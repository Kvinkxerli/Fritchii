// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
// TODO: find out which don't require stable iters
#include <unordered_map>
#include <unordered_set>
#include "video_core/page_filter.h"
#include "video_core/mirror_seq.h"
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include <vector>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <queue>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/hash.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include <ranges>
#include "common/scratch_buffer.h"
#include "common/slot_vector.h"
#include "video_core/texture_cache/pending_download.h"
#include "common/thread_worker.h"
#include "core/gpu_dirty_memory_manager.h"
#include "video_core/compatible_formats.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/fence_release_stats.h"
#include "video_core/game_dlss_status.h"
#include "video_core/nx2.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/descriptor_table.h"
#include "video_core/texture_cache/draw_template.h"
#include "video_core/texture_cache/image_base.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/lost_contents.h"
#include "video_core/texture_cache/render_targets.h"
#include "video_core/texture_cache/types.h"
#include "video_core/textures/texture.h"
#include "video_core/textures/unswizzle_worker.h"

namespace Tegra {
namespace Control {
struct ChannelState;
}
} // namespace Tegra

namespace VideoCommon {

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

struct ImageViewInOut {
    u32 index{};
    bool blacklist{};
    ImageViewId id{};
};

struct AsyncDecodeContext {
    ImageId image_id;
    Common::ScratchBuffer<u8> decoded_data;
    boost::container::small_vector<BufferImageCopy, 16> copies;
    std::mutex mutex;
    std::atomic_bool complete;
    bool cancelled{}; ///< The image was deleted meanwhile: drop the result (GPU thread only).
};

using TextureCacheGPUMap = ::Common::unordered_map<u64, std::vector<ImageId>, Common::IdentityHash<u64>>;

/// The current draw's render targets as the asynchronous shaders' skip rule sees them
/// (TextureCache::ClassifyDrawTargets): the class of the draw, and the target that gave it (the
/// first of the draw's least safe class; address 0 for a draw without targets).
struct DrawTargets {
    DrawTargetClass kind{DrawTargetClass::Recurring};
    /// kind, with targets never cleared but written in every recent render frame counted as
    /// recurring (IsRedrawnTarget; FRITCHII_ASYNC_REDRAWN).
    DrawTargetClass kind_if_redrawn{DrawTargetClass::Recurring};
    GPUVAddr gpu_addr{};
    u32 width{};
    u32 height{};
    VideoCore::Surface::PixelFormat format{};
};

class TextureCacheChannelInfo : public ChannelInfo {
public:
    TextureCacheChannelInfo() = delete;
    TextureCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept;
    TextureCacheChannelInfo(const TextureCacheChannelInfo& state) = delete;
    TextureCacheChannelInfo& operator=(const TextureCacheChannelInfo&) = delete;

    DescriptorTable<TICEntry, ImageViewId> graphics_image_table;
    DescriptorTable<TSCEntry, SamplerId> graphics_sampler_table;
    DescriptorTable<TICEntry, ImageViewId> compute_image_table;
    DescriptorTable<TSCEntry, SamplerId> compute_sampler_table;

    // TODO: still relies on bad iterators :(
    std::unordered_map<TICEntry, ImageViewId> image_views;
    std::unordered_map<TSCEntry, SamplerId> samplers;

    TextureCacheGPUMap* gpu_page_table = nullptr;
    TextureCacheGPUMap* sparse_page_table = nullptr;
};

template <class P>
class TextureCache : public VideoCommon::ChannelSetupCaches<TextureCacheChannelInfo> {
    /// Address shift for caching images into a hash table
    static constexpr u64 YUZU_PAGEBITS = 20;

    /// Enables debugging features to the texture cache
    static constexpr bool ENABLE_VALIDATION = P::ENABLE_VALIDATION;
    /// Implement blits as copies between framebuffers
    static constexpr bool FRAMEBUFFER_BLITS = P::FRAMEBUFFER_BLITS;
    /// True when some copies have to be emulated
    static constexpr bool HAS_EMULATED_COPIES = P::HAS_EMULATED_COPIES;
    /// True when the API can provide info about the memory of the device.
    static constexpr bool HAS_DEVICE_MEMORY_INFO = P::HAS_DEVICE_MEMORY_INFO;
    /// True when the API can do asynchronous texture downloads.
    static constexpr bool IMPLEMENTS_ASYNC_DOWNLOADS = P::IMPLEMENTS_ASYNC_DOWNLOADS;

    static constexpr size_t UNSET_CHANNEL{(std::numeric_limits<size_t>::max)()};

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    // Periodic statistics (TickFrame).
    u64 gc_deletes{};
    u64 gc_costly_deletes{};
    u64 gc_download_deletes{}; ///< Evictions that downloaded the image first (a full GPU wait)
    u64 gc_downloads_deferred{}; ///< EXP57: downloads left for a later frame (one wait a frame)
    u64 gc_waited_frame{~0ULL};  ///< EXP57: the frame of the collector's last download wait
    u64 async_decodes_queued{};
    u64 images_created{};
    u64 images_uploaded{};   ///< Image contents uploaded from guest memory (last 600 frames)
    u64 uploaded_bytes{};    ///< Guest bytes those uploads read
    u64 largest_upload{};    ///< Largest of them
    u64 layer_uploads{};     ///< Of them, uploads of only the written layers
    bool native_thresholds_applied{};
    u64 native_critical_memory{};
    u64 native_expected_memory{};
    u64 budget_check_frames{};
    u32 seen_pressure_events{};   ///< Allocator pressure events already answered
    u32 seen_failed_images{};     ///< Images the allocator could not make, already collected for
    u64 pressure_margin{};        ///< Extra room under the budget after images did not fit
    u64 pressure_quiet_frames{};  ///< Frames since the last such image
    u64 slack_margin{};           ///< Extra room for allocator block space the driver counts as used
    /// FRITCHII_SMALL_CARD (part Fit): room for the device memory the usage leaves out
    /// (UncountedDeviceBytes), smoothed like slack_margin; 0 while the part is off.
    u64 held_margin{};
    /// FRITCHII_VRAM_BUDGET_MODE: the thresholds as ApplyThresholdOverrides left them, before
    /// the live budget cap (budget_cap, ~0 while it does not bind) lowers them.
    u64 base_expected_memory{};
    u64 base_critical_memory{};
    u64 base_minimum_memory{};
    u64 budget_cap{~0ULL};
    bool budget_pressure{}; ///< budget_cap binds: the driver's usage passed the budget less headroom
    std::chrono::steady_clock::time_point budget_control_time{};
    /// FRITCHII_SCALE_DEPTH_MAX: depth images kept native (each counted once).
    struct DepthNativeStats {
        u32 window{}; ///< since the last 10 s line
        u64 total{};
        std::chrono::steady_clock::time_point last_log{};
    };
    DepthNativeStats depth_native_stats;
    /// What BudgetControl saw last, for the owners line.
    struct BudgetSeen {
        int mode{-1};
        u64 heap_size{};
        u64 usage{};
        u64 vk_budget{};
        u64 os_budget{}; ///< 0 without an OS sample
        u64 others{};    ///< other processes' dedicated video memory (OS counters)
        u64 budget{};    ///< the one in use
        u64 headroom{};
        u64 peak_usage{};       ///< since the last owners line
        u64 lowest_budget{~0ULL}; ///< since the last owners line
        u32 over_seconds{};     ///< samples over budget less headroom since the last owners line
        u32 spill_seconds{};    ///< samples over the budget itself since the last owners line
    } budget_seen;
    /// FRITCHII_TC_CRITICAL_ENFORCE: evictions made before creating an image.
    struct CreateEnforceStats {
        u64 passes{};
        u64 deleted{};
        u64 freed{};
        u64 capped{};  ///< passes refused by the per-frame cap
        u64 short_of{}; ///< passes that ended still over critical
    } create_enforce_stats;
    u64 last_thrash_short_of{};
    u64 create_enforce_frame{~0ULL};
    u32 create_enforce_frame_deletes{};
    /// FRITCHII_TC_THRASH_GUARD: the frame a pass walked every idle image and found none to
    /// delete; none turns up for the rest of that frame.
    u64 create_enforce_empty_frame{~0ULL};
    std::vector<ImageId> create_enforce_candidates;
    std::chrono::steady_clock::time_point vram_owners_time{};
    u64 astc_opaque_images{};
    u64 astc_alpha_images{};
    std::vector<u8> astc_scan_buffer;
    u64 images_deleted{};
    std::map<u64, std::pair<u32, GPUVAddr>> created_shapes;
    bool debug_stats{true}; ///< Collect what only the debug statistics lines use
    struct TransientTarget {
        ImageId id;
        u64 allocation_tick;
        u64 check_tick;
        u32 survived;
    };
    std::deque<TransientTarget> transient_targets;
    u64 transient_evictions{};
    u64 logged_transient_evictions{};
    [[nodiscard]] static u64 RtEvictPressureFrames() {
        static const u64 frames = [] {
            const char* value = std::getenv("EDEN_NX2_RT_EVICT_PRESSURE");
            return value != nullptr ? std::strtoull(value, nullptr, 10) : u64{8};
        }();
        return frames;
    }
    /// EDEN_NX2_MEM_TRACE: when the last memory trace lines were logged.
    std::chrono::steady_clock::time_point mem_trace_time{};
    /// EDEN_NX2_EXP90: image LRU touches made and skipped since the last report, and its time.
    u64 nx2_lru_touches{};
    u64 nx2_lru_touches_skipped{};
    std::chrono::steady_clock::time_point nx2_lru_touch_report{};

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 1_GiB + 125_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB + 625_MiB;
    /// The most device memory the collector lets the cache fill before it collects in earnest on
    /// cards up to 8 GiB; larger ones 5/8 of the driver's budget (ApplyThresholdOverrides).
    static constexpr u64 MaxCriticalMemory = 5_GiB;
    static constexpr size_t GC_EMERGENCY_COUNTS = 2;

    using Runtime = typename P::Runtime;
    using Image = typename P::Image;
    using ImageAlloc = typename P::ImageAlloc;
    using ImageView = typename P::ImageView;
    using Sampler = typename P::Sampler;
    using Framebuffer = typename P::Framebuffer;
    using AsyncBuffer = typename P::AsyncBuffer;
    using BufferType = typename P::BufferType;

    static constexpr u32 LayerUploadMinSize = 64U << 20;
    static constexpr size_t LargeScratchCapacity = 64U << 20;

    /// Splits large 2D image uploads into smaller chunks, avoiding multi-hundred MiB staging buffers.
    bool UploadInChunks(Image& image);

    /// FRITCHII_ZERO_UPLOAD (live, default 0): a whole upload whose guest bytes are all zero
    /// becomes a GPU clear to zero (no staging buffer, no untiling, no copy). True when the
    /// image was cleared (mode 1); mode 2 only counts.
    bool TryZeroUpload(Image& image);

    /// The 10 s line of FRITCHII_ZERO_UPLOAD (TickFrame).
    void LogZeroUploads();

    /// Large layered images whose refresh can decode only the layers written since the last one.
    [[nodiscard]] bool UploadsByLayer(const ImageBase& image) const noexcept;

    /// Records the layers a write of [cpu_addr, +size) touches.
    void MarkLayersWritten(Image& image, DAddr cpu_addr, size_t size);

    /// SparseTiles images: notes a written range, uploaded with the next refresh.
    void MarkRangeWritten(Image& image, DAddr cpu_addr, size_t size);

    /// SparseTiles images: binds and uploads the parts written since the last upload.
    void RefreshSparseTiles(Image& image);

    /// Image::BindTiles, counting what it binds in total_used_memory.
    bool BindSparseTiles(Image& image, std::span<const LevelRect> rects);

    /// Image::BackTilesWhole, counting what it binds in total_used_memory.
    bool BackSparseTilesWhole(Image& image);

    /// Uploads the image's dirty layers. False when it needs a full upload instead.
    bool UploadWrittenLayers(Image& image);

    /// Gives the written layers of a SparseLayers image memory and uploads them.
    void RefreshSparseLayers(Image& image);

    /// Image::BindLayers, with the memory it binds counted in total_used_memory.
    std::vector<u32> BindSparseLayers(Image& image, std::span<const u32> layers);

    /// Uploads `layers` of the image from guest memory.
    void UploadLayers(Image& image, std::span<const u32> layers);

    struct PendingUnswizzle {
        ImageId image_id;
        VideoCommon::ImageInfo info;
        size_t current_offset = 0;
        size_t total_size = 0;
        AsyncBuffer staging_buffer;
        size_t last_submitted_offset = 0;
        size_t bytes_per_slice;
        bool initialized = false;
    };

    struct BlitImages {
        ImageId dst_id;
        ImageId src_id;
        PixelFormat dst_format;
        PixelFormat src_format;
    };

public:
    /// Device pages this cache ever registered objects on (InnerInvalidation skips the rest).
    VideoCommon::PageFilter page_filter;
    /// EXP84: images registered now (non-sparse: count up at RegisterImage, down at
    /// UnregisterImage), and the pages sparse images ever covered (never cleared).
    VideoCommon::CountingPageFilter image_count_filter;
    VideoCommon::PageFilter sparse_page_filter;

    explicit TextureCache(Runtime&, Tegra::MaxwellDeviceMemoryManager&);

    /// Notify the cache that a new frame has been queued
    void TickFrame();

    /// Return a constant reference to the given image view id
    [[nodiscard]] const ImageView& GetImageView(ImageViewId id) const noexcept;

    /// Return a reference to the given image view id
    [[nodiscard]] ImageView& GetImageView(ImageViewId id) noexcept;

    /// Get the imageview from the graphics descriptor table in the specified index
    [[nodiscard]] ImageView& GetImageView(u32 index) noexcept;

    /// Mark an image as modified from the GPU
    void MarkModification(ImageId id) noexcept;

    /// Fill image_view_ids with the graphics/compute images in indices
    void FillImageViews(std::span<ImageViewInOut> views, bool compute, bool blacklist = true);

    /// Handle feedback loops during draws.
    void CheckFeedbackLoop(std::span<const ImageViewInOut> views);

    [[nodiscard]] bool SamplesRenderTarget(std::span<const ImageViewInOut> views);

    /// Get the sampler from the graphics/compute descriptor table in the specified index
    Sampler* GetSampler(u32 index, bool compute);

    /// Get the sampler id from the graphics/compute descriptor table in the specified index
    SamplerId GetSamplerId(u32 index, bool compute);

    /// Return a constant reference to the given sampler id
    [[nodiscard]] const Sampler& GetSampler(SamplerId id) const noexcept;

    /// EDEN_NX2_DEPTH_CENSUS=1 (boot, off; fable52_answer_potroom 4): per 600 frames, what the
    /// D32 render targets the cache makes are used for. One load when off.
    [[nodiscard]] bool DepthCensusOn() const noexcept {
        return depth_census_on;
    }
    /// The depth census: a draw samples `view_id` with `sampler` (null if unknown); `shader_use`
    /// is the descriptor's census_use (Shader::TextureDescriptor), `shader_dref` its is_depth.
    void CensusNoteSampled(ImageViewId view_id, const TSCEntry* sampler, u8 shader_use,
                           bool shader_dref);

    /// Return a reference to the given sampler id
    [[nodiscard]] Sampler& GetSampler(SamplerId id) noexcept;

    /// Refresh the state for graphics/compute image view and sampler descriptors
    void SynchronizeDescriptors(bool compute);

    /// Updates the Render Targets if they can be rescaled
    /// @retval True if the Render Targets have been rescaled.
    bool RescaleRenderTargets();

    /// Update bound render targets and upload memory if necessary
    /// @param is_clear True when the render targets are being used for clears
    void UpdateRenderTargets(bool is_clear);

    /// Whether a bound render target has no image because the device ran out of memory for it
    /// (FindRenderTargetView gave the null view, which only happens when JoinImages could not
    /// allocate the image). A draw or clear into it must be skipped: the null view is no depth
    /// buffer and no target of the right size, and AMD's driver crashed on such a draw
    /// (amdvlk64.dll, read of 0x0, after a 1536x1536 depth target failed on an RX 5700 XT).
    /// The targets are looked up again at the next draw. CheezyEmu a08015f893.
    [[nodiscard]] bool RenderTargetsOutOfMemory() const noexcept {
        return render_targets_out_of_memory;
    }

    /// Find a framebuffer with the currently bound render targets
    /// UpdateRenderTargets should be called before this
    Framebuffer* GetFramebuffer();

    [[nodiscard]] const RenderTargets& GetRenderTargets() const noexcept {
        return render_targets;
    }

    /// EDEN_NX2_ASYNC_DISCARD: classifies the render targets the current registers bind (the
    /// bound ones when they are current, otherwise the registered images at the registers'
    /// addresses; nothing is created). Takes the cache's lock. Meant for a draw whose pipeline is
    /// missing, so it runs per miss, not per draw.
    [[nodiscard]] DrawTargets ClassifyDrawTargets();

    /// Mark images in a range as modified from the CPU
    void WriteMemory(DAddr cpu_addr, size_t size);

    /// Download contents of host images to guest memory in a region
    void DownloadMemory(DAddr cpu_addr, size_t size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr cpu_addr, u64 size);

    /// EXP100 verify (under the mutex): logs the GPU-modified images GetFlushArea finds for a read
    /// gpu_image_filter said was clear (range, flags, nx2_gpu_counted).
    void LogGpuModifiedImagesInRegion(DAddr cpu_addr, u64 size);

    /// Remove images in a region
    void UnmapMemory(DAddr cpu_addr, size_t size);

    /// Remove images in a region
    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// Blit an image with the given parameters
    bool BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                   const Tegra::Engines::Fermi2D::Surface& src,
                   const Tegra::Engines::Fermi2D::Config& copy);

    /// Stand-in for a DLSS evaluation: temporal upscale of src into dst (see the runtime's
    /// TemporalUpscale). False when unsupported; the caller retains its native dispatch.
    bool TemporalUpscale(const Tegra::Engines::Fermi2D::Surface& dst,
                         const Tegra::Engines::Fermi2D::Surface& src,
                         const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                         const Tegra::Texture::TICEntry* depth, std::array<f32, 2> jitter,
                         std::array<f32, 2> motion_scale, bool use_history);

    /// Health check of a game's DLSS network output (dst) against its input (src); see the
    /// runtime's ProbeGameDlssOutput. Returns an earlier check's result when one is ready.
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src);

    /// Try to find a cached image view in the given CPU address
    [[nodiscard]] std::pair<ImageView*, bool> TryFindFramebufferImageView(
        const Tegra::FramebufferConfig& config, DAddr cpu_addr);

    /// Return true when there are uncommitted images to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    /// Whether CommitAsyncFlushes would commit anything (downloads or their buffers).
    [[nodiscard]] bool HasAsyncFlushWork() const noexcept;

    /// FRITCHII_FENCE_COMMIT_FAST: false only when HasAsyncFlushWork was false at the last
    /// refresh (under the lock) and nothing was queued since. Raised in the locked sections that
    /// queue a download or its buffer. Any thread, no lock.
    [[nodiscard]] bool MayHaveAsyncFlushWork() const noexcept {
        return async_flush_work_noted.load(std::memory_order_relaxed);
    }

    /// FRITCHII_FENCE_COMMIT_FAST: the note from the containers. Under the lock.
    void RefreshAsyncFlushWorkNote() noexcept {
        async_flush_work_noted.store(HasAsyncFlushWork(), std::memory_order_relaxed);
    }

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();

    /// EDEN_NX2_FENCE_RELEASE_STATS (`mutex` held): what the next PopAsyncFlushes lands.
    void Nx2DescribeNextPop(FenceReleaseStats::PopLanding& landing) {
        if (committed_downloads.empty()) {
            return;
        }
        for (const PendingDownload& download : committed_downloads.front()) {
            FenceReleaseStats::LandedItem item{};
            if (download.is_swizzle) {
                const ImageBase& image = download.GetImage(slot_images);
                item.image = true;
                item.address = image.gpu_addr;
                item.bytes = image.unswizzled_size_bytes;
                item.width = image.info.size.width;
                item.height = image.info.size.height;
                item.format = image.info.format;
                item.forced_flushed = image.info.forced_flushed;
            } else {
                const BufferDownload& buffer = slot_buffer_downloads[download.object_id];
                item.address = buffer.address;
                item.end = buffer.address + buffer.size;
                item.bytes = buffer.size;
            }
            landing.Add(item);
        }
    }

    [[nodiscard]] ImageId DmaImageId(const Tegra::DMA::ImageOperand& operand, bool is_upload);

    [[nodiscard]] std::pair<Image*, BufferImageCopy> DmaBufferImageCopy(
        const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& buffer_operand,
        const Tegra::DMA::ImageOperand& image_operand, ImageId image_id, bool modifies_image);

    void DownloadImageIntoBuffer(Image* image, BufferType buffer, size_t buffer_offset,
                                 std::span<const VideoCommon::BufferImageCopy> copies,
                                 GPUVAddr address = 0, size_t size = 0);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    [[nodiscard]] bool IsRescaling() const noexcept;

    [[nodiscard]] bool IsRescaling(const ImageViewBase& image_view) const noexcept;

    /// FRITCHII_EARLY_COMPUTE (dry): whether the view's image has a GPU write the host GPU has
    /// not run yet (its MarkModification tick; with the switch on only).
    [[nodiscard]] bool EarlyImageWritePending(ImageViewId id) noexcept;

    /// FRITCHII_ASYNC_COMPUTE_SKIP (renderer_vulkan/vk_async_compute_skip.h), the lock held:
    /// what the images on [addr, addr + size) hold, for a written storage image there.
    enum class SkipImageState : u8 {
        Redrawn,     ///< one image covers the range; every image there holds GPU writes only,
                     ///< written in each of the last REDRAWN_WRITE_FRAMES render frames (up to
                     ///< this one or the one before)
        NoImage,     ///< no image covers the range (the dispatch would make one)
        NoGpuData,   ///< an image there is not GPU-modified, or has CPU writes to upload
        NotRedrawn,  ///< an image there was not written in every one of those frames
    };
    [[nodiscard]] SkipImageState ComputeSkipImageState(DAddr addr, u64 size);

    /// Create channel state.
    void CreateChannel(Tegra::Control::ChannelState& channel) final override;

    /// Prepare an image to be used
    void PrepareImage(ImageId image_id, bool is_modification, bool invalidate);

    /// Moves the image to the young end of the LRU (EDEN_NX2_EXP90: once a frame per image).
    void TouchImageLru(ImageBase& image) noexcept;

    /// EDEN_NX2_EXP32 (texture binding memo, after the original BINDMEMO): what the
    /// graphics texture bindings of a draw are resolved against. With the same state, the same
    /// handles and the same guest TIC/TSC entries, VisitImageView and GetSamplerId give the same
    /// view and sampler ids: a view leaves a descriptor only through RemoveImageViewReferences
    /// (image deleted, joined or rescaled), which moves `epoch`, as do a new channel and the
    /// collector's out of memory table reset (null views made again); null views of unmapped
    /// addresses follow the mappings (`map_serial`). Samplers are never deleted.
    struct TextureMemoState {
        const void* channel{};
        const void* memory{};
        GPUVAddr tic_pool{};
        GPUVAddr tsc_pool{};
        u32 tic_limit{};
        u32 tsc_limit{};
        u64 map_serial{};
        u64 epoch{};

        [[nodiscard]] bool operator==(const TextureMemoState&) const noexcept = default;
    };

    /// One texture binding's guest entries as the draw that resolved it read them.
    struct TextureMemoEntry {
        u32 tic_index{}; ///< ~0U: the handle is out of the pool (null view whatever memory holds)
        u32 tsc_index{}; ///< ~0U: out of the pool (null sampler)
        TICEntry tic{};
        TSCEntry tsc{};
    };

    [[nodiscard]] TextureMemoState GraphicsTextureMemoState() const noexcept;

    /// The entries the graphics tables resolved `tic_handle` and `tsc_handle` from in this draw;
    /// false when the tables no longer hold them as resolved to `view` and `sampler`.
    [[nodiscard]] bool TakeTextureMemoEntry(u32 tic_handle, u32 tsc_handle, ImageViewId view,
                                            SamplerId sampler, TextureMemoEntry& entry) const;

    /// Whether guest memory still holds the entry's TIC and TSC entries (read as
    /// DescriptorTable::Read reads them).
    [[nodiscard]] bool TextureMemoEntryCurrent(const TextureMemoState& state,
                                               const TextureMemoEntry& entry) const;

    /// Whether preparing the view for sampling (PrepareImageView(view, false, false)) would do
    /// nothing but touch its image in the LRU: no CPU modification to upload, no aliases.
    [[nodiscard]] bool TextureMemoViewClean(ImageViewId view) const noexcept;

    /// What preparing the view does once its contents are clean: the image's LRU touch.
    void TouchTextureMemoView(ImageViewId view) noexcept;

    /// EDEN_NX2_TEX_BANK_GEN (the recomp's per-stage texture bank
    /// generations, research idea 2): a device page whose writes texture bank memos must see
    /// (pages holding texture handle words and the TIC/TSC entries draws read). A watched page
    /// counts as cached (UpdatePagesCachedCount), so guest CPU writes to it trap and mark
    /// Core::g_watch_write_pages, as do the buffer cache's writes (our own constant buffer words
    /// at FlushCaching among them) and WriteMemory. `gen` moves when a look finds the mark, and
    /// again at the next look (a trapped CPU write marks before its bytes land; a page watched
    /// just now was not trapped when the draw read it).
    struct TexBankPage {
        DAddr page{};
        u64 gen = 1;
        bool recheck = true;
    };

    /// The page's generation after taking its write mark.
    static u64 TexBankPageGen(TexBankPage& page) noexcept {
        const bool marked = Core::g_watch_write_pages.Take(page.page);
        if (marked || page.recheck) {
            ++page.gen;
        }
        page.recheck = marked;
        return page.gen;
    }

    /// The watch of `page` (page aligned), made (and the page made to trap CPU writes) when there
    /// is none (`created`). Pointers stay valid while TexBankEpoch() stands.
    [[nodiscard]] TexBankPage* TexBankWatch(DAddr page, bool& created);

    /// Moves whenever watches are dropped (TexBankReset): a memo of another epoch holds dead
    /// pointers and must not look at them.
    [[nodiscard]] u64 TexBankEpoch() const noexcept {
        return tex_bank_epoch;
    }

    [[nodiscard]] bool TexBankWatching() const noexcept {
        return !tex_bank_pages.empty();
    }

    /// Drops every watch (the pages trap as the caches' own counts say again).
    void TexBankReset();

    /// What VisitImageView does for a table entry it read before (PrepareImageView of each
    /// non-null view), for views a memo resolved. False when that deleted images: the caller
    /// resolves the full way (FillImageViews loops for the same reason).
    [[nodiscard]] bool PrepareTexBankViews(std::span<const ImageViewId> views);

    /// EXP87 (ported): what besides its handle words selects the textures
    /// a graphics draw samples now.
    [[nodiscard]] DrawTemplateContext GraphicsTextureContext(u64 owner) const noexcept;

    /// The texture and sampler descriptors the draw's textures were resolved from (`handles`:
    /// the raw handle words), for a draw template; false when one can't be kept.
    [[nodiscard]] bool KeepTemplateTextures(std::span<const u32> handles,
                                            std::span<const ImageViewInOut> views,
                                            std::span<const SamplerId> samplers,
                                            std::vector<DrawTemplateTexture>& textures);

    /// Prepares the images of the views a draw template gave, as resolving them would; false
    /// when that deleted images and the views have to be resolved again.
    [[nodiscard]] bool PrepareTemplateViews(std::span<const ImageViewInOut> views);

    /// Bumped each time the slot's view is deleted, or its image moves (RelocateImage): a kept
    /// id names the same view, and resolves from the same descriptor, while its generation holds.
    [[nodiscard]] u32 ImageViewGeneration(ImageViewId id) const noexcept {
        return id.Value() < image_view_generations.size() ? image_view_generations[id.Value()]
                                                           : 0;
    }

    /// Delta draws (D2; the original SampledViewsChange, sampled_views.h): whether
    /// resolving `views` again would change one of their images, which a delta draw keeping them
    /// does not prepare. That is PrepareImage's RefreshContents (the image is CpuModified) or
    /// SynchronizeAliases (an alias was modified after it). Views of the null image and buffer
    /// views are not counted, as there. Under the texture cache's mutex.
    [[nodiscard]] bool SampledViewsChange(std::span<const ImageViewInOut> views) const noexcept {
        for (const ImageViewInOut& view : views) {
            if (!view.id || view.id == NULL_IMAGE_VIEW_ID) {
                continue;
            }
            const ImageViewBase& image_view = slot_image_views[view.id];
            if (image_view.IsBuffer() || image_view.image_id == NULL_IMAGE_ID) {
                continue;
            }
            const ImageBase& image = slot_images[image_view.image_id];
            if (True(image.flags & ImageFlagBits::CpuModified)) {
                return true;
            }
            for (const AliasedImage& aliased : image.aliased_images) {
                if (image.modification_tick < slot_images[aliased.id].modification_tick) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Bumped whenever an image changes its scale (InvalidateScale): the rescaling words a draw
    /// pushed hold while it stays the same.
    [[nodiscard]] u64 ScaleEpoch() const noexcept {
        return scale_epoch;
    }

    /// Bumped whenever SynchronizeDescriptors finds the descriptor pools moved (EXP118=3: a
    /// delta draw that skipped it synchronizes afterwards and checks this stayed).
    [[nodiscard]] u64 TextureBindingsSerial() const noexcept {
        return texture_bindings_serial;
    }

    /// The device page `gpu_addr` lies in, or nullopt when unmapped.
    [[nodiscard]] std::optional<DAddr> TexBankDevicePage(GPUVAddr gpu_addr) const;

    /// The graphics tables' limits and native tiling, as VisitImageView and GetSamplerId use.
    [[nodiscard]] u32 GraphicsTicIndex(u32 tic_handle) const noexcept {
        return Tegra::Texture::TextureHeaderIndex(tic_handle,
                                                  channel_state->graphics_image_table.current_limit,
                                                  gpu_memory->UsesNativeTiling());
    }

    [[nodiscard]] u64 FrameTick() const noexcept {
        return frame_tick;
    }

    /// What a clear's render target lookups did (RasterizerVulkan::Clear's "Slow clear" line):
    /// ResetRenderTargetStats before UpdateRenderTargets, TakeRenderTargetStats after
    /// GetFramebuffer. Counts and times of JoinImages, RemoveFramebuffers and GetFramebufferId.
    struct RenderTargetStats {
        u64 images_created{};
        u64 images_deleted{};
        u64 uploaded_bytes{};
        u32 joins{};
        u32 overlaps_walked{};      ///< Images region_check looked at
        u32 bad_overlaps{};
        u32 framebuffers_removed{};
        u32 framebuffers_scanned{}; ///< Framebuffer map entries RemoveFramebuffers looked at
        u32 framebuffers_created{};
        u32 pooled_images{};        ///< New images that took a pooled VkImage (EXP42)
        u32 relocated_images{};     ///< Images moved instead of made (EXP43=2)
        u32 retired_images{};       ///< Predecessors deleted at once (EXP43=1)
        u64 walk_ns{};              ///< JoinImages' overlap walk
        u64 create_ns{};            ///< Making the new image (vkCreateImage, memory)
        u64 refresh_ns{};           ///< Its upload (RefreshContents) or layout transition
        u64 delete_ns{};            ///< Deleting the images it replaced
        u64 framebuffer_ns{};       ///< New framebuffers
        u64 framebuffer_map_size{}; ///< At TakeRenderTargetStats
        u64 image_count{};
    };
    void ResetRenderTargetStats() noexcept {
        rt_stats = {};
        rt_stats_base = {images_created, images_deleted, uploaded_bytes};
    }
    [[nodiscard]] RenderTargetStats TakeRenderTargetStats() const noexcept {
        RenderTargetStats stats = rt_stats;
        const auto delta = [](u64 now, u64 base) { return now >= base ? now - base : now; };
        stats.images_created = delta(images_created, rt_stats_base[0]);
        stats.images_deleted = delta(images_deleted, rt_stats_base[1]);
        stats.uploaded_bytes = delta(uploaded_bytes, rt_stats_base[2]);
        stats.framebuffer_map_size = framebuffers.size();
        stats.image_count = slot_images.size();
        return stats;
    }

    Common::SpinRecursiveMutex mutex;
    /// EXP100 (fable52_answer_imagereads 2): registered non-sparse images that were GPU-modified
    /// at some time since they were registered (CountGpuModified at the three sites that set
    /// GpuModified and at RegisterImage; removed at UnregisterImage; clears of the flag are
    /// ignored on purpose). Changed only under the mutex, read without it.
    VideoCommon::CountingPageFilter gpu_image_filter;
    /// Odd from a hold's first change of gpu_image_filter until that hold releases the mutex
    /// (the mutex's release hook, set in the constructor). After `mutex` in declaration order.
    VideoCommon::MirrorSeq image_seq{mutex, "Texture cache GPU-modified image count (EXP100)"};

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachCPUPage(DAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> YUZU_PAGEBITS;
        for (u64 page = addr >> YUZU_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    template <typename Func>
    static void ForEachGPUPage(GPUVAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> YUZU_PAGEBITS;
        for (u64 page = addr >> YUZU_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    void OnGPUASRegister(size_t map_id) final override;

    /// Runs the Garbage Collector.
    void RunGarbageCollector(bool out_of_memory);

    /// EXP121: the collector's download without a wait (false: take the waiting path).
    bool QueueCollectDownload(ImageId image_id);
    /// EXP121: deletes the images whose download the GPU has finished, drops those used again.
    void LandCollectDownloads();
    /// EXP121: forgets queued downloads (the images stay) overlapping guest bytes written now.
    void DropCollectDownloadsIn(DAddr cpu_addr, size_t size);
    void DropCollectDownload(size_t index);

    /// Find or create an image view in the guest descriptor table
    ImageViewId VisitImageView(u32 index, bool compute);

    /// Find or create a framebuffer with the given render target parameters
    FramebufferId GetFramebufferId(const RenderTargets& key);

    /// Refresh the contents (pixel data) of an image
    [[nodiscard]] bool IsDownloadable(const ImageBase& image) const noexcept;

    void RefreshContents(Image& image, ImageId image_id);

    /// Upload data from guest to an image
    template <typename StagingBuffer>
    void UploadImageContents(Image& image, StagingBuffer& staging_buffer);

    /// Create a new image view from a guest descriptor
    [[nodiscard]] ImageViewId CreateImageView(const TICEntry& config);

    /// Find or create an image from the given parameters
    [[nodiscard]] ImageId FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                            RelaxedOptions options = RelaxedOptions{});

    /// Find an image from the given parameters
    [[nodiscard]] ImageId FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                    RelaxedOptions options);

    /// Create an image from the given parameters
    [[nodiscard]] ImageId InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                      RelaxedOptions options);

    /// EDEN_NX2_IMAGE_HASH_LOG=1 (boot): a sampled image just made from a texture header; counts
    /// it by the hash of its guest bytes and logs a summary every 10 s (are a kill's effect
    /// textures made again each time? fable52_answer_backdraw (b) 4).
    void NoteNewSampledImage(ImageId image_id, GPUVAddr gpu_addr);

    /// EDEN_NX2_IMAGE_HASH_LOG: why images are deleted, so a texture made again can say what
    /// killed its previous copy at that address.
    static bool ImageHashLogEnabled() {
        static const bool enabled = [] {
            const char* const value = std::getenv("EDEN_NX2_IMAGE_HASH_LOG");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }
    enum class DeleteReason : u8 {
        Other,
        GarbageCollect,
        ColdSweep,
        Unmap,
        Join,
        SlotRetire,
        UnmapSweep,
        SupersededSweep,
        Count
    };
    DeleteReason delete_reason = DeleteReason::Other;
    struct DeleteReasonScope {
        DeleteReasonScope(TextureCache& cache_, DeleteReason reason)
            : cache{cache_}, previous{cache_.delete_reason} {
            cache.delete_reason = reason;
        }
        ~DeleteReasonScope() {
            cache.delete_reason = previous;
        }
        TextureCache& cache;
        DeleteReason previous;
    };
    /// Deleted images by address: the reason, bit 7 set when the GPU had written the image.
    std::unordered_map<GPUVAddr, u8> nx2_deleted_reasons;
    std::array<u64, static_cast<size_t>(DeleteReason::Count)> nx2_deletes{};
    std::array<u64, static_cast<size_t>(DeleteReason::Count) * 2> nx2_remade_after{};
    u64 nx2_remade_unknown = 0;
    /// FRITCHII_TC_THRASH_GUARD: what a deleted image left behind, to recognise it when the game
    /// asks for it again. Direct-mapped by address and shape; a collision forgets the older one.
    struct ThrashGhost {
        GPUVAddr gpu_addr{}; ///< 0 = empty
        u64 shape{};         ///< RelocationShapeKey
        u64 last_use{};      ///< Frame of its last use
        u32 kib{};           ///< Its device memory
        u8 cause{};          ///< ThrashCause
        bool costly{};       ///< CostlyLoad: ASTC or BCn that is decoded again
    };
    static constexpr size_t ThrashGhostSlots = 8192;
    /// DeleteReason's values, then the collector under pressure on its own: the one the floor
    /// answers.
    static constexpr size_t ThrashCausePressured = static_cast<size_t>(DeleteReason::Count);
    static constexpr size_t ThrashCauses = ThrashCausePressured + 1;
    /// Reuse intervals by size: up to 4, 8, 16 ... 1024 frames.
    static constexpr size_t ThrashBuckets = 9;
    static constexpr u64 ThrashMaxFloor = 600; ///< Frames; ~10 s at 60 fps
    std::vector<ThrashGhost> thrash_ghosts; ///< Empty until the first delete with the guard on
    std::array<u32, ThrashBuckets> thrash_window{}; ///< Pressured remakes since the last check
    u64 thrash_floor{};       ///< The collector leaves images used this recently alone
    u64 thrash_calm_frames{}; ///< Frames without pressure, for the floor's decay
    u64 thrash_image_bytes{}; ///< The live images at the last check with a floor
    struct ThrashStats {
        u64 made{};
        std::array<u64, ThrashCauses> remade{};
        u64 remade_kib{};
        u64 remade_costly{};
        u64 interval_sum{};
        u32 raised{};
        u32 capped{};
    } thrash_stats;
    /// The ghost slot of an address and shape.
    [[nodiscard]] static size_t ThrashSlot(GPUVAddr gpu_addr, u64 shape) noexcept {
        const u64 mixed = ((gpu_addr >> 8) ^ shape) * 0x9E3779B97F4A7C15ULL;
        return static_cast<size_t>(mixed >> 51); // 13 bits: ThrashGhostSlots
    }
    /// EXP76: images flagged Remapped (UnmapGPUMemory), kept after a remap (& 1), whole uploads
    /// skipped because the guest bytes hashed the same as at the last upload (& 2), and checked.
    u64 nx2_flagged_remapped = 0;
    u64 nx2_kept_after_remap = 0;
    u64 nx2_upload_skips = 0;
    u64 nx2_upload_skip_bytes = 0;
    u64 nx2_upload_hash_checks = 0;
    /// Images an unmap of their GPU memory flagged Remapped, and what became of them: the
    /// "Texture cache last 600 frames: GPU memory unmaps" line (EDEN_NX2_TC_LOG). Counted off the
    /// draw path only (unmaps, uploads, downloads, deletes, the frame end).
    struct UnmapStats {
        u64 unmaps_hit = 0;  ///< UnmapGPUMemory calls that found images
        u64 images_hit = 0;  ///< Images they found, flagged before or not
        u64 flagged = 0;     ///< Images newly flagged Remapped
        u64 refound = 0;     ///< Found again at the same address and shape (EXP76 & 1)
        u64 reuploaded = 0;  ///< Uploads into a flagged image (a cached descriptor or binding)
        u64 downloads = 0;   ///< Downloads of a flagged image (collector, guest reads)
        /// Deleted while flagged, by reason; and of those, the ones not used since the unmap
        /// (still CpuModified: no path would have downloaded them).
        std::array<u64, static_cast<size_t>(DeleteReason::Count)> died{};
        std::array<u64, static_cast<size_t>(DeleteReason::Count)> died_unused{};
        u64 collector_unused_frames = 0; ///< Frames from the unmap, summed over those
        u64 collector_unused_bytes = 0;  ///< Their device memory, summed
        u64 held_byte_frames = 0;        ///< Device memory of flagged images, summed per frame
        u64 held_peak = 0;               ///< The most at one frame end
        /// EDEN_NX2_SWEEP_UNMAPPED: entries looked at, images deleted and their memory (1); or
        /// would have deleted, and what happened to those afterwards (2).
        u64 sweep_scanned = 0;
        u64 swept = 0;
        u64 swept_bytes = 0;
        u64 would_sweep = 0;
        u64 would_sweep_bytes = 0;
        u64 would_refound = 0;     ///< Found again at the same address (EXP76 & 1)
        u64 would_reuploaded = 0;  ///< Uploaded into (a cached descriptor or binding)
        u64 would_reuploaded_gpu = 0; ///< ... with GPU-written contents from before the unmap
        u64 would_presented = 0;   ///< Shown (TryFindFramebufferImageView)
        u64 would_overwritten = 0; ///< Used again as a target written whole (nothing read)
        u64 would_died_unused = 0; ///< Deleted later by another path, unused
    };
    UnmapStats unmap_stats;
    u64 unmapped_live_bytes = 0;  ///< Device memory of the images flagged now
    u64 unmapped_live_images = 0; ///< How many
    /// The images flagged now (ImageBase::unmapped_index is the place of each).
    std::vector<ImageId> unmapped_images;
    /// The last images shown (TryFindFramebufferImageView): the sweep leaves them alone, since
    /// presenting reads an image without preparing it.
    std::array<ImageId, 8> presented_images{};
    u32 presented_next = 0;
    /// UnmapGPUMemory flagged it Remapped: counted and remembered (unmapped_tick).
    void NoteUnmapped(ImageId image_id, Image& image);
    /// Remapped cleared (found again) or the image deleted: no longer counted as flagged.
    void ForgetUnmapped(ImageId image_id, ImageBase& image);
    void LogUnmapStats();
    void NotePresented(ImageId image_id);
    /// EDEN_NX2_SWEEP_UNMAPPED (read at startup): 0 = off, 1 = delete flagged images untouched
    /// since the unmap after 2 frames, 2 = only count what 1 would delete and what used it later.
    static int SweepUnmappedMode() {
        static const int mode = [] {
            const char* const value = std::getenv("EDEN_NX2_SWEEP_UNMAPPED");
            return value == nullptr ? 0 : value[0] == '1' ? 1 : value[0] == '2' ? 2 : 0;
        }();
        return mode;
    }
    /// Deletes (or with 2 counts) flagged images nothing could still read (see the definition).
    void SweepUnmapped();
    /// EXP77: shapes of GPU-written images the cold sweep deleted, by address (at most 65,536),
    /// and the shapes guarded because a texture header asked for that address again.
    ::Common::unordered_map<GPUVAddr, u64> nx2_swept_shapes;
    u64 nx2_sweep_guards = 0;

    /// EXP76 & 4 (fable52_answer_keeptex_2, step 1, measure only): images whose whole upload is
    /// hashed (& 2) are indexed by (shape, content hash); a new sampled image hashes its guest bytes
    /// and counts which reuse would serve it. Deleted hashed images are remembered as "kept"
    /// within 512 MiB, oldest out. Nothing is reused yet.
    struct ContentKey {
        u64 shape;
        u64 lo;
        u64 hi;
        bool operator==(const ContentKey&) const = default;
    };
    struct ContentKeyHash {
        size_t operator()(const ContentKey& key) const noexcept {
            return static_cast<size_t>(key.lo ^ (key.shape * 0x9E3779B97F4A7C15ULL) ^ key.hi);
        }
    };
    [[nodiscard]] static u64 ContentShape(const ImageInfo& info) noexcept;
    void NoteContentLookup(ImageId image_id);
    void ForgetContent(ImageId image_id, const ImageBase& image, bool deleted);
    std::unordered_map<ContentKey, ImageId, ContentKeyHash> nx2_content_index;
    std::unordered_map<ContentKey, u32, ContentKeyHash> nx2_content_kept;
    std::deque<std::pair<ContentKey, u32>> nx2_content_kept_order;
    u64 nx2_content_kept_bytes = 0;
    std::array<u64, 4> nx2_content_cases{}; ///< idle twin, busy twin, kept, none
    std::array<u64, 4> nx2_content_case_bytes{};
    u64 nx2_content_hashed_bytes = 0;
    u64 nx2_content_hash_ns = 0;
    std::vector<u8> nx2_hash_scratch;
    /// CityHash128 of an image's guest bytes (in place when they are one continuous span).
    [[nodiscard]] u128 GuestContentHash(const ImageBase& image);

    /// Create a new image and join perfectly matching existing images
    /// Remove joined images from the cache
    [[nodiscard]] ImageId JoinImages(const ImageInfo& info, GPUVAddr gpu_addr, DAddr cpu_addr);

    [[nodiscard]] ImageId FindDMAImage(const ImageInfo& info, GPUVAddr gpu_addr);

    /// Return a blit image pair from the given guest blit parameters
    [[nodiscard]] std::optional<BlitImages> GetBlitImages(
        const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Config& copy);

    /// Find or create a sampler from a guest descriptor sampler
    [[nodiscard]] SamplerId FindSampler(const TSCEntry& config, bool compute);

    /// Find or create an image view for the given color buffer index
    [[nodiscard]] ImageViewId FindColorBuffer(size_t index);

    /// Find or create an image view for the depth buffer
    [[nodiscard]] ImageViewId FindDepthBuffer();

    /// Find or create a view for a render target with the given image parameters
    [[nodiscard]] ImageViewId FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr);

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachImageInRegion(DAddr cpu_addr, size_t size, Func&& func);

    template <typename Func>
    void ForEachImageInRegionGPU(size_t as_id, GPUVAddr gpu_addr, size_t size, Func&& func);

    template <typename Func>
    void ForEachSparseImageInRegion(size_t as_id, GPUVAddr gpu_addr, size_t size, Func&& func);

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachSparseSegment(ImageBase& image, Func&& func);

    /// Find or create an image view in the given image with the passed parameters
    [[nodiscard]] ImageViewId FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    /// Track CPU reads and writes for image
    void TrackImage(ImageBase& image, ImageId image_id);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageBase& image, ImageId image_id);

    /// Delete image from the cache
    void DeleteImage(ImageId image, bool immediate_delete = false);

    /// Remove image views references from the cache
    void RemoveImageViewReferences(std::span<const ImageViewId> removed_views);

    /// Per image view, the (channel, descriptor) entries of the channels' image_views maps that
    /// point to it. Deleting an image erased them by scanning every map (title 1: ~5,000 entries,
    /// 50-130 us for each deleted image, dozens a frame while the collector runs).
    std::unordered_map<ImageViewId, boost::container::small_vector<std::pair<size_t, TICEntry>, 2>>
        image_view_descriptors;

    /// Remove framebuffers using the given image views from the cache
    void RemoveFramebuffers(std::span<const ImageViewId> removed_views);

    /// Render target relocation (EXP43, fable_answer_12 2A). Per shape (format, size, samples,
    /// levels, layers), sticky for the session: a shape whose GPU-written contents were read in a
    /// later frame (sampled, rendered to without a clear) or read by the CPU never relocates or
    /// retires early, since its predecessor's contents may still be wanted.
    enum RelocationGuard : u8 {
        GuardSampled = 1,  ///< Sampled or copied from in a frame after the GPU wrote it
        GuardLoaded = 2,   ///< Rendered to without a clear in a frame after the GPU wrote it
        GuardCpuRead = 4,  ///< Read by the CPU (GetFlushArea, DownloadMemory)
    };
    struct RelocationShape {
        u8 guard{};
        bool candidate_logged{};
        bool relocated_logged{};
        u32 same_frame_refused{}; ///< Since the last report (render-frame clock line)
    };
    ::Common::unordered_map<u64, RelocationShape> relocation_shapes;
    /// The guard by address too: OoT draws every dialogue box's text into a render target at one
    /// address, sized to the box. Per shape, each new box size was relocated once before its
    /// guard tripped, and that box's text showed other images' contents (2026-11-09, the owner).
    ::Common::unordered_map<GPUVAddr, u8> relocation_guarded_addresses;
    /// Since the last report (TickFrame, every 600 frames once a candidate was seen).
    u64 relocation_candidates{};
    u64 relocation_guarded{};
    u64 relocation_relocated{};
    u64 relocation_retired{};
    u64 relocation_crowded{}; ///< Cleared targets over two or more images of their shape
    u64 relocation_same_frame{}; ///< Candidates whose only predecessors were written this frame
    /// EXP64, since the last report: retired, and those kept by reason.
    u64 slot_retired{};
    u64 slot_kept_guard{};
    u64 slot_kept_busy{};
    u64 slot_kept_kind{};
    std::array<u64, 32> slot_kept_flag{}; ///< Of slot_kept_kind, by ImageFlagBits bit
    u64 slot_kept_not_gpu{};
    u64 slot_kept_aliased{};
    u64 slot_kept_same_frame{};
    u64 slot_clears{};
    /// EXP67 (host-padded per-NPC depth targets): addresses whose depth target changed dims
    /// (padded from then on), addresses where a texture lookup found a padded image (never
    /// padded again), and counts since the last report.
    std::unordered_set<GPUVAddr> pad_slots;
    std::unordered_set<GPUVAddr> pad_unsafe_slots;
    u64 pad_made{};
    /// EXP68 bit 1 (fable52_answer_npcdips 6): a depth clear whose scissor is the real width
    /// (the zeta width register rounded up to 16 in the same 64-byte column) goes to the exact
    /// image the following draws use. Verified by the next depth bind at the address.
    /// Per address: the exact size a clear went to this frame, until a depth bind there checks it.
    std::unordered_map<GPUVAddr, std::pair<u32, u32>> clear_exact_pending;
    std::unordered_set<GPUVAddr> clear_exact_unsafe;
    u64 clear_exact_used{};
    u64 clear_exact_confirmed{};
    u64 clear_exact_missed{};
    u64 clear_exact_unchecked{};
    u64 no_pool_marked{}; ///< EXP68 bit 4: new images at size-varying addresses (skip the pool)
    u64 pad_depth_lookups{};   ///< New depth targets EXP67 looked at (render target lookups)
    u64 pad_not_varying{};     ///< ... at an address with no other-dims predecessor yet
    u64 pad_skipped_other{};   ///< ... refused: rescaling, not plain 2D, sampled address
    u64 pad_sampled{};
    u32 pad_logged{};

    /// EDEN_NX2_DEPTH_CENSUS (see DepthCensusOn). A D32 render target the cache made, followed
    /// until it is deleted or CensusMaxAge frames old, then counted.
    const bool depth_census_on = [] {
        const char* value = std::getenv("EDEN_NX2_DEPTH_CENSUS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    static constexpr u64 CensusMaxAge = 120;
    struct CensusTarget {
        GPUVAddr gpu_addr{};
        u64 frame{};             ///< frame_tick when made
        bool cleared{};          ///< a clear of depth (or stencil) with it bound
        bool drawn{};            ///< a draw with it bound as the depth target
        bool drawn_writes{};     ///< ... one with depth test and depth writes on
        bool sampled_same{};     ///< first sampled by a draw in the frame it was made
        bool sampled_later{};    ///< first sampled in a later frame
        bool cpu_read{};         ///< downloaded for a guest read
        bool dma_read{};         ///< copied to a buffer (DownloadImageIntoBuffer)
        bool via_texture{};      ///< joined into an image a texture lookup made
        u8 shader_use{};         ///< census_use bits of every sampling descriptor, or'ed
    };
    std::unordered_map<ImageId, CensusTarget> census_live;
    std::unordered_map<GPUVAddr, ImageId> census_by_addr; ///< The newest live target per address
    struct CensusMade {
        GPUVAddr gpu_addr;
        u32 width;
        u32 height;
    };
    std::vector<CensusMade> census_this_frame; ///< In the order made: the n-th of the frame
    std::vector<CensusMade> census_last_frame;
    /// Since the last report. Made: counted when made; the rest when the target is done.
    struct CensusCounts {
        u64 made{};
        u64 at_floor{};          ///< 384x384
        u64 no_predecessor{};    ///< no n-th target last frame
        u64 same_address{};      ///< at the address of last frame's n-th
        u64 moved{};
        std::array<u64, 4> side_change{}; ///< vs last frame's n-th: 0, 1-8, 9-64, more
        u64 done{};
        u64 cleared_and_drawn{};
        u64 cleared_only{};
        u64 drawn_only{};
        u64 neither{};
        u64 drawn_writes{};
        u64 sampled_same{};
        u64 sampled_later{};
        u64 never_sampled{};
        u64 via_texture{};
        u64 cpu_read{};
        u64 dma_read{};
        u64 deleted{};           ///< done by deletion (the rest aged out at CensusMaxAge)
        // Of the sampled, at the first sampling draw:
        std::array<u64, 4> address_mode{}; ///< u and v: repeat/mirror, clamp, border, mixed
        u64 compare_on{};        ///< the sampler's depth compare
        u64 shader_dref{};       ///< the shader samples with a depth reference
        u64 no_sampler{};
        // Of the sampled, any sampling descriptor (or'ed):
        u64 use_unknown{};       ///< shader info from the precompiled cache (no census bits)
        u64 use_fetch{};
        u64 use_query{};
        u64 use_offset{};
        u64 use_gather{};
        u64 use_rect{};
    } census{};
    void CensusNoteMade(ImageId image_id, const ImageInfo& info, GPUVAddr gpu_addr);
    void CensusNoteRenderTargets(bool is_clear);
    void CensusNoteRead(const ImageBase& image, bool dma);
    void CensusFinish(ImageId image_id, bool deleted);
    void CensusTickFrame();
    /// EXP64 check: images made again (not by a clear) at a retired image's address and shape
    /// within 16 render frames: a reader of the retired contents, which got guest memory instead.
    u64 slot_remade{};
    struct SlotRetired {
        GPUVAddr gpu_addr;
        u64 shape;
        u64 render_frame;
    };
    std::array<SlotRetired, 16> slot_recent{};
    size_t slot_recent_next{};
    bool slot_retired_any{};
    u32 slot_remade_logged{};
    bool relocation_seen{};

    /// The frame last_write_frame, CheckCrossFrameUse and RelocationCandidate's same-frame rule
    /// count in. frame_tick advances at the composite, a sync request the back serves wherever it
    /// is, so one host tick holds the end of one guest frame and the start of the next
    /// (fable_answer_16 1a). With EDEN_NX2_RENDER_CLOCK=1 it follows Tegra::g_nx2_render_frame
    /// instead (bumped where the frame's last GPU work is), falling back to frame_tick while that
    /// does not advance (no word learned yet, other titles). Monotonic across the switches; the
    /// LRU, the collector and the logs keep frame_tick.
    [[nodiscard]] u64 RenderFrame() const noexcept {
        const u64 source = render_clock_live
                               ? Tegra::g_nx2_render_frame.load(std::memory_order_relaxed)
                               : frame_tick;
        return render_clock_base + (source - render_clock_source_base);
    }
    /// TickFrame: follows the counter, or falls back to frame_tick after two host frames
    /// without a count; logs the rate next to the relocation line.
    void TickRenderClock();
    bool render_clock_live{};
    bool render_clock_enabled{}; ///< Last tick: the clock wanted (Resume on the way back on)
    bool render_clock_logged{};
    u64 render_clock_base{};
    u64 render_clock_source_base{};
    u64 render_clock_last_counter{};
    u64 render_clock_window_counter{}; ///< The counter at the last report
    u32 render_clock_idle{};           ///< Host frames since the counter last advanced
    u32 render_clock_live_frames{};    ///< Since the last report
    u32 render_clock_switches{};

    /// EXP43=4 (fable_answer_16 1c): relocation as 3, under the render-frame clock, and each
    /// relocated image's old range kept for 2 render frames. A read that lands on it (an image
    /// made at the old address and then sampled or rendered to without a clear, or a CPU read of
    /// the old range while the image there was GPU-written) is what relocation would have gotten
    /// wrong: counted and its first few logged. Must stay 0.
    struct RelocationTombstone {
        GPUVAddr gpu_addr;
        DAddr cpu_addr;
        u64 size;
        u64 shape_key;
        u64 render_frame;
        GPUVAddr moved_to;
        bool gpu_modified;
    };
    enum TombstoneRead : u8 { TombstoneSampled, TombstoneLoaded, TombstoneCpuRead, TombstoneReads };
    void AddRelocationTombstone(const ImageBase& old_image, GPUVAddr moved_to);
    /// JoinImages: a new image at a tombstoned address waits for its first PrepareImage.
    void CheckTombstoneCreation(ImageId image_id, const ImageBase& image, bool overwritten);
    void CheckTombstonePrepare(ImageId image_id, const ImageBase& image, bool is_modification,
                               bool invalidate);
    void CheckTombstoneCpuRead(DAddr cpu_addr, u64 size);
    void NoteTombstoneHit(const RelocationTombstone& tombstone, TombstoneRead reason,
                          GPUVAddr gpu_addr, const ImageInfo* info);
    /// TickFrame: drops tombstones older than 2 render frames; logs every 600 frames.
    void TickRelocationTombstones();
    std::vector<RelocationTombstone> relocation_tombstones;
    DAddr tombstone_cpu_begin{~DAddr{0}}; ///< Bounds of all tombstones (quick reject)
    DAddr tombstone_cpu_end{};
    struct TombstoneBorn {
        ImageId image_id;
        GPUVAddr gpu_addr;
        RelocationTombstone tombstone;
    };
    boost::container::small_vector<TombstoneBorn, 4> tombstone_born;
    std::array<u64, TombstoneReads> tombstone_hits{}; ///< Since the last report
    u64 tombstones_made{};
    u64 tombstone_overlap_reads{}; ///< New images read over one, not at its address
    u32 tombstone_hits_logged{};

    /// Cold sweep (EDEN_NX2_TC_COLD_FRAMES=K, 0 = off; fable_answer_16 3.2): a relocation-shaped
    /// image with BadOverlap (a member of a drifted chain) untouched for K host frames, with no
    /// download queued and not bound, is deleted in TickFrame (at most 16 a frame), so its
    /// VkImage goes to the EXP42 pool instead of waiting for the collector, which only starts
    /// near the device budget. Never a GPU-written image that may be read again: one that was
    /// itself read in a later frame or by the CPU, or of a shape guarded as read by the CPU, and
    /// unless EDEN_NX2_TC_COLD_GUARDED=1 of a shape guarded as sampled or rendered to without a
    /// clear in a later frame. JoinImages puts the chain members it sees on the watch list.
    void WatchColdImage(ImageId image_id);
    void SweepColdImages();
    /// FRITCHII_SWEEP_SUPERSEDED (live, frames, 0 = off; port of Skitchii 16fb54111d): deletes
    /// GPU-written images untouched for N frames whose memory a newer image of another shape
    /// covers, without a download (see the definition).
    void SweepSuperseded();
    struct SupersededStats {
        u64 passes{};        ///< Sweeps that ran
        u64 looked{};        ///< GPU-written images looked at
        u64 deleted{};       ///< Deleted
        u64 deleted_bytes{}; ///< Their device memory
        u64 kept_busy{};     ///< Superseded but bound or with a download queued
        u64 sweep_ns{};      ///< Time spent sweeping
        u64 sampled_after{}; ///< Sampled again at a swept address (a new image made there)
    } superseded_stats{};
    /// Addresses the superseded sweep deleted at (emptied at 65,536), and the frame: a texture made
    /// at one by sampling counts in sampled_after (should stay near 0).
    std::unordered_map<GPUVAddr, u64> superseded_swept;
    /// FRITCHII_COVERED_RETIRE: whether newer GPU writes (images written after this one's last
    /// write and touched since its last use) cover every byte of the image's memory.
    [[nodiscard]] bool CoveredByNewerWrites(ImageId image_id, ImageBase& image);
    /// Deletes a covered image without a download (as the collector deletes a bad overlap).
    void RetireCovered(ImageId image_id, Image& image);
    struct CoveredStats {
        u64 checked{};         ///< Coverage checks
        u64 sweep_deleted{};   ///< Deleted by the cold sweep
        u64 written_back{};    ///< Of them, written back first (FRITCHII_COVERED_RETIRE_WRITEBACK)
        u64 collect_deleted{}; ///< Deleted by the collector without the download it would make
        u64 landing_deleted{}; ///< Queued collector downloads dropped as covered, image deleted
        u64 deleted_bytes{};   ///< Their device memory
        u64 busy{};            ///< Covered but bound or with a download queued
        u64 capped_frames{};   ///< Frames the sweep hit its per-frame caps
        u64 sampled_after{};   ///< A texture made at a retired address (must stay near 0)
        u64 check_ns{};        ///< Time spent in the checks
    } covered_stats{};
    /// Addresses retired as covered (emptied at 65,536), for sampled_after.
    std::unordered_map<GPUVAddr, u64> covered_swept;
    u32 covered_sampled_logged{};
    /// FRITCHII_COLD_SWEEP_KEEP_RESAMPLED (live, default 1, 0 = off): the re-sampled set, keyed by
    /// (GPU address, RelocationShapeKey) of GPU-written targets the cold sweep (covered retire
    /// included) deleted and the game then sampled with lost contents, plus the sampled image's
    /// own address and shape; the value is the frame last seen. At most ColdResampledMax, the
    /// least recently seen out. A target made again at such an address and shape is kept by the
    /// sweep (only where it would delete it) within FRITCHII_COLD_SWEEP_KEEP_MIB of real device
    /// memory (the image plus its scaled copy) held as a resident total (cold_keep_resident):
    /// images already kept keep their share, so an overrun turns away new candidates. The default
    /// is min(256 MiB, GetDeviceLocalMemory() / 32), or min(512 MiB, GetDeviceLocalMemory() / 16)
    /// with a driver budget of 12 GiB or more (SweepColdImages). Under pressure (budget_pressure,
    /// an image that failed in the last ColdKeepFailedFrames, or the cache over critical_memory +
    /// min(max(1 GiB, critical / 4), (driver budget - critical) / 2), no margin without a
    /// reported budget) the sweep deletes kept images as before.
    /// 2 also keeps images sampled in a frame after their last GPU write (GuardSampled), from
    /// both. 3 also keeps GPU-written targets whose memory overlaps a recently lost read
    /// (lost_read_ranges, 128 ranges), under the same keep budget.
    static constexpr size_t ColdResampledMax = 4096;
    ::Common::unordered_map<u64, u64> cold_resampled;
    u32 cold_resampled_gen = 1; ///< Bumped by every change of the set (ImageBase::cold_keep_gen)
    /// The frame an image last failed to allocate (0 = never): the keep's pressure gate.
    u64 cold_keep_failed_frame{};
    static constexpr u64 ColdKeepFailedFrames = 600;
    /// The keep budget in use: the sum of every image's cold_keep_charge, and how many hold one.
    u64 cold_keep_resident{};
    u32 cold_keep_resident_count{};
    /// GetDeviceMemoryBudget(), refreshed every 60 frames by the sweep (0 = not reported).
    u64 cold_keep_driver_budget{};
    u64 cold_keep_budget_frame{};
    /// Gives back an image's share of the keep budget and clears its covered-kept bit.
    void ColdKeepUncharge(ImageBase& image) noexcept {
        image.cold_keep &= static_cast<u8>(~4);
        if (image.cold_keep_charge != 0) {
            cold_keep_resident -= (std::min)(cold_keep_resident, image.cold_keep_charge);
            cold_keep_resident_count -= cold_keep_resident_count != 0 ? 1 : 0;
            image.cold_keep_charge = 0;
        }
    }
    [[nodiscard]] static u64 ColdResampledKey(GPUVAddr gpu_addr, u64 shape_key) noexcept {
        return (gpu_addr * 0x9E3779B97F4A7C15ULL) ^ shape_key;
    }
    void LearnColdResampled(GPUVAddr gpu_addr, u64 shape_key);
    /// Whether the sweep keeps this GPU-written image under the rule (cached per set generation).
    [[nodiscard]] bool ColdKeepResampled(ImageBase& image, int mode);
    struct ColdKeepStats {
        u64 kept_new{};        ///< Images kept by the rule for the first time
        u64 kept_now{};        ///< Images kept in the report's frame
        u64 kept_now_bytes{};  ///< Their device memory
        u64 kept_sampled{};    ///< Of kept_new: by the sampled-after-write rule (mode 2)
        u64 kept_range{};      ///< Of kept_new: over a recently lost read's range (mode 3)
        u64 released{};        ///< Would be kept, deleted as memory was over the line
        u64 refused_newer{};   ///< Learned key, deleted: newer images cover or overlap it
        u64 learned{};         ///< Keys added to the set
        u64 lost_all{};        ///< Lost contents sampled (any drop)
        u64 lost_cold{};       ///< ... dropped by the cold sweep (each drop once)
        u64 lost_cold_known{}; ///< ... whose drop's key was in the set when deleted
        u64 lost_cold_repeat{}; ///< More images sampled over a cold drop already counted
        u64 lost_not_recorded{}; ///< ... with no drop on record
    } cold_keep_stats{};
    std::vector<std::pair<DAddr, DAddr>> covered_ranges; ///< Scratch for CoveredByNewerWrites
    std::vector<std::pair<ImageId, u64>> cold_watch; ///< Image, frame_tick it was put on
    ::Common::unordered_set<ImageId> cold_watch_set;
    u64 cold_deleted{};       ///< Since the last report
    u64 cold_cap_frames{};    ///< Frames the sweep stopped at its per-frame cap, since the report
    /// Images made since the last cold sweep report, per (format, width), with their heights
    /// (fable52_answer_13 section 6: in the Stalchild fight new render target sizes one pixel
    /// apart, 528x513 to 528x527, kept making images). Logged with the cold sweep line.
    struct MadeShapes {
        u64 made{};
        std::vector<u32> heights; ///< Distinct, at most 64 kept
    };
    std::map<std::pair<u32, u32>, MadeShapes> made_shapes;
    void NoteMadeShape(const ImageInfo& info);
    u64 cold_kept_guarded{};  ///< Dropped from the watch: GPU-written and guarded
    u64 cold_superseded{};    ///< EXP58: guarded but superseded and old, deleted
    /// EXP58: whether a newer render target of another shape covers this image's memory.
    [[nodiscard]] bool ColdSuperseded(ImageId image_id, const ImageBase& image);
    u64 cold_kept_read{};     ///< Dropped from the watch: GPU-written and read later or by the CPU

    [[nodiscard]] static bool RelocationShaped(const ImageInfo& info) noexcept {
        return info.type == ImageType::e2D && info.resources.levels == 1 &&
               info.resources.layers == 1 && info.num_samples == 1;
    }
    [[nodiscard]] static u64 RelocationShapeKey(const ImageInfo& info) noexcept {
        return (u64{static_cast<u32>(info.format) & 0xFF} << 56) |
               (u64{info.num_samples & 0xF} << 52) |
               (u64{static_cast<u32>(info.resources.levels) & 0xF} << 48) |
               (u64{static_cast<u32>(info.resources.layers) & 0xFFF} << 36) |
               (u64{info.size.width & 0x3FFFF} << 18) | (info.size.height & 0x3FFFF);
    }
    /// JoinImages' only bad overlap, when the new render target could take its place: the same
    /// single-level, single-layer 2D shape and layout, a drift of less than its size, nothing
    /// unusual about it (sparse, converted, aliased, remapped, rescaled, CPU-read, a download
    /// queued, bound as another target of this clear, registered in another channel). Else none.
    [[nodiscard]] ImageId RelocationCandidate(const ImageInfo& new_info, GPUVAddr gpu_addr,
                                              DAddr cpu_addr);
    /// EXP64: retires JoinImages' bad overlaps at exactly gpu_addr that a new cleared render
    /// target supersedes (see the EXP64 notes in nx2_debug_control.h); returns how many.
    u32 RetireSlotPredecessors(const ImageInfo& new_info, GPUVAddr gpu_addr);
    /// EXP43=2: moves the image to gpu_addr / cpu_addr in place of a new one (fable_answer_12 2A
    /// steps 1-6); its VkImage, memory, views, framebuffers and scale state stay.
    void RelocateImage(ImageId image_id, GPUVAddr gpu_addr, DAddr cpu_addr);
    /// Drops the image's overlapping_images links from the other images (DeleteImage, relocation).
    void DropOverlapLinks(ImageId image_id);
    /// Sets `reason` on the image's shape (once per image and reason; logs the shape's first).
    void NoteRelocationGuard(ImageBase& image, RelocationGuard reason);
    /// The cross-frame read checks of PrepareImage and the texture memo's touch.
    void CheckCrossFrameUse(ImageBase& image, bool is_modification, bool invalidate) {
        if (image.last_write_frame == ~u64{0} || invalidate ||
            image.last_write_frame == RenderFrame()) {
            return;
        }
        const RelocationGuard reason = is_modification ? GuardLoaded : GuardSampled;
        if ((image.relocation_guard_noted & reason) == 0 && RelocationShaped(image.info)) {
            NoteRelocationGuard(image, reason);
        }
    }

    /// Drops a framebuffer from framebuffers_by_view and framebuffer_keys.
    void UnindexFramebuffer(FramebufferId framebuffer_id, const RenderTargets& key);

    /// The framebuffers each image view is in, and each framebuffer's key: RemoveFramebuffers
    /// takes a deleted image's framebuffers from here instead of scanning the whole map, which
    /// grows by one entry per drifted render target a frame (fable_answer_12 2E).
    ::Common::unordered_map<ImageViewId, boost::container::small_vector<FramebufferId, 4>>
        framebuffers_by_view;
    ::Common::unordered_map<FramebufferId, RenderTargets> framebuffer_keys;

    /// Mark an image as modified from the GPU
    void MarkModification(ImageBase& image) noexcept;

    /// EXP100: counts a registered, non-sparse image that is or was GPU-modified in
    /// gpu_image_filter until it is unregistered. Holder of the mutex only.
    void CountGpuModified(ImageBase& image) noexcept;

    /// Synchronize image aliases, copying data if needed
    void SynchronizeAliases(ImageId image_id);

    /// Prepare an image view to be used
    void PrepareImageView(ImageViewId image_view_id, bool is_modification, bool invalidate);

    /// Execute copies from one image to the other, even if they are incompatible
    void CopyImage(ImageId dst_id, ImageId src_id, std::vector<ImageCopy> copies);

    /// Bind an image view as render target, downloading resources preemtively if needed
    void BindRenderTarget(ImageViewId* old_id, ImageViewId new_id);

    /// Create a render target from a given image and image view parameters
    [[nodiscard]] std::pair<FramebufferId, ImageViewId> RenderTargetFromImage(
        ImageId, const ImageViewInfo& view_info);

    /// Returns true if the current clear parameters clear the whole image of a given image view
    [[nodiscard]] bool IsFullClear(ImageViewId id);

    /// UpdateRenderTargets(true): records a full clear on the targets the clear writes (the
    /// colour target it names, the depth target when depth is cleared), for ClassifyDrawTargets.
    void NoteFullClears();

    [[nodiscard]] std::pair<u32, u32> PrepareDmaImage(ImageId dst_id, GPUVAddr base_addr,
                                                      bool mark_as_modified);

    bool ImageCanRescale(ImageBase& image);
    void InvalidateScale(Image& image);
    bool ScaleUp(Image& image);
    bool ScaleDown(Image& image);
    u64 GetScaledImageSizeBytes(const ImageBase& image);

    /// The device memory an image counts for in total_used_memory between two driver readings:
    /// none for an unbacked image (too large to back, or over unwritten sparse blocks), what is
    /// bound to its layers for a SparseLayers one.
    u64 ImageMemoryBytes(const Image& image) const;

    void QueueAsyncDecode(Image& image, ImageId image_id);
    void TickAsyncDecode();

    void QueueAsyncUnswizzle(Image& image, ImageId image_id);
    void TickAsyncUnswizzle();

    Runtime& runtime;

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    std::deque<TextureCacheGPUMap> gpu_page_table_storage;

    /// EDEN_NX2_EXP31 (RT_FAST): whether UpdateRenderTargets' clean path
    /// would change nothing: the same targets were prepared for a draw, no image was modified,
    /// created or deleted since, in the same frame, and none needs an upload, alias sync or
    /// sparse backing.
    [[nodiscard]] bool RenderTargetsStillPrepared() const;
    void NoteRenderTargetsPrepared();
    /// EXP31 bit 2: the clean path ran although RenderTargetsStillPrepared; checks it changed
    /// nothing but the targets' own modification ticks. False (and logs) on a difference.
    bool VerifyRenderTargetsPrepared();

    RenderTargets render_targets;
    bool render_targets_out_of_memory = false;
    u64 render_targets_serial = 0;
    struct RenderTargetsPrepared {
        bool valid = false;
        u64 serial = 0;
        u64 modification_tick = 0;
        u64 frame_tick = 0;
        u64 bindings_epoch = 0;
        std::array<ImageViewId, NUM_RT + 1> views{};
    } rt_prepared;
    /// EXP31 counts since the last report (TickFrame, every 600 frames while it is on).
    u64 rt_fast_skips = 0;
    u64 rt_fast_clean_paths = 0;
    u64 rt_fast_verified = 0;
    u64 rt_fast_mismatches = 0;
    /// Changes whenever a prepared render target could need preparing again: images or views
    /// created, deleted, rescaled or CPU modified, descriptor pools moved, and every frame
    /// (the original bindings_epoch).
    u64 bindings_epoch = 1;
    /// See TextureMemoState::epoch (EXP32). 0 never matches.
    u64 texture_memo_epoch = 1;
    /// EDEN_NX2_TEX_BANK_GEN: the watched pages (std::unordered_map: entries stay where they are)
    /// and the epoch of their pointers.
    std::unordered_map<DAddr, TexBankPage> tex_bank_pages;
    u64 tex_bank_epoch = 1;
    u32 rt_active_mask = 0;
    std::array<ImageId, 8> rt_image_id{};
    ImageId rt_depth_image_id{};
    u64 texture_bindings_serial = 0;
    u64 last_feedback_loop_serial = 0;
    u64 last_feedback_texture_serial = 0;
    bool last_feedback_loop_result = false;
    bool feedback_depth_dirty = true;
    FramebufferId last_framebuffer_id{};
    u64 last_framebuffer_serial = 0;

    ::Common::unordered_map<RenderTargets, FramebufferId> framebuffers;
    ::Common::unordered_map<u64, std::vector<ImageMapId>, Common::IdentityHash<u64>> page_table;
    ::Common::unordered_map<ImageId, boost::container::small_vector<ImageViewId, 16>> sparse_views;

    DAddr virtual_invalid_space{};

    /// Applies EDEN_NX2_TC_EXPECTED_MIB / EDEN_NX2_TC_CRITICAL_MIB and logs the thresholds.
    void ApplyThresholdOverrides(bool log = true);

    /// EXP68 bit 1: shrinks a padded clear target to the scissor width (within 64-byte pitch
    /// alignment) when the clear overwrites it whole, and verifies on subsequent binds.
    void ApplyClearExactSize(ImageInfo& info, GPUVAddr gpu_addr, bool want_exact);
    /// FRITCHII_VRAM_BUDGET_MODE: the base thresholds lowered to budget_cap (no-op at ~0).
    void ApplyBudgetCap();
    /// Small cards (< 5 GiB): keeps an active image floor above non-image VRAM in expected/critical.
    void ApplySmallCardImageFloor(u64 budget, bool log = false);
    /// FRITCHII_VRAM_BUDGET_MODE (live): once a second, the driver's usage of the main heap
    /// against the budget (vk: the driver's; os: also the heap less what other processes hold)
    /// less FRITCHII_VRAM_HEADROOM_MIB; past it the cap comes down, with room it goes back up.
    void BudgetControl();
    /// FRITCHII_TC_CRITICAL_ENFORCE (live): before an image of about `incoming` bytes is made
    /// past the critical memory, clean images idle 10+ frames are deleted (no downloads).
    void EnforceCriticalAtCreate(u64 incoming);
    /// FRITCHII_TC_THRASH_GUARD (live): notes a deleted image, recognises one made again, and
    /// once a second sets how recently used an image the collector must leave alone.
    void NoteThrashDelete(const Image& image);
    void NoteThrashMade(const ImageInfo& info, GPUVAddr gpu_addr);
    void TickThrashGuard();
    /// FRITCHII_VRAM_OWNERS (live, seconds, default 10): one line with who holds device memory.
    void LogVramOwners();

    /// FRITCHII_SMALL_CARD (part Fit): device memory the driver counts that TickFrame leaves out
    /// of total_used_memory (deleted images still in flight, collector downloads, frees handed to
    /// the free thread, the image pool).
    [[nodiscard]] u64 UncountedDeviceBytes() const;
    /// FRITCHII_SMALL_CARD (part Fit): the thresholds' margin under the budget, 1/16 of it and at
    /// most 512 MiB (1/8 and 1 GiB without the profile).
    [[nodiscard]] static u64 SmallCardMargin(u64 budget) noexcept {
        return (std::min)(budget / 16, u64{512} << 20);
    }

    /// A GPU-written image dropped without being written back leaves its
    /// guest memory stale (the GPU's data is gone). Its whole pages are marked here and kept
    /// tracked, so CPU writes, unmaps and write-backs clear them. An image made entirely on marked
    /// pages starts without an upload: the upload would only copy stale bytes the GPU overwrites.
    /// FRITCHII_LOST_KEEP: not an image the CPU wrote whole since (its GPU data was void already,
    /// the guest bytes are newer), nor pages a newer write-back holds.
    void MarkStale(const ImageBase& image);
    void ClearStale(DAddr addr, size_t size);
    void ClearStaleImage(const ImageBase& image);
    /// Clears every device range [gpu_addr, gpu_addr + size) maps to.
    void ClearStaleGpu(GPUVAddr gpu_addr, size_t size);
    [[nodiscard]] bool IsAllStale(DAddr addr, size_t size) const;
    void UpdateStaleTracking(u64 run_begin, u64 run_pages, s32 delta);
    // Off here unless EDEN_NX2_TC_SKIP_STALE=1: no gain at an Ocarina of Time Hyrule Field save
    // (fresh launches, walking 1% low 40.2 off vs 36.5 on); the original measured +20% in Castle Town.
    bool tc_skip_stale_boot{[] {
        const char* value = std::getenv("EDEN_NX2_TC_SKIP_STALE");
        return value != nullptr && value[0] == '1';
    }()};
    /// EDEN_NX2_TC_SKIP_STALE=1 at boot, or EXP78 != 0 (live; fable52_answer_killuploads 3-4:
    /// the fight's kill bursts re-upload GPU-drawn targets from memory nobody wrote). Pages are
    /// marked from the moment it is on.
    [[nodiscard]] bool TcSkipStale() const noexcept {
        return tc_skip_stale_boot || VideoCore::Nx2Debug::Experiment(78) != 0;
    }
    /// Device memory pages (Core::Memory::YUZU_PAGEBITS), not this cache's 1 MiB lookup pages.
    static constexpr u64 STALE_PAGE_BITS = 12;
    std::vector<u64> stale_bits; ///< One bit per device page
    u64 stale_page_count{};
    u64 tc_stale_skips{};        ///< Uploads skipped (last 600 frames)
    u64 tc_stale_skip_bytes{};
    u64 tc_stale_gpu_written{};  ///< Stale but buffer-written pages: uploaded after all
    /// Some page of [addr, addr + size) is marked stale.
    [[nodiscard]] bool IsAnyStale(DAddr addr, size_t size) const;

    /// Lost contents (the dialogue text corruption, zortos293/Lichii #1310): an image made where
    /// a GPU-written image was dropped without a write-back starts with bytes that are not the
    /// GPU's: undefined device memory (the stale skip; with the EXP42 pool, another image's
    /// texels) or guest bytes older than the GPU's writes. Harmless when the game overwrites it
    /// whole; corruption on screen when it is sampled first. Counted per image at its first
    /// sampled use, with the drop that left the memory stale.
    enum LostBits : u8 {
        LostSkipped = 1,  ///< Upload skipped as all stale (undefined unless FRITCHII_STALE_ZERO)
        LostUploaded = 2, ///< Uploaded guest bytes some of which a dropped image left stale
        LostZeroed = 4,   ///< FRITCHII_STALE_ZERO cleared it instead of leaving it undefined
        LostDrawn = 8,    ///< Drawn into since without a clear of the whole image
    };
    /// FRITCHII_LOST_WRITEBACK's drop paths (CollectDownload::lost_path).
    static constexpr u8 LostPathCollector = 1; ///< The collector's bad overlaps
    static constexpr u8 LostPathCold = 2;      ///< The cold sweep (EXP58's superseded deletes too)
    static constexpr u8 LostPathSuperseded = 3; ///< FRITCHII_SWEEP_SUPERSEDED
    static constexpr u8 LostPathSlot = 4;      ///< EXP64's slot retire
    static constexpr size_t LostPathCount = 5;
    /// The last GPU-written images dropped without a write-back (MarkStale), for attribution.
    struct StaleDrop {
        DAddr cpu_addr;
        u64 size;
        GPUVAddr gpu_addr;
        u64 frame;
        u32 format;
        u32 width;
        u32 height;
        u8 reason;   ///< DeleteReason, or LostReasonRelocate
        u8 guard;    ///< The image's relocation_guard_noted
        u8 why;      ///< LostWhy: why no write-back saved it
        u64 shape_key; ///< RelocationShapeKey (a relocation's loss guards the shape)
        u8 known;    ///< (gpu_addr, shape_key) was in cold_resampled when dropped
        u8 counted;  ///< A cold-sweep loss over this drop was counted already
    };
    static constexpr u8 LostReasonRelocate = 0x40;
    // 4096 (was 1024): covered retire alone records ~140 drops a second at the OoT market, so a
    // drop sampled again 7+ s later had left the ring ("not recorded", 9-11 a 3-minute run).
    // Scanned only for a sampled image with lost contents (tens a run).
    static constexpr size_t StaleDropRing = 4096;
    std::array<StaleDrop, StaleDropRing> stale_drops{};
    size_t stale_drop_next{};
    size_t stale_drop_count{};
    bool marking_relocation{};
    /// FRITCHII_LOST_KEEP (live, default 1): why a GPU-written image went without a write-back.
    enum LostWhy : u8 {
        LostWhyNone = 0,     ///< The drop path has no write-back (join, unmap, relocation, ...)
        LostWhyOverCap = 1,  ///< Over FRITCHII_LOST_WRITEBACK's size cap
        LostWhyRefused = 2,  ///< A newer write-back holds some of its bytes
        LostWhyNoCopy = 3,   ///< Not downloadable (MSAA, trimmed levels, unsafe copy)
        LostWhyOff = 4,      ///< FRITCHII_LOST_WRITEBACK=0, or no queue (OpenGL)
        LostWhyBudget = 5,   ///< Learned or guarded, but past FRITCHII_LOST_KEEP's byte budget
        LostWhyLanding = 6,  ///< Refused at the landing (newer bytes landed over it since)
    };
    static constexpr size_t LostWhyCount = 7;
    struct LostContentStats {
        u64 made_skipped{};      ///< Images whose upload the stale skip dropped
        u64 made_zeroed{};       ///< Of them, cleared to zero (FRITCHII_STALE_ZERO)
        u64 made_uploaded{};     ///< Images uploaded over stale pages (some or all)
        u64 overwritten{};       ///< Cleared whole before any sampled use: harmless
        u64 sampled_undrawn{};   ///< Sampled before anything drew into them
        u64 sampled_drawn{};     ///< Sampled after draws that did not clear them whole
        u64 sampled_bytes{};
        std::array<u64, 10> sampled_by_reason{}; ///< DeleteReason (8 = relocation, 9 = unknown)
        u64 sampled_guarded{};   ///< Of the sampled: the dropped image had been read in a later frame
        u64 relocation_guards{}; ///< Shapes guarded from relocation after a loss it caused
        /// FRITCHII_LOST_WRITEBACK, by LostPath: queued, landed (and bytes), refused (a newer
        /// write-back or a since-downloaded overlap owns bytes: deleted as before), kept for a
        /// later pass (queue full), over the size cap (deleted as before).
        std::array<u64, LostPathCount> wb_queued{};
        std::array<u64, LostPathCount> wb_landed{};
        std::array<u64, LostPathCount> wb_landed_bytes{};
        std::array<u64, LostPathCount> wb_refused{};
        std::array<u64, LostPathCount> wb_kept{};
        std::array<u64, LostPathCount> wb_over_cap{};
        /// FRITCHII_LOST_KEEP: over the size cap but written back all the same, as the memory
        /// was read after an earlier loss (learned) or the image was read in a later frame
        /// (guarded); over the cap with such a reason but past the byte budget (dropped).
        std::array<u64, LostPathCount> wb_learned{};
        std::array<u64, LostPathCount> wb_guarded{};
        std::array<u64, LostPathCount> wb_over_budget{};
        u64 wb_cpu_written{};    ///< Not written back: the CPU wrote it since (guest bytes newer)
        /// Of the sampled, by LostWhy of the drop (0 = no write-back tried on that path).
        std::array<u64, LostWhyCount> sampled_by_why{};
        u64 learned_new{};       ///< Ranges learned (a sampled loss)
        u64 stale_cpu_skipped{}; ///< Drops not marked stale: the CPU wrote the image since
        u64 stale_pages_kept{};  ///< Pages not marked stale: a newer write-back holds them
        u64 logged{};
        u64 window_logged{};     ///< "Lost contents sampled" lines in this 10 s window
        u64 last_log_ns{};
    } lost_stats;
    /// FRITCHII_LOST_KEEP: memory ranges sampled with lost contents (the reader's range). The
    /// game reads that memory again after the image there went cold (OoT's exposure and HUD
    /// buffers; a one-frame black flash when it read zeros), so a later drop over one is written
    /// back whatever its size, under a byte budget. Ring of LostReadRing, oldest replaced.
    struct LostReadRange {
        DAddr begin;
        DAddr end;
    };
    static constexpr size_t LostReadRing = 128;
    std::array<LostReadRange, LostReadRing> lost_read_ranges{};
    size_t lost_read_next{};
    size_t lost_read_count{};
    /// FRITCHII_LOST_KEEP's byte budget for write-backs over the size cap (token bucket, MiB/s).
    u64 lost_keep_tokens{};
    u64 lost_keep_refill_ns{};
    [[nodiscard]] bool OverlapsLostRead(DAddr addr, u64 size) const;
    void LearnLostRead(DAddr addr, u64 size);
    /// Takes `bytes` of the budget; guarded (not learned) images leave half of it to learned ones.
    bool TakeLostKeepBudget(u64 bytes, bool learned, int keep);
    /// The drop that left [addr, addr + size) stale, newest first (nullptr: none recorded).
    [[nodiscard]] StaleDrop* FindStaleDrop(DAddr addr, size_t size);
    void RecordStaleDrop(const ImageBase& image);
    /// FRITCHII_STALE_ZERO: clears an image whose upload the stale skip dropped to zero (every
    /// level and layer) instead of leaving it undefined; false where it cannot (then undefined).
    bool ClearLostToZero(Image& image);
    /// PrepareImage: counts a lost-contents image's first sampled use (or its whole overwrite).
    void NoteLostContents(ImageBase& image, bool is_modification, bool invalidate);
    /// The 10 s "Lost contents" line (TickFrame), when anything happened.
    void LogLostContents();
    /// FRITCHII_LOST_WRITEBACK (live, default 1): a GPU-written image a drop path is about to
    /// delete is written back to guest memory first, on the collector's queue (EXP121's): the
    /// landing deletes it and clears its stale marks, so an image made there later uploads the
    /// GPU's pixels. Queued: the caller leaves it to the landing. Keep: no room in the queue,
    /// the caller keeps it for a later pass. Drop: delete as before (switch off, nothing GPU-only
    /// in it, over the size cap, a newer write-back owns some of its bytes, no queue).
    enum class LostWriteback : u8 { Drop, Queued, Keep };
    LostWriteback QueueLostWriteback(ImageId image_id, Image& image, u8 path);
    /// Recent collector write-backs (device range, the image's modification_tick): an older
    /// image's write-back must not land over a newer one's bytes.
    struct RecentWriteback {
        DAddr cpu_addr;
        u64 size;
        u64 modification;
    };
    static constexpr size_t RecentWritebackRing = 256;
    std::array<RecentWriteback, RecentWritebackRing> recent_writebacks{};
    size_t recent_writeback_next{};
    u64 recent_writeback_newest{}; ///< The newest modification in the ring (MarkStale's early out)
    void NoteWriteback(const ImageBase& image);
    [[nodiscard]] bool NewerWritebackOver(const ImageBase& image) const;

    /// FRITCHII_ZERO_UPLOAD counters (the 10 s line).
    struct ZeroUploadStats {
        u64 zero_images{};     ///< All zero (cleared in mode 1, counted in mode 2)
        u64 zero_bytes{};      ///< Their guest bytes
        u64 zero_ns{};         ///< Their checks
        u64 data_images{};     ///< Checked, not all zero: uploaded as before
        u64 data_bytes{};      ///< Bytes looked at before the first non-zero one
        u64 data_ns{};
        u64 sparse_skipped{};  ///< Eligible but sparse in guest memory (footprint over 2x)
        s64 last_log_ns{};
        u32 logged{};          ///< Per-image lines so far (the first 40 of 4 MiB or more)
    } zero_upload;

    /// The CPU address an image at gpu_addr is keyed by: its backing, or a fixed stand-in.
    [[nodiscard]] std::optional<DAddr> ImageCpuAddress(GPUVAddr gpu_addr, size_t size) const;

    void BumpImageViewGeneration(ImageViewId id) {
        if (id.Value() >= image_view_generations.size()) {
            image_view_generations.resize(id.Value() + 1);
        }
        ++image_view_generations[id.Value()];
    }

    std::vector<u32> image_view_generations; ///< ImageViewGeneration
    u64 scale_epoch{};                       ///< ScaleEpoch
    bool has_deleted_images = false;
    /// True while a clear binds its render targets (UpdateRenderTargets(true)), so JoinImages can
    /// skip the upload of a new target the clear overwrites whole.
    bool clearing_render_targets = false;
    /// True while FindRenderTargetView looks a render target up (clear or draw), so JoinImages
    /// knows a new image is a render target (EDEN_NX2_RT_DEPTH_NOUPLOAD bit 1).
    bool render_target_lookup = false;
    bool is_rescaling = false;
    u64 total_used_memory = 0;
    u64 minimum_memory;
    u64 expected_memory;
    u64 critical_memory;
    size_t gpu_unswizzle_maxsize = 0;
    size_t swizzle_chunk_size = 0;
    u32 swizzle_slices_per_batch = 0;

    struct BufferDownload {
        GPUVAddr address;
        size_t size;
    };

    using PendingDownload = PendingImageDownload<Image>;

    Common::SlotVector<Image> slot_images;
    Common::SlotVector<ImageMapView> slot_map_views;
    Common::SlotVector<ImageView> slot_image_views;
    Common::SlotVector<ImageAlloc> slot_image_allocs;
    Common::SlotVector<Sampler> slot_samplers;
    Common::SlotVector<Framebuffer> slot_framebuffers;
    Common::SlotVector<BufferDownload> slot_buffer_downloads;

    // TODO: This data structure is not optimal and it should be reworked

    std::vector<PendingDownload> uncommitted_downloads;
    std::deque<std::vector<PendingDownload>> committed_downloads;
    std::vector<AsyncBuffer> uncommitted_async_buffers;
    /// MayHaveAsyncFlushWork's note: raised with every add to the two lists above.
    std::atomic<bool> async_flush_work_noted{false};
    std::deque<std::vector<AsyncBuffer>> async_buffers;
    std::deque<AsyncBuffer> async_buffers_death_ring;

    struct LRUItemParams {
        using ObjectType = ImageId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;

 #ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
 #else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Image, TICKS_TO_DESTROY> sentenced_images;
    /// Device memory of the images in each slot of sentenced_images (same index): deleted, but
    /// counted in the driver's usage until the ring destroys them.
    std::array<u64, TICKS_TO_DESTROY> sentenced_image_bytes{};
    size_t sentenced_image_index = 0;
    /// Deleted images on runtimes that report GPU progress (CurrentTick, IsFree): destroyed, with
    /// their device memory, as soon as the GPU is past them rather than TICKS_TO_DESTROY frames
    /// later. While the collector deletes dozens a frame, eight frames of them held 1-1.5 GB of
    /// VRAM above its threshold (title 1 on an 8 GB card: 6.6-7.0 GB in use against 5 GiB).
    struct GpuSentencedImage {
        u64 tick;
        u64 bytes;
        Image image;
    };
    std::deque<GpuSentencedImage> gpu_sentenced_images;
    u64 gpu_sentenced_bytes{};

    /// Fritchii's free guard (helper_bi_answer.md). An image deleted while a download of it was
    /// queued lives in the download records (RetainPendingImageDownloads) and used to be destroyed
    /// when that download landed: the GPU had passed the download's copy then, not the draws and
    /// copies recorded after it and before the delete (AMD: device lost). Each such image is also
    /// held here until the GPU passes the tick it was deleted at.
    struct RetainedImageKeeper {
        u64 tick;
        std::shared_ptr<Image> image;
    };
    std::deque<RetainedImageKeeper> retained_image_keepers;
    /// Session totals, logged every 600 frames once one is not 0.
    struct FreeGuardStats {
        u64 retained_held{};     ///< images deleted with a download queued, held by the guard
        u64 landed_early{};      ///< of them, downloads landed before the GPU was known past the
                                 ///< delete: the old code destroyed the image there, in use
        u64 freed_early{};       ///< must be 0: a landing destroyed a retired image before the
                                 ///< GPU was known past its delete (the guard's own check)
        u64 unswizzle_dropped{}; ///< async unswizzles dropped with their deleted image
        u64 logged_total{};      ///< the sum above when last logged
    };
    FreeGuardStats free_guard{};

    /// EXP121: collector downloads recorded without a wait, oldest first.
    struct CollectDownload {
        ImageId image_id;
        AsyncBuffer map;
        boost::container::small_vector<BufferImageCopy, 16> copies;
        u64 tick;              ///< GPU tick of the copy
        u64 frame;             ///< frame_tick when recorded
        u64 lru_tick;          ///< the image's LRU tick then; another one means it was used
        u64 image_modification; ///< the image's modification_tick then
        u64 cache_modification; ///< the cache's modification_tick then
        u64 staging_bytes;
        u64 memory_bytes;      ///< device memory counted as freed while queued
        bool cpu_modified;     ///< the image was CpuModified then (a CPU write since drops it)
        u8 lost_path;          ///< FRITCHII_LOST_WRITEBACK's drop path (0 = the collector's own)
    };
    std::deque<CollectDownload> collect_downloads;
    u64 collect_download_staging{};
    u64 collect_download_memory{};
    struct CollectDownloadStats {
        u64 waits, made, landed, dropped_used, dropped_written, dropped_deleted, fallback_cap,
            fallback_oom, mismatches, land_frames;
        u64 dropped_cpu;    ///< of dropped_written: a CPU write to the image since it was queued
        u64 dropped_under;  ///< of dropped_written: an overlap the GPU wrote since
        u64 landed_under;   ///< landed although such an overlap is still GPU-only (LAND_UNDER)
    } collect_stats{};

    /// FRITCHII_TC_COLLECT_OVER and FRITCHII_TC_COLLECT_EVICTABLE, since the last report.
    struct CollectPortStats {
        u64 over_passes{}; ///< Aggressive passes past the critical memory
        u64 over_extra{};  ///< Deletes those passes were allowed on top of 40
        u64 over_young{};  ///< Of them, 1 GiB or more over (images unused for 3 frames go)
        u64 over_peak{};   ///< Most bytes over the critical memory
        u64 deleted_aggressive{}; ///< Images deleted while past the critical memory
        u64 skipped{};     ///< EVICTABLE: images skipped before they used the pass's budget
    } collect_port_stats{};

    /// Bytes of images per device memory block, for the collector (refreshed every 60 frames).
    std::unordered_map<u64, u64> block_image_bytes;
    u64 block_image_bytes_frame{~0ULL};
    DelayedDestructionRing<ImageView, TICKS_TO_DESTROY> sentenced_image_view;
    DelayedDestructionRing<Framebuffer, TICKS_TO_DESTROY> sentenced_framebuffers;

    ::Common::unordered_map<GPUVAddr, ImageAllocId> image_allocs_table;

    Common::ScratchBuffer<u8> swizzle_data_buffer;
    Common::ScratchBuffer<u8> unswizzle_data_buffer;
    Tegra::Texture::UnswizzleWorker unswizzle_worker;

    u64 modification_tick = 0;
    u64 frame_tick = 0;

    Common::ThreadWorker texture_decode_worker{1, "TextureDecoder", {},
                                               Common::ThreadPlacement::Efficiency};
    std::vector<std::unique_ptr<AsyncDecodeContext>> async_decodes;
    /// Guest memory images were uploaded from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_ranges;

    std::deque<PendingUnswizzle> unswizzle_queue;
    u8 current_unswizzle_frame;

    // Join caching
    boost::container::small_vector<ImageId, 4> join_overlap_ids;
    ::Common::unordered_set<ImageId> join_overlaps_found;
    boost::container::small_vector<ImageId, 4> join_left_aliased_ids;
    boost::container::small_vector<ImageId, 4> join_right_aliased_ids;
    ::Common::unordered_set<ImageId> join_ignore_textures;
    boost::container::small_vector<ImageId, 4> join_bad_overlap_ids;
    struct JoinCopy {
        bool is_alias;
        ImageId id;
    };
    boost::container::small_vector<JoinCopy, 4> join_copies_to_do;
    ::Common::unordered_map<ImageId, size_t> join_alias_indices;

    RenderTargetStats rt_stats;
    std::array<u64, 3> rt_stats_base{}; ///< images_created, images_deleted, uploaded_bytes
};

} // namespace VideoCommon
