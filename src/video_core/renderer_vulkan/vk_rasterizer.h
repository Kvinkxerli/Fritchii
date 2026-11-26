// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string_view>

#include <boost/container/static_vector.hpp>

#include "common/common_types.h"
#include "common/dkb_cpu_trace.h"
#include "common/dkb_frame_trace.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/oot_network.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/upscaler_fallback_log.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/vk_async_compute_skip.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_flash_watch.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_mem_stage.h"
#include "video_core/renderer_vulkan/vk_seam_clock.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/fritchii_delta_path.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/nx2_runahead.h"

namespace Core {
class System;
}

namespace Core::Frontend {
class EmuWindow;
}

namespace Tegra {

namespace Engines {
class Maxwell3D;
class KeplerCompute;
}

} // namespace Tegra

namespace Vulkan {

struct FramebufferTextureInfo;

class OneFollowerDraw;
class StateTracker;

class AccelerateDMA : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache, TextureCache& texture_cache,
                           Scheduler& scheduler);

    bool BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) override;

    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;

    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;

    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    template <bool IS_IMAGE_UPLOAD>
    bool DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                            const Tegra::DMA::BufferOperand& src,
                            const Tegra::DMA::ImageOperand& dst);

    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    Scheduler& scheduler;
};

class SimLinkHog;

class RasterizerVulkan final : public VideoCore::RasterizerInterface,
                               protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                              Tegra::MaxwellDeviceMemoryManager& device_memory_,
                              const Device& device_, MemoryAllocator& memory_allocator_,
                              StateTracker& state_tracker_, Scheduler& scheduler_);
    ~RasterizerVulkan() override;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawIndirect() override;
    [[nodiscard]] bool Nx2ListDrawSupported() const override;
    [[nodiscard]] bool Nx2ListDispatchSupported() const override;
    [[nodiscard]] bool MemStageQmd(Tegra::MemoryManager& memory, GPUVAddr address,
                                   std::span<u32, 64> out) override;
    void Nx2KeepListWords(u64 keep_id, GPUVAddr address, u32 count) override;
    [[nodiscard]] bool Nx2ReadKeptListWords(u64 keep_id, std::span<u32> words) override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    [[nodiscard]] u64 GraphicsUniformBindState(size_t stage, u32 index) const override {
        return buffer_cache.UniformBindState(stage, index);
    }
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void FlushRegionForRead(DAddr addr, u64 size) override;
    void FinishRead(DAddr addr, u64 size) override;
    GpuWrittenList ReadGpuWrittenCommandList(DAddr addr, std::span<u8> out,
                                             bool past_sync) override;
    void NoteGpuWrittenCommandListSynced(DAddr page) override;
    void FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    bool WaitFenceDownload(u64 sequence) override;
    /// GetFlushArea's answer under both caches' mutexes.
    VideoCore::RasterizerDownloadArea GetFlushAreaLocked(DAddr addr, u64 size);
    /// GetFlushAreaLocked that also compares the buffer cache's page mirrors' answer with what
    /// the locked path decides (EDEN_NX2_EXP86=2).
    VideoCore::RasterizerDownloadArea GetFlushAreaLockedVerifyMirror(DAddr addr, u64 size);
    /// GetFlushAreaLocked that also compares the texture cache's count of GPU-modified images
    /// with what its locked look finds (EDEN_NX2_EXP100=2). buffer_clean: the fast path's buffer
    /// side answer, taken before the locks.
    VideoCore::RasterizerDownloadArea GetFlushAreaLockedVerifyImage(DAddr addr, u64 size,
                                                                    bool buffer_clean);
    bool HasPendingQueries(DAddr addr, u64 size) override;
    bool HasCachedReports(DAddr addr, u64 size) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) override;
    void InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    void OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) override;
    bool TryCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void InvalidateTextureDataCache() override;
    bool GatherOnStage() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    u64 SyncOperationStamped(std::function<void()>&& func) override;
    bool AppendToLastOperation(u64 stamp, void (*append)(void*), void* context) override;
    void SyncOperationKeepStamp(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force = true) override;
    bool FencesQuiet(bool strict, u64& epoch) override;
    u64 FenceWorkEpoch() const override;
    void FlushQueryResults(DAddr addr, u64 size) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    [[nodiscard]] bool TiledCacheBarrierIsNoop() const noexcept override {
        return true; // TiledCacheBarrier is empty here
    }
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateConditionalRendering() override;
    bool HasDrawTransformFeedback() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    bool SupportsGameDlssNetwork() const override;
    bool UsesNvidiaDlss() const override;
    bool CanProbeGameDlssOutput() const override {
        return true;
    }
    std::optional<VideoCore::GameDlssProbe> ProbeGameDlssOutput(
        const Tegra::Engines::Fermi2D::Surface& src,
        const Tegra::Engines::Fermi2D::Surface& dst) override;
    bool AccelerateTemporalUpscale(const Tegra::Engines::Fermi2D::Surface& src,
                                   const Tegra::Engines::Fermi2D::Surface& dst,
                                   const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                                   const Tegra::Texture::TICEntry* depth,
                                   std::array<f32, 2> jitter,
                                   std::array<f32, 2> motion_scale, bool use_history = true) override;
    VideoCommon::OotNetwork::BypassState* OotNetworkBypass() override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void AccelerateInlineToMemoryImpl(GPUVAddr address, size_t copy_size,
                                      std::span<const u8> memory);
    void AccelerateInlineWords(GPUVAddr address, size_t copy_size,
                               std::span<const u8> memory) override;
    /// AccelerateInlineToMemoryImpl's parts: what it notes first (counters, stats, traces), its
    /// EXP85 test (no GPU write ever marked the pages: the caches need no look), and that
    /// branch's work after the write; `per_word`: the invalidations accumulated word by word.
    void InlineUploadNote(GPUVAddr address, std::optional<DAddr> cpu_addr, size_t copy_size,
                          std::span<const u8> memory);
    [[nodiscard]] bool InlineSkipsCaches(DAddr cpu_addr, size_t copy_size);
    void InlineUntouchedTail(GPUVAddr address, DAddr cpu_addr, size_t copy_size, bool per_word);
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;

    void InitializeChannel(Tegra::Control::ChannelState& channel) override;

    void BindChannel(Tegra::Control::ChannelState& channel) override;

    void ReleaseChannel(s32 channel_id) override;
    std::optional<FramebufferTextureInfo> AccelerateDisplay(const Tegra::FramebufferConfig& config,
                                                            VAddr framebuffer_addr,
                                                            u32 pixel_stride);

private:
    /// One follower, part c (vk_one_follower_draw.cpp): R's draws and the verify compares reach
    /// the caches, the ring and the engine pointers.
    friend class OneFollowerDraw;
    /// Created at construction; idle unless FRITCHII_ONEF steps 3 and 4 are on.
    std::unique_ptr<OneFollowerDraw> onef_draw;
    /// R: the vertex and fragment program hashes of the last draw PrepareDraw found a pipeline
    /// for (part c's facts compare reads them after a lent draw).
    std::array<u64, 2> onef_pipeline_hashes{};
    bool onef_pipeline_seen{};
    std::string onef_precheck_note; ///< R: the pre-record check's must-be-0 text
    /// E: a render enable that can begin host conditional rendering was lent (sticks). While
    /// false, FRITCHII_ONEF_QUERY_OPS sends no override/True/False render enable.
    bool onef_condition_may_be_set{};
    [[nodiscard]] bool MayEndConditionalRendering() const noexcept override;
    /// R: inside a posted clear (FRITCHII_ONEF_OP_CLEAR): E ran its FlushCaching.
    bool onef_clear_posted{};
    /// E: an unlent list draw with a keep id was pushed since R was last drained: R records its
    /// kept words when it plays it, so a read of kept words on E drains R first.
    bool onef_list_keeps_unplayed{};
    /// DrawListWords' test on `engine` (its draw state and registers): the list draw reads its
    /// two words on the CPU (one follower: such a list draw stays lent).
    [[nodiscard]] static bool OneFollowerListOnCpu(const Tegra::Engines::Maxwell3D& engine);
    /// E (FRITCHII_ONEF_SYNC_MIRROR, fable_b 16.2): the mirror of R's samples counter "started"
    /// bit: set by every draw or clear E issues with the count enabled, cleared by a reset.
    bool onef_samples_started{};
    /// FRITCHII_ESPLIT's queued mode, E1: what onef_samples_started will be on E2 once E2 ran
    /// everything E1 queued (E1 sets it at a draw or clear with the count on, clears it when it
    /// queues a samples reset). The mirror itself is E2's then: the reports and resets that read
    /// and clear it run on E2, so E1's draws set it through E2's queue, in ring order.
    bool onef_mirror_queued{};
    /// E -> R for a lent samples report: E's bit at the report (R compares in the lent call).
    bool onef_mirror_lent_valid{};
    bool onef_mirror_lent_bit{};
    /// E: the mirror's update at a draw or clear (the count enable register, E's engine).
    void OneFollowerMirrorDraw();
    /// E (FRITCHII_ONEF_POST_FENCE): a query sync may be owed at the next fence: a samples report
    /// made with the mirror started, or any other report, since the last lent fence or wait for
    /// idle (R syncs there with E parked).
    bool onef_sync_owed{};
    /// E (fable_b 16.2): the 16-byte slots whose values R will sync at the next fence: samples
    /// reports made with the mirror started (cleared at every fence), payload releases (a
    /// superset kept until a lent fence: memory already holds their value, they are lent).
    std::vector<u64> onef_sync_samples;
    std::vector<u64> onef_sync_payloads;
    /// E: a lent fence: R synced and landed everything before it, E parked.
    void OneFollowerFenceLent();
    /// R: inside a posted fence (its closing gather ran on E).
    bool onef_fence_posted{};
    /// FRITCHII_ONEF_TICK_ON_E (default 1): a frame end R reached outside a lent call (the
    /// composite executer of an increment R played) leaves TickFrame to E, which lends it at its
    /// next fence: TickFrame ticks E's staging pools, memory stage and caches.
    std::atomic<bool> onef_frame_tick_deferred{};
    /// E: runs a frame tick R left (lent: R runs it with E parked).
    void OneFollowerDeferredTick();
    /// E: whether a fence op can be posted now (counted by why not), and the post: 0 sync
    /// point, 1 reference, 2 forced release; with FRITCHII_ONEF_SEM_E 3 a sync point whose
    /// increments E made, 4 a SignalFence whose callback E ran (`e_callback`), 5 a
    /// fence-flagged payload report E wrote (`note`).
    [[nodiscard]] bool OneFollowerFenceClean();
    struct OneFollowerPayloadNote {
        DAddr device;
        GPUVAddr gpu_addr;
        u32 payload;
        u32 subreport;
        bool has_timestamp;
    };
    /// FRITCHII_ONEF_POST_FENCE_REPORT: the counter fence report a posted fence of kind 7
    /// makes on R (its CounterReport, then its SignalFence there).
    struct OneFollowerFenceReport {
        GPUVAddr gpu_addr;
        DAddr device;
        VideoCommon::QueryType type;
        VideoCommon::QueryPropertiesFlags flags;
        u32 payload;
        u32 subreport;
        u8 mirror; ///< 0 off, 1 E's samples bit clear, 2 set
    };
    void OneFollowerPostFence(u8 kind, u32 value, std::function<void()>* e_callback = nullptr,
                              const OneFollowerPayloadNote* note = nullptr,
                              const OneFollowerFenceReport* report = nullptr);
    /// E, under the buffer lock in a Query seam: the buffers for the values R syncs next (16.2),
    /// null when none are listed; the samples list is cleared (R syncs them there).
    [[nodiscard]] std::vector<OneFollowerSyncTarget>* OneFollowerSyncTargetsLocked();
    /// R, after a posted fence or wait for idle: the unused-target check, then frees them.
    void OneFollowerCloseTargets(std::vector<OneFollowerSyncTarget>* targets);
    /// E (FRITCHII_ONEF_POST_IDLE): WaitForIdle posted.
    void OneFollowerPostIdle();
    /// R: the pages of samples reports made where E's mirror had the counter started and R's
    /// had not (E's superset), for the "sync target unused" check of the next posted fence.
    std::vector<u64> onef_superset_pages;
    void OneFollowerNoteSuperset(DAddr device);
    /// E: FRITCHII_ONEF_GATHER_E's runahead condition (logged once per change).
    [[nodiscard]] bool OneFollowerRunaheadOff(const char* where);
    /// E: the pages of the report ops it pushed (FRITCHII_ONEF_QUERY_OPS), for item 2's test.
    std::unique_ptr<VideoCommon::PageFilter> onef_report_pages;
    /// fable_b 14.4 cut 5: the union of the texture, shader, query and report page filters (each
    /// marks it too, PageFilter::AlsoMark), so FRITCHII_ONEF_UNION_FILTER tests one filter per
    /// range at E's invalidations and the four only on a hit.
    VideoCommon::PageFilter onef_union_pages;
    /// E (FRITCHII_ONEF_INVAL_E): whether this invalidation runs on E (no range on R's pages).
    [[nodiscard]] bool OneFollowerInvalidationOnEngine(
        std::span<const std::pair<DAddr, std::size_t>> sequences);
    /// E (FRITCHII_ONEF_COPY_E): whether AccelerateInlineWords runs here (stamps its pages).
    [[nodiscard]] bool OneFollowerInlineWordsOnEngine(GPUVAddr address, size_t copy_size);
    /// E (FRITCHII_ONEF_GATHER_E): applies a gather's buffer half here; false = lend it.
    [[nodiscard]] bool OneFollowerGatherOnEngine(std::span<const std::pair<DAddr, u64>> ranges);
    /// E (FRITCHII_ONEF_FLUSH_E, fable_b 17.3 (a)): whether a flush of the range would do
    /// nothing at R's place, from what E can test exactly.
    [[nodiscard]] bool OneFollowerFlushClean(DAddr addr, u64 size, VideoCommon::CacheType which);
    /// E: fable_b 12.3 fix 3's price (CountLentFence) at a lent fence.
    void OneFollowerCountFence();

    // FRITCHII_ONEF_OP_DISPATCH (vk_rasterizer.cpp, "Posted dispatches").
    /// E, DispatchCompute's first step with the switch on: true = done here (posted, or the
    /// lookup's hooks threw, which propagates as a lent dispatch's exception would). False = lend
    /// it; `onef_dispatch_lent_pipeline` is set when E looked the pipeline up already.
    [[nodiscard]] bool OneFollowerPostDispatch();
    /// FRITCHII_ASYNC_COMPUTE_SKIP (vk_async_compute_skip.h): DispatchCompute's decision for a
    /// pipeline that counts as unbuilt, both caches' locks held, before the fill. True = skip
    /// the dispatch (nothing was marked or uploaded). False = run it (it waits for the build as
    /// before, or got it within the frame's wait budget).
    [[nodiscard]] bool SkipUnbuiltDispatch(ComputePipeline& pipeline, bool indirect,
                                           bool special_pass);
    /// The written ranges of the bound dispatch (its storage pointers bound), as its marks
    /// will be; false when one is not bounded (G4). Buffer cache's lock held.
    [[nodiscard]] bool CollectComputeWrites(
        boost::container::small_vector<AsyncComputeSkip::Range, 8>& ranges);
    /// R, the dispatch op: takes ring B's packet `number` and records the dispatch with
    /// `pipeline` (E's lookup) on R's view of the compute engine.
    void OneFollowerPlayDispatch(ComputePipeline* pipeline, u64 number);
    /// R, after a played image-writing dispatch (fable_b 27.9): counts its images on pages the
    /// front reads, and with FRITCHII_ONEF_DISPATCH_RANGES compares the descriptors and ranges E
    /// recorded (`view`: the packet's OneFollowerDispatchView) with what R bound
    /// (Check::DispatchImageRange).
    void OneFollowerCheckPlayedImages(const ComputePipeline& pipeline, const void* view);
    /// E -> R through a lend (E parked): the pipeline E looked up for the lent dispatch (its
    /// hooks ran on E): DispatchCompute on R uses it instead of a second lookup.
    ComputePipeline* onef_dispatch_lent_pipeline{};
    /// R: the posted dispatch DispatchCompute plays now (its pipeline), else null.
    ComputePipeline* onef_dispatch_play{};
    /// R: E's word that the played dispatch writes images (FRITCHII_ONEF_OP_DISPATCH bit 2).
    bool onef_dispatch_writes_images{};
    /// R: its view of the bound channel's compute engine for played dispatches (no methods run
    /// on it), made for the memory manager it was made with.
    std::unique_ptr<Tegra::Engines::KeplerCompute> onef_kepler_view;
    Tegra::MemoryManager* onef_kepler_view_memory{};
    /// E: the handle values of the dispatch it fills.
    std::vector<u32> onef_dispatch_values;
    /// R, FRITCHII_ONEF_OP_DISPATCH on: each indirect dispatch's three grid words, copied on the
    /// GPU right before it into a download slot, checked against the device limit once its
    /// submission is done (posted ones: IndirectGridOverLimit, must be 0; lent ones: the dry
    /// compare). The words are GPU-written: the CPU cannot read them before the dispatch.
    struct OneFollowerGridCheck {
        StagingBufferRef ref;
        u64 tick;
        u64 shader;
        bool posted;
        bool audit; ///< FRITCHII_GRID_AUDIT's copy (not the one follower's)
    };
    std::vector<OneFollowerGridCheck> onef_grid_checks;
    /// R: checks the copies whose submission is done; the others wait for a later call.
    void OneFollowerCheckGrids();
    /// E (FRITCHII_ONEF_COPY_E): runs an inline upload here; false = lend it whole.
    [[nodiscard]] bool OneFollowerInlineOnEngine(GPUVAddr address, size_t copy_size,
                                                 std::span<const u8> memory);

    /// False when no page of the range was ever written by the GPU through the buffer cache.
    bool MayBeGpuModified(DAddr addr, u64 size);

    static constexpr const u64 NEEDS_D24[] = {
        0x01006A800016E000ULL, // SSBU
        0x0100E95004038000ULL, // XC2
        0x0100A6301214E000ULL, // FE:Engage
    };
    static constexpr size_t MAX_TEXTURES = 192;
    static constexpr size_t MAX_IMAGES = 48;
    static constexpr size_t MAX_IMAGE_VIEWS = MAX_TEXTURES + MAX_IMAGES;

    static constexpr VkDeviceSize DEFAULT_BUFFER_SIZE = 4 * sizeof(float);

    template <typename Func>
    void PrepareDraw(bool is_indexed, Func&&, bool is_indirect = false);

    void CaptureIndirectTrace();
    void CollectIndirectTrace();
    void TraceDrawFailure(std::string_view reason);
    bool IsNx2IndirectReset(bool is_indexed, u32 instance_count);
    /// The same test on `engine` and `memory` (E's own, at its place: FRITCHII_ONEF_UNLENT_RESET).
    [[nodiscard]] static bool IsNx2IndirectResetOn(Tegra::Engines::Maxwell3D& engine,
                                                   Tegra::MemoryManager& memory, bool is_indexed,
                                                   u32 instance_count);
    /// A compute dispatch was recorded since the last barrier for GPU-counted draws.
    bool compute_since_count_barrier{};
    /// Remembers render target 0 of a draw with one of Ocarina of Time's tonemap shaders.
    void TrackOotTonemappedImage(const GraphicsPipeline& pipeline);
    bool TryNx2IndirectReset(bool is_indexed, u32 instance_count);

    void FlushWork();

    void TraceRenderState(bool is_indexed, const char* outcome, const Framebuffer* framebuffer);

    /// Records the copies of the command list pages the work just recorded wrote and submits, so
    /// the GPU makes them now (BufferCache::Nx2CopyListPages). True if it recorded any.
    bool SubmitNx2ListCopies(u64 writer);

    /// Records the dynamic state of the draw being prepared with `pipeline` (null for
    /// DrawTexture's blit).
    void UpdateDynamicStates(const GraphicsPipeline* pipeline);

    void HandleTransformFeedback();

    void UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs);

    void UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineStipple(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    bool alpha_to_coverage_enable{}; ///< Last value recorded with SetAlphaToCoverageEnableEXT
    void UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs);
    void UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs, const GraphicsPipeline* pipeline);
    void UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs);

    void UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs);

    /// EXP83 (fable52_answer_microcuts 3.2): the vertex input state last recorded in this command
    /// buffer, so an identical one is not sent again. Valid while vertex_input_memo_generation
    /// equals the state tracker's and no pipeline without dynamic vertex input was bound since.
    boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> vertex_input_memo_bindings;
    boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> vertex_input_memo_attributes;
    u64 vertex_input_memo_generation{~u64{0}};
    bool vertex_input_memo_valid{};
    u64 vertex_input_sent{};
    u64 vertex_input_repeats{};
    /// FRITCHII_VERTEX_INPUT_COMPACT's counts for the "Vertex input compact" line.
    u64 vertex_input_recorded{};
    u64 vertex_input_bindings_described{};
    u64 vertex_input_bindings_referenced{};
    u64 vertex_input_attributes_described{};
    u64 vertex_input_command_bytes{};

    /// Clears a 3D color target whose slices the framebuffer can't reach. Returns true when handled.
    bool ClearVolumeSlices(const Framebuffer* framebuffer, u32 first_slice, u32 slice_count);
    void FrameDumpPass(const Framebuffer* framebuffer);
    bool OcarinaTemporalEnabled() const;
    /// At OoT's disk cache load (before any of its pipelines and images): DLSS and FSR take only images at
    /// the game's own size, so with either one upscaling the resolution scale goes to 1x.
    void ApplyOcarinaResolution(u64 title_id);
    /// `record`: the draw packet's record of the uniform slots (null: ask the buffer cache).
    void TrackOcarinaCamera(const MemStage::TrackerRecord* record);
    /// Offsets the viewport of Ocarina's scene draws (depth-tested, into the scene depth) by a
    /// sub-pixel amount that changes every frame while FSR replaces the game's upscale, so FSR
    /// gets new sample positions each frame (the game renders without jitter). The offset goes to
    /// FSR with the frame.
    void UpdateOcarinaJitter();
    void TrackOcarinaActor(u64 shader_hash, bool is_indexed, bool is_indirect,
                           const MemStage::TrackerRecord* record);
    /// FRITCHII_OOT_ACTOR_TRACE: one log line for a draw that could carry a character.
    void TraceOcarinaActorDraw(const GraphicsPipeline& pipeline, bool is_indexed);
    /// FRITCHII_MEMSTAGE_SHADOW: the trackers' record against today's query.
    void VerifyTrackerRecord(const MemStage::TrackerRecord* record);
    // One follower (one_follower.h), on E: a draw lent to R (from step 3 part (c) routes it).
    void OneFollowerLendDraw(bool is_indexed, u32 instance_count);
    void OneFollowerDrawIndirect();
    /// GpuSplit's idle hook (lent to R with one follower on); `idle`: InIdleWait on the back.
    void RunIdleHook(bool idle);
    // FRITCHII_MEMSTAGE_THREAD = 2 (vk_rasterizer.cpp).
    void SetRingBackend();
    bool StageFillDraw(VideoCommon::MemStageShadowEngine& copy, bool is_indexed, u64 position,
                       u64 seq, u64 hash, std::vector<std::pair<u64, std::size_t>>& ranges);
    void StageApplyRanges(std::span<const std::pair<DAddr, std::size_t>> ranges, bool bracket);
    /// GPUMem, StageFillDraw's join 1: stage_facts for the copy's programs; false on a miss.
    /// FRITCHII_MEMSTAGE_SETUP_KEEP: kept from the last lookup while its inputs are unchanged.
    bool StageSetupFacts(const VideoCommon::MemStageShadowEngine& copy);
    /// Step 7 (GPUMem): fills the window must not run ahead of (StageFillDraw).
    bool StageDrawWritesBuffers(const VideoCommon::MemStageShadowEngine& copy) const;
    /// `claim`: the collected ranges a fill claims (FRITCHII_MEMSTAGE_MARKS_AHEAD: marks ahead
    /// count on the ranges' pages, as WriteMemory acts on them), else constant buffer words.
    bool StageRangesOverGpuWritten(std::span<const std::pair<DAddr, std::size_t>> ranges,
                                   bool claim = false);
    /// FRITCHII_MEMSTAGE_GATHER (GPUMem, level at a gather's entry): the back runs the gather's
    /// handler now, as a request, unless `engine` has a deferred draw; then its counters.
    bool StageRunGather(Tegra::Engines::Maxwell3D& engine, u64& draw_seq);
    /// The served gather's game write ranges, while the back runs it (OnCacheInvalidations).
    void ProbeGatherRanges(std::span<const std::pair<DAddr, u64>> ranges);
    /// FRITCHII_MEMSTAGE_PASS (fable_stage_levelpoints 6): a gather's buffer half (the claims,
    /// WriteMemory per range, the verify's marks), the buffer cache's lock held; true when it
    /// recorded a command.
    bool ApplyGatherBufferHalf(std::span<const std::pair<DAddr, u64>> ranges);
    /// A gather's half for the other caches (the texture and pipeline caches), on the back.
    void ApplyGatherOtherHalf(std::span<const std::pair<DAddr, u64>> ranges);
    /// GPUMem, at a passed gather: MemoryStageLink::PassTake.
    u8 StagePassGather(std::vector<std::pair<u64, u64>>& ranges);
    /// Step 7 (GPUMem, in a served call): a download over a buffer joined ahead of the back.
    void CheckServedJoins(DAddr addr, u64 size);
    void InnerInvalidationServed(std::span<const std::pair<DAddr, std::size_t>> sequences);
    template <typename F>
    void OnStage(u8 cause, F&& body);
    bool TryOcarinaUpscale(const ComputePipeline& pipeline, bool replace);
    /// Render heights below 720p: finds the game's upscale pass by what it writes.
    void NoteLowSceneUpscale(ComputePipeline& pipeline, u32 height);
    void ReadOcarinaActor(struct OcarinaActorMesh& mesh);
    /// Records the uniform and storage buffers of the current draw or dispatch for the frame dump.
    bool FrameDumpBuffers(bool compute, std::span<const u64> shader_hashes);
    /// EDEN_NX2_PENDING_WRITE_TRACE: logs bound buffer ranges with collected, unapplied CPU writes.
    void TracePendingCpuWrites(bool compute, std::span<const u64> shader_hashes);

    Tegra::GPU& gpu;
    Tegra::MaxwellDeviceMemoryManager& device_memory;

    const Device& device;
    MemoryAllocator& memory_allocator;
    StateTracker& state_tracker;
    Scheduler& scheduler;

    StagingBufferPool staging_pool;
    /// FRITCHII_MEMSTAGE_SPLIT_ALLOC: the buffer cache's allocator and staging pool (null when
    /// off); the references below name them, or memory_allocator and staging_pool.
    std::unique_ptr<MemoryAllocator> split_buffer_allocator;
    std::unique_ptr<StagingBufferPool> split_buffer_staging;
    MemoryAllocator& buffer_memory_allocator;
    StagingBufferPool& buffer_staging_pool;
    DescriptorPool descriptor_pool;
    GuestDescriptorQueue guest_descriptor_queue;
    ComputePassDescriptorQueue compute_pass_descriptor_queue;
    DescriptorBufferRing descriptor_buffer_ring;
    BlitImageHelper blit_image;
    RenderPassCache render_pass_cache;

    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    /// The memory stage's seam (vk_mem_stage.cpp): constructed before the buffer cache runtime,
    /// which pushes its entries and asks its clock.
    MemPlayer mem_player;
    MemStage mem_stage;
    SeamClock mem_clock;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    QueryCacheRuntime query_cache_runtime;
    QueryCache query_cache;
    PipelineCache pipeline_cache;
    /// FRITCHII_ASYNC_COMPUTE_SKIP (vk_async_compute_skip.h). After pipeline_cache: its kick
    /// thread builds that cache's pipelines, so it is joined before they go.
    AsyncComputeSkip async_compute_skip;
    AccelerateDMA accelerate_dma;
    FenceManager fence_manager;
    FrameDump frame_dump;
    FlashWatch flash_watch;

    /// EDEN_NX2_RUNAHEAD (nx2_runahead.h): the caches' side, registered while this rasterizer
    /// lives. Back thread: a chunk's CPU bytes into the caches, exactly (Nx2ApplyCpuBytes).
    void Nx2ApplyRunaheadChunk(const VideoCommon::Runahead::Chunk& chunk);
    VideoCommon::Runahead::Hooks runahead_hooks{};

    boost::container::static_vector<u32, MAX_IMAGE_VIEWS> image_view_indices;
    std::array<VideoCommon::ImageViewId, MAX_IMAGE_VIEWS> image_view_ids;
    boost::container::static_vector<VkSampler, MAX_TEXTURES> sampler_handles;

    u32 draw_counter = 0;
    std::optional<u64> oot_camera_sample_frame;
    mutable std::optional<VideoCommon::OcarinaUpscaleBackend> oot_backend;
    /// AMD FSR 4 takes OoT's scene without camera jitter (Fsr4Upscaler::RunsUnjittered).
    mutable bool oot_fsr4_unjittered{};
    mutable bool oot_dlss_jitter{}; ///< DLSS gets the camera jitter (set with oot_backend)
    /// oot_upscale_output for this session (set with oot_backend): 0 = Auto (as before).
    mutable u32 oot_output_height{};
    /// The status bar's picture as published at the start (set with oot_backend), so a DLSS
    /// quality switched to or from DLAA in game shows the size that really runs.
    mutable u32 oot_picture_backend{};
    mutable u32 oot_picture_input{};
    mutable u32 oot_picture_output{};
    mutable bool oot_picture_up_to{};
    int oot_picture_native{-1}; ///< DLAA last published: -1 not yet, 0 no, 1 yes
    bool oot_resolution_checked{};
    struct OcarinaMotionTrace {
        const Tegra::MemoryManager* camera_channel{};
        const Tegra::MemoryManager* depth_channel{};
        u64 depth_frame{};
        GPUVAddr depth_address{};
        u32 depth_tic{};
        VkExtent2D extent{};
        std::array<f32, 16> current{};
        std::array<f32, 16> previous{};
    } oot_motion;
    std::unique_ptr<class OcarinaUpscaler> oot_live_dlss;
    bool oot_live_failed{};
    /// Why and how long the game's own upscale drew instead (logged when it starts and ends).
    VideoCommon::UpscalerFallbackLog oot_fallback_log;
    /// The furthest check this frame's scene draws reached without giving a camera, for the log.
    struct OcarinaCameraMiss {
        u64 frame{};
        u32 rank{};
        const char* reason{};
        std::array<f32, 4> values{};
    } oot_camera_miss;
    /// FRITCHII_DLSS_SCALED: the real sizes last logged (scene, depth, target, output) and the
    /// last colour/depth mismatch, so each new combination is logged once.
    std::array<u32, 8> oot_scaled_logged{};
    std::array<u32, 8> oot_scaled_mismatch_logged{};
    std::optional<u64> oot_dlss_frame;
    std::array<f32, 2> oot_jitter_applied{}; ///< in the viewport state now (render pixels)
    std::array<f32, 2> oot_jitter_offset{};  ///< what oot_jitter_frame's scene draws used
    std::optional<u64> oot_jitter_frame;
    u32 oot_jitter_phases{8}; ///< Halton sequence length for the last FSR frame's scale
    std::optional<u64> oot_4k_scene_frame;
    VkImage oot_4k_guest_present{};
    /// Frames that show the game's 1440p image although the upscaler made the 4K one.
    VideoCommon::UpscalerFallbackLog oot_4k_present_log;
    std::optional<u64> oot_effects_frame;
    GPUVAddr oot_effects_address{};
    VkImageView oot_effects_scene{};
    /// Fritchii's free guard: oot_effects_scene's view and its generation when taken.
    VideoCommon::ImageViewId oot_effects_view{};
    u32 oot_effects_generation{};
    u32 oot_effects_stale{}; ///< marks skipped: that view was destroyed in between
    struct OcarinaActorPose {
        u64 frame{};
        const Tegra::MemoryManager* channel{};
        GPUVAddr vertex{};
        GPUVAddr indices{};
        s32 base_vertex{};
        std::array<f32, 4> origin{};
        u64 palette{}; ///< hash of the palette's first bones (FRITCHII_OOT_ACTOR_PARTS, _TRACE)
        std::array<f32, 280 * 12> bones{};
    } oot_actor, oot_previous_actor;
    /// FRITCHII_OOT_ACTOR_PARTS, _TRACE: this frame's other draws of the actor's vertex program.
    struct OcarinaActorPart {
        const Tegra::MemoryManager* channel{};
        GPUVAddr vertex{};
        GPUVAddr indices{};
        s32 base_vertex{};
        u32 index_count{};
        u64 palette{};
        std::array<f32, 4> origin{};
    };
    std::array<OcarinaActorPart, 128> oot_actor_parts{};
    u32 oot_actor_part_count{};
    u64 oot_actor_parts_frame{};
    u32 oot_actor_trace_lines{}; ///< FRITCHII_OOT_ACTOR_TRACE: lines this frame
    u64 oot_actor_trace_frame{};
    u64 render_trace_frame = 0;
    u32 render_trace_draws = 0;
    /// The topology UpdateDynamicStates gives the draw (see there).
    VkPrimitiveTopology dynamic_host_topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    u32 render_trace_clears = 0;
    bool pending_native_work{};
    bool mesh_draw{}; ///< the draw being prepared uses a task/mesh pipeline
    /// EDEN_NX2_FASTDRAW_COUNT (nx2_fastdraw_count.h): which draws could be delta draws by
    /// the ported change's rules. Counting only; PrepareDraw calls these when it is on.
    VideoCommon::Nx2FastDrawCount::Counter fast_draw_count;
    /// Before the pipeline lookup (the original TakeReusableDraw takes the previous draw here).
    void FastDrawCountBegin();
    /// Once the caches are locked: would this draw be a delta draw, and the serial for the next.
    void FastDrawCountDecide(const GraphicsPipeline* pipeline, bool is_indexed, bool is_indirect);
    /// After the draw was recorded: whether the next draw may follow it (NoteResolvedDraw).
    void FastDrawCountResolved(const GraphicsPipeline* pipeline, bool is_indexed,
                               bool is_indirect);
    /// EXP118 (fritchii_delta_path.h): the delta path's counts, and whether the draw being
    /// decided had its vertex input moved.
    VideoCommon::FritchiiDeltaPath::Stats delta_path;
    bool delta_vertex_moved{};
    /// EXP118, after FastDrawCountBegin, before the pipeline lookup: the pipeline the draw would
    /// keep as a delta draw, or null (the reason counted). Changes nothing.
    GraphicsPipeline* DeltaPathDecide(bool is_indexed, bool is_indirect);
    /// EXP118: counts a full lookup compared with the kept pipeline; logs a difference.
    bool DeltaPathCompare(const GraphicsPipeline* kept, const GraphicsPipeline* looked, int mode);
    /// Once the caches are locked, with the count off: the serials the next draw compares with
    /// (FastDrawCountDecide takes them while the count runs).
    void DeltaPathSnapshot();
    /// List draws (nx2_list_draw.h): DrawIndirect for IndirectParams::list_words.
    void DrawListWords();
    VideoCommon::OotNetwork::BypassState oot_network_bypass;
    u32 mesh_emulation_groups{};             ///< mesh workgroups of the emulated draw
    u32 mesh_emulation_vertices_per_group{}; ///< 0 unless the draw is an emulated mesh draw
    /// Mesh emulation diagnostics with a task program: one draw's distribution arguments.
    struct {
        std::span<const u8> args;
        u64 tick{};
        u32 tasks{};
        u32 capacity{};
        bool pending{};
    } mesh_task_probe;
    /// With a task program: the vkCmdDrawIndirect arguments the distribution pass writes.
    struct {
        VkBuffer buffer{};
        VkDeviceSize offset{};
    } mesh_emulation_indirect;
    /// Mesh emulation diagnostics: one draw's output, read back once the GPU has written it.
    struct MeshEmulationProbe {
        std::span<const u8> data;
        u64 tick{};
        u64 frame{};
        u32 groups{};
        u32 block_words{};
        u32 vertex_lines{};
        u32 max_vertices{};
        u32 max_primitives{};
        std::array<u32, 4> position_lines{};
        bool pending{};
    } mesh_probe;
    void ReadMeshEmulationProbe();
    /// Mesh emulation with a task program: the output arena every task draw of a submission
    /// allocates its mesh workgroups from, and its host-visible counter of allocated words.
    struct MeshArena {
        vk::Buffer buffer;
        vk::Buffer counter;
        u64 size{};
        u64 tick{};
        bool counted{}; ///< the counter of its last submission was read
    };
    std::vector<MeshArena> mesh_arenas;
    size_t mesh_arena_current{~size_t{}};
    u64 mesh_arena_size{64ULL << 20};
    /// The current submission's arena, after its counter is reset.
    const MeshArena& AcquireMeshArena();
    /// Fragment output components the dynamic color write mask was last recorded with
    u32 last_color_output_mask{};
    /// Mesh emulation: the draw's staging buffers, held until its commands are recorded.
    /// Configure can submit (an ASTC decode, a full descriptor buffer ring), so the tick when
    /// they were requested may be older than the submission that writes them.
    boost::container::static_vector<StagingBufferRef, 4> mesh_draw_buffers;
    bool mesh_arena_used{};
    void ReleaseMeshEmulationBuffers();

    struct IndirectTrace {
        vk::Buffer readback;
        u64 tick{};
        u64 frame{};
        GPUVAddr address{};
        GPUVAddr count_address{};
        u32 max_draws{};
        u32 sampled_commands{};
        bool pending{};
        bool include_count{};
        bool byte_count{};
    };
    std::array<IndirectTrace, 8> indirect_traces;
    Common::DkbFrameTrace::RateLimit trace_rate;
    Common::DkbFrameTrace::RateLimit trace_cpu_rate;
    std::optional<Common::DkbCpuTrace::Snapshot> trace_cpu_snapshot;
    u64 trace_cpu_frame{};
    bool trace_sample_frame{true};
    u64 trace_frame{};
    u64 trace_window_frames{};
    u64 trace_draws{};
    u64 trace_skipped{};
    u64 trace_indirect{};
    u64 trace_zero_direct{};
    /// Vertex and fragment hashes of the draw being recorded, for GPU timing.
    std::pair<u64, u64> trace_draw_hashes{};
    u32 trace_captures{};
    u32 trace_capture_dropped{};
    u32 trace_failure_details{};
    /// The vertex repack mask of the draw being prepared (the stage's, from its packet, at
    /// FRITCHII_MEMSTAGE_THREAD = 2).
    u32 back_repack_mask{};
    /// FRITCHII_MEMSTAGE_LIST_DRAWS: PrepareDraw set the cache's indirect parameters for a served
    /// list draw without its packet (the fallback fill); DrawListWords clears them after.
    bool list_fallback_indirect{};
    /// GPUMem's copy of the facts of the draw it fills (the cache keeps a pointer to the sizes).
    MemStage::StagedFacts stage_facts{};
    /// FRITCHII_MEMSTAGE_SETUP_KEEP (GPUMem): what stage_facts was looked up for, the program
    /// registers and the facts table's generation then. Valid only after a lookup that found
    /// the facts.
    struct SetupKeep {
        std::array<u64, VideoCommon::MemStageShadow::ProgramWordCount> words{};
        u64 space{};
        std::array<u64, 6> code{}; ///< the words resolved (the facts table's identity)
        u64 generation{};
        bool valid{};
        // Per setup at SETUP_KEEP != 0, logged every SetupKeepLogEvery setups.
        u64 setups{};
        u64 kept{};
        u64 checked{};
        u64 disagreed{};
        /// Verify: the table was written between the keep's check and the full lookup (its
        /// generation moved): not compared; of those, the ones whose facts differed.
        u64 raced{};
        u64 raced_differed{};
    };
    SetupKeep setup_keep{};
    /// GPUMem: the raw handles of the draw it fills (SEAM 3).
    std::vector<u32> stage_handle_values;
    /// FRITCHII_SIM_LINK_HOG: last member, so its thread ends before anything else goes.
    std::unique_ptr<SimLinkHog> sim_link_hog;
};

} // namespace Vulkan
