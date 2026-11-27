// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <array>
#include <atomic>
#include <bit>
#include <bitset>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/container/unordered_map.h"
#include <boost/unordered/unordered_node_map.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/div_ceil.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include "common/range_sets.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "core/gpu_dirty_memory_manager.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/buffer_cache/buffer_base.h"
#include "video_core/buffer_cache/cache_inputs.h"
#include "video_core/buffer_cache/download_read_stats.h"
#include "video_core/memory_stage_link.h"
#include "video_core/buffer_cache/gpu_written_mirror.h"
#include "video_core/buffer_cache/read_run_stats.h"
#include "video_core/buffer_cache/read_trap_tracker.h"
#include "video_core/buffer_cache/store_footprint_live.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/fence_release_stats.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_runahead.h"
#include "video_core/nx2.h"
#include "video_core/nx2_async_readback.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_download_trace.h"
#include "video_core/rasterizer_download_area.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

using BufferId = Common::SlotId;

using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

#ifdef __APPLE__
constexpr u32 NUM_VERTEX_BUFFERS = 16;
#else
constexpr u32 NUM_VERTEX_BUFFERS = 32;
#endif
constexpr u32 NUM_TRANSFORM_FEEDBACK_BUFFERS = 4;
constexpr u32 NUM_GRAPHICS_UNIFORM_BUFFERS = 18;
constexpr u32 NUM_COMPUTE_UNIFORM_BUFFERS = 8;
// Native shaders can reference more than 16 storage buffers. Dropping a binding
// shifts all following descriptor payloads into the wrong descriptor types.
constexpr u32 NUM_STORAGE_BUFFERS = Shader::Info::MAX_SSBOS;
static_assert(NUM_STORAGE_BUFFERS <= 64, "storage binding masks must fit u64");
constexpr u32 NUM_TEXTURE_BUFFERS = 32;
constexpr u32 NUM_STAGES = 5;

static_assert(NUM_GRAPHICS_UNIFORM_BUFFERS <= 32, "fast bitmask must fit u32");

using UniformBufferSizes = std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>;
using ComputeUniformBufferSizes = std::array<u32, NUM_COMPUTE_UNIFORM_BUFFERS>;

enum class ObtainBufferSynchronize : u32 {
    NoSynchronize = 0,
    FullSynchronize = 1,
    SynchronizeNoDirty = 2,
};

enum class ObtainBufferOperation : u32 {
    DoNothing = 0,
    MarkAsWritten = 1,
    DiscardWrite = 2,
    MarkQuery = 3,
};

static constexpr BufferId NULL_BUFFER_ID{0};
static constexpr u32 DEFAULT_SKIP_CACHE_SIZE = static_cast<u32>(4_KiB);

struct Binding {
    DAddr device_addr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    BufferId buffer_id;
    u32 segment_first{};
    u32 segment_count{};
};

struct MultiRangeSegment {
    BufferId buffer_id;
    DAddr device_addr{};
    u32 size{};
};

struct TextureBufferBinding : Binding {
    PixelFormat format;
};

static constexpr Binding NULL_BINDING{
    .device_addr = 0,
    .size = 0,
    .buffer_id = NULL_BUFFER_ID,
};

template <typename Buffer>
struct HostBindings {
    boost::container::static_vector<Buffer*, NUM_VERTEX_BUFFERS> buffers;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> offsets;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> sizes;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> strides;
    u32 min_index{NUM_VERTEX_BUFFERS};
    u32 max_index{0};
    /// Full vertex binds: a run of bindings that are not dirty (the player keeps the bound size).
    bool clean{false};
};

class BufferCacheChannelInfo : public ChannelInfo {
public:
    BufferCacheChannelInfo() = delete;
    BufferCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept : ChannelInfo(state) {}
    BufferCacheChannelInfo(const BufferCacheChannelInfo& state) = delete;
    BufferCacheChannelInfo& operator=(const BufferCacheChannelInfo&) = delete;

    Binding index_buffer;
    std::array<Binding, NUM_VERTEX_BUFFERS> vertex_buffers;
    std::array<std::array<Binding, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES> uniform_buffers;
    std::array<std::array<Binding, NUM_STORAGE_BUFFERS>, NUM_STAGES> storage_buffers;
    std::array<std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS>, NUM_STAGES> texture_buffers;
    std::array<Binding, NUM_TRANSFORM_FEEDBACK_BUFFERS> transform_feedback_buffers;
    Binding count_buffer_binding;
    Binding indirect_buffer_binding;

    std::array<Binding, NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniform_buffers;
    std::array<Binding, NUM_STORAGE_BUFFERS> compute_storage_buffers;
    std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS> compute_texture_buffers;

    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    u32 enabled_compute_uniform_buffer_mask = 0;

    const UniformBufferSizes* uniform_buffer_sizes{};
    const ComputeUniformBufferSizes* compute_uniform_buffer_sizes{};

    std::array<u64, NUM_STAGES> enabled_storage_buffers{};
    std::array<u64, NUM_STAGES> written_storage_buffers{};
    u64 enabled_compute_storage_buffers = 0;
    u64 written_compute_storage_buffers = 0;
    // Bound by a native (Switch 2) compute program, whose pointers carry no size.
    u64 native_compute_storage_buffers = 0;
    // Bytes from each compute storage binding's start that a dispatch marks GPU-written.
    std::array<u32, NUM_STORAGE_BUFFERS> compute_storage_write_marks{};

    u32 total_graphics_storage_buffers = 0;
    u32 total_compute_storage_buffers = 0;

    std::array<u32, NUM_STAGES> enabled_texture_buffers{};
    std::array<u32, NUM_STAGES> written_texture_buffers{};
    std::array<u32, NUM_STAGES> image_texture_buffers{};
    u32 enabled_compute_texture_buffers = 0;
    u32 written_compute_texture_buffers = 0;
    u32 image_compute_texture_buffers = 0;

    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_hits{};
    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_shots{};

    u32 uniform_buffer_skip_cache_size = DEFAULT_SKIP_CACHE_SIZE;

    bool has_deleted_buffers = false;

    std::array<u32, NUM_STAGES> dirty_uniform_buffers{};
    std::array<u32, NUM_STAGES> fast_bound_uniform_buffers{};
    std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_buffer_binding_sizes{};
};

template <class P>
class BufferCache : public VideoCommon::ChannelSetupCaches<BufferCacheChannelInfo> {
    // Page size for caching purposes.
    // This is unrelated to the CPU page size and it can be changed as it seems optimal.
#ifdef YUZU_LEGACY
    static constexpr u32 CACHING_PAGEBITS = 12;
#else
    static constexpr u32 CACHING_PAGEBITS = 16;
#endif
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;

    static constexpr bool IS_OPENGL = P::IS_OPENGL;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS =
        P::HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT =
        P::HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = P::NEEDS_BIND_UNIFORM_INDEX;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = P::NEEDS_BIND_STORAGE_INDEX;
    static constexpr bool USE_MEMORY_MAPS = P::USE_MEMORY_MAPS;
    static constexpr bool SEPARATE_IMAGE_BUFFERS_BINDINGS = P::SEPARATE_IMAGE_BUFFER_BINDINGS;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = P::USE_MEMORY_MAPS_FOR_UPLOADS;

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 512_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB;

    // Debug Flags.

    static constexpr bool DISABLE_DOWNLOADS = true;

    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    using Runtime = typename P::Runtime;
    using Buffer = typename P::Buffer;
    using Async_Buffer = typename P::Async_Buffer;
    using MemoryTracker = typename P::MemoryTracker;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        DAddr begin;
        DAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_);

    ~BufferCache();

    void TickFrame();

    bool BindMultiRangeStorage(const Binding& binding, bool is_written,
                               std::span<const MultiRangeSegment> pool,
                               u64 write_mark_limit = ~u64{0});

    void ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                  std::vector<MultiRangeSegment>& pool);

    void RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool);

    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// `keep_page_gpu_data`: download GPU-written bytes sharing the written pages first (false
    /// when the memory is going away). `engine_write`: the GPU's own engines wrote it (inline
    /// constant buffer data, MemoryManager::FlushCaching), not the guest CPU (EXP108).
    void WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data = true,
                     bool engine_write = false);

    /// EXP91 (fable52_answer_uniformkeeps stage 0, counting only): RasterizerVulkan::
    /// InnerInvalidation's ranges (FlushCaching's cached writes, EXP88's constant buffer data
    /// among them), noted first, before any early return. Drawing thread. Changes nothing bound.
    void NoteCachedWrites(std::span<const std::pair<DAddr, std::size_t>> sequences);

    /// EXP88 & 1 (ported, part c; fable52_answer_next70 3.1): why WriteMemory of
    /// [device_addr, device_addr + size) may change something, or None when it would only make
    /// its first three marks (NoteWriteUnlocked): no page ever left the CPU-modified state (so
    /// none is tracked, cached-written or GPU-modified), no GPU write ever marked one (Switch 2
    /// titles; so no gpu_modified_ranges there) and no enabled vertex binding overlaps the range
    /// (MarkVertexBuffersStale would act). Drawing thread only (v_buffer is its state), without
    /// the lock.
    enum class WriteReach : u8 { None, EverSynchronized, MayBeGpuWritten, VertexBinding };
    [[nodiscard]] WriteReach WriteReachesBuffers(DAddr device_addr, u64 size) const;

    /// EXP88: what WriteMemory does first, on a range WriteReachesBuffers answers None for: the
    /// command list note and the two page-flag marks the native bank check and the write watch
    /// read (g_buffer_write_pages, g_watch_write_pages). Atomics or their own mutex; no lock.
    void NoteWriteUnlocked(DAddr device_addr, u64 size);

    /// EXP88 & 4, under the lock: WriteMemory of the range would change nothing but its first
    /// three marks (no GPU-modified range on its pages, every page untouched, no vertex binding).
    [[nodiscard]] bool WriteChangesNothing(DAddr device_addr, u64 size);

    /// EXP88: whether ClaimCpuWrites may find anything in `ranges` (a page ever marked
    /// GPU-modified). Without the lock.
    [[nodiscard]] bool ClaimMayFindGpuWrites(std::span<const std::pair<DAddr, u64>> ranges) const;

    void CachedWriteMemory(DAddr device_addr, u64 size);

    /// Cancels pending downloads over the range (fences' and, with EDEN_NX2_EXP49, reads'), so
    /// none writes them over newer bytes. Those bytes reach guest memory later if at all (a GPU
    /// copy, a clear, an inline or a discarded write), so read downloads already taken land
    /// first: their reads get what a synchronous flush would have written.
    void ClearDownload(DAddr base_addr, u64 size);

    /// ClearDownload where guest memory gets the newer bytes before a read can find the range: a
    /// guest CPU write, a synchronous download landing under `mutex`, or a GPU-side write whose
    /// read downloads MemoryManager landed first. Read downloads leave the range rather than land,
    /// so this never waits for the host GPU.
    void ClearReplacedDownload(DAddr device_addr, u64 size);

    /// CPU writes collected but not yet applied (SetPendingCpuWrites).
    [[nodiscard]] Common::RangeSet<DAddr> PendingCpuWriteRanges();

    /// FRITCHII_FENCE_COPY_UNLOCKED: PendingCpuWriteRanges' writes that overlap [begin, end)
    /// into `out` (cleared first), for questions inside that range only. True if any.
    bool PendingCpuWritesIn(DAddr begin, DAddr end, Common::RangeSet<DAddr>& out);

    /// FRITCHII_FENCE_COPY_UNLOCKED: WriteDownloadedMemory without the lock (inside a landing
    /// published by DownloadLanding::PublishUnlocked), its page marks after the bytes.
    void WriteDownloadedMemoryUnlocked(const Common::RangeSet<DAddr>* cpu_written,
                                       DAddr device_addr, const u8* data, u64 size);

    /// Writes downloaded bytes to guest memory, except over `cpu_written`. Callers take
    /// `cpu_written` inside a VideoCore::DownloadLanding over the bytes and write in it: the CPU
    /// writes collected after it store their bytes once the landing has ended.
    void WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written, DAddr device_addr,
                               const u8* data, u64 size);

    bool OnCPUWrite(DAddr device_addr, u64 size);

    void DownloadMemory(DAddr device_addr, u64 size);

    /// DownloadMemory for each range, waiting for the GPU once.
    void DownloadMemoryRanges(std::span<const std::pair<DAddr, u64>> ranges);

    /// EDEN_NX2_EXP49: DownloadMemory for a guest CPU read. Submits the copies without waiting
    /// for the host GPU, for FinishReadDownloads to write to guest memory. Their ranges stay
    /// GPU-modified until then, and another read of them waits for these copies (GetFlushArea:
    /// in_flight).
    void DownloadMemoryForRead(DAddr device_addr, u64 size);

    /// FRITCHII_READ_RUNS and EXP112 (`mutex` held): the pages around [begin, end) (page
    /// aligned) that hold GPU-modified bytes in a row, inside the buffer holding `begin`, up to
    /// max_bytes in all (the read's own pages count; the pages after them first, then those
    /// before), clipped to that buffer. With stop_at_preflushable a Preflushable page ends the
    /// run (fence downloads take those). Empty (first == second) when no buffer holds `begin`.
    std::pair<DAddr, DAddr> GpuModifiedRunAround(DAddr begin, DAddr end, u64 max_bytes,
                                                 bool stop_at_preflushable);

    /// Writes the DownloadMemoryForRead copies over [device_addr, device_addr + size), and those
    /// submitted before them, to guest memory, waiting for the host GPU if it must. Any thread:
    /// the reading one before it reads, the GPU's side before it writes guest memory there. Takes
    /// `mutex` itself (a caller holding it keeps it held while the host GPU finishes). Returns how
    /// long it waited for the host GPU (at least 1 ns if it did), or nothing when no download was
    /// over the range.
    std::optional<u64> FinishReadDownloads(DAddr device_addr, u64 size);

    /// A GPU write into the buffers over [device_addr, device_addr + size), after the read
    /// downloads taken so far: they still write their older bytes to guest memory, as synchronous
    /// ones would have, but a later read takes the range again. MarkWrittenBuffer does this for
    /// the writes it records; the query cache's copies of its results call it themselves
    /// (`mutex` held).
    void OutdateReadDownloads(DAddr device_addr, u64 size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr device_addr, u64 size);

    /// Without downloads at every fence, a guest CPU read is the only way GPU-written buffer
    /// data reaches guest memory, so reads must ask the buffer cache (GetFlushArea).
    bool FlushesOnCpuRead() const {
        return !AsyncBufferDownloads();
    }

    bool InlineMemory(DAddr dest_address, size_t copy_size, std::span<const u8> inlined_buffer);

    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size);

    void DisableGraphicsUniformBuffer(size_t stage, u32 index);

    void UpdateGraphicsBuffers(bool is_indexed);

    void UpdateComputeBuffers();

    void BindHostGeometryBuffers(bool is_indexed);

    void BindHostStageBuffers(size_t stage);

    void BindHostComputeBuffers();

    /// FRITCHII_EARLY_COMPUTE (dry): sets EarlyCompute::t_verdict for the dispatch being bound.
    void JudgeEarlyCompute();
    /// FRITCHII_FEEDER_LANE (dry): sets EarlyCompute::t_lane (D) for the feeder being bound.
    template <typename IsFree>
    void JudgeFeederLane(IsFree&& is_free);
    /// FRITCHII_FEEDER_LANE (dry): after a judged feeder's binds (its upload, the counts).
    void NoteFeederLaneBound();

    void SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                const UniformBufferSizes* sizes);

    void SetComputeUniformBufferState(u32 mask, const ComputeUniformBufferSizes* sizes);

    void UnbindGraphicsStorageBuffers(size_t stage);

    bool BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                   bool is_written, u32 indirect_offset = ~0U);

    void UnbindGraphicsTextureBuffers(size_t stage);

    void BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                   PixelFormat format, bool is_written, bool is_image);

    void UnbindComputeStorageBuffers();

    /// The compute program whose storage buffers are bound next (its write extents are checked
    /// per program, see ExtentTrust).
    void SetComputeShaderHash(u64 hash) noexcept {
        compute_shader_hash = hash;
    }

    /// Records the "after" copies of the write extent checks of the dispatch just recorded.
    /// Returns how many it recorded.
    u32 RecordComputeExtentProbes();
    /// One follower (FRITCHII_ONEF_OP_DISPATCH, E, the lock held): whether
    /// RecordComputeExtentProbes would record anything (a probe without its "after" copy).
    [[nodiscard]] bool HasUnrecordedExtentProbes() const noexcept {
        for (const auto& probe : extent_probes) {
            if (!probe.after) {
                return true;
            }
        }
        return false;
    }
    /// One follower (E, the lock held): writes to command list pages were noted while collecting
    /// (Nx2CollectListCopies): Nx2CopyListPages would record copies.
    [[nodiscard]] bool Nx2HasListCopyRequests() const noexcept {
        return !nx2_list_copy_requests.empty();
    }

    /// write_extent: how far the program's stores reach past the pointer (native programs).
    /// extra: FRITCHII_EXTENT_CENSUS, FRITCHII_EXTENT_CBUF and FRITCHII_STORE_FOOTPRINT
    /// (Shader::Info::storage_write_extras, null when the program has none).
    /// read_extent: FRITCHII_FEEDER_LANE (Shader::Info::storage_read_extents, null when none).
    void BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                  bool is_written, u32 indirect_offset = ~0U,
                                  const Shader::StorageWriteExtent& write_extent = {},
                                  const Shader::StorageWriteExtra* extra = nullptr,
                                  const Shader::StorageWriteExtent* read_extent = nullptr);

    void UnbindComputeTextureBuffers();

    void BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size, PixelFormat format,
                                  bool is_written, bool is_image);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                       ObtainBufferSynchronize sync_info,
                                                       ObtainBufferOperation post_op);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainCPUBuffer(DAddr gpu_addr, u32 size,
                                                          ObtainBufferSynchronize sync_info,
                                                          ObtainBufferOperation post_op);

    /// GPU work recorded now reads or writes these bytes, of a buffer obtained before (work set
    /// up once that keeps reading them, as conditional rendering does). Creates no buffer.
    void MarkUsed(DAddr device_addr, u64 size);
    void FlushCachedWrites();

    /// Where the CPU writes into GPU-written memory that were collected but not applied yet
    /// (WriteMemory) come from: downloads must not write the GPU's older copy over them.
    using PendingCpuWrites = std::function<void(const std::function<void(DAddr, size_t)>&)>;
    void SetPendingCpuWrites(PendingCpuWrites source) {
        pending_cpu_writes = std::move(source);
    }

    /// Takes CPU writes about to be applied one by one out of the GPU-written ranges first, so
    /// applying one doesn't download the GPU's copy over another (WriteMemory keeps the GPU data
    /// on the pages it writes).
    template <typename Ranges>
    void ClaimCpuWrites(const Ranges& ranges);

    /// EDEN_NX2_RUNAHEAD (nx2_runahead.h), back thread, mutex held: the guest CPU's bytes of
    /// [device_addr, device_addr + size) as a run-ahead chunk captured them (`bytes`). Registered
    /// memory gets them in its host buffer now, instead of a CPU-modified mark a later upload
    /// would read guest memory for (by then it may hold the next frame); the uniform bank
    /// snapshot of the page takes them too, so it keeps matching the host buffer. Memory no
    /// buffer holds is marked as before (a buffer made there later reads guest memory: a risk).
    void Nx2ApplyCpuBytes(DAddr device_addr, u64 size, const u8* bytes);

    /// Return true when there are uncommitted buffers to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    /// Committed fence downloads not landed in guest memory yet (under the lock).
    /// FRITCHII_ONEF_SEM_E_LANDED: a sync point's host increment then waits for them.
    [[nodiscard]] bool HasPendingDownloads() const noexcept {
        return !pending_downloads.empty();
    }

    /// FRITCHII_FENCE_COMMIT_FAST: false only when HasUncommittedFlushes was false at the last
    /// refresh (under the lock) and no fence download was queued since (QueueFenceDownload
    /// raises it, under the lock). Any thread, no lock.
    [[nodiscard]] bool MayHaveUncommittedFlushes() const noexcept {
        return uncommitted_flushes_noted.load(std::memory_order_relaxed);
    }

    /// FRITCHII_FENCE_COMMIT_FAST: the note from the range sets. Under the lock.
    void RefreshUncommittedFlushesNote() noexcept {
        uncommitted_flushes_noted.store(HasUncommittedFlushes(), std::memory_order_relaxed);
    }

    /// The memory stage's P3 (fable_stage_p3 4.2 and 6): whether a query result sync's
    /// ObtainCPUBuffer(FullSynchronize) on [begin, begin + bytes) would be a pure lookup (one
    /// existing buffer holds it, no page of it CPU-modified or untracked: the sync would then
    /// only mark the usage). Changes nothing (it may create the tracker's missing managers, as
    /// the sync would). Under the lock.
    enum class SyncProbe : u8 { Pure, NoBuffer, NotInBounds, CpuModified };
    [[nodiscard]] SyncProbe ProbeSyncTarget(DAddr begin, u64 bytes);

    void AccumulateFlushes();

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();
    void CommitAsyncFlushesHigh();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();
    void PopAsyncBuffers();

    /// FRITCHII_FENCE_COPY_UNLOCKED (helper CQ; common/fritchii_lock_census.h): `pops` of
    /// PopAsyncFlushes for the fencing thread, called without `mutex`. Each pop reads what it
    /// writes under the lock, lets it go to copy a landing's bytes into guest memory, and takes
    /// it back to clear the GPU-modified state and mark the download landed (why the result is
    /// the locked pop's: work/helper_cq_answer.md). `mode` 2 compares guest memory with the
    /// download after each copy. Fencing thread only.
    void PopAsyncBuffersUnlocked(size_t pops, int mode, Common::LockCensus::FenceCopyBatch& batch);

    /// EDEN_NX2_FENCE_RELEASE_STATS (`mutex` held): what the next PopAsyncFlushes copies to guest
    /// memory (EXP49's read downloads it also lands are not included).
    void Nx2DescribeNextPop(FenceReleaseStats::PopLanding& landing) {
        if (async_buffers.empty() || !async_buffers.front().has_value() ||
            pending_downloads.empty()) {
            return;
        }
        for (const BufferCopy& copy : pending_downloads.front()) {
            FenceReleaseStats::LandedItem item{};
            item.address = static_cast<u64>(copy.src_offset);
            item.end = item.address + copy.size;
            item.bytes = copy.size;
            landing.Add(item);
        }
    }

    bool DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount);

    bool DMAClear(GPUVAddr src_address, u64 amount, u32 value);

    /// Switch 2 GPU-written command lists (nx2.h, g_nx2_command_list_pages). While collecting (a
    /// compute dispatch or a DMA copy being recorded), writes to command list pages are noted;
    /// Nx2CopyListPages then records copies of those pages into host memory after the work.
    void Nx2CollectListCopies(bool collect) {
        nx2_collect_list_copies = collect;
        if (!collect) {
            nx2_list_copy_requests.clear();
        }
    }
    /// True when it recorded copies, which the GPU completes at `tick` (the caller submits).
    /// `writer`: the dispatch's shader hash, 0 for a DMA copy (EDEN_NX2_LIST_WAIT_TRACE).
    bool Nx2CopyListPages(u64 tick, u64 writer = 0);
    enum class Nx2ListRead { Unknown, Copied, Synced, NeedsSync };
    /// Front thread: Synced when guest memory holds the page `device_addr` lies in (flushed and
    /// written by no GPU work since), else Copied with `out` filled from the copy of its last
    /// write once the GPU is done with it. Unknown otherwise: the caller drains and flushes.
    /// Pages draws write are only Synced or Unknown, and NeedsSync until `past_sync`: the
    /// split's back ran the work before the last host-level wait (GpuSplit::WaitForSync).
    /// EXP40 & 1: Synced too when no GPU work wrote the bytes
    /// [device_addr, device_addr + out.size()) since the page was flushed, or the host wrote them
    /// since; Copied reads the words GPU work wrote from the copy and, once the page was flushed,
    /// the others from guest memory.
    Nx2ListRead Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync);
    /// Front thread, the split drained: the command list page was flushed to guest memory.
    void Nx2NoteListSynced(DAddr page);
    /// The split's back, the front waiting on it: downloads `pages` and every page draws wrote
    /// since it was last flushed with one GPU wait, and notes the latter synced. Each page's next
    /// lists then read guest memory instead of draining and flushing one page at a time.
    void Nx2FlushListPages(std::span<const DAddr> pages);
    /// CPU, inline and copy engine writes of bytes guest memory holds: they reach guest memory,
    /// so a synced page stays synced and the words they cover whole are guest memory's again, but
    /// the page's copy no longer holds the page.
    void Nx2NoteListHostWrite(DAddr device_addr, u64 size);

    /// fable_memstage_compute_brief 3: GPU-written marks (MarkWrittenBuffer) and the bytes they
    /// cover, under the cache's lock, with GPU profile 2 only. Counting only: DispatchCompute,
    /// which holds the lock from before its fill to its end, takes the difference over a
    /// dispatch (its storage binds).
    u64 nx2_written_marks{};
    u64 nx2_written_mark_bytes{};

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    // --- FRITCHII_ASYNC_COMPUTE_SKIP (renderer_vulkan/vk_async_compute_skip.h), the lock held ---

    /// After the compute storage pointer binds, before UpdateComputeBuffers: calls
    /// `func(device_addr, size)` for every range BindHostComputeStorageBuffers would mark
    /// GPU-written. False when a written binding's writes are not bounded by its mark: not a
    /// native pointer, a mark as large as the binding (no proven write extent, the 16 MiB mark),
    /// a mark the EDEN_NX2_WRITTEN_SSBO_MARK_KIB cap shortened, or several device ranges.
    template <typename Func>
    [[nodiscard]] bool ForEachComputeWriteMark(Func&& func) const;
    struct SkipRangeFacts {
        bool all_gpu_written{}; ///< every byte held in gpu_modified_ranges (as GPUMem sees it)
        bool cpu_modified{};    ///< a byte the guest CPU wrote that is not uploaded yet
        bool read_back{};       ///< a page Preflushable, read by the CPU after a GPU write, or
                                ///< on its way back in a read download
    };
    [[nodiscard]] SkipRangeFacts ComputeSkipRangeFacts(DAddr addr, u64 size);

    // --- FRITCHII_MEMSTAGE_MARKS_AHEAD (memory stage compute, part 1): GPU-written marks made
    // ahead of the back. The real sets (gpu_modified_ranges, the fence and read downloads) stay
    // "as of the back"; a writing fill on GPUMem puts its marks on a list keyed by ring B's
    // packet order, and they enter the real sets when the back has played that packet
    // (buffer_cache.h, "Marks ahead of the back"). ---

    /// Where the ahead list must be empty, or hold nothing due (CheckAheadAt).
    enum class AheadPoint : u8 { RingStart, RingEnd, RunStart };
    /// The level rules' counts (RasterizerVulkan::StageFillDraw).
    enum class AheadLevel : u8 { Claim, ClaimRefused, Bound };

    /// The back, at a frame end (a level point: GPUMem is parked): the switch as it runs (`on`:
    /// the switch and the window open) and as set. Logs "Memory stage marks ahead" every 10 s.
    void SetMarksAhead(bool on, bool wanted);
    [[nodiscard]] bool MarksAhead() const noexcept {
        return marks_ahead.load(std::memory_order_relaxed);
    }
    /// Any thread, no lock: the back has played a packet whose marks are still on the list.
    [[nodiscard]] bool AheadDue() const noexcept {
        return ahead_oldest.load(std::memory_order_acquire) <=
               ahead_played.load(std::memory_order_acquire);
    }
    /// The lock held: the due marks go into the real sets (one compare when none is due). Every
    /// reader of the real sets calls it first.
    void PromoteAhead(bool by_back = false) {
        if (AheadDue()) [[unlikely]] {
            PromoteAheadSlow(by_back);
        }
    }
    /// The lock held: whether a mark on the list covers a byte of [addr, addr + size), or of its
    /// pages.
    [[nodiscard]] bool AheadIntersects(DAddr addr, u64 size) const;
    [[nodiscard]] bool AheadIntersectsPages(DAddr addr, u64 size) const;
    /// The lock held: IsRegionGpuModified as GPUMem's own work sees it (the real set, and on
    /// GPUMem outside a served call the list too).
    [[nodiscard]] bool GpuWrittenHere(DAddr addr, u64 size) {
        return IsRegionGpuModified(addr, size) || AheadHere(addr, size);
    }
    /// GPUMem's draw fill, the lock held: the push index ring B's packet of this fill gets.
    void BeginAheadFill(u64 index);
    /// GPUMem's PushRingDraw, the lock held: packet `index` goes into ring B. True when its fill
    /// marked writes (the header's flag: the back promotes after its play).
    [[nodiscard]] bool AheadFillPushed(u64 index);
    /// GPUMem, however its fill ended (the lock held): the marks of a fill whose packet never
    /// went into ring B are taken back (the back fills that draw itself).
    void EndAheadFill();
    /// The back, at the end of ring packet `index`'s play. `marked`: its fill marked writes; the
    /// back then promotes before it goes on (the lock, or its holder promotes for it).
    void NoteRingPlayed(u64 index, bool marked);
    /// The back dropped packet `index` unplayed (must be 0): its marks are taken back.
    void NoteRingDiscarded(u64 index, bool marked);
    /// The back, at a ring's start or end (the list empty) or a run's first entry (nothing due).
    void CheckAheadAt(AheadPoint where);
    void NoteAheadLevel(AheadLevel kind) noexcept;
    /// The lock held: the list is at its bound (4,096 marks).
    [[nodiscard]] bool AheadFull() const noexcept {
        return ahead_writes.size() >= AheadBound;
    }

    /// Visits the uniform and storage buffers bound for the last draw or dispatch with the cached
    /// host buffer holding each range, if one does, for EDEN_NX2_FRAME_DUMP_SAMPLED. Nothing is
    /// created or synchronized. Stage 5 is compute.
    template <typename Func>
    void ForEachBoundBuffer(bool compute, Func&& func) {
        MemoryStageLink::CheckBackCache("ForEachBoundBuffer");
        const auto visit = [&](u32 stage, bool storage, u32 index, bool written,
                               const Binding& binding, u32 size) {
            if (binding.device_addr == 0 || size == 0) {
                return;
            }
            const Buffer* host = nullptr;
            u32 offset = 0;
            if (const BufferId id = page_table[binding.device_addr >> CACHING_PAGEBITS]) {
                const Buffer& buffer = slot_buffers[id];
                if (buffer.IsInBounds(binding.device_addr, size)) {
                    host = &buffer;
                    offset = buffer.Offset(binding.device_addr);
                }
            }
            func(stage, storage, index, written, binding, size, host, offset);
        };
        if (compute) {
            const auto* const sizes = channel_state->compute_uniform_buffer_sizes;
            ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
                const Binding& binding = channel_state->compute_uniform_buffers[index];
                visit(5, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
                const Binding& binding = channel_state->compute_storage_buffers[index];
                visit(5, true, index, ((channel_state->written_compute_storage_buffers >> index) & 1) != 0,
                      binding, binding.size);
            });
            return;
        }
        const auto* const sizes = channel_state->uniform_buffer_sizes;
        for (u32 stage = 0; stage < NUM_STAGES; ++stage) {
            ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
                const Binding& binding = channel_state->uniform_buffers[stage][index];
                visit(stage, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[stage][index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                const Binding& binding = channel_state->storage_buffers[stage][index];
                visit(stage, true, index,
                      ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0, binding,
                      binding.size);
            });
        }
    }

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(DAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(DAddr addr, size_t size);

    void SetDrawIndirect(
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect_) {
        current_draw_indirect = current_draw_indirect_;
    }

    /// Everything the cache reads from the engines (cache_inputs.h), for the bound channel.
    [[nodiscard]] CacheInputs Inputs() const noexcept {
        if (inputs_override != nullptr) [[unlikely]] {
            return *inputs_override;
        }
        return CacheInputs{maxwell3d, kepler_compute, gpu_memory, current_draw_indirect};
    }

    /// FRITCHII_MEMSTAGE_SHADOW=3: the inputs a draw's fill reads instead (null: the engines).
    void SetInputsOverride(const CacheInputs* inputs) noexcept {
        inputs_override = inputs;
    }
    [[nodiscard]] bool HasInputsOverride() const noexcept {
        return inputs_override != nullptr;
    }

    /// FRITCHII_MEMSTAGE_THREAD=2: the memory stage's copy as the bound channel's inputs.
    /// `draw_indirect_`: the copy's indirect parameters for a list draw it serves, else null.
    [[nodiscard]] CacheInputs StageInputs(
        const CacheInputs::Regs& regs_, const Tegra::Engines::Maxwell3D::State& state_,
        const Tegra::Engines::Maxwell3D::RootTables& tables_,
        const CacheInputs::DrawManagerState& draw_state_, CacheInputs::DirtyFlags& flags_,
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* draw_indirect_ =
            nullptr) const noexcept {
        return CacheInputs::OfStage(maxwell3d, regs_, state_, tables_, draw_state_, flags_,
                                    kepler_compute, gpu_memory, draw_indirect_);
    }

    /// Graphics uniform buffer `index` of `stage` as ForEachBoundBuffer visits it: its device
    /// address and bound size, or {0, 0} when it visits none.
    [[nodiscard]] std::pair<DAddr, u32> BoundGraphicsUniform(u32 stage, u32 index) const noexcept {
        if (((channel_state->enabled_uniform_buffer_masks[stage] >> index) & 1) == 0) {
            return {0, 0};
        }
        const Binding& binding = channel_state->uniform_buffers[stage][index];
        const auto* const sizes = channel_state->uniform_buffer_sizes;
        const u32 size = sizes ? (std::min)(binding.size, (*sizes)[stage][index]) : binding.size;
        if (binding.device_addr == 0 || size == 0) {
            return {0, 0};
        }
        return {binding.device_addr, size};
    }

    /// Graphics uniform buffer `index` of `stage` as the game bound it (FRITCHII_OOT_CAMERA_GUEST):
    /// its device address and bound size whether or not the current shaders use it, not cut to
    /// what they read; {0, 0} when unbound.
    [[nodiscard]] std::pair<DAddr, u32> GuestGraphicsUniform(u32 stage, u32 index) const noexcept {
        const Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.device_addr == 0 || binding.size == 0) {
            return {0, 0};
        }
        return {binding.device_addr, binding.size};
    }

    /// FRITCHII_ESPLIT_BIND_BATCH=2: a hash of everything a graphics uniform bind of `stage`,
    /// `index` writes (the binding, its bind memo, its dirty bit), to prove a dropped bind would
    /// have left it unchanged.
    [[nodiscard]] u64 UniformBindState(size_t stage, u32 index) const noexcept {
        if (stage >= NUM_STAGES || index >= NUM_GRAPHICS_UNIFORM_BUFFERS || channel_state == nullptr) {
            return 0;
        }
        const Binding& binding = channel_state->uniform_buffers[stage][index];
        const UniformBindMemo& memo = uniform_bind_memo[stage][index];
        u64 hash = 0xcbf29ce484222325ULL;
        const auto mix = [&hash](u64 value) {
            hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        };
        mix(binding.device_addr);
        mix(binding.gpu_addr);
        mix(binding.size);
        mix(binding.buffer_id.index);
        mix(binding.segment_first);
        mix(binding.segment_count);
        mix(memo.epoch);
        mix(memo.map_serial);
        mix(static_cast<u64>(memo.as_id));
        mix(reinterpret_cast<u64>(memo.channel));
        mix(memo.gpu_addr);
        mix(memo.device_addr);
        mix(memo.size);
        mix((channel_state->dirty_uniform_buffers[stage] >> index) & 1);
        return hash;
    }

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectCount();

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectBuffer();

    template <typename Func>
    void BufferOperations(Func&& func) {
        do {
            channel_state->has_deleted_buffers = false;
            func();
        } while (channel_state->has_deleted_buffers);
    }

    Common::SpinRecursiveMutex mutex;
    /// EXP86 (fable52_answer_gpuwritten): per-page mirrors of gpu_modified_ranges, the tracker's
    /// Preflushable bits and read_trapped_ranges, all changed under `mutex`, with a sequence
    /// counter that is odd from a hold's first change to its release (the mutex's release hook).
    GpuWrittenMirror gpu_written_mirror{mutex};

    /// EXP86: what the locked GetFlushArea would do for a guest read of [addr, addr + size), from
    /// the mirrors and without the lock. Clean: it would answer the preemptive page-aligned area
    /// and change nothing (EXP81's read bits aside). Anything else: take the locked path.
    [[nodiscard]] MirrorAnswer MirrorCleanForRead(DAddr addr, u64 size) const {
        if (!PageBitmap::InRange(Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
                                 Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE) -
                                     Common::AlignDown(addr, Core::DEVICE_PAGESIZE))) {
            return MirrorAnswer::Broken;
        }
        const u64 seq = gpu_written_mirror.seq.Load();
        if ((seq & 1) != 0) {
            return MirrorAnswer::Busy;
        }
        const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const u64 length = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE) - begin;
        const bool modified = gpu_written_mirror.gpu_modified_pages.Any(begin, length);
        // "Any page", as IsRegionPreflushable asks.
        const bool marked = gpu_written_mirror.preflush_pages.Any(begin, length);
        const bool armed = gpu_written_mirror.trap_pages.Any(begin, length);
        if (gpu_written_mirror.seq.Load() != seq) {
            return MirrorAnswer::Busy;
        }
        if (gpu_written_mirror.seq.Broken()) {
            return MirrorAnswer::Broken;
        }
        if (modified) {
            return MirrorAnswer::Modified;
        }
        if (armed && !marked) {
            return MirrorAnswer::ArmedUnmarked;
        }
        // F2 counter: a read answered here skips GetFlushArea (verify mode's calls are reads too).
        if (DownloadReadStats::g_enabled) [[unlikely]] {
            DownloadReadStats::Read(begin, begin + length);
        }
        // EXP81 (fable52_answer_fences2 2): this read's pages, for TickFrame's read stamps.
        if (VideoCore::Nx2Debug::Experiment(81) > 0) [[unlikely]] {
            preflush_read_pages.Set(begin, length);
        }
        return MirrorAnswer::Clean;
    }

    /// EXP86 verify (`mutex` held, before GetFlushArea): whether GetFlushArea leaves the read
    /// traps alone for this read: a Preflushable mark on its pages, or nothing armed there.
    [[nodiscard]] bool ReadLeavesTrapsAlone(DAddr addr, u64 size) {
        const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        return memory_tracker.IsRegionPreflushable(addr, size) ||
               !read_trapped_ranges.Intersects(begin, end - begin);
    }

    Runtime& runtime;
    bool any_buffer_uploaded = false;
    /// The guest words of each page of native programs' indirectly read banks when last checked:
    /// unchanged pages keep their tracking (marking them CPU-modified costs two mprotect calls
    /// per page).
    struct NativeBankPage {
        std::array<u8, Core::DEVICE_PAGESIZE> words;
        /// EDEN_NX2_EXP4 & 2: the bytes of the page compared during draw `checked_draw`.
        u64 checked_draw = 0;
        u32 checked_begin = 0;
        u32 checked_end = 0;
        /// EDEN_NX2_EXP9: frame_tick of the page's last compare, and whether the next bind
        /// compares it again (a flagged write's bytes may land after the compare that took it).
        u64 checked_frame = ~u64{0};
        /// frame_tick when the page was last seen changed: such pages are compared on every bind.
        u64 changed_frame = ~u64{0};
        bool recheck = false;
        /// FRITCHII_BANK_CENSUS (counters only, set while it is on): the write ring's sequence
        /// before the page's last compare (2), the census bind number when the page's last
        /// collected or buffer cache write flag was taken (0: none yet), and that compare's reason.
        u8 census_flag = 0; ///< 1 collected CPU write, 2 buffer cache write
        u8 census_last_reason = 0;
        /// FRITCHII_BANK_FLAGGED=2: visits left to watch a "missed" page for a late flag.
        u8 missed_watch = 0;
        u64 census_ring_seq = 0;
        u64 census_flag_bind = 0;
    };
    /// Node map: NoteNativeBankPageChanged keeps pointers to earlier pages' words until the bind's
    /// uploads are flushed, so entries must not move (std::unordered_map's lookups were ~2% of the
    /// split's back at Castle Town, 600k a second).
    boost::unordered_node_map<DAddr, NativeBankPage> native_indirect_page_snapshots;
    /// EDEN_NX2_BANK_PAGE_TABLE (live, default 0): the snapshots in a page-indexed table instead
    /// of the node map (the shape of the recomp's write-back pages: 2 MiB
    /// regions of 512 page slots under a flat top level, the last region looked up kept). Pages
    /// are allocated one by one, so their words stay where they are, as in the node map.
    static constexpr u64 NativeBankRegionBits = 21;
    static constexpr u64 NativeBankRegionPages = u64{1}
                                                 << (NativeBankRegionBits - Core::DEVICE_PAGEBITS);
    static constexpr u64 NativeBankAddressBits = Core::DevicePageFlags::AddressBits;
    struct NativeBankRegion {
        std::array<std::unique_ptr<NativeBankPage>, NativeBankRegionPages> pages;
        /// A bit per page slot holding a snapshot.
        std::array<u64, NativeBankRegionPages / 64> present{};
        /// A bit per page whose NativeBankPage::recheck is set (kept with it).
        std::array<u64, NativeBankRegionPages / 64> recheck{};
    };
    using NativeBankRegionTable =
        std::array<std::unique_ptr<NativeBankRegion>,
                   u64{1} << (NativeBankAddressBits - NativeBankRegionBits)>;
    std::unique_ptr<NativeBankRegionTable> native_bank_regions;
    /// The regions allocated in native_bank_regions, to clear them.
    std::vector<u32> native_bank_region_list;
    size_t native_bank_table_pages = 0;
    u64 native_bank_last_region_index = ~u64{0};
    NativeBankRegion* native_bank_last_region = nullptr;
    /// Which store holds the snapshots: the switch as last seen (a change empties both).
    bool native_bank_table_on = false;
    /// A page's snapshot, and the region holding it with the table (null with the node map).
    struct NativeBankSnapshotRef {
        NativeBankPage* page;
        NativeBankRegion* region;
        bool inserted;
    };
    /// Counts graphics draws (UpdateGraphicsBuffers) for EDEN_NX2_EXP4 & 2.
    u64 native_bank_draw = 0;
    /// EDEN_NX2_EXP4 != 0: bank check cost, logged every 5 s.
    struct NativeBankCheckStats {
        u64 binds = 0;
        u64 bytes_compared = 0;
        u64 bytes_trimmed = 0;
        u64 pages_compared = 0;
        u64 pages_skipped = 0;
        u64 pages_changed = 0;
        std::chrono::steady_clock::time_point last_log{};
    } native_bank_stats;
    /// EDEN_NX2_EXP9 != 0: why bank pages were compared and which of them had changed, by the
    /// first reason that applies (NativeBankReason), logged every 5 s.
    /// EDEN_NX2_EXP9: a page that changed within this many frames is compared on every bind.
    static constexpr u64 HotFrames = 120;
    enum NativeBankReason : size_t {
        BankCollected, ///< Collected CPU writes (Core::g_collected_write_pages)
        BankWritten,   ///< Writes the buffer cache saw (Core::g_buffer_write_pages)
        BankRecheck,   ///< Flagged at the previous bind
        BankCpuDirty,  ///< CPU-modified in the tracker
        BankHot,       ///< Changed within the last HotFrames frames (per-draw data)
        BankFrame,     ///< Not compared yet this frame
        BankNone,      ///< No reason: EXP9=2 compares these only to verify
        BankReasons,
    };
    /// FRITCHII_BANK_FAST (fable_b 25.2 cut b): what the last full CheckNativeBankPagesTracked
    /// decided that does not depend on the bank: the switch generation it read its switches at,
    /// the lost-store count it saw, its mode, and the slots whose settled banks it skips whole.
    struct NativeBankFast {
        u32 generation{};
        u64 overflows{};
        u32 skip_slots{};    ///< own (or flagged) slots, no verify, mode 3, the page table on
        u32 flagged_slots{}; ///< of those, FRITCHII_BANK_FLAGGED's (for its counts)
        int mode{};
        bool ready{};
    } native_bank_fast;
    u64 native_bank_fast_hits{};

    struct NativeBankTrackedStats {
        u64 binds = 0;
        u64 pages_new = 0;
        u64 pages_skipped = 0;
        std::array<u64, BankReasons> compared{};
        std::array<u64, BankReasons> changed{};
        /// EDEN_NX2_BANK_SLOTS: binds whose page loop was skipped whole, and pages of those
        /// slots the hot rule would have compared.
        u64 banks_skipped = 0;
        u64 hot_left = 0;
        /// EDEN_NX2_BANK_SKIP_VERIFY: pages compared anyway, those found changed, and pages of a
        /// skipped bank the loop had a reason for.
        u64 skip_verified = 0;
        u64 skip_missed = 0;
        u64 skip_disagreed = 0;
        /// FRITCHII_BANK_FLAGGED: the same counts for its slots alone, and the snapshot pages
        /// uploads refreshed (RefreshNativeBankSnapshots), those whose bytes differed.
        u64 flagged_binds = 0;
        u64 flagged_skipped = 0;
        u64 flagged_hot_left = 0;
        u64 flagged_verified = 0;
        u64 flagged_missed = 0;
        /// Of the missed pages, those a flag reached within their next 8 visits, and those none did.
        u64 flagged_missed_then_flagged = 0;
        u64 flagged_missed_never_flagged = 0;
        u64 flagged_disagreed = 0;
        std::array<u64, BankReasons> flagged_disagreed_by{}; ///< by the reason the loop found
        u64 refreshed_pages = 0;
        u64 refreshed_changed = 0;
        std::chrono::steady_clock::time_point last_log{};
    } native_bank_tracked_stats;
    /// FRITCHII_BANK_FLAGGED: on at the last bank bind (uploads refresh the snapshots while on),
    /// Common::g_collected_overflows as of the last snapshot drop, and how many drops so far.
    bool native_bank_flagged_on = false;
    u64 native_bank_flagged_overflows = 0;
    u64 native_bank_flagged_resets = 0;
    /// The bank marks' gate (Common::g_bank_marks) as last set, the frame it changed, and whether
    /// the snapshots were dropped two frames after it opened (the flagged rule waits for that).
    bool native_bank_marks_wanted = false;
    bool native_bank_marks_ready = false;
    u64 native_bank_marks_since = 0;
    /// EDEN_NX2_BANK_SLOT_STATS: the tracked check's counts per constant buffer index (c0..c17),
    /// logged and cleared with native_bank_tracked_stats.
    struct NativeBankSlotStats {
        u64 binds = 0;
        u64 pages = 0;     ///< pages the binds covered (mapped)
        u64 pages_new = 0; ///< pages first seen (no snapshot yet)
        u32 stages = 0;    ///< stages seen binding the index (bit per stage)
        u64 banks_skipped = 0; ///< EDEN_NX2_BANK_SLOTS: page loops skipped whole
        u64 hot_left = 0;      ///< EDEN_NX2_BANK_SLOTS: hot pages not compared
        std::array<u64, BankReasons> compared{};
        std::array<u64, BankReasons> changed{};
    };
    std::array<NativeBankSlotStats, Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers>
        native_bank_slot_stats{};
    /// FRITCHII_BANK_CENSUS: for each change the tracked check finds, what touched the page
    /// since its last compare, logged and cleared every 10 s (LogNativeBankCensus).
    struct NativeBankCensusStats {
        std::array<u64, BankReasons> changed{}; ///< changes by reason (all of them)
        u64 skip_missed = 0;  ///< EDEN_NX2_BANK_SKIP_VERIFY's "missed"
        u64 unflagged = 0;    ///< hot, frame check, verify only and skip-verify changes
        /// Unflagged changes by census binds since the page's last flag take: <= 4, <= 32,
        /// <= 256, more, never.
        std::array<u64, 5> flag_distance{};
        u64 flag_late = 0;         ///< unflagged, a flag taken <= 32 binds before
        u64 flag_late_buffer = 0;  ///< of those, the flag was a buffer cache write's
        u64 uploaded = 0;          ///< unflagged, an upload covered the page since its last compare
        u64 uploaded_after_dirty = 0; ///< of those, the last compare was for a CPU-modified page
        u64 gpu_written = 0;       ///< unflagged, the page ever had a GPU write mark
        u64 ring_found = 0;        ///< unflagged, a write ring entry since the last compare (2)
        u64 ring_lost = 0;         ///< unflagged, the ring's page mark set but no entry found (2)
        u64 ring_wrapped = 0;      ///< of those, the ring wrapped since the last compare
        /// Unflagged with no late flag, no upload since the compare and (2) no emulator write.
        u64 unseen = 0;
        u64 unseen_first = 0;      ///< of those, the page's first compare this frame
        u64 unseen_gpu_written = 0; ///< of those, the page ever had a GPU write mark
        std::array<u64, Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers> slots{}; ///< unflagged
        std::array<u64, Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers> slots_unseen{};
        /// The newest write ring entry over the page since its last compare, by site and write
        /// function (2).
        struct Writer {
            const char* site = nullptr;
            const char* via = nullptr;
            u64 count = 0;
        };
        std::array<Writer, 12> writers{};
        u64 writers_other = 0;
        std::chrono::steady_clock::time_point last_log{};
    } native_bank_census;
    /// FRITCHII_BANK_CENSUS: bank binds counted while it is on (never cleared: page distances).
    u64 native_bank_census_bind = 0;
    /// FRITCHII_BANK_CENSUS: pages an upload (UploadMemory) copied to a host buffer, taken by the
    /// page's next compare. Made at the census' first bind; read by UploadMemory without the lock.
    std::unique_ptr<Core::DevicePageFlags> native_bank_census_upload_owner;
    std::atomic<Core::DevicePageFlags*> native_bank_census_uploads{nullptr};
    /// FRITCHII_BANK_CENSUS=2 asked for the write ring, and whether it turned it on.
    bool native_bank_census_ring_wanted = false;
    bool native_bank_census_ring_owner = false;
    /// EDEN_NX2_EXP26 & 4: changed bank pages of one bind, copied to the host buffer from their
    /// snapshots (FlushNativeBankUploads) instead of being marked CPU-modified. A 64 KiB bank
    /// spans at most 17 pages.
    struct NativeBankUpload {
        DAddr page;
        const u8* words;
    };
    static constexpr size_t NativeBankUploadCapacity = 32;
    struct NativeBankUploads {
        /// Only the first `count` are set (left uninitialized: one of these per bank bind).
        std::array<NativeBankUpload, NativeBankUploadCapacity> pages;
        size_t count = 0;
    };

private:
    template <typename Mask, typename Func>
    static void ForEachEnabledBit(Mask enabled_mask, Func&& func) {
        for (u32 index = 0; enabled_mask != 0; ++index, enabled_mask >>= 1) {
            const int disabled_bits = std::countr_zero(enabled_mask);
            index += disabled_bits;
            enabled_mask >>= disabled_bits;
            func(index);
        }
    }

    template <typename Func>
    void ForEachBufferInRange(DAddr device_addr, u64 size, Func&& func) {
        const u64 page_end = Common::DivCeil(device_addr + size, CACHING_PAGESIZE);
        for (u64 page = device_addr >> CACHING_PAGEBITS; page < page_end;) {
            const BufferId buffer_id = page_table[page];
            if (!buffer_id) {
                ++page;
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            func(buffer_id, buffer);

            const DAddr end_addr = buffer.CpuAddr() + buffer.SizeBytes();
            page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
        }
    }

    static bool IsRangeGranular(DAddr device_addr, size_t size) {
        return (device_addr & ~Core::DEVICE_PAGEMASK) ==
               ((device_addr + size) & ~Core::DEVICE_PAGEMASK);
    }

    void RunGarbageCollector(bool aggressive_gc);

    void BindHostIndexBuffer();

    void BindHostVertexBuffers();

    void BindHostDrawIndirectBuffers();

    void BindHostGraphicsUniformBuffers(size_t stage);

    void BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind);

    /// EDEN_NX2_EXP9: the native uniform bank check of a bind of [device_addr, device_addr + size).
    /// EDEN_NX2_EXP26 (bits, see Experiment in nx2_debug_control.h): `trims`; with & 4 changed
    /// pages that lie whole in `buffer` are uploaded from their snapshots.
    /// `stage` and `index`: the binding's shader stage and constant buffer index
    /// (EDEN_NX2_BANK_SLOT_STATS).
    void CheckNativeBankPagesTracked(DAddr device_addr, u32 size, int mode, int trims,
                                     Buffer& buffer, size_t stage, u32 index);

    /// A native bank page whose snapshot was just (re)taken: marked CPU-modified, or with
    /// EXP26 & 4 (`upload_to` not null) queued in `uploads` when it lies whole in that buffer and
    /// holds no GPU-written data.
    void NoteNativeBankPageChanged(DAddr page, const u8* words, Buffer* upload_to,
                                   NativeBankUploads& uploads);

    /// EDEN_NX2_EXP26 & 4: copies the queued pages' snapshots into `buffer`.
    void FlushNativeBankUploads(Buffer& buffer, const NativeBankUploads& uploads);

    /// EDEN_NX2_BANK_PAGE_TABLE: picks the snapshot store from the switch (emptying both when it
    /// changed) and empties it once it holds 16384 pages, as before each bank check.
    void SelectNativeBankStore();
    /// The snapshot of `page` (page aligned), made (words left unset) when there is none.
    [[nodiscard]] NativeBankSnapshotRef NativeBankSnapshot(DAddr page);
    /// The snapshot of `page` (page aligned), or null.
    [[nodiscard]] NativeBankPage* FindNativeBankSnapshot(DAddr page);
    void ClearNativeBankSnapshots();
    /// EDEN_NX2_BANK_SLOTS, with the table only: whether every page of [first_page, end) (page
    /// aligned) has a snapshot and none waits for a recheck. False with the node map.
    [[nodiscard]] bool NativeBankPagesSettled(DAddr first_page, DAddr end);
    /// The tracked check's 5 s report (every 1024 binds it looks at the clock).
    void LogNativeBankTrackedStats(int mode);
    /// FRITCHII_BANK_CENSUS: one change the tracked check found on `page` (after the compare,
    /// before the snapshot is taken again). `unflagged`: found by no write flag (hot, frame
    /// check, verify only, skip verify); `uploaded`: the census' upload flag the compare took;
    /// `ring_marked`: the write ring's page mark the compare took (2); `ring_since`: the page's
    /// write ring sequence before this compare; `first_this_frame`: the page's first compare
    /// this frame.
    void NoteNativeBankCensusChange(DAddr page, const NativeBankPage& snapshot, int census,
                                    NativeBankReason reason, bool unflagged, bool uploaded,
                                    bool ring_marked, u64 ring_since, bool first_this_frame,
                                    u32 index);
    /// FRITCHII_BANK_FLAGGED: the snapshot bytes of [device_addr, device_addr + size) set to
    /// `bytes` (cached memory, never the write-combined staging), what an upload gives the host
    /// buffer (MappedUploadMemory, InlineMemoryImplementation). Pages without a snapshot are left
    /// alone (their first bind takes one and marks the page changed).
    void RefreshNativeBankSnapshots(DAddr device_addr, u64 size, const u8* bytes);
    /// FRITCHII_BANK_FLAGGED: whether any page of [begin, end) has a snapshot (the table's
    /// present bits, 64 pages a word; with the node map, whether it holds any snapshot).
    [[nodiscard]] bool NativeBankAnySnapshot(DAddr begin, DAddr end);
    /// FRITCHII_BANK_FLAGGED: guest bytes of an upload over snapshotted pages, read once here
    /// (cached), then copied to staging and into the snapshots.
    Common::ScratchBuffer<u8> native_bank_refresh_scratch;
    /// FRITCHII_BANK_CENSUS: the 10 s line.
    void LogNativeBankCensus(std::chrono::steady_clock::time_point now);
    /// Sets the page's recheck flag (and its region bit with the table).
    static void SetNativeBankRecheck(const NativeBankSnapshotRef& ref, DAddr page, bool value) {
        ref.page->recheck = value;
        if (ref.region != nullptr) {
            const u64 slot = (page >> Core::DEVICE_PAGEBITS) & (NativeBankRegionPages - 1);
            const u64 bit = u64{1} << (slot % 64);
            if (value) {
                ref.region->recheck[slot / 64] |= bit;
            } else {
                ref.region->recheck[slot / 64] &= ~bit;
            }
        }
    }

    void BindHostGraphicsStorageBuffers(size_t stage);

    void BindHostGraphicsTextureBuffers(size_t stage);

    void BindHostTransformFeedbackBuffers();

    void BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size, u32 stride);

    void BindHostComputeUniformBuffers();

    void BindHostComputeStorageBuffers();

    void BindHostComputeTextureBuffers();

    void DoUpdateGraphicsBuffers(bool is_indexed);

    void DoUpdateComputeBuffers();

    void UpdateIndexBuffer();

    void UpdateVertexBuffers();

    void UpdateVertexBuffer(u32 index, const CacheInputs& inputs);

    /// FRITCHII_FILL_UPDATE_KEEP (fable_b 25.2 cut f): resolve() (the translation and the buffer
    /// lookup UpdateIndexBuffer or UpdateVertexBuffer make), or the binding it gave last time for
    /// `slot` (a vertex slot; NUM_VERTEX_BUFFERS for the index buffer) while the inputs
    /// (`begin`, `extent`, `enable`), the mappings, the address space, the channel and the set of
    /// buffers (binding_epoch) are unchanged. 2 and 3 resolve anyway and compare.
    template <typename Resolve>
    Binding KeepGeometryBinding(u32 slot, GPUVAddr begin, u64 extent, u32 enable,
                                Resolve&& resolve);

    void UpdateDrawIndirect();

    void UpdateUniformBuffers(size_t stage);

    void UpdateStorageBuffers(size_t stage);

    void UpdateTextureBuffers(size_t stage);

    void UpdateTransformFeedbackBuffers();

    void UpdateTransformFeedbackBuffer(u32 index);

    void UpdateComputeUniformBuffers();

    void UpdateComputeStorageBuffers();

    void UpdateComputeTextureBuffers();

    /// `capped`: `size` is EDEN_NX2_WRITTEN_SSBO_MARK_KIB's cap, below the binding's size, so the
    /// GPU may write past it (Nx2NoteListWrite).
    void MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync = false,
                           bool capped = false);

    /// Counts a GPU write to the command list pages in the range (invalidating their copies),
    /// notes the words it may write (Nx2ListPage::gpu_words) and, while collecting, asks for new
    /// copies. `to_page_end`: the GPU may write past the range (a capped mark), so the words up to
    /// the end of its last page count as written too.
    void Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size,
                          bool to_page_end = false);

    [[nodiscard]] BufferId FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible);

    void SynchronizeBufferWrites(Buffer& buffer);

    [[nodiscard]] OverlapResult ResolveOverlaps(DAddr device_addr, u32 wanted_size);

    /// Returns whether it copied (false: skipped as wholly CPU-modified, EDEN_NX2_JOIN_SKIP_CPU).
    bool JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);
    /// The range's host contents are all stale and nothing else reads them (JoinOverlap).
    [[nodiscard]] bool JoinCopyIsDead(DAddr device_addr, u64 size);

    [[nodiscard]] BufferId CreateBuffer(DAddr device_addr, u32 wanted_size,
                                        bool sparse_compatible);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    void TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept;

    bool SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size);

    /// EDEN_NX2_EXP59 & 8: before the full walk, counts and logs what the dirty-word walk would
    /// have left out. Returns whether the dirty-word walk would have walked anything.
    bool VerifyComputeDirtyWalk(DAddr device_addr, u32 size);

    /// EDEN_NX2_STORAGE_SYNC_SCAN=2: before the full walk, counts and logs (every 10 s) the CPU
    /// and unprotected pages the scanned walk would have left out.
    void VerifyStorageSyncScan(DAddr device_addr, u32 size);

    /// EDEN_NX2_STORAGE_SYNC_MEMO: SynchronizeBuffer, skipped when the range's memo still holds;
    /// stores the memo when the synchronize left the range clean. `verify` (=2): a hit checks
    /// the range and synchronizes anyway. Returns what SynchronizeBuffer returns (true: nothing
    /// uploaded).
    bool SynchronizeStorageMemo(Buffer& buffer, DAddr device_addr, u32 size, bool verify);

    /// FRITCHII_FILL_GEOM_KEEP (fable_b 25.2 cut e): SynchronizeBuffer for vertex slot `slot`
    /// (NUM_VERTEX_BUFFERS: the index buffer), skipped at 1 while the slot's memo holds (the range
    /// and its tracker managers' dirty generation sum when a synchronize left it clean, as
    /// EDEN_NX2_STORAGE_SYNC_MEMO); 2 and 3 synchronize anyway and check a hit (must be 0).
    bool SynchronizeGeometry(Buffer& buffer, DAddr device_addr, u32 size, u32 slot);

    void UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                      std::span<BufferCopy> copies);

    void ImmediateUploadMemory(Buffer& buffer, u64 largest_copy,
                               std::span<const BufferCopy> copies);

    void MappedUploadMemory(Buffer& buffer, u64 total_size_bytes, std::span<BufferCopy> copies);

    void DownloadBufferMemory(Buffer& buffer_id);

    void DownloadBufferMemory(Buffer& buffer_id, DAddr device_addr, u64 size);

    /// Downloads the GPU-written data of several buffers with one wait for the GPU.
    void DownloadBuffersMemory(std::span<const BufferId> buffer_ids);

    void DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges);

    /// ClearDownload, for the downloads taken at fences only.
    void ClearFenceDownloads(DAddr device_addr, u64 size);

    /// Whether every GPU-modified byte in the range is in a read download's clean_ranges.
    [[nodiscard]] bool ReadDownloadsCover(DAddr device_addr, u64 size) const;

    /// Writes the read downloads up to `tick` to guest memory, oldest first (`mutex` held, the
    /// host GPU past `tick`).
    void WriteReadDownloads(u64 tick);

    /// WriteReadDownloads for those the host GPU has completed: nothing waits.
    void WriteCompletedReadDownloads();

    /// WriteReadDownloads up to the newest read download over the range, if the host GPU ran it.
    /// Returns whether it did.
    bool WriteFinishedReadDownloads(DAddr device_addr, u64 size);

    /// Records and submits a read download of the GPU-modified bytes over `ranges` that no read
    /// download brings yet.
    void RecordReadDownload(std::span<const std::pair<DAddr, u64>> ranges);

    void DeleteBuffer(BufferId buffer_id, bool do_not_mark = false);

    [[nodiscard]] Binding StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                               bool native_pointer = false,
                                               bool table_pointer = false,
                                               bool compute = false) const;

    void TraceTablePointer(bool compute, size_t stage, size_t ssbo_index, u32 cbuf_index,
                           u32 cbuf_offset, u32 indirect_offset, GPUVAddr holder,
                           const Binding& binding);

    [[nodiscard]] TextureBufferBinding GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                               PixelFormat format);

    [[nodiscard]] std::span<const u8> ImmediateBufferWithData(DAddr device_addr, size_t size);

    [[nodiscard]] std::span<u8> ImmediateBuffer(size_t wanted_capacity);

    [[nodiscard]] bool HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept;

    [[nodiscard]] Binding& VertexBufferSlot(u32 index);

    [[nodiscard]] const Binding& VertexBufferSlot(u32 index) const;

    void UpdateVertexBufferSlot(u32 index, const Binding& binding);

    /// Flags the vertex buffers overlapping a written range so host copies derived from them are
    /// rebuilt.
    void MarkVertexBuffersStale(DAddr device_addr, u64 size);

    /// Re-dirties repacked vertex buffers whose source data was written since they were bound.
    void RefreshRepackedVertexBuffers();
    [[nodiscard]] bool IsRepackedVertexBuffer(u32 index);

    void InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                    std::span<const u8> inlined_buffer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;

    Common::SlotVector<Buffer> slot_buffers;
#ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
#else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    StampedDestructionRing<Buffer, TICKS_TO_DESTROY> delayed_destruction_ring;

    const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect{};
    const CacheInputs* inputs_override{};

    u32 last_index_count = 0;

    u32 enabled_vertex_buffers_mask = 0;
    /// Streams that went from a binding to none since the last BindHostVertexBuffers: the host
    /// side forgets the buffer it remembers for them there (runtime.ForgetVertexBinding).
    u32 forget_vertex_buffers_mask = 0;
    u32 stale_vertex_buffers_mask = 0;
    u64 vertex_buffers_serial = 0;
    std::array<Binding, 32> v_buffer{};

    boost::container::small_vector<BufferCopy, 4> upload_copies;

    MemoryTracker memory_tracker;
    Common::RangeSet<DAddr> uncommitted_gpu_modified_ranges;
    /// MayHaveUncommittedFlushes's note: raised before every add to the set above (its ranges
    /// move to committed_gpu_modified_ranges only from there).
    std::atomic<bool> uncommitted_flushes_noted{false};

    /// Whether GPU-written buffer ranges are copied back to guest memory at the next fence;
    /// otherwise only when something flushes them (as the original code below High GPU accuracy).
    /// Not for Switch 2 titles: title 1 had 250-330 MB/s copied back (a fence per compute
    /// dispatch, the fencing thread holding the buffer cache while it wrote them), while its CPU
    /// read GPU-written memory at most 64 times in 200 s (counted with reactive flushing) and
    /// the native MME's DMA reads never touched those pages. Without them a fence costs ~1 us
    /// instead of ~5 and draw setup ~7% less. EDEN_NX2_BUFFER_DOWNLOADS=0/1 forces it.
    bool AsyncBufferDownloads() const {
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_BUFFER_DOWNLOADS");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        if (forced >= 0) {
            return forced == 1;
        }
        return gpu_memory == nullptr || !gpu_memory->UsesNativeTiling();
    }

    /// Experiment, EDEN_NX2_WRITTEN_SSBO_MARK_KIB=<n>: marks at most n KiB of a written Switch 2
    /// storage buffer binding GPU-written. Native pointers carry no size, so a written binding
    /// spans up to 16 MiB of its mapping and arms read traps and fence downloads over CPU data
    /// the shader never writes. Writes past the cap are not tracked, so this only measures how
    /// much of the marked range the GPU really writes.
    u32 WrittenStorageMarkSize(u32 size) const {
        static const u64 cap = [] {
            const char* value = std::getenv("EDEN_NX2_WRITTEN_SSBO_MARK_KIB");
            return value == nullptr ? u64{0} : std::strtoull(value, nullptr, 10) * 1024;
        }();
        if (cap == 0 || gpu_memory == nullptr || !gpu_memory->UsesNativeTiling()) {
            return size;
        }
        return static_cast<u32>((std::min<u64>)(size, cap));
    }

    /// Write extent self-check (fable_answer_25; Ocarina of Time's glowing triangles,
    /// 2026-11-09). The store analysis bounds how far a native compute program writes, and only
    /// that much is marked GPU-written; one program writes past it, and the CPU then read stale
    /// guest bytes back (particle positions turned into one right vertex and garbage). Each program
    /// is marked in full until a dispatch at its dims was checked: its tail past the proof, as
    /// the GPU left it after that one dispatch, is compared with guest memory from right before
    /// it (known GPU writes and unapplied CPU writes left out). The same: the proof holds at these
    /// dims (and at smaller ones). Different: the program is marked in full for the session and
    /// logged, naming it. EDEN_NX2_EXP45=1 (live) skips the check (the proof is trusted as before).
    struct ExtentTrust {
        std::array<u32, 3> block{};
        std::array<u32, 3> grid{};
        bool distrusted{};
        bool probe_pending{};
    };
    struct ExtentProbeRequest {
        bool active{};
        u64 proven{};
        u64 pointer{};
        u64 pad{};
        std::array<u32, 3> block{};
        std::array<u32, 3> grid{};
    };
    struct ExtentProbe {
        u64 shader_hash{};
        BufferId buffer_id{};
        DAddr device_addr{}; ///< The tail's start (in buffer_id)
        u64 binding_size{};
        ExtentProbeRequest request{};
        u64 span{};
        /// Guest memory of the tail at bind (fable_answer_26). Under the full mark the CPU's reads
        /// bring the GPU's bytes into guest memory, so only a copy made before the dispatch shows
        /// what the CPU would have read without the mark.
        std::vector<u8> guest;
        /// Tail offsets [begin, end) left out of the compare: known GPU writes and CPU writes not
        /// applied to the buffer yet.
        std::vector<std::pair<u64, u64>> skip;
        std::optional<Async_Buffer> after;
        u64 tick{};
    };
    u64 compute_shader_hash{};
    std::unordered_map<u64, ExtentTrust> extent_trust;
    std::array<ExtentProbeRequest, NUM_STORAGE_BUFFERS> extent_probe_requests{};
    std::vector<ExtentProbe> extent_probes;
    u64 extent_probes_trusted{};
    u64 extent_probes_distrusted{};
    u64 extent_full_marks{};

    void CheckExtentProbes();
    void ProbeExtentTail(u32 index, const Binding& binding);

    /// FRITCHII_EXTENT_CENSUS (live): why native compute bindings are marked in full, per program
    /// and binding; FRITCHII_EXTENT_CBUF (live): the store extents proven with constant buffer
    /// words (=1 marks by them, =3 only counts). One line each every 10 s (LogExtentCensus).
    enum class FullMarkWhy : u8 {
        Analysis,     ///< the store analysis left the extent unknown (ExtentUnknownWhy says why)
        IndirectGrid, ///< proven, but the dispatch is indirect (its group count is the GPU's)
        GuardOrDims,  ///< proven, but the 16-bit guard fails (or a block size is 0) at these dims
        Distrusted,   ///< an extent check caught the program writing past its proof
        Checking,     ///< marked in full while a dispatch at new dims is checked
        WholeBinding, ///< the proof covers the whole binding
        Count,
    };
    enum class CbufOutcome : u8 {
        None,       ///< no cbuf extent, or its switch (FRITCHII_EXTENT_CBUF, _SHARED, _GUARD) off
        Resolved,   ///< the words were read and the bound holds
        Fallback,   ///< a word's page is GPU-written (guest memory may be stale): full mark
        Unresolved, ///< a word not readable, the guard fails, or an indirect grid
    };
    struct ExtentCensusEntry {
        u64 dispatches{};          ///< every bind of it while the census or the cbuf switch is on
        u64 marked_bytes{};        ///< what they marked
        u64 unknown_dispatches{};  ///< with the store extent unknown
        u64 no_extra_dispatches{}; ///< of them, with no Shader::StorageWriteExtra (no reason)
        u64 full_dispatches{};
        u64 full_bytes{};
        std::array<u64, static_cast<size_t>(FullMarkWhy::Count)> whys{};
        Shader::ExtentUnknownWhy analysis{};
        u32 pointer_cbuf_index{};
        u32 pointer_cbuf_offset{};
        u64 binding_bytes{}; ///< the latest binding's size
        Shader::StorageCbufExtent cbuf_extent{};
        // FRITCHII_EXTENT_CBUF
        u64 cbuf_resolved{};
        u64 cbuf_fallback{};
        u64 cbuf_unresolved{};
        u64 cbuf_proven_bytes{};  ///< summed over the resolved dispatches
        u64 cbuf_binding_bytes{}; ///< the same dispatches' binding sizes
        u64 cbuf_marked{};        ///< =1: dispatches marked by the cbuf proof
        /// FRITCHII_EXTENT_SHARED: Known when cbuf_extent came through shared memory slots
        /// (cbuf_extent.via_shared; the counts above then follow FRITCHII_EXTENT_SHARED), else
        /// why that failed (Unrecorded: not tried)
        Shader::ExtentUnknownWhy shared_why{};
        /// FRITCHII_EXTENT_GUARD: Known when cbuf_extent was proven with the branch conditions on
        /// the stores' paths (cbuf_extent.via_guard; the counts then follow
        /// FRITCHII_EXTENT_GUARD), else why that failed (Unrecorded: not tried)
        Shader::ExtentUnknownWhy guard_why{};
    };
    std::map<std::pair<u64, u32>, ExtentCensusEntry> extent_census;
    std::chrono::steady_clock::time_point extent_census_start{};
    u64 extent_census_frames{};
    /// FRITCHII_EXTENT_CBUF=3: per storage binding of the dispatch being bound, the bytes from the
    /// binding's start its cbuf proof would mark (0: none); BindHostComputeStorageBuffers hands it
    /// to the read ring's write marks (ReadRunStats::t_proven_end).
    std::array<u32, NUM_STORAGE_BUFFERS> cbuf_dry_proven{};
    /// FRITCHII_STORE_FOOTPRINT: per storage binding of the dispatch being bound, its footprint
    /// (an id of ReadRunStats' footprint table, 0: none) and the pointer's device address;
    /// BindHostComputeStorageBuffers hands it to the write marks (ReadRunStats::t_footprint).
    std::array<ReadRunStats::FootprintRef, NUM_STORAGE_BUFFERS> footprint_dry{};
    /// FRITCHII_FEEDER_LANE: per storage binding of the dispatch being bound, the bytes from the
    /// binding's start its loads and atomics can reach (its proven read extent), ~0: unknown
    /// (the whole binding). JudgeEarlyCompute narrows the feeder's inputs with it.
    std::array<u64, NUM_STORAGE_BUFFERS> lane_read_end{};
    /// FRITCHII_STORE_FOOTPRINT (armed: non-zero at boot): who made gpu_modified_ranges' bytes
    /// GPU-modified (it follows the set from the first armed bind on), the footprint of the mark
    /// MarkWrittenBuffer is making (BindHostComputeStorageBuffers sets it around the call), and
    /// =2's sampling count.
    StoreFootprintLive::FootprintShadow footprint_shadow;
    bool footprint_following{};
    ReadRunStats::FootprintRef mark_footprint{};
    u32 footprint_skip_count{};
    /// GetFlushArea, live 1 or 2: answers the read from guest memory when the shadow says no mark
    /// since its GPU-modified bytes were last clean writes them (re-arming its pages' traps).
    bool FootprintAnswersRead(DAddr device_addr, u64 size, DAddr page_begin, DAddr page_end,
                              int live);
    /// BindComputeStorageBuffer (armed): the binding's footprint for its marks (footprint_dry):
    /// as recorded, or reduced to its power-of-two part when its modulus isn't one and a 32-bit
    /// wrap of the stores' offsets isn't ruled out at these dims.
    void NoteBindFootprint(size_t ssbo_index, const Binding& binding,
                           const Shader::StorageWriteExtent& write_extent,
                           const Shader::StorageWriteExtra* extra,
                           const std::array<u32, 3>& block,
                           const std::optional<std::array<u32, 3>>& grid, GPUVAddr pointer);
    /// =2: a download landing's bytes over claimed-outside places against guest memory.
    void FootprintCheckLanding(const Common::RangeSet<DAddr>* cpu_written, DAddr device_addr,
                               const u8* data, u64 size);

    std::optional<u64> ResolveCbufExtent(const Shader::StorageCbufExtent& extent,
                                         const std::array<u32, 3>& block,
                                         const std::optional<std::array<u32, 3>>& grid,
                                         CbufOutcome& outcome);
    void NoteExtentCensus(u32 ssbo_index, const Binding& binding, u32 cbuf_index, u32 cbuf_offset,
                          u64 mark, FullMarkWhy why, const Shader::ExtentUnknownWhy& write_unknown,
                          const Shader::StorageCbufExtent& cbuf_extent, CbufOutcome cbuf_outcome,
                          u64 cbuf_proven, bool cbuf_marked, bool extent_unknown,
                          bool has_extra, const Shader::ExtentUnknownWhy& shared_why,
                          const Shader::ExtentUnknownWhy& guard_why);
    void LogExtentCensus();

    /// Whether a GPU write to this range is copied back at the next fence. Ranges the guest CPU
    /// has read (Preflushable, set by the first reactive flush) always are: later reads skip the
    /// flush and rely on these downloads, e.g. collision meshes skinned by compute shaders.
    bool DownloadsAtFence(DAddr device_addr, u64 size) noexcept {
        return AsyncBufferDownloads() || memory_tracker.IsRegionPreflushable(device_addr, size);
    }

    /// Queues a GPU write for download at the next fence where DownloadsAtFence holds. With only
    /// Preflushable marks, just the marked pages are queued: a write that covers one marked page
    /// of a large buffer would otherwise copy the whole buffer back at every fence.
    void QueueFenceDownload(DAddr device_addr, u64 size) {
        static const bool exact = [] {
            const char* value = std::getenv("EDEN_NX2_EXACT_PREFLUSH");
            return value == nullptr || value[0] != '0';
        }();
        // FRITCHII_FENCE_COMMIT_FAST: each add below raises the note first.
        if (AsyncBufferDownloads()) {
            uncommitted_flushes_noted.store(true, std::memory_order_relaxed);
            uncommitted_gpu_modified_ranges.Add(device_addr, size);
            return;
        }
        // Preflushed writes arm the read traps too: their data reaches guest memory only when
        // the fence's download lands, and a read before that saw stale words (Ocarina of Time's
        // whole screen flashed white at the sword boulder, ~180 times in 30 s). The trapped read
        // flushes on demand only while the data is still in flight (GetFlushArea).
        if (!exact) {
            if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
                uncommitted_flushes_noted.store(true, std::memory_order_relaxed);
                uncommitted_gpu_modified_ranges.Add(device_addr, size);
            }
            ArmReadTraps(device_addr, size);
            return;
        }
        const DAddr end = device_addr + size;
        u64 queued = 0;
        memory_tracker.ForEachPreflushableRange(device_addr, size, [&](DAddr begin, u64 length) {
            const DAddr lower = (std::max)(begin, device_addr);
            const DAddr upper = (std::min)(begin + length, end);
            if (lower < upper) {
                uncommitted_flushes_noted.store(true, std::memory_order_relaxed);
                uncommitted_gpu_modified_ranges.Add(lower, upper - lower);
                queued += upper - lower;
            }
        });
        ArmReadTraps(device_addr, size);
        if (Nx2DownloadTrace::g_enabled && queued != 0) [[unlikely]] {
            Nx2DownloadTrace::Queued(device_addr, size, queued);
        }
    }

    /// GPU writes that are not downloaded at fences: trap the guest CPU's next read of their
    /// pages, which flushes them on demand (or marks them Preflushable with preflushing enabled).
    void ArmReadTraps(DAddr device_addr, u64 size) {
        const DAddr begin = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
        if (!read_trapped_ranges.Arm(begin, end - begin)) {
            return;
        }
        device_memory.ArmReadTraps(begin, end - begin);
    }

    ReadTrapTracker read_trapped_ranges;
    /// Ranges the guest CPU read after GPU writes. A read that must flush downloads the GPU
    /// writes in all of them, so the reads that usually follow it find nothing left to flush.
    Common::RangeSet<DAddr> cpu_read_ranges;
    /// EXP69: the ranges GetFlushArea marked Preflushable since the last reset.
    Common::RangeSet<DAddr> preflush_marked_ranges;
    u64 preflush_reset_bytes{};
    /// EXP81: frame_tick of each marked device page's last CPU read (folded from
    /// preflush_read_pages by TickFrame), and bytes the aged reset kept marked because they were
    /// read within the last EXP81 frames.
    std::unordered_map<u64, u64> preflush_read_frames;
    /// EXP81 (fable52_answer_fences2 2): pages guest reads asked about since the last fold, set
    /// by GetFlushArea and MirrorCleanForRead's clean answer (no lock there, hence mutable).
    mutable ReadPageBitmap preflush_read_pages;
    u64 preflush_kept_bytes{};
    /// fable52_answer_bosswave 3 (counter before the change): preflushed reads that flush
    /// although every GPU-modified byte of theirs is in a fence download already on its way
    /// (none uncommitted), and how many fences behind the newest that download was.
    std::atomic<u64> preflush_reads_flushed{};
    std::atomic<u64> preflush_reads_in_flight{};
    std::atomic<u64> preflush_reads_in_flight_depth{};
    /// EDEN_NX2_EXP72: fence downloads with copies, numbered from 1 in commit order; landed counts
    /// the ones PopAsyncBuffers wrote to guest memory (pending_download_ranges[i] is landed + i + 1).
    std::atomic<u64> fence_downloads_landed{};
    std::mutex fence_landed_mutex;
    std::condition_variable fence_landed_cv;
    std::atomic<u64> fence_waits{};
    std::atomic<u64> fence_wait_ns{};
    std::atomic<u64> fence_wait_timeouts{};

public:
    /// EDEN_NX2_EXP72 (fable52_answer_bosswave 3): a guest read whose GPU-written bytes are all in
    /// a submitted fence download waits for that download to land instead of a new copy queued
    /// behind everything since (11-17 fences at the night market). No cache lock held.
    bool WaitFenceDownload(u64 sequence, std::chrono::milliseconds timeout) {
        const auto start = std::chrono::steady_clock::now();
        // FRITCHII_READ_RUNS=2: first the writer's submission alone (counter only).
        const u64 writer_ns =
            fence_downloads_landed.load(std::memory_order_acquire) < sequence ? WaitReadWriter()
                                                                               : 0;
        std::unique_lock lock{fence_landed_mutex};
        const bool done = fence_landed_cv.wait_for(lock, timeout, [&] {
            return fence_downloads_landed.load(std::memory_order_acquire) >= sequence;
        });
        lock.unlock();
        if (ReadRunStats::t_writer_answer.cut) [[unlikely]] {
            const u64 total = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count());
            ReadRunStats::t_writer_answer.rest_wait_ns = total > writer_ns ? total - writer_ns : 0;
            MeasureRunUp();
        }
        fence_waits.fetch_add(1, std::memory_order_relaxed);
        fence_wait_ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - start)
                                                     .count()),
                                std::memory_order_relaxed);
        if (!done) {
            fence_wait_timeouts.fetch_add(1, std::memory_order_relaxed);
        }
        return done;
    }

    /// FRITCHII_READ_RUNS=2 (counter only): the stamp a write mark carries.
    [[nodiscard]] u64 ReadWriteStamp() {
        if constexpr (!IS_OPENGL) {
            if (ReadRunStats::g_writer_ticks) [[unlikely]] {
                return runtime.CurrentTick();
            }
        }
        return 0;
    }

    /// FRITCHII_READ_RUNS=2 (counter only), on the reading thread before its host wait: waits for
    /// the submission of the newest write over the read's pages (GetFlushArea found it) and
    /// returns how long. The read's own wait follows on the same timeline, so the total is
    /// unchanged. Only a submitted tick is waited for (no flush is asked for here); 0 = no cut.
    /// FRITCHII_SUBMIT_TIMES (diagnostic), on the reading thread after its whole wait (so the
    /// look-ups add nothing to the wait's parts): what the card ran from the submission after the
    /// one known done at the read to the writer's, from the per-submission GPU timestamps.
    void MeasureRunUp() {
        if constexpr (!IS_OPENGL) {
            ReadRunStats::WriterAnswer& a = ReadRunStats::t_writer_answer;
            if (!EarlyCompute::g_submit_times || !a.cut || a.cut_tick == 0) {
                return;
            }
            const auto ahead = runtime.MeasureSubmissions(a.known_tick + 1, a.cut_tick);
            a.ahead_submissions = ahead.submissions;
            a.ahead_resolved = ahead.resolved;
            a.ahead_busy_ns = ahead.busy_ns;
            a.ahead_gap_ns = ahead.gap_ns;
            a.ahead_overlap_ns = ahead.overlap_ns;
            a.ahead_span_ns = ahead.span_ns;
        }
    }

    /// `side_tick` (FRITCHII_READ_SIDE_COPY): this scheduler tick instead of the area mark's.
    u64 WaitReadWriter(std::optional<u64> side_tick = std::nullopt) {
        if constexpr (!IS_OPENGL) {
            ReadRunStats::WriterAnswer& a = ReadRunStats::t_writer_answer;
            if (!ReadRunStats::g_writer_ticks || a.cut || (!side_tick && !a.area_found)) {
                return 0;
            }
            const u64 start = ReadRunStats::NowNs();
            // Older ticks of the same queue first, oldest first, so the reader's whole wait is
            // what it was: FRITCHII_FEEDER_LANE (dry) the feeder's D (and D with its upload on
            // the lane); FRITCHII_EXTENT_CBUF / _SHARED =3 the mark the proof leaves over the
            // read's bytes (dry_cover).
            struct Step {
                u64 tick;
                u64* ns;
                bool* unsubmitted;
            };
            std::array<Step, 3> steps{};
            size_t num_steps = 0;
            bool cover_unsubmitted = false;
            if (EarlyCompute::g_lane && !side_tick && a.area_mark.lane.judged) [[unlikely]] {
                steps[num_steps++] = {a.lane_copy_tick, &a.lane_copy_ns, &a.lane_unsubmitted};
                steps[num_steps++] = {a.lane_tick, &a.lane_ns, &a.lane_unsubmitted};
            }
            if (!side_tick && a.dry_cover_found && a.dry_cover_tick != 0) [[unlikely]] {
                steps[num_steps++] = {a.dry_cover_tick, &a.dry_cover_ns, &cover_unsubmitted};
            }
            std::sort(steps.begin(), steps.begin() + num_steps,
                      [](const Step& lhs, const Step& rhs) { return lhs.tick < rhs.tick; });
            for (size_t i = 0; i < num_steps; ++i) {
                const Step& step = steps[i];
                if (step.tick == 0 || step.tick >= a.write_tick) {
                    continue;
                }
                if (!runtime.WaitTickIfSubmitted(step.tick)) {
                    *step.unsubmitted = true;
                    continue;
                }
                *step.ns = ReadRunStats::NowNs() - start;
                if (step.ns == &a.dry_cover_ns) {
                    a.dry_cover_timed = true;
                }
            }
            if (!runtime.WaitTickIfSubmitted(side_tick ? *side_tick : a.write_tick)) {
                return 0;
            }
            a.cut = true;
            a.writer_wait_ns = ReadRunStats::NowNs() - start;
            a.cut_tick = side_tick ? *side_tick : a.write_tick;
            // The cover in the area mark's submission (or later: not before it): it ran with it
            if (!side_tick && a.dry_cover_found && a.dry_cover_tick >= a.write_tick &&
                !cover_unsubmitted) [[unlikely]] {
                a.dry_cover_ns = a.writer_wait_ns;
                a.dry_cover_timed = true;
            }
            return a.writer_wait_ns;
        }
        return 0;
    }

private:
    u64 preflush_resets{};
    bool download_cpu_read_ranges{};
    /// Keeps gpu_written_mirror.gpu_modified_pages (EXP86) at every Add, Subtract and Clear.
    MirroredRangeSet<DAddr> gpu_modified_ranges;
    std::deque<Common::RangeSet<DAddr>> committed_gpu_modified_ranges;

    // Async Buffers
    std::deque<std::optional<Async_Buffer>> async_buffers;
    std::deque<boost::container::small_vector<BufferCopy, 4>> pending_downloads;
    /// For each entry of pending_downloads: the ranges it still writes to guest memory when its
    /// fence pops. A write to a range after a download was taken removes it from that download
    /// (and every other pending one): its bytes are older than the write.
    std::deque<Common::RangeSet<DAddr>> pending_download_ranges;

    /// FRITCHII_LANDING_CHECK (boot; 0 off, 1 count and log, 2 also keep the guest's bytes):
    /// at each fence download's commit the guest bytes under its copies; at its landing, the bytes
    /// the guest changed since then (a write no tracking saw) that the landing would overwrite.
    /// A tester's Jabu-Jabu freezes (2026-11-17): game objects read as null or float bits
    /// (0xbf7fffff..., 0x80000000...) right after area loads, the game's assert, then a stopped
    /// thread; with FRITCHII_ONEF_SEM_E the game's sync point wait ends before the fence's
    /// downloads land, so memory it frees and reuses after the wait can take an old landing.
    struct LandingSnapshot {
        u64 committed_ns{};
        u64 landing_seq{};            ///< landing_check_seq at the commit
        std::vector<u64> copy_offsets; ///< Each copy's bytes in `bytes`
        std::vector<u8> bytes;
    };
    std::deque<LandingSnapshot> landing_snapshots; ///< One per pending_downloads entry, when on
    /// Pages a landing wrote, with landing_check_seq then: a later landing on them is skipped by
    /// the check (the bytes changed by a download, not by the guest). Under landing_check_mutex.
    std::unordered_map<u64, u64> landing_check_pages;
    u64 landing_check_seq{}; ///< Under landing_check_mutex
    std::mutex landing_check_mutex;
    struct LandingCheckTotals {
        u64 landings{};
        u64 bytes{};
        u64 changed_landings{};
        u64 changed_bytes{};
        u64 kept_bytes{};
        u64 skipped_bytes{}; ///< On pages an earlier landing wrote after the commit
        u64 last_log_ns{};
    } landing_check_totals; ///< The landing thread's (under the buffer cache lock)
    /// The fence pop in progress on this thread (its landing_check_seq), 0: none.
    static inline thread_local u64 t_landing_check_pop_seq = 0;
    /// FRITCHII_LANDING_CHECK's mode, read once.
    static int LandingCheckMode();
    /// A fence pop with a download begins (`begin`) or ends on this thread.
    void LandingCheckPop(bool begin);
    /// The snapshot of the fence download just committed (`copies`, device addresses).
    void LandingCheckCommit(std::span<const BufferCopy> copies);
    /// Before `index`'s copy lands [start, start + size) from `data` (the staging bytes, which
    /// mode 2 patches with the guest's changed bytes). Under the buffer cache lock.
    /// `cpu_written`: collected CPU writes the landing leaves alone (not counted).
    void LandingCheckPiece(const LandingSnapshot& snapshot, const BufferCopy& copy, size_t index,
                           DAddr start, u8* data, u64 size,
                           const Common::RangeSet<DAddr>* cpu_written);
    /// PopOneUnlocked: each unlocked_pieces entry's copy index (with the check on).
    std::vector<size_t> landing_check_piece_copies;
    /// Notes the pages a download landing wrote (any landing path).
    void LandingCheckNoteLanded(DAddr device_addr, u64 size);
    PendingCpuWrites pending_cpu_writes;
    std::optional<Async_Buffer> current_buffer;

    /// FRITCHII_FENCE_COPY_UNLOCKED (PopAsyncBuffersUnlocked): the fencing thread's own scratch,
    /// touched by it alone (with or without the lock).
    struct UnlockedPiece {
        DAddr addr;
        u64 size;
        const u8* data; ///< In the download's staging memory
    };
    std::vector<UnlockedPiece> unlocked_pieces;
    Common::RangeSet<DAddr> unlocked_cpu_written;
    Common::RangeSet<DAddr> unlocked_parts;
    Common::RangeSet<DAddr> unlocked_verify_cpu;
    Common::RangeSet<DAddr> unlocked_verify_parts;
    std::vector<u8> unlocked_verify_bytes;
    std::vector<Common::RangeSet<DAddr>> unlocked_trash_ranges;
    std::vector<boost::container::small_vector<BufferCopy, 4>> unlocked_trash_copies;
    void PopOneUnlocked(std::unique_lock<Common::SpinRecursiveMutex>& lock, int mode,
                        Common::LockCensus::FenceCopyBatch& batch, u64& held_since);
    void VerifyUnlockedLanding(DAddr landing_begin, DAddr landing_end,
                               const Common::RangeSet<DAddr>& live,
                               Common::LockCensus::FenceCopyBatch& batch);

    /// EDEN_NX2_EXP49: a DownloadMemoryForRead, its copies, and what is left of them to do once
    /// the host GPU has run them.
    struct ReadDownload {
        u64 tick{}; ///< The submission with the copies
        Async_Buffer staging{};
        /// src_offset: the device address copied; dst_offset: where it is in `staging`.
        boost::container::small_vector<BufferCopy, 4> copies;
        /// Still to write to guest memory: a newer value there (ClearReplacedDownload) takes its
        /// bytes out. A newer GPU write doesn't: until a download of its own lands, guest memory
        /// is to hold what a synchronous flush would have left.
        Common::RangeSet<DAddr> write_ranges;
        /// No longer GPU-modified once written: a newer GPU write (OutdateReadDownloads) or value
        /// in guest memory takes its bytes out.
        Common::RangeSet<DAddr> clean_ranges;
        /// FRITCHII_READ_SIDE_COPY (dry): the newest write over the copied bytes was in an
        /// already submitted scheduler tick, `side_tick`: a side copy could wait for it alone.
        bool side_eligible{};
        u64 side_tick{};
        /// FRITCHII_READ_SIDE_COPY >= 2: the side copy's value (0: none). With 2 (verify) its
        /// bytes go to `verify_staging`, compared with `staging` at the landing; with 3 they are
        /// the download (no main-queue copy was recorded).
        u64 side_value{};
        bool side_verify{};
        Async_Buffer verify_staging{};
        u64 id{}; ///< numbered from 1, to find it again after a wait without the lock
    };
    /// Oldest first, and written to guest memory in that order: none lands over newer bytes.
    std::deque<ReadDownload> read_downloads;
    /// Writes one read download to guest memory (WriteReadDownloads' body, and the side copy's
    /// landing out of order: read downloads never overlap, see RecordReadDownload's gaps).
    void LandReadDownload(ReadDownload& download);
    /// All the read downloads' clean_ranges (they don't overlap): GPU-modified bytes already on
    /// their way, which later reads wait for rather than download again.
    Common::RangeSet<DAddr> read_download_ranges;

    // FRITCHII_MEMSTAGE_MARKS_AHEAD (buffer_cache.h, "Marks ahead of the back").
    /// A GPU write a fill on GPUMem marked ahead of the back: what MarkWrittenBuffer still has to
    /// do for it when the back has played ring B's packet `index`.
    struct AheadWrite {
        u64 index;
        DAddr addr;
        u32 size;
        bool capped;
        bool ampere;
        /// FRITCHII_STORE_FOOTPRINT: the mark's footprint (id 0: none), claimed at the promotion
        ReadRunStats::FootprintRef footprint{};
    };
    static constexpr size_t AheadBound = 4096;
    /// In fill order (= packet order), under `mutex`.
    std::deque<AheadWrite> ahead_writes;
    /// The counted union of their ranges, under `mutex`.
    Common::OverlapRangeSet<DAddr> ahead_ranges;
    /// The oldest entry's index (~0: none), and the last packet the back played (taken or
    /// dropped). Written under `mutex` (ahead_played: by the back alone), read anywhere.
    std::atomic<u64> ahead_oldest{~u64{0}};
    std::atomic<u64> ahead_played{0};
    /// A packet the back dropped unplayed: its marks are dropped instead of promoted (stored by
    /// the back before ahead_played).
    std::atomic<u64> ahead_discarded{0};
    std::atomic<bool> marks_ahead{false};
    bool marks_ahead_wanted{};
    /// GPUMem's open draw fill: its packet's index (0: none), and whether the packet went.
    u64 ahead_fill_index{};
    bool ahead_fill_pushed{};
    u64 ahead_last_promoted{};
    /// Must be 0 (the "Memory stage marks ahead" line).
    enum class AheadCheck : u8 {
        Claim,        ///< a level function on GPUMem's own work over a page with a mark ahead
        Order,        ///< a promotion out of packet order
        NotEmpty,     ///< the list not empty at a ring's start or end
        RunDue,       ///< a mark due at a run's first entry
        Stale,        ///< a reader of the real sets found a due mark it could not promote
        ServedMark,   ///< MarkWrittenBuffer in a call GPUMem serves for the back
        OutsideFill,  ///< a mark on GPUMem's own work outside a draw fill
        PushMismatch, ///< the packet pushed is not the one the fill was keyed with
        Count,
    };
    struct AheadStats {
        std::atomic<u64> pushed{};
        std::atomic<u64> promoted_back{};
        std::atomic<u64> promoted_other{};
        std::atomic<u64> retracted{};
        std::atomic<u64> length_sum{};
        std::atomic<u64> length_samples{};
        std::atomic<u64> length_max{};
        std::atomic<u64> back_waits{};
        std::atomic<u64> back_wait_ns{};
        std::array<std::atomic<u64>, 3> levels{}; ///< AheadLevel
        std::array<std::atomic<u64>, static_cast<size_t>(AheadCheck::Count)> checks{};
    };
    mutable AheadStats ahead_stats;
    std::array<u64, 9> ahead_reported{};
    std::array<u64, static_cast<size_t>(AheadCheck::Count)> ahead_checks_reported{};
    u64 ahead_log_ns{};

    void PromoteAheadSlow(bool by_back);
    /// MarkWrittenBuffer on GPUMem's own fill: its steps that run now. False: today's path.
    bool MarkWrittenAhead(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync,
                          bool capped);
    /// Takes the marks of packet `index` off the list (not promoted).
    void RetractAhead(u64 index);
    /// The list's part of GpuWrittenHere: GPUMem's own work only (a served call and every other
    /// thread act at the back's position, where those writes are not made yet).
    [[nodiscard]] bool AheadHere(DAddr addr, u64 size) const {
        return !ahead_writes.empty() && MemoryStageLink::JoinNeedsTest() &&
               AheadIntersects(addr, size);
    }
    /// gpu_modified_ranges.ForEachInRange with the list's ranges merged in (AheadHere's rule).
    template <typename Func>
    void ForEachGpuWrittenHere(DAddr addr, u64 size, Func&& func);
    /// At a level function's top (section 3's "level" rows): counted when GPUMem's own work
    /// changes the real sets over a page with a mark ahead (must be 0).
    void NoteAheadLevelCall(DAddr addr, u64 size, const char* where) {
        if (!ahead_writes.empty()) [[unlikely]] {
            NoteAheadLevelCallSlow(addr, size, where);
        }
    }
    void NoteAheadLevelCallSlow(DAddr addr, u64 size, const char* where);
    void NoteAheadCheck(AheadCheck check, const char* where) const noexcept;
    /// The back's promotion try (MemoryStageLink::BackPromoteMarks): true when nothing is due.
    static bool TryPromoteAhead(void* self);
    void LogMarksAhead();

    std::deque<Async_Buffer> async_buffers_death_ring;
    u64 read_download_ids{}; ///< ReadDownload::id

    size_t immediate_buffer_capacity = 0;
    Common::ScratchBuffer<u8> immediate_buffer_alloc;
    std::vector<u8> guest_uint8_indices;
    /// Guest memory large uploads came from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_buffer_ranges;

    struct LRUItemParams {
        using ObjectType = BufferId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;
    VirtualRangeCache virtual_ranges;
    std::vector<MultiRangeSegment> graphics_segments;
    std::vector<MultiRangeSegment> compute_segments;
    /// EDEN_NX2_GPU_PROFILE: SynchronizeBuffer counts its uploads as compute storage uploads
    /// (set in BindHostComputeStorageBuffers).
    bool count_compute_sync = false;
    /// EDEN_NX2_EXP59 bits 4 and 8 while BindHostComputeStorageBuffers runs (0 otherwise):
    /// SynchronizeBuffer walks only the dirty tracker words (4), or walks all and checks (8).
    int compute_dirty_walk = 0;
    /// EXP59 & 8: synchronizes checked, and those where the dirty-word walk would have differed,
    /// since the last totals line (VerifyComputeDirtyWalk).
    struct DirtyWalkVerify {
        u64 checked = 0;
        u64 mismatches = 0;
        u64 cpu_pages = 0;
        u64 untracked_pages = 0;
        u64 total_checked = 0;
        u64 total_mismatches = 0;
        u32 logged = 0;
        std::chrono::steady_clock::time_point last_report{};
    } dirty_walk_verify;
    /// EDEN_NX2_STORAGE_SYNC_SCAN while a storage bind runs (BindHostGraphicsStorageBuffers, and
    /// BindHostComputeStorageBuffers without EXP59), 0 otherwise: 1 = SynchronizeBuffer walks the
    /// scanned windows only, 2 = it walks all after counting what the scan would leave out.
    int storage_sync_scan = 0;
    /// EDEN_NX2_STORAGE_SYNC_SCAN=2 totals (VerifyStorageSyncScan).
    DirtyWalkVerify storage_scan_verify;
    /// EDEN_NX2_STORAGE_SYNC_MEMO: a range SynchronizeBuffer left clean, and the dirty generation
    /// sum of its tracker managers then (MemoryTrackerBase::DirtyGenerationSum). size 0 = empty.
    struct StorageSyncMemo {
        DAddr device_addr{};
        u32 size{};
        u64 gen_sum{};
    };
    static constexpr u32 STORAGE_SYNC_MEMO_BITS = 9;
    std::array<StorageSyncMemo, size_t{1} << STORAGE_SYNC_MEMO_BITS> storage_sync_memo{};
    /// EDEN_NX2_STORAGE_SYNC_MEMO=2 totals: hits checked, hits whose range was not clean
    /// (cpu_pages / untracked_pages unused).
    DirtyWalkVerify storage_memo_verify;
    /// FRITCHII_FILL_GEOM_KEEP while BindHostGeometryBuffers runs (0 otherwise, and when hot pages
    /// or CleanWords are on: their walks may leave a range dirty, so no memo is made behind them).
    int geom_keep = 0;
    /// FRITCHII_FILL_GEOM_KEEP: per vertex slot, then the index buffer.
    std::array<StorageSyncMemo, NUM_VERTEX_BUFFERS + 1> geom_sync_memo{};
    /// FRITCHII_FILL_UPDATE_KEEP while UpdateGraphicsBuffers runs (0 otherwise, and with accurate
    /// GPU fences: FindBuffer waits for writes there).
    int update_keep = 0;
    /// FRITCHII_FILL_UPDATE_KEEP: what a slot's binding was resolved from, and the binding.
    struct GeomUpdateMemo {
        u64 epoch{}; ///< binding_epoch after the resolve; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        GPUVAddr begin{};
        u64 extent{}; ///< vertex: the end address; index: the size from count, first and format
        u32 enable{};
        Binding binding{};
    };
    std::array<GeomUpdateMemo, NUM_VERTEX_BUFFERS + 1> geom_update_memo{};
    u64 frame_tick = 0;

    /// EXP91 (fable52_answer_uniformkeeps stage 0): counts how many of today's slot rebinds
    /// (BindSmallUniformReusing: same bytes, same tick) a keep could have made without reading
    /// guest memory. Drawing thread only; nothing here changes what is bound.
    struct UniformKeepCount {
        static constexpr size_t RingSize = 128;
        static constexpr size_t NumStages = 5;
        static constexpr size_t NumSlots = 18;
        struct Range {
            DAddr begin{};
            DAddr end{};
        };
        /// The last RingSize cached write ranges; range number `seq` sits at seq % RingSize.
        std::array<Range, RingSize> ring{};
        u64 write_seq{};
        struct Slot {
            DAddr addr{};
            u32 size{};
            bool valid{};
            u64 seen_seq{}; ///< write_seq at the slot's last bind
        };
        std::array<std::array<Slot, NumSlots>, NumStages> slots{};
        /// Since the last log line. A slot rebind lands in exactly one of kept .. pending.
        struct Totals {
            u64 binds{};        ///< BindSmallUniformReusing calls
            u64 rebinds{};      ///< every rebind (slot and EXP17 range reuse)
            u64 slot_rebinds{}; ///< today's rebind of the slot's own copy
            u64 kept{};         ///< K: no noted range touched it since the slot's last bind
            u64 no_keep{};      ///< the slot was not streamed since counting (re)started
            u64 other_range{};  ///< the slot's last bind was another range
            u64 overflow{};     ///< more than RingSize ranges noted since the slot's last bind
            u64 overlap{};      ///< a range noted since then overlaps
            u64 pending{};      ///< MemoryManager holds unflushed cached writes at the bind
        };
        Totals totals{};
        u64 calls{};
        u64 last_frame{};
        u64 log_write_seq{}; ///< write_seq at the last log line
        std::chrono::steady_clock::time_point last_log{};
    };
    UniformKeepCount uniform_keep_count;

    /// EXP91 != 0: BindSmallUniformReusing as the stream branch calls it, plus the counting.
    bool BindSmallUniformCountingKeeps(size_t stage, u32 index, u32 binding_index,
                                       DAddr device_addr, u32 size, bool range_reuse);

    /// EXP92 (fable52_answer_uniformkeeps 3 and 4; idea from a ported change, part d): uniform
    /// keeps. Drawing thread only; the ring is UniformKeepCount's (tier 1), the page generations
    /// and watches Common::UniformKeepPages (tier 2).
    struct UniformKeeps {
        static constexpr size_t NumStages = UniformKeepCount::NumStages;
        static constexpr size_t NumSlots = UniformKeepCount::NumSlots;
        /// Every 256th kept bind runs the normal path as a check with EXP92 = 1.
        static constexpr u64 TripwireInterval = 256;
        static constexpr u32 MaxStaleLogs = 32;
        struct Keep {
            DAddr addr{};
            u32 size{};
            u32 copy{};        ///< the stream reuse copy the slot bound
            u32 copy_serial{}; ///< its serial then
            std::array<u32, 2> page_gens{}; ///< first and last page's generation before the read
            u32 global_gen{};
            u64 seen_seq{};    ///< ring sequence the range is known clean up to
            u64 translation{}; ///< device memory translation generation before the read
            u64 made_frame{};
            bool valid{};
        };
        std::array<std::array<Keep, NumSlots>, NumStages> slots{};
        int mode{}; ///< EXP92 as the last bind saw it (a change drops every keep)
        bool tripped{}; ///< a tripwire check found a stale keep: keeps off for the session
        /// Why a keepable-looking bind was not kept (one per bind that was not kept).
        enum Reason : u8 {
            NoKeep,       ///< nothing kept for the slot yet (or the last bind made none)
            OtherRange,   ///< the slot's last bind was another range
            Ineligible,   ///< native bank bind or run-ahead frontier set
            Tick,         ///< the slot's copy is gone or from an older command buffer
            Translation,  ///< the device memory's translation changed
            PageGen,      ///< a write was noted on one of the range's pages
            RingOverflow, ///< more than RingSize cached writes since
            RangeWritten, ///< a cached write overlapped the range
            Pending,      ///< MemoryManager holds cached writes not flushed yet
            NumReasons,
        };
        struct Totals {
            u64 binds{};    ///< small binds through the stream branch while EXP92 is on
            u64 kept{};     ///< bound from the keep (verified ones included)
            u64 verified{}; ///< keepable binds that ran the normal path as a check
            u64 stale{};    ///< of those, the normal path made a fresh copy
            u64 denied{};   ///< not kept: a page of the range is denied (no keep made)
            std::array<u64, NumReasons> not_kept{};
        };
        Totals totals{};
        u64 kept_count{}; ///< kept binds this session (the tripwire's interval)
        u64 stale_total{};
        u32 stale_logged{};
        u64 calls{};
        /// A check's copy of the kept bytes (the normal path may overwrite the copy).
        std::array<u8, 16384> verify_bytes{}; ///< UniformStreamReuse::MaxBytes (EXP116's most)
        std::array<u8, 16384> fresh_bytes{};
        u64 log_guest_stores{}; ///< UniformKeepPages guest_stores at the last log line
        u64 log_pages_denied{};
        std::chrono::steady_clock::time_point last_log{};
    };
    UniformKeeps uniform_keeps;

    /// EXP92 != 0: the stream branch's small uniform bind with keeps (see UniformKeeps).
    bool BindSmallUniformKeeping(size_t stage, u32 index, u32 binding_index, DAddr device_addr,
                                 u32 size, bool range_reuse, bool eligible, int mode);
    void LogUniformKeeps(int mode);

    /// EDEN_NX2_EXP18: resolved bindings are kept while the GPU mappings
    /// and the set of buffers stay the same. Bumped whenever a buffer is created, joined or
    /// deleted (ChangeRegister) and when the inline index buffer is replaced.
    u64 binding_epoch = 1;

    /// The last BindGraphicsUniformBuffer of a (stage, slot) and what it translated to, valid
    /// while that binding is the one it set (EXP18 & 1).
    struct UniformBindMemo {
        u64 epoch{}; ///< binding_epoch when stored; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        GPUVAddr gpu_addr{};
        DAddr device_addr{};
        u32 size{};
    };
    std::array<std::array<UniformBindMemo, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_bind_memo{};

    /// FRITCHII_ONEF_UBO_KEEP (fable_b 25.2 cut a): each slot's last streamed bind, its pages'
    /// write generations (Common::PriceGenSum) and whether the tracker held them clean.
    struct UniformKeepMemo {
        DAddr device_addr{};
        u64 sum{};
        u32 size{};
        bool tracked{};
        bool valid{};
    };
    std::array<std::array<UniformKeepMemo, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_keep_memo{};
    struct UniformKeepStats {
        u64 binds{};     ///< streamed binds looked at
        u64 provable{};  ///< same range, unflagged, tracked then and now
        u64 kept{};      ///< bound again without the read and the compare (=2)
        u64 stale{};     ///< =1: provable, yet the bind found other bytes (must be 0)
        u64 stale_session{};
        std::chrono::steady_clock::time_point since{};
    } uniform_keep_stats;
    /// The streamed bind's keep test (true: bound again, done) and =1's verify after it.
    bool UniformKeepBefore(size_t stage, u32 index, DAddr device_addr, u32 size, bool bank,
                           bool& provable);
    void UniformKeepLog(int mode);
    /// FRITCHII_S4_COUNT (dry): hands one graphics uniform bind to the S4 counter
    /// (fritchii_s4_count.h) with the tracker's state of its range and pages.
    void S4CountBind(size_t stage, u32 index, DAddr device_addr, u32 size, bool stream,
                     bool bank);

    /// FRITCHII_BIND_SKIP (live, default 0; common/fritchii_bind_skip.h): S4-lite. What the
    /// uniform bind's body did, for the memo of its slot (BindHostGraphicsUniformBufferBody's
    /// record form; the normal form records nothing).
    struct BindSkipOutcome {
        enum class Kind : u8 {
            None,   ///< not one the skip can stand in for (root table, keeps, mapped copy, ...)
            Stream, ///< streamed through the reuse slot (BindSmallUniformReusing / UBO_KEEP)
            Cached, ///< the cached path (and a native bank's check before it)
        };
        Kind kind{Kind::None};
        bool bank{};      ///< a native bank bind (its check ran or skipped itself)
        bool held{};      ///< the run-ahead hold kept the bank check
        bool synced{};    ///< cached: SynchronizeBuffer uploaded nothing
        int bank_mode{};  ///< EXP9 of the bank check that ran (0: the plain one)
    };
    /// Each slot's last bind the skip may stand in for: the range, what bound it, and the range's
    /// write generation read before that bind read anything. The range (device_addr, size) is
    /// kept for every bind: a memo is made only when a slot binds the same range again, so
    /// slots whose range moves every bind add no generation read.
    struct BindSkipMemo {
        const void* channel{};
        const void* gpu_memory{};
        DAddr device_addr{};
        u64 gen{};
        u64 entry{};        ///< E's list entry then (0: not on E)
        u64 binding_epoch{};
        u64 translation{};  ///< the device memory's translation generation (Map, Unmap)
        u64 hash{};         ///< verify modes, cached kind: the range's bytes then
        u64 touched_frame{};
        u32 size{};
        u32 declared{};     ///< the shader's size for the slot
        u32 serial{};       ///< stream kind: the reuse slot's copy serial after the bind
        u32 skip_cache_size{};
        u32 switch_generation{};
        u32 epoch{};
        BufferId buffer_id{};
        BindSkipOutcome::Kind kind{BindSkipOutcome::Kind::None};
        bool bank{};
        bool hashed{};
        bool touch_once{};  ///< EXP18 & 4 then: a second touch in a frame does nothing
    };
    std::array<std::array<BindSkipMemo, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        bind_skip_memo{};
    enum BindSkipReason : size_t {
        BindSkipInEntry,     ///< skipped: same list entry, nothing wrote the range
        BindSkipAcross,      ///< skipped: a later entry, pages tracked, nothing wrote the range
        BindSkipNoMemo,      ///< the slot's last bind left no memo (other path, before ready)
        BindSkipOtherRange,  ///< another range, buffer, size, channel or switch setting
        BindSkipWritten,     ///< the range's write generation moved
        BindSkipUntracked,   ///< a later entry, a page untrapped (CPU-modified or cached)
        BindSkipStep1Only,   ///< a later entry, tracked, but the switch is 1 or 4
        BindSkipSlotChanged, ///< stream: the reuse slot's copy is another now
        BindSkipStreamEpoch, ///< stream: the copy is from an older stream epoch (rebound later)
        BindSkipHeld,        ///< run-ahead frontier moved, or the skip stopped for the session
        BindSkipReasonCount,
    };
    struct BindSkipWindow {
        u64 binds{};
        std::array<u64, BindSkipReasonCount> reasons{};
        u64 skipped_bytes{};
        u64 skipped_stream{};
        u64 skipped_cached{};
        u64 skipped_bank{};
        u64 checked{};
        u64 differ{};
        u64 work{}; ///< verify, cached kind: the old path uploaded or found a bank page changed
        u64 decide_ticks{};
        u64 decide_samples{};
        u64 skip_ticks{};
        u64 skip_samples{};
        u64 old_ticks{};    ///< the old path of binds the skip would have stood in for
        u64 old_samples{};
        u64 record_ticks{}; ///< the bookkeeping a bind that is not skipped adds
        u64 record_samples{};
        u64 start_ticks{};
        std::chrono::steady_clock::time_point since{};
    };
    struct BindSkipState {
        int mode{};       ///< the switch's value (0 off)
        bool step2{};     ///< 2, 3, 5: across list entries too
        bool verify{};    ///< 3, 4
        bool dry{};       ///< 5
        bool broken{};    ///< a sampled check found other bytes: no skip for the session
        u32 epoch{1};
        u64 ready_frame{};
        u64 overflows{};
        u64 samples{};
        u64 differ_session{};
        BindSkipWindow w;
    } bind_skip;
    /// Native bank pages the bank check found changed (NoteNativeBankPageChanged), for the verify.
    u64 bind_skip_bank_changes{};
    static constexpr size_t BindSkipScratchBytes = 16384;
    std::vector<u8> bind_skip_scratch;
    /// Once a draw (SetUniformBuffersState): follows the live switch.
    void BindSkipPoll();
    void BindSkipSetMode(int mode);
    /// The switch's bind: the skip, or the body with its memo and the verify around it.
    void BindSkipUniform(size_t stage, u32 index, u32 binding_index, bool needs_bind);
    /// The body of BindHostGraphicsUniformBuffer past the root table's bank (`record`: fills
    /// `outcome`; false is today's code).
    template <bool record>
    void BindHostGraphicsUniformBufferBody(size_t stage, u32 index, u32 binding_index,
                                           bool needs_bind, BindSkipOutcome* outcome);
    /// The cached path's bind after SynchronizeBuffer (shared by the body and the skip).
    void BindUniformCachedTail(size_t stage, u32 index, u32 binding_index, bool needs_bind,
                               Buffer& buffer, u32 offset, u32 size);
    [[nodiscard]] u64 BindSkipHash(DAddr device_addr, u32 size);
    void BindSkipLog();

    /// FRITCHII_BIND_ADDR_MEMO: the last address translation of each vertex stream, the index
    /// buffer and each graphics uniform slot (Tegra::MemoryManager::GpuToCpuAddressMemo), and the
    /// counts for the "Bind address memo" line.
    std::array<Tegra::MemoryManager::TranslationMemo, NUM_VERTEX_BUFFERS> vertex_addr_memo{};
    Tegra::MemoryManager::TranslationMemo index_addr_memo{};
    std::array<std::array<Tegra::MemoryManager::TranslationMemo, NUM_GRAPHICS_UNIFORM_BUFFERS>,
               NUM_STAGES>
        uniform_addr_memo{};
    Tegra::MemoryManager::TranslationMemoStats bind_addr_stats{};
    u64 bind_addr_disabled_skips{}; ///< Disabled vertex streams not translated at all
    u64 bind_addr_calls{};
    std::chrono::steady_clock::time_point bind_addr_since{};
    /// The switch, and the counts' line every 10 s (the clock every 4,096th call).
    [[nodiscard]] bool BindAddrMemo();
    /// GpuToCpuAddress, or GpuToCpuAddressMemo through `memo` when `on`.
    [[nodiscard]] std::optional<DAddr> TranslateBind(bool on, GPUVAddr gpu_addr,
                                                     Tegra::MemoryManager::TranslationMemo& memo);

    /// One storage binding's last lookup when it resolved to a single buffer (no segments)
    /// (EXP18 & 2). Index NUM_STAGES is compute.
    struct StorageBindingMemo {
        u64 epoch{}; ///< binding_epoch when stored; 0 never matches
        u64 map_serial{};
        size_t as_id{};
        const void* channel{};
        DAddr device_addr{};
        GPUVAddr gpu_addr{};
        u32 size{};
        bool is_written{};
        BufferId buffer_id{};
    };
    std::array<std::array<StorageBindingMemo, NUM_STORAGE_BUFFERS>, NUM_STAGES + 1>
        storage_binding_memo{};

    /// FindBuffer and ResolveMultiRangeStorage for one storage binding, from its memo when
    /// nothing they depend on changed.
    void ResolveStorageBinding(size_t memo_stage, u32 index, Binding& binding, bool is_written,
                               std::vector<MultiRangeSegment>& pool);

    struct FindMemo {
        u64 page{~u64{0}};
        u64 deletions{};
        DAddr begin{};
        DAddr end{};
        BufferId buffer_id{};
        bool sparse_compatible{};
    };
    std::array<FindMemo, 256> find_memos{};
    u64 buffer_deletions = 0;

    /// EDEN_NX2_TABLE_POINTER_TRACE: last table entry and size bound per table pointer slot.
    std::unordered_map<u64, std::pair<GPUVAddr, u32>> traced_table_pointers;
    u64 total_used_memory = 0;
    u64 buffer_bytes = 0; ///< The buffers' own size (total_used_memory may be the device's)
    u64 minimum_memory = 0;
    u64 critical_memory = 0;
    BufferId inline_buffer_id;
#ifdef YUZU_LEGACY
    bool immediately_free = false;
#endif

    // Covers the device address space (MaxwellDeviceTraits::device_virtual_bits).
    std::array<BufferId, ((1ULL << 36) >> CACHING_PAGEBITS)> page_table;
    Common::ScratchBuffer<u8> tmp_buffer;

    /// A command list page (Nx2CopyListPages): the GPU writes noted to it and the last copy.
    using Nx2Staging = decltype(std::declval<Runtime&>().DownloadStagingBuffer(size_t{}, true));
    static constexpr size_t Nx2ListPageWords = (size_t{1} << Tegra::Nx2WrittenPageBits) / 4;
    struct Nx2ListPage {
        u64 writes{};      ///< GPU and host writes noted to the page
        u64 copy_writes{}; ///< `writes` when the copy was made: the copy is current while equal
        u64 copy_tick{};   ///< GPU tick that completes the copy
        /// `writes` when the front last flushed the page: guest memory holds it while equal
        u64 synced_writes{~u64{0}};
        u32 copy_begin{};  ///< Bytes of the page the copy holds (a buffer can end inside it)
        u32 copy_end{};
        u64 copy_writer{}; ///< What recorded the copy (Nx2CopyListPages)
        /// Written outside a compute dispatch or a DMA copy (a draw's storage writes) at least
        /// once: the front can't tell from what it sent which draws write the page, so it reads
        /// no copy. A draw's writes reach a pushbuffer fetch only after a host-level wait, so
        /// the page's lists read guest memory once the back ran the work before the last such
        /// wait and no write came since the front flushed the page (Nx2ReadListCopy).
        bool draw_written{};
        /// The page's 4-byte words GPU work wrote that guest memory does not hold: set by the GPU
        /// writes noted to the page (the whole range a dispatch or copy may write), cleared for
        /// the words the host writes whole and for every word once the front flushed the page.
        /// Once it did (`flushed`), guest memory holds the other words, so a list on none of
        /// these reads it directly (EXP40 & 1). Before, GPU work may have written the page before
        /// it became a command list page, unnoted.
        std::bitset<Nx2ListPageWords> gpu_words;
        bool flushed{};
        std::optional<Nx2Staging> copy;
    };
    /// Sets (`set`) or clears the words of `entry.gpu_words` within [begin, end) bytes of its
    /// page: every word the bytes touch when setting, only the words they cover when clearing.
    static void Nx2MarkListWords(Nx2ListPage& entry, u64 begin, u64 end, bool set);
    /// Whether GPU work wrote any word of [begin, end) bytes of the page that guest memory lacks.
    [[nodiscard]] static bool Nx2HasListWords(const Nx2ListPage& entry, u64 begin, u64 end);
    /// Fills `out` with the page's bytes from `offset` (`device_addr`) on: the words of
    /// `entry.gpu_words` from its copy, the others from guest memory once the page was flushed
    /// (EXP40 & 1); else, or for EDEN_NX2_LIST_VERIFY's read as before, all from the copy.
    void Nx2ReadListWords(const Nx2ListPage& entry, DAddr device_addr, u64 offset,
                          std::span<u8> out);
    std::mutex nx2_list_mutex; ///< nx2_list_pages: the split's front reads the copies
    std::unordered_map<DAddr, Nx2ListPage> nx2_list_pages;
    /// Copies replaced while the GPU may still write them: freed once it is past their tick.
    std::vector<std::pair<u64, Nx2Staging>> nx2_retired_list_copies;
    boost::container::small_vector<std::pair<BufferId, DAddr>, 8> nx2_list_copy_requests;
    bool nx2_collect_list_copies{};
};

} // namespace VideoCommon
