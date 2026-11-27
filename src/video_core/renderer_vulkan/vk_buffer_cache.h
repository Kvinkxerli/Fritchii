// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <limits>
#include <utility>

#include <boost/container/small_vector.hpp>

#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/buffer_cache/usage_tracker.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/vk_buffer_recycler.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_mem_stage.h"
#include "video_core/renderer_vulkan/vk_multi_range_buffer.h"
#include "video_core/renderer_vulkan/vk_seam_clock.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/renderer_vulkan/vk_side_queue.h"

namespace Vulkan {

class Device;
class DescriptorPool;
class Scheduler;
struct HostVertexBinding;

class BufferCacheRuntime;

class Buffer : public VideoCommon::BufferBase {
public:
    explicit Buffer(BufferCacheRuntime&, VideoCommon::NullBufferParams null_params);
    explicit Buffer(BufferCacheRuntime& runtime, VAddr cpu_addr_, u64 size_bytes_,
                    bool sparse_compatible_);

    [[nodiscard]] VkBufferView View(u32 offset, u32 size, VideoCore::Surface::PixelFormat format);

    [[nodiscard]] VkBuffer Handle() const noexcept {
        return *buffer;
    }

    [[nodiscard]] VkDeviceAddress DeviceAddress() const noexcept {
        return device_address;
    }

    [[nodiscard]] bool IsSparseCompatible() const noexcept {
        return sparse_compatible;
    }

    [[nodiscard]] vk::MemoryLocation Location() const noexcept {
        return buffer.Location();
    }

    [[nodiscard]] bool IsRegionUsed(u64 offset, u64 size) const noexcept {
        return tracker.IsUsed(offset, size);
    }

    void MarkUsage(u64 offset, u64 size) noexcept;

    void ResetUsageTracking() noexcept {
        tracker.Reset();
    }

    [[nodiscard]] u64 LastUsageTick() const noexcept {
        return last_usage_tick;
    }

    [[nodiscard]] std::span<const u8> Mapped() const noexcept {
        return buffer.IsHostCoherent() ? buffer.Mapped() : std::span<const u8>{};
    }

    operator VkBuffer() const noexcept {
        return *buffer;
    }

private:
    struct BufferView {
        u32 offset;
        u32 size;
        VideoCore::Surface::PixelFormat format;
        vk::BufferView handle;
    };

    const Device* device{};
    const SeamClock* clock{};
    /// EXP104: may be larger than SizeBytes() (its size class), and goes back to the pool when
    /// this Buffer is destroyed. Declared before the views, so they are destroyed first.
    RecycledBuffer buffer;
    std::vector<BufferView> views;
    VideoCommon::UsageTracker tracker;
    VkDeviceAddress device_address{};
    u64 last_usage_tick{};
    bool is_null{};
    bool sparse_compatible{};
};

class QuadArrayIndexBuffer;
class QuadStripIndexBuffer;

class BufferCacheRuntime {
    friend Buffer;

    using PrimitiveTopology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    using IndexFormat = Tegra::Engines::Maxwell3D::Regs::IndexFormat;

public:
    explicit BufferCacheRuntime(const Device& device_, MemoryAllocator& memory_manager_,
                                Scheduler& scheduler_, StagingBufferPool& staging_pool_,
                                GuestDescriptorQueue& guest_descriptor_queue,
                                ComputePassDescriptorQueue& compute_pass_descriptor_queue,
                                DescriptorPool& descriptor_pool, MemStage& mem_stage,
                                SeamClock& clock);
    ~BufferCacheRuntime();

    /// The back's way to the seam and its player (memory stage spec 5.4: never from GPUMem).
    [[nodiscard]] MemStage& Seam() noexcept {
        MemPlayer::CheckBack("runtime.Seam()");
        return mem_stage;
    }

    [[nodiscard]] SeamClock& Clock() noexcept {
        return clock;
    }

    [[nodiscard]] MemPlayer& Player() noexcept {
        MemPlayer::CheckBack("runtime.Player()");
        return mem_stage.Player();
    }

    /// The fence commit's fill bracket (FenceManager): its downloads are played as it ends.
    [[nodiscard]] MemStage::FillPlay FenceCommitBracket() {
        return MemStage::FillPlay{mem_stage, MemStage::Site::Fence};
    }

    void TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept;

    /// Returns the mask of bindings whose repacked state changed.
    u32 SetVertexRepackLayout(const Tegra::Engines::Maxwell3D::Regs& regs, bool dynamic_input);

    bool IsVertexRepacked(u32 binding) const {
        return (vertex_repack_mask & (1U << binding)) != 0;
    }

    bool AnyVertexRepacked() const {
        return vertex_repack_mask != 0;
    }

    [[nodiscard]] u32 RepackMask() const noexcept {
        return vertex_repack_mask;
    }

    /// The clock's stamp (SeamClock: a scheduler tick, or a packet number with
    /// FRITCHII_MEMSTAGE_TICKS). IsFree, Wait, WaitSubmittedTick, WaitForHostRead and IsFreeNow
    /// take these, as does what SubmitForHostRead returns.
    u64 CurrentTick();

    bool IsFree(u64 tick);

    void Wait(u64 tick);

    /// Changes exactly when the scheduler's tick does (one thread): the stream uniform reuse's
    /// "the stream copy is still there" test (vk_mem_stage.cpp, "Packet ticks").
    [[nodiscard]] u64 StreamEpoch() const noexcept {
        return clock.Epoch();
    }

    /// Whether a buffer deleted at `stamp` may be destroyed once its frames have passed.
    [[nodiscard]] bool DeletedBufferFree(u64 stamp);
    void NoteHeldBuffers(u64 newly_held);

    /// Waits for a submitted tick without flushing: safe on threads other than the GPU thread.
    void WaitSubmittedTick(u64 tick, u64 writer = 0);
    /// `tick`: a scheduler tick (WaitSubmittedTick translated the stamp).
    void WaitSubmittedTickImpl(u64 tick);

    void Finish();

    /// EDEN_NX2_EXP49: Finish without the wait. Makes the copies so far visible to the host,
    /// submits, and returns the tick to wait for (WaitForHostRead).
    u64 SubmitForHostRead();

    /// Waits for a SubmitForHostRead tick, on any thread.
    void WaitForHostRead(u64 tick);

    /// Whether the host GPU completed `tick`, asking it when the tick is submitted but not known
    /// to be done.
    [[nodiscard]] bool IsFreeNow(u64 tick);

    /// FRITCHII_READ_RUNS=2 (counter only), on any thread: the scheduler tick `stamp` is recorded
    /// in (the open one when it is not published yet), the scheduler's current tick, and the host
    /// GPU's completed tick (refreshed).
    struct ReadTicks {
        u64 write;
        u64 current;
        u64 known;
    };
    [[nodiscard]] ReadTicks ReadWriterTicks(u64 stamp);

    /// FRITCHII_READ_RUNS=2 (counter only), on any thread: false when scheduler tick `tick` is not
    /// submitted (nothing done: no flush is asked for); else waits until the host GPU ran it.
    bool WaitTickIfSubmitted(u64 tick);

    /// FRITCHII_SUBMIT_TIMES (diagnostic), on any thread: the GPU's start and end of the
    /// scheduler ticks [first, last] (those it ran), as busy time and gaps.
    struct AheadTimes {
        u32 submissions;
        u32 resolved;
        u64 busy_ns;
        u64 gap_ns;
        u64 overlap_ns;
        u64 span_ns;
    };
    [[nodiscard]] AheadTimes MeasureSubmissions(u64 first, u64 last);

    /// FRITCHII_READ_SIDE_COPY >= 2 (vk_side_queue.h): whether the side queue is there (made on
    /// the first ask; the device must have been made with it).
    [[nodiscard]] bool SideCopyReady();

    /// One region of a side copy: from a buffer cache buffer into a download staging buffer.
    struct SideCopyRegion {
        VkBuffer src;
        VkBuffer dst;
        u64 src_offset;
        u64 dst_offset;
        u64 size;
    };
    /// Submits `regions` on the side queue after the main queue ran scheduler tick `main_tick`;
    /// returns the side value (0: not submitted). Every later main submission waits for it.
    [[nodiscard]] u64 SideCopy(u64 main_tick, std::span<const SideCopyRegion> regions);
    [[nodiscard]] bool SideDone(u64 value);
    void WaitSide(u64 value);
    /// FRITCHII_READ_SIDE_COPY: the side queue's and the main queue's counts since the last call
    /// (raised waits, slow host waits, main submissions that waited, must-be-0 unsubmitted).
    struct SideQueueStats {
        u64 raised;
        u64 slow_waits;
        u64 not_queued;
        u64 main_waits;
        u64 unsubmitted;
    };
    [[nodiscard]] SideQueueStats TakeSideQueueStats();

    u64 GetDeviceLocalMemory() const;

    u64 GetDeviceMemoryUsage() const;

    bool CanReportMemoryUsage() const;

    u32 GetUniformBufferAlignment() const;

    u32 GetStorageBufferAlignment() const;

    /// Largest range a storage buffer binding may have
    u64 GetMaxStorageBufferRange() const;

    [[nodiscard]] StagingBufferRef UploadStagingBuffer(size_t size);

    [[nodiscard]] StagingBufferRef DownloadStagingBuffer(size_t size, bool deferred = false);

    bool CanReorderUpload(Buffer& buffer, std::span<const VideoCommon::BufferCopy> copies);

    void FreeDeferredStagingBuffer(StagingBufferRef& ref);

    void PreCopyBarrier();

    void CopyBuffer(VkBuffer dst_buffer, VkBuffer src_buffer,
                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                    bool can_reorder_upload = false);

    void PostCopyBarrier();

    void ClearBuffer(VkBuffer dest_buffer, u32 offset, size_t size, u32 value);

    void BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format, u32 num_indices,
                         u32 base_vertex, VkBuffer buffer, u32 offset, u32 size);

    void BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count);

    bool WidensUint8IndicesOnHost() const {
        return device.IsMoltenVK() && !device.IsExtIndexTypeUint8Supported();
    }

    void BindHostUint8IndexBuffer(std::span<const u8> indices);

    void BindVertexBuffer(u32 index, VkBuffer buffer, u32 offset, u32 size, u32 stride);

    void BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    /// A vertex stream left without a binding: the player forgets the buffer it remembers for
    /// it, so EnsureVertexBindings never binds it again in a new command buffer (the buffer
    /// cache may have deleted it by then).
    void ForgetVertexBinding(u32 index);

    void BindTransformFeedbackBuffer(u32 index, VkBuffer buffer, u32 offset, u32 size);

    void BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings);

    /// This and the two members below are the hooks of VideoCommon::StreamUniformReusable
    /// (buffer_cache/uniform_stream_reuse.h).
    struct StreamUniformRef {
        VkBuffer buffer;
        VkDeviceAddress device_address; // base of `buffer`, 0 without buffer device address
        u32 offset;
    };

    std::span<u8> BindMappedUniformBuffer([[maybe_unused]] size_t stage,
                                          [[maybe_unused]] u32 binding_index,
                                          u32 size) {
        const StagingBufferRef ref = staging_pool.Request(
            size, MemoryUsage::Upload, false, VideoCommon::BufferChurnStats::StagingCaller::UniformStream);
        last_stream_uniform = StreamUniformRef{
            .buffer = ref.buffer,
            .device_address = ref.device_address,
            .offset = static_cast<u32>(ref.offset),
        };
        RebindStreamUniform(last_stream_uniform, size);
        return ref.mapped_span;
    }

    /// The allocation of the latest BindMappedUniformBuffer call.
    [[nodiscard]] StreamUniformRef LastStreamUniformRef() const noexcept {
        return last_stream_uniform;
    }

    /// Adds `ref` to the descriptor payload exactly as BindMappedUniformBuffer did when it made
    /// the allocation (it is the very same call). The caller guarantees the allocation is still
    /// live: made on the scheduler tick that is current now, so the staging pool has not recycled
    /// it and the GPU has not even started on it.
    void RebindStreamUniform(const StreamUniformRef& ref, u32 size) {
        mem_stage.PushDescriptor(UpdateDescriptorQueue::MakeBufferEntry(
            DescriptorForm(), ref.buffer, ref.device_address, ref.offset, size));
    }

    [[nodiscard]] VkDeviceSize SparseAlignmentFor(bool sparse_compatible) const noexcept {
        if (!sparse_compatible || !multi_range_buffers.use_sparse) {
            return 0;
        }
        return multi_range_buffers.block_size;
    }

    [[nodiscard]] bool PrefersSparseSources() const noexcept {
        return multi_range_buffers.use_sparse;
    }

    void ResetMultiRange() noexcept {
        multi_range_sources.clear();
        multi_range_total = 0;
    }

    void PushMultiRangeSource(const Buffer& buffer, u32 offset, u32 size) {
        const vk::MemoryLocation location = buffer.Location();
        multi_range_sources.push_back(MultiRangeSource{
            .handle = buffer.Handle(),
            .memory = location.memory,
            .memory_offset = location.offset,
            .offset = offset,
            .size = size,
            .write_tick = buffer.getWriteTick(),
            .memory_type = location.memory_type,
        });
        multi_range_total += size;
    }

    bool BindMultiRangeStorageBuffer(u64 key, bool is_written);

    void InvalidateMultiRange(u64 key) {
        multi_range_buffers.Invalidate(key);
    }

    void OnBufferDeleted(const Buffer& buffer) {
        multi_range_buffers.DropOwner(clock, buffer.Handle());
    }

    void BindUniformBuffer(const Buffer& buffer, u32 offset, u32 size) {
        BindBuffer(buffer, offset, size);
    }

    void BindStorageBuffer(const Buffer& buffer, u32 offset, u32 size,
                           [[maybe_unused]] bool is_written) {
        BindBuffer(buffer, offset, size);
    }

    void BindTextureBuffer(Buffer& buffer, u32 offset, u32 size,
                           VideoCore::Surface::PixelFormat format) {
        // An unmapped texture buffer has no format; its view is the null buffer's anyway.
        if (format == VideoCore::Surface::PixelFormat::Invalid) {
            format = VideoCore::Surface::PixelFormat::R8_UNORM;
        }
        const VkBufferView view = buffer.View(offset, size, format);
        mem_stage.PushDescriptor(UpdateDescriptorQueue::MakeTexelEntry(
            DescriptorForm(), view, buffer.DeviceAddress(), offset, size,
            TexelBufferFormat(format)), true);
    }

    bool ShouldLimitDynamicStorageBuffers() const {
        return limit_dynamic_storage_buffers;
    }

    u32 GetMaxDynamicStorageBuffers() const {
        return max_dynamic_storage_buffers;
    }

private:
    void BindBuffer(const Buffer& buffer, u32 offset, u32 size) {
        const VkBuffer handle = buffer.Handle();
        const bool descriptor_buffer = DescriptorForm();
        if (handle == VK_NULL_HANDLE) {
            mem_stage.PushDescriptor(UpdateDescriptorQueue::MakeBufferEntry(
                descriptor_buffer, handle, 0, 0, VK_WHOLE_SIZE));
        } else {
            mem_stage.PushDescriptor(UpdateDescriptorQueue::MakeBufferEntry(
                descriptor_buffer, handle, buffer.DeviceAddress(), offset, size));
        }
    }

    /// The form of the descriptor entries pushed now: a held fill's pipeline's (SEAM = 2 fills
    /// before Acquire), else the queue's.
    [[nodiscard]] bool DescriptorForm() const noexcept {
        return mem_stage.DescriptorForm(guest_descriptor_queue.UsesDescriptorBuffer());
    }

    VkFormat TexelBufferFormat(VideoCore::Surface::PixelFormat format) const;

    void ReserveNullBuffer();
    vk::Buffer CreateNullBuffer();

    /// A buffer cache buffer of at least `size` bytes: with EXP104, rounded up to its size class
    /// and taken from the staging pool's BufferRecycler when one is free there.
    RecycledBuffer CreateCacheBuffer(u64 size, bool sparse_compatible);

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;
    StagingBufferPool& staging_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    MemStage& mem_stage;
    SeamClock& clock;

    /// FRITCHII_READ_SIDE_COPY >= 2: made on the first SideCopyReady.
    std::unique_ptr<SideQueue> side_queue;
    std::once_flag side_queue_once;

    std::shared_ptr<QuadArrayIndexBuffer> quad_array_index_buffer;
    std::shared_ptr<QuadStripIndexBuffer> quad_strip_index_buffer;

    vk::Buffer null_buffer;

    std::unique_ptr<Uint8Pass> uint8_pass;
    std::unique_ptr<VertexRepackPass> vertex_repack_pass;
    u32 vertex_repack_mask{};
    bool static_vertex_repack{};
    QuadIndexedPass quad_index_pass;

    MultiRangeBufferCache multi_range_buffers;
    boost::container::small_vector<MultiRangeSource, 16> multi_range_sources;
    VkDeviceSize multi_range_total{};

    StreamUniformRef last_stream_uniform{};

    bool limit_dynamic_storage_buffers = false;
    u32 max_dynamic_storage_buffers = (std::numeric_limits<u32>::max)();
};

struct BufferCacheParams {
    using Runtime = Vulkan::BufferCacheRuntime;
    using Buffer = Vulkan::Buffer;
    using Async_Buffer = Vulkan::StagingBufferRef;
    using MemoryTracker = VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager>;

    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using BufferCache = VideoCommon::BufferCache<BufferCacheParams>;

} // namespace Vulkan
