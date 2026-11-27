// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <cstring>
#include <cstdlib>
#include <span>
#include <vector>

#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/memory_stage_link.h"
#include "video_core/nx2_async_readback.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "common/bit_util.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vertex_repack.h"

#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/early_compute.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/texture_cache/util.h"

namespace Vulkan {
namespace {
VkBufferCopy MakeBufferCopy(const VideoCommon::BufferCopy& copy) {
    return VkBufferCopy{
        .srcOffset = copy.src_offset,
        .dstOffset = copy.dst_offset,
        .size = copy.size,
    };
}

VkIndexType IndexTypeFromNumElements(const Device& device, u32 num_elements) {
    if (num_elements <= 0xff && device.IsExtIndexTypeUint8Supported()) {
        return VK_INDEX_TYPE_UINT8_EXT;
    }
    if (num_elements <= 0xffff) {
        return VK_INDEX_TYPE_UINT16;
    }
    return VK_INDEX_TYPE_UINT32;
}

size_t BytesPerIndex(VkIndexType index_type) {
    switch (index_type) {
    case VK_INDEX_TYPE_UINT8_EXT:
        return 1;
    case VK_INDEX_TYPE_UINT16:
        return 2;
    case VK_INDEX_TYPE_UINT32:
        return 4;
    default:
        ASSERT_MSG(false, "Invalid index type={}", index_type);
        return 1;
    }
}

vk::Buffer CreateBuffer(const Device& device, const MemoryAllocator& memory_allocator, u64 size,
                        VkDeviceSize sparse_alignment, bool within_budget = true) {
    VkBufferUsageFlags flags =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        // The counter usage too: draws can take their vertex count from any buffer
        // (DrawIndirectByteCountEXT, nx2_indirect_grid.h).
        flags |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT |
                 VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
    }
    if (device.IsExtConditionalRendering()) {
        flags |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        flags |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    const VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = size,
        .usage = flags,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (!within_budget) {
        return memory_allocator.CreateBufferOverBudget(
            buffer_ci,
            sparse_alignment <= 1 && device.IsMoltenVK() ? MemoryUsage::Stream
                                                         : MemoryUsage::DeviceLocal,
            sparse_alignment);
    }
    if (sparse_alignment > 1) {
        return memory_allocator.CreateBuffer(buffer_ci, MemoryUsage::DeviceLocal, sparse_alignment);
    }
    return memory_allocator.CreateBuffer(
        buffer_ci, device.IsMoltenVK() ? MemoryUsage::Stream : MemoryUsage::DeviceLocal);
}

[[nodiscard]] bool IsOutOfMemory(const vk::Exception& exception) noexcept {
    return exception.GetResult() == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
           exception.GetResult() == VK_ERROR_OUT_OF_HOST_MEMORY;
}

/// A guest buffer the draw being set up can't go on without. With no memory for it (low VRAM, or
/// an integrated GPU's share of the RAM), the staging pool gives back what the GPU is done with,
/// then the buffer goes over the budget: slower than fitting, but the exception would end the
/// emulator on the GPU thread. At most three creates and no wait, and the result is a real buffer
/// or the exception, never an empty one. The staging pool is emptied at most once a second: while
/// the budget stays full every new buffer lands here, and emptying it each time would only make
/// the next uploads build their staging buffers again.
vk::Buffer CreateGuestBuffer(const Device& device, const MemoryAllocator& memory_allocator,
                             StagingBufferPool& staging_pool, u64 size,
                             VkDeviceSize sparse_alignment) {
    try {
        return CreateBuffer(device, memory_allocator, size, sparse_alignment);
    } catch (const vk::Exception& exception) {
        if (!IsOutOfMemory(exception)) {
            throw;
        }
    }
    using Clock = std::chrono::steady_clock;
    static std::atomic<s64> last_release_ms{-1000};
    const s64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           Clock::now().time_since_epoch())
                           .count();
    s64 last_ms = last_release_ms.load(std::memory_order_relaxed);
    if (now_ms - last_ms >= 1000 &&
        last_release_ms.compare_exchange_strong(last_ms, now_ms, std::memory_order_relaxed)) {
        const size_t freed = staging_pool.ReleaseIdleBuffers();
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
            LOG_WARNING(Render_Vulkan,
                        "No memory for a guest buffer of {} KiB; freed {} idle staging buffers",
                        size >> 10, freed);
        }
        try {
            return CreateBuffer(device, memory_allocator, size, sparse_alignment);
        } catch (const vk::Exception& exception) {
            if (!IsOutOfMemory(exception)) {
                throw;
            }
        }
    }
    try {
        return CreateBuffer(device, memory_allocator, size, sparse_alignment, false);
    } catch (const vk::Exception& exception) {
        LOG_CRITICAL(Render_Vulkan, "No memory for a guest buffer of {} KiB, even over the budget: {}",
                     size >> 10, exception.what());
        throw;
    }
}
} // Anonymous namespace

Buffer::Buffer(BufferCacheRuntime& runtime, VideoCommon::NullBufferParams null_params)
    : VideoCommon::BufferBase(null_params), clock{&runtime.clock}, tracker{4096} {
    if (runtime.device.HasNullDescriptor()) {
        return;
    }
    device = &runtime.device;
    buffer = RecycledBuffer{runtime.CreateNullBuffer()};
    is_null = true;
    if (device->IsBufferDeviceAddressSupported()) {
        device_address = device->GetLogical().GetBufferDeviceAddress(*buffer);
    }
}

Buffer::Buffer(BufferCacheRuntime& runtime, DAddr cpu_addr_, u64 size_bytes_,
               bool sparse_compatible_)
    : VideoCommon::BufferBase(cpu_addr_, size_bytes_), device{&runtime.device},
      clock{&runtime.clock},
      buffer{runtime.CreateCacheBuffer(SizeBytes(), sparse_compatible_)}, tracker{SizeBytes()} {
    sparse_compatible = sparse_compatible_;
    if (runtime.device.HasDebuggingToolAttached()) {
        buffer.SetObjectNameEXT(fmt::format("Buffer {:#x}", CpuAddr()).c_str());
    }
    if (device->IsBufferDeviceAddressSupported()) {
        device_address = device->GetLogical().GetBufferDeviceAddress(*buffer);
    }
}

void Buffer::MarkUsage(u64 offset, u64 size) noexcept {
    tracker.Track(offset, size);
    last_usage_tick = clock->Now();
    // FRITCHII_EARLY_COMPUTE (dry): every use, for the write-after-read test.
    if (VideoCommon::EarlyCompute::g_early) [[unlikely]] {
        VideoCommon::EarlyCompute::NoteUse(CpuAddr() + offset, size, last_usage_tick);
    }
}

VkBufferView Buffer::View(u32 offset, u32 size, VideoCore::Surface::PixelFormat format) {
    if (!device) {
        // Null buffer supported, return a null descriptor
        return VK_NULL_HANDLE;
    } else if (is_null) {
        // Null buffer not supported, adjust offset and size
        offset = 0;
        size = 0;
    }
    const auto it{std::ranges::find_if(views, [offset, size, format](const BufferView& view) {
        return offset == view.offset && size == view.size && format == view.format;
    })};
    if (it != views.end()) {
        return *it->handle;
    }
    views.push_back({
        .offset = offset,
        .size = size,
        .format = format,
        .handle = device->GetLogical().CreateBufferView({
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .buffer = *buffer,
            .format = MaxwellToVK::SurfaceFormat(*device, FormatType::Buffer, false, format).format,
            .offset = offset,
            .range = size,
        }),
    });
    return *views.back().handle;
}

class QuadIndexBuffer {
public:
    QuadIndexBuffer(const Device& device_, MemoryAllocator& memory_allocator_,
                    const SeamClock& clock_, MemStage& mem_stage_,
                    StagingBufferPool& staging_pool_)
        : device{device_}, memory_allocator{memory_allocator_}, clock{clock_},
          mem_stage{mem_stage_}, staging_pool{staging_pool_} {}

    virtual ~QuadIndexBuffer() = default;

    void UpdateBuffer(u32 num_indices_) {
        while (!retired_buffers.empty() && clock.IsFree(retired_buffers.front().first)) {
            retired_buffers.erase(retired_buffers.begin());
        }
        if (num_indices_ <= num_indices) {
            return;
        }

        // Commands recorded so far may still use the old buffer: it stays until the GPU is past
        // them. Waiting for the GPU here stalled the thread ~150 ms in the middle of a load.
        if (buffer) {
            retired_buffers.emplace_back(clock.Now(), std::move(buffer));
        }

        // Grown by at least half, so that a draw count creeping up doesn't redo this every time.
        num_indices = (std::max)(num_indices_, num_indices + num_indices / 2);
        index_type = IndexTypeFromNumElements(device, num_indices);

        const u32 num_quads = GetQuadsNum(num_indices);
        const u32 num_triangle_indices = num_quads * 6;
        const u32 num_first_offset_copies = 4;
        const size_t bytes_per_index = BytesPerIndex(index_type);
        const size_t size_bytes = num_triangle_indices * bytes_per_index * num_first_offset_copies;
        const VkBufferCreateInfo buffer_ci = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size_bytes,
            .usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        };
        buffer = memory_allocator.CreateBuffer(buffer_ci, MemoryUsage::DeviceLocal);
        if (device.HasDebuggingToolAttached()) {
            buffer.SetObjectNameEXT("Quad LUT");
        }

        const bool host_visible = buffer.IsHostVisible();
        const StagingBufferRef staging = [&] {
            if (host_visible) {
                return StagingBufferRef{};
            }
            return staging_pool.Request(size_bytes, MemoryUsage::Upload);
        }();

        u8* staging_data = host_visible ? buffer.Mapped().data() : staging.mapped_span.data();
        const size_t quad_size = bytes_per_index * 6;

        for (u32 first = 0; first < num_first_offset_copies; ++first) {
            for (u32 quad = 0; quad < num_quads; ++quad) {
                MakeAndUpdateIndices(staging_data, quad_size, quad, first);
                staging_data += quad_size;
            }
        }

        if (!host_visible) {
            mem_stage.PushQuadLut(staging.buffer, staging.offset, *buffer, size_bytes);
        } else {
            buffer.Flush();
        }
    }

    void BindBuffer(u32 first) {
        const VkIndexType index_type_ = index_type;
        const size_t sub_first_offset = static_cast<size_t>(first % 4) * GetQuadsNum(num_indices);
        const size_t offset =
            (sub_first_offset + GetQuadsNum(first)) * 6ULL * BytesPerIndex(index_type);
        // BindQuadIndexBuffer forgot the remembered index buffer before this.
        mem_stage.PushIndex(MemStage::Cmd::BindIndexAlways, 0, *buffer, offset, index_type_);
    }

protected:
    virtual u32 GetQuadsNum(u32 num_indices) const = 0;

    virtual void MakeAndUpdateIndices(u8* staging_data, size_t quad_size, u32 quad, u32 first) = 0;

    const Device& device;
    MemoryAllocator& memory_allocator;
    const SeamClock& clock;
    MemStage& mem_stage;
    StagingBufferPool& staging_pool;

    vk::Buffer buffer{};
    std::vector<std::pair<u64, vk::Buffer>> retired_buffers; ///< GPU tick that frees each one
    MemoryCommit memory_commit{};
    VkIndexType index_type{};
    u32 num_indices = 0;
};

class QuadArrayIndexBuffer : public QuadIndexBuffer {
public:
    QuadArrayIndexBuffer(const Device& device_, MemoryAllocator& memory_allocator_,
                         const SeamClock& clock_, MemStage& mem_stage_,
                         StagingBufferPool& staging_pool_)
        : QuadIndexBuffer(device_, memory_allocator_, clock_, mem_stage_, staging_pool_) {}

    ~QuadArrayIndexBuffer() = default;

private:
    u32 GetQuadsNum(u32 num_indices_) const override {
        return num_indices_ / 4;
    }

    template <typename T>
    static std::array<T, 6> MakeIndices(u32 quad, u32 first) {
        std::array<T, 6> indices{0, 1, 2, 0, 2, 3};
        for (T& index : indices) {
            index = static_cast<T>(first + index + quad * 4);
        }
        return indices;
    }

    void MakeAndUpdateIndices(u8* staging_data, size_t quad_size, u32 quad, u32 first) override {
        switch (index_type) {
        case VK_INDEX_TYPE_UINT8_EXT:
            std::memcpy(staging_data, MakeIndices<u8>(quad, first).data(), quad_size);
            break;
        case VK_INDEX_TYPE_UINT16:
            std::memcpy(staging_data, MakeIndices<u16>(quad, first).data(), quad_size);
            break;
        case VK_INDEX_TYPE_UINT32:
            std::memcpy(staging_data, MakeIndices<u32>(quad, first).data(), quad_size);
            break;
        default:
            ASSERT(false);
            break;
        }
    }
};

class QuadStripIndexBuffer : public QuadIndexBuffer {
public:
    QuadStripIndexBuffer(const Device& device_, MemoryAllocator& memory_allocator_,
                         const SeamClock& clock_, MemStage& mem_stage_,
                         StagingBufferPool& staging_pool_)
        : QuadIndexBuffer(device_, memory_allocator_, clock_, mem_stage_, staging_pool_) {}

    ~QuadStripIndexBuffer() = default;

private:
    u32 GetQuadsNum(u32 num_indices_) const override {
        return num_indices_ >= 4 ? (num_indices_ - 2) / 2 : 0;
    }

    template <typename T>
    static std::array<T, 6> MakeIndices(u32 quad, u32 first) {
        std::array<T, 6> indices{0, 3, 1, 0, 2, 3};
        for (T& index : indices) {
            index = static_cast<T>(first + index + quad * 2);
        }
        return indices;
    }

    void MakeAndUpdateIndices(u8* staging_data, size_t quad_size, u32 quad, u32 first) override {
        switch (index_type) {
        case VK_INDEX_TYPE_UINT8_EXT:
            std::memcpy(staging_data, MakeIndices<u8>(quad, first).data(), quad_size);
            break;
        case VK_INDEX_TYPE_UINT16:
            std::memcpy(staging_data, MakeIndices<u16>(quad, first).data(), quad_size);
            break;
        case VK_INDEX_TYPE_UINT32:
            std::memcpy(staging_data, MakeIndices<u32>(quad, first).data(), quad_size);
            break;
        default:
            ASSERT(false);
            break;
        }
    }
};

BufferCacheRuntime::BufferCacheRuntime(const Device& device_, MemoryAllocator& memory_allocator_,
                                       Scheduler& scheduler_, StagingBufferPool& staging_pool_,
                                       GuestDescriptorQueue& guest_descriptor_queue_,
                                       ComputePassDescriptorQueue& compute_pass_descriptor_queue,
                                       DescriptorPool& descriptor_pool, MemStage& mem_stage_,
                                       SeamClock& clock_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      staging_pool{staging_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      mem_stage{mem_stage_}, clock{clock_},
      quad_index_pass(device, scheduler, descriptor_pool, staging_pool,
                      compute_pass_descriptor_queue),
      multi_range_buffers(device_) {
    if (clock.Ticks() != nullptr) {
        // Packet ticks: the stamps of this cache's staging pool and recycler are packet numbers
        // too. Without FRITCHII_MEMSTAGE_SPLIT_ALLOC that pool is also the texture cache's: on
        // one thread its stamps are just as right (the GPU thread records right after it takes
        // them), but step 6 needs the split, so the GPU thread's own pool keeps scheduler ticks.
        staging_pool.UseClock(clock);
    }
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool enable_vertex_repack = VertexRepackEnabled();
    LOG_INFO(Render_Vulkan, "69-byte vertex layout repacking {} ({})",
             enable_vertex_repack ? "enabled" : "disabled",
             "portable Vulkan layout fallback");
    static_vertex_repack =
        StaticVertexRepackEnabled(device.GetMinVertexInputBindingStrideAlignment());
    LOG_INFO(Render_Vulkan, "Vertex repacking without dynamic vertex input {}",
             static_vertex_repack ? "enabled" : "disabled");
    if (enable_vertex_repack) {
        vertex_repack_pass = std::make_unique<VertexRepackPass>(
            device, scheduler, descriptor_pool, staging_pool, compute_pass_descriptor_queue);
    }
    limit_dynamic_storage_buffers = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                    driver_id == VK_DRIVER_ID_ARM_PROPRIETARY;
    if (limit_dynamic_storage_buffers) {
        max_dynamic_storage_buffers = device.GetMaxDescriptorSetStorageBuffersDynamic();
    }    
    if (device.SupportsUint8Indices()) {
        uint8_pass = std::make_unique<Uint8Pass>(device, scheduler, descriptor_pool, staging_pool,
                                                 compute_pass_descriptor_queue);
    }
    quad_array_index_buffer = std::make_shared<QuadArrayIndexBuffer>(
        device_, memory_allocator_, clock, mem_stage, staging_pool_);
    quad_strip_index_buffer = std::make_shared<QuadStripIndexBuffer>(
        device_, memory_allocator_, clock, mem_stage, staging_pool_);
    // The three conversion passes stay here; their Record half is the player's.
    mem_stage.Player().SetPasses(uint8_pass.get(), &quad_index_pass, vertex_repack_pass.get());
}

BufferCacheRuntime::~BufferCacheRuntime() {
    mem_stage.Player().SetPasses(nullptr, nullptr, nullptr);
}

u32 BufferCacheRuntime::SetVertexRepackLayout(const Tegra::Engines::Maxwell3D::Regs& regs,
                                            bool dynamic_input) {
    const u32 old_mask = vertex_repack_mask;
    const bool input_supported = VertexRepackInputSupported(dynamic_input, static_vertex_repack);
    vertex_repack_mask = vertex_repack_pass && input_supported ? VertexRepackMask(regs) : 0;
    if (vertex_repack_mask != old_mask) {
        LOG_DEBUG(Render_Vulkan, "Vertex repack selection: mask={:#x}, dynamic_input={}",
                  vertex_repack_mask, dynamic_input);
    }
    return old_mask ^ vertex_repack_mask;
}

RecycledBuffer BufferCacheRuntime::CreateCacheBuffer(u64 size, bool sparse_compatible) {
    const VkDeviceSize sparse_alignment = SparseAlignmentFor(sparse_compatible);
    if (sparse_alignment > 1) {
        // Bound by their memory location (multi-range sparse binds): never pooled.
        return RecycledBuffer{
            CreateGuestBuffer(device, memory_allocator, staging_pool, size, sparse_alignment)};
    }
    BufferRecycler& recycler = staging_pool.Recycler();
    u64 bytes = size;
    // EDEN_NX2_EXP104 (Fable's design 8.3, step 3b): the size class, powers of two from 64 KiB.
    // A pooled buffer is one the GPU finished with (BufferRecycler::Take); CreateBuffer clears
    // the Buffer's range before any use, as for a new one.
    if (BufferRecycler::Enabled()) {
        const u32 log2 = (std::max)(BufferRecycler::MinCacheLog2,
                                    static_cast<u32>(Common::Log2Ceil<u64>(size)));
        if (log2 <= BufferRecycler::MaxLog2) {
            u64 pooled_bytes = 0;
            if (vk::Buffer pooled =
                    recycler.Take(BufferRecycler::Family::Cache, log2, &pooled_bytes)) {
                return RecycledBuffer{std::move(pooled), &recycler, pooled_bytes};
            }
            bytes = u64{1} << log2;
        }
    }
    vk::Buffer created;
    if (bytes != size) {
        try {
            created = CreateBuffer(device, memory_allocator, bytes, 0);
        } catch (const vk::Exception& exception) {
            if (!IsOutOfMemory(exception)) {
                throw;
            }
            // No room for the size class: the exact size below.
            bytes = size;
        }
    }
    if (!created) {
        created = CreateGuestBuffer(device, memory_allocator, staging_pool, bytes, 0);
    }
    return RecycledBuffer{std::move(created), &recycler, bytes};
}

StagingBufferRef BufferCacheRuntime::UploadStagingBuffer(size_t size) {
    return staging_pool.Request(size, MemoryUsage::Upload, false,
                                VideoCommon::BufferChurnStats::StagingCaller::BufferUpload);
}

StagingBufferRef BufferCacheRuntime::DownloadStagingBuffer(size_t size, bool deferred) {
    return staging_pool.Request(size, MemoryUsage::Download, deferred,
                                VideoCommon::BufferChurnStats::StagingCaller::BufferDownload);
}

VkFormat BufferCacheRuntime::TexelBufferFormat(VideoCore::Surface::PixelFormat format) const {
    return MaxwellToVK::SurfaceFormat(device, FormatType::Buffer, false, format).format;
}

void BufferCacheRuntime::FreeDeferredStagingBuffer(StagingBufferRef& ref) {
    staging_pool.FreeDeferred(ref);
}

u64 BufferCacheRuntime::GetDeviceLocalMemory() const {
    return device.GetDeviceLocalMemory();
}

u64 BufferCacheRuntime::GetDeviceMemoryUsage() const {
    if (VideoCommon::MemoryStageLink::OnStageThread()) [[unlikely]] {
        // The memory stage's thread never asks the main allocator (spec 5.3): the back's last
        // answer, which it takes at least once a frame (TickFrame).
        return device.LastDeviceMemoryUsage();
    }
    return device.GetDeviceMemoryUsage();
}

bool BufferCacheRuntime::CanReportMemoryUsage() const {
    return device.CanReportMemoryUsage();
}

u32 BufferCacheRuntime::GetUniformBufferAlignment() const {
    return static_cast<u32>(device.GetUniformBufferAlignment());
}

u32 BufferCacheRuntime::GetStorageBufferAlignment() const {
    return static_cast<u32>(device.GetStorageBufferAlignment());
}

u64 BufferCacheRuntime::GetMaxStorageBufferRange() const {
    return device.GetMaxStorageBufferRange();
}

void BufferCacheRuntime::TickFrame(Common::SlotVector<Buffer>& slot_buffers) noexcept {
    for (auto it = slot_buffers.begin(); it != slot_buffers.end(); it++) {
        if (clock.IsFree(it->LastUsageTick())) {
            it->ResetUsageTracking();
        }
    }
}

u64 BufferCacheRuntime::CurrentTick() {
    return clock.Now();
}

bool BufferCacheRuntime::IsFree(u64 tick) {
    return clock.IsFree(tick);
}

void BufferCacheRuntime::Wait(u64 tick) {
    clock.Wait(tick);
}

namespace {
// EDEN_NX2_LIST_WAIT_TRACE=1: the front's waits for GPU-written command list copies, per writer
// (the dispatch's shader hash, 0 for a DMA copy): count, time, and how many submissions the GPU
// still had to finish when the wait began. Logged every 2 s.
class ListWaitTrace {
public:
    static ListWaitTrace* Get() {
        static const bool enabled = std::getenv("EDEN_NX2_LIST_WAIT_TRACE") != nullptr;
        static ListWaitTrace trace;
        return enabled ? &trace : nullptr;
    }

    void Note(u64 writer, u64 behind, bool unsubmitted, std::chrono::steady_clock::duration wait) {
        const u64 us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(wait).count());
        Entry& entry = entries[writer];
        ++entry.count;
        entry.us += us;
        entry.max_us = (std::max)(entry.max_us, us);
        entry.behind += behind;
        entry.unsubmitted += unsubmitted ? 1 : 0;
        const auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::seconds{2}) {
            return;
        }
        const double seconds = std::chrono::duration<double>(now - last).count();
        last = now;
        std::vector<std::pair<u64, Entry>> sorted(entries.begin(), entries.end());
        std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
        u64 total = 0;
        u64 count = 0;
        for (const auto& [key, value] : sorted) {
            total += value.us;
            count += value.count;
        }
        LOG_INFO(Render_Vulkan, "List waits: {:.1f}/s, {:.2f} ms/s", count / seconds,
                 total / 1000.0 / seconds);
        for (size_t i = 0; i < (std::min<size_t>)(sorted.size(), 8); ++i) {
            const auto& [key, value] = sorted[i];
            LOG_INFO(Render_Vulkan,
                     "  writer {:016x}: {:.1f}/s, {:.2f} ms/s, max {} us, {:.1f} submissions "
                     "behind, {} unsubmitted",
                     key, value.count / seconds, value.us / 1000.0 / seconds, value.max_us,
                     static_cast<double>(value.behind) / value.count, value.unsubmitted);
        }
        entries.clear();
    }

private:
    struct Entry {
        u64 count{};
        u64 us{};
        u64 max_us{};
        u64 behind{};
        u64 unsubmitted{};
    };
    std::unordered_map<u64, Entry> entries;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
} // Anonymous namespace

// `stamp` is the clock's (a packet number with FRITCHII_MEMSTAGE_TICKS): it is turned into the
// scheduler tick it was recorded in once, here; the rest waits on that tick as before.
void BufferCacheRuntime::WaitSubmittedTick(u64 stamp, u64 writer) {
    if (clock.IsFree(stamp)) {
        return;
    }
    if (VideoCommon::MemoryStageLink::InCall() || MemStage::OneFollowerFill()) [[unlikely]] {
        // A sync the memory stage's thread must never reach (spec 5.7): the back serves it.
        // One follower: E's fill sends it to R the same way.
        if (VideoCommon::MemoryStageLink::InCall()) {
            VideoCommon::MemoryStageLink::Note(
                VideoCommon::MemoryStageLink::Check::FrontWaitOnStage, "WaitSubmittedTick");
        }
        MemStage::OnBack([&] { WaitSubmittedTick(stamp, writer); });
        return;
    }
    auto& master = scheduler.GetMasterSemaphore();
    const u64 tick = clock.TickOf(stamp);
    // The split's front waits for the GPU: the back runs meanwhile what the front sent, also the
    // entries since the front's last publish, which it does not see yet (fewer than 256 words;
    // ported).
    if (auto& split = VideoCommon::GpuSplit::Get();
        split.Active() && !VideoCommon::GpuSplit::OnBackThread()) {
        split.Publish();
    }
    if (ListWaitTrace* const trace = ListWaitTrace::Get()) [[unlikely]] {
        master.Refresh();
        const u64 known = master.KnownGpuTick();
        const bool unsubmitted = !scheduler.IsSubmitted(tick);
        const auto start = std::chrono::steady_clock::now();
        WaitSubmittedTickImpl(tick);
        trace->Note(writer, tick > known ? tick - known : 0, unsubmitted,
                    std::chrono::steady_clock::now() - start);
        return;
    }
    WaitSubmittedTickImpl(tick);
}

void BufferCacheRuntime::WaitSubmittedTickImpl(u64 tick) {
    auto& master = scheduler.GetMasterSemaphore();
    if (master.IsFree(tick)) {
        return;
    }
    // The copy's submission may be deferred (Scheduler::DeferFlush): the split's front asks the
    // back for it rather than waiting for the back's next submission. If the back has stopped,
    // nothing else uses the scheduler and Wait submits it.
    if (!scheduler.IsSubmitted(tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread()) {
        VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::ListSubmitWait};
        if (!scheduler.WaitSubmitted(tick)) {
            clock.WaitTick(tick);
            return;
        }
    }
    VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::ListGpuWait};
    master.Wait(tick);
}

void BufferCacheRuntime::Finish() {
    clock.Finish();
}

u64 BufferCacheRuntime::SubmitForHostRead() {
    return clock.SubmitForHostRead();
}

void BufferCacheRuntime::WaitForHostRead(u64 stamp) {
    if (clock.IsFree(stamp)) {
        return;
    }
    if (VideoCommon::MemoryStageLink::InCall() || MemStage::OneFollowerFill()) [[unlikely]] {
        if (VideoCommon::MemoryStageLink::InCall()) {
            VideoCommon::MemoryStageLink::Note(
                VideoCommon::MemoryStageLink::Check::FrontWaitOnStage, "WaitForHostRead");
        }
        MemStage::OnBack([&] { WaitForHostRead(stamp); });
        return;
    }
    auto& master = scheduler.GetMasterSemaphore();
    const u64 tick = clock.TickOf(stamp);
    namespace AsyncReadback = VideoCore::AsyncReadback;
    const u64 start = AsyncReadback::Now();
    // SubmitForHostRead flushed it, so it is handed to the worker: wait without flushing, which
    // only the recording thread may do. Should a tick not be submitted yet, ask the split's back
    // for it, as WaitSubmittedTickImpl does.
    if (!scheduler.IsSubmitted(tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread()) {
        if (!scheduler.WaitSubmitted(tick)) {
            clock.WaitTick(tick);
            AsyncReadback::Add(AsyncReadback::g_counters.to_submitted_ns,
                               AsyncReadback::Now() - start);
            return;
        }
    }
    const u64 submitted = AsyncReadback::Now();
    AsyncReadback::Add(AsyncReadback::g_counters.to_submitted_ns, submitted - start);
    master.Wait(tick);
    AsyncReadback::Add(AsyncReadback::g_counters.on_card_ns, AsyncReadback::Now() - submitted);
}

BufferCacheRuntime::ReadTicks BufferCacheRuntime::ReadWriterTicks(u64 stamp) {
    auto& master = scheduler.GetMasterSemaphore();
    master.Refresh();
    return ReadTicks{
        .write = clock.TickOf(stamp),
        .current = master.CurrentTick(),
        .known = master.KnownGpuTick(),
    };
}

BufferCacheRuntime::AheadTimes BufferCacheRuntime::MeasureSubmissions(u64 first, u64 last) {
    const SubmitTimes::Span span =
        scheduler.GetSubmitTimes().Measure(first, last, scheduler.GetMasterSemaphore());
    return AheadTimes{
        .submissions = span.submissions,
        .resolved = span.resolved,
        .busy_ns = span.busy_ns,
        .gap_ns = span.gap_ns,
        .overlap_ns = span.overlap_ns,
        .span_ns = span.span_ns,
    };
}

bool BufferCacheRuntime::SideCopyReady() {
    if (!device.HasSideQueue() || !device.HasTimelineSemaphore()) {
        return false;
    }
    std::call_once(side_queue_once, [this] {
        try {
            side_queue = std::make_unique<SideQueue>(device, scheduler.GetMasterSemaphore());
        } catch (const vk::Exception& error) {
            LOG_ERROR(Render_Vulkan, "FRITCHII_READ_SIDE_COPY: no side queue ({})", error.what());
        }
    });
    return side_queue != nullptr;
}

u64 BufferCacheRuntime::SideCopy(u64 main_tick, std::span<const SideCopyRegion> regions) {
    if (!SideCopyReady()) {
        return 0;
    }
    boost::container::small_vector<SideQueue::Copy, 8> copies;
    for (const SideCopyRegion& region : regions) {
        copies.push_back(SideQueue::Copy{
            .src = region.src,
            .dst = region.dst,
            .region =
                VkBufferCopy{
                    .srcOffset = region.src_offset,
                    .dstOffset = region.dst_offset,
                    .size = region.size,
                },
        });
    }
    return side_queue->Submit(main_tick, copies);
}

bool BufferCacheRuntime::SideDone(u64 value) {
    return side_queue == nullptr || side_queue->IsDone(value);
}

BufferCacheRuntime::SideQueueStats BufferCacheRuntime::TakeSideQueueStats() {
    SideQueueStats stats{};
    if (side_queue != nullptr) {
        const SideQueue::Stats side = side_queue->TakeStats();
        stats.raised = side.raised;
        stats.slow_waits = side.slow_waits;
        stats.not_queued = side.not_queued;
    }
    const MasterSemaphore::SideStats main = scheduler.GetMasterSemaphore().TakeSideStats();
    stats.main_waits = main.waits;
    stats.unsubmitted = main.unsubmitted;
    return stats;
}

void BufferCacheRuntime::WaitSide(u64 value) {
    if (side_queue != nullptr) {
        side_queue->Wait(value);
    }
}

bool BufferCacheRuntime::WaitTickIfSubmitted(u64 tick) {
    if (!scheduler.IsSubmitted(tick)) {
        return false;
    }
    auto& master = scheduler.GetMasterSemaphore();
    if (!master.IsFree(tick)) {
        master.Wait(tick);
    }
    return true;
}

bool BufferCacheRuntime::IsFreeNow(u64 stamp) {
    if (clock.IsFree(stamp)) {
        return true;
    }
    if (!scheduler.IsSubmitted(clock.TickOf(stamp))) {
        return false;
    }
    scheduler.GetMasterSemaphore().Refresh();
    return clock.IsFree(stamp);
}

bool BufferCacheRuntime::DeletedBufferFree(u64 stamp) {
    // FRITCHII_MEMSTAGE_TICKS: a deleted buffer also waits for the packet it was deleted in
    // (every packet that used it is at or before that one). With 0 the frames alone, as before.
    return clock.Ticks() == nullptr || clock.IsFree(stamp);
}

void BufferCacheRuntime::NoteHeldBuffers(u64 newly_held) {
    if (SeamTicks* const ticks = clock.Ticks()) {
        ticks->NoteHeld(newly_held);
    }
}

// "Used" is the range tracker: Buffer::MarkUsage tracks the range and stamps the buffer, and
// TickFrame resets the tracker only once that stamp is free. With FRITCHII_MEMSTAGE_TICKS the
// stamp is a packet number, free once the GPU thread played that packet and its tick completed,
// so a range a packet the GPU thread has not played yet uses still counts as used, and the
// upload is not moved ahead of that use (spec 7.2, the third user that needs more than a swap).
bool BufferCacheRuntime::CanReorderUpload(Buffer& buffer,
                                          std::span<const VideoCommon::BufferCopy> copies) {
    if (Settings::values.disable_buffer_reorder) {
        return false;
    }
    bool can_use_upload_cmdbuf =
        std::ranges::all_of(copies, [&](const VideoCommon::BufferCopy& copy) {
            return !buffer.IsRegionUsed(copy.dst_offset, copy.size);
        });
    if (!can_use_upload_cmdbuf) {
        // Only work whose packet and GPU submission have finished used the buffer: nothing in
        // flight reads the regions, so the upload can stay in UploadAhead instead of breaking
        // the open render pass. clock.IsFree() is safe from both R and E2 (SeamTicks).
        const u64 last_tick = buffer.LastUsageTick();
        can_use_upload_cmdbuf = last_tick != 0 && clock.IsFree(last_tick);
    }
    // Stamp the uploaded regions regardless of path so later uploads in the same frame stay
    // ordered behind this one.
    for (const VideoCommon::BufferCopy& copy : copies) {
        buffer.MarkUsage(copy.dst_offset, copy.size);
    }
    return can_use_upload_cmdbuf;
}

void BufferCacheRuntime::CopyBuffer(VkBuffer dst_buffer, VkBuffer src_buffer,
                                    std::span<const VideoCommon::BufferCopy> copies, bool barrier,
                                    bool can_reorder_upload) {
    if (dst_buffer == VK_NULL_HANDLE || src_buffer == VK_NULL_HANDLE) {
        return;
    }
    // Measuring a popular game, this number never exceeds the specified size once data is warmed up
    boost::container::small_vector<VkBufferCopy, 8> vk_copies(copies.size());
    std::ranges::transform(copies, vk_copies.begin(), MakeBufferCopy);
    const std::span<const VkBufferCopy> copy_span{vk_copies.data(), vk_copies.size()};
    if (src_buffer == staging_pool.StreamBuf() && can_reorder_upload) {
        mem_stage.PushCopy(MemStage::Cmd::UploadAhead, 0, src_buffer, dst_buffer, copy_span);
        return;
    }
    // EDEN_NX2_EXP62 (live, 0 = off): ported batched uploads. OoT records ~600
    // inline uploads a frame, each with two barriers; in-order ones share one command and one
    // pair of barriers (the original test: 55.9 -> 57.8 fps, GPU 8.2 -> 7.55 ms a frame).
    if (barrier && VideoCore::Nx2Debug::Experiment(62) != 0 && src_buffer == staging_pool.StreamBuf() &&
        !vk_copies.empty()) {
        // Only in-order stream uploads share barriers. Dedicated staging buffers retain their
        // normal recording path and tick-based reuse.
        mem_stage.PushCopy(MemStage::Cmd::UploadBatched, MemStage::OutsideRenderPass, src_buffer,
                           dst_buffer, copy_span);
        return;
    }
    mem_stage.PushCopy(MemStage::Cmd::Copy,
                       MemStage::OutsideRenderPass | (barrier ? MemStage::Barrier : 0),
                       src_buffer, dst_buffer, copy_span);
}

void BufferCacheRuntime::PreCopyBarrier() {
    mem_stage.Push(MemStage::Cmd::PreCopyBarrier, MemStage::OutsideRenderPass);
}

void BufferCacheRuntime::PostCopyBarrier() {
    mem_stage.Push(MemStage::Cmd::PostCopyBarrier, MemStage::OutsideRenderPass);
}

void BufferCacheRuntime::ClearBuffer(VkBuffer dest_buffer, u32 offset, size_t size, u32 value) {
    if (dest_buffer == VK_NULL_HANDLE) {
        return;
    }
    mem_stage.PushClear(dest_buffer, offset, size, value);
}

bool BufferCacheRuntime::BindMultiRangeStorageBuffer(u64 key, bool is_written) {
    if (multi_range_sources.empty() || multi_range_total == 0) {
        return false;
    }
    // A sync: Get can take the submit mutex, bind sparse memory and wait for a fence.
    // On the memory stage's thread the back serves it (MemStage::OnBack).
    MultiRangeRef ref{};
    MemStage::OnBack([&] {
        MemStage::SyncScope sync{&mem_stage, MemStage::Sync::MultiRange};
        ref = multi_range_buffers.Get(device, clock, memory_allocator, key, multi_range_sources,
                                      multi_range_total);
    });
    if (ref.handle == VK_NULL_HANDLE) {
        return false;
    }
    if (is_written && !ref.sparse) {
        return false;
    }
    if (ref.needs_gather) {
        PreCopyBarrier();
        VkDeviceSize dst_offset = 0;
        for (const MultiRangeSource& source : multi_range_sources) {
            const std::array<VideoCommon::BufferCopy, 1> copy{VideoCommon::BufferCopy{
                .src_offset = u64(source.offset),
                .dst_offset = u64(dst_offset),
                .size = size_t(source.size),
            }};
            CopyBuffer(ref.handle, source.handle, copy, false);
            dst_offset += source.size;
        }
        PostCopyBarrier();
        multi_range_buffers.MarkGathered(key);
    }
    mem_stage.PushDescriptor(UpdateDescriptorQueue::MakeBufferEntry(
        DescriptorForm(), ref.handle, ref.address, 0, ref.size));
    return true;
}

void BufferCacheRuntime::BindIndexBuffer(PrimitiveTopology topology, IndexFormat index_format,
                                         u32 base_vertex, u32 num_indices, VkBuffer buffer,
                                         u32 offset, [[maybe_unused]] u32 size) {
    VkIndexType vk_index_type = MaxwellToVK::IndexFormat(index_format);
    VkDeviceSize vk_offset = offset;
    VkBuffer vk_buffer = buffer;
    if (topology == PrimitiveTopology::Quads || topology == PrimitiveTopology::QuadStrip) {
        vk_index_type = VK_INDEX_TYPE_UINT32;
        const ComputePassRun run =
            quad_index_pass.Prepare(index_format, num_indices, base_vertex, buffer, offset,
                                    topology == PrimitiveTopology::QuadStrip);
        mem_stage.PushConvert(MemStage::Pass::QuadIndexed, run);
        vk_buffer = run.dst;
        vk_offset = run.dst_offset;
    } else if (vk_index_type == VK_INDEX_TYPE_UINT8_EXT && !device.IsExtIndexTypeUint8Supported()) {
        vk_index_type = VK_INDEX_TYPE_UINT16;
        if (uint8_pass) {
            const ComputePassRun run = uint8_pass->Prepare(num_indices, buffer, offset);
            mem_stage.PushConvert(MemStage::Pass::Uint8, run);
            vk_buffer = run.dst;
            vk_offset = run.dst_offset;
        } else if (device.GetDriverID() == VK_DRIVER_ID_QUALCOMM_PROPRIETARY) {
            ReserveNullBuffer();
            vk_buffer = *null_buffer;
            vk_offset = 0;
        }
    }
    if (vk_buffer == VK_NULL_HANDLE) {
        // Vulkan doesn't support null index buffers. Replace it with our own null buffer.
        ReserveNullBuffer();
        vk_buffer = *null_buffer;
    }
    // The player skips the rebind when the same index buffer is still bound.
    mem_stage.PushIndex(MemStage::Cmd::BindIndex, 0, vk_buffer, vk_offset, vk_index_type);
}

void BufferCacheRuntime::BindHostUint8IndexBuffer(std::span<const u8> indices) {
    const auto staging = staging_pool.Request(indices.size() * sizeof(u16), MemoryUsage::Upload);
    u16* const widened = reinterpret_cast<u16*>(staging.mapped_span.data());
    std::ranges::transform(indices, widened, [](u8 index) -> u16 { return index == 0xff ? 0xffff : index; });
    mem_stage.PushIndex(MemStage::Cmd::BindIndexAlways, 0, staging.buffer, staging.offset,
                        VK_INDEX_TYPE_UINT16);
}

void BufferCacheRuntime::BindQuadIndexBuffer(PrimitiveTopology topology, u32 first, u32 count) {
    if (count == 0) {
        ReserveNullBuffer();
        mem_stage.PushIndex(MemStage::Cmd::BindIndexAlways, 0, *null_buffer, 0,
                            VK_INDEX_TYPE_UINT32);
        return;
    }

    if (topology == PrimitiveTopology::Quads) {
        quad_array_index_buffer->UpdateBuffer(first + count);
        quad_array_index_buffer->BindBuffer(first);
    } else if (topology == PrimitiveTopology::QuadStrip) {
        quad_strip_index_buffer->UpdateBuffer(first + count);
        quad_strip_index_buffer->BindBuffer(first);
    } else {
        // Nothing bound, but the remembered index buffer is forgotten all the same.
        mem_stage.PushIndex(MemStage::Cmd::BindIndexAlways, MemStage::ForgetOnly, VK_NULL_HANDLE,
                            0, VK_INDEX_TYPE_UINT32);
    }
}

void BufferCacheRuntime::BindVertexBuffer(u32 index, VkBuffer buffer, u32 offset, u32 size, u32 stride) {
    if (index >= device.GetMaxVertexInputBindings()) {
        return;
    }
    if (buffer != VK_NULL_HANDLE && size != 0 && IsVertexRepacked(index)) {
        const auto repacked =
            vertex_repack_pass->Prepare(buffer, offset, size, u64{offset} + size);
        if (!repacked) {
            // The binding's remembered state is forgotten, nothing is bound.
            mem_stage.PushVertex(MemStage::ForgetOnly, index, VK_NULL_HANDLE, 0, 0, 0);
            return;
        }
        mem_stage.PushConvert(MemStage::Pass::VertexRepack, *repacked);
        buffer = repacked->dst;
        offset = static_cast<u32>(repacked->dst_offset);
        size = static_cast<u32>((u64{size} + 68) / 69 * 72);
        stride = 72;
    }
    if (device.IsExtExtendedDynamicStateSupported()) {
        mem_stage.PushVertex(MemStage::DynamicState, index, buffer, offset, size, stride);
    } else {
        if (!device.HasNullDescriptor() && buffer == VK_NULL_HANDLE) {
            ReserveNullBuffer();
            buffer = *null_buffer;
            offset = 0;
        }
        mem_stage.PushVertex(0, index, buffer, offset, size, stride);
    }
}

void BufferCacheRuntime::ForgetVertexBinding(u32 index) {
    if (index >= device.GetMaxVertexInputBindings()) {
        return;
    }
    mem_stage.PushVertex(MemStage::ForgetOnly, index, VK_NULL_HANDLE, 0, 0, 0);
}

void BufferCacheRuntime::BindVertexBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    boost::container::static_vector<VkBuffer, VideoCommon::NUM_VERTEX_BUFFERS> buffer_handles(bindings.buffers.size());
    for (u32 i = 0; i < bindings.buffers.size(); ++i) {
        auto handle = bindings.buffers[i]->Handle();
        if (handle == VK_NULL_HANDLE) {
            bindings.offsets[i] = 0;
            bindings.sizes[i] = VK_WHOLE_SIZE;
            if (!device.HasNullDescriptor()) {
                ReserveNullBuffer();
                handle = *null_buffer;
            }
        }
        buffer_handles[i] = handle;
        const u32 binding = bindings.min_index + i;
        if (handle != VK_NULL_HANDLE && IsVertexRepacked(binding) && bindings.strides[i] == 69 &&
            bindings.sizes[i] > 0) {
            const auto repacked = vertex_repack_pass->Prepare(
                handle, static_cast<u32>(bindings.offsets[i]), static_cast<u32>(bindings.sizes[i]),
                bindings.buffers[i]->SizeBytes());
            if (!repacked) {
                continue;
            }
            mem_stage.PushConvert(MemStage::Pass::VertexRepack, *repacked);
            buffer_handles[i] = repacked->dst;
            bindings.offsets[i] = repacked->dst_offset;
            bindings.sizes[i] = (bindings.sizes[i] + 68) / 69 * 72;
            bindings.strides[i] = 72;
            static bool reported = false;
            if (!reported) {
                LOG_INFO(Render_Vulkan, "GPU vertex repacking active: 69 -> 72 bytes");
                reported = true;
            }
        }
    }
    const u32 device_max = device.GetMaxVertexInputBindings();
    const u32 min_binding = (std::min)(bindings.min_index, device_max);
    const u32 max_binding = (std::min)(bindings.max_index, device_max);
    const u32 binding_count = max_binding - min_binding;
    if (binding_count == 0) {
        return;
    }
    // min_binding is bindings.min_index here (binding_count is 0 otherwise). The player skips the
    // rebind when the same buffers are still bound in this command buffer.
    mem_stage.PushVertices(
        static_cast<u8>(
            (device.IsExtExtendedDynamicStateSupported() ? MemStage::DynamicState : 0) |
            (bindings.clean ? MemStage::KeepSize : 0)),
        bindings.min_index,
        std::span<const VkBuffer>{buffer_handles.data(), binding_count},
        std::span<const VkDeviceSize>{bindings.offsets.data(), binding_count},
        std::span<const VkDeviceSize>{bindings.sizes.data(), binding_count},
        std::span<const VkDeviceSize>{bindings.strides.data(), binding_count});
}

void BufferCacheRuntime::BindTransformFeedbackBuffer(u32 index, VkBuffer buffer, u32 offset,
                                                     u32 size) {
    if (!device.IsExtTransformFeedbackSupported()) {
        // Already logged in the rasterizer
        return;
    }
    if (buffer == VK_NULL_HANDLE) {
        // Vulkan doesn't support null transform feedback buffers.
        // Replace it with our own null buffer.
        ReserveNullBuffer();
        buffer = *null_buffer;
        offset = 0;
        size = 0;
    }
    mem_stage.PushXfb(index, buffer, offset, size);
}

void BufferCacheRuntime::BindTransformFeedbackBuffers(VideoCommon::HostBindings<Buffer>& bindings) {
    if (!device.IsExtTransformFeedbackSupported()) {
        // Already logged in the rasterizer
        return;
    }
    boost::container::static_vector<VkBuffer, VideoCommon::NUM_VERTEX_BUFFERS> buffer_handles(bindings.buffers.size());
    for (u32 i = 0; i < bindings.buffers.size(); ++i) {
        auto handle = bindings.buffers[i]->Handle();
        if (handle == VK_NULL_HANDLE) {
            ReserveNullBuffer();
            handle = *null_buffer;
            bindings.offsets[i] = 0;
            bindings.sizes[i] = 0;
        }
        buffer_handles[i] = handle;
    }
    const size_t count = buffer_handles.size();
    mem_stage.PushXfbs(std::span<const VkBuffer>{buffer_handles.data(), count},
                       std::span<const VkDeviceSize>{bindings.offsets.data(), count},
                       std::span<const VkDeviceSize>{bindings.sizes.data(), count});
}

void BufferCacheRuntime::ReserveNullBuffer() {
    if (!null_buffer) {
        null_buffer = CreateNullBuffer();
    }
}

vk::Buffer BufferCacheRuntime::CreateNullBuffer() {
    VkBufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = 4,
        .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        create_info.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        create_info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    vk::Buffer ret = memory_allocator.CreateBuffer(create_info, MemoryUsage::DeviceLocal);
    if (device.HasDebuggingToolAttached()) {
        ret.SetObjectNameEXT("Null buffer");
    }

    mem_stage.PushNullInit(*ret);

    return ret;
}

} // namespace Vulkan
