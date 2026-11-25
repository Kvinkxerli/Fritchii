// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <dxgi1_4.h>
#endif

#include <cstring>
#include <iterator>
#include <string>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/nx2_hitch.h"
#include "common/common_types.h"
#include "common/host_video_memory.h"
#include "common/literals.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include <ranges>
#include "video_core/memory_stage_link.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/vulkan_common/unbacked_fallback.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/settings.h"
#include "video_core/small_card.h"

namespace Vulkan {
namespace {

// Helpers translating MemoryUsage to flags/usage

    [[maybe_unused]] VkMemoryPropertyFlags MemoryUsagePropertyFlags(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::DeviceLocal:
                return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            case MemoryUsage::Upload:
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            case MemoryUsage::Download:
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            case MemoryUsage::Stream:
                return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        ASSERT_MSG(false, "Invalid memory usage={}", usage);
        return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    [[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
        if (usage == MemoryUsage::Download) {
            return VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                    : VkMemoryPropertyFlagBits{};
    }

    [[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::Upload:
            case MemoryUsage::Stream:
                return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            case MemoryUsage::Download:
                return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            case MemoryUsage::DeviceLocal:
                return {};
        }
        return {};
    }

    [[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::DeviceLocal:
            case MemoryUsage::Stream:
                return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            case MemoryUsage::Upload:
            case MemoryUsage::Download:
                return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        }
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }

    // FRITCHII_VRAM_PACK (live, default 2 = auto: on with the small card profile; 0 off, 1 on):
    // buffers and commits take the best-fitting hole in a block (VMA's MIN_MEMORY strategy, as
    // images always do) instead of VMA's default, which cuts them from a hole of the next larger
    // size class and so leaves the big holes split for the next large buffer. Only where in a
    // block VMA places them changes; the search is CPU time only (no new device memory).
    [[nodiscard]] VmaAllocationCreateFlags PackStrategy() {
        const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_vram_pack,
                                                           "FRITCHII_VRAM_PACK", 2);
        const bool on = value == 1 || (value == 2 && VideoCore::SmallCard::Active());
        return on ? VmaAllocationCreateFlags{VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT}
                  : VmaAllocationCreateFlags{0};
    }
} // namespace

//MemoryCommit is now VMA-backed
MemoryCommit::MemoryCommit(VmaAllocator alloc, VmaAllocation a,
                            const VmaAllocationInfo &info) noexcept
        : allocator{alloc}, allocation{a}, memory{info.deviceMemory},
            offset{info.offset}, size{info.size}, mapped_ptr{info.pMappedData} {
    // Log GPU memory allocation
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(memory),
            static_cast<u64>(size),
            0  // Memory property flags (not easily available from VMA)
        );
    }
}

MemoryCommit::~MemoryCommit() { Release(); }

MemoryCommit::MemoryCommit(MemoryCommit &&rhs) noexcept
        : allocator{std::exchange(rhs.allocator, nullptr)},
            allocation{std::exchange(rhs.allocation, nullptr)},
            memory{std::exchange(rhs.memory, VK_NULL_HANDLE)},
            offset{std::exchange(rhs.offset, 0)},
            size{std::exchange(rhs.size, 0)},
            mapped_ptr{std::exchange(rhs.mapped_ptr, nullptr)} {}

MemoryCommit &MemoryCommit::operator=(MemoryCommit &&rhs) noexcept {
    if (this != &rhs) {
        Release();
        allocator = std::exchange(rhs.allocator, nullptr);
        allocation = std::exchange(rhs.allocation, nullptr);
        memory = std::exchange(rhs.memory, VK_NULL_HANDLE);
        offset = std::exchange(rhs.offset, 0);
        size = std::exchange(rhs.size, 0);
        mapped_ptr = std::exchange(rhs.mapped_ptr, nullptr);
    }
    return *this;
}

std::span<u8> MemoryCommit::Map()
{
    if (!allocation) return {};
    if (!mapped_ptr) {
        if (vmaMapMemory(allocator, allocation, &mapped_ptr) != VK_SUCCESS) return {};
    }
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                (std::numeric_limits<size_t>::max)()));
    return std::span<u8>{static_cast<u8 *>(mapped_ptr), n};
}

std::span<const u8> MemoryCommit::Map() const
{
    if (!allocation) return {};
    if (!mapped_ptr) {
        void *p = nullptr;
        if (vmaMapMemory(allocator, allocation, &p) != VK_SUCCESS) return {};
        const_cast<MemoryCommit *>(this)->mapped_ptr = p;
    }
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                (std::numeric_limits<size_t>::max)()));
    return std::span<const u8>{static_cast<const u8 *>(mapped_ptr), n};
}

void MemoryCommit::Unmap()
{
    if (allocation && mapped_ptr) {
        vmaUnmapMemory(allocator, allocation);
        mapped_ptr = nullptr;
    }
}

void MemoryCommit::Release() {
    if (allocation && allocator) {
        const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
        // Log GPU memory deallocation
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_memory_tracking.GetValue() &&
            memory != VK_NULL_HANDLE) {
            GPU::Logging::GPULogger::GetInstance().LogMemoryDeallocation(
                reinterpret_cast<uintptr_t>(memory)
            );
        }

        if (mapped_ptr) {
            vmaUnmapMemory(allocator, allocation);
            mapped_ptr = nullptr;
        }
        vmaFreeMemory(allocator, allocation);
    }
    allocation = nullptr;
    allocator = nullptr;
    memory = VK_NULL_HANDLE;
    offset = 0;
    size = 0;
}

MemoryAllocator::MemoryAllocator(const Device &device_, bool buffers)
        : device{device_}, allocator{buffers ? device.GetBufferAllocator() : device.GetAllocator()},
            properties{device_.GetPhysical().GetMemoryProperties().memoryProperties},
            buffer_image_granularity{
                    device_.GetPhysical().GetProperties().limits.bufferImageGranularity} {

    // Preserve the previous "RenderDoc small heap" trimming behavior that we had in original vma minus the heap bug
    if (device.HasDebuggingToolAttached())
    {
        using namespace Common::Literals;
        ForEachDeviceLocalHostVisibleHeap(device, [this](size_t heap_idx, VkMemoryHeap &heap) {
            if (heap.size <= 256_MiB) {
                for (u32 t = 0; t < properties.memoryTypeCount; ++t) {
                    if (properties.memoryTypes[t].heapIndex == heap_idx) {
                        valid_memory_types &= ~(1u << t);
                    }
                }
            }
        });
    }
}

MemoryAllocator::~MemoryAllocator() = default;

vk::Image MemoryAllocator::CreateImage(const VkImageCreateInfo &ci, bool dedicated) const
{
    vk::Image image;
    const u64 start = Common::Nx2Hitch::NowNs();
    vk::Check(AllocateImage(ci, dedicated, false, image));
    Common::Nx2Hitch::Add(Common::Nx2Hitch::Cause::ImageCreate, Common::Nx2Hitch::NowNs() - start);
    return image;
}

vk::Image MemoryAllocator::TryCreateImage(const VkImageCreateInfo &ci, bool dedicated) const {
    vk::Image image;
    const u64 start = Common::Nx2Hitch::NowNs();
    const VkResult result = AllocateImage(ci, dedicated, true, image);
    Common::Nx2Hitch::Add(Common::Nx2Hitch::Cause::ImageCreate, Common::Nx2Hitch::NowNs() - start);
    if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY && result != VK_ERROR_OUT_OF_HOST_MEMORY) {
        vk::Check(result);
    }
    return image;
}

namespace {
std::atomic<u64> g_timed_images{};
std::atomic<u64> g_timed_create_ns{};
std::atomic<u64> g_timed_alloc_ns{};
std::atomic<u64> g_timed_bind_ns{};
std::atomic<u64> g_timed_new_block{};    ///< Images whose create added a VMA memory object
std::atomic<u64> g_timed_new_block_ns{}; ///< ... their create, allocate and bind time
std::atomic<u64> g_timed_dedicated{};    ///< ... of them, dedicated allocations

u64 TimingNow() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

// vmaCreateImage in its three steps (as VMA does them, without its dedicated-allocation hint
// queries, which VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT callers set themselves), each timed.
VkResult TimedVmaCreateImage(const Device &device, VmaAllocator allocator,
                             const VkImageCreateInfo &ci, const VmaAllocationCreateInfo &alloc_ci,
                             VkImage &handle, VmaAllocation &allocation,
                             VmaAllocationInfo &alloc_info) {
    const auto &dld = device.GetDispatchLoader();
    const VkDevice logical = *device.GetLogical();
    // Outside the timed steps: did this image take a memory object VMA did not hold before?
    const u64 blocks_before = vk::VmaMemoryObjectCount(allocator);
    const u64 t0 = TimingNow();
    VkResult result = dld.vkCreateImage(logical, &ci, nullptr, &handle);
    const u64 t1 = TimingNow();
    if (result != VK_SUCCESS) {
        handle = VK_NULL_HANDLE;
        return result;
    }
    // VMA_MEMORY_USAGE_AUTO* needs the create info (vmaCreateImage only; here it fails with
    // VK_ERROR_FEATURE_NOT_PRESENT): the same preference without it.
    VmaAllocationCreateInfo explicit_ci = alloc_ci;
    if (explicit_ci.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE ||
        explicit_ci.usage == VMA_MEMORY_USAGE_AUTO) {
        explicit_ci.usage = VMA_MEMORY_USAGE_UNKNOWN;
        explicit_ci.preferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else if (explicit_ci.usage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST) {
        explicit_ci.usage = VMA_MEMORY_USAGE_UNKNOWN;
    }
    result = vmaAllocateMemoryForImage(allocator, handle, &explicit_ci, &allocation, &alloc_info);
    const u64 t2 = TimingNow();
    if (result == VK_SUCCESS) {
        result = vmaBindImageMemory(allocator, allocation, handle);
        if (result != VK_SUCCESS) {
            vmaFreeMemory(allocator, allocation);
        }
    }
    const u64 t3 = TimingNow();
    if (result != VK_SUCCESS) {
        dld.vkDestroyImage(logical, handle, nullptr);
        handle = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
        return result;
    }
    g_timed_images.fetch_add(1, std::memory_order_relaxed);
    g_timed_create_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
    g_timed_alloc_ns.fetch_add(t2 - t1, std::memory_order_relaxed);
    g_timed_bind_ns.fetch_add(t3 - t2, std::memory_order_relaxed);
    if (vk::VmaMemoryObjectCount(allocator) > blocks_before) {
        g_timed_new_block.fetch_add(1, std::memory_order_relaxed);
        g_timed_new_block_ns.fetch_add(t3 - t0, std::memory_order_relaxed);
        VmaAllocationInfo2 info2{};
        vmaGetAllocationInfo2(allocator, allocation, &info2);
        if (info2.dedicatedMemory) {
            g_timed_dedicated.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // Per image, the slow ones (over 500 us in all): is the memory a block VMA had already (a hole
    // reused) or one it just took (fable_answer_npcdips_2 section 1)? 400 lines at most.
    static std::mutex blocks_mutex;
    static std::unordered_set<VkDeviceMemory> blocks_seen;
    static u32 slow_logged = 0;
    bool new_block = false;
    {
        std::scoped_lock lock{blocks_mutex};
        new_block = blocks_seen.insert(alloc_info.deviceMemory).second;
        if (t3 - t0 < 500'000 || slow_logged >= 400) {
            return VK_SUCCESS;
        }
        ++slow_logged;
    }
    LOG_INFO(Render_Vulkan,
             "Slow image create: {}x{} format {} usage {:#x}, {} KiB at offset {} KiB of {} "
             "block; vkCreateImage {} us, allocate {} us, bind {} us",
             ci.extent.width, ci.extent.height, static_cast<u32>(ci.format), ci.usage,
             alloc_info.size >> 10, alloc_info.offset >> 10, new_block ? "a NEW" : "an old",
             (t1 - t0) / 1000, (t2 - t1) / 1000, (t3 - t2) / 1000);
    return VK_SUCCESS;
}
} // Anonymous namespace

// EDEN_NX2_DEPTH_POOL (boot, on by default, =0 off): depth/stencil attachments come from their own VMA
// pool per memory type, so the memory a freed depth target leaves is reused by depth targets only.
// The NPC-pass dips (night market): a new D32 target nearly every frame, and its vmaBindImageMemory
// took 1.35 ms on average against 0.11 ms for colour targets (IMAGE_ALLOC_TIMING): the driver
// readies the memory for depth on each bind, which memory that held depth before may skip. The
// pools live as long as the VMA allocator (never destroyed: images may outlive this object).
// At file scope so the owners line's slack split and FRITCHII_VRAM_TRIM can walk them.
struct DepthPoolEntry {
    VmaAllocator owner;
    u32 type;
    VmaPool pool;
    bool is_small; ///< FRITCHII_DEPTH_POOL_FIT's pool for small depth targets
};
static std::mutex g_depth_pool_mutex;
static std::vector<DepthPoolEntry> g_depth_pools;

/// FRITCHII_DEPTH_POOL_FIT's small depth pool: its block size (FRITCHII_DEPTH_POOL_SMALL_BLOCK_MIB,
/// boot, default 16, 4-64) and the largest padded allocation it takes (a quarter of a block).
static VkDeviceSize SmallDepthBlockBytes() {
    static const VkDeviceSize bytes = [] {
        const char* value = std::getenv("FRITCHII_DEPTH_POOL_SMALL_BLOCK_MIB");
        const unsigned long mib = value != nullptr ? std::strtoul(value, nullptr, 10) : 16UL;
        return VkDeviceSize{(std::clamp)(mib, 4UL, 64UL)} << 20;
    }();
    return bytes;
}

/// The depth pool (or with `small` FRITCHII_DEPTH_POOL_FIT's small one) of a memory type, made on
/// first use.
static VmaPool DepthPoolOfType(VmaAllocator allocator, u32 type, bool is_small);

static VmaPool DepthAttachmentPool(VmaAllocator allocator, const VkImageCreateInfo &ci,
                                   const VmaAllocationCreateInfo &alloc_ci) {
    // On by default since 2026-11-12: depth targets no longer each get their own fresh memory (the
    // driver prefers dedicated memory for them outside a pool). Night market, camera panning,
    // 120 s: binds over 5 ms 6 -> 0 (worst 162 ms -> 1.8 ms), hitches over 50 ms 5 -> 1.
    static const bool enabled = [] {
        const char *value = std::getenv("EDEN_NX2_DEPTH_POOL");
        return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    if (!enabled) {
        return VK_NULL_HANDLE;
    }
    u32 type = 0;
    if (vmaFindMemoryTypeIndexForImageInfo(allocator, &ci, &alloc_ci, &type) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return DepthPoolOfType(allocator, type, false);
}

static VmaPool DepthPoolOfType(VmaAllocator allocator, u32 type, bool is_small) {
    auto& pools = g_depth_pools;
    std::scoped_lock lock{g_depth_pool_mutex};
    for (const DepthPoolEntry& entry : pools) {
        if (entry.owner == allocator && entry.type == type && entry.is_small == is_small) {
            return entry.pool;
        }
    }
    using namespace Common::Literals;
    // EDEN_NX2_DEPTH_POOL_MIN_BLOCKS (boot, default 0; try 4 = 256 MiB): blocks the pool never gives
    // back. VMA frees an empty block, so with 0 the pool still asked the driver for new memory at
    // each peak (night market panning: 39 images on a new block, 4 binds over 20 ms, worst 67 ms).
    static const u32 min_blocks = [] {
        const char *value = std::getenv("EDEN_NX2_DEPTH_POOL_MIN_BLOCKS");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0U;
    }();
    const VmaPoolCreateInfo pool_ci{
        .memoryTypeIndex = type,
        .flags = 0,
        .blockSize =
            is_small ? SmallDepthBlockBytes()
                     : (Common::HostHasSmallVideoMemory() ? 32_MiB : 64_MiB),
        .minBlockCount = is_small ? 0U : min_blocks,
        .maxBlockCount = 0,
        .priority = 0.f,
        .minAllocationAlignment = 0,
        .pMemoryAllocateNext = nullptr,
    };
    VmaPool pool{};
    if (vmaCreatePool(allocator, &pool_ci, &pool) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    // Named, so VMA's detailed map (FRITCHII_VRAM_HOLES) tells the pools apart.
    vmaSetPoolName(allocator, pool, is_small ? "depth small" : "depth");
    LOG_INFO(Render_Vulkan, "Depth attachment pool (EDEN_NX2_DEPTH_POOL{}) for memory type {}: {} MiB "
             "blocks", is_small ? ", FRITCHII_DEPTH_POOL_FIT small" : "", type,
             pool_ci.blockSize >> 20);
    pools.push_back({allocator, type, pool, is_small});
    return pool;
}

namespace {

// FRITCHII_KILL_RT_POOL (live, default 0): colour render targets from their own VMA pool, as
// EDEN_NX2_DEPTH_POOL does for depth. Outside a pool the driver's preference gives a colour
// target its own memory, so each new one is a driver allocation on the GPU back thread. In the
// owner's Stalchild fights (fight-keepers*, 2026-11-14) the only frames over 50 ms after the load
// were single image creates of 35 and 124 ms right at kills (one of 159 ms in a playtest), with
// the back blocked and the worker, fencing and present threads idle behind it. 1 = colour targets
// of up to 32 MiB are made with vkCreateImage, allocated from the pool (64 MiB blocks) and bound;
// an image the pool refuses goes the old way. 2 = dry: nothing moves, the census below counts
// the candidates and how many of them got memory of their own. Only memory placement changes,
// never what is drawn. Images made before a switch change stay where they are.
int KillRtPoolMode() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_kill_rt_pool,
                                            "FRITCHII_KILL_RT_POOL");
}

constexpr VkDeviceSize KillPoolBlockBytes = VkDeviceSize{64} << 20;
constexpr VkDeviceSize KillPoolMaxBytes = KillPoolBlockBytes / 2;

bool IsKillPoolCandidate(const VkImageCreateInfo& ci, bool dedicated) {
    constexpr VkImageCreateFlags refused = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                           VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                                           VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                                           VK_IMAGE_CREATE_DISJOINT_BIT;
    return !dedicated && (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0 &&
           (ci.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0 &&
           (ci.flags & refused) == 0 && ci.imageType == VK_IMAGE_TYPE_2D;
}

struct KillPoolEntry {
    VmaAllocator owner;
    u32 type;
    VmaPool pool;
};
std::mutex g_kill_pool_mutex;
std::vector<KillPoolEntry> g_kill_pools;

// One pool per allocator and memory type, never destroyed (images may outlive the allocator
// object). FRITCHII_KILL_RT_POOL_BLOCKS (boot, default 2, at most 8 = 512 MiB): blocks the pool
// makes at once and never gives back, so the first targets of a fight find memory already made.
VmaPool ColourAttachmentPool(VmaAllocator allocator, const VkImageCreateInfo& ci,
                             const VmaAllocationCreateInfo& alloc_ci) {
    u32 type = 0;
    if (vmaFindMemoryTypeIndexForImageInfo(allocator, &ci, &alloc_ci, &type) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    std::scoped_lock lock{g_kill_pool_mutex};
    for (const KillPoolEntry& entry : g_kill_pools) {
        if (entry.owner == allocator && entry.type == type) {
            return entry.pool;
        }
    }
    static const u32 min_blocks = [] {
        const char* value = std::getenv("FRITCHII_KILL_RT_POOL_BLOCKS");
        const unsigned long blocks = value != nullptr ? std::strtoul(value, nullptr, 10) : 2UL;
        return static_cast<u32>((std::min)(blocks, 8UL));
    }();
    const VmaPoolCreateInfo pool_ci{
        .memoryTypeIndex = type,
        .flags = 0,
        .blockSize = KillPoolBlockBytes,
        .minBlockCount = min_blocks,
        .maxBlockCount = 0,
        .priority = 0.f,
        .minAllocationAlignment = 0,
        .pMemoryAllocateNext = nullptr,
    };
    VmaPool pool{};
    if (vmaCreatePool(allocator, &pool_ci, &pool) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    vmaSetPoolName(allocator, pool, "colour");
    LOG_INFO(Render_Vulkan,
             "Colour target pool (FRITCHII_KILL_RT_POOL) for memory type {}: 64 MiB blocks, {} kept",
             type, min_blocks);
    g_kill_pools.push_back({allocator, type, pool});
    return pool;
}

// What the census sorts image creates into.
enum KillKind : u32 {
    KillColourPool,  ///< a colour target in the FRITCHII_KILL_RT_POOL pool
    KillColourOwn,   ///< a colour target with memory of its own (a driver allocation)
    KillColourBlock, ///< a colour target in a shared VMA block
    KillDepthOwn,
    KillDepthBlock, ///< includes EDEN_NX2_DEPTH_POOL
    KillOtherOwn,
    KillOtherBlock,
    KillKinds,
};
constexpr std::array<const char*, KillKinds> kill_kind_names{
    "colour in the kill pool", "colour own memory", "colour in a block", "depth own memory",
    "depth in a block",        "other own memory",  "other in a block"};

struct KillCensus {
    std::array<u64, KillKinds> count{};
    std::array<u64, KillKinds> slow5{};
    std::array<u64, KillKinds> slow20{};
    std::array<u64, KillKinds> ns{};
    std::array<u64, KillKinds> longest{};
    u64 worst_ns = 0;
    VkExtent3D worst_extent{};
    u32 worst_format = 0;
    u32 worst_usage = 0;
    u32 worst_kind = 0;
    u64 placed = 0;      ///< taken by the pool
    u64 too_big = 0;     ///< over KillPoolMaxBytes: went the old way
    u64 refused = 0;     ///< the pool or its bind failed: went the old way
    u64 candidates = 0;  ///< dry mode: colour targets the pool would have tried
    u64 dry_own = 0;     ///< ... of them given memory of their own
    u64 window_start = 0;
};
std::mutex g_census_mutex;
KillCensus g_census;

// vkCreateImage, then the pool's memory, then the bind. False with nothing left behind when the
// pool does not take the image (too large, no pool, no room): the caller goes the old way.
bool TryKillPool(const Device& device, VmaAllocator allocator, const VkImageCreateInfo& ci,
                 const VmaAllocationCreateInfo& alloc_ci, VkImage& handle,
                 VmaAllocation& allocation, VmaAllocationInfo& alloc_info, bool& too_big) {
    too_big = false;
    const VmaPool pool = ColourAttachmentPool(allocator, ci, alloc_ci);
    if (pool == VK_NULL_HANDLE) {
        return false;
    }
    const auto& dld = device.GetDispatchLoader();
    const VkDevice logical = *device.GetLogical();
    VkImage image{};
    if (dld.vkCreateImage(logical, &ci, nullptr, &image) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements requirements{};
    dld.vkGetImageMemoryRequirements(logical, image, &requirements);
    if (requirements.size > KillPoolMaxBytes) {
        dld.vkDestroyImage(logical, image, nullptr);
        too_big = true;
        return false;
    }
    // A pool fixes the memory type; usage and preferences are not read (the AUTO usages would
    // need vmaCreateImage's create info, see TimedVmaCreateImage).
    VmaAllocationCreateInfo pool_ci = alloc_ci;
    pool_ci.flags &= ~VmaAllocationCreateFlags{VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    pool_ci.usage = VMA_MEMORY_USAGE_UNKNOWN;
    pool_ci.pool = pool;
    VmaAllocation pooled{};
    VmaAllocationInfo pooled_info{};
    if (vmaAllocateMemoryForImage(allocator, image, &pool_ci, &pooled, &pooled_info) !=
        VK_SUCCESS) {
        dld.vkDestroyImage(logical, image, nullptr);
        return false;
    }
    if (vmaBindImageMemory(allocator, pooled, image) != VK_SUCCESS) {
        vmaFreeMemory(allocator, pooled);
        dld.vkDestroyImage(logical, image, nullptr);
        return false;
    }
    handle = image;
    allocation = pooled;
    alloc_info = pooled_info;
    return true;
}

// Called after every image made with memory (same thread as the allocation: the allocator is
// externally synchronized, so the pool statistics below may be read here). Logs every 10 s
// while FRITCHII_KILL_RT_POOL is not 0, and at 0 only for a window that had a create of 20 ms
// or more.
void NoteKillCensus(VmaAllocator allocator, const VkImageCreateInfo& ci, VmaAllocation allocation,
                    u64 elapsed_ns, bool pooled, bool candidate, int mode) {
    bool own = false;
    if (allocation != VK_NULL_HANDLE) {
        VmaAllocationInfo2 info2{};
        vmaGetAllocationInfo2(allocator, allocation, &info2);
        own = info2.dedicatedMemory != VK_FALSE;
    }
    const bool colour = (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0;
    const bool depth = (ci.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
    const u32 kind = pooled  ? KillColourPool
                     : depth ? (own ? KillDepthOwn : KillDepthBlock)
                     : colour ? (own ? KillColourOwn : KillColourBlock)
                              : (own ? KillOtherOwn : KillOtherBlock);
    const u64 now = Common::Nx2Hitch::NowNs();
    std::string line;
    {
        std::scoped_lock lock{g_census_mutex};
        KillCensus& c = g_census;
        if (c.window_start == 0) {
            c.window_start = now;
        }
        ++c.count[kind];
        c.ns[kind] += elapsed_ns;
        c.slow5[kind] += elapsed_ns >= 5'000'000 ? 1 : 0;
        c.slow20[kind] += elapsed_ns >= 20'000'000 ? 1 : 0;
        c.longest[kind] = (std::max)(c.longest[kind], elapsed_ns);
        if (elapsed_ns > c.worst_ns) {
            c.worst_ns = elapsed_ns;
            c.worst_extent = ci.extent;
            c.worst_format = static_cast<u32>(ci.format);
            c.worst_usage = ci.usage;
            c.worst_kind = kind;
        }
        c.placed += pooled ? 1 : 0;
        if (mode == 2 && candidate) {
            ++c.candidates;
            c.dry_own += own ? 1 : 0;
        }
        if (now - c.window_start < 10'000'000'000ULL) {
            return;
        }
        if (mode != 0 || c.worst_ns >= 20'000'000) {
            u64 made = 0;
            u64 total_ns = 0;
            std::string kinds;
            for (u32 k = 0; k < KillKinds; ++k) {
                made += c.count[k];
                total_ns += c.ns[k];
                if (c.count[k] == 0) {
                    continue;
                }
                kinds += fmt::format("{}{} {} ({} ms, {} over 5 ms, {} over 20 ms, longest {:.1f} ms)",
                                     kinds.empty() ? "" : "; ", kill_kind_names[k], c.count[k],
                                     c.ns[k] / 1'000'000, c.slow5[k], c.slow20[k],
                                     static_cast<double>(c.longest[k]) / 1e6);
            }
            u64 pool_blocks = 0;
            u64 pool_block_bytes = 0;
            u64 pool_used_bytes = 0;
            {
                std::scoped_lock pool_lock{g_kill_pool_mutex};
                for (const KillPoolEntry& entry : g_kill_pools) {
                    if (entry.owner != allocator) {
                        continue;
                    }
                    VmaStatistics stats{};
                    vmaGetPoolStatistics(allocator, entry.pool, &stats);
                    pool_blocks += stats.blockCount;
                    pool_block_bytes += stats.blockBytes;
                    pool_used_bytes += stats.allocationBytes;
                }
            }
            line = fmt::format(
                "Kill image creates (FRITCHII_KILL_RT_POOL={}) in 10 s: {} made in {} ms; {}; worst "
                "{:.1f} ms: {}x{}x{} format {} usage {:#x} ({}); kill pool: {} placed, {} over 32 "
                "MiB, {} refused, {} blocks, {} of {} MiB used; dry: {} colour targets the pool "
                "would try, {} of them got memory of their own",
                mode, made, total_ns / 1'000'000, kinds.empty() ? "none" : kinds,
                static_cast<double>(c.worst_ns) / 1e6, c.worst_extent.width,
                c.worst_extent.height, c.worst_extent.depth, c.worst_format, c.worst_usage,
                kill_kind_names[c.worst_kind], c.placed, c.too_big, c.refused, pool_blocks,
                pool_used_bytes >> 20, pool_block_bytes >> 20, c.candidates, c.dry_own);
        }
        c = KillCensus{};
        c.window_start = now;
    }
    if (!line.empty()) {
        LOG_INFO(Render_Vulkan, "{}", line);
    }
}

void NoteKillPoolMiss(bool too_big) {
    std::scoped_lock lock{g_census_mutex};
    if (too_big) {
        ++g_census.too_big;
    } else {
        ++g_census.refused;
    }
}

static_assert(sizeof(void*) == 8, "image allocation user data needs 64-bit pointers");

/// When an image allocation is made, in 100 ms steps since the first (vk::ImageUserDataTag).
u64 ImageStamp() {
    static const auto start = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();
    return static_cast<u64>(ms / 100) & 0xFFFF'FFFFULL;
}

/// VMA user data of an image allocation: tag, padding (256-byte units, a multiple of 256 given)
/// and the stamp. VMA never reads it (no USER_DATA_COPY_STRING).
void* ImageUserData(u64 padding) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(
        vk::ImageUserDataTag | (((padding >> 8) & 0x7FFF'FFFFULL) << 32) | ImageStamp()));
}

// FRITCHII_DEPTH_POOL_FIT (live, default 0; bits): how depth targets are placed in the depth
// pool (EDEN_NX2_DEPTH_POOL). The market's depth pool held 1152-1280 MiB in 18-20 blocks of
// 64 MiB with 529-727 MiB of it in 76-177 holes (batch 29): OoT makes depth targets sized by
// screen bounds (per-NPC D32 targets, shadow cascades) next to large long-lived ones, so the hole
// a freed target leaves rarely fits the next one exactly, and a small long-lived target pins a
// whole 64 MiB block.
//  1 = size classes: the allocation is rounded up to a class, four per power of two (at most a
//      quarter more, ~1/8 on average), so a freed target's hole fits the next target of its class.
//      The padding is counted in the owners line ("depth fit padding").
//  2 = small depth targets (at most a quarter of FRITCHII_DEPTH_POOL_SMALL_BLOCK_MIB, default
//      16 MiB blocks, so up to 4 MiB) go to a depth pool of their own: they no longer fill holes
//      large targets need, and one that lives long pins a 16 MiB block instead of 64.
//  3 = both. Only where the memory sits changes; the images and their contents are the same.
int DepthFitMode() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_depth_pool_fit,
                                            "FRITCHII_DEPTH_POOL_FIT");
}

/// Size class: up to the next quarter of the power of two below `size` (at least 4 KiB steps),
/// then up to the alignment.
VkDeviceSize DepthFitClass(VkDeviceSize size, VkDeviceSize alignment) {
    const VkDeviceSize power = std::bit_floor(size);
    const VkDeviceSize step = (std::max)(power / 4, VkDeviceSize{4096});
    const VkDeviceSize rounded = Common::AlignUp(size, static_cast<size_t>(step));
    return alignment > 1 ? Common::AlignUp(rounded, static_cast<size_t>(alignment)) : rounded;
}

std::atomic<u64> g_fit_placed{};    ///< FRITCHII_DEPTH_POOL_FIT: images it placed
std::atomic<u64> g_fit_small{};     ///< ... of them in the small depth pool
std::atomic<u64> g_fit_fallbacks{}; ///< images it handed back to the old path

/// Whether the driver requires the image in memory of its own (then no pool may take it).
bool RequiresDedicated(const Device& device, VkImage image) {
    static std::mutex mutex;
    static VkDevice cached_device{};
    static PFN_vkGetImageMemoryRequirements2 requirements2{};
    const VkDevice logical = *device.GetLogical();
    PFN_vkGetImageMemoryRequirements2 function{};
    {
        std::scoped_lock lock{mutex};
        if (cached_device != logical) {
            cached_device = logical;
            requirements2 = reinterpret_cast<PFN_vkGetImageMemoryRequirements2>(
                device.GetDispatchLoader().vkGetDeviceProcAddr(logical,
                                                               "vkGetImageMemoryRequirements2"));
        }
        function = requirements2;
    }
    if (function == nullptr) {
        return false;
    }
    VkMemoryDedicatedRequirements dedicated{};
    dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    VkMemoryRequirements2 out{};
    out.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    out.pNext = &dedicated;
    VkImageMemoryRequirementsInfo2 info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
    info.image = image;
    function(logical, &info, &out);
    return dedicated.requiresDedicatedAllocation != VK_FALSE;
}

/// FRITCHII_DEPTH_POOL_FIT: vkCreateImage, the (rounded) memory from the depth pool or the small
/// one, then the bind. False with nothing left behind when it does not take the image: the caller
/// goes the old way (the depth pool through vmaCreateImage, then the fallbacks).
bool TryDepthFit(const Device& device, VmaAllocator allocator, const VkImageCreateInfo& ci,
                 const VmaAllocationCreateInfo& alloc_ci, VmaPool large_pool, int mode,
                 VkImage& handle, VmaAllocation& allocation, VmaAllocationInfo& alloc_info) {
    using namespace Common::Literals;
    // The depth pool's memory type (found once per image by the caller's DepthAttachmentPool).
    u32 type = VK_MAX_MEMORY_TYPES;
    {
        std::scoped_lock lock{g_depth_pool_mutex};
        for (const DepthPoolEntry& entry : g_depth_pools) {
            if (entry.pool == large_pool) {
                type = entry.type;
            }
        }
    }
    if (large_pool == VK_NULL_HANDLE || type == VK_MAX_MEMORY_TYPES) {
        return false;
    }
    const auto& dld = device.GetDispatchLoader();
    const VkDevice logical = *device.GetLogical();
    VkImage image{};
    if (dld.vkCreateImage(logical, &ci, nullptr, &image) != VK_SUCCESS) {
        return false;
    }
    const auto give_up = [&] {
        dld.vkDestroyImage(logical, image, nullptr);
        g_fit_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    VkMemoryRequirements requirements{};
    dld.vkGetImageMemoryRequirements(logical, image, &requirements);
    if (RequiresDedicated(device, image)) {
        return give_up();
    }
    VkDeviceSize size = requirements.size;
    if ((mode & 1) != 0) {
        size = DepthFitClass(size, requirements.alignment);
    }
    VmaPool pool = large_pool;
    VkDeviceSize block = 64_MiB;
    bool is_small = false;
    if ((mode & 2) != 0 && size <= SmallDepthBlockBytes() / 4) {
        if (const VmaPool small_pool = DepthPoolOfType(allocator, type, true)) {
            pool = small_pool;
            block = SmallDepthBlockBytes();
            is_small = true;
        }
    }
    if (size > block) {
        if (requirements.size > block) {
            return give_up(); // larger than a pool block: VMA refuses it there anyway
        }
        size = requirements.size;
    }
    const u64 padding = ((size - requirements.size) >> 8) << 8;
    VmaAllocationCreateInfo pool_ci = alloc_ci;
    pool_ci.flags &= ~VmaAllocationCreateFlags{VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT};
    pool_ci.usage = VMA_MEMORY_USAGE_UNKNOWN;
    pool_ci.memoryTypeBits = 0;
    pool_ci.pool = pool;
    pool_ci.pUserData = ImageUserData(padding);
    VkMemoryRequirements padded = requirements;
    padded.size = size;
    VmaAllocation pooled{};
    VmaAllocationInfo pooled_info{};
    if (vmaAllocateMemory(allocator, &padded, &pool_ci, &pooled, &pooled_info) != VK_SUCCESS) {
        return give_up();
    }
    if (vmaBindImageMemory(allocator, pooled, image) != VK_SUCCESS) {
        vmaFreeMemory(allocator, pooled);
        return give_up();
    }
    if (padding != 0) {
        vk::g_image_padding_bytes.fetch_add(padding, std::memory_order_relaxed);
    }
    g_fit_placed.fetch_add(1, std::memory_order_relaxed);
    if (is_small) {
        g_fit_small.fetch_add(1, std::memory_order_relaxed);
    }
    handle = image;
    allocation = pooled;
    alloc_info = pooled_info;
    return true;
}

/// The first refusals of a session get the heaps in full: which heap was short, by how much, and
/// whether VMA follows the driver's budget ("over the memory budget" with 1.1 GB used could not
/// be explained from the log).
constexpr u32 RefusalDetailEvents = 4;

/// Every heap's driver usage and budget and VMA's blocks and allocations in it, in MiB.
std::string HeapBudgetsText(VmaAllocator allocator) {
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    std::string text;
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        const VmaBudget& budget = budgets[heap];
        fmt::format_to(std::back_inserter(text),
                       "{}heap {} ({}, {} MiB): usage {} of budget {}, VMA blocks {}, allocations {}",
                       heap == 0 ? "" : "; ", heap,
                       (properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
                           ? "local"
                           : "host",
                       properties->memoryHeaps[heap].size >> 20, budget.usage >> 20,
                       budget.budget >> 20, budget.statistics.blockBytes >> 20,
                       budget.statistics.allocationBytes >> 20);
    }
    return text;
}

/// "type T (heap H, flags F)" of an allocation that was made.
std::string PlacementText(VmaAllocator allocator, u32 memory_type) {
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    if (memory_type >= properties->memoryTypeCount) {
        return fmt::format("type {}", memory_type);
    }
    const VkMemoryType& type = properties->memoryTypes[memory_type];
    return fmt::format("type {} (heap {}, flags {:#x})", memory_type, type.heapIndex,
                       static_cast<u32>(type.propertyFlags));
}

/// FRITCHII_IMAGE_DEVICE_LOCAL (boot, default 0; discrete GPUs only, 0 on integrated ones).
/// Over the budget, the first try in the default heaps (WITHIN_BUDGET, no required flags) lets VMA
/// move on to a host memory type that is still within its budget: the image silently lives in
/// system RAM, the GPU samples it over PCIe (another fork measured 9.4 GB of images there in the
/// market, 150 ms frames), and since the device heap's usage does not count it the texture cache
/// never sees pressure. 1 = that first try requires device-local memory, so a full device heap
/// refuses it and the refusal is a pressure event (the collector keeps more free); the fallbacks
/// below then place it as before (over the budget in device memory, host memory only when the
/// device really is out). 2 = dry: placement unchanged, counts the images the first try put
/// outside device-local memory.
int ImageDeviceLocalMode(const Device& device) {
    static const int mode = [&device] {
        const char* value = std::getenv("FRITCHII_IMAGE_DEVICE_LOCAL");
        const int parsed = value != nullptr ? std::atoi(value) : 0;
        if (parsed != 1 && parsed != 2) {
            return 0;
        }
        if (device.IsIntegrated()) {
            LOG_INFO(Render_Vulkan, "FRITCHII_IMAGE_DEVICE_LOCAL={}: integrated GPU, ignored",
                     parsed);
            return 0;
        }
        LOG_INFO(Render_Vulkan, "FRITCHII_IMAGE_DEVICE_LOCAL={}: {}", parsed,
                 parsed == 1 ? "images require device-local memory on the first try"
                             : "dry, counting images the first try puts in host memory");
        return parsed;
    }();
    return mode;
}

/// FRITCHII_IMAGE_DEVICE_LOCAL's counters, logged when a count reaches a power of two and then
/// every 1024.
std::atomic<u64> g_local_refused{};    ///< =1: first tries the device-local requirement refused
std::atomic<u64> g_local_over_budget{}; ///< =1: of those, placed in device memory over budget
std::atomic<u64> g_host_images{};      ///< images placed outside device-local memory
std::atomic<u64> g_host_bytes{};
std::atomic<u64> g_host_first_try{};   ///< of those, by the first try (the silent spill)

bool LogThisCount(u64 count) {
    return count != 0 && ((count & (count - 1)) == 0 || count % 1024 == 0);
}

void LogDeviceLocalCounters(int mode, const char* event) {
    LOG_INFO(Render_Vulkan,
             "Image device-local (FRITCHII_IMAGE_DEVICE_LOCAL={}, {}): first tries refused {}, "
             "of those placed over budget in device memory {}; host memory images {} ({} MiB), "
             "by the first try {}",
             mode, event, g_local_refused.load(std::memory_order_relaxed),
             g_local_over_budget.load(std::memory_order_relaxed),
             g_host_images.load(std::memory_order_relaxed),
             g_host_bytes.load(std::memory_order_relaxed) >> 20,
             g_host_first_try.load(std::memory_order_relaxed));
}

/// Counts where an image made under FRITCHII_IMAGE_DEVICE_LOCAL landed.
void NoteDeviceLocalPlacement(VmaAllocator allocator, int mode, bool refused_first,
                              bool first_try, const VmaAllocationInfo& alloc_info) {
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    const bool local = alloc_info.memoryType < properties->memoryTypeCount &&
                       (properties->memoryTypes[alloc_info.memoryType].propertyFlags &
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    if (local) {
        if (refused_first) {
            g_local_over_budget.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    g_host_bytes.fetch_add(alloc_info.size, std::memory_order_relaxed);
    if (first_try) {
        g_host_first_try.fetch_add(1, std::memory_order_relaxed);
    }
    if (LogThisCount(g_host_images.fetch_add(1, std::memory_order_relaxed) + 1)) {
        LogDeviceLocalCounters(mode, first_try ? "host image by the first try"
                                               : "host image by a fallback");
    }
}

} // Anonymous namespace

bool MemoryAllocator::ImageTimingEnabled() noexcept {
    static const bool enabled = [] {
        const char *value = std::getenv("EDEN_NX2_IMAGE_ALLOC_TIMING");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

MemoryAllocator::ImageTiming MemoryAllocator::ReadImageTiming() noexcept {
    return ImageTiming{
        .images = g_timed_images.load(std::memory_order_relaxed),
        .create_ns = g_timed_create_ns.load(std::memory_order_relaxed),
        .alloc_ns = g_timed_alloc_ns.load(std::memory_order_relaxed),
        .bind_ns = g_timed_bind_ns.load(std::memory_order_relaxed),
        .new_block_images = g_timed_new_block.load(std::memory_order_relaxed),
        .new_block_ns = g_timed_new_block_ns.load(std::memory_order_relaxed),
        .dedicated_images = g_timed_dedicated.load(std::memory_order_relaxed),
    };
}

VkResult MemoryAllocator::AllocateImage(const VkImageCreateInfo &ci, bool dedicated, bool may_fail,
                                        vk::Image &image) const {
    const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
    const VmaAllocationCreateInfo alloc_ci = {
            // Best fit: images freed by the thousand leave holes in the blocks, which new ones
            // should fill before VMA takes another block.
            .flags = static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT |
                                                           VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT) |
                     (dedicated ? VmaAllocationCreateFlags{VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT}
                               : VmaAllocationCreateFlags{0}),
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = 0,
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .memoryTypeBits = 0,
            .pool = VK_NULL_HANDLE,
            // Tag and stamp (vk::ImageUserDataTag): FRITCHII_VRAM_HOLES reads the age.
            .pUserData = ImageUserData(0),
            .priority = 0.f,
    };

    VkImage handle{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info{};
    const u64 texels = u64{ci.extent.width} * ci.extent.height * ci.extent.depth * ci.arrayLayers;
    u64 bytes = 0;
    if (texels > MaxBackedImageTexels) {
        // Its memory size decides: an image the driver won't create is too large to back.
        const auto& dld = device.GetDispatchLoader();
        VkImage probe{};
        bytes = ~u64{0};
        if (dld.vkCreateImage(*device.GetLogical(), &ci, nullptr, &probe) == VK_SUCCESS) {
            bytes = device.GetLogical().GetImageMemoryRequirements(probe).size;
            dld.vkDestroyImage(*device.GetLogical(), probe, nullptr);
        }
    }
    const bool huge = IsTooLargeToBack(texels, bytes);
    // FRITCHII_IMAGE_DEVICE_LOCAL=1: tries within the budget in the default heaps require
    // device-local memory (custom pools have their own memory type and ignore it); the
    // fallbacks without the budget check below take alloc_ci as before.
    const int device_local_mode = ImageDeviceLocalMode(device);
    VmaAllocationCreateInfo local_ci = alloc_ci;
    if (device_local_mode == 1) {
        local_ci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    // EDEN_NX2_DEPTH_POOL: the first attempt only; the fallbacks below take any memory.
    VmaAllocationCreateInfo first_ci = local_ci;
    if (!dedicated && (ci.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0) {
        first_ci.pool = DepthAttachmentPool(allocator, ci, alloc_ci);
    }
    // FRITCHII_KILL_RT_POOL: 1 = colour targets try their pool first; 2 = dry (census only).
    const u64 kill_start = Common::Nx2Hitch::NowNs();
    const int kill_mode = KillRtPoolMode();
    const bool kill_candidate = !huge && IsKillPoolCandidate(ci, dedicated);
    bool kill_pooled = false;
    if (kill_candidate && kill_mode == 1) {
        bool too_big = false;
        kill_pooled = TryKillPool(device, allocator, ci, alloc_ci, handle, allocation, alloc_info,
                                  too_big);
        if (!kill_pooled) {
            NoteKillPoolMiss(too_big);
        }
    }
    // FRITCHII_DEPTH_POOL_FIT: depth targets the depth pool takes, rounded to a size class or in
    // the small depth pool; anything it does not take goes the old way below.
    bool fit_placed = false;
    if (!kill_pooled && !huge && first_ci.pool != VK_NULL_HANDLE) {
        if (const int fit = DepthFitMode(); fit != 0) {
            fit_placed = TryDepthFit(device, allocator, ci, alloc_ci, first_ci.pool, fit, handle,
                                     allocation, alloc_info);
        }
    }
    VkResult result = kill_pooled || fit_placed ? VK_SUCCESS
                      : huge ? VK_ERROR_OUT_OF_DEVICE_MEMORY
                      : ImageTimingEnabled()
                          ? TimedVmaCreateImage(device, allocator, ci, first_ci, handle,
                                                allocation, alloc_info)
                          : vmaCreateImage(allocator, &ci, &first_ci, &handle, &allocation, &alloc_info);
    // FRITCHII_IMAGE_DEVICE_LOCAL: placed by the first try (=2 counts host placements of it).
    bool first_try_placed = result == VK_SUCCESS && !kill_pooled && !fit_placed;
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge && first_ci.pool != VK_NULL_HANDLE &&
        !kill_pooled &&
        (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_depth_pool_big,
                                          "FRITCHII_DEPTH_POOL_BIG") > 0 ||
         VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_depth_pool_retry,
                                          "FRITCHII_DEPTH_POOL_RETRY", 1) != 0)) {
        // Same fix from two helpers (BA: FRITCHII_DEPTH_POOL_BIG, BF: FRITCHII_DEPTH_POOL_RETRY);
        // either switch turns it on. On by default since 2026-11-16 (FRITCHII_DEPTH_POOL_RETRY=0
        // turns it off): only where the image is placed changes. Without it every OoT area load
        // logged a false "over the memory budget" (RX 9070 XT, 16 GB, 1.1 GB used) and the false
        // pressure event emptied the image pool and cut the cache's thresholds by 512 MiB.
        // FRITCHII_DEPTH_POOL_BIG: VMA refuses anything larger than a custom pool's block
        // (64 MiB) with VK_ERROR_OUT_OF_DEVICE_MEMORY before it looks at the budget, so a large
        // depth target (OoT, every Castle Town load: 12288x12288 D16, 288 MiB) read as "over
        // the memory budget": a false pressure event (+512 MiB kept free) and a placement
        // without the budget check. Such an image tries the normal heap first, budget kept.
        result = ImageTimingEnabled()
                     ? TimedVmaCreateImage(device, allocator, ci, local_ci, handle, allocation,
                                           alloc_info)
                     : vmaCreateImage(allocator, &ci, &local_ci, &handle, &allocation, &alloc_info);
        // The default heaps' first try for an image the depth pool refused.
        first_try_placed = result == VK_SUCCESS;
        static std::atomic<u32> placed{};
        if (const u32 n = placed.fetch_add(1, std::memory_order_relaxed); n < 8) {
            LOG_INFO(Render_Vulkan,
                     "Depth image {}x{}x{} (format {}) refused by the depth pool, outside it: {} "
                     "(FRITCHII_DEPTH_POOL_BIG, {} so far)",
                     ci.extent.width, ci.extent.height, ci.arrayLayers,
                     static_cast<u32>(ci.format),
                     result == VK_SUCCESS ? "placed within the budget" : "over the budget too",
                     n + 1);
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge && device.KeptBlockBytes() != 0) {
        // FRITCHII_KEEP_BLOCKS: the kept blocks count against the budget VMA checks; they go
        // first, and VMA reads the driver's usage again before the retry.
        if (device.ReleaseKeptBlocks() != 0) {
            vmaSetCurrentFrameIndex(allocator, 0);
            result = vmaCreateImage(allocator, &ci, &first_ci, &handle, &allocation, &alloc_info);
            first_try_placed = result == VK_SUCCESS;
        }
    }
    bool refused_first = false;
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        // Over the device memory budget (Switch 2 titles fill 8 GB cards): let the driver page
        // it out rather than failing the draw and aborting.
        if (device_local_mode == 1) {
            // FRITCHII_IMAGE_DEVICE_LOCAL=1: the device heap is full; this pressure event is
            // what makes the texture cache keep more free.
            refused_first = true;
            if (LogThisCount(g_local_refused.fetch_add(1, std::memory_order_relaxed) + 1)) {
                LogDeviceLocalCounters(device_local_mode, "first try refused");
            }
        }
        static u32 logged{};
        if (logged++ < 32) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) is over the memory "
                        "budget; allocating it anyway",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format));
        }
        static std::atomic<u32> detailed{};
        const bool detail = detailed.fetch_add(1, std::memory_order_relaxed) < RefusalDetailEvents;
        if (detail) {
            const char* const first_try = kill_pooled                       ? "the colour target pool"
                                          : first_ci.pool != VK_NULL_HANDLE ? "the depth attachment pool"
                                          : device_local_mode == 1
                                              ? "the default heaps, device-local required"
                                              : "the default heaps";
            LOG_WARNING(Render_Vulkan,
                        "  refused: {} memory, first try in {}, VMA follows the driver budget: "
                        "{}; {}",
                        dedicated ? "dedicated" : "block", first_try,
                        device.CanReportMemoryUsage() ? "yes" : "no (no VK_EXT_memory_budget)",
                        HeapBudgetsText(allocator));
        }
        pressure_events.fetch_add(1, std::memory_order_relaxed);
        VmaAllocationCreateInfo relaxed = alloc_ci;
        relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        result = vmaCreateImage(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
        if (detail) {
            LOG_WARNING(Render_Vulkan, "  without the budget check: {}",
                        result == VK_SUCCESS
                            ? "placed in " + PlacementText(allocator, alloc_info.memoryType)
                            : fmt::format("failed ({})", static_cast<int>(result)));
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        // Still out of device memory: a real image in host memory (slow, but valid) beats one
        // with nothing bound, which AMD's drivers do not survive being sampled.
        VmaAllocationCreateInfo host = alloc_ci;
        host.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        host.preferredFlags = 0;
        result = vmaCreateImage(allocator, &ci, &host, &handle, &allocation, &alloc_info);
        static u32 host_logged{};
        if (result == VK_SUCCESS && host_logged++ < 16) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) is out of device memory; "
                        "placed in host memory, {}",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format),
                        PlacementText(allocator, alloc_info.memoryType));
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        pressure_events.fetch_add(1, std::memory_order_relaxed);
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && ci.imageType != VK_IMAGE_TYPE_1D &&
        UsesUnbackedFallback(texels, bytes, may_fail)) {
        // Too big to back at all, e.g. title 1's sparse virtual texture arrays (8192x8192,
        // 68 layers, ~23 GB): on the console only resident tiles have memory. Use an image with
        // nothing bound, so views and copies still work, instead of aborting. A smaller image
        // gets one only for a caller that would throw otherwise.
        if (vk::Image sparse = CreateUnbackedImage(ci)) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) does not fit; using an "
                        "unbacked sparse image",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format));
            image = std::move(sparse);
            return VK_SUCCESS;
        }
    }
    if (result != VK_SUCCESS) {
        failed_images.fetch_add(1, std::memory_order_relaxed);
        static u32 failed_logged{};
        if (failed_logged == 0) {
            // What ran out: the device's heaps or the host's commit. Before the error line,
            // which is the one that reaches the log file when the session ends here.
            LogMemoryTrace(device);
        }
        if (failed_logged++ < 32) {
            LOG_ERROR(Render_Vulkan, "Image {}x{}x{} ({} levels, {} layers, format {}) failed: {}",
                      ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                      ci.arrayLayers, static_cast<u32>(ci.format), static_cast<int>(result));
        }
        return result;
    }

    if (device_local_mode != 0) {
        NoteDeviceLocalPlacement(allocator, device_local_mode, refused_first, first_try_placed,
                                 alloc_info);
    }

    // Log GPU memory allocation for images
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    NoteKillCensus(allocator, ci, allocation, Common::Nx2Hitch::NowNs() - kill_start, kill_pooled,
                   kill_candidate, kill_mode);

    image = vk::Image(handle, ci.usage, *device.GetLogical(), allocator, allocation,
                      device.GetDispatchLoader());
    return VK_SUCCESS;
}

vk::Image MemoryAllocator::CreateUnbackedImage(const VkImageCreateInfo &ci, bool aliased) const {
    if (ci.imageType == VK_IMAGE_TYPE_1D) {
        return vk::Image{};
    }
    VkImageCreateInfo sparse_ci = ci;
    sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    if (aliased) {
        sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
    }
    VkImage sparse{};
    if (device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &sparse_ci, nullptr,
                                                 &sparse) != VK_SUCCESS) {
        return vk::Image{};
    }
    return vk::Image(sparse, ci.usage, *device.GetLogical(), allocator, VK_NULL_HANDLE,
                     device.GetDispatchLoader());
}

VmaAllocation MemoryAllocator::AllocateSparseMemory(const VkMemoryRequirements &requirements,
                                                   VkDeviceMemory &memory,
                                                   VkDeviceSize &offset) const {
    const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
    const VmaAllocationCreateInfo alloc_ci{
        .flags = 0,
        .usage = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .preferredFlags = 0,
        .memoryTypeBits = requirements.memoryTypeBits,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    // In a pool of their own (see Device::SeparatePool); anything over half a block, or a pool
    // allocation that fails, takes the default blocks.
    u32 memory_type{};
    if (requirements.size <= Device::SeparatePoolBlockSize / 2 &&
        vmaFindMemoryTypeIndex(allocator, requirements.memoryTypeBits, &alloc_ci, &memory_type) ==
            VK_SUCCESS) {
        if (const VmaPool pool =
                device.GetSeparatePool(allocator, Device::SeparatePool::SparsePages, memory_type)) {
            VmaAllocationCreateInfo pool_ci = alloc_ci;
            pool_ci.pool = pool;
            if (vmaAllocateMemory(allocator, &requirements, &pool_ci, &allocation, &info) ==
                VK_SUCCESS) {
                memory = info.deviceMemory;
                offset = info.offset;
                return allocation;
            }
        }
    }
    if (vmaAllocateMemory(allocator, &requirements, &alloc_ci, &allocation, &info) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    memory = info.deviceMemory;
    offset = info.offset;
    return allocation;
}

void MemoryAllocator::FreeSparseMemory(VmaAllocation allocation) const {
    if (allocation != VK_NULL_HANDLE) {
        const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
        vmaFreeMemory(allocator, allocation);
    }
}

bool MemoryAllocator::CreateBufferInSeparatePool(const VkBufferCreateInfo& ci, MemoryUsage usage,
                                                 const VmaAllocationCreateInfo& alloc_ci,
                                                 VkBuffer& handle, VmaAllocation& allocation,
                                                 VmaAllocationInfo& alloc_info) const {
    // The buffer cache's buffers, churned by its collector: in a pool of their own (see
    // Device::SeparatePool). Anything over half a block takes the default blocks (dedicated);
    // the block size is FRITCHII_BUFFER_POOL_BLOCK_MB's.
    if (usage != MemoryUsage::DeviceLocal || ci.size > Device::DeviceBufferBlockSize() / 2) {
        return false;
    }
    u32 memory_type{};
    {
        std::scoped_lock lk{buffer_type_mutex};
        const auto key = (u64{ci.usage} << 32) | ci.flags;
        if (const auto it = buffer_memory_types.find(key); it != buffer_memory_types.end()) {
            memory_type = it->second;
        } else if (vmaFindMemoryTypeIndexForBufferInfo(allocator, &ci, &alloc_ci, &memory_type) ==
                   VK_SUCCESS) {
            buffer_memory_types.emplace(key, memory_type);
        } else {
            return false;
        }
    }
    const VmaPool pool =
        device.GetSeparatePool(allocator, Device::SeparatePool::DeviceBuffers, memory_type);
    if (pool == VK_NULL_HANDLE) {
        return false;
    }
    VmaAllocationCreateInfo pool_ci = alloc_ci;
    pool_ci.pool = pool;
    pool_ci.memoryTypeBits = 0;
    return vmaCreateBuffer(allocator, &ci, &pool_ci, &handle, &allocation, &alloc_info) ==
           VK_SUCCESS;
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const {
    const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
    // MESA will do memcpy() if not marked as host cached, so just force mark it for most buffers
    auto const anv_flags = (usage == MemoryUsage::Stream
        && device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
        ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0;
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage) |
                 PackStrategy(),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
        .memoryTypeBits = usage == MemoryUsage::Stream ? 0u : valid_memory_types,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };

    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    VmaAllocation allocation{};
    VkMemoryPropertyFlags property_flags{};

    if (!CreateBufferInSeparatePool(ci, usage, alloc_ci, handle, allocation, alloc_info)) {
        VkResult result = vmaCreateBuffer(allocator, &ci, &alloc_ci, &handle, &allocation,
                                          &alloc_info);
        // Staging buffers live in system memory, where the budget is the OS's soft hint: going
        // over it pages, refusing aborted the emulator (a texture upload in Ocarina's town
        // square after 36 minutes). Device-local buffers keep the budget.
        if ((result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) &&
            usage != MemoryUsage::DeviceLocal) {
            VmaAllocationCreateInfo over_budget = alloc_ci;
            over_budget.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            result = vmaCreateBuffer(allocator, &ci, &over_budget, &handle, &allocation,
                                     &alloc_info);
            LOG_WARNING(Render_Vulkan,
                        "Buffer of {} KiB (usage {}) over the memory budget: {}", ci.size >> 10,
                        static_cast<u32>(usage), result == VK_SUCCESS ? "made anyway" : "failed");
            static std::atomic<u32> detailed{};
            if (detailed.fetch_add(1, std::memory_order_relaxed) < RefusalDetailEvents) {
                LOG_WARNING(Render_Vulkan, "  {}; {}",
                            result == VK_SUCCESS
                                ? "placed in " + PlacementText(allocator, alloc_info.memoryType)
                                : std::string{"not placed"},
                            HeapBudgetsText(allocator));
            }
        }
        vk::Check(result);
    }
    vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);

    // Log GPU memory allocation for buffers
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            property_flags
        );
    }

    u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
    const std::span<u8> mapped_data = data ? std::span<u8>{data, ci.size} : std::span<u8>{};
    const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    return vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data, is_coherent,
                      location, device.GetDispatchLoader());
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                         VkDeviceSize min_alignment) const {
    if (min_alignment <= 1) {
        return CreateBuffer(ci, usage);
    }
    return CreateAlignedBuffer(ci, usage, min_alignment, true);
}

vk::Buffer MemoryAllocator::CreateBufferOverBudget(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                                   VkDeviceSize min_alignment) const {
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < RefusalDetailEvents) {
        LOG_WARNING(Render_Vulkan, "Buffer of {} KiB (usage {}) goes over the memory budget; {}",
                    ci.size >> 10, static_cast<u32>(usage), HeapBudgetsText(allocator));
    }
    return CreateAlignedBuffer(ci, usage, min_alignment > 1 ? min_alignment : VkDeviceSize{1},
                               false);
}

vk::Buffer MemoryAllocator::CreateAlignedBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                                VkDeviceSize min_alignment,
                                                bool within_budget) const {
    const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
    VkMemoryPropertyFlags anv_flags = 0;
    if (usage == MemoryUsage::Stream &&
        device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA) {
        anv_flags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
    u32 memory_type_bits = valid_memory_types;
    if (usage == MemoryUsage::Stream) {
        memory_type_bits = 0u;
    }
    VmaAllocationCreateFlags budget_flag = 0;
    if (within_budget) {
        budget_flag = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
    }
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = budget_flag | MemoryUsageVmaFlags(usage) | PackStrategy(),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
        .memoryTypeBits = memory_type_bits,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };

    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    VmaAllocation allocation{};
    VkMemoryPropertyFlags property_flags{};

    vk::Check(vmaCreateBufferWithAlignment(allocator, &ci, &alloc_ci, min_alignment, &handle,
                                           &allocation, &alloc_info));
    vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);

    u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
    std::span<u8> mapped_data{};
    if (data) {
        mapped_data = std::span<u8>{data, ci.size};
    }
    const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    return vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data, is_coherent,
                      location, device.GetDispatchLoader());
}

MemoryCommit MemoryAllocator::Commit(const VkMemoryRequirements &reqs, MemoryUsage usage)
{
    const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
    const auto vma_usage = MemoryUsageVma(usage);
    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage) | PackStrategy();
    ci.usage = vma_usage;
    ci.memoryTypeBits = reqs.memoryTypeBits & valid_memory_types;
    ci.requiredFlags = 0;
    ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);

    VmaAllocation a{};
    VmaAllocationInfo info{};

    VkResult res = vmaAllocateMemory(allocator, &reqs, &ci, &a, &info);

    if (res != VK_SUCCESS) {
        // Relax 1: drop budget constraint
        auto ci2 = ci;
        ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaAllocateMemory(allocator, &reqs, &ci2, &a, &info);

        // Relax 2: if we preferred DEVICE_LOCAL, drop that preference
        if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            auto ci3 = ci2;
            ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            res = vmaAllocateMemory(allocator, &reqs, &ci3, &a, &info);
        }
    }

    vk::Check(res);
    return MemoryCommit(allocator, a, info);
}

MemoryCommit MemoryAllocator::Commit(const vk::Buffer &buffer, MemoryUsage usage) {
    const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
    // Allocate memory appropriate for this buffer automatically
    const auto vma_usage = MemoryUsageVma(usage);

    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage) | PackStrategy();
    ci.usage = vma_usage;
    ci.requiredFlags = 0;
    ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);
    ci.pool = VK_NULL_HANDLE;
    ci.pUserData = nullptr;
    ci.priority = 0.0f;

    const VkBuffer raw = *buffer;

    VmaAllocation a{};
    VmaAllocationInfo info{};

    // Let VMA infer memory requirements from the buffer
    VkResult res = vmaAllocateMemoryForBuffer(allocator, raw, &ci, &a, &info);

    if (res != VK_SUCCESS) {
        auto ci2 = ci;
        ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaAllocateMemoryForBuffer(allocator, raw, &ci2, &a, &info);

        if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            auto ci3 = ci2;
            ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            res = vmaAllocateMemoryForBuffer(allocator, raw, &ci3, &a, &info);
        }
    }

    vk::Check(res);
    vk::Check(vmaBindBufferMemory2(allocator, a, 0, raw, nullptr));
    return MemoryCommit(allocator, a, info);
}

#ifdef _WIN32
namespace {
/// The DXGI adapter of the Vulkan device (matched by LUID), or null. Looked up again when the
/// device changes (another GPU picked between sessions).
IDXGIAdapter3* FindDxgiAdapter(const Device& device) {
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;
    device.GetPhysical().GetProperties2(properties);
    if (!id.deviceLUIDValid) {
        return nullptr;
    }
    LUID luid{};
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    static LUID cached_luid{};
    static IDXGIAdapter3* cached{};
    if (cached != nullptr && cached_luid.LowPart == luid.LowPart &&
        cached_luid.HighPart == luid.HighPart) {
        return cached;
    }
    if (cached != nullptr) {
        cached->Release();
        cached = nullptr;
    }
    cached_luid = luid;
    const HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    if (dxgi == nullptr) {
        return nullptr;
    }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create =
        reinterpret_cast<CreateFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")));
    IDXGIFactory4* factory{};
    if (create == nullptr ||
        FAILED(create(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) {
        return nullptr;
    }
    if (FAILED(factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter3),
                                          reinterpret_cast<void**>(&cached)))) {
        cached = nullptr;
    }
    factory->Release();
    return cached;
}
} // Anonymous namespace
#endif

void LogMemoryTrace(const Device& device) {
    const Common::ProcessMemoryUsage process = Common::GetProcessMemoryUsage();
    LOG_INFO(Render_Vulkan,
             "Memory trace: process commit {} MiB, working set {} MiB (private {} MiB), system "
             "commit {} of {} MiB",
             process.private_bytes >> 20, process.working_set >> 20,
             process.private_working_set >> 20,
             process.system_commit >> 20, process.system_commit_limit >> 20);
#ifdef _WIN32
    // This process's use of the card's memory (local) and of system memory for the GPU
    // (non-local: host memory types, and what the OS moved out of video memory).
    IDXGIAdapter3* const adapter = FindDxgiAdapter(device);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    DXGI_QUERY_VIDEO_MEMORY_INFO non_local{};
    if (adapter != nullptr &&
        SUCCEEDED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
        SUCCEEDED(
            adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local))) {
        LOG_INFO(Render_Vulkan,
                 "Memory trace: DXGI local {} of {} MiB, non-local {} of {} MiB",
                 local.CurrentUsage >> 20, local.Budget >> 20, non_local.CurrentUsage >> 20,
                 non_local.Budget >> 20);
    }
#endif
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    std::string heaps;
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        const VmaBudget& budget = budgets[heap];
        fmt::format_to(std::back_inserter(heaps), "{}{} ({}): driver {} of {} MiB, VMA {}/{} MiB",
                       heap == 0 ? "" : "; ", heap,
                       (properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
                           ? "local"
                           : "host",
                       budget.usage >> 20, budget.budget >> 20,
                       budget.statistics.allocationBytes >> 20,
                       budget.statistics.blockBytes >> 20);
    }
    LOG_INFO(Render_Vulkan, "Memory trace: heaps {}", heaps);
    // The heaps' VMA numbers above are the main allocator's; with FRITCHII_MEMSTAGE_SPLIT_ALLOC
    // the buffer allocator's memory types get a line of their own.
    const auto log_types = [&](VmaAllocator owner, const char* label) {
        VmaTotalStatistics stats{};
        vmaCalculateStatistics(owner, &stats);
        std::string types;
        for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
            const VmaStatistics& s = stats.memoryType[type].statistics;
            if (s.blockCount == 0) {
                continue;
            }
            fmt::format_to(std::back_inserter(types),
                           "{}{} (heap {}, flags {:#x}): {} allocs {}/{} MiB",
                           types.empty() ? "" : "; ", type, properties->memoryTypes[type].heapIndex,
                           static_cast<u32>(properties->memoryTypes[type].propertyFlags),
                           s.allocationCount, s.allocationBytes >> 20, s.blockBytes >> 20);
        }
        LOG_INFO(Render_Vulkan, "Memory trace: {} {}", label, types);
    };
    log_types(allocator, "types");
    // Only its user may walk the buffer allocator (memory stage spec 5.3).
    if (device.HasSplitAllocators() && VideoCommon::MemoryStageLink::IsCacheUser()) {
        log_types(device.GetBufferAllocator(), "buffer allocator types");
    }
}

HeapTotals GetHeapTotals(const Device& device) {
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    HeapTotals totals{};
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        const VmaBudget& budget = budgets[heap];
        if ((properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            totals.local_usage += budget.usage;
            totals.local_budget += budget.budget;
            totals.local_blocks += budget.statistics.blockBytes;
        } else {
            totals.host_usage += budget.usage;
            totals.host_budget += budget.budget;
            totals.host_blocks += budget.statistics.blockBytes;
        }
    }
    if (device.HasSplitAllocators()) {
        // FRITCHII_MEMSTAGE_SPLIT_ALLOC: the buffer allocator's blocks too. Its usage only
        // without VK_EXT_memory_budget: with it, usage is the driver's for the whole process.
        // At FRITCHII_MEMSTAGE_THREAD=1 as its user published them (memory stage spec 5.3).
        device.BufferHeapBudgets(budgets.data());
        const bool add_usage = !device.CanReportMemoryUsage();
        for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
            const VmaBudget& budget = budgets[heap];
            const u64 usage = add_usage ? budget.usage : 0;
            if ((properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                totals.local_usage += usage;
                totals.local_blocks += budget.statistics.blockBytes;
            } else {
                totals.host_usage += usage;
                totals.host_blocks += budget.statistics.blockBytes;
            }
        }
    }
    return totals;
}

void LogMemoryStatistics(const Device& device) {
    const VmaAllocator allocator = device.GetAllocator();
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator, &stats);
    for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
        const VmaStatistics& s = stats.memoryType[type].statistics;
        if (s.blockCount == 0) {
            continue;
        }
        LOG_DEBUG(Render_Vulkan, "VMA type {} (heap {}, flags {:#x}): {} blocks {} MiB, {} allocations {} MiB",
                 type, properties->memoryTypes[type].heapIndex,
                 static_cast<u32>(properties->memoryTypes[type].propertyFlags), s.blockCount,
                 s.blockBytes >> 20, s.allocationCount, s.allocationBytes >> 20);
    }
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        LOG_INFO(Render_Vulkan, "VMA heap {}: usage {} MiB of budget {} MiB, VMA blocks {} MiB",
                 heap, budgets[heap].usage >> 20, budgets[heap].budget >> 20,
                 budgets[heap].statistics.blockBytes >> 20);
    }
    if (device.HasSplitAllocators()) {
        device.BufferHeapBudgets(budgets.data());
        for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
            LOG_INFO(Render_Vulkan, "VMA heap {}: buffer allocator blocks {} MiB ({} MiB used)",
                     heap, budgets[heap].statistics.blockBytes >> 20,
                     budgets[heap].statistics.allocationBytes >> 20);
        }
    }
}

namespace {

/// A custom VMA pool of one allocator, with its memory type and a name for the log.
struct NamedPool {
    VmaPool pool;
    u32 type;
    const char* name;
};

/// Every custom pool `owner` has: the depth attachment pools (EDEN_NX2_DEPTH_POOL), the colour
/// target pools (FRITCHII_KILL_RT_POOL) and the device's separate pools (sparse pages, buffers).
std::vector<NamedPool> PoolsOf(const Device& device, VmaAllocator owner) {
    std::vector<NamedPool> pools;
    {
        std::scoped_lock lock{g_depth_pool_mutex};
        for (const DepthPoolEntry& entry : g_depth_pools) {
            if (entry.owner == owner) {
                pools.push_back(
                    {entry.pool, entry.type, entry.is_small ? "small depth pool" : "depth pool"});
            }
        }
    }
    {
        std::scoped_lock lock{g_kill_pool_mutex};
        for (const KillPoolEntry& entry : g_kill_pools) {
            if (entry.owner == owner) {
                pools.push_back({entry.pool, entry.type, "colour pool"});
            }
        }
    }
    for (u32 type = 0; type < VK_MAX_MEMORY_TYPES; ++type) {
        if (const VmaPool pool =
                device.PeekSeparatePool(owner, Device::SeparatePool::SparsePages, type)) {
            pools.push_back({pool, type, "sparse pool"});
        }
        if (const VmaPool pool =
                device.PeekSeparatePool(owner, Device::SeparatePool::DeviceBuffers, type)) {
            pools.push_back({pool, type, "buffer pool"});
        }
    }
    return pools;
}

/// FRITCHII_VRAM_TRIM: a 256-byte allocation in an existing block of the vector (the pool, or
/// the memory type's default blocks), freed at once. VMA keeps one empty block per vector; it
/// deletes the last (emptiest) block on a free when that block is empty and another free left
/// the vector with an empty block before (VmaBlockVector::Free). NEVER_ALLOCATE: no new memory
/// if no block has room. Best fit, so the probe lands in the fullest block with room (VMA tries
/// blocks fullest first), never in the empty one while another has 256 bytes.
void ProbeEmptyBlock(VmaAllocator owner, VmaPool pool, u32 type) {
    const VkMemoryRequirements requirements{
        .size = 256,
        .alignment = 256,
        .memoryTypeBits = 1U << type,
    };
    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT |
               VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;
    ci.usage = VMA_MEMORY_USAGE_UNKNOWN;
    ci.memoryTypeBits = 1U << type;
    ci.pool = pool;
    VmaAllocation probe{};
    if (vmaAllocateMemory(owner, &requirements, &ci, &probe, nullptr) == VK_SUCCESS) {
        vmaFreeMemory(owner, probe);
    }
}

} // Anonymous namespace

std::string VramSlackText(const Device& device, VmaAllocator owner, u32 heap) {
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(owner, &properties);
    if (heap >= properties->memoryHeapCount) {
        return "none";
    }
    VmaTotalStatistics total{};
    vmaCalculateStatistics(owner, &total);
    const VmaDetailedStatistics& all = total.memoryHeap[heap];
    u64 pool_block_bytes = 0;
    u64 pool_allocation_bytes = 0;
    u64 pool_holes = 0;
    u32 pool_blocks = 0;
    std::string pools;
    for (const NamedPool& named : PoolsOf(device, owner)) {
        if (named.type >= properties->memoryTypeCount ||
            properties->memoryTypes[named.type].heapIndex != heap) {
            continue;
        }
        VmaDetailedStatistics stats{};
        vmaCalculatePoolStatistics(owner, named.pool, &stats);
        const VmaStatistics& s = stats.statistics;
        pool_block_bytes += s.blockBytes;
        pool_allocation_bytes += s.allocationBytes;
        pool_holes += stats.unusedRangeCount;
        pool_blocks += s.blockCount;
        if (s.blockCount == 0) {
            continue;
        }
        fmt::format_to(std::back_inserter(pools),
                       ", {} (type {}) {} MiB ({} MiB in {} blocks, {} holes, largest {} MiB)",
                       named.name, named.type,
                       (s.blockBytes - (std::min)(s.blockBytes, s.allocationBytes)) >> 20,
                       s.blockBytes >> 20, s.blockCount, stats.unusedRangeCount,
                       stats.unusedRangeSizeMax >> 20);
    }
    // The memory types' own blocks and their dedicated allocations (a dedicated allocation is a
    // memory object of its own size: it adds no unused space).
    const u64 blocks = all.statistics.blockBytes - (std::min)(all.statistics.blockBytes,
                                                              pool_block_bytes);
    const u64 used = all.statistics.allocationBytes -
                     (std::min)(all.statistics.allocationBytes, pool_allocation_bytes);
    const u32 objects = all.statistics.blockCount - (std::min)(all.statistics.blockCount,
                                                               pool_blocks);
    return fmt::format("default {} MiB ({} MiB in {} memory objects, {} holes, largest on the "
                       "heap {} MiB){}",
                       (blocks - (std::min)(blocks, used)) >> 20, blocks >> 20, objects,
                       all.unusedRangeCount - (std::min<u64>)(all.unusedRangeCount, pool_holes),
                       all.unusedRangeSizeMax >> 20, pools);
}

u64 TrimEmptyBlocks(const Device& device, VmaAllocator owner, u32 heap) {
    const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{owner};
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(owner, &properties);
    if (heap >= properties->memoryHeapCount) {
        return 0;
    }
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> before{};
    vmaGetHeapBudgets(owner, before.data());
    for (u32 type = 0; type < properties->memoryTypeCount; ++type) {
        if (properties->memoryTypes[type].heapIndex == heap) {
            ProbeEmptyBlock(owner, VK_NULL_HANDLE, type);
        }
    }
    for (const NamedPool& named : PoolsOf(device, owner)) {
        if (named.type < properties->memoryTypeCount &&
            properties->memoryTypes[named.type].heapIndex == heap) {
            ProbeEmptyBlock(owner, named.pool, named.type);
        }
    }
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> after{};
    vmaGetHeapBudgets(owner, after.data());
    const u64 had = before[heap].statistics.blockBytes;
    const u64 has = after[heap].statistics.blockBytes;
    return had > has ? had - has : 0;
}

std::string DepthFitText() {
    const int mode = DepthFitMode();
    const u64 placed = g_fit_placed.load(std::memory_order_relaxed);
    if (mode == 0 && placed == 0) {
        return {};
    }
    return fmt::format("depth fit (FRITCHII_DEPTH_POOL_FIT={}): {} placed ({} in the small pool), "
                       "{} handed back, padding of live images {} MiB",
                       mode, placed, g_fit_small.load(std::memory_order_relaxed),
                       g_fit_fallbacks.load(std::memory_order_relaxed),
                       vk::g_image_padding_bytes.load(std::memory_order_relaxed) >> 20);
}

namespace {

/// FRITCHII_VRAM_HOLES: size classes of the histograms.
constexpr std::array<u64, 5> HoleClassLimits{u64{64} << 10, u64{256} << 10, u64{1} << 20,
                                             u64{4} << 20, u64{16} << 20};
constexpr std::array<const char*, 6> HoleClassNames{"<64K", "<256K", "<1M",
                                                    "<4M",  "<16M",  ">=16M"};

size_t HoleClass(u64 size) {
    size_t index = 0;
    while (index < HoleClassLimits.size() && size >= HoleClassLimits[index]) {
        ++index;
    }
    return index;
}

/// One block vector's numbers (a memory type's default blocks, or a named pool).
struct HoleStats {
    std::string label;
    u64 blocks{};
    u64 block_bytes{};
    std::array<u64, 6> holes{};
    std::array<u64, 6> hole_bytes{};
    std::array<u64, 6> sizes{}; ///< all allocations by size
    std::array<u64, 6> size_bytes{};
    u64 sparse_blocks{}; ///< blocks at least half free
    u64 sparse_block_bytes{};
    u64 pinners{}; ///< allocations in those blocks
    u64 pinner_bytes{};
    std::array<u64, 4> pinner_kinds{}; ///< depth, colour, other image, buffer or untyped
    std::array<u64, 4> pinner_kind_bytes{};
    std::array<u64, 6> pinner_sizes{};
    std::array<u64, 5> pinner_ages{}; ///< <10 s, <1 min, <5 min, older, untagged
};

struct SubAllocation {
    bool free{};
    bool image{};
    u64 size{};
    u64 usage{};
    u64 data{};
};

std::string Histogram(const std::array<u64, 6>& counts, const std::array<u64, 6>& bytes) {
    std::string text;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] == 0) {
            continue;
        }
        fmt::format_to(std::back_inserter(text), "{}{} {}/{} MiB", text.empty() ? "" : ", ",
                       HoleClassNames[i], counts[i], bytes[i] >> 20);
    }
    return text.empty() ? std::string{"none"} : text;
}

bool IsJsonSpace(char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

/// A JSON string starting at text[start] (the opening quote); `end` is set past the closing one.
std::string ReadJsonString(std::string_view text, size_t start, size_t& end) {
    std::string out;
    size_t j = start + 1;
    while (j < text.size() && text[j] != '"') {
        if (text[j] == '\\' && j + 1 < text.size()) {
            ++j;
        }
        out.push_back(text[j]);
        ++j;
    }
    end = j + 1;
    return out;
}

} // Anonymous namespace

std::string VramHolesText(VmaAllocator owner, u32 heap) {
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(owner, &properties);
    char* json = nullptr;
    vmaBuildStatsString(owner, &json, VK_TRUE);
    if (json == nullptr) {
        return "no map";
    }
    const std::string_view text{json};
    const u64 now = ImageStamp();
    std::vector<HoleStats> vectors;
    size_t current = SIZE_MAX; ///< index in `vectors`
    bool custom = false;
    bool dedicated = false;
    bool on_heap = false;
    u32 type = 0;
    u64 block_size = 0;
    bool in_block = false;
    std::vector<SubAllocation> block;
    SubAllocation pending{};
    bool has_pending = false;
    const auto vector_for = [&](const std::string& label) {
        for (size_t index = 0; index < vectors.size(); ++index) {
            if (vectors[index].label == label) {
                return index;
            }
        }
        vectors.push_back(HoleStats{});
        vectors.back().label = label;
        return vectors.size() - 1;
    };
    const auto flush_pending = [&] {
        if (has_pending && in_block && !dedicated && on_heap && current != SIZE_MAX) {
            block.push_back(pending);
        }
        has_pending = false;
        pending = {};
    };
    const auto flush_block = [&] {
        flush_pending();
        if (in_block && on_heap && current != SIZE_MAX && block_size != 0) {
            HoleStats& stats = vectors[current];
            ++stats.blocks;
            stats.block_bytes += block_size;
            u64 free_bytes = 0;
            for (const SubAllocation& sub : block) {
                const size_t size_class = HoleClass(sub.size);
                if (sub.free) {
                    free_bytes += sub.size;
                    ++stats.holes[size_class];
                    stats.hole_bytes[size_class] += sub.size;
                } else {
                    ++stats.sizes[size_class];
                    stats.size_bytes[size_class] += sub.size;
                }
            }
            if (free_bytes * 2 >= block_size) {
                ++stats.sparse_blocks;
                stats.sparse_block_bytes += block_size;
                for (const SubAllocation& sub : block) {
                    if (sub.free) {
                        continue;
                    }
                    ++stats.pinners;
                    stats.pinner_bytes += sub.size;
                    ++stats.pinner_sizes[HoleClass(sub.size)];
                    size_t kind = 3;
                    if (sub.image) {
                        kind = (sub.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0 ? 0
                               : (sub.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0       ? 1
                                                                                              : 2;
                    }
                    ++stats.pinner_kinds[kind];
                    stats.pinner_kind_bytes[kind] += sub.size;
                    size_t age = 4;
                    if ((sub.data & vk::ImageUserDataTag) != 0) {
                        const u64 stamp = sub.data & 0xFFFF'FFFFULL;
                        const u64 seconds = (now >= stamp ? now - stamp : 0) / 10;
                        age = seconds < 10 ? 0 : seconds < 60 ? 1 : seconds < 300 ? 2 : 3;
                    }
                    ++stats.pinner_ages[age];
                }
            }
        }
        block.clear();
        in_block = false;
        block_size = 0;
    };
    // VMA's JSON (VmaJsonWriter): "Key": value pairs in a fixed order, so a scan for keys is
    // enough: DefaultPools / CustomPools, "Type N" per memory type, "Name" per custom pool,
    // "TotalBytes" per block, then its suballocations ("Offset", "Type", "Size", "Usage",
    // "CustomData"), then the vector's "DedicatedAllocations" (left out: no unused space).
    size_t i = 0;
    while (true) {
        const size_t quote = text.find('"', i);
        if (quote == std::string_view::npos) {
            break;
        }
        size_t after = 0;
        const std::string key = ReadJsonString(text, quote, after);
        size_t j = after;
        while (j < text.size() && IsJsonSpace(text[j])) {
            ++j;
        }
        if (j >= text.size() || text[j] != ':') {
            i = after;
            continue;
        }
        ++j;
        while (j < text.size() && IsJsonSpace(text[j])) {
            ++j;
        }
        std::string string_value;
        u64 number = 0;
        bool is_string = false;
        bool is_number = false;
        if (j < text.size() && text[j] == '"') {
            string_value = ReadJsonString(text, j, i);
            is_string = true;
        } else if (j < text.size() && text[j] >= '0' && text[j] <= '9') {
            size_t k = j;
            while (k < text.size() && text[k] >= '0' && text[k] <= '9') {
                number = number * 10 + static_cast<u64>(text[k] - '0');
                ++k;
            }
            is_number = true;
            i = k;
        } else {
            i = j + 1; // an object or an array opens, or a literal
        }
        if (key == "DefaultPools" || key == "CustomPools") {
            flush_block();
            custom = key == "CustomPools";
            current = SIZE_MAX;
        } else if (key.starts_with("Type ") && !is_string && !is_number) {
            flush_block();
            type = static_cast<u32>(std::strtoul(key.c_str() + 5, nullptr, 10));
            on_heap = type < properties->memoryTypeCount &&
                      properties->memoryTypes[type].heapIndex == heap;
            dedicated = false;
            current = custom ? SIZE_MAX : vector_for(fmt::format("default (type {})", type));
        } else if (key == "Name" && is_string && custom) {
            flush_block();
            const size_t dash = string_value.find(" - ");
            const std::string name =
                dash != std::string::npos ? string_value.substr(dash + 3) : "unnamed pool";
            current = vector_for(fmt::format("{} (type {})", name, type));
            dedicated = false;
        } else if (key == "Blocks") {
            flush_block();
            dedicated = false;
        } else if (key == "DedicatedAllocations") {
            flush_block();
            dedicated = true;
        } else if (key == "TotalBytes" && is_number) {
            flush_block();
            if (!dedicated) {
                in_block = true;
                block_size = number;
            }
        } else if (key == "Offset" && is_number) {
            flush_pending();
            has_pending = true;
        } else if (key == "Type" && is_string && has_pending) {
            pending.free = string_value == "FREE";
            pending.image = string_value.starts_with("IMAGE");
        } else if (key == "Size" && is_number && has_pending) {
            pending.size = number;
        } else if (key == "Usage" && is_number && has_pending) {
            pending.usage = number;
        } else if (key == "CustomData" && is_string && has_pending) {
            pending.data = std::strtoull(string_value.c_str(), nullptr, 16);
        }
    }
    flush_block();
    vmaFreeStatsString(owner, json);
    std::string out;
    for (const HoleStats& stats : vectors) {
        if (stats.blocks == 0) {
            continue;
        }
        fmt::format_to(
            std::back_inserter(out),
            "{}{}: {} blocks {} MiB; holes {}; allocations {}; blocks at least half free {} ({} "
            "MiB) held by {} allocations {} MiB: depth {}/{} MiB, colour {}/{} MiB, other images "
            "{}/{} MiB, buffers and untyped {}/{} MiB; their sizes <64K {}, <256K {}, <1M {}, "
            "<4M {}, <16M {}, >=16M {}; their ages <10 s {}, <1 min {}, <5 min {}, older {}, "
            "unknown {}",
            out.empty() ? "" : " | ", stats.label, stats.blocks, stats.block_bytes >> 20,
            Histogram(stats.holes, stats.hole_bytes), Histogram(stats.sizes, stats.size_bytes),
            stats.sparse_blocks, stats.sparse_block_bytes >> 20, stats.pinners,
            stats.pinner_bytes >> 20, stats.pinner_kinds[0], stats.pinner_kind_bytes[0] >> 20,
            stats.pinner_kinds[1], stats.pinner_kind_bytes[1] >> 20, stats.pinner_kinds[2],
            stats.pinner_kind_bytes[2] >> 20, stats.pinner_kinds[3],
            stats.pinner_kind_bytes[3] >> 20, stats.pinner_sizes[0], stats.pinner_sizes[1],
            stats.pinner_sizes[2], stats.pinner_sizes[3], stats.pinner_sizes[4],
            stats.pinner_sizes[5], stats.pinner_ages[0], stats.pinner_ages[1],
            stats.pinner_ages[2], stats.pinner_ages[3], stats.pinner_ages[4]);
    }
    return out.empty() ? std::string{"no blocks on the heap"} : out;
}

} // namespace Vulkan
