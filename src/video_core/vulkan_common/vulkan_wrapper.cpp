// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include "video_core/memory_stage_link.h"
#include "video_core/vulkan_common/vk_enum_string_helper.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_debug_callback.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::vk {

namespace {

template <typename Func>
void SortPhysicalDevices(std::vector<VkPhysicalDevice>& devices, const InstanceDispatch& dld,
                         Func&& func) {
    // Calling GetProperties calls Vulkan more than needed. But they are supposed to be cheap
    // functions.
    std::stable_sort(devices.begin(), devices.end(),
                     [&dld, &func](VkPhysicalDevice lhs, VkPhysicalDevice rhs) {
                         return func(vk::PhysicalDevice(lhs, dld).GetProperties(),
                                     vk::PhysicalDevice(rhs, dld).GetProperties());
                     });
}

void SortPhysicalDevicesPerVendor(std::vector<VkPhysicalDevice>& devices,
                                  const InstanceDispatch& dld,
                                  std::initializer_list<u32> vendor_ids) {
    for (auto it = vendor_ids.end(); it != vendor_ids.begin();) {
        --it;
        SortPhysicalDevices(devices, dld, [id = *it](const auto& lhs, const auto& rhs) {
            return lhs.vendorID == id && rhs.vendorID != id;
        });
    }
}

bool IsMicrosoftDozen(const char* device_name) {
    return std::strstr(device_name, "Microsoft") != nullptr;
}

void SortPhysicalDevices(std::vector<VkPhysicalDevice>& devices, const InstanceDispatch& dld) {
    // Sort by name, this will set a base and make GPUs with higher numbers appear first
    // (e.g. GTX 1650 will intentionally be listed before a GTX 1080).
    SortPhysicalDevices(devices, dld, [](const auto& lhs, const auto& rhs) {
        return std::string_view{lhs.deviceName} > std::string_view{rhs.deviceName};
    });
    // Prefer discrete over non-discrete
    SortPhysicalDevices(devices, dld, [](const auto& lhs, const auto& rhs) {
        return lhs.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
               rhs.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    });
    // Prefer Nvidia over AMD, AMD over Intel, Intel over the rest.
    SortPhysicalDevicesPerVendor(devices, dld, {0x10DE, 0x1002, 0x8086});
    // Demote Microsoft's Dozen devices to the bottom.
    SortPhysicalDevices(
        devices, dld,
        [](const VkPhysicalDeviceProperties& lhs, const VkPhysicalDeviceProperties& rhs) {
            return IsMicrosoftDozen(rhs.deviceName) && !IsMicrosoftDozen(lhs.deviceName);
        });
    // Demote software rasterizers (llvmpipe/lavapipe, SwiftShader) below every real GPU.
    // The name sort above can put "llvmpipe" ahead of an integrated GPU such as
    // "Adreno (TM) 830", so device 0 (the default) becomes the CPU renderer whenever
    // a software ICD is visible.
    SortPhysicalDevices(
        devices, dld,
        [](const VkPhysicalDeviceProperties& lhs, const VkPhysicalDeviceProperties& rhs) {
            return lhs.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU &&
                   rhs.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
        });
}

template <typename T>
bool Proc(T& result, const InstanceDispatch& dld, const char* proc_name,
          VkInstance instance = nullptr) noexcept {
    result = reinterpret_cast<T>(dld.vkGetInstanceProcAddr(instance, proc_name));
    return result != nullptr;
}

template <typename T>
void Proc(T& result, const DeviceDispatch& dld, const char* proc_name, VkDevice device) noexcept {
    result = reinterpret_cast<T>(dld.vkGetDeviceProcAddr(device, proc_name));
}

void Load(VkDevice device, DeviceDispatch& dld) noexcept {
#define X(name) Proc(dld.name, dld, #name, device)
    X(vkAcquireNextImageKHR);
    X(vkAllocateCommandBuffers);
    X(vkAllocateDescriptorSets);
    X(vkAllocateMemory);
    X(vkBeginCommandBuffer);
    X(vkBindBufferMemory);
    X(vkBindImageMemory);
    X(vkCmdBeginConditionalRenderingEXT);
    X(vkCmdBeginQuery);
    X(vkCmdBeginRenderPass);
    X(vkCmdBeginTransformFeedbackEXT);
    X(vkCmdBeginDebugUtilsLabelEXT);
    X(vkCmdBindDescriptorSets);
    X(vkCmdBindIndexBuffer);
    X(vkCmdBindPipeline);
    X(vkCmdBindTransformFeedbackBuffersEXT);
    X(vkCmdBindVertexBuffers);
    X(vkCmdBlitImage);
    X(vkCmdClearAttachments);
    X(vkCmdClearColorImage);
    X(vkCmdClearDepthStencilImage);
    X(vkCmdCopyBuffer);
    X(vkCmdCopyBufferToImage);
    X(vkCmdCopyImage);
    X(vkCmdCopyImageToBuffer);
    X(vkCmdCopyQueryPoolResults);
    X(vkCmdDispatch);
    X(vkCmdDispatchIndirect);
    X(vkCmdDraw);
    X(vkCmdDrawIndexed);
    X(vkCmdDrawIndirect);
    X(vkCmdDrawIndexedIndirect);
    X(vkCmdDrawIndirectCount);
    X(vkCmdDrawIndexedIndirectCount);
    X(vkCmdDrawIndirectByteCountEXT);
    X(vkCmdDrawMeshTasksEXT);
    X(vkCmdDrawMeshTasksIndirectEXT);
    X(vkCmdEndConditionalRenderingEXT);
    X(vkCmdEndQuery);
    X(vkCmdEndRenderPass);
    X(vkCmdEndTransformFeedbackEXT);
    X(vkCmdEndDebugUtilsLabelEXT);
    X(vkCmdFillBuffer);
    X(vkCmdPipelineBarrier);
    X(vkCmdPipelineBarrier2);
    X(vkCmdPushConstants);
    X(vkCmdPushDescriptorSetWithTemplateKHR);
    X(vkCmdSetBlendConstants);
    X(vkCmdSetDepthBias);
    X(vkCmdSetDepthBias2EXT);
    X(vkCmdSetDepthBounds);
    X(vkCmdSetEvent);
    X(vkCmdSetScissor);
    X(vkCmdSetStencilCompareMask);
    X(vkCmdSetStencilReference);
    X(vkCmdSetStencilWriteMask);
    X(vkCmdSetViewport);
    X(vkCmdWaitEvents);
    X(vkCmdBindVertexBuffers2EXT);
    X(vkCmdSetCullModeEXT);
    X(vkCmdSetDepthBoundsTestEnableEXT);
    X(vkCmdSetDepthCompareOpEXT);
    X(vkCmdSetDepthTestEnableEXT);
    X(vkCmdSetDepthWriteEnableEXT);
    X(vkCmdSetPrimitiveRestartEnableEXT);
    X(vkCmdSetRasterizerDiscardEnableEXT);
    X(vkCmdSetAlphaToCoverageEnableEXT);
    X(vkCmdSetAlphaToOneEnableEXT);
    X(vkCmdSetConservativeRasterizationModeEXT);
    X(vkCmdSetLineRasterizationModeEXT);
    X(vkCmdSetLineStippleEnableEXT);
    X(vkCmdSetDepthBiasEnableEXT);
    X(vkCmdSetLogicOpEnableEXT);
    X(vkCmdSetDepthClampEnableEXT);
    X(vkCmdSetFrontFaceEXT);
    X(vkCmdSetLogicOpEXT);
    X(vkCmdSetPatchControlPointsEXT);
    X(vkCmdSetLineWidth);
    X(vkCmdSetPrimitiveTopologyEXT);
    X(vkCmdSetStencilOpEXT);
    X(vkCmdSetStencilTestEnableEXT);
    X(vkCmdSetVertexInputEXT);
    X(vkCmdSetColorWriteMaskEXT);
    X(vkCmdSetColorWriteEnableEXT);
    X(vkCmdSetColorBlendEnableEXT);
    X(vkCmdSetColorBlendEquationEXT);
    X(vkCmdResetQueryPool);
    X(vkCmdWriteTimestamp);
    X(vkCmdResolveImage);
    X(vkCreateBuffer);
    X(vkCreateBufferView);
    X(vkCreateCommandPool);
    X(vkCreateComputePipelines);
    X(vkCreateDescriptorPool);
    X(vkCreateDescriptorSetLayout);
    X(vkCreateDescriptorUpdateTemplate);
    X(vkCreateEvent);
    X(vkCreateFence);
    X(vkCreateFramebuffer);
    X(vkCreateGraphicsPipelines);
    X(vkCreateImage);
    X(vkCreateImageView);
    X(vkCreatePipelineCache);
    X(vkCreatePipelineLayout);
    X(vkCreateQueryPool);
    X(vkCreateRenderPass);
    X(vkCreateRenderPass2);
    X(vkCreateSampler);
    X(vkCreateSemaphore);
    X(vkCreateShaderModule);
    X(vkCreateSwapchainKHR);
    X(vkDestroyBuffer);
    X(vkDestroyBufferView);
    X(vkDestroyCommandPool);
    X(vkDestroyDescriptorPool);
    X(vkDestroyDescriptorSetLayout);
    X(vkDestroyDescriptorUpdateTemplate);
    X(vkDestroyEvent);
    X(vkDestroyFence);
    X(vkDestroyFramebuffer);
    X(vkDestroyImage);
    X(vkDestroyImageView);
    X(vkDestroyPipeline);
    X(vkCreatePipelineBinariesKHR);
    X(vkDestroyPipelineBinaryKHR);
    X(vkGetPipelineBinaryDataKHR);
    X(vkGetPipelineKeyKHR);
    X(vkReleaseCapturedPipelineDataKHR);
    X(vkDestroyPipelineCache);
    X(vkDestroyPipelineLayout);
    X(vkDestroyQueryPool);
    X(vkDestroyRenderPass);
    X(vkDestroySampler);
    X(vkDestroySemaphore);
    X(vkDestroyShaderModule);
    X(vkDestroySwapchainKHR);
    X(vkDeviceWaitIdle);
    X(vkEndCommandBuffer);
    X(vkFreeCommandBuffers);
    X(vkFreeDescriptorSets);
    X(vkFreeMemory);
    X(vkGetBufferMemoryRequirements2);
    X(vkGetDeviceQueue);
    X(vkGetEventStatus);
    X(vkGetFenceStatus);
    X(vkGetImageMemoryRequirements);
    X(vkGetImageSparseMemoryRequirements);
    X(vkGetPipelineCacheData);
    X(vkGetMemoryFdKHR);
#ifdef _WIN32
    X(vkGetMemoryWin32HandleKHR);
#endif
    X(vkGetQueryPoolResults);
    X(vkGetDeviceFaultInfoEXT);
    X(vkGetPipelineExecutablePropertiesKHR);
    X(vkWaitForPresentKHR);
    X(vkGetPipelineExecutableStatisticsKHR);
    X(vkGetSemaphoreCounterValue);
    X(vkMapMemory);
    X(vkQueueBindSparse);
    X(vkQueueSubmit);
    X(vkQueueSubmit2);
    X(vkResetFences);
    X(vkResetQueryPool);
    X(vkSetDebugUtilsObjectNameEXT);
    X(vkSetDebugUtilsObjectTagEXT);
    X(vkUnmapMemory);
    X(vkUpdateDescriptorSetWithTemplate);
    X(vkUpdateDescriptorSets);
    X(vkGetBufferDeviceAddress);
    X(vkGetDescriptorSetLayoutSizeEXT);
    X(vkGetDescriptorSetLayoutBindingOffsetEXT);
    X(vkGetDescriptorEXT);
    X(vkCmdBindDescriptorBuffersEXT);
    X(vkCmdSetDescriptorBufferOffsetsEXT);
    X(vkWaitForFences);
    X(vkWaitSemaphores);

    // Support for timeline semaphores is mandatory in Vulkan 1.2
    if (!dld.vkGetSemaphoreCounterValue) {
        Proc(dld.vkGetSemaphoreCounterValue, dld, "vkGetSemaphoreCounterValueKHR", device);
        Proc(dld.vkWaitSemaphores, dld, "vkWaitSemaphoresKHR", device);
    }

    // Support for host query reset is mandatory in Vulkan 1.2
    if (!dld.vkResetQueryPool) {
        Proc(dld.vkResetQueryPool, dld, "vkResetQueryPoolEXT", device);
    }

    // Support for draw indirect with count is optional in Vulkan 1.2
    if (!dld.vkCmdDrawIndirectCount) {
        Proc(dld.vkCmdDrawIndirectCount, dld, "vkCmdDrawIndirectCountKHR", device);
        Proc(dld.vkCmdDrawIndexedIndirectCount, dld, "vkCmdDrawIndexedIndirectCountKHR", device);
    }

    // Synchronization2 is core in Vulkan 1.3, otherwise requires VK_KHR_synchronization2
    if (!dld.vkCmdPipelineBarrier2) {
        Proc(dld.vkCmdPipelineBarrier2, dld, "vkCmdPipelineBarrier2KHR", device);
    }
    if (!dld.vkQueueSubmit2) {
        Proc(dld.vkQueueSubmit2, dld, "vkQueueSubmit2KHR", device);
    }

    if (!dld.vkCreateRenderPass2) {
        Proc(dld.vkCreateRenderPass2, dld, "vkCreateRenderPass2KHR", device);
    }
#undef X
}

template <typename T>
void SetObjectName(const DeviceDispatch* dld, VkDevice device, T handle, VkObjectType type,
                   const char* name) {
    const VkDebugUtilsObjectNameInfoEXT name_info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .pNext = nullptr,
        .objectType = type,
        .objectHandle = reinterpret_cast<u64>(handle),
        .pObjectName = name,
    };
    if (dld->vkSetDebugUtilsObjectNameEXT) {
        Check(dld->vkSetDebugUtilsObjectNameEXT(device, &name_info));
    }
}

} // Anonymous namespace

void NoteNullTarget(const char* command, const std::source_location& site) noexcept {
    static std::atomic<u64> skipped{};
    const u64 count = skipped.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 16 || count % 1024 == 0) {
        LOG_CRITICAL(Render_Vulkan,
                     "Null GPU target: {} (it would reach device address 0 and lose the device); "
                     "at {}:{} ({}); {} so far",
                     command, site.file_name(), site.line(), site.function_name(), count);
    }
}

bool Load(InstanceDispatch& dld) noexcept {
#define X(name) Proc(dld.name, dld, #name)
    return X(vkCreateInstance) && X(vkEnumerateInstanceExtensionProperties) &&
           X(vkEnumerateInstanceLayerProperties);
#undef X
}

bool Load(VkInstance instance, InstanceDispatch& dld) noexcept {
#define X(name) Proc(dld.name, dld, #name, instance)
    // These functions may fail to load depending on the enabled extensions.
    // Don't return a failure on these.
    X(vkCreateDebugUtilsMessengerEXT);
    X(vkCreateDebugReportCallbackEXT);
    X(vkDestroyDebugUtilsMessengerEXT);
    X(vkDestroyDebugReportCallbackEXT);
    X(vkDestroySurfaceKHR);
    X(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR);
    X(vkGetPhysicalDeviceFeatures2);
    X(vkGetPhysicalDeviceFormatProperties2);
    X(vkGetPhysicalDeviceProperties2);
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    X(vkGetPhysicalDeviceSurfaceFormatsKHR);
    X(vkGetPhysicalDeviceSurfacePresentModesKHR);
    X(vkGetPhysicalDeviceSurfaceSupportKHR);
    X(vkGetPhysicalDeviceToolProperties);
    X(vkGetSwapchainImagesKHR);
    X(vkQueuePresentKHR);

    return X(vkCreateDevice) && X(vkDestroyDevice) && X(vkDestroyDevice) &&
           X(vkEnumerateDeviceExtensionProperties) && X(vkEnumeratePhysicalDevices) &&
           X(vkGetDeviceProcAddr) && X(vkGetPhysicalDeviceFormatProperties) &&
           X(vkGetPhysicalDeviceMemoryProperties) && X(vkGetPhysicalDeviceMemoryProperties2) &&
           X(vkGetPhysicalDeviceProperties) && X(vkGetPhysicalDeviceQueueFamilyProperties);
#undef X
}

const char* Exception::what() const noexcept {
    return string_VkResult(result);
}

namespace {

// FRITCHII_RAM_VK_HOST_ALLOC: the driver's host allocations through allocation callbacks given at
// vkCreateInstance and vkCreateDevice (their children are created with null callbacks, which
// drivers serve from the parent's). The blocks come from a heap of our own and are counted, so the
// census names the driver's host memory, and what the driver frees goes back to that heap instead
// of staying in the driver's own reservations.
struct HostAllocState {
    std::array<std::atomic<u64>, 5> live{};     ///< Bytes asked for, per VkSystemAllocationScope
    std::array<std::atomic<s64>, 5> internal{}; ///< The driver's own (notified), per scope
    std::atomic<u64> blocks{};
    std::atomic<u64> total{};
    std::atomic<u64> peak{};
    std::atomic<u64> failures{};
    std::atomic<u64> foreign{};
#ifdef _WIN32
    HANDLE heap{};
#endif
};

/// Sits right below each block handed out.
struct HostBlockHeader {
    void* raw;
    u64 size;
    u32 scope;
    u32 magic;
    u64 reserved;
};
static_assert(sizeof(HostBlockHeader) == 32);
constexpr u32 HostBlockMagic = 0x46564b48; // "HKVF"

HostAllocState& HostState() {
    static HostAllocState state;
    return state;
}

size_t ScopeIndex(VkSystemAllocationScope scope) {
    return (std::min)(static_cast<size_t>(scope), size_t{4});
}

void* VKAPI_CALL HostAllocate(void* user, size_t size, size_t alignment,
                              VkSystemAllocationScope scope) {
    auto& state = *static_cast<HostAllocState*>(user);
    if (size == 0) {
        return nullptr;
    }
    alignment = (std::max)(alignment, size_t{16});
    if ((alignment & (alignment - 1)) != 0) {
        return nullptr;
    }
    // The heap's blocks are 16-byte aligned: the header and the alignment fit in this much.
    const size_t total = size + sizeof(HostBlockHeader) + alignment - 16;
#ifdef _WIN32
    void* const raw = HeapAlloc(state.heap, 0, total);
#else
    void* const raw = std::malloc(total);
#endif
    if (raw == nullptr) {
        state.failures.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    const uintptr_t first = reinterpret_cast<uintptr_t>(raw) + sizeof(HostBlockHeader);
    const uintptr_t aligned = (first + alignment - 1) & ~(uintptr_t{alignment} - 1);
    auto* const header = reinterpret_cast<HostBlockHeader*>(aligned) - 1;
    const size_t index = ScopeIndex(scope);
    *header = HostBlockHeader{raw, size, static_cast<u32>(index), HostBlockMagic, 0};
    state.live[index].fetch_add(size, std::memory_order_relaxed);
    state.blocks.fetch_add(1, std::memory_order_relaxed);
    const u64 now = state.total.fetch_add(size, std::memory_order_relaxed) + size;
    u64 peak = state.peak.load(std::memory_order_relaxed);
    while (now > peak && !state.peak.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
    }
    return reinterpret_cast<void*>(aligned);
}

void VKAPI_CALL HostFree(void* user, void* memory) {
    if (memory == nullptr) {
        return;
    }
    auto& state = *static_cast<HostAllocState*>(user);
    auto* const header = static_cast<HostBlockHeader*>(memory) - 1;
    if (header->magic != HostBlockMagic) {
        // Not one of ours: leaked rather than handed to a heap that never made it.
        if (state.foreign.fetch_add(1, std::memory_order_relaxed) == 0) {
            LOG_ERROR(Render_Vulkan, "FRITCHII_RAM_VK_HOST_ALLOC: the driver freed a block it did "
                                     "not get from the callbacks; left alone");
        }
        return;
    }
    header->magic = 0;
    state.live[header->scope].fetch_sub(header->size, std::memory_order_relaxed);
    state.blocks.fetch_sub(1, std::memory_order_relaxed);
    state.total.fetch_sub(header->size, std::memory_order_relaxed);
#ifdef _WIN32
    HeapFree(state.heap, 0, header->raw);
#else
    std::free(header->raw);
#endif
}

void* VKAPI_CALL HostReallocate(void* user, void* original, size_t size, size_t alignment,
                                VkSystemAllocationScope scope) {
    if (original == nullptr) {
        return HostAllocate(user, size, alignment, scope);
    }
    if (size == 0) {
        HostFree(user, original);
        return nullptr;
    }
    const auto* const header = static_cast<const HostBlockHeader*>(original) - 1;
    if (header->magic != HostBlockMagic) {
        return nullptr; // the original stays valid, as on a failed reallocation
    }
    void* const moved = HostAllocate(user, size, alignment, scope);
    if (moved == nullptr) {
        return nullptr;
    }
    std::memcpy(moved, original, (std::min)(static_cast<size_t>(header->size), size));
    HostFree(user, original);
    return moved;
}

void VKAPI_CALL HostInternalAllocation(void* user, size_t size, VkInternalAllocationType,
                                       VkSystemAllocationScope scope) {
    static_cast<HostAllocState*>(user)->internal[ScopeIndex(scope)].fetch_add(
        static_cast<s64>(size), std::memory_order_relaxed);
}

void VKAPI_CALL HostInternalFree(void* user, size_t size, VkInternalAllocationType,
                                 VkSystemAllocationScope scope) {
    static_cast<HostAllocState*>(user)->internal[ScopeIndex(scope)].fetch_sub(
        static_cast<s64>(size), std::memory_order_relaxed);
}

std::string HostAllocCensus() {
    const HostAllocState& state = HostState();
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    std::array<u64, 5> live{};
    s64 internal = 0;
    for (size_t i = 0; i < live.size(); ++i) {
        live[i] = state.live[i].load(std::memory_order_relaxed);
        internal += state.internal[i].load(std::memory_order_relaxed);
    }
    return fmt::format(
        "Vulkan host allocations (FRITCHII_RAM_VK_HOST_ALLOC=1, their heap is in 'other heaps'): "
        "{} MiB live in {} blocks (peak {}), by scope command {} object {} cache {} device {} "
        "instance {}; the driver's own (notified) {} MiB; {} failed, {} foreign frees",
        mib(state.total.load(std::memory_order_relaxed)),
        state.blocks.load(std::memory_order_relaxed), mib(state.peak.load(std::memory_order_relaxed)),
        mib(live[0]), mib(live[1]), mib(live[2]), mib(live[3]), mib(live[4]),
        internal > 0 ? (static_cast<u64>(internal) >> 20) : 0,
        state.failures.load(std::memory_order_relaxed),
        state.foreign.load(std::memory_order_relaxed));
}

} // namespace

const VkAllocationCallbacks* HostAllocator() noexcept {
    static const VkAllocationCallbacks* const callbacks = []() -> const VkAllocationCallbacks* {
        const char* const value = std::getenv("FRITCHII_RAM_VK_HOST_ALLOC");
        if (value == nullptr || value[0] != '1') {
            return nullptr;
        }
        HostAllocState& state = HostState();
#ifdef _WIN32
        state.heap = HeapCreate(0, 0, 0);
        if (state.heap == nullptr) {
            LOG_ERROR(Render_Vulkan, "FRITCHII_RAM_VK_HOST_ALLOC=1: HeapCreate failed ({}), off",
                      GetLastError());
            return nullptr;
        }
        ULONG low_fragmentation = 2;
        HeapSetInformation(state.heap, HeapCompatibilityInformation, &low_fragmentation,
                           sizeof(low_fragmentation));
#endif
        static const VkAllocationCallbacks table{
            .pUserData = &state,
            .pfnAllocation = HostAllocate,
            .pfnReallocation = HostReallocate,
            .pfnFree = HostFree,
            .pfnInternalAllocation = HostInternalAllocation,
            .pfnInternalFree = HostInternalFree,
        };
        LOG_INFO(Render_Vulkan, "FRITCHII_RAM_VK_HOST_ALLOC=1: the driver's host allocations "
                                "(instance and device) go through Fritchii's own counted heap");
        if (Common::MemCensus::Mode() != 0) {
            Common::MemCensus::AddReporter(&state, [] { return HostAllocCensus(); });
        }
        return &table;
    }();
    return callbacks;
}

// FRITCHII_VA_REPORT: the last destroys of objects whose memory the driver owns (pipeline code,
// views, samplers, pools), for the device-loss report: a GPU read of a freed one faults in the
// driver's own address range, which no binding report names.
namespace DestroyRing {
struct Entry {
    u64 ns;
    u64 handle;
    const char* type;
    u32 thread;
    const char* thread_name;
    std::array<u32, 8> frames; ///< return addresses as offsets into the module that holds them
};
constexpr size_t Size = 4096;
std::array<Entry, Size> g_ring{};
std::atomic<u64> g_next{};
/// The VA report's switch (VaReport::Requested, read once): off, one static bool per destroy.
bool Wanted() {
    return Vulkan::VaReport::Requested();
}
u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}
u32 ThreadId() {
#ifdef _WIN32
    return static_cast<u32>(GetCurrentThreadId());
#else
    return 0;
#endif
}
const char* ThreadName() {
    thread_local std::string name = [] {
        std::string result = "?";
#ifdef _WIN32
        PWSTR wide = nullptr;
        if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &wide)) && wide != nullptr) {
            result.clear();
            for (const wchar_t* c = wide; *c != 0 && result.size() < 31; ++c) {
                result.push_back(*c < 128 ? static_cast<char>(*c) : '?');
            }
            LocalFree(wide);
        }
#endif
        return result;
    }();
    return name.c_str();
}
void Note(const char* type, u64 handle) {
    if (!Wanted()) {
        return;
    }
    Entry entry{NowNs(), handle, type, ThreadId(), ThreadName(), {}};
#ifdef _WIN32
    void* frames[9]{};
    const USHORT count = RtlCaptureStackBackTrace(1, 9, frames, nullptr);
    static const uintptr_t base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
    for (USHORT i = 0; i < count && i < entry.frames.size(); ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(frames[i]);
        entry.frames[i] = address > base ? static_cast<u32>(address - base) : 0;
    }
#endif
    g_ring[g_next.fetch_add(1, std::memory_order_relaxed) % Size] = entry;
}
} // namespace DestroyRing

void DumpRecentDestroys(u64 window_ms) {
    if (!DestroyRing::Wanted()) {
        return;
    }
    const u64 count = DestroyRing::g_next.load(std::memory_order_relaxed);
    const u64 now = DestroyRing::NowNs();
    // Written at once to its own file as well: the process may die before the log flushes.
    static std::atomic<bool> written{};
    if (!written.exchange(true)) {
        if (const char* appdata = std::getenv("APPDATA")) {
            const std::string path = std::string{appdata} + "/fritchii/log/destroy_report.txt";
            if (std::FILE* file = std::fopen(path.c_str(), "w")) {
                const u64 begin = count > DestroyRing::Size ? count - DestroyRing::Size : 0;
                for (u64 i = count; i-- > begin;) {
                    const DestroyRing::Entry& e = DestroyRing::g_ring[i % DestroyRing::Size];
                    if (e.type == nullptr || now < e.ns || now - e.ns > window_ms * 1000000ULL) {
                        continue;
                    }
                    std::fprintf(file, "%s %llx %.3f ms ago thread %u \"%s\" stack", e.type,
                                 static_cast<unsigned long long>(e.handle),
                                 static_cast<double>(now - e.ns) / 1e6, e.thread,
                                 e.thread_name != nullptr ? e.thread_name : "?");
                    for (const u32 frame : e.frames) {
                        std::fprintf(file, " %x", frame);
                    }
                    std::fputc('\n', file);
                }
                std::fflush(file);
                std::fclose(file);
            }
        }
    }
    const u64 first = count > DestroyRing::Size ? count - DestroyRing::Size : 0;
    u32 shown = 0;
    for (u64 i = count; i-- > first && shown < 48;) {
        const DestroyRing::Entry& e = DestroyRing::g_ring[i % DestroyRing::Size];
        if (e.type == nullptr || now < e.ns || now - e.ns > window_ms * 1000000ULL) {
            continue;
        }
        ++shown;
        LOG_CRITICAL(Render_Vulkan,
                     "Destroy report: {} {:#x} destroyed {:.3f} ms ago on thread {} \"{}\"; "
                     "stack exe+{:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
                     e.type, e.handle, static_cast<double>(now - e.ns) / 1e6, e.thread,
                     e.thread_name != nullptr ? e.thread_name : "?", e.frames[0], e.frames[1],
                     e.frames[2], e.frames[3], e.frames[4], e.frames[5], e.frames[6],
                     e.frames[7]);
    }
    LOG_CRITICAL(Render_Vulkan, "Destroy report: {} destroys in the last {} ms shown ({} kept)",
                 shown, window_ms, count - first);
}

void Destroy(VkInstance instance, const InstanceDispatch& dld) noexcept {
    dld.vkDestroyInstance(instance, HostAllocator());
}

void Destroy(VkDevice device, const InstanceDispatch& dld) noexcept {
    dld.vkDestroyDevice(device, HostAllocator());
}

void Destroy(VkDevice device, VkBuffer handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("Buffer", u64(handle));
    dld.vkDestroyBuffer(device, handle, nullptr);
}

void Destroy(VkDevice device, VkBufferView handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("BufferView", u64(handle));
    dld.vkDestroyBufferView(device, handle, nullptr);
}

void Destroy(VkDevice device, VkCommandPool handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("CommandPool", u64(handle));
    dld.vkDestroyCommandPool(device, handle, nullptr);
}

void Destroy(VkDevice device, VkDescriptorPool handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("DescriptorPool", u64(handle));
    dld.vkDestroyDescriptorPool(device, handle, nullptr);
}

void Destroy(VkDevice device, VkDescriptorSetLayout handle, const DeviceDispatch& dld) noexcept {
    dld.vkDestroyDescriptorSetLayout(device, handle, nullptr);
}

void Destroy(VkDevice device, VkDescriptorUpdateTemplate handle,
             const DeviceDispatch& dld) noexcept {
    dld.vkDestroyDescriptorUpdateTemplate(device, handle, nullptr);
}

void Destroy(VkDevice device, VkDeviceMemory handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("DeviceMemory", u64(handle));
    dld.vkFreeMemory(device, handle, nullptr);
}

void Destroy(VkDevice device, VkEvent handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("Event", u64(handle));
    dld.vkDestroyEvent(device, handle, nullptr);
}

void Destroy(VkDevice device, VkFence handle, const DeviceDispatch& dld) noexcept {
    dld.vkDestroyFence(device, handle, nullptr);
}

void Destroy(VkDevice device, VkFramebuffer handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("Framebuffer", u64(handle));
    dld.vkDestroyFramebuffer(device, handle, nullptr);
}

void Destroy(VkDevice device, VkImage handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("Image", u64(handle));
    dld.vkDestroyImage(device, handle, nullptr);
}

void Destroy(VkDevice device, VkImageView handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("ImageView", u64(handle));
    dld.vkDestroyImageView(device, handle, nullptr);
}

void Destroy(VkDevice device, VkPipeline handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("Pipeline", u64(handle));
    dld.vkDestroyPipeline(device, handle, nullptr);
}

void Destroy(VkDevice device, VkPipelineCache handle, const DeviceDispatch& dld) noexcept {
    dld.vkDestroyPipelineCache(device, handle, nullptr);
}

void Destroy(VkDevice device, VkPipelineLayout handle, const DeviceDispatch& dld) noexcept {
    dld.vkDestroyPipelineLayout(device, handle, nullptr);
}

void Destroy(VkDevice device, VkQueryPool handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("QueryPool", u64(handle));
    dld.vkDestroyQueryPool(device, handle, nullptr);
}

void Destroy(VkDevice device, VkRenderPass handle, const DeviceDispatch& dld) noexcept {
    dld.vkDestroyRenderPass(device, handle, nullptr);
}

void Destroy(VkDevice device, VkSampler handle, const DeviceDispatch& dld) noexcept {
    DestroyRing::Note("Sampler", u64(handle));
    dld.vkDestroySampler(device, handle, nullptr);
}

void Destroy(VkDevice device, VkSwapchainKHR handle, const DeviceDispatch& dld) noexcept {
    dld.vkDestroySwapchainKHR(device, handle, nullptr);
}

void Destroy(VkDevice device, VkSemaphore handle, const DeviceDispatch& dld) noexcept {
    dld.vkDestroySemaphore(device, handle, nullptr);
}

void Destroy(VkDevice device, VkShaderModule handle, const DeviceDispatch& dld) noexcept {
    dld.vkDestroyShaderModule(device, handle, nullptr);
}

void Destroy(VkInstance instance, VkDebugUtilsMessengerEXT handle,
             const InstanceDispatch& dld) noexcept {
    dld.vkDestroyDebugUtilsMessengerEXT(instance, handle, nullptr);
}

void Destroy(VkInstance instance, VkDebugReportCallbackEXT handle,
             const InstanceDispatch& dld) noexcept {
    dld.vkDestroyDebugReportCallbackEXT(instance, handle, nullptr);
}

void Destroy(VkInstance instance, VkSurfaceKHR handle, const InstanceDispatch& dld) noexcept {
    dld.vkDestroySurfaceKHR(instance, handle, nullptr);
}

VkResult Free(VkDevice device, VkDescriptorPool handle, Span<VkDescriptorSet> sets,
              const DeviceDispatch& dld) noexcept {
    return dld.vkFreeDescriptorSets(device, handle, sets.size(), sets.data());
}

VkResult Free(VkDevice device, VkCommandPool handle, Span<VkCommandBuffer> buffers,
              const DeviceDispatch& dld) noexcept {
    dld.vkFreeCommandBuffers(device, handle, buffers.size(), buffers.data());
    return VK_SUCCESS;
}

Instance Instance::Create(u32 version, Span<const char*> layers, Span<const char*> extensions,
                          InstanceDispatch& dispatch) {
#ifdef __APPLE__
    constexpr VkFlags ci_flags{VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR};
#else
    constexpr VkFlags ci_flags{};
#endif
    // DO NOT TOUCH, breaks RNDA3!!
	// Don't know why, but gloom + yellow line glitch appears
    const VkApplicationInfo application_info{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "yuzu Emulator",
        .applicationVersion = VK_MAKE_VERSION(1, 3, 0),
        .pEngineName = "yuzu Emulator",
        .engineVersion = VK_MAKE_VERSION(1, 3, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = ci_flags,
        .pApplicationInfo = &application_info,
        .enabledLayerCount = layers.size(),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = extensions.size(),
        .ppEnabledExtensionNames = extensions.data(),
    };
    VkInstance instance;
    Check(dispatch.vkCreateInstance(&ci, HostAllocator(), &instance));
    if (!Proc(dispatch.vkDestroyInstance, dispatch, "vkDestroyInstance", instance)) {
        // We successfully created an instance but the destroy function couldn't be loaded.
        // This is a good moment to panic.
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }
    return Instance(instance, dispatch);
}

std::vector<VkPhysicalDevice> Instance::EnumeratePhysicalDevices() const {
    u32 num;
    Check(dld->vkEnumeratePhysicalDevices(handle, &num, nullptr));
    std::vector<VkPhysicalDevice> physical_devices(num);
    Check(dld->vkEnumeratePhysicalDevices(handle, &num, physical_devices.data()));
    SortPhysicalDevices(physical_devices, *dld);
    return physical_devices;
}

DebugUtilsMessenger Instance::CreateDebugUtilsMessenger(
    const VkDebugUtilsMessengerCreateInfoEXT& create_info) const {
    VkDebugUtilsMessengerEXT object;
    Check(dld->vkCreateDebugUtilsMessengerEXT(handle, &create_info, nullptr, &object));
    return DebugUtilsMessenger(object, handle, *dld);
}

DebugReportCallback Instance::CreateDebugReportCallback(
    const VkDebugReportCallbackCreateInfoEXT& create_info) const {
    VkDebugReportCallbackEXT object;
    Check(dld->vkCreateDebugReportCallbackEXT(handle, &create_info, nullptr, &object));
    return DebugReportCallback(object, handle, *dld);
}

void Image::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_IMAGE, name);
}

void Image::Release() const noexcept {
    if (handle) {
        // FRITCHII_DEPTH_POOL_FIT: only read while padded images may exist.
        if (allocation != nullptr &&
            g_image_padding_bytes.load(std::memory_order_relaxed) != 0) [[unlikely]] {
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(allocator, allocation, &info);
            const u64 data = static_cast<u64>(reinterpret_cast<uintptr_t>(info.pUserData));
            if ((data & ImageUserDataTag) != 0) {
                const u64 padding = ((data >> 32) & 0x7FFF'FFFFULL) << 8;
                if (padding != 0) {
                    g_image_padding_bytes.fetch_sub(padding, std::memory_order_relaxed);
                }
            }
        }
        vmaDestroyImage(allocator, handle, allocation);
    }
}

Image::MemoryBlock Image::GetMemoryBlock() const noexcept {
    if (allocation == nullptr) {
        return {};
    }
    VmaAllocationInfo2 info{};
    vmaGetAllocationInfo2(allocator, allocation, &info);
    if (info.dedicatedMemory) {
        return {0, info.blockSize, info.allocationInfo.size};
    }
    return {reinterpret_cast<u64>(info.allocationInfo.deviceMemory), info.blockSize,
            info.allocationInfo.size};
}

u64 VmaMemoryObjectCount(VmaAllocator allocator) noexcept {
    if (allocator == nullptr) {
        return 0;
    }
    const VkPhysicalDeviceMemoryProperties* properties{};
    vmaGetMemoryProperties(allocator, &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    u64 count = 0;
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        count += budgets[heap].statistics.blockCount;
    }
    return count;
}

Image::DestroyTiming Image::DestroyTimed() noexcept {
    DestroyTiming timing{};
    if (!handle) {
        return timing;
    }
    const auto now = [] {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    };
    // Counted outside the timed steps (vmaGetHeapBudgets can ask the driver every 30 operations).
    const u64 blocks_before = allocation != nullptr ? VmaMemoryObjectCount(allocator) : 0;
    const u64 t0 = now();
    dld->vkDestroyImage(owner, handle, nullptr);
    const u64 t1 = now();
    if (allocation != nullptr) {
        vmaFreeMemory(allocator, allocation);
    }
    const u64 t2 = now();
    if (allocation != nullptr) {
        timing.block_freed = VmaMemoryObjectCount(allocator) < blocks_before;
    }
    timing.destroy_ns = t1 - t0;
    timing.free_ns = t2 - t1;
    handle = VkImage{};
    allocation = nullptr;
    return timing;
}

bool Image::IsDeviceLocal() const noexcept {
    if (allocation == nullptr) {
        return false;
    }
    VkMemoryPropertyFlags flags{};
    vmaGetAllocationMemoryProperties(allocator, allocation, &flags);
    return (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
}

void Buffer::Flush() const {
    if (!is_coherent) {
        vmaFlushAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
    }
}

void Buffer::Invalidate() const {
    if (!is_coherent) {
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
    }
}

void Buffer::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_BUFFER, name);
}

void Buffer::Release() const noexcept {
    if (handle) {
        const VideoCommon::MemoryStageLink::AllocatorOp allocator_op{allocator};
        vmaDestroyBuffer(allocator, handle, allocation);
    }
}

void BufferView::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_BUFFER_VIEW, name);
}

void ImageView::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_IMAGE_VIEW, name);
}

int DeviceMemory::GetMemoryFdKHR() const {
    const VkMemoryGetFdInfoKHR get_fd_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .pNext = nullptr,
        .memory = handle,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
    };
    int fd;
    Check(dld->vkGetMemoryFdKHR(owner, &get_fd_info, &fd));
    return fd;
}

#ifdef _WIN32
HANDLE DeviceMemory::GetMemoryWin32HandleKHR() const {
    const VkMemoryGetWin32HandleInfoKHR get_win32_handle_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .pNext = nullptr,
        .memory = handle,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
    };
    HANDLE win32_handle;
    Check(dld->vkGetMemoryWin32HandleKHR(owner, &get_win32_handle_info, &win32_handle));
    return win32_handle;
}
#endif

void DeviceMemory::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_DEVICE_MEMORY, name);
}

void Fence::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_FENCE, name);
}

void Framebuffer::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_FRAMEBUFFER, name);
}

DescriptorSets DescriptorPool::Allocate(const VkDescriptorSetAllocateInfo& ai) const {
    const std::size_t num = ai.descriptorSetCount;
    std::unique_ptr sets = std::make_unique<VkDescriptorSet[]>(num);
    switch (const VkResult result = dld->vkAllocateDescriptorSets(owner, &ai, sets.get())) {
    case VK_SUCCESS:
        return DescriptorSets(std::move(sets), num, owner, handle, *dld);
    case VK_ERROR_OUT_OF_POOL_MEMORY:
    case VK_ERROR_FRAGMENTED_POOL:
        return {};
    default:
        throw Exception(result);
    }
}

void DescriptorPool::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_DESCRIPTOR_POOL, name);
}

CommandBuffers CommandPool::Allocate(std::size_t num_buffers, VkCommandBufferLevel level) const {
    const VkCommandBufferAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = handle,
        .level = level,
        .commandBufferCount = static_cast<u32>(num_buffers),
    };

    std::unique_ptr buffers = std::make_unique<VkCommandBuffer[]>(num_buffers);
    switch (const VkResult result = dld->vkAllocateCommandBuffers(owner, &ai, buffers.get())) {
    case VK_SUCCESS:
        return CommandBuffers(std::move(buffers), num_buffers, owner, handle, *dld);
    case VK_ERROR_OUT_OF_POOL_MEMORY:
    case VK_ERROR_FRAGMENTED_POOL:
        return {};
    default:
        throw Exception(result);
    }
}

void CommandPool::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_COMMAND_POOL, name);
}

std::vector<VkImage> SwapchainKHR::GetImages() const {
    u32 num;
    Check(dld->vkGetSwapchainImagesKHR(owner, handle, &num, nullptr));
    std::vector<VkImage> images(num);
    Check(dld->vkGetSwapchainImagesKHR(owner, handle, &num, images.data()));
    return images;
}

void Event::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_EVENT, name);
}

void ShaderModule::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_SHADER_MODULE, name);
}

void PipelineCache::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_PIPELINE_CACHE, name);
}

void Semaphore::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_SEMAPHORE, name);
}

Device Device::Create(VkPhysicalDevice physical_device, Span<VkDeviceQueueCreateInfo> queues_ci,
                      Span<const char*> enabled_extensions, const void* next,
                      DeviceDispatch& dispatch) {
    const VkDeviceCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = next,
        .flags = 0,
        .queueCreateInfoCount = queues_ci.size(),
        .pQueueCreateInfos = queues_ci.data(),
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = enabled_extensions.size(),
        .ppEnabledExtensionNames = enabled_extensions.data(),
        .pEnabledFeatures = nullptr,
    };
    VkDevice device;
    Check(dispatch.vkCreateDevice(physical_device, &ci, HostAllocator(), &device));
    Load(device, dispatch);
    return Device(device, dispatch);
}

Queue Device::GetQueue(u32 family_index) const noexcept {
    VkQueue queue;
    dld->vkGetDeviceQueue(handle, family_index, 0, &queue);
    return Queue(queue, *dld);
}

Queue Device::GetQueue(u32 family_index, u32 queue_index) const noexcept {
    VkQueue queue;
    dld->vkGetDeviceQueue(handle, family_index, queue_index, &queue);
    return Queue(queue, *dld);
}

BufferView Device::CreateBufferView(const VkBufferViewCreateInfo& ci) const {
    VkBufferView object;
    Check(dld->vkCreateBufferView(handle, &ci, nullptr, &object));
    return BufferView(object, handle, *dld);
}

ImageView Device::CreateImageView(const VkImageViewCreateInfo& ci) const {
    VkImageView object;
    Check(dld->vkCreateImageView(handle, &ci, nullptr, &object));
    return ImageView(object, handle, *dld);
}

Semaphore Device::CreateSemaphore() const {
    static constexpr VkSemaphoreCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    return CreateSemaphore(ci);
}

Semaphore Device::CreateSemaphore(const VkSemaphoreCreateInfo& ci) const {
    VkSemaphore object;
    Check(dld->vkCreateSemaphore(handle, &ci, nullptr, &object));
    return Semaphore(object, handle, *dld);
}

Fence Device::CreateFence(const VkFenceCreateInfo& ci) const {
    VkFence object;
    Check(dld->vkCreateFence(handle, &ci, nullptr, &object));
    return Fence(object, handle, *dld);
}

DescriptorPool Device::CreateDescriptorPool(const VkDescriptorPoolCreateInfo& ci) const {
    VkDescriptorPool object;
    Check(dld->vkCreateDescriptorPool(handle, &ci, nullptr, &object));
    return DescriptorPool(object, handle, *dld);
}

RenderPass Device::CreateRenderPass(const VkRenderPassCreateInfo& ci) const {
    VkRenderPass object;
    Check(dld->vkCreateRenderPass(handle, &ci, nullptr, &object));
    return RenderPass(object, handle, *dld);
}

RenderPass Device::CreateRenderPass2(const VkRenderPassCreateInfo2& ci) const {
    VkRenderPass object;
    Check(dld->vkCreateRenderPass2(handle, &ci, nullptr, &object));
    return RenderPass(object, handle, *dld);
}

DescriptorSetLayout Device::CreateDescriptorSetLayout(
    const VkDescriptorSetLayoutCreateInfo& ci) const {
    VkDescriptorSetLayout object;
    Check(dld->vkCreateDescriptorSetLayout(handle, &ci, nullptr, &object));
    return DescriptorSetLayout(object, handle, *dld);
}

PipelineCache Device::CreatePipelineCache(const VkPipelineCacheCreateInfo& ci) const {
    VkPipelineCache cache;
    Check(dld->vkCreatePipelineCache(handle, &ci, nullptr, &cache));
    return PipelineCache(cache, handle, *dld);
}

PipelineLayout Device::CreatePipelineLayout(const VkPipelineLayoutCreateInfo& ci) const {
    VkPipelineLayout object;
    Check(dld->vkCreatePipelineLayout(handle, &ci, nullptr, &object));
    return PipelineLayout(object, handle, *dld);
}

Pipeline Device::CreateGraphicsPipeline(const VkGraphicsPipelineCreateInfo& ci, VkPipelineCache cache) const {
    VkPipeline object = VK_NULL_HANDLE;
    auto const result = dld->vkCreateGraphicsPipelines(handle, cache, 1, &ci, nullptr, &object);
    // Adreno 5xx drivers do not properly return when a graphics pipeline fails to be created
    // Some (unkown) Mali drivers also do not properly return
    // This result code is out of spec, but should be handled as "kinda working"
    if (result == VK_INCOMPLETE)
        return Pipeline(object, handle, *dld);
    Check(result);
    return Pipeline(object, handle, *dld);
}

Pipeline Device::CreateComputePipeline(const VkComputePipelineCreateInfo& ci,
                                       VkPipelineCache cache) const {
    VkPipeline object;
    Check(dld->vkCreateComputePipelines(handle, cache, 1, &ci, nullptr, &object));
    return Pipeline(object, handle, *dld);
}

Sampler Device::CreateSampler(const VkSamplerCreateInfo& ci) const {
    VkSampler object;
    Check(dld->vkCreateSampler(handle, &ci, nullptr, &object));
    return Sampler(object, handle, *dld);
}

Framebuffer Device::CreateFramebuffer(const VkFramebufferCreateInfo& ci) const {
    VkFramebuffer object;
    Check(dld->vkCreateFramebuffer(handle, &ci, nullptr, &object));
    return Framebuffer(object, handle, *dld);
}

CommandPool Device::CreateCommandPool(const VkCommandPoolCreateInfo& ci) const {
    VkCommandPool object;
    Check(dld->vkCreateCommandPool(handle, &ci, nullptr, &object));
    return CommandPool(object, handle, *dld);
}

DescriptorUpdateTemplate Device::CreateDescriptorUpdateTemplate(
    const VkDescriptorUpdateTemplateCreateInfo& ci) const {
    VkDescriptorUpdateTemplate object;
    Check(dld->vkCreateDescriptorUpdateTemplate(handle, &ci, nullptr, &object));
    return DescriptorUpdateTemplate(object, handle, *dld);
}

QueryPool Device::CreateQueryPool(const VkQueryPoolCreateInfo& ci) const {
    VkQueryPool object;
    Check(dld->vkCreateQueryPool(handle, &ci, nullptr, &object));
    return QueryPool(object, handle, *dld);
}

ShaderModule Device::CreateShaderModule(const VkShaderModuleCreateInfo& ci) const {
    VkShaderModule object;
    Check(dld->vkCreateShaderModule(handle, &ci, nullptr, &object));
    return ShaderModule(object, handle, *dld);
}

Event Device::CreateEvent() const {
    static constexpr VkEventCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };

    VkEvent object;
    Check(dld->vkCreateEvent(handle, &ci, nullptr, &object));
    return Event(object, handle, *dld);
}

SwapchainKHR Device::CreateSwapchainKHR(const VkSwapchainCreateInfoKHR& ci) const {
    VkSwapchainKHR object;
    Check(dld->vkCreateSwapchainKHR(handle, &ci, nullptr, &object));
    return SwapchainKHR(object, handle, *dld);
}

DeviceMemory Device::TryAllocateMemory(const VkMemoryAllocateInfo& ai) const noexcept {
    VkDeviceMemory memory;
    if (dld->vkAllocateMemory(handle, &ai, nullptr, &memory) != VK_SUCCESS) {
        return {};
    }
    return DeviceMemory(memory, handle, *dld);
}

DeviceMemory Device::AllocateMemory(const VkMemoryAllocateInfo& ai) const {
    VkDeviceMemory memory;
    Check(dld->vkAllocateMemory(handle, &ai, nullptr, &memory));
    return DeviceMemory(memory, handle, *dld);
}

VkMemoryRequirements Device::GetBufferMemoryRequirements(VkBuffer buffer,
                                                         void* pnext) const noexcept {
    const VkBufferMemoryRequirementsInfo2 info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .pNext = nullptr,
        .buffer = buffer,
    };
    VkMemoryRequirements2 requirements{
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = pnext,
        .memoryRequirements{},
    };
    dld->vkGetBufferMemoryRequirements2(handle, &info, &requirements);
    return requirements.memoryRequirements;
}

VkMemoryRequirements Device::GetImageMemoryRequirements(VkImage image) const noexcept {
    VkMemoryRequirements requirements;
    dld->vkGetImageMemoryRequirements(handle, image, &requirements);
    return requirements;
}

std::vector<VkSparseImageMemoryRequirements> Device::GetImageSparseMemoryRequirements(
    VkImage image) const {
    u32 count = 0;
    dld->vkGetImageSparseMemoryRequirements(handle, image, &count, nullptr);
    std::vector<VkSparseImageMemoryRequirements> requirements(count);
    dld->vkGetImageSparseMemoryRequirements(handle, image, &count, requirements.data());
    requirements.resize(count);
    return requirements;
}

std::vector<VkPipelineExecutablePropertiesKHR> Device::GetPipelineExecutablePropertiesKHR(
    VkPipeline pipeline) const {
    const VkPipelineInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
        .pNext = nullptr,
        .pipeline = pipeline,
    };
    u32 num{};
    dld->vkGetPipelineExecutablePropertiesKHR(handle, &info, &num, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> properties(num);
    for (auto& property : properties) {
        property.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    }
    Check(dld->vkGetPipelineExecutablePropertiesKHR(handle, &info, &num, properties.data()));
    return properties;
}

std::vector<VkPipelineExecutableStatisticKHR> Device::GetPipelineExecutableStatisticsKHR(
    VkPipeline pipeline, u32 executable_index) const {
    const VkPipelineExecutableInfoKHR executable_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
        .pNext = nullptr,
        .pipeline = pipeline,
        .executableIndex = executable_index,
    };
    u32 num{};
    dld->vkGetPipelineExecutableStatisticsKHR(handle, &executable_info, &num, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> statistics(num);
    for (auto& statistic : statistics) {
        statistic.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
    }
    Check(dld->vkGetPipelineExecutableStatisticsKHR(handle, &executable_info, &num,
                                                    statistics.data()));
    return statistics;
}

void Device::UpdateDescriptorSets(Span<VkWriteDescriptorSet> writes,
                                  Span<VkCopyDescriptorSet> copies) const noexcept {
    dld->vkUpdateDescriptorSets(handle, writes.size(), writes.data(), copies.size(), copies.data());
}

VkPhysicalDeviceProperties PhysicalDevice::GetProperties() const noexcept {
    VkPhysicalDeviceProperties properties;
    dld->vkGetPhysicalDeviceProperties(physical_device, &properties);
    return properties;
}

void PhysicalDevice::GetProperties2(VkPhysicalDeviceProperties2& properties) const noexcept {
    dld->vkGetPhysicalDeviceProperties2(physical_device, &properties);
}

VkPhysicalDeviceFeatures PhysicalDevice::GetFeatures() const noexcept {
    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = nullptr;
    dld->vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features2.features;
}

void PhysicalDevice::GetFeatures2(VkPhysicalDeviceFeatures2& features) const noexcept {
    dld->vkGetPhysicalDeviceFeatures2(physical_device, &features);
}

VkFormatProperties PhysicalDevice::GetFormatProperties(VkFormat format) const noexcept {
    VkFormatProperties properties;
    dld->vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
    return properties;
}

VkFormatProperties3 PhysicalDevice::GetFormatProperties3(VkFormat format) const noexcept {
    VkFormatProperties3 properties3{
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3,
        .pNext = nullptr,
        .linearTilingFeatures = 0,
        .optimalTilingFeatures = 0,
        .bufferFeatures = 0,
    };
    VkFormatProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        .pNext = &properties3,
        .formatProperties = {},
    };
    dld->vkGetPhysicalDeviceFormatProperties2(physical_device, format, &properties2);
    return properties3;
}

std::vector<VkExtensionProperties> PhysicalDevice::EnumerateDeviceExtensionProperties() const {
    u32 num;
    dld->vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &num, nullptr);
    std::vector<VkExtensionProperties> properties(num);
    dld->vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &num, properties.data());
    return properties;
}

std::vector<VkQueueFamilyProperties> PhysicalDevice::GetQueueFamilyProperties() const {
    u32 num;
    dld->vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &num, nullptr);
    std::vector<VkQueueFamilyProperties> properties(num);
    dld->vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &num, properties.data());
    return properties;
}

std::vector<VkPhysicalDeviceToolProperties> PhysicalDevice::GetPhysicalDeviceToolProperties()
    const {
    u32 num = 0;
    if (!dld->vkGetPhysicalDeviceToolProperties) {
        return {};
    }
    dld->vkGetPhysicalDeviceToolProperties(physical_device, &num, nullptr);
    std::vector<VkPhysicalDeviceToolProperties> properties(num);
    dld->vkGetPhysicalDeviceToolProperties(physical_device, &num, properties.data());
    return properties;
}

std::vector<VkCooperativeMatrixPropertiesKHR> PhysicalDevice::GetCooperativeMatrixPropertiesKHR()
    const {
    if (!dld->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        return {};
    }
    u32 num = 0;
    if (dld->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &num, nullptr) !=
        VK_SUCCESS) {
        return {};
    }
    std::vector<VkCooperativeMatrixPropertiesKHR> properties(
        num, VkCooperativeMatrixPropertiesKHR{
                 .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                 .pNext = nullptr,
             });
    if (dld->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &num,
                                                               properties.data()) != VK_SUCCESS) {
        return {};
    }
    properties.resize(num);
    return properties;
}

bool PhysicalDevice::GetSurfaceSupportKHR(u32 queue_family_index, VkSurfaceKHR surface) const {
    VkBool32 supported;
    Check(dld->vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index, surface,
                                                    &supported));
    return supported == VK_TRUE;
}

VkSurfaceCapabilitiesKHR PhysicalDevice::GetSurfaceCapabilitiesKHR(VkSurfaceKHR surface) const {
    VkSurfaceCapabilitiesKHR capabilities;
    Check(dld->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
    return capabilities;
}

std::vector<VkSurfaceFormatKHR> PhysicalDevice::GetSurfaceFormatsKHR(VkSurfaceKHR surface) const {
    u32 num;
    Check(dld->vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &num, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(num);
    Check(
        dld->vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &num, formats.data()));
    return formats;
}

std::vector<VkPresentModeKHR> PhysicalDevice::GetSurfacePresentModesKHR(
    VkSurfaceKHR surface) const {
    u32 num;
    Check(dld->vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &num, nullptr));
    std::vector<VkPresentModeKHR> modes(num);
    Check(dld->vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &num,
                                                         modes.data()));
    return modes;
}

VkPhysicalDeviceMemoryProperties2 PhysicalDevice::GetMemoryProperties(
    void* next_structures) const noexcept {
    VkPhysicalDeviceMemoryProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    properties.pNext = next_structures;
    dld->vkGetPhysicalDeviceMemoryProperties2(physical_device, &properties);
    return properties;
}

u32 AvailableVersion(const InstanceDispatch& dld) noexcept {
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion;
    if (!Proc(vkEnumerateInstanceVersion, dld, "vkEnumerateInstanceVersion")) {
        // If the procedure is not found, Vulkan 1.0 is assumed
        return VK_API_VERSION_1_0;
    }
    u32 version;
    if (const VkResult result = vkEnumerateInstanceVersion(&version); result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "vkEnumerateInstanceVersion returned {}, assuming Vulkan 1.1",
                  string_VkResult(result));
        return VK_API_VERSION_1_1;
    }
    return version;
}

std::optional<std::vector<VkExtensionProperties>> EnumerateInstanceExtensionProperties(
    const InstanceDispatch& dld) {
    u32 num;
    if (dld.vkEnumerateInstanceExtensionProperties(nullptr, &num, nullptr) != VK_SUCCESS) {
        return std::nullopt;
    }
    std::vector<VkExtensionProperties> properties(num);
    if (dld.vkEnumerateInstanceExtensionProperties(nullptr, &num, properties.data()) !=
        VK_SUCCESS) {
        return std::nullopt;
    }
    return properties;
}

std::optional<std::vector<VkLayerProperties>> EnumerateInstanceLayerProperties(
    const InstanceDispatch& dld) {
    u32 num;
    if (dld.vkEnumerateInstanceLayerProperties(&num, nullptr) != VK_SUCCESS) {
        return std::nullopt;
    }
    std::vector<VkLayerProperties> properties(num);
    if (dld.vkEnumerateInstanceLayerProperties(&num, properties.data()) != VK_SUCCESS) {
        return std::nullopt;
    }
    return properties;
}

std::string GetDriverName(VkPhysicalDeviceDriverProperties driver) {
    switch (driver.driverID) {
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        return "AMD";
    case VK_DRIVER_ID_AMD_OPEN_SOURCE:
        return "AMDVLK";
    case VK_DRIVER_ID_MESA_RADV:
        return "RADV";
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
        return "Nvidia";
    case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
        return "Intel";
    case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
        return "ANV";
    case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
        return "PowerVR";
    case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
        return "Qualcomm";
    case VK_DRIVER_ID_ARM_PROPRIETARY:
        return "Mali";
    case VK_DRIVER_ID_SAMSUNG_PROPRIETARY:
        return "Xclipse";
    case VK_DRIVER_ID_GOOGLE_SWIFTSHADER:
        return "SwiftShader";
    case VK_DRIVER_ID_BROADCOM_PROPRIETARY:
        return "Broadcom";
    case VK_DRIVER_ID_MESA_LLVMPIPE:
        return "llvmpipe";
    case VK_DRIVER_ID_MOLTENVK:
        return "MoltenVK";
    case VK_DRIVER_ID_VERISILICON_PROPRIETARY:
        return "Vivante";
    case VK_DRIVER_ID_MESA_TURNIP:
        return "Turnip";
    case VK_DRIVER_ID_MESA_V3DV:
        return "V3DV";
    case VK_DRIVER_ID_MESA_PANVK:
        return "PanVK";
    case VK_DRIVER_ID_MESA_VENUS:
        return "Venus";
    case VK_DRIVER_ID_MESA_DOZEN:
        return "Dozen";
    case VK_DRIVER_ID_MESA_NVK:
        return "NVK";
    case VK_DRIVER_ID_IMAGINATION_OPEN_SOURCE_MESA:
        return "PVR";
    case VK_DRIVER_ID_MESA_HONEYKRISP:
        return "HoneyKrisp";
    case VK_DRIVER_ID_MESA_KOSMICKRISP:
        return "KosmicKrisp";
    case VK_DRIVER_ID_GGP_PROPRIETARY:
    case VK_DRIVER_ID_COREAVI_PROPRIETARY:
    case VK_DRIVER_ID_JUICE_PROPRIETARY:
    case VK_DRIVER_ID_VULKAN_SC_EMULATION_ON_VULKAN:
    case VK_DRIVER_ID_MAX_ENUM:
    default:
        return driver.driverName;
    }
}

} // namespace Vulkan::vk
