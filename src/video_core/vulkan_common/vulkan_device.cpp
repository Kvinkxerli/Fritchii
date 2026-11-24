// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include "video_core/fritchii_occl_flag.h"
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/host_video_memory.h"
#include "common/literals.h"
#include "common/memory_detect.h"
#include "common/nx2_hitch.h"
#include <ranges>
#include "common/settings.h"
#include "common/settings_enums.h"
#include "common/shutdown_watch.h"
#ifdef _WIN32
#include "common/windows/crash_handler.h"
#elif defined(__linux__) && !defined(__ANDROID__)
#include "common/linux/crash_handler.h"
#endif
#include "video_core/churn_counters.h"
#include "video_core/memory_stage_link.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_profile.h"
#include "video_core/small_card.h"
#include "video_core/renderer_vulkan/vk_pipeline_binary.h"
#include "video_core/vulkan_common/nsight_aftermath_tracker.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_debug_callback.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_kept_blocks.h"
#include "video_core/nx2_perf.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
// Only this declaration: one_follower.h pulls in nvdata.h, whose enum names (BadValue, ...) the
// X11 headers vulkan.h includes on Linux define as macros.
namespace VideoCommon::OneFollower {
void NoteAmdProprietaryDriver(bool integrated) noexcept;
}
#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline.h"
#endif

#if defined(__ANDROID__) && defined(ARCHITECTURE_arm64)
#include <adrenotools/bcenabler.h>
#include <android/api-level.h>
#endif

namespace Vulkan {
using namespace Common::Literals;
namespace {
namespace Alternatives {
constexpr std::array STENCIL8_UINT{
    VK_FORMAT_D16_UNORM_S8_UINT,
    VK_FORMAT_D24_UNORM_S8_UINT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array DEPTH24_UNORM_STENCIL8_UINT{
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_D16_UNORM_S8_UINT,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array DEPTH24_UNORM_DONTCARE8{
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D16_UNORM,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array DEPTH16_UNORM_STENCIL8_UINT{
    VK_FORMAT_D24_UNORM_S8_UINT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array B5G6R5_UNORM_PACK16{
    VK_FORMAT_R5G6B5_UNORM_PACK16,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array R4G4_UNORM_PACK8{
    VK_FORMAT_R8_UNORM,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array R16G16B16_SFLOAT{
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array R16G16B16_SSCALED{
    VK_FORMAT_R16G16B16A16_SSCALED,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array R8G8B8_SSCALED{
    VK_FORMAT_R8G8B8A8_SSCALED,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array VK_FORMAT_R32G32B32_SFLOAT{
    VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array VK_FORMAT_A4B4G4R4_UNORM_PACK16{
    VK_FORMAT_R4G4B4A4_UNORM_PACK16,
    VK_FORMAT_UNDEFINED,
};

constexpr std::array B10G11R11_UFLOAT_PACK32{
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_A8B8G8R8_SRGB_PACK32,
    VK_FORMAT_UNDEFINED,
};

} // namespace Alternatives

template <typename T>
void SetNext(void**& next, T& data) {
    *next = &data;
    next = &data.pNext;
}

constexpr const VkFormat* GetFormatAlternatives(VkFormat format) {
    switch (format) {
    case VK_FORMAT_S8_UINT:
        return Alternatives::STENCIL8_UINT.data();
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return Alternatives::DEPTH24_UNORM_STENCIL8_UINT.data();
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return Alternatives::DEPTH24_UNORM_DONTCARE8.data();
    case VK_FORMAT_D16_UNORM_S8_UINT:
        return Alternatives::DEPTH16_UNORM_STENCIL8_UINT.data();
    case VK_FORMAT_B5G6R5_UNORM_PACK16:
        return Alternatives::B5G6R5_UNORM_PACK16.data();
    case VK_FORMAT_R4G4_UNORM_PACK8:
        return Alternatives::R4G4_UNORM_PACK8.data();
    case VK_FORMAT_R16G16B16_SFLOAT:
        return Alternatives::R16G16B16_SFLOAT.data();
    case VK_FORMAT_R16G16B16_SSCALED:
        return Alternatives::R16G16B16_SSCALED.data();
    case VK_FORMAT_R8G8B8_SSCALED:
        return Alternatives::R8G8B8_SSCALED.data();
    case VK_FORMAT_R32G32B32_SFLOAT:
        return Alternatives::VK_FORMAT_R32G32B32_SFLOAT.data();
    case VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT:
        return Alternatives::VK_FORMAT_A4B4G4R4_UNORM_PACK16.data();
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        return Alternatives::B10G11R11_UFLOAT_PACK32.data();
    default:
        return nullptr;
    }
}

VkFormatFeatureFlags GetFormatFeatures(VkFormatProperties properties, FormatType format_type) {
    switch (format_type) {
    case FormatType::Linear:
        return properties.linearTilingFeatures;
    case FormatType::Optimal:
        return properties.optimalTilingFeatures;
    case FormatType::Buffer:
        return properties.bufferFeatures;
    default:
        return {};
    }
}

::Common::unordered_map<VkFormat, VkFormatProperties> GetFormatProperties(vk::PhysicalDevice physical) {
    static constexpr std::array formats{
        VK_FORMAT_A1R5G5B5_UNORM_PACK16,
        VK_FORMAT_A2B10G10R10_SINT_PACK32,
        VK_FORMAT_A2B10G10R10_SNORM_PACK32,
        VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
        VK_FORMAT_A2B10G10R10_UINT_PACK32,
        VK_FORMAT_A2B10G10R10_UNORM_PACK32,
        VK_FORMAT_A2B10G10R10_USCALED_PACK32,
        VK_FORMAT_A2R10G10B10_UNORM_PACK32,
        VK_FORMAT_A8B8G8R8_SINT_PACK32,
        VK_FORMAT_A8B8G8R8_SNORM_PACK32,
        VK_FORMAT_A8B8G8R8_SRGB_PACK32,
        VK_FORMAT_A8B8G8R8_UINT_PACK32,
        VK_FORMAT_A8B8G8R8_UNORM_PACK32,
        VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
        VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
        VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
        VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
        VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
        VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
        VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
        VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
        VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
        VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
        VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
        VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
        VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
        VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
        VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
        VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
        VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
        VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
        VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        VK_FORMAT_B4G4R4A4_UNORM_PACK16,
        VK_FORMAT_B5G5R5A1_UNORM_PACK16,
        VK_FORMAT_B5G6R5_UNORM_PACK16,
        VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
        VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
        VK_FORMAT_BC2_SRGB_BLOCK,
        VK_FORMAT_BC2_UNORM_BLOCK,
        VK_FORMAT_BC3_SRGB_BLOCK,
        VK_FORMAT_BC3_UNORM_BLOCK,
        VK_FORMAT_BC4_SNORM_BLOCK,
        VK_FORMAT_BC4_UNORM_BLOCK,
        VK_FORMAT_BC5_SNORM_BLOCK,
        VK_FORMAT_BC5_UNORM_BLOCK,
        VK_FORMAT_BC6H_SFLOAT_BLOCK,
        VK_FORMAT_BC6H_UFLOAT_BLOCK,
        VK_FORMAT_BC7_SRGB_BLOCK,
        VK_FORMAT_BC7_UNORM_BLOCK,
        VK_FORMAT_D16_UNORM,
        VK_FORMAT_D16_UNORM_S8_UINT,
        VK_FORMAT_X8_D24_UNORM_PACK32,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SINT,
        VK_FORMAT_R16G16B16A16_SNORM,
        VK_FORMAT_R16G16B16A16_SSCALED,
        VK_FORMAT_R16G16B16A16_UINT,
        VK_FORMAT_R16G16B16A16_UNORM,
        VK_FORMAT_R16G16B16A16_USCALED,
        VK_FORMAT_R16G16B16_SFLOAT,
        VK_FORMAT_R16G16B16_SINT,
        VK_FORMAT_R16G16B16_SNORM,
        VK_FORMAT_R16G16B16_SSCALED,
        VK_FORMAT_R16G16B16_UINT,
        VK_FORMAT_R16G16B16_UNORM,
        VK_FORMAT_R16G16B16_USCALED,
        VK_FORMAT_R16G16_SFLOAT,
        VK_FORMAT_R16G16_SINT,
        VK_FORMAT_R16G16_SNORM,
        VK_FORMAT_R16G16_SSCALED,
        VK_FORMAT_R16G16_UINT,
        VK_FORMAT_R16G16_UNORM,
        VK_FORMAT_R16G16_USCALED,
        VK_FORMAT_R16_SFLOAT,
        VK_FORMAT_R16_SINT,
        VK_FORMAT_R16_SNORM,
        VK_FORMAT_R16_SSCALED,
        VK_FORMAT_R16_UINT,
        VK_FORMAT_R16_UNORM,
        VK_FORMAT_R16_USCALED,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_FORMAT_R32G32B32A32_SINT,
        VK_FORMAT_R32G32B32A32_UINT,
        VK_FORMAT_R32G32B32_SFLOAT,
        VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R32G32B32_UINT,
        VK_FORMAT_R32G32_SFLOAT,
        VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R32G32_UINT,
        VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R32_SINT,
        VK_FORMAT_R32_UINT,
        VK_FORMAT_R4G4B4A4_UNORM_PACK16,
        VK_FORMAT_A4B4G4R4_UNORM_PACK16_EXT,
        VK_FORMAT_R4G4_UNORM_PACK8,
        VK_FORMAT_R5G5B5A1_UNORM_PACK16,
        VK_FORMAT_R5G6B5_UNORM_PACK16,
        VK_FORMAT_R8G8B8A8_SINT,
        VK_FORMAT_R8G8B8A8_SNORM,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_FORMAT_R8G8B8A8_SSCALED,
        VK_FORMAT_R8G8B8A8_UINT,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R8G8B8A8_USCALED,
        VK_FORMAT_R8G8B8_SINT,
        VK_FORMAT_R8G8B8_SNORM,
        VK_FORMAT_R8G8B8_SSCALED,
        VK_FORMAT_R8G8B8_UINT,
        VK_FORMAT_R8G8B8_UNORM,
        VK_FORMAT_R8G8B8_USCALED,
        VK_FORMAT_R8G8_SINT,
        VK_FORMAT_R8G8_SNORM,
        VK_FORMAT_R8G8_SSCALED,
        VK_FORMAT_R8G8_UINT,
        VK_FORMAT_R8G8_UNORM,
        VK_FORMAT_R8G8_USCALED,
        VK_FORMAT_R8_SINT,
        VK_FORMAT_R8_SNORM,
        VK_FORMAT_R8_SSCALED,
        VK_FORMAT_R8_UINT,
        VK_FORMAT_R8_UNORM,
        VK_FORMAT_R8_USCALED,
        VK_FORMAT_S8_UINT,
        VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
        VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
        VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
        VK_FORMAT_EAC_R11_UNORM_BLOCK,
        VK_FORMAT_EAC_R11_SNORM_BLOCK,
        VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
        VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
    };
    ::Common::unordered_map<VkFormat, VkFormatProperties> format_properties;
    for (const auto format : formats) {
        format_properties.emplace(format, physical.GetFormatProperties(format));
    }
    return format_properties;
}

#if defined(__ANDROID__) && defined(ARCHITECTURE_arm64)
void OverrideBcnFormats(::Common::unordered_map<VkFormat, VkFormatProperties>& format_properties) {
    // These properties are extracted from Adreno driver 512.687.0
    constexpr VkFormatFeatureFlags tiling_features{VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                                   VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                                   VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                   VK_FORMAT_FEATURE_TRANSFER_DST_BIT};

    constexpr VkFormatFeatureFlags buffer_features{VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT};

    static constexpr std::array bcn_formats{
        VK_FORMAT_BC1_RGBA_SRGB_BLOCK, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC2_SRGB_BLOCK,
        VK_FORMAT_BC2_UNORM_BLOCK,     VK_FORMAT_BC3_SRGB_BLOCK,       VK_FORMAT_BC3_UNORM_BLOCK,
        VK_FORMAT_BC4_SNORM_BLOCK,     VK_FORMAT_BC4_UNORM_BLOCK,      VK_FORMAT_BC5_SNORM_BLOCK,
        VK_FORMAT_BC5_UNORM_BLOCK,     VK_FORMAT_BC6H_SFLOAT_BLOCK,    VK_FORMAT_BC6H_UFLOAT_BLOCK,
        VK_FORMAT_BC7_SRGB_BLOCK,      VK_FORMAT_BC7_UNORM_BLOCK,
    };

    for (const auto format : bcn_formats) {
        format_properties[format].linearTilingFeatures = tiling_features;
        format_properties[format].optimalTilingFeatures = tiling_features;
        format_properties[format].bufferFeatures = buffer_features;
    }
}
#endif

NvidiaArchitecture GetNvidiaArchitecture(vk::PhysicalDevice physical,
                                        const std::set<std::string, std::less<>>& exts) {
    VkPhysicalDeviceProperties2 physical_properties{};
    physical_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physical_properties.pNext = nullptr;

    if (exts.contains(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) {
        VkPhysicalDeviceFragmentShadingRatePropertiesKHR shading_rate_props{};
        shading_rate_props.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;
        physical_properties.pNext = &shading_rate_props;
        physical.GetProperties2(physical_properties);
        if (shading_rate_props.primitiveFragmentShadingRateWithMultipleViewports) {
            return NvidiaArchitecture::Arch_AmpereOrNewer;
        }
        return NvidiaArchitecture::Arch_Turing;
    }

    if (exts.contains(VK_EXT_BLEND_OPERATION_ADVANCED_EXTENSION_NAME)) {
        VkPhysicalDeviceBlendOperationAdvancedPropertiesEXT advanced_blending_props{};
        advanced_blending_props.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_PROPERTIES_EXT;
        physical_properties.pNext = &advanced_blending_props;
        physical.GetProperties2(physical_properties);
        if (advanced_blending_props.advancedBlendMaxColorAttachments == 1) {
            return NvidiaArchitecture::Arch_Maxwell;
        }

        if (exts.contains(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME)) {
            VkPhysicalDeviceConservativeRasterizationPropertiesEXT conservative_raster_props{};
            conservative_raster_props.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONSERVATIVE_RASTERIZATION_PROPERTIES_EXT;
            physical_properties.pNext = &conservative_raster_props;
            physical.GetProperties2(physical_properties);
            if (conservative_raster_props.degenerateLinesRasterized) {
                return NvidiaArchitecture::Arch_Volta;
            }
            return NvidiaArchitecture::Arch_Pascal;
        }
    }

    return NvidiaArchitecture::Arch_KeplerOrOlder;
}

std::vector<const char*> ExtensionListForVulkan(
    const std::set<std::string, std::less<>>& extensions) {
    std::vector<const char*> output;
    output.reserve(extensions.size());
    for (const auto& extension : extensions) {
        output.push_back(extension.c_str());
    }
    return output;
}

constexpr u32 NvidiaDriverVersion(u32 major, u32 minor, u32 secondary) {
    return (major << 22) | (minor << 14) | (secondary << 6);
}

constexpr std::array<char, 8> STATIC_CACHE_MAGIC_NUMBER{'e', 'd', 'e', 'n', 's', 't', 'p', 'c'};
constexpr u32 STATIC_CACHE_VERSION = 1;

std::filesystem::path StaticPipelineCacheFilename() {
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    if (!Common::FS::CreateDir(shader_dir)) {
        return {};
    }
    return shader_dir / "vulkan_static_pipelines.bin";
}

// What a heap's allocations use. The driver also counts the space in VMA's blocks no allocation
// uses, but freeing an image gives its share back only once its whole block is empty: the caches,
// collecting until the usage dropped, evicted thousands of textures in a row without lowering it
// (title 1).
u64 AllocatedUsage(VkDeviceSize driver_usage, const VmaBudget& vma_budget) {
    const VmaStatistics& statistics = vma_budget.statistics;
    // Guarded: two allocators' sums (or numbers taken at different times) must not wrap.
    const u64 unused =
        statistics.blockBytes - (std::min)(statistics.blockBytes, statistics.allocationBytes);
    return driver_usage - (std::min)(driver_usage, unused);
}

// vmaGetHeapBudgets of the main allocator, with the buffer allocator's blocks and allocations
// (FRITCHII_MEMSTAGE_SPLIT_ALLOC) added to the statistics: the unused space in either allocator's
// blocks is slack. usage and budget stay the main allocator's (the driver's, for the process).
// At FRITCHII_MEMSTAGE_THREAD=1 the buffer allocator's numbers are the ones its user published
// (Device::BufferHeapBudgets): an externally synchronized allocator refreshes its budget arrays
// inside vmaGetHeapBudgets, so no other thread may call it (memory stage spec 5.3). At 0, VMA's.
void GetHeapBudgets(const Device& device, std::array<VmaBudget, VK_MAX_MEMORY_HEAPS>& budgets) {
    if (VideoCommon::MemoryStageLink::OnStageThread()) [[unlikely]] {
        // The main allocator is the back's.
        VideoCommon::MemoryStageLink::Note(VideoCommon::MemoryStageLink::Check::BudgetUser,
                                           "main allocator");
    }
    vmaGetHeapBudgets(device.GetAllocator(), budgets.data());
    if (!device.HasSplitAllocators()) {
        return;
    }
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> buffers{};
    device.BufferHeapBudgets(buffers.data());
    for (size_t heap = 0; heap < budgets.size(); ++heap) {
        budgets[heap].statistics.blockCount += buffers[heap].statistics.blockCount;
        budgets[heap].statistics.allocationCount += buffers[heap].statistics.allocationCount;
        budgets[heap].statistics.blockBytes += buffers[heap].statistics.blockBytes;
        budgets[heap].statistics.allocationBytes += buffers[heap].statistics.allocationBytes;
    }
}

/// AMD GPUs before RDNA3 have no matrix units. AMD's driver still lists 16x16x16 float16
/// cooperative matrix shapes on RDNA2 and then crashes building the HMMA compute pipelines
/// (RX 6700 XT, driver 2.0.353, amdvlk64.dll+0x2f1c963 during the boot pipeline build).
[[nodiscard]] bool IsAmdWithoutMatrixUnits(u32 vendor_id, u32 device_id) {
    if (vendor_id != 0x1002) {
        return false;
    }
    // GCN discrete, Navi 1x to Navi 23, then Navi 24 (0x7400-0x741F is CDNA2, which has matrix
    // units). RDNA3 starts at 0x7440 (Navi 31).
    if ((device_id >= 0x6600 && device_id <= 0x6FFF) ||
        (device_id >= 0x7310 && device_id <= 0x73FF) ||
        (device_id >= 0x7420 && device_id <= 0x743F)) {
        return true;
    }
    switch (device_id) {
    case 0x163F: // Van Gogh (Steam Deck)
    case 0x1435: // Sephiroth (Steam Deck OLED)
    case 0x1681: // Rembrandt
    case 0x164E: // Raphael graphics
    case 0x13C0: // Granite Ridge graphics
    case 0x1506: // Mendocino
        return true;
    default:
        return false;
    }
}

// FRITCHII_KEEP_BLOCKS: VMA's vkAllocateMemory and vkFreeMemory go through these, so a freed
// block can be kept and handed back (vulkan_kept_blocks.h). With nothing kept and the switch off
// they only pass the call on.
PFN_vkAllocateMemory g_real_allocate_memory{};
PFN_vkFreeMemory g_real_free_memory{};

// FRITCHII_MEM_CENSUS (boot): VMA's maps of host-visible memory are tagged "VMA host maps" in the
// memory census, so their resident pages are told apart from the driver's other memory. Off: VMA
// calls the driver directly.
PFN_vkMapMemory g_real_map_memory{};
PFN_vkUnmapMemory g_real_unmap_memory{};
struct CensusMemory {
    VkDeviceSize first{};  ///< Size
    void* second{};        ///< Where VMA mapped it (null: not mapped)
    bool device_local{};
    bool host_visible{};
};
std::mutex g_census_map_mutex;
std::unordered_map<VkDeviceMemory, CensusMemory> g_census_maps;

void CensusNoteAllocation(VmaAllocator allocator, uint32_t type, VkDeviceMemory memory,
                          VkDeviceSize size) {
    VkMemoryPropertyFlags flags{};
    vmaGetMemoryTypeProperties(allocator, type, &flags);
    std::scoped_lock lock{g_census_map_mutex};
    g_census_maps[memory] = {size, nullptr, (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
                             (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0};
}

void CensusForgetAllocation(VkDeviceMemory memory) {
    std::scoped_lock lock{g_census_map_mutex};
    if (const auto it = g_census_maps.find(memory); it != g_census_maps.end()) {
        if (it->second.second != nullptr) {
            Common::MemCensus::Untag(it->second.second);
        }
        g_census_maps.erase(it);
    }
}

/// The census's size matches: sizes of VMA's live device memory of one kind. 16-64 MiB private
/// allocations (315, 8.4 GB committed, 1.35 GB resident at the market, 2026-11-16) looked like
/// the driver's copies of device-local memory (VMA's 32 MiB blocks and dedicated images).
std::vector<u64> CensusMemorySizes(bool device_local_only) {
    std::vector<u64> sizes;
    std::scoped_lock lock{g_census_map_mutex};
    for (const auto& [memory, entry] : g_census_maps) {
        if (device_local_only ? (entry.device_local && !entry.host_visible)
                              : (entry.host_visible && entry.second == nullptr)) {
            sizes.push_back(entry.first);
        }
    }
    return sizes;
}

VKAPI_ATTR VkResult VKAPI_CALL CensusMapMemory(VkDevice device, VkDeviceMemory memory,
                                               VkDeviceSize offset, VkDeviceSize size,
                                               VkMemoryMapFlags flags, void** data) {
    const VkResult result = g_real_map_memory(device, memory, offset, size, flags, data);
    if (result == VK_SUCCESS && data != nullptr && *data != nullptr) {
        std::scoped_lock lock{g_census_map_mutex};
        auto& entry = g_census_maps[memory];
        const VkDeviceSize bytes =
            size != VK_WHOLE_SIZE ? size : (entry.first > offset ? entry.first - offset : 0);
        entry.second = *data;
        Common::MemCensus::Tag("VMA host maps", *data, bytes);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL CensusUnmapMemory(VkDevice device, VkDeviceMemory memory) {
    {
        std::scoped_lock lock{g_census_map_mutex};
        if (const auto it = g_census_maps.find(memory); it != g_census_maps.end() &&
                                                         it->second.second != nullptr) {
            Common::MemCensus::Untag(it->second.second);
            it->second.second = nullptr;
        }
    }
    g_real_unmap_memory(device, memory);
}

u64 NanosecondsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

VKAPI_ATTR VkResult VKAPI_CALL KeptBlocksAllocateMemory(VkDevice device,
                                                        const VkMemoryAllocateInfo* allocate_info,
                                                        const VkAllocationCallbacks* allocator,
                                                        VkDeviceMemory* memory) {
    KeptBlocks* const kept = KeptBlocks::Active(device);
    if (kept == nullptr) {
        return g_real_allocate_memory(device, allocate_info, allocator, memory);
    }
    if (kept->Take(*allocate_info, *memory)) {
        return VK_SUCCESS;
    }
    const auto start = std::chrono::steady_clock::now();
    VkResult result = g_real_allocate_memory(device, allocate_info, allocator, memory);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY ||
        result == VK_ERROR_TOO_MANY_OBJECTS) {
        // The kept blocks may hold what the driver did not find: they all go, and nothing is
        // kept until the texture cache sets the limit again.
        kept->SetLimit(0);
        if (const u64 released = kept->Release(); released != 0) {
            const VkResult failed = result;
            result = g_real_allocate_memory(device, allocate_info, allocator, memory);
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Render_Vulkan,
                            "FRITCHII_KEEP_BLOCKS: a {} MiB allocation in memory type {} failed "
                            "({}); gave back the {} MiB kept and tried again ({})",
                            allocate_info->allocationSize >> 20, allocate_info->memoryTypeIndex,
                            static_cast<int>(failed), released >> 20, static_cast<int>(result));
            }
        }
    }
    if (result == VK_SUCCESS) {
        kept->CountDriverAllocation(allocate_info->allocationSize, NanosecondsSince(start));
        kept->Track(*allocate_info, *memory);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL KeptBlocksFreeMemory(VkDevice device, VkDeviceMemory memory,
                                                const VkAllocationCallbacks* allocator) {
    KeptBlocks* const kept = KeptBlocks::Active(device);
    if (kept == nullptr) {
        g_real_free_memory(device, memory, allocator);
        return;
    }
    if (kept->Keep(memory)) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    g_real_free_memory(device, memory, allocator);
    kept->CountDriverFree(NanosecondsSince(start));
}

} // Anonymous namespace

void Device::RemoveExtension(bool& extension, const std::string& extension_name) {
    extension = false;
    loaded_extensions.erase(extension_name);
}

void Device::RemoveExtensionIfUnsuitable(bool is_suitable, const std::string& extension_name) {
    if (loaded_extensions.contains(extension_name) && !is_suitable) {
        LOG_WARNING(Render_Vulkan, "Removing unsuitable extension {}", extension_name);
        this->RemoveExtension(is_suitable, extension_name);
    }
}

template <typename Feature>
void Device::RemoveExtensionFeature(bool& extension, Feature& feature,
                                    const std::string& extension_name) {
    // Unload extension.
    this->RemoveExtension(extension, extension_name);

    // Save sType and pNext for chain.
    VkStructureType sType = feature.sType;
    void* pNext = feature.pNext;

    // Clear feature struct and restore chain.
    feature = {};
    feature.sType = sType;
    feature.pNext = pNext;
}

template <typename Feature>
void Device::RemoveExtensionFeatureIfUnsuitable(bool is_suitable, Feature& feature,
                                                const std::string& extension_name) {
    if (loaded_extensions.contains(extension_name) && !is_suitable) {
        LOG_WARNING(Render_Vulkan, "Removing features for unsuitable extension {}", extension_name);
        this->RemoveExtensionFeature(is_suitable, feature, extension_name);
    }
}

namespace {
/// AMD FSR 4 runs AMD's own shaders inside AMD's runtime (Windows, AMD's driver). They need
/// float16 and, on RDNA 4, float8 on the device; the emulator's shaders use neither for it.
bool WantsAmdFsr4(VkDriverId driver_id) {
#ifdef _WIN32
    return driver_id == VK_DRIVER_ID_AMD_PROPRIETARY &&
           Settings::values.oot_anti_aliasing.GetValue() == Settings::OotAntiAliasing::Fsr4;
#else
    return false;
#endif
}
} // Anonymous namespace

Device::Device(VkInstance instance_, vk::PhysicalDevice physical_, VkSurfaceKHR surface,
               const vk::InstanceDispatch& dld_)
    : instance{instance_}, dld{dld_}, physical{physical_},
    format_properties(GetFormatProperties(physical)) {
    // Get suitability and device properties.
    const bool is_suitable = GetSuitability(surface != VkSurfaceKHR{});

    const VkDriverId driver_id = properties.driver.driverID;

    const bool is_radv = driver_id == VK_DRIVER_ID_MESA_RADV;
    const bool is_amd_driver =
        driver_id == VK_DRIVER_ID_AMD_PROPRIETARY || driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE;
    const bool is_amd = is_amd_driver || is_radv;

    const bool is_intel_windows = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS;
    if (driver_id == VK_DRIVER_ID_AMD_PROPRIETARY) {
        // The one follower's AMD safe default reads it at the split's start (later, on the GPU
        // thread).
        VideoCommon::OneFollower::NoteAmdProprietaryDriver(
            properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    }

    const bool is_nvidia = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
    const bool is_mvk = driver_id == VK_DRIVER_ID_MOLTENVK;
    const bool is_qualcomm = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY;
    const bool is_turnip = driver_id == VK_DRIVER_ID_MESA_TURNIP;

    if (!is_suitable)
        LOG_WARNING(Render_Vulkan, "Unsuitable driver - continuing anyways");

    if (is_nvidia) {
        nvidia_arch = GetNvidiaArchitecture(physical, supported_extensions);
    }

    SetupFamilies(surface);
    const auto queue_cis = GetDeviceQueueCreateInfos();

    // GetSuitability has already configured the linked list of features for us.
    // Reuse it here.
    const void* first_next = &features2;

    VkDeviceDiagnosticsConfigCreateInfoNV diagnostics_nv{};
    const bool use_diagnostics_nv = Settings::values.enable_nsight_aftermath && extensions.device_diagnostics_config;
    if (use_diagnostics_nv) {
        nsight_aftermath_tracker = std::make_unique<NsightAftermathTracker>();

        diagnostics_nv = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
            .pNext = &features2,
            .flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
                     VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
                     VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV,
        };
        first_next = &diagnostics_nv;
    }

    // FRITCHII_AMD_OVERALLOCATE (boot, default 0; port of zortos293/Lichii #1335 via Frizti/Fritchii
    // #61): AMD's own Windows driver fails device-local allocations once the heap is full unless
    // overallocation is allowed, which lets them spill into system memory as other drivers do.
    // 27 failing sessions upstream, 24 on AMD, mostly 8 GB cards running OoT.
    VkDeviceMemoryOverallocationCreateInfoAMD overallocation_amd{
        .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OVERALLOCATION_CREATE_INFO_AMD,
        .pNext = first_next,
        .overallocationBehavior = VK_MEMORY_OVERALLOCATION_BEHAVIOR_ALLOWED_AMD,
    };
    if (const char* overallocate = std::getenv("FRITCHII_AMD_OVERALLOCATE");
        overallocate != nullptr && overallocate[0] != '\0' && overallocate[0] != '0' &&
        is_amd_driver &&
        supported_extensions.contains(VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR_EXTENSION_NAME)) {
        loaded_extensions.insert(VK_AMD_MEMORY_OVERALLOCATION_BEHAVIOR_EXTENSION_NAME);
        first_next = &overallocation_amd;
        LOG_INFO(Render_Vulkan, "FRITCHII_AMD_OVERALLOCATE: VK_AMD_memory_overallocation_behavior "
                                "on, device memory may overflow into system memory");
    }

    is_blit_depth24_stencil8_supported = TestDepthStencilBlits(VK_FORMAT_D24_UNORM_S8_UINT);
    is_blit_depth32_stencil8_supported = TestDepthStencilBlits(VK_FORMAT_D32_SFLOAT_S8_UINT);
    is_optimal_astc_supported = ComputeIsOptimalAstcSupported();
    is_warp_potentially_bigger = !extensions.subgroup_size_control ||
                                 properties.subgroup_size_control.maxSubgroupSize > GuestWarpSize;

    is_integrated = properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    is_virtual = properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU;
    is_non_gpu = properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER ||
                 properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;

    supports_d24_depth =
        IsFormatSupported(VK_FORMAT_D24_UNORM_S8_UINT,
                          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT, FormatType::Optimal);

    supports_conditional_barriers = !is_intel_windows;

    CollectPhysicalMemoryInfo();
    // FRITCHII_SMALL_CARD (part RefuseScale): the presenting device only (not Force maximum
    // clocks' second one), before the renderer's caches read the scale.
    if (surface != VkSurfaceKHR{}) {
        VideoCore::SmallCard::RefuseScaleIfNeeded();
    }
    CollectToolingInfo();

    if (is_qualcomm) {
        must_emulate_scaled_formats = true;
        LOG_WARNING(Render_Vulkan, "Qualcomm drivers require scaled vertex format emulation.");
        has_broken_descriptor_aliasing = true;
        LOG_WARNING(Render_Vulkan, "Qualcomm drivers have broken descriptor aliasing.");
        LOG_WARNING(Render_Vulkan, "Qualcomm drivers have broken color write enable.");
        RemoveExtensionFeature(extensions.color_write_enable, features.color_write_enable,
                               VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME);
        LOG_WARNING(Render_Vulkan, "Qualcomm drivers have broken shader atomic int64.");
        RemoveExtensionFeature(extensions.shader_atomic_int64, features.shader_atomic_int64,
                               VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
        features.shader_atomic_int64.shaderBufferInt64Atomics = false;
        features.shader_atomic_int64.shaderSharedInt64Atomics = false;
        features.features.shaderInt64 = false;
        LOG_WARNING(Render_Vulkan, "Qualcomm drivers have broken workgroup memory explicit layout.");
        RemoveExtensionFeature(extensions.workgroup_memory_explicit_layout,
                               features.workgroup_memory_explicit_layout,
                               VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);

#if defined(__ANDROID__) && defined(ARCHITECTURE_arm64)
        // BCn patching only safe on Android 9+ (API 28+). Older versions crash on driver load.
        const auto major = (properties.properties.driverVersion >> 24) << 2;
        const auto minor = (properties.properties.driverVersion >> 12) & 0xFFFU;
        const auto vendor = properties.properties.vendorID;
        const auto patch_status = adrenotools_get_bcn_type(major, minor, vendor);
        const int api_level = android_get_device_api_level();

        bool should_patch_bcn = api_level >= 28;
        const bool bcn_debug_override = Settings::values.patch_old_qcom_drivers.GetValue();
        if (bcn_debug_override != should_patch_bcn) {
            LOG_WARNING(Render_Vulkan,
                "BCn patch debug override active: {} (auto-detected: {})",
                bcn_debug_override, should_patch_bcn);
            should_patch_bcn = bcn_debug_override;
        }

        if (patch_status == ADRENOTOOLS_BCN_PATCH) {
            if (should_patch_bcn) {
                LOG_INFO(Render_Vulkan,
                    "Patching Adreno driver to support BCn texture formats "
                    "(Android API {}, Driver {}.{})", api_level, major, minor);
                if (adrenotools_patch_bcn(
                        reinterpret_cast<void*>(dld.vkGetPhysicalDeviceFormatProperties))) {
                    OverrideBcnFormats(format_properties);
                } else {
                    LOG_ERROR(Render_Vulkan, "BCn patch failed! Driver code may now crash");
                }
            } else {
                LOG_WARNING(Render_Vulkan,
                    "BCn texture patching skipped for stability (Android API {} < 28). "
                    "Driver version {}.{} would support patching, but may crash on older Android.",
                    api_level, major, minor);
            }
        } else if (patch_status == ADRENOTOOLS_BCN_BLOB) {
            LOG_INFO(Render_Vulkan, "Adreno driver supports BCn textures natively (no patch needed)");
        } else {
            LOG_INFO(Render_Vulkan,
                "Adreno driver does not support BCn texture patching (Android API {}, Driver {}.{})",
                api_level, major, minor);
        }
#endif
    }

    if (is_nvidia) {
        const auto arch = GetNvidiaArch();
        if (arch >= NvidiaArchitecture::Arch_AmpereOrNewer) {
            LOG_WARNING(Render_Vulkan, "Ampere and newer have broken float16 math");
            features.shader_float16_int8.shaderFloat16 = false;
        }

        // Use hardware depth/stencil blits instead when available
        if (!extensions.shader_stencil_export) {
            LOG_INFO(Render_Vulkan,
                     "NVIDIA: VK_EXT_shader_stencil_export not supported, using hardware blits "
                     "for depth/stencil operations");
            LOG_INFO(Render_Vulkan, "  D24S8 hardware blit support: {}",
                     is_blit_depth24_stencil8_supported);
            LOG_INFO(Render_Vulkan, "  D32S8 hardware blit support: {}",
                     is_blit_depth32_stencil8_supported);

            if (!is_blit_depth24_stencil8_supported && !is_blit_depth32_stencil8_supported) {
                LOG_WARNING(Render_Vulkan,
                            "NVIDIA: Neither shader export nor hardware blits available for "
                            "depth/stencil. Performance may be degraded.");
            }
        }
    }

    sets_per_pool = 64;
    // AMD GCN4 and older are told apart by their lack of float16, read before float16 is turned
    // off for newer AMD drivers below: checked after that, every AMD GPU looked like GCN4 and lost
    // VK_EXT_sampler_filter_minmax (an RDNA3 Radeon 780M logged it).
    const bool amd_gcn4_or_older = is_amd && !features.shader_float16_int8.shaderFloat16;
    // CheezyEmu 132cc46c9e: Polaris by device ID, since newer AMD drivers report shaderFloat16 on
    // Polaris and the float16 test above misses them. Polaris 10/11/20/21/30 (RX 470-590),
    // Polaris 12 (RX 540/550) and the RX 580 2048SP. Only limits pipeline build threads.
    {
        const u32 device_id = properties.properties.deviceID;
        const bool is_polaris = properties.properties.vendorID == 0x1002 &&
                                ((device_id >= 0x67C0 && device_id <= 0x67FF) ||
                                 (device_id >= 0x6980 && device_id <= 0x699F) ||
                                 device_id == 0x6FDF);
        is_amd_legacy_gcn = is_amd_driver && (is_polaris || amd_gcn4_or_older);
    }
    const bool device_float16 = features.shader_float16_int8.shaderFloat16;
    if (is_amd_driver) {
        // AMD drivers need a higher amount of Sets per Pool in certain circumstances like in XC2.
        sets_per_pool = 96;

        // Disable VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT on AMD GCN4 and lower as it is broken.
        if (amd_gcn4_or_older) {
            LOG_WARNING(Render_Vulkan,
                        "AMD GCN4 and earlier have broken VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT");
            has_broken_cube_compatibility = true;
        }

        // AMD drivers (2026+) have broken float16 math on DKCR
        if (features.shader_float16_int8.shaderFloat16) {
            LOG_WARNING(Render_Vulkan,
                        "AMD drivers (2026+) have broken float16 math");
            features.shader_float16_int8.shaderFloat16 = false;
        }
    }

    if (extensions.sampler_filter_minmax && is_amd) {
        // Disable ext_sampler_filter_minmax on AMD GCN4 and lower as it is broken.
        if (amd_gcn4_or_older) {
            LOG_WARNING(Render_Vulkan,
                        "AMD GCN4 and earlier have broken VK_EXT_sampler_filter_minmax");
            RemoveExtension(extensions.sampler_filter_minmax,
                            VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME);
        }
    }

    if (features.shader_float16_int8.shaderFloat16 && is_intel_windows) {
        // Intel's compiler crashes when using fp16 on Astral Chain, disable it for the time being.
        LOG_WARNING(Render_Vulkan, "Intel has broken float16 math");
        features.shader_float16_int8.shaderFloat16 = false;
    }

    has_broken_compute =
        CheckBrokenCompute(properties.driver.driverID, properties.properties.driverVersion) &&
        !Settings::values.enable_compute_pipelines.GetValue();

    if (is_mvk) {
        LOG_WARNING(Render_Vulkan,
                    "MVK driver breaks when using more than 16 vertex attributes/bindings");
        properties.properties.limits.maxVertexInputAttributes =
            (std::min)(properties.properties.limits.maxVertexInputAttributes, 16U);
        properties.properties.limits.maxVertexInputBindings =
            (std::min)(properties.properties.limits.maxVertexInputBindings, 16U);
    }

    if (is_turnip || is_qualcomm) {
        LOG_WARNING(Render_Vulkan, "Driver requires higher-than-reported binding limits");
        properties.properties.limits.maxVertexInputBindings = 32;
    }

    const auto dyna_state = Settings::values.dyna_state.GetValue();
    switch (dyna_state) {
    case Settings::ExtendedDynamicState::Disabled:
        // Level 0: Disable all extended dynamic state extensions
        RemoveExtensionFeature(extensions.extended_dynamic_state, features.extended_dynamic_state,
                              VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
        RemoveExtensionFeature(extensions.extended_dynamic_state2, features.extended_dynamic_state2,
                              VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME);
        RemoveExtensionFeature(extensions.extended_dynamic_state3, features.extended_dynamic_state3,
                              VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
        dynamic_state3_blending = false;
        dynamic_state3_enables = false;
        break;
    case Settings::ExtendedDynamicState::EDS1:
        // Level 1: Enable EDS1, disable EDS2 and EDS3
        RemoveExtensionFeature(extensions.extended_dynamic_state2, features.extended_dynamic_state2,
                              VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME);
        RemoveExtensionFeature(extensions.extended_dynamic_state3, features.extended_dynamic_state3,
                              VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
        dynamic_state3_blending = false;
        dynamic_state3_enables = false;
        break;
    case Settings::ExtendedDynamicState::EDS2:
        // Level 2: Enable EDS1 + EDS2, disable EDS3
        RemoveExtensionFeature(extensions.extended_dynamic_state3, features.extended_dynamic_state3,
                              VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
        dynamic_state3_blending = false;
        dynamic_state3_enables = false;
        break;
    case Settings::ExtendedDynamicState::EDS3:
    default:
        // Level 3: Enable all (EDS1 + EDS2 + EDS3)
        break;
    }

    // VK_EXT_vertex_input_dynamic_state
    if (!Settings::values.vertex_input_dynamic_state.GetValue()) {
        RemoveExtensionFeature(extensions.vertex_input_dynamic_state, features.vertex_input_dynamic_state, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    }

    // Descriptors feature list
    {
        auto& descriptor_indexing = features.descriptor_indexing;
        descriptor_indexing.shaderInputAttachmentArrayDynamicIndexing = false;
        descriptor_indexing.shaderUniformTexelBufferArrayDynamicIndexing = false;
        descriptor_indexing.shaderStorageTexelBufferArrayDynamicIndexing = false;
        descriptor_indexing.shaderUniformBufferArrayNonUniformIndexing = false;
        descriptor_indexing.shaderStorageBufferArrayNonUniformIndexing = false;
        descriptor_indexing.shaderInputAttachmentArrayNonUniformIndexing = false;
        descriptor_indexing.descriptorBindingUniformBufferUpdateAfterBind = false;
        descriptor_indexing.descriptorBindingSampledImageUpdateAfterBind = false;
        descriptor_indexing.descriptorBindingStorageImageUpdateAfterBind = false;
        descriptor_indexing.descriptorBindingStorageBufferUpdateAfterBind = false;
        descriptor_indexing.descriptorBindingUniformTexelBufferUpdateAfterBind = false;
        descriptor_indexing.descriptorBindingStorageTexelBufferUpdateAfterBind = false;
        descriptor_indexing.descriptorBindingUpdateUnusedWhilePending = false;
        descriptor_indexing.descriptorBindingVariableDescriptorCount = false;
        descriptor_indexing.runtimeDescriptorArray = false;
    }

    // VK_EXT_descriptor_buffer requires VK_KHR_buffer_device_address
    if (extensions.descriptor_buffer && !features.buffer_device_address.bufferDeviceAddress) {
        LOG_WARNING(Render_Vulkan, "Descriptor buffer needs buffer device address, disabling.");
        RemoveExtensionFeature(extensions.descriptor_buffer, features.descriptor_buffer,
                               VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
    }
    // FRITCHII_OCCL_FLAG: the session's value (a switch, else the Occlusion speed-up setting),
    // read once per game boot.
    const bool occl_flag_wanted = VideoCommon::OcclFlag::ReadAtBoot(
        VideoCommon::OcclFlag::FromSetting(Settings::values.occlusion_speedup.GetValue()));
#ifdef EGLANTINE_STREAMLINE
    // NVIDIA's DLSS needs buffer device addresses: keep the feature enabled for
    // it, but leave the emulator's own buffers as they are without descriptor buffers.
    const bool dlss_wants_device_address =
        streamline_extensions && features.buffer_device_address.bufferDeviceAddress;
    if (!extensions.descriptor_buffer && dlss_wants_device_address) {
        extensions.buffer_device_address = false;
    } else
#endif
    if (!extensions.descriptor_buffer && occl_flag_wanted &&
        instance_version >= VK_API_VERSION_1_2 &&
        features.buffer_device_address.bufferDeviceAddress) {
        // FRITCHII_OCCL_FLAG: the flag pipelines store through buffer device addresses. As for
        // DLSS above, the feature stays enabled while the emulator's own buffers and VMA stay
        // as they are without descriptor buffers.
        extensions.buffer_device_address = false;
        occl_flag_device_address = true;
    } else if (!extensions.descriptor_buffer) {
        RemoveExtensionFeature(extensions.buffer_device_address, features.buffer_device_address,
                               VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    }
    // FRITCHII_OCCL_FLAG: whichever branch ran, the feature is still enabled when it was not
    // removed (the DLSS branch keeps it too).
    if (occl_flag_wanted) {
        occl_flag_device_address = instance_version >= VK_API_VERSION_1_2 &&
                                   features.buffer_device_address.bufferDeviceAddress != 0;
        LOG_INFO(Render_Vulkan,
                 "Occlusion flags (FRITCHII_OCCL_FLAG): buffer device address {} (instance {}.{})",
                 occl_flag_device_address ? "enabled for the flag area" : "not available",
                 VK_API_VERSION_MAJOR(instance_version), VK_API_VERSION_MINOR(instance_version));
    }

    // The float16 workarounds above keep shaders from doing float16 arithmetic. Cooperative
    // matrices only pass 16-bit floats to the tensor cores and back, so they keep the device
    // feature, which no other shader then uses.
    float16_math = features.shader_float16_int8.shaderFloat16;
    if (extensions.cooperative_matrix) {
        features.shader_float16_int8.shaderFloat16 = VK_TRUE;
        LOG_INFO(Render_Vulkan, "VK_KHR_cooperative_matrix enabled for guest tensor operations");
    }

    // AMD FSR 4 runs AMD's own shaders inside AMD's runtime, which need float16 on the device.
    // Guest shaders still follow float16_math, so the AMD float16 workaround above stays in force.
    if (device_float16 && WantsAmdFsr4(driver_id)) {
        features.shader_float16_int8.shaderFloat16 = VK_TRUE;
    }
    // FRITCHII_QUEUE_PRIORITY (boot, default 0; helper BN, 2026-11-16, docs/fritchii/switches.md):
    // 1 = the device's queues ask for high global priority (VK_KHR/EXT_global_priority), 2 =
    // realtime. The driver may refuse (not permitted, or no support): the device is then made as
    // before, and the log says so.
    std::vector<VkDeviceQueueCreateInfo> priority_queue_cis(queue_cis.begin(), queue_cis.end());
    VkDeviceQueueGlobalPriorityCreateInfoKHR queue_global_priority{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
        .pNext = nullptr,
        .globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
    };
    const char* global_priority_extension = nullptr;
    if (const char* queue_priority = std::getenv("FRITCHII_QUEUE_PRIORITY");
        queue_priority != nullptr && std::atoi(queue_priority) > 0) {
        if (std::atoi(queue_priority) >= 2) {
            queue_global_priority.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR;
        }
        if (supported_extensions.contains(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME)) {
            global_priority_extension = VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME;
        } else if (supported_extensions.contains(VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME)) {
            global_priority_extension = VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME;
        } else {
            LOG_INFO(Render_Vulkan, "FRITCHII_QUEUE_PRIORITY: the driver has no global priority "
                                    "extension, queues keep the default priority");
        }
        if (global_priority_extension != nullptr &&
            !loaded_extensions.contains(global_priority_extension)) {
            loaded_extensions.insert(global_priority_extension);
            for (auto& ci : priority_queue_cis) {
                ci.pNext = &queue_global_priority;
            }
        } else {
            global_priority_extension = nullptr;
        }
    }
    if (global_priority_extension != nullptr) {
        try {
            logical = vk::Device::Create(physical, priority_queue_cis,
                                         ExtensionListForVulkan(loaded_extensions), first_next, dld);
            LOG_INFO(Render_Vulkan, "FRITCHII_QUEUE_PRIORITY: queues made with {} global priority ({})",
                     queue_global_priority.globalPriority == VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR
                         ? "realtime"
                         : "high",
                     global_priority_extension);
        } catch (const vk::Exception& exception) {
            LOG_WARNING(Render_Vulkan,
                        "FRITCHII_QUEUE_PRIORITY: the driver refused the global priority ({}), "
                        "device made with the default priority",
                        exception.what());
            loaded_extensions.erase(std::string{global_priority_extension});
            global_priority_extension = nullptr;
        }
    }
    if (global_priority_extension == nullptr) {
        logical = vk::Device::Create(physical, queue_cis, ExtensionListForVulkan(loaded_extensions), first_next, dld);
    }

    graphics_queue = logical.GetQueue(graphics_family);
    present_queue = logical.GetQueue(present_family);
    if (const std::optional<u32> family = LinkHogFamily()) {
        link_hog_queue = logical.GetQueue(*family);
        link_hog_family = *family;
        has_link_hog_queue = true;
    }
    if (WantsSideQueue()) {
        side_queue = logical.GetQueue(graphics_family, 1);
        has_side_queue = true;
        LOG_INFO(Render_Vulkan, "FRITCHII_READ_SIDE_COPY: side queue made (graphics family {}, "
                                "queue 1)",
                 graphics_family);
    }

#ifdef EGLANTINE_STREAMLINE
    if (streamline_extensions) {
        streamline_attached =
            Streamline::AttachDevice(instance, physical, *logical, graphics_family, dld);
    } else if (Streamline::WantsDevice()) {
        Streamline::Stop();
    }
#endif

    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = dld.vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = dld.vkGetDeviceProcAddr;
    // FRITCHII_KEEP_BLOCKS set at boot (any value, 0 included, so a session can toggle it live):
    // VMA's memory calls go through the kept blocks (CreateKeptBlocks). Unset: VMA calls the
    // driver directly, as before.
    if (std::getenv("FRITCHII_KEEP_BLOCKS") != nullptr && dld.vkAllocateMemory != nullptr &&
        dld.vkFreeMemory != nullptr) {
        g_real_allocate_memory = dld.vkAllocateMemory;
        g_real_free_memory = dld.vkFreeMemory;
        functions.vkAllocateMemory = KeptBlocksAllocateMemory;
        functions.vkFreeMemory = KeptBlocksFreeMemory;
    }
    const bool census_maps = Common::MemCensus::Mode() != 0 && dld.vkMapMemory != nullptr &&
                             dld.vkUnmapMemory != nullptr;
    if (census_maps) {
        g_real_map_memory = dld.vkMapMemory;
        g_real_unmap_memory = dld.vkUnmapMemory;
        functions.vkMapMemory = CensusMapMemory;
        functions.vkUnmapMemory = CensusUnmapMemory;
        Common::MemCensus::SetSizeSource("device-local memory (driver copy?)",
                                         [] { return CensusMemorySizes(true); });
        Common::MemCensus::SetSizeSource("host-visible memory not mapped",
                                         [] { return CensusMemorySizes(false); });
    }
    // EDEN_NX2_VMA_NO_PREFER_DEDICATED (boot, default on, 0 = off; ported;
    // with 32 MiB blocks market A/B 2026-11-11: off 41.4/40.3, on 41.5/42.8 fps,
    // same VRAM):
    // images the driver only *prefers* in their own memory (NVIDIA: render and depth targets) go
    // in VMA's blocks instead. OoT makes and frees ~2-3 short-lived depth targets a frame (shadow
    // cascades sized anew), and each dedicated one is a kernel allocate, bind and free on the GPU
    // thread. Required dedicated allocations stay dedicated.
    static PFN_vkGetImageMemoryRequirements2 real_image_requirements2 = nullptr;
    if (const char* value = std::getenv("EDEN_NX2_VMA_NO_PREFER_DEDICATED");
        value == nullptr || value[0] != '0') {
        real_image_requirements2 = reinterpret_cast<PFN_vkGetImageMemoryRequirements2>(
            dld.vkGetDeviceProcAddr(*logical, "vkGetImageMemoryRequirements2"));
        if (real_image_requirements2 != nullptr) {
            LOG_INFO(Render_Vulkan, "VMA: preferred dedicated allocations go in blocks");
            functions.vkGetImageMemoryRequirements2KHR =
                [](VkDevice device, const VkImageMemoryRequirementsInfo2* info,
                   VkMemoryRequirements2* requirements) {
                    real_image_requirements2(device, info, requirements);
                    for (auto* next = static_cast<VkBaseOutStructure*>(requirements->pNext);
                         next != nullptr; next = next->pNext) {
                        if (next->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS) {
                            auto* dedicated = reinterpret_cast<VkMemoryDedicatedRequirements*>(next);
                            if (!dedicated->requiresDedicatedAllocation) {
                                dedicated->prefersDedicatedAllocation = VK_FALSE;
                            }
                        }
                    }
                };
        }
    }

    // EDEN_NX2_ASYNC_IMAGE_FREE (boot, default OFF since 2026-11-11; =1 on): the texture cache destroys the images
    // it deleted on a thread of its own (with dedicated allocations each is a vkFreeMemory into
    // the driver: ~22 ms/s on the GPU back thread at the Castle Town market). VMA then has to lock
    // internally, which it does without EXTERNALLY_SYNCHRONIZED (an uncontended lock per
    // allocation call, its default mode).
    allocator_thread_safe = [] {
        // Off by default: with the free thread on, the back's unzoned stall time ("other") was
        // 185 ms/s vs 129 with it off at the market (GPU_PROFILE=2, 46.4 vs 48.8 fps): its frees
        // contend with the back's allocations for VMA's and the driver's locks. To revisit as
        // "free only while the back sleeps" (Fable's fable_answer_next.md, Q1).
        const char* value = std::getenv("EDEN_NX2_ASYNC_IMAGE_FREE");
        return value != nullptr && value[0] == '1';
    }();
    VmaAllocatorCreateFlags flags =
        allocator_thread_safe ? VmaAllocatorCreateFlags{0}
                              : VmaAllocatorCreateFlags{VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT};
    if (extensions.memory_budget) {
        flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (extensions.buffer_device_address) {
        flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    // EDEN_NX2_VRAM_LIMIT_MIB: VMA fails allocations past the limit, like a smaller card.
    std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heap_size_limits;
    heap_size_limits.fill(VK_WHOLE_SIZE);
    if (heap_size_limit != 0) {
        const auto memory_properties = physical.GetMemoryProperties().memoryProperties;
        for (u32 heap = 0; heap < memory_properties.memoryHeapCount; ++heap) {
            if (memory_properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                heap_size_limits[heap] = heap_size_limit;
            }
        }
    }
    // Counts the allocator's vkAllocateMemory / vkFreeMemory calls (blocks and dedicated
    // memory) for the "Churn (600 frames)" line.
    // With FRITCHII_MEM_CENSUS they also note each memory's size for the census's map tags.
    static const VmaDeviceMemoryCallbacks churn_callbacks{
        .pfnAllocate = [](VmaAllocator allocator, uint32_t type, VkDeviceMemory memory,
                          VkDeviceSize size, void*) {
            VideoCommon::Churn::Add(VideoCommon::Churn::DeviceAllocations);
            if (g_real_map_memory != nullptr) {
                CensusNoteAllocation(allocator, type, memory, size);
            }
        },
        .pfnFree = [](VmaAllocator, uint32_t, VkDeviceMemory memory, VkDeviceSize, void*) {
            VideoCommon::Churn::Add(VideoCommon::Churn::DeviceFrees);
            if (g_real_map_memory != nullptr) {
                CensusForgetAllocation(memory);
            }
        },
        .pUserData = nullptr,
    };
    const VmaAllocatorCreateInfo allocator_info{
            .flags = flags,
            .physicalDevice = physical,
            .device = *logical,
            // 16 MiB blocks on discrete GPUs too: with 256 MiB ones, Switch 2 titles' mix of
            // texture sizes left ~38% of the image blocks empty (2 GB of an 8 GB card in title
            // 1), which the texture cache counted as used and kept evicting textures for. With
            // 64 MiB ones textures created and deleted by the thousand still left ~1.7 GB of holes
            // after ten minutes, with 32 MiB ones ~0.5 GB: a block drains only once all of its
            // images are gone, and fewer per block drain sooner. VMA gives allocations over half a
            // block (8 MiB) their own memory, which goes back to the driver when they are freed.
            // EDEN_NX2_VMA_BLOCK_MIB (boot, default 32; ported): OoT docked
            // makes 6-9 MiB shadow-map depth targets anew every frame, each its own allocation
            // over half a 16 MiB block. 16 = the old size (fewer holes in long sessions, above).
            .preferredLargeHeapBlockSize = [] {
                const char* value = std::getenv("EDEN_NX2_VMA_BLOCK_MIB");
                const u64 default_mib = Common::HostHasSmallVideoMemory() ? 16 : 32;
                const u64 mib = value != nullptr ? std::strtoull(value, nullptr, 10) : default_mib;
                return (mib != 0 ? mib : 16) * 1024u * 1024u;
            }(),
            .pAllocationCallbacks = nullptr,
            .pDeviceMemoryCallbacks = &churn_callbacks,
            .pHeapSizeLimit = heap_size_limit != 0 ? heap_size_limits.data() : nullptr,
            .pVulkanFunctions = &functions,
            .instance = instance,
            .vulkanApiVersion = ApiVersion(),
            .pTypeExternalMemoryHandleTypes = nullptr,
    };

    vk::Check(vmaCreateAllocator(&allocator_info, &allocator));

    // FRITCHII_MEMSTAGE_SPLIT_ALLOC (boot, default on; =0 off): a second allocator, same settings,
    // for the buffer cache runtime and its staging pool (the memory stage's design, 2.4 and build
    // step 3). VMA allocators here are externally synchronized, so once the buffer cache runs on a
    // thread of its own it cannot share the back's. Images stay on the main one. Creating it takes
    // no memory: VMA allocates blocks on first use, and both see the same driver budget (with
    // VK_EXT_memory_budget the usage they compare against it is the whole process's).
    const bool split_allocators = [] {
        const char* value = std::getenv("FRITCHII_MEMSTAGE_SPLIT_ALLOC");
        return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    if (split_allocators) {
        // FRITCHII_SMALL_CARD (part BufferBlocks, boot): the buffer allocator held ~1.4 GiB of
        // blocks for ~0.7 GiB of buffers at OoT's market (8 GB and 6 GB card simulations), the
        // largest part of the block slack the driver counts as used. Buffers of 8 to 16 MiB shared
        // 32 MiB blocks; with 16 MiB blocks VMA gives them exact memory of their own (over half a
        // block), which goes back to the driver whole when they are freed. Buffers up to 8 MiB
        // keep their separate pool. FRITCHII_SMALL_CARD_BUFFER_BLOCK_MIB (boot, default 16)
        // tries other sizes.
        VmaAllocatorCreateInfo buffer_allocator_info = allocator_info;
        if (VideoCore::SmallCard::On(VideoCore::SmallCard::BufferBlocks)) {
            const char* value = std::getenv("FRITCHII_SMALL_CARD_BUFFER_BLOCK_MIB");
            const u64 mib = value != nullptr ? std::strtoull(value, nullptr, 10) : 16;
            if (mib >= 4 && mib <= 256) {
                buffer_allocator_info.preferredLargeHeapBlockSize = mib << 20;
            }
            LOG_INFO(Render_Vulkan,
                     "Small card profile: buffer allocator blocks of {} MiB (images keep {} MiB)",
                     buffer_allocator_info.preferredLargeHeapBlockSize >> 20,
                     allocator_info.preferredLargeHeapBlockSize >> 20);
        }
        if (vmaCreateAllocator(&buffer_allocator_info, &buffer_allocator) != VK_SUCCESS) {
            LOG_WARNING(Render_Vulkan, "Buffer VMA allocator could not be made; one allocator");
            buffer_allocator = nullptr;
        }
        // The memory stage counts the calls on it and who makes them (memory_stage_link.cpp).
        VideoCommon::MemoryStageLink::WatchBufferAllocator(buffer_allocator);
    }
    if (functions.vkAllocateMemory == KeptBlocksAllocateMemory) {
        CreateKeptBlocks();
    }

    owns_static_pipeline_cache = surface != VkSurfaceKHR{};
    LoadStaticPipelineCache();

    // Initialize GPU logging if enabled
    InitializeGPULogging();
}

Device::~Device() {
#ifdef EGLANTINE_STREAMLINE
    if (streamline_attached) {
        Streamline::Stop();
    }
#endif
    SaveStaticPipelineCache();
    ShutdownGPULogging();
    // FRITCHII_KEEP_BLOCKS: kept blocks go back to the driver, and the blocks VMA frees below go
    // straight to it.
    if (kept_blocks) {
        kept_blocks->Close();
    }
    // Everything allocated from either allocator (the renderer's caches) is gone by now.
    const std::array<VmaAllocator, 2> owners{allocator, buffer_allocator};
    for (size_t owner = 0; owner < owners.size(); ++owner) {
        for (auto& pools : separate_pools[owner]) {
            for (VmaPool pool : pools) {
                if (pool != VK_NULL_HANDLE) {
                    vmaDestroyPool(owners[owner], pool);
                }
            }
        }
    }
    if (buffer_allocator != nullptr) {
        VideoCommon::MemoryStageLink::WatchBufferAllocator(nullptr);
        vmaDestroyAllocator(buffer_allocator);
    }
    vmaDestroyAllocator(allocator);
}

void Device::CreateKeptBlocks() {
    const VkPhysicalDeviceMemoryProperties memory_properties =
        physical.GetMemoryProperties().memoryProperties;
    auto kept = std::make_unique<KeptBlocks>(*logical, g_real_allocate_memory, g_real_free_memory,
                                             memory_properties);
    // The memory type images take from VMA's default blocks.
    const VkImageCreateInfo probe_ci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {1024, 1024, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VmaAllocationCreateInfo image_alloc{};
    image_alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    image_alloc.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    u32 image_type = VK_MAX_MEMORY_TYPES;
    if (vmaFindMemoryTypeIndexForImageInfo(allocator, &probe_ci, &image_alloc, &image_type) !=
        VK_SUCCESS) {
        image_type = VK_MAX_MEMORY_TYPES;
    }
    if (is_integrated || !kept->Keeps(image_type)) {
        // An integrated GPU's device-local memory is system memory, and its types are mostly
        // host-visible: nothing to keep.
        LOG_INFO(Render_Vulkan,
                 "Kept blocks (FRITCHII_KEEP_BLOCKS): off, images take memory type {} which is "
                 "not kept{}",
                 image_type, is_integrated ? " (integrated GPU)" : "");
        return;
    }
    kept->Activate();
    kept_blocks = std::move(kept);
    // FRITCHII_KEEP_BLOCKS_PREMAKE=1 (boot): the cap's worth of VMA default-size blocks now.
    const u64 cap = KeptBlocks::SwitchCap();
    const char* premake = std::getenv("FRITCHII_KEEP_BLOCKS_PREMAKE");
    u32 made = 0;
    double ms = 0.0;
    const VkDeviceSize block_size = [] {
        const char* value = std::getenv("EDEN_NX2_VMA_BLOCK_MIB");
        const u64 mib = value != nullptr ? std::strtoull(value, nullptr, 10) : 32;
        return VkDeviceSize{(mib != 0 ? mib : 16) * 1024u * 1024u};
    }();
    if (premake != nullptr && std::atoi(premake) != 0 && cap != 0) {
        // The flags VMA gives its blocks (VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT).
        const VkMemoryAllocateFlags flags =
            extensions.buffer_device_address
                ? static_cast<VkMemoryAllocateFlags>(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT)
                : VkMemoryAllocateFlags{0};
        const auto start = std::chrono::steady_clock::now();
        made = kept_blocks->Make(image_type, block_size, static_cast<u32>(cap / block_size),
                                 flags);
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                 .count();
    }
    LOG_INFO(Render_Vulkan,
             "Kept blocks (FRITCHII_KEEP_BLOCKS, live): cap {} MiB now (at most 512), memory type "
             "{}; premade {} blocks of {} MiB in {:.1f} ms (FRITCHII_KEEP_BLOCKS_PREMAKE)",
             cap >> 20, image_type, made, block_size >> 20, ms);
}

void Device::LimitKeptBlocks(u64 budget, u64 critical) const {
    if (!kept_blocks) {
        return;
    }
    // The live usage (the kept blocks counted as free) may fill to the cache's critical use; the
    // kept blocks get what it leaves under that, plus half of the margin above it, so the
    // process stays at least half the margin under the budget (Windows lowers the budget as the
    // card fills).
    const u64 ceiling = critical < budget ? critical + (budget - critical) / 2 : budget;
    const u64 live = GetDeviceMemoryUsage();
    kept_blocks->SetLimit(ceiling > live ? ceiling - live : 0);
}

u64 Device::ReleaseKeptBlocks() const {
    if (!kept_blocks) {
        return 0;
    }
    return kept_blocks->Release();
}

void Device::TickKeptBlocks() const {
    if (kept_blocks) {
        kept_blocks->Tick();
    }
}

u64 Device::KeptBlockBytes() const noexcept {
    return kept_blocks ? kept_blocks->KeptBytes() : 0;
}

VkDeviceSize Device::DeviceBufferBlockSize() noexcept {
    // FRITCHII_BUFFER_POOL_BLOCK_MB=<n> (boot, default 0 = 16 MiB): n MiB blocks (1..64) for the
    // buffer cache's device-local pool, buffers over half a block take the default blocks.
    // 4 = small blocks with a 2 MiB cut-off: a 16 MiB block holds a few buffers and stays
    // pinned while any of them lives, mostly empty ("buffer pool" slack in VRAM_OWNERS).
    // FRITCHII_SMALL_CARD (part BufferPoolBlocks): 4 unless the switch says otherwise. The first
    // call comes with the first buffer, after the device decided the profile.
    static const VkDeviceSize size = [] {
        const char* value = std::getenv("FRITCHII_BUFFER_POOL_BLOCK_MB");
        if (value == nullptr || value[0] == '\0') {
            if (VideoCore::SmallCard::On(VideoCore::SmallCard::BufferPoolBlocks)) {
                LOG_INFO(Render_Vulkan, "Small card profile: device buffer pool blocks of 4 MiB, "
                                        "buffers over 2048 KiB take the default blocks");
                return VkDeviceSize{4} << 20;
            }
            return SeparatePoolBlockSize;
        }
        const long mib = std::strtol(value, nullptr, 10);
        if (mib <= 0) {
            return SeparatePoolBlockSize;
        }
        const VkDeviceSize clamped = static_cast<VkDeviceSize>((std::min)(mib, 64L)) << 20;
        LOG_INFO(Render_Vulkan,
                 "FRITCHII_BUFFER_POOL_BLOCK_MB: device buffer pool blocks {} MiB, buffers over "
                 "{} KiB take the default blocks",
                 clamped >> 20, (clamped / 2) >> 10);
        return clamped;
    }();
    return size;
}

VmaPool Device::GetSeparatePool(VmaAllocator owner, SeparatePool kind, u32 memory_type) const {
    std::scoped_lock lk{separate_pools_mutex};
    // Each allocator has its own pools: a pool belongs to the allocator that made it.
    const size_t owner_index = buffer_allocator != nullptr && owner == buffer_allocator ? 1 : 0;
    owner = owner_index == 1 ? buffer_allocator : allocator;
    VmaPool& pool = separate_pools[owner_index][static_cast<u32>(kind)][memory_type];
    if (pool == VK_NULL_HANDLE) {
        const VmaPoolCreateInfo ci{
            .memoryTypeIndex = memory_type,
            .flags = 0,
            .blockSize = kind == SeparatePool::DeviceBuffers ? DeviceBufferBlockSize()
                                                             : SeparatePoolBlockSize,
            .minBlockCount = 0,
            .maxBlockCount = 0,
            .priority = 0.f,
            .minAllocationAlignment = 0,
            .pMemoryAllocateNext = nullptr,
        };
        if (vmaCreatePool(owner, &ci, &pool) != VK_SUCCESS) {
            pool = VK_NULL_HANDLE;
        } else {
            // Named, so VMA's detailed map (FRITCHII_VRAM_HOLES) tells the pools apart.
            vmaSetPoolName(owner, pool, kind == SeparatePool::SparsePages ? "sparse" : "buffer");
        }
    }
    return pool;
}

VmaPool Device::PeekSeparatePool(VmaAllocator owner, SeparatePool kind, u32 memory_type) const {
    if (memory_type >= VK_MAX_MEMORY_TYPES) {
        return VK_NULL_HANDLE;
    }
    std::scoped_lock lk{separate_pools_mutex};
    // Only `owner`'s own pools: with one allocator (no split) the buffer slot is never used.
    if (owner == allocator) {
        return separate_pools[0][static_cast<u32>(kind)][memory_type];
    }
    if (buffer_allocator != nullptr && owner == buffer_allocator) {
        return separate_pools[1][static_cast<u32>(kind)][memory_type];
    }
    return VK_NULL_HANDLE;
}

void Device::LoadStaticPipelineCache() {
    const auto create = [this](size_t size, const void* data) {
        static_pipeline_cache = logical.CreatePipelineCache({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = size,
            .pInitialData = data,
        });
    };
    if (!owns_static_pipeline_cache) {
        create(0, nullptr);
        return;
    }
    const auto filename = StaticPipelineCacheFilename();
    if (filename.empty()) {
        create(0, nullptr);
        return;
    }
    std::vector<char> data;
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            create(0, nullptr);
            return;
        }
        file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        const size_t total = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);
        std::array<char, 8> magic{};
        u32 version{};
        if (total < magic.size() + sizeof(version)) {
            create(0, nullptr);
            return;
        }
        file.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != STATIC_CACHE_MAGIC_NUMBER || version != STATIC_CACHE_VERSION) {
            create(0, nullptr);
            return;
        }
        data.resize(total - magic.size() - sizeof(version));
        file.read(data.data(), static_cast<std::streamsize>(data.size()));
    } catch (const std::ios_base::failure& e) {
        create(0, nullptr);
        return;
    }
    try {
        create(data.size(), data.empty() ? nullptr : data.data());
    } catch (const vk::Exception& e) {
        LOG_ERROR(Render_Vulkan, "Driver rejected the static pipeline cache ({}), starting empty",
                  e.what());
        create(0, nullptr);
    }
}

void Device::SaveStaticPipelineCache() const {
    if (!owns_static_pipeline_cache || !static_pipeline_cache) {
        return;
    }
    const auto filename = StaticPipelineCacheFilename();
    if (filename.empty()) {
        return;
    }
    size_t size = 0;
    std::vector<char> data;
    static_pipeline_cache.Read(&size, nullptr);
    if (size == 0) {
        return;
    }
    data.resize(size);
    static_pipeline_cache.Read(&size, data.data());
    // Written beside the cache and renamed over it, so an exit mid-write keeps the old one.
    auto temporary = filename;
    temporary += ".tmp";
    try {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.exceptions(std::ofstream::failbit);
        if (!file.is_open()) {
            return;
        }
        file.write(STATIC_CACHE_MAGIC_NUMBER.data(), STATIC_CACHE_MAGIC_NUMBER.size())
            .write(reinterpret_cast<const char*>(&STATIC_CACHE_VERSION),
                   sizeof(STATIC_CACHE_VERSION))
            .write(data.data(), static_cast<std::streamsize>(size));
        file.close();
    } catch (const std::ios_base::failure& e) {
        Common::FS::RemoveFile(temporary);
        return;
    }
    std::error_code error;
    std::filesystem::rename(temporary, filename, error);
    if (error) {
        LOG_ERROR(Render_Vulkan, "Couldn't replace the static pipeline cache: {}", error.message());
        Common::FS::RemoveFile(temporary);
    }
}

VkFormat Device::GetSupportedFormat(VkFormat wanted_format, VkFormatFeatureFlags wanted_usage,
                                    FormatType format_type) const {
    if (IsFormatSupported(wanted_format, wanted_usage, format_type)) {
        return wanted_format;
    }
    // The wanted format is not supported by hardware, search for alternatives
    const VkFormat* alternatives = GetFormatAlternatives(wanted_format);
    if (alternatives == nullptr) {
        LOG_ERROR(Render_Vulkan,
                  "Format={} with usage={} and type={} has no defined alternatives and host "
                  "hardware does not support it",
                  wanted_format, wanted_usage, format_type);
        return wanted_format;
    }

    std::size_t i = 0;
    for (VkFormat alternative = *alternatives; alternative; alternative = alternatives[++i]) {
        if (!IsFormatSupported(alternative, wanted_usage, format_type)) {
            continue;
        }
        LOG_DEBUG(Render_Vulkan,
                  "Emulating format={} with alternative format={} with usage={} and type={}",
                  wanted_format, alternative, wanted_usage, format_type);
        return alternative;
    }

    // No alternatives found, panic
    LOG_ERROR(Render_Vulkan,
              "Format={} with usage={} and type={} is not supported by the host hardware and "
              "doesn't support any of the alternatives",
              wanted_format, wanted_usage, format_type);
    return wanted_format;
}

void Device::ReportLoss(std::source_location where) const {
    // The first thread to see the loss reports it and ends the process; the others wait for that.
    static std::atomic_flag reported;
    if (reported.test_and_set()) {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::hours{1});
        }
    }
    // The report touches the driver, the log and every thread's stack, on a process in an unknown
    // state: whatever gets stuck, the process still ends and the player gets no frozen window.
    // FRITCHII_RGD_WAIT_S (default 0, at most 600): seconds the process stays up after the report
    // so a Radeon GPU Detective capture can finish its dump (the end-process watch waits that
    // much longer too).
    static const u32 rgd_wait_s = [] {
        const char* const value = std::getenv("FRITCHII_RGD_WAIT_S");
        const long parsed = value != nullptr ? std::strtol(value, nullptr, 10) : 0;
        return static_cast<u32>(std::clamp<long>(parsed, 0, 600));
    }();
    Common::ShutdownWatch::EndProcessAfter(
        std::chrono::seconds{(nsight_aftermath_tracker ? 50 : 35) + rgd_wait_s});
    // FRITCHII_VA_REPORT: the destroys of the last 60 s (also to destroy_report.txt), first, with
    // the watch armed: a stuck dump still ends the process.
    vk::DumpRecentDestroys(60000);
    u64 heap_size{};
    u64 heap_usage{};
    u64 heap_budget{};
    const bool has_heap = MainHeapNumbers(heap_size, heap_usage, heap_budget);
    LOG_CRITICAL(Render_Vulkan,
                 "Device loss occurred! Seen at {}:{}; GPU {}, driver {} ({:#x}); video memory {}",
                 where.file_name(), where.line(), GetModelName(), GetDriverName(),
                 GetDriverVersion(),
                 has_heap ? fmt::format("{} MiB used of {} MiB, budget {} MiB", heap_usage >> 20,
                                        heap_size >> 20, heap_budget >> 20)
                          : std::string{"unknown"});
    PipelineBinary::NoteDeviceLoss();
    ReportDeviceFault();
    // FRITCHII_SLOW_GPU: what the GPU was last given (NVIDIA's fault info has no addresses).
    Common::Nx2Hitch::LogGpuBreadcrumbs("Device loss");
    // What every thread was doing, as a hang report gives it.
    Common::Nx2Hitch::LogThreadReport("Device loss", false, 0);

    if (nsight_aftermath_tracker) {
        // Give Nsight Aftermath time to write its dump.
        std::this_thread::sleep_for(std::chrono::seconds{15});
    }
    if (rgd_wait_s != 0) {
        LOG_CRITICAL(Render_Vulkan,
                     "Device loss: waiting {} s for a GPU crash dump (FRITCHII_RGD_WAIT_S)",
                     rgd_wait_s);
        std::this_thread::sleep_for(std::chrono::seconds{rgd_wait_s});
    }
    static constexpr const char* FatalError = "the GPU driver lost the Vulkan device";
#ifdef _WIN32
    Common::Windows::ReportFatalError(FatalError);
#elif defined(__linux__) && !defined(__ANDROID__)
    Common::Linux::LogFatalError(FatalError);
#endif
    LOG_CRITICAL(Render_Vulkan, "Closing Fritchii: {}", FatalError);
    // Flushes the log and skips DLL detach, which would run the driver's teardown on the lost
    // device. An exception from here would only reach std::terminate on a worker thread.
    Common::ShutdownWatch::EndProcessNow();
}

void Device::ReportDeviceFault() const {
    if (!device_fault_features.deviceFault || !dld.vkGetDeviceFaultInfoEXT) {
        LOG_CRITICAL(Render_Vulkan, "Device fault info: not supported by the driver");
        return;
    }
    VkDeviceFaultCountsEXT counts{.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    VkResult result = dld.vkGetDeviceFaultInfoEXT(*logical, &counts, nullptr);
    if (result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Device fault info: query failed ({})",
                     static_cast<int>(result));
        return;
    }
    std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor_infos(counts.vendorInfoCount);
    counts.vendorBinarySize = 0;
    VkDeviceFaultInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
        .pAddressInfos = addresses.data(),
        .pVendorInfos = vendor_infos.data(),
    };
    result = dld.vkGetDeviceFaultInfoEXT(*logical, &counts, &info);
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        LOG_CRITICAL(Render_Vulkan, "Device fault info: query failed ({})",
                     static_cast<int>(result));
        return;
    }
    LOG_CRITICAL(Render_Vulkan, "Device fault: {} ({} addresses, {} vendor records)",
                 info.description, counts.addressInfoCount, counts.vendorInfoCount);
    for (u32 i = 0; i < counts.addressInfoCount; ++i) {
        const VkDeviceFaultAddressInfoEXT& address = addresses[i];
        const char* type = "unknown";
        switch (address.addressType) {
        case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT:
            type = "none";
            break;
        case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT:
            type = "invalid read";
            break;
        case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT:
            type = "invalid write";
            break;
        case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT:
            type = "invalid execute";
            break;
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT:
            type = "instruction pointer (unknown)";
            break;
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT:
            type = "instruction pointer (invalid)";
            break;
        case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT:
            type = "instruction pointer (fault)";
            break;
        default:
            break;
        }
        const VkDeviceSize precision = (std::max)(address.addressPrecision, VkDeviceSize{1});
        const VkDeviceSize lower = address.reportedAddress & ~(precision - 1);
        LOG_CRITICAL(Render_Vulkan, "Device fault address {}: {} at {:#x} (range [{:#x}, {:#x}))",
                     i, type, address.reportedAddress, lower, lower + precision);
        // FRITCHII_VA_REPORT=1: what held that page, and whether it was still bound.
        VaReport::LogOwners(address.reportedAddress, precision);
    }
    for (u32 i = 0; i < counts.vendorInfoCount; ++i) {
        const VkDeviceFaultVendorInfoEXT& vendor = vendor_infos[i];
        LOG_CRITICAL(Render_Vulkan, "Device fault vendor record {}: {} (code {:#x}, data {:#x})", i,
                     vendor.description, vendor.vendorFaultCode, vendor.vendorFaultData);
    }
}

void Device::SaveShader(std::span<const u32> spirv) const {
    if (nsight_aftermath_tracker) {
        nsight_aftermath_tracker->SaveShader(spirv);
    }
}

bool Device::ComputeIsOptimalAstcSupported() const {
    static constexpr std::array<VkFormat, 28> astc_formats = {
        VK_FORMAT_ASTC_4x4_UNORM_BLOCK,   VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
        VK_FORMAT_ASTC_5x4_UNORM_BLOCK,   VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
        VK_FORMAT_ASTC_5x5_UNORM_BLOCK,   VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_6x5_UNORM_BLOCK,   VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_6x6_UNORM_BLOCK,   VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x5_UNORM_BLOCK,   VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x6_UNORM_BLOCK,   VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_8x8_UNORM_BLOCK,   VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x5_UNORM_BLOCK,  VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x6_UNORM_BLOCK,  VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x8_UNORM_BLOCK,  VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
        VK_FORMAT_ASTC_10x10_UNORM_BLOCK, VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
        VK_FORMAT_ASTC_12x10_UNORM_BLOCK, VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
        VK_FORMAT_ASTC_12x12_UNORM_BLOCK, VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
    };
    if (!features.features.textureCompressionASTC_LDR) {
        return false;
    }
    const VkFormatFeatureFlags format_feature_usage{
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT};
    for (const auto format : astc_formats) {
        const auto physical_format_properties{physical.GetFormatProperties(format)};
        if ((physical_format_properties.optimalTilingFeatures & format_feature_usage) !=
            format_feature_usage) {
            return false;
        }
    }
    return true;
}

bool Device::TestDepthStencilBlits(VkFormat format) const {
    static constexpr VkFormatFeatureFlags required_features =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    const auto test_features = [](VkFormatProperties props) {
        return (props.optimalTilingFeatures & required_features) == required_features;
    };
    return test_features(format_properties.at(format));
}

bool Device::IsFormatSupported(VkFormat wanted_format, VkFormatFeatureFlags wanted_usage,
                               FormatType format_type) const {
    const auto it = format_properties.find(wanted_format);
    if (it == format_properties.end()) {
        UNIMPLEMENTED_MSG("Unimplemented format query={}", wanted_format);
        return true;
    }
    const auto supported_usage = GetFormatFeatures(it->second, format_type);
    return (supported_usage & wanted_usage) == wanted_usage;
}

std::string Device::GetDriverName() const {
    return vk::GetDriverName(properties.driver);
}

bool Device::ShouldBoostClocks() const {
    const auto driver_id = properties.driver.driverID;
    const auto vendor_id = properties.properties.vendorID;
    const auto device_id = properties.properties.deviceID;

    const bool validated_driver =
        driver_id == VK_DRIVER_ID_AMD_PROPRIETARY || driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
        driver_id == VK_DRIVER_ID_MESA_RADV || driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
        driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY || driver_id == VK_DRIVER_ID_MESA_TURNIP ||
        driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY;

    const bool is_steam_deck = (vendor_id == 0x1002 && device_id == 0x163F) ||
                               (vendor_id == 0x1002 && device_id == 0x1435);

    const bool is_debugging = this->HasDebuggingToolAttached();

    return validated_driver && !is_steam_deck && !is_debugging;
}

bool Device::HasTimelineSemaphore() const {
    if (GetDriverID() == VK_DRIVER_ID_MESA_TURNIP) {
        return false;
    }
    return features.timeline_semaphore.timelineSemaphore;
}

bool Device::MustEmulateBGR565() const {
    return Settings::values.emulate_bgr565.GetValue();
}

bool Device::GetSuitability(bool requires_swapchain) {
    // Assume we will be suitable.
    bool suitable = true;

    // Configure properties.
    VkPhysicalDeviceVulkan12Features features_1_2{};
    VkPhysicalDeviceVulkan13Features features_1_3{};

    // Configure properties.
    properties.properties = physical.GetProperties();

    // Set instance version.
    instance_version = properties.properties.apiVersion;

    // Minimum of API version 1.1 is required. (This is well-supported.)
    ASSERT(instance_version >= VK_API_VERSION_1_1);

    // Get available extensions.
    auto extension_properties = physical.EnumerateDeviceExtensionProperties();

    // Get the set of supported extensions.
    supported_extensions.clear();
    for (const VkExtensionProperties& property : extension_properties) {
        supported_extensions.insert(property.extensionName);
    }

    // Generate list of extensions to load.
    loaded_extensions.clear();

#define EXTENSION(prefix, macro_name, var_name)                                                    \
    if (supported_extensions.contains(VK_##prefix##_##macro_name##_EXTENSION_NAME)) {              \
            loaded_extensions.insert(VK_##prefix##_##macro_name##_EXTENSION_NAME);                     \
            extensions.var_name = true;                                                                \
    }
#define FEATURE_EXTENSION(prefix, struct_name, macro_name, var_name)                               \
    if (supported_extensions.contains(VK_##prefix##_##macro_name##_EXTENSION_NAME)) {              \
            loaded_extensions.insert(VK_##prefix##_##macro_name##_EXTENSION_NAME);                     \
            extensions.var_name = true;                                                                \
    }

    if (instance_version < VK_API_VERSION_1_2) {
        FOR_EACH_VK_FEATURE_1_2(FEATURE_EXTENSION);
    }
    if (instance_version < VK_API_VERSION_1_3) {
        FOR_EACH_VK_FEATURE_1_3(FEATURE_EXTENSION);
    }

    FOR_EACH_VK_FEATURE_EXT(FEATURE_EXTENSION);
    FOR_EACH_VK_EXTENSION(EXTENSION);

    // Graphics pipelines linked from per-stage libraries (vk_pipeline_library): the Shader
    // libraries setting turns them on for NVIDIA GPUs and AMD's Mesa driver (Intel and others are
    // untested), EDEN_NX2_GPL=1 for any GPU, EDEN_NX2_GPL=0 turns them off. Not on AMD's Windows
    // driver: RX 5000 and 6000 owners crash building library pipelines from their cache
    // (amdvlk64.dll; unticking Shader libraries is their workaround, 2026-11-09), and the
    // precompile preparation now does the first-time builds those libraries were for.
    graphics_pipeline_library_requested = [this] {
        if (const char* value = std::getenv("EDEN_NX2_GPL"); value && *value) {
            return *value != '0';
        }
        const u32 vendor = properties.properties.vendorID;
        // Not properties.driver: it is read further down (it was 0 here, and v5.2 left the
        // libraries on for these cards). On Windows AMD's Vulkan driver is the proprietary one.
#ifdef _WIN32
        constexpr bool amd_windows_driver = true;
#else
        constexpr bool amd_windows_driver = false;
#endif
        if (vendor == 0x1002 && amd_windows_driver &&
            Settings::values.use_graphics_pipeline_library.GetValue()) {
            LOG_INFO(Render_Vulkan, "Nx2Perf gpl: shader libraries off on AMD's proprietary "
                                    "driver (EDEN_NX2_GPL=1 turns them on)");
            return false;
        }
        return Settings::values.use_graphics_pipeline_library.GetValue() &&
               (vendor == 0x10DE || vendor == 0x1002);
    }();
    const bool gpl_offered =
        supported_extensions.contains(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) &&
        supported_extensions.contains(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    if (graphics_pipeline_library_requested && gpl_offered) {
        loaded_extensions.insert(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
        loaded_extensions.insert(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    } else {
        // Says whether the driver offers them at all (an Arc B580 log could not tell, helper
        // CZ); fast linking is read only once requested, then logged below.
        LOG_INFO(Render_Vulkan,
                 "Nx2Perf gpl: shader libraries {} (the driver {} VK_EXT_graphics_pipeline_library; "
                 "EDEN_NX2_GPL=1 requests them on any GPU)",
                 graphics_pipeline_library_requested ? "requested but unavailable"
                                                     : "not requested",
                 gpl_offered ? "offers" : "does not offer");
        graphics_pipeline_library_requested = false;
    }

    // VK_KHR_pipeline_binary (FRITCHII_PIPELINE_BINARY, vk_pipeline_binary.h): enabled only with
    // the switch on (on by default for AMD's Windows driver). Its properties are read and logged
    // whenever the driver offers it.
    PipelineBinary::ChooseDefault(properties.properties.vendorID);
    pipeline_binary_offered = supported_extensions.contains(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME);
    pipeline_binary_requested = pipeline_binary_offered && PipelineBinary::SwitchValue() != 0 &&
                                supported_extensions.contains(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    if (pipeline_binary_requested) {
        loaded_extensions.insert(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME);
    }

    extensions.depth_stencil_resolve =
        extensions.depth_stencil_resolve &&
        (instance_version >= VK_API_VERSION_1_2 || extensions.create_renderpass2);
    RemoveExtensionIfUnsuitable(extensions.depth_stencil_resolve,
                                VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);

    if (supported_extensions.contains(VK_KHR_ROBUSTNESS_2_EXTENSION_NAME)) {
        loaded_extensions.erase(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
        loaded_extensions.insert(VK_KHR_ROBUSTNESS_2_EXTENSION_NAME);
        extensions.robustness_2 = true;
    } else if (supported_extensions.contains(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) {
        loaded_extensions.insert(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
        extensions.robustness_2 = true;
    } else {
        extensions.robustness_2 = false;
    }

#ifdef EGLANTINE_STREAMLINE
    // The extensions NVIDIA's DLSS asks for, where the device has them all.
    if (Streamline::WantsDevice()) {
        const std::vector<std::string> wanted = Streamline::DeviceExtensions();
        // Streamline may ask for no extensions at all.
        const bool all = std::ranges::all_of(
            wanted, [&](const std::string& name) { return supported_extensions.contains(name); });
        std::string names;
        for (const std::string& name : wanted) {
            if (all) {
                loaded_extensions.insert(name);
            }
            names += (names.empty() ? "" : ", ") + name;
        }
        streamline_extensions = all;
        LOG_INFO(Render_Vulkan, "NVIDIA DLSS device extensions: {} ({})",
                 all ? "enabled" : "unavailable", names.empty() ? "none" : names);
    }
#endif

#undef FEATURE_EXTENSION
#undef EXTENSION

// Some extensions are mandatory. Check those.
#define CHECK_EXTENSION(extension_name)                                                            \
    if (!loaded_extensions.contains(extension_name)) {                                             \
            LOG_ERROR(Render_Vulkan, "Missing required extension {}", extension_name);                 \
            suitable = false;                                                                          \
    }

#define LOG_EXTENSION(extension_name)                                                              \
    if (!loaded_extensions.contains(extension_name)) {                                             \
            LOG_INFO(Render_Vulkan, "Device doesn't support extension {}", extension_name);            \
    }

    FOR_EACH_VK_RECOMMENDED_EXTENSION(LOG_EXTENSION);
    FOR_EACH_VK_MANDATORY_EXTENSION(CHECK_EXTENSION);

    if (requires_swapchain) {
        CHECK_EXTENSION(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    if (instance_version < VK_API_VERSION_1_2) {
        CHECK_EXTENSION(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    }

#undef LOG_EXTENSION
#undef CHECK_EXTENSION

    // Generate the linked list of features to test.
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    // Set next pointer.
    void** next = &features2.pNext;

    // Vulkan 1.2 and 1.3 features
    if (instance_version >= VK_API_VERSION_1_2) {
        features_1_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        features_1_3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        features_1_2.pNext = &features_1_3;

        *next = &features_1_2;
    }

// Test all features we know about. If the feature is not available in core at our
// current API version, and was not enabled by an extension, skip testing the feature.
// We set the structure sType explicitly here as it is zeroed by the constructor.
#define FEATURE(prefix, struct_name, macro_name, var_name)                                         \
    features.var_name.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_##macro_name##_FEATURES;           \
        SetNext(next, features.var_name);

#define EXT_FEATURE(prefix, struct_name, macro_name, var_name)                                     \
    if (extensions.var_name) {                                                                     \
            features.var_name.sType =                                                                  \
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_##macro_name##_FEATURES_##prefix;                    \
            SetNext(next, features.var_name);                                                          \
    }

    FOR_EACH_VK_FEATURE_1_1(FEATURE);
    FOR_EACH_VK_FEATURE_EXT(EXT_FEATURE);
    if (instance_version >= VK_API_VERSION_1_2) {
        FOR_EACH_VK_FEATURE_1_2(FEATURE);
    } else {
        FOR_EACH_VK_FEATURE_1_2(EXT_FEATURE);
    }
    if (instance_version >= VK_API_VERSION_1_3) {
        FOR_EACH_VK_FEATURE_1_3(FEATURE);
    } else {
        FOR_EACH_VK_FEATURE_1_3(EXT_FEATURE);
    }

#undef EXT_FEATURE
#undef FEATURE

    if (graphics_pipeline_library_requested) {
        graphics_pipeline_library_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
        SetNext(next, graphics_pipeline_library_features);
    }
    // In the chain the device is created with: only when the extension is enabled.
    if (pipeline_binary_requested) {
        pipeline_binary_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR;
        SetNext(next, pipeline_binary_features);
    }

    // VK_EXT_device_fault: what the driver knows about a device loss (ReportLoss). Only the
    // report reads it; FRITCHII_DEVICE_FAULT=0 (boot) leaves the extension off.
    const char* const device_fault_switch = std::getenv("FRITCHII_DEVICE_FAULT");
    const bool has_device_fault_extension =
        supported_extensions.contains(VK_EXT_DEVICE_FAULT_EXTENSION_NAME) &&
        (device_fault_switch == nullptr || std::atoi(device_fault_switch) != 0);
    if (has_device_fault_extension) {
        device_fault_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT;
        SetNext(next, device_fault_features);
    }
    // FRITCHII_VA_REPORT=1 (boot, test tool): the driver reports each GPU address range it binds
    // or unbinds, so a device loss can name what held the faulting page. It reports through
    // VK_EXT_debug_utils, which the instance enabled for it (VaReport::InstanceReady).
    const bool has_address_binding_report =
        VaReport::Requested() && VaReport::InstanceReady() &&
        supported_extensions.contains(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
    if (VaReport::Requested() && !has_address_binding_report) {
        LOG_WARNING(Render_Vulkan, "GPU address report (FRITCHII_VA_REPORT=1): off, the driver "
                                   "has no VK_EXT_device_address_binding_report or the instance "
                                   "no VK_EXT_debug_utils");
    }
    if (has_address_binding_report) {
        address_binding_report_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ADDRESS_BINDING_REPORT_FEATURES_EXT;
        SetNext(next, address_binding_report_features);
    }

    // Perform the feature test.
    physical.GetFeatures2(features2);

    if (has_address_binding_report) {
        // Its features struct is in the creation chain: the extension is enabled either way.
        loaded_extensions.insert(VK_EXT_DEVICE_ADDRESS_BINDING_REPORT_EXTENSION_NAME);
        if (address_binding_report_features.reportAddressBinding) {
            VaReport::NoteDeviceReady();
            LOG_INFO(Render_Vulkan, "GPU address report (FRITCHII_VA_REPORT=1): the driver reports "
                                    "its GPU address binds");
        } else {
            LOG_WARNING(Render_Vulkan, "GPU address report (FRITCHII_VA_REPORT=1): off, the "
                                       "driver lists the extension without reportAddressBinding");
        }
    }
    if (has_device_fault_extension) {
        loaded_extensions.insert(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        // No vendor binary dumps: only the description, addresses and vendor codes are logged.
        device_fault_features.deviceFaultVendorBinary = VK_FALSE;
    }

    // Base Vulkan 1.0 features are always valid regardless of instance version.
    features.features = features2.features;

// Some features are mandatory. Check those.
#define CHECK_FEATURE(feature, name)                                                               \
    if (!features.feature.name) {                                                                  \
        if (IsMoltenVK() && (strcmp(#name, "geometryShader") == 0 ||                               \
                            strcmp(#name, "logicOp") == 0 ||                                       \
                            strcmp(#name, "shaderCullDistance") == 0 ||                            \
                            strcmp(#name, "wideLines") == 0)) {                                    \
            LOG_INFO(Render_Vulkan, "MoltenVK missing feature {} - using fallback", #name);       \
        } else {                                                                                    \
            LOG_ERROR(Render_Vulkan, "Missing required feature {}", #name);                        \
            suitable = false;                                                                       \
        }                                                                                           \
    }

#define LOG_FEATURE(feature, name)                                                                 \
    if (!features.feature.name) {                                                                  \
            LOG_INFO(Render_Vulkan, "Device doesn't support feature {}", #name);                       \
    }

// Optional features are enabled silently without any logging
#define OPTIONAL_FEATURE(feature, name) (void)features.feature.name;

    FOR_EACH_VK_OPTIONAL_FEATURE(OPTIONAL_FEATURE);
    FOR_EACH_VK_RECOMMENDED_FEATURE(LOG_FEATURE);
    FOR_EACH_VK_MANDATORY_FEATURE(CHECK_FEATURE);

#undef OPTIONAL_FEATURE
#undef LOG_FEATURE
#undef CHECK_FEATURE

    // Generate linked list of properties.
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

    // Set next pointer.
    next = &properties2.pNext;

    // Get driver info.
    properties.driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    SetNext(next, properties.driver);

    // Retrieve subgroup properties.
    properties.subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    SetNext(next, properties.subgroup_properties);

    // Retrieve relevant extension properties.
    if (extensions.shader_float_controls) {
        properties.float_controls.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
        SetNext(next, properties.float_controls);
    }
    if (extensions.push_descriptor) {
        properties.push_descriptor.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR;
        SetNext(next, properties.push_descriptor);
    }
    if (extensions.depth_stencil_resolve || instance_version >= VK_API_VERSION_1_2) {
        properties.depth_stencil_resolve.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES;
        SetNext(next, properties.depth_stencil_resolve);
    }
    if (extensions.descriptor_buffer) {
        properties.descriptor_buffer.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;
        SetNext(next, properties.descriptor_buffer);
    }
    if (extensions.subgroup_size_control || features.subgroup_size_control.subgroupSizeControl) {
        properties.subgroup_size_control.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
        SetNext(next, properties.subgroup_size_control);
    }
    if (extensions.transform_feedback) {
        properties.transform_feedback.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT;
        SetNext(next, properties.transform_feedback);
    }
    if (extensions.maintenance5) {
        properties.maintenance5.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES_KHR;
        SetNext(next, properties.maintenance5);
    }
    if (extensions.custom_border_color) {
        properties.custom_border_color.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_PROPERTIES_EXT;
        SetNext(next, properties.custom_border_color);
    }
    if (extensions.cooperative_matrix) {
        properties.cooperative_matrix.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        SetNext(next, properties.cooperative_matrix);
    }
    if (supported_extensions.contains(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
        properties.integer_dot_product.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
        SetNext(next, properties.integer_dot_product);
    }

    if (graphics_pipeline_library_requested) {
        properties.graphics_pipeline_library.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT;
        SetNext(next, properties.graphics_pipeline_library);
    }
    if (pipeline_binary_offered) {
        pipeline_binary_properties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_PROPERTIES_KHR;
        SetNext(next, pipeline_binary_properties);
    }

    // Perform the property fetch.
    physical.GetProperties2(properties2);

    // Store base properties
    properties.properties = properties2.properties;

    // Unload extensions if feature support is insufficient.
    RemoveUnsuitableExtensions();

    // Check limits.
    struct Limit {
        u32 minimum;
        u32 value;
        const char* name;
    };

    const VkPhysicalDeviceLimits& limits{properties.properties.limits};
    const std::array limits_report{
                                   Limit{65536, limits.maxUniformBufferRange, "maxUniformBufferRange"},
                                   Limit{16, limits.maxViewports, "maxViewports"},
                                   Limit{8, limits.maxColorAttachments, "maxColorAttachments"},
                                   Limit{8, limits.maxClipDistances, "maxClipDistances"},
                                   };

    for (const auto& [min, value, name] : limits_report) {
        if (value < min) {
            LOG_ERROR(Render_Vulkan, "{} has to be {} or greater but it is {}", name, min, value);
            suitable = false;
        }
    }

    // VK_DYNAMIC_STATE

    // Driver detection variables for workarounds in GetSuitability
    const VkDriverId driver_id = properties.driver.driverID;

    // VK_EXT_extended_dynamic_state2 below this will appear drivers that need workarounds.

    // VK_EXT_extended_dynamic_state3 below this will appear drivers that need workarounds.

    // Samsung: Broken extendedDynamicState3ColorBlendEquation
    // Disable blend equation dynamic state, force static pipeline state
    if (extensions.extended_dynamic_state3 &&
        (driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY)) {
        LOG_WARNING(Render_Vulkan,
                    "Samsung: Disabling broken extendedDynamicState3ColorBlendEquation");
        features.extended_dynamic_state3.extendedDynamicState3ColorBlendEnable = false;
        features.extended_dynamic_state3.extendedDynamicState3ColorBlendEquation = false;
    }

    // Intel Windows < 27.20.100.0: Broken VertexInputDynamicState
    // Same for NVIDIA Proprietary < 580.119.02, unknown when VIDS was first NOT broken
    // Disable VertexInputDynamicState on old Intel Windows drivers
    if (extensions.vertex_input_dynamic_state) {
        // Intel's Windows driver reports its build as (major << 14) | minor ("101.9150" is
        // 1663934), not as a Vulkan API version: compared with VK_MAKE_API_VERSION(27, 20, 100, 0)
        // every Intel driver was "older" and lost the extension. That left the precompiled
        // pipelines (all recorded with dynamic vertex input) unusable there and keyed every
        // vertex layout into its own pipeline. The broken drivers are the 100.x branch (Gen9-era
        // 27.20.100.x); Arc and Xe drivers are 101.x.
        const u32 intel_build_major = properties.properties.driverVersion >> 14;
        if ((driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS && intel_build_major < 101)
        || (driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY && properties.properties.driverVersion < NvidiaDriverVersion(580, 119, 2))) {
            LOG_WARNING(Render_Vulkan, "Disabling broken VK_EXT_vertex_input_dynamic_state");
            RemoveExtensionFeature(extensions.vertex_input_dynamic_state, features.vertex_input_dynamic_state, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
        }
    }

    if (u32(Settings::values.dyna_state.GetValue()) == 0) {
        LOG_INFO(Render_Vulkan, "Extended Dynamic State disabled by user setting, clearing all EDS features");
        features.extended_dynamic_state.extendedDynamicState = false;
        features.extended_dynamic_state2.extendedDynamicState2 = false;
        features.extended_dynamic_state3.extendedDynamicState3ColorBlendEnable = false;
        features.extended_dynamic_state3.extendedDynamicState3ColorBlendEquation = false;
        features.extended_dynamic_state3.extendedDynamicState3ColorWriteMask = false;
        features.extended_dynamic_state3.extendedDynamicState3DepthClampEnable = false;
        features.extended_dynamic_state3.extendedDynamicState3LogicOpEnable = false;
    }

    // Return whether we were suitable.
    return suitable;
}

void Device::RemoveUnsuitableExtensions() {
    // VK_KHR_present_id, VK_KHR_present_wait (the host frame pacer, vk_host_pacer.h)
    extensions.present_id = extensions.present_id && features.present_id.presentId;
    extensions.present_wait =
        extensions.present_id && extensions.present_wait && features.present_wait.presentWait;
    RemoveExtensionFeatureIfUnsuitable(extensions.present_wait, features.present_wait,
                                       VK_KHR_PRESENT_WAIT_EXTENSION_NAME);
    RemoveExtensionFeatureIfUnsuitable(extensions.present_id, features.present_id,
                                       VK_KHR_PRESENT_ID_EXTENSION_NAME);

    // VK_EXT_color_write_enable
    extensions.color_write_enable = features.color_write_enable.colorWriteEnable;
    RemoveExtensionFeatureIfUnsuitable(extensions.color_write_enable, features.color_write_enable,
                                       VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME);

    // VK_EXT_custom_border_color
    if (extensions.custom_border_color) {
        extensions.custom_border_color =
            features.custom_border_color.customBorderColors &&
            features.custom_border_color.customBorderColorWithoutFormat;
    }
    RemoveExtensionFeatureIfUnsuitable(extensions.custom_border_color, features.custom_border_color,
                                       VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);

    // VK_EXT_border_color_swizzle
    if (extensions.border_color_swizzle) {
        extensions.border_color_swizzle =
            extensions.custom_border_color && features.border_color_swizzle.borderColorSwizzle;
    }
    RemoveExtensionFeatureIfUnsuitable(extensions.border_color_swizzle,
                                       features.border_color_swizzle,
                                       VK_EXT_BORDER_COLOR_SWIZZLE_EXTENSION_NAME);

    // VK_EXT_depth_bias_control
    extensions.depth_bias_control =
        features.depth_bias_control.depthBiasControl &&
        features.depth_bias_control.leastRepresentableValueForceUnormRepresentation;
    RemoveExtensionFeatureIfUnsuitable(extensions.depth_bias_control, features.depth_bias_control,
                                       VK_EXT_DEPTH_BIAS_CONTROL_EXTENSION_NAME);

    // VK_EXT_depth_clip_control
    extensions.depth_clip_control = features.depth_clip_control.depthClipControl;
    RemoveExtensionFeatureIfUnsuitable(extensions.depth_clip_control, features.depth_clip_control,
                                       VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME);

    // VK_EXT_descriptor_buffer
    extensions.descriptor_buffer = features.descriptor_buffer.descriptorBuffer;
    RemoveExtensionFeatureIfUnsuitable(extensions.descriptor_buffer, features.descriptor_buffer,
                                       VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);

    // VK_EXT_extended_dynamic_state
    extensions.extended_dynamic_state = features.extended_dynamic_state.extendedDynamicState;
    RemoveExtensionFeatureIfUnsuitable(extensions.extended_dynamic_state,
                                       features.extended_dynamic_state,
                                       VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);

    // VK_EXT_extended_dynamic_state2
    extensions.extended_dynamic_state2 = features.extended_dynamic_state2.extendedDynamicState2;
    RemoveExtensionFeatureIfUnsuitable(extensions.extended_dynamic_state2,
                                       features.extended_dynamic_state2,
                                       VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME);

    // VK_EXT_extended_dynamic_state3
    const bool supports_color_blend_enable =
        features.extended_dynamic_state3.extendedDynamicState3ColorBlendEnable;
    const bool supports_color_blend_equation =
        features.extended_dynamic_state3.extendedDynamicState3ColorBlendEquation;
    const bool supports_color_write_mask =
        features.extended_dynamic_state3.extendedDynamicState3ColorWriteMask;
    dynamic_state3_blending = supports_color_blend_enable && supports_color_blend_equation &&
                              supports_color_write_mask;

    const bool supports_depth_clamp_enable =
        features.extended_dynamic_state3.extendedDynamicState3DepthClampEnable;
    const bool supports_logic_op_enable =
        features.extended_dynamic_state3.extendedDynamicState3LogicOpEnable;
    const bool supports_line_raster_mode =
        features.extended_dynamic_state3.extendedDynamicState3LineRasterizationMode &&
        extensions.line_rasterization && features.line_rasterization.rectangularLines;
    const bool supports_conservative_raster_mode =
        features.extended_dynamic_state3.extendedDynamicState3ConservativeRasterizationMode &&
        extensions.conservative_rasterization;
    const bool supports_line_stipple_enable =
        features.extended_dynamic_state3.extendedDynamicState3LineStippleEnable &&
        extensions.line_rasterization && features.line_rasterization.stippledRectangularLines;
    const bool supports_alpha_to_coverage =
        features.extended_dynamic_state3.extendedDynamicState3AlphaToCoverageEnable;
    const bool supports_alpha_to_one =
        features.extended_dynamic_state3.extendedDynamicState3AlphaToOneEnable &&
        features.features.alphaToOne;

    dynamic_state3_depth_clamp_enable = supports_depth_clamp_enable;
    dynamic_state3_logic_op_enable = supports_logic_op_enable;
    dynamic_state3_line_raster_mode = supports_line_raster_mode;
    dynamic_state3_conservative_raster_mode = supports_conservative_raster_mode;
    dynamic_state3_line_stipple_enable = supports_line_stipple_enable;
    dynamic_state3_alpha_to_coverage = supports_alpha_to_coverage;
    dynamic_state3_alpha_to_one = supports_alpha_to_one;

    dynamic_state3_enables = dynamic_state3_depth_clamp_enable || dynamic_state3_logic_op_enable ||
                             dynamic_state3_line_raster_mode ||
                             dynamic_state3_conservative_raster_mode ||
                             dynamic_state3_line_stipple_enable ||
                             dynamic_state3_alpha_to_coverage || dynamic_state3_alpha_to_one;

    extensions.extended_dynamic_state3 = dynamic_state3_blending || dynamic_state3_enables;
    if (!extensions.extended_dynamic_state3) {
        dynamic_state3_blending = false;
        dynamic_state3_enables = false;
        dynamic_state3_depth_clamp_enable = false;
        dynamic_state3_logic_op_enable = false;
        dynamic_state3_line_raster_mode = false;
        dynamic_state3_conservative_raster_mode = false;
        dynamic_state3_line_stipple_enable = false;
        dynamic_state3_alpha_to_coverage = false;
        dynamic_state3_alpha_to_one = false;
    }
    RemoveExtensionFeatureIfUnsuitable(extensions.extended_dynamic_state3,
                                       features.extended_dynamic_state3,
                                       VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);

    // VK_EXT_robustness2
    // AMD keeps robustBufferAccess2 (a ported AMD fix):
    // games read storage buffers out of bounds and AMD returned garbage
    // without it (OoT's fog compute shader 75d4369f3b1f5868 filled distant froxels with huge
    // values and turned the Hyrule Field sky white). It needs robustBufferAccess, which every
    // device here enables (a required feature). Only where the driver offers it: the queried
    // value is kept. AMD proprietary, AMDVLK and RADV only; every other driver turns it off as
    // before.
    if (!IsAmdDriver()) {
        features.robustness2.robustBufferAccess2 = VK_FALSE;
    }
    features.robustness2.robustImageAccess2 = VK_FALSE;
    extensions.robustness_2 = features.robustness2.nullDescriptor;

    const char* robustness2_extension_name =
        loaded_extensions.contains(VK_KHR_ROBUSTNESS_2_EXTENSION_NAME)
            ? VK_KHR_ROBUSTNESS_2_EXTENSION_NAME
            : VK_EXT_ROBUSTNESS_2_EXTENSION_NAME;

    RemoveExtensionFeatureIfUnsuitable(extensions.robustness_2, features.robustness2,
                                       robustness2_extension_name);

    // Image robustness
    extensions.robust_image_access = features.robust_image_access.robustImageAccess;
    RemoveExtensionFeatureIfUnsuitable(extensions.robust_image_access,
                                       features.robust_image_access,
                                       VK_EXT_IMAGE_ROBUSTNESS_EXTENSION_NAME);

    // VK_KHR_shader_atomic_int64
    extensions.shader_atomic_int64 = features.shader_atomic_int64.shaderBufferInt64Atomics &&
                                     features.shader_atomic_int64.shaderSharedInt64Atomics;
    RemoveExtensionFeatureIfUnsuitable(extensions.shader_atomic_int64, features.shader_atomic_int64,
                                       VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);

    // VK_EXT_shader_atomic_float: storage buffer float32 atomic add in hardware instead of a
    // compare-exchange loop. Intel only for now (the GPU it was measured on): on an Arc B580 a
    // contended loop of Ocarina of Time (1e1964184a291f00, 96 groups) took 2 ms of each frame.
    // EDEN_NX2_NO_FLOAT_ATOMICS=1 keeps the loop.
    {
        constexpr u32 intel_vendor_id = 0x8086;
        const char* const no_float_atomics = std::getenv("EDEN_NX2_NO_FLOAT_ATOMICS");
        extensions.shader_atomic_float =
            extensions.shader_atomic_float &&
            properties.properties.vendorID == intel_vendor_id &&
            features.shader_atomic_float.shaderBufferFloat32AtomicAdd &&
            !(no_float_atomics != nullptr && no_float_atomics[0] == '1');
        RemoveExtensionFeatureIfUnsuitable(extensions.shader_atomic_float,
                                           features.shader_atomic_float,
                                           VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }

    // VK_EXT_shader_demote_to_helper_invocation
    extensions.shader_demote_to_helper_invocation =
        features.shader_demote_to_helper_invocation.shaderDemoteToHelperInvocation;
    RemoveExtensionFeatureIfUnsuitable(extensions.shader_demote_to_helper_invocation,
                                       features.shader_demote_to_helper_invocation,
                                       VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME);

    // VK_EXT_subgroup_size_control
    extensions.subgroup_size_control =
        features.subgroup_size_control.subgroupSizeControl &&
        properties.subgroup_size_control.minSubgroupSize <= GuestWarpSize &&
        properties.subgroup_size_control.maxSubgroupSize >= GuestWarpSize;
    RemoveExtensionFeatureIfUnsuitable(extensions.subgroup_size_control,
                                       features.subgroup_size_control,
                                       VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    // VK_EXT_transform_feedback
    extensions.transform_feedback =
        features.transform_feedback.transformFeedback &&
        properties.transform_feedback.maxTransformFeedbackBuffers > 0;
    RemoveExtensionFeatureIfUnsuitable(extensions.transform_feedback, features.transform_feedback,
                                       VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME);

    // VK_EXT_vertex_input_dynamic_state
    extensions.vertex_input_dynamic_state =
        features.vertex_input_dynamic_state.vertexInputDynamicState;
    RemoveExtensionFeatureIfUnsuitable(extensions.vertex_input_dynamic_state,
                                       features.vertex_input_dynamic_state,
                                       VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);

    // VK_KHR_pipeline_executable_properties
    if (Settings::values.renderer_shader_feedback.GetValue()) {
        extensions.pipeline_executable_properties =
            features.pipeline_executable_properties.pipelineExecutableInfo;
        RemoveExtensionFeatureIfUnsuitable(extensions.pipeline_executable_properties,
                                           features.pipeline_executable_properties,
                                           VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    } else {
        RemoveExtensionFeature(extensions.pipeline_executable_properties,
                               features.pipeline_executable_properties,
                               VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
    }

    // VK_EXT_mesh_shader: Switch 2 titles draw some geometry with task and mesh programs. Only
    // the two stages are used; the multiview, shading rate and query features need more.
    features.mesh_shader.multiviewMeshShader = VK_FALSE;
    features.mesh_shader.primitiveFragmentShadingRateMeshShader = VK_FALSE;
    features.mesh_shader.meshShaderQueries = VK_FALSE;
    extensions.mesh_shader = features.mesh_shader.taskShader && features.mesh_shader.meshShader;
    // EDEN_NX2_NO_DYNAMIC_LOGIC_OP=1: no dynamic logic op, as AMD's Windows driver (tests the
    // conversion of NVIDIA-recorded precompiled pipelines on an NVIDIA GPU).
    if (features.extended_dynamic_state2.extendedDynamicState2LogicOp &&
        std::getenv("EDEN_NX2_NO_DYNAMIC_LOGIC_OP") != nullptr) {
        LOG_WARNING(Render_Vulkan, "EDEN_NX2_NO_DYNAMIC_LOGIC_OP: dynamic logic op disabled");
        features.extended_dynamic_state2.extendedDynamicState2LogicOp = VK_FALSE;
    }
    // EDEN_NX2_FORCE_MESH_FALLBACK=1: leave VK_EXT_mesh_shader disabled, so task/mesh draws take
    // the compute fallback a device without it uses.
    if (extensions.mesh_shader && std::getenv("EDEN_NX2_FORCE_MESH_FALLBACK") != nullptr) {
        LOG_WARNING(Render_Vulkan, "EDEN_NX2_FORCE_MESH_FALLBACK: VK_EXT_mesh_shader disabled");
        extensions.mesh_shader = false;
    }
    // The Mesh Shaders setting. Automatic leaves the extension disabled on AMD drivers: native
    // task/mesh draws hang AMD GPUs. RADV resets the GPU on RX 6000, 7000
    // and 9000 cards (Mesa 26.0 to 27.0-devel; 26.2.6 still hangs, though it recovers more often),
    // and AMD's own driver times out on the Radeon 780M. The compute fallback avoids the hangs, but
    // it can be slower in scenes full of foliage and skips the mesh programs it can't run
    // (per-primitive outputs, primitives other than triangles, layered output). Native keeps the
    // extension, for a setup that no longer hangs; Compute disables it on every device.
    const Settings::MeshShaderMode mesh_mode = Settings::values.mesh_shaders.GetValue();
    const VkDriverId mesh_driver = properties.driver.driverID;
    const bool amd_mesh_driver = mesh_driver == VK_DRIVER_ID_MESA_RADV ||
                                 mesh_driver == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                 mesh_driver == VK_DRIVER_ID_AMD_OPEN_SOURCE;
    if (extensions.mesh_shader && mesh_mode == Settings::MeshShaderMode::Compute) {
        LOG_INFO(Render_Vulkan, "Mesh Shaders setting: task/mesh draws use the compute fallback");
        extensions.mesh_shader = false;
    } else if (extensions.mesh_shader && mesh_mode == Settings::MeshShaderMode::Automatic &&
               amd_mesh_driver) {
        LOG_INFO(Render_Vulkan, "AMD driver: task/mesh draws use the compute fallback, because "
                                "native mesh draws hang the GPU (Mesh Shaders: Native keeps them)");
        extensions.mesh_shader = false;
    }
    RemoveExtensionFeatureIfUnsuitable(extensions.mesh_shader, features.mesh_shader,
                                       VK_EXT_MESH_SHADER_EXTENSION_NAME);

    // VK_EXT_graphics_pipeline_library (EDEN_NX2_GPL): used only with fast linking, which makes
    // a link cost a fraction of a millisecond.
    if (graphics_pipeline_library_requested) {
        const bool usable =
            graphics_pipeline_library_features.graphicsPipelineLibrary &&
            properties.graphics_pipeline_library.graphicsPipelineLibraryFastLinking;
        if (!usable) {
            LOG_WARNING(Render_Vulkan,
                        "Nx2Perf gpl: the device has no graphics pipeline library fast linking, "
                        "shader libraries off");
            loaded_extensions.erase(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
            loaded_extensions.erase(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
            graphics_pipeline_library_features.graphicsPipelineLibrary = VK_FALSE;
        }
        graphics_pipeline_library_enabled = usable;
        if (usable) {
            LOG_INFO(Render_Vulkan,
                     "Nx2Perf gpl: graphics pipeline libraries on (fast linking, independent "
                     "interpolation decoration {})",
                     properties.graphics_pipeline_library
                         .graphicsPipelineLibraryIndependentInterpolationDecoration);
        }
    }

    // VK_KHR_shader_quad_control
    // Quad control depends on maximal reconvergence; advertise neither unless
    // the dependency is enabled on the logical device too.
    extensions.shader_maximal_reconvergence =
        features.shader_maximal_reconvergence.shaderMaximalReconvergence;
    RemoveExtensionFeatureIfUnsuitable(extensions.shader_maximal_reconvergence,
                                       features.shader_maximal_reconvergence,
                                       VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME);
    extensions.shader_quad_control = features.shader_quad_control.shaderQuadControl &&
                                     extensions.shader_maximal_reconvergence;
    RemoveExtensionFeatureIfUnsuitable(extensions.shader_quad_control, features.shader_quad_control,
                                       VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME);

    // VK_KHR_cooperative_matrix
    // Used only for the guest's HMMA.16816.F16, as a subgroup operation of compute shaders whose
    // subgroups are the guest's 32-lane warp. The matrices hold 16-bit floats, so the device needs
    // shaderFloat16. Shaders with them use the Vulkan memory model and run with full subgroups.
    // The guest's own M=16 N=8 K=16 shape is preferred; without it, a 16x16x16 shape runs HMMA
    // through shared memory with the compute subgroups forced to 32 lanes.
    {
        // AMD's driver also advertises the extension on GPUs without matrix cores (GCN, RDNA1,
        // RDNA2) and crashes in amdvlk64.dll building these pipelines on RDNA2. RDNA3 brought
        // matrix cores together with accelerated mixed-signedness dot4, which the driver reports.
        const VkDriverId driver_id = properties.driver.driverID;
        const bool emulated_amd_matrices =
            (driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
             driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE) &&
            !properties.integer_dot_product.integerDotProduct4x8BitPackedMixedSignednessAccelerated;
        if (extensions.cooperative_matrix && emulated_amd_matrices) {
            LOG_INFO(Render_Vulkan,
                     "VK_KHR_cooperative_matrix is emulated on this AMD GPU: guest tensor "
                     "operations stay in shader emulation");
        }
        const bool no_matrix_units = IsAmdWithoutMatrixUnits(properties.properties.vendorID,
                                                             properties.properties.deviceID);
        if (no_matrix_units && extensions.cooperative_matrix) {
            LOG_INFO(Render_Vulkan, "Cooperative matrices off: AMD GPU without matrix units");
        }
        // EDEN_NX2_NO_COOP_MATRIX=1 keeps HMMA on the software path, for A/B runs and for drivers
        // the checks above miss (a reported amdvlk64.dll crash on an RX 6900 XT).
        const char* const no_coop_matrix = std::getenv("EDEN_NX2_NO_COOP_MATRIX");
        const bool forced_off = no_coop_matrix != nullptr && *no_coop_matrix != '\0' &&
                                std::string_view{no_coop_matrix} != "0";
        if (forced_off && extensions.cooperative_matrix) {
            LOG_INFO(Render_Vulkan, "Cooperative matrices off: EDEN_NX2_NO_COOP_MATRIX is set");
        }
        const bool usable = extensions.cooperative_matrix && !emulated_amd_matrices &&
                            !no_matrix_units && !forced_off &&
                            features.cooperative_matrix.cooperativeMatrix &&
                            (properties.cooperative_matrix.cooperativeMatrixSupportedStages &
                             VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                            features.shader_float16_int8.shaderFloat16 &&
                            features.vulkan_memory_model.vulkanMemoryModel &&
                            features.vulkan_memory_model.vulkanMemoryModelDeviceScope &&
                            extensions.subgroup_size_control &&
                            features.subgroup_size_control.computeFullSubgroups;
        // Gated-off devices are not asked for their shapes.
        const auto shapes = usable ? physical.GetCooperativeMatrixPropertiesKHR()
                                   : std::vector<VkCooperativeMatrixPropertiesKHR>{};
        const auto has_shape = [&](u32 rows, u32 columns) {
            return std::ranges::any_of(shapes, [=](const VkCooperativeMatrixPropertiesKHR& shape) {
                return shape.MSize == rows && shape.NSize == columns && shape.KSize == 16 &&
                       shape.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                       shape.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                       shape.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                       shape.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                       shape.saturatingAccumulation == VK_FALSE &&
                       shape.scope == VK_SCOPE_SUBGROUP_KHR;
            });
        };
        cooperative_hmma_16x8x16 =
            usable && has_shape(16, 8) &&
            (properties.subgroup_properties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
            properties.subgroup_properties.subgroupSize == GuestWarpSize &&
            // Compute subgroups may vary between these sizes (always with SPIR-V 1.6), and the
            // per-lane matrix layout needs exactly the guest's 32 lanes.
            properties.subgroup_size_control.minSubgroupSize == GuestWarpSize &&
            properties.subgroup_size_control.maxSubgroupSize == GuestWarpSize;
        cooperative_hmma_16x16x16 =
            usable && has_shape(16, 16) && IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT);
        // Intel Xe2 (ANV) only has 8x16x16: each HMMA is two products of 8 rows. Intel only, so
        // NVIDIA and AMD keep exactly the paths above (ported, measured on Mesa
        // ANV). Intel's Windows driver too: its "broken float16 math" workaround only turns off
        // float16 arithmetic in translated shaders (float16_math); the device keeps shaderFloat16
        // for the matrices (see where float16_math is set). Arc B580, driver 101.9150: the
        // software HMMA of OoT's 64x36 tile pass spilled 12 KB of scratch per lane at SIMD32 and
        // took 13.5-15 ms a frame; on 8x16x16 matrices the same dispatch took 1.3 ms.
        // EDEN_NX2_NO_COOP_MATRIX=1 goes back to the software path.
        constexpr u32 intel_vendor_id = 0x8086;
        cooperative_hmma_8x16x16 =
            usable && properties.properties.vendorID == intel_vendor_id && has_shape(8, 16) &&
            IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT);
        extensions.cooperative_matrix =
            cooperative_hmma_16x8x16 || cooperative_hmma_16x16x16 || cooperative_hmma_8x16x16;
        features.cooperative_matrix.cooperativeMatrixRobustBufferAccess = VK_FALSE;
        RemoveExtensionFeatureIfUnsuitable(extensions.cooperative_matrix,
                                           features.cooperative_matrix,
                                           VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    // Picks the movie upscaler (Automatic: the game's own network where tensor operations are
    // native). Not on Intel's 8x16x16 path: OoT's movie network (its 32-lane compute programs,
    // f9ea14a22394f5b1 and others) drew a pink line and speckled columns over the title movie on
    // an Arc B580, the bilinear copy does not. The gameplay HMMA programs keep the matrices.
    VideoCommon::Nx2Perf::g_native_tensor_ops.store(
        extensions.cooperative_matrix && !cooperative_hmma_8x16x16, std::memory_order_relaxed);

    // VK_KHR_workgroup_memory_explicit_layout
    extensions.workgroup_memory_explicit_layout =
        features.workgroup_memory_explicit_layout.workgroupMemoryExplicitLayout &&
        features.workgroup_memory_explicit_layout.workgroupMemoryExplicitLayoutScalarBlockLayout;
    RemoveExtensionFeatureIfUnsuitable(extensions.workgroup_memory_explicit_layout,
                                       features.workgroup_memory_explicit_layout,
                                       VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);

    // VK_KHR_maintenance1
    extensions.maintenance1 = loaded_extensions.contains(VK_KHR_MAINTENANCE_1_EXTENSION_NAME);
    RemoveExtensionIfUnsuitable(extensions.maintenance1, VK_KHR_MAINTENANCE_1_EXTENSION_NAME);

    // VK_KHR_maintenance2
    extensions.maintenance2 = loaded_extensions.contains(VK_KHR_MAINTENANCE_2_EXTENSION_NAME);
    RemoveExtensionIfUnsuitable(extensions.maintenance2, VK_KHR_MAINTENANCE_2_EXTENSION_NAME);

    // VK_KHR_maintenance3
    extensions.maintenance3 = loaded_extensions.contains(VK_KHR_MAINTENANCE_3_EXTENSION_NAME);
    RemoveExtensionIfUnsuitable(extensions.maintenance3, VK_KHR_MAINTENANCE_3_EXTENSION_NAME);

    // VK_KHR_maintenance4
    extensions.maintenance4 = features.maintenance4.maintenance4;
    RemoveExtensionFeatureIfUnsuitable(extensions.maintenance4, features.maintenance4,
                                       VK_KHR_MAINTENANCE_4_EXTENSION_NAME);

    // VK_KHR_maintenance5
    extensions.maintenance5 = features.maintenance5.maintenance5;
    RemoveExtensionFeatureIfUnsuitable(extensions.maintenance5, features.maintenance5,
                                       VK_KHR_MAINTENANCE_5_EXTENSION_NAME);

    // VK_KHR_pipeline_binary (FRITCHII_PIPELINE_BINARY): needs maintenance5 (the capture flag
    // is a VkPipelineCreateFlags2 bit).
    if (pipeline_binary_offered) {
        const auto& binary = pipeline_binary_properties;
        LOG_INFO(Render_Vulkan,
                 "VK_KHR_pipeline_binary: internal cache {}, internal cache control {}, prefers "
                 "internal cache {}, precompiled internal cache {}, compressed data {}",
                 binary.pipelineBinaryInternalCache, binary.pipelineBinaryInternalCacheControl,
                 binary.pipelineBinaryPrefersInternalCache,
                 binary.pipelineBinaryPrecompiledInternalCache,
                 binary.pipelineBinaryCompressedData);
    }
    if (pipeline_binary_requested) {
        const bool usable = pipeline_binary_features.pipelineBinaries && extensions.maintenance5;
        if (!usable) {
            LOG_WARNING(Render_Vulkan, "FRITCHII_PIPELINE_BINARY: the driver has no usable "
                                       "pipeline binaries (feature or maintenance5 missing), off");
            loaded_extensions.erase(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME);
            pipeline_binary_features.pipelineBinaries = VK_FALSE;
        } else if (pipeline_binary_properties.pipelineBinaryPrefersInternalCache) {
            LOG_INFO(Render_Vulkan, "FRITCHII_PIPELINE_BINARY: the driver prefers its own cache; "
                                    "the store runs anyway");
        }
        pipeline_binary_enabled = usable;
    }

    // VK_EXT_shader_float8, only for AMD FSR 4: its RDNA 4 network runs in FP8 and drew a black
    // picture without the feature on the device (found by CheezerWheezer).
    if (extensions.shader_float8 &&
        (!features.shader_float8.shaderFloat8 || !WantsAmdFsr4(properties.driver.driverID))) {
        RemoveExtensionFeature(extensions.shader_float8, features.shader_float8,
                               VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    }

    // VK_KHR_maintenance6
    extensions.maintenance6 = features.maintenance6.maintenance6;
    RemoveExtensionFeatureIfUnsuitable(extensions.maintenance6, features.maintenance6,
                                       VK_KHR_MAINTENANCE_6_EXTENSION_NAME);

    // VK_KHR_maintenance7
    extensions.maintenance7 = loaded_extensions.contains(VK_KHR_MAINTENANCE_7_EXTENSION_NAME);
    RemoveExtensionIfUnsuitable(extensions.maintenance7, VK_KHR_MAINTENANCE_7_EXTENSION_NAME);

    // VK_KHR_maintenance8
    extensions.maintenance8 = loaded_extensions.contains(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    RemoveExtensionIfUnsuitable(extensions.maintenance8, VK_KHR_MAINTENANCE_8_EXTENSION_NAME);

    // VK_KHR_synchronization2
    extensions.synchronization2 = features.synchronization2.synchronization2;
    RemoveExtensionFeatureIfUnsuitable(extensions.synchronization2, features.synchronization2,
                                       VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
}

void Device::SetupFamilies(VkSurfaceKHR surface) {
    const std::vector queue_family_properties = physical.GetQueueFamilyProperties();
    std::optional<u32> graphics;
    std::optional<u32> present;
    for (u32 index = 0; index < static_cast<u32>(queue_family_properties.size()); ++index) {
        if (graphics && (present || !surface)) {
            break;
        }
        const VkQueueFamilyProperties& queue_family = queue_family_properties[index];
        if (queue_family.queueCount == 0) {
            continue;
        }
        if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphics = index;
        }
        if (surface && physical.GetSurfaceSupportKHR(index, surface)) {
            present = index;
        }
    }
    if (!graphics) {
        LOG_ERROR(Render_Vulkan, "Device lacks a graphics queue");
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (surface && !present) {
        LOG_ERROR(Render_Vulkan, "Device lacks a present queue");
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    if (graphics) {
        graphics_family = *graphics;
        graphics_family_sparse_binding =
            (queue_family_properties[*graphics].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
    }
    if (present) {
        present_family = *present;
    }
}

bool Device::TryReserveCustomBorderColorSamplers(size_t count) const {
    const size_t limit = properties.custom_border_color.maxCustomBorderColorSamplers;
    if (limit == 0) {
        return true;
    }
    size_t used = custom_border_color_samplers_used.load(std::memory_order_relaxed);
    while (used + count <= limit) {
        if (custom_border_color_samplers_used.compare_exchange_weak(
                used, used + count, std::memory_order_relaxed, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

void Device::ReleaseCustomBorderColorSamplers(size_t count) const {
    if (count == 0) {
        return;
    }
    custom_border_color_samplers_used.fetch_sub(count, std::memory_order_relaxed);
}

u64 Device::GetDeviceMemoryUsage() const {
    // EDEN_NX2_EXP29 & 1 (the original's EDEN_NX2_MEMSTAT_MS): the buffer and texture
    // caches ask every frame; with the budget extension each answer is a driver and OS query.
    // An answer younger than 50 ms is reused (so the caches' GC sees usage up to 50 ms old).
    if ((VideoCore::Nx2Debug::Experiment(29) & 1) == 0) {
        // Kept for LastDeviceMemoryUsage (the memory stage's thread does not ask, spec 5.3).
        const u64 usage = QueryDeviceMemoryUsage();
        memory_usage_value.store(usage, std::memory_order_relaxed);
        return usage;
    }
    constexpr s64 reuse_ns = 50'000'000;
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    const s64 last = memory_usage_time.load(std::memory_order_acquire);
    if (last != 0 && now - last < reuse_ns) {
        return memory_usage_value.load(std::memory_order_relaxed);
    }
    const u64 usage = QueryDeviceMemoryUsage();
    memory_usage_value.store(usage, std::memory_order_relaxed);
    memory_usage_time.store(now, std::memory_order_release);
    return usage;
}

// The buffer allocator's heap budgets (memory stage spec 5.3). At FRITCHII_MEMSTAGE_THREAD=0 every
// caller asks VMA directly, as before the stage's thread. At 1 only the allocator's user calls VMA
// for them (the back, or GPUMem inside a hand-off: MemoryStageLink::IsCacheUser), when the
// allocator changed since they were taken (its finished-operation count moved); everyone else
// reads the published values. The publish and the reads hold buffer_budgets_mutex: two
// publishers (a broken thread rule) can't interleave, and a reader never sees half of one.
void Device::PublishBufferHeapBudgetsLocked() const {
    // Counted when each operation ended (MemoryStageLink::AllocatorOp): an operation still running
    // during vmaGetHeapBudgets is counted later, and the next call takes the budgets again.
    const u64 ops = VideoCommon::MemoryStageLink::BufferAllocatorOps();
    if (buffer_budget_ops == ops) {
        return;
    }
    if (!VideoCommon::MemoryStageLink::IsCacheUser()) {
        VideoCommon::MemoryStageLink::Note(VideoCommon::MemoryStageLink::Check::BudgetUser,
                                           "buffer allocator");
        return;
    }
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(buffer_allocator, budgets.data());
    for (size_t heap = 0; heap < budgets.size(); ++heap) {
        const VmaBudget& budget = budgets[heap];
        buffer_budgets[heap] = {budget.statistics.blockCount,
                                budget.statistics.allocationCount,
                                budget.statistics.blockBytes,
                                budget.statistics.allocationBytes,
                                budget.usage,
                                budget.budget};
    }
    buffer_budget_ops = ops;
}

void Device::PublishBufferHeapBudgets() const {
    if (buffer_allocator == nullptr || !VideoCommon::MemoryStageLink::Checking()) {
        return;
    }
    std::scoped_lock lock{buffer_budgets_mutex};
    PublishBufferHeapBudgetsLocked();
}

void Device::BufferHeapBudgets(VmaBudget* budgets) const {
    if (buffer_allocator == nullptr) {
        std::fill_n(budgets, VK_MAX_MEMORY_HEAPS, VmaBudget{});
        return;
    }
    if (!VideoCommon::MemoryStageLink::Checking()) {
        // FRITCHII_MEMSTAGE_THREAD=0: VMA's numbers of this moment, as before the stage's thread
        // (the texture cache collector's thresholds read them).
        vmaGetHeapBudgets(buffer_allocator, budgets);
        return;
    }
    std::scoped_lock lock{buffer_budgets_mutex};
    if (VideoCommon::MemoryStageLink::IsCacheUser()) {
        PublishBufferHeapBudgetsLocked();
    }
    for (size_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; ++heap) {
        const auto& in = buffer_budgets[heap];
        VmaBudget& budget = budgets[heap];
        budget.statistics.blockCount = static_cast<u32>(in[0]);
        budget.statistics.allocationCount = static_cast<u32>(in[1]);
        budget.statistics.blockBytes = in[2];
        budget.statistics.allocationBytes = in[3];
        budget.usage = in[4];
        budget.budget = in[5];
    }
}

u64 Device::QueryDeviceMemoryUsage() const {
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::MemoryQuery};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    physical.GetMemoryProperties(&budget);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> heap_budgets{};
    if (allocator != nullptr) {
        GetHeapBudgets(*this, heap_budgets);
    }
    u64 result{};
    for (const size_t heap : valid_heap_memory) {
        u64 usage = AllocatedUsage(budget.heapUsage[heap], heap_budgets[heap]);
        if (kept_blocks) {
            // FRITCHII_KEEP_BLOCKS: kept blocks are free memory to the caches.
            usage -= (std::min)(usage, kept_blocks->KeptBytes(static_cast<u32>(heap)));
        }
        result += usage;
    }
    return result;
}

u64 Device::GetDeviceMemorySlack() const {
    if (allocator == nullptr) {
        return 0;
    }
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> heap_budgets{};
    GetHeapBudgets(*this, heap_budgets);
    const VmaStatistics& statistics = heap_budgets[main_heap].statistics;
    return statistics.blockBytes - (std::min)(statistics.blockBytes, statistics.allocationBytes);
}

bool Device::MainHeapUsage(u64& usage, u64& budget_bytes) const {
    if (!extensions.memory_budget) {
        return false;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    physical.GetMemoryProperties(&budget);
    usage = budget.heapUsage[main_heap];
    budget_bytes = budget.heapBudget[main_heap];
    if (heap_size_limit != 0) {
        budget_bytes = (std::min)(budget_bytes, heap_size_limit);
    }
    return true;
}

bool Device::MainHeapNumbers(u64& size, u64& usage, u64& budget_bytes) const {
    if (!extensions.memory_budget) {
        return false;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    const auto mem_info = physical.GetMemoryProperties(&budget);
    size = mem_info.memoryProperties.memoryHeaps[main_heap].size;
    usage = budget.heapUsage[main_heap];
    budget_bytes = budget.heapBudget[main_heap];
    if (heap_size_limit != 0) {
        size = (std::min)(size, heap_size_limit);
        budget_bytes = (std::min)(budget_bytes, heap_size_limit);
    }
    return true;
}

bool Device::GetDeviceLuid(u32& low, s32& high) const {
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;
    physical.GetProperties2(properties);
    if (!id.deviceLUIDValid) {
        return false;
    }
    std::memcpy(&low, id.deviceLUID, sizeof(low));
    std::memcpy(&high, id.deviceLUID + sizeof(low), sizeof(high));
    return true;
}

void Device::LogMemoryHeaps() const {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    const auto mem_info =
        physical.GetMemoryProperties(extensions.memory_budget ? &budget : nullptr);
    const auto& properties = mem_info.memoryProperties;
    for (u32 heap = 0; heap < properties.memoryHeapCount; ++heap) {
        const VkMemoryHeap& entry = properties.memoryHeaps[heap];
        const bool local = (entry.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        const std::string state =
            extensions.memory_budget
                ? fmt::format("driver budget {} MiB, usage {} MiB", budget.heapBudget[heap] >> 20,
                              budget.heapUsage[heap] >> 20)
                : std::string{"no budget (no VK_EXT_memory_budget)"};
        LOG_INFO(Render_Vulkan, "Memory heap {}: {} MiB, {}{}, {}", heap, entry.size >> 20,
                 local ? "device-local" : "host", heap == main_heap ? " (main heap, images)" : "",
                 state);
    }
    std::string types;
    for (u32 type = 0; type < properties.memoryTypeCount; ++type) {
        const VkMemoryPropertyFlags flags = properties.memoryTypes[type].propertyFlags;
        std::string kind;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            kind += "local ";
        }
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
            kind += "visible ";
        }
        if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
            kind += "coherent ";
        }
        if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) {
            kind += "cached ";
        }
        if (!kind.empty()) {
            kind.pop_back();
        }
        fmt::format_to(std::back_inserter(types), "{}{} heap {} ({})", type == 0 ? "" : "; ",
                       type, properties.memoryTypes[type].heapIndex,
                       kind.empty() ? "none" : kind);
    }
    LOG_INFO(Render_Vulkan, "Memory types: {}", types);
}

u64 Device::GetDeviceMemoryBudget() const {
    if (!extensions.memory_budget) {
        return 0;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    physical.GetMemoryProperties(&budget);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> heap_budgets{};
    if (allocator != nullptr) {
        GetHeapBudgets(*this, heap_budgets);
    }
    // Only the main heap's budget: summing every heap's counted VRAM twice on AMD (an 8 GB card
    // reported 8.5 GB) and kept the collector's thresholds above the heap the images fill. What
    // the other heaps use is added the way GetDeviceMemoryUsage() counts it, so the usage reaches
    // this when the main heap's does its budget.
    u64 result = budget.heapBudget[main_heap];
    if (heap_size_limit != 0) {
        result = (std::min)(result, heap_size_limit);
    }
    for (const size_t heap : valid_heap_memory) {
        if (heap != main_heap) {
            result += AllocatedUsage(budget.heapUsage[heap], heap_budgets[heap]);
        }
    }
    if (is_integrated && !IsMoltenVK()) {
        // An integrated GPU's device-local heap is system memory (a Radeon 780M reported a 21 GB
        // budget): following it let the texture cache fill that much RAM before collecting, and
        // kept ASTC uncompressed. Its budget is at most the share CollectPhysicalMemoryInfo
        // gives it (4 GiB, 6 GiB with aggressive VRAM use).
        result = (std::min)(result, device_access_memory);
    }
    return result;
}

void Device::CollectPhysicalMemoryInfo() {
    // Calculate limits using memory budget
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    const auto mem_info =
        physical.GetMemoryProperties(extensions.memory_budget ? &budget : nullptr);
    const auto& mem_properties = mem_info.memoryProperties;
    const size_t num_properties = mem_properties.memoryHeapCount;
    device_access_memory = 0;
    u64 device_initial_usage = 0;
    u64 local_memory = 0;
    // EDEN_NX2_VRAM_LIMIT_MIB=<n>: treat every device-local heap as n MiB (small card tests).
    if (const char* value = std::getenv("EDEN_NX2_VRAM_LIMIT_MIB")) {
        heap_size_limit = std::strtoull(value, nullptr, 10) << 20;
        if (heap_size_limit != 0) {
            LOG_INFO(Render_Vulkan, "EDEN_NX2_VRAM_LIMIT_MIB: device-local heaps limited to {} MiB",
                     heap_size_limit >> 20);
        } else {
            LOG_WARNING(Render_Vulkan, "EDEN_NX2_VRAM_LIMIT_MIB={} is not a size in MiB, ignored",
                        value);
        }
    }
    // FRITCHII_VRAM_SIM_OTHERS (boot, default 0; with EDEN_NX2_VRAM_LIMIT_MIB): the limit alone
    // gives the simulated card a budget of the whole limit, but a real card's budget is its heap
    // less what Windows and the other programs hold (16 GB card: 16376 MiB heap, 15297 budget).
    // An 8 GB card on the same desktop has the same others, so its budget is ~7.1 GiB, not 8.
    // With the switch the limit becomes the limit less those others, measured here once: the
    // budget, the texture cache's thresholds and VMA's heap limit then sit where a real card's
    // budget would. VMA fails past it, where a real card would spill into system memory; images
    // then go to host memory, which is as slow as the spill.
    if (const char* sim = std::getenv("FRITCHII_VRAM_SIM_OTHERS");
        sim != nullptr && sim[0] != '\0' && sim[0] != '0' && heap_size_limit != 0 &&
        extensions.memory_budget && !is_integrated) {
        size_t largest = 0;
        for (size_t heap = 0; heap < num_properties; ++heap) {
            const VkMemoryHeap& candidate = mem_properties.memoryHeaps[heap];
            if ((candidate.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0 &&
                ((mem_properties.memoryHeaps[largest].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ==
                     0 ||
                 candidate.size > mem_properties.memoryHeaps[largest].size)) {
                largest = heap;
            }
        }
        const u64 heap_size = mem_properties.memoryHeaps[largest].size;
        const u64 others = heap_size - (std::min)(heap_size, u64{budget.heapBudget[largest]});
        const u64 limited = heap_size_limit - (std::min)(others, heap_size_limit / 2);
        LOG_INFO(Render_Vulkan,
                 "FRITCHII_VRAM_SIM_OTHERS: others hold {} MiB of this card's {} MiB heap; the "
                 "simulated card's limit and budget are {} MiB (EDEN_NX2_VRAM_LIMIT_MIB {})",
                 others >> 20, heap_size >> 20, limited >> 20, heap_size_limit >> 20);
        heap_size_limit = limited;
    }
    const auto limit = [this](u64 size) {
        return heap_size_limit != 0 ? (std::min)(size, heap_size_limit) : size;
    };
    for (size_t element = 0; element < num_properties; ++element) {
        const bool is_heap_local =
            (mem_properties.memoryHeaps[element].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        if (!is_integrated && !is_heap_local) {
            continue;
        }
        valid_heap_memory.push_back(element);
        const VkMemoryHeap& main = mem_properties.memoryHeaps[main_heap];
        u64 heap_size = mem_properties.memoryHeaps[element].size;
        if (is_heap_local) {
            if ((main.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0 || heap_size > main.size) {
                main_heap = element;
            }
            heap_size = limit(heap_size);
            local_memory += heap_size;
        }
        if (extensions.memory_budget) {
            device_initial_usage += budget.heapUsage[element];
            device_access_memory += is_heap_local ? limit(budget.heapBudget[element])
                                                  : budget.heapBudget[element];
            continue;
        }
        device_access_memory += heap_size;
    }
    // FRITCHII_SMALL_CARD: auto mode's verdict from the main heap's size and budget (the
    // simulated card's when EDEN_NX2_VRAM_LIMIT_MIB is set). Without VK_EXT_memory_budget the
    // heap's size stands in for the budget.
    if (num_properties != 0) {
        const u64 main_size = limit(mem_properties.memoryHeaps[main_heap].size);
        const u64 main_budget =
            extensions.memory_budget ? limit(budget.heapBudget[main_heap]) : main_size;
        VideoCore::SmallCard::Decide(main_size, main_budget, is_integrated);
    }
    if (is_integrated) {
        const s64 available_memory = static_cast<s64>(device_access_memory - device_initial_usage);
        const u64 memory_size = Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive ? 6_GiB : 4_GiB;
        device_access_memory = static_cast<u64>(std::max<s64>(std::min<s64>(available_memory - 8_GiB, memory_size), std::min<s64>(local_memory, memory_size)));
    } else {
        const u64 reserve_memory = std::min<u64>(device_access_memory / 8, 1_GiB);
        device_access_memory -= reserve_memory;
        if (Settings::values.vram_usage_mode.GetValue() != Settings::VramUsageMode::Aggressive) {
            // Account for resolution scaling in memory limits
            const size_t normal_memory = 6_GiB;
            const size_t scaler_memory = 1_GiB * Settings::values.resolution_info.ScaleUp(1);
            device_access_memory = std::min<u64>(device_access_memory, normal_memory + scaler_memory);
        }
    }
}

void Device::CollectToolingInfo() {
    if (!extensions.tooling_info) {
        return;
    }
    auto tools{physical.GetPhysicalDeviceToolProperties()};
    for (const VkPhysicalDeviceToolProperties& tool : tools) {
        const std::string_view name = tool.name;
        LOG_INFO(Render_Vulkan, "Attached debugging tool: {}", name);
        has_renderdoc = has_renderdoc || name == "RenderDoc";
        has_nsight_graphics = has_nsight_graphics || name == "NVIDIA Nsight Graphics";
        has_radeon_gpu_profiler = has_radeon_gpu_profiler || name == "Radeon GPU Profiler";
    }
#ifdef _WIN32
    if (has_renderdoc) {
        LOG_INFO(Render_Vulkan,
                 "Windows default RenderDoc output folder: %LOCALAPPDATA%\\Temp\\RenderDoc");
    }
#endif
}

bool Device::WantsSideQueue() const {
    // Read where the device is made: after main() applied nx2_switches.txt.
    static const bool wanted = [this] {
        const char* value = std::getenv("FRITCHII_READ_SIDE_COPY");
        if (value == nullptr || value[0] < '2' || value[0] > '9') {
            return false;
        }
        const auto families = physical.GetQueueFamilyProperties();
        if (graphics_family >= families.size() || families[graphics_family].queueCount < 2) {
            LOG_WARNING(Render_Vulkan, "FRITCHII_READ_SIDE_COPY: the graphics family has one "
                                       "queue, no side queue (the side copy counts dry)");
            return false;
        }
        return true;
    }();
    return wanted;
}

std::optional<u32> Device::LinkHogFamily() const {
    // Read where the device is made: after main() applied nx2_switches.txt.
    static const std::optional<u32> family = [this]() -> std::optional<u32> {
        // The mode alone is enough: a live test can then start the copies at rate 0.
        const char* const value = std::getenv("FRITCHII_SIM_LINK_HOG");
        if ((value == nullptr || std::atoi(value) <= 0) &&
            std::getenv("FRITCHII_SIM_LINK_HOG_MODE") == nullptr) {
            return std::nullopt;
        }
        // A family of its own: copy-only if there is one (a copy engine), else compute-only.
        const auto families = physical.GetQueueFamilyProperties();
        std::optional<u32> compute_only;
        for (u32 index = 0; index < families.size(); ++index) {
            const VkQueueFlags flags = families[index].queueFlags;
            if (index == graphics_family || index == present_family ||
                families[index].queueCount == 0 || (flags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                continue;
            }
            if ((flags & VK_QUEUE_COMPUTE_BIT) == 0 && (flags & VK_QUEUE_TRANSFER_BIT) != 0) {
                return index;
            }
            if ((flags & VK_QUEUE_COMPUTE_BIT) != 0 && !compute_only) {
                compute_only = index;
            }
        }
        if (!compute_only) {
            LOG_WARNING(Render_Vulkan, "FRITCHII_SIM_LINK_HOG: no queue family for it, off");
        }
        return compute_only;
    }();
    return family;
}

bool Device::PrefersDescriptorSets() const {
    static const bool prefers = [this] {
        const char* const value = std::getenv("FRITCHII_DESCRIPTOR_SETS");
        VkDeviceSize window = 0;
        const auto memory = physical.GetMemoryProperties().memoryProperties;
        for (u32 type = 0; type < memory.memoryTypeCount; ++type) {
            const VkMemoryPropertyFlags flags = memory.memoryTypes[type].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 &&
                (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
                window = (std::max)(window, memory.memoryHeaps[memory.memoryTypes[type].heapIndex].size);
            }
        }
        // =2: where AMD's driver keeps command buffers and push descriptors in host memory: a
        // discrete card whose CPU-visible window is 7 GiB or less (no resizable BAR).
        const bool amd = GetDriverID() == VK_DRIVER_ID_AMD_PROPRIETARY ||
                         GetDriverID() == VK_DRIVER_ID_AMD_OPEN_SOURCE;
        const bool small_window = window <= VkDeviceSize{7} << 30;
        // Unset: v6.2's =2 rule on AMD's proprietary driver (discrete, CPU-visible window 7 GiB or
        // less). RX 9070 XT eGPU over OCuLink, 256 MiB window, OoT fountain with one follower at
        // Full: idle 16.45 -> 16.23 ms, under PCIe load 20.70 -> 19.04 ms, GPU busy -2 ms, the
        // picture unchanged. Cards with resizable BAR keep push descriptors (not measured there).
        // 0 = as the driver's features allow (v6.2's default), 1 = always, 2 = v6.2's rule (both
        // AMD drivers).
        const bool amd_default =
            GetDriverID() == VK_DRIVER_ID_AMD_PROPRIETARY && !IsIntegrated() && small_window;
        const int mode = value != nullptr ? std::atoi(value) : (amd_default ? 2 : 0);
        const bool on = mode == 1 || (mode == 2 && amd && !IsIntegrated() && small_window);
        LOG_INFO(Render_Vulkan,
                 "Descriptors: {} (FRITCHII_DESCRIPTOR_SETS={}{}; push descriptors {}, descriptor "
                 "buffer {}; CPU-visible device memory {} MiB)",
                 on ? "sets from pools" : "as the driver's features allow", mode,
                 value == nullptr ? (amd_default ? ", the AMD small-window default" : ", unset")
                                  : "",
                 IsKhrPushDescriptorSupported() ? "supported" : "not supported",
                 IsExtDescriptorBufferSupported() ? "supported" : "off", window >> 20);
        return on;
    }();
    return prefers;
}

std::vector<VkDeviceQueueCreateInfo> Device::GetDeviceQueueCreateInfos() const {
    static constexpr float QUEUE_PRIORITY = 1.0f;
    static constexpr std::array<float, 2> QUEUE_PRIORITIES{1.0f, 1.0f};

    ::Common::unordered_set<u32> unique_queue_families{graphics_family, present_family};
    std::vector<VkDeviceQueueCreateInfo> queue_cis;
    queue_cis.reserve(unique_queue_families.size());

    for (const u32 queue_family : unique_queue_families) {
        auto& ci = queue_cis.emplace_back(VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = queue_family,
            .queueCount = 1,
            .pQueuePriorities = nullptr,
        });
        ci.pQueuePriorities = &QUEUE_PRIORITY;
        // FRITCHII_READ_SIDE_COPY >= 2: the side queue is the graphics family's second.
        if (queue_family == graphics_family && WantsSideQueue()) {
            ci.queueCount = 2;
            ci.pQueuePriorities = QUEUE_PRIORITIES.data();
        }
    }
    if (const std::optional<u32> family = LinkHogFamily()) {
        queue_cis.push_back(VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = *family,
            .queueCount = 1,
            .pQueuePriorities = &QUEUE_PRIORITY,
        });
    }

    return queue_cis;
}

void Device::InitializeGPULogging() {
    // Get log level from settings - Off is the disable.
    const auto log_level = static_cast<GPU::Logging::LogLevel>(
        static_cast<u32>(Settings::values.gpu_log_level.GetValue()));
    if (log_level == GPU::Logging::LogLevel::Off) {
        return;
    }

    // Detect driver type
    const auto driver_id = GetDriverID();
    GPU::Logging::DriverType detected_driver = GPU::Logging::DriverType::Unknown;

    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        detected_driver = GPU::Logging::DriverType::Turnip;
    } else if (driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY) {
        detected_driver = GPU::Logging::DriverType::Qualcomm;
    }

    // Initialize GPU logger
    GPU::Logging::GPULogger::GetInstance().Initialize(log_level, detected_driver);

    // Configure feature flags
    GPU::Logging::GPULogger::GetInstance().EnableVulkanCallTracking(
        Settings::values.gpu_log_vulkan_calls.GetValue());
    GPU::Logging::GPULogger::GetInstance().EnableMemoryTracking(
        Settings::values.gpu_log_memory_tracking.GetValue());
    GPU::Logging::GPULogger::GetInstance().EnableDriverDebugInfo(
        Settings::values.gpu_log_driver_debug.GetValue());
    GPU::Logging::GPULogger::GetInstance().SetRingBufferSize(
        Settings::values.gpu_log_ring_buffer_size.GetValue());

    // Log comprehensive driver and extension information
    if (Settings::values.gpu_log_driver_debug.GetValue()) {
        std::string driver_info;

        // Device information
        const auto& props = properties.properties;
        driver_info += fmt::format("Device: {}\n", props.deviceName);
        driver_info += fmt::format("Driver Name: {}\n", properties.driver.driverName);
        driver_info += fmt::format("Driver Info: {}\n", properties.driver.driverInfo);

        // Version information
        const u32 driver_version = props.driverVersion;
        const u32 api_version = props.apiVersion;
        driver_info += fmt::format("Driver Version: {}.{}.{}\n",
            VK_API_VERSION_MAJOR(driver_version),
            VK_API_VERSION_MINOR(driver_version),
            VK_API_VERSION_PATCH(driver_version));
        driver_info += fmt::format("Vulkan API Version: {}.{}.{}\n",
            VK_API_VERSION_MAJOR(api_version),
            VK_API_VERSION_MINOR(api_version),
            VK_API_VERSION_PATCH(api_version));
        driver_info += fmt::format("Driver ID: {}\n", static_cast<u32>(driver_id));

        // Vendor and device IDs
        driver_info += fmt::format("Vendor ID: {:#04x}\n", props.vendorID);
        driver_info += fmt::format("Device ID: {:#04x}\n", props.deviceID);

        // Extensions - separate QCOM extensions from others
        driver_info += "\n=== Loaded Vulkan Extensions ===\n";
        std::vector<std::string> qcom_exts;
        std::vector<std::string> other_exts;

        for (const auto& ext : loaded_extensions) {
            if (ext.find("QCOM") != std::string::npos || ext.find("qcom") != std::string::npos) {
                qcom_exts.push_back(ext);
            } else {
                other_exts.push_back(ext);
            }
        }

        // Log QCOM extensions first
        if (!qcom_exts.empty()) {
            driver_info += "\nQualcomm Proprietary Extensions:\n";
            for (const auto& ext : qcom_exts) {
                driver_info += fmt::format("  - {}\n", ext);
            }
        }

        // Log other extensions
        if (!other_exts.empty()) {
            driver_info += "\nStandard Extensions:\n";
            for (const auto& ext : other_exts) {
                driver_info += fmt::format("  - {}\n", ext);
            }
        }

        driver_info += fmt::format("\nTotal Extensions Loaded: {}\n", loaded_extensions.size());

        GPU::Logging::GPULogger::GetInstance().LogDriverDebugInfo(driver_info);
    }
}

void Device::ShutdownGPULogging() {
    if (GPU::Logging::GPULogger::GetInstance().IsInitialized()) {
        GPU::Logging::GPULogger::GetInstance().Shutdown();
    }
}

std::vector<PhysicalDeviceSummary> SummarizePhysicalDevices(
    const vk::InstanceDispatch& dld, std::span<const VkPhysicalDevice> devices) {
    std::vector<PhysicalDeviceSummary> summaries;
    summaries.reserve(devices.size());
    for (const VkPhysicalDevice handle : devices) {
        const vk::PhysicalDevice physical(handle, dld);
        const VkPhysicalDeviceProperties properties = physical.GetProperties();
        PhysicalDeviceSummary summary;
        summary.name = properties.deviceName;
        summary.type = properties.deviceType;
        const VkPhysicalDeviceMemoryProperties memory = physical.GetMemoryProperties().memoryProperties;
        for (u32 heap = 0; heap < memory.memoryHeapCount; ++heap) {
            if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                summary.local_memory = (std::max)(summary.local_memory,
                                                  u64{memory.memoryHeaps[heap].size});
            }
        }
        summaries.push_back(std::move(summary));
    }
    return summaries;
}

PhysicalDevicePick PickPhysicalDevice(std::span<const PhysicalDeviceSummary> devices,
                                      u32 saved_index, std::string_view saved_name) {
    if (devices.empty()) {
        return {0, "no device"};
    }
    if (!saved_name.empty()) {
        // The saved index first: two cards of the same model keep the one picked.
        if (saved_index < devices.size() && devices[saved_index].name == saved_name) {
            return {saved_index, "the saved choice"};
        }
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].name == saved_name) {
                return {i, "the saved choice, found by name (the device order changed)"};
            }
        }
    } else if (saved_index != 0 && saved_index < devices.size()) {
        return {saved_index, "the saved choice (index, from before names were saved)"};
    }
    // Automatic: a discrete card before an integrated GPU, then the most device-local memory.
    // Index 0 stays when nothing beats it, so single-GPU PCs never change.
    const auto rank = [](VkPhysicalDeviceType type) {
        switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return 3;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return 2;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return 1;
        default:
            return 0;
        }
    };
    size_t best = 0;
    for (size_t i = 1; i < devices.size(); ++i) {
        const int candidate = rank(devices[i].type);
        const int current = rank(devices[best].type);
        if (candidate > current ||
            (candidate == current && devices[i].local_memory > devices[best].local_memory)) {
            best = i;
        }
    }
    if (!saved_name.empty()) {
        return {best, "automatic (the saved card is not in this PC any more)"};
    }
    if (saved_index >= devices.size()) {
        return {best, "automatic (the saved index is past the device list)"};
    }
    return {best, "automatic (discrete card with the most video memory first)"};
}

} // namespace Vulkan
