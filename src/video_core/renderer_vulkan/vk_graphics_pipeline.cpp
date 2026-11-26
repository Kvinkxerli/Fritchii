// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <iostream>
#include <mutex>
#include <set>
#include <span>
#include <unordered_set>
#include <unordered_map>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#include "video_core/renderer_vulkan/pipeline_helper.h"

#include "common/bit_field.h"
#include "common/fritchii_page_flags.h"
#include "common/scope_exit.h"
#include "video_core/fritchii_price.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_frame_dump.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_binary.h"
#include "video_core/renderer_vulkan/vk_render_pass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/memstage_stats.h"
#include "video_core/shader_handle_program.h"
#include "video_core/shader_notify.h"
#include "video_core/memory_stage_link.h"
#include "video_core/texture_handle_values.h"
#include "video_core/texture_cache/formatter.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/nx2_debug_control.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower.h"
#include "video_core/renderer_vulkan/vk_one_follower_draw.h"

#if defined(__clang__) && defined(NDEBUG)
#define LAMBDA_FORCEINLINE __attribute__((always_inline))
#elif defined(_MSC_VER) && defined(NDEBUG)
#define LAMBDA_FORCEINLINE [[msvc::forceinline]]
#else
#define LAMBDA_FORCEINLINE
#endif

namespace Vulkan {
namespace {
// EDEN_NX2_DRAW_REUSE_PROBE=1: how often a draw's setup (pipeline, texture handles and samplers,
// constant buffer addresses, vertex streams, render targets) repeats, from frame to frame and
// from the previous draw of the same pipeline: whether caching it can pay. Logged every 120 frames.
struct DrawReuseProbe {
    static bool Enabled() {
        static const bool enabled = std::getenv("EDEN_NX2_DRAW_REUSE_PROBE") != nullptr;
        return enabled;
    }
    static DrawReuseProbe& Get() {
        static DrawReuseProbe probe;
        return probe;
    }
    static u64 Mix(u64 hash, u64 value) {
        hash ^= value + 0x9E3779B97F4A7C15ULL + (hash << 6) + (hash >> 2);
        return hash;
    }
    static u64 Bytes(u64 hash, const void* data, size_t size) {
        const auto* bytes = static_cast<const u8*>(data);
        for (size_t i = 0; i + 8 <= size; i += 8) {
            u64 word;
            std::memcpy(&word, bytes + i, 8);
            hash = Mix(hash, word);
        }
        return hash;
    }
    void Note(const void* pipeline, u64 full, u64 textures, u64 no_cbufs) {
        ++draws;
        no_cbufs_seen += last_no_cbufs.contains(no_cbufs) ? 1 : 0;
        auto& previous_no_cbufs = previous_no_cbufs_by_pipeline[pipeline];
        no_cbufs_same_previous += previous_no_cbufs == no_cbufs ? 1 : 0;
        previous_no_cbufs = no_cbufs;
        this_no_cbufs.insert(no_cbufs);
        full_seen += last_full.contains(full) ? 1 : 0;
        textures_seen += last_textures.contains(textures) ? 1 : 0;
        auto& previous = previous_by_pipeline[pipeline];
        full_same_previous += previous.first == full ? 1 : 0;
        textures_same_previous += previous.second == textures ? 1 : 0;
        previous = {full, textures};
        this_full.insert(full);
        this_textures.insert(textures);
    }
    void Frame() {
        std::swap(last_full, this_full);
        std::swap(last_textures, this_textures);
        std::swap(last_no_cbufs, this_no_cbufs);
        this_full.clear();
        this_textures.clear();
        this_no_cbufs.clear();
        if (++frames < 120) {
            return;
        }
        const auto pct = [this](u64 n) { return draws ? 100.0 * static_cast<double>(n) / static_cast<double>(draws) : 0.0; };
        LOG_INFO(Render_Vulkan,
                 "Draw reuse probe: {:.0f} draws/frame; setup seen last frame {:.1f}%, textures seen "
                 "last frame {:.1f}%; same as the pipeline's previous draw: setup {:.1f}%, textures "
                 "{:.1f}%; without constant buffer addresses: seen last frame {:.1f}%, same as "
                 "previous {:.1f}%",
                 static_cast<double>(draws) / frames, pct(full_seen), pct(textures_seen),
                 pct(full_same_previous), pct(textures_same_previous), pct(no_cbufs_seen),
                 pct(no_cbufs_same_previous));
        frames = draws = full_seen = textures_seen = full_same_previous = textures_same_previous = 0;
        no_cbufs_seen = no_cbufs_same_previous = 0;
    }
    std::unordered_set<u64> last_full, this_full, last_textures, this_textures;
    std::unordered_map<const void*, std::pair<u64, u64>> previous_by_pipeline;
    std::unordered_set<u64> last_no_cbufs, this_no_cbufs;
    std::unordered_map<const void*, u64> previous_no_cbufs_by_pipeline;
    u64 no_cbufs_seen{}, no_cbufs_same_previous{};
    u64 frames{}, draws{}, full_seen{}, textures_seen{}, full_same_previous{},
        textures_same_previous{};
};

// EDEN_NX2_EXP32 (texture binding memo, GraphicsPipeline::TextureMemo), bits: 1 = on, 2 = verify
// (every hit for VERIFY_ALL_FRAMES frames after the switch changes, then one hit in VERIFY_EVERY:
// the full path runs and is compared with the memo), 4 = a mismatch turns the memo off until the
// switch changes. Counts are logged every 10 s while the switch is set. Used by the GPU back
// thread only (draws and RasterizerVulkan::TickFrame).
struct TextureMemoControl {
    static constexpr u32 VERIFY_ALL_FRAMES = 120;
    static constexpr u64 VERIFY_EVERY = 64;

    static TextureMemoControl& Get() {
        static TextureMemoControl control;
        return control;
    }

    [[nodiscard]] bool Usable(int value) const {
        return (value & 1) != 0 && value != disabled_value;
    }

    /// Whether this hit is checked against the full path.
    [[nodiscard]] bool ShouldVerify(int value) {
        if ((value & 2) == 0) {
            return false;
        }
        return verify_frames_left > 0 || ++verify_counter % VERIFY_EVERY == 0;
    }

    void Frame() {
        const int value = VideoCore::Nx2Debug::Experiment(32);
        if (value != last_value) {
            last_value = value;
            disabled_value = -1;
            ++generation;
            verify_frames_left = (value & 2) != 0 ? VERIFY_ALL_FRAMES : 0;
        } else if (verify_frames_left > 0) {
            --verify_frames_left;
        }
        if (value == 0 && lookups == 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (window_start == std::chrono::steady_clock::time_point{}) {
            window_start = now;
            return;
        }
        const double seconds = std::chrono::duration<double>(now - window_start).count();
        if (seconds < 10.0) {
            return;
        }
        const double hit_share =
            lookups != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(lookups) : 0.0;
        LOG_INFO(Render_Vulkan,
                 "Texture memo (EXP32={}) last {:.1f} s: {} draws looked up, {} hits ({:.1f}%) | "
                 "misses: no memo {}, state {}, handles {}, guest entries {}, images {} | "
                 "recorded {}, refused {} | verified {}, mismatches {}{}",
                 value, seconds, lookups, hits, hit_share, miss_no_memo, miss_state, miss_handles,
                 miss_entries, miss_images, recorded, refused, verified, mismatches,
                 disabled_value == value ? " (memo off after a mismatch)" : "");
        lookups = hits = miss_no_memo = miss_state = miss_handles = miss_entries = miss_images = 0;
        recorded = refused = verified = mismatches = 0;
        window_start = now;
    }

    int last_value = 0;
    int disabled_value = -1; ///< the switch value a mismatch turned the memo off for
    u64 generation = 1;      ///< memos recorded under another generation are dropped
    u32 verify_frames_left = 0;
    u64 verify_counter = 0;
    u32 mismatches_logged = 0;
    std::chrono::steady_clock::time_point window_start{};
    u64 lookups{}, hits{}, miss_no_memo{}, miss_state{}, miss_handles{}, miss_entries{},
        miss_images{}, recorded{}, refused{}, verified{}, mismatches{};
};

// EDEN_NX2_TEX_BANK_GEN (GraphicsPipeline::TexBankMemo): counts, logged every 10 s while the
// switch is set, and the verify cadence (value 2: one hit in VERIFY_EVERY resolves the full way).
// Used by the GPU back thread only.
struct TexBankControl {
    static constexpr u64 VERIFY_EVERY = 16;

    static TexBankControl& Get() {
        static TexBankControl control;
        return control;
    }

    [[nodiscard]] bool ShouldVerify() {
        return ++verify_counter % VERIFY_EVERY == 0;
    }

    void Frame() {
        const int value = VideoCore::Nx2Debug::g_tex_bank_gen.load(std::memory_order_relaxed);
        if (value <= 0 && lookups == 0 && ineligible == 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (window_start == std::chrono::steady_clock::time_point{}) {
            window_start = now;
            return;
        }
        const double seconds = std::chrono::duration<double>(now - window_start).count();
        if (seconds < 10.0) {
            return;
        }
        const double hit_share =
            lookups != 0 ? 100.0 * static_cast<double>(hits) / static_cast<double>(lookups) : 0.0;
        LOG_INFO(Render_Vulkan,
                 "Texture bank memo (TEX_BANK_GEN={}) last {:.1f} s: {} draws looked up, {} hits "
                 "({:.1f}%), {} draws of other pipelines | misses: no memo {}, state {}, handle "
                 "bindings {}, pages {} | recorded {}, refused {}, backed off {} | prepare "
                 "fallbacks {} | verified {}, mismatches {}",
                 value, seconds, lookups, hits, hit_share, ineligible, miss_no_memo, miss_state,
                 miss_cbufs, miss_pages, recorded, refused, backed_off, prepare_fallbacks,
                 verified, mismatches);
        lookups = hits = ineligible = miss_no_memo = miss_state = miss_cbufs = miss_pages = 0;
        recorded = refused = backed_off = prepare_fallbacks = verified = mismatches = 0;
        window_start = now;
    }

    u64 verify_counter = 0;
    u32 mismatches_logged = 0;
    std::chrono::steady_clock::time_point window_start{};
    u64 lookups{}, hits{}, ineligible{}, miss_no_memo{}, miss_state{}, miss_cbufs{},
        miss_pages{}, recorded{}, refused{}, backed_off{}, prepare_fallbacks{}, verified{},
        mismatches{};
};

/// A Vulkan handle's bits, for logs (a pointer or a 64-bit integer, depending on the platform).
template <typename Handle>
u64 HandleBits(Handle handle) {
    static_assert(sizeof(Handle) <= sizeof(u64));
    u64 bits = 0;
    std::memcpy(&bits, &handle, sizeof(Handle));
    return bits;
}

using boost::container::small_vector;
using boost::container::static_vector;
using Shader::ImageBufferDescriptor;
using Shader::Backend::SPIRV::MESH_DRAW_BASE_OFFSET;
using Shader::Backend::SPIRV::MeshDrawBaseLayout;
using Shader::Backend::SPIRV::MeshFirstEnabled;
using Shader::Backend::SPIRV::RENDERAREA_LAYOUT_OFFSET;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_DOWN_FACTOR_OFFSET;
using Shader::Backend::SPIRV::RESCALING_LAYOUT_WORDS_OFFSET;
using Tegra::Texture::TexturePair;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::PixelFormatFromDepthFormat;
using VideoCore::Surface::PixelFormatFromRenderTargetFormat;

constexpr size_t NUM_STAGES = Maxwell::MaxShaderStage;

// The stage of a program: task and mesh programs sit in the vertex and tessellation slots.
VkShaderStageFlagBits StageFlag(const Shader::Info& info, VkShaderStageFlagBits slot_stage) {
    if (info.is_task) {
        return VK_SHADER_STAGE_TASK_BIT_EXT;
    }
    if (info.is_mesh) {
        return VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    return slot_stage;
}

// Votes, ballots, lane ids and shuffles span the guest's 32-thread warp. NVIDIA always runs 32
// lanes, but Intel picks SIMD16 or SIMD32 per shader: a task shader's ballot then saw half the
// warp and its meshlet compaction dropped or repeated meshlets (flickering and missing foliage on
// Arc). Only stages with these ops are pinned, as in compute pipelines; quad ops need no more
// than 4 lanes.
bool UsesWarpWidthOps(const Shader::Info& info) {
    return info.uses_subgroup_vote || info.uses_subgroup_invocation_id ||
           info.uses_subgroup_shuffles || info.uses_subgroup_mask;
}

const void* GuestWarpSizeChain(const Device& device, const Shader::Info& info,
                               VkShaderStageFlagBits stage,
                               const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT& ci) {
    return UsesWarpWidthOps(info) && device.CanRequireGuestWarpSize(stage) ? &ci : nullptr;
}

// The stage of a pipeline stage slot (vertex, tessellation control and evaluation, geometry,
// fragment).
VkShaderStageFlagBits SlotStage(size_t index) {
    static constexpr std::array stages{
        VK_SHADER_STAGE_VERTEX_BIT,
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
        VK_SHADER_STAGE_GEOMETRY_BIT,
        VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    return stages.at(index);
}

DescriptorLayoutBuilder MakeBuilder(const Device& device, std::span<const Shader::Info> infos,
                                    const std::optional<MeshEmulationSetup>& mesh_emulation = {}) {
    DescriptorLayoutBuilder builder{device};
    if (mesh_emulation) {
        builder.EnableMeshEmulation();
    }
    for (size_t index = 0; index < infos.size(); ++index) {
        static constexpr std::array stages{
            VK_SHADER_STAGE_VERTEX_BIT,
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
            VK_SHADER_STAGE_GEOMETRY_BIT,
            VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        // Mesh emulation: the mesh program's resources are read by its compute shader.
        const bool emulated{infos[index].is_mesh || infos[index].is_task};
        const VkShaderStageFlags stage{
            mesh_emulation && emulated
                ? static_cast<VkShaderStageFlags>(VK_SHADER_STAGE_COMPUTE_BIT)
                : static_cast<VkShaderStageFlags>(StageFlag(infos[index], stages.at(index)))};
        builder.Add(infos[index], stage);
    }
    if (mesh_emulation) {
        builder.AddMeshEmulationOutput(mesh_emulation->output_binding);
        if (mesh_emulation->has_task) {
            builder.AddMeshEmulationOutput(mesh_emulation->payload_binding);
            builder.AddMeshEmulationOutput(mesh_emulation->offsets_binding);
            builder.AddMeshEmulationOutput(mesh_emulation->args_binding);
        }
    }
    return builder;
}

template <class StencilFace>
VkStencilOpState GetStencilFaceState(const StencilFace& face) {
    return {
        .failOp = MaxwellToVK::StencilOp(face.ActionStencilFail()),
        .passOp = MaxwellToVK::StencilOp(face.ActionDepthPass()),
        .depthFailOp = MaxwellToVK::StencilOp(face.ActionDepthFail()),
        .compareOp = MaxwellToVK::ComparisonOp(face.TestFunc()),
        .compareMask = 0,
        .writeMask = 0,
        .reference = 0,
    };
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    static constexpr std::array unsupported_topologies{
        VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
        // VK_PRIMITIVE_TOPOLOGY_QUAD_LIST_EXT,
    };
    return std::ranges::find(unsupported_topologies, topology) == unsupported_topologies.end();
}

bool IsLine(VkPrimitiveTopology topology) {
    static constexpr std::array line_topologies{
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
        // VK_PRIMITIVE_TOPOLOGY_LINE_LOOP_EXT,
    };
    return std::ranges::find(line_topologies, topology) != line_topologies.end();
}

VkViewportSwizzleNV UnpackViewportSwizzle(u16 swizzle) {
    union Swizzle {
        u32 raw;
        BitField<0, 3, Maxwell::ViewportSwizzle> x;
        BitField<4, 3, Maxwell::ViewportSwizzle> y;
        BitField<8, 3, Maxwell::ViewportSwizzle> z;
        BitField<12, 3, Maxwell::ViewportSwizzle> w;
    };
    const Swizzle unpacked{swizzle};
    return VkViewportSwizzleNV{
        .x = MaxwellToVK::ViewportSwizzle(unpacked.x),
        .y = MaxwellToVK::ViewportSwizzle(unpacked.y),
        .z = MaxwellToVK::ViewportSwizzle(unpacked.z),
        .w = MaxwellToVK::ViewportSwizzle(unpacked.w),
    };
}

PixelFormat DecodeFormat(u8 encoded_format) {
    const auto format{static_cast<Tegra::RenderTargetFormat>(encoded_format)};
    if (format == Tegra::RenderTargetFormat::NONE) {
        return PixelFormat::Invalid;
    }
    return PixelFormatFromRenderTargetFormat(format);
}

RenderPassKey MakeRenderPassKey(const FixedPipelineState& state, const Device& device) {
    RenderPassKey key{};
    std::ranges::transform(state.color_formats, key.color_formats.begin(), DecodeFormat);
    if (state.depth_enabled != 0) {
        const auto depth_format{static_cast<Tegra::DepthFormat>(state.depth_format.Value())};
        key.depth_format = PixelFormatFromDepthFormat(depth_format);
    } else {
        key.depth_format = PixelFormat::Invalid;
    }
    key.samples = MaxwellToVK::MsaaMode(state.msaa_mode);
    const bool has_color = std::ranges::any_of(key.color_formats, [](PixelFormat format) {
        return format != PixelFormat::Invalid;
    });
    key.resolve_color =
        key.samples != VK_SAMPLE_COUNT_1_BIT && has_color && device.IsTiler();
    return key;
}

size_t NumAttachments(const FixedPipelineState& state) {
    size_t num{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; ++index) {
        const auto format{static_cast<Tegra::RenderTargetFormat>(state.color_formats[index])};
        if (format != Tegra::RenderTargetFormat::NONE) {
            num = index + 1;
        }
    }
    return num;
}

template <typename Spec>
bool Passes(const std::array<bool, NUM_STAGES>& stages,
            const std::array<Shader::Info, NUM_STAGES>& stage_infos) {
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!Spec::enabled_stages[stage] && stages[stage]) {
            return false;
        }
        const auto& info{stage_infos[stage]};
        if constexpr (!Spec::has_storage_buffers) {
            if (!info.storage_buffers_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_texture_buffers) {
            if (!info.texture_buffer_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_image_buffers) {
            if (!info.image_buffer_descriptors.empty()) {
                return false;
            }
        }
        if constexpr (!Spec::has_images) {
            if (!info.image_descriptors.empty()) {
                return false;
            }
        }
    }
    return true;
}

using ConfigureFuncPtr = bool (*)(GraphicsPipeline*, bool);

template <bool KeepLast, bool Delta, typename Spec, typename... Specs>
ConfigureFuncPtr FindSpec(const std::array<bool, NUM_STAGES>& stages,
                          const std::array<Shader::Info, NUM_STAGES>& stage_infos) {
    if constexpr (sizeof...(Specs) > 0) {
        if (!Passes<Spec>(stages, stage_infos)) {
            return FindSpec<KeepLast, Delta, Specs...>(stages, stage_infos);
        }
    }
    return GraphicsPipeline::MakeConfigureSpecFunc<Spec, KeepLast, Delta>();
}

struct SimpleVertexFragmentSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleVertexSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, false};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleStorageSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = true;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = false;
};

struct SimpleImageSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, false, false, false, true};
    static constexpr bool has_storage_buffers = false;
    static constexpr bool has_texture_buffers = false;
    static constexpr bool has_image_buffers = false;
    static constexpr bool has_images = true;
};

struct DefaultSpec {
    static constexpr std::array<bool, 5> enabled_stages{true, true, true, true, true};
    static constexpr bool has_storage_buffers = true;
    static constexpr bool has_texture_buffers = true;
    static constexpr bool has_image_buffers = true;
    static constexpr bool has_images = true;
};

/// KeepLast: the delta draw record's instantiation (GraphicsPipeline::Configure's keep_last).
/// Delta: EXP118's (GraphicsPipeline::ConfigureDelta), the same Spec as Configure's.
template <bool KeepLast = false, bool Delta = false>
ConfigureFuncPtr ConfigureFunc(const std::array<bool, NUM_STAGES>& stages,
                               const std::array<Shader::Info, NUM_STAGES>& infos) {
    return FindSpec<KeepLast, Delta, SimpleVertexSpec, SimpleVertexFragmentSpec,
                    SimpleStorageSpec, SimpleImageSpec, DefaultSpec>(stages, infos);
}

/// The stages ConfigureFunc's Spec prepares, one bit each: the memory stage's fill runs the
/// same per-stage calls (MemStage::FillDraw).
template <typename Spec, typename... Specs>
u32 FindSpecStages(const std::array<bool, NUM_STAGES>& stages,
                   const std::array<Shader::Info, NUM_STAGES>& infos) {
    if constexpr (sizeof...(Specs) > 0) {
        if (!Passes<Spec>(stages, infos)) {
            return FindSpecStages<Specs...>(stages, infos);
        }
    }
    u32 mask = 0;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        mask |= Spec::enabled_stages[stage] ? 1U << stage : 0U;
    }
    return mask;
}

u32 SpecStages(const std::array<bool, NUM_STAGES>& stages,
               const std::array<Shader::Info, NUM_STAGES>& infos) {
    return FindSpecStages<SimpleVertexSpec, SimpleVertexFragmentSpec, SimpleStorageSpec,
                          SimpleImageSpec, DefaultSpec>(stages, infos);
}

std::atomic<std::chrono::nanoseconds::rep> hold_time_ns{};
std::atomic<std::chrono::nanoseconds::rep> hold_time_total_ns{};

/// EDEN_NX2_CLAIM_HELD_BUILDS (default 1): a draw held for a worker-emitted (native) pipeline
/// whose build no worker has started yet runs that build itself instead of waiting for the
/// FIFO pool to reach it.
bool HeldDrawsClaimBuilds() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_CLAIM_HELD_BUILDS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// EDEN_NX2_WAIT_LIMIT_MS (default 16; 0 = wait as before): a draw into a depth/shadow/probe
/// target, or into a target already drawn this session, skips after the cap and retries next
/// frame. First-time color/UI still waits, so a one-shot menu or title draw is not dropped.
int WaitLimitMs() {
    static const int value = [] {
        const char* const text = std::getenv("EDEN_NX2_WAIT_LIMIT_MS");
        if (text == nullptr || text[0] == '\0') {
            return 16;
        }
        char* end{};
        const long parsed = std::strtol(text, &end, 10);
        if (end == text || parsed < 0 || parsed > 10000) {
            return 16;
        }
        return static_cast<int>(parsed);
    }();
    return value;
}

enum class WaitLimitReason : u8 { None, DepthShadow, Probe, AlreadyDrawn };

const char* WaitLimitReasonName(WaitLimitReason reason) {
    switch (reason) {
    case WaitLimitReason::DepthShadow:
        return "depth/shadow";
    case WaitLimitReason::Probe:
        return "probe";
    case WaitLimitReason::AlreadyDrawn:
        return "already-drawn";
    default:
        return "none";
    }
}

const char* DrawTargetKindName(VideoCommon::DrawTargetClass kind) {
    switch (kind) {
    case VideoCommon::DrawTargetClass::Recurring:
        return "recurring";
    case VideoCommon::DrawTargetClass::FirstTime:
        return "first-time";
    case VideoCommon::DrawTargetClass::Persistent:
        return "persistent";
    }
    return "unknown";
}

bool IsDepthOrStencilFormat(VideoCore::Surface::PixelFormat format) {
    switch (VideoCore::Surface::GetFormatType(format)) {
    case VideoCore::Surface::SurfaceType::Depth:
    case VideoCore::Surface::SurfaceType::Stencil:
    case VideoCore::Surface::SurfaceType::DepthStencil:
        return true;
    default:
        return false;
    }
}

bool IsLightingProbe(const VideoCommon::DrawTargets& targets) {
    if (targets.gpu_addr == 0 || IsDepthOrStencilFormat(targets.format)) {
        return false;
    }
    if ((std::max)(targets.width, targets.height) > 512) {
        return false;
    }
    switch (targets.format) {
    case VideoCore::Surface::PixelFormat::R16G16B16A16_FLOAT:
    case VideoCore::Surface::PixelFormat::B10G11R11_FLOAT:
    case VideoCore::Surface::PixelFormat::R32G32B32A32_FLOAT:
    case VideoCore::Surface::PixelFormat::R32G32_FLOAT:
    case VideoCore::Surface::PixelFormat::R32_FLOAT:
    case VideoCore::Surface::PixelFormat::R16_FLOAT:
    case VideoCore::Surface::PixelFormat::R16G16_FLOAT:
    case VideoCore::Surface::PixelFormat::R16G16B16X16_FLOAT:
    case VideoCore::Surface::PixelFormat::R32G32B32_FLOAT:
    case VideoCore::Surface::PixelFormat::E5B9G9R9_FLOAT:
        return true;
    default:
        return false;
    }
}

struct WaitLimitStats {
    std::mutex mutex;
    std::unordered_set<GPUVAddr> drawn;
    u64 skip_first_time{};
    u64 skip_persistent{};
    u64 skip_recurring{};
    u64 skip_depth{};
    u64 skip_probe{};
    u64 skip_already{};
    u64 finished_in_cap{};
    u64 boosted{};
    std::chrono::steady_clock::time_point window{};
    bool logged_mode{};
};

WaitLimitStats wait_limit_stats;

void NoteDrawnTarget(const VideoCommon::DrawTargets& targets) {
    if (targets.gpu_addr == 0) {
        return;
    }
    std::scoped_lock lock{wait_limit_stats.mutex};
    wait_limit_stats.drawn.insert(targets.gpu_addr);
}

bool TargetDrawnThisSession(GPUVAddr gpu_addr) {
    if (gpu_addr == 0) {
        return false;
    }
    std::scoped_lock lock{wait_limit_stats.mutex};
    return wait_limit_stats.drawn.contains(gpu_addr);
}

WaitLimitReason WaitLimitReasonFor(const VideoCommon::DrawTargets& targets) {
    if (WaitLimitMs() <= 0) {
        return WaitLimitReason::None;
    }
    if (IsDepthOrStencilFormat(targets.format)) {
        return WaitLimitReason::DepthShadow;
    }
    if (IsLightingProbe(targets)) {
        return WaitLimitReason::Probe;
    }
    if (TargetDrawnThisSession(targets.gpu_addr)) {
        return WaitLimitReason::AlreadyDrawn;
    }
    return WaitLimitReason::None;
}

void ReportWaitLimitLocked(WaitLimitStats& stats, std::chrono::steady_clock::time_point now) {
    if (stats.skip_first_time + stats.skip_persistent + stats.skip_recurring +
            stats.finished_in_cap + stats.boosted ==
        0) {
        stats.window = now;
        return;
    }
    LOG_INFO(Render_Vulkan,
             "WaitLimit in the last 10 s: skipped {} draws ({} first-time, {} persistent, "
             "{} recurring; {} depth/shadow, {} probe, {} already-drawn); {} waits finished in "
             "the cap; boosted {} in-flight builders; cap {} ms",
             stats.skip_first_time + stats.skip_persistent + stats.skip_recurring,
             stats.skip_first_time, stats.skip_persistent, stats.skip_recurring, stats.skip_depth,
             stats.skip_probe, stats.skip_already, stats.finished_in_cap, stats.boosted,
             WaitLimitMs());
    stats.skip_first_time = 0;
    stats.skip_persistent = 0;
    stats.skip_recurring = 0;
    stats.skip_depth = 0;
    stats.skip_probe = 0;
    stats.skip_already = 0;
    stats.finished_in_cap = 0;
    stats.boosted = 0;
    stats.window = now;
}

void MaybeReportWaitLimit() {
    auto& stats = wait_limit_stats;
    std::scoped_lock lock{stats.mutex};
    const auto now = std::chrono::steady_clock::now();
    if (stats.window == std::chrono::steady_clock::time_point{}) {
        stats.window = now;
        return;
    }
    if (now - stats.window >= std::chrono::seconds{10}) {
        ReportWaitLimitLocked(stats, now);
    }
}

void NoteWaitLimitSkip(const VideoCommon::DrawTargets& targets, WaitLimitReason reason,
                       std::chrono::nanoseconds held, int limit_ms) {
    auto& stats = wait_limit_stats;
    std::scoped_lock lock{stats.mutex};
    switch (targets.kind) {
    case VideoCommon::DrawTargetClass::FirstTime:
        ++stats.skip_first_time;
        break;
    case VideoCommon::DrawTargetClass::Persistent:
        ++stats.skip_persistent;
        break;
    case VideoCommon::DrawTargetClass::Recurring:
        ++stats.skip_recurring;
        break;
    }
    switch (reason) {
    case WaitLimitReason::DepthShadow:
        ++stats.skip_depth;
        break;
    case WaitLimitReason::Probe:
        ++stats.skip_probe;
        break;
    case WaitLimitReason::AlreadyDrawn:
        ++stats.skip_already;
        break;
    default:
        break;
    }
    LOG_INFO(Render_Vulkan,
             "WaitLimit: skip {} {}x{} {} {:#x} after {:.1f} ms ({}{}; EDEN_NX2_WAIT_LIMIT_MS={})",
             DrawTargetKindName(targets.kind), targets.width, targets.height, targets.format,
             targets.gpu_addr, std::chrono::duration<double, std::milli>(held).count(),
             WaitLimitReasonName(reason),
             stats.drawn.contains(targets.gpu_addr) ? ", seen before" : "", limit_ms);
}

/// Returns false if this draw should be skipped (retry next frame). Records hold time either way.
bool HoldForPipelineBuild(PipelineBuild& build, Scheduler& scheduler,
                          GraphicsPipelineLibraryCache* library_cache, TextureCache& texture_cache,
                          bool worker_only_build) {
    auto& stats = wait_limit_stats;
    {
        std::scoped_lock lock{stats.mutex};
        if (!stats.logged_mode) {
            stats.logged_mode = true;
            LOG_INFO(Render_Vulkan,
                     "WaitLimit: EDEN_NX2_WAIT_LIMIT_MS={}; depth/shadow/probe and already-drawn "
                     "targets skip the draw after the cap and retry next frame; first-time "
                     "color/UI still waits; 0 waits as before",
                     WaitLimitMs());
        }
    }
    const auto targets = texture_cache.ClassifyDrawTargets();
    const auto reason = WaitLimitReasonFor(targets);
    const int limit_ms = WaitLimitMs();
    if (reason == WaitLimitReason::None) {
        PipelineBuild* const pending = &build;
        scheduler.Record([pending, library_cache, worker_only_build](vk::CommandBuffer) {
            const auto begin = std::chrono::steady_clock::now();
            const bool boost = !pending->IsBuilt();
            pending->Wait(!worker_only_build || HeldDrawsClaimBuilds(), -1, true);
            const auto held = std::chrono::steady_clock::now() - begin;
            const auto held_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(held).count();
            hold_time_ns.fetch_add(held_ns, std::memory_order::relaxed);
            hold_time_total_ns.fetch_add(held_ns, std::memory_order::relaxed);
            if (library_cache) {
                library_cache->RecordHeldDraw(held);
            }
            if (boost) {
                std::scoped_lock lock{wait_limit_stats.mutex};
                ++wait_limit_stats.boosted;
            }
        });
        NoteDrawnTarget(targets);
        MaybeReportWaitLimit();
        return true;
    }
    build.KickUrgent();
    const auto begin = std::chrono::steady_clock::now();
    const bool ready = build.Wait(false, limit_ms, true);
    const auto held = std::chrono::steady_clock::now() - begin;
    const auto held_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(held).count();
    hold_time_ns.fetch_add(held_ns, std::memory_order::relaxed);
    hold_time_total_ns.fetch_add(held_ns, std::memory_order::relaxed);
    if (library_cache) {
        library_cache->RecordHeldDraw(held);
    }
    {
        std::scoped_lock lock{stats.mutex};
        ++stats.boosted;
        if (ready) {
            ++stats.finished_in_cap;
        }
    }
    if (!ready) {
        NoteWaitLimitSkip(targets, reason, held, limit_ms);
        MaybeReportWaitLimit();
        return false;
    }
    NoteDrawnTarget(targets);
    MaybeReportWaitLimit();
    return true;
}

// Worker builds of kept SPIR-V (the precompiled fast path): how many the driver compiled again
// instead of finding in its own cache. On an RX 7900 XT those took 0.2-1.1 s each and the draws
// waiting for them froze the game (tester log t1858), while a cache hit takes a few ms.
std::atomic<u64> kept_builds{};
std::atomic<u64> kept_builds_slow{};
std::atomic<u64> kept_builds_long{};
std::atomic<std::chrono::nanoseconds::rep> kept_builds_slow_ns{};
std::atomic<std::chrono::nanoseconds::rep> kept_builds_longest_ns{};

// EXP87 draw templates on the drawing thread (ported), logged every 10 s
// while EXP87 != 0.
struct DrawTemplateStats {
    u64 hits{};
    u64 misses{};
    u64 kept{};
    u64 not_kept{}; ///< Changing sampler variants, too many textures or descriptors not keepable
    std::array<u64, static_cast<size_t>(VideoCommon::DrawTemplateCheck::Count)> rejected{};
    u64 prepare_deleted{};
    u64 verified{};
    std::array<u64, 4> mismatches{}; ///< Views, samplers, descriptors, rescaling words
    u64 mismatched_draws{};
    u64 total_mismatched_draws{};
    u64 calls{}; ///< ReportDrawTemplates calls: the clock is read once every 1,024
    int mode{};  ///< The EXP87 value the counts are for: a toggle starts them over
    std::chrono::steady_clock::time_point next_report{};
} draw_template_stats;

void ReportDrawTemplates(int mode) {
    DrawTemplateStats& stats = draw_template_stats;
    if (stats.mode != mode) {
        // EXP87 changed: the counts start over, and the first window after it is not logged.
        stats = {.total_mismatched_draws = stats.total_mismatched_draws, .mode = mode};
    }
    if ((++stats.calls & 1023) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < stats.next_report) {
        return;
    }
    if (stats.next_report != std::chrono::steady_clock::time_point{}) {
        using Check = VideoCommon::DrawTemplateCheck;
        const auto rejected = [&](Check check) {
            return stats.rejected[static_cast<size_t>(check)];
        };
        LOG_INFO(Render_Vulkan,
                 "Draw templates in 10 s (EXP87={}): {} hits, {} misses, {} kept, {} not keepable; "
                 "rejected: {} host memory, {} scale, {} view deleted, {} descriptor bytes, {} "
                 "prepare deleted images",
                 mode, stats.hits, stats.misses, stats.kept, stats.not_kept,
                 rejected(Check::HostMemory), rejected(Check::Scale), rejected(Check::ViewDeleted),
                 rejected(Check::Descriptors), stats.prepare_deleted);
        if (mode == 2) {
            LOG_INFO(Render_Vulkan,
                     "Draw template verify in 10 s: {} checked, {} mismatched: {} views, {} "
                     "samplers, {} descriptors, {} rescaling words ({} mismatched in total)",
                     stats.verified, stats.mismatched_draws, stats.mismatches[0],
                     stats.mismatches[1], stats.mismatches[2], stats.mismatches[3],
                     stats.total_mismatched_draws);
        }
    }
    stats = {.total_mismatched_draws = stats.total_mismatched_draws,
             .mode = mode,
             .next_report = now + std::chrono::seconds{10}};
}
} // Anonymous namespace

void NoteDrawReuseProbeFrame();

void NoteDrawReuseProbeFrame() {
    if (DrawReuseProbe::Enabled()) [[unlikely]] {
        DrawReuseProbe::Get().Frame();
    }
    // The texture binding memo's frame too (EXP32: verify window, switch changes, its log).
    TextureMemoControl::Get().Frame();
    TexBankControl::Get().Frame();
}

std::chrono::nanoseconds GraphicsPipeline::TakeHoldTime() noexcept {
    return std::chrono::nanoseconds{hold_time_ns.exchange(0, std::memory_order::relaxed)};
}

std::chrono::nanoseconds GraphicsPipeline::HoldTimeTotal() noexcept {
    return std::chrono::nanoseconds{hold_time_total_ns.load(std::memory_order::relaxed)};
}

GraphicsPipeline::KeptBuildStats GraphicsPipeline::TakeKeptBuildStats() noexcept {
    return KeptBuildStats{
        .builds = kept_builds.exchange(0, std::memory_order::relaxed),
        .slow = kept_builds_slow.exchange(0, std::memory_order::relaxed),
        .long_builds = kept_builds_long.exchange(0, std::memory_order::relaxed),
        .slow_time = std::chrono::nanoseconds{
            kept_builds_slow_ns.exchange(0, std::memory_order::relaxed)},
        .longest = std::chrono::nanoseconds{
            kept_builds_longest_ns.exchange(0, std::memory_order::relaxed)},
    };
}

// TODO: This is the worst-formatted code I have EVER seen
GraphicsPipeline::GraphicsPipeline(
    Scheduler& scheduler_, BufferCache& buffer_cache_, TextureCache& texture_cache_,
    vk::PipelineCache& pipeline_cache_, VideoCore::ShaderNotify* shader_notify,
    const Device& device_, DescriptorPool& descriptor_pool,
    GuestDescriptorQueue& guest_descriptor_queue_, DescriptorBufferRing& descriptor_buffer_ring_,
    DrawTemplates& draw_templates_, Common::ThreadWorker* worker_thread,
    PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
    const GraphicsPipelineCacheKey& key_, std::array<vk::ShaderModule, NUM_STAGES> stages,
    const std::array<const Shader::Info*, NUM_STAGES>& infos, EmitStagesFunc emit_stages,
    LibraryParams library_params,
    std::optional<MeshEmulationSetup> mesh_emulation_)
    : key{key_}, device{device_}, texture_cache{texture_cache_}, buffer_cache{buffer_cache_},
      pipeline_cache(pipeline_cache_), scheduler{scheduler_},
      guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_}, draw_templates{draw_templates_},
      template_owner{draw_templates_.NewOwner()}, spv_modules{std::move(stages)},
      library_cache{library_params.library_cache},
      libraries_only{library_params.library_cache && library_params.libraries_only},
      spirv_hashes{library_params.spirv_hashes},
      worker_only_build{worker_thread && static_cast<bool>(emit_stages)},
      prepared_build{worker_only_build && library_params.prepared},
      mesh_emulation{std::move(mesh_emulation_)} {
    if (shader_notify) {
        shader_notify->MarkShaderBuilding();
    }
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info* const info{infos[stage]};
        if (!info) {
            continue;
        }
        stage_infos[stage] = *info;
        enabled_uniform_buffer_masks[stage] = info->constant_buffer_mask;
        std::ranges::copy(info->constant_buffer_used_sizes, uniform_buffer_sizes[stage].begin());
        num_image_elements += Shader::NumDescriptors(info->texture_buffer_descriptors);
        num_image_elements += Shader::NumDescriptors(info->image_buffer_descriptors);
        num_textures += Shader::NumDescriptors(info->texture_descriptors);
        num_image_elements += Shader::NumDescriptors(info->texture_descriptors);
        num_image_elements += Shader::NumDescriptors(info->image_descriptors);
        num_descriptor_entries += NumDescriptorEntries(*info);
    }
    fragment_has_color0_output = stage_infos[NUM_STAGES - 1].stores_frag_color[0];
    // The guest leaves render target components the fragment shader doesn't output untouched,
    // while Vulkan writes undefined values to them (AMD writes garbage), so they are masked out.
    for (size_t rt = 0; rt < Maxwell::NumRenderTargets; ++rt) {
        color_output_mask |= u32{stage_infos[NUM_STAGES - 1].frag_color_components[rt]}
                             << (rt * 4);
    }
    const auto any_written = [](const auto& descriptors) {
        return std::ranges::any_of(descriptors, [](const auto& desc) { return desc.is_written; });
    };
    writes_storage = std::ranges::any_of(stage_infos, [&](const Shader::Info& info) {
        return info.stores_global_memory || any_written(info.storage_buffers_descriptors) ||
               any_written(info.image_descriptors) || any_written(info.image_buffer_descriptors);
    });
    samples_nothing = std::ranges::all_of(stage_infos, [](const Shader::Info& info) {
        return info.texture_descriptors.empty() && info.texture_buffer_descriptors.empty() &&
               info.image_descriptors.empty() && info.image_buffer_descriptors.empty();
    });
    has_handle_table = std::ranges::any_of(stage_infos, [](const Shader::Info& info) {
        return std::ranges::any_of(info.texture_descriptors, [](const auto& desc) {
            return desc.count > 1 && desc.handle_program != ~0U;
        });
    });
    // EXP87 (the original texture_memo_eligible).
    template_eligible = std::ranges::all_of(stage_infos, [](const Shader::Info& info) {
        return info.texture_buffer_descriptors.empty() && info.image_buffer_descriptors.empty() &&
               info.image_descriptors.empty() &&
               std::ranges::all_of(info.texture_descriptors, [](const auto& desc) {
                   return desc.handle_program == ~0U && !desc.has_secondary;
               });
    });

    if (library_cache) {
        // Graphics pipeline libraries: one set per stage (set index = stage index), plain
        // descriptor sets (a layout has room for one push descriptor set, and the descriptor
        // buffer path is left out).
        std::array<VkDescriptorSetLayout, NUM_STAGES> set_layouts{};
        size_t descriptor_offset{};
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            DescriptorLayoutBuilder stage_builder{device};
            stage_builder.Add(stage_infos[stage], StageFlag(stage_infos[stage], SlotStage(stage)));
            stage_descriptor_offsets[stage] = descriptor_offset;
            descriptor_offset +=
                sizeof(DescriptorUpdateEntry) * NumDescriptorEntries(stage_infos[stage]);
            stage_set_signatures[stage] = stage_builder.SetLayoutSignature();
            if (stage_builder.Empty()) {
                continue;
            }
            stage_set_layouts[stage] = stage_builder.CreateDescriptorSetLayout(false);
            set_layouts[stage] = *stage_set_layouts[stage];
            has_stage_sets = true;
            if (!libraries_only) {
                stage_templates[stage] =
                    stage_builder.CreateTemplate(set_layouts[stage], VK_NULL_HANDLE, false);
                stage_allocators[stage] = descriptor_pool.Allocator(
                    device, scheduler, set_layouts[stage], stage_infos[stage]);
            }
        }
        pipeline_layout =
            DescriptorLayoutBuilder{device}.CreateIndependentPipelineLayout(set_layouts, true);
    } else {
        DescriptorLayoutBuilder builder{MakeBuilder(device, stage_infos, mesh_emulation)};
        // Mesh emulation binds one descriptor set to both the compute dispatch and the draw.
        uses_push_descriptor = !mesh_emulation && builder.CanUsePushDescriptor();
        uses_descriptor_buffer = !mesh_emulation && builder.CanUseDescriptorBuffer() &&
                                 descriptor_buffer_ring.IsValid();
        // After every stage's descriptors: the output buffer, then with a task program its
        // payload, offset and map buffers.
        const u32 emulation_buffers{mesh_emulation ? (mesh_emulation->has_task ? 4U : 1U) : 0U};
        num_descriptor_entries += emulation_buffers;
        descriptor_set_layout =
            builder.CreateDescriptorSetLayout(uses_push_descriptor, uses_descriptor_buffer);
        if (uses_descriptor_buffer) {
            descriptor_buffer_layout = builder.MakeDescriptorBufferLayout(*descriptor_set_layout);
            if (!descriptor_buffer_ring.CanAllocate(descriptor_buffer_layout.size)) {
                LOG_WARNING(Render_Vulkan,
                            "Graphics pipeline {:016X} needs {} descriptor bytes per draw, falling back "
                            "to sets",
                            key.Hash(), descriptor_buffer_layout.size);
                uses_descriptor_buffer = false;
                descriptor_buffer_layout = {};
                descriptor_set_layout = builder.CreateDescriptorSetLayout(uses_push_descriptor);
            }
        }

        const VkDescriptorSetLayout set_layout{*descriptor_set_layout};
        pipeline_layout = builder.CreatePipelineLayout(set_layout);
        if (!uses_descriptor_buffer) {
            descriptor_update_template =
                builder.CreateTemplate(set_layout, *pipeline_layout, uses_push_descriptor);
            if (!uses_push_descriptor && mesh_emulation) {
                // The set also holds the emulation's buffers.
                std::array<Shader::Info, NUM_STAGES + 1> bank_infos{};
                std::ranges::copy(stage_infos, bank_infos.begin());
                bank_infos.back().storage_buffers_descriptors.push_back({
                    .cbuf_index = 0,
                    .cbuf_offset = 0,
                    .count = emulation_buffers,
                    .is_written = true,
                });
                descriptor_allocator =
                    descriptor_pool.Allocator(device, scheduler, set_layout, bank_infos);
                if (mesh_emulation->has_task) {
                    // The distribution pass: payload, offsets, map and arguments, plus its push
                    // constants (Shader::Backend::SPIRV::MeshEmulationDistributionConstants).
                    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
                    for (u32 binding = 0; binding < bindings.size(); ++binding) {
                        bindings[binding] = {
                            .binding = binding,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                            .pImmutableSamplers = nullptr,
                        };
                    }
                    distribution_set_layout = device.GetLogical().CreateDescriptorSetLayout({
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .bindingCount = static_cast<u32>(bindings.size()),
                        .pBindings = bindings.data(),
                    });
                    const VkDescriptorSetLayout distribution_set{*distribution_set_layout};
                    const VkPushConstantRange range{
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                        .offset = 0,
                        .size = sizeof(Shader::Backend::SPIRV::MeshEmulationDistributionConstants),
                    };
                    distribution_layout = device.GetLogical().CreatePipelineLayout({
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                        .pNext = nullptr,
                        .flags = 0,
                        .setLayoutCount = 1,
                        .pSetLayouts = &distribution_set,
                        .pushConstantRangeCount = 1,
                        .pPushConstantRanges = &range,
                    });
                    distribution_allocator = descriptor_pool.Allocator(
                        device, scheduler, distribution_set, DescriptorBankInfo{
                                                                 .storage_buffers = 4,
                                                                 .score = 4,
                                                             });
                }
            } else if (!uses_push_descriptor) {
                descriptor_allocator =
                    descriptor_pool.Allocator(device, scheduler, set_layout, stage_infos);
            }
        }
    }

    auto func{[this, shader_notify, &render_pass_cache, pipeline_statistics,
                emit_stages = std::move(emit_stages)] {
        const auto begin = std::chrono::steady_clock::now();
        SCOPE_EXIT {
            if (shader_notify) {
                shader_notify->MarkShaderComplete();
            }
        };
        try {
            if (emit_stages) {
                spv_modules = emit_stages(spirv_hashes);
            }
            const VkRenderPass render_pass{render_pass_cache.Get(MakeRenderPassKey(key.state, device))};
            Validate();
            if (library_cache) {
                const bool has_module = std::ranges::any_of(
                    spv_modules, [](const vk::ShaderModule& module) { return bool(module); });
                bool made{};
                try {
                    made = MakeLibraryPipeline(render_pass);
                } catch (const vk::Exception& exception) {
                    if (libraries_only) {
                        // Warm-up: counted by the warm-up and logged once at its end.
                        LOG_DEBUG(Render_Vulkan, "Nx2Perf gpl: warm {:016x} linking failed: {}",
                                  key.Hash(), exception.what());
                        throw;
                    }
                    LOG_ERROR(Render_Vulkan, "Nx2Perf gpl: pipeline {:016x} linking failed: {}",
                              key.Hash(), exception.what());
                }
                if (!made && libraries_only && has_module) {
                    // A warm-up build whose library the driver did not make: reported as a
                    // failed build (Failed()), not counted as built.
                    throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
                }
                if (!made && !libraries_only) {
                    library_cache->RecordFallback();
                    FallBackToMonolithic(render_pass);
                }
                // The libraries hold the shaders now (the fallback has used them too).
                for (vk::ShaderModule& module : spv_modules) {
                    module.reset();
                }
            } else {
                MakePipeline(render_pass);
            }
        } catch (const Shader::Exception& exception) {
            if (libraries_only) {
                LOG_DEBUG(Render_Vulkan, "Warm graphics pipeline {:016x}: {}", key.Hash(),
                          exception.what());
            } else {
                LOG_ERROR(Render_Vulkan, "Graphics pipeline {:016x}: {}", key.Hash(),
                          exception.what());
            }
            throw;
        } catch (const vk::Exception& exception) {
            if (libraries_only) {
                // Warm-up guesses the driver refuses: counted by the warm-up, logged at its end.
                LOG_DEBUG(Render_Vulkan, "Warm graphics pipeline {:016x} build failed: {}",
                          key.Hash(), exception.what());
                throw;
            }
            LOG_CRITICAL(Render_Vulkan,
                         "Graphics pipeline build failed: {} (stages: tcs={} tes={} geom={}, "
                         "patch control points {}, device max patch size {})",
                         exception.what(), spv_modules[1] ? 1 : 0, spv_modules[2] ? 1 : 0,
                         spv_modules[3] ? 1 : 0, key.state.patch_control_points_minus_one.Value() + 1,
                         device.GetMaxTessellationPatchSize());
            throw;
        }
        if (pipeline_statistics && pipeline) {
            pipeline_statistics->Collect(device, *pipeline);
        }
        // Nothing reads the shader modules once the pipeline exists, and the driver keeps a copy
        // of each one's SPIR-V while it lives (title 1: ~1,500 pipelines, 2-5 modules each).
        for (vk::ShaderModule& module : spv_modules) {
            module.reset();
        }
        if (worker_only_build) {
            const auto elapsed = std::chrono::steady_clock::now() - begin;
            if (prepared_build) {
                kept_builds.fetch_add(1, std::memory_order::relaxed);
            }
            if (prepared_build && elapsed > KeptBuildLong) {
                kept_builds_long.fetch_add(1, std::memory_order::relaxed);
            }
            if (prepared_build && elapsed > KeptBuildSlow) {
                const auto ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
                kept_builds_slow.fetch_add(1, std::memory_order::relaxed);
                kept_builds_slow_ns.fetch_add(ns, std::memory_order::relaxed);
                auto longest = kept_builds_longest_ns.load(std::memory_order::relaxed);
                while (ns > longest && !kept_builds_longest_ns.compare_exchange_weak(
                                           longest, ns, std::memory_order::relaxed)) {
                }
            }
            static std::atomic<u32> logged{};
            if (elapsed > std::chrono::milliseconds{100} ||
                (elapsed > std::chrono::milliseconds{2} && logged.fetch_add(1) < 400)) {
                LOG_INFO(Render_Vulkan,
                         "Nx2Perf async_pipelines: {:016x} worker emission and driver build {:.1f} ms",
                         key.Hash(), std::chrono::duration<double, std::milli>(elapsed).count());
            }
        }
    }};
    std::array<bool, NUM_STAGES> present_stages{};
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        present_stages[stage] = infos[stage] != nullptr;
    }
    // Mesh pipelines can hold the mesh program in slot 1; the pipeline cache drops pipelines
    // with only one of the two tessellation stages.
    has_tessellation = !IsMeshPipeline() && (present_stages[1] || present_stages[2]);
    configure_func = ConfigureFunc(present_stages, stage_infos);
    configure_keep_func = ConfigureFunc<true>(present_stages, stage_infos);
    spec_stages = SpecStages(present_stages, stage_infos);
    has_texel_buffers = std::ranges::any_of(stage_infos, [](const Shader::Info& info) {
        return !info.texture_buffer_descriptors.empty() || !info.image_buffer_descriptors.empty();
    });
    reads_handles = has_texel_buffers || std::ranges::any_of(stage_infos, [](const Shader::Info& info) {
        return !info.texture_descriptors.empty() || !info.image_descriptors.empty() ||
               !info.handle_programs.empty();
    });
    if (samples_nothing && !IsMeshPipeline()) {
        configure_delta_func = ConfigureFunc<false, true>(present_stages, stage_infos);
    }
    build.emplace(worker_thread, std::move(func));
}

void GraphicsPipeline::AddTransition(GraphicsPipeline* transition) {
    transition_keys.push_back(transition->key);
    transition_hashes.push_back(transition->key_hash);
    transitions.push_back(transition);
}

bool GraphicsPipeline::TextureMemoHit(const TextureCache::TextureMemoState& state,
                                      std::span<const VideoCommon::ImageViewInOut> views,
                                      std::span<const u32> sampler_handles) {
    auto& control = TextureMemoControl::Get();
    ++control.lookups;
    const TextureMemo& memo = texture_memo;
    if (!memo.valid || memo.generation != control.generation ||
        memo.handles.size() != views.size() ||
        memo.sampler_handles.size() != sampler_handles.size()) {
        ++control.miss_no_memo;
        return false;
    }
    if (memo.state != state) {
        ++control.miss_state;
        return false;
    }
    for (size_t index = 0; index < views.size(); ++index) {
        if (views[index].index != memo.handles[index] ||
            sampler_handles[index] != memo.sampler_handles[index]) {
            ++control.miss_handles;
            return false;
        }
    }
    // The guest may have rewritten an entry in place: VisitImageView and GetSamplerId would read
    // it again and resolve it again.
    for (const TextureCache::TextureMemoEntry& entry : memo.entries) {
        if (!texture_cache.TextureMemoEntryCurrent(state, entry)) {
            ++control.miss_entries;
            return false;
        }
    }
    // What preparing the views would act on: CPU-modified contents to upload, aliases to sync.
    for (const VideoCommon::ImageViewId view : memo.views) {
        if (!texture_cache.TextureMemoViewClean(view)) {
            ++control.miss_images;
            return false;
        }
    }
    ++control.hits;
    return true;
}

void GraphicsPipeline::RecordTextureMemo(const TextureCache::TextureMemoState& state,
                                         std::span<const VideoCommon::ImageViewInOut> views,
                                         std::span<const u32> sampler_handles,
                                         std::span<const VideoCommon::SamplerId> samplers,
                                         std::span<const DescriptorUpdateEntry> descriptors,
                                         std::span<const u8> rescaled) {
    auto& control = TextureMemoControl::Get();
    TextureMemo& memo = texture_memo;
    memo.valid = false;
    size_t texture_arrays = 0;
    for (const Shader::Info& info : stage_infos) {
        texture_arrays += info.texture_descriptors.size();
    }
    if (sampler_handles.size() != views.size() || samplers.size() != views.size() ||
        descriptors.size() != views.size() || rescaled.size() != texture_arrays) {
        ++control.refused;
        return;
    }
    // An image made, joined, deleted or rescaled while resolving, or the pools moved: the same
    // state need not give these ids again.
    if (texture_cache.GraphicsTextureMemoState() != state) {
        ++control.refused;
        return;
    }
    memo.entries.resize(views.size());
    for (size_t index = 0; index < views.size(); ++index) {
        if (!texture_cache.TakeTextureMemoEntry(views[index].index, sampler_handles[index],
                                                views[index].id, samplers[index],
                                                memo.entries[index])) {
            ++control.refused;
            return;
        }
    }
    memo.handles.resize(views.size());
    memo.views.resize(views.size());
    for (size_t index = 0; index < views.size(); ++index) {
        memo.handles[index] = views[index].index;
        memo.views[index] = views[index].id;
    }
    memo.sampler_handles.assign(sampler_handles.begin(), sampler_handles.end());
    memo.samplers.assign(samplers.begin(), samplers.end());
    memo.descriptors.assign(descriptors.begin(), descriptors.end());
    memo.rescaled.assign(rescaled.begin(), rescaled.end());
    memo.state = state;
    memo.generation = control.generation;
    memo.touch_frame = texture_cache.FrameTick();
    memo.valid = true;
    ++control.recorded;
}

bool GraphicsPipeline::VerifyTextureMemo(std::span<const VideoCommon::ImageViewInOut> views,
                                         std::span<const VideoCommon::SamplerId> samplers,
                                         std::span<const DescriptorUpdateEntry> descriptors,
                                         std::span<const u8> rescaled, int switch_value) {
    auto& control = TextureMemoControl::Get();
    ++control.verified;
    const TextureMemo& memo = texture_memo;
    constexpr size_t no_index = ~size_t{0};
    size_t binding = no_index;
    size_t bad_array = no_index;
    const bool sizes_match = memo.views.size() == views.size() &&
                             memo.samplers.size() == samplers.size() &&
                             memo.descriptors.size() == descriptors.size() &&
                             memo.rescaled.size() == rescaled.size() &&
                             views.size() == descriptors.size() &&
                             views.size() == samplers.size();
    if (sizes_match) {
        for (size_t index = 0; index < views.size(); ++index) {
            const VkDescriptorImageInfo& kept = memo.descriptors[index].image;
            const VkDescriptorImageInfo& fresh = descriptors[index].image;
            if (memo.views[index] != views[index].id || memo.samplers[index] != samplers[index] ||
                kept.imageView != fresh.imageView || kept.sampler != fresh.sampler ||
                kept.imageLayout != fresh.imageLayout) {
                binding = index;
                break;
            }
        }
        if (binding == no_index) {
            for (size_t index = 0; index < rescaled.size(); ++index) {
                if (memo.rescaled[index] != rescaled[index]) {
                    bad_array = index;
                    break;
                }
            }
        }
        if (binding == no_index && bad_array == no_index) {
            return true;
        }
    }
    ++control.mismatches;
    if (control.mismatches_logged < 32) {
        ++control.mismatches_logged;
        if (!sizes_match) {
            LOG_WARNING(Render_Vulkan,
                        "Texture memo mismatch (EXP32): pipeline {:016x} (vs {:016x} fs {:016x}): "
                        "memo has {} views {} samplers {} descriptors {} arrays, fresh {} {} {} {}",
                        key.Hash(), key.unique_hashes[1], key.unique_hashes[5], memo.views.size(),
                        memo.samplers.size(), memo.descriptors.size(), memo.rescaled.size(),
                        views.size(), samplers.size(), descriptors.size(), rescaled.size());
        } else if (binding != no_index) {
            // The stage the binding belongs to (views are in stage order).
            size_t stage = 0;
            size_t first = 0;
            for (; stage < NUM_STAGES; ++stage) {
                const size_t count = Shader::NumDescriptors(stage_infos[stage].texture_descriptors);
                if (binding < first + count) {
                    break;
                }
                first += count;
            }
            const VkDescriptorImageInfo& kept = memo.descriptors[binding].image;
            const VkDescriptorImageInfo& fresh = descriptors[binding].image;
            LOG_WARNING(Render_Vulkan,
                        "Texture memo mismatch (EXP32): pipeline {:016x} (vs {:016x} fs {:016x}) "
                        "binding {} of {} (stage {}, element {}): handle {:#x} tsc {:#x} | memo "
                        "view {} sampler {} (image view {:#x}, sampler {:#x}, layout {}) | fresh "
                        "view {} sampler {} (image view {:#x}, sampler {:#x}, layout {})",
                        key.Hash(), key.unique_hashes[1], key.unique_hashes[5], binding,
                        views.size(), stage, binding - first, memo.handles[binding],
                        memo.sampler_handles[binding], memo.views[binding].index,
                        memo.samplers[binding].index, HandleBits(kept.imageView),
                        HandleBits(kept.sampler), static_cast<u32>(kept.imageLayout),
                        views[binding].id.index, samplers[binding].index,
                        HandleBits(fresh.imageView), HandleBits(fresh.sampler),
                        static_cast<u32>(fresh.imageLayout));
        } else {
            LOG_WARNING(Render_Vulkan,
                        "Texture memo mismatch (EXP32): pipeline {:016x} (vs {:016x} fs {:016x}) "
                        "texture array {} of {}: rescaled memo {} fresh {}",
                        key.Hash(), key.unique_hashes[1], key.unique_hashes[5], bad_array,
                        rescaled.size(), u32{memo.rescaled[bad_array]}, u32{rescaled[bad_array]});
        }
    }
    if ((switch_value & 4) != 0 && control.disabled_value != switch_value) {
        control.disabled_value = switch_value;
        LOG_WARNING(Render_Vulkan,
                    "Texture memo (EXP32={}) turned off after a mismatch; set EDEN_NX2_EXP32 to "
                    "another value to turn it on again",
                    switch_value);
    }
    return false;
}

void GraphicsPipeline::KeepDrawTemplate(DrawTemplate* verified,
                                        const VideoCommon::DrawTemplateContext& context,
                                        std::span<const u32> handles, u64 hash,
                                        std::span<const VideoCommon::ImageViewInOut> views,
                                        std::span<const VideoCommon::SamplerId> samplers,
                                        const std::array<std::pair<u32, u32>, NUM_STAGES>& ranges,
                                        const RescalingPushConstant& rescaling) {
    DrawTemplateStats& stats = draw_template_stats;
    const DescriptorUpdateEntry* const entries = guest_descriptor_queue.UpdateData();
    if (verified) {
        const auto same_image = [](const DescriptorUpdateEntry& lhs,
                                   const DescriptorUpdateEntry& rhs) {
            return lhs.image.sampler == rhs.image.sampler &&
                   lhs.image.imageView == rhs.image.imageView &&
                   lhs.image.imageLayout == rhs.image.imageLayout;
        };
        const auto& textures = verified->textures;
        bool descriptors_differ = false;
        for (size_t stage = 0; stage < NUM_STAGES && !descriptors_differ; ++stage) {
            const auto [first, count] = verified->ranges[stage];
            const DescriptorUpdateEntry* const resolved = entries + ranges[stage].first;
            descriptors_differ = count != ranges[stage].second ||
                                 !std::equal(resolved, resolved + count,
                                             verified->descriptors.data() + first, same_image);
        }
        const std::array<bool, 4> differ{
            !std::ranges::equal(
                views, textures,
                [](const auto& view, const auto& texture) { return view.id == texture.view; }),
            !std::ranges::equal(samplers, textures,
                                [](const auto& sampler, const auto& texture) {
                                    return sampler == texture.sampler;
                                }),
            descriptors_differ,
            rescaling.Data() != verified->rescaling,
        };
        ++stats.verified;
        if (std::ranges::find(differ, true) != differ.end()) {
            ++stats.mismatched_draws;
            if (stats.total_mismatched_draws++ < 16) {
                LOG_ERROR(Render_Vulkan,
                          "Draw template verify: pipeline {:016x} resolved other {}{}{}{}than "
                          "its template",
                          key.Hash(), differ[0] ? "views " : "", differ[1] ? "samplers " : "",
                          differ[2] ? "descriptors " : "", differ[3] ? "rescaling words " : "");
            }
        }
        for (size_t field = 0; field < differ.size(); ++field) {
            stats.mismatches[field] += differ[field] ? 1 : 0;
        }
    }
    constexpr size_t max_textures = 32;
    if (views.size() > max_textures || std::ranges::any_of(samplers, [&](auto id) {
            return texture_cache.GetSampler(id).HasChangingVariants();
        })) {
        ++stats.not_kept;
        return;
    }
    DrawTemplate& kept = draw_templates.Insert(context, handles, hash);
    if (!texture_cache.KeepTemplateTextures(handles, views, samplers, kept.textures)) {
        DrawTemplates::Erase(kept);
        ++stats.not_kept;
        return;
    }
    kept.descriptors.clear();
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const auto [first, count] = ranges[stage];
        kept.ranges[stage] = {static_cast<u32>(kept.descriptors.size()), count};
        kept.descriptors.insert(kept.descriptors.end(), entries + first, entries + first + count);
    }
    kept.rescaling = rescaling.Data();
    kept.host_serial = gpu_memory->HostPointerSerial();
    kept.scale_epoch = texture_cache.ScaleEpoch();
    ++stats.kept;
}

void GraphicsPipeline::KeepLastDraw(std::span<const VideoCommon::ImageViewInOut> views,
                                    std::span<const VideoCommon::SamplerId> samplers,
                                    const std::array<std::pair<u32, u32>, NUM_STAGES>& ranges,
                                    const RescalingPushConstant& rescaling, bool handles_read) {
    KeptDraw& kept = *kept_draw;
    // ConfigureImpl<Spec, true> filled handles and computed while reading the handles: one handle
    // word per view, unless they came from a memo.
    if (!handles_read || kept.handles.size() != views.size()) {
        kept.valid = false;
        return;
    }
    kept.context = texture_cache.GraphicsTextureContext(template_owner);
    kept.views.assign(views.begin(), views.end());
    kept.samplers.assign(samplers.begin(), samplers.end());
    kept.view_generations.resize(views.size());
    for (size_t i = 0; i < views.size(); ++i) {
        kept.view_generations[i] = texture_cache.ImageViewGeneration(views[i].id);
    }
    const DescriptorUpdateEntry* const entries = guest_descriptor_queue.UpdateData();
    kept.descriptors.clear();
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const auto [first, count] = ranges[stage];
        kept.ranges[stage] = {static_cast<u32>(kept.descriptors.size()), count};
        kept.descriptors.insert(kept.descriptors.end(), entries + first, entries + first + count);
    }
    kept.rescaling = rescaling.Data();
    kept.scale_epoch = texture_cache.ScaleEpoch();
    kept.host_serial = gpu_memory->HostPointerSerial();
    kept.valid = true;
}

bool GraphicsPipeline::ComputedHandlesChanged(const Tegra::Engines::Maxwell3D& maxwell3d_,
                                              Tegra::MemoryManager& gpu_memory_) const {
    for (const KeptDraw::ComputedHandle& handle : kept_draw->computed) {
        const Shader::Info& info{stage_infos[handle.stage]};
        const auto& cbufs{maxwell3d_.state.shader_stages[handle.stage].const_buffers};
        // As ConfigureImpl's read_handle (or its handle table path) computed it.
        const u32 value{VideoCommon::EvaluateTextureHandle(
            info, handle.program, gpu_memory_,
            [&cbufs](u32 cbuf) -> GPUVAddr {
                return cbuf < cbufs.size() && cbufs[cbuf].enabled ? cbufs[cbuf].address
                                                                  : GPUVAddr{};
            },
            handle.alignment, handle.element)};
        if (value != handle.value) {
            return true;
        }
    }
    return false;
}

// SEAM >= 3 (EXP123, vk_mem_stage.cpp): every raw handle this draw binds, in the order
// ConfigureImpl's handle loop reads them, into the draw packet. Called by the early fill, or by
// ConfigureImpl when no fill made them. `maxwell3d` and `gpu_memory` are the draw's (SetEngine).
void GraphicsPipeline::FillHandleValues(MemStage& seam, bool null_pointer_handles,
                                        bool in_configure) {
    seam.FillHandleValues(false, null_pointer_handles, in_configure, [&](std::vector<u32>& values) {
        values.reserve(num_image_elements);
        const auto& stages{maxwell3d->state.shader_stages};
        using Cbufs = std::remove_cvref_t<decltype(stages[0].const_buffers)>;
        VideoCommon::FillGraphicsHandleValues(
            std::span<const Shader::Info>{stage_infos}, *gpu_memory,
            [&stages](size_t stage) {
                return VideoCommon::GraphicsHandleCbufs<Cbufs>{stages[stage].const_buffers};
            },
            device.GetStorageBufferAlignment(), null_pointer_handles,
            [&values](u32 raw) { values.push_back(raw); });
    });
}

bool GraphicsPipeline::TexBankEligible() {
    TexBankMemo& memo = tex_bank_memo;
    if (memo.eligible < 0) [[unlikely]] {
        // Sampled textures only (no texture or image buffers, no images): what ConfigureImpl
        // resolves is then the textures' views and samplers, in stage order.
        bool eligible = num_textures != 0 && num_image_elements == num_textures;
        tex_bank_cbuf_slots.clear();
        const auto add_slot = [&](u32 stage, u32 index) {
            if (index >= Maxwell::MaxConstBuffers) {
                eligible = false;
                return;
            }
            const std::pair<u32, u32> slot{stage, index};
            if (std::ranges::find(tex_bank_cbuf_slots, slot) == tex_bank_cbuf_slots.end()) {
                tex_bank_cbuf_slots.push_back(slot);
            }
        };
        for (size_t stage = 0; eligible && stage < NUM_STAGES; ++stage) {
            for (const auto& desc : stage_infos[stage].texture_descriptors) {
                // Handle programs compute the handle from storage buffers: always the full way.
                if (desc.handle_program != ~0U) {
                    eligible = false;
                    break;
                }
                add_slot(static_cast<u32>(stage), desc.cbuf_index);
                if (desc.has_secondary) {
                    add_slot(static_cast<u32>(stage), desc.secondary_cbuf_index);
                }
            }
        }
        memo.eligible = eligible ? 1 : 0;
    }
    return memo.eligible == 1;
}

bool GraphicsPipeline::TexBankHit(const TextureCache::TextureMemoState& state,
                                  bool via_header_index, bool null_pointer_handles,
                                  int switch_value) {
    auto& control = TexBankControl::Get();
    ++control.lookups;
    TexBankMemo& memo = tex_bank_memo;
    // The epoch first: the page pointers of another epoch are gone.
    if (!memo.valid || memo.switch_value != switch_value ||
        memo.page_epoch != texture_cache.TexBankEpoch()) {
        ++control.miss_no_memo;
        return false;
    }
    if (memo.state != state || memo.via_header_index != via_header_index ||
        memo.null_pointer_handles != null_pointer_handles) {
        ++control.miss_state;
        return false;
    }
    const auto& stages = maxwell3d->state.shader_stages;
    for (const TexBankMemo::Cbuf& cbuf : memo.cbufs) {
        const auto& bound = stages[cbuf.stage].const_buffers[cbuf.index];
        if (bound.enabled != cbuf.enabled || bound.address != cbuf.address ||
            bound.size != cbuf.size) {
            ++control.miss_cbufs;
            return false;
        }
    }
    for (TexBankMemo::Page& page : memo.pages) {
        if (TextureCache::TexBankPageGen(*page.page) != page.gen) {
            ++control.miss_pages;
            return false;
        }
    }
    ++control.hits;
    memo.hit_since_record = true;
    return true;
}

void GraphicsPipeline::RecordTexBank(const TextureCache::TextureMemoState& state,
                                     std::span<const VideoCommon::ImageViewInOut> views,
                                     std::span<const u32> sampler_handles,
                                     std::span<const VideoCommon::SamplerId> samplers,
                                     bool via_header_index, bool null_pointer_handles,
                                     int switch_value) {
    auto& control = TexBankControl::Get();
    TexBankMemo& memo = tex_bank_memo;
    // Backoff: a memo never hit before the next record makes the next records rarer.
    if (memo.countdown != 0) {
        --memo.countdown;
        memo.valid = false;
        ++control.backed_off;
        return;
    }
    if (memo.recorded_once && !memo.hit_since_record) {
        memo.backoff = static_cast<u8>((std::min)(memo.backoff + 1, 4));
    } else {
        memo.backoff = 0;
    }
    memo.countdown = static_cast<u8>((1u << memo.backoff) - 1);
    memo.recorded_once = true;
    memo.hit_since_record = false;
    memo.valid = false;
    const auto refuse = [&control] { ++control.refused; };
    if (sampler_handles.size() != views.size() || samplers.size() != views.size()) {
        return refuse();
    }
    // An image made, joined, deleted or rescaled while resolving, or the pools moved: the same
    // state need not give these ids again.
    if (texture_cache.GraphicsTextureMemoState() != state) {
        return refuse();
    }
    const auto& stages = maxwell3d->state.shader_stages;
    memo.cbufs.clear();
    for (const auto& [stage, index] : tex_bank_cbuf_slots) {
        const auto& bound = stages[stage].const_buffers[index];
        memo.cbufs.push_back(TexBankMemo::Cbuf{
            .stage = stage,
            .index = index,
            .address = bound.address,
            .size = bound.size,
            .enabled = bound.enabled,
        });
    }
    // The GPU pages this draw read: handle words (as read_handle), TIC entries (as
    // VisitImageView: none past the limit) and TSC entries (as GetSamplerId).
    boost::container::small_vector<GPUVAddr, 32> gpu_pages;
    const auto add = [&gpu_pages](GPUVAddr address) {
        const GPUVAddr page = address & ~GPUVAddr{Core::DEVICE_PAGEMASK};
        if (std::ranges::find(gpu_pages, page) == gpu_pages.end()) {
            gpu_pages.push_back(page);
        }
    };
    size_t texture = 0;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        const Shader::Info& info = stage_infos[stage];
        const auto& cbufs = stages[stage].const_buffers;
        for (const auto& desc : info.texture_descriptors) {
            for (u32 index = 0; index < desc.count; ++index, ++texture) {
                if (texture >= views.size()) {
                    return refuse();
                }
                if (!(null_pointer_handles &&
                      VideoCommon::HandleIsStoragePointer(info, desc, index))) {
                    if (!cbufs[desc.cbuf_index].enabled) {
                        return refuse();
                    }
                    add(cbufs[desc.cbuf_index].address + desc.cbuf_offset +
                        (index << desc.size_shift));
                    if (desc.has_secondary) {
                        if (!cbufs[desc.secondary_cbuf_index].enabled) {
                            return refuse();
                        }
                        add(cbufs[desc.secondary_cbuf_index].address +
                            desc.secondary_cbuf_offset);
                    }
                }
                const u32 tic_index = texture_cache.GraphicsTicIndex(views[texture].index);
                if (tic_index <= state.tic_limit) {
                    add(state.tic_pool + u64{tic_index} * sizeof(Tegra::Texture::TICEntry));
                }
                if (sampler_handles[texture] <= state.tsc_limit) {
                    add(state.tsc_pool +
                        u64{sampler_handles[texture]} * sizeof(Tegra::Texture::TSCEntry));
                }
            }
        }
    }
    if (texture != views.size()) {
        return refuse();
    }
    const u64 epoch = texture_cache.TexBankEpoch();
    memo.pages.clear();
    for (const GPUVAddr gpu_page : gpu_pages) {
        const std::optional<DAddr> device_page = texture_cache.TexBankDevicePage(gpu_page);
        if (!device_page) {
            return refuse();
        }
        if (std::ranges::any_of(memo.pages, [&](const TexBankMemo::Page& page) {
                return page.page->page == *device_page;
            })) {
            continue;
        }
        bool created = false;
        TextureCache::TexBankPage* const page = texture_cache.TexBankWatch(*device_page, created);
        if (texture_cache.TexBankEpoch() != epoch) {
            // The watches were dropped while adding this one: the pointers taken are gone.
            return refuse();
        }
        memo.pages.push_back(TexBankMemo::Page{
            .page = page,
            .gen = TextureCache::TexBankPageGen(*page),
        });
        if (created) {
            // Not trapped yet when this draw read it: a CPU write in between marked nothing.
            // The next look moves the generation, so the next draw resolves again, trapped.
            page->recheck = true;
        }
    }
    memo.handles.resize(views.size());
    memo.views.resize(views.size());
    for (size_t index = 0; index < views.size(); ++index) {
        memo.handles[index] = views[index].index;
        memo.views[index] = views[index].id;
    }
    memo.samplers.assign(samplers.begin(), samplers.end());
    memo.state = state;
    memo.page_epoch = epoch;
    memo.switch_value = switch_value;
    memo.via_header_index = via_header_index;
    memo.null_pointer_handles = null_pointer_handles;
    memo.valid = true;
    ++control.recorded;
}

template <typename Spec, bool KeepLast, bool Delta>
bool GraphicsPipeline::ConfigureImpl(bool is_indexed) {
    // EXP118 (Delta, fritchii_delta_path.h): the pipeline samples nothing and the draw follows its
    // last with only light state changed. The descriptor pools' sync, the memo switches (the
    // decision found them off) and the texture steps are left out; with no texture, image or
    // texture buffer they bind nothing. Every other step below runs as for a full draw.
    static_assert(!(KeepLast && Delta), "a delta draw keeps nothing");
    // Delta draws (D2): with KeepLast the handle words and computed handles are kept as they are
    // read, the rest at the end (KeepLastDraw). Without it nothing below changes: every addition
    // is under `if constexpr (KeepLast)`.
    if constexpr (KeepLast) {
        if (!kept_draw) [[unlikely]] {
            kept_draw = std::make_unique<KeptDraw>();
        }
        kept_draw->valid = false;
        kept_draw->handles.clear();
        kept_draw->computed.clear();
    }
    zero_texture_handles = 0;
    // SEAM = 2 (EXP123): the memory stage filled this draw's buffer side before the texture work
    // (MemStage::FillDraw); its calls below are skipped and its sections played in their place.
    MemStage& mem_stage = buffer_cache.runtime.Seam();
    const MemStage::DrawPacket* const held = mem_stage.HeldDraw();
    // FRITCHII_NULL_RENDER (test only, the picture is broken; fable_b_one_follower 7 run 1): the
    // draw's buffer side alone, which the one-follower design gives the engine thread E, in
    // today's order: the uniform state, per stage the storage binds, the handle words (read, not
    // resolved), the texture buffer unbinds, the geometry, the descriptor reservation the binds
    // write into, per stage the buffer binds, the post-copy barrier. Left out, the render thread
    // R's part: the descriptor pools' sync, the image views and samplers (and the memos and
    // templates that keep them), the texture buffer binds (they need the views), the image
    // descriptors and rescaling, render targets, configure draw. PrepareDraw returns after this.
    // Only without a held packet (SEAM 0 and 1, the stage off): PrepareDraw checks.
    if constexpr (!Delta) {
        if (null_render_next) [[unlikely]] {
            null_render_next = false;
            namespace NullProfile = VideoCommon::Nx2Profile;
            buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks,
                                                &uniform_buffer_sizes);
            {
                NullProfile::Scope scope{NullProfile::Zone::ReadHandles};
                for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
                    if (!Spec::enabled_stages[stage]) {
                        continue;
                    }
                    const NullProfile::FineScope storage_scope{
                        NullProfile::Zone::StorageBindings};
                    buffer_cache.UnbindGraphicsStorageBuffers(stage);
                    if constexpr (Spec::has_storage_buffers) {
                        size_t ssbo_index{};
                        for (const auto& desc : stage_infos[stage].storage_buffers_descriptors) {
                            ASSERT(desc.count == 1);
                            buffer_cache.BindGraphicsStorageBuffer(
                                stage, ssbo_index, desc.cbuf_index, desc.cbuf_offset,
                                desc.is_written, desc.indirect_offset);
                            ++ssbo_index;
                        }
                    }
                }
                // The handle words, as the seam's fill reads them (FillHandleValues), kept here.
                null_render_handles.clear();
                const auto& stages{maxwell3d->state.shader_stages};
                using Cbufs = std::remove_cvref_t<decltype(stages[0].const_buffers)>;
                VideoCommon::FillGraphicsHandleValues(
                    std::span<const Shader::Info>{stage_infos}, *gpu_memory,
                    [&stages](size_t stage) {
                        return VideoCommon::GraphicsHandleCbufs<Cbufs>{
                            stages[stage].const_buffers};
                    },
                    device.GetStorageBufferAlignment(),
                    VideoCore::Nx2Debug::Experiment(21) != 0,
                    [this](u32 raw) { null_render_handles.push_back(raw); });
            }
            for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
                if (Spec::enabled_stages[stage]) {
                    buffer_cache.UnbindGraphicsTextureBuffers(stage);
                }
            }
            {
                NullProfile::Scope scope{NullProfile::Zone::UpdateBuffers};
                MemStage::FillPlay seam{mem_stage, MemStage::Site::Geometry};
                buffer_cache.UpdateGraphicsBuffers(is_indexed);
                const NullProfile::FineScope geometry_scope{NullProfile::Zone::GeometryBuffers};
                buffer_cache.BindHostGeometryBuffers(is_indexed);
            }
            {
                NullProfile::Scope scope{NullProfile::Zone::DescriptorAcquire};
                guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries,
                                               uses_descriptor_buffer);
            }
            for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
                if (!Spec::enabled_stages[stage]) {
                    continue;
                }
                NullProfile::Scope scope{NullProfile::Zone::BindStageBuffers};
                MemStage::FillPlay seam{mem_stage, MemStage::Site::Stage,
                                        static_cast<u32>(stage)};
                buffer_cache.BindHostStageBuffers(stage);
            }
            if (buffer_cache.any_buffer_uploaded) {
                buffer_cache.runtime.PostCopyBarrier();
                buffer_cache.any_buffer_uploaded = false;
            }
            return false;
        }
    }
    boost::container::small_vector<VideoCommon::ImageViewInOut, 160> views;
    boost::container::small_vector<VideoCommon::SamplerId, 96> samplers;
    views.reserve(num_image_elements);
    samplers.reserve(num_textures);
    // EDEN_NX2_EXP32: texture binding memo (TextureMemo), for pipelines whose image bindings are
    // all sampled textures. Their sampler handles are kept and resolved after the lookup.
    const int memo_switch = Delta ? 0 : VideoCore::Nx2Debug::Experiment(32);
    const bool texture_memo_on = memo_switch != 0 && num_textures != 0 &&
                                 num_image_elements == num_textures &&
                                 TextureMemoControl::Get().Usable(memo_switch);
    boost::container::small_vector<u32, 96> sampler_handles;

    namespace Profile = VideoCommon::Nx2Profile;
    // EDEN_NX2_EXP7: time each step of every call and log the calls that take over 4 ms (which
    // step holds a draw back in the hitches seen while walking).
    const bool step_trace = VideoCore::Nx2Debug::Experiment(7) != 0;
    std::array<std::chrono::steady_clock::time_point, 10> marks{};
    size_t mark_count = 0;
    const auto mark = [&] {
        if (step_trace) [[unlikely]] {
            marks[mark_count++] = std::chrono::steady_clock::now();
        }
    };
    mark();
    if constexpr (!Delta) {
        Profile::Scope scope{Profile::Zone::SyncDescriptors};
        texture_cache.SynchronizeDescriptors(false);
    }
    mark();

    if (held == nullptr) {
        buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);
    }

    const auto& regs{maxwell3d->regs};
    const bool via_header_index{regs.sampler_binding == Maxwell::SamplerBinding::ViaHeaderBinding};
    // EDEN_NX2_TEX_BANK_GEN (recomp research idea 2): the
    // textures of this pipeline's last draw, while the pages their handle words and TIC/TSC
    // entries were read from kept their generations (TexBankHit). Not with EXP32.
    // Not while CPU inner invalidation is skipped: constant buffer words then mark no page.
    const int tex_bank_switch =
        !Delta && memo_switch == 0 && !Settings::values.skip_cpu_inner_invalidation.GetValue()
            ? VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_tex_bank_gen,
                                               "EDEN_NX2_TEX_BANK_GEN")
            : 0;
    if constexpr (!Delta) {
        if (tex_bank_switch <= 0 && texture_cache.TexBankWatching()) [[unlikely]] {
            texture_cache.TexBankReset();
        }
    }
    // SEAM >= 3 (EXP123, vk_mem_stage.cpp): the raw handles come in the draw packet, with the
    // EXP21 value they were made with. When no early fill made them they are made here, before
    // the bank decision: the fill always runs, a TEX_BANK_GEN hit then does not look at them.
    const std::vector<u32>* handle_values{};
    bool packet_null_pointers{};
    // fable_stage_levelpoints 18.4 row 2 (GPU profile on, the back): the handle reads timed and
    // the draw classed by its handle range since this pipeline's last draw. The range: the words
    // each descriptor reads in its constant buffers (a key of their addresses, the write
    // generations of their pages); a handle program (it reads storage too) makes no key, so its
    // draws count as "another range". Off: the profile's test and a branch.
    const bool price_handles = !Delta && VideoCommon::FritchiiPrice::On() &&
                               VideoCommon::GpuSplit::OnBackThread();
    const auto price_handle_range{[&](u64& sum) -> u64 {
        u64 key = 0xcbf29ce484222325ULL;
        bool keyed = true;
        sum = 0;
        const auto add = [&](GPUVAddr begin, u64 bytes) {
            key = (key ^ begin) * 0x100000001b3ULL;
            key = (key ^ bytes) * 0x100000001b3ULL;
            const std::optional<DAddr> device =
                begin != 0 ? gpu_memory->GpuToCpuAddress(begin) : std::nullopt;
            const u64 part = device ? Common::PriceGenSum(*device, bytes) : ~u64{0};
            sum = part == ~u64{0} || sum == ~u64{0} ? ~u64{0} : sum + part;
        };
        for (size_t stage = 0; stage < NUM_STAGES && keyed; ++stage) {
            if (!Spec::enabled_stages[stage]) {
                continue;
            }
            const Shader::Info& info{stage_infos[stage]};
            const auto& cbufs{maxwell3d->state.shader_stages[stage].const_buffers};
            const auto base = [&cbufs](u32 index) -> GPUVAddr {
                return index < cbufs.size() && cbufs[index].enabled ? cbufs[index].address
                                                                    : GPUVAddr{};
            };
            const auto range = [&](const auto& desc) {
                if (desc.handle_program != ~0U || desc.count == 0) {
                    keyed = desc.count == 0 && keyed;
                    return;
                }
                add(base(desc.cbuf_index) + desc.cbuf_offset,
                    (u64{desc.count - 1} << desc.size_shift) + sizeof(u32));
                if constexpr (requires { desc.has_secondary; }) {
                    if (desc.has_secondary) {
                        add(base(desc.secondary_cbuf_index) + desc.secondary_cbuf_offset,
                            sizeof(u32));
                    }
                }
            };
            for (const auto& desc : info.texture_buffer_descriptors) {
                range(desc);
            }
            for (const auto& desc : info.image_buffer_descriptors) {
                range(desc);
            }
            for (const auto& desc : info.texture_descriptors) {
                range(desc);
            }
            for (const auto& desc : info.image_descriptors) {
                range(desc);
            }
        }
        return keyed ? key : 0;
    }};
    const auto price_handle_note{[&](size_t site, u64 key, u64 sum, u64 start) {
        const auto verdict =
            VideoCommon::FritchiiPrice::ClassifyHandles(key, sum, price_handle_key, price_handle_sum);
        VideoCommon::FritchiiPrice::g_handles[site].Note(static_cast<size_t>(verdict), 1,
                                                         VideoCommon::FritchiiPrice::Ticks() -
                                                             start);
    }};
    if constexpr (!Delta) {
        if (mem_stage.ValuesByValue()) {
            handle_values = mem_stage.HandleValues(false, packet_null_pointers);
            if (handle_values == nullptr) {
                if (mem_stage.HeldFromRing()) [[unlikely]] {
                    // FRITCHII_MEMSTAGE_THREAD = 2: the stage's packet came without them.
                    VideoCommon::MemoryStageLink::Note(
                        VideoCommon::MemoryStageLink::Check::BackHandles,
                        "a served draw's handles read in Configure");
                }
                Profile::Scope scope{Profile::Zone::ReadHandles};
                u64 price_key = 0;
                u64 price_sum = 0;
                u64 price_start = 0;
                if (price_handles) [[unlikely]] {
                    price_key = price_handle_range(price_sum);
                    price_start = VideoCommon::FritchiiPrice::Ticks();
                }
                FillHandleValues(mem_stage, VideoCore::Nx2Debug::Experiment(21) != 0, true);
                if (price_handles) [[unlikely]] {
                    price_handle_note(0, price_key, price_sum, price_start);
                }
                handle_values = mem_stage.HandleValues(false, packet_null_pointers);
            }
        } else if (mem_stage.HeldFromRing()) [[unlikely]] {
            // SEAM below 3 with a ring packet (the ring is ending): handles from game memory.
            VideoCommon::MemoryStageLink::Note(VideoCommon::MemoryStageLink::Check::BackHandles,
                                               "a served draw's handles read below SEAM 3");
        }
    }
    const bool verify_handles{handle_values != nullptr &&
                              (MemStage::VerifyBits() & MemStage::VerifyHandles) != 0};
    size_t handle_cursor{};
    const bool tex_bank_null_pointers{
        !Delta && (handle_values != nullptr ? packet_null_pointers
                                            : VideoCore::Nx2Debug::Experiment(21) != 0)};
    const bool tex_bank_on = tex_bank_switch > 0 && TexBankEligible();
    if (tex_bank_switch > 0 && !tex_bank_on) {
        ++TexBankControl::Get().ineligible;
    }
    bool tex_bank_hit = false;
    bool tex_bank_verify = false;
    TextureCache::TextureMemoState tex_bank_state{};
    boost::container::small_vector<u32, 96> tex_bank_sampler_handles;
    if (tex_bank_on) {
        Profile::Scope scope{Profile::Zone::ReadHandles};
        tex_bank_state = texture_cache.GraphicsTextureMemoState();
        if (TexBankHit(tex_bank_state, via_header_index, tex_bank_null_pointers,
                       tex_bank_switch)) {
            tex_bank_verify = tex_bank_switch == 2 && TexBankControl::Get().ShouldVerify();
            tex_bank_hit = !tex_bank_verify;
        }
    }
    static const bool record_sampled = [] {
        const char* const value = std::getenv("EDEN_NX2_FRAME_DUMP_SAMPLED");
        return value && *value != '\0' && *value != '0';
    }();
    // EXP87 (live; ported): the textures may come from a template an
    // earlier draw kept, also in an earlier frame, keyed on the handle words read below. Not with
    // EXP32 or TEX_BANK_GEN on (their own memos of the textures), for mesh pipelines, or while
    // EDEN_NX2_FRAME_DUMP_SAMPLED records (the original left its memo off for that too). 2 = verify.
    const int template_switch = Delta ? 0 : VideoCore::Nx2Debug::Experiment(87);
    const bool use_template = template_switch != 0 && template_eligible && memo_switch == 0 &&
                              tex_bank_switch <= 0 && !record_sampled && !IsMeshPipeline();
    const bool template_verify = template_switch == 2;
    boost::container::small_vector<u32, 96> template_handles;
    const auto config_stage{[&](size_t stage) LAMBDA_FORCEINLINE {
        const Shader::Info& info{stage_infos[stage]};
        if (held == nullptr) {
            const Profile::FineScope storage_scope{Profile::Zone::StorageBindings};
            buffer_cache.UnbindGraphicsStorageBuffers(stage);
            if constexpr (Spec::has_storage_buffers) {
                size_t ssbo_index{};
                for (const auto& desc : info.storage_buffers_descriptors) {
                    ASSERT(desc.count == 1);
                    buffer_cache.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                           desc.cbuf_offset, desc.is_written,
                                                           desc.indirect_offset);
                    ++ssbo_index;
                }
            }
        }
        if constexpr (Delta) {
            // EXP118: no handle to read.
            return true;
        }
        if (tex_bank_hit) {
            // EDEN_NX2_TEX_BANK_GEN: the textures come from the memo (no handle reads).
            return true;
        }
        const auto& cbufs{maxwell3d->state.shader_stages[stage].const_buffers};
        const auto cbuf_address{[&cbufs](u32 cbuf) -> GPUVAddr {
            return cbuf < cbufs.size() && cbufs[cbuf].enabled ? cbufs[cbuf].address : GPUVAddr{};
        }};
        // The handle words and handle programs: one function for the back's own read and the
        // memory seam's fill (texture_handle_values.h).
        const VideoCommon::GraphicsHandleCbufs<std::remove_cvref_t<decltype(cbufs)>> cbuf_view{
            cbufs};
        const u64 storage_alignment{device.GetStorageBufferAlignment()};
        // Memory stage step 7 (the 5.4 check with the window open): the game memory a handle's
        // own read takes, for MemStage::ValueLeftOut: its word or two, or for a handle program
        // every enabled constant buffer of the stage (its storage words are not known here).
        const auto handle_ranges{[&](const auto& desc, u32 index) {
            boost::container::small_vector<std::pair<u64, u64>, 18> ranges;
            if (desc.handle_program != ~0U) {
                for (const auto& cbuf : cbufs) {
                    if (cbuf.enabled && cbuf.size != 0) {
                        ranges.emplace_back(cbuf.address, cbuf.size);
                    }
                }
                return ranges;
            }
            ranges.emplace_back(cbuf_view.Base(desc.cbuf_index) + desc.cbuf_offset +
                                    (index << desc.size_shift),
                                sizeof(u32));
            if constexpr (requires { desc.has_secondary; }) {
                if (desc.has_secondary) {
                    ranges.emplace_back(cbuf_view.Base(desc.secondary_cbuf_index) +
                                            desc.secondary_cbuf_offset,
                                        sizeof(u32));
                }
            }
            return ranges;
        }};
        // SEAM >= 3: the next value of the draw packet, false past its end (a count mismatch,
        // counted when the loop ends).
        const auto next_value{[&](u32& value) {
            if (handle_cursor < handle_values->size()) {
                value = (*handle_values)[handle_cursor++];
                return true;
            }
            ++handle_cursor;
            return false;
        }};
        const auto raw_handle{[&](const auto& desc, u32 index) -> u32 {
            if (handle_values == nullptr) {
                return VideoCommon::RawTextureHandle<false>(info, desc, index, *gpu_memory,
                                                            cbuf_view, storage_alignment);
            }
            u32 value{};
            if (!next_value(value)) [[unlikely]] {
                return VideoCommon::RawTextureHandle<false>(info, desc, index, *gpu_memory,
                                                            cbuf_view, storage_alignment);
            }
            if (verify_handles) [[unlikely]] {
                // FRITCHII_MEMSTAGE_VERIFY & 1 (5.4): the back's own read.
                const u32 own{VideoCommon::RawTextureHandle<false>(
                    info, desc, index, *gpu_memory, cbuf_view, storage_alignment)};
                // With the window open GPUMem may have written the words since the fill (left
                // out and counted apart); any other difference is a mismatch.
                if (own == value ||
                    !mem_stage.ValueLeftOut(*gpu_memory, handle_ranges(desc, index))) {
                    mem_stage.NoteValueCheck(
                        MemStage::Value::Handle, own == value,
                        VideoCommon::HandleWordAddress(desc, index, cbuf_view),
                        key.unique_hashes[stage + 1], value, own);
                }
            }
            return value;
        }};
        const auto read_handle{[&](const auto& desc, u32 index) {
            const u32 raw{raw_handle(desc, index)};
            if (desc.handle_program != ~0U) {
                // Read by the shader from a storage buffer: computed for this draw.
                // (for an array, for each element: EDEN_NX2_LIGHT_TABLE_ARRAYS)
                // The program's storage offsets are relative to the buffer address aligned down
                // to the host storage alignment (StorageOffset); EDEN_NX2_HANDLE_ALIGN=0 reads
                // the words at the unaligned address instead (A/B, HandleProgramAlignment).
                if constexpr (KeepLast) {
                    kept_draw->handles.push_back(raw);
                    kept_draw->computed.push_back({
                        .stage = static_cast<u32>(stage),
                        .program = desc.handle_program,
                        .element = index,
                        .value = raw,
                        .alignment = VideoCommon::HandleProgramAlignment(storage_alignment),
                    });
                }
                if (raw == 0) {
                    ++zero_texture_handles;
                    static std::set<std::pair<u64, u32>> logged;
                    const std::pair key_program{key.unique_hashes[stage + 1], desc.handle_program};
                    if (logged.size() < 64 && logged.insert(key_program).second) {
                        const auto evaluate{[&](u64 alignment) {
                            return VideoCommon::EvaluateTextureHandle(info, desc.handle_program,
                                                                      *gpu_memory, cbuf_address,
                                                                      alignment, index);
                        }};
                        LOG_INFO(Render_Vulkan,
                                 "Texture handle program {} of stage {} (shader {:016x}) gave 0; "
                                 "aligned to {} it gives {:#x}, unaligned {:#x}",
                                 desc.handle_program, stage, key_program.first, storage_alignment,
                                 evaluate(storage_alignment), evaluate(1));
                    }
                }
                return TexturePair(raw, via_header_index);
            }
            if constexpr (KeepLast) {
                kept_draw->handles.push_back(raw);
            }
            return TexturePair(raw, via_header_index);
        }};
        const auto add_image{[&](const auto& desc, bool blacklist) LAMBDA_FORCEINLINE {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle{read_handle(desc, index)};
                views.push_back({
                    .index = handle.first,
                    .blacklist = blacklist,
                    .id = {}
                });
            }
        }};
        if constexpr (Spec::has_texture_buffers) {
            for (const auto& desc : info.texture_buffer_descriptors) {
                add_image(desc, false);
            }
        }
        if constexpr (Spec::has_image_buffers) {
            for (const auto& desc : info.image_buffer_descriptors) {
                add_image(desc, false);
            }
        }
        // EDEN_NX2_EXP21 (default 1; 0 = off): no texture for a handle that is a buffer address.
        // Read once per draw (TEX_BANK_GEN keys its memo on the same value).
        const bool null_pointer_handles{tex_bank_null_pointers};
        u32 texture{};
        for (const auto& desc : info.texture_descriptors) {
            if (desc.count > 1 && desc.handle_program != ~0U) {
                // EDEN_NX2_LIGHT_TABLE_ARRAYS: a table of handles the
                // shader indexes (one shadow map per light, say), read in one go; elements
                // repeating a sampler share its lookup. Only translated with the switch on.
                // Handle programs keep a pipeline out of EXP87's templates and TEX_BANK_GEN
                // (template_eligible, TexBankEligible); EXP32 resolves the samplers after its
                // lookup, as for every other texture.
                const size_t first{views.size()};
                u32 last_sampler_index{~0U};
                VideoCommon::SamplerId sampler{};
                const auto table_element{[&]([[maybe_unused]] u32 element, u32 raw) {
                    if constexpr (KeepLast) {
                        kept_draw->handles.push_back(raw);
                        kept_draw->computed.push_back({
                            .stage = static_cast<u32>(stage),
                            .program = desc.handle_program,
                            .element = element,
                            .value = raw,
                            .alignment = device.GetStorageBufferAlignment(),
                        });
                    }
                    const auto handle{TexturePair(raw, via_header_index)};
                    views.push_back({handle.first});
                    if (tex_bank_on) {
                        tex_bank_sampler_handles.push_back(handle.second);
                    }
                    if (texture_memo_on || use_template) {
                        sampler_handles.push_back(handle.second);
                    } else {
                        if (handle.second != last_sampler_index) {
                            sampler = texture_cache.GetSamplerId(handle.second, false);
                            last_sampler_index = handle.second;
                        }
                        samplers.push_back(sampler);
                    }
                    if (use_template) {
                        template_handles.push_back(via_header_index
                                                       ? handle.first
                                                       : handle.first | (handle.second << 20));
                    }
                }};
                if (handle_values == nullptr) {
                    VideoCommon::RawTextureHandleTable(info, desc, *gpu_memory, cbuf_view,
                                                       storage_alignment, table_element);
                } else {
                    // SEAM >= 3: the table's values from the draw packet.
                    boost::container::small_vector<u32, 16> own;
                    if (verify_handles || handle_cursor + desc.count > handle_values->size())
                        [[unlikely]] {
                        VideoCommon::RawTextureHandleTable(
                            info, desc, *gpu_memory, cbuf_view, storage_alignment,
                            [&own](u32, u32 raw) { own.push_back(raw); });
                    }
                    for (u32 element = 0; element < desc.count; ++element) {
                        u32 value{};
                        if (!next_value(value)) [[unlikely]] {
                            value = own[element];
                        } else if (verify_handles) [[unlikely]] {
                            if (own[element] == value ||
                                !mem_stage.ValueLeftOut(*gpu_memory,
                                                        handle_ranges(desc, element))) {
                                mem_stage.NoteValueCheck(
                                    MemStage::Value::Handle, own[element] == value,
                                    VideoCommon::HandleWordAddress(desc, element, cbuf_view),
                                    key.unique_hashes[stage + 1], value, own[element]);
                            }
                        }
                        table_element(element, value);
                    }
                }
                if (VideoCommon::HandleTableNotes::Wanted()) {
                    std::vector<u32> tics(desc.count);
                    for (u32 index = 0; index < desc.count; ++index) {
                        tics[index] = views[first + index].index;
                    }
                    VideoCommon::HandleTableNotes::Note(key.unique_hashes[stage + 1], texture,
                                                        tics);
                }
                ++texture;
                continue;
            }
            for (u32 index = 0; index < desc.count; ++index) {
                const bool null_handle{null_pointer_handles &&
                                       VideoCommon::HandleIsStoragePointer(info, desc, index)};
                if (null_handle && handle_values != nullptr) {
                    // SEAM >= 3: the packet keeps its place (SKIPPED_HANDLE_VALUE).
                    ++handle_cursor;
                }
                const auto handle{null_handle ? VideoCommon::NULL_TEXTURE_HANDLE
                                              : read_handle(desc, index)};
                if constexpr (KeepLast) {
                    // EXP21's null handle read no word: keep its place (one word per view).
                    if (kept_draw->handles.size() == views.size()) {
                        kept_draw->handles.push_back(~0U);
                    }
                }
                // EDEN_NX2_HANDLE_TRACE=<shader hash>: each new handle of that shader's textures.
                static const u64 trace_hash{[] {
                    const char* const value{std::getenv("EDEN_NX2_HANDLE_TRACE")};
                    return value ? std::strtoull(value, nullptr, 16) : u64{0};
                }()};
                if (trace_hash != 0 && key.unique_hashes[stage + 1] == trace_hash) {
                    static std::set<std::tuple<size_t, u32, u32>> traced;
                    const size_t slot = static_cast<size_t>(&desc - info.texture_descriptors.data());
                    if (traced.size() < 512 && traced.emplace(slot, index, handle.first).second) {
                        std::string program;
                        if (desc.handle_program != ~0U &&
                            desc.handle_program < info.handle_programs.size()) {
                            for (const u32 word : info.handle_programs[desc.handle_program]) {
                                program += fmt::format(" {:x}", word);
                            }
                        }
                        LOG_INFO(Render_Vulkan,
                                 "Handle trace {:016x} stage {} texture {}[{}]: cbuf {} offset {:#x} "
                                 "shift {} secondary {} ({} {:#x} shift {}) size_shift {} program {}:{} "
                                 "-> tic {} tsc {} (c{}[{:#x}] = {:#x})",
                                 trace_hash, stage, slot, index, desc.cbuf_index, desc.cbuf_offset,
                                 desc.shift_left, desc.has_secondary, desc.secondary_cbuf_index,
                                 desc.secondary_cbuf_offset, desc.secondary_shift_left,
                                 desc.size_shift, desc.handle_program, program, handle.first,
                                 handle.second, desc.cbuf_index, desc.cbuf_offset,
                                 desc.cbuf_index < cbufs.size() && cbufs[desc.cbuf_index].enabled
                                     ? gpu_memory->ReadHandleWord(
                                           cbufs[desc.cbuf_index].address + desc.cbuf_offset)
                                     : 0U);
                    }
                }
                views.push_back({handle.first});
                if (tex_bank_on) {
                    tex_bank_sampler_handles.push_back(handle.second);
                }

                if (texture_memo_on || use_template) {
                    // EXP87: resolved after the template lookup, unless a template gives them.
                    sampler_handles.push_back(handle.second);
                } else {
                    VideoCommon::SamplerId sampler{
                        texture_cache.GetSamplerId(handle.second, false)};
                    samplers.push_back(sampler);
                }
                if (use_template) {
                    template_handles.push_back(via_header_index
                                                   ? handle.first
                                                   : handle.first | (handle.second << 20));
                }
            }
            ++texture;
        }
        if constexpr (Spec::has_images) {
            for (const auto& desc : info.image_descriptors) {
                add_image(desc, desc.is_written);
            }
        }

        return true;
    }};
    {
        Profile::Scope scope{Profile::Zone::ReadHandles};
        // Row 2 at EXP123 below 3: the loop reads the words itself (with the storage binds and
        // the samplers' and views' indices, all in this time).
        const bool price_loop = price_handles && handle_values == nullptr && !tex_bank_hit;
        u64 price_key = 0;
        u64 price_sum = 0;
        u64 price_start = 0;
        if (price_loop) [[unlikely]] {
            price_key = price_handle_range(price_sum);
            price_start = VideoCommon::FritchiiPrice::Ticks();
        }
        SCOPE_EXIT {
            if (price_loop) [[unlikely]] {
                price_handle_note(1, price_key, price_sum, price_start);
            }
        };
        if constexpr (Spec::enabled_stages[0]) {
            config_stage(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            config_stage(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            config_stage(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            config_stage(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            config_stage(4);
        }
        if (handle_values != nullptr) {
            // None taken: a bank hit skipped the loop. Else all must have been.
            mem_stage.LookedAtHandleValues(false, handle_cursor);
        }
        if (tex_bank_hit) {
            const TexBankMemo& memo = tex_bank_memo;
            for (const u32 handle : memo.handles) {
                views.push_back({.index = handle, .blacklist = false, .id = {}});
            }
            samplers.assign(memo.samplers.begin(), memo.samplers.end());
        }
    }
    Profile::AddFine(Profile::FineCounter::TextureHandles, views.size());
    ASSERT(views.size() == num_image_elements);
    mark();
    // EXP87: a template of this pipeline with the same context and handle words, still good
    // (Check), gives the views and samplers here and the descriptors in prepare_stage.
    DrawTemplate* draw_template{};
    VideoCommon::DrawTemplateContext template_context{};
    const std::span<const u32> template_handles_span{template_handles.data(),
                                                     template_handles.size()};
    u64 template_hash{};
    bool template_hit{};
    if (use_template) {
        Profile::Scope scope{Profile::Zone::ReadHandles};
        template_context = texture_cache.GraphicsTextureContext(template_owner);
        template_hash = VideoCommon::HashDrawTemplate(template_context, template_handles_span);
        draw_template =
            draw_templates.Find(template_context, template_handles_span, template_hash);
        if (draw_template) {
            const VideoCommon::DrawTemplateCheck check =
                draw_template->Check(gpu_memory->HostPointerSerial(), texture_cache.ScaleEpoch(),
                                     [this](VideoCommon::ImageViewId id) {
                                         return texture_cache.ImageViewGeneration(id);
                                     });
            if (check != VideoCommon::DrawTemplateCheck::Hit) {
                ++draw_template_stats.rejected[static_cast<size_t>(check)];
                DrawTemplates::Erase(*draw_template);
                draw_template = nullptr;
            }
        }
        template_hit = draw_template != nullptr && !template_verify;
        if (template_hit) {
            for (size_t i = 0; i < views.size(); ++i) {
                views[i].id = draw_template->textures[i].view;
                samplers.push_back(draw_template->textures[i].sampler);
            }
            if (!texture_cache.PrepareTemplateViews(std::span(views.data(), views.size()))) {
                ++draw_template_stats.prepare_deleted;
                DrawTemplates::Erase(*draw_template);
                draw_template = nullptr;
                template_hit = false;
                samplers.clear();
            }
        }
        if (template_hit) {
            ++draw_template_stats.hits;
            template_misses = 0;
        } else if (!draw_template) {
            ++draw_template_stats.misses;
            ++template_misses;
        }
        if (!template_hit) {
            for (const u32 sampler_handle : sampler_handles) {
                samplers.push_back(texture_cache.GetSamplerId(sampler_handle, false));
            }
        }
    }
    // EXP32: a hit takes the ids (here) and the descriptors (prepare_stage) of the pipeline's
    // previous draw; a hit picked for verification resolves and pushes the full way, then
    // compares (VerifyTextureMemo). Every other draw with the memo on records it.
    bool memo_hit = false;
    bool memo_verify = false;
    TextureCache::TextureMemoState memo_state{};
    if constexpr (!Delta) {
        Profile::Scope scope{Profile::Zone::FillImageViews};
        if (texture_memo_on) {
            memo_state = texture_cache.GraphicsTextureMemoState();
            if (TextureMemoHit(memo_state, std::span(views.data(), views.size()),
                               std::span(sampler_handles.data(), sampler_handles.size()))) {
                memo_verify = TextureMemoControl::Get().ShouldVerify(memo_switch);
                memo_hit = !memo_verify;
            }
            if (memo_hit) {
                for (size_t index = 0; index < views.size(); ++index) {
                    views[index].id = texture_memo.views[index];
                }
                samplers.assign(texture_memo.samplers.begin(), texture_memo.samplers.end());
                // Preparing the views touches their images in the LRU; again in a frame is a
                // no-op.
                const u64 frame = texture_cache.FrameTick();
                if (texture_memo.touch_frame != frame) {
                    texture_memo.touch_frame = frame;
                    for (const VideoCommon::ImageViewId view_id : texture_memo.views) {
                        texture_cache.TouchTextureMemoView(view_id);
                    }
                }
            } else {
                for (const u32 sampler_handle : sampler_handles) {
                    samplers.push_back(texture_cache.GetSamplerId(sampler_handle, false));
                }
                texture_cache.FillImageViews(std::span(views.data(), views.size()), false,
                                             Spec::has_images);
            }
        } else if (tex_bank_hit) {
            // EDEN_NX2_TEX_BANK_GEN: the memo's views, prepared as VisitImageView prepares a
            // table entry it read before.
            const TexBankMemo& memo = tex_bank_memo;
            for (size_t index = 0; index < views.size(); ++index) {
                views[index].id = memo.views[index];
            }
            if (!texture_cache.PrepareTexBankViews(memo.views)) {
                // Preparing deleted images: resolve the views the full way (the handles and
                // samplers stand; samplers are never deleted).
                ++TexBankControl::Get().prepare_fallbacks;
                texture_cache.FillImageViews(std::span(views.data(), views.size()), false,
                                             Spec::has_images);
            }
        } else if (!template_hit) {
            // A template hit (EXP87) took its views and prepared them above.
            texture_cache.FillImageViews(std::span(views.data(), views.size()), false,
                                         Spec::has_images);
        }
        if (tex_bank_on && !tex_bank_hit) {
            const std::span<const VideoCommon::ImageViewInOut> views_span{views.data(),
                                                                          views.size()};
            if (tex_bank_verify) {
                // EDEN_NX2_TEX_BANK_GEN=2: a hit resolved the full way; the memo must agree.
                auto& control = TexBankControl::Get();
                ++control.verified;
                const TexBankMemo& memo = tex_bank_memo;
                size_t bad = views.size();
                if (memo.views.size() == views.size() && memo.handles.size() == views.size() &&
                    memo.samplers.size() == samplers.size() && samplers.size() == views.size()) {
                    for (size_t index = 0; index < views.size(); ++index) {
                        if (memo.handles[index] != views[index].index ||
                            memo.views[index] != views[index].id ||
                            memo.samplers[index] != samplers[index]) {
                            bad = index;
                            break;
                        }
                    }
                } else {
                    bad = 0;
                }
                if (bad != views.size()) {
                    ++control.mismatches;
                    if (control.mismatches_logged < 16) {
                        ++control.mismatches_logged;
                        const bool sized = bad < memo.views.size() && bad < memo.handles.size() &&
                                           bad < memo.samplers.size();
                        LOG_WARNING(
                            Render_Vulkan,
                            "Texture bank memo mismatch (TEX_BANK_GEN=2): pipeline {:016x} (vs "
                            "{:016x} fs {:016x}) texture {} of {}: memo handle {:#x} view {} "
                            "sampler {} | fresh handle {:#x} view {} sampler {}",
                            key.Hash(), key.unique_hashes[1], key.unique_hashes[5], bad,
                            views.size(), sized ? memo.handles[bad] : 0U,
                            sized ? memo.views[bad].index : 0U,
                            sized ? memo.samplers[bad].index : 0U, views[bad].index,
                            views[bad].id.index, bad < samplers.size() ? samplers[bad].index : 0U);
                    }
                }
            }
            RecordTexBank(tex_bank_state, views_span,
                          std::span<const u32>(tex_bank_sampler_handles.data(),
                                               tex_bank_sampler_handles.size()),
                          std::span<const VideoCommon::SamplerId>(samplers.data(),
                                                                  samplers.size()),
                          via_header_index, tex_bank_null_pointers, tex_bank_switch);
        }
    }
    ASSERT(samplers.size() == num_textures);
    mark();
    if (DrawReuseProbe::Enabled()) [[unlikely]] {
        u64 textures = 0;
        for (const auto& view : views) {
            textures = DrawReuseProbe::Mix(textures, view.index);
        }
        for (const auto sampler : samplers) {
            textures = DrawReuseProbe::Mix(textures, sampler.index);
        }
        u64 full = DrawReuseProbe::Mix(textures, reinterpret_cast<u64>(this));
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
            for (const auto& cbuf : maxwell3d->state.shader_stages[stage].const_buffers) {
                full = DrawReuseProbe::Mix(full, cbuf.enabled ? cbuf.address ^ (u64{cbuf.size} << 48) : 0);
            }
        }
        full = DrawReuseProbe::Bytes(full, &regs.vertex_streams, sizeof(regs.vertex_streams));
        full = DrawReuseProbe::Bytes(full, &regs.rt, sizeof(regs.rt));
        u64 no_cbufs = DrawReuseProbe::Mix(textures, reinterpret_cast<u64>(this));
        no_cbufs = DrawReuseProbe::Bytes(no_cbufs, &regs.vertex_streams, sizeof(regs.vertex_streams));
        no_cbufs = DrawReuseProbe::Bytes(no_cbufs, &regs.rt, sizeof(regs.rt));
        DrawReuseProbe::Get().Note(this, full, textures, no_cbufs);
    }
    if (record_sampled) {
        sampled_images.clear();
        size_t view_index{};
        size_t sampler_index{};
        for (size_t stage = 0; stage < Spec::enabled_stages.size(); ++stage) {
            if (!Spec::enabled_stages[stage]) {
                continue;
            }
            const Shader::Info& info{stage_infos[stage]};
            if constexpr (Spec::has_texture_buffers) {
                view_index += Shader::NumDescriptors(info.texture_buffer_descriptors);
            }
            if constexpr (Spec::has_image_buffers) {
                view_index += Shader::NumDescriptors(info.image_buffer_descriptors);
            }
            const u32 textures = Shader::NumDescriptors(info.texture_descriptors);
            for (u32 texture = 0; texture < textures; ++texture, ++view_index, ++sampler_index) {
                sampled_images.push_back({static_cast<u32>(stage), views[view_index].index,
                                          views[view_index].id, samplers[sampler_index]});
            }
            if constexpr (Spec::has_images) {
                view_index += Shader::NumDescriptors(info.image_descriptors);
            }
        }
    }
    if (texture_cache.DepthCensusOn()) [[unlikely]] {
        // EDEN_NX2_DEPTH_CENSUS: which of the D32 targets made this frame this draw samples,
        // with the sampler and how the shader reads it (views laid out as in record_sampled).
        size_t view_index{};
        size_t sampler_index{};
        for (size_t stage = 0; stage < Spec::enabled_stages.size(); ++stage) {
            if (!Spec::enabled_stages[stage]) {
                continue;
            }
            const Shader::Info& info{stage_infos[stage]};
            if constexpr (Spec::has_texture_buffers) {
                view_index += Shader::NumDescriptors(info.texture_buffer_descriptors);
            }
            if constexpr (Spec::has_image_buffers) {
                view_index += Shader::NumDescriptors(info.image_buffer_descriptors);
            }
            for (const auto& desc : info.texture_descriptors) {
                for (u32 i = 0; i < desc.count; ++i, ++view_index, ++sampler_index) {
                    if (view_index >= views.size()) {
                        break;
                    }
                    const bool has_sampler = sampler_index < samplers.size() &&
                                             samplers[sampler_index] != VideoCommon::SamplerId{};
                    texture_cache.CensusNoteSampled(
                        views[view_index].id,
                        has_sampler ? &texture_cache.GetSampler(samplers[sampler_index]).Descriptor()
                                    : nullptr,
                        desc.census_use, desc.is_depth);
                }
            }
            if constexpr (Spec::has_images) {
                view_index += Shader::NumDescriptors(info.image_descriptors);
            }
        }
    }

    if constexpr (Spec::has_texture_buffers || Spec::has_image_buffers) {
        VideoCommon::ImageViewInOut* texture_buffer_it{views.data()};
        const auto bind_stage_info{[&](size_t stage) LAMBDA_FORCEINLINE {
            if (held != nullptr) {
                // A held draw has no texel buffers (else it fills here, LateFill): the fill
                // unbound them.
                return;
            }
            size_t index{};
            const auto add_buffer{[&](const auto& desc) {
                constexpr bool is_image = std::is_same_v<decltype(desc), const ImageBufferDescriptor&>;
                for (u32 i = 0; i < desc.count; ++i) {
                    bool is_written{false};
                    if constexpr (is_image) {
                        is_written = desc.is_written;
                    }
                    ImageView& image_view{texture_cache.GetImageView(texture_buffer_it->id)};
                    PixelFormat format{image_view.format};
                    if constexpr (is_image) {
                        if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                            format = *explicit_format;
                        }
                    }
                    buffer_cache.BindGraphicsTextureBuffer(stage, index, image_view.GpuAddr(),
                                                           image_view.BufferSize(), format,
                                                           is_written, is_image);
                    ++index;
                    ++texture_buffer_it;
                }
            }};
            buffer_cache.UnbindGraphicsTextureBuffers(stage);

            const Shader::Info& info{stage_infos[stage]};
            if constexpr (Spec::has_texture_buffers) {
                for (const auto& desc : info.texture_buffer_descriptors) {
                    add_buffer(desc);
                }
            }
            if constexpr (Spec::has_image_buffers) {
                for (const auto& desc : info.image_buffer_descriptors) {
                    add_buffer(desc);
                }
            }
            texture_buffer_it += Shader::NumDescriptors(info.texture_descriptors);
            if constexpr (Spec::has_images) {
                texture_buffer_it += Shader::NumDescriptors(info.image_descriptors);
            }
        }};
        Profile::Scope scope{Profile::Zone::TextureBuffers};
        if constexpr (Spec::enabled_stages[0]) {
            bind_stage_info(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            bind_stage_info(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            bind_stage_info(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            bind_stage_info(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            bind_stage_info(4);
        }
    } else if (held == nullptr) {
        if constexpr (Spec::enabled_stages[0]) {
            buffer_cache.UnbindGraphicsTextureBuffers(0);
        }
        if constexpr (Spec::enabled_stages[1]) {
            buffer_cache.UnbindGraphicsTextureBuffers(1);
        }
        if constexpr (Spec::enabled_stages[2]) {
            buffer_cache.UnbindGraphicsTextureBuffers(2);
        }
        if constexpr (Spec::enabled_stages[3]) {
            buffer_cache.UnbindGraphicsTextureBuffers(3);
        }
        if constexpr (Spec::enabled_stages[4]) {
            buffer_cache.UnbindGraphicsTextureBuffers(4);
        }
    }

    if (regs.transform_feedback_enabled != 0) {
        scheduler.RequestOutsideRenderPassOperationContext();
    }
    mark();
    {
        Profile::Scope scope{Profile::Zone::UpdateBuffers};
        if (held != nullptr) {
            // SEAM = 2: the geometry section filled before the texture work, played here.
            mem_stage.PlayHeld(held->geometry);
        } else {
            // The geometry section: filled here, played as the bracket ends.
            MemStage::FillPlay seam{mem_stage, MemStage::Site::Geometry};
            buffer_cache.UpdateGraphicsBuffers(is_indexed);
            const Profile::FineScope geometry_scope{Profile::Zone::GeometryBuffers};
            buffer_cache.BindHostGeometryBuffers(is_indexed);
        }
    }
    mark();
    {
        Profile::Scope scope{Profile::Zone::DescriptorAcquire};
        guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);
    }
    mark();

    RescalingPushConstant rescaling;
    RenderAreaPushConstant render_area;
    const VideoCommon::SamplerId* samplers_it{samplers.data()};
    const VideoCommon::ImageViewInOut* views_it{views.data()};
    // EXP32: what a draw that did not hit pushes, to be recorded (or compared, when verifying).
    boost::container::small_vector<DescriptorUpdateEntry, 32> fresh_descriptors;
    std::array<std::pair<u32, u32>, NUM_STAGES> template_ranges{}; ///< EXP87, per stage
    boost::container::small_vector<u8, 32> fresh_rescaled;
    u8* fresh_rescaled_cursor{};
    size_t memo_descriptor_index{};
    size_t memo_rescaled_index{};
    if (texture_memo_on && !memo_hit) {
        fresh_descriptors.reserve(num_textures);
        size_t texture_arrays{};
        for (const Shader::Info& info : stage_infos) {
            texture_arrays += info.texture_descriptors.size();
        }
        fresh_rescaled.resize(texture_arrays);
        fresh_rescaled_cursor = fresh_rescaled.data();
    }
    const auto prepare_stage{[&](size_t stage) LAMBDA_FORCEINLINE {
        // FRITCHII_DELAY_BACK_KIND=2 ("S", fable_stage_levelpoints 18.3): every 4th stage bind,
        // on the back thread, outside the zone. Off: one relaxed load and a branch.
        VideoCommon::GpuSplit::DelayBackIf(VideoCommon::GpuSplit::BackDelayKind::StageBinds);
        {
            Profile::Scope scope{Profile::Zone::BindStageBuffers};
            if (held != nullptr) {
                // SEAM = 2: the stage's section, filled before the texture work.
                mem_stage.PlayHeld(held->stage[stage]);
            } else {
                // The stage's buffer entries and descriptors, played before its image
                // descriptors.
                MemStage::FillPlay seam{mem_stage, MemStage::Site::Stage, static_cast<u32>(stage)};
                buffer_cache.BindHostStageBuffers(stage);
            }
        }
        {
            Profile::Scope scope{Profile::Zone::PushDescriptors};
            if (library_cache && stage == NUM_STAGES - 1) {
                // Libraries: the fragment stage's rescaling bits start at fixed indices.
                rescaling.Seek(GPL_FRAGMENT_TEXTURE_SCALING_BASE, GPL_FRAGMENT_IMAGE_SCALING_BASE);
            }
            // EXP87's ranges are also the kept draw's (D2: KeepLast).
            const size_t template_first{
                use_template || KeepLast ? guest_descriptor_queue.UpdateCount() : 0};
            if (memo_hit) {
                // The stage's textures as the pipeline's previous draw pushed them (only sampled
                // textures: no image descriptors to mark modified).
                const Shader::Info& memo_info{stage_infos[stage]};
                const u32 count{Shader::NumDescriptors(memo_info.texture_descriptors)};
                guest_descriptor_queue.AddEntries(
                    texture_memo.descriptors.data() + memo_descriptor_index, count);
                memo_descriptor_index += count;
                for (size_t texture_array = 0;
                     texture_array < memo_info.texture_descriptors.size(); ++texture_array) {
                    rescaling.PushTexture(texture_memo.rescaled[memo_rescaled_index++] != 0);
                }
                samplers_it += count;
                views_it += count;
            } else if (texture_memo_on) {
                const size_t first{guest_descriptor_queue.UpdateCount()};
                PushImageDescriptors(texture_cache, guest_descriptor_queue, stage_infos[stage],
                                     rescaling, samplers_it, views_it, &fresh_rescaled_cursor);
                const DescriptorUpdateEntry* const pushed{guest_descriptor_queue.UpdateData()};
                fresh_descriptors.insert(fresh_descriptors.end(), pushed + first,
                                         pushed + guest_descriptor_queue.UpdateCount());
            } else if (template_hit) {
                // EXP87: the stage's descriptors as the template's draw pushed them.
                const auto [first, count] = draw_template->ranges[stage];
                guest_descriptor_queue.AddEntries(draw_template->descriptors.data() + first,
                                                  count);
            } else {
                PushImageDescriptors(texture_cache, guest_descriptor_queue, stage_infos[stage],
                                     rescaling, samplers_it, views_it);
            }
            if (use_template || KeepLast) {
                template_ranges[stage] = {
                    static_cast<u32>(template_first),
                    static_cast<u32>(guest_descriptor_queue.UpdateCount() - template_first)};
            }
        }
        const auto& info{stage_infos[stage]};
        if (info.uses_render_area) {
            render_area.uses_render_area = true;
            // Switch 2 titles may never program the surface clip; use colour target 0 then.
            const u32 clip_width = regs.surface_clip.width, clip_height = regs.surface_clip.height;
            const bool clip_set = clip_width != 0 && clip_height != 0;
            render_area.words = {static_cast<float>(clip_set ? clip_width : u32{regs.rt[0].width}),
                                 static_cast<float>(clip_set ? clip_height : u32{regs.rt[0].height})};
        }
    }};
    if constexpr (Spec::enabled_stages[0]) {
        prepare_stage(0);
    }
    if constexpr (Spec::enabled_stages[1]) {
        prepare_stage(1);
    }
    if constexpr (Spec::enabled_stages[2]) {
        prepare_stage(2);
    }
    if constexpr (Spec::enabled_stages[3]) {
        prepare_stage(3);
    }
    if constexpr (Spec::enabled_stages[4]) {
        prepare_stage(4);
    }
    if (texture_memo_on && !memo_hit) {
        const size_t rescaled_count{
            static_cast<size_t>(fresh_rescaled_cursor - fresh_rescaled.data())};
        const std::span<const u8> rescaled_span{fresh_rescaled.data(), rescaled_count};
        const std::span<const VideoCommon::ImageViewInOut> views_span{views.data(), views.size()};
        const std::span<const VideoCommon::SamplerId> samplers_span{samplers.data(),
                                                                    samplers.size()};
        const std::span<const DescriptorUpdateEntry> descriptors_span{fresh_descriptors.data(),
                                                                      fresh_descriptors.size()};
        if (memo_verify) {
            static_cast<void>(VerifyTextureMemo(views_span, samplers_span, descriptors_span,
                                                rescaled_span, memo_switch));
        }
        RecordTextureMemo(memo_state, views_span,
                          std::span<const u32>(sampler_handles.data(), sampler_handles.size()),
                          samplers_span, descriptors_span, rescaled_span);
    }
    if (template_hit) {
        rescaling.Assign(draw_template->rescaling);
    } else if (use_template &&
               (draw_template != nullptr || template_misses <= 16 || (template_misses & 15) == 0)) {
        // A pipeline whose draws never repeat keeps one template in 16.
        KeepDrawTemplate(draw_template, template_context, template_handles_span, template_hash,
                         std::span(views.data(), views.size()),
                         std::span(samplers.data(), samplers.size()), template_ranges, rescaling);
    }
    if (use_template) {
        ReportDrawTemplates(template_switch);
    }
    if constexpr (KeepLast) {
        KeepLastDraw(std::span(views.data(), views.size()),
                     std::span(samplers.data(), samplers.size()), template_ranges, rescaling,
                     !tex_bank_hit);
    }
    if (mesh_emulation) {
        if (!mesh_output_buffer) {
            return false;
        }
        guest_descriptor_queue.AddBuffer(mesh_output_buffer, mesh_output_offset, mesh_output_size);
        if (mesh_emulation->has_task) {
            for (size_t index = 0; index < 3; ++index) { // payload, offsets, arguments
                const MeshEmulationBuffer& task_buffer{mesh_task_buffers[index]};
                if (!task_buffer.buffer) {
                    return false;
                }
                guest_descriptor_queue.AddBuffer(task_buffer.buffer, task_buffer.offset,
                                                 task_buffer.size);
            }
        }
    }
    if (held != nullptr && (held->flags & MemStage::PostCopyBarrier) != 0) {
        // FRITCHII_MEMSTAGE_THREAD = 2: the stage's fill uploaded (the mark came in the packet).
        mem_stage.PlayPostCopyBarrier();
    } else if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }
    mark();
    {
        Profile::Scope scope{Profile::Zone::RenderTargets};
        texture_cache.UpdateRenderTargets(false);
        const std::span<const VideoCommon::ImageViewInOut> view_span{views.data(), views.size()};
        texture_cache.CheckFeedbackLoop(view_span);
        if (scheduler.HasFragmentBarrier() &&
            (scheduler.RenderPassHasStores() || texture_cache.SamplesRenderTarget(view_span))) {
            scheduler.RequestOutsideRenderPassOperationContext();
        }
    }
    if (texture_cache.RenderTargetsOutOfMemory()) [[unlikely]] {
        // CheezyEmu a08015f893: a bound target got no memory; skip the draw.
        return false;
    }
    if (IsBuilt() && (!pipeline || build->Failed())) {
        return false;
    }
    mark();
    bool configured;
    {
        Profile::Scope scope{Profile::Zone::ConfigureDraw};
        if (occl_flag_parent != nullptr) [[unlikely]] {
            occl_prepass_rescaling = rescaling.Data(); // FRITCHII_OCCL_FLAG pre-pass
            occl_prepass_render_area = render_area;
        }
        configured = ConfigureDraw(rescaling, render_area);
    }
    if (step_trace) [[unlikely]] {
        mark();
        const auto ms = [&](size_t i) {
            return std::chrono::duration<double, std::milli>(marks[i + 1] - marks[i]).count();
        };
        const double total =
            std::chrono::duration<double, std::milli>(marks[mark_count - 1] - marks[0]).count();
        static std::atomic<u32> logged{};
        if (total > 4.0 && logged.fetch_add(1, std::memory_order_relaxed) < 300) {
            LOG_WARNING(Render_Vulkan,
                        "Slow configure {:.2f} ms (shaders {:016x}/{:016x}): sync {:.2f}, handles "
                        "{:.2f}, image views {:.2f}, texture buffers {:.2f}, update buffers {:.2f}, "
                        "descriptors {:.2f}, stage buffers+push {:.2f}, render targets {:.2f}, "
                        "configure draw {:.2f}",
                        total, key.unique_hashes[1], key.unique_hashes[5], ms(0), ms(1), ms(2),
                        ms(3), ms(4), ms(5), ms(6), ms(7), ms(8));
        }
    }
    return configured;
}

bool GraphicsPipeline::ConfigureAfterOcclPrepass(const GraphicsPipeline& variant) {
    RescalingPushConstant rescaling;
    rescaling.Assign(variant.occl_prepass_rescaling);
    return ConfigureDraw(rescaling, variant.occl_prepass_render_area);
}

bool GraphicsPipeline::ConfigureDraw(const RescalingPushConstant& rescaling,
                                     const RenderAreaPushConstant& render_area) {
    if (guest_descriptor_queue.UpdateCount() != num_descriptor_entries) {
        LOG_CRITICAL(Render_Vulkan, "Graphics descriptor count mismatch: expected {}, got {}",
                     num_descriptor_entries, guest_descriptor_queue.UpdateCount());
        // One follower step 4: an unlent draw R built wrong. Counted (must be 0), not drawn,
        // and step 4 stops for the session (every later draw lent) before more go wrong.
        if (VideoCommon::OneFollower::PlayingUnlentDraw()) [[unlikely]] {
            OneFollowerDraw::ReportDescriptorMismatch(*this, num_descriptor_entries,
                                                      guest_descriptor_queue.UpdateCount());
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::BuildDescriptors,
                                           "unlent draw: descriptor count against its pipeline");
            VideoCommon::OneFollower::DisableUnlend(
                "an unlent draw's descriptor count did not match its pipeline");
        }
        return false;
    }
    const void* const descriptor_data{guest_descriptor_queue.UpdateData()};

    VkDeviceSize descriptor_buffer_offset{};
    u32 descriptor_buffer_chunk{};
    if (descriptor_set_layout && uses_descriptor_buffer) {
        const auto* const entries = static_cast<const DescriptorUpdateEntry*>(descriptor_data);
        const bool reuse_allocation =
            last_descriptor_buffer_generation == descriptor_buffer_ring.CurrentGeneration() &&
            last_descriptor_payload.size() == num_descriptor_entries &&
            std::memcmp(last_descriptor_payload.data(), entries,
                        num_descriptor_entries * sizeof(DescriptorUpdateEntry)) == 0;
        if (reuse_allocation) {
            descriptor_buffer_offset = last_descriptor_buffer_offset;
            descriptor_buffer_chunk = last_descriptor_buffer_chunk;
            descriptor_buffer_ring.TouchFrame(scheduler);
        } else {
            const DescriptorBufferRing::Allocation alloc{
                descriptor_buffer_ring.Allocate(scheduler, descriptor_buffer_layout.size)};
            if (!alloc.host) {
                LOG_DEBUG(Render_Vulkan, "Failed to reserve descriptor memory, skipping draw");
                return false;
            }
            WriteDescriptorBuffer(device, descriptor_buffer_layout, entries, alloc.host);
            descriptor_buffer_offset = alloc.offset;
            descriptor_buffer_chunk = alloc.chunk;
            last_descriptor_buffer_offset = alloc.offset;
            last_descriptor_buffer_chunk = alloc.chunk;
            last_descriptor_buffer_generation = alloc.generation;
            last_descriptor_payload.assign(entries, entries + num_descriptor_entries);
        }
    }

    if (mesh_emulation) {
        // The mesh program runs as a compute dispatch outside the render pass and writes the
        // triangles the draw then reads; both use the descriptor set committed here.
        scheduler.RequestOutsideRenderPassOperationContext();
        if (!IsBuilt() &&
            !HoldForPipelineBuild(*build, scheduler, library_cache, texture_cache,
                                  worker_only_build)) {
            return false;
        }
        // EDEN_NX2_MESH_FIRST: the task (or mesh) compute shader adds the draw's first
        // workgroup index, as the native mesh path does.
        const bool push_draw_base{MeshFirstEnabled()};
        MeshDrawBaseLayout draw_base{};
        if (push_draw_base) {
            const u32 first{maxwell3d->draw_manager.draw_state.vertex_buffer.first};
            (stage_infos[0].is_task ? draw_base.task : draw_base.mesh) = first;
        }
        scheduler.Record([this, descriptor_data, groups = mesh_groups, capacity = mesh_capacity,
                          task_buffers = mesh_task_buffers,
                          rescaling_data = rescaling.Data(), is_rescaling = texture_cache.IsRescaling(),
                          uses_render_area = render_area.uses_render_area,
                          render_area_data = render_area.words, push_draw_base,
                          draw_base](vk::CommandBuffer cmdbuf) {
            mesh_descriptor_set = VK_NULL_HANDLE;
            const bool has_task{mesh_emulation->has_task};
            if (!pipeline || !mesh_compute_pipeline || build->Failed() || groups == 0 ||
                (has_task && (!task_compute_pipeline || !distribution_pipeline))) {
                return;
            }
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            device.GetLogical().UpdateDescriptorSet(descriptor_set, *descriptor_update_template,
                                                    descriptor_data);
            mesh_descriptor_set = descriptor_set;
            // The task and mesh programs read the draw's push constants and descriptors.
            const auto bind_program_state = [&] {
                constexpr VkShaderStageFlags push_stages{VK_SHADER_STAGE_ALL_GRAPHICS |
                                                         VK_SHADER_STAGE_COMPUTE_BIT};
                cmdbuf.PushConstants(*pipeline_layout, push_stages, RESCALING_LAYOUT_WORDS_OFFSET,
                                     sizeof(rescaling_data), rescaling_data.data());
                const f32 config_down_factor{Settings::values.resolution_info.down_factor};
                const f32 down_factor{is_rescaling ? config_down_factor : 1.0f};
                cmdbuf.PushConstants(*pipeline_layout, push_stages,
                                     RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, sizeof(down_factor),
                                     &down_factor);
                if (uses_render_area) {
                    cmdbuf.PushConstants(*pipeline_layout, push_stages, RENDERAREA_LAYOUT_OFFSET,
                                         sizeof(render_area_data), &render_area_data);
                }
                if (push_draw_base) {
                    cmdbuf.PushConstants(*pipeline_layout, push_stages, MESH_DRAW_BASE_OFFSET,
                                         sizeof(draw_base), &draw_base);
                }
                cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline_layout, 0,
                                          descriptor_set, nullptr);
            };
            const auto memory_barrier = [&](VkAccessFlags src_access, VkAccessFlags dst_access,
                                            VkPipelineStageFlags src_stages,
                                            VkPipelineStageFlags dst_stages) {
                const VkMemoryBarrier barrier{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = src_access,
                    .dstAccessMask = dst_access,
                };
                cmdbuf.PipelineBarrier(src_stages, dst_stages, 0, barrier);
            };
            // The scheduler orders earlier compute writes only before the next render pass
            // (COMPUTE -> ALL_GRAPHICS); these dispatches run before it, and the programs read
            // what the game's compute passes wrote (foliage instance lists, for one).
            memory_barrier(VK_ACCESS_MEMORY_WRITE_BIT,
                           VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                           vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            if (!has_task) {
                cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *mesh_compute_pipeline);
                bind_program_state();
                cmdbuf.Dispatch(groups, 1, 1);
                memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
                return;
            }
            // 1. One task workgroup per draw group writes its payload (mesh workgroup count at
            //    word 1).
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *task_compute_pipeline);
            bind_program_state();
            cmdbuf.Dispatch(groups, 1, 1);
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            // 2. Distribution: each task's first mesh workgroup, the draw's place in the output
            //    arena, and the mesh dispatch and draw arguments.
            const VkDescriptorSet distribution_set{distribution_allocator.Commit()};
            std::array<VkDescriptorBufferInfo, 4> buffer_infos{};
            std::array<VkWriteDescriptorSet, 4> writes{};
            for (u32 binding = 0; binding < 4; ++binding) {
                buffer_infos[binding] = {
                    .buffer = task_buffers[binding].buffer,
                    .offset = task_buffers[binding].offset,
                    .range = task_buffers[binding].size,
                };
                writes[binding] = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = distribution_set,
                    .dstBinding = binding,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &buffer_infos[binding],
                    .pTexelBufferView = nullptr,
                };
            }
            device.GetLogical().UpdateDescriptorSets(writes, {});
            const Shader::Backend::SPIRV::MeshEmulationDistributionConstants constants{
                .tasks = groups,
                .payload_words = mesh_emulation->payload_stride,
                .capacity_words = capacity,
                .block_words = mesh_emulation->block_words,
                .vertices_per_group =
                    mesh_emulation->max_primitives * mesh_emulation->primitive_vertices,
                .max_groups = device.GetMaxComputeWorkGroupCount()[0],
            };
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *distribution_pipeline);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *distribution_layout, 0,
                                      distribution_set, nullptr);
            cmdbuf.PushConstants(*distribution_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(constants), &constants);
            cmdbuf.Dispatch(1, 1, 1);
            // The host reads the arena counter once the submission finishes.
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                               VK_ACCESS_HOST_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_HOST_BIT);
            // 3. The mesh workgroups, as many as the tasks asked for (up to the arena's end).
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *mesh_compute_pipeline);
            bind_program_state();
            cmdbuf.DispatchIndirect(task_buffers[2].buffer, task_buffers[2].offset);
            memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
        });
    }
    if (buffer_cache.runtime.Player().ListDrawArgsRequested()) {
        // List draws (nx2_list_draw.h): outside the render pass, before the
        // draw's counters start.
        using Link = VideoCommon::MemoryStageLink;
        MemStage& seam = buffer_cache.runtime.Seam();
        const MemStage::DrawPacket* const ring_packet =
            seam.HeldFromRing() ? seam.HeldDraw() : nullptr;
        if (ring_packet != nullptr && (ring_packet->flags & MemStage::ListArgs) != 0) {
            // FRITCHII_MEMSTAGE_LIST_DRAWS: GPUMem's fill bound the two words and took their
            // buffer and offset where this call took them (GetDrawIndirectBuffer after the
            // geometry section), at the same position in the stream.
            buffer_cache.runtime.Player().RecordListDrawArgs(
                ring_packet->indirect.buffer, ring_packet->indirect.offset, IsMeshPipeline());
            Link::Get().NoteListDrawn();
        } else {
            if (Link::BackServed()) [[unlikely]] {
                Link::Note(Link::Check::ListArgsMissing, "ConfigureDraw");
            }
            const auto [list_buffer, list_offset] = seam.Ref(
                MemStage::Site::ListArgs, [&] { return buffer_cache.GetDrawIndirectBuffer(); });
            buffer_cache.runtime.Player().RecordListDrawArgs(list_buffer->Handle(), list_offset,
                                                             IsMeshPipeline());
        }
    }
    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    if (writes_storage) {
        scheduler.NoteRenderPassStores();
    }
    if (!IsBuilt()) {
        if (!HoldForPipelineBuild(*build, scheduler, library_cache, texture_cache,
                                  worker_only_build)) {
            return false;
        }
    } else {
        NoteDrawnTarget(texture_cache.ClassifyDrawTargets());
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const bool update_rescaling{scheduler.UpdateRescaling(is_rescaling, IsMeshPipeline())};
    const bool bind_pipeline{scheduler.UpdateGraphicsPipeline(this)};
    const bool push_mesh_draw_base{IsMeshPipeline() && MeshFirstEnabled()};
    MeshDrawBaseLayout mesh_draw_base{};
    if (push_mesh_draw_base) {
        const u32 first{maxwell3d->draw_manager.draw_state.vertex_buffer.first};
        (stage_infos[0].is_task ? mesh_draw_base.task : mesh_draw_base.mesh) = first;
    }
    const bool bind_descriptor_buffer{
        descriptor_set_layout && uses_descriptor_buffer &&
        scheduler.UpdateDescriptorBufferChunk(descriptor_buffer_chunk)};

    // Log graphics pipeline binding
    if (bind_pipeline && GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string pipeline_info = fmt::format("hash={:#016x}", key.Hash());
        GPU::Logging::GPULogger::GetInstance().LogPipelineBind(false, pipeline_info);
    }

    bool update_descriptors = true;
    if ((descriptor_set_layout && !uses_push_descriptor && !uses_descriptor_buffer) ||
        has_stage_sets) {
        const auto* const entries = static_cast<const DescriptorUpdateEntry*>(descriptor_data);
        update_descriptors =
            bind_pipeline || last_descriptor_payload.size() != num_descriptor_entries ||
            std::memcmp(last_descriptor_payload.data(), entries,
                        num_descriptor_entries * sizeof(DescriptorUpdateEntry)) != 0;
        if (update_descriptors) {
            last_descriptor_payload.assign(entries, entries + num_descriptor_entries);
        }
    }
    scheduler.Record([this, descriptor_data, bind_pipeline, update_descriptors,
                      descriptor_buffer_offset, descriptor_buffer_chunk, bind_descriptor_buffer,
                      rescaling_data = rescaling.Data(), is_rescaling, update_rescaling,
                      uses_render_area = render_area.uses_render_area,
                      render_area_data = render_area.words, push_mesh_draw_base,
                      mesh_draw_base](vk::CommandBuffer cmdbuf) {
        if (bind_descriptor_buffer) {
            const VkDescriptorBufferBindingInfoEXT binding_info{
                descriptor_buffer_ring.BindingInfo(descriptor_buffer_chunk)};
            cmdbuf.BindDescriptorBuffersEXT(binding_info);
        }
        if (bind_pipeline) {
            scheduler.SetGraphicsPipelineMissing(!pipeline || build->Failed());
            if (!pipeline || build->Failed()) {
                return;
            }
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline);
            // FRITCHII_SLOW_GPU: the programs a slow submission's draws ran (one TLS load off).
            vk::NoteBoundProgram(VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 key.unique_hashes[1] != 0 ? key.unique_hashes[1]
                                                           : key.unique_hashes[0],
                                 key.unique_hashes[5]);
        }
        const VkShaderStageFlags push_stages =
            mesh_emulation ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT
            : IsMeshPipeline()
                ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT |
                      VK_SHADER_STAGE_MESH_BIT_EXT
                : VK_SHADER_STAGE_ALL_GRAPHICS;
        cmdbuf.PushConstants(*pipeline_layout, push_stages, RESCALING_LAYOUT_WORDS_OFFSET,
                             sizeof(rescaling_data), rescaling_data.data());
        if (update_rescaling || mesh_emulation) {
            const f32 config_down_factor{Settings::values.resolution_info.down_factor};
            const f32 scale_down_factor{is_rescaling ? config_down_factor : 1.0f};
            cmdbuf.PushConstants(*pipeline_layout, push_stages,
                                 RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, sizeof(scale_down_factor),
                                 &scale_down_factor);
        }
        if (uses_render_area) {
            cmdbuf.PushConstants(*pipeline_layout, push_stages, RENDERAREA_LAYOUT_OFFSET,
                                 sizeof(render_area_data), &render_area_data);
        }
        if (push_mesh_draw_base) {
            cmdbuf.PushConstants(*pipeline_layout, push_stages, MESH_DRAW_BASE_OFFSET,
                                 sizeof(mesh_draw_base), &mesh_draw_base);
        }
        if (has_stage_sets) {
            // Graphics pipeline libraries: one set per stage, at the stage's index.
            if (update_descriptors) {
                const vk::Device& dev{device.GetLogical()};
                for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
                    if (!stage_set_layouts[stage]) {
                        continue;
                    }
                    const VkDescriptorSet descriptor_set{stage_allocators[stage].Commit()};
                    dev.UpdateDescriptorSet(descriptor_set, *stage_templates[stage],
                                            static_cast<const u8*>(descriptor_data) +
                                                stage_descriptor_offsets[stage]);
                    cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout,
                                              static_cast<u32>(stage), descriptor_set, nullptr);
                }
            }
            return;
        }
        if (!descriptor_set_layout) {
            return;
        }
        if (mesh_emulation) {
            // The set the emulation's dispatch committed (see above).
            if (mesh_descriptor_set) {
                cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0,
                                          mesh_descriptor_set, nullptr);
            }
            return;
        }
        if (uses_descriptor_buffer) {
            const u32 buffer_index{};
            cmdbuf.SetDescriptorBufferOffsetsEXT(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout,
                                                 0, buffer_index, descriptor_buffer_offset);
        } else if (uses_push_descriptor) {
            cmdbuf.PushDescriptorSetWithTemplateKHR(*descriptor_update_template, *pipeline_layout,
                                                    0, descriptor_data);
        } else if (update_descriptors) {
            const VkDescriptorSet descriptor_set{descriptor_allocator.Commit()};
            const vk::Device& dev{device.GetLogical()};
            dev.UpdateDescriptorSet(descriptor_set, *descriptor_update_template, descriptor_data);
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0,
                                      descriptor_set, nullptr);
        }
    });
    return true;
}

struct GraphicsPipeline::PipelineStates {
    FixedPipelineState::DynamicState dynamic{};
    static_vector<VkVertexInputBindingDescription, 32> vertex_bindings;
    static_vector<VkVertexInputBindingDivisorDescriptionEXT, 32> vertex_binding_divisors;
    static_vector<VkVertexInputAttributeDescription, 32> vertex_attributes;
    VkPipelineVertexInputStateCreateInfo vertex_input_ci{};
    VkPipelineVertexInputDivisorStateCreateInfoEXT input_divisor_ci{};
    VkPipelineInputAssemblyStateCreateInfo input_assembly_ci{};
    VkPipelineTessellationStateCreateInfo tessellation_ci{};
    std::array<VkViewportSwizzleNV, Maxwell::NumViewports> swizzles{};
    VkPipelineViewportSwizzleStateCreateInfoNV swizzle_ci{};
    VkPipelineViewportDepthClipControlCreateInfoEXT ndc_info{};
    VkPipelineViewportStateCreateInfo viewport_ci{};
    VkPipelineRasterizationStateCreateInfo rasterization_ci{};
    VkPipelineRasterizationLineStateCreateInfoEXT line_state{};
    VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_raster{};
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex{};
    VkPipelineMultisampleStateCreateInfo multisample_ci{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil_ci{};
    static_vector<VkPipelineColorBlendAttachmentState, Maxwell::NumRenderTargets> cb_attachments;
    VkPipelineColorBlendStateCreateInfo color_blend_ci{};
    static_vector<VkDynamicState, 34> dynamic_states;
    VkPipelineDynamicStateCreateInfo dynamic_state_ci{};
    bool is_mesh{};
    bool has_tess_stages{};
    // Which structures are chained (a library key holds only what the driver is given).
    bool chains_swizzle{};
    bool chains_ndc{};
    bool chains_line_state{};
    bool chains_conservative{};
    bool chains_provoking{};
};

void GraphicsPipeline::FillStates(PipelineStates& s) const {
    FixedPipelineState::DynamicState& dynamic = s.dynamic;
    if (!key.state.extended_dynamic_state) {
        dynamic = key.state.dynamic_state;
    } else {
        dynamic.raw1 = key.state.dynamic_state.raw1;
    }
    auto& vertex_bindings = s.vertex_bindings;
    auto& vertex_binding_divisors = s.vertex_binding_divisors;
    auto& vertex_attributes = s.vertex_attributes;
    if (!key.state.dynamic_vertex_input) {
        const size_t num_vertex_arrays = (std::min)(
            Maxwell::NumVertexArrays, static_cast<size_t>(device.GetMaxVertexInputBindings()));
        for (size_t index = 0; index < num_vertex_arrays; ++index) {
            const bool instanced = key.state.binding_divisors[index] != 0;
            const auto rate =
                instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
            vertex_bindings.push_back({
                .binding = static_cast<u32>(index),
                .stride = key.state.vertex_strides[index],
                .inputRate = rate,
            });
            if (instanced) {
                vertex_binding_divisors.push_back({
                    .binding = static_cast<u32>(index),
                    .divisor = key.state.binding_divisors[index],
                });
            }
        }
        for (size_t index = 0; index < key.state.attributes.size(); ++index) {
            const auto& attribute = key.state.attributes[index];
            if (!attribute.enabled || !stage_infos[0].loads.Generic(index)) {
                continue;
            }
            vertex_attributes.push_back({
                .location = static_cast<u32>(index),
                .binding = attribute.buffer,
                .format = MaxwellToVK::VertexFormat(device, attribute.Type(), attribute.Size()),
                .offset = attribute.offset,
            });
        }
    }
    ASSERT(vertex_attributes.size() <= device.GetMaxVertexInputAttributes());

    s.vertex_input_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = static_cast<u32>(vertex_bindings.size()),
        .pVertexBindingDescriptions = vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(vertex_attributes.size()),
        .pVertexAttributeDescriptions = vertex_attributes.data(),
    };
    s.input_divisor_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .vertexBindingDivisorCount = static_cast<u32>(vertex_binding_divisors.size()),
        .pVertexBindingDivisors = vertex_binding_divisors.data(),
    };
    if (!vertex_binding_divisors.empty()) {
        s.vertex_input_ci.pNext = &s.input_divisor_ci;
    }
    const bool has_tess_stages = spv_modules[1] || spv_modules[2];
    s.has_tess_stages = has_tess_stages;
    auto input_assembly_topology = MaxwellToVK::PrimitiveTopology(device, key.state.topology);
    if (input_assembly_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {
        if (!has_tess_stages) {
            LOG_WARNING(Render_Vulkan, "Patch topology used without tessellation, using points");
            input_assembly_topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        }
    } else {
        if (has_tess_stages) {
            // The Vulkan spec requires patch list IA topology be used with tessellation
            // shader stages. Forcing it fixes a crash on some drivers
            LOG_WARNING(Render_Vulkan,
                        "Patch topology not used with tessellation, using patch list");
            input_assembly_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
        }
    }
    s.input_assembly_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = input_assembly_topology,
        .primitiveRestartEnable =
        // MoltenVK/Metal always has primitive restart enabled and cannot disable it
        device.IsMoltenVK() ? VK_TRUE :
        (dynamic.primitive_restart_enable != 0 &&
                ((input_assembly_topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
                  device.IsTopologyListPrimitiveRestartSupported()) ||
                 SupportsPrimitiveRestart(input_assembly_topology) ||
                 (input_assembly_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
                  device.IsPatchListPrimitiveRestartSupported()))
            ? VK_TRUE
            : VK_FALSE),
    };
    s.tessellation_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .patchControlPoints = std::min(key.state.patch_control_points_minus_one.Value() + 1,
                                       device.GetMaxTessellationPatchSize()),
    };
    std::ranges::transform(key.state.viewport_swizzles, s.swizzles.begin(), UnpackViewportSwizzle);
    s.swizzle_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = Maxwell::NumViewports,
        .pViewportSwizzles = s.swizzles.data(),
    };
    s.ndc_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
        .pNext = nullptr,
        .negativeOneToOne = key.state.ndc_minus_one_to_one.Value() != 0 ? VK_TRUE : VK_FALSE,
    };
    const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
    s.viewport_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = num_viewports,
        .pViewports = nullptr,
        .scissorCount = num_viewports,
        .pScissors = nullptr,
    };
    if (device.IsNvViewportSwizzleSupported()) {
        s.swizzle_ci.pNext = std::exchange(s.viewport_ci.pNext, &s.swizzle_ci);
        s.chains_swizzle = true;
    }
    if (device.IsExtDepthClipControlSupported()) {
        s.ndc_info.pNext = std::exchange(s.viewport_ci.pNext, &s.ndc_info);
        s.chains_ndc = true;
    }
    s.rasterization_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable =
        static_cast<VkBool32>(dynamic.depth_clamp_disabled == 0 ? VK_TRUE : VK_FALSE),
        .rasterizerDiscardEnable =
        static_cast<VkBool32>(dynamic.rasterize_enable == 0 ? VK_TRUE : VK_FALSE),
        .polygonMode =
        MaxwellToVK::PolygonMode(FixedPipelineState::UnpackPolygonMode(key.state.polygon_mode)),
        .cullMode = static_cast<VkCullModeFlags>(
            dynamic.cull_enable ? MaxwellToVK::CullFace(dynamic.CullFace()) : VK_CULL_MODE_NONE),
        .frontFace = MaxwellToVK::FrontFace(dynamic.FrontFace()),
        .depthBiasEnable = (dynamic.depth_bias_enable != 0 ? VK_TRUE : VK_FALSE),
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
        // TODO(alekpop): Transfer from regs
    };
    const bool smooth_lines_supported =
        device.IsExtLineRasterizationSupported() && device.SupportsSmoothLines();
    const bool stippled_lines_supported =
        device.IsExtLineRasterizationSupported() && device.SupportsStippledRectangularLines();
    s.line_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .lineRasterizationMode = key.state.smooth_lines != 0 && smooth_lines_supported
                                     ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
                                     : VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT,
        .stippledLineEnable =
            (dynamic.line_stipple_enable && stippled_lines_supported) ? VK_TRUE : VK_FALSE,
        .lineStippleFactor = key.state.line_stipple_factor,
        .lineStipplePattern = static_cast<uint16_t>(key.state.line_stipple_pattern),
    };
    s.conservative_raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = 0,
        .conservativeRasterizationMode = key.state.conservative_raster_enable != 0
                                             ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                                             : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT,
        .extraPrimitiveOverestimationSize = 0.0f,
    };
    const bool supports_provoking_first_mode = device.SupportsProvokingVertexFirstMode();
    const bool supports_provoking_last_mode = device.SupportsProvokingVertexLastMode();
    const bool requested_provoking_last_mode = key.state.provoking_vertex_last != 0;
    const VkProvokingVertexModeEXT provoking_vertex_mode =
        requested_provoking_last_mode
            ? (supports_provoking_last_mode ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                            : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT)
            : (supports_provoking_first_mode ? VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT
                                             : VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT);
    s.provoking_vertex = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .provokingVertexMode = provoking_vertex_mode,
    };

    if (IsLine(input_assembly_topology) && device.IsExtLineRasterizationSupported()) {
        s.line_state.pNext = std::exchange(s.rasterization_ci.pNext, &s.line_state);
        s.chains_line_state = true;
    }
    if (device.IsExtConservativeRasterizationSupported()) {
        s.conservative_raster.pNext =
            std::exchange(s.rasterization_ci.pNext, &s.conservative_raster);
        s.chains_conservative = true;
    }
    if (device.IsExtProvokingVertexSupported()) {
        s.provoking_vertex.pNext = std::exchange(s.rasterization_ci.pNext, &s.provoking_vertex);
        s.chains_provoking = true;
    }

    const bool supports_alpha_output = fragment_has_color0_output;
    const bool alpha_to_one_supported = device.SupportsAlphaToOne();
    s.multisample_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = MaxwellToVK::MsaaMode(key.state.msaa_mode),
        .sampleShadingEnable = Settings::values.sample_shading.GetValue() > 0 ? VK_TRUE : VK_FALSE,
        .minSampleShading = f32(Settings::values.sample_shading.GetValue()) / 100.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable =
            supports_alpha_output && key.state.alpha_to_coverage_enabled != 0 ? VK_TRUE : VK_FALSE,
        .alphaToOneEnable = supports_alpha_output && alpha_to_one_supported &&
                           key.state.alpha_to_one_enabled != 0 ? VK_TRUE : VK_FALSE,
    };
    s.depth_stencil_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthTestEnable = dynamic.depth_test_enable,
        .depthWriteEnable = dynamic.depth_write_enable,
        .depthCompareOp = dynamic.depth_test_enable
                              ? MaxwellToVK::ComparisonOp(dynamic.DepthTestFunc())
                              : VK_COMPARE_OP_ALWAYS,
        .depthBoundsTestEnable = dynamic.depth_bounds_enable && device.IsDepthBoundsSupported(),
        .stencilTestEnable = dynamic.stencil_enable,
        .front = GetStencilFaceState(dynamic.front),
        .back = GetStencilFaceState(dynamic.back),
        .minDepthBounds = f32(key.state.depth_bounds_min),
        .maxDepthBounds = f32(key.state.depth_bounds_max),
    };
    if (dynamic.depth_bounds_enable && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
    }
    auto& cb_attachments = s.cb_attachments;
    const size_t num_attachments{NumAttachments(key.state)};
    for (size_t index = 0; index < num_attachments; ++index) {
        static constexpr std::array mask_table{
            VK_COLOR_COMPONENT_R_BIT,
            VK_COLOR_COMPONENT_G_BIT,
            VK_COLOR_COMPONENT_B_BIT,
            VK_COLOR_COMPONENT_A_BIT,
        };
        const auto& blend{key.state.attachments[index]};
        const std::array mask{blend.Mask()};
        VkColorComponentFlags write_mask{};
        for (size_t i = 0; i < mask_table.size(); ++i) {
            write_mask |= mask[i] ? mask_table[i] : 0;
        }
        write_mask &= ColorOutputMask(index);
        cb_attachments.push_back({
            .blendEnable = blend.enable != 0,
            .srcColorBlendFactor = MaxwellToVK::BlendFactor(blend.SourceRGBFactor()),
            .dstColorBlendFactor = MaxwellToVK::BlendFactor(blend.DestRGBFactor()),
            .colorBlendOp = MaxwellToVK::BlendEquation(blend.EquationRGB()),
            .srcAlphaBlendFactor = MaxwellToVK::BlendFactor(blend.SourceAlphaFactor()),
            .dstAlphaBlendFactor = MaxwellToVK::BlendFactor(blend.DestAlphaFactor()),
            .alphaBlendOp = MaxwellToVK::BlendEquation(blend.EquationAlpha()),
            .colorWriteMask = write_mask,
        });
    }
    s.color_blend_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = dynamic.logic_op_enable != 0,
        .logicOp = static_cast<VkLogicOp>(dynamic.logic_op.Value()),
        .attachmentCount = static_cast<u32>(cb_attachments.size()),
        .pAttachments = cb_attachments.data(),
        .blendConstants = {}
    };
    auto& dynamic_states = s.dynamic_states;
    dynamic_states.assign({
        VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,         VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS,       VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_LINE_WIDTH,
    });
    if (key.state.extended_dynamic_state) {
        static constexpr std::array extended{
            VK_DYNAMIC_STATE_CULL_MODE_EXT,
            VK_DYNAMIC_STATE_FRONT_FACE_EXT,
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE_EXT,
            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP_EXT,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE_EXT,
            VK_DYNAMIC_STATE_STENCIL_OP_EXT,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended.begin(), extended.end());

        // VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT
        if (!key.state.dynamic_vertex_input) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT);
        }
    }

    // VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
    if (key.state.dynamic_vertex_input) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
    }

    // EDS2 - Core (3 states)
    if (key.state.extended_dynamic_state_2) {
        static constexpr std::array extended2{
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE_EXT,
            VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT,
            VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended2.begin(), extended2.end());
    }

    // EDS2 - LogicOp (granular)
    if (key.state.extended_dynamic_state_2_logic_op) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_EXT);
    }

    // EDS3 - Blending (composite: 3 states)
    if (key.state.extended_dynamic_state_3_blend) {
        static constexpr std::array extended3{
            VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
            VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
            VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
        };
        dynamic_states.insert(dynamic_states.end(), extended3.begin(), extended3.end());
    }

    // VK_EXT_color_write_enable fallback for fully on/off render targets when EDS3 blending is not available.
    if (!key.state.extended_dynamic_state_3_blend && key.state.color_write_enable_dynamic) {
        dynamic_states.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT);
    }

    // EDS3 - Enables (composite: per-feature)
    if (key.state.extended_dynamic_state_3_enables) {
        if (device.SupportsDynamicState3DepthClampEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3LogicOpEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3LineRasterizationMode()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT);
        }
        if (device.SupportsDynamicState3ConservativeRasterizationMode()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT);
        }
        if (device.SupportsDynamicState3LineStippleEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3AlphaToCoverageEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT);
        }
        if (device.SupportsDynamicState3AlphaToOneEnable()) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT);
        }
    }

    const bool is_mesh{IsMeshPipeline()};
    s.is_mesh = is_mesh;
    if (is_mesh) {
        // Mesh pipelines have no vertex input or input assembly.
        dynamic_states.erase(std::remove_if(dynamic_states.begin(), dynamic_states.end(),
                                            [](VkDynamicState state) {
                                                return state == VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY_EXT ||
                                                       state == VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE_EXT ||
                                                       state == VK_DYNAMIC_STATE_VERTEX_INPUT_EXT ||
                                                       state == VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE_EXT ||
                                                       state == VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT;
                                            }),
                             dynamic_states.end());
    }
    s.dynamic_state_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
}

void GraphicsPipeline::MakePipeline(VkRenderPass render_pass) {
    if (IsMeshPipeline() && !device.IsExtMeshShaderSupported() && !mesh_emulation) {
        return;
    }
    if (mesh_emulation) {
        // The mesh program (stage 0, or mesh_stage after a task program in stage 0) is built as
        // a compute shader, and so is the task program; the draw uses the vertex shader that
        // reads the mesh program's output.
        // Votes, ballots and shuffles span the guest's 32-thread warp, which a driver may split
        // in compute shaders too (RDNA runs them wave64).
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT compute_subgroup_size_ci{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
            .pNext = nullptr,
            .requiredSubgroupSize = GuestWarpSize,
        };
        const auto make_compute = [&](VkShaderModule module, VkPipelineLayout layout,
                                      const Shader::Info* info) {
            const bool uses_warp_ops{info && (info->uses_subgroup_vote ||
                                              info->uses_subgroup_invocation_id ||
                                              info->uses_subgroup_shuffles ||
                                              info->uses_subgroup_mask ||
                                              info->uses_quad_shuffles || info->uses_fswzadd)};
            const bool force_warp_size{uses_warp_ops && device.IsExtSubgroupSizeControlSupported() &&
                                       device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT)};
            return device.GetLogical().CreateComputePipeline(
                VkComputePipelineCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage{
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        .pNext = force_warp_size ? &compute_subgroup_size_ci : nullptr,
                        .flags = 0,
                        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                        .module = module,
                        .pName = "main",
                        .pSpecializationInfo = nullptr,
                    },
                    .layout = layout,
                    .basePipelineHandle = nullptr,
                    .basePipelineIndex = 0,
                },
                *pipeline_cache);
        };
        mesh_vertex_module = BuildShader(device, mesh_emulation->vertex_spirv);
        mesh_compute_pipeline = make_compute(*spv_modules[mesh_emulation->mesh_stage],
                                             *pipeline_layout,
                                             &stage_infos[mesh_emulation->mesh_stage]);
        if (mesh_emulation->has_task) {
            task_compute_pipeline =
                make_compute(*spv_modules[0], *pipeline_layout, &stage_infos[0]);
            distribution_module = BuildShader(device, mesh_emulation->distribution_spirv);
            distribution_pipeline =
                make_compute(*distribution_module, *distribution_layout, nullptr);
        }
    }
    PipelineStates states;
    FillStates(states);
    const bool is_mesh{states.is_mesh};
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .requiredSubgroupSize = GuestWarpSize,
    };
    static_vector<VkPipelineShaderStageCreateInfo, 5> shader_stages;
    for (size_t stage = 0; stage < Maxwell::MaxShaderStage; ++stage) {
        if (!spv_modules[stage]) {
            continue;
        }
        if (mesh_emulation && mesh_emulation->has_task && stage == mesh_emulation->mesh_stage) {
            continue; // the mesh program runs as a compute shader
        }
        // Mesh emulation: stage 0 is the vertex shader reading the compute pass's output.
        const bool emulated_vertex{mesh_emulation && stage == 0};
        const VkShaderStageFlagBits stage_flag{
            emulated_vertex ? VK_SHADER_STAGE_VERTEX_BIT
                            : StageFlag(stage_infos[stage],
                                        MaxwellToVK::ShaderStage(Shader::StageFromIndex(stage)))};
        shader_stages.emplace_back(VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = emulated_vertex ? nullptr
                                     : GuestWarpSizeChain(device, stage_infos[stage], stage_flag,
                                                          subgroup_size_ci),
            .flags = 0,
            .stage = stage_flag,
            .module = emulated_vertex ? *mesh_vertex_module : *spv_modules[stage],
            .pName = "main",
            .pSpecializationInfo = nullptr,
        });
    }
    // Mesh emulation draws a plain triangle list without vertex attributes.
    const VkPipelineVertexInputStateCreateInfo emulation_vertex_input{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };
    const VkPipelineInputAssemblyStateCreateInfo emulation_input_assembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        // The mesh program's output topology (points and lines since Fritchii's point fallback).
        .topology = !mesh_emulation                           ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                    : mesh_emulation->primitive_vertices == 1 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                    : mesh_emulation->primitive_vertices == 2 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                                                              : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };
    VkPipelineCreateFlags flags{};
    if (device.IsKhrPipelineExecutablePropertiesEnabled() && Settings::values.renderer_debug.GetValue()) {
        flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    if (uses_descriptor_buffer) {
        flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    }

    // Through Fritchii's pipeline binary store when it is on (FRITCHII_PIPELINE_BINARY).
    pipeline = PipelineBinary::CreateGraphicsPipeline(device, {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = flags,
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = mesh_emulation ? &emulation_vertex_input
                             : is_mesh      ? nullptr
                                            : &states.vertex_input_ci,
        .pInputAssemblyState = mesh_emulation ? &emulation_input_assembly
                               : is_mesh      ? nullptr
                                              : &states.input_assembly_ci,
        .pTessellationState = is_mesh ? nullptr : &states.tessellation_ci,
        .pViewportState = &states.viewport_ci,
        .pRasterizationState = &states.rasterization_ci,
        .pMultisampleState = &states.multisample_ci,
        .pDepthStencilState = &states.depth_stencil_ci,
        .pColorBlendState = &states.color_blend_ci,
        .pDynamicState = &states.dynamic_state_ci,
        .layout = *pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    }, *pipeline_cache);

    // Log graphics pipeline creation
    if (GPU::Logging::IsActive()) {
        const std::string pipeline_info = fmt::format(
            "GraphicsPipeline created: stages={}, attachments={}",
            shader_stages.size(),
            states.color_blend_ci.attachmentCount
        );
        GPU::Logging::GPULogger::GetInstance().LogPipelineStateChange(pipeline_info);
    }
}

namespace {
constexpr bool IsPreRasterizationStage(size_t stage) {
    return stage + 1 < Maxwell::MaxShaderStage;
}

bool InPart(GplPart part, size_t stage) {
    return part == GplPart::PreRasterization ? IsPreRasterizationStage(stage)
                                             : !IsPreRasterizationStage(stage);
}

/// Create flags of a shader library, keeping link time optimization info when optimized links are
/// on or fast links are not usable on `device`.
VkPipelineCreateFlags LibraryFlags(const Device& device) {
    VkPipelineCreateFlags flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
    if (GplOptimizeEnabled() || !device.CanFastLinkGraphicsLibraries()) {
        flags |= VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
    }
    return flags;
}
} // Anonymous namespace

std::string GraphicsPipeline::ShaderLibraryKey(GplPart part, const PipelineStates& s,
                                               VkRenderPass render_pass) const {
    GplKeyWriter key_writer;
    key_writer.Add(static_cast<u32>(part));
    // Same render pass handle = same formats and samples (the render pass cache keeps them).
    key_writer.Add(render_pass);
    key_writer.Add(static_cast<u32>(LibraryFlags(device)));
    const VkPushConstantRange push_range{DescriptorLayoutBuilder{device}.PushConstantRange()};
    key_writer.Add(push_range.stageFlags);
    key_writer.Add(push_range.offset);
    key_writer.Add(push_range.size);
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!InPart(part, stage)) {
            continue;
        }
        key_writer.Add(static_cast<u32>(stage));
        key_writer.Add(static_cast<u32>(spv_modules[stage] ? 1 : 0));
        key_writer.Add(static_cast<u32>(StageFlag(stage_infos[stage], SlotStage(stage))));
        key_writer.Add(spirv_hashes[stage]);
        key_writer.AddBytes(stage_set_signatures[stage]);
    }
    if (part == GplPart::PreRasterization) {
        key_writer.Add(s.viewport_ci.viewportCount);
        key_writer.Add(s.viewport_ci.scissorCount);
        key_writer.Add(static_cast<u32>(s.chains_swizzle));
        if (s.chains_swizzle) {
            key_writer.Add(s.swizzle_ci.viewportCount);
            for (const VkViewportSwizzleNV& swizzle : s.swizzles) {
                key_writer.Add(swizzle.x);
                key_writer.Add(swizzle.y);
                key_writer.Add(swizzle.z);
                key_writer.Add(swizzle.w);
            }
        }
        key_writer.Add(static_cast<u32>(s.chains_ndc));
        if (s.chains_ndc) {
            key_writer.Add(s.ndc_info.negativeOneToOne);
        }
        const VkPipelineRasterizationStateCreateInfo& raster = s.rasterization_ci;
        key_writer.Add(raster.depthClampEnable);
        key_writer.Add(raster.rasterizerDiscardEnable);
        key_writer.Add(raster.polygonMode);
        key_writer.Add(raster.cullMode);
        key_writer.Add(raster.frontFace);
        key_writer.Add(raster.depthBiasEnable);
        key_writer.Add(raster.depthBiasConstantFactor);
        key_writer.Add(raster.depthBiasClamp);
        key_writer.Add(raster.depthBiasSlopeFactor);
        key_writer.Add(raster.lineWidth);
        key_writer.Add(static_cast<u32>(s.chains_line_state));
        if (s.chains_line_state) {
            key_writer.Add(s.line_state.lineRasterizationMode);
            key_writer.Add(s.line_state.stippledLineEnable);
            key_writer.Add(s.line_state.lineStippleFactor);
            key_writer.Add(s.line_state.lineStipplePattern);
        }
        key_writer.Add(static_cast<u32>(s.chains_conservative));
        if (s.chains_conservative) {
            key_writer.Add(s.conservative_raster.conservativeRasterizationMode);
            key_writer.Add(s.conservative_raster.extraPrimitiveOverestimationSize);
        }
        key_writer.Add(static_cast<u32>(s.chains_provoking));
        if (s.chains_provoking) {
            key_writer.Add(s.provoking_vertex.provokingVertexMode);
        }
        key_writer.Add(static_cast<u32>(s.has_tess_stages));
        if (s.has_tess_stages) {
            key_writer.Add(s.tessellation_ci.patchControlPoints);
        }
    } else {
        const VkPipelineMultisampleStateCreateInfo& multisample = s.multisample_ci;
        key_writer.Add(multisample.rasterizationSamples);
        key_writer.Add(multisample.sampleShadingEnable);
        key_writer.Add(multisample.minSampleShading);
        key_writer.Add(multisample.alphaToCoverageEnable);
        key_writer.Add(multisample.alphaToOneEnable);
        const VkPipelineDepthStencilStateCreateInfo& depth = s.depth_stencil_ci;
        key_writer.Add(depth.depthTestEnable);
        key_writer.Add(depth.depthWriteEnable);
        key_writer.Add(depth.depthCompareOp);
        key_writer.Add(depth.depthBoundsTestEnable);
        key_writer.Add(depth.stencilTestEnable);
        for (const VkStencilOpState& face : {depth.front, depth.back}) {
            key_writer.Add(face.failOp);
            key_writer.Add(face.passOp);
            key_writer.Add(face.depthFailOp);
            key_writer.Add(face.compareOp);
            key_writer.Add(face.compareMask);
            key_writer.Add(face.writeMask);
            key_writer.Add(face.reference);
        }
        key_writer.Add(depth.minDepthBounds);
        key_writer.Add(depth.maxDepthBounds);
    }
    key_writer.Add(static_cast<u32>(s.dynamic_states.size()));
    for (const VkDynamicState state : s.dynamic_states) {
        key_writer.Add(state);
    }
    return key_writer.Data();
}

GplLibrary GraphicsPipeline::MakeShaderLibrary(GplPart part, const PipelineStates& s,
                                               VkRenderPass render_pass) const {
    GplLibrary library;
    // The library's own set layouts, identically defined to the linked pipeline's (same
    // signature in the key), so the library outlives the pipeline that built it.
    std::array<VkDescriptorSetLayout, NUM_STAGES> set_layouts{};
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT,
        .pNext = nullptr,
        .requiredSubgroupSize = GuestWarpSize,
    };
    static_vector<VkPipelineShaderStageCreateInfo, 5> shader_stages;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!InPart(part, stage)) {
            continue;
        }
        DescriptorLayoutBuilder builder{device};
        builder.Add(stage_infos[stage], StageFlag(stage_infos[stage], SlotStage(stage)));
        if (!builder.Empty()) {
            library.set_layouts.push_back(builder.CreateDescriptorSetLayout(false));
            set_layouts[stage] = *library.set_layouts.back();
        }
        if (!spv_modules[stage]) {
            continue;
        }
        // Same stage info as the monolithic pipeline (MakePipeline).
        const VkShaderStageFlagBits stage_flag{
            StageFlag(stage_infos[stage], MaxwellToVK::ShaderStage(Shader::StageFromIndex(stage)))};
        shader_stages.push_back({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = GuestWarpSizeChain(device, stage_infos[stage], stage_flag, subgroup_size_ci),
            .flags = 0,
            .stage = stage_flag,
            .module = *spv_modules[stage],
            .pName = "main",
            .pSpecializationInfo = nullptr,
        });
    }
    library.layout =
        DescriptorLayoutBuilder{device}.CreateIndependentPipelineLayout(set_layouts, true);
    const bool pre_rasterization = part == GplPart::PreRasterization;
    const VkGraphicsPipelineLibraryCreateInfoEXT library_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = static_cast<VkGraphicsPipelineLibraryFlagsEXT>(
            pre_rasterization ? VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT
                              : VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT),
    };
    library.pipeline = device.GetLogical().CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &library_ci,
        .flags = LibraryFlags(device),
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.empty() ? nullptr : shader_stages.data(),
        .pVertexInputState = nullptr,
        .pInputAssemblyState = nullptr,
        .pTessellationState =
            pre_rasterization && s.has_tess_stages ? &s.tessellation_ci : nullptr,
        .pViewportState = pre_rasterization ? &s.viewport_ci : nullptr,
        .pRasterizationState = pre_rasterization ? &s.rasterization_ci : nullptr,
        .pMultisampleState = pre_rasterization ? nullptr : &s.multisample_ci,
        .pDepthStencilState = pre_rasterization ? nullptr : &s.depth_stencil_ci,
        .pColorBlendState = nullptr,
        .pDynamicState = &s.dynamic_state_ci,
        .layout = *library.layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    }, *pipeline_cache);
    return library;
}

vk::Pipeline GraphicsPipeline::LinkLibraries(VkRenderPass render_pass, bool optimized) {
    const std::array<VkPipeline, 4> libraries{
        *vertex_input_library,
        *shader_libraries[0]->pipeline,
        *shader_libraries[1]->pipeline,
        *fragment_output_library,
    };
    const VkPipelineLibraryCreateInfoKHR link_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
        .pNext = nullptr,
        .libraryCount = static_cast<u32>(libraries.size()),
        .pLibraries = libraries.data(),
    };
    VkPipelineCreateFlags flags{};
    if (device.IsKhrPipelineExecutablePropertiesEnabled() && Settings::values.renderer_debug.GetValue()) {
        flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    }
    if (optimized) {
        flags |= VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
    }
    const auto begin = std::chrono::steady_clock::now();
    vk::Pipeline linked_pipeline = device.GetLogical().CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &link_ci,
        .flags = flags,
        .stageCount = 0,
        .pStages = nullptr,
        .pVertexInputState = nullptr,
        .pInputAssemblyState = nullptr,
        .pTessellationState = nullptr,
        .pViewportState = nullptr,
        .pRasterizationState = nullptr,
        .pMultisampleState = nullptr,
        .pDepthStencilState = nullptr,
        .pColorBlendState = nullptr,
        .pDynamicState = nullptr,
        .layout = *pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    }, *pipeline_cache);
    library_cache->RecordLink(optimized, std::chrono::steady_clock::now() - begin);
    return linked_pipeline;
}

bool GraphicsPipeline::MakeLibraryPipeline(VkRenderPass render_pass) {
    PipelineStates states;
    FillStates(states);
    static constexpr std::array parts{GplPart::PreRasterization, GplPart::Fragment};
    bool any_library{};
    for (size_t index = 0; index < parts.size(); ++index) {
        const GplPart part = parts[index];
        if (libraries_only) {
            // Warm-up with one stage and no partner: build only the part that has a shader.
            bool has_shader{};
            for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
                has_shader |= InPart(part, stage) && static_cast<bool>(spv_modules[stage]);
            }
            if (!has_shader) {
                continue;
            }
        }
        // The log names a library after its vertex (or fragment) program.
        const u64 name = part == GplPart::PreRasterization ? key.unique_hashes[1]
                                                           : key.unique_hashes[5];
        shader_libraries[index] = library_cache->GetOrCreate(
            part, ShaderLibraryKey(part, states, render_pass),
            [&] { return MakeShaderLibrary(part, states, render_pass); }, name);
        if (!shader_libraries[index]) {
            return false;
        }
        any_library = true;
    }
    if (libraries_only) {
        return any_library;
    }
    const VkGraphicsPipelineLibraryCreateInfoEXT vertex_input_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
    };
    vertex_input_library = device.GetLogical().CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &vertex_input_ci,
        .flags = LibraryFlags(device),
        .stageCount = 0,
        .pStages = nullptr,
        .pVertexInputState = &states.vertex_input_ci,
        .pInputAssemblyState = &states.input_assembly_ci,
        .pTessellationState = nullptr,
        .pViewportState = nullptr,
        .pRasterizationState = nullptr,
        .pMultisampleState = nullptr,
        .pDepthStencilState = nullptr,
        .pColorBlendState = nullptr,
        .pDynamicState = &states.dynamic_state_ci,
        .layout = VK_NULL_HANDLE,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    }, *pipeline_cache);
    const VkGraphicsPipelineLibraryCreateInfoEXT fragment_output_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
        .pNext = nullptr,
        .flags = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT,
    };
    fragment_output_library = device.GetLogical().CreateGraphicsPipeline({
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &fragment_output_ci,
        .flags = LibraryFlags(device),
        .stageCount = 0,
        .pStages = nullptr,
        .pVertexInputState = nullptr,
        .pInputAssemblyState = nullptr,
        .pTessellationState = nullptr,
        .pViewportState = nullptr,
        .pRasterizationState = nullptr,
        .pMultisampleState = &states.multisample_ci,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &states.color_blend_ci,
        .pDynamicState = &states.dynamic_state_ci,
        .layout = VK_NULL_HANDLE,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = nullptr,
        .basePipelineIndex = 0,
    }, *pipeline_cache);
    pipeline =
        LinkLibraries(render_pass, GplOptimizeEnabled() || !device.CanFastLinkGraphicsLibraries());
    linked = true;
    return true;
}

void GraphicsPipeline::FallBackToMonolithic(VkRenderPass render_pass) {
    // The stages were emitted with one set per stage: keep the sets, with a layout a monolithic
    // pipeline accepts (no independent sets, no null set layouts).
    fallback_empty_set_layout = device.GetLogical().CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 0,
        .pBindings = nullptr,
    });
    std::array<VkDescriptorSetLayout, NUM_STAGES> set_layouts{};
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        set_layouts[stage] =
            stage_set_layouts[stage] ? *stage_set_layouts[stage] : *fallback_empty_set_layout;
    }
    pipeline_layout =
        DescriptorLayoutBuilder{device}.CreateIndependentPipelineLayout(set_layouts, false);
    shader_libraries = {};
    vertex_input_library.reset();
    fragment_output_library.reset();
    MakePipeline(render_pass);
}

void GraphicsPipeline::Validate() {
    size_t num_images{};
    for (const auto& info : stage_infos) {
        num_images += Shader::NumDescriptors(info.texture_buffer_descriptors);
        num_images += Shader::NumDescriptors(info.image_buffer_descriptors);
        num_images += Shader::NumDescriptors(info.texture_descriptors);
        num_images += Shader::NumDescriptors(info.image_descriptors);
    }
    ASSERT(num_images == num_image_elements);
}

} // namespace Vulkan
