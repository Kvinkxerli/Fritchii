// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>
#ifdef _WIN32
#include <windows.h> // SetPriorityClass (EXP111)
#endif
#include <chrono>
#include <thread>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <tuple>
#include <set>
#include <span>
#include <fstream>
#include <mutex>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "video_core/fritchii_occl_flag.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"

#include "common/assert.h"
#include "common/cityhash.h"
#include "common/dkb_frame_trace.h"
#include "common/fritchii_write_ring.h"
#include "common/host_memory.h"
#include "common/memory_detect.h"
#include "shader_recompiler/exception.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "common/nx2_uniform_keep_pages.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/nx2_protect_stats.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/native_query_trace.h"
#include "video_core/native_indirect_reset.h"
#include "video_core/render_trace.h"
#include "video_core/indirect_reset.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/blit_image.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/framebuffer_clear_layers.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/vk_ocarina_camera_motion.h"
#include "video_core/renderer_vulkan/vk_fsr3_upscaler.h"
#include "video_core/renderer_vulkan/vk_fsr4_upscaler.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#endif
#include "video_core/renderer_vulkan/submission_policy.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_mem_stage_shadow.h"
#include "video_core/renderer_vulkan/vk_one_follower_draw.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/link_stats.h"
#include "video_core/renderer_vulkan/vk_pcie_link.h"
#include "video_core/renderer_vulkan/vk_sim_link_hog.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/util.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/memory_manager.h"
#include "video_core/buffer_churn_stats.h"
#include "video_core/fritchii_delta_path.h"
#include "video_core/fritchii_draw_kinds.h"
#include "video_core/fritchii_fill_cuts.h"
#include "video_core/fritchii_focus.h"
#include "video_core/fritchii_report_runs.h"
#include "video_core/memory_stage_link.h"
#include "video_core/texture_handle_values.h"
#include "video_core/memstage_shadow.h"
#include "video_core/memstage_shadow_engine.h"
#include "video_core/memstage_stats.h"
#include "video_core/fritchii_report_wait.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cb_path.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_list_dispatch.h"
#include "video_core/nx2_list_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower.h"

// One follower (one_follower.h), step 1: on E (the back) every entry into the renderer runs lent
// on R, E parked; on R and every other thread it runs where it is, as today. FRITCHII_ONEF=0: one
// thread-local load and a branch per entry.
#define ONEF_LEND(op, call)                                                                        \
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {                                     \
        VideoCommon::OneFollower::SetLendSite(__func__);                                           \
        auto onef_call = [&] { call; };                                                            \
        VideoCommon::OneFollower::Lend(VideoCommon::OneFollower::LentOp::op, onef_call);           \
        return;                                                                                    \
    }
#define ONEF_LEND_RESULT(op, call)                                                                 \
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {                                     \
        VideoCommon::OneFollower::SetLendSite(__func__);                                           \
        std::optional<std::remove_cvref_t<decltype(call)>> onef_result;                            \
        auto onef_call = [&] { onef_result.emplace(call); };                                       \
        VideoCommon::OneFollower::Lend(VideoCommon::OneFollower::LentOp::op, onef_call);           \
        return std::move(*onef_result);                                                            \
    }

namespace Vulkan {

namespace {
/// FRITCHII_SPLIT_BIND_FIX (boot, default 1): a submit between a draw's or dispatch's bind and
/// its record put the record into a new command buffer with nothing bound (the pipeline, the
/// descriptor sets, the dynamic state stay in the old one). AMD's driver then reads the
/// descriptor table through the hardware's stale pointer into the old command buffer's
/// recycled memory: a GPU page fault in the 32-bit descriptor range (0xbef0xxxxxxxx). 1: the
/// draw or dispatch configures again in the new command buffer (only when the tick moved, so
/// the common path costs nothing); a played dispatch (one follower, R), which cannot configure
/// again, records its bind with the dispatch and the scheduler forgets its descriptor buffer.
/// A played or served draw never configures again either (NotePlayedSplit).
/// 0: as before (counted only).
bool SplitBindFix() {
    static const bool on = [] {
        const char* value = std::getenv("FRITCHII_SPLIT_BIND_FIX");
        return value == nullptr || value[0] != '0';
    }();
    return on;
}
std::atomic<u64> g_split_draws{};
std::atomic<u64> g_split_dispatches{};
std::atomic<u64> g_split_played_kept{};
std::atomic<u64> g_split_played_lost{};
/// FRITCHII_SPLIT_BIND_FIX for a played draw (one follower's R) or a served one (a ring B packet,
/// FRITCHII_MEMSTAGE_THREAD=2): it never configures again (its packet was played at the rest, so
/// a second Configure takes the unheld path into a buffer cache this thread does not own while
/// its owner runs: the 4080 pot room crash in BindHostStageBuffers, and a descriptor count
/// mismatch that stopped step 4 for the session). `lost`: the submit came after the bind (the
/// scheduler no longer has the pipeline bound), the draw is not recorded. Otherwise the submit
/// came before the bind (the ASTC decode inside the texture work, the common case): the bind,
/// the index and the vertex binds of the packet's geometry section are in the new command buffer.
void NotePlayedSplit(bool lost, u64 shader_hash) {
    const u64 count = (lost ? g_split_played_lost : g_split_played_kept)
                          .fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 8 || (count & (count - 1)) == 0) {
        if (lost) {
            LOG_WARNING(Render_Vulkan,
                        "Split bind on a played draw: the submit came after its bind, not drawn "
                        "(shader {:016x}; {} so far)",
                        shader_hash, count);
        } else {
            LOG_INFO(Render_Vulkan,
                     "Split bind on a played draw: the submit came before its bind, nothing to "
                     "bind again (shader {:016x}; {} so far)",
                     shader_hash, count);
        }
    }
}
void NoteSplitBind(bool dispatch, u64 shader_hash) {
    const u64 count = (dispatch ? g_split_dispatches : g_split_draws)
                          .fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 8 || (count & (count - 1)) == 0) {
        LOG_WARNING(Render_Vulkan,
                    "Split bind: a submit between the {}'s bind and its record (shader {:016x}; "
                    "{} so far; FRITCHII_SPLIT_BIND_FIX={})",
                    dispatch ? "dispatch" : "draw", shader_hash, count, SplitBindFix() ? 1 : 0);
    }
}
} // Anonymous namespace

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using VideoCommon::ImageViewId;
using VideoCommon::ImageViewType;


namespace {

// EXPERIMENT EDEN_NX2_INVALIDATE_BEFORE_DISPATCH=1: apply collected guest CPU writes before every
// dispatch and draw (=list does it once per command list, in DmaPusher).
bool InvalidateBeforeUse() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_INVALIDATE_BEFORE_DISPATCH");
        return value != nullptr && value[0] != '0' && std::string_view{value} != "list";
    }();
    return enabled;
}

/// read_gpu_memory: the first address space (the bound channel's first) that maps the whole range,
/// read as the GPU reads it, without flushing GPU caches.
VideoCore::Nx2Debug::GpuReadResult ReadGpuMemory(std::span<Tegra::MemoryManager* const> spaces,
                                                 GPUVAddr address, u64 size) {
    VideoCore::Nx2Debug::GpuReadResult result{};
    if (spaces.empty()) {
        result.error = "no GPU address space is bound yet";
        return result;
    }
    for (Tegra::MemoryManager* const space : spaces) {
        if (space->IsFullyMappedRange(address, size)) {
            result.bytes.resize(size);
            space->ReadBlockUnsafe(address, result.bytes.data(), size);
            result.address_space = space->GetID();
            return result;
        }
    }
    constexpr u64 page_mask = 0xfff;
    const GPUVAddr end = address + size;
    GPUVAddr first = address;
    while (first < end) {
        const GPUVAddr next = (std::min)((first | page_mask) + 1, end);
        if (!spaces.front()->IsFullyMappedRange(first, next - first)) {
            break;
        }
        first = next;
    }
    result.error = fmt::format("GPU address {:#x} is not mapped (the first unmapped byte in the "
                               "bound channel's address space; {} address spaces checked)",
                               first, spaces.size());
    return result;
}

struct DrawParams {
    u32 base_instance;
    u32 num_instances;
    u32 base_vertex;
    u32 num_vertices;
    u32 first_index;
    bool is_indexed;
};

VkViewport GetViewportState(const Device& device, const Maxwell& regs, size_t index, float scale) {
    const auto& src = regs.viewport_transform[index];
    const auto conv = [scale](float value) {
        float new_value = value * scale;
        if (scale < 1.0f) {
            const bool sign = std::signbit(value);
            new_value = std::round(std::abs(new_value));
            new_value = sign ? -new_value : new_value;
        }
        return new_value;
    };
    const float x = conv(src.translate_x - src.scale_x);
    const float width = conv(src.scale_x * 2.0f);
    float y = conv(src.translate_y - src.scale_y);
    float height = conv(src.scale_y * 2.0f);

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const bool y_negate = !device.IsNvViewportSwizzleSupported() &&
                          src.swizzle.y == Maxwell::ViewportSwizzle::NegativeY;

    if (lower_left) {
        // Flip by surface clip height
        y += conv(static_cast<f32>(regs.surface_clip.height));
        height = -height;
    }

    if (y_negate) {
        // Flip by viewport height
        y += height;
        height = -height;
    }

    const float reduce_z = regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1.0f : 0.0f;
    VkViewport viewport{
        .x = x,
        .y = y,
        .width = width != 0.0f ? width : 1.0f,
        .height = height != 0.0f ? height : 1.0f,
        .minDepth = src.translate_z - src.scale_z * reduce_z,
        .maxDepth = src.translate_z + src.scale_z,
    };
    if (!device.IsExtDepthRangeUnrestrictedSupported()) {
        viewport.minDepth = std::clamp(viewport.minDepth, 0.0f, 1.0f);
        viewport.maxDepth = std::clamp(viewport.maxDepth, 0.0f, 1.0f);
    }
    return viewport;
}

VkRect2D GetScissorState(const Maxwell& regs, size_t index, u32 up_scale = 1, u32 down_shift = 0) {
    const auto& src = regs.scissor_test[index];
    VkRect2D scissor{};
    const auto scale_up = [&](s32 value) -> s32 {
        if (value == 0) {
            return 0U;
        }
        const s32 upset = value * up_scale;
        s32 acumm = 0;
        if ((up_scale >> down_shift) == 0) {
            acumm = upset % 2;
        }
        const s32 converted_value = (value * up_scale) >> down_shift;
        return value < 0 ? std::min<s32>(converted_value - acumm, -1)
                         : std::max<s32>(converted_value + acumm, 1);
    };

    const bool lower_left = regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft;
    const s32 clip_height = regs.surface_clip.height;

    // Flip coordinates if lower left
    s32 min_y = lower_left ? (clip_height - src.max_y) : src.min_y.Value();
    s32 max_y = lower_left ? (clip_height - src.min_y) : src.max_y.Value();

    // Bound to render area
    min_y = (std::max)(min_y, 0);
    max_y = (std::max)(max_y, 0);

    if (src.enable) {
        scissor.offset.x = scale_up(src.min_x);
        scissor.offset.y = scale_up(min_y);
        scissor.extent.width = scale_up(src.max_x - src.min_x);
        scissor.extent.height = scale_up(max_y - min_y);
    } else {
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = (std::numeric_limits<s32>::max)();
        scissor.extent.height = (std::numeric_limits<s32>::max)();
    }
    return scissor;
}

// The part of a volume slice a clear writes: the scissor (when the clear uses it) limited to the
// render area and the slice.
VkRect2D VolumeClearRect(const Maxwell& regs, VkExtent2D render_area, VideoCommon::Extent3D slice) {
    VkRect2D rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0)
                                                   : VkRect2D{{0, 0}, {slice.width, slice.height}};
    const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
        const s64 begin = std::clamp<s64>(offset, 0, limit);
        const s64 end = std::clamp<s64>(s64{offset} + extent, begin, limit);
        offset = static_cast<s32>(begin);
        extent = static_cast<u32>(end - begin);
    };
    clamp_axis(rect.offset.x, rect.extent.width, (std::min)(render_area.width, slice.width));
    clamp_axis(rect.offset.y, rect.extent.height, (std::min)(render_area.height, slice.height));
    return rect;
}

DrawParams MakeDrawParams(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state, u32 num_instances, bool is_indexed) {
    DrawParams params{
        .base_instance = draw_state.base_instance,
        .num_instances = num_instances,
        .base_vertex = is_indexed ? draw_state.base_index : draw_state.vertex_buffer.first,
        .num_vertices = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
        .first_index = is_indexed ? draw_state.index_buffer.first : 0,
        .is_indexed = is_indexed,
    };
    // 6 triangle vertices per quad, base vertex is part of the index
    // See BindQuadIndexBuffer for more details
    if (draw_state.topology == Maxwell::PrimitiveTopology::Quads) {
        params.num_vertices = (params.num_vertices / 4) * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    } else if (draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
        params.num_vertices = (params.num_vertices - 2) / 2 * 6;
        params.base_vertex = 0;
        params.is_indexed = true;
    }
    return params;
}

bool SupportsPrimitiveRestart(VkPrimitiveTopology topology) {
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return false;
    default:
        return true;
    }
}

bool IsPrimitiveRestartSupported(const Device& device, VkPrimitiveTopology topology) {
    return ((topology != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsTopologyListPrimitiveRestartSupported()) ||
            SupportsPrimitiveRestart(topology) ||
            (topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST &&
             device.IsPatchListPrimitiveRestartSupported()));
}

/// Before a dispatch: earlier writes visible to it, its indirect arguments included.
void RecordDispatchBarrier(vk::CommandBuffer cmdbuf) {
    static constexpr VkMemoryBarrier READ_BARRIER{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
    };
    cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, READ_BARRIER);
}

/// EDEN_NX2_MEM_OWNERS (on by default, =0 off): the "Memory owners" line every 10 s.
bool MemoryOwnersEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_MEM_OWNERS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// One line splitting the working set into owners (fable_answer_26 A3). The guest residency and
/// heap numbers come from Core::DeviceMemory's MemoryProbe thread (at most 10 s old), the rest is
/// read here. All in MiB. The residual is the private working set less what the line attributes
/// to the host side; its terms are estimates and may overlap (driver heaps among the heaps, VMA's
/// host heap not all resident), so a negative residual means overlap, not an error.
/// `buffer_staging`: the buffer cache's own staging pool (FRITCHII_MEMSTAGE_SPLIT_ALLOC), counted
/// with the other in the staging numbers; null when there is one pool.
void LogMemoryOwners(const Device& device, PipelineCache& pipeline_cache,
                     StagingBufferPool& staging_pool, StagingBufferPool* buffer_staging) {
    const auto cached = [&](MemoryUsage usage) {
        return staging_pool.CachedBytes(usage) +
               (buffer_staging != nullptr ? buffer_staging->CachedBytes(usage) : 0);
    };
    const u64 stream_bytes = staging_pool.StreamBufferBytes() +
                             (buffer_staging != nullptr ? buffer_staging->StreamBufferBytes() : 0);
    constexpr u64 PipelineBytes = u64{460} << 10; // ~0.45 MB of driver memory each
    const Common::ProcessMemoryUsage process = Common::GetProcessMemoryUsage();
    const auto commit = Common::HostMemory::GetCommitStats();
    const auto resident = Common::HostMemory::LastResidentSample();
    const auto pipelines = pipeline_cache.MemoryOwners();
    const HeapTotals heaps = GetHeapTotals(device);
    const Common::HeapUsage win32_heaps = Common::LastHeapUsage();
    const u64 guest_unique =
        resident.backing_bytes + resident.view_bytes - (std::min)(resident.view_bytes,
                                                                  resident.both_bytes);
    const u64 pipeline_estimate = (pipelines.graphics_built + pipelines.compute) * PipelineBytes;
    const u64 heap_committed = win32_heaps.crt_committed + win32_heaps.other_committed;
    // Older Windows builds report no private working set: the working set less the guest then.
    const u64 private_ws = process.private_working_set != 0
                               ? process.private_working_set
                               : process.working_set - (std::min)(process.working_set, guest_unique);
    const s64 residual = static_cast<s64>(private_ws) - static_cast<s64>(pipeline_estimate) -
                         static_cast<s64>(pipelines.cache_data_bytes) -
                         static_cast<s64>(heaps.host_usage) - static_cast<s64>(heap_committed);
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    LOG_INFO(Render_Vulkan,
             "Memory owners: working set {} = private {} + shared {}, private commit {}; guest "
             "{} committed, resident {} backing + {} views - {} both = {} (views map {}, "
             "sampled in {} us); pipelines {} graphics built of {} ({} never used) + {} compute "
             "({} never dispatched), {} awaiting destruction, ~{} at 0.45 MB each, pipeline cache "
             "data {}; VMA host {} used of {} "
             "({} in blocks), device-local {} of {}; staging upload {} download {} device {} "
             "stream {}; heaps CRT {} committed ({} allocated), {} others {} committed ({} "
             "allocated, {} us); residual {} (MiB)",
             mib(process.working_set), mib(private_ws),
             mib(process.working_set - (std::min)(process.working_set, private_ws)),
             mib(process.private_bytes), mib(commit.committed_bytes),
             mib(resident.backing_bytes), mib(resident.view_bytes), mib(resident.both_bytes),
             mib(guest_unique), mib(resident.view_mapped_bytes), resident.micros,
             pipelines.graphics_built, pipelines.graphics_entries, pipelines.graphics_never_used,
             pipelines.compute, pipelines.compute_never_dispatched,
             pipelines.awaiting_destruction, mib(pipeline_estimate),
             mib(pipelines.cache_data_bytes), mib(heaps.host_usage), mib(heaps.host_budget),
             mib(heaps.host_blocks), mib(heaps.local_usage), mib(heaps.local_budget),
             mib(cached(MemoryUsage::Upload)), mib(cached(MemoryUsage::Download)),
             mib(cached(MemoryUsage::DeviceLocal)), mib(stream_bytes),
             mib(win32_heaps.crt_committed),
             mib(win32_heaps.crt_allocated), win32_heaps.heaps > 0 ? win32_heaps.heaps - 1 : 0,
             mib(win32_heaps.other_committed), mib(win32_heaps.other_allocated),
             win32_heaps.micros, residual / (s64{1} << 20));
}

/// An image view's size in GPU memory: the game's size, or the resolution scale's size once the
/// texture cache scaled the image (OoT's temporal upscale with FRITCHII_DLSS_SCALED).
VkExtent2D RealExtent(const ImageView& view) {
    if (!view.IsRescaled()) {
        return {view.size.width, view.size.height};
    }
    const auto& resolution = Settings::values.resolution_info;
    return {resolution.ScaleUp(view.size.width), resolution.ScaleUp(view.size.height)};
}
} // Anonymous namespace

void RasterizerVulkan::ReadMeshEmulationProbe() {
    const MeshEmulationProbe probe{mesh_probe};
    mesh_probe.pending = false;
    const size_t num_words{probe.data.size() / sizeof(u32)};
    const auto word = [&](u64 index) -> u32 {
        if (index >= num_words) {
            return 0;
        }
        u32 value;
        std::memcpy(&value, probe.data.data() + index * sizeof(u32), sizeof(value));
        return value;
    };
    const auto position = [&](u64 base, u32 vertex) {
        std::array<f32, 4> result{};
        for (u32 element = 0; element < 4; ++element) {
            const u32 line{probe.position_lines[element]};
            if (line != ~0U) {
                result[element] = std::bit_cast<f32>(word(base + u64{vertex} * probe.vertex_lines +
                                                          line));
            }
        }
        return result;
    };
    u64 triangles{}, degenerate{}, outside{}, bad_index{};
    std::string sample;
    for (u32 group = 0; group < probe.groups; ++group) {
        const u64 base{u64{group} * probe.block_words};
        const u64 index_base{base + u64{probe.max_vertices} * probe.vertex_lines};
        for (u32 primitive = 0; primitive < probe.max_primitives; ++primitive) {
            const std::array<u32, 3> indices{word(index_base + primitive * 3),
                                             word(index_base + primitive * 3 + 1),
                                             word(index_base + primitive * 3 + 2)};
            if (indices[0] == indices[1] && indices[1] == indices[2]) {
                ++degenerate;
                continue;
            }
            ++triangles;
            bool any_bad{};
            bool any_inside{};
            for (const u32 vertex : indices) {
                if (vertex >= probe.max_vertices) {
                    any_bad = true;
                    continue;
                }
                const auto p{position(base, vertex)};
                const f32 w{std::abs(p[3])};
                any_inside |= w > 0.0f && std::abs(p[0]) <= w && std::abs(p[1]) <= w;
            }
            bad_index += any_bad;
            outside += !any_inside;
            if (sample.size() < 400 && triangles <= 2) {
                for (const u32 vertex : indices) {
                    const auto p{position(base, std::min(vertex, probe.max_vertices - 1))};
                    sample += fmt::format(" g{} v{} ({:.3g} {:.3g} {:.3g} {:.3g})", group, vertex,
                                          p[0], p[1], p[2], p[3]);
                }
            }
        }
    }
    LOG_INFO(Render_Vulkan,
             "Mesh emulation probe frame {}: {} groups, {} triangles ({} degenerate slots), {} "
             "with every vertex off screen, {} with a bad index; first triangles:{}",
             probe.frame, probe.groups, triangles, degenerate, outside, bad_index, sample);
}

const RasterizerVulkan::MeshArena& RasterizerVulkan::AcquireMeshArena() {
    const u64 tick{scheduler.CurrentTick()};
    if (mesh_arena_current < mesh_arenas.size() && mesh_arenas[mesh_arena_current].tick == tick) {
        return mesh_arenas[mesh_arena_current];
    }
    // A finished submission's counter holds the words its draws asked for, including those
    // that did not fit: grow later arenas to hold them.
    const u64 max_size{(std::min)(u64{1} << 30, u64{device.GetMaxStorageBufferRange()})};
    for (MeshArena& arena : mesh_arenas) {
        if (arena.counted || !scheduler.IsFree(arena.tick)) {
            continue;
        }
        arena.counted = true;
        arena.counter.Invalidate();
        u32 words;
        std::memcpy(&words, arena.counter.Mapped().data(), sizeof(words));
        const u64 needed{u64{words} * 4};
        if (needed > arena.size && mesh_arena_size < max_size) {
            mesh_arena_size = (std::min)(std::bit_ceil(needed + needed / 4), max_size);
            LOG_INFO(Render_Vulkan, "Mesh emulation: a submission needed {} MiB of output, arenas "
                                    "grow to {} MiB",
                     needed >> 20, mesh_arena_size >> 20);
        }
    }
    std::erase_if(mesh_arenas, [&](const MeshArena& arena) {
        return arena.counted && arena.size < mesh_arena_size;
    });
    auto it{std::ranges::find_if(mesh_arenas, [](const MeshArena& arena) { return arena.counted; })};
    if (it == mesh_arenas.end()) {
        const auto make_buffer = [&](u64 size, VkBufferUsageFlags usage, MemoryUsage memory) {
            return memory_allocator.CreateBuffer(
                VkBufferCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .size = size,
                    .usage = usage,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                },
                memory);
        };
        it = mesh_arenas.insert(
            mesh_arenas.end(),
            MeshArena{
                .buffer = make_buffer(mesh_arena_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      MemoryUsage::DeviceLocal),
                .counter = make_buffer(sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       MemoryUsage::Download),
                .size = mesh_arena_size,
            });
    }
    // The GPU is done with this arena, and this submission's commands have not been sent yet.
    const u32 zero{};
    std::memcpy(it->counter.Mapped().data(), &zero, sizeof(zero));
    it->counter.Flush();
    it->tick = tick;
    it->counted = false;
    mesh_arena_current = static_cast<size_t>(std::distance(mesh_arenas.begin(), it));
    return *it;
}

void RasterizerVulkan::ReleaseMeshEmulationBuffers() {
    for (StagingBufferRef& ref : mesh_draw_buffers) {
        staging_pool.FreeDeferred(ref);
    }
    mesh_draw_buffers.clear();
    if (mesh_arena_used) {
        mesh_arena_used = false;
        mesh_arenas[mesh_arena_current].tick = scheduler.CurrentTick();
    }
}

RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,
                                   Tegra::MaxwellDeviceMemoryManager& device_memory_,
                                   const Device& device_, MemoryAllocator& memory_allocator_,
                                   StateTracker& state_tracker_, Scheduler& scheduler_)
    : gpu{gpu_}, device_memory{device_memory_}, device{device_},
      memory_allocator{memory_allocator_}, state_tracker{state_tracker_}, scheduler{scheduler_},
      staging_pool(device, memory_allocator, scheduler,
                   device.HasSplitAllocators() && device.IsIntegrated() ? 2 : 1,
                   device.HasSplitAllocators() ? StagingBufferPool::StreamRole::Transfers
                                               : StagingBufferPool::StreamRole::Everything),
      split_buffer_allocator{device.HasSplitAllocators()
                                 ? std::make_unique<MemoryAllocator>(device, true)
                                 : nullptr},
      split_buffer_staging{split_buffer_allocator ? std::make_unique<StagingBufferPool>(
                                                        device, *split_buffer_allocator,
                                                        scheduler, 2,
                                                        StagingBufferPool::StreamRole::Buffers)
                                                  : nullptr},
      buffer_memory_allocator{split_buffer_allocator ? *split_buffer_allocator
                                                     : memory_allocator},
      buffer_staging_pool{split_buffer_staging ? *split_buffer_staging : staging_pool},
      descriptor_pool(device, scheduler),
      guest_descriptor_queue(device, UpdateDescriptorQueue::GUEST_FRAME_PAYLOAD_SIZE,
                             device.IsExtDescriptorBufferSupported()),
      compute_pass_descriptor_queue(device, UpdateDescriptorQueue::COMPUTE_FRAME_PAYLOAD_SIZE),
      descriptor_buffer_ring(device, memory_allocator),
      blit_image(device, scheduler, state_tracker, descriptor_pool), render_pass_cache(device),
      texture_cache_runtime{
          device,     scheduler,         memory_allocator, staging_pool,
          blit_image, render_pass_cache, descriptor_pool,  compute_pass_descriptor_queue},
      texture_cache(texture_cache_runtime, device_memory),
      mem_player(device, memory_allocator, scheduler, guest_descriptor_queue,
                 compute_pass_descriptor_queue),
      mem_stage(mem_player), mem_clock(scheduler, &mem_stage),
      buffer_cache_runtime(device, buffer_memory_allocator, scheduler, buffer_staging_pool,
                           guest_descriptor_queue, compute_pass_descriptor_queue, descriptor_pool,
                           mem_stage, mem_clock),
      buffer_cache(device_memory, buffer_cache_runtime),
      query_cache_runtime(this, device_memory, buffer_cache, device, memory_allocator, scheduler,
                          staging_pool, compute_pass_descriptor_queue, descriptor_pool, texture_cache),
      query_cache(gpu, *this, device_memory, query_cache_runtime),
      pipeline_cache(device_memory, device, scheduler, descriptor_pool, guest_descriptor_queue,
                     descriptor_buffer_ring, render_pass_cache, buffer_cache, texture_cache,
                     gpu.ShaderNotify()),
      accelerate_dma(buffer_cache, texture_cache, scheduler),
      fence_manager(*this, gpu, texture_cache, buffer_cache, query_cache, device, scheduler),
      frame_dump(device, memory_allocator, scheduler),
      flash_watch(device, memory_allocator, scheduler) {
    // FRITCHII_LOCK_CENSUS: the two cache locks by name (before any other thread uses them).
    // A new session: OoT's picture is published again when its upscaler is picked.
    VideoCommon::ClearOcarinaPicture();
    buffer_cache.mutex.SetCensusName("buffer cache");
    texture_cache.mutex.SetCensusName("texture cache");
    scheduler.SetQueryCache(query_cache);
    // The memory seam's read log asks this cache's tracker.
    mem_stage.Attach(buffer_cache);
    mem_stage.AttachGpu(gpu);
    mem_stage.AttachDeviceMemory(device_memory);
    if (split_buffer_staging) {
        // The memory stage's thread checks who requests from the cache's own pool.
        split_buffer_staging->MarkCacheSide();
    }
    // FRITCHII_STAGING_RING: the texture cache's pool (texture uploads at area loads).
    staging_pool.CreateUploadRing();
    if (!device.IsIntegrated()) {
        LogPcieLink(device.GetVendorID(), device.GetDeviceID());
    }
    // FRITCHII_SIM_LINK_HOG: a copy load on the transfer queue (test).
    if (device.HasLinkHogQueue()) {
        sim_link_hog = std::make_unique<SimLinkHog>(device, memory_allocator);
    }
    VideoCommon::FritchiiDrawKinds::Init();
    // EDEN_NX2_MEMSTAGE_STATS: stream writes are tested against the texture cache's pages.
    VideoCommon::MemStageStats::SetImagePageFilters(&texture_cache.page_filter,
                                                    &texture_cache.image_count_filter);
    // FRITCHII_MEMSTAGE_SHADOW: the watched pages of join 2 (image pages).
    VideoCommon::MemStageShadow::SetWatchFilter(&texture_cache.page_filter);
    // One follower: the stream-write guard's pages (section 7), and the locks its hang watchdog
    // names the holders of.
    VideoCommon::OneFollower::SetTexturePages(&texture_cache.page_filter);
    // fable_b 14.4 cut 5: the four filters E's invalidations test also mark one union.
    texture_cache.page_filter.AlsoMark(&onef_union_pages);
    pipeline_cache.page_filter.AlsoMark(&onef_union_pages);
    query_cache.page_filter.AlsoMark(&onef_union_pages);
    VideoCommon::OneFollower::SetUnionPages(&onef_union_pages); // CB_LEAN & 2's guard
    VideoCommon::OneFollower::SetShaderPages(&pipeline_cache.page_filter);
    VideoCommon::OneFollower::WatchLock("buffer cache", &buffer_cache.mutex);
    VideoCommon::OneFollower::WatchLock("texture cache", &texture_cache.mutex);
    buffer_cache.SetPendingCpuWrites(
        [this](const std::function<void(DAddr, size_t)>& callback) {
            gpu.ForEachPendingCpuWrite(callback);
        });
    VideoCommon::GpuSplit::Get().SetIdleHook(
        [](void* argument) {
            auto* const rasterizer = static_cast<RasterizerVulkan*>(argument);
            // One follower: the hook is R's work (the scheduler, the pools), lent with the idle
            // state E had (InIdleWait is E's thread-local).
            const bool idle = VideoCommon::GpuSplit::InIdleWait();
            if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
                auto run = [rasterizer, idle] { rasterizer->RunIdleHook(idle); };
                VideoCommon::OneFollower::Lend(VideoCommon::OneFollower::LentOp::FrameEnd, run);
                return;
            }
            rasterizer->RunIdleHook(idle);
        },
        this);
    SetRingBackend();
    if (split_buffer_allocator) {
        LOG_INFO(Render_Vulkan,
                 "Memory stage split allocators on (FRITCHII_MEMSTAGE_SPLIT_ALLOC=0 turns it off): "
                 "buffer allocator owns the buffer cache's buffers (EXP104 pool, quad and uint8 "
                 "index, vertex repack, multi-range, list and null buffers) and its staging pool "
                 "(stream {} MiB, uniform stream and buffer uploads/downloads); main allocator owns "
                 "images, the texture cache, query cache, descriptors, mesh emulation, upscalers "
                 "and their staging pool (stream {} MiB)",
                 buffer_staging_pool.StreamBufferBytes() >> 20,
                 staging_pool.StreamBufferBytes() >> 20);
    } else {
        LOG_INFO(Render_Vulkan, "Memory stage split allocators off: one allocator and one staging "
                                "pool for everything");
    }
    VideoCommon::GpuSplit::Get().SetFenceQuietCheck(
        [](void* argument, bool strict, u64* epoch) {
            return static_cast<RasterizerVulkan*>(argument)->FencesQuiet(strict, *epoch);
        },
        this);
    runahead_hooks = VideoCommon::Runahead::Hooks{
        .argument = this,
        .guest_pointer = [](void* argument, u64 device) -> const u8* {
            return static_cast<RasterizerVulkan*>(argument)
                ->device_memory.GetPointer<u8>(static_cast<DAddr>(device));
        },
        .may_hold_images =
            [](void* argument, u64 device, u64 size) {
                return static_cast<RasterizerVulkan*>(argument)->texture_cache.page_filter.Any(
                    static_cast<DAddr>(device), size);
            },
        .apply =
            [](void* argument, const VideoCommon::Runahead::Chunk& chunk) {
                static_cast<RasterizerVulkan*>(argument)->Nx2ApplyRunaheadChunk(chunk);
            },
        .bound_programs =
            [](void* argument) {
                // The bound channel's engine, not `maxwell3d`: with one follower at step 4 that
                // names R's view engine while R plays a draw, and this runs on E (ApplyDue).
                const auto* const rasterizer = static_cast<RasterizerVulkan*>(argument);
                if (rasterizer->channel_state == nullptr) {
                    return std::array<u64, 2>{};
                }
                const auto& regs = rasterizer->channel_state->maxwell3d.regs;
                return std::array<u64, 2>{
                    (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) |
                        regs.pipelines[1].reserved[1],
                    (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) |
                        regs.pipelines[5].reserved[1],
                };
            },
    };
    VideoCommon::Runahead::SetHooks(&runahead_hooks);
    // One follower, part c: R's draw player (idle below step 3).
    onef_draw = std::make_unique<OneFollowerDraw>(*this);
    // FRITCHII_ESPLIT: a samples reset E1 queues for E2 clears E1's copy of the mirror at once.
    VideoCommon::ESplit::g_mirror_reset = [](VideoCore::RasterizerInterface* rasterizer) {
        static_cast<RasterizerVulkan*>(rasterizer)->onef_mirror_queued = false;
    };
    VideoCommon::OneFollower::SetDrawPlayer(onef_draw.get());
}

void RasterizerVulkan::OneFollowerLendDraw(bool is_indexed, u32 instance_count) {
    namespace OneF = VideoCommon::OneFollower;
    // From step 3 part (c) routes E's draws (lent with the verify, or filled into ring B and
    // pushed at step 4): vk_one_follower_draw.cpp.
    if (OneF::Step() >= OneF::StepLevel::Build &&
        onef_draw->RouteDraw(is_indexed, instance_count)) {
        return;
    }
    // Steps 1 and 2: every draw lent. The draw's kind, from E's engine (UnlendKind), for the
    // counters and the draw span. The channel's engine, as everywhere on E (`maxwell3d` names R's
    // view while R plays a draw at step 4).
    const Tegra::Engines::Maxwell3D& engine = channel_state->maxwell3d;
    const auto& draw_state = engine.draw_manager.draw_state;
    using DrawMode = Tegra::Engines::Maxwell3D::DrawManager::DrawMode;
    u32 kind;
    if (!is_indexed && Tegra::Nx2IndirectGrid::AnyLive() &&
        Tegra::Nx2IndirectGrid::LooksLikeDrawPlaceholder(draw_state.vertex_buffer.count)) {
        kind = OneF::GpuCount;
    } else if (engine.regs.IsMeshEnabled()) {
        kind = OneF::Mesh;
    } else if (draw_state.draw_mode == DrawMode::InlineIndex ||
               !draw_state.inline_index_draw_indexes.empty()) {
        kind = OneF::InlineIndex;
    } else if (draw_state.draw_mode == DrawMode::Instance) {
        kind = OneF::InstanceLoop;
    } else {
        kind = is_indexed ? OneF::PlainIndexed : OneF::PlainArrays;
    }
    OneF::SetNextDraw(kind, is_indexed, false, instance_count);
    auto call = [this, is_indexed, instance_count] { Draw(is_indexed, instance_count); };
    OneF::Lend(OneF::LentOp::Draw, OneF::LentReason::StepBelowUnlend, call);
}

void RasterizerVulkan::OneFollowerDrawIndirect() {
    namespace OneF = VideoCommon::OneFollower;
    // E's engine is the channel's: at step 4 `maxwell3d` names R's view while R plays a draw.
    const auto& indirect = channel_state->maxwell3d.draw_manager.indirect_state;
    // Step 4 with the List bit: list draws are filled into ring B and pushed, or lent with their
    // reason and the verify (vk_one_follower_draw.cpp). Without the bit, as before.
    if (indirect.list_words && onef_draw->RouteListDraw()) {
        return;
    }
    const u32 kind = indirect.list_words ? OneF::List : OneF::Indirect;
    // Indirect draws have no R path in this branch: lent at every step (their mask bit asks for
    // nothing yet). So are list draws without the List bit.
    const OneF::LentReason reason = OneF::Step() < OneF::StepLevel::Unlend
                                        ? OneF::LentReason::StepBelowUnlend
                                        : OneF::LentReason::KindNotInMask;
    OneF::SetNextDraw(kind, indirect.is_indexed, true, 0);
    auto call = [this] { DrawIndirect(); };
    OneF::Lend(OneF::LentOp::Draw, reason, call);
}

void RasterizerVulkan::RunIdleHook(bool idle) {
    scheduler.FlushDeferred();
    // EXP99 = 1: evicted pool images (OoT's per-NPC depth targets) go while idle.
    // Only when it ran out of work: a RequestIdleHook mid-batch has a waiter (Fable).
    if (VideoCore::Nx2Debug::Experiment(99) == 1 && idle) {
        texture_cache_runtime.DestroyIdlePooledImages(2);
    }
    // EXP104: evicted pool buffers go the same way, whatever EXP104 is now (switching it
    // off evicts everything). With FRITCHII_MEMSTAGE_SPLIT_ALLOC the buffer cache's
    // recycler first, then the back's: still at most 2 a call. The first is the buffer
    // allocator's, so it runs on the cache's thread (memory stage spec 5.2: GPUMem by
    // hand-off at FRITCHII_MEMSTAGE_THREAD=1); the second stays on the back.
    if (idle) {
        BufferRecycler& buffers = buffer_staging_pool.Recycler();
        BufferRecycler& others = staging_pool.Recycler();
        if (buffers.HasDoomed()) {
            mem_stage.CacheIdle([&buffers] { buffers.DestroyIdle(2); });
        } else if (others.HasDoomed()) {
            others.DestroyIdle(2);
        }
    }
}

RasterizerVulkan::~RasterizerVulkan() {
    VideoCommon::OneFollower::SetDrawPlayer(nullptr);
    VideoCommon::Runahead::SetHooks(nullptr);
    VideoCommon::GpuSplit::Get().SetIdleHook(nullptr, nullptr);
    VideoCommon::GpuSplit::Get().SetFenceQuietCheck(nullptr, nullptr);
    scheduler.WaitWorker();
    scheduler.Finish();
}

// Experimental adaptation of a ported change: execute the reset shader's
// indirect record writes on the host until general GPU-address translation is available.
bool RasterizerVulkan::IsNx2IndirectReset(bool is_indexed, u32 instance_count) {
    return IsNx2IndirectResetOn(*maxwell3d, *gpu_memory, is_indexed, instance_count);
}

bool RasterizerVulkan::IsNx2IndirectResetOn(Tegra::Engines::Maxwell3D& engine,
                                            Tegra::MemoryManager& memory, bool is_indexed,
                                            u32 instance_count) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_RESETFIX");
        return !value || std::string_view{value} != "0";
    }();
    const auto& regs = engine.regs;
    const auto& draw = engine.draw_manager.draw_state;
    if (!enabled || is_indexed || instance_count != 1 || regs.IsMeshEnabled() ||
        draw.topology != Maxwell::PrimitiveTopology::Points ||
        !engine.GetNativeMacroEngine().native_turing) {
        return false;
    }
    const GPUVAddr program =
        (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    std::array<u32, VideoCommon::NativeIndirectReset::Program.size()> code{};
    if (!memory.IsFullyMappedRange(program + 128, sizeof(code))) {
        return false;
    }
    memory.ReadBlock(program + 128, code.data(), sizeof(code));
    return VideoCommon::NativeIndirectReset::Matches(code);
}

bool RasterizerVulkan::TryNx2IndirectReset(bool is_indexed, u32 instance_count) {
    if (!IsNx2IndirectReset(is_indexed, instance_count)) {
        return false;
    }
    VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::IndirectReset);
    const auto& draw = maxwell3d->draw_manager.draw_state;
    FlushWork();
    gpu_memory->FlushCaching();
    const auto reject = [](std::string_view reason) {
        static u32 reported_rejections{};
        if (reported_rejections++ < 8) {
            LOG_WARNING(HW_GPU, "Experimental indirect reset skipped: {}", reason);
        }
        return false;
    };
    const auto in_gpu_range = [this](u64 address, size_t size) {
        return size != 0 && address <= (std::numeric_limits<u64>::max)() - size &&
               gpu_memory->IsWithinGPUAddressRange(address) &&
               gpu_memory->IsWithinGPUAddressRange(address + size - 1);
    };
    const auto& cbuf = maxwell3d->state.shader_stages[0].const_buffers[0];
    if (!cbuf.enabled || cbuf.size < 0x120 || !in_gpu_range(cbuf.address, 0x120) ||
        !gpu_memory->IsFullyMappedRange(cbuf.address + 0x110, 16)) {
        return reject("descriptor unavailable");
    }
    std::array<u64, 2> descriptor{};
    gpu_memory->ReadBlock(cbuf.address + 0x110, descriptor.data(), sizeof(descriptor));
    const u64 list = descriptor[0];
    const u64 first = draw.vertex_buffer.first;
    const u64 count = draw.vertex_buffer.count;
    const auto range = VideoCommon::IndirectResetRange(list, static_cast<u32>(descriptor[1]),
                                                        first, count);
    if (!range || !in_gpu_range(range->begin, range->size) ||
        !gpu_memory->IsFullyMappedRange(range->begin, range->size)) {
        return reject("list outside mapped descriptor");
    }
    std::vector<u64> pointers(count);
    gpu_memory->ReadBlock(range->begin, pointers.data(), range->size);
    static constexpr std::array<u32, 5> record{0, 1, 0, 0, 0};
    for (const u64 pointer : pointers) {
        if (!pointer) {
            return reject("record is null");
        }
        if (!in_gpu_range(pointer, sizeof(record)) ||
            !gpu_memory->IsFullyMappedRange(pointer, sizeof(record))) {
            return reject("record not mapped");
        }
        if (pointer < range->begin + range->size && range->begin < pointer + sizeof(record)) {
            return reject("record overlaps pointer list");
        }
        const auto segments = gpu_memory->GetSubmappedRange(pointer, sizeof(record));
        if (segments.size() != 1 || segments[0].first != pointer ||
            segments[0].second != sizeof(record)) {
            return reject("record crosses a mapping boundary");
        }
    }
    const std::span<const u8> bytes{reinterpret_cast<const u8*>(record.data()), sizeof(record)};
    for (const u64 pointer : pointers) {
        AccelerateInlineToMemory(pointer, bytes.size(), bytes);
    }
    static u32 reported{};
    if (reported++ < 5) {
        LOG_INFO(HW_GPU, "Experimental indirect reset: {} records", count);
    }
    gpu.TickWork();
    return true;
}

namespace {

// EDEN_NX2_FASTDRAW_COUNT: the original handle masks of a pipeline (GraphicsPipeline's constructor
// there), built from its shader info on the first counted draw that needs them.
const VideoCommon::Nx2FastDrawCount::PipelineInfo& FastDrawCountInfo(
    const GraphicsPipeline& pipeline) {
    using Info = VideoCommon::Nx2FastDrawCount::PipelineInfo;
    if (pipeline.fast_draw_count_info) {
        return *pipeline.fast_draw_count_info;
    }
    auto info = std::make_unique<Info>();
    const auto& stage_infos = pipeline.StageInfos();
    const auto any_written = [](const auto& descriptors) {
        return std::ranges::any_of(descriptors,
                                   [](const auto& descriptor) { return descriptor.is_written; });
    };
    info->writes_images = std::ranges::any_of(stage_infos, [&](const Shader::Info& stage_info) {
        return any_written(stage_info.image_descriptors) ||
               any_written(stage_info.image_buffer_descriptors);
    });
    info->mesh = pipeline.IsMeshPipeline();
    for (size_t stage = 0; stage < (std::min)(stage_infos.size(), Info::NumStages); ++stage) {
        const Shader::Info& stage_info = stage_infos[stage];
        u32& handles = info->handle_cbufs[stage];
        auto& ranges = info->ranges[stage];
        std::array<std::vector<std::pair<u32, u32>>, Info::NumSlots> words;
        const auto add_range = [&](u32 slot, u32 begin, u32 end) {
            if (slot >= Info::NumSlots) {
                return;
            }
            handles |= 1U << slot;
            auto& range = ranges[slot];
            range = range.first == range.second
                        ? std::pair{begin, end}
                        : std::pair{(std::min)(range.first, begin), (std::max)(range.second, end)};
            words[slot].emplace_back(begin, end);
        };
        // The words a handle is read from. Handles computed at draw time take no part
        // (e05a72847e: they are evaluated again for each delta draw, not counted here).
        const auto add = [&](const auto& descriptor) {
            if (descriptor.handle_program != ~0U) {
                return;
            }
            const u32 last = ((std::max)(descriptor.count, 1U) - 1) << descriptor.size_shift;
            add_range(descriptor.cbuf_index, descriptor.cbuf_offset,
                      descriptor.cbuf_offset + last + 4);
            if constexpr (requires { descriptor.has_secondary; }) {
                if (descriptor.has_secondary) {
                    add_range(descriptor.secondary_cbuf_index, descriptor.secondary_cbuf_offset,
                              descriptor.secondary_cbuf_offset + 4);
                }
            }
        };
        for (const auto& descriptor : stage_info.texture_buffer_descriptors) {
            add(descriptor);
        }
        for (const auto& descriptor : stage_info.image_buffer_descriptors) {
            add(descriptor);
        }
        for (const auto& descriptor : stage_info.texture_descriptors) {
            add(descriptor);
        }
        for (const auto& descriptor : stage_info.image_descriptors) {
            add(descriptor);
        }
        for (u32 slot = 0; slot < words.size(); ++slot) {
            auto& list = words[slot];
            std::ranges::sort(list);
            const size_t first = info->words.size();
            for (const auto& range : list) {
                if (info->words.size() != first && range.first <= info->words.back().second) {
                    info->words.back().second =
                        (std::max)(info->words.back().second, range.second);
                } else {
                    info->words.push_back(range);
                }
            }
            info->word_index[stage][slot] = {static_cast<u16>(first),
                                             static_cast<u16>(info->words.size() - first)};
        }
    }
    pipeline.fast_draw_count_info = std::move(info);
    return *pipeline.fast_draw_count_info;
}

} // Anonymous namespace

void RasterizerVulkan::FastDrawCountBegin() {
    namespace Count = VideoCommon::Nx2FastDrawCount;
    Count::Counter& counter = fast_draw_count;
    // The draw before returned before its decision (no pipeline, a skipped mesh draw).
    counter.stats.not_drawn += counter.open ? 1 : 0;
    ++counter.stats.draws;
    // The pools the writes applied from here on are tested against (Fable plan D1, row 1).
    maxwell3d->UpdateFastDrawCountPools();
    counter.taken = std::exchange(counter.previous, Count::PreviousDraw{});
    counter.taken_delta = std::exchange(Count::g_delta, Count::Delta{});
    counter.take_serial = Count::g_serial.load(std::memory_order_relaxed);
    counter.take_last_break = Count::g_last_break.load(std::memory_order_relaxed);
    counter.vertex_input_dirty = maxwell3d->dirty.flags[Dirty::VertexInput];
    counter.open = true;
}

void RasterizerVulkan::FastDrawCountDecide(const GraphicsPipeline* pipeline, bool is_indexed,
                                           bool is_indirect) {
    namespace Count = VideoCommon::Nx2FastDrawCount;
    Count::Counter& counter = fast_draw_count;
    Count::Stats& stats = counter.stats;
    counter.open = false;
    // What the next draw compares with. The groups broken since the last snapshot are this
    // draw's; a widened write list is noted with them (not a break in the original). The mask is taken
    // before the serial: a break on another thread in between (BreakOn moves the serial, then
    // sets its group) keeps its group for the next draw.
    const u32 mask = Count::g_pending.exchange(0, std::memory_order_relaxed) |
                     (counter.taken_delta.widened ? u32{Count::ListOverflow} : 0U);
    const u64 serial = Count::g_serial.load(std::memory_order_relaxed);
    const u64 map_serial = gpu_memory->MapSerial();
    counter.draw_serial = serial;
    counter.draw_map_serial = map_serial;
    const Count::PreviousDraw& draw = counter.taken;
    const auto full = [&stats](Count::Full reason) { ++stats.full[static_cast<size_t>(reason)]; };
    // TakeReusableDraw's checks, in its order.
    if (!draw.valid) {
        return full(Count::Full::NoCandidate);
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (is_indirect || draw.is_indexed != is_indexed || draw.maxwell3d != maxwell3d ||
        draw.topology != static_cast<u32>(draw_state.topology) ||
        !draw_state.inline_index_draw_indexes.empty() || frame_dump.Capturing()) {
        return full(Count::Full::Kind);
    }
    if (draw.serial != counter.take_serial) {
        return counter.CountStateChanged(mask, counter.take_last_break);
    }
    if (draw.map_serial != map_serial) {
        return counter.CountStateChanged(mask | Count::Map,
                                         static_cast<u32>(Count::Reason::Map));
    }
    if (draw.tick != scheduler.CurrentTick()) {
        return full(Count::Full::Submitted);
    }
    const auto* const previous = static_cast<const GraphicsPipeline*>(draw.pipeline);
    if (draw.render_pass != scheduler.RenderPassSerial() || !scheduler.IsRenderPassActive() ||
        scheduler.BoundGraphicsPipeline() != previous) {
        return full(Count::Full::RenderPass);
    }
    const Count::PipelineInfo& info = FastDrawCountInfo(*previous);
    if (info.writes_images || info.mesh) {
        return full(Count::Full::DeltaPipeline);
    }
    // The original compares the attribute type classes in the previous pipeline's key with the
    // registers'; our keys keep only the types the programs read (KeyedAttributeTypes), so the
    // full resolve's pipeline stands for that comparison here.
    if (counter.vertex_input_dirty &&
        (!previous->HasDynamicVertexInput() || pipeline != previous)) {
        return full(Count::Full::VertexInput);
    }
    const Count::Delta& delta = counter.taken_delta;
    for (size_t stage = 0; stage < Count::PipelineInfo::NumStages && !delta.Empty(); ++stage) {
        if ((delta.binds[stage] & info.handle_cbufs[stage]) != 0) {
            return full(Count::Full::HandleBind);
        }
        if (delta.num_writes == 0) {
            continue;
        }
        const auto& cbufs = maxwell3d->state.shader_stages[stage].const_buffers;
        for (u32 slots = info.handle_cbufs[stage]; slots != 0; slots &= slots - 1) {
            const u32 slot = static_cast<u32>(std::countr_zero(slots));
            const GPUVAddr address = cbufs[slot].address;
            const auto [begin, end] = info.ranges[stage][slot];
            if (delta.Reaches(address + begin, end - begin) &&
                delta.ReachesWords(address, info.Words(stage, slot))) {
                return full(Count::Full::HandleData);
            }
        }
    }
    // DeltaStillHolds: nothing broke between the take and the locked caches.
    if (serial != counter.take_serial) {
        return counter.CountStateChanged(mask,
                                         Count::g_last_break.load(std::memory_order_relaxed));
    }
    // Its last two checks, on the draw the previous pipeline kept (Fable's port plan step D2;
    // counting keeps every draw, PrepareDraw). A kept image that a resolve would upload or copy
    // a newer alias into, or a handle program that gives another handle now: resolved in full.
    const KeptDraw* const kept = previous->KeptLastDraw();
    if (kept == nullptr) {
        return full(Count::Full::NoKeptDraw);
    }
    if (texture_cache.SampledViewsChange(kept->views)) {
        return full(Count::Full::SampledImages);
    }
    if (!kept->computed.empty() && previous->ComputedHandlesChanged(*maxwell3d, *gpu_memory)) {
        return full(Count::Full::ComputedHandles);
    }
    ++stats.delta;
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        VideoCommon::FritchiiDrawKinds::NoteWouldBeDelta();
    }
    // The full resolve picked another pipeline: a delta draw would have kept the wrong one.
    stats.other_pipeline += pipeline != previous ? 1 : 0;
}

void RasterizerVulkan::FastDrawCountResolved(const GraphicsPipeline* pipeline, bool is_indexed,
                                             bool is_indirect) {
    namespace Count = VideoCommon::Nx2FastDrawCount;
    Count::Counter& counter = fast_draw_count;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto topology = draw_state.topology;
    const auto ineligible = [&]() -> std::optional<Count::Ineligible> {
        if (is_indirect) {
            return Count::Ineligible::Indirect;
        }
        if (mesh_draw) {
            return Count::Ineligible::Mesh;
        }
        if (!pipeline->IsBuilt()) {
            return Count::Ineligible::NotBuilt;
        }
        if (FastDrawCountInfo(*pipeline).writes_images) {
            return Count::Ineligible::WritesImages;
        }
        if (maxwell3d->regs.transform_feedback_enabled != 0) {
            return Count::Ineligible::TransformFeedback;
        }
        if (!draw_state.inline_index_draw_indexes.empty()) {
            return Count::Ineligible::InlineIndex;
        }
        if (topology == Maxwell::PrimitiveTopology::Quads ||
            topology == Maxwell::PrimitiveTopology::QuadStrip) {
            return Count::Ineligible::Quads;
        }
        if (back_repack_mask != 0) {
            return Count::Ineligible::Repacked;
        }
        if (!scheduler.IsRenderPassActive()) {
            return Count::Ineligible::NoRenderPass;
        }
        if (scheduler.BoundGraphicsPipeline() != pipeline) {
            return Count::Ineligible::PipelineNotBound;
        }
        if (!VideoCommon::GpuSplit::OnBackThread()) {
            return Count::Ineligible::OffBackThread;
        }
        return std::nullopt;
    }();
    if (ineligible) {
        ++counter.stats.ineligible[static_cast<size_t>(*ineligible)];
        return;
    }
    counter.previous = Count::PreviousDraw{
        .pipeline = pipeline,
        .maxwell3d = maxwell3d,
        .serial = counter.draw_serial,
        .map_serial = counter.draw_map_serial,
        .tick = scheduler.CurrentTick(),
        .render_pass = scheduler.RenderPassSerial(),
        .topology = static_cast<u32>(topology),
        .engine_state = static_cast<u32>(maxwell3d->engine_state),
        .is_indexed = is_indexed,
        .valid = true,
    };
}

GraphicsPipeline* RasterizerVulkan::DeltaPathDecide(bool is_indexed, bool is_indirect) {
    namespace Count = VideoCommon::Nx2FastDrawCount;
    using Fallback = VideoCommon::FritchiiDeltaPath::Fallback;
    VideoCommon::FritchiiDeltaPath::Stats& stats = delta_path;
    ++stats.draws;
    delta_vertex_moved = false;
    const auto fall = [&stats](Fallback reason) -> GraphicsPipeline* {
        stats.Fall(reason);
        return nullptr;
    };
    const Count::Counter& counter = fast_draw_count;
    // The count's tests (FastDrawCountDecide), before the lookup: the draw before was eligible
    // (FastDrawCountResolved), and nothing but light state changed since its caches were locked.
    const Count::PreviousDraw& draw = counter.taken;
    if (!draw.valid) {
        return fall(Fallback::NoCandidate);
    }
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (is_indirect || draw.is_indexed != is_indexed || draw.maxwell3d != maxwell3d ||
        draw.topology != static_cast<u32>(draw_state.topology) ||
        draw.engine_state != static_cast<u32>(maxwell3d->engine_state) ||
        !draw_state.inline_index_draw_indexes.empty() || frame_dump.Capturing()) {
        return fall(Fallback::Kind);
    }
    if (draw.serial != counter.take_serial) {
        return fall(Fallback::StateChanged);
    }
    if (draw.map_serial != gpu_memory->MapSerial()) {
        return fall(Fallback::Map);
    }
    if (draw.tick != scheduler.CurrentTick()) {
        return fall(Fallback::Submitted);
    }
    auto* const previous = static_cast<GraphicsPipeline*>(const_cast<void*>(draw.pipeline));
    if (draw.render_pass != scheduler.RenderPassSerial() || !scheduler.IsRenderPassActive() ||
        scheduler.BoundGraphicsPipeline() != previous) {
        return fall(Fallback::RenderPass);
    }
    // The first cut: nothing to sample, so no handle, view or sampler can go stale.
    if (!previous->SamplesNothing() || previous->IsMeshPipeline()) {
        return fall(Fallback::Samples);
    }
    // A handle table (EDEN_NX2_LIGHT_TABLE_ARRAYS) binds a whole array of handles read from a
    // storage buffer each draw: never a delta draw, also if the cut above is ever widened to
    // pipelines that sample (a samples-nothing pipeline has none today).
    if (previous->HasHandleTable()) {
        return fall(Fallback::Samples);
    }
    // The texture memos change what Configure does for every draw (the plan: off, or no delta).
    if (VideoCore::Nx2Debug::Experiment(32) != 0 || VideoCore::Nx2Debug::Experiment(87) != 0 ||
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_tex_bank_gen,
                                         "EDEN_NX2_TEX_BANK_GEN") > 0) {
        return fall(Fallback::MemoSwitches);
    }
    // What the lookup reads besides the registers the serial covers.
    delta_vertex_moved = maxwell3d->dirty.flags[Dirty::VertexInput];
    switch (pipeline_cache.DeltaKeeps(*previous, false)) {
    case PipelineCache::DeltaLookup::Keeps:
        return previous;
    case PipelineCache::DeltaLookup::NotCurrent:
        return fall(Fallback::NotCurrent);
    case PipelineCache::DeltaLookup::Inputs:
        return fall(Fallback::LookupInputs);
    case PipelineCache::DeltaLookup::VertexInput:
        return fall(Fallback::VertexInput);
    }
    return fall(Fallback::NotCurrent);
}

bool RasterizerVulkan::DeltaPathCompare(const GraphicsPipeline* kept, const GraphicsPipeline* looked,
                                        int mode) {
    VideoCommon::FritchiiDeltaPath::Stats& stats = delta_path;
    ++stats.compared;
    if (kept == looked) {
        return true;
    }
    ++stats.differences;
    if (stats.total_differences++ < 16) {
        const auto describe = [](const GraphicsPipeline* pipeline) {
            return pipeline == nullptr
                       ? std::string{"none"}
                       : fmt::format("{:016x} (vs {:016x} fs {:016x})", pipeline->Key().Hash(),
                                     pipeline->ShaderHashes()[1], pipeline->ShaderHashes()[5]);
        };
        LOG_WARNING(Render_Vulkan,
                    "Delta path (EXP118={}) difference: kept pipeline {}, the full lookup gave "
                    "{}; vertex input moved {}, topology {}, indexed {}",
                    mode, describe(kept), describe(looked), delta_vertex_moved,
                    static_cast<u32>(maxwell3d->draw_manager.draw_state.topology),
                    fast_draw_count.taken.is_indexed);
    }
    return false;
}

void RasterizerVulkan::DeltaPathSnapshot() {
    namespace Count = VideoCommon::Nx2FastDrawCount;
    Count::Counter& counter = fast_draw_count;
    counter.draw_serial = Count::g_serial.load(std::memory_order_relaxed);
    counter.draw_map_serial = gpu_memory->MapSerial();
}

// ---------------------------------------------------------------------------------------------
// FRITCHII_MEMSTAGE_THREAD = 2: the memory stage's ring backend (memory_stage_link.cpp)
//
// What GPUMem asks of this side: the fill of a draw it serves (StageFillDraw: the facts table,
// the stage's copy as the buffer cache's inputs, the repack layout, the buffer half of the ranges
// its constant buffer words collected, then MemStage's fill into ring B), the cache's channel and
// uniform binds, its idle work. What the back asks of it: the buffer half of the stage's ranges at
// a run's start, FlushCaching at a run's end, the packets left at a run's start. And the back's
// served calls (OnStage): what the back would do in the buffer cache while it runs a served entry
// (a read flush and its gather, a MustFlushRegion question, collected ranges) runs on GPUMem, the
// cache's holder, while the back waits; inline everywhere else.

namespace {
/// The back is in DispatchCompute (fable_memstage_compute_brief 3: served calls by cause).
thread_local bool t_in_dispatch = false;

/// FRITCHII_MEMSTAGE_GATHER: what the back's run of a gather GPUMem asked for applied (its
/// OnCacheInvalidations, StageRunGather's request), for the gather line. Null otherwise.
struct GatherProbe {
    u64 ranges;
    bool gpu_written;
};
thread_local GatherProbe* t_gather_probe = nullptr;

/// List draws (nx2_list_draw.h): whether DrawListWords reads the two words back on the CPU
/// instead of drawing indirectly from them. GPUMem asks the same of its copy's registers and draw
/// state (FRITCHII_MEMSTAGE_LIST_DRAWS): such a list draw stays level. The list draw mode and
/// EDEN_NX2_MESH_FIRST are read once.
bool ListDrawReadsBack(const Tegra::Engines::Maxwell3D::DrawManager::State& draw_state,
                       const Maxwell& regs) {
    const bool quads = draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
                       draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip;
    return Tegra::Nx2ListDraw::Mode() != 1 || quads || draw_state.base_instance != 0 ||
           regs.transform_feedback_enabled != 0 || Shader::Backend::SPIRV::MeshFirstEnabled();
}
} // Anonymous namespace

bool RasterizerVulkan::OneFollowerListOnCpu(const Tegra::Engines::Maxwell3D& engine) {
    return ListDrawReadsBack(engine.draw_manager.draw_state, engine.regs);
}

template <typename F>
void RasterizerVulkan::OnStage(u8 cause, F&& body) {
    using Link = VideoCommon::MemoryStageLink;
    if (!Link::BackServed()) {
        body();
        return;
    }
    if (t_in_dispatch && VideoCommon::Nx2Profile::Fine() &&
        cause < static_cast<u8>(Link::Served::Count)) [[unlikely]] {
        using Fine = VideoCommon::Nx2Profile::FineCounter;
        static_assert(static_cast<u32>(Fine::ServedInDispatchFallback) -
                          static_cast<u32>(Fine::ServedInDispatch) + 1 ==
                      static_cast<u32>(Link::Served::Count));
        VideoCommon::Nx2Profile::AddFine(
            static_cast<Fine>(static_cast<u32>(Fine::ServedInDispatch) + cause), 1);
    }
    // The buffer cache's lock, if the back holds it (EXP118's delta draws), goes with the call.
    Common::SpinRecursiveMutex* const held =
        buffer_cache.mutex.IsHeldByCurrentThread() ? &buffer_cache.mutex : nullptr;
    const std::thread::id back = std::this_thread::get_id();
    auto run = [&] {
        if (held != nullptr) {
            held->HandOver(std::this_thread::get_id());
        }
        SCOPE_EXIT {
            if (held != nullptr) {
                held->HandOver(back);
            }
        };
        body();
    };
    Link::Get().ServeOnStage(static_cast<Link::Served>(cause), run);
}

void RasterizerVulkan::SetRingBackend() {
    using Link = VideoCommon::MemoryStageLink;
    Link::RingBackend backend{};
    backend.context = this;
    backend.fill_draw = [](void* context, VideoCommon::MemStageShadowEngine& copy, bool indexed,
                           u64 position, u64 seq, u64 hash,
                           std::vector<std::pair<u64, std::size_t>>& ranges) {
        return static_cast<RasterizerVulkan*>(context)->StageFillDraw(copy, indexed, position, seq,
                                                                      hash, ranges);
    };
    backend.bind_channel = [](void* context, s32 bind_id) {
        auto* const self = static_cast<RasterizerVulkan*>(context);
        const Link::CleanPoint clean;
        std::scoped_lock lock{self->buffer_cache.mutex};
        self->buffer_cache.BindToChannel(bind_id);
    };
    backend.bind_uniform = [](void* context, size_t stage, u32 slot, u64 gpu_address, u32 size,
                              bool enabled) {
        // As Maxwell3D::ProcessCBBind's calls (unlocked there too: the cache's user alone).
        auto& cache = static_cast<RasterizerVulkan*>(context)->buffer_cache;
        if (enabled) {
            cache.BindGraphicsUniformBuffer(stage, slot, gpu_address, size);
        } else {
            cache.DisableGraphicsUniformBuffer(stage, slot);
        }
    };
    backend.apply_ranges = [](void* context, const std::pair<u64, std::size_t>* ranges,
                              size_t count) {
        static_cast<RasterizerVulkan*>(context)->StageApplyRanges({ranges, count}, true);
    };
    backend.idle = [](void* context) {
        // The first half of the idle hook's EXP104 work (spec 5.2): the buffer allocator's.
        auto* const self = static_cast<RasterizerVulkan*>(context);
        BufferRecycler& buffers = self->buffer_staging_pool.Recycler();
        if (buffers.HasDoomed()) {
            buffers.DestroyIdle(2);
            self->device.PublishBufferHeapBudgets();
        }
    };
    backend.end_run = [](void* context) {
        auto* const self = static_cast<RasterizerVulkan*>(context);
        if (self->gpu_memory != nullptr) {
            self->gpu_memory->FlushCaching();
        }
    };
    backend.drop_packets = [](void* context, u64 position) {
        auto* const self = static_cast<RasterizerVulkan*>(context);
        const u64 dropped = self->mem_stage.DropRingPackets(position);
        if (position != ~u64{0}) {
            // FRITCHII_MEMSTAGE_MARKS_AHEAD: a run's first entry, nothing due (the ring's end is
            // checked at reset_stage).
            self->buffer_cache.CheckAheadAt(BufferCache::AheadPoint::RunStart);
        }
        return dropped;
    };
    backend.back_channel = [](void* context) {
        return static_cast<s32>(static_cast<RasterizerVulkan*>(context)->current_channel_id);
    };
    backend.ring_packets = [](void* context) {
        return static_cast<RasterizerVulkan*>(context)->mem_stage.RingPackets();
    };
    backend.ring_room = [](void* context) {
        return static_cast<RasterizerVulkan*>(context)->mem_stage.RingRoom();
    };
    Link::Get().SetCacheLock(&buffer_cache.mutex);
    backend.reset_stage = [](void* context, bool end) {
        auto* const self = static_cast<RasterizerVulkan*>(context);
        self->mem_stage.ResetStageSide(end);
        // FRITCHII_MEMSTAGE_MARKS_AHEAD: no mark ahead at a ring's start or end (both threads
        // parked, every packet played).
        self->buffer_cache.CheckAheadAt(end ? BufferCache::AheadPoint::RingEnd
                                            : BufferCache::AheadPoint::RingStart);
    };
    backend.promote_marks = [](void* context) {
        auto& cache = static_cast<RasterizerVulkan*>(context)->buffer_cache;
        std::scoped_lock lock{cache.mutex};
        cache.PromoteAhead();
    };
    backend.ranges_gpu_written = [](void* context, const std::pair<u64, std::size_t>* ranges,
                                    size_t count) {
        return static_cast<RasterizerVulkan*>(context)->StageRangesOverGpuWritten({ranges, count});
    };
    backend.list_reads_back = [](void*, const VideoCommon::MemStageShadowEngine& copy) {
        return ListDrawReadsBack(copy.draw_manager.draw_state, copy.regs);
    };
    backend.gather = [](void* context, void* engine, u64& draw_seq) {
        return static_cast<RasterizerVulkan*>(context)->StageRunGather(
            *static_cast<Tegra::Engines::Maxwell3D*>(engine), draw_seq);
    };
    backend.pass_ready = [](void* context) -> u8 {
        // fable_stage_levelpoints 6.2 step 4, on GPUMem outside any fill. Rule C, MARKS_AHEAD
        // off: only a fill that marks GPU-written data adds an uncommitted flush, and those fills
        // come level first (StageDrawWritesBuffers) or run lent, so a set found empty here is
        // empty when the back reaches this entry. No read tracking answer (13.3): the pass lists
        // its take (rule T).
        auto* const self = static_cast<RasterizerVulkan*>(context);
        auto& cache = self->buffer_cache;
        // MARKS_AHEAD on (14.7): a writing draw queues its download when the back promotes its
        // marks, after GPUMem's place, so the note here says nothing. Rule C is not needed: the
        // back groups and commits under the lock at its own place (SignalOrdering's served
        // branch). Read nothing of the cache: no due mark to miss a promotion of.
        if (cache.MarksAhead()) {
            return static_cast<u8>(Link::PassReady::Ready);
        }
        if (!cache.MayHaveUncommittedFlushes()) {
            return static_cast<u8>(Link::PassReady::Ready);
        }
        const Link::CleanPoint clean;
        std::scoped_lock lock{cache.mutex};
        const bool flushes = cache.HasUncommittedFlushes();
        cache.RefreshUncommittedFlushesNote();
        return static_cast<u8>(flushes ? Link::PassReady::Flushes : Link::PassReady::Ready);
    };
    backend.pass_gather = [](void* context, std::vector<std::pair<u64, u64>>& ranges) -> u8 {
        return static_cast<RasterizerVulkan*>(context)->StagePassGather(ranges);
    };
    backend.apply_pass = [](void* context, const std::vector<std::pair<u64, u64>>& ranges,
                            bool whole, bool listed) {
        // The back, at a passed gather's entry (or late): the half it owes. Whole: the full
        // gather path (the entry runs as a level run: the back holds the cache). A listed take
        // (rule T, fable_stage_levelpoints 13.3) leaves applying_writes right after the apply,
        // also when it throws, as GPU::Impl::InvalidateGPUCache's tracked gather does: a write
        // still listed after it was applied would hold back a landing of later GPU bytes.
        auto* const self = static_cast<RasterizerVulkan*>(context);
        SCOPE_EXIT {
            if (listed) {
                self->gpu.ReleaseTakenWrites(&ranges);
            }
        };
        const std::span<const std::pair<DAddr, u64>> span{ranges.data(), ranges.size()};
        if (whole) {
            self->OnCacheInvalidations(span);
            return;
        }
        if (Common::WriteRing::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
            self->mem_stage.NoteGather(MemStage::Gather::Apply, 0);
        }
        self->ApplyGatherOtherHalf(span);
    };
    backend.listed_writes = [](void* context) -> size_t {
        return static_cast<RasterizerVulkan*>(context)->gpu.ListedWrites();
    };
    backend.sync_probe = [](void* context, const std::pair<u64, u64>* runs, size_t count) -> u8 {
        // fable_stage_p3 6 (dry, GPUMem outside any fill): the first run whose sync would not be
        // a pure lookup, by cause.
        auto& cache = static_cast<RasterizerVulkan*>(context)->buffer_cache;
        using Probe = std::remove_cvref_t<decltype(cache)>::SyncProbe;
        const Link::CleanPoint clean;
        std::scoped_lock lock{cache.mutex};
        for (size_t i = 0; i < count; ++i) {
            switch (cache.ProbeSyncTarget(runs[i].first, runs[i].second)) {
            case Probe::Pure:
                break;
            case Probe::NoBuffer:
                return static_cast<u8>(Link::P3Outcome::NoBuffer);
            case Probe::NotInBounds:
                return static_cast<u8>(Link::P3Outcome::NotInBounds);
            case Probe::CpuModified:
                return static_cast<u8>(Link::P3Outcome::CpuModified);
            }
        }
        return static_cast<u8>(Link::P3Outcome::Passed);
    };
    backend.sync_targets = [](void* context, const std::pair<u64, u64>* runs, size_t count,
                              Link::SyncTarget* out) -> u8 {
        // FRITCHII_MEMSTAGE_PASS bit 2 (fable_stage_p3 4.2), GPUMem outside any fill, at the
        // entry's place in the cache's order (every fill before it done, none after it): the
        // query result sync's lookups, here instead of on the back. Every run is tested first;
        // only when each is a pure lookup (an existing buffer holds it, no page of it is
        // CPU-modified or untracked) is the sync's own ObtainCPUBuffer called on each, which then
        // finds that buffer, walks nothing to upload and marks the usage, as on THREAD 0 (the
        // mark keeps a later upload of those bytes after the query's copy in the submission).
        // Accurate fences are refused before this (FindBuffer would wait there).
        auto& cache = static_cast<RasterizerVulkan*>(context)->buffer_cache;
        using Probe = std::remove_cvref_t<decltype(cache)>::SyncProbe;
        const Link::CleanPoint clean;
        std::scoped_lock lock{cache.mutex};
        for (size_t i = 0; i < count; ++i) {
            switch (cache.ProbeSyncTarget(runs[i].first, runs[i].second)) {
            case Probe::Pure:
                break;
            case Probe::NoBuffer:
                return static_cast<u8>(Link::P3Outcome::NoBuffer);
            case Probe::NotInBounds:
                return static_cast<u8>(Link::P3Outcome::NotInBounds);
            case Probe::CpuModified:
                return static_cast<u8>(Link::P3Outcome::CpuModified);
            }
        }
        for (size_t i = 0; i < count; ++i) {
            const auto [buffer, offset] = cache.ObtainCPUBuffer(
                runs[i].first, static_cast<u32>(runs[i].second),
                VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                VideoCommon::ObtainBufferOperation::DoNothing);
            out[i] = Link::SyncTarget{
                .begin = runs[i].first,
                .bytes = runs[i].second,
                .buffer = std::bit_cast<u64>(buffer->Handle()),
                .offset = offset,
            };
        }
        return static_cast<u8>(Link::P3Outcome::Passed);
    };
    Link::Get().SetRingBackend(backend);
}

bool RasterizerVulkan::StageFillDraw(VideoCommon::MemStageShadowEngine& copy, bool is_indexed,
                                     u64 position, u64 seq, u64 hash,
                                     std::vector<std::pair<u64, std::size_t>>& ranges) {
    using Link = VideoCommon::MemoryStageLink;
    namespace Profile = VideoCommon::Nx2Profile;
    Link& link = Link::Get();
    // fable_stage_levelpoints 15.7: the setup's steps as fine zones (profile 2), one at a time.
    const bool fine = Profile::Fine();
    std::optional<Profile::Scope> step;
    const auto zone = [&](Profile::Zone next) {
        step.reset();
        if (fine) {
            step.emplace(next, true, false);
        }
    };
    zone(Profile::Zone::SetupWindow);
    // Step 7: the window and room in ring B, tested here, before a packet (not per entry).
    if (!link.StageBeforeFill(position)) {
        return false; // GPUMem is stopping: the draw is left to the back
    }
    zone(Profile::Zone::SetupProgram);
    // Join 1 (spec 4.4): a miss leaves the draw to the back, which publishes the facts.
    if (!StageSetupFacts(copy)) {
        link.NoteLevelDraw(4);
        return false;
    }
    step.reset();
    if (stage_facts.has_texel_buffers) {
        link.NoteLevelDraw(2); // LateFill: the texture buffer binds need the texture cache
        return false;
    }
    if (!mem_stage.Packets()) {
        link.NoteLevelDraw(7);
        return false;
    }
    if (!mem_stage.RingRoom()) {
        link.NoteLevelDraw(5);
        return false;
    }
    if (Link::Ahead() != 0) [[unlikely]] {
        zone(Profile::Zone::SetupLevel);
        // Step 7, the window open: the back answers its served calls (a read flush's download, a
        // gather's claims, MustFlushRegion) from the cache as GPUMem has it. That is the back's
        // answer while no packet ahead of the back changes what those calls read: the buffers'
        // GPU-written marks. Two fills would: a draw that writes buffers (its marks), and the
        // collected constant buffer ranges claiming GPU-written data (their claims). Those fill
        // level: GPUMem first waits for the back to reach this entry (StageLevel).
        //
        // FRITCHII_MEMSTAGE_MARKS_AHEAD: a writing fill's marks wait on the cache's list until the
        // back has played its packet (buffer_cache.h, "Marks ahead of the back"), so the writing
        // draw fills ahead. The claims stay level, tested on both sets (the list on the ranges'
        // pages); after the wait the back has played every packet before this entry, except a
        // draw it still holds deferred: marks of one on these pages refuse the fill (the entry is
        // a run: the back issues that draw, then fills this one itself, in its order). The list's
        // bound (4,096 marks) waits as a writing draw did.
        const bool marks_ahead = buffer_cache.MarksAhead();
        bool going = true;
        if (!marks_ahead && StageDrawWritesBuffers(copy)) {
            going = link.StageLevel(position, Link::LevelReason::WritingDraw);
        } else if (StageRangesOverGpuWritten(ranges, true)) {
            going = link.StageLevel(position, Link::LevelReason::Claim);
            if (marks_ahead && going) {
                buffer_cache.NoteAheadLevel(BufferCache::AheadLevel::Claim);
                const bool still = [&] {
                    const Link::CleanPoint clean_point;
                    std::scoped_lock lock{buffer_cache.mutex};
                    buffer_cache.PromoteAhead();
                    return std::ranges::any_of(ranges, [&](const auto& range) {
                        return buffer_cache.AheadIntersectsPages(range.first, range.second);
                    });
                }();
                if (still) [[unlikely]] {
                    buffer_cache.NoteAheadLevel(BufferCache::AheadLevel::ClaimRefused);
                    return false;
                }
            }
        } else if (marks_ahead && StageDrawWritesBuffers(copy)) {
            bool full;
            {
                const Link::CleanPoint clean_point;
                std::scoped_lock lock{buffer_cache.mutex};
                full = buffer_cache.AheadFull();
            }
            if (full) [[unlikely]] {
                buffer_cache.NoteAheadLevel(BufferCache::AheadLevel::Bound);
                going = link.StageLevel(position, Link::LevelReason::WritingDraw);
            }
        }
        if (!going) {
            return false; // GPUMem is stopping: the draw is left to the back
        }
    }
    zone(Profile::Zone::SetupLock);
    // A clean point: while it waits for the lock the back holds, GPUMem serves the back's call.
    std::optional<Link::CleanPoint> clean{std::in_place};
    std::scoped_lock lock{buffer_cache.mutex};
    clean.reset();
    zone(Profile::Zone::SetupInputs);
    // FRITCHII_MEMSTAGE_MARKS_AHEAD: the fill's marks are keyed with the push index of the
    // packet it makes; a fill whose packet never goes takes them back (under the lock).
    const bool marks_fill = buffer_cache.MarksAhead();
    if (marks_fill) {
        buffer_cache.BeginAheadFill(mem_stage.NextRingPacketIndex());
    }
    SCOPE_EXIT {
        if (marks_fill) {
            buffer_cache.EndAheadFill();
        }
    };
    // A list draw (FRITCHII_MEMSTAGE_LIST_DRAWS; the copy's draw manager is inside DrawEnd's
    // list branch, MemStageShadowEngine::StageDraw sent it): the fill binds its two words as the
    // back's own fill does with SetDrawIndirect, from the copy's indirect parameters.
    const bool list_draw = copy.draw_manager.indirect_state.list_words;
    // The fill reads the stage's copy (plan 5.4), as at FRITCHII_MEMSTAGE_SHADOW=3.
    const VideoCommon::CacheInputs inputs = buffer_cache.StageInputs(
        copy.regs, copy.state, copy.root_tables, copy.draw_manager.draw_state, copy.dirty.flags,
        list_draw ? &copy.draw_manager.indirect_state : nullptr);
    buffer_cache.SetInputsOverride(&inputs);
    SCOPE_EXIT {
        buffer_cache.SetInputsOverride(nullptr);
    };
    // The packet number BeginDraw opens is GPUMem's unplayed mark until the push, or until here.
    SCOPE_EXIT {
        mem_stage.EndStageFill();
    };
    // Stamps per thread (the window open): the draw's number exists before anything here can
    // stamp, so every stamp of the fill carries it under GPUMem's mark. Level mode keeps the
    // order it had.
    const bool number_first = Link::Ahead() != 0;
    if (number_first) {
        mem_stage.BeginDraw(0, 0);
    }
    const u32 repack_dirty = buffer_cache_runtime.SetVertexRepackLayout(
        inputs.RepackLayoutRegs(), stage_facts.dynamic_vertex_input);
    if (number_first) {
        mem_stage.SetDrawRepack(buffer_cache_runtime.RepackMask(), repack_dirty);
    } else {
        mem_stage.BeginDraw(buffer_cache_runtime.RepackMask(), repack_dirty);
    }
    if (repack_dirty) {
        inputs.MarkVertexBuffersDirty(repack_dirty);
    }
    const MemStage::DrawFacts facts{
        .uniform_masks = &stage_facts.uniform_masks,
        .uniform_sizes = &stage_facts.uniform_sizes,
        .stage_infos = nullptr,
        .stages = stage_facts.stages,
        .has_texel_buffers = false,
        .uses_descriptor_buffer = stage_facts.uses_descriptor_buffer,
        .hashes = stage_facts.hashes,
        .staged = &stage_facts,
    };
    // The draw's raw handles, read here at the stage's position (SEAM 3, 2c's functions, on the
    // copy's constant buffer bindings): the back takes them from the packet.
    const std::vector<u32>* values = nullptr;
    const bool null_pointer_handles = VideoCore::Nx2Debug::Experiment(21) != 0;
    if (mem_stage.ValuesByValue() && stage_facts.infos != nullptr) {
        zone(Profile::Zone::SetupHandles);
        stage_handle_values.clear();
        const auto& stages = copy.state.shader_stages;
        using Cbufs = std::remove_cvref_t<decltype(stages[0].const_buffers)>;
        VideoCommon::FillGraphicsHandleValues(
            std::span<const Shader::Info>{*stage_facts.infos}, copy.back->GetMemoryManager(),
            [&stages](size_t stage) {
                return VideoCommon::GraphicsHandleCbufs<Cbufs>{stages[stage].const_buffers};
            },
            device.GetStorageBufferAlignment(), null_pointer_handles,
            [this](u32 raw) { stage_handle_values.push_back(raw); });
        values = &stage_handle_values;
    }
    step.reset(); // the fill has zones of its own ("memory fill" and its fine zones)
    struct Pre {
        RasterizerVulkan* self;
        std::vector<std::pair<u64, std::size_t>>* ranges;
    } pre{this, &ranges};
    const bool filled = mem_stage.FillRingDraw(
        buffer_cache, facts, is_indexed, position, seq, hash,
        [](void* argument) {
            // The buffer half of the ranges the stage's words collected, before the fill reads.
            auto* const p = static_cast<Pre*>(argument);
            p->self->StageApplyRanges(*p->ranges, false);
            p->ranges->clear();
        },
        &pre, values, null_pointer_handles, list_draw);
    if (!filled) {
        link.NoteLevelDraw(7);
    }
    return filled;
}

namespace {
/// FRITCHII_MEMSTAGE_SETUP_KEEP=2: the fields in which the kept facts and the full lookup's
/// differ, for the verify's log line.
std::string DiffStagedFacts(const MemStage::StagedFacts& kept,
                            const MemStage::StagedFacts& looked) {
    std::string fields;
    const auto add = [&fields](bool differs, std::string_view name) {
        if (differs) {
            fields += fields.empty() ? "" : ", ";
            fields += name;
        }
    };
    add(kept.uniform_masks != looked.uniform_masks, "uniform_masks");
    add(kept.uniform_sizes != looked.uniform_sizes, "uniform_sizes");
    add(kept.ssbos != looked.ssbos, "ssbos");
    add(kept.ssbo_counts != looked.ssbo_counts, "ssbo_counts");
    add(kept.stages != looked.stages, "stages");
    add(kept.has_texel_buffers != looked.has_texel_buffers, "has_texel_buffers");
    add(kept.uses_descriptor_buffer != looked.uses_descriptor_buffer, "uses_descriptor_buffer");
    add(kept.dynamic_vertex_input != looked.dynamic_vertex_input,
        fmt::format("dynamic_vertex_input {} -> {}", kept.dynamic_vertex_input,
                    looked.dynamic_vertex_input));
    add(kept.hashes != looked.hashes,
        fmt::format("hashes {:016x}/{:016x} -> {:016x}/{:016x}", kept.hashes[0], kept.hashes[1],
                    looked.hashes[0], looked.hashes[1]));
    add(kept.infos != looked.infos, "infos (pointer)");
    return fields.empty() ? std::string{"no field (compare)"} : fields;
}
} // Anonymous namespace

bool RasterizerVulkan::StageSetupFacts(const VideoCommon::MemStageShadowEngine& copy) {
    // FRITCHII_MEMSTAGE_SETUP_KEEP (live; fable_stage_levelpoints 15.3 row 3): most draws repeat
    // the programs of the draw before. The lookup's inputs are the copy's program registers (the
    // 49 words StageProgramId hashes) and the facts table; the table moves its generation at
    // every write. So while both are what the last lookup that found the facts saw, stage_facts
    // still holds that lookup's answer, and the hash and the lookup (its words compare and the
    // copy of the facts, with the infos' reference count) are skipped. The words are compared
    // themselves, not a counter of program register writes: a write path that missed the
    // counter (the entry loop's fast paths, a resync) would keep stale facts; 392 bytes compare
    // in a few ns. 0 = today, 1 = keep, 2 = verify (the full lookup at every kept setup, compared
    // with what was kept: must be 0, on the link line).
    namespace Shadow = VideoCommon::MemStageShadow;
    namespace Debug = VideoCore::Nx2Debug;
    const int keep =
        Debug::SwitchValue(Debug::g_memstage_setup_keep, "FRITCHII_MEMSTAGE_SETUP_KEEP");
    Shadow::ProgramId program_id;
    if (keep == 0) {
        setup_keep.valid = false;
        Shadow::StageProgramId(copy, program_id);
        return mem_stage.LookupFacts(program_id, stage_facts);
    }
    // Before the words and the lookup: a table write after this load moves the generation, so
    // the next setup looks up again (LookupFacts' own cache keeps its generation the same way).
    const u64 generation = mem_stage.FactsGeneration();
    Shadow::StageProgramWords(copy, program_id.words);
    // The code the words name (translated only when they or the address space moved).
    Shadow::StageProgramCode(copy, program_id);
    const bool same = setup_keep.valid && setup_keep.generation == generation &&
                      setup_keep.words == program_id.words &&
                      setup_keep.space == program_id.space && setup_keep.code == program_id.code;
    ++setup_keep.setups;
    bool found;
    if (same && keep == 1) {
        ++setup_keep.kept;
        found = true;
    } else {
        program_id.key = Shadow::ProgramWordsKey(program_id.words);
        if (same) {
            // Verify: the full setup anyway, its answer used, the kept one compared with it.
            // The keep stands for a lookup at the generation its check read (`generation`): the
            // same answer a full lookup made at the check gives. A table write that lands
            // between the check and this lookup (the back publishing another pipeline of the
            // same programs: a new infos pointer, maybe another dynamic vertex input) moves the
            // generation; this lookup then answers for a later table, and today's lookup made
            // a few ns earlier would have answered as the keep did. Those are counted as raced,
            // not compared (the first 2026-11-14 gate's 32 a window were these).
            MemStage::StagedFacts looked{};
            found = mem_stage.LookupFacts(program_id, looked);
            const bool raced = mem_stage.FactsGeneration() != generation;
            ++setup_keep.checked;
            const bool differ = !found || !(looked == stage_facts);
            if (raced) {
                ++setup_keep.raced;
                setup_keep.raced_differed += differ ? 1 : 0;
            } else if (differ) [[unlikely]] {
                ++setup_keep.disagreed;
                VideoCommon::MemoryStageLink::Note(
                    VideoCommon::MemoryStageLink::Check::SetupKeepDisagree,
                    found ? "other facts" : "the lookup missed");
            }
            if (differ) [[unlikely]] {
                // Which fields: the first 16 of each kind (raced or not) in the log.
                static u32 logged_raced = 0;
                static u32 logged_disagreed = 0;
                u32& logged = raced ? logged_raced : logged_disagreed;
                if (logged < 16) {
                    ++logged;
                    LOG_WARNING(HW_GPU,
                                "Memory stage draw setup verify (FRITCHII_MEMSTAGE_SETUP_KEEP=2): "
                                "{}: {} (generation at the check {}, after the lookup {})",
                                raced ? "raced with a table write, not counted"
                                      : "disagreed (must be 0)",
                                found ? DiffStagedFacts(stage_facts, looked)
                                      : std::string{"the lookup missed"},
                                generation, mem_stage.FactsGeneration());
                }
            }
            if (found) {
                stage_facts = std::move(looked);
            }
        } else {
            found = mem_stage.LookupFacts(program_id, stage_facts);
        }
        setup_keep.valid = found;
        setup_keep.generation = generation;
        setup_keep.words = program_id.words;
        setup_keep.space = program_id.space;
        setup_keep.code = program_id.code;
    }
    constexpr u64 SetupKeepLogEvery = u64{1} << 20; // about 10 s at the market's draw rate
    if (setup_keep.setups >= SetupKeepLogEvery) [[unlikely]] {
        LOG_INFO(HW_GPU,
                 "Memory stage draw setup (FRITCHII_MEMSTAGE_SETUP_KEEP={}): {} setups, facts kept "
                 "{}, verified {}, disagreed {} (must be 0), raced with a table write {} (not "
                 "compared; {} of them differed)",
                 keep, setup_keep.setups, setup_keep.kept, setup_keep.checked,
                 setup_keep.disagreed, setup_keep.raced, setup_keep.raced_differed);
        setup_keep.setups = 0;
        setup_keep.kept = 0;
        setup_keep.checked = 0;
        setup_keep.disagreed = 0;
        setup_keep.raced = 0;
        setup_keep.raced_differed = 0;
    }
    return found;
}

void RasterizerVulkan::CheckServedJoins(DAddr addr, u64 size) {
    // Step 7, in a call GPUMem serves for the back (the cache's lock held): a download of
    // GPU-written data from a buffer a packet ahead of the back joined reads that buffer before
    // the join's copy (must be 0; the window's StageLevel rules keep the marks the back's).
    using Link = VideoCommon::MemoryStageLink;
    if (!Link::OnStageThread() || Link::Ahead() == 0 || addr == 0 || size == 0) {
        return;
    }
    Link& link = Link::Get();
    const u64 back_position = VideoCommon::GpuSplit::Get().BackEntryPosition();
    switch (link.JoinedAhead(addr, size, back_position)) {
    case Link::Joined::Yes:
        if (buffer_cache.IsRegionGpuModified(addr, size)) [[unlikely]] {
            Link::Note(Link::Check::ServedAheadJoin, "a served call's download");
        }
        break;
    case Link::Joined::Unknown:
        link.NoteJoinUnknown();
        break;
    case Link::Joined::No:
        break;
    }
    // Fable's review, A: the download writes the GPU's bytes into game memory; a constant buffer
    // write GPUMem made there, for an entry past the back's, on GPU-written data, would be lost
    // (StageWriteWords comes level before such a write; this checks it while the window's verify
    // keeps GPUMem's writes).
    if (Link::StageWritesKept()) {
        const u64 end = addr + size;
        bool hit = false;
        link.ForEachStageWriteSince(0, [&](u64 write_begin, u64 write_end, u64 position) {
            if (position <= back_position) {
                return true;
            }
            const u64 begin = (std::max)(write_begin, u64{addr});
            const u64 last = (std::min)(write_end, end);
            hit = begin < last && buffer_cache.IsRegionGpuModified(begin, last - begin);
            return hit;
        });
        if (hit) [[unlikely]] {
            Link::Note(Link::Check::ServedOverStageWrite, "a served call's download");
        }
    }
}

bool RasterizerVulkan::StageDrawWritesBuffers(const VideoCommon::MemStageShadowEngine& copy) const {
    // The fill marks GPU-written ranges for written storage buffers (BindGraphicsStorageBuffer's
    // is_written) and for transform feedback.
    if (copy.regs.transform_feedback_enabled != 0) {
        return true;
    }
    for (u32 stage = 0; stage < 5; ++stage) {
        if ((stage_facts.stages & (1U << stage)) == 0) {
            continue;
        }
        for (u32 i = 0; i < stage_facts.ssbo_counts[stage]; ++i) {
            if (stage_facts.ssbos[stage][i].is_written) {
                return true;
            }
        }
    }
    return false;
}

bool RasterizerVulkan::StageRangesOverGpuWritten(
    std::span<const std::pair<DAddr, std::size_t>> ranges, bool claim) {
    // The page bits first (they never clear, so a clear one proves nothing GPU-written is there),
    // then the cache's own answer under its lock.
    // FRITCHII_MEMSTAGE_MARKS_AHEAD: the marks GPUMem made ahead of the back count too (their
    // page bits are set at the fill). A claim looks at their pages: WriteMemory acts on the
    // written pages, and must not run over a mark ahead (the must-be-0 "claim").
    // The real set keeps today's byte test (WriteMemory's page work on it is today's).
    const bool ahead = buffer_cache.MarksAhead();
    const auto pages = [](DAddr addr, std::size_t size) {
        const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        return std::pair<DAddr, u64>{begin,
                                     Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE) - begin};
    };
    bool maybe = false;
    for (const auto& [addr, size] : ranges) {
        if (MayBeGpuModified(addr, size)) {
            maybe = true;
            break;
        }
        if (ahead && claim && size != 0) {
            const auto [begin, length] = pages(addr, size);
            if (MayBeGpuModified(begin, length)) {
                maybe = true;
                break;
            }
        }
    }
    if (!maybe) {
        return false;
    }
    const VideoCommon::MemoryStageLink::CleanPoint clean;
    std::scoped_lock lock{buffer_cache.mutex};
    for (const auto& [addr, size] : ranges) {
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (ahead) {
        for (const auto& [addr, size] : ranges) {
            if (claim ? buffer_cache.AheadIntersectsPages(addr, size)
                      : buffer_cache.AheadIntersects(addr, size)) {
                return true;
            }
        }
    }
    return false;
}

bool RasterizerVulkan::StageRunGather(Tegra::Engines::Maxwell3D& engine, u64& draw_seq) {
    // GPUMem, level at the gather's entry (MemStageShadow's StageGather): the back is parked there
    // with every packet before it played and GPUMem waits outside any fill, so the back serving
    // this request is the cache's only user, at the gather's place in the stream. It runs what its
    // own run of the entry would: InvalidateTextureDataCache's default, the same code as THREAD 0,
    // with the cache's flags handed over around it. Not when `engine` has a deferred draw: the
    // entry issues it before the gather. Counted: whether the wait for idle had query results to
    // sync (they need the buffer cache at this place: why the gather can't be GPUMem's own work),
    // whether a game write range lay by GPU-written data, the ranges.
    using Link = VideoCommon::MemoryStageLink;
    struct Run {
        RasterizerVulkan* self;
        Tegra::Engines::Maxwell3D* engine;
        bool ran;
        bool synced;
        u64 draw_seq;
        GatherProbe probe;
    } run{this, &engine, false, false, 0, GatherProbe{0, false}};
    auto body = [&run] {
        if (run.engine->draw_manager.HasDeferredDraw()) {
            return;
        }
        RasterizerVulkan& self = *run.self;
        run.synced = self.query_cache.HasPendingSync();
        VideoCommon::MemStageShadow::BackGatherLend();
        {
            // BackGatherReturn at the end of this block, also if the gather throws.
            const VideoCommon::MemStageShadow::BackGatherFlags flags;
            {
                t_gather_probe = &run.probe;
                SCOPE_EXIT {
                    t_gather_probe = nullptr;
                };
                self.InvalidateGPUCache();
            }
            self.WaitForIdle();
        }
        run.draw_seq = Link::Get().BackDrawSeq();
        run.ran = true;
    };
    Link& link = Link::Get();
    link.ServeNow(body);
    if (run.ran) {
        link.NoteGatherRun(run.synced, run.probe.ranges, run.probe.gpu_written);
    }
    draw_seq = run.draw_seq;
    return run.ran;
}

void RasterizerVulkan::ProbeGatherRanges(std::span<const std::pair<DAddr, u64>> ranges) {
    // The back serving StageRunGather's request, before the ranges are applied: would a gather on
    // GPUMem have had to come level for GPU-written data on their pages (WriteMemory downloads or
    // uploads in place there)? Counted only; the page bits first, then the cache under its lock.
    GatherProbe& probe = *t_gather_probe;
    probe.ranges += ranges.size();
    if (probe.gpu_written) {
        return;
    }
    for (const auto& [addr, size] : ranges) {
        if (addr == 0 || size == 0) {
            continue;
        }
        const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        if (!MayBeGpuModified(begin, end - begin)) {
            continue;
        }
        std::scoped_lock lock{buffer_cache.mutex};
        // FRITCHII_MEMSTAGE_MARKS_AHEAD: both sets (a deferred draw's marks can be ahead here).
        if (buffer_cache.IsRegionGpuModified(begin, end - begin) ||
            buffer_cache.AheadIntersects(begin, end - begin)) {
            probe.gpu_written = true;
            return;
        }
    }
}

void RasterizerVulkan::StageApplyRanges(std::span<const std::pair<DAddr, std::size_t>> ranges,
                                        bool bracket) {
    // InnerInvalidation's buffer half (the stage's constant buffer words land on uniform pages
    // only: words on a watched page are a level entry, join 2).
    if (ranges.empty()) {
        return;
    }
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.NoteCachedWrites(ranges);
    if (Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        return;
    }
    std::optional<MemStage::FillPlay> seam;
    if (bracket) {
        seam.emplace(mem_stage, MemStage::Site::Invalidation);
    }
    buffer_cache.ClaimCpuWrites(ranges);
    for (const auto& [addr, size] : ranges) {
        buffer_cache.WriteMemory(addr, size, true, true);
    }
}

void RasterizerVulkan::InnerInvalidationServed(
    std::span<const std::pair<DAddr, std::size_t>> sequences) {
    OnStage(static_cast<u8>(VideoCommon::MemoryStageLink::Served::InnerInvalidation),
            [&] { StageApplyRanges(sequences, true); });
    if (Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        return;
    }
    const auto any = [&](const VideoCommon::PageFilter& filter) {
        for (const auto& [addr, size] : sequences) {
            if (filter.Any(addr, size)) {
                return true;
            }
        }
        return false;
    };
    if (any(texture_cache.page_filter)) {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    if (any(query_cache.page_filter)) {
        for (const auto& [addr, size] : sequences) {
            query_cache.InvalidateRegion(addr, size);
        }
    }
    if (any(pipeline_cache.page_filter)) {
        pipeline_cache.InvalidateRegions(sequences);
    }
}

namespace {
/// FRITCHII_NULL_RENDER: whether this draw is null-rendered (the switch on and the draw
/// `eligible`), counted for the "Null render" line every 10 s (the back thread's draws only).
bool NullRenderDraw(bool eligible) {
    struct Counts {
        u64 nulled{};
        u64 drawn{};
        std::chrono::steady_clock::time_point since{};
    };
    static Counts counts;
    const bool on = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_null_render,
                                                     "FRITCHII_NULL_RENDER") > 0;
    const bool nulled = on && eligible;
    if (on) {
        ++(nulled ? counts.nulled : counts.drawn);
    }
    const auto now = std::chrono::steady_clock::now();
    if (counts.since == std::chrono::steady_clock::time_point{}) {
        counts.since = now;
    } else if (const double window = std::chrono::duration<double>(now - counts.since).count();
               window >= 10.0) {
        LOG_INFO(Render_Vulkan,
                 "Null render (FRITCHII_NULL_RENDER={}, 10 s): {:.0f} draws/s null-rendered "
                 "(buffer side only), {:.0f} draws/s drawn in full (stage on, a held packet, a "
                 "delta or mesh draw)",
                 on ? 1 : 0, static_cast<double>(counts.nulled) / window,
                 static_cast<double>(counts.drawn) / window);
        counts = {};
        counts.since = now;
    }
    return nulled;
}
} // Anonymous namespace

template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func, bool is_indirect) {

    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_draws;
    }

    // One follower step 4: R plays an unlent draw (vk_one_follower_draw.cpp). `maxwell3d` is R's
    // view; E ran TickWork and the stream's invalidations around its push, and E filled the ring
    // B packet taken below.
    const VideoCommon::OneFollower::PacketHeader* const onef_play =
        VideoCommon::OneFollower::On() && VideoCommon::OneFollower::OnRenderThread()
            ? onef_draw->Playing()
            : nullptr;
    // fable_b 10.2 rule 2's assert: on R outside a lent call the rasterizer names the view, never
    // the channel's real engine (E's, which E writes meanwhile).
    if (VideoCommon::OneFollower::OnRenderThread() && !VideoCommon::OneFollower::InLent() &&
        channel_state != nullptr && maxwell3d == &channel_state->maxwell3d) [[unlikely]] {
        VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::OwnerOnR,
                                       "a draw on R outside a lent call names E's engine");
    }
    SCOPE_EXIT {
        if (onef_play == nullptr) {
            const VideoCommon::Nx2Profile::FineScope tick_scope{
                VideoCommon::Nx2Profile::Zone::TickWork};
            gpu.TickWork();
        }
    };
    bool submitted = false;
    SCOPE_EXIT {
        auto& native = maxwell3d->GetNativeMacroEngine();
        if (native.native_turing && !submitted) {
            native.native_execution_failed = true;
            const auto& regs = maxwell3d->regs;
            const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) |
                           regs.pipelines[1].reserved[1];
            const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) |
                           regs.pipelines[5].reserved[1];
            LOG_DEBUG(HW_GPU, "Native graphics draw not recorded: frame={} vs={:#x} ps={:#x}",
                      render_trace_frame, vs, ps);
        }
    };
    SCOPE_EXIT {
        ReleaseMeshEmulationBuffers();
    };
    namespace Profile = VideoCommon::Nx2Profile;
    using Link = VideoCommon::MemoryStageLink;
    if (InvalidateBeforeUse()) {
        gpu.InvalidateGPUCache();
    }
    {
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
        if (onef_play == nullptr) {
            gpu_memory->FlushCaching();
        }
    }
    // FRITCHII_MEMSTAGE_THREAD = 2: a draw the stage filled takes its packet from ring B here, and
    // every return plays it (PlayRest, spec 4.2): the cache believes its uploads are recorded. In
    // a level run only the draws the stage filled before it found the entry hard have one; the
    // others the back fills itself, lent. A served draw without one is counted (must be 0) and
    // filled here (level mode: the stage is parked, so this is safe, only not where it belongs).
    // One follower step 4: R takes the packet E filled for this draw and plays it as a served
    // draw (Configure binds nothing in the buffer cache, only the texture cache's lock is taken).
    const bool ring_on = Link::RingActive();
    const bool served = Link::BackServed() || onef_play != nullptr;
    const MemStage::DrawPacket* ring_packet = nullptr;
    std::optional<MemStage::RestAtEnd> ring_rest;
    if (onef_play != nullptr) [[unlikely]] {
        const Profile::FineScope take_scope{Profile::Zone::RingTake};
        ring_packet = mem_stage.TakeRingDraw(onef_play->draw_seq, onef_play->position, 0, false);
        ring_rest.emplace(mem_stage);
        if (ring_packet == nullptr) {
            // R cannot fill (the buffer cache is E's): the draw is not drawn (must be 0).
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::RenderPacketMissing,
                                           "PrepareDraw");
            return;
        }
        // UNLEND's List bit: a played list draw records its copy from the packet's buffer of the
        // two words (ConfigureDraw); without it R would ask the cache (E's). Either way round a
        // disagreement is E and R seeing different draws.
        const bool list_args = (ring_packet->flags & MemStage::ListArgs) != 0;
        if (list_args != maxwell3d->draw_manager.indirect_state.list_words) [[unlikely]] {
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::PlayedListArgs,
                                           list_args ? "list arguments in a non-list draw's packet"
                                                     : "a list draw's packet without its words");
            VideoCommon::OneFollower::DisableUnlend("a played draw's list arguments disagreed");
            return;
        }
    } else if (ring_on && (served || Link::BackInRun())) [[unlikely]] {
        const Profile::FineScope take_scope{Profile::Zone::RingTake};
        const bool verify = Link::RingVerify();
        ring_packet = mem_stage.TakeRingDraw(
            Link::Get().BackDrawSeq(), VideoCommon::GpuSplit::Get().BackEntryPosition(),
            verify ? VideoCommon::MemStageShadow::BackStateHash(*maxwell3d) : 0, verify);
        ring_rest.emplace(mem_stage);
        if (ring_packet == nullptr && served) {
            Link::Note(Link::Check::PacketMissing, "PrepareDraw (the back fills it)");
            // A served list draw skipped SetDrawIndirect (DrawListWords): the back's own fill
            // binds and syncs the two words only with it, as before the list draws were served.
            if (maxwell3d->draw_manager.indirect_state.list_words) {
                buffer_cache.SetDrawIndirect(&maxwell3d->draw_manager.indirect_state);
                list_fallback_indirect = true;
            }
        }
    }
    if (frame_dump.Capturing()) {
        frame_dump.DrainIfFull();
    }
    // EDEN_NX2_FASTDRAW_COUNT: counting only, at the original TakeReusableDraw.
    // Not on R's played draws (they take the buffer cache's lock; E lends draws while either is
    // on, this covers a switch turned on between E's decision and R's play).
    const bool fast_draw_counting = VideoCommon::Nx2FastDrawCount::g_count && onef_play == nullptr;
    // EXP118 (fritchii_delta_path.h; live, but nothing unless it was on at boot, which raises
    // its breaks): a draw that follows its pipeline's last draw with only light state changed,
    // for a pipeline that samples nothing, keeps the pipeline (no lookup) and configures it
    // without the texture steps. 2 = decide only, then compare with the full lookup.
    const int delta_mode =
        VideoCommon::Nx2FastDrawCount::g_on && onef_play == nullptr
            ? VideoCore::Nx2Debug::Experiment(VideoCommon::FritchiiDeltaPath::Switch)
            : 0;
    const bool draw_tracking = fast_draw_counting || delta_mode != 0;
    // The caches' locks: a delta draw takes them here, for its last test; every other draw
    // below, after its lookup, as before. Released at the end, before the SCOPE_EXITs above.
    std::unique_lock buffer_lock{buffer_cache.mutex, std::defer_lock};
    std::unique_lock texture_lock{texture_cache.mutex, std::defer_lock};
    GraphicsPipeline* delta_pipeline = nullptr; ///< EXP118 1 or 3: this draw is a delta draw
    GraphicsPipeline* would_pipeline = nullptr; ///< EXP118=2: what a delta draw would keep
    if (draw_tracking) [[unlikely]] {
        FastDrawCountBegin();
        if (delta_mode != 0) {
            using Fallback = VideoCommon::FritchiiDeltaPath::Fallback;
            delta_path.MaybeLog(delta_mode);
            GraphicsPipeline* const candidate = DeltaPathDecide(is_indexed, is_indirect);
            if (candidate != nullptr && delta_mode == 2) {
                would_pipeline = candidate;
            } else if (candidate != nullptr) {
                bool keep = true;
                if (delta_mode == 3) {
                    // Verify harder: the full lookup as well, outside the locks as always. If
                    // it gives another pipeline, the draw is drawn the full way.
                    GraphicsPipeline* looked;
                    {
                        Profile::Scope scope{Profile::Zone::PipelineLookup};
                        looked = pipeline_cache.CurrentGraphicsPipeline();
                    }
                    keep = DeltaPathCompare(candidate, looked, delta_mode);
                    if (!keep) {
                        delta_path.Fall(Fallback::LookupDiffers);
                    }
                }
                if (keep) {
                    std::lock(buffer_lock, texture_lock);
                    // Late test: a break on another thread since FastDrawCountBegin took the
                    // serial. Then the full path: locks released first, its lookup outside them.
                    if (VideoCommon::Nx2FastDrawCount::g_serial.load(std::memory_order_relaxed) !=
                            fast_draw_count.take_serial ||
                        pipeline_cache.DeltaKeeps(*candidate, true) !=
                            PipelineCache::DeltaLookup::Keeps) {
                        texture_lock.unlock();
                        buffer_lock.unlock();
                        delta_path.Fall(Fallback::Late);
                    } else {
                        delta_pipeline = candidate;
                        ++delta_path.taken;
                        delta_path.vertex_kept += delta_vertex_moved ? 1 : 0;
                    }
                }
            }
        }
    }

    const auto report_failure = [&](std::string_view reason) {
        TraceDrawFailure(reason);
        // One follower (FRITCHII_ONEF_MIRROR_STOP=2): E's samples mirror counted this draw, R's
        // counter does not start for it (CounterEnable is past the skip).
        if (VideoCommon::OneFollower::On() && maxwell3d->regs.zpass_pixel_count_enable)
            [[unlikely]] {
            VideoCommon::OneFollower::NoteSkippedSamplesDraw();
        }
        static std::set<std::tuple<u64, u64, std::string_view>> seen;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        if (frame_dump.Capturing()) {
            frame_dump.NoteSkipped(false, ps != 0 ? ps : vs, regs.rt[0].Address(), reason);
        }
        // A missing pipeline is counted by class of target in the "Async discard" 10 s line
        // instead; this capped list is kept for EDEN_NX2_ASYNC_DISCARD=1 (the default) only.
        if (reason == "no pipeline" && pipeline_cache.LastLookupSkipped() &&
            AsyncDiscardMode() != 1) {
            return;
        }
        if (seen.size() < 64 && seen.emplace(vs, ps, reason).second) {
            LOG_WARNING(HW_GPU, "Draw skipped ({}): vs={:#x} ps={:#x} rt0={:#x} {}x{}", reason, vs, ps,
                        regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height);
        }
    };
    // Before the lookup: the facts this draw publishes describe code at least this new.
    const u64 program_code_epoch =
        VideoCommon::MemStageShadow::g_program_code_epoch.load(std::memory_order_acquire);
    // One follower step 4: a played draw's lookup must find a pipeline that exists (10.1 row 5).
    // With rule 3 the packet carries E's pipeline (no lookup on R, and graphics_cache is E's).
    const bool onef_lookup = onef_play != nullptr && onef_play->pipeline == nullptr;
    const size_t onef_entries_before = onef_lookup ? pipeline_cache.GraphicsEntries() : 0;
    // FRITCHII_OCCL_FLAG (fritchii_occl_flag.h): with one follower off, an occlusion test draw
    // may take its pipeline's flag variant; a kept flag variant is decided again.
    VideoCommon::OcclFlag::t_census = VideoCommon::OcclFlag::Census::Unknown;
    VideoCommon::OcclFlag::t_colour_prepass = false;
    VideoCommon::OcclFlag::t_discard_direct = false;
    VideoCommon::OcclFlag::t_colour_discard_prepass = false;
    const auto occl_flag = [&](GraphicsPipeline* chosen, bool allow_make) -> GraphicsPipeline* {
        if (chosen != nullptr && chosen->occl_flag_parent != nullptr) {
            chosen = chosen->occl_flag_parent; // a kept flag variant: decided again
        }
        // One follower with FRITCHII_ONEF_OCCL on (OneFollower::OcclWanted: unset = AMD's
        // proprietary driver on a discrete GPU): a played draw's flag variant is E's choice (its
        // lookup, the packet's pipeline); R only makes sure the flag area exists (made here, on
        // R), and lent draws take their variants as with one follower off. With it off, v6.2's
        // rule exactly: no flag variant with one follower on, lent draws included. (v6.2 made
        // that rule after two AMD losses read a wild address at a lent load, 2026-11-18/19;
        // RGD traced those to a stale vertex-buffer table, MemPlayer::EnsureVertexBindings,
        // not the flag store, which only stores.)
        const bool onef_occl = VideoCommon::OneFollower::OcclWanted();
        if (onef_play != nullptr && onef_occl) {
            if (VideoCommon::OcclFlag::Mode() != VideoCommon::OcclFlag::Off) {
                static_cast<void>(query_cache_runtime.OcclFlagReady());
            }
            return chosen;
        }
        if (onef_play != nullptr || (VideoCommon::OneFollower::On() && !onef_occl) ||
            chosen == nullptr || VideoCommon::OcclFlag::Mode() == VideoCommon::OcclFlag::Off ||
            !query_cache_runtime.OcclFlagReady()) {
            return chosen;
        }
        GraphicsPipeline* const flag =
            pipeline_cache.OcclFlagPipeline(chosen, maxwell3d->regs, allow_make);
        // The pre-pass records the draw twice: not for indirect and list draws (recording them
        // consumes their arguments), the delta path and draw counting (they key on the bound
        // pipeline), or a frame dump's probe.
        const bool colour_prepass = VideoCommon::OcclFlag::t_colour_prepass;
        const bool discard_prepass = flag != nullptr && flag->IsOcclDiscardPrepass() &&
                                     !VideoCommon::OcclFlag::t_discard_direct;
        if (flag != nullptr && (discard_prepass || colour_prepass) &&
            (is_indirect || draw_tracking || frame_dump.CapturingSampled())) [[unlikely]] {
            VideoCommon::OcclFlag::Add(VideoCommon::OcclFlag::g_stats.prepass_fallback);
            VideoCommon::OcclFlag::t_census =
                colour_prepass ? VideoCommon::OcclFlag::t_colour_prepass_class
                : VideoCommon::OcclFlag::t_colour_discard_prepass
                    ? VideoCommon::OcclFlag::Census::ColourDiscards
                    : VideoCommon::OcclFlag::Census::OffDiscards;
            VideoCommon::OcclFlag::t_colour_prepass = false;
            VideoCommon::OcclFlag::t_colour_discard_prepass = false;
            return chosen;
        }
        return flag;
    };
    GraphicsPipeline* const pipeline{[&]() -> GraphicsPipeline* {
        if (delta_pipeline != nullptr) {
            return occl_flag(delta_pipeline, false); // EXP118: kept, no lookup (locks held)
        }
        if (onef_play != nullptr && onef_play->pipeline != nullptr) {
            // Rule 3 (FRITCHII_ONEF_E_PIPELINE): E looked this draw's pipeline up at its own
            // position; R binds it and looks nothing up. With FRITCHII_ONEF_OCCL off it is never a
            // flag variant (v6.2: the parent). On, it may be the flag variant E picked
            // (FillAndPush): an in-place one, or a pre-pass variant bringing E's pre-pass state
            // (FRITCHII_ONEF_OCCL_PREPASS); R records the pre-pass as the off path does, or, where
            // that path falls back (draw counting, a frame dump's probe), the parent with the same
            // descriptors.
            auto* const played = const_cast<GraphicsPipeline*>(
                static_cast<const GraphicsPipeline*>(onef_play->pipeline));
            if (!VideoCommon::OneFollower::OcclWanted()) {
                return played->occl_flag_parent != nullptr ? played->occl_flag_parent : played;
            }
            const u8 occl = played->occl_flag_parent != nullptr ? onef_play->reserved[1] : u8{0};
            namespace OcclFlag = VideoCommon::OcclFlag;
            OcclFlag::t_colour_prepass = (occl & VideoCommon::OneFollower::OcclColourPrepass) != 0;
            OcclFlag::t_discard_direct = (occl & VideoCommon::OneFollower::OcclDiscardDirect) != 0;
            OcclFlag::t_colour_discard_prepass =
                (occl & VideoCommon::OneFollower::OcclColourDiscardPrepass) != 0;
            const bool prepass = OcclFlag::t_colour_prepass ||
                                 (played->IsOcclDiscardPrepass() && !OcclFlag::t_discard_direct);
            if (prepass && played->occl_flag_parent == nullptr) [[unlikely]] {
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_ERROR(Render_Vulkan,
                              "One follower: a played draw's occlusion pre-pass bits {:#x} on a "
                              "pipeline without a flag parent (kind {}, draw seq {}, flags {:#x})",
                              occl, played->occl_flag_kind, onef_play->draw_seq,
                              onef_play->flags);
                }
                OcclFlag::t_colour_prepass = false;
                OcclFlag::t_discard_direct = false;
                OcclFlag::t_colour_discard_prepass = false;
            }
            if (prepass && played->occl_flag_parent != nullptr &&
                (draw_tracking || frame_dump.CapturingSampled())) [[unlikely]] {
                OcclFlag::Add(OcclFlag::g_stats.prepass_fallback);
                OcclFlag::t_colour_prepass = false;
                OcclFlag::t_discard_direct = false;
                OcclFlag::t_colour_discard_prepass = false;
                return played->occl_flag_parent;
            }
            return played;
        }
        Profile::Scope scope{Profile::Zone::PipelineLookup};
        return occl_flag(pipeline_cache.CurrentGraphicsPipeline(), true);
    }()};
    if (onef_lookup &&
        (!pipeline || pipeline_cache.GraphicsEntries() > onef_entries_before)) [[unlikely]] {
        VideoCommon::OneFollower::Note(
            VideoCommon::OneFollower::Check::PlayedLookupMiss,
            !pipeline ? "played draw: no pipeline" : "played draw: a new pipeline built on R");
        VideoCommon::OneFollower::DisableUnlend("a played draw's pipeline lookup missed");
        if (!pipeline) {
            return;
        }
    }
    if (!pipeline) {
        report_failure("no pipeline");
        TraceRenderState(is_indexed, "no-pipeline", nullptr);
        // Mesh emulation survey: how many task/mesh draws a device without
        // VK_EXT_mesh_shader drops, how many are indirect and how many groups they launch.
        if (maxwell3d->regs.IsMeshEnabled() && !device.IsExtMeshShaderSupported()) {
            static u64 draws{}, indirect{}, groups{}, max_groups{};
            ++draws;
            if (is_indirect) {
                ++indirect;
            } else {
                const u64 count{maxwell3d->draw_manager.draw_state.vertex_buffer.count};
                groups += count;
                max_groups = std::max(max_groups, count);
            }
            if (draws == 1 || draws % 5000 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "Mesh emulation survey: {} mesh draws dropped, {} indirect, {} groups "
                            "in direct ones (max {} per draw), frame {}",
                            draws, indirect, groups, max_groups, render_trace_frame);
            }
        }
        return;
    }
    // FRITCHII_DRAW_KINDS: whether the draw's pipeline samples nothing.
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        VideoCommon::FritchiiDrawKinds::NotePipeline(pipeline->SamplesNothing());
    }
    // EDEN_NX2_MEMSTAGE_STATS: the first draw of each shader set (the memory stage's join 1).
    if (VideoCommon::MemStageStats::Enabled()) [[unlikely]] {
        VideoCommon::MemStageStats::ProgramSet(pipeline->ShaderHashes());
    }
    // FRITCHII_MEMSTAGE_SHADOW: the facts table's compare (c) and the draw kinds of spec 3.4.
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        const MemStage::DrawFacts facts = pipeline->MemFacts();
        // A diagnostic that reads the bound buffers on the back: a frame dump or a trace. The
        // OoT trackers are not one any more: they read the draw packet's record (5c, plan 5.1).
        const bool diagnostic = frame_dump.Capturing() || VideoCommon::RenderTrace::Enabled();
        VideoCommon::MemStageShadow::BackPipeline(
            pipeline->ShaderHashes(), ShadowFactsDigest(facts, pipeline->HasDynamicVertexInput()),
            (facts.has_texel_buffers ? VideoCommon::MemStageShadow::PipelineTexel : 0U) |
                (diagnostic ? VideoCommon::MemStageShadow::PipelineDiagnostic : 0U));
    }
    // TEST: up to Configure (EDEN_NX2_GPU_PROFILE=2).
    std::optional<Profile::FineScope> setup_scope;
    setup_scope.emplace(Profile::Zone::DrawSetup);
    if (VideoCommon::OotNetwork::BypassEnabled(Settings::GetCurrentProgramID())) {
        TrackOotTonemappedImage(*pipeline);
    }
    // Task/mesh pipelines: the draw's vertex count is the number of task workgroups.
    mesh_draw = pipeline->IsMeshPipeline();
    // EDEN_NX2_NO_MESH=1: skip task/mesh draws as before they were supported (A/B comparisons).
    // EDEN_NX2_NO_MESH_FILE=<path>: skip them while that file exists (checked every 1024 draws),
    // to compare the same scene with and without them in one run.
    static const bool no_mesh_always{std::getenv("EDEN_NX2_NO_MESH") != nullptr};
    static const char* const no_mesh_file{std::getenv("EDEN_NX2_NO_MESH_FILE")};
    static bool no_mesh{no_mesh_always};
    static u32 no_mesh_check{};
    if (no_mesh_file && (no_mesh_check++ % 1024) == 0) {
        no_mesh = std::filesystem::exists(no_mesh_file);
    }
    const bool emulated_mesh{mesh_draw && pipeline->IsEmulatedMesh()};
    // UNLEND's Mesh bit: native mesh draws, and with FRITCHII_ONEF_MESH_EMU emulated ones too (the
    // staging pool, the mesh arenas and the compute passes below run on R, played or lent; E
    // raises the vertex buffer flags the emulation leaves). An indirect one is never posted.
    if (onef_play != nullptr && emulated_mesh &&
        (is_indirect || !VideoCommon::OneFollower::EmulatedMeshPostable())) [[unlikely]] {
        VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::PlayedNoPath,
                                       "a played draw on an emulated mesh pipeline");
        VideoCommon::OneFollower::DisableUnlend("a played draw needed mesh emulation");
        return;
    }
    if (mesh_draw &&
        (no_mesh || (!device.IsExtMeshShaderSupported() && !emulated_mesh) ||
         (emulated_mesh && is_indirect))) {
        report_failure(no_mesh         ? "EDEN_NX2_NO_MESH"
                       : emulated_mesh ? "indirect emulated mesh draw"
                                       : "no VK_EXT_mesh_shader");
        TraceRenderState(is_indexed, "mesh-skipped", nullptr);
        return;
    }
    // Mesh emulation: one output block per mesh workgroup, written by the pipeline's compute
    // dispatch and drawn as groups * vertices-per-group triangle list vertices.
    mesh_emulation_groups = 0;
    mesh_emulation_vertices_per_group = 0;
    if (emulated_mesh) {
        // The same group count the native path hands vkCmdDrawMeshTasksEXT (MakeDrawParams:
        // the index count for indexed draws), within the device's dispatch limit.
        const auto& draw_state{maxwell3d->draw_manager.draw_state};
        const u32 groups{(std::min)(MakeDrawParams(draw_state, 1, is_indexed).num_vertices,
                                    device.GetMaxComputeWorkGroupCount()[0])};
        // Mesh emulation statistics: draws with and without groups, and the frames with any
        // groups (the game culls on the GPU and sets the counts itself).
        {
            static u64 draws{}, empty{}, total_groups{}, frames{}, frames_drawn{};
            static u64 last_frame{~0ULL};
            static bool frame_drawn{};
            if (render_trace_frame != last_frame) {
                if (last_frame != ~0ULL) {
                    ++frames;
                    frames_drawn += frame_drawn;
                }
                last_frame = render_trace_frame;
                frame_drawn = false;
            }
            ++draws;
            empty += groups == 0;
            total_groups += groups;
            frame_drawn |= groups != 0;
            if (draws % 2000 == 0) {
                LOG_INFO(Render_Vulkan,
                         "Mesh emulation draws: {} ({} with no groups), {} groups; frames with "
                         "emulated draws: {} of {}",
                         draws, empty, total_groups, frames_drawn, frames);
            }
        }
        // Read the pending probe before Request can hand its buffer out again.
        if (mesh_probe.pending && scheduler.IsFree(mesh_probe.tick)) {
            ReadMeshEmulationProbe();
        }
        // EDEN_NX2_MESH_EMULATION_PROBE=1: every 60 frames, read one draw's output back (from a
        // host-visible download buffer, as device-local ones are not mapped on every GPU).
        static const bool probe{std::getenv("EDEN_NX2_MESH_EMULATION_PROBE") != nullptr};
        const bool has_task{pipeline->MeshEmulationHasTask()};
        const bool probe_draw{probe && !has_task && !mesh_probe.pending && groups != 0 &&
                              render_trace_frame % 60 == 0};
        // With a task program the draw's groups are task workgroups. The mesh workgroups they ask
        // for are only known on the GPU, so their output goes to the submission's arena.
        const u64 block_bytes{pipeline->MeshEmulationBlockBytes()};
        StagingBufferRef output{};
        u64 bytes{};
        if (has_task) {
            const MeshArena& arena{AcquireMeshArena()};
            mesh_arena_used = true;
            bytes = arena.size;
            pipeline->SetMeshEmulationOutput(*arena.buffer, 0, bytes, groups);
        } else {
            bytes = (std::max)(u64{groups} * block_bytes, u64{4});
            output = staging_pool.Request(
                bytes, probe_draw ? MemoryUsage::Download : MemoryUsage::DeviceLocal, true,
                VideoCommon::BufferChurnStats::StagingCaller::Rasterizer);
            mesh_draw_buffers.push_back(output);
            pipeline->SetMeshEmulationOutput(output.buffer, output.offset, bytes, groups);
        }
        mesh_emulation_indirect = {};
        if (mesh_task_probe.pending && scheduler.IsFree(mesh_task_probe.tick)) {
            mesh_task_probe.pending = false;
            std::array<u32, Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS> words{};
            std::memcpy(words.data(), mesh_task_probe.args.data(), sizeof(words));
            LOG_INFO(Render_Vulkan,
                     "Mesh emulation task probe: {} tasks asked for {} mesh workgroups, {} "
                     "dispatched at arena word {} (arena {} words)",
                     mesh_task_probe.tasks,
                     words[Shader::Backend::SPIRV::MESH_EMULATION_ARGS_REQUESTED], words[0],
                     words[Shader::Backend::SPIRV::MESH_EMULATION_ARGS_BASE],
                     mesh_task_probe.capacity);
        }
        if (has_task && groups != 0) {
            const bool task_probe{probe && !mesh_task_probe.pending &&
                                  render_trace_frame % 60 == 0};
            const auto request = [&](u64 size, MemoryUsage usage = MemoryUsage::DeviceLocal) {
                size = (std::max)(size, u64{16});
                const StagingBufferRef ref{staging_pool.Request(
                    size, usage, true, VideoCommon::BufferChurnStats::StagingCaller::Rasterizer)};
                mesh_draw_buffers.push_back(ref);
                return std::make_pair(GraphicsPipeline::MeshEmulationBuffer{ref.buffer, ref.offset,
                                                                            size},
                                      ref.mapped_span.subspan(
                                          (std::min)(ref.offset, u64{ref.mapped_span.size()})));
            };
            const auto payload{
                request(u64{groups} * pipeline->MeshEmulationPayloadStride() * 4).first};
            const auto offsets{request(u64{groups} * 4).first};
            constexpr u64 args_bytes{Shader::Backend::SPIRV::MESH_EMULATION_ARGS_WORDS * 4};
            const auto [args, args_span] =
                request(args_bytes, task_probe ? MemoryUsage::Download : MemoryUsage::DeviceLocal);
            const u32 capacity_words{static_cast<u32>(
                (std::min)(bytes / 4, u64{(std::numeric_limits<u32>::max)()}))};
            if (task_probe && args_span.size() >= args_bytes) {
                mesh_task_probe = {
                    .args = args_span,
                    .tick = scheduler.CurrentTick(),
                    .tasks = groups,
                    .capacity = capacity_words,
                    .pending = true,
                };
            }
            const MeshArena& arena{mesh_arenas[mesh_arena_current]};
            pipeline->SetMeshEmulationTaskBuffers(
                payload, offsets, args, {*arena.counter, 0, sizeof(u32)}, capacity_words);
            mesh_emulation_indirect = {args.buffer, args.offset};
        }
        if (probe_draw && output.mapped_span.size() >= output.offset + bytes) {
            const MeshEmulationSetup& setup{*pipeline->MeshEmulation()};
            mesh_probe = {
                .data = output.mapped_span.subspan(output.offset, bytes),
                .tick = scheduler.CurrentTick(),
                .frame = render_trace_frame,
                .groups = groups,
                .block_words = setup.block_words,
                .vertex_lines = setup.vertex_lines,
                .max_vertices = setup.max_vertices,
                .max_primitives = setup.max_primitives,
                .position_lines = setup.position_lines,
                .pending = true,
            };
        } else if (probe_draw && output.mapped_span.empty()) {
            static bool warned{};
            if (!warned) {
                warned = true;
                LOG_WARNING(Render_Vulkan, "Mesh emulation probe: output buffer is not mapped");
            }
        }
        mesh_emulation_groups = groups;
        mesh_emulation_vertices_per_group = pipeline->MeshEmulationVerticesPerGroup();
    }
    if (!buffer_lock.owns_lock()) {
        const Profile::FineScope locks_scope{Profile::Zone::DrawLocks};
        if (ring_packet != nullptr && served) {
            texture_lock.lock(); // spec 5.6: the buffer cache's lock is the stage's
        } else {
            std::lock(buffer_lock, texture_lock);
        }
    }
    // EDEN_NX2_FASTDRAW_COUNT: the original reads the serial the next draw compares with here, once the
    // caches are locked, and checks DeltaStillHolds.
    if (fast_draw_counting) [[unlikely]] {
        FastDrawCountDecide(pipeline, is_indexed, is_indirect);
    } else if (draw_tracking) [[unlikely]] {
        DeltaPathSnapshot();
    }
    if (would_pipeline != nullptr) [[unlikely]] {
        // EXP118=2: the draw would have been a delta draw if nothing broke since the decision;
        // the full lookup must have given the pipeline it would have kept.
        if (VideoCommon::Nx2FastDrawCount::g_serial.load(std::memory_order_relaxed) !=
            fast_draw_count.take_serial) {
            delta_path.Fall(VideoCommon::FritchiiDeltaPath::Fallback::Late);
        } else {
            ++delta_path.would;
            delta_path.vertex_kept += delta_vertex_moved ? 1 : 0;
            DeltaPathCompare(would_pipeline, pipeline, delta_mode);
        }
    }
    // update engine as channel may be different.
    pipeline->SetEngine(maxwell3d, gpu_memory);
    // THREAD = 2: the facts table (join 1) learns this pipeline's facts under the stage's key.
    // One follower from step 3: E looks them up for its fills at step 4 and for the facts
    // compare at step 3 (lent and played draws publish).
    // Not from played draws (fable_b 10.1 row 2): their pipeline is R's finding on the view;
    // a lent draw's comes from the engine itself, the authority.
    // With E_PIPELINE (rule 3; implied by FRITCHII_ONEF_LEAN) nothing reads the table: E fills
    // from the pipeline's own facts.
    if (ring_on || (VideoCommon::OneFollower::On() && !VideoCommon::OneFollower::g_e_pipeline &&
                    onef_play == nullptr &&
                    VideoCommon::OneFollower::Step() >=
                        VideoCommon::OneFollower::StepLevel::Build)) [[unlikely]] {
        const Profile::FineScope facts_scope{Profile::Zone::RingFacts};
        VideoCommon::MemStageShadow::ProgramId program_id;
        VideoCommon::MemStageShadow::BackProgramId(*maxwell3d, program_id);
        mem_stage.PublishFacts(program_id, pipeline, pipeline->MemFacts(),
                               pipeline->HasDynamicVertexInput(), program_code_epoch);
    }
    if (VideoCommon::OneFollower::On()) [[unlikely]] {
        // Part (c)'s facts compare: the programs this draw ran with (R only).
        onef_pipeline_hashes = {pipeline->ShaderHashes()[1], pipeline->ShaderHashes()[5]};
        onef_pipeline_seen = true;
        // One follower step 4, before anything is recorded: the facts E filled this unlent draw's
        // packet from must name the programs of the pipeline about to be bound. Any cause of a
        // disagreement (stale facts, another channel's code, a view that is off) stops here: the
        // draw is not recorded (its packet's uploads still play), "facts stale" counts it, and
        // step 4 stops for the session.
        if (onef_play != nullptr && mem_stage.HeldHashes() != onef_pipeline_hashes)
            [[unlikely]] {
            const auto& held = mem_stage.HeldHashes();
            onef_precheck_note = fmt::format(
                "unlent draw before recording: E's facts {:#x}/{:#x}, pipeline {:#x}/{:#x}; "
                "E's address space {}, R's {}",
                held[0], held[1], onef_pipeline_hashes[0], onef_pipeline_hashes[1],
                onef_draw->FillSpace(onef_play->draw_seq), gpu_memory->GetID() + 1);
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::FactsStale,
                                           onef_precheck_note.c_str());
            VideoCommon::OneFollower::DisableUnlend(
                "an unlent draw's facts named other programs than its pipeline");
            return;
        }
    }
    // FRITCHII_MEMSTAGE_SHADOW (piece 5c): the cache's flags over the fill (compare b); with 3,
    // the fill below and the repack layout read the stage's copy of the engine (CacheInputs on
    // the copy, plan 5.4) when the shadow can vouch for it, until the draw is recorded.
    std::optional<VideoCommon::CacheInputs> stage_inputs;
    struct InputsOverride {
        BufferCache* cache{};
        ~InputsOverride() {
            if (cache != nullptr) {
                cache->SetInputsOverride(nullptr);
            }
        }
    } inputs_override;
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed) &&
        ring_packet == nullptr) [[unlikely]] {
        if (VideoCommon::MemStageShadowEngine* const copy =
                VideoCommon::MemStageShadow::BackFill(*maxwell3d, is_indirect)) {
            stage_inputs.emplace(VideoCommon::CacheInputs::OfStage(
                maxwell3d, copy->regs, copy->state, copy->root_tables,
                copy->draw_manager.draw_state, copy->dirty.flags, kepler_compute, gpu_memory,
                nullptr));
            buffer_cache.SetInputsOverride(&*stage_inputs);
            inputs_override.cache = &buffer_cache;
        }
    }
    if (ring_packet != nullptr) {
        // THREAD = 2: the stage set the repack layout and the cache's flags in its fill; the
        // vertex input state is the back's own.
        back_repack_mask = ring_packet->repack_mask;
        if (ring_packet->repack_dirty != 0) {
            maxwell3d->dirty.flags[Dirty::VertexInput] = true;
        }
        // Step 7 (fps fix 2, take 2), a served draw with the window open: the stage's fill ran
        // the cache's vertex buffer update for this draw, on its copy's flags. With one thread
        // that update clears the engine's VertexBuffers before UpdateVertexInput looks at it, so
        // a stream address, limit or stride write alone never sends the vertex input state (the
        // buffers are bound again with their strides). The back's flag, raised by those writes
        // and cleared by nothing, sent it at about 20,000 draws a second with nothing changed.
        // Cleared here as that update would have; Dirty::VertexInput still sends every change of
        // the formats, the divisors and the repack layout. A lend sets the back's cache flags
        // from GPUMem's copy again. One follower: E's fill ran that update on E's flags, so R
        // clears its view's copy here for the same reason.
        if (onef_play != nullptr || (VideoCommon::MemoryStageLink::BackServed() &&
                                     VideoCommon::MemoryStageLink::Ahead() != 0)) {
            maxwell3d->dirty.flags[VideoCommon::Dirty::VertexBuffers] = false;
        }
    } else {
        const u32 repack_dirty = buffer_cache_runtime.SetVertexRepackLayout(
            buffer_cache.Inputs().RepackLayoutRegs(), pipeline->HasDynamicVertexInput());
        back_repack_mask = buffer_cache_runtime.RepackMask();
        mem_stage.BeginDraw(buffer_cache_runtime.RepackMask(), repack_dirty);
        if (repack_dirty) {
            // The cache's flags through its inputs; the vertex input state is the back's own.
            buffer_cache.Inputs().MarkVertexBuffersDirty(repack_dirty);
            maxwell3d->dirty.flags[Dirty::VertexInput] = true;
        }
    }
    setup_scope.reset();
    // Delta draws (Fable's port plan step D2): the pipeline keeps this draw as its last while the
    // count runs or EXP106 is on (live). Off, one relaxed load a draw and the same Configure.
    const bool keep_last_draw =
        fast_draw_counting ||
        VideoCore::Nx2Debug::g_experiments[106].load(std::memory_order_relaxed) != 0;
    // SEAM = 2 (EXP123): the draw's buffer side is filled here, before the texture work;
    // Configure plays its sections and RestAtEnd what is left, however Configure ends (true,
    // false or an exception).
    const MemStage::DrawPacket* const packet =
        ring_packet != nullptr ? ring_packet
                               : mem_stage.FillDraw(buffer_cache, pipeline->MemFacts(), is_indexed);
    // The OoT trackers' record (plan 5.1): taken from the packet; without one (SEAM < 2, a late
    // fill, the trackers turned on since the frame began), today's query.
    std::optional<MemStage::TrackerRecord> tracker_record;
    if (packet != nullptr && packet->trackers.filled) [[unlikely]] {
        tracker_record = packet->trackers;
    }
    // FRITCHII_NULL_RENDER (live, default 0; test only, the picture is broken): the draw does the
    // buffer side alone (GraphicsPipeline::ConfigureImpl's null path) and returns before the
    // texture work, render targets, dynamic state, queries and the record. Only with the stage
    // off and no held packet (EXP123 below 2); a delta or mesh draw is drawn as usual. Off: one
    // relaxed load and a branch a draw.
    if (VideoCore::Nx2Debug::g_null_render.load(std::memory_order_relaxed) != 0) [[unlikely]] {
        const bool eligible = packet == nullptr && !ring_on && delta_pipeline == nullptr &&
                              !mesh_draw;
        if (NullRenderDraw(eligible)) {
            {
                Profile::Scope scope{Profile::Zone::Configure};
                const MemStage::RestAtEnd rest{mem_stage};
                pipeline->SetNullRenderNext();
                static_cast<void>(pipeline->Configure(is_indexed, keep_last_draw));
            }
            submitted = true; // not a failed native draw
            return;
        }
    }
    // FRITCHII_MEMSTAGE_SHADOW: read flushes from here to the record are counted.
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        VideoCommon::MemStageShadow::BackDrawWindow(true);
    }
    bool configured{};
    const u64 configure_tick = scheduler.CurrentTick();
    const u64 index_plays_before = mem_player.IndexPlays();
    {
        Profile::Scope scope{Profile::Zone::Configure};
        const MemStage::RestAtEnd rest{mem_stage};
        configured = delta_pipeline != nullptr ? pipeline->ConfigureDelta(is_indexed)
                                               : pipeline->Configure(is_indexed, keep_last_draw);
    }
    // FRITCHII_MEMSTAGE_LIST_DRAWS: GPUMem filled this draw as a list draw, the back's is not one
    // (no list arguments asked for): the two sides' draws differ (must be 0).
    if (ring_packet != nullptr && (ring_packet->flags & MemStage::ListArgs) != 0 &&
        !mem_player.ListDrawArgsRequested()) [[unlikely]] {
        Link::Note(Link::Check::ListArgsUnasked, "PrepareDraw");
    }
    if (!configured) {
        report_failure("configure failed");
        TraceRenderState(is_indexed, "configure-failed", nullptr);
        return;
    }
    // TEST: from Configure's end to dynamic state (EDEN_NX2_GPU_PROFILE=2).
    std::optional<Profile::FineScope> after_scope;
    after_scope.emplace(Profile::Zone::DrawAfter);
    if (delta_pipeline != nullptr && delta_mode == 3) [[unlikely]] {
        // EXP118=3: the descriptor pools' sync the delta configure left out must change nothing.
        const u64 bindings = texture_cache.TextureBindingsSerial();
        texture_cache.SynchronizeDescriptors(false);
        ++delta_path.sync_checked;
        if (texture_cache.TextureBindingsSerial() != bindings) {
            ++delta_path.sync_changed;
            if (delta_path.total_differences++ < 16) {
                LOG_WARNING(Render_Vulkan,
                            "Delta path (EXP118=3) difference: the descriptor pools moved on a "
                            "delta draw (vs {:016x} fs {:016x})",
                            pipeline->ShaderHashes()[1], pipeline->ShaderHashes()[5]);
            }
        }
    }

    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed) && !ring_on &&
        OcarinaTemporalEnabled()) [[unlikely]] {
        VerifyTrackerRecord(tracker_record ? &*tracker_record : nullptr);
    }
    TrackOcarinaCamera(tracker_record ? &*tracker_record : nullptr);
    TrackOcarinaActor(pipeline->ShaderHashes()[1], is_indexed, is_indirect,
                      tracker_record ? &*tracker_record : nullptr);
    if (VideoCommon::OcarinaActorTrace()) [[unlikely]] {
        TraceOcarinaActorDraw(*pipeline, is_indexed);
    }
    TracePendingCpuWrites(false, pipeline->ShaderHashes());
    TraceRenderState(is_indexed, "configured", texture_cache.GetFramebuffer());
    if (frame_dump.Capturing() || flash_watch.WatchesPasses()) {
        FrameDumpPass(texture_cache.GetFramebuffer());
    }
    if (flash_watch.WatchesPasses()) {
        flash_watch.NoteDraw(pipeline->ShaderHashes());
    }
    if (frame_dump.Capturing()) {
        frame_dump.NoteDraw(pipeline->ShaderHashes());
        if (frame_dump.CapturingSampled()) {
            boost::container::small_vector<FrameDump::Sampled, 16> sampled;
            for (const auto& image : pipeline->SampledImages()) {
                sampled.push_back({image.stage, image.tic, &texture_cache.GetImageView(image.id),
                                   &texture_cache.GetSampler(image.sampler)});
            }
            bool recorded = frame_dump.NoteSampled(sampled);
            recorded |= FrameDumpBuffers(false, pipeline->ShaderHashes());
            if (recorded) {
                scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
            }
        }
    }
    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
        LOG_DEBUG(Render_Vulkan,
                  "Render trace pipeline: frame={} draw={} hashes={:016x} dynamic_vertex_input={}",
                  render_trace_frame, render_trace_draws, fmt::join(pipeline->ShaderHashes(), ","),
                  pipeline->HasDynamicVertexInput());
    }

    if (is_indirect && !mesh_draw && Common::DkbFrameTrace::Enabled()) {
        static const bool readback = [] {
            const char* value = std::getenv("EDEN_DKB_INDIRECT_READBACK");
            return value && std::string_view{value} == "1";
        }();
        if (readback) {
            CaptureIndirectTrace();
        }
    }

    after_scope.reset();
    {
        Profile::Scope scope{Profile::Zone::DynamicState};
        UpdateOcarinaJitter();
        UpdateDynamicStates(pipeline);
    }

    {
        const Profile::FineScope queries_scope{Profile::Zone::DrawQueries};
        query_cache.NotifySegment(true);
        HandleTransformFeedback();
        // FRITCHII_OCCL_FLAG=1: a flag draw opens (or keeps) a slot with no host query.
        // A flag slot reports 4096 when any sample passed, not the count: only for Ocarina of
        // Time, whose occlusion tests ask "any sample" (0xe2c compares a pair, one of them 0),
        // and not when the counts are asked to be precise (accurate part 2).
        const bool flag_draw = pipeline->occl_flag_parent != nullptr;
        if (flag_draw && VideoCommon::OcclFlag::Mode() == VideoCommon::OcclFlag::On &&
            Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL &&
            !VideoCore::Nx2Debug::AccuratePart(2)) {
            query_cache_runtime.SetNextFlagDraw(true);
            // The frontend offers Full once to players whose flags act here (AMD's driver).
            if (!VideoCommon::OcclFlag::g_full_could_help.load(std::memory_order_relaxed) &&
                device.GetDriverID() == VK_DRIVER_ID_AMD_PROPRIETARY &&
                device.IsShaderInt64Supported()) [[unlikely]] {
                VideoCommon::OcclFlag::g_full_could_help.store(true, std::memory_order_relaxed);
            }
        }
        query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                  maxwell3d->regs.zpass_pixel_count_enable);
        query_cache_runtime.SetNextFlagDraw(false);
        // FRITCHII_OCCL_FLAG: a flag variant stores into the open query's slot (push constant
        // bytes 40 to 47, unused otherwise); other counted draws mark their slot as mixed.
        // Census: the class the pipeline lookup gave this draw (Unknown: no lookup).
        const auto census = std::exchange(VideoCommon::OcclFlag::t_census,
                                          VideoCommon::OcclFlag::Census::Unknown);
        if (pipeline->occl_flag_parent != nullptr) {
            const u64 address = query_cache_runtime.OcclFlagAddress();
            scheduler.Record([layout = pipeline->Layout(), stages = pipeline->PushStageFlags(),
                              address](vk::CommandBuffer cmdbuf) {
                cmdbuf.PushConstants(layout, stages, 40, sizeof(address),
                                     &address);
            });
            VideoCommon::OcclFlag::Add(VideoCommon::OcclFlag::g_stats.flag_draws);
            if (pipeline->IsOcclDiscardPrepass() && !VideoCommon::OcclFlag::t_discard_direct) {
                VideoCommon::OcclFlag::Add(VideoCommon::OcclFlag::g_stats.prepass_draws);
            }

        } else if (maxwell3d->regs.zpass_pixel_count_enable != 0) {
            query_cache_runtime.NoteOtherCountedDraw(
                VideoCommon::OcclFlag::Blocks(census)); // nothing without the flag area
            if (VideoCommon::OcclFlag::Mode() != VideoCommon::OcclFlag::Off) [[unlikely]] {
                VideoCommon::OcclFlag::Add(VideoCommon::OcclFlag::g_stats.other_counted);
                VideoCommon::OcclFlag::Add(
                    VideoCommon::OcclFlag::g_stats.census[static_cast<size_t>(census)]);
            }
        }
    }
    const std::optional<u32> probe =
        frame_dump.CapturingSampled() ? frame_dump.BeginProbe() : std::nullopt;
    {
        Profile::Scope scope{Profile::Zone::DrawRecord};
        // FRITCHII_SPLIT_BIND_FIX: Configure (or anything after it) submitted: bind again.
        if (scheduler.CurrentTick() != configure_tick) [[unlikely]] {
            const u64 split_hash = pipeline->ShaderHashes()[5] != 0 ? pipeline->ShaderHashes()[5]
                                                                    : pipeline->ShaderHashes()[1];
            NoteSplitBind(false, split_hash);
            if (SplitBindFix() && (onef_play != nullptr || ring_packet != nullptr)) {
                // A played or served draw does not configure again (NotePlayedSplit).
                if (scheduler.BoundGraphicsPipeline() != pipeline) {
                    NotePlayedSplit(true, split_hash);
                    if (onef_play != nullptr) {
                        VideoCommon::OneFollower::Note(
                            VideoCommon::OneFollower::Check::PlayedSplitBind,
                            "a played draw's bind went with a submit before its record");
                    }
                    report_failure("bind lost to a submit");
                    return;
                }
                NotePlayedSplit(false, split_hash);
                // An index bind this draw played before the submit is in the old command buffer.
                if (is_indexed && mem_player.IndexPlays() != index_plays_before) {
                    mem_player.EnsureIndexBinding();
                }
            } else if (SplitBindFix()) {
                const MemStage::RestAtEnd rest{mem_stage};
                if (!pipeline->Configure(is_indexed, keep_last_draw)) {
                    report_failure("configure failed after a split");
                    return;
                }
                UpdateDynamicStates(pipeline);
                query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                          maxwell3d->regs.zpass_pixel_count_enable);
            }
        }
        // The vertex bindings the fill left out as clean, in a command buffer they are not
        // bound in yet (one follower's R starts command buffers E cannot see).
        mem_player.EnsureVertexBindings();
        scheduler.TraceGraphicsPipeline(pipeline->ShaderHashes());
        trace_draw_hashes = {pipeline->ShaderHashes()[1], pipeline->ShaderHashes()[5]};
        if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
            VideoCommon::MemStageShadow::BackDrawWindow(false);
        }
        const bool colour_prepass = std::exchange(VideoCommon::OcclFlag::t_colour_prepass, false);
        const bool discard_direct = std::exchange(VideoCommon::OcclFlag::t_discard_direct, false);
        const bool colour_discard_prepass =
            std::exchange(VideoCommon::OcclFlag::t_colour_discard_prepass, false);
        bool direct_by_parent = false;
        if (discard_direct) {
            // Early tests count the samples the variant's discards then kill: under an open host
            // query (another counted draw opened it, or flag slots are not used) the draw's own
            // pipeline draws instead (its descriptors equal the variant's).
            if (query_cache_runtime.OcclQuerySlotOpen()) {
                direct_by_parent = true;
            } else {
                VideoCommon::OcclFlag::Add(VideoCommon::OcclFlag::g_stats.discard_direct_draws);
            }
        }
        if (direct_by_parent) [[unlikely]] {
            GraphicsPipeline& parent = *pipeline->occl_flag_parent;
            if (parent.ConfigureAfterOcclPrepass(*pipeline)) [[likely]] {
                trace_draw_hashes = {parent.ShaderHashes()[1], parent.ShaderHashes()[5]};
                draw_func();
            } else {
                parent.occl_flag_prepass_state = 2;
                VideoCommon::OcclFlag::g_stats.prepass_lost.fetch_add(1,
                                                                      std::memory_order_relaxed);
                LOG_ERROR(Render_Vulkan, "Occlusion flag discard variant: the draw could not be "
                                         "configured with its own pipeline and was not recorded");
            }
        } else
        if ((pipeline->IsOcclDiscardPrepass() && !discard_direct) || colour_prepass) [[unlikely]] {
            // FRITCHII_OCCL_FLAG pre-pass: the variant draws first with depth writes off (and,
            // for a colour pre-pass, colour writes off) and
            // stores the flag; a host query open in this slot (verify, or a window that opened
            // one) does not count it: it ends before and the window's next slot opens after.
            // In mode 1 a host query already open counts the real draw exactly: no pre-pass.
            namespace OcclFlag = VideoCommon::OcclFlag;
            const bool query_open = query_cache_runtime.OcclQuerySlotOpen();
            const bool prepass = !query_open || OcclFlag::Mode() == OcclFlag::Verify;
            if (prepass) {
                if (query_open) {
                    query_cache_runtime.OcclPrepassPause();
                    OcclFlag::Add(OcclFlag::g_stats.prepass_paused);
                }
                scheduler.Record(
                    [](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthWriteEnableEXT(false); });
                // A pre-pass for a colour draw: colour writes off (the dynamic masks cleared;
                // without them the variant's key has every attachment's writes off).
                const bool clear_masks =
                    (colour_prepass || colour_discard_prepass) &&
                    pipeline->Key().state.extended_dynamic_state_3_blend != 0;
                if (colour_prepass) {
                    OcclFlag::Add(OcclFlag::g_stats.colour_prepass_draws);
                }
                if (clear_masks) {
                    scheduler.Record([](vk::CommandBuffer cmdbuf) {
                        static constexpr std::array<VkColorComponentFlags,
                                                    Maxwell::NumRenderTargets>
                            none{};
                        cmdbuf.SetColorWriteMaskEXT(0, none);
                    });
                }
                draw_func();
                scheduler.Record([enable = maxwell3d->regs.depth_write_enabled != 0](
                                     vk::CommandBuffer cmdbuf) {
                    cmdbuf.SetDepthWriteEnableEXT(enable);
                });
                if (clear_masks) {
                    // The draw's own masks again (recorded from its registers).
                    maxwell3d->dirty.flags[Dirty::Blending] = true;
                    maxwell3d->dirty.flags[Dirty::ColorMask] = true;
                    UpdateBlending(maxwell3d->regs, pipeline->occl_flag_parent);
                }
                if (query_open) {
                    query_cache_runtime.OcclPrepassResume();
                }
            } else {
                OcclFlag::Add(OcclFlag::g_stats.prepass_skipped);
            }
            // The real draw: the parent bound again with this draw's descriptors. A submission
            // inside (a descriptor allocation) ends the dynamic states and the open query: set
            // them again.
            GraphicsPipeline& parent = *pipeline->occl_flag_parent;
            const u64 tick = scheduler.CurrentTick();
            if (parent.ConfigureAfterOcclPrepass(*pipeline)) [[likely]] {
                if (scheduler.CurrentTick() != tick) [[unlikely]] {
                    UpdateDynamicStates(&parent);
                    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                                              maxwell3d->regs.zpass_pixel_count_enable);
                }
                trace_draw_hashes = {parent.ShaderHashes()[1], parent.ShaderHashes()[5]};
                draw_func();
            } else {
                if (colour_prepass) {
                    pipeline->occl_same_descriptors = 2; // no more colour pre-passes with it
                } else if (pipeline->occl_flag_kind ==
                           GraphicsPipeline::OCCL_KIND_PREPASS_NOCOLOUR) {
                    parent.occl_flag_prepass_nocolour_state = 2; // later draws: parent alone
                } else {
                    parent.occl_flag_prepass_state = 2; // later draws take the parent alone
                }
                OcclFlag::g_stats.prepass_lost.fetch_add(1, std::memory_order_relaxed);
                LOG_ERROR(Render_Vulkan, "Occlusion flag pre-pass: the real draw could not be "
                                         "configured and was not recorded");
            }
        } else {
            draw_func();
        }
    }
    if (probe) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const auto& regs = maxwell3d->regs;
        const bool alpha_to_coverage = regs.anti_alias_alpha_control.alpha_to_coverage != 0;
        const bool dynamic_alpha_to_coverage =
            device.IsExtExtendedDynamicState3EnablesSupported() &&
            device.SupportsDynamicState3AlphaToCoverageEnable() &&
            pipeline->UsesExtendedDynamicState();
        frame_dump.EndProbe(
            *probe,
            {
                .mesh = mesh_draw,
                .indirect = is_indirect,
                .first = is_indexed ? draw_state.index_buffer.first : draw_state.vertex_buffer.first,
                .elements = is_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count,
                .instances = draw_state.instance_count,
                .alpha_test = regs.alpha_test_enabled ? static_cast<u32>(regs.alpha_test_func) : 0,
                .alpha_ref = regs.alpha_test_ref,
                .alpha_to_coverage = alpha_to_coverage,
                .host_alpha_to_coverage = dynamic_alpha_to_coverage
                                              ? alpha_to_coverage_enable
                                              : alpha_to_coverage && pipeline->SupportsAlphaToCoverage(),
                .color0 = pipeline->SupportsAlphaToCoverage(),
                .kill = pipeline->FragmentCanDiscard(),
                .zero_handles = pipeline->ZeroTextureHandles(),
            });
    }
    submitted = true;
    if (draw_tracking) [[unlikely]] {
        FastDrawCountResolved(pipeline, is_indexed, is_indirect);
    }
    constexpr u64 final_vertex = 0x16aff33d999f993fULL;
    constexpr u64 final_fragment = 0x52402d8e8ab55263ULL;
    const auto hashes = pipeline->ShaderHashes();
    if (oot_4k_scene_frame == render_trace_frame && hashes[5] == final_fragment &&
        std::ranges::find(hashes, final_vertex) != hashes.end()) {
        const auto target_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
        if (target_id) {
            const auto& target = texture_cache.GetImageView(target_id);
            if (target.size.width == 2560 && target.size.height == 1440 &&
                target.format == VideoCore::Surface::PixelFormat::A8B8G8R8_SRGB) {
                oot_4k_guest_present = target.ImageHandle();
            }
        }
    }
    // The game draws Navi's glow, fireflies and dust right after the fog. They write no depth and
    // move on their own, so FSR must not take them from its history. DLSS doesn't use the mask.
    constexpr u64 fog_fragment = 0x3ddfb1d7a854c849ULL;
    if (hashes[5] == fog_fragment && oot_live_dlss && OcarinaTemporalEnabled() &&
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr &&
        oot_camera_sample_frame == render_trace_frame && !frame_dump.Capturing()) {
        const auto scene_id = texture_cache.GetRenderTargets().color_buffer_ids[0];
        if (scene_id) {
            const auto& scene = texture_cache.GetImageView(scene_id);
            // FRITCHII_DLSS_SCALED: a scaled scene is taken at its scaled size; the mask is
            // used only when that matches the upscaler's render size (Record checks).
            if (scene.format == VideoCore::Surface::PixelFormat::B10G11R11_FLOAT &&
                scene.size.width == oot_motion.extent.width &&
                scene.size.height == oot_motion.extent.height &&
                (!scene.IsRescaled() || VideoCommon::OcarinaScaledUpscale()) &&
                scene.range == VideoCommon::SubresourceRange{}) {
                oot_effects_scene = scene.Handle(Shader::TextureType::Color2D);
                oot_effects_view = scene_id;
                oot_effects_generation = texture_cache.ImageViewGeneration(scene_id);
                oot_effects_address = scene.gpu_addr;
                oot_effects_frame = render_trace_frame;
                oot_live_dlss->SaveEffectsBase(oot_effects_scene, RealExtent(scene),
                                               render_trace_frame);
            }
        }
    }
    static u32 native_capture_count{};
    if (maxwell3d->GetNativeMacroEngine().native_turing &&
        native_capture_count < 16 && [] {
            static const bool capture = std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr;
            return capture;
        }()) {
        const u32 capture = native_capture_count++;
        const auto prefix = fmt::format("native-draw-{}", capture);
        const auto dump = [&](const std::string& name, GPUVAddr address, size_t size) {
            if (!size || !gpu_memory->IsFullyMappedRange(address, size)) return;
            std::vector<u8> bytes(size);
            gpu_memory->ReadBlockUnsafe(address, bytes.data(), bytes.size());
            std::ofstream file(prefix + name + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            LOG_INFO(HW_GPU, "Native draw capture {}{}: address={:#x}, bytes={}, nonzero={}",
                     capture, name, address, size,
                     std::count_if(bytes.begin(), bytes.end(), [](u8 byte) { return byte != 0; }));
        };
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const size_t size = static_cast<size_t>(rt.array_pitch) * 4;
        if (size && size <= 16 * 1024 * 1024) {
            if (const auto address = gpu_memory->GpuToCpuAddress(rt.Address(), size)) {
                LOG_INFO(HW_GPU, "Native draw {} render target physical address={:#x}", capture, *address);
                texture_cache.DownloadMemory(*address, size);
                dump("-rt0", rt.Address(), size);
            }
        }
        std::ofstream registers(prefix + "-registers.bin", std::ios::binary);
        registers.write(reinterpret_cast<const char*>(&regs), sizeof(regs));
        for (size_t stream = 0; stream < 4; ++stream) {
            if (regs.vertex_streams[stream].IsEnabled())
                dump(fmt::format("-vertex-{}", stream), regs.vertex_streams[stream].Address(), 128);
        }
        for (size_t stage : {0U, 4U}) {
            for (size_t bank : {2U, 3U}) {
                const auto& cb = maxwell3d->state.shader_stages[stage].const_buffers[bank];
                if (cb.enabled) dump(fmt::format("-cbuf-{}-{}", stage, bank), cb.address, std::min<size_t>(cb.size, 128));
            }
        }
    }
}


namespace {
// EDEN_NX2_WINDOW_TRACE: describe clears and draws into screen-sized colour targets (debug aid).
void TraceWindowTarget(const Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& memory,
                       const char* kind, u32 count) {
    static const bool trace = std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr;
    static u32 traced{};
    const auto& regs = maxwell3d.regs;
    static const u64 skip = [] {
        const char* value = std::getenv("EDEN_NX2_WINDOW_TRACE_SKIP");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    static u64 seen_events{};
    static const u64 only_ps = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_PS");
        return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    const u64 pixel_shader =
        (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    if (!trace || regs.rt[0].width != 3840 || traced >= 400 || seen_events++ < skip ||
        (only_ps != 0 && pixel_shader != only_ps)) {
        return;
    }
    ++traced;
    const auto& vp = regs.viewport_transform[0];
    const auto& sc = regs.scissor_test[0];
    LOG_INFO(HW_GPU,
             "nx2 window {} rt0={:#x} count={} clear={:#x} color=({},{},{},{}) mask_common={} "
             "mask0={:#x} rt_count={} vp scale=({},{}) translate=({},{}) scissor en={} x={}..{} "
             "y={}..{} blend0={} vs={:#x} ps={:#x} rasterize={}",
             kind, regs.rt[0].Address(), count, regs.clear_surface.raw, regs.clear_color[0],
             regs.clear_color[1], regs.clear_color[2], regs.clear_color[3], regs.color_mask_common,
             regs.color_mask[0].raw, regs.rt_control.count.Value(), vp.scale_x, vp.scale_y,
             vp.translate_x, vp.translate_y, sc.enable, sc.min_x.Value(), sc.max_x.Value(),
             sc.min_y.Value(), sc.max_y.Value(), regs.blend.enable[0],
             (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1],
             (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1],
             regs.rasterize_enable);
    if (kind[0] != 'd') {
        return;
    }
    // Pixel-stage constant buffer 3: colour constants of the loading-screen passes.
    if (const auto& cb3 = maxwell3d.state.shader_stages[4].const_buffers[3]; cb3.enabled) {
        std::array<f32, 8> values{};
        memory.ReadBlockUnsafe(cb3.address, values.data(), sizeof(values));
        LOG_INFO(HW_GPU, "nx2 window ps c[3]@{:#x} size={}: {}", cb3.address, cb3.size,
                 fmt::join(values, " "));
    }
    static u32 texture_traces{};
    if (texture_traces++ > 200) {
        return;
    }
    // Texture used by the pixel shader's first bound TEX (handle in c[2][0x20]) and its TIC.
    const auto& cb = maxwell3d.state.shader_stages[4].const_buffers[2];
    const u32 handle = cb.enabled ? memory.Read<u32>(cb.address + 0x20) : 0;
    const u32 tic_index = handle & 0xfffff;
    const GPUVAddr tic_address = regs.tex_header.Address() + u64{tic_index} * 32;
    std::array<u32, 8> tic{};
    memory.ReadBlockUnsafe(tic_address, tic.data(), sizeof(tic));
    LOG_INFO(HW_GPU,
             "nx2 window texture: cb2={:#x} en={} handle={:#x} tic pool={:#x} limit={} "
             "sampler pool={:#x} tic[{}]={:08x}",
             cb.address, cb.enabled, handle, regs.tex_header.Address(), regs.tex_header.limit,
             regs.tex_sampler.Address(), tic_index, fmt::join(tic, " "));
    // First bytes of the texture's memory (block-linear, so just a sample).
    const GPUVAddr texture = (GPUVAddr{tic[2] & 0xffff} << 32) | tic[1];
    std::array<u32, 16> texels{};
    memory.ReadBlockUnsafe(texture, texels.data(), sizeof(texels));
    LOG_INFO(HW_GPU, "nx2 window texture {:#x} {}x{}: {:08x}", texture, (tic[4] & 0xffff) + 1,
             (tic[5] & 0xffff) + 1, fmt::join(texels, " "));
    // EDEN_NX2_TEXTURE_DUMP=<file>: save the first 4 MiB of the last traced texture's memory.
    if (const char* path = std::getenv("EDEN_NX2_TEXTURE_DUMP"); path != nullptr) {
        std::vector<u8> bytes(2048u * 10240u);
        memory.ReadBlockUnsafe(texture, bytes.data(), bytes.size());
        if (std::FILE* file = std::fopen(path, "wb")) {
            std::fwrite(bytes.data(), 1, bytes.size(), file);
            std::fclose(file);
        }
    }
}
} // Anonymous namespace

void RasterizerVulkan::Draw(bool is_indexed, u32 instance_count) {
    // FRITCHII_DELAY_DRAW_US (live, 0 = off): the delay test "D", on the back thread only. Off:
    // one load and a branch.
    VideoCommon::GpuSplit::DelayWorkIf(VideoCore::Nx2Debug::g_delay_draw_us,
                                       VideoCommon::GpuSplit::DelayTest::Draw);
    // One follower, on E: steps 1-2 lend the draw to R; from step 3 part (c)'s RouteDraw lends it
    // with the verify or, at step 4, fills ring B and pushes it. Before anything below reads
    // `maxwell3d` (it names R's view while R plays a draw).
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        if (maxwell3d->HasPendingAsyncE2c()) [[unlikely]] {
            if (VideoCommon::OneFollower::Step() < VideoCommon::OneFollower::StepLevel::Build) {
                maxwell3d->WaitAndSyncAsyncE2c();
                if (!maxwell3d->ShouldExecute()) {
                    return;
                }
            }
        }
        OneFollowerMirrorDraw();
        OneFollowerLendDraw(is_indexed, instance_count);
        return;
    }
    // fable_stage_levelpoints 19.4 (GPU profile on, the back): the draw path's whole time, and
    // which Scheduler::Record calls fall inside it ("Price Vulkan worker").
    const VideoCommon::FritchiiPrice::DrawScope price_draw_scope;
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    // One follower from step 3, on R (vk_one_follower_draw.cpp): a draw lent from elsewhere runs
    // here with the verify; RouteDraw returns false for its own and PlayDraw's calls.
    if (VideoCommon::OneFollower::On() && onef_draw->RouteDraw(is_indexed, instance_count))
        [[unlikely]] {
        return;
    }
    const VideoCommon::FritchiiDrawKinds::Scope draw_kinds_scope{maxwell3d->regs};
    // A vertex count GPU work wrote (nx2_indirect_grid.h): drawn with that count, read on the GPU.
    // Draws that need the count here (the indirect reset, quads, a first vertex) read the words,
    // which waits for the GPU on this thread.
    std::optional<std::pair<VkBuffer, u32>> gpu_count;
    if (!is_indexed && Tegra::Nx2IndirectGrid::AnyLive()) [[unlikely]] {
        auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 count = draw_state.vertex_buffer.count;
        if (Tegra::Nx2IndirectGrid::LooksLikeDrawPlaceholder(count)) {
            auto& grid = Tegra::Nx2IndirectGrid::Get();
            Tegra::Nx2IndirectGrid::LogCounts();
            const auto source = Tegra::Nx2IndirectGrid::MatchDrawCount(count);
            if (!source) {
                grid.skipped.fetch_add(1, std::memory_order_relaxed);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1) < 8) {
                    LOG_ERROR(Render_Vulkan, "GPU-counted draw {:#x} lost its source: skipped", count);
                }
                return;
            }
            const u32 first = draw_state.vertex_buffer.first;
            const bool first_placeholder =
                first == Tegra::Nx2IndirectGrid::DrawPlaceholder(
                             count & (Tegra::Nx2IndirectGrid::Slots - 1), 1);
            const bool on_gpu = (first == 0 || first_placeholder) &&
                                draw_state.topology != Maxwell::PrimitiveTopology::Quads &&
                                draw_state.topology != Maxwell::PrimitiveTopology::QuadStrip &&
                                device.IsExtTransformFeedbackSupported() &&
                                !IsNx2IndirectReset(is_indexed, instance_count);
            if (!on_gpu) {
                std::array<u32, 2> words{};
                gpu_memory->ReadBlock(*source, words.data(), sizeof(words));
                draw_state.vertex_buffer.count = words[0];
                if (first_placeholder) {
                    draw_state.vertex_buffer.first = words[1];
                }
                // FRITCHII_MEMSTAGE_SHADOW: the stage's copy takes the draw state (plan 3.3).
                if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed))
                    [[unlikely]] {
                    VideoCommon::MemStageShadow::BackDrawStateWritten();
                }
                grid.draws_refused.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::scoped_lock lock{buffer_cache.mutex};
                const auto [buffer, offset] = mem_stage.Ref(MemStage::Site::DrawCount, [&] {
                    return buffer_cache.ObtainBuffer(
                        *source, sizeof(u32),
                        VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                        VideoCommon::ObtainBufferOperation::DoNothing);
                });
                gpu_count.emplace(buffer->Handle(), offset);
                // The count comes from a dispatch before: make its writes visible to the draw's
                // indirect read (once per batch of dispatches; it ends a render pass).
                if (std::exchange(compute_since_count_barrier, false)) {
                    static constexpr VkMemoryBarrier COUNT_BARRIER{
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
                                         VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT |
                                         VK_ACCESS_MEMORY_READ_BIT,
                    };
                    scheduler.RequestOutsideRenderPassOperationContext();
                    scheduler.Record([](vk::CommandBuffer cmdbuf) {
                        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                               0, COUNT_BARRIER);
                    });
                }
                static const bool log_counts = [] {
                    const char* value = std::getenv("EDEN_NX2_INDIRECT_GRID");
                    return value != nullptr && value[0] == '2';
                }();
                if (log_counts) {
                    std::array<u32, 2> words{};
                    gpu_memory->ReadBlock(*source, words.data(), sizeof(words));
                    static std::atomic<u32> logged{};
                    if (logged.fetch_add(1) < 40) {
                        LOG_INFO(Render_Vulkan,
                                 "GPU-counted draw from {:#x}: count {} first {} instances {} topology {}",
                                 *source, words[0], words[1], instance_count,
                                 static_cast<u32>(draw_state.topology));
                    }
                }
                draw_state.vertex_buffer.first = 0;
                grid.draws.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    {
        // A draw's first and count can come from memory a skipped (rejected) compute program
        // should have written. Past 16M vertices they are garbage: building the quad index
        // buffer for one threw out of the GPU thread and ended the process (Hyrule Warriors:
        // Age of Imprisonment, entering a new area), and indexed ones read far out of bounds.
        const auto& state = maxwell3d->draw_manager.draw_state;
        const u64 first = is_indexed ? state.index_buffer.first : state.vertex_buffer.first;
        const u64 count = is_indexed ? state.index_buffer.count : state.vertex_buffer.count;
        constexpr u64 MaxVertices = u64{1} << 24;
        if (!gpu_count && first + count > MaxVertices) {
            static std::atomic<u32> skipped{};
            if (skipped.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(HW_GPU, "Draw skipped: {} {} from {} (vertex/index count past {})",
                            count, is_indexed ? "indices" : "vertices", first, MaxVertices);
            }
            return;
        }
    }
    // FRITCHII_ONEF_UNLENT_RESET: an unlent draw E tested at its place and found no indirect
    // reset (PacketNoReset) is not tested again here (R reading the program's code).
    const VideoCommon::OneFollower::PacketHeader* const onef_played =
        VideoCommon::OneFollower::On() ? onef_draw->Playing() : nullptr;
    const bool onef_no_reset =
        onef_played != nullptr &&
        (onef_played->flags & VideoCommon::OneFollower::PacketNoReset) != 0;
    if (!gpu_count && !onef_no_reset && TryNx2IndirectReset(is_indexed, instance_count)) {
        return;
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, is_indexed ? "draw-indexed" : "draw",
                      is_indexed ? maxwell3d->draw_manager.draw_state.index_buffer.count
                                 : maxwell3d->draw_manager.draw_state.vertex_buffer.count);
    // EDEN_NX2_CB24_TRACE: for draws whose vertex shader reads bank 24, show the inline
    // constant buffer upload target (CB_ADDRESS) and its first words.
    if (static const bool trace = std::getenv("EDEN_NX2_CB24_TRACE") != nullptr; trace) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 logged{};
        if ((vs == 0x11471c9a00 || vs == 0x11471ca600 || vs == 0x11471cd300) && logged++ < 3) {
            std::array<u32, 24> words{};
            gpu_memory->ReadBlockUnsafe(regs.const_buffer.Address(), words.data(), sizeof(words));
            LOG_INFO(HW_GPU, "bank24 draw vs={:#x}: CB_ADDRESS {:#x} size {:#x} offset {:#x}: {:08x}", vs,
                     regs.const_buffer.Address(), regs.const_buffer.size, regs.const_buffer.offset,
                     fmt::join(words, " "));
            // Search every bound vertex-stage buffer for render-target-size values.
            const auto& stage = maxwell3d->state.shader_stages[0];
            for (u32 slot = 0; slot < Tegra::Engines::Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& buffer = stage.const_buffers[slot];
                if (!buffer.enabled || buffer.size == 0) continue;
                std::vector<u32> data(std::min<u32>(buffer.size, 0x10000) / 4);
                gpu_memory->ReadBlockUnsafe(buffer.address, data.data(), data.size() * 4);
                for (size_t i = 0; i < data.size(); ++i) {
                    const u32 w = data[i];
                    if (w == 0x45200000 || w == 0x44b40000 || w == 0x39cccccd || w == 0x3a360b61 ||
                        w == 0x44a00000 || w == 0x44340000 || w == 2559 || w == 1439) {
                        LOG_INFO(HW_GPU, "  slot {} +{:#x} = {:#x}", slot, i * 4, w);
                    }
                }
            }
        }
    }
    // EDEN_NX2_TRACE_VS=<address>: describe the next few draws using that vertex shader like the
    // after-upscale trace does (render target, blend, fragment textures).
    static const u64 trace_vs = [] {
        const char* value = std::getenv("EDEN_NX2_TRACE_VS");
        return value ? std::strtoull(value, nullptr, 16) : 0ULL;
    }();
    if (trace_vs != 0) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        static u32 traced{};
        if (vs == trace_vs && traced++ % 200 == 0 && traced < 1000) {
            Tegra::Engines::g_nx2_trace_draws_after_upscale = 1;
        }
    }
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        --Tegra::Engines::g_nx2_trace_draws_after_upscale;
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        LOG_INFO(HW_GPU, "after upscale: draw rt0 {:#x} {}x{} fmt {:#x} block {}/{}/{} pitch_linear {} depth {} array_pitch {:#x} ctrl {:#x} vs {:#x} ps {:#x}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.rt[0].tile_mode.block_width.Value(),
                 regs.rt[0].tile_mode.block_height.Value(), regs.rt[0].tile_mode.block_depth.Value(),
                 regs.rt[0].tile_mode.is_pitch_linear.Value(), regs.rt[0].depth.Value(),
                 regs.rt[0].array_pitch, regs.rt_control.count.Value(), vs, ps);
        // Fragment-stage texture handles (bound buffer 2) and the images they name.
        const auto& handles = maxwell3d->state.shader_stages[4].const_buffers[2];
        if (handles.enabled) {
            for (u32 i = 0; i < 24; ++i) {
                const u32 handle = gpu_memory->Read<u32>(handles.address + i * 4);
                if (handle == 0 || handle == 0xffffffff) continue;
                Tegra::Texture::TICEntry tic{};
                gpu_memory->ReadBlockUnsafe(regs.tex_header.Address() + u64{handle & 0xfffff} * sizeof(tic),
                                            &tic, sizeof(tic));
                LOG_INFO(HW_GPU, "  fs handle[{}] = {:#x}: {:#x} {}x{} format {:#x}", i, handle,
                         tic.Address(), tic.Width(), tic.Height(), static_cast<u32>(tic.format.Value()));
            }
        }
        const auto& bt = regs.blend_per_target[0];
        LOG_INFO(HW_GPU, "  blend enable {} per-target {} global op {} src {:#x} dst {:#x} | rt0 op {} src {:#x} dst {:#x} alpha {:#x}/{:#x} mask {:#x}",
                 regs.blend.enable[0], regs.blend_per_target_enabled,
                 static_cast<u32>(regs.blend.color_op), static_cast<u32>(regs.blend.color_source),
                 static_cast<u32>(regs.blend.color_dest), static_cast<u32>(bt.color_op),
                 static_cast<u32>(bt.color_source), static_cast<u32>(bt.color_dest),
                 static_cast<u32>(bt.alpha_source), static_cast<u32>(bt.alpha_dest),
                 regs.color_mask[0].raw);
    }
    // EDEN_NX2_SKIP_VS=<address>[,<address>...]: drop draws using those vertex shaders (debug aid).
    static const std::vector<u64> skip_list = [] {
        std::vector<u64> list;
        if (const char* value = std::getenv("EDEN_NX2_SKIP_VS")) {
            for (char* end = nullptr; *value; value = *end ? end + 1 : end) {
                list.push_back(std::strtoull(value, &end, 16));
                if (end == value) break;
            }
        }
        return list;
    }();
    if (!skip_list.empty()) {
        const auto& regs = maxwell3d->regs;
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        if (std::find(skip_list.begin(), skip_list.end(), vs) != skip_list.end()) return;
    }
    // EDEN_NX2_RT_TRACE: log the first draw to every distinct colour target (debug aid).
    if (static const bool trace = std::getenv("EDEN_NX2_RT_TRACE") != nullptr; trace) {
        static std::set<std::tuple<GPUVAddr, u32, u32, u64, u64>> seen;
        const auto& regs = maxwell3d->regs;
        const auto& rt = regs.rt[0];
        const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
        const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
        const auto key = std::make_tuple(rt.Address(), rt.width, static_cast<u32>(rt.format), vs, ps);
        if (rt.width >= 640 && seen.size() < 600 && seen.insert(key).second) {
            LOG_INFO(HW_GPU, "nx2 draw target {:#x} {}x{} format={:#x} vs={:#x} ps={:#x} blend={} (pairs so far {})",
                     rt.Address(), rt.width, rt.height, static_cast<u32>(rt.format), vs, ps,
                     regs.blend.enable[0], seen.size());
        }
    }
    PrepareDraw(is_indexed, [this, is_indexed, instance_count, gpu_count] {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        const u32 num_instances{instance_count};
        if (gpu_count) [[unlikely]] {
            scheduler.Record([this, counter = *gpu_count, num_instances,
                              base_instance = draw_state.base_instance](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                // Stride 1: the vertex count is the counter's value.
                cmdbuf.DrawIndirectByteCountEXT(num_instances, base_instance, counter.first,
                                                counter.second, 0, 1);
            });
            return;
        }
        const DrawParams draw_params{MakeDrawParams(draw_state, num_instances, is_indexed)};
        if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace direct: frame={} draw={} count={} instances={} first_index={} "
                      "base_vertex={} base_instance={} mesh={}",
                      render_trace_frame, render_trace_draws, draw_params.num_vertices,
                      draw_params.num_instances, draw_params.first_index, draw_params.base_vertex,
                      draw_params.base_instance, mesh_draw);
        }

        if (Common::DkbFrameTrace::Enabled() &&
            (draw_params.num_vertices == 0 || (!mesh_draw && draw_params.num_instances == 0))) {
            ++trace_zero_direct;
        }

        if (mesh_draw && mesh_emulation_vertices_per_group != 0) {
            const u32 vertices{mesh_emulation_groups * mesh_emulation_vertices_per_group};
            if (vertices == 0) {
                return;
            }
            if (mesh_emulation_indirect.buffer) {
                // With a task program the distribution pass wrote the vertex count.
                scheduler.Record([this, indirect = mesh_emulation_indirect](vk::CommandBuffer cmdbuf) {
                    if (scheduler.GraphicsPipelineMissing()) {
                        return;
                    }
                    cmdbuf.DrawIndirect(indirect.buffer,
                                        indirect.offset +
                                            Shader::Backend::SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET,
                                        1, 0);
                });
                return;
            }
            scheduler.Record([this, vertices](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.Draw(vertices, 1, 0, 0);
            });
            return;
        }
        if (mesh_draw) {
            scheduler.Record([this, groups = draw_params.num_vertices](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.DrawMeshTasksEXT(groups, 1, 1);
            });
            return;
        }
        scheduler.Record([this, draw_params, hashes = trace_draw_hashes](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            scheduler.TraceGpuWork(
                cmdbuf, hashes.first, hashes.second,
                {draw_params.num_vertices, draw_params.num_instances, 0}, [&] {
                    if (draw_params.is_indexed) {
                        cmdbuf.DrawIndexed(draw_params.num_vertices, draw_params.num_instances,
                                           draw_params.first_index, draw_params.base_vertex,
                                           draw_params.base_instance);
                    } else {
                        cmdbuf.Draw(draw_params.num_vertices, draw_params.num_instances,
                                    draw_params.base_vertex, draw_params.base_instance);
                    }
                });
        });

        // Log draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string params = is_indexed ?
                fmt::format("vertices={}, instances={}, firstIndex={}, baseVertex={}, baseInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.first_index, draw_params.base_vertex, draw_params.base_instance) :
                fmt::format("vertices={}, instances={}, firstVertex={}, firstInstance={}",
                    draw_params.num_vertices, draw_params.num_instances,
                    draw_params.base_vertex, draw_params.base_instance);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                is_indexed ? "vkCmdDrawIndexed" : "vkCmdDraw", params, VK_SUCCESS);
        }
    });
}

void RasterizerVulkan::DrawIndirect() {
    // FRITCHII_DELAY_DRAW_US (live, 0 = off): the delay test "D" (list draws included), on the
    // back thread only. Off: one load and a branch.
    VideoCommon::GpuSplit::DelayWorkIf(VideoCore::Nx2Debug::g_delay_draw_us,
                                       VideoCommon::GpuSplit::DelayTest::Draw);
    // One follower: indirect and list draws stay lent in this branch (their own UnlendKind bits).
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        OneFollowerMirrorDraw();
        OneFollowerDrawIndirect();
        return;
    }
    const VideoCommon::FritchiiPrice::DrawScope price_draw_scope; // as in Draw
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Draw};
    const VideoCommon::FritchiiDrawKinds::Scope draw_kinds_scope{maxwell3d->regs};
    const auto& params = maxwell3d->draw_manager.indirect_state;
    if (params.list_words) {
        DrawListWords();
        return;
    }
    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_indirect;
    }
    buffer_cache.SetDrawIndirect(&params);
    PrepareDraw(params.is_indexed, [this, &params] {
        if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_draws <= 2048) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace indirect: frame={} draw={} address={:#x} count_address={:#x} "
                      "bytes={} max_draws={} stride={} include_count={} byte_count={} mesh={}",
                      render_trace_frame, render_trace_draws, params.indirect_start_address,
                      params.count_start_address, params.buffer_size, params.max_draw_counts,
                      params.stride, params.include_count, params.is_byte_count, mesh_draw);
        }
        if (mesh_draw) {
            TraceDrawFailure("indirect task/mesh unsupported");
            // Indirect task/mesh draws would need vkCmdDrawMeshTasksIndirectEXT's layout.
            static bool logged{};
            if (!std::exchange(logged, true)) {
                LOG_WARNING(Render_Vulkan, "Indirect task/mesh draw skipped");
            }
            return;
        }
        const auto indirect_buffer = mem_stage.Ref(
            MemStage::Site::DrawIndirect, [&] { return buffer_cache.GetDrawIndirectBuffer(); });
        const auto& buffer = indirect_buffer.first;
        const auto& offset = indirect_buffer.second;
        if (params.is_byte_count) {
            scheduler.Record([this, buffer_obj = buffer->Handle(), offset,
                              stride = params.stride](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                cmdbuf.DrawIndirectByteCountEXT(1, 0, buffer_obj, offset, 0,
                                                static_cast<u32>(stride));
            });
            return;
        }
        if (params.include_count) {
            const auto count = mem_stage.Ref(MemStage::Site::DrawIndirect,
                                             [&] { return buffer_cache.GetDrawIndirectCount(); });
            const auto& draw_buffer = count.first;
            const auto& offset_base = count.second;
            scheduler.Record([this, draw_buffer_obj = draw_buffer->Handle(),
                              buffer_obj = buffer->Handle(), offset_base, offset,
                              params](vk::CommandBuffer cmdbuf) {
                if (scheduler.GraphicsPipelineMissing()) {
                    return;
                }
                if (params.is_indexed) {
                    cmdbuf.DrawIndexedIndirectCount(
                        buffer_obj, offset, draw_buffer_obj, offset_base,
                        static_cast<u32>(params.max_draw_counts), static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirectCount(buffer_obj, offset, draw_buffer_obj, offset_base,
                                             static_cast<u32>(params.max_draw_counts),
                                             static_cast<u32>(params.stride));
                }
            });
            return;
        }
        scheduler.Record([this, buffer_obj = buffer->Handle(), offset, params,
                          hashes = trace_draw_hashes](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            scheduler.TraceGpuWork(cmdbuf, hashes.first, hashes.second, {}, [&] {
                if (params.is_indexed) {
                    cmdbuf.DrawIndexedIndirect(buffer_obj, offset,
                                               static_cast<u32>(params.max_draw_counts),
                                               static_cast<u32>(params.stride));
                } else {
                    cmdbuf.DrawIndirect(buffer_obj, offset,
                                        static_cast<u32>(params.max_draw_counts),
                                        static_cast<u32>(params.stride));
                }
            });
        });

        // Log indirect draw call
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            const std::string log_params = fmt::format("drawCount={}, stride={}",
                params.max_draw_counts, params.stride);
            GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                params.is_indexed ? "vkCmdDrawIndexedIndirect" : "vkCmdDrawIndirect",
                log_params, VK_SUCCESS);
        }
    }, true);
    buffer_cache.SetDrawIndirect(nullptr);
}

// List draws (nx2_list_draw.h): an array draw whose vertex count and first
// vertex are the two words at indirect_state.indirect_start_address (Nx2ListDraw). The host GPU
// copies them into a draw command and draws from it; draws that need the words on the CPU read
// them back, as the front did before (a flush of the words, synchronous).
bool RasterizerVulkan::Nx2ListDrawSupported() const {
    // Task/mesh draws are drawn with vkCmdDrawMeshTasksIndirectEXT (no mesh emulation: it
    // skips indirect draws).
    return Tegra::Nx2ListDraw::Mode() != 0 && device.IsExtMeshShaderSupported();
}

// List dispatches (nx2_list_dispatch.h): DispatchCompute's indirect path takes the grid from the
// buffer cache's buffer at the list's address, which GPU work wrote: nothing is read back.
bool RasterizerVulkan::Nx2ListDispatchSupported() const {
    return Tegra::Nx2ListDispatch::Mode() != 0;
}

// The memory seam, SEAM >= 3 (vk_mem_stage.cpp): KeplerCompute's QMD read comes through the open
// dispatch packet. False below 3: the engine reads it itself, as before.
bool RasterizerVulkan::MemStageQmd(Tegra::MemoryManager& memory, GPUVAddr address,
                                   std::span<u32, 64> out) {
    if (!mem_stage.ValuesByValue()) {
        return false;
    }
    const std::array<u32, 64>& qmd = mem_stage.Qmd(memory, address);
    std::copy(qmd.begin(), qmd.end(), out.begin());
    return true;
}

// List draws and list dispatches: a copy of the list words at their place in the stream, for the
// front to read back if it needs their values after all.
void RasterizerVulkan::Nx2KeepListWords(u64 keep_id, GPUVAddr address, u32 count) {
    // FRITCHII_ONEF_OP_LISTKEEP (fable_b 15.4): E's half, the buffer (its sync's commands an op
    // packet), here; R's, the copy into the keep buffer, a render op at the same place. The
    // front's read of the kept words is lent (Nx2ReadKeptListWords), so R has played the copy.
    if (VideoCommon::OneFollower::ShouldLend() &&
        VideoCommon::OneFollower::PostOpSite(static_cast<u8>(MemStage::Site::ListWords)))
        [[unlikely]] {
        VkBuffer handle = VK_NULL_HANDLE;
        u32 offset = 0;
        {
            std::scoped_lock lock{buffer_cache.mutex};
            const auto [buffer, buffer_offset] = mem_stage.Ref(MemStage::Site::ListWords, [&] {
                return buffer_cache.ObtainBuffer(
                    address, count * sizeof(u32),
                    VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                    VideoCommon::ObtainBufferOperation::DoNothing);
            });
            if (buffer == nullptr) {
                return;
            }
            handle = buffer->Handle();
            offset = buffer_offset;
        }
        struct Op {
            MemPlayer* player;
            u64 keep_id;
            VkBuffer handle;
            u32 offset;
            u32 count;
        } const op{&mem_player, keep_id, handle, offset, count};
        VideoCommon::OneFollower::SetLendSite("Nx2KeepListWords (posted)");
        VideoCommon::OneFollower::PushOp(
            VideoCommon::OneFollower::LentOp::Copy,
            [](void* raw) {
                Op a;
                std::memcpy(&a, raw, sizeof(a));
                a.player->KeepListWords(a.keep_id, a.handle, a.offset, a.count);
            },
            &op, sizeof(op));
        return;
    }
    ONEF_LEND(Copy, Nx2KeepListWords(keep_id, address, count))
    std::scoped_lock lock{buffer_cache.mutex};
    const auto [buffer, offset] = mem_stage.Ref(MemStage::Site::ListWords, [&] {
        return buffer_cache.ObtainBuffer(address, count * sizeof(u32),
                                         VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                                         VideoCommon::ObtainBufferOperation::DoNothing);
    });
    if (buffer == nullptr) {
        return;
    }
    mem_player.KeepListWords(keep_id, buffer->Handle(), offset, count);
}

bool RasterizerVulkan::Nx2ReadKeptListWords(u64 keep_id, std::span<u32> words) {
    // One follower: R's player owns the keep list and its copies (UNLEND's List bit records
    // an unlent list draw's keeps when R plays it; FRITCHII_ONEF_OP_LISTKEEP posts the copies):
    // lent, so R has played every packet before the read.
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        onef_list_keeps_unplayed = false; // the lend below plays everything first
    }
    ONEF_LEND_RESULT(ReadBack, Nx2ReadKeptListWords(keep_id, words))
    return mem_player.ReadKeptListWords(keep_id, words);
}

void RasterizerVulkan::DrawListWords() {
    namespace ListDraw = Tegra::Nx2ListDraw;
    using Link = VideoCommon::MemoryStageLink;
    auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& params = maxwell3d->draw_manager.indirect_state;
    // The draw command has one instance from instance 0 and counts vertices as they are drawn
    // (a task/mesh pipeline ignores all three; the pipeline is not known yet). EDEN_NX2_MESH_FIRST
    // pushes the first vertex to task/mesh pipelines from the CPU: those words are read back.
    const bool on_cpu = ListDrawReadsBack(draw_state, maxwell3d->regs);
    if (on_cpu) {
        // FRITCHII_MEMSTAGE_LIST_DRAWS: GPUMem keeps these level (the same test on its copy); a
        // served one here means the two sides' state differs (must be 0).
        if (Link::BackServed()) [[unlikely]] {
            Link::Note(Link::Check::ListReadBack, "DrawListWords");
        }
        const GPUVAddr address = params.indirect_start_address;
        std::array<u32, 2> words{};
        if (gpu_memory->IsMemoryDirty(address, sizeof(words),
                                      VideoCommon::CacheType::BufferCache)) {
            gpu_memory->FlushRegion(address, sizeof(words), VideoCommon::CacheType::BufferCache);
        }
        gpu_memory->ReadBlockUnsafe(address, words.data(), sizeof(words));
        ListDraw::Count(ListDraw::g_stats.read_back);
        if (words[0] == 0) {
            ListDraw::Count(ListDraw::g_stats.read_zero);
        }
        if (words[1] != 0) {
            ListDraw::Count(ListDraw::g_stats.read_first);
        }
        mem_player.KeepListWordsRead(params.list_keep_id, words);
        draw_state.vertex_buffer.count = words[0];
        draw_state.vertex_buffer.first = words[1];
        draw_state.draw_element_count = words[0];
        Draw(false, 1);
        if (words[0] != 0 && mesh_draw) {
            ListDraw::Count(ListDraw::g_stats.read_mesh);
        }
        return;
    }
    if (Common::DkbFrameTrace::Enabled()) {
        ++trace_indirect;
    }
    // FRITCHII_MEMSTAGE_LIST_DRAWS, a list draw GPUMem served (the entry is not lent): its fill
    // bound and synchronized the two words (the geometry section) and put their buffer in the
    // packet; PrepareDraw takes the packet and Configure records the copy from that buffer
    // (GraphicsPipeline::ConfigureDraw), without asking the cache. Nor does it set the cache's
    // indirect parameters (SetDrawIndirect, what the back's own fill reads): GPUMem runs ahead and
    // uses the cache meanwhile. Until now every list draw ran lent, with GPUMem parked.
    // One follower step 4, a list draw R plays unlent (UNLEND's List bit): E's fill did the same
    // (FillRing with the list arguments); the cache is E's and is not touched here.
    const bool served = Link::BackServed() || (VideoCommon::OneFollower::OnRenderThread() &&
                                               onef_draw->Playing() != nullptr);
    if (!served) {
        buffer_cache.SetDrawIndirect(&params);
    }
    mem_player.RequestListDrawArgs(params.list_keep_id);
    bool recorded = false;
    bool mesh = false;
    PrepareDraw(false, [this, &mesh, &recorded] {
        const auto args = mem_player.TakeListDrawArgs();
        if (args.buffer == VK_NULL_HANDLE) {
            return;
        }
        recorded = true;
        mesh = args.mesh;
        scheduler.Record([this, args](vk::CommandBuffer cmdbuf) {
            if (scheduler.GraphicsPipelineMissing()) {
                return;
            }
            if (args.mesh) {
                cmdbuf.DrawMeshTasksIndirectEXT(args.buffer, args.offset, 1, 0);
            } else {
                cmdbuf.DrawIndirect(args.buffer, args.offset, 1, 0);
            }
        });
    }, true);
    // A draw that was not prepared as far as its copy leaves the request behind.
    (void)mem_player.TakeListDrawArgs();
    // Also after PrepareDraw's fallback for a served list draw without its packet (it set them).
    if (!served || std::exchange(list_fallback_indirect, false)) {
        buffer_cache.SetDrawIndirect(nullptr);
    }
    ListDraw::Count(!recorded ? ListDraw::g_stats.not_recorded
                    : mesh    ? ListDraw::g_stats.indirect_mesh
                              : ListDraw::g_stats.indirect);
}

VideoCommon::OotNetwork::BypassState* RasterizerVulkan::OotNetworkBypass() {
    // Temporal upscaling replaces the same output pass and takes precedence until it fails.
    if (OcarinaTemporalEnabled() && !(oot_live_dlss && oot_live_dlss->EvaluationFailed())) {
        static bool logged = false;
        if (!std::exchange(logged, true)) {
            LOG_INFO(Render_Vulkan, "Ocarina of Time: No sharpening is off while temporal upscaling is on");
        }
        return nullptr;
    }
    return &oot_network_bypass;
}

void RasterizerVulkan::TrackOotTonemappedImage(const GraphicsPipeline& pipeline) {
    const auto& hashes = pipeline.ShaderHashes();
    if (std::none_of(hashes.begin(), hashes.end(), VideoCommon::OotNetwork::IsTonemapShader)) {
        return;
    }
    const auto& target = maxwell3d->regs.rt[0];
    if (target.Address() == 0) {
        return;
    }
    std::scoped_lock lock{oot_network_bypass.mutex};
    if (oot_network_bypass.tonemapped.Address() != target.Address()) {
        LOG_INFO(Render_Vulkan, "Ocarina of Time: tonemapped image at {:#x} {}x{}", target.Address(),
                 target.width, target.height);
    }
    oot_network_bypass.tonemapped = target;
}

void RasterizerVulkan::TraceDrawFailure(std::string_view reason) {
    if (!Common::DkbFrameTrace::Enabled()) {
        return;
    }
    ++trace_skipped;
    if (trace_failure_details++ >= 4) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    std::array<GPUVAddr, 6> stages{};
    for (size_t i = 0; i < stages.size(); ++i) {
        if (regs.IsShaderConfigEnabled(i)) {
            stages[i] = maxwell3d->GetNativeMacroEngine().native_turing
                            ? (u64{regs.pipelines[i].reserved[0] & 0xff} << 32) |
                                  regs.pipelines[i].reserved[1]
                            : regs.program_region.Address() + regs.pipelines[i].offset;
        }
    }
    LOG_INFO(Render_Vulkan,
             "DKB draw skipped frame={} reason={} stages[0..5]={:#x} rt0={:#x} {}x{}",
             trace_frame, reason, fmt::join(stages, ","), regs.rt[0].Address(),
             regs.rt[0].width, regs.rt[0].height);
}

void RasterizerVulkan::CaptureIndirectTrace() {
    if (!trace_sample_frame || trace_captures >= 4) {
        return;
    }
    ++trace_captures;
    const auto slot = std::find_if(indirect_traces.begin(), indirect_traces.end(),
                                   [](const auto& trace) { return !trace.pending; });
    if (slot == indirect_traces.end()) {
        ++trace_capture_dropped;
        return;
    }
    const auto& params = maxwell3d->draw_manager.indirect_state;
    const auto [commands, command_offset] = buffer_cache.GetDrawIndirectBuffer();
    const u32 max_draws = static_cast<u32>(params.max_draw_counts);
    std::array<VkBufferCopy, 8> copies{};
    u32 sampled{};
    const u64 command_bytes = params.is_byte_count ? 4 : 8;
    const u64 available = command_offset <= commands->SizeBytes()
                              ? std::min<u64>(params.buffer_size,
                                              commands->SizeBytes() - command_offset)
                              : 0;
    const u32 limit = params.is_byte_count ? 1 : std::min(max_draws, 8U);
    for (u32 i = 0; i < limit; ++i) {
        if (available < command_bytes || (i && params.stride > (available - command_bytes) / i)) {
            break;
        }
        const u64 offset = command_offset + i * params.stride;
        copies[sampled++] = VkBufferCopy{offset, i * 8ULL, command_bytes};
    }
    VkBuffer count_buffer{};
    VkBufferCopy count_copy{};
    if (params.include_count && !params.is_byte_count) {
        const auto [count, offset] = buffer_cache.GetDrawIndirectCount();
        if (offset > count->SizeBytes() || sizeof(u32) > count->SizeBytes() - offset) {
            ++trace_capture_dropped;
            return;
        }
        count_buffer = count->Handle();
        count_copy = VkBufferCopy{offset, 64, sizeof(u32)};
    }
    if (!sampled && !count_buffer) {
        LOG_INFO(Render_Vulkan, "DKB indirect frame={} address={:#x} max_draws={} sampled=0",
                 trace_frame, params.indirect_start_address, max_draws);
        return;
    }
    if (!slot->readback) {
        slot->readback = memory_allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = 68,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            },
            MemoryUsage::Download);
    }
    slot->frame = trace_frame;
    slot->address = params.indirect_start_address;
    slot->count_address = params.count_start_address;
    slot->max_draws = max_draws;
    slot->sampled_commands = sampled;
    slot->include_count = count_buffer != VK_NULL_HANDLE;
    slot->byte_count = params.is_byte_count;
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([source = commands->Handle(), destination = *slot->readback, copies, sampled,
                      count_buffer, count_copy](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, before);
        if (sampled) {
            cmdbuf.CopyBuffer(source, destination, vk::Span(copies.data(), sampled));
        }
        if (count_buffer) {
            cmdbuf.CopyBuffer(count_buffer, destination, count_copy);
        }
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                               0, after);
    });
    slot->tick = scheduler.CurrentTick();
    slot->pending = true;
    scheduler.RequestRenderpass(texture_cache.GetFramebuffer());
    scheduler.NoteRenderPassStores();
}

void RasterizerVulkan::CollectIndirectTrace() {
    for (auto& trace : indirect_traces) {
        if (!trace.pending || !scheduler.IsFree(trace.tick)) {
            continue;
        }
        trace.readback.Invalidate();
        std::array<u32, 17> words{};
        std::memcpy(words.data(), trace.readback.Mapped().data(), sizeof(words));
        if (trace.byte_count) {
            LOG_INFO(Render_Vulkan, "DKB indirect frame={} address={:#x} gpu_byte_count={}",
                     trace.frame, trace.address, words[0]);
        } else {
            const u32 count = trace.include_count ? words[16] : trace.max_draws;
            const u32 requested_draws = std::min(count, trace.max_draws);
            const u32 sampled = std::min(requested_draws, trace.sampled_commands);
            u32 zero_commands{};
            std::string commands;
            for (u32 i = 0; i < sampled; ++i) {
                zero_commands += words[2 * i] == 0 || words[2 * i + 1] == 0;
                commands += fmt::format("{}{}x{}", i ? "," : "", words[2 * i], words[2 * i + 1]);
            }
            LOG_INFO(Render_Vulkan,
                     "DKB indirect frame={} address={:#x} count_address={:#x} draw_count={} "
                     "count_source={} max_draws={} requested_draws={} sampled={} zero_commands={} "
                     "vertices_or_indices_x_instances=[{}] conditional_execution=unobserved",
                     trace.frame, trace.address, trace.count_address, count,
                     trace.include_count ? "gpu" : "command-parameter",
                     trace.max_draws, requested_draws, sampled, zero_commands, commands);
        }
        trace.pending = false;
    }
}

void RasterizerVulkan::OneFollowerMirrorDraw() {
    if ((VideoCommon::OneFollower::SyncMirrorOn() || VideoCommon::OneFollower::PostFenceOn()) &&
        channel_state->maxwell3d.regs.zpass_pixel_count_enable) {
        // FRITCHII_ESPLIT's queued mode: E1 is ahead of the reports and resets E2 still has to
        // run; setting the bit here would start the mirror before an earlier reset cleared it
        // (of27b: "R's samples counter ran where E's mirror had it stopped", POST_FENCE stopped
        // for the session and every fence and wait for idle was lent from then on). The set goes
        // through E2's queue, once per started run (E1's own copy says whether E2's will be set).
        if (VideoCommon::ESplit::QueueKind(VideoCommon::ESplit::KindRasterOps)) [[unlikely]] {
            if (!std::exchange(onef_mirror_queued, true)) {
                RasterizerVulkan* const self = this;
                std::memcpy(VideoCommon::ESplit::BeginAsync(VideoCommon::ESplit::Hand::Raster),
                            &self, sizeof(self));
                VideoCommon::ESplit::PostAsync([](void* raw) {
                    RasterizerVulkan* r;
                    std::memcpy(&r, raw, sizeof(r));
                    r->onef_samples_started = true;
                });
            }
            return;
        }
        onef_samples_started = true;
        onef_mirror_queued = true;
    }
}

namespace {
/// FRITCHII_ONEF_POST_KINDS (boot, default all; a bisect tool): which fence kinds may be posted
/// (the others are lent): 1 sync points, 2 references, 4 forced releases, 8 SignalFence
/// callbacks, 16 fence-flagged payload reports, 32 counter reports.
bool OneFollowerPostKind(u32 bit) {
    static const u32 mask = [] {
        const char* value = std::getenv("FRITCHII_ONEF_POST_KINDS");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 0)) : ~0U;
    }();
    return (mask & bit) != 0;
}
} // Anonymous namespace

bool RasterizerVulkan::OneFollowerFenceClean() {
    // fable_b 16.3 and 19.6: R's fence must not enter the buffer cache. Its three ways in: the
    // commit's buffer half (E runs it at the fence's place, R skips it), the query sync (none
    // owed: the mirror's test; E obtains the buffers for the values it syncs), the closing
    // gather (E runs it before posting, R skips it). Marks ahead still due stay lent.
    namespace OneF = VideoCommon::OneFollower;
    if (!OneF::PostFenceOn()) {
        OneF::CountFenceDecision(OneF::PostFenceRefusal());
        return false;
    }
    if (onef_sync_owed) {
        OneF::CountFenceDecision(OneF::FenceDecision::SyncOwed);
        return false;
    }
    // FRITCHII_ONEF_POST_SYNCS (boot, default 1; 0 = a bisect tool): 0 lends every fence that
    // owes query syncs (16.2's posted sync targets off).
    static const bool post_syncs = [] {
        const char* value = std::getenv("FRITCHII_ONEF_POST_SYNCS");
        return value == nullptr || value[0] != '0';
    }();
    if (!post_syncs && (!onef_sync_samples.empty() || !onef_sync_payloads.empty())) {
        OneF::CountFenceDecision(OneF::FenceDecision::SyncOwed);
        return false;
    }
    if (buffer_cache.AheadDue()) {
        OneF::CountFenceDecision(OneF::FenceDecision::AheadDue);
        return false;
    }
    return true;
}

void RasterizerVulkan::OneFollowerNoteSuperset(DAddr device) {
    if (onef_superset_pages.size() >= 4096) {
        onef_superset_pages.clear(); // no posted fence for a long while: forget, stay strict
    }
    onef_superset_pages.push_back(device & ~u64{Core::DEVICE_PAGEMASK});
}

void RasterizerVulkan::OneFollowerFenceLent() {
    onef_sync_owed = false;
    onef_sync_samples.clear();
    onef_sync_payloads.clear();
    VideoCommon::OneFollower::NoteFenceLent();
}

std::vector<OneFollowerSyncTarget>* RasterizerVulkan::OneFollowerSyncTargetsLocked() {
    if (onef_sync_samples.empty() && onef_sync_payloads.empty()) {
        return nullptr;
    }
    struct Slot {
        u64 page;
        bool payload;
    };
    std::vector<Slot> slots;
    for (const u64 address : onef_sync_samples) {
        slots.push_back({address & ~u64{Core::DEVICE_PAGEMASK}, false});
    }
    for (const u64 address : onef_sync_payloads) {
        slots.push_back({address & ~u64{Core::DEVICE_PAGEMASK}, true});
    }
    std::ranges::sort(slots, [](const Slot& a, const Slot& b) {
        return a.page != b.page ? a.page < b.page : a.payload < b.payload;
    });
    auto* const targets = new std::vector<OneFollowerSyncTarget>;
    buffer_cache.BufferOperations([&] {
        targets->clear();
        for (size_t i = 0; i < slots.size();) {
            // A run of contiguous pages (R groups its values the same way, by page).
            const u64 begin = slots[i].page;
            u64 end = begin + Core::DEVICE_PAGESIZE;
            bool payload_only = true;
            for (; i < slots.size() && slots[i].page < end + Core::DEVICE_PAGESIZE &&
                   slots[i].page >= begin;
                 ++i) {
                end = (std::max)(end, slots[i].page + Core::DEVICE_PAGESIZE);
                payload_only = payload_only && slots[i].payload;
            }
            const auto [buffer, offset] = buffer_cache.ObtainCPUBuffer(
                begin, static_cast<u32>(end - begin),
                VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                VideoCommon::ObtainBufferOperation::DoNothing);
            if (buffer != nullptr) {
                targets->push_back(
                    {begin, end - begin, buffer->Handle(), offset, payload_only, false});
            }
        }
    });
    // As SyncValues does (EXP49): read downloads taken before the copies are not the newest.
    for (const u64 address : onef_sync_samples) {
        buffer_cache.OutdateReadDownloads(address, 16);
    }
    for (const u64 address : onef_sync_payloads) {
        buffer_cache.OutdateReadDownloads(address, 16);
    }
    onef_sync_samples.clear(); // R syncs them there; payloads stay listed until a wait for idle
    return targets;
}

void RasterizerVulkan::OneFollowerCloseTargets(std::vector<OneFollowerSyncTarget>* targets) {
    if (targets != nullptr) {
        for (const auto& target : *targets) {
            if (target.used || target.payload) {
                continue;
            }
            // R's counter was stopped at a report E listed (E's mirror superset): nothing to
            // sync there, explained.
            if (std::ranges::any_of(onef_superset_pages, [&](u64 page) {
                    return page >= target.begin && page < target.begin + target.bytes;
                })) {
                VideoCommon::OneFollower::CountSyncTargetSuperset();
                continue;
            }
            thread_local std::string note;
            note = fmt::format("samples sync target {:#x}+{:#x} not used", target.begin,
                               target.bytes);
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::SyncTargetUnused,
                                           note.c_str());
        }
        delete targets;
    }
    onef_superset_pages.clear(); // the samples streamer synced here
}

void RasterizerVulkan::OneFollowerPostIdle() {
    namespace OneF = VideoCommon::OneFollower;
    // fable_b 22.3 / 16.2: E needs nothing back from a wait for idle. Its buffer cache parts are
    // E's at its place, in the single back's order: the query sync's buffers (NotifyWFI's), then
    // SignalOrdering's grouping of the flushes. R records the barrier and syncs into the named
    // buffers, skipping the grouping.
    std::vector<OneFollowerSyncTarget>* targets = nullptr;
    const bool syncs = !onef_sync_samples.empty() || !onef_sync_payloads.empty();
    const bool group = buffer_cache.MayHaveUncommittedFlushes() || buffer_cache.AheadDue();
    if (syncs || group) {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Query};
        if (syncs) {
            targets = OneFollowerSyncTargetsLocked();
            // The payload streamer syncs only in NotifyWFI (its SyncWrites): all of them here.
            onef_sync_payloads.clear();
        }
        if (group) {
            const VideoCommon::MemoryStageLink::BackCacheScope back_cache;
            buffer_cache.AccumulateFlushes();
        }
    }
    struct Op {
        RasterizerVulkan* self;
        std::vector<OneFollowerSyncTarget>* targets;
    } const op{this, targets};
    static_assert(sizeof(Op) <= OneF::MaxOpPayloadBytes, "an op over the payload limit");
    OneF::SetLendSite("WaitForIdle (posted)");
    OneF::PushOp(
        OneF::LentOp::ReadBack,
        [](void* raw) {
            Op a;
            std::memcpy(&a, raw, sizeof(a));
            auto& self = *a.self;
            self.fence_manager.skip_buffer_half = true;
            t_onef_sync_targets = a.targets;
            SCOPE_EXIT {
                self.fence_manager.skip_buffer_half = false;
                t_onef_sync_targets = nullptr;
                self.OneFollowerCloseTargets(a.targets);
            };
            self.WaitForIdle();
        },
        &op, sizeof(op));
}

void RasterizerVulkan::OneFollowerPostFence(u8 kind, u32 value,
                                            std::function<void()>* e_callback,
                                            const OneFollowerPayloadNote* note,
                                            const OneFollowerFenceReport* report) {
    namespace OneF = VideoCommon::OneFollower;
    // fable_b 19.6, E at the fence's place in the single back's order: the commit's buffer half,
    // the query sync's buffers (16.2), the callback, the closing gather; then the op.
    // 16.2: the buffers for the values R syncs at this fence, obtained here at its place (their
    // uploads an op packet before the fence, the Query site posted); the op names them.
    std::vector<OneFollowerSyncTarget>* targets = nullptr;
    const bool commit = buffer_cache.MayHaveUncommittedFlushes();
    const bool syncs = !onef_sync_samples.empty() || !onef_sync_payloads.empty();
    bool committed = false;
    if (commit || syncs) {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Query};
        if (commit) {
            // As FenceManager::CommitCacheFlushes' buffer half, which R skips: the downloads'
            // copies go to R in this seam's op packet, before the fence that carries them
            // (onef_buffer_committed: made, flushed, popped at its release).
            const VideoCommon::MemoryStageLink::BackCacheScope back_cache;
            buffer_cache.PromoteAhead();
            if (buffer_cache.HasUncommittedFlushes()) {
                buffer_cache.CommitAsyncFlushes();
                committed = true;
            }
            buffer_cache.RefreshUncommittedFlushesNote();
        }
        if (syncs) {
            targets = OneFollowerSyncTargetsLocked();
        }
    }
    OneF::CountFenceDecision(committed ? OneF::FenceDecision::PostedWithCommit
                                       : OneF::FenceDecision::Posted);
    // FRITCHII_ONEF_SIGNAL_ON_R: a fence-flagged payload is written by R in the op below, at its
    // played place; its page is one this fence lands (NoteFencePosted below), E's reads of it wait.
    const bool note_on_r = note != nullptr && OneF::SignalOnR();
    if (note_on_r) {
        OneF::CountSignalOnR();
        OneF::NoteReportPage(note->device);
    }
    // fable_b 16.4: the query cache's FinishReads, E's half: the reports this fence lands (those
    // since the last fence, a counter fence report's own) get the guest reads' downloads over
    // them landed here, so R writes them without waiting for the buffer lock (FinishRead).
    if (VideoCore::g_read_downloads.load(std::memory_order_acquire) != 0) [[unlikely]] {
        for (const u64 page : OneF::UnassignedReportPages()) {
            buffer_cache.FinishReadDownloads(page << 12, u64{1} << 12);
        }
    }
    // 19.6 step 2: the callback, here (FRITCHII_ONEF_SEM_E): a SignalFence's (kind 4), the
    // payload write of a synced fence report (kind 5, after its page's landings), the sync
    // point's host increment (kind 3; its guest increment ran before the commit, as there).
    if (e_callback != nullptr) {
        (*e_callback)();
    }
    if (note != nullptr && !note_on_r) {
        OneF::BeforeEngineRead(note->device, 16);
        query_cache.WritePayloadNow(note->device, note->gpu_addr, note->payload,
                                    note->has_timestamp, true);
    }
    // FRITCHII_ONEF_SEM_E_LANDED (a tester's Jabu-Jabu freezes, 2026-11-17): the host increment
    // ends the game's wait on the sync point, and the game may then free and reuse memory the
    // fence's downloads still land in (R's fence release, later): an old landing over the game's
    // new objects. 1: the increment moves to R's fence (after its landings) while committed
    // buffer downloads are pending; 2: always (as with FRITCHII_ONEF_SEM_E=0, the guest increment
    // stays on E). 0 (default): on E as before.
    bool host_on_r = false;
    if (kind == 3) {
        static const int landed_mode = [] {
            const char* const switch_value = std::getenv("FRITCHII_ONEF_SEM_E_LANDED");
            return switch_value != nullptr ? std::atoi(switch_value) : 0;
        }();
        if (landed_mode >= 2) {
            host_on_r = true;
        } else if (landed_mode == 1) {
            if (committed) {
                host_on_r = true;
            } else {
                std::scoped_lock lock{buffer_cache.mutex};
                host_on_r = buffer_cache.HasPendingDownloads();
            }
        }
        if (landed_mode != 0) {
            static std::atomic<u64> on_e{};
            static std::atomic<u64> on_r{};
            static std::atomic<u64> last_log_ns{};
            (host_on_r ? on_r : on_e).fetch_add(1, std::memory_order_relaxed);
            const u64 now_ns = Common::Nx2Hitch::NowNs();
            u64 last = last_log_ns.load(std::memory_order_relaxed);
            if (now_ns - last >= 10'000'000'000ULL &&
                last_log_ns.compare_exchange_strong(last, now_ns, std::memory_order_relaxed)) {
                if (last != 0) {
                    LOG_INFO(Render_Vulkan,
                             "Sync point increments (FRITCHII_ONEF_SEM_E_LANDED={}) 10 s: {} on E, "
                             "{} moved to R's fence (downloads still to land)",
                             landed_mode, on_e.exchange(0, std::memory_order_relaxed),
                             on_r.exchange(0, std::memory_order_relaxed));
                }
            }
        }
        if (!host_on_r) {
            fence_manager.OneFollowerSyncPointHost(value);
        }
    }
    // The closing gather, E's half at this place (GATHER_E's path, or lent).
    InvalidateGPUCache();
    struct Op {
        RasterizerVulkan* self;
        std::vector<OneFollowerSyncTarget>* targets;
        u32 value;
        u8 kind;
        bool committed;
        bool note_on_r; ///< kind 5: R writes the payload (FRITCHII_ONEF_SIGNAL_ON_R)
        bool host_on_r; ///< kind 3: R's fence runs the host increment (FRITCHII_ONEF_SEM_E_LANDED)
        OneFollowerPayloadNote note;
        OneFollowerFenceReport report;
    } const op{this, targets, value, kind, committed, note_on_r, host_on_r,
               note != nullptr ? *note : OneFollowerPayloadNote{},
               report != nullptr ? *report : OneFollowerFenceReport{}};
    static_assert(sizeof(Op) <= OneF::MaxOpPayloadBytes, "a fence op over the payload limit");
    OneF::SetLendSite(kind == 0   ? "SignalSyncPoint (posted)"
                      : kind == 1 ? "SignalReference (posted)"
                      : kind == 2 ? "ReleaseFences (posted)"
                      : kind == 3 ? "SignalSyncPoint (posted, increments on E)"
                      : kind == 4 ? "SignalFence (posted, callback on E)"
                      : kind == 5 ? (note_on_r ? "payload fence (posted, written on R)"
                                               : "payload fence (posted, written on E)")
                                  : "fence report (posted)");
    OneF::PushOp(
        OneF::LentOp::Fence,
        [](void* raw) {
            Op a;
            std::memcpy(&a, raw, sizeof(a));
            auto& self = *a.self;
            self.onef_fence_posted = true;
            self.fence_manager.skip_buffer_half = true;
            self.fence_manager.onef_buffer_committed = a.committed;
            t_onef_sync_targets = a.targets;
            SCOPE_EXIT {
                self.onef_fence_posted = false;
                self.fence_manager.skip_buffer_half = false;
                self.fence_manager.onef_buffer_committed = false;
                t_onef_sync_targets = nullptr;
                self.OneFollowerCloseTargets(a.targets);
            };
            switch (a.kind) {
            case 0:
                self.fence_manager.SignalSyncPoint(a.value);
                break;
            case 3:
                if (a.host_on_r) {
                    // FRITCHII_ONEF_SEM_E_LANDED: the host increment at this fence's release,
                    // after its downloads (and every earlier fence's) landed.
                    auto* const fences = &self.fence_manager;
                    const u32 syncpoint = a.value;
                    std::function<void()> increment(
                        [fences, syncpoint] { fences->OneFollowerSyncPointHost(syncpoint); });
                    self.fence_manager.SignalFence(std::move(increment));
                    break;
                }
                self.fence_manager.SignalReference();
                break;
            case 1:
            case 4:
                // An empty callback (3, 4: E ran the increments or the callback).
                self.fence_manager.SignalReference();
                break;
            case 7: {
                // FRITCHII_ONEF_POST_FENCE_REPORT: the counter report, then its fence
                // (SignalFence, here on R), as the lent Query ran them.
                const OneFollowerFenceReport& r = a.report;
                if (r.mirror != 0) {
                    const bool render_bit = self.query_cache.ReportMakesPendingSync(r.type);
                    VideoCommon::OneFollower::CountSyncMirror(r.mirror == 2, render_bit);
                    if (r.mirror == 2 && !render_bit) {
                        self.OneFollowerNoteSuperset(r.device);
                    }
                }
                self.query_cache.CounterReportAt(r.gpu_addr, r.device, r.type, r.flags, r.payload,
                                                 r.subreport);
                break;
            }
            case 5:
                // E wrote the payload (or R here, SIGNAL_ON_R): the streamer's pending sync,
                // then the fence.
                if (a.note_on_r) {
                    self.query_cache.WritePayloadNow(a.note.device, a.note.gpu_addr,
                                                     a.note.payload, a.note.has_timestamp, true);
                }
                self.query_cache.NotePayloadWritten(a.note.device, a.note.has_timestamp,
                                                    a.note.payload, a.note.subreport);
                self.fence_manager.SignalReference();
                break;
            default:
                // Immediate fences: the callback runs inside SignalFence, so the wait in it ends
                // at once (PostFenceOn excludes delayed fences, whose wait is the fence thread's).
                self.fence_manager.WaitPendingFences(true);
                break;
            }
        },
        &op, sizeof(op), note_on_r);
    // fable_b 13.0: the reports since the last fence may land at this one, at R's place.
    OneF::NoteFencePosted(OneF::PushedNow());
}

// ---------------------------------------------------------------------------------------------
// Posted dispatches (FRITCHII_ONEF_OP_DISPATCH; fable_b 11.1 item 5, 13.4 group 6)
//
// Who does what (the draw's split, section 3, applied to DispatchCompute):
// - E, in stream order: the stream's invalidations (FlushCaching), the compute lookup (a plain
//   hit only: PipelineCache::OneFollowerComputePipeline; its OoT hooks run here, on E's engine,
//   exactly once), the buffer side by value into ring B (MemStage::FillRingDispatch: uniform and
//   storage binds, uploads, and the GPU-written marks the storage binds make, so later reads and
//   flushes on E see them in order), the handle values (E reads game memory at its place), what
//   R reads of the compute engine (the launch description, the pools, the code address, the
//   native mode: OneFollowerDispatchView), then the dispatch op naming the packet. E goes on.
// - R, at the op: takes the packet by name and runs DispatchCompute's played path on its view of
//   the compute engine: the texture side (image views, samplers, descriptors), the pipeline bind
//   and vkCmdDispatch. It takes no buffer cache lock and calls nothing of the buffer cache.
// - Bit 2: dispatches that write images post too. R resolves and records the written images
//   (the texture cache's state is R's); E records them for EXP10 as "may write anything" at its
//   place (their guest ranges are the texture cache's), and E's own texture-side tests already
//   wait for R or lend (FLUSH_E, INVAL_E's page filter, the read-ahead stamps). Written
//   byte-addressed images (the packed R16 shadow) stay lent.
// - Bit 4: indirect dispatches post too. The GPU reads their arguments; E obtains the buffer in
//   its fill (a section after `before`, played at the packet's rest after Configure, as the
//   lent path's Ref plays it there) and the packet names it.
// - Stays lent (each counted, DispatchCount): indirect dispatches and image-writing ones while
//   their bit is off, texel buffers, the OoT upscaler passes and the depth
//   reader the replacement keys on, a lookup that is not a plain hit (it may build or retire),
//   diagnostics, and anything while the view is not seeded. A lent dispatch E looked up already
//   takes E's pipeline into the lent call (no second lookup, no second run of the hooks).
// - After the post, two buffer-cache tails that record after the dispatch run lent (E parked, R
//   past the dispatch): the write extent probes' copies and the command list page copies.
// - The QMD release after the dispatch is the engine's (SignalFence or Query, as before):
//   nothing here moves it.
// - Rule 1 / fable_b 18: the dispatch op packs its spans by difference and clears no dirty flag
//   on E's engine; the compute lookup reads none.

namespace {
/// What R's played dispatch reads of E's compute engine, by value (ring B, after the values).
struct OneFollowerDispatchView {
    std::array<u32, Tegra::Engines::KeplerCompute::LaunchParams::NUM_LAUNCH_PARAMETERS> launch;
    std::array<u32, 3> tsc; ///< address high, low, limit
    std::array<u32, 3> tic;
    std::array<u32, 2> code_loc;
    u32 tex_cb_index;
    u32 native;
    u64 indirect;       ///< the indirect arguments' GPU address (has_indirect)
    u32 has_indirect;   ///< bit 4: an indirect dispatch (the packet's `indirect` is its buffer)
    u32 writes_images;  ///< bit 2: the program writes images (R checks it against Configure)
    /// FRITCHII_ONEF_DISPATCH_RANGES: E's description of the written image elements, in
    /// WrittenImages' order: the descriptor it read (R reads it again at its play and compares)
    /// and the range it recorded for EXP10.
    static constexpr u32 MaxImageRecords = 8;
    u32 image_count;  ///< elements described (0: none, or more than MaxImageRecords)
    u32 image_ranged; ///< 1: E recorded their ranges (else "may write anything")
    std::array<Tegra::Texture::TICEntry, MaxImageRecords> image_tic;
    std::array<u64, MaxImageRecords> image_tic_address; ///< 0: no image there (a null view)
    std::array<u64, MaxImageRecords> image_device;      ///< the recorded device range
    std::array<u64, MaxImageRecords> image_size;        ///< its size, 0: not one device range
};
static_assert(sizeof(OneFollowerDispatchView) % 8 == 0);

/// FRITCHII_ONEF_DISPATCH_RANGES (fable_b 27.9), E: the guest range of the image a written image
/// handle names, as the texture cache resolves a compute image view (VisitImageView,
/// CreateImageView): the TIC entry at the handle's index (TextureHeaderIndex, the pool's limit),
/// its view header, the image's start (its address less the base layer) and its guest size
/// (ImageInfo, CalculateGuestSizeInBytes). `tic` and `tic_address` get the entry read;
/// `null_view` is set where the texture cache binds no image (the index past the limit, an
/// invalid entry): nothing is written there. nullopt otherwise when the range is not one mapped
/// device range, or the entry is a buffer image: the record stays "may write anything".
struct WrittenImageRange {
    u64 device;
    u64 size;
};
std::optional<WrittenImageRange> DescribeWrittenImage(Tegra::MemoryManager& memory,
                                                      const Tegra::Engines::KeplerCompute& engine,
                                                      u32 raw, Tegra::Texture::TICEntry& tic,
                                                      GPUVAddr& tic_address, bool& null_view) {
    using Tegra::Texture::TICEntry;
    null_view = false;
    tic = TICEntry{};
    tic_address = 0;
    const bool via_header = engine.launch_description.linked_tsc != 0;
    const u32 view_index = Tegra::Texture::TexturePair(raw, via_header).first;
    const u32 limit = engine.regs.tic.limit;
    const bool native = memory.UsesNativeTiling();
    const u32 index = Tegra::Texture::TextureHeaderIndex(view_index, limit, native);
    if (index > limit) {
        null_view = true;
        return std::nullopt;
    }
    tic_address = engine.regs.tic.Address() + u64{index} * sizeof(TICEntry);
    if (!memory.IsFullyMappedRange(tic_address, sizeof(TICEntry))) {
        return std::nullopt;
    }
    memory.ReadBlockUnsafe(tic_address, &tic, sizeof(tic));
    if (!VideoCommon::IsValidEntry(memory, tic)) {
        null_view = true;
        return std::nullopt;
    }
    const TICEntry config = VideoCommon::TextureViewHeader(tic, native);
    const VideoCommon::ImageInfo info(config);
    if (info.type == VideoCommon::ImageType::Buffer) {
        return std::nullopt;
    }
    const u64 layer_offset = u64{config.BaseLayer()} * info.layer_stride;
    const GPUVAddr image_address = config.Address() - layer_offset;
    const u64 size = VideoCommon::CalculateGuestSizeInBytes(info);
    if (size == 0 || !memory.IsContinuousRange(image_address, size)) {
        return std::nullopt;
    }
    const std::optional<DAddr> device = memory.GpuToCpuAddress(image_address);
    if (!device) {
        return std::nullopt;
    }
    return WrittenImageRange{*device, size};
}
static_assert(sizeof(Tegra::Engines::KeplerCompute::LaunchParams) ==
              sizeof(OneFollowerDispatchView::launch));
} // Anonymous namespace

bool RasterizerVulkan::OneFollowerPostDispatch() {
    namespace OneF = VideoCommon::OneFollower;
    using Count = OneF::DispatchCount;
    bool off = false;
    const Count blocker = OneF::PostDispatchBlocker(off);
    if (off) {
        return false; // the switch is off: today's lend, uncounted
    }
    const auto lend = [](Count why) {
        OneF::CountDispatch(why);
        return false;
    };
    if (blocker != Count::Count) {
        return lend(blocker);
    }
    // Readers of the dispatch on R or of E's state on R, and the stage's own modes.
    static const bool trace_lends = [] {
        const char* value = std::getenv("FRITCHII_ONEF_TRACE_LENDS");
        return value != nullptr && value[0] == '1';
    }();
    if (onef_draw->Diagnostic() || (trace_lends && Common::DkbFrameTrace::Enabled()) ||
        InvalidateBeforeUse() ||
        (MemStage::VerifyBits() & MemStage::VerifyHandles) != 0 ||
        VideoCommon::MemStageStats::Enabled() ||
        VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed) ||
        VideoCommon::MemoryStageLink::RingActive()) {
        return lend(Count::Diagnostic);
    }
    if (!mem_stage.Packets() || !mem_stage.ValuesByValue() || mem_stage.Ticks() == nullptr ||
        !mem_stage.RingRoom() || buffer_cache.MarksAhead()) {
        return lend(Count::NoRing);
    }
    // E reads the channel's own compute engine, never the rasterizer's `kepler_compute`: R points
    // that member at its view while it plays a posted dispatch (OneFollowerPlayDispatch), at the
    // same time as E fills the next one (as `maxwell3d` names R's 3D view on R). Read through the
    // member, E took a played dispatch's launch description, pools and indirect address into the
    // next packet: wrong grids, handles and argument buffers (the flashes and GPU stall of
    // measure\of7-flash-log.txt, once nothing lent kept E and R apart in a compute pass).
    if (channel_state == nullptr) [[unlikely]] {
        return lend(Count::Transient);
    }
    Tegra::Engines::KeplerCompute& engine = channel_state->kepler_compute;
    const u32 bits = OneF::DispatchPostBits();
    // Bit 4: an indirect dispatch's arguments are read by the GPU only (DispatchIndirect); this
    // path reads them on the CPU nowhere (the frame dump's read is a diagnostic, lent above; the
    // engine's own debug read-back, EDEN_NX2_INDIRECT_GRID=2, ran before this call, on E).
    const std::optional<GPUVAddr> indirect_address = engine.GetIndirectComputeAddress();
    if (indirect_address && (bits & OneF::DispatchPostIndirect) == 0) {
        return lend(Count::Indirect);
    }
    // DispatchCompute's head, E's half: the stream's invalidations before the lookup reads.
    gpu_memory->FlushCaching();
    // A throw (a rejected program, a replaced pass) leaves DispatchCompute here on E, as a lent
    // dispatch's exception is rethrown on E.
    ComputePipeline* const pipeline = pipeline_cache.OneFollowerComputePipeline();
    if (pipeline == nullptr) {
        return lend(Count::LookupMiss); // nothing ran: the lent call looks up
    }
    // The lookup ran here: a lend from now on takes this pipeline with it.
    const auto lend_with = [&](Count why) {
        onef_dispatch_lent_pipeline = pipeline;
        OneF::CountDispatch(why);
        return false;
    };
    // FRITCHII_ASYNC_COMPUTE_SKIP: an unbuilt pipeline's dispatch is decided in the lent call
    // (DispatchCompute's SkipUnbuiltDispatch), before anything is filled or marked here. With
    // the test delay (any mode), the lent call also makes it wait as a building one would.
    if (AsyncComputeSkip::Mode() != 0 || AsyncComputeSkip::DelayMs() > 0) [[unlikely]] {
        if (async_compute_skip.CountsUnbuilt(*pipeline)) {
            return lend_with(Count::Unbuilt);
        }
        async_compute_skip.NoteRan(*pipeline);
    }
    const MemStage::DispatchFacts facts = pipeline->MemFacts();
    if (facts.has_texel_buffers) {
        return lend_with(Count::TexelBuffers);
    }
    const bool writes_images = pipeline->WritesImages();
    if (writes_images) {
        if ((bits & OneF::DispatchPostImages) == 0) {
            return lend_with(Count::WritesImage);
        }
        // The packed R16 surface path (Configure's byte-addressed shadow) can throw and makes its
        // own resources: lent.
        if (pipeline->WritesByteAddressedImages()) {
            return lend_with(Count::ImageShadow);
        }
    }
    // DispatchCompute's own OoT upscaler hooks (the replacement, its 3/4-scale pass, the depth
    // reader TryOcarinaUpscale keys on): R's members decide them, lent.
    constexpr u64 ThreeQuarterShader = 0xef542df87cacfe9eULL;
    constexpr u64 DepthReaderShader = 0x1fc7a0423116281fULL;
    const u64 hash = pipeline->ShaderHash();
    if (pipeline->CanReplaceOcarinaUpscale() || hash == ComputePipeline::OcarinaFullSizeShader ||
        hash == ComputePipeline::Ocarina1440pShader || hash == ThreeQuarterShader ||
        hash == DepthReaderShader) {
        return lend_with(Count::Upscaler);
    }
    const auto& qmd = engine.launch_description;
    const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
        return lend_with(Count::Dimensions);
    }
    // The handle values at E's place (Configure takes them from the packet on R).
    const bool null_pointer_handles = VideoCore::Nx2Debug::Experiment(21) != 0;
    pipeline->MakeHandleValues(engine, *gpu_memory, null_pointer_handles,
                               onef_dispatch_values);
    OneFollowerDispatchView view{};
    std::memcpy(view.launch.data(), &qmd, sizeof(view.launch));
    const auto& regs = engine.regs;
    view.tsc = {regs.tsc.address_high, regs.tsc.address_low, regs.tsc.limit};
    view.tic = {regs.tic.address_high, regs.tic.address_low, regs.tic.limit};
    view.code_loc = {regs.code_loc.address_high, regs.code_loc.address_low};
    view.tex_cb_index = regs.tex_cb_index;
    view.native = engine.IsNativeMode() ? 1 : 0;
    view.indirect = indirect_address.value_or(0);
    view.has_indirect = indirect_address ? 1 : 0;
    view.writes_images = writes_images ? 1 : 0;
    // FRITCHII_ONEF_DISPATCH_RANGES (fable_b 27.9): each written image's range from its
    // descriptor, at E's place (EXP10 below), and what R compares at its play.
    bool image_ranged = false;
    if (writes_images && OneF::DispatchRangesOn()) {
        u32 count = 0;
        bool all = true;
        const bool visited = pipeline->VisitWrittenImageHandles(
            onef_dispatch_values, [&](u32 raw) {
                if (count >= OneFollowerDispatchView::MaxImageRecords) {
                    ++count;
                    all = false;
                    return;
                }
                const u32 k = count++;
                GPUVAddr tic_address = 0;
                bool null_view = false;
                const auto range = DescribeWrittenImage(*gpu_memory, engine, raw,
                                                        view.image_tic[k], tic_address, null_view);
                view.image_tic_address[k] = null_view ? 0 : tic_address;
                view.image_device[k] = range ? range->device : 0;
                view.image_size[k] = range ? range->size : 0;
                all = all && (null_view || range.has_value());
            });
        const bool described = visited && count <= OneFollowerDispatchView::MaxImageRecords;
        view.image_count = described ? count : 0;
        image_ranged = described && all;
        view.image_ranged = image_ranged ? 1 : 0;
        OneF::CountDispatch(image_ranged ? Count::ImageRanged : Count::ImageUnbounded);
    }
    const u64 position = VideoCommon::GpuSplit::Get().BackEntryPosition();
    u64 number = 0;
    bool list_copies = false;
    bool probes = false;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        SCOPE_EXIT {
            mem_stage.EndStageFill(); // E's mark, however the fill ended
        };
        // As DispatchCompute: writes to command list pages are noted while the binds mark.
        buffer_cache.Nx2CollectListCopies(true);
        number = mem_stage.FillRingDispatch(buffer_cache, facts, position, onef_dispatch_values,
                                            null_pointer_handles, &view, sizeof(view),
                                            indirect_address);
        if (number != 0) {
            list_copies = buffer_cache.Nx2HasListCopyRequests();
            probes = buffer_cache.HasUnrecordedExtentProbes();
            // FRITCHII_ASYNC_COMPUTE_SKIP: its written ranges' streaks, stamped in R's frame
            // when R plays the packet (OneFollowerPlayDispatch).
            if (AsyncComputeSkip::Mode() != 0) [[unlikely]] {
                boost::container::small_vector<AsyncComputeSkip::Range, 8> writes;
                static_cast<void>(CollectComputeWrites(writes));
                if (!writes.empty()) {
                    async_compute_skip.QueueWrites(number, writes);
                }
            }
        }
        if (!list_copies) {
            buffer_cache.Nx2CollectListCopies(false);
        }
    }
    if (number == 0) {
        return lend_with(Count::FillRefused); // nothing filled: the lent call fills
    }
    // EXP10: the dispatch's buffer writes were recorded by the fill's marks on E. What
    // DispatchCompute's block after Configure adds, here at E's place:
    // - its written images: their guest ranges are the texture cache's (R's, later), so E
    //   records "may write anything" (DispatchUnbounded::Image, the record the lent path makes
    //   for an image without a guest range): a superset, so a front read across it drains;
    // - the indirect arguments ObtainBuffer dropped the GPU-written state of (12 bytes).
    if (VideoCommon::GpuSplit::RecordingDispatch()) {
        auto& split = VideoCommon::GpuSplit::Get();
        if (writes_images && image_ranged) {
            // FRITCHII_ONEF_DISPATCH_RANGES: what the dispatch can write of each image, the
            // range of the descriptor it binds (R compares the descriptor at its play).
            for (u32 k = 0; k < view.image_count; ++k) {
                if (view.image_size[k] != 0) {
                    split.RecordDispatchWrite(view.image_device[k], view.image_size[k],
                                               VideoCommon::GpuSplit::DispatchWriteKind::Image);
                }
            }
        } else if (writes_images) {
            split.RecordDispatchUnbounded(VideoCommon::GpuSplit::DispatchUnbounded::Image);
        }
        if (indirect_address) {
            if (const auto device_addr = gpu_memory->GpuToCpuAddress(*indirect_address)) {
                split.RecordDispatchWrite(*device_addr, 12,
                                          VideoCommon::GpuSplit::DispatchWriteKind::Indirect);
            }
        }
        split.ConfirmDispatch();
    }
    struct Op {
        RasterizerVulkan* self;
        ComputePipeline* pipeline;
        u64 number;
    } const op{this, pipeline, number};
    OneF::SetLendSite("DispatchCompute (posted)");
    OneF::PushOp(
        OneF::LentOp::Dispatch,
        [](void* raw) {
            Op a;
            std::memcpy(&a, raw, sizeof(a));
            a.self->OneFollowerPlayDispatch(a.pipeline, a.number);
        },
        &op, sizeof(op));
    OneF::CountDispatch(Count::Posted);
    if (writes_images) {
        OneF::CountDispatch(Count::PostedImage);
    }
    if (indirect_address) {
        OneF::CountDispatch(Count::PostedIndirect);
    }
    if (list_copies || probes) [[unlikely]] {
        // The buffer cache's records after the dispatch, in DispatchCompute's order (the probes'
        // "after" copies, then the list page copies and their submit), lent: R has played the
        // dispatch when it runs them, E parked.
        OneF::CountDispatch(list_copies ? Count::TailListCopies : Count::TailProbes);
        if (list_copies && probes) {
            OneF::CountDispatch(Count::TailProbes);
        }
        auto tail = [this, list_copies, probes, hash] {
            std::scoped_lock lock{buffer_cache.mutex};
            if (probes) {
                static_cast<void>(mem_stage.Ref(MemStage::Site::DispatchAfter, [&] {
                    return buffer_cache.RecordComputeExtentProbes();
                }));
            }
            if (list_copies) {
                static_cast<void>(SubmitNx2ListCopies(hash)); // on R: runs in place
            }
            buffer_cache.Nx2CollectListCopies(false);
        };
        OneF::SetLendSite("DispatchCompute (posted, tail)");
        OneF::Lend(OneF::LentOp::Dispatch, tail);
    }
    return true;
}

void RasterizerVulkan::OneFollowerCheckGrids() {
    namespace OneF = VideoCommon::OneFollower;
    if (onef_grid_checks.empty()) {
        return;
    }
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    std::erase_if(onef_grid_checks, [&](OneFollowerGridCheck& check) {
        if (!scheduler.IsFree(check.tick)) {
            return false;
        }
        std::array<u32, 3> grid{};
        std::memcpy(grid.data(), check.ref.mapped_span.data(), sizeof(grid));
        if (check.audit) [[unlikely]] {
            // FRITCHII_GRID_AUDIT: every indirect grid, counted; the first 16 bad ones and then
            // every 256th logged.
            static u64 audited{};
            static u64 bad{};
            ++audited;
            const bool over = grid[0] > max_dim[0] || grid[1] > max_dim[1] || grid[2] > max_dim[2];
            const bool high = grid[1] > 0xffff || grid[2] > 0xffff;
            if (over || high) {
                ++bad;
                if (bad <= 16 || bad % 256 == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "Dispatch grid audit: indirect dispatch of {:016x} ran grid "
                                "{}x{}x{} ({}{}), limit {}x{}x{}; {} bad of {} audited",
                                check.shader, grid[0], grid[1], grid[2],
                                over ? "over the device limit" : "",
                                high ? (over ? ", y/z past 16 bits" : "y/z past 16 bits") : "",
                                max_dim[0], max_dim[1], max_dim[2], bad, audited);
                }
            } else if ((audited & (audited - 1)) == 0 && audited >= 1024) {
                LOG_INFO(Render_Vulkan,
                         "Dispatch grid audit: {} indirect grids audited, {} bad (last {}x{}x{})",
                         audited, bad, grid[0], grid[1], grid[2]);
            }
            staging_pool.FreeDeferred(check.ref);
            return true;
        }
        if (grid[0] > max_dim[0] || grid[1] > max_dim[1] || grid[2] > max_dim[2]) [[unlikely]] {
            thread_local std::string note;
            note = fmt::format("{} indirect dispatch of {:016x}: grid {}x{}x{}, limit {}x{}x{}",
                               check.posted ? "a posted" : "a lent", check.shader, grid[0],
                               grid[1], grid[2], max_dim[0], max_dim[1], max_dim[2]);
            if (check.posted) {
                OneF::Note(OneF::Check::IndirectGridOverLimit, note.c_str());
                OneF::StopPostDispatch("a posted indirect dispatch's grid over the device limit");
            } else {
                OneF::CountDispatch(OneF::DispatchCount::LentGridOverLimit);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_WARNING(Render_Vulkan, "One follower (dry): {}", note);
                }
            }
        }
        staging_pool.FreeDeferred(check.ref);
        return true;
    });
}

void RasterizerVulkan::OneFollowerCheckPlayedImages(const ComputePipeline& pipeline,
                                                    const void* raw_view) {
    namespace OneF = VideoCommon::OneFollower;
    using Count = OneF::DispatchCount;
    const auto& view = *static_cast<const OneFollowerDispatchView*>(raw_view);
    std::scoped_lock lock{texture_cache.mutex};
    const auto written = pipeline.WrittenImages();
    // fable_b 27.9's count, every played image-writing dispatch: the images' real ranges (the
    // texture cache's backing images, what the lent path records) on a page the front reads.
    for (const VideoCommon::ImageViewId id : written) {
        if (!id || id == VideoCommon::NULL_IMAGE_VIEW_ID) {
            continue;
        }
        const ImageView& image_view = texture_cache.GetImageView(id);
        const Image* const image = image_view.BackingImage();
        if (image != nullptr && image->cpu_addr_end > image->cpu_addr &&
            OneF::FrontReadPagesAny(image->cpu_addr, image->cpu_addr_end - image->cpu_addr)) {
            OneF::CountDispatch(Count::ImageOnFrontPage);
        }
    }
    if (view.image_ranged == 0) {
        return; // E recorded "may write anything": nothing to compare
    }
    const auto fail = [](const char* what) {
        OneF::Note(OneF::Check::DispatchImageRange, what);
        OneF::StopDispatchRanges(what);
    };
    if (written.size() != view.image_count) {
        fail("a played dispatch wrote another number of images than E described");
        return;
    }
    for (size_t k = 0; k < written.size(); ++k) {
        const VideoCommon::ImageViewId id = written[k];
        const bool bound = id && id != VideoCommon::NULL_IMAGE_VIEW_ID;
        if (view.image_tic_address[k] == 0) {
            if (bound) {
                fail("a played dispatch writes an image where E saw none");
                return;
            }
            continue;
        }
        // The descriptor R's texture cache took at this play, read again now (the stream's
        // writes to the pool are E's; one E made after the push would differ here).
        Tegra::Texture::TICEntry tic{};
        gpu_memory->ReadBlockUnsafe(view.image_tic_address[k], &tic, sizeof(tic));
        if (std::memcmp(&tic, &view.image_tic[k], sizeof(tic)) != 0) {
            fail("a played dispatch's image descriptor differs from the one E recorded");
            return;
        }
        if (!bound) {
            continue; // no image made (no memory): it writes nothing
        }
        const Image* const image = texture_cache.GetImageView(id).BackingImage();
        if (image == nullptr || image->cpu_addr_end <= image->cpu_addr) {
            continue;
        }
        const u64 begin = view.image_device[k];
        const u64 end = begin + view.image_size[k];
        if (view.image_size[k] == 0 || image->cpu_addr < begin || image->cpu_addr_end > end) {
            OneF::CountDispatch(Count::ImageWiderThanRecord);
        }
    }
}

void RasterizerVulkan::OneFollowerPlayDispatch(ComputePipeline* pipeline, u64 number) {
    namespace OneF = VideoCommon::OneFollower;
    OneFollowerDispatchView view;
    if (mem_stage.TakeRingDispatch(number, &view, sizeof(view)) == nullptr) {
        // "ring B pairing" counted: the two sides disagree on the stream. Nothing is held.
        OneF::StopPostDispatch("a dispatch op found another packet at ring B's head");
        return;
    }
    // However this ends, the packet is played whole: the cache believes its uploads recorded.
    const MemStage::RestAtEnd rest{mem_stage};
    async_compute_skip.PlayedWrites(number); // FRITCHII_ASYNC_COMPUTE_SKIP's streaks (E's fill)
    if (channel_state == nullptr || gpu_memory == nullptr) [[unlikely]] {
        OneF::Note(OneF::Check::PlayedDispatch, "a played dispatch with no channel bound");
        OneF::StopPostDispatch("a played dispatch with no channel bound");
        return;
    }
    // R's view of the compute engine: what the played path reads of it, from E's packet.
    if (!onef_kepler_view || onef_kepler_view_memory != gpu_memory) {
        onef_kepler_view = std::make_unique<Tegra::Engines::KeplerCompute>(*gpu_memory);
        onef_kepler_view_memory = gpu_memory;
    }
    Tegra::Engines::KeplerCompute& engine = *onef_kepler_view;
    std::memcpy(&engine.launch_description, view.launch.data(), sizeof(view.launch));
    engine.regs.tsc.address_high = view.tsc[0];
    engine.regs.tsc.address_low = view.tsc[1];
    engine.regs.tsc.limit = view.tsc[2];
    engine.regs.tic.address_high = view.tic[0];
    engine.regs.tic.address_low = view.tic[1];
    engine.regs.tic.limit = view.tic[2];
    engine.regs.code_loc.address_high = view.code_loc[0];
    engine.regs.code_loc.address_low = view.code_loc[1];
    engine.regs.tex_cb_index = view.tex_cb_index;
    engine.SetViewNativeMode(view.native != 0);
    engine.SetViewIndirect(view.has_indirect != 0 ? std::optional<GPUVAddr>{view.indirect}
                                                 : std::nullopt);
    onef_dispatch_writes_images = view.writes_images != 0;
    // The R-owned objects that read the compute engine name the view for this dispatch (the
    // pipeline cache does not look up here: the packet carries E's pipeline).
    Tegra::Engines::KeplerCompute* const saved_rasterizer = ExchangeKeplerCompute(&engine);
    Tegra::Engines::KeplerCompute* const saved_texture = texture_cache.ExchangeKeplerCompute(&engine);
    onef_dispatch_play = pipeline;
    SCOPE_EXIT {
        onef_dispatch_play = nullptr;
        texture_cache.ExchangeKeplerCompute(saved_texture);
        ExchangeKeplerCompute(saved_rasterizer);
    };
    try {
        DispatchCompute();
        if (view.writes_images != 0) {
            OneFollowerCheckPlayedImages(*pipeline, &view);
        }
    } catch (const std::exception& error) {
        // The lent path would throw to E's engine (a skipped dispatch); R cannot.
        thread_local std::string note;
        note = fmt::format("a played dispatch threw: {}", error.what());
        OneF::Note(OneF::Check::PlayedDispatch, note.c_str());
        OneF::StopPostDispatch("a played dispatch threw");
    } catch (...) {
        OneF::Note(OneF::Check::PlayedDispatch, "a played dispatch threw");
        OneF::StopPostDispatch("a played dispatch threw");
    }
}

void RasterizerVulkan::DrawTexture() {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        OneFollowerMirrorDraw();
    }
    ONEF_LEND(Draw, DrawTexture())

    SCOPE_EXIT {
        gpu.TickWork();
    };
    FlushWork();

    std::scoped_lock l{texture_cache.mutex};
    texture_cache.SynchronizeDescriptors(false);
    texture_cache.UpdateRenderTargets(false);
    if (texture_cache.RenderTargetsOutOfMemory()) [[unlikely]] {
        return; // CheezyEmu a08015f893: a bound target got no memory
    }

    UpdateDynamicStates(nullptr);

    query_cache.NotifySegment(true);
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    if (maxwell3d->regs.zpass_pixel_count_enable != 0) {
        query_cache_runtime.NoteOtherCountedDraw(); // FRITCHII_OCCL_FLAG: not a flag draw
    }
    const auto& draw_texture_state = maxwell3d->draw_manager.draw_texture_state;
    const auto& sampler = texture_cache.GetSampler(draw_texture_state.src_sampler, false);
    const auto& texture = texture_cache.GetImageView(draw_texture_state.src_texture);
    const auto* framebuffer = texture_cache.GetFramebuffer();
    if (frame_dump.Capturing() || flash_watch.WatchesPasses()) {
        FrameDumpPass(framebuffer);
    }
    if (flash_watch.WatchesPasses()) {
        flash_watch.NoteDraw({});
    }

    const bool src_rescaling = texture_cache.IsRescaling() && texture.IsRescaled();
    const bool dst_rescaling = texture_cache.IsRescaling() && framebuffer->IsRescaled();

    const auto ScaleSrc = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return src_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    const auto ScaleDst = [&](auto dim_f) -> s32 {
        auto dim = static_cast<s32>(dim_f);
        return dst_rescaling ? Settings::values.resolution_info.ScaleUp(dim) : dim;
    };

    Region2D dst_region = {Offset2D{.x = ScaleDst(draw_texture_state.dst_x0),
                                    .y = ScaleDst(draw_texture_state.dst_y0)},
                           Offset2D{.x = ScaleDst(draw_texture_state.dst_x1),
                                    .y = ScaleDst(draw_texture_state.dst_y1)}};
    Region2D src_region = {Offset2D{.x = ScaleSrc(draw_texture_state.src_x0),
                                    .y = ScaleSrc(draw_texture_state.src_y0)},
                           Offset2D{.x = ScaleSrc(draw_texture_state.src_x1),
                                    .y = ScaleSrc(draw_texture_state.src_y1)}};
    Extent3D src_size = {static_cast<u32>(ScaleSrc(texture.size.width)),
                         static_cast<u32>(ScaleSrc(texture.size.height)), texture.size.depth};
    blit_image.BlitColor(framebuffer, texture.RenderTarget(), texture.ImageHandle(),
                         sampler->Handle(), dst_region, src_region, src_size);
}

// A 1-layer depth target bound with a 3D color target limits the framebuffer to one layer, so
// vkCmdClearAttachments can't reach the target's other slices. Those slices are zeroed by copying
// zeros into the clear's rectangle of exactly the slices the clear names; the rest of the volume
// keeps its contents. The rectangle can be smaller than the target: in handheld mode the game
// clears a 48-wide fog volume through a 64-wide render target.
bool RasterizerVulkan::ClearVolumeSlices(const Framebuffer* framebuffer, u32 first_slice,
                                         u32 slice_count) {
    const auto& regs = maxwell3d->regs;
    const u32 target_index = regs.clear_surface.RT;
    const auto& color_buffer_ids = texture_cache.GetRenderTargets().color_buffer_ids;
    // Depth and stencil clears take the normal path, so only a colour-only clear qualifies.
    if (target_index >= color_buffer_ids.size() || regs.clear_surface.Z || regs.clear_surface.S) {
        return false;
    }
    const auto view_id = color_buffer_ids[target_index];
    const u32 depth = regs.rt[target_index].depth;
    if (!view_id || depth <= framebuffer->Layers() || first_slice >= depth) {
        return false;
    }
    const auto& view = texture_cache.GetImageView(view_id);
    const bool is_volume = True(view.flags & VideoCommon::ImageViewFlagBits::Slice);
    const bool clears_all_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool clears_to_zero =
        std::ranges::all_of(regs.clear_color, [](f32 value) { return value == 0.0f; });
    if (!is_volume || !clears_all_channels || !clears_to_zero) {
        return false;
    }
    const VkRect2D rect = VolumeClearRect(regs, framebuffer->RenderArea(), view.size);
    if (rect.extent.width == 0 || rect.extent.height == 0) {
        return true;
    }
    const u32 width = rect.extent.width;
    const u32 height = rect.extent.height;
    // The host image can use a wider format than the guest's (B10G11R11 falls back to
    // R16G16B16A16): size the zeros for the widest color texel, 16 bytes.
    constexpr size_t WidestTexelBytes = 16;
    const size_t slice_bytes = size_t{width} * height * WidestTexelBytes;
    const auto zeros = staging_pool.Request(slice_bytes, MemoryUsage::Upload, false,
                                            VideoCommon::BufferChurnStats::StagingCaller::Rasterizer);
    std::memset(zeros.mapped_span.data(), 0, slice_bytes);
    const u32 count = std::min(slice_count, depth - first_slice);
    const VkImage image = view.ImageHandle();
    const u32 level = view.range.base.level;
    // The clear's slices count from the view's first slice.
    const u32 base_slice = static_cast<u32>(view.range.base.layer);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image, level, base_slice, rect, width, height, first_slice, count,
                      buffer = zeros.buffer, offset = zeros.offset](vk::CommandBuffer cmdbuf) {
        // Earlier passes may still be reading or writing the volume.
        constexpr VkMemoryBarrier clear_waits{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, clear_waits);
        for (u32 slice = first_slice; slice < first_slice + count; ++slice) {
            const VkBufferImageCopy zero_slice{
                .bufferOffset = offset,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                .imageOffset = {rect.offset.x, rect.offset.y, static_cast<s32>(base_slice + slice)},
                .imageExtent = {width, height, 1},
            };
            cmdbuf.CopyBufferToImage(buffer, image, VK_IMAGE_LAYOUT_GENERAL, zero_slice);
        }
        constexpr VkMemoryBarrier clear_finished{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               0, clear_finished);
    });
    return true;
}

void RasterizerVulkan::Clear(u32 layer_count) {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        OneFollowerMirrorDraw(); // a superset: R starts the count only for a clear it cannot defer
    }
    // FRITCHII_ONEF_OP_CLEAR (fable_b 13.4 group 3): E has decided ShouldExecute (DrawManager)
    // and runs the clear's one E half, FlushCaching, here; the rest is R's (render targets and
    // clear values are view registers, the texture cache and scheduler R's): a render op, its
    // R-only flags by rule 1 (PacketRenderState).
    if (VideoCommon::OneFollower::ShouldLend() && VideoCommon::OneFollower::PostClears())
        [[unlikely]] {
        gpu_memory->FlushCaching();
        struct Op {
            RasterizerVulkan* self;
            u32 layer_count;
        } const op{this, layer_count};
        VideoCommon::OneFollower::SetLendSite("Clear (posted)");
        VideoCommon::OneFollower::PushOp(
            VideoCommon::OneFollower::LentOp::Clear,
            [](void* raw) {
                Op a;
                std::memcpy(&a, raw, sizeof(a));
                a.self->onef_clear_posted = true;
                SCOPE_EXIT {
                    a.self->onef_clear_posted = false;
                };
                a.self->Clear(a.layer_count);
            },
            &op, sizeof(op), false, VideoCommon::OneFollower::PacketRenderState);
        return;
    }
    ONEF_LEND(Clear, Clear(layer_count))
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Clear};
    const VideoCommon::FritchiiDrawKinds::Scope draw_kinds_scope{maxwell3d->regs, true};
    if (Tegra::Engines::g_nx2_trace_draws_after_upscale > 0) {
        const auto& regs = maxwell3d->regs;
        LOG_INFO(HW_GPU, "after upscale: clear rt0 {:#x} {}x{} fmt {:#x} surface {:#x} colour {} {} {} {}",
                 regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height,
                 static_cast<u32>(regs.rt[0].format), regs.clear_surface.raw, regs.clear_color[0],
                 regs.clear_color[1], regs.clear_color[2], regs.clear_color[3]);
    }
    TraceWindowTarget(*maxwell3d, *gpu_memory, "clear", layer_count);
    // Slow clears (over 300 us) are logged with where the time went, at most 40 per 5 s: at the
    // Castle Town route start the back's clear zone went 0.7 -> 4.6 ms a frame while an NPC walked
    // past, at about the same number of clears.
    const auto clear_start = std::chrono::steady_clock::now();
    const bool alloc_timing = Vulkan::MemoryAllocator::ImageTimingEnabled();
    const auto alloc_before = alloc_timing ? Vulkan::MemoryAllocator::ReadImageTiming()
                                           : Vulkan::MemoryAllocator::ImageTiming{};
    FlushWork();
    if (!onef_clear_posted) {
        gpu_memory->FlushCaching(); // a posted clear's ran on E, at its place
    }
    const auto clear_flushed = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point clear_targets_ready{};
    // What the texture cache did for this clear's targets (JoinImages, framebuffers).
    std::optional<TextureCache::RenderTargetStats> clear_tc_stats;
    SCOPE_EXIT {
        const auto end = std::chrono::steady_clock::now();
        const auto us = [](auto d) {
            return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
        };
        if (end - clear_start < std::chrono::microseconds{300}) {
            return;
        }
        static std::atomic<s64> window_start{};
        static std::atomic<u32> logged{};
        const s64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               end.time_since_epoch()).count();
        if (now_ms - window_start.load(std::memory_order_relaxed) > 5000) {
            window_start.store(now_ms, std::memory_order_relaxed);
            logged.store(0, std::memory_order_relaxed);
        }
        if (logged.fetch_add(1, std::memory_order_relaxed) >= 40) {
            return;
        }
        const auto& r = maxwell3d->regs;
        const s64 targets_us = clear_targets_ready.time_since_epoch().count() != 0
                                   ? us(clear_targets_ready - clear_flushed)
                                   : -1;
        LOG_INFO(Render_Vulkan,
                 "Slow clear {} us (flush work {} us, render targets {} us): rt0 {:#x} {}x{} fmt "
                 "{:#x}, zeta {:#x} {} fmt {:#x}, surface {:#x}, layers {}",
                 us(end - clear_start), us(clear_flushed - clear_start), targets_us,
                 r.rt[0].Address(), r.rt[0].width, r.rt[0].height, static_cast<u32>(r.rt[0].format),
                 r.zeta.Address(), r.zeta_enable ? "on" : "off", static_cast<u32>(r.zeta.format),
                 r.clear_surface.raw, layer_count);
        if (clear_tc_stats) {
            const auto& s = *clear_tc_stats;
            const auto ns_us = [](u64 ns) { return ns / 1000; };
            LOG_INFO(Render_Vulkan,
                     "Slow clear texture cache: {} joins, {} created ({} pooled), {} relocated, "
                     "{} retired, {} deleted, {} "
                     "overlaps walked ({} bad), {} KiB uploaded; framebuffers {} created, {} "
                     "removed, {} scanned, {} in the map; {} images; walk {} us, create {} us, "
                     "upload {} us, delete {} us, framebuffers {} us",
                     s.joins, s.images_created, s.pooled_images, s.relocated_images,
                     s.retired_images, s.images_deleted,
                     s.overlaps_walked, s.bad_overlaps, s.uploaded_bytes >> 10,
                     s.framebuffers_created, s.framebuffers_removed, s.framebuffers_scanned,
                     s.framebuffer_map_size, s.image_count,
                     ns_us(s.walk_ns), ns_us(s.create_ns), ns_us(s.refresh_ns),
                     ns_us(s.delete_ns), ns_us(s.framebuffer_ns));
        }
        if (alloc_timing) {
            const auto after = Vulkan::MemoryAllocator::ReadImageTiming();
            LOG_INFO(Render_Vulkan,
                     "Slow clear allocator: {} images, vkCreateImage {} us, VMA allocate {} us, "
                     "bind {} us",
                     after.images - alloc_before.images,
                     (after.create_ns - alloc_before.create_ns) / 1000,
                     (after.alloc_ns - alloc_before.alloc_ns) / 1000,
                     (after.bind_ns - alloc_before.bind_ns) / 1000);
        }
    };

    auto& regs = maxwell3d->regs;
    const bool use_color = regs.clear_surface.R || regs.clear_surface.G || regs.clear_surface.B ||
                           regs.clear_surface.A;
    const bool use_depth = regs.clear_surface.Z;
    const bool use_stencil = regs.clear_surface.S;
    if (!use_color && !use_depth && !use_stencil) {
        return;
    }

    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.ResetRenderTargetStats();
    texture_cache.UpdateRenderTargets(true);
    if (texture_cache.RenderTargetsOutOfMemory()) [[unlikely]] {
        clear_tc_stats = texture_cache.TakeRenderTargetStats();
        return; // CheezyEmu a08015f893: a bound target got no memory
    }
    clear_targets_ready = std::chrono::steady_clock::now();
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    clear_tc_stats = texture_cache.TakeRenderTargetStats();
    if (frame_dump.Capturing() || flash_watch.WatchesPasses()) {
        FrameDumpPass(framebuffer);
    }
    const VkExtent2D render_area = framebuffer->RenderArea();
    const u32 base_array_layer = regs.clear_surface.layer;
    const auto clear_layers =
        GetFramebufferClearLayers(base_array_layer, layer_count, framebuffer->Layers());
    if (clear_layers.count != layer_count) {
        LOG_DEBUG(Render_Vulkan,
                  "Framebuffer clear layers: base={}, requested={}, clipped={}, framebuffer={}",
                  base_array_layer, layer_count, clear_layers.count, framebuffer->Layers());
    }
    if (ClearVolumeSlices(framebuffer, base_array_layer, layer_count)) {
        return;
    }
    if (clear_layers.count == 0) {
        return;
    }

    constexpr bool ENABLE_DEFERRED_CLEAR = true;
    const bool color_full_channels = regs.clear_surface.R && regs.clear_surface.G &&
                                     regs.clear_surface.B && regs.clear_surface.A;
    const bool stencil_partial = use_stencil && framebuffer->HasAspectStencilBit() &&
                                 regs.stencil_front_mask != 0xFF && regs.stencil_front_mask != 0;
    const bool ds_used = use_depth || use_stencil;
    const bool ds_deferrable =
        !ds_used || ((!framebuffer->HasAspectDepthBit() || use_depth) &&
                     (!framebuffer->HasAspectStencilBit() || use_stencil) && !stencil_partial);
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    VkRect2D default_scissor{};
    default_scissor.offset.x = 0;
    default_scissor.offset.y = 0;
    default_scissor.extent.width = (std::numeric_limits<s32>::max)();
    default_scissor.extent.height = (std::numeric_limits<s32>::max)();

    VkClearRect clear_rect{
        .rect = regs.clear_control.use_scissor ? GetScissorState(regs, 0, up_scale, down_shift)
                                               : default_scissor,
        .baseArrayLayer = base_array_layer,
        .layerCount = clear_layers.count,
    };
    const auto clamp_rect_to_render_area = [render_area](VkRect2D& rect) -> bool {
        const auto clamp_axis = [](s32& offset, u32& extent, u32 limit) {
            auto clamp_offset = [&offset, limit]() {
                if (limit == 0) {
                    offset = 0;
                    return;
                }
                offset = std::clamp(offset, 0, static_cast<s32>(limit));
            };

            if (extent == 0) {
                clamp_offset();
                return;
            }
            if (offset < 0) {
                const u32 shrink = (std::min)(extent, static_cast<u32>(-offset));
                extent -= shrink;
                offset = 0;
            }
            if (limit == 0) {
                extent = 0;
                offset = 0;
                return;
            }
            if (offset >= s32(limit)) {
                offset = s32(limit);
                extent = 0;
                return;
            }
            const u64 end_coord = u64(offset) + extent;
            if (end_coord > limit) {
                extent = limit - u32(offset);
            }
        };

        clamp_axis(rect.offset.x, rect.extent.width, render_area.width);
        clamp_axis(rect.offset.y, rect.extent.height, render_area.height);
        return rect.extent.width != 0 && rect.extent.height != 0;
    };
    if (!clamp_rect_to_render_area(clear_rect.rect)) {
        return;
    }

    if (VideoCommon::RenderTrace::SampleFrame(render_trace_frame) && render_trace_clears++ < 2048) {
        const u32 target = regs.clear_surface.RT;
        LOG_DEBUG(Render_Vulkan,
                  "Render trace clear: frame={} docked={} color={} depth={} stencil={} "
                  "color_target={} color_address={:#x} depth_address={:#x} depth_value={} base_layer={} layers={} "
                  "framebuffer_layers={} rect=({},{} {}x{})",
                  render_trace_frame, Settings::IsDockedMode(), use_color, use_depth, use_stencil,
                  target, target < regs.rt.size() ? regs.rt[target].Address() : 0,
                  regs.zeta.Address(), regs.clear_depth,
                  clear_rect.baseArrayLayer, clear_rect.layerCount, framebuffer->Layers(),
                  clear_rect.rect.offset.x, clear_rect.rect.offset.y,
                  clear_rect.rect.extent.width, clear_rect.rect.extent.height);
    }

    const bool clear_covers_render_area =
        clear_rect.rect.offset.x == 0 && clear_rect.rect.offset.y == 0 &&
        clear_rect.rect.extent.width >= render_area.width &&
        clear_rect.rect.extent.height >= render_area.height;
    const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && (!regs.clear_control.use_scissor || clear_covers_render_area) &&
                                 clear_layers.covers_all_layers &&
                                 !scheduler.IsRenderPassActive() &&
                                 (!use_color || color_full_channels) && ds_deferrable &&
                                 // FRITCHII_ONEF_CLEAR_NODEFER: diagnostic for the lean darkness
                                 // (posted clears with unlent array draws), clear at once.
                                 !(onef_clear_posted &&
                                   VideoCore::Nx2Debug::SwitchValue(
                                       VideoCore::Nx2Debug::g_onef_clear_nodefer,
                                       "FRITCHII_ONEF_CLEAR_NODEFER") != 0);
    if (!can_defer_clear) {
        scheduler.RequestRenderpass(framebuffer);
    }

    query_cache.NotifySegment(true);
    // Deferred load-op clears have no render pass yet. Starting a query here
    // would begin it outside the pass and end it inside the next draw's pass.
    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64,
                              !can_defer_clear && regs.zpass_pixel_count_enable);
    if (maxwell3d->regs.zpass_pixel_count_enable != 0) {
        query_cache_runtime.NoteOtherCountedDraw(); // FRITCHII_OCCL_FLAG: not a flag draw
    }
    UpdateViewportsState(regs);

    const u32 color_attachment = regs.clear_surface.RT;
    if (use_color && framebuffer->HasAspectColorBit(color_attachment)) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[color_attachment].format);
        bool is_integer = IsPixelFormatInteger(format);
        bool is_signed = IsPixelFormatSignedInteger(format);
        size_t int_size = PixelComponentSizeBitsInteger(format);
        VkClearValue clear_value{};
        if (!is_integer) {
            std::memcpy(clear_value.color.float32, regs.clear_color.data(), regs.clear_color.size() * sizeof(f32));
        } else if (!is_signed) {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.uint32[i] = u32(f32(u64(int_size) << 1U) * regs.clear_color[i]);
        } else {
            for (size_t i = 0; i < 4; i++)
                clear_value.color.int32[i] = s32(f32(s64(int_size - 1) << 1) * (regs.clear_color[i] - 0.5f));
        }

        if (color_full_channels) {
            if (can_defer_clear) {
                scheduler.DeferColorClear(framebuffer, color_attachment, clear_value);
            } else {
                scheduler.Record([color_attachment, clear_value, clear_rect](vk::CommandBuffer cmdbuf) {
                    const VkClearAttachment attachment{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .colorAttachment = color_attachment,
                        .clearValue = clear_value,
                    };
                    cmdbuf.ClearAttachments(attachment, clear_rect);
                });
            }
        } else {
            u8 color_mask = u8(regs.clear_surface.R | regs.clear_surface.G << 1 | regs.clear_surface.B << 2 | regs.clear_surface.A << 3);
            Region2D dst_region = {
                Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
                Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                         .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
            blit_image.ClearColor(framebuffer, color_mask, regs.clear_color, dst_region);
        }
    }

    if (!use_depth && !use_stencil) {
        return;
    }
    VkImageAspectFlags aspect_flags = 0;
    if (use_depth && framebuffer->HasAspectDepthBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (use_stencil && framebuffer->HasAspectStencilBit()) {
        aspect_flags |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (aspect_flags == 0) {
        return;
    }

    if (use_stencil && framebuffer->HasAspectStencilBit() && regs.stencil_front_mask != 0xFF &&
        regs.stencil_front_mask != 0) {
        Region2D dst_region = {
            Offset2D{.x = clear_rect.rect.offset.x, .y = clear_rect.rect.offset.y},
            Offset2D{.x = clear_rect.rect.offset.x + s32(clear_rect.rect.extent.width),
                     .y = clear_rect.rect.offset.y + s32(clear_rect.rect.extent.height)}};
        blit_image.ClearDepthStencil(framebuffer, use_depth, regs.clear_depth,
                                     u8(regs.stencil_front_mask), regs.clear_stencil,
                                     regs.stencil_front_func_mask, dst_region);
    } else if (can_defer_clear) {
        VkClearValue ds_value{};
        ds_value.depthStencil.depth = regs.clear_depth;
        ds_value.depthStencil.stencil = regs.clear_stencil;
        scheduler.DeferDepthStencilClear(framebuffer, ds_value);
    } else {
        scheduler.Record([clear_depth = regs.clear_depth, clear_stencil = regs.clear_stencil,
                          clear_rect, aspect_flags](vk::CommandBuffer cmdbuf) {
            VkClearAttachment attachment;
            attachment.aspectMask = aspect_flags;
            attachment.colorAttachment = 0;
            attachment.clearValue.depthStencil.depth = clear_depth;
            attachment.clearValue.depthStencil.stencil = clear_stencil;
            cmdbuf.ClearAttachments(attachment, clear_rect);
        });
    }
}

bool RasterizerVulkan::CollectComputeWrites(
    boost::container::small_vector<AsyncComputeSkip::Range, 8>& ranges) {
    ranges.clear();
    return buffer_cache.ForEachComputeWriteMark([&](DAddr addr, u64 size) {
        ranges.push_back({addr, size, AsyncComputeSkip::Kind::Buffer});
    });
}

bool RasterizerVulkan::SkipUnbuiltDispatch(ComputePipeline& pipeline, bool indirect,
                                           bool special_pass) {
    // FRITCHII_ASYNC_COMPUTE_SKIP (vk_async_compute_skip.h; idea and rules from
    // zortos293/Lichii #2104 by zortos293). Both caches' locks are held, nothing of this
    // dispatch is filled yet. The guards in order: G4 (shape, then extents), G3, G1, G2.
    using Skip = AsyncComputeSkip;
    const int mode = Skip::Mode();
    const auto run = [&](Skip::Refusal why, bool sticky_only = false) {
        async_compute_skip.NoteRefusal(why, pipeline.ShaderHash(), sticky_only);
        return false;
    };
    const MemStage::DispatchFacts facts = pipeline.MemFacts();
    // G4: a plain native direct dispatch. Global stores have no binding to judge; texel buffers
    // bind through image views (made by the texture cache); the packed R16 path makes its own
    // shadow; the OoT upscaler and effects passes have hooks of their own.
    if (!kepler_compute->IsNativeMode() || indirect || special_pass || frame_dump.Capturing() ||
        facts.has_texel_buffers || facts.info->stores_global_memory ||
        pipeline.WritesByteAddressedImages()) {
        return run(Skip::Refusal::Shape); // Configure binds as always
    }
    // Configure's pointer binds, ahead (they resolve the written ranges and mark nothing). From
    // here a dispatch that runs keeps them: Configure does not bind again.
    pipeline.BindStoragePointers(buffer_cache);
    const auto run_bound = [&](Skip::Refusal why, bool sticky_only = false) {
        pipeline.UsePreboundPointers();
        return run(why, sticky_only);
    };
    boost::container::small_vector<Skip::Range, 8> ranges;
    bool bounded = CollectComputeWrites(ranges);
    if (bounded && pipeline.WritesImages()) {
        // The written images' guest ranges from their descriptors, as E records them for EXP10
        // (DescribeWrittenImage): read, nothing made.
        thread_local std::vector<u32> values;
        pipeline.MakeHandleValues(*kepler_compute, *gpu_memory,
                                  VideoCore::Nx2Debug::Experiment(21) != 0, values);
        const bool visited = pipeline.VisitWrittenImageHandles(values, [&](u32 raw) {
            Tegra::Texture::TICEntry tic{};
            GPUVAddr tic_address{};
            bool null_view{};
            const auto range =
                DescribeWrittenImage(*gpu_memory, *kepler_compute, raw, tic, tic_address, null_view);
            if (range) {
                ranges.push_back({range->device, range->size, Skip::Kind::Image});
            } else if (!null_view) {
                bounded = false; // not one device range, or a buffer image
            }
        });
        bounded = bounded && visited;
    }
    if (!bounded) {
        return run_bound(Skip::Refusal::Extent);
    }
    for (const Skip::Range& range : ranges) {
        // G3: never a page the guest reads back (this file's sticky reads, and the buffer
        // cache's Preflushable marks, CPU-read ranges and read downloads).
        const auto buffer_facts = buffer_cache.ComputeSkipRangeFacts(range.addr, range.size);
        if (buffer_facts.read_back) {
            return run_bound(Skip::Refusal::ReadBack);
        }
        if (async_compute_skip.GuestReadSeen(range)) {
            return run_bound(Skip::Refusal::ReadBack, true);
        }
        if (range.kind == Skip::Kind::Buffer) {
            // G1: every byte GPU-written, no CPU write to upload. G2: every frame.
            if (!buffer_facts.all_gpu_written || buffer_facts.cpu_modified) {
                return run_bound(Skip::Refusal::NoHistory);
            }
            if (!async_compute_skip.BufferStreakHolds(pipeline, range)) {
                return run_bound(Skip::Refusal::NotEveryFrame);
            }
            continue;
        }
        using State = TextureCache::SkipImageState;
        switch (texture_cache.ComputeSkipImageState(range.addr, range.size)) {
        case State::Redrawn:
            break;
        case State::NoImage:
        case State::NoGpuData:
            return run_bound(Skip::Refusal::NoHistory);
        case State::NotRedrawn:
            if (!async_compute_skip.ChainHolds(pipeline, range)) {
                return run_bound(Skip::Refusal::NotEveryFrame);
            }
            break;
        }
    }
    if (mode == 2) {
        // Dry: it waits as before; counted as a skip mode 1 would make.
        async_compute_skip.NoteSkip(pipeline, ranges, true);
        pipeline.UsePreboundPointers();
        return false;
    }
    // A pipeline really building (not the test delay's) starts on a spare thread now, and may
    // still make it within what is left of the frame's wait budget.
    if (!pipeline.IsBuilt()) {
        async_compute_skip.Kick(pipeline);
        if (pipeline_cache.WaitComputeWithinBudget(pipeline)) {
            pipeline.UsePreboundPointers();
            return false;
        }
    }
    async_compute_skip.NoteSkip(pipeline, ranges, false);
    return true;
}

void RasterizerVulkan::DispatchCompute() {
    // FRITCHII_DELAY_DISPATCH_US (live, 0 = off): the delay test "K", on the back thread only.
    // Off: one load and a branch.
    VideoCommon::GpuSplit::DelayWorkIf(VideoCore::Nx2Debug::g_delay_dispatch_us,
                                       VideoCommon::GpuSplit::DelayTest::Dispatch);
    // FRITCHII_ONEF_OP_DISPATCH ("Posted dispatches" below): E posts the dispatch, or lends it
    // with the pipeline it looked up already.
    // (Not again for a lend run in place, R stopping: its pipeline is set.)
    if (VideoCommon::OneFollower::ShouldLend() && onef_dispatch_lent_pipeline == nullptr &&
        OneFollowerPostDispatch()) [[unlikely]] {
        return;
    }
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        // FRITCHII_ONEF_DISPATCH_RANGES (fable_b 27.9): the lent call records on R with E's
        // recording state (GpuSplit's flag is per thread, E's), so the lent path's own EXP10
        // block runs and confirms: its images' real ranges instead of "unconfirmed", which
        // counts as "may write anything". E is parked: R is the records' only writer then.
        const bool recording = VideoCommon::OneFollower::DispatchRangesOn() &&
                               VideoCommon::GpuSplit::RecordingDispatch();
        VideoCommon::OneFollower::SetLendSite(__func__);
        auto onef_call = [&] {
            VideoCommon::GpuSplit::AdoptRecordingDispatch(recording);
            SCOPE_EXIT {
                VideoCommon::GpuSplit::AdoptRecordingDispatch(false);
            };
            DispatchCompute();
        };
        VideoCommon::OneFollower::Lend(VideoCommon::OneFollower::LentOp::Dispatch, onef_call);
        return;
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Compute};
    // One follower: R plays a posted dispatch (E ran the stream's invalidations, the lookup and
    // the buffer side; the held ring B packet is the fill), or runs a lent one whose pipeline E
    // looked up (its hooks ran on E: no second lookup). A played dispatch touches no part of the
    // buffer cache below (E's, E running): every buffer-side step tests `onef_play`.
    ComputePipeline* const onef_play = onef_dispatch_play;
    ComputePipeline* const onef_lent = std::exchange(onef_dispatch_lent_pipeline, nullptr);
    t_in_dispatch = true;
    SCOPE_EXIT {
        t_in_dispatch = false;
    };
    if (InvalidateBeforeUse()) {
        gpu.InvalidateGPUCache();
    }
    FlushWork();
    if (onef_play == nullptr) {
        gpu_memory->FlushCaching();
    }
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.DrainIfFull();
    }
    if (flash_watch.WatchesPasses()) {
        flash_watch.EndPass();
    }

    ComputePipeline* const pipeline{[&] {
        if (onef_play != nullptr) {
            return onef_play;
        }
        if (onef_lent != nullptr) {
            return onef_lent;
        }
        const VideoCommon::Nx2Profile::Scope lookup_scope{
            VideoCommon::Nx2Profile::Zone::ComputeLookup};
        try {
            return pipeline_cache.CurrentComputePipeline();
        } catch (const Shader::Exception& error) {
            if (frame_dump.Capturing()) {
                frame_dump.NoteSkipped(true,
                                       kepler_compute->regs.code_loc.Address() +
                                           kepler_compute->launch_description.program_start,
                                       0, error.what());
            }
            throw;
        }
    }()};
    if (!pipeline) {
        if (kepler_compute->IsNativeMode()) {
            // A program rejected before returns no pipeline without throwing (ComputeShader).
            if (frame_dump.Capturing()) {
                frame_dump.NoteSkipped(true,
                                       kepler_compute->regs.code_loc.Address() +
                                           kepler_compute->launch_description.program_start,
                                       0, "rejected (cached)");
            }
            throw Shader::RuntimeError("Native compute pipeline unavailable");
        }
        return;
    }
    VideoCommon::Nx2Profile::CountComputeProgram(pipeline->ShaderHash());
    // EDEN_NX2_MEMSTAGE_STATS: the dispatch, per channel, and each program's first one.
    if (VideoCommon::MemStageStats::Enabled()) [[unlikely]] {
        VideoCommon::MemStageStats::Dispatch();
        VideoCommon::MemStageStats::ComputeProgram(pipeline->ShaderHash());
    }
    // A played dispatch takes only the texture cache's lock: the buffer cache's is E's (R waiting
    // for it outside a lent call is the "R waits for E's lock" must-be-0).
    std::unique_lock texture_lock{texture_cache.mutex, std::defer_lock};
    std::unique_lock buffer_lock{buffer_cache.mutex, std::defer_lock};
    if (onef_play == nullptr) {
        std::lock(texture_lock, buffer_lock);
    } else {
        texture_lock.lock();
    }
    // fable_memstage_compute_brief 3 (GPU profile 2): the dispatch by what its fill marks (the
    // cache's lock is held from here to the end), and what else it took.
    struct DispatchKinds {
        bool recorded{};
        bool indirect{};
        bool replaced{};
        bool list_copies{};
        bool extent_probes{};
    } dispatch_kinds;
    const bool count_dispatch = VideoCommon::Nx2Profile::Fine() && onef_play == nullptr;
    const u64 marks_before = count_dispatch ? buffer_cache.nx2_written_marks : 0;
    const u64 mark_bytes_before = count_dispatch ? buffer_cache.nx2_written_mark_bytes : 0;
    SCOPE_EXIT {
        if (!count_dispatch) {
            return;
        }
        using Fine = VideoCommon::Nx2Profile::FineCounter;
        const auto add = [](Fine counter, u64 amount = 1) {
            VideoCommon::Nx2Profile::AddFine(counter, amount);
        };
        if (buffer_cache.nx2_written_marks != marks_before) {
            add(Fine::DispatchMarksBuffer);
            add(Fine::DispatchMarkBytes, buffer_cache.nx2_written_mark_bytes - mark_bytes_before);
        } else if (!pipeline->WrittenImages().empty()) {
            add(Fine::DispatchImagesOnly);
        } else {
            add(Fine::DispatchMarksNothing);
        }
        if (dispatch_kinds.indirect) {
            add(Fine::DispatchIndirect);
        }
        if (pipeline->MemFacts().has_texel_buffers) {
            add(Fine::DispatchTexelBuffers);
        }
        if (!dispatch_kinds.recorded) {
            add(Fine::DispatchSkipped);
        }
        if (dispatch_kinds.replaced) {
            add(Fine::DispatchReplaced);
        }
        if (dispatch_kinds.list_copies) {
            add(Fine::DispatchListCopies);
        }
        if (dispatch_kinds.extent_probes) {
            add(Fine::DispatchExtentProbes);
        }
    };
    std::optional<ByteAddressedShadow> byte_addressed_shadow;
    if (onef_play == nullptr) {
        mem_stage.BeginDispatch();
        buffer_cache.Nx2CollectListCopies(true);
    }
    SCOPE_EXIT {
        if (onef_play == nullptr) {
            buffer_cache.Nx2CollectListCopies(false);
        }
        if (byte_addressed_shadow) {
            texture_cache_runtime.RetireByteAddressedShadow(*byte_addressed_shadow);
        }
    };
    const auto indirect_address = kepler_compute->GetIndirectComputeAddress();
    // A played indirect dispatch (bit 4): E obtained the arguments' buffer in its fill, the
    // packet names it (held until Configure's rest, so taken here). R's view says whether the
    // dispatch is indirect; the packet must agree.
    MemStage::BufferRef onef_indirect{};
    if (onef_play != nullptr) {
        if (const MemStage::DispatchPacket* const held = mem_stage.HeldDispatch()) {
            onef_indirect = held->indirect;
        }
        if (indirect_address.has_value() != (onef_indirect.buffer != VK_NULL_HANDLE))
            [[unlikely]] {
            VideoCommon::OneFollower::Note(
                VideoCommon::OneFollower::Check::PlayedDispatchIndirect,
                indirect_address ? "a played indirect dispatch without its arguments' buffer"
                                 : "a played direct dispatch with an arguments' buffer");
            VideoCommon::OneFollower::StopPostDispatch("a played dispatch's indirect state");
            return;
        }
    }
    // The full-size 1080p pass is replaced only by FSR (native AA); DLAA through it is untested.
    // So is the 3/4-scale pass seen on Linux RADV (added with the FSR
    // port): DLSS keeps the passes it replaced before it, after the owner saw a softer image.
    constexpr u64 ThreeQuarterShader = 0xef542df87cacfe9eULL;
    const bool fsr_only_pass = pipeline->ShaderHash() == ComputePipeline::OcarinaFullSizeShader ||
                               pipeline->ShaderHash() == ThreeQuarterShader;
    // A pass learned below 720p (NoteLowSceneUpscale) counts like the known ones.
    const bool upscale_pass =
        pipeline->CanReplaceOcarinaUpscale() || pipeline->LearnedOcarinaUpscale();
    const bool can_replace = OcarinaTemporalEnabled() && upscale_pass &&
                             (!fsr_only_pass ||
                              oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr) &&
                             !indirect_address && !frame_dump.Capturing();
    // Until a dispatch overwrites the scene, each direct one binds only after the effects are
    // marked. The indirect path records its bind at once.
    const bool marking_effects = oot_effects_frame == render_trace_frame && OcarinaTemporalEnabled() &&
                                 !indirect_address && !frame_dump.Capturing();
    ComputePipeline::BindCommand deferred_bind;
    // SEAM = 2 (EXP123): the dispatch's buffer side is filled here, before Configure's texture
    // work; Configure plays it and RestAtEnd what is left, however Configure ends (true, false
    // or an exception: the native R16 shadow checks throw).
    // A played dispatch's packet is held already (TakeRingDispatch): E filled it.
    // FRITCHII_ASYNC_COMPUTE_SKIP (vk_async_compute_skip.h): a pipeline still building is decided
    // here, before the fill marks or uploads anything. Skipped, the dispatch returns as if the
    // guest had not run it; its QMD releases are the engine's (KeplerCompute, after this call)
    // and still happen. Run, it takes Configure's own path (no fill: the decision bound its
    // storage pointers already, or Configure binds them) and waits for the build as before.
    bool unbuilt_decided = false;
    if (onef_play == nullptr &&
        (AsyncComputeSkip::Mode() != 0 || AsyncComputeSkip::DelayMs() > 0)) [[unlikely]] {
        // FRITCHII_ASYNC_COMPUTE_SKIP_DELAY_MS (any mode): a dispatch that runs while its
        // program counts as unbuilt waits in Configure as a building pipeline's does (the
        // scheduler worker held until the delay ends, "Compute dispatch waited" logged).
        if (const auto deadline = async_compute_skip.DelayDeadline(*pipeline)) {
            pipeline->SetTestWaitUntil(*deadline);
        }
    }
    if (onef_play == nullptr && AsyncComputeSkip::Mode() != 0) [[unlikely]] {
        if (async_compute_skip.CountsUnbuilt(*pipeline)) {
            unbuilt_decided = true;
            constexpr u64 DepthReaderShader = 0x1fc7a0423116281fULL;
            const u64 hash = pipeline->ShaderHash();
            const bool special_pass =
                can_replace || upscale_pass ||
                hash == ComputePipeline::OcarinaFullSizeShader ||
                hash == ComputePipeline::Ocarina1440pShader || hash == ThreeQuarterShader ||
                hash == ComputePipeline::Ocarina1440pScaledShader ||
                hash == ComputePipeline::Ocarina1440pScaledFieldShader ||
                hash == ComputePipeline::OcarinaThreeQuarterShader || hash == DepthReaderShader ||
                (marking_effects && pipeline->WritesImages());
            if (SkipUnbuiltDispatch(*pipeline, indirect_address.has_value(), special_pass)) {
                pipeline->SetTestWaitUntil({}); // no Configure: nothing waits
                // EXP10: the dispatch writes nothing.
                if (VideoCommon::GpuSplit::RecordingDispatch()) {
                    VideoCommon::GpuSplit::Get().ConfirmDispatch();
                }
                return;
            }
        } else {
            async_compute_skip.NoteRan(*pipeline);
        }
    }
    if (onef_play == nullptr && !unbuilt_decided) {
        mem_stage.FillDispatch(buffer_cache, pipeline->MemFacts());
    }
    bool configured{};
    // FRITCHII_SPLIT_BIND_FIX: the tick Configure records the bind in (bind_again_after_split).
    const u64 configure_tick = scheduler.CurrentTick();
    // A played dispatch (one follower, R) cannot configure again after a split (its packet was
    // played at the rest; R must not touch E's buffer cache): with the fix on, its bind is
    // deferred and recorded with the dispatch, and the scheduler's tracker forgets the
    // descriptor buffer afterwards (the deferred bind binds it behind the tracker's back).
    const bool defer_split_bind = SplitBindFix() && onef_play != nullptr;
    const bool defer_bind = can_replace || marking_effects || defer_split_bind;
    {
        const MemStage::RestAtEnd rest{mem_stage};
        configured = pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                                         texture_cache, texture_cache_runtime,
                                         byte_addressed_shadow,
                                         defer_bind ? &deferred_bind : nullptr);
    }
    if (!configured) {
        if (onef_play != nullptr) [[unlikely]] {
            // R cannot throw back to E's engine (the lent path's "resources unavailable").
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::PlayedDispatch,
                                           "a played dispatch's Configure returned false");
            VideoCommon::OneFollower::StopPostDispatch("a played dispatch did not configure");
            return;
        }
        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute resources unavailable");
        return;
    }
    // FRITCHII_SPLIT_BIND_FIX, right before the dispatch's record: a submit since Configure
    // (staging, the grid copy, an upscaler, ...) put the record into a new command buffer with
    // nothing bound (the submit reset the scheduler's state). A deferred bind is recorded with
    // the dispatch and needs nothing. Otherwise the dispatch configures again, as draws do: its
    // bind lands in the new command buffer and the scheduler's state stays true (no compute bind
    // bypasses Scheduler::UpdateDescriptorBufferChunk). Counted either way. False: that Configure
    // failed (the dispatch is skipped).
    const auto bind_again_after_split = [&]() -> bool {
        if (scheduler.CurrentTick() == configure_tick) [[likely]] {
            return true;
        }
        NoteSplitBind(true, pipeline->ShaderHash());
        if (!SplitBindFix() || deferred_bind || onef_play != nullptr) {
            return true;
        }
        if (byte_addressed_shadow) {
            // The first bind's packed shadow: its view is in the submitted command buffer.
            texture_cache_runtime.RetireByteAddressedShadow(*byte_addressed_shadow);
        }
        const MemStage::RestAtEnd rest{mem_stage};
        return pipeline->Configure(*kepler_compute, *gpu_memory, scheduler, buffer_cache,
                                   texture_cache, texture_cache_runtime, byte_addressed_shadow,
                                   nullptr);
    };
    // FRITCHII_ASYNC_COMPUTE_SKIP: the written ranges' streaks (G2), for a dispatch recorded
    // here; a played one's came from E's fill (QueueWrites).
    const auto note_write_streaks = [&] {
        if (onef_play == nullptr && AsyncComputeSkip::Mode() != 0) [[unlikely]] {
            boost::container::small_vector<AsyncComputeSkip::Range, 8> writes;
            static_cast<void>(CollectComputeWrites(writes));
            async_compute_skip.NoteWrites(writes);
        }
    };
    // A played dispatch: E confirmed it (it writes no image and is direct).
    // A played dispatch writes images exactly when E saw it would (bit 2: E recorded them as
    // unbounded at its place, and the packed R16 path stays lent).
    if (onef_play != nullptr &&
        (byte_addressed_shadow.has_value() ||
         pipeline->WrittenImages().empty() == onef_dispatch_writes_images)) [[unlikely]] {
        VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::PlayedDispatchImage,
                                       byte_addressed_shadow
                                           ? "a played dispatch made the packed R16 shadow"
                                           : "a played dispatch's image writes against E's");
        VideoCommon::OneFollower::StopPostDispatch("a played dispatch's image writes");
    }
    if (onef_play == nullptr && VideoCommon::GpuSplit::RecordingDispatch()) {
        // EXP10: the buffer cache recorded the buffers the dispatch marks written; add its
        // images and the indirect arguments ObtainBuffer drops the GPU-written state of.
        auto& split = VideoCommon::GpuSplit::Get();
        for (const VideoCommon::ImageViewId id : pipeline->WrittenImages()) {
            const ImageView& view = texture_cache.GetImageView(id);
            const Image* const image = view.BackingImage();
            if (image == nullptr || view.image_id == VideoCommon::NULL_IMAGE_ID) {
                continue;
            }
            if (image->cpu_addr_end > image->cpu_addr) {
                split.RecordDispatchWrite(image->cpu_addr, image->cpu_addr_end - image->cpu_addr,
                                          VideoCommon::GpuSplit::DispatchWriteKind::Image);
            } else {
                split.RecordDispatchUnbounded(VideoCommon::GpuSplit::DispatchUnbounded::Image);
            }
        }
        if (const auto indirect = kepler_compute->GetIndirectComputeAddress()) {
            if (const auto device_addr = gpu_memory->GpuToCpuAddress(*indirect)) {
                split.RecordDispatchWrite(*device_addr, 12,
                                          VideoCommon::GpuSplit::DispatchWriteKind::Indirect);
            }
        }
        split.ConfirmDispatch();
    }
    SCOPE_EXIT {
        if (!frame_dump.Capturing() && !flash_watch.WatchesPasses()) {
            return;
        }
        boost::container::small_vector<ImageView*, 8> written;
        for (const VideoCommon::ImageViewId id : pipeline->WrittenImages()) {
            written.push_back(&texture_cache.GetImageView(id));
        }
        if (frame_dump.Capturing()) {
            frame_dump.CaptureDispatch(pipeline->ShaderHash(), written);
        }
        if (flash_watch.WatchesPasses()) {
            flash_watch.NoteDispatch(pipeline->ShaderHash(), written);
        }
    };
    if (onef_play == nullptr) {
        TracePendingCpuWrites(true, std::array{pipeline->ShaderHash()}); // reads the buffer cache
    }
    if (frame_dump.CapturingSampled() && onef_play == nullptr) {
        boost::container::small_vector<FrameDump::Sampled, 16> sampled;
        for (const auto& image : pipeline->SampledImages()) {
            sampled.push_back({5, image.tic, &texture_cache.GetImageView(image.id),
                               &texture_cache.GetSampler(image.sampler)});
        }
        frame_dump.NoteDispatchSampled(pipeline->ShaderHash(), sampled);
        FrameDumpBuffers(true, std::array{pipeline->ShaderHash()});
    }

    const auto& qmd{kepler_compute->launch_description};
    // Earlier writes must be visible to the dispatch, its indirect arguments included
    // (RecordDispatchBarrier, recorded at the start of the dispatch's own command), and what it
    // writes to the next render pass (NoteComputeWrites).
    if (frame_dump.CapturingSampled() && onef_play == nullptr) {
        std::optional<std::array<u32, 3>> indirect_grid;
        if (indirect_address) {
            indirect_grid = std::array{gpu_memory->Read<u32>(*indirect_address),
                                       gpu_memory->Read<u32>(*indirect_address + 4),
                                       gpu_memory->Read<u32>(*indirect_address + 8)};
        }
        frame_dump.NoteLaunch(pipeline->ShaderHash(),
                              {qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z},
                              {qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
                              indirect_address, indirect_grid);
    }
    if (indirect_address) {
        // DispatchIndirect
        static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
        const auto post_op = VideoCommon::ObtainBufferOperation::DiscardWrite;
        // A played one: E obtained it in its fill (the packet's `indirect`, its entries played
        // at the rest after Configure, where this Ref plays them).
        const MemStage::BufferRef indirect_ref =
            onef_play != nullptr ? onef_indirect : [&] {
                const auto [buffer, offset] = mem_stage.Ref(MemStage::Site::DispatchIndirect, [&] {
                    return buffer_cache.ObtainBuffer(*indirect_address, 12, sync_info, post_op);
                });
                return MemStage::BufferRef{buffer->Handle(), offset};
            }();
        // An unmapped argument address gives the null buffer, which has no VkBuffer where the
        // device has nullDescriptor: a dispatch would read GPU address 0. Nothing to run.
        if (indirect_ref.buffer == VK_NULL_HANDLE) [[unlikely]] {
            return;
        }
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.NoteComputeWrites();
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
        // FRITCHII_ONEF_OP_DISPATCH on (one follower, R): the grid words copied on the GPU right
        // before the dispatch, checked against the device limit once the submission is done
        // (OneFollowerCheckGrids). Lent ones too: the dry compare.
        // FRITCHII_GRID_AUDIT (boot, default 0): the same copy and check for every indirect
        // dispatch, one follower or not. The grid comes from GPU memory (list dispatches, GPU-sized
        // grids), so the host never sees it before vkCmdDispatchIndirect: a grid past the device
        // limit, or y/z above the 16 bits the console's QMD reads, is logged after the fact. A
        // diagnostic for the AMD device losses: costs a copy and two barriers per indirect dispatch.
        static const bool grid_audit = [] {
            const char* value = std::getenv("FRITCHII_GRID_AUDIT");
            return value != nullptr && value[0] == '1';
        }();
        std::optional<StagingBufferRef> grid_copy;
        const bool onef_grid = VideoCommon::OneFollower::On() &&
                               VideoCommon::OneFollower::OnRenderThread() &&
                               VideoCommon::OneFollower::DispatchPostBits() != 0;
        if (onef_grid || grid_audit) [[unlikely]] {
            OneFollowerCheckGrids();
            if (onef_grid_checks.size() < 256) {
                grid_copy = staging_pool.Request(16, MemoryUsage::Download, true);
            }
        }
        if (!bind_again_after_split()) [[unlikely]] {
            return;
        }
        scheduler.Record([this, pipeline, bind = std::move(deferred_bind),
                          indirect_buffer = indirect_ref.buffer,
                          indirect_offset = indirect_ref.offset,
                          grid_buffer = grid_copy ? grid_copy->buffer : VkBuffer{},
                          grid_offset = grid_copy ? grid_copy->offset : VkDeviceSize{}](
                             vk::CommandBuffer cmdbuf) {
            if (grid_buffer != VK_NULL_HANDLE) {
                static constexpr VkMemoryBarrier COPY_READS_WRITES{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                };
                static constexpr VkMemoryBarrier HOST_READS_COPY{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
                };
                cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READS_WRITES);
                const VkBufferCopy copy{
                    .srcOffset = indirect_offset,
                    .dstOffset = grid_offset,
                    .size = 3 * sizeof(u32),
                };
                cmdbuf.CopyBuffer(indirect_buffer, grid_buffer, copy);
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                       HOST_READS_COPY);
            }
            RecordDispatchBarrier(cmdbuf);
            // The deferred bind (FRITCHII_SPLIT_BIND_FIX), in the dispatch's command buffer.
            if (bind) {
                bind(cmdbuf);
            }
            if (!pipeline->IsBound()) {
                return;
            }
            scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, {}, [&] {
                cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
            });
        });
        if (defer_split_bind) {
            // FRITCHII_SPLIT_BIND_FIX: the played dispatch's bind, recorded with it, bound a
            // descriptor buffer the scheduler did not track.
            scheduler.ForgetDescriptorBuffer();
        }
        if (grid_copy) {
            onef_grid_checks.push_back({*grid_copy, scheduler.CurrentTick(),
                                        pipeline->ShaderHash(), onef_play != nullptr,
                                        !onef_grid});
        }
        // A played dispatch's probes and list page copies are E's (a lent tail after the post).
        if (onef_play == nullptr) {
            dispatch_kinds.extent_probes = mem_stage.Ref(MemStage::Site::DispatchAfter, [&] {
                return buffer_cache.RecordComputeExtentProbes();
            }) != 0;
        }
        compute_since_count_barrier = true;
        if (onef_play == nullptr) {
            dispatch_kinds.list_copies = SubmitNx2ListCopies(pipeline->ShaderHash());
        }
        dispatch_kinds.recorded = true;
        dispatch_kinds.indirect = true;
        note_write_streaks();
        return;
    }
    const std::array<u32, 3> dim{qmd.grid_dim_x, qmd.grid_dim_y, qmd.grid_dim_z};
    const std::array<u32, 3> max_dim{device.GetMaxComputeWorkGroupCount()};
    if (dim[0] > max_dim[0] || dim[1] > max_dim[1] || dim[2] > max_dim[2]) {
        // Never recorded (a grid past maxComputeWorkGroupCount is undefined in Vulkan); counted,
        // the first 8 and then every 256th logged.
        static std::atomic<u32> skipped{};
        const u32 n = skipped.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 8 || n % 256 == 0) {
            LOG_WARNING(Render_Vulkan,
                        "Dispatch grid over the device limit, skipped: {}x{}x{} of {:016x}, limit "
                        "{}x{}x{} ({} so far)",
                        dim[0], dim[1], dim[2], pipeline->ShaderHash(), max_dim[0], max_dim[1],
                        max_dim[2], n);
        }
        return;
    }
    note_write_streaks();
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.NoteComputeWrites();
    // The upscaler's evaluation or the effects mark may be recorded next, before the dispatch:
    // the barrier goes first.
    const bool barrier_first = can_replace || marking_effects;
    if (barrier_first) {
        scheduler.Record([](vk::CommandBuffer cmdbuf) { RecordDispatchBarrier(cmdbuf); });
    }
    if (marking_effects) {
        const auto written = pipeline->WrittenImages();
        if (std::ranges::any_of(written, [this](VideoCommon::ImageViewId id) {
                return id && texture_cache.GetImageView(id).gpu_addr == oot_effects_address;
            })) {
            // Fritchii's free guard: oot_effects_scene is a raw view handle taken at the fog
            // draw. A scene image deleted since (unmap, join, relocation) and made again at the
            // same address passes the address test, but the saved view's image is destroyed
            // once the GPU passes the delete: after a submit in between, this mark would sample
            // freed memory. Its view generation moved then: no mark this frame.
            if (texture_cache.ImageViewGeneration(oot_effects_view) == oot_effects_generation) {
                oot_live_dlss->MarkEffects(oot_effects_scene, render_trace_frame);
            } else if (oot_effects_stale++ < 8) {
                LOG_WARNING(Render_Vulkan,
                            "Free guard: OoT effects mark skipped, the scene view at {:#x} was "
                            "destroyed since the fog draw ({} so far)",
                            oot_effects_address, oot_effects_stale);
            }
            oot_effects_frame.reset();
        }
    }
    // Below 720p the game's upscale pass is none of the known ones: found here, replaced from
    // its next dispatch on. Without a render height: one relaxed load.
    if (const auto render_height = Settings::AppliedOcarinaRenderHeight();
        render_height && *render_height < 720 && !can_replace && !indirect_address)
        [[unlikely]] {
        NoteLowSceneUpscale(*pipeline, *render_height);
    }
    const auto* const replacement =
        can_replace && !byte_addressed_shadow && TryOcarinaUpscale(*pipeline, true)
            ? oot_live_dlss.get()
            : nullptr;
    if (!replacement) {
        scheduler.TraceComputePipeline(pipeline->ShaderHash());
    } else {
        // Once per pass: which of the game's upscale passes the upscaler took, and with what.
        static std::array<u64, 8> logged{};
        const u64 hash = pipeline->ShaderHash();
        if (std::ranges::find(logged, hash) == logged.end()) {
            if (const auto slot = std::ranges::find(logged, 0ULL); slot != logged.end()) {
                *slot = hash;
            }
            LOG_INFO(Render_Vulkan, "Ocarina temporal: {} replaces the game's pass {:016x}",
                     oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr ? "FSR" : "DLSS",
                     hash);
        }
    }
    if (!bind_again_after_split()) [[unlikely]] {
        return;
    }
    scheduler.Record([this, pipeline, dim, byte_addressed_shadow, bind = std::move(deferred_bind),
                      barrier_recorded = barrier_first, replacement](vk::CommandBuffer cmdbuf) {
        if (!barrier_recorded) {
            RecordDispatchBarrier(cmdbuf);
        }
        if (replacement && !replacement->EvaluationFailed()) {
            return;
        }
        if (replacement) {
            LOG_WARNING(Render_Vulkan,
                        "Ocarina temporal: {} failed, the game's own upscale runs for the rest of "
                        "this session",
                        VideoCommon::OcarinaBackendName(*oot_backend));
        }
        if (bind) {
            bind(cmdbuf);
        }
        if (!pipeline->IsBound()) {
            return;
        }
        if (byte_addressed_shadow) {
            TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, false);
        }
        scheduler.TraceGpuWork(cmdbuf, pipeline->ShaderHash(), 0, dim,
                               [&] { cmdbuf.Dispatch(dim[0], dim[1], dim[2]); });
        if (byte_addressed_shadow) {
            TextureCacheRuntime::CopyByteAddressedShadow(cmdbuf, *byte_addressed_shadow, true);
        }
    });
    if (defer_split_bind) {
        // FRITCHII_SPLIT_BIND_FIX: the played dispatch's bind, recorded with it, bound a
        // descriptor buffer the scheduler did not track.
        scheduler.ForgetDescriptorBuffer();
    }
    // A played dispatch's probes and list page copies are E's (a lent tail after the post).
    if (onef_play == nullptr) {
        dispatch_kinds.extent_probes = mem_stage.Ref(MemStage::Site::DispatchAfter, [&] {
            return buffer_cache.RecordComputeExtentProbes();
        }) != 0;
    }
    if (can_replace || marking_effects) {
        // The deferred native binding may follow DLSS's command buffer bindings.
        if (onef_play != nullptr && !onef_draw->OnView()) [[unlikely]] {
            // Rule 2: off the view the state tracker names E's engine's flags. No replacement
            // ran here (posted dispatches are no upscaler pass and write no image), so the
            // invalidation is only conservative: skipped, and posting stops.
            VideoCommon::OneFollower::Note(
                VideoCommon::OneFollower::Check::PlayedDispatchNoView,
                "a played dispatch's state invalidation with R off its view");
            VideoCommon::OneFollower::StopPostDispatch("a played dispatch off the view");
        } else {
            scheduler.InvalidateState();
            if (replacement) {
                scheduler.DisturbVertexBindings(); // an upscaler's own recording
            }
        }
    }
    compute_since_count_barrier = true;
    if (onef_play == nullptr) {
        dispatch_kinds.list_copies = SubmitNx2ListCopies(pipeline->ShaderHash());
    }
    dispatch_kinds.recorded = true;
    dispatch_kinds.replaced = replacement != nullptr;

    if (!can_replace) {
        TryOcarinaUpscale(*pipeline, false);
    }

    // Log compute dispatch
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string params = fmt::format("groupCountX={}, groupCountY={}, groupCountZ={}",
            dim[0], dim[1], dim[2]);
        GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
            "vkCmdDispatch", params, VK_SUCCESS);
    }
}

void RasterizerVulkan::ResetCounter(VideoCommon::QueryType type) {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        namespace OneF = VideoCommon::OneFollower;
        if (type == VideoCommon::QueryType::ZPassPixelCount64) {
            onef_samples_started = false; // the mirror (fable_b 16.2)
            // FRITCHII_ESPLIT: E1's copy (on E2 the queue's hook cleared it when E1 queued this).
            if (!VideoCommon::ESplit::OnE2()) {
                onef_mirror_queued = false;
            }
        }
        OneF::SetQueryKind(OneF::QueryKind::Reset);
        if (OneF::QueryOps()) {
            // fable_b 11.1 item 1(b): a render op, E not parked (nothing comes back).
            struct Op {
                RasterizerVulkan* self;
                VideoCommon::QueryType type;
            } const op{this, type};
            OneF::SetLendSite("ResetCounter (posted)");
            // FRITCHII_ONEF_CARRY_OPS (fable_b 22.3 step 2): the samples streamer's reset reads
            // nothing of R's view, so it may ride in E's next packet (the others read the
            // transform feedback registers).
            OneF::PushOp(
                OneF::LentOp::Query,
                [](void* raw) {
                    Op a;
                    std::memcpy(&a, raw, sizeof(a));
                    a.self->ResetCounter(a.type);
                },
                &op, sizeof(op), false, 0,
                type == VideoCommon::QueryType::ZPassPixelCount64);
            return;
        }
    }
    ONEF_LEND(Query, ResetCounter(type))
    switch (type) {
    case VideoCommon::QueryType::ZPassPixelCount64:
    case VideoCommon::QueryType::StreamingByteCount:
    case VideoCommon::QueryType::StreamingPrimitivesSucceeded:
    case VideoCommon::QueryType::VtgPrimitivesOut:
        query_cache.CounterReset(type);
        return;
    default:
        LOG_DEBUG(Render_Vulkan, "Unimplemented counter reset={}", type);
        return;
    }
}

namespace {
/// FRITCHII_QUERY_LEND_DRY=1 (live, default 0; helper DE round 5, a dry timer that changes
/// nothing). kaksample2: E parked ~23 ms/s in Query lends. Per lent Query: why it was lent (the
/// rule that kept it from the posted paths), E's parked time, and whether its report page is one
/// the front's DMA reads snapshot (FrontReadPagesAny: a posted report there would be read stale
/// by the front unless it waited for R; must be 0 for a kind to be posted as is).
struct QueryLendDry {
    enum Why : u8 {
        Other,
        FencePayloadRule,     ///< fence payload: SEM_E's path off (not synced, EXP33, SEM_E off)
        FencePayloadNotClean, ///< fence payload: no device address or the fence not clean
        FenceCounterSwitch,   ///< fence counter report: POST_FENCE_REPORT off or EXP33
        FenceCounterAtOnce,   ///< fence counter report of a type that lands at once
        FenceCounterNotClean, ///< fence counter report: no device address or not clean
        PayloadRelease,       ///< plain payload release lent
        PlainReport,          ///< plain report (no streamer under SEM_E, or QUERY_OPS off)
        BufferPage,           ///< plain report on a registered buffer page
        Count,
    };
    static constexpr std::array<const char*, Count> names{
        "other",
        "fence payload: SEM_E path off",
        "fence payload: not clean",
        "fence counter: switch off",
        "fence counter: lands at once",
        "fence counter: not clean",
        "payload release",
        "plain report",
        "report on buffer page"};
    static bool On() {
        return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_query_lend_dry,
                                                "FRITCHII_QUERY_LEND_DRY") != 0;
    }
    static u64 Now() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }
    void Note(u8 why, bool front_page, u64 parked_ns, u64 now) {
        const size_t i = why < Count ? why : Other;
        ++count[i];
        parked[i] += parked_ns;
        front[i] += front_page ? 1 : 0;
        if (since == 0) {
            since = now;
            return;
        }
        if (now - since < 10'000'000'000ULL) {
            return;
        }
        const double seconds = static_cast<double>(now - since) / 1e9;
        std::string text;
        u64 all = 0;
        u64 all_ns = 0;
        for (size_t k = 0; k < Count; ++k) {
            all += count[k];
            all_ns += parked[k];
            if (count[k] == 0) {
                continue;
            }
            text += fmt::format("; {} {:.0f}/s {:.1f} ms/s (front read page {:.0f}/s)", names[k],
                                static_cast<double>(count[k]) / seconds,
                                static_cast<double>(parked[k]) / 1e6 / seconds,
                                static_cast<double>(front[k]) / seconds);
        }
        LOG_INFO(Render_Vulkan,
                 "Query lends (FRITCHII_QUERY_LEND_DRY=1) {:.0f} s: {:.0f}/s, E parked {:.1f} ms/s "
                 "by why lent{}",
                 seconds, static_cast<double>(all) / seconds,
                 static_cast<double>(all_ns) / 1e6 / seconds, text);
        count = {};
        parked = {};
        front = {};
        since = now;
    }
    u64 since{};
    std::array<u64, Count> count{};
    std::array<u64, Count> parked{};
    std::array<u64, Count> front{};
};
QueryLendDry g_query_lend_timer;    ///< E only
thread_local u8 t_query_lend_why{}; ///< E: why this Query is lent (set where its kind is)
} // Anonymous namespace

void RasterizerVulkan::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                             VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        namespace OneF = VideoCommon::OneFollower;
        t_query_lend_why = QueryLendDry::Other;
        // FRITCHII_ONEF_DRY_RELEASE (fable_b 19.5 step 3): E passes this release.
        if (OneF::DryReleaseOn()) [[unlikely]] {
            if (const std::optional<DAddr> device = gpu_memory->GpuToCpuAddress(gpu_addr)) {
                OneF::DryNotePass(
                    *device,
                    type != VideoCommon::QueryType::Payload ? OneF::DryKind::CounterReport
                    : True(flags & VideoCommon::QueryPropertiesFlags::IsAFence)
                        ? OneF::DryKind::PayloadFence
                        : OneF::DryKind::PayloadRelease);
            }
        }
        // FRITCHII_ONEF_POST_FENCE: what this report owes the next fence. Samples reports made
        // with the mirror started and payload releases sync into buffers E obtains for a posted
        // fence (16.2); any other kind is owed (the fence is lent). Every report's page waits for
        // the posted fence that may land it (13.0).
        if (OneF::PostFenceOn()) {
            const std::optional<DAddr> report_device = gpu_memory->GpuToCpuAddress(gpu_addr);
            const bool fence_report =
                True(flags & VideoCommon::QueryPropertiesFlags::IsAFence);
            // FRITCHII_ONEF_SEM_E: a payload E writes lands at E's place, not at a fence; a
            // fence-flagged one syncs its buffers at its own (posted) fence.
            const bool payload_on_e = type == VideoCommon::QueryType::Payload && OneF::SemOnE() &&
                                      !VideoCore::Nx2Debug::AccuratePart(1);
            if (report_device && !payload_on_e) {
                OneF::NoteReportPage(*report_device);
            }
            if (type == VideoCommon::QueryType::ZPassPixelCount64) {
                if (onef_samples_started && report_device &&
                    (!fence_report || OneF::PostFenceReportOn())) {
                    onef_sync_samples.push_back(*report_device);
                }
            } else if (type == VideoCommon::QueryType::Payload) {
                if (report_device && (!fence_report || payload_on_e)) {
                    onef_sync_payloads.push_back(*report_device);
                }
            } else {
                onef_sync_owed = true;
            }
        } else if (type != VideoCommon::QueryType::ZPassPixelCount64 || onef_samples_started) {
            onef_sync_owed = true;
        }
        if (True(flags & VideoCommon::QueryPropertiesFlags::IsAFence)) {
            // FRITCHII_ONEF_SEM_E (fable_b 19.4, 19.6): a fence-flagged payload report whose
            // operation writes at once (synced fence behaviour, EXP33 off) is a posted fence
            // whose callback, the write, E runs at its place. Otherwise lent.
            const bool synced = Settings::IsGPUFenceBehaviorDefault()
                                    ? !VideoCore::Nx2Debug::AccuratePart(1)
                                    : !Settings::IsGPUFenceBehaviorBalanced() &&
                                          !Settings::IsGPUFenceBehaviorAccurate();
            if (type == VideoCommon::QueryType::Payload && synced &&
                VideoCore::Nx2Debug::Experiment(33) == 0 && OneF::SemOnE()) {
                if (const std::optional<DAddr> device = gpu_memory->GpuToCpuAddress(gpu_addr);
                    device && OneFollowerPostKind(16) && OneFollowerFenceClean()) {
                    const OneFollowerPayloadNote note{
                        *device, gpu_addr, payload, subreport,
                        True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)};
                    OneF::CountQueryKind(OneF::QueryKind::PayloadFenceOnE);
                    if (!OneF::SignalOnR()) { // else R writes it (OneFollowerPostFence)
                        OneF::NoteEngineSignal("a fence-flagged payload write");
                    }
                    OneFollowerPostFence(5, 0, nullptr, &note);
                    return;
                }
            }
            // FRITCHII_ONEF_POST_FENCE_REPORT: a clean fence's counter report (one with a
            // streamer: its value is R's) is posted as the rasterizer's fences are: E's buffer
            // half, sync buffers (its own value listed above) and gather here, the report and
            // its fence on R, its page waiting for the op (13.0). EXP33's kept reports and
            // payload ones (SEM_E's, or lent: one home) stay as they were.
            if (type != VideoCommon::QueryType::Payload && OneF::PostFenceReportOn() &&
                VideoCore::Nx2Debug::Experiment(33) == 0 && query_cache.ReportLandsLater(type)) {
                if (const std::optional<DAddr> device = gpu_memory->GpuToCpuAddress(gpu_addr);
                    device && OneFollowerPostKind(32) && OneFollowerFenceClean()) {
                    if (onef_report_pages == nullptr) [[unlikely]] {
                        onef_report_pages = std::make_unique<VideoCommon::PageFilter>();
                    }
                    onef_report_pages->Mark(*device, 16); // as a posted plain report (item 2)
                    const OneFollowerFenceReport report{
                        gpu_addr, *device, type, flags, payload, subreport,
                        static_cast<u8>(type == VideoCommon::QueryType::ZPassPixelCount64
                                            ? (onef_samples_started ? 2 : 1)
                                            : 0)};
                    OneF::CountQueryKind(OneF::QueryKind::FenceReportPosted);
                    OneFollowerPostFence(7, 0, nullptr, nullptr, &report);
                    return;
                }
            }
            OneFollowerFenceLent(); // its fence (R's, E parked) syncs and lands all before it
            OneF::SetQueryKind(OneF::QueryKind::FenceReport); // lent (fable_b item 4)
            if (QueryLendDry::On()) [[unlikely]] {
                const bool exp33 = VideoCore::Nx2Debug::Experiment(33) != 0;
                if (type == VideoCommon::QueryType::Payload) {
                    t_query_lend_why = synced && !exp33 && OneF::SemOnE()
                                           ? QueryLendDry::FencePayloadNotClean
                                           : QueryLendDry::FencePayloadRule;
                } else if (!OneF::PostFenceReportOn() || exp33) {
                    t_query_lend_why = QueryLendDry::FenceCounterSwitch;
                } else if (!query_cache.ReportLandsLater(type)) {
                    t_query_lend_why = QueryLendDry::FenceCounterAtOnce;
                } else {
                    t_query_lend_why = QueryLendDry::FenceCounterNotClean;
                }
            }
        } else if (type == VideoCommon::QueryType::Payload) {
            // FRITCHII_ONEF_SEM_E (fable_b 19.4): a payload release CounterReport writes at once
            // (not Accurate) is written here, at E's place (after its page's landings), and
            // latches its waiters here; R gets only the streamer's pending sync (the buffers'
            // side, at the next fence: 16.2's payload rule, listed above).
            const std::optional<DAddr> device =
                OneF::SemOnE() && !VideoCore::Nx2Debug::AccuratePart(1)
                    ? gpu_memory->GpuToCpuAddress(gpu_addr)
                    : std::nullopt;
            if (device && OneF::SignalOnR()) {
                // FRITCHII_ONEF_SIGNAL_ON_R (AMD tester, test 4 §5c): the same write at R's
                // played place. E goes on (a posted op, not a lend); the guest sees the release
                // only once R recorded everything before it, as with one GPU thread. The front's
                // E-position waits wait for the op (lands_in_memory), E's reads of the page too
                // (NoteReportLandsAtLastPush). R plays in order: no earlier landing to wait for.
                const bool has_timestamp =
                    True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout);
                if (VideoCore::g_read_downloads.load(std::memory_order_acquire) != 0)
                    [[unlikely]] {
                    // As OneFollowerPostFence does for its landing pages (fable_b 16.4).
                    buffer_cache.FinishReadDownloads(*device & ~u64{0xfff}, u64{1} << 12);
                }
                struct Op {
                    RasterizerVulkan* self;
                    DAddr device;
                    GPUVAddr gpu_addr;
                    u32 payload;
                    u32 subreport;
                    bool has_timestamp;
                } const op{this, *device, gpu_addr, payload, subreport, has_timestamp};
                OneF::CountSignalOnR();
                OneF::SetQueryKind(OneF::QueryKind::PayloadReleaseOnR);
                static_assert(sizeof(Op) <= OneF::MaxOpPayloadBytes,
                              "an op over the payload limit");
                OneF::SetLendSite("payload release (written on R, SIGNAL_ON_R)");
                OneF::NoteNextGuestWriteRange(*device, 16);
                OneF::PushOp(
                    OneF::LentOp::Query,
                    [](void* raw) {
                        Op a;
                        std::memcpy(&a, raw, sizeof(a));
                        a.self->query_cache.WritePayloadNow(a.device, a.gpu_addr, a.payload,
                                                            a.has_timestamp, false);
                        a.self->query_cache.NotePayloadWritten(a.device, a.has_timestamp,
                                                               a.payload, a.subreport);
                    },
                    &op, sizeof(op), true);
                OneF::NoteReportLandsAtLastPush(*device);
                return;
            }
            if (device) {
                const bool has_timestamp =
                    True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout);
                OneF::BeforeEngineRead(*device, 16);
                OneF::NoteEngineSignal("a payload release");
                query_cache.WritePayloadNow(*device, gpu_addr, payload, has_timestamp, false);
                struct Op {
                    RasterizerVulkan* self;
                    DAddr device;
                    u32 payload;
                    u32 subreport;
                    bool has_timestamp;
                } const op{this, *device, payload, subreport, has_timestamp};
                OneF::SetQueryKind(OneF::QueryKind::PayloadReleaseOnE);
                static_assert(sizeof(Op) <= OneF::MaxOpPayloadBytes,
                              "an op over the payload limit");
                OneF::SetLendSite("payload release (written on E)");
                OneF::PushOp(
                    OneF::LentOp::Query,
                    [](void* raw) {
                        Op a;
                        std::memcpy(&a, raw, sizeof(a));
                        a.self->query_cache.NotePayloadWritten(a.device, a.has_timestamp,
                                                               a.payload, a.subreport);
                    },
                    &op, sizeof(op));
                return;
            }
            OneF::SetQueryKind(OneF::QueryKind::PayloadRelease); // lent (fable_b item 4)
            t_query_lend_why = QueryLendDry::PayloadRelease;
        } else if (OneF::SemOnE() && !query_cache.ReportLandsLater(type)) {
            // FRITCHII_ONEF_SEM_E: a type without a streamer becomes a payload write inside
            // CounterReport: lent, so every payload write is E's or lent (one home).
            OneF::SetQueryKind(OneF::QueryKind::PlainReport);
            t_query_lend_why = QueryLendDry::PlainReport;
        } else if (!OneF::QueryOps()) {
            OneF::SetQueryKind(OneF::QueryKind::PlainReport);
            t_query_lend_why = QueryLendDry::PlainReport;
        } else {
            // fable_b 11.1 item 1(c): a plain report is a render op, translated here at E's
            // stream place. Guard: a target the buffer cache has registered stays lent (E's
            // uploads would not see R's later write).
            const std::optional<DAddr> device = gpu_memory->GpuToCpuAddress(gpu_addr);
            bool registered = false;
            if (device) {
                std::scoped_lock lock{buffer_cache.mutex};
                registered = buffer_cache.IsRegionRegistered(*device, 16);
            }
            // QUERY_OPS=2: the guard is about a report that lands inside CounterReport (a type
            // without a streamer becomes a payload write there; lent, E resumes after it landed).
            // A counter report only queues: it lands at a fence or flush, which is lent, in both
            // forms, and its buffer side (SyncValues) runs there. Nothing E reads changes in
            // CounterReport itself (E's own report page filter, below, stands for the query
            // cache's until R plays it).
            const bool buffer_page_op =
                registered && OneF::QueryOpsOnBufferPages() && query_cache.ReportLandsLater(type);
            if (registered && !buffer_page_op) {
                // A lend reason, not a must-be-0: OoT's reports often land in buffers.
                OneF::SetQueryKind(OneF::QueryKind::ReportOnBufferPage);
                t_query_lend_why = QueryLendDry::BufferPage;
            } else {
                OneF::SetQueryKind(buffer_page_op ? OneF::QueryKind::BufferPageOp
                                                  : OneF::QueryKind::PlainReport);
                if (device) {
                    // Item 2: E's invalidations see this page as a query's before R plays the op
                    // (R marks the query cache's own filter only then).
                    if (onef_report_pages == nullptr) [[unlikely]] {
                        onef_report_pages = std::make_unique<VideoCommon::PageFilter>();
                        onef_report_pages->AlsoMark(&onef_union_pages); // cut 5
                    }
                    onef_report_pages->Mark(*device, 16);
                }
                struct Op {
                    RasterizerVulkan* self;
                    GPUVAddr gpu_addr;
                    std::optional<DAddr> device;
                    VideoCommon::QueryType type;
                    VideoCommon::QueryPropertiesFlags flags;
                    u32 payload;
                    u32 subreport;
                    u8 mirror; ///< 0 off, 1 E's bit clear, 2 set (FRITCHII_ONEF_SYNC_MIRROR)
                } const op{this,  gpu_addr, device, type, flags, payload, subreport,
                           static_cast<u8>(
                               (OneF::SyncMirrorOn() || OneF::PostFenceOn()) &&
                                       type == VideoCommon::QueryType::ZPassPixelCount64
                                   ? (onef_samples_started ? 2 : 1)
                                   : 0)};
                OneF::SetLendSite("Query (posted)");
                if (device) {
                    OneF::NoteNextGuestWriteRange(*device, 16);
                }
                OneF::PushOp(
                    OneF::LentOp::Query,
                    [](void* raw) {
                        Op a;
                        std::memcpy(&a, raw, sizeof(a));
                        if (a.mirror != 0) {
                            const bool render_bit =
                                a.self->query_cache.ReportMakesPendingSync(a.type);
                            VideoCommon::OneFollower::CountSyncMirror(a.mirror == 2, render_bit);
                            if (a.mirror == 2 && !render_bit && a.device) {
                                a.self->OneFollowerNoteSuperset(*a.device);
                            }
                        }
                        a.self->query_cache.CounterReportAt(a.gpu_addr, a.device, a.type, a.flags,
                                                            a.payload, a.subreport);
                    },
                    &op, sizeof(op), true, 0,
                    type == VideoCommon::QueryType::ZPassPixelCount64); // CARRY_OPS, as the reset
                return;
            }
        }
    }
    if (VideoCommon::OneFollower::ShouldLend() &&
        (VideoCommon::OneFollower::SyncMirrorOn() || VideoCommon::OneFollower::PostFenceOn()) &&
        type == VideoCommon::QueryType::ZPassPixelCount64) [[unlikely]] {
        onef_mirror_lent_valid = true; // read by R in the lent call (E parked)
        onef_mirror_lent_bit = onef_samples_started;
    }
    if (VideoCommon::OneFollower::ShouldLend() && QueryLendDry::On()) [[unlikely]] {
        // FRITCHII_QUERY_LEND_DRY: the same lend as ONEF_LEND below, timed and classified.
        const auto device = gpu_memory->GpuToCpuAddress(gpu_addr);
        const bool front_page = device && VideoCommon::OneFollower::FrontReadPagesAny(*device, 16);
        const u8 why = t_query_lend_why;
        const u64 start = QueryLendDry::Now();
        VideoCommon::OneFollower::SetLendSite(__func__);
        auto onef_call = [&] { Query(gpu_addr, type, flags, payload, subreport); };
        VideoCommon::OneFollower::Lend(VideoCommon::OneFollower::LentOp::Query, onef_call);
        const u64 now = QueryLendDry::Now();
        g_query_lend_timer.Note(why, front_page, now - start, now);
        return;
    }
    ONEF_LEND(Query, Query(gpu_addr, type, flags, payload, subreport))
    if (onef_mirror_lent_valid && VideoCommon::OneFollower::RenderInLentCall()) [[unlikely]] {
        onef_mirror_lent_valid = false;
        const bool render_bit = query_cache.ReportMakesPendingSync(type);
        VideoCommon::OneFollower::CountSyncMirror(onef_mirror_lent_bit, render_bit);
        if (onef_mirror_lent_bit && !render_bit) {
            if (const auto device = gpu_memory->GpuToCpuAddress(gpu_addr)) {
                OneFollowerNoteSuperset(*device);
            }
        }
    }
    query_cache.CounterReport(gpu_addr, type, flags, payload, subreport);
}

void RasterizerVulkan::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                 u32 size) {
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void Vulkan::RasterizerVulkan::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerVulkan::FlushAll() {}

bool RasterizerVulkan::SubmitNx2ListCopies(u64 writer) {
    ONEF_LEND_RESULT(Copy, SubmitNx2ListCopies(writer))
    // Submitted at once: ReadGpuWrittenCommands waits on these copies, so they can't sit out
    // the longer DeferFlush interval.
    if (!mem_stage.Ref(MemStage::Site::DispatchAfter,
                       [&] { return buffer_cache.Nx2CopyListPages(mem_clock.Now(), writer); })) {
        return false;
    }
    // EDEN_NX2_EXP27 & 8 (the original's DISPATCH_BATCH): the front reads few of the lists
    // (Ocarina: ~150 dispatches a frame, a submission each), so the copies go when it waits for
    // one (BufferCacheRuntime::WaitSubmittedTick asks the back), when the back goes idle, or
    // with the next submission.
    if ((VideoCore::Nx2Debug::Experiment(27) & 8) != 0 && scheduler.DeferFlushUntilWanted()) {
        return true;
    }
    if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FenceFlush) || !scheduler.DeferFlush()) {
        scheduler.Flush();
    }
    return true;
}

VideoCore::RasterizerInterface::GpuWrittenList RasterizerVulkan::ReadGpuWrittenCommandList(
    DAddr addr, std::span<u8> out, bool past_sync) {
    // No cache lock: the front only reads the copies, under their own lock.
    switch (buffer_cache.Nx2ReadListCopy(addr, out, past_sync)) {
    case BufferCache::Nx2ListRead::Copied:
        return GpuWrittenList::Copied;
    case BufferCache::Nx2ListRead::Synced:
        return GpuWrittenList::Synced;
    case BufferCache::Nx2ListRead::NeedsSync:
        return GpuWrittenList::NeedsSync;
    default:
        return GpuWrittenList::Unknown;
    }
}

void RasterizerVulkan::NoteGpuWrittenCommandListSynced(DAddr page) {
    buffer_cache.Nx2NoteListSynced(page);
}

void RasterizerVulkan::FlushGpuWrittenCommandListPages(std::span<const DAddr> pages) {
    ONEF_LEND(ReadBack, FlushGpuWrittenCommandListPages(pages))
    gpu.InvalidateGPUCache();
    std::scoped_lock lock{buffer_cache.mutex};
    MemStage::FillPlay seam{mem_stage, MemStage::Site::ListPages};
    buffer_cache.Nx2FlushListPages(pages);
}

bool RasterizerVulkan::OneFollowerFlushClean(DAddr addr, u64 size,
                                             VideoCommon::CacheType which) {
    using VideoCommon::CacheType;
    if (!VideoCommon::OneFollower::FlushOnEngine()) {
        return false;
    }
    // The buffer half: E marks the GPU-written pages when it fills the writes (before the
    // packets that record them), so a page never marked has nothing to download, at R's place
    // or anywhere. The gather before a download only orders the guest's writes ahead of it.
    if (True(which & CacheType::BufferCache) && Tegra::Nx2MayBeGpuWritten(addr, size)) {
        return false;
    }
    // The query half: R marks its filter at a lent report (E parked), E's own filter takes the
    // reports it posted.
    if (True(which & CacheType::QueryCache) &&
        (query_cache.page_filter.Any(addr, size) ||
         (onef_report_pages != nullptr && onef_report_pages->Any(addr, size)))) {
        return false;
    }
    // The texture half: R registers images (and marks the filter) when it plays the draws that
    // render to them, so a miss counts only with R caught up (nothing of E's unplayed, R not
    // marking now).
    if (True(which & CacheType::TextureCache) &&
        (VideoCommon::OneFollower::Behind() != 0 || texture_cache.page_filter.Any(addr, size))) {
        return false;
    }
    return true;
}

void RasterizerVulkan::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    // FRITCHII_ONEF_FLUSH_E (fable_b 17.3 (a)): a flush E tests clean does nothing at R's place:
    // not lent at all.
    if (VideoCommon::OneFollower::ShouldLend() && OneFollowerFlushClean(addr, size, which))
        [[unlikely]] {
        return;
    }
    ONEF_LEND(ReadBack, FlushRegion(addr, size, which))
    // One follower: R outside a lent call, an unlent draw's image upload reading game memory
    // (TextureCache::UploadImageContents -> MemoryManager::ReadBlock). The gather and the buffer
    // half below are E's cache, E running ahead (the hangs of 12 Nov: R took the buffer cache's
    // lock here, and its download waited on E). Pages the GPU never wrote have nothing to
    // download, and the gather only orders the guest's writes before a download: both are
    // skipped then, exactly. A GPU-written page is a must-be-0 that stops step 4; the image
    // takes the range again so its next use (lent) uploads it right.
    if (VideoCommon::OneFollower::OnRenderThread() &&
        !VideoCommon::OneFollower::RenderInLentCall()) [[unlikely]] {
        using VideoCommon::CacheType;
        const bool needs_engine =
            (True(which & CacheType::BufferCache) && Tegra::Nx2MayBeGpuWritten(addr, size)) ||
            (True(which & CacheType::TextureCache) && texture_cache.page_filter.Any(addr, size));
        if (needs_engine) {
            thread_local std::string note;
            note = fmt::format("FlushRegion {:#x}+{:#x} (cache bits {:#x}) in an unlent draw", addr,
                               size, static_cast<u32>(which));
            VideoCommon::OneFollower::Note(
                VideoCommon::OneFollower::Check::RenderNeedsEngineFlush, note.c_str());
            VideoCommon::OneFollower::DisableUnlend(
                "an unlent draw's image upload needed E's buffer flush");
            if (onef_dispatch_play != nullptr) {
                // The same for a posted dispatch's sampled image (FRITCHII_ONEF_OP_DISPATCH).
                VideoCommon::OneFollower::StopPostDispatch(
                    "a played dispatch's image upload needed E's buffer flush");
            }
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.WriteMemory(addr, size);
        }
        if (True(which & CacheType::QueryCache)) {
            query_cache.FlushRegion(addr, size); // R's own
        }
        return;
    }
    if (Common::WriteRing::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        mem_stage.NoteGather(MemStage::Gather::FlushRegion, static_cast<u32>(which));
    }
    // FRITCHII_MEMSTAGE_SHADOW: read flushes inside a draw (between its fill and its record).
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        using VideoCommon::CacheType;
        VideoCommon::MemStageShadow::BackFlushRegion(True(which & CacheType::TextureCache),
                                                     True(which & CacheType::BufferCache),
                                                     True(which & CacheType::QueryCache));
    }
    if (True(which & (VideoCommon::CacheType::TextureCache | VideoCommon::CacheType::BufferCache))) {
        // CPU writes into GPU-written memory are only collected when they happen. Apply them
        // first, or the download writes the stale GPU copy over them: the Yoshi demo writes a
        // native program over memory a compute pass had bound as a written texel buffer, and its
        // title menu or background read as zeros once the shader cache's read flushed it.
        gpu.InvalidateGPUCache();
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        // THREAD = 2: on GPUMem when the back runs a served entry (a read flush inside a served
        // draw's Configure).
        OnStage(static_cast<u8>(VideoCommon::MemoryStageLink::Served::FlushRegion), [&] {
            std::scoped_lock lock{buffer_cache.mutex};
            CheckServedJoins(addr, size);
            MemStage::FillPlay seam{mem_stage, MemStage::Site::Download};
            buffer_cache.DownloadMemory(addr, size);
        });
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.FlushRegion(addr, size);
    }
}

void RasterizerVulkan::FlushRegionForRead(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    ONEF_LEND(ReadBack, FlushRegionForRead(addr, size))
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        VideoCommon::MemStageShadow::BackFlushRegion(true, true, true);
    }
    // EDEN_NX2_EXP49: as FlushRegion, but the buffer cache doesn't wait for the host GPU here
    // (Scheduler::Finish, with its lock held): the reading thread does, in FinishRead.
    gpu.InvalidateGPUCache();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    OnStage(static_cast<u8>(VideoCommon::MemoryStageLink::Served::FlushRegionForRead), [&] {
        std::scoped_lock lock{buffer_cache.mutex};
        CheckServedJoins(addr, size);
        MemStage::FillPlay seam{mem_stage, MemStage::Site::DownloadForRead};
        buffer_cache.DownloadMemoryForRead(addr, size);
    });
    query_cache.FlushRegion(addr, size);
}

void RasterizerVulkan::FinishRead(DAddr addr, u64 size) {
    if (onef_fence_posted && VideoCommon::OneFollower::OnRenderThread() &&
        !VideoCommon::OneFollower::RenderInLentCall()) [[unlikely]] {
        // fable_b 16.4: R inside a posted fence (its report landings) never waits for the buffer
        // lock: E landed the downloads over these reports before posting. One still listed (a
        // guest read since) is landed only with the lock free (R then waits for the host GPU
        // alone): must be 0 either way.
        if (VideoCore::g_read_downloads.load(std::memory_order_acquire) == 0) {
            return;
        }
        if (!buffer_cache.mutex.try_lock()) {
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::RenderReadLanding,
                                           "buffer lock busy, landing skipped");
            return;
        }
        const bool landed = buffer_cache.FinishReadDownloads(addr, size).has_value();
        buffer_cache.mutex.unlock();
        if (landed) {
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::RenderReadLanding,
                                           "a read download over a report R writes");
        }
        return;
    }
    buffer_cache.FinishReadDownloads(addr, size);
}

bool RasterizerVulkan::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    // FRITCHII_ONEF_FLUSH_E: clean as FlushRegion tests it: nothing to flush.
    if (VideoCommon::OneFollower::ShouldLend() && OneFollowerFlushClean(addr, size, which))
        [[unlikely]] {
        return false;
    }
    ONEF_LEND_RESULT(ReadBack, MustFlushRegion(addr, size, which))
    if ((True(which & VideoCommon::CacheType::BufferCache)) && MayBeGpuModified(addr, size)) {
        bool modified = false;
        OnStage(static_cast<u8>(VideoCommon::MemoryStageLink::Served::MustFlush), [&] {
            std::scoped_lock lock{buffer_cache.mutex};
            modified = buffer_cache.IsRegionGpuModified(addr, size);
        });
        if (modified) {
            return true;
        }
    }
    if (!VideoCore::Nx2Debug::AccuratePart(4)) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

bool RasterizerVulkan::MayBeGpuModified(DAddr addr, u64 size) {
    // Every range the buffer cache records as GPU-modified also marks its pages GPU-written
    // (bits that never clear), so pages without the bit need no look under its mutex. The split's
    // front asks this before each macro DMA read; the back holds the mutex through each draw.
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) || size == 0) {
        return true;
    }
    const DAddr last = addr + size - 1;
    for (DAddr page = addr >> Tegra::Nx2WrittenPageBits; page <= (last >> Tegra::Nx2WrittenPageBits);
         ++page) {
        if (Tegra::Nx2IsGpuWritten(page << Tegra::Nx2WrittenPageBits)) {
            return true;
        }
    }
    return false;
}

bool RasterizerVulkan::HasPendingQueries(DAddr addr, u64 size) {
    return query_cache.IsRegionGpuModified(addr, size);
}

bool RasterizerVulkan::HasCachedReports(DAddr addr, u64 size) {
    // EXP94 (live; Fable answer_front 1): a range no query was ever registered on has none, which
    // the query cache's page filter tells without its mutex (the front asks for every macro DMA read).
    if (VideoCore::Nx2Debug::Experiment(94) != 0 && !query_cache.page_filter.Any(addr, size)) {
        return false;
    }
    return query_cache.HasCachedReports(addr, size);
}

namespace {
/// Guest reads and writes of cached memory (GetFlushArea, OnCPUWrite), logged every 5 s: the
/// lock-free answers, the locked ones, and the locked reads that must flush.
struct GuestAccessCounters {
    std::atomic<u64> read_fast{};
    std::atomic<u64> read_locked{};
    std::atomic<u64> read_flush{};
    std::atomic<u64> write_locked{};
    /// Why a read took the locked path (fable52_answer_microcuts 3.4): an image registered there
    /// now, one only ever (EXP84's count is 0), and pages the GPU ever wrote.
    std::atomic<u64> locked_image_now{};
    std::atomic<u64> locked_image_ever{};
    std::atomic<u64> locked_gpu_written{};
    /// EXP86 (fable52_answer_gpuwritten 2.6): reads on GPU-written pages the buffer cache's page
    /// mirrors answered (fast), and why they didn't: counter odd or moved, a GPU-modified byte,
    /// an armed trap without a preflush mark, or the mirror broken / out of range.
    std::atomic<u64> mirror_fast{};
    std::atomic<u64> mirror_busy{};
    std::atomic<u64> mirror_modified{};
    std::atomic<u64> mirror_armed{};
    std::atomic<u64> mirror_broken{};
    /// EXP86=2: locked reads compared with the mirrors. "Mirror clean, cache not" must stay 0 (it
    /// also counts reads the texture cache answered, which the fast path would have missed).
    std::atomic<u64> verify_checked{};
    std::atomic<u64> verify_mirror_clean_cache_not{};
    std::atomic<u64> verify_texture_answered{}; ///< of those, the texture cache answered
    std::atomic<u64> verify_cache_clean_mirror_not{};
    std::atomic<u64> verify_busy{};
    /// EXP100 (fable52_answer_imagereads 5): reads with an image on their pages (EXP100 != 0)
    /// the texture cache's count of GPU-modified images answered (fast), and why the others
    /// stayed locked: a counted image on the 64 KiB page, a page a sparse image ever covered, the
    /// counter odd or moved (or the count broken), the buffer side not clean, or excluded (EXP36,
    /// EXP81, EXP43 = 4).
    std::atomic<u64> image_fast{};
    std::atomic<u64> image_counted{};
    std::atomic<u64> image_sparse{};
    std::atomic<u64> image_busy{};
    std::atomic<u64> image_buffer{};
    std::atomic<u64> image_excluded{};
    /// EXP100=2: locked reads compared with the count, those it would have answered, and the
    /// mismatches: the count said no GPU-modified image, the locked look found one (whole
    /// session, must be 0).
    std::atomic<u64> image_verify_checked{};
    std::atomic<u64> image_verify_would_fast{};
    std::atomic<u64> image_verify_mismatch{};
    std::atomic<s64> last_log{};
};
GuestAccessCounters g_guest_access;

void CountMirrorAnswer(VideoCommon::MirrorAnswer answer) {
    auto& c = g_guest_access;
    switch (answer) {
    case VideoCommon::MirrorAnswer::Clean:
        c.mirror_fast.fetch_add(1, std::memory_order_relaxed);
        break;
    case VideoCommon::MirrorAnswer::Busy:
        c.mirror_busy.fetch_add(1, std::memory_order_relaxed);
        break;
    case VideoCommon::MirrorAnswer::Modified:
        c.mirror_modified.fetch_add(1, std::memory_order_relaxed);
        break;
    case VideoCommon::MirrorAnswer::ArmedUnmarked:
        c.mirror_armed.fetch_add(1, std::memory_order_relaxed);
        break;
    case VideoCommon::MirrorAnswer::Broken:
        c.mirror_broken.fetch_add(1, std::memory_order_relaxed);
        break;
    }
}

/// EXP86=2: the mirrors' answer against the locked path's (cache_clean: preemptive and the read
/// traps left alone).
void CountMirrorVerify(VideoCommon::MirrorAnswer answer, bool cache_clean,
                       bool texture_answered = false) {
    auto& c = g_guest_access;
    c.verify_checked.fetch_add(1, std::memory_order_relaxed);
    if (answer == VideoCommon::MirrorAnswer::Busy) {
        c.verify_busy.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const bool mirror_clean = answer == VideoCommon::MirrorAnswer::Clean;
    if (mirror_clean && !cache_clean) {
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(Render_Vulkan, "EXP86 verify: the page mirror said clean, the locked path "
                                     "did not");
        }
        c.verify_mirror_clean_cache_not.fetch_add(1, std::memory_order_relaxed);
        if (texture_answered) {
            c.verify_texture_answered.fetch_add(1, std::memory_order_relaxed);
        }
    } else if (!mirror_clean && cache_clean) {
        c.verify_cache_clean_mirror_not.fetch_add(1, std::memory_order_relaxed);
    }
}

void CountGuestAccess(std::atomic<u64>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
    auto& c = g_guest_access;
    const u64 total = c.read_fast.load(std::memory_order_relaxed) +
                      c.read_locked.load(std::memory_order_relaxed);
    if ((total & 4095) != 0) {
        return;
    }
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    s64 last = c.last_log.load(std::memory_order_relaxed);
    if (now - last < 5000 || !c.last_log.compare_exchange_strong(last, now)) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Guest access of cached memory (5 s): reads {} lock-free, {} locked ({} must "
             "flush, {} handed to the back); writes {} locked, {} collected",
             c.read_fast.exchange(0), c.read_locked.exchange(0), c.read_flush.exchange(0),
             Tegra::g_nx2_guest_reads_handed.exchange(0, std::memory_order_relaxed),
             c.write_locked.exchange(0),
             Tegra::g_nx2_guest_writes_collected.exchange(0, std::memory_order_relaxed));
    LOG_INFO(Render_Vulkan,
             "Guest reads locked because (5 s, EXP84={}): an image registered there now {}, an "
             "image only ever {}, GPU-written pages {}; EXP100={}: answered by the image filter "
             "{}, stayed locked by reason: counted image on the 64 KiB page {}, sparse {}, "
             "counter busy {}, buffer side {}, excluded {}; verify {} checked, {} would be fast, "
             "{} mismatches (whole session, must be 0)",
             VideoCore::Nx2Debug::Experiment(84), c.locked_image_now.exchange(0),
             c.locked_image_ever.exchange(0), c.locked_gpu_written.exchange(0),
             VideoCore::Nx2Debug::Experiment(100), c.image_fast.exchange(0),
             c.image_counted.exchange(0), c.image_sparse.exchange(0), c.image_busy.exchange(0),
             c.image_buffer.exchange(0), c.image_excluded.exchange(0),
             c.image_verify_checked.exchange(0), c.image_verify_would_fast.exchange(0),
             c.image_verify_mismatch.load());
    LOG_INFO(Render_Vulkan,
             "GPU-written page mirror (5 s, EXP86={}): {} fast, {} counter busy, {} modified, {} "
             "armed without a mark, {} broken; verify {} checked, {} mirror clean but cache not "
             "(whole session, must be 0; {} of them answered by the texture cache), {} cache "
             "clean but mirror not, {} busy",
             VideoCore::Nx2Debug::Experiment(86), c.mirror_fast.exchange(0),
             c.mirror_busy.exchange(0), c.mirror_modified.exchange(0), c.mirror_armed.exchange(0),
             c.mirror_broken.exchange(0), c.verify_checked.exchange(0),
             c.verify_mirror_clean_cache_not.load(), c.verify_texture_answered.load(),
             c.verify_cache_clean_mirror_not.exchange(0), c.verify_busy.exchange(0));
}
} // Anonymous namespace

bool RasterizerVulkan::WaitFenceDownload(u64 sequence) {
    return buffer_cache.WaitFenceDownload(sequence, std::chrono::milliseconds{20});
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {
    ONEF_LEND_RESULT(ReadBack, GetFlushArea(addr, size))
    // A guest read of cached memory asked under both caches' mutexes, which the split's back holds
    // through each draw: 12% of the guest cores' busy time at Castle Town went to waiting for
    // them. Pages no image was ever registered on (page_filter) and that no GPU write ever marked
    // (MayBeGpuModified, bits that never clear) need no look: neither cache finds anything there,
    // and the locked path would answer this same preemptive page-aligned area.
    const DAddr page_begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
    // EDEN_NX2_EXP84 (live, 0 = off; fable52_answer_microcuts 3.4): "no image here" from the count
    // of images registered now (and the pages sparse images ever covered) instead of page_filter,
    // whose bits never clear: the game's targets move every frame and leave them behind. An image
    // the GPU wrote and the read must see is registered when the read comes (the fence ordering
    // that makes page_filter's answer valid makes the count's valid too).
    const bool by_count = VideoCore::Nx2Debug::Experiment(84) != 0;
    const bool image_now = texture_cache.image_count_filter.Any(addr, size) ||
                           texture_cache.sparse_page_filter.Any(addr, size);
    const bool image_ever = texture_cache.page_filter.Any(addr, size);
    const bool gpu_written = MayBeGpuModified(page_begin, page_end - page_begin);
    // FRITCHII_ASYNC_COMPUTE_SKIP: guest reads of pages the GPU wrote or an image was on never
    // see a skipped dispatch's output (sticky; must-be-0 count when one does).
    if ((image_ever || gpu_written) && AsyncComputeSkip::Mode() != 0) [[unlikely]] {
        async_compute_skip.NoteGuestRead(addr, size);
    }
    // EDEN_NX2_EXP36=1 turns this off (live), for the washed-out scene hunt.
    if (VideoCore::Nx2Debug::Experiment(36) == 0 && !(by_count ? image_now : image_ever) &&
        !gpu_written) {
        CountGuestAccess(g_guest_access.read_fast);
        return VideoCore::RasterizerDownloadArea{
            .start_address = page_begin,
            .end_address = page_end,
            .preemtive = true,
        };
    }
    // EDEN_NX2_EXP86 (live, 0 = off; fable52_answer_gpuwritten): pages the GPU ever wrote, asked
    // without the locks. The buffer cache mirrors per page whether gpu_modified_ranges holds a
    // byte there now, the tracker's Preflushable mark and an armed read trap, under its mutex,
    // with a counter that is odd while a hold changes them. Clean, and marked or not armed: the
    // locked path would answer this same preemptive area and change nothing. With EXP81 the clean
    // answer sets the read's pages in preflush_read_pages, as the locked path does
    // (fable52_answer_fences2 2). 2 = verify on the locked path, no fast answers.
    const int mirror_mode = VideoCore::Nx2Debug::Experiment(86);
    const bool mirror_eligible = mirror_mode != 0 && gpu_written && size != 0 &&
                                 !(by_count ? image_now : image_ever) &&
                                 VideoCore::Nx2Debug::Experiment(36) == 0;
    if (mirror_eligible && mirror_mode == 1) {
        const VideoCommon::MirrorAnswer answer = buffer_cache.MirrorCleanForRead(addr, size);
        CountMirrorAnswer(answer);
        if (answer == VideoCommon::MirrorAnswer::Clean) {
            CountGuestAccess(g_guest_access.read_fast);
            return VideoCore::RasterizerDownloadArea{
                .start_address = page_begin,
                .end_address = page_end,
                .preemtive = true,
            };
        }
    }
    const bool verify_mirror = mirror_eligible && mirror_mode == 2;
    // EDEN_NX2_EXP100 (live, 0 = off; fable52_answer_imagereads, Fable's F1): pages with an image
    // on them, asked without the locks. The texture cache's locked look acts only on images that
    // are GpuModified; gpu_image_filter counts, per 64 KiB page, every registered non-sparse image
    // that was GPU-modified since it was registered, changed only under the texture cache's mutex
    // with image_seq odd while a hold changes it. Nothing counted on the pages, no sparse image
    // ever there, and the buffer side clean as today (no GPU write ever marked the pages, or
    // EXP86's mirrors say clean): the locked path would answer this same preemptive area and
    // change nothing. Obeys EXP36; EXP81's stamps and EXP43 = 4's tombstone check need the lock.
    // 2 = verify on the locked path, no fast answers.
    const int image_mode = VideoCore::Nx2Debug::Experiment(100);
    const bool image_block = by_count ? image_now : image_ever; // what sends it to the lock today
    bool image_eligible = false;
    if (image_mode != 0 && image_block && size != 0) {
        if (VideoCore::Nx2Debug::Experiment(36) != 0 ||
            (VideoCore::Nx2Debug::Experiment(81) != 0 && gpu_written && mirror_mode != 1) ||
            VideoCore::Nx2Debug::Experiment(43) == 4) { // tombstone check needs the lock
            g_guest_access.image_excluded.fetch_add(1, std::memory_order_relaxed);
        } else if (texture_cache.sparse_page_filter.Any(addr, size)) {
            g_guest_access.image_sparse.fetch_add(1, std::memory_order_relaxed);
        } else {
            image_eligible = true;
        }
    }
    // The buffer side, as today: no GPU write ever marked the pages, or EXP86's mirrors say clean.
    const auto image_buffer_clean = [&] {
        if (!gpu_written) {
            return true;
        }
        return mirror_mode == 1 && buffer_cache.MirrorCleanForRead(addr, size) ==
                                       VideoCommon::MirrorAnswer::Clean;
    };
    if (image_eligible && image_mode == 1) {
        const u64 seq = texture_cache.image_seq.Load();                  // 1: the counter
        if ((seq & 1) != 0 || texture_cache.image_seq.Broken()) {
            g_guest_access.image_busy.fetch_add(1, std::memory_order_relaxed);
        } else if (texture_cache.gpu_image_filter.Any(addr, size)) {     // 2: the count
            g_guest_access.image_counted.fetch_add(1, std::memory_order_relaxed);
        } else if (!image_buffer_clean()) {                              // 3: the buffer side
            g_guest_access.image_buffer.fetch_add(1, std::memory_order_relaxed);
        } else if (texture_cache.image_seq.Load() != seq ||              // 4: the counter again
                   texture_cache.image_seq.Broken()) {
            g_guest_access.image_busy.fetch_add(1, std::memory_order_relaxed);
        } else {
            g_guest_access.image_fast.fetch_add(1, std::memory_order_relaxed);
            CountGuestAccess(g_guest_access.read_fast);
            return VideoCore::RasterizerDownloadArea{
                .start_address = page_begin,
                .end_address = page_end,
                .preemtive = true,
            };
        }
    }
    const bool verify_image = image_eligible && image_mode == 2;
    const bool verify_image_buffer_clean = verify_image && image_buffer_clean();
    if (image_now) {
        g_guest_access.locked_image_now.fetch_add(1, std::memory_order_relaxed);
    } else if (image_ever) {
        g_guest_access.locked_image_ever.fetch_add(1, std::memory_order_relaxed);
    }
    if (gpu_written) {
        g_guest_access.locked_gpu_written.fetch_add(1, std::memory_order_relaxed);
    }
    // EDEN_NX2_EXP60=1: the back holds both mutexes through bursts of draws and re-takes them right
    // after each release, so a guest read waiting for them slept for milliseconds (Fable answer
    // 11). Spin a little; if they stay taken, ask for a flush instead: the back serves sync
    // requests before every draw (TickWork), and FlushRegion of a clean page does nothing.
    if (VideoCore::Nx2Debug::Experiment(60) != 0) {
        constexpr int Tries = 256;
        bool texture_locked = false;
        bool buffer_locked = false;
        for (int attempt = 0; attempt < Tries && !(texture_locked && buffer_locked); ++attempt) {
            texture_locked = texture_locked || texture_cache.mutex.try_lock();
            buffer_locked = buffer_locked || buffer_cache.mutex.try_lock();
            if (!(texture_locked && buffer_locked)) {
                std::this_thread::yield();
            }
        }
        if (!(texture_locked && buffer_locked)) {
            if (texture_locked) {
                texture_cache.mutex.unlock();
            }
            if (buffer_locked) {
                buffer_cache.mutex.unlock();
            }
            Tegra::g_nx2_guest_reads_handed.fetch_add(1, std::memory_order_relaxed);
            return VideoCore::RasterizerDownloadArea{
                .start_address = page_begin,
                .end_address = page_end,
                .preemtive = false,
            };
        }
        // Both held: the locked path takes them again recursively.
        const auto area =
            verify_image    ? GetFlushAreaLockedVerifyImage(addr, size, verify_image_buffer_clean)
            : verify_mirror ? GetFlushAreaLockedVerifyMirror(addr, size)
                            : GetFlushAreaLocked(addr, size);
        buffer_cache.mutex.unlock();
        texture_cache.mutex.unlock();
        CountGuestAccess(g_guest_access.read_locked);
        if (!area.preemtive) {
            g_guest_access.read_flush.fetch_add(1, std::memory_order_relaxed);
        }
        return area;
    }
    const auto area =
        verify_image    ? GetFlushAreaLockedVerifyImage(addr, size, verify_image_buffer_clean)
        : verify_mirror ? GetFlushAreaLockedVerifyMirror(addr, size)
                        : GetFlushAreaLocked(addr, size);
    CountGuestAccess(g_guest_access.read_locked);
    if (!area.preemtive) {
        g_guest_access.read_flush.fetch_add(1, std::memory_order_relaxed);
    }
    return area;
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushAreaLockedVerifyMirror(DAddr addr,
                                                                                   u64 size) {
    // GetFlushAreaLocked, step for step. The mirrors are asked before anything changes: under
    // the buffer cache's mutex they answer for exactly the state GetFlushArea then sees.
    {
        std::scoped_lock lock{texture_cache.mutex};
        auto area = texture_cache.GetFlushArea(addr, size);
        if (area) {
            // The fast path would have answered without the texture cache (its image test).
            CountMirrorVerify(buffer_cache.MirrorCleanForRead(addr, size), false, true);
            return *area;
        }
    }
    if (buffer_cache.FlushesOnCpuRead()) {
        std::scoped_lock lock{buffer_cache.mutex};
        const VideoCommon::MirrorAnswer answer = buffer_cache.MirrorCleanForRead(addr, size);
        const bool leaves_traps = buffer_cache.ReadLeavesTrapsAlone(addr, size);
        auto area = buffer_cache.GetFlushArea(addr, size);
        CountMirrorVerify(answer, (!area || area->preemtive) && leaves_traps);
        if (area) {
            return *area;
        }
    } else {
        // Nothing asks the buffer cache: preemptive, nothing changed.
        CountMirrorVerify(buffer_cache.MirrorCleanForRead(addr, size), true);
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushAreaLockedVerifyImage(
    DAddr addr, u64 size, bool buffer_clean) {
    // GetFlushAreaLocked, step for step (EXP100 = 2). Under the texture cache's mutex no other
    // thread is inside a hold, so the count is exact for this instant: "count says none, the
    // locked look found one" is a true mismatch (a set site was missed). The read is still
    // answered by the locked path.
    auto& c = g_guest_access;
    {
        std::scoped_lock lock{texture_cache.mutex};
        // Odd here only if this thread's own outer hold is changing the count; broken: the fast
        // path never answers. Neither is compared.
        const bool comparable = (texture_cache.image_seq.Load() & 1) == 0 &&
                                !texture_cache.image_seq.Broken();
        const bool filter_clean = !texture_cache.gpu_image_filter.Any(addr, size);
        auto area = texture_cache.GetFlushArea(addr, size);
        if (!comparable) {
            c.image_busy.fetch_add(1, std::memory_order_relaxed);
        } else {
            c.image_verify_checked.fetch_add(1, std::memory_order_relaxed);
            if (filter_clean && area) {
                const u64 mismatches = c.image_verify_mismatch.fetch_add(1) + 1;
                if (mismatches <= 16) {
                    LOG_ERROR(Render_Vulkan,
                              "EXP100 verify mismatch {}: read [{:#x} +{:#x}) has no counted "
                              "image, the locked look found a GPU-modified one",
                              mismatches, addr, size);
                    texture_cache.LogGpuModifiedImagesInRegion(addr, size);
                }
            } else if (!filter_clean) {
                c.image_counted.fetch_add(1, std::memory_order_relaxed);
            } else if (!buffer_clean) {
                c.image_buffer.fetch_add(1, std::memory_order_relaxed);
            } else {
                c.image_verify_would_fast.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (area) {
            return *area;
        }
    }
    if (buffer_cache.FlushesOnCpuRead()) {
        std::scoped_lock lock{buffer_cache.mutex};
        auto area = buffer_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushAreaLocked(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        auto area = texture_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    if (buffer_cache.FlushesOnCpuRead()) {
        std::scoped_lock lock{buffer_cache.mutex};
        auto area = buffer_cache.GetFlushArea(addr, size);
        if (area) {
            return *area;
        }
    }
    VideoCore::RasterizerDownloadArea new_area{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
    return new_area;
}

void RasterizerVulkan::PrepareRegionWrite(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0 || False(which & VideoCommon::CacheType::BufferCache)) {
        return;
    }
    ONEF_LEND(Invalidation, PrepareRegionWrite(addr, size, which))
    std::scoped_lock lock{buffer_cache.mutex};
    // MemoryManager landed the read downloads here just before (FinishRead) and writes guest
    // memory next: landing a newer one would wait for the host GPU with the lock held.
    buffer_cache.ClearReplacedDownload(addr, size);
}

void RasterizerVulkan::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    VideoCommon::ComputeSkipHook::NoteWrite(addr, size); // FRITCHII_ASYNC_COMPUTE_SKIP
    ONEF_LEND(Invalidation, InvalidateRegion(addr, size, which))
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::BufferCache))) {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
        buffer_cache.WriteMemory(addr, size);
        mem_stage.NoteCpuMark(addr, size); // step 7's verify (re-review N4)
    }
    if ((True(which & VideoCommon::CacheType::QueryCache))) {
        query_cache.InvalidateRegion(addr, size);
    }
    if ((True(which & VideoCommon::CacheType::ShaderCache))) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

bool RasterizerVulkan::OneFollowerInvalidationOnEngine(
    std::span<const std::pair<DAddr, std::size_t>> sequences) {
    namespace OneF = VideoCommon::OneFollower;
    if (!OneF::InvalOnEngine()) {
        return false;
    }
    // fable_b 11.1 item 2: a range on no page an image, a shader or a query was ever on (nor a
    // report op E pushed and R has not played) invalidates nothing of R's: E runs the rest.
    // FRITCHII_ONEF_UNION_FILTER (fable_b 14.4 cut 5): the union of the four first; the four
    // (which say which kind it is) only when it has a bit there.
    const bool union_first =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_union_filter,
                                         "FRITCHII_ONEF_UNION_FILTER") != 0;
    const auto hit = [&]() -> std::optional<OneF::InvalKind> {
        if (union_first) {
            bool any = false;
            for (const auto& [addr, size] : sequences) {
                if (onef_union_pages.Any(addr, size)) {
                    any = true;
                    break;
                }
            }
            if (!any) {
                return std::nullopt;
            }
        }
        for (const auto& [addr, size] : sequences) {
            if (texture_cache.page_filter.Any(addr, size)) {
                return OneF::InvalKind::Texture;
            }
            if (pipeline_cache.page_filter.Any(addr, size)) {
                return OneF::InvalKind::Shader;
            }
            if (query_cache.page_filter.Any(addr, size) ||
                (onef_report_pages != nullptr && onef_report_pages->Any(addr, size))) {
                return OneF::InvalKind::Query;
            }
        }
        return std::nullopt;
    };
    if (const auto first = hit()) {
        OneF::CountInvalidation(*first);
        OneF::CountLentGatherRanges(sequences.size());
        return false;
    }
    // The Dekker pair with R's image registration (it marks the texture filter, fences, then
    // reads these stamps before its upload, WrittenAheadOfRender): stamp, fence, look again.
    // Either this sees R's mark (lent, in order) or R sees the stamp (its image uploads again).
    for (const auto& [addr, size] : sequences) {
        OneF::StampPageWrites(addr, size);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (hit()) {
        OneF::CountInvalidation(OneF::InvalKind::Recheck);
        return false;
    }
    OneF::CountInvalidation(OneF::InvalKind::OnEngine);
    return true;
}

void RasterizerVulkan::InnerInvalidation(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    // FRITCHII_ONEF_INVAL_E: on E with no range on R's pages, E runs the buffer half here and
    // skips R's halves (they would find nothing); anything else is lent as before.
    const bool onef_engine_half = VideoCommon::OneFollower::ShouldLend() &&
                                  OneFollowerInvalidationOnEngine(sequences);
    if (!onef_engine_half) {
        ONEF_LEND(Invalidation, InnerInvalidation(sequences))
    }
    if (VideoCommon::MemoryStageLink::BackServed() && !sequences.empty()) [[unlikely]] {
        // FRITCHII_MEMSTAGE_THREAD = 2 (plan 5.2): collected ranges on the back outside a run.
        // Must be 0; served on GPUMem all the same.
        VideoCommon::MemoryStageLink::Note(VideoCommon::MemoryStageLink::Check::BackInvalidation,
                                           "InnerInvalidation");
        InnerInvalidationServed(sequences);
        return;
    }
    // EXP91 (stage 0, counting) and EXP92 (uniform keeps, tier 1): every cached write range, in
    // the keeps' ring, before any early return.
    buffer_cache.NoteCachedWrites(sequences);
    // Step 7's verify: marks the cache takes inline, on any thread (re-review N4).
    for (const auto& [addr, size] : sequences) {
        mem_stage.NoteCpuMark(addr, size);
    }
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        for (const auto& [addr, size] : sequences) {
            VideoCommon::Nx2FastDrawCount::NotePoolWrite(addr, size);
        }
    }
    if (Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        return;
    }
    // The GPU's own cached writes (constant buffer data, every draw): most land where no image,
    // query or shader was ever registered, which page_filter tells without the caches' locks.
    const auto any = [&](const VideoCommon::PageFilter& filter) {
        for (const auto& [addr, size] : sequences) {
            if (filter.Any(addr, size)) {
                return true;
            }
        }
        return false;
    };
    if (!onef_engine_half && any(texture_cache.page_filter)) {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : sequences) {
            texture_cache.WriteMemory(addr, size);
        }
    }
    namespace CbPath = VideoCommon::Nx2CbPath;
    const int cb_path = CbPath::Mode();
    CbPath::MaybeLog(cb_path);
    // FlushCaching's ranges are the GPU's own engines' writes (EXP108).
    if ((cb_path & CbPath::SkipBit) == 0) {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
        // As OnCacheInvalidations: out of the GPU-written ranges first, so applying one range
        // doesn't download the GPU's bytes over a later range on the same page.
        buffer_cache.ClaimCpuWrites(sequences);
        for (const auto& [addr, size] : sequences) {
            buffer_cache.WriteMemory(addr, size, true, true);
        }
    } else {
        // EXP88 & 1: ranges WriteMemory would change nothing in
        // (BufferCache::WriteReachesBuffers) keep only its three marks, without the lock. The
        // drawing thread runs FlushCaching, so the vertex bindings asked are its own.
        using Reach = BufferCache::WriteReach;
        auto& counters = CbPath::g_counters;
        const bool verify = (cb_path & CbPath::VerifyBit) != 0;
        boost::container::small_vector<std::pair<std::pair<DAddr, std::size_t>, bool>, 32> locked;
        for (const auto& range : sequences) {
            const Reach reach = buffer_cache.WriteReachesBuffers(range.first, range.second);
            if (reach == Reach::None) {
                CbPath::Count(counters.skipped);
                if (verify) {
                    locked.emplace_back(range, true);
                } else {
                    buffer_cache.NoteWriteUnlocked(range.first, range.second);
                }
                continue;
            }
            CbPath::Count(reach == Reach::EverSynchronized  ? counters.kept_synced
                          : reach == Reach::MayBeGpuWritten ? counters.kept_gpu
                                                            : counters.kept_vertex);
            locked.emplace_back(range, false);
        }
        CbPath::Count(counters.ranges, sequences.size());
        if (locked.empty()) {
            CbPath::Count(counters.lock_skips);
        } else {
            std::scoped_lock lock{buffer_cache.mutex};
            MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
            // As above: claim every range applied below first. Skipped ranges
            // (Reach::None) were never GPU-written, so the claim has nothing to take there.
            boost::container::small_vector<std::pair<DAddr, std::size_t>, 32> applied;
            for (const auto& [range, skipped] : locked) {
                applied.push_back(range);
            }
            buffer_cache.ClaimCpuWrites(applied);
            for (const auto& [range, skipped] : locked) {
                if (skipped) {
                    // Verify: the skip had to change nothing; WriteMemory runs anyway.
                    CbPath::Count(counters.verify_skip_checked);
                    if (!buffer_cache.WriteChangesNothing(range.first, range.second)) {
                        CbPath::Count(counters.verify_skip_mismatch);
                        static std::atomic<u32> logged{};
                        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                            LOG_ERROR(Render_Vulkan,
                                      "EXP88 verify: a skipped range {:#x}+{:#x} would change "
                                      "the buffer cache",
                                      range.first, range.second);
                        }
                    }
                } else {
                    CbPath::Count(counters.locked);
                }
                buffer_cache.WriteMemory(range.first, range.second, true, true);
            }
        }
    }
    if (onef_engine_half) {
        return; // R's halves: no range on their pages (OneFollowerInvalidationOnEngine)
    }
    if (any(query_cache.page_filter)) {
        for (const auto& [addr, size] : sequences) {
            query_cache.InvalidateRegion(addr, size);
        }
    }
    if (any(pipeline_cache.page_filter)) {
        pipeline_cache.InvalidateRegions(sequences);
    }
}

bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {
    VideoCommon::ComputeSkipHook::NoteWrite(addr, size); // FRITCHII_ASYNC_COMPUTE_SKIP
    ONEF_LEND_RESULT(Invalidation, OnCPUWrite(addr, size))
    DEBUG_ASSERT(addr != 0 || size != 0);
    CountGuestAccess(g_guest_access.write_locked);
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        VideoCommon::Nx2FastDrawCount::NotePoolWrite(addr, size);
    }
    {
        // EXP109: the pages the write untracks
        // change their host protection after the lock is dropped: drawing waited for those
        // system calls (~1 us each, thousands a second at OoT's water). The word state changes
        // under the lock as before. A write trapping on a page still protected until then finds
        // the page already untracked and changes nothing; tracking (+1) is never deferred.
        static thread_local VideoCommon::DeferredPageCaching deferred;
        const bool defer = VideoCommon::t_deferred_page_caching == nullptr &&
                           VideoCore::Nx2Debug::Experiment(109) != 0;
        bool gpu_modified;
        {
            std::scoped_lock lock{buffer_cache.mutex};
            if (defer) {
                VideoCommon::t_deferred_page_caching = &deferred;
            }
            gpu_modified = buffer_cache.OnCPUWrite(addr, size);
            if (defer) {
                VideoCommon::t_deferred_page_caching = nullptr;
            }
        }
        // Step 7's verify: a game core's trapped store marked the tracker directly.
        mem_stage.NoteCpuMark(addr, size);
        if (defer) {
            deferred.ApplyAll();
        }
        if (gpu_modified) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

bool RasterizerVulkan::OneFollowerGatherOnEngine(
    std::span<const std::pair<DAddr, u64>> ranges) {
    namespace OneF = VideoCommon::OneFollower;
    if (!OneF::GatherOnEngine()) {
        return false;
    }
    // fable_b 12.3 fix 5: R's half is the texture and shader caches (ApplyGatherOtherHalf; the
    // query cache is not in a gather). No range on their pages: it does nothing, and E applies
    // the buffer half (ApplyGatherBufferHalf, the served gather's). The guest's writes are in
    // memory before the take, and R fences between an image's filter mark and its upload
    // (WrittenAheadOfRender): R either uploads the written bytes or its mark is seen here.
    for (const auto& [addr, size] : ranges) {
        if (addr != 0 && size != 0 &&
            (texture_cache.page_filter.Any(addr, size) ||
             pipeline_cache.page_filter.Any(addr, size))) {
            OneF::CountInvalidation(OneF::InvalKind::GatherLent);
            OneF::CountLentGatherRanges(ranges.size());
            return false;
        }
    }
    for (const auto& [addr, size] : ranges) {
        mem_stage.NoteCpuMark(addr, size);
        if (VideoCommon::Nx2FastDrawCount::g_on && addr != 0 && size != 0) [[unlikely]] {
            VideoCommon::Nx2FastDrawCount::NotePoolWrite(addr, size);
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        (void)ApplyGatherBufferHalf(ranges);
    }
    OneF::CountInvalidation(OneF::InvalKind::GatherOnEngine);
    return true;
}

void RasterizerVulkan::OnCacheInvalidations(std::span<const std::pair<DAddr, u64>> ranges) {
    for (const auto& [range_addr, range_size] : ranges) {
        VideoCommon::ComputeSkipHook::NoteWrite(range_addr, range_size); // ASYNC_COMPUTE_SKIP
    }
    if (VideoCommon::OneFollower::ShouldLend() && OneFollowerGatherOnEngine(ranges)) {
        return;
    }
    ONEF_LEND(Invalidation, OnCacheInvalidations(ranges))
    if (t_gather_probe != nullptr) [[unlikely]] {
        ProbeGatherRanges(ranges); // FRITCHII_MEMSTAGE_GATHER's counters only
    }
    // The memory seam's read log (5.3, test D): gathers applied on this thread.
    if (Common::WriteRing::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        mem_stage.NoteGather(MemStage::Gather::Apply, 0);
    }
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        VideoCommon::MemStageShadow::BackCacheInvalidation(true, ranges.size());
    }
    if (VideoCommon::MemoryStageLink::BackServed()) [[unlikely]] {
        // FRITCHII_MEMSTAGE_THREAD = 2, a gather inside a served entry (a read flush's): the
        // buffer half on GPUMem (claims first, as below), the other caches' here. Step 7's
        // verify notes the marks the cache takes (a game write the tracker holds from now).
        for (const auto& [addr, size] : ranges) {
            mem_stage.NoteCpuMark(addr, size);
        }
        OnStage(static_cast<u8>(VideoCommon::MemoryStageLink::Served::Gather), [&] {
            std::scoped_lock lock{buffer_cache.mutex};
            ApplyGatherBufferHalf(ranges);
        });
        ApplyGatherOtherHalf(ranges);
        return;
    }
    // Step 7's verify: the marks a gather applied here, on any thread (re-review N4).
    for (const auto& [addr, size] : ranges) {
        mem_stage.NoteCpuMark(addr, size);
    }
    namespace CbPath = VideoCommon::Nx2CbPath;
    const int cb_path = CbPath::Mode();
    // EXP88 & 1 (fable52_answer_next70 3.1 step 5): the guest's collected writes take each
    // cache's lock only where that cache may act. On the back only: it owns the vertex bindings
    // WriteReachesBuffers reads (the collected writes are applied there).
    if ((cb_path & CbPath::SkipBit) != 0 && VideoCommon::GpuSplit::OnBackThread()) {
        // Delta draw count (Fable plan D1, row 1): pools are not images, test them here. The
        // other path tests each range in OnCacheInvalidation.
        if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
            for (const auto& [addr, size] : ranges) {
                VideoCommon::Nx2FastDrawCount::NotePoolWrite(addr, size);
            }
        }
        auto& counters = CbPath::g_counters;
        const bool verify = (cb_path & CbPath::VerifyBit) != 0;
        const auto note_mismatch = [&counters](DAddr addr, u64 size, const char* what) {
            CbPath::Count(counters.verify_skip_mismatch);
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Vulkan, "EXP88 verify: guest write {:#x}+{:#x}: {}", addr, size,
                          what);
            }
        };
        // ClaimCpuWrites acts only on pages a GPU-modified mark ever reached.
        if (buffer_cache.ClaimMayFindGpuWrites(ranges)) {
            std::scoped_lock lock{buffer_cache.mutex};
            MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
            buffer_cache.ClaimCpuWrites(ranges);
        } else {
            CbPath::Count(counters.guest_claim_skips);
            if (verify) {
                std::scoped_lock lock{buffer_cache.mutex};
                MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
                for (const auto& [addr, size] : ranges) {
                    CbPath::Count(counters.verify_skip_checked);
                    if (buffer_cache.IsRegionGpuModified(addr, size)) {
                        note_mismatch(addr, size, "the skipped claim had GPU-written data");
                    }
                }
                buffer_cache.ClaimCpuWrites(ranges);
            }
        }
        // FRITCHII_GATHER_ONE_LOCK (live, default 0; fable_front_drains 8.2 G1): the same calls
        // in two passes, the texture and pipeline caches' first, then the buffer cache's under
        // one hold of its lock and one seam bracket (taken at the first range that needs them),
        // instead of a lock and a bracket per range. Each cache sees its calls in the order
        // below. Exact: no range's effect here depends on another cache's effect of another
        // range. The texture cache's WriteMemory and the pipeline cache's InvalidateRegion
        // change only their own state; the buffer cache's WriteMemory changes its own state and
        // guest memory (a page download, after the claim above) and records commands, neither
        // of which those two read. What they share is the device pages' cached counts, which
        // these calls only lower: the sum, and the moment a page's count reaches zero (it is
        // unprotected once its last user has marked its data CPU-modified, in the same call),
        // do not depend on the order. The bracket plays its entries before anything else is
        // recorded (SeamHook::OnUse), so one bracket records the same commands in the same
        // order as one per range.
        if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_gather_one_lock,
                                             "FRITCHII_GATHER_ONE_LOCK") != 0) {
            for (const auto& [addr, size] : ranges) {
                if (addr == 0 || size == 0) {
                    continue;
                }
                CbPath::Count(counters.guest_ranges);
                if (texture_cache.page_filter.Any(addr, size)) {
                    std::scoped_lock lock{texture_cache.mutex};
                    texture_cache.WriteMemory(addr, size);
                }
                if (pipeline_cache.page_filter.Any(addr, size)) {
                    pipeline_cache.InvalidateRegion(addr, size);
                }
            }
            // The bracket ends before the lock is released, as in the per-range scope.
            std::unique_lock buffer_lock{buffer_cache.mutex, std::defer_lock};
            std::optional<MemStage::FillPlay> seam;
            for (const auto& [addr, size] : ranges) {
                if (addr == 0 || size == 0) {
                    continue;
                }
                const bool skip = buffer_cache.WriteReachesBuffers(addr, size) ==
                                  BufferCache::WriteReach::None;
                if (skip) {
                    CbPath::Count(counters.guest_skipped);
                }
                if (skip && !verify) {
                    buffer_cache.NoteWriteUnlocked(addr, size);
                    continue;
                }
                if (!buffer_lock.owns_lock()) {
                    buffer_lock.lock();
                    seam.emplace(mem_stage, MemStage::Site::Invalidation);
                    CbPath::Count(counters.guest_one_lock_holds);
                }
                if (skip) {
                    CbPath::Count(counters.verify_skip_checked);
                    if (!buffer_cache.WriteChangesNothing(addr, size)) {
                        note_mismatch(addr, size, "the skipped write would change the buffer cache");
                    }
                }
                buffer_cache.WriteMemory(addr, size);
            }
            seam.reset();
            return;
        }
        for (const auto& [addr, size] : ranges) {
            if (addr == 0 || size == 0) {
                continue;
            }
            CbPath::Count(counters.guest_ranges);
            // No image or shader ever registered on these pages: their caches do nothing here
            // (as InnerInvalidation asks them).
            if (texture_cache.page_filter.Any(addr, size)) {
                std::scoped_lock lock{texture_cache.mutex};
                texture_cache.WriteMemory(addr, size);
            }
            const bool skip = buffer_cache.WriteReachesBuffers(addr, size) ==
                              BufferCache::WriteReach::None;
            if (skip) {
                CbPath::Count(counters.guest_skipped);
            }
            if (skip && !verify) {
                buffer_cache.NoteWriteUnlocked(addr, size);
            } else {
                std::scoped_lock lock{buffer_cache.mutex};
                MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
                if (skip) {
                    CbPath::Count(counters.verify_skip_checked);
                    if (!buffer_cache.WriteChangesNothing(addr, size)) {
                        note_mismatch(addr, size, "the skipped write would change the buffer cache");
                    }
                }
                buffer_cache.WriteMemory(addr, size);
            }
            if (pipeline_cache.page_filter.Any(addr, size)) {
                pipeline_cache.InvalidateRegion(addr, size);
            }
        }
        return;
    }
    // Out of the GPU-written ranges first: applying one write downloads the GPU's bytes elsewhere
    // on its pages (WriteMemory), which would write over the next collected write.
    {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
        buffer_cache.ClaimCpuWrites(ranges);
    }
    for (const auto& [addr, size] : ranges) {
        OnCacheInvalidation(addr, size);
    }
}

bool RasterizerVulkan::ApplyGatherBufferHalf(std::span<const std::pair<DAddr, u64>> ranges) {
    // The buffer cache's lock held (the served gather's body, OnCacheInvalidations; and
    // FRITCHII_MEMSTAGE_PASS on GPUMem). Claims first (as OnCacheInvalidations), then WriteMemory
    // per range, in one seam bracket. True when the bracket recorded a command (the arena grew).
    MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
    const u32 arena_before = mem_stage.ArenaHere();
    buffer_cache.ClaimCpuWrites(ranges);
    for (const auto& [addr, size] : ranges) {
        if (addr != 0 && size != 0) {
            buffer_cache.WriteMemory(addr, size);
        }
    }
    return mem_stage.ArenaHere() != arena_before;
}

void RasterizerVulkan::ApplyGatherOtherHalf(std::span<const std::pair<DAddr, u64>> ranges) {
    // The served gather's tail (OnCacheInvalidations): the texture and pipeline caches, on the
    // back, range by range.
    for (const auto& [addr, size] : ranges) {
        if (addr == 0 || size == 0) {
            continue;
        }
        if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
            VideoCommon::Nx2FastDrawCount::NotePoolWrite(addr, size);
        }
        {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.WriteMemory(addr, size);
        }
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

u8 RasterizerVulkan::StagePassGather(std::vector<std::pair<u64, u64>>& ranges) {
    // GPUMem at a passed gather (fable_stage_levelpoints 6.2 steps 6 and 7, with the amendment):
    // the take, then under the buffer cache's lock the page-aligned GPU-written test (as
    // ProbeGatherRanges: WriteMemory's recording branch looks at whole pages) and, when no page
    // holds GPU-written data, the served gather's buffer half at once: on such pages WriteMemory
    // only marks (the page flags, MarkRegionAsCpuModified, MarkVertexBuffersStale, which reach
    // the copy through the mark hook) and ClaimCpuWrites does nothing, so no command is needed.
    // The verify's marks as at the served gather.
    //
    // Rule T (fable_stage_levelpoints 13.3): the take is listed on applying_writes until the
    // buffer cache has it, so a landing of downloaded GPU bytes (read downloads, fence
    // downloads: the cache's lock held) leaves those bytes out meanwhile, as for a tracked
    // gather. Unlisted under the lock right after the buffer half; on a GPU-written page (or an
    // exception) it stays listed and the record goes whole: the back unlists it after its apply.
    // Lock order: the cache's lock, then applying_writes_mutex (a landing's, and the unlist's).
    using Link = VideoCommon::MemoryStageLink;
    const bool listed = gpu.TakeCpuWrites(ranges);
    if (ranges.empty()) {
        return static_cast<u8>(Link::PassTake::Empty);
    }
    bool maybe = false;
    for (const auto& [addr, size] : ranges) {
        if (addr == 0 || size == 0) {
            continue;
        }
        const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        if (MayBeGpuModified(begin, end - begin)) {
            maybe = true;
            break;
        }
    }
    const Link::CleanPoint clean;
    std::scoped_lock lock{buffer_cache.mutex};
    if (maybe) {
        for (const auto& [addr, size] : ranges) {
            if (addr == 0 || size == 0) {
                continue;
            }
            const DAddr begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
            const DAddr end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
            // GPUMem's own question (13.4): the real set, or a mark a fill ahead of the back
            // made (FRITCHII_MEMSTAGE_MARKS_AHEAD's list): such a page makes the record whole.
            if (buffer_cache.GpuWrittenHere(begin, end - begin)) {
                return static_cast<u8>(Link::PassTake::GpuWritten);
            }
        }
    }
    for (const auto& [addr, size] : ranges) {
        mem_stage.NoteCpuMark(addr, size);
    }
    const bool recorded = ApplyGatherBufferHalf(ranges);
    if (listed) {
        gpu.ReleaseTakenWrites(&ranges); // the cache's lock still held: no landing in between
    }
    return static_cast<u8>(recorded ? Link::PassTake::Recorded : Link::PassTake::Applied);
}

bool RasterizerVulkan::GatherOnStage() {
    // FRITCHII_MEMSTAGE_PASS bit 1 (fable_stage_levelpoints 6.3), the back in a served entry with
    // writes collected (GPU::Impl::InvalidateGPUCache tested that): GPUMem takes them and applies
    // their buffer half in one served call, the other caches' half follows here, as the served
    // branch of OnCacheInvalidations. The take is always GPUMem's while the back is served.
    using Link = VideoCommon::MemoryStageLink;
    std::vector<std::pair<DAddr, u64>> writes;
    OnStage(static_cast<u8>(Link::Served::Gather), [&] {
        // Rule T (fable_stage_levelpoints 13.3): the take listed until the buffer half is done,
        // unlisted inside the call under the cache's lock (also when the half throws).
        const bool listed = gpu.TakeCpuWrites(writes);
        if (writes.empty()) {
            return;
        }
        for (const auto& [addr, size] : writes) {
            mem_stage.NoteCpuMark(addr, size);
        }
        std::scoped_lock lock{buffer_cache.mutex};
        SCOPE_EXIT {
            if (listed) {
                gpu.ReleaseTakenWrites(&writes);
            }
        };
        ApplyGatherBufferHalf(writes);
    });
    Link::Get().NoteServedTake(!writes.empty());
    if (writes.empty()) {
        return true;
    }
    // The memory seam's read log (5.3, test D) and the shadow's count, as OnCacheInvalidations.
    if (Common::WriteRing::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        mem_stage.NoteGather(MemStage::Gather::Apply, 0);
    }
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        VideoCommon::MemStageShadow::BackCacheInvalidation(true, writes.size());
    }
    ApplyGatherOtherHalf(writes);
    return true;
}

void RasterizerVulkan::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    VideoCommon::ComputeSkipHook::NoteWrite(addr, size); // FRITCHII_ASYNC_COMPUTE_SKIP
    ONEF_LEND(Invalidation, OnCacheInvalidation(addr, size))
    if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        VideoCommon::MemStageShadow::BackCacheInvalidation(false, 1);
    }
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        VideoCommon::Nx2FastDrawCount::NotePoolWrite(addr, size);
    }

    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    mem_stage.NoteCpuMark(addr, size); // step 7's verify, as in OnCacheInvalidations
    OnStage(static_cast<u8>(VideoCommon::MemoryStageLink::Served::Invalidation), [&] {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
        buffer_cache.WriteMemory(addr, size);
    });
    pipeline_cache.InvalidateRegion(addr, size);
}

void RasterizerVulkan::Nx2ApplyRunaheadChunk(const VideoCommon::Runahead::Chunk& chunk) {
    // One follower: runs on E (Runahead::ApplyDue) and touches the texture and pipeline caches,
    // which are R's: lent like OnCacheInvalidations.
    ONEF_LEND(Invalidation, Nx2ApplyRunaheadChunk(chunk))
    // As OnCacheInvalidations, with the buffer cache taking the chunk's bytes instead of a mark.
    std::vector<std::pair<DAddr, u64>> ranges;
    ranges.reserve(chunk.ranges.size());
    for (const auto& range : chunk.ranges) {
        ranges.emplace_back(static_cast<DAddr>(range.address), range.size);
    }
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        for (const auto& [addr, size] : ranges) {
            VideoCommon::Nx2FastDrawCount::NotePoolWrite(addr, size);
        }
    }
    const auto any = [&](const VideoCommon::PageFilter& filter) {
        return std::ranges::any_of(ranges, [&filter](const std::pair<DAddr, u64>& range) {
            return filter.Any(range.first, range.second);
        });
    };
    {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
        buffer_cache.ClaimCpuWrites(ranges);
    }
    // Images still upload from guest memory when next used (the front writes no word early for
    // a chunk with image pages, so the CPU has not run ahead of those bytes).
    if (any(texture_cache.page_filter)) {
        std::scoped_lock lock{texture_cache.mutex};
        for (const auto& [addr, size] : ranges) {
            if (texture_cache.page_filter.Any(addr, size)) {
                texture_cache.WriteMemory(addr, size);
            }
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Invalidation};
        for (const auto& range : chunk.ranges) {
            buffer_cache.Nx2ApplyCpuBytes(static_cast<DAddr>(range.address), range.size,
                                          chunk.bytes.data() + range.offset);
        }
    }
    if (any(pipeline_cache.page_filter)) {
        for (const auto& [addr, size] : ranges) {
            pipeline_cache.InvalidateRegion(addr, size);
        }
    }
}

bool RasterizerVulkan::TryCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) return true;
    ONEF_LEND_RESULT(Invalidation, TryCacheInvalidation(addr, size))
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        // WriteMemory downloads GPU-written bytes elsewhere on the written pages, which only the
        // GPU thread may do: leave those pages to the fallback.
        const DAddr page_begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE);
        if (buffer_cache.IsRegionGpuModified(page_begin, page_end - page_begin)) return false;
        if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
            VideoCommon::Nx2FastDrawCount::NotePoolWrite(addr, size);
        }
        texture_cache.WriteMemory(addr, size);
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return true;
}

bool RasterizerVulkan::OneFollowerRunaheadOff(const char* where) {
    const auto mode = VideoCommon::Runahead::CurrentMode();
    // FRITCHII_ONEF_GATHER_E's last condition, logged once per change.
    static std::atomic<int> logged{-1};
    if (logged.exchange(static_cast<int>(mode), std::memory_order_relaxed) !=
        static_cast<int>(mode)) [[unlikely]] {
        LOG_INFO(Render_Vulkan, "One follower: gathers on E see runahead mode {} (at {})",
                 static_cast<int>(mode), where);
    }
    return mode == VideoCommon::Runahead::Mode::Off;
}

void RasterizerVulkan::InvalidateGPUCache() {
    if (onef_fence_posted && VideoCommon::OneFollower::OnRenderThread()) [[unlikely]] {
        return; // a posted fence's closing gather: E ran it at the fence's place
    }
    // FRITCHII_ONEF_GATHER_E: the take runs on E (GPU::InvalidateGPUCache, as on the split's back
    // without one follower); OnCacheInvalidations then applies or lends. Runahead keeps the
    // whole gather lent (its boundary state).
    const bool onef_on_engine =
        VideoCommon::OneFollower::ShouldLend() && VideoCommon::OneFollower::GatherOnEngine() &&
        OneFollowerRunaheadOff("InvalidateGPUCache");
    if (!onef_on_engine) {
        ONEF_LEND(Invalidation, InvalidateGPUCache())
    }
    // FRITCHII_MEMSTAGE_PASS (fable_stage_levelpoints 6.5): at a passed gather's entry, the half
    // of GPUMem's take the back owes (the other caches', or all of it when the entry runs as a
    // level run), before this gather takes more. Records of earlier entries the back never asked
    // for are late (must be 0) and applied there.
    using Link = VideoCommon::MemoryStageLink;
    if (Link::RingActive() || Link::Get().PassPending()) [[unlikely]] {
        Link::Get().BackApplyPasses(VideoCommon::GpuSplit::Get().BackEntryPosition());
    }
    gpu.InvalidateGPUCache();
}

void RasterizerVulkan::InvalidateTextureDataCache() {
    // FRITCHII_ONEF_GATHER_E: on E (its gather as InvalidateGPUCache's; WaitForIdle stays lent).
    const bool onef_on_engine =
        VideoCommon::OneFollower::ShouldLend() && VideoCommon::OneFollower::GatherOnEngine() &&
        OneFollowerRunaheadOff("InvalidateTextureDataCache");
    if (!onef_on_engine) {
        ONEF_LEND(Invalidation, InvalidateTextureDataCache())
    }
    // FRITCHII_MEMSTAGE_GATHER: GPUMem, level at this entry, had the back run this gather already
    // (StageRunGather); the back's own run of the entry skips it. BackTakeGather counts a served
    // gather with no record (must be 0). Without the ring, and for every gather that started a
    // run, the default below.
    // A record can outlive the ring by one entry (GPUMem failed in the request): it is honoured
    // without the ring too, so a gather is never run twice.
    // FRITCHII_MEMSTAGE_PASS: a gather GPUMem passed has a pass record, asked for first
    // (fable_stage_p3 1): it runs here (InvalidateGPUCache applies the record), and
    // BackTakeGather, which counts a served gather without a request record, is not asked.
    using Link = VideoCommon::MemoryStageLink;
    Link& link = Link::Get();
    const u64 position = VideoCommon::GpuSplit::Get().BackEntryPosition();
    const bool passed =
        (Link::RingActive() || link.PassPending()) && link.BackPassRecord(position) != nullptr;
    if (!passed && (Link::RingActive() || link.GatherPending()) && link.BackTakeGather(position))
        [[unlikely]] {
        return;
    }
    InvalidateGPUCache();
    WaitForIdle();
}

void RasterizerVulkan::UnmapMemory(DAddr addr, u64 size) {
    ONEF_LEND(Invalidation, UnmapMemory(addr, size))
    // EXP92 (fable52_answer_uniformkeeps 3.3): the uniform keeps' page watches on this device
    // memory are given back here, before the backing is cleared (no cache lock held). A later keep
    // watches the page again. DeviceMemoryManager::Map and Unmap do the same for backing changes
    // that come without a GPU unmap.
    Common::UniformKeepPages::Unwatch(addr, size, [this](DAddr page_address, s32 delta) {
        device_memory.UpdatePagesCachedCount(page_address, Core::DEVICE_PAGESIZE, delta);
    });
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Unmap};
        buffer_cache.WriteMemory(addr, size, false);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerVulkan::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    ONEF_LEND(Invalidation, ModifyGPUMemory(as_id, addr, size))
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapGPUMemory(as_id, addr, size);
    }
    buffer_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerVulkan::OneFollowerCountFence() {
    // fable_b 12.3 fix 3's price: a fence whose cache commit has no buffer half to do on E.
    VideoCommon::OneFollower::CountLentFence(buffer_cache.MayHaveUncommittedFlushes() ||
                                             buffer_cache.AheadDue());
}

void RasterizerVulkan::SignalFence(std::function<void()>&& func) {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        OneFollowerCountFence();
        // FRITCHII_ONEF_SEM_E (fable_b 19.6): with fences not delayed the callback runs at once
        // inside SignalFence (a QMD reduction or max release, the GP sync): E runs it at the
        // fence's place and posts the fence with an empty one.
        // FRITCHII_ONEF_SIGNAL_ON_R: the callback writes game memory the guest acts on; it runs
        // on R at R's place, which (the callback may touch E's caches) means lent.
        if (VideoCommon::OneFollower::SemOnE() && !VideoCommon::OneFollower::SignalOnR() &&
            OneFollowerPostKind(8) && OneFollowerFenceClean()) {
            VideoCommon::OneFollower::NoteEngineSignal("a SignalFence callback");
            OneFollowerPostFence(4, 0, &func);
            return;
        }
        OneFollowerFenceLent(); // lent (its callback is the caller's): R syncs, E parked
    }
    ONEF_LEND(Fence, SignalFence(std::move(func)))
    fence_manager.SignalFence(std::move(func));
}

void RasterizerVulkan::SyncOperation(std::function<void()>&& func) {
    ONEF_LEND(Fence, SyncOperation(std::move(func)))
    fence_manager.SyncOperation(std::move(func));
}

u64 RasterizerVulkan::SyncOperationStamped(std::function<void()>&& func) {
    ONEF_LEND_RESULT(Fence, SyncOperationStamped(std::move(func)))
    return fence_manager.SyncOperationStamped(std::move(func));
}

bool RasterizerVulkan::AppendToLastOperation(u64 stamp, void (*append)(void*), void* context) {
    ONEF_LEND_RESULT(Fence, AppendToLastOperation(stamp, append, context))
    return fence_manager.AppendToLastOperation(stamp, append, context);
}

void RasterizerVulkan::SyncOperationKeepStamp(std::function<void()>&& func) {
    ONEF_LEND(Fence, SyncOperationKeepStamp(std::move(func)))
    fence_manager.SyncOperationKeepStamp(std::move(func));
}

void RasterizerVulkan::OneFollowerDeferredTick() {
    if (onef_frame_tick_deferred.load(std::memory_order_relaxed) &&
        onef_frame_tick_deferred.exchange(false, std::memory_order_acq_rel)) [[unlikely]] {
        TickFrame();
    }
}

void RasterizerVulkan::SignalSyncPoint(u32 value) {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        OneFollowerDeferredTick();
        OneFollowerCountFence();
        if (OneFollowerPostKind(1) && OneFollowerFenceClean()) {
            // FRITCHII_ONEF_SEM_E (fable_b 20.3): the guest increment, then (after the buffer
            // half, as the fence's callback runs at once) the host increment, both at E's
            // place; R makes an empty fence. The game's wait on the sync point then ends when E
            // passes it. Bound: the frame end stays lent (E parks there until R played the
            // frame), so E is at most a frame ahead of R.
            // FRITCHII_ONEF_SIGNAL_ON_R: R increments both at its played place (kind 0 below).
            if (VideoCommon::OneFollower::SemOnE() && VideoCommon::OneFollower::SignalOnR()) {
                VideoCommon::OneFollower::CountSignalOnR();
            } else if (VideoCommon::OneFollower::SemOnE()) {
                VideoCommon::OneFollower::NoteEngineSignal("a sync point increment");
                if (fence_manager.OneFollowerSyncPointGuest(value)) {
                    OneFollowerPostFence(3, value);
                }
                return;
            }
            OneFollowerPostFence(0, value);
            return;
        }
        OneFollowerFenceLent(); // lent: R syncs what is owed, E parked
    }
    ONEF_LEND(Fence, SignalSyncPoint(value))
    fence_manager.SignalSyncPoint(value);
}

void RasterizerVulkan::SignalReference() {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        OneFollowerDeferredTick();
        OneFollowerCountFence();
        if (OneFollowerPostKind(2) && OneFollowerFenceClean()) {
            OneFollowerPostFence(1, 0);
            return;
        }
        OneFollowerFenceLent();
    }
    ONEF_LEND(Fence, SignalReference())
    fence_manager.SignalReference();
}

void RasterizerVulkan::ReleaseFences(bool force) {
    // FRITCHII_ONEF_SOFT_RELEASE_E (fable_b 22.3): a non-forced release (GPU::OnCommandListEnd,
    // 1,400 a second, the lent "ReleaseFences" site) only waits for fences without a fencing
    // thread; with one (Vulkan) WaitPendingFences(false) does nothing, so E answers it.
    if (!force && VideoCommon::OneFollower::ShouldLend() &&
        VideoCommon::OneFollower::SoftReleaseOnE()) [[unlikely]] {
        return;
    }
    if (VideoCommon::OneFollower::ShouldLend() && force) [[unlikely]] {
        if (OneFollowerPostKind(4) && OneFollowerFenceClean()) {
            OneFollowerPostFence(2, 0);
            return;
        }
        OneFollowerFenceLent(); // a forced release makes a fence: R syncs there
    }
    ONEF_LEND(Fence, ReleaseFences(force))
    fence_manager.WaitPendingFences(force);
}

bool RasterizerVulkan::FencesQuiet(bool strict, u64& epoch) {
    ONEF_LEND_RESULT(Fence, FencesQuiet(strict, epoch))
    return fence_manager.FencesQuiet(strict, epoch);
}

u64 RasterizerVulkan::FenceWorkEpoch() const {
    return fence_manager.FenceWorkEpoch();
}

namespace {
/// FRITCHII_QUERY_FLUSH_DRY=1 (live, default 0; helper DE, a dry timer that changes nothing).
/// kaksample2 (Kakariko, unprofiled CPU sample): E parked ~30 ms/s in FlushQueryResults lends
/// (the front's report DMA flush runs, QueryDma::FlushPending) while the report waits line shows
/// R finding something to wait for under 1 time a second; the profile's lends line does not list
/// these lends at all. Per lend: E's parked time, whether R's query cache had reports in the
/// range (SemiFlushRegion), and whether E's own report-landing pages call the range clear
/// (OneFollower::ReportLandingClear). The lever would skip the lend where E can prove R has
/// nothing; "clear but R found" must be 0 for E's pages to be that proof. Only E touches it.
struct QueryFlushDry {
    static bool On() {
        return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_query_flush_dry,
                                                "FRITCHII_QUERY_FLUSH_DRY") != 0;
    }
    static u64 Now() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }
    void Note(bool clear, bool found, u64 parked_ns, u64 now) {
        ++lends;
        parked += parked_ns;
        if (found) {
            ++found_count;
            found_parked += parked_ns;
        }
        if (clear) {
            ++clear_count;
            clear_parked += parked_ns;
            if (found) {
                ++clear_found;
            } else {
                ceiling += parked_ns;
            }
        }
        if (since == 0) {
            since = now;
            return;
        }
        if (now - since < 10'000'000'000ULL) {
            return;
        }
        const double seconds = static_cast<double>(now - since) / 1e9;
        const auto rate = [&](u64 count) { return static_cast<double>(count) / seconds; };
        const auto ms = [&](u64 ns) { return static_cast<double>(ns) / 1e6 / seconds; };
        LOG_INFO(Render_Vulkan,
                 "Query flush lends (FRITCHII_QUERY_FLUSH_DRY=1) {:.0f} s: {:.0f}/s, E parked "
                 "{:.1f} ms/s ({:.0f} us each); R found reports in {:.0f}/s ({:.1f} ms/s); E's "
                 "landing pages clear for {:.0f}/s ({:.1f} ms/s), R found reports in {:.0f}/s of "
                 "them (must be 0 to skip on it; {} since boot); ceiling (clear, nothing found) "
                 "{:.1f} ms/s",
                 seconds, rate(lends), ms(parked),
                 lends != 0 ? static_cast<double>(parked) / 1e3 / static_cast<double>(lends) : 0.0,
                 rate(found_count), ms(found_parked), rate(clear_count), ms(clear_parked),
                 rate(clear_found), clear_found_total + clear_found, ms(ceiling));
        clear_found_total += clear_found;
        const u64 total = clear_found_total;
        *this = {};
        clear_found_total = total;
        since = now;
    }
    u64 since{};
    u64 lends{};
    u64 parked{};
    u64 found_count{};
    u64 found_parked{};
    u64 clear_count{};
    u64 clear_parked{};
    u64 clear_found{};
    u64 ceiling{};
    u64 clear_found_total{};
};
QueryFlushDry g_query_flush_timer;
/// R writes it in the lent call (SemiFlushRegion found reports); E reads it after the lend.
std::atomic<bool> g_query_flush_found{};
} // Anonymous namespace

void RasterizerVulkan::FlushQueryResults(DAddr addr, u64 size) {
    if (VideoCommon::OneFollower::ShouldLend() && QueryFlushDry::On()) [[unlikely]] {
        // FRITCHII_QUERY_FLUSH_DRY: the same lend as ONEF_LEND below, timed and classified.
        const DAddr begin = addr >= 12 ? addr - 12 : DAddr{0};
        const bool clear = VideoCommon::OneFollower::ReportLandingClear(begin, addr + size - begin);
        g_query_flush_found.store(false, std::memory_order_relaxed);
        const u64 start = QueryFlushDry::Now();
        VideoCommon::OneFollower::SetLendSite(__func__);
        auto onef_call = [&] { FlushQueryResults(addr, size); };
        VideoCommon::OneFollower::Lend(VideoCommon::OneFollower::LentOp::ReadBack, onef_call);
        const u64 now = QueryFlushDry::Now();
        g_query_flush_timer.Note(clear, g_query_flush_found.load(std::memory_order_acquire),
                               now - start, now);
        return;
    }
    ONEF_LEND(ReadBack, FlushQueryResults(addr, size))
    if (addr == 0 || size == 0 || !query_cache.SemiFlushRegion(addr, size)) {
        return;
    }
    g_query_flush_found.store(true, std::memory_order_release);
    // GPU profile zone "query DMA wait": these waits per second and their ms per second, either
    // way below (the A/B measure for EXP30).
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::QueryDmaWait};
    // FRITCHII_REPORT_WAIT (live, default 0; fritchii_report_wait.h): 0 = EXP30 decides (as
    // before), 1 = the lean wait below, 2 = verify (the lean wait, then the old one, the words
    // compared). Every value times the wait for the "Report waits" line.
    namespace ReportWait = VideoCommon::ReportWait;
    int report_wait = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_report_wait,
                                                       "FRITCHII_REPORT_WAIT");
    if (report_wait < 0 || report_wait > 2) {
        report_wait = 0;
    }
    ReportWait::MaybeReport(report_wait);
    const u64 wait_start = ReportWait::NowNs();
    // From 12 bytes before the read: a 16-byte report (value, then timestamp) starting there
    // still has words in it, and its slot is the one before the read's.
    const DAddr wait_begin = addr >= 12 ? addr - 12 : DAddr{0};
    const u64 wait_bytes = addr + size - wait_begin;
    // Below High GPU accuracy a fence runs its own callback at once, so wait for an operation
    // queued behind the pending report writes instead. On the split's back the fence's
    // submission may be deferred, and the fencing thread would wait for this thread to submit it.
    const auto old_wait = [&] {
        std::mutex wait_mutex;
        std::condition_variable wait_cv;
        bool written{};
        fence_manager.SyncOperation([&] {
            std::scoped_lock lock{wait_mutex};
            written = true;
            wait_cv.notify_all();
        });
        fence_manager.SignalFence([] {});
        scheduler.FlushDeferred();
        std::unique_lock lock{wait_mutex};
        wait_cv.wait(lock, [&written] { return written; });
    };
    if (report_wait == 2) [[unlikely]] {
        // Verify: the lean wait (polled, so a range that never settles is counted and handed to
        // the old wait instead of hanging), the words, the old wait, the words again.
        fence_manager.SignalFence([] {});
        scheduler.FlushDeferred();
        constexpr u64 lean_limit_ns = 200'000'000;
        bool settled = true;
        while (!Tegra::Nx2AsyncReportsSettled(wait_begin, wait_bytes)) {
            if (ReportWait::NowNs() - wait_start > lean_limit_ns) {
                settled = false;
                break;
            }
            std::this_thread::yield();
        }
        const u64 lean_end = ReportWait::NowNs();
        // The read FlushPending serves is one page at most.
        const size_t bytes = static_cast<size_t>((std::min<u64>)(size, 0x1000));
        std::array<u8, 0x1000> lean_words;
        std::array<u8, 0x1000> old_words;
        device_memory.ReadBlockUnsafe(addr, lean_words.data(), bytes);
        old_wait();
        const u64 old_end = ReportWait::NowNs();
        device_memory.ReadBlockUnsafe(addr, old_words.data(), bytes);
        const bool differs =
            !settled || std::memcmp(lean_words.data(), old_words.data(), bytes) != 0;
        if (ReportWait::NoteVerify(lean_end - wait_start, old_end - wait_start, differs))
            [[unlikely]] {
            size_t first = 0;
            while (first + 1 < bytes && lean_words[first] == old_words[first]) {
                ++first;
            }
            LOG_ERROR(HW_GPU,
                      "Report waits (FRITCHII_REPORT_WAIT=2, must be 0): the read of {} bytes at "
                      "{:#x} {} after the lean wait ({:.2f} ms); first different byte {} ({:#x} "
                      "after the lean wait, {:#x} after the old one)",
                      bytes, addr, settled ? "changed" : "did not settle in 200 ms",
                      static_cast<double>(lean_end - wait_start) / 1e6, first, lean_words[first],
                      old_words[first]);
        }
        ReportWait::NoteWait(old_end - wait_start, true);
        return;
    }
    if (report_wait == 1 || VideoCore::Nx2Debug::Experiment(30) != 0) {
        // EXP30: the fence carries the reports still waiting for one; on
        // the split's back its submission may be deferred, and the fencing thread would wait
        // for this thread to submit it. Then wait for the reports themselves, not for an
        // operation queued behind everything: the work submitted after them (a few ms of GPU
        // time while walking) cannot write them.
        fence_manager.SignalFence([] {});
        scheduler.FlushDeferred();
        {
            VideoCommon::BackWaitScope wait_scope{
                "RasterizerVulkan::FlushQueryResults (own reports)"};
            Tegra::Nx2WaitAsyncReportsSettled(wait_begin, wait_bytes);
        }
        ReportWait::NoteWait(ReportWait::NowNs() - wait_start, true);
        return;
    }
    old_wait();
    ReportWait::NoteWait(ReportWait::NowNs() - wait_start, false);
}

void RasterizerVulkan::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                                VideoCommon::CacheType which) {
    if (VideoCore::Nx2Debug::AccuratePart(8)) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerVulkan::WaitForIdle() {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        // FRITCHII_ONEF_POST_IDLE (fable_b 22.3): nothing comes back to E; posted when no
        // report of another kind owes the query sync (16.2's form). Posted with R behind, the
        // image flashed dark (stepflash, 8 Nov 15): the lent form also kept R caught up with E
        // 5,900 times a second. 1 posts only with R caught up (the lent form's order exactly;
        // E only stops waiting for the wait for idle itself); 2 also with R behind but no
        // posted fence with report landings unplayed (13.0), to test that dependency.
        namespace OneF = VideoCommon::OneFollower;
        if (const int mode = OneFollowerPostKind(64) ? OneF::PostIdleOn() : 0; mode != 0) {
            if (onef_sync_owed) {
                OneF::CountIdleDecision(OneF::IdleDecision::LentSyncOwed);
            } else if (OneF::RenderCaughtUp()) {
                OneF::CountIdleDecision(OneF::IdleDecision::PostedCaughtUp);
                OneFollowerPostIdle();
                return;
            } else if (mode == 1) {
                OneF::CountIdleDecision(OneF::IdleDecision::LentBehind);
            } else if (OneF::LandingOutstanding()) {
                OneF::CountIdleDecision(OneF::IdleDecision::LentLanding);
            } else {
                OneF::CountIdleDecision(OneF::IdleDecision::PostedBehind);
                OneFollowerPostIdle();
                return;
            }
        }
        // Lent: NotifyWFI syncs what is owed, E parked (it lands nothing: no fence); the
        // payload streamer syncs all of its own there.
        onef_sync_owed = false;
        onef_sync_samples.clear();
        onef_sync_payloads.clear();
    }
    ONEF_LEND(ReadBack, WaitForIdle())
    // Everything but wait pixel operations. This intentionally includes FRAGMENT_SHADER_BIT because
    // fragment shaders can still write storage buffers.
    VkPipelineStageFlags flags =
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
        VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
        VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        flags |= VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT;
    }

    // FRITCHII_MEMSTAGE_PASS (fable_stage_levelpoints 6.5): served, this is a passed entry's
    // wait for idle, whose mirror said no query sync could be pending (rule Q): one here would
    // sync through the buffer cache on the back (must be 0; it then runs as today). Bit 2 (P3,
    // fable_stage_p3 5.5): unless GPUMem resolved the sync's buffers in the entry's record, which
    // SyncValues then uses (and checks: PassSyncTarget).
    if (VideoCommon::MemoryStageLink::BackServed() && query_cache.HasPendingSync()) [[unlikely]] {
        const auto* const record = VideoCommon::MemoryStageLink::Get().BackCurrentPassRecord();
        if (record == nullptr || !record->sync) {
            VideoCommon::MemoryStageLink::Note(VideoCommon::MemoryStageLink::Check::PassQuerySync,
                                               "RasterizerVulkan::WaitForIdle");
        }
    }
    query_cache.NotifyWFI();

    if (device.IsMoltenVK() && scheduler.IsRenderPassActive()) {
        scheduler.DeferFragmentBarrier();
        fence_manager.SignalOrdering();
        return;
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([flags](vk::CommandBuffer cmdbuf) {
        // Later work waits for the earlier work and sees its writes.
        static constexpr VkMemoryBarrier WFI_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(flags, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WFI_BARRIER);
    });
    fence_manager.SignalOrdering();
}

void RasterizerVulkan::FragmentBarrier() {
    // FRITCHII_ONEF_POSTED: R's scheduler alone, nothing back to E: a render op.
    if (VideoCommon::OneFollower::ShouldLend() && VideoCommon::OneFollower::Posted() &&
        VideoCommon::OneFollower::Step() >= VideoCommon::OneFollower::StepLevel::Build)
        [[unlikely]] {
        RasterizerVulkan* const self = this;
        VideoCommon::OneFollower::SetLendSite("FragmentBarrier (posted)");
        VideoCommon::OneFollower::PushOp(
            VideoCommon::OneFollower::LentOp::Other,
            [](void* raw) {
                RasterizerVulkan* rasterizer;
                std::memcpy(&rasterizer, raw, sizeof(rasterizer));
                rasterizer->FragmentBarrier();
            },
            &self, sizeof(self));
        return;
    }
    ONEF_LEND(Other, FragmentBarrier())
    // The next draw breaks the render pass only if it reads what this one wrote: Ocarina
    // ended ~5,900 render passes a second here.
    if (device.IsMoltenVK() || VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::FragmentBarrier)) {
        scheduler.DeferFragmentBarrier();
        return;
    }
    // We already put barriers when a render pass finishes
    scheduler.RequestOutsideRenderPassOperationContext();
}

void RasterizerVulkan::TiledCacheBarrier() {
    // TODO: Implementing tiled barriers requires rewriting a good chunk of the Vulkan backend
}

void RasterizerVulkan::FlushCommands() {
    // FRITCHII_ONEF_POSTED: R's own counters (draw_counter, pending_native_work: R's draws,
    // clears and dispatches) and its scheduler, nothing back to E: a render op.
    if (VideoCommon::OneFollower::ShouldLend() && VideoCommon::OneFollower::Posted() &&
        VideoCommon::OneFollower::Step() >= VideoCommon::OneFollower::StepLevel::Build)
        [[unlikely]] {
        RasterizerVulkan* const self = this;
        VideoCommon::OneFollower::SetLendSite("FlushCommands (posted)");
        VideoCommon::OneFollower::PushOp(
            VideoCommon::OneFollower::LentOp::FrameEnd,
            [](void* raw) {
                RasterizerVulkan* rasterizer;
                std::memcpy(&rasterizer, raw, sizeof(rasterizer));
                rasterizer->FlushCommands();
            },
            &self, sizeof(self));
        return;
    }
    ONEF_LEND(FrameEnd, FlushCommands())
    if (draw_counter == 0 && !pending_native_work) {
        return;
    }
    draw_counter = 0;
    pending_native_work = false;
    // EDEN_NX2_EXP61 (live, 0 = off): nothing recorded since the last submit, nothing to send.
    // A fence with a download is flushed by InnerFence::Queue and then again here with an empty
    // command buffer: ~4,600 empty submits a second in Castle Town (fable52_answer_8 s8).
    if (VideoCore::Nx2Debug::Experiment(61) != 0 && !scheduler.RecordedSinceSubmit()) {
        return;
    }
    // Fences with downloads or queries come here right after queueing: ~9,400 a second in
    // Hyrule Warriors: Age of Imprisonment fights. Batched like the fences themselves.
    if (!scheduler.DeferFlush()) {
        scheduler.Flush();
    }
}

void RasterizerVulkan::TraceRenderState(bool is_indexed, const char* outcome,
                                        const Framebuffer* framebuffer) {
    if (!VideoCommon::RenderTrace::SampleFrame(render_trace_frame)) {
        return;
    }
    if (render_trace_draws++ >= 2048) {
        if (render_trace_draws == 2049) {
            LOG_DEBUG(Render_Vulkan, "Render trace frame={} truncated after 2048 draws",
                      render_trace_frame);
        }
        return;
    }
    const auto& regs = maxwell3d->regs;
    const auto& draw = maxwell3d->draw_manager.draw_state;
    std::array<u64, 6> shaders{};
    u32 stages{};
    for (size_t stage = 0; stage < shaders.size(); ++stage) {
        shaders[stage] = (u64{regs.pipelines[stage].reserved[0] & 0xff} << 32) |
                         regs.pipelines[stage].reserved[1];
        stages |= static_cast<u32>(regs.IsShaderConfigEnabled(stage)) << stage;
    }
    const auto extent = framebuffer ? framebuffer->RenderArea() : VkExtent2D{};
    LOG_DEBUG(Render_Vulkan,
              "Render trace frame={} draw={} docked={} outcome={} indexed={} stages={:#x} "
              "shaders={:x} guest_count={} guest_instances={} framebuffer={}x{} layers={}",
              render_trace_frame, render_trace_draws, Settings::IsDockedMode(), outcome, is_indexed,
              stages, fmt::join(shaders, ","),
              is_indexed ? draw.index_buffer.count : draw.vertex_buffer.count, draw.instance_count,
              extent.width, extent.height, framebuffer ? framebuffer->Layers() : 0);
    LOG_DEBUG(Render_Vulkan,
              "Render trace targets: color={:#x} {}x{} layers={} format={:#x} depth={:#x} "
              "{}x{} layers={} format={:#x} enabled={} clip={}x{}",
              regs.rt[0].Address(), regs.rt[0].width, regs.rt[0].height, regs.rt[0].depth.Value(),
              static_cast<u32>(regs.rt[0].format), regs.zeta.Address(), regs.zeta_size.width,
              regs.zeta_size.height, regs.zeta_size.depth.Value(), static_cast<u32>(regs.zeta.format),
              regs.zeta_enable, regs.surface_clip.width.Value(), regs.surface_clip.height.Value());
    const auto& viewport = regs.viewport_transform[0];
    const auto& scissor = regs.scissor_test[0];
    LOG_DEBUG(Render_Vulkan,
              "Render trace tests: raster={} depth_test={} depth_write={} depth_func={} "
              "cull={} face={} alpha={} func={} ref={} color_mask={:#x} "
              "viewport_scale=({},{},{}) translate=({},{},{}) scissor={} {}..{} {}..{}",
              regs.rasterize_enable, regs.depth_test_enable, regs.depth_write_enabled,
              static_cast<u32>(regs.depth_test_func), regs.gl_cull_test_enabled,
              static_cast<u32>(regs.gl_cull_face), regs.alpha_test_enabled,
              static_cast<u32>(regs.alpha_test_func), regs.alpha_test_ref, regs.color_mask[0].raw,
              viewport.scale_x, viewport.scale_y, viewport.scale_z, viewport.translate_x,
              viewport.translate_y, viewport.translate_z, scissor.enable, scissor.min_x.Value(),
              scissor.max_x.Value(), scissor.min_y.Value(), scissor.max_y.Value());
    for (size_t binding = 0; binding < regs.vertex_streams.size(); ++binding) {
        const auto& stream = regs.vertex_streams[binding];
        if (stream.IsEnabled()) {
            LOG_DEBUG(Render_Vulkan,
                      "Render trace vertex: binding={} address={:#x} limit={:#x} stride={} "
                      "frequency={} repacked={}",
                      binding, stream.Address(), regs.vertex_stream_limits[binding].Address(),
                      stream.stride.Value(), stream.frequency,
                      ((back_repack_mask >> binding) & 1) != 0);
        }
    }
    if (is_indexed) {
        LOG_DEBUG(Render_Vulkan, "Render trace index: address={:#x} limit={:#x} format={} first={}",
                  regs.index_buffer.StartAddress(), regs.index_buffer.EndAddress(),
                  static_cast<u32>(regs.index_buffer.format), draw.index_buffer.first);
    }
    LOG_DEBUG(Render_Vulkan, "Render trace condition: address={:#x} mode={} override={}",
              regs.render_enable.Address(), static_cast<u32>(regs.render_enable.mode),
              static_cast<u32>(regs.render_enable_override));
}

void RasterizerVulkan::TracePendingCpuWrites(bool compute, std::span<const u64> shader_hashes) {
    // Guest CPU writes into GPU-written buffer memory are collected, and applied to the host
    // buffers only at the next fence or cache invalidation. A hit here reads the older bytes.
    static const bool enabled = std::getenv("EDEN_NX2_PENDING_WRITE_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto*, u32) {
        if (lines > max_lines || !gpu.HasPendingCPUWrites(binding.device_addr, bound_size)) {
            return;
        }
        if (++lines > max_lines) {
            LOG_INFO(HW_GPU, "Pending write trace stopped after {} lines", max_lines);
            return;
        }
        const size_t program = compute ? 0 : stage + 1;
        LOG_INFO(HW_GPU,
                 "Pending CPU write frame={} {} shader={:016x} stage={} {}[{}]{} addr={:#x} "
                 "device={:#x} size={:#x}",
                 render_trace_frame, compute ? "dispatch" : "draw",
                 program < shader_hashes.size() ? shader_hashes[program] : 0, stage,
                 storage ? "ssbo" : "cbuf", index, written ? " written" : "", binding.gpu_addr,
                 binding.device_addr, bound_size);
    });
}

bool RasterizerVulkan::OcarinaTemporalEnabled() const {
    constexpr u64 ocarina_program = 0x0400486024DEC000ULL;
    if (program_id != ocarina_program || oot_live_failed) {
        return false;
    }
    if (oot_backend) {
        return *oot_backend != VideoCommon::OcarinaUpscaleBackend::Native;
    }
    oot_backend = ResolveOcarinaUpscaler(device);
    // FSR 4 takes the scene without camera jitter (FRITCHII_FSR4_JITTER), decided once.
    oot_fsr4_unjittered = *oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr &&
                          Fsr4Upscaler::RunsUnjittered(device);
    const bool fsr4_wanted = *oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr &&
                             Fsr4Upscaler::IsWanted(device);
    oot_dlss_jitter = VideoCommon::OcarinaDlssJitterOn(Settings::values.oot_dlss_jitter.GetValue());
    if (*oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss) {
        LOG_INFO(Render_Vulkan, "Ocarina temporal: DLSS jitter {} ({})",
                 oot_dlss_jitter ? "on" : "off",
                 VideoCommon::OcarinaDlssJitterOverride() >= 0 ? "EDEN_NX2_DLSS_JITTER" : "setting");
    }
    // Which upscaler OoT got and from what (2026-11-11: a player with DLSS on and Streamline
    // loaded had no DLSS in 10 minutes, and the log couldn't say why).
#ifdef EGLANTINE_STREAMLINE
    const char* const dlss_status =
        Streamline::IsDlssAvailable() ? "DLSS available" : "DLSS not available";
#else
    constexpr const char* dlss_status = "no DLSS in this build";
#endif
    LOG_INFO(Render_Vulkan,
             "Ocarina temporal: upscaler {}{} (DLSS setting {}, OoT anti-aliasing {}, {})",
             *oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss  ? "DLSS"
             : fsr4_wanted                                             ? "FSR 4 (else FSR 3.1)"
             : *oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr ? "FSR 3.1"
                                                                       : "the game's own",
             oot_fsr4_unjittered ? ", no camera jitter" : "",
             Settings::values.nvidia_dlss.GetValue() ? "on" : "off",
             Settings::CanonicalizeEnum(Settings::values.oot_anti_aliasing.GetValue()),
             dlss_status);
    if (*oot_backend == VideoCommon::OcarinaUpscaleBackend::Native &&
        VideoCommon::OcarinaFsrChosen(Settings::values.oot_anti_aliasing.GetValue()) &&
        !Fsr3Upscaler::IsSupported(device)) {
        LOG_WARNING(Render_Vulkan, "Ocarina FSR 3.1 unsupported on this device; using native upscale");
    }
    if (*oot_backend == VideoCommon::OcarinaUpscaleBackend::Native) {
        // No upscaler: the status bar names the resolution setting's size.
        VideoCommon::PublishOcarinaPicture(1, 0, false, 0);
    }
    if (*oot_backend != VideoCommon::OcarinaUpscaleBackend::Native) {
        // The output size for the session (oot_upscale_output), and why.
        u64 vram = 0;
        const auto output = OcarinaUpscaleOutput(device, &vram);
        oot_output_height = output.height;
        const bool max_resolution = Settings::OcarinaMaxResolutionOn();
        const bool fsr_game_size = OcarinaFsrOutputsGameSize(*oot_backend);
        // The game's scene: 1440p with Max Resolution, or what oot_render_height made it; the
        // game's dynamic size (1080p at most) without it.
        const auto render_height = Settings::AppliedOcarinaRenderHeight();
        const u32 input_height =
            max_resolution ? Settings::OcarinaRenderHeightFor(render_height) : 1080U;
        const auto plan = VideoCommon::PlanOcarinaPassOutput(
            output.height, max_resolution, Settings::OcarinaRenderHeightFor(render_height), false,
            fsr_game_size);
        const u32 shown = plan.big_height != 0   ? plan.big_height
                          : plan.small_height != 0 ? plan.small_height
                          : plan.native_aa         ? input_height
                                                   : 1440U;
        // The status bar: what runs, from what to what (DLSS Native keeps the game's size).
        const bool dlss_backend = *oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss;
        const u32 picture_backend =
            dlss_backend ? 4U
            : Settings::values.oot_anti_aliasing.GetValue() == Settings::OotAntiAliasing::Fsr4
                ? 3U
                : 2U;
        oot_picture_backend = picture_backend;
        oot_picture_input = input_height;
        oot_picture_up_to = !max_resolution;
        oot_picture_output = plan.native_aa ? input_height : shown;
        VideoCommon::PublishOcarinaPicture(picture_backend, input_height, !max_resolution,
                                           oot_picture_output);
        LOG_INFO(Render_Vulkan,
                 "Ocarina temporal: output {}x{} with Max Resolution {} (oot_upscale_output={}: "
                 "{}; {} MiB video memory{}{}){}",
                 VideoCommon::OcarinaOutputWidth(shown), shown, max_resolution ? "on" : "off",
                 Settings::values.oot_upscale_output.GetValue(), output.reason, vram >> 20,
                 plan.native_aa ? ", 1x anti-aliasing at the game's size, at most 1080p" : "",
                 output.height == 0 && fsr_game_size ? ", FRITCHII_FSR_OUTPUT" : "",
                 *oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss
                     ? "; DLSS Native keeps the scene's size"
                     : "");
        if (render_height) {
            LOG_INFO(Render_Vulkan,
                     "Ocarina temporal: the game draws its scene at {}x{} (oot_render_height={}, "
                     "built-in render patch), {} makes {}p",
                     VideoCommon::OcarinaOutputWidth(*render_height), *render_height,
                     Settings::values.oot_render_height.GetValue(),
                     VideoCommon::OcarinaBackendName(*oot_backend),
                     plan.native_aa ? input_height : shown);
        }
    }
    return *oot_backend != VideoCommon::OcarinaUpscaleBackend::Native;
}

void RasterizerVulkan::ApplyOcarinaResolution(u64 title_id) {
    if (oot_resolution_checked || title_id != 0x0400486024DEC000ULL) {
        return;
    }
    oot_resolution_checked = true;
    auto& resolution = Settings::values.resolution_info;
    // DLSS and FSR take the scene at the game's own size and refuse rescaled images, so with a
    // resolution scale other than 1x the game's own upscale ran instead (2026-11-11: "DLSS only
    // works at 1x"). They pick their own render size (the DLSS quality), so the scale goes to 1x
    // for this session; the texture cache reads this same info, and no image exists yet. Done at
    // the disk cache load, before the prepared pipelines: built at 2x they all missed the driver
    // cache (82k pipelines, 4.5 minutes at boot) and were thrown away at the switch to 1x.
    if (!resolution.active) {
        return;
    }
    const auto backend = ResolveOcarinaUpscaler(device);
    if (backend == VideoCommon::OcarinaUpscaleBackend::Native) {
        return;
    }
    // FRITCHII_DLSS_SCALED: the upscaler takes the scene at the player's scale. The scale is not
    // touched at all, so the prepared pipelines are built once, at that scale.
    if (VideoCommon::OcarinaScaledUpscale()) {
        LOG_INFO(Render_Vulkan,
                 "Ocarina temporal: resolution scale {}x kept ({} takes the scaled scene, "
                 "FRITCHII_DLSS_SCALED)",
                 resolution.up_factor,
                 backend == VideoCommon::OcarinaUpscaleBackend::Dlss ? "DLSS" : "FSR 3.1");
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Ocarina temporal: resolution scale {}x set to 1x for this session ({} does the "
             "upscaling)",
             resolution.up_factor,
             backend == VideoCommon::OcarinaUpscaleBackend::Dlss ? "DLSS" : "FSR 3.1");
    Settings::TranslateResolutionInfo(Settings::ResolutionSetup::Res1X, resolution);
}

/// Capture the two camera matrices from a draw targeting the scene color and depth. Terrain and
/// materials draw with several vertex shaders, so any draw into a scene-sized target qualifies;
/// the camera buffer's contents are checked below.
void RasterizerVulkan::TrackOcarinaCamera(const MemStage::TrackerRecord* record) {
    if (!OcarinaTemporalEnabled()) {
        return;
    }

    const auto& targets = texture_cache.GetRenderTargets();
    if (!targets.color_buffer_ids[0] || !targets.depth_buffer_id) {
        return;
    }
    const auto& color = texture_cache.GetImageView(targets.color_buffer_ids[0]);
    if (color.size.width < 640 || color.size.height < 360 ||
        color.format != VideoCore::Surface::PixelFormat::B10G11R11_FLOAT) {
        return;
    }
    // Reflections and half-size passes share the camera layout; the upscale reads the largest.
    if (oot_camera_sample_frame == render_trace_frame &&
        u64{color.size.width} * color.size.height <=
            u64{oot_motion.extent.width} * oot_motion.extent.height) {
        return;
    }

    // The furthest check this frame's scene draws reached without giving a camera, for the
    // fallback log (TryOcarinaUpscale).
    const auto miss = [this](u32 rank, const char* reason, std::array<f32, 4> values) {
        if (oot_camera_miss.frame != render_trace_frame || rank >= oot_camera_miss.rank) {
            oot_camera_miss = {render_trace_frame, rank, reason, values};
        }
    };
    // FRITCHII_OOT_CAMERA_GUEST: slot 3 as the game bound it and only the bytes the camera needs
    // (VideoCommon::OcarinaCameraReadBytes), else the whole block through the shader's view.
    const u32 camera_bytes = VideoCommon::OcarinaCameraReadBytes();
    // The camera block of vertex uniform slot 3, when bound that large and not GPU-modified.
    // `copied`: the bytes the packet's fill copied (step 7, the window open), else game memory.
    const auto sample = [&](DAddr address, const u8* copied) {
        std::array<f32, 0x230 / sizeof(f32)> camera{};
        if (copied != nullptr) {
            std::memcpy(camera.data(), copied, camera_bytes);
        } else {
            device_memory.ReadBlockUnsafe(address, camera.data(), camera_bytes);
        }
        constexpr size_t extent_offset = 0x120 / sizeof(f32);
        constexpr size_t projection_wz = 0xa8 / sizeof(f32);
        if (camera[extent_offset] != static_cast<f32>(color.size.width) ||
            camera[extent_offset + 1] != static_cast<f32>(color.size.height)) {
            miss(3, "no scene camera this frame: the camera's extent isn't the target's",
                 {camera[extent_offset], camera[extent_offset + 1],
                  static_cast<f32>(color.size.width), static_cast<f32>(color.size.height)});
            return;
        }
        if (camera[projection_wz] != -1.0f || camera[projection_wz + 1] != 0.0f) {
            miss(4, "no scene camera this frame: not a perspective projection",
                 {camera[projection_wz], camera[projection_wz + 1], 0.0f, 0.0f});
            return;
        }
        const std::span values{camera};
        oot_camera_sample_frame = render_trace_frame;
        oot_motion.camera_channel = gpu_memory;
        oot_motion.extent = {color.size.width, color.size.height};
        std::copy_n(values.begin() + 0x30 / sizeof(f32), 16, oot_motion.current.begin());
        std::copy_n(values.begin() + 0x180 / sizeof(f32), 16, oot_motion.previous.begin());
    };
    static_assert(MemStage::TrackerRecord::CameraBytes == VideoCommon::OcarinaCameraBlockBytes);
    // A slot 3 binding: too small, GPU-written, or read.
    const auto check = [&](DAddr address, u32 size, bool gpu_modified, const u8* copied) {
        if (size < camera_bytes) {
            miss(1, "no scene camera this frame: the camera buffer is bound smaller",
                 {static_cast<f32>(size), static_cast<f32>(camera_bytes), 0.0f, 0.0f});
            return;
        }
        if (gpu_modified) {
            miss(2, "no scene camera this frame: the camera buffer is GPU-written", {});
            return;
        }
        sample(address, copied);
    };
    if (record != nullptr) {
        // The draw packet's record (fable_memstage_6b_plan.md 5.1): the same answers, taken at
        // the end of the fill.
        if (record->lean_skipped != 0) [[unlikely]] {
            // FRITCHII_FILL_BRACKET_LEAN: R reads slot 0 here (must not be one left out).
            VideoCommon::FillCuts::NoteTrackerRead(record->lean_skipped, 1);
        }
        const MemStage::TrackerSlot& slot = record->slots[0];
        if (!slot.bound) {
            miss(0, "no scene camera this frame: no camera buffer bound", {});
            return;
        }
        check(slot.address, slot.size, slot.gpu_modified, mem_stage.TrackerBytes(*record, 0));
        return;
    }
    if (VideoCommon::OcarinaCameraFromGuest()) {
        const auto [address, size] = buffer_cache.GuestGraphicsUniform(0, 3);
        if (size == 0) {
            miss(0, "no scene camera this frame: no camera buffer bound", {});
            return;
        }
        check(address, size,
              size >= camera_bytes && buffer_cache.IsRegionGpuModified(address, camera_bytes),
              nullptr);
        return;
    }
    bool bound = false;
    buffer_cache.ForEachBoundBuffer(false, [&](u32 stage, bool storage, u32 index, bool,
                                                const auto& binding, u32 bound_size,
                                                const auto*, u32) {
        if (stage != 0 || storage || index != 3) {
            return;
        }
        bound = true;
        check(binding.device_addr, bound_size,
              bound_size >= camera_bytes &&
                  buffer_cache.IsRegionGpuModified(binding.device_addr, camera_bytes),
              nullptr);
    });
    if (!bound) {
        miss(0, "no scene camera this frame: no camera buffer bound", {});
    }
}

void RasterizerVulkan::UpdateOcarinaJitter() {
    std::array<f32, 2> want{};
    // Only while FSR replaced the previous frame's upscale: a frame the game upscales itself
    // would show the offset as a shake. Only draws that depth-test against the scene depth at
    // the scene size, so a depth pre-pass and the colour passes move together; shadow maps,
    // half-size passes and full-screen passes (which keep the depth bound without using it, as
    // a ported change found, some with depth writes on: with the test off nothing writes depth) stay put.
    const auto& regs = maxwell3d->regs;
    // FSR (FSR 4 without it unless FRITCHII_FSR4_JITTER: oot_fsr4_unjittered), and DLSS with
    // its jitter on (the "DLSS jitter" setting, on by default, or EDEN_NX2_DLSS_JITTER).
    const bool jittered_backend =
        (oot_backend == VideoCommon::OcarinaUpscaleBackend::Fsr && !oot_fsr4_unjittered) ||
        (oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss && oot_dlss_jitter);
    if (jittered_backend && oot_motion.extent.width != 0 &&
        oot_dlss_frame && *oot_dlss_frame + 1 >= render_trace_frame && regs.depth_test_enable) {
        const auto& targets = texture_cache.GetRenderTargets();
        if (targets.depth_buffer_id) {
            const auto& depth = texture_cache.GetImageView(targets.depth_buffer_id);
            // FRITCHII_DLSS_SCALED: a scaled scene too. The offset is added after the viewport
            // is scaled, so it stays within half a pixel of the scaled image, the unit the
            // upscaler reads it in.
            if (depth.size.width == oot_motion.extent.width &&
                depth.size.height == oot_motion.extent.height &&
                (!depth.IsRescaled() || VideoCommon::OcarinaScaledUpscale())) {
                // Halton (2, 3) with AMD's phase count for the last FSR frame's scale. On AMD's
                // drivers in [0, 1) (FRITCHII_JITTER_EDGE): they clip the viewport's fractional
                // right and bottom edges.
                const u32 index = static_cast<u32>(render_trace_frame % oot_jitter_phases) + 1;
                const auto driver = device.GetDriverID();
                const bool truncated_edges = VideoCommon::OcarinaJitterEdgeFix() &&
                                             (driver == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                              driver == VK_DRIVER_ID_AMD_OPEN_SOURCE);
                want = VideoCommon::OcarinaJitter(index, truncated_edges);
                oot_jitter_frame = render_trace_frame;
                oot_jitter_offset = want;
            }
        }
    }
    if (want != oot_jitter_applied) {
        oot_jitter_applied = want;
        maxwell3d->dirty.flags[Dirty::Viewports] = true;
    }
}

namespace {
constexpr u32 OCARINA_LINK_INDICES = 9702; ///< the index count of Link's recognized draw
constexpr size_t OCARINA_VERTEX_STRIDE = 48;
/// The palette bytes that tell skeletons apart: its first 16 bones. A false match only adds a
/// replay the scene depth check then drops.
constexpr size_t OCARINA_PALETTE_KEY_BYTES = 16 * 12 * sizeof(f32);
} // namespace

/// Capture the skinning pose of the recognized Link draw for frame-to-frame motion, and with
/// FRITCHII_OOT_ACTOR_PARTS (or _TRACE) note the other draws of the same vertex program.
void RasterizerVulkan::TrackOcarinaActor(u64 shader_hash, bool is_indexed, bool is_indirect,
                                         const MemStage::TrackerRecord* record) {
    if (shader_hash != MemStage::TrackerRecord::ActorShader || !is_indexed || is_indirect ||
        !OcarinaTemporalEnabled()) {
        return;
    }
    const bool parts = VideoCommon::OcarinaActorParts() || VideoCommon::OcarinaActorTrace();
    const bool pose_wanted = oot_actor.frame != render_trace_frame;
    if (!pose_wanted && !parts) {
        return;
    }
    const auto& draw = maxwell3d->draw_manager.draw_state;
    const auto& regs = maxwell3d->regs;
    const auto& stream = regs.vertex_streams[0];
    const bool link_draw = pose_wanted && draw.index_buffer.count == OCARINA_LINK_INDICES;
    if ((!link_draw && (!parts || draw.topology != Maxwell::PrimitiveTopology::Triangles)) ||
        draw.index_buffer.FormatSizeInBytes() != 2 || !stream.IsEnabled() ||
        stream.stride != OCARINA_VERTEX_STRIDE ||
        regs.vertex_stream_instances.IsInstancingEnabled(0)) {
        return;
    }
    const auto attribute = [&](u32 location, VkFormat format, u32 offset) {
        const auto& attrib = regs.vertex_attrib_format[location];
        return attrib.buffer == 0 && !attrib.constant && attrib.offset == offset &&
            MaxwellToVK::VertexFormat(device, attrib.type, attrib.size) == format;
    };
    if (!attribute(0, VK_FORMAT_R32G32B32_SFLOAT, 0) ||
        !attribute(4, VK_FORMAT_R8G8B8A8_UNORM, 40) || !attribute(5, VK_FORMAT_R8G8B8A8_UNORM, 44) ||
        !attribute(6, VK_FORMAT_R16G16B16A16_UINT, 12) || !attribute(7, VK_FORMAT_R16G16B16A16_UINT, 20)) {
        return;
    }

    std::array<f32, 4> origin{};
    bool origin_found{}, bones_found{};
    DAddr bones_address{};
    const u8* bones_copied{};
    // Vertex uniform slots 4 (the origin) and 6 (the bones), when not GPU-modified. `copied`: the
    // bytes the packet's fill copied (step 7, the window open), else game memory.
    using Tracker = MemStage::TrackerRecord;
    static_assert(Tracker::OriginBytes == 3 * sizeof(f32));
    static_assert(Tracker::BonesBytes == sizeof(OcarinaActorPose::bones));
    static_assert(OCARINA_PALETTE_KEY_BYTES <= Tracker::BonesBytes);
    const auto sample = [&](u32 index, DAddr address, u32 size, const u8* copied) {
        if (index == 4 && size >= Tracker::OriginOffset + Tracker::OriginBytes) {
            if (copied != nullptr) {
                std::memcpy(origin.data(), copied, Tracker::OriginBytes);
            } else {
                device_memory.ReadBlockUnsafe(address + Tracker::OriginOffset, origin.data(),
                                              Tracker::OriginBytes);
            }
            origin_found = true;
        }
        if (index == 6 && size == Tracker::BonesSlotSize) {
            bones_address = address;
            bones_copied = copied;
            bones_found = true;
        }
    };
    if (record != nullptr) {
        if (record->lean_skipped != 0) [[unlikely]] {
            // FRITCHII_FILL_BRACKET_LEAN: R reads slots 1 and 2 here.
            VideoCommon::FillCuts::NoteTrackerRead(record->lean_skipped, 6);
        }
        // The draw packet's record (fable_memstage_6b_plan.md 5.1).
        for (size_t i = 1; i < record->slots.size(); ++i) {
            const MemStage::TrackerSlot& slot = record->slots[i];
            if (slot.bound && !slot.gpu_modified) {
                sample(Tracker::Slots[i], slot.address, slot.size, mem_stage.TrackerBytes(*record, i));
            }
        }
    } else {
        buffer_cache.ForEachBoundBuffer(false, [&](u32 stage, bool storage, u32 index, bool,
                                                    const auto& binding, u32 size, const auto*,
                                                    u32) {
            if (stage != 0 || storage || (index != 4 && index != 6) ||
                buffer_cache.IsRegionGpuModified(binding.device_addr, size)) {
                return;
            }
            sample(index, binding.device_addr, size, nullptr);
        });
    }
    if (!origin_found || !bones_found) {
        return;
    }
    const auto read_bones = [&](void* destination, size_t bytes) {
        if (bones_copied != nullptr) {
            std::memcpy(destination, bones_copied, bytes);
        } else {
            device_memory.ReadBlockUnsafe(bones_address, destination, bytes);
        }
    };
    u64 palette{};
    if (parts) {
        std::array<char, OCARINA_PALETTE_KEY_BYTES> key;
        read_bones(key.data(), key.size());
        palette = Common::CityHash64(key.data(), key.size());
    }
    if (link_draw) {
        OcarinaActorPose pose{
            .frame = render_trace_frame,
            .channel = gpu_memory,
            .vertex = stream.Address(),
            .indices = draw.index_buffer.IndexStart(),
            .base_vertex = static_cast<s32>(draw.base_index),
            .origin = origin,
            .palette = palette,
        };
        read_bones(pose.bones.data(), sizeof(pose.bones));
        oot_previous_actor = oot_actor;
        oot_actor = pose;
        return;
    }
    if (oot_actor_parts_frame != render_trace_frame) {
        oot_actor_parts_frame = render_trace_frame;
        oot_actor_part_count = 0;
    }
    // Once Link's pose is known, other skeletons (the townsfolk) need no slot.
    if (oot_actor.frame == render_trace_frame && palette != oot_actor.palette) {
        return;
    }
    const OcarinaActorPart part{
        .channel = gpu_memory,
        .vertex = stream.Address(),
        .indices = draw.index_buffer.IndexStart(),
        .base_vertex = static_cast<s32>(draw.base_index),
        .index_count = draw.index_buffer.count,
        .palette = palette,
        .origin = origin,
    };
    const std::span known{oot_actor_parts.data(), oot_actor_part_count};
    const bool seen = std::ranges::any_of(known, [&](const OcarinaActorPart& other) {
        return other.vertex == part.vertex && other.indices == part.indices &&
               other.base_vertex == part.base_vertex && other.index_count == part.index_count;
    });
    if (!seen && oot_actor_part_count < oot_actor_parts.size()) {
        oot_actor_parts[oot_actor_part_count++] = part;
    }
}

void RasterizerVulkan::TraceOcarinaActorDraw(const GraphicsPipeline& pipeline, bool is_indexed) {
    if (render_trace_frame % 600 != 0 || !is_indexed || !OcarinaTemporalEnabled()) {
        return;
    }
    const auto& draw = maxwell3d->draw_manager.draw_state;
    const auto& regs = maxwell3d->regs;
    const auto& stream = regs.vertex_streams[0];
    const auto hashes = pipeline.ShaderHashes();
    if (hashes[1] != MemStage::TrackerRecord::ActorShader && stream.stride != OCARINA_VERTEX_STRIDE) {
        return;
    }
    if (oot_actor_trace_frame != render_trace_frame) {
        oot_actor_trace_frame = render_trace_frame;
        oot_actor_trace_lines = 0;
    }
    if (oot_actor_trace_lines >= 200) {
        return;
    }
    ++oot_actor_trace_lines;
    const auto& targets = texture_cache.GetRenderTargets();
    u32 width{}, height{};
    if (targets.color_buffer_ids[0]) {
        const auto& color = texture_cache.GetImageView(targets.color_buffer_ids[0]);
        width = color.size.width;
        height = color.size.height;
    }
    LOG_INFO(Render_Vulkan,
             "Ocarina actor trace: frame={} vs={:016x} fs={:016x} indices={} stride={} target={}x{} "
             "depth_test={} depth_write={} blend={} alpha_test={} alpha_to_coverage={} discard={}",
             render_trace_frame, hashes[1], hashes[5], draw.index_buffer.count, stream.stride, width,
             height, regs.depth_test_enable != 0, regs.depth_write_enabled != 0,
             regs.blend.enable[0] != 0, regs.alpha_test_enabled != 0,
             regs.anti_alias_alpha_control.alpha_to_coverage != 0, pipeline.FragmentCanDiscard());
}

/// FRITCHII_MEMSTAGE_SHADOW: the draw packet's record of uniform slots 3, 4 and 6 against what
/// the two trackers would ask the cache now (ForEachBoundBuffer, IsRegionGpuModified), slot by
/// slot: bound, address and size, and the GPU-modified answer over the bytes each tracker reads.
void RasterizerVulkan::VerifyTrackerRecord(const MemStage::TrackerRecord* record) {
    if (record == nullptr) {
        VideoCommon::MemStageShadow::BackTrackerRecord(false, 0);
        return;
    }
    MemStage::TrackerRecord today{};
    // FRITCHII_OOT_CAMERA_GUEST: slot 3 is recorded as the game bound it, with the camera's bytes.
    const bool camera_guest = VideoCommon::OcarinaCameraFromGuest();
    const u32 camera_bytes = VideoCommon::OcarinaCameraReadBytes();
    if (camera_guest) {
        const auto [address, size] = buffer_cache.GuestGraphicsUniform(0, 3);
        MemStage::TrackerSlot& slot = today.slots[0];
        slot.address = address;
        slot.size = size;
        slot.bound = size != 0;
        slot.gpu_modified = slot.bound && size >= camera_bytes &&
                            buffer_cache.IsRegionGpuModified(address, camera_bytes);
    }
    buffer_cache.ForEachBoundBuffer(false, [&](u32 stage, bool storage, u32 index, bool,
                                                const auto& binding, u32 size, const auto*, u32) {
        if (stage != 0 || storage || (camera_guest && index == 3)) {
            return;
        }
        for (size_t i = 0; i < today.slots.size(); ++i) {
            if (MemStage::TrackerRecord::Slots[i] != index) {
                continue;
            }
            MemStage::TrackerSlot& slot = today.slots[i];
            slot.address = binding.device_addr;
            slot.size = size;
            slot.bound = true;
            if (index == 3) {
                slot.gpu_modified = size >= camera_bytes &&
                                    buffer_cache.IsRegionGpuModified(binding.device_addr,
                                                                     camera_bytes);
            } else {
                slot.gpu_modified = buffer_cache.IsRegionGpuModified(binding.device_addr, size);
            }
        }
    });
    u32 mismatches = 0;
    for (size_t i = 0; i < today.slots.size(); ++i) {
        const MemStage::TrackerSlot& a = record->slots[i];
        const MemStage::TrackerSlot& b = today.slots[i];
        const bool same = a.bound == b.bound &&
                          (!a.bound || (a.address == b.address && a.size == b.size &&
                                        a.gpu_modified == b.gpu_modified));
        mismatches |= same ? 0U : (1U << i);
    }
    VideoCommon::MemStageShadow::BackTrackerRecord(true, mismatches);
}

/// Read the shared mesh only when both poses refer to the same vertex and index buffers.
void RasterizerVulkan::ReadOcarinaActor(OcarinaActorMesh& mesh) {
    const auto& now = oot_actor;
    const auto& before = oot_previous_actor;
    if (now.frame != render_trace_frame || before.frame + 1 != now.frame ||
        now.channel != gpu_memory || before.channel != now.channel || now.vertex != before.vertex ||
        now.indices != before.indices || now.base_vertex != before.base_vertex) {
        return;
    }
    const auto read = [&](GPUVAddr address, void* dst, size_t size) {
        if (!gpu_memory->IsContinuousRange(address, size) ||
            !gpu_memory->IsFullyMappedRange(address, size)) {
            return false;
        }
        const auto device_address = gpu_memory->GpuToCpuAddress(address);
        if (!device_address || buffer_cache.IsRegionGpuModified(*device_address, size) ||
            gpu.HasPendingCPUWrites(*device_address, size)) {
            return false;
        }
        gpu_memory->ReadBlockUnsafe(address, dst, size);
        return true;
    };
    constexpr size_t stride = OCARINA_VERTEX_STRIDE;
    // One draw's triangles and the vertices they use, added to the mesh.
    const auto append = [&](GPUVAddr vertex, GPUVAddr index_address, s32 base_vertex,
                            u32 index_count) {
        if (index_count == 0 || index_count > (1U << 18)) {
            return false;
        }
        std::vector<u16> indices(index_count);
        if (!read(index_address, indices.data(), indices.size() * sizeof(u16))) {
            return false;
        }
        const auto [minimum, maximum] = std::ranges::minmax(indices);
        const s64 first = static_cast<s64>(base_vertex) + minimum;
        const u32 vertex_count = static_cast<u32>(maximum) - minimum + 1;
        const size_t offset = mesh.vertices.size() / stride;
        // The replay's indices are 16 bits.
        if (first < 0 || first > (1 << 20) || vertex_count > 16384 ||
            offset + vertex_count > 65536) {
            return false;
        }
        mesh.vertices.resize((offset + vertex_count) * stride);
        if (!read(vertex + first * stride, mesh.vertices.data() + offset * stride,
                  vertex_count * stride)) {
            mesh.vertices.resize(offset * stride);
            return false;
        }
        mesh.indices.reserve(mesh.indices.size() + index_count);
        for (const u16 index : indices) {
            mesh.indices.push_back(static_cast<u16>(index - minimum + offset));
        }
        return true;
    };
    if (!append(now.vertex, now.indices, now.base_vertex, OCARINA_LINK_INDICES)) {
        return;
    }
    mesh.current_origin = now.origin;
    mesh.previous_origin = before.origin;
    mesh.current_bones = now.bones;
    mesh.previous_bones = before.bones;

    const bool add_parts = VideoCommon::OcarinaActorParts();
    const bool trace = VideoCommon::OcarinaActorTrace() && render_trace_frame % 600 == 0;
    if ((!add_parts && !trace) || oot_actor_parts_frame != render_trace_frame) {
        return;
    }
    u32 same_skeleton = 0;
    u32 added = 0;
    std::string counts;
    for (const OcarinaActorPart& part : std::span{oot_actor_parts.data(), oot_actor_part_count}) {
        if (part.channel != now.channel || part.palette != now.palette ||
            part.origin != now.origin ||
            (part.vertex == now.vertex && part.indices == now.indices &&
             part.base_vertex == now.base_vertex && part.index_count == OCARINA_LINK_INDICES)) {
            continue;
        }
        ++same_skeleton;
        if (trace) {
            counts += fmt::format(" {}", part.index_count);
        }
        if (add_parts && append(part.vertex, part.indices, part.base_vertex, part.index_count)) {
            ++added;
        }
    }
    if (trace) {
        LOG_INFO(Render_Vulkan,
                 "Ocarina actor trace: frame={} Link's draw has {} indices; {} other draws of the "
                 "program seen, {} with his skeleton (indices:{}), {} replayed; the replay has {} "
                 "vertices, {} indices",
                 render_trace_frame, OCARINA_LINK_INDICES, oot_actor_part_count, same_skeleton,
                 counts, added, mesh.vertices.size() / stride, mesh.indices.size());
    }
}

// Here and not in vk_compute_pipeline.cpp, which the Preparing shaders stamp hashes: the same test
// as the end of CanReplaceOcarinaUpscale.
bool ComputePipeline::HasOcarinaUpscaleShape() const {
    return !info.stores_global_memory && info.image_buffer_descriptors.empty() &&
           info.image_descriptors.size() == 1 && info.image_descriptors.front().count == 1 &&
           info.image_descriptors.front().is_written && !info.image_descriptors.front().is_read &&
           !info.image_descriptors.front().is_byte_addressed &&
           std::ranges::none_of(info.storage_buffers_descriptors,
                                &Shader::StorageBufferDescriptor::is_written);
}

void RasterizerVulkan::NoteLowSceneUpscale(ComputePipeline& pipeline, u32 height) {
    if (pipeline.LearnedOcarinaUpscale() || !OcarinaTemporalEnabled()) {
        return;
    }
    const auto written = pipeline.WrittenImages();
    if (written.size() != 1 || !written[0]) {
        return;
    }
    const auto& target = texture_cache.GetImageView(written[0]);
    if (target.size.width != 2560 || target.size.height != 1440 ||
        target.format != VideoCore::Surface::PixelFormat::B10G11R11_FLOAT) {
        return;
    }
    if (!pipeline.HasOcarinaUpscaleShape()) {
        static std::array<u64, 8> logged{};
        const u64 hash = pipeline.ShaderHash();
        if (std::ranges::find(logged, hash) == logged.end()) {
            if (const auto slot = std::ranges::find(logged, 0ULL); slot != logged.end()) {
                *slot = hash;
                LOG_INFO(Render_Vulkan,
                         "Ocarina temporal: pass {:016x} writes the game's 1440p target at render "
                         "height {}, but doesn't look like an upscale pass",
                         hash, height);
            }
        }
        return;
    }
    if (pipeline.LearnOcarinaUpscale()) {
        LOG_INFO(Render_Vulkan,
                 "Ocarina temporal: pass {:016x} upscales the {}p scene to the game's 1440p "
                 "target; {} takes it from its next dispatch",
                 pipeline.ShaderHash(), height, VideoCommon::OcarinaBackendName(*oot_backend));
    }
}

/// Track the scene depth on its reader dispatch, or try to replace a recognized upscale dispatch.
bool RasterizerVulkan::TryOcarinaUpscale(const ComputePipeline& pipeline, bool replace) {
    constexpr u64 depth_reader = 0x1fc7a0423116281fULL;
    const u64 upscale_hash = pipeline.ShaderHash();
    // oot_render_height: the height the game started with (the built-in render patch): its
    // scene is smaller than 2560x1440 and still upscaled to it, by the scaled 1440p passes
    // (1280x720) or the 3/4 pass (1920x1080).
    const auto render_height = Settings::AppliedOcarinaRenderHeight();
    const bool scaled_1440p_pass =
        render_height && (upscale_hash == ComputePipeline::Ocarina1440pScaledShader ||
                          upscale_hash == ComputePipeline::Ocarina1440pScaledFieldShader ||
                          pipeline.LearnedOcarinaUpscale() ||
                          (upscale_hash == ComputePipeline::OcarinaThreeQuarterShader &&
                           *render_height == 1080));
    const bool pass_1440p =
        replace && (upscale_hash == ComputePipeline::Ocarina1440pShader || scaled_1440p_pass);
    if ((!replace && upscale_hash != depth_reader) || !OcarinaTemporalEnabled()) {
        return false;
    }
    const bool native_resolution =
        oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss &&
        Settings::GetNvidiaDlssRuntimeSettings().quality == Settings::NvidiaDlssQuality::Native;
    // The DLSS quality changes in game (to or from DLAA): the status bar follows (2026-11-18: it
    // still said "up to 1080p -> 1440p" after a switch to DLAA 1080p).
    if (replace && oot_picture_backend == 4U && oot_picture_native != int{native_resolution}) {
        oot_picture_native = int{native_resolution};
        VideoCommon::PublishOcarinaPicture(oot_picture_backend, oot_picture_input,
                                           oot_picture_up_to,
                                           native_resolution ? oot_picture_input
                                                             : oot_picture_output);
    }
    // The output size: oot_upscale_output (Auto: 4K from the 1440p pass, the target's size from
    // the 3/4 pass; FRITCHII_FSR_OUTPUT may make FSR draw the game's own 1440p picture). Above
    // 1440p the output is composited with the game's interface when presented ("4K" below).
    const auto output_plan = VideoCommon::PlanOcarinaPassOutput(
        oot_output_height, pass_1440p, Settings::OcarinaRenderHeightFor(render_height),
        native_resolution, OcarinaFsrOutputsGameSize(*oot_backend));
    const bool upscale_4k = output_plan.big_height != 0;
    const VkExtent2D big_extent{VideoCommon::OcarinaOutputWidth(output_plan.big_height),
                                output_plan.big_height};
    const auto read_tic = [this](u32 index) -> std::optional<Tegra::Texture::TICEntry> {
        const auto& table = kepler_compute->regs.tic;
        using Entry = Tegra::Texture::TICEntry;
        const GPUVAddr address = table.Address() + u64{index} * sizeof(Entry);
        if (index > table.limit || !gpu_memory->IsFullyMappedRange(address, sizeof(Entry))) {
            return std::nullopt;
        }
        Entry entry{};
        gpu_memory->ReadBlockUnsafe(address, &entry, sizeof(entry));
        return entry;
    };
    if (!replace) {
        constexpr u32 depth_handle_offset = 0x128;
        const auto& qmd = kepler_compute->launch_description;
        const auto& cbuf = qmd.const_buffer_config[0];
        if (!(qmd.const_buffer_enable_mask & 1) || cbuf.size < depth_handle_offset + sizeof(u32) ||
            !gpu_memory->IsFullyMappedRange(cbuf.Address(), depth_handle_offset + sizeof(u32))) {
            return false;
        }
        const u32 index = gpu_memory->Read<u32>(cbuf.Address() + depth_handle_offset) & 0xfffff;
        const auto tic = read_tic(index);
        const bool scene_size = oot_camera_sample_frame != render_trace_frame ||
                                (tic && tic->Width() == oot_motion.extent.width &&
                                 tic->Height() == oot_motion.extent.height);
        if (tic && tic->Address() && scene_size) {
            oot_motion.depth_channel = gpu_memory;
            oot_motion.depth_frame = render_trace_frame;
            oot_motion.depth_tic = index;
            oot_motion.depth_address = tic->Address();
        }
        return false;
    }
    // Logs why the game's own upscale draws: when a stretch of such frames starts or its reason
    // changes, at most once a second (VideoCommon::UpscalerFallbackLog; the end is logged below).
    const auto fail = [this](const char* reason, std::string detail = {}) {
        if (const auto start = oot_fallback_log.Fallback(
                render_trace_frame, std::chrono::steady_clock::now(), reason)) {
            LOG_INFO(Render_Vulkan, "Ocarina temporal: the game's upscale draws frame {} ({}{}){}",
                     render_trace_frame, reason, detail,
                     start->unlogged == 0
                         ? std::string{}
                         : fmt::format(", {} more stretches since the last line",
                                       start->unlogged));
        }
        return false;
    };
    if (oot_camera_sample_frame != render_trace_frame) {
        const auto& miss = oot_camera_miss;
        if (miss.frame != render_trace_frame) {
            return fail("no scene camera this frame: no scene-sized B10G11R11 draw with depth");
        }
        switch (miss.rank) {
        case 1:
            return fail(miss.reason, fmt::format(", {} bytes, {} needed", miss.values[0],
                                                 miss.values[1]));
        case 3:
            return fail(miss.reason, fmt::format(", {}x{} for a {}x{} target", miss.values[0],
                                                 miss.values[1], miss.values[2], miss.values[3]));
        case 4:
            return fail(miss.reason, fmt::format(", ({}, {})", miss.values[0], miss.values[1]));
        default:
            return fail(miss.reason);
        }
    }
    if (oot_motion.depth_frame != render_trace_frame) {
        return fail("no scene depth this frame");
    }
    if (oot_motion.camera_channel != gpu_memory || oot_motion.depth_channel != gpu_memory) {
        return fail("camera and depth from another channel");
    }
    const auto tic = read_tic(oot_motion.depth_tic);
    if (!tic || tic->Address() != oot_motion.depth_address ||
        tic->Width() != oot_motion.extent.width || tic->Height() != oot_motion.extent.height) {
        return fail("scene depth descriptor changed");
    }

    texture_cache.SynchronizeDescriptors(true);
    std::array images{VideoCommon::ImageViewInOut{.index = oot_motion.depth_tic}};
    texture_cache.FillImageViews(images, true);
    if (!images[0].id) {
        return fail("no scene depth image");
    }
    auto& depth = texture_cache.GetImageView(images[0].id);
    const auto* backing = depth.BackingImage();
    // FRITCHII_DLSS_SCALED: scaled images are taken at their real size (checked below). Off, the
    // scale is 1x for the session and nothing is scaled.
    const bool scaled_mode = VideoCommon::OcarinaScaledUpscale();
    if (!backing || (depth.IsRescaled() && !scaled_mode) ||
        depth.format != VideoCore::Surface::PixelFormat::D32_FLOAT ||
        depth.range != VideoCommon::SubresourceRange{} || backing->info.num_samples != 1) {
        return fail("unsupported scene depth");
    }

    const auto sampled = pipeline.SampledImages();
    const auto written = pipeline.WrittenImages();
    if (sampled.size() != 1 || written.size() != 1 || !sampled[0].id || !written[0]) {
        return fail("unexpected upscale bindings");
    }
    const auto& color = texture_cache.GetImageView(sampled[0].id);
    auto& target = texture_cache.GetImageView(written[0]);
    using VideoCore::Surface::PixelFormat;
    const auto extent = oot_motion.extent;
    const bool native_size = upscale_hash == ComputePipeline::OcarinaFullSizeShader;
    // The render-height passes take the 16:9 scene the patch gave (1920x1080, 1280x720, ...).
    const u32 scene_height = Settings::OcarinaRenderHeightFor(render_height);
    const bool scene_1440p_or_scaled =
        scaled_1440p_pass ? extent.height == scene_height && extent.width == scene_height * 16 / 9
                          : extent.width == 2560 && extent.height == 1440;
    const bool valid_extents =
        pass_1440p ? scene_1440p_or_scaled && target.size.width == 2560 &&
                         target.size.height == 1440
        : native_size ? target.size.width == extent.width && target.size.height == extent.height
                      : target.size.width == extent.width * 4 / 3 &&
                            target.size.height == extent.height * 4 / 3;
    if (!valid_extents || !color.BackingImage() || !target.BackingImage() ||
        ((color.IsRescaled() || target.IsRescaled()) && !scaled_mode) ||
        color.Samples() != VK_SAMPLE_COUNT_1_BIT || target.Samples() != VK_SAMPLE_COUNT_1_BIT ||
        color.range != VideoCommon::SubresourceRange{} || target.range != VideoCommon::SubresourceRange{} ||
        color.format != PixelFormat::E5B9G9R9_FLOAT || target.format != PixelFormat::B10G11R11_FLOAT ||
        color.size.width != extent.width || color.size.height != extent.height ||
        color.ImageHandle() == target.ImageHandle() || depth.ImageHandle() == target.ImageHandle()) {
        return fail("unexpected upscale images");
    }

    // The sizes the upscaler works with. At 1x (and with FRITCHII_DLSS_SCALED off) these are the
    // game's sizes.
    VkExtent2D render_extent = extent;
    VkExtent2D target_extent{target.size.width, target.size.height};
    // A small output (720p -> 1080p) is stretched into the game's 1440p target by the output
    // pass (OcarinaUpscaler::Record samples it linearly when the sizes differ).
    const VkExtent2D small_extent{VideoCommon::OcarinaOutputWidth(output_plan.small_height),
                                  output_plan.small_height};
    VkExtent2D output_extent = output_plan.native_aa         ? extent
                               : upscale_4k                  ? big_extent
                               : output_plan.small_height != 0 ? small_extent
                                                             : target_extent;
    bool use_4k = upscale_4k;
    if (scaled_mode) {
        // The scene's colour, depth and the motion made from that depth must share one size;
        // the target may differ (a compute write keeps it at the game's size).
        const VkExtent2D color_real = RealExtent(color);
        const VkExtent2D depth_real = RealExtent(depth);
        target_extent = RealExtent(target);
        if (color_real.width != depth_real.width || color_real.height != depth_real.height) {
            const std::array<u32, 8> mismatch{color_real.width, color_real.height,
                                              depth_real.width, depth_real.height,
                                              target_extent.width, target_extent.height, 0, 0};
            if (std::exchange(oot_scaled_mismatch_logged, mismatch) != mismatch) {
                LOG_WARNING(Render_Vulkan,
                            "Ocarina temporal (FRITCHII_DLSS_SCALED): scene colour {}x{} ({}) "
                            "and depth {}x{} ({}) differ in size; the game's upscale draws",
                            color_real.width, color_real.height,
                            color.IsRescaled() ? "scaled" : "game size", depth_real.width,
                            depth_real.height, depth.IsRescaled() ? "scaled" : "game size");
            }
            return fail("scaled colour and depth sizes differ");
        }
        // The 4K output reads the target back at 2560x1440 (SaveSceneReference).
        const auto choice = VideoCommon::SelectOcarinaScaledOutput(
            {color_real.width, color_real.height}, {target_extent.width, target_extent.height},
            output_plan.native_aa, upscale_4k && !target.IsRescaled(),
            {big_extent.width, big_extent.height});
        render_extent = color_real;
        output_extent = {choice.output.width, choice.output.height};
        use_4k = choice.upscale_4k;
        const std::array<u32, 8> sizes{render_extent.width, render_extent.height,
                                       depth_real.width,    depth_real.height,
                                       target_extent.width, target_extent.height,
                                       output_extent.width, output_extent.height};
        if (std::exchange(oot_scaled_logged, sizes) != sizes) {
            LOG_INFO(Render_Vulkan,
                     "Ocarina temporal (FRITCHII_DLSS_SCALED, scale {}x): scene {}x{} ({}), "
                     "game size {}x{}; target {}x{} ({}); upscaler {}x{} -> {}x{}{}",
                     Settings::values.resolution_info.up_factor, render_extent.width,
                     render_extent.height, color.IsRescaled() ? "scaled" : "game size",
                     extent.width, extent.height, target_extent.width, target_extent.height,
                     target.IsRescaled() ? "scaled" : "game size", render_extent.width,
                     render_extent.height, output_extent.width, output_extent.height,
                     use_4k ? ", 4K output"
                     : output_extent.width > target_extent.width ? ", shrunk into the target"
                                                                 : "");
        }
    }

    constexpr VkFormat output_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    const auto color_view = color.Handle(Shader::TextureType::Color2D);
    if (!color_view || !device.IsStorageImageExtendedFormatsSupported() ||
        !device.IsFormatSupported(output_format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, FormatType::Optimal) ||
        (target.BackingImage()->UsageFlags() & VK_IMAGE_USAGE_STORAGE_BIT) == 0 ||
        MaxwellToVK::SurfaceFormat(device, FormatType::Optimal, true, target.format).format != output_format) {
        return fail("unsupported storage format");
    }
    // Without Link's pose the mesh stays empty and the motion comes from the camera alone.
    OcarinaActorMesh mesh{};
    ReadOcarinaActor(mesh);
    try {
        const auto target_view =
            target.StorageView(Shader::TextureType::Color2D, Shader::ImageFormat::Typeless, false);
        const auto scene_reference = use_4k ? target.Handle(Shader::TextureType::Color2D)
                                            : VK_NULL_HANDLE;
        if (!target_view || (use_4k && !scene_reference)) {
            return fail("no upscale target view");
        }
        if (!oot_live_dlss) {
            oot_live_dlss = std::make_unique<OcarinaUpscaler>(
                device, scheduler, descriptor_pool, compute_pass_descriptor_queue, memory_allocator,
                staging_pool, *oot_backend);
        }
        // The offset this frame's scene was drawn with (none when no scene draw was offset).
        std::array<f32, 2> jitter{};
        if (oot_jitter_frame == render_trace_frame) {
            jitter = oot_jitter_offset;
        }
        const bool recorded = oot_live_dlss->Record(color_view, depth.DepthView(), target_view,
                                                   render_extent, output_extent, target_extent, oot_motion.current,
                                                   oot_motion.previous, mesh, depth.ImageHandle(),
                                                   render_trace_frame, jitter);
        if (!recorded) {
            return fail("temporal upscaler declined the frame");
        }
        if (use_4k) {
            oot_live_dlss->SaveSceneReference(scene_reference);
            oot_4k_scene_frame = render_trace_frame;
            oot_4k_guest_present = VK_NULL_HANDLE;
        }
        oot_dlss_frame = render_trace_frame;
        // AMD's jitter phase count for FSR: 8 * (output / render width)^2, so 18 for 1440p -> 4K
        // and 8 for native AA.
        const f32 ratio =
            static_cast<f32>(output_extent.width) / static_cast<f32>(render_extent.width);
        oot_jitter_phases = std::max(1U, static_cast<u32>(8.0f * ratio * ratio));
        if (const auto end =
                oot_fallback_log.Replaced(render_trace_frame, std::chrono::steady_clock::now())) {
            LOG_INFO(Render_Vulkan,
                     "Ocarina temporal: {} replaces the game's upscale again from frame {}, after "
                     "{} frames ({:.1f} s)",
                     VideoCommon::OcarinaBackendName(*oot_backend), render_trace_frame,
                     end->frames, end->seconds);
        }
        return true;
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "Ocarina temporal upscaling unavailable: {}", error.what());
        oot_live_failed = true;
        texture_cache_runtime.ocarina_fsr_failed = true; // samplers drop the FSR LOD bias
        return false;
    }
}

bool RasterizerVulkan::FrameDumpBuffers(bool compute, std::span<const u64> shader_hashes) {
    static constexpr u32 BUFFER_CAP = 4U << 20;
    std::vector<FrameDump::BoundBuffer> buffers;
    buffer_cache.ForEachBoundBuffer(compute, [&](u32 stage, bool storage, u32 index, bool written,
                                                 const auto& binding, u32 bound_size,
                                                 const auto* host, u32 host_offset) {
        const u32 size = std::min(bound_size, BUFFER_CAP);
        const bool gpu_modified = buffer_cache.IsRegionGpuModified(binding.device_addr, size);
        FrameDump::BoundBuffer& buffer = buffers.emplace_back(FrameDump::BoundBuffer{
            .stage = stage,
            .storage = storage,
            .index = index,
            .written = written,
            .gpu_addr = binding.gpu_addr,
            .device_addr = binding.device_addr,
            .bound_size = bound_size,
            .size = size,
            .gpu_modified = gpu_modified,
            .host = host ? host->Handle() : VK_NULL_HANDLE,
            .host_offset = host_offset,
            .guest = {},
        });
        if (!gpu_modified || !host) {
            buffer.guest.resize(size);
            device_memory.ReadBlockUnsafe(binding.device_addr, buffer.guest.data(), size);
        }
        // EDEN_NX2_FRAME_DUMP_HOST: the host copy of guest-sourced ranges read from the buffer
        // cache (storage, and uniforms too large for the stream path), to find stale uploads.
        static const bool host_too = std::getenv("EDEN_NX2_FRAME_DUMP_HOST") != nullptr;
        buffer.also_host = (host_too || frame_dump.HostCopies()) && host && !gpu_modified &&
                           (storage || bound_size > 4096);
    });
    return frame_dump.NoteBuffers(compute, shader_hashes, buffers);
}

void RasterizerVulkan::FrameDumpPass(const Framebuffer* framebuffer) {
    std::array<ImageView*, VideoCommon::NUM_RT> colors{};
    const auto& ids = texture_cache.GetRenderTargets().color_buffer_ids;
    for (size_t slot = 0; slot < ids.size(); ++slot) {
        if (ids[slot]) {
            colors[slot] = &texture_cache.GetImageView(ids[slot]);
        }
    }
    bool recorded = false;
    if (frame_dump.Capturing()) {
        recorded |= frame_dump.BeginPass(framebuffer, colors);
    }
    if (flash_watch.WatchesPasses()) {
        recorded |= flash_watch.BeginPass(framebuffer, colors);
    }
    if (recorded) {
        scheduler.RequestRenderpass(framebuffer);
    }
}

void NoteDrawReuseProbeFrame(); // vk_graphics_pipeline.cpp

namespace {
/// FRITCHII_FRAME_END_TIMES=1 (boot, default 0; helper DE, a dry timer that changes nothing):
/// where TickFrame's time goes, by part, with the thread that ran it (R in one follower's lend,
/// while E is parked: the "TickFrame (frame end)" lend, 5 ms/s at Kakariko, 7-31 ms/s at the
/// market). The E-owned part (buffer cache) is what a posted frame end would still run in order
/// on E; the rest is the posting ceiling. One line every 10 s.
class FrameEndTimes {
public:
    enum Part : size_t {
        Head,     ///< focus, stats, debug requests, frame dump, upscaler trace, protect sweep
        Pipeline, ///< pipeline_cache.TickFrame
        Queues,   ///< descriptor queues and ring
        Fences,   ///< fence_manager.TickFrame
        Staging,  ///< staging pools
        Textures, ///< texture cache lock + TickFrame
        MemStage, ///< mem_stage.TickFrame
        Buffers,  ///< buffer cache lock + TickFrame (E's cache)
        Count,
    };

    static bool On() {
        static const bool on = [] {
            const char* const value = std::getenv("FRITCHII_FRAME_END_TIMES");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        return on;
    }

    void Begin() {
        mark = Now();
        begin = mark;
    }

    void End(Part part) {
        const u64 now = Now();
        ns[part] += now - mark;
        mark = now;
    }

    void Finish(bool on_render_thread) {
        const u64 total = mark - begin;
        ++calls;
        calls_on_r += on_render_thread ? 1 : 0;
        total_ns += total;
        longest_ns = (std::max)(longest_ns, total);
        if (window_start == 0) {
            window_start = mark;
            return;
        }
        const u64 span = mark - window_start;
        if (span < 10'000'000'000ULL) {
            return;
        }
        static constexpr std::array<const char*, Count> names{
            "head", "pipeline cache", "descriptor queues", "fence manager", "staging",
            "texture cache", "memory seam", "buffer cache (E's)"};
        const double seconds = static_cast<double>(span) / 1e9;
        std::string parts;
        for (size_t i = 0; i < Count; ++i) {
            parts += fmt::format("{}{} {:.2f}", i == 0 ? "" : ", ", names[i],
                                 static_cast<double>(ns[i]) / 1e6 / seconds);
        }
        LOG_INFO(Render_Vulkan,
                 "Frame end times (FRITCHII_FRAME_END_TIMES=1) 10 s: {:.1f} calls/s ({:.0f}% on "
                 "GPURender), {:.2f} ms/s, {:.0f} us each, longest {:.0f} us; by part ms/s: {}",
                 static_cast<double>(calls) / seconds,
                 calls != 0 ? 100.0 * static_cast<double>(calls_on_r) / calls : 0.0,
                 static_cast<double>(total_ns) / 1e6 / seconds,
                 calls != 0 ? static_cast<double>(total_ns) / 1e3 / calls : 0.0,
                 static_cast<double>(longest_ns) / 1e3, parts);
        ns = {};
        calls = calls_on_r = total_ns = longest_ns = 0;
        window_start = mark;
    }

private:
    static u64 Now() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }

    std::array<u64, Count> ns{};
    u64 mark{};
    u64 begin{};
    u64 window_start{};
    u64 calls{};
    u64 calls_on_r{};
    u64 total_ns{};
    u64 longest_ns{};
};
FrameEndTimes g_frame_end_times; ///< only TickFrame touches it (one thread at a time: R or E)
} // Anonymous namespace

void RasterizerVulkan::TickFrame() {
    ONEF_LEND(FrameEnd, TickFrame())
    // FRITCHII_ONEF_TICK_ON_E: on R outside a lent call E runs meanwhile: E ticks it (lent).
    if (VideoCommon::OneFollower::On() && VideoCommon::OneFollower::OnRenderThread() &&
        !VideoCommon::OneFollower::RenderInLentCall()) [[unlikely]] {
        static const bool tick_on_e = [] {
            const char* value = std::getenv("FRITCHII_ONEF_TICK_ON_E");
            return value == nullptr || std::string_view{value} != "0";
        }();
        if (tick_on_e) {
            onef_frame_tick_deferred.store(true, std::memory_order_release);
            return;
        }
    }
    async_compute_skip.TickFrame(); // FRITCHII_ASYNC_COMPUTE_SKIP's frame clock and 10 s line
    const bool frame_end_times = FrameEndTimes::On();
    if (frame_end_times) [[unlikely]] {
        g_frame_end_times.Begin();
    }
#ifdef _WIN32
    // EXP111 (live; 0 = Windows' default, 1 = above normal, 2 = high): the process priority
    // class. The owner sees more fps with the window focused: Windows boosts the foreground
    // process (longer quantum, priority boost) and the game's threads lose it when another
    // window has focus (2026-11-12). Applied when the value changes.
    {
        static int applied = 0;
        const int wanted = VideoCore::Nx2Debug::Experiment(111);
        if (wanted != applied) {
            const DWORD priority_class = wanted >= 2   ? HIGH_PRIORITY_CLASS
                                         : wanted == 1 ? ABOVE_NORMAL_PRIORITY_CLASS
                                                       : NORMAL_PRIORITY_CLASS;
            if (SetPriorityClass(GetCurrentProcess(), priority_class)) {
                LOG_INFO(Render_Vulkan, "Process priority class (EXP111={}) set", wanted);
            }
            applied = wanted;
        }
    }
    // FRITCHII_GPU_PRIORITY, FRITCHII_THREAD_QOS, FRITCHII_FOCUS_GRAB and the "Focus:" line.
    VideoCommon::FritchiiFocus::Tick();
#endif
    NoteDrawReuseProbeFrame();
    // The OoT trackers read the draw packet's record (fable_memstage_6b_plan.md 5.1): the fill
    // adds it while they are on (and draws have packets), published here once a frame.
    mem_stage.SetTrackerRecord(mem_stage.Packets() && OcarinaTemporalEnabled());
    // FRITCHII_MEMSTAGE_THREAD = 2: draws the stage must not fill (a diagnostic reads the bound
    // buffers on the back; EDEN_NX2_INVALIDATE_BEFORE_DISPATCH gathers before every draw).
    VideoCommon::MemStageShadow::g_ring_level_draws.store(
        frame_dump.Capturing() || VideoCommon::RenderTrace::Enabled() || InvalidateBeforeUse(),
        std::memory_order_relaxed);
    if (VideoCommon::FritchiiDrawKinds::g_on) [[unlikely]] {
        VideoCommon::FritchiiDrawKinds::g_frames.fetch_add(1, std::memory_order_relaxed);
    }
    VideoCommon::MemStageStats::TickFrame();
    VideoCommon::BufferChurnStats::TickFrame();
    if (VideoCore::Nx2Debug::Experiment(VideoCommon::FritchiiReportRuns::Switch) != 0)
        [[unlikely]] {
        VideoCommon::FritchiiReportRuns::g_frames.fetch_add(1, std::memory_order_relaxed);
    }
    if (VideoCommon::Nx2FastDrawCount::g_on) [[unlikely]] {
        if (VideoCommon::Nx2FastDrawCount::g_count) {
            fast_draw_count.MaybeLog();
        }
        VideoCommon::FritchiiDeltaPath::g_frames.fetch_add(1, std::memory_order_relaxed);
        VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::Frame);
    }
    if (MemoryOwnersEnabled()) {
        static std::chrono::steady_clock::time_point last_owners_log{};
        if (const auto now = std::chrono::steady_clock::now();
            now - last_owners_log >= std::chrono::seconds{10}) {
            last_owners_log = now;
            LogMemoryOwners(device, pipeline_cache, staging_pool, split_buffer_staging.get());
        }
    }
    if (VideoCore::Nx2Debug::HasRequests()) {
        if (auto capture = VideoCore::Nx2Debug::TakeCapture()) {
            // Mid-game through the frame dump (no boot switch needed); "buffers" records every
            // draw's bound buffers with their host copies, to tell stale data from a wrong read.
            if (!frame_dump.RequestNow(capture->id, capture->directory, capture->buffers,
                                       capture->buffers)) {
                VideoCore::Nx2Debug::FinishCapture(
                    capture->id, {.error = "a frame dump capture is already running"});
            }
        }
        if (const auto change = VideoCore::Nx2Debug::TakeRejectChange()) {
            pipeline_cache.RefreshDebugRejections(change->unrejected);
            VideoCore::Nx2Debug::NoteRejectApplied(change->generation);
        }
        while (const auto read = VideoCore::Nx2Debug::TakeGpuRead()) {
            std::vector<Tegra::MemoryManager*> spaces;
            if (gpu_memory != nullptr) {
                spaces.push_back(gpu_memory);
            }
            {
                std::scoped_lock lock{config_mutex};
                for (const auto& [id, space] : address_spaces) {
                    if (space.gpu_memory != gpu_memory) {
                        spaces.push_back(space.gpu_memory);
                    }
                }
            }
            VideoCore::Nx2Debug::FinishGpuRead(read->id,
                                               ReadGpuMemory(spaces, read->gpu_address, read->size));
        }
    }
    frame_dump.TickFrame();
    flash_watch.TickFrame();
    ++render_trace_frame;
    render_trace_draws = 0;
    render_trace_clears = 0;
    if (Common::DkbFrameTrace::Enabled()) {
        CollectIndirectTrace();
        const auto now = std::chrono::steady_clock::now();
        if (trace_cpu_rate.Ready(now)) {
            auto snapshot = Common::DkbCpuTrace::ReadSnapshot();
            if (trace_cpu_snapshot && trace_frame > trace_cpu_frame) {
                const u64 frames = trace_frame - trace_cpu_frame;
                std::string roles;
                for (size_t i = 0; i < Common::DkbCpuTrace::RoleCount; ++i) {
                    const auto name = Common::DkbCpuTrace::RoleNames[i];
                    const auto delta = Common::DkbCpuTrace::CpuTimeDelta(
                        trace_cpu_snapshot->threads[i], snapshot.threads[i]);
                    if (delta) {
                        const double cpu_ms = static_cast<double>(*delta) / 1e6;
                        roles += fmt::format(" {}_cpu_ms={:.3f} {}_cpu_ms_per_frame={:.3f}",
                                             name, cpu_ms, name, cpu_ms / frames);
                    } else {
                        roles += fmt::format(" {}={}", name,
                                             snapshot.threads[i].active ? "unavailable" : "inactive");
                    }
                }
                LOG_INFO(Render_Vulkan,
                         "DKB cpu through_frame={} frames={} window_wall_ms={:.3f}{}",
                         trace_frame, frames,
                         std::chrono::duration<double, std::milli>(
                             snapshot.wall_time - trace_cpu_snapshot->wall_time).count(),
                         roles);
            }
            trace_cpu_snapshot = std::move(snapshot);
            trace_cpu_frame = trace_frame;
        }
        ++trace_window_frames;
        if (trace_sample_frame) {
            LOG_INFO(Render_Vulkan,
                     "DKB draws through_frame={} frames={} attempts={} skipped={} indirect={} "
                     "zero_direct={} capture_dropped={} attempts_per_frame={:.1f} "
                     "zero_direct_per_frame={:.1f} (totals since previous line)",
                     trace_frame, trace_window_frames, trace_draws, trace_skipped, trace_indirect,
                     trace_zero_direct, trace_capture_dropped,
                     static_cast<double>(trace_draws) / trace_window_frames,
                     static_cast<double>(trace_zero_direct) / trace_window_frames);
            trace_draws = trace_skipped = trace_indirect = trace_zero_direct = 0;
            trace_window_frames = 0;
            trace_capture_dropped = trace_failure_details = 0;
            trace_rate.next = now + std::chrono::seconds{1};
        }
        ++trace_frame;
        trace_captures = 0;
        trace_sample_frame = trace_frame < 4 || trace_rate.Ready(now);
    }
    ComputePipeline::TickUpscalerTrace();
    draw_counter = 0;
    pending_native_work = false;
    // EDEN_NX2_PROTECT_STATS' frame count, then EXP89's sweep of lingering protections (takes
    // the device memory's counter_guard: no cache lock may be held here).
    Core::Nx2ProtectStats::TickFrame();
    // EXP92 (fable52_answer_uniformkeeps 3.3): a watched page the guest stored to more than 64
    // times this frame is denied (its watch given back, never kept again). Before EXP89's sweep,
    // which then releases such a page if it lingers. Nothing while no page is watched.
    Common::UniformKeepPages::TickFrame(64, [this](DAddr page_address, s32 delta) {
        device_memory.UpdatePagesCachedCount(page_address, Core::DEVICE_PAGESIZE, delta);
    });
    device_memory.ReleaseLingering();
    // FRITCHII_FRAME_END_TIMES: the parts' boundaries (nothing when off).
    const auto part_done = [frame_end_times](FrameEndTimes::Part part) {
        if (frame_end_times) [[unlikely]] {
            g_frame_end_times.End(part);
        }
    };
    part_done(FrameEndTimes::Head);
    pipeline_cache.TickFrame();
    part_done(FrameEndTimes::Pipeline);
    guest_descriptor_queue.TickFrame(scheduler);
    compute_pass_descriptor_queue.TickFrame(scheduler);
    descriptor_buffer_ring.TickFrame();
    part_done(FrameEndTimes::Queues);
    fence_manager.TickFrame();
    part_done(FrameEndTimes::Fences);
    staging_pool.TickFrame();
    if (split_buffer_staging) {
        split_buffer_staging->TickFrame();
    }
    VideoCommon::LinkStats::MaybeLog();
    part_done(FrameEndTimes::Staging);
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    part_done(FrameEndTimes::Textures);
    // SEAM (EXP123) is read here, once a frame, with nothing pending.
    mem_stage.TickFrame();
    part_done(FrameEndTimes::MemStage);
    {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::Frame};
        buffer_cache.TickFrame();
    }
    if (frame_end_times) [[unlikely]] {
        g_frame_end_times.End(FrameEndTimes::Buffers);
        g_frame_end_times.Finish(VideoCommon::OneFollower::OnRenderThread());
    }
}

bool RasterizerVulkan::MayEndConditionalRendering() const noexcept {
    return onef_condition_may_be_set || !VideoCommon::OneFollower::ShouldLend() ||
           !VideoCommon::OneFollower::QueryOps();
}

bool RasterizerVulkan::AccelerateConditionalRendering() {
    if (VideoCommon::OneFollower::ShouldLend()) [[unlikely]] {
        namespace OneF = VideoCommon::OneFollower;
        using RenderEnable = Maxwell::RenderEnable;
        // E's engine (the rasterizer's own may point at R's view).
        const auto& regs = channel_state->maxwell3d.regs;
        OneF::QueryKind kind = OneF::QueryKind::ConditionOverride;
        if (regs.render_enable_override == RenderEnable::Override::UseRenderEnable) {
            switch (regs.render_enable.mode) {
            case RenderEnable::Mode::True:
                kind = OneF::QueryKind::ConditionTrue;
                break;
            case RenderEnable::Mode::False:
                kind = OneF::QueryKind::ConditionFalse;
                break;
            case RenderEnable::Mode::Conditional:
                kind = OneF::QueryKind::ConditionConditional;
                break;
            case RenderEnable::Mode::IfEqual:
                kind = OneF::QueryKind::ConditionIfEqual;
                break;
            default:
                kind = OneF::QueryKind::ConditionIfNotEqual;
                break;
            }
        }
        const bool may_begin = kind == OneF::QueryKind::ConditionConditional ||
                               kind == OneF::QueryKind::ConditionIfEqual ||
                               kind == OneF::QueryKind::ConditionIfNotEqual;
        if (may_begin) {
            // Set with the switch off too, so turning it on later stays exact.
            onef_condition_may_be_set = true;
        } else if (!onef_condition_may_be_set && OneF::QueryOps()) {
            // fable_b 11.1 item 1(a): override or True/False only ends host conditional
            // rendering (false, no lookup), a no-op while nothing ever began it. The call's
            // FlushCaching runs here, at E's place, as without one follower: clears, copies and
            // dispatches before the next draw see the stream's writes invalidated (lent only
            // when there are ranges, which InnerInvalidation lends).
            gpu_memory->FlushCaching();
            OneF::CountQueryKind(OneF::QueryKind::ConditionSkipped);
            return false;
        }
        OneF::SetQueryKind(kind);
    }
    ONEF_LEND_RESULT(Query, AccelerateConditionalRendering())
    gpu_memory->FlushCaching();
    return query_cache.AccelerateHostConditionalRendering();
}

bool RasterizerVulkan::HasDrawTransformFeedback() {
    return device.IsTransformFeedbackDrawSupported();
}

bool RasterizerVulkan::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                             const Tegra::Engines::Fermi2D::Surface& dst,
                                             const Tegra::Engines::Fermi2D::Config& copy_config) {
    ONEF_LEND_RESULT(Copy, AccelerateSurfaceCopy(src, dst, copy_config))
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

bool RasterizerVulkan::UsesNvidiaDlss() const {
    // OoT runs NVIDIA DLSS through its own temporal path, not the texture cache's: count it too,
    // or "Use the game's own DLSS" took over the game's upscale and NVIDIA DLSS never ran
    // (2026-11-11, a player with both ticked: DLSS on, Streamline up, no DLSS all session).
    return texture_cache_runtime.UsesNvidiaDlss() ||
           (OcarinaTemporalEnabled() && oot_backend == VideoCommon::OcarinaUpscaleBackend::Dlss);
}

bool RasterizerVulkan::SupportsGameDlssNetwork() const {
    // Compute pipelines are created 32 threads wide where the device can be told to.
    return device.IsGuestWarpSizeSupported(VK_SHADER_STAGE_COMPUTE_BIT);
}

std::optional<VideoCore::GameDlssProbe> RasterizerVulkan::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst) {
    // One follower: Fermi2D asks on E; the texture cache is R's.
    ONEF_LEND_RESULT(Copy, ProbeGameDlssOutput(src, dst))
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.ProbeGameDlssOutput(dst, src);
}

bool RasterizerVulkan::AccelerateTemporalUpscale(
    const Tegra::Engines::Fermi2D::Surface& src, const Tegra::Engines::Fermi2D::Surface& dst,
    const Tegra::Engines::Fermi2D::Surface& motion_vectors, const Tegra::Texture::TICEntry* depth,
    std::array<f32, 2> jitter, std::array<f32, 2> motion_scale, bool use_history) {
    ONEF_LEND_RESULT(Copy, AccelerateTemporalUpscale(src, dst, motion_vectors, depth, jitter,
                                                     motion_scale, use_history))
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.TemporalUpscale(dst, src, motion_vectors, depth, jitter, motion_scale,
                                          use_history);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerVulkan::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerVulkan::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                                std::span<const u8> memory) {
    // One follower: the write and the buffer half are E's (section 5), the query and texture
    // halves R's. FRITCHII_ONEF_COPY_E runs it on E (R's halves lent only where hit); else lent.
    if (VideoCommon::OneFollower::ShouldLend() &&
        OneFollowerInlineOnEngine(address, copy_size, memory)) [[unlikely]] {
        return;
    }
    ONEF_LEND(Copy, AccelerateInlineToMemory(address, copy_size, memory))
    // Grid placeholders among the words (a compute macro's constant buffer of its dispatch's
    // size): the GPU copies the words they stand for over them once the write is done.
    if (Tegra::Nx2IndirectGrid::AnyLive() && !Tegra::Nx2IndirectGrid::g_writing_qmd &&
        copy_size >= 4) [[unlikely]] {
        // (source, destination, words): x alone, or x, y and z when all three are placeholders.
        boost::container::small_vector<std::tuple<GPUVAddr, GPUVAddr, u32>, 2> grids;
        const size_t words = (std::min)(copy_size, memory.size()) / sizeof(u32);
        const auto word = [&](size_t i) {
            u32 value;
            std::memcpy(&value, memory.data() + i * sizeof(u32), sizeof(value));
            return value;
        };
        for (size_t i = 0; i < words; ++i) {
            const auto match = Tegra::Nx2IndirectGrid::MatchX(word(i));
            if (!match) {
                continue;
            }
            const auto [slot, source] = *match;
            const bool all = i + 3 <= words &&
                             word(i + 1) == Tegra::Nx2IndirectGrid::Placeholder(slot, 1) &&
                             word(i + 2) == Tegra::Nx2IndirectGrid::Placeholder(slot, 2);
            grids.emplace_back(source, address + i * sizeof(u32), all ? 3U : 1U);
        }
        if (!grids.empty()) {
            AccelerateInlineToMemoryImpl(address, copy_size, memory);
            // The write's own invalidation first: it would upload the placeholders over the copy.
            gpu_memory->FlushCaching();
            std::scoped_lock lock{buffer_cache.mutex};
            MemStage::FillPlay seam{mem_stage, MemStage::Site::Dma};
            for (const auto& [source, destination, count] : grids) {
                Tegra::Nx2IndirectGrid::Get().copied.fetch_add(1, std::memory_order_relaxed);
                if (!buffer_cache.DMACopy(source, destination, count * sizeof(u32))) {
                    static std::atomic<u32> logged{};
                    if (logged.fetch_add(1) < 8) {
                        LOG_WARNING(Render_Vulkan,
                                    "GPU-sized dispatch: no host buffer to copy its size from "
                                    "{:#x} to {:#x}", source, destination);
                    }
                }
            }
            return;
        }
    }
    AccelerateInlineToMemoryImpl(address, copy_size, memory);
}

namespace {
/// EDEN_NX2_INLINE_DEFER=0: invalidate every cache at once (for comparisons).
bool InlineDefer() {
    static const bool defer = [] {
        const char* value = std::getenv("EDEN_NX2_INLINE_DEFER");
        return value == nullptr || value[0] != '0';
    }();
    return defer;
}
} // Anonymous namespace

void RasterizerVulkan::InlineUploadNote(GPUVAddr address, std::optional<DAddr> cpu_addr,
                                        size_t copy_size, std::span<const u8> memory) {
    using Fine = VideoCommon::Nx2Profile::FineCounter;
    // fable_memstage_compute_brief 3 (GPU profile 2): by size; the caller counts the branch.
    VideoCommon::Nx2Profile::AddFine(copy_size <= 4     ? Fine::InlineSize4
                                     : copy_size <= 64  ? Fine::InlineSize64
                                     : copy_size <= 256 ? Fine::InlineSize256
                                                        : Fine::InlineSizeMore,
                                     1);
    // EDEN_NX2_MEMSTAGE_STATS: an inline upload, and whether it lands on texture pages (join 2).
    VideoCommon::MemStageStats::StreamWrite(VideoCommon::MemStageStats::Write::Inline,
                                            cpu_addr.value_or(0), copy_size);
    if (gpu_memory->UsesNativeTiling()) {
        u32 first_word{};
        if (memory.size() >= sizeof(first_word)) std::memcpy(&first_word, memory.data(), sizeof(first_word));
        VideoCommon::TraceNativeQuery(11, gpu_memory->GetID(), address, cpu_addr.value_or(0),
            0, first_word, static_cast<u32>(copy_size));
        static const GPUVAddr watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
        }();
        if (watch != 0 && address < watch + 16 && address + copy_size > watch) {
            LOG_INFO(Render_Vulkan, "Semaphore watch: inline write {:#x} size {} first word {:#x}",
                     address, copy_size, first_word);
        }
    }
}

bool RasterizerVulkan::InlineSkipsCaches(DAddr cpu_addr, size_t copy_size) {
    // EDEN_NX2_EXP85 & 1 (live; fable52_answer_microcuts 1): no GPU write ever marked these pages,
    // so InlineMemory would find no GPU-modified range to upload into (both gpu_modified_ranges.Add
    // sites mark the GPU-written bit, on this thread): skip the buffer cache's lock.
    if (!InlineDefer() || (VideoCore::Nx2Debug::Experiment(85) & 1) == 0) {
        return false;
    }
    const DAddr page_begin = Common::AlignDown(cpu_addr, Core::DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(cpu_addr + copy_size, Core::DEVICE_PAGESIZE);
    return !MayBeGpuModified(page_begin, page_end - page_begin);
}

void RasterizerVulkan::InlineUntouchedTail(GPUVAddr address, DAddr cpu_addr, size_t copy_size,
                                           bool per_word) {
    // InlineSkipsCaches's branch, after the write. The list note takes its own mutex; the query
    // cache only where its filter says it holds something.
    VideoCommon::Nx2Profile::AddFine(VideoCommon::Nx2Profile::FineCounter::InlineNeverWritten, 1);
    buffer_cache.Nx2NoteListHostWrite(cpu_addr, copy_size);
    if (query_cache.page_filter.Any(cpu_addr, copy_size)) {
        query_cache.InvalidateRegion(cpu_addr, copy_size);
    }
    if (!per_word) {
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return;
    }
    // EXP124: the accumulator keeps a range per call where one range does not extend the last,
    // and skips a call inside it: words added one by one leave what the word calls left.
    for (size_t offset = 0; offset < copy_size; offset += sizeof(u32)) {
        gpu_memory->AccumulateInvalidation(address + offset,
                                           (std::min)(sizeof(u32), copy_size - offset));
    }
}

void RasterizerVulkan::AccelerateInlineWords(GPUVAddr address, size_t copy_size,
                                             std::span<const u8> memory) {
    // FRITCHII_ONEF_COPY_E: on E when no piece is on a query page (word calls go through
    // AccelerateInlineToMemory's own E path or lend alone).
    const bool onef_on_engine = VideoCommon::OneFollower::ShouldLend() &&
                                OneFollowerInlineWordsOnEngine(address, copy_size);
    if (!onef_on_engine) {
        ONEF_LEND(Copy, AccelerateInlineWords(address, copy_size, memory))
    }
    // EXP124 (fable_memstage_compute_brief 4.2): a macro's upload words, flushed at once, written
    // as their calls one by one would have been. Pieces of one 4 KiB page each take one call
    // where every word of it would take InlineSkipsCaches's branch: there each step (the write,
    // the list note, the query filter and its invalidation, and the accumulated invalidation,
    // added word by word) over the piece is the same as over its words, and the page test gives
    // every word of the page the same answer. Everything else goes word by word, as before.
    using Fine = VideoCommon::Nx2Profile::FineCounter;
    const auto add = [](Fine counter) { VideoCommon::Nx2Profile::AddFine(counter, 1); };
    const auto word_calls = [&](size_t begin, size_t end) {
        for (size_t offset = begin; offset < end; offset += sizeof(u32)) {
            const size_t size = (std::min)(sizeof(u32), end - offset);
            AccelerateInlineToMemory(address + offset, size, memory.subspan(offset, size));
        }
    };
    if (Tegra::Nx2IndirectGrid::AnyLive()) {
        // The placeholder scan copies x, y and z when it sees all three in one write, x alone in
        // a word's.
        add(Fine::InlineWordsGrid);
        word_calls(0, copy_size);
        return;
    }
    if ((address & 3) != 0) {
        // The list note clears the page's GPU word marks only for words a write covers whole:
        // word calls off a 4-byte boundary clear none, one call clears those inside it.
        add(Fine::InlineWordsUnaligned);
        word_calls(0, copy_size);
        return;
    }
    constexpr GPUVAddr page_size = GPUVAddr{1} << 12;
    const bool verify = VideoCore::Nx2Debug::Experiment(124) == 2;
    for (size_t offset = 0; offset < copy_size;) {
        const GPUVAddr piece_address = address + offset;
        const size_t end = (std::min<size_t>)(copy_size,
                                              offset + (page_size - (piece_address & (page_size - 1))));
        const size_t size = end - offset;
        const std::span<const u8> bytes = memory.subspan(offset, size);
        const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(piece_address);
        // Sparse pages: Nx2BackSparseWrite saw the words one at a time (WriteWordsCached).
        const bool sparse =
            piece_address + size > Tegra::g_nx2_sparse_low.load(std::memory_order_relaxed) &&
            piece_address < Tegra::g_nx2_sparse_high.load(std::memory_order_relaxed);
        if (!cpu_addr || sparse) {
            add(Fine::InlinePiecesUnmapped);
            word_calls(offset, end);
            offset = end;
            continue;
        }
        if (!InlineSkipsCaches(*cpu_addr, size)) {
            add(Fine::InlinePiecesWritten);
            word_calls(offset, end);
            offset = end;
            continue;
        }
        if (verify) {
            // EXP124=2: every word asked as its own call would ask; a piece whose words disagree
            // goes word by word (must stay 0).
            add(Fine::InlinePiecesChecked);
            bool agree = true;
            for (size_t word = offset; word < end && agree; word += sizeof(u32)) {
                const size_t word_size = (std::min)(sizeof(u32), end - word);
                const std::optional<DAddr> word_cpu = gpu_memory->GpuToCpuAddress(address + word);
                agree = word_cpu && *word_cpu == *cpu_addr + (word - offset) &&
                        InlineSkipsCaches(*word_cpu, word_size);
            }
            if (!agree) {
                add(Fine::InlinePiecesDisagree);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                    LOG_WARNING(Render_Vulkan,
                                "Upload words (EXP124=2): the words of the piece at {:#x} ({} "
                                "bytes, device {:#x}) disagree with its page test",
                                piece_address, size, *cpu_addr);
                }
                word_calls(offset, end);
                offset = end;
                continue;
            }
        }
        add(Fine::InlinePiecesOne);
        {
            const Common::WriteRing::SiteScope write_site{"inline upload (AccelerateInlineWords)",
                                                          false};
            const VideoCommon::Nx2Profile::FineScope inline_zone{
                VideoCommon::Nx2Profile::Zone::InlineUpload};
            InlineUploadNote(piece_address, cpu_addr, size, bytes);
            gpu_memory->WriteBlockUnsafe(piece_address, bytes.data(), size);
            InlineUntouchedTail(piece_address, *cpu_addr, size, true);
        }
        offset = end;
    }
}

bool RasterizerVulkan::OneFollowerInlineWordsOnEngine(GPUVAddr address, size_t copy_size) {
    namespace OneF = VideoCommon::OneFollower;
    if (!OneF::CopyOnEngine() || !InlineDefer() || copy_size == 0) {
        return false;
    }
    // The pieces' own path (InlineUntouchedTail) invalidates the query cache where its filter
    // is hit: R's. Any piece on a query page, or a report op E pushed: lent whole. Then the
    // stamps (item 2's pair with R's image registration), before any write.
    constexpr GPUVAddr page_size = GPUVAddr{1} << 12;
    for (size_t offset = 0; offset < copy_size;) {
        const GPUVAddr piece = address + offset;
        const size_t end =
            (std::min<size_t>)(copy_size, offset + (page_size - (piece & (page_size - 1))));
        if (const auto device = gpu_memory->GpuToCpuAddress(piece)) {
            if (query_cache.page_filter.Any(*device, end - offset) ||
                (onef_report_pages != nullptr && onef_report_pages->Any(*device, end - offset))) {
                OneF::CountInvalidation(OneF::InvalKind::InlineQuery);
                return false;
            }
        }
        offset = end;
    }
    for (size_t offset = 0; offset < copy_size;) {
        const GPUVAddr piece = address + offset;
        const size_t end =
            (std::min<size_t>)(copy_size, offset + (page_size - (piece & (page_size - 1))));
        if (const auto device = gpu_memory->GpuToCpuAddress(piece)) {
            OneF::StampPageWrites(*device, end - offset);
        }
        offset = end;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    OneF::CountInvalidation(OneF::InvalKind::InlineOnEngine);
    return true;
}

bool RasterizerVulkan::OneFollowerInlineOnEngine(GPUVAddr address, size_t copy_size,
                                                 std::span<const u8> memory) {
    namespace OneF = VideoCommon::OneFollower;
    // AccelerateInlineToMemoryImpl's deferred path (EDEN_NX2_INLINE_DEFER, the default), E's
    // parts here: the guest write, the list note, the buffer cache, the accumulated
    // invalidation (FlushCaching later: item 2's test). R's: the query cache (the whole upload
    // is lent when a range is on a query page or a report op E pushed), the texture and shader
    // caches after an in-place upload (that half lent alone, where their filters are hit).
    if (!OneF::CopyOnEngine() || !InlineDefer() || copy_size == 0 ||
        Tegra::Nx2IndirectGrid::AnyLive()) {
        return false;
    }
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) {
        return false;
    }
    if (query_cache.page_filter.Any(*cpu_addr, copy_size) ||
        (onef_report_pages != nullptr && onef_report_pages->Any(*cpu_addr, copy_size))) {
        OneF::CountInvalidation(OneF::InvalKind::InlineQuery);
        return false;
    }
    // Item 2's pair, before the write: the write's guard (GuardStreamWrite in WriteBlock) reads
    // the texture and shader filters after these stamps and drains R on a hit; R registering an
    // image meanwhile reads the stamps after its mark (WrittenAheadOfRender).
    OneF::StampPageWrites(*cpu_addr, copy_size);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const Common::WriteRing::SiteScope write_site{"inline upload (AccelerateInlineToMemory)", false};
    const VideoCommon::Nx2Profile::FineScope inline_zone{VideoCommon::Nx2Profile::Zone::InlineUpload};
    using Fine = VideoCommon::Nx2Profile::FineCounter;
    InlineUploadNote(address, cpu_addr, copy_size, memory);
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    OneF::CountInvalidation(OneF::InvalKind::InlineOnEngine);
    if (InlineSkipsCaches(*cpu_addr, copy_size)) {
        // Its query test finds nothing (tested above; only R's plays of E's report ops, in
        // E's own filter, or lent calls mark the query filter).
        InlineUntouchedTail(address, *cpu_addr, copy_size, false);
        return true;
    }
    bool uploaded;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::InlineUpload};
        buffer_cache.Nx2NoteListHostWrite(*cpu_addr, copy_size);
        uploaded = buffer_cache.InlineMemory(*cpu_addr, copy_size, memory);
    }
    VideoCommon::Nx2Profile::AddFine(uploaded ? Fine::InlineInPlace : Fine::InlineNotUploaded, 1);
    // (query_cache.InvalidateRegion: no query page here, tested above.)
    if (!uploaded) {
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return true;
    }
    // In place: the texture and shader halves now, as the Impl does, lent where they are hit.
    if (texture_cache.page_filter.Any(*cpu_addr, copy_size) ||
        pipeline_cache.page_filter.Any(*cpu_addr, copy_size)) {
        OneF::CountInvalidation(OneF::InvalKind::InlineRHalf);
        const DAddr device = *cpu_addr;
        auto r_half = [this, device, copy_size] {
            {
                std::scoped_lock lock{texture_cache.mutex};
                texture_cache.WriteMemory(device, copy_size);
            }
            pipeline_cache.InvalidateRegion(device, copy_size);
        };
        OneF::SetLendSite("inline upload, R's half");
        OneF::Lend(OneF::LentOp::Copy, r_half);
    }
    return true;
}

void RasterizerVulkan::AccelerateInlineToMemoryImpl(GPUVAddr address, size_t copy_size,
                                                    std::span<const u8> memory) {
    const Common::WriteRing::SiteScope write_site{"inline upload (AccelerateInlineToMemory)", false};
    const VideoCommon::Nx2Profile::FineScope inline_zone{VideoCommon::Nx2Profile::Zone::InlineUpload};
    using Fine = VideoCommon::Nx2Profile::FineCounter;
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    InlineUploadNote(address, cpu_addr, copy_size, memory);
    if (!cpu_addr) [[unlikely]] {
        VideoCommon::Nx2Profile::AddFine(Fine::InlineUnmapped, 1);
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    const bool defer = InlineDefer();
    if (InlineSkipsCaches(*cpu_addr, copy_size)) {
        // Counted as InlineNeverWritten (in InlineUntouchedTail): with InlineUnmapped,
        // InlineInPlace and InlineNotUploaded it sums to the size counters.
        InlineUntouchedTail(address, *cpu_addr, copy_size, false);
        return;
    }
    // One cache at a time: holding both while waiting for either (std::lock) waited longer, the
    // fencing thread and CPU write traps holding one of them most of the time.
    bool uploaded;
    {
        std::scoped_lock lock{buffer_cache.mutex};
        MemStage::FillPlay seam{mem_stage, MemStage::Site::InlineUpload};
        // Guest memory has the words now: a command list copy of the page no longer does.
        buffer_cache.Nx2NoteListHostWrite(*cpu_addr, copy_size);
        uploaded = buffer_cache.InlineMemory(*cpu_addr, copy_size, memory);
        if (!uploaded && !defer) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    VideoCommon::Nx2Profile::AddFine(uploaded ? Fine::InlineInPlace : Fine::InlineNotUploaded, 1);
    // Pending query results must not land over the new words later: drop them now.
    query_cache.InvalidateRegion(*cpu_addr, copy_size);
    if (!uploaded && defer) {
        // Title 1 makes thousands of small inline writes between draws (~10% of the GPU
        // thread invalidating every cache for each). Guest memory is written; the buffer,
        // texture and shader caches see the merged ranges at the next FlushCaching, which runs
        // before anything of theirs is used. Writes into GPU-written buffer data were uploaded
        // in place above, as before.
        gpu_memory->AccumulateInvalidation(address, copy_size);
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

std::optional<FramebufferTextureInfo> RasterizerVulkan::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr, u32 pixel_stride) {
    if (!framebuffer_addr) {
        return {};
    }
    ONEF_LEND_RESULT(FrameEnd, AccelerateDisplay(config, framebuffer_addr, pixel_stride))
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    static u32 display_trace_count{};
    static const bool frame_capture = std::getenv("EDEN_NATIVE_FRAME_CAPTURE") != nullptr;
    if (frame_capture && display_trace_count++ < 64) {
        LOG_INFO(HW_GPU, "Native display source: address={:#x}, offset={:#x}, dimensions={}x{}, stride={}, accelerated={}, mask={:#x}",
                 framebuffer_addr, config.offset, config.width, config.height, pixel_stride,
                 image_view != nullptr, config.layer_stack_mask);
    }
    if (!image_view) {
        return {};
    }
    if (frame_dump.Capturing()) {
        frame_dump.EndPass();
        frame_dump.CapturePresent(*image_view);
    }
    if (flash_watch.Enabled()) {
        flash_watch.NotePresent(*image_view);
    }

    query_cache.NotifySegment(false);

    const auto& resolution = Settings::values.resolution_info;

    FramebufferTextureInfo info{};
    info.image = image_view->ImageHandle();
    info.image_view = image_view->Handle(Shader::TextureType::Color2D);
    info.width = image_view->size.width;
    info.height = image_view->size.height;
    info.scaled_width = scaled ? resolution.ScaleUp(info.width) : info.width;
    info.scaled_height = scaled ? resolution.ScaleUp(info.height) : info.height;
    info.anti_aliased = oot_dlss_frame == render_trace_frame && !oot_live_dlss->EvaluationFailed();
    if (!oot_4k_scene_frame || oot_live_failed || !oot_live_dlss ||
        oot_live_dlss->EvaluationFailed()) {
        return info;
    }
    // Why this frame shows the game's 1440p image instead of the 4K one, for the log.
    const char* const miss =
        *oot_4k_scene_frame != render_trace_frame ? "the 4K scene is from another frame"
        : !info.image || info.image != oot_4k_guest_present || !info.image_view
            ? "not the game's final image"
        : info.width != 2560 || info.height != 1440 ||
                // FRITCHII_DLSS_SCALED: the composite samples the game's image by position.
                (scaled && !VideoCommon::OcarinaScaledUpscale()) ||
                image_view->format != VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM
            ? "unexpected presented image"
        : frame_dump.Capturing() ? "frame dump"
                                 : nullptr;
    const auto now = std::chrono::steady_clock::now();
    if (miss) {
        if (const auto start = oot_4k_present_log.Fallback(render_trace_frame, now, miss)) {
            LOG_INFO(Render_Vulkan,
                     "Ocarina temporal: frame {} presents the game's 1440p image, not the 4K {} "
                     "image ({}; 4K scene frame {}){}",
                     render_trace_frame, VideoCommon::OcarinaBackendName(*oot_backend), miss,
                     *oot_4k_scene_frame,
                     start->unlogged == 0
                         ? std::string{}
                         : fmt::format(", {} more stretches since the last line",
                                       start->unlogged));
        }
        return info;
    }
    try {
        oot_live_dlss->Composite4k(info.image_view, render_trace_frame);
        info.image = oot_live_dlss->CompositedImage();
        info.image_view = oot_live_dlss->CompositedView();
        info.scaled_width = oot_live_dlss->CompositedExtent().width;
        info.scaled_height = oot_live_dlss->CompositedExtent().height;
        info.already_upscaled = true;
    } catch (const vk::Exception& error) {
        LOG_WARNING(Render_Vulkan, "Ocarina temporal 4K presentation unavailable: {}", error.what());
        oot_live_failed = true;
        texture_cache_runtime.ocarina_fsr_failed = true; // samplers drop the FSR LOD bias
        return info;
    }
    if (const auto end = oot_4k_present_log.Replaced(render_trace_frame, now)) {
        LOG_INFO(Render_Vulkan,
                 "Ocarina temporal: the 4K {} image is presented again from frame {}, after {} "
                 "frames ({:.1f} s)",
                 VideoCommon::OcarinaBackendName(*oot_backend), render_trace_frame, end->frames,
                 end->seconds);
    }
    return info;
}

void RasterizerVulkan::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    ApplyOcarinaResolution(title_id);
    pipeline_cache.LoadDiskResources(title_id, stop_loading, callback);
}

void RasterizerVulkan::FlushWork() {
#ifdef __ANDROID__
    static constexpr bool android = true;
#else
    static constexpr bool android = false;
#endif // __ANDROID__
    const bool native = gpu_memory->UsesNativeTiling();
    // The 128-draw native budget is tuned for MoltenVK; desktop Vulkan takes the legacy 4096.
    const u32 default_native_budget =
        (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::SubmitBudget) && !device.IsMoltenVK()) ? 4096U
                                                                : SubmissionPolicy::DefaultNativeBudget;
    u32 budget = default_native_budget;
    if (native && !android) {
        static const std::optional<u32> configured_budget = []() -> std::optional<u32> {
            if (const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS")) {
                if (const auto parsed = SubmissionPolicy::ParseBudget(value)) {
                    return parsed;
                }
                LOG_WARNING(Render_Vulkan,
                            "Invalid EDEN_NX2_FLUSH_DRAWS; expected an integer from 1 to {}",
                            (std::numeric_limits<u32>::max)());
            }
            return std::nullopt;
        }();
        budget = configured_budget.value_or(default_native_budget);
    }
    u32 legacy_budget = 4096;
    if (!native && !android) {
        static const u32 forced_draws = [] {
            const char* value = std::getenv("EDEN_NX2_FLUSH_DRAWS");
            return value != nullptr
                       ? static_cast<u32>(std::max(32L, std::strtol(value, nullptr, 10)) & ~31L)
                       : 0U;
        }();
        legacy_budget = forced_draws != 0 ? forced_draws : (device.IsMoltenVK() ? 128U : 4096U);
    }
    switch (SubmissionPolicy::Advance(draw_counter, native, android, budget, legacy_budget)) {
    case SubmissionPolicy::Action::None:
        break;
    case SubmissionPolicy::Action::Dispatch:
        scheduler.DispatchWork();
        break;
    case SubmissionPolicy::Action::Flush:
        scheduler.Flush();
        pending_native_work = false;
        break;
    }
    if (native && !android) {
        pending_native_work = true;
    }
}

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_,
                             Scheduler& scheduler_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_}, scheduler{scheduler_} {}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    // FRITCHII_ONEF_OP_DMA: the buffer cache is E's: E runs it, its commands an op packet (the
    // Dma site posted), the result E's own.
    const bool onef_on_engine =
        VideoCommon::OneFollower::ShouldLend() &&
        VideoCommon::OneFollower::PostOpSite(static_cast<u8>(MemStage::Site::Dma));
    if (!onef_on_engine) {
        ONEF_LEND_RESULT(Copy, BufferClear(src_address, amount, value))
    }
    std::scoped_lock lock{buffer_cache.mutex};
    MemStage::FillPlay seam{buffer_cache.runtime.Seam(), MemStage::Site::Dma};
    return buffer_cache.DMAClear(src_address, amount, value);
}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    // FRITCHII_ONEF_OP_DMA: as BufferClear; the submit after list page copies is R's (a render
    // op at the same place).
    const bool onef_on_engine =
        VideoCommon::OneFollower::ShouldLend() &&
        VideoCommon::OneFollower::PostOpSite(static_cast<u8>(MemStage::Site::Dma));
    if (!onef_on_engine) {
        ONEF_LEND_RESULT(Copy, BufferCopy(src_address, dest_address, amount))
    }
    std::scoped_lock lock{buffer_cache.mutex};
    MemStage& seam = buffer_cache.runtime.Seam();
    bool copied;
    bool list_copied;
    {
        MemStage::Fill fill{seam, MemStage::Site::Dma};
        buffer_cache.Nx2CollectListCopies(true);
        copied = buffer_cache.DMACopy(src_address, dest_address, amount);
        list_copied = buffer_cache.Nx2CopyListPages(buffer_cache.runtime.Clock().Now());
    }
    seam.Play();
    if (list_copied) {
        if (onef_on_engine) {
            Scheduler* const target = &scheduler;
            VideoCommon::OneFollower::SetLendSite("BufferCopy's flush (posted)");
            VideoCommon::OneFollower::PushOp(
                VideoCommon::OneFollower::LentOp::Copy,
                [](void* raw) {
                    Scheduler* s;
                    std::memcpy(&s, raw, sizeof(s));
                    s->Flush();
                },
                &target, sizeof(target));
        } else {
            scheduler.Flush();
        }
    }
    buffer_cache.Nx2CollectListCopies(false);
    return copied;
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    ONEF_LEND_RESULT(Copy, DmaBufferImageCopy<IS_IMAGE_UPLOAD>(copy_info, buffer_operand,
                                                               image_operand))
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] = buffer_cache.runtime.Seam().Ref(MemStage::Site::ImageCopy, [&] {
        return buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);
    });
    if (buffer->Handle() == VK_NULL_HANDLE) [[unlikely]] {
        // The buffer operand is unmapped (ObtainBuffer's null buffer, which has no VkBuffer with
        // nullDescriptor on): a GPU copy would use device address 0. The engine's own copy does
        // it through guest memory instead, which skips the unmapped pages.
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Vulkan,
                        "DMA {} at unmapped GPU address {:#x} ({} bytes): left to the engine's "
                        "copy (no GPU copy into the null buffer)",
                        IS_IMAGE_UPLOAD ? "buffer to image" : "image to buffer",
                        buffer_operand.address, buffer_size);
        }
        return false;
    }

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

void RasterizerVulkan::UpdateDynamicStates(const GraphicsPipeline* pipeline) {
    auto& regs = maxwell3d->regs;
    auto& flags = maxwell3d->dirty.flags;
    // A mesh emulation pipeline has static input assembly and vertex input. Binding it replaces
    // that dynamic state, so leave it to the next regular draw to set again.
    const bool emulated_mesh{pipeline != nullptr && pipeline->IsEmulatedMesh()};
    if (emulated_mesh) {
        state_tracker.InvalidateInputState();
    }
    const auto topology = maxwell3d->draw_manager.draw_state.topology;
    const bool topology_changed = !emulated_mesh && state_tracker.ChangePrimitiveTopology(topology);
    // The dynamic topology follows the bound pipeline the way its static one does (FillStates):
    // a patch list with tessellation stages, points for patches without them. Intel honours the
    // dynamic topology as given (dynamicPrimitiveTopologyUnrestricted), so a tessellation
    // pipeline drawn as a triangle list put stray, stretched triangles on screen.
    dynamic_host_topology = MaxwellToVK::PrimitiveTopology(device, topology);
    if (pipeline != nullptr && pipeline->HasTessellation()) {
        dynamic_host_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    } else if (dynamic_host_topology == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) {
        dynamic_host_topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    }
    const bool host_topology_changed =
        !emulated_mesh && state_tracker.ChangeHostPrimitiveTopology(dynamic_host_topology);
    if (topology_changed || host_topology_changed) {
        flags[Dirty::DepthBiasEnable] = true;
        flags[Dirty::PrimitiveRestartEnable] = true;
    }

    UpdateViewportsState(regs);
    UpdateScissorsState(regs);
    UpdateDepthBias(regs);
    UpdateBlendConstants(regs);
    UpdateDepthBounds(regs);
    UpdateStencilFaces(regs);
    UpdateLineWidth(regs);

    if (device.IsExtExtendedDynamicStateSupported()) {
        UpdateCullMode(regs);
        UpdateDepthCompareOp(regs);
        UpdateFrontFace(regs);
        UpdateStencilOp(regs);
        if (state_tracker.TouchStateEnable()) {
            UpdateDepthBoundsTestEnable(regs);
            UpdateDepthTestEnable(regs);
            UpdateDepthWriteEnable(regs);
            UpdateStencilTestEnable(regs);
        }
        if (host_topology_changed) {
            scheduler.Record([topology_vk = dynamic_host_topology](vk::CommandBuffer cmdbuf) {
                cmdbuf.SetPrimitiveTopologyEXT(topology_vk);
            });
        }
    }

    if (device.IsExtExtendedDynamicState2Supported()) {
        if (!emulated_mesh) {
            UpdatePrimitiveRestartEnable(regs);
        }
        UpdateRasterizerDiscardEnable(regs);
        UpdateDepthBiasEnable(regs);
    }

    if (device.IsExtExtendedDynamicState2ExtrasSupported()) {
        UpdateLogicOp(regs);
    }

    if (device.IsExtExtendedDynamicState3EnablesSupported()) {
        using namespace Tegra::Engines;
        // AMD Workaround: LogicOp incompatible with float render targets
        if (device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_OPEN_SOURCE ||
            device.GetDriverID() == VkDriverIdKHR::VK_DRIVER_ID_AMD_PROPRIETARY) {
            const auto has_float = std::any_of(
                regs.vertex_attrib_format.begin(), regs.vertex_attrib_format.end(),
                [](const auto& attrib) {
                    return attrib.type == Maxwell3D::Regs::VertexAttribute::Type::Float;
                }
            );
            if (regs.logic_op.enable) {
                regs.logic_op.enable = static_cast<u32>(!has_float);
            }
        }
        UpdateLogicOpEnable(regs);
        UpdateDepthClampEnable(regs);
        UpdateLineRasterizationMode(regs);
        UpdateLineStippleEnable(regs);
        UpdateConservativeRasterizationMode(regs);
        UpdateAlphaToCoverageEnable(regs);
        UpdateAlphaToOneEnable(regs);
    }

    if (device.IsExtExtendedDynamicState3BlendingSupported()) {
        UpdateBlending(regs, pipeline);
    } else if (device.IsExtColorWriteEnableSupported()) {
        UpdateColorWriteEnable(regs);
    }

    if (device.IsExtVertexInputDynamicStateSupported() && !emulated_mesh) {
        // A draw passes the pipeline it looked up: a second lookup was ~2.5% of the back thread.
        const GraphicsPipeline* const gp =
            pipeline != nullptr ? pipeline : pipeline_cache.CurrentGraphicsPipeline();
        if (gp && gp->HasDynamicVertexInput()) {
            UpdateVertexInput(regs);
        } else if (gp) {
            // EXP83: binding a pipeline without dynamic vertex input leaves that state undefined.
            vertex_input_memo_valid = false;
        }
    }
}

void RasterizerVulkan::HandleTransformFeedback() {
    static std::once_flag warn_unsupported;

    const auto& regs = maxwell3d->regs;
    if (!device.IsExtTransformFeedbackSupported()) {
        if (regs.transform_feedback_enabled != 0) {
            std::call_once(warn_unsupported, [&] {
                LOG_WARNING(Render_Vulkan, "Transform feedback requested by guest but VK_EXT_transform_feedback is unavailable; queries disabled");
            });
        } else {
            std::call_once(warn_unsupported, [&] {
                LOG_INFO(Render_Vulkan, "VK_EXT_transform_feedback not available on device");
            });
        }
        return;
    }
    query_cache.CounterEnable(VideoCommon::QueryType::StreamingByteCount,
                              regs.transform_feedback_enabled);
    if (regs.transform_feedback_enabled != 0) {
        // Log extension usage for transform feedback
        if (GPU::Logging::IsActive()) {
            GPU::Logging::GPULogger::GetInstance().LogExtensionUsage(
                "VK_EXT_transform_feedback", "HandleTransformFeedback");
        }
        UNIMPLEMENTED_IF(regs.IsShaderConfigEnabled(Maxwell::ShaderType::TessellationInit) ||
                         regs.IsShaderConfigEnabled(Maxwell::ShaderType::Tessellation));
    }
}

void RasterizerVulkan::UpdateViewportsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchViewports()) {
        return;
    }

    maxwell3d->dirty.flags[Dirty::Scissors] = true;

    if (!regs.viewport_scale_offset_enabled) {
        float x = static_cast<float>(regs.surface_clip.x);
        float y = static_cast<float>(regs.surface_clip.y);
        float width = (std::max)(1.0f, static_cast<float>(regs.surface_clip.width));
        float height = (std::max)(1.0f, static_cast<float>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y += height;
            height = -height;
        }
        VkViewport viewport{
            .x = x + oot_jitter_applied[0],
            .y = y + oot_jitter_applied[1],
            .width = width,
            .height = height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        scheduler.Record([this, viewport](vk::CommandBuffer cmdbuf) {
            const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkViewport, Maxwell::NumViewports> viewport_list{};
            viewport_list.fill(viewport);
            const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
            cmdbuf.SetViewport(0, viewports);
        });
        return;
    }
    const bool is_rescaling{texture_cache.IsRescaling()};
    const float scale = is_rescaling ? Settings::values.resolution_info.up_factor : 1.0f;
    std::array viewport_list{
        GetViewportState(device, regs, 0, scale),  GetViewportState(device, regs, 1, scale),
        GetViewportState(device, regs, 2, scale),  GetViewportState(device, regs, 3, scale),
        GetViewportState(device, regs, 4, scale),  GetViewportState(device, regs, 5, scale),
        GetViewportState(device, regs, 6, scale),  GetViewportState(device, regs, 7, scale),
        GetViewportState(device, regs, 8, scale),  GetViewportState(device, regs, 9, scale),
        GetViewportState(device, regs, 10, scale), GetViewportState(device, regs, 11, scale),
        GetViewportState(device, regs, 12, scale), GetViewportState(device, regs, 13, scale),
        GetViewportState(device, regs, 14, scale), GetViewportState(device, regs, 15, scale),
    };
    if (oot_jitter_applied != std::array<f32, 2>{}) {
        for (VkViewport& viewport : viewport_list) {
            viewport.x += oot_jitter_applied[0];
            viewport.y += oot_jitter_applied[1];
        }
    }
    scheduler.Record([this, viewport_list](vk::CommandBuffer cmdbuf) {
        const u32 num_viewports = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkViewport> viewports(viewport_list.data(), num_viewports);
        cmdbuf.SetViewport(0, viewports);
    });
}

void RasterizerVulkan::UpdateScissorsState(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchScissors()) {
        return;
    }
    if (!regs.viewport_scale_offset_enabled) {
        u32 x = regs.surface_clip.x;
        u32 y = regs.surface_clip.y;
        u32 width = (std::max)(1u, static_cast<u32>(regs.surface_clip.width));
        u32 height = (std::max)(1u, static_cast<u32>(regs.surface_clip.height));
        if (regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft) {
            y = regs.surface_clip.height - (y + height);
        }
        VkRect2D scissor{};
        scissor.offset.x = static_cast<int32_t>(x);
        scissor.offset.y = static_cast<int32_t>(y);
        scissor.extent.width  = width;
        scissor.extent.height = height;
        scheduler.Record([this, scissor](vk::CommandBuffer cmdbuf) {
            const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
            std::array<VkRect2D, Maxwell::NumViewports> scissor_list{};
            scissor_list.fill(scissor);
            const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
            cmdbuf.SetScissor(0, scissors);
        });
        return;
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (texture_cache.IsRescaling()) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    const std::array scissor_list{
        GetScissorState(regs, 0, up_scale, down_shift),
        GetScissorState(regs, 1, up_scale, down_shift),
        GetScissorState(regs, 2, up_scale, down_shift),
        GetScissorState(regs, 3, up_scale, down_shift),
        GetScissorState(regs, 4, up_scale, down_shift),
        GetScissorState(regs, 5, up_scale, down_shift),
        GetScissorState(regs, 6, up_scale, down_shift),
        GetScissorState(regs, 7, up_scale, down_shift),
        GetScissorState(regs, 8, up_scale, down_shift),
        GetScissorState(regs, 9, up_scale, down_shift),
        GetScissorState(regs, 10, up_scale, down_shift),
        GetScissorState(regs, 11, up_scale, down_shift),
        GetScissorState(regs, 12, up_scale, down_shift),
        GetScissorState(regs, 13, up_scale, down_shift),
        GetScissorState(regs, 14, up_scale, down_shift),
        GetScissorState(regs, 15, up_scale, down_shift),
    };
    scheduler.Record([this, scissor_list](vk::CommandBuffer cmdbuf) {
        const u32 num_scissors = std::min<u32>(device.GetMaxViewports(), Maxwell::NumViewports);
        const vk::Span<VkRect2D> scissors(scissor_list.data(), num_scissors);
        cmdbuf.SetScissor(0, scissors);
    });
}

void RasterizerVulkan::UpdateDepthBias(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBias()) {
        return;
    }
    float units = regs.depth_bias / 2.0f;
    const bool is_d24 = regs.zeta.format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
                        regs.zeta.format == Tegra::DepthFormat::X8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::S8Z24_UNORM ||
                        regs.zeta.format == Tegra::DepthFormat::V8Z24_UNORM;

    if (is_d24 && !device.SupportsD24DepthBuffer()) {
        static constexpr const size_t length = sizeof(NEEDS_D24) / sizeof(NEEDS_D24[0]);

        static constexpr const u64* start = NEEDS_D24;
        static constexpr const u64* end = NEEDS_D24 + length;

        const u64* it = std::find(start, end, program_id);

        if (it != end) {
            // the base formulas can be obtained from here:
            //   https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
            const double rescale_factor =
                static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
            units = static_cast<float>(static_cast<double>(units) * rescale_factor);
        }
    }

    scheduler.Record([constant = units, clamp = regs.depth_bias_clamp,
                      factor = regs.slope_scale_depth_bias, this](vk::CommandBuffer cmdbuf) {
        if (device.IsExtDepthBiasControlSupported()) {
            static VkDepthBiasRepresentationInfoEXT bias_info{
                .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_REPRESENTATION_INFO_EXT,
                .pNext = nullptr,
                .depthBiasRepresentation =
                    VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT,
                .depthBiasExact = VK_FALSE,
            };

            cmdbuf.SetDepthBias(constant, clamp, factor, &bias_info);
        } else {
            cmdbuf.SetDepthBias(constant, clamp, factor);
        }
    });
}

void RasterizerVulkan::UpdateBlendConstants(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchBlendConstants()) {
        return;
    }
    const std::array blend_color = {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b,
                                    regs.blend_color.a};
    scheduler.Record(
        [blend_color](vk::CommandBuffer cmdbuf) { cmdbuf.SetBlendConstants(blend_color.data()); });
}

void RasterizerVulkan::UpdateDepthBounds(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBounds()) {
        return;
    }
    scheduler.Record([min = regs.depth_bounds[0], max = regs.depth_bounds[1]](
                         vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBounds(min, max); });
}

void RasterizerVulkan::UpdateStencilFaces(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilProperties()) {
        return;
    }
    bool update_references = state_tracker.TouchStencilReference();
    bool update_write_mask = state_tracker.TouchStencilWriteMask();
    bool update_compare_masks = state_tracker.TouchStencilCompare();
    if (state_tracker.TouchStencilSide(regs.stencil_two_side_enable != 0)) {
        update_references = true;
        update_write_mask = true;
        update_compare_masks = true;
    }
    if (update_references) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_back_ref);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilReferenceFront(regs.stencil_front_ref);
                const bool back_changed =
                    state_tracker.CheckStencilReferenceBack(regs.stencil_front_ref);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_ref = regs.stencil_front_ref, back_ref = regs.stencil_back_ref,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_ref != back_ref;
                // Front face
                cmdbuf.SetStencilReference(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_ref);
                if (set_back) {
                    cmdbuf.SetStencilReference(VK_STENCIL_FACE_BACK_BIT, back_ref);
                }
            });
        }();
    }
    if (update_write_mask) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_back_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilWriteMaskFront(regs.stencil_front_mask);
                const bool back_changed =
                    state_tracker.CheckStencilWriteMaskBack(regs.stencil_front_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_write_mask = regs.stencil_front_mask,
                              back_write_mask = regs.stencil_back_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_write_mask != back_write_mask;
                // Front face
                cmdbuf.SetStencilWriteMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                    : VK_STENCIL_FACE_FRONT_AND_BACK,
                                           front_write_mask);
                if (set_back) {
                    cmdbuf.SetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT, back_write_mask);
                }
            });
        }();
    }
    if (update_compare_masks) {
        [&]() {
            bool changed;
            if (regs.stencil_two_side_enable) {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_back_func_mask);
                changed = front_changed || back_changed;
            } else {
                const bool front_changed =
                    state_tracker.CheckStencilCompareMaskFront(regs.stencil_front_func_mask);
                const bool back_changed =
                    state_tracker.CheckStencilCompareMaskBack(regs.stencil_front_func_mask);
                changed = front_changed || back_changed;
            }
            if (!changed) {
                return;
            }
            scheduler.Record([front_test_mask = regs.stencil_front_func_mask,
                              back_test_mask = regs.stencil_back_func_mask,
                              two_sided = regs.stencil_two_side_enable](vk::CommandBuffer cmdbuf) {
                const bool set_back = two_sided && front_test_mask != back_test_mask;
                // Front face
                cmdbuf.SetStencilCompareMask(set_back ? VK_STENCIL_FACE_FRONT_BIT
                                                      : VK_STENCIL_FACE_FRONT_AND_BACK,
                                             front_test_mask);
                if (set_back) {
                    cmdbuf.SetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT, back_test_mask);
                }
            });
        }();
    }
    state_tracker.ClearStencilReset();
}

void RasterizerVulkan::UpdateLineWidth(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineWidth()) {
        return;
    }
    const float width =
        regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
    scheduler.Record([width](vk::CommandBuffer cmdbuf) { cmdbuf.SetLineWidth(width); });
}

void RasterizerVulkan::UpdateCullMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchCullMode()) {
        return;
    }
    scheduler.Record([enabled = regs.gl_cull_test_enabled,
                      cull_face = regs.gl_cull_face](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetCullModeEXT(enabled ? MaxwellToVK::CullFace(cull_face) : VK_CULL_MODE_NONE);
    });
}

void RasterizerVulkan::UpdateDepthBoundsTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBoundsTestEnable()) {
        return;
    }
    bool enabled = regs.depth_bounds_enable;
    if (enabled && !device.IsDepthBoundsSupported()) {
        LOG_WARNING(Render_Vulkan, "Depth bounds is enabled but not supported");
        enabled = false;
    }
    scheduler.Record([enable = enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthBoundsTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_test_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthTestEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateDepthWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthWriteEnable()) {
        return;
    }
    scheduler.Record([enable = regs.depth_write_enabled](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthWriteEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdatePrimitiveRestartEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchPrimitiveRestartEnable()) {
        return;
    }

    bool enable = regs.primitive_restart.enabled != 0;
    if (device.IsMoltenVK()) {
        enable = true;
    } else if (enable) {
        enable = IsPrimitiveRestartSupported(device, dynamic_host_topology);
    }

    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetPrimitiveRestartEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateRasterizerDiscardEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchRasterizerDiscardEnable()) {
        return;
    }
    scheduler.Record([disable = regs.rasterize_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetRasterizerDiscardEnableEXT(disable == 0);
    });
}

void RasterizerVulkan::UpdateConservativeRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchConservativeRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3ConservativeRasterizationMode()) {
        return;
    }

    scheduler.Record([enable = regs.conservative_raster_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetConservativeRasterizationModeEXT(
            enable ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                   : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
    });
}

void RasterizerVulkan::UpdateLineStippleEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLineStippleEnable()) {
        return;
    }

    if (!device.SupportsDynamicState3LineStippleEnable()) {
        return;
    }

    scheduler.Record([enable = regs.line_stipple_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineStippleEnableEXT(enable);
    });
}

void RasterizerVulkan::UpdateLineRasterizationMode(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!device.IsExtLineRasterizationSupported()) {
        return;
    }
    if (!state_tracker.TouchLineRasterizationMode()) {
        return;
    }

    if (!device.SupportsDynamicState3LineRasterizationMode()) {
        static std::once_flag warn_missing_rect;
        std::call_once(warn_missing_rect, [] {
            LOG_WARNING(Render_Vulkan,
                        "Driver lacks rectangular line rasterization support; skipping dynamic "
                        "line state updates");
        });
        return;
    }

    const bool wants_smooth = regs.line_anti_alias_enable != 0;
    VkLineRasterizationModeEXT mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT;
    if (wants_smooth) {
        if (device.SupportsSmoothLines()) {
            mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT;
        } else {
            static std::once_flag warn_missing_smooth;
            std::call_once(warn_missing_smooth, [] {
                LOG_WARNING(Render_Vulkan,
                            "Line anti-aliasing requested but smoothLines feature unavailable; "
                            "using rectangular rasterization");
            });
        }
    }
    scheduler.Record([mode](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLineRasterizationModeEXT(mode);
    });
}

void RasterizerVulkan::UpdateDepthBiasEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthBiasEnable()) {
        return;
    }
    constexpr size_t POINT = 0;
    constexpr size_t LINE = 1;
    constexpr size_t POLYGON = 2;
    static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        POINT,   // Points
        LINE,    // Lines
        LINE,    // LineLoop
        LINE,    // LineStrip
        POLYGON, // Triangles
        POLYGON, // TriangleStrip
        POLYGON, // TriangleFan
        POLYGON, // Quads
        POLYGON, // QuadStrip
        POLYGON, // Polygon
        LINE,    // LinesAdjacency
        LINE,    // LineStripAdjacency
        POLYGON, // TrianglesAdjacency
        POLYGON, // TriangleStripAdjacency
        POLYGON, // Patches
    };
    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    const u32 topology_index = u32(maxwell3d->draw_manager.draw_state.topology);
    const u32 enable = enabled_lut[POLYGON_OFFSET_ENABLE_LUT[topology_index]];
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthBiasEnableEXT(enable != 0); });
}

void RasterizerVulkan::UpdateLogicOpEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOpEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3LogicOpEnable()) {
        return;
    }
    scheduler.Record([enable = regs.logic_op.enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetLogicOpEnableEXT(enable != 0);
    });
}

void RasterizerVulkan::UpdateDepthClampEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthClampEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3DepthClampEnable()) {
        return;
    }
    bool is_enabled = !(regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::Passthrough ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ ||
                        regs.viewport_clip_control.geometry_clip ==
                            Maxwell::ViewportClipControl::GeometryClip::FrustumZ);
    scheduler.Record(
        [is_enabled](vk::CommandBuffer cmdbuf) { cmdbuf.SetDepthClampEnableEXT(is_enabled); });
}

void RasterizerVulkan::UpdateAlphaToCoverageEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    // The enable also depends on the pipeline writing colour 0, which the dirty flag does not
    // track. EDEN_NX2_A2C_REFRESH=1 records it on every draw so it never outlives its pipeline.
    static const bool refresh{std::getenv("EDEN_NX2_A2C_REFRESH") != nullptr};
    if (!state_tracker.TouchAlphaToCoverageEnable() && !refresh) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToCoverageEnable()) {
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToCoverage() &&
                        regs.anti_alias_alpha_control.alpha_to_coverage != 0;
    alpha_to_coverage_enable = enable;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToCoverageEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateAlphaToOneEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchAlphaToOneEnable()) {
        return;
    }
    if (!device.SupportsDynamicState3AlphaToOneEnable()) {
        static std::once_flag warn_alpha_to_one;
        std::call_once(warn_alpha_to_one, [] {
            LOG_WARNING(Render_Vulkan,
                        "Alpha-to-one is not supported on this device; forcing it disabled");
        });
        return;
    }
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    const bool enable = pipeline != nullptr && pipeline->SupportsAlphaToOne() &&
                        regs.anti_alias_alpha_control.alpha_to_one != 0;
    scheduler.Record([enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetAlphaToOneEnableEXT(enable ? VK_TRUE : VK_FALSE);
    });
}

void RasterizerVulkan::UpdateDepthCompareOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchDepthCompareOp()) {
        return;
    }
    scheduler.Record([func = regs.depth_test_func](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetDepthCompareOpEXT(MaxwellToVK::ComparisonOp(func));
    });
}

void RasterizerVulkan::UpdateFrontFace(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchFrontFace()) {
        return;
    }

    VkFrontFace front_face = MaxwellToVK::FrontFace(regs.gl_front_face);
    if (regs.window_origin.flip_y != 0) {
        front_face = front_face == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                           : VK_FRONT_FACE_CLOCKWISE;
    }
    scheduler.Record(
        [front_face](vk::CommandBuffer cmdbuf) { cmdbuf.SetFrontFaceEXT(front_face); });
}

void RasterizerVulkan::UpdateStencilOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilOp()) {
        return;
    }
    const Maxwell::StencilOp::Op fail = regs.stencil_front_op.fail;
    const Maxwell::StencilOp::Op zfail = regs.stencil_front_op.zfail;
    const Maxwell::StencilOp::Op zpass = regs.stencil_front_op.zpass;
    const Maxwell::ComparisonOp compare = regs.stencil_front_op.func;
    if (regs.stencil_two_side_enable) {
        // Separate stencil op per face
        const Maxwell::StencilOp::Op back_fail = regs.stencil_back_op.fail;
        const Maxwell::StencilOp::Op back_zfail = regs.stencil_back_op.zfail;
        const Maxwell::StencilOp::Op back_zpass = regs.stencil_back_op.zpass;
        const Maxwell::ComparisonOp back_compare = regs.stencil_back_op.func;
        scheduler.Record([fail, zfail, zpass, compare, back_fail, back_zfail, back_zpass,
                          back_compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_BIT, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_BACK_BIT, MaxwellToVK::StencilOp(back_fail),
                                   MaxwellToVK::StencilOp(back_zpass),
                                   MaxwellToVK::StencilOp(back_zfail),
                                   MaxwellToVK::ComparisonOp(back_compare));
        });
    } else {
        // Front face defines the stencil op of both faces
        scheduler.Record([fail, zfail, zpass, compare](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetStencilOpEXT(VK_STENCIL_FACE_FRONT_AND_BACK, MaxwellToVK::StencilOp(fail),
                                   MaxwellToVK::StencilOp(zpass), MaxwellToVK::StencilOp(zfail),
                                   MaxwellToVK::ComparisonOp(compare));
        });
    }
}

void RasterizerVulkan::UpdateLogicOp(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchLogicOp()) {
        return;
    }
    const auto op_value = static_cast<u32>(regs.logic_op.op);
    auto op = op_value >= 0x1500 && op_value < 0x1510 ? static_cast<VkLogicOp>(op_value - 0x1500)
                                                      : VK_LOGIC_OP_NO_OP;
    scheduler.Record([op](vk::CommandBuffer cmdbuf) { cmdbuf.SetLogicOpEXT(op); });
}

void RasterizerVulkan::UpdateBlending(Tegra::Engines::Maxwell3D::Regs& regs,
                                      const GraphicsPipeline* pipeline) {
    // The write mask also excludes components the fragment shader doesn't output, so it has to be
    // recorded again whenever the bound pipeline outputs a different set of components.
    const u32 color_output_mask = pipeline ? pipeline->ColorOutputMask() : ~0U;
    const bool color_output_changed = color_output_mask != last_color_output_mask;
    if (!state_tracker.TouchBlending() && !color_output_changed) {
        return;
    }

    if (state_tracker.TouchColorMask() || color_output_changed) {
        last_color_output_mask = color_output_mask;
        std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> setup_masks{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
            auto& current = setup_masks[index];
            if (mask.R) {
                current |= VK_COLOR_COMPONENT_R_BIT;
            }
            if (mask.G) {
                current |= VK_COLOR_COMPONENT_G_BIT;
            }
            if (mask.B) {
                current |= VK_COLOR_COMPONENT_B_BIT;
            }
            if (mask.A) {
                current |= VK_COLOR_COMPONENT_A_BIT;
            }
            current &= (color_output_mask >> (index * 4)) & 0xF;
        }
        scheduler.Record([setup_masks](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorWriteMaskEXT(0, setup_masks);
        });
    }

    if (state_tracker.TouchBlendEnable()) {
        std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
        for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
            bool is_integer = false;
            if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
                const auto format =
                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format);
                is_integer = IsPixelFormatInteger(format);
            }
            setup_enables[index] =
                (!is_integer && regs.blend.enable[index] != 0) ? VK_TRUE : VK_FALSE;
        }
        scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEnableEXT(0, setup_enables);
        });
    }

    if (state_tracker.TouchBlendEquations()) {
        std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> setup_blends{};

        const auto blend_setup = [&](auto& host_blend, const auto& guest_blend) {
            host_blend.srcColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_source);
            host_blend.dstColorBlendFactor = MaxwellToVK::BlendFactor(guest_blend.color_dest);
            host_blend.colorBlendOp = MaxwellToVK::BlendEquation(guest_blend.color_op);
            host_blend.srcAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_source);
            host_blend.dstAlphaBlendFactor = MaxwellToVK::BlendFactor(guest_blend.alpha_dest);
            host_blend.alphaBlendOp = MaxwellToVK::BlendEquation(guest_blend.alpha_op);
        };

        // Single blend equation for all targets
        if (!regs.blend_per_target_enabled) {
            // Temporary workaround for games that use iterated blending
            if (regs.iterated_blend.enable && Settings::values.use_squashed_iterated_blend) {
                setup_blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
                setup_blends[0].colorBlendOp = VK_BLEND_OP_ADD;
                setup_blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                setup_blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                setup_blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
            } else {
                blend_setup(setup_blends[0], regs.blend);
            }

            // Copy first blend state to all other targets
            for (size_t index = 1; index < Maxwell::NumRenderTargets; index++) {
                setup_blends[index] = setup_blends[0];
            }
        } else {
            // Per-target blending
            for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
                blend_setup(setup_blends[index], regs.blend_per_target[index]);
            }
        }

        scheduler.Record([setup_blends](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetColorBlendEquationEXT(0, setup_blends);
        });
    }
}

void RasterizerVulkan::UpdateColorWriteEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchColorMask()) {
        return;
    }
    std::array<VkBool32, Maxwell::NumRenderTargets> setup_enables{};
    for (size_t index = 0; index < Maxwell::NumRenderTargets; index++) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        setup_enables[index] = (mask.R || mask.G || mask.B || mask.A) ? VK_TRUE : VK_FALSE;
    }
    scheduler.Record([setup_enables](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetColorWriteEnableEXT(setup_enables);
    });
}

void RasterizerVulkan::UpdateStencilTestEnable(Tegra::Engines::Maxwell3D::Regs& regs) {
    if (!state_tracker.TouchStencilTestEnable()) {
        return;
    }
    scheduler.Record([enable = regs.stencil_enable](vk::CommandBuffer cmdbuf) {
        cmdbuf.SetStencilTestEnableEXT(enable);
    });
}

namespace {
// FRITCHII_VERTEX_INPUT_COMPACT: a vertex input command sized to what it carries. The full
// command captured two static_vectors of 32 Vulkan structs (about 2 KB of chunk space, copied
// into the chunk with every send); this one keeps 12 bytes a binding or attribute in one of
// three size classes and builds the Vulkan structs on the worker.
struct PackedVertexBinding {
    u32 stride;
    u32 divisor;
    u8 binding;
    u8 input_rate; // VkVertexInputRate: 0 vertex, 1 instance
};
struct PackedVertexAttribute {
    VkFormat format;
    u32 offset;
    u8 location;
    u8 binding;
};

template <size_t NumBindings, size_t NumAttributes>
struct CompactVertexInputCommand {
    std::array<PackedVertexBinding, NumBindings> bindings;
    std::array<PackedVertexAttribute, NumAttributes> attributes;
    u8 num_bindings;
    u8 num_attributes;

    void operator()(vk::CommandBuffer cmdbuf) const {
        std::array<VkVertexInputBindingDescription2EXT, NumBindings> vk_bindings;
        std::array<VkVertexInputAttributeDescription2EXT, NumAttributes> vk_attributes;
        for (u32 i = 0; i < num_bindings; ++i) {
            const PackedVertexBinding& b = bindings[i];
            vk_bindings[i] = {
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .binding = b.binding,
                .stride = b.stride,
                .inputRate = static_cast<VkVertexInputRate>(b.input_rate),
                .divisor = b.divisor,
            };
        }
        for (u32 i = 0; i < num_attributes; ++i) {
            const PackedVertexAttribute& a = attributes[i];
            vk_attributes[i] = {
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .location = a.location,
                .binding = a.binding,
                .format = a.format,
                .offset = a.offset,
            };
        }
        cmdbuf.SetVertexInputEXT(
            vk::Span<VkVertexInputBindingDescription2EXT>(vk_bindings.data(), num_bindings),
            vk::Span<VkVertexInputAttributeDescription2EXT>(vk_attributes.data(), num_attributes));
    }
};

template <size_t NumBindings, size_t NumAttributes>
void RecordCompactVertexInput(Scheduler& scheduler,
                              std::span<const VkVertexInputBindingDescription2EXT> bindings,
                              std::span<const VkVertexInputAttributeDescription2EXT> attributes) {
    CompactVertexInputCommand<NumBindings, NumAttributes> command;
    command.num_bindings = static_cast<u8>(bindings.size());
    command.num_attributes = static_cast<u8>(attributes.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        const auto& b = bindings[i];
        command.bindings[i] = {
            .stride = b.stride,
            .divisor = b.divisor,
            .binding = static_cast<u8>(b.binding),
            .input_rate = static_cast<u8>(b.inputRate),
        };
    }
    for (size_t i = 0; i < attributes.size(); ++i) {
        const auto& a = attributes[i];
        command.attributes[i] = {
            .format = a.format,
            .offset = a.offset,
            .location = static_cast<u8>(a.location),
            .binding = static_cast<u8>(a.binding),
        };
    }
    scheduler.Record(std::move(command));
}
} // Anonymous namespace

void RasterizerVulkan::UpdateVertexInput(Tegra::Engines::Maxwell3D::Regs& regs) {
    auto& dirty{maxwell3d->dirty.flags};
    const bool vertex_input_dirty = dirty[Dirty::VertexInput];
    const bool vertex_buffers_dirty = dirty[VideoCommon::Dirty::VertexBuffers];
    if (!vertex_input_dirty && !vertex_buffers_dirty) {
        return;
    }
    dirty[Dirty::VertexInput] = false;
    // Step 7 (fps fix 2), a served draw with the window open: the back's VertexBuffers flag is
    // not the cache's (GPUMem's copy holds it and its fills clear it there, plan 4.2), so on the
    // back nothing cleared it and it stayed raised: every draw sent the vertex input state again
    // (88,500 sends a second, 45,000 the same as the last). Here it only means "a stream register
    // changed, or a new command buffer, since the last send", so it is cleared with the send. A
    // lend sets the back's cache flags from GPUMem's copy again.
    if (vertex_buffers_dirty && VideoCommon::MemoryStageLink::BackServed() &&
        VideoCommon::MemoryStageLink::Ahead() != 0) {
        dirty[VideoCommon::Dirty::VertexBuffers] = false;
    }

    boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> bindings;
    boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> attributes;

    const u32 max_attributes =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexAttributes,
                                          device.GetMaxVertexInputAttributes()));
    const u32 max_bindings =
        static_cast<u32>(std::min<size_t>(Maxwell::NumVertexArrays,
                                          device.GetMaxVertexInputBindings()));

    // FRITCHII_VERTEX_INPUT_COMPACT (live, default 0; fable_perdraw_b 3.2): 1 = describe only
    // the bindings an attribute below names, in a command sized to its content. The attribute
    // list is the same in both modes (every non-constant location the registers define), so the
    // state no longer depends on which pipeline drew first: vkCmdSetVertexInputEXT replaces the
    // whole vertex input state, a shader reads locations, every location the registers define
    // is described with its binding, and a binding no attribute names cannot be read by any
    // pipeline (buffers bound to it are ignored). Strides, rates and divisors of the bindings
    // that are described are unchanged.
    const int compact = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_vertex_input_compact, "FRITCHII_VERTEX_INPUT_COMPACT");
    u32 referenced_bindings = 0;

    for (u32 index = 0; index < max_attributes; ++index) {
        const Maxwell::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        const u32 binding{attribute.buffer};
        if (attribute.constant || binding >= max_bindings) {
            continue;
        }
        referenced_bindings |= 1U << binding;
        attributes.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .location = index,
            .binding = binding,
            .format = MaxwellToVK::VertexFormat(device, attribute.type, attribute.size),
            .offset = attribute.offset +
                      (((back_repack_mask >> binding) & 1) != 0 && attribute.offset >= 65 ? 3 : 0),
        });
    }

    const auto push_binding = [&](u32 binding) {
        const auto& input_binding{regs.vertex_streams[binding]};
        const bool is_instanced{regs.vertex_stream_instances.IsInstancingEnabled(binding)};
        bindings.push_back({
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .pNext = nullptr,
            .binding = binding,
            .stride = ((back_repack_mask >> binding) & 1) != 0 ? 72U : input_binding.stride.Value(),
            .inputRate = is_instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = is_instanced ? input_binding.frequency : 1,
        });
    };
    if (compact != 0) {
        // Ascending binding order, as below (only bits under max_bindings are set).
        for (u32 mask = referenced_bindings; mask != 0; mask &= mask - 1) {
            push_binding(static_cast<u32>(std::countr_zero(mask)));
        }
    } else {
        for (u32 binding = 0; binding < max_bindings; ++binding) {
            push_binding(binding);
        }
    }

    for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
        dirty[Dirty::VertexAttribute0 + index] = false;
    }
    for (size_t index = 0; index < Maxwell::NumVertexArrays; ++index) {
        dirty[Dirty::VertexBinding0 + index] = false;
    }

    // EDEN_NX2_EXP83 (live, 0 = off; fable52_answer_microcuts 3.2): a vertex buffer address or
    // stride write raises VertexBuffers and so sent the whole state again, usually unchanged.
    // A dynamic state stays set in a command buffer until set again or a pipeline without it is
    // bound (both drop the memo), so a call with the last call's arguments changes nothing.
    // Compared field by field: the structs have padding after sType.
    const int memo_mode = VideoCore::Nx2Debug::Experiment(83);
    ++vertex_input_sent;
    if (vertex_input_memo_valid &&
        vertex_input_memo_generation == state_tracker.vertex_input_generation &&
        bindings.size() == vertex_input_memo_bindings.size() &&
        attributes.size() == vertex_input_memo_attributes.size()) {
        bool same = true;
        for (size_t i = 0; same && i < bindings.size(); ++i) {
            const auto& a = bindings[i];
            const auto& b = vertex_input_memo_bindings[i];
            same = a.binding == b.binding && a.stride == b.stride && a.inputRate == b.inputRate &&
                   a.divisor == b.divisor;
        }
        for (size_t i = 0; same && i < attributes.size(); ++i) {
            const auto& a = attributes[i];
            const auto& b = vertex_input_memo_attributes[i];
            same = a.location == b.location && a.binding == b.binding && a.format == b.format &&
                   a.offset == b.offset;
        }
        if (same) {
            ++vertex_input_repeats;
            if (memo_mode != 0) {
                return;
            }
        }
    }
    vertex_input_memo_bindings = bindings;
    vertex_input_memo_attributes = attributes;
    vertex_input_memo_generation = state_tracker.vertex_input_generation;
    vertex_input_memo_valid = true;

    // FRITCHII_VERTEX_INPUT_COMPACT's counts (kept in both modes: at 0 they say what 1 saves).
    constexpr size_t full_command_bytes = sizeof(bindings) + sizeof(attributes);
    size_t command_bytes = full_command_bytes;
    const size_t num_bindings = bindings.size();
    const size_t num_attributes = attributes.size();
    if (compact != 0) {
        const std::span<const VkVertexInputBindingDescription2EXT> binding_span(bindings.data(),
                                                                                num_bindings);
        const std::span<const VkVertexInputAttributeDescription2EXT> attribute_span(
            attributes.data(), num_attributes);
        if (num_bindings <= 4 && num_attributes <= 8) {
            command_bytes = sizeof(CompactVertexInputCommand<4, 8>);
            RecordCompactVertexInput<4, 8>(scheduler, binding_span, attribute_span);
        } else if (num_bindings <= 8 && num_attributes <= 16) {
            command_bytes = sizeof(CompactVertexInputCommand<8, 16>);
            RecordCompactVertexInput<8, 16>(scheduler, binding_span, attribute_span);
        } else {
            command_bytes = sizeof(CompactVertexInputCommand<32, 32>);
            RecordCompactVertexInput<32, 32>(scheduler, binding_span, attribute_span);
        }
    } else {
        scheduler.Record([bindings, attributes](vk::CommandBuffer cmdbuf) {
            cmdbuf.SetVertexInputEXT(bindings, attributes);
        });
    }
    ++vertex_input_recorded;
    vertex_input_bindings_described += num_bindings;
    vertex_input_bindings_referenced += static_cast<u64>(std::popcount(referenced_bindings));
    vertex_input_attributes_described += num_attributes;
    vertex_input_command_bytes += command_bytes;

    static auto last_log = std::chrono::steady_clock::now();
    if (const auto now = std::chrono::steady_clock::now();
        now - last_log >= std::chrono::seconds{10}) {
        LOG_INFO(Render_Vulkan,
                 "Vertex input (EXP83={}) last 10 s: {} sends, {} the same as the last one ({})",
                 memo_mode, vertex_input_sent, vertex_input_repeats,
                 memo_mode != 0 ? "skipped" : "sent anyway");
        const double recorded = static_cast<double>(std::max<u64>(vertex_input_recorded, 1));
        LOG_INFO(Render_Vulkan,
                 "Vertex input compact (FRITCHII_VERTEX_INPUT_COMPACT={}) last 10 s: {} recorded, "
                 "bindings described {:.1f} a send (referenced {:.1f}), attributes {:.1f} a send, "
                 "command {:.0f} bytes a send (full {})",
                 compact, vertex_input_recorded,
                 static_cast<double>(vertex_input_bindings_described) / recorded,
                 static_cast<double>(vertex_input_bindings_referenced) / recorded,
                 static_cast<double>(vertex_input_attributes_described) / recorded,
                 static_cast<double>(vertex_input_command_bytes) / recorded, full_command_bytes);
        vertex_input_sent = vertex_input_repeats = 0;
        vertex_input_recorded = vertex_input_bindings_described = 0;
        vertex_input_bindings_referenced = vertex_input_attributes_described = 0;
        vertex_input_command_bytes = 0;
        last_log = now;
    }
}

void RasterizerVulkan::InitializeChannel(Tegra::Control::ChannelState& channel) {
    ONEF_LEND(Channel, InitializeChannel(channel))
    CreateChannel(channel);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.CreateChannel(channel);
        buffer_cache.CreateChannel(channel);
    }
    pipeline_cache.CreateChannel(channel);
    query_cache.CreateChannel(channel);
    state_tracker.SetupTables(channel);
}

void RasterizerVulkan::BindChannel(Tegra::Control::ChannelState& channel) {
    ONEF_LEND(Channel, BindChannel(channel))
    const s32 channel_id = channel.bind_id;
    VideoCommon::MemStageStats::ChannelBound(channel_id);
    const size_t previous_channel = current_channel_id;
    BindToChannel(channel_id);
    // Delta draw count (Fable plan D1, row 3): only a real switch ends reuse (the state tracker
    // is reset below); binding the bound channel again doesn't.
    if (VideoCommon::Nx2FastDrawCount::g_on && current_channel_id != previous_channel)
        [[unlikely]] {
        VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::Channel);
    }
    if (!oot_resolution_checked) [[unlikely]] {
        ApplyOcarinaResolution(program_id); // no disk cache load (it does this first)
    }
    if (VideoCommon::MemoryStageLink::BackServed()) {
        // FRITCHII_MEMSTAGE_THREAD = 2: the stage binds the buffer cache's channel itself.
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
    } else {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.BindToChannel(channel_id);
        buffer_cache.BindToChannel(channel_id);
    }
    pipeline_cache.BindToChannel(channel_id);
    query_cache.BindToChannel(channel_id);
    state_tracker.ChangeChannel(channel);
    state_tracker.InvalidateState();
    if (VideoCommon::OneFollower::On()) {
        // R's view follows the bound channel's engine (seeded in full at the next lent packet).
        VideoCommon::OneFollower::SetEngine(maxwell3d);
    }
}

void RasterizerVulkan::ReleaseChannel(s32 channel_id) {
    ONEF_LEND(Channel, ReleaseChannel(channel_id))
    EraseChannel(channel_id);
    {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        texture_cache.EraseChannel(channel_id);
        buffer_cache.EraseChannel(channel_id);
    }
    pipeline_cache.EraseChannel(channel_id);
    query_cache.EraseChannel(channel_id);
}

} // namespace Vulkan
