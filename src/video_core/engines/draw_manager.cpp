// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>

#include "common/settings.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memstage_shadow.h"
#include "video_core/memstage_shadow_engine.h"
#include "video_core/memstage_stats.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/render_trace.h"

namespace Tegra::Engines {
namespace {

// The draw manager's code runs on the back's Maxwell3D and on the memory stage's copy of its
// registers (VideoCommon::MemStageShadowEngine, FRITCHII_MEMSTAGE_SHADOW): one code, two
// instantiations (the end of this file), so which draws the stage finds, and what state each one
// sees, is decided by the same lines (fable_memstage_step6_spec.md 4.2). They differ only where a
// draw is issued (the rasterizer, or the stage's StageDraw) and in the back's counters.
template <typename Engine>
constexpr bool IsBack = std::is_same_v<Engine, Maxwell3D>;

void TraceSuppressedDraw(const Maxwell3D& maxwell3d, bool indirect) {
    if (!VideoCommon::RenderTrace::Enabled()) {
        return;
    }
    const auto& regs = maxwell3d.regs;
    const u64 vs = (u64{regs.pipelines[1].reserved[0] & 0xff} << 32) | regs.pipelines[1].reserved[1];
    const u64 ps = (u64{regs.pipelines[5].reserved[0] & 0xff} << 32) | regs.pipelines[5].reserved[1];
    static thread_local std::set<std::tuple<u64, u64, u64, bool, bool>> seen;
    if (seen.size() < 256 &&
        seen.emplace(vs, ps, regs.render_enable.Address(), Settings::IsDockedMode(), indirect).second) {
        LOG_DEBUG(HW_GPU,
                  "Render trace suppressed: docked={} indirect={} vs={:#x} ps={:#x} "
                  "condition_address={:#x} mode={} override={}",
                  Settings::IsDockedMode(), indirect, vs, ps, regs.render_enable.Address(),
                  static_cast<u32>(regs.render_enable.mode),
                  static_cast<u32>(regs.render_enable_override));
    }
}

} // namespace

template <typename Engine>
void Maxwell3D::DrawManager::ProcessMethodCall(Engine& maxwell3d, u32 method, u32 argument) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(clear_surface):
        return Clear(maxwell3d, 1);
    case MAXWELL3D_REG_INDEX(draw.begin):
        return DrawBegin(maxwell3d);
    case MAXWELL3D_REG_INDEX(draw.end):
        return DrawEnd(maxwell3d);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
        break;
    case MAXWELL3D_REG_INDEX(index_buffer.count):
        draw_state.draw_indexed = true;
        break;
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
        draw_state.instance_count++;
        [[fallthrough]];
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
        return DrawIndexSmall(maxwell3d, argument);
    case MAXWELL3D_REG_INDEX(draw_inline_index):
        SetInlineIndexBuffer(maxwell3d, argument);
        break;
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_2x16.even);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_2x16.odd);
        break;
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index0);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index1);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index2);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index3);
        break;
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
        DrawArrayInstanced(maxwell3d, maxwell3d.regs.vertex_array_instance_first.topology.Value(), maxwell3d.regs.vertex_array_instance_first.start.Value(), maxwell3d.regs.vertex_array_instance_first.count.Value(), false);
        break;
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent): {
        DrawArrayInstanced(maxwell3d, maxwell3d.regs.vertex_array_instance_subsequent.topology.Value(), maxwell3d.regs.vertex_array_instance_subsequent.start.Value(), maxwell3d.regs.vertex_array_instance_subsequent.count.Value(), true);
        break;
    }
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0): {
        DrawTexture(maxwell3d);
        break;
    }
    default:
        break;
    }
}

template <typename Engine>
void Maxwell3D::DrawManager::Clear(Engine& maxwell3d, u32 layer_count) {
    // EDEN_NX2_MEMSTAGE_STATS: clears render-enable skips.
    if constexpr (IsBack<Engine>) {
        if (maxwell3d.HasPendingAsyncE2c()) [[unlikely]] {
            maxwell3d.WaitAndSyncAsyncE2c();
        }
        VideoCommon::MemStageStats::Clear(!maxwell3d.ShouldExecute());
    }
    if (maxwell3d.ShouldExecute()) {
        maxwell3d.rasterizer->Clear(layer_count);
    }
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawDeferred(Engine& maxwell3d, bool loop_continues) {
    if (draw_state.draw_mode != DrawMode::Instance || draw_state.instance_count == 0) {
        if (first_folded) [[unlikely]] {
            // The loop a folded first instance began never came: draw it on its own after all
            // (the ported fallback; the front folds only when the loop follows).
            first_folded = false;
            draw_state.draw_mode = DrawMode::General;
            DrawEnd(maxwell3d);
        }
        if (!loop_continues) {
            draw_state.drawn_instances = 0;
        }
        return;
    }
    // The deferred draw below covers a folded first instance (drawn_instances is 0 after a
    // draw.begin First, so it starts at the loop's base instance).
    first_folded = false;
    // The first draw of a batch also covers the loop's first instance; a batch that continues a
    // loop split by a register write draws only the instances after the ones already drawn.
    const bool continued = draw_state.drawn_instances != 0;
    DrawEnd(maxwell3d, draw_state.instance_count + (continued ? 0 : 1), true);
    draw_state.drawn_instances =
        loop_continues ? draw_state.drawn_instances + draw_state.instance_count : 0;
    draw_state.instance_count = 0;
}

bool Maxwell3D::DrawManager::IsInstanceLoopMethod(u32 method) noexcept {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.count):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.count):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
        return true;
    default:
        return false;
    }
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawArray(Engine& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 vertex_first, u32 vertex_count, u32 base_instance, u32 num_instances) {
    draw_state.topology = topology;
    draw_state.vertex_buffer.first = vertex_first;
    draw_state.vertex_buffer.count = vertex_count;
    draw_state.base_instance = base_instance;
    ProcessDraw(maxwell3d, false, num_instances);
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawArrayInstanced(Engine& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 vertex_first, u32 vertex_count, bool subsequent) {
    draw_state.topology = topology;
    draw_state.vertex_buffer.first = vertex_first;
    draw_state.vertex_buffer.count = vertex_count;
    if (!subsequent) {
        draw_state.instance_count = 1;
        draw_state.drawn_instances = 0;
    }
    draw_state.base_instance = draw_state.instance_count - 1;
    draw_state.draw_mode = DrawMode::Instance;
    draw_state.instance_count++;
    ProcessDraw(maxwell3d, false, 1);
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawIndex(Engine& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 index_first, u32 index_count, u32 base_index, u32 base_instance, u32 num_instances) {
    draw_state.topology = topology;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_first;
    draw_state.index_buffer.count = index_count;
    draw_state.base_index = base_index;
    draw_state.base_instance = base_instance;
    ProcessDraw(maxwell3d, true, num_instances);
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawArrayIndirect(Engine& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology) {
    draw_state.topology = topology;
    ProcessDrawIndirect(maxwell3d);
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawIndexedIndirect(Engine& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 index_first, u32 index_count) {
    draw_state.topology = topology;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_first;
    draw_state.index_buffer.count = index_count;
    ProcessDrawIndirect(maxwell3d);
}

template <typename Engine>
void Maxwell3D::DrawManager::SetInlineIndexBuffer(Engine& maxwell3d, u32 index) {
    draw_state.inline_index_draw_indexes.push_back(u8(index & 0x000000ff));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0x0000ff00) >> 8));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0x00ff0000) >> 16));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0xff000000) >> 24));
    draw_state.draw_mode = DrawMode::InlineIndex;
}

template <typename Engine>
void Maxwell3D::DrawManager::SetInlineIndexBuffer(Engine& maxwell3d, u32 method,
                                                  const u32* base_start, u32 amount) {
    auto& index_buffer = draw_state.inline_index_draw_indexes;
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw_inline_index): {
        const auto* const bytes = reinterpret_cast<const u8*>(base_start);
        index_buffer.insert(index_buffer.end(), bytes, bytes + size_t(amount) * sizeof(u32));
        break;
    }
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even): {
        const size_t offset = index_buffer.size();
        index_buffer.resize(offset + size_t(amount) * 2 * sizeof(u32));
        u8* dst = index_buffer.data() + offset;
        for (u32 i = 0; i < amount; ++i) {
            const u32 word = base_start[i];
            const u32 indexes[2]{word & 0xFFFF, word >> 16};
            std::memcpy(dst, indexes, sizeof(indexes));
            dst += sizeof(indexes);
        }
        break;
    }
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0): {
        const size_t offset = index_buffer.size();
        index_buffer.resize(offset + size_t(amount) * 4 * sizeof(u32));
        u8* dst = index_buffer.data() + offset;
        for (u32 i = 0; i < amount; ++i) {
            const u32 word = base_start[i];
            const u32 indexes[4]{word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF,
                                 word >> 24};
            std::memcpy(dst, indexes, sizeof(indexes));
            dst += sizeof(indexes);
        }
        break;
    }
    }
    draw_state.draw_mode = DrawMode::InlineIndex;
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawBegin(Engine& maxwell3d) {
    auto reset_instance_count = maxwell3d.regs.draw.instance_id == Maxwell3D::Regs::Draw::InstanceId::First;
    auto increment_instance_count = maxwell3d.regs.draw.instance_id == Maxwell3D::Regs::Draw::InstanceId::Subsequent;
    if (reset_instance_count) {
        DrawDeferred(maxwell3d);
        draw_state.instance_count = 0;
        draw_state.draw_mode = DrawMode::General;
    } else if (increment_instance_count) {
        draw_state.instance_count++;
        draw_state.draw_mode = DrawMode::Instance;
    }
    draw_state.topology = maxwell3d.regs.draw.topology;
}

template <typename Engine>
void Maxwell3D::DrawManager::AddLoopInstances(Engine& maxwell3d, u32 count) {
    // DrawBegin, `count` times, for draw.instance_id == Subsequent.
    draw_state.instance_count += count;
    draw_state.draw_mode = DrawMode::Instance;
    draw_state.topology = maxwell3d.regs.draw.topology;
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawEnd(Engine& maxwell3d, u32 instance_count, bool force_draw) {
    // List draws (nx2_list_draw.h): set between this draw.end and its draw.begin, for this draw only.
    const GPUVAddr list_draw_address = std::exchange(draw_state.list_draw_address, 0);
    const u64 list_keep_id = std::exchange(draw_state.list_keep_id, 0);
    if (fold_first_instance) [[unlikely]] {
        // EDEN_NX2_FOLD_INSTANCE (ported, its draw_manager.cpp DrawEnd).
        fold_first_instance = false;
        if (draw_state.draw_mode == DrawMode::General && !force_draw &&
            draw_state.instance_count == 0) {
            // DrawDeferred draws this instance with the loop's others.
            draw_state.draw_mode = DrawMode::Instance;
            first_folded = true;
            return;
        }
    }
    switch (draw_state.draw_mode) {
    case DrawMode::Instance:
        if (!force_draw) {
            break;
        }
        [[fallthrough]];
    case DrawMode::General:
        draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
        if (draw_state.draw_mode == DrawMode::Instance && draw_state.drawn_instances != 0) {
            draw_state.base_instance += draw_state.drawn_instances + 1;
        }
        draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
        if (list_draw_address != 0 && !draw_state.draw_indexed &&
            draw_state.draw_mode == DrawMode::General) {
            // The vertex count and the first vertex are the two words at the address: the
            // rasterizer draws from them indirectly, or reads them back if it cannot.
            draw_state.vertex_buffer = maxwell3d.regs.vertex_buffer;
            indirect_state.is_byte_count = false;
            indirect_state.is_indexed = false;
            indirect_state.include_count = false;
            indirect_state.count_start_address = 0;
            indirect_state.indirect_start_address = list_draw_address;
            indirect_state.buffer_size = 2 * sizeof(u32);
            indirect_state.max_draw_counts = 1;
            indirect_state.stride = 0;
            indirect_state.list_words = true;
            indirect_state.list_keep_id = list_keep_id;
            ProcessDrawIndirect(maxwell3d);
            indirect_state.list_words = false;
            indirect_state.list_keep_id = 0;
            break;
        }
        if (draw_state.draw_indexed) {
            draw_state.index_buffer = maxwell3d.regs.index_buffer;
            ProcessDraw(maxwell3d, true, instance_count);
        } else {
            draw_state.vertex_buffer = maxwell3d.regs.vertex_buffer;
            ProcessDraw(maxwell3d, false, instance_count);
        }
        draw_state.draw_indexed = false;
        break;
    case DrawMode::InlineIndex:
        draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
        draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
        draw_state.index_buffer = maxwell3d.regs.index_buffer;
        draw_state.index_buffer.count = u32(draw_state.inline_index_draw_indexes.size() / 4);
        draw_state.index_buffer.format = Maxwell3D::Regs::IndexFormat::UnsignedInt;
        maxwell3d.dirty.flags[VideoCommon::Dirty::IndexBuffer] = true;
        if constexpr (!IsBack<Engine>) {
            maxwell3d.NoteRaised(VideoCommon::Dirty::IndexBuffer);
        }
        ProcessDraw(maxwell3d, true, instance_count);
        draw_state.inline_index_draw_indexes.clear();
        if constexpr (!IsBack<Engine>) {
            // The fill reads the inline index bytes (memstage_shadow.cpp, SHADOW=3).
            maxwell3d.NoteChanged();
        }
        break;
    }
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawIndexSmall(Engine& maxwell3d, u32 argument) {
    Maxwell3D::Regs::IndexBufferSmall index_small_params{argument};
    draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
    draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_small_params.first;
    draw_state.index_buffer.count = index_small_params.count;
    draw_state.topology = index_small_params.topology;
    maxwell3d.dirty.flags[VideoCommon::Dirty::IndexBuffer] = true;
    if constexpr (!IsBack<Engine>) {
        maxwell3d.NoteRaised(VideoCommon::Dirty::IndexBuffer);
    }
    ProcessDraw(maxwell3d, true, 1);
}

template <typename Engine>
void Maxwell3D::DrawManager::DrawTexture(Engine& maxwell3d) {
    draw_texture_state.dst_x0 = f32(maxwell3d.regs.draw_texture.dst_x0) / 4096.f;
    draw_texture_state.dst_y0 = f32(maxwell3d.regs.draw_texture.dst_y0) / 4096.f;
    const auto dst_width = f32(maxwell3d.regs.draw_texture.dst_width) / 4096.f;
    const auto dst_height = f32(maxwell3d.regs.draw_texture.dst_height) / 4096.f;
    const bool lower_left{maxwell3d.regs.window_origin.mode != Maxwell3D::Regs::WindowOrigin::Mode::UpperLeft};
    if (lower_left) {
        draw_texture_state.dst_y0 = f32(maxwell3d.regs.surface_clip.height) - draw_texture_state.dst_y0;
    }
    draw_texture_state.dst_x1 = draw_texture_state.dst_x0 + dst_width;
    draw_texture_state.dst_y1 = draw_texture_state.dst_y0 + dst_height;
    draw_texture_state.src_x0 = f32(maxwell3d.regs.draw_texture.src_x0) / 4096.f;
    draw_texture_state.src_y0 = f32(maxwell3d.regs.draw_texture.src_y0) / 4096.f;
    draw_texture_state.src_x1 = (f32(maxwell3d.regs.draw_texture.dx_du) / 4294967296.f) * dst_width + draw_texture_state.src_x0;
    draw_texture_state.src_y1 = (f32(maxwell3d.regs.draw_texture.dy_dv) / 4294967296.f) * dst_height + draw_texture_state.src_y0;
    draw_texture_state.src_sampler = maxwell3d.regs.draw_texture.src_sampler;
    draw_texture_state.src_texture = maxwell3d.regs.draw_texture.src_texture;
    maxwell3d.rasterizer->DrawTexture();
}

template <typename Engine>
void Maxwell3D::DrawManager::UpdateTopology(Engine& maxwell3d) {
    switch (maxwell3d.regs.primitive_topology_control) {
    case Maxwell3D::Regs::PrimitiveTopologyControl::UseInBeginMethods:
        break;
    case Maxwell3D::Regs::PrimitiveTopologyControl::UseSeparateState:
        switch (maxwell3d.regs.topology_override) {
        case Maxwell3D::Regs::PrimitiveTopologyOverride::None:
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::Points:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::Points;
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::Lines:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::Lines;
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::LineStrip:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::LineStrip;
            break;
        default:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology(maxwell3d.regs.topology_override);
            break;
        }
        break;
    }
}

template <typename Engine>
void Maxwell3D::DrawManager::ProcessDraw(Engine& maxwell3d, bool draw_indexed, u32 instance_count) {
    draw_state.draw_element_count =
        draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    LOG_TRACE(HW_GPU, "called, topology={}, count={}", draw_state.topology, draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count);
    UpdateTopology(maxwell3d);
    if constexpr (!IsBack<Engine>) {
        // The stage's draw, at the point the back's is compared (before the render-enable test).
        maxwell3d.StageDraw(draw_indexed, instance_count, false);
    } else {
        // FRITCHII_MEMSTAGE_SHADOW: the stage's copy issued the same draw (compares a and b).
        if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
            VideoCommon::MemStageShadow::BackDraw(maxwell3d, draw_indexed, instance_count, false);
        }
        if (maxwell3d.HasPendingAsyncE2c()) [[unlikely]] {
            (void)maxwell3d.TrySyncAsyncE2c();
        }
        const bool e2c_pending = maxwell3d.HasPendingAsyncE2c();
        const bool should_execute = e2c_pending || maxwell3d.ShouldExecute();
        // EDEN_NX2_MEMSTAGE_STATS: draws per channel, and those render-enable skips (section 4).
        VideoCommon::MemStageStats::Draw(!should_execute);
        if (should_execute) {
            maxwell3d.rasterizer->Draw(draw_indexed, instance_count);
        } else {
            TraceSuppressedDraw(maxwell3d, false);
        }
    }
}

template <typename Engine>
void Maxwell3D::DrawManager::ProcessDrawIndirect(Engine& maxwell3d) {
    draw_state.draw_element_count = 0;
    LOG_TRACE(HW_GPU, "called, topology={}, is_indexed={}, includes_count={}, buffer_size={}, max_draw_count={}", draw_state.topology, indirect_state.is_indexed, indirect_state.include_count, indirect_state.buffer_size, indirect_state.max_draw_counts);
    UpdateTopology(maxwell3d);
    if constexpr (!IsBack<Engine>) {
        maxwell3d.StageDraw(true, 1, true);
    } else {
        if (VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) [[unlikely]] {
            VideoCommon::MemStageShadow::BackDraw(maxwell3d, indirect_state.is_indexed, 1, true);
        }
        if (maxwell3d.HasPendingAsyncE2c()) [[unlikely]] {
            maxwell3d.WaitAndSyncAsyncE2c();
        }
        VideoCommon::MemStageStats::Draw(!maxwell3d.ShouldExecute());
        if (maxwell3d.ShouldExecute()) {
            maxwell3d.rasterizer->DrawIndirect();
        } else {
            TraceSuppressedDraw(maxwell3d, true);
        }
    }
}
// The two engines the draw manager's code runs on (see IsBack above).
#define FRITCHII_DRAW_MANAGER_INSTANTIATE(E)                                                       \
    template void Maxwell3D::DrawManager::ProcessMethodCall<E>(E&, u32, u32);                      \
    template void Maxwell3D::DrawManager::Clear<E>(E&, u32);                                       \
    template void Maxwell3D::DrawManager::DrawDeferred<E>(E&, bool);                               \
    template void Maxwell3D::DrawManager::DrawArray<E>(                                            \
        E&, Maxwell3D::Regs::PrimitiveTopology, u32, u32, u32, u32);                               \
    template void Maxwell3D::DrawManager::DrawArrayInstanced<E>(                                   \
        E&, Maxwell3D::Regs::PrimitiveTopology, u32, u32, bool);                                   \
    template void Maxwell3D::DrawManager::DrawIndex<E>(                                            \
        E&, Maxwell3D::Regs::PrimitiveTopology, u32, u32, u32, u32, u32);                          \
    template void Maxwell3D::DrawManager::DrawArrayIndirect<E>(                                    \
        E&, Maxwell3D::Regs::PrimitiveTopology);                                                   \
    template void Maxwell3D::DrawManager::DrawIndexedIndirect<E>(                                  \
        E&, Maxwell3D::Regs::PrimitiveTopology, u32, u32);                                         \
    template void Maxwell3D::DrawManager::SetInlineIndexBuffer<E>(E&, u32);                        \
    template void Maxwell3D::DrawManager::SetInlineIndexBuffer<E>(E&, u32, const u32*, u32);       \
    template void Maxwell3D::DrawManager::DrawBegin<E>(E&);                                        \
    template void Maxwell3D::DrawManager::AddLoopInstances<E>(E&, u32);                            \
    template void Maxwell3D::DrawManager::DrawEnd<E>(E&, u32, bool);                               \
    template void Maxwell3D::DrawManager::DrawIndexSmall<E>(E&, u32);                              \
    template void Maxwell3D::DrawManager::DrawTexture<E>(E&);                                      \
    template void Maxwell3D::DrawManager::UpdateTopology<E>(E&);                                   \
    template void Maxwell3D::DrawManager::ProcessDraw<E>(E&, bool, u32);                           \
    template void Maxwell3D::DrawManager::ProcessDrawIndirect<E>(E&);

FRITCHII_DRAW_MANAGER_INSTANTIATE(Maxwell3D)
FRITCHII_DRAW_MANAGER_INSTANTIATE(VideoCommon::MemStageShadowEngine)
#undef FRITCHII_DRAW_MANAGER_INSTANTIATE

} // namespace Tegra::Engines
