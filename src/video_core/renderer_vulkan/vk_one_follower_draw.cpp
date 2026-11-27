// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// One follower, part c: R's draws from the packet and the view, and the verify compares
// (vk_one_follower_draw.h has the model; work\fable_b_one_follower.md 3, 5, 8.1).
//
// Where the design and the code disagree, the code wins (written down here):
// - "R plays the buffer sections (MemPlayer)": R takes the draw's ring B packet with TakeRingDraw
//   and PrepareDraw plays it as a served draw (the THREAD = 2 path), so Configure binds nothing in
//   the buffer cache. E fills it with FillRingDraw, as GPUMem does, from its own engine.
// - "E finds the pipeline's layout": E looks the facts up under BackProgramId's key (LookupFacts);
//   PrepareDraw publishes them at step 4. A miss is LentReason::PipelineMiss (R has not seen the
//   programs yet), so the first draw of a pipeline is always lent.
// - The view engine's constant buffer bindings (Maxwell3D::state) come only with lent packets: the
//   R-side readers of them (EDEN_NX2_TEX_BANK_GEN, EXP87's template, the handle verify) make every
//   draw lent while they are on. Handle words always come in the packet (ValuesByValue required).
// - The engine hint (engine_state) and the macro mode (native_turing) are not registers: E lends
//   the draw when either differs from what the last lent packet gave the view.
// - TickWork and the stream's invalidations (FlushCaching) stay E's: E runs them around the push,
//   R skips them in PrepareDraw. FlushWork (the scheduler's) is R's.
// - 0xe2c: in this branch E still decides every condition (lent) and skips False draws itself;
//   R tracks its own mode at each condition (NoteCondition) and compares it (BuildCondition), and
//   drops PacketConditional draws while it is False. Step 5 makes E send those.

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include <fmt/format.h>

#include "common/dkb_frame_trace.h"
#include "common/fritchii_page_flags.h"
#include "common/scope_exit.h"
#include "video_core/fritchii_occl_flag.h"
#include "video_core/fritchii_delta_path.h"
#include "video_core/fritchii_e_split.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/fritchii_fill_cuts.h"
#include "video_core/fritchii_occl_lean.h"
#include "video_core/fritchii_price.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/memory_stage_link.h"
#include "video_core/memstage_shadow.h"
#include "video_core/native_indirect_reset.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/nx2_indirect_grid.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower_stats.h"
#include "video_core/render_trace.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_one_follower_draw.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/shader_cache.h"
#include "video_core/texture_handle_values.h"

namespace Vulkan {

namespace {
/// The instance ReportDescriptorMismatch reports for (one rasterizer at a time).
std::atomic<OneFollowerDraw*> g_instance{nullptr};
} // Anonymous namespace

namespace OneF = VideoCommon::OneFollower;
namespace Profile = VideoCommon::Nx2Profile;
using OneF::Check;
using OneF::LentOp;
using OneF::StepLevel;
using OneF::UnlendKind;
using Maxwell = Tegra::Engines::Maxwell3D::Regs;

namespace {

/// The kinds R has a path for at step 4 through Draw; List goes through DrawIndirect
/// (RouteListDraw); the other bits stay lent (vk_one_follower_draw.h).
constexpr u32 PlayableKinds =
    OneF::PlainIndexed | OneF::PlainArrays | OneF::InstanceLoop | OneF::Mesh;

/// The register words RefreshStages reads (the programs, their region, rasterization, mesh).
using Maxwell3D = Tegra::Engines::Maxwell3D; // MAXWELL3D_REG_INDEX names it
constexpr u32 ProgramWordsBegin = static_cast<u32>(MAXWELL3D_REG_INDEX(pipelines));
constexpr u32 ProgramWordsEnd =
    ProgramWordsBegin + static_cast<u32>(sizeof(Maxwell3D::Regs::pipelines) / sizeof(u32));
constexpr u32 ProgramRegionBegin = static_cast<u32>(MAXWELL3D_REG_INDEX(program_region));
constexpr u32 ProgramRegionEnd =
    ProgramRegionBegin +
    static_cast<u32>(sizeof(Maxwell3D::Regs::program_region) / sizeof(u32));
constexpr u32 RasterizeEnableWord = static_cast<u32>(MAXWELL3D_REG_INDEX(rasterize_enable));
constexpr u32 MeshControlWord = static_cast<u32>(Maxwell3D::Regs::MeshControl);

template <typename T>
[[nodiscard]] bool SameBytes(const T& a, const T& b) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

[[nodiscard]] constexpr size_t Padded(size_t bytes) noexcept {
    return (bytes + 7) & ~size_t{7};
}

/// Where span `bit` starts in a packet's spans (one_follower.h's order), or null without it.
[[nodiscard]] const u8* SpanAt(const VideoCommon::OneFollower::PacketHeader& header,
                               const u8* spans, u8 bit) noexcept {
    namespace OneF = VideoCommon::OneFollower;
    if (spans == nullptr || (header.spans & bit) == 0) {
        return nullptr;
    }
    size_t at = 0;
    const auto skip = [&](u8 span, size_t bytes) {
        if ((header.spans & span) != 0) {
            at += Padded(bytes);
        }
    };
    const u8 order[] = {OneF::SpanDelta, OneF::SpanFlags,  OneF::SpanDraw, OneF::SpanPrograms,
                        OneF::SpanPools, OneF::SpanEngine, OneF::SpanBinds};
    for (const u8 span : order) {
        if (span == bit) {
            return spans + at;
        }
        switch (span) {
        case OneF::SpanDelta:
            skip(span, size_t{header.delta_count} * sizeof(OneF::RegPair));
            break;
        case OneF::SpanFlags:
            skip(span, sizeof(OneF::FlagBits));
            break;
        case OneF::SpanDraw:
            skip(span, sizeof(OneF::DrawValues));
            break;
        case OneF::SpanPrograms:
            skip(span, sizeof(OneF::Programs));
            break;
        case OneF::SpanPools:
            skip(span, sizeof(OneF::Pools));
            break;
        case OneF::SpanEngine:
            skip(span, sizeof(OneF::EngineValues));
            break;
        default:
            break;
        }
    }
    return nullptr;
}

/// EDEN_NX2_INVALIDATE_BEFORE_DISPATCH's draw form (vk_rasterizer.cpp's InvalidateBeforeUse).
[[nodiscard]] bool InvalidateBeforeUseOn() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_INVALIDATE_BEFORE_DISPATCH");
        return value != nullptr && value[0] != '0' && std::string_view{value} != "list";
    }();
    return enabled;
}

/// The key FixedPipelineState::Refresh builds from `engine` with every refresh step forced
/// (VertexInput, ViewportSwizzles; blending is always rebuilt without dynamic blend state). The
/// engine's flags are given back as they were.
FixedPipelineState FullKey(Tegra::Engines::Maxwell3D& engine, DynamicFeatures features) {
    static_assert(std::is_trivially_copyable_v<FixedPipelineState>);
    FixedPipelineState key;
    std::memset(&key, 0, sizeof(key));
    const auto saved = engine.dirty.flags;
    engine.dirty.flags[Dirty::VertexInput] = true;
    engine.dirty.flags[Dirty::ViewportSwizzles] = true;
    engine.dirty.flags[Dirty::Blending] = true;
    key.Refresh(engine, features);
    engine.dirty.flags = saved;
    return key;
}

/// The flags each build part consumes (attribution of a missing flag; Compare).
enum class FlagPart : u8 { Pipeline, RenderTargets, ImageViews, Dynamic, Other };

[[nodiscard]] FlagPart PartOf(size_t flag) noexcept {
    namespace CommonDirty = VideoCommon::Dirty;
    if (flag == CommonDirty::Shaders || flag == Dirty::ViewportSwizzles ||
        flag == Dirty::Blending) {
        return FlagPart::Pipeline;
    }
    if (flag >= CommonDirty::RenderTargets && flag <= CommonDirty::RescaleScissors) {
        return FlagPart::RenderTargets;
    }
    if (flag == CommonDirty::Descriptors) {
        return FlagPart::ImageViews;
    }
    if (flag > Dirty::First && flag < Dirty::Last) {
        return FlagPart::Dynamic;
    }
    return FlagPart::Other;
}

} // Anonymous namespace

// --- The view engine's scope ---------------------------------------------------------------------

class OneFollowerDraw::ViewScope {
public:
    ViewScope(RasterizerVulkan& r_, Maxwell3D& engine) : r{r_} {
        rasterizer_engine = r.ExchangeMaxwell3D(&engine);
        pipeline_engine = r.pipeline_cache.ExchangeMaxwell3D(&engine);
        texture_engine = r.texture_cache.ExchangeMaxwell3D(&engine);
        query_engine = r.query_cache.ExchangeMaxwell3D(&engine);
        tracker_flags = r.state_tracker.ExchangeFlags(&engine.dirty.flags);
    }
    ~ViewScope() {
        r.state_tracker.ExchangeFlags(tracker_flags);
        r.query_cache.ExchangeMaxwell3D(query_engine);
        r.texture_cache.ExchangeMaxwell3D(texture_engine);
        r.pipeline_cache.ExchangeMaxwell3D(pipeline_engine);
        r.ExchangeMaxwell3D(rasterizer_engine);
    }
    ViewScope(const ViewScope&) = delete;
    ViewScope& operator=(const ViewScope&) = delete;

private:
    RasterizerVulkan& r;
    Maxwell3D* rasterizer_engine{};
    Maxwell3D* pipeline_engine{};
    Maxwell3D* texture_engine{};
    Maxwell3D* query_engine{};
    Maxwell3D::DirtyState::Flags* tracker_flags{};
};

// --- Construction ----------------------------------------------------------------------------------

OneFollowerDraw::OneFollowerDraw(RasterizerVulkan& rasterizer_) : rasterizer{rasterizer_} {
    g_instance.store(this, std::memory_order_release);
    for (size_t flag = 0; flag < render_flags.size(); ++flag) {
        render_flags[flag] = OneF::OwnerOf(static_cast<u8>(flag)) != OneF::FlagOwner::E;
    }
    env_traces = std::getenv("EDEN_NX2_CB24_TRACE") != nullptr ||
                 std::getenv("EDEN_NX2_TRACE_VS") != nullptr ||
                 std::getenv("EDEN_NX2_RT_TRACE") != nullptr ||
                 std::getenv("EDEN_NX2_WINDOW_TRACE") != nullptr ||
                 std::getenv("EDEN_NATIVE_DRAW_CAPTURE") != nullptr ||
                 std::getenv("EDEN_DKB_INDIRECT_READBACK") != nullptr;
}

OneFollowerDraw::~OneFollowerDraw() {
    OneFollowerDraw* self = this;
    g_instance.compare_exchange_strong(self, nullptr, std::memory_order_acq_rel);
}

// --- E: routing a draw -----------------------------------------------------------------------------

u32 OneFollowerDraw::Classify(Maxwell3D& engine, bool is_indexed, u32 instance_count) const {
    using DrawMode = Maxwell3D::DrawManager::DrawMode;
    const auto& draw = engine.draw_manager.draw_state;
    if (draw.draw_mode == DrawMode::InlineIndex || !draw.inline_index_draw_indexes.empty()) {
        return OneF::InlineIndex;
    }
    if (draw.list_draw_address != 0) {
        return OneF::List;
    }
    if (!is_indexed && Tegra::Nx2IndirectGrid::AnyLive() &&
        Tegra::Nx2IndirectGrid::LooksLikeDrawPlaceholder(draw.vertex_buffer.count)) {
        return OneF::GpuCount;
    }
    // Task/mesh pipelines (RefreshStages: native programs with the mesh stages enabled).
    if (engine.GetNativeMacroEngine().native_turing && engine.regs.IsMeshEnabled()) {
        return OneF::Mesh;
    }
    if (draw.draw_mode == DrawMode::Instance || engine.draw_manager.first_folded) {
        return OneF::InstanceLoop;
    }
    (void)instance_count;
    return is_indexed ? OneF::PlainIndexed : OneF::PlainArrays;
}

OneF::LentReason OneFollowerDraw::UnlentBlocker(Maxwell3D& engine, bool is_indexed,
                                                u32 instance_count, u32 kind) {
    namespace Debug = VideoCore::Nx2Debug;
    // Verify (8.1 point 5): every Nth draw that would run unlent is lent and compared.
    const int every =
        Debug::SwitchValue(Debug::g_onef_verify_every, "FRITCHII_ONEF_VERIFY_EVERY",
                           OneF::DefaultVerifyEvery);
    const bool verify = every > 0 && (++unlent_candidates % static_cast<u64>(every)) == 0;
    // Read every draw, so a hit never waits for a later draw (the next draw after it is lent).
    const bool stream_write = OneF::TakeStreamWriteLend();
    push_flags = 0;
    verify_reset = {};
    const auto& draw = engine.draw_manager.draw_state;
    const bool native = engine.GetNativeMacroEngine().native_turing;
    // IsNx2IndirectReset's cheap half (Draw's early out; the rest reads the program's code).
    const bool reset_candidate = kind != OneF::List && !is_indexed && instance_count == 1 &&
                                 native && draw.topology == Maxwell::PrimitiveTopology::Points;
    if (verify) {
        // FRITCHII_ONEF_UNLENT_RESET's must-be-0: E's own test now, the lent draw's on E's engine
        // inside the call (RunLent, Check::UnlentReset).
        if (reset_candidate && OneF::UnlentResetOn()) {
            // FRITCHII_ESPLIT_COUNT: the test reads the program's code (E1 would wait for E2).
            VideoCommon::ESplit::NoteStall(VideoCommon::ESplit::StallCause::MemoryDecision);
            // FRITCHII_ESPLIT=1: E1 reads game memory here, a join.
            const VideoCommon::ESplit::JoinScope esplit_join{VideoCommon::ESplit::Join::Decision};
            verify_reset = ResetCheck{true, RasterizerVulkan::IsNx2IndirectResetOn(
                                                engine, engine.GetMemoryManager(), false, 1)};
        }
        return LentReason::Verify;
    }
    if (stream_write) {
        return LentReason::StreamWrite;
    }
    // The DKB frame trace no longer lends every draw unless FRITCHII_ONEF_TRACE_LENDS=1 asks
    // (it made every one-follower measurement a fully lent one).
    static const bool trace_lends = [] {
        const char* value = std::getenv("FRITCHII_ONEF_TRACE_LENDS");
        return value != nullptr && value[0] == '1';
    }();
    if (diagnostic.load(std::memory_order_relaxed) || env_traces ||
        (trace_lends && Common::DkbFrameTrace::Enabled())) {
        return LentReason::Diagnostic;
    }
    // Transient: what R's view or R's path cannot carry yet. Each test counts itself (fable_b
    // 19 step 2's dry counter, the "One follower transient lends" line).
    using OneF::TransientTest;
    const auto transient = [kind](TransientTest test) {
        OneF::CountTransient(test, kind);
        return LentReason::Transient;
    };
    // The lent packet seeds or syncs the view.
    if (!view_ready || seeded_step != OneF::Step()) {
        return transient(TransientTest::ViewNotSeeded);
    }
    if (synced_engine != &engine) {
        return transient(TransientTest::Channel);
    }
    // FRITCHII_ONEF_UNLENT_VALUES: at mode 1 every change since the last lent packet went to the
    // view by value; at 2 the view may be newer than that packet's values: lent until the next.
    const u8 values_mode = OneF::EngineValuesMode();
    if (values_mode == 2 || (values_mode == 0 && synced_engine_state != engine.engine_state)) {
        return transient(TransientTest::EngineHint);
    }
    if (values_mode == 0 && synced_native != native) {
        return transient(TransientTest::NativeMode);
    }
    // Readers of the constant buffer bindings on R (the view has them only from lent packets).
    if (Debug::SwitchValue(Debug::g_tex_bank_gen, "EDEN_NX2_TEX_BANK_GEN") > 0 ||
        Debug::Experiment(87) != 0 || (MemStage::VerifyBits() & MemStage::VerifyHandles) != 0) {
        return transient(TransientTest::BankReaders);
    }
    // Paths that touch the buffer cache or its lock on R, or change who fills.
    if (InvalidateBeforeUseOn() || engine.regs.transform_feedback_enabled != 0 ||
        VideoCommon::Nx2FastDrawCount::g_count ||
        (VideoCommon::Nx2FastDrawCount::g_on &&
         Debug::Experiment(VideoCommon::FritchiiDeltaPath::Switch) != 0) ||
        VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed) ||
        VideoCommon::MemoryStageLink::RingActive()) {
        return transient(TransientTest::BufferPaths);
    }
    if (kind == OneF::List) {
        return LentReason::Count; // Draw's early outs below are not DrawListWords'
    }
    // Draw's own early outs: skipped past 16M vertices, the experimental indirect reset.
    const u64 first = is_indexed ? draw.index_buffer.first : draw.vertex_buffer.first;
    const u64 count = is_indexed ? draw.index_buffer.count : draw.vertex_buffer.count;
    if (first + count > (u64{1} << 24)) {
        return transient(TransientTest::VertexLimit);
    }
    if (reset_candidate) {
        const bool reset_on = OneF::UnlentResetOn();
        bool is_reset = !reset_on;
        if (reset_on) {
            const GPUVAddr program =
                (u64{engine.regs.pipelines[1].reserved[0] & 0xff} << 32) |
                engine.regs.pipelines[1].reserved[1];
            auto& memory = engine.GetMemoryManager();
            const u64 map_serial = memory.MapSerial();
            const size_t as_id = memory.GetID();
            const u64 shader_epoch = VideoCommon::ShaderCache::InvalidationEpoch();
            const bool e_split_async = VideoCommon::ESplit::g_thread && VideoCommon::ESplit::OnE1();
            const bool shader_pending = e_split_async && VideoCommon::ESplit::ShaderPending();
            constexpr size_t code_bytes =
                sizeof(u32) * VideoCommon::NativeIndirectReset::Program.size();
            ResetCacheEntry& cached =
                reset_cache[((program >> 7) ^ (program >> 14)) & (ResetCacheSize - 1)];
            const bool hit = e_split_async && !shader_pending && cached.valid &&
                             cached.program == program && cached.map_serial == map_serial &&
                             cached.as_id == as_id && cached.inval_epoch == shader_epoch &&
                             OneF::OnShaderPages(cached.device, code_bytes);
            if (hit) {
                is_reset = cached.result;
            } else {
                // FRITCHII_ONEF_UNLENT_RESET: E runs the whole test at its own place on its own
                // engine (the code read flushes as the lent draw's read would, here) and lends only
                // a real reset; the rest are pushed with PacketNoReset, so R's Draw does not test
                // again.
                VideoCommon::ESplit::NoteStall(VideoCommon::ESplit::StallCause::MemoryDecision);
                // FRITCHII_ESPLIT=1: E1 reads game memory in the test, a join.
                const VideoCommon::ESplit::JoinScope esplit_join{VideoCommon::ESplit::Join::Decision};
                const u64 pre_test_epoch = VideoCommon::ShaderCache::InvalidationEpoch();
                is_reset = RasterizerVulkan::IsNx2IndirectResetOn(engine, memory, false, 1);
                if (e_split_async && ((program + 128) & 0xfffU) + code_bytes <= 0x1000U) {
                    if (const auto dev = memory.GpuToCpuAddress(program + 128);
                        dev && OneF::OnShaderPages(*dev, code_bytes)) {
                        cached = ResetCacheEntry{
                            .program = program,
                            .device = *dev,
                            .map_serial = map_serial,
                            .as_id = as_id,
                            .inval_epoch = pre_test_epoch,
                            .valid = true,
                            .result = is_reset,
                        };
                    }
                }
            }
        }
        if (is_reset) {
            return transient(TransientTest::IndirectReset);
        }
        push_flags |= OneF::PacketNoReset;
    }
    return LentReason::Count;
}

bool OneFollowerDraw::FillRing(Maxwell3D& engine, GraphicsPipeline* pipeline, bool is_indexed,
                               u64 position, u64 seq, u64& buffer_number, LentReason& reason,
                               bool list_args) {
    auto& r = rasterizer;
    reason = LentReason::Transient;
    VideoCommon::MemStageShadow::ProgramId program_id{};
    // fable_b 14.4 cut 4: with E's pipeline (rule 3) nothing looks the facts up by program id;
    // it only names the fill in ReportDescriptorMismatch's record (left 0 then).
    if (pipeline == nullptr) {
        VideoCommon::MemStageShadow::BackProgramId(engine, program_id);
    }
    // Rule 3: the pipeline E looked up gives the layout itself, as stage off's own fill takes
    // it (MemFacts). Else join 1 (spec 4.4): the facts R published for these programs.
    const MemStage::DrawFacts pipeline_facts =
        pipeline != nullptr ? pipeline->MemFacts() : MemStage::DrawFacts{};
    // Which refusal a Transient return is (the dry counter; RouteDraw and RouteListDraw count it).
    fill_test = OneF::TransientTest::FillOther;
    filled_refused = true;
    if (pipeline != nullptr) {
        if (pipeline_facts.has_texel_buffers || pipeline_facts.stage_infos == nullptr) {
            fill_test = OneF::TransientTest::FillTexel;
            return false; // texel buffers need the texture cache (LateFill)
        }
        facts.hashes = pipeline_facts.hashes;
        facts.uniform_masks = *pipeline_facts.uniform_masks;
        facts.stages = pipeline_facts.stages;
        for (u32 stage = 0; stage < 5; ++stage) {
            facts.ssbo_counts[stage] = static_cast<u8>(
                pipeline_facts.stage_infos[stage].storage_buffers_descriptors.size());
        }
        facts.dynamic_vertex_input = pipeline->HasDynamicVertexInput();
    } else if (!r.mem_stage.LookupFacts(program_id, facts)) {
        reason = LentReason::PipelineMiss;
        return false;
    }
    // FRITCHII_ESPLIT_DEPTH > 0, E2's async fill: E1 moved on, so no lend now; ring B's room comes
    // back as R plays (its margin holds hundreds of packets: counted, about never).
    const bool esplit_async = VideoCommon::ESplit::FillAsync();
    if (esplit_async && !r.mem_stage.RingRoom()) [[unlikely]] {
        VideoCommon::ESplit::NoteRoomWait();
        while (!r.mem_stage.RingRoom()) {
            std::this_thread::yield();
        }
    }
    if ((pipeline == nullptr && (facts.has_texel_buffers || facts.infos == nullptr)) ||
        !r.mem_stage.Packets() || !r.mem_stage.ValuesByValue() || !r.mem_stage.RingRoom() ||
        r.buffer_cache.MarksAhead()) {
        fill_test = pipeline == nullptr && (facts.has_texel_buffers || facts.infos == nullptr)
                        ? OneF::TransientTest::FillTexel
                    : r.mem_stage.Packets() && r.mem_stage.ValuesByValue() &&
                            !r.mem_stage.RingRoom()
                        ? OneF::TransientTest::FillRoom
                        : OneF::TransientTest::FillOther;
        return false; // texel buffers need the texture cache (LateFill); no room or no values
    }
    std::scoped_lock lock{r.buffer_cache.mutex};
    SCOPE_EXIT {
        r.mem_stage.EndStageFill();
    };
    // A list draw: the cache's indirect parameters are the engine's for this fill, as
    // DrawListWords sets them around its own fill (the geometry section binds and synchronizes
    // the two words; FillRingDraw takes their buffer after it, as ConfigureDraw would).
    if (list_args) {
        r.buffer_cache.SetDrawIndirect(esplit_async ? &VideoCommon::ESplit::ThreadView().indirect
                                                    : &engine.draw_manager.indirect_state);
    }
    SCOPE_EXIT {
        if (list_args) {
            r.buffer_cache.SetDrawIndirect(nullptr);
        }
    };
    // FRITCHII_ESPLIT_COUNT >= 2: E1's draw record reaches E2's view here, before the fill reads
    // anything, and the view is compared with the engine; at 3 the fill reads the view where it
    // was found equal (the buffer flags stay the engine's: E2 owns them in the split).
    // FRITCHII_ESPLIT=1 (the thread): the record reached E2's view at the hand-over; the view
    // comes from there (S1: on the 1 in 64 draws whose compare found it equal).
    const bool esplit_thread = VideoCommon::ESplit::g_thread;
    const bool esplit_records = esplit_thread || VideoCommon::ESplit::g_records;
    const VideoCommon::ESplit::View* esplit_view = nullptr;
    std::optional<VideoCommon::CacheInputs> esplit_inputs;
    if (esplit_records) [[unlikely]] {
        esplit_view = esplit_thread ? VideoCommon::ESplit::FillView(engine)
                                    : VideoCommon::ESplit::DrawHandOver(engine, list_args);
        if (esplit_view != nullptr && !r.buffer_cache.HasInputsOverride()) {
            esplit_inputs.emplace(r.buffer_cache.StageInputs(
                esplit_view->regs, esplit_view->state, esplit_view->root_tables,
                esplit_view->draw_state,
                esplit_async ? VideoCommon::ESplit::ThreadView().flags : engine.dirty.flags,
                list_args ? &esplit_view->indirect : nullptr));
            r.buffer_cache.SetInputsOverride(&*esplit_inputs);
        }
    }
    SCOPE_EXIT {
        if (esplit_inputs) {
            r.buffer_cache.SetInputsOverride(nullptr);
        }
    };
    // The repack layout and the cache's flags, as PrepareDraw's own fill sets them.
    const u32 repack_dirty = r.buffer_cache_runtime.SetVertexRepackLayout(
        r.buffer_cache.Inputs().RepackLayoutRegs(), facts.dynamic_vertex_input);
    r.mem_stage.BeginDraw(r.buffer_cache_runtime.RepackMask(), repack_dirty);
    if (repack_dirty) {
        r.buffer_cache.Inputs().MarkVertexBuffersDirty(repack_dirty);
        // The lent draws read E's engine: raise it there too (R raises its own from the packet).
        // FRITCHII_ESPLIT_DEPTH > 0: E1's engine, so E1 raises it (at its next draw or join).
        if (esplit_async) [[unlikely]] {
            VideoCommon::ESplit::DeferFlag(Dirty::VertexInput);
        } else {
            engine.dirty.flags[Dirty::VertexInput] = true;
        }
        VideoCommon::ESplit::NoteBackRaise(); // FRITCHII_ESPLIT_COUNT: an E2 to E1 write
    }
    const MemStage::DrawFacts draw_facts =
        pipeline != nullptr ? pipeline_facts
                            : MemStage::DrawFacts{
                                  .uniform_masks = &facts.uniform_masks,
                                  .uniform_sizes = &facts.uniform_sizes,
                                  .stage_infos = nullptr,
                                  .stages = facts.stages,
                                  .has_texel_buffers = false,
                                  .uses_descriptor_buffer = facts.uses_descriptor_buffer,
                                  .hashes = facts.hashes,
                                  .staged = &facts,
                              };
    // The draw's raw handles at E's position (SEAM 3): R's Configure takes them from the packet.
    const bool null_pointer_handles = VideoCore::Nx2Debug::Experiment(21) != 0;
    handle_values.clear();
    // FRITCHII_ESPLIT_COUNT=3: the bindings from E2's view (FillHandleValuesKept, the handle
    // program keep, still reads the engine's).
    const auto& stages =
        esplit_view != nullptr ? esplit_view->state.shader_stages : engine.state.shader_stages;
    using Cbufs = std::remove_cvref_t<decltype(stages[0].const_buffers)>;
    const std::span<const Shader::Info> handle_infos =
        pipeline != nullptr ? std::span<const Shader::Info>{pipeline_facts.stage_infos, 5}
                            : std::span<const Shader::Info>{*facts.infos};
    // fable_b 23.1 cut 4, dry (GPU profile on): the fill timed and the draw classed.
    const bool price_handles =
        !esplit_async && VideoCommon::FritchiiPrice::On() && pipeline != nullptr;
    const u64 handle_start = price_handles ? VideoCommon::FritchiiPrice::Ticks() : 0;
    std::optional<Profile::FineScope> handles_zone{std::in_place, Profile::Zone::DrawHandles};
    // FRITCHII_HANDLE_PROG_KEEP (fable_b 25.2 cut g): a pipeline with handle programs keeps their
    // results while every word they read is unchanged.
    if (pipeline == nullptr || pipeline_facts.reads_handles) {
        const int program_keep =
            pipeline != nullptr
                ? VideoCommon::FillCuts::Mode(VideoCommon::FillCuts::Cut::HandleProgKeep)
                : 0;
        if (program_keep != 0 &&
            std::ranges::any_of(handle_infos, [](const Shader::Info& info) {
                return !info.handle_programs.empty();
            })) [[unlikely]] {
            FillHandleValuesKept(engine, handle_infos, pipeline, facts.hashes,
                                 null_pointer_handles, program_keep);
        } else {
            VideoCommon::FillGraphicsHandleValues(
                handle_infos, engine.GetMemoryManager(),
                [&stages](size_t stage) {
                    return VideoCommon::GraphicsHandleCbufs<Cbufs>{stages[stage].const_buffers};
                },
                r.device.GetStorageBufferAlignment(), null_pointer_handles,
                [this](u32 raw) { handle_values.push_back(raw); });
        }
    }
    handles_zone.reset(); // helper CP: "draw handle values"
    if (price_handles) [[unlikely]] {
        PriceHandleValues(engine, handle_infos, pipeline,
                          VideoCommon::FritchiiPrice::Ticks() - handle_start);
    }
    buffer_number = r.mem_stage.NextRingPacketIndex();
    // FRITCHII_FILL_BRACKET_LEAN (fable_b 25.2 cut h): the tracker record slots R can read on
    // this draw, from E's registers. Slot 0: TrackOcarinaCamera reads it only when the first color
    // target's view is B10G11R11_FLOAT, which only that render target format gives, so one of
    // the eight targets has it. Slots 1 and 2: TrackOcarinaActor reads them only for an indexed
    // draw of the actor's vertex program (the pipeline's unique_hashes[1], facts.hashes[0]).
    // FillRingDraw's end sets the cut back to 0.
    if (const int bracket_lean =
            VideoCommon::FillCuts::Mode(VideoCommon::FillCuts::Cut::BracketLean);
        bracket_lean != 0) [[unlikely]] {
        u8 slots = 0;
        const auto& targets =
            esplit_async ? VideoCommon::ESplit::ThreadView().regs.rt : engine.regs.rt;
        for (const auto& rt : targets) {
            if (rt.format == Tegra::RenderTargetFormat::B10G11R11_FLOAT) {
                slots |= 1;
                break;
            }
        }
        if (is_indexed && facts.hashes[0] == MemStage::TrackerRecord::ActorShader) {
            slots |= 6;
        }
        r.mem_stage.SetRingLean(bracket_lean, slots);
        VideoCommon::FillCuts::MaybeReport();
    }
    const bool filled =
        r.mem_stage.FillRingDraw(r.buffer_cache, draw_facts, is_indexed, position, seq, 0, nullptr,
                                 nullptr, &handle_values, null_pointer_handles, list_args);
    // FRITCHII_ONEF_MESH_EMU: a lent emulated mesh draw leaves VertexBuffer0..31 raised on E's
    // engine (InvalidateInputState after its compute passes: the next draw binds its vertex
    // buffers again). Posted, R raises them on its view only, and they are E's flags: E raises
    // them here, where the lent draw left them, before the split's view takes the flags.
    if (filled && pipeline != nullptr && pipeline->IsEmulatedMesh()) [[unlikely]] {
        auto& dirty = r.buffer_cache.Inputs().Dirty();
        for (u32 index = 0; index < Maxwell3D::Regs::NumVertexArrays; ++index) {
            dirty[VideoCommon::Dirty::VertexBuffer0 + index] = true;
        }
    }
    if (esplit_records) [[unlikely]] {
        if (esplit_thread) {
            VideoCommon::ESplit::FillDone(engine, esplit_view);
        } else {
            VideoCommon::ESplit::AfterFill(engine, esplit_view);
        }
    }
    if (filled) {
        // For ReportDescriptorMismatch: what this fill used, before the push publishes it.
        FillRecord& record = fill_records[seq % FillRecords];
        record.seq = seq;
        record.key = program_id.key;
        record.space = program_id.space;
        record.code = program_id.code;
        record.words = program_id.words;
        record.hashes = facts.hashes;
        record.uniform_masks = facts.uniform_masks;
        record.ssbo_counts = facts.ssbo_counts;
        record.stages = facts.stages;
        record.buffer_number = buffer_number;
    }
    return filled;
}

void OneFollowerDraw::FillHandleValuesKept(Maxwell3D& engine, std::span<const Shader::Info> infos,
                                           const void* pipeline, const std::array<u64, 2>& hashes,
                                           bool null_pointer_handles, int mode) {
    namespace FillCuts = VideoCommon::FillCuts;
    using Record = VideoCommon::HandleProgramRecord;
    using Verdict = VideoCommon::HandleProgramVerdict;
    FillCuts::ProgramStats& stats = FillCuts::g_program;
    FillCuts::Add(stats.draws);
    FillCuts::MaybeReport();
    ProgramMemo& memo = program_memos[pipeline];
    if (memo.hashes != hashes) {
        memo.hashes = hashes;
        memo.records.clear();
    }
    auto& memory = engine.GetMemoryManager();
    // FRITCHII_ESPLIT_DEPTH > 0: an async fill reads E2's view (E1's engine is ahead).
    const auto& stages = VideoCommon::ESplit::FillAsync()
                             ? VideoCommon::ESplit::ThreadView().state.shader_stages
                             : engine.state.shader_stages;
    using Cbufs = std::remove_cvref_t<decltype(stages[0].const_buffers)>;
    // The evaluation RawTextureHandle (count 1) or RawTextureHandleTable (a table) makes, through
    // `mem` and `cbufs`, each value passed to out(raw).
    const auto evaluate = [](const Shader::Info& info, u32 program, auto& mem, auto& cbufs,
                             u32 element, u32 count, u64 alignment, auto&& out) {
        if (count == 1) {
            out(VideoCommon::EvaluateTextureHandle(info, program, mem, cbufs, alignment, element));
        } else {
            VideoCommon::EvaluateTextureHandles(info, program, mem, cbufs, alignment, count,
                                                [&out](u32, u32 raw) { out(raw); });
        }
    };
    size_t cursor = 0;
    const auto program = [&](const Shader::Info& info, u32 program_index, const auto& cbufs,
                             u32 element, u32 count, u64 alignment) {
        FillCuts::Add(stats.programs);
        if (cursor >= memo.records.size()) {
            memo.records.emplace_back();
        }
        Record& record = memo.records[cursor++];
        const std::vector<u32>& code = info.handle_programs[program_index];
        Verdict verdict = Verdict::First;
        if (record.valid && record.code == code.data() && record.code_size == code.size() &&
            record.element == element && record.count == count) {
            const bool time_check = stats.check_sample.Take();
            const u64 check_start = time_check ? FillCuts::Ticks() : 0;
            verdict = VideoCommon::CheckHandleProgramRecord(record, memory, cbufs);
            if (time_check) [[unlikely]] {
                FillCuts::Add(stats.check_ticks, FillCuts::Ticks() - check_start);
            }
        }
        switch (verdict) {
        case Verdict::Same:
            FillCuts::Add(stats.same);
            break;
        case Verdict::Moved:
            FillCuts::Add(stats.moved);
            break;
        case Verdict::Changed:
            FillCuts::Add(stats.changed);
            break;
        case Verdict::First:
            FillCuts::Add(stats.first);
            break;
        }
        if (verdict == Verdict::Same) {
            if (mode == FillCuts::On) {
                // Every read the evaluation makes reads what it read when recorded.
                handle_values.insert(handle_values.end(), record.results.begin(),
                                     record.results.end());
                return;
            }
            // 2 and 3: evaluated as before (its time is what 1 removes) and compared.
            const bool time_hit = stats.hit_sample.Take();
            const u64 hit_start = time_hit ? FillCuts::Ticks() : 0;
            const size_t first = handle_values.size();
            evaluate(info, program_index, memory, cbufs, element, count, alignment,
                     [this](u32 raw) { handle_values.push_back(raw); });
            if (time_hit) [[unlikely]] {
                FillCuts::Add(stats.hit_ticks, FillCuts::Ticks() - hit_start);
            }
            FillCuts::Add(stats.compare.checked);
            const std::span<const u32> now{handle_values.data() + first,
                                           handle_values.size() - first};
            if (!std::ranges::equal(now, record.results) && stats.compare.Differ()) [[unlikely]] {
                LOG_ERROR(Render_Vulkan,
                          "Handle program keep (FRITCHII_HANDLE_PROG_KEEP={}): program {} element "
                          "{} count {}: same {} words and {} constant buffer addresses, kept "
                          "{:#x}, evaluated {:#x}",
                          mode, program_index, element, count, record.num_reads, record.num_bases,
                          record.results.empty() ? 0U : record.results[0],
                          now.empty() ? 0U : now[0]);
            }
            return;
        }
        // Recorded again: the evaluation reads through the recording memory and addresses.
        record.Begin(code.data(), code.size(), element, count);
        VideoCommon::HandleRecordingMemory<std::remove_reference_t<decltype(memory)>> recording{
            memory, record};
        VideoCommon::HandleRecordingCbufs<std::remove_cvref_t<decltype(cbufs)>> recording_cbufs{
            cbufs, record};
        evaluate(info, program_index, recording, recording_cbufs, element, count, alignment,
                 [this, &record](u32 raw) {
                     handle_values.push_back(raw);
                     record.results.push_back(raw);
                 });
    };
    VideoCommon::FillGraphicsHandleValuesWith(
        infos, memory,
        [&stages](size_t stage) {
            return VideoCommon::GraphicsHandleCbufs<Cbufs>{stages[stage].const_buffers};
        },
        rasterizer.device.GetStorageBufferAlignment(), null_pointer_handles,
        [this](u32 raw) { handle_values.push_back(raw); }, program);
}

void OneFollowerDraw::PriceHandleValues(Maxwell3D& engine, std::span<const Shader::Info> infos,
                                        const void* pipeline, u64 ticks) {
    using Verdict = VideoCommon::FritchiiPrice::HandleValues;
    // The constant buffer words each texture and image descriptor reads (as ConfigureImpl's
    // price): a key of their addresses, the write generations of their pages, and whether the
    // buffer cache's tracker holds any of them CPU-modified (a guest store there raises no flag).
    auto& memory = engine.GetMemoryManager();
    u64 key = 0xcbf29ce484222325ULL;
    u64 sum = 0;
    bool keyed = true;
    bool cpu_modified = false;
    const auto add = [&](GPUVAddr begin, u64 bytes) {
        key = (key ^ begin) * 0x100000001b3ULL;
        key = (key ^ bytes) * 0x100000001b3ULL;
        const std::optional<DAddr> device =
            begin != 0 ? memory.GpuToCpuAddress(begin) : std::nullopt;
        const u64 part = device ? Common::PriceGenSum(*device, bytes) : ~u64{0};
        sum = part == ~u64{0} || sum == ~u64{0} ? ~u64{0} : sum + part;
        if (device && rasterizer.buffer_cache.IsRegionCpuModified(*device, bytes)) {
            cpu_modified = true;
        }
    };
    for (size_t stage = 0; stage < infos.size() && keyed; ++stage) {
        const Shader::Info& info = infos[stage];
        const auto& cbufs = engine.state.shader_stages[stage].const_buffers;
        const auto base = [&cbufs](u32 index) -> GPUVAddr {
            return index < cbufs.size() && cbufs[index].enabled ? cbufs[index].address
                                                                : GPUVAddr{};
        };
        const auto range = [&](const auto& desc) {
            if (desc.handle_program != ~0U) {
                keyed = false;
                return;
            }
            if (desc.count == 0) {
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
    HandleMemo& memo = handle_memos[pipeline];
    const bool same_values = memo.valid && memo.values == handle_values;
    const bool same_range = memo.valid && memo.key == key;
    const bool unflagged = same_range && sum != ~u64{0} && memo.sum == sum;
    Verdict verdict;
    if (!keyed) {
        verdict = Verdict::Program;
    } else if (!same_values) {
        verdict = unflagged && !cpu_modified ? Verdict::KeepWouldDiffer : Verdict::Changed;
    } else if (!same_range) {
        verdict = Verdict::SameMoved;
    } else if (!unflagged) {
        verdict = Verdict::SameFlagged;
    } else {
        verdict = cpu_modified ? Verdict::SameCpuModified : Verdict::SameKeep;
    }
    memo.values = handle_values;
    memo.key = key;
    memo.sum = sum;
    // The generations count only once they are on (NoteHandleValues turns them on): a memo
    // taken before compares nothing.
    memo.valid = keyed && Common::g_price_gens_on.load(std::memory_order_relaxed);
    VideoCommon::FritchiiPrice::NoteHandleValues(verdict, ticks);
}

void OneFollowerDraw::LendDraw(bool is_indexed, u32 instance_count, LentReason reason,
                               bool verify, bool list) {
    // The facts compare (Check::FactsStale): with the verify, the facts table entry E's fill
    // would use for these program registers now, before the lent draw publishes its own.
    FactsExpect expect{};
    if (verify) {
        expect.reset = std::exchange(verify_reset, ResetCheck{});
    }
    // With rule 3 (FRITCHII_ONEF_E_PIPELINE) E fills from the pipeline it looked up and the facts
    // table plays no part: nothing to compare.
    if (verify && !VideoCommon::OneFollower::g_e_pipeline) {
        VideoCommon::MemStageShadow::ProgramId program_id;
        VideoCommon::MemStageShadow::BackProgramId(rasterizer.channel_state->maxwell3d,
                                                   program_id);
        MemStage::StagedFacts looked{};
        if (rasterizer.mem_stage.LookupFacts(program_id, looked)) {
            expect.found = true;
            expect.hashes = looked.hashes;
            expect.space = program_id.space;
        }
    }
    auto body = [this, is_indexed, instance_count, verify, expect, list] {
        RunLent(is_indexed, instance_count, verify, expect, list);
    };
    OneF::Lend(LentOp::Draw, reason, body);
}

bool OneFollowerDraw::AsyncTailFits(const Maxwell3D& engine,
                                    const GraphicsPipeline* pipeline) const {
    // FRITCHII_ESPLIT_DEPTH > 0 (fritchii_e_split_thread.cpp, S2): what an async fill needs, all
    // known on E1. Anything else goes lockstep.
    if (!VideoCommon::ESplit::g_async || pipeline == nullptr) {
        return false; // S1, or no rule 3 pipeline (the facts table's lookup reads the engine)
    }
    // FRITCHII_ONEF_MESH_EMU: an emulated mesh draw fits too. Its fill raises VertexBuffer0..31
    // on the cache's inputs (E2's view flags when async: E2 owns the buffer flags in the split).
    // FRITCHII_ONEF_MESH_EMU_ASYNC=0: lockstep tails for them.
    static const bool mesh_async = [] {
        const char* const value = std::getenv("FRITCHII_ONEF_MESH_EMU_ASYNC");
        return value == nullptr || value[0] != '0';
    }();
    if (!mesh_async && pipeline->IsEmulatedMesh()) {
        return false;
    }
    const MemStage::DrawFacts pipeline_facts = pipeline->MemFacts();
    if (pipeline_facts.has_texel_buffers || pipeline_facts.stage_infos == nullptr) {
        return false; // FillRing would refuse (LateFill)
    }
    auto& r = rasterizer;
    if (!r.mem_stage.Packets() || !r.mem_stage.ValuesByValue() || r.buffer_cache.MarksAhead()) {
        return false;
    }
    if (!engine.draw_manager.draw_state.inline_index_draw_indexes.empty()) {
        return false; // the index bytes live on E1
    }
    // The step-in of 15 Nov (profiled): FritchiiPrice::On() is the GPU profile itself, so every
    // draw stayed lockstep. An async fill skips the price timers (they read the engine) and the
    // handle program keep reads the view's bindings (FillHandleValuesKept): neither blocks.
    return !VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed);
}

bool OneFollowerDraw::FillAndPush(Maxwell3D& engine, bool is_indexed, u64 seq, bool list,
                                  LentReason& reason) {
    // FRITCHII_ONEF_NULL_FILL=1 (test only, the picture breaks; the front split's front-wall
    // probe): no flush, fill or push, R draws nothing; TickWork as after a push. Off: one
    // relaxed load.
    if (VideoCommon::FrontSplit::NullFill()) [[unlikely]] {
        rasterizer.gpu.TickWork();
        return true;
    }
    // FRITCHII_ESPLIT_COUNT: E2's parts of the draw carved out of E1's (g_state read once).
    namespace ESplit = VideoCommon::ESplit;
    const u8 esplit = ESplit::g_state;
    // FRITCHII_ESPLIT=1 (fritchii_e_split_thread.h): here on E1, the flush and the tail run on E2.
    const bool esplit_hand = ESplit::ShouldHand();
    // S2: E2 is behind; the flush rides at the head of the tail (after E1's lookup).
    bool esplit_late_flush = false;
    if (esplit_hand) [[unlikely]] {
        ESplit::ApplyBackRaises(engine); // S2: VertexInput raised by E2's async fills
        auto& memory_manager = engine.GetMemoryManager();
        switch (ESplit::FlushPlace()) {
        case ESplit::FlushAt::Before:
            // E2 idle: the flush before the lookup, as below; only when it has something to do
            // (an empty FlushCaching changes nothing; the memory stage shadow compares even then).
            if (memory_manager.HasPendingCachedWrites() ||
                VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) {
                if (ESplit::g_async) {
                    // The queued mode: E2 is idle, E1 runs it (a join, no hand-over).
                    const ESplit::JoinScope join{ESplit::Join::Flush};
                    memory_manager.FlushCaching();
                } else {
                    auto flush = [&memory_manager] { memory_manager.FlushCaching(); };
                    ESplit::HandTo(ESplit::HandInfo{.hand = ESplit::Hand::Flush}, flush);
                }
            }
            break;
        case ESplit::FlushAt::Drained: {
            // S4: owed with E2 behind: E2 drains, then the flush (when it has work). The queued
            // mode: E1 runs it in the join that drains E2 (no round trip).
            const ESplit::JoinScope join{ESplit::Join::Flush};
            if (memory_manager.HasPendingCachedWrites() ||
                VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) {
                memory_manager.FlushCaching();
            }
            break;
        }
        case ESplit::FlushAt::Tail:
            esplit_late_flush = true;
            break;
        }
    } else {
        // Today's PrepareDraw head, E's half: the stream's invalidations before the fill.
        const ESplit::Carve flush_carve{ESplit::Class::E2, esplit};
        const ESplit::FlagProbe flush_probe{engine, ESplit::FlagSite::StreamFlush, esplit};
        const Profile::FineScope flush_zone{Profile::Zone::DrawFlush}; // helper CP
        engine.GetMemoryManager().FlushCaching();
    }
    // Rule 3 (FRITCHII_ONEF_E_PIPELINE): E looks the pipeline up itself, at its own position,
    // on its own engine (the pipeline cache never names R's view then). A miss, or a build
    // not ready, is known here: the draw is lent. The lookup consumes pipeline-key flags on
    // E's engine that R's dynamic state still reads (VertexInput and the like): they are
    // raised again, so the pack below hands them to R and clears them (rule 1).
    GraphicsPipeline* pipeline = nullptr;
    if (VideoCommon::OneFollower::g_e_pipeline) {
        // FRITCHII_ESPLIT: E1's lookup reads shader code from game memory, allowed unless a write
        // onto shader pages is pending (then E2 drains and the flush runs first: a join).
        const ESplit::LookupScope esplit_lookup;
        const ESplit::FlagProbe lookup_probe{engine, ESplit::FlagSite::Lookup, esplit};
        const auto flags_before = engine.dirty.flags;
        {
            const Profile::FineScope lookup_zone{Profile::Zone::PipelineLookup}; // helper CP
            pipeline = rasterizer.pipeline_cache.CurrentGraphicsPipeline();
        }
        engine.dirty.flags |= flags_before;
        ESplit::NoteLookup(esplit); // E1's lookup: after an E2 shader page write it would wait
        // fable_b 18.4, must be 0 (FRITCHII_ONEF_LOOKUP_CHECK, every Nth unlent draw): the parts
        // the lookup kept on a clear flag, against the same parts fresh.
        if (pipeline != nullptr && OneF::LookupCheckDue()) [[unlikely]] {
            const bool shaders_clear = !flags_before[VideoCommon::Dirty::Shaders];
            const bool blending_clear = !flags_before[Dirty::Blending];
            const bool vertex_input_clear = !flags_before[Dirty::VertexInput];
            if (const u32 differ = rasterizer.pipeline_cache.OneFollowerLookupCheck(
                    shaders_clear, blending_clear, vertex_input_clear);
                differ != 0) [[unlikely]] {
                thread_local std::string note;
                note = fmt::format("unlent draw {}: kept on a clear flag, differs from the "
                                   "registers",
                                   seq);
                if ((differ & 1) != 0) {
                    OneF::Note(Check::LookupPrograms, note.c_str());
                }
                if ((differ & 2) != 0) {
                    OneF::Note(Check::LookupBlend, note.c_str());
                }
                if ((differ & 4) != 0) {
                    OneF::Note(Check::LookupVertexInput, note.c_str());
                }
            }
        }
        if (pipeline == nullptr) {
            reason = LentReason::PipelineMiss;
            return false;
        }
        // UNLEND's Mesh bit covers native task/mesh pipelines only (RouteDraw tests the device;
        // this is the pipeline's own word for it).
        // FRITCHII_ONEF_MESH_EMU: emulated mesh pipelines are posted too (not list draws). A
        // mesh pipeline whose emulation was refused (no VkPipeline: R drops the draw) stays lent,
        // as before, so the skip counters keep their meaning.
        if (pipeline->IsEmulatedMesh() ? (list || !OneF::EmulatedMeshPostable())
                                       : (pipeline->IsMeshPipeline() &&
                                          !rasterizer.device.IsExtMeshShaderSupported()))
            [[unlikely]] {
            reason = LentReason::NoPath;
            return false;
        }
        // FRITCHII_ONEF_OCCL (OneFollower::OcclWanted: unset = AMD's proprietary driver on a
        // discrete GPU) with FRITCHII_OCCL_FLAG: E picks the draw's flag variant at its lookup
        // (the pipeline cache and the registers are E's); R records it as the lent path does
        // (the slot, the address push). Variants drawn in place, and with
        // FRITCHII_ONEF_OCCL_PREPASS the pre-pass ones (colour or discard: R records the draw
        // twice, as R's query state decides).
        // FillAndPush's callers are RouteDraw (direct draws) and RouteListDraw (list draws) only:
        // a non-list indirect draw is always lent, so it reaches the lent path's own choice,
        // where the pre-pass refuses indirect draws (review C8). Only list draws need excluding
        // here (their arguments are consumed once).
        push_occl = 0;
        if (VideoCommon::OneFollower::OcclWanted() &&
            VideoCommon::OcclFlag::Mode() != VideoCommon::OcclFlag::Off &&
            rasterizer.query_cache_runtime.OcclFlagAreaExists()) {
            VideoCommon::OcclFlag::t_colour_prepass = false;
            VideoCommon::OcclFlag::t_discard_direct = false;
            VideoCommon::OcclFlag::t_colour_discard_prepass = false;
            GraphicsPipeline* const flag =
                rasterizer.pipeline_cache.OcclFlagPipeline(pipeline, engine.regs, true);
            const bool colour_prepass = VideoCommon::OcclFlag::t_colour_prepass;
            const bool discard_direct = VideoCommon::OcclFlag::t_discard_direct;
            const bool colour_discard_prepass = VideoCommon::OcclFlag::t_colour_discard_prepass;
            const bool prepass_kind = colour_prepass || colour_discard_prepass ||
                                      (flag != nullptr && flag->IsOcclDiscardPrepass());
            VideoCommon::OcclFlag::t_colour_prepass = false;
            VideoCommon::OcclFlag::t_discard_direct = false;
            VideoCommon::OcclFlag::t_colour_discard_prepass = false;
            // FRITCHII_ONEF_OCCL_PREPASS (default 1, so it follows FRITCHII_ONEF_OCCL; 0 keeps
            // played draws to in-place variants): a pre-pass variant too (the colour or
            // discard pre-pass records the draw twice on R, decided there by R's query state as
            // the lent and off paths decide it); its pre-pass state goes in the packet. Without
            // it every such draw counted with a host query (~16k more a second at the fountain,
            // ~1.2 ms of GPU a frame on the eGPU). Not for list draws (recording them twice
            // consumes their arguments).
            static const bool onef_prepass = [] {
                const char* value = std::getenv("FRITCHII_ONEF_OCCL_PREPASS");
                return value == nullptr || value[0] != '0';
            }();
            if (flag != nullptr && flag != pipeline) {
                if (!prepass_kind) {
                    pipeline = flag;
                } else if (onef_prepass && !list) {
                    pipeline = flag;
                    if (flag->occl_flag_parent == nullptr) [[unlikely]] {
                        LOG_ERROR(Render_Vulkan, "One follower: E's pre-pass variant has no "
                                                 "flag parent (kind {})",
                                  flag->occl_flag_kind);
                    }
                    push_occl = static_cast<u8>(
                        (colour_prepass ? OneF::OcclColourPrepass : 0) |
                        (discard_direct ? OneF::OcclDiscardDirect : 0) |
                        (colour_discard_prepass ? OneF::OcclColourDiscardPrepass : 0));
                }
            }
        }
    }
    // FRITCHII_OCCL_LEAN (helper CS, fritchii_occl_lean.h): a draw that writes no colour takes
    // the same pipeline without its pixel program when that program can change nothing else; the
    // fill below then has no pixel stage and R binds no pixel textures. Off: one relaxed load.
    VideoCommon::OcclLean::DrawNote occl_note{};
    if (pipeline != nullptr) {
        if (const int occl_mode = VideoCommon::OcclLean::Mode(); occl_mode != 0) [[unlikely]] {
            pipeline = rasterizer.pipeline_cache.OcclLeanPipeline(pipeline, engine.regs,
                                                                  occl_mode, occl_note);
        }
    }
    const u64 position = VideoCommon::GpuSplit::Get().BackEntryPosition();
    if (esplit_hand && AsyncTailFits(engine, pipeline)) [[unlikely]] {
        // FRITCHII_ESPLIT_DEPTH > 0: the tail on E2 while E1 goes on. E1 builds the record
        // (taking its engine's buffer flags) and packs R's packet now; E2 fills from its view and
        // pushes the packet after the fill.
        ESplit::AsyncSlot slot{};
        if (ESplit::BeginAsyncDraw(engine, list, slot)) {
            struct AsyncTail {
                OneFollowerDraw* self;
                Maxwell3D* engine;
                GraphicsPipeline* pipeline;
                u8* packet;
                u64 position;
                u64 seq;
                u64 packet_bytes;
                u64 e2c_seq;
                OneF::PacketHeader header;
                bool is_indexed;
                bool list;
                bool flush;
            } tail{this, &engine, pipeline, slot.packet, position, seq, 0,
                   engine.PendingAsyncE2cSeq(), {}, is_indexed, list, esplit_late_flush};
            static_assert(sizeof(AsyncTail) <= ESplit::AsyncPayloadBytes);
            static_assert(std::is_trivially_copyable_v<AsyncTail>);
            tail.packet_bytes =
                OneF::PackDraw(engine, seq, push_flags, pipeline, tail.header, slot.packet);
            tail.header.reserved[1] = push_occl;
            std::memcpy(slot.payload, &tail, sizeof(tail));
            ESplit::PostAsync([](void* raw) {
                AsyncTail t;
                std::memcpy(&t, raw, sizeof(t));
                OneFollowerDraw& self = *t.self;
                if (t.e2c_seq != 0 && !t.engine->E2AsyncShouldExecute()) {
                    t.header.kind = OneF::PacketKind::Delta;
                    OneF::PushDrawPacked(t.header, t.packet, t.packet_bytes, 0);
                    return;
                }
                std::optional<VideoCommon::CacheInputs> inputs;
                auto& cache = self.rasterizer.buffer_cache;
                SCOPE_EXIT {
                    if (inputs) {
                        cache.SetInputsOverride(nullptr);
                    }
                };
                if (t.flush) {
                    auto& memory_manager = t.engine->GetMemoryManager();
                    if (memory_manager.HasPendingCachedWrites() ||
                        VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) {
                        VideoCommon::ESplit::NoteLateFlush();
                        // FRITCHII_ESPLIT_OWED's ESplitLateShader (sampled): nothing this flush
                        // invalidates after E1's lookup is shader code.
                        VideoCommon::ESplit::CheckLateFlush(&memory_manager);
                        // S4: the flush's buffer marks go to E2's flags (the view, at this draw
                        // after its record), never to E1's engine, which has moved on. No cache
                        // lock around it: the invalidation may be lent, and R takes that lock (R
                        // runs it as E, so it marks the view too).
                        VideoCommon::ESplit::View& view = VideoCommon::ESplit::ThreadView();
                        if (view.seeded && view.engine == t.engine) {
                            inputs.emplace(cache.StageInputs(
                                view.regs, view.state, view.root_tables, view.draw_state,
                                view.flags, t.list ? &view.indirect : nullptr));
                            cache.SetInputsOverride(&*inputs);
                        }
                        memory_manager.FlushCaching();
                    }
                }
                u64 number = 0;
                LentReason refused{};
                if (!self.FillRing(*t.engine, t.pipeline, t.is_indexed, t.position, t.seq, number,
                                   refused, t.list)) {
                    // E1 moved on: no lend now. R still gets E1's spans (as a delta packet);
                    // the draw is dropped (E2Refused, must be 0).
                    VideoCommon::ESplit::NoteAsyncRefused();
                    t.header.kind = OneF::PacketKind::Delta;
                    OneF::PushDrawPacked(t.header, t.packet, t.packet_bytes, 0);
                    return;
                }
                OneF::PushDrawPacked(t.header, t.packet, t.packet_bytes, number);
                self.counts.pushed.store(self.counts.pushed.load(std::memory_order_relaxed) + 1,
                                         std::memory_order_relaxed);
                self.rasterizer.gpu.TickWork();
            });
            return true;
        }
    }
    if (esplit_hand && !ESplit::g_async) [[unlikely]] {
        // FRITCHII_ESPLIT_DEPTH=0 (S1): the tail below on E2 (the record of its view first), E1
        // waits.
        bool pushed = false;
        auto tail = [&] {
            u64 number = 0;
            if (!FillRing(engine, pipeline, is_indexed, position, seq, number, reason, list)) {
                return;
            }
            OneF::PushDraw(engine, seq, number, push_flags, pipeline, push_occl);
            counts.pushed.store(counts.pushed.load(std::memory_order_relaxed) + 1,
                                std::memory_order_relaxed);
            rasterizer.gpu.TickWork();
            pushed = true;
        };
        auto& cache_mutex = rasterizer.buffer_cache.mutex;
        ESplit::HandTo(ESplit::HandInfo{.hand = ESplit::Hand::DrawTail,
                                        .record = true,
                                        .draw = true,
                                        .list_args = list,
                                        .held = cache_mutex.IsHeldByCurrentThread() ? &cache_mutex
                                                                                    : nullptr},
                       tail);
        return pushed;
    }
    // FRITCHII_ESPLIT's queued mode: a tail that cannot go async runs here, E2 drained (a join);
    // a flush E2 was behind for runs first.
    const ESplit::JoinScope esplit_tail{ESplit::Join::Tail, esplit_hand};
    if (esplit_hand) [[unlikely]] {
        ESplit::NoteTailStayed(0);
        if (esplit_late_flush) {
            auto& memory_manager = engine.GetMemoryManager();
            if (memory_manager.HasPendingCachedWrites() ||
                VideoCommon::MemStageShadow::g_on.load(std::memory_order_relaxed)) {
                ESplit::NoteLateFlush();
                ESplit::CheckLateFlush(&memory_manager);
                memory_manager.FlushCaching();
            }
        }
    }
    u64 buffer_number = 0;
    // The fill, ring B's packet, R's packet (its span packing is E1's, carved in PushDraw) and
    // TickWork: E2's.
    const ESplit::Carve fill_carve{ESplit::Class::E2, esplit};
    // FRITCHII_OCCL_LEAN's timer (1 in 16 of the draws the cut can take): the fill and the push.
    const u64 occl_start = occl_note.timed ? VideoCommon::OcclLean::NowNs() : 0;
    {
        // Helper CP: FillRing's own work; the handle values and the fill ("memory fill" and
        // its fine zones) are nested zones of their own.
        const Profile::FineScope setup_zone{Profile::Zone::DrawFillSetup};
        if (!FillRing(engine, pipeline, is_indexed, position, seq, buffer_number, reason,
                      list)) {
            return false;
        }
    }
    {
        const Profile::FineScope push_zone{Profile::Zone::DrawPush}; // helper CP
        OneF::PushDraw(engine, seq, buffer_number, push_flags, pipeline, push_occl);
    }
    if (occl_note.timed) [[unlikely]] {
        VideoCommon::OcclLean::NoteFill(occl_note, VideoCommon::OcclLean::NowNs() - occl_start);
    }
    // E is its only writer: a relaxed load and store, no locked instruction per draw.
    counts.pushed.store(counts.pushed.load(std::memory_order_relaxed) + 1,
                        std::memory_order_relaxed);
    // PrepareDraw's tail, E's half (read-backs wait on E's work).
    {
        const ESplit::FlagProbe tick_probe{engine, ESplit::FlagSite::TickWork, esplit};
        const Profile::FineScope tick_zone{Profile::Zone::TickWork}; // helper CP
        rasterizer.gpu.TickWork();
    }
    ESplit::NoteDrawFilled(esplit);
    return true;
}

bool OneFollowerDraw::RouteDraw(bool is_indexed, u32 instance_count) {
    const StepLevel step = OneF::Step();
    if (step < StepLevel::Build) {
        return false; // steps 0 to 2: today's path (part a lends it)
    }
    if (OneF::OnRenderThread()) {
        if (nested || playing != nullptr) {
            return false; // RunLent's or PlayDraw's own call of Draw
        }
        // A lent call from elsewhere (part a) runs this draw on R with E parked.
        RunLent(is_indexed, instance_count, step == StepLevel::Build, FactsExpect{});
        return true;
    }
    // E. The engine is the channel's (the rasterizer's own pointer names the view while R plays).
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    const u64 seq = ++e_draw_seq;
    const u32 kind = Classify(engine, is_indexed, instance_count);
    // The draw span's caller-known values (part a/b: the packet's kind for the counters, and
    // `indexed` and `instance_arg`, which PlayDraw reads): set for whichever packet comes next.
    OneF::SetNextDraw(kind, is_indexed, false, instance_count);
    const auto skip_suppressed_e2c = [&] {
        if (!engine.HasPendingAsyncE2c()) {
            return false;
        }
        engine.WaitAndSyncAsyncE2c();
        if (engine.ShouldExecute()) {
            return false;
        }
        OneF::ClearNextDraw();
        return true;
    };
    if (step < StepLevel::Unlend) {
        if (skip_suppressed_e2c()) {
            return true;
        }
        LendDraw(is_indexed, instance_count, LentReason::StepBelowUnlend, true);
        return true;
    }
    const u32 mask = static_cast<u32>(VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_onef_unlend, "FRITCHII_ONEF_UNLEND"));
    if ((mask & kind & PlayableKinds) == 0) {
        if (skip_suppressed_e2c()) {
            return true;
        }
        LendDraw(is_indexed, instance_count, LentReason::KindNotInMask, false);
        return true;
    }
    filled_refused = false;
    LentReason reason;
    {
        const Profile::FineScope route_zone{Profile::Zone::DrawRoute}; // helper CP
        reason = UnlentBlocker(engine, is_indexed, instance_count, kind);
    }
    // Mesh: native task/mesh pipelines only (the emulation's buffers are not R's alone). Without
    // VK_EXT_mesh_shader every mesh pipeline is emulated or skipped.
    if (reason == LentReason::Count && kind == OneF::Mesh &&
        !rasterizer.device.IsExtMeshShaderSupported() && !OneF::EmulatedMeshPostable()) {
        reason = LentReason::NoPath;
    }
    if (reason == LentReason::Count && FillAndPush(engine, is_indexed, seq, false, reason)) {
        return true;
    }
    if (skip_suppressed_e2c()) {
        return true;
    }
    if (reason == LentReason::Transient && filled_refused) {
        OneF::CountTransient(fill_test, kind); // FillRing's refusal (the dry counter)
    }
    LendDraw(is_indexed, instance_count, reason, reason == LentReason::Verify);
    return true;
}

bool OneFollowerDraw::RouteListDraw() {
    if (OneF::Step() != StepLevel::Unlend || OneF::OnRenderThread()) {
        return false; // below step 4: the caller lends it as before; on R it runs where it is
    }
    const u32 mask = static_cast<u32>(VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_onef_unlend, "FRITCHII_ONEF_UNLEND"));
    if ((mask & OneF::List) == 0) {
        return false; // the caller lends it as before (KindNotInMask)
    }
    // E. The engine is the channel's (the rasterizer's own pointer names the view while R plays).
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    const auto& indirect = engine.draw_manager.indirect_state;
    if (!indirect.list_words) {
        return false;
    }
    const u64 seq = ++e_draw_seq;
    // As OneFollowerDrawIndirect describes it: a list draw is drawn non-indexed (DrawListWords).
    OneF::SetNextDraw(OneF::List, false, true, 0);
    LentReason reason;
    filled_refused = false;
    push_flags = 0;
    verify_reset = {};
    if (engine.GetNativeMacroEngine().native_turing && engine.regs.IsMeshEnabled() &&
        (mask & OneF::Mesh) == 0) {
        reason = LentReason::KindNotInMask; // a list draw on a mesh pipeline needs both bits
    } else if (RasterizerVulkan::OneFollowerListOnCpu(engine)) {
        // DrawListWords reads the two words on the CPU (a flush, game memory): E's work.
        reason = LentReason::NoPath;
    } else {
        reason = UnlentBlocker(engine, false, 0, OneF::List);
    }
    if (reason == LentReason::Count && FillAndPush(engine, false, seq, true, reason)) {
        // R records the kept words when it plays this draw: a read of them on E drains R first.
        if (indirect.list_keep_id != 0) {
            rasterizer.onef_list_keeps_unplayed = true;
        }
        return true;
    }
    if (reason == LentReason::Transient && filled_refused) {
        OneF::CountTransient(fill_test, OneF::List); // FillRing's refusal (the dry counter)
    }
    LendDraw(false, 0, reason, reason == LentReason::Verify, true);
    return true;
}

// --- R: lent draws and the compares ----------------------------------------------------------------

void OneFollowerDraw::RunLent(bool is_indexed, u32 instance_count, bool verify,
                              const FactsExpect& expect, bool list) {
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    nested = true;
    SCOPE_EXIT {
        nested = false;
        if (view != nullptr && view_source == &engine) {
            ResyncFlags(engine);
        }
    };
    if (verify && view != nullptr && view_source == &engine) {
        // The flags compare needs the view's flags re-synced at the last lent operation: a lent
        // clear, query or dispatch in between ran R's code on E's engine, which may raise flags
        // there that the view never sees (not a forwarding miss).
        Compare(engine, OneF::Step() == StepLevel::Build && lent_since_resync == 1);
        if (list) {
            CompareList(engine);
        }
    }
    if (verify && expect.reset.decided) {
        // FRITCHII_ONEF_UNLENT_RESET: E's test before the lend against the test on E's engine at
        // the same place (E parked; the rasterizer names E's engine inside a lent call).
        const bool lent_says = rasterizer.IsNx2IndirectReset(is_indexed, instance_count);
        if (lent_says != expect.reset.is_reset) [[unlikely]] {
            OneF::Note(Check::UnlentReset, lent_says ? "E said no indirect reset, the lent draw is one"
                                                     : "E said indirect reset, the lent draw is not");
            OneF::StopUnlentReset("E's indirect reset test differs from the lent draw's");
        }
    }
    rasterizer.onef_pipeline_seen = false;
    if (list) {
        // A list draw (RouteListDraw): DrawIndirect's list path on E's engine, as the lend of
        // OneFollowerDrawIndirect runs it.
        rasterizer.DrawIndirect();
    } else {
        rasterizer.Draw(is_indexed, instance_count);
    }
    // The programs the draw ran with against the facts E's fill would have used for them: a
    // difference is a stale entry (the 2026-11-14 step 4 failure, caught at step 3).
    if (expect.found && rasterizer.onef_pipeline_seen &&
        expect.hashes != rasterizer.onef_pipeline_hashes) [[unlikely]] {
        thread_local std::string note;
        note = fmt::format("lent draw: the facts entry E would use {:#x}/{:#x}, the pipeline drawn "
                           "{:#x}/{:#x}; E's address space {}, R's {}",
                           expect.hashes[0], expect.hashes[1],
                           rasterizer.onef_pipeline_hashes[0], rasterizer.onef_pipeline_hashes[1],
                           expect.space, rasterizer.gpu_memory->GetID() + 1);
        OneF::Note(Check::FactsStale, note.c_str());
    }
}

void OneFollowerDraw::Compare(Maxwell3D& engine, bool with_flags) {
    counts.verified.fetch_add(1, std::memory_order_relaxed);
    // Part (d)'s coverage count ("build compares" on the verify line): was never fed, so step 3
    // read as vacuous while its compares ran.
    OneF::Stats::BuildCompared();
    const Maxwell& e = engine.regs;
    const Maxwell& v = view->regs;
    // Pipeline: the key each engine gives (pure, full refresh) and what RefreshStages reads.
    const DynamicFeatures& features = rasterizer.pipeline_cache.Features();
    if (!(FullKey(engine, features) == FullKey(*view, features))) {
        OneF::Note(Check::BuildPipeline, "key");
    } else if (!SameBytes(e.pipelines, v.pipelines) ||
               !SameBytes(e.program_region, v.program_region) ||
               e.rasterize_enable != v.rasterize_enable) {
        OneF::Note(Check::BuildPipeline, "programs");
    } else if (e.IsMeshEnabled() != v.IsMeshEnabled()) {
        // UNLEND's Mesh bit: whether RefreshStages takes the task/mesh programs.
        OneF::Note(Check::BuildPipeline, "mesh enable");
    }
    // Render targets: the registers UpdateRenderTargets and the framebuffer read.
    if (!SameBytes(e.rt, v.rt) || !SameBytes(e.rt_control, v.rt_control) ||
        !SameBytes(e.zeta, v.zeta) || e.zeta_enable != v.zeta_enable ||
        !SameBytes(e.zeta_size, v.zeta_size) || !SameBytes(e.surface_clip, v.surface_clip) ||
        e.anti_alias_samples_mode != v.anti_alias_samples_mode) {
        OneF::Note(Check::BuildRenderTargets, "registers");
    }
    // Image views and descriptors: the pools and the sampler binding the handles resolve in (the
    // handles themselves come in the packet).
    if (!SameBytes(e.tex_header, v.tex_header)) {
        OneF::Note(Check::BuildImageViews, "TIC pool");
    }
    if (!SameBytes(e.tex_sampler, v.tex_sampler) || e.sampler_binding != v.sampler_binding) {
        OneF::Note(Check::BuildDescriptors, "TSC pool or sampler binding");
    }
    // Draw parameters: what MakeDrawParams and the key's topology read.
    const auto& ed = engine.draw_manager.draw_state;
    const auto& vd = view->draw_manager.draw_state;
    if (ed.topology != vd.topology || ed.draw_mode != vd.draw_mode ||
        ed.base_instance != vd.base_instance || ed.base_index != vd.base_index ||
        ed.instance_count != vd.instance_count || ed.drawn_instances != vd.drawn_instances ||
        !SameBytes(ed.vertex_buffer, vd.vertex_buffer) ||
        !SameBytes(ed.index_buffer, vd.index_buffer)) {
        OneF::Note(Check::BuildDrawParams, "draw state");
    }
    if (!with_flags) {
        return; // step 4: the engine's R-owned flags were not consumed by the unlent draws
    }
    // Step 3: a flag the lent draw consumes now that R's view would not have had (both re-synced
    // after the last lent draw, so the difference is what the packets since forwarded).
    const auto missing = engine.dirty.flags & ~view->dirty.flags & render_flags;
    if (missing.none()) {
        return;
    }
    bool parts[5]{};
    for (size_t flag = 0; flag < missing.size(); ++flag) {
        if (missing[flag]) {
            parts[static_cast<size_t>(PartOf(flag))] = true;
        }
    }
    if (parts[static_cast<size_t>(FlagPart::Pipeline)]) {
        OneF::Note(Check::BuildPipeline, "flag not forwarded");
    }
    if (parts[static_cast<size_t>(FlagPart::RenderTargets)]) {
        OneF::Note(Check::BuildRenderTargets, "flag not forwarded");
    }
    if (parts[static_cast<size_t>(FlagPart::ImageViews)]) {
        OneF::Note(Check::BuildImageViews, "flag not forwarded");
    }
    if (parts[static_cast<size_t>(FlagPart::Dynamic)] ||
        parts[static_cast<size_t>(FlagPart::Other)]) {
        OneF::Note(Check::BuildDynamic, "flag not forwarded");
    }
}

void OneFollowerDraw::CompareList(Maxwell3D& engine) {
    // What a played list draw would have used: the draw span's list values (its view's indirect
    // state, SetViewList) and DrawListWords' read-back test on the view. The lent list draw uses
    // E's engine, here, with E parked inside DrawEnd's list branch.
    const auto& indirect = engine.draw_manager.indirect_state;
    if (!have_draw || last_draw.list_words != (indirect.list_words ? 1U : 0U) ||
        last_draw.list_keep_id != indirect.list_keep_id ||
        last_draw.list_address != indirect.indirect_start_address) {
        OneF::Note(Check::BuildList, "the draw span's list values against the indirect state");
        return;
    }
    // The read-back test reads the draw state and registers the view has (Compare checks them
    // too; this is the decision itself).
    if (RasterizerVulkan::OneFollowerListOnCpu(*view) !=
        RasterizerVulkan::OneFollowerListOnCpu(engine)) {
        OneF::Note(Check::BuildList, "the read-back decision on the view against the engine");
    }
}

void OneFollowerDraw::SetViewList(const OneF::DrawValues& values) {
    // DrawEnd's list branch, field by field (the rest of the indirect state is constant there).
    auto& indirect = view->draw_manager.indirect_state;
    indirect.is_byte_count = false;
    indirect.is_indexed = false;
    indirect.include_count = false;
    indirect.count_start_address = 0;
    indirect.indirect_start_address = values.list_address;
    indirect.buffer_size = 2 * sizeof(u32);
    indirect.max_draw_counts = 1;
    indirect.stride = 0;
    indirect.list_words = values.list_words != 0;
    indirect.list_keep_id = values.list_keep_id;
}

void OneFollowerDraw::ResyncFlags(const Maxwell3D& engine) {
    view->dirty.flags = (view->dirty.flags & ~render_flags) | (engine.dirty.flags & render_flags);
    lent_since_resync = 0;
}

// --- R: the packets --------------------------------------------------------------------------------

void OneFollowerDraw::ApplyViewSpans(const PacketHeader& header, const u8* spans) {
    if (spans == nullptr || view == nullptr) {
        return;
    }
    const u8* cursor = spans;
    if ((header.spans & OneF::SpanDelta) != 0) {
        for (u32 i = 0; i < header.delta_count; ++i) {
            OneF::RegPair pair;
            std::memcpy(&pair, cursor + size_t{i} * sizeof(pair), sizeof(pair));
            if (pair.index < Maxwell::NUM_REGS) {
                view->regs.reg_array[pair.index] = pair.value;
                // fable_b_one_follower 10.3 (a), a cheap belt: a program register in the delta
                // raises Shaders on the view whatever the flags span says (RefreshStages returns
                // on a clear flag and would keep the previous draw's programs).
                if ((pair.index >= ProgramWordsBegin && pair.index < ProgramWordsEnd) ||
                    (pair.index >= ProgramRegionBegin && pair.index < ProgramRegionEnd) ||
                    pair.index == RasterizeEnableWord || pair.index == MeshControlWord) {
                    view->dirty.flags[VideoCommon::Dirty::Shaders] = true;
                }
            }
        }
        cursor += Padded(size_t{header.delta_count} * sizeof(OneF::RegPair));
    }
    if ((header.spans & OneF::SpanFlags) != 0) {
        static_assert(std::is_trivially_copyable_v<OneF::FlagBits>);
        OneF::FlagBits raised;
        std::memcpy(&raised, cursor, sizeof(raised));
        view->dirty.flags |= raised;
        cursor += Padded(sizeof(OneF::FlagBits));
    }
    if ((header.spans & OneF::SpanDraw) != 0) {
        std::memcpy(&last_draw, cursor, sizeof(last_draw));
        have_draw = true;
        auto& draw = view->draw_manager.draw_state;
        draw.topology = last_draw.topology;
        draw.draw_mode = static_cast<Maxwell3D::DrawManager::DrawMode>(last_draw.draw_mode);
        draw.draw_indexed = last_draw.draw_indexed != 0;
        draw.draw_element_count = last_draw.draw_element_count;
        draw.base_index = last_draw.base_index;
        draw.vertex_buffer = last_draw.vertex_buffer;
        draw.index_buffer = last_draw.index_buffer;
        draw.base_instance = last_draw.base_instance;
        draw.instance_count = last_draw.instance_count;
        draw.drawn_instances = last_draw.drawn_instances;
        cursor += Padded(sizeof(OneF::DrawValues));
    }
    // Programs and Pools: R computes both from the registers itself (RefreshStages, the pools'
    // registers); part b's view keeps them for its compares.
    ApplyValueSpans(header, spans);
}

void OneFollowerDraw::ApplyValueSpans(const PacketHeader& header, const u8* spans) {
    if ((header.spans & (OneF::SpanEngine | OneF::SpanBinds)) == 0 || view == nullptr) {
        return;
    }
    // FRITCHII_ONEF_UNLENT_VALUES: what SyncFromEngine takes at a lent packet, by value.
    if (const u8* at = SpanAt(header, spans, OneF::SpanEngine)) {
        OneF::EngineValues values;
        std::memcpy(&values, at, sizeof(values));
        view->engine_state = static_cast<Maxwell3D::EngineHint>(values.engine_state);
        view->GetNativeMacroEngine().native_turing = values.native != 0;
    }
    if (const u8* at = SpanAt(header, spans, OneF::SpanBinds)) {
        OneF::BindsHeader binds;
        std::memcpy(&binds, at, sizeof(binds));
        const u8* entry = at + sizeof(binds);
        for (u32 i = 0; i < binds.count && i < OneF::MaxBindValues; ++i) {
            OneF::BindValue value;
            std::memcpy(&value, entry + size_t{i} * sizeof(value), sizeof(value));
            if (value.stage < Maxwell::MaxShaderStage && value.slot < Maxwell::MaxConstBuffers) {
                auto& info = view->state.shader_stages[value.stage].const_buffers[value.slot];
                info.address = value.address;
                info.size = value.size;
                info.enabled = value.enabled != 0;
            }
        }
    }
}

void OneFollowerDraw::CompareValues(Maxwell3D& engine) {
    // The by-value values against E's engine at a lent packet (E parked, the packet's own value
    // spans applied): Check::ViewValues.
    const char* what = nullptr;
    if (view->engine_state != engine.engine_state) {
        what = "the engine hint";
    } else if (view->GetNativeMacroEngine().native_turing !=
               engine.GetNativeMacroEngine().native_turing) {
        what = "the native macro mode";
    } else {
        for (size_t stage = 0; stage < Maxwell::MaxShaderStage && what == nullptr; ++stage) {
            for (size_t slot = 0; slot < Maxwell::MaxConstBuffers; ++slot) {
                const auto& v = view->state.shader_stages[stage].const_buffers[slot];
                const auto& e = engine.state.shader_stages[stage].const_buffers[slot];
                if (v.enabled != e.enabled || v.address != e.address || v.size != e.size) {
                    what = "a constant buffer binding";
                    break;
                }
            }
        }
    }
    if (what != nullptr) [[unlikely]] {
        thread_local std::string note;
        note = fmt::format("lent packet: {} the packets carried differs from E's engine", what);
        OneF::Note(Check::ViewValues, note.c_str());
        OneF::StopUnlentValues("R's view took a value E's engine does not have");
    }
}

void OneFollowerDraw::SyncFromEngine(Maxwell3D& engine, bool full) {
    if (view == nullptr || view_source != &engine) {
        // A new channel: the view follows its memory manager (a reference in Maxwell3D). The
        // R-owned objects stop naming the old view first (rule 2).
        LeaveView();
        const void* const memory = &engine.GetMemoryManager();
        const bool keep = OneF::BindSeedOn();
        Maxwell3D* next = nullptr;
        if (keep) {
            // FRITCHII_ONEF_BIND_SEED: channels alternate hundreds of times a second; the view a
            // channel had is seeded in full below, so only its construction is saved.
            for (const ViewSlot& slot : views) {
                if (slot.engine == &engine && slot.memory == memory) {
                    next = slot.view.get();
                    break;
                }
            }
        }
        if (next == nullptr) {
            if (!keep) {
                views.clear(); // one view, the old one goes (as before the switch)
            } else {
                std::erase_if(views, [&](const ViewSlot& slot) { return slot.engine == &engine; });
                // 32, not 4 (fable_b 25.1, 2026-11-15): with 5+ channels in rotation the oldest
                // slot was always the next one needed, so every bind built a view (~30 ms/s on R
                // and E's longest park). A view is a few hundred KiB.
                if (views.size() >= 32) {
                    views.erase(views.begin()); // R named none of them since LeaveView
                }
                static std::atomic<u64> built{};
                const u64 n = built.fetch_add(1, std::memory_order_relaxed) + 1;
                if ((n & (n - 1)) == 0) { // 1, 2, 4, 8 ...: a construction storm shows in the log
                    LOG_INFO(Render_Vulkan, "One follower: R built view {} (channel views kept {})",
                             n, views.size() + 1);
                }
            }
            auto fresh = std::make_unique<Maxwell3D>(engine.GetMemoryManager());
            next = fresh.get();
            views.push_back(ViewSlot{&engine, memory, std::move(fresh)});
        }
        view = next;
        view_source = &engine;
        full = true;
    }
    if (full) {
        view->regs = engine.regs;
        view->dirty.flags = engine.dirty.flags;
        auto& draw = view->draw_manager.draw_state;
        const auto& from = engine.draw_manager.draw_state;
        draw.topology = from.topology;
        draw.draw_mode = from.draw_mode;
        draw.draw_indexed = from.draw_indexed;
        draw.draw_element_count = from.draw_element_count;
        draw.base_index = from.base_index;
        draw.vertex_buffer = from.vertex_buffer;
        draw.index_buffer = from.index_buffer;
        draw.base_instance = from.base_instance;
        draw.instance_count = from.instance_count;
        draw.drawn_instances = from.drawn_instances;
        render_mode_on = engine.ShouldExecute();
        seeded_step = OneF::Step();
    }
    // What no delta carries (taken at every lent packet).
    view->state = engine.state;
    view->engine_state = engine.engine_state;
    view->GetNativeMacroEngine().native_turing = engine.GetNativeMacroEngine().native_turing;
    synced_engine = &engine;
    synced_engine_state = engine.engine_state;
    synced_native = engine.GetNativeMacroEngine().native_turing;
    view_ready = true;
}

void OneFollowerDraw::UpdateDiagnostic() {
    diagnostic.store(rasterizer.frame_dump.Capturing() || VideoCommon::RenderTrace::Enabled() ||
                         rasterizer.flash_watch.WatchesPasses(),
                     std::memory_order_relaxed);
}

void OneFollowerDraw::OnPacket(const PacketHeader& header, const u8* spans) {
    if (header.kind == OneF::PacketKind::Lent) {
        // At boot, lent packets arrive before the game binds a channel (steps 2-3 on from boot):
        // nothing to sync yet; the first lent packet after the bind seeds the view.
        if (rasterizer.channel_state == nullptr) [[unlikely]] {
            return;
        }
        // E is parked: take what deltas do not carry before the spans, which are newer.
        Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
        const bool full = (header.flags & OneF::PacketSeed) != 0 || seeded_step != OneF::Step();
        // FRITCHII_ONEF_UNLENT_VALUES' must-be-0: before the view takes the engine's values, the
        // ones the packets carried (this one's too) must equal them. At lent draws and every 16th
        // other lent packet (the bindings compare is 90 entries, inside E's park).
        if ((header.flags & OneF::PacketValues) != 0 && !full && view != nullptr &&
            view_source == &engine &&
            (header.op == static_cast<u8>(LentOp::Draw) || (header.number & 15) == 0)) {
            ApplyValueSpans(header, spans);
            CompareValues(engine);
        }
        SyncFromEngine(engine, full);
        UpdateDiagnostic();
        // This packet's own operation; a seed took the flags too, so it starts the count again.
        lent_since_resync = full ? 1 : lent_since_resync + 1;
        if (full) {
            // The seed is the engine at this lent packet, spans included. The draw span's own
            // values are still this draw's (CompareList reads the list values from it).
            if (spans != nullptr && (header.spans & OneF::SpanDraw) != 0) {
                size_t at = 0;
                if ((header.spans & OneF::SpanDelta) != 0) {
                    at += Padded(size_t{header.delta_count} * sizeof(OneF::RegPair));
                }
                if ((header.spans & OneF::SpanFlags) != 0) {
                    at += Padded(sizeof(OneF::FlagBits));
                }
                std::memcpy(&last_draw, spans + at, sizeof(last_draw));
                have_draw = true;
            }
            return;
        }
    }
    ApplyViewSpans(header, spans);
}

void OneFollowerDraw::DropDraw(const PacketHeader& header) {
    auto& stage = rasterizer.mem_stage;
    const MemStage::DrawPacket* const packet =
        stage.TakeRingDraw(header.draw_seq, header.position, 0, false);
    if (packet == nullptr) {
        OneF::Note(Check::RenderPacketMissing, "dropped draw");
        return;
    }
    // Section 5's guard: a dropped packet only brings E's buffers up to date.
    if (packet->gpu_count.buffer != VK_NULL_HANDLE || packet->indirect.buffer != VK_NULL_HANDLE ||
        packet->indirect_count.buffer != VK_NULL_HANDLE ||
        (packet->flags & MemStage::ListArgs) != 0) {
        OneF::Note(Check::DropNotPlain, "indirect or count buffers");
    }
    const bool barrier = (packet->flags & MemStage::PostCopyBarrier) != 0;
    stage.PlayRest();
    if (barrier) {
        stage.PlayPostCopyBarrier();
    }
    counts.dropped.fetch_add(1, std::memory_order_relaxed);
}

void OneFollowerDraw::PlayDraw(const PacketHeader& header) {
    UpdateDiagnostic();
    if (view == nullptr || !view_ready || !have_draw ||
        (header.spans & OneF::SpanDraw) == 0) [[unlikely]] {
        // E unlends only after a lent packet seeded the view, and every draw carries its values.
        OneF::Note(Check::RenderPacketMissing, "no view or no draw values");
        DropDraw(header);
        return;
    }
    if ((header.flags & OneF::PacketConditional) != 0 && !render_mode_on) {
        DropDraw(header);
        return;
    }
    const OneF::DrawValues values = last_draw;
    if (rasterizer.channel_state == nullptr || view_source != &rasterizer.channel_state->maxwell3d)
        [[unlikely]] {
        // The view follows another channel than the bound one: E lends after a bind until a lent
        // packet syncs the new channel's view, so this must not happen.
        OneF::Note(Check::RenderPacketMissing, "a played draw with the view of another channel");
        if (OneF::BindSeedOn()) {
            OneF::Note(Check::BindSeedView, "a played draw with the view of another channel");
            OneF::StopBindSeed("a played draw found the view on another channel");
        }
        DropDraw(header);
        return;
    }
    EnterView(); // the default on R already (rule 2); here for the first draw after a step change
    if (values.list_words != 0) {
        // UNLEND's List bit: the view's indirect state as DrawEnd's list branch set it on E, for
        // DrawListWords (its keep id, the words' address; ring B's packet has their buffer).
        SetViewList(values);
        if (RasterizerVulkan::OneFollowerListOnCpu(*view)) [[unlikely]] {
            // E lends these (NoPath): the view must decide as E's engine did.
            OneF::Note(Check::PlayedNoPath, "a played list draw that reads its words on the CPU");
            OneF::DisableUnlend("a played list draw needed its words on the CPU");
            DropDraw(header);
            view->draw_manager.indirect_state.list_words = false;
            view->draw_manager.indirect_state.list_keep_id = 0;
            return;
        }
    }
    playing = &header;
    SCOPE_EXIT {
        playing = nullptr;
        if (values.list_words != 0) {
            // As DrawEnd after ProcessDrawIndirect: the next draw is not a list draw.
            view->draw_manager.indirect_state.list_words = false;
            view->draw_manager.indirect_state.list_keep_id = 0;
        }
    };
    if (values.list_words != 0) {
        rasterizer.DrawIndirect();
    } else {
        rasterizer.Draw(values.indexed != 0, values.instance_arg);
    }
    // R is its only writer (as `pushed` on E).
    counts.played.store(counts.played.load(std::memory_order_relaxed) + 1,
                        std::memory_order_relaxed);
}

// --- R: rules 1 and 2 around lent calls (fable_b 10.2) ----------------------------------------------

void OneFollowerDraw::EnterView() {
    if (on_view || view == nullptr) {
        return;
    }
    auto& r = rasterizer;
    saved_rasterizer = r.ExchangeMaxwell3D(view);
    // Rule 3: the pipeline cache is E's then and always reads E's engine.
    if (!VideoCommon::OneFollower::g_e_pipeline) {
        saved_pipeline = r.pipeline_cache.ExchangeMaxwell3D(view);
    }
    saved_texture = r.texture_cache.ExchangeMaxwell3D(view);
    saved_query = r.query_cache.ExchangeMaxwell3D(view);
    saved_tracker = r.state_tracker.ExchangeFlags(&view->dirty.flags);
    on_view = true;
}

void OneFollowerDraw::LeaveView() {
    if (!on_view) {
        return;
    }
    auto& r = rasterizer;
    r.state_tracker.ExchangeFlags(saved_tracker);
    r.query_cache.ExchangeMaxwell3D(saved_query);
    r.texture_cache.ExchangeMaxwell3D(saved_texture);
    if (!VideoCommon::OneFollower::g_e_pipeline) {
        r.pipeline_cache.ExchangeMaxwell3D(saved_pipeline);
    }
    r.ExchangeMaxwell3D(saved_rasterizer);
    on_view = false;
}

void OneFollowerDraw::BeforeLent() {
    // Rule 2: the lent call is E's code on E's engine.
    LeaveView();
    // Rule 1: the R-owned flags the view raised and R has not consumed go to the engine, so the
    // lent call recomputes what they ask for (shared ones too: one rebind too many at worst).
    if (rasterizer.channel_state == nullptr || view == nullptr) {
        return;
    }
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    if (view_source == &engine) {
        engine.dirty.flags |= view->dirty.flags & render_flags;
    }
}

void OneFollowerDraw::AfterLent() {
    if (rasterizer.channel_state == nullptr || view == nullptr) {
        return;
    }
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    if (view_source != &engine) {
        // A channel bind ran in this call. Without FRITCHII_ONEF_BIND_SEED the next lent packet
        // makes the new channel's view (E's tracking waits for that packet's seed). With it
        // Play re-seeded E's tracking here already (ViewReady): the view follows now, E parked.
        if (!OneF::ViewReady() || OneF::Step() < StepLevel::Build) {
            return;
        }
        SyncFromEngine(engine, true);
    }
    // Rule 1: after every lent call (not only lent draws) the view takes the engine's flags:
    // what the call consumed is consumed, what it raised (a new command buffer's invalidations)
    // is raised.
    ResyncFlags(engine);
    // Rule 2: R names the view again until the next lent call.
    if (view_ready && OneF::Step() >= StepLevel::Build) {
        EnterView();
    }
}

void OneFollowerDraw::ViewOff() {
    LeaveView();
}

// --- R: a played draw that did not build ------------------------------------------------------------

void OneFollowerDraw::ReportDescriptorMismatch(const GraphicsPipeline& pipeline, u32 expected,
                                               u32 got) {
    OneFollowerDraw* const self = g_instance.load(std::memory_order_acquire);
    if (self == nullptr || self->playing == nullptr ||
        self->mismatch_reported.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    const PacketHeader& header = *self->playing;
    const FillRecord& fill = self->fill_records[header.draw_seq % FillRecords];
    const bool have_fill = fill.seq == header.draw_seq;
    // R's side: the view's program words (the rasterizer's engine is the view while it plays),
    // the pipeline's programs and its descriptors per stage.
    VideoCommon::MemStageShadow::ProgramId view_id;
    VideoCommon::MemStageShadow::BackProgramId(*self->rasterizer.maxwell3d, view_id);
    const MemStage::DrawFacts facts = pipeline.MemFacts();
    std::string stages_text;
    for (size_t stage = 0; stage < 5; ++stage) {
        const Shader::Info& info = facts.stage_infos[stage];
        const int uniforms = std::popcount((*facts.uniform_masks)[stage]);
        const size_t storage = info.storage_buffers_descriptors.size();
        const u32 textures = Shader::NumDescriptors(info.texture_descriptors) +
                             Shader::NumDescriptors(info.image_descriptors) +
                             Shader::NumDescriptors(info.texture_buffer_descriptors) +
                             Shader::NumDescriptors(info.image_buffer_descriptors);
        stages_text += fmt::format(
            "{}s{}: R uniforms {} storage {} textures {} / E uniforms {} storage {}",
            stage == 0 ? "" : "; ", stage, uniforms, storage, textures,
            have_fill ? std::popcount(fill.uniform_masks[stage]) : -1,
            have_fill ? static_cast<int>(fill.ssbo_counts[stage]) : -1);
    }
    std::string code_text;
    if (have_fill) {
        for (size_t slot = 0; slot < fill.code.size(); ++slot) {
            if (fill.code[slot] != view_id.code[slot]) {
                code_text += fmt::format("{}[{}] E {:#x} R {:#x}", code_text.empty() ? "" : ", ",
                                         slot, fill.code[slot], view_id.code[slot]);
            }
        }
    }
    std::string words_text;
    size_t differing = 0;
    if (have_fill) {
        for (size_t i = 0; i < fill.words.size(); ++i) {
            if (fill.words[i] != view_id.words[i]) {
                if (differing++ < 8) {
                    words_text += fmt::format("{}[{}] E {:#x} R {:#x}",
                                              words_text.empty() ? "" : ", ", i, fill.words[i],
                                              view_id.words[i]);
                }
            }
        }
    }
    MemStage::StagedFacts table{};
    // The locked path: the lookup cache is E's (fable_b 10.1 row 9).
    const bool in_table = self->rasterizer.mem_stage.LookupFactsLocked(view_id, table);
    const auto hashes = pipeline.ShaderHashes();
    LOG_CRITICAL(
        Render_Vulkan,
        "One follower: unlent draw descriptor mismatch (expected {}, got {}): packet {} draw seq "
        "{} ring B {} position {:#x}; E's fill {}: program key {:#x}, hashes {:#x}/{:#x}, stages "
        "{:#x}; R's pipeline: hashes {:#x} {:#x} {:#x} {:#x} {:#x} {:#x}, stages {:#x}, view "
        "program key {:#x}; address spaces (channel's memory manager id + 1) E {} R {} (R bound "
        "{}); program code addresses E vs R: {}; program words E vs R: {} differ{}{}; per stage: "
        "{}; the facts table now for R's code: {} (hashes {:#x}/{:#x}, stages {:#x})",
        expected, got, header.number, header.draw_seq, header.buffer_number, header.position,
        have_fill ? "recorded" : "not recorded (overwritten)", fill.key, fill.hashes[0],
        fill.hashes[1], fill.stages, hashes[0], hashes[1], hashes[2], hashes[3], hashes[4],
        hashes[5], facts.stages, view_id.key, have_fill ? fill.space : 0, view_id.space,
        self->rasterizer.gpu_memory->GetID() + 1, code_text.empty() ? "the same" : code_text,
        differing, differing != 0 ? ": " : "", words_text,
        stages_text, in_table ? "found" : "missing", table.hashes[0], table.hashes[1],
        table.stages);
}

// --- 0xe2c -------------------------------------------------------------------------------------------

bool OneFollowerDraw::ConditionMode(Maxwell3D& engine) {
    const auto& regs = engine.regs;
    using RenderEnable = Maxwell::RenderEnable;
    switch (regs.render_enable_override) {
    case RenderEnable::Override::AlwaysRender:
        return true;
    case RenderEnable::Override::NeverRender:
        return false;
    case RenderEnable::Override::UseRenderEnable:
        break;
    default:
        return true;
    }
    Maxwell::ReportSemaphore::Compare cmp{};
    const auto read = [&] {
        // ProcessQueryCondition read these with E parked at this place (its ReadBlock flushed
        // what was pending), so the bytes are the ones it saw.
        engine.GetMemoryManager().ReadBlockUnsafe(regs.render_enable.Address(), &cmp, sizeof(cmp));
    };
    switch (regs.render_enable.mode) {
    case RenderEnable::Mode::True:
        return true;
    case RenderEnable::Mode::False:
        return false;
    case RenderEnable::Mode::Conditional:
        read();
        return cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
    case RenderEnable::Mode::IfEqual:
        read();
        return cmp.initial_sequence == cmp.current_sequence &&
               cmp.initial_mode == cmp.current_mode;
    case RenderEnable::Mode::IfNotEqual:
        read();
        return cmp.initial_sequence != cmp.current_sequence ||
               cmp.initial_mode != cmp.current_mode;
    default:
        return true;
    }
}

void OneFollowerDraw::Condition(bool engine_execute_on) {
    Maxwell3D& engine = rasterizer.channel_state->maxwell3d;
    if (view == nullptr || view_source != &engine) {
        render_mode_on = engine_execute_on; // not seeded yet: nothing of R's to compare
        return;
    }
    render_mode_on = ConditionMode(*view);
    if (render_mode_on != engine_execute_on) {
        OneF::Note(Check::BuildCondition, engine_execute_on ? "R says False" : "R says True");
    }
}

} // namespace Vulkan
