// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// One follower, part c (work\fable_b_one_follower.md 3, 5 and 8.1): R's draws from the packet and
// the register view, and the verify compares (steps 3 and 4).
//
// The view engine. R's code reads `maxwell3d->regs`, `maxwell3d->dirty.flags` and
// `maxwell3d->draw_manager.draw_state` at about 680 sites. R keeps a Maxwell3D of its own that no
// method ever reaches (no CallMethod, no macros, no shadow RAM): the packets' spans write its
// registers, raise its flags and set its draw state, and a lent packet (E parked) gives it what
// deltas do not carry (the constant buffer bindings, the engine hint, the macro mode). While R
// plays an unlent draw, the R-owned objects (the rasterizer's own engine pointer, the pipeline,
// texture and query caches, the state tracker's flags) point at it; the buffer cache keeps E's
// engine. So the draw runs today's code, PrepareDraw included, on the values the packet carries:
// exact by construction where the view equals the engine, which steps 2 and 3 verify.
//
// The two engines' dirty flags. A lent draw reads E's engine and clears its flags; an unlent draw
// reads the view and clears the view's. A side that did not consume a flag keeps it raised, so
// its next build redoes that part from registers equal to the other side's (conservative, exact).
// After every lent draw this file runs (RunLent), the view's R-owned flags take the engine's.
//
// Draw kinds R plays at step 4 (UnlendKind): PlainIndexed, PlainArrays, InstanceLoop, Mesh (native
// task/mesh pipelines; emulated ones need the staging pool and stay lent, NoPath) and List (E's
// fill binds the two words and puts their buffer in ring B's packet, ListArgs, as GPUMem's served
// list draws did; the draw span carries the keep id and the words' address, R's view takes them
// into its indirect state for DrawListWords; a list draw that reads its words on the CPU stays
// lent, NoPath; one on a mesh pipeline needs Mesh too). Indirect, GpuCount and InlineIndex keep
// their bits but stay lent here (KindNotInMask): their R side needs values E owns (indirect
// words and counts, the GPU count's buffer, the inline index vector).

#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/one_follower.h"
#include "video_core/renderer_vulkan/vk_mem_stage.h"
#include "video_core/texture_handle_values.h"

namespace Vulkan {

class GraphicsPipeline;
class RasterizerVulkan;

class OneFollowerDraw final : public VideoCommon::OneFollower::DrawPlayer {
public:
    using Maxwell3D = Tegra::Engines::Maxwell3D;
    using PacketHeader = VideoCommon::OneFollower::PacketHeader;
    using LentReason = VideoCommon::OneFollower::LentReason;

    explicit OneFollowerDraw(RasterizerVulkan& rasterizer);
    ~OneFollowerDraw();

    OneFollowerDraw(const OneFollowerDraw&) = delete;
    OneFollowerDraw& operator=(const OneFollowerDraw&) = delete;

    /// RasterizerVulkan::Draw's first step. True: the draw is done here (on E: lent and run on R,
    /// or filled into ring B and pushed; on R inside a foreign lent call: run with the verify).
    /// False: go on with Draw (steps 0 to 2, or R inside this file's own call).
    bool RouteDraw(bool is_indexed, u32 instance_count);
    /// RasterizerVulkan::DrawIndirect's first step on E for a list draw (indirect_state.list_words)
    /// at step 4 with UNLEND's List bit: filled into ring B and pushed, or lent with its reason
    /// (and the verify). True: done. False: the caller lends it as before (step below 4, no bit).
    bool RouteListDraw();

    // DrawPlayer (R).
    void OnPacket(const PacketHeader& header, const u8* spans) override;
    void PlayDraw(const PacketHeader& header) override;
    void Condition(bool engine_execute_on) override;
    void BeforeLent() override;
    void AfterLent() override;
    void ViewOff() override;

    /// R, at GraphicsPipeline::ConfigureDraw's descriptor count mismatch on an unlent draw: logs
    /// once, at Critical, what E's fill used for the packet (program words and facts) against R's
    /// pipeline (its programs, the view's program words, descriptors per stage), and the packet.
    static void ReportDescriptorMismatch(const GraphicsPipeline& pipeline, u32 expected,
                                         u32 got);

    /// R: the address space E's fill resolved draw `seq`'s programs in (id + 1), 0 if unknown.
    [[nodiscard]] u64 FillSpace(u64 seq) const noexcept {
        const FillRecord& record = fill_records[seq % FillRecords];
        return record.seq == seq ? record.space : 0;
    }

    /// R, in PrepareDraw: the header of the unlent draw being played, or null.
    [[nodiscard]] const PacketHeader* Playing() const noexcept {
        return playing;
    }

    /// E: a diagnostic reads the work on R (frame dump, render trace, flash watch; set by R).
    [[nodiscard]] bool Diagnostic() const noexcept {
        return diagnostic.load(std::memory_order_relaxed) || env_traces;
    }
    /// R: the R-owned objects name the view now (rule 2), so a state invalidation lands there.
    [[nodiscard]] bool OnView() const noexcept {
        return on_view;
    }

    /// Played, dropped (0xe2c False) and verified draws since boot (part d's lines).
    struct Counts {
        std::atomic<u64> played{};
        std::atomic<u64> dropped{};
        std::atomic<u64> verified{};
        std::atomic<u64> pushed{}; ///< E
    };
    [[nodiscard]] const Counts& Totals() const noexcept {
        return counts;
    }

private:
    /// The R-owned objects read the view engine for one played draw, and E's after.
    class ViewScope;

    // E.
    [[nodiscard]] u32 Classify(Maxwell3D& engine, bool is_indexed, u32 instance_count) const;
    /// The first lend reason that applies to a draw of `kind` whose bit is in the mask, before
    /// the fill (Verify, StreamWrite, Diagnostic, Transient); Count when none.
    [[nodiscard]] LentReason UnlentBlocker(Maxwell3D& engine, bool is_indexed, u32 instance_count,
                                           u32 kind);
    /// Fills ring B for an unlent draw from E's engine (as GPUMem's StageFillDraw does from its
    /// copy). False with `reason` when the draw must be lent.
    /// `pipeline` (FRITCHII_ONEF_E_PIPELINE): the one E looked up; its own facts fill the draw.
    /// Null: the facts table's entry for the draw's programs.
    /// `list_args`: a list draw, the fill binds its two words (the cache's indirect parameters are
    /// the engine's for the fill) and the packet takes their buffer (ListArgs).
    bool FillRing(Maxwell3D& engine, GraphicsPipeline* pipeline, bool is_indexed, u64 position,
                  u64 seq, u64& buffer_number,
                  LentReason& reason, bool list_args = false);
    /// E's fill and push of an unlent draw (RouteDraw's and RouteListDraw's common tail). False
    /// with `reason` when it must be lent after all.
    bool FillAndPush(Maxwell3D& engine, bool is_indexed, u64 seq, bool list, LentReason& reason);
    /// FRITCHII_ESPLIT_DEPTH > 0: the draw's tail can run on E2 while E1 goes on (S2).
    [[nodiscard]] bool AsyncTailFits(const Maxwell3D& engine,
                                     const GraphicsPipeline* pipeline) const;
    /// `list`: a list draw (R runs DrawIndirect, not Draw).
    void LendDraw(bool is_indexed, u32 instance_count, LentReason reason, bool verify,
                  bool list = false);
    /// FRITCHII_ONEF_UNLENT_RESET on a forced-lent draw: E's own indirect reset test.
    struct ResetCheck {
        bool decided{}; ///< E ran the test (a candidate, the switch on)
        bool is_reset{};
    };
    /// What E's fill would use for a lent draw's programs (looked up on E before the lend).
    struct FactsExpect {
        bool found{};
        std::array<u64, 2> hashes{};
        u64 space{}; ///< the address space E looked the programs up in (id + 1)
        ResetCheck reset{};
    };

    // R.
    /// A lent draw, E parked: the compares when `verify`, Draw, then the view's flags re-synced.
    void RunLent(bool is_indexed, u32 instance_count, bool verify, const FactsExpect& expect,
                 bool list = false);
    void Compare(Maxwell3D& engine, bool with_flags);
    /// A forced-lent list draw: the list values the packet's draw span gave the view, and R's
    /// read-back decision on the view, against E's engine (Check::BuildList).
    void CompareList(Maxwell3D& engine);
    /// The view's indirect state as DrawEnd's list branch sets it, from the draw span.
    void SetViewList(const VideoCommon::OneFollower::DrawValues& values);
    void ResyncFlags(const Maxwell3D& engine);
    /// The view takes the spans of one packet.
    void ApplyViewSpans(const PacketHeader& header, const u8* spans);
    /// FRITCHII_ONEF_UNLENT_VALUES: the view takes a packet's SpanEngine and SpanBinds.
    void ApplyValueSpans(const PacketHeader& header, const u8* spans);
    /// At a lent packet with PacketValues (E parked): the view's by-value values against E's
    /// engine (Check::ViewValues; the switch stops on a difference).
    void CompareValues(Maxwell3D& engine);
    /// At a lent packet (E parked): the view takes from E's engine what deltas do not carry, or
    /// everything (`full`).
    void SyncFromEngine(Maxwell3D& engine, bool full);
    /// A dropped draw's packet: its sections played, nothing recorded.
    void DropDraw(const PacketHeader& header);
    void UpdateDiagnostic();
    /// The 0xe2c mode as Maxwell3D::ProcessQueryCondition decides it (no host condition), on
    /// `engine`'s registers and the compare words in game memory.
    [[nodiscard]] static bool ConditionMode(Maxwell3D& engine);

    RasterizerVulkan& rasterizer;

    // R only (E reads the synced_* fields below only after a lent packet released it).
    Maxwell3D* view{}; ///< the view engine of the channel R follows (one of `views`)
    /// The view engines: with FRITCHII_ONEF_BIND_SEED one per channel's engine (and its memory
    /// manager), kept across binds; without it only the current one, as before.
    struct ViewSlot {
        const Maxwell3D* engine{};
        const void* memory{};
        std::unique_ptr<Maxwell3D> view;
    };
    std::vector<ViewSlot> views;
    const Maxwell3D* view_source{}; ///< the engine the view follows (its channel)
    const PacketHeader* playing{};
    bool nested{};                  ///< inside RunLent's own Draw call
    /// fable_b 10.2 rule 2: the R-owned objects (the rasterizer's engine, the pipeline, texture
    /// and query caches' engines, the state tracker's flags) name the view while this is set,
    /// which is always on R outside lent calls once the view follows the bound channel.
    bool on_view{};
    Maxwell3D* saved_rasterizer{};
    Maxwell3D* saved_pipeline{};
    Maxwell3D* saved_texture{};
    Maxwell3D* saved_query{};
    Maxwell3D::DirtyState::Flags* saved_tracker{};
    void EnterView();
    void LeaveView();
    bool render_mode_on{true};      ///< R's 0xe2c mode
    /// Lent packets since the view's flags last took the engine's (RunLent, a seed): the step-3
    /// flags compare holds only at 1 (the draw's own packet).
    u32 lent_since_resync{2};
    VideoCommon::OneFollower::DrawValues last_draw{};
    bool have_draw{};
    /// The R-owned and shared dirty flags (OwnerOf(flag) != FlagOwner::E).
    Maxwell3D::DirtyState::Flags render_flags{};

    // Written by R at lent packets (E parked), read by E.
    bool view_ready{};
    const Maxwell3D* synced_engine{};
    Maxwell3D::EngineHint synced_engine_state{};
    bool synced_native{};
    VideoCommon::OneFollower::StepLevel seeded_step{};

    /// What E's fill used for a pushed draw (E writes before the push, R reads at a mismatch).
    struct FillRecord {
        u64 seq;
        u64 key;
        u64 space;                 ///< the address space E resolved the programs in (id + 1)
        std::array<u64, 6> code;   ///< the programs' device addresses there
        std::array<u64, VideoCommon::MemStageShadow::ProgramWordCount> words;
        std::array<u64, 2> hashes;
        std::array<u32, 5> uniform_masks;
        std::array<u8, 5> ssbo_counts;
        u32 stages;
        u64 buffer_number;
    };
    static constexpr size_t FillRecords = 64;
    std::array<FillRecord, FillRecords> fill_records{};
    std::atomic<bool> mismatch_reported{};

    // E only.
    /// fable_b 23.1 cut 4's dry price (GPU profile on): per pipeline, the handle values of its
    /// last unlent draw, the key of the constant buffer ranges they came from and the write
    /// generations of those pages (FritchiiPrice "Price handle values" line).
    struct HandleMemo {
        std::vector<u32> values;
        u64 key{};
        u64 sum{};
        bool valid{};
    };
    std::unordered_map<const void*, HandleMemo> handle_memos;
    void PriceHandleValues(Tegra::Engines::Maxwell3D& engine,
                           std::span<const Shader::Info> infos, const void* pipeline, u64 ticks);
    /// FRITCHII_HANDLE_PROG_KEEP (fable_b 25.2 cut g): per pipeline, its fill's handle program
    /// evaluations in fill order with what each read (the pipeline's program hashes name it: a
    /// pipeline made again at the same address starts over).
    struct ProgramMemo {
        std::array<u64, 2> hashes{};
        std::vector<VideoCommon::HandleProgramRecord> records;
    };
    std::unordered_map<const void*, ProgramMemo> program_memos;
    /// The handle values of a draw whose pipeline has handle programs, with the cut at `mode`
    /// (FillGraphicsHandleValues' values in its order).
    void FillHandleValuesKept(Tegra::Engines::Maxwell3D& engine,
                              std::span<const Shader::Info> infos, const void* pipeline,
                              const std::array<u64, 2>& hashes, bool null_pointer_handles,
                              int mode);
    u64 e_draw_seq{};
    u64 unlent_candidates{};
    /// FillRing's last Transient refusal (fable_b 19's dry counter), and whether the fill of the
    /// draw being routed refused (UnlentBlocker counts its own tests).
    VideoCommon::OneFollower::TransientTest fill_test{};
    bool filled_refused{};
    /// UnlentBlocker's PacketFlag bits for the draw's packet (PacketNoReset), and its reset test
    /// for a forced-lent draw (LendDraw hands it to RunLent).
    u8 push_flags{};
    u8 push_occl{}; ///< OneF::OcclPrepassBit of the draw's flag variant (E's lookup)
    ResetCheck verify_reset{};
    MemStage::StagedFacts facts{};
    std::vector<u32> handle_values;

    struct ResetCacheEntry {
        u64 program{};
        DAddr device{};
        u64 map_serial{};
        size_t as_id{};
        u64 inval_epoch{};
        bool valid{};
        bool result{};
    };
    static constexpr size_t ResetCacheSize = 128;
    std::array<ResetCacheEntry, ResetCacheSize> reset_cache{};

    /// Diagnostics that read the draw on R (frame dump, render trace, flash watch): set by R.
    std::atomic<bool> diagnostic{};
    bool env_traces{}; ///< debug environment traces in Draw's head (read once)

    Counts counts;
};

} // namespace Vulkan
