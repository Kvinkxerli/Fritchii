// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Tegra {
class MemoryManager;
}
namespace Tegra::Engines {
class Maxwell3D;
}

namespace VideoCommon {

class PageFilter;
struct MemStageShadowEngine;

/// Memory stage (fable_memstage_step6_spec.md 3.3): what a ring Call entry does, as the stage
/// reads it, in bits 32-39 of the entry's header (GpuSplit::Defer's and CallWithPayload's
/// `flags`; Execute ignores them). Level (0) means unknown: the back runs it and the stage copies
/// its registers afterwards, which is always correct. A tag other than Level and Pass promises a
/// fixed payload layout, given below; the closures that carry one are plain structs (a lambda's
/// captures are no layout to rely on), with static_asserts on the words at their call site.
enum class StageTag : u8 {
    Level = 0,
    /// The rasterizer switches channel. Payload word 0: the bind id (control/scheduler.cpp).
    Channel = 1,
    /// Maxwell3D::ApplyDrawPacket: [engine, system, count | checks << 32, then count writes as
    /// method | argument << 32, then the checks]. Each write as a Direct method entry.
    Writes = 2,
    /// Maxwell3D::FinishFrontMacro: [engine, system]. ConsumeSink, then DrawDeferred.
    FinishMacro = 3,
    /// A draw manager change, StageDrawLoop in header bits 40-47 (StageDrawLoopBits). Payload:
    /// FoldFirstInstance [engine]; AddLoopInstances [engine, count in the low 32 bits].
    DrawLoop = 4,
    /// Touches no register, no buffer cache state and no game memory the stage reads.
    Pass = 5,
    /// Uses the buffer cache or another back-side object; writes no 3D state (memstage_shadow.cpp).
    Cache = 6,
    /// Maxwell3D::ApplyFrontOp (FRITCHII_ONEF_FRONT_OPS, macro_ops_oot.h): [engine, system,
    /// header, sent mask, pair mask, arguments two to a word, check words]. The writes are
    /// OotOps::Expand's; each sent one as a Pairs word (pair mask bit) or a Direct method entry.
    Ops = 7,
    /// Maxwell3D::BackE2cCondition: evaluated at E's stream place without draining E2 when async.
    E2c = 8,
};

/// StageTag::DrawLoop's kind, in header bits 40-47 (the payloads stay as small as the lambdas
/// they replaced: one word and two).
enum class StageDrawLoop : u8 {
    FoldFirstInstance = 0, ///< DrawManager::FoldFirstInstance; payload [engine]
    AddLoopInstances = 1,  ///< DrawManager::AddLoopInstances(count); payload [engine, count]
    ListDraw = 2,          ///< the list draw's address and keep id; payload [engine, address, id]
};

/// The bits a tag takes in a Call entry's header.
[[nodiscard]] constexpr u64 StageTagBits(StageTag tag) noexcept {
    return u64{static_cast<u8>(tag)} << 32;
}

[[nodiscard]] constexpr StageTag StageTagOf(u64 header) noexcept {
    return static_cast<StageTag>((header >> 32) & 0xff);
}

/// StageTag::DrawLoop with its kind, for a Call entry's header.
[[nodiscard]] constexpr u64 StageDrawLoopBits(StageDrawLoop which) noexcept {
    return StageTagBits(StageTag::DrawLoop) | (u64{static_cast<u8>(which)} << 40);
}

[[nodiscard]] constexpr StageDrawLoop StageDrawLoopOf(u64 header) noexcept {
    return static_cast<StageDrawLoop>((header >> 40) & 0xff);
}

/// StageTag::Cache's kind, in header bits 40-47 (FRITCHII_MEMSTAGE_RELEASE): which cache call it
/// is, when the stage treats one apart. Other's payload stays the closure's own (no layout
/// promised); SemaphoreRelease's is StageReleaseWord's.
///
/// Rule (fable_stage_levelpoints 12.2, the memory stage's sync mirror): a new closure that calls
/// rasterizer->Query (a query report, a semaphore release) must carry the SemaphoreRelease kind
/// and its layout, or be untagged. GPUMem takes every other Cache-tagged closure as one that adds
/// no pending query sync; a Query in one would make a passed sync point's wait for idle find a
/// sync pending (must be 0: PassQuerySync), or a dry count wrong.
enum class StageCacheCall : u8 {
    Other = 0,            ///< any other cache call
    /// The puller's semaphore releases (WriteLong, SemaphoreRelease, SemaphoreExecute's release:
    /// QueryCache::CounterReport with the payload).
    SemaphoreRelease = 1,
};

/// StageTag::Cache with its kind, for a Call entry's header.
[[nodiscard]] constexpr u64 StageCacheCallBits(StageCacheCall which) noexcept {
    return StageTagBits(StageTag::Cache) | (u64{static_cast<u8>(which)} << 40);
}

[[nodiscard]] constexpr StageCacheCall StageCacheCallOf(u64 header) noexcept {
    return static_cast<StageCacheCall>((header >> 40) & 0xff);
}

/// StageCacheCall::SemaphoreRelease's payload (fable_stage_p3 3), a fixed layout: puller.cpp's
/// ReleaseCall, which static_asserts these words. Four words.
enum class StageReleaseWord : u8 {
    Rasterizer = 0,        ///< VideoCore::RasterizerInterface*
    Address = 1,           ///< the semaphore's GPU address
    PayloadProperties = 2, ///< the payload (low half), the QueryPropertiesFlags (high half)
    Memory = 3,            ///< the channel's Tegra::MemoryManager*
};

/// Memory stage piece 5b (spec 6.2), FRITCHII_MEMSTAGE_SHADOW=1: the stage's classify, register
/// copy, draw manager copy and handlers run on the back thread on each ring entry just before the
/// back executes it, and produce nothing; every draw the back issues is compared with the
/// shadow's. memstage_shadow.cpp has the details. Off, each hook below is one test of g_on.
namespace MemStageShadow {

/// Bumped whenever the shader cache drops a cached program because its code was written
/// (ShaderCache::InvalidatePageEntries). The memory stage's facts table is keyed by the program
/// registers (code addresses), not the code: an entry published before a code write may describe
/// the old code at the same addresses. Each entry keeps the epoch read before its pipeline was
/// looked up, and a lookup treats an entry older than the epoch as a miss (the draw is lent, and
/// its own publish brings the entry up to date). One follower: code writes reach the shader cache
/// on R in stream order with E parked, so E's lookups see every bump before its position.
inline std::atomic<u64> g_program_code_epoch{};

/// FRITCHII_MEMSTAGE_SHADOW (boot): read by ReadSwitch before the back thread starts. Atomic
/// (relaxed): FlushRegion and FlushCaching test it on CPU threads too, while a new session's
/// ReadSwitch may write it.
inline std::atomic<bool> g_on{};

/// Raised by the state tracker's flag invalidations while on (compare b of piece 5c).
inline std::atomic<u64> g_flag_invalidations{};
inline void NoteFlagInvalidation() noexcept {
    if (g_on.load(std::memory_order_relaxed)) [[unlikely]] {
        g_flag_invalidations.fetch_add(1, std::memory_order_relaxed);
    }
}

/// Reads the switch (GpuSplit::MaybeStart, before the back thread exists).
void ReadSwitch();
/// FRITCHII_MEMSTAGE_SHADOW's value (0 when off).
[[nodiscard]] int Mode();
/// The back thread starts reading the ring at position 0 (GpuSplit::BackLoop).
void Start();
/// The back is about to execute the entry at ring position `position` (GpuSplit::BackLoop).
void Before(u64 position);

/// The back's draw manager issues a draw (DrawManager::ProcessDraw / ProcessDrawIndirect, before
/// the render-enable test): compares a and b.
void BackDraw(Tegra::Engines::Maxwell3D& maxwell3d, bool indexed, u32 instances, bool indirect);
/// The back bound channel `bind_id` (the Channel-tagged closure).
void BackChannel(s32 bind_id);
/// The back's render-enable decision: true when host conditional rendering took it.
void BackCondition(bool host);
/// MemoryManager::FlushCaching on the back: the device ranges it hands InnerInvalidation
/// (compare d).
void BackFlush(const Tegra::MemoryManager& memory,
               std::span<const std::pair<DAddr, std::size_t>> ranges);

/// What the rasterizer knows of a draw once it has its pipeline (compare c, the draw kinds).
enum PipelineKind : u32 {
    PipelineTexel = 1,      ///< texel buffers: LateFill, a level draw
    PipelineDiagnostic = 2, ///< a diagnostic reads the bound buffers on the back
};
/// The pipeline of the draw the back is preparing: its program set and the digest of its facts.
void BackPipeline(const std::array<u64, 6>& hashes, u64 facts_digest, u32 kinds);

/// PrepareDraw: true after the draw's FillDraw, false before its record (draw_func). Read
/// flushes and collected-write applies in between are counted (the buffer half there would make
/// the draw level in step 6).
void BackDrawWindow(bool open);
/// PrepareDraw, before the vertex repack and the fill (piece 5c). SHADOW=3: the stage's copy the
/// fill reads instead of the engine, or null.
MemStageShadowEngine* BackFill(Tegra::Engines::Maxwell3D& maxwell3d, bool indirect);
/// The rasterizer wrote a GPU count into the back's draw state.
void BackDrawStateWritten();
/// The OoT trackers' record in the draw packet against today's query (mismatching slots, a mask).
void BackTrackerRecord(bool had_record, u32 mismatches);
/// RasterizerVulkan::FlushRegion (and FlushRegionForRead: all three) on the back: which caches'
/// halves run.
void BackFlushRegion(bool texture, bool buffer, bool query);
/// RasterizerVulkan::OnCacheInvalidations (`batch`, with its range count) or OnCacheInvalidation
/// (one range; the batch's fallback path calls it per range too) on the back.
void BackCacheInvalidation(bool batch, u64 ranges);

/// The texture cache's "pages that ever held an image" filter: constant buffer words onto those
/// pages make their entry level (join 2, spec 3.2). Set by the rasterizer at creation.
void SetWatchFilter(const PageFilter* filter);

// Piece 6b, FRITCHII_MEMSTAGE_THREAD=2 (memstage_shadow.cpp, memory_stage_link.cpp).

/// Set by the back at a frame end: every draw is level (a diagnostic, host conditional rendering).
inline std::atomic<bool> g_ring_level_draws{};

struct StageResult {
    size_t size;
    u8 cls; ///< pass, work, level, neutral
    bool hard;
    u32 packets;
    u32 launches; ///< Compute launches in the entry (0xaf, 0xb0, 0x107), for the run shape
    s32 channel;  ///< The channel the stage had bound at the entry, for the run shape
    bool ran_on_back; ///< The back ran a gather or a semaphore release of it on GPUMem's request
    bool publish_now; ///< GPUMem passed it (FRITCHII_MEMSTAGE_PASS): published at once
};
/// The back: ring mode starts (its hooks off) or ended.
void SetRingBack(bool on);
/// GPUMem: the stage's state starts over (every copy taken from the back at its first use).
void StageStart(s32 channel);
StageResult StageProcess(u64 position, bool in_run);
[[nodiscard]] bool StageExtendsRun(u64 position);
void StageLend();
/// Step 7, the window open: StageLend's flags are kept, and the back sets them at the run's first
/// entry (BackTakeLend, on the back; the publish of the run orders the two).
void StageLendDeferred();
void BackTakeLend();
/// FRITCHII_MEMSTAGE_GATHER, on the back inside GPUMem's request (GPUMem waits, level): the cache's
/// flags 15-48 to the back's engines before the gather, and back to the copies after it, as a run
/// of that one entry hands them over.
void BackGatherLend();
void BackGatherReturn();
/// BackGatherReturn when it goes out of scope, also when the gather or the closure throws (the
/// flags never stay on the back).
struct BackGatherFlags {
    BackGatherFlags() = default;
    ~BackGatherFlags() {
        BackGatherReturn();
    }
    BackGatherFlags(const BackGatherFlags&) = delete;
    BackGatherFlags& operator=(const BackGatherFlags&) = delete;
};
/// Step 7: whether processing the entry at `position` may take a full copy from the back (a copy
/// that is new or stale): GPUMem must be level there first.
[[nodiscard]] bool StageMayCopy(u64 position);
/// Returns the full copies it made.
u64 StageReturn(const void* drew_engine, const void* state_engine, s32 back_channel);
void StageTick();
[[nodiscard]] std::vector<std::pair<u64, std::size_t>>& StageRanges();
/// GPUMem, before it publishes past the entries it followed (PublishStaged) and when the ring
/// ends: FRITCHII_MEMSTAGE_CB_RUNS' run of constant buffer words ends, its bytes written and its
/// range collected, so the back (and the front's waits for it) never pass a word whose bytes are
/// not in game memory.
void StageEndWords();
/// Sets the stage's draw count to `back_seq`; returns what it was.
u64 StageDrawSeq(u64 back_seq);
/// GPUMem, right after StageProcess returned a hard entry: its first hard kind (the key of the
/// 5c "runs by first hard kind" line), and the key's name. For fable_stage_levelpoints 14.9's
/// lend to return by kind (memory_stage_link.cpp, GPU profile on).
[[nodiscard]] u64 StageEntryKey();
[[nodiscard]] std::string StageKindName(u64 key);
/// FRITCHII_DELAY_RUNKIND_CLASS's classes of a run's first hard kind (StageEntryKey): 1 the 3D
/// texture invalidate (0x3dd), 2 a compute engine method, 3 a 3D fence report, 4 the 3D wait for
/// idle (0x44), 5 a cache call (StageTag::Cache closure), 6 anything else.
[[nodiscard]] int StageKindClass(u64 key);
[[nodiscard]] u64 BackStateHash(const Tegra::Engines::Maxwell3D& maxwell3d);
/// The registers that select the programs (the six pipeline slots, then the program region), as
/// 8-byte words, and their hash: the facts table's key. The table compares the words on a key
/// match, so a hash collision is a miss, never another program's facts.
inline constexpr std::size_t ProgramWordCount = 49;
struct ProgramId {
    u64 key{};
    std::array<u64, ProgramWordCount> words{};
    /// The code itself, what the facts table is keyed by (2026-11-15: the words alone are no
    /// identity; the same program addresses in another channel, or after a remap, are other
    /// code): the address space (its memory manager's id + 1) and each program slot's device
    /// address (program region + offset through that address space; 0 when the slot is off or
    /// unmapped). ResolveProgramCode fills them.
    u64 space{};
    std::array<u64, 6> code{};
};
/// Fills `id.space` and `id.code` from `id.words` (ProgramWords' layout) through `memory`,
/// as ShaderCache::RefreshStages finds the code: with the native macro engine (`native`) a
/// slot's address is its reserved words, else program region + offset; the pixel slot is off
/// without `rasterize`; the geometry slot is off with `native` and `mesh`. Translated once per
/// change: kept per thread while the words, those three, the address space and its map serial
/// stay the same.
void ResolveProgramCode(const Tegra::MemoryManager& memory, bool native, bool rasterize,
                        bool mesh, ProgramId& id);
/// StageProgramWords' twin for the code: through the copy's back engine's address space.
void StageProgramCode(const MemStageShadowEngine& copy, ProgramId& id);
void BackProgramId(const Tegra::Engines::Maxwell3D& maxwell3d, ProgramId& out);
void StageProgramId(const MemStageShadowEngine& copy, ProgramId& out);
/// StageProgramId in two steps (FRITCHII_MEMSTAGE_SETUP_KEEP): the copy's program words, then
/// their key. StageProgramWords then ProgramWordsKey gives exactly StageProgramId's id.
void StageProgramWords(const MemStageShadowEngine& copy,
                       std::array<u64, ProgramWordCount>& words);
[[nodiscard]] u64 ProgramWordsKey(const std::array<u64, ProgramWordCount>& words);

} // namespace MemStageShadow

} // namespace VideoCommon
