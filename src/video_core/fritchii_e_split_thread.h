// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <type_traits>

#include "common/common_types.h"

namespace Common {
class SpinRecursiveMutex;
}
namespace Core {
class System;
}
namespace Tegra::Engines {
class EngineInterface;
class Maxwell3D;
}
namespace VideoCommon {
class GpuSplit;
}
namespace VideoCore {
class RasterizerInterface;
}

/// FRITCHII_ESPLIT (boot, default 0): the E split's thread, steps S1 and S2 of
/// work\E_SPLIT_DESIGN.md (2b, 5). The engine thread E (GpuSplit's back) becomes two:
///
///   E1 = the back thread itself: reads ring A, applies the register entries (engine A: the
///        real Maxwell3D, its dirty flags, the draw manager), looks the pipeline up.
///   E2 = "GPUBackE2", new: the memory work. Constant buffer words and every other E2 entry of
///        the dry classifier (fritchii_e_split.h), the draw's flush and tail (FlushCaching,
///        FillRing, the R packet's push, TickWork), a constant buffer bind's buffer cache half.
///        Its own view of the registers a fill reads (fritchii_e_split.h's View) is fed only by
///        E1's records.
///
/// S1, FRITCHII_ESPLIT_DEPTH=0 (live, default 0): lockstep. E1 hands each piece of E2 work over
/// and waits for it: one thread runs at a time, in today's order, on today's objects, so it is
/// exact by construction. E's per-call thread state moves with each hand-over (one follower's
/// draw values, lend site and query kind; the renderer's fill state, g_fill_state), E2 is E for
/// every check that asks (GpuSplit::OnBackThread, OneFollower::OnEngineThread), and a lock E1
/// holds that the work takes is handed over for the run.
///
/// Checks, logged every 10 s ("E split thread", must be 0):
///   ESplitView    E2's view against the engine after the record, 1 hand-over in 64.
///   ESplitInputs  on those draws (view equal), the fill reads the view and every input it can
///                 read is compared after the fill (as FRITCHII_ESPLIT_COUNT=3).
///   E1Touch       E1 held the buffer cache's lock, read or wrote game memory outside a join.
///   E2Order       E2 ran a hand-over out of ring order.
/// Measured: E2's busy time (zone "e split job"), E2's wait ("e split E2 wait"), E1's time in
/// hand-overs ("e split hand-over") and the hand-over cost (E1's wait less E2's run).
///
/// S2, FRITCHII_ESPLIT_DEPTH=N (live; default 16 = the queued mode): E1 never waits per piece of
/// work. It appends records to E2's queue (constant buffer writes by value, binds by value, draw
/// tails with their record, `consumed` stores) and goes on, up to N draw tails ahead; E2 drains
/// the queue on its own and sleeps only when it is empty. E1 waits only at a real join: an E2
/// entry that is not a constant buffer write (reports, sync points, wait for idle, barriers,
/// compute, cache calls, inline uploads; E1 runs it itself once E2 drained), a lend or push to R,
/// a decision that reads game memory, a write onto pages R or the lookup read, a seed. DEPTH=0 is
/// S1's lockstep, kept for the checks. The S2 section of the .cpp has the rules (who owns the
/// buffer flags, the R packet's packing on E1, the lends, the positions). Extra checks: E1Visible (a fence or query op made by E1 outside a join),
/// FrontAheadOfE2 (a `consumed` store on E1 with E2 behind), ESplitHash (a hash of what the fill
/// reads, from the engine on E1 and from the view on E2, 1 draw in 64), E2Refused (an async
/// fill refused after E1 moved on: the draw is dropped).
///
/// S3, FRITCHII_ESPLIT_PIN (boot, default 0): placement. 0 = E1 keeps E's pinned core, E2 runs
/// on the performance cores; 1 = E2 on E's old core (PinnedRole::GpuBack's mask), E1 unpinned on
/// the performance cores; 2 = E's core split by SMT sibling (E2 one hardware thread, E1 the
/// other); 3 = E2 on E's core, E1 on R's (PinnedRole::GpuMem, FRITCHII_ONEF_PIN's core). The lines
/// give each thread's CPU time per second against its busy time (a spinning thread reads about
/// 1,000 ms/s: E1 must stay under its busy share + 5%).
///
/// S4, FRITCHII_ESPLIT_CB_ASYNC (live, default 0; with DEPTH > 0): constant buffer words in two
/// halves. E1 runs the entry itself (the register half: CB_DATA, the shadow RAM, the dirty flags,
/// a deferred draw the first word issues, the offset advance) and hands the memory half (EXP107's
/// compare, the bound buffer or CB_DIRECT write, the invalidation, the stream-write guard, the
/// memory stage's stats) to E2 as a record in ring order with no wait: the words' values and
/// where they land (CbTarget) by value, several writes to a record. Writes onto texture or shader
/// pages or E's pools, across a GPU page, on Switch 2 sparse memory, unmapped, too long for a
/// record, or with the fast draw count or the price timers on, run on E1 in a join instead
/// (counted by cause). Off (and in S1): E1's own constant buffer writes outside a join (a back
/// macro's) are joins too. The .cpp's S4 section has the ordering argument. Checks (must be 0):
/// ESplitCbOrder (E2 applied a cb write out of order), ESplitCbBytes (1 cb write in 64: memory
/// read back after E2's write differs from the words), ESplitCbGuard (E2's stream-write guard
/// found tracked pages E1 found clear), ESplitCbRead (E1 read game memory outside a join with cb
/// writes queued).
///
/// FRITCHII_ESPLIT=0: nothing starts; each hand-over site pays one load and a branch (g_thread).
namespace VideoCommon::ESplit {

/// Where constant buffer words land, by value (Maxwell3D::FillConstantWordsTarget): the memory
/// half (Maxwell3D::WriteConstantWords) reads only this, so it can run after E1 moved on.
struct CbTarget {
    u64 address;  ///< CB_ADDRESS
    u64 tic;      ///< the texture header pool (EXP107's alias check; with `pools`)
    u64 tic_size;
    u64 tsc;      ///< the sampler pool
    u64 tsc_size;
    u32 size;     ///< CB_SIZE
    u32 offset;   ///< the offset before the words
    bool count_fast_draw; ///< EDEN_NX2_FASTDRAW_COUNT on (its notes read the registers)
    bool pools;           ///< the pools above are filled (the thread on); else read them then
    bool may_defer;       ///< nothing but this target is read (no fast draw count, no price timers)
};

struct View;

/// What E1 hands to E2.
enum class Hand : u8 {
    Entry,    ///< an E2 entry of ring A (GpuSplit::Execute on E2)
    Flush,    ///< a draw's FlushCaching, before E1's pipeline lookup
    DrawTail, ///< a draw's FillRing, R packet push and TickWork
    Bind,     ///< a constant buffer bind's buffer cache half, by value
    Store,    ///< S2: GpuSplit::StoreConsumed at E2's place
    CbWords,  ///< S4: constant buffer words' memory half, by value (several writes a record)
    Async,    ///< an E2 entry queued whole (a cache call, a compute engine entry)
    Raster,   ///< a 3D method's rasterizer call, by value (its register half ran on E1)
    Op,       ///< a posted op E1 packed, committed to R's queue by E2
    Count,
};

/// Where E1 runs E2-owned work itself (S1: E2 is idle anyway; S2: E2 drained first).
enum class Join : u8 {
    Entry,     ///< a join entry of the classifier (uploads, other engines, untagged calls, channel)
    Decision,  ///< E1's own decision reads game memory (the indirect reset test)
    Condition, ///< a render condition from game memory
    Idle,      ///< the idle hook and the fence check (E's idle work)
    Push,      ///< S2: E1 lends or pushes to R (R's packet queue has one producer)
    Lockstep,  ///< S2: a hand-over that waits (none in the queued mode)
    CbWrite,   ///< S4: a constant buffer write E1 makes itself (not async, or a read-back after)
    Lookup,    ///< E1's lookup after a write onto shader pages (E2 drained, flushed first)
    Flush,     ///< the draw's flush in the queued mode, run by E1 (E2 idle or drained)
    Tail,      ///< a draw tail that cannot go async in the queued mode (E1 runs it)
    Seed,      ///< E2's view seeded in the queued mode (first draw, channel switch; by E1)
    Count,
};

/// FRITCHII_ESPLIT=1 and E2 started (written by E1 before E2 exists, read everywhere).
inline bool g_thread{};
/// S2: FRITCHII_ESPLIT_DEPTH > 0 (E1 writes it at a batch's start with E2 drained).
inline bool g_async{};
/// S4: FRITCHII_ESPLIT_CB_ASYNC on while async (E1 writes it at a batch's start).
inline bool g_cb_async{};
/// 1 on E1, 2 on E2.
inline thread_local u8 t_role{};
/// E2 inside an async constant buffer record (OneFollower::GuardStreamWrite asks).
inline thread_local bool t_async_cb{};
/// E1 inside a join (it runs E2-owned work itself).
inline thread_local u32 t_join{};

/// The boot switch (read once; FRITCHII_ONEF=1 is needed, checked at the start).
[[nodiscard]] bool ThreadWanted() noexcept;
/// GpuSplit::BackLoop at its start (E1): E2 starts; `adopt` runs first on E2 (GpuSplit's own
/// thread marks).
void StartE2(Core::System& system, void (*adopt)(void*), void* adopt_argument);
/// GpuSplit::BackLoop at its end (E1): E2 drains and stops.
void StopE2();
/// E1, once a batch: FRITCHII_ESPLIT_DEPTH, the 10 s lines.
void ThreadBatch() noexcept;

[[nodiscard]] inline bool OnE1() noexcept {
    return t_role == 1;
}
[[nodiscard]] inline bool OnE2() noexcept {
    return t_role == 2;
}
/// E2's work here goes to E2: E1, the thread on, outside a join.
[[nodiscard]] inline bool ShouldHand() noexcept {
    return g_thread && t_role == 1 && t_join == 0;
}
/// E1 inside its pipeline lookup (LookupScope): the lookup's shader code reads are allowed.
inline thread_local bool t_lookup{};
/// A write onto shader pages since E1's last lookup flush (GuardStreamWrite's hit, any thread):
/// the next lookup drains E2 and flushes first (a join), so the lookup reads the new code. (S4's
/// owed rule flushes before the lookup after such work anyway; this is the belt to it.)
inline std::atomic<bool> g_shader_pending{};

void BeginLookupImpl() noexcept;
/// E1: true when a queued write onto shader pages has not yet been drained by a lookup.
[[nodiscard]] inline bool ShaderPending() noexcept {
    return g_shader_pending.load(std::memory_order_acquire);
}
/// Around E1's pipeline lookup (FillAndPush, rule 3): drains and flushes first when a shader page
/// write is pending (a "lookup join"); the lookup's reads of shader code count as allowed.
class LookupScope {
public:
    LookupScope() noexcept {
        if (g_thread && t_role == 1 && t_join == 0) [[unlikely]] {
            BeginLookupImpl();
            active = true;
        }
    }
    ~LookupScope() {
        if (active) [[unlikely]] {
            t_lookup = false;
        }
    }
    LookupScope(const LookupScope&) = delete;
    LookupScope& operator=(const LookupScope&) = delete;

private:
    bool active{};
};

/// OneFollower::GuardStreamWrite on E2 while E1 runs ahead (a write E1 did not check): E1 held
/// for the guard (it reads E1's pool ranges and may drain R). True: release with E2ExclusiveEnd.
[[nodiscard]] bool E2ExclusiveBegin() noexcept;
void E2ExclusiveEnd() noexcept;

struct HandInfo {
    Hand hand{Hand::Entry};
    bool record{};    ///< E1 builds a record for E2's view first
    bool draw{};      ///< the record carries the draw state
    bool list_args{}; ///< ... and a list draw's indirect parameters
    bool cb_entry{};  ///< a constant buffer entry: E2's view offset advances after it
    u32 cb_words{};   ///< its words, 0: what `cb_size` holds after the run
    const std::size_t* cb_size{};
    /// A lock E1 holds that the work takes: E2 holds it for the run (SpinRecursiveMutex::HandOver).
    Common::SpinRecursiveMutex* held{};
};

/// E1: `function(argument)` runs on E2; E1 waits for it (lockstep, also in S2: E2 drains first).
void Run(const HandInfo& info, void (*function)(void*), void* argument) noexcept;
template <typename F>
void HandTo(const HandInfo& info, F& body) noexcept {
    Run(info, [](void* raw) { (*static_cast<F*>(raw))(); }, &body);
}

/// FRITCHII_ESPLIT_BIND_BATCH (live, default 0; the queued mode): constant buffer binds without a
/// job each. 1 = a bind goes into E1's open record (the constant buffer words' record, S4) as a
/// segment, applied on E2 in record order between the words around it; a bind of the same value
/// as the last bind of its slot earlier in the same open record is dropped. 2 = verify: the same
/// packing, nothing dropped, the would-drop binds checked on E2 (ESplitBindDrop, must be 0: the
/// slot's state hash changed across the bind) and packed binds watched for cached writes (a count,
/// as the async jobs' one: R's concurrent writes show too). 3 = dry: binds stay one job each, the
/// line counts what 1 would drop and pack. ESplitBindOrder (1 and 2, must be 0): a packed bind
/// applied out of E1's order. E1 writes it at a batch's start.
inline int g_bind_batch{};

/// Maxwell3D::ProcessCBBind's buffer cache half, handed over by value (async in S2).
void HandBind(VideoCore::RasterizerInterface& rasterizer, std::size_t stage, u32 slot,
              bool enabled, u64 address, u32 size) noexcept;

void BeginJoin(Join why) noexcept;
void EndJoin() noexcept;
/// E1 runs E2-owned work itself here.
class JoinScope {
public:
    explicit JoinScope(Join why, bool on = true) noexcept {
        if (on && g_thread && t_role == 1) [[unlikely]] {
            BeginJoin(why);
            active = true;
        }
    }
    ~JoinScope() {
        if (active) [[unlikely]] {
            EndJoin();
        }
    }
    JoinScope(const JoinScope&) = delete;
    JoinScope& operator=(const JoinScope&) = delete;

private:
    bool active{};
};

/// E2, in a draw tail's FillRing: E2's view when the fill reads it (S1: the draw's sampled
/// compare found it equal; no inline index bytes), else null (the fill reads the engine).
[[nodiscard]] View* FillView(const Tegra::Engines::Maxwell3D& engine) noexcept;
/// After that fill (E2, or E1 in a join): the view's buffer flags, the inputs check.
void FillDone(const Tegra::Engines::Maxwell3D& engine, const View* used) noexcept;

// --- S2 (FRITCHII_ESPLIT_DEPTH > 0) -----------------------------------------------------------------

/// E2's ring position while it runs an async job (GpuSplit::BackEntryPosition on E2).
inline thread_local u64 t_e2_position{};
/// E2 asks E1 to hold at its next safe point (an E2 lend: R's lent call may read E1's state).
inline std::atomic<u32> g_hold{};
void HoldHereImpl() noexcept;
/// E1, before each ring entry: holds while E2 lends.
inline void SafePoint() noexcept {
    if (g_async) [[unlikely]] {
        if (g_hold.load(std::memory_order_relaxed) != 0) {
            HoldHereImpl();
        }
    }
}

/// E1: E2 has run everything posted.
[[nodiscard]] bool E2Idle() noexcept;
/// E1: the draw's flush can run before the lookup (lockstep: E2 idle, or S1).
[[nodiscard]] inline bool FlushBeforeLookup() noexcept {
    return !g_async || E2Idle();
}

/// Where E1 puts a draw's flush (OneFollowerDraw::FillAndPush).
enum class FlushAt : u8 {
    Before,  ///< E2 idle: before the lookup, a lockstep hand-over when it has work (today's)
    Drained, ///< owed with E2 behind: E2 drains, flushes (when it has work), then the lookup
    Tail,    ///< at the head of the draw's tail on E2, after E1's lookup
};
/// S2/S4: a flush is owed (must run before the lookup) after any work that may write game memory
/// other than queued constant buffer writes (on no shader page, by E1's test): a lockstep
/// hand-over, a join, E1's push or lend, an E2 lend that held E1. Not owed: Tail when E2 is
/// behind, or always with FRITCHII_ESPLIT_FLUSH_ASYNC. The lookup reads shader code and the
/// pipeline cache only, which a not-owed flush cannot change.
[[nodiscard]] FlushAt FlushPlace() noexcept;

/// E1, S2: a slot for an async hand-over. `packet`: room for OneFollower::PackedDrawRoom().
struct AsyncSlot {
    void* payload;
    u8* packet;
};
inline constexpr std::size_t AsyncPayloadBytes = 192;
/// E1: an async draw tail can go now (record ready); waits for depth room, builds the record
/// (taking the engine's buffer flags) and samples the hash. False: hand it over in lockstep.
[[nodiscard]] bool BeginAsyncDraw(Tegra::Engines::Maxwell3D& engine, bool list_args,
                                  AsyncSlot& slot) noexcept;
/// E1: posts the slot BeginAsyncDraw (or BeginAsync) gave: `function(payload)` runs on E2.
void PostAsync(void (*function)(void*)) noexcept;
/// E1: an async slot with a packet buffer (OneFollower::PackedDrawRoom bytes) in `packet`.
[[nodiscard]] void* BeginAsyncPacket(Hand hand, u8*& packet) noexcept;
/// E1: a slot for an async job without a record (a bind by value, a position store).
[[nodiscard]] void* BeginAsync(Hand hand) noexcept;
/// E1: a draw tail stayed with E1 (0: no async fit, 3: S1 lockstep), counted.
void NoteTailStayed(int reason) noexcept;
/// E2 in the queued mode: E2's view's buffer flags (Maxwell3D::DirtyState::Flags) when `engine` is
/// the view's, else null (CacheInputs' constructor sends the cache's flag marks there).
[[nodiscard]] void* E2FlagsFor(const Tegra::Engines::Maxwell3D* engine) noexcept;
/// E2: the running job is async (its fill reads only the view and keeps the flags there).
[[nodiscard]] bool FillAsync() noexcept;
/// E2, async fill: the repack layout raised `flag` (VertexInput) for E1's engine; E1 raises it
/// at its next draw, join, push or lockstep hand-over (ApplyBackRaises).
void DeferFlag(u32 flag) noexcept;
void ApplyBackRaises(Tegra::Engines::Maxwell3D& engine) noexcept;
/// E2, async: a ring B room wait, a flush after E1's lookup, a fill refused (E2Refused).
void NoteRoomWait() noexcept;
void NoteLateFlush() noexcept;
void NoteAsyncRefused() noexcept;
/// GpuSplit::BackLoop's `consumed` store on E1: FrontAheadOfE2 when E2 is behind.
void CheckStoreOnE1() noexcept;

/// OneFollower::Lend, PushOp, Push, Drain, FlushCarried: R's packet queue has one producer and a
/// lent call may read E's state. E1 (S2): E2 drained first, the buffer flags to the engine; E2 in
/// a lend (S2): E1 held. `visible`: a fence or query op (E1Visible when E1 makes it).
/// What E1 pushes to R (the push joins are counted and timed by it).
enum class PushKind : u8 {
    Lend,    ///< OneFollower::Lend (E parks until R ran the call)
    Op,      ///< PushOp, a posted op (R runs it later)
    Push,    ///< OneFollower::Push
    Drain,   ///< OneFollower::Drain (E waits for R to play everything)
    Carried, ///< FlushCarried (the kept ops at a `consumed` store)
    Count,
};
[[nodiscard]] bool BeforeEnginePushImpl(bool lend, bool visible, PushKind kind) noexcept;
void AfterEnginePushImpl(bool lend, bool acted) noexcept;
class PushGuard {
public:
    PushGuard(bool lend_, bool visible, PushKind kind = PushKind::Push) noexcept : lend{lend_} {
        if (g_thread) [[unlikely]] {
            acted = BeforeEnginePushImpl(lend, visible, kind);
        }
    }
    ~PushGuard() {
        if (acted) [[unlikely]] {
            AfterEnginePushImpl(lend, acted);
        }
    }
    PushGuard(const PushGuard&) = delete;
    PushGuard& operator=(const PushGuard&) = delete;

private:
    bool lend;
    bool acted{};
};

// --- S4: constant buffer words (FRITCHII_ESPLIT_CB_ASYNC) -------------------------------------------

/// Maxwell3D's constant buffer write on E1 outside a join (thread on): queued for E2 or run in a
/// join; E2's view gets the offset in E1's next record. True: done, the caller only advances the
/// offset. False (E2, a join, the thread off): the caller writes as today.
[[nodiscard]] bool CbWriteImpl(Tegra::Engines::Maxwell3D& engine, const CbTarget& target,
                               const u32* values, u32 bytes, bool block) noexcept;
[[nodiscard]] inline bool CbWrite(Tegra::Engines::Maxwell3D& engine, const CbTarget& target,
                                  const u32* values, u32 bytes, bool block) noexcept {
    if (t_role == 1 && t_join == 0 && g_thread) [[unlikely]] {
        return CbWriteImpl(engine, target, values, bytes, block);
    }
    return false;
}
/// GpuSplit::BackLoop: a constant buffer entry E1 runs itself (its writes queued for E2).
[[nodiscard]] inline bool CbEntryOnE1() noexcept {
    return g_cb_async && t_role == 1 && t_join == 0;
}
/// E1: the open constant buffer record goes to E2 now. Also done before any other hand-over,
/// drain, join, push or wait of E1's, so nothing E1 or the front does later can pass it.
void PostCbWrites() noexcept;
/// E1: queues a contiguous compute/other engine entry (or nullptr-engine call) for E2,
/// coalescing consecutive entries on the same engine into a single async job.
void QueueEngineEntry(VideoCommon::GpuSplit& split, Tegra::Engines::EngineInterface* engine,
                      const u64* entry, std::size_t words) noexcept;
/// E1: records `GpuSplit::StoreConsumed(position)` to run on E2 in job order. Non-urgent advances
/// coalesce onto the open/next job so `PostPendingBatches` is not flushed prematurely.
void QueueStore(u64 position, bool urgent) noexcept;
/// E1: drops any deferred store position when an immediately following BatchEnd job supersedes it.
void DiscardPendingStore() noexcept;
/// E1: waits until `done_seq >= target_seq` without draining unrelated later jobs.
void WaitConditionImpl(const std::atomic<u64>& done_seq, u64 target_seq) noexcept;
inline void WaitCondition(const std::atomic<u64>& done_seq, u64 target_seq) noexcept {
    if (done_seq.load(std::memory_order_acquire) >= target_seq) {
        return;
    }
    WaitConditionImpl(done_seq, target_seq);
}
/// E2's stream-write guard found tracked pages in an async constant buffer record
/// (ESplitCbGuard, must be 0: E1 found them clear when it queued the write).
void NoteCbGuardHit() noexcept;

// --- The queued mode's entry kinds (FRITCHII_ESPLIT_ASYNC_KINDS) ---------------------------------

/// Which E2 entries queue instead of joining (E2 drained, E1 runs them). Each is exact by
/// construction: E2 runs it in ring order after everything posted before it, and it reads nothing
/// E1 changes later.
///   bit 0, cache calls: Call entries tagged StageTag::Cache (fence releases, GPU cache
///          invalidations, reference signals, the puller's wait for idle and semaphore releases,
///          the compute list words). The tag says they use only back-side objects and write no
///          3D state; their payload is in ring A, which frees only at E2's place.
///   bit 1, raster ops: wait_for_idle, fragment_barrier, invalidate_texture_data_cache,
///          tiled_cache_barrier, report_semaphore.query, clear_report_value, sync_info. E1 runs
///          the method (register write, dirty flags, the values the handler reads); the one
///          rasterizer call it ends in goes to E2 by value (QueueRasterOp).
///   bit 3, posted ops: OneFollower::PushOp from E1 (an op R runs later, E not parked). E1 packs
///          the spans (its registers at the op's place) and the payload into a packet, E2
///          commits it to R's queue in ring order, as an async draw's packet; no drain.
///   bit 2, compute engine entries: every Method, Multi and Pairs entry of the compute engine,
///          its sink included, on E2 with the engine named by value. KeplerCompute and the
///          rasterizer's DispatchCompute read no Maxwell3D register (checked in the code: the
///          OoT network bypass reads the tonemap target under its own lock, recorded on R as
///          today); the engine is E2's alone then (E1 touches it only in joins, E2 drained).
enum AsyncKindBit : u32 {
    KindCacheCalls = 1,
    KindRasterOps = 2,
    KindCompute = 4,
    KindPostedOps = 8,
    KindsDefault = KindCacheCalls | KindRasterOps | KindCompute | KindPostedOps,
};
/// E1 writes it at a batch's start, E2 drained.
inline u32 g_async_kinds{};
[[nodiscard]] inline bool QueueKind(u32 bit) noexcept {
    return g_async && (g_async_kinds & bit) != 0 && t_role == 1 && t_join == 0;
}

enum class RasterOp : u8 {
    WaitForIdle,
    FragmentBarrier,
    InvalidateTextureDataCache,
    TiledCacheBarrier,
    Query,           ///< address, type, flags, payload, subreport
    ResetCounter,    ///< type
    SignalSyncPoint, ///< value
};
/// The renderer's: a samples counter reset (QueryType::ZPassPixelCount64) E1 queues for E2 clears
/// E1's copy of the one follower's samples mirror at once (RasterizerVulkan::onef_mirror_queued).
inline void (*g_mirror_reset)(VideoCore::RasterizerInterface* rasterizer) = nullptr;
void QueueRasterOpImpl(RasterOp op, VideoCore::RasterizerInterface& rasterizer, u64 address,
                       u32 a, u32 b, u32 c, u32 d) noexcept;
/// Maxwell3D, at the rasterizer call a raster-op method ends in: true when it went to E2's queue
/// (E1 outside a join, bit 1 on); false: the caller calls the rasterizer as before.
[[nodiscard]] inline bool QueueRasterOp(RasterOp op, VideoCore::RasterizerInterface& rasterizer,
                                        u64 address = 0, u32 a = 0, u32 b = 0, u32 c = 0,
                                        u32 d = 0) noexcept {
    if (QueueKind(KindRasterOps)) [[unlikely]] {
        QueueRasterOpImpl(op, rasterizer, address, a, b, c, d);
        return true;
    }
    return false;
}

// --- E1's batch end and idle (FRITCHII_ESPLIT_IDLE), the owed flush (FRITCHII_ESPLIT_OWED) -------

/// FRITCHII_ESPLIT_IDLE=1 (default): E1 never drains E2 because its own ring ran out. The batch's
/// end (the fence quiet check and the `consumed` store at that place) goes to E2 as a record, and
/// with no new ring words E1 waits for E2 to go idle or for the front to publish, whichever comes
/// first, before E's idle work (the idle hook, which then finds E2 idle: no wait). Before, each
/// batch end drained E2 (a join) while the front's new words waited for E1.
inline bool g_idle_async{};
[[nodiscard]] inline bool IdleAsync() noexcept {
    return g_idle_async && g_async && t_role == 1 && t_join == 0;
}
bool WaitE2IdleOrImpl(bool (*more)(void*), void* argument) noexcept;
/// E1 with no ring words: waits until E2 is idle (false) or `more()` (true).
template <typename More>
[[nodiscard]] bool WaitE2IdleOr(More&& more) noexcept {
    return WaitE2IdleOrImpl([](void* raw) { return (*static_cast<std::remove_reference_t<More>*>(raw))(); },
                            &more);
}

/// FRITCHII_ESPLIT_OWED=1 (default): an async job's cached writes no longer make the next flush
/// owed (S4's ESplitAsyncWrite counted any write during the job, R's included, and is a count
/// now). A late flush can only matter to E1's lookup through shader pages, so: a queued write
/// whose destination E1 knows is tested against them at the queue (report queries, semaphore
/// releases; constant buffer writes by S4's own test), a compute upload counts as one, and any
/// E thread's write that hits them sets g_shader_pending (the next lookup drains and flushes).
/// ESplitLateShader (must be 0, 1 late flush in 16): the ranges a late flush on E2 invalidates
/// after E1's lookup land on no shader page.
inline bool g_owed_shader_rule{};
/// E1: a queued write to `address` (`size` bytes) through `memory` (Tegra::MemoryManager*).
void NoteQueuedWrite(const void* memory, u64 address, u64 size) noexcept;
/// E1: FRITCHII_ESPLIT_IDLE's batch end went to E2 (counted).
void NoteBatchEnd() noexcept;
/// E1: a queued job may write game memory anywhere (a compute upload).
void NoteQueuedWriteAnywhere() noexcept;
/// E2, before a late flush (after E1's lookup): the sampled check (`memory`: Tegra::MemoryManager*).
void CheckLateFlush(const void* memory) noexcept;

/// MemoryManager::FlushCaching on E1 outside a join (queued mode): it goes to E2's queue.
[[nodiscard]] inline bool QueueFlushHere() noexcept {
    return g_async && t_role == 1 && t_join == 0;
}
/// E1: posts `memory`'s (Tegra::MemoryManager*) FlushCaching for E2 at this place.
void QueueFlushCaching(void* memory) noexcept;
void NoteCachingTouchImpl() noexcept;
/// MemoryManager's cached-write state (the accumulator, the constant buffer ranges) touched here:
/// in the queued mode only E2, E1 inside a join (E2 drained), or R in a lent call (E parked) may.
/// Anything else is ESplitCachingOwner (must be 0).
inline void NoteCachingTouch() noexcept {
    if (g_async) [[unlikely]] {
        NoteCachingTouchImpl();
    }
}

/// GpuSplit::BackLoop, the queued mode: an entry that joined, by kind (the 10 s line names the
/// top 8): ring kind, engine (0 3D, 1 compute, 2 other, 3 none), method or call tag, and a
/// call's function.
void NoteJoinEntry(u32 kind, u32 engine, u32 method_or_tag, const void* function) noexcept;

// --- E1Touch ---------------------------------------------------------------------------------------

enum class Touch : u8 { CacheLock, MemoryRead, MemoryWrite, Count };
void NoteTouchImpl(Touch kind, const void* caller) noexcept;
/// The buffer cache's lock released (its release hook, set at the cache's start with the switch).
inline void NoteCacheRelease() noexcept {
    if (t_role == 1 && t_join == 0 && g_thread) [[unlikely]] {
        NoteTouchImpl(Touch::CacheLock, nullptr);
    }
}
/// E's game memory write (OneFollower::GuardStreamWrite).
inline void NoteMemoryWrite() noexcept {
    if (t_role == 1 && t_join == 0 && g_thread) [[unlikely]] {
        NoteTouchImpl(Touch::MemoryWrite, nullptr);
    }
}

// --- The renderer's per-thread fill state (MemStage), moved like one follower's ------------------

struct FillStateHooks {
    /// The calling thread's state into `out` (FillStateBytes); `as_owner`: with the thread's own
    /// identity filled in (E1, for E2's run of its work).
    void (*capture)(void* out, bool as_owner){};
    /// Installs it; `keep_identity`: the calling thread keeps its own identity (E1, back).
    void (*install)(const void* in, bool keep_identity){};
};
inline constexpr std::size_t FillStateBytes = 128;
/// Set by the renderer at static initialization (vk_mem_stage.cpp).
inline FillStateHooks g_fill_state{};

} // namespace VideoCommon::ESplit
