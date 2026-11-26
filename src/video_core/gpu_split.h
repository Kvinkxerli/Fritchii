// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <optional>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <vector>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "common/common_types.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/fritchii_front_split.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/split_write_tracker.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon {

/// Writer ranges compared by GpuSplit::WritersMayWriteSince scans (logged with the front DMA
/// snapshot line): what the larger scan limit costs the front.
inline std::atomic<u64> g_writer_ranges_compared{0};

/// Why the split's front waited for its back (EDEN_NX2_GPU_PROFILE counts each).
enum class DrainSite : u32 {
    Other,
    FetchDrain,     ///< EDEN_NX2_FETCH_DRAIN: before every entry
    Acquire,        ///< A semaphore acquire reads the value the back may release
    SemaphoreFlush, ///< The acquired semaphore is in the buffer cache
    FenceRelease,   ///< The puller's acquires release the pending fences
    ListWriters,    ///< GPU-written command lists: the dispatches and copies sent before them
    ListSync,       ///< Command lists on draw-written pages: the work before the last host sync
    ListFlush,      ///< Command lists on GPU-written pages: drain and flush the pages
    MacroDmaRead,   ///< The front's macros read memory (MME DMA)
    MacroDmaWrites, ///< The same, waiting only for the queued work that may write it
    MacroDmaFlush,  ///< Those words are in the buffer cache
    GpSyncFence,    ///< A GP entry's sync bit (sync_memory_operations)
    Map,            ///< GPU memory maps before later command reads
    FrontStart,     ///< The front takes an engine's macros over
    FrontVerify,    ///< EDEN_NX2_FRONT_VERIFY
    MacroDmaCheck,  ///< EDEN_NX2_CMDPAGE_DIAG: a macro's memory read repeated after a full drain
    DispatchWait,   ///< EXP10: a macro memory read waits for the dispatches sent since it flushed
    FenceWriters,   ///< EXP12: a fence release waits for the writers a semaphore read can't see
    QmdRead,        ///< EXP13=2: the front's QMD scheduler copy reads a QMD queued work writes
    ListCopies,     ///< Lists on GPU-written pages: the dispatches and copies before the last sync
    LaggedCheck,    ///< EXP12=5/6: a suspended acquire's poll waits for what was sent 1 ms before
    FenceWordWriters, ///< EXP12=6: a fence release waits for the queued writers of its word
    /// FRITCHII_ONEF_SEM_E: a lagged check on a word whose newest release is a payload one,
    /// written on E (or lent): E's position
    LaggedCheckEngine,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DrainSite::Count)> drain_site_names{
    "other",       "fetch drain",     "acquire",    "semaphore flush", "fence release",
    "list writers", "list sync",      "list flush", "macro dma read",  "macro dma writes",
    "macro dma flush",
    "gp sync fence", "map",           "front start", "front verify",  "macro dma check",
    "macro dma dispatch", "fence writers", "qmd read", "list copy writers", "lagged check",
    "fence word writers", "lagged check (E)",
};

/// Maxwell3D::FrontNoteMemoryWrite's notes of 3D reports (the pending write's `what`), told apart
/// by address in the "Front 0xe2c misses" line (inline variables: one address in every unit).
/// A payload report with the timestamp (16 bytes, report Payload, not short).
inline constexpr char NotedPayloadReport[] = "3D payload report";
/// A short payload release (4 bytes, its word known).
inline constexpr char NotedPayloadRelease[] = "3D payload release";
/// Any other report: counters and sample counts (written when the host query completes).
inline constexpr char NotedCounterReport[] = "3D report";

/// EXP13=2: which queued writer set GpuSplit::QueuedGuestWriters' position (a wait cause).
enum class GuestWriterCause : u8 {
    None,
    Pending,          ///< A noted write over the bytes (NoteMemoryWrite, NoteMemoryWriteData)
    WriterRange,      ///< A compute or copy range over them (NoteWriterRange, NoteReleaseWriter)
    Ranged,           ///< A 3D ranged write over them (RangeLastWrite)
    UnboundedWriter,  ///< Compute or copy work without a range (NoteUnboundedWriter)
    Unranged,         ///< KeplerMemory, Fermi2D, syncpoint operations (NoteUnrangedWriter)
    Function,         ///< A queued function (GPU memory maps)
    MethodWrite,      ///< A 3D memory-writing method without a range (NoteMethodWrite)
    UnknownPending,   ///< A noted write of unknown range (its `what` says which)
    PendingScanLimit, ///< More noted writes queued than are remembered
    WriterScanLimit,  ///< More writer ranges queued than are compared
    OverlayEvicted,   ///< KeplerCompute's QMD copy: an evicted inline QMD may be the writer
    /// The newest ranged write RangeLastWrite dropped from its full ring (MaxRangedWrites): the
    /// ring scans no longer see its range, so it may write anything for them
    RangedOverflow,
    Count,
};

/// The QMD wait floor counters (fable_qmd_wait_brief piece 1): which term of
/// GpuSplit::WriteStamp set its answer, newest first; on a tie the earlier one here.
enum class StampTerm : u8 {
    None,           ///< Nothing: the answer is 0
    Block,          ///< A 256-byte block stamp over the bytes
    Page,           ///< A 4 KB page stamp over them
    Floor,          ///< Stamps on since, or a write over 1 MiB (write_stamp_floor)
    Unknown,        ///< A noted write of unknown range (unknown_write_position)
    Unbounded,      ///< Compute or copy work without a range
    Unranged,       ///< KeplerMemory, Fermi2D, syncpoint operations
    Function,       ///< A queued function (GPU memory maps)
    MethodWrite,    ///< A 3D memory-writing method without a range
    RangedOverflow, ///< The ranged ring's overflow (ranged_overflow_position)
    Count,
};

/// The same counters: what happened to a small inline upload patching a QMD no shadow or overlay
/// holds (KeplerCompute::FrontPatchOverlays), per 256-byte block it writes.
enum class QmdPatchOutcome : u8 {
    Holder,       ///< An overlay or shadow took it (PatchHolders)
    Made,         ///< An overlay from memory with the patch over it
    /// Of Made: built while the ranged overflow was queued (only when WriteStamp leaves it out)
    MadeFloorUp,
    Unmapped,     ///< Refused: not mapped, or GPU-written
    InlineQueued, ///< Refused: an overlay of it (an inline QMD's mostly) is still queued
    Evicted,      ///< Refused: an evicted inline QMD may be
    Stamp,        ///< Refused by the write stamps (by term: NoteQmdPatchRefused)
    Count,
};

// Splits the GPU thread in two. The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// Counts starts of the split (EXP13=2: the front's QMD scheduler copy starts over).
    [[nodiscard]] u64 Session() const {
        return session;
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }

    /// EDEN_NX2_MACRO_SINK=1 (default off): the state writes the front's 3D macros make
    /// (registers outside execution_mask) go to the back's method sink like the command list's
    /// own state writes, instead of one CallMethod each; the back applies its sink, in order,
    /// before the engine's next executable method.
    [[nodiscard]] static bool MacroSink() {
        return macro_sink;
    }
    /// EDEN_NX2_COMPUTE_SINK=1 (default off): native compute writes that only store their value
    /// (native_plain_methods), from macros or the command list, are dropped when the front's
    /// register copy shows the back holds the value already, else sent batched (SinkMethod).
    [[nodiscard]] static bool ComputeSink() {
        return compute_sink;
    }
    /// EDEN_NX2_RING_PAIRS=1 (default off): the state writes that would go to the back's method
    /// sink (the command list's 3D state writes; the 3D macros' with EDEN_NX2_MACRO_SINK, the
    /// compute ones with EDEN_NX2_COMPUTE_SINK) go as (method, value) pairs in Pairs blocks
    /// (PairWrite), which the back applies in one loop per block (EngineInterface::ApplyPairs),
    /// after its method sink.
    [[nodiscard]] static bool RingPairs() {
        return ring_pairs;
    }

    /// The front's writes that took a batched path (EDEN_NX2_MACRO_SINK, EDEN_NX2_RING_PAIRS),
    /// logged every 10 s.
    enum class Batched : u32 {
        MacroSunk,   ///< 3D macro state writes sent to the method sink (or as pairs)
        ComputeSunk, ///< Compute state writes sent to the method sink (or as pairs)
        ComputeSame, ///< Compute state writes dropped: the register held the value
        Pairs,       ///< Writes sent as pairs (any engine, any source)
        PairBlocks,  ///< Pairs blocks opened
        DeltaWrites, ///< EXP22: 3D macro state writes sent in StateDelta blocks
        DeltaBlocks, ///< EXP22: StateDelta blocks opened
        DrawBatches, ///< EXP22: DrawBatch entries sent
        DrawRepeats, ///< EXP22: instance-loop groups they stand for past their first
        SameIndexFirst, ///< EXP2 or EXP22 bit 4: same-value index_buffer.first writes not sent
        Count,
    };
    void CountBatched(Batched counter, u64 amount = 1) {
        batched[static_cast<size_t>(counter)] += amount;
    }

    /// Whether the calling thread is the back.
    /// Back: true while its idle hook runs because it ran out of work (EXP99=1 destroys only then).
    [[nodiscard]] static bool InIdleWait() {
        return in_idle_wait;
    }
    [[nodiscard]] static bool OnBackThread() {
        return on_back_thread;
    }
    /// One follower (one_follower.h): R, the render thread, runs the back's render half (lent
    /// calls with the back parked, unlent draws at step 4), so it is the back for every check
    /// that asks (submits, idle-only work, the back-only paths of the caches).
    static void MarkRenderThread() {
        on_back_thread = true;
    }
    /// One follower (fable_b_one_follower.md 8.2): two positions. `consumed` is E's (what the
    /// back ran); `played` is R's: the ring position before which R has finished every packet
    /// too. Every front wait reads PlayedNow, "R has played P", which is today's meaning (all
    /// work before P done); step 5 moves single wait kinds to E's position. Without one follower
    /// PlayedNow is `consumed`, as before.
    [[nodiscard]] u64 PlayedNow(std::memory_order order) const {
        return two_positions ? played.load(order) : consumed.load(order);
    }
    /// FRITCHII_ONEF_WAITS_E (fable_b_one_follower 11.2): whether a wait at `site` reads E's
    /// position: game memory, QMDs, semaphore flushes, lists and the dispatch records are E's;
    /// everything else (acquires, fences, the lagged check, report flushes, maps) waits for R.
    [[nodiscard]] static bool SiteOnEngine(DrainSite site) noexcept;
    /// E's own position (step 5's E-only waits; with one follower off the same as PlayedNow).
    [[nodiscard]] u64 EngineConsumedNow() const {
        return consumed.load(std::memory_order_acquire);
    }
    /// What the front published (one follower's hang watchdog).
    [[nodiscard]] u64 PublishedNow() const {
        return published.load(std::memory_order_acquire);
    }
    /// One follower's hang watchdog: the front's wait now (WaitFor, Drain). `begin_ns` 0 = not
    /// waiting; `seq` counts the waits, so one long wait tells from many short ones.
    struct FrontWait {
        std::atomic<u32> site{static_cast<u32>(DrainSite::Count)};
        std::atomic<u64> target{};
        std::atomic<u64> begin_ns{};
        std::atomic<u64> seq{};
        std::atomic<bool> on_engine{};
    };
    [[nodiscard]] const FrontWait& FrontWaitNow() const {
        return front_wait;
    }
    /// One follower: `played` grows to `position` (never shrinks), and the front is woken. E
    /// raises it with what it ran clamped to R's oldest unplayed packet; R raises it as it plays
    /// (OneFollower::EngineConsumed).
    void RaisePlayed(u64 position) {
        u64 current = played.load(std::memory_order_seq_cst);
        while (current < position &&
               !played.compare_exchange_weak(current, position, std::memory_order_seq_cst)) {
        }
        NotifyFront();
    }
    /// Back: the ring position of the entry being run.
    [[nodiscard]] u64 BackEntryPosition() const {
        // FRITCHII_ESPLIT_DEPTH > 0: on E2, the entry of the job it runs (E1 is ahead).
        if (ESplit::g_async && ESplit::t_role == 2) [[unlikely]] {
            return ESplit::t_e2_position;
        }
        return back_entry_position;
    }
    /// Called on the back before it sleeps waiting for work (the rasterizer submits deferred work).
    void SetIdleHook(void (*hook)(void*), void* argument) {
        idle_hook = hook;
        idle_argument = argument;
    }
    /// Experiment(3): asked on the back whenever it has run everything published, whether a
    /// ReleaseFences there would only invalidate the GPU cache (RasterizerInterface::FencesQuiet).
    using FenceQuietCheck = bool (*)(void* argument, bool strict, u64* epoch);
    void SetFenceQuietCheck(FenceQuietCheck check, void* argument) {
        fence_quiet_check = check;
        fence_quiet_argument = argument;
    }
    enum class FenceQuiet : u32 {
        Quiet,     ///< The back found nothing to release with everything sent executed
        Behind,    ///< Work sent is still queued (or unpublished)
        Busy,      ///< The back found fence work with everything sent executed
        Unchecked, ///< The back has not checked at this position (or its idle hook ran since)
    };
    /// Front: what the back's last check found, if it was made with everything sent executed.
    /// `epoch`: the fence work epoch the back saw (compare with FenceWorkEpoch()).
    [[nodiscard]] FenceQuiet FencesQuietNow(u64& epoch) const {
        if (PlayedNow(std::memory_order_acquire) != write_position) {
            return FenceQuiet::Behind;
        }
        if (fence_quiet_position.load(std::memory_order_acquire) == write_position) {
            epoch = fence_quiet_epoch.load(std::memory_order_relaxed);
            return FenceQuiet::Quiet;
        }
        return fence_busy_position.load(std::memory_order_relaxed) == write_position
                   ? FenceQuiet::Busy
                   : FenceQuiet::Unchecked;
    }

    /// Any thread: whether the back thread runs (false before it starts and once it stopped).
    [[nodiscard]] bool BackRunning() const {
        return back_running.load(std::memory_order_acquire);
    }
    /// Any thread: has the back call the idle hook soon, between entries or at once if it sleeps.
    void RequestIdleHook() {
        hook_requested.store(true, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    /// `effect`: what the entry may write (SplitWriteTracker; unknown unless the caller knows).
    using Effect = SplitWriteTracker::Effect;
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment, Effect effect = Effect::Unknown) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words, effect);
        if (engine->memory_write_mask[method]) {
            NoteMethodWrite(engine, method);
        }
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment,
                     Effect effect = Effect::Unknown);
    /// EXP124: a macro's writes of one method in a row, which went as one Direct MacroWrite
    /// each, as Multi entries marked Direct: the back calls the engine's CallMacroWords (each
    /// word called as its MacroWrite was, unless the engine knows taking them at once is the
    /// same), the memory stage follows each entry as it follows the words (an executable method
    /// of another engine: one hard entry).
    void MacroWords(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                    u32 amount, Effect effect = Effect::Unknown);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument,
                    Effect effect = Effect::Unknown) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1, effect);
        if (engine->memory_write_mask[method]) {
            NoteMethodWrite(engine, method);
        }
    }
    /// FRITCHII_CB_MERGE: 0 off, 1 dry count, 2 verify (joined, with the front's address before
    /// each joined run, compared on E), 3 on (joined).
    [[nodiscard]] static int CbMergeMode() {
        return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_cb_merge,
                                                "FRITCHII_CB_MERGE");
    }
    /// FRITCHII_CB_MERGE >= 2 (front, Maxwell3D::FrontWriteSend): a constant buffer size,
    /// address or offset register write that would have ended the open CbWords block (as a
    /// PairWrite) goes into the block instead, as a Sink-tagged Method word the back applies as
    /// that Pairs entry would (ConsumeSink, ApplyPairs). False (nothing sent): the caller sends
    /// it as before. Only while the block is the ring's last entry, not with the memory stage's
    /// ring or the E split (their readers take every block word as a constant buffer word).
    bool CbJoinPair(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, int mode);
    /// FRITCHII_CB_WORDS_BLOCK (Maxwell3D::FrontWrite): a 3D constant buffer data word, as the
    /// Method word Method (`direct` false) or MacroWrite (`direct` true) would send it, appended
    /// to the open CbWords block or opening one. The block is open only until anything else is
    /// reserved, the ring is published or the sent position is read (Reserve, Publish,
    /// SentPosition), and not past held-back work (pending_flush), so its words sit in the ring
    /// in the order and between the same entries as their own entries would, and a published
    /// block never changes (as a Pairs block). Each word is committed and noted at its own
    /// position (Commit, NoteMethodWrite), as its entry was; the back stores `consumed` only at
    /// entry ends, so a position inside the block counts as run exactly when the whole block ran.
    /// `verify_address` (FRITCHII_CB_WORDS_BLOCK=2): the front's address of the word, kept for a
    /// new block's first word, which the back compares with its registers.
    /// `merge_address` (FRITCHII_CB_MERGE=2): the front's address of the word, marked before it
    /// when it starts a joined run (cb_merge_mark_next); 0 otherwise.
    void CbWord(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                bool direct, Effect effect, const GPUVAddr* verify_address,
                GPUVAddr merge_address = 0) {
        const u64 word = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                         (direct ? Direct : 0) | (u64{method & 0xffff} << 8) |
                         (u64{argument} << 32);
        const bool verify = verify_address != nullptr;
        const size_t index = write_position & RingMask;
        if (cb_header != nullptr) {
            if (pending_flush == nullptr && engine == front_engine && index != 0 &&
                cb_count < MaxCbWords && verify == cb_verify &&
                write_position + 1 - consumed_cache <= RingWords) [[likely]] {
                if (write_position == cb_end) [[likely]] {
                    if (cb_merge_mark_next) [[unlikely]] {
                        // FRITCHII_CB_MERGE=2: the marker, the front's address, then the word,
                        // committed together (a publish between would strand the marker).
                        cb_merge_mark_next = false;
                        if (merge_address != 0 && index + 2 < RingWords &&
                            cb_count + 3 <= MaxCbWords &&
                            write_position + 3 - consumed_cache <= RingWords) {
                            ring[index] = static_cast<u64>(Kind::Method) | Sink |
                                          (u64{CbMarkMethod} << 8);
                            ring[index + 1] = merge_address;
                            ring[index + 2] = word;
                            cb_count += 3;
                            *cb_header = (*cb_header & 0xffff'ffffULL) | (u64{cb_count} << 32);
                            ++cb_stats.words;
                            ++cb_join.marks;
                            Commit(3, effect);
                            cb_end = write_position;
                            if (engine->memory_write_mask[method]) {
                                NoteMethodWrite(engine, method);
                            }
                            return;
                        }
                        ++cb_join.marks_skipped;
                    }
                    ring[index] = word;
                    ++cb_count;
                    *cb_header = (*cb_header & 0xffff'ffffULL) | (u64{cb_count} << 32);
                    ++cb_stats.words;
                    Commit(1, effect); // A publish here ends the block (Publish)
                    cb_end = write_position;
                    if (engine->memory_write_mask[method]) {
                        NoteMethodWrite(engine, method);
                    }
                    return;
                }
                // Something was committed after the block without ending it (must be 0).
                ++cb_stats.broken;
            } else if (cb_count >= MaxCbWords) {
                ++cb_stats.full;
            }
        }
        // FRITCHII_CB_MERGE=1/2 (dry count): what split this block from the previous one.
        cb_merge_mark_next = false;
        if (CbMergeOn()) [[unlikely]] {
            NoteCbMergeGap(engine);
        }
        SelectEngine(engine);
        const size_t head = verify ? 2 : 1;
        u64* const entry = Reserve(head + 1);
        entry[0] = static_cast<u64>(Kind::CbWords) | (verify ? CbVerify : 0) | (u64{1} << 32);
        if (verify) {
            entry[1] = *verify_address;
        }
        entry[head] = word;
        // Open before the commit, which may publish it (and so end it).
        cb_header = entry;
        cb_count = 1;
        cb_verify = verify;
        ++cb_stats.blocks;
        ++cb_stats.words;
        Commit(head + 1, effect);
        cb_end = write_position;
        if (engine->memory_write_mask[method]) {
            NoteMethodWrite(engine, method);
        }
    }
    /// EDEN_NX2_MACRO_SINK: a write that only stores a register's value, to an engine whose
    /// execution_mask has the register (native compute): the back puts it in the engine's method
    /// sink tagged EngineInterface::SplitPlainTag, applied before the engine's next other write.
    void SinkMethod(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Sink | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1, Effect::None);
    }
    /// EDEN_NX2_RING_PAIRS: a write that only stores a register's value (no execution_mask side
    /// effect; native compute: native_plain_methods), appended to the open Pairs block. A block
    /// is open only until anything else is reserved or the ring is published (Reserve, Publish),
    /// so published entries never change and a block never crosses the ring's end.
    void PairWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        SelectEngine(engine);
        const u64 pair = u64{method & 0xffff} | (u64{argument} << 32);
        const size_t index = write_position & RingMask;
        if (pairs_header != nullptr && index != 0 && pairs_count < MaxPairs &&
            write_position + 1 - consumed_cache <= RingWords) [[likely]] {
            ring[index] = pair;
            ++pairs_count;
            *pairs_header = static_cast<u64>(Kind::Pairs) | (u64{pairs_count} << 32);
            CountBatched(Batched::Pairs);
            Commit(1, Effect::None);
            return;
        }
        u64* const entry = Reserve(2);
        entry[0] = static_cast<u64>(Kind::Pairs) | (u64{1} << 32);
        entry[1] = pair;
        pairs_header = entry;
        pairs_count = 1;
        CountBatched(Batched::Pairs);
        CountBatched(Batched::PairBlocks);
        Commit(2, Effect::None);
    }

    // EDEN_NX2_EXP22, from a ported change (EDEN_NX2_FRONT_STATE, EDEN_NX2_DRAW_BATCH).
    /// Bit 1: a 3D state register write (not executable, writes no memory), which would go as a
    /// Direct MacroWrite or a PairWrite, appended to the open StateDelta block instead. The back
    /// applies the block in one loop, write by write as CallMethod and ApplyPairs would
    /// (Maxwell3D::ApplyFrontDelta). Open as a Pairs block is, and each write notes the effect
    /// its old path noted at its position.
    void DeltaWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument,
                    Effect effect) {
        SelectEngine(engine);
        const u64 word = u64{method & 0xffff} | (u64{argument} << 32);
        const size_t index = write_position & RingMask;
        if (delta_header != nullptr && index != 0 && delta_count < MaxPairs &&
            write_position + 1 - consumed_cache <= RingWords) [[likely]] {
            ring[index] = word;
            ++delta_count;
            *delta_header = static_cast<u64>(Kind::StateDelta) | (u64{delta_count} << 32);
            CountBatched(Batched::DeltaWrites);
            Commit(1, effect);
            return;
        }
        u64* const entry = Reserve(2);
        entry[0] = static_cast<u64>(Kind::StateDelta) | (u64{1} << 32);
        entry[1] = word;
        delta_header = entry;
        delta_count = 1;
        CountBatched(Batched::DeltaWrites);
        CountBatched(Batched::DeltaBlocks);
        Commit(2, effect);
    }
    /// Bit 2: `count` (method | argument << 32) writes of an instance-loop group (draw.begin
    /// Subsequent and the instance loop stores after it), run once, then `repeats - 1` more
    /// identical groups counted as instances (Maxwell3D::RunDrawBatch).
    void DrawBatch(Tegra::Engines::EngineInterface* engine, const u64* group_writes, u32 count,
                   u32 repeats, Effect effect) {
        SelectEngine(engine);
        u64* const entry = Reserve(1 + size_t{count});
        entry[0] = static_cast<u64>(Kind::DrawBatch) | (u64{count & 0xff} << 8) |
                   (u64{repeats} << 32);
        std::memcpy(entry + 1, group_writes, size_t{count} * sizeof(u64));
        CountBatched(Batched::DrawBatches);
        CountBatched(Batched::DrawRepeats, repeats - 1);
        Commit(1 + size_t{count}, effect);
    }
    /// Work the front holds back (EXP22's draw batches): `hook(argument)` sends it before any
    /// other entry is reserved, before an engine is selected, before a publish and before a
    /// drain, so it keeps its place in the method order. Another holder's work is sent first.
    using PendingFlush = void (*)(void*);
    void SetPendingFlush(PendingFlush hook, void* argument) {
        if (pending_flush != hook || pending_flush_argument != argument) {
            RunPendingFlush();
            pending_flush = hook;
            pending_flush_argument = argument;
        }
    }
    void RunPendingFlush() {
        if (pending_flush != nullptr) [[unlikely]] {
            const PendingFlush hook = pending_flush;
            pending_flush = nullptr;
            hook(pending_flush_argument);
        }
    }
    /// `argument` sends its held work itself, or goes away: forget its hook.
    void DropPendingFlush(void* argument) {
        if (pending_flush_argument == argument) {
            pending_flush = nullptr;
            pending_flush_argument = nullptr;
        }
    }
    /// Instance folding (EDEN_NX2_FOLD_INSTANCES), front: one more
    /// instance of the instance loop `context` draws. Nothing is sent: `flush(context, count)`
    /// sends the instances counted so far as one entry before the next entry, engine selection,
    /// Publish or Drain. Held as the pending flush above, so other held work (EXP22's batches,
    /// EDEN_NX2_DRAW_PACKET's packet) keeps its order against it.
    void FoldInstance(void (*flush)(void*, u32), void* context) {
        if (fold_count != 0 && fold_context != context) {
            RunPendingFlush(); // The other loop's count (FlushFoldThunk) goes first
        }
        SetPendingFlush(&FlushFoldThunk, this);
        fold_flush = flush;
        fold_context = context;
        ++fold_count;
    }
    /// Sends the instances FoldInstance counted, if any (GpuSplit clears the hook before it).
    void FlushFold() {
        if (fold_count == 0) {
            return;
        }
        const u32 count = std::exchange(fold_count, 0);
        fold_flush(fold_context, count);
    }

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    /// `flags`: Nx2FastDrawCount::Neutral() for a call that can't change what a draw resolves to
    /// (bit 7 of a Call entry, set only while EDEN_NX2_FASTDRAW_COUNT counts; Execute ignores it).
    /// Bits 32-39: the memory stage's tag (StageTagBits, memstage_shadow.h); Execute ignores them.
    template <typename Func>
    void Defer(const Func& func, Effect effect = Effect::Unknown, u64 flags = 0) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | flags | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload, effect);
    }
    void DeferFunction(std::function<void()>&& func);
    /// Runs `func(payload)` on the back after everything sent before it, with `words` payload
    /// words that `fill(u64* payload)` writes here (a Call entry with a variable payload, for
    /// EDEN_NX2_DRAW_PACKET's write lists). `fill` must not call into the split.
    /// `flags` as Defer's.
    template <typename Fill>
    void CallWithPayload(void (*func)(const u64*), size_t words, Effect effect, const Fill& fill,
                         u64 flags = 0) {
        u64* const entry = Reserve(2 + words);
        entry[0] = static_cast<u64>(Kind::Call) | flags | (u64{words} << 8);
        entry[1] = reinterpret_cast<u64>(func);
        fill(entry + 2);
        Commit(2 + words, effect);
    }
    /// Runs `func` on the back and waits for it: for work on back-owned state, such as a buffer
    /// cache download that records into the Vulkan scheduler. A drained back still submits from
    /// its idle hook, so the front never touches that state itself.
    /// `flags` as Defer's.
    template <typename Func>
    void Run(const Func& func, DrainSite site = DrainSite::Other, u64 flags = 0) {
        if (on_back_thread || !active) {
            func();
            return;
        }
        Defer(func, Effect::Unknown, flags);
        Drain(site);
    }

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();
    /// EDEN_NX2_EXP27 & 1 (the original's EDEN_NX2_PUBLISH_WAITS): before the front yields
    /// or sleeps polling suspended channels, it publishes what they sent (held-back work too),
    /// so the back doesn't idle on work the front already decoded.
    void PublishBeforeWait();
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);

    /// What sent compute or copy work that may write any memory (EXP10's log line).
    enum class WriterKind : u32 {
        Upload,  ///< Compute inline uploads the front could not bound
        Release, ///< Compute report semaphores the front could not translate
        Compute, ///< Other compute methods (unknown methods)
        Copy,    ///< Copies the front could not bound
        Other,   ///< Writers no engine front classified
        Count,
    };
    /// Compute or copy work was just sent: what can write GPU-written command lists.
    /// `classified`: the engine's front noted what that work writes (NoteWriterRange,
    /// NoteUnboundedWriter, NoteDispatchWriter); otherwise it may write anything.
    void NoteWriter(bool classified = false) {
        FrontSplit::NoteWriterSent(); // FRITCHII_FRONT_SPLIT_COUNT: one plain load off
        last_writer_position = write_position;
        if (!classified) {
            NoteUnboundedWriter(WriterKind::Other);
        }
    }
    /// Compute or copy work just sent may write any memory.
    void NoteUnboundedWriter(WriterKind kind) {
        last_unbounded_writer_position = write_position;
        ++unbounded_writers[static_cast<size_t>(kind)];
    }
    /// Compute or copy work just sent writes only [address, address + size) of `memory` (a copy's
    /// destination, a release, an inline upload); counts as unbounded when the front can't
    /// translate it as the back will.
    void NoteWriterRange(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                         WriterKind kind);
    /// How the front noted a compute report semaphore (NoteReleaseWriter, EXP10's log line).
    enum class ReleaseNote : u32 {
        Ranged,       ///< Its bytes, as one device range
        Unmapped,     ///< Translated when the back runs it, to nothing: writes nothing
        NoWrite,      ///< An operation the back ignores (KeplerCompute::FrontNoteWriter)
        PendingMap,   ///< Unbounded: a queued function may still remap it
        Sparse,       ///< Unbounded: in a lazily backed sparse span
        LateUnmapped, ///< Unbounded: unmapped now, translated only when its fence signals
        Split,        ///< Unbounded: not one device range
        Count,
    };
    /// A compute report semaphore just sent writes [address, address + size). `at_run`: the back
    /// translates it when it runs the method and writes nothing when that fails (the query
    /// cache's payload reports), so an address without a mapping (Ocarina of Time releases to GPU
    /// address 0) needs no range; otherwise as NoteWriterRange.
    void NoteReleaseWriter(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                           bool at_run);
    void CountRelease(ReleaseNote note) {
        ++release_notes[static_cast<size_t>(note)];
    }
    [[nodiscard]] const std::array<u64, static_cast<size_t>(ReleaseNote::Count)>&
    ReleaseNotes() const {
        return release_notes;
    }
    /// The noted ranges so far: a check made now covers those below it (WritersMayWriteSince).
    [[nodiscard]] u64 WriterRangeCount() const {
        writer_range_sealed = writer_range_count;
        return writer_range_count;
    }
    /// EXP10 +8: the writes found since a macro DMA read snapshot that overlap the range it
    /// checks (a 4 KiB page), as offsets into that range, so a read of other bytes of it can
    /// still skip the drain (Maxwell3D::FrontDmaRead). Past Capacity the two closest listed
    /// ranges become their union: never smaller than what was written, only coarser.
    struct ListedWrites {
        static constexpr u32 Capacity = 8;
        std::array<std::pair<u32, u32>, Capacity> ranges{};
        u32 count{};
        /// Unions made to stay within Capacity (for the log line).
        u32 merges{};
        /// Lists [begin, end) of the range at `base` (clipped to `size` bytes) unless a listed
        /// one covers it.
        void Add(u64 base, u64 size, u64 begin, u64 end) {
            const u32 first = static_cast<u32>((std::max)(begin, base) - base);
            const u32 last = static_cast<u32>((std::min)(end, base + size) - base);
            for (u32 i = 0; i < count; ++i) {
                if (ranges[i].first <= first && last <= ranges[i].second) {
                    return;
                }
            }
            if (count < Capacity) {
                ranges[count++] = {first, last};
                return;
            }
            // Sorted by start, the neighbours with the smallest gap (or the most overlap) merge.
            std::array<std::pair<u32, u32>, Capacity + 1> all{};
            std::copy(ranges.begin(), ranges.end(), all.begin());
            all[Capacity] = {first, last};
            std::sort(all.begin(), all.end());
            size_t best = 0;
            s64 best_gap = (std::numeric_limits<s64>::max)();
            for (size_t i = 0; i + 1 < all.size(); ++i) {
                const s64 gap = static_cast<s64>(all[i + 1].first) - all[i].second;
                if (gap < best_gap) {
                    best_gap = gap;
                    best = i;
                }
            }
            all[best].second = (std::max)(all[best].second, all[best + 1].second);
            for (size_t i = 0, out = 0; i < all.size(); ++i) {
                if (i != best + 1) {
                    ranges[out++] = all[i];
                }
            }
            ++merges;
        }
        /// Whether a listed write overlaps device bytes [begin, end) (the range at `base`).
        [[nodiscard]] bool Overlaps(u64 base, u64 begin, u64 end) const {
            for (u32 i = 0; i < count; ++i) {
                if (base + ranges[i].first < end && begin < base + ranges[i].second) {
                    return true;
                }
            }
            return false;
        }
    };
    /// Why WritersMayWriteSince answered true (EXP10's log line).
    enum class WriterCheck : u32 {
        Miss,      ///< Answered false
        Unbounded, ///< An unbounded writer since
        Overlap,   ///< A noted range overlaps (without `listed`)
        ScanLimit, ///< More than MaxWriterScan ranges noted since `checked`
        Forgotten, ///< Ranges since `checked` left the ring before they were compared
        Count,
    };
    /// Whether compute or copy work sent after position `since` may write device bytes
    /// [begin, begin + size): an unbounded writer, or a noted range that overlaps them.
    /// `checked` (optional): the noted ranges below it were found to miss these bytes before;
    /// set to WriterRangeCount() when nothing does. `listed` (optional, EXP10 +8): overlapping
    /// ranges are added to it instead of answering true; it and `checked` change only when the
    /// answer is false (the ranges below `checked` then miss or are listed). `why` (optional):
    /// set to the reason.
    [[nodiscard]] bool WritersMayWriteSince(u64 since, u64 begin, u64 size,
                                            u64* checked = nullptr,
                                            ListedWrites* listed = nullptr,
                                            WriterCheck* why = nullptr) const {
        u64 compared = 0;
        const auto answer = [why, &compared](WriterCheck reason) {
            if (why != nullptr) {
                *why = reason;
            }
            // Ranges compared by these scans: the front time a larger MaxWriterScan costs.
            g_writer_ranges_compared.fetch_add(compared, std::memory_order_relaxed);
            return reason != WriterCheck::Miss;
        };
        if (last_unbounded_writer_position > since) {
            return answer(WriterCheck::Unbounded);
        }
        const u64 known = checked != nullptr ? *checked : 0;
        ListedWrites found = listed != nullptr ? *listed : ListedWrites{};
        // EXP122: a watched page answers from its own list (gpu_split.cpp).
        if (watched_pages != 0) {
            const WatchAnswer watch = WatchedWritesSince(
                since, begin, size, known, listed != nullptr ? &found : nullptr);
            if (watch == WatchAnswer::Write) {
                return answer(WriterCheck::Overlap);
            }
            if (watch == WatchAnswer::NoWrite) {
                if (listed != nullptr) {
                    *listed = found;
                }
                if (checked != nullptr) {
                    *checked = WriterRangeCount();
                }
                return answer(WriterCheck::Miss);
            }
        }
        // At most MaxWriterScan ranges are compared (a drained read takes a new snapshot), and
        // more than are remembered count as overlapping.
        const u64 end = begin + size;
        const u64 oldest =
            writer_range_count - (std::min<u64>)(writer_range_count, writer_ranges.size());
        u64 index = writer_range_count;
        bool reached = false;
        for (u64 scanned = 0; index > (std::max)(known, oldest); --index, ++scanned) {
            if (scanned == MaxWriterScan) {
                return answer(WriterCheck::ScanLimit);
            }
            const WriterRange& range = writer_ranges[(index - 1) % writer_ranges.size()];
            ++compared;
            if (range.position <= since) {
                reached = true;
                break;
            }
            if (range.begin < end && begin < range.end) {
                if (listed == nullptr) {
                    return answer(WriterCheck::Overlap);
                }
                found.Add(begin, size, range.begin, range.end);
            }
        }
        if (!reached && oldest > known) {
            return answer(WriterCheck::Forgotten);
        }
        if (listed != nullptr) {
            *listed = found;
        }
        if (checked != nullptr) {
            *checked = WriterRangeCount();
        }
        return answer(WriterCheck::Miss);
    }
    /// EXP122 page watch (gpu_split.cpp): 0 off, 1 on, 2 on with a full-scan verify. Front only.
    void SetPageWatch(int mode) {
        if (mode != page_watch_mode) {
            page_watch_mode = mode;
            if (mode == 0) {
                ClearPageWatch();
            }
        }
    }
    /// EXP122: the front just took a snapshot of device page `page` (Maxwell3D::FrontDmaRead).
    void WatchPage(u64 page);
    struct PageWatchStats {
        u64 answered;       ///< WritersMayWriteSince answers from a watched page
        u64 answered_write; ///< of them "may write"
        u64 watched;        ///< Pages watched now
        u64 overflows;      ///< Byte lists that overflowed (unioned)
        u64 verified;       ///< EXP122=2: "no write" answers checked by a full scan
        u64 mismatches;     ///< of them the full scan found an overlap (must stay 0)
        u64 unwatched_absent; ///< Checks the watch left to the ring: page not in the table
        u64 unwatched_newer;  ///< page watched since after the check's snapshot
        u64 unwatched_spans;  ///< check over two pages
        u64 replaced;         ///< Watched pages pushed out by a new one
    };
    [[nodiscard]] PageWatchStats GetPageWatchStats() const;
    /// The ranges noted so far, without sealing the last one (EXP10's log line).
    [[nodiscard]] u64 NotedWriterRanges() const {
        return writer_range_count;
    }
    [[nodiscard]] u64 LastUnboundedWriterPosition() const {
        return last_unbounded_writer_position;
    }
    [[nodiscard]] const std::array<u64, static_cast<size_t>(WriterKind::Count)>&
    UnboundedWriters() const {
        return unbounded_writers;
    }

    // EXP10: what native compute dispatches write. The front can't tell (their storage buffers
    // and images come from constant buffers and descriptor pools queued work may still write),
    // so the back records, for each dispatch it runs, the device ranges it marks GPU-written in
    // the buffer cache (BufferCache::MarkWrittenBuffer: the storage and texture buffers' write
    // marks, all a flush or fence download brings to guest memory), its written images' guest
    // ranges, its indirect arguments and its QMD releases. A front read checks those once the
    // back ran every dispatch sent since its snapshot, waiting for that when asked (only that
    // far, not for what was sent after).

    /// Native compute work that may dispatch was just sent (a QMD launch or an inline QMD).
    void NoteDispatchWriter();
    /// Why a dispatch the back ran counts as writing anything.
    enum class DispatchUnbounded : u32 {
        Unconfirmed, ///< The rasterizer did not report its writes (another path or backend)
        Replaced,    ///< Replaced by other work (a network bypass, an upscaler)
        Image,       ///< A written image without a guest range
        Count,
    };
    enum class DispatchCheck : u32 {
        Miss,      ///< Every dispatch sent since ran, and none writes the bytes
        Queued,    ///< One is still queued (without the wait)
        Overlap,   ///< One writes the bytes
        Unbounded, ///< One may write anything (DispatchUnbounded, or sent with EXP10 off)
        ScanLimit, ///< Too many writes recorded since to compare
        Count,
    };
    /// Whether the native compute dispatches sent after position `since` may write device bytes
    /// [begin, begin + size). `checked`: the recorded writes below it were found to miss these
    /// bytes before (DispatchWriteCount() at the snapshot); updated on a miss. With `wait`, waits
    /// for the back to run the newest of those dispatches first (`waited` then set). `listed`
    /// (optional, EXP10 +8): as WritersMayWriteSince's, bounded overlapping writes are listed
    /// instead of answering Overlap; changed only on a Miss.
    DispatchCheck DispatchesMayWriteSince(u64 since, u64 begin, u64 size, bool wait, u64& checked,
                                          bool& waited, ListedWrites* listed = nullptr);
    /// FRITCHII_FRONT_RAN_READS, after DispatchesMayWriteSince answered ScanLimit (so every
    /// dispatch sent after `since` ran and recorded its writes): whether no dispatch write at or
    /// after `since` was recorded over [begin, begin + size), from the dispatch stamps (the
    /// newest recorded position per 64 KB block, in a hashed table whose collisions only make a
    /// stamp newer; writes over 4 GB or unbounded ones stamp everything). The answer of the scan
    /// without its limit, or stricter. `count` gets DispatchWriteCount() loaded before the
    /// stamps: every record below it is covered by the answer. `wide`: refused by a write that
    /// stamped everything.
    [[nodiscard]] bool DispatchStampsMissSince(u64 since, u64 begin, u64 size, u64& count,
                                               bool& wide) const {
        count = dispatch_write_count.load(std::memory_order_acquire);
        if (dispatch_wide_stamp.load(std::memory_order_acquire) >= since) {
            wide = true;
            return false;
        }
        wide = false;
        if (size == 0) {
            return true;
        }
        const u64 first = begin >> DispatchStampBits;
        const u64 last = (begin + size - 1) >> DispatchStampBits;
        for (u64 block = first; block <= last; ++block) {
            if (dispatch_stamps[block % DispatchStampSlots].load(std::memory_order_acquire) >=
                since) {
                return false;
            }
        }
        return true;
    }
    /// The writes the back recorded so far.
    [[nodiscard]] u64 DispatchWriteCount() const {
        return dispatch_write_count.load(std::memory_order_acquire);
    }
    /// For EXP10's log line: native compute work sent that may dispatch, the dispatches the back
    /// ran whose writes it recorded, and why the others count as writing anything.
    struct DispatchStats {
        u64 sent;
        u64 bounded;
        std::array<u64, static_cast<size_t>(DispatchUnbounded::Count)> unbounded;
    };
    [[nodiscard]] DispatchStats GetDispatchStats() const;
    /// FRITCHII_DISPATCH_WALK_SUMMARY (DispatchesMayWriteSince), front: the dispatch records the
    /// walks loaded, the walks the dispatch stamps answered instead (a miss), the stamp checks
    /// refused by a stamp on the page's block or by one that stamped everything, and with 2 the
    /// walks that found a write the stamps said was not there (must be 0). Counted with the
    /// switch off too, except the last three.
    struct DispatchWalkStats {
        u64 walked{};
        u64 summary_misses{};
        u64 refused_block{};
        u64 refused_wide{};
        u64 verify_disagreed{};
    };
    [[nodiscard]] const DispatchWalkStats& GetDispatchWalkStats() const {
        return dispatch_walk;
    }

    /// Back: whether the dispatches being run record their writes (BeginDispatchRecord).
    [[nodiscard]] static bool RecordingDispatch() {
        return recording_dispatch;
    }
    /// Back, around a QMD action that may dispatch and release (KeplerCompute).
    void BeginDispatchRecord() {
        recording_dispatch = on_back_thread && back_dispatch_records;
    }
    void EndDispatchRecord() {
        recording_dispatch = false;
    }
    /// GPUMem, around a fill it runs for the back (memory_stage_link.cpp).
    static void AdoptRecordingDispatch(bool recording) {
        recording_dispatch = recording;
    }
    /// Back, while recording: a dispatch starts; the rasterizer recorded all it writes; it ended
    /// (`ran`: it reached the rasterizer, `replaced`: other work ran instead).
    void StartDispatch() {
        dispatch_confirmed = false;
        dispatch_unbounded = false;
    }
    void ConfirmDispatch() {
        dispatch_confirmed = true;
    }
    void FinishDispatch(bool ran, bool replaced);
    /// FRITCHII_DISPATCH_WAIT_CENSUS: what a recorded dispatch write is (storage and texel
    /// buffers by their write marks, written images, indirect arguments, QMD releases, or
    /// anything), so the census can say which waits a front-side bound of the dispatches'
    /// buffer writes would answer.
    enum class DispatchWriteKind : u8 {
        Buffer,
        Image,
        Indirect,
        Release,
        Unbounded,
        Count,
    };
    /// Back, while recording: the dispatch writes device bytes [begin, begin + size).
    void RecordDispatchWrite(u64 begin, u64 size,
                             DispatchWriteKind kind = DispatchWriteKind::Buffer) {
        if (size != 0) {
            AppendDispatchWrite(begin, begin + size, kind);
        }
    }
    /// FRITCHII_DISPATCH_WAIT_CENSUS (live, default 0; 1 counts): every macro read's wait for
    /// the dispatches (DrainSite::DispatchWait), split by what the dispatches it waited for (the
    /// ones still queued when it began) turned out to write. Front only.
    struct DispatchWaitCensus {
        enum Class : u32 {
            BufferOnly,   ///< Only buffer write marks (a front bound of SSBO writes answers it)
            WithRelease,  ///< Buffers and QMD releases (the front predicts those already)
            WithIndirect, ///< Indirect arguments too (list dispatches, GPU-sized grids)
            WithImage,    ///< A written image (a bound needs the image descriptors as well)
            Unknown,      ///< Anything (unbounded), or the records left the ring meanwhile
            Nothing,      ///< No record at all (the dispatches wrote nothing the back marks)
            Count,
        };
        std::array<u64, Class::Count> waits{};
        std::array<u64, Class::Count> wait_ns{};
        u64 records{};      ///< Records of the waited-for dispatches, all waits
        u64 overlapping{};  ///< Waits where one of them wrote the page (must stay near 0)
    };
    [[nodiscard]] const DispatchWaitCensus& GetDispatchWaitCensus() const {
        return dispatch_census;
    }
    /// Non-blocking: whether a wait for `position` at `site` would return at once (WaitUntil's
    /// test, without publishing or waiting). Front only.
    [[nodiscard]] bool ReachedSite(u64 position, DrainSite site) const {
        const u64 target = (std::min)(position, write_position);
        return PlayedNow(std::memory_order_acquire) >= target ||
               Reached(target, UsesEngine(site), std::memory_order_acquire);
    }
    /// Back, while recording: the dispatch may write anything.
    void RecordDispatchUnbounded(DispatchUnbounded reason);
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();
    /// Other work that writes guest memory on the back was just sent: inline uploads, reports,
    /// puller methods (semaphores).
    void NoteMemoryWriter() {
        last_memory_writer_position = write_position;
    }
    /// Waits until the back executed every NoteWriter and NoteMemoryWriter work sent so far.
    bool WaitForMemoryWriters();
    /// The delay tests (Coz-style causal profiling, fable_front_drains sections 3 and 6): where a
    /// busy spin is added. The drains: FRITCHII_DELAY_MACRO_DRAIN_US and
    /// FRITCHII_DELAY_POLL_DRAIN_US. Work, to tell which thread paces which pass:
    /// FRITCHII_DELAY_E2C_US (front), FRITCHII_DELAY_DISPATCH_US and FRITCHII_DELAY_DRAW_US
    /// (back).
    enum class DelayTest : u32 {
        MacroDrain, ///< A macro DMA read that waits for the back (FrontDmaRead)
        PollDrain,  ///< A suspended acquire's poll that drains (Puller::CanResumeAcquire)
        E2c,        ///< Front: every call of the 0xe2c occlusion macro ("W")
        Dispatch,   ///< Back: every compute dispatch (RasterizerVulkan::DispatchCompute, "K")
        Draw,       ///< Back: every 3D draw call (RasterizerVulkan::Draw/DrawIndirect, "D")
        Back,       ///< Back: FRITCHII_DELAY_BACK_US at FRITCHII_DELAY_BACK_KIND's site (DelayBack)
        Front, ///< Front: FRITCHII_DELAY_FRONT_US at FRITCHII_DELAY_FRONT_KIND's site (DelayFront)
        Worker, ///< The Vulkan worker: FRITCHII_DELAY_WORKER_US per chunk (Scheduler::WorkerThread)
        Count,
    };
    /// FRITCHII_DELAY_FRONT_US / _KIND (live, default 0; fable_stage_levelpoints 18.3): the sites.
    enum class FrontDelayKind : u32 {
        AcquireMet = 1,   ///< "A": a suspended acquire resumes (Puller::ResumeAcquire)
        DispatchWait = 2, ///< "Q": a macro read's wait behind a dispatch (DispatchesMayWriteSince)
        Macros = 3,       ///< "C": once every 64 3D macro calls (Maxwell3D::FrontRunMacro)
        Entries = 4,      ///< "P": once every 16 list entries, before their checks (DmaPusher::Step)
    };
    /// At a FRITCHII_DELAY_FRONT_KIND site: one relaxed load and a branch while
    /// FRITCHII_DELAY_FRONT_US is 0. Kinds 1 and 2 are called on both sides of their wait
    /// (`before`: right before it begins, else right after it ended) and spin on the side
    /// FRITCHII_DELAY_BEFORE picks (0: after, the test; 1: before, the control).
    static void DelayFrontIf(FrontDelayKind kind, bool before = false) {
        if (VideoCore::Nx2Debug::g_delay_front_us.load(std::memory_order_relaxed) != 0)
            [[unlikely]] {
            DelayFront(kind, before);
        }
    }
    static void DelayFront(FrontDelayKind kind, bool before);
    /// FRITCHII_DELAY_WORKER_US (live, default 0; fable_stage_levelpoints 19.3/19.4): the Vulkan
    /// worker spins n us after each chunk it ran, inside its execution lock. Off: one relaxed
    /// load and a branch a chunk.
    static void DelayWorkerIf() {
        if (VideoCore::Nx2Debug::g_delay_worker_us.load(std::memory_order_relaxed) != 0)
            [[unlikely]] {
            if (const int us = DelayTestUs(DelayTest::Worker); us > 0) {
                SpinDelay(DelayTest::Worker, static_cast<u32>(us));
            }
        }
    }
    /// The switch of a delay test (its microseconds; 0 or less: off).
    static int DelayTestUs(DelayTest test);
    /// FRITCHII_DELAY_BACK_US / _KIND (live, default 0; fable_stage_levelpoints 18.3): the sites.
    enum class BackDelayKind : u32 {
        Methods = 1,    ///< "M": once every 64 method-zone ring entries (BackLoop, after Execute)
        StageBinds = 2, ///< "S": every 4th BindHostStageBuffers (GraphicsPipeline::ConfigureImpl)
        Fences = 3,     ///< "F": every made fence, after the skip test (FenceManager::SignalFence)
    };
    /// At a FRITCHII_DELAY_BACK_KIND site: one relaxed load and a branch while
    /// FRITCHII_DELAY_BACK_US is 0 (after its first read). `amount`: the site's items (ring
    /// entries for Methods).
    static void DelayBackIf(BackDelayKind kind, u32 amount = 1) {
        if (VideoCore::Nx2Debug::g_delay_back_us.load(std::memory_order_relaxed) != 0)
            [[unlikely]] {
            DelayBack(kind, amount);
        }
    }
    static void DelayBack(BackDelayKind kind, u32 amount);
    /// FRITCHII_DELAY_E2C_US, FRITCHII_DELAY_DISPATCH_US, FRITCHII_DELAY_DRAW_US (live, default
    /// 0): spins the switch's microseconds for `test` (E2c, Dispatch or Draw) when it is set.
    /// Dispatch and Draw spin only on the back thread (never on GPUMem, nor without the split).
    static void DelayWork(DelayTest test);
    /// DelayWork behind an inline test of `test`'s own switch, `value` (Nx2Debug::g_delay_e2c_us,
    /// g_delay_dispatch_us or g_delay_draw_us; -1 until DelayWork's first read stores it): with
    /// the switch at 0 a draw or dispatch pays one relaxed load and a branch, no call.
    static void DelayWorkIf(const std::atomic<int>& value, DelayTest test) {
        if (value.load(std::memory_order_relaxed) != 0) [[unlikely]] {
            DelayWork(test);
        }
    }
    /// Front, right before a wait of `test`'s kind (a drain): 0 unless its switch is set and the
    /// back has not run everything sent (the drain will wait). With FRITCHII_DELAY_BEFORE=1 (the
    /// control run) it spins here and returns 0; otherwise it returns the microseconds the caller
    /// spins once the drain returned (DelayAfterWait). The spin after the drain is the test: the
    /// back has nothing queued then, so the spin is added front time. A spin before it overlaps
    /// the back's backlog, which keeps shrinking meanwhile, so it costs little: the control.
    [[nodiscard]] u32 DelayBeforeWait(DelayTest test);
    static void DelayAfterWait(DelayTest test, u32 us) {
        if (us != 0) [[unlikely]] {
            SpinDelay(test, us);
        }
    }
    /// Busy-spins `us` microseconds on this thread (steady_clock with a pause, not a sleep) and
    /// counts it for the "Front waits for the back by site" line (atomic counts: the back's
    /// spins too). Publishes nothing.
    static void SpinDelay(DelayTest test, u32 us);
    /// memory-writing work whose range the front does not note was just sent (KeplerMemory
    /// and Fermi2D methods; DmaPusher is the only caller). Ends every macro DMA read snapshot.
    void NoteUnrangedWriter() {
        last_memory_writer_position = write_position;
        last_unranged_position = write_position;
        unranged_without_exp120 = write_position;
    }
    /// EXP120: a KeplerMemory or Fermi2D method that EXP120 = 0 notes as an unranged writer was
    /// just sent, and EXP120 noted its bytes instead (NoteMemoryWrite) or found it writes no
    /// memory. FrontDmaRead tells the reads this spares a drain apart (their own verify count).
    void NoteUnrangedWriterSpared() {
        unranged_without_exp120 = write_position;
    }
    /// Where LastUnrangedPosition would be with EXP120 = 0.
    [[nodiscard]] u64 LastUnrangedPositionWithoutExp120() const {
        return unranged_without_exp120;
    }
    /// What the KeplerMemory and Fermi2D methods sent were noted as (DmaPusher, EXP120), for the
    /// "DMA snapshot" log. Front only.
    enum class EngineWriteKind : u32 {
        I2mData,      ///< KeplerMemory data: the back writes the upload
        I2mRegister,  ///< KeplerMemory, any other method (exec included): registers only
        BlitLaunch,   ///< Fermi2D's blit launch (pixels_from_memory.src_y0 + 1)
        BlitRegister, ///< Fermi2D, any other method: registers only
        Count,
    };
    enum class EngineWriteNote : u32 {
        Unranged, ///< NoteUnrangedWriter
        Ranged,   ///< EXP120: NoteMemoryWrite of the bytes it may write
        Nothing,  ///< EXP120: writes no memory, not noted
        Count,
    };
    /// [call: 0 CallMethod, 1 CallMultiMethod][kind][note]
    using EngineWriteNotes =
        std::array<std::array<std::array<u64, static_cast<size_t>(EngineWriteNote::Count)>,
                              static_cast<size_t>(EngineWriteKind::Count)>,
                   2>;
    void CountEngineWrite(bool multi, EngineWriteKind kind, EngineWriteNote note) {
        ++engine_write_notes[multi ? 1 : 0][static_cast<size_t>(kind)][static_cast<size_t>(note)];
    }
    [[nodiscard]] const EngineWriteNotes& GetEngineWriteNotes() const {
        return engine_write_notes;
    }
    /// the ring position of everything sent so far. It ends the open CbWords block (CbWord): a
    /// caller that keeps the position for "before what is sent after this" (run-ahead's apply
    /// stamps) then sees later words in a later entry, as it saw their own entries.
    [[nodiscard]] u64 SentPosition() const {
        cb_header = nullptr;
        return write_position;
    }
    /// The ring position of everything sent so far, read without ending the open CbWords block
    /// (counters only: FRITCHII_ONEF_DEAD_STORES). Work held back as the pending flush is not in
    /// it yet (PendingFlushHeld).
    [[nodiscard]] u64 WritePositionPeek() const {
        return write_position;
    }
    [[nodiscard]] bool PendingFlushHeld() const {
        return pending_flush != nullptr;
    }
    /// FRITCHII_S4_COUNT (front, at a list entry): the same position without ending the open
    /// block, so the counter changes nothing the ring holds.
    [[nodiscard]] u64 WritePositionNow() const {
        return write_position;
    }
    /// the last compute/copy and unranged writer positions.
    [[nodiscard]] u64 LastWriterPosition() const {
        return last_writer_position;
    }
    [[nodiscard]] u64 LastUnrangedPosition() const {
        return last_unranged_position;
    }
    /// whether work sent after position `since` may write device bytes
    /// [begin, begin + size): compute or copies (any range), unranged writers, or a noted write
    /// that overlaps the range. Also true when more writes were noted since than are remembered.
    [[nodiscard]] bool MayBeWrittenSince(u64 since, u64 begin, u64 size) const {
        if (last_writer_position > since || last_unranged_position > since) {
            return true;
        }
        return NotedWriteSince(since, begin, size);
    }
    /// The noted-write part of MayBeWrittenSince: a NoteMemoryWrite sent after `since` overlaps
    /// [begin, begin + size), or more were noted since than are remembered.
    /// FRITCHII_NOTED_SUMMARY (live, default 0; F4 of work\FRONT_SPLIT_DESIGN.md 2d): the page
    /// buckets answer first (NotedWriteSinceSummary). Off: one relaxed load and a branch.
    /// FRITCHII_NOTED_FAST (live, default 0): the note chains answer instead of the walk
    /// (NotedWriteSinceFast). Off: one relaxed load and a branch.
    [[nodiscard]] bool NotedWriteSince(u64 since, u64 begin, u64 size) const {
        if (last_memory_writer_position <= since) {
            return false;
        }
        if (VideoCore::Nx2Debug::g_noted_fast.load(std::memory_order_relaxed) != 0)
            [[unlikely]] {
            return NotedWriteSinceFast(since, begin, size);
        }
        if (VideoCore::Nx2Debug::g_noted_summary.load(std::memory_order_relaxed) != 0)
            [[unlikely]] {
            return NotedWriteSinceSummary(since, begin, size);
        }
        u64 scanned = 0;
        return NotedWalk(since, begin, size, scanned) != NotedWalkResult::None;
    }
    /// The walk of NotedWriteSince, newest note first, down to `since`: Overlap (a note after
    /// `since` overlaps the bytes), Overflow (none did, but the table ends first: a dropped note
    /// may), None. `scanned`: the notes looked at.
    enum class NotedWalkResult : u8 { None, Overlap, Overflow };
    [[nodiscard]] NotedWalkResult NotedWalk(u64 since, u64 begin, u64 size, u64& scanned) const {
        const u64 end = begin + size;
        const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
        for (scanned = 0; scanned < remembered; ++scanned) {
            const PendingWrite& write =
                pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
            if (write.position <= since) {
                return NotedWalkResult::None;
            }
            if (write.begin < end && begin < write.end) {
                ++scanned;
                return NotedWalkResult::Overlap;
            }
        }
        return remembered == pending_writes.size() ? NotedWalkResult::Overflow
                                                   : NotedWalkResult::None;
    }
    /// FRITCHII_NOTED_SUMMARY (F4): checks that reached the walk's place, the ones the buckets
    /// answered (no noted write: the walk skipped at 1), why the others walked, the walk notes
    /// the answered checks would skip (2 and 3, and the timed ones at 1), the timed checks (1 in
    /// 64: the summary and the walk timed apart, in time stamp ticks), and the differences (the
    /// buckets said no write, the walk found an overlapping note: must be 0) and the walk
    /// overflows the buckets answered (the table's limit, not a difference). Session totals.
    struct NotedSummaryStats {
        u64 checks{};
        u64 answered{};
        u64 refused_bucket{};
        u64 refused_wide{};
        u64 refused_floor{};
        u64 refused_long{};
        u64 avoided_notes{};
        u64 timed{};
        u64 timed_answered{};
        u64 summary_ticks{};
        u64 walk_ticks_answered{};
        u64 differ{};
        u64 overflow_answered{};
    };
    [[nodiscard]] const NotedSummaryStats& GetNotedSummaryStats() const {
        return noted_summary.stats;
    }
    /// FRITCHII_NOTED_OVERFLOW 2 and 3 (Maxwell3D::FrontDmaRead, after NotedWriteSince answered a
    /// write): whether that answer was the table's overflow and the chains prove no noted write
    /// over the read since `since` (the read starting at `begin`). Clears it. Off: one plain load.
    [[nodiscard]] bool TakeNotedOverflowNo(u64 since, u64 begin) const {
        auto& overflow = noted_overflow;
        if (!overflow.proved) {
            return false;
        }
        overflow.proved = false;
        return overflow.proved_since == since && overflow.proved_begin == begin;
    }
    /// The front DMA read's check after such an answer: whether to time it (1 in 64), then what
    /// it decided. `checked` false: the ran-reads check is off, the read drains.
    [[nodiscard]] bool TimeNotedOverflowAfter() const {
        return (++noted_overflow.after_calls & 63) == 0;
    }
    void NoteOverflowAfter(bool checked, bool served, bool timed, u64 ticks) const {
        auto& s = noted_overflow.stats;
        ++s.after_reads;
        if (!checked) {
            ++s.after_unchecked;
            return;
        }
        ++(served ? s.after_served : s.after_drained);
        if (timed) {
            ++s.after_timed;
            s.after_ticks += ticks;
        }
    }
    /// FRITCHII_FRONT_RAN_READS (Maxwell3D::FrontDmaRead): whether the back ran every write noted
    /// over [begin, begin + size) after `since`, that is none of them is past `done` (ConsumedNow,
    /// loaded by the caller before it looks at anything else). The table is in ring order, newest
    /// last, so the scan stops at the first note at or before `done` (or `since`): every older one
    /// ran too, the ones the table dropped included. Queued: a note past `done` overlaps the
    /// bytes. PastTable: the table ends before the scan reaches `done`, so a dropped note past it
    /// may overlap them.
    enum class NotedRun : u8 { Ran, Queued, PastTable };
    [[nodiscard]] NotedRun NotedWritesRan(u64 since, u64 done, u64 begin, u64 size) const {
        const u64 stop = (std::max)(since, done);
        const u64 end = begin + size;
        const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
        for (u64 scanned = 0; scanned < remembered; ++scanned) {
            const PendingWrite& write =
                pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
            if (write.position <= stop) {
                return NotedRun::Ran;
            }
            if (write.begin < end && begin < write.end) {
                return NotedRun::Queued;
            }
        }
        return remembered == pending_writes.size() ? NotedRun::PastTable : NotedRun::Ran;
    }
    /// The newest NoteMemoryWrite / NoteMemoryWriteData sent after `since` over the u32 at
    /// `device` (what OverlayQueuedWrites(since) applies last to it), whether or not the back ran
    /// it since. `position` 0: none (or more were noted since than are remembered); `words` 0:
    /// its words are unknown, else `value` is the word it writes at `device`.
    struct NotedWord {
        u64 position;
        const char* what;
        u32 words;
        u32 value;
    };
    [[nodiscard]] NotedWord NewestNotedWriteTo(u64 since, u64 device) const {
        const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
        for (u64 scanned = 0; scanned < remembered; ++scanned) {
            const PendingWrite& write =
                pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
            if (write.position <= since) {
                break;
            }
            if (write.begin < device + sizeof(u32) && device < write.end) {
                NotedWord found{write.position, write.what, write.words, 0};
                if (const u64 offset = device - write.begin;
                    write.words != 0 && device >= write.begin && offset % sizeof(u32) == 0 &&
                    offset / sizeof(u32) < write.words) {
                    found.value = write.data[offset / sizeof(u32)];
                } else {
                    found.words = 0;
                }
                return found;
            }
        }
        return {};
    }
    /// "Front 0xe2c misses" (FrontDmaRead): for each of `out.size()` byte ranges [begin + i *
    /// span, begin + (i + 1) * span), the newest NoteMemoryWrite / NoteMemoryWriteData sent after
    /// `since` over it, whether or not the back ran it. `position` 0: none (or more were noted
    /// since than are remembered: `overflow`); `known`: its words are known to the front. Front
    /// only; one scan, newest first, stopping once every range has its write.
    struct NotedOver {
        u64 position;
        const char* what;
        bool known;
    };
    void NewestNotedWritesOver(u64 since, u64 begin, u64 span, std::span<NotedOver> out,
                               bool& overflow) const {
        std::ranges::fill(out, NotedOver{});
        overflow = false;
        size_t found = 0;
        const u64 remembered = (std::min<u64>)(pending_write_count, pending_writes.size());
        u64 scanned = 0;
        for (; scanned < remembered && found < out.size(); ++scanned) {
            const PendingWrite& write =
                pending_writes[(pending_write_count - 1 - scanned) % pending_writes.size()];
            if (write.position <= since) {
                return;
            }
            for (size_t i = 0; i < out.size(); ++i) {
                const u64 range_begin = begin + i * span;
                if (out[i].position == 0 && write.begin < range_begin + span &&
                    range_begin < write.end) {
                    out[i] = {write.position, write.what, write.words != 0};
                    ++found;
                }
            }
        }
        overflow = found < out.size() && scanned == pending_writes.size();
    }
    /// Work that writes the device range [begin, begin + size) was just sent (an inline upload, a
    /// report, a semaphore release). size ~0 means an unknown range.
    void NoteMemoryWrite(u64 begin, u64 size, const char* what = "unknown") {
        FrontSplit::NoteWriterSent(); // FRITCHII_FRONT_SPLIT_COUNT: one plain load off
        if (noted_summary.live) [[unlikely]] {
            NoteSummaryWrite(begin, size); // FRITCHII_NOTED_SUMMARY: one plain load off
        }
        if (write_stamps) [[unlikely]] {
            if (size == ~u64{0}) {
                unknown_write_position = write_position;
            } else {
                StampWrite(begin, begin + size, false);
            }
        }
        PendingWrite& write = pending_writes[pending_write_count++ % pending_writes.size()];
        write.what = what;
        write.begin = begin;
        write.end = size == ~u64{0} ? ~u64{0} : begin + size;
        write.position = write_position;
        write.words = 0;
        if (noted_fast.live) [[unlikely]] {
            LinkNotedFast(pending_write_count - 1); // FRITCHII_NOTED_FAST: one plain load off
        }
    }
    /// As NoteMemoryWrite, for a write whose words the front knows (a fence release payload,
    /// linear inline upload data): reads overlay them instead of waiting.
    void NoteMemoryWriteData(u64 begin, const u32* data, u32 words, const char* what = "known") {
        FrontSplit::NoteWriterSent(); // FRITCHII_FRONT_SPLIT_COUNT: one plain load off
        if (noted_summary.live && words != 0) [[unlikely]] {
            NoteSummaryWrite(begin, u64{words} * sizeof(u32)); // FRITCHII_NOTED_SUMMARY
        }
        if (write_stamps) [[unlikely]] {
            StampWrite(begin, begin + u64{words} * sizeof(u32), false);
        }
        while (words != 0) {
            const u32 count = (std::min<u32>)(words, PendingWriteWords);
            PendingWrite& write = pending_writes[pending_write_count++ % pending_writes.size()];
            write.what = what;
            write.begin = begin;
            write.end = begin + u64{count} * sizeof(u32);
            write.position = write_position;
            write.words = count;
            std::memcpy(write.data.data(), data, count * sizeof(u32));
            if (noted_fast.live) [[unlikely]] {
                LinkNotedFast(pending_write_count - 1); // FRITCHII_NOTED_FAST
            }
            begin += u64{count} * sizeof(u32);
            data += count;
            words -= count;
        }
    }
    /// The back's position, for OverlayQueuedWrites: take it before reading guest memory.
    [[nodiscard]] u64 ConsumedNow() {
        return PlayedNow(std::memory_order_acquire);
    }
    /// Applies to `out` (the words at device address `begin`, read from guest memory after
    /// `since` = ConsumedNow()) the queued writes the front knows the words of. False when an
    /// overlapping queued write's words are unknown (the caller waits instead).
    bool OverlayQueuedWrites(u64 since, u64 begin, u32* out, u32 words);
    /// Waits until the back executed the noted writes that overlap [begin, begin + size).
    /// Returns false when more writes are queued than are remembered (the caller drains).
    bool WaitForWritesTo(u64 begin, u64 size);
    /// Whether a noted write that overlaps [begin, begin + size) is still queued (or too many
    /// are queued to tell).
    bool HasQueuedWritesTo(u64 begin, u64 size);
    /// Front: a write that lands when a host query completes or a fence signals (counter
    /// reports, timed releases), not when the back runs it (Tegra::Nx2AsyncReportsSettled).
    void NoteAsyncWrite(u64 begin, u64 size) {
        // FRITCHII_FRONT_ASYNC_TABLE: off, one relaxed load, one member load and a branch.
        if ((FrontAsyncTableMode() | async_table_mode) != 0) [[unlikely]] {
            NoteAsyncWriteSwitched(begin, size);
            return;
        }
        for (u64 slot = begin >> 4; slot <= (begin + size - 1) >> 4; ++slot) {
            async_writes.Set(slot, write_position);
        }
    }
    /// Front: the ring position of the newest async write noted to these bytes, or 0.
    [[nodiscard]] u64 HasAsyncWrite(u64 begin, u64 size) {
        if ((FrontAsyncTableMode() | async_table_mode) != 0) [[unlikely]] {
            return HasAsyncWriteSwitched(begin, size); // FRITCHII_FRONT_ASYNC_TABLE
        }
        return OldAsyncNewest(begin, size);
    }

    /// A host-level wait was just sent (a syncpoint acquire, a host WFI, a GP entry with its sync
    /// bit; semaphore acquires drain): the guest's GPU work before it is complete before the
    /// commands after it are fetched.
    void NoteSync() {
        last_sync_position = write_position;
    }
    /// Waits until the back executed the work sent before the last NoteSync. Returns whether any
    /// of it was still queued.
    bool WaitForSync();
    /// Waits until the back executed the compute and copy work sent before the last NoteSync:
    /// the writers a command list fetched now is ordered after.
    /// Returns whether any of it was still queued. `site`: whose wait it is, for
    /// EDEN_NX2_GPU_PROFILE.
    bool WaitForSyncedWriters(DrainSite site = DrainSite::ListWriters);
    /// Whether the back executed the work sent before `position` (as far as it stored `consumed`,
    /// every few thousand words of a batch and at its end).
    [[nodiscard]] bool Ran(u64 position) const {
        return PlayedNow(std::memory_order_acquire) >= position;
    }
    /// Deferred work that can write memory was just sent (puller semaphore releases, maps).
    void NoteMemoryWrite() {
        last_memory_write_position = write_position;
        last_memory_write_method = DeferredWrite;
    }
    /// The 3D method just sent writes only the bytes its front noted (NoteMemoryWrite with a
    /// range, NoteMemoryWriteData): QueuedGuestWriters counts those, not the method as a writer
    /// of anything (as RangeLastWrite does for ranged writes). Only QueuedGuestWriters reads it.
    void RangedMethodWrite() {
        if (last_method_write_position == write_position) {
            last_method_write_position = previous_method_write_position;
        }
    }
    /// A method that writes memory was just sent; the method is kept for the GPU profile.
    void NoteMethodWrite(const Tegra::Engines::EngineInterface* engine, u32 method) {
        previous_memory_write_position = last_memory_write_position;
        previous_memory_write_method = last_memory_write_method;
        last_memory_write_position = write_position;
        // Engines all of whose methods write (compute, DMA, inline-to-memory, 2D) mark method 0.
        last_memory_write_method = engine->memory_write_mask[0] ? OtherEngineWrite : method;
        // QueuedGuestWriters: per-method writers only (3D); the engines all of whose methods
        // write note their ranges or count as unbounded or unranged writers. 3D wait_for_idle
        // (0x44) and sync_info (0xb2) write no guest memory themselves.
        if (!engine->memory_write_mask[0] && method != 0x44 && method != 0xb2) {
            previous_method_write_position = last_method_write_position;
            last_method_write_position = write_position;
        }
    }
    /// The method just sent writes only [begin, end): a macro memory read waits for it only where
    /// they overlap. Without this call its write counts for every read. EDEN_NX2_RANGED_WAITS=0
    /// keeps every write counting.
    /// `data` (optional): the words it writes there, in address space `space`, which a macro
    /// memory read of them takes without waiting (ForwardRead).
    /// `memory` (optional): the memory manager [begin, end) is in; with EXP126 not 0 the range is
    /// stamped by device address too (StampRangedDevice).
    void RangeLastWrite(GPUVAddr begin, GPUVAddr end, const u32* data = nullptr, size_t space = 0,
                        const Tegra::MemoryManager* memory = nullptr);
    /// For a macro memory read of `words` words at `begin` in address space `space`: when the
    /// newest work that may write them is one constant buffer upload still queued for the back,
    /// copies its words to `out` and returns true; the read then needs no wait.
    /// EDEN_NX2_FORWARD_READS=0 turns it off.
    bool ForwardRead(size_t space, GPUVAddr begin, u32 words, u32* out);
    /// For the front's macro memory reads (MME DMA) of [begin, end): waits until the back executed
    /// the methods that write memory in command order (memory_write_mask, NoteMemoryWrite; ranged
    /// writes only where they overlap), the compute and copy work and the work before the last
    /// host-level wait, all sent so far.
    bool WaitForMemoryWriters(DrainSite site, GPUVAddr begin, GPUVAddr end);
    /// Whether WaitForMemoryWriters(begin, end) would wait now: work that may write the range is
    /// still queued for the back.
    bool HasQueuedMemoryWriters(GPUVAddr begin, GPUVAddr end);
    /// Diagnostics: the newest queued work that may write [begin, end): 0 none, 1 compute or
    /// copy, 2 work before a host sync, 3 an unranged memory write (`method`), 4 a ranged write.
    u32 QueuedWriterKind(GPUVAddr begin, GPUVAddr end, u32& method);
    /// What sent a release NoteRelease records (the "Acquire resumes met by" counter).
    enum class ReleaseKind : u8 {
        Other,
        Puller,         ///< A puller semaphore release (WriteLong, SEM_EXECUTE)
        Payload3DShort, ///< A 3D payload release of 4 bytes (short query)
        Payload3DLong,  ///< A 3D payload report of 16 bytes (with the timestamp)
        ComputeReport,  ///< A compute 0x6c3 payload release
    };
    /// A semaphore release that writes `payload` at device address `device` (from GPU address
    /// `address`) was just sent (a puller release, a 3D payload report).
    void NoteRelease(u64 device, GPUVAddr address, u32 payload,
                     ReleaseKind kind = ReleaseKind::Other);
    /// The kind of the release recorded for the u32 at `device` (NoteRelease), whether or not the
    /// back ran it; Other when none is recorded there.
    [[nodiscard]] ReleaseKind RecordedReleaseKind(u64 device) const {
        const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
        return release.device == device && release.position != 0 ? release.kind
                                                                   : ReleaseKind::Other;
    }
    /// EXP13=2 (diagnostic): the newest work sent after `done` that may write guest memory over
    /// device bytes [device, device + size) (GPU addresses from `address`) when the back runs
    /// it, leaving out native compute launches and inline QMDs (the front's QMD scheduler copy
    /// keeps their words), draws and dispatches (their stores go to the host GPU's buffers; what
    /// a download brings back lands on pages marked GPU-written, which the callers check) and
    /// queued functions other than maps (fence releases: GPU-written pages, semaphore words and
    /// reports). `noted`: the newest write noted over the bytes (inline uploads, reports,
    /// semaphore releases, copies and compute releases with known ranges; those at a position in
    /// `skip` don't count); `unbounded`: the newest that may write anything (compute and copy
    /// writers without a range, unranged writers, 3D memory writes without a range, the ranged
    /// ring's overflow, maps, more writes than are remembered). 0 for none.
    /// `overflow_stamped` (EXP127, fable_fence_writers_brief 2a): the ranged overflow counts only
    /// for the dropped writes no device stamp covers (those sent before the device stamps
    /// started; all of them while the stamps are off). Exact only for a caller that has found
    /// WriteStamp over the same bytes not newer than `done` with EXP126 = 1: every other dropped
    /// write was stamped by GPU and device address before it entered the ring, so that check
    /// already says none of them writes these bytes. Its other unbounded writers count as ever.
    struct GuestWriters {
        u64 noted;
        u64 unbounded;
        GuestWriterCause noted_cause;
        GuestWriterCause unbounded_cause;
        const char* what; ///< UnknownPending: the noted write's `what`
        /// WriterRange: the range's kind and device bytes [range_begin, range_end).
        WriterKind range_kind;
        u64 range_begin;
        u64 range_end;
    };
    [[nodiscard]] GuestWriters QueuedGuestWriters(u64 done, GPUVAddr address, u64 device, u64 size,
                                                  std::span<const u64> skip = {},
                                                  bool overflow_stamped = false) const;
    /// EXP13=2/3, the inverse of QueuedGuestWriters (fable_answer_6 6f (c) a): once enabled,
    /// every write the front notes stamps the 256-byte blocks it covers (4 KB pages for ranges
    /// over 16 KB, everything for ranges over 1 MB) with its position, in hashed tables whose
    /// collisions only make stamps newer. WriteStamp is the newest position that may have
    /// written [device, device + size) or GPU [address, address + size) since: noted writes,
    /// writer ranges, 3D ranged writes, and the writers QueuedGuestWriters counts as unbounded.
    /// Everything sent before EnableWriteStamps counts as a write. Writes noted while a tag is
    /// set (SetWriteStampTag) are not counted by a WriteStamp asking with that tag (one inline
    /// upload patching what it already wrote), unless another writer stamped the block since.
    void EnableWriteStamps();
    void SetWriteStampTag(u64 tag) {
        write_stamp_tag = tag;
    }
    /// A tag no other write has (inline uploads of any engine).
    [[nodiscard]] u64 NewWriteStampTag() {
        return ++write_stamp_tags;
    }
    [[nodiscard]] u64 WriteStamp(u64 device, u64 size, GPUVAddr address, u64 skip_tag = 0) const;
    /// Whether EnableWriteStamps ran (WriteStamp answers the write position until then).
    [[nodiscard]] bool WriteStampsOn() const {
        return write_stamps;
    }
    /// EXP12=6: WriteStamp's block and page stamps over the bytes alone, without its floors
    /// (unbounded and unranged writers, 3D method writes, functions, unknown writes, the
    /// overflow): the newest write noted over them. The write position while the stamps are off.
    [[nodiscard]] u64 BytesStamp(u64 device, u64 size, GPUVAddr address) const {
        if (!write_stamps) {
            return write_position;
        }
        return (std::min)((std::max)({StampOf(device, device + size, false, 0),
                                      StampOf(address, address + size, true, 0),
                                      size != 0 ? StampReadFloor(device, size) : u64{0}}),
                          write_position);
    }
    /// The QMD wait floor counters (fable_qmd_wait_brief piece 1), front only. WriteStampWhy is
    /// WriteStamp's diagnostic twin, for the two QMD sites: `stamp` is what WriteStamp answers,
    /// `floorless` what it answers without the ranged overflow (EXP126 = 1), `term` which term
    /// set `stamp` and `term_position` that term's position.
    struct StampWhy {
        u64 stamp;
        u64 floorless;
        u64 term_position;
        StampTerm term;
    };
    [[nodiscard]] StampWhy WriteStampWhy(u64 device, u64 size, GPUVAddr address,
                                         u64 skip_tag = 0) const;
    /// For a wait or refusal set by a block or page stamp at `term_position`: what the exact scan
    /// (QueuedGuestWriters over the same bytes, writer ranges at a position in `skip` left out)
    /// finds. 0: a noted writer at or after the stamp (a real writer), 1: only an older one or
    /// an unbounded writer, 2: nothing queued (a hash collision, or a page stamp of a write
    /// elsewhere in the page).
    [[nodiscard]] u8 ExactWriterCheck(u64 done, GPUVAddr address, u64 device, u64 size,
                                      u64 term_position, std::span<const u64> skip = {}) const;
    /// The newest ranged write dropped from the full ring (0 for none).
    [[nodiscard]] u64 RangedOverflowPosition() const {
        return ranged_overflow_position;
    }
    /// EXP126 (live; fable_qmd_wait_brief piece 2): 0 WriteStamp counts the ranged overflow for
    /// every address (as before), 1 it does not, 2 as 0 while the QMD copy compares what 1
    /// would have served (NoteQmdTrial).
    [[nodiscard]] static int OverflowFloorMode();
    [[nodiscard]] static bool OverflowFloorOff() {
        return OverflowFloorMode() == 1;
    }
    /// EXP126 = 2: a QMD read where 1 would have served something else, or without the wait,
    /// compared with what this copy read (`kind`: what 1 served, 0 guest memory, 1 an overlay,
    /// 2 an upload shadow block); `differed`: the 256 bytes were not the same (must stay 0).
    void NoteQmdTrial(u8 kind, bool differed) {
        ++floor_stats.trial_compared[kind];
        if (differed) {
            ++floor_stats.trial_differed;
        }
    }
    /// EXP126 = 2: 1's copy would hold a different overlay or shadow block (a trial one).
    void NoteQmdTrialBuilt() {
        ++floor_stats.trial_built;
    }
    /// EXP126 = 2: a read of a block 1 changed in a way not followed: not compared.
    void NoteQmdTrialSkipped() {
        ++floor_stats.trial_skipped;
    }
    /// EXP126 = 2: trial entries the back had not passed dropped at the table's limit.
    void NoteQmdTrialLost(u64 count) {
        floor_stats.trial_lost += count;
    }
    void NoteQmdPatch(QmdPatchOutcome outcome) {
        ++floor_stats.patches[static_cast<size_t>(outcome)];
    }
    /// A small patch refused by the stamps; `exact`: ExactWriterCheck's answer (3: not asked).
    void NoteQmdPatchRefused(const StampWhy& why, u64 done, u8 exact);
    /// Every QMD read the front's copy makes from guest memory or its tables.
    void NoteQmdRead(u64 done);
    /// A QMD read that waits for `target`; `evicted`: an evicted inline QMD raised it past the
    /// stamps; `exact` as NoteQmdPatchRefused.
    void NoteQmdReadWait(const StampWhy& why, u64 done, u64 target, bool evicted, u8 exact);
    /// For a semaphore acquire of `device`: the payload a release still queued for the back writes
    /// there, if nothing sent after it may write the word. The back runs the acquire's channel
    /// after that release, so an acquire it meets needs no drain. EDEN_NX2_ACQUIRE_QUEUED=0 off.
    /// `miss` (when no payload): 1 no release recorded, 2 it already ran, 3 a later write.
    std::optional<u32> QueuedReleasePayload(u64 device, GPUVAddr address, u32* miss = nullptr);
    /// EDEN_NX2_ACQUIRE_PRECHECK=2's reset check (puller.cpp PrecheckAudit): the payload of the
    /// newest release recorded for the u32 at `device` (NoteRelease) while the back has not run
    /// it (its ring position is past `done`), whatever was sent after it (unlike
    /// QueuedReleasePayload); `position` gets that ring position. queued_releases keeps one
    /// release per slot: a newer release to another word in the same slot hides it.
    [[nodiscard]] std::optional<u32> QueuedReleaseTo(u64 device, u64 done, u64& position) const {
        const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
        if (release.device != device || release.position == 0 || release.position <= done) {
            return std::nullopt;
        }
        position = release.position;
        return release.payload;
    }
    /// EDEN_NX2_RUNAHEAD: the newest release recorded for the u32 at `device` (NoteRelease),
    /// whether or not the back ran it; `position` gets its ring position.
    [[nodiscard]] std::optional<u32> NewestReleaseTo(u64 device, u64& position) const {
        const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
        if (release.device != device || release.position == 0) {
            return std::nullopt;
        }
        position = release.position;
        return release.payload;
    }
    /// EDEN_NX2_RUNAHEAD: the ring position of the newest native compute work that may dispatch
    /// (NoteDispatchWriter), 0 for none this session.
    [[nodiscard]] u64 LastDispatchPosition() const {
        return last_dispatch_position;
    }
    /// EXP12: the ring position past the queued work that may write the u32 at `device` without
    /// the front knowing its words, so a semaphore read that waits for or overlays the noted
    /// writes (EDEN_NX2_SEM_FAST) can't account for it: compute and copies (their QMD and report
    /// releases among them), unranged writers, a release recorded for the word, and queued
    /// functions (maps). Draws that store to the word mark its page GPU-written once they ran.
    [[nodiscard]] u64 UntrackedWriterPosition(u64 device) const {
        u64 target =
            (std::max)({last_writer_position, last_unranged_position, last_function_position});
        if (const QueuedRelease& release = queued_releases[(device >> 2) % queued_releases.size()];
            release.device == device) {
            target = (std::max)(target, release.position);
        }
        return (std::min)(target, write_position);
    }
    /// The entry just sent also writes [address, address + size) of `memory`, noted by device
    /// page (other mappings may alias it, and the caches keep state per page).
    void NoteWrite(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size);
    /// The entry just sent writes guest memory the front can't tell.
    void NoteUnknownWrite() {
        writes.Entry(Effect::Unknown, write_position);
    }
    /// Waits for the queued functions (GPU memory maps among them) before the front translates
    /// addresses, then drops the pending writes the back has run. Returns the back's position.
    u64 SettleWrites();
    /// Ring position the back must pass to execute possible writers of [begin, end).
    /// Deferred fence writes can still be pending afterwards.
    [[nodiscard]] u64 ReadTarget(u64 begin, u64 end) const {
        return writes.ReadTarget(begin, end);
    }
    [[nodiscard]] const SplitWriteTracker& Writes() const {
        return writes;
    }
    /// Waits until the back consumed `position` (not what came after). Returns whether it had
    /// not yet.
    bool WaitUntil(u64 position, DrainSite site) {
        return WaitFor((std::min)(position, write_position), site);
    }

private:
    friend struct GpuSplitTestAccess;
    friend struct MemStageShadowAccess; // memstage_shadow.cpp reads the ring's layout
    friend struct MemoryStageRingAccess; // memory_stage_link.cpp: the stage's cursors
    friend struct ESplitEngineAccess;    // fritchii_e_split_thread.cpp: batched non-3D engine execution

    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
        Pairs,  ///< Bits 32-63 count; then one word per write: bits 0-15 method, 32-63 value
        StateDelta, ///< As Pairs, Maxwell3D macro writes (DeltaWrite)
        DrawBatch,  ///< Bits 8-15 count, 32-63 repeats; then count words (method | arg << 32)
        /// FRITCHII_CB_WORDS_BLOCK (CbWord): bits 32-63 count; with CbVerify word 1 is the
        /// front's address of the first word; then count Method words, each the word its own
        /// Method entry would have been (3D constant buffer data, no segment)
        CbWords,
    };
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)
    static constexpr u64 Sink = 1 << 7;   ///< Method: into the method sink, tagged (SinkMethod)
    static constexpr u64 CbVerify = 1 << 4; ///< CbWords: FRITCHII_CB_WORDS_BLOCK=2, word 1 held
    /// CbWords: FRITCHII_CB_MERGE >= 2 joined register writes (and marks) into the block.
    static constexpr u64 CbJoined = 1 << 5;
    /// The most words a CbWords block holds: one back gather (Execute, CallConstBufferWords).
    static constexpr u32 MaxCbWords = 64;

    // Read when the split first starts (MaybeStart), after main() applied nx2_switches.txt: a
    // static initialiser would run before it.
    static inline bool macro_sink{};
    static inline bool compute_sink{};
    static inline bool ring_pairs{};
    static constexpr u32 MaxPairs = 4096;

    /// async_writes' table: 16-byte slot -> ring position, open addressing with linear probing.
    /// As the std::unordered_map it replaces: entries are never erased (only clear(), at Reset),
    /// a note to a known slot overwrites its position. It allocates only when it doubles (at
    /// half full), not a node per new slot (FrontNoteMemoryWrite -> NoteAsyncWrite was 1.4-1.5%
    /// of the front, ~15 ms/s, fable_answer_regression_2 4.4).
    class AsyncWriteTable {
    public:
        [[nodiscard]] bool empty() const {
            return count == 0;
        }
        void clear() {
            if (count != 0) {
                std::fill(entries.begin(), entries.end(), Entry{EmptyKey, 0});
                count = 0;
            }
        }
        void Set(u64 key, u64 position) {
            if ((count + 1) * 2 > entries.size()) [[unlikely]] {
                Grow();
            }
            for (size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
                Entry& entry = entries[index];
                if (entry.key == key) {
                    entry.position = position;
                    return;
                }
                if (entry.key == EmptyKey) {
                    entry = {key, position};
                    ++count;
                    return;
                }
            }
        }
        /// The position noted to `key`, or null.
        [[nodiscard]] const u64* Find(u64 key) const {
            if (count == 0) {
                return nullptr;
            }
            for (size_t index = Hash(key) & mask;; index = (index + 1) & mask) {
                const Entry& entry = entries[index];
                if (entry.key == key) {
                    return &entry.position;
                }
                if (entry.key == EmptyKey) {
                    return nullptr;
                }
            }
        }
        /// FRITCHII_FRONT_ASYNC_TABLE: every slot and its position (switch changes copy them).
        template <typename Func>
        void ForEach(Func&& func) const {
            for (const Entry& entry : entries) {
                if (entry.key != EmptyKey) {
                    func(entry.key, entry.position);
                }
            }
        }
        /// Empties the table and frees its memory (the block table holds the slots instead).
        void Release() {
            std::vector<Entry>().swap(entries);
            mask = 0;
            count = 0;
        }
        [[nodiscard]] size_t Size() const {
            return count;
        }
        [[nodiscard]] size_t Capacity() const {
            return entries.size();
        }

    private:
        /// Slots are device addresses >> 4: never all ones.
        static constexpr u64 EmptyKey = ~u64{0};
        struct Entry {
            u64 key;
            u64 position;
        };
        [[nodiscard]] static size_t Hash(u64 key) {
            const u64 mixed = key * 0x9e3779b97f4a7c15ULL;
            return static_cast<size_t>(mixed ^ (mixed >> 29));
        }
        void Grow() {
            std::vector<Entry> old = std::move(entries);
            entries.assign(old.empty() ? size_t{1024} : old.size() * 2, Entry{EmptyKey, 0});
            mask = entries.size() - 1;
            count = 0;
            for (const Entry& entry : old) {
                if (entry.key != EmptyKey) {
                    Set(entry.key, entry.position);
                }
            }
        }
        std::vector<Entry> entries;
        size_t mask{};
        size_t count{};
    };
    AsyncWriteTable async_writes; ///< Front: 16-byte slot -> ring position

    /// FRITCHII_FRONT_ASYNC_TABLE (live, default 0; helper AA): async_writes kept per block of 8
    /// neighbouring slots (128 bytes of guest memory). A block's 8 positions fill one 64-byte line,
    /// so the reports the game writes next to each other land in one line instead of one random
    /// line each, and a reader's 3 slots (the 0xe2c read: 15 bytes before + 24) are one or two
    /// lines. A small open-addressed index maps the block number to the block, and a one-entry
    /// memo per side (notes, reads) skips the index for runs within one block. A slot never noted
    /// reads 0, which is what a missing slot gives HasAsyncWrite (a newest of 0): the answers
    /// are the same as async_writes', slot for slot. Never erased, as async_writes (only at
    /// Reset or when the switch hands the slots back).
    class AsyncBlockTable {
    public:
        static constexpr u32 SlotsPerBlock = 8;
        [[nodiscard]] bool Empty() const {
            return count == 0;
        }
        /// Empties the table and frees its memory.
        void Release() {
            std::vector<Index>().swap(index);
            std::vector<Block>().swap(blocks);
            mask = 0;
            count = 0;
            set_memo_key = EmptyKey;
            find_memo_key = EmptyKey;
        }
        void Set(u64 slot, u64 position) {
            const u64 key = slot / SlotsPerBlock;
            if (key != set_memo_key) {
                set_memo_block = Insert(key);
                set_memo_key = key;
            }
            blocks[set_memo_block].positions[slot % SlotsPerBlock] = position;
        }
        /// The newest position noted to slots [first, last], or 0.
        [[nodiscard]] u64 Newest(u64 first, u64 last) {
            u64 newest = 0;
            if (count == 0 || first > last) {
                return newest;
            }
            const u64 first_key = first / SlotsPerBlock;
            const u64 last_key = last / SlotsPerBlock;
            for (u64 key = first_key;; ++key) {
                if (const Block* const block = Lookup(key)) {
                    const u32 from = key == first_key ? static_cast<u32>(first % SlotsPerBlock) : 0;
                    const u32 to =
                        key == last_key ? static_cast<u32>(last % SlotsPerBlock) : SlotsPerBlock - 1;
                    for (u32 i = from; i <= to; ++i) {
                        newest = (std::max)(newest, block->positions[i]);
                    }
                }
                if (key == last_key) {
                    break;
                }
            }
            return newest;
        }
        /// Every slot of every block and its position (0 where never noted).
        template <typename Func>
        void ForEach(Func&& func) const {
            for (const Index& entry : index) {
                if (entry.key == EmptyKey) {
                    continue;
                }
                const Block& block = blocks[entry.block];
                for (u32 i = 0; i < SlotsPerBlock; ++i) {
                    if (block.positions[i] != 0) {
                        func(entry.key * SlotsPerBlock + i, block.positions[i]);
                    }
                }
            }
        }
        [[nodiscard]] size_t Blocks() const {
            return count;
        }

    private:
        /// Block numbers are slots / 8: never all ones.
        static constexpr u64 EmptyKey = ~u64{0};
        struct alignas(64) Block {
            std::array<u64, SlotsPerBlock> positions{};
        };
        struct Index {
            u64 key;
            u64 block;
        };
        [[nodiscard]] static size_t Hash(u64 key) {
            const u64 mixed = key * 0x9e3779b97f4a7c15ULL;
            return static_cast<size_t>(mixed ^ (mixed >> 29));
        }
        /// The block of `key`, made (all slots 0) when new.
        size_t Insert(u64 key) {
            if ((count + 1) * 2 > index.size()) [[unlikely]] {
                Grow();
            }
            for (size_t i = Hash(key) & mask;; i = (i + 1) & mask) {
                Index& entry = index[i];
                if (entry.key == key) {
                    return static_cast<size_t>(entry.block);
                }
                if (entry.key == EmptyKey) {
                    entry = {key, blocks.size()};
                    blocks.emplace_back();
                    ++count;
                    return static_cast<size_t>(entry.block);
                }
            }
        }
        [[nodiscard]] const Block* Lookup(u64 key) {
            if (key == find_memo_key) {
                return &blocks[find_memo_block];
            }
            for (size_t i = Hash(key) & mask;; i = (i + 1) & mask) {
                const Index& entry = index[i];
                if (entry.key == key) {
                    find_memo_key = key;
                    find_memo_block = static_cast<size_t>(entry.block);
                    return &blocks[find_memo_block];
                }
                if (entry.key == EmptyKey) {
                    return nullptr;
                }
            }
        }
        void Grow() {
            std::vector<Index> old = std::move(index);
            index.assign(old.empty() ? size_t{256} : old.size() * 2, Index{EmptyKey, 0});
            mask = index.size() - 1;
            for (const Index& entry : old) {
                if (entry.key == EmptyKey) {
                    continue;
                }
                for (size_t i = Hash(entry.key) & mask;; i = (i + 1) & mask) {
                    if (index[i].key == EmptyKey) {
                        index[i] = entry;
                        break;
                    }
                }
            }
        }
        std::vector<Index> index;
        std::vector<Block> blocks; ///< In the order they were made; never moved between slots
        size_t mask{};
        size_t count{};
        // Memos: block numbers stay valid (blocks are only appended, until Release).
        u64 set_memo_key{EmptyKey};
        size_t set_memo_block{};
        u64 find_memo_key{EmptyKey};
        size_t find_memo_block{};
    };
    AsyncBlockTable async_blocks; ///< Front, FRITCHII_FRONT_ASYNC_TABLE 1 to 3
    /// The FRITCHII_FRONT_ASYNC_TABLE value the tables are kept for (front only): 0 async_writes
    /// alone, 1 async_blocks alone, 2 and 3 both.
    int async_table_mode{};
    struct AsyncTableStats {
        u64 calls;       ///< switched calls (the log cadence)
        u64 notes;       ///< NoteAsyncWrite calls
        u64 finds;       ///< HasAsyncWrite calls
        u64 checked;     ///< 2: answers compared
        u64 differ;      ///< 2: of them, different (must be 0)
        u64 timed_notes; ///< 3: notes timed in both tables
        u64 note_old_ticks;
        u64 note_new_ticks;
        u64 timed_finds; ///< 3: reads timed in both tables
        u64 find_old_ticks;
        u64 find_new_ticks;
        u64 handovers; ///< switch changes that copied the slots between the tables
    };
    AsyncTableStats async_stats{};
    AsyncTableStats async_logged{};
    u64 async_window_ns{};
    u64 async_window_ticks{};
    u64 async_differ_total{};
    [[nodiscard]] static int FrontAsyncTableMode() {
        return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_front_async_table,
                                                "FRITCHII_FRONT_ASYNC_TABLE");
    }
    /// HasAsyncWrite from async_writes, as before the switch.
    [[nodiscard]] u64 OldAsyncNewest(u64 begin, u64 size) const {
        u64 newest = 0;
        if (async_writes.empty()) {
            return newest;
        }
        for (u64 slot = begin >> 4; slot <= (begin + size - 1) >> 4; ++slot) {
            if (const u64* const position = async_writes.Find(slot)) {
                newest = (std::max)(newest, *position);
            }
        }
        return newest;
    }
    /// Keeps the tables the switch value `mode` needs, copying the slots over when it changed.
    void SyncAsyncTables(int mode);
    void NoteAsyncWriteSwitched(u64 begin, u64 size);
    [[nodiscard]] u64 HasAsyncWriteSwitched(u64 begin, u64 size);
    void LogAsyncTable(int mode);

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;
    /// EXP39 & 1: the fewest words the front publishes to a back that waits for them (Commit).
    /// A back that runs faster than the front would otherwise take one entry per publish.
    static constexpr size_t PublishAfterWait = 8;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        RunPendingFlush(); // Held-back work goes before any other entry
        // What follows ends the open Pairs, StateDelta or CbWords block.
        pairs_header = nullptr;
        delta_header = nullptr;
        cb_header = nullptr;
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words, Effect effect) {
        write_position += words;
        writes.Entry(effect, write_position);
        // EXP39 & 1: the back waits for work at the last
        // position published (it stored it in back_idle_at before it spun), so it waits for
        // exactly the words the front holds: it gets them after PublishAfterWait, not after
        // publish_every. Not while a Pairs, StateDelta or CbWords block is open (publishing ends
        // it). Replaces EXP27 & 2, which published every entry while the back was idle.
        const u64 unpublished = write_position - published_local;
        if (unpublished >= publish_every ||
            (publish_waiting && unpublished >= PublishAfterWait && pairs_header == nullptr &&
             delta_header == nullptr && cb_header == nullptr &&
             back_idle_at.load(std::memory_order_relaxed) == published_local)) {
            Publish();
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        // Before the engine is compared: the held-back work selects its own.
        RunPendingFlush();
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    bool WaitFor(u64 target, DrainSite site);

    /// One step of a front wait for the back to consume `target` (`on_engine`: UsesEngine).
    void Backoff(u32 spins, u64 target, bool on_engine);
    /// SiteOnEngine with the switch and the two positions on (once per wait).
    [[nodiscard]] bool UsesEngine(DrainSite site) const;
    /// Whether the front's wait for `target` is over: R played it, or with `on_engine` E ran it
    /// and R played the newest render op before it that writes game memory (a report).
    [[nodiscard]] bool Reached(u64 target, bool on_engine, std::memory_order order) const;
    /// After the back stored `consumed`: wakes a front sleeping in Backoff.
    void NotifyFront() {
        if (front_waiting.load(std::memory_order_seq_cst)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            front_wake.notify_one();
        }
    }

    /// The back's store of `consumed` (and the front's wake). With one follower on, clamped to
    /// R's oldest unplayed packet (step 4; with every packet lent R is never behind here).
    void StoreConsumed(u64 position);

    void BackLoop(std::stop_token stop_token);
    void BackWaitForStage(std::stop_token& stop_token, u64 read_position);
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its size.
    size_t Execute(const u64* entry, size_t words);
    /// FRITCHII_ESPLIT_ASYNC_KINDS bit 2, on E2: a Method, Multi or Pairs entry of a non-3D
    /// `engine` (the compute engine), as Execute runs it with `engine` bound (E1 has moved on to
    /// other engines). Returns its size.
    size_t ExecuteOnEngine(Tegra::Engines::EngineInterface* engine, const u64* entry);
    /// The size Execute takes for a non-3D engine's Method, Multi or Pairs entry, or a Call.
    [[nodiscard]] static size_t PlainEntrySize(u64 head);
    /// Back, Kind::CbWords: the block's `count` Method words, each run as its own entry would
    /// run (Execute, whose constant buffer gather takes them at once). `verify_address`: the
    /// front's address word (FRITCHII_CB_WORDS_BLOCK=2, VerifyCbWords).
    void ExecuteCbWords(const u64* words, u32 count, const u64* verify_address);
    /// FRITCHII_CB_WORDS_BLOCK=2 on the back: the front's address against the registers', then
    /// the block applied once as a block and once word by word from the same state, registers,
    /// shadow RAM words and dirty flags compared, memory against the words' values.
    void VerifyCbWords(const u64* words, u32 count, u64 front_address);
    /// Front: the "CB words block" line every 10 s (from Publish, with ReportBatched).
    void ReportCbWords();

    Core::System* system{};
    bool active{};
    bool front_macros{};
    std::unique_ptr<u64[]> ring;

    // Front side.
    u64 write_position{};
    u64 last_writer_position{};
    u64 last_memory_writer_position{};
    u64 last_unranged_position{};
    u64 unranged_without_exp120{};
    EngineWriteNotes engine_write_notes{};
    static constexpr u32 PendingWriteWords = 8;
    struct PendingWrite {
        u64 begin;
        u64 end;
        u64 position;
        u32 words; ///< Known words from begin, or 0 (unknown contents)
        const char* what;
        std::array<u32, PendingWriteWords> data;
    };
    static constexpr size_t PendingWriteSlots = 8192;
    std::array<PendingWrite, PendingWriteSlots> pending_writes{};
    u64 pending_write_count{};
    /// FRITCHII_NOTED_SUMMARY (F4, front only): the newest note position per page bucket (4,096
    /// buckets by page hash), and one for notes over more than 16 pages or of unknown range.
    /// Notes are bucketed only while `live`; `floor` is the position when it went live (notes
    /// before it were never bucketed). Mutable: the const query turns it on and counts.
    static constexpr size_t NotedBuckets = 4096;
    static constexpr u64 NotedMaxPages = 16;
    struct NotedSummary {
        bool live{};
        u32 calls{};
        u64 floor{};
        u64 wide{};
        u64 window_ns{};
        u64 window_ticks{};
        NotedSummaryStats stats{};
        NotedSummaryStats logged{};
        std::array<u64, NotedBuckets> buckets{};
    };
    mutable NotedSummary noted_summary;
    [[nodiscard]] static size_t NotedBucket(u64 page) {
        return static_cast<size_t>((page * 0x9e3779b97f4a7c15ULL) >> 52);
    }
    /// The note path's half (front, while live): buckets the note [begin, begin + size).
    void NoteSummaryWrite(u64 begin, u64 size);
    /// NotedWriteSince with the switch on (any value): modes 1, 2 and 3.
    [[nodiscard]] bool NotedWriteSinceSummary(u64 since, u64 begin, u64 size) const;
    void LogNotedSummary(int mode) const;
    /// FRITCHII_NOTED_FAST (front only): every note in pending_writes sits in one chain, newest
    /// first: the chain of its page bucket (4,096 by page hash) when it lies inside one page,
    /// else the wide chain (more than one page, or an unknown range). `heads` and `wide_head`
    /// hold the newest note's number + 1 (pending_write_count when it was made; 0: none),
    /// `links` (per table slot) the previous note of the same chain, number + 1. Notes are
    /// linked only while `live`; `floor` is the note number when it went live (older notes are
    /// in no chain). Mutable: the const query turns it on and counts.
    struct NotedFastStats {
        u64 checks{};
        u64 answered{};       ///< answered without the walk (modes 2 and 3: would be)
        u64 answered_yes{};   ///< of them, a write since (overlap, or the table's overflow)
        u64 overflow_yes{};   ///< of those, the table's overflow (no chain looked at)
        u64 floor{};          ///< walked: the walk would reach a note older than the chains
        u64 long_reads{};     ///< walked: a read over 16 pages (or wrapping)
        u64 touched{};        ///< chain notes looked at on answered checks
        u64 walk_notes{};     ///< walk notes on answered checks whose walk ran too
        u64 walk_checks{};    ///< answered checks whose walk ran too
        u64 timed{};          ///< 1 in 64 checks: both timed
        u64 fast_ticks{};     ///< the chain answer (or its refusal) on timed checks
        u64 walk_ticks{};     ///< the walk on timed answered checks
        u64 differ{};         ///< the chains and the walk disagree (must be 0)
        u64 linked{};         ///< notes linked
        u64 linked_wide{};    ///< of them, into the wide chain
        u64 linked_unknown{}; ///< of those, of unknown range
        u64 linked_covered{}; ///< notes linked past their chain's head (it covers the head)
    };
    static constexpr size_t NotedChains = 4096;
    struct NotedFastKind {
        const char* what;
        u64 count;
    };
    struct NotedFast {
        bool live{};
        u32 calls{};
        u64 floor{};
        u64 wide_head{};
        u64 window_ns{};
        u64 window_ticks{};
        NotedFastStats stats{};
        NotedFastStats logged{};
        std::array<NotedFastKind, 8> wide_kinds{}; ///< the wide notes by `what`, this window
        std::array<u64, NotedChains> heads{};
        std::array<u64, PendingWriteSlots> links{};
        /// FRITCHII_NOTED_OVERFLOW (F7): the position of the note each link and head names (0
        /// for none), kept while live whatever the switch, so a chain still knows how far back
        /// it reaches once the table forgot the note. `floor_position`: the newest note before
        /// the floor (0 when there was none), the newest position no chain holds.
        u64 floor_position{};
        u64 wide_head_position{};
        std::array<u64, NotedChains> head_positions{};
        std::array<u64, PendingWriteSlots> link_positions{};
    };
    mutable NotedFast noted_fast;
    /// FRITCHII_NOTED_OVERFLOW (F7, front only): the overflow answers of NotedWriteSinceFast asked
    /// of the chains, why the others stay a write, the chain cost (1 in 64 timed), and what the
    /// front DMA read does after the write answer for the reads the chains proved clean
    /// (NoteOverflowAfter). Window counts; the differences also as a session total.
    struct NotedOverflowStats {
        u64 checks{};         ///< NotedWriteSinceFast checks
        u64 overflow{};       ///< of them, the table's overflow
        u64 proved_no{};      ///< of those, the chains prove no noted write since
        u64 overlap{};        ///< a chain note after `since` overlaps the read
        u64 forgotten{};      ///< a chain reaches a forgotten note after `since`
        u64 forgotten_wide{}; ///< of them, in the wide chain
        u64 floor{};          ///< notes before the chains went live may be after `since`
        u64 long_reads{};     ///< a read over 16 pages (or wrapping)
        u64 touched{};        ///< chain notes looked at
        u64 timed{};          ///< 1 in 64 overflow answers: the chain walk timed
        u64 chain_ticks{};
        u64 after_reads{};    ///< front DMA reads with a proved-clean overflow answer
        u64 after_served{};   ///< of them, the ran-reads check served them anyway
        u64 after_drained{};  ///< of them, the ran-reads check sent them to the drain
        u64 after_unchecked{}; ///< of them, no ran-reads check (switch off): a drain
        u64 after_timed{};    ///< 1 in 64 of the checked ones: the ran-reads check timed
        u64 after_ticks{};
        u64 differ{};         ///< verify: proved clean, a remembered note overlaps (must be 0)
        u64 shadow_checked{}; ///< verify: 1 in 16 proved answers checked against the shadow
        u64 shadow_differ{};  ///< of them, a shadow note after `since` overlaps (must be 0)
        u64 shadow_short{};   ///< of them, the shadow does not reach back to `since`
    };
    /// Verify's shadow (FRITCHII_NOTED_OVERFLOW=2): the last 65,536 notes' ranges and positions,
    /// eight times the table, so a proved answer can be checked past the notes the table forgot.
    struct NotedShadowNote {
        u64 begin;
        u64 end;
        u64 position;
    };
    static constexpr size_t NotedShadowSlots = 65536;
    struct NotedOverflow {
        u32 calls{};
        u32 after_calls{};
        u32 verify_calls{};
        u64 window_ns{};
        u64 window_ticks{};
        NotedOverflowStats stats{};
        NotedOverflowStats logged{};
        u64 session_differ{};
        /// The last check the chains proved clean (for TakeNotedOverflowNo).
        bool proved{};
        u64 proved_since{};
        u64 proved_begin{};
        /// Verify's shadow: filled by LinkNotedFast while the switch is 2; `shadow_start` is the
        /// first note it holds (it restarts with the chains' floor).
        std::unique_ptr<NotedShadowNote[]> shadow;
        bool shadow_live{};
        u64 shadow_start{};
        u64 shadow_floor{};
    };
    mutable NotedOverflow noted_overflow;
    enum class NotedOverflowAnswer : u8 { No, Overlap, Forgotten, ForgottenWide, Floor, Long };
    /// With the table overflowed (its oldest note after `since`): the chains' answer from the
    /// wide chain and the read's page buckets. No: no note after `since` overlaps the read, the
    /// forgotten ones included. Every other answer stays a write.
    [[nodiscard]] NotedOverflowAnswer NotedOverflowLookup(u64 since, u64 begin, u64 size,
                                                          u64& touched) const;
    /// NotedWriteSinceFast's overflow answer with the switch at 1 to 3: the lookup, counted and
    /// timed; true when the chains prove no noted write.
    [[nodiscard]] bool NotedOverflowCheck(int mode, u64 since, u64 begin, u64 size) const;
    /// Verify (FRITCHII_NOTED_OVERFLOW=2) of a proved answer against the shadow.
    void VerifyNotedOverflowShadow(u64 since, u64 begin, u64 size) const;
    void LogNotedOverflow(int mode) const;
    /// The note path's half (front, while live): links note number `seq` into its chain.
    void LinkNotedFast(u64 seq);
    /// NotedWriteSince with FRITCHII_NOTED_FAST set: modes 1, 2 and 3 (else the summary or the
    /// walk, as with it at 0).
    [[nodiscard]] bool NotedWriteSinceFast(u64 since, u64 begin, u64 size) const;
    enum class NotedFastAnswer : u8 { No, Yes, Overflow, Floor, Long };
    /// The chains' answer for NotedWriteSince: No / Yes / Overflow (exactly the walk's
    /// None / Overlap / Overflow mapped to a bool), or Floor / Long (not answered: walk).
    /// `touched`: the chain notes looked at.
    [[nodiscard]] NotedFastAnswer NotedFastLookup(u64 since, u64 begin, u64 size,
                                                  u64& touched) const;
    void LogNotedFast(int mode) const;
    /// Compute and copy writes with a known device range (NoteWriterRange), newest last.
    struct WriterRange {
        u64 position;
        u64 begin;
        u64 end;
        WriterKind kind; ///< For EXP13=2's wait histogram
    };
    // 8,192 / 4,096 (was 1,024 / 512): at the Castle Town pot room ~11,500 ranges are noted a
    // second (compute inline uploads), so a page read every few frames was always past the
    // old limit and every front DMA read of it drained. Both numbers only bound work: past
    // them the answer is "may write" without looking, so a larger limit is exact.
    std::array<WriterRange, 8192> writer_ranges{};
    static constexpr u64 MaxWriterScan = 4096;
    u64 writer_range_count{};
    u64 last_unbounded_writer_position{};
    std::array<u64, static_cast<size_t>(WriterKind::Count)> unbounded_writers{};
    std::array<u64, static_cast<size_t>(ReleaseNote::Count)> release_notes{};
    /// NoteWriterRange's translation: noted, or why not (a ReleaseNote past Ranged).
    ReleaseNote TryNoteWriterRange(const Tegra::MemoryManager& memory, GPUVAddr address,
                                   u64 size, WriterKind kind);
    /// NoteWriterRange's last range and the function position its translation holds for: the
    /// same range again (an inline upload's data words) only moves the newest range's position,
    /// unless a check may have kept what it found for it (below writer_range_sealed).
    mutable u64 writer_range_sealed{};
    const Tegra::MemoryManager* writer_range_memory{};
    GPUVAddr writer_range_address{};
    u64 writer_range_size{};
    u64 writer_range_function{~u64{0}};
    /// EXP122 page watch (gpu_split.cpp), front only.
    struct PageWatch;
    std::unique_ptr<PageWatch> page_watch;
    u32 watched_pages{};
    int page_watch_mode{};
    enum class WatchAnswer : u8 { Unwatched, NoWrite, Write };
    WatchAnswer WatchedWritesSince(u64 since, u64 begin, u64 size, u64 known,
                                   ListedWrites* found) const;
    void NoteWatchedWrite(u64 begin, u64 end, u64 position);
    void ClearPageWatch();
    /// Front, EXP10 dispatches (NoteDispatchWriter): the newest sent, the newest sent while the
    /// back did not record (EXP10 off), whether the back was last told to record, and how many.
    u64 last_dispatch_position{};
    u64 last_unrecorded_dispatch_position{};
    bool dispatch_records{};
    u64 dispatches_sent{};
    /// Dispatch writes the back recorded (begin, end; 0, ~0 for anything) with the position of
    /// the entry that ran them, newest last; written by the back, read by the front.
    struct DispatchWrite {
        std::atomic<u64> position;
        std::atomic<u64> begin;
        std::atomic<u64> end;
        std::atomic<u8> kind; ///< DispatchWriteKind (FRITCHII_DISPATCH_WAIT_CENSUS)
    };
    static constexpr u64 DispatchWriteSlots = 8192;
    static constexpr u64 MaxDispatchScan = 2048;
    std::array<DispatchWrite, DispatchWriteSlots> dispatch_writes{};
    alignas(64) std::atomic<u64> dispatch_write_count{};
    /// DispatchStampsMissSince's stamps: per 64 KB device block (hashed by block number), the
    /// newest position a dispatch write over it was recorded at; `dispatch_wide_stamp` for the
    /// writes that cover everything. Stored by AppendDispatchWrite before the count.
    static constexpr u64 DispatchStampBits = 16;
    static constexpr u64 DispatchStampSlots = 65536;
    std::array<std::atomic<u64>, DispatchStampSlots> dispatch_stamps{};
    std::atomic<u64> dispatch_wide_stamp{};
    std::atomic<u64> dispatches_bounded{};
    std::array<std::atomic<u64>, static_cast<size_t>(DispatchUnbounded::Count)>
        dispatches_unbounded{};
    void AppendDispatchWrite(u64 begin, u64 end, DispatchWriteKind kind);
    /// FRITCHII_DISPATCH_WAIT_CENSUS: classifies the records of one wait (gpu_split.cpp).
    void CensusDispatchWait(u64 consumed_before, u64 count_before, u64 begin, u64 size,
                            u64 wait_ns);
    DispatchWaitCensus dispatch_census{};
    DispatchWalkStats dispatch_walk{};
    u64 last_sync_position{};
    u64 last_memory_write_position{};
    static constexpr u32 DeferredWrite = ~0U;
    static constexpr u32 OtherEngineWrite = ~0U - 1;
    /// The 3D method behind last_memory_write_position, or one of the two values above.
    u32 last_memory_write_method{DeferredWrite};
    u64 previous_memory_write_position{};
    u32 previous_memory_write_method{DeferredWrite};
    /// The 3D memory-writing methods alone (NoteMethodWrite, RangeLastWrite), without the
    /// deferred work, the all-writing engines and the barriers last_memory_write_position also
    /// counts (QueuedGuestWriters).
    u64 last_method_write_position{};
    u64 previous_method_write_position{};
    /// The newest ranged write RangeLastWrite dropped from its full ring (fable_qmd_wait_brief
    /// 4.1). The ring scans (QueuedGuestWriters; MemoryWriterTarget through
    /// last_memory_write_position) no longer see its range, so for them it may write anything.
    /// Its bytes were stamped when it was noted, so WriteStamp needs it only as a floor of its
    /// own, which EXP126 = 1 leaves out.
    u64 ranged_overflow_position{};
    u64 session{};
    /// FRITCHII_TEST_SPLIT_RESTART: when this session started.
    std::chrono::steady_clock::time_point session_started{};
    /// EnableWriteStamps' tables: a slot keeps its newest stamp, the tag that made it, and the
    /// newest stamp of any other tag before it.
    struct StampSlot {
        u64 last;
        u64 last_tag;
        u64 before;
    };
    static constexpr size_t BlockStamps = size_t{1} << 16;
    static constexpr size_t PageStamps = size_t{1} << 14;
    std::unique_ptr<StampSlot[]> block_stamps;
    std::unique_ptr<StampSlot[]> page_stamps;
    bool write_stamps{};
    u64 write_stamp_tag{};
    u64 write_stamp_tags{};
    u64 write_stamp_floor{};
    u64 unknown_write_position{};
    void StampWrite(u64 begin, u64 end, bool gpu_address);
    [[nodiscard]] u64 StampOf(u64 begin, u64 end, bool gpu_address, u64 skip_tag) const;
    /// Writes with a known range (RangeLastWrite) the back may not have run yet, oldest first.
    struct RangedWrite {
        u64 position;
        GPUVAddr begin;
        GPUVAddr end;
        u32 method;
        size_t space;           ///< Address space of `data`; 0 when the words are not known
        std::vector<u32> data;  ///< The words written over [begin, end), when known
    };
    /// Ranged writes kept before the oldest counts for every read again (EDEN_NX2_RANGED_MAX).
    static size_t MaxRangedWrites();
    /// ranged_writes' storage, in send order as the deque it replaces: a ring of slots whose word
    /// vectors are cleared when a slot is reused, never freed (the deque destroyed one vector per
    /// upload and built a new one: ~35 ms/s of the front in vector and deque allocations,
    /// fable_answer_regression_2 4.3). Sized to MaxRangedWrites() (RangeLastWrite never holds
    /// more), grown like a deque past it. A slot's vector over 64 KiB is freed on reuse.
    class RangedRing {
    public:
        [[nodiscard]] bool empty() const {
            return count == 0;
        }
        [[nodiscard]] size_t size() const {
            return count;
        }
        /// Oldest first: 0 is the oldest, size() - 1 the newest.
        [[nodiscard]] RangedWrite& operator[](size_t index) {
            return slots[(head + index) & mask];
        }
        [[nodiscard]] const RangedWrite& operator[](size_t index) const {
            return slots[(head + index) & mask];
        }
        [[nodiscard]] RangedWrite& front() {
            return slots[head];
        }
        [[nodiscard]] RangedWrite& back() {
            return (*this)[count - 1];
        }
        [[nodiscard]] const RangedWrite& back() const {
            return (*this)[count - 1];
        }
        void pop_front() {
            head = (head + 1) & mask;
            --count;
        }
        /// As emplace_back(RangedWrite{position, begin, end, method, 0, {}}).
        RangedWrite& push_back(u64 position, GPUVAddr begin, GPUVAddr end, u32 method) {
            if (count == slots.size()) [[unlikely]] {
                Grow();
            }
            RangedWrite& write = slots[(head + count) & mask];
            ++count;
            write.position = position;
            write.begin = begin;
            write.end = end;
            write.method = method;
            write.space = 0;
            if (write.data.capacity() > MaxKeptWords) [[unlikely]] {
                std::vector<u32>{}.swap(write.data);
            } else {
                write.data.clear();
            }
            return write;
        }
        /// Empties the ring; the slots keep their vectors.
        void clear() {
            head = 0;
            count = 0;
        }

    private:
        static constexpr size_t MaxKeptWords = 16384;
        void Grow();
        std::vector<RangedWrite> slots;
        size_t mask{};
        size_t head{};
        size_t count{};
    };
    RangedRing ranged_writes;
    struct QueuedRelease {
        u64 position;
        u64 device;
        u32 payload;
        ReleaseKind kind;
    };
    std::array<QueuedRelease, 64> queued_releases{};
    /// Drops consumed ranged writes; returns the position a memory read of [begin, end) waits
    /// for and the method behind it.
    u64 MemoryWriterTarget(u64 done, GPUVAddr begin, GPUVAddr end, u32& target_method);
    u64 last_function_position{};
    SplitWriteTracker writes;
    /// NoteWrite's last translation: memory manager, GPU page, device page, the function
    /// position it was made at (a later function may remap).
    const Tegra::MemoryManager* note_memory{};
    GPUVAddr note_page{~GPUVAddr{0}};
    u64 note_device{};
    u64 note_function{};
    /// EXP126 not 0 (review fix 2 of fable_qmd_wait_brief): ranged writes are stamped by device
    /// address too, so a read of the same memory through another GPU address finds them (the
    /// GPU stamp alone did not; the overflow floor hid that once the ring was full). Since
    /// `ranged_device_since` (the switch turned on): the writes sent before were stamped by GPU
    /// address only, and count for every address with the overflow left out until they ran.
    bool ranged_device_stamps{};
    u64 ranged_device_since{};
    /// StampRangedDevice's last translation, as NoteWrite's.
    const Tegra::MemoryManager* stamp_memory{};
    GPUVAddr stamp_page{~GPUVAddr{0}};
    u64 stamp_device{};
    u64 stamp_function{};
    bool stamp_mapped{};
    /// Whether ranged writes are stamped by device address now (EXP126 not 0), noting a turn on.
    bool RangedDeviceStamps();
    void StampRangedDevice(const Tegra::MemoryManager& memory, GPUVAddr begin, GPUVAddr end);
    /// FRITCHII_STAMP_READ_PAGES (live, default 0; fable_b 21.2 cut 2): RangeLastWrite stamps
    /// (by GPU and device address) only the device pages a stamp reader (WriteStamp,
    /// WriteStampWhy, BytesStamp) ever asked about; a page first asked about takes the newest
    /// write whose stamps were skipped as its add floor, a stamp over every byte of it. 1 = dry:
    /// everything stamped as before, the pages counted. Front only, as the stamp tables.
    /// Device page -> add floor. Cleared when it grows past StampReadPagesMax (a page asked
    /// about again then takes the newest skipped write: still exact).
    mutable std::unordered_map<u64, u64> stamp_read_pages;
    static constexpr size_t StampReadPagesMax = size_t{1} << 16;
    /// The last page RangeLastWrite looked up and its answer (a reader adding a page clears it).
    mutable u64 stamp_read_cache_page{~u64{0}};
    bool stamp_read_cache_hit{};
    /// Bumped whenever stamp_read_pages changes (a page added, the table cleared, a reset): the
    /// FRITCHII_FRONT_TRIM_RANGE table below keeps an answer only while it stays the same.
    mutable u64 stamp_read_gen{1};
    /// FRITCHII_FRONT_TRIM_RANGE (live, default 0; helper Z): StampRangedRead's per-page answer
    /// (the GPU page's translation and whether a reader asked about its device page) kept per GPU
    /// page in a small direct-mapped table. The one-entry caches above miss whenever two pages
    /// take turns (constant buffer words and report writes do, ~2 million ranged writes a second
    /// at the market); an entry answers only for the same memory manager, page and queued
    /// function position as the one-entry cache's key, with the map serial and stamp_read_gen
    /// unchanged since a stable translation filled it, so it gives what a fresh lookup gives.
    /// Also RangeLastWrite's two environment switches come from `ranged_env` instead of
    /// function statics (a thread-safe static guard per call). 1 = on, 2 = verify (the old
    /// lookup runs and answers, differences counted, must be 0), 3 = dry (counted, the old
    /// lookup and the table timed 1 in 64).
    struct TrimPage {
        const Tegra::MemoryManager* memory;
        GPUVAddr page;
        u64 function; ///< last_function_position at the fill
        u64 serial;   ///< the memory manager's map serial of the stable translation
        u64 read_gen; ///< stamp_read_gen at the fill
        u64 device;   ///< The device address of the page (mapped only)
        bool mapped;
        bool read_hit; ///< A reader asked about the device page (mapped only)
    };
    static constexpr size_t TrimPages = 64;
    std::array<TrimPage, TrimPages> trim_pages{};
    /// RangeLastWrite's EDEN_NX2_RANGED_WAITS (bit 0) and EDEN_NX2_FORWARD_READS (bit 1), from
    /// the same statics the old path reads; -1 until the first call with the trim on.
    int ranged_env{-1};
    struct TrimRangeStats {
        u64 calls;    ///< RangeLastWrite calls with the trim on (the log cadence)
        u64 lookups;  ///< pages looked up in the table
        u64 answered; ///< found there (1: taken; 2 and 3: would have been)
        u64 fills;    ///< entries filled from a stable translation
        u64 checked;  ///< 2: answers compared with the old lookup
        u64 differ;   ///< 2: of them, different (must be 0)
        u64 timed;    ///< 3: answered lookups timed
        u64 old_ticks;
        u64 table_ticks;
    };
    TrimRangeStats trim_range{};
    TrimRangeStats trim_range_logged{};
    u64 trim_range_window_ns{};
    u64 trim_range_window_ticks{};
    u64 trim_range_differ_total{};
    /// The trim's switch value (FRITCHII_FRONT_TRIM_RANGE).
    [[nodiscard]] static int FrontTrimRangeMode();
    /// RangeLastWrite's environment switches as above (statics shared by both paths).
    [[nodiscard]] static int RangedEnvSwitches();
    void LogTrimRange(int mode);
    /// The newest ranged write whose stamps were skipped (0 = none this session).
    u64 stamp_skip_floor{};
    struct StampReadStats {
        u64 writes;        ///< RangeLastWrite calls with stamps on and the switch on
        u64 pages_read;    ///< their pages on a page asked about: stamped
        u64 pages_skipped; ///< on a page never asked about: skipped (1: would be)
        u64 whole;         ///< stamped as before (over 4 KiB, translation unsure, no memory)
        u64 added;         ///< pages a reader asked about for the first time
        u64 floors;        ///< reader answers raised by an add floor
        u64 clears;        ///< the page table cleared at its limit
        u64 ticks;
        std::chrono::steady_clock::time_point since;
    };
    mutable StampReadStats stamp_read_stats{};
    /// The switch's value (FRITCHII_STAMP_READ_PAGES).
    [[nodiscard]] static int StampReadPagesMode();
    /// RangeLastWrite's stamps with the switch on: per page, stamped where a reader asked about
    /// it, skipped (2) or only counted (1) where not. False: stamp as before (the caller does).
    bool StampRangedRead(const Tegra::MemoryManager& memory, GPUVAddr begin, GPUVAddr end,
                         bool skip, int trim = 0);
    /// Whether a reader asked about device page `page` (cached for runs of one page).
    bool StampPageRead(u64 page);
    /// A stamp reader of device bytes [device, device + size): marks their pages asked about and
    /// returns their add floors (0 with the switch off and nothing ever skipped).
    [[nodiscard]] u64 StampReadFloor(u64 device, u64 size) const;
    void MaybeLogStampRead(int mode);
    /// What stands for the ranged overflow when WriteStamp leaves it out: the writes not
    /// stamped by device address, or the overflow itself while device stamps are off.
    [[nodiscard]] u64 FloorlessOverflowTerm() const {
        return ranged_device_stamps ? ranged_device_since : ranged_overflow_position;
    }
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    /// EXP39 & 1, read at each Publish: Commit publishes a few words to a back that waits.
    bool publish_waiting{};
    Tegra::Engines::EngineInterface* front_engine{};
    /// The open Pairs block's head (PairWrite), and its count.
    u64* pairs_header{};
    u32 pairs_count{};
    /// The open StateDelta block's head (DeltaWrite), and its count.
    u64* delta_header{};
    u32 delta_count{};
    /// FRITCHII_CB_WORDS_BLOCK: the open CbWords block's head (CbWord), its count, whether it
    /// holds the verify word, and the position right after its last word. Mutable: SentPosition
    /// ends the block.
    mutable u64* cb_header{};
    u32 cb_count{};
    bool cb_verify{};
    u64 cb_end{};
    /// Front counts for ReportCbWords: blocks opened, words sent in blocks, blocks ended at
    /// MaxCbWords, blocks found with a commit after their end (must be 0).
    struct CbFrontStats {
        u64 blocks;
        u64 words;
        u64 full;
        u64 broken;
    };
    CbFrontStats cb_stats{};
    /// RangeLastWrite's overflows (floor_stats.overflows is reset by its own report).
    u64 ranged_overflow_count{};
    std::chrono::steady_clock::time_point cb_since{};
    CbFrontStats cb_reported{};
    u64 cb_reported_overflows{};
    /// FRITCHII_CB_MERGE=1 (dry, helper DE): when a CbWords block opens, what sits in the ring
    /// since the previous block's last word (NoteCbMergeGap). Front only, reset by ReportCbWords.
    struct CbMergeFront {
        u64 opens;     ///< blocks opened with a previous block on the same engine
        u64 offset;    ///< only constant buffer offset writes between: one entry could hold both
        u64 rebind;    ///< offset plus address or size writes between
        u64 publish;   ///< nothing between (a publish or a sent-position read ended the block)
        u64 blocked;   ///< anything else between
        u64 gap_pairs; ///< offset/address/size writes in the joinable gaps
    };
    CbMergeFront cb_merge{};
    /// The engine of the last CbWords block (the gap test needs the same engine).
    Tegra::Engines::EngineInterface* cb_last_engine{};
    /// The back's sampled timing for the same report (one writer, read relaxed): blocks by
    /// size (1, 2, 3-4, 5+ words) and the Pairs entries that hold only constant buffer offset,
    /// address or size writes, sampled 1 in 16, in time stamp counter ticks.
    struct CbMergeBack {
        std::array<std::atomic<u64>, 4> block_samples;
        std::array<std::atomic<u64>, 4> block_ticks;
        std::atomic<u64> gap_entries;
        std::atomic<u64> gap_samples;
        std::atomic<u64> gap_ticks;
    };
    alignas(128) CbMergeBack cb_merge_back{};
    u32 cb_merge_sample_left{}; ///< back only
    /// ReportCbWords' last time stamp counter and clock pair (ticks to ns) and back totals.
    u64 cb_merge_ticks_at{};
    std::chrono::steady_clock::time_point cb_merge_clock_at{};
    std::array<u64, 4> cb_merge_last_samples{};
    std::array<u64, 4> cb_merge_last_ticks{};
    u64 cb_merge_last_gap_entries{};
    u64 cb_merge_last_gap_samples{};
    u64 cb_merge_last_gap_ticks{};
    void NoteCbMergeGap(Tegra::Engines::EngineInterface* engine);
    /// The dry counters and timers run at 1 (the count) and 2 (verify), not at 3.
    [[nodiscard]] static bool CbMergeOn() {
        const int mode = CbMergeMode();
        return mode == 1 || mode == 2;
    }
    /// Back, a CbWords block with CbJoined: runs of constant buffer words as ExecuteCbWords
    /// runs them, joined register writes as their Pairs entries, mode-2 address marks checked.
    void ExecuteCbJoined(const u64* words, u32 count);
    /// FRITCHII_CB_MERGE=2: the front's address of the next joined run (its first word), as
    /// a marker word (Sink, method CbMarkMethod) and the address word, before that word.
    static constexpr u32 CbMarkMethod = 0xffff;
    bool cb_merge_mark_next{}; ///< front: the next appended word starts a joined run
    /// FRITCHII_CB_MERGE >= 2 counts (front: joined writes, refused, marks; back: blocks, writes,
    /// runs checked, differ (must be 0), bad words (must be 0)).
    struct CbJoinFront {
        u64 joined;
        u64 refused;
        u64 marks;
        u64 marks_skipped;
    };
    CbJoinFront cb_join{};
    struct CbJoinBack {
        std::atomic<u64> blocks;
        std::atomic<u64> writes;
        std::atomic<u64> checked;
        std::atomic<u64> differ;
        std::atomic<u64> bad;
    };
    alignas(128) CbJoinBack cb_join_back{};
    u64 cb_join_last_blocks{};
    u64 cb_join_last_writes{};
    /// The 3D engine the back last ran a CbWords block on (the dry timer's Pairs test).
    Tegra::Engines::EngineInterface* cb_merge_back_3d{};
    /// Held-back front work (SetPendingFlush).
    PendingFlush pending_flush{};
    void* pending_flush_argument{};
    /// Instance folding (FoldInstance): instances counted and not sent yet, and who sends them.
    u32 fold_count{};
    void (*fold_flush)(void*, u32){};
    void* fold_context{};
    static void FlushFoldThunk(void* split) {
        static_cast<GpuSplit*>(split)->FlushFold();
    }
    std::array<u64, static_cast<size_t>(Batched::Count)> batched{};
    std::chrono::steady_clock::time_point batched_since{};
    u32 publishes{};
    void ReportBatched();
    /// The QMD wait floor counters (fable_qmd_wait_brief piece 1), front only, logged every 5 s
    /// as "QMD wait floor" with EDEN_NX2_GPU_PROFILE or EXP126 on (ReportQmdFloor).
    struct FloorStats {
        u64 overflows;       ///< RangeLastWrite dropped the oldest of a full ring
        u64 qmd_reads;       ///< NoteQmdRead
        u64 reads_floor_up;  ///< ... with the overflow queued
        u64 floor_words;     ///< ... their ring words from the overflow to the write position
        u64 floor_words_max;
        std::array<u64, static_cast<size_t>(QmdPatchOutcome::Count)> patches;
        std::array<u64, static_cast<size_t>(StampTerm::Count)> patch_refused;
        u64 patch_overflow_alone; ///< Refused with nothing but the overflow queued over it
        /// Read waits by the term that set the target (the last: an evicted inline QMD).
        std::array<u64, static_cast<size_t>(StampTerm::Count) + 1> read_waits;
        std::array<u64, static_cast<size_t>(StampTerm::Count) + 1> read_wait_words;
        u64 read_overflow_alone; ///< Waits with nothing but the overflow queued over the QMD
        /// ExactWriterCheck's answers for block and page stamps: patches, reads.
        std::array<std::array<u64, 3>, 2> exact;
        u64 dma_waits;          ///< WaitForMemoryWriters(site, begin, end) that waited
        u64 dma_overflow_waits; ///< ... whose target is the ranged overflow
        u64 dma_overflow_words; ///< ... and the ring words they waited for
        /// EXP126 not 0: ranged writes stamped by device address, and those that could not be
        /// (a queued function may remap, sparse) and count for every address instead.
        u64 device_stamped;
        u64 device_floors;
        /// EXP126 = 2: reads compared (1 serving memory, an overlay, a shadow block), those that
        /// differed, trial overlays and blocks built, reads not compared, entries lost.
        std::array<u64, 3> trial_compared;
        u64 trial_differed;
        u64 trial_built;
        u64 trial_skipped;
        u64 trial_lost;
    };
    FloorStats floor_stats{};
    std::chrono::steady_clock::time_point floor_since{};
    void ReportQmdFloor();

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};
    /// The position of the entry being run (dispatch writes record it).
    u64 back_entry_position{};
    /// FRITCHII_CB_WORDS_BLOCK, written by the back only (a plain load and store), read by the
    /// front's ReportCbWords: blocks and words run; blocks whose words were not all 3D constant
    /// buffer data Method words (must be 0); with =2 the addresses compared and those that
    /// differed (must be 0), the blocks applied both ways, those that differed (must be 0), and
    /// those not applied twice (one word, a deferred draw waiting, the gather off).
    struct CbBackStats {
        std::atomic<u64> blocks;
        std::atomic<u64> words;
        std::atomic<u64> bad;
        std::atomic<u64> address_checked;
        std::atomic<u64> address_differ;
        std::atomic<u64> applied_twice;
        std::atomic<u64> apply_differ;
        std::atomic<u64> apply_skipped;
    };
    alignas(128) CbBackStats cb_back{};
    /// EXP10: the front asked the back to record dispatch writes (NoteDispatchWriter).
    bool back_dispatch_records{};
    /// The dispatch being run: the rasterizer recorded its writes, it recorded an unbounded one.
    bool dispatch_confirmed{};
    bool dispatch_unbounded{};
    static inline thread_local bool recording_dispatch{};

    static GpuSplit instance;
    static inline thread_local bool on_back_thread{};
    /// The back is in its wait for work (the idle hook's idle call, not a RequestIdleHook mid-batch).
    static inline thread_local bool in_idle_wait{};
    void (*idle_hook)(void*){};
    void* idle_argument{};
    FenceQuietCheck fence_quiet_check{};
    void* fence_quiet_argument{};
    /// Back: runs fence_quiet_check at `position` (everything published executed).
    void CheckFencesQuiet(u64 position);
    /// Back: forgets the last check (the idle hook submits, which may change the caches).
    void ForgetFencesQuiet() {
        fence_quiet_position.store(~u64{0}, std::memory_order_relaxed);
        fence_busy_position.store(~u64{0}, std::memory_order_relaxed);
    }
    /// Written by the back before `consumed`: the position of its last check that found nothing
    /// to release (with the epoch it saw), or that found work; ~0 for none.
    alignas(128) std::atomic<u64> fence_quiet_position{~u64{0}};
    std::atomic<u64> fence_busy_position{~u64{0}};
    std::atomic<u64> fence_quiet_epoch{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> back_running{};

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    /// One follower: R's position (RaisePlayed), on its own line (R and E both raise it).
    alignas(128) std::atomic<u64> played{};
    /// FRITCHII_ONEF=1 (set at MaybeStart, before the back exists): the front waits on `played`.
    bool two_positions{};
    FrontWait front_wait; ///< FrontWaitNow
    /// FRITCHII_MEMSTAGE_THREAD=2: the stage's read position (the back's limit) and its level run.
    alignas(128) std::atomic<u64> staged{};
    std::atomic<u64> lent_from{};
    std::atomic<u64> lent_until{};
    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    /// The back's position when it last ran out of work, stored before it waits for more: while
    /// it equals published_local, the back waits for the words the front holds (Commit, EXP39 &
    /// 1). The back writes it once per wait, the front reads it at every Commit. ~0: none since
    /// the split started (positions start over at 0).
    alignas(128) std::atomic<u64> back_idle_at{~u64{0}};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    /// The front sleeps in Backoff (ported: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    std::jthread back_thread;
};

} // namespace VideoCommon
