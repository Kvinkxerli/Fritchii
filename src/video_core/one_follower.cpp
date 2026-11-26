// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// One follower, part (a): the render thread R, the packet ring E -> R, lent packets, the ahead
// window and the owner asserts (one_follower.h has the design; the counters and log lines are
// part (d)'s, one_follower_stats.h).
//
// The ring: one writer (E), one reader (R), variable-size packets, never split at the ring's end
// (a pad packet fills the tail). E commits a packet by raising `sent` (release); R frees its bytes
// by raising `freed` and finishes it by raising `played` to its number. E waits for `played` at a
// lend (the park) and for `freed` or `played` at a full ring or window.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "common/dkb_cpu_trace.h"
#include "common/lowend_sim.h"
#include "common/fs/path_util.h"
#include "common/spin_recursive_mutex.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/fritchii_e_split.h"
#include "video_core/fritchii_occl_flag.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/nx2.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/one_follower.h"
#include "video_core/nx2_profile.h"
#include "video_core/one_follower_stats.h"
#include "video_core/page_filter.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif
#ifdef _WIN32
#include <windows.h> // RtlCaptureStackBackTrace (R's waits for E's lock)
#endif

namespace VideoCommon::OneFollower {

namespace {

void Pause() {
    Common::LowEndSim::CpuPauseNote();
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

constexpr u64 RingBytes = u64{2} << 20;
/// Ring A positions and push times of packets by number (EngineConsumed's floor, LentRan's reach).
/// Packets are at least sizeof(PacketHeader) bytes, so fewer than this many are ever in flight.
constexpr u64 PositionSlots = RingBytes / sizeof(PacketHeader) + 1;
/// A pad packet's kind (not in PacketKind: never pushed, only skipped).
constexpr u8 PadKind = 0xff;
/// Spins before a thread sleeps on its wake word (about as long as the back's).
constexpr u32 SpinLimit = 8192;

struct Impl {
    bool on{};   ///< FRITCHII_ONEF=1 (or FRITCHII_ONEF_LEAN=1) at boot (sticks for the process)
    bool lean{}; ///< FRITCHII_ONEF_LEAN=1 at boot: step 4 from boot, no compares
    bool read{}; ///< ReadSwitches ran
    bool pin{};  ///< FRITCHII_ONEF_PIN
    Core::System* system{};
    std::jthread thread;

    // Live switches, refreshed by Refresh when the switch generation moved.
    std::mutex refresh_mutex;
    std::atomic<u32> seen_generation{~u32{0}};
    std::atomic<u8> step{};
    std::atomic<u8> asked{}; ///< the step the switches ask for before nesting
    std::atomic<u32> unlend_mask{};
    std::atomic<int> verify_every{DefaultVerifyEvery};
    std::atomic<int> ahead{DefaultAhead};
    int logged_view{-1};
    int logged_build{-1};
    int logged_unlend{-1};

    std::unique_ptr<u8[]> ring;
    std::unique_ptr<u64[]> positions; ///< ring A position of packet n at n % PositionSlots
    std::unique_ptr<u64[]> push_ns;   ///< E's push time of packet n (with the GPU profile)

    // E's side (its own cache line).
    alignas(64) u64 write{}; ///< bytes written
    u64 pushed{};            ///< packets pushed
    u64 draw_seq{};
    std::atomic<u64> sent{};          ///< bytes committed
    std::atomic<u64> sent_packets{};  ///< packets committed
    std::atomic<u64> intended{};      ///< the consumed position E last stored (EngineConsumed)

    // R's side.
    alignas(64) u64 read_at{};
    std::atomic<u64> freed{};
    std::atomic<u64> played{};

    alignas(64) std::atomic<u32> r_wake{};
    std::atomic<bool> r_sleeping{};
    alignas(64) std::atomic<u32> e_wake{};
    std::atomic<bool> e_waiting{};

    std::exception_ptr lent_error; ///< thrown by a lent call on R, rethrown on E
    std::exception_ptr op_error;   ///< thrown by a render op on R, rethrown at E's next lend

    // The pricing counter of fable_b 11.5 (E writes; the frame end logs on R).
    std::array<std::atomic<u64>, static_cast<size_t>(QueryKind::Count)> query_count{};
    std::array<std::atomic<u64>, static_cast<size_t>(QueryKind::Count)> query_parked_ns{};
    std::array<u64, static_cast<size_t>(QueryKind::Count)> query_count_logged{};
    std::array<u64, static_cast<size_t>(QueryKind::Count)> query_ns_logged{};
    u64 query_log_ns{};
    std::atomic<bool> query_ops_logged{};
    std::atomic<u64> guest_write_op{}; ///< GuestWriteOpMark
    // FRITCHII_ONEF_CARRY_OPS (E counts; the frame-end line reads them): ops carried, the
    // packets that carried them by kind (draw, lent, op, delta), the ones sent alone by why
    // (E's consumed store, a drain, no room).
    std::atomic<u64> carried_ops{};
    std::array<std::atomic<u64>, 4> carriers{};
    std::array<std::atomic<u64>, 3> carried_alone{};
    u64 carried_ops_logged{};
    std::array<u64, 4> carriers_logged{};
    std::array<u64, 3> carried_alone_logged{};
    /// E only: the packet number of that newest op (WaitGuestWriteOps).
    u64 guest_write_number{};
    /// Lends E made holding a watched lock (handed to R for the park), the op packets' line.
    std::atomic<u64> lends_holding_lock{};
    u64 lends_holding_lock_logged{};

    // FRITCHII_ONEF_INVAL_E (item 2): E's page write stamps, the next packet number (low 32
    // bits) at each write of an invalidation E ran itself, by device page (2^22 pages = 16 GiB
    // direct, higher pages share slots: a shared slot only answers "ahead" more often).
    std::unique_ptr<std::atomic<u32>[]> stamps;
    std::atomic<bool> stamps_on{};
    std::atomic<bool> inval_refused_logged{};
    std::array<std::atomic<u64>, static_cast<size_t>(InvalKind::Count)> inval_count{};
    std::array<u64, static_cast<size_t>(InvalKind::Count)> inval_logged{};
    std::atomic<u64> ahead_checks{};
    u64 ahead_checks_logged{};

    // fable_b 12.3's dry prices (E counts; the frame-end line reads them).
    std::array<std::atomic<u64>, static_cast<size_t>(OpCount::Count)> op_count{};
    std::atomic<u64> op_bytes{};
    std::array<u64, static_cast<size_t>(OpCount::Count)> op_logged{};
    u64 op_bytes_logged{};
    std::array<std::atomic<u64>, 3> mirror_count{}; ///< compared, E set R not, R set E not
    std::atomic<bool> post_fence_stopped{};
    /// FRITCHII_ONEF_MIRROR_STOP=2 (R only): R skipped a draw with the samples count on since
    /// its last samples report; and how many "E set, R not" it explained so far.
    bool skipped_samples_draw{};
    std::atomic<u64> mirror_explained{};
    u64 mirror_explained_logged{};
    /// FRITCHII_ONEF_SIGNAL_ON_R (E counts): payload writes and sync points R made for E.
    std::atomic<u64> signals_on_r{};
    u64 signals_on_r_logged{};
    // FRITCHII_ONEF_OP_DISPATCH (E counts; the frame-end line reads them).
    std::array<std::atomic<u64>, static_cast<size_t>(DispatchCount::Count)> dispatch_count{};
    std::array<u64, static_cast<size_t>(DispatchCount::Count)> dispatch_logged{};
    std::atomic<bool> post_dispatch_stopped{};
    std::atomic<bool> dispatch_ranges_stopped{};
    /// NoteFrontReadPage's pages (made once, never freed while the process runs: R reads it).
    std::atomic<PageFilter*> front_read_pages{};
    std::atomic<bool> dispatch_refused_logged{};
    u64 lookup_checks{};
    u64 landing_max{}; ///< E: the newest packet number 13.0 assigned report pages to
    std::array<std::atomic<u64>, static_cast<size_t>(IdleDecision::Count)> idle_decisions{};
    std::array<u64, static_cast<size_t>(IdleDecision::Count)> idle_decisions_logged{};
    std::array<std::atomic<u64>, static_cast<size_t>(FenceDecision::Count)> fence_decisions{};
    std::array<u64, static_cast<size_t>(FenceDecision::Count)> fence_decisions_logged{};
    std::atomic<u64> sync_target_superset{};
    u64 sync_target_superset_logged{};
    std::atomic<bool> unlent_reset_stopped{}; ///< StopUnlentReset
    std::atomic<bool> unlent_values_stopped{}; ///< StopUnlentValues
    std::atomic<bool> bind_seed_stopped{};     ///< StopBindSeed
    // fable_b 13.0 (E only): pages with a report a posted fence lands, by that fence's packet
    // number; and the report pages since the last fence.
    std::unordered_map<u64, u64> landing_pages;
    std::vector<u64> landing_unassigned;
    std::atomic<u64> landing_waits{};
    u64 landing_waits_logged{};
    std::array<u64, 3> mirror_logged{};
    std::atomic<u64> fences_lent{};
    std::atomic<u64> fences_with_flush{};
    std::atomic<u64> gathers_lent{};
    std::atomic<u64> gather_ranges{};
    std::array<u64, 4> posting_logged{};

    // The hang watchdog's view: written relaxed by E and R (no clock), read once a second.
    std::jthread watchdog;
    std::atomic<u8> e_wait{};         ///< EWait: what E waits for
    std::atomic<u64> e_wait_seq{};    ///< E's waits begun (one that never ends keeps it still)
    std::atomic<u8> e_wait_op{};      ///< the lend's LentOp (EWait::Park)
    std::atomic<u64> e_wait_number{}; ///< the packet E waits for (park, drain) or pushes (room)
    std::atomic<u64> r_packet{};      ///< the packet R plays, 0 between packets
    std::atomic<u8> r_kind{};         ///< its PacketKind
    std::atomic<u8> r_op{};           ///< its LentOp, for a lent packet
    std::atomic<u64> r_progress{};    ///< packets and pads R finished
    std::atomic<u64> rescued{};       ///< wakes the watchdog gave
    std::atomic<std::thread::id> e_thread{};
    std::atomic<std::thread::id> e2_thread{}; ///< FRITCHII_ESPLIT: E2, E's second thread
    std::atomic<std::thread::id> r_thread{};
    std::array<std::pair<const char*, const Common::SpinRecursiveMutex*>, 4> locks{};
    std::atomic<u32> lock_count{};

    std::atomic<Maxwell3D*> engine{};
    std::atomic<bool> unlend_disabled{}; ///< DisableUnlend: step 4 off for the session

    // fable_b 19 step 2's dry counter (E counts at each transient lend; LogQueryKinds reads).
    std::array<std::array<std::atomic<u64>, Stats::KindSlots>,
               static_cast<size_t>(TransientTest::Count)>
        transient{};
    std::array<std::array<u64, Stats::KindSlots>, static_cast<size_t>(TransientTest::Count)>
        transient_logged{};
};

Impl g;

using Detail::t_render;
thread_local u32 t_lent_depth{};
/// R: inside PlayDraw of an unlent draw packet.
thread_local bool t_playing_draw{};
/// E: the kind of the query being lent or pushed (SetQueryKind), consumed by it.
thread_local QueryKind t_query_kind{QueryKind::Count};
/// fable_b 14.4 cut 1: the hot-path counters (hand-overs, behind, lends by site, query kinds,
/// invalidations) run unless LEAN without the GPU profile.
bool CountersOn() noexcept {
    return !g.lean || Nx2Profile::Enabled();
}
/// FRITCHII_ONEF_CARRY_OPS (live, default 0; fable_b 22.3 step 2): PushOp's `carry` ops ride in
/// the next packet.
bool CarryOpsOn() noexcept {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_carry_ops,
                                            "FRITCHII_ONEF_CARRY_OPS") != 0;
}

/// E: SetLendSite's name for the next Lend, consumed by it.
thread_local const char* t_lend_site{};

// FRITCHII_ONEF_TRACE (boot; fable_b 15.5): one record per lend and per posted op, by packet
// number in a ring. E writes its half (site, op, posted, its arrival, its work since its last
// record, its resume), R its own (the call's start and end, its busy time on other packets since
// its last record), GPUWatch writes the finished ones to <log>/onef_lend_trace.bin every second
// (and the site names to onef_lend_trace_sites.txt). lend_replay.py reads them.
struct TraceRecord {
    u64 number;   ///< the packet number (0: no record in this slot yet)
    u64 site;     ///< the site name's address (onef_lend_trace_sites.txt), 0 = none
    u64 e_arrive; ///< E at the push (ns, Nx2Hitch::NowNs)
    u64 r_start;  ///< R starts the packet
    u64 r_end;    ///< R ends it
    u64 e_resume; ///< E released (lent), 0 for a posted op
    u64 e_work;   ///< E's time since its previous record ended (its resume or post)
    u64 r_busy;   ///< R's time on other packets since its previous record
    u8 op;        ///< LentOp
    u8 posted;    ///< 1: a posted op (E did not wait)
    u8 pad[6];
};
static_assert(sizeof(TraceRecord) == 72);
constexpr u64 TraceSlots = u64{1} << 20;
bool g_trace{};
std::unique_ptr<TraceRecord[]> g_trace_ring;
u64 g_trace_written{}; ///< GPUWatch: the next packet number to write
u64 g_trace_records{};
u64 g_trace_lost{};
std::ofstream g_trace_file;
std::mutex g_trace_sites_mutex;
std::vector<const char*> g_trace_sites;
bool g_trace_sites_dirty{};
/// E: the end of its previous record (resume or post).
thread_local u64 t_e_last{};
/// R: its time on other packets since its previous record.
thread_local u64 t_r_busy{};

/// E: a record's E half at the push (before the packet can be played: R writes other fields).
void TraceArrive(u64 number, const char* site, LentOp op, bool posted, u64 now) {
    TraceRecord& record = g_trace_ring[number % TraceSlots];
    record.site = reinterpret_cast<u64>(site);
    record.op = static_cast<u8>(op);
    record.posted = posted ? 1 : 0;
    record.e_arrive = now;
    record.e_work = t_e_last != 0 && now > t_e_last ? now - t_e_last : 0;
    record.e_resume = 0;
    t_e_last = now;
    if (site != nullptr) {
        std::scoped_lock lock{g_trace_sites_mutex};
        if (std::ranges::find(g_trace_sites, site) == g_trace_sites.end()) {
            g_trace_sites.push_back(site);
            g_trace_sites_dirty = true;
        }
    }
    // Last: the slot is this packet's now (GPUWatch tests it).
    std::atomic_ref<u64>{record.number}.store(number, std::memory_order_release);
}

/// GPUWatch: writes every finished record (both halves in), and the site names when new.
void TraceFlush(bool final) {
    if (!g_trace || !g_trace_file) {
        return;
    }
    // A record is finished once R played it and E pushed past it (its resume was written
    // before its next push); a margin covers a posted op's E half, written right after its push.
    const u64 played = g.played.load(std::memory_order_acquire);
    const u64 sent = g.sent_packets.load(std::memory_order_acquire);
    u64 safe = (std::min)(played, sent);
    safe = final ? safe : (safe > 64 ? safe - 64 : 0);
    if (g_trace_written == 0) {
        g_trace_written = safe > TraceSlots ? safe - TraceSlots + 1 : 1;
    }
    if (safe >= g_trace_written + TraceSlots) {
        g_trace_lost += safe - TraceSlots + 1 - g_trace_written;
        g_trace_written = safe - TraceSlots + 1;
    }
    for (; g_trace_written <= safe; ++g_trace_written) {
        const TraceRecord& record = g_trace_ring[g_trace_written % TraceSlots];
        if (std::atomic_ref<u64>{const_cast<u64&>(record.number)}.load(std::memory_order_acquire) !=
            g_trace_written) {
            continue; // a draw or delta packet: no record
        }
        g_trace_file.write(reinterpret_cast<const char*>(&record), sizeof(record));
        ++g_trace_records;
    }
    g_trace_file.flush();
    std::vector<const char*> sites;
    {
        std::scoped_lock lock{g_trace_sites_mutex};
        if (!std::exchange(g_trace_sites_dirty, false) && !final) {
            return;
        }
        sites = g_trace_sites;
    }
    std::ofstream names{Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) /
                            "onef_lend_trace_sites.txt",
                        std::ios::trunc};
    for (const char* const site : sites) {
        names << fmt::format("{:#x}\t{}\n", reinterpret_cast<u64>(site), site);
    }
    if (final) {
        LOG_INFO(HW_GPU, "One follower lend trace: {} records written, {} lost (ring overrun)",
                 g_trace_records, g_trace_lost);
    }
}
/// E's lends by call site (written on E; the 10 s line reads them on R in a frame-end lend,
/// E parked). Keyed by the name's pointer (one literal per site); the rest counted together.
struct LendSite {
    const char* name{};
    u8 op{};
    u64 count{};
    u64 ns{};
    u64 count_logged{};
    u64 ns_logged{};
};
constexpr size_t LendSiteSlots = 96;
std::array<LendSite, LendSiteSlots> g_lend_sites{};
LendSite g_lend_sites_rest{};
u64 g_lend_sites_log_ns{};
/// R: the number of the packet it plays (or played last): its place for the page write stamps.
thread_local u32 t_play_number{};
constexpr u64 StampSlots = u64{1} << 22;
/// The most a packet's spans can take (every register word, the flags, the draw, the programs,
/// the pools, the engine values and every binding), and an op packet's payload after them.
constexpr std::size_t MaxSpanBytes = sizeof(RegPair) * Maxwell3D::Regs::NUM_REGS +
                                     sizeof(FlagBits) + sizeof(DrawValues) + sizeof(Programs) +
                                     sizeof(Pools) + sizeof(EngineValues) + sizeof(BindsHeader) +
                                     sizeof(BindValue) * MaxBindValues + 64;
// 128 (2026-11-15): SEM_E's payload fence op (targets, note) is 96 bytes; at 64 it ran on E
// (PushOp's oversize fallback) and recorded R's Vulkan work from E: a driver crash at boot.
constexpr std::size_t MaxPayloadBytes = MaxOpPayloadBytes;

/// FRITCHII_ONEF_CARRY_OPS (fable_b 22.3 step 2): one op E keeps for the next packet, followed by
/// its payload padded to 8 bytes.
struct CarryRecord {
    void (*function)(void*);
    u32 size;     ///< payload bytes
    u32 reserved;
};
static_assert(sizeof(CarryRecord) == 16);
/// A carrying packet's last 8 bytes: the records' count and bytes (the trailer excluded).
struct CarryTrailer {
    u32 count;
    u32 bytes;
};
static_assert(sizeof(CarryTrailer) == 8);
/// The most E keeps before it sends them alone: 12 sample reports (16 + 56 bytes each).
constexpr std::size_t MaxCarryBytes = 1024;

/// E only: the ops kept for the next packet.
struct Carry {
    alignas(8) std::array<u8, MaxCarryBytes> bytes{};
    u32 used{};
    u32 count{};
    /// Ring A position of the oldest kept op: the carrying packet's floor (EngineConsumed's
    /// `positions`), so `played` never passes a kept op before R ran it.
    u64 floor{~u64{0}};
    /// Position + 1 of the newest kept op that lands in memory (PushOp's GuestWriteOpMark), 0 none.
    u64 lands{};
};
Carry g_carry;

/// E's packet buffer (fable_b 11.1 item 0): allocated once per thread, written at a cursor, never
/// resized or cleared on the hot path (the spans' bytes are written before they are sent).
u8* PackBuffer() {
    thread_local std::unique_ptr<u8[]> buffer;
    if (!buffer) [[unlikely]] {
        buffer.reset(new u8[sizeof(PacketHeader) + MaxSpanBytes + MaxPayloadBytes + MaxCarryBytes +
                            sizeof(CarryTrailer)]);
    }
    return buffer.get();
}

/// E: puts the kept ops after the packet's body (`body_bytes` after the header; the header's
/// position is set) and returns the body's new size. `floor` is what Commit records for this
/// packet's place: the oldest kept op's position when that is earlier.
std::size_t AppendCarried(PacketHeader& header, u8* buffer, std::size_t body_bytes, u64& floor) {
    floor = header.position;
    if (g_carry.count == 0) {
        return body_bytes;
    }
    u8* const at = buffer + sizeof(PacketHeader) + body_bytes;
    std::memcpy(at, g_carry.bytes.data(), g_carry.used);
    const CarryTrailer trailer{g_carry.count, g_carry.used};
    std::memcpy(at + g_carry.used, &trailer, sizeof(trailer));
    header.flags |= PacketCarried;
    floor = (std::min)(floor, g_carry.floor);
    const size_t kind = header.kind == PacketKind::Draw   ? 0
                        : header.kind == PacketKind::Lent ? 1
                        : header.kind == PacketKind::Op   ? 2
                                                          : 3;
    g.carriers[kind].store(g.carriers[kind].load(std::memory_order_relaxed) + 1,
                           std::memory_order_relaxed);
    g.carried_ops.store(g.carried_ops.load(std::memory_order_relaxed) + g_carry.count,
                        std::memory_order_relaxed);
    return body_bytes + g_carry.used + sizeof(trailer);
}

/// E, right after Commit of a packet AppendCarried filled (`number`, the carrying packet's): the
/// kept ops are R's now.
void CarriedSent(u64 number) {
    if (g_carry.count == 0) {
        return;
    }
    if (g_carry.lands != 0) {
        // As PushOp's own mark, before E's next `consumed` store (seq_cst there).
        g.guest_write_op.store(g_carry.lands, std::memory_order_release);
        // FRITCHII_E2C_ON_E's WaitGuestWriteOps waits for the packet that holds the newest
        // report op: a carried report rides in this one (as PushOp's own mark sets it).
        g.guest_write_number = number;
    }
    g_carry.used = 0;
    g_carry.count = 0;
    g_carry.floor = ~u64{0};
    g_carry.lands = 0;
}

/// R, at the start of a PacketCarried packet (`packet` = its header): the carried ops, in order.
void RunCarried(const PacketHeader& header, const u8* packet) {
    CarryTrailer trailer;
    std::memcpy(&trailer, packet + header.bytes - sizeof(trailer), sizeof(trailer));
    const u8* at = packet + header.bytes - sizeof(trailer) - trailer.bytes;
    for (u32 i = 0; i < trailer.count; ++i) {
        CarryRecord record;
        std::memcpy(&record, at, sizeof(record));
        at += sizeof(record);
        try {
            record.function(const_cast<u8*>(at));
        } catch (...) {
            if (!g.op_error) {
                g.op_error = std::current_exception();
            }
        }
        at += (record.size + 7) & ~u32{7};
    }
}
/// E: the draw the next Lend or PackSpans is for (SetNextDraw), consumed by them.
thread_local DrawValues t_next_draw{};
thread_local bool t_next_draw_set{};

/// Applies the nesting: each step needs the one before; a refused switch is logged once per value.
void Refresh(u32 generation) {
    std::scoped_lock lock{g.refresh_mutex};
    using namespace VideoCore::Nx2Debug;
    // FRITCHII_ONEF_LEAN: steps 2 and 3 are asked for by the lean mode itself (their compares
    // stay off: no PacketVerify, VERIFY_EVERY 0); step 4 by FRITCHII_ONEF_UNLEND (lean's 3).
    const int view = g.lean ? 1 : SwitchValue(g_onef_view, "FRITCHII_ONEF_VIEW");
    const int build = g.lean ? 1 : SwitchValue(g_onef_build, "FRITCHII_ONEF_BUILD");
    const int unlend = SwitchValue(g_onef_unlend, "FRITCHII_ONEF_UNLEND");
    g.verify_every.store(SwitchValue(g_onef_verify_every, "FRITCHII_ONEF_VERIFY_EVERY",
                                     DefaultVerifyEvery),
                         std::memory_order_relaxed);
    g.ahead.store(SwitchValue(g_onef_ahead, "FRITCHII_ONEF_AHEAD", DefaultAhead),
                  std::memory_order_relaxed);
    const u8 asked = !g.on ? 0 : unlend != 0 ? 4 : build != 0 ? 3 : view != 0 ? 2 : 1;
    StepLevel level = g.on ? StepLevel::Lent : StepLevel::Off;
    const auto refuse = [](int& logged, int value, const char* name, const char* needs) {
        if (logged != value) {
            logged = value;
            LOG_WARNING(HW_GPU, "{}={} needs {}; running 0", name, value, needs);
        }
    };
    if (view != 0) {
        if (level == StepLevel::Lent) {
            level = StepLevel::View;
        } else {
            refuse(g.logged_view, view, "FRITCHII_ONEF_VIEW", "FRITCHII_ONEF=1 at boot");
        }
    }
    if (build != 0) {
        if (level == StepLevel::View) {
            level = StepLevel::Build;
        } else {
            refuse(g.logged_build, build, "FRITCHII_ONEF_BUILD", "FRITCHII_ONEF_VIEW=1");
        }
    }
    u32 mask = 0;
    if (unlend != 0) {
        if (g.unlend_disabled.load(std::memory_order_relaxed)) {
            refuse(g.logged_unlend, unlend, "FRITCHII_ONEF_UNLEND",
                   "a session without an unlent draw R could not build (it was disabled)");
        } else if (level != StepLevel::Build) {
            refuse(g.logged_unlend, unlend, "FRITCHII_ONEF_UNLEND", "FRITCHII_ONEF_BUILD=1");
        } else if (GetDrawPlayer() == nullptr) {
            refuse(g.logged_unlend, unlend, "FRITCHII_ONEF_UNLEND", "R's draw player (part c)");
        } else {
            level = StepLevel::Unlend;
            mask = static_cast<u32>(unlend) & AllKinds;
        }
    }
    const auto previous = static_cast<StepLevel>(g.step.exchange(static_cast<u8>(level)));
    g.asked.store(asked, std::memory_order_relaxed);
    g.unlend_mask.store(mask, std::memory_order_relaxed);
    if (previous >= StepLevel::View && level < StepLevel::View) {
        // Writes stop being tracked: the view is seeded again when step 2 comes back.
        Detail::ViewTrackingStopped();
    }
    g_track_writes.store(level >= StepLevel::View, std::memory_order_relaxed);
    if (previous != level && g.on) {
        LOG_INFO(HW_GPU, "One follower: step {} (unlend mask {:#x}, verify every {}, ahead {})",
                 static_cast<int>(level), mask, g.verify_every.load(std::memory_order_relaxed),
                 g.ahead.load(std::memory_order_relaxed));
    }
    if (g.lean && level != StepLevel::Unlend && unlend != 0 &&
        !g.unlend_disabled.load(std::memory_order_relaxed) && GetDrawPlayer() == nullptr) {
        return; // LEAN before R's draw player exists (boot): ask again at the next Step()
    }
    g.seen_generation.store(generation, std::memory_order_release);
}

/// What E waits for (the watchdog's report).
enum class EWait : u8 { None, Park, Room, Drain };
constexpr std::array<const char*, 4> e_wait_names{"running", "parked in a lend",
                                                  "waiting for ring room or the window",
                                                  "draining R"};

/// E waits until `ready` holds; returns the nanoseconds waited (0 at once, or without the profile).
template <typename Ready>
u64 EngineWait(Ready&& ready, EWait kind = EWait::Room, u64 number = 0) {
    if (ready()) {
        return 0;
    }
    g.e_wait_number.store(number, std::memory_order_relaxed);
    g.e_wait.store(static_cast<u8>(kind), std::memory_order_relaxed);
    g.e_wait_seq.store(g.e_wait_seq.load(std::memory_order_relaxed) + 1,
                       std::memory_order_relaxed);
    SCOPE_EXIT {
        g.e_wait.store(static_cast<u8>(EWait::None), std::memory_order_relaxed);
    };
    const Common::LowEndSim::CpuWaitScope sim_wait;
    const bool timing = Stats::Timing();
    const u64 begin = timing ? Stats::Now() : 0;
    // FRITCHII_ONEF_E_SPIN (live): pauses before E sleeps (default SpinLimit, as before).
    const u32 spin_limit = static_cast<u32>((std::max)(
        0, VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_e_spin,
                                            "FRITCHII_ONEF_E_SPIN", static_cast<int>(SpinLimit))));
    for (u32 spins = 0;; ++spins) {
        if (ready()) {
            break;
        }
        if (spins < spin_limit) {
            Pause();
            continue;
        }
        g.e_waiting.store(true, std::memory_order_seq_cst);
        // The flag before `ready`'s loads (they are acquire loads): WakeEngine's fence pairs.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const u32 seen = g.e_wake.load(std::memory_order_seq_cst);
        if (!ready()) {
            g.e_wake.wait(seen, std::memory_order_seq_cst);
        }
        g.e_waiting.store(false, std::memory_order_relaxed);
    }
    return timing ? Stats::Now() - begin : 1;
}

// The wake handshakes are Dekker's: the sleeper stores its flag and then reads the condition; the
// waker stores the condition and then reads the flag. The waker's condition stores (`sent`,
// `freed`) are release stores, which x86 may hold in its store buffer past the flag's load
// (store-load reordering): both sides then read the other's old value, the sleeper sleeps, the
// waker sees no sleeper, and E (parked in a lend) and R (asleep with the lend in the ring) wait
// for each other for good. The full fence before the flag's load closes it.
void WakeRender() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (g.r_sleeping.load(std::memory_order_seq_cst)) {
        g.r_wake.fetch_add(1, std::memory_order_seq_cst);
        g.r_wake.notify_one();
    }
}

void WakeEngine() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (g.e_waiting.load(std::memory_order_seq_cst)) {
        g.e_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e_wake.notify_one();
    }
}

/// E: copies a finished packet (its header first, `bytes` in all) into the ring. `windowed`: an
/// unparked packet, held to FRITCHII_ONEF_AHEAD (a lent packet waits for its own play anyway).
void Commit(const u8* packet, u32 bytes, u64 number, u64 position, bool windowed) {
    u64 waited = 0;
    // A packet never wraps: the tail goes to a pad when it does not fit.
    const u64 offset = g.write % RingBytes;
    if (offset + bytes > RingBytes) {
        const u64 pad = RingBytes - offset;
        waited += EngineWait(
            [&] { return g.write + pad - g.freed.load(std::memory_order_acquire) <= RingBytes; },
            EWait::Room, number);
        PacketHeader header{};
        header.bytes = static_cast<u32>(pad);
        header.kind = static_cast<PacketKind>(PadKind);
        std::memcpy(g.ring.get() + offset, &header, sizeof(header));
        g.write += pad;
        g.sent.store(g.write, std::memory_order_release);
        WakeRender();
    }
    const int ahead = g.ahead.load(std::memory_order_relaxed);
    waited += EngineWait([&] {
        if (g.write + bytes - g.freed.load(std::memory_order_acquire) > RingBytes) {
            return false;
        }
        // The window: packets pushed before this one that R has not played.
        return !windowed || ahead <= 0 ||
               number - 1 - g.played.load(std::memory_order_acquire) < static_cast<u64>(ahead);
    }, EWait::Room, number);
    if (waited != 0) {
        Stats::WindowWait(waited);
    }
    std::memcpy(g.ring.get() + g.write % RingBytes, packet, bytes);
    g.positions[number % PositionSlots] = position;
    if (Stats::Timing()) {
        g.push_ns[number % PositionSlots] = Stats::Now();
    }
    g.write += bytes;
    // `positions` before `sent_packets`: R and EngineConsumed read it after an acquire of it.
    g.sent_packets.store(number, std::memory_order_release);
    g.sent.store(g.write, std::memory_order_release);
    WakeRender();
}

/// The oldest unplayed packet's ring A position, ~0 when R has played everything committed.
u64 Floor(u64 played, u64 sent_packets) {
    return played < sent_packets ? g.positions[(played + 1) % PositionSlots] : ~u64{0};
}

/// R (after each unparked packet) and E (EngineConsumed): GpuSplit's `played` (8.2) =
/// min(what E ran, R's oldest unplayed packet). The front's waits read it.
void RaisePlayed() {
    // A Dekker pair: E stores `intended` then reads `played`; R stores `played` then reads
    // `intended`. Without this fence E's load could pass its store (x86 reorders a store with a
    // later load) while R read the old `intended`: both raised GpuSplit's `played` short of the
    // truth and neither raised it again, the front waiting for a position E and R had both passed
    // (the FRITCHII_ONEF_QUERY_OPS hang: E and R idle, GPUWatch silent; render ops made E end its
    // batches with R's packets unplayed, which opened the window). With it at least one side
    // reads the other's store.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // `intended` before `sent_packets` (acquire both): an `intended` E stored after committing a
    // packet comes with that packet's count.
    const u64 intended = g.intended.load(std::memory_order_acquire);
    const u64 played = g.played.load(std::memory_order_acquire);
    const u64 sent_packets = g.sent_packets.load(std::memory_order_acquire);
    GpuSplit::Get().RaisePlayed((std::min)(intended, Floor(played, sent_packets)));
}

/// FRITCHII_ONEF_RAISE_AFTER_LEND: lends after which `played` was short (E's count, logged).
std::atomic<u64> g_lend_short{};
std::atomic<u64> g_lend_short_positions{};
std::atomic<u64> g_lend_short_logged{};
std::atomic<u64> g_lend_short_log_ns{};

/// E, after a lend's park. R raises `played` after every packet but a lent one, and its raise
/// before the lent packet stopped at that packet's floor. A lend made after E's last
/// EngineConsumed (the frame end op when E goes idle) thus left `played` short of what E and R had
/// both passed, and nobody raised it until E consumed more: the front, waiting on R's position
/// before it sends more, waited for the hang watchdog's rescue ("a lost raise of `played`", the
/// 2.5 s freeze in a Castle Town load). Counted always; with the switch on E raises it here.
void CheckRaiseAfterLend() {
    std::atomic_thread_fence(std::memory_order_seq_cst); // as RaisePlayed: pairs with R's store
    const u64 intended = g.intended.load(std::memory_order_acquire);
    const u64 played = g.played.load(std::memory_order_acquire);
    const u64 sent_packets = g.sent_packets.load(std::memory_order_acquire);
    const u64 target = (std::min)(intended, Floor(played, sent_packets));
    const u64 now_played = GpuSplit::Get().PlayedNow(std::memory_order_acquire);
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_raise_after_lend,
                                                      "FRITCHII_ONEF_RAISE_AFTER_LEND");
    if (target > now_played) {
        g_lend_short.fetch_add(1, std::memory_order_relaxed);
        g_lend_short_positions.fetch_add(target - now_played, std::memory_order_relaxed);
        if (mode != 0) {
            RaisePlayed();
        }
    }
    const u64 now = Common::Nx2Hitch::NowNs();
    u64 last = g_lend_short_log_ns.load(std::memory_order_relaxed);
    if (last == 0) {
        g_lend_short_log_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000ULL ||
        !g_lend_short_log_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
        return;
    }
    const u64 total = g_lend_short.load(std::memory_order_relaxed);
    const u64 positions = g_lend_short_positions.exchange(0, std::memory_order_relaxed);
    const u64 count = total - g_lend_short_logged.exchange(total, std::memory_order_relaxed);
    if (count == 0) {
        return;
    }
    const double seconds = static_cast<double>(now - last) / 1e9;
    LOG_INFO(HW_GPU,
             "One follower raise after lend (FRITCHII_ONEF_RAISE_AFTER_LEND={}) {:.0f} s: {:.1f} "
             "lends/s left `played` short of what E and R had both passed ({:.0f} ring words "
             "short on average; {}), {} this session",
             mode, seconds, static_cast<double>(count) / seconds,
             static_cast<double>(positions) / static_cast<double>(count),
             mode != 0 ? "raised after the park" : "left to E's next consumed position or the hang "
                                                   "watchdog",
             total);
}

/// E, after a lend's park: its site's count and time.
void NoteLendSite(LentOp op, u64 ns) {
    const char* const name = std::exchange(t_lend_site, nullptr);
    LendSite* slot = &g_lend_sites_rest;
    for (auto& site : g_lend_sites) {
        if (site.name == name && site.op == static_cast<u8>(op)) {
            slot = &site;
            break;
        }
        if (site.name == nullptr && site.count == 0) {
            site.name = name; // null: Lend called directly (draws, verify compares)
            site.op = static_cast<u8>(op);
            slot = &site;
            break;
        }
    }
    ++slot->count;
    slot->ns += ns;
}

/// The lends by site, every 10 s (R, at frame-end lends: E parked), the 16 costliest.
void LogLendSites() {
    const u64 now = Common::Nx2Hitch::NowNs();
    if (g_lend_sites_log_ns == 0) {
        g_lend_sites_log_ns = now;
        return;
    }
    if (now - g_lend_sites_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - g_lend_sites_log_ns) / 1e9;
    g_lend_sites_log_ns = now;
    struct Row {
        const LendSite* site;
        u64 count;
        u64 ns;
    };
    std::vector<Row> rows;
    const auto take = [&](LendSite& site) {
        const Row row{&site, site.count - site.count_logged, site.ns - site.ns_logged};
        site.count_logged = site.count;
        site.ns_logged = site.ns;
        if (row.count != 0) {
            rows.push_back(row);
        }
    };
    for (auto& site : g_lend_sites) {
        take(site);
    }
    take(g_lend_sites_rest);
    std::ranges::sort(rows, [](const Row& a, const Row& b) { return a.ns > b.ns; });
    std::string text;
    for (size_t i = 0; i < rows.size() && i < 16; ++i) {
        const LendSite& site = *rows[i].site;
        const char* const op = site.op < lent_op_names.size() ? lent_op_names[site.op] : "?";
        text += fmt::format("{}{} ({}) {:.0f}/s {:.1f} ms/s", text.empty() ? "" : "; ",
                            &site == &g_lend_sites_rest ? "other sites"
                            : site.name != nullptr      ? site.name
                                                        : "direct",
                            op, static_cast<double>(rows[i].count) / seconds,
                            static_cast<double>(rows[i].ns) / 1e6 / seconds);
    }
    LOG_INFO(HW_GPU, "One follower lends by site (E parked, 10 s, costliest first): {}", text);
}

/// The query kinds' 10 s line (R, at frame-end lends).
void LogQueryKinds() {
    const u64 now = Common::Nx2Hitch::NowNs();
    if (g.query_log_ns == 0) {
        g.query_log_ns = now;
        return;
    }
    if (now - g.query_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - g.query_log_ns) / 1e9;
    g.query_log_ns = now;
    std::string text;
    for (size_t i = 0; i < query_kind_names.size(); ++i) {
        const u64 count = g.query_count[i].load(std::memory_order_relaxed);
        const u64 ns = g.query_parked_ns[i].load(std::memory_order_relaxed);
        text += fmt::format("{}{} {:.0f}/s {:.1f} ms/s", i == 0 ? "" : ", ", query_kind_names[i],
                            static_cast<double>(count - g.query_count_logged[i]) / seconds,
                            static_cast<double>(ns - g.query_ns_logged[i]) / 1e6 / seconds);
        g.query_count_logged[i] = count;
        g.query_ns_logged[i] = ns;
    }
    // fable_b 11.3: the acquire latches by thread (session totals): R's is the expected one.
    static constexpr std::array<const char*, static_cast<size_t>(InvalKind::Count)> inval_names{
        "run on E", "lent: texture page", "lent: shader page", "lent: query page",
        "lent: hit at the second look", "inline uploads on E",
        "inline uploads with R's half lent", "inline uploads lent: query page", "gathers on E",
        "gathers lent: texture or shader page"};
    std::string inval;
    for (size_t i = 0; i < inval_names.size(); ++i) {
        const u64 count = g.inval_count[i].load(std::memory_order_relaxed);
        inval += fmt::format("{}{} {:.0f}/s", i == 0 ? "" : ", ", inval_names[i],
                             static_cast<double>(count - g.inval_logged[i]) / seconds);
        g.inval_logged[i] = count;
    }
    const u64 checks = g.ahead_checks.load(std::memory_order_relaxed);
    LOG_INFO(HW_GPU,
             "One follower invalidations (FRITCHII_ONEF_INVAL_E={}) {:.0f} s: {}; image uploads "
             "checked against the page write stamps {:.0f}/s (\"texture read ahead of R\" in the "
             "checks line, must be 0)",
             InvalOnEngine() ? 1 : 0, seconds, inval,
             static_cast<double>(checks - g.ahead_checks_logged) / seconds);
    g.ahead_checks_logged = checks;
    const std::array<u64, 4> posting{g.fences_lent.load(std::memory_order_relaxed),
                                     g.fences_with_flush.load(std::memory_order_relaxed),
                                     g.gathers_lent.load(std::memory_order_relaxed),
                                     g.gather_ranges.load(std::memory_order_relaxed)};
    const auto rate = [&](size_t i) {
        return static_cast<double>(posting[i] - g.posting_logged[i]) / seconds;
    };
    const double gathers = rate(2);
    LOG_INFO(HW_GPU,
             "One follower posting (dry, FRITCHII_ONEF_POSTED={}) {:.0f} s: lent fences {:.0f}/s, "
             "{:.0f}/s with a buffer flush pending (the rest need no E half); lent gathers "
             "{:.0f}/s, {:.1f} ranges each",
             Posted() ? 1 : 0, seconds, rate(0), rate(1), gathers,
             gathers > 0 ? rate(3) / gathers : 0.0);
    g.posting_logged = posting;
    {
        static constexpr std::array<const char*, static_cast<size_t>(FenceDecision::Count)>
            names{"posted",           "posted with E's buffer commit", "lent: POST_FENCE off",
                  "lent: delayed",    "lent: stopped",                 "lent: sync owed",
                  "lent: marks ahead due"};
        std::string text;
        for (size_t i = 0; i < names.size(); ++i) {
            const u64 count = g.fence_decisions[i].load(std::memory_order_relaxed);
            text += fmt::format("{}{} {:.0f}/s", i == 0 ? "" : ", ", names[i],
                                static_cast<double>(count - g.fence_decisions_logged[i]) /
                                    seconds);
            g.fence_decisions_logged[i] = count;
        }
        static constexpr std::array<const char*, static_cast<size_t>(IdleDecision::Count)>
            idle_names{"posted, R caught up", "posted, R behind", "lent: R behind",
                       "lent: landing due", "lent: sync owed"};
        std::string idle;
        for (size_t i = 0; i < idle_names.size(); ++i) {
            const u64 count = g.idle_decisions[i].load(std::memory_order_relaxed);
            idle += fmt::format("{}{} {:.0f}/s", i == 0 ? "" : ", ", idle_names[i],
                                static_cast<double>(count - g.idle_decisions_logged[i]) /
                                    seconds);
            g.idle_decisions_logged[i] = count;
        }
        const u64 superset = g.sync_target_superset.load(std::memory_order_relaxed);
        const u64 on_r = g.signals_on_r.load(std::memory_order_relaxed);
        const u64 explained = g.mirror_explained.load(std::memory_order_relaxed);
        namespace Debug = VideoCore::Nx2Debug;
        LOG_INFO(HW_GPU,
                 "One follower fence decisions (POST_FENCE={}, SEM_E={}, SIGNAL_ON_R={}, "
                 "WAITS_R={}, MIRROR_STOP={}) {:.0f} s: {}; samples sync targets R did not need, "
                 "E's mirror ahead {:.1f}/s; waits for idle (POST_IDLE={}): {}; signals made on "
                 "R for E {:.0f}/s; E's mirror ahead after a skipped draw {:.1f}/s",
                 PostFenceOn() ? 1 : 0, g_sem_e.load(std::memory_order_relaxed) ? 1 : 0,
                 Debug::SwitchValue(Debug::g_onef_signal_on_r, "FRITCHII_ONEF_SIGNAL_ON_R"),
                 WaitsR() ? 1 : 0,
                 Debug::SwitchValue(Debug::g_onef_mirror_stop, "FRITCHII_ONEF_MIRROR_STOP"),
                 seconds, text,
                 static_cast<double>(superset - g.sync_target_superset_logged) / seconds,
                 PostIdleOn(), idle,
                 static_cast<double>(on_r - g.signals_on_r_logged) / seconds,
                 static_cast<double>(explained - g.mirror_explained_logged) / seconds);
        g.sync_target_superset_logged = superset;
        g.signals_on_r_logged = on_r;
        g.mirror_explained_logged = explained;
    }
    if (SyncMirrorOn()) {
        std::array<double, 3> rates{};
        for (size_t i = 0; i < rates.size(); ++i) {
            const u64 count = g.mirror_count[i].load(std::memory_order_relaxed);
            rates[i] = static_cast<double>(count - g.mirror_logged[i]) / seconds;
            g.mirror_logged[i] = count;
        }
        LOG_INFO(HW_GPU,
                 "One follower sync mirror (fable_b 16.2, FRITCHII_ONEF_SYNC_MIRROR=1) {:.0f} s: "
                 "samples reports compared {:.0f}/s, E set and R not {:.0f}/s, R set and E not "
                 "{:.0f}/s (\"sync mirror\" in the checks line, must be 0)",
                 seconds, rates[0], rates[1], rates[2]);
    }
    static constexpr std::array<const char*, static_cast<size_t>(OpCount::Count)> op_names{
        "posted", "lent: a sync in a held fill", "lent: inside a draw fill", "lent: no room",
        "lent: no packet ticks"};
    std::string ops;
    for (size_t i = 0; i < op_names.size(); ++i) {
        const u64 count = g.op_count[i].load(std::memory_order_relaxed);
        ops += fmt::format("{}{} {:.0f}/s", i == 0 ? "" : ", ", op_names[i],
                           static_cast<double>(count - g.op_logged[i]) / seconds);
        g.op_logged[i] = count;
    }
    const u64 op_bytes = g.op_bytes.load(std::memory_order_relaxed);
    const double posted = static_cast<double>(g.op_count[0].load(std::memory_order_relaxed));
    (void)posted;
    LOG_INFO(HW_GPU,
             "One follower op packets (fable_b 13.1; FRITCHII_ONEF_OP_INLINE={}) {:.0f} s: {}; "
             "{:.1f} KiB/s of entries posted; lends parked holding a lock {:.0f}/s (handed to "
             "R); E reads that waited for a posted fence's report landing {:.0f}/s",
             PostOpSite(15) ? 1 : 0, seconds, ops,
             static_cast<double>(op_bytes - g.op_bytes_logged) / 1024.0 / seconds,
             static_cast<double>(g.lends_holding_lock.load(std::memory_order_relaxed) -
                                 g.lends_holding_lock_logged) /
                 seconds,
             static_cast<double>(g.landing_waits.load(std::memory_order_relaxed) -
                                 g.landing_waits_logged) /
                 seconds);
    g.landing_waits_logged = g.landing_waits.load(std::memory_order_relaxed);
    g.lends_holding_lock_logged = g.lends_holding_lock.load(std::memory_order_relaxed);
    g.op_bytes_logged = op_bytes;
    {
        // FRITCHII_ONEF_CARRY_OPS (fable_b 22.3 step 2): the ops that rode in another packet.
        const auto rate = [seconds](const std::atomic<u64>& counter, u64& logged) {
            const u64 now = counter.load(std::memory_order_relaxed);
            const double value = static_cast<double>(now - logged) / seconds;
            logged = now;
            return value;
        };
        const double carried = rate(g.carried_ops, g.carried_ops_logged);
        std::array<double, 4> carriers{};
        for (size_t i = 0; i < carriers.size(); ++i) {
            carriers[i] = rate(g.carriers[i], g.carriers_logged[i]);
        }
        std::array<double, 3> alone{};
        for (size_t i = 0; i < alone.size(); ++i) {
            alone[i] = rate(g.carried_alone[i], g.carried_alone_logged[i]);
        }
        if (carried != 0.0 || CarryOpsOn()) {
            LOG_INFO(HW_GPU,
                     "One follower carried ops (fable_b 22.3, FRITCHII_ONEF_CARRY_OPS={}) {:.0f} "
                     "s: {:.0f}/s rode in {:.0f} packets/s: draw {:.0f}, lent {:.0f}, op {:.0f}, "
                     "alone {:.0f} (at E's consumed store {:.0f}, a drain {:.0f}, no room {:.0f})",
                     CarryOpsOn() ? 1 : 0, seconds, carried,
                     carriers[0] + carriers[1] + carriers[2] + carriers[3], carriers[0],
                     carriers[1], carriers[2], carriers[3], alone[0], alone[1], alone[2]);
        }
    }
    {
        // FRITCHII_ONEF_OP_DISPATCH: where each dispatch went (lent ones by reason), and the
        // posted share. Only while the switch is on (step 4).
        std::array<double, static_cast<size_t>(DispatchCount::Count)> rates{};
        double total = 0.0;
        bool any = false;
        for (size_t i = 0; i < rates.size(); ++i) {
            const u64 count = g.dispatch_count[i].load(std::memory_order_relaxed);
            rates[i] = static_cast<double>(count - g.dispatch_logged[i]) / seconds;
            g.dispatch_logged[i] = count;
            any = any || rates[i] > 0.0;
            const auto kind = static_cast<DispatchCount>(i);
            if (!DispatchCountIsExtra(kind)) {
                total += rates[i]; // the extras are posted dispatches counted once more
            }
        }
        if (any) {
            std::string dispatches;
            for (size_t i = 0; i < rates.size(); ++i) {
                dispatches += fmt::format("{}{} {:.0f}/s", i == 0 ? "" : ", ",
                                          dispatch_count_names[i], rates[i]);
            }
            LOG_INFO(HW_GPU,
                     "One follower dispatches (FRITCHII_ONEF_OP_DISPATCH={}, BIND_SEED={}, "
                     "DISPATCH_RANGES={}; fable_b 13.4 group 6) {:.0f} s: {:.0f}/s, {:.1f}% "
                     "posted: {}{}",
                     DispatchPostBits(), BindSeedOn() ? 1 : 0, DispatchRangesOn() ? 1 : 0,
                     seconds, total,
                     total > 0.0 ? 100.0 * rates[0] / total : 0.0, dispatches,
                     g.post_dispatch_stopped.load(std::memory_order_relaxed)
                         ? " (STOPPED for the session: see the checks line)"
                         : "");
        }
    }
    // fable_b 19 step 2: transient lent draws by the test that fired, each split by kind.
    {
        std::string tests;
        u64 total = 0;
        for (size_t test = 0; test < transient_test_names.size(); ++test) {
            u64 sum = 0;
            std::string kinds;
            for (size_t slot = 0; slot < Stats::KindSlots; ++slot) {
                const u64 count = g.transient[test][slot].load(std::memory_order_relaxed);
                const u64 delta = count - g.transient_logged[test][slot];
                g.transient_logged[test][slot] = count;
                if (delta == 0) {
                    continue;
                }
                sum += delta;
                kinds += fmt::format("{}{} {:.0f}", kinds.empty() ? "" : ", ",
                                     Stats::kind_names[slot],
                                     static_cast<double>(delta) / seconds);
            }
            total += sum;
            tests += fmt::format("{}{} {:.0f}/s{}{}{}", test == 0 ? "" : ", ",
                                 transient_test_names[test],
                                 static_cast<double>(sum) / seconds, kinds.empty() ? "" : " (",
                                 kinds, kinds.empty() ? "" : ")");
        }
        LOG_INFO(HW_GPU,
                 "One follower transient lends (fable_b 19) {:.0f} s: {:.0f}/s; by test: {}",
                 seconds, static_cast<double>(total) / seconds, tests);
    }
    LOG_INFO(HW_GPU,
             "One follower queries by kind (FRITCHII_ONEF_QUERY_OPS={}) {:.0f} s: {}; acquire "
             "latches this session by R {}, E {}, other threads {}",
             QueryOps() ? 1 : 0, seconds, text, g_latches[0].load(std::memory_order_relaxed),
             g_latches[1].load(std::memory_order_relaxed),
             g_latches[2].load(std::memory_order_relaxed));
}

void LogMaybe() {
    LogQueryKinds();
    LogLendSites();
    Stats::Logged now{};
    now.step = static_cast<StepLevel>(g.step.load(std::memory_order_relaxed));
    now.asked = g.asked.load(std::memory_order_relaxed);
    now.unlend_mask = g.unlend_mask.load(std::memory_order_relaxed);
    now.ahead = g.ahead.load(std::memory_order_relaxed);
    now.verify_every = g.verify_every.load(std::memory_order_relaxed);
    Stats::MaybeLog(now);
}

/// R: one packet. Returns where its time goes (Stats::RenderTime).
Stats::RenderTime Play(const PacketHeader& header, const u8* spans) {
    t_play_number = static_cast<u32>(header.number);
    if ((header.flags & PacketCarried) != 0) {
        // FRITCHII_ONEF_CARRY_OPS: the ops E kept since its last packet were made before this
        // packet's own place, so they run first (they read nothing of the view the spans move).
        RunCarried(header, spans - sizeof(PacketHeader));
    }
    RegisterView& view = View();
    Maxwell3D* const engine = g.engine.load(std::memory_order_acquire);
    const bool view_on = Step() >= StepLevel::View;
    if (header.kind == PacketKind::Lent && view_on && engine != nullptr &&
        ((header.flags & PacketSeed) != 0 || !view.seeded)) {
        SeedView(view, *engine); // E is parked: its engine is at this packet's place
    }
    if (header.spans != 0 && view.seeded) {
        ApplySpans(view, header, spans);
    }
    // Part (c)'s view engine (from step 3): every packet after the spans, a lent one before its
    // call runs (E parked). Below step 3 it is idle and costs nothing.
    DrawPlayer* const player = Step() >= StepLevel::Build ? GetDrawPlayer() : nullptr;
    if (player != nullptr) {
        player->OnPacket(header, spans);
    } else if (DrawPlayer* const idle = GetDrawPlayer()) {
        idle->ViewOff(); // below step 3 the view is not kept: R reads E's engine (lent only)
    }
    switch (header.kind) {
    case PacketKind::Lent: {
        const bool around = view_on && view.seeded && engine != nullptr;
        if (player != nullptr) {
            player->BeforeLent(); // rules 1 and 2: the lent call reads E's engine
        }
        if (around) {
            BeforeLentCall(view, *engine);
            if ((header.flags & PacketVerify) != 0) {
                VerifyView(view, *engine);
            }
        }
        const bool timing = Stats::Timing();
        const u64 begin = timing ? Stats::Now() : 0;
        ++t_lent_depth;
        try {
            header.call.function(header.call.argument);
        } catch (...) {
            g.lent_error = std::current_exception();
        }
        --t_lent_depth;
        if (around) {
            AfterLentCall(view, *engine);
        }
        // FRITCHII_ONEF_BIND_SEED: the call bound another channel (SetEngine), E is still
        // parked: E's tracking and the views start over from the new engine here, not at the
        // next lent packet (the draw after a bind was lent only to carry that seed).
        if (view_on) {
            Maxwell3D* const bound = g.engine.load(std::memory_order_acquire);
            if (bound != nullptr && bound != engine && BindSeedOn()) {
                Detail::SeedAfterBind(view, *bound);
            }
        }
        if (player != nullptr) {
            player->AfterLent(); // the views take the engine's flags; R is on the view again
        }
        const LentOp op = static_cast<LentOp>(header.op);
        if (timing) {
            const u64 end = Stats::Now();
            const u64 pushed_at = g.push_ns[header.number % PositionSlots];
            Stats::LentRan(op, begin > pushed_at && pushed_at != 0 ? begin - pushed_at : 0,
                           end - begin);
        } else {
            Stats::LentRan(op, 0, 0);
        }
        if (op == LentOp::FrameEnd) {
            LogMaybe();
        }
        return Stats::RenderTime::LentCalls;
    }
    case PacketKind::Draw:
        if (DrawPlayer* const draws = GetDrawPlayer(); draws != nullptr) {
            t_playing_draw = true;
            SCOPE_EXIT {
                t_playing_draw = false;
            };
            draws->PlayDraw(header);
        } else {
            Note(Check::RenderPacketMissing, "a draw packet with no player");
        }
        return Stats::RenderTime::DrawPackets;
    case PacketKind::Delta:
        break;
    case PacketKind::Op:
        // A render op at its stream place (E not parked): R's own code on R's state.
        try {
            header.call.function(const_cast<u8*>(spans) + header.reserved2);
        } catch (...) {
            if (!g.op_error) {
                g.op_error = std::current_exception();
            }
        }
        break;
    }
    return Stats::RenderTime::OtherPackets;
}

/// R's contended-lock hook (SpinRecursiveMutex::t_contended_hook): R waiting for a lock E holds
/// outside a lent call is R reaching into E's cache. Counted with the call stack (first reports):
/// the deadlock of 12 Nov had R in a draw packet waiting for the buffer cache's lock while E
/// was parked in a lend holding it.
bool RenderContended(std::thread::id holder, bool first) {
    const bool holder_is_e = holder == g.e_thread.load(std::memory_order_relaxed) ||
                             holder == g.e2_thread.load(std::memory_order_relaxed);
    if (!first || !holder_is_e || t_lent_depth != 0) {
        return false;
    }
    std::string stack;
#ifdef _WIN32
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) < 12) {
        void* frames[32];
        const USHORT count = RtlCaptureStackBackTrace(1, 32, frames, nullptr);
        for (USHORT i = 0; i < count; ++i) {
            HMODULE module{};
            char name[MAX_PATH] = "?";
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   static_cast<LPCSTR>(frames[i]), &module)) {
                GetModuleFileNameA(module, name, sizeof(name));
            }
            const char* const base_name = std::strrchr(name, '\\');
            stack += fmt::format(" {}+{:#x}", base_name != nullptr ? base_name + 1 : name,
                                 reinterpret_cast<uintptr_t>(frames[i]) -
                                     reinterpret_cast<uintptr_t>(module));
        }
    }
#endif
    thread_local std::string note;
    note = fmt::format("R (packet {}) waits for a lock E holds; stack:{}",
                       g.r_packet.load(std::memory_order_relaxed),
                       stack.empty() ? std::string{" (not logged)"} : stack);
    Note(Check::RenderWaitsEngineLock, note.c_str());
    return false;
}

void RenderMain(std::stop_token stop) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuMem};
    Common::SetCurrentThreadName("GPURender");
    t_render = true;
    Common::SpinRecursiveMutex::t_contended_hook = &RenderContended;
    GpuSplit::MarkRenderThread();
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    if (g.pin) {
        // FRITCHII_ONEF_PIN (default 1): the core ThreadPinning keeps for GPUMem (thread.cpp
        // reserves it when FRITCHII_ONEF=1).
        Common::PinCurrentThread(Common::PinnedRole::GpuMem);
    }
    g.system->RegisterHostThread();
    g.r_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    for (;;) {
        u64 sent = g.sent.load(std::memory_order_acquire);
        if (sent == g.read_at && !stop.stop_requested()) {
            const bool timing = Stats::Timing();
            const u64 idle_begin = timing ? Stats::Now() : 0;
            // FRITCHII_ONEF_R_SPIN (live): pauses before R sleeps (default SpinLimit, as before).
            const u32 spin_limit = static_cast<u32>((std::max)(
                0, VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_r_spin,
                                                    "FRITCHII_ONEF_R_SPIN",
                                                    static_cast<int>(SpinLimit))));
            for (u32 spins = 0; sent == g.read_at && !stop.stop_requested(); ++spins) {
                if (spins < spin_limit) {
                    Pause();
                } else {
                    g.r_sleeping.store(true, std::memory_order_seq_cst);
                    const u32 seen = g.r_wake.load(std::memory_order_seq_cst);
                    if (g.sent.load(std::memory_order_seq_cst) == g.read_at &&
                        !stop.stop_requested()) {
                        Stats::RenderSlept();
                        g.r_wake.wait(seen, std::memory_order_seq_cst);
                    }
                    g.r_sleeping.store(false, std::memory_order_relaxed);
                    spins = 0;
                }
                sent = g.sent.load(std::memory_order_acquire);
            }
            if (timing) {
                Stats::RenderTimeSpent(Stats::RenderTime::Idle, Stats::Now() - idle_begin);
            }
        }
        if (sent == g.read_at) {
            break; // stop requested with nothing left
        }
        const u8* const at = g.ring.get() + g.read_at % RingBytes;
        PacketHeader header;
        std::memcpy(&header, at, sizeof(header));
        if (static_cast<u8>(header.kind) == PadKind) {
            g.read_at += header.bytes;
            g.freed.store(g.read_at, std::memory_order_release);
            g.r_progress.store(g.r_progress.load(std::memory_order_relaxed) + 1,
                               std::memory_order_relaxed);
            WakeEngine();
            continue;
        }
        const bool timing = Stats::Timing();
        const u64 begin = timing ? Stats::Now() : 0;
        g.r_kind.store(static_cast<u8>(header.kind), std::memory_order_relaxed);
        g.r_op.store(header.op, std::memory_order_relaxed);
        g.r_packet.store(header.number, std::memory_order_relaxed);
        const u64 trace_start = g_trace ? Common::Nx2Hitch::NowNs() : 0;
        // FRITCHII_SIM_CPU_SLOW: playing one packet is R's work.
        Common::LowEndSim::CpuWorkBegin(Common::LowEndSim::CpuClass::Render);
        const Stats::RenderTime where = Play(header, at + sizeof(PacketHeader));
        Common::LowEndSim::CpuWorkEnd();
        if (g_trace) [[unlikely]] {
            const u64 trace_end = Common::Nx2Hitch::NowNs();
            if (header.kind == PacketKind::Lent || header.kind == PacketKind::Op) {
                TraceRecord& record = g_trace_ring[header.number % TraceSlots];
                record.r_start = trace_start;
                record.r_end = trace_end;
                record.r_busy = std::exchange(t_r_busy, 0);
            } else {
                t_r_busy += trace_end - trace_start;
            }
        }
        g.r_packet.store(0, std::memory_order_relaxed);
        g.r_progress.store(g.r_progress.load(std::memory_order_relaxed) + 1,
                           std::memory_order_relaxed);
        if (timing && where != Stats::RenderTime::LentCalls) {
            Stats::RenderTimeSpent(where, Stats::Now() - begin);
        }
        g.read_at += header.bytes;
        g.freed.store(g.read_at, std::memory_order_release);
        g.played.store(header.number, std::memory_order_seq_cst);
        WakeEngine();
        if (header.kind != PacketKind::Lent) {
            // Step 4: E raised `played` only to this packet while it was unplayed.
            RaisePlayed();
        }
    }
}

/// The watchdog's names for the threads it knows.
std::string ThreadName(std::thread::id id) {
    if (id == std::thread::id{}) {
        return "free";
    }
    if (id == g.e_thread.load(std::memory_order_relaxed)) {
        return "GPUBack (E)";
    }
    if (id == g.e2_thread.load(std::memory_order_relaxed)) {
        return "GPUBackE2 (E's second thread)";
    }
    if (id == g.r_thread.load(std::memory_order_relaxed)) {
        return "GPURender (R)";
    }
    return fmt::format("another thread ({:#x})", std::hash<std::thread::id>{}(id));
}

/// The watchdog's second line: the front's wait and the acquires it may be held by. `who` starts
/// the line; the hang report (ReportHangState) lists the acquires itself (`with_waiters` false).
void ReportFrontAndAcquires(const char* who, bool front_stuck, bool e_outside, u64 now_ns,
                            bool with_waiters = true) {
    auto& split = GpuSplit::Get();
    const auto& front = split.FrontWaitNow();
    const u64 begin = front.begin_ns.load(std::memory_order_acquire);
    const u32 site = front.site.load(std::memory_order_relaxed);
    std::string waiters;
    u32 count = 0;
    // Never blocks for long: a report runs while something may hold the waiters' lock.
    const bool listed = !with_waiters || Tegra::Nx2SemaphoreWaiters::Get().TryForEach(
        [&](u64 address, u32 value, u32 operation, bool latched, const u32* word, double) {
            if (++count > 16) {
                return;
            }
            waiters += fmt::format(
                "{}{:#x} {} {:#x}{}{}", waiters.empty() ? "" : "; ", address,
                operation == Tegra::Nx2SemaphoreWaiters::AcquireEqual    ? "=="
                : operation == Tegra::Nx2SemaphoreWaiters::AcquireGequal ? ">="
                : operation == Tegra::Nx2SemaphoreWaiters::AcquireMask   ? "&"
                                                                          : "?",
                value, latched ? " latched" : "",
                word != nullptr ? fmt::format(" (word now {:#x})",
                                              std::atomic_ref<u32>{*const_cast<u32*>(word)}
                                                  .load(std::memory_order_relaxed))
                                : std::string{});
        });
    LOG_CRITICAL(
        HW_GPU,
        "{}: front {} (site {}, target {}, {}, for {:.1f} s, wait #{}); "
        "published {}, consumed (E) {}, played (R) {}; E {}; acquire waiters {}: {}; latches by "
        "R {}, E {}, other {}",
        who, begin != 0 ? (front_stuck ? "stuck in a wait" : "in a wait") : "not in a wait",
        site < drain_site_names.size() ? drain_site_names[site] : "none",
        front.target.load(std::memory_order_relaxed),
        front.on_engine.load(std::memory_order_relaxed) ? "on E's position" : "on R's position",
        begin != 0 ? static_cast<double>(now_ns - begin) / 1e9 : 0.0,
        front.seq.load(std::memory_order_relaxed), split.PublishedNow(),
        split.EngineConsumedNow(), split.PlayedNow(std::memory_order_acquire),
        e_outside ? "owes work and has not moved for 2 s outside its one-follower waits"
                  : "not stuck outside its waits",
        count,
        !with_waiters     ? std::string{"in the hang report's acquire line"}
        : !listed         ? std::string{"not read: their lock stayed held for 200 ms"}
        : waiters.empty() ? std::string{"none"}
                          : waiters,
        g_latches[0].load(std::memory_order_relaxed), g_latches[1].load(std::memory_order_relaxed),
        g_latches[2].load(std::memory_order_relaxed));
}

/// The watchdog's state line from `step` on, read fresh (also the hang report's).
std::string StateText() {
    auto& split = GpuSplit::Get();
    std::string locks;
    const u32 lock_count = g.lock_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < lock_count; ++i) {
        locks += fmt::format("{}{} held by {}", i == 0 ? "" : ", ", g.locks[i].first,
                             ThreadName(g.locks[i].second->Owner()));
    }
    const char* const back_site = back_wait_site.load(std::memory_order_relaxed);
    const u64 r_packet = g.r_packet.load(std::memory_order_relaxed);
    const u8 e_wait = g.e_wait.load(std::memory_order_relaxed);
    const u8 r_kind = g.r_kind.load(std::memory_order_relaxed);
    const u8 r_op = g.r_op.load(std::memory_order_relaxed);
    const u8 e_op = g.e_wait_op.load(std::memory_order_relaxed);
    return fmt::format(
        "step {}; positions: published {}, consumed (E) {}, played (R) {}; packets: pushed {}, "
        "played {}, ring bytes sent {} freed {}; E: {} (op {}, packet {}), e_waiting {}, back "
        "wait site {}; R: {} (packet {}, kind {}, op {}), r_sleeping {}; locks: {}",
        static_cast<int>(g.step.load(std::memory_order_relaxed)), split.PublishedNow(),
        split.EngineConsumedNow(), split.PlayedNow(std::memory_order_acquire),
        g.sent_packets.load(std::memory_order_acquire), g.played.load(std::memory_order_acquire),
        g.sent.load(std::memory_order_acquire), g.freed.load(std::memory_order_acquire),
        e_wait_names[(std::min)(static_cast<size_t>(e_wait), e_wait_names.size() - 1)],
        e_op < lent_op_names.size() ? lent_op_names[e_op] : "?",
        g.e_wait_number.load(std::memory_order_relaxed),
        g.e_waiting.load(std::memory_order_relaxed),
        back_site != nullptr ? back_site : "none tracked",
        r_packet != 0 ? "in a packet"
        : g.r_sleeping.load(std::memory_order_relaxed) ? "asleep"
                                                       : "between packets",
        r_packet,
        r_kind == static_cast<u8>(PacketKind::Lent)    ? "lent"
        : r_kind == static_cast<u8>(PacketKind::Draw)  ? "draw"
        : r_kind == static_cast<u8>(PacketKind::Delta) ? "delta"
                                                       : "?",
        r_kind == static_cast<u8>(PacketKind::Lent) && r_op < lent_op_names.size()
            ? lent_op_names[r_op]
            : "-",
        g.r_sleeping.load(std::memory_order_relaxed),
        locks.empty() ? std::string{"none registered"} : locks);
}

/// The hang watchdog (FRITCHII_ONEF=1): one wake a second. Two seconds without progress where
/// progress is owed (R has a packet it does not finish, or packets it does not take; E in one wait
/// that does not end) logs the whole state once, at Critical, then gives both sides a wake (a
/// lost wakeup ends there, and the log says so when progress resumes).
void WatchMain(std::stop_token stop) {
    Common::SetCurrentThreadName("GPUWatch");
    struct Seen {
        u64 played{};
        u64 r_packet{};
        u64 r_progress{};
        u64 e_wait_seq{};
        u64 front_seq{};
        u64 consumed{};
    } last{};
    u32 r_strikes = 0;
    u32 e_strikes = 0;
    u32 e_out_strikes = 0;
    bool reported = false;
    bool lost_wakeup = false;
    for (;;) {
        for (int i = 0; i < 10; ++i) {
            if (stop.stop_requested()) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        TraceFlush(false);
        const u64 played = g.played.load(std::memory_order_acquire);
        const u64 sent_packets = g.sent_packets.load(std::memory_order_acquire);
        const u64 r_packet = g.r_packet.load(std::memory_order_relaxed);
        const u64 r_progress = g.r_progress.load(std::memory_order_relaxed);
        const u8 e_wait = g.e_wait.load(std::memory_order_relaxed);
        const u64 e_wait_seq = g.e_wait_seq.load(std::memory_order_relaxed);
        const u64 sent = g.sent.load(std::memory_order_acquire);
        const u64 freed = g.freed.load(std::memory_order_acquire);
        // R owes progress when it holds a packet, or when packets wait in the ring.
        const bool r_owes = r_packet != 0 || sent != freed || sent_packets > played;
        const bool r_still = r_owes && r_packet == last.r_packet && r_progress == last.r_progress;
        const bool e_still = e_wait != 0 && e_wait_seq == last.e_wait_seq;
        r_strikes = r_still ? r_strikes + 1 : 0;
        e_strikes = e_still ? e_strikes + 1 : 0;
        // The front in one wait for over 2 s (GpuSplit's WaitFor or Drain), and E owing work
        // (the front published past its position) without moving and outside its own waits: both
        // were invisible to the R and E strikes (the first QUERY_OPS hang).
        auto& split_now = GpuSplit::Get();
        const auto& front = split_now.FrontWaitNow();
        const u64 front_begin = front.begin_ns.load(std::memory_order_acquire);
        const u64 front_seq = front.seq.load(std::memory_order_relaxed);
        const u64 now_ns = Common::Nx2Hitch::NowNs();
        const bool front_stuck = front_begin != 0 && now_ns - front_begin > 2'000'000'000ULL;
        const u64 consumed_now = split_now.EngineConsumedNow();
        const bool e_out_still = e_wait == 0 && split_now.PublishedNow() > consumed_now &&
                                 consumed_now == last.consumed;
        e_out_strikes = e_out_still ? e_out_strikes + 1 : 0;
        last = Seen{played, r_packet, r_progress, e_wait_seq, front_seq, consumed_now};
        const bool hung = r_strikes >= 2 || e_strikes >= 2 || front_stuck || e_out_strikes >= 2;
        if (!hung) {
            if (reported) {
                LOG_CRITICAL(HW_GPU, "One follower hang watchdog: progress again ({}; {} wakes "
                                     "given this session)",
                             lost_wakeup ? "a lost wakeup, ended by the watchdog's wake"
                                         : "a long wait that ended",
                             g.rescued.load(std::memory_order_relaxed));
                reported = false;
            }
            continue;
        }
        if (!reported) {
            reported = true;
            // A lost wakeup: a side asleep although what it waits for is there.
            lost_wakeup = (g.r_sleeping.load(std::memory_order_seq_cst) && sent != freed) ||
                          (e_wait == static_cast<u8>(EWait::Park) &&
                           g.e_waiting.load(std::memory_order_seq_cst) &&
                           played >= g.e_wait_number.load(std::memory_order_relaxed));
            auto& split = GpuSplit::Get();
            ReportFrontAndAcquires("One follower hang watchdog", front_stuck, e_out_strikes >= 2,
                                   now_ns);
            // A lost raise: R has nothing unplayed and E is idle, yet `played` is short of
            // E's position. Raise it (the min of the two, as RaisePlayed always takes) and say so.
            const u64 intended = g.intended.load(std::memory_order_seq_cst);
            if (!r_owes && split.PlayedNow(std::memory_order_seq_cst) < intended) {
                LOG_CRITICAL(HW_GPU,
                             "One follower hang watchdog: a lost raise of `played` ({} short of "
                             "E's {}, R with nothing unplayed): raised now",
                             split.PlayedNow(std::memory_order_relaxed), intended);
                RaisePlayed();
            }
            LOG_CRITICAL(
                HW_GPU, "One follower hang watchdog: no progress for 2 s ({}{}{}{}{}); {}",
                r_strikes >= 2 ? "R" : "", e_strikes >= 2 ? (r_strikes >= 2 ? " and E" : "E") : "",
                front_stuck ? " the front in one wait" : "",
                e_out_strikes >= 2 ? " E owing work outside its waits" : "",
                lost_wakeup ? ", a lost wakeup: a side asleep with its condition met" : "",
                StateText());
        }
        // A wake to both sides: harmless if they are not asleep (each rechecks its condition).
        g.rescued.fetch_add(1, std::memory_order_relaxed);
        g.r_wake.fetch_add(1, std::memory_order_seq_cst);
        g.r_wake.notify_one();
        g.e_wake.fetch_add(1, std::memory_order_seq_cst);
        g.e_wake.notify_one();
    }
}

/// E: the header's draw fields from SetNextDraw (consumed), for Stats and the draw span.
u32 TakeDrawKind(PacketHeader& header) {
    if (!t_next_draw_set) {
        return 0;
    }
    header.reserved[0] = static_cast<u8>(t_next_draw.kind);
    return t_next_draw.kind;
}

} // Anonymous namespace

namespace Detail {
const DrawValues* NextDraw() noexcept {
    return t_next_draw_set ? &t_next_draw : nullptr;
}
} // namespace Detail

namespace {
std::atomic<bool> g_amd_proprietary{};
std::atomic<bool> g_integrated_gpu{};
/// FRITCHII_ONEF_OCCL (boot), decided in ReadSwitches.
bool g_occl_wanted{};
} // Anonymous namespace

void NoteAmdProprietaryDriver(bool integrated) noexcept {
    g_integrated_gpu.store(integrated, std::memory_order_relaxed);
    g_amd_proprietary.store(true, std::memory_order_relaxed);
}

bool IsAmdProprietaryDriver() noexcept {
    return g_amd_proprietary.load(std::memory_order_relaxed);
}

bool IsAmdProprietaryDiscrete() noexcept {
    return g_amd_proprietary.load(std::memory_order_relaxed) &&
           !g_integrated_gpu.load(std::memory_order_relaxed);
}

bool OcclWanted() noexcept {
    return g_occl_wanted;
}

void ReadSwitches() {
    if (g.read) {
        return; // boot switches: the first session's values stick
    }
    g.read = true;
    using namespace VideoCore::Nx2Debug;
    // FRITCHII_ONEF_OCCL (boot): occlusion flag variants (FRITCHII_OCCL_FLAG) with one follower
    // on, lent draws and played ones (E picks a played draw's variant). An explicit value wins;
    // unset, on only on AMD's proprietary driver on a discrete GPU (where it was measured, with
    // EnsureVertexBindings: the AMD losses were a stale vertex-buffer table, not the flag store),
    // off elsewhere until an A/B there. 0 is v6.2's rule: no flag variant with one follower on.
    {
        const char* const value = std::getenv("FRITCHII_ONEF_OCCL");
        const bool set = value != nullptr && value[0] != '\0';
        g_occl_wanted = set ? value[0] != '0' : IsAmdProprietaryDiscrete();
        LOG_INFO(HW_GPU, "One follower: occlusion flags {} (FRITCHII_ONEF_OCCL={})",
                 g_occl_wanted ? "on" : "off",
                 set ? std::string{value}
                     : fmt::format("unset: {}", IsAmdProprietaryDiscrete()
                                                    ? "AMD proprietary driver, discrete GPU"
                                                    : "not an AMD proprietary discrete GPU"));
    }
    // AMD (FRITCHII_ONEF_AMD, boot). 2026-11-16: the desktop RX 7900 XT lost the device in game
    // with the one follower on, so 6.2 kept it off on AMD's proprietary driver. 2026-11-19: the
    // loss was found with Radeon GPU Detective (AMD's vertex-buffer table in command-buffer
    // embedded memory read through a stale pointer; MemPlayer::EnsureVertexBindings), and with
    // occlusion flags in played draws (FRITCHII_ONEF_OCCL, its pre-pass variants) one follower
    // beats off on an RX 9070 XT eGPU (fountain -10% idle, -3% under PCIe load; 0 losses in about
    // 30 launches and a soak). So on by default there, only where measured: a discrete GPU, and
    // the occlusion speed-up at Full (9) in effect (the win exists only with the pre-pass
    // variants; at Standard one follower on against off was never measured on AMD). Otherwise
    // off, as in 6.2. FRITCHII_ONEF_AMD=0 turns it off, =1 keeps it on anyway (tests). RADV and
    // other vendors unchanged.
    if (g_amd_proprietary.load(std::memory_order_relaxed)) {
        const char* const keep = std::getenv("FRITCHII_ONEF_AMD");
        const bool keep_set = keep != nullptr && keep[0] != '\0';
        const bool wanted = (std::getenv("FRITCHII_ONEF_LEAN") != nullptr &&
                             std::getenv("FRITCHII_ONEF_LEAN")[0] == '1') ||
                            (std::getenv("FRITCHII_ONEF") != nullptr &&
                             std::getenv("FRITCHII_ONEF")[0] == '1');
        const char* off_why = nullptr;
        if (keep_set) {
            off_why = keep[0] == '0' ? "FRITCHII_ONEF_AMD=0" : nullptr;
        } else if (g_integrated_gpu.load(std::memory_order_relaxed)) {
            off_why = "an integrated GPU: measured on a discrete one only";
        } else if (VideoCommon::OcclFlag::RawMode() != VideoCommon::OcclFlag::Full) {
            off_why = "the occlusion speed-up is not Full (FRITCHII_OCCL_FLAG=9): one follower "
                      "beats off there only with its pre-pass variants";
        }
        if (wanted && off_why != nullptr) {
            LOG_WARNING(HW_GPU,
                        "One follower: off on AMD's proprietary driver ({}); FRITCHII_ONEF_AMD=1 "
                        "keeps it on",
                        off_why);
            return; // g.on and g.lean stay false: every FRITCHII_ONEF_* switch reads as off
        }
        if (wanted) {
            LOG_INFO(HW_GPU,
                     "One follower on AMD's proprietary driver ({}); FRITCHII_ONEF_AMD=0 turns it "
                     "off",
                     keep_set ? "FRITCHII_ONEF_AMD=1"
                              : "discrete GPU, occlusion speed-up Full");
        }
    }
    g.lean = SwitchValue(g_onef_lean, "FRITCHII_ONEF_LEAN") == 1;
    // Before E starts (ReadSwitches runs ahead of the back thread): LEAN is always at step 2 or
    // more once it runs, and tracking from boot only marks words the first seed then covers.
    g_track_writes_always = g.lean;
    g.on = g.lean || SwitchValue(g_onef, "FRITCHII_ONEF") == 1;
    g.pin = SwitchValue(g_onef_pin, "FRITCHII_ONEF_PIN", 1) != 0;
    g_trace = g.on && SwitchValue(g_onef_trace, "FRITCHII_ONEF_TRACE") == 1;
    if (g.lean) {
        // The verified configuration as defaults; a switch set in the environment (the test
        // block) or read before this keeps its value.
        const auto preset = [](std::atomic<int>& value, const char* name, int lean) {
            int unset = -1;
            if (std::getenv(name) == nullptr && value.compare_exchange_strong(unset, lean)) {
                BumpSwitchGeneration();
            }
            return SwitchValue(value, name);
        };
        const int unlend = preset(g_onef_unlend, "FRITCHII_ONEF_UNLEND", 3);
        const int e_pipeline = preset(g_onef_e_pipeline, "FRITCHII_ONEF_E_PIPELINE", 1);
        const int query_ops = preset(g_onef_query_ops, "FRITCHII_ONEF_QUERY_OPS", 1);
        const int waits_e = preset(g_onef_waits_e, "FRITCHII_ONEF_WAITS_E", 1);
        const int inval_e = preset(g_onef_inval_e, "FRITCHII_ONEF_INVAL_E", 1);
        const int verify_every =
            preset(g_onef_verify_every, "FRITCHII_ONEF_VERIFY_EVERY", 0);
        // EXP12=5, the lagged acquire check (fable_b 11.3), from boot. Loaded first, so the
        // experiments' own load does not replace it later.
        (void)Experiment(12);
        if (std::getenv("EDEN_NX2_EXP12") == nullptr) {
            g_experiments[12].store(5, std::memory_order_relaxed);
            BumpSwitchGeneration();
        }
        LOG_INFO(HW_GPU,
                 "One follower LEAN (FRITCHII_ONEF_LEAN=1): step 4 from boot, view and build "
                 "compares off, no forced verify; UNLEND={:#x} E_PIPELINE={} QUERY_OPS={} "
                 "WAITS_E={} INVAL_E={} VERIFY_EVERY={} EXP12={}",
                 unlend, e_pipeline, query_ops, waits_e, inval_e, verify_every, Experiment(12));
    }
    // Default 1 (the step 4 probes ran with it; 0 keeps the facts table path).
    g_e_pipeline = g.on && SwitchValue(g_onef_e_pipeline, "FRITCHII_ONEF_E_PIPELINE", 1) == 1;
    if (g_e_pipeline) {
        LOG_INFO(HW_GPU, "One follower: E owns the pipeline lookup (FRITCHII_ONEF_E_PIPELINE=1)");
        const char* const mesh_emu = std::getenv("FRITCHII_ONEF_MESH_EMU");
        LOG_INFO(HW_GPU, "One follower: emulated mesh draws {} (FRITCHII_ONEF_MESH_EMU={})",
                 EmulatedMeshPostable() ? "posted" : "lent",
                 mesh_emu != nullptr && mesh_emu[0] != '\0'
                     ? std::string{mesh_emu}
                     : std::string{IsAmdProprietaryDriver() ? "unset: AMD proprietary driver"
                                                            : "unset: not AMD's proprietary driver"});
    }
    if (g.on) {
        LOG_INFO(HW_GPU,
                 "One follower (FRITCHII_ONEF=1): the render thread R runs the renderer, every "
                 "draw and render operation lent; R {}pinned; the memory stage stays off",
                 g.pin ? "" : "not ");
    }
}

bool On() noexcept {
    return g.on;
}

StepLevel Step() noexcept {
    if (!g.on) {
        return StepLevel::Off;
    }
    const u32 generation =
        VideoCore::Nx2Debug::g_switch_generation.load(std::memory_order_acquire);
    if (generation != g.seen_generation.load(std::memory_order_acquire)) [[unlikely]] {
        Refresh(generation);
    }
    return static_cast<StepLevel>(g.step.load(std::memory_order_relaxed));
}

bool OnRenderThread() noexcept {
    return t_render;
}

bool RenderInLentCall() noexcept {
    return t_lent_depth != 0;
}

bool InLent() noexcept {
    return t_render && t_lent_depth != 0;
}

bool PlayingUnlentDraw() noexcept {
    return t_playing_draw;
}

void DisableUnlend(const char* why) noexcept {
    if (g.unlend_disabled.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    LOG_CRITICAL(HW_GPU,
                 "One follower: step 4 stopped for this session ({}); FRITCHII_ONEF_UNLEND runs 0, "
                 "every draw is lent again",
                 why != nullptr ? why : "?");
    g.seen_generation.store(~u32{0}, std::memory_order_release); // Refresh at the next Step()
}

void MarkEngineThread() noexcept {
    if (g.on) {
        Detail::t_engine = true;
        g.e_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    }
}

void MarkEngineHelperThread() noexcept {
    if (g.on) {
        Detail::t_engine = true;
        g.e2_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);
    }
}

void TakeEngineTls(EngineTls& out) noexcept {
    out.next_draw = t_next_draw;
    out.next_draw_set = t_next_draw_set;
    out.query_kind = t_query_kind;
    out.lend_site = t_lend_site;
    out.e_last = t_e_last;
    t_next_draw_set = false;
    t_query_kind = QueryKind::Count;
    t_lend_site = nullptr;
}

void PutEngineTls(const EngineTls& in) noexcept {
    t_next_draw = in.next_draw;
    t_next_draw_set = in.next_draw_set;
    t_query_kind = in.query_kind;
    t_lend_site = in.lend_site;
    t_e_last = in.e_last;
}

void Start(Core::System& system) {
    if (!g.on || g.thread.joinable()) {
        return;
    }
    g.system = &system;
    if (!g.ring) {
        g.ring = std::make_unique<u8[]>(RingBytes);
        g.positions = std::make_unique<u64[]>(PositionSlots);
        g.push_ns = std::make_unique<u64[]>(PositionSlots);
    }
    // A session starts with empty positions (GpuSplit's ring starts at 0 again).
    g.write = 0;
    g.pushed = 0;
    g.draw_seq = 0;
    g_carry.used = 0; // E is not running: nothing kept survives a session
    g_carry.count = 0;
    g_carry.floor = ~u64{0};
    g_carry.lands = 0;
    g.sent.store(0, std::memory_order_relaxed);
    g.sent_packets.store(0, std::memory_order_relaxed);
    g.intended.store(0, std::memory_order_relaxed);
    g.read_at = 0;
    g.freed.store(0, std::memory_order_relaxed);
    g.played.store(0, std::memory_order_relaxed);
    g.engine.store(nullptr, std::memory_order_relaxed);
    View() = RegisterView{};
    Detail::ViewTrackingStopped();
    g.seen_generation.store(~u32{0}, std::memory_order_relaxed);
    (void)Step();
    if (g_trace && !g_trace_ring) {
        g_trace_ring = std::make_unique<TraceRecord[]>(TraceSlots);
        const auto path =
            Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "onef_lend_trace.bin";
        g_trace_file.open(path, std::ios::binary | std::ios::trunc);
        LOG_INFO(HW_GPU, "One follower lend trace (FRITCHII_ONEF_TRACE=1): {} ({} MiB ring)",
                 path.string(), TraceSlots * sizeof(TraceRecord) >> 20);
    }
    g.thread = std::jthread([](std::stop_token stop) { RenderMain(stop); });
    g.watchdog = std::jthread([](std::stop_token stop) { WatchMain(stop); });
}

void ReportHangState() {
    const u64 now_ns = Common::Nx2Hitch::NowNs();
    const u64 begin = GpuSplit::Get().FrontWaitNow().begin_ns.load(std::memory_order_acquire);
    // The front's wait (GpuSplit, with or without the one follower); its acquires are in the
    // report's own line (GPU::LogHangState).
    ReportFrontAndAcquires("Hang report: GPU split",
                           begin != 0 && now_ns - begin > 2'000'000'000ULL, false, now_ns, false);
    if (!g.on || !g.thread.joinable()) {
        LOG_CRITICAL(HW_GPU, "Hang report: one follower not running (FRITCHII_ONEF off): no E/R "
                             "packet state");
        return;
    }
    LOG_CRITICAL(HW_GPU, "Hang report: one follower {}; watchdog wakes given {}", StateText(),
                 g.rescued.load(std::memory_order_relaxed));
}

void Stop() {
    if (!g.thread.joinable()) {
        return;
    }
    if (g.watchdog.joinable()) {
        g.watchdog.request_stop();
        g.watchdog.join();
    }
    g.thread.request_stop();
    g.r_wake.fetch_add(1, std::memory_order_seq_cst);
    g.r_wake.notify_one();
    g.thread.join();
    TraceFlush(true);
    if (g_trace_file.is_open()) {
        g_trace_file.close();
    }
}

void WatchLock(const char* name, const Common::SpinRecursiveMutex* mutex) noexcept {
    const u32 count = g.lock_count.load(std::memory_order_relaxed);
    for (u32 i = 0; i < count; ++i) {
        if (std::strcmp(g.locks[i].first, name) == 0) {
            g.locks[i].second = mutex; // a new rasterizer (session): its lock replaces the old
            return;
        }
    }
    const u32 index = count;
    if (index >= g.locks.size()) {
        return;
    }
    g.locks[index] = {name, mutex};
    g.lock_count.store(index + 1, std::memory_order_release);
}

void SetNextDraw(u32 kind, bool indexed, bool indirect, u32 instance_arg) noexcept {
    t_next_draw = DrawValues{};
    t_next_draw.kind = kind;
    t_next_draw.indexed = indexed ? 1 : 0;
    t_next_draw.indirect = indirect ? 1 : 0;
    t_next_draw.instance_arg = instance_arg;
    t_next_draw_set = true;
}

void ClearNextDraw() noexcept {
    t_next_draw_set = false;
}

namespace {
void PushPacked(PacketHeader& header, u8* buffer, std::size_t body_bytes);
void SendCarriedAlone(size_t why);
} // Anonymous namespace

void FlushCarried() noexcept {
    if (g_carry.count != 0 && Detail::t_engine && g.thread.joinable()) {
        // FRITCHII_ESPLIT S2: one producer
        const ESplit::PushGuard esplit_push{false, false, ESplit::PushKind::Carried};
        SendCarriedAlone(0);
    }
}

void Lend(LentOp op, LentReason reason, void (*function)(void*), void* argument) {
    if (!Detail::t_engine || !g.thread.joinable()) {
        if (op == LentOp::Draw) {
            t_next_draw_set = false;
        }
        t_lend_site = nullptr;
        function(argument); // R, another thread, or R not running: in place, as today
        return;
    }
    // FRITCHII_ESPLIT (S2): E1 drains E2 first (one producer, the flags on the engine); a lend
    // of E2's holds E1 at a safe point while R runs it (fritchii_e_split_thread.h).
    const ESplit::PushGuard esplit_push{true, op == LentOp::Fence || op == LentOp::Query,
                                        ESplit::PushKind::Lend};
    // FRITCHII_ESPLIT_COUNT: the whole lend (pack, park, R's run) is the "lend" class.
    const ESplit::LendScope esplit_lend;
    const StepLevel step = Step();
    const bool timing = Stats::Timing();
    const u64 begin = timing ? Stats::Now() : 0;
    const bool counting = CountersOn();
    const u64 begin_clock = counting ? Common::Nx2Hitch::NowNs() : 0;
    const u64 behind = counting ? Behind() : 0;
    const u64 number = ++g.pushed;
    if (g_trace) [[unlikely]] {
        TraceArrive(number, t_lend_site, op, false, Common::Nx2Hitch::NowNs());
    }
    u8* const buffer = PackBuffer();
    std::size_t spans_bytes = 0;
    PacketHeader header{};
    header.kind = PacketKind::Lent;
    header.op = static_cast<u8>(op);
    header.reason = static_cast<u8>(reason);
    header.number = number;
    header.position = GpuSplit::Get().BackEntryPosition();
    header.draw_seq = op == LentOp::Draw ? ++g.draw_seq : g.draw_seq;
    header.call = LentCall{function, argument};
    const u32 kind = op == LentOp::Draw ? TakeDrawKind(header) : 0;
    // FRITCHII_ESPLIT S2: a lend of E2's while E1 runs ahead packs nothing of E1's (R's view
    // stays at E2's place; E1's changes go with its next packet).
    if (step >= StepLevel::View && !(ESplit::g_async && ESplit::OnE2())) {
        if (Maxwell3D* const engine = g.engine.load(std::memory_order_acquire)) {
            // The spans: everything E changed since the last packet, after the header.
            spans_bytes = PackSpans(*engine, header,
                                    std::span<u8>{buffer + sizeof(PacketHeader), MaxSpanBytes});
            if (op == LentOp::Draw && !g.lean) {
                header.flags |= PacketVerify; // FRITCHII_ONEF_LEAN: no view compare
            }
        }
    }
    if (op == LentOp::Draw) {
        t_next_draw_set = false;
    }
    u64 floor = header.position;
    spans_bytes = AppendCarried(header, buffer, spans_bytes, floor);
    header.bytes = static_cast<u32>(sizeof(PacketHeader) + spans_bytes);
    std::memcpy(buffer, &header, sizeof(header));
    Commit(buffer, header.bytes, number, floor, false);
    CarriedSent(number);
    if (counting) {
        Stats::Pushed(header, kind, behind);
    }
    // E never parks holding a lock R may take (the deadlock of 12 Nov: E parked in a seam play's
    // lend holding the buffer cache's lock, R in an earlier draw packet waiting for it): a
    // watched lock E holds goes to R for the park (R takes it as its own, recursively; guest
    // threads stay out) and comes back after. R reaching into the buffer cache outside the lent
    // call still counts as "owner on R" (its entry points' asserts name where).
    std::array<Common::SpinRecursiveMutex*, 4> handed{};
    u32 handed_count = 0;
    {
        const std::thread::id self = std::this_thread::get_id();
        const std::thread::id render = g.r_thread.load(std::memory_order_relaxed);
        const u32 lock_count = g.lock_count.load(std::memory_order_acquire);
        for (u32 i = 0; i < lock_count && handed_count < handed.size(); ++i) {
            auto* const mutex = const_cast<Common::SpinRecursiveMutex*>(g.locks[i].second);
            if (mutex != nullptr && mutex->Owner() == self) {
                mutex->HandOver(render);
                handed[handed_count++] = mutex;
            }
        }
        if (handed_count != 0) {
            g.lends_holding_lock.fetch_add(1, std::memory_order_relaxed);
        }
    }
    SCOPE_EXIT {
        const std::thread::id self = std::this_thread::get_id();
        for (u32 i = 0; i < handed_count; ++i) {
            handed[i]->HandOver(self);
        }
    };
    // The park: R plays everything before, runs the call, releases E. Its own profile zone
    // (fable_b 14.1): without it the park landed in whatever zone the lend was made in ("other"
    // for a call entry, "method" for a method's).
    g.e_wait_op.store(static_cast<u8>(op), std::memory_order_relaxed);
    {
        const Nx2Profile::Scope parked_zone{Nx2Profile::Zone::OnefParked};
        EngineWait([&] { return g.played.load(std::memory_order_acquire) >= number; },
                   EWait::Park, number);
    }
    CheckRaiseAfterLend();
    if (g_trace) [[unlikely]] {
        const u64 now = Common::Nx2Hitch::NowNs();
        g_trace_ring[number % TraceSlots].e_resume = now;
        t_e_last = now;
    }
    const u64 parked_ns = counting ? Common::Nx2Hitch::NowNs() - begin_clock : 0;
    if (counting) {
        NoteLendSite(op, parked_ns);
    } else {
        t_lend_site = nullptr;
        t_query_kind = QueryKind::Count; // consumed uncounted
    }
    if (counting && op == LentOp::Query && t_query_kind != QueryKind::Count) {
        const size_t kind_index = static_cast<size_t>(t_query_kind);
        g.query_count[kind_index].fetch_add(1, std::memory_order_relaxed);
        g.query_parked_ns[kind_index].fetch_add(parked_ns, std::memory_order_relaxed);
        t_query_kind = QueryKind::Count;
    }
    if (timing) {
        Stats::Parked(op, Stats::Now() - begin);
    }
    if (g.lent_error) [[unlikely]] {
        std::rethrow_exception(std::exchange(g.lent_error, nullptr));
    }
    if (g.op_error) [[unlikely]] {
        std::rethrow_exception(std::exchange(g.op_error, nullptr));
    }
}

void PushOp(LentOp op, void (*function)(void*), const void* payload, u32 size,
            bool lands_in_memory, u8 packet_flags, bool carry) {
    if (!Detail::t_engine || !g.thread.joinable() || size > MaxPayloadBytes) {
        if (size > MaxPayloadBytes && Detail::t_engine && g.thread.joinable()) [[unlikely]] {
            // An R op run on E: a bug in the op's payload, never silent (its effects are R's).
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_CRITICAL(HW_GPU, "One follower: an op payload of {} bytes (max {}) at {} ran on E",
                             size, MaxPayloadBytes, t_lend_site != nullptr ? t_lend_site : "?");
            }
        }
        t_lend_site = nullptr;
        function(const_cast<void*>(payload));
        return;
    }
    // FRITCHII_ESPLIT_ASYNC_KINDS bit 3: a posted op from E1 queues. E1 packs the spans (its
    // registers at the op's place) and the payload; E2 commits the packet in ring order (R's
    // queue keeps E2 as its one producer), as an async draw's packet. No drain.
    if (ESplit::QueueKind(ESplit::KindPostedOps) && !(carry && packet_flags == 0 && CarryOpsOn()))
        [[unlikely]] {
        if (op == LentOp::Query && t_query_kind != QueryKind::Count) {
            g.query_count[static_cast<size_t>(t_query_kind)].fetch_add(1,
                                                                       std::memory_order_relaxed);
            t_query_kind = QueryKind::Count;
        }
        u8* buffer = nullptr;
        void* const slot = ESplit::BeginAsyncPacket(ESplit::Hand::Op, buffer);
        struct QueuedOp {
            PacketHeader header;
            u8* buffer;
            const char* trace_site;
            std::size_t body_bytes;
            LentOp op;
            bool lands_in_memory;
        } job{};
        job.header.kind = PacketKind::Op;
        job.header.flags = packet_flags;
        job.header.op = static_cast<u8>(op);
        job.header.call = LentCall{function, nullptr};
        std::size_t spans_bytes = 0;
        if (Step() >= StepLevel::View) {
            if (Maxwell3D* const engine = g.engine.load(std::memory_order_acquire)) {
                spans_bytes = PackSpans(*engine, job.header,
                                        std::span<u8>{buffer + sizeof(PacketHeader), MaxSpanBytes});
            }
        }
        job.header.reserved2 = static_cast<u32>(spans_bytes);
        std::memcpy(buffer + sizeof(PacketHeader) + spans_bytes, payload, size);
        job.buffer = buffer;
        job.trace_site = std::exchange(t_lend_site, nullptr);
        job.body_bytes = spans_bytes + ((size + 7) & ~u32{7});
        job.op = op;
        job.lands_in_memory = lands_in_memory;
        static_assert(sizeof(QueuedOp) <= ESplit::AsyncPayloadBytes);
        static_assert(std::is_trivially_copyable_v<QueuedOp>);
        std::memcpy(slot, &job, sizeof(job));
        ESplit::PostAsync([](void* raw) {
            QueuedOp q;
            std::memcpy(&q, raw, sizeof(q));
            PushPacked(q.header, q.buffer, q.body_bytes);
            if (g_trace) [[unlikely]] {
                TraceArrive(q.header.number, q.trace_site, q.op, true, Common::Nx2Hitch::NowNs());
            }
            if (q.lands_in_memory) {
                g.guest_write_op.store(q.header.position + 1, std::memory_order_release);
                g.guest_write_number = q.header.number;
            }
        });
        return;
    }
    const ESplit::PushGuard esplit_push{false, op == LentOp::Fence || op == LentOp::Query,
                                        ESplit::PushKind::Op};
    if (op == LentOp::Query && t_query_kind != QueryKind::Count) {
        g.query_count[static_cast<size_t>(t_query_kind)].fetch_add(1, std::memory_order_relaxed);
        t_query_kind = QueryKind::Count;
    }
    if (carry && packet_flags == 0 && CarryOpsOn()) {
        // FRITCHII_ONEF_CARRY_OPS: kept for the next packet (no spans: the op reads nothing of
        // R's view, and E's changes since the last packet go with the next one).
        const std::size_t record_bytes = sizeof(CarryRecord) + ((size + 7) & ~u32{7});
        if (g_carry.used + record_bytes > MaxCarryBytes) [[unlikely]] {
            SendCarriedAlone(2);
        }
        const u64 position = GpuSplit::Get().BackEntryPosition();
        const CarryRecord record{function, size, 0};
        u8* const at = g_carry.bytes.data() + g_carry.used;
        std::memcpy(at, &record, sizeof(record));
        std::memcpy(at + sizeof(record), payload, size);
        g_carry.used += static_cast<u32>(record_bytes);
        ++g_carry.count;
        g_carry.floor = (std::min)(g_carry.floor, position);
        if (lands_in_memory) {
            g_carry.lands = position + 1;
        }
        t_lend_site = nullptr;
        return;
    }
    u8* const buffer = PackBuffer();
    PacketHeader header{};
    header.kind = PacketKind::Op;
    header.flags = packet_flags;
    header.op = static_cast<u8>(op);
    header.call = LentCall{function, nullptr};
    std::size_t spans_bytes = 0;
    if (Step() >= StepLevel::View && !(ESplit::g_async && ESplit::OnE2())) {
        if (Maxwell3D* const engine = g.engine.load(std::memory_order_acquire)) {
            // E's changes before the op, so R's view is at the op's place when it runs.
            spans_bytes = PackSpans(*engine, header,
                                    std::span<u8>{buffer + sizeof(PacketHeader), MaxSpanBytes});
        }
    }
    header.reserved2 = static_cast<u32>(spans_bytes);
    std::memcpy(buffer + sizeof(PacketHeader) + spans_bytes, payload, size);
    const char* const trace_site = std::exchange(t_lend_site, nullptr);
    PushPacked(header, buffer, spans_bytes + ((size + 7) & ~u32{7}));
    if (g_trace) [[unlikely]] {
        TraceArrive(header.number, trace_site, op, true, Common::Nx2Hitch::NowNs());
    }
    if (lands_in_memory) {
        // Before E's next `consumed` store (seq_cst there): a front that sees E past this entry
        // sees this mark too.
        g.guest_write_op.store(header.position + 1, std::memory_order_release);
        g.guest_write_number = header.number;
    }
}

bool WaitGuestWriteOps() noexcept {
    if (!Detail::t_engine) {
        return false;
    }
    const u64 number = g.guest_write_number;
    if (number == 0 || g.played.load(std::memory_order_acquire) >= number) {
        return false;
    }
    EngineWait([&] { return g.played.load(std::memory_order_acquire) >= number; }, EWait::Drain,
               number);
    return true;
}

bool WaitsE() noexcept {
    return g.on &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_waits_e,
                                            "FRITCHII_ONEF_WAITS_E") != 0 &&
           !WaitsR();
}

bool WaitsR() noexcept {
    return g.on && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_waits_r,
                                                     "FRITCHII_ONEF_WAITS_R") != 0;
}

u64 GuestWriteOpMark() noexcept {
    return g.guest_write_op.load(std::memory_order_acquire);
}

bool QueryOps() noexcept {
    using namespace VideoCore::Nx2Debug;
    // From step 3 only: below it R's objects read E's engine, which a render op (E not parked)
    // must not.
    const bool on = g.on &&
                    (SwitchValue(g_onef_query_ops, "FRITCHII_ONEF_QUERY_OPS") != 0 ||
                     SwitchValue(g_onef_posted, "FRITCHII_ONEF_POSTED") != 0) &&
                    Step() >= StepLevel::Build;
    if (on && !g.query_ops_logged.load(std::memory_order_relaxed) &&
        !g.query_ops_logged.exchange(true)) [[unlikely]] {
        // fable_b 11.3: with reports landing on R the full-drain acquire poll (EXP12 = 4) drains
        // to R; the lagged check (EXP12 = 5) is the mode to run with, its target R's position
        // (`played`). Not switched here: the wait mode of acquires already suspended does not
        // change under them (the 4 -> 5 flip went with the first QUERY_OPS hang). Set it from
        // boot.
        const int exp12 = Experiment(12);
        if (exp12 != 5 && exp12 != 6) {
            LOG_WARNING(HW_GPU,
                        "One follower: FRITCHII_ONEF_QUERY_OPS=1 with EXP12={}: set EXP12=5 from "
                        "boot (the lagged acquire check, fable_b 11.3)",
                        exp12);
        }
        LOG_INFO(HW_GPU,
                 "One follower: FRITCHII_ONEF_QUERY_OPS=1 (render enables that cannot set host "
                 "conditional rendering not sent, resets and plain reports as render ops); EXP12 "
                 "{}",
                 exp12);
    }
    return on;
}

bool InvalOnEngine() noexcept {
    using namespace VideoCore::Nx2Debug;
    if (!g.on || SwitchValue(g_onef_inval_e, "FRITCHII_ONEF_INVAL_E") == 0) {
        return false;
    }
    if (!g_e_pipeline) {
        // Exact only while E owns the shader registrations (rule 3): R's own lookups would mark
        // the shader filter without E seeing it in order.
        if (!g.inval_refused_logged.exchange(true, std::memory_order_relaxed)) {
            LOG_WARNING(HW_GPU, "One follower: FRITCHII_ONEF_INVAL_E=1 needs "
                                "FRITCHII_ONEF_E_PIPELINE=1 (boot); running 0");
        }
        return false;
    }
    if (Step() != StepLevel::Unlend) {
        return false;
    }
    if (!g.stamps_on.load(std::memory_order_acquire)) [[unlikely]] {
        // The first call with the switch on: R starts reading the table once `stamps_on` shows
        // it; never freed while the process runs (R may be reading it).
        static std::once_flag once;
        std::call_once(once, [] {
            g.stamps = std::make_unique<std::atomic<u32>[]>(StampSlots);
            g.stamps_on.store(true, std::memory_order_release);
            LOG_INFO(HW_GPU,
                     "One follower: page write stamps on (FRITCHII_ONEF_INVAL_E=1, {} MiB)",
                     StampSlots * sizeof(u32) >> 20);
        });
    }
    return true;
}

bool SyncMirrorOn() noexcept {
    return g.on && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_sync_mirror,
                                                    "FRITCHII_ONEF_SYNC_MIRROR") != 0;
}

bool LookupCheckDue() noexcept {
    const int every = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_lookup_check,
                                                       "FRITCHII_ONEF_LOOKUP_CHECK");
    return g.on && every > 0 && ++g.lookup_checks % static_cast<u64>(every) == 0;
}

namespace {
/// fable_b 19.5 step 3's dry price: a release per slot (by its word), as E passed it and as
/// some thread wrote it. Racy by design (a counter): E stamps the pass, the writer the run, the
/// front reads both at a resume.
struct DrySlot {
    std::atomic<u64> device{~u64{0}};
    std::atomic<u64> pass_ns{};
    std::atomic<u64> pass_stall{};
    std::atomic<u64> run_ns{};
    std::atomic<u64> run_stall{};
    std::atomic<u8> kind{};
};
constexpr size_t DrySlots = 4096;
std::array<DrySlot, DrySlots> g_dry_slots{};
std::atomic<u64> g_dry_stall_total{};
std::atomic<u64> g_dry_stall_since{};
DrySlot& DrySlotFor(u64 device) {
    return g_dry_slots[(device >> 2) % DrySlots];
}
/// The front's window (front thread only).
struct DryWindow {
    u64 since_ns{};
    u64 last_ns{};
    std::array<u64, static_cast<size_t>(DryKind::Count)> count{};
    std::array<u64, static_cast<size_t>(DryKind::Count)> wall_ns{};
    std::array<u64, static_cast<size_t>(DryKind::Count)> stalled_ns{};
    u64 resumes{};
    u64 not_passed{}; ///< no release E passed for the word
    u64 not_run{};    ///< E passed one, not written yet (met another way)
    u64 before{};     ///< written before the suspension
};
DryWindow g_dry_window{};
} // Anonymous namespace

bool DryReleaseOn() noexcept {
    const bool on = g.on && VideoCore::Nx2Debug::SwitchValue(
                                VideoCore::Nx2Debug::g_onef_dry_release,
                                "FRITCHII_ONEF_DRY_RELEASE") != 0;
    if (g_dry_release.load(std::memory_order_relaxed) != on) {
        g_dry_release.store(on, std::memory_order_relaxed);
    }
    return on;
}

void DryNotePass(u64 device, DryKind kind) noexcept {
    DrySlot& slot = DrySlotFor(device);
    slot.run_ns.store(0, std::memory_order_relaxed);
    slot.kind.store(static_cast<u8>(kind), std::memory_order_relaxed);
    slot.pass_stall.store(DryStallNow(), std::memory_order_relaxed);
    slot.pass_ns.store(Common::Nx2Hitch::NowNs(), std::memory_order_relaxed);
    slot.device.store(device, std::memory_order_release);
}

void DryNoteRun(u64 device) noexcept {
    DrySlot& slot = DrySlotFor(device);
    if (slot.device.load(std::memory_order_acquire) != device ||
        slot.run_ns.load(std::memory_order_relaxed) != 0) {
        return;
    }
    slot.run_stall.store(DryStallNow(), std::memory_order_relaxed);
    slot.run_ns.store(Common::Nx2Hitch::NowNs(), std::memory_order_release);
}

u64 DryStallNow() noexcept {
    const u64 since = g_dry_stall_since.load(std::memory_order_acquire);
    const u64 total = g_dry_stall_total.load(std::memory_order_acquire);
    return since == 0 ? total : total + (Common::Nx2Hitch::NowNs() - since);
}

void DryFrontStall(bool begin) noexcept {
    const u64 now = Common::Nx2Hitch::NowNs();
    if (begin) {
        g_dry_stall_since.store(now, std::memory_order_release);
        return;
    }
    const u64 since = g_dry_stall_since.exchange(0, std::memory_order_acq_rel);
    if (since != 0) {
        g_dry_stall_total.fetch_add(now - since, std::memory_order_acq_rel);
    }
}

void DryNoteResume(u64 device, u64 since_ns, u64 stall_at_since) noexcept {
    auto& w = g_dry_window;
    const u64 now = Common::Nx2Hitch::NowNs();
    if (w.since_ns != 0 && now - w.last_ns > 5'000'000'000ULL) {
        w = {}; // the switch was off, or nothing resumed: start over
    }
    if (w.since_ns == 0) {
        w.since_ns = now;
    }
    w.last_ns = now;
    ++w.resumes;
    DrySlot& slot = DrySlotFor(device);
    const u64 run_ns = slot.run_ns.load(std::memory_order_acquire);
    if (slot.device.load(std::memory_order_acquire) != device) {
        ++w.not_passed;
    } else if (run_ns == 0) {
        ++w.not_run;
    } else if (run_ns < since_ns) {
        ++w.before;
    } else {
        const size_t kind = (std::min)(static_cast<size_t>(slot.kind.load(std::memory_order_relaxed)),
                                       static_cast<size_t>(DryKind::Count) - 1);
        const u64 pass_ns = (std::max)(slot.pass_ns.load(std::memory_order_relaxed), since_ns);
        const u64 pass_stall =
            (std::max)(slot.pass_stall.load(std::memory_order_relaxed), stall_at_since);
        const u64 run_stall = slot.run_stall.load(std::memory_order_relaxed);
        ++w.count[kind];
        w.wall_ns[kind] += run_ns > pass_ns ? run_ns - pass_ns : 0;
        w.stalled_ns[kind] += run_stall > pass_stall ? run_stall - pass_stall : 0;
    }
    const double window = static_cast<double>(now - w.since_ns) / 1e9;
    if (window < 5.0) {
        return;
    }
    static constexpr std::array<const char*, static_cast<size_t>(DryKind::Count)> names{
        "payload release", "fence-flagged payload", "counter report", "QMD reduction",
        "QMD max"};
    std::string kinds;
    double stalled_all = 0;
    for (size_t k = 0; k < names.size(); ++k) {
        if (w.count[k] == 0) {
            continue;
        }
        const double stalled = static_cast<double>(w.stalled_ns[k]) / 1e6 / window;
        stalled_all += stalled;
        kinds += fmt::format("{}{} {:.0f}/s (E ahead of the write {:.1f} ms/s, the front "
                             "stalled in it {:.1f} ms/s)",
                             kinds.empty() ? "" : "; ", names[k],
                             static_cast<double>(w.count[k]) / window,
                             static_cast<double>(w.wall_ns[k]) / 1e6 / window, stalled);
    }
    LOG_INFO(HW_GPU,
             "One follower dry release price (fable_b 19.5 step 3): {:.0f} resumed acquires/s; "
             "met after the suspension by a release E passed: {}; price (front stalled while "
             "E was past the release and it was not written) {:.1f} ms/s; no release E passed "
             "{:.0f}/s, passed but not written {:.0f}/s, written before the suspension "
             "{:.0f}/s",
             static_cast<double>(w.resumes) / window, kinds.empty() ? "none" : kinds,
             stalled_all, static_cast<double>(w.not_passed) / window,
             static_cast<double>(w.not_run) / window, static_cast<double>(w.before) / window);
    const u64 last = w.last_ns;
    w = {};
    w.since_ns = now;
    w.last_ns = last;
}

bool SemOnE() noexcept {
    const bool on = PostFenceOn() && VideoCore::Nx2Debug::SwitchValue(
                                         VideoCore::Nx2Debug::g_onef_sem_e,
                                         "FRITCHII_ONEF_SEM_E") != 0;
    if (g_sem_e.load(std::memory_order_relaxed) != on) {
        g_sem_e.store(on, std::memory_order_relaxed);
    }
    return on;
}

bool SignalOnR() noexcept {
    return g.on && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_signal_on_r,
                                                     "FRITCHII_ONEF_SIGNAL_ON_R") != 0;
}

void CountSignalOnR() noexcept {
    g.signals_on_r.fetch_add(1, std::memory_order_relaxed);
}

void NoteReportLandsAtLastPush(u64 device) noexcept {
    // The op is the newest packet E pushed (no carry): its number is `pushed`. Erased once R
    // played it (WaitReportLanding) or at the next lent fence (NoteFenceLent).
    const u64 number = g.pushed;
    g.landing_pages[device >> 12] = number;
    g.landing_max = (std::max)(g.landing_max, number);
    g_report_landing_watch.store(true, std::memory_order_relaxed);
}

bool SoftReleaseOnE() noexcept {
    return g.on && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_soft_release_e,
                                                    "FRITCHII_ONEF_SOFT_RELEASE_E") != 0;
}

int PostIdleOn() noexcept {
    if (!PostFenceOn()) {
        return 0;
    }
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_post_idle,
                                            "FRITCHII_ONEF_POST_IDLE");
}

bool RenderCaughtUp() noexcept {
    return g.played.load(std::memory_order_acquire) >= g.pushed;
}

bool LandingOutstanding() noexcept {
    return g.landing_max > g.played.load(std::memory_order_acquire);
}

void NoteEngineSignal(const char* what) noexcept {
    if (LandingOutstanding()) [[unlikely]] {
        thread_local std::string note;
        note = fmt::format("{} on E while R has not played packet {} (it lands reports; R at {})",
                           what, g.landing_max, g.played.load(std::memory_order_relaxed));
        Note(Check::SignalAheadOfLanding, note.c_str());
    }
}

void CountIdleDecision(IdleDecision decision) noexcept {
    g.idle_decisions[static_cast<size_t>(decision)].fetch_add(1, std::memory_order_relaxed);
}

bool PostFenceReportOn() noexcept {
    return PostFenceOn() &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_post_fence_report,
                                            "FRITCHII_ONEF_POST_FENCE_REPORT") != 0;
}

const std::vector<u64>& UnassignedReportPages() noexcept {
    return g.landing_unassigned;
}

void CountFenceDecision(FenceDecision decision) noexcept {
    g.fence_decisions[static_cast<size_t>(decision)].fetch_add(1, std::memory_order_relaxed);
}

FenceDecision PostFenceRefusal() noexcept {
    if (g.post_fence_stopped.load(std::memory_order_relaxed)) {
        return FenceDecision::Stopped;
    }
    if (VideoCore::Nx2Debug::DelayedFences()) {
        return FenceDecision::Delayed;
    }
    return FenceDecision::Off;
}

void CountSyncTargetSuperset() noexcept {
    g.sync_target_superset.fetch_add(1, std::memory_order_relaxed);
}

namespace {
/// fable_b 19.6's "semaphore order": the last write's place per word (a direct-mapped table; a
/// collision forgets the older word).
struct SemOrderSlot {
    std::atomic<u64> device{~u64{0}};
    std::atomic<u64> place{};
};
std::array<SemOrderSlot, 4096> g_sem_order{};
} // Anonymous namespace

void NoteSemaphoreOrder(u64 device) noexcept {
    // E's place is after the packets it pushed (n, then 1); R's the packet it plays (n, then 0):
    // a packet E pushes comes after E's writes before it and before E's writes after it. The
    // packet numbers are compared on their low 32 bits, modulo.
    const bool engine = Detail::t_engine;
    const u32 number = engine ? static_cast<u32>(g.pushed) : t_play_number;
    const u64 place = (u64{number} << 1) | (engine ? 1 : 0);
    SemOrderSlot& slot = g_sem_order[(device >> 2) % g_sem_order.size()];
    if (slot.device.load(std::memory_order_acquire) == device) {
        const u64 last = slot.place.load(std::memory_order_relaxed);
        const s32 ahead = static_cast<s32>(number - static_cast<u32>(last >> 1));
        if (ahead < 0 || (ahead == 0 && (place & 1) < (last & 1))) [[unlikely]] {
            thread_local std::string note;
            note = fmt::format("word {:#x}: {} wrote it at packet {} after {} wrote it at {}",
                               device, engine ? "E" : "R", number, (last & 1) != 0 ? "E" : "R",
                               last >> 1);
            Note(Check::SemaphoreOrder, note.c_str());
        }
    }
    slot.place.store(place, std::memory_order_relaxed);
    slot.device.store(device, std::memory_order_release);
}

void StopPostFence(const char* why) noexcept {
    if (!g.post_fence_stopped.exchange(true, std::memory_order_relaxed)) {
        LOG_CRITICAL(HW_GPU, "One follower: FRITCHII_ONEF_POST_FENCE stops for this session ({})",
                     why != nullptr ? why : "?");
    }
}

u32 DispatchPostBits() noexcept {
    return static_cast<u32>((std::max)(
        0, VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_op_dispatch,
                                            "FRITCHII_ONEF_OP_DISPATCH")));
}

DispatchCount PostDispatchBlocker(bool& off) noexcept {
    using namespace VideoCore::Nx2Debug;
    off = !g.on || Step() != StepLevel::Unlend ||
          SwitchValue(g_onef_op_dispatch, "FRITCHII_ONEF_OP_DISPATCH") == 0;
    if (off) {
        return DispatchCount::Count;
    }
    if (!g_e_pipeline) {
        // The compute lookup and its OoT hooks run on E (rule 3's reasoning, compute side): with
        // R owning the pipeline cache E may not look up.
        if (!g.dispatch_refused_logged.exchange(true, std::memory_order_relaxed)) {
            LOG_WARNING(HW_GPU, "One follower: FRITCHII_ONEF_OP_DISPATCH=1 needs "
                                "FRITCHII_ONEF_E_PIPELINE=1 (boot); every dispatch stays lent");
        }
        return DispatchCount::NeedsEPipeline;
    }
    if (g.post_dispatch_stopped.load(std::memory_order_relaxed)) {
        return DispatchCount::Stopped;
    }
    if (!ViewReady()) {
        return DispatchCount::Transient;
    }
    return DispatchCount::Count;
}

void StopPostDispatch(const char* why) noexcept {
    if (!g.post_dispatch_stopped.exchange(true, std::memory_order_relaxed)) {
        LOG_CRITICAL(HW_GPU,
                     "One follower: FRITCHII_ONEF_OP_DISPATCH stops for this session ({}); every "
                     "dispatch is lent again",
                     why != nullptr ? why : "?");
    }
}

bool DispatchRangesOn() noexcept {
    return g.on &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_dispatch_ranges,
                                            "FRITCHII_ONEF_DISPATCH_RANGES") != 0 &&
           !g.dispatch_ranges_stopped.load(std::memory_order_relaxed);
}

void StopDispatchRanges(const char* why) noexcept {
    if (!g.dispatch_ranges_stopped.exchange(true, std::memory_order_relaxed)) {
        LOG_CRITICAL(HW_GPU,
                     "One follower: FRITCHII_ONEF_DISPATCH_RANGES stops for this session ({}); "
                     "image-writing dispatches record \"may write anything\" again",
                     why != nullptr ? why : "?");
    }
}

void NoteFrontReadPage(u64 page) noexcept {
    if (!g.on) {
        return;
    }
    PageFilter* filter = g.front_read_pages.load(std::memory_order_acquire);
    if (filter == nullptr) [[unlikely]] {
        static std::once_flag once;
        std::call_once(once, [] {
            g.front_read_pages.store(new PageFilter, std::memory_order_release);
        });
        filter = g.front_read_pages.load(std::memory_order_acquire);
    }
    filter->Mark(page << 12, u64{1} << 12);
}

bool FrontReadPagesAny(u64 address, u64 size) noexcept {
    const PageFilter* const filter = g.front_read_pages.load(std::memory_order_acquire);
    return filter != nullptr && filter->Any(address, size);
}

void CountDispatch(DispatchCount kind) noexcept {
    g.dispatch_count[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

u64 PushedNow() noexcept {
    return g.pushed;
}

void NoteReportPage(u64 device) noexcept {
    const u64 page = device >> 12;
    if (g.landing_unassigned.empty() || g.landing_unassigned.back() != page) {
        g.landing_unassigned.push_back(page);
    }
}

void NoteFencePosted(u64 number) noexcept {
    if (g.landing_unassigned.empty()) {
        return;
    }
    const u64 played = g.played.load(std::memory_order_acquire);
    if (g.landing_pages.size() > 4096) {
        std::erase_if(g.landing_pages, [played](const auto& entry) { return entry.second <= played; });
    }
    for (const u64 page : g.landing_unassigned) {
        g.landing_pages[page] = number;
    }
    g.landing_unassigned.clear();
    g.landing_max = (std::max)(g.landing_max, number);
    g_report_landing_watch.store(true, std::memory_order_relaxed);
}

void NoteFenceLent() noexcept {
    // R played everything before the lend, and a fence it ran there landed its reports.
    g.landing_pages.clear();
    g.landing_unassigned.clear();
    g_report_landing_watch.store(false, std::memory_order_relaxed);
}

bool ReportLandingClear(u64 address, u64 size) noexcept {
    if (size == 0) {
        return true;
    }
    const u64 first = address >> 12;
    const u64 last = (address + size - 1) >> 12;
    for (const u64 page : g.landing_unassigned) {
        if (page >= first && page <= last) {
            return false;
        }
    }
    if (g.landing_pages.empty()) {
        return true;
    }
    const u64 played = g.played.load(std::memory_order_acquire);
    for (u64 page = first; page <= last; ++page) {
        const auto it = g.landing_pages.find(page);
        if (it != g.landing_pages.end() && played < it->second) {
            return false;
        }
    }
    return true;
}

void WaitReportLanding(u64 address, u64 size) noexcept {
    if (size == 0 || g.landing_pages.empty()) {
        g_report_landing_watch.store(!g.landing_pages.empty(), std::memory_order_relaxed);
        return;
    }
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        const auto it = g.landing_pages.find(page);
        if (it == g.landing_pages.end()) {
            continue;
        }
        const u64 number = it->second;
        if (g.played.load(std::memory_order_acquire) < number) {
            g.landing_waits.fetch_add(1, std::memory_order_relaxed);
            EngineWait([&] { return g.played.load(std::memory_order_acquire) >= number; },
                       EWait::Drain, number);
        }
        g.landing_pages.erase(it);
    }
    g_report_landing_watch.store(!g.landing_pages.empty(), std::memory_order_relaxed);
}

bool PostFenceOn() noexcept {
    return g.on && Step() == StepLevel::Unlend &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_post_fence,
                                            "FRITCHII_ONEF_POST_FENCE") != 0 &&
           !VideoCore::Nx2Debug::DelayedFences() &&
           !g.post_fence_stopped.load(std::memory_order_relaxed);
}

void CountSyncMirror(bool engine_bit, bool render_bit) noexcept {
    g.mirror_count[0].fetch_add(1, std::memory_order_relaxed);
    const bool skipped = std::exchange(g.skipped_samples_draw, false); // R: since its last report
    if (engine_bit == render_bit) {
        return;
    }
    if (render_bit && !g.post_fence_stopped.exchange(true, std::memory_order_relaxed)) {
        // E's "no sync owed" test reads the mirror as a superset of R's counter: wrong now.
        LOG_CRITICAL(HW_GPU, "One follower: R's samples counter ran where E's mirror had it "
                             "stopped; FRITCHII_ONEF_POST_FENCE stops for this session");
    }
    // FRITCHII_ONEF_MIRROR_STOP (live, default 0): E's mirror started, R's counter not. A superset
    // (E lists the report for a sync R then does not need: OneFollowerCloseTargets counts it), the
    // kind a draw R skipped makes (E's mirror counts every lent draw with the count on; R starts
    // its counter only for a draw it records). 1: stop posting fences as the opposite case does;
    // 2: explained, no must-be-0 note, when R skipped such a draw since its last samples report.
    if (engine_bit) {
        const int mirror_stop = VideoCore::Nx2Debug::SwitchValue(
            VideoCore::Nx2Debug::g_onef_mirror_stop, "FRITCHII_ONEF_MIRROR_STOP");
        if (mirror_stop == 1 && !g.post_fence_stopped.exchange(true, std::memory_order_relaxed)) {
            LOG_CRITICAL(HW_GPU, "One follower: E's samples mirror started where R's counter did "
                                 "not{}; FRITCHII_ONEF_POST_FENCE stops for this session "
                                 "(FRITCHII_ONEF_MIRROR_STOP=1)",
                         skipped ? " (after a draw R skipped)" : "");
        }
        if (mirror_stop == 2 && skipped) {
            g.mirror_explained.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    g.mirror_count[engine_bit ? 1 : 2].fetch_add(1, std::memory_order_relaxed);
    Note(Check::SyncMirror, engine_bit ? "a samples report: E's mirror started, R's counter not"
                                       : "a samples report: R's counter started, E's mirror not");
}

void NoteSkippedSamplesDraw() noexcept {
    if (g.on && OnRenderThread()) {
        g.skipped_samples_draw = true;
    }
}

bool FlushOnEngine() noexcept {
    return g.on && Step() == StepLevel::Unlend &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_flush_e,
                                            "FRITCHII_ONEF_FLUSH_E") != 0;
}

bool PostClears() noexcept {
    return g.on && Step() == StepLevel::Unlend &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_op_clear,
                                            "FRITCHII_ONEF_OP_CLEAR") != 0 &&
           ViewReady();
}

bool PostOpSite(u8 site) noexcept {
    if (!g.on || Step() != StepLevel::Unlend) {
        return false;
    }
    using namespace VideoCore::Nx2Debug;
    switch (site) {
    case 15: // MemStage::Site::InlineUpload
        return SwitchValue(g_onef_op_inline, "FRITCHII_ONEF_OP_INLINE") != 0;
    case 10: // MemStage::Site::Invalidation: E's buffer halves (INVAL_E, GATHER_E)
        return SwitchValue(g_onef_op_inval, "FRITCHII_ONEF_OP_INVAL") != 0;
    case 7: // MemStage::Site::ListWords: Nx2KeepListWords' buffer side
        return SwitchValue(g_onef_op_listkeep, "FRITCHII_ONEF_OP_LISTKEEP") != 0;
    case 14: // MemStage::Site::Dma: AccelerateDMA's buffer copies and clears on E
        return SwitchValue(g_onef_op_dma, "FRITCHII_ONEF_OP_DMA") != 0;
    case 18: // MemStage::Site::Query: the sync buffers E obtains for a posted fence (16.2)
        return PostFenceOn();
    default:
        return false;
    }
}

void CountOpPacket(OpCount kind, u64 bytes) noexcept {
    g.op_count[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
    if (bytes != 0) {
        g.op_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

bool GatherOnEngine() noexcept {
    const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_gather_e,
                                                       "FRITCHII_ONEF_GATHER_E");
    const bool inval = value != 0 && InvalOnEngine();
    // The guard's parts, logged once per change (the GATHER_E probe found it never true).
    static std::atomic<int> logged{-1};
    const int state = (value != 0 ? 1 : 0) | (inval ? 2 : 0) |
                      (static_cast<int>(Step()) << 2) | (g_e_pipeline ? 32 : 0);
    if (logged.exchange(state, std::memory_order_relaxed) != state) [[unlikely]] {
        LOG_INFO(HW_GPU,
                 "One follower: FRITCHII_ONEF_GATHER_E={} (INVAL_E's conditions {}, step {}, "
                 "E_PIPELINE {}, INVAL_E switch {})",
                 value, inval ? "met" : "not met", static_cast<int>(Step()), g_e_pipeline ? 1 : 0,
                 VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_inval_e,
                                                  "FRITCHII_ONEF_INVAL_E"));
    }
    return inval;
}

bool CopyOnEngine() noexcept {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_copy_e,
                                            "FRITCHII_ONEF_COPY_E") != 0 &&
           InvalOnEngine();
}

void StampPageWrites(u64 address, u64 size) noexcept {
    if (size == 0 || !g.stamps_on.load(std::memory_order_acquire)) {
        return;
    }
    const u32 stamp = static_cast<u32>(g.pushed + 1);
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        g.stamps[page & (StampSlots - 1)].store(stamp, std::memory_order_relaxed);
    }
}

void CountInvalidation(InvalKind kind) noexcept {
    if (!CountersOn()) {
        return;
    }
    g.inval_count[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

bool WrittenAheadOfRender(u64 address, u64 size) noexcept {
    if (!Detail::t_render || size == 0 || !g.stamps_on.load(std::memory_order_acquire)) {
        return false;
    }
    // The other half of E's pair: R marked the texture filter (registering the image) before
    // this; E stamps, fences, then looks at the filter again. Either E saw the mark (the
    // invalidation went to R in order) or this sees E's stamp.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    g.ahead_checks.fetch_add(1, std::memory_order_relaxed);
    const u32 place = t_play_number;
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        const u32 stamp = g.stamps[page & (StampSlots - 1)].load(std::memory_order_relaxed);
        if (stamp != 0 && static_cast<s32>(stamp - place) > 0) {
            thread_local std::string note;
            note = fmt::format("image {:#x}+{:#x}: page {:#x} written by E before packet {}, R "
                               "at packet {}",
                               address, size, page << 12, stamp, place);
            Note(Check::TextureReadAhead, note.c_str());
            return true;
        }
    }
    return false;
}

bool QueryOpsOnBufferPages() noexcept {
    return QueryOps() && (Posted() || VideoCore::Nx2Debug::SwitchValue(
                                          VideoCore::Nx2Debug::g_onef_query_ops,
                                          "FRITCHII_ONEF_QUERY_OPS") >= 2);
}

bool Posted() noexcept {
    return g.on && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_posted,
                                                    "FRITCHII_ONEF_POSTED") != 0;
}

void CountLentFence(bool buffer_flush) noexcept {
    g.fences_lent.fetch_add(1, std::memory_order_relaxed);
    if (buffer_flush) {
        g.fences_with_flush.fetch_add(1, std::memory_order_relaxed);
    }
}

void CountLentGatherRanges(u64 ranges) noexcept {
    g.gathers_lent.fetch_add(1, std::memory_order_relaxed);
    g.gather_ranges.fetch_add(ranges, std::memory_order_relaxed);
}

void SetLendSite(const char* site) noexcept {
    t_lend_site = site;
}

void CountTransient(TransientTest test, u32 kind) noexcept {
    const size_t index = (std::min)(static_cast<size_t>(test),
                                    static_cast<size_t>(TransientTest::Count) - 1);
    g.transient[index][Stats::KindSlot(kind)].fetch_add(1, std::memory_order_relaxed);
}

bool UnlentResetOn() noexcept {
    return g.on && Step() == StepLevel::Unlend &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_unlent_reset,
                                            "FRITCHII_ONEF_UNLENT_RESET") != 0 &&
           !g.unlent_reset_stopped.load(std::memory_order_relaxed);
}

void StopUnlentReset(const char* why) noexcept {
    if (!g.unlent_reset_stopped.exchange(true, std::memory_order_relaxed)) {
        LOG_CRITICAL(HW_GPU, "One follower: FRITCHII_ONEF_UNLENT_RESET stops for this session ({})",
                     why != nullptr ? why : "?");
    }
}

bool UnlentValuesOn() noexcept {
    return g.on && Step() == StepLevel::Unlend &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_unlent_values,
                                            "FRITCHII_ONEF_UNLENT_VALUES") != 0 &&
           !g.unlent_values_stopped.load(std::memory_order_relaxed);
}

void StopUnlentValues(const char* why) noexcept {
    if (!g.unlent_values_stopped.exchange(true, std::memory_order_relaxed)) {
        LOG_CRITICAL(HW_GPU,
                     "One follower: FRITCHII_ONEF_UNLENT_VALUES stops for this session ({})",
                     why != nullptr ? why : "?");
    }
}

bool BindSeedOn() noexcept {
    return g.on && Step() == StepLevel::Unlend &&
           VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_bind_seed,
                                            "FRITCHII_ONEF_BIND_SEED") != 0 &&
           !g.bind_seed_stopped.load(std::memory_order_relaxed);
}

void StopBindSeed(const char* why) noexcept {
    if (!g.bind_seed_stopped.exchange(true, std::memory_order_relaxed)) {
        LOG_CRITICAL(HW_GPU, "One follower: FRITCHII_ONEF_BIND_SEED stops for this session ({})",
                     why != nullptr ? why : "?");
    }
}

void SetQueryKind(QueryKind kind) noexcept {
    t_query_kind = kind;
}

void CountQueryKind(QueryKind kind) noexcept {
    if (!CountersOn()) {
        return;
    }
    g.query_count[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

namespace {
/// E: commits a packet whose body (`body_bytes` after the header) is already in PackBuffer().
void PushPacked(PacketHeader& header, u8* buffer, std::size_t body_bytes) {
    const bool counting = CountersOn();
    const u64 behind = counting ? Behind() : 0;
    header.number = ++g.pushed;
    header.position = GpuSplit::Get().BackEntryPosition();
    if (header.draw_seq == 0) {
        header.draw_seq = header.kind == PacketKind::Draw ? ++g.draw_seq : g.draw_seq;
    }
    u64 floor = header.position;
    body_bytes = AppendCarried(header, buffer, body_bytes, floor);
    header.bytes = static_cast<u32>(sizeof(PacketHeader) + body_bytes);
    const u32 kind = header.kind == PacketKind::Draw ? TakeDrawKind(header) : 0;
    t_next_draw_set = false;
    std::memcpy(buffer, &header, sizeof(header));
    Commit(buffer, header.bytes, header.number, floor, true);
    CarriedSent(header.number);
    if (counting) {
        Stats::Pushed(header, kind, behind);
    }
}

/// E: the kept ops as a packet of their own (a Delta with no spans: E's changes since the last
/// packet stay for the next one). `why`: 0 E's consumed store, 1 a drain, 2 no room.
void SendCarriedAlone(size_t why) {
    if (g_carry.count == 0) {
        return;
    }
    g.carried_alone[why].store(g.carried_alone[why].load(std::memory_order_relaxed) + 1,
                               std::memory_order_relaxed);
    u8* const buffer = PackBuffer();
    PacketHeader header{};
    header.kind = PacketKind::Delta;
    header.number = ++g.pushed;
    header.position = GpuSplit::Get().BackEntryPosition();
    header.draw_seq = g.draw_seq;
    u64 floor = header.position;
    const std::size_t body_bytes = AppendCarried(header, buffer, 0, floor);
    header.bytes = static_cast<u32>(sizeof(PacketHeader) + body_bytes);
    std::memcpy(buffer, &header, sizeof(header));
    Commit(buffer, header.bytes, header.number, floor, true);
    CarriedSent(header.number);
}
} // Anonymous namespace

void Push(PacketHeader& header, std::span<const u8> spans) {
    if (!Detail::t_engine || !g.thread.joinable()) {
        t_next_draw_set = false;
        return;
    }
    const ESplit::PushGuard esplit_push{false, false, ESplit::PushKind::Push};
    u8* const buffer = PackBuffer();
    if (!spans.empty()) {
        std::memcpy(buffer + sizeof(PacketHeader), spans.data(), spans.size());
    }
    PushPacked(header, buffer, spans.size());
}

void Drain() {
    if (!Detail::t_engine) {
        return;
    }
    const ESplit::PushGuard esplit_push{false, false, ESplit::PushKind::Drain}; // E2's first
    if (g_carry.count != 0 && g.thread.joinable()) {
        SendCarriedAlone(1); // R plays everything E made, the kept ops too
    }
    const u64 target = g.pushed;
    const u64 waited =
        EngineWait([&] { return g.played.load(std::memory_order_acquire) >= target; },
                   EWait::Drain, target);
    if (waited != 0) {
        Stats::Drained(Stats::Timing() ? waited : 0);
    }
}

u64 Behind() noexcept {
    return g.pushed - g.played.load(std::memory_order_acquire);
}

void EngineConsumed(u64 position) noexcept {
    g.intended.store(position, std::memory_order_seq_cst); // RaisePlayed's fence pairs with R's
    RaisePlayed();
}

void PushDraw(const Maxwell3D& engine, u64 draw_seq, u64 buffer_number, u8 flags,
              const void* pipeline, u8 occl_prepass) {
    if (!Detail::t_engine || !g.thread.joinable()) {
        t_next_draw_set = false;
        return;
    }
    // Part (c)'s unlent draw: its ring B packet is filled; this packet carries the spans (the
    // draw span always) and names it. E goes on within FRITCHII_ONEF_AHEAD.
    PacketHeader header{};
    header.kind = PacketKind::Draw;
    header.flags = flags;
    header.draw_seq = draw_seq;
    header.buffer_number = buffer_number;
    header.pipeline = pipeline;
    header.reserved[1] = occl_prepass;
    // Packed straight into the packet buffer, after the header: no copy, no allocation.
    u8* const buffer = PackBuffer();
    std::size_t bytes;
    {
        // FRITCHII_ESPLIT_COUNT: packing R's spans is E1's (the push itself stays E2's).
        const ESplit::Carve esplit_pack{ESplit::Class::E1};
        bytes = PackSpans(engine, header, std::span<u8>{buffer + sizeof(PacketHeader), MaxSpanBytes});
    }
    PushPacked(header, buffer, bytes);
}

std::size_t PackedDrawRoom() noexcept {
    return sizeof(PacketHeader) + MaxSpanBytes + MaxPayloadBytes + MaxCarryBytes +
           sizeof(CarryTrailer);
}

std::size_t PackDraw(const Maxwell3D& engine, u64 draw_seq, u8 flags, const void* pipeline,
                     PacketHeader& header, u8* buffer) noexcept {
    header = PacketHeader{};
    header.kind = PacketKind::Draw;
    header.flags = flags;
    header.draw_seq = draw_seq;
    header.pipeline = pipeline;
    if (!Detail::t_engine || !g.thread.joinable()) {
        return 0;
    }
    return PackSpans(engine, header, std::span<u8>{buffer + sizeof(PacketHeader), MaxSpanBytes});
}

void PushDrawPacked(PacketHeader& header, u8* buffer, std::size_t body_bytes,
                    u64 buffer_number) noexcept {
    if (!Detail::t_engine || !g.thread.joinable()) {
        t_next_draw_set = false;
        return;
    }
    header.buffer_number = buffer_number;
    PushPacked(header, buffer, body_bytes);
}

void SetEngine(Maxwell3D* engine) noexcept {
    if (g.engine.exchange(engine, std::memory_order_acq_rel) != engine) {
        // A new channel's engine: the view takes it in full at the next lent packet, and E's
        // tracking starts over (the channel bind is lent, so E is parked here).
        Detail::ViewTrackingStopped();
    }
}

Maxwell3D* Engine() noexcept {
    return g.engine.load(std::memory_order_acquire);
}

void CheckRenderOwned(const char* where) noexcept {
    if (!Detail::t_engine) {
        return; // R itself, or a thread the caches' locks already order
    }
    // E with every packet played is R parked or idle: today's serial order.
    if (g.played.load(std::memory_order_acquire) >= g.pushed) {
        return;
    }
    Note(Check::OwnerOnE, where);
}

void CheckEngineOwned(const char* where) noexcept {
    if (t_render && t_lent_depth == 0) {
        Note(Check::OwnerOnR, where);
    }
}

u32 UnlendMask() noexcept {
    (void)Step();
    return g.unlend_mask.load(std::memory_order_relaxed);
}

int VerifyEvery() noexcept {
    (void)Step();
    return g.verify_every.load(std::memory_order_relaxed);
}

int Ahead() noexcept {
    (void)Step();
    return g.ahead.load(std::memory_order_relaxed);
}

} // namespace VideoCommon::OneFollower
