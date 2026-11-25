// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <tuple>
#include "common/cpu_features.h"
#include "common/cpu_features.h"

#ifdef _WIN32
#include "common/windows/timer_resolution.h"
#endif

#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core_timing.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/nx2_thread_stats.h"
#include "video_core/nx2_debug_control.h"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace Core::Timing {

namespace {

/// EDEN_NX2_EXP75 (live, us, 0 = off; fable52_answer_pipeline section 3): the timer thread spins
/// the last this-many us before a precise event (the kernel's hardware timer: guest sleeps and
/// timeouts) instead of waking on the OS timer's 0.5 ms tick.
s64 TimerSpinNs() {
    return s64{VideoCore::Nx2Debug::Experiment(75)} * 1000;
}

void SpinPause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

/// FRITCHII_HOSTTIMING_SPIN_BUDGET (live, percent of the timer thread's time, 0 = no limit, as
/// before): EXP75's spins are paid from a budget that grows by this share of the time that passes,
/// holding at most SpinBurstNs. With less left than EXP75's window, the thread spins only what the
/// budget holds and sleeps on the OS timer before that. Guest sleeps of a few hundred us back to
/// back (OoT's present poll sleeps 250 us at a time while it waits for a frame) otherwise keep a
/// whole core spinning: 2.3 s of CPU in a 2.5 s Castle Town load freeze. Idea from
/// zortos293/Lichii #1733 (a fifth of the thread's time, 2 ms of burst).
int SpinBudgetPercent() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_hosttiming_spin_budget,
                                            "FRITCHII_HOSTTIMING_SPIN_BUDGET");
}
constexpr s64 SpinBurstNs = 2'000'000;

/// FRITCHII_HOSTTIMING_PRIORITY (live, default 0 = highest, as before): 1 = the timer thread runs
/// time critical, the priority of the guest cores and the GPU threads. At highest it is one step
/// below them and below DWM and audio threads: in market windows where other processes took 2.5+
/// CPUs, 400-900 timer callbacks every 10 s ran 1 ms or more late (guest sleeps, timeouts and the
/// syncpoint polls the game's GPU waits are answered by) and single callbacks took 4-9 ms of wall
/// time, most likely the thread off the CPU. Only the thread's OS priority changes.
/// FRITCHII_HOSTTIMING_OFF_GPU (live, default 0; helper DE). Where the timer thread spins.
/// 0 (as before): BackgroundWorkCpuMask at the first spin after any GPU thread pinned, never
/// again; the GPU threads pin one after another at boot, so a spin between the front's pin and
/// the back's left HostTiming allowed onto the back's core (b37 census: HostTiming 295-496 ms/s
/// allowed on 0x30, the busiest thread that could share E's core). 1: every processor but the
/// physical cores of the pinned GPU threads (front, back, R; both hardware threads), placed
/// again whenever another GPU thread pins; refused on small CPUs (Common::HostTimingOffGpuMask).
/// Back to 0 restores the placement 0 made. Each placement is logged with where the thread is.
void PlaceTimerSpin() {
    static bool placed = false;          // mode 0's own placement
    static u64 placed_mask = 0;          // what mode 0 set (0: none yet)
    static int applied_mode = 0;         // the mode the thread's mask follows
    static u32 applied_generation = ~0U; // mode 1: the pinned set it was placed for
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_hosttiming_off_gpu,
                                                      "FRITCHII_HOSTTIMING_OFF_GPU") != 0
                         ? 1
                         : 0;
    const auto log_place = [](const char* why, u64 mask, u64 gpu) {
        const int processor = Common::CurrentProcessorNumber();
        LOG_INFO(Core_Timing,
                 "HostTiming placed ({}): mask {:#x}, GPU threads' cores {:#x}, running on "
                 "processor {}",
                 why, mask, gpu, processor);
    };
    if (mode == 0) {
        if (applied_mode != 0) {
            // Back from 1: the placement mode 0 had (or every processor if it had none).
            Common::SetCurrentThreadCpuMask(placed_mask);
            applied_mode = 0;
            log_place("FRITCHII_HOSTTIMING_OFF_GPU=0", placed_mask, 0);
        }
        if (!placed) {
            // Off the GPU threads' cores, once they are pinned.
            if (const u64 mask = Common::BackgroundWorkCpuMask(); mask != 0) {
                Common::SetCurrentThreadCpuMask(mask);
                placed = true;
                placed_mask = mask;
                log_place("first spin after a GPU pin", mask, 0);
            }
        }
        return;
    }
    const u32 generation = Common::PinGeneration();
    if (applied_mode == 1 && generation == applied_generation) [[likely]] {
        return;
    }
    u64 gpu_cores = 0;
    const u64 mask = Common::HostTimingOffGpuMask(gpu_cores);
    applied_generation = generation;
    if (mask == 0) {
        // Nothing pinned yet (looked at again at the next pin) or refused on this CPU.
        applied_mode = 1;
        return;
    }
    Common::SetCurrentThreadCpuMask(mask);
    applied_mode = 1;
    log_place("FRITCHII_HOSTTIMING_OFF_GPU=1", mask, gpu_cores);
}

int TimerPriorityMode() {
    return VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_hosttiming_priority,
                                            "FRITCHII_HOSTTIMING_PRIORITY");
}

/// On the timer thread: applies FRITCHII_HOSTTIMING_PRIORITY when it changed. Off: one relaxed
/// load and a compare per pass.
void ApplyTimerPriority() {
    static thread_local int applied = 0; // The thread starts at highest (Initialize)
    const int wanted = TimerPriorityMode();
    if (wanted == applied) [[likely]] {
        return;
    }
    Common::SetCurrentThreadPriority(wanted == 1 ? Common::ThreadPriority::Critical
                                                 : Common::ThreadPriority::VeryHigh);
    LOG_INFO(Core_Timing, "HostTiming priority (FRITCHII_HOSTTIMING_PRIORITY={}): {}", wanted,
             wanted == 1 ? "time critical" : "highest");
    applied = wanted;
}

/// What the timer thread did, logged every 10 s from it ("HostTiming in 10 s"). The spins and the
/// callbacks of 1 ms or more also go to the hitch line, so a hitch where HostTiming is busy says
/// which of the two it was. Thread local: Advance also runs on the CPU thread in single core mode.
struct TimerStats {
    s64 window_start{};
    u64 waits{};          ///< OS waits (the high resolution timer or the event)
    u64 spins{};          ///< EXP75 spins
    s64 spin_ns{};        ///< time in them
    s64 longest_spin{};   ///< the longest one
    u64 budget_cuts{};    ///< spins the budget shortened or turned into an OS wait
    u64 events{};         ///< callbacks run
    s64 callback_ns{};    ///< time in them
    s64 longest_callback{};
    std::string longest_callback_name;
    u64 slow_callbacks{}; ///< callbacks of 1 ms or more
    u64 most_in_pass{};   ///< the most callbacks one Advance ran (a catch-up after a delay)
    u64 late_1ms{};       ///< callbacks that ran 1 ms or more after their time
    s64 worst_late{};
    std::string worst_late_name;
};
thread_local TimerStats t_timer_stats;

void NoteCallback(const EventType& type, s64 began, s64 ended, s64 late) {
    TimerStats& s = t_timer_stats;
    const s64 took = ended - began;
    ++s.events;
    s.callback_ns += took;
    if (took > s.longest_callback) {
        s.longest_callback = took;
        s.longest_callback_name = type.name;
    }
    if (took >= 1'000'000) {
        ++s.slow_callbacks;
        Common::Nx2Hitch::Add(Common::Nx2Hitch::Cause::SlowTimerEvent, static_cast<u64>(took));
    }
    if (late >= 1'000'000) {
        ++s.late_1ms;
        if (late > s.worst_late) {
            s.worst_late = late;
            s.worst_late_name = type.name;
        }
    }
}

void NoteSpin(s64 ns) {
    TimerStats& s = t_timer_stats;
    ++s.spins;
    s.spin_ns += ns;
    s.longest_spin = (std::max)(s.longest_spin, ns);
    Common::Nx2Hitch::Add(Common::Nx2Hitch::Cause::TimerSpin, static_cast<u64>((std::max)(ns, s64{0})));
}

void MaybeLogTimerStats(s64 now) {
    TimerStats& s = t_timer_stats;
    if (s.window_start == 0) {
        s.window_start = now;
        return;
    }
    if (now - s.window_start < 10'000'000'000) {
        return;
    }
    const double seconds = static_cast<double>(now - s.window_start) / 1e9;
    LOG_INFO(Core_Timing,
             "HostTiming in {:.0f} s: spun {:.1f} ms/s before guest timers ({} spins, longest {} us; "
             "EXP75={} us, FRITCHII_HOSTTIMING_SPIN_BUDGET={}%: {} cut by it); {} OS waits; {} "
             "events, callbacks {:.1f} ms/s, longest {} us ({}), {} of 1 ms or more; most in one "
             "pass {}; {} ran 1 ms or more late (worst {:.1f} ms, {}); "
             "FRITCHII_HOSTTIMING_PRIORITY={}",
             seconds, static_cast<double>(s.spin_ns) / 1e6 / seconds, s.spins,
             s.longest_spin / 1000, TimerSpinNs() / 1000, SpinBudgetPercent(), s.budget_cuts,
             s.waits, s.events, static_cast<double>(s.callback_ns) / 1e6 / seconds,
             s.longest_callback / 1000,
             s.longest_callback_name.empty() ? "none" : s.longest_callback_name.c_str(),
             s.slow_callbacks, s.most_in_pass, s.late_1ms, static_cast<double>(s.worst_late) / 1e6,
             s.worst_late_name.empty() ? "none" : s.worst_late_name.c_str(),
             TimerPriorityMode());
    Common::NoteHostTimingWindow(seconds, s.late_1ms);
    s = TimerStats{};
    s.window_start = now;
}

} // Anonymous namespace

constexpr s64 MAX_SLICE_LENGTH = 10000;

constexpr u32 CPU_CLOCK_BASE_MHZ = 1020;
constexpr u32 CPU_CLOCK_BOOST_MHZ = 1734;
constexpr u32 CPU_CLOCK_OVERCLOCK_MHZ = 2040;

constexpr u32 CpuClockTargetMhz(Settings::CpuClock clock) {
    switch (clock) {
    case Settings::CpuClock::Boost:
        return CPU_CLOCK_BOOST_MHZ;
    case Settings::CpuClock::Overclock:
        return CPU_CLOCK_OVERCLOCK_MHZ;
    default:
        return CPU_CLOCK_BASE_MHZ;
    }
}

std::atomic<u64> g_inline_cntpct_factor{};
static_assert(sizeof(std::atomic<u64>) == sizeof(u64) && std::atomic<u64>::is_always_lock_free);

bool InlineCntpctEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("FRITCHII_JIT_INLINE_CNTPCT");
        if (value == nullptr || value[0] == '\0' || value[0] == '0') {
            return false;
        }
#if defined(ARCHITECTURE_x86_64)
        if (!Common::g_wall_clock.IsNative()) {
            LOG_INFO(Core_Timing, "FRITCHII_JIT_INLINE_CNTPCT: off, the wall clock is not the TSC "
                                  "(counter reads stay host calls)");
            return false;
        }
        LOG_INFO(Core_Timing, "FRITCHII_JIT_INLINE_CNTPCT: guest counter reads computed in the "
                              "JIT (host call while the CPU clock is not Normal, core speed sync "
                              "or guest thread stats are on)");
        return true;
#else
        return false;
#endif
    }();
    return enabled;
}

void RefreshInlineCntpct() {
    u64 factor = 0;
#if defined(ARCHITECTURE_x86_64)
    if (InlineCntpctEnabled() &&
        CpuClockTargetMhz(Settings::values.cpu_clock.GetValue()) == CPU_CLOCK_BASE_MHZ &&
        !Settings::values.sync_core_speed.GetValue() && !Kernel::Nx2ThreadStats::g_enabled) {
        factor = Common::g_wall_clock.cntpct_rdtsc_factor.load(std::memory_order_relaxed);
    }
#endif
    g_inline_cntpct_factor.store(factor, std::memory_order_relaxed);
}

std::shared_ptr<EventType> CreateEvent(std::string name, TimedCallback&& callback) {
    return std::make_shared<EventType>(std::move(callback), std::move(name));
}

struct CoreTiming::Event {
    s64 time;
    u64 fifo_order;
    std::weak_ptr<EventType> type;
    s64 reschedule_time;
    heap_t::handle_type handle{};
    bool precise{};

    // Sort by time, unless the times are the same, in which case sort by
    // the order added to the queue
    friend bool operator>(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) > std::tie(right.time, right.fifo_order);
    }

    friend bool operator<(const Event& left, const Event& right) {
        return std::tie(left.time, left.fifo_order) < std::tie(right.time, right.fifo_order);
    }
};

CoreTiming::CoreTiming() = default;
CoreTiming::~CoreTiming() {
    Reset();
#ifdef _WIN32
    Common::Windows::CloseWaitHandle(wake_handle);
#endif
}

void CoreTiming::Wake() {
    wake_count.fetch_add(1, std::memory_order_release);
    event.Set();
#ifdef _WIN32
    Common::Windows::SignalWakeEvent(wake_handle);
#endif
}

void CoreTiming::Initialize(std::function<void()>&& on_thread_init_) {
    Reset();
    on_thread_init = std::move(on_thread_init_);
    event_fifo_id = 0;
    cpu_ticks = 0;
    if (is_multicore) {
#ifdef _WIN32
        if (wake_handle == nullptr) {
            wake_handle = Common::Windows::CreateWakeEvent();
        }
#endif
        timer_thread = std::jthread([this](std::stop_token stop_token) {
            Common::SetCurrentThreadName("HostTiming");
            Common::SetCurrentThreadPriority(Common::ThreadPriority::VeryHigh);
            Common::SetCurrentThreadToPerformanceCores();
            on_thread_init();
#ifdef _WIN32
            // EDEN_HIRES_CLOCK=0 keeps the plain timed wait.
            const char* const hires_switch = std::getenv("EDEN_HIRES_CLOCK");
            void* const hires_timer = (hires_switch != nullptr && hires_switch[0] == '0')
                                          ? nullptr
                                          : Common::Windows::CreateHighResolutionTimer();
            SCOPE_EXIT {
                Common::Windows::CloseWaitHandle(hires_timer);
            };
#endif
            has_started = true;

            // FRITCHII_HOSTTIMING_SPIN_BUDGET: what is left to spin, and when it last grew.
            s64 spin_budget_ns = SpinBurstNs;
            s64 spin_budget_at = 0;

            // base frequency in MHz: 1ns (10^-9) = 1GHz (10^9)
            while (!stop_token.stop_requested()) {
                while (!paused && !stop_token.stop_requested()) {
                    paused_set = false;
                    ApplyTimerPriority();
                    if (auto const next_time = Advance(); next_time) {
                        const s64 now_ns = GetGlobalTimeNs().count();
                        MaybeLogTimerStats(now_ns);
                        // There are more events left in the queue, wait until the next event.
                        auto wait_time = *next_time - now_ns;
                        // EXP75: a precise event due within the spin window: spin to the next
                        // event (never later than that precise one), else wake on the OS timer
                        // that far ahead of it. A Wake (an earlier event scheduled) ends the spin.
                        if (const s64 spin_window = TimerSpinNs();
                            spin_window > 0 && wait_time > 0 &&
                            next_precise_ns != (std::numeric_limits<s64>::max)()) {
                            const s64 precise_in = next_precise_ns - now_ns;
                            s64 spin_ns = spin_window;
                            if (const int budget = SpinBudgetPercent(); budget > 0) {
                                if (spin_budget_at != 0) {
                                    spin_budget_ns = (std::min)(
                                        SpinBurstNs,
                                        spin_budget_ns + (now_ns - spin_budget_at) * budget / 100);
                                }
                                spin_budget_at = now_ns;
                                spin_ns = std::clamp<s64>(spin_budget_ns, 0, spin_window);
                                if (precise_in <= spin_window && precise_in > spin_ns) {
                                    ++t_timer_stats.budget_cuts;
                                }
                            } else {
                                spin_budget_ns = SpinBurstNs;
                                spin_budget_at = 0;
                            }
                            if (precise_in <= spin_ns) {
                                PlaceTimerSpin();
                                const u32 seen = wake_count.load(std::memory_order_acquire);
                                s64 spun_until = now_ns;
                                while ((spun_until = GetGlobalTimeNs().count()) < *next_time &&
                                       wake_count.load(std::memory_order_acquire) == seen &&
                                       !paused && !stop_token.stop_requested()) {
                                    SpinPause();
                                }
                                NoteSpin(spun_until - now_ns);
                                spin_budget_ns -= spun_until - now_ns;
                                continue;
                            }
                            wait_time = (std::min)(wait_time, precise_in - spin_ns);
                        }
                        if (wait_time > 0) {
                            ++t_timer_stats.waits;
#ifdef _WIN32
                            // A high resolution timer keeps the guest's events (vsync among them)
                            // on time even where Windows doesn't honour our timer resolution
                            // request (Windows 11: players gained ~30% fps from making it global).
                            if (!Common::Windows::WaitHighResolution(
                                    hires_timer, wake_handle, std::chrono::nanoseconds(wait_time)))
#endif
                                event.WaitFor(std::chrono::nanoseconds(wait_time));
                        }
                    } else {
                        // Queue is empty, wait until another event is scheduled and signals us to
                        // continue.
                        wait_set = true;
                        event.Wait();
                    }
                    wait_set = false;
                }
                paused_set = true;
                pause_event.Wait();
            }
        });
    }
}

void CoreTiming::ClearPendingEvents() {
    std::scoped_lock lock{advance_lock, basic_lock};
    event_queue.clear();
    Wake();
}

void CoreTiming::Pause(bool is_paused) {
    paused = is_paused;
    pause_event.Set();

    if (!is_paused) {
        pause_end_time = GetGlobalTimeNs().count();
    }
}

void CoreTiming::SyncPause(bool is_paused) {
    if (is_paused == paused && paused_set == paused) {
        return;
    }

    Pause(is_paused);
    if (timer_thread.joinable()) {
        if (!is_paused) {
            pause_event.Set();
        }
        Wake();
        while (paused_set != is_paused)
            ;
    }

    if (!is_paused) {
        pause_end_time = GetGlobalTimeNs().count();
    }
}

bool CoreTiming::IsRunning() const {
    return !paused_set;
}

bool CoreTiming::HasPendingEvents() const {
    std::scoped_lock lock{basic_lock};
    return !(wait_set && event_queue.empty());
}

void CoreTiming::ScheduleEvent(std::chrono::nanoseconds ns_into_future,
                               const std::shared_ptr<EventType>& event_type, bool absolute_time) {
    {
        std::scoped_lock scope{basic_lock};
        const auto next_time{absolute_time ? ns_into_future : GetGlobalTimeNs() + ns_into_future};

        const bool precise = event_type && event_type->precise;
        auto h{event_queue.emplace(Event{next_time.count(), event_fifo_id++, event_type, 0, {}, precise})};
        (*h).handle = h;
    }

    Wake();
}

void CoreTiming::ScheduleLoopingEvent(std::chrono::nanoseconds start_time,
                                      std::chrono::nanoseconds resched_time,
                                      const std::shared_ptr<EventType>& event_type,
                                      bool absolute_time) {
    {
        std::scoped_lock scope{basic_lock};
        const auto next_time{absolute_time ? start_time : GetGlobalTimeNs() + start_time};

        const bool precise = event_type && event_type->precise;
        auto h{event_queue.emplace(
            Event{next_time.count(), event_fifo_id++, event_type, resched_time.count(), {}, precise})};
        (*h).handle = h;
    }

    Wake();
}

void CoreTiming::UnscheduleEvent(const std::shared_ptr<EventType>& event_type,
                                 UnscheduleEventType type) {
    {
        std::scoped_lock lk{basic_lock};

        std::vector<heap_t::handle_type> to_remove;
        for (auto itr = event_queue.begin(); itr != event_queue.end(); itr++) {
            const Event& e = *itr;
            if (e.type.lock().get() == event_type.get()) {
                to_remove.push_back(itr->handle);
            }
        }

        for (auto& h : to_remove) {
            event_queue.erase(h);
        }

        event_type->sequence_number++;
    }

    // Force any in-progress events to finish
    if (type == UnscheduleEventType::Wait) {
        std::scoped_lock lk{advance_lock};
    }
}

static u64 GetNextTickCount(u64 next_ticks) {
    if (Settings::values.use_custom_cpu_ticks.GetValue()) {
        return Settings::values.cpu_ticks.GetValue();
    }
    return next_ticks;
}

void CoreTiming::AddTicks(u64 ticks_to_add) {
    const u64 ticks = GetNextTickCount(ticks_to_add);
    cpu_ticks += ticks;
    downcount -= static_cast<s64>(ticks);
}

void CoreTiming::Idle() {
    AddTicks(1000U);
}

void CoreTiming::ResetTicks() {
    downcount = MAX_SLICE_LENGTH;
}

u64 CoreTiming::GetClockTicks() const {
    u64 fres = is_multicore ? Common::g_wall_clock.GetCNTPCT() : Common::g_wall_clock.CPUTickToGuestCNTPCT(cpu_ticks);
    if (const u32 target = CpuClockTargetMhz(Settings::values.cpu_clock.GetValue());
        target != CPU_CLOCK_BASE_MHZ) {
        fres = fres * target / CPU_CLOCK_BASE_MHZ;
    }
    if (::Settings::values.sync_core_speed.GetValue()) {
        auto const ticks = f64(fres);
        auto const speed_limit = f64(Settings::SpeedLimit()) * 0.01;
        return u64(ticks / speed_limit);
    }
    return fres;
}

u64 CoreTiming::GetGPUTicks() const {
    return is_multicore
        ? Common::g_wall_clock.GetGPUTick()
        : Common::WallClock::CPUTickToGPUTick(cpu_ticks);
}

std::optional<s64> CoreTiming::Advance() {
    std::scoped_lock lock{advance_lock, basic_lock};
    global_timer = GetGlobalTimeNs().count();
    u64 run_in_pass = 0;
    SCOPE_EXIT {
        t_timer_stats.most_in_pass = (std::max)(t_timer_stats.most_in_pass, run_in_pass);
    };

    while (!event_queue.empty() && event_queue.top().time <= global_timer) {
        const Event& evt = event_queue.top();

        if (const auto event_type{evt.type.lock()}) {
            const auto evt_time = evt.time;
            const auto evt_sequence_num = event_type->sequence_number;
            ++run_in_pass;

            if (evt.reschedule_time == 0) {
                event_queue.pop();

                basic_lock.unlock();

                const s64 began = GetGlobalTimeNs().count();
                event_type->callback(evt_time, std::chrono::nanoseconds{began - evt_time});
                NoteCallback(*event_type, began, GetGlobalTimeNs().count(), began - evt_time);

                basic_lock.lock();
            } else {
                basic_lock.unlock();

                // Time spent paused is not lateness: a callback that makes up for being late (the
                // sysmix pump) would otherwise wait out the pause again after it.
                const s64 now_ns{GetGlobalTimeNs().count()};
                const s64 late_ns{now_ns - evt_time};
                const std::chrono::nanoseconds late_since_pause{
                    evt_time < pause_end_time ? (std::min)(late_ns, now_ns - pause_end_time)
                                              : late_ns};
                const auto new_schedule_time{event_type->callback(evt_time, late_since_pause)};
                NoteCallback(*event_type, now_ns, GetGlobalTimeNs().count(),
                             late_since_pause.count());

                basic_lock.lock();

                if (evt_sequence_num != event_type->sequence_number) {
                    // Heap handle is invalidated after external modification.
                    continue;
                }

                const auto next_schedule_time{new_schedule_time.has_value()
                                                  ? new_schedule_time.value().count()
                                                  : evt.reschedule_time};

                // If this event was scheduled into a pause, its time now is going to be way
                // behind. Re-set this event to continue from the end of the pause.
                auto next_time{evt.time + next_schedule_time};
                if (evt.time < pause_end_time) {
                    next_time = pause_end_time + next_schedule_time;
                }

                event_queue.update(evt.handle, Event{next_time, event_fifo_id++, evt.type,
                                                     next_schedule_time, evt.handle,
                                                     event_type->precise});
            }
        } else {
            event_queue.pop();
        }

        global_timer = GetGlobalTimeNs().count();
    }

    // The earliest precise event (the heap orders by time only; a few dozen events at most).
    next_precise_ns = (std::numeric_limits<s64>::max)();
    if (TimerSpinNs() > 0) {
        for (const Event& queued : event_queue) {
            if (queued.precise && queued.time < next_precise_ns) {
                next_precise_ns = queued.time;
            }
        }
    }

    if (!event_queue.empty()) {
        return event_queue.top().time;
    } else {
        return std::nullopt;
    }
}

void CoreTiming::Reset() {
    paused = true;
    // Request the stop before waking the thread: woken first, it could take the wake-up, see no
    // stop request yet, and go back to waiting on pause_event for good (the join below then hung
    // the frontend at boot).
    if (timer_thread.joinable()) {
        timer_thread.request_stop();
    }
    pause_event.Set();
    Wake();
    if (timer_thread.joinable()) {
        timer_thread.join();
    }
    has_started = false;
}

/// @brief Returns current time in nanoseconds.
std::chrono::nanoseconds CoreTiming::GetGlobalTimeNs() const noexcept {
    return is_multicore
        ? Common::g_wall_clock.GetTimeNS()
        : std::chrono::nanoseconds{Common::WallClock::CPUTickToNS(cpu_ticks)};
}

/// @brief Returns current time in microseconds.
std::chrono::microseconds CoreTiming::GetGlobalTimeUs() const noexcept {
    return is_multicore
        ? Common::g_wall_clock.GetTimeUS()
        : std::chrono::microseconds{Common::WallClock::CPUTickToUS(cpu_ticks)};
}

#ifdef _WIN32
void CoreTiming::SetTimerResolutionNs(std::chrono::nanoseconds ns) {
    timer_resolution_ns = ns.count();
}
#endif

} // namespace Core::Timing
