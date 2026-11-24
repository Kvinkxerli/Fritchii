// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <thread>
#include "common/common_types.h"
#include "common/polyfill_thread.h"

namespace Common {

class Event {
public:
#ifdef _WIN32
    ~Event();
#endif

    void Set() {
        std::scoped_lock lk{mutex};
        if (!is_set) {
            is_set = true;
#ifdef _WIN32
            if (sleepers != 0) {
                WakeSleeper();
            }
#endif
            condvar.notify_one();
        }
    }

    void Wait() {
        std::unique_lock lk{mutex};
        condvar.wait(lk, [&] { return is_set.load(); });
        is_set = false;
    }

    bool WaitFor(const std::chrono::nanoseconds time);

    /// Like WaitFor, but always blocks in the OS. On Windows, WaitFor keeps the thread running
    /// in a hardware wait; this sleeps on a high-resolution waitable timer instead (ported;
    /// only Tegra::Nx2PendingWake uses it, under EDEN_NX2_GPU_WAKE).
    bool SleepFor(const std::chrono::nanoseconds time);

    template<class Clock, class Duration>
    bool WaitUntil(const std::chrono::time_point<Clock, Duration> time) {
        std::unique_lock lk{mutex};
        if (!condvar.wait_until(lk, time, [this] { return is_set.load(); }))
            return false;
        is_set = false;
        return true;
    }

    void Reset() {
        std::unique_lock lk{mutex};
        // no other action required, since wait loops on the predicate and any lingering signal will
        // get cleared on the first iteration
        is_set = false;
    }

    [[nodiscard]] bool IsSet() const {
        return is_set;
    }

private:
#ifdef _WIN32
    void WakeSleeper();
#endif

    alignas(64) std::atomic<bool> is_set{false};
    std::condition_variable condvar;
    std::mutex mutex;
#ifdef _WIN32
    void* sleep_wake{}; ///< SleepFor's wake event, made on its first use
    u32 sleepers{};     ///< Threads in SleepFor; Set signals sleep_wake only then
#endif
};

class Barrier {
public:
    explicit Barrier(std::size_t count_) : count(count_) {}

    /// Blocks until all "count" threads have called Sync()
    bool Sync(std::stop_token token = {}) {
        std::unique_lock lk{mutex};
        const std::size_t current_generation = generation;

        if (++waiting == count) {
            generation++;
            waiting = 0;
            condvar.notify_all();
            return true;
        } else {
            condvar.wait(lk, token,
                        [this, current_generation] { return current_generation != generation; });
            return !token.stop_requested();
        }
    }

private:
    std::condition_variable_any condvar;
    std::mutex mutex;
    std::size_t count;
    std::size_t waiting = 0;
    std::size_t generation = 0; // Incremented once each time the barrier is used
};

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

enum class ThreadPlacement : u32 {
    Default = 0,
    Background = 1,
    Efficiency = 2,
};

void SetCurrentThreadPriority(ThreadPriority new_priority);
/// OS thread id for SetNativeThreadPriority: GetCurrentThreadId on Windows, gettid on Linux.
[[nodiscard]] u64 GetCurrentNativeThreadId();
/// Raise or lower another thread of this process. no-op if native_id is 0.
void SetNativeThreadPriority(u64 native_id, ThreadPriority new_priority);
/// Saved native priority (Win32 GetThreadPriority / Linux nice). Used to restore a boosted builder.
[[nodiscard]] int GetNativeThreadNice(u64 native_id);
/// Restore a native priority previously read with GetNativeThreadNice.
void SetNativeThreadNice(u64 native_id, int nice_value);
void SetCurrentThreadName(const char* name);
void SetCurrentThreadToPerformanceCores();

/// Windows: one physical performance core each for the split GPU thread's halves and the guest
/// cores kept off them, so an SMT sibling never shares the GPU threads' core. See
/// ThreadPinning::MakePlan for when it applies; EDEN_NX2_PIN_THREADS=0 turns it off, 1 forces it.
/// GpuMem (FRITCHII_PIN_GPUMEM=1, boot, default off): the memory stage's thread gets one physical
/// core of its own, taken from the guest set (Fable B, 2026-11-14: unpinned, it shared cores and
/// unchanged front/back work cost 9-21% more a draw with the stage on).
enum class PinnedRole : u32 { GpuFront, GpuBack, GuestCore, GpuMem };
/// `guest_core`: the guest core index for GuestCore. With EDEN_NX2_GUEST_PIN_EACH=1 each guest core
/// gets one physical core of the guest set to itself (its SMT siblings) instead of the whole set.
void PinCurrentThread(PinnedRole role, int guest_core = -1);
/// Windows: the processor group 0 logical processors background work should use to stay off the
/// cores pinned to the GPU threads (and their L3 cache where the CPU has several). 0 while nothing
/// is pinned, and elsewhere.
[[nodiscard]] u64 BackgroundWorkCpuMask();
/// Windows: hard affinity to `mask` (group 0), or with 0 to every CPU the process may use.
void SetCurrentThreadCpuMask(u64 mask);

// GPU core siblings (helper DE, 2026-11-17). The pinned GPU threads own a whole physical core each
// (front 0xc, back 0x30 on a 7800X3D) but use one of its two logical processors; the other one is
// in no thread's mask, so it is often the only idle logical processor and unpinned busy threads
// (VulkanWorker ~400 ms/s at Critical priority, GPUFencingThread, VulkanPresent) land on it and
// share the GPU thread's core.
/// The unpinned threads that note where they run for FRITCHII_SIBLING_CENSUS.
enum class CensusThread : u32 { VulkanWorker, Fencing, Present, Count };
/// FRITCHII_SIBLING_CENSUS=1 (boot, default 0; a dry counter, changes nothing): notes the logical
/// processor the calling thread runs on (one call per unit of work). A flag test when off.
void NoteCensusCpu(CensusThread who);
/// Once a second, on the Pulse thread. FRITCHII_GPU_SIBLINGS_FREE=2: every other thread of the
/// process nobody placed (still on the whole process mask: driver threads, audio, VSync...) is
/// kept off the pinned GPU threads' cores too; another value gives them their masks back.
/// FRITCHII_SIBLING_CENSUS=1: returns what ran on each pinned GPU thread's physical core besides
/// that thread (ms/s, DPC + interrupt time apart), this process's busiest threads allowed onto
/// it by name, and where the noted threads ran. Empty when the census is off.
[[nodiscard]] std::string GpuCoreTick();
/// FRITCHII_GPU_SIBLINGS_FREE (live, default 0): 1 or 2 = the calling thread keeps off the pinned
/// GPU threads' cores (BackgroundWorkCpuMask); back to 0 = every CPU the process may use. Refused
/// (logged once, nothing changes) when that would leave fewer than MinAutoGuestProcessors + 2
/// fast logical processors (6-core CPUs with SMT, 8 cores without SMT). `applied` is the caller's
/// own record (start with 0: nothing changes while the switch is 0). Called once per unit of
/// work; one relaxed load when nothing changed.
void KeepOffGpuCores(int& applied);
/// Bumped each time a GPU role thread pins (front, back, R/GPUMem): placements that depend on
/// the pinned set look again when it moved (one relaxed load).
[[nodiscard]] u32 PinGeneration();
/// The logical processor (group 0 number) the calling thread runs on; -1 where unknown.
[[nodiscard]] int CurrentProcessorNumber();
/// FRITCHII_HOSTTIMING_OFF_GPU (live, default 0; core_timing.cpp): the mask HostTiming's spin
/// runs on with 1: every logical processor of the process except the physical cores (both
/// hardware threads) of the pinned GPU threads (front, back, R/GPUMem). 0 while none is pinned,
/// or when that would leave fewer than MinAutoGuestProcessors + 2 fast logical processors
/// (refused, logged once). `gpu_cores` gets the excluded cores.
[[nodiscard]] u64 HostTimingOffGpuMask(u64& gpu_cores);
/// THREAD_PRIORITY_IDLE on Windows, ThreadPriority::Low elsewhere.
void SetCurrentThreadIdlePriority();
void SetCurrentThreadToEfficiencyCores();
void SetCurrentThreadToBackgroundWork();
void SetCurrentThreadToAllCores();

void RefreshThreadPolicies();
void PollThreadPolicies();

/// How long to poll before blocking next time, after a wait of `waited` that polled `poll`, with
/// `limit` the longest poll. This is the cpuidle-haltpoll governor's rule (Linux,
/// drivers/cpuidle/governors/haltpoll.c): a wait a longer poll would have caught doubles the poll,
/// and one longer than the limit, which no poll catches, halves it. The poll never drops below an
/// eighth of the limit, so the quickest wake-ups stay cheap.
constexpr std::chrono::nanoseconds NextPollWindow(std::chrono::nanoseconds poll,
                                                  std::chrono::nanoseconds waited,
                                                  std::chrono::nanoseconds limit) {
    if (waited <= poll) {
        return poll;
    }
    if (waited <= limit) {
        return std::min(poll * 2, limit);
    }
    return std::max(poll / 2, limit / 8);
}

/// Waits until flag is true or the deadline passes, and returns the flag. Where the CPU has a
/// user-mode wait on an address (MWAITX on AMD, UMWAIT on Intel) the thread waits in it: a store to
/// the flag's cache line ends the wait within a few hundred nanoseconds, and the core neither
/// issues instructions nor enters the OS meanwhile. Other CPUs yield until then.
/// (only Tegra::Nx2PendingWake uses it, under EDEN_NX2_GPU_WAKE.)
bool WaitForFlag(const std::atomic<bool>& flag, std::chrono::steady_clock::time_point deadline);

/// FRITCHII_HWWAIT (boot, default 0), read once: 1 = spin waits that opt in wait in UMWAIT (Intel
/// WAITPKG, the fast-wake C0.1 state), 2 = MWAITX on AMD too. Returns what this CPU runs: 0 off
/// (also when it lacks the instruction or an invariant TSC), 1 UMWAIT, 2 MWAITX. Logged once.
[[nodiscard]] int HardwareWaitMode();

/// HardwareWaitMode() != 0: waits in hardware while `word` still holds `seen`, at most `cycles`
/// TSC cycles. A store to the word's cache line or an interrupt ends it early. The word is checked
/// again after the monitor is armed, so a store just before the wait is never missed.
void HardwareWaitWhileEqual(const std::atomic<u64>& word, u64 seen, u32 cycles);

/// TSC cycles one PAUSE takes on this CPU (measured once; 1 where there is no TSC).
[[nodiscard]] u32 PauseCycles();

/// Windows: one "CPU placement:" line with the CPU's performance, efficiency and low-power
/// efficiency logical processors (by efficiency class and last-level cache) and what the hybrid
/// placement does with them (EDEN_HYBRID_CORES, thread.cpp). Logging only.
void LogCpuPlacement();

/// Windows, an Intel CPU with performance and efficiency cores: the hybrid_cpu_placement setting
/// applies. False everywhere else.
[[nodiscard]] bool IntelHybridCpu();
/// EDEN_HYBRID_CORES's value (it overrides hybrid_cpu_placement), -1 when it isn't set.
[[nodiscard]] int HybridCoresSwitch();
/// ThreadPinning already gives the GPU threads performance cores of their own on this CPU, so
/// pinning them adds nothing.
[[nodiscard]] bool GpuThreadsHaveOwnCores();

/// Windows hybrid CPUs, from HostTiming's 10 s report: `late_1ms` guest timers ran 1 ms or more
/// late in `seconds`. Logs one "Hybrid placement in 10 s:" line (mode, time-critical threads,
/// how busy the performance / efficiency / low-power cores were) and runs strict placement's
/// timer storm guard. Nothing on other CPUs.
void NoteHostTimingWindow(double seconds, u64 late_1ms);

} // namespace Common
