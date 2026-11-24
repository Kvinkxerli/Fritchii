// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/adpf.h"
#include "common/error.h"
#include "common/event_wait.h"
#include "common/fritchii_lock_census.h"
#include "common/fritchii_write_ring.h"
#include "common/logging.h"
#include "common/lowend_sim.h"
#include "common/nx2_sample_profile.h"
#include "common/assert.h"
#include "common/thread.h"
#include "common/frame_budget.h"
#include "common/nx2_hitch.h"
#include "common/thread_pinning.h"
#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__HAIKU__)
#include <kernel/OS.h>
#elif defined(_WIN32)
#include <windows.h>
#include "common/settings.h"
#include "common/string_util.h"
#include "common/windows/fritchii_focus.h"
#include "common/windows/timer_resolution.h"
#else
#if defined(__FreeBSD__)
#include <sys/cpuset.h>
#include <sys/_cpuset.h>
#include <pthread_np.h>
// Compatibility with CPUset
#define cpu_set_t cpuset_t
#elif defined(__DragonFly__) || defined(__OpenBSD__) || defined(__Bitrig__)
#include <pthread_np.h>
#endif
#include <pthread.h>
#include <sched.h>
#endif

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __linux__
#include <sys/resource.h>
#include <algorithm>

namespace {
constexpr int NICE_AUDIO = -16;
constexpr int NICE_URGENT_DISPLAY = -8;
constexpr int NICE_DISPLAY = -4;
constexpr int NICE_DEFAULT = 0;
constexpr int NICE_BACKGROUND = 10;

int LowestAllowedNice() {
    static const int lowest = [] {
        rlimit limit{};
        if (getrlimit(RLIMIT_NICE, &limit) != 0) {
            return 0;
        }
        if (limit.rlim_cur >= 40) {
            return -20;
        }
        return 20 - static_cast<int>(limit.rlim_cur);
    }();
    return lowest;
}

int NiceValueForPriority(Common::ThreadPriority priority) {
    const int wanted = [priority] {
        switch (priority) {
        case Common::ThreadPriority::Low: return NICE_BACKGROUND;
        case Common::ThreadPriority::Normal: return NICE_DEFAULT;
        case Common::ThreadPriority::High: return NICE_DISPLAY;
        case Common::ThreadPriority::VeryHigh: return NICE_URGENT_DISPLAY;
        case Common::ThreadPriority::Critical: return NICE_AUDIO;
        default: return NICE_DEFAULT;
        }
    }();
    return (std::max)(wanted, (std::min)(NICE_DEFAULT, LowestAllowedNice()));
}
} // Anonymous namespace
#endif

#ifdef __ANDROID__
#include <sys/utsname.h>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <vector>

namespace {
constexpr size_t ANDROID_MINIMUM_PERFORMANCE_CORES = 4;

// A core counts as a performance core while it is within this much of the fastest one.
constexpr s64 ANDROID_PERFORMANCE_CAPACITY_PERCENT = 50;

constexpr std::chrono::nanoseconds ANDROID_POLICY_POLL_INTERVAL = std::chrono::milliseconds{500};

enum class CoreGroup {
    Unrestricted,
    Performance,
    Efficiency,
};

struct ThreadPolicy {
    pid_t tid;
    CoreGroup group;
    s32 nice_value;
    bool has_nice;
};

struct CoreInfo {
    s64 weight;
    u64 midr;
    s32 cpu;
};

struct CpuTopologyState {
    std::mutex topology_mutex;
    cpu_set_t allowed{};
    cpu_set_t performance{};
    cpu_set_t efficiency{};
    bool separated = false;
    bool initialized = false;

    pid_t canary_tid = 0;
    cpu_set_t canary_mask{};
    bool canary_valid = false;

    std::atomic<s64> next_poll_ns{0};

    std::mutex policy_mutex;
    std::vector<ThreadPolicy> policies;
};

CpuTopologyState& State() {
    static CpuTopologyState* const state = new CpuTopologyState();
    return *state;
}

struct PolicyRegistration {
    ~PolicyRegistration() {
        const pid_t tid = gettid();
        ::Common::ADPF::RemoveCurrentThread();
        CpuTopologyState& state = State();
        std::scoped_lock lock{state.policy_mutex};
        std::erase_if(state.policies,
                      [tid](const ThreadPolicy& policy) { return policy.tid == tid; });
    }
};

thread_local PolicyRegistration t_policy_registration;

s32 PossibleCpuCount() {
    std::ifstream file("/sys/devices/system/cpu/possible");
    std::string list;
    if (file && std::getline(file, list) && !list.empty()) {
        s64 highest = -1;
        const char* cursor = list.c_str();
        while (*cursor != '\0') {
            char* end = nullptr;
            const s64 value = std::strtol(cursor, &end, 10);
            if (end == cursor) {
                break;
            }
            highest = (std::max)(highest, value);
            cursor = end;
            while (*cursor == '-' || *cursor == ',') {
                ++cursor;
            }
        }
        if (highest >= 0) {
            return static_cast<s32>((std::min<s64>)(highest + 1, CPU_SETSIZE));
        }
    }
    const s64 configured = sysconf(_SC_NPROCESSORS_CONF);
    if (configured > 0) {
        return static_cast<s32>((std::min<s64>)(configured, CPU_SETSIZE));
    }
    return static_cast<s32>((std::min<s64>)(std::thread::hardware_concurrency(), CPU_SETSIZE));
}

s64 ReadCpuScalar(s32 cpu, const char* node) {
    s64 value = 0;
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + node);
    if (!file || !(file >> value) || value <= 0) {
        return 0;
    }
    return value;
}

u64 ReadCpuMidr(s32 cpu) {
    u64 midr = 0;
    std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                       "/regs/identification/midr_el1");
    if (!file || !(file >> std::hex >> midr)) {
        return 0;
    }
    return midr;
}

std::vector<CoreInfo> CollectCores(const cpu_set_t& allowed, s32 total, const char* node,
                                   bool require_all) {
    std::vector<CoreInfo> cores;
    for (s32 cpu = 0; cpu < total; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        const s64 weight = ReadCpuScalar(cpu, node);
        if (weight <= 0) {
            if (require_all) {
                return {};
            }
            LOG_WARNING(Common, "Could not read {} for CPU {}, treating it as an efficiency core",
                        node, cpu);
            continue;
        }
        cores.push_back(CoreInfo{weight, ReadCpuMidr(cpu), cpu});
    }
    return cores;
}

bool WeightsAreUniform(const std::vector<CoreInfo>& cores) {
    return std::all_of(cores.begin(), cores.end(),
                       [&](const CoreInfo& core) { return core.weight == cores.front().weight; });
}

bool MidrsAreDistinct(const std::vector<CoreInfo>& cores) {
    return std::none_of(cores.begin(), cores.end(),
                        [](const CoreInfo& core) { return core.midr == 0; }) &&
           std::any_of(cores.begin(), cores.end(),
                       [&](const CoreInfo& core) { return core.midr != cores.front().midr; });
}

void ComputeTopologyLocked(CpuTopologyState& state) {
    state.initialized = true;
    state.separated = false;
    CPU_ZERO(&state.allowed);
    CPU_ZERO(&state.performance);
    CPU_ZERO(&state.efficiency);

    if (sched_getaffinity(getpid(), sizeof(state.allowed), &state.allowed) != 0) {
        LOG_WARNING(Common, "Could not query process CPU affinity: {}",
                    ::Common::GetLastErrorMsg());
        return;
    }

    const s32 total = PossibleCpuCount();
    auto cores = CollectCores(state.allowed, total, "cpu_capacity", true);
    if (cores.empty() || (WeightsAreUniform(cores) && MidrsAreDistinct(cores))) {
        auto by_frequency = CollectCores(state.allowed, total, "cpufreq/cpuinfo_max_freq", false);
        if (!by_frequency.empty()) {
            cores = std::move(by_frequency);
        }
    }
    if (cores.empty()) {
        LOG_WARNING(Common, "Could not determine CPU topology, thread placement is disabled");
        return;
    }

    if (WeightsAreUniform(cores)) {
        if (MidrsAreDistinct(cores)) {
            LOG_WARNING(Common, "CPU clusters differ but rank identically, thread placement is "
                                "disabled");
        } else {
            LOG_INFO(Common, "CPU cores are symmetric, thread placement is disabled");
        }
        return;
    }

    std::sort(cores.begin(), cores.end(), [](const CoreInfo& lhs, const CoreInfo& rhs) {
        if (lhs.weight != rhs.weight) {
            return lhs.weight > rhs.weight;
        }
        return lhs.cpu < rhs.cpu;
    });

    const s64 fastest = cores.front().weight;
    size_t taken = 0;
    for (const auto& core : cores) {
        const bool fast_enough =
            core.weight * 100 >= fastest * ANDROID_PERFORMANCE_CAPACITY_PERCENT;
        if (!fast_enough && taken >= ANDROID_MINIMUM_PERFORMANCE_CORES) {
            break;
        }
        CPU_SET(core.cpu, &state.performance);
        ++taken;
    }
    if (taken == 0) {
        return;
    }

    for (s32 cpu = 0; cpu < total; ++cpu) {
        if (CPU_ISSET(cpu, &state.allowed) && !CPU_ISSET(cpu, &state.performance)) {
            CPU_SET(cpu, &state.efficiency);
        }
    }

    state.separated = CPU_COUNT(&state.efficiency) > 0;
    LOG_INFO(Common, "CPU topology: {} performance cores, {} efficiency cores, separation {}",
             CPU_COUNT(&state.performance), CPU_COUNT(&state.efficiency),
             state.separated ? "enabled" : "unavailable");
}

void EnsureTopologyLocked(CpuTopologyState& state) {
    if (!state.initialized) {
        ComputeTopologyLocked(state);
    }
}

void RefreshTopologyLocked(CpuTopologyState& state) {
    if (!state.initialized) {
        ComputeTopologyLocked(state);
        return;
    }
    cpu_set_t current;
    CPU_ZERO(&current);
    if (sched_getaffinity(getpid(), sizeof(current), &current) != 0) {
        return;
    }
    if (std::memcmp(&current, &state.allowed, sizeof(current)) != 0) {
        ComputeTopologyLocked(state);
    }
}

bool ApplyCoreGroupLocked(CpuTopologyState& state, pid_t tid, CoreGroup group,
                          bool* gone = nullptr) {
    const bool restrict_group = group != CoreGroup::Unrestricted && state.separated;
    const cpu_set_t* mask = &state.allowed;
    if (restrict_group) {
        mask = group == CoreGroup::Performance ? &state.performance : &state.efficiency;
    }
    if (CPU_COUNT(mask) == 0) {
        return false;
    }
    if (sched_setaffinity(tid, sizeof(*mask), mask) != 0) {
        if (gone != nullptr && errno == ESRCH) {
            *gone = true;
            return false;
        }
        LOG_WARNING(Common, "Could not restrict thread {} to its core group: {}", tid,
                    ::Common::GetLastErrorMsg());
        return false;
    }
    return true;
}

bool KernelPreservesRequestedAffinity() {
    utsname info{};
    if (uname(&info) != 0) {
        return false;
    }
    s32 major = 0;
    s32 minor = 0;
    if (std::sscanf(info.release, "%d.%d", &major, &minor) != 2) {
        return false;
    }
    return major > 6 || (major == 6 && minor >= 2);
}

void SnapshotCanaryLocked(CpuTopologyState& state) {
    state.canary_valid = false;
    if (!state.separated) {
        return;
    }
    for (const auto& policy : state.policies) {
        if (policy.group == CoreGroup::Unrestricted) {
            continue;
        }
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(policy.tid, sizeof(mask), &mask) != 0) {
            continue;
        }
        state.canary_tid = policy.tid;
        state.canary_mask = mask;
        state.canary_valid = true;
        return;
    }
}

bool DueForPoll(CpuTopologyState& state) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    s64 next = state.next_poll_ns.load(std::memory_order_relaxed);
    if (now_ns < next) {
        return false;
    }
    return state.next_poll_ns.compare_exchange_strong(
        next, now_ns + ANDROID_POLICY_POLL_INTERVAL.count(), std::memory_order_relaxed);
}

ThreadPolicy& AcquirePolicyLocked(CpuTopologyState& state, pid_t tid) {
    for (auto& policy : state.policies) {
        if (policy.tid == tid) {
            return policy;
        }
    }
    return state.policies.emplace_back(ThreadPolicy{tid, CoreGroup::Unrestricted, 0, false});
}

void SetCurrentThreadCoreGroup(CoreGroup group) {
    const pid_t tid = gettid();
    (void)&t_policy_registration;

    CoreGroup effective = group;
    if (tid == getpid() && group != CoreGroup::Unrestricted) {
        LOG_WARNING(Common, "Refusing to place the main thread: the CPU topology is read from it");
        effective = CoreGroup::Unrestricted;
    }

    CpuTopologyState& state = State();
    std::scoped_lock topology_lock{state.topology_mutex};
    EnsureTopologyLocked(state);
    ApplyCoreGroupLocked(state, tid, effective);

    std::scoped_lock policy_lock{state.policy_mutex};
    AcquirePolicyLocked(state, tid).group = effective;
    if (!state.canary_valid) {
        SnapshotCanaryLocked(state);
    }
}

void RememberCurrentThreadNice(pid_t tid, s32 nice_value) {
    (void)&t_policy_registration;
    CpuTopologyState& state = State();
    std::scoped_lock lock{state.policy_mutex};
    ThreadPolicy& policy = AcquirePolicyLocked(state, tid);
    policy.nice_value = nice_value;
    policy.has_nice = true;
}
} // Anonymous namespace
#endif

#include "common/cpu_features.h"
#ifdef ARCHITECTURE_x86_64
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "common/x64/rdtsc.h"
#include "video_core/nx2_debug_control.h"
#endif

#ifdef _WIN32
namespace {
// Hybrid CPUs (Intel P/E cores), EDEN_HYBRID_CORES (boot):
// - 2, the default: soft. The time-critical threads get no CPU set: the process is high QoS
//   (timer_resolution.cpp SetHighQoS), so Windows runs them on the performance cores first and
//   spills to the efficiency cores when those are full. Only the idle and low priority background
//   workers that ask for it (quiet pipeline prebuilds, TextureDecoder) keep to the other cores.
// - 1: strict with guards. The time-critical threads are held to the performance cores with CPU
//   sets, but only while they fit (no more of them than the performance cores hold, an SMT
//   sibling counted as half a core, one core left for Windows and the driver) and while guest
//   timers stay on time (two 10 s windows with over 50 late timers a second release it).
// - 3: strict without guards (v6.1's placement, to compare against).
// - 4: GPU pinned, also the "Pin GPU to P-cores" setting (hybrid_cpu_placement, Intel only, read
//   when a game starts; the switch wins when set). As soft, but the threads ThreadPinning would
//   give cores of their own (GPU front, GPU back, and the third one: the render thread R with one
//   follower, GPUMem with the memory stage) are held to the performance cores with CPU sets.
//   VulkanWorker stays soft: ThreadPinning leaves it free too, and on 4 P-core CPUs it wouldn't
//   fit beside the guest cores. Guards as in strict: they must fit, and two 10 s windows with
//   over 50 late timers a second release them. Where ThreadPinning already pins those threads
//   (6 or more P-cores with enough left for the guest cores) there is nothing to add.
// - 0: off, every thread on every core.
// Why soft (helper ES, 2026-11-18): about 12 threads call SetCurrentThreadToPerformanceCores,
// several of them spin (HostTiming up to 600 us per event, the GPU workers). On a 6 P-core Core
// Ultra 7 255H strict placement queued them behind each other on 6 cores while 10 efficiency
// cores idled: guest timers ran late (2 -> 175 a second), the game scheduled 7x the timer events,
// the P-cores' clock fell (225% -> 158%) and fps fell 30.9 -> 13.3. No Intel hybrid CPU has
// enough P-cores for that thread count.
enum class HybridMode { Off, Soft, Strict, StrictUnguarded, GpuPinned };

struct WindowsCpuSets {
    std::vector<ULONG> performance;
    std::vector<ULONG> efficiency;
    /// Group 0 logical processors per class, for the busy split (NoteHostTimingWindow).
    u64 performance_mask = 0;
    u64 efficiency_mask = 0; ///< efficiency cores beside the performance cores (shared cache)
    u64 low_power_mask = 0;  ///< an island of slower cores without the performance cores' cache
    size_t performance_cores = 0; ///< physical
    /// Time-critical threads strict placement keeps on the performance cores (an SMT sibling is
    /// half a core, one core left for Windows and the driver).
    size_t capacity = 0;
    HybridMode mode = HybridMode::Off; ///< from EDEN_HYBRID_CORES; see ActiveHybridMode
    bool mode_from_switch = false;     ///< EDEN_HYBRID_CORES is set: the setting doesn't count
    bool cpu_hybrid = false; ///< the CPU has performance and efficiency cores
    bool hybrid = false;     ///< cpu_hybrid and placement not off
};

/// EDEN_HYBRID_CORES's value, -1 when it isn't set.
int HybridSwitchValue() {
    const char* const text = std::getenv("EDEN_HYBRID_CORES");
    if (text == nullptr || text[0] == '\0') {
        return -1;
    }
    return static_cast<int>(std::strtol(text, nullptr, 10));
}

HybridMode ParseHybridMode() {
    switch (HybridSwitchValue()) {
    case 0:
        return HybridMode::Off;
    case 1:
        return HybridMode::Strict;
    case 3:
        return HybridMode::StrictUnguarded;
    case 4:
        return HybridMode::GpuPinned;
    default:
        return HybridMode::Soft;
    }
}

const char* HybridModeName(HybridMode mode, bool from_switch) {
    switch (mode) {
    case HybridMode::Off:
        return "off (EDEN_HYBRID_CORES=0)";
    case HybridMode::Soft:
        return "soft (EDEN_HYBRID_CORES=2, the default)";
    case HybridMode::Strict:
        return "strict with guards (EDEN_HYBRID_CORES=1)";
    case HybridMode::StrictUnguarded:
        return "strict without guards (EDEN_HYBRID_CORES=3, v6.1's placement)";
    case HybridMode::GpuPinned:
        return from_switch ? "GPU pinned to P-cores (EDEN_HYBRID_CORES=4)"
                           : "GPU pinned to P-cores (the Pin GPU to P-cores setting)";
    }
    return "?";
}

const WindowsCpuSets& CpuSets() {
    static const WindowsCpuSets sets = [] {
        WindowsCpuSets result;
        result.mode = ParseHybridMode();
        result.mode_from_switch = HybridSwitchValue() >= 0;
        using PFN_GetSystemCpuSetInformation = BOOL(WINAPI*)(PSYSTEM_CPU_SET_INFORMATION, ULONG,
                                                             PULONG, HANDLE, ULONG);
        const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        const auto get = reinterpret_cast<PFN_GetSystemCpuSetInformation>(
            reinterpret_cast<void*>(GetProcAddress(kernel32, "GetSystemCpuSetInformation")));
        if (get == nullptr) {
            return result;
        }
        ULONG length = 0;
        get(nullptr, 0, &length, GetCurrentProcess(), 0);
        if (length == 0) {
            return result;
        }
        std::vector<u8> buffer(length);
        if (!get(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), length, &length,
                 GetCurrentProcess(), 0)) {
            return result;
        }
        struct Entry {
            ULONG id;
            u8 efficiency_class;
            u32 cache; ///< (group << 8) | last-level cache index
            u32 core;  ///< (group << 8) | core index
            u8 group;
            u8 processor;
        };
        std::vector<Entry> entries;
        u8 best = 0;
        for (ULONG offset = 0; offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= length;) {
            const auto* info =
                reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
            if (info->Size == 0) {
                break;
            }
            if (info->Type == CpuSetInformation) {
                const u32 group = info->CpuSet.Group;
                entries.push_back({info->CpuSet.Id, info->CpuSet.EfficiencyClass,
                                   (group << 8) | info->CpuSet.LastLevelCacheIndex,
                                   (group << 8) | info->CpuSet.CoreIndex,
                                   static_cast<u8>(group), info->CpuSet.LogicalProcessorIndex});
                best = (std::max)(best, static_cast<u8>(info->CpuSet.EfficiencyClass));
            }
            offset += info->Size;
        }
        std::vector<u32> top_caches;
        std::vector<u32> top_cores;
        for (const Entry& entry : entries) {
            if (entry.efficiency_class != best) {
                continue;
            }
            if (std::find(top_caches.begin(), top_caches.end(), entry.cache) == top_caches.end()) {
                top_caches.push_back(entry.cache);
            }
            if (std::find(top_cores.begin(), top_cores.end(), entry.core) == top_cores.end()) {
                top_cores.push_back(entry.core);
            }
        }
        u64 low_power = 0;
        for (const Entry& entry : entries) {
            const u64 bit = entry.group == 0 && entry.processor < 64
                                ? u64{1} << entry.processor
                                : 0;
            if (entry.efficiency_class == best) {
                result.performance.push_back(entry.id);
                result.performance_mask |= bit;
            } else {
                result.efficiency.push_back(entry.id);
                if (std::find(top_caches.begin(), top_caches.end(), entry.cache) !=
                    top_caches.end()) {
                    result.efficiency_mask |= bit;
                } else {
                    low_power |= bit;
                }
            }
        }
        if (result.efficiency_mask == 0) {
            // Only an island of slower cores: those are the efficiency cores.
            result.efficiency_mask = low_power;
        } else {
            result.low_power_mask = low_power;
        }
        result.performance_cores = top_cores.size();
        const size_t siblings = result.performance.size() - (std::min)(result.performance.size(),
                                                                         top_cores.size());
        const size_t whole = result.performance_cores + siblings / 2;
        result.capacity = whole > 1 ? whole - 1 : 1;
        result.cpu_hybrid = !result.efficiency.empty() && result.performance.size() >= 2;
        result.hybrid = result.cpu_hybrid && result.mode != HybridMode::Off;
        return result;
    }();
    return sets;
}

bool IntelCpu() {
#ifdef ARCHITECTURE_x86_64
    return Common::g_cpu_caps.manufacturer == Common::CPUCaps::Manufacturer::Intel;
#else
    return false;
#endif
}

/// The placement now: EDEN_HYBRID_CORES when set, else the setting (Intel only, a config copied
/// from another PC does nothing elsewhere). The setting can't change while a game runs.
HybridMode ActiveHybridMode(const WindowsCpuSets& sets) {
    if (sets.mode_from_switch) {
        return sets.mode;
    }
    return sets.cpu_hybrid && IntelCpu() &&
                   Settings::values.hybrid_cpu_placement.GetValue() ==
                       Settings::HybridCpuPlacement::PinGpu
               ? HybridMode::GpuPinned
               : HybridMode::Soft;
}

/// Threads that asked for the performance cores (SetCurrentThreadToPerformanceCores), live ones
/// only, with a handle strict placement can release them through.
struct TimeCriticalThread {
    HANDLE handle;
    DWORD id;
};
std::mutex g_time_critical_mutex;
std::vector<TimeCriticalThread> g_time_critical;
std::atomic<size_t> g_time_critical_count{};
/// Strict placement given up (too many threads, or a timer storm); every thread on every core.
std::atomic<bool> g_strict_released{false};
/// GPU pinned: the threads held to the performance cores (under g_time_critical_mutex).
std::vector<TimeCriticalThread> g_gpu_pinned;
std::atomic<size_t> g_gpu_pinned_count{};
/// GPU pinned given up for this game session.
std::atomic<bool> g_gpu_pin_released{false};

void DropExitedThreads(std::vector<TimeCriticalThread>& threads) {
    std::erase_if(threads, [](const TimeCriticalThread& thread) {
        DWORD code = 0;
        if (GetExitCodeThread(thread.handle, &code) && code != STILL_ACTIVE) {
            CloseHandle(thread.handle);
            return true;
        }
        return false;
    });
}

using SetThreadSelectedCpuSetsFn = BOOL(WINAPI*)(HANDLE, const ULONG*, ULONG);
SetThreadSelectedCpuSetsFn SelectCpuSetsFn() {
    static const auto set = reinterpret_cast<SetThreadSelectedCpuSetsFn>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadSelectedCpuSets")));
    return set;
}

/// Adds the calling thread (once) and returns how many time-critical threads are alive.
size_t RegisterTimeCriticalThread() {
    const DWORD id = GetCurrentThreadId();
    std::scoped_lock lock{g_time_critical_mutex};
    DropExitedThreads(g_time_critical);
    const bool known = std::any_of(g_time_critical.begin(), g_time_critical.end(),
                                   [id](const TimeCriticalThread& thread) { return thread.id == id; });
    HANDLE handle = nullptr;
    if (!known && DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                                  &handle,
                                  THREAD_SET_LIMITED_INFORMATION |
                                      THREAD_QUERY_LIMITED_INFORMATION,
                                  FALSE, 0)) {
        g_time_critical.push_back({handle, id});
    }
    g_time_critical_count.store(g_time_critical.size(), std::memory_order_relaxed);
    return g_time_critical.size();
}

/// Strict placement: every time-critical thread back on every core, once.
void ReleaseStrictPlacement(const std::string& why) {
    if (g_strict_released.exchange(true)) {
        return;
    }
    size_t released = 0;
    {
        std::scoped_lock lock{g_time_critical_mutex};
        if (const auto set = SelectCpuSetsFn(); set != nullptr) {
            for (const TimeCriticalThread& thread : g_time_critical) {
                released += set(thread.handle, nullptr, 0) ? 1 : 0;
            }
        }
    }
    LOG_WARNING(Common,
                "Hybrid placement: strict placement released ({}); {} time-critical threads back "
                "on every core, Windows places them from now on",
                why, released);
}

void SelectCurrentThreadCpuSets(const std::vector<ULONG>& ids) {
    const auto set = SelectCpuSetsFn();
    if (set == nullptr) {
        return;
    }
    if (!set(GetCurrentThread(), ids.empty() ? nullptr : ids.data(),
             static_cast<ULONG>(ids.size()))) {
        LOG_WARNING(Common, "SetThreadSelectedCpuSets failed: {}", Common::GetLastErrorMsg());
    }
}

/// GPU pinned: the held GPU threads back on every core, once per game session.
void ReleaseGpuPin(const std::string& why) {
    if (g_gpu_pin_released.exchange(true)) {
        return;
    }
    size_t released = 0;
    {
        std::scoped_lock lock{g_time_critical_mutex};
        if (const auto set = SelectCpuSetsFn(); set != nullptr) {
            for (const TimeCriticalThread& thread : g_gpu_pinned) {
                released += set(thread.handle, nullptr, 0) ? 1 : 0;
            }
        }
    }
    LOG_WARNING(Common,
                "Hybrid placement: GPU pin released ({}); {} GPU threads back on every core, "
                "Windows places them from now on",
                why, released);
}

/// GPU pinned: holds the calling GPU thread to the performance cores, while they all fit.
void HoldGpuThreadOnPerformanceCores() {
    const WindowsCpuSets& sets = CpuSets();
    if (!sets.hybrid || ActiveHybridMode(sets) != HybridMode::GpuPinned) {
        return;
    }
    const DWORD id = GetCurrentThreadId();
    size_t live = 0;
    {
        std::scoped_lock lock{g_time_critical_mutex};
        DropExitedThreads(g_gpu_pinned);
        if (g_gpu_pinned.empty()) {
            g_gpu_pin_released.store(false); // A new game session tries again.
        }
        const bool known =
            std::any_of(g_gpu_pinned.begin(), g_gpu_pinned.end(),
                        [id](const TimeCriticalThread& thread) { return thread.id == id; });
        HANDLE handle = nullptr;
        if (!known && DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
                                      GetCurrentProcess(), &handle,
                                      THREAD_SET_LIMITED_INFORMATION |
                                          THREAD_QUERY_LIMITED_INFORMATION,
                                      FALSE, 0)) {
            g_gpu_pinned.push_back({handle, id});
        }
        live = g_gpu_pinned.size();
        g_gpu_pinned_count.store(live, std::memory_order_relaxed);
    }
    if (g_gpu_pin_released.load()) {
        return;
    }
    SelectCurrentThreadCpuSets(sets.performance);
    if (g_gpu_pin_released.load()) {
        // Released between the check and the selection: the release may have missed this one.
        SelectCurrentThreadCpuSets({});
        return;
    }
    if (live > sets.capacity) {
        ReleaseGpuPin(fmt::format("{} GPU threads, the {} performance cores hold {}", live,
                                  sets.performance_cores, sets.capacity));
    }
}

/// Masks of processor group 0's logical processors that share a `relation` (a physical core, or
/// with RelationCache an L3 cache).
std::vector<KAFFINITY> GroupZeroMasks(LOGICAL_PROCESSOR_RELATIONSHIP relation) {
    std::vector<KAFFINITY> result;
    DWORD length = 0;
    GetLogicalProcessorInformationEx(relation, nullptr, &length);
    std::vector<u8> buffer(length);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
    if (length == 0 || !GetLogicalProcessorInformationEx(relation, info, &length)) {
        return result;
    }
    for (DWORD offset = 0; offset < length;) {
        auto* entry =
            reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (entry->Relationship == RelationProcessorCore && entry->Processor.GroupCount > 0 &&
            entry->Processor.GroupMask[0].Group == 0) {
            result.push_back(entry->Processor.GroupMask[0].Mask);
        } else if (entry->Relationship == RelationCache && entry->Cache.Level == 3 &&
                   entry->Cache.GroupMask.Group == 0) {
            result.push_back(entry->Cache.GroupMask.Mask);
        }
        offset += entry->Size;
    }
    return result;
}

/// Logical processors PinCurrentThread gave the GPU front and back threads.
std::atomic<u64> pinned_gpu_mask{};

/// Each pinned GPU role's mask and thread id, the latest thread pinned to the role wins (round 1
/// kept the first: the memory stage's GPUMem pins the GpuMem role before one follower's R does,
/// so R's own time read 0). For FRITCHII_SIBLING_CENSUS's split and FRITCHII_GPU_SIBLINGS_FREE=2,
/// which never moves these threads.
struct PinnedThread {
    std::atomic<u64> mask{};
    std::atomic<u32> tid{};
};
std::array<PinnedThread, 4> pinned_threads{}; ///< by PinnedRole (GuestCore unused)
std::atomic<u32> pin_generation{}; ///< PinGeneration

bool SiblingCensusOn() {
    static const bool on = [] {
        const char* const value = std::getenv("FRITCHII_SIBLING_CENSUS");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return on;
}

/// FRITCHII_SIBLING_CENSUS: notes by thread and logical processor (group 0).
std::array<std::array<std::atomic<u32>, 64>, static_cast<size_t>(Common::CensusThread::Count)>
    census_notes{};

Common::ThreadPinning::Topology ReadPinningTopology() {
    Common::ThreadPinning::Topology topology;
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &length);
    std::vector<u8> buffer(length);
    if (length == 0 ||
        !GetLogicalProcessorInformationEx(
            RelationAll, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()),
            &length)) {
        return topology;
    }
    for (DWORD offset = 0; offset < length;) {
        const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            buffer.data() + offset);
        if (entry->Size == 0) {
            break;
        }
        if (entry->Relationship == RelationProcessorCore) {
            const PROCESSOR_RELATIONSHIP& core = entry->Processor;
            for (WORD i = 0; i < core.GroupCount; ++i) {
                topology.logical_processors +=
                    static_cast<u32>(std::popcount(static_cast<u64>(core.GroupMask[i].Mask)));
            }
            if (core.GroupCount > 0 && core.GroupMask[0].Group == 0) {
                topology.cores.push_back(
                    {static_cast<u64>(core.GroupMask[0].Mask), core.EfficiencyClass});
            }
        } else if (entry->Relationship == RelationCache) {
            const CACHE_RELATIONSHIP& cache = entry->Cache;
            if (cache.Level == 3 && cache.Type != CacheInstruction &&
                cache.GroupMask.Group == 0) {
                topology.l3.push_back({static_cast<u64>(cache.GroupMask.Mask), cache.CacheSize});
            }
        }
        offset += entry->Size;
    }
    // FRITCHII_SIM_CPU: the plan sees the simulated CPU, as it would on a PC with those cores.
    if (const u64 allowed = Common::LowEndSim::CpuMask(); allowed != 0) {
        std::erase_if(topology.cores, [allowed](auto& core) {
            core.mask &= allowed;
            return core.mask == 0;
        });
        std::erase_if(topology.l3, [allowed](auto& cache) {
            cache.mask &= allowed;
            return cache.mask == 0;
        });
        topology.logical_processors = static_cast<u32>(std::popcount(allowed));
    }
    return topology;
}
} // Anonymous namespace
#endif

namespace Common {

void SetCurrentThreadPriority(ThreadPriority new_priority) {
#ifdef _WIN32
    int windows_priority = [&]() {
        switch (new_priority) {
        case ThreadPriority::Low: return THREAD_PRIORITY_BELOW_NORMAL;
        case ThreadPriority::Normal: return THREAD_PRIORITY_NORMAL;
        case ThreadPriority::High: return THREAD_PRIORITY_ABOVE_NORMAL;
        case ThreadPriority::VeryHigh: return THREAD_PRIORITY_HIGHEST;
        case ThreadPriority::Critical: return THREAD_PRIORITY_TIME_CRITICAL;
        default: return THREAD_PRIORITY_NORMAL;
        }
    }();
    SetThreadPriority(GetCurrentThread(), windows_priority);
    // FRITCHII_THREAD_QOS tags these later (docs/fritchii/switches.md).
    if (new_priority >= ThreadPriority::High) {
        Windows::FritchiiFocus::RegisterHotThread();
    }
#elif defined(__HAIKU__)
    // TODO: We have priorities for 3D rendering applications - may help lavapipe?
    int priority = [&]() {
        switch (new_priority) {
        case ThreadPriority::Low: return B_LOW_PRIORITY;
        case ThreadPriority::Normal: return B_NORMAL_PRIORITY;
        case ThreadPriority::High: return B_DISPLAY_PRIORITY;
        case ThreadPriority::VeryHigh: return B_URGENT_DISPLAY_PRIORITY;
        case ThreadPriority::Critical: return B_URGENT_PRIORITY;
        default: return B_NORMAL_PRIORITY;
        }
    }();
    set_thread_priority(find_thread(NULL), priority);
#elif defined(__ANDROID__)
    const int nice_value = NiceValueForPriority(new_priority);
    const pid_t tid = gettid();
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), nice_value) != 0) {
        LOG_WARNING(Common, "Could not set thread nice value to {}: {}", nice_value,
                    GetLastErrorMsg());
        return;
    }
    RememberCurrentThreadNice(tid, nice_value);
#elif defined(__linux__)
    const int nice_value = NiceValueForPriority(new_priority);
    if (setpriority(PRIO_PROCESS, 0, nice_value) != 0) {
        LOG_DEBUG(Common, "Could not set thread nice value to {}: {}", nice_value,
                  GetLastErrorMsg());
    }
#else
    const s32 max_prio = sched_get_priority_max(SCHED_OTHER);
    const s32 min_prio = sched_get_priority_min(SCHED_OTHER);
    if (max_prio > min_prio) {
        const u32 level = (std::min)(static_cast<u32>(new_priority), 4U);
        sched_param params{};
        params.sched_priority =
            min_prio + static_cast<s32>(static_cast<u32>(max_prio - min_prio) * level) / 4;
        pthread_setschedparam(pthread_self(), SCHED_OTHER, &params);
    }
#endif
}

u64 GetCurrentNativeThreadId() {
#ifdef _WIN32
    return static_cast<u64>(GetCurrentThreadId());
#elif defined(__linux__)
    return static_cast<u64>(gettid());
#else
    return 0;
#endif
}

int GetNativeThreadNice(u64 native_id) {
#ifdef _WIN32
    if (native_id == 0) {
        return THREAD_PRIORITY_ERROR_RETURN;
    }
    HANDLE handle = OpenThread(THREAD_QUERY_INFORMATION, FALSE, static_cast<DWORD>(native_id));
    if (handle == nullptr) {
        return THREAD_PRIORITY_ERROR_RETURN;
    }
    const int value = GetThreadPriority(handle);
    CloseHandle(handle);
    return value;
#elif defined(__linux__)
    if (native_id == 0) {
        return 0;
    }
    errno = 0;
    const int value = getpriority(PRIO_PROCESS, static_cast<id_t>(native_id));
    return errno != 0 ? 0 : value;
#else
    (void)native_id;
    return 0;
#endif
}

void SetNativeThreadPriority(u64 native_id, ThreadPriority new_priority) {
    if (native_id == 0) {
        return;
    }
#ifdef _WIN32
    int windows_priority = THREAD_PRIORITY_NORMAL;
    switch (new_priority) {
    case ThreadPriority::Low:
        windows_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    case ThreadPriority::Normal:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    case ThreadPriority::High:
        windows_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case ThreadPriority::VeryHigh:
        windows_priority = THREAD_PRIORITY_HIGHEST;
        break;
    case ThreadPriority::Critical:
        windows_priority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    default:
        break;
    }
    HANDLE handle = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE,
                               static_cast<DWORD>(native_id));
    if (handle == nullptr) {
        return;
    }
    SetThreadPriority(handle, windows_priority);
    CloseHandle(handle);
#elif defined(__linux__)
    const int nice_value = NiceValueForPriority(new_priority);
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(native_id), nice_value) != 0) {
        LOG_DEBUG(Common, "Could not set thread {} nice value to {}: {}", native_id, nice_value,
                  GetLastErrorMsg());
    }
#else
    (void)new_priority;
#endif
}

void SetNativeThreadNice(u64 native_id, int nice_value) {
    if (native_id == 0) {
        return;
    }
#ifdef _WIN32
    if (nice_value == THREAD_PRIORITY_ERROR_RETURN) {
        return;
    }
    HANDLE handle = OpenThread(THREAD_SET_INFORMATION, FALSE, static_cast<DWORD>(native_id));
    if (handle == nullptr) {
        return;
    }
    SetThreadPriority(handle, nice_value);
    CloseHandle(handle);
#elif defined(__linux__)
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(native_id), nice_value) != 0) {
        LOG_DEBUG(Common, "Could not restore thread {} nice value to {}: {}", native_id, nice_value,
                  GetLastErrorMsg());
    }
#else
    (void)nice_value;
#endif
}

void SetCurrentThreadName(const char* name) {
    Nx2SampleProfile::OnThreadNamed(name);
    Nx2Hitch::RegisterThread(name);
    LockCensus::NameThread(name);
    FrameBudget::RegisterThread(name);
    WriteRing::NameThread(name);
#ifdef _MSC_VER
    // Sets the debugger-visible name of the current thread.
    if (auto pf = (decltype(&SetThreadDescription))(void*)GetProcAddress(GetModuleHandle(TEXT("KernelBase.dll")), "SetThreadDescription"); pf)
        pf(GetCurrentThread(), UTF8ToUTF16W(name).data()); // Windows 10+
    else
        ; // No-op
#elif  defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__HAIKU__)
    rename_thread(find_thread(NULL), name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__) || defined(__CYGWIN__) || defined(__sun__) || defined(__glibc__) || defined(__managarm__)
    int ret = pthread_setname_np(pthread_self(), name);
    if (ret == ERANGE) {
        // Linux limits thread names to 15 characters and will outright reject any
        // attempt to set a longer name with ERANGE.
        char buf[16];
        size_t const len = std::min<size_t>(std::strlen(name), sizeof(buf) - 1);
        std::memcpy(buf, name, len);
        buf[len] = '\0';
        pthread_setname_np(pthread_self(), buf);
    }
#elif defined(_WIN32)
    // MinGW with the POSIX threading model does not support pthread_setname_np
    // See for reference
    // https://gitlab.freedesktop.org/mesa/mesa/-/blame/main/src/util/u_thread.c?ref_type=heads#L75
    (void)name;
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void SetCurrentThreadToPerformanceCores() {
#if defined(__ANDROID__)
    if (ADPF::AddCurrentThread(ADPF::Session::Render)) {
        SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
        return;
    }
    SetCurrentThreadCoreGroup(CoreGroup::Performance);
#elif defined(_WIN32)
    const WindowsCpuSets& sets = CpuSets();
    if (!sets.hybrid) {
        return;
    }
    // Every mode counts them (the "Hybrid placement in 10 s" line); soft leaves them to Windows,
    // and so does GPU pinned (its GPU threads are held in PinCurrentThread).
    const size_t live = RegisterTimeCriticalThread();
    const HybridMode mode = ActiveHybridMode(sets);
    if (mode == HybridMode::Soft || mode == HybridMode::GpuPinned ||
        g_strict_released.load(std::memory_order_relaxed)) {
        return;
    }
    SelectCurrentThreadCpuSets(sets.performance);
    if (g_strict_released.load()) {
        // Released between the check and the selection: the release may have missed this one.
        SelectCurrentThreadCpuSets({});
        return;
    }
    if (mode == HybridMode::Strict && live > sets.capacity) {
        ReleaseStrictPlacement(fmt::format("{} time-critical threads, the {} performance cores "
                                           "hold {}",
                                           live, sets.performance_cores, sets.capacity));
    }
#endif
}

void PinCurrentThread(PinnedRole role, int guest_core) {
#ifdef _WIN32
    // Off with EDEN_NX2_PIN_THREADS=0, forced with 1, otherwise on with enough performance cores
    // (ThreadPinning::MakePlan): the split's back did about 8% more draws a second at Lake Hylia
    // (Ryzen 7 7800X3D), its worst window 29 -> 54 fps, once no SMT sibling shared its core.
    static u64 mem_mask = 0;
    static const ThreadPinning::Plan plan = [] {
        const auto mode = ThreadPinning::ParseMode(std::getenv("EDEN_NX2_PIN_THREADS"));
        const ThreadPinning::Topology topology = ReadPinningTopology();
        ThreadPinning::Plan result = ThreadPinning::MakePlan(topology, mode);
        LOG_INFO(Common, "Thread pinning: {}", ThreadPinning::Describe(result, topology, mode));
        // FRITCHII_PIN_GPUMEM=1: the last physical core of the guest set goes to GPUMem alone.
        const char* const pin_mem = std::getenv("FRITCHII_PIN_GPUMEM");
        // One follower (FRITCHII_ONEF=1, boot): its render thread R takes the same core unless
        // FRITCHII_ONEF_PIN=0 (default 1; video_core/one_follower.h). R and GPUMem never run
        // together: the memory stage stays off with one follower on.
        // FRITCHII_ONEF_LEAN=1 turns one follower on too.
        const char* const onef = std::getenv("FRITCHII_ONEF");
        const char* const onef_lean = std::getenv("FRITCHII_ONEF_LEAN");
        const char* const onef_pin = std::getenv("FRITCHII_ONEF_PIN");
        const bool pin_render = ((onef != nullptr && onef[0] == '1') ||
                                 (onef_lean != nullptr && onef_lean[0] == '1')) &&
                                (onef_pin == nullptr || onef_pin[0] != '0');
        // FRITCHII_ONEF_PIN=3 (fable_b 24): R on the front's idle SMT sibling; the front keeps
        // one hardware thread of its core and the guest cores keep all of normal's threads.
        if (pin_render && onef_pin != nullptr && onef_pin[0] == '3' &&
            std::popcount(result.front) > 1) {
            mem_mask = u64{1} << (63 - std::countl_zero(result.front));
            result.front &= ~mem_mask;
            LOG_INFO(Common,
                     "Thread pinning: GPURender {:#x} (FRITCHII_ONEF_PIN=3, the front's sibling), "
                     "front now {:#x}, guest cores {:#x}",
                     mem_mask, result.front, result.guest);
        } else if (((pin_mem != nullptr && pin_mem[0] == '1') || pin_render) &&
                   result.guest != 0) {
            for (auto it = topology.cores.rbegin(); it != topology.cores.rend(); ++it) {
                if (it->mask != 0 && (it->mask & ~result.guest) == 0 &&
                    (result.guest & ~it->mask) != 0) {
                    mem_mask = it->mask;
                    // FRITCHII_ONEF_PIN=2 (fable_b 20.7): R on one hardware thread of the core,
                    // its SMT sibling stays with the guest cores (the whole core cost the game
                    // about 7% of its path; unpinned, R slowed the GPU side more).
                    if (pin_render && onef_pin != nullptr && onef_pin[0] == '2' &&
                        std::popcount(mem_mask) > 1) {
                        mem_mask = u64{1} << (63 - std::countl_zero(mem_mask));
                    }
                    result.guest &= ~mem_mask;
                    break;
                }
            }
            LOG_INFO(Common, "Thread pinning: {} {:#x} ({}), guest cores now {:#x}",
                     pin_render ? "GPURender" : "GPUMem", mem_mask,
                     pin_render ? "FRITCHII_ONEF=1" : "FRITCHII_PIN_GPUMEM=1", result.guest);
        }
        return result;
    }();
    u64 mask{};
    switch (role) {
    case PinnedRole::GpuFront:
        mask = plan.front;
        break;
    case PinnedRole::GpuBack:
        mask = plan.back;
        break;
    case PinnedRole::GpuMem:
        mask = mem_mask;
        break;
    case PinnedRole::GuestCore: {
        mask = plan.guest;
        // EDEN_NX2_GUEST_PIN_EACH=1: one physical core each (the guest cores' JIT code was 73-80%
        // of their busy time at Castle Town; two on one core's SMT siblings share its units).
        static const std::vector<u64> guest_cores = [] {
            std::vector<u64> masks;
            const char* value = std::getenv("EDEN_NX2_GUEST_PIN_EACH");
            if (plan.guest == 0 || value == nullptr || value[0] != '1') {
                return masks;
            }
            for (const ThreadPinning::Core& core : ReadPinningTopology().cores) {
                if (core.mask != 0 && (core.mask & ~plan.guest) == 0) {
                    masks.push_back(core.mask);
                }
            }
            LOG_INFO(Common, "Thread pinning: each guest core on its own physical core ({} cores)",
                     masks.size());
            return masks;
        }();
        if (guest_core >= 0 && !guest_cores.empty()) {
            mask = guest_cores[static_cast<size_t>(guest_core) % guest_cores.size()];
        }
        break;
    }
    }
    if (mask != 0) {
        GROUP_AFFINITY affinity{};
        affinity.Mask = static_cast<KAFFINITY>(mask);
        affinity.Group = 0;
        if (!SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr)) {
            LOG_WARNING(Common, "Could not pin thread: {}", GetLastErrorMsg());
        } else if (role != PinnedRole::GuestCore) {
            pinned_gpu_mask.fetch_or(mask, std::memory_order_relaxed);
            // FRITCHII_SIBLING_CENSUS / FRITCHII_GPU_SIBLINGS_FREE=2 (the Pulse thread reads it).
            PinnedThread& pinned = pinned_threads[static_cast<size_t>(role)];
            pinned.mask.store(mask, std::memory_order_relaxed);
            pinned.tid.store(static_cast<u32>(GetCurrentThreadId()), std::memory_order_release);
            pin_generation.fetch_add(1, std::memory_order_release);
        }
    } else if (role != PinnedRole::GuestCore && !plan.Enabled()) {
        // No core of its own on this CPU: EDEN_HYBRID_CORES=4 or the setting can still hold it
        // to the performance cores.
        HoldGpuThreadOnPerformanceCores();
    }
#else
    (void)role;
#endif
}

bool IntelHybridCpu() {
#ifdef _WIN32
    return CpuSets().cpu_hybrid && IntelCpu();
#else
    return false;
#endif
}

int HybridCoresSwitch() {
#ifdef _WIN32
    return HybridSwitchValue();
#else
    return -1;
#endif
}

bool GpuThreadsHaveOwnCores() {
#ifdef _WIN32
    static const bool pinned =
        ThreadPinning::MakePlan(ReadPinningTopology(),
                                ThreadPinning::ParseMode(std::getenv("EDEN_NX2_PIN_THREADS")))
            .Enabled();
    return pinned;
#else
    return false;
#endif
}

u64 BackgroundWorkCpuMask() {
#ifdef _WIN32
    const u64 pinned = pinned_gpu_mask.load(std::memory_order_relaxed);
    if (pinned == 0) {
        return 0;
    }
    DWORD_PTR process{};
    DWORD_PTR system{};
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system)) {
        return 0;
    }
    const u64 rest = static_cast<u64>(process) & ~pinned;
    if (rest == 0) {
        return 0;
    }
    // Several L3 caches (Ryzen 9 with two CCDs): the other CCD's cores, so the work does not
    // evict what the GPU threads keep in theirs, when that leaves enough of them.
    static const std::vector<KAFFINITY> l3_caches = GroupZeroMasks(RelationCache);
    u64 other_caches = rest;
    for (const KAFFINITY cache : l3_caches) {
        if ((static_cast<u64>(cache) & pinned) != 0) {
            other_caches &= ~static_cast<u64>(cache);
        }
    }
    return std::popcount(other_caches) >= 4 ? other_caches : rest;
#else
    return 0;
#endif
}

void SetCurrentThreadCpuMask(u64 mask) {
#ifdef _WIN32
    if (mask == 0) {
        DWORD_PTR process{};
        DWORD_PTR system{};
        if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system) || process == 0) {
            return;
        }
        mask = process;
    }
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(mask));
#else
    static_cast<void>(mask);
#endif
}

void NoteCensusCpu(CensusThread who) {
#ifdef _WIN32
    if (!SiblingCensusOn()) {
        return;
    }
    PROCESSOR_NUMBER processor{};
    GetCurrentProcessorNumberEx(&processor);
    if (processor.Group == 0 && processor.Number < 64) {
        census_notes[static_cast<size_t>(who)][processor.Number].fetch_add(
            1, std::memory_order_relaxed);
    }
#else
    static_cast<void>(who);
#endif
}

#if defined(_WIN32) && defined(_WIN64)
namespace {

using NtQuerySystemFn = LONG(WINAPI*)(ULONG, PVOID, ULONG, PULONG);
using NtQueryThreadFn = LONG(WINAPI*)(HANDLE, ULONG, PVOID, ULONG, PULONG);

NtQuerySystemFn NtQuerySystem() {
    static const auto function = reinterpret_cast<NtQuerySystemFn>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation")));
    return function;
}

NtQueryThreadFn NtQueryThread() {
    static const auto function = reinterpret_cast<NtQueryThreadFn>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread")));
    return function;
}

/// The logical processors of group 0's fastest cores (all of them on a CPU that is not hybrid).
u64 FastestProcessors() {
    static const u64 mask = [] {
        const Common::ThreadPinning::Topology topology = ReadPinningTopology();
        u8 fastest = 0;
        for (const auto& core : topology.cores) {
            fastest = (std::max)(fastest, core.efficiency_class);
        }
        u64 result = 0;
        for (const auto& core : topology.cores) {
            if (core.efficiency_class == fastest) {
                result |= core.mask;
            }
        }
        return result;
    }();
    return mask;
}

/// FRITCHII_GPU_SIBLINGS_FREE's target mask: BackgroundWorkCpuMask, but only where it leaves
/// the guest cores room (at least MinAutoGuestProcessors + 2 fast logical processors), so a
/// smaller CPU never gets more threads on fewer processors: a 7800X3D keeps 11 (on); a 6-core
/// with SMT keeps 7 once R takes its hardware thread (off); 8 cores without SMT keep 5 (off).
/// 0 while nothing is pinned, or when refused (`refused` set, logged once).
u64 SiblingsFreeMask(bool& refused) {
    refused = false;
    const u64 mask = Common::BackgroundWorkCpuMask();
    if (mask == 0) {
        return 0;
    }
    const int left = std::popcount(mask & FastestProcessors());
    const int needed = static_cast<int>(Common::ThreadPinning::MinAutoGuestProcessors) + 2;
    if (left < needed) {
        refused = true;
        static std::atomic<bool> logged{};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            LOG_INFO(Common,
                     "FRITCHII_GPU_SIBLINGS_FREE: off on this CPU: {} fast logical processors "
                     "left beside the GPU threads' cores, {} needed",
                     left, needed);
        }
        return 0;
    }
    return mask;
}

// NtQuerySystemInformation(SystemProcessInformation) records, x64 layout.
struct ProcessRecord {
    ULONG next_entry_offset;
    ULONG number_of_threads;
    u8 skipped[0x50 - 8];
    HANDLE unique_process_id; ///< 0x50
    u8 rest[0x100 - 0x58];
};
static_assert(sizeof(ProcessRecord) == 0x100);
struct ThreadRecord {
    LARGE_INTEGER kernel_time;
    LARGE_INTEGER user_time;
    LARGE_INTEGER create_time;
    ULONG wait_time;
    PVOID start_address;
    HANDLE unique_process;
    HANDLE unique_thread; ///< 0x30
    LONG priority;
    LONG base_priority;
    ULONG context_switches;
    ULONG thread_state;
    ULONG wait_reason;
};
static_assert(sizeof(ThreadRecord) == 0x50);

/// NtQueryInformationThread(ThreadBasicInformation).
struct ThreadBasic {
    LONG exit_status;
    PVOID teb;
    HANDLE unique_process;
    HANDLE unique_thread;
    KAFFINITY affinity;
    LONG priority;
    LONG base_priority;
};

/// One of this process's threads as the Pulse thread last saw it.
struct OwnThread {
    HANDLE handle{};
    std::string name;
    u64 cpu_100ns{};     ///< kernel + user, the latest read
    u64 cpu_delta{};     ///< since the read before
    u64 affinity{};      ///< the latest read
    bool seen{};         ///< in the latest enumeration
    u64 moved_from{};    ///< FRITCHII_GPU_SIBLINGS_FREE=2: the mask it had before (0 = not moved)
    u64 moved_to{};
};

/// Only the Pulse thread touches these.
std::unordered_map<u32, OwnThread> g_own_threads;
std::vector<u8> g_process_buffer;

/// The thread's description (SetThreadDescription), else its start routine's module and id.
std::string NameOf(HANDLE handle, u32 tid) {
    using GetDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get_description = reinterpret_cast<GetDescriptionFn>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
    if (get_description != nullptr) {
        PWSTR description = nullptr;
        if (SUCCEEDED(get_description(handle, &description)) && description != nullptr) {
            std::string name = Common::UTF16ToUTF8(std::wstring_view{description});
            LocalFree(description);
            if (!name.empty()) {
                return name;
            }
        }
    }
    std::string module = "?";
    PVOID start = nullptr;
    constexpr ULONG ThreadQuerySetWin32StartAddress = 9;
    if (const auto query = NtQueryThread();
        query != nullptr &&
        query(handle, ThreadQuerySetWin32StartAddress, &start, sizeof(start), nullptr) >= 0 &&
        start != nullptr) {
        HMODULE owner = nullptr;
        wchar_t path[MAX_PATH]{};
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCWSTR>(start), &owner) &&
            GetModuleFileNameW(owner, path, MAX_PATH) != 0) {
            const std::wstring_view full{path};
            const size_t slash = full.find_last_of(L"\\/");
            module = Common::UTF16ToUTF8(
                slash == std::wstring_view::npos ? full : full.substr(slash + 1));
        }
    }
    return fmt::format("tid {} {}", tid, module);
}

/// Reads this process's threads (one NtQuerySystemInformation call, ~150 threads here): CPU
/// time, affinity, a cached handle and name. False when the call fails.
bool RefreshOwnThreads() {
    const auto query = NtQuerySystem();
    if (query == nullptr) {
        return false;
    }
    constexpr ULONG SystemProcessInformation = 5;
    if (g_process_buffer.empty()) {
        g_process_buffer.resize(1 << 20);
    }
    ULONG length = 0;
    LONG status = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
        status = query(SystemProcessInformation, g_process_buffer.data(),
                       static_cast<ULONG>(g_process_buffer.size()), &length);
        constexpr LONG InfoLengthMismatch = static_cast<LONG>(0xC0000004L);
        if (status != InfoLengthMismatch) {
            break;
        }
        g_process_buffer.resize((std::max<size_t>)(length + (64 << 10), g_process_buffer.size() * 2));
    }
    if (status < 0) {
        return false;
    }
    const auto own_pid = static_cast<uintptr_t>(GetCurrentProcessId());
    const u8* record = g_process_buffer.data();
    const ProcessRecord* process = nullptr;
    for (;;) {
        const auto* candidate = reinterpret_cast<const ProcessRecord*>(record);
        if (reinterpret_cast<uintptr_t>(candidate->unique_process_id) == own_pid) {
            process = candidate;
            break;
        }
        if (candidate->next_entry_offset == 0) {
            break;
        }
        record += candidate->next_entry_offset;
    }
    if (process == nullptr) {
        return false;
    }
    for (auto& [tid, thread] : g_own_threads) {
        thread.seen = false;
    }
    const auto* threads = reinterpret_cast<const ThreadRecord*>(process + 1);
    const auto query_thread = NtQueryThread();
    for (ULONG i = 0; i < process->number_of_threads; ++i) {
        const ThreadRecord& info = threads[i];
        const auto tid = static_cast<u32>(reinterpret_cast<uintptr_t>(info.unique_thread));
        const u64 cpu = static_cast<u64>(info.kernel_time.QuadPart + info.user_time.QuadPart);
        auto [it, inserted] = g_own_threads.try_emplace(tid);
        OwnThread& thread = it->second;
        if (inserted) {
            thread.handle = OpenThread(THREAD_QUERY_INFORMATION | THREAD_SET_INFORMATION, FALSE,
                                       static_cast<DWORD>(tid));
            thread.name = thread.handle != nullptr ? NameOf(thread.handle, tid)
                                                   : fmt::format("tid {}", tid);
            thread.cpu_100ns = cpu;
        }
        thread.cpu_delta = cpu >= thread.cpu_100ns ? cpu - thread.cpu_100ns : 0;
        thread.cpu_100ns = cpu;
        thread.seen = true;
        ThreadBasic basic{};
        constexpr ULONG ThreadBasicInformation = 0;
        if (thread.handle != nullptr && query_thread != nullptr &&
            query_thread(thread.handle, ThreadBasicInformation, &basic, sizeof(basic), nullptr) >=
                0) {
            thread.affinity = static_cast<u64>(basic.affinity);
        }
    }
    for (auto it = g_own_threads.begin(); it != g_own_threads.end();) {
        if (!it->second.seen) {
            if (it->second.handle != nullptr) {
                CloseHandle(it->second.handle);
            }
            it = g_own_threads.erase(it);
        } else {
            ++it;
        }
    }
    return true;
}

/// Threads that place themselves (HostTiming's spin, the pipeline prebuild workers) or that
/// KeepOffGpuCores places per unit of work: mode 2 leaves them alone, so turning it off never
/// undoes their own placement.
bool SelfPlaced(const std::string& name) {
    return name.starts_with("HostTiming") || name.starts_with("VkPipelinePrebuild") ||
           name.starts_with("VkPrecompile") || name == "VulkanWorker" ||
           name == "GPUFencingThread" || name == "VulkanPresent";
}

bool PinnedTid(u32 tid) {
    for (const auto& pinned : pinned_threads) {
        if (pinned.tid.load(std::memory_order_relaxed) == tid && tid != 0) {
            return true;
        }
    }
    return false;
}

/// FRITCHII_GPU_SIBLINGS_FREE=2, once a second on the Pulse thread: every thread of this process
/// still on the process's whole mask (nobody placed it: driver threads, audio, VSync, the Pulse
/// thread, image frees...) gets the siblings-free mask; the pinned GPU threads and anything with
/// a narrower mask (guest cores, HostTiming, pipeline workers, E2) keep theirs. Any other value
/// gives the moved threads their old mask back (if nobody changed it since). Returns the threads
/// moved this pass.
u32 SweepOffGpuCores(int mode, u64& kept_off) {
    u32 moved = 0;
    kept_off = 0;
    DWORD_PTR process_mask{};
    DWORD_PTR system_mask{};
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        return 0;
    }
    bool refused = false;
    const u64 target = mode == 2 ? SiblingsFreeMask(refused) : 0;
    for (auto& [tid, thread] : g_own_threads) {
        if (thread.handle == nullptr) {
            continue;
        }
        if (target == 0) {
            if (thread.moved_from != 0) {
                if (thread.affinity == thread.moved_to) {
                    SetThreadAffinityMask(thread.handle, static_cast<DWORD_PTR>(thread.moved_from));
                }
                thread.moved_from = thread.moved_to = 0;
            }
            continue;
        }
        if (thread.moved_from != 0) {
            ++kept_off;
            continue;
        }
        if (thread.affinity != static_cast<u64>(process_mask) || PinnedTid(tid) ||
            SelfPlaced(thread.name)) {
            continue;
        }
        if (SetThreadAffinityMask(thread.handle, static_cast<DWORD_PTR>(target)) != 0) {
            thread.moved_from = thread.affinity;
            thread.moved_to = target;
            thread.affinity = target;
            ++moved;
            ++kept_off;
        }
    }
    return moved;
}

} // Anonymous namespace
#endif

u32 PinGeneration() {
#ifdef _WIN32
    return pin_generation.load(std::memory_order_acquire);
#else
    return 0;
#endif
}

int CurrentProcessorNumber() {
#ifdef _WIN32
    PROCESSOR_NUMBER processor{};
    GetCurrentProcessorNumberEx(&processor);
    return processor.Number;
#else
    return -1;
#endif
}

u64 HostTimingOffGpuMask(u64& gpu_cores) {
    gpu_cores = 0;
#if defined(_WIN32) && defined(_WIN64)
    static const std::vector<KAFFINITY> cores = GroupZeroMasks(RelationProcessorCore);
    for (size_t role = 0; role < pinned_threads.size(); ++role) {
        const u64 mask = pinned_threads[role].mask.load(std::memory_order_relaxed);
        if (role == static_cast<size_t>(PinnedRole::GuestCore) || mask == 0) {
            continue;
        }
        for (const KAFFINITY core : cores) {
            if ((static_cast<u64>(core) & mask) != 0) {
                gpu_cores |= static_cast<u64>(core);
            }
        }
    }
    if (gpu_cores == 0) {
        return 0;
    }
    DWORD_PTR process{};
    DWORD_PTR system{};
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system)) {
        return 0;
    }
    const u64 mask = static_cast<u64>(process) & ~gpu_cores;
    const int left = std::popcount(mask & FastestProcessors());
    const int needed = static_cast<int>(ThreadPinning::MinAutoGuestProcessors) + 2;
    if (left < needed) {
        static std::atomic<bool> logged{};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            LOG_INFO(Common,
                     "FRITCHII_HOSTTIMING_OFF_GPU: off on this CPU: {} fast logical processors "
                     "left beside the GPU threads' cores {:#x}, {} needed",
                     left, gpu_cores, needed);
        }
        return 0;
    }
    return mask;
#else
    return 0;
#endif
}

void KeepOffGpuCores(int& applied) {
#if defined(_WIN32) && defined(_WIN64) && defined(ARCHITECTURE_x86_64)
    // `applied`: bit 0 = the switch value handled (0 or on), bit 1 = this thread's mask was set.
    const int wanted = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_gpu_siblings_free,
                                                        "FRITCHII_GPU_SIBLINGS_FREE") != 0
                           ? 1
                           : 0;
    if ((applied & 1) == wanted) [[likely]] {
        return;
    }
    if (wanted != 0) {
        bool refused = false;
        const u64 mask = SiblingsFreeMask(refused);
        if (mask == 0 && !refused) {
            return; // Nothing pinned yet (the GPU threads pin as they start): next time.
        }
        if (mask != 0) {
            SetCurrentThreadCpuMask(mask);
            applied = 3;
        } else {
            applied = 1; // Refused on this CPU: nothing changes.
        }
    } else {
        if ((applied & 2) != 0) {
            SetCurrentThreadCpuMask(0);
        }
        applied = 0;
    }
#else
    static_cast<void>(applied);
#endif
}

std::string GpuCoreTick() {
#if defined(_WIN32) && defined(_WIN64)
    const int siblings_free =
#ifdef ARCHITECTURE_x86_64
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_gpu_siblings_free,
                                         "FRITCHII_GPU_SIBLINGS_FREE");
#else
        0;
#endif
    const bool census = SiblingCensusOn();
    // The thread list is read when something needs it: the census, mode 2, or threads mode 2
    // moved that wait to get their mask back.
    static bool sweeping = false;
    if (!census && siblings_free != 2 && !sweeping) {
        return {};
    }
    const bool threads_ok = RefreshOwnThreads();
    u32 moved = 0;
    u64 kept_off = 0;
    if (threads_ok && (siblings_free == 2 || sweeping)) {
        moved = SweepOffGpuCores(siblings_free, kept_off);
        sweeping = siblings_free == 2;
    }
    if (!census) {
        return {};
    }

    // Per logical processor busy time (kernel - idle + user, 100 ns units; kernel holds idle,
    // DPCs and interrupts) and DPC + interrupt time, from
    // NtQuerySystemInformation(SystemProcessorPerformanceInformation).
    struct ProcessorTimes {
        LARGE_INTEGER idle;
        LARGE_INTEGER kernel;
        LARGE_INTEGER user;
        LARGE_INTEGER dpc;
        LARGE_INTEGER interrupt;
        ULONG interrupts;
    };
    const auto query = NtQuerySystem();
    if (query == nullptr) {
        return {};
    }
    constexpr ULONG SystemProcessorPerformanceInformation = 8;
    std::array<ProcessorTimes, 64> times{};
    ULONG length = 0;
    if (query(SystemProcessorPerformanceInformation, times.data(),
              static_cast<ULONG>(sizeof(times)), &length) < 0) {
        return {};
    }
    const size_t processors = (std::min<size_t>)(length / sizeof(ProcessorTimes), times.size());
    // The physical core around each role's mask (R's mask is one logical processor of its core).
    static const std::vector<KAFFINITY> cores = GroupZeroMasks(RelationProcessorCore);
    const auto core_of = [&](u64 mask) {
        u64 core = 0;
        for (const KAFFINITY c : cores) {
            if ((static_cast<u64>(c) & mask) != 0) {
                core |= static_cast<u64>(c);
            }
        }
        return core;
    };

    struct Last {
        u64 ns{};
        std::array<u64, 64> busy{};
        std::array<u64, 64> dpc{};
        std::array<std::array<u32, 64>, static_cast<size_t>(CensusThread::Count)> notes{};
    };
    static Last last;
    const u64 now_ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now().time_since_epoch())
                                            .count());
    std::array<u64, 64> busy{};
    std::array<u64, 64> dpc{};
    for (size_t i = 0; i < processors; ++i) {
        busy[i] = static_cast<u64>(times[i].kernel.QuadPart - times[i].idle.QuadPart +
                                   times[i].user.QuadPart);
        dpc[i] = static_cast<u64>(times[i].dpc.QuadPart + times[i].interrupt.QuadPart);
    }
    std::array<std::array<u32, 64>, static_cast<size_t>(CensusThread::Count)> notes{};
    for (size_t who = 0; who < notes.size(); ++who) {
        for (size_t i = 0; i < 64; ++i) {
            notes[who][i] = census_notes[who][i].load(std::memory_order_relaxed);
        }
    }
    const bool first = last.ns == 0;
    const double span_ms = static_cast<double>(now_ns - last.ns) / 1e6;
    std::string text;
    if (!first && span_ms > 0.0) {
        // ms of CPU per second of wall time (the OS counts in clock ticks, ~15.6 ms steps).
        const auto per_s = [&](u64 delta_100ns) {
            return static_cast<double>(delta_100ns) / 1e4 / span_ms * 1000.0;
        };
        static constexpr std::array<const char*, 4> role_names{"front", "back", "", "R"};
        static constexpr std::array<const char*, 4> thread_names{"GPU", "GPUBack", "",
                                                                 "GPURender/GPUMem"};
        static constexpr std::array<const char*, static_cast<size_t>(CensusThread::Count)>
            who_names{"VulkanWorker", "GPUFencingThread", "VulkanPresent"};
        text = "; GPU cores (FRITCHII_SIBLING_CENSUS):";
        std::array<u64, 4> role_core{};
        for (size_t role = 0; role < pinned_threads.size(); ++role) {
            const u64 mask = pinned_threads[role].mask.load(std::memory_order_relaxed);
            if (role == static_cast<size_t>(PinnedRole::GuestCore) || mask == 0) {
                continue;
            }
            const u64 core = core_of(mask);
            role_core[role] = core;
            u64 core_busy = 0;
            u64 core_dpc = 0;
            for (size_t i = 0; i < processors; ++i) {
                if ((core >> i) & 1) {
                    core_busy += busy[i] - last.busy[i];
                    core_dpc += dpc[i] - last.dpc[i];
                }
            }
            const u32 owner_tid = pinned_threads[role].tid.load(std::memory_order_acquire);
            u64 own = 0;
            if (const auto it = g_own_threads.find(owner_tid); it != g_own_threads.end()) {
                own = it->second.cpu_delta;
            }
            text += fmt::format(" {} core {:#x} busy {:.0f}, {} {:.0f}, others {:.0f} (DPC+ISR "
                                "{:.0f}) ms/s",
                                role_names[role], core, per_s(core_busy), thread_names[role],
                                per_s(own), per_s(core_busy > own ? core_busy - own : 0),
                                per_s(core_dpc));
            // The others, named as far as Windows tells without a kernel trace: this process's
            // busiest threads whose affinity lets them onto the core's free hardware thread (they
            // may have run anywhere their mask allows), and what all such threads used in all.
            const u64 free_part = core & ~mask;
            std::vector<const OwnThread*> allowed;
            u64 allowed_cpu = 0;
            for (const auto& [tid, thread] : g_own_threads) {
                if (tid == owner_tid || (thread.affinity & (free_part != 0 ? free_part : core)) == 0 ||
                    thread.cpu_delta == 0) {
                    continue;
                }
                allowed.push_back(&thread);
                allowed_cpu += thread.cpu_delta;
            }
            const size_t shown = (std::min<size_t>)(allowed.size(), 3);
            std::partial_sort(allowed.begin(), allowed.begin() + shown, allowed.end(),
                              [](const OwnThread* a, const OwnThread* b) {
                                  return a->cpu_delta > b->cpu_delta;
                              });
            text += fmt::format(", own threads allowed there {:.0f} ms/s in {}", per_s(allowed_cpu),
                                allowed.size());
            for (size_t i = 0; i < shown; ++i) {
                text += fmt::format("{} {} {:.0f}", i == 0 ? ": " : ", ", allowed[i]->name,
                                    per_s(allowed[i]->cpu_delta));
            }
            text += ";";
        }
        // Where the noted threads ran: their notes on each role's core, of all their notes.
        for (size_t who = 0; who < notes.size(); ++who) {
            u64 all = 0;
            std::array<u64, 4> on_role{};
            for (size_t i = 0; i < 64; ++i) {
                const u64 delta = notes[who][i] - last.notes[who][i];
                all += delta;
                for (size_t role = 0; role < role_core.size(); ++role) {
                    if ((role_core[role] >> i) & 1) {
                        on_role[role] += delta;
                    }
                }
            }
            if (all == 0) {
                continue;
            }
            const auto share = [&](PinnedRole role) {
                return 100.0 * static_cast<double>(on_role[static_cast<size_t>(role)]) /
                       static_cast<double>(all);
            };
            text += fmt::format(" {} {:.0f}/s on front/back/R cores {:.0f}/{:.0f}/{:.0f}%;",
                                who_names[who], static_cast<double>(all) * 1000.0 / span_ms,
                                share(PinnedRole::GpuFront), share(PinnedRole::GpuBack),
                                share(PinnedRole::GpuMem));
        }
        text += fmt::format(" FRITCHII_GPU_SIBLINGS_FREE={}", siblings_free);
        if (siblings_free == 2 || kept_off != 0) {
            text += fmt::format(" (threads kept off {}, moved this second {})", kept_off, moved);
        }
        if (!threads_ok) {
            text += " (thread list unreadable)";
        }
    }
    last.ns = now_ns;
    last.busy = busy;
    last.dpc = dpc;
    last.notes = notes;
    return text;
#else
    return {};
#endif
}

void SetCurrentThreadIdlePriority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
#else
    SetCurrentThreadPriority(ThreadPriority::Low);
#endif
}

void SetCurrentThreadToEfficiencyCores() {
#if defined(__ANDROID__)
    if (ADPF::AddCurrentThread(ADPF::Session::Background)) {
        SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
        return;
    }
    SetCurrentThreadCoreGroup(CoreGroup::Efficiency);
#elif defined(_WIN32)
    if (const WindowsCpuSets& sets = CpuSets(); sets.hybrid) {
        SelectCurrentThreadCpuSets(sets.efficiency);
    }
#endif
}

void SetCurrentThreadToBackgroundWork() {
#if defined(__ANDROID__)
    ADPF::AddCurrentThread(ADPF::Session::Background);
    SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
#elif defined(_WIN32)
    if (const WindowsCpuSets& sets = CpuSets(); sets.hybrid) {
        SelectCurrentThreadCpuSets({});
    }
#endif
}

void SetCurrentThreadToAllCores() {
#if defined(__ANDROID__)
    ADPF::RemoveCurrentThread();
    SetCurrentThreadCoreGroup(CoreGroup::Unrestricted);
#elif defined(_WIN32)
    if (const WindowsCpuSets& sets = CpuSets(); sets.hybrid) {
        SelectCurrentThreadCpuSets({});
    }
#endif
}

void RefreshThreadPolicies() {
#if defined(__ANDROID__)
    CpuTopologyState& state = State();
    std::scoped_lock topology_lock{state.topology_mutex};
    RefreshTopologyLocked(state);

    std::scoped_lock policy_lock{state.policy_mutex};
    std::erase_if(state.policies, [&state](const ThreadPolicy& policy) {
        bool gone = false;
        if (policy.has_nice &&
            setpriority(PRIO_PROCESS, static_cast<id_t>(policy.tid), policy.nice_value) != 0 &&
            errno == ESRCH) {
            gone = true;
        }
        if (!gone) {
            ApplyCoreGroupLocked(state, policy.tid, policy.group, &gone);
        }
        return gone;
    });
    SnapshotCanaryLocked(state);
#endif
}

void PollThreadPolicies() {
#if defined(__ANDROID__)
    static const bool needed = !KernelPreservesRequestedAffinity();
    if (!needed) {
        return;
    }

    CpuTopologyState& state = State();
    if (!DueForPoll(state)) {
        return;
    }

    pid_t tid;
    cpu_set_t expected;
    {
        std::scoped_lock lock{state.topology_mutex};
        if (!state.canary_valid) {
            return;
        }
        tid = state.canary_tid;
        expected = state.canary_mask;
    }

    cpu_set_t current;
    CPU_ZERO(&current);
    if (sched_getaffinity(tid, sizeof(current), &current) == 0 &&
        std::memcmp(&current, &expected, sizeof(current)) == 0) {
        return;
    }
    RefreshThreadPolicies();
#endif
}

#ifdef ARCHITECTURE_x86_64
// On Linux and UNIX systems, a futex would nominally be used to cover the costs
// the idea is that it's intuitivelly cheaper to use a direct instruction as opposed to a full futex call
// the underlying libc++ implementation uses pthread_cond_timedwait which MAY invoke a futex
// Let's pretend the OS is too expensive to jump into, and avoid ANY context switches
// this should *IN THEORY* lower CPU usage while just waiting for stuff effectively
// For windows the minimal quanta resolution is about 500us, and normal CRT cond var is 1.5ms(?)
// so may as well avoid that too
// Let's just give ALL platforms the same mechanisms (almost) for when they have umonitor OR waitpkg
#ifdef __clang__
__attribute__((target("waitpkg,mwaitx")))
#elif defined(__GNUC__)
#pragma GCC target("waitpkg")
#pragma GCC target("mwaitx")
#endif
bool Event::WaitFor(const std::chrono::nanoseconds time) {
#ifdef _WIN32
    auto const& caps = Common::g_cpu_caps;
    const auto deadline = EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()), time);
    const auto remaining = [deadline] {
        return deadline - std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch());
    };
    const u64 ticks_per_second = Common::g_wall_clock.NsToTicks(std::chrono::seconds{1});
    if (caps.monitorx) {
        while (true) {
            // Armed monitor, as per manual, MWAITX must be conditional if the condition isn't satisfied
            // to prevent a race condition.
            _mm_monitorx(reinterpret_cast<u64*>(std::addressof(is_set)), 0, 0);
            if (!is_set.load()) {
                const u32 cycles = MwaitxTimeoutCycles(remaining(), ticks_per_second);
                if (cycles != 0) {
                    // See here: https://github.com/torvalds/linux/blob/948a64995aca6820abefd17f1a4258f5835c5ad9/arch/x86/lib/delay.c#L93
                    // MWAITX accepts a 32-bit input timer which determines the total number of cycles to wait for
                    // NOT THE TOTAL ABSOLUTE TSC VALUE, it's just a delta
                    // BIT[1] = use a timer
                    // Hint = 0: Use C1 state when sleepy (means slower wakeup but better savings)
                    _mm_mwaitx(1 << 1, 0u, cycles);
                    continue;
                } else
                    return false; //timeout
            }
            bool expected = true;
            if (is_set.compare_exchange_weak(expected, false, std::memory_order_release))
                return true;
        }
    }
    if (caps.waitpkg) {
        // #UD If CPUID.7.0:ECX.WAITPKG[bit 5]=0.
        while (true) {
            _umonitor(std::addressof(is_set));
            if (!is_set.load()) {
                // UMWAIT also wakes early on interrupts, and its result (CF) only reports the OS
                // limit (IA32_UMWAIT_CONTROL). Wait again until the deadline, as for MWAITX. The
                // bounded budget keeps the absolute TSC deadline from wrapping.
                const u32 cycles = MwaitxTimeoutCycles(remaining(), ticks_per_second);
                if (cycles == 0)
                    return false; // timeout
                _umwait(0, Common::X64::FencedRDTSC() + cycles); // umwait is absolute time!!!
                continue;
            }
            bool expected = true;
            if (is_set.compare_exchange_weak(expected, false, std::memory_order_release))
                return true;
        }
    }
    while (!is_set.load() && remaining() > std::chrono::nanoseconds::zero())
        Common::Windows::SleepForOneTick();
    bool expected = true;
    return is_set.compare_exchange_strong(expected, false, std::memory_order_release);
#else
    std::unique_lock lk{mutex};
    const auto now = std::chrono::steady_clock::now();
    const auto deadline = std::chrono::steady_clock::time_point{EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()), time)};
    if (!condvar.wait_until(lk, deadline, [this] { return is_set.load(); }))
        return false;
    is_set = false;
    return true;
#endif
}
#else
bool Event::WaitFor(const std::chrono::nanoseconds time) {
#ifdef _WIN32
    auto const end = Common::g_wall_clock.GetTimeNS() + time;
    while (!is_set.load() && end > Common::g_wall_clock.GetTimeNS())
        Common::Windows::SleepForOneTick();
    if (is_set.load())
        Reset();
    return true;
#else
    std::unique_lock lk{mutex};
    const auto now = std::chrono::steady_clock::now();
    const auto deadline = std::chrono::steady_clock::time_point{EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()), time)};
    if (!condvar.wait_until(lk, deadline, [this] { return is_set.load(); }))
        return false;
    is_set = false;
    return true;
#endif
}
#endif

// WaitForFlag and Event::SleepFor: ported changes.
#if defined(ARCHITECTURE_x86_64) && defined(__clang__)
__attribute__((target("waitpkg,mwaitx")))
#endif
bool WaitForFlag(const std::atomic<bool>& flag, std::chrono::steady_clock::time_point deadline) {
#ifdef ARCHITECTURE_x86_64
    if (const auto& caps = Common::g_cpu_caps; caps.monitorx || caps.waitpkg) {
        auto* const line = const_cast<std::atomic<bool>*>(std::addressof(flag));
        const u64 ticks_per_second = Common::g_wall_clock.NsToTicks(std::chrono::seconds{1});
        while (true) {
            if (caps.monitorx) {
                _mm_monitorx(line, 0, 0);
            } else {
                _umonitor(line);
            }
            if (flag.load(std::memory_order_acquire)) {
                return true;
            }
            // Armed before the load, so a store after it ends the wait. Interrupts and stores to
            // the rest of the line end it early too: wait again until the deadline.
            const u32 cycles = MwaitxTimeoutCycles(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    deadline - std::chrono::steady_clock::now()),
                ticks_per_second);
            if (cycles == 0) {
                return false;
            }
            if (caps.monitorx) {
                _mm_mwaitx(1 << 1, 0u, cycles);
            } else {
                _umwait(0, Common::X64::FencedRDTSC() + cycles);
            }
        }
    }
#endif
    // EXP103 (live, pauses; fable52_answer_yield 7): without MONITORX/WAITPKG (this owner's PC:
    // g_cpu_caps.monitorx false, ~795 ms/s of SwitchToThread over six cores) poll with N pauses
    // between checks instead of yielding to the scheduler: a better neighbour on a shared core.
    if (const int pauses = VideoCore::Nx2Debug::Experiment(103); pauses > 0) {
        while (!flag.load(std::memory_order_acquire)) {
            for (int i = 0; i < pauses; ++i) {
#ifdef ARCHITECTURE_x86_64
                _mm_pause();
#endif
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
        }
        return flag.load(std::memory_order_acquire);
    }
    while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    return flag.load(std::memory_order_acquire);
}

int HardwareWaitMode() {
    static const int mode = [] {
        const char* text = std::getenv("FRITCHII_HWWAIT");
        const int requested = text == nullptr ? 0 : std::atoi(text);
        if (requested <= 0) {
            return 0;
        }
        int result = 0;
#ifdef ARCHITECTURE_x86_64
        const auto& caps = Common::g_cpu_caps;
        if (Common::g_wall_clock.IsNative()) {
            if (caps.waitpkg) {
                result = 1;
            } else if (requested >= 2 && caps.monitorx) {
                result = 2;
            }
        }
        LOG_INFO(Common, "FRITCHII_HWWAIT={}: {} (CPU: WAITPKG {}, MONITORX {}, TSC clock {})",
                 requested,
                 result == 1   ? "spin waits use UMWAIT (C0.1)"
                 : result == 2 ? "spin waits use MWAITX"
                               : "off, this CPU has no usable hardware wait",
                 caps.waitpkg ? "yes" : "no", caps.monitorx ? "yes" : "no",
                 Common::g_wall_clock.IsNative() ? "yes" : "no");
#else
        LOG_INFO(Common, "FRITCHII_HWWAIT={}: off, x86-64 only", requested);
#endif
        return result;
    }();
    return mode;
}

#if defined(ARCHITECTURE_x86_64) && defined(__clang__)
__attribute__((target("waitpkg,mwaitx")))
#endif
void HardwareWaitWhileEqual(const std::atomic<u64>& word, u64 seen, u32 cycles) {
#ifdef ARCHITECTURE_x86_64
    const int mode = HardwareWaitMode();
    if (mode == 0 || cycles == 0) {
        return;
    }
    auto* const line = const_cast<std::atomic<u64>*>(std::addressof(word));
    if (mode == 1) {
        _umonitor(line);
    } else {
        _mm_monitorx(line, 0, 0);
    }
    if (word.load(std::memory_order_acquire) != seen) {
        return;
    }
    if (mode == 1) {
        // Control bit 0 set: C0.1, the faster wake (Event::WaitFor uses C0.2). Absolute deadline.
        _umwait(1, Common::X64::FencedRDTSC() + cycles);
    } else {
        _mm_mwaitx(1 << 1, 0u, cycles);
    }
#else
    (void)word;
    (void)seen;
    (void)cycles;
#endif
}

void LogCpuPlacement() {
#ifdef _WIN32
    const WindowsCpuSets& sets = CpuSets();
    if (sets.performance.empty() && sets.efficiency.empty()) {
        LOG_INFO(Common, "CPU placement: no CPU set information on this Windows");
        return;
    }
    const int efficiency = std::popcount(sets.efficiency_mask);
    const int low_power = std::popcount(sets.low_power_mask);
    const HybridMode mode = ActiveHybridMode(sets);
    const char* const mode_name = HybridModeName(mode, sets.mode_from_switch);
    std::string what;
    if (!sets.cpu_hybrid) {
        what = "off (not a hybrid CPU)";
    } else {
        switch (mode) {
        case HybridMode::Off:
            what = mode_name;
            break;
        case HybridMode::Soft:
            what = fmt::format("{}: time-critical threads left to Windows (performance cores "
                               "first, the others when those are full), quiet background work "
                               "on the {} others",
                               mode_name, sets.efficiency.size());
            break;
        case HybridMode::Strict:
        case HybridMode::StrictUnguarded:
            what = fmt::format("{}: time-critical threads on the {} performance processors "
                               "({} physical cores, room for {} threads{}), background work on "
                               "the {} others",
                               mode_name, sets.performance.size(), sets.performance_cores,
                               sets.capacity,
                               mode == HybridMode::Strict
                                   ? ", released past that or in a timer storm"
                                   : "",
                               sets.efficiency.size());
            break;
        case HybridMode::GpuPinned:
            what = GpuThreadsHaveOwnCores()
                       ? fmt::format("{}: nothing to add, the GPU threads already have their own "
                                     "performance cores (see the Thread pinning line)",
                                     mode_name)
                       : fmt::format("{}: GPU front, back and render threads on the {} "
                                     "performance processors ({} physical cores, room for {} "
                                     "threads, released past that or in a timer storm), other "
                                     "time-critical threads left to Windows, quiet background "
                                     "work on the {} others",
                                     mode_name, sets.performance.size(), sets.performance_cores,
                                     sets.capacity, sets.efficiency.size());
            break;
        }
    }
    LOG_INFO(Common,
             "CPU placement: {} logical processors, {} performance / {} efficiency / {} low-power "
             "efficiency (efficiency class and last-level cache); hybrid placement {}",
             sets.performance.size() + sets.efficiency.size(), sets.performance.size(), efficiency,
             low_power, what);
#endif
}

void NoteHostTimingWindow(double seconds, u64 late_1ms) {
#ifdef _WIN32
    const WindowsCpuSets& sets = CpuSets();
    if (!sets.cpu_hybrid || seconds <= 0.0) {
        return;
    }
    // Strict placement's timer storm guard: two windows running with over 50 late guest timers a
    // second. The 255H's runs: 0.2-9 a second off (the 9 while loading), 150-235 in the storm.
    static int storm_windows = 0;
    const double late_per_second = static_cast<double>(late_1ms) / seconds;
    storm_windows = late_per_second > 50.0 ? storm_windows + 1 : 0;
    const HybridMode mode = ActiveHybridMode(sets);
    if (mode == HybridMode::Strict && storm_windows >= 2 &&
        !g_strict_released.load(std::memory_order_relaxed)) {
        ReleaseStrictPlacement(fmt::format("timer storm: {:.0f} late guest timers a second, two "
                                           "10 s windows running",
                                           late_per_second));
    }
    const size_t gpu_pinned = g_gpu_pinned_count.load(std::memory_order_relaxed);
    if (mode == HybridMode::GpuPinned && storm_windows >= 2 && gpu_pinned != 0 &&
        !g_gpu_pin_released.load(std::memory_order_relaxed)) {
        ReleaseGpuPin(fmt::format("timer storm: {:.0f} late guest timers a second, two 10 s "
                                  "windows running",
                                  late_per_second));
    }

    // How busy each kind of core was (the whole system, this process included), from
    // NtQuerySystemInformation(SystemProcessorPerformanceInformation): kernel time holds idle.
    struct ProcessorTimes {
        LARGE_INTEGER idle;
        LARGE_INTEGER kernel;
        LARGE_INTEGER user;
        LARGE_INTEGER dpc;
        LARGE_INTEGER interrupt;
        ULONG interrupts;
    };
    using NtQuerySystemFn = LONG(WINAPI*)(ULONG, PVOID, ULONG, PULONG);
    static const auto query = reinterpret_cast<NtQuerySystemFn>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation")));
    constexpr ULONG SystemProcessorPerformanceInformation = 8;
    static std::array<u64, 64> last_busy{};
    static std::array<u64, 64> last_all{};
    std::array<ProcessorTimes, 64> times{};
    ULONG length = 0;
    std::string busy_text;
    if (query != nullptr && query(SystemProcessorPerformanceInformation, times.data(),
                                  static_cast<ULONG>(sizeof(times)), &length) >= 0) {
        const size_t processors =
            (std::min<size_t>)(length / sizeof(ProcessorTimes), times.size());
        // Busy and total 100 ns units per kind: performance, efficiency, low-power.
        std::array<u64, 3> busy{};
        std::array<u64, 3> all{};
        bool first = true;
        for (size_t i = 0; i < processors; ++i) {
            const u64 total = static_cast<u64>(times[i].kernel.QuadPart + times[i].user.QuadPart);
            const u64 work = total - (std::min)(total, static_cast<u64>(times[i].idle.QuadPart));
            first = first && last_all[i] == 0;
            const u64 d_all = total - (std::min)(total, last_all[i]);
            const u64 d_busy = work - (std::min)(work, last_busy[i]);
            last_all[i] = total;
            last_busy[i] = work;
            const u64 bit = u64{1} << i;
            const size_t kind = (sets.performance_mask & bit) != 0  ? 0
                                : (sets.efficiency_mask & bit) != 0 ? 1
                                : (sets.low_power_mask & bit) != 0  ? 2
                                                                    : 3;
            if (kind < 3) {
                busy[kind] += d_busy;
                all[kind] += d_all;
            }
        }
        const auto percent = [&](size_t kind) {
            return all[kind] == 0 ? 0.0
                                  : 100.0 * static_cast<double>(busy[kind]) /
                                        static_cast<double>(all[kind]);
        };
        if (!first) {
            busy_text = fmt::format("; cores busy (whole PC): performance {:.0f}%, efficiency "
                                    "{:.0f}%",
                                    percent(0), percent(1));
            if (sets.low_power_mask != 0) {
                busy_text += fmt::format(", low-power {:.0f}%", percent(2));
            }
        }
    }
    const bool strict = mode == HybridMode::Strict || mode == HybridMode::StrictUnguarded;
    std::string state;
    if (strict) {
        state = g_strict_released.load(std::memory_order_relaxed)
                    ? std::string{", released"}
                    : fmt::format(", holding (room for {})", sets.capacity);
    } else if (mode == HybridMode::GpuPinned) {
        state = GpuThreadsHaveOwnCores() ? std::string{", GPU threads already on their own cores"}
                : g_gpu_pin_released.load(std::memory_order_relaxed)
                    ? std::string{", released"}
                    : fmt::format(", holding {} GPU threads (room for {})", gpu_pinned,
                                  sets.capacity);
    }
    LOG_INFO(Common,
             "Hybrid placement in {:.0f} s: {}{}; {} time-critical threads ({}); late guest "
             "timers {:.1f}/s{}",
             seconds, HybridModeName(mode, sets.mode_from_switch), state,
             g_time_critical_count.load(std::memory_order_relaxed),
             mode == HybridMode::Soft || mode == HybridMode::GpuPinned ? "left to Windows"
             : mode == HybridMode::Off ? "every core"
             : g_strict_released.load(std::memory_order_relaxed)
                 ? "every core"
                 : "on the performance cores",
             late_per_second, busy_text);
#else
    static_cast<void>(seconds);
    static_cast<void>(late_1ms);
#endif
}

u32 PauseCycles() {
    static const u32 cycles = [] {
#ifdef ARCHITECTURE_x86_64
        u64 best = ~u64{0};
        for (int run = 0; run < 3; ++run) {
            const u64 start = Common::X64::FencedRDTSC();
            for (int i = 0; i < 1024; ++i) {
                _mm_pause();
            }
            best = (std::min)(best, Common::X64::FencedRDTSC() - start);
        }
        return static_cast<u32>((std::max<u64>)(best / 1024, 1));
#else
        return 1U;
#endif
    }();
    return cycles;
}

#ifdef _WIN32
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

Event::~Event() {
    if (sleep_wake) {
        CloseHandle(sleep_wake);
    }
}

void Event::WakeSleeper() {
    SetEvent(sleep_wake);
}

// Timer and wake event as in a ported change: a high-resolution timer
// can expire between timer-resolution ticks.
bool Event::SleepFor(const std::chrono::nanoseconds time) {
    struct Timer {
        Timer() {
            handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                            TIMER_ALL_ACCESS);
            if (!handle) {
                handle = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
            }
        }
        ~Timer() {
            if (handle) {
                CloseHandle(handle);
            }
        }
        HANDLE handle;
    };
    static thread_local Timer timer;
    using TimerTicks = std::chrono::duration<s64, std::ratio<1, 10'000'000>>;

    std::unique_lock lk{mutex};
    const auto deadline = std::chrono::steady_clock::time_point{EventWaitDeadline(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()), time)};
    if (!sleep_wake) {
        sleep_wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    }
    while (!is_set.load()) {
        const auto remaining = deadline - std::chrono::steady_clock::now();
        if (remaining <= std::chrono::nanoseconds::zero()) {
            return false;
        }
        LARGE_INTEGER due{};
        due.QuadPart = -std::chrono::ceil<TimerTicks>(remaining).count();
        if (!sleep_wake || !timer.handle ||
            !SetWaitableTimerEx(timer.handle, &due, 0, nullptr, nullptr, nullptr, 0)) {
            condvar.wait_until(lk, deadline, [this] { return is_set.load(); });
            continue;
        }
        // Set() signals sleep_wake while a sleeper waits; the timeout only backs up the timer.
        ++sleepers;
        lk.unlock();
        const HANDLE handles[]{static_cast<HANDLE>(sleep_wake), timer.handle};
        const auto backstop = std::chrono::ceil<std::chrono::milliseconds>(remaining).count() + 1;
        const DWORD result = WaitForMultipleObjects(
            2, handles, FALSE, static_cast<DWORD>((std::min<s64>)(backstop, INFINITE - 1)));
        lk.lock();
        --sleepers;
        if (result == WAIT_FAILED) {
            condvar.wait_until(lk, deadline, [this] { return is_set.load(); });
        }
    }
    is_set = false;
    return true;
}
#else
bool Event::SleepFor(const std::chrono::nanoseconds time) {
    return WaitFor(time);
}
#endif

} // namespace Common
