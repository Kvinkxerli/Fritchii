// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MasterSemaphore;

/// FRITCHII_SUBMIT_TIMES=1 (diagnostic, read at first use, default 0; helper_cu_answer.md Round
/// 3): a GPU timestamp at the start of every main-queue submission (top of its upload command
/// buffer) and at its end (bottom of its command buffer), read back lazily, when someone asks
/// about a run of ticks. The read-back counters ask, at each read's cut at its writer's tick
/// (FRITCHII_READ_RUNS=2), how the submissions the card had to run before the writer split into
/// GPU busy time and the gaps between them (submission and scheduling overhead).
class SubmitTimes {
public:
    /// What the submissions [first, last] took on the GPU.
    struct Span {
        u32 submissions{}; ///< asked for (capped at MaxSpan)
        u32 resolved{};    ///< with both timestamps read
        u64 busy_ns{};     ///< sum of end - start
        u64 gap_ns{};      ///< sum of the idle time between one's end and the next one's start
        u64 overlap_ns{};  ///< sum of the time a submission started before the previous ended
        u64 span_ns{};     ///< the first resolved one's start to the last one's end
    };

    static constexpr u32 Slots = 4096;
    static constexpr u32 MaxSpan = 64;

    [[nodiscard]] static bool Enabled();

    /// The worker, after the upload command buffer began: the start stamp of the next submission.
    void Begin(const Device& device, vk::CommandBuffer upload_cmdbuf);
    /// The worker, before the command buffer ends: its end stamp, for scheduler tick `tick`.
    void End(vk::CommandBuffer cmdbuf, u64 tick);

    /// Any thread: the submissions [first, last], those the host GPU completed (`master`).
    [[nodiscard]] Span Measure(u64 first, u64 last, MasterSemaphore& master) const;

private:
    bool ReadSlot(u32 slot, u64& start_ns, u64& end_ns) const;

    const Device* device{};
    vk::QueryPool pool;
    double period_ns{};
    u64 mask{};
    bool failed{};
    bool recording{};
    u32 slot{};      ///< the slot the submission being recorded uses
    u64 sequence{};  ///< submissions stamped so far
    /// Per slot: the tick it holds (0: none yet). Per tick % Slots: the slot.
    std::unique_ptr<std::atomic<u64>[]> slot_tick{new std::atomic<u64>[Slots]{}};
    std::unique_ptr<std::atomic<u32>[]> tick_slot{new std::atomic<u32>[Slots]{}};
};

/// FRITCHII_SLOW_GPU (live, default 1; docs/fritchii/switches.md, helper_eb_answer.md): a GPU
/// timestamp at the start (top of the upload command buffer) and end (bottom of the command
/// buffer) of every main-queue submission, read back on the Vulkan worker once the tick is done,
/// never waited for. A submission over FRITCHII_SLOW_GPU_MS (live, default 50) logs one "Slow GPU
/// submission:" line with what it held (vk::CmdStats, counted by the command buffer wrappers on
/// the worker), at most 20 a minute, and a "Slow GPU 10 s:" summary follows when any did. The
/// newest submissions' contents are kept as breadcrumbs for the hang report and a device loss
/// (LogBreadcrumbs, through Common::Nx2Hitch::LogGpuBreadcrumbs).
class SlowGpu {
public:
    /// Where a submission came from: the recording thread's side of Scheduler::SubmitExecution.
    struct Origin {
        std::array<char, 16> thread{}; ///< the recording thread's name (GPUBack = E, GPURender = R)
        const char* file{};            ///< Flush or Finish call site
        u32 line{};
        bool finish{};  ///< Scheduler::Finish: the recording thread waits for it at once
        bool signals{}; ///< signals a binary semaphore (a frame for the present thread)
        bool waits{};   ///< waits on a binary semaphore
    };

    static constexpr u32 Slots = 256;
    static constexpr u32 Crumbs = 8;   ///< submissions a report lists, newest
    static constexpr u32 History = 256; ///< submissions kept, to find the oldest unfinished
    static constexpr u32 OldestChecked = 8; ///< from the first not reported done

    SlowGpu() = default;
    /// Waits for the GPU to be done with the newest stamped submission (the pool goes with this).
    /// The scheduler's worker has joined by then (worker_thread is declared after slow_gpu).
    ~SlowGpu();
    SlowGpu(const SlowGpu&) = delete;
    SlowGpu& operator=(const SlowGpu&) = delete;

    /// The worker, with each new pair of command buffers (and the constructor, once): reads the
    /// finished submissions back (slow lines, the summary), stamps the next one's start and
    /// starts counting its commands. Off: reads back what is pending, nothing else.
    void Begin(const Device& device, vk::CommandBuffer upload_cmdbuf, MasterSemaphore& master);
    /// The worker, before each chunk: where the wrappers count (null: they don't).
    [[nodiscard]] vk::CmdStats* Counting() noexcept {
        return counting ? &current : nullptr;
    }
    /// The worker, in the submission before cmdbuf.End(): its end stamp; keeps its contents.
    void End(vk::CommandBuffer cmdbuf, u64 tick, const Origin& origin);
    /// The worker, after vkQueueSubmit returned for `tick`: when it was called (NowNs).
    void Submitted(u64 tick, u64 call_ns);
    /// Any thread (hang report, device loss): the oldest submission the GPU has not finished and
    /// the newest submissions, what they held and whether the GPU finished or started them. Never
    /// waits for the GPU, never reports a loss itself; skips a lock held 100 ms.
    void LogBreadcrumbs(const char* prefix, MasterSemaphore& master) const;
    /// The newest tick with stamps the GPU may still write (0: none).
    [[nodiscard]] u64 LastStampedTick() const noexcept {
        return last_stamped_tick;
    }

    [[nodiscard]] static u64 NowNs() noexcept;

private:
    struct Entry {
        u64 tick{};
        u64 submit_ns{};     ///< steady clock, just before vkQueueSubmit (0: not submitted)
        u32 submit_call_us{};
        Origin origin{};
        vk::CmdStats stats{};
    };
    struct Crumb {
        Entry entry{};
        u32 slot{~0u};       ///< its timestamp slot (~0: not stamped)
    };

    bool CreatePool(const Device& device);
    void Poll(MasterSemaphore& master, u64 now);
    void Process(const Entry& entry, u64 start_raw, u64 end_raw, u64 sequence, u64 now);
    void Summary(u64 now, bool verbose);
    u64 Unwrap(u64 raw);
    /// The start stamp of `slot` if it still belongs to `tick`: 1 written, 0 not yet, -1 unknown.
    int Started(u32 slot, u64 tick) const {
        return StampWritten(slot, tick, 0);
    }
    /// The same for the end stamp.
    int Ended(u32 slot, u64 tick) const {
        return StampWritten(slot, tick, 1);
    }
    int StampWritten(u32 slot, u64 tick, u32 which) const;

    const Device* device{};
    MasterSemaphore* master_semaphore{}; ///< from Begin, for the destructor's wait
    vk::QueryPool pool;
    double period_ns{};
    u64 mask{};
    bool failed{};
    bool host_reset{};
    bool counting{};
    bool stamping{};      ///< the submission being recorded has a start stamp in stamp_next's slot
    vk::CmdStats current{};

    std::unique_ptr<Entry[]> ring{new Entry[Slots]{}}; ///< stamped submissions, by stamp sequence
    std::unique_ptr<std::atomic<u64>[]> slot_tick{new std::atomic<u64>[Slots]{}};
    u64 stamp_next{};     ///< stamped submissions so far
    u64 stamp_read{};     ///< the oldest not read back
    u64 last_stamped_tick{};
    u64 last_end_tick{};  ///< the newest tick End kept (Submitted's test)
    u64 unstamped{};      ///< submissions not stamped while on (ring full), this window

    // Read-back state (worker only).
    u64 last_poll_ns{};
    u64 last_raw{};
    u64 last_unwrapped{};
    bool have_raw{};
    double previous_end_ns{};  ///< the previous read submission's end (GPU clock, ns)
    u64 previous_sequence{~u64{0}};
    double offset_min{1e300};  ///< min over this window of GPU start - CPU submit (ns)
    double offset_min_previous{1e300};

    // Lines and the 10 s summary (worker only).
    u64 window_start_ns{};
    u64 window_timed{};
    u64 window_over{};
    u64 window_held{};
    double window_busy_ms{};
    double window_max_ms{};
    u64 window_max_tick{};
    u64 minute_start_ns{};
    u32 minute_lines{};

    mutable std::mutex crumb_mutex;
    std::unique_ptr<Crumb[]> crumbs{new Crumb[History]{}};
    u64 crumb_next{};     ///< crumbs written (under crumb_mutex)
};

} // namespace Vulkan
