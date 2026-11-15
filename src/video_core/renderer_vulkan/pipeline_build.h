// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "common/logging.h"
#include "common/thread.h"
#include "common/thread_worker.h"
#include "video_core/renderer_vulkan/async_draw_policy.h"

namespace Vulkan {

class EmissionBudget {
public:
    class Reservation {
    public:
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;

        ~Reservation() {
            if (pending) {
                pending->fetch_sub(1, std::memory_order_relaxed);
            }
        }

    private:
        friend class EmissionBudget;
        Reservation() = default;
        std::shared_ptr<std::atomic<size_t>> pending;
    };

    explicit EmissionBudget(size_t capacity_) : capacity{capacity_} {}

    std::unique_ptr<Reservation> TryAcquire(bool enabled, bool native) {
        size_t count = pending->load(std::memory_order_relaxed);
        if (!DefersShaderEmission(enabled, native, count, capacity)) {
            return nullptr;
        }
        auto reservation = std::unique_ptr<Reservation>{new Reservation};
        while (DefersShaderEmission(enabled, native, count, capacity)) {
            if (pending->compare_exchange_weak(count, count + 1, std::memory_order_relaxed)) {
                reservation->pending = pending;
                return reservation;
            }
        }
        return nullptr;
    }

    size_t Pending() const {
        return pending->load(std::memory_order_relaxed);
    }

private:
    const size_t capacity;
    std::shared_ptr<std::atomic<size_t>> pending = std::make_shared<std::atomic<size_t>>(0);
};

class PipelineBuild {
    struct State {
        explicit State(std::function<void()> task_) : task{std::move(task_)} {}

        void Run() noexcept {
            if (claimed.exchange(true)) {
                return;
            }
            runner_id.store(Common::GetCurrentNativeThreadId(), std::memory_order_release);
            try {
                task();
            } catch (const std::exception& exception) {
                failed = true;
                LOG_ERROR(Render_Vulkan, "Graphics pipeline worker failed: {}", exception.what());
            } catch (...) {
                failed = true;
                LOG_ERROR(Render_Vulkan,
                          "Graphics pipeline worker failed with an unknown exception");
            }
            task = {};
            {
                std::scoped_lock lock{mutex};
                complete.store(true, std::memory_order_release);
                condition.notify_all();
            }
            runner_id.store(0, std::memory_order_release);
        }

        std::function<void()> task;
        bool failed{};
        std::atomic_bool claimed{};
        std::atomic_bool complete{};
        std::atomic<u64> runner_id{};
        std::mutex mutex;
        std::condition_variable condition;
    };

public:
    PipelineBuild(Common::ThreadWorker* workers, std::function<void()> task)
        : state{std::make_shared<State>(std::move(task))} {
        if (workers) {
            workers->QueueWork([state = state] { state->Run(); });
        } else {
            state->Run();
        }
    }

    ~PipelineBuild() {
        Wait(true);
    }

    bool IsBuilt() const {
        return state->complete.load(std::memory_order_acquire);
    }

    bool Failed() const {
        return IsBuilt() && state->failed;
    }

    /// If no worker has started this build yet, start it on a short-lived normal-priority
    /// thread so a time-limited wait does not have to run the compile on the GPU thread.
    void KickUrgent() const {
        const auto shared = state;
        if (shared->complete.load(std::memory_order_acquire) ||
            shared->claimed.load(std::memory_order_acquire)) {
            return;
        }
        static std::atomic<int> live{0};
        if (live.load(std::memory_order_relaxed) >= 4) {
            return;
        }
        live.fetch_add(1, std::memory_order_relaxed);
        try {
            std::thread([shared] {
                Common::SetCurrentThreadName("VkUrgentBuild");
                Common::SetCurrentThreadPriority(Common::ThreadPriority::Normal);
                shared->Run();
                live.fetch_sub(1, std::memory_order_relaxed);
            }).detach();
        } catch (...) {
            live.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    /// Wait until the build is done. `may_build` lets this thread claim an unstarted build
    /// (HeldDrawsClaimBuilds). `timeout_ms` < 0 waits forever; otherwise returns after the cap
    /// whether or not the build finished. `boost_runner` raises an already-started worker to
    /// normal priority for the wait and restores it afterwards.
    /// Returns whether the build completed.
    bool Wait(bool may_build = false, int timeout_ms = -1, bool boost_runner = false) const {
        const auto shared = state;
        if (may_build) {
            shared->Run();
        }
        if (shared->complete.load(std::memory_order_acquire)) {
            return true;
        }
        struct BoostGuard {
            std::shared_ptr<State> state;
            u64 tid{};
            int previous_nice{};
            bool armed{};

            explicit BoostGuard(std::shared_ptr<State> state_) : state{std::move(state_)} {
                tid = state->runner_id.load(std::memory_order_acquire);
                if (tid == 0) {
                    return;
                }
                previous_nice = Common::GetNativeThreadNice(tid);
                Common::SetNativeThreadPriority(tid, Common::ThreadPriority::Normal);
                armed = true;
            }

            ~BoostGuard() {
                if (!armed) {
                    return;
                }
                if (state->runner_id.load(std::memory_order_acquire) != tid) {
                    return;
                }
                Common::SetNativeThreadNice(tid, previous_nice);
            }
        };
        std::optional<BoostGuard> boost;
        if (boost_runner) {
            boost.emplace(shared);
        }
        std::unique_lock lock{shared->mutex};
        const auto done = [&] { return shared->complete.load(std::memory_order_acquire); };
        if (timeout_ms < 0) {
            shared->condition.wait(lock, done);
            return true;
        }
        shared->condition.wait_for(lock, std::chrono::milliseconds{timeout_ms}, done);
        return shared->complete.load(std::memory_order_acquire);
    }

    PipelineBuild(const PipelineBuild&) = delete;
    PipelineBuild& operator=(const PipelineBuild&) = delete;

private:
    std::shared_ptr<State> state;
};

} // namespace Vulkan
