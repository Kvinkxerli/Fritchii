// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <list>
#include <memory>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "common/assert.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/gpu_dirty_memory_manager.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "core/hle/service/nvdrv/nvdata.h"
#include "core/perf_stats.h"
#include "video_core/buffer_cache/read_run_stats.h"
#include "video_core/buffer_cache/store_footprint_live.h"
#include "video_core/cdma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu.h"
#include "video_core/pending_maps.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_timestamp.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/memory_stage_link.h"
#include "video_core/nx2.h"
#include "video_core/nx2_async_readback.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_runahead.h"
#include "video_core/one_follower.h"
#include "video_core/renderer_base.h"
#include "video_core/shader_notify.h"

namespace Tegra {

struct GPU::Impl {
    explicit Impl(Core::System& system_, bool is_async_, bool use_nvdec_)
        : system{system_}
        , gpu_thread{system_}
        , use_nvdec{use_nvdec_}
        , shader_notify()
        , is_async{is_async_}
    {}

    ~Impl() {
        NotifyShutdown();
        // Rasterizer teardown flushes commands and invalidates the bound engine's dirty flags.
        // Keep channels, their scheduler references, and the graphics context alive until then.
        renderer.reset();
    }

    std::shared_ptr<Control::ChannelState> CreateChannel(s32 channel_id) {
        auto channel_state = std::make_shared<Tegra::Control::ChannelState>(channel_id);
        channels.emplace(channel_id, channel_state);
        scheduler.DeclareChannel(channel_state);
        return channel_state;
    }

    void BindChannel(s32 channel_id) {
        if (bound_channel != channel_id) {
            auto it = channels.find(channel_id);
            ASSERT(it != channels.end());
            bound_channel = channel_id;
            current_channel = it->second.get();
            renderer->ReadRasterizer()->BindChannel(*current_channel);
        }
    }

    std::shared_ptr<Control::ChannelState> AllocateChannel() {
        return CreateChannel(new_channel_id++);
    }

    void InitChannel(Control::ChannelState& to_init, u64 program_id) {
        to_init.Init(system, program_id);
        to_init.BindRasterizer(renderer->ReadRasterizer());
        renderer->ReadRasterizer()->InitializeChannel(to_init);
    }

    void InitAddressSpace(Tegra::MemoryManager& memory_manager) {
        memory_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    void ReleaseChannel(Control::ChannelState& to_release) {
        UNIMPLEMENTED();
    }

    /// Binds a renderer to the GPU.
    void BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer_) {
        renderer = std::move(renderer_);
        system.Host1x().memory_manager.BindInterface(renderer->ReadRasterizer());
        system.Host1x().gmmu_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    /// Flush all current written commands into the host GPU for execution.
    void FlushCommands() {
        renderer->ReadRasterizer()->FlushCommands();
    }

    /// FRITCHII_GATHER_EMPTY_TEST's counts, one line every 10 s (any gathering thread; the clock
    /// is read every 1,024th test).
    static void NoteGatherEmptyTest(int mode, bool empty) {
        static std::atomic<u64> asked{};
        static std::atomic<u64> skipped{};
        static std::atomic<s64> since{};
        if (empty) {
            skipped.fetch_add(1, std::memory_order_relaxed);
        }
        if ((asked.fetch_add(1, std::memory_order_relaxed) & 1023) != 0) {
            return;
        }
        const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
        s64 last = since.load(std::memory_order_relaxed);
        if (last == 0) {
            since.compare_exchange_strong(last, now, std::memory_order_relaxed);
            asked.store(0, std::memory_order_relaxed);
            skipped.store(0, std::memory_order_relaxed);
            return;
        }
        if (now - last < 10'000 ||
            !since.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
            return;
        }
        const double window = static_cast<double>(now - last) / 1000.0;
        const u64 asked_count = asked.exchange(0, std::memory_order_relaxed);
        const u64 skipped_count = skipped.exchange(0, std::memory_order_relaxed);
        LOG_INFO(HW_GPU,
                 "Gather empty test (FRITCHII_GATHER_EMPTY_TEST={}) {:.0f} s: gathers asked "
                 "{:.0f}/s, skipped as empty {:.0f}/s ({:.1f}%)",
                 mode, window, static_cast<double>(asked_count) / window,
                 static_cast<double>(skipped_count) / window,
                 asked_count != 0 ? 100.0 * static_cast<double>(skipped_count) /
                                        static_cast<double>(asked_count)
                                  : 0.0);
    }

    /// Synchronizes CPU writes with Host GPU memory.
    void InvalidateGPUCache() {
        // EDEN_NX2_RUNAHEAD: behind the frontier the guest's newest bytes are not the draws'
        // (the front's next boundary takes them); past it they are applied at once (exact).
        if (VideoCommon::Runahead::HandleInvalidate(system)) {
            return;
        }
        // FRITCHII_GATHER_EMPTY_TEST (live, default 0; fable_front_drains 8.2 G3): most gathers
        // find nothing. Each manager answers that without a lock (NothingCollected: a write
        // collected before this test is seen, or a gather already took it); a write collected
        // after it is taken by the next gather, as when it lands after the managers' locks.
        if (const int empty_test = VideoCore::Nx2Debug::SwitchValue(
                VideoCore::Nx2Debug::g_gather_empty_test, "FRITCHII_GATHER_EMPTY_TEST");
            empty_test != 0) {
            const bool empty = std::ranges::all_of(
                system.GetGPUDirtyMemoryManager(),
                [](const Core::GPUDirtyMemoryManager& manager) {
                    return manager.NothingCollected();
                });
            NoteGatherEmptyTest(empty_test, empty);
            if (empty) {
                return;
            }
        }
        // FRITCHII_MEMSTAGE_PASS bit 1 (fable_stage_levelpoints 6.3, 13.3 item 5): the back in a
        // served entry takes nothing itself, read tracking or not. Empty collectors: nothing to
        // do (a write collected after this test is taken at the next gather, as one collected
        // after the managers' locks). Else the take and the buffer half run on GPUMem in one
        // served call, so only GPUMem takes while the back is served (no write sits between two
        // takers while a fill passes); that take is listed as below (rule T).
        // Bit 2 too (P3): a passed fence entry's fence ends in this gather while served.
        if (VideoCommon::MemoryStageLink::BackServed() && VideoCommon::MemoryStageLink::PassOn(3)) {
            if (std::ranges::all_of(system.GetGPUDirtyMemoryManager(),
                                    [](const Core::GPUDirtyMemoryManager& manager) {
                                        return manager.NothingCollected();
                                    })) {
                return;
            }
            if (renderer->ReadRasterizer()->GatherOnStage()) {
                return;
            }
        }
        std::vector<std::pair<DAddr, u64>> writes;
        std::function<void(PAddr, size_t)> callback_writes(
            [&writes](PAddr address, size_t size) { writes.emplace_back(address, size); });
        // EDEN_NX2_EXP49: gathered writes stay pending (ForEachPendingCpuWrite) until they are
        // applied. A read download written to guest memory meanwhile on a reading thread
        // (FinishRead) would otherwise write the GPU's older bytes over them. Read downloads are
        // recorded on the thread that applies these (FlushRegionForRead calls this first), so
        // with none pending and the switch off nothing is tracked and no lock is taken.
        const bool track = VideoCore::g_read_downloads.load(std::memory_order_acquire) != 0 ||
                           VideoCore::Nx2Debug::Experiment(49) != 0;
        if (!track) {
            system.GatherGPUDirtyMemory(callback_writes);
            if (!writes.empty()) {
                renderer->ReadRasterizer()->OnCacheInvalidations(writes);
            }
            return;
        }
        {
            std::scoped_lock lock{applying_writes_mutex};
            system.GatherGPUDirtyMemory(callback_writes);
            if (writes.empty()) {
                return;
            }
            applying_writes.push_back(&writes);
        }
        // Also if OnCacheInvalidations throws: ForEachPendingCpuWrite must not reach `writes`
        // once it is gone.
        SCOPE_EXIT {
            std::scoped_lock lock{applying_writes_mutex};
            std::erase(applying_writes, &writes);
        };
        renderer->ReadRasterizer()->OnCacheInvalidations(writes);
    }

    /// FRITCHII_MEMSTAGE_PASS (GPUMem): the take alone (GPU::TakeCpuWrites), listed on
    /// applying_writes whatever EXP49 says (fable_stage_levelpoints 13.3, rule T): until the
    /// buffer cache has the writes, a landing of downloaded GPU bytes leaves them out, as for a
    /// tracked gather. `out` must not move or change while listed. True when listed.
    bool TakeCpuWrites(std::vector<std::pair<DAddr, u64>>& out) {
        if (std::ranges::all_of(system.GetGPUDirtyMemoryManager(),
                                [](const Core::GPUDirtyMemoryManager& manager) {
                                    return manager.NothingCollected();
                                })) {
            return false;
        }
        std::function<void(PAddr, size_t)> callback(
            [&out](PAddr address, size_t size) { out.emplace_back(address, size); });
        // Lock order: applying_writes_mutex, then the managers' (as ForEachPendingCpuWrite).
        std::scoped_lock lock{applying_writes_mutex};
        system.GatherGPUDirtyMemory(callback);
        if (out.empty()) {
            return false;
        }
        applying_writes.push_back(&out);
        return true;
    }

    /// Rule T: the list TakeCpuWrites listed leaves applying_writes (the buffer cache has it).
    void ReleaseTakenWrites(const std::vector<std::pair<DAddr, u64>>* list) {
        std::scoped_lock lock{applying_writes_mutex};
        std::erase(applying_writes, list);
    }

    /// Rule T's check: how many lists are on applying_writes (0 at the ring's end, no gather
    /// running).
    size_t ListedWrites() {
        std::scoped_lock lock{applying_writes_mutex};
        return applying_writes.size();
    }

    void ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
        std::scoped_lock lock{applying_writes_mutex};
        for (auto& manager : system.GetGPUDirtyMemoryManager()) {
            manager.ForEachPending(callback);
        }
        for (const auto* writes : applying_writes) {
            for (const auto& [address, size] : *writes) {
                callback(address, size);
            }
        }
        // EDEN_NX2_RUNAHEAD: writes taken into chunks the back has not applied yet.
        if (VideoCommon::Runahead::g_exact_used.load(std::memory_order_acquire)) {
            VideoCommon::Runahead::ForEachUnappliedRange(
                [](void* argument, u64 address, u64 size) {
                    (*static_cast<const std::function<void(DAddr, size_t)>*>(argument))(
                        address, static_cast<size_t>(size));
                },
                const_cast<std::function<void(DAddr, size_t)>*>(&callback));
        }
    }

    /// Signal the ending of command list.
    void OnCommandListEnd() {
        renderer->ReadRasterizer()->ReleaseFences(false);
        Settings::UpdateGPUAccuracy();
    }

    /// Request a host GPU memory flush from the CPU.
    template <typename Func>
    [[nodiscard]] u64 RequestSyncOperation(Func&& action) {
        std::unique_lock lck{sync_request_mutex};
        const u64 fence = ++last_sync_fence;
        if (sync_requests_dropped.load(std::memory_order_acquire)) {
            // The GPU threads stopped (NotifyShutdown): nothing runs it, and its waiter returns at
            // once. Not queued, so a stray TickWork can't run it after its waiter's frame is gone.
            return fence;
        }
        sync_requests.emplace_back(std::forward<Func>(action));
        has_sync_requests.store(true, std::memory_order_release);
        return fence;
    }

    /// Obtains current flush request fence id.
    [[nodiscard]] u64 CurrentSyncRequestFence() const {
        return current_sync_fence.load(std::memory_order_acquire);
    }

    /// Returns once the request of `fence` ran, or once the GPU threads stopped for good
    /// (NotifyShutdown): then it never runs.
    void WaitForSyncOperation(const u64 fence) {
        if (CurrentSyncRequestFence() >= fence ||
            sync_requests_dropped.load(std::memory_order_acquire)) {
            return;
        }
        for (int i = 0; i < 200; ++i) {
#if defined(_M_X64) || defined(__x86_64__)
            _mm_pause();
#else
            std::this_thread::yield();
#endif
            if (CurrentSyncRequestFence() >= fence ||
                sync_requests_dropped.load(std::memory_order_acquire)) {
                return;
            }
        }
        std::unique_lock lck{sync_request_mutex};
        sync_request_cv.wait(lck, [this, fence] {
            return CurrentSyncRequestFence() >= fence ||
                   sync_requests_dropped.load(std::memory_order_acquire);
        });
    }

    void WaitForIdle() {
        const u64 fence = RequestSyncOperation([] {});
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
    }

    /// Tick pending requests within the GPU.
    void TickWork() {
        // EDEN_NX2_EXP29 & 2 (the original's EDEN_NX2_OPT_TICKWORK): TickWork runs for
        // every draw, clear and dispatch; check without the mutex whether a request is queued
        // at all. A request queued after this check runs at the next TickWork, as one queued
        // just after the loop below empties the queue does; whoever queued it and then had this
        // thread tick (WaitForIdle's TickGPU) is ordered after its store, so that tick sees it.
        if ((VideoCore::Nx2Debug::Experiment(29) & 2) != 0 &&
            !has_sync_requests.load(std::memory_order_acquire)) {
            return;
        }
        std::unique_lock lck{sync_request_mutex};
        while (!sync_requests.empty()) {
            auto request = std::move(sync_requests.front());
            sync_requests.pop_front();
            sync_request_mutex.unlock();
            VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::TickWork);
            request();
            current_sync_fence.fetch_add(1, std::memory_order_release);
            sync_request_mutex.lock();
            sync_request_cv.notify_all();
        }
        // Under the lock with the queue empty: no request can be queued in between.
        has_sync_requests.store(false, std::memory_order_relaxed);
    }

    [[nodiscard]] u64 GetTicks() const {
        return ReadGpuTimestamp(system.CoreTiming(), g_nx2_ampere.load(std::memory_order_relaxed),
                                Settings::values.gpu_clock.GetValue(),
                                present_ns.load(std::memory_order_relaxed));
    }

    void RendererFrameEndNotify() {
        present_ns.store(static_cast<u64>(system.CoreTiming().GetGlobalTimeNs().count()),
                         std::memory_order_relaxed);
        system.GetPerfStats().EndGameFrame();
    }

    /// Performs any additional setup necessary in order to begin GPU emulation.
    /// This can be used to launch any necessary threads and register any necessary
    /// core timing events.
    void Start() {
        Settings::UpdateGPUAccuracy();
        gpu_thread.StartThread(*renderer, renderer->Context(), scheduler);
    }

    void NotifyShutdown() {
        // Stops the GPU thread and the split's back (GpuSplit::Stop): no TickWork runs after it.
        gpu_thread.NotifyShutdown();
        {
            // The GPU threads stopped: no sync operation runs anymore. Guest threads still run
            // until the kernel suspends them, and one waiting for a read's flush (OnCPURead)
            // would wait forever, in host code where the kernel cannot suspend it
            // (KernelCore::SuspendEmulation then waits for it forever). Drop the requests, which
            // may refer to their waiters' frames, and release the waiters.
            std::scoped_lock requests_lock{sync_request_mutex};
            sync_requests.clear();
            has_sync_requests.store(false, std::memory_order_relaxed);
            sync_requests_dropped.store(true, std::memory_order_release);
        }
        sync_request_cv.notify_all();
        std::unique_lock lk{sync_mutex};
        shutting_down.store(true, std::memory_order::relaxed);
        sync_cv.notify_all();
    }

    /// Obtain the CPU Context
    void ObtainContext() {
        if (!cpu_context) {
            cpu_context = renderer->GetRenderWindow().CreateSharedContext();
        }
        cpu_context->MakeCurrent();
    }

    /// Release the CPU Context
    void ReleaseContext() {
        cpu_context->DoneCurrent();
    }

    /// Push GPU command entries to be processed
    void PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
        gpu_thread.SubmitList(channel, std::move(entries), is_async);
    }

    void RunInOrder(std::function<void()>&& func) {
        gpu_thread.Run(std::move(func), is_async);
    }

    void MapInOrder(GPUVAddr gpu_addr, u64 size, std::function<void()>&& func) {
        const u64 id = pending_maps.Add(gpu_addr, size);
        gpu_thread.Run(
            [this, id, func = std::move(func)] {
                func();
                pending_maps.Remove(id);
            },
            is_async, true);
    }

    bool IsMapPending(GPUVAddr gpu_addr, u64 size) const {
        return pending_maps.Overlaps(gpu_addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(DAddr addr, u64 size) {
        gpu_thread.FlushRegion(addr, size, is_async);
    }

    VideoCore::RasterizerDownloadArea OnCPURead(DAddr addr, u64 size) {
        namespace ReadRunStats = VideoCommon::ReadRunStats;
        if (ReadRunStats::g_enabled) [[unlikely]] {
            ReadRunStats::t_run_pages = 0;
            ReadRunStats::t_read_bytes = {};
            ReadRunStats::t_writer_answer = {};
        }
        auto raster_area = renderer->ReadRasterizer()->GetFlushArea(addr, size);
        // FRITCHII_STORE_FOOTPRINT=2: a sampled read compares its bytes once its wait is over
        // (nothing pending: one acquire load)
        SCOPE_EXIT {
            VideoCommon::StoreFootprintLive::FinishVerify(
                addr, size, [this](u64 address, u8* out, u64 bytes) {
                    system.Host1x().MemoryManager().ReadBlockUnsafe(address, out, bytes);
                });
        };
        if (raster_area.preemtive) {
            return raster_area;
        }
        // FRITCHII_READ_RUNS: this read goes to the back or waits for a download.
        const u64 run_pages = ReadRunStats::t_run_pages;
        const ReadRunStats::ReadBytes read_bytes =
            ReadRunStats::g_enabled ? ReadRunStats::t_read_bytes : ReadRunStats::ReadBytes{};
        const u64 run_start = ReadRunStats::g_enabled ? VideoCore::AsyncReadback::Now() : 0;
        // EDEN_NX2_EXP72: everything this read lacks is in a fence download already submitted:
        // wait for it to land instead of a new copy behind all the work recorded since.
        if (raster_area.fence_download != 0 &&
            renderer->ReadRasterizer()->WaitFenceDownload(raster_area.fence_download)) {
            raster_area.preemtive = true;
            if (ReadRunStats::g_enabled) [[unlikely]] {
                const u64 wait_ns = VideoCore::AsyncReadback::Now() - run_start;
                ReadRunStats::NoteRead(ReadRunStats::Kind::FenceDownload,
                                       raster_area.start_address, run_pages, wait_ns, size,
                                       read_bytes);
                NoteReadWriter(ReadRunStats::Kind::FenceDownload, wait_ns);
            }
            return raster_area;
        }
        raster_area.preemtive = true;
        // A guest CPU read of GPU-written memory (reactive flushing): count them, they decide
        // whether buffer data must be copied back at every fence (EDEN_NX2_BUFFER_DOWNLOADS).
        if (static std::atomic<u64> reads{}; (reads.fetch_add(1, std::memory_order_relaxed) & 63) == 0) {
            LOG_INFO(HW_GPU, "CPU read of GPU-written memory [{:#x}, {:#x}) ({} so far)",
                     raster_area.start_address, raster_area.end_address,
                     reads.load(std::memory_order_relaxed));
        }
        namespace AsyncReadback = VideoCore::AsyncReadback;
        const u64 request = AsyncReadback::Now();
        const u64 area_size = raster_area.end_address - raster_area.start_address;
        // EDEN_NX2_EXP49 (live): the GPU thread submits the read's
        // downloads without waiting for the host GPU, and this thread waits for them. An area an
        // earlier read's downloads cover (in_flight) only waits; with the switch off it flushes
        // as before, which lands or downloads those bytes again.
        const bool async = VideoCore::Nx2Debug::Experiment(49) != 0;
        if (!async || !raster_area.in_flight) {
            const u64 fence = RequestSyncOperation([this, raster_area, area_size, async]() {
                // Its own zone: it runs in whatever zone TickWork runs in (a draw's, mostly).
                const VideoCommon::Nx2Profile::Scope read_scope{
                    VideoCommon::Nx2Profile::Zone::ReadFlush};
                const u64 start = AsyncReadback::Now();
                // Apply collected guest writes before downloading GPU data over guest memory.
                InvalidateGPUCache();
                if (async) {
                    renderer->ReadRasterizer()->FlushRegionForRead(raster_area.start_address,
                                                                   area_size);
                } else {
                    renderer->ReadRasterizer()->FlushRegion(raster_area.start_address, area_size);
                }
                AsyncReadback::Add(AsyncReadback::g_counters.flush_ns,
                                   AsyncReadback::Now() - start);
            });
            gpu_thread.TickGPU(is_async);
            WaitForSyncOperation(fence);
            AsyncReadback::Add(AsyncReadback::g_counters.to_flushed_ns,
                               AsyncReadback::Now() - request);
        }
        AsyncReadback::Read kind = AsyncReadback::Read::Sync;
        if (async) {
            // Written to guest memory before this thread reads, while the GPU thread goes on.
            AsyncReadback::t_reading = true;
            renderer->ReadRasterizer()->FinishRead(raster_area.start_address, area_size);
            AsyncReadback::t_reading = false;
            kind = raster_area.in_flight ? AsyncReadback::Read::InFlight
                                         : AsyncReadback::Read::Async;
        }
        AsyncReadback::CountRead(kind, AsyncReadback::Now() - request);
        AsyncReadback::MaybeReport();
        if (ReadRunStats::g_enabled) [[unlikely]] {
            const ReadRunStats::Kind read_kind = kind == AsyncReadback::Read::InFlight
                                                     ? ReadRunStats::Kind::InFlight
                                                     : ReadRunStats::Kind::Flushed;
            const u64 wait_ns = AsyncReadback::Now() - run_start;
            ReadRunStats::NoteRead(read_kind, raster_area.start_address, run_pages, wait_ns, size,
                                   read_bytes);
            NoteReadWriter(read_kind, wait_ns);
        }
        return raster_area;
    }

    /// FRITCHII_READ_RUNS: the read's writer, as GetFlushArea found it, and whether the read's
    /// bytes in guest memory differ now that its download landed from before it landed.
    void NoteReadWriter(VideoCommon::ReadRunStats::Kind kind, u64 wait_ns) {
        namespace ReadRunStats = VideoCommon::ReadRunStats;
        const ReadRunStats::WriterAnswer& answer = ReadRunStats::t_writer_answer;
        std::array<u8, ReadRunStats::CompareBytes> after{};
        if (answer.compared != 0) {
            system.Host1x().MemoryManager().ReadBlockUnsafe(answer.address, after.data(),
                                                            answer.compared);
        }
        ReadRunStats::NoteWriter(kind, wait_ns, answer,
                                 answer.compared != 0 ? after.data() : nullptr);
    }

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
        if (preserve_gpu_writes) {
            VideoCore::RasterizerInterface* rasterizer = renderer->ReadRasterizer();
            if (!rasterizer->TryCacheInvalidation(addr, size)) {
                const u64 fence = RequestSyncOperation([rasterizer, addr, size] {
                    const VideoCommon::Nx2Profile::Scope read_scope{
                        VideoCommon::Nx2Profile::Zone::ReadFlush};
                    rasterizer->FlushRegion(addr, size, VideoCommon::CacheType::BufferCache);
                    rasterizer->OnCacheInvalidation(addr, size);
                });
                gpu_thread.TickGPU(is_async);
                WaitForSyncOperation(fence);
            }
            return;
        }
        gpu_thread.InvalidateRegion(addr, size);
    }

    bool OnCPUWrite(DAddr addr, u64 size) {
        return renderer->ReadRasterizer()->OnCPUWrite(addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed and invalidated
    void FlushAndInvalidateRegion(DAddr addr, u64 size) {
        gpu_thread.FlushAndInvalidateRegion(addr, size, is_async);
    }

    void RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers, std::vector<Service::Nvidia::NvFence>&& fences) {
        const size_t num_fences{fences.size()};
        size_t current_request_counter{};
        if (num_fences != 0) {
            std::unique_lock<std::mutex> lk(request_swap_mutex);
            if (free_swap_counters.empty()) {
                current_request_counter = request_swap_counters.size();
                request_swap_counters.emplace_back(num_fences);
            } else {
                current_request_counter = free_swap_counters.front();
                request_swap_counters[current_request_counter] = num_fences;
                free_swap_counters.pop_front();
            }
        }
        pending_composite_fence = RequestSyncOperation(
            [this, current_request_counter, num_fences, composite_layers = std::move(layers),
             composite_fences = std::move(fences)] {
                if (num_fences == 0) {
                    renderer->Composite(composite_layers);
                    return;
                }
                auto& syncpoint_manager = system.Host1x().GetSyncpointManager();
                const auto executer = [this, current_request_counter, composite_layers]() {
                    {
                        std::unique_lock<std::mutex> lk(request_swap_mutex);
                        if (--request_swap_counters[current_request_counter] != 0) {
                            return;
                        }
                        free_swap_counters.push_back(current_request_counter);
                    }
                    renderer->Composite(composite_layers);
                };
                for (size_t i = 0; i < num_fences; i++) {
                    syncpoint_manager.RegisterGuestAction(composite_fences[i].id,
                                                          composite_fences[i].value, executer);
                }
            });
        gpu_thread.TickGPU(is_async);
    }

    void WaitForComposite() {
        const u64 fence = pending_composite_fence;
        if (fence == 0) {
            return;
        }
        pending_composite_fence = 0;
        if (shutting_down.load(std::memory_order_relaxed)) {
            return;
        }
        WaitForSyncOperation(fence);
    }

    std::vector<u8> GetAppletCaptureBuffer() {
        std::vector<u8> out;

        const auto wait_fence =
            RequestSyncOperation([&] { out = renderer->GetAppletCaptureBuffer(); });
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(wait_fence);

        return out;
    }

    Core::System& system;

    // Destruction of thread must be done before all (non trivial)
    // previous members has been destroyed
    VideoCommon::GPUThread::ThreadManager gpu_thread;

    std::unique_ptr<VideoCore::RendererBase> renderer;
    const bool use_nvdec;

    s32 new_channel_id{1};
    /// Shader build notifier
    VideoCore::ShaderNotify shader_notify;
    /// When true, we are about to shut down emulation session, so terminate outstanding tasks
    std::atomic_bool shutting_down{};

    std::array<std::atomic<u32>, Service::Nvidia::MaxSyncPoints> syncpoints{};

    std::array<std::list<u32>, Service::Nvidia::MaxSyncPoints> syncpt_interrupts;

    std::mutex sync_mutex;
    std::mutex device_mutex;

    std::condition_variable sync_cv;

    std::list<std::function<void()>> sync_requests;
    /// Set with each request queued, cleared under the lock once TickWork emptied the queue
    /// (EXP29 & 2 reads it without the lock).
    std::atomic<bool> has_sync_requests{};
    std::atomic<u64> current_sync_fence{};
    std::atomic<u64> present_ns{};
    u64 last_sync_fence{};

    VideoCommon::PendingMaps pending_maps;
    std::mutex sync_request_mutex;
    std::condition_variable sync_request_cv;

    /// InvalidateGPUCache: the writes it gathered and is applying, on each thread doing so
    /// (EDEN_NX2_EXP49).
    std::vector<const std::vector<std::pair<DAddr, u64>>*> applying_writes;
    std::mutex applying_writes_mutex;
    /// The GPU threads stopped and the requests were dropped (NotifyShutdown, under the mutex).
    std::atomic<bool> sync_requests_dropped{};

    const bool is_async;

    std::unique_ptr<Core::Frontend::GraphicsContext> cpu_context;

    Tegra::Control::Scheduler scheduler;
    ::Common::unordered_map<s32, std::shared_ptr<Tegra::Control::ChannelState>> channels;
    Tegra::Control::ChannelState* current_channel;
    s32 bound_channel{-1};

    std::deque<size_t> free_swap_counters;
    std::deque<size_t> request_swap_counters;
    std::mutex request_swap_mutex;
    u64 pending_composite_fence{};
};

GPU::GPU(Core::System& system, bool is_async, bool use_nvdec)
    : impl{std::make_unique<Impl>(system, is_async, use_nvdec)}
{}

GPU::~GPU() = default;

std::shared_ptr<Control::ChannelState> GPU::AllocateChannel() {
    return impl->AllocateChannel();
}

void GPU::InitChannel(Control::ChannelState& to_init, u64 program_id) {
    impl->InitChannel(to_init, program_id);
}

void GPU::BindChannel(s32 channel_id) {
    impl->BindChannel(channel_id);
}

void GPU::ReleaseChannel(Control::ChannelState& to_release) {
    impl->ReleaseChannel(to_release);
}

void GPU::InitAddressSpace(Tegra::MemoryManager& memory_manager) {
    impl->InitAddressSpace(memory_manager);
}

void GPU::BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer) {
    impl->BindRenderer(std::move(renderer));
}

void GPU::FlushCommands() {
    impl->FlushCommands();
}

void GPU::InvalidateGPUCache() {
    impl->InvalidateGPUCache();
}

bool GPU::TakeCpuWrites(std::vector<std::pair<DAddr, u64>>& out) {
    return impl->TakeCpuWrites(out);
}

void GPU::ReleaseTakenWrites(const std::vector<std::pair<DAddr, u64>>* list) {
    impl->ReleaseTakenWrites(list);
}

size_t GPU::ListedWrites() {
    return impl->ListedWrites();
}

bool GPU::HasPendingCPUWrites(DAddr addr, u64 size) {
    for (auto& manager : impl->system.GetGPUDirtyMemoryManager()) {
        if (manager.Overlaps(addr, size)) {
            return true;
        }
    }
    return false;
}

void GPU::ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
    impl->ForEachPendingCpuWrite(callback);
}

void GPU::OnCommandListEnd() {
    impl->OnCommandListEnd();
}

u64 GPU::RequestFlush(DAddr addr, std::size_t size) {
    return impl->RequestSyncOperation([this, addr, size]() {
        const VideoCommon::Nx2Profile::Scope read_scope{VideoCommon::Nx2Profile::Zone::ReadFlush};
        impl->renderer->ReadRasterizer()->FlushRegion(addr, size);
    });
}

u64 GPU::RunOnGpuThread(std::function<void()>&& action) {
    return impl->RequestSyncOperation(std::move(action));
}

u64 GPU::CurrentSyncRequestFence() const {
    return impl->CurrentSyncRequestFence();
}

void GPU::WaitForSyncOperation(u64 fence) {
    return impl->WaitForSyncOperation(fence);
}

void GPU::TickWork() {
    impl->TickWork();
}

/// Gets a mutable reference to the Host1x interface
Host1x::Host1x& GPU::Host1x() {
    return impl->system.Host1x();
}

/// Gets an immutable reference to the Host1x interface.
const Host1x::Host1x& GPU::Host1x() const {
    return impl->system.Host1x();
}

Engines::Maxwell3D& GPU::Maxwell3D() {
    return impl->current_channel->payload->maxwell_3d;
}

const Engines::Maxwell3D& GPU::Maxwell3D() const {
    return impl->current_channel->payload->maxwell_3d;
}

Engines::KeplerCompute& GPU::KeplerCompute() {
    return impl->current_channel->payload->kepler_compute;
}

const Engines::KeplerCompute& GPU::KeplerCompute() const {
    return impl->current_channel->payload->kepler_compute;
}

Tegra::DmaPusher& GPU::DmaPusher() {
    return impl->current_channel->payload->dma_pusher;
}

const Tegra::DmaPusher& GPU::DmaPusher() const {
    return impl->current_channel->payload->dma_pusher;
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

const VideoCore::RendererBase& GPU::Renderer() const {
    return *impl->renderer;
}

VideoCore::ShaderNotify& GPU::ShaderNotify() {
    return impl->shader_notify;
}

const VideoCore::ShaderNotify& GPU::ShaderNotify() const {
    return impl->shader_notify;
}

void GPU::RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers,
                           std::vector<Service::Nvidia::NvFence>&& fences) {
    impl->RequestComposite(std::move(layers), std::move(fences));
}

void GPU::WaitForComposite() {
    impl->WaitForComposite();
}

std::vector<u8> GPU::GetAppletCaptureBuffer() {
    return impl->GetAppletCaptureBuffer();
}

u64 GPU::GetTicks() const {
    return impl->GetTicks();
}

bool GPU::IsAsync() const {
    return impl->is_async;
}

bool GPU::HasSuspendedChannels() const {
    return impl->scheduler.HasPendingChannelsAnyThread();
}

void GPU::LogHangState() {
    auto& host1x = impl->system.Host1x();
    std::string acquires;
    u32 count = 0;
    // Every suspended acquire registers here (Puller::ProcessSemaphoreAcquire), the channel's own
    // and the memory stage's watches alike. The word is read as the guest sees it now.
    const bool listed = Nx2SemaphoreWaiters::Get().TryForEach(
        [&](u64 address, u32 value, u32 operation, bool latched, const u32* word, double seconds) {
            if (++count > 16) {
                return;
            }
            const u32 now = word != nullptr
                                ? std::atomic_ref<u32>{*const_cast<u32*>(word)}.load(
                                      std::memory_order_relaxed)
                                : host1x.MemoryManager().Read<u32>(address);
            acquires += fmt::format(
                "{}device {:#x} {} {:#x}, word now {:#x}{}, for {:.1f} s",
                acquires.empty() ? "" : "; ", address,
                operation == Nx2SemaphoreWaiters::AcquireEqual    ? "=="
                : operation == Nx2SemaphoreWaiters::AcquireGequal ? ">="
                : operation == Nx2SemaphoreWaiters::AcquireMask   ? "&"
                                                                  : "?",
                value, now, latched ? " (latched by a GPU write)" : "", seconds);
        });
    LOG_CRITICAL(HW_GPU,
                 "Hang report: channels suspended on an acquire: {}; acquires waiting ({}): {}",
                 HasSuspendedChannels() ? "yes" : "no", count,
                 !listed            ? std::string{"not read: their lock stayed held for 200 ms"}
                 : acquires.empty() ? std::string{"none"}
                                    : acquires);
    LOG_CRITICAL(HW_GPU, "Hang report: syncpoint waits: {}",
                 host1x.GetSyncpointManager().DescribeWaits());
    VideoCommon::OneFollower::ReportHangState();
}

bool GPU::UseNvdec() const {
    return impl->use_nvdec;
}

void GPU::RendererFrameEndNotify() {
    impl->RendererFrameEndNotify();
}

void GPU::Start() {
    impl->Start();
}

void GPU::NotifyShutdown() {
    impl->NotifyShutdown();
}

void GPU::WaitForIdle() {
    impl->WaitForIdle();
}

void GPU::ObtainContext() {
    impl->ObtainContext();
}

void GPU::ReleaseContext() {
    impl->ReleaseContext();
}

void GPU::PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
    impl->PushGPUEntries(channel, std::move(entries));
}

VideoCore::RasterizerDownloadArea GPU::OnCPURead(PAddr addr, u64 size) {
    return impl->OnCPURead(addr, size);
}

void GPU::FlushRegion(DAddr addr, u64 size) {
    impl->FlushRegion(addr, size);
}

void GPU::RunInOrder(std::function<void()>&& func) {
    impl->RunInOrder(std::move(func));
}

void GPU::MapInOrder(GPUVAddr gpu_addr, u64 size, std::function<void()>&& func) {
    impl->MapInOrder(gpu_addr, size, std::move(func));
}

bool GPU::IsMapPending(GPUVAddr gpu_addr, u64 size) const {
    return impl->IsMapPending(gpu_addr, size);
}

void GPU::InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
    impl->InvalidateRegion(addr, size, preserve_gpu_writes);
}

bool GPU::OnCPUWrite(DAddr addr, u64 size) {
    return impl->OnCPUWrite(addr, size);
}

void GPU::FlushAndInvalidateRegion(DAddr addr, u64 size) {
    impl->FlushAndInvalidateRegion(addr, size);
}

} // namespace Tegra

namespace VideoCore {

bool MayHoldGpuWrites(DAddr addr, u64 size) {
    return !Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
           Tegra::Nx2MayBeGpuWritten(addr, size);
}

} // namespace VideoCore
