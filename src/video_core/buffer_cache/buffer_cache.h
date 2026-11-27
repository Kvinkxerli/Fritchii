// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/buffer_cache/uniform_cache_history.h"

#include <algorithm>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <set>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fmt/ranges.h>

#include "common/cityhash.h"
#include "common/fritchii_bind_skip.h"
#include "common/fritchii_write_ring.h"
#include "common/nx2_uniform_keep_pages.h"
#include "common/range_sets.inc"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/mem_read_log.h"
#include "video_core/compute_skip_hook.h"
#include "video_core/buffer_cache/multi_range_refresh.h"
#include "video_core/buffer_cache/nx2_bytes_equal.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/back_read_audit.h"
#include "video_core/buffer_churn_stats.h"
#include "video_core/fritchii_fill_cuts.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/churn_counters.h"
#include "video_core/fritchii_s4_count.h"
#include "video_core/gpu_split.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_stage_link.h"
#include "video_core/one_follower_hooks.h"
#include "video_core/memstage_stats.h"
#include "video_core/nx2.h"
#include "video_core/link_stats.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_list_verify.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/rasterizer_download_area.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

/// EDEN_NX2_BANK_PAGE_TABLE / EDEN_NX2_BANK_SLOTS defaults: on (table, c1) for Ocarina of Time only,
/// whose bank layout they were measured on (2026-11-11, market: c1 compared ~580k pages/s and never
/// changed; skip verify 0 missed / 0 disagreed over ~2.4M compares per 5 s; live toggle 46.8 -> 47.3
/// fps, bind stage buffers 142.7 -> 135.7 ms/s). Other titles keep both off unless set.
[[nodiscard]] inline bool NativeBankDefaultsForOcarina() {
    return Settings::GetCurrentProgramID() == 0x0400486024DEC000ULL;
}

using Core::DEVICE_PAGESIZE;

template <class P>
BufferCache<P>::BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_)
    : runtime{runtime_}, device_memory{device_memory_}, memory_tracker{device_memory} {
    // A new runtime has a new scheduler (its ticks restart) and a new stream buffer: whatever the
    // stream uniform reuse remembers refers to the old ones. Before the early return below.
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    // EXP86 (fable52_answer_gpuwritten 2.1): the page mirrors follow their range sets, and the
    // mutex's release closes the counter a hold opened. Before any other thread sees the cache.
    gpu_modified_ranges.Attach(&gpu_written_mirror.gpu_modified_pages, &gpu_written_mirror.seq);
    read_trapped_ranges.AttachMirror(&gpu_written_mirror.trap_pages, &gpu_written_mirror.seq);
    if (VideoCommon::ESplit::ThreadWanted()) [[unlikely]] {
        // FRITCHII_ESPLIT=1: E1's holds of this lock count as E1Touch (fritchii_e_split_thread.h).
        mutex.SetReleaseHook(
            [](void* seq) {
                MirrorSeq::ReleaseHook(seq);
                VideoCommon::ESplit::NoteCacheRelease();
            },
            &gpu_written_mirror.seq);
    } else {
        mutex.SetReleaseHook(&MirrorSeq::ReleaseHook, &gpu_written_mirror.seq);
    }
    // Ensure the first slot is used for the null buffer
    void(slot_buffers.insert(runtime, NullBufferParams{}));
    gpu_modified_ranges.Clear();
    inline_buffer_id = NULL_BUFFER_ID;
#ifdef YUZU_LEGACY
    immediately_free = (Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive);
#endif
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}

template <class P>
BufferCache<P>::~BufferCache() {
    VideoCore::g_read_downloads.fetch_sub(static_cast<u32>(read_downloads.size()),
                                          std::memory_order_relaxed);
}

template <class P>
void BufferCache<P>::RunGarbageCollector(bool aggressive_gc) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    const u64 ticks_to_destroy = aggressive_gc ? 60 : 120;
    int num_iterations = aggressive_gc ? 64 : 32;
    // Their GPU-written data is downloaded first, all with one wait for the GPU: one wait per
    // buffer stalled the GPU thread every frame on title 1 (an 8 GB GPU near its budget, no
    // per-fence downloads keeping the buffers clean), ~2.5% of its time.
    boost::container::small_vector<BufferId, 64> evicted;
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, [&](BufferId buffer_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        // Below the critical threshold, buffers holding GPU-written data stay: evicting one
        // means downloading it first. The thresholds compare the whole device's use (mostly
        // textures) with fixed headroom, so on an 8 GB GPU the collector ran every frame once a
        // title 1 course was loaded, and the downloads were ~2.7% of the GPU thread.
        const Buffer& buffer = slot_buffers[buffer_id];
        if (!aggressive_gc && gpu_modified_ranges.Intersects(buffer.CpuAddr(), buffer.SizeBytes())) {
            return false;
        }
        evicted.push_back(buffer_id);
        return false;
    });
    if constexpr (USE_MEMORY_MAPS) {
        DownloadBuffersMemory(evicted);
    } else {
        for (const BufferId buffer_id : evicted) {
            DownloadBufferMemory(slot_buffers[buffer_id]);
        }
    }
    for (const BufferId buffer_id : evicted) {
        DeleteBuffer(buffer_id);
    }
}

template <class P>
void BufferCache<P>::TickFrame() {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    VideoCommon::g_hot_page_epoch.fetch_add(1, std::memory_order_relaxed);
    VideoCommon::g_hot_pages_on.store(VideoCommon::HotPagesEnabled(), std::memory_order_relaxed);
    // Homebrew console apps don't create or bind any channels, so this will be nullptr.
    if (!channel_state) {
        return;
    }
    runtime.TickFrame(slot_buffers);
    if (!extent_probes.empty()) {
        CheckExtentProbes();
    }
    static const bool mem_trace = std::getenv("EDEN_NX2_MEM_TRACE") != nullptr;
    if (static std::chrono::steady_clock::time_point last{}; mem_trace) {
        if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{2}) {
            last = now;
            LOG_INFO(HW_GPU, "Memory trace: buffer cache {} MiB", buffer_bytes >> 20);
        }
    }
    // EXP81 (fable52_answer_fences2 2): every frame, the marked pages read since the last frame
    // (both read paths set preflush_read_pages, with or without the lock) get this frame's stamp.
    if (VideoCore::Nx2Debug::Experiment(81) > 0 && VideoCore::Nx2Debug::Experiment(69) > 0 &&
        !preflush_marked_ranges.Empty()) {
        preflush_marked_ranges.ForEach([&](DAddr begin, DAddr end) {
            preflush_read_pages.Take(begin, end - begin, [&](u64 page) {
                preflush_read_frames[page >> Core::DEVICE_PAGEBITS] = frame_tick;
            });
        });
    }
    // EDEN_NX2_EXP69 (live, frames, 0 = off; fable52_answer_stalls 5.1): Preflushable marks never
    // cleared, so every later GPU write to a page the CPU once read was downloaded at its fence
    // (night market: ~100 MB in ~8,200 copies a second). Every N frames the marks go; a range the
    // CPU still reads is trapped on its next read (preflushed writes arm the traps too, see
    // QueueFenceDownload) and marked again by that read's flush.
    if (const int reset_frames = VideoCore::Nx2Debug::Experiment(69);
        reset_frames > 0 && frame_tick % static_cast<u64>(reset_frames) == 0 &&
        !preflush_marked_ranges.Empty()) {
        // EDEN_NX2_EXP81 (live, frames, 0 = off; fable52_answer_readback_card 3): unmark only the
        // pages the CPU has not read for this many frames. Unmarking every page made each page
        // the game still reads need one async read (~40 a second at the market) to be marked
        // again; a mark only decides whether a write is copied at the fence or at the read.
        const u64 idle_frames = static_cast<u64>((std::max)(VideoCore::Nx2Debug::Experiment(81), 0));
        if (idle_frames == 0) {
            preflush_marked_ranges.ForEach([&](DAddr begin, DAddr end) {
                // EXP86: the mirror bit goes before the tracker's (a mark missing early only
                // sends a read to the locked path).
                gpu_written_mirror.preflush_pages.Clear(begin, end - begin, gpu_written_mirror.seq);
                memory_tracker.UnmarkRegionAsPreflushable(begin, end - begin);
                preflush_reset_bytes += end - begin;
            });
            preflush_marked_ranges.Clear();
        } else {
            std::vector<DAddr> stale_pages;
            preflush_marked_ranges.ForEach([&](DAddr begin, DAddr end) {
                for (DAddr page = begin; page < end; page += Core::DEVICE_PAGESIZE) {
                    const auto it = preflush_read_frames.find(page >> Core::DEVICE_PAGEBITS);
                    if (it == preflush_read_frames.end() || frame_tick - it->second >= idle_frames) {
                        stale_pages.push_back(page);
                    } else {
                        preflush_kept_bytes += Core::DEVICE_PAGESIZE;
                    }
                }
            });
            for (const DAddr page : stale_pages) {
                gpu_written_mirror.preflush_pages.Clear(page, Core::DEVICE_PAGESIZE,
                                                        gpu_written_mirror.seq);
                memory_tracker.UnmarkRegionAsPreflushable(page, Core::DEVICE_PAGESIZE);
                preflush_marked_ranges.Subtract(page, Core::DEVICE_PAGESIZE);
                preflush_read_frames.erase(page >> Core::DEVICE_PAGEBITS);
                preflush_reset_bytes += Core::DEVICE_PAGESIZE;
            }
        }
        ++preflush_resets;
    }
    if (frame_tick % 600 == 0 && preflush_reads_flushed.load(std::memory_order_relaxed) != 0) {
        const u64 flushed = preflush_reads_flushed.exchange(0, std::memory_order_relaxed);
        const u64 in_flight = preflush_reads_in_flight.exchange(0, std::memory_order_relaxed);
        const u64 depth = preflush_reads_in_flight_depth.exchange(0, std::memory_order_relaxed);
        LOG_INFO(HW_GPU,
                 "Preflushed reads last 600 frames: {} flushed on demand, {} of them with every "
                 "GPU-written byte already in a fence download on its way ({:.1f} fences from "
                 "the newest on average)",
                 flushed, in_flight, in_flight != 0 ? static_cast<double>(depth) / in_flight : 0.0);
        const u64 waits = fence_waits.exchange(0, std::memory_order_relaxed);
        if (waits != 0) {
            const u64 ns = fence_wait_ns.exchange(0, std::memory_order_relaxed);
            LOG_INFO(HW_GPU,
                     "Fence download waits (EXP72={}) last 600 frames: {}, {:.2f} ms average, {} "
                     "timed out (flushed as before)",
                     VideoCore::Nx2Debug::Experiment(72), waits,
                     static_cast<double>(ns) / 1e6 / static_cast<double>(waits),
                     fence_wait_timeouts.exchange(0, std::memory_order_relaxed));
        }
    }
    if (DownloadReadStats::g_enabled && frame_tick % 600 == 0) [[unlikely]] {
        DownloadReadStats::Log();
    }
    // FRITCHII_EXTENT_CENSUS / _CBUF / _SHARED / _GUARD: the 10 s lines (empty while all are off)
    if (!extent_census.empty()) [[unlikely]] {
        LogExtentCensus();
    }
    // FRITCHII_STORE_FOOTPRINT (armed): the live line every 10 s
    if (footprint_following) [[unlikely]] {
        StoreFootprintLive::MaybeLog(footprint_shadow.NumEntries());
    }
    if (ReadRunStats::g_enabled) [[unlikely]] {
        ReadRunStats::g_frame.fetch_add(1, std::memory_order_relaxed);
        ReadRunStats::FrameTick();
        // FRITCHII_STORE_FOOTPRINT / FRITCHII_FEEDER_LANE: their 10 s lines
        ReadRunStats::LogDry();
        if (frame_tick % 600 == 0) {
            if constexpr (!IS_OPENGL) {
                if (EarlyCompute::g_side_copy.Get() >= 2) {
                    const auto queue = runtime.TakeSideQueueStats();
                    EarlyCompute::NoteQueueStats(queue.raised, queue.slow_waits, queue.not_queued,
                                                 queue.main_waits, queue.unsubmitted);
                }
            }
            ReadRunStats::Log(VideoCore::Nx2Debug::Experiment(112));
        }
    }
    if (frame_tick % 600 == 0 && preflush_resets != 0) {
        LOG_INFO(HW_GPU,
                 "Preflush reset (EXP69={}, EXP81={}) last 600 frames: {} resets, {} KiB unmarked, "
                 "{} KiB kept as read recently",
                 VideoCore::Nx2Debug::Experiment(69), VideoCore::Nx2Debug::Experiment(81),
                 preflush_resets, preflush_reset_bytes >> 10, preflush_kept_bytes >> 10);
        preflush_resets = preflush_reset_bytes = preflush_kept_bytes = 0;
    }
    if (frame_tick % 600 == 0 && !uploaded_buffer_ranges.empty()) {
        for (const auto& [address, size] : uploaded_buffer_ranges) {
            device_memory.HintCold(address, size);
        }
        uploaded_buffer_ranges.clear();
    }
    if (static u64 frames{}; ++frames % 600 == 0) {
        u64 total{};
        size_t count{};
        std::vector<std::pair<u64, VAddr>> largest;
        for (const auto [buffer_id, buffer] : slot_buffers) {
            total += buffer->SizeBytes();
            ++count;
            largest.emplace_back(buffer->SizeBytes(), buffer->CpuAddr());
        }
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()),
                          largest.end(), std::greater<>{});
        LOG_INFO(HW_GPU,
                 "Buffer cache: {} buffers {} MiB; largest {}; compute write extents: {} checks "
                 "held, {} programs marked in full, {} full marks for checks",
                 count, total >> 20,
                 fmt::join(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()), ", "),
                 extent_probes_trusted, extent_probes_distrusted, extent_full_marks);
        extent_full_marks = 0;
    }

    // Calculate hits and shots and move hit bits to the right
    const u32 hits = std::reduce(channel_state->uniform_cache_hits.begin(),
                                 channel_state->uniform_cache_hits.end());
    const u32 shots = std::reduce(channel_state->uniform_cache_shots.begin(),
                                  channel_state->uniform_cache_shots.end());
    AdvanceUniformCacheHistory(channel_state->uniform_cache_hits);
    AdvanceUniformCacheHistory(channel_state->uniform_cache_shots);

    const bool skip_preferred = hits * 256 < shots * 251;
    // EXP116: a larger stream size on Vulkan (OpenGL's fast uniform buffers hold 4 KiB).
    u32 stream_max = DEFAULT_SKIP_CACHE_SIZE;
    if constexpr (!IS_OPENGL) {
        stream_max = VideoCommon::UniformStreamMaxBytes();
    }
    channel_state->uniform_buffer_skip_cache_size = skip_preferred ? stream_max : 0;

    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
    }
    if (gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
        // Switch 2 titles fill the device with textures (their collector watches the budget);
        // title 1's buffers were ~200 MiB of an 8 GB GPU's ~6.8 GB in use, yet this collector
        // ran aggressively every frame on the device-wide figure, evicting up to 64 buffers (and
        // downloading the GPU-written ones, ~5% of the GPU thread) that were recreated later.
        // Collect when the buffers themselves are large. At 1 GiB it held geometry of scenes
        // long gone: title 1's title screen grew to ~0.9 GB of buffers in ten minutes.
        if (buffer_bytes >= 512_MiB) {
            RunGarbageCollector(buffer_bytes >= 1_GiB);
        }
    } else if (total_used_memory >= minimum_memory) {
        RunGarbageCollector(total_used_memory >= critical_memory);
    }
    ++frame_tick;
    // Deleted buffers wait their frames, and with FRITCHII_MEMSTAGE_TICKS also the packet they
    // were deleted in: frames are the stage's, and the GPU thread may not have played that far.
    if constexpr (IS_OPENGL) {
        delayed_destruction_ring.Tick([](u64) { return true; });
    } else {
        runtime.NoteHeldBuffers(delayed_destruction_ring.Tick(
            [this](u64 stamp) { return runtime.DeletedBufferFree(stamp); }));
    }

    // EDEN_NX2_EXP49: read downloads no thread waited for (the other ranges the guest read
    // before) land once the host GPU ran them.
    if constexpr (!IS_OPENGL) {
        if (!read_downloads.empty()) {
            (void)runtime.IsFreeNow(read_downloads.front().tick);
            WriteCompletedReadDownloads();
        }
    }
    for (auto& buffer : async_buffers_death_ring) {
        runtime.FreeDeferredStagingBuffer(buffer);
    }
    async_buffers_death_ring.clear();
}

template <class P>
void BufferCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        virtual_ranges.Unmap(as_id, gpu_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data,
                                 bool engine_write) {
    VideoCommon::MemoryStageLink::CheckBackCache("WriteMemory");
    PromoteAhead();
    NoteAheadLevelCall(device_addr, size, "WriteMemory");
    Nx2NoteListHostWrite(device_addr, size);
    Core::g_buffer_write_pages.Mark(device_addr, size);
    Core::g_watch_write_pages.Mark(device_addr, size);
    // The written range lies within its pages: when they hold no GPU-written data, both lookups
    // below find nothing, so one lookup answers them.
    if (Nx2Perf::On(Nx2Perf::Opt::WriteIntersections) && keep_page_gpu_data) {
        const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
        // EDEN_NX2_EXP85 & 2 (live; fable52_answer_microcuts 2b): pages no GPU write ever marked
        // hold no GPU-modified range (every gpu_modified_ranges.Add marks the bit first): the
        // never-clearing bit answers before the range set's lookup.
        const bool never_gpu_written = (VideoCore::Nx2Debug::Experiment(85) & 2) != 0 &&
                                       Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
                                       !Tegra::Nx2MayBeGpuWritten(page_begin, page_end - page_begin);
        if (never_gpu_written || !gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
            if (!Nx2Perf::Verify()) {
                // EXP108 (ported): untracking the pages
                // and tracking them again at the upload changed their host protection twice (two
                // VirtualProtect calls each, ~4% of the GPU thread at OoT's water) for writes no
                // guest CPU access saw. Ours also wants no GPU-modified bit on the pages: a
                // tracked page's GPU bits count (an untracked one's are masked).
                if (engine_write && VideoCore::Nx2Debug::Experiment(108) != 0 &&
                    !memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin)) {
                    memory_tracker.MarkRegionAsWrittenTracked(device_addr, size);
                } else {
                    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
                }
                MarkVertexBuffersStale(device_addr, size);
                return;
            }
            static std::atomic<u32> logged{};
            if (IsRegionGpuModified(device_addr, size) &&
                logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: GPU-written data in {:#x}+{:#x} outside its pages",
                          device_addr, size);
            }
        }
    }
    if (IsRegionGpuModified(device_addr, size)) {
        // Guest memory holds the newer bytes already: read downloads leave the range.
        ClearReplacedDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
    }
    // The tracker marks whole pages CPU-modified, which drops their GPU-modified state: GPU-written
    // bytes elsewhere on these pages would never be downloaded, and the page's next upload would
    // send the stale guest copy over them. Hyrule Warriors: Age of Imprisonment writes small
    // structures next to the draw arguments its compute dispatches fill in; the draws then read
    // leftover bytes as vertex counts. Download those bytes first (the CPU's own were subtracted
    // above, so they aren't overwritten).
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (keep_page_gpu_data && gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
        // Switch 2 titles: upload the CPU's bytes into the buffer instead, as CachedWriteMemory
        // does, so the page stays GPU-owned and nothing waits for the host GPU.
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_INLINE_PAGE_WRITES");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        const bool inline_writes =
            forced >= 0 ? forced == 1 : Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
        if (inline_writes && IsRegionRegistered(device_addr, size)) {
            tmp_buffer.resize_destructive(size);
            VideoCommon::OneFollower::BeforeEngineRead(device_addr, size);
            device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);
            MemReadLog::Bytes(MemReadLog::Kind::InlinePageWrite, device_memory, device_addr, size);
            InlineMemoryImplementation(device_addr, size, tmp_buffer);
            return;
        }
        DownloadMemory(page_begin, page_end - page_begin);
    }
    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
}

template <class P>
typename BufferCache<P>::WriteReach BufferCache<P>::WriteReachesBuffers(DAddr device_addr,
                                                                        u64 size) const {
    // Exact (fable52_answer_next70 3.1): on pages that never left the CPU-modified state and no
    // GPU write marked, WriteMemory finds no GPU-modified range and its CPU mark sets bits that
    // are set: no page changes state, dirty_gen doesn't move (only a gained bit moves it) and
    // EXP48's hot bookkeeping acts only on tracked pages. What is left is the three marks.
    if (memory_tracker.IsRegionEverSynchronized(device_addr, size)) {
        return WriteReach::EverSynchronized;
    }
    // Every gpu_modified_ranges.Add marks the GPU-written bit (Switch 2 titles only).
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!Tegra::g_nx2_ampere.load(std::memory_order_relaxed) ||
        Tegra::Nx2MayBeGpuWritten(page_begin, page_end - page_begin)) {
        return WriteReach::MayBeGpuWritten;
    }
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        for (u32 mask = enabled_vertex_buffers_mask; mask != 0; mask &= mask - 1) {
            const Binding& binding = v_buffer[std::countr_zero(mask)];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                return WriteReach::VertexBinding;
            }
        }
    }
    return WriteReach::None;
}

template <class P>
void BufferCache<P>::NoteWriteUnlocked(DAddr device_addr, u64 size) {
    // WriteMemory's first three lines: the bank compare (d79572d28a) reads these page flags.
    Nx2NoteListHostWrite(device_addr, size);
    Core::g_buffer_write_pages.Mark(device_addr, size);
    Core::g_watch_write_pages.Mark(device_addr, size);
}

template <class P>
bool BufferCache<P>::WriteChangesNothing(DAddr device_addr, u64 size) {
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (gpu_modified_ranges.Intersects(page_begin, page_end - page_begin) ||
        AheadHere(page_begin, page_end - page_begin) ||
        !memory_tracker.IsRegionUntouched(device_addr, size)) {
        return false;
    }
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        for (u32 mask = enabled_vertex_buffers_mask; mask != 0; mask &= mask - 1) {
            const Binding& binding = v_buffer[std::countr_zero(mask)];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                return false;
            }
        }
    }
    return true;
}

template <class P>
bool BufferCache<P>::ClaimMayFindGpuWrites(
    std::span<const std::pair<DAddr, u64>> ranges) const {
    // ClaimCpuWrites acts only where IsRegionGpuModified, which needs a GPU-modified mark.
    return std::ranges::any_of(ranges, [this](const std::pair<DAddr, u64>& range) {
        return memory_tracker.IsRegionEverGpuModified(range.first, range.second);
    });
}

template <class P>
void BufferCache<P>::CachedWriteMemory(DAddr device_addr, u64 size) {
    PromoteAhead();
    NoteAheadLevelCall(device_addr, size, "CachedWriteMemory");
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return;
    }
    DAddr aligned_start = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        WriteMemory(device_addr, size);
        return;
    }

    Nx2NoteListHostWrite(device_addr, size);
    tmp_buffer.resize_destructive(size);
    VideoCommon::OneFollower::BeforeEngineRead(device_addr, size);
    device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);

    InlineMemoryImplementation(device_addr, size, tmp_buffer);
}

template <class P>
bool BufferCache<P>::OnCPUWrite(DAddr device_addr, u64 size) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return false;
    }
    if (memory_tracker.IsRegionGpuModified(device_addr, size)) {
        return true;
    }
    WriteMemory(device_addr, size);
    return false;
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> BufferCache<P>::GetFlushArea(DAddr device_addr,
                                                                              u64 size) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    area.emplace();
    DAddr device_addr_start_aligned = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    DAddr device_addr_end_aligned = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
    area->start_address = device_addr_start_aligned;
    area->end_address = device_addr_end_aligned;
    // F2 counter (EDEN_NX2_DOWNLOAD_READ_STATS): the guest reads these pages, whatever this
    // answers (and whether it disarms their read traps below).
    if (DownloadReadStats::g_enabled) [[unlikely]] {
        DownloadReadStats::Read(device_addr_start_aligned, device_addr_end_aligned);
    }
    // EXP81 (fable52_answer_fences2 2): this read's pages, stamped by TickFrame's fold. The
    // lock-free clean answer (MirrorCleanForRead) sets them too, so EXP86 sees every read.
    if (VideoCore::Nx2Debug::Experiment(81) > 0) [[unlikely]] {
        preflush_read_pages.Set(device_addr_start_aligned,
                                device_addr_end_aligned - device_addr_start_aligned);
    }
    // FRITCHII_READ_RUNS: the GPU-written run around the read's pages, before anything lands,
    // and whether the read's own bytes are GPU-written or only share their pages.
    if (ReadRunStats::g_enabled) [[unlikely]] {
        ReadRunStats::t_run_pages = 0;
        ReadRunStats::t_read_bytes = {};
        ReadRunStats::t_writer_answer = {};
        if (IsRegionGpuModified(device_addr_start_aligned,
                                device_addr_end_aligned - device_addr_start_aligned)) {
            // The newest write mark over the read's bytes, and the bytes as guest memory holds
            // them before this read's download lands (fence downloads land under this lock).
            ReadRunStats::FindWriter(device_addr, size, device_addr_start_aligned,
                                     device_addr_end_aligned);
            ReadRunStats::WriterAnswer& writer = ReadRunStats::t_writer_answer;
            if constexpr (!IS_OPENGL) {
                // FRITCHII_READ_RUNS=2: where the host GPU is with the newest write over the
                // read's pages ("Read writer ticks").
                if (ReadRunStats::g_writer_ticks && writer.area_found) {
                    const auto ticks = runtime.ReadWriterTicks(writer.area_mark.stamp);
                    writer.write_tick = ticks.write;
                    writer.read_tick = ticks.current;
                    writer.known_tick = ticks.known;
                    // FRITCHII_EXTENT_CBUF / _SHARED =3: the tick of the mark the proof leaves
                    // over the read's bytes (WaitReadWriter times it)
                    if (writer.dry_cover_found && writer.dry_cover.stamp != 0) [[unlikely]] {
                        writer.dry_cover_tick =
                            runtime.ReadWriterTicks(writer.dry_cover.stamp).write;
                    }
                    // FRITCHII_FEEDER_LANE: the writing feeder's D as ticks (0: nothing unrun)
                    const EarlyCompute::Lane& lane = writer.area_mark.lane;
                    if (EarlyCompute::g_lane && lane.judged) [[unlikely]] {
                        writer.lane_tick =
                            lane.d != 0 ? runtime.ReadWriterTicks(lane.d).write : 0;
                        writer.lane_copy_tick =
                            lane.d_copy != 0 ? runtime.ReadWriterTicks(lane.d_copy).write : 0;
                    }
                }
            }
            writer.compared =
                static_cast<u32>((std::min<u64>)(size, ReadRunStats::CompareBytes));
            VideoCommon::OneFollower::BeforeEngineRead(device_addr, writer.compared);
            device_memory.ReadBlockUnsafe(device_addr, writer.before.data(), writer.compared);
            ReadRunStats::ByteCheck check{.begin = device_addr, .end = device_addr + size};
            gpu_modified_ranges.ForEachInRange(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned,
                [&](DAddr lower, DAddr upper) { check.Add(lower, upper); });
            ReadRunStats::t_read_bytes.modified = check.Answer();
            const auto [run_begin, run_end] = GpuModifiedRunAround(
                device_addr_start_aligned, device_addr_end_aligned,
                ReadRunStats::RunCountPages * Core::DEVICE_PAGESIZE, false);
            ReadRunStats::t_run_pages =
                (Common::AlignUp(run_end, Core::DEVICE_PAGESIZE) -
                 Common::AlignDown(run_begin, Core::DEVICE_PAGESIZE)) >>
                Core::DEVICE_PAGEBITS;
            writer.run_begin = run_begin;
            writer.run_end = run_end;
        }
    }
    const bool preflushable = memory_tracker.IsRegionPreflushable(device_addr, size);
    // FRITCHII_STORE_FOOTPRINT=1/2 (live once armed): every GPU-modified byte of the read is held
    // only by marks whose store footprints exclude it, so no GPU write since those bytes were
    // last clean touched them and guest memory holds them. Answered without a wait; nothing else
    // changes (FootprintAnswersRead re-arms the pages' traps). The core may keep only the read's
    // own bytes as its read area.
    if (footprint_following && !preflushable) [[unlikely]] {
        const int live = StoreFootprintLive::Mode();
        if ((live == 1 || live == 2) &&
            FootprintAnswersRead(device_addr, size, device_addr_start_aligned,
                                 device_addr_end_aligned, live)) {
            area->start_address = device_addr;
            area->end_address = device_addr + size;
            area->preemtive = true;
            return area;
        }
    }
    // EDEN_NX2_EXP49: all of the area's GPU-modified bytes on their way for an earlier read. No
    // new copies: the read waits for those (in_flight), or finds them landed.
    if (!read_downloads.empty() &&
        IsRegionGpuModified(device_addr_start_aligned,
                            device_addr_end_aligned - device_addr_start_aligned) &&
        ReadDownloadsCover(device_addr_start_aligned,
                           device_addr_end_aligned - device_addr_start_aligned)) {
        if (!preflushable) {
            // The core took the trap, as below.
            read_trapped_ranges.Subtract(device_addr_start_aligned,
                                         device_addr_end_aligned - device_addr_start_aligned);
        }
        if (!WriteFinishedReadDownloads(device_addr_start_aligned,
                                        device_addr_end_aligned - device_addr_start_aligned)) {
            area->preemtive = false;
            area->in_flight = true;
            if (ReadRunStats::g_enabled) [[unlikely]] {
                // FRITCHII_READ_RUNS: the read download's bytes in the read's pages.
                ReadRunStats::ByteCheck check{.begin = device_addr, .end = device_addr + size};
                read_download_ranges.ForEachInRange(
                    device_addr_start_aligned,
                    device_addr_end_aligned - device_addr_start_aligned,
                    [&](DAddr lower, DAddr upper) { check.Add(lower, upper); });
                ReadRunStats::t_read_bytes.on_way = check.Answer();
            }
            return area;
        }
        // The host GPU had run them: in guest memory now. Anything still GPU-modified is newer
        // and flushes below.
    }
    if (preflushable) {
        // A preflushed region's data reaches guest memory when its fence pops (PopAsyncBuffers
        // clears gpu_modified_ranges then). A read before that saw stale words: Ocarina of Time's
        // whole screen flashed white near the sword boulder with EDEN_NX2_PREFLUSH=1 (232 flash
        // watch reports a session, 3 without preflush). Such a read flushes on demand instead.
        area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                               device_addr_end_aligned - device_addr_start_aligned);
        if (!area->preemtive) {
            // Counter only: would waiting for a fence download already on its way have done?
            const DAddr begin = device_addr_start_aligned;
            const u64 length = device_addr_end_aligned - device_addr_start_aligned;
            bool all_in_flight = !uncommitted_gpu_modified_ranges.Intersects(begin, length);
            size_t newest = 0;
            if (all_in_flight) {
                gpu_modified_ranges.ForEachInRange(begin, length, [&](DAddr lower, DAddr upper) {
                    bool found = false;
                    size_t index = 0;
                    for (const auto& ranges : pending_download_ranges) {
                        if (ranges.Intersects(lower, upper - lower)) {
                            found = true;
                            newest = (std::max)(newest, index);
                        }
                        ++index;
                    }
                    all_in_flight = all_in_flight && found;
                });
            }
            preflush_reads_flushed.fetch_add(1, std::memory_order_relaxed);
            if (all_in_flight) {
                if (VideoCore::Nx2Debug::Experiment(72) != 0) {
                    area->fence_download =
                        fence_downloads_landed.load(std::memory_order_acquire) + newest + 1;
                }
                preflush_reads_in_flight.fetch_add(1, std::memory_order_relaxed);
                preflush_reads_in_flight_depth.fetch_add(
                    pending_download_ranges.size() - newest, std::memory_order_relaxed);
                if (ReadRunStats::g_enabled) [[unlikely]] {
                    // FRITCHII_READ_RUNS: the fence downloads the read waits for (up to and
                    // including `newest`), their bytes in the read's pages.
                    ReadRunStats::ByteCheck check{.begin = device_addr,
                                                  .end = device_addr + size};
                    for (size_t index = 0;
                         index <= newest && index < pending_download_ranges.size(); ++index) {
                        pending_download_ranges[index].ForEachInRange(
                            begin, length,
                            [&](DAddr lower, DAddr upper) { check.Add(lower, upper); });
                    }
                    ReadRunStats::t_read_bytes.on_way = check.Answer();
                }
            }
        }
        return area;
    }
    area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                           device_addr_end_aligned - device_addr_start_aligned);
    // A preflushable mark persists and downloads later GPU writes at every fence. Native titles
    // instead keep read traps and flush on demand; EDEN_NX2_PREFLUSH=1 restores preflushing.
    // Not Ocarina of Time either: preflushing its CPU's ~170 reads a second of GPU-written pages
    // was worth ~3 fps, but a third of loads into Castle Town then rendered overexposed and
    // tinted (stale or wrong exposure data), against none with the traps.
    static const std::optional<bool> forced_preflush = []() -> std::optional<bool> {
        const char* value = std::getenv("EDEN_NX2_PREFLUSH");
        if (value != nullptr) {
            return value[0] != '0';
        }
        return std::nullopt;
    }();
    const bool preflush =
        forced_preflush.value_or(!Tegra::g_nx2_ampere.load(std::memory_order_relaxed));
    if (!area->preemtive) {
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Marked(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned, preflush);
        }
        if (preflush) {
            memory_tracker.MarkRegionAsPreflushable(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
            // EXP86: the mirror bit after the tracker's (one missing late only sends a read to
            // the locked path).
            gpu_written_mirror.preflush_pages.Set(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned,
                gpu_written_mirror.seq);
            preflush_marked_ranges.Add(device_addr_start_aligned,
                                       device_addr_end_aligned - device_addr_start_aligned);
        } else {
            if constexpr (USE_MEMORY_MAPS) {
                cpu_read_ranges.Add(area->start_address, area->end_address - area->start_address);
                download_cpu_read_ranges = true;
            }
            // The core consumed this trap before GetFlushArea. Forget it so the next GPU write
            // really rearms the core instead of being skipped by the deduplication cache.
            read_trapped_ranges.Subtract(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        }
    } else {
        // The core keeps this area and has no mark. The next GPU write here must arm the range
        // again, which ends the area and sets the trap.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
    }
    return area;
}

template <class P>
void BufferCache<P>::DownloadMemory(DAddr device_addr, u64 size) {
    VideoCommon::MemoryStageLink::CheckBackCache("DownloadMemory");
    PromoteAhead();
    const Common::WriteRing::SiteScope write_site{"buffer download (DownloadMemory)", true};
    if constexpr (USE_MEMORY_MAPS) {
        if (download_cpu_read_ranges) {
            download_cpu_read_ranges = false;
            boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges{{device_addr, size}};
            cpu_read_ranges.ForEach([&](DAddr lower, DAddr upper) {
                gpu_modified_ranges.ForEachInRange(lower, upper - lower,
                                                   [&](DAddr start, DAddr end) {
                                                       ranges.emplace_back(start, end - start);
                                                   });
            });
            DownloadRangesMemory(ranges);
            return;
        }
    }
    ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
        const DAddr start = (std::max)(device_addr, buffer.CpuAddr());
        const DAddr end = (std::min)(device_addr + size, buffer.CpuAddr() + buffer.SizeBytes());
        if (start < end) {
            DownloadBufferMemory(buffer, start, end - start);
        }
    });
}

template <class P>
void BufferCache<P>::DownloadMemoryForRead(DAddr device_addr, u64 size) {
    VideoCommon::MemoryStageLink::CheckBackCache("DownloadMemoryForRead");
    PromoteAhead();
    const Common::WriteRing::SiteScope write_site{"buffer download (DownloadMemoryForRead)", true};
    if constexpr (IS_OPENGL || !USE_MEMORY_MAPS) {
        DownloadMemory(device_addr, size);
    } else {
        WriteCompletedReadDownloads();
        // The ranges DownloadMemory takes: the guest CPU soon reads those it read before too.
        boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges{{device_addr, size}};
        if (download_cpu_read_ranges) {
            download_cpu_read_ranges = false;
            cpu_read_ranges.ForEach([&](DAddr lower, DAddr upper) {
                gpu_modified_ranges.ForEachInRange(
                    lower, upper - lower,
                    [&](DAddr start, DAddr end) { ranges.emplace_back(start, end - start); });
            });
        }
        // EXP112 (live, 0 = off; fable52_answer_fountain 4): the GPU-written run around the
        // read's pages too, up to 256 KiB, in this same download. RecordReadDownload takes only
        // GPU-modified bytes there (and none already on their way), so it is what the reads of
        // the neighbouring pages would download one round trip each. Once it lands they are clean
        // (WriteReadDownloads); their read trap marks go now, so EXP86's mirrors answer those
        // reads without the lock. Taking a mark early is always safe: the next GPU write there
        // arms the pages again. Neighbouring Preflushable pages end the run (their fence
        // downloads take them). The read's own pages may be Preflushable: with preflushing on,
        // GetFlushArea marks them for this very read before it comes here (the first version
        // skipped those and never ran at the fountain).
        if (VideoCore::Nx2Debug::Experiment(112) != 0) {
            using Why = ReadRunStats::NoRun;
            constexpr u64 RunFlushBytes = 256_KiB;
            const DAddr begin = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
            const DAddr end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
            if (ReadRunStats::g_enabled) [[unlikely]] {
                ReadRunStats::g_run_asked.fetch_add(1, std::memory_order_relaxed);
                if (memory_tracker.IsRegionPreflushable(begin, end - begin)) {
                    ReadRunStats::g_run_own_marked.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (!IsRegionGpuModified(begin, end - begin)) {
                ReadRunStats::CountNoRun(Why::NotGpuModified);
            } else {
                const auto [run_begin, run_end] =
                    GpuModifiedRunAround(begin, end, RunFlushBytes, true);
                if (run_begin == run_end) {
                    ReadRunStats::CountNoRun(Why::NoBuffer);
                } else if (run_begin >= begin && run_end <= end) {
                    if (ReadRunStats::g_enabled) [[unlikely]] {
                        // Would the run have gone on without the Preflushable stop?
                        const auto [any_begin, any_end] =
                            GpuModifiedRunAround(begin, end, RunFlushBytes, false);
                        ReadRunStats::CountNoRun(any_begin < begin || any_end > end
                                                     ? Why::NeighbourPreflushable
                                                     : Why::NoNeighbour);
                    }
                } else {
                    ranges.emplace_back(run_begin, run_end - run_begin);
                    const DAddr pages_begin = Common::AlignDown(run_begin, Core::DEVICE_PAGESIZE);
                    const DAddr pages_end = Common::AlignUp(run_end, Core::DEVICE_PAGESIZE);
                    read_trapped_ranges.Subtract(pages_begin, pages_end - pages_begin);
                    if (ReadRunStats::g_enabled) [[unlikely]] {
                        ReadRunStats::g_run_flushes.fetch_add(1, std::memory_order_relaxed);
                        const u64 before = pages_begin < begin ? begin - pages_begin : 0;
                        const u64 after = pages_end > end ? pages_end - end : 0;
                        ReadRunStats::g_run_bytes.fetch_add(before + after,
                                                            std::memory_order_relaxed);
                    }
                }
            }
        }
        RecordReadDownload(ranges);
    }
}

template <class P>
std::pair<DAddr, DAddr> BufferCache<P>::GpuModifiedRunAround(DAddr begin, DAddr end,
                                                             u64 max_bytes,
                                                             bool stop_at_preflushable) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    DAddr buffer_begin = 0;
    DAddr buffer_end = 0;
    ForEachBufferInRange(begin, end - begin, [&](BufferId, Buffer& buffer) {
        if (buffer_end == 0 && buffer.CpuAddr() <= begin &&
            begin < buffer.CpuAddr() + buffer.SizeBytes()) {
            buffer_begin = buffer.CpuAddr();
            buffer_end = buffer.CpuAddr() + buffer.SizeBytes();
        }
    });
    if (buffer_end == 0) {
        // The read's first page starts before any buffer: look from its first byte in a buffer.
        ForEachBufferInRange(begin, end - begin, [&](BufferId, Buffer& buffer) {
            if (buffer_end == 0) {
                buffer_begin = buffer.CpuAddr();
                buffer_end = buffer.CpuAddr() + buffer.SizeBytes();
            }
        });
        if (buffer_end == 0) {
            return {begin, begin};
        }
    }
    // Whether the page at `page` holds GPU-modified bytes inside the buffer.
    const auto written = [&](DAddr page) {
        const DAddr lower = (std::max)(page, buffer_begin);
        const DAddr upper = (std::min)(page + Core::DEVICE_PAGESIZE, buffer_end);
        if (lower >= upper || !gpu_modified_ranges.Intersects(lower, upper - lower)) {
            return false;
        }
        return !stop_at_preflushable || !memory_tracker.IsRegionPreflushable(lower, upper - lower);
    };
    DAddr run_begin = begin;
    DAddr run_end = end;
    while (run_end - run_begin < max_bytes && run_end < buffer_end && written(run_end)) {
        run_end += Core::DEVICE_PAGESIZE;
    }
    while (run_end - run_begin < max_bytes && run_begin > buffer_begin &&
           written(run_begin - Core::DEVICE_PAGESIZE)) {
        run_begin -= Core::DEVICE_PAGESIZE;
    }
    return {(std::max)(run_begin, buffer_begin), (std::min)(run_end, buffer_end)};
}

template <class P>
void BufferCache<P>::RecordReadDownload(std::span<const std::pair<DAddr, u64>> ranges) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    if constexpr (!IS_OPENGL && USE_MEMORY_MAPS) {
        struct Copy {
            BufferId buffer_id;
            BufferCopy copy;
        };
        boost::container::small_vector<Copy, 16> copies;
        ReadDownload download;
        u64 total_size_bytes = 0;
        const auto add_copy = [&](BufferId buffer_id, DAddr buffer_addr, DAddr begin, DAddr end) {
            copies.push_back({buffer_id, BufferCopy{
                                             .src_offset = begin - buffer_addr,
                                             .dst_offset = total_size_bytes,
                                             .size = end - begin,
                                         }});
            total_size_bytes += Common::AlignUp(end - begin, 64);
            download.write_ranges.Add(begin, end - begin);
            download.clean_ranges.Add(begin, end - begin);
            read_download_ranges.Add(begin, end - begin);
        };
        boost::container::small_vector<std::pair<DAddr, DAddr>, 8> gaps;
        for (const auto& [range_addr, range_size] : ranges) {
            ForEachBufferInRange(range_addr, range_size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(range_addr, buffer_addr);
                const DAddr end =
                    (std::min)(range_addr + range_size, buffer_addr + buffer.SizeBytes());
                if (begin >= end) {
                    return;
                }
                // The bytes DownloadRangesMemory takes, but they stay GPU-modified (in the tracker
                // too) until they are in guest memory (WriteReadDownloads).
                memory_tracker.ForEachDownloadRange(
                    begin, end - begin, false, [&](u64 run_addr, u64 run_size) {
                        gpu_modified_ranges.ForEachInRange(
                            run_addr, run_size, [&](DAddr start, DAddr range_end) {
                                // Without the bytes a read download already brings (an earlier
                                // one, or this one: the ranges can overlap).
                                gaps.clear();
                                DAddr next = start;
                                read_download_ranges.ForEachInRange(
                                    start, range_end - start, [&](DAddr lower, DAddr upper) {
                                        if (lower > next) {
                                            gaps.emplace_back(next, lower);
                                        }
                                        next = (std::max)(next, upper);
                                    });
                                if (next < range_end) {
                                    gaps.emplace_back(next, range_end);
                                }
                                for (const auto& [gap_begin, gap_end] : gaps) {
                                    add_copy(buffer_id, buffer_addr, gap_begin, gap_end);
                                }
                            });
                        // The downloads taken at fences are older: they must not land over these.
                        ClearFenceDownloads(run_addr, run_size);
                    });
            });
        }
        if (copies.empty()) {
            return;
        }
        download.id = ++read_download_ids;
        // FRITCHII_READ_SIDE_COPY: where the newest write over these bytes is (before anything
        // is recorded: the current tick is the one the submit below closes).
        const int side_mode = EarlyCompute::g_side_copy.Get();
        if (side_mode != 0) [[unlikely]] {
            boost::container::small_vector<std::pair<DAddr, u64>, 8> copied;
            for (const Copy& entry : copies) {
                copied.emplace_back(slot_buffers[entry.buffer_id].CpuAddr() +
                                        entry.copy.src_offset,
                                    entry.copy.size);
            }
            u64 stamp = 0;
            if (!ReadRunStats::NewestMarkOver(copied, stamp)) {
                EarlyCompute::NoteSideDownload(EarlyCompute::SideWriter::NoMark);
            } else {
                const auto ticks = runtime.ReadWriterTicks(stamp);
                download.side_eligible = ticks.write < ticks.current;
                download.side_tick = ticks.write;
                EarlyCompute::NoteSideDownload(download.side_eligible
                                                   ? EarlyCompute::SideWriter::Submitted
                                                   : EarlyCompute::SideWriter::Open);
            }
        }
        // >= 2: the copy on the side queue too (2, into a second staging buffer, compared at the
        // landing) or instead (3). Only for a writer already submitted (eligible).
        const bool side_copy =
            side_mode >= 2 && download.side_eligible && runtime.SideCopyReady();
        const bool main_copy = !side_copy || side_mode == 2;
        auto staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
        boost::container::small_vector<typename Runtime::SideCopyRegion, 8> side_regions;
        std::optional<Async_Buffer> verify_staging;
        if (side_copy && side_mode == 2) {
            verify_staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
        }
        std::ranges::stable_sort(copies, {}, &Copy::buffer_id);
        for (Copy& entry : copies) {
            Buffer& buffer = slot_buffers[entry.buffer_id];
            const u64 relative_dst = entry.copy.dst_offset;
            entry.copy.dst_offset += staging.offset;
            buffer.MarkUsage(entry.copy.src_offset, entry.copy.size);
            if (side_copy) {
                const Async_Buffer& target = verify_staging ? *verify_staging : staging;
                side_regions.push_back({
                    .src = buffer.Handle(),
                    .dst = target.buffer,
                    .src_offset = entry.copy.src_offset,
                    .dst_offset = target.offset + relative_dst,
                    .size = entry.copy.size,
                });
            }
            download.copies.push_back(BufferCopy{
                .src_offset = buffer.CpuAddr() + entry.copy.src_offset,
                .dst_offset = entry.copy.dst_offset,
                .size = entry.copy.size,
            });
        }
        const auto copy_batched = [&](const Async_Buffer& dst_staging) {
            boost::container::small_vector<BufferCopy, 16> buffer_copies;
            for (size_t i = 0; i < copies.size();) {
                const BufferId buffer_id = copies[i].buffer_id;
                Buffer& buffer = slot_buffers[buffer_id];
                buffer_copies.clear();
                while (i < copies.size() && copies[i].buffer_id == buffer_id) {
                    buffer_copies.push_back(copies[i++].copy);
                }
                // Mapped buffers too: the GPU thread's later work may write them before the
                // copy is read.
                runtime.CopyBuffer(dst_staging.buffer, buffer, buffer_copies, true);
            }
        };
        if (main_copy) {
            copy_batched(staging);
        }
        download.staging = staging;
        if (side_copy) {
            download.side_value = runtime.SideCopy(download.side_tick, side_regions);
            if (download.side_value == 0 && !main_copy) {
                // The side submit failed: the main copies after all.
                copy_batched(staging);
            }
            if (verify_staging) {
                if (download.side_value != 0) {
                    download.side_verify = true;
                    download.verify_staging = *verify_staging;
                } else {
                    async_buffers_death_ring.push_back(*verify_staging);
                }
            }
            EarlyCompute::NoteSideSubmit(download.side_value != 0);
        }
        // Submitted now, not waited for: the reading thread waits (FinishReadDownloads).
        download.tick = runtime.SubmitForHostRead();
        read_downloads.push_back(std::move(download));
        VideoCore::g_read_downloads.fetch_add(1, std::memory_order_release);
    }
}

template <class P>
std::optional<u64> BufferCache<P>::FinishReadDownloads(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        // Pushed under `mutex` before the read's sync operation completed, or on this thread.
        if (VideoCore::g_read_downloads.load(std::memory_order_acquire) == 0) {
            return std::nullopt;
        }
        std::unique_lock lock{mutex};
        // The newest one over the range: those before it are written first.
        std::optional<u64> tick;
        bool side_eligible = false;
        u64 side_tick = 0;
        u64 side_value = 0;
        bool side_verify = false;
        u64 download_id = 0;
        u32 downloads_over = 0;
        for (const ReadDownload& download : read_downloads) {
            if (download.write_ranges.Intersects(device_addr, size)) {
                tick = download.tick;
                side_eligible = download.side_eligible;
                side_tick = download.side_tick;
                side_value = download.side_value;
                side_verify = download.side_verify;
                download_id = download.id;
                ++downloads_over;
            }
        }
        if (!tick) {
            return std::nullopt;
        }
        const bool reading = VideoCore::AsyncReadback::t_reading;
        // FRITCHII_READ_SIDE_COPY=3: the download is the side copy alone. Its reader waits for
        // that copy only and lands it at once, out of order (read downloads never overlap), when
        // no other download covers the range. Anyone else waits for the main tick as before:
        // its submission waited for the side copy.
        if (side_value != 0 && !side_verify && reading && downloads_over == 1) [[unlikely]] {
            u64 waited = 0;
            if (!runtime.SideDone(side_value)) {
                lock.unlock();
                const u64 wait_start = VideoCore::AsyncReadback::Now();
                runtime.WaitSide(side_value);
                waited = (std::max)(VideoCore::AsyncReadback::Now() - wait_start, u64{1});
                VideoCore::AsyncReadback::CountHostWait(waited);
                VideoCore::AsyncReadback::Add(VideoCore::AsyncReadback::g_counters.on_card_ns,
                                              waited);
                lock.lock();
            }
            EarlyCompute::NoteSideLanded(waited);
            for (auto it = read_downloads.begin(); it != read_downloads.end(); ++it) {
                if (it->id == download_id) {
                    LandReadDownload(*it);
                    read_downloads.erase(it);
                    VideoCore::g_read_downloads.fetch_sub(1, std::memory_order_release);
                    break;
                }
            }
            return waited;
        }
        u64 waited = 0;
        if (!runtime.IsFree(*tick)) {
            lock.unlock();
            const u64 wait_start = VideoCore::AsyncReadback::Now();
            // FRITCHII_READ_SIDE_COPY=2 (verify): the side copy's own wait first, then the main
            // copy's as before (both land, and are compared, at the landing).
            u64 side_ns = 0;
            if (side_verify && reading) [[unlikely]] {
                runtime.WaitSide(side_value);
                side_ns = (std::max)(VideoCore::AsyncReadback::Now() - wait_start, u64{1});
            }
            // FRITCHII_READ_RUNS=2 (counter only): the reader's wait, first until its writer's
            // submission ran. The "on the host GPU" figure keeps both parts.
            // FRITCHII_READ_SIDE_COPY (dry): an eligible download's wait is cut at the newest
            // write over all its bytes instead (RecordReadDownload), the side copy's own wait.
            u64 writer_ns = 0;
            const bool side_cut = side_eligible && EarlyCompute::g_side_copy && reading &&
                                  side_ns == 0;
            if (side_cut) [[unlikely]] {
                writer_ns = WaitReadWriter(side_tick);
                VideoCore::AsyncReadback::Add(VideoCore::AsyncReadback::g_counters.on_card_ns,
                                              writer_ns);
            } else if (ReadRunStats::g_writer_ticks && VideoCore::AsyncReadback::t_reading)
                [[unlikely]] {
                writer_ns = WaitReadWriter();
                VideoCore::AsyncReadback::Add(VideoCore::AsyncReadback::g_counters.on_card_ns,
                                              writer_ns);
            }
            runtime.WaitForHostRead(*tick);
            waited = (std::max)(VideoCore::AsyncReadback::Now() - wait_start, u64{1});
            VideoCore::AsyncReadback::CountHostWait(waited);
            if (ReadRunStats::t_writer_answer.cut) [[unlikely]] {
                ReadRunStats::t_writer_answer.rest_wait_ns =
                    waited > writer_ns ? waited - writer_ns : 0;
                if (side_cut) {
                    EarlyCompute::NoteSideCut(writer_ns,
                                              ReadRunStats::t_writer_answer.rest_wait_ns);
                }
                MeasureRunUp();
            }
            if (side_ns != 0) [[unlikely]] {
                EarlyCompute::NoteSideVerifyWait(side_ns, waited);
            }
            lock.lock();
        }
        WriteReadDownloads(*tick);
        return waited;
    }
    return std::nullopt;
}

template <class P>
void BufferCache<P>::OutdateReadDownloads(DAddr device_addr, u64 size) {
    // The bytes stay GPU-modified when the downloads land, so a later read downloads them.
    if (!read_downloads.empty()) [[unlikely]] {
        for (ReadDownload& download : read_downloads) {
            download.clean_ranges.Subtract(device_addr, size);
        }
        read_download_ranges.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::WriteFinishedReadDownloads(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        std::optional<u64> tick;
        for (const ReadDownload& download : read_downloads) {
            if (download.write_ranges.Intersects(device_addr, size)) {
                tick = download.tick;
            }
        }
        if (!tick || !runtime.IsFreeNow(*tick)) {
            return false;
        }
        WriteReadDownloads(*tick);
        return true;
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::WriteReadDownloads(u64 tick) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    const Common::WriteRing::SiteScope write_site{"buffer read download (WriteReadDownloads)", true};
    if constexpr (!IS_OPENGL) {
        while (!read_downloads.empty() && read_downloads.front().tick <= tick) {
            LandReadDownload(read_downloads.front());
            read_downloads.pop_front();
            VideoCore::g_read_downloads.fetch_sub(1, std::memory_order_release);
        }
    }
}

template <class P>
void BufferCache<P>::LandReadDownload(ReadDownload& download) {
    if constexpr (!IS_OPENGL) {
        {
            // FRITCHII_READ_SIDE_COPY=2: the side copy's bytes against the main copy's. The main
            // tick is done here or the side value was waited for: every main submission after
            // the side submit waited for it, so both are in.
            if (download.side_verify) [[unlikely]] {
                runtime.WaitSide(download.side_value);
                u64 bytes = 0;
                u64 differing = 0;
                const u8* const main_base = download.staging.mapped_span.data();
                const u8* const side_base = download.verify_staging.mapped_span.data();
                for (const BufferCopy& copy : download.copies) {
                    const u64 relative = copy.dst_offset - download.staging.offset;
                    bytes += copy.size;
                    if (std::memcmp(main_base + relative, side_base + relative, copy.size) != 0) {
                        ++differing;
                    }
                }
                EarlyCompute::NoteSideVerify(download.copies.size(), differing, bytes);
                async_buffers_death_ring.push_back(download.verify_staging);
                download.side_verify = false;
            }
            // Guest memory this writes. The guest CPU writes collected from now on store their
            // bytes once this download is in guest memory.
            DAddr landing_begin = ~DAddr{0};
            DAddr landing_end = 0;
            for (const BufferCopy& copy : download.copies) {
                landing_begin = (std::min)(landing_begin, static_cast<DAddr>(copy.src_offset));
                landing_end =
                    (std::max)(landing_end, static_cast<DAddr>(copy.src_offset + copy.size));
            }
            {
                const VideoCore::DownloadLanding landing{landing_begin, landing_end};
                // CPU writes are collected before they reach guest memory, and stay pending
                // until applied, which takes them out of write_ranges (ClaimCpuWrites): either
                // way they keep their bytes.
                const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
                const u8* const base = download.staging.mapped_span.data();
                for (const BufferCopy& copy : download.copies) {
                    const DAddr copy_addr = static_cast<DAddr>(copy.src_offset);
                    const u8* const data = base + (copy.dst_offset - download.staging.offset);
                    download.write_ranges.ForEachInRange(
                        copy_addr, copy.size, [&](DAddr start, DAddr end) {
                            WriteDownloadedMemory(cpu_written, start, data + (start - copy_addr),
                                                  end - start);
                        });
                }
            }
            download.clean_ranges.ForEach([&](DAddr start, DAddr end) {
                gpu_modified_ranges.Subtract(start, end - start);
                read_download_ranges.Subtract(start, end - start);
            });
            // As a synchronous download leaves them: pages without GPU-modified bytes left are
            // clean in the tracker too.
            download.clean_ranges.ForEach([&](DAddr start, DAddr end) {
                const DAddr last = Common::AlignUp(end, DEVICE_PAGESIZE);
                for (DAddr page = Common::AlignDown(start, DEVICE_PAGESIZE); page < last;
                     page += DEVICE_PAGESIZE) {
                    // A mark ahead of the back keeps its bit (its step e ran at the fill).
                    if (!gpu_modified_ranges.Intersects(page, DEVICE_PAGESIZE) &&
                        !AheadIntersects(page, DEVICE_PAGESIZE)) {
                        memory_tracker.UnmarkRegionAsGpuModified(page, DEVICE_PAGESIZE);
                    }
                }
            });
            // Freed by the GPU thread (TickFrame).
            async_buffers_death_ring.push_back(download.staging);
        }
    }
}

template <class P>
void BufferCache<P>::WriteCompletedReadDownloads() {
    if constexpr (!IS_OPENGL) {
        while (!read_downloads.empty() && runtime.IsFree(read_downloads.front().tick)) {
            WriteReadDownloads(read_downloads.front().tick);
        }
    }
}

template <class P>
bool BufferCache<P>::ReadDownloadsCover(DAddr device_addr, u64 size) const {
    bool covered = true;
    gpu_modified_ranges.ForEachInRange(device_addr, size, [&](DAddr start, DAddr end) {
        DAddr next = start;
        read_download_ranges.ForEachInRange(start, end - start, [&](DAddr lower, DAddr upper) {
            covered = covered && lower <= next;
            next = (std::max)(next, upper);
        });
        covered = covered && next >= end;
    });
    return covered;
}

template <class P>
void BufferCache<P>::DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    boost::container::small_vector<Download, 16> downloads;
    u64 total_size_bytes = 0;
    for (const auto& [range_addr, range_size] : ranges) {
        ForEachBufferInRange(range_addr, range_size, [&](BufferId buffer_id, Buffer& buffer) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const DAddr start = (std::max)(range_addr, buffer_addr);
            const DAddr end = (std::min)(range_addr + range_size, buffer_addr + buffer.SizeBytes());
            if (start >= end) {
                return;
            }
            memory_tracker.ForEachDownloadRangeAndClear(
                start, end - start, [&](u64 device_addr_out, u64 out_size) {
                    gpu_modified_ranges.ForEachInRange(
                        device_addr_out, out_size, [&](DAddr lower, DAddr upper) {
                            downloads.push_back({buffer_id, BufferCopy{
                                                                .src_offset = lower - buffer_addr,
                                                                .dst_offset = total_size_bytes,
                                                                .size = upper - lower,
                                                            }});
                            total_size_bytes += Common::AlignUp(upper - lower, 64);
                        });
                    // Lands under `mutex` once the host GPU ran it: read downloads leave the range.
                    ClearReplacedDownload(device_addr_out, out_size);
                    gpu_modified_ranges.Subtract(device_addr_out, out_size);
                });
        });
    }
    if (total_size_bytes == 0) {
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    std::ranges::stable_sort(downloads, {}, &Download::buffer_id);
    boost::container::small_vector<BufferCopy, 16> buffer_copies;
    for (size_t i = 0; i < downloads.size();) {
        const BufferId buffer_id = downloads[i].buffer_id;
        Buffer& buffer = slot_buffers[buffer_id];
        buffer_copies.clear();
        while (i < downloads.size() && downloads[i].buffer_id == buffer_id) {
            Download& download = downloads[i++];
            download.copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(download.copy.src_offset, download.copy.size);
            buffer_copies.push_back(download.copy);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, buffer_copies, true);
    }
    runtime.Finish();
    const u8* const mapped_memory = download_staging.mapped_span.data();
    // The CPU writes collected from now on store their bytes once this is in guest memory.
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const Download& download : downloads) {
        const DAddr address = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        landing_begin = (std::min)(landing_begin, address);
        landing_end = (std::max)(landing_end, address + download.copy.size);
    }
    const VideoCore::DownloadLanding landing{landing_begin, landing_end};
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        WriteDownloadedMemory(
            cpu_written, device_addr,
            mapped_memory + (download.copy.dst_offset - download_staging.offset),
            download.copy.size);
    }
}

template <class P>
void BufferCache<P>::DownloadMemoryRanges(std::span<const std::pair<DAddr, u64>> ranges) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    constexpr bool has_mapped = requires(Buffer& buffer) {
        buffer.Mapped();
        buffer.LastUsageTick();
    };
    if constexpr (!USE_MEMORY_MAPS) {
        for (const auto& [device_addr, size] : ranges) {
            DownloadMemory(device_addr, size);
        }
    } else {
        struct Download {
            BufferId buffer_id;
            BufferCopy copy;
        };
        std::vector<Download> staged;
        std::vector<Download> mapped;
        u64 total_size_bytes = 0;
        u64 mapped_tick = 0;
        for (const auto& [device_addr, size] : ranges) {
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(device_addr, buffer_addr);
                const DAddr end = (std::min)(device_addr + size, buffer_addr + buffer.SizeBytes());
                if (begin >= end) {
                    return;
                }
                bool is_mapped = false;
                if constexpr (has_mapped) {
                    is_mapped = !buffer.Mapped().empty();
                    if (is_mapped) {
                        mapped_tick = (std::max)(mapped_tick, buffer.LastUsageTick());
                    }
                }
                memory_tracker.ForEachDownloadRangeAndClear(
                    begin, end - begin, [&](u64 device_addr_out, u64 range_size) {
                        gpu_modified_ranges.ForEachInRange(
                            device_addr_out, range_size, [&](DAddr start, DAddr range_end) {
                                const BufferCopy copy{
                                    .src_offset = start - buffer_addr,
                                    .dst_offset = total_size_bytes,
                                    .size = range_end - start,
                                };
                                if (is_mapped) {
                                    mapped.push_back({buffer_id, copy});
                                    return;
                                }
                                staged.push_back({buffer_id, copy});
                                total_size_bytes += Common::AlignUp(range_end - start, 64);
                            });
                        // Lands under `mutex` once the host GPU ran it: read downloads leave
                        // the range.
                        ClearReplacedDownload(device_addr_out, range_size);
                        gpu_modified_ranges.Subtract(device_addr_out, range_size);
                    });
            });
        }
        if (staged.empty() && mapped.empty()) {
            return;
        }
        // Once the host GPU ran the copies, inside a landing over their bytes: the CPU writes
        // collected meanwhile keep their bytes too (they were looked at before the wait).
        const auto land = [&](const Async_Buffer* staging) {
            DAddr landing_begin = ~DAddr{0};
            DAddr landing_end = 0;
            const auto cover = [&](const Download& download) {
                const DAddr address =
                    slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
                landing_begin = (std::min)(landing_begin, address);
                landing_end = (std::max)(landing_end, address + download.copy.size);
            };
            std::ranges::for_each(staged, cover);
            std::ranges::for_each(mapped, cover);
            const VideoCore::DownloadLanding landing{landing_begin, landing_end};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            if (staging != nullptr) {
                for (const Download& download : staged) {
                    WriteDownloadedMemory(
                        cpu_written,
                        slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset,
                        staging->mapped_span.data() + (download.copy.dst_offset - staging->offset),
                        download.copy.size);
                }
            }
            if constexpr (has_mapped) {
                for (const Download& download : mapped) {
                    Buffer& buffer = slot_buffers[download.buffer_id];
                    WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + download.copy.src_offset,
                                          buffer.Mapped().data() + download.copy.src_offset,
                                          download.copy.size);
                }
            }
        };
        if (staged.empty()) {
            if constexpr (has_mapped) {
                runtime.Wait(mapped_tick);
            }
            land(nullptr);
        } else {
            auto staging = runtime.DownloadStagingBuffer(total_size_bytes);
            for (Download& download : staged) {
                download.copy.dst_offset += staging.offset;
                Buffer& buffer = slot_buffers[download.buffer_id];
                buffer.MarkUsage(download.copy.src_offset, download.copy.size);
                runtime.CopyBuffer(staging.buffer, buffer, std::span{&download.copy, 1}, true);
            }
            runtime.Finish();
            land(&staging);
        }
    }
}

template <class P>
void BufferCache<P>::ClearDownload(DAddr device_addr, u64 size) {
    if (!read_downloads.empty()) [[unlikely]] {
        if (const std::optional<u64> waited = FinishReadDownloads(device_addr, size)) {
            VideoCore::AsyncReadback::CountClearLanding(*waited);
        }
    }
    ClearReplacedDownload(device_addr, size);
}

template <class P>
void BufferCache<P>::ClearReplacedDownload(DAddr device_addr, u64 size) {
    ClearFenceDownloads(device_addr, size);
    if (!read_downloads.empty()) [[unlikely]] {
        for (ReadDownload& download : read_downloads) {
            download.write_ranges.Subtract(device_addr, size);
            download.clean_ranges.Subtract(device_addr, size);
        }
        read_download_ranges.Subtract(device_addr, size);
    }
}

template <class P>
void BufferCache<P>::ClearFenceDownloads(DAddr device_addr, u64 size) {
    // FRITCHII_FENCE_COPY_UNLOCKED: the fencing thread may be copying a fence download here
    // without the lock (PopAsyncBuffersUnlocked), from ranges it read before this takes them
    // away. The caller writes newer bytes here next (a synchronous download, a GPU-side write
    // after PrepareRegionWrite), or a guest write is collected here (its bytes store after any
    // landing over them): the copy lands first, as when the pop held the lock throughout. One
    // acquire load while no such copy is under way.
    VideoCore::DownloadLanding::AwaitUnlocked(device_addr, size);
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    uncommitted_gpu_modified_ranges.Subtract(device_addr, size);
    for (auto& interval_set : committed_gpu_modified_ranges) {
        interval_set.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    const Common::WriteRing::SiteScope write_site{"DMA copy (buffer cache)", false};
    const std::optional<DAddr> cpu_src_address = gpu_memory->GpuToCpuAddress(src_address);
    const std::optional<DAddr> cpu_dest_address = gpu_memory->GpuToCpuAddress(dest_address);
    if (!cpu_src_address || !cpu_dest_address) {
        return false;
    }
    PromoteAhead();
    NoteAheadLevelCall(*cpu_src_address, amount, "DMACopy");
    NoteAheadLevelCall(*cpu_dest_address, amount, "DMACopy");
    const bool source_dirty = IsRegionRegistered(*cpu_src_address, amount);
    const bool dest_dirty = IsRegionRegistered(*cpu_dest_address, amount);
    if (!source_dirty && !dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dest_address, amount);
    MarkVertexBuffersStale(*cpu_dest_address, amount);
    // Guest memory gets the copy below (the scoped write at the end), which holds every byte the
    // source has in guest memory: command list pages in the destination read those from it, and
    // their copies are stale. The bytes copied from GPU-written
    // source ranges are noted as GPU writes after this.
    Nx2NoteListHostWrite(*cpu_dest_address, amount);

    BufferId buffer_a;
    BufferId buffer_b;
    do {
        channel_state->has_deleted_buffers = false;
        buffer_a = FindBuffer(*cpu_src_address, static_cast<u32>(amount), false);
        buffer_b = FindBuffer(*cpu_dest_address, static_cast<u32>(amount), false);
    } while (channel_state->has_deleted_buffers);
    auto& src_buffer = slot_buffers[buffer_a];
    auto& dest_buffer = slot_buffers[buffer_b];
    SynchronizeBuffer(src_buffer, *cpu_src_address, static_cast<u32>(amount));
    SynchronizeBuffer(dest_buffer, *cpu_dest_address, static_cast<u32>(amount));
    std::array copies{BufferCopy{
        .src_offset = src_buffer.Offset(*cpu_src_address),
        .dst_offset = dest_buffer.Offset(*cpu_dest_address),
        .size = amount,
    }};

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> tmp_intervals;
    auto mirror = [&](DAddr base_address, DAddr base_address_end) {
        const u64 size = base_address_end - base_address;
        const DAddr diff = base_address - *cpu_src_address;
        const DAddr new_base_address = *cpu_dest_address + diff;
        tmp_intervals.push_back({new_base_address, size});
        if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            Tegra::Nx2MarkGpuWritten(new_base_address, size);
        }
        QueueFenceDownload(new_base_address, size);
    };
    gpu_modified_ranges.ForEachInRange(*cpu_src_address, amount, mirror);
    // This subtraction in this order is important for overlapping copies.
    gpu_modified_ranges.Subtract(*cpu_dest_address, amount);
    const bool has_new_downloads = tmp_intervals.size() != 0;
    for (const auto& pair : tmp_intervals) {
        gpu_modified_ranges.Add(pair.first, pair.second);
        // FRITCHII_READ_RUNS: a GPU copy of GPU-written bytes is a write mark too (writer Other).
        if (ReadRunStats::g_enabled) [[unlikely]] {
            ReadRunStats::NoteWrite(pair.first, pair.second, ReadWriteStamp());
        }
        if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            Nx2NoteListWrite(buffer_b, pair.first, pair.second);
        }
    }
    const auto& copy = copies[0];
    src_buffer.MarkUsage(copy.src_offset, copy.size);
    dest_buffer.MarkUsage(copy.dst_offset, copy.size);
    runtime.CopyBuffer(dest_buffer, src_buffer, copies, true);
    if (has_new_downloads) {
        memory_tracker.MarkRegionAsGpuModified(*cpu_dest_address, amount);
        const bool should_sync = Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if (should_sync) {
            runtime.Finish();
        }
    }

    Tegra::Memory::DeviceGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::UnsafeReadWrite>
        tmp(device_memory, *cpu_src_address, amount, &tmp_buffer);
    tmp.SetAddressAndSize(*cpu_dest_address, amount);
    return true;
}

template <class P>
bool BufferCache<P>::DMAClear(GPUVAddr dst_address, u64 amount, u32 value) {
    const std::optional<DAddr> cpu_dst_address = gpu_memory->GpuToCpuAddress(dst_address);
    if (!cpu_dst_address) {
        return false;
    }
    // `amount` counts words: the check looked at a quarter of the range (fixed in a
    // ported change).
    const size_t size = amount * sizeof(u32);
    PromoteAhead();
    NoteAheadLevelCall(*cpu_dst_address, size, "DMAClear");
    const bool dest_dirty = IsRegionRegistered(*cpu_dst_address, size);
    if (!dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dst_address, size);
    gpu_modified_ranges.Subtract(*cpu_dst_address, size);
    // The caller writes the cleared words to guest memory after this, as an inline write: command
    // list pages read them from guest memory, and their copies are stale.
    Nx2NoteListHostWrite(*cpu_dst_address, size);

    const BufferId buffer = FindBuffer(*cpu_dst_address, static_cast<u32>(size), false);
    Buffer& dest_buffer = slot_buffers[buffer];
    const u32 offset = dest_buffer.Offset(*cpu_dst_address);
    runtime.ClearBuffer(dest_buffer, offset, size, value);
    dest_buffer.MarkUsage(offset, size);
    return true;
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                                 ObtainBufferSynchronize sync_info,
                                                                 ObtainBufferOperation post_op) {
    VideoCommon::MemoryStageLink::CheckBackCache("ObtainBuffer");
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return {&slot_buffers[NULL_BUFFER_ID], 0};
    }
    return ObtainCPUBuffer(*device_addr, size, sync_info, post_op);
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainCPUBuffer(
    DAddr device_addr, u32 size, ObtainBufferSynchronize sync_info, ObtainBufferOperation post_op) {
    const BufferId buffer_id = FindBuffer(device_addr, size, false);
    Buffer& buffer = slot_buffers[buffer_id];

    // synchronize op
    switch (sync_info) {
    case ObtainBufferSynchronize::FullSynchronize:
        SynchronizeBuffer(buffer, device_addr, size);
        break;
    default:
        break;
    }

    // Each caller records GPU work that reads or writes these bytes now (indirect dispatches, DMA
    // copies between images and buffers, query results, conditional rendering): an upload of them
    // later in this submission must run after it (CanReorderUpload).
    buffer.MarkUsage(buffer.Offset(device_addr), size);
    switch (post_op) {
    case ObtainBufferOperation::MarkAsWritten:
        MarkWrittenBuffer(buffer_id, device_addr, size);
        break;
    case ObtainBufferOperation::DiscardWrite: {
        PromoteAhead();
        NoteAheadLevelCall(device_addr, size, "ObtainCPUBuffer (DiscardWrite)");
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
        break;
    }
    default:
        break;
    }

    return {&buffer, buffer.Offset(device_addr)};
}

template <class P>
void BufferCache<P>::MarkUsed(DAddr device_addr, u64 size) {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || size == 0 || page >= page_table.size()) {
        return;
    }
    const BufferId buffer_id = page_table[page];
    if (!buffer_id) {
        return;
    }
    Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        buffer.MarkUsage(buffer.Offset(device_addr), size);
    }
}

template <class P>
bool BufferCache<P>::BindAddrMemo() {
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bind_addr_memo,
                                                      "FRITCHII_BIND_ADDR_MEMO");
    if (mode == 0) [[likely]] {
        return false;
    }
    if ((++bind_addr_calls & 4095) == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (bind_addr_since == std::chrono::steady_clock::time_point{}) {
            bind_addr_since = now;
            bind_addr_stats = {};
            bind_addr_disabled_skips = 0;
        } else if (now - bind_addr_since >= std::chrono::seconds{10}) {
            const double window = std::chrono::duration<double>(now - bind_addr_since).count();
            LOG_INFO(HW_GPU,
                     "Bind address memo (FRITCHII_BIND_ADDR_MEMO={}) {:.0f} s: hits {:.0f}/s, "
                     "translations {:.0f}/s ({:.0f}/s not kept: a map change overlapped), "
                     "disabled vertex streams not translated {:.0f}/s",
                     mode, window, static_cast<double>(bind_addr_stats.hits) / window,
                     static_cast<double>(bind_addr_stats.lookups) / window,
                     static_cast<double>(bind_addr_stats.unstable) / window,
                     static_cast<double>(bind_addr_disabled_skips) / window);
            bind_addr_since = now;
            bind_addr_stats = {};
            bind_addr_disabled_skips = 0;
        }
    }
    return true;
}

template <class P>
std::optional<DAddr> BufferCache<P>::TranslateBind(bool on, GPUVAddr gpu_addr,
                                                   Tegra::MemoryManager::TranslationMemo& memo) {
    if (!on) {
        return gpu_memory->GpuToCpuAddress(gpu_addr);
    }
    return gpu_memory->GpuToCpuAddressMemo(gpu_addr, memo, bind_addr_stats);
}

template <class P>
void BufferCache<P>::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                               u32 size) {
    VideoCommon::MemoryStageLink::CheckBackCache("BindGraphicsUniformBuffer");
    UniformBindMemo& memo = uniform_bind_memo[stage][index];
    Binding& current = channel_state->uniform_buffers[stage][index];
    // EDEN_NX2_EXP18 & 1: the same GPU range bound again while the mappings and the set of
    // buffers are unchanged translates to the same device range, and the buffer it resolved to
    // is still what FindBuffer returns: keep the binding with its buffer_id. The contents are
    // read at the draw as before. FindBuffer waits for writes with accurate GPU fences: keep that.
    const bool use_memo = (VideoCore::Nx2Debug::Experiment(18) & 1) != 0 &&
                          !Settings::IsGPUFenceBehaviorAccurate();
    if (use_memo && memo.epoch == binding_epoch && memo.gpu_addr == gpu_addr &&
        memo.size == size && memo.channel == static_cast<const void*>(channel_state) &&
        current.device_addr == memo.device_addr && current.size == size &&
        memo.as_id == gpu_memory->GetID() && memo.map_serial == gpu_memory->MapSerial()) {
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        return;
    }
    // Before the translation: a map in between makes the memo miss next time.
    const u64 map_serial = gpu_memory->MapSerial();
    const std::optional<DAddr> device_addr =
        TranslateBind(BindAddrMemo(), gpu_addr, uniform_addr_memo[stage][index]);
    if (!device_addr) {
        current = NULL_BINDING;
        memo.epoch = 0;
        return;
    }
    current = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = BufferId{},
    };
    if (use_memo) {
        memo = UniformBindMemo{
            .epoch = binding_epoch,
            .map_serial = map_serial,
            .as_id = gpu_memory->GetID(),
            .channel = channel_state,
            .gpu_addr = gpu_addr,
            .device_addr = *device_addr,
            .size = size,
        };
    } else {
        memo.epoch = 0;
    }
}

template <class P>
void BufferCache<P>::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    VideoCommon::MemoryStageLink::CheckBackCache("DisableGraphicsUniformBuffer");
    uniform_bind_memo[stage][index].epoch = 0;
    auto& current = channel_state->uniform_buffers[stage][index];
    if (current.device_addr == 0 && current.size == 0 && current.buffer_id == NULL_BUFFER_ID)
        [[likely]] {
        return;
    }
    current = NULL_BINDING;
}

template <class P>
void BufferCache<P>::UpdateGraphicsBuffers(bool is_indexed) {
    VideoCommon::MemoryStageLink::CheckBackCache("UpdateGraphicsBuffers");
    VideoCommon::OneFollower::EngineOwned("BufferCache::UpdateGraphicsBuffers");
    ++native_bank_draw;
    // FRITCHII_FILL_UPDATE_KEEP (fable_b 25.2 cut f): read once a draw. Not with accurate GPU
    // fences (FindBuffer waits for a buffer's writes there, as EXP18's memos leave it).
    update_keep = VideoCommon::FillCuts::Mode(VideoCommon::FillCuts::Cut::UpdateKeep);
    if (update_keep != 0) [[unlikely]] {
        if (Settings::IsGPUFenceBehaviorAccurate()) {
            update_keep = 0;
        }
        VideoCommon::FillCuts::MaybeReport();
    }
    SCOPE_EXIT {
        update_keep = 0;
    };
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateGraphicsBuffers(is_indexed);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::UpdateComputeBuffers() {
    VideoCommon::MemoryStageLink::CheckBackCache("UpdateComputeBuffers");
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateComputeBuffers();
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {
    VideoCommon::MemoryStageLink::CheckBackCache("BindHostGeometryBuffers");
    // FRITCHII_FILL_GEOM_KEEP (fable_b 25.2 cut e): read once a draw. No memo behind the walks
    // that may leave a range dirty (hot pages, CleanWords), as EDEN_NX2_STORAGE_SYNC_MEMO.
    geom_keep = VideoCommon::FillCuts::Mode(VideoCommon::FillCuts::Cut::GeomKeep);
    if (geom_keep != 0) [[unlikely]] {
        if (HotPagesOn() || Nx2Perf::On(Nx2Perf::Opt::CleanWords)) {
            geom_keep = 0;
        }
        VideoCommon::FillCuts::MaybeReport();
    }
    SCOPE_EXIT {
        geom_keep = 0;
    };
    if (is_indexed) {
        BindHostIndexBuffer();
    } else if constexpr (!HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const auto& draw_state = Inputs().DrawState();
        if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
            draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
    }
    BindHostVertexBuffers();
    BindHostTransformFeedbackBuffers();
    if (Inputs().DrawIndirect()) {
        BindHostDrawIndirectBuffers();
    }
}

template <class P>
void BufferCache<P>::BindHostStageBuffers(size_t stage) {
    VideoCommon::MemoryStageLink::CheckBackCache("BindHostStageBuffers");
    {
        const Nx2Profile::FineScope uniform_scope{Nx2Profile::Zone::UniformBuffers};
        BindHostGraphicsUniformBuffers(stage);
    }
    Nx2Profile::AddFine(
        Nx2Profile::FineCounter::StorageBinds,
        static_cast<u64>(std::popcount(channel_state->enabled_storage_buffers[stage])));
    if (channel_state->enabled_storage_buffers[stage] != 0) {
        const Nx2Profile::FineScope storage_scope{Nx2Profile::Zone::StageStorage};
        BindHostGraphicsStorageBuffers(stage);
    }
    if (channel_state->enabled_texture_buffers[stage] != 0) {
        const Nx2Profile::FineScope texture_scope{Nx2Profile::Zone::StageTexBuffers};
        BindHostGraphicsTextureBuffers(stage);
    }
}

template <class P>
void BufferCache<P>::BindHostComputeBuffers() {
    VideoCommon::MemoryStageLink::CheckBackCache("BindHostComputeBuffers");
    if (ReadRunStats::g_enabled) [[unlikely]] {
        ReadRunStats::g_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    // FRITCHII_EARLY_COMPUTE (dry): could this feeder dispatch run at once on a second queue?
    // Judged before its own binds; its write marks carry the verdict to the reads.
    if constexpr (!IS_OPENGL) {
        if (EarlyCompute::g_early && EarlyCompute::IsFeeder(compute_shader_hash)) [[unlikely]] {
            JudgeEarlyCompute();
        }
    }
    SCOPE_EXIT {
        EarlyCompute::t_verdict = EarlyCompute::Verdict::NotJudged;
    };
    BindHostComputeUniformBuffers();
    if (channel_state->enabled_compute_storage_buffers != 0) {
        BindHostComputeStorageBuffers();
    }
    if (channel_state->enabled_compute_texture_buffers != 0) {
        BindHostComputeTextureBuffers();
    }
    if (EarlyCompute::t_verdict != EarlyCompute::Verdict::NotJudged) [[unlikely]] {
        EarlyCompute::NoteJudged(EarlyCompute::t_verdict, any_buffer_uploaded);
        if (EarlyCompute::t_lane.judged) {
            NoteFeederLaneBound();
            EarlyCompute::t_lane = {};
        }
    }
    if (any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        any_buffer_uploaded = false;
    }
}

template <class P>
void BufferCache<P>::NoteFeederLaneBound() {
    // FRITCHII_FEEDER_LANE (dry): a bind that uploaded CPU data puts D in its own chunk (the
    // copy is recorded right before the dispatch); D_copy is what the lane would wait for if it
    // took the copy too.
    if constexpr (!IS_OPENGL) {
        EarlyCompute::Lane& lane = EarlyCompute::t_lane;
        if (any_buffer_uploaded && lane.kind != EarlyCompute::LaneKind::Wrapped) {
            lane.d = (std::max)(lane.d, runtime.CurrentTick());
            lane.kind = EarlyCompute::LaneKind::Upload;
            ReadRunStats::PatchLaneUpload(
                ReadRunStats::g_dispatches.load(std::memory_order_relaxed), compute_shader_hash,
                lane.d);
        }
        u32 storage_bindings = 0;
        u32 read_extents = 0;
        ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
            ++storage_bindings;
            read_extents += lane_read_end[index] != ~u64{0} ? 1 : 0;
        });
        EarlyCompute::NoteLaneJudged(compute_shader_hash, lane, any_buffer_uploaded,
                                     storage_bindings, read_extents);
    }
}

template <class P>
void BufferCache<P>::JudgeEarlyCompute() {
    if constexpr (!IS_OPENGL) {
        // Every bound range, and those the dispatch writes.
        boost::container::small_vector<std::pair<DAddr, DAddr>, 24> bound;
        boost::container::small_vector<std::pair<DAddr, DAddr>, 8> written;
        const auto add = [&](const Binding& binding, bool is_written) {
            if (binding.segment_count != 0) {
                for (u32 i = 0; i < binding.segment_count &&
                                binding.segment_first + i < compute_segments.size();
                     ++i) {
                    const MultiRangeSegment& segment = compute_segments[binding.segment_first + i];
                    bound.emplace_back(segment.device_addr, segment.device_addr + segment.size);
                    if (is_written) {
                        written.emplace_back(segment.device_addr,
                                             segment.device_addr + segment.size);
                    }
                }
                return;
            }
            if (binding.size == 0 || binding.device_addr == 0) {
                return;
            }
            bound.emplace_back(binding.device_addr, binding.device_addr + binding.size);
            if (is_written) {
                written.emplace_back(binding.device_addr, binding.device_addr + binding.size);
            }
        };
        ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
            add(channel_state->compute_uniform_buffers[index], false);
        });
        ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
            add(channel_state->compute_storage_buffers[index],
                ((channel_state->written_compute_storage_buffers >> index) & 1) != 0);
        });
        ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
            add(channel_state->compute_texture_buffers[index],
                ((channel_state->written_compute_texture_buffers >> index) & 1) != 0);
        });
        (void)runtime.ReadWriterTicks(0); // refreshes the host GPU's known tick
        const auto is_free = [this](u64 stamp) { return runtime.IsFree(stamp); };
        EarlyCompute::Verdict verdict = EarlyCompute::Verdict::Qualifies;
        if (ReadRunStats::AnyPendingMark(bound, is_free)) {
            verdict = EarlyCompute::Verdict::PendingWrite;
        } else {
            for (const auto& [begin, end] : written) {
                if (EarlyCompute::PendingUse(begin, end, is_free)) {
                    verdict = EarlyCompute::Verdict::PendingUse;
                    break;
                }
            }
        }
        EarlyCompute::t_verdict = verdict;
        if (EarlyCompute::g_lane) {
            JudgeFeederLane(is_free);
        }
    }
}

template <class P>
template <typename IsFree>
void BufferCache<P>::JudgeFeederLane(IsFree&& is_free) {
    if constexpr (!IS_OPENGL) {
        // FRITCHII_FEEDER_LANE (dry): D, the newest unrun main-queue work this feeder really
        // depends on. Its inputs: uniform and texture buffers whole; a native storage binding's
        // reads up to its proven read extent (else the whole binding) plus, when written, its
        // write mark (write after write). Its outputs (write after read): the write marks.
        boost::container::small_vector<std::pair<DAddr, DAddr>, 24> inputs;
        boost::container::small_vector<std::pair<DAddr, DAddr>, 8> outputs;
        const auto add_whole = [&](const Binding& binding, bool is_written) {
            if (binding.segment_count != 0) {
                for (u32 i = 0; i < binding.segment_count &&
                                binding.segment_first + i < compute_segments.size();
                     ++i) {
                    const MultiRangeSegment& segment = compute_segments[binding.segment_first + i];
                    inputs.emplace_back(segment.device_addr, segment.device_addr + segment.size);
                    if (is_written) {
                        outputs.emplace_back(segment.device_addr,
                                             segment.device_addr + segment.size);
                    }
                }
                return;
            }
            if (binding.size == 0 || binding.device_addr == 0) {
                return;
            }
            inputs.emplace_back(binding.device_addr, binding.device_addr + binding.size);
            if (is_written) {
                outputs.emplace_back(binding.device_addr, binding.device_addr + binding.size);
            }
        };
        ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
            add_whole(channel_state->compute_uniform_buffers[index], false);
        });
        ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
            const Binding& binding = channel_state->compute_storage_buffers[index];
            const bool is_written =
                ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
            if (binding.segment_count != 0 || binding.size == 0 || binding.device_addr == 0) {
                add_whole(binding, is_written);
                return;
            }
            const bool native =
                ((channel_state->native_compute_storage_buffers >> index) & 1) != 0;
            const u64 read_end = (std::min<u64>)(lane_read_end[index], binding.size);
            const u64 write_end =
                native ? (std::min<u64>)(channel_state->compute_storage_write_marks[index],
                                         binding.size)
                       : binding.size;
            if (read_end != 0) {
                inputs.emplace_back(binding.device_addr, binding.device_addr + read_end);
            }
            if (is_written && write_end != 0) {
                inputs.emplace_back(binding.device_addr, binding.device_addr + write_end);
                outputs.emplace_back(binding.device_addr, binding.device_addr + write_end);
            }
        });
        ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
            add_whole(channel_state->compute_texture_buffers[index],
                      ((channel_state->written_compute_texture_buffers >> index) & 1) != 0);
        });
        EarlyCompute::LaneDep dep = ReadRunStats::NewestPendingMark(inputs, is_free);
        dep.Merge(EarlyCompute::NewestPendingUse(outputs, is_free));
        EarlyCompute::Lane lane{
            .d = dep.d,
            .d_copy = dep.d_copy,
            .kind = dep.d != 0 ? dep.kind : EarlyCompute::LaneKind::None,
            .copy_kind = dep.d_copy != 0 ? dep.copy_kind : EarlyCompute::LaneKind::None,
            .judged = true,
        };
        if (dep.incomplete) {
            // Not every dependency is known: its own chunk
            lane.d = lane.d_copy = runtime.CurrentTick();
            lane.kind = lane.copy_kind = EarlyCompute::LaneKind::Wrapped;
        }
        EarlyCompute::t_lane = lane;
    }
}

template <class P>
void BufferCache<P>::SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                            const UniformBufferSizes* sizes) {
    VideoCommon::MemoryStageLink::CheckBackCache("SetUniformBuffersState");
    // FRITCHII_S4_COUNT: the counter follows its live switch, read once a draw here.
    VideoCommon::S4Count::Poll();
    // FRITCHII_BIND_SKIP: the same, for the uniform bind skip.
    BindSkipPoll();
    if (channel_state->enabled_uniform_buffer_masks == mask &&
        channel_state->uniform_buffer_sizes == sizes) [[likely]] {
        return;
    }
    if (channel_state->enabled_uniform_buffer_masks != mask) {
        channel_state->fast_bound_uniform_buffers.fill(0);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers.fill(~u32{0});
            channel_state->uniform_buffer_binding_sizes.fill({});
        }
        channel_state->enabled_uniform_buffer_masks = mask;
    }
    channel_state->uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::SetComputeUniformBufferState(u32 mask,
                                                  const ComputeUniformBufferSizes* sizes) {
    channel_state->enabled_compute_uniform_buffer_mask = mask;
    channel_state->compute_uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::UnbindGraphicsStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) {
        return;
    }
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_graphics_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_storage_buffers[stage]));
        }
    }
    channel_state->enabled_storage_buffers[stage] = 0;
    channel_state->written_storage_buffers[stage] = 0;
}

template <class P>
bool BufferCache<P>::BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                               u32 cbuf_offset, bool is_written,
                                               u32 indirect_offset) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) {
        LOG_ERROR(HW_GPU, "Invalid graphics storage buffer binding: stage {}, index {}",
                  stage, ssbo_index);
        return false;
    }
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_graphics_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping graphics storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return false;
            }
        }
    }
    channel_state->enabled_storage_buffers[stage] |= u64{1} << ssbo_index;
    channel_state->written_storage_buffers[stage] |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }

    GPUVAddr ssbo_addr = Inputs().StageConstBuffer(stage, cbuf_index).address + cbuf_offset;
    GPUVAddr holder{};
    if (indirect_offset != ~0U) {
        // The pointer is an entry of a table the constant buffer points to.
        holder = Inputs().ReadGameU64(ssbo_addr);
        MemStageStats::Read(MemStageStats::Kind::StorageAddress, sizeof(u64));
        MemReadLog::Value(MemReadLog::Kind::TableAddress, gpu_memory, ssbo_addr, holder);
        if (holder == 0) {
            channel_state->storage_buffers[stage][ssbo_index] = NULL_BINDING;
            return false;
        }
        ssbo_addr = holder + indirect_offset;
    }
    // EDEN_NX2_GRAPHICS_SSBO_BARE=0 takes the word after a native graphics pointer as its size
    // again: much smaller bindings to synchronize per draw, for A/B runs.
    static const bool bare_pointers = [] {
        const char* const value = std::getenv("EDEN_NX2_GRAPHICS_SSBO_BARE");
        return value == nullptr || value[0] != '0';
    }();
    channel_state->storage_buffers[stage][ssbo_index] =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written,
                             (bare_pointers && Inputs().NativeTuring()) ||
                                 indirect_offset != ~0U,
                             indirect_offset != ~0U);
    if (indirect_offset != ~0U) {
        TraceTablePointer(false, stage, ssbo_index, cbuf_index, cbuf_offset, indirect_offset, holder,
                          channel_state->storage_buffers[stage][ssbo_index]);
    }
    return (channel_state->storage_buffers[stage][ssbo_index].buffer_id != NULL_BUFFER_ID);
}

template <class P>
void BufferCache<P>::UnbindGraphicsTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) {
        return;
    }
    channel_state->enabled_texture_buffers[stage] = 0;
    channel_state->written_texture_buffers[stage] = 0;
    channel_state->image_texture_buffers[stage] = 0;
}

template <class P>
void BufferCache<P>::BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr,
                                               u32 size, PixelFormat format, bool is_written,
                                               bool is_image) {
    channel_state->enabled_texture_buffers[stage] |= 1U << tbo_index;
    channel_state->written_texture_buffers[stage] |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_texture_buffers[stage] |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->texture_buffers[stage][tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::UnbindComputeStorageBuffers() {
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_compute_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_compute_storage_buffers));
        }
    }
    channel_state->enabled_compute_storage_buffers = 0;
    channel_state->written_compute_storage_buffers = 0;
    for (ExtentProbeRequest& request : extent_probe_requests) {
        request.active = false;
    }
    cbuf_dry_proven.fill(0);
    footprint_dry.fill({});
    lane_read_end.fill(~u64{0});
    channel_state->native_compute_storage_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                              bool is_written, u32 indirect_offset,
                                              const Shader::StorageWriteExtent& write_extent,
                                              const Shader::StorageWriteExtra* extra,
                                              const Shader::StorageWriteExtent* read_extent) {
    if (ssbo_index >= channel_state->compute_storage_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Storage buffer index {} exceeds maximum storage buffer count",
                  ssbo_index);
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_compute_storage_buffers >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_compute_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping compute storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return;
            }
        }
    }
    channel_state->enabled_compute_storage_buffers |= u64{1} << ssbo_index;
    channel_state->written_compute_storage_buffers |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_compute_storage_buffers;
        }
    }

    const auto& launch_desc = Inputs().LaunchDescription();
    if (((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) == 0) {
        LOG_WARNING(HW_GPU, "Skipped binding SSBO: cbuf index {} is not enabled", cbuf_index);
        return;
    }
    ASSERT(((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) != 0);

    const auto& cbufs = launch_desc.const_buffer_config;
    GPUVAddr ssbo_addr = cbufs[cbuf_index].Address() + cbuf_offset;
    GPUVAddr holder{};
    const bool table_pointer = indirect_offset != ~0U;
    if (table_pointer) {
        holder = Inputs().ReadGameU64(ssbo_addr);
        MemStageStats::Read(MemStageStats::Kind::CStorageAddress, sizeof(u64));
        MemReadLog::Value(MemReadLog::Kind::TableAddress, gpu_memory, ssbo_addr, holder);
        ssbo_addr = holder + indirect_offset;
    }
    const bool native_pointer = Inputs().ComputeNativeMode();
    channel_state->native_compute_storage_buffers =
        (channel_state->native_compute_storage_buffers & ~(u64{1} << ssbo_index)) |
        (u64{native_pointer ? 1U : 0U} << ssbo_index);
    const Binding binding =
        table_pointer && holder == 0
            ? NULL_BINDING
            : StorageBufferBinding(ssbo_addr, cbuf_index, is_written, native_pointer,
                                   table_pointer, true);
    channel_state->compute_storage_buffers[ssbo_index] = binding;
    if (table_pointer && holder != 0) {
        TraceTablePointer(true, 0, ssbo_index, cbuf_index, cbuf_offset, indirect_offset, holder,
                          binding);
    }

    // How much of the binding the dispatch marks GPU-written. A native pointer has no size, so the
    // binding is up to 16 MiB past it (FRITCHII_BARE_SSBO_MIB), and marking all of it defers the
    // CPU's later writes to those pages until the next fence and has fences copy the over-marked
    // range back.
    u64 mark = binding.size;
    // EDEN_NX2_EXP1=1: mark the whole binding as before (the original A/B).
    if (native_pointer && is_written && binding.size != 0 &&
        VideoCore::Nx2Debug::Experiment(1) == 0) {
        switch (Settings::values.native_compute_write_mark.GetValue()) {
        case Settings::NativeComputeWriteMark::Auto: {
            const auto& qmd = launch_desc;
            const std::array<u32, 3> block{qmd.block_dim_x.Value(), qmd.block_dim_y.Value(),
                                           qmd.block_dim_z.Value()};
            std::optional<std::array<u32, 3>> grid;
            if (!Inputs().IndirectComputeAddress()) {
                grid = std::array<u32, 3>{qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                                          qmd.grid_dim_z.Value()};
            }
            std::optional<u64> extent =
                Shader::ResolveStorageWriteExtent(write_extent, block, grid);
            static constexpr Shader::StorageWriteExtra NoExtra{};
            // The census names the analysis' reason only where the extent is unknown.
            const Shader::ExtentUnknownWhy write_unknown =
                write_extent.bounded
                    ? Shader::ExtentUnknownWhy{.kind = Shader::ExtentUnknown::Known}
                    : (extra != nullptr ? *extra : NoExtra).write_unknown;
            const Shader::StorageCbufExtent& cbuf_extent =
                (extra != nullptr ? *extra : NoExtra).cbuf_extent;
            // FRITCHII_EXTENT_CENSUS / FRITCHII_EXTENT_CBUF (live, default 0): off, one load each.
            const int census = VideoCore::Nx2Debug::SwitchValue(
                VideoCore::Nx2Debug::g_extent_census, "FRITCHII_EXTENT_CENSUS", 0);
            const int cbuf_switch = VideoCore::Nx2Debug::SwitchValue(
                VideoCore::Nx2Debug::g_extent_cbuf, "FRITCHII_EXTENT_CBUF", 0);
            // FRITCHII_EXTENT_SHARED (live, default 0): the same for a cbuf extent proven by
            // following offsets through the program's private shared memory slots (via_shared);
            // it, not FRITCHII_EXTENT_CBUF, decides about those.
            const int shared_switch = VideoCore::Nx2Debug::SwitchValue(
                VideoCore::Nx2Debug::g_extent_shared, "FRITCHII_EXTENT_SHARED", 0);
            // FRITCHII_EXTENT_GUARD (live, default 0): the same for a cbuf extent proven with the
            // branch conditions on the stores' paths (via_guard).
            const int guard_switch = VideoCore::Nx2Debug::SwitchValue(
                VideoCore::Nx2Debug::g_extent_guard, "FRITCHII_EXTENT_GUARD", 0);
            const int cbuf_mode = cbuf_extent.via_guard    ? guard_switch
                                  : cbuf_extent.via_shared ? shared_switch
                                                           : cbuf_switch;
            FullMarkWhy full_why = FullMarkWhy::Analysis;
            if (!extent && write_extent.bounded) {
                full_why = grid ? FullMarkWhy::GuardOrDims : FullMarkWhy::IndirectGrid;
            }
            // FRITCHII_EXTENT_CBUF: a store extent with constant buffer words, its words read
            // now like the pointer word. =1 marks by it (through the ExtentTrust checks below),
            // =3 only counts what it would free.
            CbufOutcome cbuf_outcome = CbufOutcome::None;
            std::optional<u64> cbuf_bytes;
            if (!extent && (cbuf_mode == 1 || cbuf_mode == 3) && cbuf_extent.ids.bounded)
                [[unlikely]] {
                cbuf_bytes = ResolveCbufExtent(cbuf_extent, block, grid, cbuf_outcome);
            }
            // Only with the ExtentTrust checks (they need Vulkan and EXP45 off).
            const bool cbuf_marks = !IS_OPENGL && cbuf_mode == 1 && cbuf_bytes.has_value() &&
                                    VideoCore::Nx2Debug::Experiment(45) == 0;
            if (cbuf_marks) {
                extent = cbuf_bytes;
            }
            // The pointer word, read once (the extent counts from the pointer, the binding from
            // the pointer aligned down).
            std::optional<GPUVAddr> pointer_word;
            const auto read_pointer = [&]() -> GPUVAddr {
                if (!pointer_word) {
                    pointer_word = Inputs().ReadGameU64(ssbo_addr);
                    MemReadLog::Value(MemReadLog::Kind::ComputePointer, gpu_memory, ssbo_addr,
                                      *pointer_word);
                }
                return *pointer_word;
            };
            u64 cbuf_proven = 0;
            if (extent) {
                const GPUVAddr pointer = read_pointer();
                const u64 pad = pointer >= binding.gpu_addr ? pointer - binding.gpu_addr : 0;
                const u64 proven = std::min<u64>(binding.size, pad + *extent);
                mark = proven;
                if (cbuf_marks) {
                    cbuf_proven = proven;
                }
                if (proven >= binding.size) {
                    full_why = FullMarkWhy::WholeBinding;
                }
                // Not yet checked at these dims, or caught writing past it: mark in full (see
                // ExtentTrust), and check this dispatch.
                if (!IS_OPENGL && proven < binding.size &&
                    VideoCore::Nx2Debug::Experiment(45) == 0) {
                    const std::array<u32, 3> dims_grid =
                        grid.value_or(std::array<u32, 3>{~0U, ~0U, ~0U});
                    const ExtentTrust& trust = extent_trust[compute_shader_hash];
                    bool bigger = false;
                    for (size_t i = 0; i < 3; ++i) {
                        bigger |= block[i] > trust.block[i] || dims_grid[i] > trust.grid[i];
                    }
                    if (trust.distrusted || bigger) {
                        mark = binding.size;
                        ++extent_full_marks;
                        full_why =
                            trust.distrusted ? FullMarkWhy::Distrusted : FullMarkWhy::Checking;
                        if (!trust.distrusted && !trust.probe_pending) {
                            extent_probe_requests[ssbo_index] = ExtentProbeRequest{
                                .active = true,
                                .proven = proven,
                                .pointer = pointer,
                                .pad = pad,
                                .block = block,
                                .grid = dims_grid,
                            };
                        }
                    }
                }
            } else if (cbuf_mode == 3 && cbuf_bytes) [[unlikely]] {
                // Dry: the full mark stays; the read ring learns where the proof would end it.
                const GPUVAddr pointer = read_pointer();
                const u64 pad = pointer >= binding.gpu_addr ? pointer - binding.gpu_addr : 0;
                cbuf_proven = std::min<u64>(binding.size, pad + *cbuf_bytes);
                if (cbuf_proven < binding.size) {
                    cbuf_dry_proven[ssbo_index] = static_cast<u32>(cbuf_proven);
                }
            }
            // FRITCHII_STORE_FOOTPRINT=3 (dry): the binding's store footprint for its write marks,
            // counted from the pointer (a binding of several device ranges has none).
            // FRITCHII_STORE_FOOTPRINT armed (any value at boot): the live shadow follows
            // gpu_modified_ranges from the first such bind on, and the marks carry the same
            // footprint (BindHostComputeStorageBuffers).
            if (ReadRunStats::g_store_footprint) [[unlikely]] {
                if (!footprint_following) {
                    footprint_shadow.Start(gpu_modified_ranges);
                    gpu_modified_ranges.SetFollower(&footprint_shadow);
                    footprint_following = true;
                }
                NoteBindFootprint(ssbo_index, binding, write_extent, extra, block, grid,
                                  read_pointer());
            }
            if (census != 0 || cbuf_switch != 0 || shared_switch != 0 || guard_switch != 0)
                [[unlikely]] {
                // Every native written binding (the census ranks them by read-back waits too)
                NoteExtentCensus(static_cast<u32>(ssbo_index), binding, cbuf_index, cbuf_offset,
                                 mark, full_why, write_unknown, cbuf_extent, cbuf_outcome,
                                 cbuf_proven, cbuf_marks && mark < binding.size,
                                 !write_extent.bounded, extra != nullptr,
                                 (extra != nullptr ? *extra : NoExtra).shared_why,
                                 (extra != nullptr ? *extra : NoExtra).guard_why);
                if (census != 0 &&
                    !ReadRunStats::g_wait_census.load(std::memory_order_relaxed)) {
                    ReadRunStats::g_wait_census.store(true, std::memory_order_relaxed);
                }
            }
            break;
        }
        case Settings::NativeComputeWriteMark::Full:
            break;
        case Settings::NativeComputeWriteMark::Cap64K:
            mark = std::min<u64>(binding.size, 64_KiB);
            break;
        }
    }
    channel_state->compute_storage_write_marks[ssbo_index] = static_cast<u32>(mark);
    // FRITCHII_FEEDER_LANE (dry): how far a native binding's loads and atomics reach, for the
    // feeder's D (JudgeEarlyCompute). Unknown (the whole binding) without a proven read extent.
    if constexpr (!IS_OPENGL) {
        if (EarlyCompute::g_lane && native_pointer && read_extent != nullptr &&
            read_extent->bounded && binding.size != 0 && binding.segment_count == 0)
            [[unlikely]] {
            const auto& qmd = launch_desc;
            const std::array<u32, 3> block{qmd.block_dim_x.Value(), qmd.block_dim_y.Value(),
                                           qmd.block_dim_z.Value()};
            std::optional<std::array<u32, 3>> grid;
            if (!Inputs().IndirectComputeAddress()) {
                grid = std::array<u32, 3>{qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                                          qmd.grid_dim_z.Value()};
            }
            if (const std::optional<u64> bytes =
                    Shader::ResolveStorageWriteExtent(*read_extent, block, grid)) {
                const GPUVAddr pointer = Inputs().ReadGameU64(ssbo_addr);
                const u64 pad = pointer >= binding.gpu_addr ? pointer - binding.gpu_addr : 0;
                lane_read_end[ssbo_index] = std::min<u64>(binding.size, pad + *bytes);
            }
        }
    }
}

template <class P>
std::optional<u64> BufferCache<P>::ResolveCbufExtent(const Shader::StorageCbufExtent& extent,
                                                     const std::array<u32, 3>& block,
                                                     const std::optional<std::array<u32, 3>>& grid,
                                                     CbufOutcome& outcome) {
    // FRITCHII_EXTENT_CBUF: each term's word as the dispatch's uniform upload reads it (game
    // memory at bind). A word on a GPU-written page may be stale in game memory: full mark.
    const auto& launch_desc = Inputs().LaunchDescription();
    const auto& cbufs = launch_desc.const_buffer_config;
    const auto read_word = [&](const Shader::StorageExtentTerm& term, u32& value) {
        if (term.cbuf_index >= cbufs.size() ||
            ((launch_desc.const_buffer_enable_mask >> term.cbuf_index) & 1) == 0) {
            outcome = CbufOutcome::Unresolved;
            return false;
        }
        const GPUVAddr address = cbufs[term.cbuf_index].Address() + term.cbuf_offset;
        const std::optional<DAddr> device = gpu_memory->GpuToCpuAddress(address);
        if (!device) {
            outcome = CbufOutcome::Unresolved;
            return false;
        }
        const DAddr page = Common::AlignDown(*device, Core::DEVICE_PAGESIZE);
        if (GpuWrittenHere(page, Core::DEVICE_PAGESIZE)) {
            outcome = CbufOutcome::Fallback;
            return false;
        }
        value = Inputs().ReadGameU32(address);
        return true;
    };
    std::array<u32, Shader::MaxExtentTerms> values{};
    std::array<u32, Shader::MaxExtentTerms> guard_values{};
    for (size_t i = 0; i < extent.num_terms && i < Shader::MaxExtentTerms; ++i) {
        if (!read_word(extent.terms[i], values[i])) {
            return std::nullopt;
        }
    }
    for (size_t i = 0; i < extent.num_guard_terms && i < Shader::MaxExtentTerms; ++i) {
        if (!read_word(extent.guard_terms[i], guard_values[i])) {
            return std::nullopt;
        }
    }
    // FRITCHII_EXTENT_GUARD: the words a clamp needs at least a value in, read the same way
    std::array<u32, Shader::MaxExtentFloors> floor_values{};
    for (size_t i = 0; i < extent.num_floors && i < Shader::MaxExtentFloors; ++i) {
        const Shader::StorageExtentTerm word{.cbuf_index = extent.floors[i].cbuf_index,
                                             .cbuf_offset = extent.floors[i].cbuf_offset};
        if (!read_word(word, floor_values[i])) {
            return std::nullopt;
        }
    }
    const std::optional<u64> bytes = Shader::ResolveStorageCbufExtent(
        extent, block, grid, values, guard_values, floor_values);
    outcome = bytes ? CbufOutcome::Resolved : CbufOutcome::Unresolved;
    return bytes;
}

template <class P>
void BufferCache<P>::NoteBindFootprint(size_t ssbo_index, const Binding& binding,
                                       const Shader::StorageWriteExtent& write_extent,
                                       const Shader::StorageWriteExtra* extra,
                                       const std::array<u32, 3>& block,
                                       const std::optional<std::array<u32, 3>>& grid,
                                       GPUVAddr pointer) {
    namespace Live = StoreFootprintLive;
    Live::Stats& stats = Live::GetStats();
    if (binding.segment_count != 0 || binding.size == 0) {
        Live::Bump(stats.bind_none);
        return;
    }
    static constexpr Shader::StorageStoreFootprint NoFootprint{};
    const Shader::StorageStoreFootprint& recorded =
        extra != nullptr ? extra->footprint : NoFootprint;
    const u64 pad = pointer >= binding.gpu_addr ? pointer - binding.gpu_addr : 0;
    const u32 binding_index = static_cast<u32>(ssbo_index);
    u32 id = 0;
    if (!recorded.Valid() || recorded.covers_all || recorded.PowerOfTwo()) {
        // A power-of-two modulus holds whatever the 32-bit offsets do; none claims nothing
        id = ReadRunStats::InternFootprint(compute_shader_hash, binding_index, recorded);
        Live::Bump(recorded.Valid() && !recorded.covers_all ? stats.bind_full : stats.bind_none);
    } else {
        // Another modulus holds only while pointer + offset doesn't wrap 32 bits. Offsets made
        // from the ids alone are below the plain store extent at these dims: then no store that
        // lands in the binding wrapped (one below the pointer by more than the pad lands past
        // 4 GiB - 16 MiB, outside it).
        const std::optional<u64> plain = Shader::ResolveStorageWriteExtent(write_extent, block, grid);
        if (recorded.ids_only && plain && pad + *plain <= (u64{1} << 32)) {
            id = ReadRunStats::InternFootprint(compute_shader_hash, binding_index, recorded);
            Live::Bump(stats.bind_full);
        } else {
            id = ReadRunStats::InternFootprint(compute_shader_hash, binding_index | 0x80000000U,
                                               Shader::ReduceStoreFootprint(recorded));
            Live::Bump(stats.bind_reduced);
        }
    }
    footprint_dry[ssbo_index] = ReadRunStats::FootprintRef{
        .id = id,
        .base = binding.device_addr + pad,
    };
}

template <class P>
bool BufferCache<P>::FootprintAnswersRead(DAddr device_addr, u64 size, DAddr page_begin,
                                          DAddr page_end, int live) {
    namespace Live = StoreFootprintLive;
    Live::Stats& stats = Live::GetStats();
    Live::Bump(stats.asked);
    switch (footprint_shadow.Check(device_addr, size, gpu_modified_ranges)) {
    case Live::Answer::NotModified:
        Live::Bump(stats.not_modified);
        return false;
    case Live::Answer::Unclaimed:
        Live::Bump(stats.unclaimed);
        return false;
    case Live::Answer::Inside:
        Live::Bump(stats.inside);
        return false;
    case Live::Answer::Outside:
        break;
    }
    if (live == 2 && ++footprint_skip_count % Live::VerifyEvery == 0) {
        // Waits as before; its bytes now are compared with the landed ones (GPU::OnCPURead)
        std::array<u8, 64> bytes{};
        const u32 compared = static_cast<u32>((std::min<u64>)(size, bytes.size()));
        VideoCommon::OneFollower::BeforeEngineRead(device_addr, compared);
        device_memory.ReadBlockUnsafe(device_addr, bytes.data(), compared);
        Live::AddVerify(device_addr, compared, bytes.data());
        return false;
    }
    // The core took this read's page traps (it keeps the pages readable until a GPU write arms
    // them again), but the pages still hold GPU-modified bytes this read did not need: arm them
    // now so the next read of one traps and is asked here as before.
    read_trapped_ranges.Subtract(page_begin, page_end - page_begin);
    ArmReadTraps(page_begin, page_end - page_begin);
    Live::Bump(stats.rearmed, (page_end - page_begin) >> Core::DEVICE_PAGEBITS);
    Live::Bump(stats.skipped);
    return true;
}

template <class P>
void BufferCache<P>::FootprintCheckLanding(const Common::RangeSet<DAddr>* cpu_written,
                                           DAddr device_addr, const u8* data, u64 size) {
    namespace Live = StoreFootprintLive;
    if (!footprint_shadow.AnyClaimed(device_addr, size)) {
        return;
    }
    Live::Stats& stats = Live::GetStats();
    const u64 checked = (std::min<u64>)(size, Live::LandingCheckBytes);
    std::array<u8, 4096> guest{};
    u64 differed = 0;
    u64 first_differed = 0;
    for (u64 offset = 0; offset < checked; offset += guest.size()) {
        const u64 chunk = (std::min<u64>)(guest.size(), checked - offset);
        device_memory.ReadBlockUnsafe(device_addr + offset, guest.data(), chunk);
        if (std::memcmp(guest.data(), data + offset, chunk) == 0) {
            continue;
        }
        for (u64 i = 0; i < chunk; ++i) {
            if (guest[i] == data[offset + i]) {
                continue;
            }
            const DAddr address = device_addr + offset + i;
            if (cpu_written != nullptr && cpu_written->Intersects(address, 1)) {
                continue; // newer guest bytes: the landing leaves them
            }
            if (footprint_shadow.ClaimedOutside(address)) {
                if (differed == 0) {
                    first_differed = address;
                }
                ++differed;
            }
        }
    }
    Live::Bump(stats.landings_checked);
    Live::Bump(stats.landing_bytes_checked, checked);
    if (differed != 0) {
        const u64 count = stats.landings_differed.fetch_add(1, std::memory_order_relaxed) + 1;
        Live::Bump(stats.landing_bytes_differed, differed);
        if (count <= 16) {
            LOG_ERROR(HW_GPU,
                      "Store footprint (FRITCHII_STORE_FOOTPRINT=2): a download landing at {:#x} "
                      "+{:#x} would overwrite {} guest bytes outside every footprint over them, "
                      "first at {:#x} (must be 0)",
                      device_addr, size, differed, first_differed);
        }
    }
}

template <class P>
void BufferCache<P>::NoteExtentCensus(u32 ssbo_index, const Binding& binding, u32 cbuf_index,
                                      u32 cbuf_offset, u64 mark, FullMarkWhy why,
                                      const Shader::ExtentUnknownWhy& write_unknown,
                                      const Shader::StorageCbufExtent& cbuf_extent,
                                      CbufOutcome cbuf_outcome, u64 cbuf_proven,
                                      bool cbuf_marked, bool extent_unknown, bool has_extra,
                                      const Shader::ExtentUnknownWhy& shared_why,
                                      const Shader::ExtentUnknownWhy& guard_why) {
    if (extent_census.empty() && extent_census_frames == 0) {
        extent_census_start = std::chrono::steady_clock::now();
    }
    ExtentCensusEntry& entry = extent_census[{compute_shader_hash, ssbo_index}];
    entry.analysis = write_unknown;
    entry.shared_why = shared_why;
    entry.guard_why = guard_why;
    entry.pointer_cbuf_index = cbuf_index;
    entry.pointer_cbuf_offset = cbuf_offset;
    entry.binding_bytes = binding.size;
    entry.cbuf_extent = cbuf_extent;
    ++entry.dispatches;
    entry.marked_bytes += (std::min<u64>)(mark, binding.size);
    if (extent_unknown) {
        ++entry.unknown_dispatches;
        entry.no_extra_dispatches += has_extra ? 0 : 1;
    }
    if (mark >= binding.size) {
        ++entry.full_dispatches;
        entry.full_bytes += binding.size;
        ++entry.whys[static_cast<size_t>(why)];
    }
    switch (cbuf_outcome) {
    case CbufOutcome::Resolved:
        ++entry.cbuf_resolved;
        entry.cbuf_proven_bytes += cbuf_proven;
        entry.cbuf_binding_bytes += binding.size;
        break;
    case CbufOutcome::Fallback:
        ++entry.cbuf_fallback;
        break;
    case CbufOutcome::Unresolved:
        ++entry.cbuf_unresolved;
        break;
    case CbufOutcome::None:
        break;
    }
    entry.cbuf_marked += cbuf_marked ? 1 : 0;
}

template <class P>
void BufferCache<P>::LogExtentCensus() {
    // TickFrame, every frame while FRITCHII_EXTENT_CENSUS, FRITCHII_EXTENT_CBUF,
    // FRITCHII_EXTENT_SHARED or FRITCHII_EXTENT_GUARD is on: the lines every 10 s.
    if (extent_census.empty()) {
        extent_census_frames = 0;
        return;
    }
    ++extent_census_frames;
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - extent_census_start).count();
    if (seconds < 10.0) {
        return;
    }
    const int census = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_extent_census,
                                                        "FRITCHII_EXTENT_CENSUS", 0);
    const int cbuf_mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_extent_cbuf,
                                                           "FRITCHII_EXTENT_CBUF", 0);
    const int shared_mode = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_extent_shared, "FRITCHII_EXTENT_SHARED", 0);
    const int guard_mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_extent_guard,
                                                            "FRITCHII_EXTENT_GUARD", 0);
    // Which switch decides about a cbuf extent: 0 FRITCHII_EXTENT_CBUF, 1 _SHARED (via_shared),
    // 2 _GUARD (via_guard)
    const auto proof_kind = [](const Shader::StorageCbufExtent& extent) -> size_t {
        return extent.via_guard ? 2 : extent.via_shared ? 1 : 0;
    };
    static constexpr std::array<const char*, 3> proof_word{"cbuf", "shared", "guard"};
    static constexpr std::array<const char*, 3> proof_switch{
        "FRITCHII_EXTENT_CBUF", "FRITCHII_EXTENT_SHARED", "FRITCHII_EXTENT_GUARD"};
    const std::array<int, 3> proof_mode{cbuf_mode, shared_mode, guard_mode};
    const double frames = static_cast<double>((std::max<u64>)(extent_census_frames, 1));
    const auto opcode_name = [](u16 opcode) {
        return std::string{Shader::IR::NameOf(static_cast<Shader::IR::Opcode>(opcode))};
    };
    const auto why_text = [&](const Shader::ExtentUnknownWhy& why) {
        std::string text{Shader::ExtentUnknownName(why.kind)};
        switch (why.kind) {
        case Shader::ExtentUnknown::OtherCbufWord:
            text += fmt::format(" c{}[{:#x}]", why.cbuf_index, why.cbuf_offset);
            break;
        case Shader::ExtentUnknown::DynamicCbuf:
        case Shader::ExtentUnknown::LoadedFromMemory:
        case Shader::ExtentUnknown::AtomicResult:
        case Shader::ExtentUnknown::SignedOr64Bit:
        case Shader::ExtentUnknown::VariableProduct:
        case Shader::ExtentUnknown::OtherOp:
        case Shader::ExtentUnknown::LoopCarried:
        case Shader::ExtentUnknown::AddressNotTracked:
        case Shader::ExtentUnknown::NoPointerTerm:
        case Shader::ExtentUnknown::TooLarge:
        case Shader::ExtentUnknown::GuardFailed:
        case Shader::ExtentUnknown::SharedNotPrivate:
        case Shader::ExtentUnknown::SharedPartialWrite:
        case Shader::ExtentUnknown::SharedNotDominated:
        case Shader::ExtentUnknown::GuardNotDominating:
        case Shader::ExtentUnknown::GuardSigned:
        case Shader::ExtentUnknown::GuardWraps:
        case Shader::ExtentUnknown::GuardLimitUnknown:
        case Shader::ExtentUnknown::GuardInLoop:
            if (why.opcode != 0) {
                text += fmt::format(" ({})", opcode_name(why.opcode));
            }
            break;
        default:
            break;
        }
        return text;
    };
    const auto terms_text = [](const Shader::StorageCbufExtent& extent) -> std::string {
        if (!extent.ids.bounded) {
            return "none";
        }
        std::string text;
        for (size_t i = 0; i < extent.num_terms; ++i) {
            text += fmt::format("{}c{}[{:#x}] x{:#x}", i == 0 ? "" : " + ",
                                extent.terms[i].cbuf_index, extent.terms[i].cbuf_offset,
                                extent.terms[i].coef);
        }
        if (extent.num_guard_terms != 0) {
            text += " (16-bit guard on";
            for (size_t i = 0; i < extent.num_guard_terms; ++i) {
                text += fmt::format(" c{}[{:#x}] x{}", extent.guard_terms[i].cbuf_index,
                                    extent.guard_terms[i].cbuf_offset,
                                    extent.guard_terms[i].coef);
            }
            text += ")";
        }
        if (extent.num_floors != 0) {
            // FRITCHII_EXTENT_GUARD: the clamps' words, each at least its min at bind
            text += text.empty() ? "(needs" : " (needs";
            for (size_t i = 0; i < extent.num_floors; ++i) {
                text += fmt::format("{} c{}[{:#x}] >= {:#x}", i == 0 ? "" : ",",
                                    extent.floors[i].cbuf_index, extent.floors[i].cbuf_offset,
                                    extent.floors[i].min);
            }
            text += ")";
        }
        return text.empty() ? std::string{"no words (proven without them)"} : text;
    };
    // FRITCHII_EXTENT_GUARD: the extent proven with the stores' path facts, or why not
    const auto guard_text = [&](const ExtentCensusEntry& entry) -> std::string {
        if (entry.cbuf_extent.via_guard && entry.cbuf_extent.ids.bounded) {
            const Shader::StorageWriteExtent& ids = entry.cbuf_extent.ids;
            return fmt::format("end {:#x} + local {:#x}/{:#x}/{:#x} + group {:#x}/{:#x}/{:#x}{} + "
                               "words: {}",
                               ids.const_end, ids.local_stride[0], ids.local_stride[1],
                               ids.local_stride[2], ids.group_stride[0], ids.group_stride[1],
                               ids.group_stride[2], ids.guarded ? " (16-bit guard)" : "",
                               terms_text(entry.cbuf_extent));
        }
        if (entry.guard_why.kind == Shader::ExtentUnknown::Unrecorded) {
            return "not tried";
        }
        return "refused: " + why_text(entry.guard_why);
    };
    // FRITCHII_EXTENT_SHARED: the extent proven through shared memory slots, or why not
    const auto shared_text = [&](const ExtentCensusEntry& entry) -> std::string {
        if (entry.cbuf_extent.via_shared && entry.cbuf_extent.ids.bounded) {
            const Shader::StorageWriteExtent& ids = entry.cbuf_extent.ids;
            return fmt::format("{} (end {:#x} + local {:#x}/{:#x}/{:#x} + group {:#x}/{:#x}/{:#x}"
                               "{})",
                               entry.cbuf_extent.num_terms == 0 ? "ids only" : "with words",
                               ids.const_end, ids.local_stride[0], ids.local_stride[1],
                               ids.local_stride[2], ids.group_stride[0], ids.group_stride[1],
                               ids.group_stride[2], ids.guarded ? ", 16-bit guard" : "");
        }
        if (entry.shared_why.kind == Shader::ExtentUnknown::Unrecorded) {
            return "not tried";
        }
        return "refused: " + why_text(entry.shared_why);
    };
    using Entry = std::pair<const std::pair<u64, u32>*, const ExtentCensusEntry*>;
    constexpr size_t Top = 8;
    if (census == 0) {
        ReadRunStats::g_wait_census.store(false, std::memory_order_relaxed);
    } else {
        // Read-back waits by writer (the read ring: FRITCHII_EXTENT_CENSUS or
        // FRITCHII_EXTENT_CBUF=3 at boot, or FRITCHII_READ_RUNS)
        const auto waits = ReadRunStats::TakeWaits();
        const auto wait_of =
            [&](const std::pair<u64, u32>& key) -> const ReadRunStats::WaitStats* {
            const ReadRunStats::Writer writer{.program = key.first,
                                              .binding = key.second,
                                              .kind = ReadRunStats::WriterKind::ComputeStorage};
            const auto it = waits.find(writer);
            return it != waits.end() ? &it->second : nullptr;
        };
        u64 dispatches = 0;
        u64 full_dispatches = 0;
        u64 full_bytes = 0;
        u64 unknown = 0;
        u64 no_extra = 0;
        u64 full_bindings = 0;
        for (const auto& [key, entry] : extent_census) {
            dispatches += entry.dispatches;
            full_dispatches += entry.full_dispatches;
            full_bytes += entry.full_bytes;
            unknown += entry.unknown_dispatches;
            no_extra += entry.no_extra_dispatches;
            full_bindings += entry.full_dispatches != 0 ? 1 : 0;
        }
        u64 wait_ns = 0;
        for (const auto& [writer, stats] : waits) {
            wait_ns += stats.wait_ns;
        }
        LOG_INFO(HW_GPU,
                 "Extent census (FRITCHII_EXTENT_CENSUS) last {:.1f} s: {:.0f} native written "
                 "bindings a second, {:.0f} of them marked in full ({:.0f} MiB/s) over {} program "
                 "bindings; extent unknown {:.0f}/s, {:.0f}/s of those with no reason recorded "
                 "(must be 0 with EDEN_NX2_PRECOMPILED_FAST=0) | read-back waits {:.2f} ms a frame "
                 "over {} writers{}",
                 seconds, static_cast<double>(dispatches) / seconds,
                 static_cast<double>(full_dispatches) / seconds,
                 static_cast<double>(full_bytes) / 1048576.0 / seconds, full_bindings,
                 static_cast<double>(unknown) / seconds, static_cast<double>(no_extra) / seconds,
                 static_cast<double>(wait_ns) / 1e6 / frames, waits.size(),
                 ReadRunStats::g_enabled ? ""
                                         : " (read ring off: FRITCHII_EXTENT_CENSUS at boot)");
        const auto log_entry = [&](const char* rank, size_t i, size_t n,
                                   const std::pair<u64, u32>& key,
                                   const ExtentCensusEntry& entry) {
            const ReadRunStats::WaitStats* waited = wait_of(key);
            const auto& w = entry.whys;
            LOG_INFO(HW_GPU,
                     "Extent census {} {}/{}: program {:016x} binding {} (pointer c{}[{:#x}]) | "
                     "read-back {} reads, {:.2f} ms a frame | {:.1f} dispatches/s, {:.1f} "
                     "full-marked/s, {:.1f} MiB marked/s ({} KiB binding) | why unknown: {} | "
                     "full mark because: unknown {}, indirect grid {}, 16-bit guard or dims {}, "
                     "distrusted {}, checking new dims {}, proof covers the binding {} | cbuf "
                     "extent: {} | shared extent: {} | guard extent: {}",
                     rank, i + 1, n, key.first, key.second, entry.pointer_cbuf_index,
                     entry.pointer_cbuf_offset, waited != nullptr ? waited->reads : 0,
                     waited != nullptr ? static_cast<double>(waited->wait_ns) / 1e6 / frames : 0.0,
                     static_cast<double>(entry.dispatches) / seconds,
                     static_cast<double>(entry.full_dispatches) / seconds,
                     static_cast<double>(entry.marked_bytes) / 1048576.0 / seconds,
                     entry.binding_bytes >> 10, why_text(entry.analysis),
                     w[static_cast<size_t>(FullMarkWhy::Analysis)],
                     w[static_cast<size_t>(FullMarkWhy::IndirectGrid)],
                     w[static_cast<size_t>(FullMarkWhy::GuardOrDims)],
                     w[static_cast<size_t>(FullMarkWhy::Distrusted)],
                     w[static_cast<size_t>(FullMarkWhy::Checking)],
                     w[static_cast<size_t>(FullMarkWhy::WholeBinding)],
                     entry.cbuf_extent.via_guard    ? std::string{"none (see guard extent)"}
                     : entry.cbuf_extent.via_shared ? std::string{"none (see shared extent)"}
                                                    : terms_text(entry.cbuf_extent),
                     shared_text(entry), guard_text(entry));
        };
        // 1. The writers the guest waited for most, whatever they are (the ones that matter)
        std::vector<std::pair<const ReadRunStats::Writer*, const ReadRunStats::WaitStats*>>
            by_wait;
        for (const auto& [writer, stats] : waits) {
            by_wait.emplace_back(&writer, &stats);
        }
        const size_t wait_shown = (std::min)(Top, by_wait.size());
        std::partial_sort(by_wait.begin(),
                          by_wait.begin() + static_cast<std::ptrdiff_t>(wait_shown), by_wait.end(),
                          [](const auto& a, const auto& b) {
                              return a.second->wait_ns > b.second->wait_ns;
                          });
        std::set<std::pair<u64, u32>> shown_keys;
        for (size_t i = 0; i < wait_shown; ++i) {
            const ReadRunStats::Writer& writer = *by_wait[i].first;
            const ReadRunStats::WaitStats& stats = *by_wait[i].second;
            const std::pair<u64, u32> key{writer.program, writer.binding};
            if (writer.kind == ReadRunStats::WriterKind::ComputeStorage) {
                if (const auto it = extent_census.find(key); it != extent_census.end()) {
                    log_entry("read-back", i, wait_shown, key, it->second);
                    shown_keys.insert(key);
                    continue;
                }
            }
            const bool draw = writer.kind == ReadRunStats::WriterKind::DrawStorage ||
                              writer.kind == ReadRunStats::WriterKind::DrawTexture;
            LOG_INFO(HW_GPU,
                     "Extent census read-back {}/{}: {} program {:016x} {} {} | read-back {} "
                     "reads, {:.2f} ms a frame | no bare native written binding of it bound in "
                     "this window (a sized buffer, a texture buffer, a draw or a copy): no store "
                     "extent applies",
                     i + 1, wait_shown, ReadRunStats::WriterKindName(writer.kind), writer.program,
                     draw ? "stage/binding" : "binding",
                     draw ? fmt::format("{}/{}", writer.binding / 64, writer.binding % 64)
                          : fmt::format("{}", writer.binding),
                     stats.reads, static_cast<double>(stats.wait_ns) / 1e6 / frames);
        }
        // 2. The bindings marked in full the most, not shown above
        std::vector<Entry> order;
        for (const auto& [key, entry] : extent_census) {
            if (entry.full_dispatches != 0 && !shown_keys.contains(key)) {
                order.emplace_back(&key, &entry);
            }
        }
        const size_t shown = (std::min)(Top, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                          order.end(), [](const Entry& a, const Entry& b) {
                              return a.second->full_bytes > b.second->full_bytes;
                          });
        for (size_t i = 0; i < shown; ++i) {
            log_entry("marked", i, shown, *order[i].first, *order[i].second);
        }
    }
    const bool cbuf_on = cbuf_mode == 1 || cbuf_mode == 3;
    const bool shared_on = shared_mode == 1 || shared_mode == 3;
    const bool guard_on = guard_mode == 1 || guard_mode == 3;
    if (cbuf_on || shared_on || guard_on) {
        // The dry read counts (=3: the read ring's marks carry the proof's end). The extents
        // proven through shared memory slots (FRITCHII_EXTENT_SHARED) and with the stores' path
        // facts (FRITCHII_EXTENT_GUARD) have their own lines.
        auto dry = ReadRunStats::TakeDry();
        std::vector<Entry> order;
        std::array<size_t, 3> proof_entries{};
        for (const auto& [key, entry] : extent_census) {
            if (entry.cbuf_resolved + entry.cbuf_fallback + entry.cbuf_unresolved != 0) {
                order.emplace_back(&key, &entry);
                ++proof_entries[proof_kind(entry.cbuf_extent)];
            }
        }
        const size_t cbuf_entries = proof_entries[0];
        const size_t shared_entries = proof_entries[1];
        const size_t guard_entries = proof_entries[2];
        if ((cbuf_on && cbuf_entries == 0) || (shared_on && shared_entries == 0) ||
            (guard_on && guard_entries == 0)) {
            // Say so rather than stay silent: no binding had such an extent to try
            u64 unknown = 0;
            u64 no_extra = 0;
            u64 shared_refused = 0;
            u64 guard_refused = 0;
            const auto refused = [](const Shader::ExtentUnknownWhy& why) {
                return why.kind != Shader::ExtentUnknown::Unrecorded &&
                       why.kind != Shader::ExtentUnknown::Known;
            };
            for (const auto& [key, entry] : extent_census) {
                unknown += entry.unknown_dispatches;
                no_extra += entry.no_extra_dispatches;
                shared_refused += refused(entry.shared_why) ? entry.dispatches : 0;
                guard_refused += refused(entry.guard_why) ? entry.dispatches : 0;
            }
            if (guard_on && guard_entries == 0) {
                LOG_INFO(HW_GPU,
                         "Extent guard (FRITCHII_EXTENT_GUARD={}) last {:.1f} s: no native "
                         "written binding had an extent proven with its stores' path facts; "
                         "refused {:.0f} dispatches/s (the census' guard extent says why); "
                         "extent unknown {:.0f} dispatches/s, {:.0f}/s of them with no reason "
                         "recorded (run with EDEN_NX2_PRECOMPILED_FAST=0)",
                         guard_mode, seconds, static_cast<double>(guard_refused) / seconds,
                         static_cast<double>(unknown) / seconds,
                         static_cast<double>(no_extra) / seconds);
            }
            if (cbuf_on && cbuf_entries == 0) {
                LOG_INFO(HW_GPU,
                         "Extent cbuf (FRITCHII_EXTENT_CBUF={}) last {:.1f} s: no native written "
                         "binding had a cbuf extent; extent unknown {:.0f} dispatches/s, {:.0f}/s "
                         "of them with no reason recorded (no Shader::StorageWriteExtra)",
                         cbuf_mode, seconds, static_cast<double>(unknown) / seconds,
                         static_cast<double>(no_extra) / seconds);
            }
            if (shared_on && shared_entries == 0) {
                LOG_INFO(HW_GPU,
                         "Extent shared (FRITCHII_EXTENT_SHARED={}) last {:.1f} s: no native "
                         "written binding had an extent proven through shared memory slots; "
                         "refused {:.0f} dispatches/s (the census' shared extent says why); "
                         "extent unknown {:.0f} dispatches/s, {:.0f}/s of them with no reason "
                         "recorded (run with EDEN_NX2_PRECOMPILED_FAST=0)",
                         shared_mode, seconds, static_cast<double>(shared_refused) / seconds,
                         static_cast<double>(unknown) / seconds,
                         static_cast<double>(no_extra) / seconds);
            }
        }
        const auto dry_of = [&](const std::pair<u64, u32>& key) -> const ReadRunStats::DryStats* {
            const ReadRunStats::Writer writer{.program = key.first,
                                              .binding = key.second,
                                              .kind = ReadRunStats::WriterKind::ComputeStorage};
            const auto it = dry.find(writer);
            return it != dry.end() ? &it->second : nullptr;
        };
        const auto weight = [&](const Entry& e) {
            const ReadRunStats::DryStats* d = dry_of(*e.first);
            u64 wait = 0;
            if (d != nullptr) {
                for (const u64 ns : d->wait_ns) {
                    wait += ns;
                }
            }
            return std::pair<u64, u64>{wait, e.second->cbuf_resolved};
        };
        const size_t shown = (std::min)(Top, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown),
                          order.end(),
                          [&](const Entry& a, const Entry& b) { return weight(a) > weight(b); });
        for (size_t i = 0; i < shown; ++i) {
            const auto& [key, entry] = order[i];
            const ReadRunStats::DryStats none{};
            const ReadRunStats::DryStats* found = dry_of(*key);
            const ReadRunStats::DryStats& d = found != nullptr ? *found : none;
            using DR = ReadRunStats::DryRead;
            const auto reads = [&](DR kind) { return d.reads[static_cast<size_t>(kind)]; };
            const auto ms_frame = [&](DR kind) {
                return static_cast<double>(d.wait_ns[static_cast<size_t>(kind)]) / 1e6 / frames;
            };
            const double kib_proven =
                entry->cbuf_resolved != 0
                    ? static_cast<double>(entry->cbuf_proven_bytes) / 1024.0 /
                          static_cast<double>(entry->cbuf_resolved)
                    : 0.0;
            const double kib_binding =
                entry->cbuf_resolved != 0
                    ? static_cast<double>(entry->cbuf_binding_bytes) / 1024.0 /
                          static_cast<double>(entry->cbuf_resolved)
                    : 0.0;
            const size_t kind = proof_kind(entry->cbuf_extent);
            const std::string proof_text = kind == 2   ? guard_text(*entry)
                                           : kind == 1 ? shared_text(*entry)
                                                       : terms_text(entry->cbuf_extent);
            LOG_INFO(HW_GPU,
                     "Extent {} ({}={}) {}/{} last {:.1f} s: program {:016x} "
                     "binding {} | {} extent {} | dispatches/s: proof {:.1f}, fallback (word on "
                     "a GPU-written page) {:.1f}, unresolved {:.1f}, marked by the proof (=1) "
                     "{:.1f} | proven {:.1f} KiB of {:.1f} KiB on average | reads to the back by "
                     "its marks (=3): inside the proof {} ({:.2f} ms a frame), would be freed {} "
                     "({:.2f} ms a frame), past the proof but under an older mark of the same "
                     "writer {} / another writer {} ({:.2f} / {:.2f} ms a frame) | must be 0: "
                     "freed reads whose bytes changed at landing {} of {} compared (inside: {} of "
                     "{})",
                     proof_word[kind], proof_switch[kind], proof_mode[kind], i + 1, shown,
                     seconds, key->first, key->second, proof_word[kind], proof_text,
                     static_cast<double>(entry->cbuf_resolved) / seconds,
                     static_cast<double>(entry->cbuf_fallback) / seconds,
                     static_cast<double>(entry->cbuf_unresolved) / seconds,
                     static_cast<double>(entry->cbuf_marked) / seconds, kib_proven, kib_binding,
                     reads(DR::Inside), ms_frame(DR::Inside), reads(DR::Freed),
                     ms_frame(DR::Freed), reads(DR::CoveredSame), reads(DR::CoveredOther),
                     ms_frame(DR::CoveredSame), ms_frame(DR::CoveredOther), d.freed_differed,
                     d.freed_compared, d.inside_differed, d.inside_compared);
            // The reads past the proof under an older mark: whose marks they would still wait
            // for, and how long (timed with the write ticks: FRITCHII_EXTENT_SHARED=3, _GUARD=3 or
            // FRITCHII_READ_RUNS=2 at boot)
            u64 all_wait = 0;
            for (const u64 ns : d.wait_ns) {
                all_wait += ns;
            }
            u64 timed_wait = 0;
            u64 timed_predicted = 0;
            std::vector<std::pair<const ReadRunStats::Writer*, const ReadRunStats::DryStats::Cover*>>
                covers;
            for (const auto& [writer, cover] : d.covers) {
                covers.emplace_back(&writer, &cover);
                timed_wait += cover.timed_wait_ns;
                timed_predicted += cover.predicted_ns;
            }
            constexpr size_t TopCovers = 4;
            const size_t covers_shown = (std::min)(TopCovers, covers.size());
            std::partial_sort(covers.begin(),
                              covers.begin() + static_cast<std::ptrdiff_t>(covers_shown),
                              covers.end(), [](const auto& a, const auto& b) {
                                  return a.second->wait_ns > b.second->wait_ns;
                              });
            std::string covered_by;
            for (size_t c = 0; c < covers_shown; ++c) {
                const ReadRunStats::Writer& writer = *covers[c].first;
                const ReadRunStats::DryStats::Cover& cover = *covers[c].second;
                std::string why = "not in the census (not a bare native written binding)";
                if (writer.kind == ReadRunStats::WriterKind::ComputeStorage) {
                    if (const auto it = extent_census.find({writer.program, writer.binding});
                        it != extent_census.end()) {
                        why = fmt::format("{}, {:.1f} full-marked/s, pointer c{}[{:#x}]",
                                          why_text(it->second.analysis),
                                          static_cast<double>(it->second.full_dispatches) /
                                              seconds,
                                          it->second.pointer_cbuf_index,
                                          it->second.pointer_cbuf_offset);
                    }
                }
                const bool draw = writer.kind == ReadRunStats::WriterKind::DrawStorage ||
                                  writer.kind == ReadRunStats::WriterKind::DrawTexture;
                covered_by += fmt::format(
                    "{}{} {:016x} {} {}: {} reads, {:.2f} ms a frame, timed {} would wait {:.2f} "
                    "ms a frame instead of {:.2f} (why unknown: {})",
                    c == 0 ? "" : "; ", ReadRunStats::WriterKindName(writer.kind),
                    writer.program, draw ? "stage/binding" : "binding",
                    draw ? fmt::format("{}/{}", writer.binding / 64, writer.binding % 64)
                         : fmt::format("{}", writer.binding),
                    cover.reads, static_cast<double>(cover.wait_ns) / 1e6 / frames, cover.timed,
                    static_cast<double>(cover.predicted_ns) / 1e6 / frames,
                    static_cast<double>(cover.timed_wait_ns) / 1e6 / frames, why);
            }
            LOG_INFO(HW_GPU,
                     "Extent {} ({}={}) {}/{} with the proof: its reads would wait {:.2f} ms a "
                     "frame instead of {:.2f} (freed reads as 0, covered reads until their cover "
                     "ran: timed {}, untimed {} kept as waited; the timed ones {:.2f} instead of "
                     "{:.2f}) | covered by {} writers: {}",
                     proof_word[kind], proof_switch[kind], proof_mode[kind], i + 1, shown,
                     static_cast<double>(d.proof_wait_ns) / 1e6 / frames,
                     static_cast<double>(all_wait) / 1e6 / frames, d.covered_timed,
                     d.covered_untimed, static_cast<double>(timed_predicted) / 1e6 / frames,
                     static_cast<double>(timed_wait) / 1e6 / frames, covers.size(),
                     covered_by.empty() ? std::string{"none"} : covered_by);
        }
    }
    extent_census.clear();
    extent_census_frames = 0;
}

template <class P>
void BufferCache<P>::UnbindComputeTextureBuffers() {
    channel_state->enabled_compute_texture_buffers = 0;
    channel_state->written_compute_texture_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                              PixelFormat format, bool is_written, bool is_image) {
    if (tbo_index >= channel_state->compute_texture_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Texture buffer index {} exceeds maximum texture buffer count",
                  tbo_index);
        return;
    }
    channel_state->enabled_compute_texture_buffers |= 1U << tbo_index;
    channel_state->written_compute_texture_buffers |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_compute_texture_buffers |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->compute_texture_buffers[tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::FlushCachedWrites() {
    memory_tracker.FlushCachedWrites();
}

template <class P>
typename BufferCache<P>::SyncProbe BufferCache<P>::ProbeSyncTarget(DAddr begin, u64 bytes) {
    // FindBuffer's hit (else CreateBuffer), then SynchronizeBuffer's test: ForEachUploadRange
    // clears CPU-modified bits even where it uploads nothing, and re-tracks untracked pages (a
    // cached write's), so either one is a change only the back may make at its place.
    const u64 page = begin >> CACHING_PAGEBITS;
    if (begin == 0 || page >= page_table.size()) {
        return SyncProbe::NoBuffer;
    }
    const BufferId buffer_id = page_table[page];
    if (!buffer_id) {
        return SyncProbe::NoBuffer;
    }
    if (!slot_buffers[buffer_id].IsInBounds(begin, bytes)) {
        return SyncProbe::NotInBounds;
    }
    if (!memory_tracker.IsRegionSyncClean(begin, bytes)) {
        return SyncProbe::CpuModified;
    }
    return SyncProbe::Pure;
}

template <class P>
bool BufferCache<P>::HasUncommittedFlushes() const noexcept {
    // FRITCHII_MEMSTAGE_MARKS_AHEAD: its callers promote first (the fence commit, FencesQuiet).
    if (AheadDue()) [[unlikely]] {
        NoteAheadCheck(AheadCheck::Stale, "HasUncommittedFlushes");
    }
    return !uncommitted_gpu_modified_ranges.Empty() || !committed_gpu_modified_ranges.empty();
}

template <class P>
void BufferCache<P>::AccumulateFlushes() {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    if (uncommitted_gpu_modified_ranges.Empty()) {
        return;
    }
    committed_gpu_modified_ranges.emplace_back(std::move(uncommitted_gpu_modified_ranges));
}

template <class P>
bool BufferCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return (!async_buffers.empty() && async_buffers.front().has_value());
}

template <class P>
void BufferCache<P>::CommitAsyncFlushesHigh() {
    AccumulateFlushes();

    if (committed_gpu_modified_ranges.empty()) {
        if (DownloadReadStats::g_enabled) [[unlikely]] {
            DownloadReadStats::Committed(0, 0);
        }
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }

    auto it = committed_gpu_modified_ranges.begin();
    while (it != committed_gpu_modified_ranges.end()) {
        auto& current_intervals = *it;
        auto next_it = std::next(it);
        while (next_it != committed_gpu_modified_ranges.end()) {
            next_it->ForEach([&current_intervals](DAddr start, DAddr end) {
                current_intervals.Subtract(start, end - start);
            });
            next_it++;
        }
        it++;
    }

    boost::container::small_vector<std::pair<BufferCopy, BufferId>, 16> downloads;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    for (const Common::RangeSet<DAddr>& range_set : committed_gpu_modified_ranges) {
        range_set.ForEach([&](DAddr interval_lower, DAddr interval_upper) {
            const std::size_t size = interval_upper - interval_lower;
            const DAddr device_addr = interval_lower;
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_start = buffer.CpuAddr();
                const DAddr buffer_end = buffer_start + buffer.SizeBytes();
                const DAddr new_start = (std::max)(buffer_start, device_addr);
                const DAddr new_end = (std::min)(buffer_end, device_addr + size);
                memory_tracker.ForEachDownloadRange(
                    new_start, new_end - new_start, false,
                    [&](u64 device_addr_out, u64 range_size) {
                        const DAddr buffer_addr = buffer.CpuAddr();
                        const auto add_download = [&](DAddr start, DAddr end) {
                            const u64 new_offset = start - buffer_addr;
                            const u64 new_size = end - start;
                            downloads.push_back({
                                BufferCopy{
                                    .src_offset = new_offset,
                                    .dst_offset = total_size_bytes,
                                    .size = new_size,
                                },
                                buffer_id,
                            });
                            // Align up to avoid cache conflicts
                            constexpr u64 align = 64ULL;
                            constexpr u64 mask = ~(align - 1ULL);
                            total_size_bytes += (new_size + align - 1) & mask;
                            largest_copy = (std::max)(largest_copy, new_size);
                        };

                        gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                           add_download);
                    });
            });
        });
    }
    committed_gpu_modified_ranges.clear();
    if (DownloadReadStats::g_enabled) [[unlikely]] {
        DownloadReadStats::Committed(downloads.size(), total_size_bytes);
    }
    if (downloads.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }
    Nx2Profile::Add(Nx2Profile::Counter::DownloadBytes, total_size_bytes);
    Nx2Profile::Add(Nx2Profile::Counter::DownloadCopies, downloads.size());
    // EDEN_NX2_BACK_FENCE_STATS (fable52_answer_fences2 4): the staging buffer's part of the fence.
    const bool time_staging = VideoCore::Nx2Debug::BackFenceStats();
    const auto staging_start = time_staging ? FenceReleaseStats::Clock::now()
                                            : FenceReleaseStats::Clock::time_point{};
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
    if (time_staging) [[unlikely]] {
        FenceReleaseStats::BackFence::t_staging_ns += FenceReleaseStats::NsSince(staging_start);
    }
    boost::container::small_vector<BufferCopy, 4> normalized_copies;
    Common::RangeSet<DAddr> download_ranges;
    runtime.PreCopyBarrier();
    for (auto& [copy, buffer_id] : downloads) {
        copy.dst_offset += download_staging.offset;
        const std::array copies{copy};
        BufferCopy second_copy{copy};
        Buffer& buffer = slot_buffers[buffer_id];
        second_copy.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
        const DAddr orig_device_addr = static_cast<DAddr>(second_copy.src_offset);
        if (Nx2Profile::Enabled()) {
            u64 on_dma_pages = 0;
            const auto& pages = Nx2Profile::DmaReadPages();
            for (DAddr page = orig_device_addr >> 12; page <= (orig_device_addr + copy.size - 1) >> 12;
                 ++page) {
                if (pages.contains(page)) {
                    const DAddr begin = (std::max)(orig_device_addr, page << 12);
                    const DAddr end = (std::min)(orig_device_addr + copy.size, (page + 1) << 12);
                    on_dma_pages += end - begin;
                }
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadBytesOnDmaPages, on_dma_pages);
        }
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Copied(orig_device_addr, copy.size);
        }
        download_ranges.Add(orig_device_addr, copy.size);
        buffer.MarkUsage(copy.src_offset, copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, copies, false);
        normalized_copies.push_back(second_copy);
    }
    runtime.PostCopyBarrier();
    if (LandingCheckMode() != 0) [[unlikely]] {
        LandingCheckCommit(std::span<const BufferCopy>(normalized_copies.data(),
                                                       normalized_copies.size()));
    }
    pending_downloads.emplace_back(std::move(normalized_copies));
    pending_download_ranges.emplace_back(std::move(download_ranges));
    async_buffers.emplace_back(download_staging);
}

template <class P>
void BufferCache<P>::CommitAsyncFlushes() {
    CommitAsyncFlushesHigh();
}

template <class P>
void BufferCache<P>::PopAsyncFlushes() {
    PopAsyncBuffers();
}

template <class P>
template <typename Ranges>
void BufferCache<P>::ClaimCpuWrites(const Ranges& ranges) {
    VideoCommon::MemoryStageLink::CheckBackCache("ClaimCpuWrites");
    for (const auto& [device_addr, size] : ranges) {
        NoteAheadLevelCall(device_addr, size, "ClaimCpuWrites");
        if (IsRegionGpuModified(device_addr, size)) {
            // The CPU's bytes are in guest memory: read downloads leave the range.
            ClearReplacedDownload(device_addr, size);
            gpu_modified_ranges.Subtract(device_addr, size);
        }
    }
}

template <class P>
void BufferCache<P>::Nx2ApplyCpuBytes(DAddr device_addr, u64 size, const u8* bytes) {
    if (size == 0) {
        return;
    }
    const DAddr page = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    if (device_addr + size <= page + Core::DEVICE_PAGESIZE) {
        if (NativeBankPage* const snapshot = FindNativeBankSnapshot(page)) {
            std::memcpy(snapshot->words.data() + (device_addr - page), bytes, size);
        }
    }
    if (!IsRegionRegistered(device_addr, static_cast<size_t>(size))) {
        WriteMemory(device_addr, size);
        return;
    }
    InlineMemoryImplementation(device_addr, static_cast<size_t>(size),
                               std::span<const u8>(bytes, static_cast<size_t>(size)));
}

template <class P>
Common::RangeSet<DAddr> BufferCache<P>::PendingCpuWriteRanges() {
    Common::RangeSet<DAddr> ranges;
    if (pending_cpu_writes) {
        pending_cpu_writes(
            [&ranges](DAddr device_addr, size_t size) { ranges.Add(device_addr, size); });
    }
    return ranges;
}

template <class P>
void BufferCache<P>::WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written,
                                           DAddr device_addr, const u8* data, u64 size) {
    const Common::WriteRing::SiteScope write_site{"buffer download", true};
    // FRITCHII_S4_COUNT: GPU results landing in game memory.
    if (Common::S4Count::On()) [[unlikely]] {
        Common::S4Count::MarkGens(Common::S4Count::Writer::Gpu, device_addr, size);
    }
    Core::g_buffer_write_pages.Mark(device_addr, size);
    Core::g_watch_write_pages.Mark(device_addr, size);
    if (LandingCheckMode() != 0) [[unlikely]] {
        LandingCheckNoteLanded(device_addr, size);
    }
    // FRITCHII_STORE_FOOTPRINT=2: would this landing overwrite guest bytes no footprint writes?
    if (footprint_following && StoreFootprintLive::Mode() == 2) [[unlikely]] {
        FootprintCheckLanding(&cpu_written, device_addr, data, size);
    }
    if (!cpu_written.Intersects(device_addr, size)) {
        device_memory.WriteBlockUnsafe(device_addr, data, size);
        return;
    }
    Common::RangeSet<DAddr> parts;
    parts.Add(device_addr, size);
    cpu_written.ForEachInRange(device_addr, size, [&parts](DAddr start, DAddr end) {
        parts.Subtract(start, end - start);
    });
    parts.ForEach([&](DAddr start, DAddr end) {
        device_memory.WriteBlockUnsafe(start, data + (start - device_addr), end - start);
    });
}

template <class P>
int BufferCache<P>::LandingCheckMode() {
    static const int mode = [] {
        const char* const value = std::getenv("FRITCHII_LANDING_CHECK");
        const int parsed = value != nullptr ? std::atoi(value) : 0;
        if (parsed != 0) {
            LOG_INFO(HW_GPU,
                     "Landing check (FRITCHII_LANDING_CHECK={}): fence downloads compare the guest "
                     "bytes they land over with the bytes at their commit{}",
                     parsed, parsed == 2 ? "; bytes the guest changed since are kept" : "");
        }
        return parsed;
    }();
    return mode;
}

template <class P>
void BufferCache<P>::LandingCheckCommit(std::span<const BufferCopy> copies) {
    LandingSnapshot snapshot;
    snapshot.committed_ns = Common::Nx2Hitch::NowNs();
    {
        std::scoped_lock lock{landing_check_mutex};
        snapshot.landing_seq = landing_check_seq;
    }
    u64 total = 0;
    snapshot.copy_offsets.reserve(copies.size());
    for (const BufferCopy& copy : copies) {
        snapshot.copy_offsets.push_back(total);
        total += copy.size;
    }
    snapshot.bytes.resize(total);
    for (size_t index = 0; index < copies.size(); ++index) {
        device_memory.ReadBlockUnsafe(static_cast<DAddr>(copies[index].src_offset),
                                      snapshot.bytes.data() + snapshot.copy_offsets[index],
                                      copies[index].size);
    }
    landing_snapshots.push_back(std::move(snapshot));
}

template <class P>
void BufferCache<P>::LandingCheckPop(bool begin) {
    std::scoped_lock lock{landing_check_mutex};
    if (begin) {
        t_landing_check_pop_seq = ++landing_check_seq;
        return;
    }
    t_landing_check_pop_seq = 0;
    if (landing_check_pages.size() > 65536) {
        // Only landings after the oldest pending commit matter (the popped snapshot is gone).
        const u64 oldest = !landing_snapshots.empty() ? landing_snapshots.front().landing_seq
                                                      : landing_check_seq;
        std::erase_if(landing_check_pages, [oldest](const auto& entry) {
            return entry.second <= oldest;
        });
    }
}

template <class P>
void BufferCache<P>::LandingCheckNoteLanded(DAddr device_addr, u64 size) {
    if (size == 0) {
        return;
    }
    std::scoped_lock lock{landing_check_mutex};
    const u64 seq =
        t_landing_check_pop_seq != 0 ? t_landing_check_pop_seq : ++landing_check_seq;
    for (u64 page = device_addr >> Core::DEVICE_PAGEBITS;
         page <= (device_addr + size - 1) >> Core::DEVICE_PAGEBITS; ++page) {
        landing_check_pages[page] = seq;
    }
}

template <class P>
void BufferCache<P>::LandingCheckPiece(const LandingSnapshot& snapshot, const BufferCopy& copy,
                                       size_t index, DAddr start, u8* data, u64 size,
                                       const Common::RangeSet<DAddr>* cpu_written) {
    const int mode = LandingCheckMode();
    const DAddr copy_addr = static_cast<DAddr>(copy.src_offset);
    if (size == 0 || index >= snapshot.copy_offsets.size() || start < copy_addr ||
        start + size > copy_addr + copy.size) {
        return;
    }
    const u8* const old = snapshot.bytes.data() + snapshot.copy_offsets[index] + (start - copy_addr);
    std::vector<u8> now(size);
    device_memory.ReadBlockUnsafe(start, now.data(), size);
    // Collected CPU writes: the landing leaves those bytes alone. Treated as unchanged.
    if (cpu_written != nullptr && cpu_written->Intersects(start, size)) {
        cpu_written->ForEachInRange(start, size, [&](DAddr lower, DAddr upper) {
            for (DAddr address = (std::max)(lower, start); address < (std::min)(upper, start + size);
                 ++address) {
                now[address - start] = old[address - start];
            }
        });
    }
    u64 changed = 0;
    u64 skipped = 0;
    u64 first_changed = ~u64{0};
    std::array<u8, 8> first_now{}, first_old{}, first_landing{};
    {
        std::scoped_lock lock{landing_check_mutex};
        u64 i = 0;
        while (i < size) {
            const u64 page = (start + i) >> Core::DEVICE_PAGEBITS;
            const u64 page_end =
                (std::min)(size, ((page + 1) << Core::DEVICE_PAGEBITS) - start);
            const auto it = landing_check_pages.find(page);
            // A landing (or read download) after the commit wrote this page: its bytes changed
            // by a download, not by the guest.
            const bool landed_since = it != landing_check_pages.end() &&
                                      it->second > snapshot.landing_seq &&
                                      it->second != t_landing_check_pop_seq;
            for (; i < page_end; ++i) {
                if (now[i] == old[i] || now[i] == data[i]) {
                    continue;
                }
                if (landed_since) {
                    ++skipped;
                    continue;
                }
                if (first_changed == ~u64{0}) {
                    first_changed = i;
                    for (u64 b = 0; b < 8 && i + b < size; ++b) {
                        first_now[b] = now[i + b];
                        first_old[b] = old[i + b];
                        first_landing[b] = data[i + b];
                    }
                }
                ++changed;
                if (mode == 2) {
                    data[i] = now[i]; // the landing writes the guest's own byte back
                }
            }
        }
    }
    auto& totals = landing_check_totals;
    ++totals.landings;
    totals.bytes += size;
    totals.skipped_bytes += skipped;
    if (changed != 0) {
        ++totals.changed_landings;
        totals.changed_bytes += changed;
        if (mode == 2) {
            totals.kept_bytes += changed;
        }
        static std::atomic<u64> logged{};
        if (const u64 n = logged.fetch_add(1, std::memory_order_relaxed); n < 32 || n % 1000 == 0) {
            const auto hex = [](const std::array<u8, 8>& bytes) {
                u64 value = 0;
                std::memcpy(&value, bytes.data(), sizeof(value));
                return value;
            };
            LOG_CRITICAL(
                HW_GPU,
                "Landing check: a fence download lands over bytes the guest changed since its copy "
                "was taken (must not happen): device [{:#x}, {:#x}), {} of {} bytes changed, "
                "first at +{:#x}: guest now {:#018x}, at the commit {:#018x}, the landing "
                "{:#018x} (little-endian words); {:.1f} ms after the commit, {} fences pending; "
                "{} (#{})",
                start, start + size, changed, size, first_changed, hex(first_now), hex(first_old),
                hex(first_landing),
                static_cast<double>(Common::Nx2Hitch::NowNs() - snapshot.committed_ns) / 1e6,
                pending_downloads.size(),
                mode == 2 ? "kept the guest's bytes" : "landed anyway (FRITCHII_LANDING_CHECK=2 keeps them)",
                n + 1);
        }
    }
    const u64 now_ns = Common::Nx2Hitch::NowNs();
    if (now_ns - totals.last_log_ns >= 10'000'000'000ULL) {
        if (totals.last_log_ns != 0) {
            LOG_INFO(HW_GPU,
                     "Landing check (FRITCHII_LANDING_CHECK={}) 10 s: {} landings, {} KiB checked; "
                     "changed by the guest since the copy: {} landings, {} bytes (must be 0; {} "
                     "kept); {} bytes on pages a later download wrote (skipped)",
                     mode, totals.landings, totals.bytes >> 10, totals.changed_landings,
                     totals.changed_bytes, totals.kept_bytes, totals.skipped_bytes);
        }
        totals = LandingCheckTotals{};
        totals.last_log_ns = now_ns;
    }
}

template <class P>
void BufferCache<P>::PopAsyncBuffers() {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    const Common::WriteRing::SiteScope write_site{"buffer download landing (PopAsyncBuffers)", true};
    // EDEN_NX2_EXP49: the host GPU is past this fence. The read downloads taken before its copies
    // are complete and go to guest memory first; those taken after took their bytes out of its
    // copies (ClearFenceDownloads).
    if constexpr (!IS_OPENGL) {
        if (!read_downloads.empty()) [[unlikely]] {
            (void)runtime.IsFreeNow(read_downloads.front().tick);
            WriteCompletedReadDownloads();
        }
    }
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    auto& downloads = pending_downloads.front();
    auto& ranges = pending_download_ranges.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    // FRITCHII_LANDING_CHECK: this fence's snapshot (taken at its commit).
    const LandingSnapshot* const landing_snapshot =
        LandingCheckMode() != 0 && !landing_snapshots.empty() ? &landing_snapshots.front()
                                                              : nullptr;
    if (landing_snapshot != nullptr) [[unlikely]] {
        LandingCheckPop(true);
    }
    // F2 counter (EDEN_NX2_DOWNLOAD_READ_STATS): the ranges this fence writes to guest memory.
    std::vector<std::pair<u64, u64>> stats_landed;
    // Landed per copy (copies within a page of each other together), each inside a
    // VideoCore::DownloadLanding that it looks at the collected CPU writes in: with preflush a
    // fence's copies span unrelated pages, and one landing over their hull would hold guest
    // writes anywhere in it until all of them landed.
    size_t first = 0;
    while (first < downloads.size()) {
        DAddr landing_begin = static_cast<DAddr>(downloads[first].src_offset);
        DAddr landing_end = landing_begin + downloads[first].size;
        size_t last = first + 1;
        for (; last < downloads.size(); ++last) {
            const DAddr begin = static_cast<DAddr>(downloads[last].src_offset);
            const DAddr end = begin + downloads[last].size;
            if (end + Core::DEVICE_PAGESIZE < landing_begin ||
                begin > landing_end + Core::DEVICE_PAGESIZE) {
                break;
            }
            landing_begin = (std::min)(landing_begin, begin);
            landing_end = (std::max)(landing_end, end);
        }
        const VideoCore::DownloadLanding landing{landing_begin, landing_end};
        // This runs on the fencing thread while the CPU keeps writing: CPU writes collected since
        // the copy was taken are newer than it, and the GPU thread hasn't dropped them from
        // `ranges` yet.
        // EDEN_NX2_FENCE_RELEASE_STATS (fable52_answer_fences2 3): the two parts that can wait.
        const bool time_parts = FenceReleaseStats::g_enabled;
        auto part_start = time_parts ? FenceReleaseStats::Clock::now()
                                     : FenceReleaseStats::Clock::time_point{};
        const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
        if (time_parts) [[unlikely]] {
            FenceReleaseStats::t_pop_parts.pending_ns += FenceReleaseStats::NsSince(part_start);
            ++FenceReleaseStats::t_pop_parts.pending_calls;
        }
        for (size_t index = first; index < last; ++index) {
            const auto& copy = downloads[index];
            const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
            const u64 dst_offset = copy.dst_offset - base_offset;
            const u8* read_mapped_memory = base + dst_offset;
            // Only what no write replaced since the copy was taken: the rest is older than guest
            // memory (CPU writes) or than the GPU's copy (GPU writes, still to be downloaded).
            ranges.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end) {
                if (time_parts) [[unlikely]] {
                    part_start = FenceReleaseStats::Clock::now();
                }
                if (landing_snapshot != nullptr) [[unlikely]] {
                    // The staging bytes are ours (mode 2 patches the guest's changes in).
                    LandingCheckPiece(*landing_snapshot, copy, index, start,
                                      const_cast<u8*>(&read_mapped_memory[start - device_addr]),
                                      end - start, &cpu_written);
                }
                WriteDownloadedMemory(cpu_written, start, &read_mapped_memory[start - device_addr],
                                      end - start);
                if (time_parts) [[unlikely]] {
                    FenceReleaseStats::t_pop_parts.write_ns +=
                        FenceReleaseStats::NsSince(part_start);
                    ++FenceReleaseStats::t_pop_parts.writes;
                }
                gpu_modified_ranges.Subtract(start, end - start);
                if (DownloadReadStats::g_enabled) [[unlikely]] {
                    stats_landed.emplace_back(start, end);
                }
            });
        }
        first = last;
    }
    if (DownloadReadStats::g_enabled) [[unlikely]] {
        DownloadReadStats::Landed(stats_landed);
    }
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    pending_downloads.pop_front();
    pending_download_ranges.pop_front();
    if (LandingCheckMode() != 0 && !landing_snapshots.empty()) [[unlikely]] {
        landing_snapshots.pop_front();
        LandingCheckPop(false);
    }
    {
        // EXP72: in guest memory now (the copies above wrote it).
        std::scoped_lock lock{fence_landed_mutex};
        fence_downloads_landed.fetch_add(1, std::memory_order_release);
    }
    fence_landed_cv.notify_all();
}

template <class P>
bool BufferCache<P>::PendingCpuWritesIn(DAddr begin, DAddr end, Common::RangeSet<DAddr>& out) {
    out.Clear();
    if (!pending_cpu_writes) {
        return false;
    }
    // Only the writes over [begin, end): the callers ask about ranges inside it alone, so the
    // answers are PendingCpuWriteRanges' (without building a set of every collected write).
    struct Filter {
        DAddr begin;
        DAddr end;
        Common::RangeSet<DAddr>* out;
        bool any;
    } filter{begin, end, &out, false};
    pending_cpu_writes([filter_ptr = &filter](DAddr address, size_t size) {
        if (size != 0 && address < filter_ptr->end && address + size > filter_ptr->begin) {
            filter_ptr->out->Add(address, size);
            filter_ptr->any = true;
        }
    });
    return filter.any;
}

template <class P>
void BufferCache<P>::WriteDownloadedMemoryUnlocked(const Common::RangeSet<DAddr>* cpu_written,
                                                   DAddr device_addr, const u8* data, u64 size) {
    const Common::WriteRing::SiteScope write_site{"buffer download", true};
    if (LandingCheckMode() != 0) [[unlikely]] {
        LandingCheckNoteLanded(device_addr, size);
    }
    if (cpu_written == nullptr || !cpu_written->Intersects(device_addr, size)) {
        device_memory.WriteBlockUnsafe(device_addr, data, size);
    } else {
        unlocked_parts.Clear();
        unlocked_parts.Add(device_addr, size);
        cpu_written->ForEachInRange(device_addr, size, [this](DAddr start, DAddr end) {
            unlocked_parts.Subtract(start, end - start);
        });
        unlocked_parts.ForEach([&](DAddr start, DAddr end) {
            device_memory.WriteBlockUnsafe(start, data + (start - device_addr), end - start);
        });
    }
    // WriteDownloadedMemory's marks, after the bytes landed: it marks first, which holds only
    // under the lock (the native bank check takes g_buffer_write_pages under it). A mark taken
    // between a mark and the bytes would leave the bytes unseen; after, a look before the mark
    // sees the page as before this fence (as a draw before the pop did), the next one sees it.
    if (Common::S4Count::On()) [[unlikely]] {
        Common::S4Count::MarkGens(Common::S4Count::Writer::Gpu, device_addr, size);
    }
    Core::g_buffer_write_pages.Mark(device_addr, size);
    Core::g_watch_write_pages.Mark(device_addr, size);
}

template <class P>
void BufferCache<P>::VerifyUnlockedLanding(DAddr landing_begin, DAddr landing_end,
                                           const Common::RangeSet<DAddr>& live,
                                           Common::LockCensus::FenceCopyBatch& batch) {
    // Under the lock, in a landing of its own: guest CPU writes collected from here on store
    // their bytes after the compare. What the locked pop would have written: the pieces' bytes
    // where the fence still writes (`live`), except over the CPU writes it found collected then
    // (unlocked_cpu_written) and those collected since (they store after the copy: newer).
    static std::atomic<u32> logged{};
    const VideoCore::DownloadLanding landing{landing_begin, landing_end};
    PendingCpuWritesIn(landing_begin, landing_end, unlocked_verify_cpu);
    for (const UnlockedPiece& piece : unlocked_pieces) {
        bool bad_piece = false;
        live.ForEachInRange(piece.addr, piece.size, [&](DAddr start, DAddr end) {
            unlocked_verify_parts.Clear();
            unlocked_verify_parts.Add(start, end - start);
            const auto cut = [this](DAddr lower, DAddr upper) {
                unlocked_verify_parts.Subtract(lower, upper - lower);
            };
            unlocked_cpu_written.ForEachInRange(start, end - start, cut);
            unlocked_verify_cpu.ForEachInRange(start, end - start, cut);
            unlocked_verify_parts.ForEach([&](DAddr lower, DAddr upper) {
                const u64 bytes = upper - lower;
                if (unlocked_verify_bytes.size() < bytes) {
                    unlocked_verify_bytes.resize(bytes);
                }
                device_memory.ReadBlockUnsafe(lower, unlocked_verify_bytes.data(), bytes);
                const u8* const expected = piece.data + (lower - piece.addr);
                batch.verify_bytes += bytes;
                if (std::memcmp(unlocked_verify_bytes.data(), expected, bytes) == 0) [[likely]] {
                    return;
                }
                u64 differ = 0;
                u64 first_differ = bytes;
                for (u64 i = 0; i < bytes; ++i) {
                    if (unlocked_verify_bytes[i] != expected[i]) {
                        ++differ;
                        first_differ = (std::min)(first_differ, i);
                    }
                }
                batch.verify_bad_bytes += differ;
                bad_piece = true;
                if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                    LOG_ERROR(HW_GPU,
                              "Fence copy unlocked verify: {} of {} bytes at {:#x} differ from "
                              "the fence's download (first at {:#x}; copy {:#x}+{:#x})",
                              differ, bytes, lower, lower + first_differ, piece.addr,
                              piece.size);
                }
            });
        });
        if (bad_piece) {
            ++batch.verify_bad_pieces;
        }
    }
}

template <class P>
void BufferCache<P>::PopAsyncBuffersUnlocked(size_t pops, int mode,
                                             Common::LockCensus::FenceCopyBatch& batch) {
    // The popped lists are moved aside and freed after the lock is let go; room made now, out
    // of the lock.
    if (unlocked_trash_ranges.capacity() < pops) {
        unlocked_trash_ranges.reserve((std::max)(pops, size_t{64}));
    }
    if (unlocked_trash_copies.capacity() < pops) {
        unlocked_trash_copies.reserve((std::max)(pops, size_t{64}));
    }
    std::unique_lock lock{mutex};
    u64 held_since = Common::Nx2Hitch::NowNs();
    for (size_t i = 0; i < pops; ++i) {
        PopOneUnlocked(lock, mode, batch, held_since);
    }
    batch.held_ns += Common::Nx2Hitch::NowNs() - held_since;
    lock.unlock();
    unlocked_trash_ranges.clear();
    unlocked_trash_copies.clear();
    batch.pops += pops;
    VideoCore::DownloadLanding::TakeUnlockedWaits(batch.holder_waits, batch.holder_wait_ns);
}

template <class P>
void BufferCache<P>::PopOneUnlocked(std::unique_lock<Common::SpinRecursiveMutex>& lock, int mode,
                                    Common::LockCensus::FenceCopyBatch& batch, u64& held_since) {
    // PopAsyncBuffers with each landing's copy out of the lock. Held on entry and on return.
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    const Common::WriteRing::SiteScope write_site{"buffer download landing (PopAsyncBuffers)", true};
    if constexpr (!IS_OPENGL) {
        if (!read_downloads.empty()) [[unlikely]] {
            (void)runtime.IsFreeNow(read_downloads.front().tick);
            WriteCompletedReadDownloads();
        }
    }
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    // References to the fronts stay good while the lock is let go: only this thread pops these
    // deques, and a push at the back moves no element. They are read under the lock only; the
    // copy outside it uses unlocked_pieces alone.
    auto& downloads = pending_downloads.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    std::vector<std::pair<u64, u64>> stats_landed;
    // FRITCHII_LANDING_CHECK: this fence's snapshot (checked under the lock, before each copy).
    const LandingSnapshot* const landing_snapshot =
        LandingCheckMode() != 0 && !landing_snapshots.empty() ? &landing_snapshots.front()
                                                              : nullptr;
    if (landing_snapshot != nullptr) [[unlikely]] {
        LandingCheckPop(true);
    }
    // Below this many bytes a landing is copied under the lock as before: letting the lock go
    // and taking it back costs more than such a copy (and E may take it in between). The
    // verify mode copies every landing outside the lock.
    constexpr u64 UnlockedMinBytes = 512;
    size_t first = 0;
    while (first < downloads.size()) {
        // PopAsyncBuffers' landings: copies within a page of each other together.
        DAddr landing_begin = static_cast<DAddr>(downloads[first].src_offset);
        DAddr landing_end = landing_begin + downloads[first].size;
        size_t last = first + 1;
        for (; last < downloads.size(); ++last) {
            const DAddr begin = static_cast<DAddr>(downloads[last].src_offset);
            const DAddr end = begin + downloads[last].size;
            if (end + Core::DEVICE_PAGESIZE < landing_begin ||
                begin > landing_end + Core::DEVICE_PAGESIZE) {
                break;
            }
            landing_begin = (std::min)(landing_begin, begin);
            landing_end = (std::max)(landing_end, end);
        }
        // What the fence still writes (PopAsyncBuffers' ForEachInRange), read under the lock: a
        // writer that takes a range away later waits for the copy (ClearFenceDownloads).
        const Common::RangeSet<DAddr>& ranges = pending_download_ranges.front();
        unlocked_pieces.clear();
        landing_check_piece_copies.clear();
        u64 bytes = 0;
        for (size_t index = first; index < last; ++index) {
            const auto& copy = downloads[index];
            const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
            const u8* const read_mapped_memory = base + (copy.dst_offset - base_offset);
            ranges.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end) {
                if (landing_snapshot != nullptr) [[unlikely]] {
                    landing_check_piece_copies.push_back(index);
                }
                unlocked_pieces.push_back(
                    UnlockedPiece{start, end - start, read_mapped_memory + (start - device_addr)});
                bytes += end - start;
            });
        }
        if (unlocked_pieces.empty()) {
            // Nothing left to write (PopAsyncBuffers' landing writes nothing either).
            first = last;
            continue;
        }
        // FRITCHII_LANDING_CHECK: each piece, once the collected CPU writes are known.
        const auto check_pieces = [&] {
            for (size_t piece = 0;
                 piece < unlocked_pieces.size() && piece < landing_check_piece_copies.size();
                 ++piece) {
                const size_t index = landing_check_piece_copies[piece];
                LandingCheckPiece(*landing_snapshot, downloads[index], index,
                                  unlocked_pieces[piece].addr,
                                  const_cast<u8*>(unlocked_pieces[piece].data),
                                  unlocked_pieces[piece].size, &unlocked_cpu_written);
            }
        };
        if (bytes < UnlockedMinBytes && mode != 2) {
            const VideoCore::DownloadLanding landing{landing_begin, landing_end};
            PendingCpuWritesIn(landing_begin, landing_end, unlocked_cpu_written);
            if (landing_snapshot != nullptr) [[unlikely]] {
                check_pieces();
            }
            for (const UnlockedPiece& piece : unlocked_pieces) {
                WriteDownloadedMemory(unlocked_cpu_written, piece.addr, piece.data, piece.size);
                gpu_modified_ranges.Subtract(piece.addr, piece.size);
                if (DownloadReadStats::g_enabled) [[unlikely]] {
                    stats_landed.emplace_back(piece.addr, piece.addr + piece.size);
                }
            }
            ++batch.groups_locked;
            first = last;
            continue;
        }
        // The landing and the collected CPU writes under the lock, in PopAsyncBuffers' order
        // (published, then looked at): a write collected later stores its bytes once the landing
        // ended, which is after the copy.
        VideoCore::DownloadLanding::PublishUnlocked(landing_begin, landing_end);
        const bool any_cpu = PendingCpuWritesIn(landing_begin, landing_end, unlocked_cpu_written);
        if (landing_snapshot != nullptr) [[unlikely]] {
            check_pieces();
        }
        // FRITCHII_STORE_FOOTPRINT=2: the unlocked landing's pieces, checked under the lock
        if (footprint_following && StoreFootprintLive::Mode() == 2) [[unlikely]] {
            for (const UnlockedPiece& piece : unlocked_pieces) {
                FootprintCheckLanding(any_cpu ? &unlocked_cpu_written : nullptr, piece.addr,
                                      piece.data, piece.size);
            }
        }
        const u64 unlock_at = Common::Nx2Hitch::NowNs();
        batch.held_ns += unlock_at - held_since;
        lock.unlock();
        for (const UnlockedPiece& piece : unlocked_pieces) {
            WriteDownloadedMemoryUnlocked(any_cpu ? &unlocked_cpu_written : nullptr, piece.addr,
                                          piece.data, piece.size);
        }
        VideoCore::DownloadLanding::EndUnlocked();
        const u64 relock_at = Common::Nx2Hitch::NowNs();
        batch.unlocked_ns += relock_at - unlock_at;
        lock.lock();
        held_since = Common::Nx2Hitch::NowNs();
        batch.relock_ns += held_since - relock_at;
        ++batch.groups_unlocked;
        batch.pieces += unlocked_pieces.size();
        batch.bytes += bytes;
        // In guest memory now. The GPU-modified state goes where the fence still writes: a range
        // taken away meanwhile was taken by a newer write, which set its own state (a GPU write
        // keeps the range GPU-modified, a CPU write or a synchronous download cleared it).
        PromoteAhead(); // a reader of the real sets
        const Common::RangeSet<DAddr>& live = pending_download_ranges.front();
        if (mode == 2) [[unlikely]] {
            VerifyUnlockedLanding(landing_begin, landing_end, live, batch);
        }
        for (const UnlockedPiece& piece : unlocked_pieces) {
            live.ForEachInRange(piece.addr, piece.size, [&](DAddr start, DAddr end) {
                gpu_modified_ranges.Subtract(start, end - start);
                if (DownloadReadStats::g_enabled) [[unlikely]] {
                    stats_landed.emplace_back(start, end);
                }
            });
        }
        first = last;
    }
    if (DownloadReadStats::g_enabled) [[unlikely]] {
        DownloadReadStats::Landed(stats_landed);
    }
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    unlocked_trash_copies.emplace_back(std::move(pending_downloads.front()));
    pending_downloads.pop_front();
    unlocked_trash_ranges.emplace_back(std::move(pending_download_ranges.front()));
    pending_download_ranges.pop_front();
    if (LandingCheckMode() != 0 && !landing_snapshots.empty()) [[unlikely]] {
        landing_snapshots.pop_front();
        LandingCheckPop(false);
    }
    {
        // EXP72: in guest memory now (the copies above wrote it).
        std::scoped_lock landed_lock{fence_landed_mutex};
        fence_downloads_landed.fetch_add(1, std::memory_order_release);
    }
    fence_landed_cv.notify_all();
}

template <class P>
bool BufferCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    VideoCommon::MemoryStageLink::CheckBackCache("IsRegionGpuModified");
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: the back's question (GPUMem's: GpuWrittenHere)
    // One lookup instead of visiting every range inside (~4% of the GPU thread for title 1's
    // inline uploads once GPU-written ranges are no longer cleared by downloads).
    return gpu_modified_ranges.Intersects(addr, size);
}

template <class P>
template <typename Func>
bool BufferCache<P>::ForEachComputeWriteMark(Func&& func) const {
    // As BindHostComputeStorageBuffers marks: min(the cap, the native write mark).
    bool bounded = true;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers &
                          channel_state->written_compute_storage_buffers,
                      [&](u32 index) {
                          const Binding& binding = channel_state->compute_storage_buffers[index];
                          if (binding.device_addr == 0 || binding.size == 0) {
                              return; // a null binding: nothing marked
                          }
                          const bool native =
                              ((channel_state->native_compute_storage_buffers >> index) & 1) != 0;
                          const u64 limit = channel_state->compute_storage_write_marks[index];
                          const u64 capped = WrittenStorageMarkSize(binding.size);
                          if (!native || binding.segment_count != 0 || limit >= binding.size ||
                              (capped < binding.size && capped <= limit)) {
                              bounded = false;
                              return;
                          }
                          if (limit != 0) {
                              func(binding.device_addr, limit);
                          }
                      });
    return bounded;
}

template <class P>
auto BufferCache<P>::ComputeSkipRangeFacts(DAddr addr, u64 size) -> SkipRangeFacts {
    PromoteAhead();
    SkipRangeFacts facts{};
    const DAddr end = addr + size;
    u64 covered = 0;
    ForEachGpuWrittenHere(addr, size, [&](DAddr lower, DAddr upper) {
        const DAddr begin = (std::max)(lower, addr);
        const DAddr stop = (std::min)(upper, end);
        if (begin < stop) {
            covered += stop - begin;
        }
    });
    facts.all_gpu_written = covered >= size;
    facts.cpu_modified = memory_tracker.IsRegionCpuModified(addr, size);
    const DAddr page_begin = Common::AlignDown(addr, Core::DEVICE_PAGESIZE);
    const u64 page_size = Common::AlignUp(end, Core::DEVICE_PAGESIZE) - page_begin;
    facts.read_back = memory_tracker.IsRegionPreflushable(page_begin, page_size) ||
                      cpu_read_ranges.Intersects(page_begin, page_size) ||
                      read_download_ranges.Intersects(page_begin, page_size);
    return facts;
}

template <class P>
bool BufferCache<P>::IsRegionRegistered(DAddr addr, size_t size) {
    const DAddr end_addr = addr + size;
    const u64 page_end = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    for (u64 page = addr >> CACHING_PAGEBITS; page < page_end;) {
        const BufferId buffer_id = page_table[page];
        if (!buffer_id) {
            ++page;
            continue;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buf_start_addr = buffer.CpuAddr();
        const DAddr buf_end_addr = buf_start_addr + buffer.SizeBytes();
        if (buf_start_addr < end_addr && addr < buf_end_addr) {
            return true;
        }
        page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    }
    return false;
}

template <class P>
bool BufferCache<P>::IsRegionCpuModified(DAddr addr, size_t size) {
    return memory_tracker.IsRegionCpuModified(addr, size);
}

template <class P>
void BufferCache<P>::BindHostIndexBuffer() {
    Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
    TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
    const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
    const u32 size = channel_state->index_buffer.size;
    const auto& draw_state = Inputs().DrawState();
    MemStageStats::Read(MemStageStats::Kind::Index, size);
    if constexpr (requires { runtime.BindHostUint8IndexBuffer(std::span<const u8>{}); }) {
        if (draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte &&
            draw_state.topology != Maxwell::PrimitiveTopology::Quads &&
            draw_state.topology != Maxwell::PrimitiveTopology::QuadStrip &&
            runtime.WidensUint8IndicesOnHost()) {
            std::span<const u8> indices;
            const auto& index_buffer = draw_state.index_buffer;
            const u32 guest_size = (std::min)(
                static_cast<u32>(index_buffer.EndAddress() - index_buffer.StartAddress()),
                index_buffer.first + index_buffer.count);
            if (!draw_state.inline_index_draw_indexes.empty()) {
                indices = std::span{draw_state.inline_index_draw_indexes.data(), size};
            } else if (guest_size != 0 &&
                       !GpuWrittenHere(channel_state->index_buffer.device_addr, guest_size)) {
                guest_uint8_indices.resize(guest_size);
                VideoCommon::OneFollower::BeforeEngineRead(channel_state->index_buffer.device_addr, guest_size);
                device_memory.ReadBlockUnsafe(channel_state->index_buffer.device_addr,
                                              guest_uint8_indices.data(), guest_size);
                indices = guest_uint8_indices;
            }
            if (!indices.empty()) {
                runtime.BindHostUint8IndexBuffer(indices);
                return;
            }
        }
    }
    if (draw_state.inline_index_draw_indexes.empty()) {
        if (geom_keep != 0) [[unlikely]] {
            SynchronizeGeometry(buffer, channel_state->index_buffer.device_addr, size,
                                NUM_VERTEX_BUFFERS);
        } else {
            SynchronizeBuffer(buffer, channel_state->index_buffer.device_addr, size);
        }
    } else {
        if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
            auto upload_staging = runtime.UploadStagingBuffer(size);
            std::array<BufferCopy, 1> copies{{BufferCopy{.src_offset = upload_staging.offset, .dst_offset = 0, .size = size}}};
            std::memcpy(upload_staging.mapped_span.data(), draw_state.inline_index_draw_indexes.data(), size);
            runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true);
        } else {
            buffer.ImmediateUpload(0, draw_state.inline_index_draw_indexes);
        }
    }
    if constexpr (HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const u32 new_offset = offset + draw_state.index_buffer.first * u32(draw_state.index_buffer.FormatSizeInBytes());
        runtime.BindIndexBuffer(buffer, new_offset, size);
    } else {
        buffer.MarkUsage(offset, size);
        runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format, draw_state.index_buffer.first, draw_state.index_buffer.count, buffer, offset, size);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size,
                                          u32 stride) {
    if constexpr (IS_OPENGL) {
        runtime.BindVertexBuffer(index, buffer, offset, size, stride);
    } else {
        runtime.BindVertexBuffer(index, buffer.Handle(), offset, size, stride);
    }
}

template <class P>
Binding& BufferCache<P>::VertexBufferSlot(u32 index) {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
const Binding& BufferCache<P>::VertexBufferSlot(u32 index) const {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
void BufferCache<P>::UpdateVertexBufferSlot(u32 index, const Binding& binding) {
    Binding& slot = VertexBufferSlot(index);
    if (slot.device_addr != binding.device_addr || slot.size != binding.size) {
        ++vertex_buffers_serial;
    }
    slot = binding;
    const u32 bit = 1u << index;
    if (binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
        enabled_vertex_buffers_mask |= bit;
    } else {
        // A stream going from a binding to none (switched off, or its buffer deleted) is not
        // bound again, so the host's remembered binding would outlive the buffer: forgotten at
        // the next BindHostVertexBuffers, in the draw's own stream of binds.
        forget_vertex_buffers_mask |= enabled_vertex_buffers_mask & bit;
        enabled_vertex_buffers_mask &= ~bit;
    }
}

template <class P>
void BufferCache<P>::MarkVertexBuffersStale(DAddr device_addr, u64 size) {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = enabled_vertex_buffers_mask;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            const Binding& binding = v_buffer[index];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                stale_vertex_buffers_mask |= 1U << index;
            }
        }
    }
}

template <class P>
bool BufferCache<P>::IsRepackedVertexBuffer(u32 index) {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        return runtime.IsVertexRepacked(index);
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::RefreshRepackedVertexBuffers() {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = std::exchange(stale_vertex_buffers_mask, 0) & enabled_vertex_buffers_mask;
        auto& flags = Inputs().Dirty();
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            if (runtime.IsVertexRepacked(index)) {
                flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffers() {
    RefreshRepackedVertexBuffers();
    if (forget_vertex_buffers_mask != 0) [[unlikely]] {
        if constexpr (requires { runtime.ForgetVertexBinding(u32{}); }) {
            for (u32 mask = forget_vertex_buffers_mask; mask != 0; mask &= mask - 1) {
                runtime.ForgetVertexBinding(static_cast<u32>(std::countr_zero(mask)));
            }
        }
        forget_vertex_buffers_mask = 0;
    }

#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        auto& flags = Inputs().Dirty();
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings{};
        u32 last_index = (std::numeric_limits<u32>::max)();
        // FRITCHII_MEMSTAGE_FULL_BINDS (memory stage spec 4.2): the bindings that are not dirty
        // are bound too, in runs of their own and without a usage mark, and the player drops
        // the ones still bound, so the recorded commands stay the same. Repacked ones are left
        // out: binding one again runs its conversion again.
        const bool full_binds = VideoCommon::MemoryStageLink::FullVertexBinds();
        const bool geom_memo = geom_keep != 0; // FRITCHII_FILL_GEOM_KEEP
        bool run_dirty = true;
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            bindings.clean = !run_dirty;
            runtime.BindVertexBuffers(bindings);
            bindings = HostBindings<Buffer>{};
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = std::countr_zero(enabled_mask);
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            MemStageStats::Read(MemStageStats::Kind::Vertex, binding.size);
            if (geom_memo) [[unlikely]] {
                SynchronizeGeometry(buffer, binding.device_addr, binding.size, index);
            } else {
                SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            }
            const bool dirty = flags[Dirty::VertexBuffer0 + index];
            if (!dirty && (!full_binds || IsRepackedVertexBuffer(index))) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = Inputs().VertexStreamStride(index);
            const u32 offset = buffer.Offset(binding.device_addr);
            if (dirty) {
                buffer.MarkUsage(offset, binding.size);
            }
            if (!bindings.buffers.empty() && (index != last_index + 1 || dirty != run_dirty)) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
                run_dirty = dirty;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
    } else {
        HostBindings<typename P::Buffer> host_bindings;
        bool any_valid{false};
        auto& flags = Inputs().Dirty();
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding& binding = channel_state->vertex_buffers[index];
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            MemStageStats::Read(MemStageStats::Kind::Vertex, binding.size,
                                binding.size != 0 ? 1 : 0);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;

            host_bindings.min_index = (std::min)(host_bindings.min_index, index);
            host_bindings.max_index = (std::max)(host_bindings.max_index, index);
            any_valid = true;
        }

        if (any_valid) {
            host_bindings.max_index++;
            for (u32 index = host_bindings.min_index; index < host_bindings.max_index; index++) {
                flags[Dirty::VertexBuffer0 + index] = false;

                const Binding& binding = channel_state->vertex_buffers[index];
                Buffer& buffer = slot_buffers[binding.buffer_id];

                const u32 stride = Inputs().VertexStreamStride(index);
                const u32 offset = buffer.Offset(binding.device_addr);
                buffer.MarkUsage(offset, binding.size);

                host_bindings.buffers.push_back(&buffer);
                host_bindings.offsets.push_back(offset);
                host_bindings.sizes.push_back(binding.size);
                host_bindings.strides.push_back(stride);
            }
            runtime.BindVertexBuffers(host_bindings);
        }
    }
}

template <class P>
void BufferCache<P>::BindHostDrawIndirectBuffers() {
    const auto bind_buffer = [this](const Binding& binding) {
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        MemStageStats::Read(MemStageStats::Kind::Indirect, binding.size);
        SynchronizeBuffer(buffer, binding.device_addr, binding.size);
        // The draw reads the arguments (GetDrawIndirectBuffer, GetDrawIndirectCount).
        if (binding.buffer_id != NULL_BUFFER_ID) {
            buffer.MarkUsage(buffer.Offset(binding.device_addr), binding.size);
        }
    };
    if (Inputs().DrawIndirect()->include_count) {
        bind_buffer(channel_state->count_buffer_binding);
    }
    bind_buffer(channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffers(size_t stage) {
    u32 dirty = ~0U;
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        dirty = std::exchange(channel_state->dirty_uniform_buffers[stage], 0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        const bool needs_bind = ((dirty >> index) & 1) != 0;
        BindHostGraphicsUniformBuffer(stage, index, binding_index, needs_bind);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            ++binding_index;
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind) {
    ++channel_state->uniform_cache_shots[0];
    // Bank 24 (root table 0, NVN2's push constants) is read from bank 17, which such games do not
    // bind: stream the table's current words as that buffer.
    if (index == 17 && Inputs().RootTables().loaded && channel_state->uniform_buffers[stage][index].size == 0) {
        const auto& words = Inputs().RootTables().tables[0];
        const u32 size = (*channel_state->uniform_buffer_sizes)[stage][index];
        const u32 copied = (std::min)(size, static_cast<u32>(sizeof(words)));
        if constexpr (IS_OPENGL) {
            std::array<u8, sizeof(words)> data{};
            std::memcpy(data.data(), words.data(), copied);
            runtime.BindFastUniformBuffer(stage, binding_index, copied);
            runtime.PushFastUniformBuffer(stage, binding_index, std::span<const u8>(data.data(), copied));
        } else {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            std::memset(span.data(), 0, span.size());
            std::memcpy(span.data(), words.data(), copied);
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformOther, 1);
        Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformBytes, size);
        return;
    }
    // FRITCHII_BIND_SKIP (live, default 0): off, one member test.
    if constexpr (!IS_OPENGL) {
        if (bind_skip.mode != 0) [[unlikely]] {
            BindSkipUniform(stage, index, binding_index, needs_bind);
            return;
        }
    }
    BindHostGraphicsUniformBufferBody<false>(stage, index, binding_index, needs_bind, nullptr);
}

template <class P>
template <bool record>
void BufferCache<P>::BindHostGraphicsUniformBufferBody(size_t stage, u32 index,
                                                       u32 binding_index, bool needs_bind,
                                                       [[maybe_unused]] BindSkipOutcome* outcome) {
    const Binding& binding = channel_state->uniform_buffers[stage][index];
    const DAddr device_addr = binding.device_addr;
    const u32 size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    Buffer& buffer = slot_buffers[binding.buffer_id];
    TouchBuffer(buffer, binding.buffer_id);
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    const u32 offset = has_host_buffer ? buffer.Offset(device_addr) : 0;
    const bool needs_alignment_stream = [&]() {
        if constexpr (IS_OPENGL) {
            return false;
        } else {
            if (!has_host_buffer) {
                return false;
            }
            const u32 alignment = runtime.GetUniformBufferAlignment();
            return alignment > 1 && (offset % alignment) != 0;
        }
    }();
    // Native programs keep tables next to their code in a 64 KiB bank read with indirect offsets
    // (Splatoon Raiders: the dialogue box corners in c1). Writes to that memory can go unnoticed by
    // the memory tracker, leaving stale host contents, so check such a bank on every bind: games
    // rewrite per-draw data between draws, and checking once per frame left it stale for a frame
    // now and then (white flashes in Ocarina of Time).
    // Decided before the bank check, which changes nothing it reads (marking pages CPU-modified
    // leaves their GPU bits alone).
    // EDEN_NX2_RUNAHEAD: behind the run-ahead frontier guest memory may already hold the next
    // frame's bytes; the host buffer, which the run-ahead chunks keep exact, is bound instead of
    // a stream of guest memory. A misaligned bind still streams (counted as a risk).
    if (needs_alignment_stream) {
        (void)VideoCommon::Runahead::BackHolds(VideoCommon::Runahead::Hold::AlignedStream);
    }
    const bool use_fast_buffer = needs_alignment_stream
        || (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size
            && !memory_tracker.IsRegionGpuModified(device_addr, size)
            && !VideoCommon::Runahead::BackHolds(VideoCommon::Runahead::Hold::FastUniform));
    const bool native_bank_bind = has_host_buffer &&
                                  (*channel_state->uniform_buffer_sizes)[stage][index] == 0x10000 &&
                                  size != 0 && gpu_memory != nullptr &&
                                  gpu_memory->UsesNativeTiling() &&
                                  !memory_tracker.IsRegionGpuModified(device_addr, size);
    // EDEN_NX2_EXP26 & 1 (the original's EDEN_NX2_BUFCACHE_BANK): a bind that streams the
    // buffer skips the check. The stream copies guest memory itself; the write flags the check
    // would take stay set for the next bind that reads the host buffer.
    const int bank_trims = native_bank_bind ? VideoCore::Nx2Debug::Experiment(26) : 0;
    const bool native_bank = native_bank_bind && !(use_fast_buffer && (bank_trims & 1) != 0);
    if constexpr (record) {
        // FRITCHII_BIND_SKIP: the path, filled in below (a path the skip cannot stand in for
        // leaves Kind::None).
        outcome->bank = native_bank;
    }
    // FRITCHII_S4_COUNT (dry): the bind before anything below reads its bytes.
    if (Common::S4Count::On()) [[unlikely]] {
        S4CountBind(stage, index, device_addr, size, use_fast_buffer, native_bank);
    }
    // EDEN_NX2_MEMSTAGE_STATS: what this bind reads (the stream copies guest memory; the cached
    // path synchronizes the host buffer; the bank check compares the bank's pages).
    if (MemStageStats::Enabled()) [[unlikely]] {
        MemStageStats::Read(use_fast_buffer ? MemStageStats::Kind::UniformStream
                                            : MemStageStats::Kind::UniformCached,
                            size);
        if (native_bank) {
            MemStageStats::Read(MemStageStats::Kind::BankCheck, size);
        }
    }
    // TEST (EDEN_NX2_GPU_PROFILE=2): the bank check below, only for the binds that run it.
    std::optional<Nx2Profile::Scope> bank_scope;
    if (native_bank && Nx2Profile::Fine()) [[unlikely]] {
        bank_scope.emplace(Nx2Profile::Zone::UniformBank, true, true);
    }
    // EDEN_NX2_RUNAHEAD: behind the frontier the bank's guest pages may hold the next frame (the
    // chunks put this draw's bytes in the host buffer and the snapshots): compared again at a
    // bind past it (the write flags stay set until then).
    if (native_bank &&
        VideoCommon::Runahead::BackHolds(VideoCommon::Runahead::Hold::BankCompare)) {
        // Held: nothing compared, nothing taken.
        if constexpr (record) {
            outcome->held = true;
        }
    } else if (const int tracked_mode = native_bank ? VideoCore::Nx2Debug::Experiment(9) : 0;
        tracked_mode != 0) {
        if constexpr (record) {
            outcome->bank_mode = tracked_mode;
        }
        CheckNativeBankPagesTracked(device_addr, size, tracked_mode, bank_trims, buffer, stage,
                                    index);
    } else if (native_bank) {
        // The bank size is only an upper limit: stop at the end of the mapped memory.
        u64 mapped_size = 0;
        while (mapped_size < size &&
               device_memory.template GetPointer<u8>(device_addr + mapped_size) != nullptr) {
            mapped_size += Core::DEVICE_PAGESIZE - ((device_addr + mapped_size) & Core::DEVICE_PAGEMASK);
        }
        const u64 check_size = (std::min<u64>)(mapped_size, size);
        // Banks slide through larger buffers, a new address per draw: compare each page with
        // its last contents and mark only the pages that changed.
        SelectNativeBankStore();
        const DAddr first_page = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr end = device_addr + check_size;
        // EDEN_NX2_EXP4 (experiment, 0 = off): & 1 compares only the bytes of each page the
        // binding covers; & 2 compares each byte at most once per draw. The guest writes a
        // draw's data before it submits the draw, so comparing again within the same draw only
        // finds writes meant for later draws, which their own binds check. A changed page is
        // still copied and marked whole.
        const int cost_mode = VideoCore::Nx2Debug::Experiment(4);
        const bool trim = (cost_mode & 1) != 0;
        const bool once_per_draw = (cost_mode & 2) != 0;
        auto& cost = native_bank_stats;
        if (cost_mode != 0) {
            ++cost.binds;
        }
        // EDEN_NX2_RUNAHEAD: once the frontier moved this session, a changed page goes up from
        // its snapshot now: a CPU-modified mark would read guest memory after a later boundary.
        Buffer* const bank_upload_to =
            (bank_trims & 4) != 0 ||
                    VideoCommon::Runahead::g_frontier.load(std::memory_order_relaxed) != 0
                ? &buffer
                : nullptr;
        NativeBankUploads bank_uploads;
        for (DAddr page = first_page; page < end; page += Core::DEVICE_PAGESIZE) {
            VideoCommon::OneFollower::BeforeEngineRead(page, Core::DEVICE_PAGESIZE);
            const u8* const guest = device_memory.template GetPointer<u8>(page);
            if (guest == nullptr) {
                continue;
            }
            const u32 begin =
                trim ? static_cast<u32>((std::max)(page, device_addr) - page) : 0;
            const u32 stop = trim ? static_cast<u32>((std::min<DAddr>)(page + Core::DEVICE_PAGESIZE,
                                                                      end) - page)
                                  : static_cast<u32>(Core::DEVICE_PAGESIZE);
            const NativeBankSnapshotRef ref = NativeBankSnapshot(page);
            const bool inserted = ref.inserted;
            NativeBankPage& snapshot = *ref.page;
            if (once_per_draw && !inserted && snapshot.checked_draw == native_bank_draw &&
                snapshot.checked_begin <= begin && stop <= snapshot.checked_end) {
                ++cost.pages_skipped;
                continue;
            }
            // EDEN_NX2_EXP26 & 2: an inline SSE2 compare instead of the ucrt memcmp.
            const bool same =
                !inserted && ((bank_trims & 2) != 0
                                  ? VideoCommon::Nx2BytesEqual(snapshot.words.data() + begin,
                                                               guest + begin, stop - begin)
                                  : std::memcmp(snapshot.words.data() + begin, guest + begin,
                                                stop - begin) == 0);
            if (cost_mode != 0 && !inserted) {
                ++cost.pages_compared;
                cost.bytes_compared += stop - begin;
                cost.bytes_trimmed += Core::DEVICE_PAGESIZE - (stop - begin);
            }
            if (once_per_draw && same) {
                if (snapshot.checked_draw != native_bank_draw ||
                    begin > snapshot.checked_end || stop < snapshot.checked_begin) {
                    // Disjoint from what this draw compared before: keep the larger range.
                    if (snapshot.checked_draw != native_bank_draw ||
                        stop - begin > snapshot.checked_end - snapshot.checked_begin) {
                        snapshot.checked_begin = begin;
                        snapshot.checked_end = stop;
                    }
                } else {
                    snapshot.checked_begin = (std::min)(snapshot.checked_begin, begin);
                    snapshot.checked_end = (std::max)(snapshot.checked_end, stop);
                }
                snapshot.checked_draw = native_bank_draw;
            }
            // EDEN_NX2_MME_DMA_STATS: how many checked pages had changed.
            if (static const bool stats = std::getenv("EDEN_NX2_MME_DMA_STATS") != nullptr;
                stats) {
                static std::array<u64, 3> counts{};
                static std::array<u64, 4> changed_state{};
                ++counts[same ? 0 : (inserted ? 1 : 2)];
                if (!same && !inserted) {
                    // Did the tracker already know (CPU-modified), or was the page clean, and
                    // was it GPU-modified?
                    const bool known =
                        memory_tracker.IsRegionCpuModified(page, Core::DEVICE_PAGESIZE);
                    const bool gpu =
                        memory_tracker.IsRegionGpuModified(page, Core::DEVICE_PAGESIZE);
                    ++changed_state[(known ? 1 : 0) | (gpu ? 2 : 0)];
                }
                if (((counts[0] + counts[1] + counts[2]) & 0x3ffff) == 0) {
                    LOG_INFO(HW_GPU,
                             "Native bank pages checked: same {}, new {}, changed {} (tracker "
                             "clean {}, CPU-modified {}, GPU {}, both {})",
                             counts[0], counts[1], counts[2], changed_state[0],
                             changed_state[1], changed_state[2], changed_state[3]);
                }
            }
            if (same) {
                continue;
            }
            std::memcpy(snapshot.words.data(), guest, Core::DEVICE_PAGESIZE);
            NoteNativeBankPageChanged(page, snapshot.words.data(), bank_upload_to, bank_uploads);
            if (once_per_draw) {
                snapshot.checked_draw = native_bank_draw;
                snapshot.checked_begin = 0;
                snapshot.checked_end = static_cast<u32>(Core::DEVICE_PAGESIZE);
            }
            if (cost_mode != 0 && !inserted) {
                ++cost.pages_changed;
            }
        }
        if (bank_uploads.count != 0) {
            FlushNativeBankUploads(buffer, bank_uploads);
        }
        if (cost_mode != 0) {
            const auto now = std::chrono::steady_clock::now();
            if (now - cost.last_log >= std::chrono::seconds{5}) {
                LOG_INFO(HW_GPU,
                         "Native bank check (EXP4={}, 5 s): {} binds, {} pages compared ({} KiB), "
                         "{} KiB outside the bindings not compared, {} pages skipped (compared "
                         "earlier in the draw), {} pages changed",
                         cost_mode, cost.binds, cost.pages_compared, cost.bytes_compared / 1024,
                         cost.bytes_trimmed / 1024, cost.pages_skipped, cost.pages_changed);
                cost = {};
                cost.last_log = now;
            }
        }
    }
    bank_scope.reset();
    Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformBytes, size);
    if (use_fast_buffer) {
        MemReadLog::Bytes(MemReadLog::Kind::Uniform, device_memory, device_addr, size);
        const Nx2Profile::FineScope stream_scope{Nx2Profile::Zone::UniformStream};
        if constexpr (IS_OPENGL) {
            if (runtime.HasFastBufferSubData()) {
                // Fast path for Nvidia
                const bool should_fast_bind =
                    !HasFastUniformBufferBound(stage, binding_index) ||
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
                if (should_fast_bind) {
                    // We only have to bind when the currently bound buffer is not the fast version
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                    runtime.BindFastUniformBuffer(stage, binding_index, size);
                }
                const auto span = ImmediateBufferWithData(device_addr, size);
                runtime.PushFastUniformBuffer(stage, binding_index, span);
                Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformOther, 1);
                return;
            }
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        // fable_b 14.4 cut 7, dry (GPU profile on): the price verdict splits the unflagged binds
        // by whether a guest store on their pages would trap and raise a flag at all.
        if (FritchiiPrice::On()) [[unlikely]] {
            FritchiiPrice::t_uniform_cpu_modified =
                memory_tracker.IsRegionCpuModified(device_addr, size);
        }
        // Stream buffer path to avoid stalling on non-Nvidia drivers or Vulkan
        bool streamed = false;
        if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
            // The same bytes as this slot's previous bind, in the same command buffer: bind that
            // stream allocation again instead of allocating and copying (uniform_stream_reuse.h).
            // EDEN_NX2_UBOREUSE=0 turns it off. EDEN_NX2_EXP17=1: any stage or slot's copy of
            // the same range is bound again too.
            if (VideoCommon::UniformStreamReuseEnabled()) {
                // TEST: rebinds counted (EDEN_NX2_GPU_PROFILE=2).
                const u64 rebinds =
                    VideoCommon::UniformStreamReuseState<typename Runtime::StreamUniformRef>()
                        .rebinds;
                if (const int keep_mode = VideoCore::Nx2Debug::Experiment(92);
                    keep_mode != 0 || uniform_keeps.mode != 0) [[unlikely]] {
                    // EXP92: uniform keeps (a change to 0 drops them, then today's path). Not on
                    // native bank binds or behind a run-ahead frontier.
                    streamed = BindSmallUniformKeeping(
                        stage, index, binding_index, device_addr, size,
                        VideoCore::Nx2Debug::Experiment(17) != 0,
                        !native_bank_bind &&
                            VideoCommon::Runahead::g_frontier.load(std::memory_order_relaxed) == 0,
                        keep_mode);
                } else if (VideoCore::Nx2Debug::Experiment(91) != 0) [[unlikely]] {
                    // EXP91 (stage 0): the same call, counted (binds nothing differently).
                    streamed = BindSmallUniformCountingKeeps(
                        stage, index, binding_index, device_addr, size,
                        VideoCore::Nx2Debug::Experiment(17) != 0);
                } else {
                    // FRITCHII_ONEF_UBO_KEEP (fable_b 25.2 cut a): the price line's cut class
                    // bound again whole; =1 verifies the proof against the compare.
                    bool provable = false;
                    streamed = UniformKeepBefore(stage, index, device_addr, size,
                                                 native_bank_bind, provable);
                    if (!streamed) {
                        using Ref = typename Runtime::StreamUniformRef;
                        const u32 serial_before =
                            provable ? VideoCommon::UniformSlotSerial<Ref>(stage, index) : 0;
                        streamed = VideoCommon::BindSmallUniformReusing(
                            runtime, device_memory, stage, index, binding_index, device_addr,
                            size, VideoCore::Nx2Debug::Experiment(17) != 0);
                        if (provable && serial_before != 0 &&
                            VideoCommon::UniformSlotSerial<Ref>(stage, index) != serial_before)
                            [[unlikely]] {
                            ++uniform_keep_stats.stale;
                            ++uniform_keep_stats.stale_session;
                        }
                    }
                    if constexpr (record) {
                        // FRITCHII_BIND_SKIP: the reuse slot holds this bind's bytes now.
                        if (streamed) {
                            outcome->kind = BindSkipOutcome::Kind::Stream;
                        }
                    }
                }
                const u64 rebound =
                    VideoCommon::UniformStreamReuseState<typename Runtime::StreamUniformRef>()
                        .rebinds -
                    rebinds;
                Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformRebinds, rebound);
                Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformStreamed,
                                    streamed ? 1 - rebound : 0);
            }
        }
        if (!streamed) {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            VideoCommon::OneFollower::BeforeEngineRead(device_addr, size);
            device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
            Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformStreamed, 1);
        }
        return;
    }
    Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformCached, 1);
    const Nx2Profile::FineScope classic_scope{Nx2Profile::Zone::UniformClassic};
    // Classic cached path
    const bool synced = SynchronizeBuffer(buffer, device_addr, size);
    if (synced) {
        ++channel_state->uniform_cache_hits[0];
    }
    if constexpr (record) {
        outcome->kind = BindSkipOutcome::Kind::Cached;
        outcome->synced = synced;
    }
    BindUniformCachedTail(stage, index, binding_index, needs_bind, buffer, offset, size);
}

template <class P>
void BufferCache<P>::BindUniformCachedTail(size_t stage, [[maybe_unused]] u32 index,
                                           u32 binding_index, bool needs_bind, Buffer& buffer,
                                           u32 offset, u32 size) {
    // Skip binding if it's not needed and if the bound buffer is not the fast version
    // This exists to avoid instances where the fast buffer is bound and a GPU write happens
    needs_bind |= HasFastUniformBufferBound(stage, binding_index);
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        needs_bind |= channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
    }
    if (!needs_bind) {
        return;
    }
    if constexpr (IS_OPENGL) {
        // Mark the index as dirty if offset doesn't match
        const bool is_copy_bind = offset != 0 && !runtime.SupportsNonZeroUniformOffset();
        channel_state->dirty_uniform_buffers[stage] |= (is_copy_bind ? 1U : 0U) << index;
    }
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
    }
    buffer.MarkUsage(offset, size);
    if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
        runtime.BindUniformBuffer(stage, binding_index, buffer, offset, size);
    } else {
        runtime.BindUniformBuffer(buffer, offset, size);
    }
    channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
}

template <class P>
bool BufferCache<P>::UniformKeepBefore(size_t stage, u32 index, DAddr device_addr, u32 size,
                                       bool bank, bool& provable) {
    provable = false;
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_onef_ubo_keep,
                                                      "FRITCHII_ONEF_UBO_KEEP");
    if (mode == 0) [[likely]] {
        return false;
    }
    // The write generations count only while they are on: from the first keep test.
    if (!Common::g_price_gens_on.load(std::memory_order_relaxed)) [[unlikely]] {
        Common::g_price_gens_on.store(true, std::memory_order_relaxed);
    }
    UniformKeepStats& stats = uniform_keep_stats;
    ++stats.binds;
    UniformKeepLog(mode);
    // Every writer of these bytes raises a generation on their pages: our constant buffer words
    // and every buffer cache write (g_buffer_write_pages, always), the guest's stores on pages
    // the tracker holds clean (they trap: g_collected_write_pages, again after the bytes landed
    // while the bank marks are on). A CPU-modified page lets guest stores through unseen, a
    // GPU-modified range never streams, a native bank bind or a run-ahead frontier keeps its own
    // rules: no proof then.
    const bool usable = !bank && Common::BankMarks() &&
                        VideoCommon::Runahead::g_frontier.load(std::memory_order_relaxed) == 0;
    const u64 sum = usable ? Common::PriceGenSum(device_addr, size) : ~u64{0};
    const bool tracked = usable && !memory_tracker.IsRegionCpuModified(device_addr, size);
    UniformKeepMemo& memo = uniform_keep_memo[stage][index];
    provable = memo.valid && memo.tracked && tracked && memo.device_addr == device_addr &&
               memo.size == size && sum != ~u64{0} && memo.sum == sum;
    memo = UniformKeepMemo{device_addr, sum, size, tracked, usable && sum != ~u64{0}};
    if (!provable) {
        return false;
    }
    ++stats.provable;
    if (mode < 2) {
        return false; // =1: the usual bind runs and its result is checked
    }
    if (!VideoCommon::RebindUniformSlotCopy(runtime, stage, index, size)) {
        return false; // the copy is not this epoch's (or not the slot's): the usual bind
    }
    ++stats.kept;
    return true;
}

template <class P>
void BufferCache<P>::UniformKeepLog(int mode) {
    UniformKeepStats& stats = uniform_keep_stats;
    if ((stats.binds & 4095) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (stats.since == std::chrono::steady_clock::time_point{}) {
        stats.since = now;
        return;
    }
    if (now - stats.since < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - stats.since).count();
    LOG_INFO(HW_GPU,
             "Uniform keep (fable_b 25.2 cut a, FRITCHII_ONEF_UBO_KEEP={}) {:.0f} s: streamed "
             "binds {:.0f}/s, provable (same range, unflagged, pages tracked) {:.0f}/s, bound "
             "again without the read {:.0f}/s; provable but other bytes {:.0f}/s (must be 0; {} "
             "this session)",
             mode, seconds, static_cast<double>(stats.binds) / seconds,
             static_cast<double>(stats.provable) / seconds,
             static_cast<double>(stats.kept) / seconds, static_cast<double>(stats.stale) / seconds,
             stats.stale_session);
    const u64 session = stats.stale_session;
    stats = UniformKeepStats{};
    stats.stale_session = session;
    stats.since = now;
}

template <class P>
void BufferCache<P>::S4CountBind(size_t stage, u32 index, DAddr device_addr, u32 size,
                                 bool stream, bool bank) {
    if (size == 0 || device_addr == 0) {
        return;
    }
    VideoCommon::S4Count::BindFacts facts;
    facts.device_addr = device_addr;
    facts.size = size;
    facts.stage = static_cast<u32>(stage);
    facts.slot = index;
    facts.stream = stream;
    facts.bank = bank;
    facts.cpu_modified = memory_tracker.IsRegionCpuModified(device_addr, size);
    facts.gpu_modified = memory_tracker.IsRegionGpuModified(device_addr, size);
    const VideoCommon::S4Count::PageTest cpu_page = [](void* self, u64 page) {
        return static_cast<BufferCache*>(self)->memory_tracker.IsRegionCpuModified(
            page, Core::DEVICE_PAGESIZE);
    };
    const VideoCommon::S4Count::PageTest gpu_page = [](void* self, u64 page) {
        return static_cast<BufferCache*>(self)->memory_tracker.IsRegionGpuModified(
            page, Core::DEVICE_PAGESIZE);
    };
    VideoCommon::S4Count::Bind(device_memory, facts, cpu_page, gpu_page, this);
}

template <class P>
void BufferCache<P>::BindSkipPoll() {
    if constexpr (IS_OPENGL) {
        return; // the skip stands in for Vulkan's stream copies only
    } else {
        const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bind_skip,
                                                           "FRITCHII_BIND_SKIP");
        const int mode = value >= 1 && value <= 5 ? value : 0;
        if (mode != bind_skip.mode) [[unlikely]] {
            BindSkipSetMode(mode);
        }
    }
}

template <class P>
void BufferCache<P>::BindSkipSetMode(int mode) {
    BindSkipState& s = bind_skip;
    const bool was_on = s.mode != 0;
    s.mode = mode;
    s.step2 = mode == 2 || mode == 3 || mode == 5;
    s.verify = mode == 3 || mode == 4;
    s.dry = mode == 5;
    // No memo from before is trusted (what a memo holds depends on the value: the hash).
    ++s.epoch;
    if (mode != 0 && !was_on) {
        // The marks the proof stands on: the price generations (they stay on, as their other
        // users leave them), the bank marks gate, the tracker's hooks, the front's list entry
        // marks. A writer that loaded a gate shut set no mark: no memo is made for two frames
        // (every writer's next load sees the gates by then), as the bank check does.
        Common::g_price_gens_on.store(true, std::memory_order_seq_cst);
        Common::SetBankMarks(Common::BankMarksBindSkip, true);
        Common::BindSkip::g_on.store(true, std::memory_order_seq_cst);
        VideoCommon::S4Count::SetBindSkipEntries(true);
        s.ready_frame = frame_tick + 2;
        s.overflows = Common::g_collected_overflows.load(std::memory_order_seq_cst);
    } else if (mode == 0 && was_on) {
        Common::SetBankMarks(Common::BankMarksBindSkip, false);
        Common::BindSkip::g_on.store(false, std::memory_order_seq_cst);
        VideoCommon::S4Count::SetBindSkipEntries(false);
    }
    if (bind_skip_scratch.size() < BindSkipScratchBytes) {
        bind_skip_scratch.resize(BindSkipScratchBytes);
    }
    s.w = BindSkipWindow{};
    s.w.since = std::chrono::steady_clock::now();
    s.w.start_ticks = Nx2Profile::Ticks();
    LOG_INFO(HW_GPU, "Bind skip (FRITCHII_BIND_SKIP={}): {}{}", mode,
             mode == 0   ? "off"
             : mode == 5 ? "dry, decides, counts and times only"
                         : (s.step2 ? "within a list entry and across entries on tracked pages"
                                    : "within a list entry"),
             s.verify ? ", verified (the old path runs too)" : "");
}

template <class P>
u64 BufferCache<P>::BindSkipHash(DAddr device_addr, u32 size) {
    // The mapped pages of the range, as the bank check reads them (an unmapped page stands as a
    // marker of its own).
    u64 hash = 0x9E3779B97F4A7C15ULL ^ size;
    DAddr at = device_addr;
    u64 left = size;
    while (left != 0) {
        const u64 chunk =
            (std::min<u64>)(left, Core::DEVICE_PAGESIZE - (at & Core::DEVICE_PAGEMASK));
        const u8* const pointer = VideoCommon::BackReadAudit::Unaudited(
            [&] { return device_memory.template GetPointer<u8>(at); });
        if (pointer == nullptr) {
            hash = (hash ^ 0xDEADULL) * 0x9E3779B97F4A7C15ULL;
        } else {
            hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(pointer),
                                              static_cast<size_t>(chunk), hash);
        }
        at += chunk;
        left -= chunk;
    }
    return hash;
}

template <class P>
void BufferCache<P>::BindSkipUniform(size_t stage, u32 index, u32 binding_index,
                                     bool needs_bind) {
    if constexpr (IS_OPENGL || !VideoCommon::StreamUniformReusable<Runtime>) {
        BindHostGraphicsUniformBufferBody<false>(stage, index, binding_index, needs_bind, nullptr);
    } else {
        using Ref = typename Runtime::StreamUniformRef;
        using Reuse = VideoCommon::UniformStreamReuse<Ref>;
        using Kind = typename BindSkipOutcome::Kind;
        static_assert(Reuse::NumStages == NUM_STAGES &&
                      Reuse::NumSlots == NUM_GRAPHICS_UNIFORM_BUFFERS);
        static_assert(Reuse::MaxBytes <= BindSkipScratchBytes);
        BindSkipState& s = bind_skip;
        BindSkipWindow& w = s.w;
        ++w.binds;
        // One bind in 64 is timed (decide, skip, the upkeep a bind that is not skipped adds).
        const bool timed = (w.binds & 63) == 0;
        const u64 start = timed ? Nx2Profile::Ticks() : 0;
        const Binding& binding = channel_state->uniform_buffers[stage][index];
        const DAddr device_addr = binding.device_addr;
        const u32 declared = (*channel_state->uniform_buffer_sizes)[stage][index];
        const u32 size = (std::min)(binding.size, declared);
        BindSkipMemo& memo = bind_skip_memo[stage][index];
        // Every memo ends when a guest store's after-store mark was lost (its bytes landed
        // without a mark) or while a run-ahead frontier is up (guest memory may hold a later
        // frame).
        const bool frontier =
            VideoCommon::Runahead::g_frontier.load(std::memory_order_relaxed) != 0;
        const u64 overflows = Common::g_collected_overflows.load(std::memory_order_relaxed);
        if (overflows != s.overflows || frontier) [[unlikely]] {
            s.overflows = overflows;
            ++s.epoch;
        }
        const u64 entry = VideoCommon::S4Count::EngineEntryIndex();
        const u32 switch_generation =
            VideoCore::Nx2Debug::g_switch_generation.load(std::memory_order_acquire);
        // A device page mapped anew reads other host memory with no write anyone saw.
        const u64 translation = device_memory.TranslationGeneration();
        Reuse& reuse = VideoCommon::UniformStreamReuseState<Ref>();
        u64 gen = ~u64{0};
        bool gen_read = false;
        BindSkipReason reason = BindSkipNoMemo;
        if (s.broken || frontier) {
            reason = BindSkipHeld;
        } else if (memo.epoch != s.epoch || memo.kind == Kind::None) {
            reason = BindSkipNoMemo;
        } else if (memo.device_addr != device_addr || memo.size != size ||
                   memo.declared != declared || memo.buffer_id != binding.buffer_id ||
                   memo.binding_epoch != binding_epoch || memo.channel != channel_state ||
                   memo.gpu_memory != gpu_memory || memo.switch_generation != switch_generation ||
                   memo.skip_cache_size != channel_state->uniform_buffer_skip_cache_size ||
                   memo.translation != translation) {
            reason = BindSkipOtherRange;
        } else {
            // A report landing on the range is waited for, as a read of it waits (its write
            // generation shows it then).
            VideoCommon::OneFollower::BeforeEngineRead(device_addr, size);
            gen = Common::BindSkip::RangeGen(device_addr, size);
            gen_read = true;
            if (gen == ~u64{0} || gen != memo.gen) {
                reason = BindSkipWritten;
            } else if (entry != 0 && memo.entry == entry) {
                reason = BindSkipInEntry;
            } else if (!memory_tracker.IsRegionSyncClean(device_addr, size)) {
                // Tracked now with no state change since the memo (the generation) is tracked
                // all along: guest stores there trap and mark.
                reason = BindSkipUntracked;
            } else {
                reason = s.step2 ? BindSkipAcross : BindSkipStep1Only;
            }
            if ((reason == BindSkipInEntry || reason == BindSkipAcross) &&
                memo.kind == Kind::Stream) {
                const typename Reuse::Slot& slot = reuse.slots[stage][index];
                const typename Reuse::Copy& copy = reuse.copies[slot.copy];
                if (slot.serial == 0 || slot.serial != memo.serial || slot.direct != 0 ||
                    copy.serial != slot.serial || copy.size != size) {
                    reason = BindSkipSlotChanged;
                } else if (copy.tick != runtime.StreamEpoch()) {
                    // BindSmallUniformReusing fills a new allocation from the copy then (with
                    // EXP17 it looks for a range copy first): left to it.
                    reason = BindSkipStreamEpoch;
                }
            }
        }
        ++w.reasons[reason];
        const bool would_skip = reason == BindSkipInEntry || reason == BindSkipAcross;
        // Verify modes check every bind the skip would stand in for; the others one in 1,024.
        const bool check = would_skip && !s.dry && (s.verify || (++s.samples & 1023) == 0);
        if (timed) {
            w.decide_ticks += Nx2Profile::Ticks() - start;
            ++w.decide_samples;
        }
        if (would_skip && !s.dry && !check) {
            // --- The skip: what the old path does that stays, without its reads ---
            const u64 skip_start = timed ? Nx2Profile::Ticks() : 0;
            if (!memo.touch_once || memo.touched_frame != frame_tick) {
                TouchBuffer(slot_buffers[binding.buffer_id], binding.buffer_id);
                memo.touched_frame = frame_tick;
            }
            if (Common::S4Count::On()) [[unlikely]] {
                S4CountBind(stage, index, device_addr, size, memo.kind == Kind::Stream,
                            memo.bank);
            }
            if (memo.kind == Kind::Stream) {
                // BindSmallUniformReusing's own rebind of the slot's copy (same bytes, same
                // stream epoch), and the stream branch's binding state.
                typename Reuse::Slot& slot = reuse.slots[stage][index];
                const typename Reuse::Copy& copy = reuse.copies[slot.copy];
                channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                runtime.RebindStreamUniform(copy.ref, size);
                ++reuse.rebinds;
                slot.misses = 0;
                Nx2Profile::AddFine(Nx2Profile::FineCounter::UniformRebinds, 1);
                ++w.skipped_stream;
            } else {
                // The cached path after a synchronize that finds nothing to upload (a hit) and
                // a bank check that finds nothing changed.
                Buffer& buffer = slot_buffers[binding.buffer_id];
                ++channel_state->uniform_cache_hits[0];
                BindUniformCachedTail(stage, index, binding_index, needs_bind, buffer,
                                      buffer.Offset(device_addr), size);
                ++w.skipped_cached;
                w.skipped_bank += memo.bank ? 1 : 0;
            }
            w.skipped_bytes += size;
            if (timed) {
                w.skip_ticks += Nx2Profile::Ticks() - skip_start;
                ++w.skip_samples;
            }
            if ((w.binds & 4095) == 0) {
                BindSkipLog();
            }
            return;
        }
        // --- The old path, the check around it, and the slot's memo after it ---
        bool differ = false;
        if (check) {
            ++w.checked;
            if (memo.kind == Kind::Stream) {
                // What the skip would bind (the slot's copy) against guest memory now.
                const typename Reuse::Slot& slot = reuse.slots[stage][index];
                u8* const scratch = bind_skip_scratch.data();
                VideoCommon::BackReadAudit::Unaudited(
                    [&] { device_memory.ReadBlockUnsafe(device_addr, scratch, size); });
                differ = std::memcmp(reuse.Bytes(slot.copy), scratch, size) != 0;
            } else if (memo.hashed) {
                differ = BindSkipHash(device_addr, size) != memo.hash;
            }
        }
        u64 upkeep_ticks = 0;
        // A memo is made when the slot binds its last range again (the first bind of a range
        // only notes it): slots whose range moves every bind pay no generation read.
        const bool repeat = memo.device_addr == device_addr && memo.size == size;
        if (!gen_read && repeat) {
            // Read before the body reads the range (a write landing after this moves it).
            const u64 gen_start = timed ? Nx2Profile::Ticks() : 0;
            gen = Common::BindSkip::RangeGen(device_addr, size);
            if (timed) {
                upkeep_ticks += Nx2Profile::Ticks() - gen_start;
            }
        }
        const u64 bank_changes = bind_skip_bank_changes;
        // The old path's time for binds the skip would stand in for: every check, and in the
        // dry mode one in 16.
        const bool time_old = would_skip && (check || (s.dry && (++s.samples & 15) == 0));
        const u64 old_start = time_old ? Nx2Profile::Ticks() : 0;
        BindSkipOutcome outcome{};
        BindHostGraphicsUniformBufferBody<true>(stage, index, binding_index, needs_bind, &outcome);
        if (time_old) {
            w.old_ticks += Nx2Profile::Ticks() - old_start;
            ++w.old_samples;
        }
        const bool bank_changed = bind_skip_bank_changes != bank_changes;
        if (check) {
            // The path itself changed: an input of the decision moved unseen (counted as differ).
            differ |= outcome.kind != memo.kind;
            if (outcome.kind == Kind::Cached && (!outcome.synced || bank_changed)) {
                ++w.work;
            }
            if (differ) [[unlikely]] {
                ++w.differ;
                ++s.differ_session;
                if (s.differ_session <= 16) {
                    LOG_WARNING(HW_GPU,
                                "Bind skip: would skip but the bytes differ: stage {} slot {} at "
                                "{:#x}+{:#x} ({}, {}, path then {} now {})",
                                stage, index, device_addr, size,
                                memo.kind == Kind::Stream ? "stream" : "cached",
                                reason == BindSkipInEntry ? "in its list entry" : "a later entry",
                                static_cast<int>(memo.kind), static_cast<int>(outcome.kind));
                }
                if (!s.verify && !s.broken) {
                    s.broken = true;
                    LOG_ERROR(HW_GPU,
                              "Bind skip (FRITCHII_BIND_SKIP={}): a sampled check found other "
                              "bytes; no skip for the rest of the session",
                              s.mode);
                }
            }
        }
        const u64 record_start = timed ? Nx2Profile::Ticks() : 0;
        memo.kind = Kind::None;
        // The plain bank check and EXP9=3 only: EXP9=1 compares each page once a frame for
        // writers no flag sees, and 2 counts every compare, on purpose.
        const bool bank_ok = !outcome.bank || outcome.bank_mode == 0 || outcome.bank_mode == 3;
        if (outcome.kind != Kind::None && binding.buffer_id != NULL_BUFFER_ID &&
            gen != ~u64{0} && !s.broken && !frontier &&
            frame_tick >= s.ready_frame && !outcome.held && bank_ok &&
            (outcome.kind == Kind::Stream || (outcome.synced && !bank_changed))) {
            const u32 serial = outcome.kind == Kind::Stream
                                   ? VideoCommon::UniformSlotSerial<Ref>(stage, index)
                                   : 0;
            if (outcome.kind != Kind::Stream || serial != 0) {
                memo.channel = channel_state;
                memo.gpu_memory = gpu_memory;
                memo.device_addr = device_addr;
                memo.gen = gen;
                memo.entry = entry;
                memo.binding_epoch = binding_epoch;
                memo.translation = translation;
                memo.touched_frame = frame_tick;
                memo.size = size;
                memo.declared = declared;
                memo.serial = serial;
                memo.skip_cache_size = channel_state->uniform_buffer_skip_cache_size;
                memo.switch_generation = switch_generation;
                memo.epoch = s.epoch;
                memo.buffer_id = binding.buffer_id;
                memo.kind = outcome.kind;
                memo.bank = outcome.bank;
                memo.touch_once = (VideoCore::Nx2Debug::Experiment(18) & 4) != 0;
                memo.hashed = s.verify && outcome.kind == Kind::Cached;
                memo.hash = memo.hashed ? BindSkipHash(device_addr, size) : 0;
            }
        }
        memo.device_addr = device_addr;
        memo.size = size;
        if (timed) {
            w.record_ticks += upkeep_ticks + (Nx2Profile::Ticks() - record_start);
            ++w.record_samples;
        }
        if ((w.binds & 4095) == 0) {
            BindSkipLog();
        }
    }
}

template <class P>
void BufferCache<P>::BindSkipLog() {
    BindSkipState& s = bind_skip;
    BindSkipWindow& w = s.w;
    const auto now = std::chrono::steady_clock::now();
    if (w.since == std::chrono::steady_clock::time_point{}) {
        w.since = now;
        w.start_ticks = Nx2Profile::Ticks();
        return;
    }
    if (now - w.since < std::chrono::seconds{10}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - w.since).count();
    const u64 ticks = Nx2Profile::Ticks() - w.start_ticks;
    const double ns_per_tick = ticks != 0 ? seconds * 1e9 / static_cast<double>(ticks) : 0.0;
    const auto rate = [seconds](u64 count) { return static_cast<double>(count) / seconds; };
    const auto mean_ns = [ns_per_tick](u64 sum, u64 samples) {
        return samples != 0 ? static_cast<double>(sum) * ns_per_tick / static_cast<double>(samples)
                            : 0.0;
    };
    const double decide_ns = mean_ns(w.decide_ticks, w.decide_samples);
    const double skip_ns = mean_ns(w.skip_ticks, w.skip_samples);
    const double old_ns = mean_ns(w.old_ticks, w.old_samples);
    const double upkeep_ns = mean_ns(w.record_ticks, w.record_samples);
    const u64 would = w.reasons[BindSkipInEntry] + w.reasons[BindSkipAcross];
    const u64 skipped = s.dry ? would : w.skipped_stream + w.skipped_cached;
    const u64 not_skipped = w.binds - (std::min)(skipped, w.binds);
    // E's time saved: the skipped binds' old path less the skip, less what every bind's decide
    // and the other binds' memo upkeep add (the dry mode has no skip time: an upper bound).
    const double saved_ns = static_cast<double>(skipped) * (old_ns - skip_ns) -
                            static_cast<double>(w.binds) * decide_ns -
                            static_cast<double>(not_skipped) * upkeep_ns;
    LOG_INFO(HW_GPU,
             "Bind skip (FRITCHII_BIND_SKIP={}) {:.0f} s: binds {:.0f}/s, {} {:.0f}/s (in entry "
             "{:.0f}, across entries {:.0f}; stream {:.0f}, cached {:.0f}, native banks among "
             "them {:.0f}), bytes not read {:.1f} MB/s; not skipped: no memo {:.0f}/s, other "
             "range {:.0f}, written {:.0f}, untrapped page {:.0f}, step 1 only {:.0f}, slot copy "
             "changed {:.0f}, older stream epoch {:.0f}, held {:.0f}{}; time (1 in 64 sampled): "
             "decide {:.0f} ns a bind, skip {:.0f} ns, old path of skippable binds {:.0f} ns, "
             "memo upkeep of other binds {:.0f} ns: about {:.1f} ms/s of E saved{}; verify: "
             "checked {:.0f}/s, differ {} (must be 0, {} session), old path did work on a "
             "cached would-skip {:.0f}/s",
             s.mode, seconds, rate(w.binds), s.dry ? "would skip" : "skipped", rate(skipped),
             rate(w.reasons[BindSkipInEntry]), rate(w.reasons[BindSkipAcross]),
             rate(w.skipped_stream), rate(w.skipped_cached), rate(w.skipped_bank),
             static_cast<double>(w.skipped_bytes) / seconds / (1024.0 * 1024.0),
             rate(w.reasons[BindSkipNoMemo]), rate(w.reasons[BindSkipOtherRange]),
             rate(w.reasons[BindSkipWritten]), rate(w.reasons[BindSkipUntracked]),
             rate(w.reasons[BindSkipStep1Only]), rate(w.reasons[BindSkipSlotChanged]),
             rate(w.reasons[BindSkipStreamEpoch]), rate(w.reasons[BindSkipHeld]),
             s.broken ? " (STOPPED: a sampled check found other bytes)" : "", decide_ns,
             skip_ns, old_ns, upkeep_ns, saved_ns / seconds / 1e6,
             s.dry ? " (dry: skip time not known, an upper bound)" : "", rate(w.checked),
             w.differ, s.differ_session, rate(w.work));
    s.w = BindSkipWindow{};
    s.w.since = now;
    s.w.start_ticks = Nx2Profile::Ticks();
}

template <class P>
void BufferCache<P>::NoteCachedWrites(std::span<const std::pair<DAddr, std::size_t>> sequences) {
    VideoCommon::MemoryStageLink::CheckBackCache("NoteCachedWrites");
    UniformKeepCount& count = uniform_keep_count;
    if (VideoCore::Nx2Debug::Experiment(91) == 0 && VideoCore::Nx2Debug::Experiment(92) == 0) {
        // Not noted: whatever a keep or count remembers is past the ring now (EXP92 turned on
        // again before a bind saw it off still finds every older keep overflowed).
        if (!sequences.empty()) {
            count.write_seq += UniformKeepCount::RingSize + 1;
        }
        return;
    }
    for (const auto& [addr, size] : sequences) {
        ++count.write_seq;
        count.ring[count.write_seq % UniformKeepCount::RingSize] = {addr, addr + size};
    }
}

template <class P>
bool BufferCache<P>::BindSmallUniformCountingKeeps(size_t stage, u32 index, u32 binding_index,
                                                   DAddr device_addr, u32 size,
                                                   bool range_reuse) {
    if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
        using Ref = typename Runtime::StreamUniformRef;
        using Reuse = VideoCommon::UniformStreamReuse<Ref>;
        using Count = UniformKeepCount;
        static_assert(Count::NumStages == Reuse::NumStages && Count::NumSlots == Reuse::NumSlots);
        Reuse& reuse = VideoCommon::UniformStreamReuseState<Ref>();
        Count& count = uniform_keep_count;
        // A whole frame without a counted bind: EXP91 was off (nothing noted then) or nothing
        // drew. Start over, so no bind is judged across ranges the ring did not see.
        if (count.last_log == std::chrono::steady_clock::time_point{} ||
            count.last_frame + 1 < frame_tick) {
            for (auto& stage_slots : count.slots) {
                stage_slots.fill(typename Count::Slot{});
            }
            count.totals = {};
            count.log_write_seq = count.write_seq;
            count.last_log = std::chrono::steady_clock::now();
        }
        count.last_frame = frame_tick;
        const bool in_slot = stage < Count::NumStages && index < Count::NumSlots;
        typename Reuse::Slot before{};
        if (in_slot) {
            before = reuse.slots[stage][index];
        }
        const u64 rebinds = reuse.rebinds;
        const bool streamed = VideoCommon::BindSmallUniformReusing(
            runtime, device_memory, stage, index, binding_index, device_addr, size, range_reuse);
        const u64 rebound = reuse.rebinds - rebinds;
        typename Count::Totals& totals = count.totals;
        ++totals.binds;
        totals.rebinds += rebound;
        if (in_slot) {
            typename Count::Slot& keep = count.slots[stage][index];
            const typename Reuse::Slot& after = reuse.slots[stage][index];
            // The slot still names the same copy and serial only after today's slot rebind (the
            // compare said "same" and the copy's tick is the current one); a range-reuse rebind
            // or a fresh copy changes it.
            if (rebound != 0 && after.copy == before.copy && after.serial == before.serial) {
                ++totals.slot_rebinds;
                const u64 since = count.write_seq - keep.seen_seq;
                const auto overlaps = [&] {
                    const DAddr end = device_addr + size;
                    for (u64 seq = keep.seen_seq + 1; seq <= count.write_seq; ++seq) {
                        const typename Count::Range& range = count.ring[seq % Count::RingSize];
                        if (range.begin < end && device_addr < range.end) {
                            return true;
                        }
                    }
                    return false;
                };
                if (!keep.valid) {
                    ++totals.no_keep;
                } else if (keep.addr != device_addr || keep.size != size) {
                    ++totals.other_range;
                } else if (since > Count::RingSize) {
                    ++totals.overflow;
                } else if (overlaps()) {
                    ++totals.overlap;
                } else if (gpu_memory != nullptr && gpu_memory->HasPendingCachedWrites()) {
                    ++totals.pending;
                } else {
                    ++totals.kept;
                }
            }
            if (streamed) {
                keep = typename Count::Slot{device_addr, size, true, count.write_seq};
            } else {
                keep.valid = false;
            }
        }
        if ((++count.calls & 1023) == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - count.last_log).count();
            if (seconds >= 5.0) {
                const auto rate = [seconds](u64 n) {
                    return static_cast<u64>(static_cast<double>(n) / seconds + 0.5);
                };
                LOG_INFO(HW_GPU,
                         "Uniform keeps count (EXP91, per second over {:.1f} s): small binds {}, "
                         "rebinds today {} (slot {}), K (keepable) {}; not kept: range written "
                         "{}, ring overflow {}, pending cached writes {}, other range {}, no "
                         "keep yet {}; ranges noted {}",
                         seconds, rate(totals.binds), rate(totals.rebinds),
                         rate(totals.slot_rebinds), rate(totals.kept), rate(totals.overlap),
                         rate(totals.overflow), rate(totals.pending), rate(totals.other_range),
                         rate(totals.no_keep), rate(count.write_seq - count.log_write_seq));
                count.totals = {};
                count.log_write_seq = count.write_seq;
                count.last_log = now;
            }
        }
        return streamed;
    } else {
        (void)stage;
        (void)index;
        (void)binding_index;
        (void)device_addr;
        (void)size;
        (void)range_reuse;
        return false;
    }
}

template <class P>
bool BufferCache<P>::BindSmallUniformKeeping(size_t stage, u32 index, u32 binding_index,
                                             DAddr device_addr, u32 size, bool range_reuse,
                                             bool eligible, int mode) {
    if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
        using Ref = typename Runtime::StreamUniformRef;
        using Reuse = VideoCommon::UniformStreamReuse<Ref>;
        using Keeps = UniformKeeps;
        using Keep = typename Keeps::Keep;
        using Count = UniformKeepCount;
        namespace KeepPages = Common::UniformKeepPages;
        static_assert(Keeps::NumStages == Reuse::NumStages && Keeps::NumSlots == Reuse::NumSlots);
        static_assert(KeepPages::PageBits == Core::DEVICE_PAGEBITS);
        static_assert(Reuse::MaxBytes <= sizeof(Keeps::verify_bytes));
        Reuse& reuse = VideoCommon::UniformStreamReuseState<Ref>();
        Keeps& keeps = uniform_keeps;
        const auto bind_normally = [&] {
            return VideoCommon::BindSmallUniformReusing(runtime, device_memory, stage, index,
                                                        binding_index, device_addr, size,
                                                        range_reuse);
        };
        if (mode != keeps.mode) {
            // Read live: any change drops every keep (1 -> 0 at any time included).
            for (auto& stage_keeps : keeps.slots) {
                stage_keeps.fill(Keep{});
            }
            if (keeps.mode == 0) {
                keeps.totals = {};
                keeps.last_log = std::chrono::steady_clock::now();
                keeps.log_guest_stores =
                    KeepPages::g_state.guest_stores.load(std::memory_order_relaxed);
                keeps.log_pages_denied =
                    KeepPages::g_state.pages_denied.load(std::memory_order_relaxed);
            }
            keeps.mode = mode;
        }
        if (mode == 0 || size == 0 || size > VideoCommon::UniformStreamMaxBytes() ||
            stage >= Keeps::NumStages || index >= Keeps::NumSlots) {
            return bind_normally();
        }
        typename Keeps::Totals& totals = keeps.totals;
        ++totals.binds;
        if (keeps.tripped) {
            // The tripwire fired: today's path for the rest of the session (still logged).
            if ((++keeps.calls & 1023) == 0) {
                LogUniformKeeps(mode);
            }
            return bind_normally();
        }
        Keep& keep = keeps.slots[stage][index];
        const Count& ring = uniform_keep_count;
        const DAddr end = device_addr + size;
        const u64 first_page = device_addr >> Core::DEVICE_PAGEBITS;
        const u64 last_page = (end - 1) >> Core::DEVICE_PAGEBITS;
        const u64 tick = runtime.StreamEpoch();

        // 3.5: whether the slot's keep may be bound without reading guest memory.
        u64 scanned = 0;
        const auto overlaps = [&] {
            for (u64 seq = keep.seen_seq + 1; seq <= ring.write_seq; ++seq) {
                ++scanned;
                const typename Count::Range& range = ring.ring[seq % Count::RingSize];
                if (range.begin < end && device_addr < range.end) {
                    return true;
                }
            }
            return false;
        };
        using Reason = typename Keeps::Reason;
        Reason reason = Keeps::NumReasons; // keepable
        if (!keep.valid) {
            reason = Keeps::NoKeep;
        } else if (keep.addr != device_addr || keep.size != size) {
            reason = Keeps::OtherRange;
        } else if (!eligible) {
            reason = Keeps::Ineligible;
        } else if (const typename Reuse::Copy& copy = reuse.copies[keep.copy];
                   copy.serial != keep.copy_serial || copy.tick != tick ||
                   reuse.slots[stage][index].copy != keep.copy ||
                   reuse.slots[stage][index].serial != keep.copy_serial) {
            reason = Keeps::Tick;
        } else if (device_memory.TranslationGeneration() != keep.translation) {
            reason = Keeps::Translation;
        } else if (KeepPages::GlobalGen() != keep.global_gen ||
                   KeepPages::PageGen(first_page) != keep.page_gens[0] ||
                   KeepPages::PageGen(last_page) != keep.page_gens[1]) {
            reason = Keeps::PageGen;
        } else if (ring.write_seq - keep.seen_seq > Count::RingSize) {
            reason = Keeps::RingOverflow;
        } else if (overlaps()) {
            reason = Keeps::RangeWritten;
        } else if (gpu_memory != nullptr && gpu_memory->HasPendingCachedWrites()) {
            reason = Keeps::Pending;
        }
        const bool keepable = reason == Keeps::NumReasons;
        bool check = false;
        if (keepable) {
            ++totals.kept;
            ++keeps.kept_count;
            check = mode == 2 || keeps.kept_count % Keeps::TripwireInterval == 0;
            if (!check) {
                keep.seen_seq = ring.write_seq;
                // What BindSmallUniformReusing does when its compare says "same" and the tick
                // matches (the stream branch already set the two channel_state lines).
                const typename Reuse::Copy& copy = reuse.copies[keep.copy];
                runtime.RebindStreamUniform(copy.ref, size);
                ++reuse.rebinds;
                reuse.slots[stage][index] = typename Reuse::Slot{keep.copy, keep.copy_serial};
                if ((++keeps.calls & 1023) == 0) {
                    LogUniformKeeps(mode);
                }
                return true;
            }
            std::memcpy(keeps.verify_bytes.data(), reuse.Bytes(keep.copy), size);
        } else {
            ++totals.not_kept[reason];
        }
        const Keep previous = keep;

        // 3.4: making a keep. 1. mark and watch the pages; 2. read their generations (then the
        // watches again: a page unwatched meanwhile can't hold a keep); 3. the normal bind;
        // 4. store the keep.
        bool can_keep = eligible;
        std::array<u32, 2> gens{};
        u32 global_gen = 0;
        u64 translation = 0;
        if (can_keep) {
            if (!KeepPages::Watch(device_addr, size, [this](DAddr page_address, s32 delta) {
                    device_memory.UpdatePagesCachedCount(page_address, Core::DEVICE_PAGESIZE,
                                                         delta);
                })) {
                can_keep = false;
                ++totals.denied;
            } else {
                translation = device_memory.TranslationGeneration();
                global_gen = KeepPages::GlobalGen();
                gens = {KeepPages::PageGen(first_page), KeepPages::PageGen(last_page)};
                can_keep = KeepPages::AllWatched(device_addr, size);
            }
        }
        const u64 rebinds_before = reuse.rebinds;
        const bool streamed = bind_normally();
        if (check) {
            // 4: the normal path must have bound the kept copy again ("same", same tick).
            ++totals.verified;
            const typename Reuse::Slot& after = reuse.slots[stage][index];
            const bool same = streamed && reuse.rebinds != rebinds_before &&
                              after.copy == previous.copy && after.serial == previous.copy_serial;
            if (!same) {
                ++totals.stale;
                ++keeps.stale_total;
                if (keeps.stale_logged < Keeps::MaxStaleLogs) {
                    ++keeps.stale_logged;
                    // Where the bytes differ, against guest memory now.
                    VideoCommon::OneFollower::BeforeEngineRead(device_addr, size);
                    device_memory.ReadBlockUnsafe(device_addr, keeps.fresh_bytes.data(), size);
                    u32 first_diff = size;
                    for (u32 i = 0; i < size; ++i) {
                        if (keeps.fresh_bytes[i] != keeps.verify_bytes[i]) {
                            first_diff = i;
                            break;
                        }
                    }
                    const auto& pages = KeepPages::g_state;
                    LOG_ERROR(HW_GPU,
                              "Uniform keeps (EXP92={}): STALE bind #{}: stage {} slot {} address "
                              "{:#x} size {} kept {} frames ago; quiet checks: range and copy "
                              "same, copy tick current, translation {:#x} unchanged, page gens "
                              "{}/{} and global {} unchanged, ring clean ({} ranges scanned since "
                              "seq {}), no pending cached writes; pages now: watched {}/{}, "
                              "denied {}/{}; first changed byte {} of {}",
                              mode, keeps.stale_total, stage, index, device_addr, size,
                              frame_tick - previous.made_frame, previous.translation,
                              previous.page_gens[0], previous.page_gens[1], previous.global_gen,
                              scanned, previous.seen_seq, pages.watched.Test(first_page),
                              pages.watched.Test(last_page), pages.denied.Test(first_page),
                              pages.denied.Test(last_page), first_diff, size);
                }
                if (mode == 1 && !keeps.tripped) {
                    keeps.tripped = true;
                    LOG_ERROR(HW_GPU,
                              "Uniform keeps (EXP92=1): the tripwire found a stale kept bind "
                              "(stage {} slot {} address {:#x} size {}); keeps are off for this "
                              "session",
                              stage, index, device_addr, size);
                }
            }
        }
        if (streamed && can_keep) {
            const typename Reuse::Slot& slot = reuse.slots[stage][index];
            keep = Keep{
                .addr = device_addr,
                .size = size,
                .copy = slot.copy,
                .copy_serial = slot.serial,
                .page_gens = gens,
                .global_gen = global_gen,
                .seen_seq = ring.write_seq,
                .translation = translation,
                .made_frame = frame_tick,
                .valid = true,
            };
        } else {
            keep.valid = false;
        }
        if ((++keeps.calls & 1023) == 0) {
            LogUniformKeeps(mode);
        }
        return streamed;
    } else {
        (void)stage;
        (void)index;
        (void)binding_index;
        (void)device_addr;
        (void)size;
        (void)range_reuse;
        (void)eligible;
        (void)mode;
        return false;
    }
}

template <class P>
void BufferCache<P>::LogUniformKeeps(int mode) {
    UniformKeeps& keeps = uniform_keeps;
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - keeps.last_log).count();
    if (seconds < 5.0) {
        return;
    }
    namespace KeepPages = Common::UniformKeepPages;
    const auto& pages = KeepPages::g_state;
    const u64 guest_stores = pages.guest_stores.load(std::memory_order_relaxed);
    const u64 pages_denied = pages.pages_denied.load(std::memory_order_relaxed);
    const auto rate = [seconds](u64 n) {
        return static_cast<u64>(static_cast<double>(n) / seconds + 0.5);
    };
    const typename UniformKeeps::Totals& totals = keeps.totals;
    const auto& reasons = totals.not_kept;
    LOG_INFO(HW_GPU,
             "Uniform keeps (EXP92={}, per second over {:.1f} s): small binds {}, kept {}; not "
             "kept: range written {}, page generation {}, tick {}, other range {}, ring overflow "
             "{}, pending writes {}, translation {}, no keep yet {}, bank/run-ahead {}, denied "
             "page {}; watched pages {}, guest stores on them {}, pages denied {} (total {}); "
             "verified {} and stale {} in {:.1f} s, stale this session {}{}",
             mode, seconds, rate(totals.binds), rate(totals.kept),
             rate(reasons[UniformKeeps::RangeWritten]), rate(reasons[UniformKeeps::PageGen]),
             rate(reasons[UniformKeeps::Tick]), rate(reasons[UniformKeeps::OtherRange]),
             rate(reasons[UniformKeeps::RingOverflow]), rate(reasons[UniformKeeps::Pending]),
             rate(reasons[UniformKeeps::Translation]), rate(reasons[UniformKeeps::NoKeep]),
             rate(reasons[UniformKeeps::Ineligible]), rate(totals.denied),
             pages.watched_count.load(std::memory_order_relaxed),
             rate(guest_stores - keeps.log_guest_stores), pages_denied - keeps.log_pages_denied,
             pages_denied, totals.verified, totals.stale, seconds, keeps.stale_total,
             keeps.tripped ? " (TRIPPED: keeps off)" : "");
    keeps.totals = {};
    keeps.log_guest_stores = guest_stores;
    keeps.log_pages_denied = pages_denied;
    keeps.last_log = now;
}

template <class P>
void BufferCache<P>::RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        VideoCommon::RefreshMultiRangeSegments(
            std::span{pool}, [this](BufferId id) { return slot_buffers.contains(id); },
            [this](DAddr address, u32 size) {
                return FindBuffer(address, size, runtime.PrefersSparseSources());
            });
    }
}

template <class P>
void BufferCache<P>::ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                              std::vector<MultiRangeSegment>& pool) {
    binding.segment_first = 0;
    binding.segment_count = 0;
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.gpu_addr == 0 || binding.size == 0) {
            return;
        }
        if (is_written && !runtime.PrefersSparseSources()) {
            return;
        }
        // Counted for compute bindings only (the compute bindings line).
        const bool count = Nx2Profile::Enabled() && &pool == &compute_segments;
        bool hit = false;
        const VirtualSegments* found = virtual_ranges.Query(*gpu_memory, binding.gpu_addr,
                                                            binding.size, count ? &hit : nullptr);
        if (count) {
            Nx2Profile::Add(hit ? Nx2Profile::Counter::RangeCacheHits
                                : Nx2Profile::Counter::RangeCacheMisses,
                            1);
        }
        if (!found || found->size() < 2) {
            return;
        }
        const VirtualSegments segments = *found;
        const u32 first = static_cast<u32>(pool.size());
        const bool prefer_sparse = runtime.PrefersSparseSources();
        for (const VirtualSegment& segment : segments) {
            const BufferId buffer_id =
                FindBuffer(segment.device_addr, segment.size, prefer_sparse);
            if (!buffer_id) {
                pool.resize(first);
                return;
            }
            pool.push_back(MultiRangeSegment{
                .buffer_id = buffer_id,
                .device_addr = segment.device_addr,
                .size = segment.size,
            });
        }
        binding.segment_first = first;
        binding.segment_count = static_cast<u32>(segments.size());
    }
}

template <class P>
bool BufferCache<P>::BindMultiRangeStorage(const Binding& binding, bool is_written,
                                           std::span<const MultiRangeSegment> pool,
                                           u64 write_mark_limit) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.segment_count < 2) {
            return false;
        }
        if (binding.segment_first > pool.size() ||
            binding.segment_count > pool.size() - binding.segment_first) {
            return false;
        }
        const auto segments = pool.subspan(binding.segment_first, binding.segment_count);
        if (!MultiRangeSegmentsLive(segments,
                                    [this](BufferId id) { return slot_buffers.contains(id); })) {
            return false;
        }
        const u64 key = (static_cast<u64>(gpu_memory->GetID()) << 48) ^ binding.gpu_addr;
        runtime.ResetMultiRange();
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const MultiRangeSegment& segment = pool[binding.segment_first + index];
            Buffer& buffer = slot_buffers[segment.buffer_id];
            TouchBuffer(buffer, segment.buffer_id);
            SynchronizeBuffer(buffer, segment.device_addr, segment.size);
            const u32 offset = buffer.Offset(segment.device_addr);
            buffer.MarkUsage(offset, segment.size);
            if (is_written && write_mark_limit != 0) {
                // write_mark_limit: the native write mark (BindComputeStorageBuffer), spread over
                // the segments in order.
                const u32 mark = static_cast<u32>(std::min<u64>(segment.size, write_mark_limit));
                MarkWrittenBuffer(segment.buffer_id, segment.device_addr, mark);
                if (write_mark_limit != ~u64{0}) {
                    write_mark_limit -= mark;
                }
            }
            runtime.PushMultiRangeSource(buffer, offset, segment.size);
        }
        return BindFreshMultiRangeStorage(runtime, key, is_written);
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::CheckNativeBankPagesTracked(DAddr device_addr, u32 size, int mode,
                                                 int trims, Buffer& buffer, size_t stage,
                                                 u32 index) {
    // EDEN_NX2_EXP9 (experiment, 0 = off): compare only the pages whose guest memory may have
    // changed since their last compare. 1: pages guest CPU writes were collected for (applied
    // only at the next InvalidateGPUCache, while the tracker holds the page clean), pages the
    // buffer cache saw written, pages flagged at the previous bind, CPU-modified pages, and each
    // page once per frame for writers none of these see. 3: the same without the frame check.
    // 2: every page as with 0, counting the changed pages the rules of 1 and 3 would have skipped.
    static_assert(Core::DevicePageFlags::AddressBits >=
                  Tegra::MaxwellDeviceTraits::device_virtual_bits);
    // FRITCHII_BANK_FAST (live, default 0; fable_b 25.2 cut b): a settled bank of a slot the last
    // full call would skip whole is skipped before the switch reads and the upkeep below, while
    // nothing that upkeep reads moved: the switches (their generation), the lost-store count,
    // the marks gate's state, the snapshot store's size, the inner invalidation setting. The
    // test is the same four as the skip's own.
    const u32 switch_generation =
        VideoCore::Nx2Debug::g_switch_generation.load(std::memory_order_acquire);
    if (native_bank_fast.ready && index < 32 && ((native_bank_fast.skip_slots >> index) & 1) != 0 &&
        mode == native_bank_fast.mode && switch_generation == native_bank_fast.generation &&
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bank_fast, "FRITCHII_BANK_FAST") !=
            0 &&
        Common::g_collected_overflows.load(std::memory_order_seq_cst) ==
            native_bank_fast.overflows &&
        (native_bank_marks_ready || !native_bank_marks_wanted) && native_bank_table_on &&
        native_bank_table_pages < 16384 &&
        !Settings::values.skip_cpu_inner_invalidation.GetValue()) {
        const DAddr bank_first = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr bank_end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
        const u64 bank_bytes = bank_end - bank_first;
        if (!Core::g_collected_write_pages.Any(bank_first, bank_bytes) &&
            !Core::g_buffer_write_pages.Any(bank_first, bank_bytes) &&
            NativeBankPagesSettled(bank_first, bank_end) &&
            !memory_tracker.IsRegionCpuModified(bank_first, bank_bytes)) {
            auto& fast_stats = native_bank_tracked_stats;
            const bool flagged_slot = ((native_bank_fast.flagged_slots >> index) & 1) != 0;
            ++fast_stats.binds;
            ++fast_stats.banks_skipped;
            fast_stats.flagged_binds += flagged_slot ? 1 : 0;
            fast_stats.flagged_skipped += flagged_slot ? 1 : 0;
            ++native_bank_fast_hits;
            LogNativeBankTrackedStats(mode);
            return;
        }
    }
    SelectNativeBankStore();
    auto& stats = native_bank_tracked_stats;
    ++stats.binds;
    // FRITCHII_BANK_FLAGGED (1 on, 2 verify): the slots of FRITCHII_BANK_FLAGGED_SLOTS drop the
    // hot rule, as EDEN_NX2_BANK_SLOTS' do (the exactness argument is in docs/fritchii/
    // switches.md). It needs every snapshot to hold what its page's host copy holds, which
    // uploads keep from now on (RefreshNativeBankSnapshots, mapped uploads only): when it turns on
    // every snapshot is dropped, so each page's next bind takes a new one and marks the page
    // changed. A guest store whose range the after-store point lost (the overflow count) set no
    // flag after its bytes landed: every snapshot is dropped then too.
    const int flagged_switch =
        USE_MEMORY_MAPS_FOR_UPLOADS
            ? VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bank_flagged,
                                               "FRITCHII_BANK_FLAGGED")
            : 0;
    // FRITCHII_BANK_CENSUS (counters only): for each change found, what touched the page since
    // its last compare (NoteNativeBankCensusChange). 2 also turns the write ring on.
    const int census = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bank_census,
                                                        "FRITCHII_BANK_CENSUS");
    // The bank marks' gate (Common::g_bank_marks): the writers' marks the hot rule made unneeded
    // are set only while one of the two switches is on. A writer that loaded the gate shut set no
    // mark, so nothing snapshotted before it opened is trusted: two frames after it opened (every
    // writer's next load sees it by then) every snapshot is dropped once, and the flagged rule
    // works only from then on.
    const bool want_marks = flagged_switch != 0 || census != 0;
    if (want_marks != native_bank_marks_wanted) [[unlikely]] {
        native_bank_marks_wanted = want_marks;
        native_bank_marks_ready = false;
        native_bank_marks_since = frame_tick;
        Common::SetBankMarks(Common::BankMarksBankCheck, want_marks);
    }
    if (want_marks && !native_bank_marks_ready &&
        frame_tick - native_bank_marks_since >= 2) [[unlikely]] {
        native_bank_marks_ready = true;
        native_bank_flagged_overflows =
            Common::g_collected_overflows.load(std::memory_order_seq_cst);
        ClearNativeBankSnapshots();
        ++native_bank_flagged_resets;
    }
    const int flagged = native_bank_marks_ready ? flagged_switch : 0;
    if (flagged != 0) [[unlikely]] {
        const u64 overflows = Common::g_collected_overflows.load(std::memory_order_seq_cst);
        if (!native_bank_flagged_on || overflows != native_bank_flagged_overflows) {
            native_bank_flagged_overflows = overflows;
            native_bank_flagged_on = true;
            ClearNativeBankSnapshots();
            ++native_bank_flagged_resets;
        }
    } else {
        native_bank_flagged_on = false;
    }
    Core::DevicePageFlags* census_uploads = nullptr;
    if (census != 0 || native_bank_census_ring_wanted) [[unlikely]] {
        const bool want_ring = census >= 2;
        if (want_ring != native_bank_census_ring_wanted) {
            native_bank_census_ring_wanted = want_ring;
            if (want_ring) {
                native_bank_census_ring_owner =
                    !Common::WriteRing::g_on.exchange(true, std::memory_order_relaxed);
            } else if (native_bank_census_ring_owner) {
                Common::WriteRing::g_on.store(false, std::memory_order_relaxed);
                native_bank_census_ring_owner = false;
            }
        }
        if (census != 0) {
            if (!native_bank_census_upload_owner) {
                native_bank_census_upload_owner = std::make_unique<Core::DevicePageFlags>();
                native_bank_census_uploads.store(native_bank_census_upload_owner.get(),
                                                 std::memory_order_release);
            }
            census_uploads = native_bank_census_upload_owner.get();
            ++native_bank_census_bind;
        }
    }
    // EDEN_NX2_BANK_SLOT_STATS (recomp research idea 1, step 1): the same counts per constant
    // buffer index, to see which of NVN2's banks (c0 driver, c1 program, c2 handles, c3+ the
    // game's) the 64 KiB binds are and which of them ever change outside our own writes.
    NativeBankSlotStats* const slot_stats =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bank_slot_stats,
                                         "EDEN_NX2_BANK_SLOT_STATS") != 0 &&
                index < native_bank_slot_stats.size()
            ? &native_bank_slot_stats[index]
            : nullptr;
    if (slot_stats != nullptr) [[unlikely]] {
        ++slot_stats->binds;
        slot_stats->stages |= 1u << stage;
    }
    // EDEN_NX2_BANK_SLOTS (recomp research idea 1): banks NVN2's driver
    // writes only through the command stream (c0, c2) or never (c1) change only through writes
    // that set a flag: our own constant buffer words reach the buffer cache's WriteMemory at
    // FlushCaching (g_buffer_write_pages), guest CPU writes on tracked pages are collected
    // (g_collected_write_pages) and untracked ones leave the page CPU-modified. For those slots
    // the hot rule goes, and a bank with no flag anywhere, no pending recheck, no CPU-modified
    // page and a snapshot of every page has nothing the page loop would compare: skipped whole.
    // The game's banks (c3+) keep the loop and the hot rule.
    const u32 own_slots = static_cast<u32>(
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bank_slots, "EDEN_NX2_BANK_SLOTS",
                                         NativeBankDefaultsForOcarina() ? 0x2 : 0));
    // FRITCHII_BANK_FLAGGED: the game's slots the census cleared (unseen 0), with the same rule.
    const u32 flagged_slots =
        flagged != 0 ? static_cast<u32>(VideoCore::Nx2Debug::SwitchValue(
                           VideoCore::Nx2Debug::g_bank_flagged_slots,
                           "FRITCHII_BANK_FLAGGED_SLOTS", 0x40))
                     : 0;
    const bool flagged_slot = index < 32 && ((flagged_slots >> index) & 1) != 0;
    // Not while CPU inner invalidation is skipped: our constant buffer words then set no flag.
    const bool own_slot = index < 32 && (((own_slots >> index) & 1) != 0 || flagged_slot) &&
                          !Settings::values.skip_cpu_inner_invalidation.GetValue();
    // FRITCHII_BANK_FLAGGED=2 verifies its slots as EDEN_NX2_BANK_SKIP_VERIFY does.
    const bool slot_verify =
        own_slot &&
        ((flagged_slot && flagged == 2) ||
         VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bank_skip_verify,
                                          "EDEN_NX2_BANK_SKIP_VERIFY") != 0);
    if (own_slot && flagged_slot) [[unlikely]] {
        ++stats.flagged_binds;
    }
    // FRITCHII_BANK_FAST: what this call decided for the slot, for the next binds' fast test
    // (only while no census, slot counts or ring, which the fast test does not keep).
    if (index < 32) {
        NativeBankFast& fast = native_bank_fast;
        const bool steady = census == 0 && slot_stats == nullptr &&
                            !native_bank_census_ring_wanted;
        if (!steady || fast.generation != switch_generation || fast.mode != mode ||
            fast.overflows != native_bank_flagged_overflows) {
            fast = NativeBankFast{};
        }
        if (steady) {
            fast.generation = switch_generation;
            fast.mode = mode;
            fast.overflows = native_bank_flagged_overflows;
            const u32 bit = 1u << index;
            const bool skips = own_slot && mode == 3 && native_bank_table_on && !slot_verify &&
                               (flagged == 0 || native_bank_flagged_on);
            fast.skip_slots = skips ? fast.skip_slots | bit : fast.skip_slots & ~bit;
            fast.flagged_slots =
                skips && flagged_slot ? fast.flagged_slots | bit : fast.flagged_slots & ~bit;
            fast.ready = true;
        }
    }
    bool skip_verify = false;
    if (own_slot && mode == 3 && native_bank_table_on) {
        // The whole bound range, page aligned: a superset of what the loop below walks.
        const DAddr bank_first = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr bank_end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
        const u64 bank_bytes = bank_end - bank_first;
        if (!Core::g_collected_write_pages.Any(bank_first, bank_bytes) &&
            !Core::g_buffer_write_pages.Any(bank_first, bank_bytes) &&
            NativeBankPagesSettled(bank_first, bank_end) &&
            !memory_tracker.IsRegionCpuModified(bank_first, bank_bytes)) {
            ++stats.banks_skipped;
            stats.flagged_skipped += flagged_slot ? 1 : 0;
            if (slot_stats != nullptr) [[unlikely]] {
                ++slot_stats->banks_skipped;
            }
            if (!slot_verify) {
                LogNativeBankTrackedStats(mode);
                return;
            }
            // EDEN_NX2_BANK_SKIP_VERIFY: walk it anyway, comparing every page.
            skip_verify = true;
        }
    }
    u64 mapped_size = 0;
    while (mapped_size < size &&
           device_memory.template GetPointer<u8>(device_addr + mapped_size) != nullptr) {
        mapped_size += Core::DEVICE_PAGESIZE - ((device_addr + mapped_size) & Core::DEVICE_PAGEMASK);
    }
    const DAddr first_page = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    const DAddr end = device_addr + (std::min<u64>)(mapped_size, size);
    const bool verify = mode == 2;
    const bool frame_check = mode != 3;
    // Most banks have no CPU-modified page: ask per page only when one has.
    const bool any_cpu_dirty = memory_tracker.IsRegionCpuModified(first_page, end - first_page);
    // EDEN_NX2_RUNAHEAD: as in the untracked check (BindHostGraphicsUniformBuffer).
    Buffer* const upload_to =
        (trims & 4) != 0 || VideoCommon::Runahead::g_frontier.load(std::memory_order_relaxed) != 0
            ? &buffer
            : nullptr;
    NativeBankUploads uploads;
    for (DAddr page = first_page; page < end; page += Core::DEVICE_PAGESIZE) {
        VideoCommon::OneFollower::BeforeEngineRead(page, Core::DEVICE_PAGESIZE);
        const u8* const guest = device_memory.template GetPointer<u8>(page);
        if (guest == nullptr) {
            continue;
        }
        if (slot_stats != nullptr) [[unlikely]] {
            ++slot_stats->pages;
        }
        // Take the flags before reading the page: a write flagged later is compared next bind.
        const bool collected = Core::g_collected_write_pages.Take(page);
        const bool written = Core::g_buffer_write_pages.Take(page);
        const NativeBankSnapshotRef ref = NativeBankSnapshot(page);
        NativeBankPage& snapshot = *ref.page;
        if (ref.inserted) {
            ++stats.pages_new;
            if (slot_stats != nullptr) [[unlikely]] {
                ++slot_stats->pages_new;
            }
            if (census != 0) [[unlikely]] {
                // FRITCHII_BANK_CENSUS: the census' marks start at this snapshot.
                (void)census_uploads->Take(page);
                (void)Common::g_ring_write_pages.Take(page);
                snapshot.census_ring_seq = census >= 2 ? Common::WriteRing::Sequence() : 0;
                snapshot.census_flag = collected ? 1 : (written ? 2 : 0);
                snapshot.census_flag_bind = collected || written ? native_bank_census_bind : 0;
                snapshot.census_last_reason = static_cast<u8>(BankNone);
            }
            std::memcpy(snapshot.words.data(), guest, Core::DEVICE_PAGESIZE);
            snapshot.checked_frame = frame_tick;
            SetNativeBankRecheck(ref, page, collected || written);
            NoteNativeBankPageChanged(page, snapshot.words.data(), upload_to, uploads);
            continue;
        }
        // FRITCHII_BANK_FLAGGED=2: a page whose change was "missed" is watched for its next 8
        // visits: a flag then is the late mark of a writer that ran during the bind (its bytes
        // landed before its mark); none means a writer no mark covers.
        if (snapshot.missed_watch != 0) [[unlikely]] {
            if (collected || written) {
                snapshot.missed_watch = 0;
                ++stats.flagged_missed_then_flagged;
            } else if (--snapshot.missed_watch == 0) {
                ++stats.flagged_missed_never_flagged;
            }
        }
        // A flagged write's bytes may land after this compare: compare again at the next bind.
        const bool recheck = snapshot.recheck;
        if (recheck != (collected || written)) {
            SetNativeBankRecheck(ref, page, collected || written);
        }
        // Changed lately: likely per-draw data, rewritten by paths none of the flags see
        // (EXP9=2 found ~1 such change in 15 s without this rule). Not for EDEN_NX2_BANK_SLOTS.
        const bool hot = snapshot.changed_frame != ~u64{0} &&
                         frame_tick - snapshot.changed_frame < HotFrames;
        NativeBankReason reason = BankNone;
        if (collected) {
            reason = BankCollected;
        } else if (written) {
            reason = BankWritten;
        } else if (recheck) {
            reason = BankRecheck;
        } else if (any_cpu_dirty &&
                   memory_tracker.IsRegionCpuModified(page, Core::DEVICE_PAGESIZE)) {
            reason = BankCpuDirty;
        } else if (hot && !own_slot) {
            reason = BankHot;
        } else if (frame_check && snapshot.checked_frame != frame_tick) {
            reason = BankFrame;
        }
        if (skip_verify && reason != BankNone) [[unlikely]] {
            // The whole-bank skip passed over a page the loop has a reason for. The skip's flag
            // looks (Any) come before the loop's takes: a writer on another thread can mark in
            // between, and its mark stays for the next bind. Counted by the reason found.
            ++stats.skip_disagreed;
            if (flagged_slot) {
                ++stats.flagged_disagreed;
                ++stats.flagged_disagreed_by[reason];
            }
        }
        const bool own_hot = reason == BankNone && own_slot && hot;
        if (own_hot) {
            ++stats.hot_left;
            stats.flagged_hot_left += flagged_slot ? 1 : 0;
            if (slot_stats != nullptr) [[unlikely]] {
                ++slot_stats->hot_left;
            }
        }
        // EDEN_NX2_BANK_SKIP_VERIFY: what the slot rule left alone is compared all the same.
        // FRITCHII_BANK_FLAGGED=2: every page of its slots without a reason, not only the hot ones
        // (a change no flag announced is "missed" whether or not the hot rule would have seen it).
        const bool verify_skip =
            reason == BankNone && (skip_verify || (own_hot && slot_verify) ||
                                   (flagged_slot && own_slot && flagged == 2));
        const bool first_this_frame = snapshot.checked_frame != frame_tick;
        if (reason != BankNone) {
            snapshot.checked_frame = frame_tick;
        } else if (!verify && !verify_skip) {
            ++stats.pages_skipped;
            continue;
        }
        // FRITCHII_BANK_CENSUS: what this compare's change (if any) is attributed to, taken
        // before the bytes are read, as the write flags are.
        bool census_uploaded = false;
        bool census_ring_marked = false;
        u64 census_ring_since = 0;
        if (census != 0) [[unlikely]] {
            census_uploaded = census_uploads->Take(page);
            census_ring_since = snapshot.census_ring_seq;
            if (census >= 2) {
                // The ring's page mark: an emulator write since the last compare, even when the
                // ring wrapped over its entry.
                census_ring_marked = Common::g_ring_write_pages.Take(page);
                snapshot.census_ring_seq = Common::WriteRing::Sequence();
            }
            if (collected || written) {
                snapshot.census_flag = collected ? 1 : 2;
                snapshot.census_flag_bind = native_bank_census_bind;
            }
        }
        if (verify_skip) {
            ++stats.skip_verified;
            stats.flagged_verified += flagged_slot ? 1 : 0;
        } else {
            ++stats.compared[reason];
            if (slot_stats != nullptr) [[unlikely]] {
                ++slot_stats->compared[reason];
            }
        }
        // EDEN_NX2_EXP26 & 2: an inline SSE2 compare instead of the ucrt memcmp.
        if ((trims & 2) != 0
                ? VideoCommon::Nx2BytesEqual(snapshot.words.data(), guest, Core::DEVICE_PAGESIZE)
                : std::memcmp(snapshot.words.data(), guest, Core::DEVICE_PAGESIZE) == 0) {
            if (census != 0) [[unlikely]] {
                snapshot.census_last_reason = static_cast<u8>(reason);
            }
            continue;
        }
        if (census != 0) [[unlikely]] {
            // Unflagged: found by the hot rule, the frame check or a verify compare only.
            const bool unflagged = verify_skip || reason == BankHot || reason == BankFrame ||
                                   reason == BankNone;
            NoteNativeBankCensusChange(page, snapshot, census, reason, unflagged,
                                       census_uploaded, census_ring_marked, census_ring_since,
                                       first_this_frame, index);
            if (verify_skip) {
                ++native_bank_census.skip_missed;
            }
            snapshot.census_last_reason = static_cast<u8>(reason);
        }
        if (verify_skip) {
            // Missed by the slot rule: taken as any change is (marked), and counted.
            ++stats.skip_missed;
            if (flagged_slot) {
                ++stats.flagged_missed;
                snapshot.missed_watch = 8;
            }
        } else {
            ++stats.changed[reason];
            if (slot_stats != nullptr) [[unlikely]] {
                ++slot_stats->changed[reason];
            }
        }
        snapshot.changed_frame = frame_tick;
        std::memcpy(snapshot.words.data(), guest, Core::DEVICE_PAGESIZE);
        NoteNativeBankPageChanged(page, snapshot.words.data(), upload_to, uploads);
    }
    if (uploads.count != 0) {
        FlushNativeBankUploads(buffer, uploads);
    }
    LogNativeBankTrackedStats(mode);
}

template <class P>
void BufferCache<P>::LogNativeBankTrackedStats(int mode) {
    auto& stats = native_bank_tracked_stats;
    const bool verify = mode == 2;
    // The clock only every 1,024 binds (~100k binds a second at Castle Town).
    if ((stats.binds & 1023) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (VideoCore::Nx2Debug::g_bank_census.load(std::memory_order_relaxed) > 0) [[unlikely]] {
        LogNativeBankCensus(now);
    }
    if (now - stats.last_log < std::chrono::seconds{5}) {
        return;
    }
    // With EXP9=2 the frame check runs too: "missed" counts what 1 would have skipped, and
    // "missed without the frame check" adds what only the frame check found (3's misses).
    const auto& compared = stats.compared;
    const auto& changed = stats.changed;
    LOG_INFO(HW_GPU,
             "Native bank check (EXP9={}, 5 s): {} binds, {} new pages, {} skipped; compared / "
             "changed: collected CPU writes {} / {}, buffer cache writes {} / {}, recheck {} / {}, "
             "CPU-modified {} / {}, hot {} / {}, frame check {} / {}, verify only {} / {}; missed "
             "{}, missed without the frame check {}",
             mode, stats.binds, stats.pages_new, stats.pages_skipped, compared[BankCollected],
             changed[BankCollected], compared[BankWritten], changed[BankWritten],
             compared[BankRecheck], changed[BankRecheck], compared[BankCpuDirty],
             changed[BankCpuDirty], compared[BankHot], changed[BankHot], compared[BankFrame],
             changed[BankFrame], compared[BankNone],
             changed[BankNone], verify ? changed[BankNone] : 0,
             verify ? changed[BankNone] + changed[BankFrame] : 0);
    // EDEN_NX2_BANK_SLOT_STATS: the same window per constant buffer index. "hot" changes are the
    // ones no flag saw; "written" ones are writes the buffer cache saw (our own constant buffer
    // words, inline uploads, downloads); "collected" ones are guest CPU writes.
    std::string slot_lines;
    for (size_t slot = 0; slot < native_bank_slot_stats.size(); ++slot) {
        const NativeBankSlotStats& s = native_bank_slot_stats[slot];
        if (s.binds == 0) {
            continue;
        }
        const auto& c = s.compared;
        const auto& d = s.changed;
        slot_lines += fmt::format(
            " | c{} (stages {:#x}): {} binds ({} skipped whole), {} pages, {} new; compared / "
            "changed: collected {} / {}, written {} / {}, recheck {} / {}, CPU-modified {} / {}, "
            "hot {} / {} ({} hot left by BANK_SLOTS), frame {} / {}, verify only {} / {}",
            slot, s.stages, s.binds, s.banks_skipped, s.pages, s.pages_new, c[BankCollected],
            d[BankCollected], c[BankWritten], d[BankWritten], c[BankRecheck], d[BankRecheck],
            c[BankCpuDirty], d[BankCpuDirty], c[BankHot], d[BankHot], s.hot_left, c[BankFrame],
            d[BankFrame], c[BankNone], d[BankNone]);
    }
    if (!slot_lines.empty()) {
        LOG_INFO(HW_GPU, "Native bank slots (EXP9={}, 5 s){}", mode, slot_lines);
    }
    // EDEN_NX2_BANK_SLOTS: what the slot rule skipped, and with EDEN_NX2_BANK_SKIP_VERIFY what
    // comparing it anyway found ("missed" and "disagreed" must stay 0).
    if (stats.banks_skipped != 0 || stats.hot_left != 0 || stats.skip_verified != 0) {
        LOG_INFO(HW_GPU,
                 "Native bank slot skip (BANK_SLOTS={:#x}, 5 s): {} binds skipped whole, {} hot "
                 "pages not compared; verify: {} compared, {} missed, {} disagreed",
                 VideoCore::Nx2Debug::g_bank_slots.load(std::memory_order_relaxed),
                 stats.banks_skipped, stats.hot_left, stats.skip_verified, stats.skip_missed,
                 stats.skip_disagreed);
    }
    // FRITCHII_BANK_FLAGGED: the same for its slots alone, with the snapshot refreshes uploads
    // made. With =2: "missed" pages whose page was flagged within its next 8 visits were writers
    // running during the bind (bytes before mark); "never flagged" must stay 0. "Disagreed" by
    // reason: collected or buffer cache marks set between the skip's look and the loop's take.
    if (stats.flagged_binds != 0) {
        const auto& by = stats.flagged_disagreed_by;
        LOG_INFO(HW_GPU,
                 "Native bank flagged (FRITCHII_BANK_FLAGGED={}, slots {:#x}, 5 s): {} binds, {} "
                 "skipped whole, {} hot pages not compared; verify: {} compared, {} missed (then "
                 "flagged {}, never flagged {}), {} disagreed (collected {}, buffer cache {}, "
                 "recheck {}, CPU-modified {}); uploads refreshed {} pages ({} changed); "
                 "snapshots dropped {} times so far (switch on, lost guest store ranges)",
                 VideoCore::Nx2Debug::g_bank_flagged.load(std::memory_order_relaxed),
                 VideoCore::Nx2Debug::g_bank_flagged_slots.load(std::memory_order_relaxed),
                 stats.flagged_binds, stats.flagged_skipped, stats.flagged_hot_left,
                 stats.flagged_verified, stats.flagged_missed, stats.flagged_missed_then_flagged,
                 stats.flagged_missed_never_flagged, stats.flagged_disagreed, by[BankCollected],
                 by[BankWritten], by[BankRecheck], by[BankCpuDirty], stats.refreshed_pages,
                 stats.refreshed_changed, native_bank_flagged_resets);
    }
    native_bank_slot_stats = {};
    stats = {};
    stats.last_log = now;
}

template <class P>
void BufferCache<P>::RefreshNativeBankSnapshots(DAddr device_addr, u64 size, const u8* bytes) {
    auto& stats = native_bank_tracked_stats;
    const DAddr end = device_addr + size;
    for (DAddr address = device_addr; address < end;) {
        const DAddr page = Common::AlignDown(address, Core::DEVICE_PAGESIZE);
        const DAddr next = (std::min)(end, page + Core::DEVICE_PAGESIZE);
        if (NativeBankPage* const snapshot = FindNativeBankSnapshot(page)) {
            u8* const words = snapshot->words.data() + (address - page);
            const u8* const staged = bytes + (address - device_addr);
            const size_t amount = static_cast<size_t>(next - address);
            ++stats.refreshed_pages;
            if (std::memcmp(words, staged, amount) != 0) {
                std::memcpy(words, staged, amount);
                // As when the bank check finds the change itself: the page is hot again, so the
                // slots that keep the hot rule compare it as before.
                snapshot->changed_frame = frame_tick;
                ++stats.refreshed_changed;
            }
        }
        address = next;
    }
}

template <class P>
void BufferCache<P>::NoteNativeBankCensusChange(DAddr page, const NativeBankPage& snapshot,
                                                int census, NativeBankReason reason,
                                                bool unflagged, bool uploaded, bool ring_marked,
                                                u64 ring_since, bool first_this_frame,
                                                u32 index) {
    auto& c = native_bank_census;
    ++c.changed[reason];
    if (!unflagged) {
        // Found by a flag or the tracker's CPU-modified state: that names its writer.
        return;
    }
    ++c.unflagged;
    if (index < c.slots.size()) {
        ++c.slots[index];
    }
    // A flag taken lately, its bytes landing after that compare and the one recheck after it
    // (the trapping writers mark before they store).
    const u64 since_flag =
        snapshot.census_flag_bind != 0 ? native_bank_census_bind - snapshot.census_flag_bind
                                       : ~u64{0};
    const size_t bucket = since_flag == ~u64{0} ? 4
                          : since_flag <= 4     ? 0
                          : since_flag <= 32    ? 1
                          : since_flag <= 256   ? 2
                                                : 3;
    ++c.flag_distance[bucket];
    const bool flag_late = since_flag <= 32;
    if (flag_late) {
        ++c.flag_late;
        if (snapshot.census_flag == 2) {
            ++c.flag_late_buffer;
        }
    }
    // An upload copied guest memory to the host buffer after the snapshot was taken: the host
    // may hold these bytes already (a guest store between the last compare and the upload).
    if (uploaded) {
        ++c.uploaded;
        if (snapshot.census_last_reason == static_cast<u8>(BankCpuDirty)) {
            ++c.uploaded_after_dirty;
        }
    }
    const bool gpu_written =
        Tegra::Nx2MayBeGpuWritten(page, Core::DEVICE_PAGESIZE) ||
        memory_tracker.IsRegionEverGpuModified(page, Core::DEVICE_PAGESIZE);
    if (gpu_written) {
        ++c.gpu_written;
    }
    // FRITCHII_BANK_CENSUS=2: the newest emulator write over the page since the last compare.
    // The ring's page mark says whether there was one at all; the ring names it unless it
    // wrapped over the entry.
    bool ring_found = false;
    if (census >= 2 && ring_marked) {
        const u64 page_end = page + Core::DEVICE_PAGESIZE;
        Common::WriteRing::Entry newest{};
        const bool complete =
            Common::WriteRing::ForEachSince(ring_since, [&](const Common::WriteRing::Entry& e) {
                if (e.address < page_end && page < e.address + e.size) {
                    newest = e;
                    ring_found = true;
                    return true;
                }
                return false;
            });
        if (ring_found) {
            ++c.ring_found;
            const char* const site = newest.site != nullptr ? newest.site : newest.context;
            auto it = std::ranges::find_if(c.writers, [&](const auto& w) {
                return w.count != 0 && w.site == site && w.via == newest.via;
            });
            if (it == c.writers.end()) {
                it = std::ranges::find_if(c.writers, [](const auto& w) { return w.count == 0; });
            }
            if (it != c.writers.end()) {
                it->site = site;
                it->via = newest.via;
                ++it->count;
            } else {
                ++c.writers_other;
            }
        } else {
            // An emulator write landed on the page, its entry gone (the ring wrapped) or, when
            // complete, written before the compare but recorded after it.
            ++c.ring_lost;
            c.ring_wrapped += complete ? 0 : 1;
        }
    }
    // Unseen: no late flag, no upload after the compare, no emulator write (2; with 1 the ring's
    // writers count here too). A GPU write mark on the page explains nothing by itself.
    const bool ring_writer = census >= 2 ? ring_marked : false;
    if (!flag_late && !uploaded && !ring_writer) {
        ++c.unseen;
        if (first_this_frame) {
            ++c.unseen_first;
        }
        if (gpu_written) {
            ++c.unseen_gpu_written;
        }
        if (index < c.slots_unseen.size()) {
            ++c.slots_unseen[index];
        }
    }
}

template <class P>
void BufferCache<P>::LogNativeBankCensus(std::chrono::steady_clock::time_point now) {
    auto& c = native_bank_census;
    if (now - c.last_log < std::chrono::seconds{10}) {
        return;
    }
    const auto& d = c.changed;
    std::string slots;
    for (size_t slot = 0; slot < c.slots.size(); ++slot) {
        if (c.slots[slot] != 0) {
            slots += fmt::format(" c{} {} / {}", slot, c.slots[slot], c.slots_unseen[slot]);
        }
    }
    std::string writers;
    for (const auto& w : c.writers) {
        if (w.count != 0) {
            writers += fmt::format(" | {} / {}: {}", w.site != nullptr ? w.site : "no site",
                                   w.via != nullptr ? w.via : "?", w.count);
        }
    }
    if (c.writers_other != 0) {
        writers += fmt::format(" | others: {}", c.writers_other);
    }
    const auto& f = c.flag_distance;
    LOG_INFO(HW_GPU,
             "Native bank census (FRITCHII_BANK_CENSUS={}, 10 s): changes: collected {}, buffer "
             "cache {}, recheck {}, CPU-modified {}, hot {}, frame check {}, verify only {} (slot "
             "skip missed {}); unflagged {}: flag taken <=4 / <=32 / <=256 / more / never binds "
             "before {} / {} / {} / {} / {} (late flag {}, of them buffer cache {}), uploaded since "
             "the last compare {} (that compare CPU-modified {}), ever GPU-written {}, ring writer "
             "named {}, ring writer lost {} (ring wrapped {}), unseen {} (first compare this frame "
             "{}, ever GPU-written {}); slots (unflagged / unseen):{}{}",
             VideoCore::Nx2Debug::g_bank_census.load(std::memory_order_relaxed),
             d[BankCollected], d[BankWritten], d[BankRecheck], d[BankCpuDirty], d[BankHot],
             d[BankFrame], d[BankNone], c.skip_missed, c.unflagged, f[0], f[1], f[2], f[3], f[4],
             c.flag_late, c.flag_late_buffer, c.uploaded, c.uploaded_after_dirty, c.gpu_written,
             c.ring_found, c.ring_lost, c.ring_wrapped, c.unseen, c.unseen_first,
             c.unseen_gpu_written, slots.empty() ? " none" : slots, writers);
    c = {};
    c.last_log = now;
}

template <class P>
void BufferCache<P>::NoteNativeBankPageChanged(DAddr page, [[maybe_unused]] const u8* words,
                                               [[maybe_unused]] Buffer* upload_to,
                                               [[maybe_unused]] NativeBankUploads& uploads) {
    // FRITCHII_BIND_SKIP: its verify sees whether a bank check found a change.
    ++bind_skip_bank_changes;
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        // EDEN_NX2_EXP26 & 4 (the original's UploadNativeBank): copy the page from its
        // snapshot now and leave its tracking alone (marking it CPU-modified costs two
        // VirtualProtect calls). Only for a page that lies whole in the bound buffer and holds no
        // GPU-written data, so the guest bytes are what its host copy should hold.
        if (upload_to != nullptr && uploads.count < uploads.pages.size() &&
            page >= upload_to->CpuAddr() &&
            page + Core::DEVICE_PAGESIZE <= upload_to->CpuAddr() + upload_to->SizeBytes() &&
            !memory_tracker.IsRegionGpuModified(page, Core::DEVICE_PAGESIZE) &&
            !gpu_modified_ranges.Intersects(page, Core::DEVICE_PAGESIZE) &&
            !AheadHere(page, Core::DEVICE_PAGESIZE)) {
            uploads.pages[uploads.count++] = NativeBankUpload{.page = page, .words = words};
            return;
        }
    }
    memory_tracker.MarkRegionAsCpuModified(page, Core::DEVICE_PAGESIZE);
}

template <class P>
void BufferCache<P>::FlushNativeBankUploads([[maybe_unused]] Buffer& buffer,
                                            [[maybe_unused]] const NativeBankUploads& uploads) {
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(uploads.count * Core::DEVICE_PAGESIZE);
        std::array<BufferCopy, NativeBankUploadCapacity> copies{};
        size_t num_copies = 0;
        u64 staging_offset = 0;
        for (size_t i = 0; i < uploads.count; ++i) {
            const NativeBankUpload& upload = uploads.pages[i];
            // From the snapshot, not guest memory again: the host buffer then holds exactly the
            // bytes compared, and a later guest write shows at the next compare.
            std::memcpy(upload_staging.mapped_span.data() + staging_offset, upload.words,
                        Core::DEVICE_PAGESIZE);
            const u64 dst_offset = buffer.Offset(upload.page);
            if (num_copies != 0 &&
                copies[num_copies - 1].dst_offset + copies[num_copies - 1].size == dst_offset) {
                copies[num_copies - 1].size += Core::DEVICE_PAGESIZE;
            } else {
                copies[num_copies++] = BufferCopy{
                    .src_offset = upload_staging.offset + staging_offset,
                    .dst_offset = dst_offset,
                    .size = Core::DEVICE_PAGESIZE,
                };
            }
            staging_offset += Core::DEVICE_PAGESIZE;
        }
        const std::span<const BufferCopy> copy_span(copies.data(), num_copies);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copy_span);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copy_span, true, can_reorder);
    }
}

template <class P>
void BufferCache<P>::SelectNativeBankStore() {
    const bool table = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bank_page_table,
                                                        "EDEN_NX2_BANK_PAGE_TABLE",
                                                        NativeBankDefaultsForOcarina() ? 1 : 0) != 0;
    if (table != native_bank_table_on) [[unlikely]] {
        // Both emptied: every page's next bind takes a new snapshot and marks the page changed.
        ClearNativeBankSnapshots();
        native_bank_table_on = table;
    }
    const size_t pages =
        table ? native_bank_table_pages : native_indirect_page_snapshots.size();
    if (pages >= 16384) {
        ClearNativeBankSnapshots();
    }
}

template <class P>
typename BufferCache<P>::NativeBankSnapshotRef BufferCache<P>::NativeBankSnapshot(DAddr page) {
    static_assert(Tegra::MaxwellDeviceTraits::device_virtual_bits <= NativeBankAddressBits);
    if (!native_bank_table_on || (page >> NativeBankAddressBits) != 0) [[unlikely]] {
        auto [it, inserted] = native_indirect_page_snapshots.try_emplace(page);
        return NativeBankSnapshotRef{&it->second, nullptr, inserted};
    }
    const u64 region_index = page >> NativeBankRegionBits;
    NativeBankRegion* region = native_bank_last_region;
    if (region_index != native_bank_last_region_index) [[unlikely]] {
        if (!native_bank_regions) {
            native_bank_regions = std::make_unique<NativeBankRegionTable>();
        }
        std::unique_ptr<NativeBankRegion>& entry = (*native_bank_regions)[region_index];
        if (!entry) {
            entry = std::make_unique<NativeBankRegion>();
            native_bank_region_list.push_back(static_cast<u32>(region_index));
        }
        region = entry.get();
        native_bank_last_region_index = region_index;
        native_bank_last_region = region;
    }
    const u64 slot = (page >> Core::DEVICE_PAGEBITS) & (NativeBankRegionPages - 1);
    std::unique_ptr<NativeBankPage>& snapshot = region->pages[slot];
    if (snapshot) [[likely]] {
        return NativeBankSnapshotRef{snapshot.get(), region, false};
    }
    // Default-initialized words: the caller copies the page in before any compare.
    snapshot.reset(new NativeBankPage);
    region->present[slot / 64] |= u64{1} << (slot % 64);
    ++native_bank_table_pages;
    return NativeBankSnapshotRef{snapshot.get(), region, true};
}

template <class P>
typename BufferCache<P>::NativeBankPage* BufferCache<P>::FindNativeBankSnapshot(DAddr page) {
    if (!native_bank_table_on || (page >> NativeBankAddressBits) != 0) {
        const auto it = native_indirect_page_snapshots.find(page);
        return it != native_indirect_page_snapshots.end() ? &it->second : nullptr;
    }
    if (!native_bank_regions) {
        return nullptr;
    }
    const NativeBankRegion* const region =
        (*native_bank_regions)[page >> NativeBankRegionBits].get();
    if (region == nullptr) {
        return nullptr;
    }
    return region->pages[(page >> Core::DEVICE_PAGEBITS) & (NativeBankRegionPages - 1)].get();
}

template <class P>
bool BufferCache<P>::NativeBankPagesSettled(DAddr first_page, DAddr end) {
    if (!native_bank_table_on || !native_bank_regions || first_page >= end ||
        ((end - 1) >> NativeBankAddressBits) != 0) {
        return false;
    }
    for (DAddr page = first_page; page < end;) {
        const u64 region_index = page >> NativeBankRegionBits;
        const NativeBankRegion* const region =
            region_index == native_bank_last_region_index
                ? native_bank_last_region
                : (*native_bank_regions)[region_index].get();
        if (region == nullptr) {
            return false;
        }
        const DAddr region_end =
            (std::min<DAddr>)(end, (region_index + 1) << NativeBankRegionBits);
        u64 slot = (page >> Core::DEVICE_PAGEBITS) & (NativeBankRegionPages - 1);
        const u64 slot_end = slot + ((region_end - page) >> Core::DEVICE_PAGEBITS);
        while (slot < slot_end) {
            const u64 first_bit = slot % 64;
            const u64 bits = (std::min<u64>)(64 - first_bit, slot_end - slot);
            const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
            if ((region->present[slot / 64] & mask) != mask ||
                (region->recheck[slot / 64] & mask) != 0) {
                return false;
            }
            slot += bits;
        }
        page = region_end;
    }
    return true;
}

template <class P>
bool BufferCache<P>::NativeBankAnySnapshot(DAddr begin, DAddr end) {
    if (begin >= end) {
        return false;
    }
    if (!native_bank_table_on || ((end - 1) >> NativeBankAddressBits) != 0) {
        // The node map: no range lookup, so any snapshot at all counts.
        return !native_indirect_page_snapshots.empty() ||
               (native_bank_table_on && native_bank_table_pages != 0);
    }
    if (!native_bank_regions || native_bank_table_pages == 0) {
        return false;
    }
    // The table's present bits, a word (64 pages) at a time.
    for (DAddr page = Common::AlignDown(begin, Core::DEVICE_PAGESIZE); page < end;) {
        const u64 region_index = page >> NativeBankRegionBits;
        const DAddr region_end_full = (region_index + 1) << NativeBankRegionBits;
        const NativeBankRegion* const region =
            region_index == native_bank_last_region_index
                ? native_bank_last_region
                : (*native_bank_regions)[region_index].get();
        if (region != nullptr) {
            const DAddr region_end = (std::min<DAddr>)(end, region_end_full);
            u64 slot = (page >> Core::DEVICE_PAGEBITS) & (NativeBankRegionPages - 1);
            const u64 last = ((region_end - 1) >> Core::DEVICE_PAGEBITS) &
                             (NativeBankRegionPages - 1);
            while (slot <= last) {
                const u64 first_bit = slot % 64;
                const u64 bits = (std::min<u64>)(64 - first_bit, last - slot + 1);
                const u64 mask = (bits == 64 ? ~u64{0} : (u64{1} << bits) - 1) << first_bit;
                if ((region->present[slot / 64] & mask) != 0) {
                    return true;
                }
                slot += bits;
            }
        }
        page = region_end_full;
    }
    return false;
}

template <class P>
void BufferCache<P>::ClearNativeBankSnapshots() {
    native_indirect_page_snapshots.clear();
    if (native_bank_regions) {
        for (const u32 region_index : native_bank_region_list) {
            (*native_bank_regions)[region_index].reset();
        }
    }
    native_bank_region_list.clear();
    native_bank_table_pages = 0;
    native_bank_last_region_index = ~u64{0};
    native_bank_last_region = nullptr;
}

template <class P>
void BufferCache<P>::BindHostGraphicsStorageBuffers(size_t stage) {
    u32 binding_index = 0;
    // EDEN_NX2_GPU_PROFILE=2: what each binding costs (zones "storage synchronize" and "storage
    // written marks" under "stage storage buffers", and the storage fine counters).
    const bool fine = Nx2Profile::Fine();
    // Both on by default since 2026-11-11: verify modes (=2) found 0 differences in ~14M checked
    // syncs each at the market, waterfall, Lost Woods and the Stalchild fight; bind stage buffers
    // 136 -> 115 ms/s on the back with both. =0 turns each off.
    // EDEN_NX2_STORAGE_SYNC_SCAN: the synchronizes below walk only the tracker words that hold
    // something to upload or re-protect.
    storage_sync_scan = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_storage_sync_scan,
                                                         "EDEN_NX2_STORAGE_SYNC_SCAN", 1);
    SCOPE_EXIT {
        storage_sync_scan = 0;
        if (ReadRunStats::g_enabled) [[unlikely]] {
            ReadRunStats::t_writer = {};
        }
    };
    // EDEN_NX2_STORAGE_SYNC_MEMO: ranges left clean are not walked again while no page of their
    // tracker managers gained a CPU or Untracked bit. Not with hot pages (always dirty), and
    // only behind a walk that leaves the range clean from the words themselves (the full walk
    // or the scanned one, not the CleanWords walk).
    const int memo_mode = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_storage_sync_memo, "EDEN_NX2_STORAGE_SYNC_MEMO", 1);
    const bool sync_memo = memo_mode != 0 && !HotPagesOn() &&
                           (storage_sync_scan == 1 || !Nx2Perf::On(Nx2Perf::Opt::CleanWords));
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        const Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        if (ReadRunStats::g_enabled) [[unlikely]] {
            ReadRunStats::t_writer = {.binding = static_cast<u32>(stage) * 64 + index,
                                      .kind = ReadRunStats::WriterKind::DrawStorage};
        }
        MemStageStats::Read(MemStageStats::Kind::Storage, binding.size);
        if (BindMultiRangeStorage(binding, is_written, graphics_segments)) {
            Nx2Profile::AddFine(Nx2Profile::FineCounter::StorageMulti, 1);
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        {
            const Nx2Profile::FineScope sync_scope{Nx2Profile::Zone::StorageSync};
            const bool no_upload =
                sync_memo ? SynchronizeStorageMemo(buffer, binding.device_addr, size,
                                                   memo_mode == 2)
                          : SynchronizeBuffer(buffer, binding.device_addr, size);
            if (fine) [[unlikely]] {
                Nx2Profile::AddFine(Nx2Profile::FineCounter::StorageKiB, size >> 10);
                Nx2Profile::AddFine(no_upload ? Nx2Profile::FineCounter::StorageNoUpload
                                              : Nx2Profile::FineCounter::StorageUploads,
                                    1);
            }
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            const Nx2Profile::FineScope mark_scope{Nx2Profile::Zone::StorageMark};
            Nx2Profile::AddFine(Nx2Profile::FineCounter::StorageWritten, 1);
            const u32 mark = WrittenStorageMarkSize(size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, mark, false, mark < size);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindStorageBuffer(stage, binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsTextureBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        MemStageStats::Read(MemStageStats::Kind::TextureBuffer, size);
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written = ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
        if (is_written) {
            if (ReadRunStats::g_enabled) [[unlikely]] {
                ReadRunStats::t_writer = {.binding = static_cast<u32>(stage) * 64 + index,
                                          .kind = ReadRunStats::WriterKind::DrawTexture};
            }
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            if (ReadRunStats::g_enabled) [[unlikely]] {
                ReadRunStats::t_writer = {};
            }
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostTransformFeedbackBuffers() {
    if (!Inputs().TransformFeedbackEnabled()) {
        return;
    }
    HostBindings<typename P::Buffer> host_bindings;
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        const Binding& binding = channel_state->transform_feedback_buffers[index];
        const auto& control = Inputs().TransformFeedbackControl(index);
        const bool has_layout = control.varying_count != 0 || control.stride != 0;

        Buffer* host_buffer = &slot_buffers[NULL_BUFFER_ID];
        u32 offset = 0;
        u32 size = 0;

        if (has_layout && binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            size = binding.size;
            SynchronizeBuffer(buffer, binding.device_addr, size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, size);
            host_buffer = &buffer;
        }

        host_bindings.buffers.push_back(host_buffer);
        host_bindings.offsets.push_back(offset);
        host_bindings.sizes.push_back(size);
    }
    runtime.BindTransformFeedbackBuffers(host_bindings);
}

template <class P>
void BufferCache<P>::BindHostComputeUniformBuffers() {
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        // Mark all uniform buffers as dirty
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->fast_bound_uniform_buffers.fill(0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        const Binding& binding = channel_state->compute_uniform_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size =
            (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
        const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
        const u32 offset = has_host_buffer ? buffer.Offset(binding.device_addr) : 0;
        const bool needs_alignment_stream = [&]() {
            if constexpr (IS_OPENGL) {
                return false;
            } else {
                if (!has_host_buffer) {
                    return false;
                }
                const u32 alignment = runtime.GetUniformBufferAlignment();
                return alignment > 1 && (offset % alignment) != 0;
            }
        }();
        // Small buffers the GPU has not written are streamed, as graphics ones are: synchronizing
        // them made every dispatch re-protect pages an inline upload had just unprotected (two
        // VirtualProtect calls per dispatch, ~4,000 dispatches a second in title 1).
        const bool use_fast_buffer =
            needs_alignment_stream ||
            (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size &&
             !memory_tracker.IsRegionGpuModified(binding.device_addr, size));
        if (MemStageStats::Enabled()) [[unlikely]] {
            MemStageStats::Read(use_fast_buffer ? MemStageStats::Kind::CUniformStream
                                                : MemStageStats::Kind::CUniformCached,
                                size);
        }
        if constexpr (!IS_OPENGL) {
            if (use_fast_buffer) {
                MemReadLog::Bytes(MemReadLog::Kind::Uniform, device_memory, binding.device_addr,
                                  size);
                const std::span<u8> span =
                    runtime.BindMappedUniformBuffer(0, binding_index, size);
                VideoCommon::OneFollower::BeforeEngineRead(binding.device_addr, size);
                device_memory.ReadBlockUnsafe(binding.device_addr, span.data(), size);
                return;
            }
        }

        SynchronizeBuffer(buffer, binding.device_addr, size);

        buffer.MarkUsage(offset, size);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            runtime.BindComputeUniformBuffer(binding_index, buffer, offset, size);
            ++binding_index;
        } else {
            runtime.BindUniformBuffer(buffer, offset, size);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeStorageBuffers() {
    // The unmapped device read lines' context (a few dispatches a frame: two stores each).
    const Common::WriteRing::ReadSiteScope read_site{"compute storage bind"};
    u32 binding_index = 0;
    // EDEN_NX2_GPU_PROFILE: bytes bound against bytes uploaded (the compute bindings line).
    const bool profile = Nx2Profile::Enabled();
    count_compute_sync = profile;
    // EDEN_NX2_EXP59 & 4: the dirty-word walk for these synchronizes only; & 8 verifies it.
    // Bits 1 and 2 belong to the read extents.
    compute_dirty_walk = VideoCore::Nx2Debug::Experiment(59) & (4 | 8);
    // EDEN_NX2_STORAGE_SYNC_SCAN, while EXP59 does not pick the walk.
    storage_sync_scan =
        compute_dirty_walk != 0
            ? 0
            : VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_storage_sync_scan,
                                               "EDEN_NX2_STORAGE_SYNC_SCAN", 1);
    SCOPE_EXIT {
        count_compute_sync = false;
        compute_dirty_walk = 0;
        storage_sync_scan = 0;
        if (ReadRunStats::g_enabled) [[unlikely]] {
            ReadRunStats::t_writer = {};
            ReadRunStats::t_proven_end = 0;
            ReadRunStats::t_footprint = {};
        }
    };
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        const Binding& binding = channel_state->compute_storage_buffers[index];
        if (ReadRunStats::g_enabled) [[unlikely]] {
            ReadRunStats::t_writer = {.program = compute_shader_hash,
                                      .binding = index,
                                      .kind = ReadRunStats::WriterKind::ComputeStorage};
            // FRITCHII_EXTENT_CBUF=3: where the cbuf proof would have ended this binding's mark
            ReadRunStats::t_proven_end =
                cbuf_dry_proven[index] != 0 ? binding.device_addr + cbuf_dry_proven[index] : 0;
            // FRITCHII_STORE_FOOTPRINT=3: this binding's footprint (none: id 0)
            ReadRunStats::t_footprint = footprint_dry[index];
        }
        if (profile) {
            Nx2Profile::Add(Nx2Profile::Counter::ComputeStorageBindings, 1);
            Nx2Profile::Add(Nx2Profile::Counter::ComputeStorageBytes, binding.size);
        }
        MemStageStats::Read(MemStageStats::Kind::CStorage, binding.size);
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        // A native program's binding is min(layout, 16 MiB) because its pointer has no size, and
        // marking all of it GPU-written defers the CPU's later writes to those pages. Mark only
        // what BindComputeStorageBuffer worked out the dispatch can write; the binding keeps its
        // size so the shader can still read all of it.
        const bool native =
            ((channel_state->native_compute_storage_buffers >> index) & 1) != 0;
        const u64 write_mark_limit =
            native ? u64{channel_state->compute_storage_write_marks[index]} : ~u64{0};
        if (BindMultiRangeStorage(binding, is_written, compute_segments, write_mark_limit)) {
            if (native && is_written) {
                ProbeExtentTail(index, binding);
            }
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (native && is_written) {
            ProbeExtentTail(index, binding);
        }

        // The lower of EDEN_NX2_WRITTEN_SSBO_MARK_KIB's cap and the native write mark.
        const u32 capped_size = WrittenStorageMarkSize(size);
        const u32 mark =
            is_written ? static_cast<u32>(std::min<u64>(capped_size, write_mark_limit)) : 0;
        if (mark != 0) {
            // FRITCHII_STORE_FOOTPRINT (armed): the mark carries the binding's footprint
            if (footprint_following) [[unlikely]] {
                mark_footprint = footprint_dry[index];
            }
            // Only the cap leaves writes past the mark; the native mark bounds them.
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, mark, true,
                              capped_size < size && capped_size <= write_mark_limit);
            mark_footprint = {};
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindComputeStorageBuffer(binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::ProbeExtentTail(u32 index, const Binding& binding) {
    // Write extent check (ExtentTrust): the tail past the proof, as it is right before this
    // dispatch (after the bind's uploads). RecordComputeExtentProbes copies it again right after.
    if constexpr (!IS_OPENGL) {
        const ExtentProbeRequest& request = extent_probe_requests[index];
        if (!request.active || request.proven >= binding.size) {
            return;
        }
        BufferId buffer_id = binding.buffer_id;
        DAddr tail = binding.device_addr + request.proven;
        u64 room = binding.size - request.proven;
        if (binding.segment_count >= 2) {
            // A binding over several buffers: its segments cover it in order.
            if (binding.segment_first > compute_segments.size() ||
                binding.segment_count > compute_segments.size() - binding.segment_first) {
                return;
            }
            u64 skip = request.proven;
            bool found = false;
            for (u32 i = 0; i < binding.segment_count; ++i) {
                const MultiRangeSegment& segment = compute_segments[binding.segment_first + i];
                if (skip < segment.size) {
                    buffer_id = segment.buffer_id;
                    tail = segment.device_addr + skip;
                    room = segment.size - skip;
                    found = true;
                    break;
                }
                skip -= segment.size;
            }
            if (!found) {
                return;
            }
        }
        if (!slot_buffers.contains(buffer_id)) {
            return;
        }
        const u64 span = (std::min<u64>)(room, 2_MiB);
        ExtentProbe probe{
            .shader_hash = compute_shader_hash,
            .buffer_id = buffer_id,
            .device_addr = tail,
            .binding_size = binding.size,
            .request = request,
            .span = span,
        };
        // What the CPU would read there without the mark, taken before the dispatch. Bytes the
        // cache already knows the GPU wrote, and CPU writes not in the buffer yet, may differ for
        // other reasons: they are left out.
        probe.guest.resize(span);
        VideoCommon::OneFollower::BeforeEngineRead(tail, span);
        const Common::WriteRing::ReadSiteScope read_site{"extent probe"};
        device_memory.ReadBlockUnsafe(tail, probe.guest.data(), span);
        const auto skip = [&](DAddr begin, DAddr end) {
            begin = (std::max)(begin, tail);
            end = (std::min)(end, tail + span);
            if (begin < end) {
                probe.skip.emplace_back(begin - tail, end - tail);
            }
        };
        ForEachGpuWrittenHere(tail, span, skip);
        PendingCpuWriteRanges().ForEachInRange(tail, span, skip);
        extent_probes.push_back(std::move(probe));
        extent_trust[compute_shader_hash].probe_pending = true;
    }
}

template <class P>
u32 BufferCache<P>::RecordComputeExtentProbes() {
    if constexpr (IS_OPENGL) {
        return 0;
    } else {
    u32 recorded = 0;
    for (ExtentProbe& probe : extent_probes) {
        if (probe.after || !slot_buffers.contains(probe.buffer_id)) {
            continue;
        }
        Buffer& buffer = slot_buffers[probe.buffer_id];
        probe.after = runtime.DownloadStagingBuffer(probe.span, true);
        const BufferCopy copy{
            .src_offset = buffer.Offset(probe.device_addr),
            .dst_offset = probe.after->offset,
            .size = probe.span,
        };
        runtime.CopyBuffer(probe.after->buffer, buffer, std::span{&copy, 1}, true);
        probe.tick = runtime.CurrentTick();
        ++recorded;
    }
    return recorded;
    }
}

template <class P>
void BufferCache<P>::CheckExtentProbes() {
    if constexpr (IS_OPENGL) {
        return;
    } else {
    std::erase_if(extent_probes, [this](ExtentProbe& probe) {
        if (!probe.after) {
            // The dispatch was dropped after its bind: the next one records the copy, and a
            // difference from it can only distrust the program (the safe side).
            return false;
        }
        if (!runtime.IsFree(probe.tick)) {
            return false;
        }
        // The GPU's tail after the dispatch against guest memory before it: a difference is GPU
        // data the cache would not have known about without the full mark.
        const u8* const guest = probe.guest.data();
        const u8* const after = probe.after->mapped_span.data();
        u64 differing = 0;
        u64 first = ~u64{0};
        u64 last = 0;
        if (std::memcmp(guest, after, probe.span) != 0) {
            std::vector<bool> skipped(probe.span);
            for (const auto& [begin, end] : probe.skip) {
                std::fill(skipped.begin() + static_cast<std::ptrdiff_t>(begin),
                          skipped.begin() + static_cast<std::ptrdiff_t>(end), true);
            }
            for (u64 i = 0; i < probe.span; ++i) {
                if (guest[i] != after[i] && !skipped[i]) {
                    ++differing;
                    first = (std::min)(first, i);
                    last = i;
                }
            }
        }
        ExtentTrust& trust = extent_trust[probe.shader_hash];
        trust.probe_pending = false;
        const ExtentProbeRequest& request = probe.request;
        if (differing != 0) {
            if (!trust.distrusted) {
                trust.distrusted = true;
                ++extent_probes_distrusted;
                LOG_WARNING(HW_GPU,
                            "Compute write extent too small: shader {:016x} pointer {:#x} pad {} "
                            "proven {} of {} bytes, block {}x{}x{} grid {}x{}x{}: {} bytes written "
                            "past the proof (first +{}, last +{}); marked in full from now on",
                            probe.shader_hash, request.pointer, request.pad, request.proven,
                            probe.binding_size, request.block[0], request.block[1],
                            request.block[2], request.grid[0], request.grid[1], request.grid[2],
                            differing, first, last);
            }
        } else if (!trust.distrusted) {
            // Trusted at these dims, and so at any dims within them. The componentwise maximum
            // with dims checked before also counts as checked (a dispatch past both is rare).
            for (size_t i = 0; i < 3; ++i) {
                trust.block[i] = (std::max)(trust.block[i], request.block[i]);
                trust.grid[i] = (std::max)(trust.grid[i], request.grid[i]);
            }
            ++extent_probes_trusted;
        }
        runtime.FreeDeferredStagingBuffer(*probe.after);
        return true;
    });
    }
}

template <class P>
void BufferCache<P>::BindHostComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->compute_texture_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        MemStageStats::Read(MemStageStats::Kind::CTextureBuffer, size);
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written =
            ((channel_state->written_compute_texture_buffers >> index) & 1) != 0;
        if (is_written) {
            if (ReadRunStats::g_enabled) [[unlikely]] {
                ReadRunStats::t_writer = {.program = compute_shader_hash,
                                          .binding = index,
                                          .kind = ReadRunStats::WriterKind::ComputeTexture};
            }
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            if (ReadRunStats::g_enabled) [[unlikely]] {
                ReadRunStats::t_writer = {};
            }
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_compute_texture_buffers >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {
    graphics_segments.clear();
    BufferOperations([&]() {
        if (is_indexed) {
            UpdateIndexBuffer();
        }
        UpdateVertexBuffers();
        UpdateTransformFeedbackBuffers();
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            UpdateUniformBuffers(stage);
            UpdateStorageBuffers(stage);
            UpdateTextureBuffers(stage);
        }
        if (Inputs().DrawIndirect()) {
            UpdateDrawIndirect();
        }
        RefreshMultiRangeSegments(graphics_segments);
    });
}

template <class P>
void BufferCache<P>::DoUpdateComputeBuffers() {
    compute_segments.clear();
    BufferOperations([&]() {
        UpdateComputeUniformBuffers();
        UpdateComputeStorageBuffers();
        UpdateComputeTextureBuffers();
        RefreshMultiRangeSegments(compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateIndexBuffer() {
    // We have to check for the dirty flags and index count
    // The index count is currently changed without updating the dirty flags
    const auto& draw_state = Inputs().DrawState();
    const auto& index_buffer_ref = draw_state.index_buffer;
    auto& flags = Inputs().Dirty();
    if (!flags[Dirty::IndexBuffer]) {
        return;
    }
    flags[Dirty::IndexBuffer] = false;
    if (!draw_state.inline_index_draw_indexes.empty()) [[unlikely]] {
        auto inline_index_size = static_cast<u32>(draw_state.inline_index_draw_indexes.size());
        u32 buffer_size = Common::AlignUp(inline_index_size, CACHING_PAGESIZE);
        if (inline_buffer_id == NULL_BUFFER_ID) [[unlikely]] {
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        if (slot_buffers[inline_buffer_id].SizeBytes() < buffer_size) [[unlikely]] {
            // Through DeleteBuffer: it is a registered buffer with an LRU entry, and draws already
            // recorded may still read it (the delayed destruction ring keeps it for them).
            DeleteBuffer(inline_buffer_id);
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        channel_state->index_buffer = Binding{
            .device_addr = 0,
            .size = inline_index_size,
            .buffer_id = inline_buffer_id,
        };
        return;
    }

    const GPUVAddr gpu_addr_begin = index_buffer_ref.StartAddress();
    const GPUVAddr gpu_addr_end = index_buffer_ref.EndAddress();
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    const u32 draw_size = (index_buffer_ref.count + index_buffer_ref.first) * u32(index_buffer_ref.FormatSizeInBytes());
    const u32 size = (std::min)(address_size, draw_size);
    const auto resolve = [&]() -> Binding {
        const std::optional<DAddr> device_addr =
            TranslateBind(BindAddrMemo(), gpu_addr_begin, index_addr_memo);
        if (size == 0 || !device_addr) {
            return NULL_BINDING;
        }
        return Binding{
            .device_addr = *device_addr,
            .size = size,
            .buffer_id = FindBuffer(*device_addr, size, false),
        };
    };
    if (update_keep != 0) [[unlikely]] {
        // FRITCHII_FILL_UPDATE_KEEP: the binding is a function of the start address and the size
        // (the translation, then FindBuffer over that range).
        channel_state->index_buffer =
            KeepGeometryBinding(NUM_VERTEX_BUFFERS, gpu_addr_begin, size, 1, resolve);
        return;
    }
    channel_state->index_buffer = resolve();
}

template <class P>
void BufferCache<P>::UpdateVertexBuffers() {
    auto& flags = Inputs().Dirty();
    if (!flags[Dirty::VertexBuffers]) {
        return;
    }
    flags[Dirty::VertexBuffers] = false;

    // One copy of the inputs for the loop (nothing in it changes the override, the engine or the
    // flags' view). A switched-off stream's dirty bit is never cleared (BindHostVertexBuffers
    // clears only enabled streams), so every pass met ~30 sticky-dirty disabled streams and
    // resolved each to NULL_BINDING: they take that binding directly now, with the same two
    // stores, and the flags are still tested live per stream (a mark raised mid-loop is seen).
    const CacheInputs inputs = Inputs();
    for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
        if (!flags[Dirty::VertexBuffer0 + index]) {
            continue;
        }
        if (update_keep == 0 && inputs.VertexStream(index).enable == 0) [[likely]] {
            if (BindAddrMemo()) {
                ++bind_addr_disabled_skips;
            }
            channel_state->vertex_buffers[index] = NULL_BINDING;
            UpdateVertexBufferSlot(index, NULL_BINDING);
            continue;
        }
        UpdateVertexBuffer(index, inputs);
    }
}

template <class P>
void BufferCache<P>::UpdateVertexBuffer(u32 index, const CacheInputs& inputs) {
    const auto& array = inputs.VertexStream(index);
    const auto& limit = inputs.VertexStreamLimit(index);
    const GPUVAddr gpu_addr_begin = array.Address();
    const bool native = inputs.NativeTuring();
    const u64 native_size = native ? inputs.NativeVertexStreamSize(index) : 0;
    if (native && array.enable != 0 && native_size > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error("Native vertex stream exceeds supported 32-bit buffer size");
    }
    const GPUVAddr gpu_addr_end = native ? gpu_addr_begin + native_size : limit.Address() + 1;
    const auto resolve = [&]() -> Binding {
        // FRITCHII_BIND_ADDR_MEMO: through the stream's memo, and a disabled stream is not
        // translated (it becomes NULL_BINDING below whatever the translation gives).
        const bool addr_memo = BindAddrMemo();
        std::optional<DAddr> device_addr;
        if (!addr_memo || array.enable != 0) {
            device_addr = TranslateBind(addr_memo, gpu_addr_begin, vertex_addr_memo[index]);
        } else {
            ++bind_addr_disabled_skips;
        }
        const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
        u32 size = address_size; // TODO: Analyze stride and number of vertices
        if (array.enable == 0 || size == 0 || !device_addr) {
            return NULL_BINDING;
        }
        if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
            size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr_begin, size));
        }
        return Binding{
            .device_addr = *device_addr,
            .size = size,
            .buffer_id = FindBuffer(*device_addr, size, false),
        };
    };
    // FRITCHII_FILL_UPDATE_KEEP: the binding is a function of the start and end addresses and the
    // enable (the translation, the mapped range, then FindBuffer over it).
    const Binding binding =
        update_keep != 0 ? KeepGeometryBinding(index, gpu_addr_begin, gpu_addr_end,
                                               static_cast<u32>(array.enable), resolve)
                         : resolve();
    channel_state->vertex_buffers[index] = binding;
    UpdateVertexBufferSlot(index, binding);
}

template <class P>
template <typename Resolve>
Binding BufferCache<P>::KeepGeometryBinding(u32 slot, GPUVAddr begin, u64 extent, u32 enable,
                                            Resolve&& resolve) {
    namespace FillCuts = VideoCommon::FillCuts;
    FillCuts::UpdateStats& stats = FillCuts::g_update;
    const bool is_index = slot == NUM_VERTEX_BUFFERS;
    GeomUpdateMemo& memo = geom_update_memo[slot];
    FillCuts::Add(is_index ? stats.index_updates : stats.vertex_updates);
    const bool time_check = stats.check_sample.Take();
    const u64 check_start = time_check ? FillCuts::Ticks() : 0;
    // Before the resolve: a page table change during it makes the memo miss next time.
    const u64 map_serial = gpu_memory->MapSerial();
    const size_t as_id = gpu_memory->GetID();
    const bool hit = memo.epoch == binding_epoch && memo.map_serial == map_serial &&
                     memo.as_id == as_id &&
                     memo.channel == static_cast<const void*>(channel_state) &&
                     memo.begin == begin && memo.extent == extent && memo.enable == enable;
    if (time_check) [[unlikely]] {
        FillCuts::Add(stats.check_ticks, FillCuts::Ticks() - check_start);
    }
    if (hit) {
        FillCuts::Add(is_index ? stats.index_hits : stats.vertex_hits);
        if (update_keep == FillCuts::On) {
            // The same inputs translate to the same range (no page table change), and FindBuffer
            // finds the same buffer over it (no buffer registered or unregistered since).
            return memo.binding;
        }
    }
    const bool time_hit = hit && stats.hit_sample.Take();
    const u64 hit_start = time_hit ? FillCuts::Ticks() : 0;
    const Binding binding = resolve();
    if (time_hit) [[unlikely]] {
        FillCuts::Add(stats.hit_ticks, FillCuts::Ticks() - hit_start);
    }
    if (hit) {
        // 2 and 3: the kept binding must be the one the resolve gives.
        FillCuts::Add(stats.compare.checked);
        if ((binding.device_addr != memo.binding.device_addr ||
             binding.size != memo.binding.size || binding.buffer_id != memo.binding.buffer_id) &&
            stats.compare.Differ()) [[unlikely]] {
            LOG_ERROR(HW_GPU,
                      "Fill update keep (FRITCHII_FILL_UPDATE_KEEP={}): {} {} kept {:#x}+{:#x} "
                      "buffer {}, the resolve gives {:#x}+{:#x} buffer {}",
                      update_keep, is_index ? "index buffer" : "vertex slot", slot,
                      memo.binding.device_addr, memo.binding.size, memo.binding.buffer_id.index,
                      binding.device_addr, binding.size, binding.buffer_id.index);
        }
    }
    // binding_epoch after the resolve: a buffer it created is part of what the memo holds.
    memo = GeomUpdateMemo{
        .epoch = binding_epoch,
        .map_serial = map_serial,
        .as_id = as_id,
        .channel = channel_state,
        .begin = begin,
        .extent = extent,
        .enable = enable,
        .binding = binding,
    };
    return binding;
}

template <class P>
void BufferCache<P>::UpdateDrawIndirect() {
    const auto update = [this](GPUVAddr gpu_addr, size_t size, Binding& binding) {
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!device_addr) {
            binding = NULL_BINDING;
            return;
        }
        binding = Binding{
            .device_addr = *device_addr,
            .size = static_cast<u32>(size),
            .buffer_id = FindBuffer(*device_addr, static_cast<u32>(size), false),
        };
    };
    const auto* const indirect = Inputs().DrawIndirect();
    if (indirect->include_count) {
        update(indirect->count_start_address, sizeof(u32), channel_state->count_buffer_binding);
    }
    update(indirect->indirect_start_address, indirect->buffer_size,
           channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::UpdateUniformBuffers(size_t stage) {
    if (channel_state->enabled_uniform_buffer_masks[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.buffer_id) {
            // Already updated
            return;
        }
        // Mark as dirty
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        // Resolve buffer
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        ResolveStorageBinding(stage, index, binding, is_written, graphics_segments);
    });
}

template <class P>
void BufferCache<P>::ResolveStorageBinding(size_t memo_stage, u32 index, Binding& binding,
                                           bool is_written, std::vector<MultiRangeSegment>& pool) {
    StorageBindingMemo& memo = storage_binding_memo[memo_stage][index];
    // EDEN_NX2_EXP18 & 2: what FindBuffer and a ResolveMultiRangeStorage finding one range gave
    // last time, while the binding, the mappings and the set of buffers are unchanged.
    const bool use_memo = (VideoCore::Nx2Debug::Experiment(18) & 2) != 0 &&
                          !Settings::IsGPUFenceBehaviorAccurate();
    // Before the lookup: a mapping change during it leaves a memo that never matches.
    const u64 map_serial = gpu_memory->MapSerial();
    const size_t as_id = gpu_memory->GetID();
    if (use_memo && memo.epoch == binding_epoch && memo.device_addr == binding.device_addr &&
        memo.gpu_addr == binding.gpu_addr && memo.size == binding.size &&
        memo.is_written == is_written && memo.map_serial == map_serial && memo.as_id == as_id &&
        memo.channel == static_cast<const void*>(channel_state)) {
        binding.buffer_id = memo.buffer_id;
        binding.segment_first = 0;
        binding.segment_count = 0;
        return;
    }
    binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    ResolveMultiRangeStorage(binding, is_written, pool);
    memo.epoch = 0;
    if (use_memo && binding.segment_count == 0) {
        // binding_epoch after the lookup: a buffer it created is part of what the memo holds.
        memo = StorageBindingMemo{
            .epoch = binding_epoch,
            .map_serial = map_serial,
            .as_id = as_id,
            .channel = channel_state,
            .device_addr = binding.device_addr,
            .gpu_addr = binding.gpu_addr,
            .size = binding.size,
            .is_written = is_written,
            .buffer_id = binding.buffer_id,
        };
    }
}

template <class P>
void BufferCache<P>::UpdateTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->texture_buffers[stage][index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffers() {
    if (!Inputs().TransformFeedbackEnabled()) {
        return;
    }
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        UpdateTransformFeedbackBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffer(u32 index) {
    const auto& binding = Inputs().TransformFeedbackBuffer(index);
    const GPUVAddr gpu_addr = binding.Address() + binding.start_offset;
    const u32 size = binding.size;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (binding.enable == 0 || size == 0 || !device_addr) {
        channel_state->transform_feedback_buffers[index] = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->transform_feedback_buffers[index] = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
void BufferCache<P>::UpdateComputeUniformBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        Binding& binding = channel_state->compute_uniform_buffers[index];
        binding = NULL_BINDING;
        const auto& launch_desc = Inputs().LaunchDescription();
        if (((launch_desc.const_buffer_enable_mask >> index) & 1) != 0) {
            const auto& cbuf = launch_desc.const_buffer_config[index];
            const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(cbuf.Address());
            if (device_addr) {
                binding.device_addr = *device_addr;
                binding.size = cbuf.size;
            }
        }
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeStorageBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        ResolveStorageBinding(NUM_STAGES, index, binding, is_written, compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_texture_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync,
                                       bool capped) {
    // FRITCHII_ASYNC_COMPUTE_SKIP: these bytes are no longer last written by a skipped dispatch
    // (in stream order, wherever the mark is made).
    VideoCommon::ComputeSkipHook::NoteWrite(device_addr, size);
    // FRITCHII_MEMSTAGE_MARKS_AHEAD: GPUMem's own fill puts the mark on the list ("Marks ahead of
    // the back", below). Anywhere else the whole of it runs here, after the due promotions.
    if (marks_ahead.load(std::memory_order_relaxed)) [[unlikely]] {
        if (MarkWrittenAhead(buffer_id, device_addr, size, needs_sync, capped)) {
            return;
        }
    }
    PromoteAhead();
    if (VideoCommon::GpuSplit::RecordingDispatch()) [[unlikely]] {
        VideoCommon::GpuSplit::Get().RecordDispatchWrite(device_addr, size);
    }
    if (VideoCommon::Nx2Profile::Fine()) [[unlikely]] {
        ++nx2_written_marks;
        nx2_written_mark_bytes += size;
    }
    // FRITCHII_READ_RUNS: the write mark, with the writer its bind set (the "Read writer" lines).
    if (ReadRunStats::g_enabled) [[unlikely]] {
        ReadRunStats::NoteWrite(device_addr, size, ReadWriteStamp());
    }
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    // The downloads already taken over this range hold older bytes: when their fences pop
    // they must not write them over this write's result in guest memory, nor drop the range
    // from gpu_modified_ranges (the next fence downloads it instead).
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    OutdateReadDownloads(device_addr, size);
    memory_tracker.MarkRegionAsGpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    if (mark_footprint.id != 0) [[unlikely]] {
        // FRITCHII_STORE_FOOTPRINT: who made these bytes GPU-modified (the shadow first)
        footprint_shadow.AddClaimed(mark_footprint.id, mark_footprint.base, device_addr, size);
        gpu_modified_ranges.Add(device_addr, size, true);
    } else {
        gpu_modified_ranges.Add(device_addr, size);
    }
    const bool ampere = Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
    if (ampere) {
        // Before QueueFenceDownload arms read traps: guest reads skip the buffer cache's lock on
        // pages without the mark (RasterizerVulkan::GetFlushArea), and the arm is what ends a
        // core's read area.
        Tegra::Nx2MarkGpuWritten(device_addr, size);
    }
    QueueFenceDownload(device_addr, size);
    if (ampere) {
        Nx2NoteListWrite(buffer_id, device_addr, size, capped);
    }
}

// ---------------------------------------------------------------------------------------------
// Marks ahead of the back: FRITCHII_MEMSTAGE_MARKS_AHEAD (live, read at a frame end; default 0;
// it runs only with the window open: FRITCHII_MEMSTAGE_THREAD=2 and FRITCHII_MEMSTAGE_AHEAD > 0).
// Fable's design (fable_stage_compute.md, part 1 with its section 10).
//
// Off, a draw whose fill writes buffers (storage writes, transform feedback) makes GPUMem come
// level first (StageLevel, "writing draw"): the back's served calls and fences read the
// GPU-written marks, so no packet ahead of the back may carry new ones. On, that wait goes:
//   - every existing set keeps meaning "as of the back": gpu_modified_ranges, the uncommitted
//     and committed fence sets, pending_download_ranges, the read downloads, the read traps;
//   - MarkWrittenBuffer on GPUMem's own fill (MemoryStageLink::JoinNeedsTest: GPUMem, not in a
//     served call, the window open) runs at once the steps GPUMem's later fills need: the
//     dispatch write record, the buffer's write tick, the tracker's GPU-modified bit, the vertex
//     buffers' stale flags, the page bits that never clear. The rest (the taken fence downloads'
//     subtract, the read downloads' outdate, gpu_modified_ranges, the fence download and its
//     read traps, the command list note) waits on `ahead_writes`, keyed by the push index of the
//     ring B packet the fill makes: ring B's packet order, the back's issue order (ring B is
//     taken strictly in order, and a deferred draw's packet stays in it until the back issues
//     the draw);
//   - the back promotes them right after it played that packet (the header's flag,
//     NoteRingPlayed): under the lock the deferred steps run with today's code, in packet order.
//     It never blocks inside the lock for it: while GPUMem holds the lock (its fill) the back
//     retries and serves GPUMem's requests, and a GPUMem that waits for the back meanwhile (a
//     sync's or a join's catch-up) promotes for it. Every reader of the real sets promotes first;
//   - GPUMem's own work reads both (GpuWrittenHere, ForEachGpuWrittenHere): a write ahead of the
//     back is before it in the stream;
//   - the promotion sets the tracker's bit again: a served WriteMemory at the back's position,
//     before the write in stream order, may have dropped it since the fill, and THREAD 0's order
//     has the bit set after such a write;
//   - a fill whose packet never went into ring B (it threw, or made none) takes its marks back,
//     as does a packet the back drops unplayed (must be 0): the back fills that draw itself.
// The level rules that stay test both sets (vk_rasterizer.cpp, StageFillDraw; the words' test).
// Must be 0, logged with the "Memory stage marks ahead" line every 10 s: the AheadCheck list.

namespace MarksAheadDetail {
/// A stats counter: a plain load and store (one writer at a time: the lock, or one thread).
inline void AheadBump(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}
inline u64 AheadNowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}
} // namespace MarksAheadDetail
using MarksAheadDetail::AheadBump;
using MarksAheadDetail::AheadNowNs;

template <class P>
bool BufferCache<P>::MarkWrittenAhead(BufferId buffer_id, DAddr device_addr, u32 size,
                                      bool needs_sync, bool capped) {
    using Link = VideoCommon::MemoryStageLink;
    if (Link::ServingBack()) [[unlikely]] {
        // At the back's position: today's code (none does today).
        NoteAheadCheck(AheadCheck::ServedMark, "MarkWrittenBuffer");
        return false;
    }
    if (!Link::JoinNeedsTest()) {
        return false; // the back (a run), another thread, or the window closed: today's code
    }
    if (ahead_fill_index == 0) [[unlikely]] {
        NoteAheadCheck(AheadCheck::OutsideFill, "MarkWrittenBuffer");
        return false;
    }
    PromoteAhead();
    // The steps that run now (the spec's a, b, e, f, h), in today's order.
    if (VideoCommon::GpuSplit::RecordingDispatch()) [[unlikely]] {
        VideoCommon::GpuSplit::Get().RecordDispatchWrite(device_addr, size);
    }
    if (VideoCommon::Nx2Profile::Fine()) [[unlikely]] {
        ++nx2_written_marks;
        nx2_written_mark_bytes += size;
    }
    if (ReadRunStats::g_enabled) [[unlikely]] {
        ReadRunStats::NoteWrite(device_addr, size, ReadWriteStamp());
    }
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    memory_tracker.MarkRegionAsGpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    const bool ampere = Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
    if (ampere) {
        Tegra::Nx2MarkGpuWritten(device_addr, size);
    }
    // The rest at the promotion (c, d, g, i, j).
    if (ahead_writes.empty()) {
        ahead_oldest.store(ahead_fill_index, std::memory_order_release);
    }
    ahead_writes.push_back(AheadWrite{
        .index = ahead_fill_index,
        .addr = device_addr,
        .size = size,
        .capped = capped,
        .ampere = ampere,
        .footprint = mark_footprint,
    });
    ahead_ranges.Add(device_addr, size);
    AheadBump(ahead_stats.pushed);
    const u64 length = ahead_writes.size();
    AheadBump(ahead_stats.length_sum, length);
    AheadBump(ahead_stats.length_samples);
    if (length > ahead_stats.length_max.load(std::memory_order_relaxed)) {
        ahead_stats.length_max.store(length, std::memory_order_relaxed);
    }
    return true;
}

template <class P>
void BufferCache<P>::PromoteAheadSlow(bool by_back) {
    if (!mutex.IsHeldByCurrentThread()) [[unlikely]] {
        // Every reader of the real sets holds the lock; one that does not cannot promote.
        NoteAheadCheck(AheadCheck::Stale, "a reader of the real sets without the cache's lock");
        return;
    }
    const u64 discarded = ahead_discarded.load(std::memory_order_acquire);
    const u64 played = ahead_played.load(std::memory_order_acquire);
    u64 promoted = 0;
    while (!ahead_writes.empty() && ahead_writes.front().index <= played) {
        const AheadWrite write = ahead_writes.front();
        ahead_writes.pop_front();
        ahead_ranges.Subtract(write.addr, write.size);
        if (write.index == discarded) [[unlikely]] {
            AheadBump(ahead_stats.retracted); // its packet was dropped unplayed: not written
            continue;
        }
        if (write.index < ahead_last_promoted) [[unlikely]] {
            NoteAheadCheck(AheadCheck::Order, "PromoteAhead");
        }
        ahead_last_promoted = write.index;
        // MarkWrittenBuffer's deferred steps, in its order. The downloads already taken hold
        // older bytes (c), and so do the read downloads (d).
        for (auto& ranges : pending_download_ranges) {
            ranges.Subtract(write.addr, write.size);
        }
        OutdateReadDownloads(write.addr, write.size);
        // e again (the block comment above): THREAD 0 has the bit set after anything before the
        // write in stream order.
        memory_tracker.MarkRegionAsGpuModified(write.addr, write.size);
        if (write.footprint.id != 0) [[unlikely]] {
            // FRITCHII_STORE_FOOTPRINT: who made these bytes GPU-modified (the shadow first)
            footprint_shadow.AddClaimed(write.footprint.id, write.footprint.base, write.addr,
                                        write.size);
            gpu_modified_ranges.Add(write.addr, write.size, true);
        } else {
            gpu_modified_ranges.Add(write.addr, write.size);
        }
        // h ran at the fill, before this arms the read traps.
        QueueFenceDownload(write.addr, write.size);
        if (write.ampere) {
            // The buffer there now: its id is used only while list copies are collected.
            const BufferId buffer_id = page_table[write.addr >> CACHING_PAGEBITS];
            Nx2NoteListWrite(buffer_id, write.addr, write.size, write.capped);
        }
        ++promoted;
    }
    ahead_oldest.store(ahead_writes.empty() ? ~u64{0} : ahead_writes.front().index,
                       std::memory_order_release);
    AheadBump(by_back ? ahead_stats.promoted_back : ahead_stats.promoted_other, promoted);
}

template <class P>
void BufferCache<P>::RetractAhead(u64 index) {
    for (auto it = ahead_writes.begin(); it != ahead_writes.end();) {
        if (it->index != index) {
            ++it;
            continue;
        }
        ahead_ranges.Subtract(it->addr, it->size);
        it = ahead_writes.erase(it);
        AheadBump(ahead_stats.retracted);
    }
    ahead_oldest.store(ahead_writes.empty() ? ~u64{0} : ahead_writes.front().index,
                       std::memory_order_release);
}

template <class P>
bool BufferCache<P>::AheadIntersects(DAddr addr, u64 size) const {
    if (ahead_writes.empty() || size == 0) {
        return false;
    }
    bool hit = false;
    ahead_ranges.ForEachInRange(addr, size, [&hit](DAddr lower, DAddr upper, s32) {
        hit = hit || lower < upper;
    });
    return hit;
}

template <class P>
bool BufferCache<P>::AheadIntersectsPages(DAddr addr, u64 size) const {
    if (ahead_writes.empty() || size == 0) {
        return false;
    }
    const DAddr begin = Common::AlignDown(addr, DEVICE_PAGESIZE);
    const DAddr end = Common::AlignUp(addr + size, DEVICE_PAGESIZE);
    return AheadIntersects(begin, end - begin);
}

template <class P>
template <typename Func>
void BufferCache<P>::ForEachGpuWrittenHere(DAddr addr, u64 size, Func&& func) {
    if (ahead_writes.empty() || !VideoCommon::MemoryStageLink::JoinNeedsTest()) [[likely]] {
        gpu_modified_ranges.ForEachInRange(addr, size, func);
        return;
    }
    // Both sets, merged in address order (callers walk the gaps between the ranges).
    boost::container::small_vector<std::pair<DAddr, DAddr>, 16> spans;
    gpu_modified_ranges.ForEachInRange(addr, size, [&spans](DAddr lower, DAddr upper) {
        spans.emplace_back(lower, upper);
    });
    ahead_ranges.ForEachInRange(addr, size, [&spans](DAddr lower, DAddr upper, s32) {
        if (lower < upper) {
            spans.emplace_back(lower, upper);
        }
    });
    if (spans.empty()) {
        return;
    }
    std::sort(spans.begin(), spans.end());
    std::pair<DAddr, DAddr> current = spans.front();
    for (size_t i = 1; i < spans.size(); ++i) {
        if (spans[i].first <= current.second) {
            current.second = (std::max)(current.second, spans[i].second);
            continue;
        }
        func(current.first, current.second);
        current = spans[i];
    }
    func(current.first, current.second);
}

template <class P>
void BufferCache<P>::BeginAheadFill(u64 index) {
    // GPUMem at a fill's start: one compare when nothing is due.
    PromoteAhead();
    ahead_fill_index = index;
    ahead_fill_pushed = false;
}

template <class P>
bool BufferCache<P>::AheadFillPushed(u64 index) {
    if (ahead_fill_index == 0) {
        return false;
    }
    ahead_fill_pushed = true;
    if (index != ahead_fill_index) [[unlikely]] {
        // The fill's marks are the newest: they go with the packet that carries the fill.
        NoteAheadCheck(AheadCheck::PushMismatch, "PushRingDraw");
        for (auto it = ahead_writes.rbegin();
             it != ahead_writes.rend() && it->index == ahead_fill_index; ++it) {
            it->index = index;
        }
        ahead_fill_index = index;
        ahead_oldest.store(ahead_writes.empty() ? ~u64{0} : ahead_writes.front().index,
                           std::memory_order_release);
    }
    return !ahead_writes.empty() && ahead_writes.back().index == index;
}

template <class P>
void BufferCache<P>::EndAheadFill() {
    if (ahead_fill_index == 0) {
        return;
    }
    if (!ahead_fill_pushed && !ahead_writes.empty() &&
        ahead_writes.back().index == ahead_fill_index) [[unlikely]] {
        // No packet carries them: the back fills this draw itself and marks it then.
        while (!ahead_writes.empty() && ahead_writes.back().index == ahead_fill_index) {
            ahead_ranges.Subtract(ahead_writes.back().addr, ahead_writes.back().size);
            ahead_writes.pop_back();
            AheadBump(ahead_stats.retracted);
        }
        ahead_oldest.store(ahead_writes.empty() ? ~u64{0} : ahead_writes.front().index,
                           std::memory_order_release);
    }
    ahead_fill_index = 0;
    ahead_fill_pushed = false;
}

template <class P>
bool BufferCache<P>::TryPromoteAhead(void* self_) {
    BufferCache& self = *static_cast<BufferCache*>(self_);
    if (!self.AheadDue()) {
        return true; // promoted, by the back or by the lock's holder
    }
    if (!self.mutex.try_lock()) {
        return false;
    }
    std::scoped_lock lock{std::adopt_lock, self.mutex};
    self.PromoteAhead(true);
    return true;
}

template <class P>
void BufferCache<P>::NoteRingPlayed(u64 index, bool marked) {
    ahead_played.store(index, std::memory_order_release);
    if (!marked || TryPromoteAhead(this)) [[likely]] {
        return;
    }
    // GPUMem (its fill) or another thread holds the lock.
    const u64 start = AheadNowNs();
    VideoCommon::MemoryStageLink::Get().BackPromoteMarks(&TryPromoteAhead, this);
    AheadBump(ahead_stats.back_waits);
    AheadBump(ahead_stats.back_wait_ns, AheadNowNs() - start);
}

template <class P>
void BufferCache<P>::NoteRingDiscarded(u64 index, bool marked) {
    if (marked) {
        ahead_discarded.store(index, std::memory_order_release);
    }
    NoteRingPlayed(index, marked);
}

template <class P>
void BufferCache<P>::CheckAheadAt(AheadPoint where) {
    if (where == AheadPoint::RunStart) {
        // Marks of packets the run takes later (its first entry's draws GPUMem filled) stay;
        // the back promoted every packet it played.
        if (AheadDue()) [[unlikely]] {
            NoteAheadCheck(AheadCheck::RunDue, "a run's first entry");
            std::scoped_lock lock{mutex};
            PromoteAhead(true);
        }
        return;
    }
    if (ahead_oldest.load(std::memory_order_acquire) == ~u64{0}) [[likely]] {
        return; // empty
    }
    std::scoped_lock lock{mutex};
    PromoteAhead(true);
    if (ahead_writes.empty()) {
        return;
    }
    NoteAheadCheck(AheadCheck::NotEmpty,
                   where == AheadPoint::RingStart ? "the ring's start" : "the ring's end");
    // No packet will play them: dropped (their early steps only made the cache more careful).
    while (!ahead_writes.empty()) {
        ahead_ranges.Subtract(ahead_writes.back().addr, ahead_writes.back().size);
        ahead_writes.pop_back();
        AheadBump(ahead_stats.retracted);
    }
    ahead_oldest.store(~u64{0}, std::memory_order_release);
}

template <class P>
void BufferCache<P>::NoteAheadLevel(AheadLevel kind) noexcept {
    AheadBump(ahead_stats.levels[static_cast<size_t>(kind)]);
}

template <class P>
void BufferCache<P>::NoteAheadLevelCallSlow(DAddr addr, u64 size, const char* where) {
    if (VideoCommon::MemoryStageLink::JoinNeedsTest() && AheadIntersectsPages(addr, size)) {
        NoteAheadCheck(AheadCheck::Claim, where);
    }
}

template <class P>
void BufferCache<P>::NoteAheadCheck(AheadCheck check, const char* where) const noexcept {
    static constexpr std::array<const char*, static_cast<size_t>(AheadCheck::Count)> names{
        "a level function on GPUMem's own work over a page with a mark ahead",
        "a promotion out of packet order",
        "the list not empty",
        "a mark due at a run's first entry",
        "a due mark a reader could not promote",
        "a GPU-written mark in a call GPUMem serves for the back",
        "a GPU-written mark on GPUMem's own work outside a draw fill",
        "the packet pushed is not the one the fill's marks were keyed with",
    };
    const size_t index = static_cast<size_t>(check);
    ahead_stats.checks[index].fetch_add(1, std::memory_order_relaxed);
    static std::array<std::atomic<u32>, static_cast<size_t>(AheadCheck::Count)> logged{};
    if (logged[index].fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_ERROR(HW_GPU, "Memory stage marks ahead (must be 0): {} ({})", names[index], where);
    }
}

template <class P>
void BufferCache<P>::SetMarksAhead(bool on, bool wanted) {
    if (on != marks_ahead.load(std::memory_order_relaxed) || wanted != marks_ahead_wanted) {
        LOG_INFO(HW_GPU, "Memory stage marks ahead (FRITCHII_MEMSTAGE_MARKS_AHEAD={}): {}",
                 wanted ? 1 : 0,
                 on       ? "GPUMem's writing fills mark ahead of the back, no level wait"
                 : wanted ? "off while the window is closed (FRITCHII_MEMSTAGE_THREAD=2 and "
                            "FRITCHII_MEMSTAGE_AHEAD > 0)"
                          : "off");
        marks_ahead_wanted = wanted;
        marks_ahead.store(on, std::memory_order_relaxed);
    }
    LogMarksAhead();
}

template <class P>
void BufferCache<P>::LogMarksAhead() {
    const u64 now = AheadNowNs();
    if (ahead_log_ns == 0) {
        ahead_log_ns = now;
        return;
    }
    if (now - ahead_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - ahead_log_ns) / 1e9;
    ahead_log_ns = now;
    const auto take = [&](size_t slot, const std::atomic<u64>& counter) {
        const u64 value = counter.load(std::memory_order_relaxed);
        const u64 delta = value - ahead_reported[slot];
        ahead_reported[slot] = value;
        return delta;
    };
    AheadStats& s = ahead_stats;
    const u64 pushed = take(0, s.pushed);
    const u64 by_back = take(1, s.promoted_back);
    const u64 by_other = take(2, s.promoted_other);
    const u64 retracted = take(3, s.retracted);
    const u64 length_sum = take(4, s.length_sum);
    const u64 samples = take(5, s.length_samples);
    const u64 waits = take(6, s.back_waits);
    const u64 wait_ns = take(7, s.back_wait_ns);
    const u64 claims = s.levels[0].exchange(0, std::memory_order_relaxed);
    const u64 refused = s.levels[1].exchange(0, std::memory_order_relaxed);
    const u64 bound = s.levels[2].exchange(0, std::memory_order_relaxed);
    const u64 length_max = s.length_max.exchange(0, std::memory_order_relaxed);
    std::string checks;
    u64 any_check = 0;
    for (size_t i = 0; i < s.checks.size(); ++i) {
        const u64 value = s.checks[i].load(std::memory_order_relaxed);
        any_check += value - ahead_checks_reported[i];
        checks += fmt::format("{}{}", i == 0 ? "" : " ", value);
        ahead_checks_reported[i] = value;
    }
    if (!marks_ahead.load(std::memory_order_relaxed) && pushed == 0 && by_back == 0 &&
        by_other == 0 && any_check == 0) {
        return;
    }
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    LOG_INFO(HW_GPU,
             "Memory stage marks ahead {:.0f} s: put on the list {:.0f}/s, promoted by the back at "
             "the play {:.0f}/s, by others {:.0f}/s, taken back {:.1f}/s; list length avg {:.1f} "
             "max {}; the back's waits for the lock at a promotion {:.1f}/s ({:.2f} ms/s); level "
             "waits: claims {:.1f}/s (refused after the wait {:.1f}/s), at the bound {:.1f}/s; "
             "must be 0 (claim, order, not empty, run due, stale, served mark, outside a fill, push "
             "mismatch; totals): {}",
             seconds, rate(pushed), rate(by_back), rate(by_other), rate(retracted),
             samples != 0 ? static_cast<double>(length_sum) / static_cast<double>(samples) : 0.0,
             length_max, rate(waits), static_cast<double>(wait_ns) / 1e6 / seconds, rate(claims),
             rate(refused), rate(bound), checks);
}

template <class P>
void BufferCache<P>::Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size,
                                      bool to_page_end) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page.
            const u64 word =
                Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                (page % 64);
            if (word == 0) {
                page = (page / 64 + 1) * 64;
                continue;
            }
            if (word & 1) {
                const DAddr page_addr = page << page_bits;
                {
                    std::scoped_lock lock{nx2_list_mutex};
                    Nx2ListPage& entry = nx2_list_pages[page_addr];
                    ++entry.writes;
                    constexpr DAddr page_size = DAddr{1} << page_bits;
                    const DAddr write_end = to_page_end && page == last
                                                ? page_addr + page_size
                                                : (std::min)(device_addr + size,
                                                             page_addr + page_size);
                    Nx2MarkListWords(entry, (std::max)(device_addr, page_addr) - page_addr,
                                     write_end - page_addr, true);
                    if (!nx2_collect_list_copies && !entry.draw_written) {
                        entry.draw_written = true;
                        LOG_INFO(HW_GPU, "Command list page {:#x} is written by draws too: its "
                                         "lists wait for the work before the last host sync",
                                 page_addr);
                    }
                }
                if (nx2_collect_list_copies) {
                    nx2_list_copy_requests.emplace_back(buffer_id, page_addr);
                }
                if (Tegra::Nx2CmdPageDiag::g_enabled) {
                    Tegra::Nx2CmdPageDiag::NoteWrite(page_addr, device_addr, size,
                                                     !nx2_collect_list_copies);
                }
            }
            ++page;
        }
    }
}

template <class P>
bool BufferCache<P>::Nx2CopyListPages(u64 tick, u64 writer) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (nx2_list_copy_requests.empty()) {
            return false;
        }
        std::ranges::sort(nx2_list_copy_requests, {}, &std::pair<BufferId, DAddr>::second);
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        bool recorded = false;
        for (size_t i = 0; i < nx2_list_copy_requests.size(); ++i) {
            const auto [buffer_id, page] = nx2_list_copy_requests[i];
            if (i + 1 < nx2_list_copy_requests.size() &&
                nx2_list_copy_requests[i + 1].second == page) {
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            const DAddr begin = (std::max<DAddr>)(page, buffer.CpuAddr());
            const DAddr end =
                (std::min<DAddr>)(page + page_size, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin >= end) {
                continue;
            }
            Nx2Staging staging = runtime.DownloadStagingBuffer(page_size, true);
            const BufferCopy copy{
                .src_offset = buffer.Offset(begin),
                .dst_offset = staging.offset + (begin - page),
                .size = end - begin,
            };
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(staging.buffer, buffer, std::span{&copy, 1}, true);
            recorded = true;
            std::scoped_lock lock{nx2_list_mutex};
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.copy) {
                nx2_retired_list_copies.emplace_back(entry.copy_tick, std::move(*entry.copy));
            }
            entry.copy = staging;
            entry.copy_writes = entry.writes;
            entry.copy_tick = tick;
            entry.copy_writer = writer;
            entry.copy_begin = static_cast<u32>(begin - page);
            entry.copy_end = static_cast<u32>(end - page);
        }
        nx2_list_copy_requests.clear();
        std::erase_if(nx2_retired_list_copies, [this](auto& retired) {
            if (!runtime.IsFree(retired.first)) {
                return false;
            }
            runtime.FreeDeferredStagingBuffer(retired.second);
            return true;
        });
        return recorded;
    }
}

template <class P>
auto BufferCache<P>::Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync)
    -> Nx2ListRead {
    if constexpr (IS_OPENGL) {
        return Nx2ListRead::Unknown;
    } else {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (offset + out.size() > page_size) {
            return Nx2ListRead::Unknown;
        }
        const auto current = [&](const Nx2ListPage& entry) {
            return entry.copy && !entry.draw_written && entry.copy_writes == entry.writes &&
                   offset >= entry.copy_begin && offset + out.size() <= entry.copy_end;
        };
        // EXP40 & 1: guest memory serves the words no GPU work wrote since the page was flushed.
        // Not for EDEN_NX2_LIST_VERIFY's read as before.
        const bool words = (VideoCore::Nx2Debug::Experiment(40) & 1) != 0 &&
                           !Tegra::Nx2ListVerify::t_reference;
        u64 tick{};
        u64 writes{};
        u64 writer{};
        {
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page);
            if (it == nx2_list_pages.end()) {
                return Nx2ListRead::Unknown;
            }
            const Nx2ListPage& entry = it->second;
            if (entry.draw_written) {
                // EDEN_NX2_CMDPAGE_DRAIN=1: drain and flush for every list on such pages.
                static const bool always_drain = std::getenv("EDEN_NX2_CMDPAGE_DRAIN") != nullptr;
                if (always_drain) {
                    return Nx2ListRead::Unknown;
                }
                if (!past_sync) {
                    return Nx2ListRead::NeedsSync;
                }
                return entry.synced_writes == entry.writes ? Nx2ListRead::Synced
                                                           : Nx2ListRead::Unknown;
            }
            // Pages the CPU writes the lists of (GPU-written once, long before) stay synced, and
            // with EXP40 & 1 a list on none of the words GPU work wrote since the page was flushed
            // is in guest memory too: it waits for no copy.
            if (entry.synced_writes == entry.writes ||
                (words && entry.flushed && !Nx2HasListWords(entry, offset, offset + out.size()))) {
                return Nx2ListRead::Synced;
            }
            if (!current(entry)) {
                return Nx2ListRead::Unknown;
            }
            // The lists after the first of a batch find the copy done: read it at once.
            if (runtime.IsFree(entry.copy_tick)) {
                Nx2ReadListWords(entry, device_addr, offset, out);
                return Nx2ListRead::Copied;
            }
            tick = entry.copy_tick;
            writes = entry.copy_writes;
            writer = entry.copy_writer;
        }
        // Submitted with the work that wrote the page (Nx2CopyListPages's caller flushes).
        runtime.WaitSubmittedTick(tick, writer);
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || !current(it->second) || it->second.copy_tick != tick ||
            it->second.copy_writes != writes) {
            return Nx2ListRead::Unknown;
        }
        Nx2ReadListWords(it->second, device_addr, offset, out);
        return Nx2ListRead::Copied;
    }
}

template <class P>
void BufferCache<P>::Nx2ReadListWords(const Nx2ListPage& entry, DAddr device_addr, u64 offset,
                                      std::span<u8> out) {
    if constexpr (!IS_OPENGL) {
        // The copy holds the words GPU work wrote. Guest memory holds the others once the page
        // was flushed: the host wrote them, and the buffer the copy came from need not have had
        // those writes uploaded. EDEN_NX2_LIST_VERIFY's read as before takes the whole copy.
        const u8* const copy = entry.copy->mapped_span.data();
        const u64 end = offset + out.size();
        bool all_gpu = true;
        for (u64 word = offset / 4; word < (end + 3) / 4 && all_gpu; ++word) {
            all_gpu = entry.gpu_words.test(static_cast<size_t>(word));
        }
        if (all_gpu || !entry.flushed || Tegra::Nx2ListVerify::t_reference ||
            (VideoCore::Nx2Debug::Experiment(40) & 1) == 0) {
            std::memcpy(out.data(), copy + offset, out.size());
            return;
        }
        VideoCommon::OneFollower::BeforeEngineRead(device_addr, out.size());
        device_memory.ReadBlockUnsafe(device_addr, out.data(), out.size());
        for (u64 byte = offset; byte < end;) {
            const u64 next = (std::min)((byte / 4 + 1) * 4, end);
            if (entry.gpu_words.test(static_cast<size_t>(byte / 4))) {
                std::memcpy(out.data() + (byte - offset), copy + byte, next - byte);
            }
            byte = next;
        }
    }
}

template <class P>
void BufferCache<P>::Nx2MarkListWords(Nx2ListPage& entry, u64 begin, u64 end, bool set) {
    if (begin >= end) {
        return;
    }
    // Words GPU work may have written in part count whole; the host clears only whole words.
    const u64 first = set ? begin / 4 : (begin + 3) / 4;
    const u64 last = (std::min<u64>)(set ? (end + 3) / 4 : end / 4, Nx2ListPageWords);
    for (u64 word = first; word < last; ++word) {
        entry.gpu_words.set(static_cast<size_t>(word), set);
    }
}

template <class P>
bool BufferCache<P>::Nx2HasListWords(const Nx2ListPage& entry, u64 begin, u64 end) {
    const u64 last = (std::min<u64>)((end + 3) / 4, Nx2ListPageWords);
    for (u64 word = begin / 4; word < last; ++word) {
        if (entry.gpu_words.test(static_cast<size_t>(word))) {
            return true;
        }
    }
    return false;
}

template <class P>
void BufferCache<P>::Nx2NoteListSynced(DAddr page) {
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        Nx2ListPage& entry = nx2_list_pages[page];
        entry.synced_writes = entry.writes;
        entry.gpu_words.reset();
        entry.flushed = true;
    }
}

template <class P>
void BufferCache<P>::Nx2FlushListPages(std::span<const DAddr> pages) {
    const Common::WriteRing::SiteScope write_site{"list page flush (Nx2FlushListPages)", true};
    constexpr u64 page_size = u64{1} << Tegra::Nx2WrittenPageBits;
    std::vector<std::pair<DAddr, u64>> ranges;
    for (const DAddr page : pages) {
        ranges.emplace_back(page, page_size);
    }
    std::vector<std::pair<DAddr, u64>> drawn;
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, entry] : nx2_list_pages) {
            if (entry.draw_written && entry.synced_writes != entry.writes) {
                ranges.emplace_back(page, page_size);
                drawn.emplace_back(page, entry.writes);
            }
        }
    }
    DownloadMemoryRanges(ranges);
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, writes] : drawn) {
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.writes == writes) {
                entry.synced_writes = writes;
                entry.gpu_words.reset();
                entry.flushed = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListHostWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last; ++page) {
            if (!((Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                   (page % 64)) &
                  1)) {
                continue;
            }
            std::scoped_lock lock{nx2_list_mutex};
            const DAddr page_addr = page << page_bits;
            const auto it = nx2_list_pages.find(page_addr);
            if (it == nx2_list_pages.end()) {
                continue;
            }
            Nx2ListPage& entry = it->second;
            const bool synced = entry.synced_writes == entry.writes;
            ++entry.writes;
            if (synced) {
                entry.synced_writes = entry.writes;
            }
            Nx2MarkListWords(entry, (std::max)(device_addr, page_addr) - page_addr,
                             (std::min)(device_addr + size, page_addr + (DAddr{1} << page_bits)) -
                                 page_addr,
                             false);
        }
    }
}

template <class P>
BufferId BufferCache<P>::FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible) {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || page >= page_table.size()) {
        return NULL_BUFFER_ID;
    }
    const bool should_sync = Settings::IsGPUFenceBehaviorAccurate();
    FindMemo& memo = find_memos[page & (find_memos.size() - 1)];
    if (!should_sync && memo.page == page && memo.deletions == buffer_deletions &&
        device_addr >= memo.begin && device_addr + size <= memo.end &&
        (!sparse_compatible || memo.sparse_compatible)) {
        return memo.buffer_id;
    }
    const BufferId buffer_id = page_table[page];
    if (buffer_id) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsInBounds(device_addr, size)) {
            if (should_sync) {
                SynchronizeBufferWrites(buffer);
            }
            bool usable = true;
            bool is_sparse = false;
            if constexpr (requires { buffer.IsSparseCompatible(); }) {
                is_sparse = buffer.IsSparseCompatible();
                if (sparse_compatible && !is_sparse) {
                    usable = false;
                }
            }
            if (usable) {
                if (!should_sync) {
                    memo = FindMemo{
                        .page = page,
                        .deletions = buffer_deletions,
                        .begin = buffer.CpuAddr(),
                        .end = buffer.CpuAddr() + buffer.SizeBytes(),
                        .buffer_id = buffer_id,
                        .sparse_compatible = is_sparse,
                    };
                }
                return buffer_id;
            }
        }
    }
    return CreateBuffer(device_addr, size, sparse_compatible);
}

template <class P>
void BufferCache<P>::SynchronizeBufferWrites(Buffer& buffer) {
    if constexpr (!IS_OPENGL) {
        const u64 buffer_tick = buffer.getWriteTick();
        if (!runtime.IsFree(buffer_tick)) {
            runtime.Wait(buffer_tick);
        }
    }
}

template <class P>
typename BufferCache<P>::OverlapResult BufferCache<P>::ResolveOverlaps(DAddr device_addr,
                                                                       u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    DAddr begin = device_addr;
    DAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    auto expand_begin = [&](DAddr add_value) {
        static constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = Core::DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    auto expand_end = [&](DAddr add_value) {
        static constexpr DAddr max_page = 1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS];
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.IsPicked()) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.Pick();
        const DAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const DAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_end(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

template <class P>
bool BufferCache<P>::JoinCopyIsDead(DAddr device_addr, u64 size) {
    // Every page CPU-modified: SynchronizeBuffer uploads such pages whole from guest memory
    // (ForEachModifiedRange reports whole pages) before any bind, DMA copy or download reads
    // the new buffer there, so the bytes a join copy puts in would be overwritten unread. The
    // exceptions keep host data the upload leaves alone or read the host buffer later: a page
    // GPU-modified or Preflushable, GPU writes the upload skips (gpu_modified_ranges), and GPU
    // writes queued or in flight for download.
    if (!memory_tracker.IsRegionWhollyCpuModified(device_addr, size) ||
        gpu_modified_ranges.Intersects(device_addr, size) || AheadHere(device_addr, size) ||
        uncommitted_gpu_modified_ranges.Intersects(device_addr, size)) {
        return false;
    }
    const auto intersects = [&](const Common::RangeSet<DAddr>& ranges) {
        return ranges.Intersects(device_addr, size);
    };
    return std::ranges::none_of(committed_gpu_modified_ranges, intersects) &&
           std::ranges::none_of(pending_download_ranges, intersects);
}

template <class P>
bool BufferCache<P>::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                                 bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    boost::container::small_vector<BufferCopy, 10> copies;
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    copies.push_back(BufferCopy{
        .src_offset = 0,
        .dst_offset = dst_base_offset,
        .size = overlap.SizeBytes(),
    });
    new_buffer.MarkUsage(copies[0].dst_offset, copies[0].size);
    VideoCommon::Churn::Add(VideoCommon::Churn::BuffersJoined);
    // Memory stage step 7: a join in a fill ahead of the back. Over GPU-written data the back
    // first plays the packets ahead (a download it records later must not read the new buffer
    // before this copy); kept for the served calls' check either way.
    if (VideoCommon::MemoryStageLink::JoinNeedsTest() &&
        GpuWrittenHere(overlap.CpuAddr(), overlap.SizeBytes())) [[unlikely]] {
        VideoCommon::MemoryStageLink::JoinOverGpuWritten();
    }
    VideoCommon::MemoryStageLink::NoteBufferJoin(overlap.CpuAddr(), overlap.SizeBytes());
    // EDEN_NX2_JOIN_SKIP_CPU=1 (fable_answer_17 4): a drifted per-frame range the guest rewrote
    // whole since its last upload needs no copy, the first bind uploads all of it anyway
    // (JoinCopyIsDead). Read on first use: nx2_switches.txt is applied after static init.
    static const bool skip_cpu = [] {
        const char* value = std::getenv("EDEN_NX2_JOIN_SKIP_CPU");
        return value != nullptr && value[0] == '1';
    }();
    const bool copy = !skip_cpu || !JoinCopyIsDead(overlap.CpuAddr(), overlap.SizeBytes());
    if (copy) {
        VideoCommon::Churn::Add(VideoCommon::Churn::BufferJoinBytes, overlap.SizeBytes());
        runtime.CopyBuffer(new_buffer, overlap, copies, true);
    } else {
        VideoCommon::Churn::Add(VideoCommon::Churn::BufferJoinsSkipped);
        VideoCommon::Churn::Add(VideoCommon::Churn::BufferJoinSkippedBytes, overlap.SizeBytes());
    }
#ifdef YUZU_LEGACY
    if (immediately_free)
        runtime.Finish();
#endif
    DeleteBuffer(overlap_id, true);
    return copy;
}

template <class P>
BufferId BufferCache<P>::CreateBuffer(DAddr device_addr, u32 wanted_size,
                                      bool sparse_compatible) {
    DAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(runtime, overlap.begin, size, sparse_compatible);
    VideoCommon::Churn::Add(VideoCommon::Churn::BuffersCreated);
    auto& new_buffer = slot_buffers[new_buffer_id];
    const size_t size_bytes = new_buffer.SizeBytes();
    MemStageStats::BufferCreated(size_bytes);
    // EDEN_NX2_BUFFER_CHURN_STATS: why (Fable's design 8.3).
    if (VideoCommon::BufferChurnStats::Enabled()) [[unlikely]] {
        using VideoCommon::BufferChurnStats::Created;
        const Created cause = device_addr == 0         ? Created::Inline
                              : overlap.ids.empty()     ? Created::NewRange
                              : overlap.ids.size() == 1 ? Created::Grow
                                                        : Created::Join;
        VideoCommon::BufferChurnStats::BufferCreated(cause, size_bytes);
    }
    runtime.ClearBuffer(new_buffer, 0, size_bytes, 0);
    new_buffer.MarkUsage(0, size_bytes);
    bool copied = false;
    bool skipped = false;
    for (const BufferId overlap_id : overlap.ids) {
        const bool joined_copy = JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
        copied |= joined_copy;
        skipped |= !joined_copy;
    }
    if (skipped && !copied) {
        // The barrier a join copy ends with (CopyBuffer) also ordered the clear above before a
        // reordered upload in a later submission; keep it when every copy was skipped.
        runtime.PostCopyBarrier();
    }
    Register(new_buffer_id);
    TouchBuffer(new_buffer, new_buffer_id);
    return new_buffer_id;
}

template <class P>
void BufferCache<P>::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

template <class P>
void BufferCache<P>::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <class P>
template <bool insert>
void BufferCache<P>::ChangeRegister(BufferId buffer_id) {
    ++binding_epoch;
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    if (insert) {
        buffer_bytes += Common::AlignUp(size, 1024);
        total_used_memory += Common::AlignUp(size, 1024);
        buffer.setLRUID(lru_cache.Insert(buffer_id, frame_tick));
    } else {
        buffer_bytes -= Common::AlignUp(size, 1024);
        total_used_memory -= Common::AlignUp(size, 1024);
        lru_cache.Free(buffer.getLRUID());
    }
    const DAddr device_addr_begin = buffer.CpuAddr();
    const DAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page] = buffer_id;
        } else {
            page_table[page] = BufferId{};
        }
    }
}

template <class P>
void BufferCache<P>::TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept {
    if (buffer_id == NULL_BUFFER_ID) {
        return;
    }
    // EDEN_NX2_EXP18 & 4: the buffer's own field instead of its LRU node (a cache miss) for every
    // bind of the frame; a second touch in the same frame changes nothing.
    if ((VideoCore::Nx2Debug::Experiment(18) & 4) != 0) {
        if (buffer.lru_touched_tick == frame_tick) {
            return;
        }
        buffer.lru_touched_tick = frame_tick;
    }
    lru_cache.Touch(buffer.getLRUID(), frame_tick);
}

template <class P>
bool BufferCache<P>::SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size) {
    MemReadLog::Range(MemReadLog::Kind::Bound, device_addr, size);
    // An upload over a clean word of the tracker (word_manager.h) finds and changes nothing:
    // walk only the dirty ones. Verify walks every word and checks the clean words.
    const bool skip_clean = Nx2Perf::On(Nx2Perf::Opt::CleanWords) && !Nx2Perf::Verify();
    if (Nx2Perf::On(Nx2Perf::Opt::CleanWords) && Nx2Perf::Verify()) {
        static std::atomic<u32> logged{};
        if (!memory_tracker.CleanWordsHold(device_addr, size) &&
            logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_ERROR(HW_GPU, "Nx2Perf draw-binds: clean tracker words wrong near {:#x}+{:#x}",
                      device_addr, size);
        }
    }
    // EDEN_NX2_EXP59 (compute storage binds only, compute_dirty_walk): & 4 walks the dirty words
    // like CleanWords above; & 8 keeps the full walk and checks what & 4 would have left out.
    const bool compute_verify = (compute_dirty_walk & 8) != 0;
    const bool compute_fast = !compute_verify && (compute_dirty_walk & 4) != 0;
    if (compute_verify) [[unlikely]] {
        const bool walks = VerifyComputeDirtyWalk(device_addr, size);
        if (count_compute_sync) {
            Nx2Profile::Add(Nx2Profile::Counter::ComputeSyncDirtyWalks, 1);
            Nx2Profile::Add(Nx2Profile::Counter::ComputeSyncDirtySkips, walks ? 0 : 1);
        }
    }
    // EDEN_NX2_STORAGE_SYNC_SCAN (storage binds only, storage_sync_scan): 1 walks the scanned
    // windows; 2 keeps the full walk and counts what 1 would have left out.
    if (storage_sync_scan == 2) [[unlikely]] {
        VerifyStorageSyncScan(device_addr, size);
    }
    const auto for_each_upload_range = [&](auto&& func) {
        if (compute_fast) {
            const bool walked = memory_tracker.ForEachDirtyUploadRange(device_addr, size, func);
            if (count_compute_sync) {
                Nx2Profile::Add(Nx2Profile::Counter::ComputeSyncDirtyWalks, 1);
                Nx2Profile::Add(Nx2Profile::Counter::ComputeSyncDirtySkips, walked ? 0 : 1);
            }
        } else if (storage_sync_scan == 1) {
            memory_tracker.ForEachScannedUploadRange(device_addr, size, func);
        } else if (skip_clean && !compute_verify) {
            memory_tracker.ForEachDirtyUploadRange(device_addr, size, func);
        } else {
            memory_tracker.ForEachUploadRange(device_addr, size, func);
        }
    };
#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        upload_copies.clear();
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        const DAddr buffer_start = buffer.cpu_addr_cached;
        const auto add_upload = [&](DAddr start, DAddr end) {
            if (start == end) return;
            const u64 range_size = end - start;
            upload_copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = start - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        };
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            DAddr upload_start = device_addr_out;
            ForEachGpuWrittenHere(device_addr_out, range_size, [&](DAddr gpu_start, DAddr gpu_end) {
                add_upload(upload_start, gpu_start);
                upload_start = gpu_end;
            });
            add_upload(upload_start, device_addr_out + range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        if (count_compute_sync) [[unlikely]] {
            Nx2Profile::Add(Nx2Profile::Counter::ComputeSyncUploads, 1);
            Nx2Profile::Add(Nx2Profile::Counter::ComputeSyncBytes, total_size_bytes);
        }
        const std::span<BufferCopy> copies_span(upload_copies.data(), upload_copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        // Mapped uploads carry their own barriers (or run in the upload command buffer, which
        // ends with one); another barrier here only ended the render pass (~50 times a frame in
        // Pokopia).
        if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
            any_buffer_uploaded = true;
        }
        return false;
    } else {
        boost::container::small_vector<BufferCopy, 4> copies;
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        DAddr buffer_start = buffer.CpuAddr();
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = device_addr_out - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        if (count_compute_sync) [[unlikely]] {
            Nx2Profile::Add(Nx2Profile::Counter::ComputeSyncUploads, 1);
            Nx2Profile::Add(Nx2Profile::Counter::ComputeSyncBytes, total_size_bytes);
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        return false;
    }
}

template <class P>
bool BufferCache<P>::VerifyComputeDirtyWalk(DAddr device_addr, u32 size) {
    // The full walk runs after this and is what gets used. A CPU page outside the dirty-word
    // window is an upload the dirty-word walk would have missed; an Untracked one is a page it
    // would have left unprotected. Both must stay 0.
    const auto check = memory_tracker.CheckDirtyWalk(device_addr, size);
    auto& verify = dirty_walk_verify;
    ++verify.checked;
    ++verify.total_checked;
    if (check.cpu_pages != 0 || check.untracked_pages != 0) [[unlikely]] {
        ++verify.mismatches;
        ++verify.total_mismatches;
        verify.cpu_pages += check.cpu_pages;
        verify.untracked_pages += check.untracked_pages;
        if (verify.logged < 16) {
            ++verify.logged;
            LOG_ERROR(HW_GPU,
                      "Nx2 EXP59 verify: the dirty-word walk would differ at {:#x}+{:#x}: {} CPU "
                      "pages it would not upload, {} Untracked pages it would not re-track",
                      device_addr, size, check.cpu_pages, check.untracked_pages);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (verify.last_report == std::chrono::steady_clock::time_point{}) {
        verify.last_report = now;
    } else if (now - verify.last_report >= std::chrono::seconds{10}) {
        if (verify.mismatches != 0) {
            LOG_ERROR(HW_GPU,
                      "Nx2 EXP59 verify, last 10 s: {} of {} compute synchronizes differ ({} CPU "
                      "pages, {} Untracked pages); since start {} of {}",
                      verify.mismatches, verify.checked, verify.cpu_pages, verify.untracked_pages,
                      verify.total_mismatches, verify.total_checked);
        } else {
            LOG_INFO(HW_GPU,
                     "Nx2 EXP59 verify, last 10 s: 0 of {} compute synchronizes differ; since "
                     "start {} of {}",
                     verify.checked, verify.total_mismatches, verify.total_checked);
        }
        verify.checked = 0;
        verify.mismatches = 0;
        verify.cpu_pages = 0;
        verify.untracked_pages = 0;
        verify.last_report = now;
    }
    return check.walks;
}

template <class P>
void BufferCache<P>::VerifyStorageSyncScan(DAddr device_addr, u32 size) {
    // The full walk runs after this and is what gets used. A CPU page outside the scanned
    // windows is an upload the scanned walk would have missed; an unprotected one a page it
    // would have left untracked. Both must stay 0.
    const auto check = memory_tracker.CheckScannedWalk(device_addr, size);
    auto& verify = storage_scan_verify;
    ++verify.checked;
    ++verify.total_checked;
    if (check.cpu_pages != 0 || check.untracked_pages != 0) [[unlikely]] {
        ++verify.mismatches;
        ++verify.total_mismatches;
        verify.cpu_pages += check.cpu_pages;
        verify.untracked_pages += check.untracked_pages;
        if (verify.logged < 16) {
            ++verify.logged;
            LOG_ERROR(HW_GPU,
                      "Nx2 storage sync scan verify: the scanned walk would differ at {:#x}+{:#x}: "
                      "{} CPU pages it would not upload, {} unprotected pages it would not "
                      "re-track",
                      device_addr, size, check.cpu_pages, check.untracked_pages);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (verify.last_report == std::chrono::steady_clock::time_point{}) {
        verify.last_report = now;
    } else if (now - verify.last_report >= std::chrono::seconds{10}) {
        if (verify.mismatches != 0) {
            LOG_ERROR(HW_GPU,
                      "Nx2 storage sync scan verify, last 10 s: {} of {} storage synchronizes "
                      "differ ({} CPU pages, {} unprotected pages); since start {} of {}",
                      verify.mismatches, verify.checked, verify.cpu_pages, verify.untracked_pages,
                      verify.total_mismatches, verify.total_checked);
        } else {
            LOG_INFO(HW_GPU,
                     "Nx2 storage sync scan verify, last 10 s: 0 of {} storage synchronizes "
                     "differ; since start {} of {}",
                     verify.checked, verify.total_mismatches, verify.total_checked);
        }
        verify.checked = 0;
        verify.mismatches = 0;
        verify.cpu_pages = 0;
        verify.untracked_pages = 0;
        verify.last_report = now;
    }
}

template <class P>
bool BufferCache<P>::SynchronizeStorageMemo(Buffer& buffer, DAddr device_addr, u32 size,
                                            bool verify) {
    StorageSyncMemo& memo =
        storage_sync_memo[((device_addr ^ (u64{size} << 40)) * 0x9E3779B97F4A7C15ULL) >>
                          (64 - STORAGE_SYNC_MEMO_BITS)];
    u64 gen_before = 0;
    const bool tracked = memory_tracker.DirtyGenerationSum(device_addr, size, gen_before);
    if (tracked && memo.size == size && memo.device_addr == device_addr &&
        memo.gen_sum == gen_before) {
        Nx2Profile::AddFine(Nx2Profile::FineCounter::StorageMemoHits, 1);
        if (!verify) {
            // Left clean at this sum, and no page of its managers gained a CPU or Untracked bit
            // since: the walk would upload nothing and re-protect nothing.
            return true;
        }
        // EDEN_NX2_STORAGE_SYNC_MEMO=2: the range must hold no CPU or Untracked page. Then
        // synchronize as without the memo.
        auto& check = storage_memo_verify;
        ++check.checked;
        ++check.total_checked;
        if (memory_tracker.CheckScannedWalk(device_addr, size).walks) [[unlikely]] {
            ++check.mismatches;
            ++check.total_mismatches;
            if (check.logged < 16) {
                ++check.logged;
                LOG_ERROR(HW_GPU,
                          "Nx2 storage sync memo verify: a memo hit at {:#x}+{:#x} has CPU or "
                          "unprotected pages",
                          device_addr, size);
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (check.last_report == std::chrono::steady_clock::time_point{}) {
            check.last_report = now;
        } else if (now - check.last_report >= std::chrono::seconds{10}) {
            if (check.mismatches != 0) {
                LOG_ERROR(HW_GPU,
                          "Nx2 storage sync memo verify, last 10 s: {} of {} memo hits wrong; "
                          "since start {} of {}",
                          check.mismatches, check.checked, check.total_mismatches,
                          check.total_checked);
            } else {
                LOG_INFO(HW_GPU,
                         "Nx2 storage sync memo verify, last 10 s: 0 of {} memo hits wrong; "
                         "since start {} of {}",
                         check.checked, check.total_mismatches, check.total_checked);
            }
            check.checked = 0;
            check.mismatches = 0;
            check.last_report = now;
        }
    }
    const bool no_upload = SynchronizeBuffer(buffer, device_addr, size);
    // The walk clears every CPU and Untracked bit of the range (hot pages are off) and sets
    // none: with the sum unchanged across it, nothing marked the range during it either.
    u64 gen_after = 0;
    if (tracked && memory_tracker.DirtyGenerationSum(device_addr, size, gen_after) &&
        gen_after == gen_before) {
        memo = StorageSyncMemo{
            .device_addr = device_addr,
            .size = size,
            .gen_sum = gen_after,
        };
    }
    return no_upload;
}

template <class P>
bool BufferCache<P>::SynchronizeGeometry(Buffer& buffer, DAddr device_addr, u32 size, u32 slot) {
    namespace FillCuts = VideoCommon::FillCuts;
    FillCuts::GeomStats& stats = FillCuts::g_geom;
    StorageSyncMemo& memo = geom_sync_memo[slot];
    FillCuts::Add(slot == NUM_VERTEX_BUFFERS ? stats.index_syncs : stats.vertex_syncs);
    const bool time_check = stats.check_sample.Take();
    const u64 check_start = time_check ? FillCuts::Ticks() : 0;
    u64 gen_before = 0;
    const bool tracked = memory_tracker.DirtyGenerationSum(device_addr, size, gen_before);
    const bool hit = tracked && memo.size == size && memo.device_addr == device_addr &&
                     memo.gen_sum == gen_before;
    if (time_check) [[unlikely]] {
        FillCuts::Add(stats.check_ticks, FillCuts::Ticks() - check_start);
    }
    if (hit) {
        FillCuts::Add(stats.hits);
        if (geom_keep == FillCuts::On) {
            // Left clean at this sum, and no page of its managers gained a CPU or Untracked bit
            // since: the walk would upload nothing and re-protect nothing. The read log's range
            // is logged as SynchronizeBuffer logs it.
            MemReadLog::Range(MemReadLog::Kind::Bound, device_addr, size);
            return true;
        }
    }
    // 2 (verify): a hit's range must hold no CPU or Untracked page (before the walk takes them).
    const bool dirty_on_hit = hit && geom_keep == FillCuts::Verify &&
                              memory_tracker.CheckScannedWalk(device_addr, size).walks;
    const bool time_hit = hit && stats.hit_sample.Take();
    const u64 hit_start = time_hit ? FillCuts::Ticks() : 0;
    const bool no_upload = SynchronizeBuffer(buffer, device_addr, size);
    if (time_hit) [[unlikely]] {
        FillCuts::Add(stats.hit_ticks, FillCuts::Ticks() - hit_start);
    }
    if (hit) {
        // 2 and 3: a hit's synchronize must upload nothing (and at 2 find no dirty page).
        FillCuts::Add(stats.compare.checked);
        if ((!no_upload || dirty_on_hit) && stats.compare.Differ()) [[unlikely]] {
            LOG_ERROR(HW_GPU,
                      "Fill geometry keep (FRITCHII_FILL_GEOM_KEEP={}): a memo hit at {:#x}+{:#x} "
                      "(slot {}) {}",
                      geom_keep, device_addr, size, slot,
                      no_upload ? "has CPU or unprotected pages" : "uploaded");
        }
    }
    // The walk clears every CPU and Untracked bit of the range (hot pages are off) and sets
    // none: with the sum unchanged across it, nothing marked the range during it either.
    u64 gen_after = 0;
    if (tracked && memory_tracker.DirtyGenerationSum(device_addr, size, gen_after) &&
        gen_after == gen_before) {
        memo = StorageSyncMemo{
            .device_addr = device_addr,
            .size = size,
            .gen_sum = gen_after,
        };
    } else {
        memo.size = 0;
    }
    return no_upload;
}

template <class P>
void BufferCache<P>::UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                                  std::span<BufferCopy> copies) {
    MemStageStats::Uploaded(copies.size(), total_size_bytes);
    const Common::WriteRing::ReadSiteScope read_site{"buffer upload"};
    // FRITCHII_BANK_CENSUS: the pages these copies read from guest memory (the bank check's next
    // compare of each takes the mark).
    if (Core::DevicePageFlags* const census_uploads =
            native_bank_census_uploads.load(std::memory_order_acquire);
        census_uploads != nullptr &&
        VideoCore::Nx2Debug::g_bank_census.load(std::memory_order_relaxed) > 0) [[unlikely]] {
        for (const BufferCopy& copy : copies) {
            census_uploads->Mark(buffer.CpuAddr() + copy.dst_offset, copy.size);
        }
    }
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        MappedUploadMemory(buffer, total_size_bytes, copies);
    } else {
        ImmediateUploadMemory(buffer, largest_copy, copies);
    }
}

template <class P>
void BufferCache<P>::ImmediateUploadMemory([[maybe_unused]] Buffer& buffer,
                                           [[maybe_unused]] u64 largest_copy,
                                           [[maybe_unused]] std::span<const BufferCopy> copies) {
    if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
        std::span<u8> immediate_buffer;
        for (const BufferCopy& copy : copies) {
            std::span<const u8> upload_span;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            VideoCommon::OneFollower::BeforeEngineRead(device_addr, copy.size);
            if (IsRangeGranular(device_addr, copy.size)) {
                auto* const ptr = device_memory.GetPointer<u8>(device_addr);
                if (ptr != nullptr) {
                    upload_span = std::span(ptr, copy.size);
                }
            } else {
                if (immediate_buffer.empty()) {
                    immediate_buffer = ImmediateBuffer(largest_copy);
                }
                device_memory.ReadBlockUnsafe(device_addr, immediate_buffer.data(), copy.size);
                upload_span = immediate_buffer.subspan(0, copy.size);
            }
            buffer.ImmediateUpload(copy.dst_offset, upload_span);
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemory([[maybe_unused]] Buffer& buffer,
                                        [[maybe_unused]] u64 total_size_bytes,
                                        [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            VideoCommon::OneFollower::BeforeEngineRead(device_addr, copy.size);
            // FRITCHII_BANK_FLAGGED: the bank snapshots of these bytes become what the host
            // buffer gets (guest memory may hold newer bytes already). Never read back from the
            // staging memory: it is write-combined, uncached for the CPU (reading it back cost
            // 60% of the back thread at 2,500 pages a second). A copy over a snapshotted page
            // reads guest memory once into cached scratch, which fills staging and snapshots.
            if (native_bank_flagged_on &&
                NativeBankAnySnapshot(device_addr, device_addr + copy.size)) [[unlikely]] {
                native_bank_refresh_scratch.resize_destructive(copy.size);
                u8* const scratch = native_bank_refresh_scratch.data();
                device_memory.ReadBlockUnsafe(device_addr, scratch, copy.size);
                std::memcpy(src_pointer, scratch, copy.size);
                RefreshNativeBankSnapshots(device_addr, copy.size, scratch);
            } else {
                device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            }
            MemReadLog::Staged(MemReadLog::Kind::Upload, device_memory, device_addr, src_pointer,
                               copy.size);
            // FRITCHII_LINK_STATS=2: the same bytes as last time? From guest memory, the staging
            // mapping is write-combined.
            if (LinkStats::ChunkCensusOn()) [[unlikely]] {
                thread_local std::vector<u8> census_bytes;
                census_bytes.resize(copy.size);
                device_memory.ReadBlockUnsafe(device_addr, census_bytes.data(), copy.size);
                LinkStats::NoteUploadChunks(device_addr, census_bytes.data(), copy.size);
            }
            // Large uploads are mostly geometry the host GPU now has its own copy of: TickFrame
            // marks their guest memory cold (see TextureCache::RefreshContents).
            if (copy.size >= 256_KiB) {
                uploaded_buffer_ranges.emplace_back(device_addr, copy.size);
            }
            // Apply the staging offset
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    }
}

template <class P>
bool BufferCache<P>::InlineMemory(DAddr dest_address, size_t copy_size,
                                  std::span<const u8> inlined_buffer) {
    PromoteAhead();
    NoteAheadLevelCall(dest_address, copy_size, "InlineMemory");
    const bool is_dirty = IsRegionRegistered(dest_address, copy_size);
    if (!is_dirty) {
        return false;
    }
    DAddr aligned_start = Common::AlignDown(dest_address, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(dest_address + copy_size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        return false;
    }

    InlineMemoryImplementation(dest_address, copy_size, inlined_buffer);

    return true;
}

template <class P>
void BufferCache<P>::InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                                std::span<const u8> inlined_buffer) {
    NoteAheadLevelCall(dest_address, copy_size, "InlineMemoryImplementation");
    Core::g_buffer_write_pages.Mark(dest_address, copy_size);
    Core::g_watch_write_pages.Mark(dest_address, copy_size);
    ClearDownload(dest_address, copy_size);
    gpu_modified_ranges.Subtract(dest_address, copy_size);
    MarkVertexBuffersStale(dest_address, copy_size);

    BufferId buffer_id = FindBuffer(dest_address, static_cast<u32>(copy_size), false);
    auto& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, dest_address, static_cast<u32>(copy_size));

    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(copy_size);
        std::array copies{BufferCopy{
            .src_offset = upload_staging.offset,
            .dst_offset = buffer.Offset(dest_address),
            .size = copy_size,
        }};
        u8* const src_pointer = upload_staging.mapped_span.data();
        std::memcpy(src_pointer, inlined_buffer.data(), copy_size);
        // FRITCHII_BANK_FLAGGED: these bytes are the host copy's now, as an upload's are (from
        // the caller's cached bytes, never the write-combined staging memory).
        if (native_bank_flagged_on &&
            NativeBankAnySnapshot(dest_address, dest_address + copy_size)) [[unlikely]] {
            RefreshNativeBankSnapshots(dest_address, copy_size, inlined_buffer.data());
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    } else {
        buffer.ImmediateUpload(buffer.Offset(dest_address), inlined_buffer.first(copy_size));
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer) {
    DownloadBufferMemory(buffer, buffer.CpuAddr(), buffer.SizeBytes());
}

template <class P>
void BufferCache<P>::DownloadBuffersMemory(std::span<const BufferId> buffer_ids) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    const Common::WriteRing::SiteScope write_site{"buffer download (DownloadBuffersMemory)", true};
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    std::vector<Download> downloads;
    u64 total_size_bytes = 0;
    for (const BufferId buffer_id : buffer_ids) {
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buffer_addr = buffer.CpuAddr();
        memory_tracker.ForEachDownloadRangeAndClear(
            buffer_addr, buffer.SizeBytes(), [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                   [&](DAddr start, DAddr end) {
                    downloads.push_back({buffer_id, BufferCopy{
                                                        .src_offset = start - buffer_addr,
                                                        .dst_offset = total_size_bytes,
                                                        .size = end - start,
                                                    }});
                    total_size_bytes += Common::AlignUp(end - start, 64);
                });
                ClearReplacedDownload(device_addr_out, range_size);
                gpu_modified_ranges.Subtract(device_addr_out, range_size);
            });
    }
    if (total_size_bytes == 0) {
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    std::ranges::stable_sort(downloads, {}, &Download::buffer_id);
    boost::container::small_vector<BufferCopy, 16> buffer_copies;
    for (size_t i = 0; i < downloads.size();) {
        const BufferId buffer_id = downloads[i].buffer_id;
        Buffer& buffer = slot_buffers[buffer_id];
        buffer_copies.clear();
        while (i < downloads.size() && downloads[i].buffer_id == buffer_id) {
            Download& download = downloads[i++];
            download.copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(download.copy.src_offset, download.copy.size);
            buffer_copies.push_back(download.copy);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, buffer_copies, true);
    }
    runtime.Finish();
    const u8* const mapped_memory = download_staging.mapped_span.data();
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const Download& download : downloads) {
        const DAddr address = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        landing_begin = (std::min)(landing_begin, address);
        landing_end = (std::max)(landing_end, address + download.copy.size);
    }
    const VideoCore::DownloadLanding landing{landing_begin, landing_end};
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        WriteDownloadedMemory(
            cpu_written, device_addr,
            mapped_memory + (download.copy.dst_offset - download_staging.offset),
            download.copy.size);
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer, DAddr device_addr, u64 size) {
    PromoteAhead(); // FRITCHII_MEMSTAGE_MARKS_AHEAD: a reader of the real sets
    boost::container::small_vector<BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    memory_tracker.ForEachDownloadRangeAndClear(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](DAddr start, DAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(BufferCopy{
                    .src_offset = new_offset,
                    .dst_offset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
                largest_copy = (std::max)(largest_copy, new_size);
            };

            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            ClearReplacedDownload(device_addr_out, range_size);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    // Guest memory is written once the host GPU ran the copies, inside a landing over their
    // bytes: the CPU writes collected meanwhile keep their bytes too.
    DAddr landing_begin = ~DAddr{0};
    DAddr landing_end = 0;
    for (const BufferCopy& copy : copies) {
        landing_begin = (std::min)(landing_begin, buffer.CpuAddr() + copy.src_offset);
        landing_end = (std::max)(landing_end, buffer.CpuAddr() + copy.src_offset + copy.size);
    }
    // EDEN_NX2_DOWNLOAD_STATS=1: how buffer downloads (reactive flushes among them) are made and
    // what they wait for, logged every 5 s.
    static const bool download_stats = std::getenv("EDEN_NX2_DOWNLOAD_STATS") != nullptr;
    struct DownloadStats {
        u64 mapped{}, staged{}, mapped_ns{}, staged_ns{}, bytes{}, writer_done{};
        std::chrono::steady_clock::time_point last{std::chrono::steady_clock::now()};
    };
    static DownloadStats stats;
    const auto stats_begin = download_stats ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
    const auto note_download = [&](bool mapped_path) {
        if (!download_stats) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const u64 ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - stats_begin).count());
        (mapped_path ? stats.mapped : stats.staged) += 1;
        (mapped_path ? stats.mapped_ns : stats.staged_ns) += ns;
        stats.bytes += total_size_bytes;
        if (now - stats.last >= std::chrono::seconds{5}) {
            const double window = std::chrono::duration<double>(now - stats.last).count();
            LOG_INFO(HW_GPU,
                     "Download stats: {:.0f} mapped/s ({:.1f} ms/s), {:.0f} staged/s ({:.1f} ms/s), "
                     "{:.0f} KiB/s, writer already done {:.0f}/s",
                     stats.mapped / window, stats.mapped_ns / 1e6 / window, stats.staged / window,
                     stats.staged_ns / 1e6 / window, stats.bytes / 1024.0 / window,
                     stats.writer_done / window);
            stats = DownloadStats{};
        }
    };
    if constexpr (!IS_OPENGL) {
        if (download_stats && runtime.IsFree(buffer.getWriteTick())) {
            ++stats.writer_done;
        }
    }

    if constexpr (requires { buffer.Mapped(); }) {
        if (const auto mapped = buffer.Mapped(); !mapped.empty()) {
            runtime.Wait(buffer.LastUsageTick());
            const VideoCore::DownloadLanding landing{landing_begin, landing_end};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            for (const BufferCopy& copy : copies) {
                WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + copy.src_offset,
                                      mapped.data() + copy.src_offset, copy.size);
            }
            note_download(true);
            return;
        }
    }
    if constexpr (USE_MEMORY_MAPS) {
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        const u8* const mapped_memory = download_staging.mapped_span.data();
        const std::span<BufferCopy> copies_span(copies.data(), copies.data() + copies.size());
        for (BufferCopy& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(copy.src_offset, copy.size);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, copies_span, true);
        runtime.Finish();
        const VideoCore::DownloadLanding landing{landing_begin, landing_end};
        const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
        for (const BufferCopy& copy : copies) {
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            // Undo the modified offset
            const u64 dst_offset = copy.dst_offset - download_staging.offset;
            const u8* copy_mapped_memory = mapped_memory + dst_offset;
            WriteDownloadedMemory(cpu_written, copy_device_addr, copy_mapped_memory, copy.size);
        }
        note_download(false);
    } else {
        const std::span<u8> immediate_buffer = ImmediateBuffer(largest_copy);
        for (const BufferCopy& copy : copies) {
            buffer.ImmediateDownload(copy.src_offset, immediate_buffer.subspan(0, copy.size));
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            const VideoCore::DownloadLanding landing{copy_device_addr,
                                                     copy_device_addr + copy.size};
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            WriteDownloadedMemory(cpu_written, copy_device_addr, immediate_buffer.data(), copy.size);
        }
    }
}

template <class P>
void BufferCache<P>::DeleteBuffer(BufferId buffer_id, bool do_not_mark) {
    ++buffer_deletions;
    VideoCommon::Churn::Add(VideoCommon::Churn::BuffersDeleted);
    if (MemStageStats::Enabled()) [[unlikely]] {
        MemStageStats::BufferDeleted(slot_buffers[buffer_id].SizeBytes());
    }
    // EDEN_NX2_BUFFER_CHURN_STATS: joins pass do_not_mark, the collector does not.
    VideoCommon::BufferChurnStats::BufferDeleted(
        buffer_id == inline_buffer_id ? VideoCommon::BufferChurnStats::Deleted::Inline
        : do_not_mark                 ? VideoCommon::BufferChurnStats::Deleted::Join
                                      : VideoCommon::BufferChurnStats::Deleted::Collector,
        slot_buffers[buffer_id].SizeBytes());
    if constexpr (requires { runtime.OnBufferDeleted(slot_buffers[buffer_id]); }) {
        runtime.OnBufferDeleted(slot_buffers[buffer_id]);
    }
    read_trapped_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(),
                                 slot_buffers[buffer_id].SizeBytes());
    cpu_read_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(), slot_buffers[buffer_id].SizeBytes());
    const auto scalar_replace = [buffer_id](Binding& binding) {
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
        }
    };
    const auto replace = [scalar_replace](auto& bindings) {
        std::ranges::for_each(bindings, scalar_replace);
    };
    // Every channel keeps its own bindings, and a resolved uniform buffer id survives across its
    // draws. A channel left out would address whatever buffer reuses the slot next at the old
    // offset (Ocarina of Time draws from several channels: a frame of garbage constants).
    const auto scrub = [&](BufferCacheChannelInfo& channel) {
        const CacheInputs inputs = CacheInputs::Of(channel);
        if (channel.index_buffer.buffer_id == buffer_id) {
            channel.index_buffer.buffer_id = BufferId{};
            inputs.MarkIndexBufferDirty();
        }
        for (u32 index = 0; index < channel.vertex_buffers.size(); index++) {
            auto& binding = channel.vertex_buffers[index];
            if (binding.buffer_id != buffer_id) {
                continue;
            }
            binding.buffer_id = BufferId{};
            if (&channel == channel_state) {
                UpdateVertexBufferSlot(index, binding);
            }
            inputs.MarkVertexBuffersDirty(1U << index);
        }
        std::ranges::for_each(channel.uniform_buffers, replace);
        std::ranges::for_each(channel.storage_buffers, replace);
        std::ranges::for_each(channel.texture_buffers, replace);
        replace(channel.transform_feedback_buffers);
        scalar_replace(channel.count_buffer_binding);
        scalar_replace(channel.indirect_buffer_binding);
        replace(channel.compute_uniform_buffers);
        replace(channel.compute_storage_buffers);
        replace(channel.compute_texture_buffers);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel.dirty_uniform_buffers.fill(~u32{0});
            channel.uniform_buffer_binding_sizes.fill({});
        }
        channel.has_deleted_buffers = true;
    };
    scrub(*channel_state);
    for (const size_t channel_id : active_channel_ids) {
        if (&channel_storage[channel_id] != channel_state) {
            scrub(channel_storage[channel_id]);
        }
    }

    // Adapted from the pre-refresh branch: merged buffers invalidate segment references too.
    InvalidateMultiRangeSegments(std::span{graphics_segments}, buffer_id);
    InvalidateMultiRangeSegments(std::span{compute_segments}, buffer_id);

    // Mark the whole buffer as CPU written to stop tracking CPU writes
    if (!do_not_mark) {
        Buffer& buffer = slot_buffers[buffer_id];
        memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
    }

    Unregister(buffer_id);
    // The collector can delete the inline index buffer too: the next inline draw makes a new one.
    if (buffer_id == inline_buffer_id) {
        inline_buffer_id = NULL_BUFFER_ID;
    }

#ifdef YUZU_LEGACY
    if (!do_not_mark || !immediately_free)
#endif
    {
        u64 current_tick = 0;
        if constexpr (!IS_OPENGL) {
            current_tick = runtime.CurrentTick();
        }
        delayed_destruction_ring.Push(std::move(slot_buffers[buffer_id]), current_tick);
    }

    slot_buffers.erase(buffer_id);
}

template <class P>
void BufferCache<P>::TraceTablePointer(bool compute, size_t stage, size_t ssbo_index,
                                       u32 cbuf_index, u32 cbuf_offset, u32 indirect_offset,
                                       GPUVAddr holder, const Binding& binding) {
    // A table entry is read here from guest memory, and again by the shader from the holder's
    // host buffer. Logs each change of a slot's entry or size, and whether the two copies of the
    // entry can differ: GPU-written (host newer) or CPU-written and not yet uploaded (guest newer).
    static const bool enabled = std::getenv("EDEN_NX2_TABLE_POINTER_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    const GPUVAddr entry_addr = holder + indirect_offset;
    const GPUVAddr entry = Inputs().ReadGameU64(entry_addr);
    const u64 key = (u64{compute} << 63) | (u64{stage & 7} << 60) | (u64{ssbo_index & 63} << 54) |
                    (u64{cbuf_index & 31} << 49) | (u64{cbuf_offset & 0x7ffff} << 30) |
                    (indirect_offset & 0x3fffffff);
    const std::pair<GPUVAddr, u32> state{entry, binding.size};
    const auto [it, inserted] = traced_table_pointers.try_emplace(key, state);
    if (!inserted && it->second == state) {
        return;
    }
    it->second = state;
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    if (++lines > max_lines) {
        if (lines == max_lines + 1) {
            LOG_INFO(HW_GPU, "Table pointer trace stopped after {} lines", max_lines);
        }
        return;
    }
    const std::optional<DAddr> entry_device = gpu_memory->GpuToCpuAddress(entry_addr);
    const bool gpu_modified =
        entry_device && memory_tracker.IsRegionGpuModified(*entry_device, sizeof(u64));
    const bool cpu_modified =
        entry_device && memory_tracker.IsRegionCpuModified(*entry_device, sizeof(u64));
    LOG_INFO(HW_GPU,
             "Table pointer frame={} {} stage={} ssbo={} c{}[{:#x}]+{:#x}: holder={:#x} "
             "entry={:#x} bound={:#x} size={:#x} entry_gpu_modified={} entry_cpu_modified={}",
             frame_tick, compute ? "compute" : "graphics", stage, ssbo_index, cbuf_index,
             cbuf_offset, indirect_offset, holder, entry, binding.device_addr, binding.size,
             gpu_modified, cpu_modified);
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                             bool native_pointer, bool table_pointer,
                                             bool compute) const {
    const GPUVAddr gpu_addr = Inputs().ReadGameU64(ssbo_addr);
    MemReadLog::Value(MemReadLog::Kind::StorageAddress, gpu_memory, ssbo_addr, gpu_addr);
    // EDEN_NX2_MEMSTAGE_STATS: the pointer word.
    const MemStageStats::Kind address_kind =
        compute ? MemStageStats::Kind::CStorageAddress : MemStageStats::Kind::StorageAddress;
    MemStageStats::Read(address_kind, sizeof(u64));

    if (gpu_addr == 0) {
        return NULL_BINDING;
    }

    // The size word after a plain pointer is part of the answer; a read-only native compute
    // pointer reads it too (a size word past the bare reach grows the binding). Pool pointers
    // ignore it.
    const bool reads_size_word =
        !(table_pointer && !is_written) && (!native_pointer || (compute && !is_written));
    const u64 next_qword = reads_size_word ? Inputs().ReadGameU64(ssbo_addr + 8) : 0;
    if (reads_size_word) {
        MemReadLog::Value(MemReadLog::Kind::StorageSize, gpu_memory, ssbo_addr + 8, next_qword);
    }
    // EDEN_NX2_MEMSTAGE_STATS: the size word, in the pointer's range.
    MemStageStats::Read(address_kind, reads_size_word ? sizeof(u64) : 0, 0);
    // Every draw rebinds each storage buffer: the layout lookup (a mutex), the range walk and
    // two page walks cost Ocarina's GPU thread ~6%. Answers hold until the page table changes.
    struct CachedBinding {
        size_t owner; ///< MemoryManager id + 1, 0 when empty
        GPUVAddr gpu_addr;
        u64 next_qword;
        u64 serial;
        u64 backing; ///< FRITCHII_BARE_SSBO_CLAMP: the device backing's generation, else 0
        u32 flags;
        Binding binding;
    };
    constexpr size_t Ways = 4;
    thread_local std::array<std::array<CachedBinding, Ways>, 512> binding_cache{};
    // FRITCHII_BARE_SSBO_MIB is live: its reach is part of the answer's key (MiB, from bit 3).
    const u32 bare_reach = native_pointer ? VideoCore::Nx2Debug::BareStorageReach() : 0;
    // FRITCHII_BARE_SSBO_CLAMP (live, default 0): a bare binding stops at the first device page
    // with no host backing; the answer then also holds only until the backing changes (bit 31).
    const bool bare_clamp =
        native_pointer && VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_bare_ssbo_clamp,
                                                           "FRITCHII_BARE_SSBO_CLAMP") != 0;
    const u32 flags = (is_written ? 1U : 0U) | (native_pointer ? 2U : 0U) |
                      (table_pointer ? 4U : 0U) | ((bare_reach >> 20) << 3) |
                      (bare_clamp ? 1U << 31 : 0U);
    const size_t owner = gpu_memory->GetID() + 1;
    const u64 serial = gpu_memory->MapSerial();
    const u64 backing = bare_clamp ? device_memory.TranslationGeneration() : 0;
    auto& set = binding_cache[((gpu_addr ^ next_qword ^ flags) * 0x9E3779B97F4A7C15ULL) >> 55];
    for (size_t way = 0; way < Ways; ++way) {
        const CachedBinding& cached = set[way];
        if (cached.owner == owner && cached.gpu_addr == gpu_addr &&
            cached.next_qword == next_qword && cached.flags == flags && cached.serial == serial &&
            cached.backing == backing) {
            const Binding hit = cached.binding;
            std::rotate(set.begin(), set.begin() + way, set.begin() + way + 1);
            if (compute) {
                Nx2Profile::Add(Nx2Profile::Counter::StorageBindingHits, 1);
            }
            return hit;
        }
    }
    if (compute) {
        Nx2Profile::Add(Nx2Profile::Counter::StorageBindingMisses, 1);
    }
    const Binding binding = [&]() -> Binding {
        u32 size = [&]() {
            const u32 memory_layout_size = static_cast<u32>((std::min)(
                u64{gpu_memory->GetMemoryLayoutSize(gpu_addr)}, u64{(std::numeric_limits<u32>::max)()}));
            const u32 packed_size = static_cast<u32>(next_qword);
            const bool next_qword_is_size = static_cast<u32>(next_qword >> 32) == 0 &&
                                            packed_size != 0 && packed_size <= memory_layout_size;
            const u64 reach = (std::min)(runtime.GetMaxStorageBufferRange(), u64{1_GiB});
            const u32 limit = static_cast<u32>(reach - (std::min)(reach, u64{64_KiB}));
            // A pointer read from a table in memory is the base of an object pool, read anywhere in
            // it: bound to the end of its mapping, as far as a storage buffer may reach (at most
            // 1 GiB). Ocarina of Time's grass instances lie 12-18 MiB past theirs (an 8 MiB cap
            // culled every blade), and its light probe records reach about 130 MiB past theirs: a
            // 64 MiB cap left the far probes outside the binding, and the fog took one overbright
            // constant from them. Written pools keep a cap: a write marks the whole binding
            // GPU-modified.
            if (table_pointer && !is_written) {
                return (std::min)(memory_layout_size, limit);
            }
            // A native (Switch 2) compute program's pointer is bare: the word after it is another
            // parameter (a dimension, stride or texture handle), not an NVN buffer size. Taking it
            // as one bound Fortune's Weave's DLSS input features to 266 bytes and dropped the rest.
            if (native_pointer) {
                // A bare pointer reaches FRITCHII_BARE_SSBO_MIB (16 MiB by default; 8 before).
                // OoT's light/decal cluster list is mapped about 9.5 MiB past its pointer: a
                // compute program writes it, and the water and the mist over it read it. Capped at
                // 8 MiB the writes past the cap were dropped and the reads past it got zero or old
                // entries (the console has no bounds there): stair-stepped tiles around Link in
                // water, waterfalls and river mist. Written bindings stop at 16 MiB at most (a
                // written graphics binding is marked GPU-modified in full); the object pools mapped
                // ~83 MiB past their pointers stay cut at the reach.
                const u32 cap =
                    is_written ? (std::min)(bare_reach, static_cast<u32>(16_MiB)) : bare_reach;
                const u32 bare = (std::min)({memory_layout_size, cap, limit});
                if (memory_layout_size > 8_MiB) {
                    // The first 24 different bindings the old 8 MiB cap cuts, for the switch's A/B.
                    static std::atomic<bool> log_full{};
                    if (!log_full.load(std::memory_order_relaxed)) [[unlikely]] {
                        static std::mutex log_mutex;
                        static std::vector<std::pair<GPUVAddr, u32>> logged;
                        std::scoped_lock lock{log_mutex};
                        const std::pair entry{gpu_addr, flags};
                        if (logged.size() < 24 && std::ranges::find(logged, entry) == logged.end()) {
                            logged.push_back(entry);
                            log_full.store(logged.size() >= 24, std::memory_order_relaxed);
                            LOG_INFO(HW_GPU,
                                     "Bare {} storage binding at {:#x} ({}): mapping {:#x} bytes, "
                                     "bound {:#x} (FRITCHII_BARE_SSBO_MIB={})",
                                     compute ? "compute" : "graphics", gpu_addr,
                                     is_written ? "written" : "read-only", memory_layout_size, bare,
                                     bare_reach >> 20);
                        }
                    }
                }
                // Mario Kart World's lighting passes read past 8 MiB from theirs, inside the size
                // word after the pointer (a ported change reads it): capped at 8 MiB those reads
                // came back zero and lit land and characters too dark. A read-only compute binding
                // takes a larger size word. Binding to the end of the mapping instead spanned
                // Ocarina of Time's ~1 GiB heap and backed it with a 1 GiB buffer.
                if (compute && !is_written && next_qword_is_size && packed_size > bare) {
                    return (std::min)(packed_size, limit);
                }
                return bare;
            }
            if (next_qword_is_size) {
                return packed_size;
            }
            return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
        }();
        if ((native_pointer || table_pointer) && size != 0) {
            // Stop where the mapping or its device range does: the binding is one device range, and
            // a written one isn't split into separate ranges without sparse sources.
            size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr, size));
        }
        // FRITCHII_BARE_SSBO_CLAMP: stop at the first device page with no host backing (read as
        // zeros: the AMD tester's market load read 16 MiB of them at one bare compute window,
        // test 4 §5b); none backed at the start is the null binding (size 0 below).
        if (bare_clamp && size != 0) {
            if (const std::optional<DAddr> start = gpu_memory->GpuToCpuAddress(gpu_addr)) {
                const size_t backed = device_memory.BackedPrefix(*start, size);
                if (backed < size) {
                    static std::atomic<u32> logged{};
                    if (logged.fetch_add(1, std::memory_order_relaxed) < 24) [[unlikely]] {
                        LOG_WARNING(HW_GPU,
                                    "Bare {} storage binding at {:#x} (device {:#x}, {}): {:#x} "
                                    "bytes, host-backed {:#x}: {} (FRITCHII_BARE_SSBO_CLAMP=1)",
                                    compute ? "compute" : "graphics", gpu_addr, *start,
                                    is_written ? "written" : "read-only", size, backed,
                                    backed == 0 ? "null binding" : "clamped");
                    }
                    size = static_cast<u32>(backed);
                }
            }
        }

        // Alignment only applies to the offset of the buffer
        const u32 alignment = runtime.GetStorageBufferAlignment();
        const GPUVAddr aligned_gpu_addr = Common::AlignDown(gpu_addr, alignment);
        const u32 aligned_size = static_cast<u32>(gpu_addr - aligned_gpu_addr) + size;

        const std::optional<DAddr> aligned_device_addr =
            gpu_memory->GpuToCpuAddress(aligned_gpu_addr);
        if (!aligned_device_addr || size == 0) {
            LOG_DEBUG(HW_GPU, "Failed to find storage buffer for cbuf index {}", cbuf_index);
            return NULL_BINDING;
        }
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        ASSERT_MSG(device_addr, "Unaligned storage buffer address not found for cbuf index {}",
                   cbuf_index);
        // The end address used for size calculation does not need to be aligned
        const DAddr cpu_end = Common::AlignUp(*device_addr + size, Core::DEVICE_PAGESIZE);

        u32 binding_size = static_cast<u32>(cpu_end - *aligned_device_addr);
        if (is_written) {
            binding_size = aligned_size;
        }
        return Binding{
            .device_addr = *aligned_device_addr,
            .gpu_addr = aligned_gpu_addr,
            .size = binding_size,
            .buffer_id = BufferId{},
        };
    }();
    std::rotate(set.begin(), set.end() - 1, set.end());
    set[0] = {owner, gpu_addr, next_qword, serial, backing, flags, binding};
    return binding;
}

template <class P>
TextureBufferBinding BufferCache<P>::GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                             PixelFormat format) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    TextureBufferBinding binding;
    if (!device_addr || size == 0) {
        binding.device_addr = 0;
        binding.size = 0;
        binding.buffer_id = NULL_BUFFER_ID;
        binding.format = PixelFormat::Invalid;
    } else {
        binding.device_addr = *device_addr;
        binding.size = size;
        binding.buffer_id = BufferId{};
        binding.format = format;
    }
    return binding;
}

template <class P>
std::span<const u8> BufferCache<P>::ImmediateBufferWithData(DAddr device_addr, size_t size) {
    VideoCommon::OneFollower::BeforeEngineRead(device_addr, size);
    u8* const base_pointer = device_memory.GetPointer<u8>(device_addr);
    if (IsRangeGranular(device_addr, size) ||
        base_pointer + size == device_memory.GetPointer<u8>(device_addr + size)) {
        return std::span(base_pointer, size);
    } else {
        const std::span<u8> span = ImmediateBuffer(size);
        const Common::WriteRing::ReadSiteScope read_site{"immediate buffer"};
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        return span;
    }
}

template <class P>
std::span<u8> BufferCache<P>::ImmediateBuffer(size_t wanted_capacity) {
    immediate_buffer_alloc.resize_destructive(wanted_capacity);
    return std::span<u8>(immediate_buffer_alloc.data(), wanted_capacity);
}

template <class P>
bool BufferCache<P>::HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept {
    return ((channel_state->fast_bound_uniform_buffers[stage] >> binding_index) & 1u) != 0;
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectCount() {
    VideoCommon::MemoryStageLink::CheckBackCache("GetDrawIndirectCount");
    auto& buffer = slot_buffers[channel_state->count_buffer_binding.buffer_id];
    return std::make_pair(&buffer, buffer.Offset(channel_state->count_buffer_binding.device_addr));
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectBuffer() {
    VideoCommon::MemoryStageLink::CheckBackCache("GetDrawIndirectBuffer");
    auto& buffer = slot_buffers[channel_state->indirect_buffer_binding.buffer_id];
    return std::make_pair(&buffer,
                          buffer.Offset(channel_state->indirect_buffer_binding.device_addr));
}

} // namespace VideoCommon
