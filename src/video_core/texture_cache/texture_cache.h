// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <vector>
#include <optional>
#include <bit>
#include <type_traits>
#include <unordered_map>
#include "common/cityhash.h"
#include "common/container/unordered_map.h"
#include <boost/container/small_vector.hpp>

#include "common/alignment.h"
#include "common/host_video_memory.h"
#include "common/memory_detect.h"
#include "common/memory_detect.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "video_core/buffer_cache/mem_read_log.h"
#include "video_core/compute_skip_hook.h"
#include "video_core/control/channel_state.h"
#include "video_core/dirty_flags.h"
#include "video_core/early_compute.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_stage_link.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/small_card.h"
#include "video_core/texture_cache/host_memory_limit.h"
#include "video_core/texture_cache/image_view_base.h"
#include "video_core/texture_cache/conversion.h"
#include "video_core/texture_cache/samples_helper.h"
#include "video_core/texture_cache/texture_cache_base.h"
#include "video_core/texture_cache/relocation_memory.h"
#include "video_core/texture_cache/util.h"
#include "video_core/nx2.h"
#include "video_core/churn_counters.h"
#include "video_core/nx2_profile.h"
#include "common/nx2_hitch.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/nx2_report_census.h"
#include "video_core/one_follower_hooks.h"
#include "video_core/render_trace.h"
#include "video_core/textures/decoders.h"

namespace VideoCommon {

/// EDEN_NATIVE_DRAW_DIAGNOSTICS is read once: getenv scans the whole environment block on Windows.
[[nodiscard]] inline bool NativeDrawDiagnostics() {
    static const bool enabled = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
    return enabled;
}

using Tegra::Texture::TICEntry;
using Tegra::Texture::TSCEntry;
using VideoCore::Surface::GetFormatType;
using VideoCore::Surface::PixelFormat;
using VideoCore::Surface::SurfaceType;
using namespace Common::Literals;

/// Nanoseconds since `start` (the render target statistics of the slow clear log).
[[nodiscard]] inline u64 NsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

template <class P>
TextureCache<P>::TextureCache(Runtime& runtime_, Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : runtime{runtime_}, device_memory{device_memory_} {
    // EXP100 (fable52_answer_imagereads 2): the mutex's release closes the counter a hold opened
    // for gpu_image_filter. Before any other thread sees the cache.
    mutex.SetReleaseHook(&VideoCommon::MirrorSeq::ReleaseHook, &image_seq);
    // FRITCHII_RELOCATION_MEMORY: shapes guarded in earlier sessions (and the seed) never
    // relocate, so their first read this session isn't lost (relocation_memory.h).
    if (RelocationMemory::Enabled()) {
        for (const auto& [shape_key, guard] : RelocationMemory::Load()) {
            relocation_shapes[shape_key].guard |= guard;
        }
    }
    // Configure null sampler
    TSCEntry sampler_descriptor{};
    sampler_descriptor.min_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mag_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    sampler_descriptor.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Linear);
    sampler_descriptor.cubemap_anisotropy.Assign(1);

    // These values were chosen based on typical peak swizzle data sizes seen in some titles
    constexpr size_t SWIZZLE_DATA_BUFFER_INITIAL_CAPACITY = 8_MiB;
    constexpr size_t UNSWIZZLE_DATA_BUFFER_INITIAL_CAPACITY = 1_MiB;
    swizzle_data_buffer.resize_destructive(SWIZZLE_DATA_BUFFER_INITIAL_CAPACITY);
    unswizzle_data_buffer.resize_destructive(UNSWIZZLE_DATA_BUFFER_INITIAL_CAPACITY);

    // Make sure the first index is reserved for the null resources
    // This way the null resource becomes a compile time constant
    void(slot_images.insert(NullImageParams{}));
    void(slot_image_views.insert(runtime, NullImageViewParams{}));
    void(slot_samplers.insert(runtime, sampler_descriptor));

    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
        const s64 min_spacing_expected = device_local_memory - 1_GiB;
        const s64 min_spacing_critical = device_local_memory - 512_MiB;
        const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
        const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
        const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
        expected_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                     DEFAULT_EXPECTED_MEMORY));
        critical_memory = static_cast<u64>(
            (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                     DEFAULT_CRITICAL_MEMORY));
        minimum_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
    } else {
        expected_memory = DEFAULT_EXPECTED_MEMORY + 512_MiB;
        critical_memory = DEFAULT_CRITICAL_MEMORY + 1_GiB;
        minimum_memory = 0;
    }
    ApplyThresholdOverrides();

    const bool gpu_unswizzle_enabled = Settings::values.gpu_unswizzle_enabled.GetValue();

    if (gpu_unswizzle_enabled) {
        switch (Settings::values.gpu_unswizzle_texture_size.GetValue()) {
            case Settings::GpuUnswizzleSize::VerySmall:    gpu_unswizzle_maxsize = 16_MiB; break;
            case Settings::GpuUnswizzleSize::Small:        gpu_unswizzle_maxsize = 32_MiB; break;
            case Settings::GpuUnswizzleSize::Normal:       gpu_unswizzle_maxsize = 128_MiB; break;
            case Settings::GpuUnswizzleSize::Large:        gpu_unswizzle_maxsize = 256_MiB; break;
            case Settings::GpuUnswizzleSize::VeryLarge:    gpu_unswizzle_maxsize = 512_MiB; break;
            default:                                       gpu_unswizzle_maxsize = 128_MiB; break;
        }

        switch (Settings::values.gpu_unswizzle_stream_size.GetValue()) {
            case Settings::GpuUnswizzle::VeryLow: swizzle_chunk_size = 4_MiB; break;
            case Settings::GpuUnswizzle::Low:     swizzle_chunk_size = 8_MiB; break;
            case Settings::GpuUnswizzle::Normal:  swizzle_chunk_size = 16_MiB; break;
            case Settings::GpuUnswizzle::Medium:  swizzle_chunk_size = 32_MiB; break;
            case Settings::GpuUnswizzle::High:    swizzle_chunk_size = 64_MiB; break;
            default:                              swizzle_chunk_size = 16_MiB;
        }

        switch (Settings::values.gpu_unswizzle_chunk_size.GetValue()) {
            case Settings::GpuUnswizzleChunk::VeryLow: swizzle_slices_per_batch = 32; break;
            case Settings::GpuUnswizzleChunk::Low:     swizzle_slices_per_batch = 64; break;
            case Settings::GpuUnswizzleChunk::Normal:  swizzle_slices_per_batch = 128; break;
            case Settings::GpuUnswizzleChunk::Medium:  swizzle_slices_per_batch = 256; break;
            case Settings::GpuUnswizzleChunk::High:    swizzle_slices_per_batch = 512; break;
            default:                                   swizzle_slices_per_batch = 128;
        }
    } else {
        gpu_unswizzle_maxsize = 0;
        swizzle_chunk_size = 0;
        swizzle_slices_per_batch = 0;
    }
}

/// EDEN_NX2_ASYNC_RELAXED (1 = on, 0 = off; on by default on small cards only): relocated targets
/// keep their clear history and a clear one render frame back counts for asynchronous shaders.
[[nodiscard]] inline bool AsyncRelaxedTargets() {
    static const bool relaxed = [] {
        const char* value = std::getenv("EDEN_NX2_ASYNC_RELAXED");
        return value != nullptr ? value[0] != '0' : Common::HostHasSmallVideoMemory();
    }();
    return relaxed;
}

template <class P>
void TextureCache<P>::RunGarbageCollector(bool out_of_memory) {
    const Common::WriteRing::SiteScope write_site{"texture download (garbage collector)", true};
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::GarbageCollect};
    bool high_priority_mode = false;
    bool aggressive_mode = false;
    bool evict_old_costly = false;
    u64 ticks_to_destroy = 0;
    size_t num_iterations = 0;
    const auto Configure = [&](bool allow_aggressive) {
        high_priority_mode = out_of_memory || total_used_memory >= expected_memory;
        aggressive_mode =
            allow_aggressive && (out_of_memory || total_used_memory >= critical_memory);
        ticks_to_destroy = aggressive_mode ? 10ULL : high_priority_mode ? 25ULL : 300ULL;
        // EDEN_NX2_TC_MIN_AGE=<frames>: never evict images used more recently (thrash tests).
        static const u64 min_age = [] {
            const char* value = std::getenv("EDEN_NX2_TC_MIN_AGE");
            return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
        }();
        ticks_to_destroy = (std::max)(ticks_to_destroy, min_age);
        num_iterations = aggressive_mode ? 40 : (high_priority_mode ? 20 : 10);
        // FRITCHII_TC_COLLECT_OVER (live, default 0): past the critical memory, 40 deletes a
        // frame is not enough where the game makes more images than that. Breaking pots in
        // OoT's Hyrule Castle makes ~55 a frame (about 25 new 384x384 D32 shadow maps and the
        // frame's targets, at addresses that move every frame), none replacing an older one: the
        // cache grew ~15 images a frame until device memory ran out and the driver crashed on
        // the next image (zortos #1306, RX 9070 XT). So one more delete per 2 MiB over the
        // critical memory (up to 1,024 more), and from 1 GiB over, images unused for 3 frames go
        // too instead of 10.
        // FRITCHII_VRAM_BUDGET_MODE: also while the budget is short (budget_pressure).
        if (aggressive_mode && total_used_memory > critical_memory &&
            (budget_pressure ||
             VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_tc_collect_over,
                                              "FRITCHII_TC_COLLECT_OVER") != 0)) {
            const u64 over = total_used_memory - critical_memory;
            const size_t extra = static_cast<size_t>((std::min)(over / 2_MiB, u64{1024}));
            num_iterations += extra;
            auto& stats = collect_port_stats;
            ++stats.over_passes;
            stats.over_extra += extra;
            stats.over_peak = (std::max)(stats.over_peak, over);
            if (over >= 1_GiB) {
                ticks_to_destroy = (std::max)(min_age, u64{3});
                ++stats.over_young;
            }
        }
        // FRITCHII_TC_THRASH_GUARD: images the game keeps coming back to stay, however far past
        // critical the cache is. An allocation that failed needs the memory now.
        if (!out_of_memory) {
            ticks_to_destroy = (std::max)(ticks_to_destroy, thrash_floor);
        }
    };
    // FRITCHII_TC_COLLECT_EVICTABLE (live, default 0; zortos #1731): images this pass may not
    // delete are skipped before they count against its budget. Without it, 10 or 20 old costly
    // (ASTC) images at the old end of the list used up a normal or high-priority pass, which then
    // deleted nothing although deletable images sat right behind them.
    const bool skip_kept = VideoCore::Nx2Debug::SwitchValue(
                               VideoCore::Nx2Debug::g_tc_collect_evictable,
                               "FRITCHII_TC_COLLECT_EVICTABLE") != 0;
    const auto CanEvict = [this, &high_priority_mode, &aggressive_mode,
                           &evict_old_costly](const Image& image, bool& must_download) {
        if (True(image.flags & ImageFlagBits::IsDecoding)) {
            return false;
        }
        must_download = IsDownloadable(image) && False(image.flags & ImageFlagBits::BadOverlap);
        return !((!aggressive_mode && !evict_old_costly &&
                  True(image.flags & ImageFlagBits::CostlyLoad)) ||
                 (!high_priority_mode && must_download));
    };
    const bool covered_on = VideoCore::Nx2Debug::SwitchValue(
                                VideoCore::Nx2Debug::g_covered_retire,
                                "FRITCHII_COVERED_RETIRE") > 0;
    const auto Cleanup = [this, &num_iterations, &high_priority_mode, &aggressive_mode,
                          &CanEvict, skip_kept, out_of_memory, covered_on](ImageId image_id) {
        if (num_iterations == 0) {
            return true;
        }
        auto& image = slot_images[image_id];
        if (True(image.flags & ImageFlagBits::CollectPending)) {
            return false; // EXP121: already counted as freed
        }
        bool must_download = false;
        if (skip_kept) {
            if (!CanEvict(image, must_download)) {
                ++collect_port_stats.skipped;
                return false;
            }
            --num_iterations;
        } else {
            --num_iterations;
            if (!CanEvict(image, must_download)) {
                return false;
            }
        }
        // FRITCHII_COVERED_RETIRE: newer GPU writes cover every byte of it, so a download would
        // only put bytes in guest memory that those images' own downloads replace. Checked again
        // only after a newer image came to overlap it.
        if (must_download && covered_on &&
            image.covered_overlap_frame > image.covered_check_frame &&
            CoveredByNewerWrites(image_id, image)) {
            must_download = false;
            ++covered_stats.collect_deleted;
            covered_stats.deleted_bytes += ImageMemoryBytes(image);
            if (covered_swept.size() >= 65536) {
                covered_swept.clear();
            }
            covered_swept[image.gpu_addr] = frame_tick;
        }
        // EXP121 (live, 0 = off): record the download and delete the image at a later frame end,
        // when the GPU has written it out (LandCollectDownloads), instead of waiting here. An
        // allocation that failed needs the memory now, so it still waits.
        bool queued = false;
        // FRITCHII_LOST_WRITEBACK (default 1; zortos293/Lichii #1501's idea): a GPU-written bad
        // overlap is written back before it goes, through the same queue; it was dropped, and the
        // next image there showed other memory (the dialogue text corruption, #1310). An
        // allocation that failed needs the memory now: dropped as before.
        if (!must_download && !out_of_memory && True(image.flags & ImageFlagBits::BadOverlap)) {
            switch (QueueLostWriteback(image_id, image, LostPathCollector)) {
            case LostWriteback::Queued:
                queued = true;
                break;
            case LostWriteback::Keep:
                return false;
            case LostWriteback::Drop:
                break;
            }
        }
        if (must_download && VideoCore::Nx2Debug::Experiment(121) != 0) {
            if (out_of_memory) {
                ++collect_stats.fallback_oom;
            } else {
                queued = QueueCollectDownload(image_id);
            }
        }
        // EDEN_NX2_EXP57 (live, 0 = off): at most one download with a full wait for the GPU per
        // frame unless an allocation failed. A player's first high-priority pass downloaded
        // image after image and froze for 3 s (fable52_answer_3 s7); the rest wait for later
        // frames.
        if (!queued && must_download && !out_of_memory &&
            VideoCore::Nx2Debug::Experiment(57) != 0 && gc_waited_frame == frame_tick) {
            ++gc_downloads_deferred;
            return false;
        }
        if (!queued) {
            if (must_download) {
                gc_waited_frame = frame_tick;
                auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
                const auto copies = FullDownloadCopies(image.info);
                image.DownloadMemory(map, copies);
                runtime.Finish();
                ++collect_stats.waits;
                DropCollectDownloadsIn(image.cpu_addr, image.guest_size_bytes);
                SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, map.mapped_span, swizzle_data_buffer);
                ++gc_download_deletes;
                NoteWriteback(image); // FRITCHII_LOST_WRITEBACK's order check
                unmap_stats.downloads += image.unmapped_tick != 0 ? 1 : 0;
                // Guest memory holds the GPU's data now (DeleteImage must not mark it stale).
                image.flags &= ~ImageFlagBits::GpuModified;
                ClearStaleImage(image);
            } else {
                MarkStale(image); // before UntrackImage (see MarkStale)
            }
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, image_id);
            }
            // FRITCHII_TC_COLLECT_EVICTABLE: the block's fill stays current between the 60-frame
            // refreshes, so later passes still find the emptiest blocks.
            if constexpr (requires(const Image& img) { img.MemoryBlock(); }) {
                if (skip_kept) {
                    const auto block = image.MemoryBlock();
                    const auto it = block.block != 0 ? block_image_bytes.find(block.block)
                                                     : block_image_bytes.end();
                    if (it != block_image_bytes.end()) {
                        it->second -= (std::min)(it->second, block.size);
                    }
                }
            }
            ++(True(image.flags & ImageFlagBits::CostlyLoad) ? gc_costly_deletes : gc_deletes);
            collect_port_stats.deleted_aggressive += aggressive_mode ? 1 : 0;
            UnregisterImage(image_id);
            DeleteImage(image_id, image.scale_tick > frame_tick + 5);
        }
        if (out_of_memory) {
            return false;
        }
        if (aggressive_mode && total_used_memory < critical_memory) {
            num_iterations >>= 2;
            aggressive_mode = false;
        } else if (high_priority_mode && total_used_memory < expected_memory) {
            num_iterations >>= 1;
            high_priority_mode = false;
        }
        return false;
    };
    // Deleting by age alone left every device memory block partly used: the collector deletes
    // thousands of images a minute and new ones land in the holes, so blocks seldom empty (title
    // 1: ~0.8 GB of holes in ~190 32 MiB blocks after ten minutes, which the driver counts as
    // used). Among the oldest candidates (8 per image this pass may delete), those in the
    // emptiest blocks go first, so blocks drain and go back to the driver; placement already
    // prefers the fullest blocks (best fit). Images with memory of their own sit in the middle.
    const auto Collect = [&](u64 age) {
        if constexpr (requires(const Image& image) { image.MemoryBlock(); }) {
            std::vector<ImageId> candidates;
            const size_t limit = num_iterations * 8;
            lru_cache.ForEachItemBelow(frame_tick - age, [&](ImageId image_id) {
                if (skip_kept) {
                    // FRITCHII_TC_COLLECT_EVICTABLE: only images this pass may delete.
                    const Image& image = slot_images[image_id];
                    bool must_download = false;
                    if (True(image.flags & ImageFlagBits::CollectPending) ||
                        !CanEvict(image, must_download)) {
                        return false;
                    }
                }
                candidates.push_back(image_id);
                return candidates.size() >= limit;
            });
            if (candidates.empty()) {
                return;
            }
            if (block_image_bytes_frame == ~0ULL || frame_tick - block_image_bytes_frame >= 60) {
                block_image_bytes_frame = frame_tick;
                block_image_bytes.clear();
                for (const auto [image_id, image] : slot_images) {
                    const auto block = image->MemoryBlock();
                    if (block.block != 0) {
                        block_image_bytes[block.block] += block.size;
                    }
                }
            }
            std::vector<std::pair<double, ImageId>> ordered;
            ordered.reserve(candidates.size());
            for (const ImageId image_id : candidates) {
                const auto block = slot_images[image_id].MemoryBlock();
                double fill = 0.5;
                if (block.block != 0 && block.block_size != 0) {
                    const auto it = block_image_bytes.find(block.block);
                    fill = it != block_image_bytes.end()
                               ? static_cast<double>(it->second) / static_cast<double>(block.block_size)
                               : 0.0;
                }
                ordered.emplace_back(fill, image_id);
            }
            std::stable_sort(ordered.begin(), ordered.end(),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [fill, image_id] : ordered) {
                if (Cleanup(image_id)) {
                    break;
                }
            }
        } else {
            lru_cache.ForEachItemBelow(frame_tick - age, Cleanup);
        }
    };
    Configure(false);
    static const u64 old_age = [] {
        const char* value = std::getenv("EDEN_NX2_TC_OLD_AGE");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 1800ULL;
    }();
    if (high_priority_mode && old_age != 0 && frame_tick > old_age) {
        evict_old_costly = true;
        if (skip_kept) {
            // FRITCHII_TC_COLLECT_EVICTABLE: old costly images by block fill too, so their
            // deletes empty blocks as the other passes' do.
            Collect(old_age);
        } else {
            lru_cache.ForEachItemBelow(frame_tick - old_age, Cleanup);
        }
        evict_old_costly = false;
    }
    Collect(ticks_to_destroy);
    if (out_of_memory || total_used_memory >= critical_memory) {
        Configure(true);
        Collect(ticks_to_destroy);
    }
}

template <class P>
bool TextureCache<P>::QueueCollectDownload(ImageId image_id) {
    if constexpr (requires { runtime.CurrentTick(); }) {
        // Caps on what is held at once; past them the collector waits as before.
        constexpr size_t MaxQueued = 64;
        constexpr u64 MaxStaging = 256_MiB;
        Image& image = slot_images[image_id];
        if (True(image.flags & ImageFlagBits::Sparse)) {
            return false;
        }
        const u64 staging_bytes = image.unswizzled_size_bytes;
        if (collect_downloads.size() >= MaxQueued ||
            collect_download_staging + staging_bytes > MaxStaging) {
            ++collect_stats.fallback_cap;
            return false;
        }
        // Deferred: the pool must not hand the buffer to anyone else until it is freed here.
        auto map = runtime.DownloadStagingBuffer(staging_bytes, true);
        const auto copies = FullDownloadCopies(image.info);
        image.DownloadMemory(map, copies);
        const u64 memory_bytes = (image.HasScaled() ? GetScaledImageSizeBytes(image) : 0) +
                                 ImageMemoryBytes(image);
        collect_downloads.push_back(CollectDownload{
            .image_id = image_id,
            .map = map,
            .copies = copies,
            .tick = runtime.CurrentTick(),
            .frame = frame_tick,
            .lru_tick = lru_cache.Tick(image.lru_index),
            .image_modification = image.modification_tick,
            .cache_modification = modification_tick,
            .staging_bytes = staging_bytes,
            .memory_bytes = memory_bytes,
            .cpu_modified = True(image.flags & ImageFlagBits::CpuModified),
            .lost_path = 0,
        });
        image.flags |= ImageFlagBits::CollectPending;
        collect_download_staging += staging_bytes;
        collect_download_memory += memory_bytes;
        total_used_memory -= (std::min)(total_used_memory, memory_bytes);
        ++collect_stats.made;
        return true;
    } else {
        (void)image_id;
        return false;
    }
}

template <class P>
void TextureCache<P>::DropCollectDownload(size_t index) {
    CollectDownload& entry = collect_downloads[index];
    slot_images[entry.image_id].flags &= ~ImageFlagBits::CollectPending;
    // The image stays: its memory counts again (TickFrame leaves out the rest still queued).
    total_used_memory += entry.memory_bytes;
    collect_download_staging -= entry.staging_bytes;
    collect_download_memory -= entry.memory_bytes;
    runtime.FreeDeferredStagingBuffer(entry.map);
    collect_downloads.erase(collect_downloads.begin() + static_cast<std::ptrdiff_t>(index));
}

template <class P>
void TextureCache<P>::DropCollectDownloadsIn(DAddr cpu_addr, size_t size) {
    // Newer bytes went to guest memory: the queued, older ones must not land over them.
    for (size_t i = collect_downloads.size(); i-- > 0;) {
        const ImageBase& image = slot_images[collect_downloads[i].image_id];
        if (image.cpu_addr < cpu_addr + size && cpu_addr < image.cpu_addr + image.guest_size_bytes) {
            DropCollectDownload(i);
            ++collect_stats.dropped_written;
        }
    }
}

template <class P>
void TextureCache<P>::LandCollectDownloads() {
    const Common::WriteRing::SiteScope write_site{"EXP121 landing (LandCollectDownloads)", true};
    if constexpr (requires { runtime.CurrentTick(); }) {
        const DeleteReasonScope delete_reason_scope{*this, DeleteReason::GarbageCollect};
        for (size_t i = 0; i < collect_downloads.size();) {
            CollectDownload& entry = collect_downloads[i];
            Image& image = slot_images[entry.image_id];
            // Used again (bound, drawn to, copied; registered again): keep it, as if never picked.
            if (False(image.flags & ImageFlagBits::Registered) ||
                lru_cache.Tick(image.lru_index) != entry.lru_tick) {
                DropCollectDownload(i);
                ++collect_stats.dropped_used;
                continue;
            }
            // Written again by the GPU or the CPU, or downloaded by a guest read.
            bool written = !IsDownloadable(image) ||
                           image.modification_tick != entry.image_modification;
            // A CPU write since it was queued (WriteMemory marks it CpuModified; nothing drops
            // the entry there): landing would put the GPU's older bytes over the CPU's.
            if (!written && !entry.cpu_modified && True(image.flags & ImageFlagBits::CpuModified)) {
                written = true;
                ++collect_stats.dropped_cpu;
            }
            bool overlap_written = false;
            bool landed_under = false;
            if (!written && runtime.IsFree(entry.tick)) {
                // An overlapping image the GPU wrote since holds newer bytes for this memory.
                // FRITCHII_COLLECT_LAND_UNDER: unless it is still GPU-only. Its bytes are not in
                // guest memory, its own download lands over ours later, and one that lands first
                // drops ours (DropCollectDownloadsIn), so landing ours is what the waiting path
                // does. Dropped, the image stayed and the next pass queued it again.
                // FRITCHII_LOST_WRITEBACK's entries always land under such overlaps.
                const bool land_under = entry.lost_path != 0 ||
                                        VideoCore::Nx2Debug::SwitchValue(
                                            VideoCore::Nx2Debug::g_collect_land_under,
                                            "FRITCHII_COLLECT_LAND_UNDER") > 0;
                const ImageId self = entry.image_id;
                const u64 since = entry.cache_modification;
                ForEachImageInRegion(image.cpu_addr, image.guest_size_bytes,
                                     [&](ImageId other_id, ImageBase& other) {
                                         if (other_id == self || other.modification_tick <= since) {
                                             return;
                                         }
                                         if (land_under &&
                                             True(other.flags & ImageFlagBits::GpuModified)) {
                                             landed_under = true;
                                             return;
                                         }
                                         overlap_written = true;
                                     });
            }
            // FRITCHII_LOST_WRITEBACK: a newer write-back landed over some of its bytes since.
            const bool newer_landed = entry.lost_path != 0 && !written && !overlap_written &&
                                      runtime.IsFree(entry.tick) && NewerWritebackOver(image);
            if (overlap_written || newer_landed) {
                // FRITCHII_COVERED_RETIRE: newer GPU writes cover all of it, so nothing of it is
                // left to land: deleted now instead of kept for the next pass to queue again.
                if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_covered_retire,
                                                     "FRITCHII_COVERED_RETIRE") > 0 &&
                    CoveredByNewerWrites(entry.image_id, image)) {
                    const ImageId image_id = entry.image_id;
                    DropCollectDownload(i);
                    ++collect_stats.dropped_written;
                    ++collect_stats.dropped_under;
                    ++covered_stats.landing_deleted;
                    RetireCovered(image_id, image);
                    continue;
                }
                written = true;
                ++collect_stats.dropped_under;
                if (entry.lost_path != 0) {
                    // FRITCHII_LOST_WRITEBACK: newer bytes in its memory now; deleted without the
                    // write-back, as before (queued again, it would download every pass).
                    const ImageId image_id = entry.image_id;
                    ++lost_stats.wb_refused[entry.lost_path];
                    DropCollectDownload(i);
                    ++collect_stats.dropped_written;
                    image.lost_drop_why = LostWhyLanding;
                    MarkStale(image); // before UntrackImage (see MarkStale)
                    if (True(image.flags & ImageFlagBits::Tracked)) {
                        UntrackImage(image, image_id);
                    }
                    ++gc_deletes;
                    UnregisterImage(image_id);
                    DeleteImage(image_id, image.scale_tick > frame_tick + 5);
                    continue;
                }
            }
            if (written) {
                DropCollectDownload(i);
                ++collect_stats.dropped_written;
                continue;
            }
            if (!runtime.IsFree(entry.tick)) {
                ++i;
                continue;
            }
            // EXP121=2: compare with a download that waits (nothing touched the image since).
            if (VideoCore::Nx2Debug::Experiment(121) == 2) {
                auto fresh = runtime.DownloadStagingBuffer(entry.staging_bytes);
                image.DownloadMemory(fresh, entry.copies);
                runtime.Finish();
                if (std::memcmp(fresh.mapped_span.data(), entry.map.mapped_span.data(),
                                entry.staging_bytes) != 0) {
                    if (++collect_stats.mismatches <= 8) {
                        LOG_WARNING(HW_GPU,
                                    "Collector download differs from a waited one: {:#x} {}x{} "
                                    "format {}, {} frames after it was recorded",
                                    image.gpu_addr, image.info.size.width,
                                    image.info.size.height, static_cast<u32>(image.info.format),
                                    frame_tick - entry.frame);
                    }
                }
            }
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, entry.copies,
                         entry.map.mapped_span, swizzle_data_buffer);
            ++collect_stats.landed;
            NoteWriteback(image); // FRITCHII_LOST_WRITEBACK's order check
            if (entry.lost_path != 0) {
                ++lost_stats.wb_landed[entry.lost_path];
                lost_stats.wb_landed_bytes[entry.lost_path] += image.guest_size_bytes;
            }
            collect_stats.landed_under += landed_under ? 1 : 0;
            collect_stats.land_frames += frame_tick - entry.frame;
            unmap_stats.downloads += image.unmapped_tick != 0 ? 1 : 0;
            const ImageId image_id = entry.image_id;
            DropCollectDownload(i);
            // As the waiting path, a few frames later.
            image.flags &= ~ImageFlagBits::GpuModified;
            ClearStaleImage(image);
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, image_id);
            }
            ++(True(image.flags & ImageFlagBits::CostlyLoad) ? gc_costly_deletes : gc_deletes);
            UnregisterImage(image_id);
            DeleteImage(image_id, image.scale_tick > frame_tick + 5);
        }
    }
}

template <class P>
void TextureCache<P>::UpdateStaleTracking(u64 run_begin, u64 run_pages, s32 delta) {
    if (run_pages != 0) {
        device_memory.UpdatePagesCachedCount(run_begin << STALE_PAGE_BITS,
                                             run_pages << STALE_PAGE_BITS, delta);
    }
}

template <class P>
void TextureCache<P>::MarkStale(const ImageBase& image) {
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    const bool keep = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_lost_keep,
                                                       "FRITCHII_LOST_KEEP", 1) > 0;
    // FRITCHII_LOST_KEEP: the CPU wrote it since the GPU did (all of it dirty: its next use would
    // upload the guest bytes over the GPU's). Nothing GPU-only is lost; marked, the CPU's bytes
    // were skipped as stale and the next image there showed zeros.
    if (keep && True(image.flags & ImageFlagBits::CpuModified) && image.dirty_layers.empty() &&
        image.dirty_ranges.empty()) {
        ++lost_stats.stale_cpu_skipped;
        return;
    }
    RecordStaleDrop(image); // the lost contents counter's attribution
    if (!TcSkipStale() || True(image.flags & (ImageFlagBits::Sparse | ImageFlagBits::Unbacked)) ||
        image.guest_size_bytes == 0) {
        return;
    }
    // Whole pages only: a page the image shares with other data keeps that data's state.
    constexpr u64 page_size = u64{1} << STALE_PAGE_BITS;
    const u64 first = (image.cpu_addr + page_size - 1) >> STALE_PAGE_BITS;
    const u64 end = (image.cpu_addr + image.guest_size_bytes) >> STALE_PAGE_BITS;
    if (end <= first || end > (u64{1} << 28)) {
        return;
    }
    if (stale_bits.size() * 64 < end) {
        stale_bits.resize((end + 63) / 64);
    }
    // FRITCHII_LOST_KEEP: pages a newer image's write-back landed on hold bytes newer than this
    // image's (the refused write-backs: OoT draws HUD masks of several sizes at one address, the
    // newest landed first, the older one's drop marked it all stale and the next one there was
    // zero). Left unmarked, as pages shared with other data are.
    boost::container::small_vector<std::pair<u64, u64>, 8> newer_pages;
    if (keep && recent_writeback_newest > image.modification_tick) {
        for (const RecentWriteback& w : recent_writebacks) {
            if (w.size != 0 && w.modification > image.modification_tick &&
                w.cpu_addr < image.cpu_addr + image.guest_size_bytes &&
                image.cpu_addr < w.cpu_addr + w.size) {
                newer_pages.emplace_back(w.cpu_addr >> STALE_PAGE_BITS,
                                         (w.cpu_addr + w.size + page_size - 1) >> STALE_PAGE_BITS);
            }
        }
    }
    const auto newer = [&newer_pages](u64 page) {
        return std::ranges::any_of(newer_pages, [page](const std::pair<u64, u64>& range) {
            return range.first <= page && page < range.second;
        });
    };
    // Tracked from here on so CPU writes reach WriteMemory. Marking before the image's own
    // tracking goes keeps the pages' write protection from switching off and on again.
    u64 run_begin = 0;
    u64 run_pages = 0;
    for (u64 page = first; page < end; ++page) {
        u64& word = stale_bits[page / 64];
        const u64 bit = u64{1} << (page % 64);
        if (!newer_pages.empty() && (word & bit) == 0 && newer(page)) {
            ++lost_stats.stale_pages_kept;
            UpdateStaleTracking(run_begin, run_pages, 1);
            run_pages = 0;
            continue;
        }
        if ((word & bit) != 0) {
            UpdateStaleTracking(run_begin, run_pages, 1);
            run_pages = 0;
            continue;
        }
        word |= bit;
        ++stale_page_count;
        if (run_pages == 0) {
            run_begin = page;
        }
        ++run_pages;
    }
    UpdateStaleTracking(run_begin, run_pages, 1);
}

template <class P>
void TextureCache<P>::ClearStale(DAddr addr, size_t size) {
    if (stale_page_count == 0 || size == 0) {
        return;
    }
    constexpr u64 page_size = u64{1} << STALE_PAGE_BITS;
    const u64 first = addr >> STALE_PAGE_BITS;
    const u64 end = (std::min<u64>)((addr + size + page_size - 1) >> STALE_PAGE_BITS,
                                    stale_bits.size() * 64);
    u64 run_begin = 0;
    u64 run_pages = 0;
    for (u64 page = first; page < end;) {
        u64& word = stale_bits[page / 64];
        if (word == 0 && page % 64 == 0) {
            UpdateStaleTracking(run_begin, run_pages, -1);
            run_pages = 0;
            page += 64;
            continue;
        }
        const u64 bit = u64{1} << (page % 64);
        if ((word & bit) != 0) {
            word &= ~bit;
            --stale_page_count;
            if (run_pages == 0) {
                run_begin = page;
            }
            ++run_pages;
        } else {
            UpdateStaleTracking(run_begin, run_pages, -1);
            run_pages = 0;
        }
        ++page;
    }
    UpdateStaleTracking(run_begin, run_pages, -1);
}

template <class P>
void TextureCache<P>::ClearStaleImage(const ImageBase& image) {
    if (stale_page_count == 0) {
        return;
    }
    if (False(image.flags & ImageFlagBits::Sparse)) {
        ClearStale(image.cpu_addr, image.guest_size_bytes);
        return;
    }
    ClearStaleGpu(image.gpu_addr, image.guest_size_bytes);
}

template <class P>
void TextureCache<P>::ClearStaleGpu(GPUVAddr gpu_addr, size_t size) {
    if (stale_page_count == 0) {
        return;
    }
    // Each piece is contiguous in device memory (GetSubmappedRange splits where it is not).
    for (const auto& [piece_addr, piece_size] : gpu_memory->GetSubmappedRange(gpu_addr, size)) {
        if (const auto device_addr = gpu_memory->GpuToCpuAddress(piece_addr)) {
            ClearStale(*device_addr, piece_size);
        }
    }
}

template <class P>
bool TextureCache<P>::IsAllStale(DAddr addr, size_t size) const {
    if (stale_page_count == 0 || size == 0) {
        return false;
    }
    constexpr u64 page_size = u64{1} << STALE_PAGE_BITS;
    const u64 first = addr >> STALE_PAGE_BITS;
    const u64 end = (addr + size + page_size - 1) >> STALE_PAGE_BITS;
    if (end > stale_bits.size() * 64) {
        return false;
    }
    for (u64 page = first; page < end;) {
        const u64 word = stale_bits[page / 64];
        if (page % 64 == 0 && page + 64 <= end) {
            if (word != ~u64{0}) {
                return false;
            }
            page += 64;
            continue;
        }
        if (((word >> (page % 64)) & 1) == 0) {
            return false;
        }
        ++page;
    }
    return true;
}

template <class P>
bool TextureCache<P>::IsAnyStale(DAddr addr, size_t size) const {
    if (stale_page_count == 0 || size == 0) {
        return false;
    }
    const u64 first = addr >> STALE_PAGE_BITS;
    const u64 end = (std::min<u64>)((addr + size + (u64{1} << STALE_PAGE_BITS) - 1) >>
                                        STALE_PAGE_BITS,
                                    stale_bits.size() * 64);
    for (u64 page = first; page < end;) {
        const u64 word = stale_bits[page / 64];
        if (page % 64 == 0 && page + 64 <= end) {
            if (word != 0) {
                return true;
            }
            page += 64;
            continue;
        }
        if (((word >> (page % 64)) & 1) != 0) {
            return true;
        }
        ++page;
    }
    return false;
}

template <class P>
void TextureCache<P>::RecordStaleDrop(const ImageBase& image) {
    if (image.guest_size_bytes == 0 || True(image.flags & ImageFlagBits::Sparse)) {
        return;
    }
    // DeleteImage marks again what its caller just marked: one entry per drop.
    if (stale_drop_count != 0) {
        const StaleDrop& last = stale_drops[(stale_drop_next + StaleDropRing - 1) % StaleDropRing];
        if (last.cpu_addr == image.cpu_addr && last.size == image.guest_size_bytes &&
            last.frame == frame_tick) {
            return;
        }
    }
    const u64 shape_key = RelocationShapeKey(image.info);
    stale_drops[stale_drop_next] = StaleDrop{
        .cpu_addr = image.cpu_addr,
        .size = image.guest_size_bytes,
        .gpu_addr = image.gpu_addr,
        .frame = frame_tick,
        .format = static_cast<u32>(image.info.format),
        .width = image.info.size.width,
        .height = image.info.size.height,
        .reason = marking_relocation ? LostReasonRelocate : static_cast<u8>(delete_reason),
        .guard = image.relocation_guard_noted,
        .why = image.lost_drop_why,
        .shape_key = shape_key,
        .known = static_cast<u8>(
            !cold_resampled.empty() &&
            cold_resampled.contains(ColdResampledKey(image.gpu_addr, shape_key))),
        .counted = 0,
    };
    stale_drop_next = (stale_drop_next + 1) % StaleDropRing;
    stale_drop_count = (std::min)(stale_drop_count + 1, StaleDropRing);
}

template <class P>
auto TextureCache<P>::FindStaleDrop(DAddr addr, size_t size) -> StaleDrop* {
    // A join copied its overlaps into the new image before deleting them: another drop there,
    // if any, is the one that lost data.
    StaleDrop* join = nullptr;
    for (size_t i = 0; i < stale_drop_count; ++i) {
        StaleDrop& drop =
            stale_drops[(stale_drop_next + StaleDropRing - 1 - i) % StaleDropRing];
        if (drop.cpu_addr < addr + size && addr < drop.cpu_addr + drop.size) {
            if (drop.reason != static_cast<u8>(DeleteReason::Join)) {
                return &drop;
            }
            join = join != nullptr ? join : &drop;
        }
    }
    return join;
}

template <class P>
void TextureCache<P>::NoteLostContents(ImageBase& image, bool is_modification, bool invalidate) {
    LostContentStats& s = lost_stats;
    if (invalidate) {
        // Overwritten whole (a full clear, a copy over all of it): nothing lost shows.
        ++s.overwritten;
        image.lost_contents = 0;
        return;
    }
    if (is_modification) {
        image.lost_contents |= LostDrawn;
        return;
    }
    const u8 bits = image.lost_contents;
    image.lost_contents = 0; // counted once
    ++((bits & LostDrawn) != 0 ? s.sampled_drawn : s.sampled_undrawn);
    s.sampled_bytes += image.guest_size_bytes;
    StaleDrop* const drop = FindStaleDrop(image.cpu_addr, image.guest_size_bytes);
    const size_t reason = drop == nullptr                         ? 9
                          : drop->reason == LostReasonRelocate     ? 8
                          : (std::min<size_t>)(drop->reason, 7);
    ++s.sampled_by_reason[reason];
    // FRITCHII_COLD_SWEEP_KEEP_RESAMPLED: the game reads this memory again after the cold sweep
    // (covered retire included) dropped it: the dropped target's address and shape, and this
    // image's, are kept by the sweep from now on (while memory allows). Each drop counts once
    // (later images over it are repeats); "known" is whether its key was in the set when deleted.
    const int cold_keep_mode = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_cold_sweep_keep_resampled, "FRITCHII_COLD_SWEEP_KEEP_RESAMPLED", 1);
    ++cold_keep_stats.lost_all;
    cold_keep_stats.lost_not_recorded += drop == nullptr ? 1 : 0;
    if (drop != nullptr && drop->reason == static_cast<u8>(DeleteReason::ColdSweep)) {
        if (drop->counted != 0) {
            ++cold_keep_stats.lost_cold_repeat;
        } else {
            drop->counted = 1;
            ++cold_keep_stats.lost_cold;
            cold_keep_stats.lost_cold_known += drop->known;
        }
        if (cold_keep_mode > 0) {
            LearnColdResampled(drop->gpu_addr, drop->shape_key);
            if (RelocationShaped(image.info)) {
                LearnColdResampled(image.gpu_addr, RelocationShapeKey(image.info));
            }
        }
    }
    if (drop != nullptr && (drop->guard & (GuardSampled | GuardLoaded)) != 0) {
        ++s.sampled_guarded;
    }
    // A relocation moved the image away while its old address was still read: that shape and
    // address never relocate again (as the sampled guard does when it sees the read in time).
    if (drop != nullptr && reason == 8 &&
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_lost_writeback,
                                         "FRITCHII_LOST_WRITEBACK", 1024) > 0) {
        relocation_guarded_addresses[drop->gpu_addr] |= GuardSampled;
        relocation_shapes[drop->shape_key].guard |= GuardSampled;
        ++s.relocation_guards;
        if (RelocationMemory::Enabled()) {
            RelocationMemory::Note(drop->shape_key, GuardSampled);
        }
    }
    const u8 why = drop != nullptr
                       ? (std::min)(drop->why, static_cast<u8>(LostWhyCount - 1))
                       : static_cast<u8>(LostWhyNone);
    ++s.sampled_by_why[why];
    // FRITCHII_LOST_KEEP: this memory is read after going cold; the next drop over it is written
    // back whatever its size (relocation has its guards instead). FRITCHII_COLD_SWEEP_KEEP_RESAMPLED=3
    // keeps GPU-written targets over these ranges from the sweep, so it learns them too.
    if (reason != 8 && (cold_keep_mode >= 3 ||
                        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_lost_keep,
                                                         "FRITCHII_LOST_KEEP", 1) > 0)) {
        LearnLostRead(image.cpu_addr, image.guest_size_bytes);
    }
    // The first 40 of the session, then 2 per 10 s window (the flashes a tester reports come
    // late in long sessions).
    if (s.logged < 40 || s.window_logged < 2) {
        ++s.logged;
        ++s.window_logged;
        static constexpr std::array<const char*, 10> names{
            "other",       "garbage collect", "cold sweep",       "unmap",      "join",
            "slot retire", "unmap sweep",     "superseded sweep", "relocation", "not recorded"};
        static constexpr std::array<const char*, LostWhyCount> whys{
            "no write-back on that path", "over the size cap", "refused (newer write-back)",
            "not downloadable", "write-back off", "over the keep budget", "refused at landing"};
        LOG_WARNING(HW_GPU,
                    "Lost contents sampled: format {} {}x{} levels {} at {:#x} ({} KiB), {}{}{}; "
                    "the memory's GPU data was dropped by: {} (format {} {}x{} at {:#x}, {} "
                    "frames ago, guard {:#x}, {}{})",
                    static_cast<u32>(image.info.format), image.info.size.width,
                    image.info.size.height, image.info.resources.levels, image.gpu_addr,
                    image.guest_size_bytes >> 10,
                    (bits & LostSkipped) != 0 ? "upload skipped as stale" : "stale guest bytes uploaded",
                    (bits & LostZeroed) != 0 ? ", cleared to zero" : "",
                    (bits & LostDrawn) != 0 ? ", drawn into without a full clear" : ", not drawn into",
                    names[reason], drop ? drop->format : 0u, drop ? drop->width : 0u,
                    drop ? drop->height : 0u, drop ? drop->gpu_addr : GPUVAddr{0},
                    drop ? frame_tick - drop->frame : 0, drop ? drop->guard : u8{0}, whys[why],
                    drop != nullptr && drop->known != 0 ? ", key in set at drop" : "");
    }
}

template <class P>
bool TextureCache<P>::OverlapsLostRead(DAddr addr, u64 size) const {
    for (size_t i = 0; i < lost_read_count; ++i) {
        const LostReadRange& range = lost_read_ranges[i];
        if (range.begin < addr + size && addr < range.end) {
            return true;
        }
    }
    return false;
}

template <class P>
void TextureCache<P>::LearnLostRead(DAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    for (size_t i = 0; i < lost_read_count; ++i) {
        const LostReadRange& range = lost_read_ranges[i];
        if (range.begin <= addr && addr + size <= range.end) {
            return; // known
        }
    }
    lost_read_ranges[lost_read_next] = LostReadRange{addr, addr + size};
    lost_read_next = (lost_read_next + 1) % LostReadRing;
    lost_read_count = (std::min)(lost_read_count + 1, LostReadRing);
    ++lost_stats.learned_new;
}

template <class P>
bool TextureCache<P>::TakeLostKeepBudget(u64 bytes, bool learned, int keep) {
    // N MiB a second (1 = 32), up to a second's worth saved. 32 MiB/s is ~1/20 of what
    // FRITCHII_LOST_WRITEBACK=1 wrote back in the market (-3 fps there), and only memory with a
    // reason is in it.
    const u64 rate = u64{keep > 1 ? (std::min)(static_cast<u32>(keep), 4096U) : 32U} << 20;
    const u64 now = Common::Nx2Hitch::NowNs();
    if (lost_keep_refill_ns == 0) {
        lost_keep_tokens = rate;
        lost_keep_refill_ns = now;
    } else if (now > lost_keep_refill_ns) {
        // At most 2^32 * 10^9 < 2^62.
        const u64 elapsed = (std::min<u64>)(now - lost_keep_refill_ns, 1'000'000'000);
        const u64 refill = rate * elapsed / 1'000'000'000;
        if (refill != 0) {
            lost_keep_tokens = (std::min)(rate, lost_keep_tokens + refill);
            lost_keep_refill_ns = now; // else the remainder keeps adding up
        }
    }
    // Guarded images (read in a later frame, not yet seen lost) leave half to learned memory.
    const u64 floor = learned ? 0 : rate / 2;
    if (lost_keep_tokens < floor + bytes) {
        return false;
    }
    lost_keep_tokens -= bytes;
    return true;
}

template <class P>
void TextureCache<P>::LogLostContents() {
    LostContentStats& s = lost_stats;
    const u64 now = Common::Nx2Hitch::NowNs();
    if (s.last_log_ns == 0) {
        s.last_log_ns = now;
        return;
    }
    if (now - s.last_log_ns < 10'000'000'000) {
        return;
    }
    s.last_log_ns = now;
    const u64 sampled = s.sampled_undrawn + s.sampled_drawn;
    u64 wb_any = 0;
    for (size_t i = 0; i < LostPathCount; ++i) {
        wb_any += s.wb_queued[i] + s.wb_landed[i] + s.wb_refused[i] + s.wb_kept[i] +
                  s.wb_over_cap[i];
    }
    if (s.made_skipped + s.made_uploaded + sampled + wb_any + s.stale_cpu_skipped +
            s.stale_pages_kept + s.wb_cpu_written !=
        0) {
        const auto& r = s.sampled_by_reason;
        // queued/landed (MiB) refused R kept K over cap C, per path.
        std::string wb;
        static constexpr std::array<const char*, LostPathCount> paths{
            "", "collector", "cold sweep", "superseded sweep", "slot retire"};
        for (size_t i = 1; i < LostPathCount; ++i) {
            wb += fmt::format("{}{} {}/{} ({:.1f} MiB) refused {} kept {} over cap {} (written "
                              "back all the same: learned {}, guarded {}; over budget {})",
                              i == 1 ? "" : "; ", paths[i], s.wb_queued[i], s.wb_landed[i],
                              static_cast<double>(s.wb_landed_bytes[i]) / (1024.0 * 1024.0),
                              s.wb_refused[i], s.wb_kept[i], s.wb_over_cap[i], s.wb_learned[i],
                              s.wb_guarded[i], s.wb_over_budget[i]);
        }
        // FRITCHII_LOST_KEEP: why the sampled ones' drops had no write-back (over cap, refused
        // and over budget are the ones it works on: they must stay near 0 with it on).
        const auto& y = s.sampled_by_why;
        wb += fmt::format(
            "; FRITCHII_LOST_KEEP={}: sampled ones' drops without a write-back because: no "
            "write-back on that path {}, over cap {}, refused {}, not downloadable {}, off {}, "
            "over budget {}, refused at landing {}; learned ranges {} (+{}), pages kept from "
            "stale under newer write-backs {}, CPU-written images not marked stale {} (not "
            "written back {})",
            VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_lost_keep,
                                             "FRITCHII_LOST_KEEP", 1),
            y[LostWhyNone], y[LostWhyOverCap], y[LostWhyRefused], y[LostWhyNoCopy], y[LostWhyOff],
            y[LostWhyBudget], y[LostWhyLanding], lost_read_count, s.learned_new,
            s.stale_pages_kept, s.stale_cpu_skipped, s.wb_cpu_written);
        LOG_INFO(HW_GPU,
                 "Lost contents in 10 s (FRITCHII_STALE_ZERO={}, FRITCHII_LOST_WRITEBACK={}): "
                 "images made over memory whose GPU data was dropped: {} upload skipped ({} "
                 "cleared to zero), {} uploaded stale guest bytes; {} overwritten whole first; "
                 "SAMPLED with lost contents: {} ({} not drawn into, {} drawn over; {} KiB; {} "
                 "were read in a later frame before the drop) by drop: collector {}, cold sweep "
                 "{}, superseded sweep {}, slot retire {}, join {}, relocation {}, unmap {}/{}, "
                 "other {}, not recorded {}; relocation guards {}; write-backs queued/landed: {}",
                 VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_stale_zero,
                                                  "FRITCHII_STALE_ZERO", 1),
                 VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_lost_writeback,
                                                  "FRITCHII_LOST_WRITEBACK", 1024),
                 s.made_skipped, s.made_zeroed, s.made_uploaded, s.overwritten, sampled,
                 s.sampled_undrawn, s.sampled_drawn, s.sampled_bytes >> 10, s.sampled_guarded,
                 r[1], r[2], r[7], r[5], r[4], r[8], r[3], r[6], r[0], r[9], s.relocation_guards,
                 wb);
    }
    const u64 last = s.last_log_ns;
    const u64 logged = s.logged;
    s = {};
    s.last_log_ns = last;
    s.logged = logged;
}

template <class P>
auto TextureCache<P>::QueueLostWriteback(ImageId image_id, Image& image, u8 path)
    -> LostWriteback {
    // 1 (default): every image; N > 1: only images up to N KiB (larger ones dropped as before).
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_lost_writeback,
                                                      "FRITCHII_LOST_WRITEBACK", 1024);
    const int keep = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_lost_keep,
                                                      "FRITCHII_LOST_KEEP", 1);
    if (False(image.flags & ImageFlagBits::GpuModified) ||
        True(image.flags & ImageFlagBits::CollectPending)) {
        return LostWriteback::Drop;
    }
    LostContentStats& s = lost_stats;
    // FRITCHII_LOST_KEEP: the CPU wrote all of it since the GPU did: the guest bytes are the
    // newer ones (MarkStale leaves them unmarked), and a write-back would land the GPU's older
    // bytes over them.
    if (keep > 0 && True(image.flags & ImageFlagBits::CpuModified) &&
        image.dirty_layers.empty() && image.dirty_ranges.empty()) {
        ++s.wb_cpu_written;
        return LostWriteback::Drop;
    }
    if (mode <= 0) {
        image.lost_drop_why = LostWhyOff;
        return LostWriteback::Drop;
    }
    if (!IsDownloadable(image)) {
        image.lost_drop_why = LostWhyNoCopy;
        return LostWriteback::Drop;
    }
    if constexpr (!requires { runtime.CurrentTick(); }) {
        image.lost_drop_why = LostWhyOff;
        return LostWriteback::Drop; // no queue (OpenGL)
    } else {
        // Bytes a newer image already wrote back: landing this one would put older bytes there.
        // (Before the size cap: a refused image costs no budget.)
        const bool over_cap =
            mode > 1 && image.guest_size_bytes > u64{static_cast<u32>(mode)} * 1024;
        if (!over_cap || keep > 0) {
            if (NewerWritebackOver(image)) {
                ++s.wb_refused[path];
                image.lost_drop_why = LostWhyRefused;
                return LostWriteback::Drop;
            }
        }
        // FRITCHII_LOST_KEEP: over the cap, but memory read after an earlier loss there
        // (learned) or an image read in a later frame than its write (guarded) is written back
        // all the same, within the byte budget.
        u64 budget_taken = 0;
        if (over_cap) {
            ++s.wb_over_cap[path];
            const bool learned =
                keep > 0 && OverlapsLostRead(image.cpu_addr, image.guest_size_bytes);
            const bool guarded =
                keep > 0 && (image.relocation_guard_noted &
                             (GuardSampled | GuardLoaded | GuardCpuRead)) != 0;
            if (!learned && !guarded) {
                image.lost_drop_why = LostWhyOverCap;
                return LostWriteback::Drop;
            }
            if (!TakeLostKeepBudget(image.guest_size_bytes, learned, keep)) {
                ++s.wb_over_budget[path];
                image.lost_drop_why = LostWhyBudget;
                return LostWriteback::Drop;
            }
            budget_taken = image.guest_size_bytes;
            ++(learned ? s.wb_learned[path] : s.wb_guarded[path]);
        }
        if (!QueueCollectDownload(image_id)) {
            lost_keep_tokens += budget_taken; // not spent
            ++s.wb_kept[path];
            return LostWriteback::Keep;
        }
        collect_downloads.back().lost_path = path;
        ++s.wb_queued[path];
        return LostWriteback::Queued;
    }
}

template <class P>
void TextureCache<P>::NoteWriteback(const ImageBase& image) {
    recent_writebacks[recent_writeback_next] = RecentWriteback{
        .cpu_addr = image.cpu_addr,
        .size = image.guest_size_bytes,
        .modification = image.modification_tick,
    };
    recent_writeback_next = (recent_writeback_next + 1) % RecentWritebackRing;
    // Only grows: an entry the ring dropped leaves it high, which costs MarkStale a scan.
    recent_writeback_newest = (std::max)(recent_writeback_newest, image.modification_tick);
}

template <class P>
bool TextureCache<P>::NewerWritebackOver(const ImageBase& image) const {
    for (const RecentWriteback& w : recent_writebacks) {
        if (w.size != 0 && w.modification > image.modification_tick &&
            w.cpu_addr < image.cpu_addr + image.guest_size_bytes &&
            image.cpu_addr < w.cpu_addr + w.size) {
            return true;
        }
    }
    return false;
}

template <class P>
u64 TextureCache<P>::UncountedDeviceBytes() const {
    // The same subtractions TickFrame makes from the driver's usage.
    u64 bytes = std::accumulate(sentenced_image_bytes.begin(), sentenced_image_bytes.end(),
                                gpu_sentenced_bytes) +
                collect_download_memory;
    if constexpr (requires { runtime.PendingImageFreeBytes(); }) {
        bytes += runtime.PendingImageFreeBytes();
    }
    if constexpr (requires { runtime.PooledImageBytes(); }) {
        bytes += runtime.PooledImageBytes();
    }
    return bytes;
}

template <class P>
void TextureCache<P>::ApplyThresholdOverrides(bool log) {
    // Past what a frame needs, more device memory only keeps the textures of earlier scenes
    // (title 1 filled 6.7 GB of an 8 GB card and draws its races in ~5 GB), so collect from
    // MaxCriticalMemory on rather than near the card's budget. Larger cards get 5/8 of the
    // driver's budget (GetDeviceLocalMemory is capped near 7 GiB): at 5 GiB title 1's races
    // recreate and upload ~8 MiB of render targets a frame (16 GB card: 53 -> 59 fps at 10 GiB).
    // 5/8 of the budget only passes 5 GiB above an 8 GiB budget, so cards up to 8 GB (and
    // integrated GPUs, whose budget is capped at 4-6 GiB) keep the 5 GiB cap.
    u64 max_critical = MaxCriticalMemory;
    if constexpr (requires { runtime.GetDeviceMemoryBudget(); }) {
        const u64 budget = runtime.GetDeviceMemoryBudget();
        max_critical = (std::max)(max_critical, budget / 8 * 5);
        // Cards with 10 GiB or more of budget: 5/8 left ~7.3 GiB on a 12 GB RX 6700 XT, under what
        // Ocarina's Kokiri Forest keeps alive, so every aggressive pass evicted its CostlyLoad ASTC
        // textures (a 6144x6144 5x5 one, 31 MiB) and the game re-created them ~128 frames later:
        // a 24 ms texture stall every ~3 s. Keep 2.5 GiB of the budget for everything else instead.
        if (budget >= 10_GiB) {
            max_critical = (std::max)(max_critical, budget - 2560_MiB);
        } else if (budget > 2_GiB && VideoCore::SmallCard::On(VideoCore::SmallCard::Fit)) {
            // FRITCHII_SMALL_CARD (part Fit): the budget check in TickFrame subtracts everything
            // the usage leaves out, so its result needs no fixed cap; the cap only kept textures
            // from the room the uncounted image pool and in-flight frees then took.
            max_critical = (std::max)(max_critical, budget - SmallCardMargin(budget));
        } else if (budget > 4_GiB &&
                   VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_vram_critical_fit,
                                                    "FRITCHII_VRAM_CRITICAL_FIT") != 0) {
            // FRITCHII_VRAM_CRITICAL_FIT (live, default 0; with FRITCHII_VRAM_POOL_FIT): an 8 GB
            // card (budget ~7.1 GiB with Windows and other programs) stopped at the fixed 5 GiB
            // and its last ~2 GiB of budget went to the image pool and margins while OoT's market
            // (~5.1-5.3 GiB in the cache's view on a 16 GB card) evicted and re-made textures.
            // Here the cap is the budget less the cache's margin (1/8 of it, at most 1 GiB) and
            // 512 MiB for deleted images still in flight; the fitted pool yields the rest.
            const u64 margin = (std::min)(budget / 8, u64{1_GiB});
            max_critical = (std::max)(max_critical, budget - margin - 512_MiB);
        }
    }
    critical_memory = (std::min)(critical_memory, max_critical);
    expected_memory =
        (std::min)(expected_memory, critical_memory - (std::min)(critical_memory, u64{768_MiB}));
    minimum_memory =
        (std::min)(minimum_memory, expected_memory - (std::min)(expected_memory, u64{1_GiB}));
    // EDEN_NX2_TC_CEILING_MIB: a ceiling for the critical threshold (expected 1.5 GiB under it),
    // which only ever lowers them, so cards with a smaller budget keep theirs. Fritchii builds it
    // in at 7680 (2026-11-10, owner's playtests: 16 GB card, VRAM 10 -> ~6 GB, 4-6 GB of RAM kept
    // free instead of ~1.5 since Windows charges device memory to commit, 58.8 fps over 30 min).
    if (const char* value = std::getenv("EDEN_NX2_TC_CEILING_MIB")) {
        const u64 ceiling = std::strtoull(value, nullptr, 10) << 20;
        if (ceiling >= 2_GiB) {
            critical_memory = (std::min)(critical_memory, ceiling);
            expected_memory = (std::min)(expected_memory, ceiling - 1536_MiB);
            minimum_memory = (std::min)(minimum_memory,
                                        expected_memory - (std::min)(expected_memory, u64{1_GiB}));
        }
    }
    // EDEN_NX2_TC_EXPECTED_MIB / EDEN_NX2_TC_CRITICAL_MIB override the collector's thresholds.
    for (const auto& [name, target] : {std::pair{"EDEN_NX2_TC_EXPECTED_MIB", &expected_memory},
                                       std::pair{"EDEN_NX2_TC_CRITICAL_MIB", &critical_memory}}) {
        if (const char* value = std::getenv(name)) {
            *target = std::strtoull(value, nullptr, 10) << 20;
        }
    }
    // FRITCHII_VRAM_BUDGET_MODE: these are the base thresholds; the live budget cap only ever
    // lowers them, and goes when the room under the budget is back.
    base_expected_memory = expected_memory;
    base_critical_memory = critical_memory;
    base_minimum_memory = minimum_memory;
    ApplyBudgetCap();
    if (log) {
        LOG_INFO(HW_GPU, "Texture cache thresholds: expected {} MiB, critical {} MiB, minimum {} MiB",
                 expected_memory >> 20, critical_memory >> 20, minimum_memory >> 20);
    }
}

template <class P>
void TextureCache<P>::ApplyBudgetCap() {
    if (base_critical_memory == 0) {
        return;
    }
    critical_memory = base_critical_memory;
    expected_memory = base_expected_memory;
    minimum_memory = base_minimum_memory;
    if (budget_cap >= critical_memory) {
        return;
    }
    // The cap keeps the base thresholds' distance (1.5 GiB with Fritchii's ceiling) between
    // expected and critical, at least 256 MiB and at most half the cap (or 192 MiB on small cards).
    const u64 raw_gap = base_critical_memory - (std::min)(base_critical_memory, base_expected_memory);
    const u64 max_gap = budget_cap < 3_GiB ? u64{192_MiB} : (budget_cap / 2);
    const u64 min_gap = budget_cap < 3_GiB ? u64{128_MiB} : u64{256_MiB};
    critical_memory = budget_cap;
    expected_memory = (std::min)(
        expected_memory,
        critical_memory - (std::min)(max_gap, (std::max)(raw_gap, min_gap)));
    minimum_memory =
        (std::min)(minimum_memory, expected_memory - (std::min)(expected_memory, u64{1_GiB}));
}

template <class P>
void TextureCache<P>::ApplySmallCardImageFloor(u64 budget, bool log) {
    // Small cards (< 5 GiB): leave images a floor over non-image VRAM so the
    // collector does not thrash downloading and re-transcoding active textures.
    static const std::optional<u64> forced_floor = []() -> std::optional<u64> {
        const char* value = std::getenv("EDEN_NX2_TC_IMAGE_FLOOR_MIB");
        return value != nullptr
                   ? std::optional<u64>{std::strtoull(value, nullptr, 10) << 20}
                   : std::nullopt;
    }();
    const u64 image_floor =
        forced_floor.value_or(budget < 5_GiB ? (std::min)(budget / 2, u64{1536_MiB}) : u64{0});
    if (image_floor == 0) {
        return;
    }
    u64 image_bytes{};
    for (const auto [image_id, image] : slot_images) {
        image_bytes += ImageMemoryBytes(*image);
    }
    const u64 other = total_used_memory - (std::min)(total_used_memory, image_bytes);
    const u64 floor_expected = other + image_floor;
    if (floor_expected > expected_memory) {
        if (log) {
            static bool image_floor_logged{};
            if (!image_floor_logged) {
                image_floor_logged = true;
                LOG_INFO(HW_GPU,
                         "Texture cache image floor: {} MiB in use is not images, "
                         "images get {} MiB (expected {} -> {} MiB)",
                         other >> 20, image_floor >> 20, expected_memory >> 20,
                         floor_expected >> 20);
            }
        }
        expected_memory = floor_expected;
        critical_memory = (std::max)(critical_memory, floor_expected + image_floor / 4);
    }
}

template <class P>
void TextureCache<P>::BudgetControl() {
    // FRITCHII_VRAM_BUDGET_MODE (live; 0 fixed, 1 vk = default, 2 os). The thresholds above follow
    // the budget only as far as Fritchii's ceiling (EDEN_NX2_TC_CEILING_MIB, 7680) and the 5/8 cap
    // allow, and they count the cache's view of the usage: the image pool, deleted images still in
    // flight, the unused space in the allocators' blocks and whatever the driver holds outside
    // them are left out. A 16 GB RX 9070 XT at 4K (zortos tester, 2026-11-16) kept the cache's
    // critical at 7,680 MiB while the driver's count for the process reached the budget, the
    // driver lowered the budget to it and moved ~10 GB to system RAM (freezes of 15-16 s; a 1365^2
    // image create took 11.9 s). With a 5-7 GiB hog on the card the budget fell to ~11.6 GB and
    // nothing followed it.
    //
    // Here the driver's own count of the main heap (all of this process's memory in it) is held
    // under the budget less a headroom: once a second, past it the cache's critical memory comes
    // down by the excess (at most 1 GiB a second, never under half the budget), with room again it
    // goes back up (256 MiB a second) until the base thresholds rule again. While the cap binds
    // (budget_pressure) the collector deletes past critical as with FRITCHII_TC_COLLECT_OVER, the
    // image pool fits the room (FRITCHII_VRAM_POOL_FIT), kept blocks go back, and images made past
    // critical first delete clean idle ones (FRITCHII_TC_CRITICAL_ENFORCE). Nothing changes while
    // the usage stays under the budget less the headroom: an RTX 4080 at 1080p sits ~7 GB under it.
    // os also counts what other processes hold on the adapter (PDH, sampled on a thread of its
    // own): Windows' budget for the process in front assumes it can push the others out, which a
    // program that keeps using its memory defeats (the tester's VRAM hog).
    if constexpr (requires(u64 value) { runtime.MainHeapNumbers(value, value, value); }) {
        const auto now = std::chrono::steady_clock::now();
        if (now - budget_control_time < std::chrono::seconds{1}) {
            return;
        }
        budget_control_time = now;
        const int mode = VideoCore::Nx2Debug::VramBudgetMode();
        u64 size{};
        u64 usage{};
        u64 vk_budget{};
        if (!runtime.MainHeapNumbers(size, usage, vk_budget) || vk_budget == 0) {
            return;
        }
        BudgetSeen& seen = budget_seen;
        seen.heap_size = size;
        seen.usage = usage;
        seen.vk_budget = vk_budget;
        seen.os_budget = 0;
        seen.others = 0;
        u64 budget = vk_budget;
        if (mode == 2) {
            if (const std::optional<u64> others = runtime.OtherProcessesVideoMemory()) {
                seen.others = *others;
                seen.os_budget = size - (std::min)(size, *others);
                budget = (std::min)(budget, seen.os_budget);
            }
        }
        const int headroom_mib = VideoCore::Nx2Debug::SwitchValue(
            VideoCore::Nx2Debug::g_vram_headroom_mib, "FRITCHII_VRAM_HEADROOM_MIB", 0);
        // Auto: 1/16 of the budget, 256 MiB to 1 GiB (~950 MiB on a 16 GB card, ~450 on 8 GB).
        const u64 headroom = headroom_mib > 0
                                 ? u64{static_cast<u32>(headroom_mib)} << 20
                                 : std::clamp(budget / 16, u64{256_MiB}, u64{1_GiB});
        const u64 target = budget - (std::min)(budget, headroom);
        seen.budget = budget;
        seen.headroom = headroom;
        seen.peak_usage = (std::max)(seen.peak_usage, usage);
        seen.lowest_budget = (std::min)(seen.lowest_budget, budget);
        seen.over_seconds += usage > target ? 1 : 0;
        seen.spill_seconds += usage > budget ? 1 : 0;
        if (mode != seen.mode) {
            LOG_INFO(HW_GPU,
                     "VRAM budget control: FRITCHII_VRAM_BUDGET_MODE={} (0 fixed, 1 vk, 2 os), "
                     "headroom {} MiB; main heap {} MiB, driver budget {} MiB, usage {} MiB",
                     mode, headroom >> 20, size >> 20, vk_budget >> 20, usage >> 20);
            seen.mode = mode;
        }
        const bool was_pressure = budget_pressure;
        const u64 old_cap = budget_cap;
        if (mode == 0) {
            budget_cap = ~0ULL;
        } else if (usage > target) {
            const u64 over = usage - target;
            const u64 from = (std::min)(budget_cap, total_used_memory);
            budget_cap = (std::max)(from - (std::min)(from, (std::min)(over, u64{1_GiB})),
                                    budget / 2);
        } else if (budget_cap != ~0ULL) {
            // A dead band of half the headroom, so the cap does not chase the usage up and down.
            const u64 room = target - usage;
            if (room > headroom / 2) {
                budget_cap += (std::min)(room - headroom / 2, u64{256_MiB});
            }
        }
        if (budget_cap >= base_critical_memory) {
            budget_cap = ~0ULL;
        }
        // The pressure also holds while the usage is past the target with the cap at its floor
        // or above the base (a 16 GB card: half the budget is over Fritchii's 7,680 MiB ceiling,
        // so the 4K session's spill came with the cache already past a critical it could not
        // hold): the collector's extra deletes, the fitted pool and the evictions before creates
        // then act on the base thresholds. It ends half a headroom under the target.
        const bool over_target = mode != 0 && usage > target;
        const bool well_under = usage + headroom / 2 <= target;
        budget_pressure =
            budget_cap != ~0ULL || over_target || (budget_pressure && mode != 0 && !well_under);
        if constexpr (requires { runtime.SetBudgetPressure(true); }) {
            runtime.SetBudgetPressure(budget_pressure);
        }
        ApplyBudgetCap();
        if (!budget_pressure && budget > 2_GiB) {
            ApplySmallCardImageFloor(budget, false);
        } else if (budget > 2_GiB && budget < 5_GiB && usage <= budget) {
            u64 live_image_bytes{};
            for (const auto [image_id, image] : slot_images) {
                if (False(image->flags & ImageFlagBits::Unbacked)) {
                    live_image_bytes += ImageMemoryBytes(*image);
                }
            }
            const u64 other = total_used_memory - (std::min)(total_used_memory, live_image_bytes);
            const u64 small_img_floor = (std::min)(budget / 4, u64{768_MiB});
            const u64 cap_expected = (std::min)(other + small_img_floor,
                                                base_expected_memory);
            if (expected_memory < cap_expected) {
                expected_memory = cap_expected;
                critical_memory = (std::max)(critical_memory, expected_memory + 192_MiB);
            }
        }
        const u64 change = budget_cap > old_cap ? budget_cap - old_cap : old_cap - budget_cap;
        if (budget_pressure != was_pressure ||
            (budget_cap != ~0ULL && old_cap != ~0ULL && change >= 256_MiB) ||
            ((budget_cap == ~0ULL) != (old_cap == ~0ULL))) {
            LOG_INFO(HW_GPU,
                     "VRAM budget control ({}): driver usage {} MiB of budget {} MiB (vk {} MiB, "
                     "os {} MiB, other processes {} MiB), headroom {} MiB: {}; texture cache "
                     "{} MiB, expected {} MiB, critical {} MiB (base {} MiB)",
                     mode == 2 ? "os" : (mode == 1 ? "vk" : "fixed"), usage >> 20, budget >> 20,
                     vk_budget >> 20, seen.os_budget >> 20, seen.others >> 20, headroom >> 20,
                     !budget_pressure ? "pressure off, cap off"
                     : budget_cap == ~0ULL
                         ? "pressure on, no cap under the base thresholds"
                         : (old_cap == ~0ULL ? "pressure on, cap on" : "pressure on, cap moved"),
                     total_used_memory >> 20, expected_memory >> 20, critical_memory >> 20,
                     base_critical_memory >> 20);
        }
    }
}

template <class P>
void TextureCache<P>::EnforceCriticalAtCreate(u64 incoming) {
    // FRITCHII_TC_CRITICAL_ENFORCE (live; 0 off, 1 = default: while the budget cap binds, 2: past
    // critical at any time). The collector runs once a frame and deletes a bounded number of
    // images; an area load makes dozens in one frame, each on top of a card already at its budget
    // (the 11.9 s image create above was the driver making room). Before such an image is made,
    // clean images (no GPU writes, so no download and no wait) idle for 10+ frames are deleted
    // until the cache is back under critical with room for it: at most 32 a pass and 256 a frame,
    // looking at most 512 of the oldest. Images used in the last 10 frames are never taken (what
    // the draw being set up prepared was touched in this frame), nor bound render targets.
    // Deleting here is what a join does in the middle of a lookup already (has_deleted_images:
    // every caller of InsertImage looks up again or gives up the operation).
    const int enforce = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_tc_critical_enforce, "FRITCHII_TC_CRITICAL_ENFORCE", 1);
    const bool small_card = Common::HostHasSmallVideoMemory();
    if (enforce <= 0 || (enforce == 1 && !budget_pressure && !small_card) ||
        total_used_memory + incoming <= critical_memory) [[likely]] {
        return;
    }
    static const u64 min_age = [] {
        const char* value = std::getenv("EDEN_NX2_TC_MIN_AGE");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 0ULL;
    }();
    const u64 idle_frames = (std::max)({u64{10}, min_age, thrash_floor});
    const u32 max_per_frame =
        (small_card && total_used_memory > critical_memory + 512_MiB) ? 1024U : 256U;
    constexpr size_t MaxPerPass = 32;
    constexpr size_t MaxLooked = 512;
    if (create_enforce_frame != frame_tick) {
        create_enforce_frame = frame_tick;
        create_enforce_frame_deletes = 0;
    }
    auto& stats = create_enforce_stats;
    if (create_enforce_frame_deletes >= max_per_frame) {
        ++stats.capped;
        return;
    }
    if (frame_tick <= idle_frames) {
        return;
    }
    // FRITCHII_TC_THRASH_GUARD: with the cache over critical for good, every image made walked
    // the idle list again for nothing (26,000 passes in 10 s, 0 images deleted).
    const bool skip_empty = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_tc_thrash_guard,
                                                             "FRITCHII_TC_THRASH_GUARD", 1) > 0;
    if (skip_empty && create_enforce_empty_frame == frame_tick) {
        return;
    }
    ++stats.passes;
    // Render targets bound now are kept whatever their age: RescaleRenderTargets binds a target it
    // found and keeps its id while it looks up the next one (which may get here), and touches
    // them only after.
    std::array<ImageId, NUM_RT + 1> bound{};
    const auto bound_image = [this](ImageViewId view_id) {
        return view_id != NULL_IMAGE_VIEW_ID && view_id != ImageViewId{}
                   ? slot_image_views[view_id].image_id
                   : ImageId{};
    };
    for (size_t index = 0; index < NUM_RT; ++index) {
        bound[index] = bound_image(render_targets.color_buffer_ids[index]);
    }
    bound[NUM_RT] = bound_image(render_targets.depth_buffer_id);
    auto& candidates = create_enforce_candidates;
    candidates.clear();
    const u64 emergency_age =
        (small_card && total_used_memory > critical_memory + 512_MiB) ? u64{4} : u64{10};
    const u64 min_idle = (std::max)(emergency_age, min_age);
    size_t looked = 0;
    const auto collect_candidates = [&](u64 floor_frames) {
        candidates.clear();
        looked = 0;
        if (frame_tick <= floor_frames) {
            return;
        }
        lru_cache.ForEachItemBelow(frame_tick - floor_frames, [&](ImageId image_id) {
            const Image& image = slot_images[image_id];
            if (False(image.flags & (ImageFlagBits::IsDecoding | ImageFlagBits::CollectPending |
                                     ImageFlagBits::GpuModified | ImageFlagBits::BadOverlap)) &&
                std::ranges::find(bound, image_id) == bound.end()) {
                candidates.push_back(image_id);
            }
            return candidates.size() >= MaxPerPass || ++looked >= MaxLooked;
        });
    };
    collect_candidates(idle_frames);
    if (candidates.empty() && idle_frames > min_idle) {
        // ThrashGuard's floor locked out evictions while VRAM is already past critical; fall back
        // to min_idle and relax thrash_floor so subsequent frames don't stall for 0 MiB freed.
        collect_candidates(min_idle);
        if (thrash_floor > min_idle) {
            thrash_floor /= 2;
        }
    }
    if (candidates.empty() && looked < MaxLooked) {
        create_enforce_empty_frame = frame_tick;
    }
    if (!candidates.empty()) {
        // Evict cheap-to-reload images before costly (e.g. ASTC) ones to avoid re-decode hitches.
        std::ranges::stable_partition(candidates, [&](ImageId id) {
            return False(slot_images[id].flags & ImageFlagBits::CostlyLoad);
        });
    }
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::GarbageCollect};
    for (const ImageId image_id : candidates) {
        if (total_used_memory + incoming <= critical_memory ||
            create_enforce_frame_deletes >= max_per_frame) {
            break;
        }
        auto& image = slot_images[image_id];
        const u64 before = total_used_memory;
        MarkStale(image); // before UntrackImage (see MarkStale)
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
        ++(True(image.flags & ImageFlagBits::CostlyLoad) ? gc_costly_deletes : gc_deletes);
        UnregisterImage(image_id);
        DeleteImage(image_id, image.scale_tick > frame_tick + 5);
        ++stats.deleted;
        ++create_enforce_frame_deletes;
        stats.freed += before - (std::min)(before, total_used_memory);
    }
    if (total_used_memory + incoming > critical_memory) {
        ++stats.short_of;
    }
}

template <class P>
void TextureCache<P>::NoteThrashDelete(const Image& image) {
    if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_tc_thrash_guard,
                                         "FRITCHII_TC_THRASH_GUARD", 1) <= 0) {
        return;
    }
    if (thrash_ghosts.empty()) {
        thrash_ghosts.resize(ThrashGhostSlots);
    }
    const u64 shape = RelocationShapeKey(image.info);
    const bool pressured = delete_reason == DeleteReason::GarbageCollect &&
                           (budget_pressure || total_used_memory >= critical_memory);
    thrash_ghosts[ThrashSlot(image.gpu_addr, shape)] = ThrashGhost{
        .gpu_addr = image.gpu_addr,
        .shape = shape,
        .last_use = image.last_use_tick,
        .kib = static_cast<u32>(ImageMemoryBytes(image) >> 10),
        .cause = static_cast<u8>(pressured ? ThrashCausePressured
                                           : static_cast<size_t>(delete_reason)),
        .costly = True(image.flags & ImageFlagBits::CostlyLoad),
    };
}

template <class P>
void TextureCache<P>::NoteThrashMade(const ImageInfo& info, GPUVAddr gpu_addr) {
    ++thrash_stats.made;
    if (thrash_ghosts.empty()) {
        return;
    }
    const u64 shape = RelocationShapeKey(info);
    ThrashGhost& ghost = thrash_ghosts[ThrashSlot(gpu_addr, shape)];
    if (ghost.gpu_addr != gpu_addr || ghost.shape != shape) {
        return;
    }
    const u64 interval = frame_tick - (std::min)(frame_tick, ghost.last_use);
    const size_t cause = ghost.cause;
    const bool costly = ghost.costly;
    const u32 kib = ghost.kib;
    ghost = ThrashGhost{};
    // Back after half a minute or more is a scene coming back, not a loop.
    if (interval > 1800) {
        return;
    }
    auto& stats = thrash_stats;
    ++stats.remade[cause];
    stats.remade_kib += kib;
    stats.remade_costly += costly ? 1 : 0;
    stats.interval_sum += interval;
    if (cause == ThrashCausePressured && interval <= ThrashMaxFloor) {
        size_t bucket = 0;
        while (bucket + 1 < ThrashBuckets && interval > (u64{4} << bucket)) {
            ++bucket;
        }
        ++thrash_window[bucket];
    }
}

template <class P>
void TextureCache<P>::TickThrashGuard() {
    // FRITCHII_TC_THRASH_GUARD (live; 0 off, 1 = default, 2 counts only). The collector works from
    // the driver's usage, which on a small card holds far more than images: Castle Town on a
    // simulated 4 GB card (budget 3,328 MiB) had 4,100-4,900 MiB in the cache's view with 1,500 in
    // images, against a critical of 2,100-2,400. Deleting every image could not reach it, so the
    // collector stayed 1 GiB+ over, where it takes up to 1,064 images a frame unused for just 3
    // frames, ASTC included. Textures the game comes back to every few frames were deleted and
    // made again 55-60 times a frame (normal: under 1), each an image, its memory and an ASTC
    // decode on the GPU, 44 fps with 20 fps lows against 75.
    //
    // Deleted images are remembered by address and shape. When the game asks for one again soon
    // after the collector took it under pressure, the time since its last use says how long it
    // should have stayed: while 8+ a second come back, the collector and the evictions before
    // creates leave alone what was used within 1.25x the 90th percentile of those gaps. Only
    // keeping images longer: what they hold is what a big card keeps anyway. Bounds: 600
    // frames, and live images at 3/4 of the budget, where the floor halves (past that, memory
    // comes first). It only acts under pressure, so cards with room never get one.
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_tc_thrash_guard,
                                                      "FRITCHII_TC_THRASH_GUARD", 1);
    if (mode <= 0) {
        thrash_floor = 0;
        thrash_window = {};
        thrash_stats = {};
        if (!thrash_ghosts.empty()) {
            thrash_ghosts = std::vector<ThrashGhost>{};
        }
        return;
    }
    constexpr u64 CheckFrames = 60;
    constexpr u32 MinRemakes = 8;
    if (frame_tick % CheckFrames == 0) {
        const bool pressure = budget_pressure || total_used_memory >= critical_memory;
        thrash_calm_frames = pressure ? 0 : thrash_calm_frames + CheckFrames;
        const u32 remakes = std::accumulate(thrash_window.begin(), thrash_window.end(), u32{0});
        u64 wanted = 0;
        if (remakes >= MinRemakes) {
            u32 seen = 0;
            for (size_t bucket = 0; bucket < ThrashBuckets; ++bucket) {
                seen += thrash_window[bucket];
                if (seen * 10 >= remakes * 9) {
                    const u64 gap = u64{4} << bucket;
                    wanted = (std::min)(gap + gap / 4 + 2, ThrashMaxFloor);
                    break;
                }
            }
        }
        thrash_window = {};
        u64 floor = thrash_floor;
        if (mode == 1 && wanted > floor) {
            floor = wanted;
            ++thrash_stats.raised;
        } else if (mode != 1) {
            floor = 0;
        } else if (thrash_calm_frames >= 600 && floor != 0) {
            // Inert without pressure; going down slowly it is ready for the next spell.
            floor -= floor / 8 + 1;
        }
        if (floor != 0) {
            const bool new_short = create_enforce_stats.short_of > last_thrash_short_of;
            last_thrash_short_of = create_enforce_stats.short_of;
            u64 bytes = 0;
            for (const auto [image_id, image] : slot_images) {
                if (False(image->flags & ImageFlagBits::Unbacked)) {
                    bytes += ImageMemoryBytes(*image);
                }
            }
            thrash_image_bytes = bytes;
            if constexpr (requires { runtime.GetDeviceMemoryBudget(); }) {
                const u64 budget = runtime.GetDeviceMemoryBudget();
                const u64 cap_threshold = budget < 5_GiB ? budget / 2 : budget / 4 * 3;
                if (budget != 0 && budget < 5_GiB && floor > 120) {
                    floor = 120;
                    ++thrash_stats.capped;
                } else if (budget != 0 &&
                           (bytes > cap_threshold ||
                            (pressure && new_short && floor > 60))) {
                    floor /= 2;
                    ++thrash_stats.capped;
                }
            }
        }
        thrash_floor = floor;
    }
    if (frame_tick % 600 != 0) {
        return;
    }
    auto& s = thrash_stats;
    const u64 remade = std::accumulate(s.remade.begin(), s.remade.end(), u64{0});
    if (remade != 0 || thrash_floor != 0 || s.capped != 0) {
        const auto at = [&s](DeleteReason reason) { return s.remade[static_cast<size_t>(reason)]; };
        LOG_INFO(HW_GPU,
                 "Image churn (FRITCHII_TC_THRASH_GUARD={}) last 600 frames: {} images made, {} "
                 "made again after a delete ({} MiB, {} costly to load, {:.1f} frames after their "
                 "last use); deleted by: collector under pressure {}, collector {}, cold sweep "
                 "{}, superseded sweep {}, slot retire {}, join {}, unmap {}, unmap sweep {}, "
                 "other {}; collector age floor {} frames (raised {} times, halved {} times at "
                 "3/4 of the budget), live images {} MiB",
                 mode, s.made, remade, s.remade_kib >> 10, s.remade_costly,
                 remade != 0 ? static_cast<double>(s.interval_sum) / static_cast<double>(remade)
                             : 0.0,
                 s.remade[ThrashCausePressured], at(DeleteReason::GarbageCollect),
                 at(DeleteReason::ColdSweep), at(DeleteReason::SupersededSweep),
                 at(DeleteReason::SlotRetire), at(DeleteReason::Join), at(DeleteReason::Unmap),
                 at(DeleteReason::UnmapSweep), at(DeleteReason::Other), thrash_floor, s.raised,
                 s.capped, thrash_image_bytes >> 20);
    }
    s = {};
}

template <class P>
void TextureCache<P>::LogVramOwners() {
    // FRITCHII_VRAM_OWNERS (live, seconds between lines, default 10, 0 = off): who holds device
    // memory, next to the budget. A walk of the live images (a few thousand) every 10 s.
    if constexpr (requires { runtime.VramOwnersText(u64{}, u64{}); }) {
        const int seconds = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_vram_owners,
                                                             "FRITCHII_VRAM_OWNERS", 10);
        if (seconds <= 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (vram_owners_time == std::chrono::steady_clock::time_point{}) {
            vram_owners_time = now;
            return;
        }
        if (now - vram_owners_time < std::chrono::seconds{seconds}) {
            return;
        }
        vram_owners_time = now;
        u64 image_bytes{};
        u64 gpu_written_bytes{};
        size_t images{};
        size_t unbacked{};
        for (const auto [image_id, image] : slot_images) {
            ++images;
            if (True(image->flags & ImageFlagBits::Unbacked)) {
                ++unbacked;
                continue;
            }
            const u64 bytes = ImageMemoryBytes(*image);
            image_bytes += bytes;
            if (True(image->flags & ImageFlagBits::GpuModified)) {
                gpu_written_bytes += bytes;
            }
        }
        const u64 deleted = std::accumulate(sentenced_image_bytes.begin(),
                                            sentenced_image_bytes.end(), gpu_sentenced_bytes);
        BudgetSeen& seen = budget_seen;
        auto& enforce = create_enforce_stats;
        LOG_INFO(HW_GPU,
                 "VRAM owners: {}; texture cache {} images ({} MiB GPU-written, {} unbacked), "
                 "its view {} MiB, expected {} MiB, critical {} MiB (base {} MiB{}); budget mode {} "
                 "(0 fixed, 1 vk, 2 os), budget {} MiB (vk {} MiB, os {} MiB, other processes {} "
                 "MiB), headroom {} MiB, last {} s: peak usage {} MiB, lowest budget {} MiB, {} s "
                 "over budget less headroom, {} s over the budget; evictions before creates {} "
                 "passes, {} images, {} MiB, {} short, {} at the frame cap",
                 runtime.VramOwnersText(image_bytes, deleted), images, gpu_written_bytes >> 20,
                 unbacked, total_used_memory >> 20, expected_memory >> 20, critical_memory >> 20,
                 base_critical_memory >> 20, budget_pressure ? ", budget cap on" : "",
                 seen.mode, seen.budget >> 20, seen.vk_budget >> 20, seen.os_budget >> 20,
                 seen.others >> 20, seen.headroom >> 20, seconds, seen.peak_usage >> 20,
                 seen.lowest_budget == ~0ULL ? 0 : seen.lowest_budget >> 20, seen.over_seconds,
                 seen.spill_seconds, enforce.passes, enforce.deleted, enforce.freed >> 20,
                 enforce.short_of, enforce.capped);
        seen.peak_usage = 0;
        seen.lowest_budget = ~0ULL;
        seen.over_seconds = 0;
        seen.spill_seconds = 0;
        enforce = {};
        last_thrash_short_of = 0;
    }
}

template <class P>
void TextureCache<P>::TickFrame() {
    // Upload buffers grown for one huge image (title 1's 341 MiB terrain layer pools) are freed
    // rather than kept for the rest of the session.
    for (auto* const scratch : {&swizzle_data_buffer, &unswizzle_data_buffer}) {
        if (scratch->capacity() > LargeScratchCapacity) {
            scratch->release();
        }
    }
    LogZeroUploads();
    LogLostContents();
    if (depth_native_stats.window != 0) {
        // FRITCHII_SCALE_DEPTH_MAX: depth images kept native, at most one line per 10 s.
        const auto now = std::chrono::steady_clock::now();
        if (now - depth_native_stats.last_log >= std::chrono::seconds{10}) {
            LOG_INFO(HW_GPU,
                     "Scale depth max (FRITCHII_SCALE_DEPTH_MAX={}): {} depth images kept at "
                     "native size in the last 10 s, {} this session",
                     VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_scale_depth_max,
                                                      "FRITCHII_SCALE_DEPTH_MAX", 8192),
                     depth_native_stats.window, depth_native_stats.total);
            depth_native_stats.window = 0;
            depth_native_stats.last_log = now;
        }
    }
    // EXP121: before the usage is read below, so landed images leave the queued bytes.
    if (!collect_downloads.empty()) {
        LandCollectDownloads();
    }
    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        // Switch 2 titles (native tiling) stream far more texture data than the original thresholds
        // were tuned for: title 1's attract loop alone keeps ~5.3 GB resident, so the collector
        // sat in aggressive mode evicting decoded ASTC (and decoding it again). Total use is
        // measured from the driver, so collect only near the device budget instead.
        if (!native_thresholds_applied && gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
            native_thresholds_applied = true;
            const u64 device_local = runtime.GetDeviceLocalMemory();
            critical_memory = (std::max)(critical_memory, device_local - 256_MiB);
            expected_memory = (std::max)(expected_memory, device_local - 1280_MiB);
            ApplyThresholdOverrides();
            native_critical_memory = critical_memory;
            native_expected_memory = expected_memory;
        }
        // Follow the driver's budget (re-read now and then): it shrinks when other programs use
        // VRAM (another game: 7.0 -> 6.4 GB), and collecting only past it means the driver pages
        // textures out. It is also the room to use: the emulator's device memory figure is capped near
        // 7 GB, which on a 12 GB card (11.2 GB budget) kept title 1 evicting textures with 3 GB
        // unused. Both thresholds stay under it with a margin (1/8 of it, at most 1 GiB): past it
        // allocations fail and images end up unbacked (garbage textures), and Windows shrinks the
        // budget as the card fills (16 GB card: 13.5 GB left mid-race), hence the re-read every 30
        // frames. EDEN_NX2_TC_*_MIB overrides still win.
        if constexpr (requires { runtime.GetDeviceMemoryBudget(); }) {
            if constexpr (requires { runtime.MemoryPressureEvents(); }) {
                // Images that did not fit (made in host memory or with nothing bound) mean the
                // thresholds sit too high for this card and driver: leave more room until the
                // pressure has been gone for a while.
                if (native_thresholds_applied) {
                    const u32 events = runtime.MemoryPressureEvents();
                    if (events != seen_pressure_events) {
                        seen_pressure_events = events;
                        // FRITCHII_SMALL_CARD (part Fit): at most 1/8 of the budget. An RTX
                        // 2080 (budget ~7 GiB) loading Castle Town had ~60 refused images in
                        // 0.5 s: the margin climbed to 3 GiB, critical fell to ~3.6 GiB and the
                        // collector evicted and downloaded the scene it was drawing (read-backs
                        // of 50-120 ms for seconds). Fit already keeps the uncounted memory out.
                        u64 pressure_cap = 3_GiB;
                        if (VideoCore::SmallCard::On(VideoCore::SmallCard::Fit)) {
                            if constexpr (requires { runtime.GetDeviceMemoryBudget(); }) {
                                pressure_cap = (std::max)(runtime.GetDeviceMemoryBudget() / 8,
                                                          u64{512_MiB});
                            }
                        }
                        pressure_margin = (std::min)(pressure_margin + 512_MiB, pressure_cap);
                        pressure_quiet_frames = 0;
                        budget_check_frames = 0; // thresholds are recomputed below
                        LOG_WARNING(HW_GPU, "Out of device memory: keeping {} MiB more free",
                                    pressure_margin >> 20);
                    } else if (pressure_margin != 0 && ++pressure_quiet_frames % 600 == 0) {
                        pressure_margin -= (std::min)(pressure_margin, 128_MiB);
                        budget_check_frames = 0;
                    }
                }
            }
            if (native_thresholds_applied && budget_check_frames++ % 30 == 0) {
                const u64 budget = runtime.GetDeviceMemoryBudget();
                if constexpr (requires { runtime.NoteBudgetPressure(u64{}, u64{}); }) {
                    runtime.NoteBudgetPressure(runtime.GetDeviceMemoryUsage(), budget);
                }
                if (budget > 2_GiB) {
                    const u64 old_critical = critical_memory;
                    const u64 margin = (std::min)(budget / 8, 1_GiB);
                    if constexpr (requires { runtime.GetDeviceMemorySlack(); }) {
                        // The usage counted here leaves out the unused part of the allocator's
                        // blocks, which the driver counts and a full heap cannot hand to a new
                        // image (RX 7800 XT: 1.06 GiB of it, the whole margin, with the heap at
                        // its budget for a minute before images stopped fitting). Reserve it
                        // too: at once when it grows, slowly when it shrinks, so evicting images
                        // (which leaves partly used blocks) does not chase the threshold down.
                        const u64 slack = (std::min)(runtime.GetDeviceMemorySlack(), 1536_MiB);
                        slack_margin = slack >= slack_margin
                                           ? slack
                                           : slack_margin - (std::min)(slack_margin - slack, 16_MiB);
                    }
                    // FRITCHII_SMALL_CARD (part Fit): what the usage leaves out (the image pool,
                    // deleted images still in flight, collector downloads) is memory the driver
                    // counts. Kept out of the thresholds, the cache filled to critical and those
                    // went on top: up to ~1.2 GiB past the 1/8 margin at the market (pool 1 GiB),
                    // over the budget of an 8 GB card at 1440p and of every 6 GB card. Reserved
                    // like the slack: at once when it grows, 16 MiB a check when it shrinks.
                    // The margin then only covers growth between checks and the budget
                    // shrinking: 1/16 of the budget, at most 512 MiB.
                    u64 critical_margin = margin;
                    if (VideoCore::SmallCard::On(VideoCore::SmallCard::Fit)) {
                        const u64 held = UncountedDeviceBytes();
                        held_margin = held >= held_margin
                                          ? held
                                          : held_margin - (std::min)(held_margin - held, 16_MiB);
                        critical_margin = SmallCardMargin(budget) + held_margin;
                        pressure_margin = (std::min)(
                            pressure_margin, (std::max)(budget / 8, u64{512_MiB}));
                    } else {
                        held_margin = 0;
                    }
                    critical_memory = budget - (std::min)(critical_margin + pressure_margin +
                                                              slack_margin,
                                                          budget / 2);
                    // Ported fix: the device's budget is not all the room there is.
                    // Windows charges device memory to the system's commit limit, so the cache
                    // may also hold no more than the device memory in use plus the commit still
                    // free, less 2 GiB (HostCriticalMemory; an OoT out-of-memory crash on a PC
                    // with 16 GiB of RAM). The usage is the driver's.
                    const std::optional<u64> commit_room = Common::GetSystemCommitRoom();
                    const u64 host_critical =
                        commit_room
                            ? HostCriticalMemory(runtime.GetDeviceMemoryUsage(), *commit_room)
                            : (std::numeric_limits<u64>::max)();
                    critical_memory = (std::min)(critical_memory, host_critical);
                    expected_memory = critical_memory - (std::min)(margin, critical_memory / 2);
                    ApplyThresholdOverrides(false);
                    ApplySmallCardImageFloor(budget, true);
                    // FRITCHII_KEEP_BLOCKS: the kept blocks get what the live usage leaves under
                    // the critical use and half the room above it.
                    if constexpr (requires { runtime.LimitKeptBlocks(u64{}, u64{}); }) {
                        runtime.LimitKeptBlocks(budget, critical_memory);
                    }
                    // Log the first values and real changes (the budget moves by a few MiB all
                    // the time). Compared after the overrides: the value before them is above
                    // the 5/8-of-budget cap on large cards, so every check looked like a change
                    // and logged the same thresholds every 30 frames.
                    const u64 change = critical_memory > old_critical
                                           ? critical_memory - old_critical
                                           : old_critical - critical_memory;
                    const bool log = budget_check_frames == 1 || change >= 256_MiB;
                    if (log) {
                        LOG_INFO(HW_GPU,
                                 "Texture cache thresholds: expected {} MiB, critical {} MiB, "
                                 "minimum {} MiB",
                                 expected_memory >> 20, critical_memory >> 20,
                                 minimum_memory >> 20);
                        LOG_INFO(HW_GPU,
                                 "Memory margins: base {} MiB, out of memory {} MiB, block slack {} "
                                 "MiB, uncounted {} MiB (small card profile {})",
                                 (critical_margin - held_margin) >> 20, pressure_margin >> 20,
                                 slack_margin >> 20, held_margin >> 20,
                                 VideoCore::SmallCard::On(VideoCore::SmallCard::Fit) ? "on"
                                                                                     : "off");
                        if (commit_room) {
                            LOG_INFO(HW_GPU, "Host memory: {} MiB of commit left, {}",
                                     *commit_room >> 20,
                                     critical_memory == host_critical
                                         ? "which bounds the thresholds"
                                         : "the device budget bounds the thresholds");
                        }
                    }
                }
            }
        }
    }
    if constexpr (requires { runtime.CurrentTick(); }) {
        while (!gpu_sentenced_images.empty() &&
               runtime.IsFree(gpu_sentenced_images.front().tick)) {
            // EXP42: the GPU is done with it, so its VkImage may go to the runtime's pool for
            // the next image of its shape (fable_answer_12 2B).
            if constexpr (requires(Image& image) { image.ReleaseToPool(); }) {
                gpu_sentenced_images.front().image.ReleaseToPool();
            }
            // EDEN_NX2_ASYNC_IMAGE_FREE: what is left of it is destroyed on the runtime's free
            // thread (moved out here; pop_front then destroys an empty image), not here.
            if constexpr (requires(Image& image) { runtime.FreeImageLater(image); }) {
                runtime.FreeImageLater(gpu_sentenced_images.front().image);
            }
            gpu_sentenced_bytes -= gpu_sentenced_images.front().bytes;
            gpu_sentenced_images.pop_front();
        }
        // Fritchii's free guard: images deleted with a download queued. The download records
        // may still hold them (then the landing destroys them, the GPU being past this tick).
        while (!retained_image_keepers.empty() &&
               runtime.IsFree(retained_image_keepers.front().tick)) {
            retained_image_keepers.pop_front();
        }
        if (frame_tick % 600 == 0) {
            const u64 total = free_guard.retained_held + free_guard.landed_early +
                              free_guard.freed_early + free_guard.unswizzle_dropped;
            if (total != free_guard.logged_total) {
                free_guard.logged_total = total;
                LOG_INFO(HW_GPU,
                         "Free guard (session): {} images deleted with a download queued held "
                         "until the GPU passed their delete ({} of those downloads landed "
                         "before it: freed in use without the guard), {} held now, {} destroyed "
                         "early anyway (must be 0); {} async unswizzles dropped with their image",
                         free_guard.retained_held, free_guard.landed_early,
                         retained_image_keepers.size(), free_guard.freed_early,
                         free_guard.unswizzle_dropped);
            }
        }
        if constexpr (requires { runtime.SubmitImageFrees(); }) {
            runtime.SubmitImageFrees();
        }
    }
    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
        // Images the collector deleted keep their memory for TICKS_TO_DESTROY frames: counting
        // it kept the collector in aggressive mode, deleting 40+ images a frame for frames after
        // it had freed enough (title 1: frames over 100 ms for seconds at a time).
        for (const u64 bytes : sentenced_image_bytes) {
            total_used_memory -= (std::min)(total_used_memory, bytes);
        }
        total_used_memory -= (std::min)(total_used_memory, gpu_sentenced_bytes);
        // EXP121: images whose download is queued are as good as deleted.
        total_used_memory -= (std::min)(total_used_memory, collect_download_memory);
        // EDEN_NX2_ASYNC_IMAGE_FREE: deleted images handed to the free thread and not destroyed
        // yet are still counted by the driver; left out, so the collector sees what it saw when
        // they were destroyed above. Read after the usage: a free in between leaves the usage
        // counted high, never low.
        if constexpr (requires { runtime.PendingImageFreeBytes(); }) {
            total_used_memory -= (std::min)(total_used_memory, runtime.PendingImageFreeBytes());
        }
        // EXP42's pooled images are deleted images too: the pool caps and ages them itself.
        if constexpr (requires { runtime.PooledImageBytes(); }) {
            total_used_memory -= (std::min)(total_used_memory, runtime.PooledImageBytes());
        }
    }
    // FRITCHII_VRAM_BUDGET_MODE / FRITCHII_VRAM_OWNERS: once a second / every 10 s, with the
    // usage just read (the collector below already sees a moved cap).
    BudgetControl();
    LogVramOwners();
    // EDEN_NX2_MEM_TRACE=1 (ported): every 2 s, what the cache holds next to
    // what the process and the driver report (memory climbs).
    static const bool mem_trace = std::getenv("EDEN_NX2_MEM_TRACE") != nullptr;
    if (const auto now = std::chrono::steady_clock::now();
        mem_trace && now - mem_trace_time >= std::chrono::seconds{2}) {
        mem_trace_time = now;
        u64 image_bytes{};
        u64 gpu_written_bytes{};
        size_t images{};
        size_t unbacked{};
        for (const auto [image_id, image] : slot_images) {
            ++images;
            if (True(image->flags & ImageFlagBits::Unbacked)) {
                ++unbacked;
                continue;
            }
            const u64 bytes = GetScaledImageSizeBytes(*image);
            image_bytes += bytes;
            if (True(image->flags & ImageFlagBits::GpuModified)) {
                gpu_written_bytes += bytes;
            }
        }
        u64 decode_bytes{};
        for (const auto& decode : async_decodes) {
            // The worker sizes the buffer before it publishes complete.
            if (decode->complete.load(std::memory_order_acquire)) {
                decode_bytes += decode->decoded_data.capacity();
            }
        }
        const u64 sentenced = std::accumulate(sentenced_image_bytes.begin(),
                                              sentenced_image_bytes.end(), gpu_sentenced_bytes);
        LOG_INFO(HW_GPU,
                 "Memory trace: texture cache {} images {} MiB ({} MiB GPU-written, {} unbacked), "
                 "device usage {} MiB (expected {} MiB, critical {} MiB), sentenced {} MiB, {} "
                 "async decodes {} MiB, scratch {} MiB",
                 images, image_bytes >> 20, gpu_written_bytes >> 20, unbacked,
                 total_used_memory >> 20, expected_memory >> 20, critical_memory >> 20,
                 sentenced >> 20, async_decodes.size(), decode_bytes >> 20,
                 (swizzle_data_buffer.capacity() + unswizzle_data_buffer.capacity()) >> 20);
        if constexpr (requires { runtime.LogMemoryTrace(); }) {
            runtime.LogMemoryTrace();
        }
    }
    // Every 600 frames: device memory in use against the collector's thresholds.
    // Skip expensive heap allocation, map construction, and sorting unless debug logging is explicitly enabled.
    static const bool log_tc = std::getenv("EDEN_NX2_TC_LOG") != nullptr;
    static u64 frames{};
    if (log_tc && ++frames % 600 == 0) {
        u64 image_bytes{};
        size_t images{};
        std::vector<std::pair<u64, const Image*>> largest;
        std::map<u32, std::pair<u64, u32>> by_format;
        for (const auto [image_id, image] : slot_images) {
            const u64 bytes = GetScaledImageSizeBytes(*image);
            image_bytes += bytes;
            ++images;
            if (debug_stats) {
                largest.emplace_back(bytes, image);
                auto& format = by_format[static_cast<u32>(image->info.format)];
                format.first += bytes;
                ++format.second;
            }
        }
        LOG_INFO(HW_GPU, "Texture cache: device memory {} MiB (expected {} MiB, critical {} MiB), {} images {} MiB",
                 total_used_memory >> 20, expected_memory >> 20, critical_memory >> 20, images,
                 image_bytes >> 20);
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(12, largest.size()),
                          largest.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t i = 0; i < std::min<size_t>(12, largest.size()); ++i) {
            const Image& image = *largest[i].second;
            LOG_DEBUG(HW_GPU, "  {} MiB: {:#x} {}x{}x{} layers {} levels {} format {} type {} flags {:#x}",
                     largest[i].first >> 20, image.gpu_addr, image.info.size.width, image.info.size.height,
                     image.info.size.depth, image.info.resources.layers, image.info.resources.levels,
                     static_cast<u32>(image.info.format), static_cast<u32>(image.info.type),
                     static_cast<u32>(image.flags));
        }
        std::vector<std::pair<u64, u32>> formats;
        for (const auto& [format, total] : by_format) formats.emplace_back(total.first, format);
        std::sort(formats.rbegin(), formats.rend());
        for (size_t i = 0; i < std::min<size_t>(8, formats.size()); ++i) {
            LOG_DEBUG(HW_GPU, "  format {}: {} MiB in {} images", formats[i].second, formats[i].first >> 20,
                     by_format[formats[i].second].second);
        }
        if constexpr (requires { runtime.LogMemoryStatistics(); }) {
            runtime.LogMemoryStatistics();
        }
        LOG_INFO(HW_GPU, "Texture cache last 600 frames: created {} images, deleted {} (collector {}, {} costly, {} downloaded, {} downloads deferred), {} async decodes, ASTC {} opaque / {} alpha",
                 images_created, images_deleted, gc_deletes + gc_costly_deletes, gc_costly_deletes,
                 gc_download_deletes, gc_downloads_deferred,
                 async_decodes_queued, astc_opaque_images, astc_alpha_images);
        LOG_INFO(HW_GPU, "Texture cache last 600 frames: {} uploads from guest memory, {} MiB (largest {} MiB, {} by layer)",
                 images_uploaded, uploaded_bytes >> 20, largest_upload >> 20, layer_uploads);
        LOG_INFO(HW_GPU,
                 "Texture cache last 600 frames: stale uploads skipped {} ({} MiB), {} uploaded "
                 "for buffer-written pages; {} MiB of pages marked stale",
                 tc_stale_skips, tc_stale_skip_bytes >> 20, tc_stale_gpu_written,
                 (stale_page_count << STALE_PAGE_BITS) >> 20);
        tc_stale_skips = tc_stale_skip_bytes = tc_stale_gpu_written = 0;
        images_uploaded = uploaded_bytes = largest_upload = layer_uploads = 0;
        astc_opaque_images = astc_alpha_images = 0;
        std::vector<std::pair<u32, u64>> shapes;
        for (const auto& [key, value] : created_shapes) shapes.emplace_back(value.first, key);
        std::sort(shapes.rbegin(), shapes.rend());
        for (size_t i = 0; i < std::min<size_t>(10, shapes.size()); ++i) {
            const u64 key = shapes[i].second;
            LOG_DEBUG(HW_GPU, "  created {}x: format {} type {} levels {} layers {} {}x{} (last at {:#x})",
                     shapes[i].first, key >> 56, (key >> 52) & 0xF, (key >> 48) & 0xF,
                     (key >> 36) & 0xFFF, (key >> 18) & 0x3FFFF, key & 0x3FFFF,
                     created_shapes[key].second);
        }
        created_shapes.clear();
        // The per-image and per-shape tallies only feed debug lines: skip them while those are off.
        debug_stats = Common::Log::IsLevelEnabled(Common::Log::Class::HW_GPU, Common::Log::Level::Debug);
        gc_deletes = gc_costly_deletes = gc_download_deletes = gc_downloads_deferred =
            async_decodes_queued = 0;
        images_created = images_deleted = 0;
    }
    // EXP121's counts, and the collector downloads that still wait for the GPU.
    if (frame_tick % 600 == 0 && (log_tc || VideoCore::Nx2Debug::Experiment(121) != 0)) {
        auto& s = collect_stats;
        if (s.waits != 0 || s.made != 0 || !collect_downloads.empty()) {
            LOG_INFO(HW_GPU,
                     "Texture cache last 600 frames: collector downloads {} waited, {} queued, "
                     "{} landed ({:.1f} frames after), dropped {} used again / {} written again "
                     "/ {} deleted, fell back {} cap / {} out of memory, {} mismatches, {} queued "
                     "now ({} MiB staging); written again: {} by a newer overlap, {} by the CPU; "
                     "{} landed under a GPU-only overlap (FRITCHII_COLLECT_LAND_UNDER)",
                     s.waits, s.made, s.landed,
                     s.landed != 0 ? static_cast<double>(s.land_frames) / s.landed : 0.0,
                     s.dropped_used, s.dropped_written, s.dropped_deleted, s.fallback_cap,
                     s.fallback_oom, s.mismatches, collect_downloads.size(),
                     collect_download_staging >> 20, s.dropped_under, s.dropped_cpu,
                     s.landed_under);
        }
        s = {};
    }
    // FRITCHII_TC_COLLECT_OVER and FRITCHII_TC_COLLECT_EVICTABLE: what they did, every 600 frames.
    if (frame_tick % 600 == 0) {
        auto& s = collect_port_stats;
        const int over = VideoCore::Nx2Debug::g_tc_collect_over.load(std::memory_order_relaxed);
        const int evictable =
            VideoCore::Nx2Debug::g_tc_collect_evictable.load(std::memory_order_relaxed);
        if (over > 0 || evictable > 0) {
            LOG_INFO(HW_GPU,
                     "Collector ports (FRITCHII_TC_COLLECT_OVER={} FRITCHII_TC_COLLECT_EVICTABLE={}) "
                     "last 600 frames: {} passes past critical (+{} deletes allowed, {} at 1 GiB+ "
                     "over, peak {} MiB over), {} images deleted past critical; {} skipped as not "
                     "deletable; device memory {} MiB (critical {} MiB)",
                     (std::max)(over, 0), (std::max)(evictable, 0), s.over_passes, s.over_extra,
                     s.over_young, s.over_peak >> 20, s.deleted_aggressive, s.skipped,
                     total_used_memory >> 20, critical_memory >> 20);
        }
        s = {};
    }
    unmap_stats.held_byte_frames += unmapped_live_bytes;
    unmap_stats.held_peak = (std::max)(unmap_stats.held_peak, unmapped_live_bytes);
    if (frame_tick % 600 == 0) {
        if (log_tc || SweepUnmappedMode() != 0) {
            LogUnmapStats();
        }
        unmap_stats = {};
    }
    SweepUnmapped();
    SweepSuperseded();
    if (const u64 rt_evict = RtEvictPressureFrames();
        rt_evict != 0 && total_used_memory > expected_memory) {
        const DeleteReasonScope delete_reason_scope{*this, DeleteReason::ColdSweep};
        for (size_t budget = transient_targets.size(); budget != 0 && !transient_targets.empty();
             --budget) {
            TransientTarget entry = transient_targets.front();
            if (frame_tick - entry.check_tick < rt_evict) {
                break;
            }
            transient_targets.pop_front();
            if (!slot_images.contains(entry.id)) {
                continue;
            }
            Image& image = slot_images[entry.id];
            if (image.allocation_tick != entry.allocation_tick ||
                False(image.flags & ImageFlagBits::TransientTarget) ||
                False(image.flags & ImageFlagBits::Registered)) {
                continue;
            }
            if (True(image.flags & ImageFlagBits::GpuModified)) {
                image.flags &= ~ImageFlagBits::TransientTarget;
                continue;
            }
            const u64 last_use = lru_cache.Tick(image.lru_index);
            if (frame_tick - last_use < rt_evict || True(image.flags & ImageFlagBits::IsDecoding)) {
                if (++entry.survived < 4) {
                    entry.check_tick = frame_tick;
                    transient_targets.push_back(entry);
                } else {
                    image.flags &= ~ImageFlagBits::TransientTarget;
                }
                continue;
            }
            MarkStale(image);
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, entry.id);
            }
            UnregisterImage(entry.id);
            DeleteImage(entry.id, false);
            ++transient_evictions;
        }
    } else if (transient_targets.size() > 8192) {
        transient_targets.erase(transient_targets.begin(),
                                transient_targets.begin() + (transient_targets.size() - 8192));
    }
    if (frame_tick % 600 == 0 && transient_evictions != logged_transient_evictions) {
        LOG_INFO(HW_GPU, "RT evict: {} unused fresh render targets deleted early so far, {} waiting",
                 transient_evictions, transient_targets.size());
        logged_transient_evictions = transient_evictions;
    }
    SweepColdImages();
    TickThrashGuard();
    // An image that could not be made at all: the driver is out of memory whatever the counts
    // say, so collect as at the critical threshold.
    bool out_of_memory = false;
    if constexpr (requires { runtime.FailedImages(); }) {
        const u32 failed = runtime.FailedImages();
        out_of_memory = failed != seen_failed_images;
        seen_failed_images = failed;
    }
    if (out_of_memory) {
        cold_keep_failed_frame = (std::max)(frame_tick, u64{1}); // the cold sweep keep's gate
    }
    if (out_of_memory || total_used_memory > minimum_memory) {
        RunGarbageCollector(out_of_memory);
    }
    if (out_of_memory) {
        // Textures left out for lack of memory are made again on their next read, once a frame
        // while images keep failing.
        for (size_t c : active_channel_ids) {
            channel_storage[c].graphics_image_table.Invalidate();
            channel_storage[c].compute_image_table.Invalidate();
        }
        // Texture memos of their null views would skip that read (EXP32).
        ++texture_memo_epoch;
    }
    sentenced_images.Tick();
    sentenced_image_index = (sentenced_image_index + 1) % TICKS_TO_DESTROY;
    sentenced_image_bytes[sentenced_image_index] = 0;
    sentenced_framebuffers.Tick();
    sentenced_image_view.Tick();
    TickAsyncDecode();
    TickAsyncUnswizzle();
    if (frame_tick % 600 == 0 && !uploaded_ranges.empty()) {
        // EDEN_NX2_EXP8 bit 8: none. An image created again over cold memory (re-created after
        // eviction, or a render target at a new size) faults all of it back in while uploading.
        if ((VideoCore::Nx2Debug::Experiment(8) & 8) == 0) {
            for (const auto& [address, size] : uploaded_ranges) {
                device_memory.HintCold(address, size);
            }
        }
        uploaded_ranges.clear();
    }

    runtime.TickFrame();
    ++frame_tick;
    if (const int lru_switch = VideoCore::Nx2Debug::Experiment(90); lru_switch != 0) {
        const auto now = std::chrono::steady_clock::now();
        if (nx2_lru_touch_report == std::chrono::steady_clock::time_point{}) {
            nx2_lru_touch_report = now;
        } else if (now - nx2_lru_touch_report >= std::chrono::seconds{5}) {
            const u64 all = nx2_lru_touches + nx2_lru_touches_skipped;
            LOG_INFO(HW_GPU,
                     "Image LRU touch (EXP90={}) in 5 s: {} touched, {} skipped ({:.1f}% of {})",
                     lru_switch, nx2_lru_touches, nx2_lru_touches_skipped,
                     all != 0 ? 100.0 * static_cast<double>(nx2_lru_touches_skipped) /
                                    static_cast<double>(all)
                              : 0.0,
                     all);
            nx2_lru_touch_report = now;
            nx2_lru_touches = 0;
            nx2_lru_touches_skipped = 0;
        }
    }
    // Prepared render targets skip the LRU touch of their images: keep them to one frame.
    ++bindings_epoch;
    if (VideoCore::Nx2Debug::Experiment(31) != 0 && frame_tick % 600 == 0) {
        LOG_INFO(HW_GPU,
                 "RT_FAST (EXP31) last 600 frames: {} of {} clean render target updates skipped; "
                 "verified {}, mismatches {}",
                 rt_fast_skips, rt_fast_clean_paths, rt_fast_verified, rt_fast_mismatches);
        rt_fast_skips = rt_fast_clean_paths = rt_fast_verified = rt_fast_mismatches = 0;
    }
    if (relocation_seen && frame_tick % 600 == 0) {
        u32 guarded_shapes = 0;
        for (const auto& [key, shape] : relocation_shapes) {
            guarded_shapes += shape.guard != 0 ? 1 : 0;
        }
        LOG_INFO(HW_GPU,
                 "Render target relocation (EXP43={}) last 600 frames: {} candidates, {} kept by "
                 "the guard, {} relocated, {} retired early, {} over 2+ images of their shape; "
                 "{} of {} shapes guarded",
                 VideoCore::Nx2Debug::Experiment(43), relocation_candidates, relocation_guarded,
                 relocation_relocated, relocation_retired, relocation_crowded, guarded_shapes,
                 relocation_shapes.size());
        relocation_candidates = relocation_guarded = relocation_relocated = relocation_retired =
            relocation_crowded = 0;
    }
    clear_exact_unchecked += clear_exact_pending.size();
    clear_exact_pending.clear();
    if ((clear_exact_used != 0 || clear_exact_missed != 0 || no_pool_marked != 0) &&
        frame_tick % 600 == 0) {
        LOG_INFO(HW_GPU,
                 "Clear to exact size (EXP68={}) last 600 frames: {} depth clears sent to the "
                 "scissor's width, {} confirmed by the next bind, {} missed, {} not bound again "
                 "that frame; {} addresses off; {} images kept out of the pool",
                 VideoCore::Nx2Debug::Experiment(68), clear_exact_used, clear_exact_confirmed,
                 clear_exact_missed, clear_exact_unchecked, clear_exact_unsafe.size(),
                 no_pool_marked);
        clear_exact_used = clear_exact_confirmed = clear_exact_missed = clear_exact_unchecked = 0;
        no_pool_marked = 0;
    }
    if (depth_census_on) [[unlikely]] {
        CensusTickFrame();
    }
    if ((pad_depth_lookups != 0 || pad_sampled != 0) && frame_tick % 600 == 0) {
        LOG_INFO(HW_GPU,
                 "Host pad (EXP67={}) last 600 frames: {} new depth targets, {} padded, {} at an "
                 "address without a resize yet, {} refused (lookup kind, rescaling, shape, "
                 "sampled address); {} found by a texture lookup; {} varying addresses, {} stopped",
                 VideoCore::Nx2Debug::Experiment(67), pad_depth_lookups, pad_made,
                 pad_not_varying, pad_skipped_other, pad_sampled, pad_slots.size(),
                 pad_unsafe_slots.size());
        pad_made = pad_sampled = pad_depth_lookups = pad_not_varying = pad_skipped_other = 0;
    }
    if (slot_clears != 0 && frame_tick % 600 == 0) {
        LOG_INFO(HW_GPU,
                 "Slot retire (EXP64={}) last 600 frames: {} cleared targets over same-address "
                 "images, {} retired; kept: {} guarded (sampled or CPU-read later), {} busy "
                 "(bound, download queued), {} of an excluded kind, {} written this frame; {} made "
                 "again within 16 frames of a retire",
                 VideoCore::Nx2Debug::Experiment(64), slot_clears, slot_retired, slot_kept_guard,
                 slot_kept_busy, slot_kept_kind, slot_kept_same_frame, slot_remade);
        slot_clears = slot_retired = slot_kept_guard = slot_kept_busy = slot_kept_kind =
            slot_kept_same_frame = slot_remade = 0;
        std::string flags_text;
        for (u32 bit = 0; bit < 32; ++bit) {
            if (slot_kept_flag[bit] != 0) {
                flags_text += fmt::format(" flag bit {}: {};", bit, slot_kept_flag[bit]);
            }
        }
        LOG_INFO(HW_GPU, "Slot retire (EXP64) excluded kinds:{} not GPU-written or unregistered: {}; "
                 "scaled or aliased: {}", flags_text, slot_kept_not_gpu, slot_kept_aliased);
        slot_kept_flag.fill(0);
        slot_kept_not_gpu = slot_kept_aliased = 0;
    }
    TickRenderClock();
    TickRelocationTombstones();
    // What the caches made and destroyed (fable_answer_16 4; counted always, logged on request).
    static const bool churn_log = [] {
        const char* value = std::getenv("EDEN_NX2_CHURN_LOG");
        return value != nullptr && value[0] == '1';
    }();
    if (churn_log && frame_tick % 600 == 0) {
        const auto churn = Churn::Take();
        LOG_INFO(HW_GPU,
                 "Churn (600 frames): images created {}, deleted {}, relocated {}, pooled {}; "
                 "image views created {}, destroyed {}; buffers created {}, deleted {}, joined {} "
                 "({} KiB copied; {} skipped as CPU-modified, {} KiB not copied); staging "
                 "buffers allocated {} ({} MiB); device memory "
                 "allocations {}, frees {}; descriptor pools created {}; {} images now",
                 churn[Churn::ImagesCreated], churn[Churn::ImagesDeleted],
                 churn[Churn::ImagesRelocated], churn[Churn::ImagesPooled],
                 churn[Churn::ImageViewsCreated], churn[Churn::ImageViewsDestroyed],
                 churn[Churn::BuffersCreated], churn[Churn::BuffersDeleted],
                 churn[Churn::BuffersJoined], churn[Churn::BufferJoinBytes] >> 10,
                 churn[Churn::BufferJoinsSkipped], churn[Churn::BufferJoinSkippedBytes] >> 10,
                 churn[Churn::StagingBuffers], churn[Churn::StagingBytes] >> 20,
                 churn[Churn::DeviceAllocations], churn[Churn::DeviceFrees],
                 churn[Churn::DescriptorPools], slot_images.size());
    }

    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        for (auto& buffer : async_buffers_death_ring) {
            runtime.FreeDeferredStagingBuffer(buffer);
        }
        async_buffers_death_ring.clear();
    }
}

/// EDEN_NX2_TC_COLD_FRAMES (the cold sweep's K, 0 = off).
inline u64 TcColdFrames() {
    static const u64 frames = [] {
        // 4 by default since 2026-11-08 (device memory 8.8 -> ~6.4 GB at 2 min in Castle Town,
        // 11 loads 0 washed out); EDEN_NX2_TC_COLD_FRAMES=0 turns it off.
        const char* value = std::getenv("EDEN_NX2_TC_COLD_FRAMES");
        return value != nullptr ? std::strtoull(value, nullptr, 10) : 4ULL;
    }();
    return frames;
}

template <class P>
void TextureCache<P>::WatchColdImage(ImageId image_id) {
    if (TcColdFrames() == 0 || !RelocationShaped(slot_images[image_id].info)) {
        return;
    }
    if (cold_watch_set.insert(image_id).second) {
        cold_watch.emplace_back(image_id, frame_tick);
    }
}

template <class P>
void TextureCache<P>::LearnColdResampled(GPUVAddr gpu_addr, u64 shape_key) {
    const u64 key = ColdResampledKey(gpu_addr, shape_key);
    if (const auto it = cold_resampled.find(key); it != cold_resampled.end()) {
        it->second = frame_tick;
        return;
    }
    if (cold_resampled.size() >= ColdResampledMax) {
        // Least recently seen out (a scan: keys are added a few times a minute).
        auto oldest = cold_resampled.begin();
        for (auto it = cold_resampled.begin(); it != cold_resampled.end(); ++it) {
            if (it->second < oldest->second) {
                oldest = it;
            }
        }
        cold_resampled.erase(oldest);
    }
    cold_resampled.emplace(key, frame_tick);
    ++cold_resampled_gen;
    ++cold_keep_stats.learned;
}

template <class P>
bool TextureCache<P>::ColdKeepResampled(ImageBase& image, int mode) {
    if (mode >= 2 && (image.relocation_guard_noted & GuardSampled) != 0) {
        if ((image.cold_keep & 2) == 0) {
            image.cold_keep |= 2;
            ++cold_keep_stats.kept_new;
            ++cold_keep_stats.kept_sampled;
        }
        return true;
    }
    if (cold_resampled.empty()) {
        return false;
    }
    if (image.cold_keep_gen != cold_resampled_gen) {
        image.cold_keep_gen = cold_resampled_gen;
        const auto it =
            cold_resampled.find(ColdResampledKey(image.gpu_addr, RelocationShapeKey(image.info)));
        const bool found = it != cold_resampled.end();
        if (found) {
            it->second = frame_tick; // seen again: stays in the set
        }
        image.cold_keep = static_cast<u8>((image.cold_keep & ~1) | (found ? 1 : 0));
    }
    if ((image.cold_keep & 1) == 0) {
        return false;
    }
    if ((image.cold_keep & 2) == 0) {
        image.cold_keep |= 2;
        ++cold_keep_stats.kept_new;
    }
    return true;
}

template <class P>
bool TextureCache<P>::ColdSuperseded(ImageId image_id, const ImageBase& image) {
    const u64 shape = RelocationShapeKey(image.info);
    const u64 touched = lru_cache.Tick(image.lru_index);
    bool superseded = false;
    ForEachImageInRegion(image.cpu_addr, image.guest_size_bytes,
                         [&](ImageId other_id, Image& other) {
                             if (other_id == image_id ||
                                 False(other.flags & ImageFlagBits::Registered) ||
                                 RelocationShapeKey(other.info) == shape ||
                                 lru_cache.Tick(other.lru_index) <= touched) {
                                 return false;
                             }
                             superseded = true;
                             return true;
                         });
    return superseded;
}

template <class P>
bool TextureCache<P>::CoveredByNewerWrites(ImageId image_id, ImageBase& image) {
    // FRITCHII_COVERED_RETIRE. Newer: written by the GPU after this image's last write
    // (modification_tick is one counter for the whole cache) and touched no earlier than this
    // image last was, so nothing read this image after those writes landed over it. Such
    // images together reach every byte: the memory holds none of this image's contents any
    // more, on the console as here, and a read of it (a CPU read flushes the GPU-written
    // images there) gets the newer bytes either way.
    const auto start = std::chrono::steady_clock::now();
    image.covered_check_frame = frame_tick;
    ++covered_stats.checked;
    if (image.guest_size_bytes == 0 || True(image.flags & ImageFlagBits::Sparse)) {
        return false;
    }
    const DAddr begin = image.cpu_addr;
    const DAddr end = begin + image.guest_size_bytes;
    const u64 touched = lru_cache.Tick(image.lru_index);
    const u64 written = image.modification_tick;
    covered_ranges.clear();
    ForEachImageInRegion(begin, image.guest_size_bytes, [&](ImageId other_id, Image& other) {
        if (other_id == image_id || False(other.flags & ImageFlagBits::Registered) ||
            True(other.flags & ImageFlagBits::Sparse) || other.modification_tick <= written ||
            lru_cache.Tick(other.lru_index) < touched) {
            return;
        }
        const DAddr other_begin = (std::max)(static_cast<DAddr>(other.cpu_addr), begin);
        const DAddr other_end =
            (std::min)(static_cast<DAddr>(other.cpu_addr + other.guest_size_bytes), end);
        if (other_begin < other_end) {
            covered_ranges.emplace_back(other_begin, other_end);
        }
    });
    std::ranges::sort(covered_ranges);
    DAddr reach = begin;
    for (const auto& [range_begin, range_end] : covered_ranges) {
        if (range_begin > reach) {
            break;
        }
        reach = (std::max)(reach, range_end);
        if (reach >= end) {
            break;
        }
    }
    covered_stats.check_ns += NsSince(start);
    return reach >= end;
}

template <class P>
void TextureCache<P>::RetireCovered(ImageId image_id, Image& image) {
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::ColdSweep};
    covered_stats.deleted_bytes += ImageMemoryBytes(image);
    if (covered_swept.size() >= 65536) {
        covered_swept.clear(); // the oldest are the least likely to be read again
    }
    covered_swept[image.gpu_addr] = frame_tick;
    // As the collector deletes a bad overlap: no download.
    MarkStale(image); // before UntrackImage (see MarkStale)
    if (True(image.flags & ImageFlagBits::Tracked)) {
        UntrackImage(image, image_id);
    }
    UnregisterImage(image_id);
    DeleteImage(image_id);
}

template <class P>
void TextureCache<P>::SweepSuperseded() {
    // FRITCHII_SWEEP_SUPERSEDED=N (live, frames, 0 = off; 1 = 60): a GPU-written image untouched
    // for N frames whose memory a newer image of another shape (touched since) also covers: the
    // game reused that memory for another layout, so nothing reads the old contents (a CPU read
    // there goes to the newer image). EXP58 does this for the cold sweep's chain images after 60
    // frames; this looks at every GPU-written image, every min(N, 30) frames. Skitchii runs 3:
    // OoT's Castle Town guard house at 3440x1440 moves ~20 render targets a frame while the
    // camera turns, and 60 frames of dead copies took driver usage to 13-15 GB on a 12 GB card
    // (orbit 16-24 -> 42-51 fps; 2 lost fps standing). Deleted without a download, as the
    // collector would, so their VkImages go back to the EXP42 pool in time for the next set.
    const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_sweep_superseded,
                                                       "FRITCHII_SWEEP_SUPERSEDED");
    auto& s = superseded_stats;
    if (frame_tick % 600 == 0 && (value > 0 || s.passes != 0)) {
        LOG_INFO(HW_GPU,
                 "Superseded sweep (FRITCHII_SWEEP_SUPERSEDED={}) last 600 frames: {} passes, {} "
                 "GPU-written images looked at, {} deleted ({} MiB), {} kept as bound or "
                 "downloading, {:.2f} ms spent; {} sampled again where one was deleted",
                 (std::max)(value, 0), s.passes, s.looked, s.deleted, s.deleted_bytes >> 20,
                 s.kept_busy, static_cast<double>(s.sweep_ns) / 1e6, s.sampled_after);
        s = {};
        if (value <= 0) {
            superseded_swept.clear();
        }
    }
    if (value <= 0) {
        return;
    }
    const u64 frames = value == 1 ? u64{60} : static_cast<u64>(value);
    if (frame_tick < frames || frame_tick % (std::min)(frames, u64{30}) != 0) {
        return;
    }
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::SupersededSweep};
    const auto start = std::chrono::steady_clock::now();
    constexpr ImageFlagBits excluded =
        ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
        ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked | ImageFlagBits::IsDecoding |
        ImageFlagBits::AsynchronousDecode | ImageFlagBits::Remapped |
        ImageFlagBits::CollectPending;
    std::vector<ImageId> dead;
    lru_cache.ForEachItemBelow(frame_tick - frames, [&](ImageId image_id) {
        const Image& image = slot_images[image_id];
        if (False(image.flags & ImageFlagBits::GpuModified) ||
            False(image.flags & ImageFlagBits::Registered) || True(image.flags & excluded)) {
            return false;
        }
        ++s.looked;
        if (ColdSuperseded(image_id, image)) {
            dead.push_back(image_id);
        }
        return false;
    });
    const auto queued = [](ImageId image_id, const auto& downloads) {
        return std::ranges::any_of(downloads, [image_id](const auto& download) {
            return download.is_swizzle && download.object_id == image_id;
        });
    };
    const auto bound = [this](const ImageBase& image) {
        return std::ranges::any_of(image.image_view_ids, [this](ImageViewId view) {
            return render_targets.depth_buffer_id == view ||
                   std::ranges::find(render_targets.color_buffer_ids, view) !=
                       render_targets.color_buffer_ids.end();
        });
    };
    for (const ImageId image_id : dead) {
        Image& image = slot_images[image_id];
        if (bound(image) || queued(image_id, uncommitted_downloads) ||
            std::ranges::any_of(committed_downloads,
                                [&](const auto& downloads) { return queued(image_id, downloads); })) {
            ++s.kept_busy;
            continue;
        }
        // FRITCHII_LOST_WRITEBACK: written back first (the landing deletes it), or kept for a
        // later pass while the queue is full.
        const LostWriteback writeback = QueueLostWriteback(image_id, image, LostPathSuperseded);
        if (writeback == LostWriteback::Keep) {
            ++s.kept_busy;
            continue;
        }
        s.deleted_bytes += ImageMemoryBytes(image);
        if (superseded_swept.size() >= 65536) {
            superseded_swept.clear(); // the oldest are the least likely to be read again
        }
        superseded_swept[image.gpu_addr] = frame_tick;
        if (writeback == LostWriteback::Queued) {
            ++s.deleted;
            continue;
        }
        MarkStale(image); // before UntrackImage (see MarkStale)
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
        UnregisterImage(image_id);
        DeleteImage(image_id);
        ++s.deleted;
    }
    ++s.passes;
    s.sweep_ns += NsSince(start);
}

template <class P>
void TextureCache<P>::SweepColdImages() {
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::ColdSweep};
    const u64 cold_frames = TcColdFrames();
    if (cold_frames == 0) {
        return;
    }
    static const bool sweep_guarded = [] {
        const char* value = std::getenv("EDEN_NX2_TC_COLD_GUARDED");
        return value != nullptr && value[0] == '1';
    }();
    // FRITCHII_COLD_SWEEP_MAX (live, default 0 = 16): images deleted a frame at most. zortos
    // #1571 retires up to 512 a frame: in the guard house 7,858 dead targets built up within
    // seconds there. The cap line below says whether 16 is ever reached here.
    const int cap_value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_cold_sweep_max,
                                                           "FRITCHII_COLD_SWEEP_MAX");
    const u32 MaxPerFrame = cap_value > 0 ? static_cast<u32>(cap_value) : 16;
    constexpr u64 MaxWatchFrames = 600; // then a watched image that never got BadOverlap leaves
    constexpr ImageFlagBits excluded =
        ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
        ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked | ImageFlagBits::Converted |
        ImageFlagBits::AsynchronousDecode | ImageFlagBits::IsDecoding | ImageFlagBits::Remapped |
        ImageFlagBits::CostlyLoad | ImageFlagBits::CollectPending;
    const auto queued = [](ImageId image_id, const auto& downloads) {
        return std::ranges::any_of(downloads, [image_id](const auto& download) {
            return download.is_swizzle && download.object_id == image_id;
        });
    };
    const auto bound = [this](const ImageBase& image) {
        return std::ranges::any_of(image.image_view_ids, [this](ImageViewId view) {
            return render_targets.depth_buffer_id == view ||
                   std::ranges::find(render_targets.color_buffer_ids, view) !=
                       render_targets.color_buffer_ids.end();
        });
    };
    // FRITCHII_COVERED_RETIRE=N: a watched GPU-written image untouched for N frames whose memory
    // newer GPU writes cover whole goes now, guarded or not (the guards keep contents a later
    // frame may read; covered ones have none left). Checked once per new overlap, under caps of
    // their own: the pot room makes ~30 targets a frame.
    const int covered_value = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_covered_retire, "FRITCHII_COVERED_RETIRE");
    const u64 covered_frames = covered_value > 0 ? static_cast<u64>(covered_value) : 0;
    constexpr u32 MaxCoveredChecks = 64;
    constexpr u32 MaxCoveredDeletes = 64;
    u32 covered_checks = 0;
    u32 covered_deleted = 0;
    bool covered_capped = false;
    // FRITCHII_COLD_SWEEP_KEEP_RESAMPLED (live, default 1): GPU-written targets at an address and
    // shape the game sampled again after an earlier sweep lost them (cold_resampled) stay, by
    // the regular sweep and by covered retire, within a keep budget of real device memory
    // (FRITCHII_COLD_SWEEP_KEEP_MIB) and only while nothing signals memory pressure (keep_fits).
    const int keep_mode = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_cold_sweep_keep_resampled, "FRITCHII_COLD_SWEEP_KEEP_RESAMPLED", 1);
    // The expected threshold: the collector starts deleting by age above minimum_memory (1.5 GiB),
    // which play exceeds at once on large cards, so that line kept nothing in practice.
    // The critical threshold: at the market the cache runs at 6.8-7.6 GiB during the sweep, above
    // the expected line (6 GiB on a 16 GiB card), so that line kept nothing; the kept targets are
    // small (~40 mostly 1280x720 a run) and above critical the collector's rules apply as before.
    const u64 keep_limit = critical_memory;
    // FRITCHII_COLD_SWEEP_KEEP_MIB (live, MiB; 0 = the critical line above instead): kept
    // targets count against a budget of real device memory, whatever the cache's size. Play went
    // past the critical line (9.7 GiB on a 16 GiB card), where the line released 1,515 of 1,642
    // kept targets; the budget bounds the extra memory and the collector stays the backstop.
    // The budget is resident (cold_keep_resident): an image keeps its share from its first keep
    // until it is deleted, released, relocated or leaves cold_watch, so images kept earlier stay
    // and an overrun turns away new candidates instead.
    // Unset, the default is min(256 MiB, GetDeviceLocalMemory() / 32). GetDeviceLocalMemory is
    // the driver's budget less a reserve, capped at 6 GiB + the resolution scale's GiB, not the
    // heap: ~96 MiB on a 4 GB card, ~200 MiB on 8 GB. With a driver budget of 12 GiB or more it
    // is min(512 MiB, GetDeviceLocalMemory() / 16): on the 16 GB 9070 XT 448 MiB at 1x and
    // 512 MiB at 2x, as the measured peak there (408 images, 115 MiB of guest bytes) is up to
    // ~575 MiB real when rescaled (guest + 4x scaled copy) and 256 MiB would cut it. Not yet
    // measured at that size: the log's "in use" figure against the budget tells.
    if constexpr (requires { runtime.GetDeviceMemoryBudget(); }) {
        if (cold_keep_budget_frame == 0 || frame_tick - cold_keep_budget_frame >= 60) {
            cold_keep_budget_frame = (std::max)(frame_tick, u64{1});
            cold_keep_driver_budget = runtime.GetDeviceMemoryBudget();
        }
    }
    int keep_default_mib = 256;
    if constexpr (HAS_DEVICE_MEMORY_INFO) {
        const u64 device_local = runtime.GetDeviceLocalMemory();
        if (device_local != 0) {
            const bool large = cold_keep_driver_budget >= 12_GiB;
            keep_default_mib = static_cast<int>((std::max)(
                u64{1}, (std::min)(u64{large ? 512u : 256u},
                                   (device_local / (large ? 16 : 32)) >> 20)));
        }
    }
    const int keep_mib = VideoCore::Nx2Debug::SwitchValue(
        VideoCore::Nx2Debug::g_cold_sweep_keep_mib, "FRITCHII_COLD_SWEEP_KEEP_MIB",
        keep_default_mib);
    const u64 keep_budget = keep_mib > 0 ? static_cast<u64>(keep_mib) << 20 : 0;
    // The pressure gate: nothing is kept (the sweep deletes as in v6.2, kept images included)
    // while the driver's budget binds (budget_pressure), for ColdKeepFailedFrames after an image
    // could not be made (the collector's out-of-memory signal, runtime.FailedImages()), or while
    // the cache is past the critical line by a margin of min(max(1 GiB, critical / 4), (driver
    // budget - critical) / 2). The plain critical line is too low: on the 9070 XT play ran the
    // cache at 6.8-9.7 GiB with critical at 7.68 GiB, and that line released 1,515 of 1,642 kept
    // targets; the margin puts the line at ~9.6 GiB there. The second bound keeps the line on
    // the card where critical + 1 GiB would pass it, so it holds without budget_pressure
    // (FRITCHII_VRAM_BUDGET_MODE=0) or failures (WDDM demotes rather than fails). Without a
    // reported budget (no VK_EXT_memory_budget, OpenGL) there is no margin: the plain critical
    // line.
    // Only large cards (a driver budget of 12 GiB or more, as the default budget's test) get the
    // margin; it was made for the 16 GB 9070 XT, where play runs the cache past critical. On
    // smaller cards, in the band above critical, the collector is at its most eager and the
    // thrash guard may have raised its floor (which also shields young covered keeps), so a
    // keep there would hold up to its budget where the collector can't reach it: the plain
    // critical line there, as without a reported budget (low-end PCs first).
    u64 keep_margin = 0;
    if (cold_keep_driver_budget >= 12_GiB && cold_keep_driver_budget > critical_memory) {
        keep_margin = (std::min)((std::max)(static_cast<u64>(1_GiB), critical_memory / 4),
                                 (cold_keep_driver_budget - critical_memory) / 2);
    }
    bool keep_failed = false;
    if constexpr (requires { runtime.FailedImages(); }) {
        // TickFrame records a failure after the sweep: one not yet recorded counts as well.
        keep_failed = runtime.FailedImages() != seen_failed_images;
    }
    keep_failed = keep_failed || (cold_keep_failed_frame != 0 &&
                                  frame_tick - cold_keep_failed_frame < ColdKeepFailedFrames);
    const u64 keep_pressure_line = critical_memory + keep_margin;
    const bool keep_pressure =
        budget_pressure || keep_failed || total_used_memory > keep_pressure_line;
    // Whether a kept image may stay; on false the caller deletes it (after ColdKeepUncharge).
    const auto keep_fits = [&](Image& image) {
        if (keep_pressure) {
            return false;
        }
        if (keep_budget == 0) {
            const u8 covered_bit = image.cold_keep & 4;
            ColdKeepUncharge(image); // the line holds it, not the budget
            image.cold_keep |= covered_bit;
            return total_used_memory < keep_limit;
        }
        // Real device memory: the image plus its scaled copy when rescaled (Vulkan's ScaleUp
        // keeps both), as total_used_memory counts it. Its own earlier share is not counted
        // twice, so a kept image stays unless the budget shrank under what is kept.
        const u64 bytes = (std::max)(
            u64{1}, ImageMemoryBytes(image) +
                        (image.HasScaled() ? GetScaledImageSizeBytes(image) : 0));
        const u64 others =
            cold_keep_resident - (std::min)(cold_keep_resident, image.cold_keep_charge);
        if (others + bytes > keep_budget) {
            return false;
        }
        cold_keep_resident_count += image.cold_keep_charge == 0 ? 1 : 0;
        cold_keep_resident = others + bytes;
        image.cold_keep_charge = bytes;
        return true;
    };
    const bool keep_room = total_used_memory < keep_limit;
    // 3 also keeps targets whose memory overlaps a recently lost read (lost_read_ranges: at most
    // LostReadRing ranges, scanned only for GPU-written candidates the set did not keep).
    const auto keep_rule = [&](Image& image) {
        if (keep_mode <= 0 || False(image.flags & ImageFlagBits::GpuModified)) {
            return false;
        }
        if (ColdKeepResampled(image, keep_mode)) {
            return true;
        }
        if (keep_mode < 3 || lost_read_count == 0 ||
            !OverlapsLostRead(image.cpu_addr, image.guest_size_bytes)) {
            return false;
        }
        if ((image.cold_keep & 2) == 0) {
            image.cold_keep |= 2;
            ++cold_keep_stats.kept_new;
            ++cold_keep_stats.kept_range;
        }
        return true;
    };
    // FRITCHII_COLD_SWEEP_KEEP_COVERED (live, default 1): whether the keep may hold a target that
    // newer images of another shape cover (covered retire) or overlap since its last use
    // (ColdSuperseded). Such a target is not synced with them (they aren't aliases), so a lookup
    // that finds it again samples its pre-cover picture. Off, those go as in v6.2 (no keep, no
    // budget), and only targets no newer image of another shape touched since are kept: their
    // contents are still the memory's. 1 keeps them as well (where the cover is only the images'
    // ranges, not their real writes, that picture is the right one: the pause menu's blocky
    // background was such a target).
    const bool keep_covered =
        VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_cold_sweep_keep_covered,
                                         "FRITCHII_COLD_SWEEP_KEEP_COVERED", 1) > 0;
    // Whether a GPU-written target's address and shape are in the re-sampled set: what mode 1
    // keeps by (no side effects: the rule's own lookup marks the image kept and counts it).
    const auto key_known = [&](const Image& image) {
        return !cold_resampled.empty() &&
               cold_resampled.find(ColdResampledKey(
                   image.gpu_addr, RelocationShapeKey(image.info))) != cold_resampled.end();
    };
    // Counts a target the keep turns away as covered or superseded, when the rule would have
    // kept it by its learned key.
    const auto refuse_newer = [&](const Image& image) {
        if (keep_mode > 0 && True(image.flags & ImageFlagBits::GpuModified) && key_known(image)) {
            ++cold_keep_stats.refused_newer;
        }
    };
    const auto count_kept = [&](Image& image) {
        const u32 frame = static_cast<u32>(frame_tick);
        if (frame_tick % 600 == 0 && image.cold_keep_frame != frame) {
            image.cold_keep_frame = frame;
            ++cold_keep_stats.kept_now;
            cold_keep_stats.kept_now_bytes +=
                ImageMemoryBytes(image) + (image.HasScaled() ? GetScaledImageSizeBytes(image) : 0);
        }
    };
    u32 deleted = 0;
    size_t index = 0;
    while (index < cold_watch.size()) {
        const ImageId image_id = cold_watch[index].first;
        const u64 watched_at = cold_watch[index].second;
        const auto drop = [&] {
            // Out of cold_watch, the keep no longer holds it: its share of the budget goes back.
            if (slot_images.contains(image_id)) {
                ColdKeepUncharge(slot_images[image_id]);
            }
            cold_watch_set.erase(image_id);
            cold_watch[index] = cold_watch.back();
            cold_watch.pop_back();
        };
        if (!slot_images.contains(image_id)) {
            drop();
            continue;
        }
        Image& image = slot_images[image_id];
        if (!RelocationShaped(image.info) || False(image.flags & ImageFlagBits::Registered) ||
            True(image.flags & excluded)) {
            drop();
            continue;
        }
        const bool covered_candidate = covered_frames != 0 &&
                                       True(image.flags & ImageFlagBits::GpuModified) &&
                                       False(image.flags & ImageFlagBits::CollectPending) &&
                                       frame_tick - lru_cache.Tick(image.lru_index) >= covered_frames;
        // Kept by covered retire (cold_keep bit 2): looked at again every frame without a new
        // overlap (its memory was covered; coverage only grows), so the budget and the pressure
        // gate release it; used again (or no longer a candidate), it is ordinary cache contents
        // from here, and its share of the budget goes back (charged again if it is kept again):
        // the budget covers only images that are cold now, or a long session fills it with
        // targets that are alive anyway and turns new candidates away.
        bool covered_kept = (image.cold_keep & 4) != 0;
        if (covered_kept && !covered_candidate) {
            ColdKeepUncharge(image); // clears bit 2 too
            covered_kept = false;
        }
        if (covered_candidate &&
            (covered_kept || image.covered_overlap_frame > image.covered_check_frame)) {
            if ((!covered_kept && covered_checks >= MaxCoveredChecks) ||
                covered_deleted >= MaxCoveredDeletes) {
                covered_capped = true; // looked at in a later frame
            } else {
                covered_checks += covered_kept ? 0 : 1;
                if (covered_kept || CoveredByNewerWrites(image_id, image)) {
                    if (queued(image_id, uncommitted_downloads) ||
                        std::ranges::any_of(committed_downloads, [&](const auto& downloads) {
                            return queued(image_id, downloads);
                        }) ||
                        bound(image)) {
                        ++covered_stats.busy;
                        image.covered_check_frame = image.covered_overlap_frame - 1; // again
                        ++index;
                        continue;
                    }
                    // FRITCHII_COLD_SWEEP_KEEP_RESAMPLED: only here, where covered retire would
                    // delete it (covered retire was what lost them: 28 of 29 cold sweep losses
                    // in a 3-minute market run). Kept, it is looked at here every frame (bit 2).
                    // Covered, so only with FRITCHII_COLD_SWEEP_KEEP_COVERED=1 (above).
                    if (!keep_covered) {
                        refuse_newer(image);
                    }
                    const bool keep = keep_covered && keep_rule(image);
                    if (keep && keep_fits(image)) {
                        image.cold_keep |= 4;
                        count_kept(image);
                        ++index;
                        continue;
                    }
                    ColdKeepUncharge(image); // released (or the rule no longer holds)
                    covered_kept = false;
                    // FRITCHII_COVERED_RETIRE_WRITEBACK (live, default 0 = off): written back to guest memory first, as
                    // the cold sweep does (LostWriteback), so a later sample of this target is not cleared to
                    // zero; deleted when the write-back lands, kept if used again first. Market A/B: cold sweep
                    // losses 29 -> 12-15, but 11-14k write-backs a run for ~20 re-reads; off until a cheaper
                    // selection (only shapes likely to be read again) is measured.
                    if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_covered_retire_writeback,
                                                         "FRITCHII_COVERED_RETIRE_WRITEBACK", 0) != 0) {
                        const LostWriteback wb = QueueLostWriteback(image_id, image, LostPathCold);
                        if (wb == LostWriteback::Queued) {
                            ++covered_deleted;
                            ++covered_stats.sweep_deleted;
                            ++covered_stats.written_back;
                            drop();
                            continue;
                        }
                        if (wb == LostWriteback::Keep) {
                            ++index; // the queue is full: a later frame
                            continue;
                        }
                    }
                    ++covered_deleted;
                    ++covered_stats.sweep_deleted;
                    cold_keep_stats.released += keep ? 1 : 0;
                    RetireCovered(image_id, image);
                    drop();
                    continue;
                }
            }
        }
        if (covered_kept) {
            ++index; // kept by covered retire, its caps reached this frame: looked at next frame
            continue;
        }
        if (False(image.flags & ImageFlagBits::BadOverlap)) {
            // The newest image of a chain has one overlap; the next drift makes it bad.
            if (frame_tick - watched_at > MaxWatchFrames) {
                drop();
            } else {
                ++index;
            }
            continue;
        }
        // Touched (prepared, bound, sampled) in the last K host frames. A kept one is hot again:
        // its share of the keep's budget goes back (charged again if it is kept again).
        if (frame_tick - lru_cache.Tick(image.lru_index) < cold_frames) {
            ColdKeepUncharge(image);
            ++index;
            continue;
        }
        bool superseded = false; // EXP58 chose it
        constexpr u64 SupersededFrames = 60;
        // EDEN_NX2_EXP58=2 (live): images the CPU read once go the same way (below). Owner's
        // Castle Town market + shooting gallery session: 150-170k "kept as read later" a window,
        // ~6,000 images and device memory at the collector's line with EXP58=1.
        const int sweep_mode = VideoCore::Nx2Debug::Experiment(58);
        if (True(image.flags & ImageFlagBits::GpuModified)) {
            // Its contents exist only on the GPU and the collector would not download them
            // either (BadOverlap): only where nothing says they may be read again.
            if (image.relocation_guard_noted != 0 || image.info.forced_flushed ||
                image.info.dma_downloaded) {
                const u64 age = frame_tick - lru_cache.Tick(image.lru_index);
                if (sweep_mode >= 2 && age < SupersededFrames &&
                    frame_tick - watched_at <= MaxWatchFrames) {
                    ++index; // looked at again when old enough
                    continue;
                }
                // A newer target of another shape owns that memory now: a CPU read there goes
                // to it, not to this image.
                if (sweep_mode < 2 || age < SupersededFrames || !ColdSuperseded(image_id, image)) {
                    ++cold_kept_read;
                    drop();
                    continue;
                }
                superseded = true;
            }
            const auto shape = relocation_shapes.find(RelocationShapeKey(image.info));
            const u8 guard = shape != relocation_shapes.end() ? shape->second.guard : 0;
            if (!superseded &&
                ((guard & GuardCpuRead) != 0 ||
                 (!sweep_guarded && (guard & (GuardSampled | GuardLoaded)) != 0))) {
                // EDEN_NX2_EXP58 (live, 0 = off; fable52_answer_2 3.3 step 4): a guarded image
                // still dies once it has been untouched for SupersededFrames and a render target
                // of another shape, touched since, covers its memory: the game reused that memory
                // for another layout, so the old contents have no reader left. Deleted without a
                // download, as the collector would. The guard was made for relocation (same
                // shape, live contents); these piled up (350k "kept" in 600 frames) until the
                // collector's bursts. Shapes the CPU read only with EXP58=2.
                const bool superseded_sweep =
                    sweep_mode != 0 && ((guard & GuardCpuRead) == 0 || sweep_mode >= 2);
                const u64 age = frame_tick - lru_cache.Tick(image.lru_index);
                if (superseded_sweep && age < SupersededFrames &&
                    frame_tick - watched_at <= MaxWatchFrames) {
                    ++index; // looked at again when old enough
                    continue;
                }
                if (!superseded_sweep || age < SupersededFrames ||
                    !ColdSuperseded(image_id, image)) {
                    ++cold_kept_guarded;
                    drop();
                    continue;
                }
                superseded = true;
            }
        }
        if (queued(image_id, uncommitted_downloads) ||
            std::ranges::any_of(committed_downloads,
                                [&](const auto& downloads) { return queued(image_id, downloads); }) ||
            bound(image)) {
            ++index;
            continue;
        }
        if (deleted >= MaxPerFrame) {
            if (covered_frames == 0) {
                break;
            }
            ++index; // FRITCHII_COVERED_RETIRE: the rest still get their coverage checks
            continue;
        }
        // FRITCHII_COLD_SWEEP_KEEP_RESAMPLED (above): only where the image would be deleted
        // (after the GpuModified guards), so guarded images neither use the budget nor stay in
        // cold_watch. Superseded, or overlapped by a newer image of another shape touched since
        // (looked at again every frame while kept), only with FRITCHII_COLD_SWEEP_KEEP_COVERED=1.
        // ColdSuperseded runs only for GPU-written images the rule could keep (mode 1: by their
        // key, a hash lookup first).
        bool keep = false;
        if (keep_covered) {
            keep = keep_rule(image);
        } else if (keep_mode > 0 && True(image.flags & ImageFlagBits::GpuModified)) {
            const bool newer =
                superseded ||
                ((keep_mode >= 2 || key_known(image)) && ColdSuperseded(image_id, image));
            if (newer) {
                refuse_newer(image);
            } else {
                keep = keep_rule(image);
            }
        }
        if (keep && keep_fits(image)) {
            count_kept(image);
            ++index;
            continue;
        }
        ColdKeepUncharge(image); // released (or the rule no longer holds)
        // FRITCHII_LOST_WRITEBACK (default 1): written back to guest memory first, and deleted
        // when that lands. Dropped, an image made there later (the dialogue box's text read
        // again after the UI was hidden: OoT draws it once) had no contents (#1310).
        switch (QueueLostWriteback(image_id, image, LostPathCold)) {
        case LostWriteback::Queued:
            ++deleted;
            cold_superseded += superseded ? 1 : 0;
            cold_keep_stats.released += keep ? 1 : 0;
            drop(); // the landing deletes it; used again first, it stays
            continue;
        case LostWriteback::Keep:
            ++index; // the queue is full: a later frame
            continue;
        case LostWriteback::Drop:
            break;
        }
        // As the collector deletes an image it does not download.
        MarkStale(image); // before UntrackImage (see MarkStale)
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
        UnregisterImage(image_id);
        DeleteImage(image_id);
        ++deleted;
        cold_superseded += superseded ? 1 : 0;
        cold_keep_stats.released += keep ? 1 : 0;
        drop();
    }
    cold_deleted += deleted;
    cold_cap_frames += deleted >= MaxPerFrame ? 1 : 0;
    covered_stats.capped_frames += covered_capped ? 1 : 0;
    if (frame_tick % 600 == 0) {
        auto& c = covered_stats;
        if (covered_frames != 0 || c.checked != 0) {
            LOG_INFO(HW_GPU,
                     "Covered retire (FRITCHII_COVERED_RETIRE={}) last 600 frames: {} checked "
                     "({:.2f} ms), deleted {} by the cold sweep ({} written back first), {} by the collector without a "
                     "download, {} queued downloads dropped as covered ({} MiB in all); {} kept "
                     "as busy, caps reached in {} frames; {} sampled again where one was "
                     "deleted (must stay near 0)",
                     covered_frames, c.checked, static_cast<double>(c.check_ns) / 1e6,
                     c.sweep_deleted, c.written_back, c.collect_deleted, c.landing_deleted,
                     c.deleted_bytes >> 20, c.busy, c.capped_frames, c.sampled_after);
        }
        c = {};
        if (covered_frames == 0) {
            covered_swept.clear();
        }
    }
    if (frame_tick % 600 == 0 && (cold_cap_frames != 0 || cap_value > 0)) {
        LOG_INFO(HW_GPU,
                 "Cold sweep cap (FRITCHII_COLD_SWEEP_MAX={}) last 600 frames: {} a frame at "
                 "most, reached in {} frames, {} watched now",
                 (std::max)(cap_value, 0), MaxPerFrame, cold_cap_frames, cold_watch.size());
    }
    if (frame_tick % 600 == 0) {
        cold_cap_frames = 0;
        LOG_INFO(HW_GPU,
                 "Cold sweep (EDEN_NX2_TC_COLD_FRAMES={}{}) last 600 frames: {} chain images "
                 "deleted ({} of them guarded but superseded, EXP58), {} kept as GPU-written and "
                 "guarded, {} kept as read later; {} watched; {} images, device memory {} MiB "
                 "(expected {} MiB); EXP77 shapes guarded after a sweep: {} so far",
                 cold_frames, sweep_guarded ? ", guarded too" : "", cold_deleted, cold_superseded,
                 cold_kept_guarded, cold_kept_read, cold_watch.size(), slot_images.size(),
                 total_used_memory >> 20, expected_memory >> 20, nx2_sweep_guards);
        cold_deleted = cold_kept_guarded = cold_kept_read = cold_superseded = 0;
        auto& k = cold_keep_stats;
        LOG_INFO(HW_GPU,
                 "Cold sweep keep (FRITCHII_COLD_SWEEP_KEEP_RESAMPLED={}) last 600 frames: {} "
                 "images newly kept by the rule ({} as sampled after their write, {} by range), "
                 "{} kept now ({} MiB), {} deleted all the same as memory was over the line, "
                 "{} with a learned key deleted as newer images cover or overlap them "
                 "(FRITCHII_COLD_SWEEP_KEEP_COVERED={}); "
                 "re-sampled set {} entries (+{}); lost contents sampled {} ({} dropped by the "
                 "cold sweep, {} of those in the set when deleted; {} more images over a drop "
                 "already counted; not recorded {}); device memory {} MiB, keep line {} MiB{}; "
                 "keep budget {} MiB (in use {} MiB by {} images, real device memory, resident), "
                 "release line {} MiB{}",
                 keep_mode, k.kept_new, k.kept_sampled, k.kept_range, k.kept_now,
                 k.kept_now_bytes >> 20, k.released, k.refused_newer, keep_covered ? 1 : 0,
                 cold_resampled.size(), k.learned, k.lost_all,
                 k.lost_cold, k.lost_cold_known, k.lost_cold_repeat, k.lost_not_recorded,
                 total_used_memory >> 20,
                 keep_limit >> 20, keep_room ? "" : " (over the line)", keep_budget >> 20,
                 cold_keep_resident >> 20, cold_keep_resident_count, keep_pressure_line >> 20,
                 !keep_pressure     ? ""
                 : budget_pressure ? ", releasing: budget pressure"
                 : keep_failed     ? ", releasing: an image failed"
                 : keep_margin != 0 ? ", releasing: over critical + margin"
                                    : ", releasing: over critical (no margin on this card)");
        k = {};
        // The image families made in those frames, most made first (fable52_answer_13 sec. 6).
        std::vector<std::pair<std::pair<u32, u32>, const MadeShapes*>> families;
        u64 made_total = 0;
        for (const auto& [key, shapes] : made_shapes) {
            families.emplace_back(key, &shapes);
            made_total += shapes.made;
        }
        if (made_total != 0) {
            std::ranges::sort(families, [](const auto& a, const auto& b) {
                return a.second->made > b.second->made;
            });
            std::string text;
            for (size_t i = 0; i < (std::min)(families.size(), size_t{6}); ++i) {
                const auto& [key, shapes] = families[i];
                const auto [low, high] = std::ranges::minmax(shapes->heights);
                text += fmt::format("{}format {} width {}: {} made, {} heights {}-{}",
                                    i == 0 ? "" : "; ", key.first, key.second, shapes->made,
                                    shapes->heights.size(), low, high);
            }
            LOG_INFO(HW_GPU, "Images made in the last 600 frames: {} in {} (format, width) "
                             "families; most made: {}",
                     made_total, families.size(), text);
        }
        made_shapes.clear();
    }
}

template <class P>
void TextureCache<P>::NoteMadeShape(const ImageInfo& info) {
    auto& shapes = made_shapes[{static_cast<u32>(info.format), info.size.width}];
    ++shapes.made;
    if (shapes.heights.size() < 64 &&
        std::ranges::find(shapes.heights, info.size.height) == shapes.heights.end()) {
        shapes.heights.push_back(info.size.height);
    }
}

template <class P>
void TextureCache<P>::TickRenderClock() {
    // EXP43=4 relocates under this clock (with tombstones).
    const bool enabled =
        Tegra::Nx2RenderClockEnv() != 0 || VideoCore::Nx2Debug::Experiment(43) == 4;
    if (enabled && !render_clock_enabled) {
        Tegra::g_nx2_frame_fence.Resume(); // nothing was noted while it was off
    }
    render_clock_enabled = enabled;
    Tegra::g_nx2_render_clock_wanted.store(enabled, std::memory_order_relaxed);
    const u64 counter = Tegra::g_nx2_render_frame.load(std::memory_order_relaxed);
    render_clock_idle = counter != render_clock_last_counter ? 0 : render_clock_idle + 1;
    render_clock_last_counter = counter;
    // Live while the counter advanced in this host frame or the one before (a count may land in
    // the next tick); two host frames without one and frame_tick takes over again.
    const bool live = enabled && counter != 0 && render_clock_idle < 2;
    if (live != render_clock_live) {
        // Continue from the current value, so the clock never goes back across a switch.
        render_clock_base = RenderFrame() + 1;
        render_clock_source_base = live ? counter : frame_tick;
        render_clock_live = live;
        ++render_clock_switches;
        if (live && !render_clock_logged) {
            render_clock_logged = true;
            LOG_INFO(HW_GPU,
                     "Render frame clock (EDEN_NX2_RENDER_CLOCK): live at host frame {} ({}); "
                     "last_write_frame, cross-frame reads and the same-frame rule follow it",
                     frame_tick,
                     Tegra::Nx2RenderClockSyncpt() >= 0
                         ? fmt::format("syncpoint {}", Tegra::Nx2RenderClockSyncpt())
                     : Tegra::Nx2RenderClockWordSource()
                         ? std::string{"the present poll's learned word"}
                         : std::string{"the flip's frame fence"});
        }
    }
    render_clock_live_frames += render_clock_live ? 1 : 0;
    if (frame_tick % 600 != 0 || (!enabled && !relocation_seen)) {
        return;
    }
    // The shapes refused most often as written this frame.
    std::vector<std::pair<u32, u64>> refused;
    for (auto& [key, shape] : relocation_shapes) {
        if (shape.same_frame_refused != 0) {
            refused.emplace_back(shape.same_frame_refused, key);
            shape.same_frame_refused = 0;
        }
    }
    const size_t top = (std::min)(refused.size(), size_t{3});
    std::partial_sort(refused.begin(), refused.begin() + top, refused.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string refused_text;
    for (size_t i = 0; i < top; ++i) {
        const u64 key = refused[i].second;
        refused_text += fmt::format("{}format {} {}x{} {}", i == 0 ? "" : ", ", key >> 56,
                                    (key >> 18) & 0x3FFFF, key & 0x3FFFF, refused[i].first);
    }
    // Syncpoints signalled in these frames, most first: one at about 1 per host frame may end
    // each frame (EDEN_NX2_RENDER_CLOCK_SYNCPT).
    std::string syncpt_text;
    if (enabled) {
        std::vector<std::pair<u32, u32>> signals;
        for (u32 id = 0; id < Tegra::Nx2SyncpointSignalSlots; ++id) {
            const u32 count =
                Tegra::g_nx2_syncpoint_signals[id].exchange(0, std::memory_order_relaxed);
            if (count != 0) {
                signals.emplace_back(count, id);
            }
        }
        std::sort(signals.rbegin(), signals.rend());
        for (size_t i = 0; i < (std::min)(signals.size(), size_t{4}); ++i) {
            syncpt_text += fmt::format(" {}:{:.2f}", signals[i].second, signals[i].first / 600.0);
        }
    }
    // The flip-fence source's confirmations (video_core/nx2_frame_fence.h): equal and ahead are
    // exact boundaries, late ones came at the flip (as frame_tick), early must stay 0.
    std::string fence_text;
    if (enabled && Tegra::Nx2RenderClockFlipSource()) {
        const auto fence = Tegra::g_nx2_frame_fence.TakeWindow();
        const auto per_frame = [](u32 count) { return count / 600.0; };
        fence_text = fmt::format(
            "; frame fence {} {}: {} flips ({:.2f} per host frame), {} equal, {} ahead, {} late "
            "({} merged), {} early{}; bumps {} armed, {} flip, {} late; arms refused {} old, {} "
            "not largest, {} reached, {} dropped, {} held, {} relearns",
            fence.syncpoint == ~0U ? std::string{"none"}
                                   : fmt::format("syncpoint {}", fence.syncpoint),
            fence.slot_key == ~0ULL ? std::string{"(no slot armed it)"}
                                    : fmt::format("(session {} slot {:#x})", fence.slot_key >> 32,
                                                  static_cast<u32>(fence.slot_key)),
            fence.flips, per_frame(fence.flips), fence.equal, fence.ahead, fence.late,
            fence.merged, fence.early, fence.fault ? ", armed source off" : "",
            fence.bumps_armed, fence.bumps_flip, fence.bumps_late, fence.refused_old,
            fence.refused_not_largest, fence.refused_reached, fence.dropped, fence.held,
            fence.relearns);
    }
    const u64 counted = counter - render_clock_window_counter;
    render_clock_window_counter = counter;
    LOG_INFO(HW_GPU,
             "Render frame clock (EDEN_NX2_RENDER_CLOCK={}) last 600 frames: {} render frames "
             "({:.2f} per host frame), live {} of 600 frames, {} switches; {} relocation "
             "candidates refused as written this frame{}{}; syncpoint signals per host frame:{}{}",
             enabled ? 1 : 0, counted, counted / 600.0, render_clock_live_frames,
             render_clock_switches, relocation_same_frame, top != 0 ? " (" : "",
             top != 0 ? refused_text + ")" : std::string{},
             syncpt_text.empty() ? std::string{" none"} : syncpt_text, fence_text);
    render_clock_live_frames = 0;
    render_clock_switches = 0;
    relocation_same_frame = 0;
}

template <class P>
const typename P::ImageView& TextureCache<P>::GetImageView(ImageViewId id) const noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::ImageView& TextureCache<P>::GetImageView(ImageViewId id) noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::ImageView& TextureCache<P>::GetImageView(u32 index) noexcept {
    // Not compute!
    const auto image_view_id = VisitImageView(index, false);
    return slot_image_views[image_view_id];
}

template <class P>
void TextureCache<P>::MarkModification(ImageId id) noexcept {
    MarkModification(slot_images[id]);
}

template <class P>
void TextureCache<P>::FillImageViews(std::span<ImageViewInOut> views, bool compute, bool blacklist) {
    bool has_blacklisted = false;
    do {
        has_deleted_images = false;
        if (blacklist) {
            has_blacklisted = false;
        }
        // EDEN_NX2_LIGHT_TABLE_ARRAYS only: an element repeating the one
        // before (a handle table holding one shadow map in many elements, say) is the same view:
        // visiting it again only prepares it again. Not once this pass has deleted image views
        // (scaling an image down below does): that view may have gone with them.
        const bool reuse_repeats{VideoCore::Nx2Debug::LightTableArrays()};
        const ImageViewInOut* previous{};
        for (ImageViewInOut& view : views) {
            view.id = reuse_repeats && previous != nullptr && previous->index == view.index &&
                              !has_deleted_images
                          ? previous->id
                          : VisitImageView(view.index, compute);
            previous = &view;
            if (blacklist) {
                if (view.blacklist && view.id != NULL_IMAGE_VIEW_ID) {
                    const ImageViewBase& image_view = slot_image_views[view.id];
                    if (image_view.image_id == NULL_IMAGE_ID) {
                        continue;
                    }
                    auto& image = slot_images[image_view.image_id];
                    has_blacklisted |= ScaleDown(image);
                    image.scale_rating = 0;
                }
            }
        }
    } while (has_deleted_images || (blacklist && has_blacklisted));
}

template <class P>
void TextureCache<P>::CheckFeedbackLoop(std::span<const ImageViewInOut> views) {
    if (!Settings::values.barrier_feedback_loops.GetValue()) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    const bool writes_depth_stencil = regs.depth_write_enabled != 0 || regs.stencil_enable != 0;
    const auto barrier = [&] {
        if (feedback_depth_dirty) {
            runtime.BarrierFeedbackLoop();
            feedback_depth_dirty = false;
        }
    };
    SCOPE_EXIT {
        if (writes_depth_stencil) {
            feedback_depth_dirty = true;
        }
    };

    if (render_targets_serial == last_feedback_loop_serial &&
        texture_bindings_serial == last_feedback_texture_serial) {
        if (last_feedback_loop_result) {
            barrier();
        }
        return;
    }
    if (render_targets_serial != last_feedback_loop_serial) {
        feedback_depth_dirty = true;
    }

    if (rt_active_mask == 0) {
        last_feedback_loop_serial = render_targets_serial;
        last_feedback_texture_serial = texture_bindings_serial;
        last_feedback_loop_result = false;
        return;
    }
    const u32 depth_bit = 1u << NUM_RT;
    const bool depth_active = (rt_active_mask & depth_bit) != 0;

    const bool requires_barrier = [&] {
        for (const auto& view : views) {
            if (!view.id) {
                continue;
            }

            {
                bool is_continue = false;
                for (size_t i = 0; i < 8; ++i)
                    is_continue |= (rt_active_mask & (1u << i)) && view.id == render_targets.color_buffer_ids[i];
                if (is_continue)
                    continue;
            }

            if (depth_active && view.id == render_targets.depth_buffer_id)
                continue;

            const ImageId view_image_id = slot_image_views[view.id].image_id;
            {
                bool is_continue = false;
                for (size_t i = 0; i < 8; ++i)
                    is_continue |= (rt_active_mask & (1u << i)) && view_image_id == rt_image_id[i];
                if (is_continue)
                    continue;
            }
            if (depth_active && view_image_id == rt_depth_image_id) {
                return true;
            }
        }

        return false;
    }();

    last_feedback_loop_serial = render_targets_serial;
    last_feedback_texture_serial = texture_bindings_serial;
    last_feedback_loop_result = requires_barrier;
    if (requires_barrier) {
        barrier();
    }
}

template <class P>
bool TextureCache<P>::SamplesRenderTarget(std::span<const ImageViewInOut> views) {
    if (rt_active_mask == 0) {
        return false;
    }
    const bool depth_active = (rt_active_mask & (1u << NUM_RT)) != 0;
    for (const auto& view : views) {
        if (!view.id) {
            continue;
        }
        const ImageId image_id = slot_image_views[view.id].image_id;
        if (depth_active && image_id == rt_depth_image_id) {
            return true;
        }
        for (size_t i = 0; i < NUM_RT; ++i) {
            if ((rt_active_mask & (1u << i)) != 0 && image_id == rt_image_id[i]) {
                return true;
            }
        }
    }
    return false;
}

template <class P>
typename P::Sampler* TextureCache<P>::GetSampler(u32 index, bool compute) {
    return &slot_samplers[GetSamplerId(index, compute)];
}

template <class P>
SamplerId TextureCache<P>::GetSamplerId(u32 index, bool compute) {
    auto& table = compute ? channel_state->compute_sampler_table : channel_state->graphics_sampler_table;
    if (index > table.current_limit) {
        LOG_TRACE(HW_GPU, "Invalid sampler index={}", index);
        return NULL_SAMPLER_ID;
    }
    auto const [descriptor, is_new] = table.Read(*gpu_memory, index);
    if (is_new) {
        auto const id = FindSampler(descriptor, compute);
        table.bound_ids[index] = id;
        return id;
    }
    return table.bound_ids[index];
}

template <class P>
typename TextureCache<P>::TextureMemoState TextureCache<P>::GraphicsTextureMemoState()
    const noexcept {
    const auto& image_table = channel_state->graphics_image_table;
    const auto& sampler_table = channel_state->graphics_sampler_table;
    return TextureMemoState{
        .channel = channel_state,
        .memory = gpu_memory,
        .tic_pool = image_table.current_gpu_addr,
        .tsc_pool = sampler_table.current_gpu_addr,
        .tic_limit = image_table.current_limit,
        .tsc_limit = sampler_table.current_limit,
        .map_serial = gpu_memory->MapSerial(),
        .epoch = texture_memo_epoch,
    };
}

template <class P>
bool TextureCache<P>::TakeTextureMemoEntry(u32 tic_handle, u32 tsc_handle, ImageViewId view,
                                           SamplerId sampler, TextureMemoEntry& entry) const {
    // The entry must be in the table as this draw read it, and still give the resolved id
    // (as the original RecordBinding checks).
    const auto read_entry = [](const auto& table, u32 index, auto id) {
        return !table.invalidated && index / 64 < table.read_descriptors.size() &&
               index < table.descriptors.size() &&
               (table.read_descriptors[index / 64] & (1ULL << (index % 64))) != 0 &&
               table.bound_ids[index] == id;
    };
    const auto& image_table = channel_state->graphics_image_table;
    const auto& sampler_table = channel_state->graphics_sampler_table;
    const u32 header_index = Tegra::Texture::TextureHeaderIndex(
        tic_handle, image_table.current_limit, gpu_memory->UsesNativeTiling());
    if (header_index <= image_table.current_limit) {
        if (!read_entry(image_table, header_index, view)) {
            return false;
        }
        entry.tic_index = header_index;
        entry.tic = image_table.descriptors[header_index];
    } else if (view != NULL_IMAGE_VIEW_ID) {
        return false;
    } else {
        entry.tic_index = ~0U;
        entry.tic = TICEntry{};
    }
    if (tsc_handle <= sampler_table.current_limit) {
        if (!read_entry(sampler_table, tsc_handle, sampler)) {
            return false;
        }
        entry.tsc_index = tsc_handle;
        entry.tsc = sampler_table.descriptors[tsc_handle];
    } else if (sampler != NULL_SAMPLER_ID) {
        return false;
    } else {
        entry.tsc_index = ~0U;
        entry.tsc = TSCEntry{};
    }
    return true;
}

template <class P>
bool TextureCache<P>::TextureMemoEntryCurrent(const TextureMemoState& state,
                                              const TextureMemoEntry& entry) const {
    const auto same = [this](GPUVAddr gpu_addr, const auto& expected) {
        using T = std::remove_cvref_t<decltype(expected)>;
        T guest{};
        const bool one_page =
            (gpu_addr & Core::Memory::YUZU_PAGEMASK) + sizeof(T) <= Core::Memory::YUZU_PAGESIZE;
        if (const u8* const host = one_page ? gpu_memory->GetPointerForDescriptor(gpu_addr)
                                            : nullptr) {
            std::memcpy(&guest, host, sizeof(T));
        } else {
            gpu_memory->ReadBlockUnsafe(gpu_addr, &guest, sizeof(T));
        }
        return std::memcmp(&guest, &expected, sizeof(T)) == 0;
    };
    if (entry.tic_index != ~0U &&
        !same(state.tic_pool + u64{entry.tic_index} * sizeof(TICEntry), entry.tic)) {
        return false;
    }
    if (entry.tsc_index != ~0U &&
        !same(state.tsc_pool + u64{entry.tsc_index} * sizeof(TSCEntry), entry.tsc)) {
        return false;
    }
    return true;
}

template <class P>
bool TextureCache<P>::TextureMemoViewClean(ImageViewId view) const noexcept {
    if (!view || view == NULL_IMAGE_VIEW_ID) {
        // VisitImageView prepares nothing for the null view.
        return true;
    }
    const ImageViewBase& image_view = slot_image_views[view];
    if (image_view.IsBuffer()) {
        return true;
    }
    // What PrepareImage(id, false, false) acts on besides the LRU: RefreshContents and
    // SynchronizeAliases.
    const ImageBase& image = slot_images[image_view.image_id];
    return False(image.flags & ImageFlagBits::CpuModified) && image.aliased_images.empty();
}

template <class P>
void TextureCache<P>::TouchTextureMemoView(ImageViewId view) noexcept {
    if (!view || view == NULL_IMAGE_VIEW_ID) {
        return;
    }
    const ImageViewBase& image_view = slot_image_views[view];
    if (image_view.IsBuffer()) {
        return;
    }
    ImageBase& image = slot_images[image_view.image_id];
    CheckCrossFrameUse(image, false, false); // as PrepareImage(id, false, false) would
    TouchImageLru(image);
}

template <class P>
typename TextureCache<P>::TexBankPage* TextureCache<P>::TexBankWatch(DAddr page,
                                                                    bool& created) {
    if (const auto it = tex_bank_pages.find(page); it != tex_bank_pages.end()) {
        created = false;
        return &it->second;
    }
    created = true;
    // A bound on the trapped pages (handle words and pool entries of the draws seen lately).
    if (tex_bank_pages.size() >= 4096) {
        TexBankReset();
    }
    // Trapped from now on. Not before: the entry's first look moves its generation (recheck).
    device_memory.UpdatePagesCachedCount(page, Core::DEVICE_PAGESIZE, 1);
    TexBankPage& entry = tex_bank_pages[page];
    entry.page = page;
    return &entry;
}

template <class P>
void TextureCache<P>::TexBankReset() {
    if (tex_bank_pages.empty()) {
        return;
    }
    for (const auto& [page, entry] : tex_bank_pages) {
        device_memory.UpdatePagesCachedCount(page, Core::DEVICE_PAGESIZE, -1);
    }
    tex_bank_pages.clear();
    ++tex_bank_epoch;
}

template <class P>
bool TextureCache<P>::PrepareTexBankViews(std::span<const ImageViewId> views) {
    has_deleted_images = false;
    for (const ImageViewId view : views) {
        if (view != NULL_IMAGE_VIEW_ID) {
            PrepareImageView(view, false, false);
        }
    }
    return !has_deleted_images;
}

template <class P>
DrawTemplateContext TextureCache<P>::GraphicsTextureContext(u64 owner) const noexcept {
    const auto& image_table = channel_state->graphics_image_table;
    const auto& sampler_table = channel_state->graphics_sampler_table;
    return {
        .owner = owner,
        .memory = reinterpret_cast<uintptr_t>(gpu_memory),
        .image_table = reinterpret_cast<uintptr_t>(&image_table),
        .tic_address = image_table.current_gpu_addr,
        .tsc_address = sampler_table.current_gpu_addr,
        .tic_limit = image_table.current_limit,
        .tsc_limit = sampler_table.current_limit,
        .via_header_index =
            maxwell3d->regs.sampler_binding ==
                    Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding
                ? 1U
                : 0U,
        .native_tiling = gpu_memory->UsesNativeTiling() ? 1U : 0U,
    };
}

template <class P>
bool TextureCache<P>::KeepTemplateTextures(std::span<const u32> handles,
                                           std::span<const ImageViewInOut> views,
                                           std::span<const SamplerId> samplers,
                                           std::vector<DrawTemplateTexture>& textures) {
    textures.clear();
    const auto& image_table = channel_state->graphics_image_table;
    const auto& sampler_table = channel_state->graphics_sampler_table;
    if (image_table.invalidated || sampler_table.invalidated || handles.size() != views.size() ||
        handles.size() != samplers.size()) {
        return false;
    }
    const bool via_header_index = maxwell3d->regs.sampler_binding ==
                                  Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding;
    const bool native_tiling = gpu_memory->UsesNativeTiling();
    const auto host_of = [&](GPUVAddr address, size_t size) -> const u8* {
        if ((address & Core::Memory::YUZU_PAGEMASK) + size > Core::Memory::YUZU_PAGESIZE) {
            return nullptr;
        }
        return gpu_memory->GetPointer(address);
    };
    for (size_t i = 0; i < handles.size(); ++i) {
        const auto [view_index, sampler_index] =
            Tegra::Texture::TexturePair(handles[i], via_header_index);
        const u32 tic_index = Tegra::Texture::TextureHeaderIndex(
            view_index, image_table.current_limit, native_tiling);
        // A null view can stand for an image that failed to allocate, which the next read of the
        // descriptor after TickFrame's collection makes again: a template would keep it null.
        if (!views[i].id || views[i].id == NULL_IMAGE_VIEW_ID || !samplers[i] ||
            tic_index > image_table.current_limit ||
            sampler_index > sampler_table.current_limit) {
            return false;
        }
        const DrawTemplateTexture& texture = textures.emplace_back(DrawTemplateTexture{
            .tic_host = host_of(image_table.current_gpu_addr + tic_index * sizeof(TICEntry),
                                sizeof(TICEntry)),
            .tsc_host = host_of(sampler_table.current_gpu_addr + sampler_index * sizeof(TSCEntry),
                                sizeof(TSCEntry)),
            .tic = image_table.descriptors[tic_index],
            .tsc = sampler_table.descriptors[sampler_index],
            .view = views[i].id,
            .view_generation = ImageViewGeneration(views[i].id),
            .sampler = samplers[i],
        });
        if (texture.tic_host == nullptr || texture.tsc_host == nullptr ||
            std::memcmp(texture.tic_host, &texture.tic, sizeof(texture.tic)) != 0 ||
            std::memcmp(texture.tsc_host, &texture.tsc, sizeof(texture.tsc)) != 0) {
            return false;
        }
    }
    return true;
}

template <class P>
bool TextureCache<P>::PrepareTemplateViews(std::span<const ImageViewInOut> views) {
    has_deleted_images = false;
    for (const ImageViewInOut& view : views) {
        PrepareImageView(view.id, false, false);
    }
    return !has_deleted_images;
}

template <class P>
std::optional<DAddr> TextureCache<P>::TexBankDevicePage(GPUVAddr gpu_addr) const {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return std::nullopt;
    }
    return Common::AlignDown(*device_addr, Core::DEVICE_PAGESIZE);
}

template <class P>
const typename P::Sampler& TextureCache<P>::GetSampler(SamplerId id) const noexcept {
    return slot_samplers[id];
}

template <class P>
typename P::Sampler& TextureCache<P>::GetSampler(SamplerId id) noexcept {
    return slot_samplers[id];
}

template <class P>
void TextureCache<P>::SynchronizeDescriptors(bool compute) {
    if (compute) {
        const bool linked_tsc = kepler_compute->launch_description.linked_tsc;
        const u32 tic_limit = kepler_compute->regs.tic.limit;
        const u32 tsc_limit = linked_tsc ? tic_limit : kepler_compute->regs.tsc.limit;
        bool bindings_changed = false;
        if (channel_state->compute_sampler_table.Synchronize(kepler_compute->regs.tsc.Address(), tsc_limit))
            bindings_changed = true;
        if (channel_state->compute_image_table.Synchronize(kepler_compute->regs.tic.Address(), tic_limit))
            bindings_changed = true;
        if (bindings_changed) {
            ++texture_bindings_serial;
            ++bindings_epoch;
        }
    } else {
        const bool linked_tsc = maxwell3d->regs.sampler_binding == Tegra::Engines::Maxwell3D::Regs::SamplerBinding::ViaHeaderBinding;
        const u32 tic_limit = maxwell3d->regs.tex_header.limit;
        const u32 tsc_limit = linked_tsc ? tic_limit : maxwell3d->regs.tex_sampler.limit;
        bool bindings_changed = false;
        if (channel_state->graphics_sampler_table.Synchronize(maxwell3d->regs.tex_sampler.Address(), tsc_limit))
            bindings_changed = true;
        if (channel_state->graphics_image_table.Synchronize(maxwell3d->regs.tex_header.Address(), tic_limit))
            bindings_changed = true;
        if (bindings_changed) {
            ++texture_bindings_serial;
            ++bindings_epoch;
        }
    }
}

template <class P>
bool TextureCache<P>::RescaleRenderTargets() {
    auto& flags = maxwell3d->dirty.flags;
    u32 scale_rating = 0;
    bool rescaled = false;
    std::array<ImageId, NUM_RT> tmp_color_images{};
    ImageId tmp_depth_image{};
    do {
        flags[Dirty::RenderTargets] = false;

        has_deleted_images = false;
        // Render target control is used on all render targets, so force look ups when this one is
        // up
        const bool force = flags[Dirty::RenderTargetControl];
        flags[Dirty::RenderTargetControl] = false;

        scale_rating = 0;
        bool any_rescaled = false;
        bool can_rescale = true;
        const auto check_rescale = [&](ImageViewId view_id, ImageId& id_save) {
            if (view_id != NULL_IMAGE_VIEW_ID && view_id != ImageViewId{}) {
                const auto& view = slot_image_views[view_id];
                const auto image_id = view.image_id;
                id_save = image_id;
                auto& image = slot_images[image_id];
                can_rescale &= ImageCanRescale(image);
                any_rescaled |= True(image.flags & ImageFlagBits::Rescaled) ||
                                GetFormatType(image.info.format) != SurfaceType::ColorTexture;
                scale_rating = std::max<u32>(scale_rating, image.scale_tick <= frame_tick
                                                               ? image.scale_rating + 1U
                                                               : image.scale_rating);
            } else {
                id_save = CORRUPT_ID;
            }
        };
        for (size_t index = 0; index < NUM_RT; ++index) {
            ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
            if (flags[Dirty::ColorBuffer0 + index] || force) {
                flags[Dirty::ColorBuffer0 + index] = false;
                BindRenderTarget(&color_buffer_id, FindColorBuffer(index));
            }
            check_rescale(color_buffer_id, tmp_color_images[index]);
        }
        if (flags[Dirty::ZetaBuffer] || force) {
            flags[Dirty::ZetaBuffer] = false;
            BindRenderTarget(&render_targets.depth_buffer_id, FindDepthBuffer());
        }
        check_rescale(render_targets.depth_buffer_id, tmp_depth_image);
        // EXP117 (live, 0 = off): a square depth target with no colour target is a shadow map.
        // Scaling it does not sharpen the picture; it multiplies the map's memory by the square of
        // the scale (OoT's 12288x12288 sun shadow map: 648 MiB at 1.5x). A full-screen depth
        // prepass is not square, so it still scales.
        if (can_rescale && tmp_depth_image != CORRUPT_ID &&
            VideoCore::Nx2Debug::Experiment(117) != 0 &&
            std::ranges::all_of(tmp_color_images, [](ImageId id) { return id == CORRUPT_ID; })) {
            const Extent3D& size = slot_images[tmp_depth_image].info.size;
            can_rescale = size.width != size.height;
        }

        if (can_rescale) {
            rescaled = any_rescaled || scale_rating >= 2;
            const auto scale_up = [this](ImageId image_id) {
                if (image_id != CORRUPT_ID) {
                    Image& image = slot_images[image_id];
                    ScaleUp(image);
                }
            };
            if (rescaled) {
                for (size_t index = 0; index < NUM_RT; ++index) {
                    scale_up(tmp_color_images[index]);
                }
                scale_up(tmp_depth_image);
                scale_rating = 2;
            }
        } else {
            rescaled = false;
            const auto scale_down = [this](ImageId image_id) {
                if (image_id != CORRUPT_ID) {
                    Image& image = slot_images[image_id];
                    ScaleDown(image);
                }
            };
            for (size_t index = 0; index < NUM_RT; ++index) {
                scale_down(tmp_color_images[index]);
            }
            scale_down(tmp_depth_image);
            scale_rating = 1;
        }
    } while (has_deleted_images);
    const auto set_rating = [this, scale_rating](ImageId image_id) {
        if (image_id != CORRUPT_ID) {
            Image& image = slot_images[image_id];
            image.scale_rating = scale_rating;
            if (image.scale_tick <= frame_tick) {
                image.scale_tick = frame_tick + 1;
            }
        }
    };
    for (size_t index = 0; index < NUM_RT; ++index) {
        set_rating(tmp_color_images[index]);
    }
    set_rating(tmp_depth_image);

    return rescaled;
}

template <class P>
void TextureCache<P>::UpdateRenderTargets(bool is_clear) {
    VideoCommon::OneFollower::RenderOwned("TextureCache::UpdateRenderTargets");
    using namespace VideoCommon::Dirty;
    struct ClearingScope {
        bool& flag;
        ~ClearingScope() {
            flag = false;
        }
    } clearing_scope{clearing_render_targets};
    clearing_render_targets = is_clear;
    // EDEN_NX2_DEPTH_CENSUS: the depth target bound once this update is done, on every path out.
    struct CensusScope {
        TextureCache& cache;
        bool is_clear;
        ~CensusScope() {
            if (cache.depth_census_on) [[unlikely]] {
                cache.CensusNoteRenderTargets(is_clear);
            }
        }
    } census_scope{*this, is_clear};
    auto& flags = maxwell3d->dirty.flags;
    if (render_targets_out_of_memory) [[unlikely]] {
        // A target had no memory at the last update: look every target up again, even if
        // something cleared the dirty flags since.
        flags[Dirty::RenderTargets] = true;
        flags[Dirty::RenderTargetControl] = true;
    }
    if (!flags[Dirty::RenderTargets]) {
        // EDEN_NX2_EXP31 (RT_FAST): bit 1 = skip the loop below while it
        // would change nothing, bit 2 = run it anyway and check that it changed nothing.
        const int rt_fast = VideoCore::Nx2Debug::Experiment(31);
        bool verify = false;
        if (rt_fast != 0 && !is_clear) {
            ++rt_fast_clean_paths;
            if ((rt_fast & 1) != 0 && RenderTargetsStillPrepared()) {
                if ((rt_fast & 2) == 0) {
                    ++rt_fast_skips;
                    return;
                }
                verify = true;
            }
        }
        if (verify) {
            if (!VerifyRenderTargetsPrepared()) {
                rt_prepared.valid = false;
                return;
            }
            NoteRenderTargetsPrepared();
            return;
        }
        for (size_t index = 0; index < NUM_RT; ++index) {
            ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
            PrepareImageView(color_buffer_id, true, is_clear && IsFullClear(color_buffer_id));
        }
        const ImageViewId depth_buffer_id = render_targets.depth_buffer_id;
        PrepareImageView(depth_buffer_id, true, is_clear && IsFullClear(depth_buffer_id));
        if (is_clear) {
            NoteFullClears();
            rt_prepared.valid = false;
        } else if ((rt_fast & 1) != 0) {
            NoteRenderTargetsPrepared();
        }
        return;
    }
    rt_prepared.valid = false;

    const VideoCommon::RenderTargets previous_render_targets = render_targets;
    const bool rescaled = RescaleRenderTargets();
    if (is_rescaling != rescaled) {
        flags[Dirty::RescaleViewports] = true;
        flags[Dirty::RescaleScissors] = true;
        is_rescaling = rescaled;
    }

    for (size_t index = 0; index < NUM_RT; ++index) {
        ImageViewId& color_buffer_id = render_targets.color_buffer_ids[index];
        PrepareImageView(color_buffer_id, true, is_clear && IsFullClear(color_buffer_id));
    }
    const ImageViewId depth_buffer_id = render_targets.depth_buffer_id;

    PrepareImageView(depth_buffer_id, true, is_clear && IsFullClear(depth_buffer_id));
    if (is_clear) {
        NoteFullClears();
    }

    rt_active_mask = 0;
    rt_image_id = {};
    for (size_t i = 0; i < rt_image_id.size(); ++i) {
        if (ImageViewId const view = render_targets.color_buffer_ids[i]; view) {
            rt_active_mask |= 1u << i;
            rt_image_id[i] = slot_image_views[view].image_id;
        }
    }
    if (depth_buffer_id) {
        rt_active_mask |= (1u << NUM_RT);
        rt_depth_image_id = slot_image_views[depth_buffer_id].image_id;
    } else {
        rt_depth_image_id = ImageId{};
    }

    for (size_t index = 0; index < NUM_RT; ++index) {
        render_targets.draw_buffers[index] = static_cast<u8>(maxwell3d->regs.rt_control.Map(index));
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    if (is_rescaling) {
        up_scale = Settings::values.resolution_info.up_scale;
        down_shift = Settings::values.resolution_info.down_shift;
    }
    render_targets.size = Extent2D{
        (maxwell3d->regs.surface_clip.width * up_scale) >> down_shift,
        (maxwell3d->regs.surface_clip.height * up_scale) >> down_shift,
    };
    render_targets.is_rescaled = is_rescaling;

    if (render_targets != previous_render_targets) {
        ++render_targets_serial;
    }

    // CheezyEmu a08015f893: a target that got no memory (FindRenderTargetView gives the null
    // view only when its image could not be allocated; unused slots are ImageViewId{}). Its
    // draws and clears are skipped, and every target is looked up again at the next one, by
    // when the collector may have freed some. Never set while allocations succeed.
    render_targets_out_of_memory =
        render_targets.depth_buffer_id == NULL_IMAGE_VIEW_ID ||
        std::ranges::any_of(render_targets.color_buffer_ids,
                            [](ImageViewId id) { return id == NULL_IMAGE_VIEW_ID; });
    if (render_targets_out_of_memory) [[unlikely]] {
        flags[Dirty::RenderTargets] = true;
        flags[Dirty::RenderTargetControl] = true; // forces every target's lookup
        static u32 logged{};
        if (logged++ < 8) {
            LOG_ERROR(HW_GPU, "A render target has no memory: draws into it are skipped");
        }
    }

    flags[Dirty::DepthBiasGlobal] = true;
}

template <class P>
bool TextureCache<P>::RenderTargetsStillPrepared() const {
    const auto& prepared = rt_prepared;
    if (!prepared.valid || prepared.serial != render_targets_serial ||
        prepared.modification_tick != modification_tick || prepared.frame_tick != frame_tick ||
        prepared.bindings_epoch != bindings_epoch) {
        return false;
    }
    for (size_t index = 0; index <= NUM_RT; ++index) {
        const ImageViewId id = index < NUM_RT ? render_targets.color_buffer_ids[index]
                                              : render_targets.depth_buffer_id;
        if (id != prepared.views[index]) {
            return false;
        }
        if (!id) {
            continue;
        }
        const ImageViewBase& image_view = slot_image_views[id];
        if (image_view.IsBuffer()) {
            continue;
        }
        // What PrepareImage(is_modification, !invalidate) would act on.
        const ImageBase& image = slot_images[image_view.image_id];
        if (False(image.flags & ImageFlagBits::GpuModified) ||
            True(image.flags & (ImageFlagBits::CpuModified | ImageFlagBits::SparseTiles |
                                ImageFlagBits::SparseLayers | ImageFlagBits::UnwrittenSparse)) ||
            !image.aliased_images.empty()) {
            return false;
        }
    }
    // MarkModification would give them the newest ticks again, in the same order: they have them,
    // nothing was marked since. LRU touches within a frame are no-ops.
    return true;
}

template <class P>
void TextureCache<P>::NoteRenderTargetsPrepared() {
    auto& prepared = rt_prepared;
    prepared.valid = true;
    prepared.serial = render_targets_serial;
    prepared.modification_tick = modification_tick;
    prepared.frame_tick = frame_tick;
    prepared.bindings_epoch = bindings_epoch;
    for (size_t index = 0; index < NUM_RT; ++index) {
        prepared.views[index] = render_targets.color_buffer_ids[index];
    }
    prepared.views[NUM_RT] = render_targets.depth_buffer_id;
}

template <class P>
bool TextureCache<P>::VerifyRenderTargetsPrepared() {
    // The images the clean path prepares, with their flags before it runs.
    std::array<ImageId, NUM_RT + 1> images{};
    std::array<ImageFlagBits, NUM_RT + 1> flags_before{};
    u64 marked = 0;
    for (size_t index = 0; index <= NUM_RT; ++index) {
        const ImageViewId id = index < NUM_RT ? render_targets.color_buffer_ids[index]
                                              : render_targets.depth_buffer_id;
        if (!id || slot_image_views[id].IsBuffer()) {
            continue;
        }
        images[index] = slot_image_views[id].image_id;
        flags_before[index] = slot_images[images[index]].flags;
        ++marked;
    }
    const u64 tick_before = modification_tick;
    const u64 epoch_before = bindings_epoch;
    for (size_t index = 0; index < NUM_RT; ++index) {
        PrepareImageView(render_targets.color_buffer_ids[index], true, false);
    }
    PrepareImageView(render_targets.depth_buffer_id, true, false);
    ++rt_fast_verified;
    // Skipping it would have left the ticks alone: MarkModification once per target is all it
    // may have done.
    bool same = modification_tick - tick_before == marked && bindings_epoch == epoch_before;
    size_t first_difference = NUM_RT + 1;
    for (size_t index = 0; index <= NUM_RT && same; ++index) {
        if (images[index] && slot_images[images[index]].flags != flags_before[index]) {
            same = false;
            first_difference = index;
        }
    }
    if (!same) {
        ++rt_fast_mismatches;
        static u32 logged = 0;
        if (logged < 32) {
            ++logged;
            const ImageId image =
                first_difference <= NUM_RT ? images[first_difference] : ImageId{};
            LOG_WARNING(HW_GPU,
                        "RT_FAST (EXP31) mismatch: the clean path changed what the skip assumed: "
                        "modification tick +{} for {} targets, epoch {} -> {}, target {} flags "
                        "{:#x} -> {:#x}",
                        modification_tick - tick_before, marked, epoch_before, bindings_epoch,
                        first_difference,
                        image ? static_cast<u32>(flags_before[first_difference]) : 0U,
                        image ? static_cast<u32>(slot_images[image].flags) : 0U);
        }
    }
    return same;
}

template <class P>
typename P::Framebuffer* TextureCache<P>::GetFramebuffer() {
    if (last_framebuffer_id && last_framebuffer_serial == render_targets_serial) {
        return &slot_framebuffers[last_framebuffer_id];
    }
    const FramebufferId framebuffer_id = GetFramebufferId(render_targets);
    last_framebuffer_id = framebuffer_id;
    last_framebuffer_serial = render_targets_serial;
    return &slot_framebuffers[framebuffer_id];
}

template <class P>
ImageViewId TextureCache<P>::VisitImageView(u32 index, bool compute) {
    auto& table = compute ? channel_state->compute_image_table : channel_state->graphics_image_table;
    index = Tegra::Texture::TextureHeaderIndex(index, table.current_limit,
                                               gpu_memory->UsesNativeTiling());
    if (index > table.current_limit) {
        LOG_TRACE(HW_GPU, "Invalid image view index={}", index);
        return NULL_IMAGE_VIEW_ID;
    }
    auto const [descriptor, is_new] = table.Read(*gpu_memory, index);
    if (is_new) {
        if (IsValidEntry(*gpu_memory, descriptor)) {
            const auto [pair, is_new_tc] = channel_state->image_views.try_emplace(descriptor);
            if (is_new_tc) {
                const auto failed_images = [&]() -> u32 {
                    if constexpr (requires { runtime.FailedImages(); }) {
                        return runtime.FailedImages();
                    } else {
                        return 0;
                    }
                };
                const u32 failed_before = failed_images();
                pair->second = CreateImageView(descriptor);
                if (pair->second == NULL_IMAGE_VIEW_ID && failed_images() != failed_before) {
                    // No memory for its image: TickFrame invalidates the table after collecting,
                    // and the descriptor's next read makes the image again.
                    channel_state->image_views.erase(pair);
                    table.bound_ids[index] = NULL_IMAGE_VIEW_ID;
                    return NULL_IMAGE_VIEW_ID;
                }
                if (pair->second != NULL_IMAGE_VIEW_ID) {
                    image_view_descriptors[pair->second].emplace_back(current_channel_id,
                                                                      descriptor);
                }
            }
            PrepareImageView(pair->second, false, false);
            table.bound_ids[index] = pair->second;
            return pair->second;
        }
        table.bound_ids[index] = NULL_IMAGE_VIEW_ID;
        return NULL_IMAGE_VIEW_ID;
    }
    const auto id = table.bound_ids[index];
    if (id != NULL_IMAGE_VIEW_ID)
        PrepareImageView(id, false, false);
    return id;
}

template <class P>
FramebufferId TextureCache<P>::GetFramebufferId(const RenderTargets& key) {
    const auto [pair, is_new] = framebuffers.try_emplace(key);
    FramebufferId& framebuffer_id = pair->second;
    if (!is_new) {
        return framebuffer_id;
    }
    std::array<ImageView*, NUM_RT> color_buffers;
    std::ranges::transform(key.color_buffer_ids, color_buffers.begin(), [this](ImageViewId id) {
        return id ? &slot_image_views[id] : nullptr;
    });
    ImageView* const depth_buffer = key.depth_buffer_id ? &slot_image_views[key.depth_buffer_id] : nullptr;
    const auto framebuffer_start = std::chrono::steady_clock::now();
    runtime.FlushDeferredClear();
    framebuffer_id = slot_framebuffers.insert(runtime, color_buffers, depth_buffer, key);
    framebuffer_keys.emplace(framebuffer_id, key);
    const auto index_view = [this, id = framebuffer_id](ImageViewId view) {
        if (!view) {
            return;
        }
        auto& ids = framebuffers_by_view[view];
        if (std::ranges::find(ids, id) == ids.end()) {
            ids.push_back(id);
        }
    };
    std::ranges::for_each(key.color_buffer_ids, index_view);
    index_view(key.depth_buffer_id);
    ++rt_stats.framebuffers_created;
    rt_stats.framebuffer_ns += NsSince(framebuffer_start);
    return framebuffer_id;
}

template <class P>
void TextureCache<P>::WriteMemory(DAddr cpu_addr, size_t size) {
    // EDEN_NX2_TEX_BANK_GEN: InvalidateRegion may reach only this cache.
    Core::g_watch_write_pages.Mark(cpu_addr, size);
    ClearStale(cpu_addr, size);
    ForEachImageInRegion(cpu_addr, size, [this, cpu_addr, size](ImageId image_id, Image& image) {
        ++bindings_epoch;
        VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::Invalidation);
        if (True(image.flags & ImageFlagBits::UnwrittenSparse)) {
            // Written at last: a backed image replaces it where it is used next.
            image.flags |= ImageFlagBits::CpuModified | ImageFlagBits::Remapped;
            return;
        }
        if (True(image.flags & ImageFlagBits::SparseTiles)) {
            MarkRangeWritten(image, cpu_addr, size);
            return;
        }
        if (UploadsByLayer(image)) {
            MarkLayersWritten(image, cpu_addr, size);
            return;
        }
        if (True(image.flags & ImageFlagBits::CpuModified)) {
            return;
        }
        image.flags |= ImageFlagBits::CpuModified;
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
    });
}

template <class P>
bool TextureCache<P>::IsDownloadable(const ImageBase& image) const noexcept {
    if (!image.IsSafeGpuCopy() || True(image.flags & (ImageFlagBits::Unbacked | ImageFlagBits::TrimmedLevels))) {
        return false;
    }
    if (image.info.num_samples == 1) {
        return true;
    }
    if constexpr (P::HAS_MSAA_DOWNLOADS) {
        return runtime.CanDownloadMsaa(image.info);
    } else {
        return false;
    }
}

template <class P>
void TextureCache<P>::DownloadMemory(DAddr cpu_addr, size_t size) {
    const Common::WriteRing::SiteScope write_site{"texture download (DownloadMemory)", true};
    if (!relocation_tombstones.empty()) {
        CheckTombstoneCpuRead(cpu_addr, size); // EXP43=4
    }
    boost::container::small_vector<ImageId, 16> images;
    ForEachImageInRegion(cpu_addr, size, [this, &images](ImageId image_id, ImageBase& image) {
        if (!IsDownloadable(image)) {
            return;
        }
        if (True(image.flags & ImageFlagBits::GpuModified) &&
            (image.relocation_guard_noted & GuardCpuRead) == 0 && RelocationShaped(image.info)) {
            NoteRelocationGuard(image, GuardCpuRead);
        }
        unmap_stats.downloads += image.unmapped_tick != 0 ? 1 : 0;
        image.flags &= ~ImageFlagBits::GpuModified;
        images.push_back(image_id);
    });
    if (images.empty()) {
        return;
    }
    // EDEN_NX2_FLUSH_AREA_LOG=1: image downloads that wait for the GPU (first ones, then a count
    // every 10 s).
    static const bool image_flush_log = std::getenv("EDEN_NX2_FLUSH_AREA_LOG") != nullptr;
    if (image_flush_log) {
        static int logged = 0;
        static u64 count = 0, bytes = 0;
        static auto last = std::chrono::steady_clock::now();
        for (const ImageId id : images) {
            const ImageBase& image = slot_images[id];
            ++count;
            bytes += image.unswizzled_size_bytes;
            if (logged < 32) {
                ++logged;
                LOG_INFO(HW_GPU,
                         "Image flush: read [{:#x} +{:#x}) downloads image cpu={:#x} gpu={:#x} "
                         "format={} {}x{}x{} levels={} layers={}",
                         cpu_addr, size, image.cpu_addr, image.gpu_addr,
                         static_cast<u32>(image.info.format), image.info.size.width,
                         image.info.size.height, image.info.size.depth,
                         image.info.resources.levels, image.info.resources.layers);
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last > std::chrono::seconds(10)) {
            LOG_INFO(HW_GPU, "Image flush: {} image downloads, {} KiB in 10 s", count,
                     bytes / 1024);
            count = 0;
            bytes = 0;
            last = now;
        }
    }
    std::ranges::sort(images, [this](ImageId lhs, ImageId rhs) {
        return slot_images[lhs].modification_tick < slot_images[rhs].modification_tick;
    });
    // Record the copies of a batch, wait for the GPU once, then swizzle in modification order
    // (the copies only read, so they do not depend on each other). A batch is capped so the
    // staging memory held at once stays bounded.
    constexpr size_t MaxBatchBytes = 64_MiB;
    using Map = decltype(runtime.DownloadStagingBuffer(size_t{}));
    using Copies = decltype(FullDownloadCopies(std::declval<const ImageInfo&>()));
    struct Pending {
        ImageId image_id;
        Map map;
        Copies copies;
    };
    boost::container::small_vector<Pending, 16> pending;
    size_t pending_bytes = 0;
    const auto flush_pending = [&] {
        if (pending.empty()) {
            return;
        }
        runtime.Finish();
        for (Pending& entry : pending) {
            Image& image = slot_images[entry.image_id];
            DropCollectDownloadsIn(image.cpu_addr, image.guest_size_bytes);
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, entry.copies,
                         entry.map.mapped_span, swizzle_data_buffer);
            ClearStaleImage(image);
        }
        pending.clear();
        pending_bytes = 0;
    };
    for (const ImageId image_id : images) {
        Image& image = slot_images[image_id];
        if (!pending.empty() && pending_bytes + image.unswizzled_size_bytes > MaxBatchBytes) {
            flush_pending();
        }
        auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);
        auto copies = FullDownloadCopies(image.info);
        if (depth_census_on) [[unlikely]] {
            CensusNoteRead(image, false);
        }
        image.DownloadMemory(map, copies);
        pending_bytes += image.unswizzled_size_bytes;
        pending.push_back(Pending{image_id, map, std::move(copies)});
    }
    flush_pending();
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> TextureCache<P>::GetFlushArea(DAddr cpu_addr,
                                                                               u64 size) {
    if (!relocation_tombstones.empty()) {
        CheckTombstoneCpuRead(cpu_addr, size); // EXP43=4
    }
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    ForEachImageInRegion(cpu_addr, size, [&](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return;
        }
        if (!area) {
            area.emplace();
            area->start_address = cpu_addr;
            area->end_address = cpu_addr + size;
            area->preemtive = true;
        }
        area->start_address = (std::min)(area->start_address, image.cpu_addr);
        area->end_address = (std::max)(area->end_address, image.cpu_addr_end);
        for (auto image_view_id : image.image_view_ids) {
            auto& image_view = slot_image_views[image_view_id];
            image_view.flags |= ImageViewFlagBits::PreemtiveDownload;
        }
        area->preemtive &= image.info.forced_flushed;
        // EDEN_NX2_FLUSH_AREA_LOG=1: which CPU reads turn an image into a per-bind download.
        static const bool flush_area_log = std::getenv("EDEN_NX2_FLUSH_AREA_LOG") != nullptr;
        static int flush_area_logged = 0;
        if (flush_area_log && !image.info.forced_flushed && flush_area_logged < 64) {
            ++flush_area_logged;
            LOG_INFO(HW_GPU,
                     "Flush area: CPU read [{:#x} +{:#x}) marks image cpu={:#x}..{:#x} gpu={:#x} "
                     "format={} {}x{}x{} levels={} layers={} pitch_linear={}",
                     cpu_addr, size, image.cpu_addr, image.cpu_addr_end, image.gpu_addr,
                     static_cast<u32>(image.info.format), image.info.size.width,
                     image.info.size.height, image.info.size.depth, image.info.resources.levels,
                     image.info.resources.layers, image.info.type == ImageType::Linear);
        }
        image.info.forced_flushed = true;
        if ((image.relocation_guard_noted & GuardCpuRead) == 0 && RelocationShaped(image.info)) {
            NoteRelocationGuard(image, GuardCpuRead);
        }
    });
    return area;
}

template <class P>
void TextureCache<P>::LogGpuModifiedImagesInRegion(DAddr cpu_addr, u64 size) {
    // The images GetFlushArea's lambda acted on: same walk, same test, under the same hold.
    ForEachImageInRegion(cpu_addr, size, [&](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return;
        }
        LOG_ERROR(HW_GPU,
                  "EXP100 verify: read [{:#x} +{:#x}) found GPU-modified image cpu={:#x}..{:#x} "
                  "size={:#x} gpu={:#x} flags={:#x} nx2_gpu_counted={} counted={:#x} +{:#x}",
                  cpu_addr, size, image.cpu_addr, image.cpu_addr_end, image.guest_size_bytes,
                  image.gpu_addr, static_cast<u32>(image.flags & ~ImageFlagBits::Picked),
                  image.nx2_gpu_counted,
                  image.nx2_counted_addr, image.nx2_counted_size);
    });
}

template <class P>
void TextureCache<P>::UnmapMemory(DAddr cpu_addr, size_t size) {
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::Unmap};
    // EDEN_NX2_TEX_BANK_GEN: a watched page leaving its mapping drops every watch (their trap
    // counts go back while the pages are still mapped; memos of the old epoch miss).
    if (!tex_bank_pages.empty()) {
        const DAddr first = Common::AlignDown(cpu_addr, Core::DEVICE_PAGESIZE);
        for (const auto& [page, entry] : tex_bank_pages) {
            if (page >= first && page < cpu_addr + size) {
                TexBankReset();
                break;
            }
        }
    }
    boost::container::small_vector<ImageId, 16> deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        Image& image = slot_images[id];
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, id);
        }
        UnregisterImage(id);
        DeleteImage(id);
    }
    ClearStale(cpu_addr, size);
}

template <class P>
void TextureCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    boost::container::small_vector<ImageId, 16> deleted_images;
    ForEachImageInRegionGPU(as_id, gpu_addr, size,
                            [&](ImageId id, Image&) { deleted_images.push_back(id); });
    if (VideoCommon::Nx2FastDrawCount::g_on && !deleted_images.empty()) [[unlikely]] {
        VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::Map);
    }
    unmap_stats.unmaps_hit += deleted_images.empty() ? 0 : 1;
    unmap_stats.images_hit += deleted_images.size();
    for (const ImageId id : deleted_images) {
        Image& image = slot_images[id];
        ++bindings_epoch;
        if (False(image.flags & ImageFlagBits::CpuModified)) {
            image.flags |= ImageFlagBits::CpuModified;
            if (True(image.flags & ImageFlagBits::Tracked)) {
                UntrackImage(image, id);
            }
        }
        image.dirty_layers.clear(); // all of it
        if (True(image.flags & ImageFlagBits::Remapped)) {
            continue;
        }
        image.flags |= ImageFlagBits::Remapped;
        ++nx2_flagged_remapped;
        NoteUnmapped(id, image);
    }
}

template <class P>
void TextureCache<P>::NoteUnmapped(ImageId image_id, Image& image) {
    image.unmapped_tick = frame_tick + 1;
    image.unmapped_bytes = ImageMemoryBytes(image);
    image.unmapped_index = static_cast<u32>(unmapped_images.size());
    image.unmapped_verify = 0;
    unmapped_images.push_back(image_id);
    unmapped_live_bytes += image.unmapped_bytes;
    ++unmapped_live_images;
    ++unmap_stats.flagged;
}

template <class P>
void TextureCache<P>::LogUnmapStats() {
    auto& s = unmap_stats;
    const auto at = [](const auto& counts, DeleteReason reason) {
        return counts[static_cast<size_t>(reason)];
    };
    const auto others = [&](const auto& counts) {
        return at(counts, DeleteReason::Other) + at(counts, DeleteReason::ColdSweep) +
               at(counts, DeleteReason::SlotRetire) + at(counts, DeleteReason::SupersededSweep);
    };
    const u64 collector_unused = at(s.died_unused, DeleteReason::GarbageCollect);
    const double per_death = collector_unused != 0 ? static_cast<double>(collector_unused) : 1.0;
    LOG_INFO(HW_GPU,
             "Texture cache last 600 frames: GPU memory unmaps {} hit {} images, {} newly flagged "
             "remapped; flagged images found again {}, uploaded into {}, downloaded {}; deleted "
             "while flagged (of them unused since the unmap): collector {} ({}, {:.1f} frames "
             "after it, {} KiB each), join {} ({}), unmap {} ({}), other {} ({}); flagged now {} "
             "images {} MiB, held {} MiB on average, {} MiB peak",
             s.unmaps_hit, s.images_hit, s.flagged, s.refound, s.reuploaded, s.downloads,
             at(s.died, DeleteReason::GarbageCollect), collector_unused,
             static_cast<double>(s.collector_unused_frames) / per_death,
             static_cast<u64>(static_cast<double>(s.collector_unused_bytes) / per_death) >> 10,
             at(s.died, DeleteReason::Join), at(s.died_unused, DeleteReason::Join),
             at(s.died, DeleteReason::Unmap), at(s.died_unused, DeleteReason::Unmap),
             others(s.died), others(s.died_unused), unmapped_live_images,
             unmapped_live_bytes >> 20, (s.held_byte_frames / 600) >> 20, s.held_peak >> 20);
    const int mode = SweepUnmappedMode();
    if (mode == 0) {
        return;
    }
    LOG_INFO(HW_GPU,
             "Texture cache last 600 frames: unmapped sweep (EDEN_NX2_SWEEP_UNMAPPED={}) deleted "
             "{} ({} MiB), {} looked at; would delete {} ({} MiB), of those later found again {}, "
             "uploaded into {} ({} GPU-written before the unmap), presented {}, written whole {}, "
             "deleted unused by another path {}",
             mode, s.swept, s.swept_bytes >> 20, s.sweep_scanned, s.would_sweep,
             s.would_sweep_bytes >> 20, s.would_refound, s.would_reuploaded,
             s.would_reuploaded_gpu, s.would_presented, s.would_overwritten, s.would_died_unused);
}

template <class P>
void TextureCache<P>::ForgetUnmapped(ImageId image_id, ImageBase& image) {
    unmapped_live_bytes -= (std::min)(unmapped_live_bytes, image.unmapped_bytes);
    unmapped_live_images -= unmapped_live_images != 0 ? 1 : 0;
    const u32 index = image.unmapped_index;
    if (index < unmapped_images.size() && unmapped_images[index] == image_id) {
        const ImageId last = unmapped_images.back();
        unmapped_images[index] = last;
        slot_images[last].unmapped_index = index;
        unmapped_images.pop_back();
    }
    image.unmapped_tick = 0;
    image.unmapped_bytes = 0;
    image.unmapped_index = ~0U;
    image.unmapped_verify = 0;
}

template <class P>
void TextureCache<P>::NotePresented(ImageId image_id) {
    if (std::ranges::find(presented_images, image_id) == presented_images.end()) {
        presented_images[presented_next] = image_id;
        presented_next = (presented_next + 1) % static_cast<u32>(presented_images.size());
    }
    ImageBase& image = slot_images[image_id];
    if (image.unmapped_verify == 1) {
        ++unmap_stats.would_presented;
        image.unmapped_verify = 2;
    }
}

template <class P>
void TextureCache<P>::SweepUnmapped() {
    // UnmapGPUMemory flags the images in an unmapped GPU range Remapped and CpuModified. Lookups
    // skip them (FindImage unless EXP76 & 1 keeps one at the same address and shape, FindDMAImage,
    // joins, which delete them without a download when a new image covers their memory), so most
    // of them sit until the collector's age limit (50 frames, 10 near the budget). Ocarina of
    // Time at 3440x1440 unmaps and maps its render targets again every frame or two while the
    // camera turns in the Castle Town guard house; the dead sets overflowed 12 GB cards.
    //
    // Deleted here, 2 frames after their last use and without a download, only when nothing can
    // still need their contents, so the result is what the collector does later:
    // - CpuModified still set: nothing uploaded into it or wrote it whole since the unmap (a
    //   cached descriptor or a render target binding can still reach it; that upload clears the
    //   flag). Such an image is never downloaded (IsSafeGpuCopy), by the collector or a guest
    //   read, and its next use would upload guest memory again, as a new image does. The GPU
    //   data from before the unmap is already given up when the unmap set the flag.
    // - No aliases: SynchronizeAliases copies from an alias without preparing it.
    // - Not among the last images shown: presenting reads an image without preparing it.
    // - Not a bound render target, no download queued, no decode in flight, not sparse or
    //   unbacked (their own handling), and EXP121 has not taken it.
    // MarkStale and untracking as the collector does for an image it does not download.
    const int mode = SweepUnmappedMode();
    if (mode == 0 || unmapped_images.empty()) {
        return;
    }
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::UnmapSweep};
    constexpr u64 IdleFrames = 2;
    constexpr ImageFlagBits excluded =
        ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
        ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked | ImageFlagBits::IsDecoding |
        ImageFlagBits::CollectPending;
    const auto queued = [](ImageId image_id, const auto& downloads) {
        return std::ranges::any_of(downloads, [image_id](const auto& download) {
            return download.is_swizzle && download.object_id == image_id;
        });
    };
    const auto bound = [this](const ImageBase& image) {
        return std::ranges::any_of(image.image_view_ids, [this](ImageViewId view) {
            return render_targets.depth_buffer_id == view ||
                   std::ranges::find(render_targets.color_buffer_ids, view) !=
                       render_targets.color_buffer_ids.end();
        });
    };
    const auto download_queued = [&](ImageId image_id) {
        return queued(image_id, uncommitted_downloads) ||
               std::ranges::any_of(committed_downloads, [&](const auto& downloads) {
                   return queued(image_id, downloads);
               });
    };
    for (size_t index = 0; index < unmapped_images.size();) {
        ++unmap_stats.sweep_scanned;
        const ImageId image_id = unmapped_images[index];
        Image& image = slot_images[image_id];
        if (False(image.flags & ImageFlagBits::Remapped) ||
            False(image.flags & ImageFlagBits::Registered) ||
            False(image.flags & ImageFlagBits::CpuModified) || True(image.flags & excluded) ||
            !image.aliased_images.empty() || image.unmapped_verify != 0 ||
            frame_tick - lru_cache.Tick(image.lru_index) < IdleFrames || bound(image) ||
            std::ranges::find(presented_images, image_id) != presented_images.end() ||
            download_queued(image_id)) {
            ++index;
            continue;
        }
        if (mode == 2) {
            image.unmapped_verify = 1;
            ++unmap_stats.would_sweep;
            unmap_stats.would_sweep_bytes += image.unmapped_bytes;
            ++index;
            continue;
        }
        ++unmap_stats.swept;
        unmap_stats.swept_bytes += image.unmapped_bytes;
        MarkStale(image); // before UntrackImage (see MarkStale)
        if (True(image.flags & ImageFlagBits::Tracked)) {
            UntrackImage(image, image_id);
        }
        UnregisterImage(image_id);
        DeleteImage(image_id); // takes it out of the list: the last entry moves to this index
    }
}

template <class P>
bool TextureCache<P>::BlitImage(const Tegra::Engines::Fermi2D::Surface& dst,
                                const Tegra::Engines::Fermi2D::Surface& src,
                                const Tegra::Engines::Fermi2D::Config& copy) {
    // A deferred clear parked before this copy has to land before it: realizing it afterwards
    // starts its clear pass after the copy and wipes what the copy wrote. That is what made
    // Ocarina of Time draw its terrain layer over the scene: the game clears the
    // occlusion depth, blits the scene depth over it, and reads it back during the layer draw.
    // GetFramebufferId only flushes when it creates a framebuffer, so the bug starts on the
    // second frame. Texture cache copies bypass render passes and keep stream order here.
    runtime.FlushDeferredClear();
    const auto result = GetBlitImages(dst, src, copy);
    if (!result) {
        return false;
    }
    const BlitImages images = *result;
    const ImageId dst_id = images.dst_id;
    const ImageId src_id = images.src_id;

    PrepareImage(src_id, false, false);
    PrepareImage(dst_id, true, false);

    Image& dst_image = slot_images[dst_id];
    Image& src_image = slot_images[src_id];
    if (True(dst_image.flags & ImageFlagBits::Unbacked)) {
        static u32 logged{};
        if (logged++ < 64) {
            LOG_INFO(HW_GPU, "2D blit into unbacked image {:#x}: dst {:#x} {}x{} from {:#x} {}x{}",
                     dst_image.gpu_addr, dst.Address(), dst.width, dst.height, src.Address(),
                     src.width, src.height);
        }
    }
    bool is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
    bool is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);

    const bool is_resolve = src_image.info.num_samples != 1 && dst_image.info.num_samples == 1;
    if (is_src_rescaled != is_dst_rescaled) {
        if (ImageCanRescale(src_image)) {
            ScaleUp(src_image);
            is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
            if (is_resolve) {
                dst_image.info.rescaleable = true;
                for (const auto& alias : dst_image.aliased_images) {
                    Image& other_image = slot_images[alias.id];
                    other_image.info.rescaleable = true;
                }
            }
        }
        if (ImageCanRescale(dst_image)) {
            ScaleUp(dst_image);
            is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);
        }
    }
    if (is_resolve && (is_src_rescaled != is_dst_rescaled)) {
        // A resolve requires both images to be the same dimensions. Resize down if needed.
        ScaleDown(src_image);
        ScaleDown(dst_image);
        is_src_rescaled = True(src_image.flags & ImageFlagBits::Rescaled);
        is_dst_rescaled = True(dst_image.flags & ImageFlagBits::Rescaled);
    }
    const auto& resolution = Settings::values.resolution_info;
    const auto scale_region = [&](Region2D& region) {
        region.start.x = resolution.ScaleUp(region.start.x);
        region.start.y = resolution.ScaleUp(region.start.y);
        region.end.x = resolution.ScaleUp(region.end.x);
        region.end.y = resolution.ScaleUp(region.end.y);
    };

    // TODO: Deduplicate
    const std::optional src_base = src_image.TryFindBase(src.Address());
    const SubresourceRange src_range{.base = src_base.value(), .extent = {1, 1}};
    const ImageViewInfo src_view_info(ImageViewType::e2D, images.src_format, src_range);
    const auto [src_framebuffer_id, src_view_id] = RenderTargetFromImage(src_id, src_view_info);
    const auto [src_samples_x, src_samples_y] = SamplesLog2(src_image.info.num_samples);
    Region2D src_region{
        Offset2D{.x = copy.src_x0 >> src_samples_x, .y = copy.src_y0 >> src_samples_y},
        Offset2D{.x = copy.src_x1 >> src_samples_x, .y = copy.src_y1 >> src_samples_y},
    };
    if (is_src_rescaled) {
        scale_region(src_region);
    }

    const std::optional dst_base = dst_image.TryFindBase(dst.Address());
    const SubresourceRange dst_range{.base = dst_base.value(), .extent = {1, 1}};
    const ImageViewInfo dst_view_info(ImageViewType::e2D, images.dst_format, dst_range);
    const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
    const auto [dst_samples_x, dst_samples_y] = SamplesLog2(dst_image.info.num_samples);
    Region2D dst_region{
        Offset2D{.x = copy.dst_x0 >> dst_samples_x, .y = copy.dst_y0 >> dst_samples_y},
        Offset2D{.x = copy.dst_x1 >> dst_samples_x, .y = copy.dst_y1 >> dst_samples_y},
    };
    if (is_dst_rescaled) {
        scale_region(dst_region);
    }

    // Always call this after src_framebuffer_id was queried, as the address might be invalidated.
    Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
    if constexpr (FRAMEBUFFER_BLITS) {
        // OpenGL blits from framebuffers, not images
        Framebuffer* const src_framebuffer = &slot_framebuffers[src_framebuffer_id];
        runtime.BlitFramebuffer(dst_framebuffer, src_framebuffer, dst_region, src_region,
                                copy.filter, copy.operation);
    } else {
        // Vulkan can blit images, but it lacks format reinterpretations
        // Provide a framebuffer in case it's necessary
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        runtime.BlitImage(dst_framebuffer, dst_view, src_view, dst_region, src_region, copy.filter,
                          copy.operation);
    }
    return true;
}

template <class P>
std::optional<VideoCore::GameDlssProbe> TextureCache<P>::ProbeGameDlssOutput(
    const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src) {
    if constexpr (FRAMEBUFFER_BLITS) {
        return std::nullopt;
    } else {
        const Tegra::Engines::Fermi2D::Config copy{
            .operation = Tegra::Engines::Fermi2D::Operation::SrcCopy,
            .filter = Tegra::Engines::Fermi2D::Filter::Point,
            .must_accelerate = true,
            .dst_x0 = 0,
            .dst_y0 = 0,
            .dst_x1 = static_cast<s32>(dst.width),
            .dst_y1 = static_cast<s32>(dst.height),
            .src_x0 = 0,
            .src_y0 = 0,
            .src_x1 = static_cast<s32>(src.width),
            .src_y1 = static_cast<s32>(src.height),
        };
        const auto result = GetBlitImages(dst, src, copy);
        if (!result) {
            return std::nullopt;
        }
        const ImageId dst_id = result->dst_id;
        const ImageId src_id = result->src_id;
        PrepareImage(src_id, false, false);
        PrepareImage(dst_id, false, false);
        for (const ImageId id : {src_id, dst_id}) {
            // The network runs only at the game's own resolution.
            if (slot_images[id].info.num_samples != 1 ||
                True(slot_images[id].flags & ImageFlagBits::Rescaled)) {
                return std::nullopt;
            }
        }
        const auto view = [&](ImageId id, const Tegra::Engines::Fermi2D::Surface& surface,
                              PixelFormat format) -> std::optional<ImageViewId> {
            const std::optional base = slot_images[id].TryFindBase(surface.Address());
            if (!base) {
                return std::nullopt;
            }
            const SubresourceRange range{.base = *base, .extent = {1, 1}};
            return FindOrEmplaceImageView(id, ImageViewInfo(ImageViewType::e2D, format, range));
        };
        const auto src_view = view(src_id, src, result->src_format);
        const auto dst_view = view(dst_id, dst, result->dst_format);
        if (!src_view || !dst_view) {
            return std::nullopt;
        }
        return runtime.ProbeGameDlssOutput(slot_image_views[*src_view],
                                           slot_image_views[*dst_view]);
    }
}

template <class P>
bool TextureCache<P>::TemporalUpscale(const Tegra::Engines::Fermi2D::Surface& dst,
                                      const Tegra::Engines::Fermi2D::Surface& src,
                                      const Tegra::Engines::Fermi2D::Surface& motion_vectors,
                                      const Tegra::Texture::TICEntry* depth,
                                      std::array<f32, 2> jitter, std::array<f32, 2> motion_scale,
                                      bool use_history) {
    if constexpr (FRAMEBUFFER_BLITS) {
        return false;
    } else {
        const Tegra::Engines::Fermi2D::Config copy{
            .operation = Tegra::Engines::Fermi2D::Operation::SrcCopy,
            .filter = Tegra::Engines::Fermi2D::Filter::Bilinear,
            .must_accelerate = true,
            .dst_x0 = 0,
            .dst_y0 = 0,
            .dst_x1 = static_cast<s32>(dst.width),
            .dst_y1 = static_cast<s32>(dst.height),
            .src_x0 = 0,
            .src_y0 = 0,
            .src_x1 = static_cast<s32>(src.width),
            .src_y1 = static_cast<s32>(src.height),
        };
        const auto result = GetBlitImages(dst, src, copy);
        if (!result) {
            return false;
        }
        const auto& motion_surface = use_history ? motion_vectors : src;
        const ImageInfo motion_info(motion_surface);
        has_deleted_images = false;
        const ImageId motion_id =
            FindOrInsertImage(motion_info, motion_surface.Address(), RelaxedOptions{});
        // The depth only feeds NVIDIA DLSS and FSR 3.1 (looked up only when they can run); without
        // it the stand-in runs.
        ImageId depth_id{};
        if (use_history && depth && motion_id && runtime.WantsTemporalUpscaleDepth()) {
            depth_id = FindOrInsertImage(ImageInfo(*depth), depth->Address(), RelaxedOptions{});
            if (!depth_id) {
                return false;
            }
        }
        if (has_deleted_images || !motion_id) {
            return false;
        }
        const ImageId dst_id = result->dst_id;
        const ImageId src_id = result->src_id;
        PrepareImage(src_id, false, false);
        PrepareImage(motion_id, false, false);
        PrepareImage(dst_id, true, false);
        if (depth_id) {
            PrepareImage(depth_id, false, false);
            if (slot_images[depth_id].info.num_samples != 1 ||
                slot_images[depth_id].info.format != PixelFormat::D32_FLOAT) {
                return false;
            }
        }
        for (const ImageId id : {src_id, motion_id, dst_id}) {
            if (slot_images[id].info.num_samples != 1) {
                return false;
            }
        }
        // With resolution scaling, bring the images that can follow to the same scale, as BlitImage
        // does; the stand-in reads each image at its own size, so any left behind still work.
        const auto rescaled = [&](ImageId id) {
            return True(slot_images[id].flags & ImageFlagBits::Rescaled);
        };
        if (rescaled(src_id) || rescaled(motion_id) || rescaled(dst_id) ||
            (depth_id && rescaled(depth_id))) {
            for (const ImageId id : {src_id, motion_id, dst_id, depth_id}) {
                if (id && !rescaled(id) && ImageCanRescale(slot_images[id])) {
                    ScaleUp(slot_images[id]);
                }
            }
        }
        // FSR 3.1 reads colour, motion and depth texel for texel: all at the colour's scale.
        if (depth_id && (rescaled(depth_id) != rescaled(src_id) ||
                         rescaled(motion_id) != rescaled(src_id))) {
            return false;
        }
        const Extent3D src_size = slot_images[src_id].info.size;
        if (src_size.width != src.width || src_size.height != src.height) {
            return false;
        }
        const std::array<u32, 2> render_size =
            rescaled(src_id)
                ? std::array{Settings::values.resolution_info.ScaleUp(src_size.width),
                             Settings::values.resolution_info.ScaleUp(src_size.height)}
                : std::array{src_size.width, src_size.height};
        // The jitter is in the input's guest pixels.
        if (rescaled(src_id)) {
            const f32 factor = Settings::values.resolution_info.up_factor;
            jitter = {jitter[0] * factor, jitter[1] * factor};
        }
        const auto view = [&](ImageId id, const Tegra::Engines::Fermi2D::Surface& surface,
                              PixelFormat format) -> std::optional<std::pair<FramebufferId, ImageViewId>> {
            const std::optional base = slot_images[id].TryFindBase(surface.Address());
            if (!base) {
                return std::nullopt;
            }
            const SubresourceRange range{.base = *base, .extent = {1, 1}};
            return RenderTargetFromImage(id, ImageViewInfo(ImageViewType::e2D, format, range));
        };
        const auto src_view = view(src_id, src, result->src_format);
        const auto motion_view = view(motion_id, motion_surface, motion_info.format);
        const auto dst_view = view(dst_id, dst, result->dst_format);
        if (!src_view || !motion_view || !dst_view) {
            return false;
        }
        ImageViewId depth_view{};
        if (depth_id) {
            if (const std::optional base = slot_images[depth_id].TryFindBase(depth->Address())) {
                const SubresourceRange range{.base = *base, .extent = {1, 1}};
                depth_view = FindOrEmplaceImageView(
                    depth_id, ImageViewInfo(ImageViewType::e2D, PixelFormat::D32_FLOAT, range));
            }
            if (!depth_view) {
                return false;
            }
        }
        // After the views: creating them may reallocate the framebuffer slots.
        Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_view->first];
        return runtime.TemporalUpscale(
            dst_framebuffer, slot_image_views[dst_view->second], slot_image_views[src_view->second],
            slot_image_views[motion_view->second],
            depth_view ? &slot_image_views[depth_view] : nullptr, render_size, jitter,
            motion_scale, use_history);
    }
}

template <class P>
std::pair<typename P::ImageView*, bool> TextureCache<P>::TryFindFramebufferImageView(
    const Tegra::FramebufferConfig& config, DAddr cpu_addr) {
    // TODO: Properly implement this
    const auto it = page_table.find(cpu_addr >> YUZU_PAGEBITS);
    if (it == page_table.end()) {
        return {};
    }
    const auto view_format = [&]() {
        switch (config.pixel_format) {
        case Service::android::PixelFormat::Rgb565:
            return PixelFormat::R5G6B5_UNORM;
        case Service::android::PixelFormat::Bgra8888:
            return PixelFormat::B8G8R8A8_UNORM;
        default:
            return PixelFormat::A8B8G8R8_UNORM;
        }
    }();

    const auto& image_map_ids = it->second;
    boost::container::small_vector<ImageId, 4> valid_image_ids;
    for (const ImageMapId map_id : image_map_ids) {
        const ImageMapView& map = slot_map_views[map_id];
        const ImageBase& image = slot_images[map.image_id];
        if (image.cpu_addr != cpu_addr) {
            continue;
        }
        if (image.image_view_ids.empty()) {
            continue;
        }
        if (!VideoCore::Surface::IsViewCompatible(image.info.format, view_format, false, true)) {
            static u32 rejected_alias_count{};
            if (NativeDrawDiagnostics() && rejected_alias_count++ < 8) {
                LOG_WARNING(HW_GPU, "Ignoring incompatible framebuffer alias: address={:#x}, image={}, view={}",
                            cpu_addr, image.info.format, view_format);
            }
            continue;
        }
        valid_image_ids.push_back(map.image_id);
    }

    const auto GetImageViewForFramebuffer = [&](ImageId image_id) {
        ImageViewInfo info{ImageViewType::e2D, view_format};
        if (config.blending == Tegra::BlendMode::Opaque) {
            info.x_source = static_cast<u8>(SwizzleSource::R);
            info.y_source = static_cast<u8>(SwizzleSource::G);
            info.z_source = static_cast<u8>(SwizzleSource::B);
            info.w_source = static_cast<u8>(SwizzleSource::OneFloat);
        }
        NotePresented(image_id);
        return std::make_pair(&slot_image_views[FindOrEmplaceImageView(image_id, info)],
                              slot_images[image_id].IsRescaled());
    };

    if (valid_image_ids.size() == 1) [[likely]] {
        return GetImageViewForFramebuffer(valid_image_ids.front());
    }

    if (valid_image_ids.size() > 0) [[unlikely]] {
        auto most_recent = std::ranges::max_element(valid_image_ids, [&](auto a, auto b) {
            return slot_images[a].modification_tick < slot_images[b].modification_tick;
        });
        return GetImageViewForFramebuffer(*most_recent);
    }

    return {};
}

template <class P>
bool TextureCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_downloads.empty();
}

template <class P>
bool TextureCache<P>::HasAsyncFlushWork() const noexcept {
    return !uncommitted_downloads.empty() || !uncommitted_async_buffers.empty();
}

template <class P>
bool TextureCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return !committed_downloads.empty() && !committed_downloads.front().empty();
}

template <class P>
void TextureCache<P>::CommitAsyncFlushes() {
    // This is intentionally passing the value by copy
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        auto& download_ids = uncommitted_downloads;
        const size_t redundant_images = DeduplicatePendingImageDownloads(download_ids, slot_images);
        if (download_ids.empty()) {
            committed_downloads.emplace_back(std::move(uncommitted_downloads));
            uncommitted_downloads.clear();
            async_buffers.emplace_back(std::move(uncommitted_async_buffers));
            uncommitted_async_buffers.clear();
            return;
        }
        size_t total_size_bytes = 0;
        size_t last_async_buffer_id = uncommitted_async_buffers.size();
        bool any_none_dma = false;
        for (PendingDownload& download_info : download_ids) {
            if (download_info.is_swizzle) {
                total_size_bytes +=
                    Common::AlignUp(download_info.GetImage(slot_images).unswizzled_size_bytes, 64);
                any_none_dma = true;
                download_info.async_buffer_id = last_async_buffer_id;
            }
        }

        if (any_none_dma) {
            if (NativeDrawDiagnostics() && total_size_bytes >= 64_MiB) {
                LOG_WARNING(HW_GPU, "Large image readback batch: bytes={}, entries={}, duplicates_removed={}, submitted_batches={}, retained_buffers={}",
                            total_size_bytes, download_ids.size(), redundant_images,
                            committed_downloads.size(), async_buffers_death_ring.size());
                size_t traced_images = 0;
                for (const auto& download : download_ids) {
                    if (!download.is_swizzle || traced_images++ >= 8) continue;
                    const auto& image = download.GetImage(slot_images);
                    LOG_WARNING(HW_GPU, "Readback image: gpu={:#x}, cpu={:#x}, format={}, size={}x{}x{}, levels={}, layers={}, layer_stride={}, guest_bytes={}, download_bytes={}",
                                image.gpu_addr, image.cpu_addr, image.info.format,
                                image.info.size.width, image.info.size.height, image.info.size.depth,
                                image.info.resources.levels, image.info.resources.layers,
                                image.info.layer_stride, image.guest_size_bytes, image.unswizzled_size_bytes);
                }
            }
            auto download_map = runtime.DownloadStagingBuffer(total_size_bytes, true);
            for (const PendingDownload& download_info : download_ids) {
                if (download_info.is_swizzle) {
                    Image& image = download_info.GetImage(slot_images);
                    const auto copies = FullDownloadCopies(image.info);
                    if (depth_census_on) [[unlikely]] {
                        CensusNoteRead(image, false);
                    }
                    image.DownloadMemory(download_map, copies);
                    download_map.offset += Common::AlignUp(image.unswizzled_size_bytes, 64);
                }
            }
            uncommitted_async_buffers.emplace_back(download_map);
        }

        async_buffers.emplace_back(std::move(uncommitted_async_buffers));
        uncommitted_async_buffers.clear();
    }
    committed_downloads.emplace_back(std::move(uncommitted_downloads));
    uncommitted_downloads.clear();
}

template <class P>
void TextureCache<P>::PopAsyncFlushes() {
    const Common::WriteRing::SiteScope write_site{"texture download landing (PopAsyncFlushes)", true};
    if (committed_downloads.empty()) {
        return;
    }
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        const auto& download_ids = committed_downloads.front();
        if (download_ids.empty()) {
            committed_downloads.pop_front();
            async_buffers.pop_front();
            return;
        }
        auto download_map = std::move(async_buffers.front());
        for (size_t i = download_ids.size(); i > 0; i--) {
            auto& download_info = download_ids[i - 1];
            auto& download_buffer = download_map[download_info.async_buffer_id];
            if (download_info.is_swizzle) {
                const ImageBase& image = download_info.GetImage(slot_images);
                if constexpr (requires { runtime.IsFree(u64{}); }) {
                    // Fritchii's free guard: the old code destroyed a retired image when its
                    // download landed (the pop below). Counted when the GPU is not known past
                    // its delete yet; retained_image_keepers holds it now.
                    if (download_info.retired_image && !runtime.IsFree(download_info.retired_tick)) {
                        // The pop below drops this reference: the last one destroys the image.
                        if (download_info.retired_image.use_count() == 1) {
                            ++free_guard.freed_early;
                        }
                        if (free_guard.landed_early++ < 8) {
                            LOG_WARNING(HW_GPU,
                                        "Free guard: image {:#x} ({}x{}) deleted with a download "
                                        "queued; the download landed before the GPU passed the "
                                        "delete (tick {}), kept until it does",
                                        image.gpu_addr, image.info.size.width,
                                        image.info.size.height, download_info.retired_tick);
                        }
                    }
                }
                const auto copies = FullDownloadCopies(image.info);
                download_buffer.offset -= Common::AlignUp(image.unswizzled_size_bytes, 64);
                std::span<u8> download_span =
                    download_buffer.mapped_span.subspan(download_buffer.offset);
                DropCollectDownloadsIn(image.cpu_addr, image.guest_size_bytes);
                SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, download_span, swizzle_data_buffer);
                ClearStaleImage(image);
            } else {
                const BufferDownload& buffer_info = slot_buffer_downloads[download_info.object_id];
                std::span<u8> download_span =
                    download_buffer.mapped_span.subspan(download_buffer.offset);
                if (!collect_downloads.empty()) {
                    if (const auto cpu_addr = gpu_memory->GpuToCpuAddress(buffer_info.address)) {
                        DropCollectDownloadsIn(*cpu_addr, buffer_info.size);
                    }
                }
                MemReadLog::GuestWrite(*gpu_memory, buffer_info.address, buffer_info.size);
                gpu_memory->WriteBlockUnsafe(buffer_info.address, download_span.data(),
                                             buffer_info.size);
                ClearStaleGpu(buffer_info.address, buffer_info.size);
                slot_buffer_downloads.erase(download_info.object_id);
            }
        }
        for (auto& download_buffer : download_map) {
            async_buffers_death_ring.emplace_back(download_buffer);
        }
        committed_downloads.pop_front();
        async_buffers.pop_front();
    } else {
        const auto& download_ids = committed_downloads.front();
        if (download_ids.empty()) {
            committed_downloads.pop_front();
            return;
        }
        size_t total_size_bytes = 0;
        for (const PendingDownload& download_info : download_ids) {
            if (download_info.is_swizzle) {
                total_size_bytes += download_info.GetImage(slot_images).unswizzled_size_bytes;
            }
        }
        auto download_map = runtime.DownloadStagingBuffer(total_size_bytes);
        const size_t original_offset = download_map.offset;
        for (const PendingDownload& download_info : download_ids) {
            if (!download_info.is_swizzle) {
                continue;
            }
            Image& image = download_info.GetImage(slot_images);
            const auto copies = FullDownloadCopies(image.info);
            if (depth_census_on) [[unlikely]] {
                CensusNoteRead(image, false);
            }
            image.DownloadMemory(download_map, copies);
            download_map.offset += image.unswizzled_size_bytes;
        }
        // Wait for downloads to finish
        runtime.Finish();
        download_map.offset = original_offset;
        std::span<u8> download_span = download_map.mapped_span;
        for (const PendingDownload& download_info : download_ids) {
            if (!download_info.is_swizzle) {
                continue;
            }
            const ImageBase& image = download_info.GetImage(slot_images);
            const auto copies = FullDownloadCopies(image.info);
            DropCollectDownloadsIn(image.cpu_addr, image.guest_size_bytes);
            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, download_span, swizzle_data_buffer);
            ClearStaleImage(image);
            download_map.offset += image.unswizzled_size_bytes;
            download_span = download_span.subspan(image.unswizzled_size_bytes);
        }
        committed_downloads.pop_front();
    }
}

template <class P>
ImageId TextureCache<P>::DmaImageId(const Tegra::DMA::ImageOperand& operand, bool is_upload) {
    const ImageInfo dst_info(operand);
    const ImageId dst_id = FindDMAImage(dst_info, operand.address);
    if (!dst_id) {
        return NULL_IMAGE_ID;
    }
    auto& image = slot_images[dst_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        // No need to waste time on an image that's synced with guest
        return NULL_IMAGE_ID;
    }
    if (image.info.type == ImageType::e3D) {
        // Don't accelerate 3D images.
        return NULL_IMAGE_ID;
    }
    if (!is_upload && !image.info.dma_downloaded) {
        // Force a full sync.
        image.info.dma_downloaded = true;
        return NULL_IMAGE_ID;
    }
    const auto base = image.TryFindBase(operand.address);
    if (!base) {
        return NULL_IMAGE_ID;
    }
    return dst_id;
}

template <class P>
bool TextureCache<P>::IsRescaling() const noexcept {
    return is_rescaling;
}

template <class P>
bool TextureCache<P>::IsRescaling(const ImageViewBase& image_view) const noexcept {
    if (image_view.type == ImageViewType::Buffer) {
        return false;
    }
    const ImageBase& image = slot_images[image_view.image_id];
    return True(image.flags & ImageFlagBits::Rescaled);
}

template <class P>
bool TextureCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    bool is_modified = false;
    ForEachImageInRegion(addr, size, [&is_modified](ImageId, ImageBase& image) {
        if (False(image.flags & ImageFlagBits::GpuModified)) {
            return false;
        }
        is_modified = true;
        return true;
    });
    return is_modified;
}

template <class P>
std::pair<typename TextureCache<P>::Image*, BufferImageCopy> TextureCache<P>::DmaBufferImageCopy(
    const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& buffer_operand,
    const Tegra::DMA::ImageOperand& image_operand, ImageId image_id, bool modifies_image) {
    const auto [level, base] = PrepareDmaImage(image_id, image_operand.address, modifies_image);
    auto* image = &slot_images[image_id];
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    const u32 bpp = VideoCore::Surface::BytesPerBlock(image->info.format);
    const auto convert = [old_bpp = image_operand.bytes_per_pixel, bpp](u32 value) {
        return (old_bpp * value) / bpp;
    };
    const u32 base_x = convert(image_operand.params.origin.x.Value());
    const u32 base_y = image_operand.params.origin.y.Value();
    const u32 length_x = convert(copy_info.length_x);
    const u32 length_y = copy_info.length_y;

    const BufferImageCopy copy{
        .buffer_offset = 0,
        .buffer_size = buffer_size,
        .buffer_row_length = convert(buffer_operand.pitch),
        .buffer_image_height = buffer_operand.height,
        .image_subresource =
            {
                .base_level = static_cast<s32>(level),
                .base_layer = static_cast<s32>(base),
                .num_layers = 1,
            },
        .image_offset =
            {
                .x = static_cast<s32>(base_x),
                .y = static_cast<s32>(base_y),
                .z = 0,
            },
        .image_extent =
            {
                .width = length_x,
                .height = length_y,
                .depth = 1,
            },
    };
    return {image, copy};
}

template <class P>
void TextureCache<P>::DownloadImageIntoBuffer(typename TextureCache<P>::Image* image,
                                              typename TextureCache<P>::BufferType buffer,
                                              size_t buffer_offset,
                                              std::span<const VideoCommon::BufferImageCopy> copies,
                                              GPUVAddr address, size_t size) {
    if (depth_census_on) [[unlikely]] {
        CensusNoteRead(*image, true);
    }
    if constexpr (IMPLEMENTS_ASYNC_DOWNLOADS) {
        const BufferDownload new_buffer_download{address, size};
        auto slot = slot_buffer_downloads.insert(new_buffer_download);
        const PendingDownload new_download{false, uncommitted_async_buffers.size(), slot};
        async_flush_work_noted.store(true, std::memory_order_relaxed);
        uncommitted_downloads.emplace_back(new_download);
        auto download_map = runtime.DownloadStagingBuffer(size, true);
        uncommitted_async_buffers.emplace_back(download_map);
        std::array buffers{
            buffer,
            download_map.buffer,
        };
        std::array<size_t, 2> buffer_offsets{
            buffer_offset,
            download_map.offset,
        };
        image->DownloadMemory(buffers, buffer_offsets, copies);
    } else {
        image->DownloadMemory(buffer, buffer_offset, copies);
    }
}

template <class P>
void TextureCache<P>::RefreshContents(Image& image, ImageId image_id) {
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        // Only upload modified images
        return;
    }
    // FRITCHII_ONEF_INVAL_E: pages E wrote after R's place, in an invalidation it ran without
    // R (must be 0): the image takes that write as WriteMemory would have at its place, once
    // this upload is done, so its next use uploads again.
    const DAddr onef_addr = image.cpu_addr;
    const size_t onef_size = image.guest_size_bytes;
    const bool onef_ahead =
        VideoCommon::OneFollower::WrittenAheadOfRender(onef_addr, onef_size);
    SCOPE_EXIT {
        if (onef_ahead) {
            WriteMemory(onef_addr, onef_size);
        }
    };
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::TextureUpload};

    if (image.unmapped_tick != 0) {
        ++unmap_stats.reuploaded; // still Remapped: reached through a cached descriptor or binding
        if (image.unmapped_verify == 1) {
            ++unmap_stats.would_reuploaded;
            unmap_stats.would_reuploaded_gpu += True(image.flags & ImageFlagBits::GpuModified);
            image.unmapped_verify = 2;
        }
    }
    image.flags &= ~ImageFlagBits::CpuModified;
    // EXP76 & 2: the hash of the last whole upload, cleared here so any other way this refresh
    // changes the texels (layers, sparse pieces, the stale skip) leaves no hash behind.
    const u64 last_hash_lo = image.nx2_content_hash_lo;
    const u64 last_hash_hi = image.nx2_content_hash_hi;
    if ((VideoCore::Nx2Debug::Experiment(76) & 4) != 0) {
        ForgetContent(image_id, image, false);
    }
    image.nx2_content_hash_lo = 0;
    image.nx2_content_hash_hi = 0;

    // Images uploaded by layer stay tracked while some of their layers are dirty.
    if (False(image.flags & ImageFlagBits::Tracked)) {
        TrackImage(image, image_id);
    }

    // Made where a dropped GPU-written image left stale bytes (MarkStale): nothing to upload.
    // Pages the buffer cache wrote on the GPU may hold data written back since: upload those.
    if (TcSkipStale() && image.dirty_layers.empty() &&
        False(image.flags & (ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
                             ImageFlagBits::SparseTiles | ImageFlagBits::Unbacked)) &&
        IsAllStale(image.cpu_addr, image.guest_size_bytes)) {
        if (!Tegra::Nx2MayBeGpuWritten(image.cpu_addr, image.guest_size_bytes)) {
            ++tc_stale_skips;
            tc_stale_skip_bytes += image.guest_size_bytes;
            ++lost_stats.made_skipped;
            image.lost_contents = LostSkipped;
            // FRITCHII_STALE_ZERO (live, default 1): zero instead of undefined. A new image's
            // memory often held another image (the EXP42 pool hands out deleted targets'
            // VkImages): where the game samples before drawing it whole, that image's texels
            // showed (other text over a dialogue box, zortos293/Lichii #1310).
            if (ClearLostToZero(image)) {
                ++lost_stats.made_zeroed;
                image.lost_contents |= LostZeroed;
            } else {
                runtime.TransitionImageLayout(image);
            }
            return;
        }
        ++tc_stale_gpu_written;
    } else if (stale_page_count != 0 && IsAnyStale(image.cpu_addr, image.guest_size_bytes)) {
        // The upload below brings guest bytes older than the dropped image's GPU writes.
        ++lost_stats.made_uploaded;
        image.lost_contents = LostUploaded;
    }

    if (True(image.flags & ImageFlagBits::SparseLayers)) {
        RefreshSparseLayers(image);
        return;
    }
    if (True(image.flags & ImageFlagBits::SparseTiles)) {
        RefreshSparseTiles(image);
        return;
    }
    if (!image.dirty_layers.empty() && UploadWrittenLayers(image)) {
        runtime.InsertUploadMemoryBarrier();
        return;
    }
    image.dirty_layers.clear();

    if (True(image.flags & ImageFlagBits::Unbacked)) {
        // Nothing to upload into (and staging it could take tens of GB).
        runtime.TransitionImageLayout(image);
        return;
    }

    if (image.info.num_samples > 1 && !runtime.CanUploadMSAA()) {
        LOG_WARNING(HW_GPU, "MSAA image uploads are not implemented");
        runtime.TransitionImageLayout(image);
        return;
    }
    // EXP76 & 2: a whole upload whose guest bytes hash as they did at this image's last whole
    // upload already has its texels (the image was tracked again above, before the bytes are
    // read, so a CPU write after this marks it again). Never for GPU-written images.
    const int keep_mode = VideoCore::Nx2Debug::Experiment(76);
    constexpr u32 HashMinBytes = 64 * 1024;
    const bool hashable = (keep_mode & 2) != 0 && image.guest_size_bytes >= HashMinBytes &&
                          False(image.flags & ImageFlagBits::GpuModified);
    if (hashable && (last_hash_lo | last_hash_hi) != 0) {
        ++nx2_upload_hash_checks;
        const u128 hash = GuestContentHash(image);
        if (hash[0] == last_hash_lo && (hash[1] | 1) == last_hash_hi &&
            False(image.flags & ImageFlagBits::CpuModified)) {
            ++nx2_upload_skips;
            nx2_upload_skip_bytes += image.guest_size_bytes;
            image.nx2_content_hash_lo = last_hash_lo;
            image.nx2_content_hash_hi = last_hash_hi;
            if ((keep_mode & 4) != 0) {
                nx2_content_index[ContentKey{ContentShape(image.info), last_hash_lo,
                                             last_hash_hi}] = image_id;
            }
            return;
        }
    }

    if (True(image.flags & ImageFlagBits::AsynchronousDecode)) {
        QueueAsyncDecode(image, image_id);
        return;
    }

    const bool gpu_unswizzle_enabled = Settings::values.gpu_unswizzle_enabled.GetValue();

    if (gpu_unswizzle_enabled &&
        IsPixelFormatBCn(image.info.format) &&
        image.info.type == ImageType::e3D &&
        image.info.resources.levels == 1 &&
        image.info.resources.layers == 1 &&
        MapSizeBytes(image) >= gpu_unswizzle_maxsize &&
        False(image.flags & ImageFlagBits::GpuModified)) {

        QueueAsyncUnswizzle(image, image_id);
        return;
    }
    ++images_uploaded;
    uploaded_bytes += image.guest_size_bytes;
    largest_upload = (std::max<u64>)(largest_upload, image.guest_size_bytes);
    // EDEN_NX2_EXP7: log the uploads that hold a draw back (over 2 ms).
    const bool upload_trace = VideoCore::Nx2Debug::Experiment(7) != 0;
    const auto upload_begin = upload_trace ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
    auto staging_end = upload_begin;
    // Page faults of the whole process meanwhile: ~256 per MiB when the guest memory was paged
    // out (cold-hinted) or the staging memory is new.
    const u64 faults_begin = upload_trace ? Common::GetProcessPageFaults() : 0;
    if (TryZeroUpload(image)) {
        // FRITCHII_ZERO_UPLOAD: cleared on the GPU, what the upload of these bytes writes.
    } else if (!UploadInChunks(image)) {
        auto staging = runtime.UploadStagingBuffer(MapSizeBytes(image));
        if (upload_trace) [[unlikely]] {
            staging_end = std::chrono::steady_clock::now();
        }
        // FRITCHII_HITCH_SPLIT: the CPU side of texture uploads in the hitch line.
        const Common::Nx2Hitch::SplitScope hitch_timed{Common::Nx2Hitch::Cause::TextureUpload,
                                                       MapSizeBytes(image)};
        UploadImageContents(image, staging);
        runtime.InsertUploadMemoryBarrier();
    }
    // EXP76 & 2: the bytes just uploaded, hashed after the upload read them. A CPU write since
    // the image was tracked marked it CpuModified again: then no hash (the next refresh uploads).
    if (hashable && False(image.flags & ImageFlagBits::CpuModified)) {
        const u128 hash = GuestContentHash(image);
        if (False(image.flags & ImageFlagBits::CpuModified)) {
            image.nx2_content_hash_lo = hash[0];
            image.nx2_content_hash_hi = hash[1] | 1; // never 0/0
            if ((keep_mode & 4) != 0) {
                nx2_content_index[ContentKey{ContentShape(image.info), image.nx2_content_hash_lo,
                                             image.nx2_content_hash_hi}] = image_id;
            }
        }
    }
    if (upload_trace) [[unlikely]] {
        using Ms = std::chrono::duration<double, std::milli>;
        const double ms = Ms(std::chrono::steady_clock::now() - upload_begin).count();
        static std::atomic<u32> logged{};
        if (ms > 2.0 && logged.fetch_add(1, std::memory_order_relaxed) < 300) {
            LOG_WARNING(HW_GPU,
                        "Slow upload {:.2f} ms (staging {:.2f} ms, {} page faults): format {} "
                        "type {} {}x{}x{} levels {} layers {}, {} KiB guest, {} KiB host, "
                        "flags {:#x}",
                        ms, Ms(staging_end - upload_begin).count(),
                        Common::GetProcessPageFaults() - faults_begin,
                        static_cast<u32>(image.info.format), static_cast<u32>(image.info.type),
                        image.info.size.width, image.info.size.height, image.info.size.depth,
                        image.info.resources.levels, image.info.resources.layers,
                        image.guest_size_bytes >> 10, MapSizeBytes(image) >> 10,
                        static_cast<u32>(image.flags));
        }
    }
    // The host GPU has its own copy now and the guest rarely reads this memory again: TickFrame
    // marks it cold, so the OS pages it out before anything else when it needs RAM (title 1
    // streams several GB of textures through it).
    constexpr u64 ColdHintMinBytes = 64_KiB;
    if (image.guest_size_bytes >= ColdHintMinBytes &&
        False(image.flags & ImageFlagBits::GpuModified)) {
        if (True(image.flags & ImageFlagBits::Sparse)) {
            // Non-contiguous: each mapped piece (title 1 maps its own memory into these).
            ForEachSparseSegment(image, [&](GPUVAddr, DAddr address, size_t size) {
                uploaded_ranges.emplace_back(address, size);
            });
        } else {
            uploaded_ranges.emplace_back(image.cpu_addr, image.guest_size_bytes);
        }
    }
}

/// FRITCHII_ZERO_UPLOAD: bytes looked at until the first non-zero one (all of them when zero).
[[nodiscard]] inline size_t ZeroPrefixBytes(std::span<const u8> bytes) noexcept {
    const u8* const data = bytes.data();
    const size_t size = bytes.size();
    // 4 KiB at a time: OR-ing words vectorizes; a block with data stops the walk.
    constexpr size_t Block = 4096;
    size_t offset = 0;
    for (; offset + Block <= size; offset += Block) {
        u64 bits = 0;
        for (size_t i = 0; i < Block; i += sizeof(u64)) {
            u64 word;
            std::memcpy(&word, data + offset + i, sizeof(word));
            bits |= word;
        }
        if (bits != 0) {
            return offset;
        }
    }
    for (; offset < size; ++offset) {
        if (data[offset] != 0) {
            return offset;
        }
    }
    return size;
}

template <class P>
bool TextureCache<P>::TryZeroUpload(Image& image) {
    const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_zero_upload,
                                                      "FRITCHII_ZERO_UPLOAD");
    if (mode <= 0) {
        return false;
    }
    if constexpr (!requires { image.ClearToZero(); }) {
        return false;
    } else {
        // Only where zero guest bytes upload as zero texels in every level and layer and the
        // upload covers the whole host image: plain (not converted, not GPU-decoded), uncompressed
        // single-sample images whose host image is the guest image (not scaled, padded or
        // sparse), not a buffer.
        constexpr ImageFlagBits excluded =
            ImageFlagBits::AcceleratedUpload | ImageFlagBits::Converted |
            ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
            ImageFlagBits::Unbacked | ImageFlagBits::TrimmedLevels |
            ImageFlagBits::AsynchronousDecode | ImageFlagBits::IsDecoding |
            ImageFlagBits::ZeroFilled | ImageFlagBits::UnwrittenSparse | ImageFlagBits::Rescaled |
            ImageFlagBits::HostPadded | ImageFlagBits::ScaleLimited;
        constexpr u64 MinBytes = 256_KiB;
        if (True(image.flags & excluded) || image.info.type == ImageType::Buffer ||
            image.info.num_samples != 1 || image.host_level_offset != 0 ||
            image.guest_size_bytes < MinBytes ||
            VideoCore::Surface::DefaultBlockWidth(image.info.format) != 1 ||
            VideoCore::Surface::DefaultBlockHeight(image.info.format) != 1) {
            return false;
        }
        // A slice of a deep block-linear 3D texture spans 16x its texels in guest memory: looking
        // at all of it costs more than the upload's own read.
        if (u64{image.guest_size_bytes} > 2 * u64{image.unswizzled_size_bytes}) {
            ++zero_upload.sparse_skipped;
            return false;
        }
        const u64 start = Common::Nx2Hitch::NowNs();
        // The bytes the upload would read (GpuGuestMemory, unmapped pages as zeros), in pieces so
        // a range that is not contiguous copies at most one piece.
        constexpr size_t Piece = 4_MiB;
        const size_t total = image.guest_size_bytes;
        size_t looked = 0;
        bool zero = true;
        for (size_t offset = 0; offset < total; offset += Piece) {
            const size_t size = (std::min)(Piece, total - offset);
            Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> bytes(
                *gpu_memory, image.gpu_addr + offset, size, &swizzle_data_buffer);
            const size_t prefix = ZeroPrefixBytes(std::span<const u8>(bytes.data(), size));
            looked += prefix;
            if (prefix != size) {
                zero = false;
                break;
            }
        }
        const u64 ns = Common::Nx2Hitch::NowNs() - start;
        Common::Nx2Hitch::Add(Common::Nx2Hitch::Cause::ZeroUpload, ns, looked);
        if (!zero) {
            ++zero_upload.data_images;
            zero_upload.data_bytes += looked;
            zero_upload.data_ns += ns;
            return false;
        }
        ++zero_upload.zero_images;
        zero_upload.zero_bytes += total;
        zero_upload.zero_ns += ns;
        if (total >= 4_MiB && zero_upload.logged < 40) {
            ++zero_upload.logged;
            LOG_INFO(HW_GPU,
                     "Zero upload (FRITCHII_ZERO_UPLOAD={}): format {} type {} {}x{}x{} levels {} "
                     "layers {} at {:#x}, {} KiB guest all zero, {} (checked in {:.2f} ms)",
                     mode, static_cast<u32>(image.info.format), static_cast<u32>(image.info.type),
                     image.info.size.width, image.info.size.height, image.info.size.depth,
                     image.info.resources.levels, image.info.resources.layers, image.gpu_addr,
                     total >> 10, mode == 1 ? "cleared on the GPU" : "uploaded (count only)",
                     static_cast<double>(ns) / 1e6);
        }
        if (mode != 1) {
            return false; // 2: count only, the upload runs as before
        }
        image.ClearToZero();
        return true;
    }
}

template <class P>
bool TextureCache<P>::ClearLostToZero(Image& image) {
    if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_stale_zero, "FRITCHII_STALE_ZERO",
                                         1) <= 0) {
        return false;
    }
    if constexpr (!requires { image.ClearToZero(true); }) {
        return false;
    } else {
        // Formats a transfer clear can write (uncompressed, not converted), single-sample.
        constexpr ImageFlagBits excluded =
            ImageFlagBits::Converted | ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
            ImageFlagBits::SparseTiles | ImageFlagBits::Unbacked |
            ImageFlagBits::AsynchronousDecode | ImageFlagBits::IsDecoding;
        if (True(image.flags & excluded) || image.info.type == ImageType::Buffer ||
            image.info.num_samples != 1 ||
            VideoCore::Surface::DefaultBlockWidth(image.info.format) != 1 ||
            VideoCore::Surface::DefaultBlockHeight(image.info.format) != 1) {
            return false;
        }
        image.ClearToZero(true); // the image in use, scaled or not
        return true;
    }
}

template <class P>
void TextureCache<P>::LogZeroUploads() {
    ZeroUploadStats& s = zero_upload;
    const s64 now = static_cast<s64>(Common::Nx2Hitch::NowNs());
    if (s.last_log_ns == 0) {
        s.last_log_ns = now;
        return;
    }
    if (now - s.last_log_ns < 10'000'000'000) {
        return;
    }
    s.last_log_ns = now;
    if (s.zero_images + s.data_images + s.sparse_skipped != 0) {
        const int mode = VideoCore::Nx2Debug::g_zero_upload.load(std::memory_order_relaxed);
        LOG_INFO(HW_GPU,
                 "Zero uploads in 10 s (FRITCHII_ZERO_UPLOAD={}): {} uploads all zero ({} MiB "
                 "guest, checked in {:.1f} ms), {}; {} had data ({} MiB looked at before the first "
                 "non-zero byte, {:.1f} ms); {} not checked (3D slices, guest footprint over 2x)",
                 mode, s.zero_images, s.zero_bytes >> 20, static_cast<double>(s.zero_ns) / 1e6,
                 mode == 1 ? "cleared on the GPU" : "uploaded as before (count only)",
                 s.data_images, s.data_bytes >> 20, static_cast<double>(s.data_ns) / 1e6,
                 s.sparse_skipped);
    }
    s.zero_images = s.zero_bytes = s.zero_ns = 0;
    s.data_images = s.data_bytes = s.data_ns = 0;
    s.sparse_skipped = 0;
}

template <class P>
bool TextureCache<P>::UploadInChunks(Image& image) {
    // A whole upload of a very large image needs a staging buffer of its full size (rounded up to
    // a power of two: 1 GiB for 600 MiB of data), which can fail when memory is tight.
    constexpr u64 min_size = 64_MiB;
    constexpr u64 chunk_size = 32_MiB;
    constexpr ImageFlagBits excluded = ImageFlagBits::Sparse | ImageFlagBits::Converted |
                                       ImageFlagBits::AcceleratedUpload | ImageFlagBits::Unbacked |
                                       ImageFlagBits::TrimmedLevels;
    if (True(image.flags & excluded) || image.info.type != ImageType::e2D ||
        MapSizeBytes(image) < min_size) {
        return false;
    }
    for (u64 offset = 0; offset < image.guest_size_bytes; offset += chunk_size) {
        const std::array range{
            std::pair<u64, u64>{offset, std::min<u64>(offset + chunk_size, image.guest_size_bytes)}};
        const auto plan = PlanUnswizzleImageRows(image.gpu_addr, image.info, range);
        if (!plan.layout_valid) {
            if (offset == 0) {
                return false; // layout not handled: one staging buffer after all
            }
            continue;
        }
        if (plan.copies.empty()) {
            continue; // padding between levels or layers
        }
        auto staging = runtime.UploadStagingBuffer(plan.total_host_bytes);
        ExecuteUnswizzlePlan(*gpu_memory, plan, staging.mapped_span);
        image.UploadMemory(staging, plan.copies);
    }
    runtime.InsertUploadMemoryBarrier();
    return true;
}

template <class P>
bool TextureCache<P>::UploadsByLayer(const ImageBase& image) const noexcept {
    // Title 1's terrain arrays (8192x8192x68 ASTC, 1.45 GB each) get a few layers streamed in at a
    // time; re-decoding all of them for each took a 2 GiB staging buffer and stalled the GPU.
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        if (True(image.flags & ImageFlagBits::SparseLayers)) {
            return true;
        }
        return True(image.flags & ImageFlagBits::AcceleratedUpload) &&
               image.info.type == ImageType::e2D && image.info.resources.layers > 1 &&
               image.guest_size_bytes >= LayerUploadMinSize &&
               False(image.flags & (ImageFlagBits::Unbacked | ImageFlagBits::Sparse |
                                    ImageFlagBits::Remapped));
    } else {
        return false;
    }
}

template <class P>
void TextureCache<P>::MarkLayersWritten(Image& image, DAddr cpu_addr, size_t size) {
    if (True(image.flags & ImageFlagBits::CpuModified) && image.dirty_layers.empty()) {
        return; // all of it is dirty already
    }
    const u32 layers = static_cast<u32>(image.info.resources.layers);
    if (False(image.flags & ImageFlagBits::CpuModified)) {
        image.flags |= ImageFlagBits::CpuModified;
        image.dirty_layers.assign(Common::DivCeil(layers, 64U), 0);
    }
    const DAddr begin = (std::max<DAddr>)(cpu_addr, image.cpu_addr) - image.cpu_addr;
    const DAddr end = (std::min<DAddr>)(cpu_addr + size, image.cpu_addr + image.guest_size_bytes) -
                      image.cpu_addr;
    const u32 stride = image.info.layer_stride;
    if (end <= begin || stride == 0) {
        return;
    }
    const u32 last = (std::min)(static_cast<u32>((end - 1) / stride), layers - 1);
    for (u32 layer = static_cast<u32>(begin / stride); layer <= last; ++layer) {
        image.dirty_layers[layer / 64] |= u64{1} << (layer % 64);
    }
}

template <class P>
void TextureCache<P>::MarkRangeWritten(Image& image, DAddr cpu_addr, size_t size) {
    if (True(image.flags & ImageFlagBits::CpuModified) && image.dirty_ranges.empty()) {
        return; // all of it is dirty already
    }
    const DAddr begin = (std::max<DAddr>)(cpu_addr, image.cpu_addr);
    const DAddr end = (std::min<DAddr>)(cpu_addr + size, image.cpu_addr + image.guest_size_bytes);
    if (end <= begin) {
        return;
    }
    image.flags |= ImageFlagBits::CpuModified;
    image.dirty_ranges.emplace_back(begin - image.cpu_addr, end - begin);
}

template <class P>
void TextureCache<P>::RefreshSparseTiles(Image& image) {
    if constexpr (requires { &Image::BindTiles; }) {
        // The ranges written since the last upload, or, for a new image, all of it that can hold
        // data (the tiles the game maps there, or the blocks written).
        std::vector<std::pair<u64, u64>> ranges = std::move(image.dirty_ranges);
        image.dirty_ranges.clear();
        if (ranges.empty()) {
            ranges = Tegra::Nx2SparseDataRanges(image.gpu_addr, image.guest_size_bytes);
        }
        const std::vector<LevelRect> rects =
            SparseTileRects(image.info, ranges, image.SparseTileWidth(), image.SparseTileHeight(),
                            image.SparseTailLevel());
        if (rects.empty() || !BindSparseTiles(image, rects)) {
            runtime.TransitionImageLayout(image);
            return;
        }
        const size_t bytes = UnswizzledRectsBytes(image.info, rects);
        auto staging = runtime.UploadStagingBuffer(bytes);
        const std::vector<BufferImageCopy> copies =
            UnswizzleImageRects(*gpu_memory, image.gpu_addr, image.info, rects, staging.mapped_span);
        image.UploadMemory(staging, copies);
        runtime.InsertUploadMemoryBarrier();
        ++images_uploaded;
        uploaded_bytes += bytes;
    }
}

template <class P>
bool TextureCache<P>::BindSparseTiles(Image& image, std::span<const LevelRect> rects) {
    if constexpr (requires { &Image::BindTiles; }) {
        const u64 bound_bytes = ImageMemoryBytes(image);
        const bool bound = image.BindTiles(rects);
        // RegisterImage counts what an image has bound by then; later binds are added here.
        if (True(image.flags & ImageFlagBits::Registered)) {
            total_used_memory += ImageMemoryBytes(image) - bound_bytes;
        }
        return bound;
    } else {
        return false;
    }
}

template <class P>
bool TextureCache<P>::BackSparseTilesWhole(Image& image) {
    if constexpr (requires { &Image::BackTilesWhole; }) {
        const u64 bound_bytes = ImageMemoryBytes(image);
        const bool backed = image.BackTilesWhole();
        if (True(image.flags & ImageFlagBits::Registered)) {
            total_used_memory += ImageMemoryBytes(image) - bound_bytes;
        }
        return backed;
    } else {
        return false;
    }
}

template <class P>
bool TextureCache<P>::UploadWrittenLayers(Image& image) {
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        const u32 num_layers = static_cast<u32>(image.info.resources.layers);
        std::vector<u32> layers;
        for (u32 layer = 0; layer < num_layers; ++layer) {
            if ((image.dirty_layers[layer / 64] >> (layer % 64)) & 1) {
                layers.push_back(layer);
            }
        }
        image.dirty_layers.clear();
        if (layers.empty() || layers.size() == num_layers) {
            return false;
        }
        UploadLayers(image, layers);
        return true;
    } else {
        return false;
    }
}

template <class P>
void TextureCache<P>::RefreshSparseLayers(Image& image) {
    if constexpr (requires { &Image::BindLayers; }) {
        // The layers written since the last upload, or, for a new image, every layer any of whose
        // blocks has been written.
        const u32 num_layers = static_cast<u32>(image.info.resources.layers);
        const u64 stride = image.info.layer_stride;
        std::vector<u32> layers;
        for (u32 layer = 0; layer < num_layers; ++layer) {
            const bool written =
                image.dirty_layers.empty()
                    ? !Tegra::Nx2SparseUnwritten(image.gpu_addr + layer * stride, stride)
                    : ((image.dirty_layers[layer / 64] >> (layer % 64)) & 1) != 0;
            if (written) {
                layers.push_back(layer);
            }
        }
        image.dirty_layers.clear();
        const std::vector<u32> bound =
            layers.empty() ? std::vector<u32>{} : BindSparseLayers(image, layers);
        if (bound.empty()) {
            runtime.TransitionImageLayout(image);
            return;
        }
        UploadLayers(image, bound);
        runtime.InsertUploadMemoryBarrier();
    }
}

template <class P>
std::vector<u32> TextureCache<P>::BindSparseLayers(Image& image, std::span<const u32> layers) {
    if constexpr (requires { &Image::BindLayers; }) {
        const u64 bound_bytes = ImageMemoryBytes(image);
        std::vector<u32> bound = image.BindLayers(layers);
        // RegisterImage counts what an image has bound by then; later binds are added here.
        if (True(image.flags & ImageFlagBits::Registered)) {
            total_used_memory += ImageMemoryBytes(image) - bound_bytes;
        }
        return bound;
    } else {
        return {};
    }
}

template <class P>
void TextureCache<P>::UploadLayers(Image& image, std::span<const u32> layers) {
    if constexpr (requires { &Runtime::AccelerateImageUploadLayers; }) {
        // At most 32 MiB of layers per staging buffer, as UploadInChunks: all of a sparse array's
        // bound layers at once took a 128 MiB one that an 8 GB laptop had no memory for.
        constexpr u64 ChunkBytes = 32_MiB;
        const size_t chunk_layers = static_cast<size_t>(
            (std::max<u64>)(1, ChunkBytes / (std::max<u64>)(image.info.layer_stride, 1)));
        if (layers.size() > chunk_layers) {
            for (size_t first = 0; first < layers.size(); first += chunk_layers) {
                UploadLayers(image, layers.subspan(first, (std::min)(chunk_layers,
                                                                     layers.size() - first)));
            }
            return;
        }
        if (False(image.flags & ImageFlagBits::AcceleratedUpload)) {
            // Plain formats: each layer unswizzled on its own into its layer of the image.
            ImageInfo layer_info = image.info;
            layer_info.resources.layers = 1;
            const size_t layer_size = CalculateUnswizzledSizeBytes(layer_info);
            auto staging = runtime.UploadStagingBuffer(layer_size * layers.size());
            std::vector<BufferImageCopy> copies;
            for (size_t i = 0; i < layers.size(); ++i) {
                const GPUVAddr layer_addr = image.gpu_addr + u64{layers[i]} * image.info.layer_stride;
                Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead>
                    swizzle_data(*gpu_memory, layer_addr, image.info.layer_stride,
                                 &swizzle_data_buffer);
                auto layer_copies =
                    UnswizzleImage(*gpu_memory, layer_addr, layer_info, swizzle_data,
                                   staging.mapped_span.subspan(i * layer_size, layer_size));
                for (BufferImageCopy copy : layer_copies) {
                    copy.buffer_offset += i * layer_size;
                    copy.image_subresource.base_layer = static_cast<s32>(layers[i]);
                    copies.push_back(copy);
                }
            }
            image.UploadMemory(staging, copies);
            ++images_uploaded;
            uploaded_bytes += u64{image.info.layer_stride} * layers.size();
            ++layer_uploads;
            return;
        }
        // Each layer's mip chain from the first level the host keeps on (as UploadImageContents).
        auto uploads = FullUploadSwizzles(image.info);
        const size_t level_offset = uploads[image.host_level_offset].buffer_offset;
        const size_t tail = image.info.layer_stride - level_offset;
        auto staging = runtime.UploadStagingBuffer(tail * layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            gpu_memory->ReadBlock(image.gpu_addr + u64{layers[i]} * image.info.layer_stride +
                                      level_offset,
                                  staging.mapped_span.data() + i * tail, tail,
                                  VideoCommon::CacheType::NoTextureCache);
        }
        uploads.erase(uploads.begin(), uploads.begin() + image.host_level_offset);
        for (auto& upload : uploads) {
            upload.buffer_offset -= level_offset;
        }
        runtime.AccelerateImageUploadLayers(image, staging, uploads, layers);
        ++images_uploaded;
        uploaded_bytes += tail * layers.size();
        ++layer_uploads;
    }
}

template <class P>
template <typename StagingBuffer>
void TextureCache<P>::UploadImageContents(Image& image, StagingBuffer& staging) {
    const std::span<u8> mapped_span = staging.mapped_span;
    const GPUVAddr gpu_addr = image.gpu_addr;

    if (True(image.flags & ImageFlagBits::AcceleratedUpload) && image.host_level_offset != 0) {
        // Only levels >= host_level_offset exist on the host: read each layer's tail of the mip
        // chain into a compact buffer (layer stride = that tail's size) and decode those levels.
        auto uploads = FullUploadSwizzles(image.info);
        const size_t level_offset = uploads[image.host_level_offset].buffer_offset;
        const size_t tail = image.info.layer_stride - level_offset;
        for (s32 layer = 0; layer < image.info.resources.layers; ++layer) {
            gpu_memory->ReadBlock(gpu_addr + layer * image.info.layer_stride + level_offset,
                                  mapped_span.data() + layer * tail, tail,
                                  VideoCommon::CacheType::NoTextureCache);
        }
        uploads.erase(uploads.begin(), uploads.begin() + image.host_level_offset);
        for (auto& upload : uploads) {
            upload.buffer_offset -= level_offset;
        }
        runtime.AccelerateImageUpload(image, staging, uploads, 0, 0);
        return;
    }
    if (True(image.flags & ImageFlagBits::AcceleratedUpload)) {
        gpu_memory->ReadBlock(gpu_addr, mapped_span.data(), mapped_span.size_bytes(),
                              VideoCommon::CacheType::NoTextureCache);
        const auto uploads = FullUploadSwizzles(image.info);
        runtime.AccelerateImageUpload(image, staging, uploads, 0, 0);
        return;
    }

    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> swizzle_data(
        *gpu_memory, gpu_addr, image.guest_size_bytes, &swizzle_data_buffer);
    if (True(image.flags & ImageFlagBits::Converted)) {
        unswizzle_data_buffer.resize_destructive(image.unswizzled_size_bytes);
        auto copies = UnswizzleImage(*gpu_memory, gpu_addr, image.info, swizzle_data,
                                    unswizzle_data_buffer, &unswizzle_worker);
        ConvertImage(unswizzle_data_buffer, image.info, mapped_span, copies);
        image.UploadMemory(staging, copies);
    } else {
        const auto copies = UnswizzleImage(*gpu_memory, gpu_addr, image.info, swizzle_data,
                                          mapped_span, &unswizzle_worker);
        image.UploadMemory(staging, copies);
    }
}

template <class P>
ImageViewId TextureCache<P>::CreateImageView(const TICEntry& raw_config) {
    const bool native = gpu_memory->UsesNativeTiling();
    if (native && raw_config.format.Value() == Tegra::Texture::TextureFormat{} &&
        RenderTrace::Enabled()) {
        static u32 traces{};
        if (traces++ < 64) {
            LOG_WARNING(HW_GPU,
                        "Zero-format image TIC: image={:#x} size={}x{}x{} header={} type={} "
                        "raw={:016x}/{:016x}/{:016x}/{:016x}",
                        raw_config.Address(), raw_config.Width(), raw_config.Height(),
                        raw_config.Depth(), static_cast<u32>(raw_config.header_version.Value()),
                        static_cast<u32>(raw_config.texture_type.Value()), raw_config.raw[0],
                        raw_config.raw[1], raw_config.raw[2], raw_config.raw[3]);
        }
    }
    if (native && raw_config.BaseLayer() && NativeDrawDiagnostics()) {
        static u32 texture_view_traces{};
        if (texture_view_traces++ < 8) {
            LOG_WARNING(HW_GPU, "Native texture view header: address={:#x}, dimensions={}x{}x{}, type={}, legacy_layer={}, raw={:016x}/{:016x}/{:016x}/{:016x}",
                        raw_config.Address(), raw_config.Width(), raw_config.Height(), raw_config.Depth(),
                        static_cast<u32>(raw_config.texture_type.Value()), raw_config.BaseLayer(),
                        raw_config.raw[0], raw_config.raw[1], raw_config.raw[2], raw_config.raw[3]);
        }
    }
    const TICEntry config = TextureViewHeader(raw_config, native);
    const ImageInfo info(config);
    ++bindings_epoch;
    if (info.type == ImageType::Buffer) {
        const ImageViewInfo view_info(config, 0);
        Churn::Add(Churn::ImageViewsCreated);
        return slot_image_views.insert(runtime, info, view_info, config.Address());
    }
    const u32 layer_offset = config.BaseLayer() * info.layer_stride;
    const GPUVAddr image_gpu_addr = config.Address() - layer_offset;
    ImageId image_id = FindImage(info, image_gpu_addr, RelaxedOptions{});
    if (!image_id) {
        image_id = InsertImage(info, image_gpu_addr, RelaxedOptions{});
        // EXP77 (live; fable52_answer_keeptex_2 2b): the game samples memory where the cold
        // sweep just took a GPU-written target: that shape's contents are read in later frames,
        // so the sweep keeps it from now on (GuardSampled), as if the target had been sampled.
        if (image_id && !nx2_swept_shapes.empty()) [[unlikely]] {
            if (const auto swept = nx2_swept_shapes.find(image_gpu_addr);
                swept != nx2_swept_shapes.end()) {
                RelocationShape& shape = relocation_shapes[swept->second];
                if ((shape.guard & GuardSampled) == 0) {
                    shape.guard |= GuardSampled;
                    ++nx2_sweep_guards;
                }
                relocation_guarded_addresses[image_gpu_addr] |= GuardSampled;
                nx2_swept_shapes.erase(swept);
            }
        }
        // FRITCHII_SWEEP_SUPERSEDED: a texture made where the sweep deleted an image means the
        // game read that memory again (its contents came from guest memory, not the GPU's).
        if (image_id && !superseded_swept.empty()) [[unlikely]] {
            if (const auto swept = superseded_swept.find(image_gpu_addr);
                swept != superseded_swept.end()) {
                ++superseded_stats.sampled_after;
                superseded_swept.erase(swept);
            }
        }
        // FRITCHII_COVERED_RETIRE: the same for images retired as covered.
        if (image_id && !covered_swept.empty()) [[unlikely]] {
            if (const auto swept = covered_swept.find(image_gpu_addr);
                swept != covered_swept.end()) {
                ++covered_stats.sampled_after;
                if (covered_sampled_logged < 20) {
                    ++covered_sampled_logged;
                    const ImageBase& made = slot_images[image_id];
                    LOG_WARNING(HW_GPU,
                                "Covered retire: format {} {}x{} at {:#x} sampled {} frames after "
                                "an image there was retired as covered",
                                static_cast<u32>(made.info.format), made.info.size.width,
                                made.info.size.height, image_gpu_addr,
                                frame_tick - swept->second);
                }
                covered_swept.erase(swept);
            }
        }
        if (image_id) {
            NoteNewSampledImage(image_id, image_gpu_addr);
            if ((VideoCore::Nx2Debug::Experiment(76) & 4) != 0) [[unlikely]] {
                NoteContentLookup(image_id);
            }
        }
    }
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    ImageBase& image = slot_images[image_id];
    if (True(image.flags & ImageFlagBits::HostPadded)) [[unlikely]] {
        // EXP67: sampled with normalized coordinates over the padded host image this time;
        // its address is never padded again.
        ++pad_sampled;
        pad_unsafe_slots.insert(image.gpu_addr);
        if (pad_logged < 20) {
            ++pad_logged;
            LOG_WARNING(HW_GPU,
                        "Host pad (EXP67): format {} {}x{} at {:#x} is sampled as a texture; "
                        "its address is no longer padded",
                        static_cast<u32>(image.info.format), image.info.size.width,
                        image.info.size.height, image.gpu_addr);
        }
    }
    const SubresourceBase base = image.TryFindBase(config.Address()).value();
    ASSERT(base.level == 0);
    const ImageViewInfo view_info(config, base.layer);
    const ImageViewId image_view_id = FindOrEmplaceImageView(image_id, view_info);
    ImageViewBase& image_view = slot_image_views[image_view_id];
    image_view.flags |= ImageViewFlagBits::Strong;
    image.flags |= ImageFlagBits::Strong;
    return image_view_id;
}

template <class P>
ImageId TextureCache<P>::FindOrInsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                           RelaxedOptions options) {
    if (const ImageId image_id = FindImage(info, gpu_addr, options); image_id) {
        return image_id;
    }
    return InsertImage(info, gpu_addr, options);
}

template <class P>
ImageId TextureCache<P>::FindImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                   RelaxedOptions options) {
    const std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
    if (!cpu_addr) {
        return ImageId{};
    }
    const bool broken_views =
        runtime.HasBrokenTextureViewFormats() || True(options & RelaxedOptions::ForceBrokenViews);
    const bool native_bgr = runtime.HasNativeBgr();
    const bool flexible_formats = True(options & RelaxedOptions::Format);
    ImageId image_id{};
    boost::container::small_vector<ImageId, 8> image_ids;
    const int keep_mode = VideoCore::Nx2Debug::Experiment(76);
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (True(existing_image.flags & ImageFlagBits::Remapped)) {
            // EXP76 & 1 (fable52_answer_keeptex 2a): the game remapped the range and the same
            // texture is asked for at the same GPU and device address: keep the image instead of
            // making it again. It stays CpuModified, so its contents are uploaded from guest
            // memory as for a new image (or skipped by EXP76 & 2 when the bytes are unchanged).
            const ImageInfo& old = existing_image.info;
            constexpr ImageFlagBits not_kept =
                ImageFlagBits::GpuModified | ImageFlagBits::Alias | ImageFlagBits::BadOverlap |
                ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
                ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked;
            if ((keep_mode & 1) == 0 || existing_image.gpu_addr != gpu_addr ||
                existing_image.cpu_addr != *cpu_addr || True(existing_image.flags & not_kept) ||
                !existing_image.aliased_images.empty() || old.type != info.type ||
                old.format != info.format || old.size.width != info.size.width ||
                old.size.height != info.size.height || old.size.depth != info.size.depth ||
                old.resources.levels != info.resources.levels ||
                old.resources.layers != info.resources.layers ||
                old.num_samples != info.num_samples || old.block.width != info.block.width ||
                old.block.height != info.block.height || old.block.depth != info.block.depth ||
                old.layer_stride != info.layer_stride || old.pitch != info.pitch ||
                old.tile_width_spacing != info.tile_width_spacing) {
                return false;
            }
            existing_image.flags &= ~ImageFlagBits::Remapped;
            existing_image.flags |= ImageFlagBits::CpuModified;
            ++bindings_epoch;
            ++nx2_kept_after_remap;
            if (existing_image.unmapped_tick != 0) {
                ++unmap_stats.refound;
                unmap_stats.would_refound += existing_image.unmapped_verify == 1 ? 1 : 0;
                ForgetUnmapped(existing_image_id, existing_image);
            }
            image_id = existing_image_id;
            image_ids.clear();
            image_ids.push_back(existing_image_id);
            return true;
        }
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
            [[unlikely]] {
            const bool strict_size = False(options & RelaxedOptions::Size) &&
                                     True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format, broken_views, native_bgr)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                return !flexible_formats && existing.format == info.format;
            }
        } else if (IsSubresource(info, existing_image, gpu_addr, options, broken_views,
                                 native_bgr)) {
            image_id = existing_image_id;
            image_ids.push_back(existing_image_id);
            return !flexible_formats && existing_image.info.format == info.format;
        }
        return false;
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    if (image_ids.size() <= 1) [[likely]] {
        return image_id;
    }
    auto image_ids_compare = [this](ImageId a, ImageId b) {
        auto& image_a = slot_images[a];
        auto& image_b = slot_images[b];
        return image_a.modification_tick < image_b.modification_tick;
    };
    return *std::ranges::max_element(image_ids, image_ids_compare);
}

template <class P>
bool TextureCache<P>::ImageCanRescale(ImageBase& image) {
    if (!image.info.rescaleable) {
        return false;
    }
    if constexpr (requires { runtime.GetMaxScaledExtent(); }) {
        // FRITCHII_SCALED_EXTENT_CAP (default 1): past the device's limits the scaled image is
        // invalid. OoT's 12288x12288 sun shadow map is 18432 wide at 1.5x and 24576 at 2x, over
        // AMD's 16384, and drawing into it hangs the GPU (device loss entering Castle Town on
        // AMD). Before the cached IsRescalable answer, so a scale changed in game is checked too.
        if (VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_scaled_extent_cap,
                                             "FRITCHII_SCALED_EXTENT_CAP", 1) != 0 &&
            !ScaledImageFits(image.info, Settings::values.resolution_info,
                             runtime.GetMaxScaledExtent())) {
            if (False(image.flags & ImageFlagBits::ScaleLimited)) {
                image.flags |= ImageFlagBits::ScaleLimited;
                LOG_INFO(HW_GPU,
                         "Image {}x{} (format {}, {} samples) stays at native size: scaled it "
                         "would be {}x{}, past the device's limit of {} "
                         "(FRITCHII_SCALED_EXTENT_CAP)",
                         image.info.size.width, image.info.size.height,
                         static_cast<u32>(image.info.format), image.info.num_samples,
                         Settings::values.resolution_info.ScaleUp(image.info.size.width),
                         Settings::values.resolution_info.ScaleUp(image.info.size.height),
                         runtime.GetMaxScaledExtent());
            }
            return false;
        }
    }
    // FRITCHII_SCALE_DEPTH_MAX (default 8192, 0 off): a depth/stencil image whose scaled width or
    // height would pass N stays at native size. OoT's 12288x12288 D16 sun shadow map (288 MiB
    // native, made on every Castle Town load) is 648 MiB at 1.5x and 1.15 GiB at 2x on NVIDIA
    // (32768 limit, so the extent cap above keeps scaling it), and 450 MiB at 1.25x on AMD.
    // A shadow map that size gains nothing visible from scaling; its memory is what tips a card
    // over the budget. Same native path as the extent cap: the render target set of that pass
    // stays native, samplers, copies and blits handle a native image next to scaled ones as for
    // any image that is not rescaleable. Like the cap, checked before the cached answer.
    if (const int depth_max = VideoCore::Nx2Debug::SwitchValue(
            VideoCore::Nx2Debug::g_scale_depth_max, "FRITCHII_SCALE_DEPTH_MAX", 8192);
        depth_max > 0 && GetFormatType(image.info.format) != SurfaceType::ColorTexture &&
        !ScaledImageFits(image.info, Settings::values.resolution_info,
                         static_cast<u32>(depth_max))) {
        if (False(image.flags & ImageFlagBits::DepthNative)) {
            image.flags |= ImageFlagBits::DepthNative;
            ++depth_native_stats.window;
            if (const u64 n = ++depth_native_stats.total; n <= 16) {
                LOG_INFO(HW_GPU,
                         "Depth image {}x{} (format {}, {} samples) stays at native size: scaled "
                         "it would be {}x{}, past {} (FRITCHII_SCALE_DEPTH_MAX, {} so far)",
                         image.info.size.width, image.info.size.height,
                         static_cast<u32>(image.info.format), image.info.num_samples,
                         Settings::values.resolution_info.ScaleUp(image.info.size.width),
                         Settings::values.resolution_info.ScaleUp(image.info.size.height),
                         depth_max, n);
            }
        }
        return false;
    }
    if (Settings::values.resolution_info.downscale && !image.info.downscaleable) {
        return false;
    }
    if (True(image.flags & (ImageFlagBits::Rescaled | ImageFlagBits::CheckingRescalable))) {
        return true;
    }
    if (True(image.flags & ImageFlagBits::IsRescalable)) {
        return true;
    }
    image.flags |= ImageFlagBits::CheckingRescalable;
    for (const auto& alias : image.aliased_images) {
        Image& other_image = slot_images[alias.id];
        if (!ImageCanRescale(other_image)) {
            image.flags &= ~ImageFlagBits::CheckingRescalable;
            return false;
        }
    }
    image.flags &= ~ImageFlagBits::CheckingRescalable;
    image.flags |= ImageFlagBits::IsRescalable;
    return true;
}

template <class P>
void TextureCache<P>::InvalidateScale(Image& image) {
    ++bindings_epoch;
    VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::Invalidation);
    ++scale_epoch; // ScaleEpoch: draw templates (EXP87) keep rescaling words
    if (image.scale_tick <= frame_tick) {
        image.scale_tick = frame_tick + 1;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    for (const ImageViewId image_view_id : image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);
    for (const ImageViewId image_view_id : image_view_ids) {
        sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        slot_image_views.erase(image_view_id);
        BumpImageViewGeneration(image_view_id);
        Churn::Add(Churn::ImageViewsDestroyed);
    }
    image.image_view_ids.clear();
    image.image_view_infos.clear();
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];

#ifndef NDEBUG
        if constexpr (ENABLE_VALIDATION) {
            std::ranges::fill(channel_info.graphics_image_table.LiveBoundIds(), CORRUPT_ID);
            std::ranges::fill(channel_info.compute_image_table.LiveBoundIds(), CORRUPT_ID);
        }
#endif
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
u64 TextureCache<P>::GetScaledImageSizeBytes(const ImageBase& image) {
    const u64 scale_up = static_cast<u64>(Settings::values.resolution_info.up_scale *
                                          Settings::values.resolution_info.up_scale);
    const u64 down_shift = static_cast<u64>(Settings::values.resolution_info.down_shift +
                                            Settings::values.resolution_info.down_shift);
    const u64 image_size_bytes =
        static_cast<u64>((std::max)(image.guest_size_bytes, image.unswizzled_size_bytes));
    const u64 tentative_size = (image_size_bytes * scale_up) >> down_shift;
    const u64 fitted_size = Common::AlignUp(tentative_size, 1024);
    return fitted_size;
}

template <class P>
u64 TextureCache<P>::ImageMemoryBytes(const Image& image) const {
    if (True(image.flags & ImageFlagBits::Unbacked)) {
        // No device memory but what BindLayers gave a SparseLayers image: at its full size (GBs
        // for title 1's sparse terrain images), deleting it would look to the collector like
        // freeing that much.
        if constexpr (requires { image.SparseBoundBytes(); }) {
            return image.SparseBoundBytes();
        }
        return 0;
    }
    u64 tentative_size = (std::max)(image.guest_size_bytes, image.unswizzled_size_bytes);
    if ((IsPixelFormatASTC(image.info.format) &&
         True(image.flags & ImageFlagBits::AcceleratedUpload)) ||
        True(image.flags & ImageFlagBits::Converted)) {
        tentative_size = TranscodedAstcSize(tentative_size, image.info.format);
    }
    // Each top level the host leaves out (TrimmedLevels) held about 3/4 of what was left.
    tentative_size >>= 2 * image.host_level_offset;
    return Common::AlignUp(tentative_size, 1024);
}

template <class P>
void TextureCache<P>::QueueAsyncDecode(Image& image, ImageId image_id) {
    UNIMPLEMENTED_IF(False(image.flags & ImageFlagBits::Converted));
    LOG_DEBUG(HW_GPU, "Queuing async texture decode {:#x}", image.gpu_addr);

    image.flags |= ImageFlagBits::IsDecoding;
    ++async_decodes_queued;
    auto decode = std::make_unique<AsyncDecodeContext>();
    auto* decode_ptr = decode.get();
    decode->image_id = image_id;
    async_decodes.push_back(std::move(decode));

    std::vector<u8> local_unswizzle_data_buffer(image.unswizzled_size_bytes, 0);
    Tegra::Memory::GpuGuestMemory<u8, Tegra::Memory::GuestMemoryFlags::UnsafeRead> swizzle_data(*gpu_memory, image.gpu_addr, image.guest_size_bytes, &swizzle_data_buffer);
    auto copies = UnswizzleImage(*gpu_memory, image.gpu_addr, image.info, swizzle_data, local_unswizzle_data_buffer);
    const size_t out_size = MapSizeBytes(image);
    if (False(image.flags & ImageFlagBits::ZeroFilled)) {
        // Until the decode lands (a second or more while title 1 streams ~40 textures a second)
        // a new image showed whatever its memory last held, often other textures. Zero blocks
        // read as opaque black (BC1) or transparent (BC3) instead.
        image.flags |= ImageFlagBits::ZeroFilled;
        auto zero_copies = copies;
        ConvertedLayout(image.info, zero_copies);
        auto staging = runtime.UploadStagingBuffer(out_size);
        std::memset(staging.mapped_span.data(), 0, out_size);
        image.UploadMemory(staging, zero_copies);
        runtime.InsertUploadMemoryBarrier();
    }

    auto func = [out_size, copies, info = image.info,
                 input = std::move(local_unswizzle_data_buffer),
                 async_decode = decode_ptr]() mutable {
        async_decode->decoded_data.resize_destructive(out_size);
        std::span copies_span{copies.data(), copies.size()};
        ConvertImage(input, info, async_decode->decoded_data, copies_span);

        // TODO: Do we need this lock?
        std::unique_lock lock{async_decode->mutex};
        async_decode->copies = std::move(copies);
        async_decode->complete = true;
    };
    texture_decode_worker.QueueWork(std::move(func));
}

template <class P>
void TextureCache<P>::QueueAsyncUnswizzle(Image& image, ImageId image_id) {
    if (True(image.flags & ImageFlagBits::IsDecoding)) {
        return;
    }

    image.flags |= ImageFlagBits::IsDecoding;

    unswizzle_queue.push_back({
        .image_id = image_id,
        .info = image.info
    });
}

template <class P>
void TextureCache<P>::TickAsyncDecode() {
    bool has_uploads{};
    auto i = async_decodes.begin();
    while (i != async_decodes.end()) {
        auto* async_decode = i->get();
        {
            // Release the lock before the context is erased below: unlocking it afterwards wrote
            // into freed memory (heap corruption, crashes minutes into title 1).
            std::unique_lock lock{async_decode->mutex};
            if (!async_decode->complete) {
                ++i;
                continue;
            }
        }
        if (async_decode->cancelled) {
            i = async_decodes.erase(i);
            continue;
        }
        Image& image = slot_images[async_decode->image_id];
        if (async_decode->decoded_data.size() > MapSizeBytes(image)) {
            LOG_ERROR(HW_GPU, "Async decode of {} bytes for a {}-byte image at {:#x}, dropped",
                      async_decode->decoded_data.size(), MapSizeBytes(image), image.gpu_addr);
            i = async_decodes.erase(i);
            continue;
        }
        auto staging = runtime.UploadStagingBuffer(MapSizeBytes(image));
        std::memcpy(staging.mapped_span.data(), async_decode->decoded_data.data(),
                    async_decode->decoded_data.size());
        image.UploadMemory(staging, async_decode->copies);
        image.flags &= ~ImageFlagBits::IsDecoding;
        has_uploads = true;
        i = async_decodes.erase(i);
    }
    if (has_uploads) {
        runtime.InsertUploadMemoryBarrier();
    }
}

template <class P>
void TextureCache<P>::TickAsyncUnswizzle() {
    if (unswizzle_queue.empty()) {
        return;
    }

    if(current_unswizzle_frame > 0) {
        current_unswizzle_frame--;
        return;
    }

    PendingUnswizzle& task = unswizzle_queue.front();
    Image& image = slot_images[task.image_id];

    if (!task.initialized) {
        task.total_size = MapSizeBytes(image);
        task.staging_buffer = runtime.UploadStagingBuffer(task.total_size, true);

        const auto& info = image.info;
        const u32 bytes_per_block = BytesPerBlock(info.format);
        const u32 width_blocks = Common::DivCeil(info.size.width, 4u);
        const u32 height_blocks = Common::DivCeil(info.size.height, 4u);

        const u32 stride = width_blocks * bytes_per_block;
        const u32 aligned_height = height_blocks;
        task.bytes_per_slice = static_cast<size_t>(stride) * aligned_height;
        task.last_submitted_offset = 0;
        task.initialized = true;
    }

    // Read data
    if (task.current_offset < task.total_size) {
        const size_t remaining = task.total_size - task.current_offset;

        size_t copy_amount = (std::min)(swizzle_chunk_size, remaining);

        if (remaining > swizzle_chunk_size) {
            copy_amount = (copy_amount / task.bytes_per_slice) * task.bytes_per_slice;
            if (copy_amount == 0) copy_amount = task.bytes_per_slice;
        }

        gpu_memory->ReadBlock(image.gpu_addr + task.current_offset,
                              task.staging_buffer.mapped_span.data() + task.current_offset,
                              copy_amount);
        task.current_offset += copy_amount;
    }

    const bool is_final_batch = task.current_offset >= task.total_size;
    const size_t bytes_ready = task.current_offset - task.last_submitted_offset;
    const u32 complete_slices = static_cast<u32>(bytes_ready / task.bytes_per_slice);

    if (complete_slices >= swizzle_slices_per_batch || (is_final_batch && complete_slices > 0)) {
        const u32 z_start = static_cast<u32>(task.last_submitted_offset / task.bytes_per_slice);
        const u32 slices_to_process = (std::min)(complete_slices, swizzle_slices_per_batch);
        const u32 z_count = (std::min)(slices_to_process, image.info.size.depth - z_start);

        if (z_count > 0) {
            const auto uploads = FullUploadSwizzles(task.info);
            runtime.AccelerateImageUpload(image, task.staging_buffer, uploads, z_start, z_count);
            task.last_submitted_offset += (static_cast<size_t>(z_count) * task.bytes_per_slice);
        }
    }

    // Check if complete
    const u32 slices_submitted = static_cast<u32>(task.last_submitted_offset / task.bytes_per_slice);
    const bool all_slices_submitted = slices_submitted >= image.info.size.depth;

    if (is_final_batch && all_slices_submitted) {
        runtime.FreeDeferredStagingBuffer(task.staging_buffer);
        image.flags &= ~ImageFlagBits::IsDecoding;
        unswizzle_queue.pop_front();

        // Wait 4 frames to process the next entry
        current_unswizzle_frame = 4u;
    }
}

template <class P>
bool TextureCache<P>::ScaleUp(Image& image) {
    const bool has_copy = image.HasScaled();
    const bool rescaled = image.ScaleUp();
    if (!has_copy && image.HasScaled()) {
        total_used_memory += GetScaledImageSizeBytes(image);
    }
    if (!rescaled) {
        return false;
    }
    InvalidateScale(image);
    return true;
}

template <class P>
bool TextureCache<P>::ScaleDown(Image& image) {
    const bool rescaled = image.ScaleDown();
    if (!rescaled) {
        return false;
    }
    InvalidateScale(image);
    return true;
}

template <class P>
std::optional<DAddr> TextureCache<P>::ImageCpuAddress(GPUVAddr gpu_addr, size_t size) const {
    if (const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr)) {
        return cpu_addr;
    }
    if (const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr, size)) {
        return cpu_addr;
    }
    // No CPU backing (title 1 renders a bloom chain at GPU 0xf0000-0x15b600): a stand-in address
    // fixed by the GPU address, at or above ~(1 << 40) so TrackImage skips it. A fresh address per
    // lookup made every use create another image (thousands a minute, never found again).
    constexpr GPUVAddr StandInRange = 1ULL << 40;
    if (gpu_addr >= StandInRange || size > StandInRange - gpu_addr - 1) {
        return std::nullopt;
    }
    return ~(1ULL << 40ULL) + gpu_addr;
}

template <class P>
ImageId TextureCache<P>::InsertImage(const ImageInfo& info, GPUVAddr gpu_addr,
                                     RelaxedOptions options) {
    // FRITCHII_TC_CRITICAL_ENFORCE: room first when the cache is past critical and the budget is
    // short (the guest size stands in for the host image's).
    const auto guest_size = CalculateGuestSizeInBytes(info);
    EnforceCriticalAtCreate(guest_size);
    std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, guest_size);
    if (!cpu_addr) {
        const DAddr fake_addr = ~(1ULL << 40ULL) + virtual_invalid_space;
        virtual_invalid_space += Common::AlignUp(CalculateGuestSizeInBytes(info), 32);
        cpu_addr = std::optional<DAddr>(fake_addr);
    }
    ASSERT_MSG(cpu_addr, "Tried to insert an image to an invalid gpu_addr={:#x}", gpu_addr);
    ImageInfo storage_info = info;
    storage_info.native_tiling = gpu_memory->UsesNativeTiling();
    if (storage_info.native_tiling && info.type == ImageType::e3D) {
        // Switch 2 3D images (lighting/fog volumes) are rare: log each one's shape.
        LOG_DEBUG(HW_GPU, "3D image at {:#x}: {}x{}x{} levels {} format {} block {}x{}x{}",
                 gpu_addr, info.size.width, info.size.height, info.size.depth, info.resources.levels,
                 static_cast<u32>(info.format), info.block.width, info.block.height, info.block.depth);
    }
    const ImageId image_id = JoinImages(storage_info, gpu_addr, *cpu_addr);
    if (!image_id) {
        return image_id;
    }
    const Image& image = slot_images[image_id];
    // Using "image.gpu_addr" instead of "gpu_addr" is important because it might be different
    const auto [it, is_new] = image_allocs_table.try_emplace(image.gpu_addr);
    if (is_new) {
        it->second = slot_image_allocs.insert();
    }
    slot_image_allocs[it->second].images.push_back(image_id);
    return image_id;
}

template <class P>
u64 TextureCache<P>::ContentShape(const ImageInfo& info) noexcept {
    u64 shape = static_cast<u64>(info.format);
    const auto mix = [&shape](u64 value) { shape = (shape ^ value) * 0x100000001B3ULL; };
    mix(static_cast<u64>(info.type));
    mix(info.size.width);
    mix(info.size.height);
    mix(info.size.depth);
    mix(static_cast<u64>(info.resources.levels));
    mix(static_cast<u64>(info.resources.layers));
    mix(info.num_samples);
    mix(info.block.width);
    mix(info.block.height);
    mix(info.block.depth);
    mix(info.layer_stride);
    mix(info.pitch);
    mix(info.tile_width_spacing);
    return shape;
}

template <class P>
void TextureCache<P>::ForgetContent(ImageId image_id, const ImageBase& image, bool deleted) {
    if ((image.nx2_content_hash_lo | image.nx2_content_hash_hi) == 0) {
        return;
    }
    const ContentKey key{ContentShape(image.info), image.nx2_content_hash_lo,
                         image.nx2_content_hash_hi};
    if (const auto it = nx2_content_index.find(key);
        it != nx2_content_index.end() && it->second == image_id) {
        nx2_content_index.erase(it);
    }
    if (!deleted || True(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    // Measure only: as if the host image were kept, within 512 MiB, oldest out.
    constexpr u64 KeptBudget = u64{512} << 20;
    if (nx2_content_kept.emplace(key, image.guest_size_bytes).second) {
        nx2_content_kept_order.emplace_back(key, image.guest_size_bytes);
        nx2_content_kept_bytes += image.guest_size_bytes;
    }
    while (nx2_content_kept_bytes > KeptBudget && !nx2_content_kept_order.empty()) {
        const auto [old_key, bytes] = nx2_content_kept_order.front();
        nx2_content_kept_order.pop_front();
        if (nx2_content_kept.erase(old_key) != 0) {
            nx2_content_kept_bytes -= bytes;
        }
    }
}

template <class P>
void TextureCache<P>::NoteContentLookup(ImageId image_id) {
    const ImageBase& image = slot_images[image_id];
    constexpr u32 HashMinBytes = 64 * 1024;
    if (image.guest_size_bytes < HashMinBytes ||
        True(image.flags & (ImageFlagBits::GpuModified | ImageFlagBits::Sparse |
                            ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
                            ImageFlagBits::Unbacked))) {
        return;
    }
    const u128 hash = GuestContentHash(image);
    // As RefreshContents stores it (bit 0 of the high word set).
    const ContentKey key{ContentShape(image.info), hash[0], hash[1] | 1};
    size_t which = 3;
    if (const auto it = nx2_content_index.find(key);
        it != nx2_content_index.end() && it->second != image_id &&
        slot_images.contains(it->second)) {
        const ImageBase& twin = slot_images[it->second];
        const bool same = twin.nx2_content_hash_lo == key.lo && twin.nx2_content_hash_hi == key.hi &&
                          False(twin.flags & ImageFlagBits::GpuModified);
        if (same) {
            const bool idle = twin.lru_index == SIZE_MAX ||
                              lru_cache.Tick(twin.lru_index) + 1 < frame_tick;
            which = idle ? 0 : 1;
        }
    }
    if (which == 3 && nx2_content_kept.contains(key)) {
        which = 2;
    }
    ++nx2_content_cases[which];
    nx2_content_case_bytes[which] += image.guest_size_bytes;
}

template <class P>
u128 TextureCache<P>::GuestContentHash(const ImageBase& image) {
    const size_t size = image.guest_size_bytes;
    const auto start = std::chrono::steady_clock::now();
    u128 hash;
    if (const u8* const span = gpu_memory->GetSpan(image.gpu_addr, size); span != nullptr) {
        hash = Common::CityHash128(reinterpret_cast<const char*>(span), size);
    } else {
        nx2_hash_scratch.resize(size);
        gpu_memory->ReadBlockUnsafe(image.gpu_addr, nx2_hash_scratch.data(), size);
        hash = Common::CityHash128(reinterpret_cast<const char*>(nx2_hash_scratch.data()), size);
    }
    nx2_content_hash_ns += static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                             start)
            .count());
    nx2_content_hashed_bytes += size;
    return hash;
}

template <class P>
void TextureCache<P>::NoteNewSampledImage(ImageId image_id, GPUVAddr gpu_addr) {
    if (!ImageHashLogEnabled()) [[likely]] {
        return;
    }
    // What deleted the previous image at this address (bit 7: the GPU had written it).
    if (const auto it = nx2_deleted_reasons.find(gpu_addr); it != nx2_deleted_reasons.end()) {
        const size_t reason = it->second & 0x7f;
        const size_t gpu_written = (it->second & 0x80) != 0 ? 1 : 0;
        ++nx2_remade_after[reason * 2 + gpu_written];
        nx2_deleted_reasons.erase(it);
    } else {
        ++nx2_remade_unknown;
    }
    struct Seen {
        u32 count = 0;
        u32 width = 0;
        u32 height = 0;
        u32 format = 0;
    };
    static std::unordered_map<u64, Seen> seen;
    static u64 made = 0;
    static u64 made_again = 0;
    static u64 bytes_again = 0;
    static auto last_log = std::chrono::steady_clock::now();
    const ImageBase& image = slot_images[image_id];
    // Up to the first 256 KiB of the guest bytes, with the shape: enough to tell the same
    // texture made again from a new one.
    const size_t size =
        (std::min)(static_cast<size_t>(image.guest_size_bytes), size_t{256} << 10);
    std::vector<u8> bytes(size);
    gpu_memory->ReadBlockUnsafe(gpu_addr, bytes.data(), size);
    u64 hash = Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), size);
    hash ^= (u64{image.info.size.width} << 32 | image.info.size.height) * 0x9E3779B97F4A7C15ULL;
    hash ^= static_cast<u64>(image.info.format) << 56;
    Seen& entry = seen[hash];
    ++made;
    if (entry.count++ > 0) {
        ++made_again;
        bytes_again += image.guest_size_bytes;
    }
    entry.width = image.info.size.width;
    entry.height = image.info.size.height;
    entry.format = static_cast<u32>(image.info.format);
    const auto now = std::chrono::steady_clock::now();
    if (now - last_log < std::chrono::seconds{10}) {
        return;
    }
    last_log = now;
    std::vector<std::pair<u64, Seen>> top(seen.begin(), seen.end());
    const size_t keep = (std::min)(top.size(), size_t{8});
    std::partial_sort(top.begin(), top.begin() + keep, top.end(),
                      [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    std::string text;
    for (size_t i = 0; i < keep && top[i].second.count > 1; ++i) {
        text += fmt::format(" {}x{} fmt {} x{};", top[i].second.width, top[i].second.height,
                            top[i].second.format, top[i].second.count);
    }
    LOG_INFO(HW_GPU,
             "New sampled images (EDEN_NX2_IMAGE_HASH_LOG) last 10 s: {} made, {} with bytes made "
             "before ({} KiB); {} different so far; most made:{}",
             made, made_again, bytes_again >> 10, seen.size(), text.empty() ? " none twice" : text);
    // Reason order: other, garbage collect, cold sweep, unmap, join, slot retire, unmap sweep,
    // superseded sweep.
    static constexpr std::array<const char*, static_cast<size_t>(DeleteReason::Count)> names{
        "other",       "garbage collect", "cold sweep", "unmap", "join", "slot retire",
        "unmap sweep", "superseded sweep"};
    std::string deletes;
    std::string remade;
    for (size_t r = 0; r < names.size(); ++r) {
        deletes += fmt::format(" {} {};", names[r], nx2_deletes[r]);
        remade += fmt::format(" {} {}/{};", names[r], nx2_remade_after[r * 2],
                              nx2_remade_after[r * 2 + 1]);
    }
    LOG_INFO(HW_GPU,
             "Images deleted last 10 s by:{} | new sampled images whose previous image at that "
             "address was deleted by (not GPU-written/GPU-written):{} no earlier image there {}",
             deletes, remade, nx2_remade_unknown);
    LOG_INFO(HW_GPU,
             "Remaps (EXP76={}) last 10 s: {} images flagged remapped, {} kept in place; whole "
             "uploads: {} hash checks, {} skipped as unchanged ({} KiB)",
             VideoCore::Nx2Debug::Experiment(76), nx2_flagged_remapped, nx2_kept_after_remap,
             nx2_upload_hash_checks, nx2_upload_skips, nx2_upload_skip_bytes >> 10);
    const double hash_ms = static_cast<double>(nx2_content_hash_ns) / 1e6;
    LOG_INFO(HW_GPU,
             "Content reuse (EXP76 & 4, measure only) last 10 s: new sampled images that a live "
             "idle twin would serve {} ({} KiB), a live busy twin {} ({} KiB), a kept image {} "
             "({} KiB), nothing {} ({} KiB); hashed {} MiB in {:.1f} ms ({:.1f} GB/s); index {} "
             "live, {} kept ({} MiB)",
             nx2_content_cases[0], nx2_content_case_bytes[0] >> 10, nx2_content_cases[1],
             nx2_content_case_bytes[1] >> 10, nx2_content_cases[2],
             nx2_content_case_bytes[2] >> 10, nx2_content_cases[3],
             nx2_content_case_bytes[3] >> 10, nx2_content_hashed_bytes >> 20, hash_ms,
             hash_ms > 0 ? static_cast<double>(nx2_content_hashed_bytes) / 1e6 / hash_ms : 0.0,
             nx2_content_index.size(), nx2_content_kept.size(), nx2_content_kept_bytes >> 20);
    nx2_content_cases = {};
    nx2_content_case_bytes = {};
    nx2_content_hash_ns = 0;
    nx2_content_hashed_bytes = 0;
    nx2_flagged_remapped = 0;
    nx2_kept_after_remap = 0;
    nx2_upload_hash_checks = 0;
    nx2_upload_skips = 0;
    nx2_upload_skip_bytes = 0;
    nx2_deletes = {};
    nx2_remade_after = {};
    nx2_remade_unknown = 0;
    made = 0;
    made_again = 0;
    bytes_again = 0;
}

template <class P>
ImageId TextureCache<P>::JoinImages(const ImageInfo& info, GPUVAddr gpu_addr, DAddr cpu_addr) {
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::Join};
    ImageInfo new_info = info;
    const size_t size_bytes = CalculateGuestSizeInBytes(new_info);
    const bool broken_views = runtime.HasBrokenTextureViewFormats();
    const bool native_bgr = runtime.HasNativeBgr();
    join_overlap_ids.clear();
    join_overlaps_found.clear();
    join_left_aliased_ids.clear();
    join_right_aliased_ids.clear();
    join_ignore_textures.clear();
    join_bad_overlap_ids.clear();
    join_copies_to_do.clear();
    join_alias_indices.clear();
    ++rt_stats.joins;
    const auto walk_start = std::chrono::steady_clock::now();
    const bool this_is_linear = info.type == ImageType::Linear;
    const auto region_check = [&](ImageId overlap_id, ImageBase& overlap) {
        ++rt_stats.overlaps_walked;
        if (True(overlap.flags & ImageFlagBits::Remapped)) {
            join_ignore_textures.insert(overlap_id);
            return;
        }
        const bool overlap_is_linear = overlap.info.type == ImageType::Linear;
        if (this_is_linear != overlap_is_linear) {
            return;
        }
        if (this_is_linear && overlap_is_linear) {
            if (info.pitch == overlap.info.pitch && gpu_addr == overlap.gpu_addr) {
                // Alias linear images with the same pitch
                join_left_aliased_ids.push_back(overlap_id);
            }
            return;
        }
        join_overlaps_found.insert(overlap_id);
        // A drift: a single-level, single-layer 2D target placed a little off a previous one
        // (Ocarina of Time's bump-allocated per-frame targets move 40 KB - 2 MB a frame). The
        // offset is under one layer, and not 0, so it is no mip or layer base of either image:
        // ResolveOverlap and both IsSubresource checks below would fail. Skips them.
        if (new_info.type == ImageType::e2D && overlap.info.type == ImageType::e2D &&
            new_info.resources.levels == 1 && overlap.info.resources.levels == 1 &&
            new_info.resources.layers == 1 && overlap.info.resources.layers == 1 &&
            new_info.layer_stride == overlap.info.layer_stride && gpu_addr != overlap.gpu_addr) {
            const u64 distance = gpu_addr > overlap.gpu_addr ? gpu_addr - overlap.gpu_addr
                                                             : overlap.gpu_addr - gpu_addr;
            const u64 limit = new_info.layer_stride != 0 ? new_info.layer_stride : size_bytes;
            if (distance < limit && distance < size_bytes &&
                distance < overlap.guest_size_bytes) {
                join_bad_overlap_ids.push_back(overlap_id);
                return;
            }
        }
        constexpr bool strict_size = true;
        const std::optional<OverlapResult> solution = ResolveOverlap(
            new_info, gpu_addr, cpu_addr, overlap, strict_size, broken_views, native_bgr);
        if (solution) {
            gpu_addr = solution->gpu_addr;
            cpu_addr = solution->cpu_addr;
            new_info.resources = solution->resources;
            join_overlap_ids.push_back(overlap_id);
            join_copies_to_do.emplace_back(JoinCopy{false, overlap_id});
            return;
        }
        constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
        const ImageBase new_image_base(new_info, gpu_addr, cpu_addr);
        if (IsSubresource(new_info, overlap, gpu_addr, options, broken_views, native_bgr)) {
            join_left_aliased_ids.push_back(overlap_id);
            overlap.flags |= ImageFlagBits::Alias;
            join_copies_to_do.emplace_back(JoinCopy{true, overlap_id});
        } else if (IsSubresource(overlap.info, new_image_base, overlap.gpu_addr, options,
                                 broken_views, native_bgr)) {
            join_right_aliased_ids.push_back(overlap_id);
            overlap.flags |= ImageFlagBits::Alias;
            join_copies_to_do.emplace_back(JoinCopy{true, overlap_id});
        } else {
            join_bad_overlap_ids.push_back(overlap_id);
        }
    };
    ForEachImageInRegion(cpu_addr, size_bytes, region_check);
    const auto region_check_gpu = [&](ImageId overlap_id, ImageBase& overlap) {
        if (!join_overlaps_found.contains(overlap_id)) {
            if (True(overlap.flags & ImageFlagBits::Remapped)) {
                join_ignore_textures.insert(overlap_id);
            }
            if (overlap.gpu_addr == gpu_addr && overlap.guest_size_bytes == size_bytes) {
                join_ignore_textures.insert(overlap_id);
            }
        }
    };
    ForEachSparseImageInRegion(channel_state->gpu_memory.GetID(), gpu_addr, size_bytes,
                               region_check_gpu);
    rt_stats.walk_ns += NsSince(walk_start);
    rt_stats.bad_overlaps += static_cast<u32>(join_bad_overlap_ids.size());

    bool can_rescale = info.rescaleable;
    bool any_rescaled = false;
    for (const auto& copy : join_copies_to_do) {
        if (!can_rescale) {
            break;
        }
        Image& sibling = slot_images[copy.id];
        can_rescale &= ImageCanRescale(sibling);
        any_rescaled |= True(sibling.flags & ImageFlagBits::Rescaled);
    }

    can_rescale &= any_rescaled;

    if (can_rescale) {
        for (const auto& copy : join_copies_to_do) {
            Image& sibling = slot_images[copy.id];
            ScaleUp(sibling);
        }
    } else {
        for (const auto& copy : join_copies_to_do) {
            Image& sibling = slot_images[copy.id];
            ScaleDown(sibling);
        }
    }

    if constexpr (requires { runtime.UseOpaqueAstcFormat(); }) {
        // Opaque ASTC goes to BC1 rather than BC3 when it is re-compressed (the Vulkan runtime
        // decides). The guest data is already there when an image is first used.
        constexpr size_t MaxScanBytes = 64_MiB;
        if (IsPixelFormatASTC(new_info.format) && runtime.UseOpaqueAstcFormat()) {
            const size_t scan_bytes = CalculateGuestSizeInBytes(new_info);
            if (scan_bytes <= MaxScanBytes) {
                astc_scan_buffer.resize(scan_bytes);
                gpu_memory->ReadBlockUnsafe(gpu_addr, astc_scan_buffer.data(), scan_bytes);
                new_info.astc_opaque = AstcDataLooksOpaque(astc_scan_buffer);
                ++(new_info.astc_opaque ? astc_opaque_images : astc_alpha_images);
            }
        }
    }
    // A new render target that the clear binding it overwrites whole (every channel, the whole
    // image) needs no upload from guest memory first: Ocarina of Time creates ~4 such targets a
    // frame outdoors, and uploading them was ~15% of the GPU back thread.
    // Which test refused the last call (EXP7's new-image line).
    const char* clear_refusal = "";
    const auto overwritten_by_clear = [&] {
        if (!clearing_render_targets) {
            return (clear_refusal = "not clearing", false);
        }
        if (!join_copies_to_do.empty()) {
            return (clear_refusal = "join copies pending", false);
        }
        if (new_info.type != ImageType::e2D || new_info.resources.levels != 1 ||
            new_info.resources.layers != 1 || new_info.num_samples != 1) {
            return (clear_refusal = "not a plain 2D target", false);
        }
        const auto& regs = maxwell3d->regs;
        const auto& surface = regs.clear_surface;
        if (surface.layer != 0) {
            return (clear_refusal = "clear layer", false);
        }
        const u32 width = new_info.size.width;
        const u32 height = new_info.size.height;
        if (regs.clear_control.use_scissor != 0) {
            const auto& scissor = regs.scissor_test[0];
            if (scissor.min_x != 0 || scissor.min_y != 0 || scissor.max_x < width ||
                scissor.max_y < height) {
                // EXP7: the numbers, for the per-object depth targets whose width is padded to
                // 16 (fable52_answer_4: is the clear only a few pixels of padding short?).
                static std::atomic<u32> scissor_logged{};
                if (VideoCore::Nx2Debug::Experiment(7) != 0 &&
                    scissor_logged.fetch_add(1, std::memory_order_relaxed) < 40) {
                    LOG_WARNING(HW_GPU,
                                "Clear rule: scissor ({},{})-({},{}) vs new image format {} "
                                "{}x{} at {:#x}",
                                scissor.min_x.Value(), scissor.min_y.Value(),
                                scissor.max_x.Value(), scissor.max_y.Value(),
                                static_cast<u32>(new_info.format), width, height, gpu_addr);
                }
                return (clear_refusal = "scissor", false);
            }
        }
        // The clear covers the framebuffer's render area, the smallest bound target.
        // EDEN_NX2_EXP56 (live, 0 = off): not for the clear's own depth target, which follows
        // IsFullClear as an existing one does (its contents are dropped, the colour targets'
        // sizes aside). Ocarina binds a 1920x1024 depth buffer under 1280x720 colour and this
        // rule uploaded its 7.5 MiB from guest memory every frame (fable52_answer_3 s7).
        const bool depth_follows_full_clear = VideoCore::Nx2Debug::Experiment(56) != 0 &&
                                              regs.zeta_enable && regs.zeta.Address() == gpu_addr;
        for (u32 index = 0; index < regs.rt_control.count && index < NUM_RT; ++index) {
            const auto& rt = regs.rt[index];
            if (!depth_follows_full_clear && rt.Address() != 0 &&
                rt.format != Tegra::RenderTargetFormat::NONE &&
                (rt.width < width || rt.height < height)) {
                return (clear_refusal = "a smaller colour target", false);
            }
        }
        if (regs.zeta_enable && regs.zeta.Address() != 0 &&
            (regs.zeta_size.width < width || regs.zeta_size.height < height)) {
            return (clear_refusal = "a smaller depth target", false);
        }
        const bool is_depth_target = regs.zeta_enable && regs.zeta.Address() == gpu_addr;
        const bool full_stencil = surface.S && regs.stencil_front_mask == 0xFF;
        switch (VideoCore::Surface::GetFormatType(new_info.format)) {
        case VideoCore::Surface::SurfaceType::ColorTexture: {
            const u32 index = surface.RT;
            const bool covered = surface.R && surface.G && surface.B && surface.A &&
                                 index < NUM_RT && index < regs.rt_control.count &&
                                 regs.rt[index].Address() == gpu_addr;
            clear_refusal = covered ? "" : "colour channels or target";
            return covered;
        }
        case VideoCore::Surface::SurfaceType::Depth:
            clear_refusal = (is_depth_target && surface.Z) ? "" : "depth: not the bound target or no Z";
            return is_depth_target && surface.Z;
        case VideoCore::Surface::SurfaceType::Stencil:
            clear_refusal = (is_depth_target && full_stencil) ? "" : "stencil: not the bound target or mask";
            return is_depth_target && full_stencil;
        case VideoCore::Surface::SurfaceType::DepthStencil:
            clear_refusal = (is_depth_target && surface.Z && full_stencil) ? "" : "depth-stencil: not bound, no Z or stencil mask";
            return is_depth_target && surface.Z && full_stencil;
        default:
            return false;
        }
    };
    // EXP43 (fable_answer_12 2A): a cleared render target that drifted off exactly one image of
    // its shape takes that image's place (2: the image moves here) or retires it at once (1).
    if (clearing_render_targets && !join_bad_overlap_ids.empty() && join_copies_to_do.empty() &&
        join_left_aliased_ids.empty() && join_right_aliased_ids.empty() &&
        join_ignore_textures.empty()) {
        if (const ImageId old_id = RelocationCandidate(new_info, gpu_addr, cpu_addr);
            old_id && overwritten_by_clear()) {
            const GPUVAddr old_gpu_addr = slot_images[old_id].gpu_addr;
            const s64 drift = static_cast<s64>(gpu_addr) - static_cast<s64>(old_gpu_addr);
            RelocationShape& shape = relocation_shapes[RelocationShapeKey(new_info)];
            const int mode = VideoCore::Nx2Debug::Experiment(43);
            relocation_seen = true;
            ++relocation_candidates;
            if (shape.guard != 0 || relocation_guarded_addresses.contains(gpu_addr) ||
                relocation_guarded_addresses.contains(old_gpu_addr)) {
                ++relocation_guarded;
            } else {
                if (!shape.candidate_logged) {
                    shape.candidate_logged = true;
                    LOG_INFO(HW_GPU,
                             "Render target relocation candidate (EXP43={}): format {} {}x{} at "
                             "{:#x}, drifted {:+} bytes from the image at {:#x}",
                             mode, static_cast<u32>(new_info.format), new_info.size.width,
                             new_info.size.height, gpu_addr, drift, old_gpu_addr);
                }
                const bool relocate = mode == 2 || mode == 3 || mode == 4;
                if ((relocate || mode == 1) && !shape.relocated_logged) {
                    shape.relocated_logged = true;
                    LOG_INFO(HW_GPU,
                             "Render target {} (EXP43={}): format {} {}x{}, {:#x} -> {:#x}",
                             relocate ? "relocated" : "retired early", mode,
                             static_cast<u32>(new_info.format), new_info.size.width,
                             new_info.size.height, old_gpu_addr, gpu_addr);
                }
                if (relocate) {
                    ++relocation_relocated;
                    ++rt_stats.relocated_images;
                    Churn::Add(Churn::ImagesRelocated);
                    if (mode == 4) {
                        AddRelocationTombstone(slot_images[old_id], gpu_addr);
                    }
                    RelocateImage(old_id, gpu_addr, cpu_addr);
                    // The other bad overlaps are linked to it as to a new image (below).
                    Image& moved = slot_images[old_id];
                    for (const ImageId other_id : join_bad_overlap_ids) {
                        if (other_id == old_id) {
                            continue;
                        }
                        ImageBase& other = slot_images[other_id];
                        other.overlapping_images.push_back(old_id);
                        moved.overlapping_images.push_back(other_id);
                        other.covered_overlap_frame = frame_tick + 1; // FRITCHII_COVERED_RETIRE
                        if (other.info.resources.levels == 1 && other.info.block.depth == 0 &&
                            other.overlapping_images.size() > 1) {
                            other.flags |= ImageFlagBits::BadOverlap;
                        }
                        if (moved.info.resources.levels == 1 && moved.info.block.depth == 0 &&
                            moved.overlapping_images.size() > 1) {
                            moved.flags |= ImageFlagBits::BadOverlap;
                        }
                        WatchColdImage(other_id);
                    }
                    WatchColdImage(old_id);
                    return old_id;
                }
                if (mode == 1) {
                    // 2C: the predecessor goes now instead of 50 frames later; a new image is
                    // made below as without the switch.
                    ++relocation_retired;
                    ++rt_stats.retired_images;
                    Image& old = slot_images[old_id];
                    MarkStale(old); // before UntrackImage (see MarkStale)
                    if (True(old.flags & ImageFlagBits::Tracked)) {
                        UntrackImage(old, old_id);
                    }
                    UnregisterImage(old_id);
                    DeleteImage(old_id);
                    join_bad_overlap_ids.erase(std::remove(join_bad_overlap_ids.begin(),
                                                           join_bad_overlap_ids.end(), old_id),
                                               join_bad_overlap_ids.end());
                }
            }
        }
    }
    if (!clearing_render_targets && slot_retired_any) {
        const u64 shape = RelocationShapeKey(new_info);
        const u64 frame = RenderFrame();
        for (const SlotRetired& retired : slot_recent) {
            if (retired.gpu_addr == gpu_addr && retired.shape == shape &&
                frame - retired.render_frame <= 16) {
                ++slot_remade;
                if (slot_remade_logged < 20) {
                    ++slot_remade_logged;
                    LOG_WARNING(HW_GPU,
                                "Slot retire (EXP64): format {} {}x{} at {:#x} made again {} "
                                "frames after it was retired (its contents were wanted)",
                                static_cast<u32>(new_info.format), new_info.size.width,
                                new_info.size.height, gpu_addr, frame - retired.render_frame);
                }
                break;
            }
        }
    }
    // EXP67 (live, bucket in pixels, 0 = off): a depth render target at an address whose target
    // changes dims (OoT's per-NPC depth targets, sized by screen bounds every frame) gets a host
    // image rounded up to the bucket, so the image pool hands the next frame's target the same
    // VkImage instead of a new create and a 0.2-1.3 ms depth bind (IMAGE_ALLOC_TIMING).
    const auto pad_type = VideoCore::Surface::GetFormatType(new_info.format);
    const bool pad_depth = pad_type == VideoCore::Surface::SurfaceType::Depth ||
                           pad_type == VideoCore::Surface::SurfaceType::Stencil ||
                           pad_type == VideoCore::Surface::SurfaceType::DepthStencil;
    const int pad_bucket = VideoCore::Nx2Debug::Experiment(67);
    // Not is_rescaling: it is true whenever a depth target is bound, at 1x too (RescaleRenderTargets
    // counts depth as rescaled), which refused every target in the first test.
    const bool pad_native_scale = Settings::values.resolution_info.up_scale == 1 &&
                                  Settings::values.resolution_info.down_shift == 0;
    if (pad_bucket > 0 && pad_depth) {
        ++pad_depth_lookups;
        if (!render_target_lookup || !pad_native_scale || !RelocationShaped(new_info) ||
            new_info.num_samples != 1 || pad_unsafe_slots.contains(gpu_addr)) {
            ++pad_skipped_other;
        }
    }
    if (const int bucket = pad_bucket;
        bucket > 0 && pad_depth && render_target_lookup && pad_native_scale &&
        RelocationShaped(new_info) && new_info.num_samples == 1 &&
        !pad_unsafe_slots.contains(gpu_addr)) {
        {
            bool varying = pad_slots.contains(gpu_addr);
            if (!varying) {
                for (const ImageId overlap_id : join_bad_overlap_ids) {
                    const ImageBase& overlap = slot_images[overlap_id];
                    if (overlap.gpu_addr == gpu_addr && overlap.info.format == new_info.format &&
                        (overlap.info.size.width != new_info.size.width ||
                         overlap.info.size.height != new_info.size.height)) {
                        varying = true;
                        pad_slots.insert(gpu_addr);
                        break;
                    }
                }
            }
            if (varying) {
                const u32 b = static_cast<u32>(bucket);
                const u32 width = (new_info.size.width + b - 1) / b * b;
                const u32 height = (new_info.size.height + b - 1) / b * b;
                if constexpr (requires(Runtime& r) { r.SetNextHostPad(0U, 0U); }) {
                    runtime.SetNextHostPad(width, height);
                    ++pad_made;
                }
            } else {
                ++pad_not_varying;
            }
        }
    }
    // EXP64 (fable_answer_npcdips 3): the cleared target's same-address predecessors go now.
    if (clearing_render_targets && !join_bad_overlap_ids.empty() && join_copies_to_do.empty() &&
        join_left_aliased_ids.empty() && join_right_aliased_ids.empty() &&
        join_ignore_textures.empty() && VideoCore::Nx2Debug::Experiment(64) != 0) {
        const auto& regs = maxwell3d->regs;
        const auto& surface = regs.clear_surface;
        const bool depth_cleared = regs.zeta_enable && regs.zeta.Address() == gpu_addr &&
                                   (surface.Z || surface.S);
        const bool colour_cleared = (surface.R || surface.G || surface.B || surface.A) &&
                                    surface.RT < NUM_RT && regs.rt[surface.RT].Address() == gpu_addr;
        if (depth_cleared || colour_cleared) {
            RetireSlotPredecessors(new_info, gpu_addr);
        }
    }
    const auto create_start = std::chrono::steady_clock::now();
    const ImageId new_image_id = slot_images.insert(runtime, new_info, gpu_addr, cpu_addr);
    rt_stats.create_ns += NsSince(create_start);
    if (depth_census_on && !render_target_lookup) [[unlikely]] {
        // EDEN_NX2_DEPTH_CENSUS: a texture lookup's image takes census targets in.
        const auto mark = [this](ImageId id) {
            if (const auto it = census_live.find(id); it != census_live.end()) {
                it->second.via_texture = true;
            }
        };
        for (const auto& copy : join_copies_to_do) {
            mark(copy.id);
        }
        std::ranges::for_each(join_left_aliased_ids, mark);
        std::ranges::for_each(join_right_aliased_ids, mark);
    }
    NoteMadeShape(new_info);
    NoteThrashMade(new_info, gpu_addr);
    if constexpr (requires(const Image& image) { image.FromPool(); }) {
        rt_stats.pooled_images += slot_images[new_image_id].FromPool() ? 1 : 0;
        Churn::Add(Churn::ImagesPooled, slot_images[new_image_id].FromPool() ? 1 : 0);
    }
    if constexpr (requires(const Image& image) { image.OutOfMemory(); }) {
        if (slot_images[new_image_id].OutOfMemory()) {
            // Left out of the cache: lookups see no image (a null view) and make it again once
            // the collector has freed memory.
            slot_images.erase(new_image_id);
            static u32 logged{};
            if (logged++ < 32) {
                LOG_ERROR(HW_GPU, "No memory for the {}x{}x{} image at {:#x} (format {}); skipping it",
                          new_info.size.width, new_info.size.height, new_info.size.depth, gpu_addr,
                          static_cast<u32>(new_info.format));
            }
            return ImageId{};
        }
    }
    Image& new_image = slot_images[new_image_id];
    if ((VideoCore::Nx2Debug::Experiment(68) & 4) != 0 && RelocationShaped(new_info)) {
        // EXP68 bit 4 (fable52_answer_npcdips 2, item 3): same address and format, other dims
        // among the overlaps: a size-varying slot. Its images skip the pool.
        const auto other_dims = [&](ImageId id) {
            const ImageBase& other = slot_images[id];
            return other.gpu_addr == gpu_addr && other.info.format == new_info.format &&
                   (other.info.size.width != new_info.size.width ||
                    other.info.size.height != new_info.size.height);
        };
        bool varying = false;
        for (const auto* ids : {&join_bad_overlap_ids}) {
            for (const ImageId id : *ids) {
                if (other_dims(id)) {
                    varying = true;
                    slot_images[id].flags |= ImageFlagBits::NoPool;
                }
            }
        }
        for (const ImageId id : join_left_aliased_ids) {
            if (other_dims(id)) {
                varying = true;
                slot_images[id].flags |= ImageFlagBits::NoPool;
            }
        }
        for (const ImageId id : join_right_aliased_ids) {
            if (other_dims(id)) {
                varying = true;
                slot_images[id].flags |= ImageFlagBits::NoPool;
            }
        }
        if (varying) {
            new_image.flags |= ImageFlagBits::NoPool;
            ++no_pool_marked;
        }
    }
    ++bindings_epoch;

    new_image.allocation_tick = frame_tick;
    {
        // Which shapes are being created (periodic log): format, type, levels, layers, size.
        const u64 key = (u64{static_cast<u32>(new_info.format)} << 56) |
                        (u64{static_cast<u32>(new_info.type)} << 52) |
                        (u64{static_cast<u32>(new_info.resources.levels) & 0xF} << 48) |
                        (u64{static_cast<u32>(new_info.resources.layers) & 0xFFF} << 36) |
                        (u64{new_info.size.width & 0x3FFFF} << 18) | (new_info.size.height & 0x3FFFF);
        if (debug_stats) {
            auto& shape = created_shapes[key];
            ++shape.first;
            shape.second = gpu_addr;
        }
        ++images_created;
        Churn::Add(Churn::ImagesCreated);
    }

    if (!gpu_memory->IsContinuousRange(new_image.gpu_addr, new_image.guest_size_bytes) &&
        new_info.is_sparse) {
        new_image.flags |= ImageFlagBits::Sparse;
    }

    const auto ignored_delete_start = std::chrono::steady_clock::now();
    for (const ImageId overlap_id : join_ignore_textures) {
        Image& overlap = slot_images[overlap_id];
        // Remapped images no longer describe the current guest memory. Discard their GPU
        // contents rather than downloading them through the replacement mapping.
        if (True(overlap.flags & ImageFlagBits::GpuModified) &&
            False(overlap.flags & ImageFlagBits::Remapped)) {
            UNIMPLEMENTED();
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap, overlap_id);
        }
        UnregisterImage(overlap_id);
        DeleteImage(overlap_id);
    }
    rt_stats.delete_ns += NsSince(ignored_delete_start);

    // Whether one of the GPU copies below writes all of a single-level 2D image, so an upload
    // before it is never seen. Ocarina of Time creates a D32 image at each new size of a
    // dynamic-resolution target over the same memory, uploading 6-14 MiB of it each time.
    const auto overwritten_by_copies = [&] {
        const ImageInfo& image_info = new_image.info;
        constexpr ImageFlagBits unusual = ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
                                          ImageFlagBits::SparseTiles |
                                          ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked;
        if (image_info.type != ImageType::e2D || image_info.resources.levels != 1 ||
            image_info.resources.layers != 1 || image_info.num_samples != 1 ||
            True(new_image.flags & (unusual | ImageFlagBits::Converted |
                                    ImageFlagBits::AsynchronousDecode))) {
            return false;
        }
        const auto blocks = [](const ImageInfo& of, s32 level) {
            const Extent3D size = MipSize(of.size, static_cast<u32>(level));
            return Extent2D{
                .width = Common::DivCeil(size.width, DefaultBlockWidth(of.format)),
                .height = Common::DivCeil(size.height, DefaultBlockHeight(of.format)),
            };
        };
        const Extent2D new_blocks = blocks(image_info, 0);
        const auto covers = [&](const ImageInfo& overlap_info, s32 level) {
            const Extent2D overlap_blocks = blocks(overlap_info, level);
            return overlap_blocks.width >= new_blocks.width &&
                   overlap_blocks.height >= new_blocks.height;
        };
        for (const JoinCopy& copy : join_copies_to_do) {
            const ImageBase& overlap = slot_images[copy.id];
            if (overlap.info.type != ImageType::e2D || overlap.info.num_samples != 1 ||
                True(overlap.flags & unusual) ||
                GetFormatType(overlap.info.format) != GetFormatType(image_info.format) ||
                BytesPerBlock(overlap.info.format) != BytesPerBlock(image_info.format) ||
                overlap.info.astc_opaque != image_info.astc_opaque) {
                continue;
            }
            if (!copy.is_alias) {
                // Copied whole (MakeShrinkImageCopies) when the GPU wrote it.
                const std::optional<SubresourceBase> base = new_image.TryFindBase(overlap.gpu_addr);
                if (True(overlap.flags & ImageFlagBits::GpuModified) && base &&
                    base->level == 0 && base->layer == 0) {
                    return true;
                }
                continue;
            }
            if (!overlap.IsSafeGpuCopy()) {
                continue;
            }
            // The extent AddImageAlias gives the copy: the smaller of the two at the base found.
            constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
            if (std::ranges::find(join_left_aliased_ids, copy.id) != join_left_aliased_ids.end()) {
                const std::optional<SubresourceBase> base =
                    FindSubresource(image_info, overlap, new_image.gpu_addr, options, false, true);
                if (base && covers(overlap.info, base->level)) {
                    return true;
                }
            } else {
                const std::optional<SubresourceBase> base =
                    FindSubresource(overlap.info, new_image, overlap.gpu_addr, options, false, true);
                if (base && base->level == 0 && base->layer == 0 && covers(overlap.info, 0)) {
                    return true;
                }
            }
        }
        return false;
    };
    // Whether the copies from overlaps the GPU wrote each write a part of the image and together
    // all of it, for the layered and mipmapped 2D images overwritten_by_copies leaves out (ported):
    // Mario Kart World renders each layer of its shadow map and colour
    // arrays as a target of its own, then samples the array, and the upload under those copies
    // took 13-31 ms an array. Only at 1x, where the copies below are unscaled.
    const auto overwritten_by_layer_copies = [&] {
        constexpr ImageFlagBits unusual = ImageFlagBits::Sparse | ImageFlagBits::SparseLayers |
                                          ImageFlagBits::SparseTiles |
                                          ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked;
        if (can_rescale || new_info.num_samples != 1 ||
            True(new_image.flags & (unusual | ImageFlagBits::Converted |
                                    ImageFlagBits::AsynchronousDecode))) {
            return false;
        }
        std::vector<u8> written(
            static_cast<size_t>(new_info.resources.levels * new_info.resources.layers));
        for (const JoinCopy& copy : join_copies_to_do) {
            const Image& overlap = slot_images[copy.id];
            if (copy.is_alias || False(overlap.flags & ImageFlagBits::GpuModified) ||
                overlap.info.num_samples != 1) {
                continue;
            }
            const std::optional<SubresourceBase> base = new_image.TryFindBase(overlap.gpu_addr);
            if (!base) {
                continue;
            }
            MarkSubresourcesWrittenWhole(new_image, overlap,
                                         MakeShrinkImageCopies(new_info, overlap.info, *base, 1, 0),
                                         written);
        }
        return std::ranges::all_of(written, [](u8 whole) { return whole != 0; });
    };
    // EDEN_NX2_RT_DEPTH_NOUPLOAD (live, bits, 0 = off, the default; ported from its
    // texture_cache.h nup_skip_depth_upload): a new depth/stencil image starts without
    // the upload from guest memory. Guest memory never receives the GPU's depth (no write-back),
    // so the upload only copies stale bytes; Ocarina of Time makes ~2.5 such targets a frame
    // (shadow cascades sized anew each frame, 259x259 ... 888x888 D32). Not exact: what the
    // image holds where the game neither clears nor draws is undefined instead of stale guest
    // bytes (the original names shadow popping with it), so it comes after the exact
    // rules: a clear the target follows through IsFullClear (EXP56) and copies that write it
    // whole (EXP8 bit 2) decide first.
    // Bit 1 = a render target lookup's new target with nothing to join (no older image to copy).
    // Bit 2 = a new image whose every join copy comes from a GPU-written image (sampled shadow
    // maps made for a descriptor shaped unlike the target); texels the copies miss are undefined.
    // 3 = as the original shipped.
    const auto depth_noupload = [&] {
        const int mode = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_rt_depth_noupload,
                                                          "EDEN_NX2_RT_DEPTH_NOUPLOAD");
        if (mode == 0) {
            return false;
        }
        const auto type = VideoCore::Surface::GetFormatType(new_info.format);
        if (type != VideoCore::Surface::SurfaceType::Depth &&
            type != VideoCore::Surface::SurfaceType::Stencil &&
            type != VideoCore::Surface::SurfaceType::DepthStencil) {
            return false;
        }
        if (join_copies_to_do.empty()) {
            return (mode & 1) != 0 && render_target_lookup;
        }
        return (mode & 2) != 0 && std::ranges::all_of(join_copies_to_do, [this](const JoinCopy& copy) {
                   return True(slot_images[copy.id].flags & ImageFlagBits::GpuModified);
               });
    };
    const bool clear_skip = overwritten_by_clear();
    // EDEN_NX2_EXP8 bit 2: skip the upload the copies overwrite.
    const bool copy_skip = !clear_skip && (VideoCore::Nx2Debug::Experiment(8) & 2) != 0 &&
                           (overwritten_by_copies() || overwritten_by_layer_copies());
    const bool depth_skip = !clear_skip && !copy_skip && depth_noupload();
    if (VideoCore::Nx2Debug::Experiment(7) != 0 && new_image.guest_size_bytes >= 1_MiB)
        [[unlikely]] {
        // Why a large new image was (or was not) uploaded from guest memory.
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 300) {
            u32 aliases = 0;
            u32 gpu_written = 0;
            for (const JoinCopy& copy : join_copies_to_do) {
                aliases += copy.is_alias ? 1 : 0;
                const bool gpu = True(slot_images[copy.id].flags & ImageFlagBits::GpuModified);
                gpu_written += gpu ? 1 : 0;
            }
            LOG_WARNING(HW_GPU,
                        "New image format {} {}x{}x{} levels {} layers {} at {:#x}: {} copies "
                        "({} aliases, {} GPU-written), {} bad overlaps, clearing {}, covered {}, "
                        "upload {}{}{}",
                        static_cast<u32>(new_info.format), new_info.size.width,
                        new_info.size.height, new_info.size.depth, new_info.resources.levels,
                        new_info.resources.layers, new_image.gpu_addr, join_copies_to_do.size(),
                        aliases, gpu_written, join_bad_overlap_ids.size(),
                        clearing_render_targets,
                        overwritten_by_copies() || overwritten_by_layer_copies(),
                        clear_skip   ? "skipped (clear)"
                        : copy_skip  ? "skipped (copies)"
                        : depth_skip ? "skipped (depth)"
                                     : "done",
                        clear_skip || clear_refusal[0] == '\0' ? "" : ", clear refused: ",
                        clear_skip ? "" : clear_refusal);
        }
    }
    const auto refresh_start = std::chrono::steady_clock::now();
    if (clear_skip) {
        // Stays CpuModified: UpdateRenderTargets' PrepareImage drops the upload only when
        // IsFullClear agrees the clear covers the whole image, and uploads it otherwise.
        runtime.TransitionImageLayout(new_image);
    } else if (copy_skip || depth_skip) {
        // As RefreshContents leaves it, with the contents the copies write (depth_skip: and
        // undefined elsewhere; also when a partial clear creates it, PrepareImage would upload it).
        new_image.flags &= ~ImageFlagBits::CpuModified;
        TrackImage(new_image, new_image_id);
        runtime.TransitionImageLayout(new_image);
    } else {
        // TODO: Only upload what we need
        RefreshContents(new_image, new_image_id);
    }
    rt_stats.refresh_ns += NsSince(refresh_start);
    if (!relocation_tombstones.empty()) {
        CheckTombstoneCreation(new_image_id, new_image, clear_skip); // EXP43=4
    }

    if (can_rescale) {
        ScaleUp(new_image);
    } else {
        ScaleDown(new_image);
    }

    std::ranges::sort(join_copies_to_do, [this](const JoinCopy& lhs, const JoinCopy& rhs) {
        const ImageBase& lhs_image = slot_images[lhs.id];
        const ImageBase& rhs_image = slot_images[rhs.id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });

    ImageBase& new_image_base = new_image;
    for (const ImageId aliased_id : join_right_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        size_t alias_index = new_image_base.aliased_images.size();
        if (!AddImageAlias(new_image_base, aliased, new_image_id, aliased_id)) {
            continue;
        }
        join_alias_indices.emplace(aliased_id, alias_index);
        new_image.flags |= ImageFlagBits::Alias;
    }
    for (const ImageId aliased_id : join_left_aliased_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        size_t alias_index = new_image_base.aliased_images.size();
        if (!AddImageAlias(aliased, new_image_base, aliased_id, new_image_id)) {
            continue;
        }
        join_alias_indices.emplace(aliased_id, alias_index);
        new_image.flags |= ImageFlagBits::Alias;
    }
    for (const ImageId aliased_id : join_bad_overlap_ids) {
        ImageBase& aliased = slot_images[aliased_id];
        aliased.overlapping_images.push_back(new_image_id);
        new_image.overlapping_images.push_back(aliased_id);
        aliased.covered_overlap_frame = frame_tick + 1; // FRITCHII_COVERED_RETIRE: check again
        if (aliased.info.resources.levels == 1 && aliased.info.block.depth == 0 &&
            aliased.overlapping_images.size() > 1) {
            aliased.flags |= ImageFlagBits::BadOverlap;
        }
        if (new_image.info.resources.levels == 1 && new_image.info.block.depth == 0 &&
            new_image.overlapping_images.size() > 1) {
            new_image.flags |= ImageFlagBits::BadOverlap;
        }
        WatchColdImage(aliased_id);
    }
    if (!join_bad_overlap_ids.empty()) {
        WatchColdImage(new_image_id);
    }

    const auto joined_delete_start = std::chrono::steady_clock::now();
    for (const auto& copy_object : join_copies_to_do) {
        Image& overlap = slot_images[copy_object.id];
        if (copy_object.is_alias) {
            if (!overlap.IsSafeGpuCopy()) {
                continue;
            }
            if (overlap.info.num_samples != new_image.info.num_samples) {
                continue;
            }
            const auto alias_pointer = join_alias_indices.find(copy_object.id);
            if (alias_pointer == join_alias_indices.end()) {
                continue;
            }
            const AliasedImage& aliased = new_image.aliased_images[alias_pointer->second];
            CopyImage(new_image_id, aliased.id, aliased.copies);
            new_image.modification_tick = overlap.modification_tick;
            continue;
        }
        if (True(overlap.flags & ImageFlagBits::GpuModified)) {
            new_image.flags |= ImageFlagBits::GpuModified;
            CountGpuModified(new_image); // EXP100 (a no-op until RegisterImage below counts it)
            const auto& resolution = Settings::values.resolution_info;
            const SubresourceBase base = new_image.TryFindBase(overlap.gpu_addr).value();
            const u32 up_scale = can_rescale ? resolution.up_scale : 1;
            const u32 down_shift = can_rescale ? resolution.down_shift : 0;
            auto copies = MakeShrinkImageCopies(new_info, overlap.info, base, up_scale, down_shift);
            if (overlap.info.num_samples != new_image.info.num_samples) {
                runtime.CopyImageMSAA(new_image, overlap, copies);
            } else {
                runtime.CopyImage(new_image, overlap, copies);
            }
            new_image.modification_tick = overlap.modification_tick;
        }
        if (True(overlap.flags & ImageFlagBits::Tracked)) {
            UntrackImage(overlap, copy_object.id);
        }
        UnregisterImage(copy_object.id);
        DeleteImage(copy_object.id);
    }
    rt_stats.delete_ns += NsSince(joined_delete_start);

    if (depth_census_on && render_target_lookup &&
        new_info.format == VideoCore::Surface::PixelFormat::D32_FLOAT) [[unlikely]] {
        CensusNoteMade(new_image_id, new_info, gpu_addr);
    }
    RegisterImage(new_image_id);
    return new_image_id;
}

template <class P>
std::optional<typename TextureCache<P>::BlitImages> TextureCache<P>::GetBlitImages(
    const Tegra::Engines::Fermi2D::Surface& dst, const Tegra::Engines::Fermi2D::Surface& src,
    const Tegra::Engines::Fermi2D::Config& copy) {

    constexpr auto FIND_OPTIONS = RelaxedOptions::Samples;
    const GPUVAddr dst_addr = dst.Address();
    const GPUVAddr src_addr = src.Address();
    ImageInfo dst_info(dst);
    ImageInfo src_info(src);
    const bool can_be_depth_blit =
        dst_info.format == src_info.format && copy.filter == Tegra::Engines::Fermi2D::Filter::Point;
    ImageId dst_id;
    ImageId src_id;
    RelaxedOptions try_options = FIND_OPTIONS;
    if (can_be_depth_blit) {
        try_options |= RelaxedOptions::Format;
    }
    do {
        has_deleted_images = false;
        src_id = FindImage(src_info, src_addr, try_options);
        dst_id = FindImage(dst_info, dst_addr, try_options);
        if (!copy.must_accelerate) {
            do {
                if (!src_id && !dst_id) {
                    return std::nullopt;
                }
                if (src_id && True(slot_images[src_id].flags & ImageFlagBits::GpuModified)) {
                    break;
                }
                if (dst_id && True(slot_images[dst_id].flags & ImageFlagBits::GpuModified)) {
                    break;
                }
                return std::nullopt;
            } while (false);
        }
        const ImageBase* const src_image = src_id ? &slot_images[src_id] : nullptr;
        if (src_image && src_image->info.num_samples > 1) {
            RelaxedOptions find_options{FIND_OPTIONS | RelaxedOptions::ForceBrokenViews};
            src_id = FindOrInsertImage(src_info, src_addr, find_options);
            dst_id = FindOrInsertImage(dst_info, dst_addr, find_options);
            if (has_deleted_images) {
                continue;
            }
            break;
        }
        if (can_be_depth_blit) {
            const ImageBase* const dst_image = dst_id ? &slot_images[dst_id] : nullptr;
            DeduceBlitImages(dst_info, src_info, dst_image, src_image);
            if (GetFormatType(dst_info.format) != GetFormatType(src_info.format)) {
                continue;
            }
        }
        if (!src_id) {
            src_id = InsertImage(src_info, src_addr, RelaxedOptions{});
        }
        if (!dst_id) {
            dst_id = InsertImage(dst_info, dst_addr, RelaxedOptions{});
        }
    } while (has_deleted_images);
    if (!src_id || !dst_id) {
        return std::nullopt;
    }
    const ImageBase& src_image = slot_images[src_id];
    const ImageBase& dst_image = slot_images[dst_id];
    const bool native_bgr = runtime.HasNativeBgr();
    if (GetFormatType(dst_info.format) != GetFormatType(dst_image.info.format) ||
        GetFormatType(src_info.format) != GetFormatType(src_image.info.format) ||
        !VideoCore::Surface::IsViewCompatible(dst_info.format, dst_image.info.format, false,
                                              native_bgr) ||
        !VideoCore::Surface::IsViewCompatible(src_info.format, src_image.info.format, false,
                                              native_bgr)) {
        // Make sure the images match the expected format.
        do {
            has_deleted_images = false;
            src_id = FindOrInsertImage(src_info, src_addr, RelaxedOptions{});
            dst_id = FindOrInsertImage(dst_info, dst_addr, RelaxedOptions{});
        } while (has_deleted_images);
        if (!src_id || !dst_id) {
            return std::nullopt;
        }
    }
    return {BlitImages{
        .dst_id = dst_id,
        .src_id = src_id,
        .dst_format = dst_info.format,
        .src_format = src_info.format,
    }};
}

template <class P>
ImageId TextureCache<P>::FindDMAImage(const ImageInfo& info, GPUVAddr gpu_addr) {
    const std::optional<DAddr> cpu_addr = ImageCpuAddress(gpu_addr, CalculateGuestSizeInBytes(info));
    if (!cpu_addr) {
        return ImageId{};
    }
    ImageId image_id{};
    boost::container::small_vector<ImageId, 8> image_ids;
    const auto lambda = [&](ImageId existing_image_id, ImageBase& existing_image) {
        if (True(existing_image.flags & ImageFlagBits::Remapped)) {
            return false;
        }
        if (info.type == ImageType::Linear || existing_image.info.type == ImageType::Linear)
            [[unlikely]] {
            const bool strict_size = True(existing_image.flags & ImageFlagBits::Strong);
            const ImageInfo& existing = existing_image.info;
            if (existing_image.gpu_addr == gpu_addr && existing.type == info.type &&
                existing.pitch == info.pitch &&
                IsPitchLinearSameSize(existing, info, strict_size) &&
                IsViewCompatible(existing.format, info.format, false, true)) {
                image_id = existing_image_id;
                image_ids.push_back(existing_image_id);
                return true;
            }
        } else if (IsSubCopy(info, existing_image, gpu_addr)) {
            if (True(existing_image.flags & ImageFlagBits::Unbacked)) {
                static u32 logged{};
                if (logged++ < 64) {
                    LOG_INFO(HW_GPU, "DMA copy at {:#x} ({}x{}) into unbacked image {:#x}", gpu_addr,
                             info.size.width, info.size.height, existing_image.gpu_addr);
                }
            }
            image_id = existing_image_id;
            image_ids.push_back(existing_image_id);
            return true;
        }
        return false;
    };
    ForEachImageInRegion(*cpu_addr, CalculateGuestSizeInBytes(info), lambda);
    if (image_ids.size() <= 1) [[likely]] {
        return image_id;
    }
    auto image_ids_compare = [this](ImageId a, ImageId b) {
        auto& image_a = slot_images[a];
        auto& image_b = slot_images[b];
        return image_a.modification_tick < image_b.modification_tick;
    };
    return *std::ranges::max_element(image_ids, image_ids_compare);
}

template <class P>
std::pair<u32, u32> TextureCache<P>::PrepareDmaImage(ImageId dst_id, GPUVAddr base_addr,
                                                     bool mark_as_modified) {
    // As in BlitImage: land a parked deferred clear before the DMA write.
    runtime.FlushDeferredClear();
    const auto& image = slot_images[dst_id];
    const auto base = image.TryFindBase(base_addr);
    PrepareImage(dst_id, mark_as_modified, false);
    auto& new_image = slot_images[dst_id];
    TouchImageLru(new_image);
    return std::make_pair(base->level, base->layer);
}

template <class P>
SamplerId TextureCache<P>::FindSampler(const TSCEntry& config, bool compute) {
    if (std::ranges::all_of(config.raw, [](u64 value) { return value == 0; })) {
        return NULL_SAMPLER_ID;
    }
    const auto [pair, is_new] = channel_state->samplers.try_emplace(config);
    if (is_new) {
        pair->second = slot_samplers.insert(runtime, config);
    }
    return pair->second;
}

template <class P>
void TextureCache<P>::CensusNoteMade(ImageId image_id, const ImageInfo& info, GPUVAddr gpu_addr) {
    const size_t n = census_this_frame.size();
    ++census.made;
    if (info.size.width == 384 && info.size.height == 384) {
        ++census.at_floor;
    }
    if (n < census_last_frame.size()) {
        const CensusMade& prev = census_last_frame[n];
        ++(prev.gpu_addr == gpu_addr ? census.same_address : census.moved);
        const u32 dw = prev.width > info.size.width ? prev.width - info.size.width
                                                    : info.size.width - prev.width;
        const u32 dh = prev.height > info.size.height ? prev.height - info.size.height
                                                      : info.size.height - prev.height;
        const u32 side = (std::max)(dw, dh);
        ++census.side_change[side == 0 ? 0 : side <= 8 ? 1 : side <= 64 ? 2 : 3];
    } else {
        ++census.no_predecessor;
    }
    census_this_frame.push_back({gpu_addr, info.size.width, info.size.height});
    census_live[image_id] = CensusTarget{.gpu_addr = gpu_addr, .frame = frame_tick};
    census_by_addr[gpu_addr] = image_id;
}

template <class P>
void TextureCache<P>::CensusNoteRenderTargets(bool is_clear) {
    const ImageViewId view_id = render_targets.depth_buffer_id;
    if (view_id == NULL_IMAGE_VIEW_ID || view_id == ImageViewId{}) {
        return;
    }
    const auto it = census_live.find(slot_image_views[view_id].image_id);
    if (it == census_live.end()) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    CensusTarget& target = it->second;
    if (is_clear) {
        target.cleared |= regs.clear_surface.Z != 0 || regs.clear_surface.S != 0;
    } else {
        target.drawn = true;
        target.drawn_writes |= regs.depth_test_enable != 0 && regs.depth_write_enabled != 0;
    }
}

template <class P>
void TextureCache<P>::CensusNoteSampled(ImageViewId view_id, const TSCEntry* sampler,
                                        u8 shader_use, bool shader_dref) {
    if (view_id == NULL_IMAGE_VIEW_ID || view_id == ImageViewId{}) {
        return;
    }
    const auto it = census_live.find(slot_image_views[view_id].image_id);
    if (it == census_live.end()) {
        return;
    }
    CensusTarget& target = it->second;
    target.shader_use |= shader_use;
    if (target.sampled_same || target.sampled_later) {
        return;
    }
    (target.frame == frame_tick ? target.sampled_same : target.sampled_later) = true;
    census.shader_dref += shader_dref ? 1 : 0;
    if (sampler == nullptr) {
        ++census.no_sampler;
        return;
    }
    using Tegra::Texture::WrapMode;
    const auto kind = [](WrapMode mode) {
        switch (mode) {
        case WrapMode::Wrap:
        case WrapMode::Mirror:
            return 0;
        case WrapMode::Border:
        case WrapMode::MirrorOnceBorder:
            return 2;
        default:
            return 1; // clamp to edge, clamp, the mirror-once clamps
        }
    };
    const int u = kind(sampler->wrap_u);
    const int v = kind(sampler->wrap_v);
    ++census.address_mode[u == v ? u : 3];
    census.compare_on += sampler->depth_compare_enabled != 0 ? 1 : 0;
}

template <class P>
void TextureCache<P>::CensusNoteRead(const ImageBase& image, bool dma) {
    const auto by_addr = census_by_addr.find(image.gpu_addr);
    if (by_addr == census_by_addr.end() || &slot_images[by_addr->second] != &image) {
        return;
    }
    const auto it = census_live.find(by_addr->second);
    if (it != census_live.end()) {
        (dma ? it->second.dma_read : it->second.cpu_read) = true;
    }
}

template <class P>
void TextureCache<P>::CensusFinish(ImageId image_id, bool deleted) {
    const auto it = census_live.find(image_id);
    if (it == census_live.end()) {
        return;
    }
    const CensusTarget& t = it->second;
    ++census.done;
    census.deleted += deleted ? 1 : 0;
    if (t.cleared && t.drawn) {
        ++census.cleared_and_drawn;
    } else if (t.cleared) {
        ++census.cleared_only;
    } else if (t.drawn) {
        ++census.drawn_only;
    } else {
        ++census.neither;
    }
    census.drawn_writes += t.drawn_writes ? 1 : 0;
    census.via_texture += t.via_texture ? 1 : 0;
    census.cpu_read += t.cpu_read ? 1 : 0;
    census.dma_read += t.dma_read ? 1 : 0;
    if (t.sampled_same || t.sampled_later) {
        ++(t.sampled_same ? census.sampled_same : census.sampled_later);
        using Desc = Shader::TextureDescriptor;
        if ((t.shader_use & Desc::CensusKnown) == 0) {
            ++census.use_unknown;
        } else {
            census.use_fetch += (t.shader_use & Desc::CensusFetch) != 0 ? 1 : 0;
            census.use_query += (t.shader_use & Desc::CensusQuery) != 0 ? 1 : 0;
            census.use_offset += (t.shader_use & Desc::CensusOffset) != 0 ? 1 : 0;
            census.use_gather += (t.shader_use & Desc::CensusGather) != 0 ? 1 : 0;
            census.use_rect += (t.shader_use & Desc::CensusRect) != 0 ? 1 : 0;
        }
    } else {
        ++census.never_sampled;
    }
    if (const auto by_addr = census_by_addr.find(t.gpu_addr);
        by_addr != census_by_addr.end() && by_addr->second == image_id) {
        census_by_addr.erase(by_addr);
    }
    census_live.erase(it);
}

template <class P>
void TextureCache<P>::CensusTickFrame() {
    census_last_frame.swap(census_this_frame);
    census_this_frame.clear();
    std::vector<ImageId> aged;
    for (const auto& [image_id, target] : census_live) {
        if (frame_tick - target.frame > CensusMaxAge) {
            aged.push_back(image_id);
        }
    }
    for (const ImageId image_id : aged) {
        CensusFinish(image_id, false);
    }
    if (frame_tick % 600 != 0) {
        return;
    }
    const CensusCounts& c = census;
    LOG_INFO(HW_GPU,
             "Depth census (EDEN_NX2_DEPTH_CENSUS) last 600 frames: {} D32 targets made, {} at "
             "384x384; vs last frame's n-th: {} same address, {} moved, {} none; side change "
             "0/1-8/9-64/more {}/{}/{}/{}. Done {} ({} deleted, the rest {} frames old): "
             "cleared+drawn {}, cleared only {}, drawn only {}, neither {} ({} drawn with depth "
             "writes); sampled same frame {}, later {}, never {}; joined into a texture lookup's "
             "image {}; CPU-read {}, copied to a buffer {}. Sampled, first sampler: address "
             "repeat/clamp/border/mixed {}/{}/{}/{}, compare on {}, shader depth-reference {}, "
             "no sampler {}; shader reads: fetch {}, size query {}, offsets {}, gather {}, rect "
             "{}, unknown (precompiled info) {}; {} live",
             c.made, c.at_floor, c.same_address, c.moved, c.no_predecessor, c.side_change[0],
             c.side_change[1], c.side_change[2], c.side_change[3], c.done, c.deleted,
             CensusMaxAge, c.cleared_and_drawn, c.cleared_only, c.drawn_only, c.neither,
             c.drawn_writes, c.sampled_same, c.sampled_later, c.never_sampled, c.via_texture,
             c.cpu_read, c.dma_read, c.address_mode[0], c.address_mode[1], c.address_mode[2],
             c.address_mode[3], c.compare_on, c.shader_dref, c.no_sampler, c.use_fetch,
             c.use_query, c.use_offset, c.use_gather, c.use_rect, c.use_unknown,
             census_live.size());
    census = {};
    // The same switch: the occlusion reports and the 0xe2c reads of them (potroom 4b).
    VideoCommon::Nx2ReportCensus::LogAndReset();
}

template <class P>
ImageViewId TextureCache<P>::FindColorBuffer(size_t index) {
    const auto& regs = maxwell3d->regs;
    if (index >= regs.rt_control.count) {
        return ImageViewId{};
    }
    const auto& rt = regs.rt[index];
    const GPUVAddr gpu_addr = rt.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    if (rt.format == Tegra::RenderTargetFormat::NONE) {
        return ImageViewId{};
    }
    const ImageInfo info(regs.rt[index], regs.anti_alias_samples_mode);
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindDepthBuffer() {
    const auto& regs = maxwell3d->regs;
    if (!regs.zeta_enable) {
        return ImageViewId{};
    }
    const GPUVAddr gpu_addr = regs.zeta.Address();
    if (gpu_addr == 0) {
        return ImageViewId{};
    }
    ImageInfo info(regs.zeta, regs.zeta_size, regs.anti_alias_samples_mode);
    if (!clearing_render_targets && !clear_exact_pending.empty()) {
        // EXP68 bit 1 check: the first depth bind at an address after its clear uses the exact
        // size the clear went to (several NPC targets are cleared before their draws).
        if (const auto it = clear_exact_pending.find(gpu_addr); it != clear_exact_pending.end()) {
            if (info.size.width == it->second.first && info.size.height == it->second.second) {
                ++clear_exact_confirmed;
            } else {
                ++clear_exact_missed;
                clear_exact_unsafe.insert(gpu_addr);
                LOG_WARNING(HW_GPU,
                            "Clear to exact size (EXP68): the bind after the clear at {:#x} is "
                            "{}x{}, not {}x{}; this address keeps the register size",
                            gpu_addr, info.size.width, info.size.height, it->second.first,
                            it->second.second);
            }
            clear_exact_pending.erase(it);
        }
    }
    if (clearing_render_targets && (VideoCore::Nx2Debug::Experiment(68) & 1) != 0 &&
        regs.clear_surface.Z && regs.clear_control.use_scissor != 0 &&
        info.type == ImageType::e2D && info.resources.levels == 1 && info.resources.layers == 1 &&
        info.num_samples == 1 && !clear_exact_unsafe.contains(gpu_addr)) {
        // The game clears the per-NPC depth target with the width register rounded up to 16 and
        // a scissor of the real square, then draws into the real square: two images a frame
        // with an upload and alias copies between them (fable52_answer_npcdips 1).
        const auto& scissor = regs.scissor_test[0];
        const u32 bytes = VideoCore::Surface::BytesPerBlock(info.format);
        const u32 scissor_width = scissor.max_x;
        if (scissor.min_x == 0 && scissor.min_y == 0 && scissor.max_y >= info.size.height &&
            scissor_width != 0 && scissor_width < info.size.width && bytes != 0 &&
            Common::AlignUp(scissor_width * bytes, 64U) ==
                Common::AlignUp(info.size.width * bytes, 64U)) {
            info.size.width = scissor_width;
            clear_exact_pending[gpu_addr] = {info.size.width, info.size.height};
            ++clear_exact_used;
        }
    }
    return FindRenderTargetView(info, gpu_addr);
}

template <class P>
ImageViewId TextureCache<P>::FindRenderTargetView(const ImageInfo& info, GPUVAddr gpu_addr) {
    static constexpr u32 MaxRenderTargetSize = 32768;
    static constexpr s32 MaxRenderTargetLayers = 2048;
    if (info.size.width == 0 || info.size.height == 0 || info.size.width > MaxRenderTargetSize ||
        info.size.height > MaxRenderTargetSize || info.resources.layers > MaxRenderTargetLayers) {
        LOG_ERROR(HW_GPU, "Ignoring {}x{} render target with {} layers at {:#x}", info.size.width,
                  info.size.height, info.resources.layers, gpu_addr);
        return ImageViewId{};
    }
    ImageId image_id{};
    bool delete_state = has_deleted_images;
    render_target_lookup = true; // EDEN_NX2_RT_DEPTH_NOUPLOAD: JoinImages sees a render target
    do {
        has_deleted_images = false;
        image_id = FindOrInsertImage(info, gpu_addr);
        delete_state |= has_deleted_images;
    } while (has_deleted_images);
    render_target_lookup = false;
    has_deleted_images = delete_state;
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }
    Image& image = slot_images[image_id];
    if (RtEvictPressureFrames() != 0 && Tegra::g_nx2_ampere.load(std::memory_order_relaxed) &&
        image.modification_tick == 0 && True(image.flags & ImageFlagBits::CpuModified) &&
        False(image.flags & ImageFlagBits::GpuModified) &&
        False(image.flags & (ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
                             ImageFlagBits::Unbacked)) &&
        image.dirty_layers.empty()) {
        image.flags &= ~ImageFlagBits::CpuModified;
        if (False(image.flags & ImageFlagBits::Tracked)) {
            TrackImage(image, image_id);
        }
        if (False(image.flags & ImageFlagBits::TransientTarget)) {
            image.flags |= ImageFlagBits::TransientTarget;
            transient_targets.push_back({image_id, image.allocation_tick, frame_tick, 0});
        }
    }
    const ImageViewType view_type = RenderTargetImageViewType(info);
    SubresourceBase base;
    if (image.info.type == ImageType::Linear) {
        base = SubresourceBase{.level = 0, .layer = 0};
    } else {
        base = image.TryFindBase(gpu_addr).value();
    }
    const s32 layers = image.info.type == ImageType::e3D ? info.size.depth : info.resources.layers;
    if (RenderTrace::SampleFrame(frame_tick)) {
        LOG_DEBUG(HW_GPU,
                  "Render trace target match: docked={} requested={:#x} {}x{} layers={} format={} "
                  "stride={} block={}/{}/{} selected={:#x} {}x{} layers={} format={} stride={} "
                  "block={}/{}/{} base_level={} base_layer={} aliases={} overlaps={}",
                  Settings::IsDockedMode(), gpu_addr, info.size.width, info.size.height, layers,
                  info.format, info.layer_stride, info.block.width, info.block.height,
                  info.block.depth, image.gpu_addr, image.info.size.width, image.info.size.height,
                  image.info.resources.layers, image.info.format, image.info.layer_stride,
                  image.info.block.width, image.info.block.height, image.info.block.depth,
                  base.level, base.layer, image.aliased_images.size(), image.overlapping_images.size());
    }
    if (True(image.flags & ImageFlagBits::Unbacked)) {
        static u32 logged{};
        if (logged++ < 64) {
            LOG_INFO(HW_GPU, "Render target in unbacked image {:#x} at {:#x}: {}x{} level {} layer {}",
                     image.gpu_addr, gpu_addr, info.size.width, info.size.height, base.level, base.layer);
        }
    }
    const SubresourceRange range{
        .base = base,
        .extent = {.levels = 1, .layers = layers},
    };
    return FindOrEmplaceImageView(image_id, ImageViewInfo(view_type, info.format, range));
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegion(DAddr cpu_addr, size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 32> images;
    boost::container::small_vector<ImageMapId, 32> maps;
    ForEachCPUPage(cpu_addr, size, [this, &images, &maps, cpu_addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const ImageMapId map_id : it->second) {
            ImageMapView& map = slot_map_views[map_id];
            if (map.picked) {
                continue;
            }
            if (!map.Overlaps(cpu_addr, size)) {
                continue;
            }
            map.picked = true;
            maps.push_back(map_id);
            Image& image = slot_images[map.image_id];
            if (True(image.flags & ImageFlagBits::Picked)) {
                continue;
            }
            image.flags |= ImageFlagBits::Picked;
            images.push_back(map.image_id);
            if constexpr (BOOL_BREAK) {
                if (func(map.image_id, image)) {
                    return true;
                }
            } else {
                func(map.image_id, image);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
    for (const ImageMapId map_id : maps) {
        slot_map_views[map_id].picked = false;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachImageInRegionGPU(size_t as_id, GPUVAddr gpu_addr, size_t size,
                                              Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 8> images;
    auto storage_id = getStorageID(as_id);
    if (!storage_id) {
        return;
    }
    auto& gpu_page_table = gpu_page_table_storage[*storage_id * 2];
    ForEachGPUPage(gpu_addr, size,
                   [this, &gpu_page_table, &images, gpu_addr, size, func](u64 page) {
                       const auto it = gpu_page_table.find(page);
                       if (it == gpu_page_table.end()) {
                           if constexpr (BOOL_BREAK) {
                               return false;
                           } else {
                               return;
                           }
                       }
                       for (const ImageId image_id : it->second) {
                           Image& image = slot_images[image_id];
                           if (True(image.flags & ImageFlagBits::Picked)) {
                               continue;
                           }
                           if (!image.OverlapsGPU(gpu_addr, size)) {
                               continue;
                           }
                           image.flags |= ImageFlagBits::Picked;
                           images.push_back(image_id);
                           if constexpr (BOOL_BREAK) {
                               if (func(image_id, image)) {
                                   return true;
                               }
                           } else {
                               func(image_id, image);
                           }
                       }
                       if constexpr (BOOL_BREAK) {
                           return false;
                       }
                   });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachSparseImageInRegion(size_t as_id, GPUVAddr gpu_addr, size_t size,
                                                 Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
    constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<ImageId, 8> images;
    auto storage_id = getStorageID(as_id);
    if (!storage_id) {
        return;
    }
    auto& sparse_page_table = gpu_page_table_storage[*storage_id * 2 + 1];
    ForEachGPUPage(gpu_addr, size,
                   [this, &sparse_page_table, &images, gpu_addr, size, func](u64 page) {
                       const auto it = sparse_page_table.find(page);
                       if (it == sparse_page_table.end()) {
                           if constexpr (BOOL_BREAK) {
                               return false;
                           } else {
                               return;
                           }
                       }
                       for (const ImageId image_id : it->second) {
                           Image& image = slot_images[image_id];
                           if (True(image.flags & ImageFlagBits::Picked)) {
                               continue;
                           }
                           if (!image.OverlapsGPU(gpu_addr, size)) {
                               continue;
                           }
                           image.flags |= ImageFlagBits::Picked;
                           images.push_back(image_id);
                           if constexpr (BOOL_BREAK) {
                               if (func(image_id, image)) {
                                   return true;
                               }
                           } else {
                               func(image_id, image);
                           }
                       }
                       if constexpr (BOOL_BREAK) {
                           return false;
                       }
                   });
    for (const ImageId image_id : images) {
        slot_images[image_id].flags &= ~ImageFlagBits::Picked;
    }
}

template <class P>
template <typename Func>
void TextureCache<P>::ForEachSparseSegment(ImageBase& image, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, GPUVAddr, DAddr, size_t>::type;
    constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
    const auto segments = gpu_memory->GetSubmappedRange(image.gpu_addr, image.guest_size_bytes);
    for (const auto& [gpu_addr, size] : segments) {
        const std::optional<DAddr> cpu_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!cpu_addr) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(HW_GPU,
                            "Skipping sparse texture segment {:#x}+{:#x}: no device address",
                            gpu_addr, size);
            }
            continue;
        }
        if constexpr (RETURNS_BOOL) {
            if (func(gpu_addr, *cpu_addr, size)) {
                break;
            }
        } else {
            func(gpu_addr, *cpu_addr, size);
        }
    }
}

template <class P>
ImageViewId TextureCache<P>::FindOrEmplaceImageView(ImageId image_id, const ImageViewInfo& info) {
    Image& image = slot_images[image_id];
    if (const ImageViewId image_view_id = image.FindView(info); image_view_id) {
        return image_view_id;
    }
    if (True(image.flags & ImageFlagBits::Unbacked) &&
        False(image.flags & ImageFlagBits::SparseLayers)) {
        // How the game uses an image too large to back (e.g. a sparse virtual texture).
        LOG_INFO(HW_GPU, "View of unbacked image {:#x}: type {} format {} levels {}+{} layers {}+{}",
                 image.gpu_addr, static_cast<u32>(info.type), static_cast<u32>(info.format),
                 info.range.base.level, info.range.extent.levels, info.range.base.layer,
                 info.range.extent.layers);
    }
    const ImageViewId image_view_id = slot_image_views.insert(runtime, info, image_id, image, slot_images);
    Churn::Add(Churn::ImageViewsCreated);
    image.InsertView(info, image_view_id);
    ++bindings_epoch;
    return image_view_id;
}

template <class P>
void TextureCache<P>::RegisterImage(ImageId image_id) {
    ImageBase& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += ImageMemoryBytes(slot_images[image_id]);
    image.lru_index = lru_cache.Insert(image_id, frame_tick);
    image.lru_touched_tick = ~u64{0};

    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        (*channel_state->gpu_page_table)[page].push_back(image_id);
    });
    if (False(image.flags & ImageFlagBits::Sparse)) {
        auto map_id =
            slot_map_views.insert(image.gpu_addr, image.cpu_addr, image.guest_size_bytes, image_id);
        page_filter.Mark(image.cpu_addr, image.guest_size_bytes);
        image_count_filter.Add(image.cpu_addr, image.guest_size_bytes);
        // Memory stage step 7 (risk 9.1): a constant buffer write the stage made ahead of the
        // back under this image (counted, must be 0).
        VideoCommon::MemoryStageLink::NoteImageRegistered(image.cpu_addr, image.guest_size_bytes);
        image.nx2_counted_addr = image.cpu_addr;
        image.nx2_counted_size = image.guest_size_bytes;
        if (True(image.flags & ImageFlagBits::GpuModified)) {
            CountGpuModified(image); // EXP100: registered while already GPU-modified
        }
        ForEachCPUPage(image.cpu_addr, image.guest_size_bytes,
                       [this, map_id](u64 page) { page_table[page].push_back(map_id); });
        image.map_view_id = map_id;
        return;
    }
    boost::container::small_vector<ImageViewId, 16> sparse_maps;
    ForEachSparseSegment(
        image, [this, image_id, &sparse_maps](GPUVAddr gpu_addr, DAddr cpu_addr, size_t size) {
            auto map_id = slot_map_views.insert(gpu_addr, cpu_addr, size, image_id);
            page_filter.Mark(cpu_addr, size);
            sparse_page_filter.Mark(cpu_addr, size);
            ForEachCPUPage(cpu_addr, size,
                           [this, map_id](u64 page) { page_table[page].push_back(map_id); });
            sparse_maps.push_back(map_id);
        });
    sparse_views.emplace(image_id, std::move(sparse_maps));
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        (*channel_state->sparse_page_table)[page].push_back(image_id);
    });
}

template <class P>
void TextureCache<P>::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already registered image");
    image.flags &= ~ImageFlagBits::Registered;
    image.flags &= ~ImageFlagBits::BadOverlap;
    image.last_use_tick = lru_cache.Tick(image.lru_index);
    lru_cache.Free(image.lru_index);
    const auto& clear_page_table =
        [image_id](u64 page, ::Common::unordered_map<u64, std::vector<ImageId>, Common::IdentityHash<u64>>& selected_page_table) {
            const auto page_it = selected_page_table.find(page);
            if (page_it == selected_page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageId>& image_ids = page_it->second;
            const auto vector_it = std::ranges::find(image_ids, image_id);
            if (vector_it == image_ids.end()) {
                ASSERT_MSG(false, "Unregistering unregistered image in page={:#x}",
                           page << YUZU_PAGEBITS);
                return;
            }
            image_ids.erase(vector_it);
        };
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, &clear_page_table](u64 page) {
        clear_page_table(page, (*channel_state->gpu_page_table));
    });
    if (False(image.flags & ImageFlagBits::Sparse)) {
        const auto map_id = image.map_view_id;
        image_count_filter.Remove(image.nx2_counted_addr, image.nx2_counted_size);
        if (image.nx2_gpu_counted) { // EXP100
            image_seq.BeforeChange();
            gpu_image_filter.Remove(image.nx2_counted_addr, image.nx2_counted_size);
            image.nx2_gpu_counted = false;
        }
        image.nx2_counted_size = 0;
        ForEachCPUPage(image.cpu_addr, image.guest_size_bytes, [this, map_id](u64 page) {
            const auto page_it = page_table.find(page);
            if (page_it == page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageMapId>& image_map_ids = page_it->second;
            const auto vector_it = std::ranges::find(image_map_ids, map_id);
            if (vector_it == image_map_ids.end()) {
                ASSERT_MSG(false, "Unregistering unregistered image in page={:#x}",
                           page << YUZU_PAGEBITS);
                return;
            }
            image_map_ids.erase(vector_it);
        });
        slot_map_views.erase(map_id);
        return;
    }
    ForEachGPUPage(image.gpu_addr, image.guest_size_bytes, [this, &clear_page_table](u64 page) {
        clear_page_table(page, (*channel_state->sparse_page_table));
    });
    auto it = sparse_views.find(image_id);
    ASSERT(it != sparse_views.end());
    auto& sparse_maps = it->second;
    for (auto& map_view_id : sparse_maps) {
        const auto& map_range = slot_map_views[map_view_id];
        const DAddr cpu_addr = map_range.cpu_addr;
        const std::size_t size = map_range.size;
        ForEachCPUPage(cpu_addr, size, [this, image_id](u64 page) {
            const auto page_it = page_table.find(page);
            if (page_it == page_table.end()) {
                ASSERT_MSG(false, "Unregistering unregistered page={:#x}", page << YUZU_PAGEBITS);
                return;
            }
            std::vector<ImageMapId>& image_map_ids = page_it->second;
            auto vector_it = image_map_ids.begin();
            while (vector_it != image_map_ids.end()) {
                ImageMapView& map = slot_map_views[*vector_it];
                if (map.image_id != image_id) {
                    vector_it++;
                    continue;
                }
                if (!map.picked) {
                    map.picked = true;
                }
                vector_it = image_map_ids.erase(vector_it);
            }
        });
        slot_map_views.erase(map_view_id);
    }
    sparse_views.erase(it);
}

template <class P>
void TextureCache<P>::TrackImage(ImageBase& image, ImageId image_id) {
    ASSERT(False(image.flags & ImageFlagBits::Tracked));
    image.flags |= ImageFlagBits::Tracked;
    if (False(image.flags & ImageFlagBits::Sparse)) {
        if (image.cpu_addr < ~(1ULL << 40)) {
            device_memory.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, 1);
        }
        return;
    }
    if (True(image.flags & ImageFlagBits::Registered)) {
        auto it = sparse_views.find(image_id);
        ASSERT(it != sparse_views.end());
        auto& sparse_maps = it->second;
        for (auto& map_view_id : sparse_maps) {
            const auto& map = slot_map_views[map_view_id];
            const DAddr cpu_addr = map.cpu_addr;
            const std::size_t size = map.size;
            device_memory.UpdatePagesCachedCount(cpu_addr, size, 1);
        }
        return;
    }
    ForEachSparseSegment(image,
                         [this]([[maybe_unused]] GPUVAddr gpu_addr, DAddr cpu_addr, size_t size) {
                             device_memory.UpdatePagesCachedCount(cpu_addr, size, 1);
                         });
}

template <class P>
void TextureCache<P>::UntrackImage(ImageBase& image, ImageId image_id) {
    ASSERT(True(image.flags & ImageFlagBits::Tracked));
    image.flags &= ~ImageFlagBits::Tracked;
    if (False(image.flags & ImageFlagBits::Sparse)) {
        if (image.cpu_addr < ~(1ULL << 40)) {
            device_memory.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, -1);
        }
        return;
    }
    ASSERT(True(image.flags & ImageFlagBits::Registered));
    auto it = sparse_views.find(image_id);
    ASSERT(it != sparse_views.end());
    auto& sparse_maps = it->second;
    for (auto& map_view_id : sparse_maps) {
        const auto& map = slot_map_views[map_view_id];
        const DAddr cpu_addr = map.cpu_addr;
        const std::size_t size = map.size;
        device_memory.UpdatePagesCachedCount(cpu_addr, size, -1);
    }
}

template <class P>
void TextureCache<P>::DeleteImage(ImageId image_id, bool immediate_delete) {
    if (depth_census_on) [[unlikely]] {
        CensusFinish(image_id, true);
    }
    ImageBase& image = slot_images[image_id];
    ColdKeepUncharge(image); // whichever path deletes it (the collector included)
    if (True(image.flags & ImageFlagBits::CollectPending)) [[unlikely]] {
        // EXP121: deleted by another path before its queued download landed.
        for (size_t i = 0; i < collect_downloads.size(); ++i) {
            if (collect_downloads[i].image_id == image_id) {
                DropCollectDownload(i);
                ++collect_stats.dropped_deleted;
                break;
            }
        }
    }
    ++images_deleted;
    Churn::Add(Churn::ImagesDeleted);
    NoteThrashDelete(slot_images[image_id]);
    if (image.unmapped_tick != 0) {
        // Still flagged by an unmap. Unused since: no upload or full overwrite took the
        // CpuModified the unmap set, so nothing could have downloaded it either.
        const auto reason = static_cast<size_t>(delete_reason);
        ++unmap_stats.died[reason];
        if (True(image.flags & ImageFlagBits::CpuModified)) {
            ++unmap_stats.died_unused[reason];
            if (delete_reason == DeleteReason::GarbageCollect) {
                unmap_stats.collector_unused_frames += frame_tick + 1 - image.unmapped_tick;
                unmap_stats.collector_unused_bytes += image.unmapped_bytes;
            }
        }
        if (image.unmapped_verify == 1) {
            // EDEN_NX2_SWEEP_UNMAPPED=2: nothing uploaded into it since it was picked. Without
            // CpuModified, a target written whole took it (the invalidate path): nothing read.
            ++(True(image.flags & ImageFlagBits::CpuModified) ? unmap_stats.would_died_unused
                                                               : unmap_stats.would_overwritten);
        }
        ForgetUnmapped(image_id, image);
    }
    ++bindings_epoch;
    VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::Invalidation);
    if ((VideoCore::Nx2Debug::Experiment(76) & 4) != 0) [[unlikely]] {
        ForgetContent(image_id, image, true);
    }
    // EXP77: a GPU-written chain target the cold sweep takes leaves its shape at its address.
    if (delete_reason == DeleteReason::ColdSweep &&
        True(image.flags & ImageFlagBits::GpuModified) && RelocationShaped(image.info) &&
        VideoCore::Nx2Debug::Experiment(77) != 0 && nx2_swept_shapes.size() < 65536) {
        nx2_swept_shapes[image.gpu_addr] = RelocationShapeKey(image.info);
    }
    if (ImageHashLogEnabled()) [[unlikely]] {
        const auto reason = static_cast<size_t>(delete_reason);
        ++nx2_deletes[reason];
        if (nx2_deleted_reasons.size() < 200'000) {
            nx2_deleted_reasons[image.gpu_addr] = static_cast<u8>(
                reason | (True(image.flags & ImageFlagBits::GpuModified) ? 0x80 : 0));
        }
    }
    MarkStale(image);
    // Drop every decode still queued for it, whatever IsDecoding says: an image re-decoded while
    // a decode was in flight has two, and the first to finish clears the flag. A decode left
    // behind was uploaded into the next image to take this slot (another texture's data: garbage
    // textures, and a staging overrun when that image was smaller).
    for (auto& async_decode : async_decodes) {
        if (async_decode->image_id == image_id) {
            async_decode->cancelled = true;
        }
    }
    // Fritchii's free guard: an async unswizzle still queued for it (GPU unswizzle, 3D BCn) went
    // on reading and uploading through this slot in later frames, when the slot is gone or holds
    // another image: compute writes into a destroyed or wrong-sized image. Only the collector
    // skipped decoding images; unmaps, joins and the sweeps did not. Dropped here with its
    // staging buffer, which the copies it recorded so far (older than this delete) still use:
    // FreeDeferredStagingBuffer stamps it with the current tick.
    for (auto it = unswizzle_queue.begin(); it != unswizzle_queue.end();) {
        if (it->image_id != image_id) {
            ++it;
            continue;
        }
        if (it->initialized) {
            runtime.FreeDeferredStagingBuffer(it->staging_buffer);
        }
        if (free_guard.unswizzle_dropped++ < 8) {
            LOG_WARNING(HW_GPU,
                        "Free guard: async unswizzle of image {:#x} dropped with the image ({} of "
                        "{} bytes read)",
                        image.gpu_addr, it->current_offset, it->total_size);
        }
        it = unswizzle_queue.erase(it);
    }
    if (image.HasScaled()) {
        total_used_memory -= GetScaledImageSizeBytes(image);
    }
    // TickFrame sets the total from the driver, which can be under the sum of these estimates.
    total_used_memory -= (std::min)(total_used_memory, ImageMemoryBytes(slot_images[image_id]));
    const u64 freed_bytes = (image.HasScaled() ? GetScaledImageSizeBytes(image) : 0) +
                            ImageMemoryBytes(slot_images[image_id]);
    const GPUVAddr gpu_addr = image.gpu_addr;
    const auto alloc_it = image_allocs_table.find(gpu_addr);
    if (alloc_it == image_allocs_table.end()) {
        ASSERT_MSG(false, "Trying to delete an image alloc that does not exist in address {:#x}",
                   gpu_addr);
        return;
    }
    const ImageAllocId alloc_id = alloc_it->second;
    std::vector<ImageId>& alloc_images = slot_image_allocs[alloc_id].images;
    const auto alloc_image_it = std::ranges::find(alloc_images, image_id);
    if (alloc_image_it == alloc_images.end()) {
        ASSERT_MSG(false, "Trying to delete an image that does not exist");
        return;
    }
    ASSERT_MSG(False(image.flags & ImageFlagBits::Tracked), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Mark render targets as dirty
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    const std::span<const ImageViewId> image_view_ids = image.image_view_ids;
    for (const ImageViewId image_view_id : image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);

    for (const AliasedImage& alias : image.aliased_images) {
        ImageBase& other_image = slot_images[alias.id];
        [[maybe_unused]] const size_t num_removed_aliases =
            std::erase_if(other_image.aliased_images, [image_id](const AliasedImage& other_alias) {
                return other_alias.id == image_id;
            });
        other_image.CheckAliasState();
        ASSERT_MSG(num_removed_aliases == 1, "Invalid number of removed aliases: {}",
                   num_removed_aliases);
    }
    DropOverlapLinks(image_id);
    for (const ImageViewId image_view_id : image_view_ids) {
        if (!immediate_delete) {
            sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        }
        slot_image_views.erase(image_view_id);
        BumpImageViewGeneration(image_view_id);
        Churn::Add(Churn::ImageViewsDestroyed);
    }
    u64 delete_tick = 0;
    if constexpr (requires { runtime.CurrentTick(); }) {
        delete_tick = runtime.CurrentTick();
    }
    std::shared_ptr<Image> retained =
        RetainPendingImageDownloads(image_id, slot_images[image_id], uncommitted_downloads,
                                    committed_downloads, delete_tick);
    if (retained) {
        LOG_INFO(Render_Vulkan, "Retained image for queued download: id={}", image_id.Value());
        // Fritchii's free guard: the download's landing must not be what frees it while the
        // GPU still runs work recorded before this delete (TickFrame lets go at IsFree).
        if constexpr (requires { runtime.CurrentTick(); }) {
            retained_image_keepers.push_back({delete_tick, std::move(retained)});
            ++free_guard.retained_held;
        }
    } else if (!immediate_delete) {
        if constexpr (requires { runtime.CurrentTick(); }) {
            gpu_sentenced_images.push_back(
                {runtime.CurrentTick(), freed_bytes, std::move(slot_images[image_id])});
            gpu_sentenced_bytes += freed_bytes;
        } else {
            sentenced_images.Push(std::move(slot_images[image_id]));
            sentenced_image_bytes[sentenced_image_index] += freed_bytes;
        }
    }
    slot_images.erase(image_id);

    alloc_images.erase(alloc_image_it);
    if (alloc_images.empty()) {
        image_allocs_table.erase(alloc_it);
        slot_image_allocs.erase(alloc_id);
    }
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
#ifndef NDEBUG
        if constexpr (ENABLE_VALIDATION) {
            std::ranges::fill(channel_info.graphics_image_table.LiveBoundIds(), CORRUPT_ID);
            std::ranges::fill(channel_info.compute_image_table.LiveBoundIds(), CORRUPT_ID);
        }
#endif
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
}

template <class P>
void TextureCache<P>::DropOverlapLinks(ImageId image_id) {
    ImageBase& image = slot_images[image_id];
    for (const ImageId overlap_id : image.overlapping_images) {
        ImageBase& other_image = slot_images[overlap_id];
        [[maybe_unused]] const size_t num_removed_overlaps = std::erase_if(
            other_image.overlapping_images,
            [image_id](const ImageId other_overlap_id) { return other_overlap_id == image_id; });
        other_image.CheckBadOverlapState();
        ASSERT_MSG(num_removed_overlaps == 1, "Invalid number of removed overlapps: {}",
                   num_removed_overlaps);
    }
}

template <class P>
void TextureCache<P>::RelocateImage(ImageId image_id, GPUVAddr gpu_addr, DAddr cpu_addr) {
    Image& image = slot_images[image_id];
    // 1. The GPU's data at the old address is lost, as when the collector deletes the image.
    marking_relocation = true; // the lost contents counter's attribution
    MarkStale(image); // before UntrackImage (see MarkStale)
    marking_relocation = false;
    if (True(image.flags & ImageFlagBits::Tracked)) {
        UntrackImage(image, image_id);
    }
    UnregisterImage(image_id);
    // RegisterImage below counts it again.
    total_used_memory -= (std::min)(total_used_memory, ImageMemoryBytes(image));
    // 2. No overlap links: the chain of drifted images does not form.
    DropOverlapLinks(image_id);
    image.overlapping_images.clear();
    image.CheckBadOverlapState();
    // 3. Out of the alloc entry of the old address (InsertImage files it under the new one).
    if (const auto alloc_it = image_allocs_table.find(image.gpu_addr);
        alloc_it != image_allocs_table.end()) {
        const ImageAllocId alloc_id = alloc_it->second;
        std::vector<ImageId>& alloc_images = slot_image_allocs[alloc_id].images;
        std::erase(alloc_images, image_id);
        if (alloc_images.empty()) {
            image_allocs_table.erase(alloc_it);
            slot_image_allocs.erase(alloc_id);
        }
    } else {
        ASSERT_MSG(false, "Relocating an image without an alloc at {:#x}", image.gpu_addr);
    }
    // Lookups by address (descriptors, bound render targets) must not keep finding its views
    // under the old address; the views themselves, and the framebuffers keyed by them, stay.
    for (const ImageViewId image_view_id : image.image_view_ids) {
        std::ranges::replace(render_targets.color_buffer_ids, image_view_id, ImageViewId{});
        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = ImageViewId{};
        }
    }
    RemoveImageViewReferences(image.image_view_ids);
    // The views stay, but their descriptors now resolve to other views (the image is no longer
    // at the old address): draw templates (EXP87) keeping them must not hold.
    for (const ImageViewId image_view_id : image.image_view_ids) {
        BumpImageViewGeneration(image_view_id);
    }
    // 4. The new address. Contents count as the guest's (the clear overwrites them whole, and
    // PrepareImage's invalidate path drops the upload, as for a new image).
    image.gpu_addr = gpu_addr;
    image.cpu_addr = cpu_addr;
    image.cpu_addr_end = cpu_addr + image.guest_size_bytes;
    // The cold sweep keep's cached answer was for the old address (ColdKeepResampled), and its
    // share of the keep budget goes back (the contents it was kept for are gone).
    ColdKeepUncharge(image);
    image.cold_keep_gen = 0;
    image.cold_keep = 0;
    image.flags &= ~ImageFlagBits::GpuModified;
    image.flags |= ImageFlagBits::CpuModified;
    image.last_write_frame = ~u64{0};
    image.write_streak = 0;
    image.relocation_guard_noted = 0;
    image.lost_contents = 0;
    // Relocated per-frame targets on small cards keep clear history under EDEN_NX2_ASYNC_RELAXED
    // so asynchronous shader builds don't stall on every camera relocation.
    if (!AsyncRelaxedTargets()) {
        image.last_clear_frame = ~u64{0};
        image.previous_clear_frame = ~u64{0};
    }
    image.drawn_since_clear = false;
    // 5. Page tables, map view, page filter, LRU.
    RegisterImage(image_id);
    // 6. As DeleteImage: render targets and descriptor tables are looked up again.
    ++bindings_epoch;
    auto& dirty = maxwell3d->dirty.flags;
    dirty[Dirty::RenderTargets] = true;
    dirty[Dirty::ZetaBuffer] = true;
    for (size_t rt = 0; rt < NUM_RT; ++rt) {
        dirty[Dirty::ColorBuffer0 + rt] = true;
    }
    for (size_t c : active_channel_ids) {
        auto& channel_info = channel_storage[c];
        channel_info.graphics_image_table.Invalidate();
        channel_info.compute_image_table.Invalidate();
    }
    has_deleted_images = true;
    // EDEN_NX2_FASTDRAW_COUNT (Fable plan D1, row 2): view generations and tables changed.
    VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::Relocate);
}

template <class P>
void TextureCache<P>::RemoveImageViewReferences(std::span<const ImageViewId> removed_views) {
    ++bindings_epoch;
    // Their descriptors resolve to other views now: every texture memo goes (EXP32).
    ++texture_memo_epoch;
    for (const ImageViewId removed : removed_views) {
        const auto refs = image_view_descriptors.find(removed);
        if (refs == image_view_descriptors.end()) {
            continue;
        }
        for (const auto& [channel, descriptor] : refs->second) {
            // The channel may be gone, or its entry point to another view since.
            if (std::ranges::find(active_channel_ids, channel) == active_channel_ids.end()) {
                continue;
            }
            auto& image_views = channel_storage[channel].image_views;
            if (const auto it = image_views.find(descriptor);
                it != image_views.end() && it->second == removed) {
                image_views.erase(it);
            }
        }
        image_view_descriptors.erase(refs);
    }
}

template <class P>
void TextureCache<P>::RemoveFramebuffers(std::span<const ImageViewId> removed_views) {
    runtime.FlushDeferredClear();
    const auto destroy = [this](FramebufferId framebuffer_id) {
        ASSERT(framebuffer_id);
        ++rt_stats.framebuffers_removed;
        if (framebuffer_id == last_framebuffer_id) {
            last_framebuffer_id = {};
            last_framebuffer_serial = 0;
        }
        sentenced_framebuffers.Push(std::move(slot_framebuffers[framebuffer_id]));
        slot_framebuffers.erase(framebuffer_id);
    };
    // A null id matches every unused slot of a key: only the full scan finds those.
    if (std::ranges::any_of(removed_views, [](ImageViewId view) { return !view; })) {
        rt_stats.framebuffers_scanned += static_cast<u32>(framebuffers.size());
        auto it = framebuffers.begin();
        while (it != framebuffers.end()) {
            if (it->first.Contains(removed_views)) {
                const FramebufferId framebuffer_id = it->second;
                UnindexFramebuffer(framebuffer_id, it->first);
                destroy(framebuffer_id);
                it = framebuffers.erase(it);
            } else {
                ++it;
            }
        }
        return;
    }
    // The same framebuffers the scan finds (every key that holds one of the views), by index.
    boost::container::small_vector<FramebufferId, 16> removed;
    for (const ImageViewId view : removed_views) {
        const auto it = framebuffers_by_view.find(view);
        if (it == framebuffers_by_view.end()) {
            continue;
        }
        for (const FramebufferId framebuffer_id : it->second) {
            if (std::ranges::find(removed, framebuffer_id) == removed.end()) {
                removed.push_back(framebuffer_id);
            }
        }
    }
    rt_stats.framebuffers_scanned += static_cast<u32>(removed.size());
    for (const FramebufferId framebuffer_id : removed) {
        const auto key_it = framebuffer_keys.find(framebuffer_id);
        ASSERT(key_it != framebuffer_keys.end());
        const RenderTargets key = key_it->second;
        UnindexFramebuffer(framebuffer_id, key);
        [[maybe_unused]] const size_t erased = framebuffers.erase(key);
        ASSERT(erased == 1);
        destroy(framebuffer_id);
    }
}

template <class P>
void TextureCache<P>::UnindexFramebuffer(FramebufferId framebuffer_id, const RenderTargets& key) {
    const auto unindex_view = [this, framebuffer_id](ImageViewId view) {
        if (!view) {
            return;
        }
        const auto it = framebuffers_by_view.find(view);
        if (it == framebuffers_by_view.end()) {
            return;
        }
        auto& ids = it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), framebuffer_id), ids.end());
        if (ids.empty()) {
            framebuffers_by_view.erase(it);
        }
    };
    std::ranges::for_each(key.color_buffer_ids, unindex_view);
    unindex_view(key.depth_buffer_id);
    framebuffer_keys.erase(framebuffer_id);
}

// EXP100 (fable52_answer_imagereads 2). Holder of the mutex only (BeforeChange breaks the count
// for good otherwise, and logs). A registered, non-sparse image that is or was GPU-modified: its
// pages stay counted until it is unregistered. Clears of the flag are ignored on purpose.
template <class P>
void TextureCache<P>::CountGpuModified(ImageBase& image) noexcept {
    if (image.nx2_gpu_counted || False(image.flags & ImageFlagBits::Registered) ||
        True(image.flags & ImageFlagBits::Sparse) || image.nx2_counted_size == 0) {
        return;
    }
    image_seq.BeforeChange();
    gpu_image_filter.Add(image.nx2_counted_addr, image.nx2_counted_size);
    image.nx2_gpu_counted = true;
}

template <class P>
bool TextureCache<P>::EarlyImageWritePending(ImageViewId id) noexcept {
    if constexpr (requires { runtime.IsFree(u64{}); }) {
        const ImageViewBase& view = slot_image_views[id];
        if (!view.image_id) {
            return false;
        }
        const ImageBase& image = slot_images[view.image_id];
        return image.early_write_tick != 0 && !runtime.IsFree(image.early_write_tick);
    }
    return false;
}

template <class P>
auto TextureCache<P>::ComputeSkipImageState(DAddr addr, u64 size) -> SkipImageState {
    const u64 render_frame = RenderFrame();
    const DAddr end = addr + size;
    bool covered = false;
    SkipImageState state = SkipImageState::Redrawn;
    ForEachImageInRegion(addr, size, [&](ImageId, ImageBase& image) {
        if (state != SkipImageState::Redrawn) {
            return;
        }
        if (False(image.flags & ImageFlagBits::GpuModified) ||
            True(image.flags & ImageFlagBits::CpuModified)) {
            state = SkipImageState::NoGpuData;
            return;
        }
        // Written in this frame and the 7 before, or in the last frame and the 7 before that.
        const bool streak_now =
            IsRedrawnTarget(image.last_write_frame, image.write_streak, render_frame);
        const bool streak_before = render_frame != 0 &&
                                   IsRedrawnTarget(image.last_write_frame, image.write_streak,
                                                   render_frame - 1);
        if (!streak_now && !streak_before) {
            state = SkipImageState::NotRedrawn;
            return;
        }
        covered = covered || (image.cpu_addr <= addr && end <= image.cpu_addr_end);
    });
    if (state == SkipImageState::Redrawn && !covered) {
        return SkipImageState::NoImage;
    }
    return state;
}

template <class P>
void TextureCache<P>::MarkModification(ImageBase& image) noexcept {
    // FRITCHII_ASYNC_COMPUTE_SKIP: a GPU write over the image (the whole of it counts).
    if (image.cpu_addr_end > image.cpu_addr) {
        VideoCommon::ComputeSkipHook::NoteWrite(image.cpu_addr,
                                                image.cpu_addr_end - image.cpu_addr);
    }
    image.flags |= ImageFlagBits::GpuModified;
    CountGpuModified(image); // EXP100
    // EXP76: GPU-written texels no longer match a hash of guest bytes.
    image.nx2_content_hash_lo = 0;
    image.nx2_content_hash_hi = 0;
    image.modification_tick = ++modification_tick;
    if constexpr (requires { runtime.CurrentTick(); }) {
        if (VideoCommon::EarlyCompute::g_early) [[unlikely]] {
            image.early_write_tick = runtime.CurrentTick();
        }
    }
    const u64 render_frame = RenderFrame();
    image.write_streak = NextWriteStreak(image.last_write_frame, image.write_streak, render_frame);
    image.last_write_frame = render_frame;
    if (!clearing_render_targets) {
        image.drawn_since_clear = true;
    }
}

template <class P>
u32 TextureCache<P>::RetireSlotPredecessors(const ImageInfo& new_info, GPUVAddr gpu_addr) {
    const DeleteReasonScope delete_reason_scope{*this, DeleteReason::SlotRetire};
    if (!RelocationShaped(new_info)) {
        return 0;
    }
    // Depth targets only (the per-NPC ones): in the owner's 2026-11-11 playtest the wider rule
    // retired colour targets (480x270 format 21, 17 times) and the 2560x1440 scene depth that
    // were wanted again in the same frame. Same-size predecessors are left to EXP43 and the sweep.
    const auto type = VideoCore::Surface::GetFormatType(new_info.format);
    if (type != VideoCore::Surface::SurfaceType::Depth &&
        type != VideoCore::Surface::SurfaceType::Stencil &&
        type != VideoCore::Surface::SurfaceType::DepthStencil) {
        return 0;
    }
    // Not Alias (the per-NPC pile is all Alias: 2,000-3,200 kept a 600 frames for it): the cold
    // sweep deletes those too; live alias links are refused below (aliased_images).
    constexpr ImageFlagBits excluded =
        ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
        ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked | ImageFlagBits::Converted |
        ImageFlagBits::AsynchronousDecode | ImageFlagBits::IsDecoding | ImageFlagBits::Remapped |
        ImageFlagBits::CostlyLoad | ImageFlagBits::Rescaled |
        ImageFlagBits::TrimmedLevels;
    const auto queued = [](ImageId image_id, const auto& downloads) {
        return std::ranges::any_of(downloads, [image_id](const auto& download) {
            return download.is_swizzle && download.object_id == image_id;
        });
    };
    const auto bound = [this](const ImageBase& image) {
        return std::ranges::any_of(image.image_view_ids, [this](ImageViewId view) {
            return render_targets.depth_buffer_id == view ||
                   std::ranges::find(render_targets.color_buffer_ids, view) !=
                       render_targets.color_buffer_ids.end();
        });
    };
    const u64 render_frame = RenderFrame();
    const u64 min_age = static_cast<u64>((std::max)(1, VideoCore::Nx2Debug::Experiment(64)));
    bool seen = false;
    boost::container::small_vector<ImageId, 8> retire;
    for (const ImageId overlap_id : join_bad_overlap_ids) {
        const Image& old = slot_images[overlap_id];
        // The same slot, reused for another layout of the same format.
        if (old.gpu_addr != gpu_addr || old.info.format != new_info.format ||
            !RelocationShaped(old.info) ||
            (old.info.size.width == new_info.size.width &&
             old.info.size.height == new_info.size.height)) {
            continue;
        }
        seen = true;
        if (True(old.flags & excluded)) {
            ++slot_kept_kind;
            const auto flag_bits = static_cast<u32>(old.flags & excluded);
            for (u32 bit = 0; bit < 32; ++bit) {
                if ((flag_bits >> bit) & 1) {
                    ++slot_kept_flag[bit];
                }
            }
            continue;
        }
        // Not GPU-written: its contents are guest memory's, nothing is lost. Alias links: the
        // cold sweep deletes such images too, and DeleteImage drops the links (2026-11-11 night
        // market: 1,700-3,000 of the pile a 600 frames had them, ~500 were never GPU-written).
        if (False(old.flags & ImageFlagBits::Registered)) {
            ++slot_kept_kind;
            ++slot_kept_not_gpu;
            continue;
        }
        if (old.HasScaled()) {
            ++slot_kept_kind;
            ++slot_kept_aliased;
            continue;
        }
        if (old.last_write_frame == render_frame) {
            ++slot_kept_same_frame; // may still be read in this frame
            continue;
        }
        // EXP64=N: only images nothing touched (bound, sampled, drawn) for N host frames. With
        // 1, 29 retired targets were made again 1-4 frames later (drawn into without a clear:
        // the game wanted their depth), 2026-11-11 night market.
        if (frame_tick - lru_cache.Tick(old.lru_index) < min_age) {
            ++slot_kept_same_frame;
            continue;
        }
        const auto shape = relocation_shapes.find(RelocationShapeKey(old.info));
        const u8 guard = shape != relocation_shapes.end() ? shape->second.guard : 0;
        constexpr u8 read_later = GuardSampled | GuardCpuRead;
        if (old.info.forced_flushed || old.info.dma_downloaded ||
            ((guard | old.relocation_guard_noted) & read_later) != 0) {
            ++slot_kept_guard;
            continue;
        }
        if (bound(old) || queued(overlap_id, uncommitted_downloads) ||
            std::ranges::any_of(committed_downloads, [&](const auto& downloads) {
                return queued(overlap_id, downloads);
            })) {
            ++slot_kept_busy;
            continue;
        }
        retire.push_back(overlap_id);
    }
    slot_clears += seen ? 1 : 0;
    // FRITCHII_LOST_WRITEBACK (default 1): written back first, deleted when that lands; until then
    // it stays a bad overlap of the new image (so it leaves the retire list), as one the queue
    // has no room for.
    retire.erase(std::remove_if(retire.begin(), retire.end(),
                                [this](ImageId old_id) {
                                    return QueueLostWriteback(old_id, slot_images[old_id],
                                                              LostPathSlot) != LostWriteback::Drop;
                                }),
                 retire.end());
    for (const ImageId old_id : retire) {
        // As EXP43=1 and the cold sweep delete one (no download: the contents have no reader).
        Image& old = slot_images[old_id];
        MarkStale(old); // before UntrackImage (see MarkStale)
        if (True(old.flags & ImageFlagBits::Tracked)) {
            UntrackImage(old, old_id);
        }
        UnregisterImage(old_id);
        slot_recent[slot_recent_next] = SlotRetired{old.gpu_addr, RelocationShapeKey(old.info),
                                                    render_frame};
        slot_recent_next = (slot_recent_next + 1) % slot_recent.size();
        DeleteImage(old_id);
        ++rt_stats.retired_images;
    }
    if (!retire.empty()) {
        join_bad_overlap_ids.erase(std::remove_if(join_bad_overlap_ids.begin(),
                                                  join_bad_overlap_ids.end(),
                                                  [&retire](ImageId id) {
                                                      return std::ranges::find(retire, id) !=
                                                             retire.end();
                                                  }),
                                   join_bad_overlap_ids.end());
    }
    slot_retired += retire.size();
    slot_retired_any = slot_retired_any || !retire.empty();
    return static_cast<u32>(retire.size());
}

template <class P>
ImageId TextureCache<P>::RelocationCandidate(const ImageInfo& new_info, GPUVAddr gpu_addr,
                                             DAddr cpu_addr) {
    if (!RelocationShaped(new_info)) {
        return ImageId{};
    }
    // Exactly one bad overlap of the same shape and layout; others (a neighbouring target of
    // another shape the drift moved into) stay bad overlaps, as for a new image.
    const auto same_shape = [&new_info](const ImageInfo& info) {
        return RelocationShaped(info) && info.format == new_info.format &&
               info.size.width == new_info.size.width &&
               info.size.height == new_info.size.height &&
               info.size.depth == new_info.size.depth && info.block == new_info.block &&
               info.layer_stride == new_info.layer_stride &&
               info.maybe_unaligned_layer_stride == new_info.maybe_unaligned_layer_stride &&
               info.tile_width_spacing == new_info.tile_width_spacing &&
               info.native_tiling == new_info.native_tiling &&
               info.rescaleable == new_info.rescaleable &&
               info.downscaleable == new_info.downscaleable &&
               info.is_sparse == new_info.is_sparse && info.astc_opaque == new_info.astc_opaque;
    };
    // Predecessors: images of the shape the GPU did not write in this frame (one it did may
    // still be read in this frame; it is never taken, and not counted).
    ImageId old_id{};
    u32 predecessors = 0;
    u32 same_frame = 0;
    const u64 render_frame = RenderFrame();
    for (const ImageId overlap_id : join_bad_overlap_ids) {
        const ImageBase& overlap = slot_images[overlap_id];
        if (!same_shape(overlap.info)) {
            continue;
        }
        if (overlap.last_write_frame == render_frame) {
            ++same_frame;
            continue;
        }
        ++predecessors;
        // EXP43=3 takes the most recently modified of several.
        if (!old_id || overlap.modification_tick > slot_images[old_id].modification_tick) {
            old_id = overlap_id;
        }
    }
    if (predecessors == 0) {
        if (same_frame != 0) {
            // Refused as written this frame (the render-frame clock line counts them by shape).
            ++relocation_same_frame;
            ++relocation_shapes[RelocationShapeKey(new_info)].same_frame_refused;
            relocation_seen = true;
        }
        return ImageId{};
    }
    if (predecessors > 1) {
        // A chain from before the switch (or two targets of the shape in reach): EXP43=1 and 2
        // need exactly one.
        ++relocation_crowded;
        relocation_seen = true;
        if (VideoCore::Nx2Debug::Experiment(43) < 3) {
            return ImageId{};
        }
    }
    const ImageBase& old = slot_images[old_id];
    // Read by the CPU.
    if (old.info.forced_flushed || old.info.dma_downloaded) {
        return ImageId{};
    }
    constexpr ImageFlagBits excluded =
        ImageFlagBits::Sparse | ImageFlagBits::SparseLayers | ImageFlagBits::SparseTiles |
        ImageFlagBits::UnwrittenSparse | ImageFlagBits::Unbacked | ImageFlagBits::Converted |
        ImageFlagBits::AsynchronousDecode | ImageFlagBits::IsDecoding | ImageFlagBits::Remapped |
        ImageFlagBits::Alias | ImageFlagBits::CostlyLoad | ImageFlagBits::Rescaled |
        ImageFlagBits::TrimmedLevels;
    if (True(old.flags & excluded) || False(old.flags & ImageFlagBits::Registered) ||
        old.HasScaled() || !old.aliased_images.empty()) {
        return ImageId{};
    }
    // A drift, not a reuse of distant memory; real CPU addresses on both sides (no stand-ins).
    const u64 distance =
        gpu_addr > old.gpu_addr ? gpu_addr - old.gpu_addr : old.gpu_addr - gpu_addr;
    constexpr DAddr StandInBase = ~(1ULL << 40);
    if (distance == 0 || distance >= old.guest_size_bytes || cpu_addr >= StandInBase ||
        old.cpu_addr >= StandInBase) {
        return ImageId{};
    }
    if (new_info.is_sparse && !gpu_memory->IsContinuousRange(gpu_addr, old.guest_size_bytes)) {
        return ImageId{};
    }
    // Registered in this channel's GPU page table (UnregisterImage clears that one).
    const auto& gpu_pages = *channel_state->gpu_page_table;
    const auto page = gpu_pages.find(old.gpu_addr >> YUZU_PAGEBITS);
    if (page == gpu_pages.end() || std::ranges::find(page->second, old_id) == page->second.end()) {
        return ImageId{};
    }
    // No download queued for it.
    const auto queued = [old_id](const auto& downloads) {
        return std::ranges::any_of(downloads, [old_id](const auto& download) {
            return download.is_swizzle && download.object_id == old_id;
        });
    };
    if (queued(uncommitted_downloads) || std::ranges::any_of(committed_downloads, queued)) {
        return ImageId{};
    }
    // Not bound as another target of this clear.
    const auto& regs = maxwell3d->regs;
    for (u32 index = 0; index < regs.rt_control.count && index < NUM_RT; ++index) {
        const auto& rt = regs.rt[index];
        const GPUVAddr address = rt.Address();
        if (address != 0 && address != gpu_addr && rt.format != Tegra::RenderTargetFormat::NONE &&
            old.OverlapsGPU(address, 1)) {
            return ImageId{};
        }
    }
    if (regs.zeta_enable && regs.zeta.Address() != 0 && regs.zeta.Address() != gpu_addr &&
        old.OverlapsGPU(regs.zeta.Address(), 1)) {
        return ImageId{};
    }
    return old_id;
}

template <class P>
void TextureCache<P>::NoteRelocationGuard(ImageBase& image, RelocationGuard reason) {
    image.relocation_guard_noted |= reason;
    relocation_guarded_addresses[image.gpu_addr] |= reason;
    RelocationShape& shape = relocation_shapes[RelocationShapeKey(image.info)];
    if ((shape.guard & reason) != 0) {
        return;
    }
    shape.guard |= reason;
    // Each shape's first trip of each reason (a handful per session).
    LOG_INFO(HW_GPU,
             "Render target relocation guard (EXP43): format {} {}x{} now never relocates: {} "
             "(image at {:#x}, written in frame {}, frame {})",
             static_cast<u32>(image.info.format), image.info.size.width, image.info.size.height,
             reason == GuardSampled  ? "sampled in a later frame"
             : reason == GuardLoaded ? "rendered to without a clear in a later frame"
                                     : "read by the CPU",
             image.gpu_addr,
             image.last_write_frame == ~u64{0} ? s64{-1} : static_cast<s64>(image.last_write_frame),
             RenderFrame());
}

template <class P>
void TextureCache<P>::AddRelocationTombstone(const ImageBase& old_image, GPUVAddr moved_to) {
    relocation_tombstones.push_back(RelocationTombstone{
        .gpu_addr = old_image.gpu_addr,
        .cpu_addr = old_image.cpu_addr,
        .size = old_image.guest_size_bytes,
        .shape_key = RelocationShapeKey(old_image.info),
        .render_frame = RenderFrame(),
        .moved_to = moved_to,
        .gpu_modified = True(old_image.flags & ImageFlagBits::GpuModified),
    });
    tombstone_cpu_begin = (std::min)(tombstone_cpu_begin, old_image.cpu_addr);
    tombstone_cpu_end = (std::max)(tombstone_cpu_end, old_image.cpu_addr + old_image.guest_size_bytes);
    ++tombstones_made;
}

template <class P>
void TextureCache<P>::CheckTombstoneCreation(ImageId image_id, const ImageBase& image,
                                             bool overwritten) {
    for (const RelocationTombstone& tombstone : relocation_tombstones) {
        if (image.gpu_addr == tombstone.gpu_addr) {
            // A lookup that, without the relocation, would have found the old image. A clear
            // that overwrites it whole reads nothing; anything else is settled by its first use.
            if (!overwritten) {
                tombstone_born.push_back(TombstoneBorn{
                    .image_id = image_id, .gpu_addr = image.gpu_addr, .tombstone = tombstone});
            }
            return;
        }
    }
    if (!overwritten && image.cpu_addr < tombstone_cpu_end &&
        tombstone_cpu_begin < image.cpu_addr_end) {
        for (const RelocationTombstone& tombstone : relocation_tombstones) {
            if (image.cpu_addr < tombstone.cpu_addr + tombstone.size &&
                tombstone.cpu_addr < image.cpu_addr_end) {
                // Not at the old address: it read guest memory with or without the relocation
                // (a bad overlap gets no copies). Counted only.
                ++tombstone_overlap_reads;
                return;
            }
        }
    }
}

template <class P>
void TextureCache<P>::CheckTombstonePrepare(ImageId image_id, const ImageBase& image,
                                            bool is_modification, bool invalidate) {
    const auto it = std::ranges::find_if(tombstone_born, [&](const TombstoneBorn& born) {
        return born.image_id == image_id && born.gpu_addr == image.gpu_addr;
    });
    if (it == tombstone_born.end()) {
        return;
    }
    const TombstoneBorn born = *it;
    tombstone_born.erase(it);
    if (invalidate) {
        return; // overwritten whole: nothing read
    }
    NoteTombstoneHit(born.tombstone, is_modification ? TombstoneLoaded : TombstoneSampled,
                     image.gpu_addr, &image.info);
}

template <class P>
void TextureCache<P>::CheckTombstoneCpuRead(DAddr cpu_addr, u64 size) {
    if (cpu_addr >= tombstone_cpu_end || cpu_addr + size <= tombstone_cpu_begin) {
        return;
    }
    for (const RelocationTombstone& tombstone : relocation_tombstones) {
        if (tombstone.gpu_modified && cpu_addr < tombstone.cpu_addr + tombstone.size &&
            tombstone.cpu_addr < cpu_addr + size) {
            NoteTombstoneHit(tombstone, TombstoneCpuRead, cpu_addr, nullptr);
            return;
        }
    }
}

template <class P>
void TextureCache<P>::NoteTombstoneHit(const RelocationTombstone& tombstone, TombstoneRead reason,
                                       GPUVAddr address, const ImageInfo* info) {
    ++tombstone_hits[reason];
    if (tombstone_hits_logged >= 8) {
        return;
    }
    ++tombstone_hits_logged;
    const u64 key = tombstone.shape_key;
    LOG_WARNING(HW_GPU,
                "Relocation tombstone hit (EXP43=4): {} at {:#x}{}; the image at {:#x} (format {} "
                "{}x{}, {}GPU-written) moved to {:#x} {} render frame(s) ago",
                reason == TombstoneSampled  ? "sampled"
                : reason == TombstoneLoaded ? "rendered to without a clear"
                                            : "read by the CPU",
                address,
                info != nullptr ? fmt::format(" (format {} {}x{})", static_cast<u32>(info->format),
                                              info->size.width, info->size.height)
                                : std::string{},
                tombstone.gpu_addr, key >> 56, (key >> 18) & 0x3FFFF, key & 0x3FFFF,
                tombstone.gpu_modified ? "" : "not ", tombstone.moved_to,
                RenderFrame() - tombstone.render_frame);
}

template <class P>
void TextureCache<P>::TickRelocationTombstones() {
    if (!relocation_tombstones.empty()) {
        const u64 render_frame = RenderFrame();
        const auto expired = [render_frame](const RelocationTombstone& tombstone) {
            return render_frame > tombstone.render_frame + 2;
        };
        std::erase_if(relocation_tombstones, expired);
        tombstone_born.erase(std::remove_if(tombstone_born.begin(), tombstone_born.end(),
                                            [&](const TombstoneBorn& born) {
                                                return expired(born.tombstone);
                                            }),
                             tombstone_born.end());
        tombstone_cpu_begin = ~DAddr{0};
        tombstone_cpu_end = 0;
        for (const RelocationTombstone& tombstone : relocation_tombstones) {
            tombstone_cpu_begin = (std::min)(tombstone_cpu_begin, tombstone.cpu_addr);
            tombstone_cpu_end = (std::max)(tombstone_cpu_end, tombstone.cpu_addr + tombstone.size);
        }
    }
    if (frame_tick % 600 != 0 || (tombstones_made == 0 && VideoCore::Nx2Debug::Experiment(43) != 4)) {
        return;
    }
    const u64 hits = tombstone_hits[TombstoneSampled] + tombstone_hits[TombstoneLoaded] +
                     tombstone_hits[TombstoneCpuRead];
    LOG_INFO(HW_GPU,
             "Relocation tombstones (EXP43=4) last 600 frames: {} hits ({} sampled, {} rendered to "
             "without a clear, {} CPU read); {} tombstones made, {} live; {} new images read over "
             "one, not at its address",
             hits, tombstone_hits[TombstoneSampled], tombstone_hits[TombstoneLoaded],
             tombstone_hits[TombstoneCpuRead], tombstones_made, relocation_tombstones.size(),
             tombstone_overlap_reads);
    tombstone_hits = {};
    tombstones_made = 0;
    tombstone_overlap_reads = 0;
}

template <class P>
void TextureCache<P>::SynchronizeAliases(ImageId image_id) {
    boost::container::small_vector<const AliasedImage*, 8> aliased_images;
    Image& image = slot_images[image_id];
    bool any_rescaled = True(image.flags & ImageFlagBits::Rescaled);
    bool any_modified = True(image.flags & ImageFlagBits::GpuModified);
    u64 most_recent_tick = image.modification_tick;
    for (const AliasedImage& aliased : image.aliased_images) {
        ImageBase& aliased_image = slot_images[aliased.id];
        if (image.modification_tick < aliased_image.modification_tick) {
            most_recent_tick = (std::max)(most_recent_tick, aliased_image.modification_tick);
            aliased_images.push_back(&aliased);
            any_rescaled |= True(aliased_image.flags & ImageFlagBits::Rescaled);
            any_modified |= True(aliased_image.flags & ImageFlagBits::GpuModified);
        }
    }
    if (aliased_images.empty()) {
        return;
    }
    const bool can_rescale = ImageCanRescale(image);
    if (any_rescaled) {
        if (can_rescale) {
            ScaleUp(image);
        } else {
            ScaleDown(image);
        }
    }
    image.modification_tick = most_recent_tick;
    if (any_modified) {
        image.flags |= ImageFlagBits::GpuModified;
        CountGpuModified(image); // EXP100
        image.nx2_content_hash_lo = 0; // EXP76: aliases copied GPU-written texels in
        image.nx2_content_hash_hi = 0;
    }
    std::ranges::sort(aliased_images, [this](const AliasedImage* lhs, const AliasedImage* rhs) {
        const ImageBase& lhs_image = slot_images[lhs->id];
        const ImageBase& rhs_image = slot_images[rhs->id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });
    const auto& resolution = Settings::values.resolution_info;
    for (const AliasedImage* const aliased : aliased_images) {
        if (!resolution.active || !any_rescaled) {
            CopyImage(image_id, aliased->id, aliased->copies);
            continue;
        }
        Image& aliased_image = slot_images[aliased->id];
        if (!can_rescale) {
            ScaleDown(aliased_image);
            CopyImage(image_id, aliased->id, aliased->copies);
            continue;
        }
        ScaleUp(aliased_image);
        CopyImage(image_id, aliased->id, aliased->copies);
    }
}

template <class P>
void TextureCache<P>::PrepareImage(ImageId image_id, bool is_modification, bool invalidate) {
    Image& image = slot_images[image_id];
    CheckCrossFrameUse(image, is_modification, invalidate);
    if (!tombstone_born.empty()) {
        CheckTombstonePrepare(image_id, image, is_modification, invalidate); // EXP43=4
    }
    if (invalidate) {
        image.flags &= ~(ImageFlagBits::CpuModified | ImageFlagBits::GpuModified);
        if (False(image.flags & ImageFlagBits::Tracked)) {
            TrackImage(image, image_id);
        }
    } else {
        RefreshContents(image, image_id);
        if (!image.aliased_images.empty()) {
            SynchronizeAliases(image_id);
        }
    }
    if (image.lost_contents != 0) [[unlikely]] {
        NoteLostContents(image, is_modification, invalidate);
    }
    if (is_modification) {
        if constexpr (requires { &Image::BackTilesWhole; }) {
            if (True(image.flags & ImageFlagBits::SparseTiles) && image.SparseTilesShareZeroBlock()) {
                // The render would write the zeroed block all empty blocks share: each gets memory
                // of its own first, and the parts that hold data are loaded again.
                LOG_INFO(HW_GPU, "GPU write into sparse tiled image {:#x}; backing it whole",
                         image.gpu_addr);
                if (BackSparseTilesWhole(image)) {
                    image.dirty_ranges.clear();
                    RefreshSparseTiles(image);
                } else {
                    LOG_ERROR(HW_GPU,
                              "Sparse tiled image {:#x}: out of memory backing it whole, the GPU "
                              "write lands in the shared zeroed block",
                              image.gpu_addr);
                }
            }
        }
        if constexpr (requires { &Image::BindLayers; }) {
            if (True(image.flags & ImageFlagBits::SparseLayers)) {
                std::vector<u32> layers(static_cast<size_t>(image.info.resources.layers));
                std::iota(layers.begin(), layers.end(), 0U);
                // Layers over the image's memory cap stay unbacked (logged once).
                BindSparseLayers(image, layers);
            }
        }
        if (True(image.flags & ImageFlagBits::UnwrittenSparse) &&
            False(image.flags & ImageFlagBits::Remapped)) {
            // The GPU renders into it: these writes are lost, later uses get a backed image.
            LOG_WARNING(HW_GPU, "GPU write into unwritten sparse image {:#x}; replacing it",
                        image.gpu_addr);
            Tegra::Nx2SparseMarkWritten(image.gpu_addr, image.guest_size_bytes);
            image.flags |= ImageFlagBits::CpuModified | ImageFlagBits::Remapped;
            ++bindings_epoch;
            VideoCommon::Nx2FastDrawCount::Break(VideoCommon::Nx2FastDrawCount::Reason::Invalidation);
        }
        MarkModification(image);
    }
    TouchImageLru(image);
}

template <class P>
void TextureCache<P>::TouchImageLru(ImageBase& image) noexcept {
    // EDEN_NX2_EXP90 (Fable, fable52_answer_after88 2): decide from the image's own field, which
    // PrepareImage has just read, instead of the LRU node (a cache miss) on every use. The node
    // already ignores a second touch with the same tick (LeastRecentlyUsedCache::Touch), so the
    // LRU order and ticks stay exactly what they were.
    if (VideoCore::Nx2Debug::Experiment(90) != 0) {
        if (image.lru_touched_tick == frame_tick) {
            ++nx2_lru_touches_skipped;
            return;
        }
        image.lru_touched_tick = frame_tick;
        ++nx2_lru_touches;
    }
    lru_cache.Touch(image.lru_index, frame_tick);
}

template <class P>
void TextureCache<P>::PrepareImageView(ImageViewId image_view_id, bool is_modification,
                                       bool invalidate) {
    if (!image_view_id) {
        return;
    }
    const ImageViewBase& image_view = slot_image_views[image_view_id];
    if (image_view.IsBuffer()) {
        return;
    }
    PrepareImage(image_view.image_id, is_modification, invalidate);
}

template <class P>
void TextureCache<P>::CopyImage(ImageId dst_id, ImageId src_id, std::vector<ImageCopy> copies) {
    // As in BlitImage: land a parked deferred clear before the copy overwrites the image.
    runtime.FlushDeferredClear();
    Image& dst = slot_images[dst_id];
    Image& src = slot_images[src_id];
    if (True(dst.flags & ImageFlagBits::UnwrittenSparse) ||
        True(src.flags & ImageFlagBits::UnwrittenSparse)) {
        return;
    }
    if (dst.info.astc_opaque != src.info.astc_opaque) {
        // BC1 and BC3 blocks differ in size, so the host can't copy between them; both hold the
        // same guest data, which is uploaded from guest memory instead.
        return;
    }
    const bool is_rescaled = True(src.flags & ImageFlagBits::Rescaled);
    if (is_rescaled) {
        ASSERT(True(dst.flags & ImageFlagBits::Rescaled));
        const bool both_2d{src.info.type == ImageType::e2D && dst.info.type == ImageType::e2D};
        const auto& resolution = Settings::values.resolution_info;
        for (auto& copy : copies) {
            copy.src_offset.x = resolution.ScaleUp(copy.src_offset.x);
            copy.dst_offset.x = resolution.ScaleUp(copy.dst_offset.x);
            copy.extent.width = resolution.ScaleUp(copy.extent.width);
            if (both_2d) {
                copy.src_offset.y = resolution.ScaleUp(copy.src_offset.y);
                copy.dst_offset.y = resolution.ScaleUp(copy.dst_offset.y);
                copy.extent.height = resolution.ScaleUp(copy.extent.height);
            }
        }
    }
    const auto dst_format_type = GetFormatType(dst.info.format);
    const auto src_format_type = GetFormatType(src.info.format);
    if (src_format_type == dst_format_type) {
        if constexpr (HAS_EMULATED_COPIES) {
            if (!runtime.CanImageBeCopied(dst, src)) {
                return runtime.EmulateCopyImage(dst, src, copies);
            }
        }
        return runtime.CopyImage(dst, src, copies);
    }
    UNIMPLEMENTED_IF(dst.info.type != ImageType::e2D);
    UNIMPLEMENTED_IF(src.info.type != ImageType::e2D);
    if (runtime.ShouldReinterpret(dst, src)) {
        return runtime.ReinterpretImage(dst, src, copies);
    }
    for (const ImageCopy& copy : MakeImageCopyLayers(copies)) {
        UNIMPLEMENTED_IF(copy.dst_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_offset != Offset3D{});
        UNIMPLEMENTED_IF(copy.dst_offset != Offset3D{});

        const SubresourceBase dst_base{
            .level = copy.dst_subresource.base_level,
            .layer = copy.dst_subresource.base_layer,
        };
        const SubresourceBase src_base{
            .level = copy.src_subresource.base_level,
            .layer = copy.src_subresource.base_layer,
        };
        const SubresourceExtent dst_extent{.levels = 1, .layers = 1};
        const SubresourceExtent src_extent{.levels = 1, .layers = 1};
        const SubresourceRange dst_range{.base = dst_base, .extent = dst_extent};
        const SubresourceRange src_range{.base = src_base, .extent = src_extent};
        PixelFormat dst_format = GetAliasConversionFormat(dst.info.format, src.info.format);
        const PixelFormat src_format = GetAliasConversionFormat(src.info.format, dst.info.format);
        if (GetFormatType(src.info.format) == SurfaceType::DepthStencil &&
            GetFormatType(dst_format) == SurfaceType::ColorTexture &&
            BytesPerBlock(dst_format) == 4) {
            dst_format = PixelFormat::A8B8G8R8_UNORM;
        }
        const ImageViewInfo dst_view_info(ImageViewType::e2D, dst_format, dst_range);
        const ImageViewInfo src_view_info(ImageViewType::e2D, src_format, src_range);
        const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
        Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
        const ImageViewId src_view_id = FindOrEmplaceImageView(src_id, src_view_info);
        ImageView& dst_view = slot_image_views[dst_view_id];
        ImageView& src_view = slot_image_views[src_view_id];
        [[maybe_unused]] const Extent3D expected_size{
            .width = (std::min)(dst_view.size.width, src_view.size.width),
            .height = (std::min)(dst_view.size.height, src_view.size.height),
            .depth = (std::min)(dst_view.size.depth, src_view.size.depth),
        };
        const Extent3D scaled_extent = [is_rescaled, expected_size]() {
            if (!is_rescaled) {
                return expected_size;
            }
            const auto& resolution = Settings::values.resolution_info;
            return Extent3D{
                .width = resolution.ScaleUp(expected_size.width),
                .height = resolution.ScaleUp(expected_size.height),
                .depth = expected_size.depth,
            };
        }();
        UNIMPLEMENTED_IF(copy.extent != scaled_extent);

        runtime.ConvertImage(dst_framebuffer, dst_view, src_view);
    }
}

template <class P>
void TextureCache<P>::BindRenderTarget(ImageViewId* old_id, ImageViewId new_id) {
    if (*old_id == new_id) {
        return;
    }
    if (new_id) {
        const ImageViewBase& old_view = slot_image_views[new_id];
        if (True(old_view.flags & ImageViewFlagBits::PreemtiveDownload) &&
            False(slot_images[old_view.image_id].flags & (ImageFlagBits::Unbacked | ImageFlagBits::TrimmedLevels))) {
            const PendingDownload new_download{true, 0, old_view.image_id};
            async_flush_work_noted.store(true, std::memory_order_relaxed);
            uncommitted_downloads.emplace_back(new_download);
        }
    }
    *old_id = new_id;
}

template <class P>
std::pair<FramebufferId, ImageViewId> TextureCache<P>::RenderTargetFromImage(
    ImageId image_id, const ImageViewInfo& view_info) {
    const ImageViewId view_id = FindOrEmplaceImageView(image_id, view_info);
    const ImageBase& image = slot_images[image_id];
    const bool is_rescaled = True(image.flags & ImageFlagBits::Rescaled);
    const bool is_color = GetFormatType(image.info.format) == SurfaceType::ColorTexture;
    const ImageViewId color_view_id = is_color ? view_id : ImageViewId{};
    const ImageViewId depth_view_id = is_color ? ImageViewId{} : view_id;
    Extent3D extent = MipSize(image.info.size, view_info.range.base.level);
    if (is_rescaled) {
        const auto& resolution = Settings::values.resolution_info;
        extent.width = resolution.ScaleUp(extent.width);
        if (image.info.type == ImageType::e2D) {
            extent.height = resolution.ScaleUp(extent.height);
        }
    }
    const u32 num_samples = image.info.num_samples;
    const auto [samples_x, samples_y] = SamplesLog2(num_samples);
    const FramebufferId framebuffer_id = GetFramebufferId(RenderTargets{
        .color_buffer_ids = {color_view_id},
        .depth_buffer_id = depth_view_id,
        .size = {extent.width >> samples_x, extent.height >> samples_y},
        .is_rescaled = is_rescaled,
    });
    return {framebuffer_id, view_id};
}

template <class P>
bool TextureCache<P>::IsFullClear(ImageViewId id) {
    if (!id) {
        return true;
    }
    const ImageViewBase& image_view = slot_image_views[id];
    const ImageBase& image = slot_images[image_view.image_id];
    const Extent3D size = image_view.size;
    const auto& regs = maxwell3d->regs;
    const auto& scissor = regs.scissor_test[0];
    if (image.info.resources.levels > 1 || image.info.resources.layers > 1) {
        // Images with multiple resources can't be cleared in a single call
        return false;
    }
    if (regs.clear_control.use_scissor == 0) {
        // If scissor testing is disabled, the clear is always full
        return true;
    }
    // Make sure the clear covers all texels in the subresource
    return scissor.min_x == 0 && scissor.min_y == 0 && scissor.max_x >= size.width &&
           scissor.max_y >= size.height;
}

template <class P>
void TextureCache<P>::NoteFullClears() {
    // Not PrepareImage's invalidate path itself: UpdateRenderTargets(true) invalidates every bound
    // target, also those the clear leaves alone (the other colour targets, the depth target of a
    // colour clear), and those must not count as cleared here.
    const auto& clear = maxwell3d->regs.clear_surface;
    const bool color = clear.R || clear.G || clear.B || clear.A;
    const u64 render_frame = RenderFrame();
    const auto note = [&](ImageViewId view_id) {
        if (!view_id || view_id == NULL_IMAGE_VIEW_ID) {
            return;
        }
        const ImageViewBase& view = slot_image_views[view_id];
        if (view.IsBuffer() || !IsFullClear(view_id)) {
            return;
        }
        ImageBase& image = slot_images[view.image_id];
        if (image.last_clear_frame != render_frame) {
            // Only a clear after a write starts a new one: two clears with nothing between are
            // the same clear, wherever the frame's tick fell.
            if (image.drawn_since_clear) {
                image.previous_clear_frame = image.last_clear_frame;
            }
            image.last_clear_frame = render_frame;
        }
        image.drawn_since_clear = false;
    };
    if (color && clear.RT < NUM_RT) {
        note(render_targets.color_buffer_ids[clear.RT]);
    }
    // A stencil-only clear leaves the depth contents.
    if (clear.Z) {
        note(render_targets.depth_buffer_id);
    }
}

template <class P>
DrawTargets TextureCache<P>::ClassifyDrawTargets() {
    std::scoped_lock lock{mutex};
    const auto& regs = maxwell3d->regs;
    // The bound targets are the registers' while their dirty flag is clear (UpdateRenderTargets
    // finds them again when it is set); otherwise the images at the registers' addresses.
    const bool bound_current = !maxwell3d->dirty.flags[Dirty::RenderTargets];
    const u64 render_frame = RenderFrame();
    DrawTargets result{};
    bool any_target{};
    const auto take = [&](ImageViewId bound, const ImageInfo& info, GPUVAddr gpu_addr,
                          VideoCore::Surface::PixelFormat format) {
        any_target = true;
        const ImageBase* image{};
        if (bound_current) {
            if (bound && bound != NULL_IMAGE_VIEW_ID && !slot_image_views[bound].IsBuffer()) {
                image = &slot_images[slot_image_views[bound].image_id];
            }
        } else if (const ImageId image_id = FindImage(info, gpu_addr, RelaxedOptions{});
                   image_id) {
            image = &slot_images[image_id];
        }
        // Not registered yet: the draw is the first use of the target.
        // A render-frame tick between a target's clear and its draws made a per-frame target
        // look persistent (its clear was in the frame before): with EDEN_NX2_ASYNC_RELAXED (on
        // by default on small cards) a clear one render frame back counts as this frame's.
        u64 clear_frame = render_frame;
        if (image && AsyncRelaxedTargets() && image->last_clear_frame + 1 == render_frame) {
            clear_frame = image->last_clear_frame;
        }
        const DrawTargetClass kind =
            image ? ClassifyClearedTarget(image->last_clear_frame, image->previous_clear_frame,
                                          clear_frame)
                  : DrawTargetClass::FirstTime;
        // A target never cleared but written in every recent frame is per-frame too.
        const DrawTargetClass redrawn_kind =
            kind == DrawTargetClass::Persistent && image &&
                    IsRedrawnTarget(image->last_write_frame, image->write_streak, render_frame)
                ? DrawTargetClass::Recurring
                : kind;
        result.kind_if_redrawn = (std::max)(result.kind_if_redrawn, redrawn_kind);
        if (kind > result.kind) {
            result = DrawTargets{
                .kind = kind,
                .kind_if_redrawn = result.kind_if_redrawn,
                .gpu_addr = gpu_addr,
                .width = image ? image->info.size.width : info.size.width,
                .height = image ? image->info.size.height : info.size.height,
                .format = image ? image->info.format : format,
            };
        }
    };
    const u32 color_count = (std::min)(static_cast<u32>(regs.rt_control.count),
                                       static_cast<u32>(NUM_RT));
    for (u32 index = 0; index < color_count; ++index) {
        const auto& rt = regs.rt[index];
        const GPUVAddr gpu_addr = rt.Address();
        if (gpu_addr == 0 || rt.format == Tegra::RenderTargetFormat::NONE) {
            continue;
        }
        const ImageInfo info(rt, regs.anti_alias_samples_mode);
        take(render_targets.color_buffer_ids[index], info, gpu_addr, info.format);
    }
    if (regs.zeta_enable && regs.zeta.Address() != 0) {
        const ImageInfo info(regs.zeta, regs.zeta_size, regs.anti_alias_samples_mode);
        take(render_targets.depth_buffer_id, info, regs.zeta.Address(), info.format);
    }
    if (!any_target) {
        // No colour or depth target: what the draw does stays in storage it writes.
        result.kind = DrawTargetClass::Persistent;
        result.kind_if_redrawn = DrawTargetClass::Persistent;
    }
    return result;
}

template <class P>
void TextureCache<P>::CreateChannel(struct Tegra::Control::ChannelState& channel) {
    VideoCommon::ChannelSetupCaches<TextureCacheChannelInfo>::CreateChannel(channel);
    ++texture_memo_epoch;
    ++bindings_epoch;
    const auto it = channel_map.find(channel.bind_id);
    auto* this_state = &channel_storage[it->second];
    const auto& this_as_ref = address_spaces[channel.memory_manager->GetID()];
    this_state->gpu_page_table = &gpu_page_table_storage[this_as_ref.storage_id * 2];
    this_state->sparse_page_table = &gpu_page_table_storage[this_as_ref.storage_id * 2 + 1];
}

/// Bind a channel for execution.
template <class P>
void TextureCache<P>::OnGPUASRegister([[maybe_unused]] size_t map_id) {
    gpu_page_table_storage.emplace_back();
    gpu_page_table_storage.emplace_back();
}

} // namespace VideoCommon
