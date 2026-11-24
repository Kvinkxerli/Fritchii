// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include "video_core/page_filter.h"
#include "common/container/unordered_map.h"
#include <utility>

#include "common/assert.h"
#include "common/bit_field.h"
#include "common/common_types.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/query_cache/query_base.h"
#include "video_core/query_cache/types.h"

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra {
class GPU;
}

namespace VideoCommon {

struct LookupData {
    VAddr address;
    QueryBase* found_query;
};

template <typename Traits>
class QueryCacheBase : public VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
    using RuntimeType = typename Traits::RuntimeType;

public:
    /// Device pages this cache ever registered objects on (InnerInvalidation skips the rest).
    VideoCommon::PageFilter page_filter;

    union QueryLocation {
        BitField<27, 5, u32> stream_id;
        BitField<0, 27, u32> query_id;
        u32 raw;

        std::pair<size_t, size_t> unpack() const {
            return {static_cast<size_t>(stream_id.Value()), static_cast<size_t>(query_id.Value())};
        }
    };

    explicit QueryCacheBase(Tegra::GPU& gpu, VideoCore::RasterizerInterface& rasterizer_,
                            Tegra::MaxwellDeviceMemoryManager& device_memory_,
                            RuntimeType& runtime_);

    ~QueryCacheBase();

    void InvalidateRegion(VAddr addr, std::size_t size) {
        IterateCache<true>(addr, size,
                           [this](QueryLocation location) { InvalidateQuery(location); });
    }

    void FlushRegion(VAddr addr, std::size_t size) {
        if (SemiFlushRegion(addr, size)) {
            RequestGuestHostSync();
        }
    }

    /// Writes out the finalized reports in the region. True when some still wait for host results.
    bool SemiFlushRegion(VAddr addr, std::size_t size) {
        bool result = false;
        // Visits every query in the range: each final value not written yet is written here.
        IterateCache<false>(addr, size, [this, &result](QueryLocation location) {
            result |= SemiFlushQueryDirty(location);
        });
        return result;
    }

    static u64 BuildMask(std::span<const QueryType> types) {
        u64 mask = 0;
        for (auto query_type : types) {
            mask |= 1ULL << (static_cast<u64>(query_type));
        }
        return mask;
    }

    /// Whether a report in the region is still listed (pending host results, or written and not
    /// unregistered yet). Only the cache mutex: callable from the split front.
    [[nodiscard]] bool HasCachedReports(VAddr addr, size_t size) {
        bool result = false;
        IterateCache<false>(addr, size, [&result](QueryLocation) {
            result = true;
            return true;
        });
        return result;
    }

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size) {
        bool result = false;
        IterateCache<false>(addr, size, [this, &result](QueryLocation location) {
            result = IsQueryDirty(location);
            return result;
        });
        return result;
    }

    void CounterEnable(QueryType counter_type, bool is_enabled);

    void CounterReset(QueryType counter_type);

    void CounterClose(QueryType counter_type);

    void CounterReport(GPUVAddr addr, QueryType counter_type, QueryPropertiesFlags flags,
                       u32 payload, u32 subreport);

    /// One follower (FRITCHII_ONEF_QUERY_OPS=2): whether a report of `type` lands later, at a
    /// fence or flush (a streamer exists), not inside CounterReport (the payload write a type
    /// without a streamer becomes).
    [[nodiscard]] bool ReportLandsLater(QueryType type) const;
    /// One follower (fable_b 16.2): whether a report of `type` written now makes a pending sync.
    [[nodiscard]] bool ReportMakesPendingSync(QueryType type) const;
    /// One follower (fable_b 19.4, FRITCHII_ONEF_SEM_E), E at the report's place: a payload
    /// write to game memory, the waiters and the listeners, as CounterReport's immediate write
    /// (`completion` false) or a synced fence report's operation (true) does it. The streamers
    /// (R's) are not touched.
    void WritePayloadNow(DAddr cpu_addr, GPUVAddr addr, u32 payload, bool has_timestamp,
                         bool completion);
    /// R: the streamer's side of a payload write E made, its pending sync for the next fence.
    void NotePayloadWritten(DAddr cpu_addr, bool has_timestamp, u32 payload, u32 subreport);

    /// CounterReport with `addr` already translated (FRITCHII_ONEF_QUERY_OPS: E translates at
    /// its stream place, R writes the report).
    void CounterReportAt(GPUVAddr addr, std::optional<DAddr> device, QueryType counter_type,
                         QueryPropertiesFlags flags, u32 payload, u32 subreport);

    void NotifyWFI();

    bool AccelerateHostConditionalRendering();

    // Async downloads
    void CommitAsyncFlushes();

    bool HasUncommittedFlushes() const;

    /// Whether a new fence has queries to flush or host writes to sync for this cache.
    /// `ignore_payload_sync` (EDEN_NX2_EXP28 & 1, the original's FENCE_TRIM): payload
    /// releases' host buffer syncs don't count; they wait for a later fence or wait for idle.
    bool HasPendingFenceWork(bool ignore_payload_sync = false) const;

    /// Whether NotifyWFI would sync query results now (a streamer has pending syncs).
    bool HasPendingSync() const;

    /// EXP28 & 1: a fence was skipped although payload releases wait for their host sync.
    void NotePayloadSyncDeferred();

    bool ShouldWaitAsyncFlushes();

    void PopAsyncFlushes();

    void NotifySegment(bool resume);

    void BindToChannel(s32 id) override;

protected:
    template <bool remove_from_cache, typename Func>
    void IterateCache(VAddr addr, std::size_t size, Func&& func) {
        // invoke_result_t: with invoke_result (the trait, never bool) callbacks returning bool
        // never ended the walk early.
        static constexpr bool RETURNS_BOOL =
            std::is_same_v<std::invoke_result_t<Func, QueryLocation>, bool>;
        const u64 addr_begin = addr;
        const u64 addr_end = addr_begin + size;

        const u64 page_end = addr_end >> Core::DEVICE_PAGEBITS;
        std::scoped_lock lock(cache_mutex);
        for (u64 page = addr_begin >> Core::DEVICE_PAGEBITS; page <= page_end; ++page) {
            const u64 page_start = page << Core::DEVICE_PAGEBITS;
            const auto in_range = [page_start, addr_begin, addr_end](const u32 query_location) {
                const u64 cache_begin = page_start + query_location;
                const u64 cache_end = cache_begin + sizeof(u32);
                return cache_begin < addr_end && addr_begin < cache_end;
            };
            const auto& it = cached_queries.find(page);
            if (it == std::end(cached_queries)) {
                continue;
            }
            auto& contents = it->second;
            if constexpr (!remove_from_cache) {
                // Macro DMA often reads a few words on a page with hundreds of occlusion
                // reports. Probe the possible starts of overlapping reports on that page.
                constexpr u64 MaxProbes = 32;
                constexpr u64 ReportOverlap = sizeof(u32) - 1;
                const u64 first = (std::max)(page_start + ReportOverlap, addr_begin) -
                                  page_start - ReportOverlap;
                const u64 limit = (std::min<u64>)(addr_end - page_start, Core::DEVICE_PAGESIZE);
                if (limit > first && limit - first <= MaxProbes && limit - first < contents.size()) {
                    auto match = contents.end();
                    u64 offset = first;
                    for (; offset < limit; ++offset) {
                        const auto query = contents.find(static_cast<u32>(offset));
                        if (query == contents.end()) {
                            continue;
                        }
                        if constexpr (RETURNS_BOOL) {
                            if (func(query->second)) {
                                return;
                            }
                        } else {
                            // Keep the original write order when several reports are flushed.
                            if (match != contents.end()) {
                                break;
                            }
                            match = query;
                        }
                    }
                    if (offset == limit) {
                        if constexpr (!RETURNS_BOOL) {
                            if (match != contents.end()) {
                                func(match->second);
                            }
                        }
                        continue;
                    }
                }
            }
            bool matched = false;
            for (auto& query : contents) {
                if (!in_range(query.first)) {
                    continue;
                }
                matched = true;
                if constexpr (RETURNS_BOOL) {
                    if (func(query.second)) {
                        return;
                    }
                } else {
                    func(query.second);
                }
            }
            // Only pages with a query in the range: the scan above already found none elsewhere.
            if constexpr (remove_from_cache) {
                if (!matched) {
                    continue;
                }
                const auto in_range2 = [&](const std::pair<u32, QueryLocation>& pair) {
                    return in_range(pair.first);
                };
                erase_if(contents, in_range2);
            }
        }
    }

    using ContentCache = ::Common::unordered_map<u64, ::Common::unordered_map<u32, QueryLocation>>;

    void InvalidateQuery(QueryLocation location);
    bool IsQueryDirty(QueryLocation location);
    bool SemiFlushQueryDirty(QueryLocation location);
    void RequestGuestHostSync();
    /// Before a report's bytes go to guest memory: a download for a guest read still on its way
    /// would write the GPU's older copy over them later (as MemoryManager::Write; EXP49).
    void FinishReads(DAddr address, size_t size);
    void UnregisterPending();

    ::Common::unordered_map<u64, ::Common::unordered_map<u32, QueryLocation>> cached_queries;
    std::mutex cache_mutex;

    struct QueryCacheBaseImpl;
    friend struct QueryCacheBaseImpl;
    friend RuntimeType;

    std::unique_ptr<QueryCacheBaseImpl> impl;
};

} // namespace VideoCommon
