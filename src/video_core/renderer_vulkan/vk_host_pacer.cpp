// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>

#include "common/logging.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_perf.h"
#include "video_core/renderer_vulkan/vk_host_pacer.h"

#ifdef _WIN32
#include <cwchar>
#include <string_view>
#include <vector>
#include <windows.h>
#include <dwmapi.h>
#include "common/string_util.h"
#pragma comment(lib, "dwmapi.lib")
#endif

namespace Vulkan {
namespace {
using Ns = std::chrono::nanoseconds;
using Ms = std::chrono::duration<double, std::milli>;

Ns ToNs(double ns) {
    return Ns{static_cast<s64>(ns)};
}
} // Anonymous namespace

#ifdef _WIN32
namespace {
struct DwmPeriod {
    std::optional<double> measured; ///< The compositor's measured period
    std::optional<double> nominal;
};

DwmPeriod QueryDwmPeriod() {
    DwmPeriod result;
    DWM_TIMING_INFO info{};
    info.cbSize = sizeof(info);
    if (FAILED(DwmGetCompositionTimingInfo(nullptr, &info)) ||
        info.rateRefresh.uiNumerator == 0 || info.rateRefresh.uiDenominator == 0) {
        return result;
    }
    result.nominal = 1e9 * info.rateRefresh.uiDenominator / info.rateRefresh.uiNumerator;
    LARGE_INTEGER frequency{};
    if (info.qpcRefreshPeriod != 0 && QueryPerformanceFrequency(&frequency) &&
        frequency.QuadPart != 0) {
        result.measured = 1e9 * static_cast<double>(info.qpcRefreshPeriod) /
                          static_cast<double>(frequency.QuadPart);
    }
    return result;
}

bool Agrees(double a, double b) {
    return std::abs(a - b) < b * 0.01;
}

/// The exact rate of the active display path whose source is GDI display `device` (119.982 Hz
/// rather than 120), or 0.
double QueryPathPeriod(const wchar_t* device) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        UINT32 path_count = 0;
        UINT32 mode_count = 0;
        if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &path_count, &mode_count) !=
            ERROR_SUCCESS) {
            return 0.0;
        }
        std::vector<DISPLAYCONFIG_PATH_INFO> paths(path_count);
        std::vector<DISPLAYCONFIG_MODE_INFO> modes(mode_count);
        const LONG status = QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &path_count, paths.data(),
                                               &mode_count, modes.data(), nullptr);
        if (status == ERROR_INSUFFICIENT_BUFFER) {
            continue; // The configuration changed in between
        }
        if (status != ERROR_SUCCESS) {
            return 0.0;
        }
        for (UINT32 i = 0; i < path_count; ++i) {
            const DISPLAYCONFIG_PATH_INFO& path = paths[i];
            DISPLAYCONFIG_SOURCE_DEVICE_NAME source{};
            source.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
            source.header.size = sizeof(source);
            source.header.adapterId = path.sourceInfo.adapterId;
            source.header.id = path.sourceInfo.id;
            if (DisplayConfigGetDeviceInfo(&source.header) != ERROR_SUCCESS ||
                std::wcscmp(source.viewGdiDeviceName, device) != 0) {
                continue;
            }
            DISPLAYCONFIG_RATIONAL rate = path.targetInfo.refreshRate;
            if ((rate.Numerator == 0 || rate.Denominator == 0) &&
                path.targetInfo.modeInfoIdx < mode_count &&
                modes[path.targetInfo.modeInfoIdx].infoType ==
                    DISPLAYCONFIG_MODE_INFO_TYPE_TARGET) {
                rate = modes[path.targetInfo.modeInfoIdx]
                           .targetMode.targetVideoSignalInfo.vSyncFreq;
            }
            if (rate.Numerator != 0 && rate.Denominator != 0) {
                return 1e9 * static_cast<double>(rate.Denominator) /
                       static_cast<double>(rate.Numerator);
            }
        }
        return 0.0;
    }
    return 0.0;
}
} // Anonymous namespace
#endif

std::optional<DisplayRefresh> DisplayRefreshQuery::Query([[maybe_unused]] void* native_window) {
#ifdef _WIN32
    const DwmPeriod dwm = QueryDwmPeriod();

    const HMONITOR monitor =
        native_window != nullptr
            ? MonitorFromWindow(static_cast<HWND>(native_window), MONITOR_DEFAULTTONEAREST)
            : MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFOEXW info{};
    info.cbSize = sizeof(info);
    if (monitor != nullptr && GetMonitorInfoW(monitor, &info)) {
        const bool primary = (info.dwFlags & MONITORINFOF_PRIMARY) != 0;
        const std::string gdi_name = Common::UTF16ToUTF8(std::wstring_view{info.szDevice});
        // "\\.\DISPLAY2" -> "DISPLAY2"
        std::string name = gdi_name.substr(gdi_name.find_last_of('\\') + 1);
        if (primary) {
            name += ", primary";
        }

        DEVMODEW mode{};
        mode.dmSize = sizeof(mode);
        const u32 hz = EnumDisplaySettingsW(info.szDevice, ENUM_CURRENT_SETTINGS, &mode) &&
                               mode.dmDisplayFrequency > 1
                           ? static_cast<u32>(mode.dmDisplayFrequency)
                           : 0;
        // The display configuration is the slow part: read it again only when the window moved
        // to another display or that display's mode changed, or now and then.
        const auto now = std::chrono::steady_clock::now();
        if (gdi_name != device || hz != nominal_hz || now >= stale_at) {
            device = gdi_name;
            nominal_hz = hz;
            rational_ns = QueryPathPeriod(info.szDevice);
            // Windows' dynamic refresh rate keeps the desktop at its low rate (60 Hz) and raises
            // it only for moving content, so the current mode can be below what the game gets
            fastest_hz = hz;
            for (DWORD index = 0;; ++index) {
                DEVMODEW other{};
                other.dmSize = sizeof(other);
                if (!EnumDisplaySettingsW(info.szDevice, index, &other)) {
                    break;
                }
                if (other.dmPelsWidth == mode.dmPelsWidth &&
                    other.dmPelsHeight == mode.dmPelsHeight && other.dmDisplayFrequency > 1) {
                    fastest_hz = (std::max)(fastest_hz, static_cast<u32>(other.dmDisplayFrequency));
                }
            }
            stale_at = now + std::chrono::seconds{30};
        }

        std::optional<DisplayRefresh> result;
        if (rational_ns > 0.0) {
            result = DisplayRefresh{rational_ns, name + ", display config"};
        } else if (hz != 0) {
            result = DisplayRefresh{1e9 / hz, name + ", display settings"};
        }
        if (result && fastest_hz > hz) {
            result->fastest_ns = 1e9 / fastest_hz;
        }
        if (result) {
            // DWM's timing is the primary display's. Its measured period is the most precise
            // one there, and the game's frame clock follows it (genlock).
            if (primary && dwm.measured && Agrees(*dwm.measured, result->period_ns)) {
                return DisplayRefresh{*dwm.measured, name + ", DWM", result->fastest_ns};
            }
            return result;
        }
    }

    // The window's display is unknown: the primary display from DWM, measured when it agrees
    // with the nominal rate (a "120 Hz" display may run at 119.98).
    if (dwm.nominal) {
        if (dwm.measured && Agrees(*dwm.measured, *dwm.nominal)) {
            return DisplayRefresh{*dwm.measured, "primary, DWM"};
        }
        return DisplayRefresh{*dwm.nominal, "primary, DWM nominal"};
    }
    DEVMODEW mode{};
    mode.dmSize = sizeof(mode);
    if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode) &&
        mode.dmDisplayFrequency > 1) {
        return DisplayRefresh{1e9 / mode.dmDisplayFrequency, "primary, display settings"};
    }
#endif
    return std::nullopt;
}

u32 PrimaryDisplayHeight() {
#ifdef _WIN32
    DEVMODEW mode{};
    mode.dmSize = sizeof(mode);
    if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode) && mode.dmPelsHeight > 0) {
        return static_cast<u32>(mode.dmPelsHeight);
    }
#endif
    return 0;
}

void HostPacer::NoteQueued(Clock::time_point queued) {
    if (last_queued) {
        // A load screen's or a stall's gap would take seconds to forget: clamp each sample.
        const double sample =
            std::clamp(static_cast<double>(Ns{queued - *last_queued}.count()), refresh_ns * 0.5,
                       100e6);
        content_ns += (sample - content_ns) * 0.05;
    }
    last_queued = queued;
}

void HostPacer::Refresh(Clock::time_point now, bool allowed, void* native_window) {
    if (now >= next_refresh_query) {
        next_refresh_query = now + std::chrono::seconds{2};
        if (const auto refresh = refresh_query.Query(native_window);
            refresh && refresh->period_ns > 2e6 && refresh->period_ns < 50e6) {
            const double period = refresh->period_ns;
            if (!refresh_known || std::abs(period - refresh_ns) > refresh_ns * 0.01 ||
                refresh->display != refresh_display || refresh->fastest_ns != fastest_ns) {
                LOG_INFO(Render_Vulkan,
                         "Host pacing: display refresh {:.4f} Hz (monitor {}){}", 1e9 / period,
                         refresh->display,
                         refresh->fastest_ns > 0.0
                             ? fmt::format("; up to {:.0f} Hz with dynamic refresh rate or "
                                           "adaptive sync",
                                           1e9 / refresh->fastest_ns)
                             : std::string{});
                // A new rate or another display: its own refresh grid.
                vblank.reset();
                refresh_display = refresh->display;
            }
            refresh_ns = period;
            fastest_ns = refresh->fastest_ns;
            refresh_known = true;
        }
    }
    pacing = allowed && refresh_known && !broken &&
             VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_host_pace,
                                              "EDEN_NX2_HOST_PACE", 1) != 0;
    if (const int value = VideoCore::Nx2Debug::SwitchValue(VideoCore::Nx2Debug::g_pace_fix,
                                                           "FRITCHII_PACE_FIX", 0);
        value != fix) [[unlikely]] {
        LOG_INFO(Render_Vulkan, "Host pacing: FRITCHII_PACE_FIX {} (was {})", value, fix);
        fix = value;
        guard_extra_ns = 0.0;
    }
    VideoCommon::Nx2Perf::g_host_refresh_ns.store(
        pacing ? static_cast<long long>(refresh_ns) : 0, std::memory_order_relaxed);
    // The high refresh rate cap ("Display"): the fastest rate the display can show the game at
    VideoCommon::Nx2Perf::g_display_refresh_ns.store(
        refresh_known ? static_cast<long long>(fastest_ns > 0.0 ? fastest_ns : refresh_ns) : 0,
        std::memory_order_relaxed);
}

double HostPacer::LeadNs() const {
    if ((fix & 1) == 0) {
        return lead_ns;
    }
    // FRITCHII_PACE_FIX 1: present just after the refresh before the intended one. The frame
    // cannot be taken at that refresh any more (vblank follows when frames were shown, which is
    // after the refresh), so it is shown at the intended one, and the copy and the compositor
    // get nearly a whole refresh instead of the ~3 ms the adaptive lead settles at.
    const double guard =
        (std::min)(std::clamp(refresh_ns * 0.15, 1e6, 3e6) + guard_extra_ns, refresh_ns * 0.5);
    return refresh_ns - guard;
}

HostPacer::Clock::time_point HostPacer::Target(Clock::time_point now) {
    pending_intended.reset();
    pending_target.reset();
    pending_woke.reset();
    pending_catch_up = false;
    if (!pacing || !vblank || outstanding != 0) {
        return now;
    }
    // Refreshes per game frame, from the frame rate the game asks for: 2 for 60 fps at 120 Hz,
    // also while it runs at 55 (those frames then show for 2 or 3), 4 for 30 fps. Not from how
    // fast frames arrive: holding frames slows the game, and the pacer then locked itself to
    // the slower rate (a loading screen's 25 ms frames held Ocarina at 40 fps). Unknown: only
    // one frame per refresh.
    const long long guest_ns =
        VideoCommon::Nx2Perf::g_guest_frame_interval_ns.load(std::memory_order_relaxed);
    const s64 k =
        guest_ns > 0
            ? (std::max<s64>)(1, static_cast<s64>(std::floor(guest_ns / refresh_ns + 0.02)))
            : 1;
    auto intended = *vblank + ToNs(static_cast<double>(k) * refresh_ns);
    const double lead = LeadNs();
    // A late frame moves the schedule a refresh later for good, and the frames after it then
    // wait that refresh in the queue (input lag). Once frames could have been shown a refresh
    // earlier for 2 s (1 s at 30 fps and below) in a row, show one a refresh early (one short
    // frame) to drop that wait. At 30 fps without it, Ocarina's frames waited ~30 ms each after
    // a few hitches: nearly a frame of extra input lag.
    // FRITCHII_PACE_FIX 1: by when the frame reached the present thread; the thread itself only
    // looks after the previous frame was shown, too late for the early lead's earlier target.
    const auto ready = (fix & 1) != 0 && last_queued ? (std::min)(*last_queued, now) : now;
    if (k >= 2 && intended - ToNs(refresh_ns + lead + 1e6) > ready) {
        if (++early_streak >= (k == 2 ? 120U : 30U)) {
            early_streak = 0;
            intended -= ToNs(refresh_ns);
            ++caught_up;
            pending_catch_up = true;
        }
    } else {
        early_streak = 0;
    }
    if (now >= intended - ToNs(1e6)) {
        ++game_late; // Reached the present thread too late for its refresh
    }
    const auto target = intended - ToNs(lead);
    if (target <= now) {
        // FRITCHII_PACE_FIX 1: presented at once, but still in time: keep it for the miss count
        // (the early target makes this common, and those frames must not drop out of it).
        if ((fix & 1) != 0 && now < intended - ToNs(1e6)) {
            pending_intended = intended;
            pending_woke = now;
        }
        return now;
    }
    pending_intended = intended;
    pending_target = target;
    return target;
}

long long HostPacer::NsToNextRefresh(Clock::time_point now) const {
    if (!vblank || !pacing) {
        return -1;
    }
    const double since = static_cast<double>(Ns{now - *vblank}.count());
    const double into = std::fmod(std::fmod(since, refresh_ns) + refresh_ns, refresh_ns);
    return static_cast<long long>(refresh_ns - into);
}

void HostPacer::NoteSubmitted(u64 id, Clock::time_point submitted_at, bool was_held) {
    if (id == 0) {
        outstanding = 0;
        pending_intended.reset();
        pending_target.reset();
        pending_woke.reset();
        return;
    }
    // Held frames, and with FRITCHII_PACE_FIX 1 the ones presented at once but in time (Target
    // left no target for them).
    const bool tracked = was_held || (pending_intended && !pending_target);
    submitted.push_back({id, submitted_at, tracked ? pending_intended : std::nullopt,
                         pending_target, pending_woke, pending_catch_up});
    while (submitted.size() > 16) {
        submitted.pop_front();
    }
    if (tracked && pending_woke) {
        const double copy = Ms{submitted_at - *pending_woke}.count();
        ++copies;
        copy_sum_ms += copy;
        copy_max_ms = (std::max)(copy_max_ms, copy);
    }
    pending_intended.reset();
    pending_target.reset();
    pending_woke.reset();
    outstanding = id;
}

void HostPacer::NoteShown(u64 id, Clock::time_point shown, bool was_lost) {
    while (!submitted.empty() && submitted.front().id < id) {
        submitted.pop_front();
    }
    std::optional<Submitted> entry;
    if (!submitted.empty() && submitted.front().id == id) {
        entry = submitted.front();
        submitted.pop_front();
    }
    if (outstanding != 0 && id >= outstanding) {
        outstanding = 0;
    }
    if (report_start == Clock::time_point{}) {
        report_start = shown;
    }
    if (was_lost) {
        ++lost;
        vblank.reset();
        last_shown.reset();
        return;
    }
    ++frames;
    if (entry) {
        latency_sum_ms += Ms{shown - entry->submitted}.count();
    }
    if (last_shown && last_shown_id + 1 == id) {
        const double interval = static_cast<double>(Ns{shown - *last_shown}.count());
        interval_sum_ms += interval / 1e6;
        interval_max_ms = (std::max)(interval_max_ms, interval / 1e6);
        const auto count = std::clamp<s64>(std::llround(interval / refresh_ns), 0, 5);
        ++refreshes_shown[static_cast<size_t>(count)];
    }
    // Follow the refresh grid: snap to the refresh this was shown at and correct the phase a
    // little, so the wake-up noise of the wait does not move the targets.
    if (vblank) {
        const double since = static_cast<double>(Ns{shown - *vblank}.count());
        const s64 refreshes = std::llround(since / refresh_ns);
        if (refreshes >= 1) {
            const auto predicted = *vblank + ToNs(static_cast<double>(refreshes) * refresh_ns);
            const double residual = static_cast<double>(Ns{shown - predicted}.count());
            vblank = std::abs(residual) < refresh_ns * 0.5 ? predicted + ToNs(residual * 0.1)
                                                            : shown;
        }
    } else {
        vblank = shown;
    }
    if (entry && entry->intended) {
        if (shown > *entry->intended + ToNs(refresh_ns * 0.5)) {
            ++missed;
            lead_ns = (std::min)(lead_ns + 0.5e6, refresh_ns * 0.75);
            // Where its lead went: the hold woke late, the copy and present calls took it, or
            // the frame was handed over in time and the GPU copy or the compositor was late.
            const double left = Ms{*entry->intended - entry->submitted}.count();
            if (entry->catch_up) {
                ++missed_catch_up;
            } else if (entry->target && entry->woke &&
                       *entry->woke > *entry->target + std::chrono::milliseconds{1}) {
                ++missed_woke;
            } else if (left < 0.5) {
                ++missed_copy;
            } else {
                ++missed_after;
                missed_left_sum_ms += left;
                missed_left_max_ms = (std::max)(missed_left_max_ms, left);
            }
        } else {
            lead_ns = (std::max)(lead_ns - 0.01e6, 0.5e6);
            if (shown < *entry->intended - ToNs(refresh_ns * 0.5)) {
                // Shown a refresh before the one it was held for.
                ++early;
                if ((fix & 1) != 0) {
                    guard_extra_ns = (std::min)(guard_extra_ns + 0.25e6, refresh_ns * 0.25);
                }
            }
        }
    }
    last_shown = shown;
    last_shown_id = id;
    Report(shown);
}

void HostPacer::NoteShownWait(bool shown) {
    if (shown) {
        shown_wait_misses = 0;
        return;
    }
    if (++shown_wait_misses >= 8 && !broken) {
        broken = true;
        pacing = false;
        VideoCommon::Nx2Perf::g_host_refresh_ns.store(0, std::memory_order_relaxed);
        LOG_WARNING(Render_Vulkan, "Host pacing: frames are not reported shown in time (present "
                                   "wait does not signal here); pacing is off for this session");
    }
}

void HostPacer::NoteHold(Clock::time_point start, Clock::time_point woke) {
    ++held;
    hold_sum_ms += Ms{woke - start}.count();
    pending_woke = woke;
}

void HostPacer::Report(Clock::time_point now) {
    if (now - report_start < std::chrono::seconds{10} || frames == 0) {
        return;
    }
    u32 intervals = 0;
    for (const u32 count : refreshes_shown) {
        intervals += count;
    }
    LOG_INFO(Render_Vulkan,
             "Host pacing {}: {} frames at {:.3f} Hz; shown for 1/2/3/4/5+ refreshes: "
             "{}/{}/{}/{}/{} (same refresh {}); interval mean {:.2f} ms, max {:.2f}; held {} "
             "(mean {:.2f} ms, missed {}), lead {:.2f} ms; present to shown {:.2f} ms; game "
             "frame {:.2f} ms; {} lost; caught up {}; misses: {} catch-up, {} woke late, {} slow "
             "copy, {} after present (time left {:.2f} ms mean, {:.2f} max); copy to present "
             "{:.2f} ms mean, {:.2f} max; game late {}; early {}; fix {}",
             pacing ? "on" : "off", frames, 1e9 / refresh_ns, refreshes_shown[1],
             refreshes_shown[2], refreshes_shown[3], refreshes_shown[4], refreshes_shown[5],
             refreshes_shown[0], intervals ? interval_sum_ms / intervals : 0.0, interval_max_ms,
             held, held ? hold_sum_ms / held : 0.0, missed, LeadNs() / 1e6,
             latency_sum_ms / frames, content_ns / 1e6, lost, caught_up, missed_catch_up,
             missed_woke, missed_copy, missed_after,
             missed_after ? missed_left_sum_ms / missed_after : 0.0, missed_left_max_ms,
             copies ? copy_sum_ms / copies : 0.0, copy_max_ms, game_late, early, fix);
    report_start = now;
    refreshes_shown = {};
    frames = held = missed = lost = caught_up = 0;
    interval_sum_ms = interval_max_ms = hold_sum_ms = latency_sum_ms = 0;
    missed_catch_up = missed_woke = missed_copy = missed_after = game_late = early = copies = 0;
    missed_left_sum_ms = missed_left_max_ms = copy_sum_ms = copy_max_ms = 0;
}

} // namespace Vulkan
