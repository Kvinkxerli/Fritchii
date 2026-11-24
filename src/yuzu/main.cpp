// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <fmt/format.h>

#include "common/scm_rev.h"

#include <QApplication>
#include <QMessageBox>
#include <cstdlib>
#include <exception>

#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/lowend_sim.h"
#include "common/memory_detect.h"
#ifdef _WIN32
#include "common/windows/crash_handler.h"
#endif
#include "common/shutdown_watch.h"
#include "startup_checks.h"

#ifdef __unix__
#include "qt_common/gui_settings.h"
#endif

#ifndef _WIN32
#include <sys/resource.h>
#endif

#if defined(__APPLE__)
#include <climits>
#include <cstdlib>
#include <cstring>
#endif

#include "main_window.h"
#include "fritchii/fritchii_ui.h"
#include "fritchii/switches_window.h"
#include "util/profiling_mode.h"
#ifdef ENABLE_UPDATE_CHECKER
#include "updater/update_dialog.h"
#endif

#ifdef _WIN32
#include <QScreen>

static void OverrideWindowsFont() {
    // Qt5 chooses these fonts on Windows and they have fairly ugly alphanumeric/cyrillic characters
    // Asking to use "MS Shell Dlg 2" gives better other chars while leaving the Chinese Characters.
    const QString startup_font = QApplication::font().family();
    const QStringList ugly_fonts = {QStringLiteral("SimSun"), QStringLiteral("PMingLiU")};
    if (ugly_fonts.contains(startup_font)) {
        QApplication::setFont(QFont(QStringLiteral("MS Shell Dlg 2"), 9, QFont::Normal));
    }
}
#endif

// Fritchii: caps Windows' scale (passed through) so the primary screen keeps at least 1350x900
// units for the windows, by a QT_SCALE_FACTOR below 1 when it would not. Leaves a
// QT_SCALE_FACTOR the user set alone.
static void FitHighDpiScaleToScreen() {
#ifdef _WIN32
    if (qEnvironmentVariableIsSet("QT_SCALE_FACTOR")) {
        return;
    }
    qreal fit = 1.0;
    {
        int temp_argc = 0;
        char** temp_argv = nullptr;
        QApplication temp{temp_argc, temp_argv};
        const QScreen* primary_screen = QGuiApplication::primaryScreen();
        if (primary_screen == nullptr) {
            return;
        }
        const qreal ratio = primary_screen->devicePixelRatio();
        const QRect area = primary_screen->availableGeometry();
        const qreal pixels_wide = area.width() * ratio;
        const qreal pixels_high = area.height() * ratio;
        constexpr qreal minimum_width = 1350.0;
        constexpr qreal minimum_height = 900.0;
        const qreal largest = std::min(pixels_wide / minimum_width, pixels_high / minimum_height);
        if (ratio > 1.0 && largest < ratio) {
            fit = std::max(1.0, largest) / ratio;
        }
    }
    if (fit < 0.99) {
        qputenv("QT_SCALE_FACTOR", QByteArray::number(fit, 'f', 3));
    }
#endif
}

static Qt::HighDpiScaleFactorRoundingPolicy GetHighDpiRoundingPolicy() {
#ifdef _WIN32
    // For Windows, we want to avoid scaling artifacts on fractional scaling ratios.
    // This is done by setting the optimal scaling policy for the primary screen.

    // Create a temporary QApplication.
    int temp_argc = 0;
    char** temp_argv = nullptr;
    QApplication temp{temp_argc, temp_argv};

    // Get the current screen geometry.
    const QScreen* primary_screen = QGuiApplication::primaryScreen();
    if (primary_screen == nullptr) {
        return Qt::HighDpiScaleFactorRoundingPolicy::PassThrough;
    }

    const QRect screen_rect = primary_screen->geometry();
    const qreal real_ratio = primary_screen->devicePixelRatio();
    const qreal real_width = std::trunc(screen_rect.width() * real_ratio);
    const qreal real_height = std::trunc(screen_rect.height() * real_ratio);

    // Recommended minimum width and height for proper window fit.
    // Any screen with a lower resolution than this will still have a scale of 1.
    constexpr qreal minimum_width = 1350.0;
    constexpr qreal minimum_height = 900.0;

    const qreal width_ratio = std::max(1.0, real_width / minimum_width);
    const qreal height_ratio = std::max(1.0, real_height / minimum_height);

    // Get the lower of the 2 ratios and truncate, this is the maximum integer scale.
    const qreal max_ratio = std::trunc(std::min(width_ratio, height_ratio));
    return max_ratio > real_ratio ? Qt::HighDpiScaleFactorRoundingPolicy::Round
                                  : Qt::HighDpiScaleFactorRoundingPolicy::Floor;
#else
    // Other OSes should be better than Windows at fractional scaling.
    return Qt::HighDpiScaleFactorRoundingPolicy::PassThrough;
#endif
}

// The NX2 switches are read from the environment (EDEN_NX2_*), which a GUI launch has no way to
// set. nx2_switches.txt holds "NAME=VALUE" lines, one per line, '#' comments. A variable already
// in the real environment always wins, so a launcher script can still override one run; between
// files the later one wins, so the copy next to the executable (shipped with the build) can be
// overridden by %APPDATA%\fritchii\nx2_switches.txt without editing the one that gets replaced.
// Fritchii 6.2, once per switch file: v6.1 tied the GPU threads to the P-cores of hybrid Intel
// CPUs, and players worked around the fps it cost with EDEN_HYBRID_CORES=0. 6.2's default lets
// Windows place them, and the switch would keep it (and the Intel P-cores setting) from applying.
// The line is turned into a comment saying why, a marker line records the check, and a line the
// player adds back afterwards stays. Returns what was done, for the switch summary in the log.
static std::string RetireHybridCoresOff(const std::filesystem::path& path) {
    constexpr std::string_view Marker = "# Fritchii 6.2 checked EDEN_HYBRID_CORES";
    std::ifstream in{path, std::ios::binary};
    if (!in) {
        return {};
    }
    const std::string text{std::istreambuf_iterator<char>(in), {}};
    in.close();
    if (text.find(Marker) != std::string::npos) {
        return {};
    }
    const std::string newline = text.find("\r\n") != std::string::npos ? "\r\n" : "\n";
    std::string out;
    bool retired = false;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        const bool last = end == std::string::npos;
        std::string line = text.substr(start, last ? std::string::npos : end - start);
        start = last ? text.size() : end + 1;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::string bare = line;
        bare.erase(std::remove_if(bare.begin(), bare.end(),
                                  [](char c) { return c == ' ' || c == '\t'; }),
                   bare.end());
        if (bare == "EDEN_HYBRID_CORES=0") {
            out += "# " + line + "   (turned off by Fritchii 6.2: no longer needed. Remove the # "
                   "to use it again.)" + newline;
            retired = true;
        } else {
            out += line;
            if (!last) {
                out += newline;
            }
        }
    }
    if (!out.empty() && out.substr(out.size() - newline.size()) != newline) {
        out += newline;
    }
    out += std::string{Marker} + newline;
    std::ofstream file{path, std::ios::binary | std::ios::trunc};
    if (!file || !file.write(out.data(), static_cast<std::streamsize>(out.size()))) {
        return {};
    }
    return retired ? "EDEN_HYBRID_CORES=0 turned off in " + path.filename().string() +
                         " (no longer needed with 6.2)"
                   : std::string{};
}

static void ParseSwitchFile(const std::filesystem::path& path,
                            std::map<std::string, std::string>& entries) {
    std::ifstream file{path};
    if (!file) {
        return;
    }
    constexpr const char* blank = " \t\r";
    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(blank));
        line.erase(line.find_last_not_of(blank) + 1);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string name = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        name.erase(name.find_last_not_of(blank) + 1);
        value.erase(0, value.find_first_not_of(blank));
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }
        if (!name.empty()) {
            entries.insert_or_assign(std::move(name), std::move(value));
        }
    }
}

static std::filesystem::path GetSwitchExeDirectory(const char* argv0) {
#ifdef _WIN32
    return Common::FS::GetExeDirectory();
#else
    std::error_code ec;
#ifdef __linux__
    if (auto exe = std::filesystem::read_symlink("/proc/self/exe", ec); !ec) {
        return exe.parent_path();
    }
#endif
    const auto exe = std::filesystem::absolute(argv0, ec);
    return ec ? std::filesystem::path{} : exe.parent_path();
#endif
}

// Built into the Frizti builds so players don't have to set anything: the switches tuned for
// Ocarina of Time. Either nx2_switches.txt overrides any of them (=0 turns one off).
// Not EDEN_NX2_WRITTEN_SSBO_MARK_KIB=64 (built in from v4 to v5.5): writes past its cap were not
// tracked, one of the gaps behind the huge glowing triangles (the other: native compute write
// extents, see BufferCache::ExtentTrust). Castle Town: 43.4/45.0 fps with it, 44.6/43.0 without;
// the triangles still showed without it alone (2026-11-09).
constexpr std::pair<const char*, const char*> BuiltinSwitches[] = {
    {"EDEN_NX2_GRAPHICS_SSBO_BARE", "0"},
    {"EDEN_NX2_ACQUIRE_PRECHECK", "2"},   {"EDEN_NX2_MACRO_SINK", "1"},
    {"EDEN_NX2_COMPUTE_SINK", "1"},       {"EDEN_NX2_RING_PAIRS", "1"},
    {"EDEN_NX2_UBO_INPLACE", "1"},        {"EDEN_NX2_PIPELINE_CLEAN", "1"},
    {"EDEN_NX2_LIST_WAIT_SKIP", "1"},     {"EDEN_NX2_PREFLUSH", "1"},
    {"EDEN_NX2_SEM_FAST", "2"},
    // No 40 MB/s pacing of game file reads (the original code's fix for video freezes in other games):
    // OoT's videos play fine without it and it slept the game at loads and kills (2026-11-10).
    {"EDEN_NX2_READ_THROTTLE", "0"},
    // The texture cache collects from 7.5 GiB at most (a ceiling: smaller cards keep their own
    // thresholds). Owner's call for v5.7 after the 2026-11-10 playtests (VRAM 10 -> ~6 GB, more
    // RAM free, no new stutter).
    {"EDEN_NX2_TC_CEILING_MIB", "7680"},
    // Idle CPU cores compile ahead the game code another core compiled (each of OoT's 6 cores
    // has its own recompiler). Owner's call for v5.7 after the 2026-11-10 playtests: Kakariko
    // felt much better; repeat compiles after reloads 294 -> 8 ms in the night tests.
    {"EDEN_NX2_JIT_WARM", "1"},
    // The GPU thread waits for suspended channels in a wait a latch or a new command ends,
    // instead of 1 ms sleeps. Built in with EXP12 = 4 (2026-11-14, owner's call): without it a
    // latched resume could sit out a sleep and mode 4 lost 0.5 fps at the market.
    {"EDEN_NX2_GPU_WAKE", "1"},
    // 2026-11-14 (autopilot; owner: v6 gets every exact win): the keepers, each exact by its
    // verify mode, toggled together at the market in one session: 62.7 -> 64.1 fps, 1% low 44.4 ->
    // 45.5, 0.1% low 37.7 -> 40.4, the back 7.27 -> 6.99 us a draw (V6_PLAN.md, V6_GOALS.md).
    // Each stays live: =0 in nx2_switches.txt turns it off.
    {"FRITCHII_GATHER_ONE_LOCK", "1"},    {"FRITCHII_GATHER_EMPTY_TEST", "1"},
    {"FRITCHII_FENCE_COMMIT_FAST", "1"},  {"FRITCHII_FRONT_RAN_READS", "1"},
    {"FRITCHII_VERTEX_INPUT_COMPACT", "1"}, {"FRITCHII_BIND_ADDR_MEMO", "1"},
    {"FRITCHII_WORKER_SPIN", "20"},       {"FRITCHII_CB_WORDS_BLOCK", "1"},
    // 2026-11-14: uniform bank pages nobody flagged as written are not compared again. Market
    // toggle 61.3 -> 62.5 fps, 1% low 43.5 -> 45.5, 0.1% low 32.8 -> 37.4; FLAGGED=2 verify: never
    // flagged, recheck and CPU-modified 0 in all 29 windows (the other counts are timed races).
    {"FRITCHII_BANK_FLAGGED", "1"},
    // v6.0 (owner's release order 2026-11-18: bake the switch set into the defaults): the set
    // every v6 test, PGO training and tester build 1-9 ran with, so a player with no
    // nx2_switches.txt gets exactly the tested build. Values as tested; each stays live or
    // overridable in nx2_switches.txt (=0 turns one off). Linux gets the set its test builds ran.
    {"EDEN_NX2_SPLIT", "2"},              {"EDEN_NX2_FENCE_BATCH", "1"},
    {"FRITCHII_DEPTH_POOL_FIT", "3"},     {"FRITCHII_HOSTTIMING_SPIN_BUDGET", "20"},
    {"FRITCHII_HOSTTIMING_OFF_GPU", "1"}, {"FRITCHII_DLSS_LOW_VRAM_CNN", "1"},
    // Recompiler
    {"FRITCHII_JIT_INDIRECT_CACHE", "2"}, {"FRITCHII_JIT_RSQRT_ZERO", "2"},
    {"FRITCHII_JIT_NO_FRAME_POINTER", "1"}, {"FRITCHII_JIT_FLAG_BRANCH", "2"},
    {"FRITCHII_JIT_FAR_CODE", "1"},       {"FRITCHII_JIT_CHAIN", "6"},
    {"FRITCHII_JIT_BLOCK_LIST", "1"},     {"FRITCHII_JIT_INLINE_CNTPCT", "1"},
    {"FRITCHII_JIT_STATE_REFS", "1"},     {"FRITCHII_JIT_SHARED", "1"},
    {"FRITCHII_STORE_MEMO", "2"},
    // GPU front and back
    {"EDEN_NX2_EXP37", "1"},              {"EDEN_NX2_EXP103", "64"},
    {"EDEN_NX2_EXP123", "3"},             {"EDEN_NX2_LIGHT_TABLE_ARRAYS", "1"},
    {"FRITCHII_MEMSTAGE_TICKS", "1"},     {"FRITCHII_STAMP_READ_PAGES", "2"},
    {"FRITCHII_LIST_COPY_SKIP", "1"},     {"FRITCHII_DISPATCH_WALK_SUMMARY", "1"},
    {"FRITCHII_FRONT_CALL_LEAN", "1"},    {"FRITCHII_FRONT_LIST_LANE", "3"},
    {"FRITCHII_NOTED_FAST", "1"},         {"FRITCHII_FRONT_TRIM_RANGE", "1"},
    {"FRITCHII_FRONT_TRIM_QMD", "1"},     {"FRITCHII_FRONT_ASYNC_TABLE", "1"},
    {"FRITCHII_DEPTH_POOL_BIG", "1"},     {"FRITCHII_RAM_GPU_ENTRIES", "1"},
    {"FRITCHII_SMALL_CARD", "2"},         {"FRITCHII_READ_SIDE_COPY", "3"},
    {"FRITCHII_GPU_SIBLINGS_FREE", "1"},  {"FRITCHII_PACE_FIX", "3"},
    // OoT's zstd decodes (area loads, streamed files) on the host's libzstd; verify mode compared
    // 995 stream chunks with 0 mismatches (2026-11-19). 0 turns it off.
    {"FRITCHII_HOST_ZSTD", "3"},
    // FRITCHII_OCCL_FLAG is not here: the graphics setting "OoT occlusion speed-up" gives its
    // value when no switch sets it (Standard, the default, is the core: 3).
    // One follower (E/R GPU threads)
    {"FRITCHII_ONEF_LEAN", "1"},          {"FRITCHII_ONEF_AHEAD", "0"},
    {"FRITCHII_ONEF_POSTED", "1"},        {"FRITCHII_ONEF_COPY_E", "1"},
    {"FRITCHII_ONEF_UNLEND", "55"},       {"FRITCHII_ONEF_GATHER_E", "1"},
    {"FRITCHII_ONEF_OP_CLEAR", "1"},      {"FRITCHII_ONEF_OP_INVAL", "1"},
    {"FRITCHII_ONEF_POST_FENCE", "1"},    {"FRITCHII_ONEF_FLUSH_E", "1"},
    {"FRITCHII_ONEF_OP_DMA", "1"},        {"FRITCHII_ONEF_OP_LISTKEEP", "1"},
    {"FRITCHII_ONEF_SEM_E", "1"},         {"FRITCHII_ONEF_UNLENT_RESET", "1"},
    {"FRITCHII_ONEF_UNLENT_VALUES", "1"}, {"FRITCHII_ONEF_BIND_SEED", "1"},
    {"FRITCHII_ONEF_OP_DISPATCH", "7"},   {"FRITCHII_ONEF_UNION_FILTER", "1"},
    {"FRITCHII_ONEF_SOFT_RELEASE_E", "1"}, {"FRITCHII_ONEF_POST_FENCE_REPORT", "1"},
    {"FRITCHII_ONEF_CARRY_OPS", "1"},     {"FRITCHII_ONEF_KEY_KEEP", "2"},
    {"FRITCHII_ONEF_CB_LEAN", "3"},       {"FRITCHII_ONEF_DISPATCH_RANGES", "1"},
    {"FRITCHII_ONEF_POST_IDLE", "2"},     {"FRITCHII_ONEF_RAISE_AFTER_LEND", "1"},
    // zortos293/Lichii #2104 (owner 2026-11-18: keep zortos's proven stutter fixes even where
    // our tests show no difference): a native dispatch whose compute pipeline isn't built yet is
    // skipped for that frame when its guards prove nothing reads the output. Works only while a
    // pipeline builds (no cost with warm caches); cold A/B 11-18: 0 skipped, must-be-0 counts 0.
    {"FRITCHII_ASYNC_COMPUTE_SKIP", "1"},
    // Pot room (guard house): moving makes per-object shadow depth images every frame. A
    // GPU-written target whose memory newer GPU writes fully cover, untouched for 4 frames, is
    // retired; queued collector downloads still land under a GPU-only overlap. 11-14 owner run:
    // peak VRAM 12.45 -> 9.1 GB. It was only in the play build, so v6.0 shipped without it.
    {"FRITCHII_COVERED_RETIRE", "4"},
    {"FRITCHII_COLLECT_LAND_UNDER", "1"},
#ifdef _WIN32
    // Tested on Windows only (the Linux test builds ran without them).
    {"FRITCHII_JIT_PIN", "1"},            {"FRITCHII_JIT_PIN_REGS", "x8,x0"},
    {"FRITCHII_ONEF_PIN", "2"},           {"FRITCHII_GPU_PRIORITY", "2"},
    {"FRITCHII_ZERO_TRIM_BATCH", "1"},    {"FRITCHII_ZERO_TRIM_CHUNK", "1"},
    {"FRITCHII_ZERO_UPLOAD", "1"},        {"FRITCHII_REPORT_WAIT", "1"},
    {"FRITCHII_FENCE_COPY_UNLOCKED", "1"},
#endif
};

#ifndef _WIN32
extern char** environ;
#endif

// Every test switch this launch runs with and where it came from, logged at each game start
// ("Test switches:", main_window.cpp) so a stale one left in a file or a launcher's environment
// can't skew a test unseen.
static std::string g_switch_summary;
// The same, one entry per source: source name, "A=1, B=2" (Tools > Switches).
static std::vector<std::pair<std::string, std::string>> g_switch_groups;

std::string Nx2SwitchSummary() {
    return g_switch_summary;
}

std::vector<std::pair<std::string, std::string>> Nx2SwitchGroups() {
    return g_switch_groups;
}

static bool IsTestSwitch(std::string_view name) {
    return name.starts_with("FRITCHII_") || name.starts_with("EDEN_NX2_") ||
           name.starts_with("LICHII_");
}

bool Nx2IsTestSwitch(std::string_view name) {
    return IsTestSwitch(name);
}

// The player's switch file, in Fritchii's data folder: the one Tools > Switches writes.
std::filesystem::path Nx2UserSwitchFile() {
#ifdef _WIN32
    const auto app_data = Common::FS::GetAppDataRoamingDirectory();
#else
    const auto app_data = Common::FS::GetDataDirectory("XDG_CONFIG_HOME");
#endif
    return app_data / EDEN_DIR / "nx2_switches.txt";
}

// Before the first launch's import has copied it, the player's file is still in the old folder
// (Fritchii v5.8 moved from eden to fritchii): read, never written.
std::filesystem::path Nx2LegacyUserSwitchFile() {
    return Nx2UserSwitchFile().parent_path().parent_path() / EDEN_LEGACY_DIR / "nx2_switches.txt";
}

static void LoadNx2Switches(const char* argv0) {
    // The test switches already in the real environment (a launcher script's), before the files
    // add theirs.
    std::map<std::string, std::string> from_environment;
#ifdef _WIN32
    char** const environment = _environ;
#else
    char** const environment = environ;
#endif
    for (char** it = environment; it != nullptr && *it != nullptr; ++it) {
        const std::string_view entry{*it};
        const auto eq = entry.find('=');
        if (eq != std::string_view::npos && IsTestSwitch(entry.substr(0, eq))) {
            from_environment.emplace(entry.substr(0, eq), entry.substr(eq + 1));
        }
    }

    // Later files overwrite earlier entries; the real environment is checked only at the end.
    std::map<std::string, std::string> entries;
    std::map<std::string, int> origin; // index into the source names below
    const auto take = [&](std::map<std::string, std::string>&& file, int source) {
        for (auto& [name, value] : file) {
            origin.insert_or_assign(name, source);
            entries.insert_or_assign(name, std::move(value));
        }
    };
    for (const auto& [name, value] : BuiltinSwitches) {
        entries.emplace(name, value);
        origin.emplace(name, 3);
    }
    if (const auto exe_dir = GetSwitchExeDirectory(argv0); !exe_dir.empty()) {
        std::map<std::string, std::string> file;
        ParseSwitchFile(exe_dir / "nx2_switches.txt", file);
        take(std::move(file), 2);
    }
    auto user_file = Nx2UserSwitchFile();
    if (!std::filesystem::exists(user_file)) {
        user_file = Nx2LegacyUserSwitchFile();
    }
    const std::string hybrid_note = RetireHybridCoresOff(user_file);
    {
        std::map<std::string, std::string> file;
        ParseSwitchFile(user_file, file);
        take(std::move(file), 1);
    }

    // Profiling mode (Help > Profiling Mode): its switches fill in what the files and the
    // environment leave unset; they beat only a built-in value.
    std::string profiling_applied;
    if (ProfilingMode::OnAtLaunch()) {
        for (const auto& [name, value] : ProfilingMode::Switches()) {
            const auto it = origin.find(name);
            if ((it == origin.end() || it->second == 3) && !from_environment.contains(name)) {
                entries.insert_or_assign(name, value);
                origin.insert_or_assign(name, 4);
                profiling_applied +=
                    fmt::format("{}{}={}", profiling_applied.empty() ? "" : ", ", name, value);
            }
        }
    }
    ProfilingMode::SetApplied(std::move(profiling_applied));

    // "environment: A=1; user nx2_switches.txt: B=2; nx2_switches.txt next to the exe: C=1;
    // built in: D=1, E=0; profiling mode: F=1". Anything else under these prefixes (a typo, a
    // retired name) shows too, so it can be spotted.
    constexpr const char* SourceNames[] = {"environment", "user nx2_switches.txt",
                                           "nx2_switches.txt next to the exe", "built in",
                                           "profiling mode"};
    std::vector<std::string> groups(std::size(SourceNames));
    const auto add = [&groups](int source, const std::string& name, const std::string& value) {
        groups[source] += fmt::format("{}{}={}", groups[source].empty() ? "" : ", ", name, value);
    };
    for (const auto& [name, value] : from_environment) {
        add(0, name, value);
    }
    for (const auto& [name, value] : entries) {
        if (!from_environment.contains(name)) {
            add(origin[name], name, value);
        }
    }
    g_switch_summary.clear();
    g_switch_groups.clear();
    for (size_t i = 0; i < groups.size(); ++i) {
        if (!groups[i].empty()) {
            g_switch_summary += fmt::format("{}{}: {}", g_switch_summary.empty() ? "" : "; ",
                                            SourceNames[i], groups[i]);
            g_switch_groups.emplace_back(SourceNames[i], groups[i]);
        }
    }

    if (!hybrid_note.empty()) {
        g_switch_summary += fmt::format("{}{}", g_switch_summary.empty() ? "" : "; ", hybrid_note);
    }

    for (const auto& [name, value] : entries) {
        if (std::getenv(name.c_str()) != nullptr) {
            continue;
        }
#ifdef _WIN32
        _putenv_s(name.c_str(), value.c_str());
#else
        setenv(name.c_str(), value.c_str(), 1);
#endif
    }
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // The helper process of a fatal error's report: it writes the dump and ends here.
    if (int helper_exit_code = 0; Common::Windows::RunDumpHelper(helper_exit_code)) {
        return helper_exit_code;
    }
#endif
#ifdef ENABLE_UPDATE_CHECKER
    // Before the switches are added to the environment, which a restart must not inherit.
    Updater::PrepareLaunch();
#endif
    // Profiling mode's saved setting and the clean environment for its restart, before the
    // switches are added.
    ProfilingMode::ReadAtLaunch();
    LoadNx2Switches(argv[0]);
    Common::InstallCommitLimitNewHandler();
    // Test only, off by default: the low-end PC simulation's limits on this process.
    Common::LowEndSim::ApplyProcessLimits();
    // Fritchii: NVIDIA's driver shader cache must hold OoT's ~120,000 prepared pipelines. Its size
    // is a Control Panel setting newer drivers no longer show ("Shader Cache Size"); the driver
    // also reads these variables, for this process only (nothing system-wide changes). A value
    // already set (the player's, or nx2_switches.txt) wins. Every Fritchii build, the classic UI
    // (FRITCHII_CLASSIC_UI=1) included: the cache is about the game, not the interface.
    if (std::string_view{Common::g_build_name} == "Fritchii") {
        if (!qEnvironmentVariableIsSet("__GL_SHADER_DISK_CACHE_SKIP_CLEANUP")) {
            qputenv("__GL_SHADER_DISK_CACHE_SKIP_CLEANUP", "1");
        }
        if (!qEnvironmentVariableIsSet("__GL_SHADER_DISK_CACHE_SIZE")) {
            qputenv("__GL_SHADER_DISK_CACHE_SIZE", "10737418240"); // 10 GiB
        }
    }
    bool has_broken_vulkan = false;
    bool is_child = false;
    if (CheckEnvVars(&is_child)) {
        return 0;
    }

    if (StartupChecks(argv[0], &has_broken_vulkan,
                      Settings::values.perform_vulkan_check.GetValue())) {
        return 0;
    }

#ifdef YUZU_CRASH_DUMPS
    Breakpad::InstallCrashHandler();
#endif

    // Init settings params
    QCoreApplication::setOrganizationName(QStringLiteral("eden"));
    QCoreApplication::setApplicationName(QStringLiteral("Lichii"));

    // Increases the maximum open file limit.
    // TODO: This should be common to all frontends.
#ifdef _WIN32
    // MSVCRT limits this to 2048 for some inexplicable (and likely arcane) reason,
    // so we have to account for that as well.
#ifdef __MSVCRT__
    _setmaxstdio(2048);
#else
    _setmaxstdio(8192);
#endif // __MSVCRT__
#elif defined(__unix__) || defined(__APPLE__)
    // Raise the open file limit to at least 8192, or the hard limit.
    // Most sane systems should not hit the hard limit here.
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        rl.rlim_cur = std::min<rlim_t>(1048576, rl.rlim_max);
        setrlimit(RLIMIT_NOFILE, &rl);
    }
#endif // _WIN32

#if defined(__APPLE__)
    // Convert the relative path to an absolute path before the chdir
    char resolved[PATH_MAX];
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "-input-profile") == 0) {
            ++i;
        } else if (argv[i][0] != '-' && argv[i][0] != '/' && realpath(argv[i], resolved)) {
            argv[i] = strdup(resolved);
        }
    }

    // If you start a bundle (binary) on OSX without the Terminal, the working directory is "/".
    // But since we require the working directory to be the executable path for the location of
    // the user folder in the Qt Frontend, we need to cd into that working directory
    const auto bin_path = Common::FS::GetBundleDirectory() / "..";
    chdir(Common::FS::PathToUTF8String(bin_path).c_str());
#endif

#ifdef __unix__
    // Set the DISPLAY variable in order to open web browsers
    // TODO (lat9nq): Find a better solution for AppImages to start external applications
    if (QString::fromLocal8Bit(qgetenv("DISPLAY")).isEmpty()) {
        qputenv("DISPLAY", ":0");
    }

    if (GraphicsBackend::GetForceX11() && qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
        qputenv("QT_QPA_PLATFORM", "xcb");

    // Fix the Wayland appId. This needs to match the name of the .desktop file without the .desktop
    // suffix.
    QGuiApplication::setDesktopFileName(QStringLiteral("io.github.zortos293.lichii"));
#endif

    // Fritchii: Windows' own scale (125%, 150%, 175%), as large as the screen leaves the windows
    // the room they were laid out for (1350x900, the rounding's minimum below). The rounding
    // turned a 4K screen at 150% into 200% and 125% into 100%; passing a large scale through as
    // is left a 1440p screen at 175% only ~820 units tall, below Game Settings' OK button.
    auto rounding_policy = GetHighDpiRoundingPolicy();
    if (FritchiiUi::Enabled()) {
        rounding_policy = Qt::HighDpiScaleFactorRoundingPolicy::PassThrough;
        FitHighDpiScaleToScreen();
    }
    QApplication::setHighDpiScaleFactorRoundingPolicy(rounding_policy);

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    // Disables the "?" button on all dialogs. Disabled by default on Qt6.
    QCoreApplication::setAttribute(Qt::AA_DisableWindowContextHelpButton);
#endif

    // Enables the core to make the qt created contexts current on std::threads
    QCoreApplication::setAttribute(Qt::AA_DontCheckOpenGLContextThreadAffinity);

#ifdef _WIN32
    QApplication::setStyle(QStringLiteral("windowsvista"));
#endif

    // Shutdown watch steps of the process exit (common/shutdown_watch.h), named as each object
    // below goes: this one after the QApplication.
    struct ExitStep {
        const char* name;
        ~ExitStep() {
            Common::ShutdownWatch::Exiting(name);
        }
    };
    const ExitStep after_app{"QApplication destroyed: static destructors, then the process exit"};

    QApplication app(argc, argv);

#ifdef _WIN32
    OverrideWindowsFont();
#endif

    // Workaround for QTBUG-85409, for Suzhou numerals the number 1 is actually \u3021
    // so we can see if we get \u3008 instead
    // TL;DR all other number formats are consecutive in unicode code points
    // This bug is fixed in Qt6, specifically 6.0.0-alpha1
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    const QLocale locale = QLocale::system();
    if (QStringLiteral("\u3008") == locale.toString(1)) {
        QLocale::setDefault(QLocale::system().name());
    }
#endif

    // Qt changes the locale and causes issues in float conversion using std::to_string() when
    // generating shaders
    setlocale(LC_ALL, "C");

    try {
        const ExitStep after_window{"main window destroyed: QApplication destructor (thread pool)"};
        MainWindow main_window{has_broken_vulkan};
        main_window.show();
        app.connect(&app, &QGuiApplication::applicationStateChanged, &main_window,
                    &MainWindow::OnAppFocusStateChanged);
        return app.exec();
    } catch (const std::exception& exception) {
        QMessageBox::critical(nullptr, QObject::tr("Fritchii startup stopped"),
                              QString::fromUtf8(exception.what()));
        return EXIT_FAILURE;
    }
}
