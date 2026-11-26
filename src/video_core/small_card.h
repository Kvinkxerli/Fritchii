// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/nx2_debug_control.h"

/// Fritchii's small card profile (FRITCHII_SMALL_CARD, docs/fritchii/switches.md): what an 8 GB
/// card (budget ~6.7-7.4 GiB with Windows and other programs) or a 6 GB one needs so the process
/// stays under the driver's budget in real play, instead of spilling into system memory (an RTX
/// 2080 at 1440p: 7.9 GB used of a 6.7-7.2 GB budget, read-backs stalling for seconds). Every part
/// frees memory that is counted wrong or held for nothing on such a card; nothing reserves more.
/// Big cards (16 GB) never turn it on in auto mode, so they keep today's behaviour.
namespace VideoCore::SmallCard {

/// FRITCHII_SMALL_CARD (live): 0 off (default for now), 1 on, 2 auto (on when the main
/// device-local heap is under 9 GiB or its budget at boot under 7.5 GiB, never on integrated GPUs).
/// The boot parts (BufferBlocks, RefuseScale, Stream) take the value at boot.
inline std::atomic<int> g_small_card{-1};

/// FRITCHII_SMALL_CARD_PARTS (boot, bit mask, default DefaultParts): which parts the profile turns
/// on, for measuring them one by one.
enum Part : u32 {
    /// Texture cache thresholds from the budget less everything the cache leaves out of its usage
    /// (image pool, deleted images still in flight, collector downloads, block slack), with a
    /// margin of 1/16 of the budget (at most 512 MiB) instead of 1/8 (at most 1 GiB), no fixed
    /// 5 GiB cap, and the out-of-memory margin capped at 1/8 of the budget instead of 3 GiB.
    Fit = 1U << 0,
    /// Image pool (EXP42) cap: 1/16 of the budget, at least 256 MiB, instead of EXP66's 1 GiB.
    Pool = 1U << 1,
    /// Buffer allocator blocks of 16 MiB instead of 32 (boot): buffers over 8 MiB get exact
    /// memory instead of sharing 32 MiB blocks they leave half empty.
    BufferBlocks = 1U << 2,
    /// A resolution scale over 1x is refused for the session (boot), with a status bar note.
    RefuseScale = 1U << 3,
    /// Stream buffer 64 MiB instead of 128 (boot). Not in the default set.
    Stream = 1U << 4,
    /// The buffer cache's device-local pool in 4 MiB blocks instead of 16 (boot;
    /// FRITCHII_BUFFER_POOL_BLOCK_MB overrides, 0 keeps 16): at Castle Town ~140 MiB of small
    /// buffers held 50 blocks, 800 MiB, a block staying while any buffer in it lives.
    BufferPoolBlocks = 1U << 5,
};
inline constexpr u32 DefaultParts =
    Fit | Pool | BufferBlocks | RefuseScale | Stream | BufferPoolBlocks;

/// Auto mode's limits.
inline constexpr u64 AutoHeapBytes = u64{9} << 30;
inline constexpr u64 AutoBudgetBytes = u64{7680} << 20;

/// -1 until Decide ran, then 0 or 1: auto mode's verdict for this card.
inline std::atomic<int> g_detected{-1};
/// The main heap's budget at boot (EDEN_NX2_VRAM_LIMIT_MIB and FRITCHII_VRAM_SIM_OTHERS applied).
inline std::atomic<u64> g_boot_budget{0};
/// The resolution scale (up factor x 100) RefuseScale turned down to 1x, 0 if none.
inline std::atomic<u32> g_refused_scale_percent{0};

[[nodiscard]] inline u32 Parts() {
    static const u32 parts = [] {
        const char* const value = std::getenv("FRITCHII_SMALL_CARD_PARTS");
        return value != nullptr && value[0] != '\0'
                   ? static_cast<u32>(std::strtoul(value, nullptr, 0))
                   : DefaultParts;
    }();
    return parts;
}

[[nodiscard]] inline int Mode() {
    return Nx2Debug::SwitchValue(g_small_card, "FRITCHII_SMALL_CARD", 0);
}

/// Whether the profile is on now (live: the debug server's switch can flip it).
[[nodiscard]] inline bool Active() {
    const int mode = Mode();
    return mode == 1 || (mode == 2 && g_detected.load(std::memory_order_relaxed) == 1);
}

/// Whether `part` is on now.
[[nodiscard]] inline bool On(Part part) {
    return (Parts() & part) != 0 && Active();
}

/// Auto mode's verdict, from the main device-local heap's size and budget at boot (both with the
/// VRAM limit of a simulated card applied). Every device made decides again (another GPU may be
/// picked between games); the line is logged when the verdict is new or changed.
inline void Decide(u64 heap_size, u64 budget, bool integrated) {
    const bool is_small = !integrated && heap_size != 0 &&
                       (heap_size < AutoHeapBytes || (budget != 0 && budget < AutoBudgetBytes));
    const int before = g_detected.exchange(is_small ? 1 : 0, std::memory_order_relaxed);
    g_boot_budget.store(budget, std::memory_order_relaxed);
    if (before == (is_small ? 1 : 0)) {
        return;
    }
    LOG_INFO(Render_Vulkan,
             "Small card profile (FRITCHII_SMALL_CARD={}): main heap {} MiB, budget {} MiB, {} "
             "card{}; profile {} (parts {:#x}: fit {}, pool {}, buffer blocks {}, refuse scale "
             "{}, stream {}, buffer pool blocks {})",
             Mode(), heap_size >> 20, budget >> 20, is_small ? "a small" : "not a small",
             integrated ? " (integrated GPU)" : "", Active() ? "on" : "off", Parts(),
             (Parts() & Fit) != 0 ? 1 : 0, (Parts() & Pool) != 0 ? 1 : 0,
             (Parts() & BufferBlocks) != 0 ? 1 : 0, (Parts() & RefuseScale) != 0 ? 1 : 0,
             (Parts() & Stream) != 0 ? 1 : 0, (Parts() & BufferPoolBlocks) != 0 ? 1 : 0);
}

/// Part RefuseScale (boot): a resolution scale over 1x turned down to 1x for this session. OoT
/// renders 2560x1440 with Max Resolution, so 2x is 5120x2880 targets (4x the memory) where an
/// 8 GB card's 1x market already uses ~6.2 of ~7.1-7.4 GiB. Only the session's scaling info
/// changes (the renderer reads nothing else); the setting stays, so a bigger card scales again.
/// Called by the device that presents, before the caches exist.
inline void RefuseScaleIfNeeded() {
    g_refused_scale_percent.store(0, std::memory_order_relaxed);
    auto& info = Settings::values.resolution_info;
    if (!On(RefuseScale) || info.up_factor <= 1.0f) {
        return;
    }
    const float refused = info.up_factor;
    Settings::TranslateResolutionInfo(Settings::ResolutionSetup::Res1X, info);
    g_refused_scale_percent.store(static_cast<u32>(refused * 100.0f + 0.5f),
                                  std::memory_order_relaxed);
    LOG_WARNING(Render_Vulkan,
                "Small card profile: resolution scale {:.2f}x does not fit a {} MiB video memory "
                "budget, this session runs at 1x (FRITCHII_SMALL_CARD_PARTS without 0x8 keeps it)",
                refused, g_boot_budget.load(std::memory_order_relaxed) >> 20);
}

/// The scale RefuseScale turned down this session (e.g. 2.0), 0 if none.
[[nodiscard]] inline float RefusedScale() {
    return static_cast<float>(g_refused_scale_percent.load(std::memory_order_relaxed)) / 100.0f;
}

} // namespace VideoCore::SmallCard
