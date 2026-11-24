// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <span>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "common/assert.h"
#include "common/hex_util.h"
#include "common/host_video_memory.h"
#include "common/logging.h"
#include "common/settings.h"
#ifndef _WIN32
#include "common/string_util.h"
#endif

#include "core/arm/debug.h"
#include "core/core.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/content_archive.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/ips_layer.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/vfs/vfs_cached.h"
#include "core/file_sys/vfs/vfs_layered.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/hle/service/ns/language.h"
#include "core/hle/service/set/settings_server.h"
#include "core/loader/loader.h"
#include "core/loader/nso.h"
#include "core/memory/cheat_engine.h"

namespace FileSys {
namespace {

constexpr u32 SINGLE_BYTE_MODULUS = 0x100;

constexpr std::array<const char*, 14> EXEFS_FILE_NAMES{
    "main",    "main.npdm", "rtld",    "sdk",     "subsdk0", "subsdk1", "subsdk2",
    "subsdk3", "subsdk4",   "subsdk5", "subsdk6", "subsdk7", "subsdk8", "subsdk9",
};

/// Bytes a built-in patch writes at an offset into the module's code (after the NSO header; an
/// exefs IPS patch's offset is 0x100 more). With `checked`, the module must hold `original` there.
struct BuiltinPatchRecord {
    size_t module_offset;
    std::span<const u8> bytes;
    bool checked;
    u32 original;
    /// When set, this environment variable at 1 leaves this one record out: a hook stays the
    /// game's word while the rest of the patch applies (tests of one site of a patch).
    const char* skip_env = nullptr;
    /// When set, this record applies only while this environment variable is 1: the other form of
    /// a word whose usual record has the same variable as its skip_env.
    const char* need_env = nullptr;
};

struct BuiltinPatch {
    std::string_view name;
    u64 title_id;
    std::string_view module;
    std::string_view build_id;
    std::span<const BuiltinPatchRecord> records;
    /// When set, the patch applies only while this environment variable is 1 (A/B before it
    /// becomes a default).
    const char* enable_env = nullptr;
    /// When set, this environment variable at 1 skips the patch and at 0 applies it even when
    /// Add-Ons has it off (A/B runs without touching the player's Add-Ons).
    const char* disable_env = nullptr;
    /// When set, the patch applies (and is listed in Add-Ons) only while this returns true.
    bool (*setting_enabled)() = nullptr;
    /// Another built-in this one contains: skipped once this one is applied.
    std::string_view replaces{};
    /// When set, guest state the patch keeps in the module (offset, size): it must lie in BSS.
    size_t state_offset = 0;
    size_t state_size = 0;
    /// OoT's render height this patch sets (oot_render_height, Settings::OcarinaRenderHeight). It
    /// needs Max Resolution applied (it changes that path), and the renderer reads what was
    /// applied (Settings::AppliedOcarinaRenderHeight). 0 = not a render-height patch.
    u32 render_height = 0;
};

// Max Resolution turns the game's dynamic resolution off, so the 3D stays at 1440p docked. A
// small card (< 5 GiB) then spends heavy frame time at 1440p; keep dynamic resolution unless overridden.
bool MaxResolutionFits() {
    if (const char* const value = std::getenv("EDEN_NX2_NO_MAX_RES"); value && *value == '0') {
        return true;
    }
    const bool small = Common::HostHasSmallVideoMemory();
    static bool logged{};
    if (small && !std::exchange(logged, true)) {
        LOG_INFO(Loader, "Max Resolution (built-in) left out: {} MiB of video memory, the game "
                         "keeps its dynamic resolution",
                 Common::HostDedicatedVideoMemory() >> 20);
    }
    return !small;
}

// Ocarina of Time takes its "dynamic resolution disabled" path, so the 3D renders at full scale
// instead of following the GPU-time controller down to its 480p floor.
constexpr std::array<u8, 4> MAX_RESOLUTION_BRANCH{0x0F, 0x02, 0x00, 0x14}; // b 0x80640c
constexpr std::array MAX_RESOLUTION_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x805BD0,
                       .bytes = MAX_RESOLUTION_BRANCH,
                       .checked = true,
                       .original = 0x360041E8}, // tbz w8, #0, 0x80640c
};

constexpr std::string_view MAX_RESOLUTION_NAME = "Max Resolution (built-in)";

// OoT's render heights (oot_render_height; before, FRITCHII_DLSS_RENDER_SCALE's DLSS qualities:
// ProfessorTee's community patch as ported to Lichii): Max Resolution's fixed-scale path loads
// 1.0 of 2560x1440 (main+0x806420: mov w9, #0x3f800000, the float's top 16 bits). Lower scales
// make the game draw a smaller scene that DLSS or FSR enlarge; the emulator's resolution scale
// stays 1x, so every pass of the game, its compute passes too, runs at the game's own size.
constexpr u32 RENDER_SCALE_ONE = 0x52A7F009; // 1.0
// mov w9, #imm16, lsl #16 with the scale's top 16 bits (each scale below is exact in them).
constexpr std::array<u8, 4> RenderScaleBytes(u32 float_top) {
    const u32 instruction = 0x52A00009U | (float_top << 5);
    return {static_cast<u8>(instruction), static_cast<u8>(instruction >> 8),
            static_cast<u8>(instruction >> 16), static_cast<u8>(instruction >> 24)};
}
constexpr std::array<u8, 4> RENDER_SCALE_360 = RenderScaleBytes(0x3E80);  // 0.25, 640x360
constexpr std::array<u8, 4> RENDER_SCALE_540 = RenderScaleBytes(0x3EC0);  // 0.375, 960x540
constexpr std::array<u8, 4> RENDER_SCALE_720 = RenderScaleBytes(0x3F00);  // 0.5, 1280x720
constexpr std::array<u8, 4> RENDER_SCALE_810 = RenderScaleBytes(0x3F10);  // 0.5625, 1440x810
constexpr std::array<u8, 4> RENDER_SCALE_900 = RenderScaleBytes(0x3F20);  // 0.625, 1600x900
constexpr std::array<u8, 4> RENDER_SCALE_1080 = RenderScaleBytes(0x3F40); // 0.75, 1920x1080
constexpr std::array<u8, 4> RENDER_SCALE_1260 = RenderScaleBytes(0x3F60); // 0.875, 2240x1260
static_assert(RenderScaleBytes(0x3F80) == std::array<u8, 4>{0x09, 0xF0, 0xA7, 0x52});
static_assert(RENDER_SCALE_1080 == std::array<u8, 4>{0x09, 0xE8, 0xA7, 0x52});
static_assert(RENDER_SCALE_720 == std::array<u8, 4>{0x09, 0xE0, 0xA7, 0x52});
static_assert(RENDER_SCALE_540 == std::array<u8, 4>{0x09, 0xD8, 0xA7, 0x52});
static_assert(RENDER_SCALE_360 == std::array<u8, 4>{0x09, 0xD0, 0xA7, 0x52});
constexpr std::array RENDER_360_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x806420,
                       .bytes = RENDER_SCALE_360,
                       .checked = true,
                       .original = RENDER_SCALE_ONE},
};
bool OcarinaRender360On() {
    return Settings::OcarinaRenderHeight() == u32{360};
}
constexpr std::array RENDER_540_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x806420,
                       .bytes = RENDER_SCALE_540,
                       .checked = true,
                       .original = RENDER_SCALE_ONE},
};
bool OcarinaRender540On() {
    return Settings::OcarinaRenderHeight() == u32{540};
}
constexpr std::array RENDER_720_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x806420,
                       .bytes = RENDER_SCALE_720,
                       .checked = true,
                       .original = RENDER_SCALE_ONE},
};
bool OcarinaRender720On() {
    return Settings::OcarinaRenderHeight() == u32{720};
}
constexpr std::array RENDER_810_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x806420,
                       .bytes = RENDER_SCALE_810,
                       .checked = true,
                       .original = RENDER_SCALE_ONE},
};
bool OcarinaRender810On() {
    return Settings::OcarinaRenderHeight() == u32{810};
}
constexpr std::array RENDER_900_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x806420,
                       .bytes = RENDER_SCALE_900,
                       .checked = true,
                       .original = RENDER_SCALE_ONE},
};
bool OcarinaRender900On() {
    return Settings::OcarinaRenderHeight() == u32{900};
}
constexpr std::array RENDER_1080_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x806420,
                       .bytes = RENDER_SCALE_1080,
                       .checked = true,
                       .original = RENDER_SCALE_ONE},
};
bool OcarinaRender1080On() {
    return Settings::OcarinaRenderHeight() == u32{1080};
}
constexpr std::array RENDER_1260_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x806420,
                       .bytes = RENDER_SCALE_1260,
                       .checked = true,
                       .original = RENDER_SCALE_ONE},
};
bool OcarinaRender1260On() {
    return Settings::OcarinaRenderHeight() == u32{1260};
}

// Texture pool patch (a ported community patch): with Max Resolution the
// render targets outgrow the graphics allocator's initial pool and the Castle Town guard house
// (the pot room) dereferences a null allocation and freezes. main+0xC2A36C (stp x8, x9,
// [sp, #0x10]) stores the initial pool size (x8); it now calls a routine that doubles x8 first
// (lsl x8, x8, #1 ; stp x8, x9, [sp, #0x10] ; ret). The original routine sits at main+0x3C462C0,
// inside the Hyrule Cheat Menu mod's code (main+0x3C46280..0x3C46398); ours goes after our own
// cave on the same executable page (main+0x3C46F10, the page runs to main+0x3C47000), so the
// call is re-encoded for it (bl +0x301CBA4).
constexpr size_t TEXTURE_POOL_ROUTINE_OFFSET = 0x3C46F10;
constexpr std::array<u8, 4> TEXTURE_POOL_CALL{0xE9, 0x72, 0xC0, 0x94};
constexpr std::array<u8, 12> TEXTURE_POOL_ROUTINE{
    0x08, 0xF9, 0x7F, 0xD3, 0xE8, 0x27, 0x01, 0xA9, 0xC0, 0x03, 0x5F, 0xD6,
};
constexpr std::array TEXTURE_POOL_RECORDS{
    BuiltinPatchRecord{.module_offset = 0xC2A36C,
                       .bytes = TEXTURE_POOL_CALL,
                       .checked = true,
                       .original = 0xA90127E8}, // stp x8, x9, [sp, #0x10]
    BuiltinPatchRecord{.module_offset = TEXTURE_POOL_ROUTINE_OFFSET, .bytes = TEXTURE_POOL_ROUTINE},
};

// Dynamic FPS by Dark, bundled with his permission: the game's frame step follows
// the time measured with the system counter instead of assuming 30 fps, so it runs at full speed
// at any frame rate. A call (bl 0x3c46200) to a routine in unused space after the code.
constexpr std::array<u8, 4> DYNAMIC_FPS_CALL{0x4F, 0x7D, 0xD5, 0x94};
constexpr std::array<u8, 120> DYNAMIC_FPS_ROUTINE{
    0x28, 0x18, 0x20, 0x1E, 0xCA, 0x92, 0x00, 0xB0, 0x4A, 0x01, 0x38, 0x91,
    0x2B, 0xE0, 0x3B, 0xD5, 0x4C, 0x01, 0x40, 0xF9, 0x4B, 0x01, 0x00, 0xF9,
    0xCC, 0x02, 0x00, 0xB4, 0x6B, 0x01, 0x0C, 0xCB, 0x0D, 0xE0, 0x3B, 0xD5,
    0x8E, 0x02, 0x80, 0xD2, 0xAE, 0x09, 0xCE, 0x9A, 0x7F, 0x01, 0x0E, 0xEB,
    0xCB, 0x81, 0x8B, 0x9A, 0x4C, 0x05, 0x40, 0xF9, 0x4C, 0x00, 0x00, 0xB5,
    0xEC, 0x03, 0x0B, 0xAA, 0x6E, 0x01, 0x0C, 0xCB, 0x8C, 0x0D, 0x8E, 0x8B,
    0x4C, 0x05, 0x00, 0xF9, 0x82, 0x01, 0x23, 0x9E, 0xA3, 0x01, 0x23, 0x9E,
    0x42, 0x18, 0x23, 0x1E, 0x42, 0x08, 0x21, 0x1E, 0xC3, 0x00, 0x00, 0x1C,
    0x03, 0x09, 0x23, 0x1E, 0x40, 0x20, 0x23, 0x1E, 0x4D, 0x00, 0x00, 0x54,
    0x48, 0x40, 0x20, 0x1E, 0xC0, 0x03, 0x5F, 0xD6, 0x0A, 0xD7, 0x83, 0x3F,
};

// High refresh rate (experimental, Settings oot_high_refresh): Dark's routine with the tail of
// The "FPS Unlock" patch (a ported community patch): the smoothed real-time step is always
// returned, also below 1.0, so the game keeps real speed above 60 fps (0.5 a frame at 120).
// Through the fmul at +0x58 it is Dark's routine byte for byte (the 50 ms delta cap included;
// the ported patch raised it to 100 ms). The tail clamps the step to [0.02, 3.0] (3.0 = that cap; the ported patch: 6.0) and multiplies it
// by a speed scale word that the present pacing sets from the speed limit (Turbo, Slow; 1.0 at
// 100%), since pacing alone no longer changes the game's speed. The first call (no earlier
// tick) still returns the game's own step from +0x70 (the head's cbz target).
//   +0x5C ldr s3, +0x80 ; fmaxnm s2, s2, s3 ; ldr s3, +0x84 ; fminnm s2, s2, s3 ; b +0x74
//   +0x70 ret
//   +0x74 ldr s3, +0x88 ; fmul s8, s2, s3 ; ret
//   +0x80 0.02f ; 3.0f ; speed scale (1.0f)
constexpr std::array<u8, 140> HIGH_REFRESH_ROUTINE{
    0x28, 0x18, 0x20, 0x1E, 0xCA, 0x92, 0x00, 0xB0, 0x4A, 0x01, 0x38, 0x91,
    0x2B, 0xE0, 0x3B, 0xD5, 0x4C, 0x01, 0x40, 0xF9, 0x4B, 0x01, 0x00, 0xF9,
    0xCC, 0x02, 0x00, 0xB4, 0x6B, 0x01, 0x0C, 0xCB, 0x0D, 0xE0, 0x3B, 0xD5,
    0x8E, 0x02, 0x80, 0xD2, 0xAE, 0x09, 0xCE, 0x9A, 0x7F, 0x01, 0x0E, 0xEB,
    0xCB, 0x81, 0x8B, 0x9A, 0x4C, 0x05, 0x40, 0xF9, 0x4C, 0x00, 0x00, 0xB5,
    0xEC, 0x03, 0x0B, 0xAA, 0x6E, 0x01, 0x0C, 0xCB, 0x8C, 0x0D, 0x8E, 0x8B,
    0x4C, 0x05, 0x00, 0xF9, 0x82, 0x01, 0x23, 0x9E, 0xA3, 0x01, 0x23, 0x9E,
    0x42, 0x18, 0x23, 0x1E, 0x42, 0x08, 0x21, 0x1E, 0x23, 0x01, 0x00, 0x1C,
    0x42, 0x68, 0x23, 0x1E, 0x03, 0x01, 0x00, 0x1C, 0x42, 0x78, 0x23, 0x1E,
    0x02, 0x00, 0x00, 0x14, 0xC0, 0x03, 0x5F, 0xD6, 0xA3, 0x00, 0x00, 0x1C,
    0x48, 0x08, 0x23, 0x1E, 0xC0, 0x03, 0x5F, 0xD6, 0x0A, 0xD7, 0xA3, 0x3C,
    0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x80, 0x3F,
};
// Its own cave at the free end of the page Dark's routine is on (the page runs to main+0x3C47000),
// since other mods sit right after Dark's 120 bytes (the Hyrule Cheat Menu at main+0x3C46280..
// 0x3C46398, 2026-11-10: it shared our old spot after Dark's, so Ctrl+U refused with it on). The
// same page: its adrp to Dark's timer state is the same word; only the call to it is re-encoded.
constexpr size_t HIGH_REFRESH_ROUTINE_OFFSET = 0x3C46E00;
constexpr std::array<u8, 4> HIGH_REFRESH_CALL{0x4F, 0x80, 0xD5, 0x94}; // bl 0x3c46e00
constexpr size_t HIGH_REFRESH_SCALE = 0x88; // the speed scale word, in the routine

// Water ripples at any frame rate: a port of Blitchii PR #1180 (480e01394 and its review
// follow-up), found and verified live by a tester. The water renderer (main+0x472E0) runs
// iterations * trunc(step) ripple passes a frame (w8 = 6 from ldr w8, [x19, #0x4c]; fcvtzs w9, s0
// at main+0x475F8; mul w8, w8, w9 at main+0x4760C). Below a step of 1 some or all frames run no
// pass: the ripples freeze, and the grid that follows Link moves without a pass to apply the
// shift, so the surface jumps (on camera turns above 60 fps). Our earlier accumulator returned
// trunc(step + rest), which kept the average rate but still ran 0 and 6 passes in turn at a step
// of 0.5. Now the fcvtzs calls a routine (bl 0x3c46f20) and the mul is a nop: the routine adds
// iterations * step to a rest kept per water object (x21), runs the rounded whole passes, at least
// one while time moves (none while the step is 0: paused), and keeps total - passes, never below
// -0.5. At a step of 0.5: 3 passes every frame.
// In: w8 iterations, s0 step, x21 the water object. Out: w8 passes. Clobbers x9-x12, s0-s2, the
// flags and x30 (the bl). Checked on the game's code (2026-11-17): the function saves x30 in its
// prologue (stp x29, x30, [sp, #-0x60]! at main+0x472E4) and reloads it before its retab, makes
// calls of its own, and on no path from the call to its return is x9-x12, x30 or a flag read
// before it is written; s0-s2 are only ever passed into calls that every loop around them already
// clobbers (our earlier accumulator changed s0 and s1 too).
//   +0x00 s0 = w8 * s0 ; s0 <= 0: +0x8C (w8 = 0 ; ret)
//   +0x10 x9 = 0x4e9fe20 (8 slots of {u64 object, f32 rest}), x10 = 0x4e9fea0 (next slot to take)
//   +0x20 find x21's slot; none (+0x38): take slot [next], object = x21, rest = 0,
//         next = (next + 1) & 7
//   +0x54 s0 += rest ; w8 = max(fcvtzs(s0 + 0.5), 1) ; rest = max(s0 - w8, -0.5) ; ret
// The routine is the port's byte for byte, but for the slot base (add x9, x9, #0xe20 at +0x14;
// there #0xe10): the slots stay where our accumulator had them, after Dark's timer state.
constexpr size_t WATER_HOOK_OFFSET = 0x475F8;
constexpr u32 WATER_FCVTZS = 0x1E380009;                            // fcvtzs w9, s0
constexpr std::array<u8, 4> WATER_ORIGINAL{0x09, 0x00, 0x38, 0x1E}; // fcvtzs w9, s0
constexpr std::array<u8, 4> WATER_HOOK{0x4A, 0xFE, 0xEF, 0x94};     // bl 0x3c46f20
constexpr size_t WATER_MUL_OFFSET = 0x4760C;
constexpr u32 WATER_MUL = 0x1B097D08;                                   // mul w8, w8, w9
constexpr std::array<u8, 4> WATER_MUL_ORIGINAL{0x08, 0x7D, 0x09, 0x1B}; // mul w8, w8, w9
constexpr std::array<u8, 4> WATER_MUL_NOP{0x1F, 0x20, 0x03, 0xD5};      // nop
constexpr size_t WATER_ROUTINE_OFFSET = 0x3C46F20;
constexpr std::array<u8, 148> WATER_ROUTINE{
    0x01, 0x01, 0x22, 0x1E, 0x00, 0x08, 0x21, 0x1E, 0x08, 0x20, 0x20, 0x1E, 0x0D, 0x04, 0x00, 0x54,
    0xC9, 0x92, 0x00, 0xB0, 0x29, 0x81, 0x38, 0x91, 0x2A, 0x01, 0x02, 0x91, 0xEB, 0x03, 0x09, 0xAA,
    0x6C, 0x01, 0x40, 0xF9, 0x9F, 0x01, 0x15, 0xEB, 0x60, 0x01, 0x00, 0x54, 0x6B, 0x41, 0x00, 0x91,
    0x7F, 0x01, 0x0A, 0xEB, 0x61, 0xFF, 0xFF, 0x54, 0x4C, 0x01, 0x40, 0xB9, 0x2B, 0x51, 0x2C, 0x8B,
    0x75, 0x01, 0x00, 0xF9, 0x7F, 0x09, 0x00, 0xB9, 0x8C, 0x05, 0x00, 0x11, 0x8C, 0x09, 0x00, 0x12,
    0x4C, 0x01, 0x00, 0xB9, 0x61, 0x09, 0x40, 0xBD, 0x00, 0x28, 0x21, 0x1E, 0x01, 0x10, 0x2C, 0x1E,
    0x02, 0x28, 0x21, 0x1E, 0x48, 0x00, 0x38, 0x1E, 0x1F, 0x05, 0x00, 0x71, 0x4A, 0x00, 0x00, 0x54,
    0x28, 0x00, 0x80, 0x52, 0x01, 0x01, 0x22, 0x1E, 0x00, 0x38, 0x21, 0x1E, 0x01, 0x10, 0x3C, 0x1E,
    0x00, 0x48, 0x21, 0x1E, 0x60, 0x09, 0x00, 0xBD, 0xC0, 0x03, 0x5F, 0xD6, 0xE8, 0x03, 0x1F, 0x2A,
    0xC0, 0x03, 0x5F, 0xD6,
};
// The routine's place with the ripples off: free.
constexpr std::array<u8, WATER_ROUTINE.size()> FREE_WATER_ROUTINE{};
constexpr size_t WATER_SLOTS_OFFSET = 0x4E9FE20;
constexpr size_t WATER_NEXT_OFFSET = 0x4E9FEA0; // u32, the next slot to take
// Dark's timer state (+0x00..+0x10), then the ripple slots (+0x20..+0xA0) and the next-slot word
// (+0xA0; +0x10 held our earlier accumulator's, unused now).
constexpr size_t HIGH_REFRESH_STATE_OFFSET = 0x4E9FE00;
constexpr size_t HIGH_REFRESH_STATE_SIZE = 0xB0;

// What the encodings say, checked against the offsets they are meant for.
constexpr u32 InstructionAt(std::span<const u8> bytes, size_t offset) {
    return u32{bytes[offset]} | u32{bytes[offset + 1]} << 8 | u32{bytes[offset + 2]} << 16 |
           u32{bytes[offset + 3]} << 24;
}
constexpr u32 EncodeBl(size_t from, size_t to) {
    return 0x94000000U |
           (static_cast<u32>((static_cast<s64>(to) - static_cast<s64>(from)) / 4) & 0x03FFFFFFU);
}
constexpr size_t AdrpTarget(size_t pc, u32 word) {
    const s64 pages = static_cast<s64>((((word >> 5) & 0x7FFFF) << 2) | ((word >> 29) & 3));
    const s64 signed_pages = pages >= (s64{1} << 20) ? pages - (s64{1} << 21) : pages;
    return static_cast<size_t>(static_cast<s64>(pc & ~size_t{0xFFF}) + signed_pages * 0x1000);
}
constexpr size_t AddImmediate(u32 word) { // add xd, xn, #imm (shift 0)
    return (word >> 10) & 0xFFF;
}
static_assert(InstructionAt(WATER_HOOK, 0) == EncodeBl(WATER_HOOK_OFFSET, WATER_ROUTINE_OFFSET));
static_assert(InstructionAt(WATER_ORIGINAL, 0) == WATER_FCVTZS);
static_assert(InstructionAt(WATER_MUL_ORIGINAL, 0) == WATER_MUL);
// The routine's only position-dependent word is its adrp (x9 = page 0x4e9f000 from any spot on
// the page 0x3c46000); its branches are relative and stay inside it.
static_assert(InstructionAt(WATER_ROUTINE, 0x10) == 0xB00092C9 && // adrp x9, ...
              InstructionAt(WATER_ROUTINE, 0x14) == 0x91388129 && // add x9, x9, #0xe20
              InstructionAt(WATER_ROUTINE, 0x18) == 0x9102012A);  // add x10, x9, #0x80
static_assert(AdrpTarget(WATER_ROUTINE_OFFSET + 0x10, InstructionAt(WATER_ROUTINE, 0x10)) +
                  AddImmediate(InstructionAt(WATER_ROUTINE, 0x14)) ==
              WATER_SLOTS_OFFSET);
static_assert(WATER_SLOTS_OFFSET + AddImmediate(InstructionAt(WATER_ROUTINE, 0x18)) ==
              WATER_NEXT_OFFSET);
static_assert(WATER_NEXT_OFFSET == WATER_SLOTS_OFFSET + 8 * 16);
// The slots after Dark's 16 bytes, the next-slot word inside the state.
static_assert(WATER_SLOTS_OFFSET >= HIGH_REFRESH_STATE_OFFSET + 0x10 &&
              WATER_NEXT_OFFSET + 4 <= HIGH_REFRESH_STATE_OFFSET + HIGH_REFRESH_STATE_SIZE);

// Our cave (main+0x3C46E00, executable padding after the end of the code: the text page runs on to
// main+0x3C47000): the high refresh routine. Then, on the same page, the texture pool routine
// (main+0x3C46F10..0x3C46F1C) and the water ripple routine (main+0x3C46F20..0x3C46FB4). Every word
// of ours must be free (zero) in the module.
constexpr size_t OOT_CAVE_SIZE = HIGH_REFRESH_ROUTINE.size();
static_assert(HIGH_REFRESH_ROUTINE_OFFSET + OOT_CAVE_SIZE <= TEXTURE_POOL_ROUTINE_OFFSET);
static_assert(TEXTURE_POOL_ROUTINE_OFFSET + TEXTURE_POOL_ROUTINE.size() <= WATER_ROUTINE_OFFSET);
static_assert(WATER_ROUTINE_OFFSET + WATER_ROUTINE.size() <= 0x3C47000);
static_assert(OOT_CAVE_SIZE % 4 == 0 && WATER_ROUTINE.size() % 4 == 0);
// Our cave with the high refresh mode off: free.
constexpr std::array<u8, OOT_CAVE_SIZE> FREE_CAVE{};
static_assert(OotHighRefreshCode::CallOffset == 0x6E6CC4 &&
              OotHighRefreshCode::WaterOffset == WATER_HOOK_OFFSET &&
              OotHighRefreshCode::WaterMulOffset == WATER_MUL_OFFSET &&
              OotHighRefreshCode::DarkOffset == 0x3C46200 &&
              OotHighRefreshCode::DarkSize == DYNAMIC_FPS_ROUTINE.size() &&
              OotHighRefreshCode::CaveOffset == HIGH_REFRESH_ROUTINE_OFFSET &&
              OotHighRefreshCode::CaveSize == OOT_CAVE_SIZE &&
              OotHighRefreshCode::RippleOffset == WATER_ROUTINE_OFFSET &&
              OotHighRefreshCode::RippleSize == WATER_ROUTINE.size() &&
              OotHighRefreshCode::ScaleOffset == HIGH_REFRESH_ROUTINE_OFFSET + HIGH_REFRESH_SCALE);

// The call (to our routine or to Dark's), Dark's routine (so the mode can switch live), the water
// hook (over its fcvtzs) and nop (over its mul), then our cave and the ripple routine one word at
// a time, each checked to be zero in the module.
// Since v5.8 (a ported finding, 2026-11-10) Dark's mode gets the same hook, nop, cave and ripple
// routine too, through the separate ripple built-in: with his Dynamic FPS alone the ripples ran
// trunc(step) passes, so at a 60 fps cap (step 0.998-1.003) some frames ran none and below 60
// they ran slow (1.33 -> 1). With both, the call word is the only difference between the modes.
constexpr auto MakeOotRecords(std::span<const u8> call) {
    std::array<BuiltinPatchRecord, 4 + OOT_CAVE_SIZE / 4 + WATER_ROUTINE.size() / 4> records{};
    records[0] = BuiltinPatchRecord{.module_offset = 0x6E6CC4, .bytes = call};
    records[1] = BuiltinPatchRecord{.module_offset = 0x3C46200, .bytes = DYNAMIC_FPS_ROUTINE};
    records[2] = BuiltinPatchRecord{.module_offset = WATER_HOOK_OFFSET,
                                    .bytes = WATER_HOOK,
                                    .checked = true,
                                    .original = WATER_FCVTZS};
    records[3] = BuiltinPatchRecord{.module_offset = WATER_MUL_OFFSET,
                                    .bytes = WATER_MUL_NOP,
                                    .checked = true,
                                    .original = WATER_MUL};
    size_t i = 4;
    for (size_t offset = 0; offset < OOT_CAVE_SIZE; ++i, offset += 4) {
        records[i] = BuiltinPatchRecord{.module_offset = HIGH_REFRESH_ROUTINE_OFFSET + offset,
                                        .bytes = std::span<const u8>{HIGH_REFRESH_ROUTINE}.subspan(offset, 4),
                                        .checked = true,
                                        .original = 0};
    }
    for (size_t offset = 0; offset < WATER_ROUTINE.size(); ++i, offset += 4) {
        records[i] = BuiltinPatchRecord{.module_offset = WATER_ROUTINE_OFFSET + offset,
                                        .bytes = std::span<const u8>{WATER_ROUTINE}.subspan(offset, 4),
                                        .checked = true,
                                        .original = 0};
    }
    return records;
}
constexpr auto HIGH_REFRESH_RECORDS = MakeOotRecords(HIGH_REFRESH_CALL);
// Dark's Dynamic FPS: his call and routine only, as always (IsBuiltinDynamicFpsMod tells his
// standalone mod by these two records).
constexpr std::array DYNAMIC_FPS_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x6E6CC4, .bytes = DYNAMIC_FPS_CALL},
    BuiltinPatchRecord{.module_offset = 0x3C46200, .bytes = DYNAMIC_FPS_ROUTINE},
};
// The water ripples with Dark's Dynamic FPS (its own built-in, applied after his): the water hook
// and nop, our cave and the ripple routine, the same words the high refresh mode has, without its
// call. Skipped when the high refresh mode already wrote them, or when another mod holds any of
// those spots.
constexpr auto WATER_RIPPLE_RECORDS = [] {
    std::array<BuiltinPatchRecord, 2 + OOT_CAVE_SIZE / 4 + WATER_ROUTINE.size() / 4> records{};
    const auto all = MakeOotRecords(DYNAMIC_FPS_CALL);
    std::ranges::copy(std::span{all}.subspan(2), records.begin());
    return records;
}();
constexpr std::string_view WATER_RIPPLE_NAME = "Water ripples at any frame rate (built-in)";
constexpr std::string_view DYNAMIC_FPS_NAME = "Dynamic FPS by Dark (built-in)";
constexpr std::string_view HIGH_REFRESH_NAME = "High refresh rate (built-in, experimental)";
// The mod that took the high refresh patch's place at the last boot (OotHighRefreshBlockedBy).
std::mutex g_high_refresh_blocked_mutex;
std::string g_high_refresh_blocked_by;

// Pickup counter above 60 fps: the number shown above Link after a pickup (rupees, sticks...)
// counts toward the wallet in the indicator's update (main+0xA67E50): shown (+0x360) moves toward
// the target (+0x35C) by trunc(step * max(distance / 50, 1)) (fcvtzs w12, s0 at main+0xA68300), and
// the indicator fades only once they meet. In the high refresh mode the step is under 1.0 above
// 60 fps, so the cut is 0 every frame and the count never moves: the indicator stays up until a
// door or a talk replaces it. That fcvtzs branches to a stub at main+0x3C46B00 that keeps the
// fraction per counter (4 slots of {u64 counter address, f32 rest} at main+0x4E9FEB0 + 0x10, a u32
// round-robin index at +0) and returns the whole units in it; a new slot returns at least 1. It
// does that only while the frame step calls the high refresh routine (the word at main+0x6E6CC4
// is HIGH_REFRESH_CALL, read at each call, so Ctrl+U's live switch is followed); otherwise it
// returns the game's own fcvtzs, so Dark's mode is unchanged. The same bug and fix as the
// "pickup window count-up" of zortos293/Lichii#1587 (by atk3001) and NotRexed/Hookshot#2006.
//   +0x00 save x13-x16, d16-d17 ; w12 = fcvtzs(s0)
//   +0x10 adrp x13, 0x6e6000 ; ldr w13, [x13, #0xcc4] ; w14 = 0x94d5804f ; cmp ; b.ne done
//   +0x28 fcmp s0, #0.0 ; b.le done ; x14 = x19 + 0x360 ; x13 = 0x4e9feb0 ; find x14's slot
//   +0x5C (miss) take slot [index], index = (index + 1) & 3, key = x14, rest = 0,
//         w12 = max(fcvtzs(s0), 1)
//   +0x88 (hit) rest += s0 ; w12 = fcvtzs(rest) ; rest -= w12
//   +0xA0 done: restore ; b 0xa68304
constexpr size_t PICKUP_HOOK_OFFSET = 0xA68300;
constexpr u32 PICKUP_FCVTZS = 0x1E38000C;                         // fcvtzs w12, s0
constexpr std::array<u8, 4> PICKUP_HOOK{0x00, 0x7A, 0xC7, 0x14}; // b 0x3c46b00
constexpr size_t PICKUP_STUB_OFFSET = 0x3C46B00;
constexpr std::array<u8, 176> PICKUP_STUB{
    0xED, 0x3B, 0xBD, 0xA9, 0xEF, 0x43, 0x01, 0xA9, 0xF0, 0x47, 0x02, 0x6D,
    0x0C, 0x00, 0x38, 0x1E, 0x0D, 0x55, 0xFE, 0x90, 0xAD, 0xC5, 0x4C, 0xB9,
    0xEE, 0x09, 0x90, 0x52, 0xAE, 0x9A, 0xB2, 0x72, 0xBF, 0x01, 0x0E, 0x6B,
    0xE1, 0x03, 0x00, 0x54, 0x08, 0x20, 0x20, 0x1E, 0xAD, 0x03, 0x00, 0x54,
    0x6E, 0x82, 0x0D, 0x91, 0xCD, 0x92, 0x00, 0xB0, 0xAD, 0xC1, 0x3A, 0x91,
    0xAF, 0x41, 0x00, 0x91, 0x90, 0x00, 0x80, 0x52, 0xEC, 0x01, 0x40, 0xF9,
    0x9F, 0x01, 0x0E, 0xEB, 0xE0, 0x01, 0x00, 0x54, 0xEF, 0x41, 0x00, 0x91,
    0x10, 0x06, 0x00, 0x71, 0x61, 0xFF, 0xFF, 0x54, 0xB0, 0x01, 0x40, 0xB9,
    0x0C, 0x06, 0x00, 0x11, 0x8C, 0x05, 0x00, 0x12, 0xAC, 0x01, 0x00, 0xB9,
    0xAF, 0x51, 0x30, 0x8B, 0xEE, 0x09, 0x00, 0xF9, 0xFF, 0x19, 0x00, 0xB9,
    0x0C, 0x00, 0x38, 0x1E, 0x9F, 0x05, 0x00, 0x71, 0x8C, 0xA5, 0x9F, 0x1A,
    0x07, 0x00, 0x00, 0x14, 0xF0, 0x09, 0x40, 0xBD, 0x10, 0x2A, 0x20, 0x1E,
    0x0C, 0x02, 0x38, 0x1E, 0x91, 0x01, 0x22, 0x1E, 0x10, 0x3A, 0x31, 0x1E,
    0xF0, 0x09, 0x00, 0xBD, 0xF0, 0x47, 0x42, 0x6D, 0xEF, 0x43, 0x41, 0xA9,
    0xED, 0x3B, 0xC3, 0xA8, 0xD6, 0x85, 0x38, 0x17, // +0xAC b 0xa68304
};
// Its slots: right after Dark's timer state, the ripple slots and their next-slot word
// (+0x00..+0xB0). The BSS of this main runs to main+0x4EA0000.
constexpr size_t PICKUP_STATE_OFFSET = 0x4E9FEB0;
constexpr size_t PICKUP_STATE_SIZE = 0x50;
static_assert(PICKUP_STATE_OFFSET >= HIGH_REFRESH_STATE_OFFSET + HIGH_REFRESH_STATE_SIZE);
// Free padding on the page of Dark's routine, before the spots other mods are known to use
// (a community ultrawide mod from main+0x3C46C00) and our own shadow steps (main+0x3C46DC0).
static_assert(PICKUP_STUB_OFFSET + PICKUP_STUB.size() <= 0x3C46C00);
// The hook over its fcvtzs, then the stub one word at a time, each checked to be zero.
constexpr auto PICKUP_COUNTER_RECORDS = [] {
    std::array<BuiltinPatchRecord, 1 + PICKUP_STUB.size() / 4> records{};
    records[0] = BuiltinPatchRecord{.module_offset = PICKUP_HOOK_OFFSET,
                                    .bytes = PICKUP_HOOK,
                                    .checked = true,
                                    .original = PICKUP_FCVTZS};
    for (size_t i = 1, offset = 0; offset < PICKUP_STUB.size(); ++i, offset += 4) {
        records[i] = BuiltinPatchRecord{.module_offset = PICKUP_STUB_OFFSET + offset,
                                        .bytes = std::span<const u8>{PICKUP_STUB}.subspan(offset, 4),
                                        .checked = true,
                                        .original = 0};
    }
    return records;
}();
constexpr std::string_view PICKUP_COUNTER_NAME = "Pickup counter at any frame rate (built-in)";

// High refresh fixes (ported findings, 2026-11-16): game code that counts frames, or cuts the step
// to whole units, goes wrong once the high refresh mode's step drops below 1.0 above 60 fps. Each
// hook is a branch to a stub in the free padding on the page of Dark's routine (from
// main+0x3C463A0: after the Hyrule Cheat Menu's main+0x3C46280..0x3C46398, before the pickup stub
// at main+0x3C46B00). Like the pickup counter, every stub first reads the frame step's call word
// (main+0x6E6CC4) and changes nothing unless it is HIGH_REFRESH_CALL: otherwise it runs the game's
// own instruction and branches back, so Dark's mode is unchanged and Ctrl+U's live switch is
// followed. The gate (32 bytes, x16/x17 saved on the stack):
//   stp x16, x17, [sp, #-16]! ; adrp x16, 0x6e6000 ; ldr w16, [x16, #0xcc4] ;
//   mov w17, #0x804f ; movk w17, #0x94d5, lsl #16 ; cmp w16, w17 ; ldp x16, x17, [sp], #16 ;
//   b.ne <the game's own instruction>
// The stubs keep every register but NZCV, which is dead at each hook (the next flag use sets it
// first). The game's timer is T = [main+0x4E94B20] (the step at T+0x10, step / 60 at T+0x18).
// State: BSS at main+0x4E9EE00, on a page of its own below the pickup slots (the BSS of this main
// runs to main+0x4EA0000; the game's own variables end below main+0x4E9B000). Not after the pickup
// slots (main+0x4E9FF00, until 2026-11-16): tomvita's cheats for this build keep flags at
// main+0x4E9FF80..0x4E9FF94 and 0x4E9FFA0..0x4E9FFB0, inside the step slots (zortos293/Lichii#1981
// found it; no mod or cheat it surveyed writes main+0x4E9EE00..0x4E9EF10). Assembled with llvm-mc
// and ld.lld; the state's adrp/add pairs (SOUND_GATE +0x20, STEP_TICKS +0x4 and +0x2C) re-encoded
// by script.
struct OotFixHook {
    size_t offset;
    std::array<u8, 4> bytes;
    u32 original;
    const char* skip_env = nullptr; // =1 leaves this hook out (BuiltinPatchRecord::skip_env)
};
struct OotFixStub {
    size_t offset;
    std::span<const u8> bytes;
};
constexpr size_t HR_FIX_CAVE_START = 0x3C463A0;
constexpr size_t HR_FIX_STATE_OFFSET = 0x4E9EE00;

// Sound at 60 Hz: the sound engine counts its fades, delays and ramps in whole 60 fps steps per
// update. Below a whole step some sounds are never released when an area unloads; they fill the
// voice budget, and after a few loading zones most new sounds and the music stop until a restart.
// The sound manager job (main+0x415300, phase in w2: 0 reads the step and updates the engine, 1
// follows) enters through SOUND_GATE: in phase 0 a step of 1 or more runs it as before, a smaller
// one is added to an accumulator (state +0) and the job runs only once a whole step has built up
// (the rest kept), else returns; phase 1 follows that frame's choice (flag at state +4). The job's
// step load (main+0x415334) goes through SOUND_FEED, max(step, 1.0), so the engine always gets
// whole steps and sound keeps real time.
//   gate ; x16 = state ; cbnz w2, other ; s16 = step ; s17 = 1.0 ; step >= 1: every
//   s18 = [x16] + step ; s18 < 1: wait ; [x16] = s18 - 1 ; every: [x16 + 4] = 1 ; b run
//   wait: [x16] = s18 ; [x16 + 4] = 0 ; ret ; other: [x16 + 4] == 0: ret
//   run: pacibsp ; b 0x415304
// SOUND_FEED: ldr s0, [x8, #0x10] ; gate ; q16 saved ; fmaxnm s0, s0, 1.0 ; b 0x415338
constexpr std::array<u8, 136> SOUND_GATE{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x01, 0x03, 0x00, 0x54, 0xD0, 0x92, 0x00, 0x90,
    0x10, 0x02, 0x38, 0x91, 0x62, 0x02, 0x00, 0x35, 0x69, 0x92, 0x00, 0xD0,
    0x29, 0x91, 0x45, 0xF9, 0x30, 0x11, 0x40, 0xBD, 0x11, 0x10, 0x2E, 0x1E,
    0x00, 0x22, 0x31, 0x1E, 0xEA, 0x00, 0x00, 0x54, 0x12, 0x02, 0x40, 0xBD,
    0x52, 0x2A, 0x30, 0x1E, 0x40, 0x22, 0x31, 0x1E, 0xCB, 0x00, 0x00, 0x54,
    0x52, 0x3A, 0x31, 0x1E, 0x12, 0x02, 0x00, 0xBD, 0x29, 0x00, 0x80, 0x52,
    0x09, 0x06, 0x00, 0xB9, 0x06, 0x00, 0x00, 0x14, 0x12, 0x02, 0x00, 0xBD,
    0x1F, 0x06, 0x00, 0xB9, 0xC0, 0x03, 0x5F, 0xD6, 0x09, 0x06, 0x40, 0xB9,
    0x69, 0x00, 0x00, 0x34, 0x7F, 0x23, 0x03, 0xD5, 0xB9, 0x3B, 0x1F, 0x17, // b 0x415304
    0xC0, 0x03, 0x5F, 0xD6,
};
constexpr std::array<u8, 56> SOUND_FEED{
    0x00, 0x11, 0x40, 0xBD, 0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90,
    0x10, 0xC6, 0x4C, 0xB9, 0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72,
    0x1F, 0x02, 0x11, 0x6B, 0xF0, 0x47, 0xC1, 0xA8, 0xA1, 0x00, 0x00, 0x54,
    0xF0, 0x0F, 0x9F, 0x3C, 0x10, 0x10, 0x2E, 0x1E, 0x00, 0x68, 0x30, 0x1E,
    0xF0, 0x07, 0xC1, 0x3C, 0x50, 0xE4, 0x6C, 0x17, // b 0x177f5a4 (HR_SOUND_COUNT, then 0x415338)
};

// Step counters: three counters cut the step or count frames. The ocarina playback tick adds
// trunc(step) (fcvtzu w8, s0 at main+0x3293410) and stalls below 1; the step rate manager's
// countdown takes max(round(step), 1) a frame (from fcvtzs s0, s8 at main+0x2F53268 to
// main+0x2F53294) and the ambient wave phase adds 1 a frame (add w26, w27, #1 at main+0x53F914),
// so both run fast. Each takes whole ticks of the step from STEP_TICKS instead, which keeps the
// rest per counter. (The pickup counter, the fourth such counter, has its own built-in.)
// STEP_TICKS: x13 = the counter's address, s0 = step -> w13 = whole ticks; uses x14-x16 and
// NZCV, keeps q0/q1. 15 slots of {u64 address, f32 rest, pad} at state +0x10 (to main+0x4E9EF00)
// and a u32 replacement index at state +8; a new or replaced slot returns max(trunc(step), 1).
//   find x13's slot (an empty one is taken) ; none: take [index], index = (index + 1) % 15
//   new: key = x13, rest = 0, w13 = max(fcvtzs(s0), 1) ; hit: rest += s0, w13 = trunc, rest -= w13
// The callers (after the gate) save x13-x16 and x30 and call it:
//   playback: x13 = x19 + x20 ; w8 = ticks ; b 0x3293414
//   countdown: x13 = x24 + 0x98, s0 = s8 ; w9 = ticks ; s0 = s8 - trunc(s8) (as the game leaves
//              it) ; w10 = [x24, #0x98] ; b 0x2f53294 (the words it skips are checked as well)
//   ambient: x13 = x28 + x1 * 4, s0 = [T + 0x10] ; w26 = w27 + ticks ; s0 = [x0, #0x18] ;
//            b 0x53f918
constexpr std::array<u8, 136> STEP_TICKS{
    0xE0, 0x07, 0xBF, 0xAD, 0xCE, 0x92, 0x00, 0x90, 0xCE, 0x41, 0x38, 0x91,
    0xEF, 0x01, 0x80, 0x52, 0xD0, 0x01, 0x40, 0xF9, 0x1F, 0x02, 0x0D, 0xEB,
    0x80, 0x02, 0x00, 0x54, 0xB0, 0x01, 0x00, 0xB4, 0xCE, 0x41, 0x00, 0x91,
    0xEF, 0x05, 0x00, 0x71, 0x41, 0xFF, 0xFF, 0x54, 0xCE, 0x92, 0x00, 0x90,
    0xCE, 0x21, 0x38, 0x91, 0xD0, 0x01, 0x40, 0xB9, 0x0F, 0x06, 0x00, 0x11,
    0xFF, 0x3D, 0x00, 0x71, 0xEF, 0x31, 0x9F, 0x1A, 0xCF, 0x01, 0x00, 0xB9,
    0xCE, 0x21, 0x00, 0x91, 0xCE, 0x11, 0x10, 0x8B, 0xCD, 0x01, 0x00, 0xF9,
    0xDF, 0x09, 0x00, 0xB9, 0x0D, 0x00, 0x38, 0x1E, 0xBF, 0x05, 0x00, 0x71,
    0xAD, 0xA5, 0x9F, 0x1A, 0x07, 0x00, 0x00, 0x14, 0xC1, 0x09, 0x40, 0xBD,
    0x00, 0x28, 0x21, 0x1E, 0x01, 0xC0, 0x25, 0x1E, 0x00, 0x38, 0x21, 0x1E,
    0xC0, 0x09, 0x00, 0xBD, 0x2D, 0x00, 0x38, 0x1E, 0xE0, 0x07, 0xC1, 0xAC,
    0xC0, 0x03, 0x5F, 0xD6,
};
constexpr std::array<u8, 80> STEP_REPLAY{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x61, 0x01, 0x00, 0x54, 0xED, 0x3B, 0xBD, 0xA9,
    0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x6D, 0x02, 0x14, 0x8B,
    0xD0, 0xFF, 0xFF, 0x97, 0xE8, 0x03, 0x0D, 0x2A, 0xFE, 0x13, 0x40, 0xF9,
    0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0xB4, 0x33, 0xD9, 0x17,
    0x08, 0x00, 0x39, 0x1E, 0xB2, 0x33, 0xD9, 0x17, // fcvtzu w8, s0 ; b 0x3293414
};
constexpr std::array<u8, 96> STEP_HITSTOP{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0xE1, 0x01, 0x00, 0x54, 0xED, 0x3B, 0xBD, 0xA9,
    0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x0D, 0x63, 0x02, 0x91,
    0x00, 0x41, 0x20, 0x1E, 0xBB, 0xFF, 0xFF, 0x97, 0xE9, 0x03, 0x0D, 0x2A,
    0xFE, 0x13, 0x40, 0xF9, 0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8,
    0x00, 0xC1, 0x25, 0x1E, 0x00, 0x39, 0x20, 0x1E, 0x0A, 0x9B, 0x40, 0xB9,
    0x3C, 0x33, 0xCC, 0x17, 0x00, 0xB9, 0xA1, 0x5E, 0x30, 0x33, 0xCC, 0x17, // fcvtzs s0, s8 ; b 0x2f5326c
};
constexpr std::array<u8, 96> STEP_AMBIENT{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0xE1, 0x01, 0x00, 0x54, 0xED, 0x3B, 0xBD, 0xA9,
    0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x8D, 0x0B, 0x01, 0x8B,
    0x6E, 0x92, 0x00, 0xD0, 0xCE, 0x91, 0x45, 0xF9, 0xC0, 0x11, 0x40, 0xBD,
    0xA1, 0xFF, 0xFF, 0x97, 0x7A, 0x03, 0x0D, 0x0B, 0x00, 0x18, 0x40, 0xBD,
    0xFE, 0x13, 0x40, 0xF9, 0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8,
    0xC5, 0xE4, 0x23, 0x17, 0x7A, 0x07, 0x00, 0x11, 0xC3, 0xE4, 0x23, 0x17, // add w26, w27, #1 ; b 0x53f918
};

// Moving platforms: the character controller takes a tenth of its ground's velocity once a frame
// rate value is over 100 (fcmp s0, s1 with 100.0 ; b.gt, at main+0x13A070, 0x13D710 and
// 0x389AC2C), so above 100 fps a moving platform slides away under the player. Each b.gt goes to
// a stub: not greater, back as before; greater, the gate: the high refresh mode goes on as if not
// greater, Dark's takes the branch. Both redo the fcmp first (s0 and s1 are untouched), so the
// flags are the game's. The fcmp before each hook is checked as well.
//   b.gt 1f ; b next ; 1: gate (else 2f) ; fcmp s0, s1 ; b next ; 2: fcmp s0, s1 ; b target
constexpr std::array<u8, 56> GROUND_A{
    0x4C, 0x00, 0x00, 0x54, 0x98, 0xCE, 0x13, 0x17, 0xF0, 0x47, 0xBF, 0xA9,
    0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9, 0xF1, 0x09, 0x90, 0x52,
    0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B, 0xF0, 0x47, 0xC1, 0xA8,
    0x61, 0x00, 0x00, 0x54, 0x00, 0x20, 0x21, 0x1E, 0x8E, 0xCE, 0x13, 0x17,
    0x00, 0x20, 0x21, 0x1E, 0x56, 0xCF, 0x13, 0x17, // b 0x13a39c
};
constexpr std::array<u8, 56> GROUND_B{
    0x4C, 0x00, 0x00, 0x54, 0x30, 0xDC, 0x13, 0x17, 0xF0, 0x47, 0xBF, 0xA9,
    0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9, 0xF1, 0x09, 0x90, 0x52,
    0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B, 0xF0, 0x47, 0xC1, 0xA8,
    0x61, 0x00, 0x00, 0x54, 0x00, 0x20, 0x21, 0x1E, 0x26, 0xDC, 0x13, 0x17,
    0x00, 0x20, 0x21, 0x1E, 0x14, 0xDD, 0x13, 0x17, // b 0x13dad4
};
constexpr std::array<u8, 56> GROUND_C{
    0x4C, 0x00, 0x00, 0x54, 0x67, 0x51, 0xF1, 0x17, 0xF0, 0x47, 0xBF, 0xA9,
    0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9, 0xF1, 0x09, 0x90, 0x52,
    0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B, 0xF0, 0x47, 0xC1, 0xA8,
    0x61, 0x00, 0x00, 0x54, 0x00, 0x20, 0x21, 0x1E, 0x5D, 0x51, 0xF1, 0x17,
    0x00, 0x20, 0x21, 0x1E, 0x18, 0x52, 0xF1, 0x17, // b 0x389af24
};
// Turn speed (Gohma steers with it): the turn action at main+0x261E8E0 ends by clamping its turn
// speed, kept per 60 Hz frame, to the change it applied times the step (fmul s1, s0, s1 ;
// fmul s0, s0, s8 at main+0x261EB18, s8 = step, s1 = [sp, #0xc]). The change already carries the
// step, so below a step of 1 the speed decays by step^2 each frame and Gohma walks into her
// room's wall. High refresh, below a step of 1: the bounds are +-(change / step) instead.
//   gate (else dark) ; s1 = 1.0 ; s8 >= 1: orig ; s0 = s0 / s8 ; s1 = -s0 ; b 0x261eb20
//   orig: ldr s1, [sp, #0xc] ; dark: fmul s1, s0, s1 ; fmul s0, s0, s8 ; b 0x261eb20
constexpr std::array<u8, 72> TURN_SPEED{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x01, 0x01, 0x00, 0x54, 0x01, 0x10, 0x2E, 0x1E,
    0x00, 0x21, 0x21, 0x1E, 0x8A, 0x00, 0x00, 0x54, 0x00, 0x18, 0x28, 0x1E,
    0x01, 0x40, 0x21, 0x1E, 0x07, 0x61, 0xA7, 0x17, 0xE1, 0x0F, 0x40, 0xBD,
    0x01, 0x08, 0x21, 0x1E, 0x00, 0x08, 0x28, 0x1E, 0x03, 0x61, 0xA7, 0x17,
};

// Movie cutscenes: a movie player adds 1 to its count at +0x2F8 per update that shows a picture
// (add w8, w8, #1 at main+0x3B5C7C8), and a cutscene that plays a movie places its voices,
// overlays and its end by that count. Picture and sound follow the clock, so above 60 fps the
// cues run ahead of them (the sages' lines over each other). High refresh: the count adds the
// whole ticks of the step for its address (STEP_TICKS) instead. Dark's mode keeps the +1 (below
// 60 fps it drifts the other way, as before).
//   gate (else dark) ; save x13-x16, x30, q0 ; x13 = x20 + 0x2f8 ; s0 = [T + 0x10] ;
//   w8 += ticks ; restore ; b 0x3b5c7cc ; dark: add w8, w8, #1 ; b 0x3b5c7cc
constexpr std::array<u8, 100> MOVIE_FRAMES{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x01, 0x02, 0x00, 0x54, 0xED, 0x3B, 0xBD, 0xA9,
    0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0xE0, 0x0F, 0x9F, 0x3C,
    0x8D, 0xE2, 0x0B, 0x91, 0x6E, 0x92, 0x00, 0xD0, 0xCE, 0x91, 0x45, 0xF9,
    0xC0, 0x11, 0x40, 0xBD, 0x44, 0xFF, 0xFF, 0x97, 0x08, 0x01, 0x0D, 0x0B,
    0xE0, 0x07, 0xC1, 0x3C, 0xFE, 0x13, 0x40, 0xF9, 0xEF, 0x43, 0x41, 0xA9,
    0xED, 0x3B, 0xC3, 0xA8, 0x15, 0x58, 0xFC, 0x17, 0x08, 0x05, 0x00, 0x11,
    0x13, 0x58, 0xFC, 0x17,
};

// Scarecrow's Song: the recording keeps each note's and gap's length in the elapsed counter the
// playback tick advances, and adds trunc(step) to it in two more places (fcvtzu w8, s0): while a
// note or gap is held (main+0x3296B08, the counter at x9) and when the last note is timed
// (main+0x32926C8, the counter at x20 + 0x5FC). Below a step of 1 the cut is 0, so the lengths
// come out zero and the song stops after its first note. High refresh: whole ticks of the step
// for that counter (STEP_TICKS), as the playback tick.
//   gate (else dark) ; save x13-x16, x30 ; x13 = x9 | x20 + 0x5fc ; w8 = ticks ; restore ;
//   b back ; dark: fcvtzu w8, s0 ; b back
constexpr std::array<u8, 80> SCARECROW_RECORD{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x61, 0x01, 0x00, 0x54, 0xED, 0x3B, 0xBD, 0xA9,
    0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0xED, 0x03, 0x09, 0xAA,
    0x2C, 0xFF, 0xFF, 0x97, 0xE8, 0x03, 0x0D, 0x2A, 0xFE, 0x13, 0x40, 0xF9,
    0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0xCE, 0x40, 0xD9, 0x17,
    0x08, 0x00, 0x39, 0x1E, 0xCC, 0x40, 0xD9, 0x17, // fcvtzu w8, s0 ; b 0x3296b0c
};
constexpr std::array<u8, 80> SCARECROW_LAST{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x61, 0x01, 0x00, 0x54, 0xED, 0x3B, 0xBD, 0xA9,
    0xEF, 0x43, 0x01, 0xA9, 0xFE, 0x13, 0x00, 0xF9, 0x8D, 0xF2, 0x17, 0x91,
    0x18, 0xFF, 0xFF, 0x97, 0xE8, 0x03, 0x0D, 0x2A, 0xFE, 0x13, 0x40, 0xF9,
    0xEF, 0x43, 0x41, 0xA9, 0xED, 0x3B, 0xC3, 0xA8, 0xAA, 0x2F, 0xD9, 0x17,
    0x08, 0x00, 0x39, 0x1E, 0xA8, 0x2F, 0xD9, 0x17, // fcvtzu w8, s0 ; b 0x32926cc
};

// Cloth and hair: the cloth update (clothes, hair, the wind action's phase) builds its time step
// as scale * 1/60 without the frame step, so above 60 fps every frame still simulates 1/60 s and
// cloth sways too fast (twice at 120 fps). Four sites load the 1/60 (0x3C888889): ldr s0,
// [x8, #0x66c] at main+0x14632C and 0x2C207A4, fmov s0, w8 at 0x2AB804, ldr s2, [x8, #0x66c] at
// 0x2C2208C. High refresh: min(T+0x18 (step / 60), 1/60), 1/60 when that is not positive (the
// bits compared as integers), so a step of 1 and more still gives the game's 1/60 exactly. One
// stub per site (no bl: x30 stays as it is), x16/x17 saved.
// Below 60 fps (a step over 1) it was the other way round: each frame simulated 1/60 s while the
// game moved on step / 60, so the cloth trailed and over-swung (Link's cap flared out after he
// stopped and took about a second to settle at 18 fps). Now, in both frame-step modes, the time is
// T+0x18 itself, at most 4/60: the gate's b.ne to the game's load is a nop and the bound is 4/60
// (one word each). A step of exactly 1 still gives 1/60 (the game's 60 fps value), and above 60 fps
// nothing changes. Both step routines keep the step in (0, 3] at normal speed, so the bound only
// limits Turbo; a step of 0 or less would also give 4/60, and neither routine returns one.
// FRITCHII_NO_CLOTH_SLOW=1 puts back the two words of each stub (HR_FIX_VARIANTS): the high
// refresh mode's min(T+0x18, 1/60) and Dark's mode's game load, as before.
//   gate (nop: always on) ; save x16, x17 ; w17 = [T + 0x18] ; w16 = 0x3d888888 ;
//   w17 = (w17 - 1 < w16 ? w17 - 1 : w16) + 1 ; fmov s0|s2, w17 ; restore ; b back
//   dark (unreached): the game's load ; b back
constexpr std::array<u8, 92> CLOTH_DT_A{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x1F, 0x20, 0x03, 0xD5, 0xF0, 0x47, 0xBF, 0xA9,
    0x70, 0x92, 0x00, 0xD0, 0x10, 0x92, 0x45, 0xF9, 0x11, 0x1A, 0x40, 0xB9,
    0x10, 0x11, 0x91, 0x52, 0x10, 0xB1, 0xA7, 0x72, 0x31, 0x06, 0x00, 0x51,
    0x3F, 0x02, 0x10, 0x6B, 0x31, 0x32, 0x90, 0x1A, 0x31, 0x06, 0x00, 0x11,
    0x20, 0x02, 0x27, 0x1E, 0xF0, 0x47, 0xC1, 0xA8, 0xAC, 0xFE, 0x13, 0x17,
    0x00, 0x6D, 0x46, 0xBD, 0xAA, 0xFE, 0x13, 0x17, // ldr s0, [x8, #0x66c] ; b 0x146330
};
constexpr std::array<u8, 92> CLOTH_DT_B{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x1F, 0x20, 0x03, 0xD5, 0xF0, 0x47, 0xBF, 0xA9,
    0x70, 0x92, 0x00, 0xD0, 0x10, 0x92, 0x45, 0xF9, 0x11, 0x1A, 0x40, 0xB9,
    0x10, 0x11, 0x91, 0x52, 0x10, 0xB1, 0xA7, 0x72, 0x31, 0x06, 0x00, 0x51,
    0x3F, 0x02, 0x10, 0x6B, 0x31, 0x32, 0x90, 0x1A, 0x31, 0x06, 0x00, 0x11,
    0x20, 0x02, 0x27, 0x1E, 0xF0, 0x47, 0xC1, 0xA8, 0xCA, 0x93, 0x19, 0x17,
    0x00, 0x01, 0x27, 0x1E, 0xC8, 0x93, 0x19, 0x17, // fmov s0, w8 ; b 0x2ab808
};
constexpr std::array<u8, 92> CLOTH_DT_C{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x1F, 0x20, 0x03, 0xD5, 0xF0, 0x47, 0xBF, 0xA9,
    0x70, 0x92, 0x00, 0xD0, 0x10, 0x92, 0x45, 0xF9, 0x11, 0x1A, 0x40, 0xB9,
    0x10, 0x11, 0x91, 0x52, 0x10, 0xB1, 0xA7, 0x72, 0x31, 0x06, 0x00, 0x51,
    0x3F, 0x02, 0x10, 0x6B, 0x31, 0x32, 0x90, 0x1A, 0x31, 0x06, 0x00, 0x11,
    0x20, 0x02, 0x27, 0x1E, 0xF0, 0x47, 0xC1, 0xA8, 0x9A, 0x67, 0xBF, 0x17,
    0x00, 0x6D, 0x46, 0xBD, 0x98, 0x67, 0xBF, 0x17, // ldr s0, [x8, #0x66c] ; b 0x2c207a8
};
constexpr std::array<u8, 92> CLOTH_DT_D{
    0xF0, 0x47, 0xBF, 0xA9, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xF0, 0x47, 0xC1, 0xA8, 0x1F, 0x20, 0x03, 0xD5, 0xF0, 0x47, 0xBF, 0xA9,
    0x70, 0x92, 0x00, 0xD0, 0x10, 0x92, 0x45, 0xF9, 0x11, 0x1A, 0x40, 0xB9,
    0x10, 0x11, 0x91, 0x52, 0x10, 0xB1, 0xA7, 0x72, 0x31, 0x06, 0x00, 0x51,
    0x3F, 0x02, 0x10, 0x6B, 0x31, 0x32, 0x90, 0x1A, 0x31, 0x06, 0x00, 0x11,
    0x22, 0x02, 0x27, 0x1E, 0xF0, 0x47, 0xC1, 0xA8, 0xBC, 0x6D, 0xBF, 0x17,
    0x02, 0x6D, 0x46, 0xBD, 0xBA, 0x6D, 0xBF, 0x17, // ldr s2, [x8, #0x66c] ; b 0x2c22090
};

// Counters that cut an amount of the step to whole units, with the amount of a 60 fps frame known
// at the site. HR_WHOLE gives the whole units of that amount for one counter and keeps the rest:
// in x13 the counter's address, w14 the bits of a 60 fps frame's amount, s0 the amount; out w12.
// It gives the game's own cut unless the frame step calls the high refresh routine and
// 0 < amount <= the 60 fps amount, so 60 fps and below, and Dark's mode, get the game's value
// exactly (at the 60 fps amount the rest, under 1, cannot change the cut). Unlike STEP_TICKS it
// starts a new counter from 0 and has slots of its own: 8 of {u64 address, f32 rest, pad} at state
// +0x100 and the next slot to take at +0x180, so the ambient and step rate counters that fill
// STEP_TICKS's table cannot push these out. Two things let it save nothing: every caller is
// followed by a call before x12-x17, s16, s17 or the flags are read, and it keeps s0.
//   w12 = fcvtzs(s0) ; call word is not ours, or bits - 1 >= w14 unsigned (0, negative, NaN and
//   larger amounts): ret ; find x13's slot, none: take [next], next = (next + 1) & 7, key = x13,
//   rest = 0 ; rest += s0 ; w12 = fcvtzs(rest) ; rest -= w12 ; ret
// It sits after the high refresh routine, the one free run this long that no known cheat writes.
constexpr size_t HR_WHOLE_OFFSET = 0x3C46E8C;
constexpr std::array<u8, 132> HR_WHOLE{
    0x0C, 0x00, 0x38, 0x1E, 0x0F, 0x55, 0xFE, 0x90, 0xEF, 0xC5, 0x4C, 0xB9,
    0xF0, 0x09, 0x90, 0x52, 0xB0, 0x9A, 0xB2, 0x72, 0xFF, 0x01, 0x10, 0x6B,
    0x41, 0x03, 0x00, 0x54, 0x0F, 0x00, 0x26, 0x1E, 0xEF, 0x05, 0x00, 0x51,
    0xFF, 0x01, 0x0E, 0x6B, 0xC2, 0x02, 0x00, 0x54, 0xD0, 0x92, 0x00, 0x90,
    0x10, 0x02, 0x3C, 0x91, 0x0E, 0x06, 0x41, 0xF8, 0xDF, 0x01, 0x0D, 0xEB,
    0x60, 0x01, 0x00, 0x54, 0x1F, 0x1A, 0x40, 0xF2, 0x81, 0xFF, 0xFF, 0x54,
    0x0E, 0x02, 0x40, 0xB9, 0xCF, 0x05, 0x00, 0x11, 0xEF, 0x09, 0x00, 0x12,
    0x0F, 0x02, 0x00, 0xB9, 0x10, 0x02, 0x02, 0xD1, 0x10, 0x12, 0x0E, 0x8B,
    0x0D, 0x06, 0x01, 0xF8, 0x1F, 0x82, 0x1F, 0xB8, 0x10, 0x82, 0x5F, 0xBC,
    0x10, 0x2A, 0x20, 0x1E, 0x0C, 0x02, 0x38, 0x1E, 0x91, 0x01, 0x22, 0x1E,
    0x10, 0x3A, 0x31, 0x1E, 0x10, 0x82, 0x1F, 0xBC, 0xC0, 0x03, 0x5F, 0xD6,
};
constexpr size_t HR_WHOLE_SLOTS_OFFSET = HR_FIX_STATE_OFFSET + 0x100;
// The end-of-slots test is on the address, so the slots must start on a 0x80 boundary.
static_assert(HR_WHOLE_SLOTS_OFFSET % 0x80 == 0);
static_assert(AdrpTarget(HR_WHOLE_OFFSET + 0x2C, InstructionAt(HR_WHOLE, 0x2C)) +
                  ((InstructionAt(HR_WHOLE, 0x30) >> 10) & 0xFFF) ==
              HR_WHOLE_SLOTS_OFFSET);
static_assert(HIGH_REFRESH_ROUTINE_OFFSET + OOT_CAVE_SIZE <= HR_WHOLE_OFFSET &&
              HR_WHOLE_OFFSET + HR_WHOLE.size() <= TEXTURE_POOL_ROUTINE_OFFSET);

// Time of day: each frame the clock's update (main+0x3CDF70) stores trunc(2 * step) game seconds
// to +0x3C, which are then added to the seconds of the day. Above 60 fps the cut loses time: the
// day runs slow below 120 fps (83% at 100) and stands still above it, with the sun swinging over
// the frozen hour. The cut stored to +0x3C (fcvtzs w8, s0 at main+0x3CDFC0) goes through HR_WHOLE
// with the counter x19 + 0x3C. The function's other cut, in the branch that sets the time from
// another source (main+0x3CE160), stays the game's.
//   x13 = x19 + 0x3c ; w14 = 2.0 ; bl HR_WHOLE ; w8 = w12 ; b 0x3cdfc4
constexpr std::array<u8, 20> HR_TIME_STUB{
    0x6D, 0xF2, 0x00, 0x91, 0x0E, 0x00, 0xA8, 0x52, 0x35, 0x01, 0x00, 0x94,
    0xE8, 0x03, 0x0C, 0x2A, 0x81, 0x1D, 0x1E, 0x17,
};

// The Deku Tree's floor web: its update (main+0x632CB0, s8 = the elapsed time) runs
// trunc(time / (1/60)) simulation steps, clamped to 0..3. Above 60 fps the cut is 0 every frame
// and the web does not react when Link lands on it. The cut (fcvtzs w22, s0 at main+0x632D00) goes
// through HR_WHOLE with a rest per web object (the counter x19 + 0x8F8), before the clamp, which
// stays: a long pause still gives at most 3 steps and leaves no backlog.
//   x13 = x19 + 0x8f8 ; w14 = 1.0 ; bl HR_WHOLE ; w22 = w12 ; b 0x632d04
constexpr std::array<u8, 20> HR_WEB_STUB{
    0x6D, 0xE2, 0x23, 0x91, 0x0E, 0xF0, 0xA7, 0x52, 0x30, 0x01, 0x00, 0x94,
    0xF6, 0x03, 0x0C, 0x2A, 0xCC, 0xB0, 0x27, 0x17,
};

// The physics solver's steps: each update the game sets them to trunc(step * the steps of a 60 fps
// frame + 0.001), stored to the solver's +0x2C8 (main+0x38AEB80 and main+0x38AFC10, the same code
// in both). Below a step of 1 the count is cut short and changes from frame to frame as the
// measured step wanders (8 x 0.49 gives 3, 8 x 0.5 gives 4), and it reaches 0 at high enough
// rates, so moving platforms slide out from under the player. High refresh mode, step under 1:
// the count is ceil(scaled - 0.1), at least 1, so a solver step never covers more time than at
// 60 fps (give or take the 0.1 for jitter, which keeps a steady rate at a steady count). That is
// never below the game's count. A step of 1 or more keeps the game's count, as does Dark's mode.
// Both sites call it (bl over fcvtzs w1, s0): a blr follows at once, so x16, x17, s16, s17 and
// x30 are free there. In: s0 = the game's float, s1 = the steps of a 60 fps frame. Out: w1.
//   w1 = fcvtzs(s0) ; call word is not ours: ret ; s16 = s0 - 0.001 ; s16 <= 0 or s16 >= s1: ret ;
//   w1 = max(fcvtzs(ceil(s16 - 0.1)), 1) ; ret
constexpr std::array<u8, 8> HR_SOLVER_CONSTANTS{
    0x6F, 0x12, 0x83, 0x3A, 0xCD, 0xCC, 0xCC, 0x3D, // 0.001f (the game's own), 0.1f
};
constexpr std::array<u8, 80> HR_SOLVER{
    0x01, 0x00, 0x38, 0x1E, 0x10, 0x55, 0xFE, 0x90, 0x10, 0xC6, 0x4C, 0xB9,
    0xF1, 0x09, 0x90, 0x52, 0xB1, 0x9A, 0xB2, 0x72, 0x1F, 0x02, 0x11, 0x6B,
    0xA1, 0x01, 0x00, 0x54, 0x90, 0xF8, 0xFF, 0x1C, 0x10, 0x38, 0x30, 0x1E,
    0x08, 0x22, 0x20, 0x1E, 0x2D, 0x01, 0x00, 0x54, 0x00, 0x22, 0x21, 0x1E,
    0xEA, 0x00, 0x00, 0x54, 0xF1, 0xF7, 0xFF, 0x1C, 0x10, 0x3A, 0x31, 0x1E,
    0x10, 0xC2, 0x24, 0x1E, 0x10, 0x02, 0x38, 0x1E, 0x1F, 0x06, 0x00, 0x71,
    0x01, 0xC6, 0x9F, 0x1A, 0xC0, 0x03, 0x5F, 0xD6,
};

// Epona's fence jump: the horse's jump state machine (main+0x2C39EB0, s10 = its step) sets the
// blend frames into its JumpReady animation so the blend ends at takeoff, except below a step of 1,
// which at 60 fps means slow motion: then it adds the step to last frame's blend frames instead
// (fcmp s10, #1.0 ; b.pl at main+0x2C3AE1C). In the high refresh mode every step is under 1, so
// the blend frames grew each frame, the jump animation never blended in, and Epona galloped
// through the fence at ground level while her body jumped. Now the b.pl goes to a stub that also
// skips the slow-motion branch when the high refresh mode is on and the step is the frame's own
// (within about 0.1% of [T+0x10], compared as bits): real slow motion, a step under the frame's,
// and every other case keep the game's test. x9, x10 and the flags are dead at both targets.
//   b.pl skip ; call word is not ours: slow ; w9 = |bits(s10) - bits([T + 0x10])| ;
//   w9 <= 8192: skip ; slow: b 0x2c3ae24 ; skip: b 0x2c3ae48
constexpr std::array<u8, 68> HR_HORSE_STUB{
    0x05, 0x02, 0x00, 0x54, 0x09, 0x55, 0xFE, 0x90, 0x29, 0xC5, 0x4C, 0xB9,
    0xEA, 0x09, 0x90, 0x52, 0xAA, 0x9A, 0xB2, 0x72, 0x3F, 0x01, 0x0A, 0x6B,
    0x21, 0x01, 0x00, 0x54, 0x69, 0x92, 0x00, 0xD0, 0x29, 0x91, 0x45, 0xF9,
    0x2A, 0x11, 0x40, 0xB9, 0x49, 0x01, 0x26, 0x1E, 0x29, 0x01, 0x0A, 0x6B,
    0x29, 0x55, 0x89, 0x5A, 0x3F, 0x09, 0x40, 0x71, 0x49, 0x00, 0x00, 0x54,
    0x8E, 0xD0, 0xBF, 0x17, 0x96, 0xD0, 0xBF, 0x17,
};

// Song notes: after a song is played, its playback (main+0x3292AF0) goes on to the next note on
// each update it finds the note flag set (f_832b7b50's result, w0), a word the sound engine sets
// when the melody reaches a note and clears on its next run. SOUND_GATE runs the sound job at
// 60 Hz above 60 fps, so the playback found the flag set on two or three updates in a row and
// showed that many notes at once, ahead of the music. The same holds for the ocarina's other
// melody playback (main+0x3292DC0). HR_SOUND_COUNT counts the sound job's runs (SOUND_FEED's way
// back now goes through it, in both modes); each playback keeps the count at its last set read,
// and a flag read set again before another run reads clear. Only in the high refresh mode; at
// 60 fps and below the job runs every frame, so every once-a-frame read sees a new count and the
// game's value. The two tbz w0, #0 after the calls (main+0x3292C88, 0x3292ED4) go to small stubs
// that save x16/x17 and call HR_SONG_CUE with the playback's slot (0 or 4) in w17.
//   stub: save ; w17 = slot ; bl HR_SONG_CUE ; restore ; tbz w0, #0, clear ; b set ; clear: b ...
//   HR_SONG_CUE: w0 clear or call word not ours: ret ; runs = [state + 8] ; old = [state + w17] ;
//                [state + w17] = runs ; old == runs: w0 = 0 ; ret (x14, x15 saved)
//   HR_SOUND_COUNT: save x16, x17 ; [state + 8] += 1 ; restore ; b 0x415338
// HR_SONG_B_STUB and HR_SOUND_COUNT sit in two of the 28-byte gaps between the game's functions
// (main+0x177F364, 0x177F5A4), past the ones mods are known to claim (to main+0x177D1C0).
constexpr std::array<u8, 28> HR_SONG_A_STUB{
    0xF0, 0x47, 0xBF, 0xA9, 0x11, 0x00, 0x80, 0x52, 0x73, 0x01, 0x00, 0x94,
    0xF0, 0x47, 0xC1, 0xA8, 0x40, 0x00, 0x00, 0x36, 0xA6, 0x30, 0xD9, 0x17,
    0xA7, 0x30, 0xD9, 0x17,
};
constexpr std::array<u8, 28> HR_SONG_B_STUB{
    0xF0, 0x47, 0xBF, 0xA9, 0x91, 0x00, 0x80, 0x52, 0x12, 0x1F, 0x93, 0x94,
    0xF0, 0x47, 0xC1, 0xA8, 0x40, 0x00, 0x00, 0x36, 0xD8, 0x4E, 0x6C, 0x14,
    0x32, 0x4F, 0x6C, 0x14,
};
constexpr std::array<u8, 72> HR_SONG_CUE{
    0xEE, 0x3F, 0xBF, 0xA9, 0xE0, 0x01, 0x00, 0x36, 0x0E, 0x55, 0xFE, 0x90,
    0xCE, 0xC5, 0x4C, 0xB9, 0xEF, 0x09, 0x90, 0x52, 0xAF, 0x9A, 0xB2, 0x72,
    0xDF, 0x01, 0x0F, 0x6B, 0x21, 0x01, 0x00, 0x54, 0xD0, 0x92, 0x00, 0x90,
    0x10, 0x42, 0x3E, 0x91, 0x0E, 0x0A, 0x40, 0xB9, 0x0F, 0x6A, 0x71, 0xB8,
    0x0E, 0x6A, 0x31, 0xB8, 0xFF, 0x01, 0x0E, 0x6B, 0x41, 0x00, 0x00, 0x54,
    0xE0, 0x03, 0x1F, 0x2A, 0xEE, 0x3F, 0xC1, 0xA8, 0xC0, 0x03, 0x5F, 0xD6,
};
constexpr std::array<u8, 28> HR_SOUND_COUNT{
    0xF0, 0x47, 0xBF, 0xA9, 0xF0, 0xB8, 0x01, 0xF0, 0x11, 0x9A, 0x4F, 0xB9,
    0x31, 0x06, 0x00, 0x11, 0x11, 0x9A, 0x0F, 0xB9, 0xF0, 0x47, 0xC1, 0xA8,
    0x5F, 0x57, 0xB2, 0x17,
};
constexpr size_t HR_SONG_STATE_OFFSET = HR_FIX_STATE_OFFSET + 0x190; // u32 A, u32 B, u32 runs
static_assert(AdrpTarget(0x3C46FB4 + 0x20, InstructionAt(HR_SONG_CUE, 0x20)) +
                  AddImmediate(InstructionAt(HR_SONG_CUE, 0x24)) ==
              HR_SONG_STATE_OFFSET);
static_assert(AdrpTarget(0x177F5A4 + 0x4, InstructionAt(HR_SOUND_COUNT, 0x4)) +
                  ((InstructionAt(HR_SOUND_COUNT, 0x8) >> 10) & 0xFFF) * 4 ==
              HR_SONG_STATE_OFFSET + 8);
static_assert(InstructionAt(SOUND_FEED, 0x34) == EncodeBl(0x3C46464, 0x177F5A4) - 0x80000000U &&
              InstructionAt(HR_SOUND_COUNT, 0x18) == EncodeBl(0x177F5BC, 0x415338) - 0x80000000U);

// The hooks (each checked to hold the game's word), then the stubs in address order.
constexpr std::array HR_FIX_HOOKS{
    OotFixHook{0x415300, {0x28, 0xC4, 0xE0, 0x14}, 0xD503237F}, // pacibsp -> b 0x3c463a0
    OotFixHook{0x415334, {0x3F, 0xC4, 0xE0, 0x14}, 0xBD401100}, // ldr s0, [x8, #0x10] -> b 0x3c46430
    OotFixHook{0x3293410, {0x3C, 0xCC, 0x26, 0x14}, 0x1E390008}, // fcvtzu w8, s0 -> b 0x3c46500
    OotFixHook{0x2F53268, {0xBA, 0xCC, 0x33, 0x14}, 0x5EA1B900}, // fcvtzs s0, s8 -> b 0x3c46550
    // The countdown's words its stub skips in the high refresh mode, kept as they are.
    OotFixHook{0x2F5326C, {0x09, 0x01, 0x38, 0x1E}, 0x1E380109},
    OotFixHook{0x2F53270, {0x00, 0xD8, 0x21, 0x5E}, 0x5E21D800},
    OotFixHook{0x2F53274, {0x00, 0x39, 0x20, 0x1E}, 0x1E203900},
    OotFixHook{0x2F53278, {0x00, 0x20, 0x30, 0x1E}, 0x1E302000},
    OotFixHook{0x2F5327C, {0xAA, 0x17, 0x00, 0x54}, 0x540017AA},
    OotFixHook{0x2F53280, {0x00, 0x20, 0x31, 0x1E}, 0x1E312000},
    OotFixHook{0x2F53284, {0x09, 0x18, 0x00, 0x54}, 0x54001809},
    OotFixHook{0x2F53288, {0x0A, 0x9B, 0x40, 0xB9}, 0xB9409B0A},
    OotFixHook{0x2F5328C, {0x3F, 0x05, 0x00, 0x71}, 0x7100053F},
    OotFixHook{0x2F53290, {0x29, 0xC5, 0x9F, 0x1A}, 0x1A9FC529},
    OotFixHook{0x53F914, {0x27, 0x1B, 0xDC, 0x14}, 0x1100077A}, // add w26, w27, #1 -> b 0x3c465b0
    // The ground velocity's fcmp s0, s1 (kept) and b.gt -> its stub, three times; the turn
    // speed's two fmul (the second kept, unreached in the high refresh mode).
    OotFixHook{0x13A06C, {0x00, 0x20, 0x21, 0x1E}, 0x1E212000},
    OotFixHook{0x13A070, {0x68, 0x31, 0xEC, 0x14}, 0x5400196C}, // b.gt 0x13a39c -> b 0x3c46610
    OotFixHook{0x13D70C, {0x00, 0x20, 0x21, 0x1E}, 0x1E212000},
    OotFixHook{0x13D710, {0xD0, 0x23, 0xEC, 0x14}, 0x54001E2C}, // b.gt 0x13dad4 -> b 0x3c46650
    OotFixHook{0x389AC28, {0x00, 0x20, 0x21, 0x1E}, 0x1E212000},
    OotFixHook{0x389AC2C, {0x99, 0xAE, 0x0E, 0x14}, 0x540017CC}, // b.gt 0x389af24 -> b 0x3c46690
    OotFixHook{0x261EB18, {0xEE, 0x9E, 0x58, 0x14}, 0x1E210801}, // fmul s1, s0, s1 -> b 0x3c466d0
    OotFixHook{0x261EB1C, {0x00, 0x08, 0x28, 0x1E}, 0x1E280800},
    OotFixHook{0x3B5C7C8, {0xD6, 0xA7, 0x03, 0x14}, 0x11000508}, // add w8, w8, #1 -> b 0x3c46720
    OotFixHook{0x3296B08, {0x22, 0xBF, 0x26, 0x14}, 0x1E390008}, // fcvtzu w8, s0 -> b 0x3c46790
    OotFixHook{0x32926C8, {0x46, 0xD0, 0x26, 0x14}, 0x1E390008}, // fcvtzu w8, s0 -> b 0x3c467e0
    OotFixHook{0x14632C, {0x41, 0x01, 0xEC, 0x14}, 0xBD466D00}, // ldr s0, [x8, #0x66c] -> b 0x3c46830
    OotFixHook{0x2AB804, {0x23, 0x6C, 0xE6, 0x14}, 0x1E270100}, // fmov s0, w8 -> b 0x3c46890
    OotFixHook{0x2C207A4, {0x53, 0x98, 0x40, 0x14}, 0xBD466D00}, // ldr s0, [x8, #0x66c] -> b 0x3c468f0
    OotFixHook{0x2C2208C, {0x31, 0x92, 0x40, 0x14}, 0xBD466D02}, // ldr s2, [x8, #0x66c] -> b 0x3c46950
    OotFixHook{0x3CDFC0, {0x7C, 0xE2, 0xE1, 0x14}, 0x1E380008, // fcvtzs w8, s0 -> b 0x3c469b0
               "FRITCHII_NO_HR_TIME"},
    OotFixHook{0x632D00, {0x31, 0x4F, 0xD8, 0x14}, 0x1E380016, // fcvtzs w22, s0 -> b 0x3c469c4
               "FRITCHII_NO_HR_WEB"},
    OotFixHook{0x38AEBF4, {0xAE, 0x5F, 0x0E, 0x94}, 0x1E380001, // fcvtzs w1, s0 -> bl 0x3c46aac
               "FRITCHII_NO_HR_SOLVER"},
    OotFixHook{0x38AFD20, {0x63, 0x5B, 0x0E, 0x94}, 0x1E380001, // fcvtzs w1, s0 -> bl 0x3c46aac
               "FRITCHII_NO_HR_SOLVER"},
    OotFixHook{0x2C3AE20, {0x64, 0x2F, 0x40, 0x14}, 0x54000145, // b.pl 0x2c3ae48 -> b 0x3c46bb0
               "FRITCHII_NO_HR_HORSE"},
    OotFixHook{0x3292C88, {0x56, 0xCF, 0x26, 0x14}, 0x36000060, // tbz w0, #0 -> b 0x3c469e0
               "FRITCHII_NO_HR_SONG"},
    OotFixHook{0x3292ED4, {0x24, 0xB1, 0x93, 0x17}, 0x36000B80, // tbz w0, #0 -> b 0x177f364
               "FRITCHII_NO_HR_SONG"},
};
constexpr std::array HR_FIX_STUBS{
    OotFixStub{0x3C463A0, SOUND_GATE},
    OotFixStub{0x3C46430, SOUND_FEED},
    OotFixStub{0x3C46470, STEP_TICKS},
    OotFixStub{0x3C46500, STEP_REPLAY},
    OotFixStub{0x3C46550, STEP_HITSTOP},
    OotFixStub{0x3C465B0, STEP_AMBIENT},
    OotFixStub{0x3C46610, GROUND_A},
    OotFixStub{0x3C46650, GROUND_B},
    OotFixStub{0x3C46690, GROUND_C},
    OotFixStub{0x3C466D0, TURN_SPEED},
    OotFixStub{0x3C46720, MOVIE_FRAMES},
    OotFixStub{0x3C46790, SCARECROW_RECORD},
    OotFixStub{0x3C467E0, SCARECROW_LAST},
    OotFixStub{0x3C46830, CLOTH_DT_A},
    OotFixStub{0x3C46890, CLOTH_DT_B},
    OotFixStub{0x3C468F0, CLOTH_DT_C},
    OotFixStub{0x3C46950, CLOTH_DT_D},
    OotFixStub{0x3C469B0, HR_TIME_STUB},
    OotFixStub{0x3C469C4, HR_WEB_STUB},
    OotFixStub{0x3C469D8, HR_SOLVER_CONSTANTS},
    OotFixStub{0x3C469E0, HR_SONG_A_STUB},
    OotFixStub{0x3C46AAC, HR_SOLVER},
};
// Stubs in the gaps between the game's functions (the free 28-byte ones past main+0x177D1C0).
constexpr std::array HR_FIX_SLOT_STUBS{
    OotFixStub{0x177F364, HR_SONG_B_STUB},
    OotFixStub{0x177F5A4, HR_SOUND_COUNT},
};
static_assert([] {
    size_t end = 0x177D1C0;
    for (const OotFixStub& stub : HR_FIX_SLOT_STUBS) {
        if (stub.offset < end || stub.offset % 4 != 0 || stub.bytes.size() > 28 ||
            stub.offset + stub.bytes.size() > 0x177F6A8) {
            return false;
        }
        end = stub.offset + stub.bytes.size();
    }
    return true;
}());
// Stubs past the pickup stub, also in address order.
constexpr std::array HR_FIX_LATE_STUBS{
    OotFixStub{0x3C46BB0, HR_HORSE_STUB},
    OotFixStub{HR_WHOLE_OFFSET, HR_WHOLE},
    OotFixStub{0x3C46FB4, HR_SONG_CUE},
};
// The sound accumulator and flag at +0, the step slots' index at +8, the 15 slots from +0x10, and
// from +0x100 HR_WHOLE's 8 slots and its next-slot word, from +0x190 the song notes' two slots and
// the sound job's run count.
constexpr size_t HR_FIX_STATE_SIZE = 0x1A0;
// The stubs in order, in free padding between the Hyrule Cheat Menu and the pickup stub; their
// state inside the BSS, clear of Dark's timer state, the ripple and pickup slots and the cheats'
// flags from main+0x4E9FF80. Known cheats write their own code to main+0x3C46A00..0x3C46AAC and
// main+0x3C46CC0..0x3C46DB0 while the game runs, which no check at load can see: no stub there.
constexpr bool HrFixStubsInOrder(std::span<const OotFixStub> stubs, size_t start, size_t limit) {
    size_t end = start;
    for (const OotFixStub& stub : stubs) {
        if (stub.offset < end || stub.offset % 4 != 0 || stub.bytes.size() % 4 != 0) {
            return false;
        }
        end = stub.offset + stub.bytes.size();
        const auto touches = [&](size_t first, size_t last) {
            return stub.offset < last && first < end;
        };
        if (touches(0x3C46A00, 0x3C46AAC) || touches(0x3C46CC0, 0x3C46DB0)) {
            return false;
        }
    }
    return end <= limit;
}
static_assert(HR_FIX_CAVE_START >= 0x3C46398 &&
              HrFixStubsInOrder(HR_FIX_STUBS, HR_FIX_CAVE_START, PICKUP_STUB_OFFSET));
// The late stubs: after the pickup stub, to the end of the page, around the words other built-ins
// write there and the community ultrawide mod's spot (main+0x3C46C00..0x3C46DC0).
static_assert(HrFixStubsInOrder(HR_FIX_LATE_STUBS, PICKUP_STUB_OFFSET + PICKUP_STUB.size(),
                                0x3C47000));
static_assert([] {
    constexpr std::array<std::pair<size_t, size_t>, 5> taken{{
        {0x3C46C00, 0x3C46E00}, // the ultrawide mod's spot, the shadow steps, the texture heap gap
        {HIGH_REFRESH_ROUTINE_OFFSET, HIGH_REFRESH_ROUTINE_OFFSET + OOT_CAVE_SIZE},
        {TEXTURE_POOL_ROUTINE_OFFSET, TEXTURE_POOL_ROUTINE_OFFSET + TEXTURE_POOL_ROUTINE.size()},
        {WATER_ROUTINE_OFFSET, WATER_ROUTINE_OFFSET + WATER_ROUTINE.size()},
        {PICKUP_STUB_OFFSET, PICKUP_STUB_OFFSET + PICKUP_STUB.size()},
    }};
    for (const OotFixStub& stub : HR_FIX_LATE_STUBS) {
        for (const auto& [first, last] : taken) {
            if (stub.offset < last && first < stub.offset + stub.bytes.size()) {
                return false;
            }
        }
    }
    return true;
}());
static_assert(HR_FIX_STATE_OFFSET >= 0x4E9B000 &&
              HR_FIX_STATE_OFFSET + HR_FIX_STATE_SIZE <= HIGH_REFRESH_STATE_OFFSET &&
              HR_FIX_STATE_OFFSET + HR_FIX_STATE_SIZE <= PICKUP_STATE_OFFSET &&
              HR_WHOLE_SLOTS_OFFSET + 0x84 <= HR_SONG_STATE_OFFSET &&
              HR_SONG_STATE_OFFSET + 0xC <= HR_FIX_STATE_OFFSET + HR_FIX_STATE_SIZE);
// Stub words with a second form a test switch puts back: the stub's word applies unless the switch
// is 1, the old word only while it is 1.
struct OotFixVariant {
    size_t offset;
    std::array<u8, 4> old_bytes;
    const char* env;
};
constexpr std::array<u8, 4> CLOTH_OLD_GATE{0xC1, 0x01, 0x00, 0x54};  // b.ne +0x38 (the game's load)
constexpr std::array<u8, 4> CLOTH_OLD_BOUND{0x10, 0x91, 0xA7, 0x72}; // movk w16, #0x3c88, lsl #16
constexpr std::array HR_FIX_VARIANTS{
    OotFixVariant{0x3C46830 + 0x1C, CLOTH_OLD_GATE, "FRITCHII_NO_CLOTH_SLOW"},
    OotFixVariant{0x3C46830 + 0x34, CLOTH_OLD_BOUND, "FRITCHII_NO_CLOTH_SLOW"},
    OotFixVariant{0x3C46890 + 0x1C, CLOTH_OLD_GATE, "FRITCHII_NO_CLOTH_SLOW"},
    OotFixVariant{0x3C46890 + 0x34, CLOTH_OLD_BOUND, "FRITCHII_NO_CLOTH_SLOW"},
    OotFixVariant{0x3C468F0 + 0x1C, CLOTH_OLD_GATE, "FRITCHII_NO_CLOTH_SLOW"},
    OotFixVariant{0x3C468F0 + 0x34, CLOTH_OLD_BOUND, "FRITCHII_NO_CLOTH_SLOW"},
    OotFixVariant{0x3C46950 + 0x1C, CLOTH_OLD_GATE, "FRITCHII_NO_CLOTH_SLOW"},
    OotFixVariant{0x3C46950 + 0x34, CLOTH_OLD_BOUND, "FRITCHII_NO_CLOTH_SLOW"},
};
// The cloth stubs' two changed words: a nop over the gate's b.ne, the 4/60 bound's movk.
static_assert(InstructionAt(CLOTH_DT_A, 0x1C) == 0xD503201F && InstructionAt(CLOTH_DT_B, 0x1C) == 0xD503201F &&
              InstructionAt(CLOTH_DT_C, 0x1C) == 0xD503201F && InstructionAt(CLOTH_DT_D, 0x1C) == 0xD503201F);
static_assert(InstructionAt(CLOTH_DT_A, 0x34) == 0x72A7B110 && InstructionAt(CLOTH_DT_B, 0x34) == 0x72A7B110 &&
              InstructionAt(CLOTH_DT_C, 0x34) == 0x72A7B110 && InstructionAt(CLOTH_DT_D, 0x34) == 0x72A7B110);
// The hooks over the game's words, then every stub word, each checked to be zero.
constexpr size_t HR_FIX_RECORD_COUNT = HR_FIX_HOOKS.size() + HR_FIX_VARIANTS.size() + [] {
    size_t words = 0;
    for (const OotFixStub& stub : HR_FIX_STUBS) {
        words += stub.bytes.size() / 4;
    }
    for (const OotFixStub& stub : HR_FIX_LATE_STUBS) {
        words += stub.bytes.size() / 4;
    }
    for (const OotFixStub& stub : HR_FIX_SLOT_STUBS) {
        words += stub.bytes.size() / 4;
    }
    return words;
}();
constexpr auto HR_FIX_RECORDS = [] {
    std::array<BuiltinPatchRecord, HR_FIX_RECORD_COUNT> records{};
    size_t i = 0;
    for (const OotFixHook& hook : HR_FIX_HOOKS) {
        records[i++] = BuiltinPatchRecord{.module_offset = hook.offset,
                                          .bytes = hook.bytes,
                                          .checked = true,
                                          .original = hook.original,
                                          .skip_env = hook.skip_env};
    }
    const auto add_stubs = [&](std::span<const OotFixStub> stubs) {
        for (const OotFixStub& stub : stubs) {
            for (size_t offset = 0; offset < stub.bytes.size(); offset += 4) {
                const char* skip_env = nullptr;
                for (const OotFixVariant& variant : HR_FIX_VARIANTS) {
                    if (variant.offset == stub.offset + offset) {
                        skip_env = variant.env;
                    }
                }
                records[i++] = BuiltinPatchRecord{.module_offset = stub.offset + offset,
                                                  .bytes = stub.bytes.subspan(offset, 4),
                                                  .checked = true,
                                                  .original = 0,
                                                  .skip_env = skip_env};
            }
        }
    };
    add_stubs(HR_FIX_STUBS);
    add_stubs(HR_FIX_LATE_STUBS);
    add_stubs(HR_FIX_SLOT_STUBS);
    for (const OotFixVariant& variant : HR_FIX_VARIANTS) {
        records[i++] = BuiltinPatchRecord{.module_offset = variant.offset,
                                          .bytes = variant.old_bytes,
                                          .checked = true,
                                          .original = 0,
                                          .need_env = variant.env};
    }
    return records;
}();
constexpr std::string_view HR_FIX_NAME = "High refresh fixes (built-in)";
// Every variant replaces a word some stub writes.
static_assert([] {
    for (const OotFixVariant& variant : HR_FIX_VARIANTS) {
        size_t writers = 0;
        for (const BuiltinPatchRecord& record : HR_FIX_RECORDS) {
            writers += record.module_offset == variant.offset && record.skip_env == variant.env;
        }
        if (writers != 1) {
            return false;
        }
    }
    return true;
}());

bool BuiltinRecordSkipped(const BuiltinPatchRecord& record) {
    const auto set = [](const char* name) {
        const char* const value = std::getenv(name);
        return value != nullptr && std::string_view{value} == "1";
    };
    return (record.skip_env != nullptr && set(record.skip_env)) ||
           (record.need_env != nullptr && !set(record.need_env));
}

bool HighRefreshSettingOn() {
    // Tests boot either way without touching the player's config: EDEN_NX2_OOT_HIGH_REFRESH=0/1.
    if (const char* const value = std::getenv("EDEN_NX2_OOT_HIGH_REFRESH"); value && *value) {
        return *value != '0';
    }
    return Settings::values.oot_high_refresh.GetValue();
}

// Occlusion Off (a ported community patch): the game creates no
// z-prepass occlusion queries and no samples-passed queries (ldrb w8, [x0] -> mov w8, wzr), so it
// issues no occlusion reports and no conditional renders; every candidate object is drawn.
constexpr std::array<u8, 4> MOV_W8_WZR{0xE8, 0x03, 0x1F, 0x2A};
constexpr u32 LDRB_W8_X0 = 0x39400008;
constexpr std::array OCCLUSION_OFF_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x11ACE68, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11B828C, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11B82A4, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11ACEC8, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11B82DC, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
    BuiltinPatchRecord{.module_offset = 0x11B82F4, .bytes = MOV_W8_WZR, .checked = true, .original = LDRB_W8_X0},
};

// Resource poll backoff: the resource request thread (main+0xF7A0)
// rescans its pending requests at once when one is not ready (`tbz w8, #0, loop` at main+0x14ABC,
// w8 = "all done"). That branch now goes through the zero padding at main+0x17088: `svc #0xf3`
// (a short sleep while EXP25 is on, core/hle/kernel/svc/svc_nx2_oot_poll.cpp) and `b loop`.
constexpr std::array<u8, 4> POLL_RESCAN_BRANCH{0x68, 0x2E, 0x01, 0x36};  // tbz w8, #0, 0x17088
constexpr std::array<u8, 4> POLL_BACKOFF_SVC{0x61, 0x1E, 0x00, 0xD4};    // svc #0xf3
constexpr std::array<u8, 4> POLL_BACKOFF_RETURN{0x0D, 0xE2, 0xFF, 0x17}; // b 0xf8c0
constexpr std::array POLL_BACKOFF_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x14ABC,
                       .bytes = POLL_RESCAN_BRANCH,
                       .checked = true,
                       .original = 0x36057028}, // tbz w8, #0, 0xf8c0
    BuiltinPatchRecord{.module_offset = 0x17088, .bytes = POLL_BACKOFF_SVC, .checked = true, .original = 0},
    BuiltinPatchRecord{.module_offset = 0x1708C, .bytes = POLL_BACKOFF_RETURN, .checked = true, .original = 0},
};

// Shadow size steps: OoT gives each nearby object its own square depth shadow map
// ("local_shadow_depth"), sized by the object's size on screen. While the player moves, every
// side changes every frame, so the host makes ~10-13 new depth images a frame in the Castle Town
// pot room (2-5 ms a frame). The one read of the side before the texture request (main+0x2440A0,
// `ldr w10, [x21, #0x4e8]` at main+0x244100) now goes through four words of free space that round
// it up to a multiple of 64: 384..1,920 becomes 25 sizes the image pool can reuse. The projection
// is unchanged, so a map gets at most 63 more texels a side. Research: work\potroom_patch_research.md.
constexpr std::array<u8, 4> SHADOW_STEPS_BRANCH{0x30, 0x0B, 0xE8, 0x14}; // b 0x3c46dc0
constexpr std::array<u8, 4> SHADOW_STEPS_LOAD{0xAA, 0xEA, 0x44, 0xB9};   // ldr w10, [x21, #0x4e8]
constexpr std::array<u8, 4> SHADOW_STEPS_ADD{0x4A, 0xFD, 0x00, 0x11};    // add w10, w10, #0x3f
constexpr std::array<u8, 4> SHADOW_STEPS_AND{0x4A, 0x65, 0x1A, 0x12};    // and w10, w10, #0xffffffc0
constexpr std::array<u8, 4> SHADOW_STEPS_RETURN{0xCE, 0xF4, 0x17, 0x17}; // b 0x244104
constexpr std::array SHADOW_STEPS_RECORDS{
    BuiltinPatchRecord{.module_offset = 0x244100,
                       .bytes = SHADOW_STEPS_BRANCH,
                       .checked = true,
                       .original = 0xB944EAAA}, // ldr w10, [x21, #0x4e8]
    BuiltinPatchRecord{.module_offset = 0x3C46DC0, .bytes = SHADOW_STEPS_LOAD, .checked = true, .original = 0},
    BuiltinPatchRecord{.module_offset = 0x3C46DC4, .bytes = SHADOW_STEPS_ADD, .checked = true, .original = 0},
    BuiltinPatchRecord{.module_offset = 0x3C46DC8, .bytes = SHADOW_STEPS_AND, .checked = true, .original = 0},
    BuiltinPatchRecord{.module_offset = 0x3C46DCC, .bytes = SHADOW_STEPS_RETURN, .checked = true, .original = 0},
};

// Texture heap tidy-up at most once a second: the game keeps its textures' storage in one heap and
// tidies it in rounds on a thread of its own (aglTextureDataMgr::Compaction). The texture data
// manager's per-frame step (main+0x777240, under its lock) asks for a new round at every frame no
// round runs: its flags byte (+0x3C0) has bit 1 set once at setup. Every texture a round moves is
// made again at its new address, which for us is a new image, a new upload and new descriptors: in
// the market the heap sits at its 5% free line and the rounds never stop (the thread wanted the
// cores 231-261 ms a second in the 6.0 market profile). Now a round may start at most once a second
// (the guest's 19.2 MHz counter); the round itself, what it moves and how, is the game's, and a
// started round always finishes. Not tied to the frame rate: it applies at any fps and in both
// frame-step modes. The one risk: the heap is more fragmented than the game expects between rounds.
// tbz w8, #1, 0x777288 at main+0x777278 branches to:
//   tbz w8, #1, skip ; x9 = cntpct ; x8 = x9 - [last] ; x8 < 19.2 M: skip ;
// (CNTPCT_EL0, the counter the game itself reads: the JIT has no CNTVCT_EL0 and aborted on it.)
//   [last] = x9 ; b 0x77727c (ask) ; skip: b 0x777288
// x8-x11 and the flags are dead at both targets. State: the last ask's counter value.
constexpr size_t TEX_GAP_HOOK_OFFSET = 0x777278;
constexpr size_t TEX_GAP_STUB_OFFSET = 0x3C46DD0;
constexpr std::array<u8, 4> TEX_GAP_HOOK{0xD6, 0x3E, 0xD3, 0x14}; // b 0x3c46dd0
constexpr std::array<u8, 48> TEX_GAP_STUB{
    0x68, 0x01, 0x08, 0x36, 0x29, 0xE0, 0x3B, 0xD5, 0xCA, 0x92, 0x00, 0x90,
    0x48, 0xD1, 0x47, 0xF9, 0x28, 0x01, 0x08, 0xCB, 0x0B, 0x00, 0x9F, 0x52,
    0x8B, 0x24, 0xA0, 0x72, 0x1F, 0x01, 0x0B, 0xEB, 0x63, 0x00, 0x00, 0x54,
    0x49, 0xD1, 0x07, 0xF9, 0x21, 0xC1, 0x2C, 0x17, 0x23, 0xC1, 0x2C, 0x17,
};
constexpr size_t TEX_GAP_STATE_OFFSET = 0x4E9EFA0;
constexpr size_t TEX_GAP_STATE_SIZE = 0x8;
static_assert(InstructionAt(TEX_GAP_HOOK, 0) == EncodeBl(TEX_GAP_HOOK_OFFSET, TEX_GAP_STUB_OFFSET) - 0x80000000U);
static_assert(AdrpTarget(TEX_GAP_STUB_OFFSET + 0x8, InstructionAt(TEX_GAP_STUB, 0x8)) +
                  ((InstructionAt(TEX_GAP_STUB, 0xC) >> 10) & 0xFFF) * 8 ==
              TEX_GAP_STATE_OFFSET);
// After the shadow steps' four words, before the high refresh routine, clear of the code known
// cheats write from main+0x3C46CC0 to 0x3C46DB0.
static_assert(TEX_GAP_STUB_OFFSET >= 0x3C46DD0 &&
              TEX_GAP_STUB_OFFSET + TEX_GAP_STUB.size() <= HIGH_REFRESH_ROUTINE_OFFSET);
// Its state after the high refresh fixes' (BSS no mod is known to write below main+0x4E9F000).
static_assert(HR_FIX_STATE_OFFSET + HR_FIX_STATE_SIZE <= TEX_GAP_STATE_OFFSET &&
              TEX_GAP_STATE_OFFSET % 8 == 0 &&
              TEX_GAP_STATE_OFFSET + TEX_GAP_STATE_SIZE <= 0x4E9F000);
constexpr auto TEX_GAP_RECORDS = [] {
    std::array<BuiltinPatchRecord, 1 + TEX_GAP_STUB.size() / 4> records{};
    records[0] = BuiltinPatchRecord{.module_offset = TEX_GAP_HOOK_OFFSET,
                                    .bytes = TEX_GAP_HOOK,
                                    .checked = true,
                                    .original = 0x36080088}; // tbz w8, #1, 0x777288
    for (size_t i = 1, offset = 0; offset < TEX_GAP_STUB.size(); ++i, offset += 4) {
        records[i] = BuiltinPatchRecord{.module_offset = TEX_GAP_STUB_OFFSET + offset,
                                        .bytes = std::span<const u8>{TEX_GAP_STUB}.subspan(offset, 4),
                                        .checked = true,
                                        .original = 0};
    }
    return records;
}();

constexpr std::array BUILTIN_PATCHES{
    BuiltinPatch{
        .name = MAX_RESOLUTION_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = MAX_RESOLUTION_RECORDS,
        .disable_env = "EDEN_NX2_NO_MAX_RES",
        .setting_enabled = &MaxResolutionFits,
    },
    // With Max Resolution the guard house outgrows the initial pool and dereferences a null
    // allocation (the pot room freeze in Castle Town).
    BuiltinPatch{
        .name = "Texture Pool x2 (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = TEXTURE_POOL_RECORDS,
    },
    // Before Dark's: it contains his routine and replaces it while applied. Off by default; only
    // listed in Add-Ons while the setting is on.
    BuiltinPatch{
        .name = HIGH_REFRESH_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = HIGH_REFRESH_RECORDS,
        .setting_enabled = &HighRefreshSettingOn,
        .replaces = DYNAMIC_FPS_NAME,
        .state_offset = HIGH_REFRESH_STATE_OFFSET,
        .state_size = HIGH_REFRESH_STATE_SIZE,
    },
    BuiltinPatch{
        .name = DYNAMIC_FPS_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = DYNAMIC_FPS_RECORDS,
    },
    // After Dark's (v5.8, a ported finding): his mode ran trunc(step) ripple
    // passes, so at a 60 fps cap some frames ran none and below 60 the ripples ran slow.
    BuiltinPatch{
        .name = WATER_RIPPLE_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = WATER_RIPPLE_RECORDS,
        .state_offset = HIGH_REFRESH_STATE_OFFSET,
        .state_size = HIGH_REFRESH_STATE_SIZE,
    },
    // In both modes, so Ctrl+U can switch the high refresh mode on live; the stub only changes
    // anything while the frame step calls the high refresh routine (Dark's mode: the game's own
    // cut, bit for bit).
    BuiltinPatch{
        .name = PICKUP_COUNTER_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = PICKUP_COUNTER_RECORDS,
        .state_offset = PICKUP_STATE_OFFSET,
        .state_size = PICKUP_STATE_SIZE,
    },
    // In both modes too, for the same reason: each stub changes nothing unless the frame step
    // calls the high refresh routine.
    BuiltinPatch{
        .name = HR_FIX_NAME,
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = HR_FIX_RECORDS,
        .disable_env = "FRITCHII_NO_HR_FIXES", // =1 skips it (A/B runs, a fix that misbehaves)
        .state_offset = HR_FIX_STATE_OFFSET,
        .state_size = HR_FIX_STATE_SIZE,
    },
    BuiltinPatch{
        .name = "Occlusion Off (built-in, EDEN_NX2_OCCLUSION_OFF=1)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = OCCLUSION_OFF_RECORDS,
        .enable_env = "EDEN_NX2_OCCLUSION_OFF",
    },
    BuiltinPatch{
        .name = "Resource poll backoff (built-in, EDEN_NX2_OOT_POLL_BACKOFF=1)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = POLL_BACKOFF_RECORDS,
        .enable_env = "EDEN_NX2_OOT_POLL_BACKOFF",
    },
    BuiltinPatch{
        .name = "Shadow Size Steps (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = SHADOW_STEPS_RECORDS,
        // On by default since the pot room run (2026-11-12): new depth images while strafing at
        // the town wall 4,000-10,000 -> 30-90 per 600 frames, frames 20.5-26 -> 18.4-18.8 ms,
        // shadows unchanged in the screenshot. FRITCHII_NO_SHADOW_STEPS=1 turns it off.
        .disable_env = "FRITCHII_NO_SHADOW_STEPS",
    },
    // Applies to everyone at any frame rate. FRITCHII_NO_TEX_COMPACT_GAP=1 turns it off (the game
    // asks for a round at every frame, as before), =0 applies it even when Add-Ons has it off.
    BuiltinPatch{
        .name = "Texture heap tidy-up once a second (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = TEX_GAP_RECORDS,
        .disable_env = "FRITCHII_NO_TEX_COMPACT_GAP",
        .state_offset = TEX_GAP_STATE_OFFSET,
        .state_size = TEX_GAP_STATE_SIZE,
    },
    // OoT's render height (oot_render_height, Settings::OcarinaRenderHeight) under Max
    // Resolution with DLSS or FSR. The picture menu owns it: not in Add-Ons, where unticking one
    // would leave the menu naming a height the game doesn't draw.
    BuiltinPatch{
        .name = "360p render (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_360_RECORDS,
        .setting_enabled = &OcarinaRender360On,
        .render_height = 360,
    },
    BuiltinPatch{
        .name = "540p render (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_540_RECORDS,
        .setting_enabled = &OcarinaRender540On,
        .render_height = 540,
    },
    BuiltinPatch{
        .name = "720p render (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_720_RECORDS,
        .setting_enabled = &OcarinaRender720On,
        .render_height = 720,
    },
    BuiltinPatch{
        .name = "810p render (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_810_RECORDS,
        .setting_enabled = &OcarinaRender810On,
        .render_height = 810,
    },
    BuiltinPatch{
        .name = "900p render (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_900_RECORDS,
        .setting_enabled = &OcarinaRender900On,
        .render_height = 900,
    },
    BuiltinPatch{
        .name = "1080p render (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_1080_RECORDS,
        .setting_enabled = &OcarinaRender1080On,
        .render_height = 1080,
    },
    BuiltinPatch{
        .name = "1260p render (built-in)",
        .title_id = 0x0400486024DEC000,
        .module = "main",
        .build_id = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348",
        .records = RENDER_1260_RECORDS,
        .setting_enabled = &OcarinaRender1260On,
        .render_height = 1260,
    },
};

// The JIT translates only these system registers (dynarmic's A64 impl/system.cpp): an MRS or MSR
// of any other ends the game with "unreachable" the first time the patched code runs, which no
// test sees until a game reaches that code (the texture heap stub's CNTVCT_EL0, 2026-11-19). So
// every word of every built-in patch is checked here, at build time. Words are compared with
// their register field (Rt) cleared.
constexpr bool PatchWordRunsOnJit(u32 word) {
    const u32 no_rt = word & ~u32{0x1F};
    if ((word & 0xFFF00000U) == 0xD5300000U) { // MRS
        constexpr std::array<u32, 9> readable{
            0xD53BE000U, // CNTFRQ_EL0
            0xD53BE020U, // CNTPCT_EL0
            0xD53B0020U, // CTR_EL0
            0xD53B00E0U, // DCZID_EL0
            0xD53B4400U, // FPCR
            0xD53B4420U, // FPSR
            0xD53B4200U, // NZCV
            0xD53BD040U, // TPIDR_EL0
            0xD53BD060U, // TPIDRRO_EL0
        };
        return std::ranges::find(readable, no_rt) != readable.end();
    }
    if ((word & 0xFFF00000U) == 0xD5100000U) { // MSR (register)
        constexpr std::array<u32, 4> writable{
            0xD51B4400U, // FPCR
            0xD51B4420U, // FPSR
            0xD51B4200U, // NZCV
            0xD51BD040U, // TPIDR_EL0
        };
        return std::ranges::find(writable, no_rt) != writable.end();
    }
    return true;
}
static_assert(!PatchWordRunsOnJit(0xD53BE049U), "CNTVCT_EL0 must be refused");
static_assert(PatchWordRunsOnJit(0xD53BE029U), "CNTPCT_EL0 must pass");
static_assert(std::ranges::all_of(BUILTIN_PATCHES, [](const BuiltinPatch& patch) {
                  return std::ranges::all_of(patch.records, [](const BuiltinPatchRecord& record) {
                      if (record.module_offset % 4 != 0 || record.bytes.size() % 4 != 0) {
                          return true;
                      }
                      for (size_t i = 0; i < record.bytes.size(); i += 4) {
                          const u32 word = u32{record.bytes[i]} | (u32{record.bytes[i + 1]} << 8) |
                                           (u32{record.bytes[i + 2]} << 16) |
                                           (u32{record.bytes[i + 3]} << 24);
                          if (!PatchWordRunsOnJit(word)) {
                              return false;
                          }
                      }
                      return true;
                  });
              }),
              "a built-in patch reads or writes a system register the JIT does not translate");

// Whether [offset, offset + size) of a module (offsets after its NSO header) lies in its BSS.
bool InModuleBss(const Loader::NSOHeader& header, u64 offset, u64 size) {
    const auto& data = header.segments[2];
    const u64 bss_start = u64{data.location} + data.size;
    const u64 bss_end = bss_start + data.bss_size;
    return offset >= bss_start && offset <= bss_end && size <= bss_end - offset;
}

enum class TitleVersionFormat : u8 {
    ThreeElements, ///< vX.Y.Z
    FourElements,  ///< vX.Y.Z.W
};

std::string FormatTitleVersion(u32 version,
                               TitleVersionFormat format = TitleVersionFormat::ThreeElements) {
    std::array<u8, sizeof(u32)> bytes{};
    bytes[0] = static_cast<u8>(version % SINGLE_BYTE_MODULUS);
    for (std::size_t i = 1; i < bytes.size(); ++i) {
        version /= SINGLE_BYTE_MODULUS;
        bytes[i] = static_cast<u8>(version % SINGLE_BYTE_MODULUS);
    }

    if (format == TitleVersionFormat::FourElements) {
        return fmt::format("v{}.{}.{}.{}", bytes[3], bytes[2], bytes[1], bytes[0]);
    }
    return fmt::format("v{}.{}.{}", bytes[3], bytes[2], bytes[1]);
}

// Returns a directory with name matching name case-insensitive. Returns nullptr if directory
// doesn't have a directory with name.
VirtualDir FindSubdirectoryCaseless(const VirtualDir dir, std::string_view name) {
#ifdef _WIN32
    return dir->GetSubdirectory(name);
#else
    const auto subdirs = dir->GetSubdirectories();
    for (const auto& subdir : subdirs) {
        std::string dir_name = Common::ToLower(subdir->GetName());
        if (dir_name == name) {
            return subdir;
        }
    }

    return nullptr;
#endif
}

std::optional<std::vector<Core::Memory::CheatEntry>> ReadCheatFileFromFolder(
    u64 title_id, const PatchManager::BuildID& build_id_, const VirtualDir& base_path, bool upper) {
    const auto build_id_raw = Common::HexToString(build_id_, upper);
    const auto build_id = build_id_raw.substr(0, sizeof(u64) * 2);
    const auto file = base_path->GetFile(fmt::format("{}.txt", build_id));

    if (file == nullptr) {
        LOG_INFO(Common_Filesystem, "No cheats file found for title_id={:016X}, build_id={}",
                 title_id, build_id);
        return std::nullopt;
    }

    std::vector<u8> data(file->GetSize());
    if (file->Read(data.data(), data.size()) != data.size()) {
        LOG_INFO(Common_Filesystem, "Failed to read cheats file for title_id={:016X}, build_id={}",
                 title_id, build_id);
        return std::nullopt;
    }

    const Core::Memory::TextCheatParser parser;
    return parser.Parse(std::string_view(reinterpret_cast<const char*>(data.data()), data.size()));
}

void AppendCommaIfNotEmpty(std::string& to, std::string_view with) {
    if (to.empty()) {
        to += with;
    } else {
        to += ", ";
        to += with;
    }
}

bool IsDirValidAndNonEmpty(const VirtualDir& dir) {
    return dir != nullptr && (!dir->GetFiles().empty() || !dir->GetSubdirectories().empty());
}

bool IsVersionedExternalUpdateDisabled(const std::vector<std::string>& disabled, u32 version) {
    const std::string disabled_key = fmt::format("Update@{}", version);
    return std::find(disabled.cbegin(), disabled.cend(), disabled_key) != disabled.cend() ||
           std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();
}

std::string GetUpdateVersionStringFromSlot(const ContentProvider* provider, u64 update_tid) {
    if (provider == nullptr) {
        return {};
    }

    auto control_nca = provider->GetEntry(update_tid, ContentRecordType::Control);
    if (control_nca == nullptr ||
        control_nca->GetStatus() != Loader::ResultStatus::Success) {
        return {};
    }

    const auto romfs = control_nca->GetRomFS();
    if (romfs == nullptr) {
        return {};
    }

    const auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return {};
    }

    auto nacp_file = extracted->GetFile("control.nacp");
    if (nacp_file == nullptr) {
        nacp_file = extracted->GetFile("Control.nacp");
    }
    if (nacp_file == nullptr) {
        return {};
    }

    NACP nacp{nacp_file};
    return nacp.GetVersionString();
}
YUZU_NO_INLINE std::optional<u64> GetParentApplicationId(const ContentProvider* provider, u64 title_id) {
    return provider == nullptr ? std::nullopt : provider->GetParentApplicationId(title_id);
}

// The IPS and IPSwitch patches in a mod's exefs for the NSO with this build id (hex, padded with
// zeros to 64 digits).
std::vector<VirtualFile> ModPatchFiles(const VirtualDir& mod, const std::string& nso_build_id) {
    std::vector<VirtualFile> out;
    const auto exefs_dir = FindSubdirectoryCaseless(mod, "exefs");
    if (exefs_dir == nullptr) {
        return out;
    }
    for (const auto& file : exefs_dir->GetFiles()) {
        if (file->GetExtension() == "ips") {
            auto name = file->GetName();
            const auto this_build_id = fmt::format("{:0<64}", name.substr(0, name.find('.')));
            if (nso_build_id == this_build_id)
                out.push_back(file);
        } else if (file->GetExtension() == "pchtxt") {
            IPSwitchCompiler compiler{file};
            const auto this_build_id = Common::HexToString(compiler.GetBuildID());
            if (nso_build_id == this_build_id)
                out.push_back(file);
        }
    }
    return out;
}

// Whether a mod is Dark's standalone Dynamic FPS, which Fritchii has built in: one of its patches
// for Ocarina's main writes into the built-in's records (the call or the routine). Such a mod is
// left out (not applied, not listed in Add-Ons) so the built-in, which the high refresh rate mode
// switches live, is applied in its place. Mods overlapping the other built-ins still win.
bool IsBuiltinDynamicFpsMod(u64 title_id, const VirtualDir& mod) {
    const auto it = std::ranges::find(BUILTIN_PATCHES, DYNAMIC_FPS_NAME, &BuiltinPatch::name);
    if (it == BUILTIN_PATCHES.end() || it->title_id != title_id || mod == nullptr) {
        return false;
    }
    // A player who turned the built-in off in Add-Ons keeps their mod (as before), or they would
    // get no Dynamic FPS at all.
    if (const auto disabled = Settings::values.disabled_addons.find(title_id);
        disabled != Settings::values.disabled_addons.end() &&
        std::ranges::find(disabled->second, std::string{DYNAMIC_FPS_NAME}) != disabled->second.end()) {
        return false;
    }
    const auto overlaps = [&](const std::pair<u64, u64>& range) {
        return std::ranges::any_of(it->records, [&](const BuiltinPatchRecord& record) {
            const u64 begin = sizeof(Loader::NSOHeader) + record.module_offset;
            return range.first < begin + record.bytes.size() && begin < range.second;
        });
    };
    for (const auto& file : ModPatchFiles(mod, fmt::format("{:0<64}", it->build_id))) {
        const auto ranges = file->GetExtension() == "ips" ? GetIPSWriteRanges(file)
                                                          : IPSwitchCompiler{file}.GetWriteRanges();
        if (std::ranges::any_of(ranges, overlaps)) {
            return true;
        }
    }
    return false;
}

// What a mod folder holds, as Add-Ons lists it ("IPS, LayeredFS"); empty when nothing loads.
std::string ModTypes(const VirtualDir& mod) {
    std::string types;
    const auto exefs_dir = FindSubdirectoryCaseless(mod, "exefs");
    if (IsDirValidAndNonEmpty(exefs_dir)) {
        bool ips = false;
        bool ipswitch = false;
        bool layeredfs = false;
        for (const auto& file : exefs_dir->GetFiles()) {
            if (file->GetExtension() == "ips") {
                ips = true;
            } else if (file->GetExtension() == "pchtxt") {
                ipswitch = true;
            } else if (std::find(EXEFS_FILE_NAMES.begin(), EXEFS_FILE_NAMES.end(),
                                 file->GetName()) != EXEFS_FILE_NAMES.end()) {
                layeredfs = true;
            }
        }
        if (ips)
            AppendCommaIfNotEmpty(types, "IPS");
        if (ipswitch)
            AppendCommaIfNotEmpty(types, "IPSwitch");
        if (layeredfs)
            AppendCommaIfNotEmpty(types, "LayeredExeFS");
    }
    if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfs")) ||
        IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfslite")))
        AppendCommaIfNotEmpty(types, "LayeredFS");
    if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "romfs_ext")))
        AppendCommaIfNotEmpty(types, "ExtLayeredFS");
    if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(mod, "cheats")))
        AppendCommaIfNotEmpty(types, "Cheats");
    return types;
}

// The same for the SD card's atmosphere/contents/<title> folder, listed as one add-on "SDMC".
std::string SdmcModTypes(const VirtualDir& sdmc_mod_dir) {
    std::string types;
    if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "exefs")))
        AppendCommaIfNotEmpty(types, "LayeredExeFS");
    if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfs")) ||
        IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfslite")))
        AppendCommaIfNotEmpty(types, "LayeredFS");
    if (IsDirValidAndNonEmpty(FindSubdirectoryCaseless(sdmc_mod_dir, "romfs_ext")))
        AppendCommaIfNotEmpty(types, "ExtLayeredFS");
    return types;
}

std::mutex addons_mutex;
std::function<void(u64)> new_mods_callback;

// New mods start disabled: a mod (a folder or cheat_ file in load/<title>/, or the SD card's
// folder) Fritchii has not seen for this title before goes on the disabled list, and both lists
// are saved at once. Seen mods keep the player's choice. Updates, DLC and the built-ins are
// never touched (Max Resolution's default is set once in the config, frontend_common/config.cpp).
void DisableNewModsFor(u64 title_id, const VirtualDir& mod_dir, const VirtualDir& sdmc_mod_dir) {
    if (title_id == 0) {
        return;
    }
    std::vector<std::string> seen;
    {
        std::scoped_lock lock{addons_mutex};
        if (!Settings::values.seen_addons_tracking) {
            return;
        }
        if (const auto it = Settings::values.seen_addons.find(title_id);
            it != Settings::values.seen_addons.end()) {
            seen = it->second;
        }
    }
    const auto is_seen = [&](const std::string& name) {
        return std::find(seen.begin(), seen.end(), name) != seen.end();
    };
    // File reads outside the lock. `keep`: seen, state left as is (Dark's own Dynamic FPS, which
    // the built-in replaces while that is on: if the player turns the built-in off, theirs runs).
    std::vector<std::string> fresh;
    std::vector<std::string> keep;
    if (mod_dir != nullptr) {
        for (const auto& file : mod_dir->GetFiles()) {
            if (auto name = file->GetName(); name.starts_with("cheat_") && !is_seen(name)) {
                fresh.push_back(std::move(name));
            }
        }
        for (const auto& mod : mod_dir->GetSubdirectories()) {
            auto name = mod->GetName();
            if (is_seen(name) || ModTypes(mod).empty()) {
                continue;
            }
            (IsBuiltinDynamicFpsMod(title_id, mod) ? keep : fresh).push_back(std::move(name));
        }
    }
    if (sdmc_mod_dir != nullptr && !is_seen("SDMC") && !SdmcModTypes(sdmc_mod_dir).empty()) {
        fresh.emplace_back("SDMC");
    }
    if (fresh.empty() && keep.empty()) {
        return;
    }

    std::vector<std::string> disabled_now;
    std::function<void(u64)> callback;
    {
        std::scoped_lock lock{addons_mutex};
        callback = new_mods_callback;
        auto& seen_list = Settings::values.seen_addons[title_id];
        auto& disabled = Settings::values.disabled_addons[title_id];
        const auto add = [](std::vector<std::string>& list, const std::string& name) {
            if (std::find(list.begin(), list.end(), name) != list.end()) {
                return false;
            }
            list.push_back(name);
            return true;
        };
        for (const auto& name : keep) {
            add(seen_list, name);
        }
        for (const auto& name : fresh) {
            if (add(seen_list, name) && add(disabled, name)) {
                disabled_now.push_back(name);
            }
        }
    }
    for (const auto& name : disabled_now) {
        LOG_INFO(Loader,
                 "Mods: new mod {} for {:016X} starts disabled; enable it in the game's Add-Ons",
                 name, title_id);
    }
    // Saved even when only `keep` changed, so the seen list survives a crash.
    if (callback) {
        callback(title_id);
    }
}
} // Anonymous namespace

void PatchManager::SetNewModsCallback(std::function<void(u64 title_id)> callback) {
    std::scoped_lock lock{addons_mutex};
    new_mods_callback = std::move(callback);
}

std::unique_lock<std::mutex> PatchManager::LockAddOns() {
    return std::unique_lock{addons_mutex};
}

void PatchManager::DisableNewMods() const {
    {
        std::scoped_lock lock{addons_mutex};
        if (!Settings::values.seen_addons_tracking) {
            return;
        }
    }
    DisableNewModsFor(title_id, fs_controller.GetModificationLoadRoot(title_id),
                      fs_controller.GetSDMCModificationLoadRoot(title_id));
    if (parent_title_id) {
        DisableNewModsFor(*parent_title_id,
                          fs_controller.GetModificationLoadRoot(*parent_title_id),
                          fs_controller.GetSDMCModificationLoadRoot(*parent_title_id));
    }
}

PatchManager::PatchManager(u64 title_id_,
                           const Service::FileSystem::FileSystemController& fs_controller_,
                           const ContentProvider& content_provider_)
    : title_id{title_id_}, parent_title_id{GetParentApplicationId(std::addressof(content_provider_), title_id_)}, fs_controller{fs_controller_}, content_provider{content_provider_} {}

PatchManager::~PatchManager() = default;

u64 PatchManager::GetTitleID() const {
    return title_id;
}

VirtualDir PatchManager::GetModificationLoadRoot(bool sdmc) const {
    const auto get_root = [&](u64 id) {
        return sdmc ? fs_controller.GetSDMCModificationLoadRoot(id) : fs_controller.GetModificationLoadRoot(id);
    };
    auto root = get_root(title_id);
    if (!parent_title_id) {
        return root;
    }
    std::vector<VirtualDir> roots{std::move(root), get_root(*parent_title_id)};
    std::erase(roots, nullptr);
    return LayeredVfsDirectory::MakeLayeredDirectory(std::move(roots));
}

std::vector<std::string> PatchManager::GetDisabledAddons() const {
    DisableNewMods();
    const auto lock = LockAddOns();
    auto disabled = Settings::values.disabled_addons[title_id];
    if (parent_title_id) {
        const auto& shared = Settings::values.disabled_addons[*parent_title_id];
        disabled.insert(disabled.end(), shared.begin(), shared.end());
    }
    return disabled;
}

VirtualDir PatchManager::PatchExeFS(VirtualDir exefs) const {
    LOG_INFO(Loader, "Patching ExeFS for title_id={:016X}", title_id);

    if (exefs == nullptr)
        return exefs;

    const auto disabled = GetDisabledAddons();

    bool update_disabled = true;
    std::optional<u32> enabled_version;
    bool checked_external = false;
    bool checked_manual = false;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);
    const auto update_tid = GetUpdateTitleID(title_id);

    if (content_union) {
        // First, check ExternalContentProvider
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            if (!update_versions.empty()) {
                checked_external = true;
                for (const auto& update_entry : update_versions) {
                    if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                        update_disabled = false;
                        enabled_version = update_entry.version;
                        break;
                    }
                }
            }
        }

        // Also check ManualContentProvider (for Android)
        if (!checked_external) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

                if (!manual_update_versions.empty()) {
                    checked_manual = true;
                    for (const auto& update_entry : manual_update_versions) {
                        if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                            update_disabled = false;
                            enabled_version = update_entry.version;
                            break;
                        }
                    }
                }
            }
        }
    }

    // check for original NAND style
    // Check NAND if: no external updates exist, OR all external updates are disabled
    if (!checked_external && !checked_manual) {
        // Only enable NAND update if it exists AND is not disabled
        // We need to check if an update actually exists in the content provider
        const bool has_nand_update = content_provider.HasEntry(update_tid, ContentRecordType::Program);

        if (has_nand_update) {
            const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
            const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();
            const bool generic_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();

            if (!nand_disabled && !sdmc_disabled && !generic_disabled) {
                update_disabled = false;
            }
        }
    } else if (update_disabled && content_union) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();

        if (!nand_disabled || !sdmc_disabled) {
            const auto nand_sdmc_entries = content_union->ListEntriesFilterOrigin(
                std::nullopt, TitleType::Update, ContentRecordType::Program, update_tid);

            for (const auto& [slot, entry] : nand_sdmc_entries) {
                if (slot == ContentProviderUnionSlot::UserNAND ||
                    slot == ContentProviderUnionSlot::SysNAND) {
                    if (!nand_disabled) {
                        update_disabled = false;
                        break;
                    }
                } else if (slot == ContentProviderUnionSlot::SDMC) {
                    if (!sdmc_disabled) {
                        update_disabled = false;
                        break;
                    }
                }
            }
        }
    }

    // Game Updates
    std::unique_ptr<NCA> update = nullptr;

    // If we have a specific enabled version from external provider, use it
    if (enabled_version.has_value() && content_union) {
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            auto file = external_provider->GetEntryForVersion(update_tid, ContentRecordType::Program, *enabled_version);
            if (file != nullptr) {
                update = std::make_unique<NCA>(file);
            }
        }

        // Also try ManualContentProvider
        if (update == nullptr) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                auto file = manual_provider->GetEntryForVersion(update_tid, ContentRecordType::Program, *enabled_version);
                if (file != nullptr) {
                    update = std::make_unique<NCA>(file);
                }
            }
        }
    }

    // Fallback to regular content provider if no external update was loaded
    if (update == nullptr && !update_disabled) {
        update = content_provider.GetEntry(update_tid, ContentRecordType::Program);
    }

    if (!update_disabled && update != nullptr && update->GetExeFS() != nullptr) {
        LOG_INFO(Loader, "    ExeFS: Update ({}) applied successfully",
                 FormatTitleVersion(content_provider.GetEntryVersion(update_tid).value_or(0)));
        exefs = update->GetExeFS();
    }

    // LayeredExeFS
    const auto load_dir = GetModificationLoadRoot();
    const auto sdmc_load_dir = GetModificationLoadRoot(true);

    std::vector<VirtualDir> patch_dirs;
    if (sdmc_load_dir != nullptr) {
        patch_dirs.push_back(sdmc_load_dir);
    }
    if (load_dir != nullptr) {
        const auto load_patch_dirs = load_dir->GetSubdirectories();
        patch_dirs.insert(patch_dirs.end(), load_patch_dirs.begin(), load_patch_dirs.end());
    }

    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    std::vector<VirtualDir> layers;
    layers.reserve(patch_dirs.size() + 1);
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.begin(), disabled.end(), subdir->GetName()) != disabled.end())
            continue;

        auto exefs_dir = FindSubdirectoryCaseless(subdir, "exefs");
        if (exefs_dir != nullptr)
            layers.push_back(std::move(exefs_dir));
    }
    layers.push_back(exefs);

    auto layered = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers));
    if (layered != nullptr) {
        LOG_INFO(Loader, "    ExeFS: LayeredExeFS patches applied successfully");
        exefs = std::move(layered);
    }

    if (Settings::values.dump_exefs) {
        LOG_INFO(Loader, "Dumping ExeFS for title_id={:016X}", title_id);
        const auto dump_dir = fs_controller.GetModificationDumpRoot(title_id);
        if (dump_dir != nullptr) {
            const auto exefs_dir = GetOrCreateDirectoryRelative(dump_dir, "/exefs");
            VfsRawCopyD(exefs, exefs_dir);
        }
    }

    return exefs;
}

std::vector<VirtualFile> PatchManager::CollectPatches(const std::vector<VirtualDir>& patch_dirs, const std::string& build_id) const {
    const auto disabled = GetDisabledAddons();
    const auto nso_build_id = fmt::format("{:0<64}", build_id);

    std::vector<VirtualFile> out;
    out.reserve(patch_dirs.size());
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) != disabled.cend())
            continue;

        const auto files = ModPatchFiles(subdir, nso_build_id);
        out.insert(out.end(), files.begin(), files.end());
    }
    return out;
}

std::vector<u8> PatchManager::PatchNSO(const std::vector<u8>& nso, const std::string& name) const {
    if (name == "main") {
        Settings::SetAppliedOcarinaRenderHeight(std::nullopt);
    }
    if (nso.size() < sizeof(Loader::NSOHeader)) {
        return nso;
    }

    Loader::NSOHeader header;
    std::memcpy(&header, nso.data(), sizeof(header));

    if (header.magic != Common::MakeMagic('N', 'S', 'O', '0')) {
        return nso;
    }

    const auto build_id_raw = Common::HexToString(header.build_id);
    const auto build_id = build_id_raw.substr(0, build_id_raw.find_last_not_of('0') + 1);

    if (Settings::values.dump_nso) {
        LOG_INFO(Loader, "Dumping NSO for name={}, build_id={}, title_id={:016X}", name, build_id,
                 title_id);
        const auto dump_dir = fs_controller.GetModificationDumpRoot(title_id);
        if (dump_dir != nullptr) {
            const auto nso_dir = GetOrCreateDirectoryRelative(dump_dir, "/nso");
            const auto file = nso_dir->CreateFile(fmt::format("{}-{}.nso", name, build_id));

            file->Resize(nso.size());
            file->WriteBytes(nso);
        }
    }

    LOG_INFO(Loader, "Patching NSO for name={}, build_id={}", name, build_id);

    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return nso;
    }

    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });
    // Dark's standalone Dynamic FPS: the built-in is applied instead (the player's files stay).
    std::erase_if(patch_dirs, [&](const VirtualDir& mod) {
        if (!IsBuiltinDynamicFpsMod(title_id, mod)) {
            return false;
        }
        if (name == "main") {
            LOG_INFO(Loader,
                     "    - Skipping mod \"{}\": it is Dark's Dynamic FPS, which Fritchii has "
                     "built in",
                     mod->GetName());
        }
        return true;
    });
    const auto patches = CollectPatches(patch_dirs, build_id);

    auto out = nso;
    // Which mod wrote over a built-in patch's bytes: the skip line and Ctrl+U name it. Only the
    // built-in records' bytes are compared before and after each mod (a few KiB, not the module).
    std::map<size_t, std::string> builtin_owner;
    const auto builtin_bytes = [&] {
        std::vector<std::pair<size_t, std::vector<u8>>> bytes;
        for (const auto& patch : BUILTIN_PATCHES) {
            if (patch.title_id != title_id || patch.module != name) {
                continue;
            }
            for (const BuiltinPatchRecord& record : patch.records) {
                const size_t offset = sizeof(Loader::NSOHeader) + record.module_offset;
                if (out.size() >= offset + record.bytes.size()) {
                    bytes.emplace_back(record.module_offset,
                                       std::vector<u8>(out.begin() + offset,
                                                       out.begin() + offset + record.bytes.size()));
                }
            }
        }
        return bytes;
    };
    const auto note_owner = [&](const std::vector<std::pair<size_t, std::vector<u8>>>& before,
                                const std::string& mod) {
        for (const auto& [module_offset, old_bytes] : before) {
            const size_t offset = sizeof(Loader::NSOHeader) + module_offset;
            if (out.size() >= offset + old_bytes.size() &&
                !std::equal(old_bytes.begin(), old_bytes.end(), out.begin() + offset)) {
                builtin_owner.try_emplace(module_offset, mod);
            }
        }
    };
    for (const auto& patch_file : patches) {
        const std::string mod_name =
            patch_file->GetContainingDirectory()->GetParentDirectory()->GetName();
        const auto before = builtin_bytes();
        if (patch_file->GetExtension() == "ips") {
            LOG_INFO(Loader, "    - Applying IPS patch from mod \"{}\"",
                     patch_file->GetContainingDirectory()->GetParentDirectory()->GetName());
            const auto patched = PatchIPS(std::make_shared<VectorVfsFile>(out), patch_file);
            if (patched != nullptr)
                out = patched->ReadAllBytes();
        } else if (patch_file->GetExtension() == "pchtxt") {
            LOG_INFO(Loader, "    - Applying IPSwitch patch from mod \"{}\"",
                     patch_file->GetContainingDirectory()->GetParentDirectory()->GetName());
            const IPSwitchCompiler compiler{patch_file};
            const auto patched = compiler.Apply(std::make_shared<VectorVfsFile>(out));
            if (patched != nullptr)
                out = patched->ReadAllBytes();
        }
        note_owner(before, mod_name);
    }
    if (title_id == 0x0400486024DEC000ULL && name == "main") {
        std::scoped_lock lock{g_high_refresh_blocked_mutex};
        g_high_refresh_blocked_by.clear();
    }

    const auto disabled = GetDisabledAddons();
    std::vector<std::string_view> replaced;
    std::vector<std::string_view> applied;
    for (const auto& patch : BUILTIN_PATCHES) {
        if (patch.title_id != title_id || patch.module != name) {
            continue;
        }
        if (patch.enable_env != nullptr) {
            const char* const value = std::getenv(patch.enable_env);
            if (value == nullptr || std::string_view{value} != "1") {
                continue;
            }
        }
        if (patch.setting_enabled != nullptr && !patch.setting_enabled()) {
            continue;
        }
        if (std::find(replaced.cbegin(), replaced.cend(), patch.name) != replaced.cend()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": included in an applied built-in patch",
                     patch.name);
            continue;
        }
        // disable_env=1 skips the patch, =0 applies it even when Add-Ons has it off (A/B runs).
        const char* const ab_value = patch.disable_env ? std::getenv(patch.disable_env) : nullptr;
        const std::string_view ab{ab_value ? ab_value : ""};
        if (ab == "1") {
            LOG_INFO(Loader, "    - Skipping \"{}\": {}=1", patch.name, patch.disable_env);
            continue;
        }
        // A render height unticked in an earlier build counts for nothing: the row is gone.
        if (ab != "0" && patch.render_height == 0 &&
            std::find(disabled.cbegin(), disabled.cend(), patch.name) != disabled.cend()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": disabled in Add-Ons", patch.name);
            continue;
        }
        if (patch.build_id != build_id) {
            LOG_INFO(Loader, "    - Skipping \"{}\": build id {} is not {}", patch.name, build_id,
                     patch.build_id);
            continue;
        }
        if (patch.render_height != 0 &&
            std::find(applied.cbegin(), applied.cend(), MAX_RESOLUTION_NAME) == applied.cend()) {
            LOG_INFO(Loader, "    - Skipping \"{}\": it needs \"{}\"", patch.name,
                     MAX_RESOLUTION_NAME);
            continue;
        }
        if (patch.state_size != 0 && !InModuleBss(header, patch.state_offset, patch.state_size)) {
            LOG_WARNING(Loader, "    - Skipping \"{}\": its state at {}+{:#x} is outside BSS",
                        patch.name, name, patch.state_offset);
            continue;
        }
        // Every record must fit and hold its original bytes, and none may have been patched by a
        // user mod already (the mod's patch is used then, for all records).
        bool apply = true;
        for (const BuiltinPatchRecord& record : patch.records) {
            if (BuiltinRecordSkipped(record)) {
                continue;
            }
            const size_t offset = sizeof(Loader::NSOHeader) + record.module_offset;
            if (nso.size() < offset + record.bytes.size() ||
                out.size() < offset + record.bytes.size()) {
                LOG_WARNING(Loader, "    - Skipping \"{}\": {}+{:#x} is past the module's end",
                            patch.name, name, record.module_offset);
                apply = false;
                break;
            }
            if (record.checked) {
                u32 original;
                std::memcpy(&original, nso.data() + offset, sizeof(original));
                if (original != record.original) {
                    LOG_WARNING(Loader,
                                "    - Skipping \"{}\": {}+{:#x} holds {:#010x}, expected {:#010x}",
                                patch.name, name, record.module_offset, original, record.original);
                    apply = false;
                    break;
                }
            }
            if (!std::equal(nso.begin() + offset, nso.begin() + offset + record.bytes.size(),
                            out.begin() + offset)) {
                const auto owner = builtin_owner.find(record.module_offset);
                const std::string mod =
                    owner != builtin_owner.end() ? owner->second : std::string{"a user mod"};
                LOG_INFO(Loader,
                         "    - Skipping \"{}\": the mod \"{}\" already patched {}+{:#x}, the "
                         "mod's patch is used",
                         patch.name, mod, name, record.module_offset);
                if (patch.name == HIGH_REFRESH_NAME) {
                    std::scoped_lock lock{g_high_refresh_blocked_mutex};
                    g_high_refresh_blocked_by = mod;
                }
                apply = false;
                break;
            }
        }
        if (!apply) {
            continue;
        }
        for (const BuiltinPatchRecord& record : patch.records) {
            if (BuiltinRecordSkipped(record)) {
                // (a word kept for its switch, skipped while the switch is not 1: no line)
                if (record.skip_env != nullptr) {
                    LOG_INFO(Loader, "    - \"{}\": leaving {}+{:#x} out, {}=1", patch.name, name,
                             record.module_offset, record.skip_env);
                }
                continue;
            }
            std::memcpy(out.data() + sizeof(Loader::NSOHeader) + record.module_offset,
                        record.bytes.data(), record.bytes.size());
        }
        LOG_INFO(Loader, "    - Applying built-in patch \"{}\" ({} records) to {}", patch.name,
                 patch.records.size(), name);
        if (!patch.replaces.empty()) {
            replaced.push_back(patch.replaces);
        }
        applied.push_back(patch.name);
        if (patch.render_height != 0) {
            Settings::SetAppliedOcarinaRenderHeight(patch.render_height);
        }
    }

    if (out.size() < sizeof(Loader::NSOHeader)) {
        return nso;
    }

    std::memcpy(out.data(), &header, sizeof(header));
    return out;
}

bool PatchManager::HasNSOPatch(const BuildID& build_id_, std::string_view name) const {
    const auto build_id_raw = Common::HexToString(build_id_);
    const auto build_id = build_id_raw.substr(0, build_id_raw.find_last_not_of('0') + 1);

    LOG_INFO(Loader, "Querying NSO patch existence for build_id={}, name={}", build_id, name);

    if (std::ranges::any_of(BUILTIN_PATCHES, [&](const BuiltinPatch& patch) {
            return patch.title_id == title_id && patch.module == name;
        })) {
        return true;
    }

    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return false;
    }

    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    return !CollectPatches(patch_dirs, build_id).empty();
}

std::vector<Core::Memory::CheatEntry> PatchManager::CreateCheatList(const BuildID& build_id_) const {
    const auto load_dir = GetModificationLoadRoot();
    if (load_dir == nullptr) {
        LOG_ERROR(Loader, "Cannot load mods for invalid title_id={:016X}", title_id);
        return {};
    }

    const auto disabled = GetDisabledAddons();
    auto patch_dirs = load_dir->GetSubdirectories();
    std::sort(patch_dirs.begin(), patch_dirs.end(), [](auto const& l, auto const& r) { return l->GetName() < r->GetName(); });

    // <mod dir> / <folder> / cheats / <build id>.txt
    std::vector<Core::Memory::CheatEntry> out;
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) == disabled.cend()) {
            if (auto cheats_dir = FindSubdirectoryCaseless(subdir, "cheats"); cheats_dir != nullptr) {
                if (auto const res = ReadCheatFileFromFolder(title_id, build_id_, cheats_dir, true))
                    std::copy(res->begin(), res->end(), std::back_inserter(out));
                if (auto const res = ReadCheatFileFromFolder(title_id, build_id_, cheats_dir, false))
                    std::copy(res->begin(), res->end(), std::back_inserter(out));
            }
        }
    }
    // Uncareless user-friendly loading of patches (must start with 'cheat_')
    // <mod dir> / <cheat file>.txt
    for (auto const& f : load_dir->GetFiles()) {
        auto const name = f->GetName();
        if (name.starts_with("cheat_") && std::find(disabled.cbegin(), disabled.cend(), name) == disabled.cend()) {
            std::vector<u8> data(f->GetSize());
            if (f->Read(data.data(), data.size()) == data.size()) {
                const Core::Memory::TextCheatParser parser;
                auto const res = parser.Parse(std::string_view(reinterpret_cast<const char*>(data.data()), data.size()));
                std::copy(res.begin(), res.end(), std::back_inserter(out));
            } else {
                LOG_INFO(Common_Filesystem, "Failed to read cheats file for title_id={:016X}", title_id);
            }
        }
    }
    return out;
}

void PatchManager::ApplyLayeredFS(VirtualFile& romfs, ContentRecordType type) const {
    const auto load_dir = GetModificationLoadRoot();
    const auto sdmc_load_dir = GetModificationLoadRoot(true);
    if ((type != ContentRecordType::Program && type != ContentRecordType::Data &&
         type != ContentRecordType::HtmlDocument) ||
        (load_dir == nullptr && sdmc_load_dir == nullptr)) {
        return;
    }

    const auto disabled = GetDisabledAddons();
    std::vector<VirtualDir> patch_dirs;
    if (load_dir != nullptr) {
        patch_dirs = load_dir->GetSubdirectories();
    }
    if (std::find(disabled.cbegin(), disabled.cend(), "SDMC") == disabled.cend() &&
        sdmc_load_dir != nullptr) {
        patch_dirs.push_back(sdmc_load_dir);
    }
    std::sort(patch_dirs.begin(), patch_dirs.end(),
              [](const VirtualDir& l, const VirtualDir& r) { return l->GetName() < r->GetName(); });

    std::vector<VirtualDir> layers;
    std::vector<VirtualDir> layers_ext;
    layers.reserve(patch_dirs.size() + 1);
    layers_ext.reserve(patch_dirs.size() + 1);
    for (const auto& subdir : patch_dirs) {
        if (std::find(disabled.cbegin(), disabled.cend(), subdir->GetName()) != disabled.cend()) {
            continue;
        }

        auto romfs_dir = FindSubdirectoryCaseless(subdir, "romfs");
        if (romfs_dir != nullptr)
            layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(romfs_dir)));

        // Support for romfslite introduced in Atmosphere 1.9.5
        auto romfslite_dir = FindSubdirectoryCaseless(subdir, "romfslite");
        if (romfslite_dir != nullptr)
            layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(romfslite_dir)));

        auto ext_dir = FindSubdirectoryCaseless(subdir, "romfs_ext");
        if (ext_dir != nullptr)
            layers_ext.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(ext_dir)));

        if (type == ContentRecordType::HtmlDocument) {
            auto manual_dir = FindSubdirectoryCaseless(subdir, "manual_html");
            if (manual_dir != nullptr)
                layers.emplace_back(std::make_shared<CachedVfsDirectory>(std::move(manual_dir)));
        }
    }

    // When there are no layers to apply, return early as there is no need to rebuild the RomFS
    if (layers.empty() && layers_ext.empty()) {
        return;
    }

    auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return;
    }

    layers.emplace_back(std::move(extracted));

    auto layered = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers));
    auto layered_ext = LayeredVfsDirectory::MakeLayeredDirectory(std::move(layers_ext));
    if (layered == nullptr && layered_ext == nullptr) {
        return;
    }

    auto packed = CreateRomFS(std::move(layered), std::move(layered_ext));
    if (packed == nullptr) {
        return;
    }

    LOG_INFO(Loader, "    RomFS: LayeredFS patches applied successfully");
    romfs = std::move(packed);
}

VirtualFile PatchManager::PatchRomFS(const NCA* base_nca, VirtualFile base_romfs,
                                     ContentRecordType type, VirtualFile packed_update_raw,
                                     bool apply_layeredfs) const {
    const auto log_string = fmt::format("Patching RomFS for title_id={:016X}, type={:02X}",
                                        title_id, static_cast<u8>(type));
    if (type == ContentRecordType::Program || type == ContentRecordType::Data) {
        LOG_INFO(Loader, "{}", log_string);
    } else {
        LOG_DEBUG(Loader, "{}", log_string);
    }

    auto romfs = base_romfs;

    // Game Updates
    const auto update_tid = GetUpdateTitleID(title_id);
    const auto disabled = GetDisabledAddons();

    bool update_disabled = true;
    std::optional<u32> enabled_version;
    VirtualFile update_raw = nullptr;
    bool checked_external = false;
    bool checked_manual = false;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);
    if (content_union) {
        // First, check ExternalContentProvider
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            if (!update_versions.empty()) {
                checked_external = true;
                for (const auto& update_entry : update_versions) {
                    if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                        update_disabled = false;
                        enabled_version = update_entry.version;
                        update_raw = external_provider->GetEntryForVersion(update_tid, type, update_entry.version);
                        break;
                    }
                }
            }
        }

        if (!checked_external) {
            const auto* manual_provider = static_cast<const ManualContentProvider*>(
                content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
            if (manual_provider) {
                const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

                if (!manual_update_versions.empty()) {
                    checked_manual = true;
                    for (const auto& update_entry : manual_update_versions) {
                        if (!IsVersionedExternalUpdateDisabled(disabled, update_entry.version)) {
                            update_disabled = false;
                            enabled_version = update_entry.version;
                            update_raw = manual_provider->GetEntryForVersion(update_tid, type, update_entry.version);
                            break;
                        }
                    }
                }
            }
        }
    }

    if (!checked_external && !checked_manual) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();
        const bool generic_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();

        if (!nand_disabled && !sdmc_disabled && !generic_disabled) {
            update_disabled = false;
        }
        if (!update_disabled) {
            update_raw = content_provider.GetEntryRaw(update_tid, type);
        }
    } else if (update_disabled && content_union) {
        const bool nand_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (NAND)") != disabled.cend();
        const bool sdmc_disabled = std::find(disabled.cbegin(), disabled.cend(), "Update (SDMC)") != disabled.cend();

        if (!nand_disabled || !sdmc_disabled) {
            const auto nand_sdmc_entries = content_union->ListEntriesFilterOrigin(
                std::nullopt, TitleType::Update, type, update_tid);

            for (const auto& [slot, entry] : nand_sdmc_entries) {
                if (slot == ContentProviderUnionSlot::UserNAND ||
                    slot == ContentProviderUnionSlot::SysNAND) {
                    if (!nand_disabled) {
                        update_disabled = false;
                        update_raw = content_provider.GetEntryRaw(update_tid, type);
                        break;
                    }
                } else if (slot == ContentProviderUnionSlot::SDMC) {
                    if (!sdmc_disabled) {
                        update_disabled = false;
                        update_raw = content_provider.GetEntryRaw(update_tid, type);
                        break;
                    }
                }
            }
        }
    }

    if (!update_disabled && update_raw != nullptr && base_nca != nullptr) {
        const auto new_nca = std::make_shared<NCA>(update_raw, base_nca);
        if (new_nca->GetStatus() == Loader::ResultStatus::Success &&
            new_nca->GetRomFS() != nullptr) {
            LOG_INFO(Loader, "    RomFS: Update ({}) applied successfully",
                     enabled_version.has_value() ? FormatTitleVersion(*enabled_version) :
                     FormatTitleVersion(content_provider.GetEntryVersion(update_tid).value_or(0)));
            romfs = new_nca->GetRomFS();
        }
    } else if (!update_disabled && packed_update_raw != nullptr && base_nca != nullptr) {
        const auto new_nca = std::make_shared<NCA>(packed_update_raw, base_nca);
        if (new_nca->GetStatus() == Loader::ResultStatus::Success &&
            new_nca->GetRomFS() != nullptr) {
            LOG_INFO(Loader, "    RomFS: Update (PACKED) applied successfully");
            romfs = new_nca->GetRomFS();
        }
    }

    // LayeredFS
    if (apply_layeredfs) {
        ApplyLayeredFS(romfs, type);
    }

    return romfs;
}

std::vector<Patch> PatchManager::GetPatches(VirtualFile update_raw) const {
    if (title_id == 0) {
        return {};
    }

    std::vector<Patch> out;
    DisableNewMods();
    const auto disabled = [this] {
        const auto lock = LockAddOns();
        return Settings::values.disabled_addons[title_id];
    }();

    // Game Updates
    const auto update_tid = GetUpdateTitleID(title_id);

    std::vector<Patch> external_update_patches;

    const auto* content_union = static_cast<const ContentProviderUnion*>(&content_provider);

    if (content_union) {
        // First, check ExternalContentProvider for updates
        const auto* external_provider = content_union->GetExternalProvider();
        if (external_provider) {
            const auto update_versions = external_provider->ListUpdateVersions(update_tid);

            for (const auto& update_entry : update_versions) {
                std::string version_str = update_entry.version_string;
                if (version_str.empty()) {
                    version_str = FormatTitleVersion(update_entry.version);
                }

                const auto update_disabled =
                    IsVersionedExternalUpdateDisabled(disabled, update_entry.version);

                Patch update_patch = {.enabled = !update_disabled,
                                      .name = "Update",
                                      .version = version_str,
                                      .type = PatchType::Update,
                                      .program_id = title_id,
                                      .title_id = update_tid,
                                      .source = PatchSource::External,
                                      .numeric_version = update_entry.version};

                external_update_patches.push_back(update_patch);
            }
        }

        const auto* manual_provider = static_cast<const ManualContentProvider*>(
            content_union->GetSlotProvider(ContentProviderUnionSlot::FrontendManual));
        if (manual_provider && external_update_patches.empty()) {
            const auto manual_update_versions = manual_provider->ListUpdateVersions(update_tid);

            for (const auto& update_entry : manual_update_versions) {
                std::string version_str = update_entry.version_string;
                if (version_str.empty()) {
                    version_str = FormatTitleVersion(update_entry.version);
                }

                const auto update_disabled =
                    IsVersionedExternalUpdateDisabled(disabled, update_entry.version);


                Patch update_patch = {.enabled = !update_disabled,
                                      .name = "Update",
                                      .version = version_str,
                                      .type = PatchType::Update,
                                      .program_id = title_id,
                                      .title_id = update_tid,
                                      .source = PatchSource::External,
                                      .numeric_version = update_entry.version};

                external_update_patches.push_back(update_patch);
            }
        }

        if (external_update_patches.size() > 1) {
            bool found_enabled = false;
            for (auto& patch : external_update_patches) {
                if (patch.enabled) {
                    if (found_enabled) {
                        patch.enabled = false;
                    } else {
                        found_enabled = true;
                    }
                }
            }
        }

        for (auto& patch : external_update_patches) {
            out.push_back(std::move(patch));
        }

        const auto all_updates = content_union->ListEntriesFilterOrigin(
            std::nullopt, std::nullopt, ContentRecordType::Program, update_tid);

        for (const auto& [slot, entry] : all_updates) {
            (void)entry;
            if (slot == ContentProviderUnionSlot::External ||
                slot == ContentProviderUnionSlot::FrontendManual) {
                continue;
            }

            PatchSource source_type = PatchSource::Unknown;
            std::string source_suffix;

            switch (slot) {
                case ContentProviderUnionSlot::UserNAND:
                case ContentProviderUnionSlot::SysNAND:
                    source_type = PatchSource::NAND;
                    source_suffix = " (NAND)";
                    break;
                case ContentProviderUnionSlot::SDMC:
                    source_type = PatchSource::SDMC;
                    source_suffix = " (SDMC)";
                    break;
                default:
                    break;
            }

            std::string version_str;
            u32 numeric_ver = 0;
            const auto* slot_provider = content_union->GetSlotProvider(slot);
            version_str = GetUpdateVersionStringFromSlot(slot_provider, update_tid);

            if (slot_provider != nullptr) {
                const auto slot_ver = slot_provider->GetEntryVersion(update_tid);
                if (slot_ver.has_value()) {
                    numeric_ver = *slot_ver;
                    if (version_str.empty() && numeric_ver != 0) {
                        version_str = FormatTitleVersion(numeric_ver);
                    }
                }
            }

            std::string patch_name = "Update" + source_suffix;

            bool update_disabled =
                std::find(disabled.cbegin(), disabled.cend(), patch_name) != disabled.cend();

            Patch update_patch = {.enabled = !update_disabled,
                                  .name = patch_name,
                                  .version = version_str,
                                  .type = PatchType::Update,
                                  .program_id = title_id,
                                  .title_id = update_tid,
                                  .source = source_type,
                                  .numeric_version = numeric_ver};

            out.push_back(update_patch);
        }
    } else {
        PatchManager update{update_tid, fs_controller, content_provider};
        const auto metadata = update.GetControlMetadata();
        const auto& nacp = metadata.first;

        bool update_disabled =
            std::find(disabled.cbegin(), disabled.cend(), "Update") != disabled.cend();
        Patch update_patch = {.enabled = !update_disabled,
                              .name = "Update",
                              .version = "",
                              .type = PatchType::Update,
                              .program_id = title_id,
                              .title_id = title_id,
                              .source = PatchSource::Unknown,
                              .numeric_version = 0};

        if (nacp != nullptr) {
            update_patch.version = nacp->GetVersionString();
            out.push_back(update_patch);
        } else {
            if (content_provider.HasEntry(update_tid, ContentRecordType::Program)) {
                const auto meta_ver = content_provider.GetEntryVersion(update_tid);
                if (meta_ver.value_or(0) == 0) {
                    out.push_back(update_patch);
                } else {
                    update_patch.version = FormatTitleVersion(*meta_ver);
                    update_patch.numeric_version = *meta_ver;
                    out.push_back(update_patch);
                }
            } else if (update_raw != nullptr) {
                update_patch.version = "PACKED";
                update_patch.source = PatchSource::Packed;
                out.push_back(update_patch);
            }
        }
    }

    // General Mods (LayeredFS and IPS)
    const auto mod_dir = fs_controller.GetModificationLoadRoot(title_id);
    if (mod_dir != nullptr) {
        for (auto const& f : mod_dir->GetFiles())
            if (auto const name = f->GetName(); name.starts_with("cheat_")) {
                auto const mod_disabled = std::find(disabled.begin(), disabled.end(), name) != disabled.end();
                out.push_back({
                    .enabled = !mod_disabled,
                    .name = name,
                    .version = "Cheats",
                    .type = PatchType::Mod,
                    .program_id = title_id,
                    .title_id = title_id,
                    .source = PatchSource::Unknown,
                    .location = f->GetFullPath(),
                });
            }

        for (const auto& mod : mod_dir->GetSubdirectories()) {
            // Not listed: Fritchii's built-in Dynamic FPS (listed below) is used instead.
            if (IsBuiltinDynamicFpsMod(title_id, mod)) {
                continue;
            }
            auto types = ModTypes(mod);
            if (types.empty())
                continue;

            const auto mod_disabled = std::find(disabled.begin(), disabled.end(), mod->GetName()) != disabled.end();
            out.push_back({.enabled = !mod_disabled,
                           .name = mod->GetName(),
                           .version = types,
                           .type = PatchType::Mod,
                           .program_id = title_id,
                           .title_id = title_id,
                           .source = PatchSource::Unknown,
                           .location = mod->GetFullPath()});
        }
    }

    // SDMC mod directory (RomFS LayeredFS)
    const auto sdmc_mod_dir = fs_controller.GetSDMCModificationLoadRoot(title_id);
    if (sdmc_mod_dir != nullptr) {
        auto types = SdmcModTypes(sdmc_mod_dir);
        if (!types.empty()) {
            const auto mod_disabled =
                std::find(disabled.begin(), disabled.end(), "SDMC") != disabled.end();
            out.push_back({.enabled = !mod_disabled,
                           .name = "SDMC",
                           .version = types,
                           .type = PatchType::Mod,
                           .program_id = title_id,
                           .title_id = title_id,
                           .source = PatchSource::Unknown});
        }
    }

    for (const auto& patch : BUILTIN_PATCHES) {
        if (patch.title_id != title_id || patch.render_height != 0 ||
            (patch.enable_env != nullptr &&
             std::string_view{std::getenv(patch.enable_env) ? std::getenv(patch.enable_env) : ""} !=
                 "1") ||
            (patch.setting_enabled != nullptr && !patch.setting_enabled())) {
            continue;
        }
        const auto patch_disabled =
            std::find(disabled.begin(), disabled.end(), patch.name) != disabled.end();
        out.push_back({.enabled = !patch_disabled,
                       .name = std::string{patch.name},
                       .version = "IPS",
                       .type = PatchType::Mod,
                       .program_id = title_id,
                       .title_id = title_id,
                       .source = PatchSource::Unknown});
    }

    // DLC
    std::vector<ContentProviderEntry> dlc_match;
    bool has_external_dlc = false;
    bool has_nand_dlc = false;
    bool has_sdmc_dlc = false;
    bool has_other_dlc = false;
    const auto dlc_entries_with_origin =
        content_union->ListEntriesFilterOrigin(std::nullopt, TitleType::AOC, ContentRecordType::Data);

    dlc_match.reserve(dlc_entries_with_origin.size());
    for (const auto& [slot, entry] : dlc_entries_with_origin) {
        const auto base_tid = GetBaseTitleID(entry.title_id);
        const bool matches_base = base_tid == title_id;
        if (!matches_base) {
            LOG_DEBUG(Loader, "DLC {:016X} base {:016X} doesn't match title {:016X}",
                      entry.title_id, base_tid, title_id);
            continue;
        }

        const auto* slot_provider = content_union->GetSlotProvider(slot);
        if (slot_provider == nullptr) {
            continue;
        }

        auto nca = slot_provider->GetEntry(entry);
        if (!nca) {
            LOG_DEBUG(Loader, "Failed to get NCA for DLC {:016X}", entry.title_id);
            continue;
        }

        const auto status = nca->GetStatus();
        if (status != Loader::ResultStatus::Success) {
            LOG_DEBUG(Loader, "DLC {:016X} NCA has status {}", entry.title_id,
                      static_cast<int>(status));
            continue;
        }

        switch (slot) {
            case ContentProviderUnionSlot::External:
            case ContentProviderUnionSlot::FrontendManual:
                has_external_dlc = true;
                break;
            case ContentProviderUnionSlot::UserNAND:
            case ContentProviderUnionSlot::SysNAND:
                has_nand_dlc = true;
                break;
            case ContentProviderUnionSlot::SDMC:
                has_sdmc_dlc = true;
                break;
            default:
                has_other_dlc = true;
                break;
        }
        dlc_match.push_back(entry);
    }

    if (!dlc_match.empty()) {
        // Ensure sorted so DLC IDs show in order.
        std::sort(dlc_match.begin(), dlc_match.end());

        std::string list;
        for (size_t i = 0; i < dlc_match.size() - 1; ++i)
            list += fmt::format("{}, ", dlc_match[i].title_id & 0x7FF);

        list += fmt::format("{}", dlc_match.back().title_id & 0x7FF);

        const auto dlc_disabled =
            std::find(disabled.begin(), disabled.end(), "DLC") != disabled.end();
        PatchSource dlc_source = PatchSource::Unknown;
        if (has_external_dlc && !has_nand_dlc && !has_sdmc_dlc && !has_other_dlc) {
            dlc_source = PatchSource::External;
        } else if (has_nand_dlc && !has_external_dlc && !has_sdmc_dlc && !has_other_dlc) {
            dlc_source = PatchSource::NAND;
        } else if (has_sdmc_dlc && !has_external_dlc && !has_nand_dlc && !has_other_dlc) {
            dlc_source = PatchSource::SDMC;
        }

        out.push_back({.enabled = !dlc_disabled,
                       .name = "DLC",
                       .version = std::move(list),
                       .type = PatchType::DLC,
                       .program_id = title_id,
                       .title_id = dlc_match.back().title_id,
                       .source = dlc_source});
    }

    return out;
}

std::optional<u32> PatchManager::GetGameVersion() const {
    const auto update_tid = GetUpdateTitleID(title_id);
    if (content_provider.HasEntry(update_tid, ContentRecordType::Program)) {
        const auto version = content_provider.GetEntryVersion(update_tid);
        return version || !parent_title_id ? version : content_provider.GetEntryVersion(GetUpdateTitleID(*parent_title_id));
    }

    const auto version = content_provider.GetEntryVersion(title_id);
    return version || !parent_title_id ? version : content_provider.GetEntryVersion(*parent_title_id);
}

PatchManager::Metadata PatchManager::GetControlMetadata() const {
    const auto base_control_nca = content_provider.GetEntry(title_id, ContentRecordType::Control);
    if (base_control_nca == nullptr) {
        if (parent_title_id) {
            const auto control_id = content_provider.HasEntry(*parent_title_id, ContentRecordType::Control) ? *parent_title_id : GetUpdateTitleID(*parent_title_id);
            const PatchManager parent{control_id, fs_controller, content_provider};
            return parent.GetControlMetadata();
        }
        return {};
    }

    return ParseControlNCA(*base_control_nca);
}

PatchManager::Metadata PatchManager::ParseControlNCA(const NCA& nca) const {
    const auto base_romfs = nca.GetRomFS();
    if (base_romfs == nullptr) {
        return {};
    }

    return ParseControlRomFS(PatchRomFS(&nca, base_romfs, ContentRecordType::Control));
}

PatchManager::Metadata PatchManager::ParseControlRomFS(const VirtualFile& romfs) {
    if (romfs == nullptr) {
        return {};
    }

    const auto extracted = ExtractRomFS(romfs);
    if (extracted == nullptr) {
        return {};
    }

    auto nacp_file = extracted->GetFile("control.nacp");
    if (nacp_file == nullptr) {
        nacp_file = extracted->GetFile("Control.nacp");
    }

    auto nacp = nacp_file == nullptr ? nullptr : std::make_unique<NACP>(nacp_file);

    // Get language code from settings
    const auto language_code = Service::Set::GetLanguageCodeFromIndex(
        static_cast<u32>(Settings::values.language_index.GetValue()));

    // Convert to application language and get priority list
    const auto application_language =
        Service::NS::ConvertToApplicationLanguage(language_code)
            .value_or(Service::NS::ApplicationLanguage::AmericanEnglish);
    const auto language_priority_list =
        Service::NS::GetApplicationLanguagePriorityList(application_language);

    // Convert to language names
    auto priority_language_names = FileSys::LANGUAGE_NAMES; // Copy
    if (language_priority_list) {
        for (size_t i = 0; i < priority_language_names.size(); ++i) {
            // Relies on FileSys::LANGUAGE_NAMES being in the same order as
            // Service::NS::ApplicationLanguage
            const auto language_index = static_cast<u8>(language_priority_list->at(i));

            if (language_index < FileSys::LANGUAGE_NAMES.size()) {
                priority_language_names[i] = FileSys::LANGUAGE_NAMES[language_index];
            } else {
                // Not a catastrophe, unlikely to happen
                LOG_WARNING(Loader, "Invalid language index {}", language_index);
            }
        }
    }

    // Get first matching icon
    VirtualFile icon_file;
    for (const auto& language : priority_language_names) {
        icon_file = extracted->GetFile(std::string("icon_").append(language).append(".dat"));
        if (icon_file != nullptr) {
            break;
        }
    }

    return {std::move(nacp), icon_file};
}

[[nodiscard]] PatchManager::Metadata PatchManager::GetMetadataFromBaseOrUpdate(Core::System& system, u64 application_id) noexcept {
    const FileSys::PatchManager pm{application_id, system.GetFileSystemController(), system.GetContentProvider()};
    auto metadata = pm.GetControlMetadata();
    if (metadata.first != nullptr)
        return metadata;
    const FileSys::PatchManager pm_update{FileSys::GetUpdateTitleID(application_id), system.GetFileSystemController(), system.GetContentProvider()};
    return pm_update.GetControlMetadata();
}

std::optional<size_t> OotHighRefreshScaleOffset(std::span<const u8> code) {
    const auto parts = ReadOotHighRefreshCode(code);
    if (!parts || GetOotHighRefreshLayout(*parts) != OotHighRefreshLayout::On) {
        return std::nullopt;
    }
    return OotHighRefreshCode::ScaleOffset;
}

std::optional<OotHighRefreshCode> ReadOotHighRefreshCode(std::span<const u8> code) {
    using Code = OotHighRefreshCode;
    static_assert(Code::WaterOffset < Code::WaterMulOffset && Code::WaterMulOffset < Code::CallOffset &&
                  Code::CallOffset < Code::DarkOffset && Code::DarkOffset < Code::CaveOffset &&
                  Code::CaveOffset + Code::CaveSize <= Code::RippleOffset);
    if (code.size() < Code::RippleOffset + Code::RippleSize) {
        return std::nullopt;
    }
    Code parts;
    std::ranges::copy(code.subspan(Code::CallOffset, parts.call.size()), parts.call.begin());
    std::ranges::copy(code.subspan(Code::WaterOffset, parts.water.size()), parts.water.begin());
    std::ranges::copy(code.subspan(Code::WaterMulOffset, parts.water_mul.size()),
                      parts.water_mul.begin());
    std::ranges::copy(code.subspan(Code::DarkOffset, parts.dark.size()), parts.dark.begin());
    std::ranges::copy(code.subspan(Code::CaveOffset, parts.cave.size()), parts.cave.begin());
    std::ranges::copy(code.subspan(Code::RippleOffset, parts.ripples.size()), parts.ripples.begin());
    return parts;
}

OotHighRefreshLayout GetOotHighRefreshLayout(const OotHighRefreshCode& code) {
    // Dark's routine is there in both modes (switching off calls it again).
    if (code.dark != DYNAMIC_FPS_ROUTINE) {
        return OotHighRefreshLayout::Unknown;
    }
    // Before v5.8 (and with the ripple built-in off) Dark's mode has the water renderer's own
    // fcvtzs and mul and leaves our cave and the ripple routine's place free; still switchable.
    if (code.call == DYNAMIC_FPS_CALL && code.water == WATER_ORIGINAL &&
        code.water_mul == WATER_MUL_ORIGINAL && code.cave == FREE_CAVE &&
        code.ripples == FREE_WATER_ROUTINE) {
        return OotHighRefreshLayout::Off;
    }
    // The cave but the speed scale word, which the present pacing sets (Turbo, Slow).
    const auto cave = std::span<const u8>{code.cave};
    const auto ours = std::span<const u8>{HIGH_REFRESH_ROUTINE};
    constexpr size_t after_scale = HIGH_REFRESH_SCALE + 4;
    const bool our_cave =
        code.water == WATER_HOOK && code.water_mul == WATER_MUL_NOP &&
        code.ripples == WATER_ROUTINE &&
        std::ranges::equal(cave.first(HIGH_REFRESH_SCALE), ours.first(HIGH_REFRESH_SCALE)) &&
        std::ranges::equal(cave.subspan(after_scale), ours.subspan(after_scale));
    if (our_cave && code.call == HIGH_REFRESH_CALL) {
        return OotHighRefreshLayout::On;
    }
    if (our_cave && code.call == DYNAMIC_FPS_CALL) {
        return OotHighRefreshLayout::Off;
    }
    return OotHighRefreshLayout::Unknown;
}

std::optional<OotHighRefreshCode> SwitchOotHighRefreshCode(const OotHighRefreshCode& code,
                                                           bool on, bool ripples) {
    const OotHighRefreshLayout layout = GetOotHighRefreshLayout(code);
    if (layout == OotHighRefreshLayout::Unknown) {
        return std::nullopt;
    }
    if ((layout == OotHighRefreshLayout::On) == on) {
        return code;
    }
    OotHighRefreshCode switched = code;
    switched.call = on ? HIGH_REFRESH_CALL : DYNAMIC_FPS_CALL;
    // Both modes keep the water hook and nop, the cave and the ripple routine (v5.8): only the
    // call and the speed scale word change, so no code a thread may be running changes under it.
    // Unless the player turned the ripple built-in off: then Dark's mode has none of them, as
    // before (the fcvtzs and the mul both back, both routines' places free).
    const bool keep = on || ripples;
    switched.water = keep ? WATER_HOOK : WATER_ORIGINAL;
    switched.water_mul = keep ? WATER_MUL_NOP : WATER_MUL_ORIGINAL;
    switched.cave = keep ? HIGH_REFRESH_ROUTINE : FREE_CAVE;
    switched.ripples = keep ? WATER_ROUTINE : FREE_WATER_ROUTINE;
    return switched;
}

bool OotHighRefreshStateInBss(u64 bss_start, u64 bss_end) {
    return HIGH_REFRESH_STATE_OFFSET >= bss_start && HIGH_REFRESH_STATE_OFFSET <= bss_end &&
           HIGH_REFRESH_STATE_SIZE <= bss_end - HIGH_REFRESH_STATE_OFFSET;
}

namespace {

// One attempt, with the application stalled: no guest thread of it runs on any core, and each
// one's saved context is its real state.
OotHighRefreshSwitch SwitchOotHighRefreshStalled(Kernel::KProcess& process, u64 code_address,
                                                 bool on) {
    using Code = OotHighRefreshCode;
    const u64 cave_address = code_address + Code::CaveOffset;
    const u64 ripple_address = code_address + Code::RippleOffset;
    // No thread may stand inside our cave or the ripple routine, where the bytes under it change
    // (it could have been stopped at a block boundary in it), nor between the water hook and its
    // nop (main+0x475FC..0x4760C: the routine's return lands at 0x475FC, a block start): it ran
    // one form of the hook and would run the other form of the mul (passes * w9, where w9 now
    // holds a slot address, would run the ripples for ever). The call is a single word that
    // leads into complete code either way, and Dark's routine never changes.
    {
        const auto inside = [](u64 pc, u64 start, u64 size) {
            return pc >= start && pc < start + size;
        };
        const u64 between_water_words = code_address + Code::WaterOffset + 4;
        Kernel::KScopedLightLock lk{process.GetListLock()};
        for (auto& thread : process.GetThreadList()) {
            const u64 pc = thread.GetContext().pc;
            if (inside(pc, cave_address, Code::CaveSize) ||
                inside(pc, ripple_address, Code::RippleSize) ||
                inside(pc, between_water_words, Code::WaterMulOffset + 4 - (Code::WaterOffset + 4))) {
                return OotHighRefreshSwitch::Busy;
            }
        }
    }

    // The present pacing writes the speed scale word under this lock too (it runs on a service
    // thread, which the stall does not stop).
    std::scoped_lock lk{process.GetOotHighRefreshLock()};
    auto& memory = process.GetMemory();
    Code current;
    if (!memory.ReadBlock(code_address + Code::CallOffset, current.call.data(), current.call.size()) ||
        !memory.ReadBlock(code_address + Code::WaterOffset, current.water.data(),
                          current.water.size()) ||
        !memory.ReadBlock(code_address + Code::WaterMulOffset, current.water_mul.data(),
                          current.water_mul.size()) ||
        !memory.ReadBlock(code_address + Code::DarkOffset, current.dark.data(),
                          current.dark.size()) ||
        !memory.ReadBlock(cave_address, current.cave.data(), current.cave.size()) ||
        !memory.ReadBlock(ripple_address, current.ripples.data(), current.ripples.size())) {
        return OotHighRefreshSwitch::UnknownCode;
    }
    const auto disabled = Settings::values.disabled_addons.find(0x0400486024DEC000ULL);
    const bool ripples = disabled == Settings::values.disabled_addons.end() ||
                         std::ranges::find(disabled->second, std::string{WATER_RIPPLE_NAME}) ==
                             disabled->second.end();
    const auto target = SwitchOotHighRefreshCode(current, on, ripples);
    if (!target) {
        return OotHighRefreshSwitch::UnknownCode;
    }
    if (!on) {
        process.SetOotHighRefreshScaleAddress(0);
        Settings::SetOotHighRefreshPatched(false);
    }
    // On: our code first, then the words leading into it; off: the words first, then the code
    // cleared (the guest is stalled either way; this order never leaves a word pointing at a
    // cave that isn't complete).
    const auto write_block = [&](u64 address, std::span<const u8> bytes,
                                 std::span<const u8> was) {
        if (!std::ranges::equal(bytes, was)) {
            memory.WriteBlock(address, bytes.data(), bytes.size());
            Core::InvalidateInstructionCacheRange(&process, address, bytes.size());
        }
    };
    const auto write_cave = [&] {
        write_block(cave_address, target->cave, current.cave);
        write_block(ripple_address, target->ripples, current.ripples);
    };
    const auto write_word = [&](size_t offset, const std::array<u8, 4>& word,
                                const std::array<u8, 4>& was) {
        if (word != was) {
            memory.WriteBlock(code_address + offset, word.data(), word.size());
            Core::InvalidateInstructionCacheRange(&process, code_address + offset, word.size());
        }
    };
    if (on) {
        write_cave();
    }
    write_word(Code::WaterMulOffset, target->water_mul, current.water_mul);
    write_word(Code::WaterOffset, target->water, current.water);
    write_word(Code::CallOffset, target->call, current.call);
    if (!on) {
        write_cave();
    }
    if (on) {
        process.SetOotHighRefreshScaleAddress(code_address + Code::ScaleOffset);
        Settings::SetOotHighRefreshPatched(true);
    }
    return OotHighRefreshSwitch::Switched;
}

} // Anonymous namespace

std::string OotHighRefreshBlockedBy() {
    std::scoped_lock lock{g_high_refresh_blocked_mutex};
    return g_high_refresh_blocked_by;
}

OotHighRefreshSwitch SwitchOotHighRefreshLive(Core::System& system, bool on) {
    constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;
    Kernel::KProcess* const process = system.ApplicationProcess();
    if (!system.IsPoweredOn() || system.IsShuttingDown() || process == nullptr ||
        process->GetProgramId() != OcarinaOfTime ||
        process->GetOotMainCodeAddress() == 0) {
        return OotHighRefreshSwitch::NotRunning;
    }
    if (on) {
        const auto it = Settings::values.disabled_addons.find(OcarinaOfTime);
        if (it != Settings::values.disabled_addons.end() &&
            std::ranges::find(it->second, HIGH_REFRESH_NAME) != it->second.end()) {
            return OotHighRefreshSwitch::DisabledInAddOns;
        }
    }
    // A thread stopped inside the cave is all but impossible (a few dozen instructions once a
    // frame); try a few times anyway, letting the game run a moment in between.
    OotHighRefreshSwitch result = OotHighRefreshSwitch::Busy;
    for (int attempt = 0; attempt < 8 && result == OotHighRefreshSwitch::Busy; ++attempt) {
        if (attempt != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{2});
        }
        auto stall = system.StallApplication();
        result = SwitchOotHighRefreshStalled(*process, process->GetOotMainCodeAddress(), on);
        system.UnstallApplication();
    }
    LOG_INFO(Loader, "Ocarina of Time: high refresh rate switched live {}: {}", on ? "on" : "off",
             result == OotHighRefreshSwitch::Switched ? "done"
             : result == OotHighRefreshSwitch::Busy   ? "a thread stayed in the cave"
                                                      : "its code holds neither mode");
    return result;
}

} // namespace FileSys
