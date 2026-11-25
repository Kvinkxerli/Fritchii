// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <optional>
#include <cstdlib>
#include <string_view>

#include "common/settings_enums.h"

namespace VideoCommon {

/// OoT's picture this session, for the status bar: the upscaler that really runs and its sizes.
/// The rasterizer publishes it once when it picks OoT's upscaler (the "Ocarina temporal: output"
/// log line); empty before that and outside OoT. Backend: 1 the game's own, 2 FSR 3.1, 3 FSR 4,
/// 4 DLSS. A 0 output means no upscaler (the resolution setting applies).
struct OcarinaPicture {
    std::uint32_t backend;
    std::uint32_t input_height;
    bool input_up_to; ///< The game's dynamic resolution (Max Resolution off): at most input_height
    std::uint32_t output_height;
};

inline std::atomic<std::uint64_t> g_ocarina_picture{0};

inline void PublishOcarinaPicture(std::uint32_t backend, std::uint32_t input_height, bool up_to,
                                  std::uint32_t output_height) {
    g_ocarina_picture.store((std::uint64_t{1} << 63) | (std::uint64_t{backend & 0xff} << 48) |
                                (std::uint64_t{up_to} << 47) |
                                (std::uint64_t{input_height & 0xffff} << 16) |
                                std::uint64_t{output_height & 0xffff},
                            std::memory_order_relaxed);
}

inline void ClearOcarinaPicture() {
    g_ocarina_picture.store(0, std::memory_order_relaxed);
}

[[nodiscard]] inline std::optional<OcarinaPicture> ReadOcarinaPicture() {
    const std::uint64_t value = g_ocarina_picture.load(std::memory_order_relaxed);
    if ((value >> 63) == 0) {
        return std::nullopt;
    }
    return OcarinaPicture{static_cast<std::uint32_t>((value >> 48) & 0xff),
                          static_cast<std::uint32_t>((value >> 16) & 0xffff),
                          ((value >> 47) & 1) != 0, static_cast<std::uint32_t>(value & 0xffff)};
}

enum class OcarinaUpscaleBackend { Native, Fsr, Dlss };

constexpr const char* OcarinaBackendName(OcarinaUpscaleBackend backend) {
    switch (backend) {
    case OcarinaUpscaleBackend::Native:
        return "the game's upscale";
    case OcarinaUpscaleBackend::Fsr:
        return "FSR";
    case OcarinaUpscaleBackend::Dlss:
        return "NVIDIA DLSS";
    }
    return "";
}

namespace OcarinaSwitch {
/// A boot switch read once: unset or empty gives `fallback`.
inline int Read(const char* name, int fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? std::atoi(value) : fallback;
}
} // namespace OcarinaSwitch

/// FRITCHII_DLSS_SANITIZE (boot, 1 = on by default): the upscaler's colour input has NaN and
/// infinite texels set to 0 and is clamped to [0, 1] (the game's colour is already tonemapped).
/// DLSS spreads such a texel into a black block: the small black squares around far birds and
/// particles in Hyrule Field. 0 = the exact copy as before.
inline bool OcarinaDlssSanitize() {
    static const bool on = OcarinaSwitch::Read("FRITCHII_DLSS_SANITIZE", 1) != 0;
    return on;
}

/// FRITCHII_FSR4_JITTER (boot, 0 = off): AMD FSR 4 gets the camera jitter and texture LOD bias
/// FSR 3.1 uses. Off, FSR 4 takes the scene as the game draws it, without jitter or bias, as in
/// CheezyEmu, whose FSR 4 AMD testers report as working well (RX 9070 XT). FSR 3.1 and DLSS
/// are unchanged.
inline bool OcarinaFsr4Jitter() {
    static const bool on = OcarinaSwitch::Read("FRITCHII_FSR4_JITTER", 0) != 0;
    return on;
}

/// FRITCHII_FSR4_LINEAR (boot, 0 = off): AMD FSR 4 is told the colour is linear, as before.
/// Off, it is told the colour is perceptual (gamma encoded), which OoT's tonemapped colour is
/// (CheezyEmu's FSR 4 setup).
inline bool OcarinaFsr4LinearColor() {
    static const bool on = OcarinaSwitch::Read("FRITCHII_FSR4_LINEAR", 0) != 0;
    return on;
}

/// FRITCHII_FSR_OUTPUT (boot, 0 = as before): the size OoT's FSR (3.1 and 4) draws at when the
/// game's own pass would make its 2560x1440 picture. 0: 3840x2160, then composited with the
/// game's interface at 4K and scaled to the window (on a 1440p screen that 4K picture is scaled
/// back down). 1: the game's own 2560x1440 picture (FSR as anti-aliasing at 1x, no 4K
/// composite), 4/9 of the pixels FSR outputs at 4K: FSR 4 on Radeon RX 7000 cards runs its
/// network on the shader cores and spends several ms a frame there. 2: 1 when the primary
/// display is at most 1440 pixels tall, else 0. DLSS is unchanged.
inline int OcarinaFsrOutputMode() {
    static const int mode = OcarinaSwitch::Read("FRITCHII_FSR_OUTPUT", 0);
    return mode;
}

/// Whether FSR draws the game's own 2560x1440 picture instead of a 4K one (FRITCHII_FSR_OUTPUT);
/// `display_height` 0 = unknown.
constexpr bool OcarinaFsrOutputsGameSize(int mode, OcarinaUpscaleBackend backend,
                                         unsigned display_height) {
    if (backend != OcarinaUpscaleBackend::Fsr) {
        return false;
    }
    return mode == 1 || (mode == 2 && display_height != 0 && display_height <= 1440);
}

/// oot_upscale_output: the output heights the player may pick for OoT's DLSS or FSR (docked,
/// 16:9). 0 = Auto, as before: 4K from Max Resolution's 1440p pass (the game's 1440p picture
/// with FRITCHII_FSR_OUTPUT), the game's 1440p from its 3/4 pass without Max Resolution.
inline constexpr std::array<unsigned, 6> OcarinaUpscaleOutputs{1080, 1440, 1800, 2160, 2880, 4320};

constexpr bool OcarinaUpscaleOutputKnown(unsigned height) {
    for (const unsigned known : OcarinaUpscaleOutputs) {
        if (known == height) {
            return true;
        }
    }
    return false;
}

/// The 16:9 width of an output height (1800 -> 3200, 4320 -> 7680).
constexpr unsigned OcarinaOutputWidth(unsigned height) {
    return height * 16 / 9;
}

/// Video memory an output above the game's 1440p picture takes, estimated: the upscaler's
/// output-sized images (FSR 3.1: two RGBA16F histories, the RGBA16F output and a lock mask, 25
/// bytes a pixel; DLSS and FSR 4 keep similar ones) and the RGBA8 composite with the game's
/// interface, rounded up to 40 bytes a pixel. 4K ~316 MiB, 2880p ~563 MiB, 8K ~1.24 GiB.
constexpr unsigned long long OcarinaOutputBytes(unsigned height) {
    return 40ULL * OcarinaOutputWidth(height) * height;
}

/// Whether a card whose main video memory heap has `vram` bytes (0 = unknown) takes an output:
/// up to 4K always (as before); above it when the estimate is at most an eighth of the card
/// (2880p from 6 GB, 8K from 12 GB), so 8 GB cards keep their room for the game.
constexpr bool OcarinaUpscaleOutputFits(unsigned height, unsigned long long vram) {
    return height <= 2160 || vram == 0 || OcarinaOutputBytes(height) * 8 <= vram;
}

/// oot_upscale_output as the game starts with it, and why (the boot log line).
struct OcarinaOutputSetting {
    unsigned height{};   ///< 0 = Auto, else one of OcarinaUpscaleOutputs
    const char* reason{};
};

constexpr OcarinaOutputSetting ResolveOcarinaUpscaleOutput(unsigned requested,
                                                           unsigned long long vram,
                                                           bool small_card) {
    if (requested == 0) {
        if (small_card || (vram != 0 && vram < (5ULL << 30))) {
            return {1080, "Auto capped to 1080p under 5 GiB"};
        }
        return {0, "Auto, as before"};
    }
    if (!OcarinaUpscaleOutputKnown(requested)) {
        return {0, "not one of the offered heights, Auto instead"};
    }
    if (requested > 2160 && small_card) {
        return {2160, "the small card profile is on, 4K instead"};
    }
    if (!OcarinaUpscaleOutputFits(requested, vram)) {
        return {2160, "more than an eighth of this card's video memory, 4K instead"};
    }
    return {requested, "picked in the picture menu"};
}

/// What one replaced pass makes. `native_aa`: the output keeps the scene's size (DLAA, or FSR
/// at 1x). `big_height`: an output above the game's 1440p picture (4K before), composited with
/// the game's interface when presented; 0 = none, the upscaler writes into the game's target
/// at the target's size.
struct OcarinaPassOutput {
    bool native_aa{};
    unsigned big_height{};
    /// An output between a lowered scene (oot_render_height) and the game's 1440p picture
    /// (720p -> 1080p): the upscaler makes that size and the output pass stretches it into the
    /// game's target. 0 = none.
    unsigned small_height{};
    constexpr bool operator==(const OcarinaPassOutput&) const = default;
};

/// The output a scene below 720p gets (oot_render_height 540 and 360): the largest offered height
/// up to `requested` and within 3x the scene. Auto (0) asks for 1080p here, not 4K: 4K is 4x from
/// 540p and 6x from 360p, beyond what DLSS and FSR take, and 1080p is an exact DLSS mode from both
/// (Performance 2x, Ultra Performance 3x). 0 = no offered height fits.
constexpr unsigned OcarinaLowSceneOutput(unsigned requested, unsigned scene_height) {
    const unsigned wanted = requested == 0 ? 1080U : requested;
    unsigned best = 0;
    for (const unsigned height : OcarinaUpscaleOutputs) {
        if (height <= wanted && height <= 3 * scene_height) {
            best = height;
        }
    }
    return best;
}

/// `output`: oot_upscale_output resolved (0 = Auto). `pass_1440p`: Max Resolution's 1440p pass,
/// whose target is the game's 1440p picture. `scene_height`: the height the game drew (1440, or
/// what the DLSS render scale made it). `dlss_native`: the DLSS quality Native (DLAA), which
/// keeps the scene's size whatever the output. `fsr_game_size`: FRITCHII_FSR_OUTPUT (Auto
/// only). DLSS and FSR only enlarge, by up to 3x a side (FSR 3.1's range, DLSS Ultra
/// Performance), so an output at or below the scene is native AA at the scene's size (the
/// window's scaling takes it down), and one more than 3x the scene drops to the largest offered
/// height within 3x. With a lowered scene (scene_height below 1440: oot_render_height) an output
/// below 1440p is made at its own size (small_height) or, at most the scene, as native AA; with
/// the game's 1440p scene the outputs up to 1440p stay the game's target, as before. A scene below
/// 720p takes OcarinaLowSceneOutput: even the game's 1440p target is over 3x from 360p.
constexpr OcarinaPassOutput PlanOcarinaPassOutput(unsigned output, bool pass_1440p,
                                                  unsigned scene_height, bool dlss_native,
                                                  bool fsr_game_size) {
    if (dlss_native) {
        return {true, 0};
    }
    if (pass_1440p && scene_height < 720) {
        const unsigned low = OcarinaLowSceneOutput(output, scene_height);
        if (low == 0) {
            return {true, 0};
        }
        return {false, 0, low < 1440 ? low : 0U};
    }
    if (output == 0) {
        return {false, pass_1440p && !fsr_game_size ? 2160U : 0U};
    }
    if (!pass_1440p) {
        // The 3/4 pass: its target is 4/3 of the scene (1440p at most).
        return {output <= 1080, 0};
    }
    if (output < 1440 && scene_height < 1440) {
        if (output <= scene_height) {
            return {true, 0};
        }
        return {false, 0, output};
    }
    if (output <= 1440) {
        return {false, 0};
    }
    unsigned big = 0;
    for (const unsigned height : OcarinaUpscaleOutputs) {
        if (height > 1440 && height <= output && height <= 3 * scene_height) {
            big = height;
        }
    }
    return {false, big};
}

/// FRITCHII_OOT_CAMERA_GUEST (boot, 0 = off): the camera tracker reads vertex uniform slot 3 as
/// the game bound it, not as the draw's shader sees it. The shader's view leaves out slots it does
/// not use and stops at the last byte it reads, so draws that read less than the camera block gave
/// no camera and the upscaler stepped aside ("no scene camera this frame", e.g. when the camera
/// turns in the Water Temple). On, only the 0x1c0 bytes up to the end of the previous matrix
/// are needed.
inline bool OcarinaCameraFromGuest() {
    static const bool on = OcarinaSwitch::Read("FRITCHII_OOT_CAMERA_GUEST", 0) != 0;
    return on;
}
/// The camera bytes the tracker needs: through the previous view-projection matrix (0x180 + 64)
/// with FRITCHII_OOT_CAMERA_GUEST, the whole block as before without it.
inline constexpr unsigned OcarinaCameraBlockBytes = 0x230;
inline constexpr unsigned OcarinaCameraNeededBytes = 0x1c0;
inline unsigned OcarinaCameraReadBytes() {
    return OcarinaCameraFromGuest() ? OcarinaCameraNeededBytes : OcarinaCameraBlockBytes;
}

/// FRITCHII_JITTER_EDGE (boot, 1 = on by default): on AMD's drivers the jitter offsets are
/// Halton (2, 3) in [0, 1) instead of [-0.5, 0.5). AMD clips each viewport to a whole pixel, so a
/// negative offset dropped the last column and row of every jittered draw on half the frames
/// (a ripple along the right and bottom edges with FSR). Other drivers are unchanged.
inline bool OcarinaJitterEdgeFix() {
    static const bool on = OcarinaSwitch::Read("FRITCHII_JITTER_EDGE", 1) != 0;
    return on;
}

/// The camera jitter for a phase (from 1) in render pixels: Halton (2, 3), centred on 0, or in
/// [0, 1) for drivers that clip the viewport's fractional right and bottom edges.
inline std::array<float, 2> OcarinaJitter(unsigned index, bool truncated_viewport_edges) {
    const auto halton = [](unsigned value, unsigned base) {
        float fraction = 1.0f;
        float result = 0.0f;
        while (value > 0) {
            fraction /= static_cast<float>(base);
            result += fraction * static_cast<float>(value % base);
            value /= base;
        }
        return result;
    };
    const float center = truncated_viewport_edges ? 0.0f : 0.5f;
    return {halton(index, 2) - center, halton(index, 3) - center};
}

/// FRITCHII_OOT_ACTOR_PARTS (boot, 0 = off): the skinned motion replay also takes the other draws
/// of the actor's vertex program that use Link's bone palette and origin, not only his recognized
/// 9702-index draw. Parts left out keep the camera's motion: with a still camera that is none, so
/// DLSS and FSR blend a part that moves (the hat swaying) with where it was. A wrong part costs
/// nothing but its replay: the scene depth check drops what the game did not draw there.
inline bool OcarinaActorParts() {
    static const bool on = OcarinaSwitch::Read("FRITCHII_OOT_ACTOR_PARTS", 0) != 0;
    return on;
}

/// FRITCHII_OOT_ACTOR_TOLERANCE (boot, 16 by default): how far, in steps of the depth format, the
/// replay's depth may be from the scene's for a pixel to take the skinned motion.
inline std::uint32_t OcarinaActorDepthTolerance() {
    static const std::uint32_t tolerance = static_cast<std::uint32_t>(
        std::max(OcarinaSwitch::Read("FRITCHII_OOT_ACTOR_TOLERANCE", 16), 0));
    return tolerance;
}

/// FRITCHII_OOT_ACTOR_TRACE (boot, 0 = off): every 600th frame the log lists the indexed draws
/// that could carry a character (the actor's vertex program, or its 48-byte vertices).
inline bool OcarinaActorTrace() {
    static const bool on = OcarinaSwitch::Read("FRITCHII_OOT_ACTOR_TRACE", 0) != 0;
    return on;
}

/// FRITCHII_OOT_MOTION_VIEW (boot, 0 = off): a test view. The game's target gets the motion
/// vectors instead of the upscaled scene: black where nothing moves, blue with red and green for
/// the size of the motion elsewhere. With a still camera, only the pixels the skinned replay
/// reached light up. Shown with outputs up to 1440p (above, the composite shows the scene).
inline bool OcarinaMotionView() {
    static const bool on = OcarinaSwitch::Read("FRITCHII_OOT_MOTION_VIEW", 0) != 0;
    return on;
}

/// FRITCHII_OOT_LOD_RATIO (boot, 1 = by default): the texture LOD bias follows the render and
/// output sizes: FSR log2(render / output) - 1 (AMD), DLSS with its jitter log2(render / output)
/// (NVIDIA). 2: DLSS gets FSR's formula too, so the same sharpness as FSR (-1.58 for
/// 1440p -> 4K). 0: the fixed numbers as before (FSR -1.58, DLSS with jitter -0.58).
inline int OcarinaLodRatioMode() {
    static const int mode = OcarinaSwitch::Read("FRITCHII_OOT_LOD_RATIO", 1);
    return mode;
}

constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(bool fsr_enabled, bool dlss_enabled,
                                                    bool dlss_available,
                                                    std::string_view override) {
    if (override == "native") {
        return OcarinaUpscaleBackend::Native;
    }
    if (override == "fsr") {
        return OcarinaUpscaleBackend::Fsr;
    }
    if ((dlss_enabled || override == "dlss") && dlss_available) {
        return OcarinaUpscaleBackend::Dlss;
    }
    return fsr_enabled ? OcarinaUpscaleBackend::Fsr : OcarinaUpscaleBackend::Native;
}

/// Whether the "OoT anti-aliasing" choice asks for AMD FSR (3.1, or 4 with FSR 3.1 as fallback).
constexpr bool OcarinaFsrChosen(Settings::OotAntiAliasing choice) {
    return choice == Settings::OotAntiAliasing::Fsr3 || choice == Settings::OotAntiAliasing::Fsr4;
}

/// The backend for the "OoT anti-aliasing" choice and the NVIDIA DLSS setting. An FSR choice wins
/// over DLSS; with Game or No sharpening, DLSS replaces the game's pass when it's on and available
/// (as before FSR came in).
constexpr OcarinaUpscaleBackend SelectOcarinaUpscaler(Settings::OotAntiAliasing choice,
                                                    bool dlss_enabled, bool dlss_available,
                                                    std::string_view override) {
    const bool fsr = OcarinaFsrChosen(choice);
    return SelectOcarinaUpscaler(fsr, dlss_enabled && !fsr, dlss_available, override);
}

/// EDEN_NX2_DLSS_JITTER (boot): overrides the "DLSS jitter" setting. Unset: the setting (on by
/// default). 0 = off, 1 = on, 2 = on with the vertical sign flipped for DLSS (a test only: the
/// offset DLSS gets is the one the scene moved by, the convention of NVIDIA's own Streamline
/// sample and of AMD's FSR, so 1 is right). -1 when unset.
inline int OcarinaDlssJitterOverride() {
    static const int mode = OcarinaSwitch::Read("EDEN_NX2_DLSS_JITTER", -1);
    return mode;
}

/// Whether DLSS gets the camera jitter and the texture bias that goes with it (DLSS is made for
/// jittered input): the override when set, else the "DLSS jitter" setting.
constexpr bool OcarinaDlssJitterOn(int override_mode, bool setting) {
    return override_mode >= 0 ? override_mode != 0 : setting;
}
inline bool OcarinaDlssJitterOn(bool setting) {
    return OcarinaDlssJitterOn(OcarinaDlssJitterOverride(), setting);
}

/// EDEN_NX2_DLSS_JITTER=2: DLSS is told the offset with y flipped.
inline bool OcarinaDlssJitterFlipY() {
    return OcarinaDlssJitterOverride() == 2;
}

/// FRITCHII_DLSS_SCALED (boot, 0 = off): with DLSS or FSR picked for OoT, keep the player's
/// resolution scale instead of setting it to 1x for the session. The upscaler then takes the
/// scene at the scaled size (2x: four times the pixels, 0.5x: a quarter) and picks its mode from
/// the real render and output sizes. Off: the scale goes to 1x as before.
inline bool OcarinaScaledUpscale() {
    static const bool on = [] {
        const char* value = std::getenv("FRITCHII_DLSS_SCALED");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return on;
}

struct OcarinaExtent {
    unsigned width{};
    unsigned height{};
    constexpr bool operator==(const OcarinaExtent&) const = default;
};

/// What the temporal upscaler writes for one frame with the scene at the resolution scale.
/// `render`: the scene's real size (the scaled size when the scaler scaled it). `target`: the real
/// size of the game's upscale target. `native_aa`: the DLSS quality "Native" (the output keeps the
/// render size). `want_4k`: the 1440p pass with the big output (`big`: 4K unless the player
/// picked another, PlanOcarinaPassOutput). DLSS and FSR only enlarge, so a scene larger than the
/// target is upscaled to its own size (native AA) and the output pass shrinks the result into
/// the target: supersampling.
struct OcarinaScaledOutput {
    OcarinaExtent output;
    bool upscale_4k{};
};

constexpr OcarinaScaledOutput SelectOcarinaScaledOutput(OcarinaExtent render, OcarinaExtent target,
                                                        bool native_aa, bool want_4k,
                                                        OcarinaExtent big = {3840, 2160}) {
    const auto fits = [render](OcarinaExtent output) {
        return render.width <= output.width && render.height <= output.height;
    };
    if (native_aa) {
        return {render, false};
    }
    if (want_4k && fits(big)) {
        return {big, true};
    }
    return {fits(target) ? target : render, false};
}

/// Texture LOD bias added to Ocarina's mipmapped samplers while FSR upscales. AMD recommends
/// log2(render / output) - 1: -1.58 for docked 1440p -> 4K. With the DLSS jitter, DLSS gets
/// NVIDIA's log2(render / output): -0.58. The fixed numbers of FRITCHII_OOT_LOD_RATIO=0.
inline float OcarinaFsrLodBias(OcarinaUpscaleBackend backend, bool dlss_jitter = false) {
    if (backend == OcarinaUpscaleBackend::Fsr) {
        return -1.58f;
    }
    return backend == OcarinaUpscaleBackend::Dlss && dlss_jitter ? -0.58f : 0.0f;
}

/// The bias for the render and output sizes (one side). `mode` 1: AMD's and NVIDIA's formulas,
/// 2: FSR's for DLSS too. DLSS gets a bias only with its jitter on (without jitter a sharper mip
/// only shimmers).
inline float OcarinaLodBiasForSizes(OcarinaUpscaleBackend backend, unsigned render_size,
                                    unsigned output_size, bool dlss_jitter, int mode) {
    if (render_size == 0 || output_size == 0 || backend == OcarinaUpscaleBackend::Native) {
        return 0.0f;
    }
    const float ratio = std::log2(static_cast<float>(render_size) /
                                  static_cast<float>(output_size));
    if (backend == OcarinaUpscaleBackend::Fsr) {
        return ratio - 1.0f;
    }
    if (!dlss_jitter) {
        return 0.0f;
    }
    return mode >= 2 ? ratio - 1.0f : ratio;
}

/// The "Texture sharpness" choice: Auto is the bias for the sizes (or the fixed numbers with
/// FRITCHII_OOT_LOD_RATIO=0), Sharper half a mip more, Custom the player's value as is (DLSS
/// without its jitter too: the player asked for it). Nothing for the game's own upscale.
inline float OcarinaTextureLodBias(OcarinaUpscaleBackend backend, unsigned render_size,
                                   unsigned output_size, bool dlss_jitter, int ratio_mode,
                                   Settings::OotTextureSharpness choice, float custom) {
    if (backend == OcarinaUpscaleBackend::Native) {
        return 0.0f;
    }
    if (choice == Settings::OotTextureSharpness::Custom) {
        return std::isfinite(custom) ? custom : 0.0f;
    }
    const float automatic =
        ratio_mode == 0
            ? OcarinaFsrLodBias(backend, dlss_jitter)
            : OcarinaLodBiasForSizes(backend, render_size, output_size, dlss_jitter, ratio_mode);
    if (choice != Settings::OotTextureSharpness::Sharper ||
        (backend == OcarinaUpscaleBackend::Dlss && !dlss_jitter)) {
        return automatic;
    }
    return automatic - 0.5f;
}

/// Whether a sampler with that bias keeps it for an image: only docked (handheld isn't measured),
/// not for images the GPU wrote (render targets would read a mip too sharp), and not once the
/// upscaler failed this session (the game's own pass runs again).
constexpr bool KeepOcarinaFsrLodBias(bool docked, bool upscaler_failed, bool gpu_written) {
    return docked && !upscaler_failed && !gpu_written;
}

} // namespace VideoCommon
