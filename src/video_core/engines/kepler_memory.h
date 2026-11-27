// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include "common/bit_field.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/engine_upload.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra::Engines {

/**
 * This Engine is known as P2MF. Documentation can be found in:
 * https://github.com/envytools/envytools/blob/master/rnndb/graph/gk104_p2mf.xml
 * https://cgit.freedesktop.org/mesa/mesa/tree/src/gallium/drivers/nouveau/nvc0/nve4_p2mf.xml.h
 */

#define KEPLERMEMORY_REG_INDEX(field_name)                                                         \
    (offsetof(Tegra::Engines::KeplerMemory::Regs, field_name) / sizeof(u32))

class KeplerMemory final : public EngineInterface {
public:
    explicit KeplerMemory(MemoryManager& memory_manager);
    ~KeplerMemory() override;

    /// Binds a rasterizer to this engine.
    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer);

    /// Write the value to the register identified by method.
    void CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) override;

    /// Write multiple values to the register identified by method.
    void CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) override;

    struct Regs {
        static constexpr size_t NUM_REGS = 0x7F;

        union {
            struct {
                INSERT_PADDING_WORDS_NOINIT(0x60);

                Upload::Registers upload;

                struct {
                    union {
                        BitField<0, 1, u32> linear;
                    };
                } exec;

                u32 data;

                INSERT_PADDING_WORDS_NOINIT(0x11);
            };
            std::array<u32, NUM_REGS> reg_array;
        };
    } regs{};

    /// GPU thread split's front (DmaPusher, EXP120): keeps a copy of the upload registers as
    /// the methods are sent (the back owns `regs`). `argument`: the last word of a run.
    void FrontNoteMethod(u32 method, u32 argument);
    /// The GPU range [begin, begin + size) the data method sent next may write, as
    /// Upload::State::ProcessData does for a linear upload. False for a block linear upload,
    /// before the front saw an exec and every register the range needs, or for an empty range.
    /// False for a block linear upload, before the front saw an exec and every register the range
    /// needs, or for an empty range.
    [[nodiscard]] bool FrontDataRange(GPUVAddr& begin, u64& size) const;
    [[nodiscard]] const MemoryManager& GetMemoryManager() const noexcept {
        return upload_state.GetMemoryManager();
    }

private:
    void ConsumeSinkImpl(Core::System& system) override;
    Upload::State upload_state;

    /// FrontNoteMethod's copy of registers 0x60 (upload) to 0x6C (exec).
    static constexpr u32 FrontFirst = 0x60;
    static constexpr u32 FrontCount = 0x0D;
    std::array<u32, FrontCount> front_regs{};
    u32 front_seen{};       ///< Bit n: front_regs[n] was sent
    u64 front_exec_size{};  ///< line_length_in * line_count at the last exec (copy_size)
    bool front_linear{};    ///< exec.linear at the last exec
};

#define ASSERT_REG_POSITION(field_name, position)                                                  \
    static_assert(offsetof(KeplerMemory::Regs, field_name) == position * 4,                        \
                  "Field " #field_name " has invalid position")

ASSERT_REG_POSITION(upload, 0x60);
ASSERT_REG_POSITION(exec, 0x6C);
ASSERT_REG_POSITION(data, 0x6D);
#undef ASSERT_REG_POSITION

} // namespace Tegra::Engines
