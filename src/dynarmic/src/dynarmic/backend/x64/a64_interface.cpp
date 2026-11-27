// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <exception>
#include <memory>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <thread>

#include <boost/icl/interval_set.hpp>
#include "common/assert.h"
#include "common/logging.h"
#include "common/nx2_hitch.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include <bit>

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/backend/x64/shared_code.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt_passes.h"

namespace Dynarmic::A64 {

using namespace Backend::X64;

static RunCodeCallbacks GenRunCodeCallbacks(A64::UserCallbacks* cb, CodePtr (*LookupBlock)(void* lookup_block_arg), void* arg, const A64::UserConfig& conf) {
    return RunCodeCallbacks{
        ArgCallback(LookupBlock, reinterpret_cast<u64>(arg)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::AddTicks>(cb)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::GetTicksRemaining>(cb)),
        conf.enable_cycle_counting,
    };
}

static std::function<void(BlockOfCode&)> GenRCP(const A64::UserConfig& conf) {
    return [conf](BlockOfCode& code) {
        if (conf.page_table) {
            code.mov(code.r14, std::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.mov(code.r13, *conf.fastmem_pointer);
        }
    };
}

namespace {

/// Fritchii: FRITCHII_JIT_DUMP (boot, default 0; Fable A 4.1 and 8.5). Writes the code each
/// block was compiled to, with its guest words, so the host code can be read offline.
/// 1 = written every 4096 blocks, 2 = only at a full cache clear and at shutdown. Host bytes
/// are read when written, so links show as patched by then.
/// FRITCHII_JIT_DUMP_CORE: the core that dumps (default 0, -1 = every core).
/// FRITCHII_JIT_DUMP_FILE: the file (default jit-dump-core{core}.bin in the working directory;
/// "{core}" is replaced by the core, and with every core dumping ".core{N}" is appended when
/// the name has no "{core}").
/// FRITCHII_JIT_DUMP_LIST: only these blocks, one hex guest pc per line (first word of a line,
/// so hot lists work too). Values below the image base are offsets from it (the lists made by
/// earlier launches, as the base is random per launch), others absolute.
/// File: a 64-byte header ("FJITDMP2", version 2, core, code space base and size, image base,
/// far region start or 0, far code flag, entry count of the list), then 64-byte records, each
/// followed by its guest words, near bytes and far bytes (see CodeDumpRecord).
struct CodeDumpSettings {
    int mode = 0;
    int core = 0;
    std::string file;
    std::vector<u64> list; // as written in the list file
};

const CodeDumpSettings& GetCodeDumpSettings() {
    static const CodeDumpSettings settings = [] {
        CodeDumpSettings s;
        const auto text = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr ? std::string{value} : std::string{};
        };
        s.mode = std::atoi(text("FRITCHII_JIT_DUMP").c_str());
        if (s.mode < 0 || s.mode > 2) {
            s.mode = 0;
        }
        if (s.mode == 0) {
            return s;
        }
        s.core = std::atoi(text("FRITCHII_JIT_DUMP_CORE").c_str());
        s.file = text("FRITCHII_JIT_DUMP_FILE");
        if (const std::string list_path = text("FRITCHII_JIT_DUMP_LIST"); !list_path.empty()) {
            if (std::FILE* const list = std::fopen(list_path.c_str(), "r")) {
                char line[256];
                while (std::fgets(line, sizeof(line), list) != nullptr) {
                    char* end = nullptr;
                    const u64 value = std::strtoull(line, &end, 16);
                    if (end != line) {
                        s.list.push_back(value);
                    }
                }
                std::fclose(list);
            }
            LOG_INFO(Core_ARM, "FRITCHII_JIT_DUMP_LIST: {} pcs from {}", s.list.size(), list_path);
        }
        LOG_INFO(Core_ARM, "FRITCHII_JIT_DUMP={}: core {}, {}", s.mode, s.core,
                 s.mode == 1 ? "written every 4096 blocks" : "written at cache clears and shutdown");
        return s;
    }();
    return settings;
}

#pragma pack(push, 1)
struct CodeDumpHeader {
    char magic[8];
    u32 version;
    u32 core;
    u64 code_base;
    u64 code_size;
    u64 image_base;
    u64 far_begin;     ///< address of the far region, 0 without one
    u32 flags;         ///< bit 0: far code
    u32 list_entries;  ///< FRITCHII_JIT_DUMP_LIST entries (0 = every block)
    u64 reserved;
};
struct CodeDumpRecord {
    u16 kind;          ///< 1 block, 2 prelude (dispatcher, handlers, constant pool)
    u16 flags;         ///< bit 0: far part present, bit 1: chained behind the block before
    u32 guest_count;   ///< guest instruction words that follow
    u64 descriptor;    ///< location descriptor (pc, fpcr, single step)
    u64 guest_pc;      ///< absolute guest pc
    s64 guest_offset;  ///< guest pc minus the image base (the same in every launch)
    u64 near_addr;
    u64 far_addr;      ///< 0 without a far part
    u32 near_size;
    u32 far_size;
    u64 sequence;      ///< emission order since the JIT was made
};
#pragma pack(pop)
static_assert(sizeof(CodeDumpHeader) == 64 && sizeof(CodeDumpRecord) == 64);

class CodeDump {
public:
    CodeDump(const UserConfig& conf, const BlockOfCode& code) : conf{conf}, code{code} {
        const CodeDumpSettings& s = GetCodeDumpSettings();
        if (s.mode == 0 || (s.core != -1 && s.core != int(conf.processor_id))) {
            return;
        }
        mode = s.mode;
        std::string name = s.file.empty() ? std::string{"jit-dump-core{core}.bin"} : s.file;
        if (const size_t at = name.find("{core}"); at != std::string::npos) {
            name.replace(at, 6, std::to_string(conf.processor_id));
        } else if (s.core == -1) {
            name += ".core" + std::to_string(conf.processor_id);
        }
        path = std::move(name);
        list_entries = u32(s.list.size());
        for (const u64 value : s.list) {
            list.insert(value < conf.image_base ? value + conf.image_base : value);
        }
    }

    ~CodeDump() {
        Flush();
        if (file != nullptr) {
            std::fclose(file);
        }
    }

    bool Active() const {
        return mode != 0;
    }

    /// After a block was emitted. `get_code` reads its guest words.
    template<typename GetCode>
    void Record(const IR::Block& block, CodePtr near_addr, size_t near_size, CodePtr far_addr, size_t far_size, bool chained, GetCode&& get_code) {
        ++sequence;
        const A64::LocationDescriptor location{block.Location()};
        const u64 pc = location.PC();
        if (!list.empty() && list.count(pc) == 0) {
            return;
        }
        Pending& p = pending.emplace_back();
        p.record.kind = 1;
        p.record.flags = u16((far_size != 0 ? 1 : 0) | (chained ? 2 : 0));
        p.record.descriptor = location.UniqueHash();
        p.record.guest_pc = pc;
        p.record.guest_offset = s64(pc - conf.image_base);
        p.record.near_addr = u64(reinterpret_cast<uintptr_t>(near_addr));
        p.record.far_addr = far_size != 0 ? u64(reinterpret_cast<uintptr_t>(far_addr)) : 0;
        p.record.near_size = u32(near_size);
        p.record.far_size = u32(far_size);
        p.record.sequence = sequence;
        const u64 end_pc = A64::LocationDescriptor{block.EndLocation()}.PC();
        const u64 count = end_pc > pc ? (end_pc - pc) / 4 : 0;
        p.words.reserve(size_t(count));
        for (u64 i = 0; i < count; ++i) {
            p.words.push_back(get_code(pc + i * 4).value_or(0));
        }
        p.record.guest_count = u32(p.words.size());
        if (mode == 1 && pending.size() >= 4096) {
            Flush();
        }
    }

    /// Writes what is pending (the host bytes as they are now). Call before the code goes.
    void Flush() {
        if (pending.empty() || !Open()) {
            pending.clear();
            return;
        }
        for (const Pending& p : pending) {
            WriteRecord(p.record, p.words.data());
        }
        std::fflush(file);
        pending.clear();
    }

private:
    struct Pending {
        CodeDumpRecord record{};
        std::vector<u32> words;
    };

    bool Open() {
        if (file != nullptr) {
            return true;
        }
        if (failed) {
            return false;
        }
        file = std::fopen(path.c_str(), "wb");
        if (file == nullptr) {
            failed = true;
            LOG_WARNING(Core_ARM, "FRITCHII_JIT_DUMP: can't open {}", path);
            return false;
        }
        CodeDumpHeader header{};
        std::memcpy(header.magic, "FJITDMP2", 8);
        header.version = 2;
        header.core = conf.processor_id;
        header.code_base = u64(reinterpret_cast<uintptr_t>(code.getCode()));
        header.code_size = code.GetTotalCodeSize();
        header.image_base = conf.image_base;
        header.far_begin = code.HasFarCode() ? u64(reinterpret_cast<uintptr_t>(code.getCode())) + code.FarCodeBegin() : 0;
        header.flags = code.HasFarCode() ? 1 : 0;
        header.list_entries = list_entries;
        std::fwrite(&header, sizeof(header), 1, file);
        // The prelude: everything before the first block, as it never moves.
        CodeDumpRecord prelude{};
        prelude.kind = 2;
        prelude.near_addr = header.code_base;
        prelude.near_size = u32(static_cast<const u8*>(code.GetCodeBegin()) - code.getCode());
        WriteRecord(prelude, nullptr);
        LOG_INFO(Core_ARM, "FRITCHII_JIT_DUMP: core {} writes {} (image base {:#x})",
                 conf.processor_id, path, conf.image_base);
        return true;
    }

    void WriteRecord(const CodeDumpRecord& record, const u32* words) {
        std::fwrite(&record, sizeof(record), 1, file);
        if (record.guest_count != 0) {
            std::fwrite(words, sizeof(u32), record.guest_count, file);
        }
        if (record.near_size != 0) {
            std::fwrite(reinterpret_cast<const void*>(uintptr_t(record.near_addr)), 1, record.near_size, file);
        }
        if (record.far_size != 0) {
            std::fwrite(reinterpret_cast<const void*>(uintptr_t(record.far_addr)), 1, record.far_size, file);
        }
    }

    const UserConfig& conf;
    const BlockOfCode& code;
    int mode = 0;
    std::string path;
    u32 list_entries = 0;
    ::Common::unordered_set<u64> list;
    std::vector<Pending> pending;
    u64 sequence = 0;
    std::FILE* file = nullptr;
    bool failed = false;
};

} // namespace

/// Fritchii: FRITCHII_JIT_HALT_LATENCY (boot, default 0). 1 = each core times a halt request
/// from another thread (HaltExecution: interrupts, cache invalidations) to the JIT's return from
/// Run, and logs every 10 s: halts, p50, p99, longest (with the pc it stopped at), and "over
/// 1 ms: N (must be 0)". It bounds how long a halt can wait for a check, which
/// FRITCHII_JIT_CHAIN bit 2 makes rarer (Fable A 9.4). Requests the core makes itself from a
/// callback (svc, aborts) are not timed: a CheckHalt follows them at once.
/// Off: one predictable branch in HaltExecution and Run, no change to generated code.
static bool HaltLatencyMode() {
    static const bool on = [] {
        const char* text = std::getenv("FRITCHII_JIT_HALT_LATENCY");
        const bool value = text != nullptr && std::atoi(text) == 1;
        if (value) {
            LOG_INFO(Core_ARM, "FRITCHII_JIT_HALT_LATENCY=1: halt request to JIT return timed per core, "
                               "line every 10 s");
        }
        return value;
    }();
    return on;
}

/// The JIT whose Run this thread is inside (FRITCHII_JIT_HALT_LATENCY only).
static thread_local const void* t_running_jit = nullptr;

static s64 SteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static Optimization::PolyfillOptions GenPolyfillOptions(const BlockOfCode& code) {
    return Optimization::PolyfillOptions{
        .sha256 = !code.HasHostFeature(HostFeature::SHA),
        .vector_multiply_widen = true,
    };
}

namespace {

/// Fritchii (FRITCHII_JIT_SHARED): the code space, its emitter and the compile scratch. One per
/// JIT in mode 0 (the old layout); in shared mode one per process, used by every guest core's JIT,
/// each of which keeps its own state block, dispatch table, return stack buffer, indirect slots
/// and run loop. The prelude (dispatcher, handlers, fallbacks) reads every per-core value through
/// the state register (FRITCHII_JIT_STATE_REFS), so one copy serves all cores.
struct CodeEngine {
    CodeEngine(const UserConfig& conf, CodePtr (*lookup)(void*), void* jit_impl, Jit* jit, const A64JitState& state, bool shared)
        : shared_code(shared ? std::make_unique<SharedCode>() : nullptr)
        , code(GenRunCodeCallbacks(conf.callbacks, lookup, jit_impl, conf), JitStateInfo{state}, conf.code_cache_size, GenRCP(conf),
               A64EmitX64::PinSetup(conf), A64EmitX64::StateRefSetup(conf, jit_impl))
        , emitter(code, conf, jit)
        , polyfill_options(GenPolyfillOptions(code))
        , creator(jit_impl)
        , creator_conf(conf) {
        emitter.shared_code = shared_code.get();
    }

    std::unique_ptr<SharedCode> shared_code;  ///< null = one cache per core
    BlockOfCode code;
    A64EmitX64 emitter;
    Optimization::PolyfillOptions polyfill_options;
    IR::Block ir_block{LocationDescriptor(0, FP::FPCR(0), false)};
    const void* creator;  ///< the JIT that made it (its dispatch table is the emitter's)
    const UserConfig creator_conf;
    /// Shared mode: the JITs that run this cache, by core (halted for a whole-cache clear).
    std::mutex members_mutex;
    std::array<void*, SharedCode::max_cores> members{};
    /// Shared mode: the counters' line (any core's Run, every 10 s at most).
    std::mutex log_mutex;
    s64 log_ns = 0;
    u64 logged_compiled = 0;
    u64 logged_lock_wait_ns = 0;
    u64 logged_other_core_lookups = 0;
};

/// Fritchii (FRITCHII_JIT_SHARED): why this JIT keeps its own cache (nullptr = it can share).
const char* SharedCodeRefusal(const UserConfig& conf) {
#ifdef DYNARMIC_ENABLE_NO_EXECUTE_SUPPORT
    static_cast<void>(conf);
    return "W^X builds change the code's protection while other cores run it";
#else
    if (A64EmitX64::StateRefsMode() == 0) {
        return "it needs FRITCHII_JIT_STATE_REFS=1 (blocks must not hold per-core addresses)";
    }
    if (conf.page_table == nullptr) {
        return "there is no page table";
    }
    if (conf.enable_cycle_counting) {
        return "cycle counting is on";
    }
    if (conf.check_halt_on_memory_access) {
        return "the debugger is on";
    }
    if (conf.very_verbose_debugging_output) {
        return "verbose debugging output is on";
    }
    if (conf.processor_id >= SharedCode::max_cores) {
        return "the core number is too large";
    }
    if (conf.HasOptimization(OptimizationFlag::BlockLinking) &&
        ((A64EmitX64::ChainBits() & 5) == 0 || !A64EmitX64::FarCodeEnabled() || conf.code_cache_size < (u32{64} << 20))) {
        return "it needs compact links (FRITCHII_JIT_CHAIN=4 or 1 with FRITCHII_JIT_FAR_CODE=1), the only links that "
               "can be patched while other cores run them";
    }
    return nullptr;
#endif
}

/// Fritchii (FRITCHII_JIT_SHARED): every setting that shapes generated code is the same (only the
/// per-core objects, which blocks read through the state register, may differ).
bool SameCodeShape(const UserConfig& a, const UserConfig& b) {
    return a.page_table == b.page_table && a.fastmem_pointer == b.fastmem_pointer && a.global_monitor == b.global_monitor &&
           a.optimizations == b.optimizations && a.unsafe_optimizations == b.unsafe_optimizations &&
           a.page_table_address_space_bits == b.page_table_address_space_bits &&
           a.page_table_pointer_mask == b.page_table_pointer_mask && a.page_table_log2_stride == b.page_table_log2_stride &&
           a.page_table_marked_bit == b.page_table_marked_bit && a.page_table_sign_extension == b.page_table_sign_extension &&
           a.cntfrq_el0 == b.cntfrq_el0 && a.ctr_el0 == b.ctr_el0 && a.dczid_el0 == b.dczid_el0 &&
           a.fastmem_address_space_bits == b.fastmem_address_space_bits && a.code_cache_size == b.code_cache_size &&
           a.detect_misaligned_access_via_page_table == b.detect_misaligned_access_via_page_table &&
           a.hook_data_cache_operations == b.hook_data_cache_operations && a.hook_isb == b.hook_isb &&
           a.hook_hint_instructions == b.hook_hint_instructions && a.silently_mirror_page_table == b.silently_mirror_page_table &&
           a.absolute_offset_page_table == b.absolute_offset_page_table &&
           a.only_detect_misalignment_via_page_table_on_page_boundary == b.only_detect_misalignment_via_page_table_on_page_boundary &&
           a.recompile_on_fastmem_failure == b.recompile_on_fastmem_failure && a.silently_mirror_fastmem == b.silently_mirror_fastmem &&
           a.fastmem_exclusive_access == b.fastmem_exclusive_access &&
           a.recompile_on_exclusive_fastmem_failure == b.recompile_on_exclusive_fastmem_failure &&
           a.define_unpredictable_behaviour == b.define_unpredictable_behaviour && a.wall_clock_cntpct == b.wall_clock_cntpct &&
           a.inline_cntpct_factor == b.inline_cntpct_factor &&
           (a.inline_cntpct_last == nullptr) == (b.inline_cntpct_last == nullptr) &&
           a.inline_cntpct_spin_ticks == b.inline_cntpct_spin_ticks && a.check_halt_on_memory_access == b.check_halt_on_memory_access &&
           a.enable_cycle_counting == b.enable_cycle_counting && (a.tpidr_el0 == nullptr) == (b.tpidr_el0 == nullptr) &&
           (a.tpidrro_el0 == nullptr) == (b.tpidrro_el0 == nullptr) &&
           a.very_verbose_debugging_output == b.very_verbose_debugging_output;
}

/// Fritchii (FRITCHII_JIT_SHARED): this JIT's engine: its own (mode 0, or refused), or the
/// process's (keyed by page table and fastmem arena), made by the first core that asks.
std::shared_ptr<CodeEngine> AcquireCodeEngine(const UserConfig& conf, CodePtr (*lookup)(void*), void* jit_impl, Jit* jit,
                                              const A64JitState& state) {
    const auto own = [&](bool shared) {
        return std::make_shared<CodeEngine>(conf, lookup, jit_impl, jit, state, shared);
    };
    if (A64EmitX64::SharedMode() == 0) {
        return own(false);
    }
    if (const char* const why = SharedCodeRefusal(conf)) {
        LOG_WARNING(Core_ARM, "JIT shared code (FRITCHII_JIT_SHARED): core {} keeps its own code cache: {}", conf.processor_id, why);
        return own(false);
    }
    static std::mutex registry_mutex;
    static std::map<std::pair<const void*, std::uintptr_t>, std::weak_ptr<CodeEngine>> registry;
    std::scoped_lock lock{registry_mutex};
    std::erase_if(registry, [](const auto& entry) { return entry.second.expired(); });
    std::weak_ptr<CodeEngine>& entry = registry[{static_cast<const void*>(conf.page_table), conf.fastmem_pointer.value_or(0)}];
    if (auto engine = entry.lock()) {
        std::scoped_lock members_lock{engine->members_mutex};
        SharedCode::CoreSlot& slot = engine->shared_code->cores[conf.processor_id];
        if (!SameCodeShape(engine->creator_conf, conf)) {
            LOG_WARNING(Core_ARM, "JIT shared code (FRITCHII_JIT_SHARED): core {} keeps its own code cache: its settings "
                                  "differ from core {}'s", conf.processor_id, engine->creator_conf.processor_id);
            return own(false);
        }
        if (slot.registered.load(std::memory_order_relaxed)) {
            LOG_WARNING(Core_ARM, "JIT shared code (FRITCHII_JIT_SHARED): core {} keeps its own code cache: another JIT of "
                                  "that core runs the process's", conf.processor_id);
            return own(false);
        }
        slot.registered.store(true, std::memory_order_relaxed);
        return engine;
    }
    auto engine = own(true);
    if (conf.HasOptimization(OptimizationFlag::BlockLinking) && !engine->emitter.compact_links) {
        LOG_ERROR(Core_ARM, "JIT shared code (FRITCHII_JIT_SHARED): compact links did not come on, core {} keeps its own "
                            "code cache", conf.processor_id);
        return own(false);
    }
    engine->shared_code->cores[conf.processor_id].registered.store(true, std::memory_order_relaxed);
    entry = engine;
    return engine;
}

} // namespace

struct Jit::Impl final {
public:
    Impl(Jit* jit, UserConfig conf_)
        : conf(conf_)
        , engine(AcquireCodeEngine(conf, &GetCurrentBlockThunk, this, jit, jit_state))
        , block_of_code(engine->code)
        , emitter(engine->emitter)
        , polyfill_options(engine->polyfill_options)
        , ir_block(engine->ir_block)
        , shared(engine->shared_code.get())
        , shared_verify(shared != nullptr && A64EmitX64::SharedMode() == 2)
    {
        ASSERT(conf.page_table_address_space_bits >= 12 && conf.page_table_address_space_bits <= 64);
        // Fritchii (FRITCHII_JIT_VERIFY=2): the guest registers whose slots hold a poison while
        // generated code runs.
        if (block_of_code.Pins().shadow) {
            for (const PinnedRegister& pin : block_of_code.Pins().regs) {
                pin_poison_mask |= u64(1) << (pin.poison & 31);
            }
        }
        halt_latency = HaltLatencyMode();
        // Fritchii (FRITCHII_JIT_STATE_REFS): the per-core addresses generated code reads from
        // the state block (all 0 with the switch off, then nothing reads them).
        static_assert(size_t(CoreRef::Count) <= A64JitState::core_ref_count);
        if (shared != nullptr) {
            JoinSharedCode();  // FRITCHII_JIT_SHARED: this core's own values and tables
            return;
        }
        for (size_t i = 0; i < size_t(CoreRef::Count); ++i) {
            jit_state.core_refs[i] = block_of_code.StateRefs().values[i];
        }
        // FRITCHII_JIT_STATE_REFS=2: the scan also looks for this object (state block, code
        // object, emitter).
        emitter.AddStateRefScanRange(this, sizeof(*this), "JIT object");
    }

    ~Impl() {
        if (shared != nullptr) {
            std::scoped_lock lock{engine->members_mutex};
            engine->members[conf.processor_id] = nullptr;
            shared->cores[conf.processor_id].registered.store(false, std::memory_order_relaxed);
        }
    }

    HaltReason Run() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        is_executing = true;
        if (shared != nullptr) {
            EnterCode();  // FRITCHII_JIT_SHARED
        }
        // TODO: Check code alignment
        const CodePtr current_code_ptr = [this] {
            // RSB optimization
            const u32 new_rsb_ptr = (jit_state.rsb_ptr - 1) & A64JitState::RSB_PTR_MASK;
            if (jit_state.GetUniqueHash() == jit_state.rsb_location_descriptors[new_rsb_ptr]) {
                jit_state.rsb_ptr = new_rsb_ptr;
                return CodePtr(jit_state.rsb_codeptrs[new_rsb_ptr]);
            }
            // Fritchii (FRITCHII_JIT_CHAIN=1): chained blocks start unaligned.
            if (emitter.chain_successors) {
                return GetCurrentBlock();
            }
            return CodePtr((uintptr_t(GetCurrentBlock()) + 15) & ~uintptr_t(15));
        }();
        // Fritchii (FRITCHII_JIT_HALT_LATENCY): a request from before this point either set a
        // halt bit (the entry returns at once) or was served; only later ones are timed.
        s64 run_start_ns = 0;
        if (halt_latency) [[unlikely]] {
            t_running_jit = this;
            run_start_ns = SteadyNs();
            halt_request_ns.store(0, std::memory_order_relaxed);
        }
        const HaltReason hr = block_of_code.RunCode(&jit_state, current_code_ptr);
        if (shared != nullptr) {
            LeaveCode();
        }
        if (halt_latency) [[unlikely]] {
            t_running_jit = nullptr;
            NoteHaltLatency(run_start_ns);
        }
        if (pin_poison_mask != 0) [[unlikely]] {
            CheckPinsAfterExit();
        }
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        // Fritchii: FRITCHII_JIT_INDIRECT_CACHE's counters, written by this thread's code.
        if (emitter.indirect_cache_mode != 0 && (++runs_since_counter_look & 0x3ff) == 0) [[unlikely]] {
            if (shared != nullptr) {
                emitter.LogIndirectCounters(indirect_counters, indirect_log, conf.processor_id);
            } else {
                emitter.LogIndirectCounters();
            }
        }
        // Fritchii: FRITCHII_JIT_VERIFY's must-be-0 count, written by this thread's code. With
        // FRITCHII_JIT_SHARED the emitter's "logged" marks are shared: one core logs at a time.
        if (emitter.pin_verify && (++runs_since_pin_look & 0x3f) == 0) [[unlikely]] {
            std::unique_lock<std::mutex> log_lock;
            if (shared != nullptr) {
                log_lock = std::unique_lock<std::mutex>{engine->log_mutex, std::try_to_lock};
            }
            if (shared == nullptr || log_lock.owns_lock()) {
                if (emitter.pin_shadow) {
                    emitter.LogPinChecks(jit_state, pin_poison_reads.load(std::memory_order_relaxed), pin_exit_poison);
                } else {
                    emitter.LogPinMismatches(jit_state.pin_mismatches);
                }
            }
        }
        if (emitter.pin_stats && (++runs_since_pin_look & 0x3ff) == 0) [[unlikely]] {
            std::unique_lock<std::mutex> log_lock;
            if (shared != nullptr) {
                log_lock = std::unique_lock<std::mutex>{engine->log_mutex, std::try_to_lock};
            }
            if (shared == nullptr || log_lock.owns_lock()) {
                emitter.LogPinStats(jit_state.pin_stats);
            }
        }
        if (shared != nullptr && (++runs_since_shared_look & 0x3ff) == 0) [[unlikely]] {
            LogSharedCode();
        }
        return hr;
    }

    HaltReason Step() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        is_executing = true;
        if (shared != nullptr) {
            EnterCode();  // FRITCHII_JIT_SHARED
        }
        const HaltReason hr = block_of_code.StepCode(&jit_state, GetCurrentSingleStep());
        if (shared != nullptr) {
            LeaveCode();
        }
        if (pin_poison_mask != 0) [[unlikely]] {
            CheckPinsAfterExit();  // Fritchii (FRITCHII_JIT_VERIFY=2)
        }
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        return hr;
    }

    void ClearCache() {
        if (shared != nullptr) {
            RequestSharedClear();  // FRITCHII_JIT_SHARED: every core stops, then one clears
            return;
        }
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(u64 start_address, size_t length) {
        std::unique_lock lock{invalidation_mutex};
        const auto end_address = static_cast<u64>(start_address + length - 1);
        const auto range = boost::icl::discrete_interval<u64>::closed(start_address, end_address);
        invalid_cache_ranges.add(range);
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        ASSERT(!is_executing);
        const auto core_refs = jit_state.core_refs; // Fritchii (FRITCHII_JIT_STATE_REFS)
        jit_state = {};
        jit_state.core_refs = core_refs;
    }

    void HaltExecution(HaltReason hr) {
        // Fritchii (FRITCHII_JIT_HALT_LATENCY): the earliest request from another thread not yet
        // served is timed; stamped before the bit, so a return that saw the bit finds the stamp.
        if (halt_latency && t_running_jit != this) [[unlikely]] {
            s64 expected = 0;
            halt_request_ns.compare_exchange_strong(expected, SteadyNs(), std::memory_order_relaxed);
        }
        Atomic::Or(&jit_state.halt_reason, static_cast<u32>(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&jit_state.halt_reason, ~static_cast<u32>(hr));
    }

    u64 GetSP() const {
        return CheckPoisonRead(31, jit_state.sp);
    }

    void SetSP(u64 value) {
        jit_state.sp = value;
    }

    u64 GetPC() const {
        return jit_state.pc;
    }

    void SetPC(u64 value) {
        jit_state.pc = value;
    }

    u64 GetRegister(size_t index) const {
        if (index == 31)
            return GetSP();
        return CheckPoisonRead(index, jit_state.reg.at(index));
    }

    void SetRegister(size_t index, u64 value) {
        if (index == 31)
            return SetSP(value);
        jit_state.reg.at(index) = value;
    }

    std::array<u64, 31> GetRegisters() const {
        if (pin_poison_mask != 0) [[unlikely]] {
            for (size_t i = 0; i < 31; ++i) {
                CheckPoisonRead(i, jit_state.reg[i]);
            }
        }
        return jit_state.reg;
    }

    void SetRegisters(const std::array<u64, 31>& value) {
        jit_state.reg = value;
    }

    Vector GetVector(size_t index) const {
        return {jit_state.vec.at(index * 2), jit_state.vec.at(index * 2 + 1)};
    }

    void SetVector(size_t index, Vector value) {
        jit_state.vec.at(index * 2) = value[0];
        jit_state.vec.at(index * 2 + 1) = value[1];
    }

    std::array<Vector, 32> GetVectors() const {
        std::array<Vector, 32> ret;
        static_assert(sizeof(ret) == sizeof(jit_state.vec));
        std::memcpy(ret.data(), jit_state.vec.data(), sizeof(jit_state.vec));
        return ret;
    }

    void SetVectors(const std::array<Vector, 32>& value) {
        static_assert(sizeof(value) == sizeof(jit_state.vec));
        std::memcpy(jit_state.vec.data(), value.data(), sizeof(jit_state.vec));
    }

    u32 GetFpcr() const {
        return jit_state.GetFpcr();
    }

    void SetFpcr(u32 value) {
        jit_state.SetFpcr(value);
    }

    u32 GetFpsr() const {
        return jit_state.GetFpsr();
    }

    void SetFpsr(u32 value) {
        jit_state.SetFpsr(value);
    }

    u32 GetPstate() const {
        return jit_state.GetPstate();
    }

    void SetPstate(u32 value) {
        jit_state.SetPstate(value);
    }

    void ClearExclusiveState() {
        jit_state.exclusive_state = 0;
    }

    bool IsExecuting() const {
        return is_executing;
    }

    std::string Disassemble() const {
        const size_t size = reinterpret_cast<const char*>(block_of_code.getCurr()) - reinterpret_cast<const char*>(block_of_code.GetCodeBegin());
        auto const* p = reinterpret_cast<const char*>(block_of_code.GetCodeBegin());
        return Common::DisassembleX64(p, p + size);
    }

private:
    static CodePtr GetCurrentBlockThunk(void* thisptr) {
        Jit::Impl* this_ = static_cast<Jit::Impl*>(thisptr);
        return this_->GetCurrentBlock();
    }

    IR::LocationDescriptor GetCurrentLocation() const {
        return IR::LocationDescriptor{jit_state.GetUniqueHash()};
    }

    CodePtr GetCurrentBlock() {
        if (conf.counters) {
            ++conf.counters->block_lookups;
        }
        return GetBlock(GetCurrentLocation());
    }

public:
    bool PrecompileBlock(u64 location) {
        if (is_executing) {
            return false;
        }
        if (shared != nullptr) {
            return PrecompileShared(location);
        }
        // Fritchii (FRITCHII_JIT_HOT_PACK): a whole-cache clear asked for since the last run is
        // done now, as the next Run would do it first, instead of compiling into the code it
        // throws away.
        bool clear_pending = false;
        {
            std::unique_lock lock{invalidation_mutex};
            clear_pending = invalidate_entire_cache;
        }
        if (clear_pending) {
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
        }
        const IR::LocationDescriptor descriptor{location};
        if (emitter.GetBasicBlock(descriptor)) {
            return false;
        }
        if (!WarmSpaceLeft()) {
            return false;
        }
        static_cast<void>(GetBlock(descriptor));
        return true;
    }

private:
    /// Room for a warm compile (EDEN_NX2_JIT_WARM, FRITCHII_JIT_BLOCK_LIST).
    bool WarmSpaceLeft() {
        // Fritchii (FRITCHII_JIT_FAR_CODE): half of the smaller region, as 256 MiB is of 512.
        const size_t keep_free = block_of_code.HasFarCode() ? block_of_code.SmallestRegionSize() / 2
                                                            : (size_t{256} << 20);
        if (block_of_code.SpaceRemaining() < keep_free) {
            // Fritchii (FRITCHII_HITCH_SPLIT): the warm-up stops here; the blocks it skips are
            // compiled while the game waits, by every core that runs them.
            if (::Common::Nx2Hitch::SplitEnabled()) {
                ::Common::Nx2Hitch::AddCount(::Common::Nx2Hitch::Cause::WarmRefusedSpace, 0, 1);
                if (!warm_refusal_logged) {
                    warm_refusal_logged = true;
                    LOG_INFO(Core_ARM,
                             "JIT warm compiles refused from now on (FRITCHII_HITCH_SPLIT) core {}: "
                             "code cache keeps {} MiB free, near {} MiB used, far {} MiB used",
                             conf.processor_id, keep_free >> 20, block_of_code.NearCodeUsed() >> 20,
                             block_of_code.FarCodeUsed() >> 20);
                }
            }
            return false;
        }
        return true;
    }

    CodePtr GetCurrentSingleStep() {
        return GetBlock(A64::LocationDescriptor{GetCurrentLocation()}.SetSingleStepping(true));
    }

    static constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;

    CodePtr GetBlock(IR::LocationDescriptor descriptor) {
        if (shared != nullptr) {
            return GetSharedBlock(descriptor);  // FRITCHII_JIT_SHARED
        }
        if (auto block = emitter.GetBasicBlock(descriptor))
            return block->entrypoint;
        // Fritchii hitch attribution (2026-11-10): new game code compiled while the game waits.
        // Each emulated core has its own JIT (fable52_answer_12): count code another core has.
        const ::Common::Nx2Hitch::Scope hitch_timed{
            ::Common::Nx2Hitch::CompileCause(descriptor.Value(), conf.processor_id)};

        if (block_of_code.SpaceRemaining() < MINIMUM_REMAINING_CODESIZE) {
            // Immediately evacuate cache
            invalidate_entire_cache = true;
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
        }
        return CompileBlock(descriptor);
    }

    /// Translates, emits and links the block (FRITCHII_JIT_SHARED: under the compile lock).
    CodePtr CompileBlock(IR::LocationDescriptor descriptor) {
        block_of_code.EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);

        // JIT Compile
        const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
        // LocationDescriptor ctor() does important ops (like tflags) do not skip
        auto const arch_descriptor = A64::LocationDescriptor{descriptor};
        ir_block.Reset(arch_descriptor);
        // Fritchii: called from generated code (block lookup), so a throw cannot unwind past it
        // and the process would end without a word: log it, then abort (the crash handler's
        // report and dump follow).
        try {
            A64::Translate(ir_block, arch_descriptor, get_code, {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});
            Optimization::Optimize(ir_block, conf, polyfill_options);
        } catch (const std::exception& error) {
            ::Common::Log::FmtCrashMessage(__FILE__, __LINE__, __func__,
                                           "JIT translate of {:#x} (core {}) threw: {}",
                                           arch_descriptor.PC(), conf.processor_id, error.what());
            std::abort();
        }
        if (conf.counters) {
            ++conf.counters->blocks_compiled;
        }
        if (shared != nullptr) {
            shared->compiled.fetch_add(1, std::memory_order_relaxed);
        }
        const auto bdesc = emitter.Emit(ir_block);
        if (code_dump.Active()) [[unlikely]] {
            code_dump.Record(ir_block, bdesc.entrypoint, bdesc.size, emitter.last_far_entry,
                             emitter.last_far_size, false, get_code);
        }
        if (emitter.chain_successors) {
            ChainSuccessors(get_code);
        }
        return bdesc.entrypoint;
    }

    /// Fritchii (FRITCHII_JIT_CHAIN=1): while the block just emitted ends in a link slot to a
    /// block not compiled yet, that block is compiled right behind it (the slot becomes a nop).
    /// Only readable code, never a cache clear in between.
    template<typename GetCode>
    void ChainSuccessors(const GetCode& get_code) {
        for (int depth = 0; depth < A64EmitX64::ChainDepth(); ++depth) {
            const auto next = emitter.ChainCandidate();
            if (!next || emitter.GetBasicBlock(*next)) {
                return;
            }
            const auto next_descriptor = A64::LocationDescriptor{*next};
            if (next_descriptor.SingleStepping() || !get_code(next_descriptor.PC())) {
                return;
            }
            if (block_of_code.SpaceRemaining() < 2 * MINIMUM_REMAINING_CODESIZE) {
                return;
            }
            block_of_code.EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);
            ir_block.Reset(next_descriptor);
            try {
                A64::Translate(ir_block, next_descriptor, get_code, {conf.define_unpredictable_behaviour, conf.wall_clock_cntpct});
                Optimization::Optimize(ir_block, conf, polyfill_options);
            } catch (const std::exception& error) {
                ::Common::Log::FmtCrashMessage(__FILE__, __LINE__, __func__,
                                               "JIT translate of chained {:#x} (core {}) threw: {}",
                                               next_descriptor.PC(), conf.processor_id, error.what());
                std::abort();
            }
            if (conf.counters) {
                ++conf.counters->blocks_compiled;
            }
            if (shared != nullptr) {
                shared->compiled.fetch_add(1, std::memory_order_relaxed);
            }
            const auto bdesc = emitter.Emit(ir_block, true);
            if (code_dump.Active()) [[unlikely]] {
                code_dump.Record(ir_block, bdesc.entrypoint, bdesc.size, emitter.last_far_entry,
                                 emitter.last_far_size, true, get_code);
            }
        }
    }

    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation) && shared != nullptr) {
            PerformSharedInvalidation();  // FRITCHII_JIT_SHARED
            return;
        }
        if (Has(hr, HaltReason::CacheInvalidation)) {
            std::unique_lock lock{invalidation_mutex};

            ClearHalt(HaltReason::CacheInvalidation);

            if (!invalidate_entire_cache && invalid_cache_ranges.empty()) {
                return;
            }

            jit_state.ResetRSB();
            if (invalidate_entire_cache) {
                if (conf.counters) {
                    ++conf.counters->cache_clears;
                }
                code_dump.Flush(); // FRITCHII_JIT_DUMP: before the code goes
                if (block_of_code.HasFarCode()) {
                    // FRITCHII_JIT_FAR_CODE: how full each region was (the 5/8 : 3/8 split).
                    LOG_INFO(Core_ARM, "JIT cache clear (FRITCHII_JIT_FAR_CODE) core {}: near {} MiB used, far {} MiB used",
                             conf.processor_id, block_of_code.NearCodeUsed() >> 20, block_of_code.FarCodeUsed() >> 20);
                }
                block_of_code.ClearCache();
                emitter.ClearCache();
            } else {
                emitter.InvalidateCacheRanges(invalid_cache_ranges);
            }
            invalid_cache_ranges.clear();
            invalidate_entire_cache = false;
        }
    }

    // Fritchii: FRITCHII_JIT_SHARED. Every guest core's JIT runs the process's code cache. The
    // hot path takes no lock: running code, block-to-block links, the per-core dispatch table and
    // the shared map's lookup. Compiling, linking, removing and the fault handler's patch table
    // take the compile lock (one compiler at a time). A whole-cache clear sets clear_pending,
    // halts every core and runs once all of them are out of generated code (in_code, set at every
    // entry before clear_pending is read: a core that enters later backs off and waits). Removed
    // blocks keep their code until such a clear, so a core still inside one finishes safely; each
    // core drops removed blocks from its own dispatch table, slots and return stack buffer at its
    // next entry (SharedCode's removal log), as each core's own invalidation did before.

    /// This core's values in its state block, its own tables, and its place in the engine.
    void JoinSharedCode() {
        const bool creator = engine->creator == this;
        dispatch_table = emitter.fast_dispatch_table.get();
        indirect_slots = emitter.indirect_slots.get();
        indirect_counters = emitter.indirect_counters.data();
        if (!creator) {
            own_dispatch_table.reset(new A64EmitX64::FastDispatchEntry[emitter.fast_dispatch_table_size]);
            dispatch_table = own_dispatch_table.get();
            if (emitter.indirect_slots) {
                own_indirect_slots.reset(new A64EmitX64::FastDispatchEntry[A64EmitX64::indirect_slot_capacity]);
                indirect_slots = own_indirect_slots.get();
            }
            indirect_counters = own_indirect_counters.data();
        }
        const StateRefConfig own = A64EmitX64::StateRefSetup(conf, this);
        for (size_t i = 0; i < size_t(CoreRef::Count); ++i) {
            jit_state.core_refs[i] = own.values[i];
        }
        const auto set = [this](CoreRef which, const void* value) {
            jit_state.core_refs[size_t(which)] = u64(reinterpret_cast<uintptr_t>(value));
        };
        // The exclusive accesses through the monitor in C++ take this core's config (its
        // processor id and callbacks), not the engine's copy of the first core's.
        set(CoreRef::Conf, &conf);
        set(CoreRef::FastDispatchTable, dispatch_table);
        set(CoreRef::IndirectSlots, indirect_slots);
        set(CoreRef::IndirectCounters, indirect_counters);
        core_bit = SharedCode::CoreBit(conf.processor_id);
        {
            std::scoped_lock lock{shared->compile_mutex};
            // FRITCHII_JIT_STATE_REFS=2: blocks are scanned for every core's objects.
            emitter.AddStateRefScanRange(this, sizeof(*this), "JIT object");
            if (!creator) {
                emitter.AddStateRefScanRange(conf.callbacks, sizeof(void*), "callbacks");
                emitter.AddStateRefScanRange(conf.tpidr_el0, sizeof(u64), "tpidr_el0");
                emitter.AddStateRefScanRange(conf.tpidrro_el0, sizeof(u64), "tpidrro_el0");
                emitter.AddStateRefScanRange(reinterpret_cast<const void*>(own.values[size_t(CoreRef::MonitorAddress)]), sizeof(u64), "own monitor address");
                emitter.AddStateRefScanRange(reinterpret_cast<const void*>(own.values[size_t(CoreRef::MonitorValue)]), 16, "own monitor value");
                emitter.AddStateRefScanRange(conf.inline_cntpct_last, conf.inline_cntpct_last ? sizeof(u64) : 0, "inline cntpct last read");
                emitter.AddStateRefScanRange(dispatch_table, emitter.fast_dispatch_table_size * sizeof(A64EmitX64::FastDispatchEntry), "fast dispatch table");
                emitter.AddStateRefScanRange(indirect_slots, indirect_slots ? A64EmitX64::indirect_slot_capacity * sizeof(A64EmitX64::FastDispatchEntry) : 0, "indirect slots");
            }
            seen_clear_epoch = shared->clear_epoch.load(std::memory_order_acquire);
        }
        {
            std::scoped_lock lock{shared->removal_mutex};
            seen_removal = shared->removal_seq.load(std::memory_order_relaxed);
        }
        {
            std::scoped_lock lock{engine->members_mutex};
            engine->members[conf.processor_id] = this;
        }
        LOG_INFO(Core_ARM, "JIT shared code (FRITCHII_JIT_SHARED={}) core {} {} the process's code cache ({} MiB{})",
                 A64EmitX64::SharedMode(), conf.processor_id, creator ? "made" : "joined", block_of_code.GetTotalCodeSize() >> 20,
                 emitter.compact_links ? ", compact links" : ", no block links");
    }

    /// Before generated code runs (Run, Step, a warm compile): marks this core inside it, unless a
    /// whole-cache clear is pending (then helps it finish first), and catches up with removals.
    void EnterCode() {
        SharedCode::CoreSlot& slot = shared->cores[conf.processor_id];
        for (;;) {
            slot.in_code.store(true, std::memory_order_seq_cst);
            if (!shared->clear_pending.load(std::memory_order_seq_cst)) {
                break;
            }
            slot.in_code.store(false, std::memory_order_seq_cst);
            PerformPendingClear();
        }
        SyncSharedCode();
        if (shared_verify) [[unlikely]] {
            VerifyCoreRefs();
        }
    }

    void LeaveCode() {
        SharedCode::CoreSlot& slot = shared->cores[conf.processor_id];
        slot.in_code.store(false, std::memory_order_release);
        slot.generation.fetch_add(1, std::memory_order_acq_rel);
    }

    /// Asks for a whole-cache clear: every core of the process stops (the kernel's halt), the
    /// first one out with all the others out performs it.
    void RequestSharedClear() {
        shared->clear_pending.store(true, std::memory_order_seq_cst);
        std::scoped_lock lock{engine->members_mutex};
        for (void* const member : engine->members) {
            if (member != nullptr) {
                static_cast<Impl*>(member)->HaltExecution(HaltReason::CacheInvalidation);
            }
        }
    }

    /// The halt's work, outside generated code: a pending clear, this core's ranges, catch-up.
    void PerformSharedInvalidation() {
        ClearHalt(HaltReason::CacheInvalidation);
        if (shared->clear_pending.load(std::memory_order_seq_cst)) {
            PerformPendingClear();
        }
        boost::icl::interval_set<u64> ranges;
        {
            std::unique_lock lock{invalidation_mutex};
            invalidate_entire_cache = false;
            ranges = std::move(invalid_cache_ranges);
            invalid_cache_ranges.clear();
        }
        if (!ranges.empty()) {
            // The first core to serve a range removes its blocks for all; the others find none.
            std::scoped_lock lock{shared->compile_mutex};
            emitter.InvalidateCacheRanges(ranges);
        }
        SyncSharedCode();
    }

    /// The whole-cache clear, once every other core is out of generated code (they were all
    /// halted; none holds the compile lock while waiting to get out). This core is out too.
    void PerformPendingClear() {
        const size_t own = conf.processor_id;
        for (;;) {
            bool inside = false;
            for (size_t c = 0; c < SharedCode::max_cores; ++c) {
                if (c != own && shared->cores[c].in_code.load(std::memory_order_seq_cst)) {
                    inside = true;
                    break;
                }
            }
            if (!inside) {
                break;
            }
            if (!shared->clear_pending.load(std::memory_order_seq_cst)) {
                return;  // another core did it
            }
            std::this_thread::yield();
        }
        std::scoped_lock lock{shared->compile_mutex};
        if (!shared->clear_pending.load(std::memory_order_seq_cst)) {
            return;
        }
        code_dump.Flush(); // FRITCHII_JIT_DUMP: before the code goes (this core's records)
        LOG_INFO(Core_ARM, "JIT shared code (FRITCHII_JIT_SHARED) whole-cache clear by core {}: {} blocks, near {} MiB used, far {} MiB used",
                 conf.processor_id, emitter.block_descriptors.size(), block_of_code.NearCodeUsed() >> 20, block_of_code.FarCodeUsed() >> 20);
        if (conf.counters) {
            ++conf.counters->cache_clears;
        }
        block_of_code.ClearCache();
        emitter.ClearCache();  // the map and the removal log too
        shared->full_clears.fetch_add(1, std::memory_order_relaxed);
        shared->clear_epoch.fetch_add(1, std::memory_order_seq_cst);
        shared->clear_pending.store(false, std::memory_order_seq_cst);
    }

    /// This core's dispatch table, indirect slots and return stack buffer drop what the shared
    /// cache removed (one entry per removed block) or, after a whole-cache clear, everything.
    void SyncSharedCode() {
        const u64 epoch = shared->clear_epoch.load(std::memory_order_acquire);
        if (epoch != seen_clear_epoch) {
            seen_clear_epoch = epoch;
            {
                std::scoped_lock lock{shared->removal_mutex};
                seen_removal = shared->removal_seq.load(std::memory_order_relaxed);
            }
            ResetOwnTables(true);
            return;
        }
        if (shared->removal_seq.load(std::memory_order_acquire) == seen_removal) {
            return;
        }
        constexpr size_t max_selective_removals = 8;
        std::array<u64, max_selective_removals> removed_keys{};
        size_t removed_count = 0;
        bool selective = false;
        {
            std::scoped_lock lock{shared->removal_mutex};
            const u64 seq = shared->removal_seq.load(std::memory_order_relaxed);
            if (seen_removal < shared->removal_base) {
                seen_removal = seq;
                ResetOwnTables(true);  // fell behind the log
                return;
            }
            const u64 delta = seq - seen_removal;
            if (conf.HasOptimization(OptimizationFlag::FastDispatch) && emitter.fast_dispatch_table_lookup_in != nullptr) {
                for (u64 i = seen_removal; i < seq; ++i) {
                    const u64 key = shared->removal_log[size_t(i - shared->removal_base)];
                    A64EmitX64::FastDispatchEntry& entry = (*emitter.fast_dispatch_table_lookup_in)(key, dispatch_table);
                    if (entry.location_descriptor == key) {
                        entry = {};
                    }
                    if (removed_count < max_selective_removals) {
                        removed_keys[removed_count++] = key;
                    }
                }
            } else if (delta <= max_selective_removals) {
                for (u64 i = seen_removal; i < seq; ++i) {
                    removed_keys[removed_count++] = shared->removal_log[size_t(i - shared->removal_base)];
                }
            }
            selective = delta <= max_selective_removals;
            seen_removal = seq;
        }
        if (selective) {
            // Selective invalidation keeps 99.9%+ of cached indirect branch targets and RSB
            // entries warm across isolated fastmem faults instead of wiping 1 MiB of slots.
            if (indirect_slots != nullptr) {
                const size_t used = (std::min)(emitter.indirect_slots_used.load(std::memory_order_relaxed),
                                               A64EmitX64::indirect_slot_capacity);
                for (size_t s = 0; s < used; ++s) {
                    const u64 slot_desc = indirect_slots[s].location_descriptor;
                    for (size_t r = 0; r < removed_count; ++r) {
                        if (slot_desc == removed_keys[r]) {
                            indirect_slots[s] = {};
                            break;
                        }
                    }
                }
            }
            for (size_t i = 0; i < A64JitState::RSB_SIZE; ++i) {
                const u64 desc = jit_state.rsb_location_descriptors[i];
                for (size_t r = 0; r < removed_count; ++r) {
                    if (desc == removed_keys[r]) {
                        jit_state.rsb_location_descriptors[i] = 0xFFFFFFFFFFFFFFFFull;
                        jit_state.rsb_codeptrs[i] = 0;
                        break;
                    }
                }
            }
        } else {
            ResetOwnTables(false);
        }
    }

    void ResetOwnTables(bool everything) {
        if (everything && dispatch_table != nullptr && conf.HasOptimization(OptimizationFlag::FastDispatch)) {
            std::fill_n(dispatch_table, emitter.fast_dispatch_table_size, A64EmitX64::FastDispatchEntry{});
        }
        if (indirect_slots != nullptr) {
            const size_t used = everything ? A64EmitX64::indirect_slot_capacity
                                           : (std::min)(emitter.indirect_slots_used.load(std::memory_order_relaxed), A64EmitX64::indirect_slot_capacity);
            std::fill_n(indirect_slots, used, A64EmitX64::FastDispatchEntry{});
        }
        jit_state.ResetRSB();
    }

    /// The block for `descriptor`: the map without a lock, else compiled under the lock (or taken
    /// from the core that compiled it while this one waited).
    CodePtr GetSharedBlock(IR::LocationDescriptor descriptor) {
        const u64 key = descriptor.Value();
        const bool check = shared_verify && (++verify_tick & 63) == 0;
        const u64 removals_before = check ? shared->removal_seq.load(std::memory_order_acquire) : 0;
        if (const CodePtr hit = shared->Find(key, core_bit)) {
            if (check) [[unlikely]] {
                VerifyLookup(descriptor, hit, removals_before);
            }
            return hit;
        }
        // Fritchii hitch attribution: game code compiled (or waited for) while the game waits.
        const ::Common::Nx2Hitch::Scope hitch_timed{::Common::Nx2Hitch::CompileCause(key, conf.processor_id)};
        std::unique_lock lock{shared->compile_mutex, std::try_to_lock};
        if (!lock.owns_lock()) {
            const s64 begin = SteadyNs();
            lock.lock();
            shared->lock_waits.fetch_add(1, std::memory_order_relaxed);
            shared->lock_wait_ns.fetch_add(u64(SteadyNs() - begin), std::memory_order_relaxed);
            if (const auto block = emitter.GetBasicBlock(descriptor)) {
                shared->taken_after_wait.fetch_add(1, std::memory_order_relaxed);
                return block->entrypoint;
            }
        } else if (const auto block = emitter.GetBasicBlock(descriptor)) {
            return block->entrypoint;
        }
        const size_t remaining = block_of_code.SpaceRemaining();
        if (remaining < shared_clear_reserve) {
            // Nearly full: every core stops for a clear soon; compiles go on in the reserve.
            if (!shared->clear_pending.load(std::memory_order_relaxed)) {
                RequestSharedClear();
            }
            if (remaining < 2 * MINIMUM_REMAINING_CODESIZE) {
                // Out of room: leave generated code (the pc is in the state block, the halt bit
                // is set); the clear runs once every core is out, then this block compiles.
                return block_of_code.GetForceReturnFromRunCodeAddress();
            }
        }
        shared->compiling_core_bit = core_bit;
        return CompileBlock(descriptor);
    }

    /// A warm compile (EDEN_NX2_JIT_WARM, FRITCHII_JIT_BLOCK_LIST) into the shared cache: a block
    /// any core compiled is found and skipped, so the list is compiled once for all cores.
    bool PrecompileShared(u64 location) {
        if (shared->clear_pending.load(std::memory_order_seq_cst)) {
            PerformPendingClear();
        }
        EnterCode();
        bool compiled = false;
        if (shared->Find(location, 0) == nullptr) {
            const IR::LocationDescriptor descriptor{location};
            std::scoped_lock lock{shared->compile_mutex};
            if (!emitter.GetBasicBlock(descriptor) && WarmSpaceLeft()) {
                shared->compiling_core_bit = core_bit;
                static_cast<void>(CompileBlock(descriptor));
                compiled = true;
            }
        }
        LeaveCode();
        return compiled;
    }

    /// FRITCHII_JIT_SHARED's line, every 10 s at most (any core's Run).
    void LogSharedCode() {
        std::unique_lock log_lock{engine->log_mutex, std::try_to_lock};
        if (!log_lock.owns_lock()) {
            return;
        }
        const s64 now = SteadyNs();
        const u64 compiled = shared->compiled.load(std::memory_order_relaxed);
        const u64 wait_ns = shared->lock_wait_ns.load(std::memory_order_relaxed);
        const u64 other = shared->other_core_lookups.load(std::memory_order_relaxed);
        if (engine->log_ns == 0 || now - engine->log_ns < 10'000'000'000LL) {
            if (engine->log_ns == 0) {
                engine->log_ns = now;
                engine->logged_compiled = compiled;
                engine->logged_lock_wait_ns = wait_ns;
                engine->logged_other_core_lookups = other;
            }
            return;
        }
        size_t near_mib = 0, far_mib = 0, keys = 0;
        {
            std::scoped_lock lock{shared->compile_mutex};
            near_mib = block_of_code.NearCodeUsed() >> 20;
            far_mib = block_of_code.FarCodeUsed() >> 20;
            keys = shared->Entries();
        }
        std::string users;
        size_t last = 1;
        for (size_t n = 1; n < shared->users.size(); ++n) {
            if (shared->users[n].load(std::memory_order_relaxed) != 0) {
                last = n;
            }
        }
        for (size_t n = 1; n <= last; ++n) {
            users += fmt::format("{}{}:{}", n == 1 ? "" : " ", n, s64(shared->users[n].load(std::memory_order_relaxed)));
        }
        const std::string verify = shared_verify
                                       ? fmt::format("; checks {}, mismatches {} (must be 0)",
                                                     shared->verify_checks.load(std::memory_order_relaxed),
                                                     shared->verify_mismatches.load(std::memory_order_relaxed))
                                       : std::string{};
        LOG_INFO(Core_ARM,
                 "JIT shared code (FRITCHII_JIT_SHARED={}) {:.1f} s: {} blocks compiled once (+{}); blocks by cores that "
                 "looked them up {}; first lookups of a block another core had {} (+{}); compile lock: {} waits, "
                 "{:.1f} ms waited (+{:.1f}), {} found compiled by the core waited for; removals {}, whole-cache clears {}; "
                 "near {} MiB, far {} MiB used, map {} keys; patch shape errors {} (must be 0){}",
                 A64EmitX64::SharedMode(), double(now - engine->log_ns) / 1e9, compiled, compiled - engine->logged_compiled,
                 users, other, other - engine->logged_other_core_lookups, shared->lock_waits.load(std::memory_order_relaxed),
                 double(wait_ns) / 1e6, double(wait_ns - engine->logged_lock_wait_ns) / 1e6,
                 shared->taken_after_wait.load(std::memory_order_relaxed), shared->removals.load(std::memory_order_relaxed),
                 shared->full_clears.load(std::memory_order_relaxed), near_mib, far_mib, keys,
                 shared->patch_shape_errors.load(std::memory_order_relaxed), verify);
        engine->log_ns = now;
        engine->logged_compiled = compiled;
        engine->logged_lock_wait_ns = wait_ns;
        engine->logged_other_core_lookups = other;
    }

    /// FRITCHII_JIT_SHARED=2: a lock-free lookup gave what the map under the lock gives (skipped
    /// when the lock is busy or blocks were removed in between).
    void VerifyLookup(IR::LocationDescriptor descriptor, CodePtr hit, u64 removals_before) {
        std::unique_lock lock{shared->compile_mutex, std::try_to_lock};
        if (!lock.owns_lock() || shared->removal_seq.load(std::memory_order_acquire) != removals_before) {
            return;
        }
        shared->verify_checks.fetch_add(1, std::memory_order_relaxed);
        const auto block = emitter.GetBasicBlock(descriptor);
        if (!block || block->entrypoint != hit) {
            const u64 count = shared->verify_mismatches.fetch_add(1, std::memory_order_relaxed);
            if (count < 8) {
                LOG_ERROR(Core_ARM, "JIT shared code (FRITCHII_JIT_SHARED=2) core {}: lookup of {:#x} gave {} but the map holds {}",
                          conf.processor_id, A64::LocationDescriptor{descriptor}.PC(), fmt::ptr(hit),
                          fmt::ptr(block ? block->entrypoint : nullptr));
            }
        }
    }

    /// FRITCHII_JIT_SHARED=2: the state block still holds this core's own objects at every entry
    /// (a wrong one would make generated code call another core's callbacks).
    void VerifyCoreRefs() {
        const auto is = [this](CoreRef which, const void* value) {
            return jit_state.core_refs[size_t(which)] == u64(reinterpret_cast<uintptr_t>(value));
        };
        if ((verify_tick++ & 63) != 0) {
            return;
        }
        shared->verify_checks.fetch_add(1, std::memory_order_relaxed);
        if (!is(CoreRef::Jit, this) || !is(CoreRef::Callbacks, conf.callbacks) || !is(CoreRef::Conf, &conf) ||
            !is(CoreRef::Tpidr, conf.tpidr_el0) || !is(CoreRef::FastDispatchTable, dispatch_table)) {
            const u64 count = shared->verify_mismatches.fetch_add(1, std::memory_order_relaxed);
            if (count < 8) {
                LOG_ERROR(Core_ARM, "JIT shared code (FRITCHII_JIT_SHARED=2) core {}: the state block holds another object", conf.processor_id);
            }
        }
    }

    /// FRITCHII_JIT_SHARED: a clear is asked for with this much left, so compiles keep room
    /// while every core stops.
    static constexpr size_t shared_clear_reserve = size_t{32} << 20;

    /// Fritchii (FRITCHII_JIT_VERIFY=2): a C++ read of a pinned register that returns its poison
    /// means it read the slot while the pins were live and no store had run (a call out that
    /// bypassed the pin hook, or a read from another thread while the core runs). Counted
    /// (must be 0), the first few logged; the value is left as it is, so it shows in a crash.
    u64 CheckPoisonRead(size_t index, u64 value) const {
        if ((pin_poison_mask >> index) & 1) [[unlikely]] {
            if (value == A64JitState::PinPoison(index)) {
                const u64 count = pin_poison_reads.fetch_add(1, std::memory_order_relaxed);
                if (count < 8) {
                    LOG_ERROR(Core_ARM, "JIT pin check (FRITCHII_JIT_VERIFY=2) core {}: C++ read the poison of {} "
                                        "(pc {:#x}, executing {}): a call out without the pin store",
                              conf.processor_id, index == 31 ? std::string("sp") : fmt::format("x{}", index),
                              jit_state.pc, is_executing);
                }
            }
        }
        return value;
    }

    /// Fritchii (FRITCHII_JIT_VERIFY=2): after the JIT returned every pinned slot must hold the
    /// real value (the exit's store). A poison here means a way out without it: counted
    /// (must be 0) and repaired from the shadow, which holds the value.
    void CheckPinsAfterExit() {
        const auto& regs = block_of_code.Pins().regs;
        for (size_t k = 0; k < regs.size(); ++k) {
            const size_t index = size_t(regs[k].poison & 31);
            u64& slot = index == 31 ? jit_state.sp : jit_state.reg[index];
            if (slot == regs[k].poison) [[unlikely]] {
                if (pin_exit_poison++ < 8) {
                    LOG_ERROR(Core_ARM, "JIT pin check (FRITCHII_JIT_VERIFY=2) core {}: {} still poisoned after the "
                                        "JIT returned (pc {:#x}): a way out without the pin store",
                              conf.processor_id, index == 31 ? std::string("sp") : fmt::format("x{}", index),
                              jit_state.pc);
                }
                slot = jit_state.pin_shadow[k];
            }
        }
    }

    /// Fritchii (FRITCHII_JIT_HALT_LATENCY): the time from the earliest halt request of this run
    /// to the return; a run that returned without one (none came, or it came before the run and
    /// its bit made the entry return at once) has no sample.
    void NoteHaltLatency(s64 run_start_ns) {
        const s64 requested = halt_request_ns.exchange(0, std::memory_order_relaxed);
        const s64 now = SteadyNs();
        if (requested != 0) {
            const s64 waited = now - (std::max)(requested, run_start_ns);
            const u64 us = waited <= 0 ? 0 : u64(waited) / 1000;
            size_t bucket = 0;  // [2^(b-1), 2^b) us, bucket 0 = under 1 us
            while (bucket + 1 < halt_buckets.size() && (u64(1) << bucket) <= us) {
                ++bucket;
            }
            ++halt_buckets[bucket];
            ++halt_count;
            if (waited > 1'000'000) {
                ++halt_over_1ms;
                ++halt_over_1ms_total;
            }
            if (waited > halt_longest_ns) {
                halt_longest_ns = waited;
                halt_longest_pc = jit_state.pc;
            }
        }
        if (halt_log_ns == 0) {
            halt_log_ns = now;
            return;
        }
        if (now - halt_log_ns < 10'000'000'000LL) {
            return;
        }
        // Percentiles as the upper edge of the bucket they fall in.
        const auto percentile = [&](u64 part) -> u64 {
            const u64 rank = (halt_count * part + 99) / 100;
            u64 seen = 0;
            for (size_t b = 0; b < halt_buckets.size(); ++b) {
                seen += halt_buckets[b];
                if (seen >= rank && rank != 0) {
                    return u64(1) << b;
                }
            }
            return 0;
        };
        LOG_INFO(Core_ARM, "JIT halt latency (FRITCHII_JIT_HALT_LATENCY=1) core {} {:.1f} s: {} halts from other threads, "
                           "p50 under {} us, p99 under {} us, longest {} us (pc {:#x}), over 1 ms: {} (must be 0), "
                           "{} since boot",
                 conf.processor_id, double(now - halt_log_ns) / 1e9, halt_count, percentile(50), percentile(99),
                 halt_longest_ns / 1000, halt_longest_pc, halt_over_1ms, halt_over_1ms_total);
        halt_buckets.fill(0);
        halt_count = 0;
        halt_over_1ms = 0;
        halt_longest_ns = 0;
        halt_longest_pc = 0;
        halt_log_ns = now;
    }

    const UserConfig conf;
    A64JitState jit_state;
    /// Fritchii (FRITCHII_JIT_SHARED): the code cache, emitter and compile scratch; this JIT's own
    /// in mode 0, the process's in shared mode (shared non-null).
    std::shared_ptr<CodeEngine> engine;
    BlockOfCode& block_of_code;
    A64EmitX64& emitter;
    Optimization::PolyfillOptions& polyfill_options;
    IR::Block& ir_block;
    SharedCode* const shared;
    const bool shared_verify;  ///< FRITCHII_JIT_SHARED=2
    CodeDump code_dump{conf, block_of_code}; // after the code: written out before it goes
    // FRITCHII_JIT_SHARED: this core's dispatch table, indirect slots and counters (the emitter's
    // for the core that made the engine, the own_* ones for the others), and what of the shared
    // cache's clears and removals they have caught up with.
    A64EmitX64::FastDispatchEntry* dispatch_table = nullptr;
    A64EmitX64::FastDispatchEntry* indirect_slots = nullptr;
    u64* indirect_counters = nullptr;
    std::unique_ptr<A64EmitX64::FastDispatchEntry[]> own_dispatch_table;
    std::unique_ptr<A64EmitX64::FastDispatchEntry[]> own_indirect_slots;
    std::array<u64, A64EmitX64::IndirectCounterCount> own_indirect_counters{};
    A64EmitX64::IndirectLog indirect_log;
    u64 core_bit = 0;
    u64 seen_clear_epoch = 0;
    u64 seen_removal = 0;
    u32 runs_since_shared_look = 0;
    u32 verify_tick = 0;
    bool is_executing = false;
    u32 runs_since_counter_look = 0;
    u32 runs_since_pin_look = 0;
    /// Fritchii (FRITCHII_JIT_VERIFY=2): bit i = guest register i (31 = SP) is pinned and
    /// poisoned while generated code runs; 0 in every other mode.
    u64 pin_poison_mask = 0;
    mutable std::atomic<u64> pin_poison_reads{0};
    u64 pin_exit_poison = 0;
    /// Fritchii (FRITCHII_JIT_HALT_LATENCY): the stamp is written by any thread, the rest by
    /// the owner thread in Run.
    bool halt_latency = false;
    std::atomic<s64> halt_request_ns{0};
    std::array<u64, 24> halt_buckets{};
    u64 halt_count = 0;
    u64 halt_over_1ms = 0;
    u64 halt_over_1ms_total = 0;
    s64 halt_longest_ns = 0;
    u64 halt_longest_pc = 0;
    s64 halt_log_ns = 0;
    bool invalidate_entire_cache = false;
    bool warm_refusal_logged = false; ///< Fritchii (FRITCHII_HITCH_SPLIT): logged once per core
    boost::icl::interval_set<u64> invalid_cache_ranges;
    std::mutex invalidation_mutex;
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Jit::Impl>(this, conf)) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

bool Jit::PrecompileBlock(u64 location) {
    return impl->PrecompileBlock(location);
}

void Jit::InvalidateCacheRange(u64 start_address, size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

u64 Jit::GetSP() const {
    return impl->GetSP();
}

void Jit::SetSP(u64 value) {
    impl->SetSP(value);
}

u64 Jit::GetPC() const {
    return impl->GetPC();
}

void Jit::SetPC(u64 value) {
    impl->SetPC(value);
}

u64 Jit::GetRegister(size_t index) const {
    return impl->GetRegister(index);
}

void Jit::SetRegister(size_t index, u64 value) {
    impl->SetRegister(index, value);
}

std::array<u64, 31> Jit::GetRegisters() const {
    return impl->GetRegisters();
}

void Jit::SetRegisters(const std::array<u64, 31>& value) {
    impl->SetRegisters(value);
}

Vector Jit::GetVector(size_t index) const {
    return impl->GetVector(index);
}

void Jit::SetVector(size_t index, Vector value) {
    impl->SetVector(index, value);
}

std::array<Vector, 32> Jit::GetVectors() const {
    return impl->GetVectors();
}

void Jit::SetVectors(const std::array<Vector, 32>& value) {
    impl->SetVectors(value);
}

u32 Jit::GetFpcr() const {
    return impl->GetFpcr();
}

void Jit::SetFpcr(u32 value) {
    impl->SetFpcr(value);
}

u32 Jit::GetFpsr() const {
    return impl->GetFpsr();
}

void Jit::SetFpsr(u32 value) {
    impl->SetFpsr(value);
}

u32 Jit::GetPstate() const {
    return impl->GetPstate();
}

void Jit::SetPstate(u32 value) {
    impl->SetPstate(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

bool Jit::IsExecuting() const {
    return impl->IsExecuting();
}

std::string Jit::Disassemble() const {
    return impl->Disassemble();
}

}  // namespace Dynarmic::A64
