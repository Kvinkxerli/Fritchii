// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

// The memory stage's seam (fable_memstage_step2_spec.md, sub-step 2a).
//
// Every Vulkan command the buffer cache runtime used to record itself is now an entry: the
// runtime function keeps its own work (null replacement, the route of an upload, staging) and
// pushes an entry; MemPlayer::Exec turns the entry into the scheduler or descriptor queue calls
// the function made before. Exec, with the typed plays it decodes to, is the only copy of that
// code.
//
// EDEN_NX2_EXP123 (SEAM, live, read once a frame in TickFrame):
//   0: every entry is executed the moment it is pushed: today's calls at today's moments.
//      Without packet ticks and FRITCHII_MEMSTAGE_SEAM_STATS this is the seam lean: a bracket
//      only counts its depth (no owner claim, no watch, no site counts), and an entry run as
//      pushed goes straight to its typed play (MemPlayer::PlayCopy ...) without being written.
//      Entries pushed as they run take that typed path at every SEAM.
//   1: inside a fill bracket (MemStage::Fill, FillPlay, Ref) the entries wait in the arena and
//      are played where the call site plays them: right after the call, as today. Outside a
//      bracket (a site that has none, another thread) they still run at once.
//   2: as 1, and each draw and dispatch is filled once, before the texture work (sub-step 2b):
//      FillDraw at the rasterizer's draw site (SetUniformBuffersState, the storage binds, the
//      texture buffer unbinds, UpdateGraphicsBuffers, BindHostGeometryBuffers, every stage's
//      BindHostStageBuffers), FillDispatch before ComputePipeline::Configure (the uniform state,
//      the storage binds, the shader hash, the texture buffer unbinds, UpdateComputeBuffers,
//      BindHostComputeBuffers). The packet is held apart from the arena (held_arena), so the
//      texture side's records do not play it; Configure plays its sections at today's places
//      (PlayHeld: the geometry after the texture buffer binds, each stage after Acquire, the
//      dispatch's `before` before the image descriptors) and RestAtEnd plays whatever is left
//      when Configure ends, true, false or by an exception: the cache believes its uploads are
//      on their way. A fill that throws ends without a held packet (EndHeldFill). Descriptors left over then are dropped (the draw is dropped too). Pipelines with
//      texel buffers keep the fill where the calls are (LateFill: the texture buffer binds need
//      the texture cache's views). The descriptor entries take the pipeline's form (the fill runs
//      before Acquire; DescriptorForm).
//
// SEAM 2 and 3 need packet ticks (FRITCHII_MEMSTAGE_TICKS=1 or 2): a held packet's numbers stay
// open until its PlayRest ("Packet ticks" below), so what its early fill stamped is never free
// before the tick its copies are recorded in. Without them TickFrame runs SEAM 1 (logged once,
// counted on the stats line); there is no tick guard.
//
// SEAM = 2's checks:
//   5.5, flushed while held: held packets whose fill and PlayRest are in different command
//        buffers (a flush or finish in between, the clock's syncs included). Expected near 0;
//        the first 16 are logged with their caller. A vertex binding the fill did not see dirty
//        is not bound again in the new command buffer (step 6's full vertex bindings).
//   Texture writes while held: the texture cache writes game memory inside Configure (GC
//        downloads, EXP121's LandCollectDownloads, a flushed download; SwizzleImage and the
//        buffer download's WriteBlockUnsafe call MemReadLog::GuestWrite), after the early fill
//        read. Each held fill keeps the device ranges it read (every MemReadLog line, read log on
//        or not); a write over one is counted and the first 16 logged ("Memory seam: texture
//        write over fill read"). Must be 0 for SEAM 2 to be exact.
//   5.3, the read log (FRITCHII_MEMSTAGE_READ_LOG=N, boot): every Nth draw, dispatch and operation
//        packet logs what its fill read from game memory (the buffer cache's MemReadLog lines:
//        uniforms copied to the stream, buffer uploads with their staged bytes, bound cached
//        ranges, storage and table addresses, compute pointers, the inline page write). At the
//        old read position (the packet's first play; an operation's end) the bytes and values are
//        read again and compared, and each bound range is asked whether the CPU wrote it since.
//        Differences are counted by kind. Game threads write those pages all the time, so the
//        gate is "0 unexplained" (fable_memstage_readlog1.md, step 6 spec 10): every difference
//        gets a class and a place, the first 16 of each class are logged with all of it.
//   Who wrote (the read log's classes). From boot with READ_LOG, every write emulator code makes
//        to game memory, on any thread, goes to Common::WriteRing (fritchii_write_ring.h), one
//        ring of 4,096 {device address, size, site, context, write function, thread name,
//        sequence}, after its bytes landed. The write functions that note: DeviceMemoryManager
//        Write, WriteBlock, WriteBlockUnsafe and its in-place span stores (NoteHostStore);
//        Tegra::MemoryManager Write, WriteBlock(Unsafe/Cached) and WriteWordsCached (through
//        WriteBlockImpl), WriteConstantBufferData, its span stores (CopyBlock, the engines'
//        scoped writes); the GPU write listener (query completions, immediate and semi-flush
//        reports, the 3D stamps, compute and QMD releases); the run-ahead front's release words;
//        Core::Memory's unsafe writes (services storing past the write trap). The sites that name
//        them (SiteScope; innermost names the site, outermost the context; "landing" = GPU bytes
//        written back): buffer downloads (WriteDownloadedMemory in DownloadMemory,
//        DownloadMemoryForRead, DownloadBuffersMemory, PopAsyncBuffers, WriteReadDownloads,
//        Nx2FlushListPages), texture downloads (SwizzleImage in DownloadMemory, PopAsyncFlushes,
//        the garbage collector), EXP121's LandCollectDownloads: landings; the buffer cache's DMA
//        copy, the DMA engine, inline uploads (engine upload, AccelerateInlineToMemory), the 3D
//        constant buffer data, the software blit, the VIC's output: not landings.
//        At a difference (ClassifyDiff):
//        A  an entry newer than the fill over the changed bytes names the writer: "emulator
//           write, download landing" or "emulator write" (by site on the stats line).
//        B  none, on a tracked read (an upload, the inline page write):
//           GPU::ForEachPendingCpuWrite over the changed bytes. A pending game-side mark is
//           "game write, marked" (applied at the next gather); none is "game write, mark already
//           taken" (the store landed after its mark was gathered: core\memory.cpp marks before
//           the store; nothing repairs it). An untracked read (a stream uniform, a value) has no
//           mark to expect and is read again by the next fill: "untracked read changed"
//           (fable_memstage_readlog2.md 4).
//        D  a bound range (the tracker says CPU-written): if this thread applied a gather since
//           the fill (OnCacheInvalidations; FlushRegion gathers first), "gather on the back inside
//           Configure", with the first gather's stack; else the tracker was marked directly by a
//           trapped game store ("game write, marked").
//        C  the place: an upload's changed bytes outside every bound range, uniform and value the
//           packet read are "over-read" (the cache uploaded the page, the draw does not use them);
//           everything else is "inside a range the packet binds" (for a dispatch, whole storage
//           buffers).
//        Unclassified: unmapped, no GPU, or the ring wrapped since the fill.
//        The gate line fails on unexplained (emulator writes that are not landings +
//        unclassified) and on a wrapped ring (the run is not valid); "mark already taken",
//        "untracked read changed" and "changed inside a range the packet binds" are reported
//        apart, not failures.
//        Off (no READ_LOG) every hook is one relaxed load and a branch.
//   The held descriptors' form is compared with the queue's after Acquire. Must be 0.
// The zones "memory fill" (fills at SEAM = 2, with what the cache does inside them) and "memory
// play" (the plays at SEAM = 2) go to EDEN_NX2_GPU_PROFILE: "memory fill" ms/s is what leaves the
// back in step 6.
//
// The order rule: pending entries are played before anything else is recorded. The scheduler's
// record functions, its flush, finish, dispatch and state reset, and the descriptor queues' adds
// and Acquire, call SeamHook::Watch; while a fill is open or entries are pending, OnUse plays them
// first and counts:
//   around (5.1): something recorded or added a descriptor inside a fill without going through
//                 the player or a sync: a runtime path that was not converted. Must be 0.
//   late (5.2):   entries were still pending outside a fill when something else recorded: a call
//                 site misses its play. Must be 0. The picture is right either way.
// A sync (SeamClock::Finish, Wait, SubmitForHostRead, the multi-range bind, the kept words wait)
// plays the pending entries first and is not counted.
//
// Only the thread that owns the arena defers (the first to open a fill while none is pending);
// a fill on another thread at the same moment runs its entries at once. Pending entries are only
// ever played by the thread that pushed them.
//
// SEAM = 3 (sub-step 2c, spec 6.3 and 9.1-9.3): everything SEAM = 2 does (the early fill and its
// checks: the code compares SEAM >= 2), and what the back reads from game memory that the stream
// writes comes in the packet by value.
//   Handle values: the raw handle of every image element a draw or dispatch binds, in the order the
//     back's handle loops read them (per stage: texture buffers, image buffers, textures with
//     tables expanded, images), made by VideoCommon::FillStageHandleValues
//     (texture_handle_values.h). The back's loops take the next value instead of reading. The same
//     RawTextureHandle makes the value on both sides, so a handle program's constant buffer words,
//     storage buffer address and storage words are read by the fill only. The fill always runs;
//     the back's memo paths (TEX_BANK_GEN hits) decide later whether to look: those draws are
//     counted ("skipped by the back"), they now pay for reads the back did not make.
//     Where the fill runs: not in 2b's early fill (yet): the pipeline's Configure fills at the top
//     of its handle work, before the bank decision ("in Configure"), on the same thread as the
//     back's reads.
//   The QMD: KeplerCompute's read of the 256 bytes (kepler_compute.cpp, the native QMD
//     scheduler's read) goes through Qmd(), which opens the dispatch packet, reads into it and
//     returns it. BeginDispatch adopts the open packet. The error path's 512-byte read stays a
//     direct read.
//   Unbound pointer slots: the compute pipeline lookup (PipelineCache::StaleUnboundPointerSlot)
//     reads a pointer from a QMD constant buffer (and a pointer-table entry through it) before the
//     dispatch's fill can run; it reads through
//     ReadPointerSlot(), which keeps the value in the open dispatch packet (null_slot_values).
// The handle values live in their own arrays beside the entry arena (draw_values,
// dispatch_values), not in it: the arena is played as an entry stream. packet.handle_values spans
// them.
//
// Ring B and the facts table (FRITCHII_MEMSTAGE_THREAD = 2, piece 6b; memory_stage_link.cpp has the
// loop). The stage fills the draws it serves on GPUMem (FillRingDraw: the same FillDrawHere, from
// the facts table instead of the pipeline) and the packet goes to ring B instead of the held arena:
// a 4 MiB byte ring, one writer, one reader, packets never wrapped (a pad header at the end), each
// a RingHeader (position, number, the draw's sequence number, the verify hash, the entries a sync
// played already), the DrawPacket and the arena's bytes. That is the split of MemStage (spec 5.5):
// the fill side (arena, the packet being built, ring B's writer) is GPUMem's; the play side (ring
// B's reader, the held arena, played_draw, the player) the back's. The back takes the packet of
// its draw by sequence number (TakeRingDraw: every draw on both threads counts, so the numbers
// match; packets of earlier draws the back never prepared are played whole first) into the held
// arena, and from there plays it exactly as a held packet (PlayHeld, PlayRest). The read log and
// the texture write check stay off for ring packets (they compare one fill with its own play on
// one thread); with the window open (step 7) the read log runs for them in its own form, compared
// at the take ("Step 7's verify" below). The facts table: the back publishes each pipeline's
// facts (by value) under the stage's program key at every draw it prepares; GPUMem copies an
// entry out under its sequence word (each side keeps a cache of its own: step 7). A miss makes
// the draw level (the back prepares it, and publishes).
//
// FRITCHII_MEMSTAGE_VERIFY (boot, bit flags; spec 5.4; not 2b's FRITCHII_MEMSTAGE_READ_LOG): with
// SEAM = 3 the back also makes its own read and compares it with the packet's value: 1 handle
// values, 2 the QMD, 4 null pointer slots.
// Mismatches are counted by kind and the first 16 are logged with address, kind and shader. They
// must read 0: same thread, same instant. Shape mismatches (the back took a different number of
// handle values than the fill made) are counted always.

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <fmt/format.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // RtlCaptureStackBackTrace (the read log's gather callers)
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "common/assert.h"
#include "common/fritchii_write_ring.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/spin_recursive_mutex.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/gpu.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/memory_stage_link.h"
#include "video_core/fritchii_e_split_thread.h"
#include "video_core/fritchii_fill_cuts.h"
#include "video_core/nx2_debug_control.h"
#include "video_core/nx2_fastdraw_count.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2_runahead.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/link_stats.h"
#include "video_core/renderer_vulkan/vk_mem_stage.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/one_follower.h"

namespace Vulkan {
namespace {

/// Step 7: how much of the next ring B packet the back prefetches when it takes one (the header,
/// the draw and the first entries; a packet is usually 10 to 12 lines).
constexpr u64 RingPrefetchBytes = 1024;

void PrefetchLine(const u8* address) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__clang__) || defined(__GNUC__)
    __builtin_prefetch(address, 0, 3);
#else
    static_cast<void>(address);
#endif
}

// The fill depth is MemStage::t_fill_depth (the header: the lean bracket counts it inline).
thread_local u32 t_play_depth = 0;
thread_local MemStage::Site t_site = MemStage::Site::Count;
thread_local char t_thread_marker = 0;
/// FRITCHII_MEMSTAGE_THREAD = 1: while the back serves GPUMem's request, GPUMem's identity.
thread_local uintptr_t t_identity = 0;
/// This thread holds a filled draw or dispatch packet (SEAM = 2) until its plays.
thread_local bool t_holding = false;

/// Per packet, the bytes the read log copies or reads again at the compare.
constexpr u64 READ_LOG_BUDGET = 1ULL << 20;

/// The read log's test D: gathers of game writes applied and FlushRegion calls on this thread
/// (counted only while the read log runs).
thread_local u64 t_gathers = 0;
thread_local u64 t_flush_regions = 0;

namespace Profile = VideoCommon::Nx2Profile;

u64 NowNs() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

std::atomic<MemStage*> g_stage{nullptr};

uintptr_t ThisThread() noexcept {
    return t_identity != 0 ? t_identity : reinterpret_cast<uintptr_t>(&t_thread_marker);
}

using Link = VideoCommon::MemoryStageLink;

// Ring B with one follower (FRITCHII_ONEF step 4): E fills its unlent draws' packets
// (FillRingDraw, as GPUMem does at THREAD = 2) while R takes and plays them, so the fill side's
// rules apply to E: it never plays the held packet (that is R's), and the packet ticks keep two
// marks (E's fill, R's held packet) with ring B's oldest packet between them. Audited for this:
// BeginDraw, ClaimHeld and UpdateUnplayed. The other OnStageThread tests are GPUMem's own checks
// and served calls, which one follower does not use (its E-side work that needs R is lent).
[[nodiscard]] static bool OneFollowerRing() noexcept {
    return VideoCommon::OneFollower::On() &&
           VideoCommon::OneFollower::Step() == VideoCommon::OneFollower::StepLevel::Unlend;
}
/// The thread that fills ring B for another to play: GPUMem, or E with one follower at step 4.
[[nodiscard]] static bool FillSide() noexcept {
    return Link::OnStageThread() ||
           (VideoCommon::OneFollower::OnEngineThread() && OneFollowerRing());
}
/// Ring B has its filler and its player on two threads.
[[nodiscard]] static bool TwoThreadRing() noexcept {
    return Link::RingActive() || OneFollowerRing();
}

struct PlayDepth {
    PlayDepth() noexcept {
        ++t_play_depth;
    }
    ~PlayDepth() {
        --t_play_depth;
    }
};

/// Stats counters: a plain load and store, no locked instruction. A count racing with another
/// thread's may be lost; the stats line does not need more.
void Bump(std::atomic<u64>& counter, u64 amount = 1) noexcept {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

constexpr u32 Align8(u32 value) noexcept {
    return (value + 7U) & ~7U;
}

template <typename T>
const T* PayloadAt(const MemStage::Entry& entry, u32 offset) noexcept {
    return reinterpret_cast<const T*>(reinterpret_cast<const u8*>(&entry) + offset);
}

/// Where an entry executed at once is written: per thread, reused for every push.
// One scratch arena per nesting level of immediate pushes: an Exec that pushed again on the same
// thread would otherwise reset (or grow and free) the arena holding the entry it is still
// reading. No path does that today (review of 2a); this keeps it safe if one appears.
thread_local u32 t_scratch_depth = 0;

MemStage::Arena& Scratch() {
    constexpr u32 Levels = 4;
    thread_local std::array<std::unique_ptr<MemStage::Arena>, Levels> arenas;
    ASSERT_MSG(t_scratch_depth < Levels, "memory seam: immediate pushes nested too deep");
    auto& slot = arenas[t_scratch_depth];
    if (!slot) {
        slot = std::make_unique<MemStage::Arena>(4096);
    }
    return *slot;
}

struct ScratchDepth {
    ScratchDepth() noexcept {
        ++t_scratch_depth;
    }
    ~ScratchDepth() {
        --t_scratch_depth;
    }
};

void HostReadBarrier(vk::CommandBuffer cmdbuf) {
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                           barrier);
}

constexpr VkMemoryBarrier COPY_READ_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
};
constexpr VkMemoryBarrier COPY_WRITE_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
};

constexpr std::array<const char*, static_cast<size_t>(MemStage::Cmd::Count)> CMD_NAMES{
    "upload ahead", "upload batched", "copy",  "pre-copy barrier", "post-copy barrier",
    "clear",        "quad table",     "null init", "index",        "index always",
    "vertex",       "vertices",       "xfb",   "xfbs",             "convert",
    "descriptor",
};
constexpr std::array<const char*, static_cast<size_t>(MemStage::Site::Count)> SITE_NAMES{
    "geometry",      "stage",      "dispatch before", "dispatch indirect", "dispatch after",
    "draw count",    "draw indirect", "list words",   "list args",         "list pages",
    "invalidation",  "download",   "download for read", "frame",           "dma",
    "inline upload", "unmap",      "image copy",      "query",             "fence",
    "draw fill",     "dispatch fill",
};
constexpr std::array<const char*, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)>
    READ_KIND_NAMES{
        "uniform",          "upload",         "bound range",         "storage address",
        "storage size",     "table address",  "compute pointer",     "inline page write",
        "handle program word", "handle program storage base", "handle program storage word",
};
constexpr std::array<const char*, static_cast<size_t>(MemStage::DiffClass::Count)>
    DIFF_CLASS_NAMES{
        "emulator write, download landing", "emulator write",
        "game write, marked",               "game write, mark already taken",
        "untracked read changed",           "gather on the back inside Configure",
        "unclassified",                     "async GPU completion, over-read",
    };
constexpr std::array<const char*, static_cast<size_t>(MemStage::Sync::Count)> SYNC_NAMES{
    "finish", "wait", "submit for host read", "multi-range", "kept words",
};

const char* UseName(SeamHook::Use use) {
    switch (use) {
    case SeamHook::Use::Record:
        return "record";
    case SeamHook::Use::RecordUpload:
        return "record upload";
    case SeamHook::Use::OutsideRenderPass:
        return "outside render pass";
    case SeamHook::Use::RenderPass:
        return "render pass";
    case SeamHook::Use::Flush:
        return "flush";
    case SeamHook::Use::Finish:
        return "finish";
    case SeamHook::Use::DispatchWork:
        return "dispatch work";
    case SeamHook::Use::InvalidateState:
        return "invalidate state";
    case SeamHook::Use::DescriptorAdd:
        return "descriptor add";
    case SeamHook::Use::DescriptorAcquire:
        return "descriptor acquire";
    }
    return "?";
}

/// A fill at one of these sites is a packet of its own (the others belong to a draw or dispatch).
bool IsOpSite(MemStage::Site site) {
    switch (site) {
    case MemStage::Site::Invalidation:
    case MemStage::Site::Download:
    case MemStage::Site::DownloadForRead:
    case MemStage::Site::Frame:
    case MemStage::Site::Dma:
    case MemStage::Site::InlineUpload:
    case MemStage::Site::Unmap:
    case MemStage::Site::ImageCopy:
    case MemStage::Site::Query:
    case MemStage::Site::Fence:
        return true;
    default:
        return false;
    }
}

/// FRITCHII_MEMSTAGE_TICKS (boot): 0 the scheduler's ticks, 1 packet ticks, 2 packet ticks
/// checked against the scheduler's answer.
u32 TicksMode() {
    const char* const value = std::getenv("FRITCHII_MEMSTAGE_TICKS");
    if (value == nullptr || value[0] < '0' || value[0] > '9') {
        return 0;
    }
    return (std::min)(static_cast<u32>(std::atoi(value)), 2U);
}

/// Entries a sync inside a held fill leaves for the packet's plays: descriptors and the command
/// buffer's binding state.
bool WaitsForPlay(MemStage::Cmd kind) {
    switch (kind) {
    case MemStage::Cmd::Descriptors:
    case MemStage::Cmd::BindIndex:
    case MemStage::Cmd::BindIndexAlways:
    case MemStage::Cmd::BindVertex:
    case MemStage::Cmd::BindVertices:
    case MemStage::Cmd::BindXfb:
    case MemStage::Cmd::BindXfbs:
        return true;
    default:
        return false;
    }
}

bool IsRecord(SeamHook::Use use) {
    switch (use) {
    case SeamHook::Use::Record:
    case SeamHook::Use::RecordUpload:
    case SeamHook::Use::OutsideRenderPass:
    case SeamHook::Use::RenderPass:
    case SeamHook::Use::DescriptorAdd:
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// ---------------------------------------------------------------------------------------------
// The hook

bool SeamHook::Holding() noexcept {
    return t_holding;
}

void SeamHook::OnUse(Use use, const std::source_location& location) {
    if (Link::OnStageThread()) [[unlikely]] {
        // The scheduler and the descriptor queues are the back's (spec 5.5): must be 0.
        Link::Note(Link::Check::StageScheduler, location.function_name());
    }
    if (t_holding && (use == Use::Flush || use == Use::Finish)) [[unlikely]] {
        // SEAM = 2: a submit between a held packet's fill and its PlayRest, syncs included.
        if (MemStage* const stage = g_stage.load(std::memory_order_acquire)) {
            stage->NoteHeldFlush(use, location);
        }
    }
    if (t_play_depth != 0) {
        return;
    }
    MemStage* const stage = g_stage.load(std::memory_order_acquire);
    if (stage == nullptr) {
        return;
    }
    const bool in_fill = MemStage::t_fill_depth != 0;
    if (in_fill) {
        // A use inside a fill that is not the player's: a record is 5.1; a flush, finish or
        // state reset that did not go through the clock is counted beside it.
        stage->Note(use, location, true);
    }
    if (stage->OwnedHere() && stage->pending) {
        if (!in_fill) {
            stage->Note(use, location, false);
        }
        stage->PlayPending(true);
    }
}

void SeamHook::OnTickEnd(u64 tick) {
    MemStage* const stage = g_stage.load(std::memory_order_acquire);
    if (stage != nullptr && stage->ticks) {
        stage->ticks->Close(tick);
    }
}

// ---------------------------------------------------------------------------------------------
// Arena

MemStage::Arena::Arena(u32 initial_capacity)
    : data{std::make_unique_for_overwrite<u8[]>(initial_capacity)}, capacity{initial_capacity} {}

MemStage::Arena::~Arena() = default;

u32 MemStage::Arena::Reserve(u32 bytes) {
    if (bytes > capacity - cursor) {
        u64 grown = capacity;
        while (grown < u64{cursor} + bytes) {
            grown *= 2;
        }
        ASSERT(grown <= 0x80000000ULL);
        auto larger = std::make_unique_for_overwrite<u8[]>(static_cast<size_t>(grown));
        std::memcpy(larger.get(), data.get(), cursor);
        data = std::move(larger);
        capacity = static_cast<u32>(grown);
    }
    const u32 offset = cursor;
    cursor += bytes;
    return offset;
}

// ---------------------------------------------------------------------------------------------
// Brackets

// The full bracket. The lean one (header) does only the depth: at SEAM 0 nothing defers
// (claim_deferring is SEAM's, so the claim below only made this thread the owner of an arena it
// never writes), no packet ticks number operations, no zone or read log runs (SEAM >= 2), and
// the watch, the site and its counts serve the stats line and the 5.1 checks of the deferring
// modes. Lean is set only with all of that off (TickFrame).
void MemStage::Fill::Open(Site site_, u32 index_) {
    site = site_;
    index = index_;
    previous_site = t_site;
    if (t_fill_depth == 1) {
        const uintptr_t me = ThisThread();
        uintptr_t expected = 0;
        if (stage.owner.load(std::memory_order_relaxed) != me &&
            stage.owner.compare_exchange_strong(expected, me, std::memory_order_acquire)) {
            // The arena is reset when the next packet begins: nothing in it is pending.
            stage.claim_deferring = stage.seam_on.load(std::memory_order_relaxed);
            stage.arena.Reset();
            stage.play_cursor = 0;
            stage.last_entry = ~0U;
        }
        if (stage.seam_value.load(std::memory_order_relaxed) >= 2) {
            // SEAM = 2: the outermost bracket times the fill; an operation packet (not a held
            // draw or dispatch) may log its reads.
            zone.emplace(Profile::Zone::MemoryFill);
            if (stage.read_log_every > 0 && !stage.filling_held && stage.OwnedHere()) {
                reads = stage.StartReadLog(stage.op_log, ++stage.op_fills);
            }
        }
        if (stage.ticks && IsOpSite(site)) {
            u64 number;
            // FRITCHII_ONEF_OP_MARK (boot, default 1): with one follower, E's operation opens its
            // number under the close's lock with E's mark, as GPUMem's does: R closes ticks while
            // E fills (posted fences), and an unmarked number could be published (its staging
            // freed or reused) before R recorded the copy that reads it (onef-audit finding 1).
            static const bool op_mark = [] {
                const char* value = std::getenv("FRITCHII_ONEF_OP_MARK");
                return value == nullptr || value[0] != '0';
            }();
            const bool engine_mark = op_mark && VideoCommon::OneFollower::OnEngineThread() &&
                                     FillSide() && TwoThreadRing();
            if ((stage.ticks->PerThread() && Link::OnStageThread()) || engine_mark) {
                // Step 7, stamps per thread: an operation on GPUMem (a served call) opens its
                // number under the close's lock with GPUMem's mark, as a draw does, and its
                // stamps carry it until the bracket ends.
                number = stage.ticks->BeginStagePacket();
                stage.stage_op_from = number;
                stage_op = true;
                SeamTicks::SetStamp(number, true);
            } else {
                number = stage.ticks->BeginPacket();
                // No GPUMem mark covers it: not a stage stamp, so a leftover one is checked
                // against `published` if per-thread stamps turn on before the next open.
                SeamTicks::SetStamp(number, false);
            }
            stage.op_number.store(number, std::memory_order_relaxed);
        }
    }
    begin = stage.OwnedHere() ? stage.arena.Size() : 0;
    t_site = site;
    // One follower (fable_b 14.4 cut 3): the watch makes scheduler uses play pending entries.
    // E never uses the scheduler (its uses are lent, "owner on E" counts one that is not), and
    // R's uses do not play E's arena (not its owner): E's brackets leave the shared count alone.
    if (!VideoCommon::OneFollower::OnEngineThread()) {
        SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
    }
    Bump(stage.counters.site_fills[static_cast<size_t>(site)]);
}

void MemStage::Fill::Close() {
    if (!VideoCommon::OneFollower::OnEngineThread()) {
        SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed); // as Open
    }
    if (stage.OwnedHere()) {
        stage.CloseSection(site, index, begin);
    }
    if (reads) {
        // An operation packet is played where it is filled: its old read position is here.
        stage.StopReadLog(stage.op_log, false);
        stage.CompareReads(stage.op_log, SITE_NAMES[static_cast<size_t>(site)], stage.op_fills, 0,
                           0);
    }
    t_site = previous_site;
    if (--t_fill_depth == 0 && stage.OwnedHere() && !stage.pending) {
        stage.owner.store(0, std::memory_order_release);
    }
    if (stage_op) {
        // GPUMem's operation is over: its pending entries (played next, FillPlay) keep their own
        // mark; GPUMem stamps with no packet of its own until the next.
        stage.stage_op_from = 0;
        SeamTicks::SetStamp(0, true);
        stage.UpdateUnplayed();
    }
}

MemStage::SyncScope::SyncScope(MemStage* stage, Sync kind) {
    if (stage != nullptr) {
        stage->EnterSync(kind);
    }
    ++t_play_depth;
}

MemStage::SyncScope::~SyncScope() {
    --t_play_depth;
}

// ---------------------------------------------------------------------------------------------
// MemStage

MemStage::MemStage(MemPlayer& player_)
    : player{player_}, arena{256 * 1024}, held_arena{256 * 1024},
      ring_b{std::make_unique_for_overwrite<u8[]>(RingBytes)},
      facts_table{std::make_unique<FactsSlot[]>(FactsSlots)},
      published_memo{std::make_unique<PublishedMemo[]>(MemoSlots)},
      stage_facts_cache{std::make_unique<StageFactsSlot[]>(StageFactsSlots)} {
    player.stage = this;
    if (const char* const value = std::getenv("FRITCHII_MEMSTAGE_READ_LOG")) {
        read_log_every = (std::max)(0, std::atoi(value));
    }
    if (read_log_every > 0) {
        // The read log's classes: every emulator write to game memory goes to the ring from now.
        Common::WriteRing::g_on.store(true, std::memory_order_relaxed);
        ring_owner = true;
        // Step 7's verify: the cache's CPU-written marks (NoteCpuMark).
        cpu_marks = std::make_unique<CpuMark[]>(CpuMarkSlots);
    }
    if (const u32 mode = TicksMode(); mode != 0) {
        ticks = std::make_unique<SeamTicks>(player.scheduler.GetMasterSemaphore(), mode == 2);
        // Ring B's packets are unplayed too: a close asks for the oldest itself (THREAD = 2). It
        // reads the back's cursor: every close runs on the recording thread, the back (GPUMem's
        // syncs are served there), except Serve's fallback at shutdown, which runs on GPUMem only
        // once the back has left its loop and touches the cursor no more.
        ticks->SetRingOldest(
            [](void* context) { return static_cast<MemStage*>(context)->RingOldestNumber(); },
            this);
        LOG_INFO(Render_Vulkan, "Memory seam: packet ticks (FRITCHII_MEMSTAGE_TICKS={}){}", mode,
                 mode == 2 ? ", checked against the scheduler's ticks" : "");
    }
    g_stage.store(this, std::memory_order_release);
    if (ticks) {
        SeamHook::g_tick_watch.store(true, std::memory_order_release);
    }
}

MemStage::~MemStage() {
    MemStage* expected = this;
    if (g_stage.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel) && ticks) {
        SeamHook::g_tick_watch.store(false, std::memory_order_release);
    }
    if (pending) {
        ClearPendingWatch();
    }
    if (held != HeldKind::None) {
        SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
    }
    VideoCommon::MemReadLog::g_on.store(false, std::memory_order_relaxed);
    VideoCommon::MemReadLog::g_held.store(false, std::memory_order_relaxed);
    if (ring_owner) {
        Common::WriteRing::g_on.store(false, std::memory_order_relaxed);
    }
}

void MemStage::Attach(Cache& cache_) noexcept {
    cache = &cache_;
}

void MemStage::AttachGpu(Tegra::GPU& gpu_) noexcept {
    gpu = &gpu_;
}

bool MemStage::OwnedHere() const noexcept {
    return owner.load(std::memory_order_relaxed) == ThisThread();
}

bool MemStage::Deferring() const noexcept {
    return t_fill_depth != 0 && t_play_depth == 0 && OwnedHere() && claim_deferring;
}

void MemStage::SetPending() {
    if (!pending) {
        pending = true;
        pending_site = t_site;
        if (ticks) {
            // The caller's packet (step 7, stamps per thread; `open` otherwise, as before).
            pending_from = ticks->Mine();
            // FRITCHII_ONEF_OP_MARK: E's operation stamped its staging from its own number on;
            // its pending entries keep that number (Mine() may be later once E's mark clears).
            if (stage_op_from != 0 && stage_op_from < pending_from && FillSide()) {
                pending_from = stage_op_from;
            }
            UpdateUnplayed();
        }
        if (OneFollowerFill()) {
            pending_engine = true;
            SeamHook::g_watch_engine.value.store(1, std::memory_order_relaxed);
        } else {
            SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void MemStage::PlayPending([[maybe_unused]] bool late) {
    if (!pending) {
        return;
    }
    if (Link::InCall() || OneFollowerFill()) [[unlikely]] {
        // One follower, fable_b 13.1: an op site posted (E recorded, R only plays): the pending
        // entries go by value in a ring B op packet, E goes on.
        if (!Link::InCall() && PostPendingOp()) {
            return;
        }
        // GPUMem (FRITCHII_MEMSTAGE_THREAD = 1): a play inside a fill (a sync's, a Ref's) records,
        // so the back runs it, as GPUMem. One follower: E's play goes to R the same way.
        OnBack([this, late] { PlayPending(late); });
        return;
    }
    std::optional<Profile::Scope> zone;
    if (seam_value.load(std::memory_order_relaxed) >= 2) {
        zone.emplace(Profile::Zone::MemoryPlay);
    }
    PlayDepth depth;
    Bump(counters.plays);
    if (filling_held) [[unlikely]] {
        // A sync inside a held fill (SEAM = 2): what the sync needs (copies, uploads, clears,
        // barriers, conversions) runs now; the descriptors (no Acquire yet) and the index,
        // vertex and xfb binds wait for the packet's plays. A bind recorded now would be lost if
        // the texture work flushes before the draw. Delaying a bind past a copy into the same
        // buffer is fine (a bind reads nothing); the copies still come before the draw. Played
        // entries are marked, and the cursor stays at the first entry left.
        u32 first_left = ~0U;
        for (u32 at = play_cursor; at < arena.Size();) {
            Entry& entry = *reinterpret_cast<Entry*>(arena.At(at));
            const u32 here = at;
            at += entry.bytes;
            if ((entry.flags & Played) != 0) {
                continue;
            }
            if (WaitsForPlay(entry.kind)) {
                first_left = (std::min)(first_left, here);
                continue;
            }
            entry.flags |= Played;
            player.Exec(entry);
        }
        play_cursor = first_left != ~0U ? first_left : arena.Size();
    } else {
        while (play_cursor < arena.Size()) {
            const Entry& entry = *reinterpret_cast<const Entry*>(arena.At(play_cursor));
            play_cursor += entry.bytes;
            // Played: only after a held fill that threw (EndHeldFill) left its entries pending.
            if ((entry.flags & Played) == 0) [[likely]] {
                player.Exec(entry);
            }
        }
    }
    pending = false;
    if (ticks) {
        UpdateUnplayed();
    }
    last_entry = ~0U;
    ClearPendingWatch();
}

void MemStage::Play() {
    if (!OwnedHere()) {
        return;
    }
    PlayPending(false);
    if (t_fill_depth == 0) {
        owner.store(0, std::memory_order_release);
    }
}

void MemStage::CloseSection(Site site, u32 index, u32 begin) {
    if (!claim_deferring) {
        return;
    }
    const Span span{begin, arena.Size()};
    switch (site) {
    case Site::Geometry:
        draw.geometry = span;
        break;
    case Site::Stage:
        if (index < draw.stage.size()) {
            draw.stage[index] = span;
        }
        break;
    case Site::DispatchBefore:
        dispatch.before = span;
        break;
    case Site::DispatchAfter:
        dispatch.after = span;
        break;
    case Site::Draw:
    case Site::Dispatch:
        // A held packet's whole fill: its sections are the brackets inside.
        break;
    default:
        if (t_fill_depth == 1) {
            op.number = ticks ? op_number.load(std::memory_order_relaxed) : ++packet_number;
            op.entries = span;
        }
        break;
    }
}

void MemStage::EnterSync(Sync kind) {
    Bump(counters.syncs[static_cast<size_t>(kind)]);
    if (OwnedHere()) {
        PlayPending(false);
    }
}

void MemStage::BeginDraw(u32 repack_mask, u32 repack_dirty) {
    // A held packet ends at its Configure's RestAtEnd; this is only a safety net. (On GPUMem,
    // THREAD = 2, the held packet is the back's; with one follower at step 4, E's is R's.)
    if (!FillSide()) {
        PlayRest();
    }
    Bump(counters.draws);
    RetireDrawValues();
    draw = DrawPacket{};
    if (ticks && FillSide() && TwoThreadRing()) {
        // THREAD = 2: GPUMem's mark is the draw's number from the moment it exists (the back
        // may close a tick meanwhile); EndStageFill clears it however the fill ends.
        draw.number = ticks->BeginStagePacket();
        if (stage_fill_from == 0) {
            stage_fill_from = draw.number;
        }
        SeamTicks::SetStamp(draw.number, true); // its stamps carry its number (stamps per thread)
    } else {
        draw.number = ticks ? ticks->BeginPacket() : ++packet_number;
        // No GPUMem mark covers it (see Fill::Fill): checked against `published` if left over.
        SeamTicks::SetStamp(draw.number, false);
    }
    draw.repack_mask = repack_mask;
    draw.repack_dirty = repack_dirty;
}

void MemStage::BeginDispatch() {
    PlayRest();
    Bump(counters.dispatches);
    if (dispatch_open) {
        // SEAM = 3: the packet opened at the QMD read (or the null pointer slot read).
        dispatch_open = false;
    } else {
        RetireDispatchValues();
        dispatch = DispatchPacket{};
        dispatch.number = ticks ? ticks->BeginPacket() : ++packet_number;
    }
    // Stamps per thread: a dispatch is filled where it is played (the back, or GPUMem by
    // hand-off at THREAD 1, where stamps per thread are off).
    SeamTicks::SetStamp(dispatch.number, false);
}

// ---------------------------------------------------------------------------------------------
// By value (SEAM = 3)

u32 MemStage::VerifyBits() noexcept {
    static const u32 bits = [] {
        const char* const value = std::getenv("FRITCHII_MEMSTAGE_VERIFY");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 0)) : 0U;
    }();
    return bits;
}

void MemStage::CloseHandleValues(bool dispatch_side, bool null_pointer_handles,
                                 bool in_configure) {
    const size_t side = dispatch_side ? 1 : 0;
    const u32 count = static_cast<u32>((dispatch_side ? dispatch_values : draw_values).size());
    // THREAD = 2: a draw from ring B keeps its values' state in its own copy (played_draw).
    DrawPacket& values_draw = held_ring ? played_draw : draw;
    u32& flags = dispatch_side ? dispatch.flags : values_draw.flags;
    flags &= ~(HandleNullPointers | HandleValuesLooked | HandleValuesRetired);
    flags |= HandleValuesFilled | (null_pointer_handles ? HandleNullPointers : 0U);
    (dispatch_side ? dispatch.handle_values : values_draw.handle_values) = Span{0, count};
    Bump(value_counters.fills[side]);
    Bump(value_counters.values[side], count);
    if (in_configure) {
        Bump(value_counters.in_configure[side]);
    }
}

const std::vector<u32>* MemStage::HandleValues(bool dispatch_side,
                                               bool& null_pointer_handles) const noexcept {
    // One follower: a dispatch R holds from ring B has its values in its own copy.
    const bool ring_dispatch = dispatch_side && HeldDispatchFromRing();
    const u32 flags = ring_dispatch ? played_dispatch.flags
                      : dispatch_side ? dispatch.flags
                                      : (held_ring ? played_draw.flags : draw.flags);
    if ((flags & HandleValuesFilled) == 0) {
        return nullptr;
    }
    null_pointer_handles = (flags & HandleNullPointers) != 0;
    return ring_dispatch ? &played_dispatch_values
           : dispatch_side ? &dispatch_values
                           : &draw_values;
}

void MemStage::LookedAtHandleValues(bool dispatch_side, size_t taken) {
    const bool ring_dispatch = dispatch_side && HeldDispatchFromRing();
    u32& flags = ring_dispatch ? played_dispatch.flags
                 : dispatch_side ? dispatch.flags
                                 : (held_ring ? played_draw.flags : draw.flags);
    const size_t count = (ring_dispatch   ? played_dispatch_values
                          : dispatch_side ? dispatch_values
                                          : draw_values)
                             .size();
    if (taken == 0 && count != 0) {
        // The back skipped its handle loop (a TEX_BANK_GEN hit): retired as skipped.
        return;
    }
    flags |= HandleValuesLooked;
    if (taken != count) {
        NoteValueCheck(Value::Shape, false, 0, 0, count, taken);
    }
}

void MemStage::RetireDrawValues() {
    if ((draw.flags & (HandleValuesFilled | HandleValuesLooked | HandleValuesRetired)) ==
        HandleValuesFilled) {
        Bump(value_counters.skipped[0]);
        Bump(value_counters.skipped_values[0], draw_values.size());
    }
    draw.flags |= HandleValuesRetired;
}

void MemStage::RetireDispatchValues() {
    if ((dispatch.flags & (HandleValuesFilled | HandleValuesLooked | HandleValuesRetired)) ==
        HandleValuesFilled) {
        Bump(value_counters.skipped[1]);
        Bump(value_counters.skipped_values[1], dispatch_values.size());
    }
    dispatch.flags |= HandleValuesRetired;
}

void MemStage::OpenDispatchByValue() {
    if (dispatch_open) {
        return;
    }
    RetireDispatchValues();
    dispatch = DispatchPacket{};
    dispatch.number = ticks ? ticks->BeginPacket() : ++packet_number;
    SeamTicks::SetStamp(dispatch.number, false);
    dispatch_open = true;
}

const std::array<u32, 64>& MemStage::Qmd(Tegra::MemoryManager& gpu_memory, GPUVAddr address) {
    if (dispatch_open) {
        // A QMD read always begins a new dispatch (the pipeline lookup's pointer slots come after
        // it). An open packet is one that ran no dispatch: a QMD the scheduler read for a load or
        // that waits on its dependencies, or only a pointer slot read by a lookup that found no
        // pipeline (review of 2c: it was carried into the next dispatch).
        if ((dispatch.flags & QmdByValue) != 0) {
            Bump(value_counters.qmds_unused);
        }
        dispatch_open = false;
    }
    OpenDispatchByValue();
    gpu_memory.ReadBlockUnsafe(address, dispatch.qmd.data(), sizeof(dispatch.qmd));
    dispatch.flags |= QmdByValue;
    Bump(value_counters.qmds);
    if ((VerifyBits() & VerifyQmd) != 0) [[unlikely]] {
        std::array<u32, 64> own;
        gpu_memory.ReadBlockUnsafe(address, own.data(), sizeof(own));
        for (size_t word = 0; word < own.size(); ++word) {
            const bool same = own[word] == dispatch.qmd[word];
            if (!same || word + 1 == own.size()) {
                // One check per QMD; the first word that differs is the one logged. The shader
                // is the program address (QMD words 48-49).
                NoteValueCheck(Value::Qmd, same, address + word * sizeof(u32),
                               (u64{own[49]} << 32) | own[48], dispatch.qmd[word], own[word]);
                break;
            }
        }
    }
    return dispatch.qmd;
}

u64 MemStage::ReadPointerSlot(Tegra::MemoryManager& gpu_memory, GPUVAddr address) {
    OpenDispatchByValue();
    const u64 value = gpu_memory.Read<u64>(address);
    if (dispatch.null_slot_count < dispatch.null_slot_values.size()) {
        dispatch.null_slot_values[dispatch.null_slot_count++] = value;
    } else {
        Bump(value_counters.null_slots_unkept);
    }
    Bump(value_counters.null_slots);
    if ((VerifyBits() & VerifyNullSlots) != 0) [[unlikely]] {
        const u64 own = gpu_memory.Read<u64>(address);
        NoteValueCheck(Value::NullSlot, own == value, address,
                       (u64{dispatch.qmd[49]} << 32) | dispatch.qmd[48], value, own);
    }
    return value;
}

void MemStage::NoteValueCheck(Value kind, bool same, u64 address, u64 shader, u64 packet_value,
                              u64 own_value) {
    const size_t index = static_cast<size_t>(kind);
    if (kind != Value::Shape) {
        Bump(value_counters.checks[index]);
    }
    if (same) {
        return;
    }
    Bump(value_counters.mismatches[index]);
    static constexpr std::array<const char*, static_cast<size_t>(Value::Count)> names{
        "handle value", "QMD word", "null pointer slot", "handle count"};
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Vulkan,
                    "Memory seam by value (5.4): {} differs at {:#x}, shader {:016x}: packet "
                    "{:#x}, the back's own read {:#x}",
                    names[index], address, shader, packet_value, own_value);
    }
}

void MemStage::SnapshotValues(ValueCounts& out) const {
    const auto load = [](const std::atomic<u64>& value) {
        return value.load(std::memory_order_relaxed);
    };
    for (size_t side = 0; side < 2; ++side) {
        out.fills[side] = load(value_counters.fills[side]);
        out.values[side] = load(value_counters.values[side]);
        out.in_configure[side] = load(value_counters.in_configure[side]);
        out.skipped[side] = load(value_counters.skipped[side]);
        out.skipped_values[side] = load(value_counters.skipped_values[side]);
    }
    out.qmds = load(value_counters.qmds);
    out.qmds_unused = load(value_counters.qmds_unused);
    out.null_slots = load(value_counters.null_slots);
    out.null_slots_unkept = load(value_counters.null_slots_unkept);
    for (size_t kind = 0; kind < out.checks.size(); ++kind) {
        out.checks[kind] = load(value_counters.checks[kind]);
        out.mismatches[kind] = load(value_counters.mismatches[kind]);
    }
    out.left_out = load(value_counters.left_out);
}

bool MemStage::ValueLeftOut(Tegra::MemoryManager& gpu_memory,
                            std::span<const std::pair<u64, u64>> ranges) {
    // The back holds a ring packet and GPUMem keeps its writes (the window open, its verify on).
    if (!held_ring || held != HeldKind::Draw || !Link::StageWritesKept()) {
        return false;
    }
    const Link& link = Link::Get();
    bool hit = false;
    for (const auto& [gpu_address, size] : ranges) {
        // Per GPU page: the device range it maps to.
        constexpr u64 Page = 0x1000;
        for (u64 piece = gpu_address; piece < gpu_address + size && !hit;) {
            const u64 piece_end = (std::min)(gpu_address + size, (piece + Page) & ~(Page - 1));
            if (const auto device = gpu_memory.GpuToCpuAddress(piece)) {
                const u64 begin = *device;
                const u64 end = begin + (piece_end - piece);
                link.ForEachStageWriteSince(played_stage_writes,
                                            [&](u64 write_begin, u64 write_end, u64) {
                                                hit = write_begin < end && begin < write_end;
                                                return hit;
                                            });
            }
            piece = piece_end;
        }
        if (hit) {
            break;
        }
    }
    if (hit) {
        Bump(value_counters.checks[static_cast<size_t>(Value::Handle)]);
        Bump(value_counters.left_out);
    }
    return hit;
}

void MemStage::LogValueStats(double seconds, bool first) {
    if (first) {
        SnapshotValues(reported_values);
        return;
    }
    ValueCounts current;
    SnapshotValues(current);
    const ValueCounts& last = reported_values;
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const auto each = [](u64 values, u64 fills) {
        return fills != 0 ? static_cast<double>(values) / static_cast<double>(fills) : 0.0;
    };
    if (ValuesByValue() || current.fills[0] != last.fills[0] || current.qmds != last.qmds) {
        std::array<std::string, 2> sides;
        for (size_t side = 0; side < 2; ++side) {
            const u64 fills = current.fills[side] - last.fills[side];
            const u64 skipped = current.skipped[side] - last.skipped[side];
            sides[side] = fmt::format(
                "{:.0f}/s ({:.1f} values each, {:.0f}/s filled in Configure), skipped by the back "
                "{:.0f}/s ({:.0f} values/s)",
                rate(fills), each(current.values[side] - last.values[side], fills),
                rate(current.in_configure[side] - last.in_configure[side]), rate(skipped),
                rate(current.skipped_values[side] - last.skipped_values[side]));
        }
        const auto delta = [&](size_t kind, bool checks) {
            return checks ? current.checks[kind] - last.checks[kind]
                          : current.mismatches[kind] - last.mismatches[kind];
        };
        LOG_INFO(Render_Vulkan,
                 "  memory seam by value: draws {}; dispatches {}; QMDs {:.0f}/s ({:.0f}/s ran no "
                 "dispatch); null pointer slots {:.0f}/s ({} not kept); verify {:#x}: checks "
                 "handle {} QMD {} slot {}, mismatches (5.4) handle {} QMD {} slot {}, count {}; "
                 "handle values left out as GPUMem's own later writes {}",
                 sides[0], sides[1], rate(current.qmds - last.qmds),
                 rate(current.qmds_unused - last.qmds_unused),
                 rate(current.null_slots - last.null_slots),
                 current.null_slots_unkept - last.null_slots_unkept, VerifyBits(),
                 delta(0, true), delta(1, true), delta(2, true), delta(0, false),
                 delta(1, false), delta(2, false), delta(3, false),
                 current.left_out - last.left_out);
    }
    reported_values = current;
}

// ---------------------------------------------------------------------------------------------
// SEAM = 2: the held draw and dispatch packets

bool MemStage::ClaimHeld() {
    if (t_fill_depth != 0 || t_play_depth != 0) {
        // Inside another fill or a play: not a draw site. Fill where the calls are.
        return false;
    }
    if (!FillSide()) {
        PlayRest();
    }
    const uintptr_t me = ThisThread();
    uintptr_t expected = 0;
    if (owner.load(std::memory_order_relaxed) != me &&
        !owner.compare_exchange_strong(expected, me, std::memory_order_acquire)) {
        // Another thread's fill holds the arena this instant.
        return false;
    }
    if (pending) {
        // Left pending by a site that misses its play (5.2): played before the packet begins.
        Note(SeamHook::Use::Flush, std::source_location::current(), false);
        PlayPending(true);
    }
    arena.Reset();
    play_cursor = 0;
    last_entry = ~0U;
    claim_deferring = true;
    return true;
}

// Packet ticks: the numbers whose entries wait for their play stay open past a tick's end
// (SeamTicks::Close carries from the first). Two sources: the pending entries (as without SEAM 2)
// and a held packet, from its fill to its PlayRest. A sync inside the held fill plays the
// scheduler entries, but the descriptors and later entries still wait; an operation packet
// played inside Configure does not end the held one. Without a held packet this is SetPending's
// and PlayPending's set and clear.
void MemStage::UpdateUnplayed() {
    if (TwoThreadRing()) {
        // THREAD = 2 ("Packet ticks with the stage ahead"): each side writes only its own mark,
        // from its own fields; the close takes the lower one and ring B's oldest packet itself.
        // GPUMem: the draw it fills (from BeginDraw to the push) and its pending entries. The
        // back: the packet it holds, its own fill in a run, its pending entries. (The arena and
        // its pending flag belong to whichever thread owns it: GPUMem's while it fills; the back
        // reads them as GPUMem's only while it serves GPUMem's sync with GPUMem's identity, and
        // then writes GPUMem's mark.)
        const bool owned = OwnedHere();
        if (FillSide() || t_identity != 0) {
            u64 from = stage_fill_from;
            if (stage_op_from != 0 && (from == 0 || stage_op_from < from)) {
                from = stage_op_from;
            }
            if (owned && pending && (from == 0 || pending_from < from)) {
                from = pending_from;
            }
            ticks->SetStageUnplayed(from);
            return;
        }
        u64 from = held != HeldKind::None || (owned && filling_held) ? held_from : 0;
        if (owned && pending && (from == 0 || pending_from < from)) {
            from = pending_from;
        }
        if (from != 0) {
            ticks->SetUnplayed(from);
        } else {
            ticks->ClearUnplayed();
        }
        return;
    }
    u64 from = held != HeldKind::None || filling_held ? held_from : 0;
    // THREAD = 2: the packets in ring B are not played yet either.
    if (const u64 oldest = RingOldestNumber(); oldest != 0 && (from == 0 || oldest < from)) {
        from = oldest;
    }
    if (pending && (from == 0 || pending_from < from)) {
        from = pending_from;
    }
    if (from != 0) {
        ticks->SetUnplayed(from);
    } else {
        ticks->ClearUnplayed();
    }
}

void MemStage::TakeHeld(HeldKind kind) {
    // The arena holding the packet becomes the held one; the next fill starts on an empty arena.
    // Entries a sync played during the fill are behind play_cursor.
    held_arena.Swap(arena);
    held_cursor = play_cursor;
    arena.Reset();
    play_cursor = 0;
    last_entry = ~0U;
    if (pending) {
        pending = false;
        ClearPendingWatch();
    }
    held = kind;
    t_holding = true;
    VideoCommon::MemReadLog::g_held.store(true, std::memory_order_relaxed);
    if (ticks) {
        UpdateUnplayed();
    }
    // Watched while held: a flush before the plays is logged with its caller.
    SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
}

/// Ends a held fill however it ends: TakeHeld ran, or the cache threw inside the fill.
struct MemStage::HeldFillEnd {
    MemStage& stage;
    ~HeldFillEnd() {
        stage.EndHeldFill();
    }
};

void MemStage::EndHeldFill() {
    filling_held = false;
    fill_form = -1;
    t_reads = nullptr;
    // A ring push ended the fill too. ring_pushed is FillRingDraw's answer: read here, not taken
    // (taking it here made every served draw look unfilled to the stage, hence level). A ring fill
    // asks only that: `held` is the back's (it may hold another packet right now).
    if (ring_fill ? ring_pushed : held != HeldKind::None) {
        return;
    }
    // The fill threw before TakeHeld: nothing is held. Its entries stay pending and the next
    // play runs them (the ones a sync played are marked and skipped).
    t_holding = false;
    if (t_log == &held_log) {
        t_log = nullptr;
        held_log.Clear();
    }
    if (ring_fill && stage_log) {
        // A ring fill that made no packet: its log goes back to the pool.
        if (t_log == stage_log.get()) {
            t_log = nullptr;
        }
        RecycleRingLog(std::move(stage_log));
    }
    if (ticks) {
        UpdateUnplayed();
    }
}

const MemStage::DrawPacket* MemStage::FillDrawHere(Cache& cache_, const DrawFacts& facts,
                                                   bool is_indexed) {
    if (seam_value.load(std::memory_order_relaxed) < 2) {
        return nullptr;
    }
    if (facts.has_texel_buffers) {
        // The texture buffer binds take their address, size and format from the texture
        // cache's views (FillImageViews): this draw fills where the calls are.
        draw.flags |= LateFill;
        Bump(counters.late_draws);
        return nullptr;
    }
    if (!ClaimHeld()) {
        Bump(counters.unheld);
        return nullptr;
    }
    const bool timed = Profile::Enabled();
    const u64 start = timed ? NowNs() : 0;
    Bump(counters.held_draws);
    draw.flags |= Held;
    const bool form = player.guest_descriptor_queue.FormFor(facts.uses_descriptor_buffer);
    if (ring_fill) {
        // THREAD = 2: the fill side's own fields (the back holds and plays another packet at the
        // same time with the window open); they go to the back in the ring header.
        fill_hashes = facts.hashes;
        fill_held_form = form;
    } else {
        held_hashes = facts.hashes;
        held_form = form;
        held_flush_seen = false;
        held_reads.clear();
    }
    fill_form = form ? 1 : 0;
    filling_held = true;
    filling_kind = HeldKind::Draw;
    t_holding = true;
    // THREAD = 2: no texture write check and no read log for ring packets (Ring B above).
    t_reads = ring_fill ? nullptr : &held_reads;
    const HeldFillEnd fill_end{*this};
    if (ticks) {
        if (!ring_fill) {
            held_from = draw.number;
        }
        UpdateUnplayed();
    }
    // A ring packet's log is the stage's own (step 7, the window open; compared at the take).
    const bool logging =
        read_log_every > 0 &&
        (ring_fill ? StartRingReadLog(draw.number) : StartReadLog(held_log, draw.number));
    {
        // Today's calls in today's relative order (ConfigureImpl): the uniform state, per stage
        // the storage binds, the texture buffer unbinds, then the geometry and each stage's
        // buffers, each in its own section.
        Fill whole{*this, Site::Draw};
        // fable_stage_levelpoints 15.7: the fill's steps as fine zones inside "memory fill"
        // (profile 2), one at a time; the brackets' own work stays in "memory fill".
        const bool fine = Profile::Fine();
        std::optional<Profile::Scope> step;
        const auto zone = [&](Profile::Zone next) {
            step.reset();
            if (fine) {
                step.emplace(next, true, false);
            }
        };
        if (ring_fill && ring_pre != nullptr) {
            zone(Profile::Zone::FillRanges);
            // THREAD = 2: the buffer half of the stage's collected ranges, where FlushCaching is
            // today (the top of PrepareDraw); what it pushes is played with the packet.
            ring_pre(ring_pre_argument);
        }
        zone(Profile::Zone::FillBindSetup);
        cache_.SetUniformBuffersState(*facts.uniform_masks, facts.uniform_sizes);
        for (u32 stage = 0; stage < 5; ++stage) {
            if ((facts.stages & (1U << stage)) == 0) {
                continue;
            }
            cache_.UnbindGraphicsStorageBuffers(stage);
            cache_.UnbindGraphicsTextureBuffers(stage);
            if (facts.staged != nullptr) {
                const StagedFacts& staged = *facts.staged;
                for (u32 i = 0; i < staged.ssbo_counts[stage]; ++i) {
                    const FactsSsbo& desc = staged.ssbos[stage][i];
                    cache_.BindGraphicsStorageBuffer(stage, i, desc.cbuf_index, desc.cbuf_offset,
                                                     desc.is_written, desc.indirect_offset);
                }
                continue;
            }
            size_t ssbo_index = 0;
            for (const auto& desc : facts.stage_infos[stage].storage_buffers_descriptors) {
                ASSERT(desc.count == 1);
                cache_.BindGraphicsStorageBuffer(stage, ssbo_index, desc.cbuf_index,
                                                 desc.cbuf_offset, desc.is_written,
                                                 desc.indirect_offset);
                ++ssbo_index;
            }
        }
        last_entry = ~0U;
        zone(Profile::Zone::FillGeometry);
        {
            Fill geometry{*this, Site::Geometry};
            cache_.UpdateGraphicsBuffers(is_indexed);
            cache_.BindHostGeometryBuffers(is_indexed);
        }
        zone(Profile::Zone::FillStages);
        for (u32 stage = 0; stage < 5; ++stage) {
            if ((facts.stages & (1U << stage)) == 0) {
                continue;
            }
            // A stage's descriptors never join the previous stage's entry: each is played at
            // its own stage, before that stage's image descriptors.
            last_entry = ~0U;
            Fill section{*this, Site::Stage, stage};
            cache_.BindHostStageBuffers(stage);
        }
        zone(Profile::Zone::FillPacket);
        last_entry = ~0U;
        if (logging) {
            if (ring_fill) {
                StopRingReadLog();
            } else {
                StopReadLog(held_log, true);
            }
        }
        if (ring_fill) {
            // The cache's "uploaded" mark is the back's question in Configure: it goes in the
            // packet (the back records the barrier where it does today).
            if (cache_.any_buffer_uploaded) {
                draw.flags |= PostCopyBarrier;
                cache_.any_buffer_uploaded = false;
            }
            if (ring_list_args) {
                // A list draw (FRITCHII_MEMSTAGE_LIST_DRAWS): the geometry section bound and
                // synchronized its two words (the inputs' indirect parameters); their buffer
                // and offset are the back's question in Configure (ConfigureDraw), answered here
                // as the cache answers it there, after the fill.
                const auto [list_buffer, list_offset] = cache_.GetDrawIndirectBuffer();
                draw.indirect = BufferRef{list_buffer->Handle(), list_offset};
                draw.flags |= ListArgs;
            }
            // The trackers' record goes in the packet: taken before the push.
            fill_tracker_data.clear();
            if (tracker_record.load(std::memory_order_relaxed)) [[unlikely]] {
                if (ring_lean != 0) [[unlikely]] {
                    FillTrackerLean(cache_); // FRITCHII_FILL_BRACKET_LEAN
                } else {
                    FillTrackerRecord(cache_, draw.trackers);
                }
                if (Link::Ahead() != 0) {
                    // Step 7: with the window open the back draws after GPUMem wrote later
                    // constant buffer words: the bytes the trackers read come with the packet.
                    CopyTrackerBytes(draw.trackers, facts.hashes[0]);
                }
            }
            PushRingDraw();
        } else {
            TakeHeld(HeldKind::Draw);
        }
    }
    // The OoT trackers' record (fable_memstage_6b_plan.md 5.1): the bindings and GPU-modified
    // marks they ask the cache for after the fill, taken here, where the fill ends.
    if (!ring_fill && tracker_record.load(std::memory_order_relaxed)) [[unlikely]] {
        FillTrackerRecord(cache_, draw.trackers);
    }
    if (timed) {
        Bump(counters.fill_ns, NowNs() - start);
    }
    return &draw;
}

void MemStage::CopyTrackerBytes(TrackerRecord& record, u64 vertex_program) {
    // What TrackOcarinaCamera and TrackOcarinaActor read of game memory, under the same tests of
    // the record (the trackers' other tests, on the render targets and the registers, are the
    // back's and come after): the camera block of slot 3, the origin in slot 4 and the bones of
    // slot 6 of the actor's draw.
    if (device_memory == nullptr) {
        return;
    }
    const auto copy = [&](size_t slot, u64 address, u32 bytes) {
        const size_t at = fill_tracker_data.size();
        fill_tracker_data.resize(at + bytes);
        device_memory->ReadBlockUnsafe(address, fill_tracker_data.data() + at, bytes);
        record.copied[slot] = bytes;
    };
    const TrackerSlot& camera = record.slots[0];
    const u32 camera_bytes = VideoCommon::OcarinaCameraReadBytes(); // FRITCHII_OOT_CAMERA_GUEST
    if (camera.bound && camera.size >= camera_bytes && !camera.gpu_modified) {
        copy(0, camera.address, camera_bytes);
    }
    if (vertex_program != TrackerRecord::ActorShader) {
        return;
    }
    const TrackerSlot& origin = record.slots[1];
    if (origin.bound && !origin.gpu_modified &&
        origin.size >= TrackerRecord::OriginOffset + TrackerRecord::OriginBytes) {
        copy(1, origin.address + TrackerRecord::OriginOffset, TrackerRecord::OriginBytes);
    }
    const TrackerSlot& bones = record.slots[2];
    if (bones.bound && !bones.gpu_modified && bones.size == TrackerRecord::BonesSlotSize) {
        copy(2, bones.address, TrackerRecord::BonesBytes);
    }
}

const u8* MemStage::TrackerBytes(const TrackerRecord& record, size_t slot) const noexcept {
    if (slot >= record.copied.size() || record.copied[slot] == 0) {
        return nullptr;
    }
    size_t at = 0;
    for (size_t i = 0; i < slot; ++i) {
        at += record.copied[i];
    }
    if (at + record.copied[slot] > played_tracker_data.size()) {
        return nullptr;
    }
    return played_tracker_data.data() + at;
}

// FRITCHII_OOT_CAMERA_GUEST: the camera's slot 3 as the game bound it (the draw's shader may not
// read it, or read less than the camera block); the other slots as ForEachBoundBuffer visits them.
std::pair<DAddr, u32> MemStage::TrackerBinding(Cache& cache_, u32 index) {
    if (index == 3 && VideoCommon::OcarinaCameraFromGuest()) {
        return cache_.GuestGraphicsUniform(0, index);
    }
    return cache_.BoundGraphicsUniform(0, index);
}

void MemStage::FillTrackerRecord(Cache& cache_, TrackerRecord& record) {
    for (size_t i = 0; i < record.slots.size(); ++i) {
        const u32 index = TrackerRecord::Slots[i];
        const auto [address, size] = TrackerBinding(cache_, index);
        TrackerSlot& slot = record.slots[i];
        slot.address = address;
        slot.size = size;
        slot.bound = size != 0;
        // The bytes each tracker reads: the camera block of slot 3 (only when it is bound that
        // large), the whole binding of slots 4 and 6.
        const u32 camera_bytes = VideoCommon::OcarinaCameraReadBytes();
        const u64 bytes = index == 3 ? u64{camera_bytes} : u64{size};
        slot.gpu_modified = slot.bound && (index != 3 || size >= camera_bytes) &&
                            cache_.GpuWrittenHere(address, bytes); // with marks ahead on GPUMem
    }
    record.filled = true;
}

void MemStage::FillTrackerSlots(Cache& cache_, TrackerRecord& record, u8 slots) {
    for (size_t i = 0; i < record.slots.size(); ++i) {
        TrackerSlot& slot = record.slots[i];
        if ((slots & (1U << i)) == 0) {
            slot = TrackerSlot{};
            continue;
        }
        const u32 index = TrackerRecord::Slots[i];
        const auto [address, size] = TrackerBinding(cache_, index);
        slot.address = address;
        slot.size = size;
        slot.bound = size != 0;
        // As FillTrackerRecord.
        const u32 camera_bytes = VideoCommon::OcarinaCameraReadBytes();
        const u64 bytes = index == 3 ? u64{camera_bytes} : u64{size};
        slot.gpu_modified = slot.bound && (index != 3 || size >= camera_bytes) &&
                            cache_.GpuWrittenHere(address, bytes);
    }
    record.filled = true;
}

// FRITCHII_FILL_BRACKET_LEAN (fable_b 25.2 cut h): E's FillRing found which slots R can read on
// this draw (ring_lean_slots): R reads slot 0 only for a draw whose first color target is
// B10G11R11_FLOAT (TrackOcarinaCamera), slots 1 and 2 only for an indexed draw of the actor's
// vertex program (TrackOcarinaActor). At 1 the others stay unbound; at 2 and 3 the whole record
// is made. At every value the left-out slots are marked, and R counts reading one (must be 0; at
// 1 it turns the cut off for the session).
void MemStage::FillTrackerLean(Cache& cache_) {
    namespace FillCuts = VideoCommon::FillCuts;
    FillCuts::BracketStats& stats = FillCuts::g_bracket;
    FillCuts::Add(stats.tracker_draws);
    const u8 slots = ring_lean_slots & 7;
    const u8 skipped = static_cast<u8>(~slots & 7);
    FillCuts::Add(stats.lean_slots, static_cast<u64>(std::popcount(skipped)));
    const bool timed = stats.tracker_sample.Take();
    if (ring_lean == FillCuts::On) {
        const u64 start = timed ? FillCuts::Ticks() : 0;
        FillTrackerSlots(cache_, draw.trackers, slots);
        if (timed) [[unlikely]] {
            FillCuts::Add(stats.lean_ticks, FillCuts::Ticks() - start);
        }
        draw.trackers.lean_skipped = skipped;
        return;
    }
    if (timed) [[unlikely]] {
        // The lean record's time on the same draw (into a scratch record), then the whole one.
        TrackerRecord scratch{};
        const u64 lean_start = FillCuts::Ticks();
        FillTrackerSlots(cache_, scratch, slots);
        const u64 full_start = FillCuts::Ticks();
        FillCuts::Add(stats.lean_ticks, full_start - lean_start);
        FillTrackerRecord(cache_, draw.trackers);
        FillCuts::Add(stats.full_ticks, FillCuts::Ticks() - full_start);
    } else {
        FillTrackerRecord(cache_, draw.trackers);
    }
    draw.trackers.lean_skipped = skipped;
}

const MemStage::DispatchPacket* MemStage::FillDispatchHere(Cache& cache_,
                                                           const DispatchFacts& facts) {
    // fable_memstage_compute_brief 3: the dispatch's fill, a fine zone of its own.
    const Profile::FineScope fill_zone{Profile::Zone::DispatchFill};
    if (seam_value.load(std::memory_order_relaxed) < 2) {
        return nullptr;
    }
    if (facts.has_texel_buffers) {
        dispatch.flags |= LateFill;
        Bump(counters.late_dispatches);
        return nullptr;
    }
    if (!ClaimHeld()) {
        Bump(counters.unheld);
        return nullptr;
    }
    const bool timed = Profile::Enabled();
    const u64 start = timed ? NowNs() : 0;
    Bump(counters.held_dispatches);
    dispatch.flags |= Held;
    held_hashes = {facts.shader_hash, 0};
    held_form = player.guest_descriptor_queue.FormFor(facts.uses_descriptor_buffer);
    fill_form = held_form ? 1 : 0;
    filling_held = true;
    filling_kind = HeldKind::Dispatch;
    held_flush_seen = false;
    t_holding = true;
    held_reads.clear();
    t_reads = &held_reads;
    const HeldFillEnd fill_end{*this};
    if (ticks) {
        held_from = dispatch.number;
        UpdateUnplayed();
    }
    const bool logging = read_log_every > 0 && StartReadLog(held_log, dispatch.number);
    {
        // ComputePipeline::Configure's calls in its order: the uniform state, the storage binds
        // and the shader hash, the texture buffer unbinds, then the `before` section.
        Fill whole{*this, Site::Dispatch};
        cache_.SetComputeUniformBufferState(facts.uniform_mask, facts.uniform_sizes);
        cache_.UnbindComputeStorageBuffers();
        cache_.SetComputeShaderHash(facts.shader_hash);
        size_t ssbo_index = 0;
        for (const auto& desc : facts.info->storage_buffers_descriptors) {
            ASSERT(desc.count == 1);
            cache_.BindComputeStorageBuffer(
                ssbo_index, desc.cbuf_index, desc.cbuf_offset, desc.is_written,
                desc.indirect_offset, desc.write_extent,
                Shader::StorageWriteExtraOf(*facts.info, ssbo_index),
                Shader::StorageReadExtentOf(*facts.info, ssbo_index));
            ++ssbo_index;
        }
        cache_.UnbindComputeTextureBuffers();
        last_entry = ~0U;
        {
            Fill before{*this, Site::DispatchBefore};
            cache_.UpdateComputeBuffers();
            cache_.BindHostComputeBuffers();
        }
        last_entry = ~0U;
        if (logging) {
            StopReadLog(held_log, true);
        }
        TakeHeld(HeldKind::Dispatch);
    }
    if (timed) {
        Bump(counters.fill_ns, NowNs() - start);
    }
    return &dispatch;
}

// ---------------------------------------------------------------------------------------------
// FRITCHII_MEMSTAGE_THREAD = 1 (fable_memstage_step6_spec.md 6.1; the thread and its link:
// memory_stage_link.cpp). The draw's and the dispatch's fills run on GPUMem while the back waits.
//
// Before the hand-off the back does what ClaimHeld does first on one thread (plays a held packet
// left over and entries left pending, lets go of the arena), so GPUMem starts with nothing to
// play. The buffer cache's lock, which the back holds at both sites, goes with the fill: GPUMem
// holds it while it runs (the cache's page mirrors change bits only under their holder,
// MirrorSeq), and the back holds it again for every request it serves and when the fill returns.
// The seam's thread-local state (fill and play depth, the site, the held flag, the read log) is
// GPUMem's during the fill; a request the back serves (OnBack) runs with that state and with
// GPUMem's identity as the arena's owner, so a sync inside a held fill plays exactly what it
// plays on one thread. After the hand-off the back holds the packet: its plays and the texture
// write check run on the back as before. When the fill changed the buffer allocator, GPUMem
// publishes its heap budgets (spec 5.3).

bool MemStage::HandsOff() const noexcept {
    return Link::Get().Mode() == 1 && seam_value.load(std::memory_order_relaxed) >= 2 &&
           t_fill_depth == 0 && t_play_depth == 0 && !Link::OnStageThread();
}

template <typename Body>
void MemStage::HandOff(Cache& cache_, u8 kind, Body& body) {
    PlayRest();
    if (OwnedHere()) {
        if (pending) {
            Note(SeamHook::Use::Flush, std::source_location::current(), false);
            PlayPending(true);
        }
        owner.store(0, std::memory_order_release);
    }
    lent_mutex = cache_.mutex.IsHeldByCurrentThread() ? &cache_.mutex : nullptr;
    if (lent_mutex == nullptr) [[unlikely]] {
        // The fills run under the back's buffer cache lock, which goes with them (spec 5.1).
        // Without it GPUMem would fill unlocked: counted and logged (must be 0), and run anyway.
        Link::Note(Link::Check::HandOffUnlocked,
                   kind == static_cast<u8>(Link::HandOff::Draw) ? "draw fill" : "dispatch fill");
    }
    lent_back = std::this_thread::get_id();
    handing_back = ThisThread();
    SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
    SCOPE_EXIT {
        SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
        lent_mutex = nullptr;
        t_holding = held != HeldKind::None;
    };
    auto run = [&] {
        lent_stage = std::this_thread::get_id();
        if (lent_mutex != nullptr) {
            lent_mutex->HandOver(lent_stage);
        }
        SCOPE_EXIT {
            // A fill that threw leaves its entries pending: the back plays them.
            if (OwnedHere()) {
                owner.store(pending ? handing_back : 0, std::memory_order_release);
            }
            t_holding = false;
            player.device.PublishBufferHeapBudgets();
            if (lent_mutex != nullptr) {
                lent_mutex->HandOver(lent_back);
            }
        };
        body();
    };
    Link::Get().Call(static_cast<Link::HandOff>(kind), run);
    if (held_log_on_stage) [[unlikely]] {
        // The read log's "this thread since the fill" counts the holding thread's gathers and
        // FlushRegion calls: the back's, as at FRITCHII_MEMSTAGE_THREAD=0 (syncs it served
        // during the fill count as inside the fill, as they would on one thread).
        held_log_on_stage = false;
        held_log.gathers = t_gathers;
        held_log.flush_regions = t_flush_regions;
    }
}

const MemStage::DrawPacket* MemStage::FillDraw(Cache& cache_, const DrawFacts& facts,
                                               bool is_indexed) {
    if (facts.has_texel_buffers || !HandsOff()) {
        return FillDrawHere(cache_, facts, is_indexed);
    }
    const DrawPacket* packet = nullptr;
    auto body = [&] { packet = FillDrawHere(cache_, facts, is_indexed); };
    HandOff(cache_, static_cast<u8>(Link::HandOff::Draw), body);
    return packet;
}

const MemStage::DispatchPacket* MemStage::FillDispatch(Cache& cache_,
                                                       const DispatchFacts& facts) {
    if (facts.has_texel_buffers || !HandsOff()) {
        return FillDispatchHere(cache_, facts);
    }
    const DispatchPacket* packet = nullptr;
    auto body = [&] { packet = FillDispatchHere(cache_, facts); };
    HandOff(cache_, static_cast<u8>(Link::HandOff::Dispatch), body);
    return packet;
}

// ---------------------------------------------------------------------------------------------
// FRITCHII_MEMSTAGE_THREAD = 2: ring B and the facts table ("Ring B" in the header comment)

bool MemStage::FillRingDraw(Cache& cache_, const DrawFacts& facts, bool is_indexed, u64 position,
                            u64 seq, u64 hash, void (*pre)(void*), void* pre_argument,
                            const std::vector<u32>* values, bool null_pointer_handles,
                            bool list_args) {
    ring_values = values;
    ring_null_pointers = null_pointer_handles;
    ring_list_args = list_args;
    ring_fill = true;
    ring_pushed = false;
    ring_fill_position = position;
    ring_fill_seq = seq;
    ring_fill_hash = hash;
    ring_pre = pre;
    ring_pre_argument = pre_argument;
    // The cache's lock, held by GPUMem, goes to the back with each sync it serves (OnBackImpl).
    lent_mutex = cache_.mutex.IsHeldByCurrentThread() ? &cache_.mutex : nullptr;
    SCOPE_EXIT {
        ring_fill = false;
        ring_pre = nullptr;
        ring_pre_argument = nullptr;
        ring_values = nullptr;
        ring_list_args = false;
        lent_mutex = nullptr;
        ring_lean = 0; // FRITCHII_FILL_BRACKET_LEAN: set by E for this fill only
        ring_lean_slots = 0x7;
    };
    const DrawPacket* const packet = FillDrawHere(cache_, facts, is_indexed);
    return packet != nullptr && std::exchange(ring_pushed, false);
}

void MemStage::EndStageFill() {
    // GPUMem's mark, however its draw's fill ended (a push clears it first; this covers a fill
    // that made no packet, or threw).
    if (stage_fill_from != 0) {
        stage_fill_from = 0;
        SeamTicks::SetStamp(0, true);
        if (ticks) {
            UpdateUnplayed();
        }
    }
}

void MemStage::ResetStageSide(bool end) {
    if (end && pending) {
        // GPUMem's entries left pending by a fill that threw: played on the back, in order,
        // before anything else is recorded (GPUMem is out of its loop).
        owner.store(ThisThread(), std::memory_order_relaxed);
        PlayPending(true);
        owner.store(0, std::memory_order_release);
    }
    stage_fill_from = 0;
    stage_op_from = 0;
    if (ticks) {
        ticks->SetStageUnplayed(0);
        if (end) {
            ticks->SetPerThread(false); // the ring is off: `open` again until the next frame end
        }
        UpdateUnplayed();
    }
}

bool MemStage::RingRoom() const noexcept {
    // Several packets can be in flight in level mode: the draws of one entry, and a deferred
    // draw the stage's copy flushed at a sink-bound write, whose packet waits for the back's
    // next ConsumeSink while the stage goes on with the entries after it. A packet is a few KiB;
    // the margin holds hundreds. Short of it the draw is level (StageFillDraw).
    constexpr u64 Margin = u64{1} << 20;
    // seq_cst: GPUMem's room wait tests it after announcing its sleep (NoteRingFreed's handshake).
    return RingBytes - (ring_b_write - ring_b_freed.load(std::memory_order_seq_cst)) >= Margin;
}

void MemStage::PushRingDraw() {
    const u32 entries = arena.Size();
    const u32 value_count = ring_values != nullptr ? static_cast<u32>(ring_values->size()) : 0U;
    const u32 tracker_bytes = static_cast<u32>(fill_tracker_data.size());
    const u64 bytes = Align8(static_cast<u32>(sizeof(RingHeader) + sizeof(DrawPacket) + entries +
                                              value_count * sizeof(u32) + tracker_bytes));
    u64 index = ring_b_write % RingBytes;
    if (index + bytes > RingBytes) {
        // Packets never wrap: a pad to the end (no header when fewer bytes than one are left).
        if (RingBytes - index >= sizeof(RingHeader)) {
            RingHeader pad{};
            pad.bytes = static_cast<u32>(RingBytes - index);
            pad.kind = 0;
            std::memcpy(ring_b.get() + index, &pad, sizeof(pad));
        }
        ring_b_write += RingBytes - index;
        index = 0;
    }
    if (ring_lean == 0) [[likely]] {
        if (ring_b_write + bytes - ring_b_freed.load(std::memory_order_acquire) > RingBytes)
            [[unlikely]] {
            // RingRoom's margin makes this impossible below 1 MiB a packet.
            Link::Note(Link::Check::RingFull, "a draw packet past ring B's room");
        }
    } else {
        PushRoomLean(bytes); // FRITCHII_FILL_BRACKET_LEAN
    }
    RingHeader header{};
    header.bytes = static_cast<u32>(bytes);
    header.kind = 1;
    header.flags = fill_held_form ? 1 : 0;
    header.part = 0;
    header.position = ring_fill_position;
    header.number = draw.number;
    header.seq = ring_fill_seq;
    header.hash = ring_fill_hash;
    header.held_cursor = play_cursor;
    header.entry_bytes = entries;
    header.hashes = fill_hashes;
    header.value_count = value_count;
    header.value_flags = (ring_values != nullptr ? 1U : 0U) | (ring_null_pointers ? 2U : 0U);
    header.tracker_bytes = tracker_bytes;
    header.stage_writes = Link::Get().StageWriteCount();
    // FRITCHII_MEMSTAGE_MARKS_AHEAD: the fill's marks wait for this packet's play (GPUMem holds
    // the cache's lock in its fill).
    if (cache != nullptr && cache->AheadFillPushed(NextRingPacketIndex())) {
        header.flags |= RingMarkedWrites;
    }
    if (stage_log) {
        // Step 7's verify: the fill's read log, in packet order, before the packet is visible.
        header.flags |= 2;
        std::scoped_lock lock{ring_logs_mutex};
        ring_logs.emplace_back(draw.number, std::move(stage_log));
    }
    u8* const out = ring_b.get() + index;
    std::memcpy(out, &header, sizeof(header));
    std::memcpy(out + sizeof(header), &draw, sizeof(DrawPacket));
    if (entries != 0) {
        std::memcpy(out + sizeof(header) + sizeof(DrawPacket), arena.At(0), entries);
    }
    if (value_count != 0) {
        std::memcpy(out + sizeof(header) + sizeof(DrawPacket) + entries, ring_values->data(),
                    value_count * sizeof(u32));
    }
    if (tracker_bytes != 0) {
        std::memcpy(out + sizeof(header) + sizeof(DrawPacket) + entries + value_count * sizeof(u32),
                    fill_tracker_data.data(), tracker_bytes);
    }
    ring_b_write += bytes;
    ring_b_sent.store(ring_b_write, std::memory_order_release);
    // GPUMem's mark goes only after the packet is in ring B (a close reads the mark first).
    stage_fill_from = 0;
    SeamTicks::SetStamp(0, true); // no packet of GPUMem's own until the next (stamps per thread)
    const u64 sent = ring_packets_sent.load(std::memory_order_relaxed) + 1;
    ring_packets_sent.store(sent, std::memory_order_release);
    if (ring_lean == 0) [[likely]] {
        Link::Get().NotePacketSent(bytes,
                                   sent - ring_packets_done.load(std::memory_order_seq_cst));
    } else {
        PushStatsLean(bytes, sent); // FRITCHII_FILL_BRACKET_LEAN
    }
    // As TakeHeld: the arena is done with; the next fill starts on an empty one.
    arena.Reset();
    play_cursor = 0;
    last_entry = ~0U;
    if (pending) {
        pending = false;
        ClearPendingWatch();
    }
    ring_pushed = true;
    t_holding = false;
    if (ticks) {
        UpdateUnplayed();
    }
}

// FRITCHII_FILL_BRACKET_LEAN (fable_b 25.2 cut h): the room check without ring_b_freed's line
// (the back writes it for every packet it takes). The pusher keeps the last value it read:
// ring_b_freed only grows, so a packet that fits against it fits now, and the check could not
// fire. Only when it does not fit is the line read (and kept), and the check made as before.
// 2 and 3 read it every time and count a fit by the kept value the real one refuses (must be 0).
void MemStage::PushRoomLean(u64 bytes) {
    namespace FillCuts = VideoCommon::FillCuts;
    FillCuts::BracketStats& stats = FillCuts::g_bracket;
    ring_lean_timed = stats.push_sample.Take();
    const bool fits_kept = ring_b_write + bytes - ring_lean_freed <= RingBytes;
    if (ring_lean == FillCuts::On && fits_kept) {
        return;
    }
    const bool timed = ring_lean_timed && ring_lean == FillCuts::Dry && fits_kept;
    const u64 start = timed ? FillCuts::Ticks() : 0;
    const u64 freed = ring_b_freed.load(std::memory_order_acquire);
    const bool full = ring_b_write + bytes - freed > RingBytes;
    if (timed) [[unlikely]] {
        FillCuts::Add(stats.push_ticks, FillCuts::Ticks() - start);
    }
    const u64 kept = std::exchange(ring_lean_freed, freed);
    if (ring_lean != FillCuts::On && fits_kept) {
        FillCuts::Add(stats.push_compare.checked);
        if (full && stats.push_compare.Differ()) [[unlikely]] {
            LOG_ERROR(Render_Vulkan,
                      "Fill bracket lean (FRITCHII_FILL_BRACKET_LEAN={}): a draw packet of {} "
                      "bytes fit by the kept freed count {:#x} but not by the real one {:#x}",
                      ring_lean, bytes, kept, freed);
        }
    }
    if (full) [[unlikely]] {
        // RingRoom's margin makes this impossible below 1 MiB a packet.
        Link::Note(Link::Check::RingFull, "a draw packet past ring B's room");
    }
}

// FRITCHII_FILL_BRACKET_LEAN: at 1 the link counts the packet and its bytes only; the window's
// counters (packets in ring B after the push, read from the back's line) are GPUMem's window
// question (step 7), not E's. 2 and 3 count as before (3 times it).
void MemStage::PushStatsLean(u64 bytes, u64 sent) {
    namespace FillCuts = VideoCommon::FillCuts;
    FillCuts::BracketStats& stats = FillCuts::g_bracket;
    FillCuts::Add(stats.pushes);
    if (ring_lean == FillCuts::On) {
        Link::Get().NotePacketSentLean(bytes);
        return;
    }
    const bool timed = ring_lean_timed && ring_lean == FillCuts::Dry;
    const u64 start = timed ? FillCuts::Ticks() : 0;
    Link::Get().NotePacketSent(bytes, sent - ring_packets_done.load(std::memory_order_seq_cst));
    if (timed) [[unlikely]] {
        FillCuts::Add(stats.push_ticks, FillCuts::Ticks() - start);
    }
}

bool MemStage::PostPendingOp() {
    namespace OneF = VideoCommon::OneFollower;
    static_assert(static_cast<u8>(Site::InlineUpload) == 15 &&
                      static_cast<u8>(Site::Invalidation) == 10 &&
                      static_cast<u8>(Site::ListWords) == 7 && static_cast<u8>(Site::Dma) == 14 &&
                      static_cast<u8>(Site::Query) == 18,
                  "PostOpSite's site numbers");
    if (!OneF::PostOpSite(static_cast<u8>(pending_site))) {
        return false;
    }
    // 13.2 R3: a sync inside a held fill plays part of the arena (its binds wait): R needs E's
    // arena then, lent as before. E's own draw fill makes its own packet (FillRingDraw).
    if (filling_held) {
        OneF::CountOpPacket(OneF::OpCount::LentHeldFill);
        return false;
    }
    if (ring_fill) {
        OneF::CountOpPacket(OneF::OpCount::LentRingFill);
        return false;
    }
    // 13.2 R5: the entries' buffers and staging live until R has played the packet: its number
    // is the one their stamps carry (pending_from), and ring B's oldest packet counts in every
    // close (UpdateUnplayed, RingOldestNumber).
    if (!ticks) {
        OneF::CountOpPacket(OneF::OpCount::LentNoTicks);
        return false;
    }
    const u32 begin = play_cursor;
    const u32 end = arena.Size();
    const u32 entry_bytes = end > begin ? end - begin : 0;
    const u64 bytes = Align8(static_cast<u32>(sizeof(RingHeader) + entry_bytes));
    if (!RingRoom() || bytes > (u64{1} << 20)) {
        OneF::CountOpPacket(OneF::OpCount::LentNoRoom);
        return false;
    }
    const u64 number = pending_from;
    if (entry_bytes != 0) {
        u64 index = ring_b_write % RingBytes;
        if (index + bytes > RingBytes) {
            // Packets never wrap (as PushRingDraw).
            if (RingBytes - index >= sizeof(RingHeader)) {
                RingHeader pad{};
                pad.bytes = static_cast<u32>(RingBytes - index);
                pad.kind = 0;
                std::memcpy(ring_b.get() + index, &pad, sizeof(pad));
            }
            ring_b_write += RingBytes - index;
            index = 0;
        }
        RingHeader header{};
        header.bytes = static_cast<u32>(bytes);
        header.kind = RingKindOp;
        header.part = static_cast<u16>(pending_site);
        header.position = VideoCommon::GpuSplit::Get().BackEntryPosition();
        header.number = number;
        header.seq = 0; // an op takes no draw number (R takes it by `number`, R1)
        header.entry_bytes = entry_bytes;
        u8* const out = ring_b.get() + index;
        std::memcpy(out, &header, sizeof(header));
        std::memcpy(out + sizeof(header), arena.At(begin), entry_bytes);
        // 13.2 R2: the packet is in ring B before the name that sends R to it.
        ring_b_write += bytes;
        ring_b_sent.store(ring_b_write, std::memory_order_release);
        const u64 sent = ring_packets_sent.load(std::memory_order_relaxed) + 1;
        ring_packets_sent.store(sent, std::memory_order_release);
        struct Op {
            MemStage* stage;
            u64 number;
        } const op{this, number};
        OneF::SetLendSite("op packet (posted)");
        OneF::PushOp(
            OneF::LentOp::Copy,
            [](void* raw) {
                Op a;
                std::memcpy(&a, raw, sizeof(a));
                a.stage->PlayRingOp(a.number);
            },
            &op, sizeof(op));
    }
    OneF::CountOpPacket(OneF::OpCount::Posted, entry_bytes);
    // As a play: the entries are R's now.
    play_cursor = end;
    pending = false;
    last_entry = ~0U;
    ClearPendingWatch();
    UpdateUnplayed();
    return true;
}

void MemStage::PlayRingOp(u64 number) {
    // 13.2 R4: a held draw packet's rest first (E's calls' order is the record order).
    PlayRest();
    const RingHeader* const header = PeekRing();
    if (header == nullptr || header->kind != RingKindOp || header->number != number) [[unlikely]] {
        thread_local std::string note;
        note = fmt::format("BufferOp {}: ring B's head is {}", number,
                           header == nullptr ? std::string{"empty"}
                                             : fmt::format("kind {} number {}", header->kind,
                                                           header->number));
        VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::RingPairing,
                                       note.c_str());
        return;
    }
    // 13.3: R's own player state, the packet's stamp (its staging and buffers' number); no fill
    // state of E's, no lock handed over: no entry's Exec enters the buffer cache ("owner on R"
    // counts one that does).
    SeamTicks::SetStamp(number, false);
    {
        PlayDepth depth;
        const u8* const entries = reinterpret_cast<const u8*>(header) + sizeof(RingHeader);
        for (u32 at = 0; at < header->entry_bytes;) {
            const Entry& entry = *reinterpret_cast<const Entry*>(entries + at);
            at += entry.bytes;
            if ((entry.flags & Played) == 0) [[likely]] {
                player.Exec(entry);
            }
        }
    }
    ring_b_read += header->bytes;
    ring_b_freed.store(ring_b_read, std::memory_order_seq_cst);
    ring_packets_done.fetch_add(1, std::memory_order_seq_cst);
    Link::Get().NoteRingFreed();
    if (ticks) {
        UpdateUnplayed();
    }
}

// ---------------------------------------------------------------------------------------------
// Ring B dispatch packets (one follower, FRITCHII_ONEF_OP_DISPATCH; fable_b 13.4 group 6)
//
// E fills a posted dispatch's buffer side as FillDispatchHere does for a held one, with the fill
// side's rules of a ring draw (FillRingDraw): its number is E's mark from the moment it exists
// (BeginStagePacket), a sync inside the fill goes to R with E's fill state (OnRenderImpl), the
// cache's "uploaded" mark goes in the packet (PostCopyBarrier), and the packet goes to ring B
// whole: RingHeader (kind RingKindDispatch), DispatchPacket, the entries, the handle values, the
// view's bytes. E then pushes the dispatch op naming its number. R, at the op, takes it by name
// (TakeRingDispatch) into the held arena and plays it as a held dispatch: Configure's PlayHeld of
// `before`, RestAtEnd for the rest. Neither side touches the other's packet: E fills `dispatch`
// and `dispatch_values`, R plays `played_dispatch` and `played_dispatch_values`.

u64 MemStage::FillRingDispatch(Cache& cache_, const DispatchFacts& facts, u64 position,
                               const std::vector<u32>& values, bool null_pointer_handles,
                               const void* view, u32 view_bytes,
                               std::optional<GPUVAddr> indirect) {
    const Profile::FineScope fill_zone{Profile::Zone::DispatchFill};
    if (seam_value.load(std::memory_order_relaxed) < 2 || facts.has_texel_buffers || !ticks ||
        !FillSide() || !TwoThreadRing() || (view_bytes & 7) != 0 || !RingRoom()) {
        return 0;
    }
    if (!ClaimHeld()) {
        Bump(counters.unheld);
        return 0;
    }
    const bool timed = Profile::Enabled();
    const u64 start = timed ? NowNs() : 0;
    Bump(counters.dispatches);
    Bump(counters.held_dispatches);
    // The packet the QMD read opened (SEAM 3) is this dispatch's, as BeginDispatch adopts it;
    // its number is replaced by E's ring mark.
    RetireDispatchValues();
    dispatch = DispatchPacket{};
    dispatch_open = false;
    dispatch.number = ticks->BeginStagePacket();
    if (stage_fill_from == 0) {
        stage_fill_from = dispatch.number;
    }
    SeamTicks::SetStamp(dispatch.number, true);
    dispatch.flags |= Held;
    const bool form = player.guest_descriptor_queue.FormFor(facts.uses_descriptor_buffer);
    fill_hashes = {facts.shader_hash, 0};
    fill_held_form = form;
    fill_form = form ? 1 : 0;
    filling_held = true;
    filling_kind = HeldKind::Dispatch;
    t_holding = true;
    t_reads = nullptr; // no texture write check, no read log for ring packets (Ring B above)
    // As FillRingDraw: EndHeldFill reads ring_pushed; a sync E's fill lends hands R the lock.
    ring_fill = true;
    ring_pushed = false;
    lent_mutex = cache_.mutex.IsHeldByCurrentThread() ? &cache_.mutex : nullptr;
    SCOPE_EXIT {
        ring_fill = false;
        lent_mutex = nullptr;
    };
    const HeldFillEnd fill_end{*this};
    UpdateUnplayed();
    {
        // FillDispatchHere's calls, in ComputePipeline::Configure's order.
        Fill whole{*this, Site::Dispatch};
        cache_.SetComputeUniformBufferState(facts.uniform_mask, facts.uniform_sizes);
        cache_.UnbindComputeStorageBuffers();
        cache_.SetComputeShaderHash(facts.shader_hash);
        size_t ssbo_index = 0;
        for (const auto& desc : facts.info->storage_buffers_descriptors) {
            ASSERT(desc.count == 1);
            cache_.BindComputeStorageBuffer(
                ssbo_index, desc.cbuf_index, desc.cbuf_offset, desc.is_written,
                desc.indirect_offset, desc.write_extent,
                Shader::StorageWriteExtraOf(*facts.info, ssbo_index),
                Shader::StorageReadExtentOf(*facts.info, ssbo_index));
            ++ssbo_index;
        }
        cache_.UnbindComputeTextureBuffers();
        last_entry = ~0U;
        {
            Fill before{*this, Site::DispatchBefore};
            cache_.UpdateComputeBuffers();
            cache_.BindHostComputeBuffers();
        }
        last_entry = ~0U;
        // Configure's question to the cache after the section, answered here.
        if (cache_.any_buffer_uploaded) {
            dispatch.flags |= PostCopyBarrier;
            cache_.any_buffer_uploaded = false;
        }
        if (indirect) {
            // DispatchCompute's indirect step (after Configure there, so after the barrier
            // question too): the arguments synchronized and their GPU-written state dropped, as
            // its Ref(DispatchIndirect) does. Its entries play at the packet's rest, which on R
            // follows Configure's bind record, as there.
            last_entry = ~0U;
            Fill arguments{*this, Site::DispatchIndirect};
            const auto [buffer, offset] = cache_.ObtainBuffer(
                *indirect, 12, VideoCommon::ObtainBufferSynchronize::FullSynchronize,
                VideoCommon::ObtainBufferOperation::DiscardWrite);
            dispatch.indirect = BufferRef{buffer->Handle(), offset};
        }
        last_entry = ~0U;
        PushRingDispatch(values, null_pointer_handles, view, view_bytes, position);
    }
    if (timed) {
        Bump(counters.fill_ns, NowNs() - start);
    }
    return ring_pushed ? dispatch.number : 0;
}

void MemStage::PushRingDispatch(const std::vector<u32>& values, bool null_pointer_handles,
                                const void* view, u32 view_bytes, u64 position) {
    const u32 entries = arena.Size();
    const u32 value_count = static_cast<u32>(values.size());
    const u64 bytes = Align8(static_cast<u32>(sizeof(RingHeader) + sizeof(DispatchPacket) +
                                              entries + value_count * sizeof(u32) + view_bytes));
    u64 index = ring_b_write % RingBytes;
    if (index + bytes > RingBytes) {
        // Packets never wrap (as PushRingDraw).
        if (RingBytes - index >= sizeof(RingHeader)) {
            RingHeader pad{};
            pad.bytes = static_cast<u32>(RingBytes - index);
            pad.kind = 0;
            std::memcpy(ring_b.get() + index, &pad, sizeof(pad));
        }
        ring_b_write += RingBytes - index;
        index = 0;
    }
    if (ring_b_write + bytes - ring_b_freed.load(std::memory_order_acquire) > RingBytes)
        [[unlikely]] {
        Link::Note(Link::Check::RingFull, "a dispatch packet past ring B's room");
    }
    // The values as the packet's own (R's HandleValues reads them from played_dispatch).
    dispatch.flags &=
        ~(HandleValuesFilled | HandleNullPointers | HandleValuesLooked | HandleValuesRetired);
    dispatch.flags |= HandleValuesFilled | (null_pointer_handles ? HandleNullPointers : 0U);
    dispatch.handle_values = Span{0, value_count};
    Bump(value_counters.fills[1]);
    Bump(value_counters.values[1], value_count);
    RingHeader header{};
    header.bytes = static_cast<u32>(bytes);
    header.kind = RingKindDispatch;
    header.flags = fill_held_form ? 1 : 0;
    header.part = static_cast<u16>(Site::Dispatch);
    header.position = position;
    header.number = dispatch.number;
    header.seq = 0; // a dispatch takes no draw number (R takes it by `number`, R1)
    header.held_cursor = play_cursor;
    header.entry_bytes = entries;
    header.hashes = fill_hashes;
    header.value_count = value_count;
    header.value_flags = 1U | (null_pointer_handles ? 2U : 0U);
    header.tracker_bytes = view_bytes;
    u8* const out = ring_b.get() + index;
    u8* cursor = out;
    std::memcpy(cursor, &header, sizeof(header));
    cursor += sizeof(header);
    std::memcpy(cursor, &dispatch, sizeof(DispatchPacket));
    cursor += sizeof(DispatchPacket);
    if (entries != 0) {
        std::memcpy(cursor, arena.At(0), entries);
        cursor += entries;
    }
    if (value_count != 0) {
        std::memcpy(cursor, values.data(), value_count * sizeof(u32));
        cursor += value_count * sizeof(u32);
    }
    if (view_bytes != 0) {
        std::memcpy(cursor, view, view_bytes);
    }
    // 13.2 R2: the packet is in ring B before the op that names it.
    ring_b_write += bytes;
    ring_b_sent.store(ring_b_write, std::memory_order_release);
    // E's mark goes only after the packet is in ring B (a close reads the mark first).
    stage_fill_from = 0;
    SeamTicks::SetStamp(0, true);
    const u64 sent = ring_packets_sent.load(std::memory_order_relaxed) + 1;
    ring_packets_sent.store(sent, std::memory_order_release);
    // As TakeHeld: the arena is done with; the next fill starts on an empty one.
    arena.Reset();
    play_cursor = 0;
    last_entry = ~0U;
    if (pending) {
        pending = false;
        ClearPendingWatch();
    }
    ring_pushed = true;
    t_holding = false;
    dispatch.flags |= HandleValuesRetired; // E's copy: its values went with the packet
    UpdateUnplayed();
}

const MemStage::DispatchPacket* MemStage::TakeRingDispatch(u64 number, void* view,
                                                          u32 view_bytes) {
    // 13.2 R4: a held packet's rest first (E's calls' order is the record order).
    PlayRest();
    const RingHeader* const header = PeekRing();
    if (header == nullptr || header->kind != RingKindDispatch || header->number != number ||
        header->tracker_bytes != view_bytes || (header->value_flags & 1) == 0) [[unlikely]] {
        thread_local std::string note;
        note = fmt::format("dispatch op {}: ring B's head is {}", number,
                           header == nullptr ? std::string{"empty"}
                                             : fmt::format("kind {} number {} view bytes {}",
                                                           header->kind, header->number,
                                                           header->tracker_bytes));
        VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::RingPairing,
                                       note.c_str());
        return nullptr;
    }
    const RingHeader taken = *header;
    const u8* cursor = reinterpret_cast<const u8*>(header) + sizeof(RingHeader);
    std::memcpy(&played_dispatch, cursor, sizeof(DispatchPacket));
    cursor += sizeof(DispatchPacket);
    held_arena.Reset();
    if (taken.entry_bytes != 0) {
        const u32 at = held_arena.Reserve(taken.entry_bytes);
        std::memcpy(held_arena.At(at), cursor, taken.entry_bytes);
        cursor += taken.entry_bytes;
    }
    const auto* const values = reinterpret_cast<const u32*>(cursor);
    played_dispatch_values.assign(values, values + taken.value_count);
    cursor += size_t{taken.value_count} * sizeof(u32);
    if (view_bytes != 0) {
        std::memcpy(view, cursor, view_bytes);
    }
    played_dispatch.flags &=
        ~(HandleValuesFilled | HandleNullPointers | HandleValuesLooked | HandleValuesRetired);
    played_dispatch.flags |=
        HandleValuesFilled | ((taken.value_flags & 2) != 0 ? HandleNullPointers : 0U);
    played_dispatch.handle_values = Span{0, taken.value_count};
    held_cursor = taken.held_cursor;
    held_form = (taken.flags & 1) != 0;
    held_hashes = taken.hashes;
    // Stamps per thread: R stamps with the packet it plays.
    SeamTicks::SetStamp(number, false);
    ring_b_read += taken.bytes;
    ring_b_freed.store(ring_b_read, std::memory_order_seq_cst);
    held_packet_index = ring_packets_done.fetch_add(1, std::memory_order_seq_cst) + 1;
    held_marks = false; // E refuses a dispatch while marks run ahead
    Link::Get().NoteRingFreed();
    held = HeldKind::Dispatch;
    held_ring = true;
    held_reads.clear();
    held_flush_seen = false;
    held_log_pending = false;
    t_holding = true;
    VideoCommon::MemReadLog::g_held.store(true, std::memory_order_relaxed);
    SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
    held_from = number;
    UpdateUnplayed();
    return &played_dispatch;
}

const MemStage::RingHeader* MemStage::PeekRing() {
    for (;;) {
        const u64 sent = ring_b_sent.load(std::memory_order_acquire);
        if (ring_b_read == sent) {
            return nullptr;
        }
        const u64 index = ring_b_read % RingBytes;
        if (RingBytes - index < sizeof(RingHeader)) {
            ring_b_read += RingBytes - index;
            ring_b_freed.store(ring_b_read, std::memory_order_release);
            continue;
        }
        const auto* const header = reinterpret_cast<const RingHeader*>(ring_b.get() + index);
        if (header->kind == 0) {
            ring_b_read += header->bytes;
            ring_b_freed.store(ring_b_read, std::memory_order_release);
            continue;
        }
        return header;
    }
}

u64 MemStage::RingOldestNumber() {
    u64 read = ring_b_read;
    const u64 sent = ring_b_sent.load(std::memory_order_acquire);
    while (read != sent) {
        const u64 index = read % RingBytes;
        if (RingBytes - index < sizeof(RingHeader)) {
            read += RingBytes - index;
            continue;
        }
        const auto* const header = reinterpret_cast<const RingHeader*>(ring_b.get() + index);
        if (header->kind == 0) {
            read += header->bytes;
            continue;
        }
        return header->number;
    }
    return 0;
}

void MemStage::TakeRingHeader(const RingHeader& header) {
    // The back holds the packet as a held one: PlayHeld and PlayRest play it as today.
    PlayRest();
    // TEST (EDEN_NX2_GPU_PROFILE=2): the copies out of ring B, up to the tracker bytes.
    std::optional<Profile::FineScope> copy_scope;
    copy_scope.emplace(Profile::Zone::RingTakeCopy);
    const u8* const data = reinterpret_cast<const u8*>(&header);
    held_arena.Reset();
    if (header.entry_bytes != 0) {
        const u32 at = held_arena.Reserve(header.entry_bytes);
        std::memcpy(held_arena.At(at), data + sizeof(RingHeader) + sizeof(DrawPacket),
                    header.entry_bytes);
    }
    std::memcpy(&played_draw, data + sizeof(RingHeader), sizeof(DrawPacket));
    // SEAM 3: the raw handles the stage read at its fill, at its position in the stream; the
    // back's handle loops take them (HandleValues) instead of reading game memory, which the
    // stage may already have written past this draw.
    played_draw.flags &=
        ~(HandleValuesFilled | HandleNullPointers | HandleValuesLooked | HandleValuesRetired);
    if ((header.value_flags & 1) != 0) {
        const auto* const values = reinterpret_cast<const u32*>(
            data + sizeof(RingHeader) + sizeof(DrawPacket) + header.entry_bytes);
        draw_values.assign(values, values + header.value_count);
        played_draw.flags |=
            HandleValuesFilled | ((header.value_flags & 2) != 0 ? HandleNullPointers : 0U);
        played_draw.handle_values = Span{0, header.value_count};
        Bump(value_counters.fills[0]);
        Bump(value_counters.values[0], header.value_count);
    }
    // Step 7: the trackers' bytes the fill copied (TrackerBytes).
    const u8* const tracker_data = data + sizeof(RingHeader) + sizeof(DrawPacket) +
                                   header.entry_bytes + size_t{header.value_count} * sizeof(u32);
    played_tracker_data.assign(tracker_data, tracker_data + header.tracker_bytes);
    copy_scope.reset();
    held_cursor = header.held_cursor;
    held_form = (header.flags & 1) != 0;
    held_hashes = header.hashes;
    played_stage_writes = header.stage_writes;
    const bool logged = (header.flags & 2) != 0;
    const bool marks = (header.flags & RingMarkedWrites) != 0;
    const u64 number = header.number;
    // Stamps per thread: the back stamps with the packet it plays, and after it with the last.
    SeamTicks::SetStamp(number, false);
    ring_b_read += header.bytes;
    ring_b_freed.store(ring_b_read, std::memory_order_seq_cst);
    if (Link::Ahead() != 0) {
        // Step 7: the next packet is usually committed already (the window keeps several
        // ahead); GPUMem's core wrote its lines microseconds ago. They are fetched while this
        // draw runs, not at the next take. Only committed bytes, up to the end of the ring.
        const u64 committed = ring_b_sent.load(std::memory_order_acquire) - ring_b_read;
        const u64 index = ring_b_read % RingBytes;
        const u64 bytes = (std::min)({committed, RingBytes - index, RingPrefetchBytes});
        const u8* const next = ring_b.get() + index;
        for (u64 offset = 0; offset < bytes; offset += 64) {
            PrefetchLine(next + offset);
        }
    }
    // Step 7: the window and the room GPUMem may wait for (it is woken if it sleeps).
    // FRITCHII_MEMSTAGE_MARKS_AHEAD: packets are taken or dropped in push order, so the count is
    // this one's push index.
    held_packet_index = ring_packets_done.fetch_add(1, std::memory_order_seq_cst) + 1;
    held_marks = marks;
    Link::Get().NoteRingFreed();
    held = HeldKind::Draw;
    held_ring = true;
    held_reads.clear();
    held_flush_seen = false; // the back's packet now (the ring fill leaves it to the back)
    held_log_pending = false;
    t_holding = true;
    VideoCommon::MemReadLog::g_held.store(true, std::memory_order_relaxed);
    SeamHook::g_watch.fetch_add(1, std::memory_order_relaxed);
    if (ticks) {
        held_from = number;
        UpdateUnplayed();
    }
    if (logged) {
        // Step 7's verify (spec 6.4): the reads of the packet's fill against game memory now, at
        // the back's position (before anything of this draw runs).
        if (std::unique_ptr<ReadLog> log = TakeRingLog(number)) {
            CompareRingReads(*log, number, held_hashes[0], held_hashes[1]);
            RecycleRingLog(std::move(log));
        }
    }
}

const MemStage::DrawPacket* MemStage::TakeRingDraw(u64 seq, u64 back_position, u64 hash,
                                                   bool verify) {
    for (;;) {
        const RingHeader* const header = PeekRing();
        if (header != nullptr && (header->kind == RingKindOp || header->kind == RingKindDispatch))
            [[unlikely]] {
            // 13.2 R1: an op packet is taken by its BufferOp only, a dispatch packet by its op.
            VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::RingPairing,
                                           header->kind == RingKindOp
                                               ? "a draw found an op packet at ring B's head"
                                               : "a draw found a dispatch packet at ring B's head");
            return nullptr;
        }
        if (header == nullptr || header->seq > seq) {
            return nullptr;
        }
        if (header->seq < seq) {
            // A draw the back never prepared (its Draw returned before PrepareDraw), must be 0.
            // The packet is played whole, so the uploads the cache counts on are recorded. Ring B
            // is played in order, and every draw the back fills itself takes this path first (in
            // a run too), so no newer upload of the same bytes has been recorded before it.
            Link::Note(Link::Check::PacketOrphan, "a draw the back did not prepare");
            TakeRingHeader(*header);
            PlayRest();
            continue;
        }
        if (header->position > back_position) {
            // The number matches but the packet belongs to a later entry than the back's: the
            // two sides paired the draws differently (must be 0). Another draw's packet is never
            // played: it is dropped unplayed, and the back fills this draw itself.
            Link::Note(Link::Check::PacketMismatch, "the packet's entry is past the back's");
            DiscardRingHeader(*header);
            return nullptr;
        }
        const u64 stage_hash = header->hash;
        const u64 position = header->position;
        TakeRingHeader(*header);
        if (verify && stage_hash != 0) {
            Link::Get().NoteHash(stage_hash == hash);
            if (stage_hash != hash) {
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                    LOG_WARNING(Render_Vulkan,
                                "Memory stage ring verify (compare b, two threads): draw {} of "
                                "entry {:#x}: the stage's state hash {:#x}, the back's {:#x}",
                                seq, position, stage_hash, hash);
                }
            }
        }
        Link::Get().NotePacketTaken();
        return &played_draw;
    }
}

void MemStage::PlayPostCopyBarrier() {
    // As BufferCacheRuntime::PostCopyBarrier's entry, executed at once by the play side.
    const Entry entry{
        .kind = Cmd::PostCopyBarrier,
        .flags = OutsideRenderPass,
        .count = 0,
        .bytes = static_cast<u32>(sizeof(Entry)),
    };
    PlayDepth depth;
    player.Exec(entry);
}

void MemStage::DiscardRingHeader(const RingHeader& header) {
    if ((header.flags & 2) != 0) {
        // Not played: its read log is not compared.
        RecycleRingLog(TakeRingLog(header.number));
    }
    const bool marked = (header.flags & RingMarkedWrites) != 0; // before the bytes are freed
    ring_b_read += header.bytes;
    ring_b_freed.store(ring_b_read, std::memory_order_seq_cst);
    const u64 index = ring_packets_done.fetch_add(1, std::memory_order_seq_cst) + 1;
    Link::Get().NoteRingFreed();
    if (ticks) {
        UpdateUnplayed();
    }
    // FRITCHII_MEMSTAGE_MARKS_AHEAD: its fill's marks are dropped, not promoted (the back fills
    // the draw itself).
    if (cache != nullptr) {
        cache->NoteRingDiscarded(index, marked);
    }
}

u64 MemStage::DropRingPackets(u64 position) {
    u64 dropped = 0;
    for (const RingHeader* header = PeekRing(); header != nullptr && header->position < position;
         header = PeekRing()) {
        TakeRingHeader(*header);
        PlayRest();
        ++dropped;
    }
    return dropped;
}

namespace {
/// The facts table's key: a hash of the code identity (the address space and each program slot's
/// device address), never 0.
u64 CodeKey(const VideoCommon::MemStageShadow::ProgramId& id) {
    u64 hash = id.space * 0x9E3779B185EBCA87ULL;
    for (const u64 address : id.code) {
        hash = std::rotl(hash ^ address, 27) * 0xC2B2AE3D27D4EB4FULL + 0x165667B19E3779F9ULL;
    }
    return hash != 0 ? hash : 1;
}
} // Anonymous namespace

void MemStage::PublishFacts(const ProgramId& id, const void* source, const DrawFacts& facts,
                            bool dynamic_input, u64 code_epoch) {
    const u64 key = CodeKey(id);
    // The back's memo (step 7): every write of the table goes through here and updates it, so a
    // hit means the table holds this source under these registers already. No lock then. The
    // registers are compared too: one key for two programs is a miss, never the other's entry.
    // The epoch too: a republish after a code write anywhere makes the entry good for it again.
    PublishedMemo& memo = published_memo[FactsHash(key) & (MemoSlots - 1)];
    if (memo.key == key && memo.source == source && memo.epoch >= code_epoch) {
        if (memo.space == id.space && memo.code == id.code) {
            return;
        }
        Link::Note(Link::Check::FactsKeyCollision, "the back's facts memo");
    }
    constexpr size_t Mask = FactsSlots - 1;
    const size_t home = FactsHash(key) & Mask;
    std::scoped_lock lock{facts_mutex};
    FactsSlot* target = &facts_table[home];
    for (size_t probe = 0, index = home; probe < 16; ++probe, index = (index + 1) & Mask) {
        FactsSlot& slot = facts_table[index];
        if (slot.key == key) {
            if (slot.space != id.space || slot.code != id.code) {
                // Other code under the same key: a separate entry.
                Link::Note(Link::Check::FactsKeyCollision, "the facts table (publish)");
                continue;
            }
            if (slot.source == source) {
                // Published already: the same pipeline's facts are good for this epoch too.
                if (slot.epoch < code_epoch) {
                    slot.epoch = code_epoch;
                    facts_generation.fetch_add(1, std::memory_order_release);
                }
                memo = PublishedMemo{key, source, id.space, id.code, slot.epoch};
                return;
            }
            target = &slot;
            break;
        }
        if (slot.key == 0) {
            target = &slot;
            break;
        }
    }
    StagedFacts staged{};
    staged.uniform_masks = *facts.uniform_masks;
    staged.uniform_sizes = *facts.uniform_sizes;
    for (u32 stage = 0; stage < 5; ++stage) {
        if ((facts.stages & (1U << stage)) == 0 || facts.stage_infos == nullptr) {
            continue;
        }
        u8 count = 0;
        for (const auto& desc : facts.stage_infos[stage].storage_buffers_descriptors) {
            if (desc.count != 1 || count >= MaxFactsSsbos) {
                return; // not by value: the stage misses and the draw stays level
            }
            staged.ssbos[stage][count++] = FactsSsbo{
                .cbuf_index = desc.cbuf_index,
                .cbuf_offset = desc.cbuf_offset,
                .indirect_offset = desc.indirect_offset,
                .is_written = desc.is_written,
            };
        }
        staged.ssbo_counts[stage] = count;
    }
    staged.stages = facts.stages;
    staged.has_texel_buffers = facts.has_texel_buffers;
    staged.uses_descriptor_buffer = facts.uses_descriptor_buffer;
    staged.dynamic_vertex_input = dynamic_input;
    staged.hashes = facts.hashes;
    if (facts.stage_infos != nullptr) {
        auto infos = std::make_shared<std::array<Shader::Info, 5>>();
        for (size_t stage = 0; stage < infos->size(); ++stage) {
            (*infos)[stage] = facts.stage_infos[stage];
        }
        staged.infos = std::move(infos);
    }
    if (target->key != 0 &&
        (target->key != key || target->space != id.space || target->code != id.code)) {
        // An entry pushed out of a full probe run: its memo must not say "published" any more
        // (cleared on the key alone: a memo cleared needlessly costs one locked look).
        PublishedMemo& evicted = published_memo[FactsHash(target->key) & (MemoSlots - 1)];
        if (evicted.key == target->key) {
            evicted = PublishedMemo{};
        }
    }
    target->key = key;
    target->source = source;
    target->space = id.space;
    target->code = id.code;
    target->facts = std::move(staged);
    target->epoch = code_epoch;
    memo = PublishedMemo{key, source, id.space, id.code, code_epoch};
    facts_generation.fetch_add(1, std::memory_order_release);
}

bool MemStage::LookupFactsLocked(const ProgramId& id, StagedFacts& out) const {
    const u64 key = CodeKey(id);
    const u64 epoch =
        VideoCommon::MemStageShadow::g_program_code_epoch.load(std::memory_order_acquire);
    constexpr size_t Mask = FactsSlots - 1;
    std::scoped_lock lock{facts_mutex};
    for (size_t probe = 0, index = FactsHash(key) & Mask; probe < 16;
         ++probe, index = (index + 1) & Mask) {
        const FactsSlot& slot = facts_table[index];
        if (slot.key == key && slot.space == id.space && slot.code == id.code) {
            if (slot.epoch < epoch) {
                return false;
            }
            out = slot.facts;
            return true;
        }
        if (slot.key == 0) {
            return false;
        }
    }
    return false;
}

bool MemStage::LookupFacts(const ProgramId& id, StagedFacts& out) const {
    const u64 key = CodeKey(id);
    // An entry published before the latest program code write may describe the old code at these
    // addresses: a miss (MemStageShadow::g_program_code_epoch).
    const u64 epoch =
        VideoCommon::MemStageShadow::g_program_code_epoch.load(std::memory_order_acquire);
    // GPUMem's cache (step 7): valid while no write happened since it was filled. A write right
    // after this test is one the lookup could have missed under the lock too (the back publishes
    // at its own, earlier, position).
    const u64 generation = facts_generation.load(std::memory_order_acquire);
    if (generation != stage_facts_generation) {
        for (size_t i = 0; i < StageFactsSlots; ++i) {
            stage_facts_cache[i].key = 0;
        }
        stage_facts_generation = generation;
    }
    // Entries match on the key and the registers it hashes: one key for two programs is a miss
    // (counted, must be 0 in practice), never the other program's facts.
    StageFactsSlot& cached = stage_facts_cache[FactsHash(key) & (StageFactsSlots - 1)];
    if (cached.key == key) {
        if (cached.space == id.space && cached.code == id.code) {
            if (cached.epoch < epoch) {
                return false; // stale since a code write
            }
            out = cached.facts;
            return true;
        }
        Link::Note(Link::Check::FactsKeyCollision, "GPUMem's facts cache");
    }
    constexpr size_t Mask = FactsSlots - 1;
    const size_t home = FactsHash(key) & Mask;
    std::scoped_lock lock{facts_mutex};
    for (size_t probe = 0, index = home; probe < 16; ++probe, index = (index + 1) & Mask) {
        const FactsSlot& slot = facts_table[index];
        if (slot.key == key) {
            if (slot.space != id.space || slot.code != id.code) {
                Link::Note(Link::Check::FactsKeyCollision, "the facts table (lookup)");
                continue;
            }
            if (slot.epoch < epoch) {
                return false; // published before the latest code write: maybe the old code's
            }
            out = slot.facts;
            if (facts_generation.load(std::memory_order_relaxed) == generation) {
                // Kept only when the table is still the generation the cache stands for.
                cached.key = key;
                cached.space = id.space;
                cached.code = id.code;
                cached.facts = slot.facts;
                cached.epoch = slot.epoch;
            }
            return true;
        }
        if (slot.key == 0) {
            return false;
        }
    }
    return false;
}

bool MemStage::OneFollowerFill() noexcept {
    return VideoCommon::OneFollower::OnEngineThread() && OneFollowerRing();
}

void MemStage::OnBackImpl(void (*function)(void*), void* argument) {
    if (OneFollowerFill()) [[unlikely]] {
        OnRenderImpl(function, argument);
        return;
    }
    if (!Link::InCall()) {
        function(argument);
        return;
    }
    struct State {
        uintptr_t identity;
        u32 fill_depth;
        bool fill_lean;
        u32 play_depth;
        Site site;
        bool holding;
        ReadLog* log;
        std::vector<ReadRange>* reads;
        u64 stamp;       ///< Stamps per thread (step 7): a served sync stamps with GPUMem's packet
        bool stamp_stage;
    };
    const auto capture = [] {
        return State{t_identity,
                     t_fill_depth,
                     t_fill_lean,
                     t_play_depth,
                     t_site,
                     t_holding,
                     t_log,
                     t_reads,
                     SeamTicks::CurrentStamp(),
                     SeamTicks::CurrentStampStage()};
    };
    const auto install = [](const State& state) {
        t_identity = state.identity;
        t_fill_depth = state.fill_depth;
        t_fill_lean = state.fill_lean;
        t_play_depth = state.play_depth;
        t_site = state.site;
        t_holding = state.holding;
        t_log = state.log;
        t_reads = state.reads;
        SeamTicks::SetStamp(state.stamp, state.stamp_stage);
    };
    State gpumem = capture();
    gpumem.identity = ThisThread();
    MemStage* const stage = g_stage.load(std::memory_order_acquire);
    const std::thread::id gpumem_thread = std::this_thread::get_id();
    auto served = [&] {
        const State back = capture();
        install(gpumem);
        Common::SpinRecursiveMutex* const lent = stage != nullptr ? stage->lent_mutex : nullptr;
        if (lent != nullptr) {
            lent->HandOver(std::this_thread::get_id());
        }
        SCOPE_EXIT {
            if (lent != nullptr) {
                lent->HandOver(gpumem_thread);
            }
            gpumem = capture();
            install(back);
        };
        function(argument);
    };
    Link::Get().Serve(served);
    gpumem.identity = 0;
    install(gpumem);
}

void MemStage::OnRenderImpl(void (*function)(void*), void* argument) {
    // As OnBackImpl's served call, with E in GPUMem's place and R in the back's: R runs the sync
    // or the entry with E's fill state (its arena identity, depths, site, holding, logs, stamp),
    // holding the cache's lock E holds, while E is parked in the lend.
    struct State {
        uintptr_t identity;
        u32 fill_depth;
        bool fill_lean;
        u32 play_depth;
        Site site;
        bool holding;
        ReadLog* log;
        std::vector<ReadRange>* reads;
        u64 stamp;
        bool stamp_stage;
    };
    const auto capture = [] {
        return State{t_identity, t_fill_depth, t_fill_lean, t_play_depth, t_site, t_holding,
                     t_log,      t_reads,      SeamTicks::CurrentStamp(),  SeamTicks::CurrentStampStage()};
    };
    const auto install = [](const State& state) {
        t_identity = state.identity;
        t_fill_depth = state.fill_depth;
        t_fill_lean = state.fill_lean;
        t_play_depth = state.play_depth;
        t_site = state.site;
        t_holding = state.holding;
        t_log = state.log;
        t_reads = state.reads;
        SeamTicks::SetStamp(state.stamp, state.stamp_stage);
    };
    State engine = capture();
    engine.identity = ThisThread();
    MemStage* const stage = g_stage.load(std::memory_order_acquire);
    const std::thread::id engine_thread = std::this_thread::get_id();
    auto served = [&] {
        const State render = capture();
        install(engine);
        Common::SpinRecursiveMutex* const lent = stage != nullptr ? stage->lent_mutex : nullptr;
        if (lent != nullptr) {
            lent->HandOver(std::this_thread::get_id());
        }
        SCOPE_EXIT {
            if (lent != nullptr) {
                lent->HandOver(engine_thread);
            }
            engine = capture();
            install(render);
        };
        function(argument);
    };
    // Named by the seam site the play came from (the lends by site line, the lend trace).
    VideoCommon::OneFollower::SetLendSite(
        t_site < Site::Count ? SITE_NAMES[static_cast<size_t>(t_site)] : "seam play");
    VideoCommon::OneFollower::Lend(VideoCommon::OneFollower::LentOp::Other, served);
    engine.identity = 0;
    install(engine);
}

namespace {
/// FRITCHII_ESPLIT: what OnRenderImpl's State moves, as plain bytes in the hand-over's job.
struct ESplitState {
    uintptr_t identity;
    void* log;   ///< MemStage::ReadLog*
    void* reads; ///< std::vector<MemStage::ReadRange>*
    u64 stamp;
    u32 fill_depth;
    u32 play_depth;
    u8 site; ///< MemStage::Site
    bool fill_lean;
    bool holding;
    bool stamp_stage;
};
static_assert(sizeof(ESplitState) <= VideoCommon::ESplit::FillStateBytes);
static_assert(std::is_trivially_copyable_v<ESplitState>);
} // Anonymous namespace

void MemStage::ESplitCapture(void* out, bool as_owner) {
    const ESplitState state{
        .identity = as_owner ? ThisThread() : t_identity,
        .log = t_log,
        .reads = t_reads,
        .stamp = SeamTicks::CurrentStamp(),
        .fill_depth = t_fill_depth,
        .play_depth = t_play_depth,
        .site = static_cast<u8>(t_site),
        .fill_lean = t_fill_lean,
        .holding = t_holding,
        .stamp_stage = SeamTicks::CurrentStampStage(),
    };
    std::memcpy(out, &state, sizeof(state));
}

void MemStage::ESplitInstall(const void* in, bool keep_identity) {
    ESplitState state;
    std::memcpy(&state, in, sizeof(state));
    if (!keep_identity) {
        t_identity = state.identity;
    }
    t_log = static_cast<ReadLog*>(state.log);
    t_reads = static_cast<std::vector<ReadRange>*>(state.reads);
    t_fill_depth = state.fill_depth;
    t_fill_lean = state.fill_lean;
    t_play_depth = state.play_depth;
    t_site = static_cast<Site>(state.site);
    t_holding = state.holding;
    SeamTicks::SetStamp(state.stamp, state.stamp_stage);
}

namespace {
[[maybe_unused]] const bool g_esplit_fill_state = [] {
    VideoCommon::ESplit::g_fill_state = {&MemStage::ESplitCapture, &MemStage::ESplitInstall};
    return true;
}();
} // Anonymous namespace

void MemStage::CacheIdleImpl(void (*function)(void*), void* argument) {
    if (Link::RingActive()) {
        // THREAD = 2: GPUMem runs the cache's idle work itself, while the back is parked
        // (the ring backend's idle).
        return;
    }
    auto run = [&] {
        function(argument);
        player.device.PublishBufferHeapBudgets();
    };
    Link::Get().Call(Link::HandOff::Idle, run);
}

void MemStage::CheckHeld() {
    if (held_log_pending) {
        // 5.3: the packet's first play is the old read position.
        held_log_pending = false;
        const bool is_draw = held == HeldKind::Draw;
        CompareReads(held_log, is_draw ? "draw" : "dispatch",
                     is_draw ? (held_ring ? played_draw.number : draw.number) : dispatch.number,
                     held_hashes[0], held_hashes[1]);
    }
}

void MemStage::PlayHeld(Span span) {
    if (held == HeldKind::None) {
        return;
    }
    PlayHeldTo(span.end, false);
}

void MemStage::PlayHeldTo(u32 end, bool rest) {
    const bool timed = Profile::Enabled();
    const u64 start = timed ? NowNs() : 0;
    const Profile::Scope zone{Profile::Zone::MemoryPlay};
    CheckHeld();
    const u32 stop = (std::min)(end, held_arena.Size());
    if (held_cursor >= stop) {
        return;
    }
    if (rest) {
        // Configure returned before playing every section (false, or a path without the play).
        Bump(counters.rest_plays);
        (held == HeldKind::Draw ? (held_ring ? played_draw.flags : draw.flags)
                                : (held_ring ? played_dispatch.flags : dispatch.flags)) |=
            RestPlayed;
    }
    PlayDepth depth;
    while (held_cursor < stop) {
        const Entry& entry = *reinterpret_cast<const Entry*>(held_arena.At(held_cursor));
        held_cursor += entry.bytes;
        if ((entry.flags & Played) != 0) {
            continue;
        }
        if (entry.kind == Cmd::Descriptors) {
            if (rest) {
                // No Acquire for them: the draw or dispatch is dropped.
                Bump(counters.rest_descriptors, entry.count);
                continue;
            }
            if (player.guest_descriptor_queue.UsesDescriptorBuffer() != held_form) [[unlikely]] {
                Bump(counters.form_mismatch);
            }
        }
        player.Exec(entry);
    }
    if (timed) {
        Bump(counters.play_ns, NowNs() - start);
    }
}

void MemStage::PlayRest() {
    if (held == HeldKind::None) {
        return;
    }
    PlayHeldTo(held_arena.Size(), true);
    const Profile::FineScope tail_scope{Profile::Zone::RestTail};
    const bool ring_packet = held_ring;
    held = HeldKind::None;
    held_ring = false;
    t_holding = false;
    VideoCommon::MemReadLog::g_held.store(false, std::memory_order_relaxed);
    held_reads.clear();
    if (ticks) {
        UpdateUnplayed();
    }
    SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
    held_arena.Reset();
    held_cursor = 0;
    held_log.Clear();
    held_log_pending = false;
    // FRITCHII_MEMSTAGE_MARKS_AHEAD: the packet is played whole; the marks its fill made ahead
    // go into the buffer cache's real sets now, before the back goes on.
    if (ring_packet && cache != nullptr) {
        cache->NoteRingPlayed(held_packet_index, std::exchange(held_marks, false));
    }
}

void MemStage::NoteHeldFlush(SeamHook::Use use, const std::source_location& location) {
    if (held_flush_seen.exchange(true)) {
        return;
    }
    Bump(counters.held_flushed);
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        // THREAD = 2: a ring packet the back holds is played_draw (GPUMem's fields are its own).
        const bool ring = held_ring && held == HeldKind::Draw;
        const bool is_draw =
            ring || (filling_held.load() ? filling_kind.load() : held) == HeldKind::Draw;
        LOG_WARNING(Render_Vulkan,
                    "Memory seam (5.5): a {}{} between the fill and the PlayRest of {} {}: {}:{} {}",
                    UseName(use), t_play_depth != 0 ? " (in a sync)" : "",
                    is_draw ? "draw" : "dispatch",
                    ring              ? played_draw.number
                    : is_draw         ? draw.number
                    : HeldDispatchFromRing() ? played_dispatch.number
                                             : dispatch.number,
                    location.file_name(), location.line(), location.function_name());
    }
}

void MemPlayer::EnsureVertexBindings() {
    const u64 epoch = scheduler.VertexEpoch();
    if (epoch == ensured_vertex_epoch) {
        return;
    }
    ensured_vertex_epoch = epoch;
    for (u32 index = 0; index < NUM_BOUND_VERTEX_BUFFERS; ++index) {
        BoundVertexBuffer& bound = bound_vertex_buffers[index];
        if (bound.buffer == VK_NULL_HANDLE || bound.epoch == epoch || bound.epoch == ~u64{0}) {
            continue;
        }
        bound.epoch = epoch;
        Bump(vertex_runs_rebound);
        if (bound.dynamic) {
            scheduler.Record([index, buffer = bound.buffer, offset = bound.offset,
                              size = bound.size, stride = bound.stride](vk::CommandBuffer cmdbuf) {
                cmdbuf.BindVertexBuffers2EXT(index, 1, &buffer, &offset, &size, &stride);
            });
        } else {
            scheduler.Record([index, buffer = bound.buffer,
                              offset = bound.offset](vk::CommandBuffer cmdbuf) {
                cmdbuf.BindVertexBuffers(index, 1, &buffer, &offset);
            });
        }
    }
}

void MemPlayer::EnsureIndexBinding() {
    const u64 epoch = scheduler.VertexEpoch();
    if (!last_index_valid || last_index_buffer == VK_NULL_HANDLE ||
        last_index_vertex_epoch == epoch) {
        return;
    }
    last_index_vertex_epoch = epoch;
    bound_index_valid = false; // the rebind skip starts again from the next play
    scheduler.Record([buffer = last_index_buffer, offset = last_index_offset,
                      type = last_index_type](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindIndexBuffer(buffer, offset, type);
    });
}

// ---------------------------------------------------------------------------------------------
// The read log (5.3)

bool MemStage::StartReadLog(ReadLog& log, u64 number) {
    if (read_log_every <= 0 || number % static_cast<u64>(read_log_every) != 0 || t_log != nullptr) {
        return false;
    }
    log.Clear();
    log.ring_seq = Common::WriteRing::Sequence();
    t_log = &log;
    return true;
}

void MemStage::StopReadLog(ReadLog& log, bool held_packet) {
    t_log = nullptr;
    log.gathers = t_gathers;
    log.flush_regions = t_flush_regions;
    if (held_packet) {
        gather_frame_count = 0;
        // On GPUMem these are GPUMem's counters; HandOff takes the back's when the call returns.
        held_log_on_stage = Link::OnStageThread();
    }
    for (ReadRecord& record : log.records) {
        if (record.how != ReadRecord::How::Range) {
            continue;
        }
        // The tracker's answer when the fill ends; at the compare a range the CPU wrote since
        // (clean now, modified then) is a difference. An operation is compared at once.
        record.value = !held_packet || cache == nullptr ||
                               cache->IsRegionCpuModified(record.address, record.size)
                           ? 1
                           : 0;
    }
    if (held_packet) {
        held_log_pending = true;
    }
}

void MemStage::CompareReads(ReadLog& log, const char* what, u64 number, u64 hash_a, u64 hash_b) {
    using How = ReadRecord::How;
    for (const ReadRecord& record : log.records) {
        Bump(counters.read_kind_records[static_cast<size_t>(record.kind)]);
        bool differs = false;
        // The first and last byte that differ, from the record's address.
        u64 first = 0;
        u64 last = record.size != 0 ? record.size - 1 : 0;
        switch (record.how) {
        case How::Bytes:
        case How::Staged: {
            compare_scratch.resize(record.size);
            static_cast<Tegra::MaxwellDeviceMemoryManager*>(record.memory)
                ->ReadBlockUnsafe(record.address, compare_scratch.data(), record.size);
            const u8* const expected = record.how == How::Bytes
                                           ? log.bytes.data() + record.value
                                           : static_cast<const u8*>(record.data);
            differs = std::memcmp(compare_scratch.data(), expected, record.size) != 0;
            if (differs) {
                while (compare_scratch[first] == expected[first]) {
                    ++first;
                }
                while (compare_scratch[last] == expected[last]) {
                    --last;
                }
            }
            break;
        }
        case How::Value: {
            const auto* const memory = static_cast<const Tegra::MemoryManager*>(record.memory);
            const u64 now = record.size == 4 ? u64{memory->Read<u32>(record.address)}
                                             : memory->Read<u64>(record.address);
            differs = now != record.value;
            if (differs) {
                // Little endian: the lowest changed bit's byte is the first.
                const u64 changed = now ^ record.value;
                first = static_cast<u64>(std::countr_zero(changed)) / 8;
                last = static_cast<u64>(63 - std::countl_zero(changed)) / 8;
            }
            break;
        }
        case How::Range:
            differs = record.value == 0 && cache != nullptr &&
                      cache->IsRegionCpuModified(record.address, record.size);
            break;
        }
        if (!differs) {
            continue;
        }
        Bump(counters.read_diffs[static_cast<size_t>(record.kind)]);
        ClassifyDiff(log, record, first, last, what, number, hash_a, hash_b);
    }
    Bump(counters.read_packets);
    Bump(counters.read_records, log.records.size());
    Bump(counters.read_skipped, log.skipped);
    log.Clear();
}

// ---------------------------------------------------------------------------------------------
// Step 7's verify: the read compare of ring packets (spec 6.4; the design's 2.5)
//
// With the window open (FRITCHII_MEMSTAGE_AHEAD > 0) and FRITCHII_MEMSTAGE_READ_LOG=N, every Nth
// ring packet's fill logs its reads of game memory on GPUMem as the read log does (StartRingReadLog;
// the hooks are the same), the log goes to the back with the packet (ring_logs, in packet order),
// and the back reads every range again at the take, before anything of that draw runs, and
// compares. Left out: bytes GPUMem itself wrote after the fill (its constant buffer words for
// later entries, from its own ring of writes with their entries, MemoryStageLink::NoteStageWrite;
// those writes are kept out of the emulator's write ring meanwhile, where they would only wrap it
// over a lead of a frame). Counted as "left out: GPUMem's own later writes". What is left is
// classed as every read log difference ("Who wrote" above), so the gate is the read log's:
// unexplained 0 (emulator writes that are not download landings, unclassified), ring wrapped 0
// (GPUMem's own ring counts there too). A bound range has no tracker on the back (the cache is
// GPUMem's): it differs when an emulator write noted since the fill, a mark the cache took since
// or a pending game-side mark covers it, and is classed the same way. The marks the cache takes
// (a served gather or invalidation, a game core's trapped store) go to a small ring of their own
// while the compare runs (NoteCpuMark): a difference over one, with no emulator write, is "game
// write, marked" (the tracker holds it, the next bind uploads), not "mark already taken". Every
// ring that wrapped since a fill counts as "ring wrapped". Only GPUMem's own constant buffer
// writes are kept out of the emulator's write ring (counted): every other compare ends before
// GPUMem writes again (held packets in runs, where GPUMem is parked; operations, compared at once).

void MemStage::NoteCpuMarkSlow(u64 address, u64 size) noexcept {
    if (!cpu_marks || size == 0) {
        return;
    }
    const u64 number = cpu_mark_count.fetch_add(1, std::memory_order_relaxed);
    CpuMark& slot = cpu_marks[number % CpuMarkSlots];
    slot.seq.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    slot.begin.store(address, std::memory_order_relaxed);
    slot.end.store(address + size, std::memory_order_relaxed);
    slot.seq.store(number + 1, std::memory_order_release);
}

bool MemStage::CpuMarkedSince(u64 since, u64 begin, u64 end, bool& complete) const {
    if (!cpu_marks) {
        return false;
    }
    const u64 count = cpu_mark_count.load(std::memory_order_acquire);
    if (count <= since) {
        return false;
    }
    if (count - since > CpuMarkSlots) {
        complete = false;
    }
    const u64 oldest = count - since > CpuMarkSlots ? count - CpuMarkSlots : since;
    for (u64 n = count; n > oldest; --n) {
        const CpuMark& slot = cpu_marks[(n - 1) % CpuMarkSlots];
        if (slot.seq.load(std::memory_order_acquire) != n) {
            continue; // being written (a later mark), or reused
        }
        const u64 mark_begin = slot.begin.load(std::memory_order_relaxed);
        const u64 mark_end = slot.end.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (slot.seq.load(std::memory_order_relaxed) != n) {
            continue;
        }
        if (mark_begin < end && begin < mark_end) {
            return true;
        }
    }
    return false;
}

bool MemStage::StartRingReadLog(u64 number) {
    if (Link::Ahead() == 0 || read_log_every <= 0 ||
        number % static_cast<u64>(read_log_every) != 0 || t_log != nullptr) {
        return false;
    }
    {
        std::scoped_lock lock{ring_logs_mutex};
        if (!ring_log_pool.empty()) {
            stage_log = std::move(ring_log_pool.back());
            ring_log_pool.pop_back();
        }
    }
    if (!stage_log) {
        stage_log = std::make_unique<ReadLog>();
    }
    stage_log->Clear();
    stage_log->ring_seq = Common::WriteRing::Sequence();
    stage_log->position = ring_fill_position;
    t_log = stage_log.get();
    return true;
}

void MemStage::StopRingReadLog() {
    t_log = nullptr;
    if (!stage_log) {
        return;
    }
    ReadLog& log = *stage_log;
    log.stage_writes = Link::Get().StageWriteCount();
    log.cpu_marks = cpu_mark_count.load(std::memory_order_acquire);
    for (ReadRecord& record : log.records) {
        if (record.how == ReadRecord::How::Range) {
            // The tracker's answer at the fill's end (GPUMem holds the cache).
            record.value =
                cache == nullptr || cache->IsRegionCpuModified(record.address, record.size) ? 1 : 0;
        }
    }
}

std::unique_ptr<MemStage::ReadLog> MemStage::TakeRingLog(u64 number) {
    std::scoped_lock lock{ring_logs_mutex};
    // Logs of packets taken out of order cannot exist: ring B is played in order. An older one
    // left (a packet dropped some other way) goes back to the pool.
    while (!ring_logs.empty() && ring_logs.front().first < number) {
        ring_log_pool.push_back(std::move(ring_logs.front().second));
        ring_logs.pop_front();
    }
    if (ring_logs.empty() || ring_logs.front().first != number) {
        return nullptr;
    }
    std::unique_ptr<ReadLog> log = std::move(ring_logs.front().second);
    ring_logs.pop_front();
    return log;
}

void MemStage::RecycleRingLog(std::unique_ptr<ReadLog> log) {
    if (!log) {
        return;
    }
    log->Clear();
    std::scoped_lock lock{ring_logs_mutex};
    if (ring_log_pool.size() < 256) {
        ring_log_pool.push_back(std::move(log));
    }
}

void MemStage::CompareRingReads(ReadLog& log, u64 number, u64 hash_a, u64 hash_b) {
    using How = ReadRecord::How;
    thread_local std::vector<u8> scratch;
    // Class D (gathers on this thread since the fill): the compare is at the take, before any.
    log.gathers = t_gathers;
    log.flush_regions = t_flush_regions;
    Link& link = Link::Get();
    // GPUMem's writes after the fill over [begin, end): the intervals the compare leaves out.
    boost::container::small_vector<std::pair<u64, u64>, 8> later;
    const auto later_writes = [&](u64 begin, u64 end) {
        later.clear();
        const bool complete = link.ForEachStageWriteSince(
            log.stage_writes, [&](u64 write_begin, u64 write_end, u64) {
                if (write_begin < end && begin < write_end) {
                    later.emplace_back((std::max)(write_begin, begin), (std::min)(write_end, end));
                }
                return false;
            });
        if (!complete) {
            Bump(counters.ring_wrapped);
        }
    };
    const auto left_out = [&](u64 device) {
        return std::any_of(later.begin(), later.end(), [device](const std::pair<u64, u64>& w) {
            return device >= w.first && device < w.second;
        });
    };
    for (const ReadRecord& record : log.records) {
        Bump(counters.read_kind_records[static_cast<size_t>(record.kind)]);
        bool differs = false;
        bool own_later = false;
        u64 first = 0;
        u64 last = record.size != 0 ? record.size - 1 : 0;
        switch (record.how) {
        case How::Bytes:
        case How::Staged: {
            scratch.resize(record.size);
            static_cast<Tegra::MaxwellDeviceMemoryManager*>(record.memory)
                ->ReadBlockUnsafe(record.address, scratch.data(), record.size);
            const u8* const expected = record.how == How::Bytes
                                           ? log.bytes.data() + record.value
                                           : static_cast<const u8*>(record.data);
            if (record.size == 0 || std::memcmp(scratch.data(), expected, record.size) == 0) {
                break;
            }
            later_writes(record.address, record.address + record.size);
            bool found = false;
            for (u64 i = 0; i < record.size; ++i) {
                if (scratch[i] == expected[i]) {
                    continue;
                }
                if (left_out(record.address + i)) {
                    own_later = true;
                    continue;
                }
                if (!found) {
                    first = i;
                    found = true;
                }
                last = i;
            }
            differs = found;
            break;
        }
        case How::Value: {
            const auto* const memory = static_cast<const Tegra::MemoryManager*>(record.memory);
            const u64 now = record.size == 4 ? u64{memory->Read<u32>(record.address)}
                                             : memory->Read<u64>(record.address);
            if (now == record.value) {
                break;
            }
            const u64 changed = now ^ record.value;
            first = static_cast<u64>(std::countr_zero(changed)) / 8;
            last = static_cast<u64>(63 - std::countl_zero(changed)) / 8;
            if (const auto device = memory->GpuToCpuAddress(record.address)) {
                // Left out only when every changed byte is one GPUMem wrote later (R5c). A
                // value is 4 or 8 aligned bytes: one page, so the device bytes follow on.
                later_writes(*device, *device + record.size);
                bool all_own = true;
                bool any_own = false;
                u64 kept_first = ~u64{0};
                u64 kept_last = 0;
                for (u64 i = first; i <= last; ++i) {
                    if (((changed >> (i * 8)) & 0xff) == 0) {
                        continue;
                    }
                    if (left_out(*device + i)) {
                        any_own = true;
                        continue;
                    }
                    all_own = false;
                    kept_first = (std::min)(kept_first, i);
                    kept_last = i;
                }
                own_later = any_own;
                if (all_own) {
                    break;
                }
                first = kept_first;
                last = kept_last;
            }
            differs = true;
            break;
        }
        case How::Range: {
            if (record.value != 0) {
                break; // CPU-written at the fill already: uploaded or read as such
            }
            // No tracker on the back: an emulator write noted since the fill, a mark the cache
            // took since (CpuMarks), or a game-side mark pending now, over the range.
            const u64 begin = record.address;
            const u64 end = begin + record.size;
            const bool complete = Common::WriteRing::ForEachSince(
                log.ring_seq, [&](const Common::WriteRing::Entry& e) {
                    differs = e.address < end && begin < e.address + e.size;
                    return differs;
                });
            if (!complete) {
                // R5a: writers since the fill unknown, as ClassifyDiff counts it.
                Bump(counters.ring_wrapped);
            }
            if (!differs) {
                bool marks_complete = true;
                differs = CpuMarkedSince(log.cpu_marks, begin, end, marks_complete);
                if (!marks_complete) {
                    Bump(counters.ring_wrapped);
                }
            }
            if (!differs && gpu != nullptr) {
                gpu->ForEachPendingCpuWrite([&](DAddr address, size_t size) {
                    differs = differs || (address < end && begin < address + size);
                });
            }
            break;
        }
        }
        if (own_later) {
            Bump(counters.diff_stage_own);
        }
        if (!differs) {
            continue;
        }
        Bump(counters.read_diffs[static_cast<size_t>(record.kind)]);
        // Fable's review, 2: a game store the cache took since the fill (a served gather, a
        // trapped store) leaves no pending mark; the tracker holds it, the next bind uploads.
        bool tracker_holds = false;
        std::optional<u64> device;
        if (record.how == How::Value) {
            if (const auto* const memory = static_cast<const Tegra::MemoryManager*>(record.memory)) {
                device = memory->GpuToCpuAddress(record.address + first);
            }
        } else {
            device = record.address + first;
        }
        if (device) {
            bool marks_complete = true;
            tracker_holds =
                CpuMarkedSince(log.cpu_marks, *device, *device + (last - first + 1), marks_complete);
            if (!marks_complete) {
                Bump(counters.ring_wrapped);
            }
        }
        ClassifyDiff(log, record, first, last, "ring draw", number, hash_a, hash_b, tracker_holds);
    }
    Bump(counters.read_packets);
    Bump(counters.ring_reads);
    if (cpu_mark_count.load(std::memory_order_acquire) != log.cpu_marks) {
        // The cache took a game-side mark (anywhere) between this packet's fill and its take.
        Bump(counters.ring_reads_marked);
    }
    Bump(counters.read_records, log.records.size());
    Bump(counters.read_skipped, log.skipped);
    log.Clear();
}

// Every difference gets a class (A, B, D) and a place (C); see "Who wrote" at the top.
void MemStage::ClassifyDiff(const ReadLog& log, const ReadRecord& record, u64 first, u64 last,
                            const char* what, u64 number, u64 hash_a, u64 hash_b,
                            bool tracker_holds) {
    using How = ReadRecord::How;
    using Kind = VideoCommon::MemReadLog::Kind;
    // The changed bytes as a device range (a value is read by GPU address).
    std::optional<u64> device;
    if (record.how == How::Value) {
        if (const auto* const memory = static_cast<const Tegra::MemoryManager*>(record.memory)) {
            device = memory->GpuToCpuAddress(record.address + first);
        }
    } else {
        device = record.address + first;
    }
    const u64 length = last - first + 1;
    const u64 begin = device.value_or(0);
    const u64 end = begin + length;
    const auto overlaps = [begin, end](u64 address, u64 size) {
        return address < end && begin < address + size;
    };

    // C: an upload covers whole tracked words; the packet uses only its bound ranges, uniforms
    // and values. Every other read is inside a range the packet binds by what it is (for a
    // dispatch that is whole storage buffers: whether the shader reads the bytes is not known).
    bool own = true;
    if (record.kind == Kind::Upload && device) {
        own = std::any_of(log.used.begin(), log.used.end(), [&](const std::pair<u64, u64>& used) {
            return overlaps(used.first, used.second);
        });
    }

    // A: an emulator write noted after the fill began.
    Common::WriteRing::Entry writer{};
    bool found = false;
    bool complete = true;
    if (device) {
        complete = Common::WriteRing::ForEachSince(
            log.ring_seq, [&](const Common::WriteRing::Entry& entry) {
                if (overlaps(entry.address, entry.size)) {
                    writer = entry;
                    found = true;
                    return true;
                }
                return false;
            });
        if (!complete) {
            Bump(counters.ring_wrapped);
        }
    }

    // B: a game-side mark over the changed bytes not applied yet.
    bool asked = false;
    bool pending = false;
    if (device && gpu != nullptr) {
        asked = true;
        gpu->ForEachPendingCpuWrite([&](DAddr address, size_t size) {
            pending = pending || overlaps(address, size);
        });
    }

    // D: gathers on this thread since the fill ended (a held packet; an operation compares at
    // once, they never move there).
    const u64 gathers = t_gathers - log.gathers;
    const u64 flush_regions = t_flush_regions - log.flush_regions;

    DiffClass kind;
    if (record.how == How::Range && gathers != 0) {
        kind = DiffClass::BackGather;
    } else if (found && !writer.landing && !own && writer.via != nullptr &&
               std::strcmp(writer.via, "GPU write: query-completion") == 0 &&
               std::strncmp(writer.thread.data(), "GPUFencingThread", 16) == 0) {
        // A query result the fencing thread wrote when the GPU finished (not at a stream
        // position, so a fill can meet it at any distance from its play), on bytes the packet
        // does not bind: reported apart. Inside a range the packet binds it stays a failure.
        kind = DiffClass::AsyncOverRead;
    } else if (found) {
        kind = writer.landing ? DiffClass::Landing : DiffClass::EmulatorWrite;
    } else if (tracker_holds && device && complete) {
        // Step 7: the cache took a game-side mark over these bytes since the fill (CpuMarks).
        kind = DiffClass::GameMarked;
    } else if (!device || !complete || !asked) {
        kind = DiffClass::Unclassified;
    } else if (record.how == How::Range) {
        // The tracker holds the mark: a game core's trapped store marked it directly.
        kind = DiffClass::GameMarked;
    } else if (record.kind != Kind::Upload && record.kind != Kind::InlinePageWrite) {
        // No tracker behind the read (a stream uniform, a value): no mark is expected and the
        // next fill reads it again, so "mark already taken" stays the unrepaired tracked case.
        kind = DiffClass::Untracked;
    } else {
        kind = pending ? DiffClass::GameMarked : DiffClass::MarkTaken;
    }
    const size_t index = static_cast<size_t>(kind);
    Bump(counters.diff_classes[index]);
    Bump(own ? counters.diff_own : counters.diff_over);
    if (found && (kind == DiffClass::Landing || kind == DiffClass::EmulatorWrite)) {
        CountDiffSite(writer.site != nullptr ? writer.site : writer.via, writer.landing);
    }
    if (class_logged[index].fetch_add(1, std::memory_order_relaxed) >= 16) {
        return;
    }
    std::string who;
    if (found) {
        who = fmt::format("; written by {}{}{} via {} on {}, {} ring entries after the fill began",
                          writer.site != nullptr ? writer.site : "an unscoped site",
                          writer.context != nullptr && writer.context != writer.site
                              ? fmt::format(" (in {})", writer.context)
                              : std::string{},
                          writer.landing ? ", a landing" : "", writer.via,
                          writer.thread[0] != '\0' ? writer.thread.data() : "an unnamed thread",
                          writer.seq - log.ring_seq);
    } else if (device) {
        who = complete ? "; no emulator write in the ring" : "; the ring wrapped since the fill";
    }
    const std::string mark = asked ? (pending ? "; a game-side mark is pending over them"
                                              : "; no game-side mark pending over them")
                                   : "; marks not asked (no GPU or unmapped)";
    std::string back;
    if (record.how == How::Range || gathers != 0 || flush_regions != 0) {
        back = fmt::format("; this thread since the fill: {} gathers applied, {} FlushRegion{}",
                           gathers, flush_regions,
                           kind == DiffClass::BackGather && gather_frame_count != 0
                               ? fmt::format("; first gather from:{}", GatherFrames())
                               : std::string{});
    }
    LOG_WARNING(Render_Vulkan,
                "Memory seam (5.3) {}: {} {} read {} at {:#x}+{:#x}, bytes +{:#x}..+{:#x} changed "
                "(device {}), {}{}{}{} (programs {:016x} {:016x})",
                DIFF_CLASS_NAMES[index], what, number,
                READ_KIND_NAMES[static_cast<size_t>(record.kind)], record.address, record.size,
                first, last,
                device ? fmt::format("{:#x}+{:#x}", *device, length) : std::string{"unmapped"},
                own ? "inside a range the packet binds"
                    : "over-read (outside every range the packet binds)",
                who, mark, back, hash_a, hash_b);
}

void MemStage::CountDiffSite(const char* name, bool landing) {
    std::scoped_lock lock{diff_sites_mutex};
    for (DiffSite& site : diff_site_table) {
        const char* const known = site.name.load(std::memory_order_relaxed);
        if (known == nullptr) {
            site.landing.store(landing, std::memory_order_relaxed);
            site.count.store(1, std::memory_order_relaxed);
            site.name.store(name, std::memory_order_release);
            return;
        }
        if (site.landing.load(std::memory_order_relaxed) == landing &&
            std::strcmp(known, name) == 0) {
            Bump(site.count);
            return;
        }
    }
    Bump(counters.diff_sites_other);
}

void MemStage::NoteGather(Gather kind, u32 which) {
    if (kind == Gather::Apply) {
        ++t_gathers;
    } else {
        ++t_flush_regions;
    }
    if (!t_holding) {
        return;
    }
    if (kind == Gather::FlushRegion) {
        for (u32 bit = 0; bit < 3; ++bit) {
            if ((which & (1U << bit)) != 0) {
                Bump(counters.flush_regions_held[bit]);
            }
        }
        return;
    }
    Bump(counters.gathers_held);
    if (held_log_pending && gather_frame_count == 0) {
#ifdef _WIN32
        gather_frame_count = RtlCaptureStackBackTrace(
            1, static_cast<ULONG>(gather_frames.size()), gather_frames.data(), nullptr);
#else
        gather_frames[0] = __builtin_return_address(0);
        gather_frame_count = 1;
#endif
    }
}

std::string MemStage::GatherFrames() const {
    std::string text;
    for (u32 i = 0; i < gather_frame_count; ++i) {
#ifdef _WIN32
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(gather_frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* const base_name = std::strrchr(name, '\\');
        text += fmt::format(" {}+{:#x}", base_name != nullptr ? base_name + 1 : name,
                            reinterpret_cast<uintptr_t>(gather_frames[i]) -
                                reinterpret_cast<uintptr_t>(module));
#else
        text += fmt::format(" {}", gather_frames[i]);
#endif
    }
    return text;
}

void MemStage::CountPush(Cmd kind) noexcept {
    Bump(counters.entries[static_cast<size_t>(kind)]);
    if (t_fill_depth != 0 && t_site != Site::Count) {
        Bump(counters.site_entries[static_cast<size_t>(t_site)]);
    } else {
        Bump(counters.outside[static_cast<size_t>(kind)]);
    }
}

// PushParts's path for an entry that is not deferred, without the entry: the same checks in the
// same order, then the typed play the entry's Exec would have called (MemPlayer::PlayCopy ...).
// The seam lean skips the stats counts only (no stats line reads them while it holds).
template <typename PlayFn>
void MemStage::RunNow(Cmd kind, PlayFn&& play) {
    if (!lean.load(std::memory_order_relaxed)) {
        CountPush(kind);
    }
    Link::CheckCacheUser(Link::Check::PushUser);
    if (Link::Checking() && pending && OwnedHere()) [[unlikely]] {
        Link::Note(Link::Check::PushWhilePending, CMD_NAMES[static_cast<size_t>(kind)]);
    }
    if (Link::InCall() || OneFollowerFill()) [[unlikely]] {
        // GPUMem has nowhere to put it (spec 5.5): the back records it, counted. One follower:
        // E has no player either; R records it (lent).
        if (Link::InCall()) {
            Link::Note(Link::Check::PushNowhere, CMD_NAMES[static_cast<size_t>(kind)]);
        }
        OnBack([&play] {
            PlayDepth depth;
            play();
        });
        return;
    }
    PlayDepth depth;
    play();
}

void MemStage::PushParts(Cmd kind, u8 flags, u32 count, std::initializer_list<Part> parts) {
    u32 bytes = static_cast<u32>(sizeof(Entry));
    for (const Part& part : parts) {
        bytes += Align8(part.bytes);
    }
    if (!lean.load(std::memory_order_relaxed)) {
        CountPush(kind);
    }
    Link::CheckCacheUser(Link::Check::PushUser);
    const bool defer = Deferring();
    Arena& target = defer ? arena : Scratch();
    if (!defer) {
        target.Reset();
    }
    const u32 offset = target.Reserve(bytes);
    u8* const out = target.At(offset);
    const Entry entry{
        .kind = kind,
        .flags = flags,
        .count = static_cast<u16>((std::min)(count, 0xffffU)),
        .bytes = bytes,
    };
    std::memcpy(out, &entry, sizeof(Entry));
    u32 at = static_cast<u32>(sizeof(Entry));
    for (const Part& part : parts) {
        if (part.bytes != 0) {
            std::memcpy(out + at, part.data, part.bytes);
        }
        at += Align8(part.bytes);
    }
    if (defer) {
        last_entry = offset;
        Bump(counters.deferred);
        SetPending();
        return;
    }
    // FRITCHII_MEMSTAGE_THREAD >= 1 only: at 0 this is the seam's own case, logged nowhere new.
    if (Link::Checking() && pending && OwnedHere()) [[unlikely]] {
        Link::Note(Link::Check::PushWhilePending, CMD_NAMES[static_cast<size_t>(kind)]);
    }
    const ScratchDepth scratch_depth;
    if (Link::InCall() || OneFollowerFill()) [[unlikely]] {
        // GPUMem has nowhere to put it (spec 5.5): the back records it, counted. One follower:
        // R records it (lent).
        if (Link::InCall()) {
            Link::Note(Link::Check::PushNowhere, CMD_NAMES[static_cast<size_t>(kind)]);
        }
        OnBack([this, out] {
            PlayDepth depth;
            player.Exec(*reinterpret_cast<const Entry*>(out));
        });
        return;
    }
    PlayDepth depth;
    player.Exec(*reinterpret_cast<const Entry*>(out));
}

void MemStage::Push(Cmd kind, u8 flags) {
    PushParts(kind, flags, 0, {});
}

void MemStage::PushCopy(Cmd kind, u8 flags, VkBuffer src, VkBuffer dst,
                        std::span<const VkBufferCopy> copies) {
    if (!Deferring()) {
        RunNow(kind, [&] { player.PlayCopy(kind, flags, src, dst, copies); });
        return;
    }
    const CopyPayload payload{
        .src = src,
        .dst = dst,
        .count = static_cast<u32>(copies.size()),
        .reserved = 0,
    };
    PushParts(kind, flags, payload.count, {PartOf(payload), PartOf(copies)});
}

void MemStage::PushClear(VkBuffer dst, u64 offset, u64 size, u32 value) {
    const ClearPayload payload{.dst = dst, .offset = offset, .size = size, .value = value};
    PushParts(Cmd::Clear, OutsideRenderPass, 1, {PartOf(payload)});
}

void MemStage::PushQuadLut(VkBuffer src, VkDeviceSize src_offset, VkBuffer dst,
                           VkDeviceSize size) {
    const QuadLutPayload payload{.src = src, .src_offset = src_offset, .dst = dst, .size = size};
    PushParts(Cmd::QuadLutUpload, OutsideRenderPass, 1, {PartOf(payload)});
}

void MemStage::PushNullInit(VkBuffer buffer) {
    const NullInitPayload payload{.buffer = buffer};
    PushParts(Cmd::NullInit, OutsideRenderPass, 1, {PartOf(payload)});
}

void MemStage::PushIndex(Cmd kind, u8 flags, VkBuffer buffer, VkDeviceSize offset,
                         VkIndexType type) {
    if (!Deferring()) {
        RunNow(kind, [&] { player.PlayIndex(kind, flags, buffer, offset, type); });
        return;
    }
    const IndexPayload payload{.buffer = buffer, .offset = offset, .type = type};
    PushParts(kind, flags, 1, {PartOf(payload)});
}

void MemStage::PushVertex(u8 flags, u32 index, VkBuffer buffer, u32 offset, u32 size,
                          u32 stride) {
    if (!Deferring()) {
        RunNow(Cmd::BindVertex,
               [&] { player.PlayVertex(flags, index, buffer, offset, size, stride); });
        return;
    }
    const VertexOne payload{
        .buffer = buffer, .index = index, .offset = offset, .size = size, .stride = stride};
    PushParts(Cmd::BindVertex, flags, 1, {PartOf(payload)});
}

void MemStage::PushVertices(u8 flags, u32 first, std::span<const VkBuffer> buffers,
                            std::span<const VkDeviceSize> offsets,
                            std::span<const VkDeviceSize> sizes,
                            std::span<const VkDeviceSize> strides) {
    const u32 count = static_cast<u32>(buffers.size());
    if (!Deferring()) {
        RunNow(Cmd::BindVertices, [&] {
            player.PlayVertices(flags, first, count, buffers.data(), offsets.first(count).data(),
                                sizes.first(count).data(), strides.first(count).data());
        });
        return;
    }
    const VerticesPayload payload{.first = first, .count = count};
    PushParts(Cmd::BindVertices, flags, count,
              {PartOf(payload), PartOf(buffers), PartOf(offsets.first(count)),
               PartOf(sizes.first(count)), PartOf(strides.first(count))});
}

void MemStage::PushXfb(u32 index, VkBuffer buffer, u32 offset, u32 size) {
    const XfbOne payload{.buffer = buffer, .index = index, .offset = offset, .size = size};
    PushParts(Cmd::BindXfb, 0, 1, {PartOf(payload)});
}

void MemStage::PushXfbs(std::span<const VkBuffer> buffers, std::span<const VkDeviceSize> offsets,
                        std::span<const VkDeviceSize> sizes) {
    const u32 count = static_cast<u32>(buffers.size());
    const XfbsPayload payload{.count = count, .reserved = 0};
    PushParts(Cmd::BindXfbs, 0, count,
              {PartOf(payload), PartOf(buffers), PartOf(offsets.first(count)),
               PartOf(sizes.first(count))});
}

void MemStage::PushConvert(Pass pass, const ComputePassRun& run) {
    // No OutsideRenderPass flag: the pass's Record asks for it itself, after its descriptor work.
    const ConvertPayload payload{.pass = pass, .run = run};
    PushParts(Cmd::Convert, 0, 1, {PartOf(payload)});
}

void MemStage::PushDescriptor(const DescriptorUpdateEntry& descriptor, bool texel) {
    static_assert(sizeof(DescriptorUpdateEntry) % 8 == 0);
    // The seam lean: no stats line reads the counts.
    const bool counted = !lean.load(std::memory_order_relaxed);
    if (texel && counted) {
        Bump(counters.texel);
    }
    Link::CheckCacheUser(Link::Check::PushUser);
    if (!Deferring()) {
        if (counted) {
            CountPush(Cmd::Descriptors);
        }
        if (Link::Checking() && pending && OwnedHere()) [[unlikely]] {
            Link::Note(Link::Check::PushWhilePending, "descriptor");
        }
        if (Link::InCall() || OneFollowerFill()) [[unlikely]] {
            if (Link::InCall()) {
                Link::Note(Link::Check::PushNowhere, "descriptor");
            }
            OnBack([this, &descriptor] {
                PlayDepth depth;
                player.AddDescriptors(&descriptor, 1);
            });
            return;
        }
        PlayDepth depth;
        player.AddDescriptors(&descriptor, 1);
        return;
    }
    // Consecutive descriptors make one entry: one AddEntries call when played.
    if (last_entry != ~0U) {
        const Entry& last = *reinterpret_cast<const Entry*>(arena.At(last_entry));
        if (last.kind == Cmd::Descriptors && last_entry + last.bytes == arena.Size()) {
            CountPush(Cmd::Descriptors);
            const u32 at = arena.Reserve(static_cast<u32>(sizeof(DescriptorUpdateEntry)));
            std::memcpy(arena.At(at), &descriptor, sizeof(DescriptorUpdateEntry));
            Entry& grown = *reinterpret_cast<Entry*>(arena.At(last_entry));
            grown.bytes += static_cast<u32>(sizeof(DescriptorUpdateEntry));
            auto& payload =
                *reinterpret_cast<DescriptorsPayload*>(arena.At(last_entry + sizeof(Entry)));
            ++payload.count;
            grown.count = static_cast<u16>((std::min)(payload.count, 0xffffU));
            Bump(counters.deferred);
            return;
        }
    }
    const DescriptorsPayload payload{.count = 1, .reserved = 0};
    PushParts(Cmd::Descriptors, 0, 1, {PartOf(payload), PartOf(descriptor)});
}

void MemStage::Note(SeamHook::Use use, const std::source_location& location, bool around) {
    if (around && !IsRecord(use)) {
        Bump(counters.unclocked);
    } else {
        Bump(around ? counters.around : counters.late);
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Vulkan,
                    "Memory seam: {} ({}) {}, site {}: {}:{} {}",
                    around ? (IsRecord(use) ? "a record around the packet"
                                            : "a scheduler use not through the clock")
                           : "a late play",
                    around ? "5.1" : "5.2", UseName(use),
                    t_site == Site::Count ? "none" : SITE_NAMES[static_cast<size_t>(t_site)],
                    location.file_name(), location.line(), location.function_name());
    }
}

void MemStage::TickFrame() {
    int value = VideoCore::Nx2Debug::Experiment(123);
    // AMD safe default (2026-11-16): SEAM 3 lost the device on an RX 7900 XT and an RX 9070 XT at
    // the title screen (the ASTC decoder's submit inside a held fill); SEAM 1 ran clean on both.
    // Since 2026-11-19 that submit is skipped by default (FRITCHII_ASTC_HELD_DEFER=1), and with
    // one follower on SEAM 3 is what the AMD one-follower result and soak ran at: kept on AMD's
    // proprietary driver when one follower is on (unset FRITCHII_SEAM_AMD). Held to 1 with one
    // follower off (there SEAM 3 is about 0.6 ms slower than 1 on AMD) or FRITCHII_SEAM_AMD=0;
    // FRITCHII_SEAM_AMD=1 keeps the asked level either way (tests).
    if (value >= 2 && VideoCommon::OneFollower::IsAmdProprietaryDriver()) [[unlikely]] {
        // The unset keep also needs the ASTC held defer: it is what made SEAM 3 survive the title
        // on AMD (FRITCHII_ASTC_HELD_DEFER=0 would bring the title loss back).
        static const bool defer = AstcHeldDeferOn(true);
        static const bool keep = [] {
            const char* const v = std::getenv("FRITCHII_SEAM_AMD");
            if (v != nullptr && v[0] != '\0') {
                return v[0] != '0';
            }
            return VideoCommon::OneFollower::On() && defer;
        }();
        static bool logged = false;
        if (!std::exchange(logged, true)) {
            if (!keep && VideoCommon::OneFollower::On() && !defer &&
                std::getenv("FRITCHII_SEAM_AMD") == nullptr) {
                LOG_WARNING(Render_Vulkan,
                            "Memory seam: EXP123={} held to 1 on AMD's proprietary driver (one "
                            "follower on, but FRITCHII_ASTC_HELD_DEFER is off: SEAM 3 lost the "
                            "device at the title without it); FRITCHII_SEAM_AMD=1 keeps it",
                            value);
            } else if (keep) {
                LOG_INFO(Render_Vulkan,
                         "Memory seam: EXP123={} kept on AMD's proprietary driver ({}); "
                         "FRITCHII_SEAM_AMD=0 holds it to 1",
                         value,
                         std::getenv("FRITCHII_SEAM_AMD") != nullptr ? "FRITCHII_SEAM_AMD=1"
                                                                     : "one follower on");
            } else {
                LOG_WARNING(Render_Vulkan,
                            "Memory seam: EXP123={} held to 1 on AMD's proprietary driver ({}); "
                            "FRITCHII_SEAM_AMD=1 keeps it",
                            value,
                            std::getenv("FRITCHII_SEAM_AMD") != nullptr ? "FRITCHII_SEAM_AMD=0"
                                                                        : "one follower off");
            }
        }
        if (!keep) {
            value = 1;
        }
    }
    if (value >= 2 && !ticks) [[unlikely]] {
        // The early fill needs packet ticks (the header comment): run SEAM 1.
        Bump(counters.seam_clamped);
        if (!clamp_logged) {
            clamp_logged = true;
            LOG_WARNING(Render_Vulkan, "SEAM 2/3 need FRITCHII_MEMSTAGE_TICKS=1 (or 2); running "
                                       "SEAM 1");
        }
        value = 1;
    }
    const bool on = value != 0;
    // A held packet never outlives its draw or dispatch; a safety net, as in BeginDraw.
    PlayRest();
    if (value != seam_level.load(std::memory_order_relaxed)) {
        if ((value >= 3) != ValuesByValue()) {
            LOG_INFO(Render_Vulkan, "Memory seam (EXP123={}): {}", value,
                     value >= 3 ? "handle values, the QMD and null pointer slots by value"
                                : "the back reads handles, the QMD and null pointer slots itself");
        }
        seam_level.store(value, std::memory_order_relaxed);
    }
    RetireDrawValues();
    if (dispatch_open) {
        dispatch_open = false;
        Bump(value_counters.qmds_unused);
        RetireDispatchValues();
    }
    if (OwnedHere() && pending) {
        // Nothing should be pending between frames: a site that misses its play.
        Note(SeamHook::Use::Flush, std::source_location::current(), false);
        PlayPending(true);
    }
    if (OwnedHere() && t_fill_depth == 0) {
        owner.store(0, std::memory_order_release);
    }
    // The seam lean: entries run as pushed and nothing reports the bracket's work. Turned off
    // before a SEAM above 0 shows, turned on after SEAM 0 does.
    static const bool stats_env = std::getenv("FRITCHII_MEMSTAGE_SEAM_STATS") != nullptr;
    const bool lean_now = value == 0 && !ticks && !stats_env;
    if (!lean_now) {
        lean.store(false, std::memory_order_relaxed);
    }
    if (value != seam_value.load(std::memory_order_relaxed)) {
        seam_value.store(value, std::memory_order_relaxed);
        seam_on.store(on, std::memory_order_relaxed);
        player.ForgetBindings();
        // At 2 and 3 always: a held fill keeps its read ranges for the texture write check.
        VideoCommon::MemReadLog::g_on.store(value >= 2, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "Memory seam (EXP123): {}{}",
                 value >= 2 ? fmt::format("{}, one fill per draw and dispatch before the texture "
                                          "work{}",
                                          value, value >= 3 ? ", values by value" : "")
                 : on       ? std::string{"1, recorder and player"}
                            : std::string{"0, entries run as pushed"},
                 value >= 2 && read_log_every > 0
                     ? fmt::format(" (FRITCHII_MEMSTAGE_READ_LOG: every {}th packet's reads "
                                   "compared)",
                                   read_log_every)
                     : std::string{});
    }
    if (lean_now) {
        lean.store(true, std::memory_order_relaxed);
    }
    TickThread(value);
    if (on || stats_env) {
        LogStats();
    }
    if (ticks) {
        ticks->LogStats();
    }
}

// FRITCHII_MEMSTAGE_THREAD (live, read here at a frame end, where both threads are level) and
// FRITCHII_MEMSTAGE_FULL_BINDS (live). THREAD = 1 needs the early fill (SEAM >= 2, which needs
// packet ticks) and the buffer cache's own allocator (FRITCHII_MEMSTAGE_SPLIT_ALLOC); without them
// it runs as 0, logged once. The link refuses too when GPUMem does not run (no GPU thread split).
// THREAD = 2 (piece 6b) needs the same, and EXP53 off (with it every conditional block is a level
// entry the registers do not decide); without it, it runs as 1, logged once. It forces full vertex
// binds (plan 4.2: the flags the state tracker raises on the back are dropped at each lend).
// FRITCHII_MEMSTAGE_AHEAD (live, here too) is step 7's window (memory_stage_link.cpp); 0 is 6b.
void MemStage::TickThread(int seam) {
    namespace Debug = VideoCore::Nx2Debug;
    const int wanted = Debug::SwitchValue(Debug::g_memstage_thread, "FRITCHII_MEMSTAGE_THREAD");
    u32 mode = wanted >= 2 ? 2U : wanted > 0 ? 1U : 0U;
    if (mode != 0 && VideoCommon::OneFollower::On()) {
        // One follower (FRITCHII_ONEF=1) replaces GPUMem: the stage stays off.
        if (!thread_clamp_logged) {
            thread_clamp_logged = true;
            LOG_WARNING(Render_Vulkan, "FRITCHII_MEMSTAGE_THREAD={} with FRITCHII_ONEF=1; running 0",
                        wanted);
        }
        mode = 0;
    }
    if (mode != 0 && (seam < 2 || !ticks || !player.device.HasSplitAllocators())) {
        if (!thread_clamp_logged) {
            thread_clamp_logged = true;
            LOG_WARNING(Render_Vulkan, "FRITCHII_MEMSTAGE_THREAD={} needs {}; running 0", wanted,
                        seam < 2 || !ticks
                            ? "EXP123 >= 2 (with FRITCHII_MEMSTAGE_TICKS >= 1)"
                            : "the buffer cache's own allocator (FRITCHII_MEMSTAGE_SPLIT_ALLOC)");
        }
        mode = 0;
    }
    if (mode == 2 && seam < 3) {
        // Finding 1 of the 6b review: a served draw's handles must come in its packet (values by
        // value); the back would read them after the stage wrote later constant buffer words.
        if (!ring_seam_logged) {
            ring_seam_logged = true;
            LOG_WARNING(Render_Vulkan, "FRITCHII_MEMSTAGE_THREAD=2 needs EXP123=3 (values by "
                                       "value); running 1");
        }
        mode = 1;
    }
    if (mode == 2 && Debug::Experiment(53) != 0) {
        if (!ring_clamp_logged) {
            ring_clamp_logged = true;
            LOG_WARNING(Render_Vulkan, "FRITCHII_MEMSTAGE_THREAD=2 needs EDEN_NX2_EXP53 off "
                                       "(conditional rendering decided on the front); running 1");
        }
        mode = 1;
    }
    Link& link = Link::Get();
    link.SetMode(mode);
    // Step 7's window (live, here at a frame end, a level run: GPUMem is parked on it and reads
    // the new value at its next entry). Packets GPUMem may fill ahead of the back's take; 0 is
    // 6b's level mode, 65,536 and above unlimited (the harness sends 2147483647).
    const int ahead = Debug::SwitchValue(Debug::g_memstage_ahead, "FRITCHII_MEMSTAGE_AHEAD");
    // EDEN_NX2_RUNAHEAD applies due CPU-write chunks on the back before each entry: GPUMem would
    // fill those entries before the chunks are applied. The window runs 0 while it is on. The
    // same with the delta path on at boot (EXP118, or its count): a delta draw holds the buffer
    // cache's lock through Configure, where a read flush is a served call (StageContendedHook
    // covers that too; this keeps the case out of the window altogether).
    const char* const blocked =
        VideoCommon::Runahead::CurrentMode() != VideoCommon::Runahead::Mode::Off
            ? "EDEN_NX2_RUNAHEAD"
        : VideoCommon::Nx2FastDrawCount::g_on ? "the delta path (EXP118)"
                                              : nullptr;
    link.SetAhead(link.Mode() == 2 && ahead > 0 ? static_cast<u32>(ahead) : 0U, blocked);
    // List draws served by GPUMem (live, default on): only while the window is open
    // (Link::ListDrawsServed); level mode and THREAD 0/1 never serve them.
    link.SetListDraws(
        Debug::SwitchValue(Debug::g_memstage_list_draws, "FRITCHII_MEMSTAGE_LIST_DRAWS", 1) > 0);
    // GPUMem's fast entry loop (live, default on): only while the window is open too
    // (Link::FastLoop); level mode and THREAD 0/1 keep the loop as it was.
    link.SetFastLoop(
        Debug::SwitchValue(Debug::g_memstage_fast_loop, "FRITCHII_MEMSTAGE_FAST_LOOP", 1) > 0);
    // GPUMem's short entry path (live, default off): only while the window is open too
    // (Link::EntryFast).
    link.SetEntryFast(
        Debug::SwitchValue(Debug::g_memstage_entry_fast, "FRITCHII_MEMSTAGE_ENTRY_FAST", 0) > 0);
    // GPUMem's runs of constant buffer words (live, default off): only while the window is open
    // too (Link::CbRuns).
    link.SetCbRuns(
        Debug::SwitchValue(Debug::g_memstage_cb_runs, "FRITCHII_MEMSTAGE_CB_RUNS", 0) > 0);
    // Gathers run at their entry with GPUMem level instead of a run (live, default off): only
    // while the window is open (Link::GatherServed); level mode and THREAD 0/1 never.
    link.SetGather(
        Debug::SwitchValue(Debug::g_memstage_gather, "FRITCHII_MEMSTAGE_GATHER", 0) > 0);
    // The puller's semaphore releases the same way (live, default off; the window open only).
    link.SetRelease(
        Debug::SwitchValue(Debug::g_memstage_release, "FRITCHII_MEMSTAGE_RELEASE", 0) > 0);
    // Passed gathers and waits for idle (live, default off; the window open only; bit 1).
    link.SetPass(static_cast<u32>(
        (std::max)(Debug::SwitchValue(Debug::g_memstage_pass, "FRITCHII_MEMSTAGE_PASS", 0), 0)));
    // GPU-written marks made ahead of the back (live, default off; buffer_cache.h, "Marks ahead
    // of the back"): they run only with the window open. Here both threads are level, so the
    // cache's list is empty when it turns on or off.
    {
        const bool marks = Debug::SwitchValue(Debug::g_memstage_marks_ahead,
                                              "FRITCHII_MEMSTAGE_MARKS_AHEAD", 0) > 0;
        if (cache != nullptr) {
            cache->SetMarksAhead(marks && link.Mode() == 2 && Link::Ahead() != 0, marks);
        }
    }
    // The window's verify (its own-write ring and the image check) only with a verify on.
    Link::SetVerify(read_log_every > 0 || Link::RingVerify());
    ring_read_log.store(read_log_every > 0 && Link::Ahead() != 0, std::memory_order_relaxed);
    // Stamps per thread while the window is open and the ring runs (here both threads are level;
    // the ring's end turns it off, ResetStageSide). Otherwise `open`, as before.
    if (ticks) {
        ticks->SetPerThread(Link::RingActive() && Link::Ahead() != 0);
    }
    // One follower: on unless FRITCHII_MEMSTAGE_FULL_BINDS=0. R starts command buffers E cannot
    // see (posted fences, the E split), and the cmdbuf invalidation's vertex binding flags are
    // E's (one_follower_view.cpp): the unchanged bindings were never bound in the new command
    // buffer. AMD then reads the old command buffer's vertex table: a GPU page fault (Radeon GPU
    // Detective, 2026-11-19). The player drops the bindings still bound (VertexEpoch).
    // (With one follower the player now binds the remembered bindings again itself in a new
    // command buffer, MemPlayer::EnsureVertexBindings: full binds stay opt-in.)
    const bool full =
        Debug::SwitchValue(Debug::g_memstage_full_binds, "FRITCHII_MEMSTAGE_FULL_BINDS") > 0 ||
        link.Mode() == 2;
    if (full != Link::FullVertexBinds()) {
        Link::SetFullVertexBinds(full);
        LOG_INFO(Render_Vulkan, "Memory seam full vertex bindings (FRITCHII_MEMSTAGE_FULL_BINDS={}): {}",
                 full ? 1 : 0,
                 full ? "every enabled vertex binding is bound on every draw; the player drops the "
                        "ones still bound"
                      : "vertex bindings are bound when dirty");
    }
    // The back is the cache's user at a frame end: the buffer allocator's numbers, fresh.
    player.device.PublishBufferHeapBudgets();
    link.LogStats();
}

void MemStage::LogStats() {
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    if (last_log_ns == 0) {
        last_log_ns = now;
        Snapshot(reported);
        LogValueStats(0.0, true);
        return;
    }
    if (now - last_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - last_log_ns) / 1e9;
    last_log_ns = now;
    Counts current;
    Snapshot(current);
    const auto delta = [&](u64 Counts::*member) { return current.*member - reported.*member; };
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };

    std::string kinds;
    std::string outside;
    for (size_t i = 0; i < CMD_NAMES.size(); ++i) {
        if (const u64 n = current.entries[i] - reported.entries[i]; n != 0) {
            kinds += fmt::format("{}{} {:.0f}", kinds.empty() ? "" : ", ", CMD_NAMES[i], rate(n));
        }
        if (const u64 n = current.outside[i] - reported.outside[i]; n != 0) {
            outside +=
                fmt::format("{}{} {:.0f}", outside.empty() ? "" : ", ", CMD_NAMES[i], rate(n));
        }
    }
    std::string sites;
    u64 draw_entries = 0;
    u64 dispatch_entries = 0;
    for (size_t i = 0; i < SITE_NAMES.size(); ++i) {
        const u64 fills = current.site_fills[i] - reported.site_fills[i];
        const u64 entries = current.site_entries[i] - reported.site_entries[i];
        switch (static_cast<Site>(i)) {
        case Site::Geometry:
        case Site::Stage:
        case Site::DrawCount:
        case Site::DrawIndirect:
        case Site::ListArgs:
        case Site::Draw:
            draw_entries += entries;
            break;
        case Site::Dispatch:
        case Site::DispatchBefore:
        case Site::DispatchIndirect:
        case Site::DispatchAfter:
            dispatch_entries += entries;
            break;
        default:
            break;
        }
        if (fills != 0) {
            sites += fmt::format("{}{} {:.0f} ({:.2f} entries each)", sites.empty() ? "" : ", ",
                                 SITE_NAMES[i], rate(fills),
                                 static_cast<double>(entries) / static_cast<double>(fills));
        }
    }
    std::string syncs;
    for (size_t i = 0; i < SYNC_NAMES.size(); ++i) {
        if (const u64 n = current.syncs[i] - reported.syncs[i]; n != 0) {
            syncs += fmt::format("{}{} {:.0f}", syncs.empty() ? "" : ", ", SYNC_NAMES[i], rate(n));
        }
    }
    const u64 draws = delta(&Counts::draws);
    const u64 dispatches = delta(&Counts::dispatches);
    const size_t fence = static_cast<size_t>(Site::Fence);
    const u64 fence_fills = current.site_fills[fence] - reported.site_fills[fence];
    const u64 fence_entries = current.site_entries[fence] - reported.site_entries[fence];
    LOG_INFO(Render_Vulkan,
             "Memory seam (EXP123={}) {:.0f} s: draws {:.0f}/s ({:.2f} entries each), "
             "dispatches {:.0f}/s ({:.2f} each), fence commits {:.0f}/s ({:.2f} each); "
             "deferred {:.0f}/s, plays {:.0f}/s; around the packet (5.1) {}, late plays (5.2) {}, "
             "scheduler uses not through the clock {}{}",
             seam_value.load(std::memory_order_relaxed), seconds, rate(draws),
             draws != 0 ? static_cast<double>(draw_entries) / static_cast<double>(draws) : 0.0,
             rate(dispatches),
             dispatches != 0 ? static_cast<double>(dispatch_entries) / static_cast<double>(dispatches)
                             : 0.0,
             rate(fence_fills),
             fence_fills != 0 ? static_cast<double>(fence_entries) / static_cast<double>(fence_fills)
                              : 0.0,
             rate(delta(&Counts::deferred)), rate(delta(&Counts::plays)),
             delta(&Counts::around), delta(&Counts::late), delta(&Counts::unclocked),
             delta(&Counts::seam_clamped) != 0
                 ? fmt::format("; SEAM 2/3 run as 1 without packet ticks: {} frames",
                               delta(&Counts::seam_clamped))
                 : std::string{});
    LOG_INFO(Render_Vulkan, "  memory seam entries/s by kind: {}; texel buffer descriptors {:.0f}/s",
             kinds.empty() ? "none" : kinds, rate(delta(&Counts::texel)));
    LOG_INFO(Render_Vulkan, "  memory seam fills/s by site: {}", sites.empty() ? "none" : sites);
    LOG_INFO(Render_Vulkan, "  memory seam syncs/s: {}; entries/s run outside any fill: {}",
             syncs.empty() ? "none" : syncs, outside.empty() ? "none" : outside);
    if (seam_value.load(std::memory_order_relaxed) >= 2) {
        std::string diffs;
        u64 diff_total = 0;
        for (size_t i = 0; i < READ_KIND_NAMES.size(); ++i) {
            if (const u64 n = current.read_diffs[i] - reported.read_diffs[i]; n != 0) {
                diffs += fmt::format("{}{} {}", diffs.empty() ? "" : ", ", READ_KIND_NAMES[i], n);
                diff_total += n;
            }
        }
        const u64 held_draws = delta(&Counts::held_draws);
        const u64 held_dispatches = delta(&Counts::held_dispatches);
        const u64 held_packets = held_draws + held_dispatches;
        LOG_INFO(Render_Vulkan,
                 "  memory seam 2b: held draws {:.0f}/s, late fill draws {:.0f}/s (texel "
                 "buffers), held dispatches {:.0f}/s, late fill dispatches {:.0f}/s, not held "
                 "(arena busy) {}; flushed while held (5.5) {}; texture writes while held {} "
                 "(over fill reads {}, must be 0); rest plays {} ({} descriptors dropped); "
                 "descriptor form mismatches {}; fill {:.3f} ms/s ({:.2f} us a packet), play "
                 "{:.3f} ms/s (inclusive, with EDEN_NX2_GPU_PROFILE)",
                 rate(held_draws), rate(delta(&Counts::late_draws)), rate(held_dispatches),
                 rate(delta(&Counts::late_dispatches)), delta(&Counts::unheld),
                 delta(&Counts::held_flushed), delta(&Counts::texture_writes),
                 delta(&Counts::texture_overlaps), delta(&Counts::rest_plays),
                 delta(&Counts::rest_descriptors), delta(&Counts::form_mismatch),
                 static_cast<double>(delta(&Counts::fill_ns)) / 1e6 / seconds,
                 held_packets != 0 ? static_cast<double>(delta(&Counts::fill_ns)) / 1e3 /
                                         static_cast<double>(held_packets)
                                   : 0.0,
                 static_cast<double>(delta(&Counts::play_ns)) / 1e6 / seconds);
        if (read_log_every > 0) {
            std::string compared;
            for (size_t i = 0; i < READ_KIND_NAMES.size(); ++i) {
                if (const u64 n = current.read_kind_records[i] - reported.read_kind_records[i];
                    n != 0) {
                    compared += fmt::format("{}{} {}", compared.empty() ? "" : ", ",
                                            READ_KIND_NAMES[i], n);
                }
            }
            // Step 7: GPUMem's writes kept out of the emulator's write ring (in its own ring), and
            // the cache's CPU-written marks noted for the compare.
            const u64 quiet = Common::WriteRing::g_quiet.load(std::memory_order_relaxed);
            const u64 marks = cpu_mark_count.load(std::memory_order_relaxed);
            LOG_INFO(Render_Vulkan,
                     "  memory seam read log (FRITCHII_MEMSTAGE_READ_LOG={}): {} packets, {} reads "
                     "compared{}{}{}, {} not logged (budget); differences (5.3) {}{}{}; ring "
                     "packets compared at the take (step 7) {} ({} with a CPU-written mark since "
                     "the fill), left out as GPUMem's own later writes {}; GPUMem's writes kept "
                     "out of the write ring {}; CPU-written marks noted {}",
                     read_log_every, delta(&Counts::read_packets), delta(&Counts::read_records),
                     compared.empty() ? "" : " (", compared, compared.empty() ? "" : ")",
                     delta(&Counts::read_skipped), diff_total, diffs.empty() ? "" : ": ", diffs,
                     delta(&Counts::ring_reads), delta(&Counts::ring_reads_marked),
                     delta(&Counts::diff_stage_own), quiet - reported_quiet,
                     marks - reported_marks);
            reported_quiet = quiet;
            reported_marks = marks;
            const auto by_class = [&](DiffClass kind) {
                const size_t i = static_cast<size_t>(kind);
                return current.diff_classes[i] - reported.diff_classes[i];
            };
            std::string sites;
            for (size_t i = 0; i < DIFF_SITES; ++i) {
                const char* const name = diff_site_table[i].name.load(std::memory_order_acquire);
                if (name == nullptr) {
                    break;
                }
                if (const u64 n = current.diff_sites[i] - reported.diff_sites[i]; n != 0) {
                    sites += fmt::format(
                        "{}{} {}{}", sites.empty() ? "" : ", ", name, n,
                        diff_site_table[i].landing.load(std::memory_order_relaxed) ? " (landing)"
                                                                                   : "");
                }
            }
            if (const u64 n = delta(&Counts::diff_sites_other); n != 0) {
                sites += fmt::format("{}other sites {}", sites.empty() ? "" : ", ", n);
            }
            const u64 landings = by_class(DiffClass::Landing);
            const u64 emulator = by_class(DiffClass::EmulatorWrite);
            const u64 unclassified = by_class(DiffClass::Unclassified);
            const u64 own = delta(&Counts::diff_own);
            const u64 taken = by_class(DiffClass::MarkTaken);
            const u64 untracked = by_class(DiffClass::Untracked);
            const u64 wrapped = delta(&Counts::ring_wrapped);
            LOG_INFO(Render_Vulkan,
                     "  memory seam read log classes: emulator write {}{}{}{}, game write marked "
                     "{}, game write mark already taken {}, untracked read changed {}, gather on "
                     "the back inside Configure {}, unclassified {}; over-read {}, changed inside "
                     "a range the packet binds {}; ring wrapped {}; on the holding thread: "
                     "gathers applied {}, FlushRegion texture {} query {} buffer {}; async GPU "
                     "completions over-read {}",
                     landings + emulator, sites.empty() ? "" : " (", sites,
                     sites.empty() ? "" : ")", by_class(DiffClass::GameMarked), taken, untracked,
                     by_class(DiffClass::BackGather), unclassified, delta(&Counts::diff_over), own,
                     wrapped, delta(&Counts::gathers_held),
                     current.flush_regions_held[0] - reported.flush_regions_held[0],
                     current.flush_regions_held[1] - reported.flush_regions_held[1],
                     current.flush_regions_held[2] - reported.flush_regions_held[2],
                     by_class(DiffClass::AsyncOverRead));
            // Fails on any unexplained difference, and on a wrapped ring (the run is not valid:
            // the writers since some fills are unknown).
            const bool gate_pass = emulator + unclassified == 0 && wrapped == 0;
            LOG_INFO(Render_Vulkan,
                     "  read log gate {}: unexplained {} (emulator writes that are not download "
                     "landings {}, unclassified {}); ring wrapped {}{}; mark already taken {} "
                     "(tracked reads); untracked read changed {}; changed inside a range the "
                     "packet binds {}; async GPU completions over-read {} (reported)",
                     gate_pass ? "passed" : "FAILED", emulator + unclassified, emulator,
                     unclassified, wrapped, wrapped != 0 ? " (run not valid)" : "", taken,
                     untracked, own, by_class(DiffClass::AsyncOverRead));
        }
    }
    const std::array<u64, 2> vertex_runs{
        player.vertex_runs_recorded.load(std::memory_order_relaxed),
        player.vertex_runs_dropped.load(std::memory_order_relaxed)};
    if (Link::FullVertexBinds()) {
        LOG_INFO(Render_Vulkan,
                 "  memory seam full vertex bindings (FRITCHII_MEMSTAGE_FULL_BINDS=1): vertex bind "
                 "runs recorded {:.0f}/s, dropped as still bound {:.0f}/s",
                 rate(vertex_runs[0] - reported_vertex_runs[0]),
                 rate(vertex_runs[1] - reported_vertex_runs[1]));
    }
    {
        static u64 reported_rebound = 0;
        static u64 reported_forgotten = 0;
        const u64 rebound = player.vertex_runs_rebound.load(std::memory_order_relaxed);
        const u64 forgotten = player.vertex_forgotten.load(std::memory_order_relaxed);
        if (rebound != reported_rebound || forgotten != reported_forgotten) {
            LOG_INFO(Render_Vulkan,
                     "  memory seam vertex bindings bound again in a new command buffer "
                     "(EnsureVertexBindings): {:.0f}/s; forgotten (stream left without a "
                     "binding, never bound again) {:.0f}/s",
                     rate(rebound - reported_rebound), rate(forgotten - reported_forgotten));
        }
        reported_rebound = rebound;
        reported_forgotten = forgotten;
    }
    reported_vertex_runs = vertex_runs;
    reported = current;
    LogValueStats(seconds, false);
}

void MemStage::Snapshot(Counts& out) const {
    for (size_t i = 0; i < out.entries.size(); ++i) {
        out.entries[i] = counters.entries[i].load(std::memory_order_relaxed);
        out.outside[i] = counters.outside[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < out.site_fills.size(); ++i) {
        out.site_fills[i] = counters.site_fills[i].load(std::memory_order_relaxed);
        out.site_entries[i] = counters.site_entries[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < out.syncs.size(); ++i) {
        out.syncs[i] = counters.syncs[i].load(std::memory_order_relaxed);
    }
    out.draws = counters.draws.load(std::memory_order_relaxed);
    out.dispatches = counters.dispatches.load(std::memory_order_relaxed);
    out.deferred = counters.deferred.load(std::memory_order_relaxed);
    out.plays = counters.plays.load(std::memory_order_relaxed);
    out.around = counters.around.load(std::memory_order_relaxed);
    out.late = counters.late.load(std::memory_order_relaxed);
    out.unclocked = counters.unclocked.load(std::memory_order_relaxed);
    out.texel = counters.texel.load(std::memory_order_relaxed);
    out.held_draws = counters.held_draws.load(std::memory_order_relaxed);
    out.late_draws = counters.late_draws.load(std::memory_order_relaxed);
    out.held_dispatches = counters.held_dispatches.load(std::memory_order_relaxed);
    out.late_dispatches = counters.late_dispatches.load(std::memory_order_relaxed);
    out.unheld = counters.unheld.load(std::memory_order_relaxed);
    out.held_flushed = counters.held_flushed.load(std::memory_order_relaxed);
    out.seam_clamped = counters.seam_clamped.load(std::memory_order_relaxed);
    out.texture_writes = counters.texture_writes.load(std::memory_order_relaxed);
    out.texture_overlaps = counters.texture_overlaps.load(std::memory_order_relaxed);
    out.rest_plays = counters.rest_plays.load(std::memory_order_relaxed);
    out.rest_descriptors = counters.rest_descriptors.load(std::memory_order_relaxed);
    out.form_mismatch = counters.form_mismatch.load(std::memory_order_relaxed);
    out.read_packets = counters.read_packets.load(std::memory_order_relaxed);
    out.read_records = counters.read_records.load(std::memory_order_relaxed);
    out.read_skipped = counters.read_skipped.load(std::memory_order_relaxed);
    out.fill_ns = counters.fill_ns.load(std::memory_order_relaxed);
    out.play_ns = counters.play_ns.load(std::memory_order_relaxed);
    for (size_t i = 0; i < out.read_diffs.size(); ++i) {
        out.read_diffs[i] = counters.read_diffs[i].load(std::memory_order_relaxed);
        out.read_kind_records[i] = counters.read_kind_records[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < out.diff_classes.size(); ++i) {
        out.diff_classes[i] = counters.diff_classes[i].load(std::memory_order_relaxed);
    }
    out.diff_own = counters.diff_own.load(std::memory_order_relaxed);
    out.diff_over = counters.diff_over.load(std::memory_order_relaxed);
    out.diff_stage_own = counters.diff_stage_own.load(std::memory_order_relaxed);
    out.ring_reads = counters.ring_reads.load(std::memory_order_relaxed);
    out.ring_reads_marked = counters.ring_reads_marked.load(std::memory_order_relaxed);
    out.ring_wrapped = counters.ring_wrapped.load(std::memory_order_relaxed);
    out.gathers_held = counters.gathers_held.load(std::memory_order_relaxed);
    for (size_t i = 0; i < out.flush_regions_held.size(); ++i) {
        out.flush_regions_held[i] = counters.flush_regions_held[i].load(std::memory_order_relaxed);
    }
    for (size_t i = 0; i < out.diff_sites.size(); ++i) {
        out.diff_sites[i] = diff_site_table[i].count.load(std::memory_order_relaxed);
    }
    out.diff_sites_other = counters.diff_sites_other.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------
// SeamClock. With FRITCHII_MEMSTAGE_TICKS = 0 (or no stage), Now, IsFree and KnownGpu read the
// scheduler's master semaphore, exactly what Scheduler::CurrentTick and IsFree read; with 1 or 2
// they ask SeamTicks (below). The syncs play the pending entries first.

SeamClock::SeamClock(Scheduler& scheduler_, MemStage* stage_)
    : scheduler{&scheduler_}, master{&scheduler_.GetMasterSemaphore()}, stage{stage_},
      ticks{stage_ != nullptr ? stage_->Ticks() : nullptr} {}

// On GPUMem (FRITCHII_MEMSTAGE_THREAD = 1) each sync is served by the back (MemStage::OnBack).
void SeamClock::Finish(std::source_location location) {
    MemStage::OnBack([&] {
        MemStage::SyncScope sync{stage, MemStage::Sync::Finish};
        scheduler->Finish({}, {}, location);
    });
}

void SeamClock::Wait(u64 stamp) {
    MemStage::OnBack([&] {
        MemStage::SyncScope sync{stage, MemStage::Sync::Wait};
        // Translated after the scope played the pending entries: a number still unpublished then
        // is recorded in the current tick, which Scheduler::Wait flushes (and so publishes).
        scheduler->Wait(ticks != nullptr ? ticks->WaitTickOf(stamp) : stamp);
    });
}

void SeamClock::WaitTick(u64 tick) {
    MemStage::OnBack([&] {
        MemStage::SyncScope sync{stage, MemStage::Sync::Wait};
        scheduler->Wait(tick);
    });
}

u64 SeamClock::SubmitForHostRead(std::source_location location) {
    u64 result{};
    MemStage::OnBack([&] {
        MemStage::SyncScope sync{stage, MemStage::Sync::SubmitForHostRead};
        scheduler->RequestOutsideRenderPassOperationContext(location);
        scheduler->Record(HostReadBarrier);
        if (ticks == nullptr) {
            result = scheduler->Flush({}, {}, location);
            return;
        }
        // Nothing is pending after the scope: the number current now is the one this flush
        // closes, and the stamp of the tick it submits.
        const u64 stamp = ticks->Now();
        const u64 tick = scheduler->Flush({}, {}, location);
        ticks->CheckHostRead(stamp, tick);
        result = stamp;
    });
    return result;
}

// ---------------------------------------------------------------------------------------------
// Packet ticks: FRITCHII_MEMSTAGE_TICKS >= 1 (boot), fable_memstage_step2_spec.md 7.2, build
// step 4.
//
// A stamp is a packet number, not a scheduler tick. The stage counts its packets up (a draw, a
// dispatch and an operation packet each take the next number: BeginPacket), and every stamp the
// buffer cache side takes (SeamClock::Now: buffer usage, staging and stream regions, recycled
// buffers, write ticks, probes, list page copies, deleted buffers) is the number current then.
// The thread that records publishes, at the end of every scheduler tick, "every number up to N
// was recorded in tick K" (Close, called by Scheduler::SubmitExecution right before the tick
// advances). A stamp is free when the tick its number was published with is complete.
//
// The published table is a short ring of (N, K), in order, for the ticks not yet seen complete;
// `free_number` is the highest N whose K is known complete. IsFree is a load or two in the
// common cases (at or below free_number: free; above the last published N: not free) and
// otherwise advances free_number to the master semaphore's known GPU tick under the lock. So it
// never asks the GPU more than Scheduler::IsFree did (it does not refresh).
//
// What N a close publishes. Entries the stage filled and the player has not played are not
// recorded yet: their number (`unplayed_from`, set when the first entry starts waiting) and every
// later one stay open, and are published with a later tick, the one their play lands in. That is
// why SEAM 2 and 3 need packet ticks (and 2b has no tick guard): a staging buffer stamped at an
// early fill is not free before the tick its copy is recorded in.
// Otherwise the close takes the current number and moves the stage on to the next, so a stamp
// taken after the tick ended is never published with it. On one thread each number then lives
// in exactly one tick, and every answer is today's: IsFree(N) == IsFree(the tick N was handed
// out in).
//
// Step 6 (the stage on its own thread, ahead of the GPU thread) keeps the stamps, the ring and
// the reads as they are. Only Close changes: the GPU thread publishes the last packet it has
// played and never moves `open` (the stage alone does). Waits on a stamp not yet published
// (TickOf falls back to the current tick, right only when the two are level) become a join:
// the stage asks the GPU thread to play up to that packet and flush.
//
// Packet ticks with the stage ahead (FRITCHII_MEMSTAGE_THREAD = 2, step 7's window). GPUMem fills
// while the back records, so one `unplayed_from` written by both would lose the lower mark (the
// back clearing it from a stale look at ring B while GPUMem pushes). While the ring runs each side
// writes only its own mark: the back's (`unplayed_from`: the packet it holds, its own fill in a
// level run, its pending entries) and GPUMem's (`stage_unplayed`: the draw it fills, from the
// number BeginDraw opens under the close's lock (BeginStagePacket) to the push, and its pending
// entries). Close, on the recording thread, takes the lowest of the two and of ring B's oldest
// packet, which it reads itself after GPUMem's mark: a push puts the packet in ring B before the
// mark clears. So every number a packet in flight carries (in GPUMem's fill, in ring B, held by
// the back) stays open until its entries are recorded: its buffers, staging and stream regions
// outlive it. At AHEAD = 0 the two threads take turns and the three marks give the old answer.
//
// Stamps per thread (step 7, the window open and the ring running; Fable's split answer). Now()
// returned `open`, the newest number anyone opened: on one thread that is the caller's packet,
// with two it is not. GPUMem stamping with `open` while it held no mark (before its draw's
// number, between packets, in an operation opened with a plain BeginPacket) let a close by the
// back publish that number with the tick being closed: the use was freed a tick early. So each
// thread stamps with its own packet (SeamTicks::Mine): GPUMem with the draw it fills (BeginDraw
// opens it before anything in StageFillDraw can stamp) or the operation it runs (opened under
// the close's lock with GPUMem's mark, stage_op_from), and with `open` only between them; the
// back with the packet it holds or fills, and after it with the last one while that is not
// published (a host read then waits for its own tick, not the tick of GPUMem's newest packet,
// which was the "late" of the AHEAD 8 verify). Off (level mode, the ring off): `open`, as before.
//
// Users that compare stamps for equality (the stream uniform reuse and its keeps: "same tick, so
// the stream copy is still there") take Epoch(): the number of closes. A copy made at epoch E
// carries a number not yet published; its stream region is only recycled once that number is
// published and complete, which is a later close, so "still epoch E" keeps meaning "not
// recycled" in step 6. On one thread there is one close per tick and the epoch is the tick.
//
// FRITCHII_MEMSTAGE_TICKS = 2 checks every answer against the scheduler's: each stamp handed out
// notes the tick current then (a ring keyed by number, 2^18 numbers back), and IsFree, the
// stream regions' known-GPU test, the waits, the host read flushes and the epoch compare with
// what that tick says. The "Memory seam ticks" line every 10 s:
//   early:     the packet answer frees (or waits for) less than the tick answer. Unsafe; must be 0.
//   late:      the packet answer holds longer. Must be 0 on one thread.
//   carried:   numbers a tick ended with unplayed (an early fill across a flush): skipped, the
//              packet answer is the right one there (the tick answer would free too early).
//   split:     a number handed out in two ticks without a carry (a tick ended without a close).
//              Must be 0.
//   unchecked: stamps older than the ring.
//   (step 7) splits that raced the tick advance: a split by one tick of a number above the
//              `open` the close of the first tick saw (a stamp taken on the other thread between
//              that close and the tick's advance). Reported, not a failure; the rest of the
//              splits stay must be 0. Carried ranges are kept until older than the stamp ring.
//   (step 7) stamps on a number already published: a stamp whose number a close published
//              before it was taken (its use would be freed early). Must be 0.
//   epoch off: Epoch() differed from the scheduler's tick. Must be 0. (With step 7's window,
//              GPUMem's reads one apart from the tick race a close, which bumps the epoch just
//              before the tick advances: counted apart as "GPUMem's epoch reads racing a close".)
// Deleted buffers held past their frames (buffer_cache.h's ring waits for their stamp as well)
// are counted with 1 and 2; on one thread they must be 0.

namespace {
constexpr u64 StampSlots = u64{1} << 18;
constexpr std::array<const char*, 4> CHECK_NAMES{"is free", "stream known", "wait", "host read"};

thread_local u64 t_known_tick = 0;
std::atomic<u32> g_ticks_logged{0};

u64 SteadyNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}
} // Anonymous namespace

SeamTicks::SeamTicks(MasterSemaphore& master_, bool verify_) : master{&master_}, verify{verify_} {
    epoch.store(master->CurrentTick(), std::memory_order_release);
    if (verify) {
        slots = std::make_unique<StampSlot[]>(StampSlots);
    }
}

SeamTicks::~SeamTicks() = default;

u64 SeamTicks::Advance(u64 known) {
    std::scoped_lock lock{mutex};
    return AdvanceLocked(known);
}

u64 SeamTicks::AdvanceLocked(u64 known) {
    u64 free = free_number.load(std::memory_order_relaxed);
    while (!ring.empty() && ring.front().tick <= known) {
        free = ring.front().last;
        ring.pop_front();
    }
    // free_number before advanced_gpu: a reader that sees the new known GPU tick there sees the
    // numbers it frees.
    free_number.store(free, std::memory_order_release);
    if (known > advanced_gpu.load(std::memory_order_relaxed)) {
        advanced_gpu.store(known, std::memory_order_release);
    }
    return free;
}

u64 SeamTicks::BeginStagePacket() {
    std::scoped_lock lock{mutex};
    const u64 number = open.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (stage_unplayed.load(std::memory_order_relaxed) == 0) {
        stage_unplayed.store(number, std::memory_order_release);
    }
    return number;
}

void SeamTicks::Close(u64 tick) {
    std::scoped_lock lock{mutex};
    // The back's mark, then GPUMem's, then ring B's oldest packet ("Packet ticks with the stage
    // ahead"): GPUMem's mark before the ring, because a packet it pushes is in ring B before its
    // mark clears. Off THREAD = 2 the last two are 0.
    u64 unplayed = unplayed_from.load(std::memory_order_acquire);
    const auto lower = [&unplayed](u64 number) {
        if (number != 0 && (unplayed == 0 || number < unplayed)) {
            unplayed = number;
        }
    };
    lower(stage_unplayed.load(std::memory_order_acquire));
    if (ring_oldest != nullptr) {
        lower(ring_oldest(ring_context));
    }
    u64 last;
    u64 open_seen;
    if (unplayed != 0) {
        last = unplayed - 1;
        open_seen = open.load(std::memory_order_acquire);
        Bump(counters.carried);
        if (verify) {
            const u64 to = open_seen;
            carried_ranges.push_back(Range{.from = unplayed, .to = to});
            // Dropped by number, once older than the stamp ring (a stamp there is unchecked
            // anyway), not by count: with the window open nearly every close carries.
            while (!carried_ranges.empty() && carried_ranges.front().to + StampSlots < to) {
                carried_ranges.pop_front();
            }
            carried_to.store(to, std::memory_order_release);
        }
    } else {
        last = open.fetch_add(1, std::memory_order_acq_rel);
        open_seen = last;
    }
    if (verify) {
        // The split check's "raced the tick advance": the `open` this close saw, by its tick.
        close_opens.push_back(CloseOpen{.tick = tick, .open = open_seen});
        if (close_opens.size() > 4096) {
            close_opens.pop_front();
        }
    }
    if (last > published.load(std::memory_order_relaxed)) {
        ring.push_back(Published{.last = last, .tick = tick});
        published.store(last, std::memory_order_release);
    }
    epoch.fetch_add(1, std::memory_order_acq_rel);
    Bump(counters.closes);
    AdvanceLocked(master->KnownGpuTick());
    if (const u64 depth = ring.size(); depth > counters.max_depth.load(std::memory_order_relaxed)) {
        counters.max_depth.store(depth, std::memory_order_relaxed);
    }
}

u64 SeamTicks::TickOf(u64 stamp) {
    std::scoped_lock lock{mutex};
    if (stamp <= free_number.load(std::memory_order_relaxed)) {
        // Complete: any complete tick does for a wait or a submitted test.
        return advanced_gpu.load(std::memory_order_relaxed);
    }
    const auto it = std::lower_bound(
        ring.begin(), ring.end(), stamp,
        [](const Published& entry, u64 value) { return entry.last < value; });
    // Not published: still open, so its uses are in the tick being recorded.
    return it != ring.end() ? it->tick : master->CurrentTick();
}

u64 SeamTicks::WaitTickOf(u64 stamp) {
    const u64 tick = TickOf(stamp);
    if (verify) [[unlikely]] {
        u64 old_tick{};
        if (stamp > free_number.load(std::memory_order_acquire) && OldTick(stamp, old_tick)) {
            Bump(counters.checks[static_cast<size_t>(Check::Wait)]);
            if (tick != old_tick) {
                Disagree(Check::Wait, tick < old_tick, stamp, old_tick, tick);
            }
        }
    }
    return tick;
}

void SeamTicks::CheckHostRead(u64 stamp, u64 tick) {
    if (!verify) {
        return;
    }
    const u64 published_tick = TickOf(stamp);
    if (stamp <= free_number.load(std::memory_order_acquire)) {
        return;
    }
    Bump(counters.checks[static_cast<size_t>(Check::HostRead)]);
    if (published_tick != tick) {
        Disagree(Check::HostRead, published_tick < tick, stamp, tick, published_tick);
    }
}

void SeamTicks::NoteHeld(u64 newly_held) noexcept {
    if (newly_held != 0) {
        Bump(counters.held, newly_held);
    }
}

void SeamTicks::NoteStamp(u64 number) noexcept {
    // The real case of a split, caught where it happens (Fable's split answer): a stamp on a
    // number a close published already. Its use is recorded later than that close's tick, so it
    // would be freed early. Must be 0 (it does not depend on ticks).
    if (number != 0 && number <= published.load(std::memory_order_acquire)) [[unlikely]] {
        Bump(counters.stamp_published);
        if (g_ticks_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Render_Vulkan,
                        "Memory seam ticks: a stamp on packet {}, published already (published up "
                        "to {}, open {})",
                        number, published.load(std::memory_order_relaxed),
                        open.load(std::memory_order_relaxed));
        }
    }
    StampSlot& slot = slots[number & (StampSlots - 1)];
    const u64 tick = master->CurrentTick();
    if (slot.number.load(std::memory_order_acquire) != number) {
        slot.first.store(tick, std::memory_order_relaxed);
        slot.last.store(tick, std::memory_order_relaxed);
        slot.number.store(number, std::memory_order_release);
    } else if (slot.last.load(std::memory_order_relaxed) != tick) {
        slot.last.store(tick, std::memory_order_relaxed);
    }
}

bool SeamTicks::OldTick(u64 stamp, u64& tick) {
    if (stamp == 0 || stamp == Never) {
        // Never stamped (free) and the deferred staging buffers' "never free".
        tick = stamp;
        return true;
    }
    const StampSlot& slot = slots[stamp & (StampSlots - 1)];
    if (slot.number.load(std::memory_order_acquire) != stamp) {
        Bump(counters.unchecked);
        return false;
    }
    const u64 first = slot.first.load(std::memory_order_relaxed);
    const u64 last = slot.last.load(std::memory_order_relaxed);
    bool carried = false;
    if (stamp <= carried_to.load(std::memory_order_acquire)) {
        std::scoped_lock lock{mutex};
        carried = std::ranges::any_of(carried_ranges, [stamp](const Range& range) {
            return stamp >= range.from && stamp <= range.to;
        });
    }
    if (carried) {
        Bump(counters.carried_skipped);
        return false;
    }
    if (first != last) {
        // "Raced the tick advance" (reported, not a failure): the stamp was taken on another
        // thread while the back was between a close and the tick's advance, so it read the old
        // tick for a number the close of that tick did not publish (above the `open` it saw).
        if (first + 1 == last) {
            std::scoped_lock lock{mutex};
            const auto close = std::ranges::find_if(
                close_opens, [first](const CloseOpen& entry) { return entry.tick == first; });
            if (close != close_opens.end() && stamp > close->open) {
                Bump(counters.split_raced);
                return false;
            }
        }
        Bump(counters.split);
        if (g_ticks_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Render_Vulkan,
                        "Memory seam ticks: packet {} was handed out in ticks {} and {} without "
                        "a carry (a tick ended without a close)",
                        stamp, first, last);
        }
        return false;
    }
    tick = first;
    return true;
}

bool SeamTicks::IsFreeChecked(u64 stamp) {
    const u64 known_before = master->KnownGpuTick();
    const bool free = IsFreeFast(stamp);
    const u64 known_after = master->KnownGpuTick();
    u64 old_tick{};
    if (!OldTick(stamp, old_tick)) {
        return free;
    }
    Bump(counters.checks[static_cast<size_t>(Check::IsFree)]);
    // The packet answer came from a known GPU tick between the two reads.
    if (free && old_tick > known_after) {
        Disagree(Check::IsFree, true, stamp, old_tick, known_after);
    } else if (!free && old_tick <= known_before) {
        Disagree(Check::IsFree, false, stamp, old_tick, known_before);
    }
    return free;
}

u64 SeamTicks::KnownFreeChecked(u64 known) {
    std::scoped_lock lock{mutex};
    const u64 free = AdvanceLocked(known);
    // The tick that answer stands for (another thread may have advanced past `known`).
    t_known_tick = advanced_gpu.load(std::memory_order_relaxed);
    return free;
}

void SeamTicks::CheckKnown(u64 stamp, bool free) {
    u64 old_tick{};
    if (!OldTick(stamp, old_tick)) {
        return;
    }
    Bump(counters.checks[static_cast<size_t>(Check::Known)]);
    const bool old_free = old_tick <= t_known_tick;
    if (free != old_free) {
        Disagree(Check::Known, free, stamp, old_tick, t_known_tick);
    }
}

void SeamTicks::CheckEpoch(u64 value) {
    const u64 tick = master->CurrentTick();
    if (value != tick && Link::OnStageThread() && value + 1 >= tick && value <= tick + 1) {
        // Step 7: GPUMem reads the epoch while the back closes ticks. Close bumps the epoch just
        // before the scheduler's tick advances (SubmitExecution), and the tick can advance between
        // GPUMem's two reads, so one apart is a read racing a close, not an epoch that is off.
        // (The stream copies' test stays safe: an epoch read a close early only fails a reuse.)
        Bump(counters.epoch_racing);
        return;
    }
    if (value != tick) {
        Bump(counters.epoch_off);
        if (g_ticks_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Render_Vulkan, "Memory seam ticks: epoch {} while the tick is {}", value,
                        tick);
        }
    }
}

void SeamTicks::Disagree(Check kind, bool early, u64 stamp, u64 old_tick, u64 detail) {
    const size_t index = static_cast<size_t>(kind);
    Bump(early ? counters.early[index] : counters.late[index]);
    if (g_ticks_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Vulkan,
                    "Memory seam ticks: {} answer {} for packet {}: tick {} by the scheduler's "
                    "ticks, {} by the packet ticks (open {}, free up to {}, published up to {})",
                    CHECK_NAMES[index], early ? "early" : "late", stamp, old_tick, detail,
                    open.load(std::memory_order_relaxed),
                    free_number.load(std::memory_order_relaxed),
                    published.load(std::memory_order_relaxed));
    }
}

void SeamTicks::Take(Snap& out) const noexcept {
    out.numbers = open.load(std::memory_order_relaxed);
    out.closes = counters.closes.load(std::memory_order_relaxed);
    out.carried = counters.carried.load(std::memory_order_relaxed);
    out.held = counters.held.load(std::memory_order_relaxed);
    for (size_t i = 0; i < out.checks.size(); ++i) {
        out.checks[i] = counters.checks[i].load(std::memory_order_relaxed);
        out.early[i] = counters.early[i].load(std::memory_order_relaxed);
        out.late[i] = counters.late[i].load(std::memory_order_relaxed);
    }
    out.carried_skipped = counters.carried_skipped.load(std::memory_order_relaxed);
    out.unchecked = counters.unchecked.load(std::memory_order_relaxed);
    out.split = counters.split.load(std::memory_order_relaxed);
    out.epoch_off = counters.epoch_off.load(std::memory_order_relaxed);
    out.epoch_racing = counters.epoch_racing.load(std::memory_order_relaxed);
    out.stamp_published = counters.stamp_published.load(std::memory_order_relaxed);
    out.split_raced = counters.split_raced.load(std::memory_order_relaxed);
}

void SeamTicks::LogStats() {
    const u64 now = SteadyNs();
    if (last_log_ns == 0) {
        last_log_ns = now;
        Take(reported);
        return;
    }
    if (now - last_log_ns < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - last_log_ns) / 1e9;
    last_log_ns = now;
    Snap current;
    Take(current);
    const auto rate = [seconds](u64 value) { return static_cast<double>(value) / seconds; };
    const u64 depth = counters.max_depth.exchange(0, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan,
             "Memory seam ticks (FRITCHII_MEMSTAGE_TICKS={}) {:.0f} s: packets {:.0f}/s, tick "
             "closes {:.0f}/s, published ticks waiting at most {}, carried closes {}, deleted "
             "buffers held past their frames {}",
             verify ? 2 : 1, seconds, rate(current.numbers - reported.numbers),
             rate(current.closes - reported.closes), depth, current.carried - reported.carried,
             current.held - reported.held);
    if (verify) {
        std::string checks;
        u64 early = 0;
        u64 late = 0;
        for (size_t i = 0; i < CHECK_NAMES.size(); ++i) {
            const u64 e = current.early[i] - reported.early[i];
            const u64 l = current.late[i] - reported.late[i];
            early += e;
            late += l;
            checks += fmt::format("{}{} {} (early {}, late {})", checks.empty() ? "" : ", ",
                                  CHECK_NAMES[i], current.checks[i] - reported.checks[i], e, l);
        }
        LOG_INFO(Render_Vulkan,
                 "  memory seam ticks check: early {}, late {}; {}; carried skipped {}, split {}, "
                 "unchecked {}, epoch off {}; GPUMem's epoch reads racing a close {}; splits that "
                 "raced the tick advance {} (reported); stamps on a number already published {} "
                 "(must be 0)",
                 early, late, checks, current.carried_skipped - reported.carried_skipped,
                 current.split - reported.split, current.unchecked - reported.unchecked,
                 current.epoch_off - reported.epoch_off,
                 current.epoch_racing - reported.epoch_racing,
                 current.split_raced - reported.split_raced,
                 current.stamp_published - reported.stamp_published);
    }
    reported = current;
}

// ---------------------------------------------------------------------------------------------
// MemPlayer

MemPlayer::MemPlayer(const Device& device_, MemoryAllocator& memory_allocator_,
                     Scheduler& scheduler_, GuestDescriptorQueue& guest_descriptor_queue_,
                     ComputePassDescriptorQueue& compute_pass_descriptor_queue_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      guest_descriptor_queue{guest_descriptor_queue_},
      compute_pass_descriptor_queue{compute_pass_descriptor_queue_} {
    static_assert(NUM_BOUND_VERTEX_BUFFERS >= VideoCommon::NUM_VERTEX_BUFFERS);
}

MemPlayer::~MemPlayer() = default;

void MemPlayer::SetPasses(Uint8Pass* uint8, QuadIndexedPass* quad_indexed,
                          VertexRepackPass* vertex_repack) noexcept {
    uint8_pass = uint8;
    quad_indexed_pass = quad_indexed;
    vertex_repack_pass = vertex_repack;
}

void MemPlayer::ForgetBindings() noexcept {
    bound_index_valid = false;
    last_index_valid = false;
    // Every vertex bind reaches this player whatever the SEAM (entries run at once call the
    // same plays), so the remembered buffers are still the ones bound: only the "still bound"
    // drop stops trusting them. Forgotten (~0), a stream the fill keeps as clean stayed unbound
    // in the next command buffer R starts (review R1 should-fix 3).
    for (BoundVertexBuffer& bound : bound_vertex_buffers) {
        if (bound.epoch != ~u64{0}) {
            bound.epoch = StaleEpoch;
        }
    }
}

// Spec 5.4: the player is the back's. Nothing reaches it on GPUMem: entries GPUMem must have run
// now go to the back (OnBack), and the runtime's Player() and Seam() check too. Must be 0.
void MemPlayer::CheckBack(const char* where) noexcept {
    if (Link::OnStageThread()) [[unlikely]] {
        Link::Note(Link::Check::PlayerOnStage, where);
    }
    // One follower at step 4: the player is R's; E reaching it while it fills is "owner on E"
    // (the seam itself is E's way to fill: not counted).
    if (MemStage::OneFollowerFill() && std::string_view{where} != "runtime.Seam()") [[unlikely]] {
        VideoCommon::OneFollower::Note(VideoCommon::OneFollower::Check::OwnerOnE, where);
    }
}

void MemPlayer::AddDescriptors(const DescriptorUpdateEntry* entries, size_t count) {
    CheckBack("AddDescriptors");
    guest_descriptor_queue.AddEntries(entries, count);
}

void MemPlayer::Exec(const MemStage::Entry& entry) {
    using Cmd = MemStage::Cmd;
    constexpr u32 HEAD = static_cast<u32>(sizeof(MemStage::Entry));
    // The kinds with a typed play (entries run at once call it directly, MemStage::RunNow).
    switch (entry.kind) {
    case Cmd::UploadAhead:
    case Cmd::UploadBatched:
    case Cmd::Copy: {
        const auto& payload = *PayloadAt<MemStage::CopyPayload>(entry, HEAD);
        const VkBufferCopy* const copies = PayloadAt<VkBufferCopy>(
            entry, HEAD + Align8(static_cast<u32>(sizeof(MemStage::CopyPayload))));
        PlayCopy(entry.kind, entry.flags, payload.src, payload.dst,
                 std::span<const VkBufferCopy>{copies, payload.count});
        return;
    }
    case Cmd::BindIndex:
    case Cmd::BindIndexAlways: {
        const auto& payload = *PayloadAt<MemStage::IndexPayload>(entry, HEAD);
        PlayIndex(entry.kind, entry.flags, payload.buffer, payload.offset, payload.type);
        return;
    }
    case Cmd::BindVertex: {
        const auto& payload = *PayloadAt<MemStage::VertexOne>(entry, HEAD);
        PlayVertex(entry.flags, payload.index, payload.buffer, payload.offset, payload.size,
                   payload.stride);
        return;
    }
    case Cmd::BindVertices: {
        const auto& payload = *PayloadAt<MemStage::VerticesPayload>(entry, HEAD);
        const u32 binding_count = payload.count;
        const u32 handles_at = HEAD + Align8(static_cast<u32>(sizeof(MemStage::VerticesPayload)));
        const u32 array_bytes = binding_count * 8;
        PlayVertices(entry.flags, payload.first, binding_count,
                     PayloadAt<VkBuffer>(entry, handles_at),
                     PayloadAt<VkDeviceSize>(entry, handles_at + array_bytes),
                     PayloadAt<VkDeviceSize>(entry, handles_at + 2 * array_bytes),
                     PayloadAt<VkDeviceSize>(entry, handles_at + 3 * array_bytes));
        return;
    }
    default:
        break;
    }
    BeginPlay(entry.flags);
    switch (entry.kind) {
    case Cmd::UploadAhead:
    case Cmd::UploadBatched:
    case Cmd::Copy:
    case Cmd::BindIndex:
    case Cmd::BindIndexAlways:
    case Cmd::BindVertex:
    case Cmd::BindVertices:
        // Played above.
        break;
    case Cmd::PreCopyBarrier:
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
        });
        break;
    case Cmd::PostCopyBarrier:
        scheduler.Record([](vk::CommandBuffer cmdbuf) {
            // Later transfers too, as after CopyBuffer's own barrier: the downloads, gathers and
            // compute uploads this closes are in-order copies a later reordered upload of the
            // same bytes would not wait for.
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0,
                                   COPY_WRITE_BARRIER);
        });
        break;
    case Cmd::Clear: {
        const auto& payload = *PayloadAt<MemStage::ClearPayload>(entry, HEAD);
        scheduler.Record([dest_buffer = payload.dst, offset = payload.offset, size = payload.size,
                          value = payload.value](vk::CommandBuffer cmdbuf) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
            cmdbuf.FillBuffer(dest_buffer, offset, size, value);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE, 0, COPY_WRITE_BARRIER);
        });
        break;
    }
    case Cmd::QuadLutUpload: {
        const auto& payload = *PayloadAt<MemStage::QuadLutPayload>(entry, HEAD);
        scheduler.Record([src_buffer = payload.src, src_offset = payload.src_offset,
                          dst_buffer = payload.dst,
                          size_bytes = payload.size](vk::CommandBuffer cmdbuf) {
            const VkBufferCopy copy{
                .srcOffset = src_offset,
                .dstOffset = 0,
                .size = size_bytes,
            };
            const VkBufferMemoryBarrier write_barrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_INDEX_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = dst_buffer,
                .offset = 0,
                .size = size_bytes,
            };
            cmdbuf.CopyBuffer(src_buffer, dst_buffer, copy);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, write_barrier);
        });
        break;
    }
    case Cmd::NullInit: {
        const auto& payload = *PayloadAt<MemStage::NullInitPayload>(entry, HEAD);
        scheduler.Record([buffer = payload.buffer](vk::CommandBuffer cmdbuf) {
            cmdbuf.FillBuffer(buffer, 0, VK_WHOLE_SIZE, 0);
        });
        break;
    }
    case Cmd::BindXfb: {
        const auto& payload = *PayloadAt<MemStage::XfbOne>(entry, HEAD);
        scheduler.Record([index = payload.index, buffer = payload.buffer, offset = payload.offset,
                          size = payload.size](vk::CommandBuffer cmdbuf) {
            const VkDeviceSize vk_offset = offset;
            const VkDeviceSize vk_size = size;
            cmdbuf.BindTransformFeedbackBuffersEXT(index, 1, &buffer, &vk_offset, &vk_size);
        });
        break;
    }
    case Cmd::BindXfbs: {
        const auto& payload = *PayloadAt<MemStage::XfbsPayload>(entry, HEAD);
        const u32 count = payload.count;
        const u32 handles_at = HEAD + Align8(static_cast<u32>(sizeof(MemStage::XfbsPayload)));
        const VkBuffer* const handles = PayloadAt<VkBuffer>(entry, handles_at);
        const VkDeviceSize* const offsets = PayloadAt<VkDeviceSize>(entry, handles_at + count * 8);
        const VkDeviceSize* const sizes =
            PayloadAt<VkDeviceSize>(entry, handles_at + 2 * count * 8);
        using Handles = boost::container::static_vector<VkBuffer, NUM_BOUND_VERTEX_BUFFERS>;
        using Sizes = boost::container::static_vector<VkDeviceSize, NUM_BOUND_VERTEX_BUFFERS>;
        scheduler.Record([buffer_handles_ = Handles(handles, handles + count),
                          offsets_ = Sizes(offsets, offsets + count),
                          sizes_ = Sizes(sizes, sizes + count)](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindTransformFeedbackBuffersEXT(0, static_cast<u32>(buffer_handles_.size()),
                                                   buffer_handles_.data(), offsets_.data(),
                                                   sizes_.data());
        });
        break;
    }
    case Cmd::Convert: {
        const auto& payload = *PayloadAt<MemStage::ConvertPayload>(entry, HEAD);
        switch (payload.pass) {
        case MemStage::Pass::Uint8:
            uint8_pass->Record(payload.run);
            break;
        case MemStage::Pass::QuadIndexed:
            quad_indexed_pass->Record(payload.run);
            break;
        case MemStage::Pass::VertexRepack:
            vertex_repack_pass->Record(payload.run);
            break;
        }
        break;
    }
    case Cmd::Descriptors: {
        const auto& payload = *PayloadAt<MemStage::DescriptorsPayload>(entry, HEAD);
        AddDescriptors(PayloadAt<DescriptorUpdateEntry>(
                           entry, HEAD + Align8(static_cast<u32>(sizeof(MemStage::DescriptorsPayload)))),
                       payload.count);
        break;
    }
    case Cmd::Count:
        ASSERT(false);
        break;
    }
}

// The typed plays: Exec decodes an entry to these; an entry run as pushed comes here directly
// (MemStage::RunNow), with the same arguments the entry would have carried.

void MemPlayer::BeginPlay(u8 flags) {
    CheckBack("Exec");
    if ((flags & MemStage::OutsideRenderPass) != 0) {
        scheduler.RequestOutsideRenderPassOperationContext();
    }
}

void MemPlayer::PlayCopy(MemStage::Cmd kind, u8 flags, VkBuffer src_buffer, VkBuffer dst_buffer,
                         std::span<const VkBufferCopy> copies) {
    using Cmd = MemStage::Cmd;
    BeginPlay(flags);
    if (kind == Cmd::UploadBatched) {
        scheduler.RecordUpload(src_buffer, dst_buffer, copies);
        return;
    }
    const boost::container::small_vector<VkBufferCopy, 8> vk_copies(copies.begin(),
                                                                     copies.end());
    if (kind == Cmd::UploadAhead) {
        scheduler.RecordWithUploadBuffer([src_buffer, dst_buffer, vk_copies](
                                             vk::CommandBuffer, vk::CommandBuffer upload_cmdbuf) {
            upload_cmdbuf.CopyBuffer(src_buffer, dst_buffer, vk_copies);
        });
        return;
    }
    const bool barrier = (flags & MemStage::Barrier) != 0;
    scheduler.Record([src_buffer, dst_buffer, vk_copies, barrier](vk::CommandBuffer cmdbuf) {
        if (barrier) {
            cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, COPY_READ_BARRIER);
        }
        cmdbuf.CopyBuffer(src_buffer, dst_buffer, vk_copies);
        if (barrier) {
            // Later transfers too: a reordered upload of the same bytes in a later submission
            // (RecordWithUploadBuffer, ahead of that submission's work) has no barrier of its
            // own before it, and in-order copies don't count as uses for CanReorderUpload.
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER, 0,
                                   COPY_WRITE_BARRIER);
        }
    });
}

void MemPlayer::PlayIndex(MemStage::Cmd kind, u8 flags, VkBuffer vk_buffer, VkDeviceSize vk_offset,
                          VkIndexType vk_index_type) {
    BeginPlay(flags);
    ++index_plays;
    // EnsureIndexBinding: bound (or still bound) in this command buffer from here on.
    last_index_valid = (flags & MemStage::ForgetOnly) == 0;
    last_index_buffer = vk_buffer;
    last_index_offset = vk_offset;
    last_index_type = vk_index_type;
    last_index_vertex_epoch = scheduler.VertexEpoch();
    if (kind == MemStage::Cmd::BindIndexAlways) {
        bound_index_valid = false;
        if ((flags & MemStage::ForgetOnly) != 0) {
            return;
        }
        scheduler.Record([buffer = vk_buffer, offset = vk_offset,
                          type = vk_index_type](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindIndexBuffer(buffer, offset, type);
        });
        return;
    }
    // The same index buffer is still bound in this command buffer: skip the rebind.
    if (VideoCommon::Nx2Perf::On(VideoCommon::Nx2Perf::Opt::IndexRebind) &&
        !VideoCommon::Nx2Perf::Verify()) {
        const u64 epoch = scheduler.RecordingEpoch();
        if (bound_index_valid && vk_buffer == bound_index_buffer &&
            vk_offset == bound_index_offset && vk_index_type == bound_index_type &&
            epoch == bound_index_epoch) {
            return;
        }
        bound_index_valid = true;
        bound_index_buffer = vk_buffer;
        bound_index_offset = vk_offset;
        bound_index_type = vk_index_type;
        bound_index_epoch = epoch;
    }
    scheduler.Record([vk_buffer, vk_offset, vk_index_type](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindIndexBuffer(vk_buffer, vk_offset, vk_index_type);
    });
}

void MemPlayer::PlayVertex(u8 flags, u32 index, VkBuffer buffer, u32 offset, u32 size,
                           u32 stride) {
    BeginPlay(flags);
    // Remembered as bound (EnsureVertexBindings binds it again in a new command buffer); a
    // forget, or a null binding, leaves nothing to bind again.
    if (index < VideoCommon::NUM_VERTEX_BUFFERS) {
        BoundVertexBuffer& bound = bound_vertex_buffers[index];
        if ((flags & MemStage::ForgetOnly) != 0 || buffer == VK_NULL_HANDLE) {
            bound.epoch = ~u64{0};
        } else {
            bound = {buffer,
                     offset,
                     size,
                     stride,
                     scheduler.VertexEpoch(),
                     (flags & MemStage::DynamicState) != 0};
        }
    }
    if ((flags & MemStage::ForgetOnly) != 0) {
        Bump(vertex_forgotten);
        return;
    }
    if ((flags & MemStage::DynamicState) != 0) {
        scheduler.Record([index, buffer, offset, size, stride](vk::CommandBuffer cmdbuf) {
            const VkDeviceSize vk_offset = buffer != VK_NULL_HANDLE ? offset : 0;
            const VkDeviceSize vk_size = buffer != VK_NULL_HANDLE ? size : VK_WHOLE_SIZE;
            const VkDeviceSize vk_stride = stride;
            cmdbuf.BindVertexBuffers2EXT(index, 1, &buffer, &vk_offset, &vk_size, &vk_stride);
        });
    } else {
        scheduler.Record([index, buffer, offset](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindVertexBuffer(index, buffer, offset);
        });
    }
}

void MemPlayer::PlayVertices(u8 flags, u32 first, u32 binding_count, const VkBuffer* handles,
                             const VkDeviceSize* offsets, const VkDeviceSize* sizes,
                             const VkDeviceSize* strides) {
    BeginPlay(flags);
    // The same buffers are still bound in this command buffer (as for index buffers): skip
    // the rebind. Draws re-resolve their vertex streams when the guest rewrites unchanged
    // registers. Verify never drops one, but still remembers what it binds: EnsureVertexBindings
    // binds the memo again in a new command buffer, and a memo left from before Verify could
    // name a buffer deleted since (review R1 should-fix 3).
    const u64 epoch = scheduler.VertexEpoch();
    if (!VideoCommon::Nx2Perf::Verify()) {
        // A clean run (full vertex binds) only re-states what is bound: still bound at any size,
        // as before full binds (its size follows the draw, the binding did not change).
        const bool keep_size = (flags & MemStage::KeepSize) != 0;
        bool unchanged = true;
        for (u32 i = 0; unchanged && i < binding_count; ++i) {
            const BoundVertexBuffer& bound = bound_vertex_buffers[first + i];
            unchanged = bound.epoch == epoch && bound.buffer == handles[i] &&
                        bound.offset == offsets[i] && (keep_size || bound.size == sizes[i]) &&
                        bound.stride == strides[i];
        }
        if (unchanged) {
            Bump(vertex_runs_dropped);
            return;
        }
    }
    {
        const bool dynamic = (flags & MemStage::DynamicState) != 0;
        for (u32 i = 0; i < binding_count; ++i) {
            bound_vertex_buffers[first + i] = {handles[i], offsets[i], sizes[i], strides[i],
                                               epoch, dynamic};
        }
    }
    Bump(vertex_runs_recorded);
    using Handles = boost::container::static_vector<VkBuffer, NUM_BOUND_VERTEX_BUFFERS>;
    using Sizes = boost::container::static_vector<VkDeviceSize, NUM_BOUND_VERTEX_BUFFERS>;
    Handles buffer_handles(handles, handles + binding_count);
    Sizes buffer_offsets(offsets, offsets + binding_count);
    if ((flags & MemStage::DynamicState) != 0) {
        Sizes buffer_sizes(sizes, sizes + binding_count);
        Sizes buffer_strides(strides, strides + binding_count);
        scheduler.Record([first, binding_count, buffer_handles_ = std::move(buffer_handles),
                          offsets_ = std::move(buffer_offsets),
                          sizes_ = std::move(buffer_sizes),
                          strides_ = std::move(buffer_strides)](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindVertexBuffers2EXT(first, binding_count, buffer_handles_.data(),
                                         offsets_.data(), sizes_.data(), strides_.data());
        });
    } else {
        scheduler.Record([first, binding_count, buffer_handles_ = std::move(buffer_handles),
                          offsets_ = std::move(buffer_offsets)](vk::CommandBuffer cmdbuf) {
            cmdbuf.BindVertexBuffers(first, binding_count, buffer_handles_.data(),
                                     offsets_.data());
        });
    }
}

// ---------------------------------------------------------------------------------------------
// The list draw facility. Its stamps are the tick a copy was recorded in, its wait is on that
// tick and the draw takes its result here: facts about what the GPU thread recorded. Its two
// buffers are created with the main allocator.

void MemPlayer::RecordListDrawArgs(VkBuffer source, VkDeviceSize offset, bool mesh) {
    CheckBack("RecordListDrawArgs");
    if (source == VK_NULL_HANDLE) {
        return;
    }
    stage->Play();
    PlayDepth depth;
    if (!list_draw_buffer) {
        const VkBufferCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = VkDeviceSize{LIST_DRAW_SLOTS} * LIST_DRAW_SLOT_BYTES,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        };
        // FRITCHII_LIST_ARGS_LOCAL (boot, default 0): in host memory the GPU writes the two words
        // across the link and its command processor reads them back, a round trip per list draw
        // on a slow link. In device memory neither leaves the card.
        static const bool args_local = [] {
            const char* const value = std::getenv("FRITCHII_LIST_ARGS_LOCAL");
            return value != nullptr && value[0] != '0';
        }();
        list_draw_args_local = args_local;
        if (args_local) {
            VkBufferCreateInfo template_info = create_info;
            template_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            list_draw_template = memory_allocator.CreateBuffer(template_info, MemoryUsage::Upload);
            list_draw_buffer = memory_allocator.CreateBuffer(create_info, MemoryUsage::DeviceLocal);
            list_draw_fill_pending = true;
        } else {
            list_draw_buffer = memory_allocator.CreateBuffer(create_info, MemoryUsage::Upload);
        }
        vk::Buffer& words_buffer = args_local ? list_draw_template : list_draw_buffer;
        if (device.HasDebuggingToolAttached()) {
            list_draw_buffer.SetObjectNameEXT("List draw arguments");
        }
        // Written once: the host GPU only ever copies over words 0 and 2 of the vertex command
        // and word 0 of the task/mesh command.
        const std::span<u8> mapped = words_buffer.Mapped();
        constexpr std::array<u32, LIST_DRAW_SLOT_BYTES / sizeof(u32)> slot{0, 1, 0, 0, 0, 1, 1, 0};
        for (u32 i = 0; i < LIST_DRAW_SLOTS; ++i) {
            std::memcpy(mapped.data() + size_t{i} * LIST_DRAW_SLOT_BYTES, slot.data(),
                        LIST_DRAW_SLOT_BYTES);
        }
        words_buffer.Flush();
    }
    if (VideoCommon::LinkStats::On()) [[unlikely]] {
        VideoCommon::LinkStats::NoteListDraw(!list_draw_args_local);
    }
    const VkBuffer fill_source = std::exchange(list_draw_fill_pending, false)
                                     ? *list_draw_template
                                     : VkBuffer{VK_NULL_HANDLE};
    // The copies and the draws that read them run in the order they are recorded: a slot is
    // free again as soon as the next copy into it is recorded after its draw.
    const VkDeviceSize slot = VkDeviceSize{list_draw_cursor} * LIST_DRAW_SLOT_BYTES;
    list_draw_cursor = (list_draw_cursor + 1) % LIST_DRAW_SLOTS;
    const VkDeviceSize args = slot + (mesh ? 16 : 0);
    scheduler.RequestOutsideRenderPassOperationContext();
    // The front's copy of the words, with the same barriers.
    VkBuffer keep_buffer = VK_NULL_HANDLE;
    VkDeviceSize keep_offset = 0;
    if (const u64 keep_id = list_draw_keep_id; keep_id != 0) {
        EnsureListKeepBuffer();
        const u32 keep_slot = static_cast<u32>(keep_id % LIST_KEEP_SLOTS);
        keep_buffer = *list_keep_buffer;
        keep_offset = VkDeviceSize{keep_slot} * LIST_KEEP_SLOT_BYTES;
        list_keeps[keep_slot] = {keep_id, scheduler.CurrentTick(), true, 2, {}};
    }
    scheduler.Record([source, offset, destination = *list_draw_buffer, args, mesh, keep_buffer,
                      keep_offset, fill_source](vk::CommandBuffer cmdbuf) {
        static constexpr VkMemoryBarrier READ_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        // (count, first) -> word 0 and, for vertex draws, word 2.
        const std::array<VkBufferCopy, 2> copies{{
            {.srcOffset = offset, .dstOffset = args, .size = sizeof(u32)},
            {.srcOffset = offset + sizeof(u32), .dstOffset = args + 2 * sizeof(u32),
             .size = sizeof(u32)},
        }};
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, READ_BARRIER);
        if (fill_source != VK_NULL_HANDLE) {
            // FRITCHII_LIST_ARGS_LOCAL: the fixed words, before the first draw's two.
            static constexpr VkMemoryBarrier FILLED_BARRIER{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            };
            const VkBufferCopy fill{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = VkDeviceSize{LIST_DRAW_SLOTS} * LIST_DRAW_SLOT_BYTES,
            };
            cmdbuf.CopyBuffer(fill_source, destination, fill);
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   0, FILLED_BARRIER);
        }
        cmdbuf.CopyBuffer(source, destination,
                          std::span<const VkBufferCopy>{copies.data(), mesh ? 1U : 2U});
        if (keep_buffer != VK_NULL_HANDLE) {
            const VkBufferCopy keep{
                .srcOffset = offset, .dstOffset = keep_offset, .size = 2 * sizeof(u32)};
            cmdbuf.CopyBuffer(source, keep_buffer, keep);
        }
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               keep_buffer != VK_NULL_HANDLE
                                   ? VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_HOST_BIT
                                   : VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                               0, WRITE_BARRIER);
    });
    list_draw_args = {*list_draw_buffer, args, mesh};
}

void MemPlayer::EnsureListKeepBuffer() {
    if (list_keep_buffer) {
        return;
    }
    const VkBufferCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = VkDeviceSize{LIST_KEEP_SLOTS} * LIST_KEEP_SLOT_BYTES,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    list_keep_buffer = memory_allocator.CreateBuffer(create_info, MemoryUsage::Download);
    if (device.HasDebuggingToolAttached()) {
        list_keep_buffer.SetObjectNameEXT("Kept list words");
    }
}

void MemPlayer::KeepListWords(u64 keep_id, VkBuffer source, VkDeviceSize offset, u32 count) {
    CheckBack("KeepListWords");
    if (keep_id == 0 || source == VK_NULL_HANDLE || count == 0 || count > 3) {
        return;
    }
    stage->Play();
    PlayDepth depth;
    EnsureListKeepBuffer();
    const u32 slot = static_cast<u32>(keep_id % LIST_KEEP_SLOTS);
    const VkDeviceSize keep_offset = VkDeviceSize{slot} * LIST_KEEP_SLOT_BYTES;
    scheduler.RequestOutsideRenderPassOperationContext();
    list_keeps[slot] = {keep_id, scheduler.CurrentTick(), true, count, {}};
    scheduler.Record([source, offset, destination = *list_keep_buffer, keep_offset,
                      count](vk::CommandBuffer cmdbuf) {
        static constexpr VkMemoryBarrier READ_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        };
        static constexpr VkMemoryBarrier HOST_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, READ_BARRIER);
        const VkBufferCopy copy{
            .srcOffset = offset, .dstOffset = keep_offset, .size = VkDeviceSize{count} * 4};
        cmdbuf.CopyBuffer(source, destination, copy);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                               HOST_BARRIER);
    });
}

void MemPlayer::KeepListWordsRead(u64 keep_id, std::span<const u32> words) {
    CheckBack("KeepListWordsRead");
    if (keep_id == 0 || words.empty() || words.size() > 3) {
        return;
    }
    ListKeep& keep = list_keeps[keep_id % LIST_KEEP_SLOTS];
    keep = {keep_id, 0, false, static_cast<u32>(words.size()), {}};
    std::ranges::copy(words, keep.words.begin());
}

bool MemPlayer::ReadKeptListWords(u64 keep_id, std::span<u32> words) {
    CheckBack("ReadKeptListWords");
    const ListKeep& keep = list_keeps[keep_id % LIST_KEEP_SLOTS];
    if (keep_id == 0 || keep.keep_id != keep_id || words.size() > keep.count) {
        return false;
    }
    if (!keep.on_gpu) {
        std::copy_n(keep.words.begin(), words.size(), words.begin());
        return true;
    }
    {
        MemStage::SyncScope sync{stage, MemStage::Sync::KeptWords};
        scheduler.Wait(keep.tick);
    }
    list_keep_buffer.Invalidate();
    const VkDeviceSize keep_offset = VkDeviceSize{keep_id % LIST_KEEP_SLOTS} * LIST_KEEP_SLOT_BYTES;
    std::memcpy(words.data(), list_keep_buffer.Mapped().data() + keep_offset, words.size_bytes());
    return true;
}

} // namespace Vulkan

// The buffer cache's read log lines (buffer_cache.h). g_on is set while SEAM = 2 runs with
// FRITCHII_MEMSTAGE_READ_LOG; only a fill that logs (every Nth packet, on its own thread) has a
// log.
namespace VideoCommon::MemReadLog {

using Vulkan::MemStage;

void NoteBytes(Kind kind, Tegra::MaxwellDeviceMemoryManager& memory, u64 address, u64 size) {
    MemStage::NoteRead(kind, address, size);
    MemStage::ReadLog* const log = MemStage::t_log;
    if (log == nullptr || size == 0) {
        return;
    }
    // Also past the budget: the over-read test needs every range the packet uses.
    log->used.emplace_back(address, size);
    if (log->budget + size > Vulkan::READ_LOG_BUDGET) {
        ++log->skipped;
        return;
    }
    log->budget += size;
    const size_t at = log->bytes.size();
    log->bytes.resize(at + static_cast<size_t>(size));
    memory.ReadBlockUnsafe(address, log->bytes.data() + at, static_cast<size_t>(size));
    log->records.push_back({
        .kind = kind,
        .how = MemStage::ReadRecord::How::Bytes,
        .size = static_cast<u32>(size),
        .address = address,
        .value = at,
        .data = nullptr,
        .memory = &memory,
    });
}

void NoteStaged(Kind kind, Tegra::MaxwellDeviceMemoryManager& memory, u64 address,
                const u8* staged, u64 size) {
    MemStage::NoteRead(kind, address, size);
    MemStage::ReadLog* const log = MemStage::t_log;
    if (log == nullptr || size == 0) {
        return;
    }
    if (log->budget + size > Vulkan::READ_LOG_BUDGET) {
        ++log->skipped;
        return;
    }
    log->budget += size;
    log->records.push_back({
        .kind = kind,
        .how = MemStage::ReadRecord::How::Staged,
        .size = static_cast<u32>(size),
        .address = address,
        .value = 0,
        .data = staged,
        .memory = &memory,
    });
}

void NoteValue(Kind kind, Tegra::MemoryManager* memory, u64 address, u64 value, u32 size) {
    if (MemStage::t_reads != nullptr && memory != nullptr) {
        // A GPU address: kept as the device range it maps to.
        if (const auto device_address = memory->GpuToCpuAddress(address)) {
            MemStage::NoteRead(kind, *device_address, size);
        }
    }
    MemStage::ReadLog* const log = MemStage::t_log;
    if (log == nullptr || memory == nullptr) {
        return;
    }
    if (const auto device_address = memory->GpuToCpuAddress(address)) {
        log->used.emplace_back(*device_address, size);
    }
    log->records.push_back({
        .kind = kind,
        .how = MemStage::ReadRecord::How::Value,
        .size = size,
        .address = address,
        .value = value,
        .data = nullptr,
        .memory = memory,
    });
}

void NoteRange(Kind kind, u64 address, u64 size) {
    MemStage::NoteRead(kind, address, size);
    MemStage::ReadLog* const log = MemStage::t_log;
    if (log == nullptr || size == 0) {
        return;
    }
    log->used.emplace_back(address, size);
    log->records.push_back({
        .kind = kind,
        .how = MemStage::ReadRecord::How::Range,
        .size = static_cast<u32>(size),
        .address = address,
        .value = 0,
        .data = nullptr,
        .memory = nullptr,
    });
}

// The texture cache wrote game memory while a draw or dispatch is held (SEAM 2/3): its GC
// downloads, EXP121's landed downloads, a flushed download. The fill read before the texture
// work; at SEAM 0 and 1 the same reads come after this write. Counted; a write over one of the
// fill's read ranges is a crossing the early fill gets wrong (must be 0).
void NoteGuestWrite(Tegra::MemoryManager& memory, u64 gpu_addr, u64 size) {
    MemStage* const stage = Vulkan::g_stage.load(std::memory_order_acquire);
    if (stage == nullptr || size == 0) {
        return;
    }
    Vulkan::Bump(stage->counters.texture_writes);
    if (!Vulkan::t_holding || stage->held == MemStage::HeldKind::None) {
        // Another thread's write: counted only (the ranges belong to the holding thread).
        return;
    }
    for (const auto& [device_address, length] : memory.GetSubmappedDeviceRange(gpu_addr, size)) {
        const u64 end = device_address + length;
        for (const MemStage::ReadRange& read : stage->held_reads) {
            if (device_address >= read.end || read.begin >= end) {
                continue;
            }
            Vulkan::Bump(stage->counters.texture_overlaps);
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                const bool is_draw = stage->held == MemStage::HeldKind::Draw;
                LOG_WARNING(Render_Vulkan,
                            "Memory seam: texture write over fill read: {:#x}+{:#x} (device "
                            "{:#x}) over the {} read at {:#x}+{:#x} of {} {} (programs {:016x} "
                            "{:016x})",
                            gpu_addr, size, device_address,
                            Vulkan::READ_KIND_NAMES[static_cast<size_t>(read.kind)], read.begin,
                            read.end - read.begin, is_draw ? "draw" : "dispatch",
                            is_draw ? stage->draw.number : stage->dispatch.number,
                            stage->held_hashes[0], stage->held_hashes[1]);
            }
            return;
        }
    }
}

} // namespace VideoCommon::MemReadLog
