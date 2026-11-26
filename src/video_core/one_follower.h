// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// One follower (work\fable_b_one_follower.md): the renderer's front end split behind the engine.
//
//   F, the GPU thread (GpuSplit's front): command lists and macros, unchanged.
//   E, the engine thread = GpuSplit's back thread: GpuSplit::Execute on the real engines, game
//      memory writes from the stream, the buffer cache, each draw's buffer fill (by value,
//      MemStage::FillRingDraw into ring B).
//   R, the render thread (new, this file): the pipeline cache, texture cache, query cache, fences,
//      state tracker and the Vulkan scheduler. It has no engine: it plays packets and keeps a
//      view of the registers it reads, fed by deltas E sends.
//
// Everything R cannot do from a packet yet runs "lent": E pushes a lent packet and parks; R plays
// every packet before it, runs the operation reading E's engine directly, then releases E. That is
// today's serial order run on two threads, so every step below keeps the picture.
//
// Steps (section 8 / 8.1 of the design), one switch each, nested (each needs the one before and
// refuses loudly without it; the effective step is what Step() returns):
//   1 FRITCHII_ONEF=1 (boot): R exists, every draw and every R-owned operation lent.
//   2 FRITCHII_ONEF_VIEW=1 (live): E packs the delta spans; R applies them to its view and, at
//     every lent draw, compares the view with the engine (Check::View*).
//   3 FRITCHII_ONEF_BUILD=1 (live): R also builds each lent draw from the packet and its view and
//     compares that with what the lent path used (Check::Build*).
//   4 FRITCHII_ONEF_UNLEND=<mask> (live): the draw kinds in the mask run on R from the packet
//     alone while E goes on (UnlendKind). Every other draw and op stays lent, and every lent draw
//     counts its reason (LentReason). FRITCHII_ONEF_VERIFY_EVERY=N forces every Nth unlent draw
//     lent and runs step 3's compare on it: step 4's own must-be-0.
//   FRITCHII_ONEF_AHEAD (live): E's window in packets R has not played (1, 8, 0 = the ring).
//   FRITCHII_ONEF_PIN (boot, default 1): R gets PinnedRole::GpuMem's physical core.
//
// Ownership (section 5): state has one owner. E never touches an R-owned object (scheduler,
// texture, pipeline and query caches, fence manager, state tracker) unless R is parked behind a
// lent packet or drained; R never touches the buffer cache, the engines or game memory writes
// outside a lent packet. Check::OwnerOnE / OwnerOnR count violations (must be 0).
//
// Out of this branch (step 5): reports, counter resets, clears, fences with served landings,
// compute dispatches and 0xe2c on R. They stay lent here.

#include <array>
#include <atomic>
#include <cstddef>
#include <span>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/one_follower_hooks.h"

namespace Core {
class System;
}

namespace Common {
class SpinRecursiveMutex;
}

namespace VideoCommon {
class PageFilter;
}

namespace VideoCommon::OneFollower {

using Maxwell3D = Tegra::Engines::Maxwell3D;

// --- Switches ------------------------------------------------------------------------------------
// Stored in VideoCore::Nx2Debug (nx2_debug_control.h): g_onef (boot), g_onef_view, g_onef_build,
// g_onef_unlend, g_onef_verify_every, g_onef_ahead (live), g_onef_pin (boot).

/// The steps, as Step() reports them after nesting.
enum class StepLevel : u8 {
    Off = 0,    ///< FRITCHII_ONEF=0 at boot: today's code path, untouched
    Lent = 1,   ///< R exists, everything lent
    View = 2,   ///< + the register view and its delta, verify only
    Build = 3,  ///< + draws built from the packet, verify only
    Unlend = 4, ///< + the draw kinds of FRITCHII_ONEF_UNLEND unlent
};

/// FRITCHII_ONEF_UNLEND's bits: the draw kinds that run on R from the packet alone at step 4.
/// Plain draws first; every other kind starts lent and gets its own bit.
enum UnlendKind : u32 {
    PlainIndexed = 1u << 0, ///< draw.end, indexed, one instance or a plain instanced draw
    PlainArrays = 1u << 1,  ///< the same, non-indexed
    InstanceLoop = 1u << 2, ///< a deferred instance loop (DrawDeferred, folded first instance)
    Indirect = 1u << 3,     ///< DrawIndirect (counts and arguments in GPU memory)
    /// List draws (nx2_list_draw.h: two words in GPU memory). E's fill binds the words and puts
    /// their buffer in ring B's packet (ListArgs); the draw span carries the keep id and the
    /// address. A list draw on a mesh pipeline needs Mesh too; one that reads its words on the
    /// CPU stays lent (LentReason::NoPath).
    List = 1u << 4,
    /// Task/mesh pipelines, direct draws. Native ones only (VK_EXT_mesh_shader); emulated ones
    /// stay lent (LentReason::NoPath).
    Mesh = 1u << 5,
    GpuCount = 1u << 6,     ///< a vertex count GPU work wrote (nx2_indirect_grid.h)
    InlineIndex = 1u << 7,  ///< inline index draws (the indexes live in a vector on E)
    AllKinds = 0xff,
};

/// FRITCHII_ONEF_VERIFY_EVERY's default: every 64th unlent draw is forced lent and compared.
inline constexpr int DefaultVerifyEvery = 64;
/// FRITCHII_ONEF_AHEAD's default (packets R has not played that E may run ahead by; 0 = ring).
inline constexpr int DefaultAhead = 1;

// --- Why something ran lent (counted per reason; not a must-be-0) ---------------------------------

/// Lent draws by reason. A draw counts exactly one: the first that applies, in this order.
enum class LentReason : u8 {
    StepBelowUnlend, ///< step 1 to 3: every draw is lent
    KindNotInMask,   ///< its UnlendKind bit is off (counted per kind as well, LentKind)
    PipelineMiss,    ///< R's lookup on its view found no ready pipeline
    Verify,          ///< FRITCHII_ONEF_VERIFY_EVERY: forced lent to compare
    StreamWrite,     ///< the stream-write guard hit since the last packet (Check::StreamWrite)
    Diagnostic,      ///< a frame dump, render trace, probe or capture reads E's state
    Transient,       ///< state R's view cannot carry yet (channel switch pending, view not seeded)
    /// Its kind's bit is in the mask but this draw needs what R's path cannot do: a list draw
    /// whose two words are read on the CPU (ListDrawReadsBack), an emulated mesh draw (the
    /// staging pool and the mesh arenas).
    NoPath,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(LentReason::Count)> lent_reason_names{
    "step below 4", "kind not in mask", "pipeline miss", "verify", "stream write", "diagnostic",
    "transient", "no R path",
};

/// Lent operations other than draws (step 5's kinds and the rest): E parks for each.
enum class LentOp : u8 {
    Draw,         ///< a draw (its reason: LentReason)
    Clear,        ///< RasterizerVulkan::Clear
    Dispatch,     ///< DispatchCompute (lent, or posted with FRITCHII_ONEF_OP_DISPATCH)
    Query,        ///< Query, ResetCounter, report and counter work
    Fence,        ///< SignalFence, SyncOperation*, SignalSyncPoint, SignalReference, ReleaseFences
    ReadBack,     ///< FlushRegion*, FinishRead, GetFlushArea, WaitForIdle, TickWork's texture half
    Invalidation, ///< InvalidateRegion, OnCacheInvalidation(s), OnCPUWrite, UnmapMemory, gathers
    Copy,         ///< accelerated DMA, surface copies, inline uploads onto images
    FrameEnd,     ///< TickFrame, present, AccelerateDisplay, FlushCommands
    Channel,      ///< a channel bind (the view is seeded again from the new channel's engine)
    Other,        ///< any other call into an R-owned object
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(LentOp::Count)> lent_op_names{
    "draw", "clear", "dispatch", "query", "fence", "read back", "invalidation", "copy",
    "frame end", "channel", "other",
};

// --- Must-be-0 checks ------------------------------------------------------------------------------
// Logged every 10 s on one line memstage_gate.py reads, in its pair form:
//   "One follower checks (must be 0; this window / session): view words 0 / 0, view flags 0 / 0, ..."
// with the names below.

enum class Check : u8 {
    // Step 2, at every lent draw: R's view against E's engine.
    ViewWords,     ///< a watched register word differs
    ViewFlags,     ///< an R-owned or shared dirty flag differs
    ViewDraw,      ///< the packet's DrawValues differ from the engine's draw manager
    ViewPrograms,  ///< the six program hashes differ
    ViewPools,     ///< the TIC or TSC pool differs
    // Step 3 (and step 4's forced-lent draws): the draw built from the packet and the view against
    // what the lent path used.
    BuildPipeline,    ///< the pipeline (key) differs
    BuildImageViews,  ///< the image view ids differ
    BuildDynamic,     ///< the dynamic state differs
    BuildDescriptors, ///< the descriptors (image/sampler entries) differ
    BuildRenderTargets, ///< the render targets differ
    // Section 7: E's game memory write from the stream onto a page R's texture reads still need
    // (a page the texture cache tracks, while R has unplayed packets). A hit forces that write
    // lent (step 4); counted at every step.
    StreamWrite,
    // Ownership.
    OwnerOnE, ///< E used an R-owned object while R was neither parked nor drained
    OwnerOnR, ///< R used the buffer cache or an engine outside a lent packet
    // Part c (vk_one_follower_draw.cpp), appended so the values above keep theirs.
    BuildDrawParams,     ///< step 3: the draw's first, count, base and instances differ
    BuildCondition,      ///< step 3: R's 0xe2c mode on its view differs from E's execute_on
    RenderPacketMissing, ///< step 4: R found no ring B packet for an unlent draw (not drawn)
    DropNotPlain,        ///< 0xe2c: a dropped draw's packet held more than uploads and binds
    /// Step 3 (and step 4's forced-lent draws): the facts table entry E's fill would use for the
    /// draw's program registers names other programs than the pipeline the draw ran with (code
    /// rewritten at the same addresses and the entry not retired).
    FactsStale,
    /// Step 4 (fable_b 10.1 row 5): a played draw's pipeline lookup missed (no pipeline, or a
    /// new one built on R from the view, whose constant buffer bindings are only as new as the
    /// last lent packet). Step 4 stops for the session.
    PlayedLookupMiss,
    /// FRITCHII_ONEF_INVAL_E: R uploaded an image from pages E wrote after R's place in the
    /// stream (an invalidation E ran without R, on pages no image had yet): the section 5 hazard
    /// the page filter cannot see. The image uploads again at its next use.
    TextureReadAhead,
    /// fable_b 13.2 R1: R at a BufferOp found another packet at ring B's head (its kind or
    /// number differs from the name); the op's entries are not played.
    RingPairing,
    /// R waited for a lock E holds (the buffer cache's), outside a lent call: R reaches into E's
    /// cache ("player in the cache" as a lock-try); its stack is logged for the first reports.
    RenderWaitsEngineLock,
    /// R outside a lent call (an unlent draw's image upload reads game memory through
    /// MemoryManager::ReadBlock) needed a flush of E's buffer cache: pages the GPU may have
    /// written. R cannot run it (E's cache, E running ahead): the upload reads game memory as it
    /// is, the image takes the range again (it uploads at its next use) and step 4 stops for the
    /// session, so every later draw is lent.
    RenderNeedsEngineFlush,
    /// Step 4's forced-lent list draws (FRITCHII_ONEF_VERIFY_EVERY): the list values the packet
    /// carries (keep id, words address) or R's read-back decision on its view differ from E's
    /// engine's.
    BuildList,
    /// A played draw whose ring B packet's list arguments (MemStage::ListArgs) disagree with the
    /// draw being a list draw on R's view; the draw is not recorded and step 4 stops.
    PlayedListArgs,
    /// A played draw of a kind R has no path for after all (a list draw that reads its words on
    /// the CPU, an emulated mesh draw); not recorded, step 4 stops.
    PlayedNoPath,
    /// FRITCHII_ONEF_SYNC_MIRROR (dry): at a samples report, E's mirror of R's "counter
    /// started" bit (fable_b 16.2) differs from R's streamer.
    SyncMirror,
    /// fable_b 16.2: R, syncing query values in a posted fence, found a range none of the op's
    /// targets (the buffers E obtained) holds: the value is not copied, fences post no more.
    SyncTargetMissing,
    /// fable_b 16.2: a target E obtained for a samples report that R had no value for (E's
    /// mirror said a sync was owed): its early upload may have cleared a mark serial keeps.
    SyncTargetUnused,
    /// FRITCHII_ONEF_OP_DISPATCH: R could not record a posted dispatch as the lent path would
    /// (Configure returned false, an exception, the view's channel gone): the dispatch is not
    /// recorded (its buffer side is played) and dispatches post no more this session.
    PlayedDispatch,
    /// FRITCHII_ONEF_OP_DISPATCH: R played a dispatch while not on its register view (rule 2):
    /// its state invalidation would have reached E's engine. Skipped there; posting stops.
    PlayedDispatchNoView,
    /// fable_b 18.4 (FRITCHII_ONEF_LOOKUP_CHECK): at an unlent draw's lookup on E, a part of the
    /// key the lookup kept because its flag was clear differs from the same part computed fresh
    /// (canonicalized as the lookup does): a pack or path took the flag before E's lookup read
    /// it. One check per part: the programs, the blend, the vertex input.
    LookupPrograms,
    LookupBlend,
    LookupVertexInput,
    /// fable_b 19.6 (FRITCHII_ONEF_SEM_E): a semaphore write on E or R to a word last written
    /// from a later place in the stream (E's place: after the packets it pushed; R's: the
    /// packet it plays). Every writer of a word must have one home.
    SemaphoreOrder,
    /// fable_b 16.4 (POST_FENCE): R, writing a report inside a posted fence, met a guest read's
    /// download over it that E did not land before posting, or found the buffer lock busy and
    /// skipped the landing (R never waits for that lock in a posted op).
    RenderReadLanding,
    /// SEM_E: E made a game-visible signal (a sync point increment, a callback, a payload
    /// write) while R had not yet played a posted fence that lands reports E queued before it
    /// (the order a lent wait for idle kept: R caught up there, 5,900 times a second).
    SignalAheadOfLanding,
    /// FRITCHII_ONEF_UNLENT_RESET (fable_b 19 step 2): a forced-lent draw where E's own indirect
    /// reset test (made before the lend) differs from the test on E's engine inside the lent
    /// call. The switch stops for the session.
    UnlentReset,
    /// FRITCHII_ONEF_UNLENT_VALUES: at a lent packet, the engine hint, the macro mode or a
    /// constant buffer binding R's view took by value differs from E's engine. The switch stops
    /// for the session.
    ViewValues,
    /// FRITCHII_ONEF_BIND_SEED: a played draw found R's view on another channel than the bound
    /// one (the re-seed inside the lent bind did not hold). The switch stops for the session.
    BindSeedView,
    /// FRITCHII_ONEF_OP_DISPATCH bit 4: R played an indirect dispatch whose packet carries no
    /// indirect buffer, or a direct one whose packet does (E and R disagree on the dispatch). Not
    /// recorded; posting stops.
    PlayedDispatchIndirect,
    /// FRITCHII_ONEF_OP_DISPATCH bit 2: a played dispatch wrote an image of a byte-addressed
    /// kind E sends lent (the R16 shadow), or wrote images E did not see in the program's info.
    /// Posting stops.
    PlayedDispatchImage,
    /// FRITCHII_ONEF_OP_DISPATCH bit 4: a posted indirect dispatch's grid words (copied on the
    /// GPU right before it, read once its submission is done) exceed the device's workgroup
    /// count limit. Seen after the fact (the dispatch ran); posting stops.
    IndirectGridOverLimit,
    /// FRITCHII_ONEF_DISPATCH_RANGES (fable_b 27.9): a played image-writing dispatch binds an
    /// image other than the one E recorded the range of (the descriptor R reads at its play
    /// differs from E's, or E had no image where R binds one). The switch stops.
    DispatchImageRange,
    /// FRITCHII_SPLIT_BIND_FIX: a played draw whose bind went with a submit between its
    /// Configure and its record (the scheduler no longer had its pipeline bound). R cannot
    /// configure it again (E's buffer cache): the draw is not recorded.
    PlayedSplitBind,
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Check::Count)> check_names{
    "view words",    "view flags",       "view draw",         "view programs",
    "view pools",    "build pipeline",   "build image views", "build dynamic",
    "build descriptors", "build render targets", "stream write", "owner on E", "owner on R",
    "build draw params", "build condition", "render packet missing", "drop not plain",
    "facts stale", "played lookup miss", "texture read ahead of R", "ring B pairing",
    "R waits for E's lock", "R needs E's buffer flush",
    "build list draw", "played list args", "played kind without path", "sync mirror",
    "sync target missing", "sync target unused", "played dispatch", "played dispatch off view", "lookup programs",
    "lookup blend", "lookup vertex input",
    "semaphore order", "R read landing",
    "signal ahead of landing", "unlent reset", "view values", "bind seed view",
    "played dispatch indirect", "played dispatch image", "indirect grid over the limit",
    "dispatch image range", "played split bind",
};

// --- The packet (E to R) -----------------------------------------------------------------------------
// One ring, one writer (E), one reader (R). Every packet starts with a PacketHeader; its spans
// follow in this order, each present when its bit is in `spans`, each padded to 8 bytes:
//   RegPair[delta_count], FlagBits, DrawValues, Programs, Pools, EngineValues, Binds.
// The buffer half of a draw is MemStage's DrawPacket in ring B (vk_mem_stage.h), filled by E with
// FillRingDraw and taken by R with TakeRingDraw; `buffer_number` names it. Ring B keeps its format.

enum class PacketKind : u8 {
    Draw = 0,  ///< an unlent draw: R plays it from the packet and its view (step 4)
    Lent = 1,  ///< E is parked: R plays what is before, runs `call`, releases E
    Delta = 2, ///< spans only (the delta before a lent op, so R's view is current when it runs)
    /// A render op (fable_b 11.1): R runs `call.function` on the payload after the spans
    /// (`reserved2` = the spans' bytes) at its stream place, E not parked (PushOp).
    Op = 3,
};

enum SpanBit : u8 {
    SpanDelta = 1,    ///< RegPair[delta_count]
    SpanFlags = 2,    ///< FlagBits
    SpanDraw = 4,     ///< DrawValues
    SpanPrograms = 8, ///< Programs
    SpanPools = 16,   ///< Pools
    /// FRITCHII_ONEF_UNLENT_VALUES: EngineValues (sent when they changed since the last packet).
    SpanEngine = 32,
    /// FRITCHII_ONEF_UNLENT_VALUES: BindsHeader then BindValue[count], the constant buffer
    /// bindings E bound since the last packet, with their values at the packet.
    SpanBinds = 64,
};

enum PacketFlag : u8 {
    /// Step 2/3: R compares its view (and at 3 its build) with the engine while E is parked.
    PacketVerify = 1,
    /// The view is seeded from the engine in full before the spans (channel bind, step change,
    /// first packet): R copies regs and flags from E's engine while E is parked (lent only).
    PacketSeed = 2,
    /// A draw under a 0xe2c condition E did not decide (section 5): R plays it only when its own
    /// mode (DrawPlayer::Condition, at the condition's stream place) is not False; a dropped
    /// draw's ring B packet is still played (its uploads and binds). Step 5 sets it; R honours it.
    PacketConditional = 4,
    /// An op packet whose call reads and consumes R's render state (a clear): its R-only dirty
    /// flags go by rule 1, as a draw packet's (PackSpans forwards each set one and clears it on
    /// E's engine), not by difference.
    PacketRenderState = 8,
    /// A draw packet (FRITCHII_ONEF_UNLENT_RESET): E ran IsNx2IndirectReset's test on its own
    /// engine at its place and the draw is not an indirect reset; R does not test it again
    /// (the test reads the program's code in game memory, which R must not do out of a lend).
    PacketNoReset = 16,
    /// A lent packet (FRITCHII_ONEF_UNLENT_VALUES): every change of the engine hint, the macro
    /// mode and the constant buffer bindings since the last lent packet went to R by value, this
    /// packet's spans included; R may compare its view's values with E's engine before it takes
    /// the engine's (Check::ViewValues).
    PacketValues = 32,
    /// FRITCHII_ONEF_CARRY_OPS (fable_b 22.3 step 2): the packet carries render ops E deferred
    /// since the last packet (counter resets and sample reports, which read nothing of R's view).
    /// They follow the packet's body: CarryRecord + payload each, then a CarryTrailer as the
    /// packet's last 8 bytes. R runs them, in order, before anything else of the packet.
    PacketCarried = 64,
};

/// One changed register word R watches.
struct RegPair {
    u32 index;
    u32 value;
};
static_assert(sizeof(RegPair) == 8);

/// The R-owned and shared dirty flags E raised since the last packet (Maxwell3D::DirtyState).
using FlagBits = Maxwell3D::DirtyState::Flags;

/// The draw manager's state for one draw, by value (no inline index vector: those draws stay lent).
/// A list draw (indirect_state.list_words, inside DrawEnd's list branch) also carries what R's
/// view needs of the indirect state: DrawEnd sets every other field of it to a constant.
struct DrawValues {
    Maxwell3D::Regs::PrimitiveTopology topology;
    u32 draw_mode; ///< Maxwell3D::DrawManager::DrawMode
    u32 base_index;
    u32 base_instance;
    u32 instance_count;
    u32 drawn_instances;
    u32 draw_element_count;
    Maxwell3D::Regs::VertexBuffer vertex_buffer;
    Maxwell3D::Regs::IndexBuffer index_buffer;
    u32 kind;      ///< the draw's UnlendKind (one bit)
    u8 indexed;    ///< RasterizerVulkan::Draw's is_indexed
    u8 indirect;   ///< DrawIndirect
    u8 draw_indexed; ///< the draw manager's own draw_indexed
    u8 reserved;
    u32 instance_arg; ///< RasterizerVulkan::Draw's instance_count argument
    u32 list_words;   ///< indirect_state.list_words (1: a list draw)
    u64 list_keep_id; ///< indirect_state.list_keep_id (0 unless a list draw)
    GPUVAddr list_address; ///< indirect_state.indirect_start_address (0 unless a list draw)
};
static_assert(sizeof(DrawValues) % 8 == 0 && offsetof(DrawValues, list_keep_id) % 8 == 0);

/// The six program hashes RefreshStages gives (ShaderCache::RefreshStages's unique_hashes).
struct Programs {
    std::array<u64, 6> hashes;
};

/// The texture and sampler pools (regs.tex_header, regs.tex_sampler and the linked-TSC mode).
struct Pools {
    GPUVAddr tic_address;
    u32 tic_limit;
    u32 tsc_linked; ///< regs.sampler_binding (1: the TSC index comes from the TIC handle)
    GPUVAddr tsc_address;
    u32 tsc_limit;
    u32 reserved;
};

/// What SyncFromEngine copies at a lent packet besides the bindings, by value (SpanEngine).
struct EngineValues {
    u32 engine_state; ///< Maxwell3D::EngineHint
    u8 native;        ///< the native macro mode (MacroEngine::native_turing)
    u8 reserved[3];
};
static_assert(sizeof(EngineValues) == 8);

/// SpanBinds: the count, then that many BindValue.
struct BindsHeader {
    u32 count;
    u32 reserved;
};
/// One constant buffer binding (Maxwell3D::state.shader_stages[stage].const_buffers[slot]).
struct BindValue {
    u8 stage;
    u8 slot;
    u8 enabled;
    u8 reserved;
    u32 size;
    GPUVAddr address;
};
static_assert(sizeof(BindValue) == 16);
/// Bindings in the engine's state (5 stages of 18): BindValue entries a packet carries at most.
inline constexpr u32 MaxBindValues =
    static_cast<u32>(Maxwell3D::Regs::MaxShaderStage * Maxwell3D::Regs::MaxConstBuffers);

/// A lent operation: run on R with E parked. `function(argument)` is E's code, unchanged.
struct LentCall {
    void (*function)(void*);
    void* argument;
};

struct PacketHeader {
    u32 bytes;        ///< the whole packet with its spans, a multiple of 8
    PacketKind kind;
    u8 spans;         ///< SpanBit
    u8 flags;         ///< PacketFlag
    u8 reason;        ///< LentReason (a lent draw) or 0
    u8 op;            ///< LentOp (a lent packet)
    /// reserved[0]: the draw's UnlendKind bit (TakeDrawKind). reserved[1], a draw packet with E's
    /// flag variant: the occlusion flag pre-pass bits E's lookup gave (OcclPrepassBit), which R's
    /// thread-local pre-pass state takes before the draw.
    u8 reserved[3];
    u32 delta_count;  ///< RegPair count (SpanDelta)
    u64 number;       ///< packets since R started, E's count (1 = the first)
    u64 position;     ///< ring A position of E's entry (GpuSplit::BackEntryPosition)
    u64 draw_seq;     ///< E's draw count since R started
    u64 buffer_number; ///< the draw's MemStage ring B packet number, 0 = none (lent draws)
    LentCall call;    ///< PacketKind::Lent
    s32 channel;      ///< the channel bound at the packet
    u32 reserved2;
    /// PacketKind::Draw with FRITCHII_ONEF_E_PIPELINE: the GraphicsPipeline E looked up for the
    /// draw (R binds it, no lookup of its own). Null otherwise.
    const void* pipeline;
};
static_assert(sizeof(PacketHeader) % 8 == 0);

// --- The register view (R) -----------------------------------------------------------------------

/// R's view: only packets write it (or a seed from E's engine under a lent packet). R's code reads
/// `regs` and `flags` the way it reads the engine's today.
struct RegisterView {
    Maxwell3D::Regs regs{};
    FlagBits flags{};
    DrawValues draw{};
    Programs programs{};
    Pools pools{};
    u64 packets{}; ///< packets applied
    bool seeded{};
};

/// The words E sends in deltas and R's verify compares: every register word except those R never
/// reads (constant buffer data words, macro upload and firmware words, the shadow scratch). Built
/// once (one_follower_view.cpp).
[[nodiscard]] bool Watched(u32 index) noexcept;

/// Which side owns a dirty flag (section 2): flags only the buffer side reads stay on E; flags
/// only R reads are forwarded; flags both read exist on both, raised on both, cleared by each.
enum class FlagOwner : u8 { E, R, Both };
[[nodiscard]] FlagOwner OwnerOf(u8 flag) noexcept;

// --- Lifetime and the thread (one_follower.cpp) ----------------------------------------------------

/// Reads the switches (GpuSplit::MaybeStart, before the back thread exists). Boot ones stick.
void ReadSwitches();
/// The Vulkan device names its driver (Device's constructor, before ReadSwitches): on AMD's
/// proprietary driver the one follower stays off unless FRITCHII_ONEF_AMD=1. `integrated`: the
/// device is an integrated GPU (an APU).
void NoteAmdProprietaryDriver(bool integrated) noexcept;
/// True once the Vulkan device said it runs AMD's proprietary driver (AMD safe defaults read it).
[[nodiscard]] bool IsAmdProprietaryDriver() noexcept;
/// AMD's proprietary driver on a discrete GPU (not an integrated one).
[[nodiscard]] bool IsAmdProprietaryDiscrete() noexcept;
/// FRITCHII_ONEF_OCCL (boot, read in ReadSwitches): occlusion flag variants with one follower on
/// (lent draws through the off path's choice, played draws through E's). An explicit value wins;
/// unset = on only on AMD's proprietary driver on a discrete GPU. False reproduces v6.2's rule
/// (no flag variant with one follower on).
[[nodiscard]] bool OcclWanted() noexcept;
/// FRITCHII_ONEF=1 at boot (R exists).
[[nodiscard]] bool On() noexcept;
/// The effective step after nesting (StepLevel), refreshed at frame ends and lent packets.
[[nodiscard]] StepLevel Step() noexcept;
/// True on R.
[[nodiscard]] bool OnRenderThread() noexcept;
/// R inside a lent call (E parked at its place): E's caches may be used.
[[nodiscard]] bool RenderInLentCall() noexcept;
/// R runs a lent call now (true on R inside a lent packet).
[[nodiscard]] bool InLent() noexcept;

/// E: run `function(argument)` on R with E parked. R first plays every packet before it. Counted
/// by `op` (and `reason` for a draw); the round trip is timed (the hop's price, step 1's number).
void Lend(LentOp op, LentReason reason, void (*function)(void*), void* argument);
template <typename F>
void Lend(LentOp op, LentReason reason, F& function) {
    Lend(op, reason, [](void* argument) { (*static_cast<F*>(argument))(); }, &function);
}
template <typename F>
void Lend(LentOp op, F& function) {
    Lend(op, LentReason::StepBelowUnlend, function);
}

/// E: the call site of the next Lend (the rasterizer's ONEF_LEND passes its function's name),
/// for the "lends by site" line (fable_b 11.5's pricing of items 3 to 7). A string literal.
void SetLendSite(const char* site) noexcept;

/// E: wait until R has played every packet pushed (E is about to touch an R-owned object itself).
void Drain();
/// E: packets R has not played.
[[nodiscard]] u64 Behind() noexcept;

/// Owner asserts: an R-owned object is used here (scheduler, texture/pipeline/query caches, fence
/// manager, state tracker); counted as Check::OwnerOnE unless on R, or on E with R parked or
/// drained. `where` names the object for the first reports.
void CheckRenderOwned(const char* where) noexcept;
/// The buffer cache or an engine is used here: Check::OwnerOnR when on R outside a lent call.
void CheckEngineOwned(const char* where) noexcept;

/// Counts a must-be-0 event (any thread). `what` names it in the first reports.
void Note(Check check, const char* what) noexcept;

// --- The engine side of the delta (part b, one_follower_view.cpp) ---------------------------------

/// Maxwell3D's write paths: register word `index` took a new value on E. Off: one load and a branch.
void NoteRegWrite(u32 index) noexcept;
/// E raised dirty flag `flag` outside the method write path (the dirty tables cover that path).
void NoteFlagRaised(u8 flag) noexcept;

/// E, at a draw or before a lent op (step 2 on): appends the spans that changed since the last
/// packet to `header` (SpanDelta, SpanFlags, SpanDraw, SpanPrograms, SpanPools) at `out`;
/// returns the bytes written.
std::size_t PackSpans(const Maxwell3D& engine, PacketHeader& header, std::span<u8> out);

/// R: applies a packet's spans to the view.
void ApplySpans(RegisterView& view, const PacketHeader& header, const u8* spans);

/// R, at a lent draw with PacketVerify (step 2): compares the view with E's engine (E parked);
/// counts Check::View*.
void VerifyView(const RegisterView& view, const Maxwell3D& engine);

/// R's view (R only, or E while R is parked).
[[nodiscard]] RegisterView& View() noexcept;

// --- Section 7: the stream-write guard -----------------------------------------------------------

/// E writes game memory from the stream (constant buffer words, inline uploads, copies, reports):
/// true when it lands on a page the texture cache tracks while R has unplayed packets. The caller
/// then makes that write lent (Check::StreamWrite counted). GuardStreamWrite below does both: it
/// drains R before the write, which is the write run lent, and the next draw is lent as well
/// (TakeStreamWriteLend, LentReason::StreamWrite).
[[nodiscard]] bool StreamWriteHits(DAddr address, u64 size) noexcept;
/// E, at a draw: true once after StreamWriteHits returned true (that draw is lent with
/// LentReason::StreamWrite). Implemented beside StreamWriteHits.
[[nodiscard]] bool TakeStreamWriteLend() noexcept;
/// E (FRITCHII_ESPLIT_CB_ASYNC's E1): true when a game memory write at [address, address + size)
/// lands on no page the stream-write guard acts on (texture or shader pages, E's pools), so it
/// can wait in E2's queue. False also when the filters are not set.
[[nodiscard]] bool StreamPagesClear(DAddr address, u64 size) noexcept;
/// FRITCHII_ESPLIT: whether [address, address + size) touches a page that held shader code (false
/// when the filter is not set).
[[nodiscard]] bool OnShaderPages(DAddr address, u64 size) noexcept;

// --- R's draws (part c: renderer_vulkan/vk_one_follower_draw.cpp, one_follower_draw.cpp) ----------

/// R's side of the packets: the Vulkan rasterizer registers one at boot (SetDrawPlayer).
/// R's packet loop (part a) calls OnPacket for every packet after ApplySpans (a Lent packet before
/// its call runs, with E parked), then PlayDraw for a PacketKind::Draw.
class DrawPlayer {
public:
    /// Every packet: R's view engine takes the same spans; at a Lent packet (E parked) it also
    /// takes what deltas do not carry (constant buffer bindings, the engine hint, the macro mode,
    /// a new channel), and with PacketSeed everything.
    virtual void OnPacket(const PacketHeader& header, const u8* spans) = 0;
    /// PacketKind::Draw (step 4): records the draw from the packet and the view while E goes on.
    virtual void PlayDraw(const PacketHeader& header) = 0;
    /// R at a 0xe2c condition's stream place (E parked): R's mode on its view; from step 3 it is
    /// compared with E's `execute_on` (Check::BuildCondition). PlayDraw drops PacketConditional
    /// draws while it is False.
    virtual void Condition(bool engine_execute_on) = 0;
    /// fable_b 10.2 rules 1 and 2, around EVERY lent call (E parked), from step 3: before it,
    /// the R-owned objects go back to E's engine and the engine takes the R-owned flags the view
    /// has not consumed; after it, the view takes the engine's flags and the R-owned objects
    /// name the view again (the default on R outside lent calls).
    virtual void BeforeLent() {}
    virtual void AfterLent() {}
    /// R, a packet while the step is below 3: the R-owned objects name E's engine again.
    virtual void ViewOff() {}

protected:
    ~DrawPlayer() = default;
};
void SetDrawPlayer(DrawPlayer* player) noexcept;
[[nodiscard]] DrawPlayer* GetDrawPlayer() noexcept;

/// E (part a, ring B's owner of the hand-over): pushes a PacketKind::Draw packet whose header
/// carries `draw_seq` (the seq E filled ring B's packet with), `buffer_number` (that packet's push
/// index) and `flags` (PacketFlag), with the spans PackSpans gives; E goes on within
/// FRITCHII_ONEF_AHEAD.
void PushDraw(const Maxwell3D& engine, u64 draw_seq, u64 buffer_number, u8 flags,
              const void* pipeline = nullptr, u8 occl_prepass = 0);

/// PacketHeader::reserved[1] of a draw packet: OcclFlag's pre-pass state for E's flag variant.
enum OcclPrepassBit : u8 {
    OcclColourPrepass = 1,        ///< OcclFlag::t_colour_prepass
    OcclDiscardDirect = 2,        ///< OcclFlag::t_discard_direct
    OcclColourDiscardPrepass = 4, ///< OcclFlag::t_colour_discard_prepass
};

/// E, at the end of Maxwell3D::ProcessQueryCondition when the host did not take the condition:
/// from step 3, R computes the mode on its view at this place (in the lent call when one is
/// running, else in a lent one of its own) and compares (DrawPlayer::Condition).
void NoteCondition(bool execute_on);

// --- Added with part (a): the thread, the ring, the lend (one_follower.cpp) ------------------------

// Detail::t_engine and Detail::t_render (ShouldLend's one load): one_follower_hooks.h.

/// E's rasterizer entry points: true on E with one follower on; the call is then lent to R (or
/// pushed, step 4). On R and on every other thread the call runs where it is, as today.
[[nodiscard]] inline bool ShouldLend() noexcept {
    return Detail::t_engine;
}
/// GpuSplit::BackLoop at its start: this thread is E (no-op with FRITCHII_ONEF=0).
void MarkEngineThread() noexcept;
[[nodiscard]] inline bool OnEngineThread() noexcept {
    return Detail::t_engine;
}

/// GpuSplit::MaybeStart, before the back thread starts / GpuSplit::Stop, after it joined.
void Start(Core::System& system);
void Stop();

/// The hang report (FRITCHII_HANG_REPORT_S, no frame for that long while the game runs): logs
/// the front's wait and, with the one follower running, the watchdog's state line (positions,
/// E and R, watched locks). Read only, from any thread: wakes and changes nothing.
void ReportHangState();

/// The hang watchdog (always on with FRITCHII_ONEF=1, one wake a second): a lock whose holder it
/// names when E or R stop making progress (the rasterizer registers the buffer and texture
/// caches' mutexes). Up to four; `name` must outlive the session.
void WatchLock(const char* name, const Common::SpinRecursiveMutex* mutex) noexcept;

// R's draw player and E's routing of draws from step 3: part (c)'s DrawPlayer, SetDrawPlayer,
// PushDraw (above) and Vulkan::OneFollowerDraw::RouteDraw. Below step 3 the rasterizer lends
// every draw itself (RasterizerVulkan::OneFollowerLendDraw).

/// E, before Lend or PushDraw of a draw: what the draw is (DrawValues' kind, indexed, indirect
/// and the instance count argument). Lend puts the kind in the header (reserved[0], the UnlendKind
/// bit) for the counters, PackSpans puts all of it in the draw span. Consumed by the next packet;
/// a draw packet without it still carries the draw manager's values (kind 0).
void SetNextDraw(u32 kind, bool indexed, bool indirect, u32 instance_arg) noexcept;
void ClearNextDraw() noexcept;

/// E: pushes a packet R plays while E goes on (PacketKind::Draw or Delta), within the window
/// (FRITCHII_ONEF_AHEAD) and the ring's room. Fills number, position and draw_seq.
void Push(PacketHeader& header, std::span<const u8> spans);

/// The engine R's view mirrors (the bound channel's Maxwell3D): set by the channel bind (on R
/// under a lent packet, or on E). PackSpans reads it on E, the seed and VerifyView on R.
void SetEngine(Maxwell3D* engine) noexcept;
[[nodiscard]] Maxwell3D* Engine() noexcept;

/// GpuSplit's two positions (8.2): E stores its own `consumed`, then calls this, which raises
/// GpuSplit's `played` (what every front wait reads) to that position clamped to the oldest packet
/// R has not played; R raises `played` again after each packet it plays (step 4).
void EngineConsumed(u64 position) noexcept;

/// R: true while it plays an unlent draw packet (DrawPlayer::PlayDraw).
[[nodiscard]] bool PlayingUnlentDraw() noexcept;
/// Step 4's safety stop: an unlent draw R could not build (a descriptor count that does not match
/// its pipeline). Logs once at Critical; FRITCHII_ONEF_UNLEND then runs 0 for the session (every
/// draw lent again). Any thread.
void DisableUnlend(const char* why) noexcept;

/// FRITCHII_ONEF_QUERY_OPS (live; fable_b 11.1 item 1): render enables that cannot have set host
/// conditional rendering send nothing, counter resets and plain reports are render ops (no park).
[[nodiscard]] bool QueryOps() noexcept;
/// FRITCHII_ONEF_POSTED (live; fable_b 12.2): an R-side operation that returns nothing to E is
/// posted by value, lent only where R must read the engine or E needs the result. Applied so far
/// (fable_b 12.3's order): fix 1, the reports on buffer pages (as QUERY_OPS=2; QUERY_OPS on).
[[nodiscard]] bool Posted() noexcept;
/// fable_b 12.3's prices before building (the "One follower posting (dry)" line): E, at a lent
/// fence (a buffer flush pending or not) and at a lent gather (its ranges).
void CountLentFence(bool buffer_flush) noexcept;
void CountLentGatherRanges(u64 ranges) noexcept;
/// FRITCHII_ONEF_QUERY_OPS=2: plain reports on buffer cache pages are render ops too, when their
/// type lands later (at a fence or flush, lent), not inside CounterReport.
[[nodiscard]] bool QueryOpsOnBufferPages() noexcept;

/// E: a render op R runs at its stream place while E goes on: `function(payload copy)` on R,
/// `size` bytes (at most MaxOpPayloadBytes) copied into the packet. Elsewhere it runs in place.
inline constexpr std::size_t MaxOpPayloadBytes = 128;
/// `lands_in_memory`: the op writes game memory later on R (a report): the front's E-position
/// waits (FRITCHII_ONEF_WAITS_E) also wait for R to play it (GuestWriteOpMark).
/// `packet_flags`: PacketFlag bits for the op's header (PacketRenderState).
/// `carry` (FRITCHII_ONEF_CARRY_OPS on, fable_b 22.3 step 2): an op that reads nothing of R's
/// view (a samples counter reset or report) is not a packet of its own: E keeps it and the next
/// packet it pushes carries it (PacketCarried), R running it first. E sends what it keeps before
/// its `consumed` store (FlushCarried), so the front's waits on E's position stay exact.
void PushOp(LentOp op, void (*function)(void*), const void* payload, u32 size,
            bool lands_in_memory = false, u8 packet_flags = 0, bool carry = false);
/// E: sends the ops it keeps for the next packet (FRITCHII_ONEF_CARRY_OPS) as a packet of
/// their own, now. GpuSplit::StoreConsumed calls it before E's `consumed` store; Drain too.
void FlushCarried() noexcept;
/// FRITCHII_ONEF_OP_CLEAR (live; fable_b 13.4 group 3; step 4, the view tracked): clears as render
/// ops R plays from its view; E runs the clear's FlushCaching first.
[[nodiscard]] bool PostClears() noexcept;
/// FRITCHII_ONEF_FLUSH_E (live; fable_b 17.3 (a); step 4): E answers a flush it tests clean
/// itself instead of lending it (FlushRegion, MustFlushRegion).
[[nodiscard]] bool FlushOnEngine() noexcept;
/// FRITCHII_ONEF_SYNC_MIRROR (live; fable_b 16.2, dry): E keeps a mirror of R's samples counter
/// "started" bit and R compares it at every samples report ("sync mirror", must be 0).
[[nodiscard]] bool SyncMirrorOn() noexcept;
/// FRITCHII_ONEF_POST_FENCE (live; fable_b 16.3; step 4; immediate fences only): a sync point, a
/// reference or a forced fence release E finds clean (no query sync owed, no buffer flush to
/// commit) is posted; E runs the closing gather itself first. Stopped for the session by a "sync
/// mirror" mismatch of the kind that makes E's test wrong (R started, E not).
[[nodiscard]] bool PostFenceOn() noexcept;
/// FRITCHII_ONEF_LOOKUP_CHECK: whether this unlent draw's lookup is checked (every Nth).
[[nodiscard]] bool LookupCheckDue() noexcept;
/// FRITCHII_ONEF_DRY_RELEASE: on (refreshes g_dry_release). E.
[[nodiscard]] bool DryReleaseOn() noexcept;
/// FRITCHII_ONEF_SEM_E (with POST_FENCE on; refreshes g_sem_e): fable_b 19.4, 19.6, 20.3.
[[nodiscard]] bool SemOnE() noexcept;
/// FRITCHII_ONEF_SIGNAL_ON_R (with SEM_E on): SEM_E's guest-visible signals happen at R's played
/// place (payload writes on R in posted ops, sync points signalled by R, SignalFence lent). E.
[[nodiscard]] bool SignalOnR() noexcept;
/// E: a signal SEM_E would have made on E went to R instead (the fence decisions line).
void CountSignalOnR() noexcept;
/// E, right after the PushOp of an op that writes the report at `device` on R: E's reads of its
/// page wait until R played that op (13.0's landing pages, keyed to the op itself).
void NoteReportLandsAtLastPush(u64 device) noexcept;
/// FRITCHII_ONEF_SOFT_RELEASE_E: a non-forced ReleaseFences is answered on E.
[[nodiscard]] bool SoftReleaseOnE() noexcept;
/// FRITCHII_ONEF_POST_IDLE (with POST_FENCE on): WaitForIdle posted.
[[nodiscard]] int PostIdleOn() noexcept;
/// E: R played every packet E pushed.
[[nodiscard]] bool RenderCaughtUp() noexcept;
/// E: a posted fence that lands reports is not played by R yet (13.0's pages).
[[nodiscard]] bool LandingOutstanding() noexcept;
/// E (SEM_E): before a game-visible signal on E; the "signal ahead of landing" check.
void NoteEngineSignal(const char* what) noexcept;
/// E, at each WaitForIdle it could post (the fence decisions line).
enum class IdleDecision : u8 {
    PostedCaughtUp, ///< R had played everything
    PostedBehind,   ///< POST_IDLE=2: R behind, no landing due
    LentBehind,     ///< POST_IDLE=1: R behind
    LentLanding,    ///< POST_IDLE=2: a landing due
    LentSyncOwed,   ///< a report of another kind owes the sync
    Count,
};
void CountIdleDecision(IdleDecision decision) noexcept;
/// FRITCHII_ONEF_POST_FENCE_REPORT (with POST_FENCE on): counter fence reports posted.
[[nodiscard]] bool PostFenceReportOn() noexcept;
/// E: the report pages (device >> 12) since the last fence, which the next posted fence lands.
[[nodiscard]] const std::vector<u64>& UnassignedReportPages() noexcept;
/// E, at each fence it could post: what it did (dry counts, the fence decisions line).
enum class FenceDecision : u8 {
    Posted,           ///< posted, nothing for the buffer half
    PostedWithCommit, ///< posted, E ran the buffer half's commit (19.6 step 1)
    Off,              ///< lent: POST_FENCE off (or not at step 4)
    Delayed,          ///< lent: delayed fences (accuracy)
    Stopped,          ///< lent: POST_FENCE stopped for the session
    SyncOwed,         ///< lent: a report of another kind owes the query sync
    AheadDue,         ///< lent: memory stage marks ahead still to promote
    Count,
};
void CountFenceDecision(FenceDecision decision) noexcept;
/// Why PostFenceOn refuses: Off, Delayed or Stopped.
[[nodiscard]] FenceDecision PostFenceRefusal() noexcept;
/// R: an unused samples sync target explained by E's mirror superset (R's counter was stopped
/// at the reports E listed): counted, not a check.
void CountSyncTargetSuperset() noexcept;
/// Stops FRITCHII_ONEF_POST_FENCE for the session (a must-be-0 of its own fired).
void StopPostFence(const char* why) noexcept;

/// FRITCHII_ONEF_OP_DISPATCH (live, default 0; fable_b 11.1 item 5, 13.4 group 6; step 4, needs
/// FRITCHII_ONEF_E_PIPELINE=1): compute dispatches as posted packets. E looks the compute
/// pipeline up (a plain hit only), fills the buffer side into ring B by value (uniform and
/// storage binds, uploads, the written marks), reads the handle values, and pushes a dispatch op;
/// R takes the ring B packet by name and records the dispatch (pipeline, image views,
/// descriptors, vkCmdDispatch) while E goes on. Everything else stays lent, counted by reason.
/// Where each dispatch went (the "One follower dispatches" line). A dispatch counts one: the
/// first that applies, in this order. The two Tail kinds count posted dispatches that also lent
/// a part after the post.
enum class DispatchCount : u8 {
    Posted,
    NeedsEPipeline, ///< lent: FRITCHII_ONEF_E_PIPELINE=0 (R owns the lookup then)
    Stopped,        ///< lent: a must-be-0 stopped posting for the session
    Transient,      ///< lent: the view not seeded yet: after a channel bind with
                    ///< FRITCHII_ONEF_BIND_SEED=0 (the seed waits for the next lent packet)
    Diagnostic,     ///< lent: a frame dump, trace, flash watch, verify bits or the stage's ring
    NoRing,         ///< lent: no packets, no values by value, no packet ticks, no room, marks ahead
    Indirect,       ///< lent: indirect, bit 4 off
    LookupMiss,     ///< lent: the compute lookup is not a plain hit (R looks up in the lent call)
    TexelBuffers,   ///< lent: texture or image buffers (their binds need the texture cache)
    WritesImage,    ///< lent: writes an image, bit 2 off
    Upscaler,       ///< lent: an OoT upscaler pass or its depth reader (the replacement's hooks)
    Dimensions,     ///< lent: a grid over the device's limit (the lent path drops it)
    FillRefused,    ///< lent: the seam did not fill it (arena busy, no room at the fill)
    TailListCopies, ///< posted; its command list page copies lent after it
    TailProbes,     ///< posted; its write extent probes' copies lent after it
    ImageShadow,    ///< lent (bit 2 on): writes a byte-addressed image (the R16 shadow path)
    PostedImage,    ///< posted (bit 2): writes an image (EXP10: recorded unbounded on E)
    PostedIndirect, ///< posted (bit 4): indirect, its arguments' buffer obtained by E
    /// lent indirect dispatch whose grid words exceeded the device limit (the dry compare of
    /// IndirectGridOverLimit: the game's own, if not 0)
    LentGridOverLimit,
    /// FRITCHII_ONEF_DISPATCH_RANGES: a posted image-writing dispatch E recorded with its images'
    /// ranges (EXP10), not as "may write anything"
    ImageRanged,
    /// ... one E still recorded unbounded (an image without one device range, more than 8)
    ImageUnbounded,
    /// fable_b 27.9's count (R, every played image-writing dispatch): an image's real guest
    /// range touches a page the front snapshots for its DMA reads (expected about 0)
    ImageOnFrontPage,
    /// R: an image's real range (the texture cache's backing image) reaches past E's record
    /// (a view into a larger image: the lent path would record more; not a must-be-0)
    ImageWiderThanRecord,
    /// lent: FRITCHII_ASYNC_COMPUTE_SKIP on and the pipeline not built yet (the lent call
    /// decides to skip or wait before anything is marked)
    Unbuilt,
    Count,
};
/// The kinds that count a posted dispatch a second time (not in the line's total).
[[nodiscard]] constexpr bool DispatchCountIsExtra(DispatchCount kind) noexcept {
    return kind == DispatchCount::TailListCopies || kind == DispatchCount::TailProbes ||
           kind == DispatchCount::PostedImage || kind == DispatchCount::PostedIndirect ||
           kind == DispatchCount::LentGridOverLimit || kind == DispatchCount::ImageRanged ||
           kind == DispatchCount::ImageUnbounded || kind == DispatchCount::ImageOnFrontPage ||
           kind == DispatchCount::ImageWiderThanRecord;
}
/// FRITCHII_ONEF_OP_DISPATCH's bits (any non-zero value turns the switch on).
enum DispatchPostBit : u32 {
    DispatchPostPlain = 1,    ///< direct dispatches that write no image (any value != 0)
    DispatchPostImages = 2,   ///< + dispatches that write images (R records the texture side)
    DispatchPostIndirect = 4, ///< + indirect dispatches (E obtains the arguments' buffer)
};
/// E: FRITCHII_ONEF_OP_DISPATCH's value (0 = off).
[[nodiscard]] u32 DispatchPostBits() noexcept;
inline constexpr std::array<const char*, static_cast<size_t>(DispatchCount::Count)>
    dispatch_count_names{
        "posted",           "lent: E_PIPELINE off", "lent: stopped",       "lent: view not ready",
        "lent: diagnostic", "lent: no ring",        "lent: indirect",      "lent: lookup miss",
        "lent: texel buffers", "lent: writes an image", "lent: upscaler", "lent: grid limit",
        "lent: fill refused", "posted + list copies lent", "posted + extent probes lent",
        "lent: byte-addressed image write", "posted writing images", "posted indirect",
        "lent indirect grid over the limit (dry)", "posted images ranged",
        "posted images unbounded", "images on a front read page", "images wider than E's record",
        "lent: unbuilt pipeline (FRITCHII_ASYNC_COMPUTE_SKIP)",
};
/// E: the first reason a dispatch cannot be posted before its lookup, Count when it may be;
/// DispatchCount::Posted (never returned) is not a reason. Returns Count too when the switch is
/// off or the step is below 4 with `off` set: nothing is counted then.
[[nodiscard]] DispatchCount PostDispatchBlocker(bool& off) noexcept;
/// Stops FRITCHII_ONEF_OP_DISPATCH for the session (a must-be-0 of its own fired). Any thread.
void StopPostDispatch(const char* why) noexcept;
/// E: counts where a dispatch went.
void CountDispatch(DispatchCount kind) noexcept;
/// FRITCHII_ONEF_DISPATCH_RANGES (live, default 0; fable_b 27.9): E records a posted
/// image-writing dispatch's images by their ranges, from their descriptors, and a lent dispatch
/// records on R with E's recording state (the lent path's own records, confirmed). False once a
/// "dispatch image range" stopped it.
[[nodiscard]] bool DispatchRangesOn() noexcept;
void StopDispatchRanges(const char* why) noexcept;
/// The front (Maxwell3D::FrontDmaRead): it took a snapshot of device page `page` (4 KiB), a page
/// its DMA reads are answered from. Kept with one follower on only; never cleared.
void NoteFrontReadPage(u64 page) noexcept;
/// Any thread: whether [address, address + size) touches a page NoteFrontReadPage noted.
[[nodiscard]] bool FrontReadPagesAny(u64 address, u64 size) noexcept;
/// E: the packet number of the packet it pushed last.
[[nodiscard]] u64 PushedNow() noexcept;
/// fable_b 13.0 (POST_FENCE), E: a report to `device`'s page was made; a fence op with `number`
/// was posted (the reports since the last fence land there); a fence was lent (everything
/// before it landed with E parked).
void NoteReportPage(u64 device) noexcept;
void NoteFencePosted(u64 number) noexcept;
void NoteFenceLent() noexcept;
/// R, at a samples report: E's bit against R's streamer.
void CountSyncMirror(bool engine_bit, bool render_bit) noexcept;
/// R, at a draw it skips (no pipeline, configure failed, ...) with the samples count enabled in
/// its view: E's mirror counted the draw, R's counter will not start for it
/// (FRITCHII_ONEF_MIRROR_STOP=2 explains the next "E's mirror started, R's counter not").
void NoteSkippedSamplesDraw() noexcept;
/// E: the view is tracked and seeded at R (a posted op reads it).
[[nodiscard]] bool ViewReady() noexcept;

/// FRITCHII_ONEF_WAITS_E (live; fable_b 11.2): the front's waits on game memory, QMDs,
/// semaphore flushes and lists read E's position (`consumed`), the rest R's (`played`).
[[nodiscard]] bool WaitsE() noexcept;
/// FRITCHII_ONEF_WAITS_R (live): overrides WAITS_E and SEM_E's lagged acquire check, every front
/// wait ends at R's played position.
[[nodiscard]] bool WaitsR() noexcept;
/// The newest render op that writes game memory on R (a plain report): its ring A entry
/// position + 1, 0 = none yet. E stores it before it stores a `consumed` past that entry.
[[nodiscard]] u64 GuestWriteOpMark() noexcept;
/// E (fable_b 19.5 step 4, FRITCHII_E2C_ON_E): before E reads game memory at its stream place
/// that a plain report op it pushed may write (the 0xe2c read), waits until R played the newest
/// such op, the rule the front's E-position waits use (GpuSplit::Reached). True when it waited.
/// Not on E (one follower off, the single back): false at once, nothing to wait for.
bool WaitGuestWriteOps() noexcept;

/// FRITCHII_ONEF_INVAL_E (live; fable_b 11.1 item 2; needs FRITCHII_ONEF_E_PIPELINE=1 and step
/// 4): E runs the draw's invalidation (the buffer half) itself when no range touches a page an
/// image, a shader, a query or a pushed report op was ever on; otherwise it stays lent.
[[nodiscard]] bool InvalOnEngine() noexcept;
/// FRITCHII_ONEF_COPY_E (live; with FRITCHII_ONEF_INVAL_E's conditions and stamps): inline
/// uploads (AccelerateInlineToMemory) run on E; R's halves only where their pages are hit.
[[nodiscard]] bool CopyOnEngine() noexcept;
/// FRITCHII_ONEF_GATHER_E (live; with FRITCHII_ONEF_INVAL_E's conditions): the guest's collected
/// writes (InvalidateGPUCache's gather) are taken and applied to the buffer cache on E; R's
/// half (texture, shader caches) only where their filters are hit (then the gather is lent).
[[nodiscard]] bool GatherOnEngine() noexcept;
/// fable_b 13.1: whether E's pending buffer commands of memory seam site `site` (MemStage::Site)
/// are posted as a ring B op packet (R plays them at a BufferOp) instead of lent. Group 1:
/// FRITCHII_ONEF_OP_INLINE, the inline upload site. Step 4 only.
[[nodiscard]] bool PostOpSite(u8 site) noexcept;
/// The op packets' line (E counts): posted (with their entry bytes), or lent and why.
enum class OpCount : u8 {
    Posted,
    LentHeldFill,   ///< a sync inside a held (draw) fill: R plays part of E's arena (R3)
    LentRingFill,   ///< inside E's draw fill (FillRingDraw): its own ring B packet
    LentNoRoom,     ///< ring B short of room
    LentNoTicks,    ///< no packet ticks (R5 needs the op's number in their accounting)
    Count,
};
void CountOpPacket(OpCount kind, u64 bytes = 0) noexcept;
/// E: stamps [address, address + size)'s pages with the next packet's number (before the
/// filters' second look: WrittenAheadOfRender's Dekker pair).
void StampPageWrites(u64 address, u64 size) noexcept;
/// Where each of E's invalidations went (the "One follower invalidations" line).
enum class InvalKind : u8 {
    OnEngine, ///< run on E, nothing for R
    Texture,  ///< lent: a range on a texture page
    Shader,   ///< lent: a range on a shader page
    Query,    ///< lent: a range on a query page or a pushed report op's
    Recheck,  ///< lent: no hit, then a hit at the second look (R registered meanwhile)
    InlineOnEngine,  ///< FRITCHII_ONEF_COPY_E: an inline upload run on E
    InlineRHalf,     ///< ... whose texture/shader half was lent alone (in-place upload, a hit)
    InlineQuery,     ///< ... lent whole: a query page or a pushed report op's
    GatherOnEngine,  ///< FRITCHII_ONEF_GATHER_E: a gather applied on E
    GatherLent,      ///< ... lent: a range on a texture or shader page
    Count,
};
void CountInvalidation(InvalKind kind) noexcept;

/// The pricing counter of fable_b 11.5: LentOp::Query by call site.
enum class QueryKind : u8 {
    ConditionSkipped,      ///< item 1(a): a render enable that could not change anything, not sent
    ConditionOverride,     ///< lent: override always or never
    ConditionTrue,         ///< lent: mode True
    ConditionFalse,        ///< lent: mode False
    ConditionConditional,  ///< lent: mode Conditional
    ConditionIfEqual,      ///< lent: mode IfEqual
    ConditionIfNotEqual,   ///< lent: mode IfNotEqual
    Reset,                 ///< a counter reset (lent, or a render op with QUERY_OPS)
    PlainReport,           ///< a report, not a fence (lent, or a render op with QUERY_OPS)
    ReportOnBufferPage,    ///< a plain report kept lent by the guard (a lend reason, common)
    BufferPageOp,          ///< QUERY_OPS=2: a plain report on a buffer page as a render op
    FenceReport,           ///< a fence-flagged report (lent; item 4)
    PayloadRelease,        ///< a payload release (lent; item 4)
    PayloadReleaseOnE,     ///< FRITCHII_ONEF_SEM_E: a payload release E wrote (R notes the sync)
    PayloadFenceOnE,       ///< FRITCHII_ONEF_SEM_E: a fence-flagged payload report E wrote
    FenceReportPosted,     ///< FRITCHII_ONEF_POST_FENCE_REPORT: a counter fence report posted
    PayloadReleaseOnR,     ///< FRITCHII_ONEF_SIGNAL_ON_R: a payload release R writes (posted op)
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(QueryKind::Count)> query_kind_names{
    "condition skipped", "condition override", "condition true", "condition false",
    "condition conditional", "condition if equal", "condition if not equal", "reset",
    "plain report", "report on buffer page", "report on buffer page as op", "fence report",
    "payload release", "payload release on E", "payload fence on E", "fence report posted",
    "payload release on R",
};
/// fable_b 19 step 2's dry counter: which test made a draw lend with LentReason::Transient, in
/// the order UnlentBlocker and FillRing (vk_one_follower_draw.cpp) try them. Logged per second
/// and per draw kind on the "One follower transient lends" line (with the hand-over line's 10 s
/// window), so each fix below is checked against its own count.
enum class TransientTest : u8 {
    ViewNotSeeded, ///< no lent packet seeded R's view yet, or the step changed since the seed
    Channel,       ///< the view follows another channel's engine (a bind since the last lent packet)
    EngineHint,    ///< Maxwell3D::engine_state differs from what the last lent packet gave the view
    NativeMode,    ///< the native macro mode differs likewise
    BankReaders,   ///< R-side readers of the constant buffer bindings are on (TEX_BANK_GEN, ...)
    BufferPaths,   ///< a path that touches the buffer cache on R or changes who fills
    VertexLimit,   ///< first + count past 16M vertices (Draw skips it)
    IndirectReset, ///< a non-indexed single-instance native draw of points (IsNx2IndirectReset)
    FillTexel,     ///< E's fill refused: texel buffers (they need the texture cache)
    FillRoom,      ///< E's fill refused: no room in ring B
    FillOther,     ///< E's fill refused: packets or values by value off, marks ahead, the fill
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(TransientTest::Count)>
    transient_test_names{
        "view not seeded", "channel",      "engine hint",     "native mode",
        "bank readers",    "buffer paths", "vertex limit",    "indirect reset",
        "fill: texel buffers", "fill: no ring room", "fill: other",
    };
/// E: a draw of UnlendKind `kind` is lent as Transient because of `test`.
void CountTransient(TransientTest test, u32 kind) noexcept;
/// FRITCHII_ONEF_UNLENT_RESET (live; fable_b 19 step 2; step 4): a non-indexed single-instance
/// native draw of points is lent only when E's own IsNx2IndirectReset test says it is one;
/// the others are pushed with PacketNoReset. Stopped for the session by "unlent reset".
[[nodiscard]] bool UnlentResetOn() noexcept;
void StopUnlentReset(const char* why) noexcept;
/// FRITCHII_ONEF_UNLENT_VALUES (live; fable_b 19 step 2; step 4): the engine hint, the macro mode
/// and the constant buffer bindings (what SyncFromEngine copies at a lent packet besides the
/// registers) go to R's view by value in the packets when they change (SpanEngine, SpanBinds),
/// so a draw after a change of them is not lent to sync the view. E takes it on at a lent packet
/// (where R's view takes the engine's values) and gives it up the same way. Stopped for the
/// session by "view values".
[[nodiscard]] bool UnlentValuesOn() noexcept;
void StopUnlentValues(const char* why) noexcept;
/// E: where FRITCHII_ONEF_UNLENT_VALUES stands for R's view: 0 = off (the values come with lent
/// packets only), 1 = by value since the last lent packet (every change is in a packet), 2 =
/// the switch went off since (the view may be newer than the last lent packet: draws are lent
/// until the next lent packet).
[[nodiscard]] u8 EngineValuesMode() noexcept;
/// FRITCHII_ONEF_BIND_SEED (live; fable_b 19 step 2; step 4): a lent call that binds another
/// channel (RasterizerVulkan::BindChannel, LentOp::Channel) re-seeds E's tracking and R's views
/// from the new channel's engine at its end, E still parked: what the next lent packet's seed
/// did. So the first draw after a bind is not lent ("channel" on the transient lends line). R's
/// part (c) view engines are kept per channel. Stopped for the session by "bind seed view".
[[nodiscard]] bool BindSeedOn() noexcept;
void StopBindSeed(const char* why) noexcept;

/// E, before the Lend or PushOp of a query: its kind (counted with the park's time).
void SetQueryKind(QueryKind kind) noexcept;
/// E: a query handled without a packet (item 1(a)).
void CountQueryKind(QueryKind kind) noexcept;

/// Live switch values (refreshed with Step()).
[[nodiscard]] u32 UnlendMask() noexcept;
[[nodiscard]] int VerifyEvery() noexcept;
[[nodiscard]] int Ahead() noexcept;

// The counters and the 10 s lines are part (d)'s: one_follower_stats.h (Note() is defined there).

// --- Added with part (b): the engine side's write tracking (one_follower_view.cpp) ---------------

// g_track_writes and NoteRegWrite: one_follower_hooks.h.

/// E: the view's spans need a full seed first (tracking just started, a step change or a channel
/// bind): the next lent packet carries PacketSeed. Unlent draws wait for it (LentReason::Transient).
[[nodiscard]] bool ViewReady() noexcept;

/// The texture cache's "pages that ever held an image" filter (set by the rasterizer at creation):
/// StreamWriteHits looks there.
void SetTexturePages(const PageFilter* filter) noexcept;
/// The shader cache's code pages (fable_b 10.1 row 8): R reads programs there too.
void SetShaderPages(const PageFilter* filter) noexcept;
/// The rasterizer's union of the texture, shader, query and report filters (fable_b 14.4 cut 5):
/// with FRITCHII_ONEF_CB_LEAN & 2 the stream-write guard tests it first (a superset of the two
/// it needs).
void SetUnionPages(const PageFilter* filter) noexcept;
/// FRITCHII_ONEF_CB_LEAN (live, default 0; fable_b 23.1 cut 3), bits: 1 = a single constant
/// buffer word goes through the block write (Maxwell3D::CallConstBufferWords) as runs of two or
/// more do; 2 = the stream-write guard tests the union filter first.
[[nodiscard]] int CbLean() noexcept;
/// E's stream writes (Maxwell3D's constant buffer words, memory_manager.cpp's cached writes): on a
/// texture page they are counted (StreamWritesOnTexturePages, informational at every step), and
/// with R behind (step 4) Check::StreamWrite is counted and E drains R before the write lands.
void GuardStreamWrite(DAddr address, u64 size) noexcept;
/// Writes that landed on texture pages, any step (section 7's run 4 number; not a must-be-0).
[[nodiscard]] u64 StreamWritesOnTexturePages() noexcept;

/// R, under a lent packet with PacketSeed (or a channel bind): the view takes the engine's
/// registers, flags, draw state and pools in full.
void SeedView(RegisterView& view, const Maxwell3D& engine);
/// R, around a lent call (step 2+): before, at step 4 the engine's forwarded flags take the
/// view's (R's truth); after, the view takes what the lent code left in the engine.
void BeforeLentCall(RegisterView& view, Maxwell3D& engine);
void AfterLentCall(RegisterView& view, Maxwell3D& engine);

/// E, step 4 (part c's pusher), right after its buffer fill of an unlent draw: the forwarded flags
/// the fill left are E's baseline for the next packet's flag span (the fill clears E's copies of
/// the shared flags; their raises were packed before it).
void NoteEngineFlags(const Maxwell3D& engine) noexcept;

/// FRITCHII_ESPLIT (fritchii_e_split_thread.h): E's per-call state on its thread (the draw
/// SetNextDraw described, the query kind and lend site of the next lend, the trace's last record
/// end). E2 runs pieces of E's work; the state moves with each one (TakeEngineTls clears the
/// caller's copy, PutEngineTls installs one), so every call sees what it would on one thread.
struct EngineTls {
    DrawValues next_draw;
    bool next_draw_set;
    QueryKind query_kind;
    const char* lend_site;
    u64 e_last;
};
void TakeEngineTls(EngineTls& out) noexcept;
void PutEngineTls(const EngineTls& in) noexcept;
/// FRITCHII_ESPLIT: this thread (E2) is E too (lends, pushes, the engine thread's checks); the
/// watchdog and R's contended-lock report name it as E.
void MarkEngineHelperThread() noexcept;
/// FRITCHII_ESPLIT_DEPTH > 0: PushDraw in two halves. E1 packs the draw packet's spans at the
/// draw (its registers are past E2's place by the time E2 pushes), into `buffer` (room for
/// PackedDrawRoom() bytes); E2 pushes it after its fill with the ring B packet's number.
[[nodiscard]] std::size_t PackedDrawRoom() noexcept;
[[nodiscard]] std::size_t PackDraw(const Maxwell3D& engine, u64 draw_seq, u8 flags,
                                   const void* pipeline, PacketHeader& header, u8* buffer) noexcept;
void PushDrawPacked(PacketHeader& header, u8* buffer, std::size_t body_bytes,
                    u64 buffer_number) noexcept;

namespace Detail {
/// Between one_follower.cpp and one_follower_view.cpp.
/// The draw SetNextDraw described, or null (E; PackSpans's draw span).
[[nodiscard]] const DrawValues* NextDraw() noexcept;
/// E's write tracking stops (step below 2) or its engine changed: the next lent packet seeds the
/// view in full. On E, or on R under a lent packet (E parked).
void ViewTrackingStopped() noexcept;
/// FRITCHII_ONEF_BIND_SEED, R at the end of a lent call that changed the bound channel (E
/// parked): E's tracking starts over from `engine` and `view` is seeded from it, as PackSpans'
/// seed and Play's SeedView do at the next lent packet with PacketSeed.
void SeedAfterBind(RegisterView& view, const Maxwell3D& engine);
} // namespace Detail

} // namespace VideoCommon::OneFollower
