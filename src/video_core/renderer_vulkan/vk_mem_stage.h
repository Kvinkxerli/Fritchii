// SPDX-FileCopyrightText: Copyright 2026 Fritchii
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/mem_read_log.h"
#include "video_core/memstage_shadow.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_compute_pass.h"
#include "video_core/renderer_vulkan/vk_mem_stage_hook.h"
#include "video_core/renderer_vulkan/vk_seam_clock.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Common {
class SpinRecursiveMutex;
}

namespace Tegra {
class GPU;
class MemoryManager;
}

namespace Vulkan {

struct BufferCacheParams;
class Device;
class MemoryAllocator;
class MemPlayer;
class Scheduler;

class MemStage {
public:
    enum class Cmd : u8 {
        UploadAhead,
        UploadBatched,
        Copy,
        PreCopyBarrier,
        PostCopyBarrier,
        Clear,
        QuadLutUpload,
        NullInit,
        BindIndex,
        BindIndexAlways,
        BindVertex,
        BindVertices,
        BindXfb,
        BindXfbs,
        Convert,
        Descriptors,
        Count,
    };

    enum Flag : u8 {
        OutsideRenderPass = 1,
        Barrier = 2,
        DynamicState = 4,
        ForgetOnly = 8,
        KeepSize = 0x10, ///< Full vertex binds: a clean run, dropped while still bound at any size
        Played = 0x80, ///< Played early by a sync inside a held fill (SEAM = 2)
    };

    struct Entry {
        Cmd kind;
        u8 flags;
        u16 count;
        u32 bytes;
    };

    struct CopyPayload {
        VkBuffer src;
        VkBuffer dst;
        u32 count;
        u32 reserved;
    };
    struct ClearPayload {
        VkBuffer dst;
        u64 offset;
        u64 size;
        u32 value;
    };
    struct QuadLutPayload {
        VkBuffer src;
        VkDeviceSize src_offset;
        VkBuffer dst;
        VkDeviceSize size;
    };
    struct NullInitPayload {
        VkBuffer buffer;
    };
    struct IndexPayload {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkIndexType type;
    };
    struct VertexOne {
        VkBuffer buffer;
        u32 index;
        u32 offset;
        u32 size;
        u32 stride;
    };
    struct VerticesPayload {
        u32 first;
        u32 count;
    };
    struct XfbOne {
        VkBuffer buffer;
        u32 index;
        u32 offset;
        u32 size;
    };
    struct XfbsPayload {
        u32 count;
        u32 reserved;
    };
    enum class Pass : u8 { Uint8, QuadIndexed, VertexRepack };
    struct ConvertPayload {
        Pass pass;
        ComputePassRun run;
    };
    struct DescriptorsPayload {
        u32 count;
        u32 reserved;
    };

    struct Span {
        u32 begin;
        u32 end;
    };
    struct BufferRef {
        VkBuffer buffer;
        u32 offset;
    };
    /// The OoT temporal trackers' uniform slots in a draw packet (fable_memstage_6b_plan.md 5.1):
    /// vertex stage slots 3, 4 and 6 as ForEachBoundBuffer visits them, and whether the bytes
    /// the trackers read are GPU-modified (slot 3: the first 0x230; slots 4 and 6: the binding).
    /// Filled by FillDraw while the trackers are on, so they need not ask the cache.
    struct TrackerSlot {
        DAddr address;
        u32 size;
        bool bound;
        bool gpu_modified;
    };
    struct TrackerRecord {
        std::array<TrackerSlot, 3> slots;
        bool filled;
        /// FRITCHII_FILL_BRACKET_LEAN (fable_b 25.2 cut h): bit i, slot i is one the lean record
        /// leaves out (at 1 left unbound; at 2 and 3 filled anyway). R counts reading one (must
        /// be 0; FillCuts::NoteTrackerRead).
        u8 lean_skipped;
        /// Step 7 (the window open): the bytes each tracker reads, copied at the fill, as many
        /// per slot (0: not copied, the tracker reads game memory); MemStage::TrackerBytes.
        std::array<u32, 3> copied;
        static constexpr std::array<u32, 3> Slots{3, 4, 6};
        static constexpr u32 CameraBytes = 0x230;
        /// The actor tracker's draw (its vertex program) and what it reads of slots 4 and 6.
        static constexpr u64 ActorShader = 0x28c6e527634f814aULL;
        static constexpr u32 OriginOffset = 0x40;
        static constexpr u32 OriginBytes = 3 * sizeof(float);
        static constexpr u32 BonesSlotSize = 13568; ///< the binding the actor's draw has
        static constexpr u32 BonesBytes = 280 * 12 * sizeof(float); ///< what the tracker reads
    };
    struct DrawPacket {
        u64 number;
        u32 flags;
        u32 repack_mask;
        u32 repack_dirty;
        Span geometry;
        std::array<Span, 5> stage;
        std::array<Span, 5> stage_descriptors;
        BufferRef gpu_count;
        BufferRef indirect;
        BufferRef indirect_count;
        Span reads;
        Span handle_values;
        TrackerRecord trackers;
    };
    struct DispatchPacket {
        u64 number;
        u32 flags;
        Span before;
        Span descriptors;
        BufferRef indirect;
        Span after;
        Span reads;
        Span handle_values;
        std::array<u32, 64> qmd;
        std::array<u64, 8> null_slot_values;
        u32 null_slot_count;
    };
    struct OpPacket {
        u64 number;
        Span entries;
        BufferRef result;
    };

    enum class Site : u8 {
        Geometry,
        Stage,
        DispatchBefore,
        DispatchIndirect,
        DispatchAfter,
        DrawCount,
        DrawIndirect,
        ListWords,
        ListArgs,
        ListPages,
        Invalidation,
        Download,
        DownloadForRead,
        Frame,
        Dma,
        InlineUpload,
        Unmap,
        ImageCopy,
        Query,
        Fence,
        Draw,
        Dispatch,
        Count,
    };

    /// DrawPacket and DispatchPacket flags.
    enum PacketFlag : u32 {
        LateFill = 1,   ///< Texel buffers: filled where the calls are (as with SEAM = 1)
        Held = 2,       ///< Filled before the texture work, its sections played later
        RestPlayed = 4, ///< Configure did not play every section: PlayRest did
        PostCopyBarrier = 8, ///< THREAD = 2: the fill uploaded; the back records the barrier
        /// THREAD = 2, a list draw GPUMem served (FRITCHII_MEMSTAGE_LIST_DRAWS): `indirect` is the
        /// buffer and offset of its two words (the cache's GetDrawIndirectBuffer at the fill).
        ListArgs = 16,
    };

    /// THREAD = 2: a facts table entry (spec 4.4), copied by value from the pipeline.
    struct FactsSsbo {
        u32 cbuf_index;
        u32 cbuf_offset;
        u32 indirect_offset;
        bool is_written;
        bool operator==(const FactsSsbo&) const = default;
    };
    static constexpr u32 MaxFactsSsbos = 16;
    struct StagedFacts {
        std::array<u32, 5> uniform_masks;
        VideoCommon::UniformBufferSizes uniform_sizes;
        std::array<std::array<FactsSsbo, MaxFactsSsbos>, 5> ssbos;
        std::array<u8, 5> ssbo_counts;
        u32 stages;
        bool has_texel_buffers;
        bool uses_descriptor_buffer;
        bool dynamic_vertex_input;
        std::array<u64, 2> hashes;
        /// The programs' infos, for the handle values the stage reads at its fill.
        std::shared_ptr<const std::array<Shader::Info, 5>> infos;
        /// FRITCHII_MEMSTAGE_SETUP_KEEP=2: every field, the infos by pointer (one table entry
        /// hands out one pointer; a new one comes with a new generation).
        bool operator==(const StagedFacts&) const = default;
    };

    /// What FillDraw needs from the graphics pipeline (spec 4.2's StageFacts).
    struct DrawFacts {
        const std::array<u32, 5>* uniform_masks;
        const VideoCommon::UniformBufferSizes* uniform_sizes;
        const Shader::Info* stage_infos; ///< 5 of them
        u32 stages;                      ///< The stages Configure prepares, one bit each
        bool has_texel_buffers;
        bool uses_descriptor_buffer;
        std::array<u64, 2> hashes; ///< Vertex and fragment program, for the logs
        const StagedFacts* staged{}; ///< THREAD = 2: the storage binds from here
        bool reads_handles{};        ///< True when at least one graphics stage reads handles
    };

    /// What FillDispatch needs from the compute pipeline (spec 4.3's ComputeFacts).
    struct DispatchFacts {
        u32 uniform_mask;
        const VideoCommon::ComputeUniformBufferSizes* uniform_sizes;
        const Shader::Info* info;
        u64 shader_hash;
        bool has_texel_buffers;
        bool uses_descriptor_buffer;
    };

    using Cache = VideoCommon::BufferCache<BufferCacheParams>;

    enum class Sync : u8 {
        Finish,
        Wait,
        SubmitForHostRead,
        MultiRange,
        KeptWords,
        Count,
    };

    class Arena {
    public:
        explicit Arena(u32 initial_capacity);
        ~Arena();

        [[nodiscard]] u32 Size() const noexcept {
            return cursor;
        }
        void Reset() noexcept {
            cursor = 0;
        }
        /// Room for `bytes` more (a multiple of 8) at the end; returns its offset.
        u32 Reserve(u32 bytes);
        [[nodiscard]] u8* At(u32 offset) noexcept {
            return data.get() + offset;
        }
        [[nodiscard]] const u8* At(u32 offset) const noexcept {
            return data.get() + offset;
        }
        void Swap(Arena& other) noexcept {
            data.swap(other.data);
            std::swap(capacity, other.capacity);
            std::swap(cursor, other.cursor);
        }

    private:
        std::unique_ptr<u8[]> data;
        u32 capacity;
        u32 cursor{};
    };

    /// A fill bracket: with SEAM on, the entries pushed inside wait for a play.
    /// The seam lean (Lean(): EXP123 = 0, no packet ticks, no stats line): the bracket only keeps
    /// this thread's fill depth; nested brackets follow their outermost one (t_fill_lean).
    class Fill {
    public:
        Fill(MemStage& stage_, Site site_, u32 index_ = 0) : stage{stage_} {
            if (t_fill_depth++ == 0) {
                t_fill_lean = stage_.lean.load(std::memory_order_relaxed);
            }
            lean = t_fill_lean;
            if (!lean) [[unlikely]] {
                Open(site_, index_);
            }
        }
        ~Fill() {
            if (lean) [[likely]] {
                --t_fill_depth;
                return;
            }
            Close();
        }
        Fill(const Fill&) = delete;
        Fill& operator=(const Fill&) = delete;

    private:
        /// The full bracket (t_fill_depth already counts this one) and its end.
        void Open(Site site_, u32 index_);
        void Close();

        MemStage& stage;
        Site site{};
        u32 index{};
        Site previous_site{};
        u32 begin{};
        bool lean{};
        bool reads{}; ///< This bracket started an operation packet's read log
        bool stage_op{}; ///< Step 7: GPUMem's operation, opened with its mark (stamps per thread)
        std::optional<VideoCommon::Nx2Profile::Scope> zone;
    };

    /// A fill bracket that plays what it filled when it ends.
    class FillPlay {
    public:
        FillPlay(MemStage& stage_, Site site, u32 index = 0)
            : stage{stage_}, fill{std::in_place, stage_, site, index} {}
        ~FillPlay() {
            fill.reset();
            stage.PlayIfOwned();
        }
        FillPlay(const FillPlay&) = delete;
        FillPlay& operator=(const FillPlay&) = delete;

    private:
        MemStage& stage;
        std::optional<Fill> fill;
    };

    /// Plays what is pending, then runs a sync: what the sync records is not counted.
    class SyncScope {
    public:
        SyncScope(MemStage* stage, Sync kind);
        ~SyncScope();
        SyncScope(const SyncScope&) = delete;
        SyncScope& operator=(const SyncScope&) = delete;
    };

    explicit MemStage(MemPlayer& player);
    ~MemStage();

    MemStage(const MemStage&) = delete;
    MemStage& operator=(const MemStage&) = delete;

    [[nodiscard]] MemPlayer& Player() noexcept {
        return player;
    }

    /// FRITCHII_MEMSTAGE_TICKS >= 1: the packet numbers and published ticks; null at 0.
    [[nodiscard]] SeamTicks* Ticks() const noexcept {
        return ticks.get();
    }

    /// Runs `fill` in a fill bracket, then plays: for the calls that return a buffer the caller
    /// records with.
    template <typename Func>
    decltype(auto) Ref(Site site, Func&& fill) {
        struct PlayAtEnd {
            MemStage& stage;
            ~PlayAtEnd() {
                stage.PlayIfOwned();
            }
        } play{*this};
        Fill bracket{*this, site};
        return fill();
    }

    /// Plays every pending entry this thread pushed.
    void Play();
    /// Play, without the call when no thread owns the arena: Play does nothing then (it plays only
    /// for the owner, and no thread's identity is 0). The seam lean's brackets claim nothing.
    void PlayIfOwned() {
        if (owner.load(std::memory_order_relaxed) != 0) {
            Play();
        }
    }

    void BeginDraw(u32 repack_mask, u32 repack_dirty);
    /// The repack layout of the draw BeginDraw opened (GPUMem opens it first, step 7).
    void SetDrawRepack(u32 repack_mask, u32 repack_dirty) noexcept {
        draw.repack_mask = repack_mask;
        draw.repack_dirty = repack_dirty;
    }
    void BeginDispatch();

    /// The cache whose ranges the read log asks about.
    void Attach(Cache& cache) noexcept;
    /// The read log's pending-mark question (GPU::ForEachPendingCpuWrite).
    void AttachGpu(Tegra::GPU& gpu) noexcept;

    /// The read log (Common::WriteRing::g_on): a gather of game writes applied
    /// (OnCacheInvalidations) or a FlushRegion (`which`: its CacheType bits), on this thread.
    enum class Gather : u8 { Apply, FlushRegion };
    void NoteGather(Gather kind, u32 which);

    /// The read log's classes of a difference (5.3).
    enum class DiffClass : u8 {
        Landing,       ///< an emulator write the ring names, a download landing
        EmulatorWrite, ///< an emulator write the ring names, anything else
        GameMarked,    ///< no emulator write; a game-side mark is pending (or the tracker holds it)
        MarkTaken,     ///< tracked read, no emulator write, no pending mark: a store in flight at
                       ///< the gather (nothing repairs it)
        Untracked,     ///< untracked read (a stream uniform, a value), no emulator write: no mark
                       ///< is expected, the next fill reads it again
        BackGather,    ///< bound range: a gather ran on this thread between the fill and the play
        Unclassified,
        AsyncOverRead, ///< a query completion GPUFencingThread wrote, on bytes the packet does not
                       ///< bind: it lands when the GPU finishes, not at a stream position (step 7)
        Count,
    };

    /// SEAM = 2: the draw's whole fill, before the texture work. Null when the draw is not held
    /// (another SEAM, texel buffers: LateFill, or the arena busy on another thread); Configure
    /// then fills where the calls are. A held packet is always played whole: RestAtEnd.
    /// FRITCHII_MEMSTAGE_THREAD = 1: run on GPUMem by hand-off.
    const DrawPacket* FillDraw(Cache& cache, const DrawFacts& facts, bool is_indexed);
    /// SEAM = 2: the dispatch's `before` section, before Configure. Null as for FillDraw.
    const DispatchPacket* FillDispatch(Cache& cache, const DispatchFacts& facts);
    /// The OoT trackers are on (RasterizerVulkan::TickFrame, at each frame end): FillDraw adds
    /// their record to the draw packet.
    void SetTrackerRecord(bool on) noexcept {
        tracker_record.store(on, std::memory_order_relaxed);
    }
    /// SEAM >= 2: draws get packets.
    [[nodiscard]] bool Packets() const noexcept {
        return seam_value.load(std::memory_order_relaxed) >= 2;
    }
    /// The arena's size when this thread owns it, else 0: inside a fill bracket, a growth means
    /// an entry was recorded (FRITCHII_MEMSTAGE_PASS's "a passed gather recorded a command").
    [[nodiscard]] u32 ArenaHere() const noexcept {
        return OwnedHere() ? arena.Size() : 0;
    }
    /// The record of slots 3, 4 and 6 from the cache as it is now (FillDraw; the verify).
    static void FillTrackerRecord(Cache& cache, TrackerRecord& record);
    /// One tracker slot's binding (slot 3 as the game bound it with FRITCHII_OOT_CAMERA_GUEST).
    static std::pair<DAddr, u32> TrackerBinding(Cache& cache, u32 index);
    /// The back: the bytes a ring packet's fill copied for tracker slot `slot` (0 to 2) of the
    /// draw it holds, or null (read game memory then).
    [[nodiscard]] const u8* TrackerBytes(const TrackerRecord& record, size_t slot) const noexcept;
    /// Step 7's verify: the cache marked [address, address + size) CPU-written for a game write
    /// (a served gather or invalidation, a game core's trapped store). Any thread; only while ring
    /// packets' reads are compared. A difference over one is "game write, marked": the tracker
    /// holds it and the next bind uploads.
    void NoteCpuMark(u64 address, u64 size) noexcept {
        if (ring_read_log.load(std::memory_order_relaxed)) [[unlikely]] {
            NoteCpuMarkSlow(address, size);
        }
    }
    /// The device memory the trackers' bytes are copied from (the rasterizer's).
    void AttachDeviceMemory(Tegra::MaxwellDeviceMemoryManager& memory) noexcept {
        device_memory = &memory;
    }

    /// On GPUMem inside a hand-off: runs `body` on the back with this thread's seam state (a
    /// sync, an entry that must run now). Anywhere else: runs it here.
    template <typename F>
    static void OnBack(F&& body) {
        using Body = std::remove_reference_t<F>;
        OnBackImpl([](void* argument) { (*static_cast<Body*>(argument))(); }, &body);
    }
    static void OnBackImpl(void (*function)(void*), void* argument);
    /// One follower at step 4, on E filling ring B: E has no player and no scheduler (R's), as
    /// GPUMem in a hand-off has none. Every entry that must run now and every sync goes to R
    /// (OnBack lends it, E parked), as THREAD = 2 sends GPUMem's to the back.
    [[nodiscard]] static bool OneFollowerFill() noexcept;
    /// OnBack's one-follower route: lends `function` to R with E's seam state installed there and
    /// the buffer cache's lock handed to R (the hand-off's served call, over one follower's lend).
    static void OnRenderImpl(void (*function)(void*), void* argument);
    /// FRITCHII_ESPLIT (fritchii_e_split_thread.h, ESplit::g_fill_state): E's seam state moved to
    /// E2 for its run of E's work and back, as OnRenderImpl moves it to R. `as_owner`: the
    /// calling thread's identity filled in; `keep_identity`: the caller keeps its own.
    static void ESplitCapture(void* out, bool as_owner);
    static void ESplitInstall(const void* in, bool keep_identity);

    /// The buffer cache's idle work, on its user's thread (hand-off at FRITCHII_MEMSTAGE_THREAD=1).
    template <typename F>
    void CacheIdle(F&& body) {
        using Body = std::remove_reference_t<F>;
        CacheIdleImpl([](void* argument) { (*static_cast<Body*>(argument))(); }, &body);
    }
    void CacheIdleImpl(void (*function)(void*), void* argument);

    [[nodiscard]] const DrawPacket* HeldDraw() const noexcept {
        return held == HeldKind::Draw ? (held_ring ? &played_draw : &draw) : nullptr;
    }

    // FRITCHII_MEMSTAGE_THREAD = 2 (vk_mem_stage.cpp, "Ring B").
    /// GPUMem: FillDraw for a draw the stage serves (after BeginDraw); the packet goes to ring B.
    /// `pre` runs inside the fill bracket first (the buffer half of the stage's ranges).
    /// `values`: the draw's raw handles the stage read at its position (SEAM 3), or null.
    /// `list_args`: a list draw (the cache's inputs carry its indirect parameters): the packet
    /// takes the buffer of its two words (ListArgs).
    bool FillRingDraw(Cache& cache, const DrawFacts& facts, bool is_indexed, u64 position, u64 seq,
                      u64 hash, void (*pre)(void*), void* pre_argument,
                      const std::vector<u32>* values, bool null_pointer_handles,
                      bool list_args = false);
    /// The back holds a draw packet from ring B.
    [[nodiscard]] bool HeldFromRing() const noexcept {
        return held == HeldKind::Draw && held_ring;
    }
    /// GPUMem: room for another packet.
    [[nodiscard]] bool RingRoom() const noexcept;
    /// FRITCHII_FILL_BRACKET_LEAN (fable_b 25.2 cut h), E, for its next FillRingDraw: the cut's
    /// value and the tracker record slots R can read on that draw (bit i: slot i).
    void SetRingLean(int mode, u8 tracker_slots) noexcept {
        ring_lean = mode;
        ring_lean_slots = tracker_slots;
    }
    /// GPUMem: its draw's fill is over, packet or not (BeginDraw opened its packet number).
    void EndStageFill();
    /// The back, both threads parked at the ring's start or end (`end`): GPUMem's fill side starts
    /// clean; at the end its pending entries (a fill that threw) are played here first.
    void ResetStageSide(bool end);
    /// Packets in ring B the back has not taken or dropped yet (step 7's window).
    [[nodiscard]] u64 RingPackets() const noexcept {
        return ring_packets_sent.load(std::memory_order_acquire) -
               ring_packets_done.load(std::memory_order_seq_cst);
    }
    /// GPUMem (the only pusher): the push index the next packet gets, counted from 1 (packets
    /// are taken or dropped in this order: FRITCHII_MEMSTAGE_MARKS_AHEAD keys its marks by it).
    [[nodiscard]] u64 NextRingPacketIndex() const noexcept {
        return ring_packets_sent.load(std::memory_order_relaxed) + 1;
    }
    /// The back: the packet of its draw `seq` (packets of earlier draws are played whole first),
    /// held as a filled draw is; null when there is none. `hash`: the back's compare b hash.
    const DrawPacket* TakeRingDraw(u64 seq, u64 back_position, u64 hash, bool verify);
    /// The back: plays whole the packets of entries before `position`; returns how many.
    u64 DropRingPackets(u64 position);
    /// The back: the barrier after a ring packet's uploads (Configure's any_buffer_uploaded).
    void PlayPostCopyBarrier();
    /// The back publishes a pipeline's facts under the stage's program key; GPUMem looks them up.
    /// Entries match on the key and the registers it hashes (a key collision is a miss).
    using ProgramId = VideoCommon::MemStageShadow::ProgramId;
    /// `code_epoch`: MemStageShadow::g_program_code_epoch read before the pipeline was looked
    /// up (the code the facts describe is at least that new).
    void PublishFacts(const ProgramId& id, const void* source, const DrawFacts& facts,
                      bool dynamic_input, u64 code_epoch);
    [[nodiscard]] bool LookupFacts(const ProgramId& id, StagedFacts& out) const;
    /// LookupFacts without its one-reader cache (any thread; the cache is E's or GPUMem's).
    [[nodiscard]] bool LookupFactsLocked(const ProgramId& id, StagedFacts& out) const;
    /// The facts table's generation (bumped under its lock at every write): while it holds, a
    /// lookup of the same registers gives what it gave (FRITCHII_MEMSTAGE_SETUP_KEEP).
    /// The back's held ring packet: the vertex and fragment program hashes of the facts its fill
    /// used (one follower's pre-record check against the pipeline about to be bound).
    [[nodiscard]] const std::array<u64, 2>& HeldHashes() const noexcept {
        return held_hashes;
    }
    [[nodiscard]] u64 FactsGeneration() const noexcept {
        // A code write (the program code epoch) changes what a lookup gives as a table write does.
        return facts_generation.load(std::memory_order_acquire) +
               VideoCommon::MemStageShadow::g_program_code_epoch.load(std::memory_order_acquire);
    }
    [[nodiscard]] const DispatchPacket* HeldDispatch() const noexcept {
        return held == HeldKind::Dispatch ? (held_ring ? &played_dispatch : &dispatch) : nullptr;
    }
    /// One follower: the held dispatch came from ring B (R plays a posted dispatch E filled).
    [[nodiscard]] bool HeldDispatchFromRing() const noexcept {
        return held == HeldKind::Dispatch && held_ring;
    }

    // One follower, posted dispatches (FRITCHII_ONEF_OP_DISPATCH; vk_mem_stage.cpp, "Ring B
    // dispatch packets").
    /// E: the dispatch's buffer side (FillDispatchHere's calls) filled by value into a ring B
    /// packet of its own kind, with the handle values E read and `view` (`view_bytes`, a multiple
    /// of 8: what R's play needs of E's compute engine). Returns the packet's number, the name R
    /// takes it by; 0 = nothing filled or pushed (the caller lends the dispatch).
    /// `indirect` (FRITCHII_ONEF_OP_DISPATCH bit 4): an indirect dispatch's arguments: after the
    /// `before` section E obtains their buffer as DispatchCompute does (its entries in a section
    /// of their own, played at the packet's rest, after Configure) and the packet's `indirect`
    /// names it.
    u64 FillRingDispatch(Cache& cache, const DispatchFacts& facts, u64 position,
                         const std::vector<u32>& values, bool null_pointer_handles,
                         const void* view, u32 view_bytes,
                         std::optional<GPUVAddr> indirect = std::nullopt);
    /// R, at the dispatch's op: takes ring B's packet `number` (13.2 R1: by name) and holds it as
    /// a filled dispatch (Configure plays it, RestAtEnd ends it); copies its view into `view`.
    /// Null when ring B's head is not that packet ("ring B pairing"): nothing held.
    const DispatchPacket* TakeRingDispatch(u64 number, void* view, u32 view_bytes);

    /// Plays the held packet up to the end of `span` (the old position of that section).
    void PlayHeld(Span span);
    /// Plays what is left of the held packet (descriptors left out) and ends it.
    void PlayRest();

    /// PlayRest when Configure ends: every return, true or false, and an exception.
    class RestAtEnd {
    public:
        explicit RestAtEnd(MemStage& stage_) noexcept : stage{stage_} {}
        ~RestAtEnd() {
            stage.PlayRest();
        }
        RestAtEnd(const RestAtEnd&) = delete;
        RestAtEnd& operator=(const RestAtEnd&) = delete;

    private:
        MemStage& stage;
    };

    /// The descriptor form the runtime's descriptor entries take: the held fill's pipeline's
    /// (it runs before Acquire), else the queue's.
    [[nodiscard]] bool DescriptorForm(bool queue_form) const noexcept {
        return fill_form < 0 ? queue_form : fill_form != 0;
    }

    enum class Value : u8 {
        Handle,
        Qmd,
        NullSlot,
        Shape,
        Count,
    };
    enum ValueFlag : u32 {
        HandleValuesFilled = 1U << 16,
        HandleNullPointers = 1U << 17,
        HandleValuesLooked = 1U << 18,
        HandleValuesRetired = 1U << 19,
        QmdByValue = 1U << 20,
    };
    enum VerifyBit : u32 {
        VerifyHandles = 1,
        VerifyQmd = 2,
        VerifyNullSlots = 4,
    };

    [[nodiscard]] bool ValuesByValue() const noexcept {
        return seam_level.load(std::memory_order_relaxed) >= 3;
    }
    [[nodiscard]] static u32 VerifyBits() noexcept;

    template <typename FillFn>
    void FillHandleValues(bool dispatch_side, bool null_pointer_handles, bool in_configure,
                          FillFn&& fill) {
        std::vector<u32>& values = dispatch_side ? dispatch_values : draw_values;
        values.clear();
        fill(values);
        CloseHandleValues(dispatch_side, null_pointer_handles, in_configure);
    }
    [[nodiscard]] const std::vector<u32>* HandleValues(bool dispatch_side,
                                                       bool& null_pointer_handles) const noexcept;
    void LookedAtHandleValues(bool dispatch_side, size_t taken);
    /// Step 7, a 5.4 check of the ring packet the back holds that differs: whether GPUMem wrote
    /// any of these GPU ranges since the packet's fill (its constant buffer words for later
    /// entries; the back's own read is at play, after them). Then the value is left out and
    /// counted apart ("left out as GPUMem's own later writes"), not as a mismatch. Only while
    /// GPUMem keeps its own writes (the window open, its verify on).
    [[nodiscard]] bool ValueLeftOut(Tegra::MemoryManager& gpu_memory,
                                    std::span<const std::pair<u64, u64>> ranges);
    void NoteValueCheck(Value kind, bool same, u64 address, u64 shader, u64 packet_value,
                        u64 own_value);

    const std::array<u32, 64>& Qmd(Tegra::MemoryManager& gpu_memory, GPUVAddr address);
    u64 ReadPointerSlot(Tegra::MemoryManager& gpu_memory, GPUVAddr address);

    /// Once a frame on the GPU thread: reads SEAM (EDEN_NX2_EXP123) and logs the stats line.
    void TickFrame();

    void Push(Cmd kind, u8 flags = 0);
    void PushCopy(Cmd kind, u8 flags, VkBuffer src, VkBuffer dst,
                  std::span<const VkBufferCopy> copies);
    void PushClear(VkBuffer dst, u64 offset, u64 size, u32 value);
    void PushQuadLut(VkBuffer src, VkDeviceSize src_offset, VkBuffer dst, VkDeviceSize size);
    void PushNullInit(VkBuffer buffer);
    void PushIndex(Cmd kind, u8 flags, VkBuffer buffer, VkDeviceSize offset, VkIndexType type);
    void PushVertex(u8 flags, u32 index, VkBuffer buffer, u32 offset, u32 size, u32 stride);
    void PushVertices(u8 flags, u32 first, std::span<const VkBuffer> buffers,
                      std::span<const VkDeviceSize> offsets, std::span<const VkDeviceSize> sizes,
                      std::span<const VkDeviceSize> strides);
    void PushXfb(u32 index, VkBuffer buffer, u32 offset, u32 size);
    void PushXfbs(std::span<const VkBuffer> buffers, std::span<const VkDeviceSize> offsets,
                  std::span<const VkDeviceSize> sizes);
    void PushConvert(Pass pass, const ComputePassRun& run);
    /// `texel`: a texel buffer's descriptor (counted apart in the stats line).
    void PushDescriptor(const DescriptorUpdateEntry& entry, bool texel = false);

private:
    friend class SeamClock;
    friend void SeamHook::OnUse(SeamHook::Use use, const std::source_location& location);
    friend void SeamHook::OnTickEnd(u64 tick);
    friend void VideoCommon::MemReadLog::NoteBytes(VideoCommon::MemReadLog::Kind,
                                                   Tegra::MaxwellDeviceMemoryManager&, u64, u64);
    friend void VideoCommon::MemReadLog::NoteStaged(VideoCommon::MemReadLog::Kind,
                                                    Tegra::MaxwellDeviceMemoryManager&, u64,
                                                    const u8*, u64);
    friend void VideoCommon::MemReadLog::NoteValue(VideoCommon::MemReadLog::Kind,
                                                   Tegra::MemoryManager*, u64, u64, u32);
    friend void VideoCommon::MemReadLog::NoteRange(VideoCommon::MemReadLog::Kind, u64, u64);
    friend void VideoCommon::MemReadLog::NoteGuestWrite(Tegra::MemoryManager&, u64, u64);

    enum class HeldKind : u8 { None, Draw, Dispatch };

    /// One read of game memory a fill made (spec 5.3).
    struct ReadRecord {
        enum class How : u8 { Bytes, Staged, Value, Range };
        VideoCommon::MemReadLog::Kind kind;
        How how;
        u32 size;
        u64 address;
        u64 value;        ///< Value: the value read; Bytes: offset of the copy; Range: fill state
        const void* data; ///< Staged: the staged bytes
        void* memory;     ///< The device or GPU memory read
    };
    struct ReadLog {
        std::vector<ReadRecord> records;
        std::vector<u8> bytes;
        u64 skipped{}; ///< Not logged: past the byte budget
        u64 budget{};  ///< Bytes to read again at the compare
        /// Device ranges the packet uses: bound ranges, uniforms, values (over-read test).
        std::vector<std::pair<u64, u64>> used;
        u64 ring_seq{};      ///< Common::WriteRing's sequence when the fill began
        u64 gathers{};       ///< This thread's gathers applied when the fill ended
        u64 flush_regions{}; ///< This thread's FlushRegion calls when the fill ended
        /// Step 7, a ring packet's log: GPUMem's own writes noted when the fill ended (the
        /// compare leaves out the ones after), the CPU-written marks noted then (CpuMarks), and
        /// the packet's entry.
        u64 stage_writes{};
        u64 cpu_marks{};
        u64 position{};
        void Clear() noexcept {
            records.clear();
            bytes.clear();
            skipped = 0;
            budget = 0;
            used.clear();
            ring_seq = 0;
            gathers = 0;
            flush_regions = 0;
            stage_writes = 0;
            cpu_marks = 0;
            position = 0;
        }
    };

    /// Emulator writes the ring named at a difference, by site.
    struct DiffSite {
        std::atomic<const char*> name{};
        std::atomic<bool> landing{};
        std::atomic<u64> count{};
    };
    static constexpr size_t DIFF_SITES = 32;

    inline static thread_local ReadLog* t_log = nullptr;

    /// Fill brackets open on this thread, and whether the outermost one opened lean (Fill).
    inline static thread_local u32 t_fill_depth = 0;
    inline static thread_local bool t_fill_lean = false;

    /// A device range a held fill read (any kind), for the texture write check.
    struct ReadRange {
        u64 begin;
        u64 end;
        VideoCommon::MemReadLog::Kind kind;
    };
    inline static thread_local std::vector<ReadRange>* t_reads = nullptr;
    static void NoteRead(VideoCommon::MemReadLog::Kind kind, u64 address, u64 size) {
        if (t_reads != nullptr && size != 0) {
            t_reads->push_back({address, address + size, kind});
        }
    }

    struct Part {
        const void* data;
        u32 bytes;
    };

    template <typename T>
    static Part PartOf(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        return Part{&value, static_cast<u32>(sizeof(T))};
    }
    template <typename T>
    static Part PartOf(std::span<const T> values) {
        static_assert(std::is_trivially_copyable_v<T>);
        return Part{values.data(), static_cast<u32>(values.size_bytes())};
    }

    struct Counts {
        std::array<u64, static_cast<size_t>(Cmd::Count)> entries{};
        std::array<u64, static_cast<size_t>(Cmd::Count)> outside{};
        std::array<u64, static_cast<size_t>(Site::Count)> site_fills{};
        std::array<u64, static_cast<size_t>(Site::Count)> site_entries{};
        std::array<u64, static_cast<size_t>(Sync::Count)> syncs{};
        u64 draws{};
        u64 dispatches{};
        u64 deferred{};
        u64 plays{};
        u64 around{};
        u64 late{};
        u64 unclocked{};
        u64 texel{};
        u64 held_draws{};
        u64 late_draws{};
        u64 held_dispatches{};
        u64 late_dispatches{};
        u64 unheld{};
        u64 held_flushed{};
        u64 seam_clamped{};
        u64 texture_writes{};
        u64 texture_overlaps{};
        u64 rest_plays{};
        u64 rest_descriptors{};
        u64 form_mismatch{};
        u64 read_packets{};
        u64 read_records{};
        u64 read_skipped{};
        u64 fill_ns{};
        u64 play_ns{};
        std::array<u64, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)> read_diffs{};
        std::array<u64, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)>
            read_kind_records{};
        std::array<u64, static_cast<size_t>(DiffClass::Count)> diff_classes{};
        u64 diff_own{};
        u64 diff_over{};
        u64 diff_stage_own{};
        u64 ring_reads{};
        u64 ring_reads_marked{};
        u64 ring_wrapped{};
        u64 gathers_held{};
        std::array<u64, 3> flush_regions_held{};
        std::array<u64, DIFF_SITES> diff_sites{};
        u64 diff_sites_other{};
    };

    void PushParts(Cmd kind, u8 flags, u32 count, std::initializer_list<Part> parts);
    /// An entry that runs as pushed (not Deferring): its typed play, with PushParts's checks.
    template <typename PlayFn>
    void RunNow(Cmd kind, PlayFn&& play);
    void CountPush(Cmd kind) noexcept;
    [[nodiscard]] bool Deferring() const noexcept;
    [[nodiscard]] bool OwnedHere() const noexcept;
    void SetPending();
    void PlayPending(bool late);
    /// One follower (fable_b 13.1): E posts its pending entries as one ring B op packet and
    /// pushes BufferOp{its number}; false = not posted (the caller lends, as before).
    bool PostPendingOp();
    /// R, at a BufferOp: plays the op packet `number` (13.2 R1: by name).
    void PlayRingOp(u64 number);
    void CloseSection(Site site, u32 index, u32 begin);
    void EnterSync(Sync kind);
    void Note(SeamHook::Use use, const std::source_location& location, bool around);
    void LogStats();
    void Snapshot(Counts& out) const;
    void CloseHandleValues(bool dispatch_side, bool null_pointer_handles, bool in_configure);
    void RetireDrawValues();
    void RetireDispatchValues();
    void OpenDispatchByValue();
    void LogValueStats(double seconds, bool first);

    const DrawPacket* FillDrawHere(Cache& cache, const DrawFacts& facts, bool is_indexed);
    const DispatchPacket* FillDispatchHere(Cache& cache, const DispatchFacts& facts);
    template <typename Body>
    void HandOff(Cache& cache, u8 kind, Body& body);
    [[nodiscard]] bool HandsOff() const noexcept;
    void TickThread(int seam);

    bool ClaimHeld();
    void UpdateUnplayed();
    void TakeHeld(HeldKind kind);
    struct HeldFillEnd;
    void EndHeldFill();
    void CheckHeld();
    void PlayHeldTo(u32 end, bool rest);
    bool StartReadLog(ReadLog& log, u64 number);
    void StopReadLog(ReadLog& log, bool held_packet);
    void NoteHeldFlush(SeamHook::Use use, const std::source_location& location);
    void CompareReads(ReadLog& log, const char* what, u64 number, u64 hash_a, u64 hash_b);
    void ClassifyDiff(const ReadLog& log, const ReadRecord& record, u64 first, u64 last,
                      const char* what, u64 number, u64 hash_a, u64 hash_b,
                      bool tracker_holds = false);
    void CountDiffSite(const char* name, bool landing);
    std::string GatherFrames() const;

    MemPlayer& player;
    Arena arena;
    std::atomic<uintptr_t> owner{0};
    std::atomic<bool> seam_on{false};
    std::atomic<int> seam_value{0};
    /// The seam lean (TickFrame): SEAM 0, no packet ticks, no stats line. Brackets keep only their
    /// depth, and entries pushed at once skip the stats counts.
    std::atomic<bool> lean{false};
    std::atomic<bool> tracker_record{false};
    /// FRITCHII_MEMSTAGE_READ_LOG: every Nth packet's reads are compared, 0 off
    int read_log_every{};
    Cache* cache{};
    /// FRITCHII_MEMSTAGE_THREAD = 1: the cache's lock while a hand-off lends it to GPUMem.
    Common::SpinRecursiveMutex* lent_mutex{};
    std::thread::id lent_back{};
    std::thread::id lent_stage{};
    /// The back's identity while a hand-off runs (an owner it takes back with pending entries).
    uintptr_t handing_back{};
    /// SEAM 2/3 asked for without packet ticks: run as SEAM 1 (logged once).
    bool clamp_logged{};

    HeldKind held{HeldKind::None};
    /// Atomic (step 7): the arena owner's, read by the other thread's logs and checks.
    std::atomic<bool> filling_held{};
    std::atomic<HeldKind> filling_kind{HeldKind::None};
    /// The held packet saw a flush between its fill and its PlayRest (counted once). Atomic: the
    /// thread that holds a packet sets it, another thread's sync can reach NoteHeldFlush.
    std::atomic<bool> held_flush_seen{};
    s8 fill_form{-1};
    Arena held_arena;
    u32 held_cursor{};
    bool held_form{};
    std::array<u64, 2> held_hashes{};
    std::vector<ReadRange> held_reads;
    ReadLog op_log;
    ReadLog held_log;
    bool held_log_pending{};
    /// FRITCHII_MEMSTAGE_THREAD=1: held_log was stopped on GPUMem; HandOff stamps it with the
    /// back's gather counters (they are per thread, and the back holds the packet).
    bool held_log_on_stage{};
    u64 op_fills{};
    std::vector<u8> compare_scratch;
    Tegra::GPU* gpu{};
    /// This stage turned Common::WriteRing on (FRITCHII_MEMSTAGE_READ_LOG).
    bool ring_owner{};
    std::array<std::atomic<u32>, static_cast<size_t>(DiffClass::Count)> class_logged{};
    /// The first gather on this thread after the held fill ended: where it came from.
    std::array<void*, 16> gather_frames{};
    u32 gather_frame_count{};
    std::mutex diff_sites_mutex;
    std::array<DiffSite, DIFF_SITES> diff_site_table{};
    bool claim_deferring{};
    bool pending{};
    bool pending_engine{}; ///< `pending` counted in SeamHook::g_watch_engine (E's fill)
    /// Uncounts `pending` from the word SetPending counted it in.
    void ClearPendingWatch() noexcept {
        if (std::exchange(pending_engine, false)) {
            SeamHook::g_watch_engine.value.store(0, std::memory_order_relaxed);
        } else {
            SeamHook::g_watch.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    /// The site of the bracket the first pending entry was recorded in (PostPendingOp).
    Site pending_site{Site::Count};
    u32 play_cursor{};
    u32 last_entry{~0U};

    u64 packet_number{};
    std::unique_ptr<SeamTicks> ticks;
    /// Atomic (step 7): an operation's fill can open on any thread (a guest core's invalidation).
    std::atomic<u64> op_number{};
    /// Packet ticks: the first number of the pending entries and of the held packet (SEAM >= 2).
    u64 pending_from{};
    u64 held_from{};
    DrawPacket draw{};
    DispatchPacket dispatch{};
    OpPacket op{};

    struct Counters {
        std::array<std::atomic<u64>, static_cast<size_t>(Cmd::Count)> entries{};
        std::array<std::atomic<u64>, static_cast<size_t>(Cmd::Count)> outside{};
        std::array<std::atomic<u64>, static_cast<size_t>(Site::Count)> site_fills{};
        std::array<std::atomic<u64>, static_cast<size_t>(Site::Count)> site_entries{};
        std::array<std::atomic<u64>, static_cast<size_t>(Sync::Count)> syncs{};
        std::atomic<u64> draws{};
        std::atomic<u64> dispatches{};
        std::atomic<u64> deferred{};
        std::atomic<u64> plays{};
        std::atomic<u64> around{};
        std::atomic<u64> late{};
        std::atomic<u64> unclocked{};
        std::atomic<u64> texel{};
        std::atomic<u64> held_draws{};
        std::atomic<u64> late_draws{};
        std::atomic<u64> held_dispatches{};
        std::atomic<u64> late_dispatches{};
        std::atomic<u64> unheld{};
        std::atomic<u64> held_flushed{};
        std::atomic<u64> seam_clamped{};
        std::atomic<u64> texture_writes{};
        std::atomic<u64> texture_overlaps{};
        std::atomic<u64> rest_plays{};
        std::atomic<u64> rest_descriptors{};
        std::atomic<u64> form_mismatch{};
        std::atomic<u64> read_packets{};
        std::atomic<u64> read_records{};
        std::atomic<u64> read_skipped{};
        std::atomic<u64> fill_ns{};
        std::atomic<u64> play_ns{};
        std::array<std::atomic<u64>, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)>
            read_diffs{};
        std::array<std::atomic<u64>, static_cast<size_t>(VideoCommon::MemReadLog::Kind::Count)>
            read_kind_records{};
        std::array<std::atomic<u64>, static_cast<size_t>(DiffClass::Count)> diff_classes{};
        std::atomic<u64> diff_own{};
        std::atomic<u64> diff_over{};
        std::atomic<u64> diff_stage_own{}; ///< Step 7: changed only by GPUMem's later writes
        std::atomic<u64> ring_reads{};     ///< Step 7: ring packets whose reads were compared
        std::atomic<u64> ring_reads_marked{}; ///< ... with a CPU-written mark since the fill
        std::atomic<u64> ring_wrapped{};
        std::atomic<u64> gathers_held{};
        std::array<std::atomic<u64>, 3> flush_regions_held{};
        std::atomic<u64> diff_sites_other{};
    };
    Counters counters;
    Counts reported;
    u64 last_log_ns{};
    std::array<u64, 2> reported_vertex_runs{};
    bool thread_clamp_logged{};
    bool ring_clamp_logged{};
    bool ring_seam_logged{};

    std::atomic<int> seam_level{0};
    bool dispatch_open{};

    // Ring B (THREAD = 2): the stage's draw packets, one writer (GPUMem), one reader (the back).
    struct RingHeader {
        u32 bytes;
        u8 kind;
        u8 flags; ///< 1: held form, 2: read log, 4: its fill marked writes (RingMarkedWrites)
        u16 part;
        u64 position; ///< Ring A position of the entry the packet belongs to
        u64 number;
        u64 seq;      ///< The draw's sequence number since the ring started
        u64 hash;     ///< The stage's compare b hash (FRITCHII_MEMSTAGE_RING_VERIFY), else 0
        u32 held_cursor;
        u32 entry_bytes;
        std::array<u64, 2> hashes;
        u32 value_count;  ///< Handle values after the entries
        u32 value_flags;  ///< 1: filled, 2: EXP21's null pointer handles
        u32 tracker_bytes; ///< Step 7: the trackers' bytes after the values
        u32 reserved;
        u64 stage_writes;  ///< Step 7: GPUMem's own writes noted when the fill ended
    };
    static constexpr u64 RingBytes = u64{4} << 20;
    /// RingHeader::kind of an op packet (fable_b 13.1): 0 pad, 1 draw.
    static constexpr u8 RingKindOp = 2;
    /// RingHeader::kind of a posted dispatch (FillRingDispatch): RingHeader, DispatchPacket, the
    /// entries, the handle values, then the view's bytes (`tracker_bytes` holds their size).
    static constexpr u8 RingKindDispatch = 3;
    /// R: the dispatch packet it holds from ring B, and its handle values (E's `dispatch` and
    /// `dispatch_values` are E's while it fills the next one).
    DispatchPacket played_dispatch{};
    std::vector<u32> played_dispatch_values;
    void PushRingDispatch(const std::vector<u32>& values, bool null_pointer_handles,
                          const void* view, u32 view_bytes, u64 position);
    static_assert(sizeof(RingHeader) % 8 == 0, "op entries are played in place (8-aligned)");
    std::unique_ptr<u8[]> ring_b;
    // Each thread's control fields on their own cache line (step 7: with the window open both
    // write theirs for every packet, and one shared line bounced between the cores).
    alignas(64) u64 ring_b_write{};       ///< GPUMem
    std::atomic<u64> ring_b_sent{};       ///< Bytes GPUMem committed
    std::atomic<u64> ring_packets_sent{}; ///< GPUMem: packets pushed
    alignas(64) u64 ring_b_read{};        ///< The back
    std::atomic<u64> ring_b_freed{};      ///< Bytes the back is done with
    std::atomic<u64> ring_packets_done{}; ///< The back: packets taken or dropped
    /// The fill in progress goes to ring B (FillRingDraw).
    alignas(64) bool ring_fill{};
    bool ring_pushed{};
    /// FRITCHII_FILL_BRACKET_LEAN for the ring fill in progress (E's FillRing sets it, the fill's
    /// end clears it; 0 for every other fill), and that draw's tracker slots R can read. On the
    /// fill side's line, as the fields below.
    int ring_lean{};
    u8 ring_lean_slots{0x7};
    /// The pusher's last read of ring_b_freed in a lean push (it only grows: room against it is
    /// room now), and whether this push's lean parts are timed (1 in 64).
    u64 ring_lean_freed{};
    bool ring_lean_timed{};
    u64 ring_fill_position{};
    u64 ring_fill_seq{};
    u64 ring_fill_hash{};
    void (*ring_pre)(void*){};
    void* ring_pre_argument{};
    const std::vector<u32>* ring_values{};
    bool ring_null_pointers{};
    bool ring_list_args{};
    /// The held draw came from ring B: its packet is played_draw.
    bool held_ring{};
    DrawPacket played_draw{};
    /// FRITCHII_MEMSTAGE_MARKS_AHEAD: the held ring packet's push index, and whether its fill
    /// marked writes ahead of the back (RingMarkedWrites: promoted when its play ends).
    u64 held_packet_index{};
    bool held_marks{};
    static constexpr u8 RingMarkedWrites = 4; ///< RingHeader::flags
    /// GPUMem's side of a ring fill (THREAD = 2; the back's held_* fields are the back's): the
    /// open packet number (its unplayed mark), the program hashes and the descriptor form.
    u64 stage_fill_from{};
    /// Step 7, stamps per thread: GPUMem's open operation's number (its mark while it runs).
    u64 stage_op_from{};
    std::array<u64, 2> fill_hashes{};
    bool fill_held_form{};
    /// Step 7: the trackers' bytes, GPUMem's for the fill, the back's for the packet it holds.
    std::vector<u8> fill_tracker_data;
    std::vector<u8> played_tracker_data;
    /// The held ring packet's ring header stage_writes (ValueLeftOut).
    u64 played_stage_writes{};
    Tegra::MaxwellDeviceMemoryManager* device_memory{};
    void CopyTrackerBytes(TrackerRecord& record, u64 vertex_program);
    /// FRITCHII_FILL_BRACKET_LEAN: the trackers' record of a ring fill with the cut on.
    void FillTrackerLean(Cache& cache);
    /// The record of the slots in `slots` (bit i: slot i) as FillTrackerRecord makes it; the
    /// others unbound.
    static void FillTrackerSlots(Cache& cache, TrackerRecord& record, u8 slots);
    /// FRITCHII_FILL_BRACKET_LEAN: PushRingDraw's room check and its stats count.
    void PushRoomLean(u64 bytes);
    void PushStatsLean(u64 bytes, u64 sent);
    /// Step 7's verify (spec 6.4, the design's 2.5 read compare): a ring packet's read log, filled
    /// by GPUMem (stage_log while it fills), handed over in packet order, compared by the back at
    /// the take. Logs are pooled.
    std::mutex ring_logs_mutex;
    std::deque<std::pair<u64, std::unique_ptr<ReadLog>>> ring_logs;
    std::vector<std::unique_ptr<ReadLog>> ring_log_pool;
    std::unique_ptr<ReadLog> stage_log;
    /// Ring packets' reads are compared (the window open, FRITCHII_MEMSTAGE_READ_LOG).
    std::atomic<bool> ring_read_log{false};
    /// The CPU-written marks (NoteCpuMark): several writers, a sequence word per slot.
    struct CpuMark {
        std::atomic<u64> seq{};
        std::atomic<u64> begin{};
        std::atomic<u64> end{};
    };
    static constexpr u64 CpuMarkSlots = 8192;
    std::unique_ptr<CpuMark[]> cpu_marks;
    std::atomic<u64> cpu_mark_count{0};
    u64 reported_quiet{};
    u64 reported_marks{};
    void NoteCpuMarkSlow(u64 address, u64 size) noexcept;
    /// Whether a mark noted since `since` covers [begin, end); `complete` false when the ring
    /// wrapped since.
    bool CpuMarkedSince(u64 since, u64 begin, u64 end, bool& complete) const;
    bool StartRingReadLog(u64 number);
    void StopRingReadLog();
    std::unique_ptr<ReadLog> TakeRingLog(u64 number);
    void RecycleRingLog(std::unique_ptr<ReadLog> log);
    void CompareRingReads(ReadLog& log, u64 number, u64 hash_a, u64 hash_b);
    void PushRingDraw();
    const RingHeader* PeekRing();
    void TakeRingHeader(const RingHeader& header);
    void DiscardRingHeader(const RingHeader& header);
    [[nodiscard]] u64 RingOldestNumber();

    // The facts table (spec 4.4): open addressing under a lock (written rarely, by the back).
    using ProgramWords = std::array<u64, VideoCommon::MemStageShadow::ProgramWordCount>;
    /// Slots match on the code identity (ProgramId's space and code), the key is its hash.
    struct FactsSlot {
        u64 key{};
        const void* source{};
        u64 space{};
        std::array<u64, 6> code{};
        StagedFacts facts{};
        u64 epoch{}; ///< the program code epoch the facts are good for (a lookup's minimum)
    };
    static constexpr size_t FactsSlots = 4096;
    std::unique_ptr<FactsSlot[]> facts_table;
    mutable std::mutex facts_mutex;
    /// Step 7: with the window open both threads use the table at every draw, so each keeps a
    /// cache of its own and takes the lock only on a miss. Bumped (under the lock) at every write.
    std::atomic<u64> facts_generation{0};
    /// The back's: what it published, one slot per key hash (no lock while it holds).
    struct PublishedMemo {
        u64 key{};
        const void* source{};
        u64 space{};
        std::array<u64, 6> code{};
        u64 epoch{};
    };
    static constexpr size_t MemoSlots = 1024;
    std::unique_ptr<PublishedMemo[]> published_memo;
    /// GPUMem's: the entries it looked up, valid while the generation it saw holds.
    struct StageFactsSlot {
        u64 key{};
        u64 space{};
        std::array<u64, 6> code{};
        StagedFacts facts{};
        u64 epoch{};
    };
    static constexpr size_t StageFactsSlots = 512;
    mutable std::unique_ptr<StageFactsSlot[]> stage_facts_cache;
    mutable u64 stage_facts_generation{~u64{0}};
    static size_t FactsHash(u64 key) noexcept {
        return static_cast<size_t>((key * 0x9e3779b97f4a7c15ULL) >> 52);
    }
    std::vector<u32> draw_values;
    std::vector<u32> dispatch_values;

    struct ValueCounts {
        std::array<u64, 2> fills{};
        std::array<u64, 2> values{};
        std::array<u64, 2> in_configure{};
        std::array<u64, 2> skipped{};
        std::array<u64, 2> skipped_values{};
        u64 qmds{};
        u64 qmds_unused{};
        u64 null_slots{};
        u64 null_slots_unkept{};
        std::array<u64, static_cast<size_t>(Value::Count)> checks{};
        std::array<u64, static_cast<size_t>(Value::Count)> mismatches{};
        u64 left_out{};
    };
    struct ValueCounters {
        std::array<std::atomic<u64>, 2> fills{};
        std::array<std::atomic<u64>, 2> values{};
        std::array<std::atomic<u64>, 2> in_configure{};
        std::array<std::atomic<u64>, 2> skipped{};
        std::array<std::atomic<u64>, 2> skipped_values{};
        std::atomic<u64> qmds{};
        std::atomic<u64> qmds_unused{};
        std::atomic<u64> null_slots{};
        std::atomic<u64> null_slots_unkept{};
        std::array<std::atomic<u64>, static_cast<size_t>(Value::Count)> checks{};
        std::array<std::atomic<u64>, static_cast<size_t>(Value::Count)> mismatches{};
        std::atomic<u64> left_out{}; ///< Step 7: handle values GPUMem wrote since the fill
    };
    ValueCounters value_counters;
    ValueCounts reported_values;
    void SnapshotValues(ValueCounts& out) const;
};

/// Turns the memory stage's entries into scheduler and descriptor queue calls: the only copy of
/// that Vulkan code. Owns what the GPU thread recorded: the remembered bindings and the list
/// draw facility.
class MemPlayer {
public:
    struct ListDrawArgs {
        VkBuffer buffer{};
        VkDeviceSize offset{};
        bool mesh{}; ///< A task/mesh draw command (group counts)
    };

    explicit MemPlayer(const Device& device, MemoryAllocator& memory_allocator,
                       Scheduler& scheduler, GuestDescriptorQueue& guest_descriptor_queue,
                       ComputePassDescriptorQueue& compute_pass_descriptor_queue);
    ~MemPlayer();

    MemPlayer(const MemPlayer&) = delete;
    MemPlayer& operator=(const MemPlayer&) = delete;

    void SetPasses(Uint8Pass* uint8, QuadIndexedPass* quad_indexed,
                   VertexRepackPass* vertex_repack) noexcept;

    void Exec(const MemStage::Entry& entry);

    /// The typed plays Exec decodes these kinds to; an entry run as pushed calls them directly.
    void PlayCopy(MemStage::Cmd kind, u8 flags, VkBuffer src_buffer, VkBuffer dst_buffer,
                  std::span<const VkBufferCopy> copies);
    void PlayIndex(MemStage::Cmd kind, u8 flags, VkBuffer vk_buffer, VkDeviceSize vk_offset,
                   VkIndexType vk_index_type);
    void PlayVertex(u8 flags, u32 index, VkBuffer buffer, u32 offset, u32 size, u32 stride);
    void PlayVertices(u8 flags, u32 first, u32 binding_count, const VkBuffer* handles,
                      const VkDeviceSize* offsets, const VkDeviceSize* sizes,
                      const VkDeviceSize* strides);

    void AddDescriptors(const DescriptorUpdateEntry* entries, size_t count);

    /// Before a draw is recorded (R, or the back): in a command buffer the remembered vertex
    /// bindings were not bound in yet (VertexEpoch moved), they are bound again. The bindings
    /// the fill leaves out as clean (their flags are E's under one follower, and R's new command
    /// buffers do not raise them) are then still bound: the AMD device loss at the Market load
    /// read the old command buffer's vertex table. Once per command buffer; nothing otherwise.
    void EnsureVertexBindings();

    /// Index binds played (recorded or found still bound); a draw compares it around its
    /// Configure to know whether the last one is its own.
    [[nodiscard]] u64 IndexPlays() const noexcept {
        return index_plays;
    }
    /// FRITCHII_SPLIT_BIND_FIX, a played or served draw after a submit inside its Configure: the
    /// index buffer its play bound is bound again when that was in an earlier command buffer
    /// (VertexEpoch moved since). Only right after a play of this draw (IndexPlays moved).
    void EnsureIndexBinding();

    /// The remembered index buffer is forgotten and the vertex ones are no longer trusted as
    /// bound (SEAM changed): each is recorded again at its next bind, and bound again in a new
    /// command buffer (EnsureVertexBindings).
    void ForgetBindings() noexcept;

    /// The player is the back's: counts a use on GPUMem (vk_mem_stage.cpp).
    static void CheckBack(const char* where) noexcept;

    /// `keep_id`: the id the copy of the words is kept as (KeepListWords), 0 for none.
    void RequestListDrawArgs(u64 keep_id = 0) noexcept {
        CheckBack("RequestListDrawArgs");
        list_draw_requested = true;
        list_draw_args = {};
        list_draw_keep_id = keep_id;
    }
    [[nodiscard]] bool ListDrawArgsRequested() const noexcept {
        CheckBack("ListDrawArgsRequested");
        return list_draw_requested;
    }
    /// Records the copy of the words at `offset` of `source`, outside the render pass, into a
    /// vertex draw command or (`mesh`) a task/mesh one.
    void RecordListDrawArgs(VkBuffer source, VkDeviceSize offset, bool mesh);
    /// The arguments recorded since the request (no buffer if none were), ending the request.
    [[nodiscard]] ListDrawArgs TakeListDrawArgs() noexcept {
        CheckBack("TakeListDrawArgs");
        list_draw_requested = false;
        list_draw_keep_id = 0;
        return std::exchange(list_draw_args, {});
    }
    /// Records, outside the render pass, a copy of `count` (up to 3) words at `offset` of
    /// `source` as `keep_id`.
    void KeepListWords(u64 keep_id, VkBuffer source, VkDeviceSize offset, u32 count);
    /// Keeps words the host read itself as `keep_id`.
    void KeepListWordsRead(u64 keep_id, std::span<const u32> words);
    /// The words kept as `keep_id`, waiting for the host GPU to copy them if it must. False if
    /// none are (a later copy took the slot).
    [[nodiscard]] bool ReadKeptListWords(u64 keep_id, std::span<u32> words);

private:
    friend class MemStage;

    /// Every play's start: the player check and the flag's render pass exit.
    void BeginPlay(u8 flags);
    void EnsureListKeepBuffer();

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    ComputePassDescriptorQueue& compute_pass_descriptor_queue;
    MemStage* stage{};

    Uint8Pass* uint8_pass{};
    QuadIndexedPass* quad_indexed_pass{};
    VertexRepackPass* vertex_repack_pass{};

    VkBuffer bound_index_buffer = VK_NULL_HANDLE;
    VkDeviceSize bound_index_offset = 0;
    VkIndexType bound_index_type = VK_INDEX_TYPE_UINT32;
    u64 bound_index_epoch = 0;
    bool bound_index_valid = false;
    /// The last index bind played (EnsureIndexBinding), whatever the rebind skip does.
    VkBuffer last_index_buffer = VK_NULL_HANDLE;
    VkDeviceSize last_index_offset = 0;
    VkIndexType last_index_type = VK_INDEX_TYPE_UINT32;
    u64 last_index_vertex_epoch = 0; ///< VertexEpoch of the command buffer it is bound in
    bool last_index_valid = false;
    u64 index_plays = 0;

    /// A vertex binding the player remembers. epoch: the VertexEpoch it was last bound at;
    /// ~0 = forgotten (nothing to bind again: never bound, left without a buffer, or a repack
    /// that failed); StaleEpoch = remembered but not known to be bound in this command buffer
    /// (the next PlayVertices records it, EnsureVertexBindings binds it in a new one).
    static constexpr u64 StaleEpoch = ~u64{0} - 1;
    struct BoundVertexBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        VkDeviceSize stride = 0;
        u64 epoch = ~u64{0};
        bool dynamic = false; ///< bound with BindVertexBuffers2EXT (its size and stride)
    };
    u64 ensured_vertex_epoch = ~u64{0};
    static constexpr size_t NUM_BOUND_VERTEX_BUFFERS = 32;
    std::array<BoundVertexBuffer, NUM_BOUND_VERTEX_BUFFERS> bound_vertex_buffers{};
    /// BindVertices entries recorded, and dropped as still bound (FRITCHII_MEMSTAGE_FULL_BINDS).
    std::atomic<u64> vertex_runs_recorded{};
    std::atomic<u64> vertex_runs_rebound{}; ///< EnsureVertexBindings: bindings bound again
    std::atomic<u64> vertex_forgotten{};    ///< single-slot forgets (ForgetOnly) played
    std::atomic<u64> vertex_runs_dropped{};

    static constexpr u32 LIST_DRAW_SLOTS = 1024;
    static constexpr u32 LIST_DRAW_SLOT_BYTES = 32;
    vk::Buffer list_draw_buffer;
    /// FRITCHII_LIST_ARGS_LOCAL: the arguments stay in device memory; this host copy of their
    /// fixed words fills them once, with the first list draw.
    vk::Buffer list_draw_template;
    bool list_draw_fill_pending{};
    bool list_draw_args_local{};
    u32 list_draw_cursor{};
    bool list_draw_requested{};
    ListDrawArgs list_draw_args{};
    u64 list_draw_keep_id{};

    static constexpr u32 LIST_KEEP_SLOTS = 1024;
    static constexpr u32 LIST_KEEP_SLOT_BYTES = 16;
    struct ListKeep {
        u64 keep_id{};
        u64 tick{}; ///< The submission that copies them, if on_gpu
        bool on_gpu{};
        u32 count{};
        std::array<u32, 3> words{}; ///< If not on_gpu
    };
    vk::Buffer list_keep_buffer;
    std::array<ListKeep, LIST_KEEP_SLOTS> list_keeps{};
};

} // namespace Vulkan
