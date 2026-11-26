// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <span>
#include <type_traits>
#include <vector>

#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/evicted_pointer_slots.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_library.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/texture_cache/draw_template.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

struct GraphicsPipelineCacheKey {
    std::array<u64, 6> unique_hashes;
    FixedPipelineState state;

    size_t Hash() const noexcept;

    bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;

    bool operator!=(const GraphicsPipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }

    size_t Size() const noexcept {
        return sizeof(unique_hashes) + state.Size();
    }

    /// FRITCHII_PIPELINE_KEY_HASH: a cheap hash of exactly the bytes operator== compares (the
    /// first Size() bytes; Size() is read from raw1, which those bytes hold, so equal keys hash
    /// the same). Two rotate-xor lanes over 8-byte words, mixed at the end: keys of one size that
    /// differ in one word never hash the same. A prefilter only: the full compare decides.
    u64 Fingerprint() const noexcept {
        const size_t size = Size();
        const u8* const bytes = reinterpret_cast<const u8*>(this);
        u64 lane_a = 0;
        u64 lane_b = 0;
        size_t at = 0;
        for (; at + 16 <= size; at += 16) {
            u64 word_a;
            u64 word_b;
            std::memcpy(&word_a, bytes + at, sizeof(word_a));
            std::memcpy(&word_b, bytes + at + 8, sizeof(word_b));
            lane_a = std::rotl(lane_a, 5) ^ word_a;
            lane_b = std::rotl(lane_b, 7) ^ word_b;
        }
        if (at + 8 <= size) {
            u64 word;
            std::memcpy(&word, bytes + at, sizeof(word));
            lane_a = std::rotl(lane_a, 5) ^ word;
            at += 8;
        }
        if (at < size) {
            u64 word = 0;
            std::memcpy(&word, bytes + at, size - at);
            lane_b = std::rotl(lane_b, 7) ^ word;
        }
        return (lane_a * 0x9e3779b97f4a7c15ULL) ^ (lane_b * 0xc2b2ae3d27d4eb4fULL) ^ size;
    }
};
static_assert(std::has_unique_object_representations_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<GraphicsPipelineCacheKey>);

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::GraphicsPipelineCacheKey> {
    size_t operator()(const Vulkan::GraphicsPipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class Device;
class PipelineStatistics;
class RenderPassCache;
class RescalingPushConstant;
class RenderAreaPushConstant;
class Scheduler;

/// Graphics pipeline libraries (EDEN_NX2_GPL): non-null `library_cache` builds the pipeline
/// from shared per-stage libraries; `libraries_only` builds only the libraries (warm-up).
struct GraphicsPipelineLibraryParams {
    GraphicsPipelineLibraryCache* library_cache{};
    std::array<u128, Tegra::Engines::Maxwell3D::Regs::MaxShaderStage> spirv_hashes{};
    bool libraries_only{};
    /// Not a library setting (carried here to keep the constructor as it is): a prepared
    /// pipeline built from kept SPIR-V, timed for TakeKeptBuildStats. Runtime pipelines whose
    /// SPIR-V a worker emits are new to the driver anyway, so they are not counted.
    bool prepared{};
};

/// Mesh emulation on a device without VK_EXT_mesh_shader: the mesh program (stage 0) is built as
/// a compute shader that writes its output to a storage buffer, and this vertex shader draws it
/// (see Shader::Backend::SPIRV::EmitMeshEmulationVertex).
struct MeshEmulationSetup {
    std::vector<u32> vertex_spirv;
    u32 output_binding{};
    u32 block_words{};    ///< output words per mesh workgroup
    u32 max_primitives{}; ///< primitives per mesh workgroup
    /// Vertices per primitive: 3 triangles, 2 lines, 1 points (the draw's topology).
    u32 primitive_vertices{3};
    /// Generic location carrying the mesh program's PrimitiveId to the pixel program, ~0 if none
    u32 primitive_id_location{~0U};
    // With a task program (stage 0, the mesh program at mesh_stage): its payload buffer and the
    // distribution pass's offsets and arguments (Shader::RuntimeInfo), and that pass's shader.
    bool has_task{};
    u32 mesh_stage{};
    u32 payload_stride{};
    u32 payload_binding{};
    u32 offsets_binding{};
    u32 args_binding{};
    std::vector<u32> distribution_spirv;
    // Output layout, for the readback diagnostics.
    u32 max_vertices{};
    u32 vertex_lines{};
    std::array<u32, 4> position_lines{~0U, ~0U, ~0U, ~0U}; ///< line of PositionX..W
};

/// The sampled textures and image descriptors of a draw, kept across frames (ConfigureImpl,
/// EXP87, ported).
using DrawTemplate =
    VideoCommon::DrawTemplate<DescriptorUpdateEntry,
                              Tegra::Engines::Maxwell3D::Regs::MaxShaderStage,
                              Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS>;
using DrawTemplates = VideoCommon::DrawTemplateTable<DrawTemplate>;

/// A pipeline's previous draw (delta draws, Fable's port plan step D2; ConfigureImpl).
using KeptDraw =
    VideoCommon::KeptDraw<DescriptorUpdateEntry, VideoCommon::ImageViewInOut,
                          Tegra::Engines::Maxwell3D::Regs::MaxShaderStage,
                          Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS>;

class GraphicsPipeline {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

public:
    /// Emits the stages' SPIR-V on a worker; with libraries it also returns each stage's SPIR-V
    /// hash (zero for absent stages).
    using EmitStagesFunc = std::function<std::array<vk::ShaderModule, NUM_STAGES>(
        std::array<u128, NUM_STAGES>& spirv_hashes)>;

    /// Graphics pipeline libraries (EDEN_NX2_GPL): non-null `library_cache` builds the pipeline
    /// from per-stage libraries; the stages were emitted with one descriptor set per stage.
    /// `libraries_only` builds the shader libraries and nothing else (warm-up, no draw).
    // Declared outside the class: GCC rejects a default argument `= {}` of a nested struct
    // with member initializers inside its enclosing class.
    using LibraryParams = GraphicsPipelineLibraryParams;

    explicit GraphicsPipeline(
        Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache,
        vk::PipelineCache& pipeline_cache, VideoCore::ShaderNotify* shader_notify,
        const Device& device, DescriptorPool& descriptor_pool,
        GuestDescriptorQueue& guest_descriptor_queue,
        DescriptorBufferRing& descriptor_buffer_ring, DrawTemplates& draw_templates,
        Common::ThreadWorker* worker_thread, PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
        const GraphicsPipelineCacheKey& key, std::array<vk::ShaderModule, NUM_STAGES> stages,
        const std::array<const Shader::Info*, NUM_STAGES>& infos,
        EmitStagesFunc emit_stages = {}, LibraryParams library_params = {},
        std::optional<MeshEmulationSetup> mesh_emulation = std::nullopt);

    static std::chrono::nanoseconds TakeHoldTime() noexcept;
    /// Draws held for unbuilt pipelines this session (never reset; the worst frame line takes
    /// differences of it).
    static std::chrono::nanoseconds HoldTimeTotal() noexcept;

    /// Worker builds of kept SPIR-V since the last call: over KeptBuildSlow means the driver
    /// compiled the pipeline again (not in its cache), under it a cache hit. Over KeptBuildLong
    /// is a compile long enough that a draw waiting for it is a visible freeze.
    static constexpr std::chrono::milliseconds KeptBuildSlow{20};
    static constexpr std::chrono::milliseconds KeptBuildLong{250};
    struct KeptBuildStats {
        u64 builds;
        u64 slow;
        u64 long_builds;
        std::chrono::nanoseconds slow_time;
        std::chrono::nanoseconds longest;
    };
    static KeptBuildStats TakeKeptBuildStats() noexcept;

    bool HasDynamicVertexInput() const noexcept { return key.state.dynamic_vertex_input; }
    [[nodiscard]] const GraphicsPipelineCacheKey& Key() const noexcept {
        return key;
    }
    const std::array<u64, 6>& ShaderHashes() const noexcept {
        return key.unique_hashes;
    }

    /// What the memory stage's FillDraw needs (SEAM = 2, vk_mem_stage.cpp).
    [[nodiscard]] MemStage::DrawFacts MemFacts() const noexcept {
        return {
            .uniform_masks = &enabled_uniform_buffer_masks,
            .uniform_sizes = &uniform_buffer_sizes,
            .stage_infos = stage_infos.data(),
            .stages = spec_stages,
            .has_texel_buffers = has_texel_buffers,
            .uses_descriptor_buffer = uses_descriptor_buffer,
            .hashes = {key.unique_hashes[1], key.unique_hashes[5]},
            .reads_handles = reads_handles,
        };
    }

    struct SampledImage {
        u32 stage;
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
    };

    /// The sampled textures of the last configured draw, kept while a sampled frame dump runs.
    std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    bool SupportsAlphaToCoverage() const noexcept {
        return fragment_has_color0_output;
    }

    bool SupportsAlphaToOne() const noexcept {
        return fragment_has_color0_output;
    }

    /// No texture, image, texture buffer or image buffer descriptor in any stage: a draw binds no
    /// image (FRITCHII_DRAW_KINDS counts these; the delta path's first cut, EXP118).
    [[nodiscard]] bool SamplesNothing() const noexcept {
        return samples_nothing;
    }

    /// A texture array whose handles a handle program reads from a table (one shadow map per
    /// light, EDEN_NX2_LIGHT_TABLE_ARRAYS). Such a pipeline never takes a delta draw
    /// (DeltaPathDecide).
    [[nodiscard]] bool HasHandleTable() const noexcept {
        return has_handle_table;
    }

    /// Texture handles of the last configured draw whose handle program gave 0.
    u32 ZeroTextureHandles() const noexcept {
        return zero_texture_handles;
    }

    bool FragmentCanDiscard() const noexcept {
        return stage_infos[NUM_STAGES - 1].uses_demote_to_helper_invocation;
    }

    /// Color components of a render target the fragment shader writes, as VkColorComponentFlags
    VkColorComponentFlags ColorOutputMask(size_t render_target) const noexcept {
        return (color_output_mask >> (render_target * 4)) & 0xF;
    }

    /// Color components the fragment shader writes, 4 bits per render target
    u32 ColorOutputMask() const noexcept {
        return color_output_mask;
    }

    bool UsesExtendedDynamicState() const noexcept {
        return key.state.extended_dynamic_state != 0;
    }

    /// Switch 2 task/mesh pipeline: drawn with vkCmdDrawMeshTasksEXT.
    [[nodiscard]] bool IsMeshPipeline() const noexcept {
        return stage_infos[0].is_task || stage_infos[0].is_mesh;
    }

    /// Has tessellation control and evaluation stages: drawn as a patch list.
    [[nodiscard]] bool HasTessellation() const noexcept {
        return has_tessellation;
    }

    /// Mesh pipeline drawn through the compute emulation (MeshEmulationSetup).
    [[nodiscard]] bool IsEmulatedMesh() const noexcept {
        return mesh_emulation.has_value();
    }

    /// Output bytes the emulation needs for one mesh workgroup.
    [[nodiscard]] u64 MeshEmulationBlockBytes() const noexcept {
        return mesh_emulation ? u64{mesh_emulation->block_words} * 4 : 0;
    }

    /// Vertices of the emulation's point, line or triangle list for one mesh workgroup.
    [[nodiscard]] u32 MeshEmulationVerticesPerGroup() const noexcept {
        return mesh_emulation
                   ? mesh_emulation->max_primitives * mesh_emulation->primitive_vertices
                   : 0;
    }

    /// The emulation's setup (output layout), or null.
    [[nodiscard]] const MeshEmulationSetup* MeshEmulation() const noexcept {
        return mesh_emulation ? &*mesh_emulation : nullptr;
    }

    /// The emulation's output buffer and workgroup count for the next Configure.
    void SetMeshEmulationOutput(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                u32 groups) noexcept {
        mesh_output_buffer = buffer;
        mesh_output_offset = offset;
        mesh_output_size = size;
        mesh_groups = groups;
    }

    struct MeshEmulationBuffer {
        VkBuffer buffer{};
        VkDeviceSize offset{};
        VkDeviceSize size{};
    };
    /// With a task program: the task payload, offset and argument buffers, the arena counter,
    /// and the words of the output arena (the draw's group count is the task's).
    void SetMeshEmulationTaskBuffers(const MeshEmulationBuffer& payload,
                                     const MeshEmulationBuffer& offsets,
                                     const MeshEmulationBuffer& args,
                                     const MeshEmulationBuffer& counter,
                                     u32 capacity_words) noexcept {
        mesh_task_buffers = {payload, offsets, args, counter};
        mesh_capacity = capacity_words;
    }

    [[nodiscard]] bool MeshEmulationHasTask() const noexcept {
        return mesh_emulation && mesh_emulation->has_task;
    }

    [[nodiscard]] u32 MeshEmulationPayloadStride() const noexcept {
        return mesh_emulation ? mesh_emulation->payload_stride : 0;
    }
    GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept = delete;
    GraphicsPipeline(GraphicsPipeline&&) noexcept = delete;

    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(const GraphicsPipeline&) = delete;

    void AddTransition(GraphicsPipeline* transition);

    /// `keep_last` (EDEN_NX2_FASTDRAW_COUNT or EXP106 on): the draw is also kept as the
    /// pipeline's last draw (KeptLastDraw), by a second instantiation of ConfigureImpl; with it
    /// off the draw runs the same code as before.
    bool Configure(bool is_indexed, bool keep_last = false) {
        if (keep_last) [[unlikely]] {
            return configure_keep_func(this, is_indexed);
        }
        return configure_func(this, is_indexed);
    }

    /// EXP118 (fritchii_delta_path.h), for a pipeline that samples nothing (SamplesNothing), on a
    /// draw that follows this pipeline's last with only light state changed: Configure without
    /// the descriptor pools' sync and the texture steps (nothing to bind). Everything else runs
    /// as in Configure.
    bool ConfigureDelta(bool is_indexed) {
        return configure_delta_func(this, is_indexed);
    }

    /// FRITCHII_NULL_RENDER (test only): the next Configure does only the buffer side of the draw
    /// (fable_b_one_follower 7 run 1: thread E's part) and returns false. RasterizerVulkan::
    /// PrepareDraw sets it for a draw it null-renders; ConfigureImpl clears it.
    void SetNullRenderNext() {
        null_render_next = true;
    }

    /// Delta draws (D2): the previous draw as its Configure kept it with `keep_last`, or null
    /// when it wasn't kept whole. Only meaningful for the pipeline's last configured draw.
    [[nodiscard]] const KeptDraw* KeptLastDraw() const noexcept {
        return kept_draw && kept_draw->valid ? kept_draw.get() : nullptr;
    }

    /// Delta draws (D2; the original ComputedHandlesChanged): whether a handle the
    /// kept draw computed with a handle program (from constant and storage buffers) evaluates to
    /// another value now with `maxwell3d_`'s constant buffers: a delta draw would keep views and
    /// samplers resolved from the old one. Needs a KeptLastDraw.
    [[nodiscard]] bool ComputedHandlesChanged(const Tegra::Engines::Maxwell3D& maxwell3d_,
                                              Tegra::MemoryManager& gpu_memory_) const;

    void FillHandleValues(MemStage& seam, bool null_pointer_handles, bool in_configure);

    [[nodiscard]] GraphicsPipeline* Next(const GraphicsPipelineCacheKey& current_key) noexcept {
        if (key == current_key) {
            return this;
        }
        const auto it{std::find(transition_keys.begin(), transition_keys.end(), current_key)};
        return it != transition_keys.end() ? transitions[std::distance(transition_keys.begin(), it)]
                                           : nullptr;
    }

    /// What Next tests, in Next's order (FRITCHII_PIPELINE_KEY_HASH counts them).
    struct NextCounts {
        u64 candidates{}; ///< Keys looked at (this pipeline's and the transitions')
        u64 compared{};   ///< Fingerprints that matched: the full key compare ran
        u64 collisions{}; ///< ... and the keys differed
    };

    /// Next with the fingerprints first (FRITCHII_PIPELINE_KEY_HASH): `current_hash` is
    /// current_key.Fingerprint(). Equal keys have equal fingerprints, so a candidate whose
    /// fingerprint differs is one Next's compare rejects; the others get Next's compare. The same
    /// candidates in the same order: the same answer.
    [[nodiscard]] GraphicsPipeline* NextHashed(const GraphicsPipelineCacheKey& current_key,
                                               u64 current_hash, NextCounts& counts) noexcept {
        ++counts.candidates;
        if (key_hash == current_hash) {
            ++counts.compared;
            if (key == current_key) {
                return this;
            }
            ++counts.collisions;
        }
        const size_t count = transition_hashes.size();
        counts.candidates += count;
        for (size_t i = 0; i < count; ++i) {
            if (transition_hashes[i] != current_hash) {
                continue;
            }
            ++counts.compared;
            if (transition_keys[i] == current_key) {
                return transitions[i];
            }
            ++counts.collisions;
        }
        return nullptr;
    }

    /// The frame the pipeline cache last handed this pipeline out (evicts long unused ones).
    void MarkUsed(u64 frame) noexcept {
        last_use_frame = frame;
        used = true;
    }

    [[nodiscard]] u64 LastUse() const noexcept {
        return last_use_frame;
    }

    /// Whether the cache has handed this pipeline out at all (frame 0 included), as opposed to a
    /// pipeline prebuilt from the disk cache that no draw has used yet.
    [[nodiscard]] bool WasUsed() const noexcept {
        return used;
    }

    /// The constant buffer slots each program binds as storage buffer pointers
    [[nodiscard]] const VideoCommon::EvictedPointerSlots::Slots& MappedPointers() const noexcept {
        return mapped_pointers;
    }

    void SetMappedPointers(VideoCommon::EvictedPointerSlots::Slots slots) noexcept {
        mapped_pointers = std::move(slots);
    }

    /// Forgets the transitions, which may point to pipelines the cache evicted.
    void ClearTransitions() noexcept {
        transition_keys.clear();
        transition_hashes.clear();
        transitions.clear();
        occl_sibling = nullptr; // FRITCHII_OCCL_LEAN: found again by key
    }

    /// FRITCHII_OCCL_LEAN (fritchii_occl_lean.h; the pipeline cache's, E only): 0 not classed
    /// yet, else OcclLean::PipelineClass + 1; and the same key without the pixel program once
    /// found or made (cleared with the transitions, as the cache may evict it).
    u8 occl_class{};
    GraphicsPipeline* occl_sibling{};

    // FRITCHII_STAMP_SKIP_BEGIN: left out of the Preparing shaders stamp (CMakeModules/Nx2ShaderStamp.cmake).
    // Only what can't change a pipeline the driver compiles belongs here.
    /// FRITCHII_PREP_OCCL_VARIANTS: on a pipeline made from the kept infos and SPIR-V
    /// (CreatePrecompiledGraphicsPipeline), the index it came from (a PrecompiledKept::Index)
    /// and its record there (a PrecompiledKept::Record, alive as long as the index): its
    /// occlusion flag variants are made from them (PipelineCache::MakeKeptOcclVariant).
    /// kept_occl_tried: bit per kind looked up.
    std::shared_ptr<const void> kept_index;
    const void* kept_record{};
    u8 kept_occl_tried{};
    /// FRITCHII_PREP_OCCL_VARIANTS: a pipeline built another way (the disk cache's prebuild, a
    /// translation) whose key the kept index holds: its record looked up at its first variant
    /// request (PipelineCache::LookUpKeptParent; the reads on the occlusion lane), the record's
    /// infos kept here (a KeptParentLookup, vk_pipeline_cache.cpp). kept_lookup_state: 0 not
    /// looked up, 1 reading, 2 done (bound to a record, or none fits). kept_path: how this
    /// pipeline was built, for the prepared variants' counter (0 other, 1 fast path, 2 disk cache).
    std::shared_ptr<void> kept_lookup;
    u8 kept_lookup_state{};
    u8 kept_path{};
    /// FRITCHII_OCCL_FLAG (fritchii_occl_flag.h): this pipeline's flag variant (the same key with
    /// the host flag pixel program), owned here and never in the cache; on a flag variant, the
    /// pipeline it was made from. occl_flag_state: 0 not tried, 1 made, 2 refused for good.
    std::unique_ptr<GraphicsPipeline> occl_flag;
    GraphicsPipeline* occl_flag_parent{};
    u8 occl_flag_state{};
    /// FRITCHII_OCCL_FLAG: the colour flag variant, for colour-writing draws (the same key with
    /// the guest pixel program translated with RuntimeInfo::occl_flag_store). One pipeline can
    /// draw with colour on and off (dynamic write masks), so it has a slot of its own.
    /// occl_flag_colour_state: 0 not tried, 1 made, 2 refused for good.
    std::unique_ptr<GraphicsPipeline> occl_flag_colour;
    u8 occl_flag_colour_state{};
    /// FRITCHII_OCCL_FLAG: the pre-pass variant, for counted colour-off draws whose program
    /// discards (the guest pixel program with early fragment tests and the flag store at its
    /// end, drawn first with depth writes off; the real draw follows with this pipeline).
    std::unique_ptr<GraphicsPipeline> occl_flag_prepass;
    u8 occl_flag_prepass_state{};
    /// FRITCHII_OCCL_FLAG colour pre-pass without dynamic colour write masks: the host flag
    /// variant built from this key with every attachment's write mask and blend cleared.
    std::unique_ptr<GraphicsPipeline> occl_flag_nocolour;
    u8 occl_flag_nocolour_state{};
    /// FRITCHII_OCCL_FLAG: the discard pre-pass for colour draws without dynamic colour write
    /// masks: the pre-pass variant built from this key with every attachment's writes off.
    std::unique_ptr<GraphicsPipeline> occl_flag_prepass_nocolour;
    u8 occl_flag_prepass_nocolour_state{};
    /// On a flag variant, its kind: 1 host flag program, 2 colour, 3 pre-pass, 4 host flag
    /// program with colour writes off in its key, 5 pre-pass with colour writes off in its key.
    static constexpr u8 OCCL_KIND_PREPASS = 3;
    static constexpr u8 OCCL_KIND_PREPASS_NOCOLOUR = 5;
    /// The stages its push constant range covers (what every push to its layout names).
    [[nodiscard]] VkShaderStageFlags PushStageFlags() const noexcept {
        return mesh_emulation ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT
               : IsMeshPipeline()
                   ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT |
                         VK_SHADER_STAGE_MESH_BIT_EXT
                   : VK_SHADER_STAGE_ALL_GRAPHICS;
    }
    /// Built from graphics pipeline libraries (its stages compiled apart from each other).
    [[nodiscard]] bool UsesLibraries() const noexcept {
        return library_cache != nullptr;
    }
    /// A discard pre-pass variant (drawn first, then the draw itself with its pipeline).
    [[nodiscard]] bool IsOcclDiscardPrepass() const noexcept {
        return occl_flag_kind == OCCL_KIND_PREPASS || occl_flag_kind == OCCL_KIND_PREPASS_NOCOLOUR;
    }
    u8 occl_flag_kind{};
    /// On a flag variant: 1 its translation's descriptors equal its parent's (the parent can
    /// bind them for a real draw after a pre-pass), 2 they differ.
    u8 occl_same_descriptors{};
    /// FRITCHII_OCCL_FLAG_PERSIST, as a parent (GPU thread, computed once; 0 not yet): the hash
    /// of its descriptor lists (PipelineCache's OcclDescriptorSignature), and its refusal memo
    /// hash (its key without the driver fields, that signature and this build).
    u64 occl_persist_signature{};
    u64 occl_persist_memo{};
    /// On a flag variant: its last Configure's rescaling words and render area, which the
    /// parent pushes again when it is bound for the real draw after a pre-pass
    /// (ConfigureAfterOcclPrepass).
    std::remove_cvref_t<decltype(std::declval<const RescalingPushConstant&>().Data())>
        occl_prepass_rescaling{};
    RenderAreaPushConstant occl_prepass_render_area{};

    /// FRITCHII_OCCL_FLAG pre-pass: binds this pipeline (the parent) for the real draw that
    /// follows `variant`'s pre-pass draw, with the descriptors `variant`'s Configure acquired for
    /// this draw (the same stage infos: the same entries) and its push constants. False when the
    /// draw cannot be configured (the real draw is not recorded then).
    bool ConfigureAfterOcclPrepass(const GraphicsPipeline& variant);

    /// FRITCHII_OCCL_FLAG: the layout the flag slot address is pushed with.
    [[nodiscard]] VkPipelineLayout Layout() const noexcept {
        return *pipeline_layout;
    }

    // FRITCHII_STAMP_SKIP_END
    [[nodiscard]] bool IsBuilt() const noexcept {
        return build->IsBuilt();
    }

    /// FRITCHII_PIPELINE_WAIT_BUDGET: waits at most `timeout` for the build (never building it
    /// on the calling thread). True once built.
    [[nodiscard]] bool WaitBuiltFor(std::chrono::nanoseconds timeout) const {
        return build->WaitFor(timeout);
    }

    /// The programs' shader info, by stage (EDEN_NX2_FASTDRAW_COUNT reads the handle words).
    [[nodiscard]] const std::array<Shader::Info, NUM_STAGES>& StageInfos() const noexcept {
        return stage_infos;
    }

    /// EDEN_NX2_FASTDRAW_COUNT: what the counter derived from StageInfos, on its first draw.
    mutable std::unique_ptr<VideoCommon::Nx2FastDrawCount::PipelineInfo> fast_draw_count_info;

    [[nodiscard]] bool Failed() const noexcept {
        return build->Failed();
    }

    template <typename Spec, bool KeepLast = false, bool Delta = false>
    static auto MakeConfigureSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed) {
            return pl->ConfigureImpl<Spec, KeepLast, Delta>(is_indexed);
        };
    }

    void SetEngine(Tegra::Engines::Maxwell3D* maxwell3d_, Tegra::MemoryManager* gpu_memory_) {
        maxwell3d = maxwell3d_;
        gpu_memory = gpu_memory_;
    }

private:
    /// EDEN_NX2_EXP32 (texture binding memo; the original BINDMEMO reduced to "the same
    /// textures as this pipeline's previous draw"): that draw's texture bindings as resolved and
    /// pushed, for pipelines whose image bindings are all sampled textures (no texture or image
    /// buffers, no storage images). A draw reading the same handles with the same texture cache
    /// state (TextureCache::TextureMemoState), the same guest TIC/TSC entries and clean images
    /// takes the view and sampler ids and the descriptor entries from it instead of resolving them
    /// (GetSamplerId, FillImageViews) and building them (PushImageDescriptors).
    struct TextureMemo {
        bool valid{};
        u64 generation{};  ///< the switch generation it was recorded in (a switch change drops it)
        u64 touch_frame{}; ///< the frame its images were last touched in the LRU
        TextureCache::TextureMemoState state{};
        std::vector<u32> handles;         ///< per texture: the TIC handle read
        std::vector<u32> sampler_handles; ///< per texture: the TSC handle read
        std::vector<TextureCache::TextureMemoEntry> entries;
        std::vector<VideoCommon::ImageViewId> views;
        std::vector<VideoCommon::SamplerId> samplers;
        std::vector<DescriptorUpdateEntry> descriptors; ///< per texture, in push order
        std::vector<u8> rescaled; ///< per texture descriptor (array), in push order
    };

    /// Whether the memo holds this draw's textures (counts why not).
    [[nodiscard]] bool TextureMemoHit(const TextureCache::TextureMemoState& state,
                                      std::span<const VideoCommon::ImageViewInOut> views,
                                      std::span<const u32> sampler_handles);

    /// Remembers the textures this draw resolved and pushed, when nothing in the texture cache
    /// changed since `state` was taken and the tables still hold them as resolved.
    void RecordTextureMemo(const TextureCache::TextureMemoState& state,
                           std::span<const VideoCommon::ImageViewInOut> views,
                           std::span<const u32> sampler_handles,
                           std::span<const VideoCommon::SamplerId> samplers,
                           std::span<const DescriptorUpdateEntry> descriptors,
                           std::span<const u8> rescaled);

    /// EXP32 verify: a memo hit resolved and pushed the full way; compares the memo with it,
    /// logs and counts a difference, and turns the memo off when the switch asks (bit 4).
    bool VerifyTextureMemo(std::span<const VideoCommon::ImageViewInOut> views,
                           std::span<const VideoCommon::SamplerId> samplers,
                           std::span<const DescriptorUpdateEntry> descriptors,
                           std::span<const u8> rescaled, int switch_value);

    /// EDEN_NX2_TEX_BANK_GEN (the recomp's per-stage texture bank
    /// generations): this pipeline's last resolved textures and what they were resolved from.
    /// Only for pipelines whose bindings are all sampled textures with handles read straight
    /// from constant buffers (TexBankEligible).
    struct TexBankMemo {
        bool valid{};
        s8 eligible{-1}; ///< -1: not decided yet
        int switch_value{};
        u64 page_epoch{}; ///< TextureCache::TexBankEpoch() the page pointers belong to
        TextureCache::TextureMemoState state{};
        bool via_header_index{};
        bool null_pointer_handles{}; ///< EXP21
        /// Record backoff: a memo that was never hit before the next record makes the pipeline
        /// skip 2^backoff - 1 records (exact either way: only whether to keep a memo).
        bool hit_since_record{};
        bool recorded_once{};
        u8 backoff{};
        u8 countdown{};
        struct Cbuf {
            u32 stage;
            u32 index;
            GPUVAddr address;
            u32 size;
            bool enabled;
        };
        std::vector<Cbuf> cbufs; ///< the bindings the handle words are read from
        struct Page {
            TextureCache::TexBankPage* page;
            u64 gen;
        };
        std::vector<Page> pages; ///< handle word, TIC and TSC entry pages, at record time
        std::vector<u32> handles; ///< per texture: the TIC handle (views[i].index)
        std::vector<VideoCommon::ImageViewId> views;
        std::vector<VideoCommon::SamplerId> samplers;
    };

    /// Whether this pipeline can keep a texture bank memo (decided once).
    [[nodiscard]] bool TexBankEligible();

    /// Whether the memo holds this draw's textures: same state, same handle constant buffer
    /// bindings, every page at its generation (counts why not).
    [[nodiscard]] bool TexBankHit(const TextureCache::TextureMemoState& state,
                                  bool via_header_index, bool null_pointer_handles,
                                  int switch_value);

    /// Remembers what this draw resolved the full way, with the pages it read from.
    void RecordTexBank(const TextureCache::TextureMemoState& state,
                       std::span<const VideoCommon::ImageViewInOut> views,
                       std::span<const u32> sampler_handles,
                       std::span<const VideoCommon::SamplerId> samplers, bool via_header_index,
                       bool null_pointer_handles, int switch_value);

    /// EXP87: keeps the draw's textures and image descriptors, which the guest descriptor queue
    /// holds at `ranges`, for later draws with the same key. With `verified`, the template the key
    /// found is compared with them first (EXP87 = 2).
    void KeepDrawTemplate(DrawTemplate* verified, const VideoCommon::DrawTemplateContext& context,
                          std::span<const u32> handles, u64 hash,
                          std::span<const VideoCommon::ImageViewInOut> views,
                          std::span<const VideoCommon::SamplerId> samplers,
                          const std::array<std::pair<u32, u32>, NUM_STAGES>& ranges,
                          const RescalingPushConstant& rescaling);

    /// Delta draws (D2): fills kept_draw from the draw ConfigureImpl<Spec, true> just resolved
    /// and pushed (`ranges`: each stage's image descriptor entries in the guest descriptor
    /// queue). `handles_read`: the draw read its handles (not a TEX_BANK_GEN hit).
    void KeepLastDraw(std::span<const VideoCommon::ImageViewInOut> views,
                      std::span<const VideoCommon::SamplerId> samplers,
                      const std::array<std::pair<u32, u32>, NUM_STAGES>& ranges,
                      const RescalingPushConstant& rescaling, bool handles_read);

    /// KeepLast: also keep the draw as the pipeline's last (Configure's keep_last).
    /// Delta: ConfigureDelta (EXP118; pipelines that sample nothing, never with KeepLast).
    template <typename Spec, bool KeepLast = false, bool Delta = false>
    bool ConfigureImpl(bool is_indexed);

    bool ConfigureDraw(const RescalingPushConstant& rescaling,
                       const RenderAreaPushConstant& render_are);

    struct PipelineStates;

    /// The fixed-function state of the pipeline (shared by the monolithic and library builds).
    void FillStates(PipelineStates& states) const;

    void MakePipeline(VkRenderPass render_pass);

    /// Library build (EDEN_NX2_GPL): gets or builds the two shader libraries, then (unless
    /// libraries_only) links them with this pipeline's vertex input and fragment output parts.
    /// False when a library failed; the caller then builds the pipeline the monolithic way.
    bool MakeLibraryPipeline(VkRenderPass render_pass);

    GplLibrary MakeShaderLibrary(GplPart part, const PipelineStates& states,
                                 VkRenderPass render_pass) const;

    std::string ShaderLibraryKey(GplPart part, const PipelineStates& states,
                                 VkRenderPass render_pass) const;

    vk::Pipeline LinkLibraries(VkRenderPass render_pass, bool optimized);

    /// The library route failed: per-stage sets kept, with a layout a monolithic pipeline takes.
    void FallBackToMonolithic(VkRenderPass render_pass);

    void Validate();

    const GraphicsPipelineCacheKey key;
    /// key.Fingerprint(): the key never changes (NextHashed).
    const u64 key_hash{key.Fingerprint()};
    Tegra::Engines::Maxwell3D* maxwell3d;
    Tegra::MemoryManager* gpu_memory;
    const Device& device;
    TextureCache& texture_cache;
    BufferCache& buffer_cache;
    vk::PipelineCache& pipeline_cache;
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    DrawTemplates& draw_templates;
    const u64 template_owner;
    u32 template_misses{}; ///< Draws in a row no template was found for (EXP87)
    /// Every texture is sampled with a handle read from a constant buffer, nothing else is bound
    /// as an image: ConfigureImpl can take the textures from a draw template (EXP87).
    bool template_eligible{};

    bool (*configure_func)(GraphicsPipeline*, bool){};
    /// ConfigureImpl<Spec, true> for the same Spec (Configure with keep_last).
    bool (*configure_keep_func)(GraphicsPipeline*, bool){};
    /// ConfigureImpl<Spec, false, true> (ConfigureDelta); null unless the pipeline samples nothing.
    bool (*configure_delta_func)(GraphicsPipeline*, bool){};
    /// Delta draws (D2): the last draw configured with keep_last; made on the first.
    std::unique_ptr<KeptDraw> kept_draw;
    /// FRITCHII_NULL_RENDER: SetNullRenderNext, and the handle words its draws read.
    bool null_render_next{};
    std::vector<u32> null_render_handles;
    /// fable_stage_levelpoints 18.4 row 2 (GPU profile on): the last draw's handle range key and
    /// its pages' write generations (FritchiiPrice::ClassifyHandles).
    u64 price_handle_key{};
    u64 price_handle_sum{};

    std::vector<GraphicsPipelineCacheKey> transition_keys;
    /// Each transition key's Fingerprint(), kept with transition_keys (AddTransition and
    /// ClearTransitions are their only writers).
    std::vector<u64> transition_hashes;
    std::vector<GraphicsPipeline*> transitions;
    VideoCommon::EvictedPointerSlots::Slots mapped_pointers;
    u64 last_use_frame{};
    bool used{};

    std::array<vk::ShaderModule, NUM_STAGES> spv_modules;

    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    u32 num_descriptor_entries{};
    size_t num_image_elements{};
    std::vector<SampledImage> sampled_images;
    u32 num_textures{};
    bool has_tessellation{};
    bool fragment_has_color0_output{};
    u32 zero_texture_handles{};
    u32 color_output_mask{};
    bool writes_storage{};
    bool samples_nothing{}; ///< SamplesNothing
    bool has_handle_table{}; ///< HasHandleTable
    /// The memory stage's facts: the stages Configure prepares (its Spec's), and whether any
    /// stage binds texture or image buffers (the fill then stays in Configure: LateFill).
    u32 spec_stages{};
    bool has_texel_buffers{};
    bool reads_handles{};
    TextureMemo texture_memo;
    TexBankMemo tex_bank_memo;
    /// EDEN_NX2_TEX_BANK_GEN: (stage, constant buffer index) of every handle word read.
    std::vector<std::pair<u32, u32>> tex_bank_cbuf_slots;

    /// Graphics pipeline libraries (EDEN_NX2_GPL): one descriptor set per stage (set index =
    /// stage index), the shader libraries, and this pipeline's own vertex input and fragment
    /// output libraries.
    GraphicsPipelineLibraryCache* const library_cache;
    const bool libraries_only;
    std::array<u128, NUM_STAGES> spirv_hashes{};
    std::array<vk::DescriptorSetLayout, NUM_STAGES> stage_set_layouts;
    std::array<std::string, NUM_STAGES> stage_set_signatures;
    std::array<DescriptorAllocator, NUM_STAGES> stage_allocators;
    std::array<vk::DescriptorUpdateTemplate, NUM_STAGES> stage_templates;
    std::array<size_t, NUM_STAGES> stage_descriptor_offsets{}; ///< Bytes into the update data
    bool has_stage_sets{};
    std::array<std::shared_ptr<const GplLibrary>, 2> shader_libraries;
    vk::Pipeline vertex_input_library;
    vk::Pipeline fragment_output_library;
    vk::DescriptorSetLayout fallback_empty_set_layout;
    bool linked{};

    vk::DescriptorSetLayout descriptor_set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    DescriptorBufferLayout descriptor_buffer_layout;
    std::vector<DescriptorUpdateEntry> last_descriptor_payload;
    VkDeviceSize last_descriptor_buffer_offset{};
    u32 last_descriptor_buffer_chunk{};
    u64 last_descriptor_buffer_generation{};

    const bool worker_only_build;
    /// LibraryParams::prepared: a worker build of kept SPIR-V (TakeKeptBuildStats).
    const bool prepared_build;
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};

    // Mesh emulation (MeshEmulationSetup).
    std::optional<MeshEmulationSetup> mesh_emulation;
    vk::ShaderModule mesh_vertex_module;
    vk::Pipeline mesh_compute_pipeline;
    VkDescriptorSet mesh_descriptor_set{}; ///< shared by the dispatch and the draw it feeds
    VkBuffer mesh_output_buffer{};
    VkDeviceSize mesh_output_offset{};
    VkDeviceSize mesh_output_size{};
    u32 mesh_groups{};
    // With a task program: its compute pipeline, the distribution pass between it and the mesh
    // dispatch, and the per-draw buffers (payload, offsets, arguments, arena counter).
    vk::Pipeline task_compute_pipeline;
    vk::ShaderModule distribution_module;
    vk::DescriptorSetLayout distribution_set_layout;
    vk::PipelineLayout distribution_layout;
    vk::Pipeline distribution_pipeline;
    DescriptorAllocator distribution_allocator;
    std::array<MeshEmulationBuffer, 4> mesh_task_buffers{};
    u32 mesh_capacity{};

    std::optional<PipelineBuild> build;
};

} // namespace Vulkan
