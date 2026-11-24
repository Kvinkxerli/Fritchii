// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view_info.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

enum class ImageFlagBits : u32 {
    AcceleratedUpload = 1 << 0, ///< Upload can be accelerated in the GPU
    Converted = 1 << 1,   ///< Guest format is not supported natively and it has to be converted
    CpuModified = 1 << 2, ///< Contents have been modified from the CPU
    GpuModified = 1 << 3, ///< Contents have been modified from the GPU
    Tracked = 1 << 4,     ///< Writes and reads are being hooked from the CPU JIT
    Strong = 1 << 5,      ///< Exists in the image table, the dimensions are can be trusted
    Registered = 1 << 6,  ///< True when the image is registered
    Picked = 1 << 7,      ///< Temporary flag to mark the image as picked
    Remapped = 1 << 8,    ///< Image has been remapped.
    Sparse = 1 << 9,      ///< Image has non continuous submemory.

    // Garbage Collection Flags
    BadOverlap = 1 << 10, ///< This image overlaps other but doesn't fit, has higher
                          ///< garbage collection priority
    Alias = 1 << 11,      ///< This image has aliases and has priority on garbage
                          ///< collection
    CostlyLoad = 1 << 12, ///< Protected from low-tier GC as it is costly to load back.

    // Rescaler
    Rescaled = 1 << 13,
    CheckingRescalable = 1 << 14,
    IsRescalable = 1 << 15,

    AsynchronousDecode = 1 << 16,
    IsDecoding = 1 << 17, ///< Is currently being decoded asynchronously.
    Unbacked = 1 << 18,   ///< Host image has no memory (too big to back); never upload/download.
    /// The host image keeps only mip levels >= host_level_offset (its decoded copy would not fit
    /// in VRAM); uploads skip the rest, downloads are never done.
    TrimmedLevels = 1 << 19,
    ZeroFilled = 1 << 20, ///< Cleared to zero blocks before its first asynchronous decode.
    /// Over Switch 2 sparse blocks nobody wrote: all zeros, so the host image has no memory
    /// (Unbacked) until the first write, which makes a backed image replace it.
    UnwrittenSparse = 1 << 21,
    /// A large array over Switch 2 sparse blocks: unbacked, each layer gets device memory when
    /// something is written to it (Image::BindLayers) and only those layers are uploaded.
    SparseLayers = 1 << 22,
    /// A large single image over Switch 2 sparse blocks: unbacked, each host sparse block gets
    /// device memory when the guest memory under it can hold data (Image::BindTiles), and only
    /// the parts written are uploaded.
    SparseTiles = 1 << 23,
    /// EXP67: the host image is larger than the guest size (rounded up to a bucket) so the image
    /// pool can reuse it for the next frame's target of nearby dims; the guest size is the top
    /// left rectangle. Never sampled with normalized coordinates (a texture lookup that finds one
    /// stops the padding at its address).
    HostPadded = 1 << 24,
    /// EXP68 bit 4: made at an address whose target keeps changing size: freed when the GPU is
    /// done instead of going to the EXP42 pool (those dims never come back).
    NoPool = 1 << 25,
    /// EXP121: the collector recorded its download; deleted once the GPU has written it out.
    CollectPending = 1 << 26,
    /// FRITCHII_SCALED_EXTENT_CAP: its scaled size would pass the device's limits, so it stays
    /// native; logged once.
    ScaleLimited = 1 << 27,
    /// FRITCHII_SCALE_DEPTH_MAX: a depth image kept native for its scaled size; counted once.
    DepthNative = 1 << 28,
    TransientTarget = 1 << 29,
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

struct ImageViewInfo;

struct AliasedImage {
    std::vector<ImageCopy> copies;
    ImageId id;
};

struct NullImageParams {};

struct ImageBase {
    explicit ImageBase(const ImageInfo& info, GPUVAddr gpu_addr, VAddr cpu_addr);
    explicit ImageBase(const NullImageParams&);

    [[nodiscard]] std::optional<SubresourceBase> TryFindBase(GPUVAddr other_addr) const noexcept;

    [[nodiscard]] ImageViewId FindView(const ImageViewInfo& view_info) const noexcept;

    void InsertView(const ImageViewInfo& view_info, ImageViewId image_view_id);

    [[nodiscard]] bool IsSafeGpuCopy() const noexcept;

    [[nodiscard]] bool Overlaps(VAddr overlap_cpu_addr, size_t overlap_size) const noexcept {
        const VAddr overlap_end = overlap_cpu_addr + overlap_size;
        return cpu_addr < overlap_end && overlap_cpu_addr < cpu_addr_end;
    }

    [[nodiscard]] bool OverlapsGPU(GPUVAddr overlap_gpu_addr, size_t overlap_size) const noexcept {
        const VAddr overlap_end = overlap_gpu_addr + overlap_size;
        const GPUVAddr gpu_addr_end = gpu_addr + guest_size_bytes;
        return gpu_addr < overlap_end && overlap_gpu_addr < gpu_addr_end;
    }

    void CheckBadOverlapState();
    void CheckAliasState();

    bool HasScaled() const {
        return has_scaled;
    }

    ImageInfo info;

    u32 guest_size_bytes = 0;
    /// EXP76 & 2: CityHash128 of the guest bytes as they were when the image was last uploaded
    /// whole (0/0: none). An upload whose guest bytes still hash the same is skipped.
    u64 nx2_content_hash_lo = 0;
    u64 nx2_content_hash_hi = 0;
    /// EXP84: the range RegisterImage counted in image_count_filter (UnregisterImage removes
    /// exactly this one, whatever cpu_addr or the size say by then).
    u64 nx2_counted_addr = 0;
    u64 nx2_counted_size = 0;
    /// EXP100: this image's nx2_counted range is in TextureCache::gpu_image_filter (it was
    /// GPU-modified while registered); UnregisterImage removes it.
    bool nx2_gpu_counted = false;
    u32 unswizzled_size_bytes = 0;
    u32 converted_size_bytes = 0;
    u32 scale_rating = 0;
    u32 host_level_offset = 0; ///< Guest mip level held by host mip 0 (see TrimmedLevels).
    /// Large layered images uploaded by layer: one bit per layer written since the last upload
    /// while CpuModified. Empty while CpuModified means the whole image.
    std::vector<u64> dirty_layers;
    /// SparseTiles images: the guest byte ranges (offset, size) written since the last upload while
    /// CpuModified. Empty while CpuModified means all of it that can hold data.
    std::vector<std::pair<u64, u64>> dirty_ranges;
    u64 scale_tick = 0;
    bool has_scaled = false;

    size_t channel = 0;

    ImageFlagBits flags = ImageFlagBits::CpuModified;

    GPUVAddr gpu_addr = 0;
    VAddr cpu_addr = 0;
    VAddr cpu_addr_end = 0;

    u64 modification_tick = 0;
    /// FRITCHII_EARLY_COMPUTE (dry): the scheduler tick of its last GPU write (MarkModification).
    u64 early_write_tick = 0;
    size_t lru_index = SIZE_MAX;
    /// The frame tick of the last LRU touch (TextureCache::TouchImageLru, EDEN_NX2_EXP90); reset
    /// when the image gets a new LRU node.
    u64 lru_touched_tick = ~u64{0};
    /// The frame of its last use, kept when it leaves the LRU (UnregisterImage), for
    /// FRITCHII_TC_THRASH_GUARD.
    u64 last_use_tick = 0;

    /// Render target relocation guard (EXP43): the frame the GPU last wrote the image (~0 =
    /// never, or contents discarded), and the guard reasons this image already reported.
    u64 last_write_frame = ~u64{0};
    /// Consecutive render frames with a write, up to last_write_frame (NextWriteStreak).
    u32 write_streak = 0;
    u8 relocation_guard_noted = 0;
    /// FRITCHII_COVERED_RETIRE: the frame a newer image last came to overlap this one (a bad
    /// overlap in JoinImages), and the frame its coverage was last checked.
    u64 covered_overlap_frame = 0;
    u64 covered_check_frame = 0;
    /// Made where a GPU-written image was dropped without a write-back (TextureCache's lost
    /// contents counter): bits of TextureCache::LostBits, 0 once counted or overwritten whole.
    u8 lost_contents = 0;
    /// Why a drop path deleted it without a write-back (TextureCache::LostWhy), for the lost
    /// contents counter's attribution; 0 = no write-back was tried.
    u8 lost_drop_why = 0;
    /// FRITCHII_COLD_SWEEP_KEEP_RESAMPLED: the re-sampled set's generation this image was last
    /// looked up at (0 = never) and the answer (bit 0), whether it was counted as newly kept
    /// (bit 1), whether covered retire kept it (bit 2: looked at again every frame until it goes
    /// or is used again), and the frame it was last counted in the kept-now total.
    /// cold_keep_charge: the device memory it holds of the keep budget (0 = none), counted in
    /// TextureCache::cold_keep_resident until it is deleted, released, relocated or leaves
    /// cold_watch (TextureCache::ColdKeepUncharge).
    u32 cold_keep_gen = 0;
    u32 cold_keep_frame = 0;
    u8 cold_keep = 0;
    u64 cold_keep_charge = 0;

    /// Unmapped images (TextureCache::UnmapGPUMemory): the frame + 1 at which an unmap of its GPU
    /// memory flagged it Remapped (0 = none did, or it was found again since) and its device
    /// memory then.
    u64 unmapped_tick = 0;
    u64 unmapped_bytes = 0;
    /// Its place in the cache's list of such images (~0 = not in it), and with
    /// EDEN_NX2_SWEEP_UNMAPPED=2: 1 once the sweep would have deleted it, 2 once a use after that
    /// was counted.
    u32 unmapped_index = ~0U;
    u8 unmapped_verify = 0;

    /// Asynchronous shaders' skip rule (EDEN_NX2_ASYNC_DISCARD=2, ClassifyClearedTarget): the
    /// render frames of the image's last two full clears that had a write between them (~0 =
    /// none), and whether anything but a clear wrote it since the last one. Two clears with no
    /// draw between count as one, so a tick that falls between them cannot make a layer the game
    /// renders once look like a per-frame target.
    u64 last_clear_frame = ~u64{0};
    u64 previous_clear_frame = ~u64{0};
    bool drawn_since_clear = false;

    std::array<u32, MAX_MIP_LEVELS> mip_level_offsets{};

    std::vector<ImageViewInfo> image_view_infos;
    std::vector<ImageViewId> image_view_ids;

    boost::container::small_vector<u32, 16> slice_offsets;
    boost::container::small_vector<SubresourceBase, 16> slice_subresources;

    std::vector<AliasedImage> aliased_images;
    std::vector<ImageId> overlapping_images;
    ImageMapId map_view_id{};
};

struct ImageMapView {
    explicit ImageMapView(GPUVAddr gpu_addr, VAddr cpu_addr, size_t size, ImageId image_id);

    [[nodiscard]] bool Overlaps(VAddr overlap_cpu_addr, size_t overlap_size) const noexcept {
        const VAddr overlap_end = overlap_cpu_addr + overlap_size;
        const VAddr cpu_addr_end = cpu_addr + size;
        return cpu_addr < overlap_end && overlap_cpu_addr < cpu_addr_end;
    }

    [[nodiscard]] bool OverlapsGPU(GPUVAddr overlap_gpu_addr, size_t overlap_size) const noexcept {
        const GPUVAddr overlap_end = overlap_gpu_addr + overlap_size;
        const GPUVAddr gpu_addr_end = gpu_addr + size;
        return gpu_addr < overlap_end && overlap_gpu_addr < gpu_addr_end;
    }

    GPUVAddr gpu_addr;
    VAddr cpu_addr;
    size_t size;
    ImageId image_id;
    bool picked{};
};

struct ImageAllocBase {
    std::vector<ImageId> images;
};

bool AddImageAlias(ImageBase& lhs, ImageBase& rhs, ImageId lhs_id, ImageId rhs_id);

} // namespace VideoCommon
