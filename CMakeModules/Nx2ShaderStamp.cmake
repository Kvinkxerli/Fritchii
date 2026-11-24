# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_policy(SET CMP0012 NEW)
if(POLICY CMP0130)
    cmake_policy(SET CMP0130 NEW)
endif()

# Build-time script (cmake -P): hashes the sources that decide what the GPU driver compiles for a
# pipeline (the shader recompiler, pipeline creation, the precompile step) into NX2_SHADER_STAMP.
# "Preparing shaders" stamps its work with it instead of the release version (2026-11-10: every
# release re-ran the preparation for every player, also releases that changed no shader). The
# header is only rewritten when the hash changes, so nothing rebuilds otherwise.
# In: INPUT_LIST (a file listing the sources, one per line), OUT (the header to write).
#
# A NEW STAMP SENDS EVERY PLAYER THROUGH THE PREPARATION AGAIN. A comment or a renamed call in one
# of these files is enough, so:
# - Code that can't change a compiled pipeline stays out of these files. Where a class in them has
#   to gain such a member, it goes between lines holding FRITCHII_STAMP_SKIP_BEGIN and
#   FRITCHII_STAMP_SKIP_END, whole lines, only in the headers named in SkipBlocksIn below. Those
#   lines are hashed as if they weren't there.
# - KnownStamps is what released builds carry: the build stops on any other value. If shaders
#   really changed, add the new stamps there in the same commit (a CRLF checkout, as on Windows,
#   and an LF one hash differently). For a test build, set FRITCHII_SHADER_STAMP_ANY=1.
# - SameAsReleased maps a hash to a released stamp when the changed files provably prepare the
#   same pipelines, byte for byte, as that release (each entry says why). The map holds exact
#   hashes: any later edit to these files leaves it, and the build stops again until someone
#   checks that edit too.

set(SkipBlocksIn "vk_graphics_pipeline.h;vk_compute_pipeline.h")
# v6.1, kept by 6.2: Windows (CRLF checkout), Linux (LF checkout).
set(KnownStamps "143b6220486fb09b;3dc8b3ea00f9b392")
# 6.2, the occlusion flag extensions (FRITCHII_OCCL_FLAG=9, "Full"): emit_spirv.*,
# emit_spirv_special.cpp, spirv_emit_context.*, runtime_info.h and vk_graphics_pipeline.cpp
# changed. Every new SPIR-V path runs only for a module translated with
# RuntimeInfo::occl_flag_store or occl_flag_store_at_end (EmitContext::UsesOcclFlagStore), which
# only PipelineCache::FindOrMakeOcclFlag sets: a variant made during play, owned by its parent,
# never stored in the disk cache (CreateGraphicsPipeline(nullptr, false)), so never prepared as a
# pipeline of its own. (Later, FRITCHII_PREP_OCCL_VARIANTS=1, off by default, makes "Preparing
# shaders" build such variants too, kept only in the kept file beside their parents: its stamp
# then names them with OcclVariantTranslationVersion in vk_pipeline_cache.cpp, which is not
# hashed here and is bumped by hand with any change to their translation, and the boot sample
# translates some of them again. The prepared stamp and the parents are unchanged.)
# Any other module's SPIR-V is byte-identical (the two other edits: a static_vector's capacity
# and "force_early_z || UsesOcclFlagStore()"). vk_graphics_pipeline.cpp gained a draw-time copy
# of the push constants (ConfigureImpl) and ConfigureAfterOcclPrepass: no create info changed.
# So the prepared pipelines are 6.1's, with Full on or off.
set(SameAsReleased "096d887be421a8e2=143b6220486fb09b;6f72127d45416a88=3dc8b3ea00f9b392")

# The SHA-256 of a file as if its FRITCHII_STAMP_SKIP blocks weren't there, line endings as in the
# file. file(READ) gives LF line endings on every platform, so CRLF goes back on before hashing.
function(hash_without_skip_blocks path out)
    file(READ "${path}" content)
    file(READ "${path}" head LIMIT 256 HEX)
    string(FIND "${head}" "0d0a" crlf)
    while(TRUE)
        string(FIND "${content}" "FRITCHII_STAMP_SKIP_BEGIN" begin)
        if(begin EQUAL -1)
            break()
        endif()
        string(SUBSTRING "${content}" 0 ${begin} before)
        string(FIND "${before}" "\n" line_start REVERSE)
        math(EXPR line_start "${line_start} + 1")
        string(SUBSTRING "${content}" 0 ${line_start} before)
        string(SUBSTRING "${content}" ${begin} -1 rest)
        string(FIND "${rest}" "FRITCHII_STAMP_SKIP_END" end)
        if(end EQUAL -1)
            message(FATAL_ERROR "${path}: FRITCHII_STAMP_SKIP_BEGIN without an END")
        endif()
        string(SUBSTRING "${rest}" ${end} -1 rest)
        string(FIND "${rest}" "\n" line_end)
        math(EXPR line_end "${line_end} + 1")
        string(SUBSTRING "${rest}" ${line_end} -1 rest)
        set(content "${before}${rest}")
    endwhile()
    if(NOT crlf EQUAL -1)
        string(REPLACE "\n" "\r\n" content "${content}")
    endif()
    string(SHA256 hash "${content}")
    set(${out} "${hash}" PARENT_SCOPE)
endfunction()

file(STRINGS "${INPUT_LIST}" inputs)
list(SORT inputs)
set(all "")
foreach(path IN LISTS inputs)
    if (EXISTS "${path}")
        get_filename_component(name "${path}" NAME)
        list(FIND SkipBlocksIn "${name}" skip_index)
        if(skip_index EQUAL -1)
            file(SHA256 "${path}" hash)
        else()
            hash_without_skip_blocks("${path}" hash)
        endif()
        string(APPEND all "${name}:${hash};")
    endif()
endforeach()
string(SHA256 stamp "${all}")
string(SUBSTRING "${stamp}" 0 16 stamp)
foreach(pair IN LISTS SameAsReleased)
    string(REPLACE "=" ";" pair "${pair}")
    list(GET pair 0 hashed)
    list(GET pair 1 released)
    if(stamp STREQUAL hashed)
        set(stamp "${released}")
        break()
    endif()
endforeach()
list(FIND KnownStamps "${stamp}" known_index)
if(known_index EQUAL -1 AND NOT "$ENV{FRITCHII_SHADER_STAMP_ANY}")
    message(FATAL_ERROR
        "The Preparing shaders stamp changed to ${stamp}: every player would prepare all shaders "
        "again. If a change in src/shader_recompiler, vk_graphics_pipeline.*, vk_compute_pipeline.*, "
        "fixed_pipeline_state.* or nx2_precompile.* can't change a compiled pipeline, undo it or "
        "move it out (CMakeModules/Nx2ShaderStamp.cmake says how). If shaders really changed, add "
        "the new stamps to KnownStamps; if they provably did not, to SameAsReleased with the "
        "reason. For a test build: FRITCHII_SHADER_STAMP_ANY=1.")
endif()
set(content "// Generated by CMakeModules/Nx2ShaderStamp.cmake: do not edit.\n#pragma once\n#define NX2_SHADER_STAMP \"${stamp}\"\n")
set(old "")
if (EXISTS "${OUT}")
    file(READ "${OUT}" old)
endif()
if (NOT old STREQUAL content)
    file(WRITE "${OUT}" "${content}")
endif()
