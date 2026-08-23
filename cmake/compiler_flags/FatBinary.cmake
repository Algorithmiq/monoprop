#.rst:
#
# The fat binary: one copy of the propagation engine per x86-64 ISA tier, all four in the same wheel,
# one of them selected when ``monoprop`` is imported.
#
# Why the ISA is bound on the command line and not by an attribute. Two things, both measured.
#
# An attribute cannot widen code defined outside it. ``target("arch=x86-64-v4,avx512vpopcntdq")`` is
# perfectly valid -- a comma separates options in a plain ``target``, unlike ``target_clones`` where it
# separates clones -- but ``ix86_can_inline_p`` refuses to inline across an ``arch`` mismatch, so a
# targeted wrapper around this engine compiles to a ``jmp``. ``flatten`` does not override that, and
# ``#pragma GCC target`` does not capture a template that was defined outside its region. Header-resident
# code is widened by its TU's command line or not at all.
#
# And ``flatten`` plus ``target_clones`` is not affordable here even so: ``build_layer`` fans out over
# ``with_algebra`` x ``with_store`` x ``with_kernel_width``, and four flattened clones of that took one
# TU from 16.7 s to >20 min at ~100 GB of compiler memory -- 24.5 GB even with the width axis collapsed,
# against 740 MB for the same code as an ordinary compile. As separate TUs the multiplication is linear
# and parallel; inside one function it is not. See ``docs/content/docs/fat-binary.mdx``.
#
# Why not glibc-hwcaps, which would need no code at all: its subdirectory names are the four psABI
# levels, and the top tier here is ``x86-64-v4`` *plus* ``avx512vpopcntdq``. Installing it as
# ``x86-64-v4`` would hand it to Skylake-X and Cascade Lake, which are v4 and have no vector popcount,
# and they would take SIGILL. The predicate has to be ours.
#
# Two shapes, chosen by monoprop_FAT_BINARY_MODE, differing only in how much gets duplicated.
#
# ``whole-library`` compiles every engine source once per tier and ships one extension module per tier,
# picked in Python before any engine code loads (src/monoprop/_bootstrap.py). Simple, and nothing in the
# wheel can be at the wrong ISA by construction.
#
# ``narrow-seam`` compiles one translation unit per tier -- the one holding build_layer's instantiation
# tree -- and everything else once, at the baseline; a single module carries all the tiers and picks one
# on first use inside the library (cpp/monoprop/detail/evolution/TierDispatch.cpp). Smaller (1.9 MB of
# payload against 5.0 MB) and quicker to build (84 s of engine compile against 197 s), and it does not
# work: see the message() below. Kept because the measurement it produced is worth keeping and because
# the two fixes for it are real -- per-tier shared objects, or a per-tier namespace around the algorithm
# headers -- and both start from this code.
#
# Variables used::
#
#   monoprop_ENABLE_FAT_BINARY   declared in the top-level CMakeLists.txt
#   monoprop_FAT_BINARY_MODE
#   monoprop_FAT_MTUNE
#
# Variables defined::
#
#   monoprop_FAT_TIERS               tier ids, baseline first
#   monoprop_ENGINE_OBJ_TARGETS      object libraries holding the shared engine sources
#   monoprop_TIERED_OBJ_TARGETS      object libraries holding the tiered sources, one per tier
#   monoprop_FAT_NARROW_SEAM         TRUE in narrow-seam mode
#   ARCH_FLAG                        overwritten with the baseline tier's flags
#
# Provides::
#
#   monoprop_engine_sources(<sources>...)          sources compiled once per engine object library
#   monoprop_tiered_engine_sources(<sources>...)   sources compiled once per ISA tier

# What every tier tunes for. Not part of the ISA: -mtune never widens the instruction set, it only
# changes the cost model and the schedule, so it is free to name a core no tier requires. skylake is
# the measured choice -- it reaches the top cluster on every vectorization counter at the smallest
# .text, and is the oldest core in that cluster, so it is the least likely to schedule badly across the
# 2015-onwards range. There is no vendor-neutral alternative: GCC rejects -mtune=x86-64-v3.
set(
  monoprop_FAT_MTUNE
  "skylake"
  CACHE STRING
  "-mtune value applied to every fat-binary tier (scheduling only; never widens the ISA)"
)

# How much of the library each tier gets its own copy of. See the two paragraphs at the top of this
# file; the trade is wheel size and build time against a dispatch boundary inside the engine.
set(
  monoprop_FAT_BINARY_MODE
  "whole-library"
  CACHE STRING
  "Fat-binary shape: whole-library (a module per tier) or narrow-seam (one tiered TU in one module)"
)
set_property(
  CACHE
    monoprop_FAT_BINARY_MODE
  PROPERTY
    STRINGS
      "whole-library"
      "narrow-seam"
)

if(NOT monoprop_FAT_BINARY_MODE MATCHES "^(whole-library|narrow-seam)$")
  message(
    FATAL_ERROR
    "monoprop_FAT_BINARY_MODE must be 'whole-library' or 'narrow-seam', got '${monoprop_FAT_BINARY_MODE}'"
  )
endif()

# Tier ids, baseline first. An id is the install directory name, the value monoprop.__variant__
# reports and the value monoprop_VARIANT accepts, so it is user-visible and appears in benchmark
# artifacts: renaming one orphans whatever tracked those.
set(
  monoprop_FAT_TIERS
  "x86-64-v1"
  "x86-64-v2"
  "x86-64-v3"
  "x86-64-v4-vpopcntdq"
)

# Resolve a tier id to the flags it compiles with and the CPU features it requires.
#
# The two are deliberately separate: MARCH_VAR is what the compiler is told, CPU_TOKENS_VAR is what the
# loader checks, and they are not the same list. -march=x86-64-v3 permits the compiler to use every v3
# instruction, but __builtin_cpu_supports("x86-64-v3") is one query covering the whole level, so the
# token list is shorter than the flag list rather than being derived from it.
function(_monoprop_tier_spec)
  set(
    _one_value_args
    TIER
    MARCH_VAR
    CPU_TOKENS_VAR
  )
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "${_one_value_args}" "")

  if(_arg_TIER STREQUAL "x86-64-v1")
    set(_march "-march=x86-64")
    set(_tokens "")
  elseif(_arg_TIER STREQUAL "x86-64-v2")
    set(_march "-march=x86-64-v2")
    set(_tokens "x86-64-v2")
  elseif(_arg_TIER STREQUAL "x86-64-v3")
    set(_march "-march=x86-64-v3")
    set(_tokens "x86-64-v3")
  elseif(_arg_TIER STREQUAL "x86-64-v4-vpopcntdq")
    # v4 alone buys nothing here: ablating the eight AVX-512 extensions one at a time,
    # -mavx512vpopcntdq accounted for the entire v4 -> v4x gain and the other seven for exactly zero.
    # This codebase is std::popcount word loops and holds no intrinsics, so a vector popcount is the
    # only extension it has anything to bite on.
    set(
      _march
      "-march=x86-64-v4"
      "-mavx512vpopcntdq"
    )
    set(
      _tokens
      "x86-64-v4"
      "avx512vpopcntdq"
    )
  else()
    message(FATAL_ERROR "_monoprop_tier_spec: unknown tier '${_arg_TIER}'")
  endif()

  list(APPEND _march "-mtune=${monoprop_FAT_MTUNE}")

  # Reproducibility across tiers, and the reason it needs saying: without this, -march=x86-64-v3 and up
  # contract a*b+c into FMA3, which changes the rounding of the coefficient accumulation. Measured: all
  # evolved terms stay bit-identical but the energy moves by 1-2 ULP from v3 up, and an arm built with
  # -ffp-contract=off came back byte-identical to the baseline. In a fat binary that would make a
  # result depend on which CPU the user happens to run on, which is not something a propagation library
  # gets to do -- and it would end `just diff-baseline` as a byte-wise gate. The project-wide flag in
  # GNU.CXX.cmake already covers this; it is repeated here so a tier cannot lose it by reordering.
  list(APPEND _march "-ffp-contract=off")

  if(_arg_MARCH_VAR)
    set(${_arg_MARCH_VAR} "${_march}" PARENT_SCOPE)
  endif()
  if(_arg_CPU_TOKENS_VAR)
    set(${_arg_CPU_TOKENS_VAR} "${_tokens}" PARENT_SCOPE)
  endif()
endfunction()

# Sanitize a tier id into something usable as a CMake target-name suffix.
function(_monoprop_tier_slug tier output_variable)
  string(
    REPLACE "-"
    "_"
    _slug
    "${tier}"
  )
  set(${output_variable} "${_slug}" PARENT_SCOPE)
endfunction()

if(NOT monoprop_ENABLE_FAT_BINARY)
  set(monoprop_ENGINE_OBJ_TARGETS "monoprop-objs")
  set(monoprop_TIERED_OBJ_TARGETS "")
  set(monoprop_FAT_NARROW_SEAM FALSE)

  macro(monoprop_engine_sources)
    target_sources(monoprop-objs PRIVATE ${ARGN})
  endmacro()

  # A single-ISA build has one tier, so "tiered" and "shared" are the same thing. The seam still exists
  # in the source -- TierDispatch.cpp resolves it to a direct call -- so that the two build shapes do
  # not diverge in what they compile.
  macro(monoprop_tiered_engine_sources)
    monoprop_engine_sources(${ARGN})
  endmacro()

  return()
endif()

# ---------------------------------------------------------------------------------------------------
# From here on the fat binary is enabled.
# ---------------------------------------------------------------------------------------------------

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  message(
    FATAL_ERROR
    "monoprop_ENABLE_FAT_BINARY is x86-64 only (CMAKE_SYSTEM_PROCESSOR is '${CMAKE_SYSTEM_PROCESSOR}'). Turn it OFF; a single-ISA build is the correct shape on this target."
  )
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  message(
    FATAL_ERROR
    "monoprop_ENABLE_FAT_BINARY needs the -march/-mtune spelling and __builtin_cpu_supports, i.e. GCC or Clang. Detected '${CMAKE_CXX_COMPILER_ID}'."
  )
endif()

# The loader's variant table, best ISA first. Generated rather than hand-written in isa.cpp so that the
# tier list above is the only place a tier is declared: a tier that is built but never selected, or
# selected but never built, is a silent loss of the whole feature.
set(_monoprop_variant_table_body "")
set(_monoprop_variant_names "")
set(_monoprop_variant_tiers "")
list(REVERSE monoprop_FAT_TIERS)
foreach(_tier IN LISTS monoprop_FAT_TIERS)
  _monoprop_tier_spec(TIER "${_tier}" CPU_TOKENS_VAR _tokens)
  _monoprop_tier_slug("${_tier}" _slug)
  if(_tokens STREQUAL "")
    set(_predicate "true")
  else()
    set(_predicate "")
    foreach(_token IN LISTS _tokens)
      if(NOT _predicate STREQUAL "")
        string(APPEND _predicate " && ")
      endif()
      string(APPEND _predicate "__builtin_cpu_supports(\"${_token}\")")
    endforeach()
  endif()
  string(
    APPEND _monoprop_variant_table_body
    "    X(\"${_tier}\", ${_predicate})                                       \\\n"
  )
  string(
    APPEND _monoprop_variant_names
    "    X(\"${_tier}\")                  \\\n"
  )
  string(
    APPEND _monoprop_variant_tiers
    "    X(\"${_tier}\", ${_slug}, ${_predicate})                             \\\n"
  )
endforeach()
list(REVERSE monoprop_FAT_TIERS)

file(
  WRITE "${PROJECT_BINARY_DIR}/include/monoprop/FatVariants.h"
  "// Generated by cmake/compiler_flags/FatBinary.cmake -- do not edit.
#pragma once

/// Every shipped ISA variant, best first, as X(id, predicate) where the predicate holds exactly when
/// the running CPU can execute that variant. Best-first is the selection order, so the table's order
/// is load-bearing and not cosmetic.
#define monoprop_FAT_VARIANT_TABLE(X)                                                              \\
${_monoprop_variant_table_body}    /* end */

/// Every shipped ISA variant, best first, as X(id) -- the same list without the predicates.
#define monoprop_FAT_VARIANT_NAMES(X)                                                              \\
${_monoprop_variant_names}    /* end */

/// Every shipped ISA variant, best first, as X(id, slug, predicate). The slug is the C++ identifier the
/// tiered translation unit's namespace is named with, so it is how narrow-seam mode reaches one tier's
/// copy of the kernel; it is the tier id with '-' replaced by '_'.
#define monoprop_FAT_VARIANT_TIERS(X)                                                              \\
${_monoprop_variant_tiers}    /* end */
"
)

# The baseline tier is not a fifth build: it *is* monoprop-objs, and so also what libmonoprop.so, the
# C++ unit tests and any C++ consumer of the installed package get. That is deliberate -- a wheel's
# tiers are chosen for the machines that run it, while a source build has -march=native available and
# does not need any of this -- and it means CI's C++ tests exercise the portable floor.
list(GET monoprop_FAT_TIERS 0 monoprop_FAT_BASELINE_TIER)
_monoprop_tier_spec(TIER "${monoprop_FAT_BASELINE_TIER}" MARCH_VAR ARCH_FLAG)

# The baseline is also the *floor*, and it has to be global rather than per target. monoprop-objs is
# not the only source of code in a wheel: nanobind's static library, and anything else a dependency
# adds as its own target, is compiled with whatever -march the toolchain defaults to -- and that
# default is not the psABI baseline. GCC as shipped by Ubuntu here is configured
# --with-arch-64=x86-64-v3, so without this floor the v1 and v2 variants would carry AVX2 in their
# nanobind glue, and _isa -- the probe that exists precisely to keep us off machines that cannot run
# a variant -- would itself fault on those machines.
#
# Putting it in CMAKE_CXX_FLAGS rather than on targets is what makes it a floor: it is emitted before
# every target's own options, so a tier's wider -march still wins for that tier's objects, while
# everything nobody widened stays at the baseline.
string(
  REPLACE ";"
  " "
  _monoprop_baseline_flags
  "${ARCH_FLAG}"
)
string(APPEND CMAKE_CXX_FLAGS " ${_monoprop_baseline_flags}")

if(monoprop_FAT_BINARY_MODE STREQUAL "narrow-seam")
  set(monoprop_FAT_NARROW_SEAM TRUE)
  # Not shippable as it stands, and the reason is not something a test notices. Almost everything the
  # tiered TU emits is a template instantiation -- a weak COMDAT whose mangled name carries no tier --
  # so the linker keeps one definition per name and the four tiers collapse into whichever the link
  # line lists first. Measured: the pinned tier makes no difference to run time, and reversing the link
  # order makes every pin run at the top tier's speed. Everything else still passes, because the tiers
  # do agree on the answers; they just stop differing in the instructions.
  #
  # tools/check-tier-symbols.py is the gate on that, and it currently fails by design. See the
  # narrow-seam section of docs/content/docs/fat-binary.mdx for the numbers and the two ways out.
  message(
    WARNING
    "monoprop_FAT_BINARY_MODE=narrow-seam is an experiment, not a shippable configuration: COMDAT deduplication collapses the ISA tiers at link time, so the run-time dispatch has nothing to choose between. Run tools/check-tier-symbols.py on this build tree. Use whole-library for anything that ships."
  )
else()
  set(monoprop_FAT_NARROW_SEAM FALSE)
endif()

# One Variants.h per tier so each can say which it is, and the object libraries the tiers compile into.
#
# whole-library: one engine object library per tier, holding every engine source. The baseline reuses
# monoprop-objs rather than adding a target, so N tiers cost N compiles and not N+1.
#
# narrow-seam: monoprop-objs alone holds the shared sources, and each tier gets a small object library
# holding only the tiered ones. The baseline gets one too, even though its flags are monoprop-objs' own
# flags: the tiered TU needs a per-target monoprop_TIER_SLUG, and putting the baseline's copy in
# monoprop-objs would mean defining that on a source file instead of a target -- which is how the same
# source ends up compiled twice under two slugs the day a second tiered TU is added.
set(monoprop_ENGINE_OBJ_TARGETS "")
set(monoprop_TIERED_OBJ_TARGETS "")
foreach(_tier IN LISTS monoprop_FAT_TIERS)
  _monoprop_tier_spec(TIER "${_tier}" MARCH_VAR _tier_flags)
  _monoprop_generate_variant_header(
    VARIANT_ID "${_tier}"
    FLAGS ${_tier_flags}
    OUTPUT_DIR "${PROJECT_BINARY_DIR}/variants/${_tier}/include"
  )
  _monoprop_tier_slug("${_tier}" _slug)

  if(monoprop_FAT_NARROW_SEAM)
    add_library(monoprop-tier-${_slug} OBJECT "")
    list(APPEND monoprop_TIERED_OBJ_TARGETS "monoprop-tier-${_slug}")
  elseif(_tier STREQUAL monoprop_FAT_BASELINE_TIER)
    list(APPEND monoprop_ENGINE_OBJ_TARGETS "monoprop-objs")
  else()
    add_library(monoprop-objs-${_slug} OBJECT "")
    list(APPEND monoprop_ENGINE_OBJ_TARGETS "monoprop-objs-${_slug}")
  endif()
endforeach()

if(monoprop_FAT_NARROW_SEAM)
  set(monoprop_ENGINE_OBJ_TARGETS "monoprop-objs")
endif()

# The engine object library that carries a given tier, and the include directory holding that tier's
# Variants.h. Both are derived, so callers never spell a target name or a path themselves.
#
# Usage:
#   monoprop_tier_targets(TIER <id> OBJS_VAR <var> VARIANT_INCLUDE_DIR_VAR <var>)
function(monoprop_tier_targets)
  set(
    _one_value_args
    TIER
    OBJS_VAR
    VARIANT_INCLUDE_DIR_VAR
  )
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "${_one_value_args}" "")

  if(monoprop_FAT_NARROW_SEAM)
    message(
      FATAL_ERROR
      "monoprop_tier_targets is whole-library only: narrow-seam mode has one engine object library and one binding module, not one per tier. Use monoprop_tiered_targets."
    )
  endif()

  if(_arg_TIER STREQUAL monoprop_FAT_BASELINE_TIER)
    set(_objs "monoprop-objs")
  else()
    _monoprop_tier_slug("${_arg_TIER}" _slug)
    set(_objs "monoprop-objs-${_slug}")
  endif()

  if(_arg_OBJS_VAR)
    set(${_arg_OBJS_VAR} "${_objs}" PARENT_SCOPE)
  endif()
  if(_arg_VARIANT_INCLUDE_DIR_VAR)
    set(
      ${_arg_VARIANT_INCLUDE_DIR_VAR}
      "${PROJECT_BINARY_DIR}/variants/${_arg_TIER}/include"
      PARENT_SCOPE
    )
  endif()
endfunction()

# Fan a source list out over every engine object library. A macro and not a function, so that relative
# source paths still resolve against the CMakeLists.txt that named them.
macro(monoprop_engine_sources)
  foreach(_engine_target IN LISTS monoprop_ENGINE_OBJ_TARGETS)
    target_sources(${_engine_target} PRIVATE ${ARGN})
  endforeach()
endmacro()

# Sources that get one copy per ISA tier. In whole-library mode that is every engine object library, so
# this is monoprop_engine_sources; in narrow-seam mode it is the per-tier libraries and *not*
# monoprop-objs, which is what keeps the shared build at one copy.
#
# A source belongs here when the ISA is what the code is for -- i.e. it holds a hot instantiation tree.
# Adding one is not free: it is compiled once per tier, so it multiplies both the wheel and the build.
macro(monoprop_tiered_engine_sources)
  if(monoprop_FAT_NARROW_SEAM)
    foreach(_tiered_target IN LISTS monoprop_TIERED_OBJ_TARGETS)
      target_sources(${_tiered_target} PRIVATE ${ARGN})
    endforeach()
  else()
    monoprop_engine_sources(${ARGN})
  endif()
endmacro()

# The object libraries holding one tier's copy of the tiered sources, and the tier id each was built
# for, as two parallel lists. Only narrow-seam mode has any.
#
# Usage:
#   monoprop_tiered_targets(TARGETS_VAR <var> TIERS_VAR <var>)
function(monoprop_tiered_targets)
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "TARGETS_VAR;TIERS_VAR" "")
  if(_arg_TARGETS_VAR)
    set(${_arg_TARGETS_VAR} "${monoprop_TIERED_OBJ_TARGETS}" PARENT_SCOPE)
  endif()
  if(_arg_TIERS_VAR)
    if(monoprop_FAT_NARROW_SEAM)
      set(${_arg_TIERS_VAR} "${monoprop_FAT_TIERS}" PARENT_SCOPE)
    else()
      set(${_arg_TIERS_VAR} "" PARENT_SCOPE)
    endif()
  endif()
endfunction()
