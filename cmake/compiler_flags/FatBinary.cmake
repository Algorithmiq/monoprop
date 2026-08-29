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
# And ``flatten`` plus ``target_clones`` is not affordable here even so. Flattening pulls the layer
# engine's whole instantiation fan-out into one function body and the clones then multiply it: measured
# on the runtime-width engine, where that fan-out is one TU, four clones took it from 16.7 s to >20 min
# at ~100 GB of compiler memory -- 24.5 GB even with the widest axis collapsed, against 740 MB for the
# same code as an ordinary compile. As separate TUs the multiplication is linear and parallel; inside
# one function it is not. Here the fan-out is wider still, the engine being templated on the mode
# count. See ``docs/content/docs/fat-binary.mdx``.
#
# Why not glibc-hwcaps, which would need no code at all: its subdirectory names are the four psABI
# levels, and the top tier here is ``x86-64-v4`` *plus* ``avx512vpopcntdq``. Installing it as
# ``x86-64-v4`` would hand it to Skylake-X and Cascade Lake, which are v4 and have no vector popcount,
# and they would take SIGILL. The predicate has to be ours.
#
# Variables used::
#
#   monoprop_ENABLE_FAT_BINARY   declared in the top-level CMakeLists.txt
#   monoprop_FAT_MTUNE
#
# Variables defined::
#
#   monoprop_FAT_TIERS               tier ids, baseline first
#   monoprop_ENGINE_OBJ_TARGETS      object libraries holding the engine, one per tier
#   ARCH_FLAG                        overwritten with the baseline tier's flags
#
# Provides::
#
#   monoprop_engine_sources(<sources>...)   add sources to every engine object library

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

# Tier ids, baseline first. An id is the install directory name, the value monoprop.__variant__
# reports and the value monoprop_VARIANT accepts, so it is user-visible and appears in benchmark
# artifacts: renaming one orphans whatever tracked those.
#
# The last two are one instruction set at two vector widths: identical -march, identical
# __builtin_cpu_supports requirements, differing only in -mprefer-vector-width. They are a pair because
# with -mprefer-vector-width unset GCC takes the AVX-512 vector width from the tuning tables, i.e. from
# monoprop_FAT_MTUNE -- a core chosen for its *schedule*, on the reasoning that -mtune never changes
# which instructions come out. For AVX-512 widths it does. So the two tiers pin it and the run time
# picks, using monoprop_FAT_NARROW_VECTOR_CORES below. Their order is load-bearing: the generated table
# is this list reversed, so the 512-bit tier is tried first and a narrow-datapath core falls through to
# the 256-bit one.
set(
  monoprop_FAT_TIERS
  "x86-64-v1"
  "x86-64-v2"
  "x86-64-v3"
  "x86-64-v4-vpopcntdq-vw256"
  "x86-64-v4-vpopcntdq-vw512"
)

# Cores that should be given 256-bit vectors even though they can execute 512-bit ones.
#
# This is a table of core names and not a feature query because there is no feature to query. Whether
# 512-bit vectors are worth using is a property of the *implementation* -- how wide the datapath behind
# the registers really is, and what the core charges in clock frequency for lighting it up -- and CPUID
# reports neither. -mprefer-vector-width is a tuning flag for exactly that reason, and a build that
# fixes it has guessed on the user's behalf; carrying both and choosing at run time is the only way the
# answer can be right on more than one machine.
#
# A core is on the list when it has either of the two reasons to keep 512-bit code out, and off it when
# it has neither:
#
#   a split datapath -- 512-bit operations run on 256-bit hardware, so the width halves the instruction
#   count and buys no throughput. AMD Zen 4 is the case, and the one entry here that is measured rather
#   than reasoned: pinning the two tiers on an EPYC 9R14, the 256-bit tier wins the 127-qubit
#   kicked-Ising model by 1.1% with the two three-sample ranges disjoint (358.9-361.3 ms against
#   363.5-365.7) and ties on the 120-mode Hubbard one (2058.7 against 2063.2, spreads overlapping). GCC
#   tunes znver4 the other way -- -mtune=znver4 resolves to width 512 -- so this is a correction to it.
#
#   a frequency penalty -- the core drops its clock while 512-bit code is in flight. Ice Lake through
#   Rocket Lake pay a measured ~175 MHz of peak, which is what put -mprefer-vector-width=256 in both
#   compilers' Intel tuning to begin with.
#
# Not on the list: Zen 5, which has the full-width datapath Zen 4 lacks, and Sapphire Rapids onwards,
# where the frequency penalty went away (llvm/llvm-project#102047) -- a decision against GCC's own
# -mtune tables, which still say 256 for every Intel AVX-512 core. Neither can be measured here, so
# both are the mechanism argument rather than a number, and both are falsifiable with monoprop_VARIANT.
#
# Also not on the list, and not by choice: the parts the 256-bit default was introduced for in the first
# place. Skylake-SP, Cascade Lake and Cooper Lake are x86-64-v4 with no vector popcount, so they never
# reach this tier at all and their names would be dead entries.
#
# A name GCC does not recognise is dropped with a warning below rather than being an error: the cost is
# that that core gets the 512-bit tier, which is a percent or so, and the alternative is a build that
# fails on an older toolchain over a tuning hint.
set(
  monoprop_FAT_NARROW_VECTOR_CORES
  "znver4"
  "icelake-client"
  "icelake-server"
  "tigerlake"
  "rocketlake"
)

# Resolve a tier id to the flags it compiles with, the CPU features it requires, and anything else its
# selection predicate needs.
#
# The first two are deliberately separate: MARCH_VAR is what the compiler is told, CPU_TOKENS_VAR is what
# the loader checks, and they are not the same list. -march=x86-64-v3 permits the compiler to use every
# v3 instruction, but __builtin_cpu_supports("x86-64-v3") is one query covering the whole level, so the
# token list is shorter than the flag list rather than being derived from it.
function(_monoprop_tier_spec)
  set(
    _one_value_args
    TIER
    MARCH_VAR
    CPU_TOKENS_VAR
    EXTRA_PREDICATE_VAR
  )
  cmake_parse_arguments(PARSE_ARGV 0 _arg "" "${_one_value_args}" "")

  # Anything a __builtin_cpu_supports token cannot say. Empty for every tier but one.
  set(_extra "")

  if(_arg_TIER STREQUAL "x86-64-v1")
    set(_march "-march=x86-64")
    set(_tokens "")
  elseif(_arg_TIER STREQUAL "x86-64-v2")
    set(_march "-march=x86-64-v2")
    set(_tokens "x86-64-v2")
  elseif(_arg_TIER STREQUAL "x86-64-v3")
    set(_march "-march=x86-64-v3")
    set(_tokens "x86-64-v3")
  elseif(_arg_TIER MATCHES "^x86-64-v4-vpopcntdq-vw(256|512)$")
    # v4 alone buys nothing here: ablating the eight AVX-512 extensions one at a time,
    # -mavx512vpopcntdq accounted for the entire v4 -> v4x gain and the other seven for exactly zero.
    # This codebase is std::popcount word loops and holds no intrinsics, so a vector popcount is the
    # only extension it has anything to bite on.
    #
    # The width is spelled out rather than left to -mtune, and that is the point of the pair:
    # -mtune=skylake and -mtune=znver4 resolve to 512, -mtune=icelake-server and -mtune=sapphirerapids
    # to 256. See monoprop_FAT_TIERS above.
    set(_width "${CMAKE_MATCH_1}")
    set(
      _march
      "-march=x86-64-v4"
      "-mavx512vpopcntdq"
      "-mprefer-vector-width=${_width}"
    )
    set(
      _tokens
      "x86-64-v4"
      "avx512vpopcntdq"
    )
    if(_width STREQUAL "512")
      # The only tier with a predicate term that is not a feature bit, and the reason the 512 tier is
      # tried first: a narrow-datapath core fails this and falls through to the 256 tier, which asks for
      # nothing but the instructions.
      set(_extra "!monoprop::detail::prefers_narrow_vectors()")
    endif()
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
  if(_arg_EXTRA_PREDICATE_VAR)
    set(${_arg_EXTRA_PREDICATE_VAR} "${_extra}" PARENT_SCOPE)
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

  macro(monoprop_engine_sources)
    target_sources(monoprop-objs PRIVATE ${ARGN})
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

# Which of the core names in monoprop_FAT_NARROW_VECTOR_CORES this compiler actually knows. Checked
# rather than assumed: __builtin_cpu_is rejects a name its libgcc has no model number for, and the name
# set grows with every GCC release, so a hard-coded list is a build failure waiting for the oldest
# toolchain in the matrix. A dropped name means that core takes the 512-bit tier instead of the 256-bit
# one -- a tuning difference, never a fault, since the two tiers require identical CPU features.
include(CheckCXXSourceCompiles)
set(_monoprop_narrow_cores "")
foreach(_core IN LISTS monoprop_FAT_NARROW_VECTOR_CORES)
  string(MAKE_C_IDENTIFIER "monoprop_HAVE_CPU_IS_${_core}" _monoprop_have_core)
  check_cxx_source_compiles(
    "int main() { return __builtin_cpu_is(\"${_core}\"); }"
    ${_monoprop_have_core}
  )
  if(${_monoprop_have_core})
    list(APPEND _monoprop_narrow_cores "${_core}")
  else()
    message(
      WARNING
      "${CMAKE_CXX_COMPILER_ID} does not know the core name '${_core}', so __builtin_cpu_is cannot test for it. That CPU will be given the 512-bit vector tier instead of the 256-bit one; both require the same instructions, so this costs tuning and not correctness."
    )
  endif()
endforeach()

set(_monoprop_narrow_core_list "")
foreach(_core IN LISTS _monoprop_narrow_cores)
  string(
    APPEND _monoprop_narrow_core_list
    "    X(\"${_core}\")                                                                           \\\n"
  )
endforeach()

# The loader's variant table, best ISA first. Generated rather than hand-written in isa.cpp so that the
# tier list above is the only place a tier is declared: a tier that is built but never selected, or
# selected but never built, is a silent loss of the whole feature.
set(_monoprop_variant_table_body "")
list(REVERSE monoprop_FAT_TIERS)
foreach(_tier IN LISTS monoprop_FAT_TIERS)
  _monoprop_tier_spec(
    TIER "${_tier}"
    CPU_TOKENS_VAR _tokens
    EXTRA_PREDICATE_VAR _extra_predicate
  )
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
  # Two predicates, and the distinction is load-bearing: the first is whether this CPU *can execute*
  # the tier, which is what gates a monoprop_VARIANT pin, and the second is whether it should be *given*
  # the tier, which is what the automatic selection reads. They differ for exactly one tier -- the
  # 512-bit one, which any AVX-512 CPU can run and only some should have -- and conflating them makes
  # that tier unpinnable on the machines somebody would want to compare it on.
  set(_preferred "${_predicate}")
  if(NOT _extra_predicate STREQUAL "")
    set(_preferred "${_predicate} && ${_extra_predicate}")
  endif()
  string(
    APPEND _monoprop_variant_table_body
    "    X(\"${_tier}\", ${_predicate}, ${_preferred})                                   \\\n"
  )
endforeach()
list(REVERSE monoprop_FAT_TIERS)

file(
  WRITE "${PROJECT_BINARY_DIR}/include/monoprop/FatVariants.h"
  "// Generated by cmake/compiler_flags/FatBinary.cmake -- do not edit.
#pragma once

/// Cores that should be given the 256-bit-preferring variant, as X(core name). May be empty.
#define monoprop_FAT_NARROW_VECTOR_CORES(X)                                                        \\
${_monoprop_narrow_core_list}    /* end */

namespace monoprop::detail {
/// Whether this CPU should be handed 256-bit vectors in preference to 512-bit ones.
///
/// Not a capability question -- both variants that ask it require exactly the same instructions -- but
/// an implementation one: how wide the datapath behind the registers is, and what the core charges in
/// clock frequency for using all of it. CPUID reports neither, so this is a name table, and a core
/// nobody listed gets the wider variant. See monoprop_FAT_NARROW_VECTOR_CORES in FatBinary.cmake.
inline auto prefers_narrow_vectors() -> bool {
#if defined(__x86_64__) || defined(_M_X64)
    // Required before any other __builtin_cpu_* call in a translation unit that may run before libgcc's
    // own constructor has. Idempotent, so callers that have already probed are not a problem.
    __builtin_cpu_init();
#define monoprop_FAT_NARROW_CORE(core) \\
    if (__builtin_cpu_is(core)) {      \\
        return true;                   \\
    }
    monoprop_FAT_NARROW_VECTOR_CORES(monoprop_FAT_NARROW_CORE)
#undef monoprop_FAT_NARROW_CORE
#endif
    return false;
}
} // namespace monoprop::detail

/// Every shipped ISA variant, best first, as X(id, runnable, preferred).
///
/// The second holds when the CPU can execute the variant at all, the third when it should be given the
/// variant without being asked. They differ only for the 512-bit vector variant, and only because
/// vector width is a tuning question rather than a capability one -- see prefers_narrow_vectors().
///
/// Best-first is the selection order, so the table's order is load-bearing and not cosmetic.
#define monoprop_FAT_VARIANT_TABLE(X)                                                              \\
${_monoprop_variant_table_body}    /* end */
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

# One engine object library per tier, and one Variants.h per tier so each can say which it is. The
# baseline tier reuses monoprop-objs rather than adding a target, so N tiers cost N compiles and not
# N+1.
set(monoprop_ENGINE_OBJ_TARGETS "")
foreach(_tier IN LISTS monoprop_FAT_TIERS)
  _monoprop_tier_spec(TIER "${_tier}" MARCH_VAR _tier_flags)
  _monoprop_generate_variant_header(
    VARIANT_ID "${_tier}"
    ARCH_FLAGS ${_tier_flags}
    OUTPUT_DIR "${PROJECT_BINARY_DIR}/variants/${_tier}/include"
  )

  if(_tier STREQUAL monoprop_FAT_BASELINE_TIER)
    list(APPEND monoprop_ENGINE_OBJ_TARGETS "monoprop-objs")
  else()
    _monoprop_tier_slug("${_tier}" _slug)
    add_library(monoprop-objs-${_slug} OBJECT "")
    list(APPEND monoprop_ENGINE_OBJ_TARGETS "monoprop-objs-${_slug}")
  endif()
endforeach()

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

# Fan a source list out over every tier. A macro and not a function, so that relative source paths
# still resolve against the CMakeLists.txt that named them.
macro(monoprop_engine_sources)
  foreach(_engine_target IN LISTS monoprop_ENGINE_OBJ_TARGETS)
    target_sources(${_engine_target} PRIVATE ${ARGN})
  endforeach()
endmacro()
