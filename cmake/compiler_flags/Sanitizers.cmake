set(
  monoprop_SANITIZER
  "none"
  CACHE STRING
  "Sanitizer profile: none, asan-ubsan, or tsan"
)
set_property(
  CACHE
    monoprop_SANITIZER
  PROPERTY
    STRINGS
      "none"
      "asan-ubsan"
      "tsan"
)

add_library(monoprop-sanitizers INTERFACE)

if(monoprop_SANITIZER STREQUAL "none")
  return()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "monoprop_SANITIZER requires Linux")
endif()

# CI deliberately runs this profile under GCC: the manylinux wheels and the gcov-based
# coverage job are GCC-built, so sanitizing with GCC exercises the codegen that actually
# ships. Clang is supported as an escape hatch -- it has -fsanitize-ignorelist=, which can
# exempt nanobind's casters by name, where GCC only offers a translation-unit-wide
# -fno-sanitize= flag.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(
    FATAL_ERROR
    "monoprop_SANITIZER requires a GNU or Clang compiler, got ${CMAKE_CXX_COMPILER_ID}"
  )
endif()

set(
  _monoprop_sanitizer_common_flags
  -O1
  -g3
  -fno-omit-frame-pointer
  -fno-optimize-sibling-calls
  # identical-code folding merges distinct functions, so a sanitizer backtrace can name the
  # wrong one; no Clang equivalent is needed because Clang does not fold at -O1
  $<$<CXX_COMPILER_ID:GNU>:-fno-ipa-icf>
  -fno-sanitize-recover=all
)

if(monoprop_SANITIZER STREQUAL "asan-ubsan")
  # Checks outside the `undefined` group are named explicitly. The array-bounds spelling
  # differs: GCC's `bounds-strict` also instruments trailing arrays in structs and is not
  # implied by `undefined`; Clang has no such value and uses `bounds`.
  #
  # pointer-compare/pointer-subtract only emit the instrumentation -- whether it reports is a
  # per-leg runtime choice via ASAN_OPTIONS=detect_invalid_pointer_pairs, because the Python
  # leg loads an uninstrumented CPython and would false-positive on ordinary comparisons.
  if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    set(_monoprop_sanitizer_bounds "bounds-strict")
  else()
    set(_monoprop_sanitizer_bounds "bounds")
  endif()
  set(
    _monoprop_sanitizer_runtime_flags
    "-fsanitize=address,undefined,${_monoprop_sanitizer_bounds},float-cast-overflow,pointer-compare,pointer-subtract"
  )
elseif(monoprop_SANITIZER STREQUAL "tsan")
  set(_monoprop_sanitizer_runtime_flags -fsanitize=thread)
else()
  message(FATAL_ERROR "Unknown monoprop_SANITIZER value: ${monoprop_SANITIZER}")
endif()

target_compile_options(
  monoprop-sanitizers
  INTERFACE
    ${_monoprop_sanitizer_common_flags}
    ${_monoprop_sanitizer_runtime_flags}
)
target_link_options(
  monoprop-sanitizers
  INTERFACE
    -fno-sanitize-recover=all
    ${_monoprop_sanitizer_runtime_flags}
)
