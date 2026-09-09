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

# CI sanitizes with Clang: -fsanitize-ignorelist= can exempt individual nanobind casters, where
# GCC can only disable a sanitizer for a whole translation unit. GCC still builds asan-ubsan.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(
    FATAL_ERROR
    "monoprop_SANITIZER requires a GNU or Clang compiler, got ${CMAKE_CXX_COMPILER_ID}"
  )
endif()

# TSan cannot see OpenMP synchronization through GCC's libgomp, so every barrier looks like a race.
# LLVM's libomp has the annotations (Archer) that TSan needs.
if(
  monoprop_SANITIZER
    STREQUAL
    "tsan"
  AND
    NOT
      CMAKE_CXX_COMPILER_ID
        MATCHES
        "Clang"
)
  message(
    FATAL_ERROR
    "monoprop_SANITIZER=tsan requires Clang and its OpenMP runtime, got "
    "${CMAKE_CXX_COMPILER_ID}: libgomp has no ThreadSanitizer annotations, so every OpenMP "
    "barrier reports as a data race. Build with CC=clang CXX=clang++."
  )
endif()

set(
  _monoprop_sanitizer_common_flags
  -O1
  -g3
  -fno-omit-frame-pointer
  -fno-optimize-sibling-calls
  # Identical-code folding can misidentify functions in backtraces; Clang does not fold at -O1.
  $<$<CXX_COMPILER_ID:GNU>:-fno-ipa-icf>
  -fno-sanitize-recover=all
)

if(monoprop_SANITIZER STREQUAL "asan-ubsan")
  # Explicit checks complement `undefined`. GCC's `bounds-strict` covers trailing struct
  # arrays; Clang uses `bounds`.
  #
  # ASAN_OPTIONS=detect_invalid_pointer_pairs controls pointer reports because the Python leg
  # uses an uninstrumented CPython.
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
