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

# CI uses GCC, matching the wheel and gcov coverage builds. Clang is also supported; its
# -fsanitize-ignorelist= can exempt named nanobind casters, whereas GCC only supports
# translation-unit-wide -fno-sanitize=.
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
