set(
  monoprop_GCC_SANITIZER
  "none"
  CACHE STRING
  "GCC sanitizer profile: none, asan-ubsan, or tsan"
)
set_property(
  CACHE
    monoprop_GCC_SANITIZER
  PROPERTY
    STRINGS
      "none"
      "asan-ubsan"
      "tsan"
)

add_library(monoprop-sanitizers INTERFACE)

if(monoprop_GCC_SANITIZER STREQUAL "none")
  message(STATUS "GCC sanitizer profile  : none")
  return()
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "monoprop_GCC_SANITIZER requires Linux")
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES GNU)
  message(FATAL_ERROR "monoprop_GCC_SANITIZER requires a GNU compiler")
endif()

set(
  _monoprop_sanitizer_common_flags
  -O1
  -g3
  -fno-omit-frame-pointer
  -fno-optimize-sibling-calls
  -fno-ipa-icf
  -fno-sanitize-recover=all
)

if(monoprop_GCC_SANITIZER STREQUAL "asan-ubsan")
  set(
    _monoprop_sanitizer_runtime_flags
    -fsanitize=address,undefined,bounds-strict,float-cast-overflow,pointer-compare,pointer-subtract
  )
elseif(monoprop_GCC_SANITIZER STREQUAL "tsan")
  set(_monoprop_sanitizer_runtime_flags -fsanitize=thread)
else()
  message(
    FATAL_ERROR
    "Unknown monoprop_GCC_SANITIZER value: ${monoprop_GCC_SANITIZER}"
  )
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

message(STATUS "GCC sanitizer profile  : ${monoprop_GCC_SANITIZER}")
