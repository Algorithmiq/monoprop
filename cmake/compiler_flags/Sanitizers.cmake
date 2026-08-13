#.rst:
#
# Opt-in sanitizer configuration.
#
# Sanitizers are wired in here rather than as a build type because they are
# orthogonal to optimization level: the partition threading layer is only worth
# auditing at the optimization level it ships with, so
# ``monoprop_ENABLE_TSAN=ON`` composes with any ``CMAKE_BUILD_TYPE`` instead of
# replacing it.
#
# The flags are appended to ``CMAKE_CXX_FLAGS`` (not to ``monoprop_CXX_FLAGS``,
# which reaches ``target_compile_options`` only) because ``-fsanitize=thread``
# must appear on the *link* line as well as every compile line; CMake passes
# ``CMAKE_CXX_FLAGS`` to both. The explicit linker-flag appends below cover
# link steps that a toolchain file may drive without ``CMAKE_CXX_FLAGS``.
#
# Variables used::
#
#   monoprop_ENABLE_TSAN
#
# Variables modified::
#
#   CMAKE_CXX_FLAGS
#   CMAKE_EXE_LINKER_FLAGS
#   CMAKE_SHARED_LINKER_FLAGS
#   CMAKE_MODULE_LINKER_FLAGS

option(
  monoprop_ENABLE_TSAN
  "Build with ThreadSanitizer (-fsanitize=thread); opt-in, never on by default"
  OFF
)

if(monoprop_ENABLE_TSAN)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(
      FATAL_ERROR
      "monoprop_ENABLE_TSAN requires GCC or Clang. Detected: ${CMAKE_CXX_COMPILER_ID}"
    )
  endif()

  # -g is forced even in Release: without line tables a TSan report names only
  # addresses, and a report you cannot attribute to a memory ordering is not an
  # audit. -fno-omit-frame-pointer is already in monoprop_CXX_FLAGS, but repeat
  # it here so the setting survives an EXTRA_CXXFLAGS override of that list.
  set(
    monoprop_TSAN_FLAGS
    "-fsanitize=thread -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -g"
  )
  string(APPEND CMAKE_CXX_FLAGS " ${monoprop_TSAN_FLAGS}")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=thread")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -fsanitize=thread")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS " -fsanitize=thread")
endif()
