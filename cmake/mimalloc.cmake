# Fetch and configure mimalloc (https://github.com/microsoft/mimalloc) so that
# it overrides the default `malloc`/`free` interface used by monoprop.
#
# mimalloc is built as a position-independent static library with `MI_OVERRIDE`
# enabled. Linking `mimalloc-static` into the shared `monoprop` library replaces
# the standard allocator entry points (`malloc`, `calloc`, `realloc`, `free`,
# `new`, `delete`, ...) with the mimalloc implementations for the whole library
# and every target that links against it (including the `_core` Python module).
#
# The pinned version is kept here so it is easy to bump in a single place.
set(
  monoprop_MIMALLOC_VERSION
  "3.3.2"
  CACHE STRING
  "mimalloc version to build against"
)

cpmaddpackage(
  NAME "mimalloc"
  VERSION "${monoprop_MIMALLOC_VERSION}"
  GIT_REPOSITORY "https://github.com/microsoft/mimalloc"
  GIT_TAG "v${monoprop_MIMALLOC_VERSION}"
  OPTIONS
    "MI_OVERRIDE ON" # define entry points for malloc/free/new/delete
    "MI_BUILD_STATIC ON"
    "MI_BUILD_SHARED OFF"
    "MI_BUILD_OBJECT OFF"
    "MI_BUILD_TESTS OFF"
    "MI_INSTALL_TOPLEVEL OFF"
)

if(NOT TARGET mimalloc-static)
  message(
    FATAL_ERROR
    "mimalloc was requested but the 'mimalloc-static' target is unavailable."
  )
endif()

message(
  STATUS
  "Overriding the default allocator with mimalloc v${monoprop_MIMALLOC_VERSION}"
)
