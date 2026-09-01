# Standalone project: it is configured against an *installed* monoprop, not from the main
# build, so it is deliberately not added via add_subdirectory. `just test-find-package`
# points monoprop_DIR at the installed package and builds it.
#
# What it checks is the exported package itself -- that monopropConfig.cmake resolves, that
# it finds its own dependencies, and that the usage requirements it publishes are enough to
# compile and link a consumer. The in-tree targets cannot see any of that: they never go
# through the config file.

cmake_minimum_required(VERSION 3.28)
project(monoprop_find_package_smoke LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

find_package(monoprop CONFIG REQUIRED)

# The same source as the in-tree link-export probe, so the two cannot drift. Its explicit
# class-template instantiations compile every member of MonomialPropagator against the
# installed headers, and its main() runs the graph and partition chains -- the second is
# why an MPI-enabled package that fails to propagate MPI::MPI_CXX shows up here as a link
# error rather than as a mystery at a downstream user's runtime.
add_executable(smoke ../link_export_probe/link_export_probe.cpp)
target_link_libraries(smoke PRIVATE monoprop::monoprop)
