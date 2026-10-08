cmake_minimum_required(VERSION 3.28)
project(monoprop_find_package_smoke LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

find_package(monoprop CONFIG REQUIRED)

add_executable(
  smoke
  ../link_export_probe/link_export_probe.cpp
  ../link_export_probe/sharded_state_chain.cpp
  ../link_export_probe/sharded_construction_chain.cpp
  ../link_export_probe/sharded_evaluation_chain.cpp
)
target_link_libraries(smoke PRIVATE monoprop::monoprop)

# The MPI setting the caller expects of this package; defaults to what the package reports. Pass
# -DSMOKE_EXPECT_MPI=ON/OFF to check an installed package independently.
if(NOT DEFINED SMOKE_EXPECT_MPI)
  set(SMOKE_EXPECT_MPI "${monoprop_ENABLE_MPI}")
endif()
target_compile_definitions(
  smoke
  PRIVATE
    monoprop_EXPECT_MPI=$<BOOL:${SMOKE_EXPECT_MPI}>
)
