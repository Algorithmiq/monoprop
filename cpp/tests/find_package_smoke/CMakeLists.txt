cmake_minimum_required(VERSION 3.28)
project(monoprop_find_package_smoke LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

option(
  monoprop_SMOKE_MPI
  "Consume the MPI component (monoprop::monoprop_mpi)"
  OFF
)

option(
  monoprop_SMOKE_OPTIONAL_MPI
  "Request the MPI component as optional and consume the serial target"
  OFF
)

if(monoprop_SMOKE_MPI)
  find_package(monoprop CONFIG REQUIRED COMPONENTS MPI)
  set(_monoprop_target monoprop::monoprop_mpi)
elseif(monoprop_SMOKE_OPTIONAL_MPI)
  find_package(monoprop CONFIG REQUIRED OPTIONAL_COMPONENTS MPI)
  set(_monoprop_target monoprop::monoprop)
else()
  find_package(monoprop CONFIG REQUIRED)
  set(_monoprop_target monoprop::monoprop)
endif()

add_executable(
  smoke
  ../link_export_probe/link_export_probe.cpp
  ../link_export_probe/sharded_state_chain.cpp
  ../link_export_probe/sharded_construction_chain.cpp
  ../link_export_probe/sharded_evaluation_chain.cpp
)
target_link_libraries(smoke PRIVATE ${_monoprop_target})

# Only the MPI component's target carries monoprop_ENABLE_MPI; the serial one never does.
target_compile_definitions(
  smoke
  PRIVATE
    monoprop_EXPECT_MPI=$<BOOL:${monoprop_SMOKE_MPI}>
)
