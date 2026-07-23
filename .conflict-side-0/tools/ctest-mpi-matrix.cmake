# Run CTest across both MPI OFF and MPI ON configs
cmake_minimum_required(VERSION 3.28)

get_filename_component(_script_dir "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(CTEST_SOURCE_DIRECTORY "${_script_dir}/.." ABSOLUTE)

if(NOT DEFINED CTEST_CMAKE_GENERATOR)
  set(CTEST_CMAKE_GENERATOR "Ninja")
endif()

set(_common_config "-DCMAKE_BUILD_TYPE=Release")
set(
  _matrix
  "nompi|-Dmonoprop_ENABLE_MPI=OFF"
  "mpi|-Dmonoprop_ENABLE_MPI=ON"
)

foreach(_entry IN LISTS _matrix)
  string(
    REPLACE "|"
    ";"
    _parts
    "${_entry}"
  )
  list(GET _parts 0 config_name)
  list(GET _parts 1 extra_opts)

  set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/build/${config_name}")
  file(MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")

  message("==> Configuring ${config_name}")
  ctest_configure(
    BUILD "${CTEST_BINARY_DIRECTORY}"
    SOURCE "${CTEST_SOURCE_DIRECTORY}"
    OPTIONS "${_common_config} ${extra_opts}"
  )

  message("==> Building ${config_name}")
  ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE _build_failed)
  if(_build_failed)
    message(FATAL_ERROR "Build failed for ${config_name}")
  endif()

  message("==> Testing ${config_name}")
  ctest_test(BUILD "${CTEST_BINARY_DIRECTORY}" RETURN_VALUE _test_failed)
  if(_test_failed)
    message(FATAL_ERROR "Tests failed for ${config_name}")
  endif()
endforeach()
