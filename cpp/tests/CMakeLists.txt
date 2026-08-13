find_package(Boost 1.85 COMPONENTS unit_test_framework REQUIRED)

include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)

cpmaddpackage(
  NAME "msgpack-cxx"
  GIT_REPOSITORY "https://github.com/msgpack/msgpack-c"
  GIT_TAG "cpp-7.0.0"
  OPTIONS "MSGPACK_USE_BOOST OFF" "MSGPACK_CXX20 ON"
  SYSTEM YES
  EXCLUDE_FROM_ALL YES
)

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/TestUtilities.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/TestUtilities.h
  @ONLY
)

file(
  GLOB _tests_cpps
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

add_executable(monoprop_unit_tests.x ${_tests_cpps})

target_compile_definitions(
  monoprop_unit_tests.x
  PUBLIC
    BOOST_TEST_DYN_LINK
    BOOST_TEST_NO_MAIN
)

target_include_directories(
  monoprop_unit_tests.x
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(
  monoprop_unit_tests.x
  PRIVATE
    monoprop-objs
    Boost::unit_test_framework
    msgpack-cxx
)

include(${CMAKE_CURRENT_LIST_DIR}/boost-test.cmake)

# Under monoprop_ENABLE_TSAN every test process needs the suppressions file, so it
# is attached as a test property rather than left to the caller's shell.
# history_size raises the shadow-memory depth so a report about a barrier word
# can still name the thread that wrote the previous generation; the spin loops
# retire far more accesses between two interesting ones than the default 2 keeps.
set(_monoprop_test_env)
if(monoprop_ENABLE_TSAN)
  set(
    _monoprop_test_env
    ENVIRONMENT
    "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_LIST_DIR}/tsan.supp:history_size=4:second_deadlock_stack=1"
  )
endif()

# Automatic discovery of unit tests.
# Default CTest run includes per-case serial tests plus suite-level MPI variants
# for monoprop_MPI_TEST_PROCS.
discover_tests(monoprop_unit_tests.x
  PROPERTIES
    LABELS "unit"
    ${_monoprop_test_env}
)
