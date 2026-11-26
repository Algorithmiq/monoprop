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
# Its own main() ends every scenario by aborting the job, so it stays out of the unit runner.
list(FILTER _tests_cpps EXCLUDE REGEX "/mpi_failure_driver\\.cpp$")

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
    monoprop-sanitizers
    ${monoprop_CPP_TEST_OBJS}
    Boost::unit_test_framework
    msgpack-cxx
)

include(${CMAKE_CURRENT_LIST_DIR}/boost-test.cmake)

# CTest launches each Boost case as a world-size-1 process, so excluding fabric components cuts
# MPI_Init from 2.03 s to 0.61 s without affecting communication; real MPI tests must keep the
# full component set.
#
# Use SERIAL_ENVIRONMENT with exclusions (^), not a positive component list: Open MPI component
# names change across versions, and unknown OMPI_MCA_* variables are harmless under other MPIs.
option(
  monoprop_TEST_EXCLUDE_MPI_FABRIC
  "Skip fabric init in the single-process (serial) test variants to cut MPI_Init cost"
  ON
)
set(_monoprop_serial_env_entries)
if(monoprop_TEST_EXCLUDE_MPI_FABRIC AND monoprop_ENABLE_MPI)
  list(
    APPEND _monoprop_serial_env_entries
    "OMPI_MCA_pml=^ucx"
    "OMPI_MCA_btl=^openib,ofi,uct"
  )
endif()

# The exact team T of a fresh-process launch: monoprop_NUM_THREADS = OMP_NUM_THREADS = T with OMP_DYNAMIC off, named in
# monoprop_TEST_EXPECT_TEAM, which the suites read through sharded_test::expected_team().
function(monoprop_team_environment OUT THREADS)
  set(
    ${OUT}
    "monoprop_NUM_THREADS=${THREADS};OMP_NUM_THREADS=${THREADS};OMP_DYNAMIC=FALSE;monoprop_TEST_EXPECT_TEAM=${THREADS}"
    PARENT_SCOPE
  )
endfunction()

# The launch-time thread policy of every discovered case, set by the registration because the runner supplies no
# budget: ordinary (raw-layout) cases run at exactly T = 1, per case and in the whole-suite MPI variants alike; teams of
# T > 1 are dedicated fresh-process launches below, and kernel cases request their own teams explicitly.
monoprop_team_environment(_monoprop_unit_environment 1)

# Automatic discovery of unit tests.
# Default CTest run includes per-case serial tests plus suite-level MPI variants
# for monoprop_MPI_TEST_PROCS.
discover_tests(
  monoprop_unit_tests.x
  PROPERTIES
  LABELS
  "unit"
  ENVIRONMENT
  ${_monoprop_unit_environment}
  SERIAL_ENVIRONMENT
  ${_monoprop_serial_env_entries}
)

# CTest's parallel scheduler (`ctest -j N`, CTEST_PARALLEL_LEVEL) charges each test its PROCESSORS, so a launch claims
# its ranks x threads and a parallel run never oversubscribes the host; a test claiming more than N runs alone. The
# discovered per-case tests run at T = 1 and keep the default of one.
function(monoprop_set_test_processors NAME RANKS THREADS)
  math(EXPR _processors "${RANKS} * ${THREADS}")
  set_tests_properties(
    ${NAME}
    PROPERTIES
      PROCESSORS
        ${_processors}
  )
endfunction()

# The cases that depend on T rerun in one fresh process at T = 4, since an object captures monoprop_NUM_THREADS once,
# when it is constructed: reference energies, the fused and two-pass sweeps, the multi-shard coefficient and
# expectation-value probes, the root's shards and raw-access rule, budget capture, unchanged OpenMP settings and the
# invalid-owner rule. A timeout is a hang.
set(
  _monoprop_team_cases
  "build_graph_*,fused_sweep_*,two_pass_apply_matches_graph_replay,evolved_coefficients_*,term_expectation_values_*,sharded_root_*,openmp_root_*,openmp_propagator_*,openmp_failed_build_invalidates_the_owner"
)
monoprop_team_environment(_monoprop_team_environment 4)
add_test(
  NAME sharded_env_t4
  COMMAND
    monoprop_unit_tests.x --run_test=${_monoprop_team_cases}
)
set(
  _monoprop_team_launch_environment
  ${_monoprop_serial_env_entries}
  ${_monoprop_team_environment}
)
set_tests_properties(
  sharded_env_t4
  PROPERTIES
    LABELS
      "unit"
    TIMEOUT
      600
    ENVIRONMENT
      "${_monoprop_team_launch_environment}"
)
monoprop_set_test_processors(sharded_env_t4 1 4)

# Binding is left to the OpenMP settings: Open MPI would otherwise bind each rank to one core and squeeze its T threads
# onto it (unknown OMPI_MCA_* variables are harmless under other MPIs). Oversubscription is allowed in both Open MPI
# spellings, so the launches fit onto smaller hosts.
set(
  _monoprop_mpi_launch_env
  "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_hwloc_base_binding_policy=none;OMPI_MCA_rmaps_base_oversubscribe=1;PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe"
)

if(monoprop_ENABLE_MPI AND _monoprop_mpiexec)
  # Two ranks at T = 2 each: the root against one process, the physical round and the serial-against-world suites. The
  # whole-suite MPI variants run the same cases at T = 1. Oversubscribed, spinning OpenMP waiters starve the ranks' MPI
  # progress, so waiting workers sleep. A timeout is a hang.
  set(
    _monoprop_multirank_cases
    "sharded_root_multirank_*,sharded_exchange_*,rank_count_energy_within_fp_tolerance,gradient_rank_count_within_fp_tolerance,pauli_rank_count_energy_within_fp_tolerance,mpi_fresh_insert_schrodinger_majorana_serial_world_equiv,mpi_fresh_insert_schrodinger_pauli_serial_world_equiv"
  )
  monoprop_team_environment(_monoprop_multirank_team 2)
  add_test(
    NAME sharded_mpi2_t2
    COMMAND
      ${_monoprop_mpiexec} ${_monoprop_mpiexec_numproc_flag} 2
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:monoprop_unit_tests.x>
      --run_test=${_monoprop_multirank_cases}
  )
  set_tests_properties(
    sharded_mpi2_t2
    PROPERTIES
      LABELS
        "mpi;mpi-2"
      TIMEOUT
        300
      ENVIRONMENT
        "${_monoprop_mpi_launch_env};${_monoprop_multirank_team};OMP_WAIT_POLICY=PASSIVE"
  )
  monoprop_set_test_processors(sharded_mpi2_t2 2 2)

  # Distributed failure and thread-support scenarios, each a separate two-rank launch with a 30 s timeout. Labelled
  # mpi, so only the MPI test leg runs them (mpi_failure_driver.cpp lists them).
  add_executable(monoprop_mpi_failure_driver.x mpi_failure_driver.cpp)
  target_include_directories(
    monoprop_mpi_failure_driver.x
    PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}
      ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_link_libraries(
    monoprop_mpi_failure_driver.x
    PRIVATE
      monoprop-sanitizers
      ${monoprop_CPP_TEST_OBJS}
  )

  set(_monoprop_abort_diagnostic "aborting the communicator")
  function(
    monoprop_add_failure_scenario
    NAME
    SCENARIO
    OPTION
    EXPECT
    FORBID
  )
    cmake_parse_arguments("" "SUCCESS" "" "" ${ARGN})
    add_test(
      NAME mpi_failure_${NAME}
      COMMAND
        ${CMAKE_COMMAND} "-DMPIEXEC=${_monoprop_mpiexec}"
        "-DNUMPROC_FLAG=${_monoprop_mpiexec_numproc_flag}"
        "-DPREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DDRIVER=$<TARGET_FILE:monoprop_mpi_failure_driver.x>"
        "-DSCENARIO=${SCENARIO}" "-DOPTION=${OPTION}" "-DEXPECT=${EXPECT}"
        "-DFORBID=${FORBID}" "-DEXPECT_SUCCESS=${_SUCCESS}" -DTIMEOUT_SECONDS=30
        -P "${CMAKE_CURRENT_LIST_DIR}/run_mpi_failure_scenario.cmake"
    )
    set_tests_properties(
      mpi_failure_${NAME}
      PROPERTIES
        LABELS
          "mpi;mpi_failure"
        TIMEOUT
          90
        # Two ranks; the thread-support scenarios run two threads per rank.
        PROCESSORS
          4
        SKIP_RETURN_CODE
          77
        # As for the suite's MPI variants: CI containers may run as root. Binding is left to OpenMP, so a rank's two
        # threads are not squeezed onto one core.
        ENVIRONMENT
          "${_monoprop_mpi_launch_env};OMP_DYNAMIC=FALSE"
    )
  endfunction()

  # Every failure scenario must end in the library's abort path, never in an escaped exception.
  monoprop_add_failure_scenario(
    build_failure build-failure "" "injected traversal failure.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    active_ticket active-ticket "" "injected failure while the replay Ticket.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )

  # Thread support: FUNNELED is requested and required. Acceptance runs real distributed operations at two threads per
  # rank and must complete; SINGLE is refused at construction, before any communication; a guarded call from another
  # thread fails fast locally and must not call MPI_Abort (or anything else in MPI).
  monoprop_add_failure_scenario(
    host_initialized_funneled host-level funneled "thread-support operations completed with provided=MPI_THREAD_FUNNELED" "UNGUARDED|${_monoprop_abort_diagnostic}"
    SUCCESS
  )
  monoprop_add_failure_scenario(
    insufficient_thread_level_single insufficient-thread-level single "requires MPI_THREAD_FUNNELED.*provides MPI_THREAD_SINGLE" "UNGUARDED|despite"
  )
  monoprop_add_failure_scenario(
    wrong_thread_entry_funneled wrong-thread-entry funneled "thread that initialized MPI" "${_monoprop_abort_diagnostic}|completed"
  )
endif()

add_subdirectory(link_export_probe)

add_test(
  NAME monoprop_mpi_inspection
  COMMAND
    ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_LIST_DIR}/cmake/test_mpi_inspection.cmake
)
set_tests_properties(
  monoprop_mpi_inspection
  PROPERTIES
    LABELS
      "unit"
)
