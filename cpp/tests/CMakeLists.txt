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
    monoprop-objs
    Boost::unit_test_framework
    msgpack-cxx
    PkgConfig::HWLOC
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

# Automatic discovery of unit tests.
# Default CTest run includes per-case serial tests plus suite-level MPI variants
# for monoprop_MPI_TEST_PROCS.
discover_tests(
  monoprop_unit_tests.x
  PROPERTIES
  LABELS
  "unit"
  SERIAL_ENVIRONMENT
  ${_monoprop_serial_env_entries}
)

# Fresh processes for the construction-time thread budget: each entry reruns one unit case under its own
# launch environment, since an object captures monoprop_NUM_THREADS once, when it is constructed.
function(monoprop_add_launch_env_case NAME CASE)
  cmake_parse_arguments("" "" "" "ENVIRONMENT;ENVIRONMENT_MODIFICATION" ${ARGN})
  add_test(
    NAME ${NAME}
    COMMAND
      monoprop_unit_tests.x --run_test=${CASE}
  )
  set(
    _environment
    ${_monoprop_serial_env_entries}
    ${_ENVIRONMENT}
  )
  set_tests_properties(
    ${NAME}
    PROPERTIES
      LABELS
        "unit;openmp_env"
      ENVIRONMENT
        "${_environment}"
  )
  if(_ENVIRONMENT_MODIFICATION)
    set_tests_properties(
      ${NAME}
      PROPERTIES
        ENVIRONMENT_MODIFICATION
          "${_ENVIRONMENT_MODIFICATION}"
    )
  endif()
endfunction()

monoprop_add_launch_env_case(
  openmp_env_prototype_budget_1
  openmp_prototype_budget_matches_the_launch_environment
  ENVIRONMENT "monoprop_NUM_THREADS=1;monoprop_TEST_EXPECT_BUDGET=1"
)
monoprop_add_launch_env_case(
  openmp_env_prototype_budget_3
  openmp_prototype_budget_matches_the_launch_environment
  ENVIRONMENT "monoprop_NUM_THREADS=3;monoprop_TEST_EXPECT_BUDGET=3"
)
# Unset library override: the OpenMP runtime default decides.
monoprop_add_launch_env_case(
  openmp_env_prototype_budget_openmp_default_3
  openmp_prototype_budget_matches_the_launch_environment
  ENVIRONMENT "OMP_NUM_THREADS=3;monoprop_TEST_EXPECT_BUDGET=3"
  ENVIRONMENT_MODIFICATION "monoprop_NUM_THREADS=unset:"
)
monoprop_add_launch_env_case(
  openmp_env_runtime_thread_limit_2
  openmp_runtime_limited_team_keeps_the_captured_budget
  ENVIRONMENT "monoprop_NUM_THREADS=3;OMP_THREAD_LIMIT=2;monoprop_TEST_EXPECT_BUDGET=3"
)
monoprop_add_launch_env_case(
  openmp_env_no_affinity_change
  openmp_propagator_work_does_not_change_affinity
  ENVIRONMENT "OMP_PROC_BIND=false"
)
# The threaded cosine and fused-apply kernels under runtime-limited and dynamic teams: exact results and
# participation must not depend on receiving the full budget.
set(
  _monoprop_kernel_cases
  "openmp_cos_mask_exact,openmp_cos_mask_workers_participate,openmp_cos_lazy_workers_participate,openmp_fused_apply_exact_on_disjoint_records,openmp_fused_apply_workers_participate,openmp_kernel_worker_failures_join_before_rethrow,combined_threaded_lazy_fold_matches_oracle,fused_real_records_have_one_add_owner_and_apply_exactly,fused_threaded_prototype_matches_serial_prototype_exactly"
)
monoprop_add_launch_env_case(
  openmp_env_kernels_thread_limit_2
  "${_monoprop_kernel_cases}"
  ENVIRONMENT "OMP_THREAD_LIMIT=2"
)
monoprop_add_launch_env_case(
  openmp_env_kernels_dynamic_teams
  "${_monoprop_kernel_cases}"
  ENVIRONMENT "OMP_DYNAMIC=TRUE"
)

# Distributed failure scenarios, each a separate two-rank launch with a 30 s timeout. Labelled mpi, so
# only the MPI test leg runs them.
if(monoprop_ENABLE_MPI AND _monoprop_mpiexec)
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
      monoprop-objs
      PkgConfig::HWLOC
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
    add_test(
      NAME mpi_failure_${NAME}
      COMMAND
        ${CMAKE_COMMAND} "-DMPIEXEC=${_monoprop_mpiexec}"
        "-DNUMPROC_FLAG=${_monoprop_mpiexec_numproc_flag}"
        "-DPREFLAGS=${MPIEXEC_PREFLAGS}"
        "-DDRIVER=$<TARGET_FILE:monoprop_mpi_failure_driver.x>"
        "-DSCENARIO=${SCENARIO}" "-DOPTION=${OPTION}" "-DEXPECT=${EXPECT}"
        "-DFORBID=${FORBID}" -DTIMEOUT_SECONDS=30 -P
        "${CMAKE_CURRENT_LIST_DIR}/run_mpi_failure_scenario.cmake"
    )
    set_tests_properties(
      mpi_failure_${NAME}
      PROPERTIES
        LABELS
          "mpi;mpi_failure"
        TIMEOUT
          90
        SKIP_RETURN_CODE
          77
        # As for the suite's MPI variants: CI containers may run as root.
        ENVIRONMENT
          "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1"
    )
  endfunction()

  monoprop_add_failure_scenario(worker_throw worker-throw "" "injected worker failure" "UNGUARDED")
  monoprop_add_failure_scenario(
    before_exchange before-exchange "" "injected failure in the scan" "UNGUARDED|NOT REACHED"
  )
  monoprop_add_failure_scenario(
    active_ticket active-ticket ticket "injected failure while the replay Ticket" "UNGUARDED"
  )
  # Worker 1 of the kernel's two-worker team owns range 2, so the failure starts off the calling thread.
  monoprop_add_failure_scenario(
    active_ticket_cosine_worker active-ticket cosine-worker "cosine range 2 on OpenMP worker 1 while the replay Ticket" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    active_pending active-ticket pending "injected failure while a PendingAlltoallv" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    insufficient_thread_level_single insufficient-thread-level single "requires MPI_THREAD_SERIALIZED" "UNGUARDED|despite"
  )
  monoprop_add_failure_scenario(
    insufficient_thread_level_funneled insufficient-thread-level funneled "requires MPI_THREAD_SERIALIZED" "UNGUARDED|despite"
  )
  # Local fail-fast: the offending thread must not call MPI_Abort (or anything else in MPI).
  monoprop_add_failure_scenario(
    wrong_thread_entry_serialized wrong-thread-entry serialized "thread that initialized MPI" "${_monoprop_abort_diagnostic}|completed"
  )
  monoprop_add_failure_scenario(
    wrong_thread_entry_multiple wrong-thread-entry multiple "thread that initialized MPI" "${_monoprop_abort_diagnostic}|completed"
  )
endif()

add_subdirectory(link_export_probe)
