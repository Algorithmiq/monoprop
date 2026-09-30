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
# The threaded cosine, fused-apply, scan and resolve kernels under runtime-limited and dynamic teams: exact results and
# participation must not depend on receiving the full budget.
set(
  _monoprop_kernel_cases
  "openmp_cos_mask_exact,openmp_cos_mask_workers_participate,openmp_cos_lazy_workers_participate,openmp_fused_apply_exact_on_disjoint_records,openmp_fused_apply_workers_participate,openmp_kernel_worker_failures_join_before_rethrow,combined_threaded_lazy_fold_matches_oracle,fused_real_records_have_one_add_owner_and_apply_exactly,fused_threaded_prototype_matches_serial_prototype_exactly,openmp_scan_ranges_match_serial_majorana,openmp_scan_workers_participate,openmp_scan_worker_failure_joins_before_rethrow,openmp_scan_prepares_cold_caches_on_the_caller,openmp_scan_pauli_ranges_match_serial,openmp_scan_threaded_construction_matches_serial_exactly,openmp_scan_construction_workers_participate,openmp_probe_frozen_positions_matches_one_batch,openmp_probe_frozen_positions_workers_participate,openmp_incoming_probe_threaded_matches_serial,openmp_incoming_decode_and_probe_workers_participate,openmp_incoming_decode_worker_failure_joins_before_probing,openmp_resolve_scatter_threaded_matches_serial,openmp_resolve_scatter_workers_participate,openmp_self_windows_match_serial,openmp_mixed_self_and_remote_publication_order,openmp_self_probe_construction_workers_participate"
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

# The fixed-team phase primitive (detail/sharded/Team.h) assumes exactly T workers, so its cases run only under
# exact-team launches, never under the limited or dynamic teams above, which test for_blocks' reduced-team contract.
# Each T is a fresh process; cases that need a nonprimary worker are left out of T = 1. The passive-wait launch parks
# waiting workers, which widens the windows the checkpoint-generation cases probe.
set(
  _monoprop_sharded_team_single_cases
  "sharded_team_visits_each_owner,sharded_team_observes_worker_identity,sharded_team_keeps_owner_across_phases,sharded_team_primary_only_phase_runs_on_the_caller,sharded_team_empty_owners_reach_the_checkpoint,sharded_team_publishes_writes_to_the_next_phase,sharded_team_single_worker_failure_is_returned_after_join,sharded_team_failure_is_returned_not_rethrown,sharded_team_failure_selects_the_lowest_recorded_slot,sharded_team_preserves_the_original_exception,sharded_team_fresh_invocation_has_no_stale_failure,sharded_team_rejects_a_nonpositive_budget_before_the_region,sharded_team_leaves_runtime_settings_unchanged,sharded_team_launch_supplies_the_expected_team"
)
foreach(_threads IN ITEMS 1 2 4)
  if(_threads EQUAL 1)
    set(_cases "${_monoprop_sharded_team_single_cases}")
  else()
    set(_cases "sharded_team_*")
  endif()
  set(
    _team_environment
    "monoprop_NUM_THREADS=${_threads};OMP_NUM_THREADS=${_threads};OMP_DYNAMIC=FALSE;monoprop_TEST_EXPECT_TEAM=${_threads}"
  )
  monoprop_add_launch_env_case(
    sharded_team_env_t${_threads}
    "${_cases}"
    ENVIRONMENT "${_team_environment}"
  )
  set_tests_properties(
    sharded_team_env_t${_threads}
    PROPERTIES
      TIMEOUT
        120
  )
endforeach()
monoprop_add_launch_env_case(
  sharded_team_env_t4_passive
  "sharded_team_*"
  ENVIRONMENT "monoprop_NUM_THREADS=4;OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE;OMP_WAIT_POLICY=PASSIVE;monoprop_TEST_EXPECT_TEAM=4"
)
set_tests_properties(
  sharded_team_env_t4_passive
  PROPERTIES
    TIMEOUT
      120
)

# Owner-initialized shard state (detail/sharded/State.h) under the same exact-team launches: one fresh process per T,
# never a limited or dynamic team. Cases that need a nonprimary worker are left out of T = 1.
set(
  _monoprop_sharded_state_single_cases
  "sharded_state_make_shards_initializes_each_owner_once,sharded_state_seed_matches_the_partition_oracle,sharded_state_seed_matches_an_independent_reference,sharded_state_routed_rows_belong_to_their_flat_owner,sharded_state_keeps_sparse_heisenberg_and_dense_schrodinger_caches,sharded_state_counts_and_accounting_sum_over_shards,sharded_state_rejects_invalid_arguments_before_the_team,sharded_state_copy_is_exact_and_owner_allocated,sharded_state_copies_mutate_independently,sharded_state_graph_cores_are_shared_and_lists_independent,sharded_state_copy_keeps_the_captured_budget,sharded_state_invalid_source_is_rejected_by_the_owner_guard,sharded_state_initializer_failure_on_the_primary_is_returned,sharded_state_seed_allocation_failures_are_catchable,sharded_state_copy_allocation_failures_leave_sources_intact,sharded_state_legacy_constructor_allocation_failures_are_catchable"
)
foreach(_threads IN ITEMS 1 2 4)
  if(_threads EQUAL 1)
    set(_cases "${_monoprop_sharded_state_single_cases}")
  else()
    set(_cases "sharded_state_*")
  endif()
  monoprop_add_launch_env_case(
    sharded_state_env_t${_threads}
    "${_cases}"
    ENVIRONMENT "monoprop_NUM_THREADS=${_threads};OMP_NUM_THREADS=${_threads};OMP_DYNAMIC=FALSE"
  )
  set_tests_properties(
    sharded_state_env_t${_threads}
    PROPERTIES
      TIMEOUT
        300
  )
endforeach()

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
  # Worker 1 of the scan's two-worker team owns word range 1; peers enter the query exchange normally.
  monoprop_add_failure_scenario(
    before_exchange_scan_worker before-exchange scan-worker "scan range 1 on OpenMP worker 1" "UNGUARDED|NOT REACHED"
  )
  # Resolve workers: the self probe fails before the query exchange; the incoming decode, frozen probe and
  # scatter fail after the query round while the peers enter the response round. Worker 1 owns the last range.
  foreach(_phase self-probe decode incoming-probe scatter)
    string(
      REPLACE "-"
      "_"
      _phase_name
      "${_phase}"
    )
    monoprop_add_failure_scenario(
      resolve_worker_${_phase_name} resolve-worker ${_phase} "${_phase} range [0-9]+ on OpenMP worker 1" "UNGUARDED|NOT REACHED"
    )
  endforeach()
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
