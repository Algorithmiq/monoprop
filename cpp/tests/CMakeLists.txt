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

# The launch-time thread policy of every discovered case, set by the registration because the runner supplies no
# budget: ordinary (raw-layout) cases run at exactly T = 1, per case and in the whole-suite MPI variants alike; teams of
# T > 1 are dedicated fresh-process launches below, and kernel cases request their own teams explicitly.
set(
  _monoprop_unit_environment
  "monoprop_NUM_THREADS=1"
  "OMP_NUM_THREADS=1"
  "OMP_DYNAMIC=FALSE"
)

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

# Budget capture at the launch: every root captures it, and its T is its shard count, so each launch names an exact
# team. The unset launch removes monoprop_NUM_THREADS whatever the caller's environment holds, so the OpenMP runtime
# default decides; neither CTest nor the runner may supply it.
monoprop_add_launch_env_case(
  openmp_env_root_budget_1
  openmp_root_budget_matches_the_launch_environment
  ENVIRONMENT "OMP_DYNAMIC=FALSE;monoprop_NUM_THREADS=1;monoprop_TEST_EXPECT_BUDGET=1"
)
monoprop_add_launch_env_case(
  openmp_env_root_budget_3
  openmp_root_budget_matches_the_launch_environment
  ENVIRONMENT "OMP_DYNAMIC=FALSE;monoprop_NUM_THREADS=3;monoprop_TEST_EXPECT_BUDGET=3"
)
monoprop_add_launch_env_case(
  openmp_env_root_budget_openmp_default_3
  openmp_root_budget_matches_the_launch_environment
  ENVIRONMENT "OMP_DYNAMIC=FALSE;OMP_NUM_THREADS=3;monoprop_TEST_EXPECT_BUDGET=3;monoprop_TEST_EXPECT_UNSET_BUDGET=3"
  ENVIRONMENT_MODIFICATION "monoprop_NUM_THREADS=unset:"
)
# The runtime contract at an exact team of four: settings and affinity untouched, the budget read once, and the
# invalid-owner rule, all inside real multi-owner operations.
monoprop_add_launch_env_case(
  openmp_env_root_t4
  "openmp_root_budget_matches_the_launch_environment,openmp_root_captures_once_and_copies_preserve_it,openmp_propagator_work_leaves_openmp_settings_unchanged,openmp_failed_build_invalidates_the_owner,openmp_validation_errors_leave_the_owner_usable,openmp_invalid_owner_can_still_be_destroyed"
  ENVIRONMENT "OMP_DYNAMIC=FALSE;monoprop_NUM_THREADS=4;OMP_NUM_THREADS=4;monoprop_TEST_EXPECT_BUDGET=4"
)
monoprop_add_launch_env_case(
  openmp_env_no_affinity_change
  openmp_propagator_work_does_not_change_affinity
  ENVIRONMENT "OMP_PROC_BIND=false;OMP_DYNAMIC=FALSE;monoprop_NUM_THREADS=2;OMP_NUM_THREADS=2"
)

# The threaded cosine, fused-apply, scan and resolve kernels under runtime-limited and dynamic teams: exact results and
# participation must not depend on receiving the full budget.
set(
  _monoprop_kernel_cases
  "openmp_cos_mask_exact,openmp_cos_mask_workers_participate,openmp_cos_lazy_workers_participate,openmp_fused_apply_exact_on_disjoint_records,openmp_fused_apply_workers_participate,openmp_kernel_worker_failures_join_before_rethrow,combined_threaded_lazy_fold_matches_oracle,openmp_scan_ranges_match_serial_majorana,openmp_scan_workers_participate,openmp_scan_worker_failure_joins_before_rethrow,openmp_scan_prepares_cold_caches_on_the_caller,openmp_scan_pauli_ranges_match_serial,openmp_probe_frozen_positions_matches_one_batch,openmp_probe_frozen_positions_workers_participate,openmp_incoming_probe_threaded_matches_serial,openmp_incoming_decode_and_probe_workers_participate,openmp_incoming_decode_worker_failure_joins_before_probing,openmp_resolve_scatter_threaded_matches_serial,openmp_resolve_scatter_workers_participate,openmp_self_windows_match_serial,openmp_mixed_self_and_remote_publication_order"
)
# These exercise the kernels' own explicit budgets and for_blocks' reduced-team contract, which is independent of the
# root: no fixed-team root operation runs under these limited or dynamic teams.
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
  "sharded_team_visits_each_owner,sharded_team_observes_worker_identity,sharded_team_keeps_owner_across_phases,sharded_team_primary_only_phase_runs_on_the_caller,sharded_team_empty_owners_reach_the_checkpoint,sharded_team_publishes_writes_to_the_next_phase,sharded_team_single_worker_failure_is_returned_after_join,sharded_team_failure_is_returned_not_rethrown,sharded_team_failure_selects_the_lowest_recorded_slot,sharded_team_preserves_the_original_exception,sharded_team_fresh_invocation_has_no_stale_failure,sharded_team_rejects_a_nonpositive_budget_before_the_region,sharded_team_leaves_runtime_settings_unchanged,sharded_team_launch_supplies_the_expected_team,sharded_team_section_limit_bounds_concurrent_holders,sharded_team_section_limit_releases_on_exception,sharded_team_section_limit_admits_at_least_one"
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
  "sharded_state_make_shards_initializes_each_owner_once,sharded_state_seed_matches_an_independent_reference,sharded_state_routed_rows_belong_to_their_flat_owner,sharded_state_keeps_sparse_heisenberg_and_dense_schrodinger_caches,sharded_state_counts_and_accounting_sum_over_shards,sharded_state_rejects_invalid_arguments_before_the_team,sharded_state_copy_is_exact_and_owner_allocated,sharded_state_copies_mutate_independently,sharded_state_graph_cores_are_shared_and_lists_independent,sharded_state_copy_keeps_the_captured_budget,sharded_state_initializer_failure_on_the_primary_is_returned,sharded_state_seed_allocation_failures_are_catchable,sharded_state_copy_allocation_failures_leave_sources_intact"
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

# Direct-buffer construction and propagation (detail/sharded/Construction.h) under the same exact-team launches: one
# fresh process per T. Cases that need a nonprimary worker are left out of T = 1. A timeout is a hang, not a pass.
set(
  _monoprop_sharded_construction_single_cases
  "sharded_construction_rows_follow_the_ordering_contract,sharded_construction_propagation_matches_an_independent_reference,sharded_construction_owners_do_their_own_work,sharded_construction_opaque_cutoff_traverses_on_the_primary,sharded_construction_empty_operators_circuits_and_identity,sharded_construction_published_views_cover_the_window,sharded_construction_published_resolution_matches_owning_blocks,sharded_construction_large_self_streams_match_the_root,sharded_construction_wide_terms_match_the_root,sharded_construction_failure_in_each_phase_suppresses_later_work,sharded_construction_allocation_failures_are_contained,sharded_construction_rejects_invalid_arguments_before_the_team"
)
foreach(_threads IN ITEMS 1 2 4)
  if(_threads EQUAL 1)
    set(_cases "${_monoprop_sharded_construction_single_cases}")
  else()
    set(_cases "sharded_construction_*")
  endif()
  monoprop_add_launch_env_case(
    sharded_construction_env_t${_threads}
    "${_cases}"
    ENVIRONMENT "monoprop_NUM_THREADS=${_threads};OMP_NUM_THREADS=${_threads};OMP_DYNAMIC=FALSE"
  )
  set_tests_properties(
    sharded_construction_env_t${_threads}
    PROPERTIES
      TIMEOUT
        900
  )
endforeach()

# Snapshot-safe replay, energy, gradients, retained/pared functionals and coefficient-informed construction
# (detail/sharded/Evaluation.h, Construction.h) under the same exact-team launches: one fresh process per T. Cases that
# need a nonprimary worker are left out of T = 1. A timeout is a hang, not a pass.
set(
  _monoprop_sharded_evaluation_single_cases
  "sharded_evaluation_identity_once_and_empty_parameters,sharded_evaluation_rejects_invalid_arguments_before_the_team,sharded_evaluation_matches_independent_references,sharded_evaluation_deep_circuit_gradients,sharded_evaluation_repeated_parameters_and_duplicate_records,sharded_evaluation_pared_functionals_match_the_root,sharded_evaluation_cosine_counts_reserve_records_exactly,sharded_evaluation_empty_stored_mask_applies_nothing,sharded_evaluation_repeated_instances_copies_and_index_growth,sharded_evaluation_owners_do_their_own_work,sharded_evaluation_opaque_callbacks_run_on_the_primary,sharded_evaluation_combination_folds_in_shard_order,sharded_evaluation_failure_in_each_phase_suppresses_later_work,sharded_evaluation_callback_paring_and_construction_failures,sharded_evaluation_allocation_failures_are_contained"
)
foreach(_threads IN ITEMS 1 2 4)
  if(_threads EQUAL 1)
    set(_cases "${_monoprop_sharded_evaluation_single_cases}")
  else()
    set(_cases "sharded_evaluation_*")
  endif()
  monoprop_add_launch_env_case(
    sharded_evaluation_env_t${_threads}
    "${_cases}"
    ENVIRONMENT "monoprop_NUM_THREADS=${_threads};OMP_NUM_THREADS=${_threads};OMP_DYNAMIC=FALSE"
  )
  set_tests_properties(
    sharded_evaluation_env_t${_threads}
    PROPERTIES
      TIMEOUT
        900
  )
endforeach()

# The root (MonomialPropagator over detail/sharded/*) through its public operations, with private shard inspection,
# phase observation and fault injection (PropagatorTestAccess.h). One fresh process per T, exact teams only; a timeout
# is a hang, not a pass. Cases needing a nonprimary worker skip themselves at T = 1.
foreach(_threads IN ITEMS 1 2 4)
  monoprop_add_launch_env_case(
    sharded_root_env_t${_threads}
    "sharded_root_*"
    ENVIRONMENT "monoprop_NUM_THREADS=${_threads};OMP_NUM_THREADS=${_threads};OMP_DYNAMIC=FALSE;monoprop_TEST_EXPECT_TEAM=${_threads}"
  )
  set_tests_properties(
    sharded_root_env_t${_threads}
    PROPERTIES
      TIMEOUT
        900
  )
endforeach()

# The fused-sweep, two-pass and real-record cases (fused_cos_sweep_tests.cpp) at fixed teams of two and four: each
# launch audits its own shards' records and agrees with itself bitwise; geometries are compared through complete
# retained maps in tests/test_sharded_openmp.py, never inside one process.
foreach(_threads IN ITEMS 2 4)
  monoprop_add_launch_env_case(
    fused_env_t${_threads}
    "fused_*,two_pass_apply_matches_graph_replay"
    ENVIRONMENT "monoprop_NUM_THREADS=${_threads};OMP_NUM_THREADS=${_threads};OMP_DYNAMIC=FALSE"
  )
  set_tests_properties(
    fused_env_t${_threads}
    PROPERTIES
      TIMEOUT
        900
  )
endforeach()

# No monoprop_NUM_THREADS at all, whatever the caller's environment holds: the root takes the OpenMP default, so
# nothing between CTest and the library may supply the variable.
monoprop_add_launch_env_case(
  sharded_root_env_openmp_default_t3
  "sharded_root_unset_budget_takes_the_openmp_default,sharded_root_owns_t_routed_shards"
  ENVIRONMENT "OMP_NUM_THREADS=3;OMP_DYNAMIC=FALSE;monoprop_TEST_EXPECT_TEAM=3;monoprop_TEST_EXPECT_UNSET_BUDGET=3"
  ENVIRONMENT_MODIFICATION "monoprop_NUM_THREADS=unset:"
)
set_tests_properties(
  sharded_root_env_openmp_default_t3
  PROPERTIES
    TIMEOUT
      900
)

# Multi-process launches of the sharded runtime: P ranks x T threads each, every rank with an exact team. The
# multi-rank cases run only where monoprop_TEST_SHARDED_RANKS names the world size, so the whole-suite MPI variants
# (default budget) skip them. Binding is left to the OpenMP settings: Open MPI would otherwise bind each rank to one
# core and squeeze its T threads onto it (unknown OMPI_MCA_* variables are harmless under other MPIs).
function(monoprop_add_sharded_mpi_launch NAME RANKS THREADS CASES)
  cmake_parse_arguments("" "" "" "ENVIRONMENT" ${ARGN})
  add_test(
    NAME ${NAME}
    COMMAND
      ${_monoprop_mpiexec} ${_monoprop_mpiexec_numproc_flag} ${RANKS}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:monoprop_unit_tests.x>
      --run_test=${CASES}
  )
  set_tests_properties(
    ${NAME}
    PROPERTIES
      LABELS
        "mpi;mpi-${RANKS}"
      TIMEOUT
        300
      ENVIRONMENT
        "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_hwloc_base_binding_policy=none;monoprop_NUM_THREADS=${THREADS};OMP_NUM_THREADS=${THREADS};OMP_DYNAMIC=FALSE;monoprop_TEST_EXPECT_TEAM=${THREADS};monoprop_TEST_SHARDED_RANKS=${RANKS};${_ENVIRONMENT}"
  )
endfunction()

# The physical round (detail/sharded/Exchange.h) between processes, in every MPI build: two and three ranks at
# T = 1, 2 and 4. At world size 1 (the per-case serial runs) the multi-rank cases skip. A timeout is a hang.
if(monoprop_ENABLE_MPI AND _monoprop_mpiexec)
  foreach(_ranks IN ITEMS 2 3)
    foreach(_threads IN ITEMS 1 2 4)
      monoprop_add_sharded_mpi_launch(
        sharded_exchange_mpi${_ranks}_t${_threads} ${_ranks} ${_threads} "sharded_exchange_*"
      )
    endforeach()
  endforeach()
endif()

# The root over P ranks: two and four ranks (linear routing) at T = 1, 2 and 4; three ranks under splitmix; and three
# ranks under linear routing, which every rank must reject together. A hang is a failure.
if(monoprop_ENABLE_MPI AND _monoprop_mpiexec)
  foreach(_ranks IN ITEMS 2 4)
    foreach(_threads IN ITEMS 1 2 4)
      monoprop_add_sharded_mpi_launch(
        sharded_root_mpi${_ranks}_t${_threads} ${_ranks} ${_threads} "sharded_root_multirank_*"
      )
    endforeach()
  endforeach()
  monoprop_add_sharded_mpi_launch(
    sharded_root_mpi3_t2_splitmix 3 2 "sharded_root_multirank_*" ENVIRONMENT "monoprop_ROUTING=splitmix"
  )
  monoprop_add_sharded_mpi_launch(
    sharded_root_mpi3_t2_linear_rejected 3 2 "sharded_root_multirank_rejects_an_unroutable_geometry"
  )
  # The serial-against-world equivalence suites, through the public API at P ranks x T.
  set(
    _monoprop_world_equivalence_cases
    "rank_count_energy_within_fp_tolerance,gradient_rank_count_within_fp_tolerance,pauli_rank_count_energy_within_fp_tolerance,mpi_fresh_insert_schrodinger_majorana_serial_world_equiv,mpi_fresh_insert_schrodinger_pauli_serial_world_equiv"
  )
  foreach(_ranks IN ITEMS 2 4)
    foreach(_threads IN ITEMS 1 2 4)
      monoprop_add_sharded_mpi_launch(
        sharded_world_equivalence_mpi${_ranks}_t${_threads} ${_ranks} ${_threads} "${_monoprop_world_equivalence_cases}"
      )
    endforeach()
  endforeach()
endif()

# Distributed failure and thread-support scenarios, each a separate two-rank launch with a 30 s timeout. Labelled mpi,
# so only the MPI test leg runs them (mpi_failure_driver.cpp lists them).
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
        SKIP_RETURN_CODE
          77
        # As for the suite's MPI variants: CI containers may run as root. Binding is left to OpenMP, so a rank's two
        # threads are not squeezed onto one core.
        ENVIRONMENT
          "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_hwloc_base_binding_policy=none;OMP_DYNAMIC=FALSE"
    )
  endfunction()

  # Every failure scenario must end in the library's abort (or local fail-fast) path, never in an escaped exception.
  monoprop_add_failure_scenario(
    sharded_worker_throw worker-throw "" "injected replay-finish failure on OpenMP worker 1.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    sharded_before_exchange before-exchange "" "injected traversal failure.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    sharded_active_round active-ticket round "injected cosine failure while the replay round is posted.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    sharded_active_counts active-ticket counts "injected pack failure while the count round is posted.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    sharded_active_local_round active-ticket local-round "injected cosine failure while a caller-local replay round is posted.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    sharded_malformed_receive malformed-receive "" "QueryWire: malformed record.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  # The low-level one-owner-per-rank engine and the exported replay functions, on the sole shard of a T = 1 root.
  # Worker 1 of a two-worker kernel team owns the failing range, so the failure starts off the calling thread.
  monoprop_add_failure_scenario(
    active_ticket active-ticket ticket "injected failure while the replay Ticket.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    active_ticket_cosine_worker active-ticket cosine-worker "cosine range 2 on OpenMP worker 1 while the replay Ticket.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    active_pending active-ticket pending "injected failure while a PendingAlltoallv.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    before_exchange_scan_worker before-exchange scan-worker "scan range 1 on OpenMP worker 1.*${_monoprop_abort_diagnostic}" "UNGUARDED|NOT REACHED"
  )
  # Resolve workers: the self probe fails before the query exchange; the incoming decode, frozen probe and scatter fail
  # after the query round while the peers enter the response round. Worker 1 owns the last range.
  foreach(_phase self-probe decode incoming-probe scatter)
    string(
      REPLACE "-"
      "_"
      _phase_name
      "${_phase}"
    )
    monoprop_add_failure_scenario(
      resolve_worker_${_phase_name} resolve-worker ${_phase} "${_phase} range [0-9]+ on OpenMP worker 1.*${_monoprop_abort_diagnostic}" "UNGUARDED|NOT REACHED"
    )
  endforeach()

  # Thread support: FUNNELED is requested and required. Acceptance runs real distributed operations at two threads per
  # rank and must complete; SINGLE is refused at construction, before any communication.
  monoprop_add_failure_scenario(
    library_initialized_level library-level "" "provided=MPI_THREAD_(FUNNELED|SERIALIZED|MULTIPLE).*thread-support operations completed" "UNGUARDED|${_monoprop_abort_diagnostic}"
    SUCCESS
  )
  foreach(_level funneled serialized multiple)
    string(TOUPPER "${_level}" _upper)
    monoprop_add_failure_scenario(
      host_initialized_${_level} host-level ${_level} "thread-support operations completed with provided=MPI_THREAD_${_upper}" "UNGUARDED|${_monoprop_abort_diagnostic}"
      SUCCESS
    )
  endforeach()
  monoprop_add_failure_scenario(
    insufficient_thread_level_single insufficient-thread-level single "requires MPI_THREAD_FUNNELED.*provides MPI_THREAD_SINGLE" "UNGUARDED|despite"
  )
  # Local fail-fast at every supported level: the offending thread must not call MPI_Abort (or anything else in MPI).
  foreach(_level funneled serialized multiple)
    monoprop_add_failure_scenario(
      wrong_thread_entry_${_level} wrong-thread-entry ${_level} "thread that initialized MPI" "${_monoprop_abort_diagnostic}|completed"
    )
  endforeach()
endif()

add_subdirectory(link_export_probe)
