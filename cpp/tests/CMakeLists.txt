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
# monoprop_TEST_EXPECT_TEAM, which only the sharded_team and sharded_root suites read.
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
# discovered per-case tests run at T = 1 and keep the default of one. THREADS is the launch's team; when empty it is read
# from ENVIRONMENT. A launch that names no team takes the OpenMP default (every core): it claims four, enough to keep it
# from sharing a small runner.
function(monoprop_set_test_processors NAME RANKS THREADS ENVIRONMENT)
  if(THREADS)
    set(_threads ${THREADS})
  elseif(ENVIRONMENT MATCHES "(^|;)monoprop_NUM_THREADS=([0-9]+)")
    set(_threads ${CMAKE_MATCH_2})
  elseif(ENVIRONMENT MATCHES "(^|;)OMP_NUM_THREADS=([0-9]+)")
    set(_threads ${CMAKE_MATCH_2})
  elseif(ENVIRONMENT MATCHES "(^|;)OMP_THREAD_LIMIT=([0-9]+)")
    set(_threads ${CMAKE_MATCH_2})
  else()
    set(_threads 4)
  endif()
  math(EXPR _processors "${RANKS} * ${_threads}")
  set_tests_properties(
    ${NAME}
    PROPERTIES
      PROCESSORS
        ${_processors}
  )
endfunction()

# Fresh processes for the construction-time thread budget: each entry reruns one unit case under its own
# launch environment, since an object captures monoprop_NUM_THREADS once, when it is constructed. THREADS T puts the
# exact team (monoprop_team_environment) ahead of ENVIRONMENT; TIMEOUT bounds the launch.
function(monoprop_add_launch_env_case NAME CASE)
  cmake_parse_arguments(
    ""
    ""
    "THREADS;TIMEOUT"
    "ENVIRONMENT;ENVIRONMENT_MODIFICATION"
    ${ARGN}
  )
  if(_THREADS)
    monoprop_team_environment(_team ${_THREADS})
    list(PREPEND _ENVIRONMENT ${_team})
  endif()
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
  monoprop_set_test_processors(${NAME} 1 "${_THREADS}" "${_ENVIRONMENT}")
  if(_TIMEOUT)
    set_tests_properties(
      ${NAME}
      PROPERTIES
        TIMEOUT
          ${_TIMEOUT}
    )
  endif()
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
# team.
foreach(_budget IN ITEMS 1 3)
  monoprop_add_launch_env_case(
    openmp_env_root_budget_${_budget}
    openmp_root_budget_matches_the_launch_environment
    ENVIRONMENT "OMP_DYNAMIC=FALSE;monoprop_NUM_THREADS=${_budget};monoprop_TEST_EXPECT_BUDGET=${_budget}"
  )
endforeach()
# The runtime contract at an exact team of four: settings and affinity untouched, the budget read once, and the
# invalid-owner rule, all inside real multi-owner operations.
monoprop_add_launch_env_case(
  openmp_env_root_t4
  "openmp_root_budget_matches_the_launch_environment,openmp_root_captures_once_and_copies_preserve_it,openmp_propagator_work_leaves_openmp_settings_unchanged,openmp_failed_build_invalidates_the_owner,openmp_invalid_owner_can_still_be_destroyed"
  THREADS 4
  ENVIRONMENT "monoprop_TEST_EXPECT_BUDGET=4"
)
monoprop_add_launch_env_case(
  openmp_env_no_affinity_change
  openmp_propagator_work_does_not_change_affinity
  THREADS 2
  ENVIRONMENT "OMP_PROC_BIND=false"
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

# One exact-team launch per T of the cases matching PATTERN, each a fresh process named <PREFIX>_env_t<T> under
# monoprop_team_environment, never a limited or dynamic team. Cases that need a nonprimary worker skip themselves at
# T = 1. A timeout is a hang, not a pass: the discovered per-case runs share the T = 1 environment
# (_monoprop_unit_environment) but only CTest's default timeout, so a suite with a TIMEOUT keeps its _env_t1 launch.
function(monoprop_add_team_launches PREFIX PATTERN)
  cmake_parse_arguments("" "" "TIMEOUT" "THREADS" ${ARGN})
  foreach(_threads IN LISTS _THREADS)
    monoprop_add_launch_env_case(
      ${PREFIX}_env_t${_threads}
      "${PATTERN}"
      THREADS ${_threads}
      TIMEOUT "${_TIMEOUT}"
    )
  endforeach()
endfunction()

# The fixed-team phase primitive (detail/sharded/Team.h) assumes exactly T workers, so its cases run only under
# exact-team launches, never under the limited or dynamic teams above, which test for_blocks' reduced-team contract.
# The passive-wait launch parks waiting workers, which widens the windows the checkpoint-generation cases probe.
monoprop_add_team_launches(sharded_team "sharded_team_*" THREADS 1 2 4 TIMEOUT 120)
monoprop_add_launch_env_case(
  sharded_team_env_t4_passive
  "sharded_team_*"
  THREADS 4
  TIMEOUT 120
  ENVIRONMENT "OMP_WAIT_POLICY=PASSIVE"
)

# Owner-initialized shard state (detail/sharded/State.h).
monoprop_add_team_launches(sharded_state "sharded_state_*" THREADS 1 2 4 TIMEOUT 300)

# evolved_operator_coefficients() probes every shard of the rank: its suite reruns at T = 2 and 4, where the keys spread
# over several shards, against the shard-transparent evolved_operator_terms() oracle at the same T.
monoprop_add_team_launches(evolved_coefficients "evolved_coefficients_*" THREADS 2 4)

# term_expectation_values() sweeps every shard's state back through its layers and gathers each key from the one shard
# holding it: its suite reruns at T = 2 and 4 against the shard-transparent expectation_value() oracle at the same T.
monoprop_add_team_launches(term_expectation_values "term_expectation_values_*" THREADS 2 4)

# Direct-buffer construction and propagation (detail/sharded/Construction.h).
monoprop_add_team_launches(sharded_construction "sharded_construction_*" THREADS 1 2 4 TIMEOUT 900)

# Snapshot-safe replay, energy, gradients, retained/pared functionals and coefficient-informed construction
# (detail/sharded/Evaluation.h, Construction.h).
monoprop_add_team_launches(sharded_evaluation "sharded_evaluation_*" THREADS 1 2 4 TIMEOUT 900)

# The root (MonomialPropagator over detail/sharded/*) through its public operations, with private shard inspection,
# phase observation and fault injection (PropagatorTestAccess.h).
monoprop_add_team_launches(sharded_root "sharded_root_*" THREADS 1 2 4 TIMEOUT 900)

# The fused-sweep, two-pass and real-record cases (fused_cos_sweep_tests.cpp) at fixed teams of two and four: each
# launch audits its own shards' records and agrees with itself bitwise; geometries are compared through complete
# retained maps in tests/test_sharded_openmp.py, never inside one process.
monoprop_add_team_launches(fused "fused_*,two_pass_apply_matches_graph_replay" THREADS 2 4 TIMEOUT 900)

# No monoprop_NUM_THREADS at all, whatever the caller's environment holds: the root takes the OpenMP default, so
# nothing between CTest and the library may supply the variable.
monoprop_add_launch_env_case(
  sharded_root_env_openmp_default_t3
  "sharded_root_unset_budget_takes_the_openmp_default,sharded_root_owns_t_routed_shards"
  TIMEOUT 900
  ENVIRONMENT "OMP_NUM_THREADS=3;OMP_DYNAMIC=FALSE;monoprop_TEST_EXPECT_TEAM=3;monoprop_TEST_EXPECT_UNSET_BUDGET=3"
  ENVIRONMENT_MODIFICATION "monoprop_NUM_THREADS=unset:"
)

# Multi-process launches of the sharded runtime: P ranks x T threads each, every rank with an exact team. The
# multi-rank cases run only where monoprop_TEST_SHARDED_RANKS names the world size, so the whole-suite MPI variants
# (default budget) skip them. Binding is left to the OpenMP settings: Open MPI would otherwise bind each rank to one
# core and squeeze its T threads onto it (unknown OMPI_MCA_* variables are harmless under other MPIs). Oversubscription
# is allowed in both Open MPI spellings, so a direct `ctest -L mpi` fits 3- and 4-rank launches onto smaller hosts.
# Oversubscribed, spinning OpenMP waiters starve the ranks' MPI progress (4 ranks x 4 threads on a 4-core runner take
# minutes instead of seconds), so waiting workers sleep.
set(
  _monoprop_mpi_launch_env
  "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_hwloc_base_binding_policy=none;OMPI_MCA_rmaps_base_oversubscribe=1;PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe"
)
function(monoprop_add_sharded_mpi_launch NAME RANKS THREADS CASES)
  cmake_parse_arguments("" "" "" "ENVIRONMENT" ${ARGN})
  add_test(
    NAME ${NAME}
    COMMAND
      ${_monoprop_mpiexec} ${_monoprop_mpiexec_numproc_flag} ${RANKS}
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:monoprop_unit_tests.x>
      --run_test=${CASES}
  )
  monoprop_team_environment(_team ${THREADS})
  set(
    _environment
    "${_monoprop_mpi_launch_env};${_team};OMP_WAIT_POLICY=PASSIVE;monoprop_TEST_SHARDED_RANKS=${RANKS};${_ENVIRONMENT}"
  )
  set_tests_properties(
    ${NAME}
    PROPERTIES
      LABELS
        "mpi;mpi-${RANKS}"
      TIMEOUT
        300
      ENVIRONMENT
        "${_environment}"
  )
  monoprop_set_test_processors(${NAME} ${RANKS} ${THREADS} "")
endfunction()

if(monoprop_ENABLE_MPI AND _monoprop_mpiexec)
  # The physical round (detail/sharded/Exchange.h) between processes, in every MPI build: two and three ranks at
  # T = 1, 2 and 4. At world size 1 (the per-case serial runs) the multi-rank cases skip. A timeout is a hang.
  foreach(_ranks IN ITEMS 2 3)
    foreach(_threads IN ITEMS 1 2 4)
      monoprop_add_sharded_mpi_launch(
        sharded_exchange_mpi${_ranks}_t${_threads} ${_ranks} ${_threads} "sharded_exchange_*"
      )
    endforeach()
  endforeach()

  # The root over P ranks: two and four ranks (linear routing) at T = 1, 2 and 4; three ranks under splitmix; and
  # three ranks under linear routing, which every rank must reject together. A hang is a failure.
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
  # The exported replay functions, on the sole shard of a T = 1 root. Worker 1 of a two-worker kernel team owns the
  # failing range, so the failure starts off the calling thread.
  monoprop_add_failure_scenario(
    active_ticket active-ticket ticket "injected failure while the replay Ticket.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )
  monoprop_add_failure_scenario(
    active_ticket_cosine_worker active-ticket cosine-worker "cosine range 2 on OpenMP worker 1 while the replay Ticket.*${_monoprop_abort_diagnostic}" "UNGUARDED"
  )

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
