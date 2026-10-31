# Link-time export-visibility probe: unlike monoprop_unit_tests.x (linked against monoprop-objs, the
# plain OBJECT library, which never crosses a dynamic-visibility boundary), this target links against
# the installed "monoprop" SHARED target -- exactly as an external find_package(monoprop CONFIG)
# consumer would. "-Wl,--no-undefined" (GNU ld) makes the link step itself fail on any symbol the
# shared library does not export, instead of deferring the failure to a client's runtime load. Apple
# ld64 errors on undefined symbols by default, so no flag is needed there ("-Wl,-undefined,error" is
# accepted but deprecated, and warns).

# The probe's sources, built once per consumer: the serial libmonoprop and every MPI-enabled one.
set(
  _monoprop_probe_sources
  link_export_probe.cpp
  sharded_state_chain.cpp
  sharded_construction_chain.cpp
  sharded_evaluation_chain.cpp
)

# monoprop-sanitizers is a no-op interface library outside a sanitizer build. Under one, libmonoprop.so
# is ASan-instrumented; without this, the probe executable itself is not, so the ASan runtime reaches
# the process only as a transitive dependency instead of as its own first library, and ASan refuses to
# start ("runtime does not come first in initial library list"). A real external consumer building
# against an ASan-instrumented monoprop would face the same requirement to sanitize their own code.
#
# (j): the consumer must inherit exactly its target's MPI setting: the serial libmonoprop never carries
# monoprop_ENABLE_MPI, every MPI-enabled one always does.
function(monoprop_add_link_export_probe NAME LIBRARY EXPECT_MPI)
  add_executable(${NAME}.x ${_monoprop_probe_sources})
  target_link_libraries(
    ${NAME}.x
    PRIVATE
      monoprop-sanitizers
      ${LIBRARY}
  )
  target_compile_definitions(
    ${NAME}.x
    PRIVATE
      monoprop_EXPECT_MPI=${EXPECT_MPI}
  )
  if(NOT APPLE)
    target_link_options(${NAME}.x PRIVATE "-Wl,--no-undefined")
  endif()
endfunction()

monoprop_add_link_export_probe(monoprop_link_export_probe monoprop 0)
add_test(NAME monoprop_link_export_probe COMMAND monoprop_link_export_probe.x)
# A fixed team of two, so the multi-shard paths (and the raw-access rejection) run here.
set_tests_properties(
  monoprop_link_export_probe
  PROPERTIES
    LABELS
      "unit"
    ENVIRONMENT
      "monoprop_NUM_THREADS=2;OMP_NUM_THREADS=2;OMP_DYNAMIC=FALSE"
)

# The same probe against every MPI-enabled libmonoprop, in singleton MPI (no mpiexec).
foreach(_abi IN LISTS monoprop_MPI_ABI_NAMES)
  set(_probe monoprop_mpi_${_abi}_link_export_probe)
  monoprop_add_link_export_probe(${_probe} monoprop_mpi_${_abi} 1)
  add_test(NAME ${_probe} COMMAND ${_probe}.x)
  set_tests_properties(
    ${_probe}
    PROPERTIES
      LABELS
        "unit"
      ENVIRONMENT
        "${_monoprop_serial_env_entries};monoprop_NUM_THREADS=2;OMP_NUM_THREADS=2;OMP_DYNAMIC=FALSE"
  )
endforeach()

# Several ranks of the same consumer: the multi-rank root chain (l) runs only here, at T = 1 and T = 2, against the
# MPI variant the unit tests use (the one mpiexec launches).
if(monoprop_MPI_ABI_NAMES AND _monoprop_mpiexec)
  list(GET monoprop_MPI_ABI_NAMES 0 _abi)
  foreach(_threads IN ITEMS 1 2)
    add_test(
      NAME monoprop_link_export_probe_mpi2_t${_threads}
      COMMAND
        ${_monoprop_mpiexec} ${_monoprop_mpiexec_numproc_flag} 2
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:monoprop_mpi_${_abi}_link_export_probe.x>
    )
    set_tests_properties(
      monoprop_link_export_probe_mpi2_t${_threads}
      PROPERTIES
        LABELS
          "mpi;mpi-2"
        TIMEOUT
          300
        ENVIRONMENT
          "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1;OMPI_MCA_hwloc_base_binding_policy=none;OMPI_MCA_rmaps_base_oversubscribe=1;PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe;monoprop_NUM_THREADS=${_threads};OMP_NUM_THREADS=${_threads};OMP_DYNAMIC=FALSE;monoprop_TEST_EXPECT_RANKS=2"
    )
  endforeach()
endif()
