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
# (g): the consumer must inherit exactly its target's MPI setting: the serial libmonoprop never carries
# monoprop_ENABLE_MPI, every MPI-enabled one always does.
#
# The test runs at a fixed team of two, so the multi-shard paths (and the raw-access rejection) run; ENV_PREFIX entries
# go ahead of that team in its environment.
function(monoprop_add_link_export_probe NAME LIBRARY EXPECT_MPI ENV_PREFIX)
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
  add_test(NAME ${NAME} COMMAND ${NAME}.x)
  monoprop_team_environment(_team 2)
  set(
    _environment
    ${ENV_PREFIX}
    ${_team}
  )
  set_tests_properties(
    ${NAME}
    PROPERTIES
      LABELS
        "unit"
      ENVIRONMENT
        "${_environment}"
  )
  monoprop_set_test_processors(${NAME} 1 2 "")
endfunction()

monoprop_add_link_export_probe(monoprop_link_export_probe monoprop 0 "")

# The same probe against every MPI-enabled libmonoprop, in singleton MPI (no mpiexec).
foreach(_abi IN LISTS monoprop_MPI_ABI_NAMES)
  monoprop_add_link_export_probe(
    monoprop_mpi_${_abi}_link_export_probe monoprop_mpi_${_abi} 1 "${_monoprop_serial_env_entries}"
  )
endforeach()
