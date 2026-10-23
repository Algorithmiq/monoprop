# Link-time export-visibility probe: unlike monoprop_unit_tests.x (linked against monoprop-objs, the
# plain OBJECT library, which never crosses a dynamic-visibility boundary), this target links against
# the installed "monoprop" SHARED target -- exactly as an external find_package(monoprop CONFIG)
# consumer would. "-Wl,--no-undefined" (GNU ld) makes the link step itself fail on any symbol the
# shared library does not export, instead of deferring the failure to a client's runtime load. Apple
# ld64 errors on undefined symbols by default, so no flag is needed there ("-Wl,-undefined,error" is
# accepted but deprecated, and warns).

add_executable(monoprop_link_export_probe.x link_export_probe.cpp)

# monoprop-sanitizers is a no-op interface library outside a sanitizer build. Under one, libmonoprop.so
# is ASan-instrumented; without this, the probe executable itself is not, so the ASan runtime reaches
# the process only as a transitive dependency instead of as its own first library, and ASan refuses to
# start ("runtime does not come first in initial library list"). A real external consumer building
# against an ASan-instrumented monoprop would face the same requirement to sanitize their own code.
target_link_libraries(
  monoprop_link_export_probe.x
  PRIVATE
    monoprop-sanitizers
    monoprop
)

if(NOT APPLE)
  target_link_options(monoprop_link_export_probe.x PRIVATE "-Wl,--no-undefined")
endif()

add_test(NAME monoprop_link_export_probe COMMAND monoprop_link_export_probe.x)
set_tests_properties(
  monoprop_link_export_probe
  PROPERTIES
    LABELS
      "unit"
)

# The same probe against every MPI-enabled libmonoprop, in singleton MPI (no mpiexec).
foreach(_abi IN LISTS monoprop_MPI_ABI_NAMES)
  set(_probe monoprop_mpi_${_abi}_link_export_probe)
  add_executable(${_probe}.x link_export_probe.cpp)
  target_link_libraries(
    ${_probe}.x
    PRIVATE
      monoprop-sanitizers
      monoprop_mpi_${_abi}
  )
  if(NOT APPLE)
    target_link_options(${_probe}.x PRIVATE "-Wl,--no-undefined")
  endif()
  add_test(NAME ${_probe} COMMAND ${_probe}.x)
  set_tests_properties(
    ${_probe}
    PROPERTIES
      LABELS
        "unit"
      ENVIRONMENT
        "${_monoprop_serial_env_entries}"
  )
endforeach()
