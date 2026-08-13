add_subdirectory(include/monoprop)

# transform monoprop_CXX_FLAGS and EXTRA_CXXFLAGS to ;-separated lists
if(DEFINED monoprop_CXX_FLAGS)
  string(
    REPLACE " "
    ";"
    monoprop_CXX_FLAGS
    ${monoprop_CXX_FLAGS}
  )
endif()
if(DEFINED EXTRA_CXXFLAGS)
  string(
    REPLACE " "
    ";"
    EXTRA_CXXFLAGS
    ${EXTRA_CXXFLAGS}
  )
endif()

add_subdirectory(monoprop)

include(CMakePackageConfigHelpers)

set(monoprop_INSTALL_CMAKEDIR ${PROJECT_NAME}/cmake)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/monopropConfig.cmake.in
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${monoprop_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${SKBUILD_PROJECT_VERSION_FULL}
  COMPATIBILITY SameMajorVersion
)

install(
  EXPORT monopropTargets
  FILE monopropTargets.cmake
  NAMESPACE monoprop::
  DESTINATION ${monoprop_INSTALL_CMAKEDIR}
)

install(
  FILES
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${monoprop_INSTALL_CMAKEDIR}
)

if(monoprop_ENABLE_CXX_UNIT_TESTS)
  add_subdirectory(tests)
endif()
