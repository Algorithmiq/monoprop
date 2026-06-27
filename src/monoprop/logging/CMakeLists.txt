cpmaddpackage("gh:odygrd/quill#v11.1.0")

# create static library for logging
add_library(monoproplog STATIC "")

target_compile_definitions(
  monoproplog
  PRIVATE
    # keep only the QUILL_LOG_<LEVEL> macros
    QUILL_DISABLE_NON_PREFIXED_MACROS
)

set(monoprop_DEFAULT_LOG_LEVEL "critical")
# Check if CIBUILDWHEEL environment variable is defined and set to "true"
if(DEFINED ENV{CIBUILDWHEEL} AND "$ENV{CIBUILDWHEEL}" STREQUAL "true")
  message(STATUS "Creating Python wheel with cibuildwheel")
  target_compile_definitions(
    monoproplog
    PRIVATE
      # generate code for INFO level (and above)
      QUILL_COMPILE_ACTIVE_LOG_LEVEL=QUILL_COMPILE_ACTIVE_LOG_LEVEL_INFO
      # remove function name
      QUILL_DISABLE_FUNCTION_NAME
      # remove file and line info
      QUILL_DISABLE_FILE_INFO
  )
  set(_caller_file_line "")
else()
  message(STATUS "Running a local build")
  target_compile_definitions(
    monoproplog
    PRIVATE
      # generate code for all levels
      QUILL_COMPILE_ACTIVE_LOG_LEVEL=QUILL_COMPILE_ACTIVE_LOG_LEVEL_TRACE_L3
  )
  set(_caller_file_line "[%(caller_function)@%(file_name):%(line_number)]")
endif()

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/QuillWrapper.cpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/QuillWrapper.cpp
)

target_sources(monoproplog PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/QuillWrapper.cpp)

# flags to prepend
target_compile_options(
  monoproplog
  BEFORE
  PRIVATE
    "${monoprop_CXX_FLAGS}"
    "${ARCH_FLAG}"
)

# flags to append
target_compile_options(monoproplog PRIVATE "${EXTRA_CXXFLAGS}")

target_include_directories(
  monoproplog
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

target_link_libraries(monoproplog PRIVATE quill::quill)
