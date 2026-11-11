cmake_minimum_required(VERSION 3.28...4.0)
project(index_miniapp LANGUAGES CXX)
if(
  CMAKE_CXX_COMPILER_ID
    STREQUAL
    "GNU"
  AND
    CMAKE_CXX_COMPILER_VERSION
      VERSION_LESS
      14
)
  message(FATAL_ERROR "GCC 14 or newer is required")
elseif(
  CMAKE_CXX_COMPILER_ID
    MATCHES
    "Clang"
  AND
    CMAKE_CXX_COMPILER_VERSION
      VERSION_LESS
      18
)
  message(FATAL_ERROR "Clang 18 or newer is required")
endif()
# The study is MPI-off by construction; it must never inherit the library's MPI switch.
if(DEFINED monoprop_ENABLE_MPI)
  message(
    FATAL_ERROR
    "index-miniapp is MPI-off only; do not define monoprop_ENABLE_MPI"
  )
endif()
find_package(Boost 1.85 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
add_executable(index-miniapp main.cpp)
target_compile_features(index-miniapp PRIVATE cxx_std_23)
target_include_directories(
  index-miniapp
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../cpp"
)
target_link_libraries(
  index-miniapp
  PRIVATE
    Boost::boost
    OpenMP::OpenMP_CXX
    Threads::Threads
)
enable_testing()
add_test(
  NAME index-miniapp-self-test
  COMMAND
    index-miniapp --self-test
)
