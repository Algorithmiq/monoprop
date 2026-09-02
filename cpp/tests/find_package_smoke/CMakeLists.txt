cmake_minimum_required(VERSION 3.28)
project(monoprop_find_package_smoke LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

find_package(monoprop CONFIG REQUIRED)

add_executable(smoke ../link_export_probe/link_export_probe.cpp)
target_link_libraries(smoke PRIVATE monoprop::monoprop)
