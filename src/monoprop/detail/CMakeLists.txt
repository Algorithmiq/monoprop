target_sources(
  monoprop
  PUBLIC
    FILE_SET headers
      TYPE HEADERS
      FILES
      "EnvConfig.h"
)

add_subdirectory(evolution)
add_subdirectory(graph)
add_subdirectory(graph_encoding)
add_subdirectory(monomial_propagator)
add_subdirectory(mpi)
add_subdirectory(operator)
add_subdirectory(pare)
add_subdirectory(partition)
