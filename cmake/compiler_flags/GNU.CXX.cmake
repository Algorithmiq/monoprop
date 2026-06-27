if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  set(
    monoprop_CXX_FLAGS
    "-Wall -Wno-unknown-pragmas -Wno-sign-compare -Woverloaded-virtual -Wwrite-strings -Wno-unused -Wextra -Wconversion -Wnon-virtual-dtor -Wcast-align -Wunused-parameter -fdiagnostics-color=always"
  )

  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -DDEBUG")
  set(CMAKE_CXX_FLAGS_COVERAGE "-O1 --coverage -g")
endif()
