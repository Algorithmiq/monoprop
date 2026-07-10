if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(
      FATAL_ERROR
      "monoprop requires GNU compiler version >= 14. Detected version: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()

  set(
    monoprop_CXX_FLAGS
    "-Wall -Wno-unknown-pragmas -Wno-sign-compare -Woverloaded-virtual -Wwrite-strings -Wextra -Wconversion -Wnon-virtual-dtor -Wcast-align -Wunused-parameter -fdiagnostics-color=always -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
  )

  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -DDEBUG")
  set(CMAKE_CXX_FLAGS_COVERAGE "-O1 --coverage -g")
endif()
