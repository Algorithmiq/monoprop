if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  set(
    monoprop_CXX_FLAGS
    "-Wall -Wno-padded -Wno-unknown-pragmas -Woverloaded-virtual -Wwrite-strings -fcolor-diagnostics -Wno-c++98-compat -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
  )
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "-O3 -g3 -DDEBUG -glldb -fno-limit-debug-info"
  )
  set(
    CMAKE_CXX_FLAGS_DEBUG
    "-O0 -g3 -DDEBUG -glldb -fno-limit-debug-info -Weffc++ -Wdeprecated -Wdocumentation"
  )
endif()
