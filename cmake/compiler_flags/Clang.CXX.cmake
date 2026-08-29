# same flags are used for both Clang and AppleClang
if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  # check version when compiler id is exactly Clang (valid for Linux)
  if(
    CMAKE_CXX_COMPILER_ID
      STREQUAL
      Clang
    AND
      CMAKE_CXX_COMPILER_VERSION
        VERSION_LESS
        18
  )
    message(
      FATAL_ERROR
      "monoprop requires Clang compiler version >= 18. Detected version: ${CMAKE_CXX_COMPILER_VERSION}"
    )
  endif()

  # -ffp-contract=off, and why it is not a micro-optimization to be traded away: without it the
  # compiler fuses a*b+c into an FMA wherever the target has one, which changes the rounding of the
  # coefficient accumulation. Measured across ISA levels, every evolved term stays bit-identical and
  # only the energy moves, by 1-2 ULP from -march=x86-64-v3 up. That is small and it is also exactly
  # the wrong shape: with a fat binary the same wheel would answer differently depending on which CPU
  # it landed on, and `just diff-baseline` could no longer be a byte-wise gate. The project has no
  # -ffast-math and treats accumulation order as a contract, so contraction is off everywhere -- not
  # only in the tiers -- to keep a source build, a wheel and every tier bit-comparable.
  set(
    monoprop_CXX_FLAGS
    "-Wall -Wno-padded -Wno-unknown-pragmas -Woverloaded-virtual -Wwrite-strings -fcolor-diagnostics -Wno-c++98-compat -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -ffp-contract=off"
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
