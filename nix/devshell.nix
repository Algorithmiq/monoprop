{
  lib,
  mkShell,
  stdenv,
  python,

  # C++ toolchain
  cmake,
  ninja,
  pkg-config,
  clang-tools,
  gdb,
  lcov,
  doxygen,

  # C++ dependencies
  boost,
  hwloc,
  openmpi,

  # workflow tooling
  git,
  just,
  nodejs,
  uv,
  zlib,
}:

mkShell {
  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    clang-tools
    gdb
    lcov
    doxygen
    git
    just
    nodejs
    python
    uv
  ];

  # Host inputs, so that CMake's setup hook puts them on NIXPKGS_CMAKE_PREFIX_PATH
  # and PKG_CONFIG_PATH for the CMake run scikit-build-core drives.
  buildInputs = [
    boost
    hwloc
    openmpi
  ];

  shellHook = ''
    # `[tool.uv] python-preference = "only-managed"` pulls prebuilt interpreters
    # that expect a loader NixOS does not provide; use this shell's CPython.
    export UV_PYTHON_PREFERENCE=only-system
    export UV_PYTHON="${python}/bin/python"

    # manylinux wheels (numpy, qiskit, ...) resolve libstdc++ through the ambient
    # loader path, which is empty on NixOS.
    export LD_LIBRARY_PATH="${
      lib.makeLibraryPath [
        stdenv.cc.cc.lib
        zlib
      ]
    }''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  '';
}
