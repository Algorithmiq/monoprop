{
  lib,
  buildPythonPackage,
  python,

  # build tooling
  cmake,
  ninja,
  pkg-config,
  nanobind,
  nanobind-backend,
  scikit-build-core,
  setuptools-scm,

  # C++ dependencies
  boost,
  hwloc,

  # runtime dependencies
  msgpack,
  numpy,

  # optional MPI support
  mpi,
  mpi4py,
  withMPI ? false,

  # `-march=native` is the upstream default; it is off here because a store path
  # may be built on one machine and substituted onto another.
  enableArchFlags ? false,

  # setuptools-scm derives development versions from git metadata, which the build
  # sandbox does not see. VERSION records the latest stable release.
  version ? lib.removeSuffix "\n" (builtins.readFile ../VERSION),
}:

buildPythonPackage {
  pname = "monoprop";
  inherit version;
  pyproject = true;

  src = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../CMakeLists.txt
      ../LICENSE
      ../README.md
      ../VERSION
      ../pyproject.toml
      # `tools/generate-dispatch.py` stamps this header onto the generated files.
      ../.github/license-header.txt
      ../cmake
      ../cpp
      ../src
      ../tools
    ];
  };

  # scikit-build-core invokes CMake itself; the nixpkgs hook must not configure
  # the tree first.
  dontUseCmakeConfigure = true;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ];

  build-system = [
    nanobind
    # Split mode: `nanobind_add_module(... BACKEND_MODULE nanobind_backend)` builds
    # `_core` without a backend of its own, so the module is needed at build time
    # for the stub pass and at run time for the import.
    nanobind-backend
    scikit-build-core
    setuptools-scm
  ]
  # Upstream provisions mpi4py through a scikit-build-core override keyed on the
  # `monoprop_ENABLE_MPI` environment variable, which would resolve it from PyPI.
  # The sandbox has no network, so the store copy is supplied here instead and the
  # variable is deliberately left unset -- see `env` below.
  ++ lib.optionals withMPI [ mpi4py ];

  buildInputs = [
    boost
    hwloc
  ]
  ++ lib.optionals withMPI [ mpi ];

  dependencies = [
    msgpack
    nanobind-backend
    numpy
  ]
  ++ lib.optionals withMPI [ mpi4py ];

  env = {
    SETUPTOOLS_SCM_PRETEND_VERSION = version;
    SKBUILD_CMAKE_DEFINE = lib.concatStringsSep ";" (
      [
        # The C++ test target resolves msgpack-cxx through CPM, i.e. a git fetch
        # the sandbox denies; C++ tests belong to the dev shell anyway.
        "monoprop_ENABLE_CXX_UNIT_TESTS=OFF"
        "monoprop_ENABLE_ARCH_FLAGS=${if enableArchFlags then "ON" else "OFF"}"
        # nanobind's CMake config lives inside its Python package, which is not
        # on the interpreter's own site-packages path under nixpkgs.
        "nanobind_DIR=${nanobind}/${python.sitePackages}/nanobind/cmake"
      ]
      # The CMake option is set directly rather than through upstream's
      # `monoprop_ENABLE_MPI` environment switch: that switch also appends mpi4py to
      # `build.requires`, i.e. a PyPI resolution the sandbox denies.
      ++ lib.optionals withMPI [ "monoprop_ENABLE_MPI=ON" ]
    );
  };

  # Overrides the single-job default that `[tool.scikit-build]` sets for laptops.
  preBuild = ''
    export SKBUILD_BUILD_TOOL_ARGS="-j$NIX_BUILD_CORES"
  '';

  # The test suite lives outside the wheel and needs fixtures excluded from src.
  doCheck = false;
  pythonImportsCheck = [ "monoprop" ];

  meta = {
    description = "High-performance Majorana and Pauli propagation";
    homepage = "https://github.com/Algorithmiq/monoprop";
    changelog = "https://github.com/Algorithmiq/monoprop/releases";
    license = lib.licenses.asl20;
    platforms = lib.platforms.unix;
  };
}
