{
  buildPythonPackage,
  isPyPy,

  # build tooling
  cmake,
  ninja,
  scikit-build-core,

  nanobind_3,
}:

# The compiled half of nanobind's split mode: monoprop's `_core` carries no backend
# of its own and calls into this module at import time. Upstream publishes it as
# binary wheels only, so it is built here from the `nanobind-backend/` subdirectory
# of the nanobind checkout.
buildPythonPackage {
  pname = "nanobind-backend";

  # Not nanobind's version: `nanobind-backend/CMakeLists.txt` refuses to build unless
  # this matches the NB_BACKEND_* ABI macros in `include/nanobind/nb_backend.h`.
  version = "1.0.0";
  pyproject = true;

  disabled = isPyPy;

  # The subproject's CMake resolves `../cmake/nanobind-config.cmake` and compiles
  # headers from `../include`, so the whole checkout is unpacked and only the build
  # directory moves down. Sharing nanobind's source keeps the backend ABI in step
  # with the headers monoprop compiles against.
  inherit (nanobind_3) src;
  sourceRoot = "${nanobind_3.src.name}/nanobind-backend";

  # scikit-build-core invokes CMake itself; the nixpkgs hook must not configure
  # the tree first.
  dontUseCmakeConfigure = true;

  nativeBuildInputs = [
    cmake
    ninja
  ];

  build-system = [ scikit-build-core ];

  # `fill()` dispatches lazily on the ABI major, so importing the package alone
  # would not load the compiled module.
  pythonImportsCheck = [
    "nanobind_backend"
    "nanobind_backend._nb_backend_v1"
  ];

  meta = {
    inherit (nanobind_3.meta) homepage license platforms;
    description = "Compiled nanobind backend for extensions built in split mode";
  };
}
