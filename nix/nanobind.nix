{
  fetchFromGitHub,
  nanobind,
}:

# `[build-system] requires` pins `nanobind==3.0.0` for split mode, while nixpkgs is
# still on the 2.x series. Drop this override once nixpkgs catches up; the pin is
# exact, so a newer nixpkgs nanobind will not satisfy it either.
nanobind.overridePythonAttrs (_: {
  version = "3.0.0";

  # `nanobind_add_backend` reads `ext/robin_map`, a submodule the release tarball
  # omits; nix/nanobind-backend.nix builds out of this same tree.
  src = fetchFromGitHub {
    owner = "wjakob";
    repo = "nanobind";
    tag = "v3.0.0";
    fetchSubmodules = true;
    hash = "sha256-Rx+ZJRpsUwrTTqLva38cT/rp0QmHOo+9CEEouSWZ4NU=";
  };
})
