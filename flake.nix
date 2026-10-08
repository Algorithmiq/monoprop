{
  description = "monoprop: because your operators deserve to propagate at escape velocity.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    let
      # Interpreters with flake outputs.
      pythonVersions = [
        "312"
        "313"
        "314"
      ];
      # Backs the unsuffixed packages, the `nix run` app and the dev shell. Tracks
      # nixpkgs' `python3`: Hydra builds and caches only the maintained package sets,
      # older ones are compiled locally and break more often.
      defaultPython = "python314";

      # Adds `monoprop`, `monoprop-mpi` and `nanobind-backend` to every Python package
      # set, so `python313.pkgs.monoprop` and friends resolve.
      monopropPythonExtension =
        pyfinal: pyprev:
        let
          # On Python 3.12 the pinned nixpkgs fails anyio's TLS tests, which
          # scikit-build-core's test suite depends on, so neither it nor nanobind_3
          # is cached there. Skipping that test suite unblocks the build; newer
          # interpreters use the upstream, cached packages untouched. Drop this once
          # anyio builds upstream. It stays local to this extension: overriding the
          # package set would rebuild every scikit-build-core consumer.
          anyioBroken = pyprev.pythonOlder "3.13";
          scikit-build-core =
            if anyioBroken then
              pyprev.scikit-build-core.overridePythonAttrs (_: {
                doCheck = false;
              })
            else
              pyprev.scikit-build-core;
          nanobind_3 =
            if anyioBroken then
              pyprev.nanobind_3.override { inherit scikit-build-core; }
            else
              pyprev.nanobind_3;
        in
        {
          nanobind-backend = pyfinal.callPackage ./nix/nanobind-backend.nix {
            inherit scikit-build-core nanobind_3;
          };
          monoprop = pyfinal.callPackage ./nix/monoprop.nix {
            inherit nanobind_3;
            # monoprop's `[build-system] requires` wants >= 1.0.3, newer than
            # nixpkgs ships.
            scikit-build-core = scikit-build-core.overridePythonAttrs (_: rec {
              version = "1.0.3";
              src = pyfinal.fetchPypi {
                pname = "scikit_build_core";
                inherit version;
                hash = "sha256-pNegWXjuN5dcN3Q1EMiZHi3rzn74OvsKB8DFdv1PFug=";
              };
              # The nixpkgs test setup targets the version nixpkgs ships.
              doCheck = false;
            });
          };
          monoprop-mpi = pyfinal.monoprop.override { withMPI = true; };
        };

      monopropOverlay = final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [ monopropPythonExtension ];

        monoprop = final.${defaultPython}.pkgs.monoprop;
        monoprop-mpi = final.${defaultPython}.pkgs.monoprop-mpi;
      };
    in
    {
      overlays.default = monopropOverlay;
    }
    //
      flake-utils.lib.eachSystem
        [
          "x86_64-linux"
          "aarch64-linux"
          "aarch64-darwin"
        ]
        (
          system:
          let
            pkgs = import nixpkgs {
              inherit system;
              overlays = [ monopropOverlay ];
            };
            inherit (pkgs) lib monoprop monoprop-mpi;
            python = pkgs.${defaultPython};

            # monoprop-py312, monoprop-py312-mpi, monoprop-py313, ...
            perPython = lib.mergeAttrsList (
              map (
                version:
                let
                  ps = pkgs."python${version}".pkgs;
                in
                {
                  "monoprop-py${version}" = ps.monoprop;
                  "monoprop-py${version}-mpi" = ps.monoprop-mpi;
                }
              ) pythonVersions
            );

            replEnv = python.withPackages (ps: [
              ps.monoprop
              ps.numpy
            ]);
          in
          {
            packages = {
              default = monoprop;
              inherit monoprop monoprop-mpi;
            }
            // perPython;

            apps.default = {
              type = "app";
              program = "${replEnv}/bin/python";
              meta.description = "Python interpreter with monoprop importable";
            };

            devShells.default = pkgs.callPackage ./nix/devshell.nix { inherit python; };

            # One serial build per interpreter; the MPI variants share everything but
            # the CMake switch and are built explicitly in CI.
            checks = lib.filterAttrs (name: _: !lib.hasSuffix "-mpi" name) perPython;

            formatter = pkgs.nixfmt-tree;
          }
        );
}
