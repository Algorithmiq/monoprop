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
      monopropOverlay = final: _prev: {
        monoprop =
          let
            python = final.python312;
            scikit-build-core = python.pkgs.scikit-build-core.overridePythonAttrs (_: rec {
              version = "1.0.3";
              src = final.fetchPypi {
                pname = "scikit_build_core";
                inherit version;
                hash = "sha256-pNegWXjuN5dcN3Q1EMiZHi3rzn74OvsKB8DFdv1PFug=";
              };
              doCheck = false;
            });
            nanobind = python.pkgs.callPackage ./nix/nanobind.nix { };
            nanobind-backend = python.pkgs.callPackage ./nix/nanobind-backend.nix {
              inherit nanobind;
            };
          in
          python.pkgs.callPackage ./nix/monoprop.nix {
            inherit scikit-build-core nanobind nanobind-backend;
          };

        monoprop-mpi = final.monoprop.override { withMPI = true; };
      };
    in
    {
      overlays.default = monopropOverlay;
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ monopropOverlay ];
        };
        python = pkgs.python312;
        inherit (pkgs) monoprop monoprop-mpi;

        replEnv = python.withPackages (ps: [
          monoprop
          ps.numpy
        ]);
      in
      {
        packages = {
          default = monoprop;
          inherit monoprop monoprop-mpi;
        };

        apps.default = {
          type = "app";
          program = "${replEnv}/bin/python";
          meta.description = "Python interpreter with monoprop importable";
        };

        devShells.default = pkgs.callPackage ./nix/devshell.nix { inherit python; };

        checks = {
          inherit monoprop;
        };

        formatter = pkgs.nixfmt-tree;
      }
    );
}
