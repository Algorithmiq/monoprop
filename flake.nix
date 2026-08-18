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
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        scikit-build-core = python.pkgs.scikit-build-core.overridePythonAttrs (_: rec {
          version = "1.0.3";
          src = pkgs.fetchPypi {
            pname = "scikit_build_core";
            inherit version;
            hash = "sha256-pNegWXjuN5dcN3Q1EMiZHi3rzn74OvsKB8DFdv1PFug=";
          };
          doCheck = false;
        });

        monoprop = python.pkgs.callPackage ./nix/monoprop.nix { inherit scikit-build-core; };
        monoprop-mpi = monoprop.override { withMPI = true; };

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
