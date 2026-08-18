{
  description = "monoprop: Majorana and Pauli propagation";

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

        # Pinned rather than taken from `pkgs.python3` so that a nixpkgs bump of
        # the default interpreter cannot silently change the ABI of the wheel.
        python = pkgs.python312;

        # nixpkgs still ships 1.0.2; `[build-system] requires` asks for 1.0.3.
        # Drop this once nixpkgs catches up.
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
