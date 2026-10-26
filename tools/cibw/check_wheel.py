# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: INP001, T201

"""Audit a repaired monoprop wheel: MPI linkage, RPATHs, bundled libraries and the ABI table.

Each MPI extension and MPI libmonoprop must depend on exactly its ABI family's MPI soname, left
unresolved for the user's MPI; nothing else may depend on MPI; no binary may carry an absolute
RPATH, except the macOS MPI search directories on MPI binaries; no MPI library may be bundled;
the type stub and ``py.typed`` must be present.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import NamedTuple

MPI_SONAMES: dict[str, tuple[str, str]] = {
    "mpich": ("libmpi.so.12", "libmpi.12.dylib"),
    "openmpi": ("libmpi.so.40", "libmpi.40.dylib"),
    "mpiabi": ("libmpi_abi.so.1", "libmpi_abi.1.dylib"),
}
ALL_MPI_SONAMES = {name for names in MPI_SONAMES.values() for name in names}
MACOS_MPI_RPATHS = ("/usr/local/lib", "/opt/homebrew/lib", "/opt/local/lib")
BUNDLED_MPI_PREFIXES = (
    "libmpi",
    "libpmpi",
    "libopen-",
    "libpmix",
    "libprrte",
    "libfabric",
    "libuc",
    "libevent",
)
_MPI_BINARY = re.compile(
    r"^(?:_core_mpi\.|libmonoprop_mpi_)(?P<abi>mpich|openmpi|mpiabi)[._]"
)
_RELATIVE_RPATH = ("$ORIGIN", "@loader_path")
REQUIRED_FILES = ("monoprop/_core.pyi", "monoprop/py.typed")


class Binary(NamedTuple):
    """A shared object in the wheel and what its dynamic section records."""

    path: str
    needed: tuple[str, ...]
    rpaths: tuple[str, ...]


def parse_readelf(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (NEEDED entries, RPATH/RUNPATH entries) from ``readelf -d`` output."""
    needed = tuple(re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", text))
    rpaths: list[str] = []
    for value in re.findall(
        r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[([^\]]*)\]", text
    ):
        # An empty entry is kept: the loader resolves it to the current directory. An empty
        # tag value has no entries at all, so it is skipped.
        if value:
            rpaths.extend(value.split(":"))
    return needed, tuple(rpaths)


def parse_otool(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (LC_LOAD_DYLIB names, LC_RPATH paths) from ``otool -l`` output."""
    needed: list[str] = []
    rpaths: list[str] = []
    command = None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "cmd":
            command = fields[1]
        elif (
            len(fields) >= 2
            and fields[0] == "name"
            and command in {"LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB"}
        ):
            needed.append(fields[1])
        elif len(fields) >= 2 and fields[0] == "path" and command == "LC_RPATH":
            rpaths.append(fields[1])
    return tuple(needed), tuple(rpaths)


def audit_binary(binary: Binary, *, macos: bool) -> list[str]:
    """Return one message per rule ``binary`` breaks."""
    errors: list[str] = []
    match = _MPI_BINARY.match(Path(binary.path).name)
    abi = match["abi"] if match else None
    mpi_deps = [dep for dep in binary.needed if Path(dep).name in ALL_MPI_SONAMES]
    if abi is None:
        if mpi_deps:
            errors.append(f"{binary.path}: serial binary depends on MPI: {mpi_deps}")
    else:
        want = MPI_SONAMES[abi][1 if macos else 0]
        if [Path(dep).name for dep in mpi_deps] != [want]:
            errors.append(
                f"{binary.path}: needs {want} and no other MPI library, has {mpi_deps}"
            )
        if macos and any(not dep.startswith("@rpath/") for dep in mpi_deps):
            errors.append(
                f"{binary.path}: MPI must be loaded through @rpath: {mpi_deps}"
            )
    allowed = MACOS_MPI_RPATHS if macos and abi else ()
    for rpath in binary.rpaths:
        if rpath == "":
            errors.append(
                f"{binary.path}: empty RPATH entry (resolves to the current directory)"
            )
        elif not rpath.startswith(_RELATIVE_RPATH) and rpath not in allowed:
            errors.append(
                f"{binary.path}: RPATH entry {rpath!r} is not relative to the binary"
            )
    return errors


def audit_bundled(paths: list[str]) -> list[str]:
    """Return one message per MPI library vendored into the wheel."""
    return [
        f"{path}: MPI library bundled into the wheel"
        for path in paths
        if Path(path).name.startswith(BUNDLED_MPI_PREFIXES)
    ]


def audit_required_files(paths: list[str]) -> list[str]:
    """Return one message per required file missing from the wheel.

    Stub generation at install time is non-fatal, so a failure there only shows up as a missing
    ``_core.pyi``.
    """
    return [
        f"{required}: missing from the wheel"
        for required in REQUIRED_FILES
        if required not in paths
    ]


def _literal(source: str, name: str) -> object:
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
    msg = f"_mpi_abis.py defines no {name}"
    raise ValueError(msg)


def audit_abi_table(
    paths: list[str], source: str, expected: tuple[str, ...]
) -> list[str]:
    """Check ``_mpi_abis.py`` against the expected ABIs and the ``_core_mpi.*`` files shipped."""
    abis = tuple(_literal(source, "ABIS"))
    module_files = dict(_literal(source, "MODULE_FILES"))
    shipped = {
        ".".join(Path(path).name.split(".")[:2])
        for path in paths
        if Path(path).parent.name == "monoprop"
        and Path(path).name.startswith("_core_mpi.")
    }
    errors: list[str] = []
    if abis != expected:
        errors.append(f"_mpi_abis.py: ABIS is {abis}, expected {expected}")
    if set(module_files) != set(abis):
        errors.append(
            f"_mpi_abis.py: MODULE_FILES keys {sorted(module_files)} differ from ABIS {abis}"
        )
    if set(module_files.values()) != shipped:
        errors.append(
            f"_mpi_abis.py: MODULE_FILES {sorted(module_files.values())} but the wheel ships {sorted(shipped)}"
        )
    return errors


def _read_binary(root: Path, path: Path, *, macos: bool) -> Binary:
    command = ["otool", "-l", str(path)] if macos else ["readelf", "-d", str(path)]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout  # noqa: S603
    needed, rpaths = parse_otool(output) if macos else parse_readelf(output)
    return Binary(path.relative_to(root).as_posix(), needed, rpaths)


def main() -> int:
    """Audit the wheel named on the command line; print findings and return 1 if any."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--expect-abis", required=True, help="comma-separated, in build order"
    )
    args = parser.parse_args()
    expected = tuple(abi for abi in args.expect_abis.split(",") if abi)
    macos = sys.platform == "darwin"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.wheel) as wheel:
            wheel.extractall(root)
            paths = wheel.namelist()
        binaries = [
            root / path
            for path in paths
            if path.endswith((".so", ".dylib")) or ".so." in Path(path).name
        ]
        errors = [
            error
            for path in binaries
            for error in audit_binary(
                _read_binary(root, path, macos=macos), macos=macos
            )
        ]
        errors += audit_bundled(paths)
        errors += audit_required_files(paths)
        errors += audit_abi_table(
            paths, (root / "monoprop" / "_mpi_abis.py").read_text(), expected
        )

    for error in errors:
        print(error, file=sys.stderr)
    print(f"{args.wheel.name}: {len(binaries)} binaries, {len(errors)} finding(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
