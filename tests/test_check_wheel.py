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

"""The post-repair wheel audit run by tools/cibw/repair-wheel.sh."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "cibw" / "check_wheel.py"
_spec = importlib.util.spec_from_file_location("check_wheel", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
check_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_wheel)

Binary = check_wheel.Binary

READELF = """
Dynamic section at offset 0x1 contains 4 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libmpi.so.12]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN/../monoprop.libs:$ORIGIN/lib64]
"""

OTOOL = """
x.so:
Load command 10
          cmd LC_LOAD_DYLIB
      cmdsize 56
         name @rpath/libmpi.40.dylib (offset 24)
Load command 11
          cmd LC_LOAD_DYLIB
      cmdsize 56
         name /usr/lib/libc++.1.dylib (offset 24)
Load command 12
          cmd LC_RPATH
      cmdsize 32
         path @loader_path/lib (offset 12)
Load command 13
          cmd LC_RPATH
      cmdsize 32
         path /opt/homebrew/lib (offset 12)
"""


def test_parse_readelf() -> None:
    assert check_wheel.parse_readelf(READELF) == (
        ("libmpi.so.12", "libstdc++.so.6"),
        ("$ORIGIN/../monoprop.libs", "$ORIGIN/lib64"),
    )


def test_parse_readelf_keeps_empty_entries() -> None:
    text = (
        " 0x000000000000001d (RUNPATH)            Library runpath: [:$ORIGIN/lib64]\n"
    )
    assert check_wheel.parse_readelf(text) == ((), ("", "$ORIGIN/lib64"))


def test_parse_otool() -> None:
    assert check_wheel.parse_otool(OTOOL) == (
        ("@rpath/libmpi.40.dylib", "/usr/lib/libc++.1.dylib"),
        ("@loader_path/lib", "/opt/homebrew/lib"),
    )


def test_mpi_module_with_its_soname_passes() -> None:
    binary = Binary(
        "monoprop/_core_mpi.mpich.abi3.so",
        ("libmpi.so.12", "libc.so.6"),
        ("$ORIGIN/../monoprop.libs",),
    )
    assert check_wheel.audit_binary(binary, macos=False) == []


def test_mpi_module_with_the_wrong_soname_fails() -> None:
    binary = Binary("monoprop/_core_mpi.mpich.abi3.so", ("libmpi.so.40",), ())
    assert check_wheel.audit_binary(binary, macos=False)


def test_mpi_module_without_mpi_fails() -> None:
    binary = Binary("monoprop/lib64/libmonoprop_mpi_openmpi.so", ("libc.so.6",), ())
    assert check_wheel.audit_binary(binary, macos=False)


def test_serial_binary_with_mpi_fails() -> None:
    binary = Binary("monoprop/_core.abi3.so", ("libmpi.so.40",), ())
    assert check_wheel.audit_binary(binary, macos=False)


def test_absolute_rpath_fails() -> None:
    binary = Binary("monoprop/_core.abi3.so", (), ("/opt/mpi/openmpi/lib",))
    assert check_wheel.audit_binary(binary, macos=False)


def test_empty_rpath_entry_fails() -> None:
    binary = Binary("monoprop/_core.abi3.so", (), ("",))
    assert check_wheel.audit_binary(binary, macos=False) == [
        "monoprop/_core.abi3.so: empty RPATH entry (resolves to the current directory)"
    ]


def test_macos_mpi_module_searches_the_mpi_prefixes() -> None:
    binary = Binary(
        "monoprop/_core_mpi.openmpi.abi3.so",
        ("@rpath/libmpi.40.dylib",),
        ("@loader_path/lib", *check_wheel.MACOS_MPI_RPATHS),
    )
    assert check_wheel.audit_binary(binary, macos=True) == []


def test_macos_mpi_module_missing_an_mpi_prefix_fails() -> None:
    # delocate strips absolute RPATHs by default, which left MPI unloadable from Homebrew
    binary = Binary(
        "monoprop/_core_mpi.openmpi.abi3.so",
        ("@rpath/libmpi.40.dylib",),
        ("@loader_path/lib", "/usr/local/lib"),
    )
    assert check_wheel.audit_binary(binary, macos=True) == [
        (
            "monoprop/_core_mpi.openmpi.abi3.so: RPATH lacks the MPI search directories "
            "['/opt/homebrew/lib', '/opt/local/lib']"
        )
    ]


def test_macos_serial_module_may_not_search_homebrew() -> None:
    binary = Binary("monoprop/_core.abi3.so", (), ("/opt/homebrew/lib",))
    assert check_wheel.audit_binary(binary, macos=True)


def test_macos_mpi_dependency_must_go_through_rpath() -> None:
    binary = Binary(
        "monoprop/_core_mpi.openmpi.abi3.so",
        ("/opt/mpi/openmpi/lib/libmpi.40.dylib",),
        (),
    )
    assert check_wheel.audit_binary(binary, macos=True)


def test_bundled_mpi_library_fails() -> None:
    assert check_wheel.audit_bundled(["monoprop.libs/libhwloc-1a2b.so.15"]) == []
    assert check_wheel.audit_bundled(["monoprop.libs/libopen-pal-1a2b.so.80"])
    assert check_wheel.audit_bundled(["monoprop/.dylibs/libmpi.40.dylib"])


TABLE = (
    'ABIS: tuple[str, ...] = ("mpich", "openmpi", )\n'
    'MODULE_FILES: dict[str, str] = {"mpich": "_core_mpi.mpich", "openmpi": "_core_mpi.openmpi", }\n'
)
SHIPPED = [
    "monoprop/_core.abi3.so",
    "monoprop/_core_mpi.mpich.abi3.so",
    "monoprop/_core_mpi.openmpi.abi3.so",
]


def test_abi_table_matching_the_files_passes() -> None:
    assert check_wheel.audit_abi_table(SHIPPED, TABLE, ("mpich", "openmpi")) == []


def test_abi_table_with_a_missing_module_fails() -> None:
    assert check_wheel.audit_abi_table(SHIPPED[:2], TABLE, ("mpich", "openmpi"))


def test_abi_table_with_unexpected_abis_fails() -> None:
    assert check_wheel.audit_abi_table(SHIPPED, TABLE, ("mpich",))


def test_audit_required_files_passes_when_stub_and_marker_present() -> None:
    paths = ["monoprop/__init__.py", "monoprop/_core.pyi", "monoprop/py.typed"]
    assert check_wheel.audit_required_files(paths) == []


def test_audit_required_files_reports_each_missing_file() -> None:
    errors = check_wheel.audit_required_files(["monoprop/__init__.py"])
    assert len(errors) == 2
    assert any("monoprop/_core.pyi" in error for error in errors)
    assert any("monoprop/py.typed" in error for error in errors)
    errors = check_wheel.audit_required_files(["monoprop/py.typed"])
    assert len(errors) == 1
    assert "monoprop/_core.pyi" in errors[0]
