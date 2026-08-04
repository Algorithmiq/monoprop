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

# ruff: noqa: D100, INP001

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

# Run as a script, so sys.path[0] is tools/ regardless of the working directory CMake picks.
from _binding_layout import binding_block, binding_blocks


def python_comment_header_from_text(header_text: str) -> str:
    """Convert a plain-text license header to Python comment lines."""
    lines = header_text.strip().splitlines()
    return "\n".join(f"# {line}" if line else "#" for line in lines)


def _simulator_adapter_source() -> str:
    return """class _SimulatorAdapter:
    __slots__ = ("_core", "_logical_num_modes")

    def __init__(
        self,
        core_type,
        logical_num_modes: int,
        initial_operator: dict[tuple[int, ...], complex],
        cutoff: int,
        initial_state: list[int],
        comm=None,
        schrodinger_cutoff: int | None = None,
        lower_atol: float | None = None,
        upper_atol: float | None = None,
        cutoff_type: str = "length",
        basis_change: list[list[int]] | None = None,
        basis: str = "majorana",
    ) -> None:
        object.__setattr__(self, "_logical_num_modes", logical_num_modes)
        object.__setattr__(
            self,
            "_core",
            core_type(
                initial_operator=initial_operator,
                cutoff=cutoff,
                initial_state=initial_state,
                comm=comm,
                schrodinger_cutoff=schrodinger_cutoff,
                lower_atol=lower_atol,
                upper_atol=upper_atol,
                cutoff_type=cutoff_type,
                basis_change=basis_change,
                logical_num_modes=logical_num_modes,
                basis=basis,
            ),
        )

    @property
    def num_modes(self) -> int:
        return self._logical_num_modes

    def __getattr__(self, name: str):
        return getattr(self._core, name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        setattr(self._core, name, value)

    def __repr__(self) -> str:
        return repr(self._core)
"""


def _cores_table(blocks: list[int], class_name: str) -> str:
    """Emit a dict literal mapping each storage-block width to its core type alias."""
    entries = ",\n".join(
        f"    {block}: _{class_name}{block:03d}Core" for block in blocks
    )
    return f"_CORES: dict[int, type] = {{\n{entries},\n}}"


def _core_dispatch_source(
    *,
    max_logical_num_modes: int,
    blocks: list[int],
    license_header: str,
    class_name: str,
    module_name: str,
) -> str:
    import_line = (
        "(\n"
        + ",\n".join(
            f"    {class_name}{block:03d} as _{class_name}{block:03d}Core"
            for block in blocks
        )
        + "\n)"
    )
    modes_per_block = binding_block(1)  # == MODES_PER_STORAGE_BLOCK
    cores_table = _cores_table(blocks, class_name)

    return f'''{license_header}

"""Typing dispatch for the {class_name}.


This file was automatically generated on {date.today()}. Do *NOT EDIT*. Do *NOT COMMIT*.
"""

# ruff: noqa: I001, ANN001, ANN204

from __future__ import annotations

from functools import partial

from {module_name}._core import {import_line}
from monoprop.exceptions import NumberOfModesInvalidError

_MAX_LOGICAL_NUM_MODES: int = {max_logical_num_modes}
_MODES_PER_STORAGE_BLOCK: int = {modes_per_block}


{_simulator_adapter_source()}


{cores_table}


def _binding_block(n: int) -> int:
    blocks = (n + _MODES_PER_STORAGE_BLOCK - 1) // _MODES_PER_STORAGE_BLOCK
    return max(_MODES_PER_STORAGE_BLOCK, blocks * _MODES_PER_STORAGE_BLOCK)


def dispatch(num_modes: int) -> partial[_SimulatorAdapter]:
    """Return a factory for a {class_name} adapter bound to *num_modes*."""
    if not isinstance(num_modes, int) or num_modes <= 0:
        errmsg = f"Number of Fermionic modes {{num_modes}} invalid. num_modes must be > 0."
        raise NumberOfModesInvalidError(errmsg)
    if num_modes > _MAX_LOGICAL_NUM_MODES:
        errmsg = (
            f"Number of Fermionic modes {{num_modes}} invalid."
            f" num_modes must be <= {{_MAX_LOGICAL_NUM_MODES}}."
            f" Contact monoprop developers if more than {{_MAX_LOGICAL_NUM_MODES}}"
            " Fermionic modes are required."
        )
        raise NumberOfModesInvalidError(errmsg)
    return partial(_SimulatorAdapter, _CORES[_binding_block(num_modes)], num_modes)
'''


def main(
    max_logical_num_modes: int,
    class_name: str,
    module_name: str,
) -> None:
    """Generate dispatch modules for core Python bindings."""
    blocks = binding_blocks(max_logical_num_modes)

    project_root = Path(__file__).resolve().parents[1]
    license_header_path = project_root / ".github" / "license-header.txt"
    license_header = python_comment_header_from_text(license_header_path.read_text())

    Path("_dispatch.py").write_text(
        _core_dispatch_source(
            max_logical_num_modes=max_logical_num_modes,
            blocks=blocks,
            license_header=license_header,
            class_name=class_name,
            module_name=module_name,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dispatch modules.")

    parser.add_argument(
        "--max-num-modes",
        type=int,
        required=True,
        help="Maximum number of simulable Fermionic modes",
    )

    parser.add_argument(
        "--class-name",
        type=str,
        required=True,
        help="Name of the class for the dispatch module",
    )

    parser.add_argument(
        "--module-name",
        type=str,
        required=True,
        help="Name of the module for the dispatch module",
    )

    args = parser.parse_args()

    main(args.max_num_modes, args.class_name, args.module_name)
