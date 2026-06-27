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
from textwrap import indent


def python_comment_header_from_text(header_text: str) -> str:
    """Convert a plain-text license header to Python comment lines."""
    lines = header_text.strip().splitlines()
    return "\n".join(f"# {line}" if line else "#" for line in lines)


def binding_block(logical_num_modes: int) -> int:
    """Return the storage-width binding block for a logical mode count."""
    return max(32, ((logical_num_modes + 31) // 32) * 32)


def binding_blocks(max_logical_num_modes: int) -> list[int]:
    """Return the storage-width binding blocks needed to cover max_logical_num_modes."""
    max_storage_modes = binding_block(max_logical_num_modes)
    return list(range(32, max_storage_modes + 1, 32))


def _simulator_adapter_source() -> str:
    return """class _SimulatorAdapter:
    __slots__ = ("_core", "_logical_num_modes")

    def __init__(
        self,
        core_type,
        logical_num_modes: int,
        initial_operator: dict[tuple[int, ...], complex],
        cutoff: int,
        slater_determinant: list[int],
        comm=None,
        schrodinger_cutoff: int | None = None,
        lower_atol: float | None = None,
        upper_atol: float | None = None,
        cutoff_type: str = "length",
        basis_change: list[list[int]] | None = None,
    ) -> None:
        object.__setattr__(self, "_logical_num_modes", logical_num_modes)
        object.__setattr__(
            self,
            "_core",
            core_type(
                initial_operator=initial_operator,
                cutoff=cutoff,
                slater_determinant=slater_determinant,
                comm=comm,
                schrodinger_cutoff=schrodinger_cutoff,
                lower_atol=lower_atol,
                upper_atol=upper_atol,
                cutoff_type=cutoff_type,
                basis_change=basis_change,
                logical_num_modes=logical_num_modes,
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


def _class_defs(
    exact_modes: list[int], class_prefix: str, core_alias_prefix: str
) -> str:
    class_defs = []
    for mode in exact_modes:
        block = binding_block(mode)
        class_defs.append(
            f"""class {class_prefix}{mode:03d}(_SimulatorAdapter):
    def __init__(
        self,
        initial_operator: dict[tuple[int, ...], complex],
        cutoff: int,
        slater_determinant: list[int],
        comm=None,
        schrodinger_cutoff: int | None = None,
        lower_atol: float | None = None,
        upper_atol: float | None = None,
        cutoff_type: str = "length",
        basis_change: list[list[int]] | None = None,
    ) -> None:
        super().__init__(
            {core_alias_prefix}{block:03d}Core,
            {mode},
            initial_operator,
            cutoff,
            slater_determinant,
            comm,
            schrodinger_cutoff,
            lower_atol,
            upper_atol,
            cutoff_type,
            basis_change,
        )
"""
        )
    return "\n\n".join(class_defs)


def _dispatch_cases(exact_modes: list[int], class_prefix: str) -> str:
    return indent(
        "\n".join(
            f"case {mode:d}:\n    cls = {class_prefix}{mode:03d}"
            for mode in exact_modes
        ),
        " " * 8,
    )


def _core_dispatch_source(
    *,
    max_logical_num_modes: int,
    blocks: list[int],
    exact_modes: list[int],
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
    classes = _class_defs(exact_modes, class_name, f"_{class_name}")
    cases = _dispatch_cases(exact_modes, class_name)

    return f'''{license_header}

"""Typing dispatch for the {class_name}.


This file was automatically generated on {date.today()}. Do *NOT EDIT*. Do *NOT COMMIT*.
"""

# ruff: noqa: I001, ANN001, ANN204, C901, PLR0912

from __future__ import annotations

from {module_name}._core import {import_line}
from monoprop.exceptions import NumberOfModesInvalidError


{_simulator_adapter_source()}


{classes}


def dispatch(num_modes: int) -> type[_SimulatorAdapter]:
    """Dispatches the appropriate {class_name} class based on number of Fermionic modes."""
    match num_modes:
        case n if n <= 0:
            errmsg = f"Number of Fermionic modes {{n}} invalid. num_modes must be > 0."
            raise NumberOfModesInvalidError(errmsg)
{cases}
        case n if n > {max_logical_num_modes}:
            errmsg = f"Number of Fermionic modes {{n}} invalid. num_modes must be <= {max_logical_num_modes}."
            raise NumberOfModesInvalidError(errmsg)

    return cls
'''


def main(
    max_logical_num_modes: int,
    class_name: str,
    module_name: str,
) -> None:
    """Generate dispatch modules for core Python bindings."""
    blocks = binding_blocks(max_logical_num_modes)
    exact_modes = list(range(1, max_logical_num_modes + 1))

    project_root = Path(__file__).resolve().parents[1]
    license_header_path = project_root / ".github" / "license-header.txt"
    license_header = python_comment_header_from_text(license_header_path.read_text())

    Path("_dispatch.py").write_text(
        _core_dispatch_source(
            max_logical_num_modes=max_logical_num_modes,
            blocks=blocks,
            exact_modes=exact_modes,
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
