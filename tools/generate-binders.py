#!/usr/bin/env python3

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

"""Generate C++ binding batches for core nanobind modules."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from textwrap import indent

# Run as a script, so sys.path[0] is tools/ regardless of the working directory CMake picks.
from _binding_layout import binding_blocks

FN_SIGNATURE_TEMPLATE = """/**
 * @brief Binds the {binding_label} class to Python for NumModes in [{modes}].
 *
 * @param mod The module to which the class will be bound.
 */
auto bind_up_to_{end:03d}(nanobind::module_ &mod) -> void;\n\n"""

HEADER = """/* This file was automatically generated on {date}. Do *NOT EDIT*. Do *NOT COMMIT*. */
#pragma once


#include <nanobind/nanobind.h>


namespace monoprop {{
namespace bindings {{
{body}
}} // namespace bindings
}} // namespace monoprop
"""

CPP = """/* This file was automatically generated on {date}. Do *NOT EDIT*. Do *NOT COMMIT*. */
#include "generated/bind.h"

#include <nanobind/nanobind.h>

#include "binder.h"

namespace monoprop::bindings {{
auto bind_up_to_{end:03d}(nanobind::module_ &mod) -> void {{
{body}
}}
}} // namespace monoprop::bindings
"""


def main(
    max_logical_num_modes: int,
    batch_size: int,
    output_dir: Path,
    binding_kind: str,
) -> None:
    """Generate binding code.

    Args:
        max_logical_num_modes: maximum number of simulable Fermionic modes.
        batch_size: number of modes per batch.
        output_dir: output directory.
        binding_kind: selects whether to generate core class registrations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_binding in output_dir.glob("bind_up_to_*.cpp"):
        old_binding.unlink()

    binding_fn = {
        "core": "detail::bind_monomial_propagator",
    }[binding_kind]
    binding_label = {
        "core": "MonomialPropagator",
    }[binding_kind]

    call = ""
    blocks = binding_blocks(max_logical_num_modes)
    num_batches = (len(blocks) + batch_size - 1) // batch_size
    body_h = ""
    for n in range(num_batches):
        s = n * batch_size
        e = s + batch_size
        batch = blocks[s:e]
        modes = ", ".join(f"{i}" for i in batch)
        end = batch[-1]
        body_h += FN_SIGNATURE_TEMPLATE.format(
            binding_label=binding_label,
            modes=modes,
            end=end,
        )

        body_cpp = ""
        body_cpp += indent(
            "\n".join(f"{binding_fn}<{_}>(mod);" for _ in batch),
            " " * 4,
        )
        with (output_dir / f"bind_up_to_{end:03d}.cpp").open("w") as f:
            f.write(CPP.format(date=date.today(), end=end, body=body_cpp))
        call += f"    bindings::bind_up_to_{end:03d}(m);\n"

    with (output_dir / "bind.h").open("w") as f:
        f.write(HEADER.format(date=date.today(), body=body_h))

    with (output_dir / "call.txt").open("w") as f:
        f.write(call)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binding code.")

    parser.add_argument(
        "--max-num-modes",
        type=int,
        required=True,
        help="Maximum number of simulable Fermionic modes",
    )

    parser.add_argument(
        "--batch-size", type=int, required=True, help="Number of modes per batch"
    )

    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )

    parser.add_argument(
        "--binding-kind",
        type=str,
        choices=["core"],
        required=True,
        help="Whether to generate core bindings",
    )

    args = parser.parse_args()

    main(args.max_num_modes, args.batch_size, args.output_dir, args.binding_kind)
