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

# ruff: noqa: INP001

"""Storage-block layout shared by the binding and dispatch generators.

`MonomialPropagator<NumModes>` is instantiated only at multiples of 32, because a `Monomial<N>`
is a `Bitset<2*N>` over 64-bit words: 32 modes is exactly one word. A logical mode count is
therefore served by the next block up, and the two generators have to agree on that rounding --
`generate-binders.py` decides which templates exist, `generate-dispatch.py` decides which one
each logical mode count is routed to. A disagreement routes Python at a template that was never
instantiated, so the rule lives here rather than in both.
"""

from __future__ import annotations

MODES_PER_STORAGE_BLOCK = 32


def binding_block(logical_num_modes: int) -> int:
    """Return the storage width serving a logical mode count.

    Args:
        logical_num_modes: number of logical Fermionic modes (or qubits).

    Returns:
        The smallest instantiated storage width that holds `logical_num_modes`.
    """
    blocks = (
        logical_num_modes + MODES_PER_STORAGE_BLOCK - 1
    ) // MODES_PER_STORAGE_BLOCK
    return max(MODES_PER_STORAGE_BLOCK, blocks * MODES_PER_STORAGE_BLOCK)


def binding_blocks(max_logical_num_modes: int) -> list[int]:
    """Return every storage width needed to cover 1..max_logical_num_modes, ascending.

    Args:
        max_logical_num_modes: the largest logical mode count the bindings must serve.

    Returns:
        The instantiated storage widths, from 32 up to `binding_block(max_logical_num_modes)`.
    """
    return list(
        range(
            MODES_PER_STORAGE_BLOCK,
            binding_block(max_logical_num_modes) + 1,
            MODES_PER_STORAGE_BLOCK,
        )
    )
