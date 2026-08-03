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

"""Utility functions for monoprop module."""

from __future__ import annotations


def jordan_wigner_basis_change(n_qubits: int) -> list[list[int]]:
    """Return the Jordan-Wigner basis change: the ``2 * n_qubits`` Majoranas as slot lists.

    Entry ``k`` is the slot support of the Jordan-Wigner image of Majorana ``m_k``, in the same
    two-slots-per-qubit encoding the engine's native Pauli basis uses (``X_q`` is ``2q``, ``Y_q``
    is ``2q+1``, ``Z_q`` both), so ``m_{2i}`` spans ``2i+1`` slots via its ``Z`` prefix.
    """
    basis = []
    for i in range(n_qubits):
        z_str = list(range(2 * i))
        # m_2i -> X_i Z_i-1 ... Z_0
        basis.append([*z_str, 2 * i])
        # m_2i+1 -> Y_i Z_i-1 ... Z_0
        basis.append([*z_str, 2 * i + 1])
    return basis


def validate_basis_change(
    basis_change: None | list[list[int]],
    num_modes: int,
) -> None:
    """Validate the basis change.

    Raises:
        ValueError: If ``basis_change`` is not ``None`` and does not have ``2 * num_modes`` entries.
    """
    if basis_change is not None and len(basis_change) != 2 * num_modes:
        raise ValueError(
            f"Basis change must have length {2 * num_modes}, but got {len(basis_change)}."
        )
