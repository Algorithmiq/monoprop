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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


def normalize_parameters(
    parameters: list[float] | np.ndarray | None,
    parameter_mapping: list[int] | np.ndarray | None,
    gen_coeffs: list[float] | np.ndarray | None,
) -> tuple[list[float] | np.ndarray, list[int] | np.ndarray, list[float] | np.ndarray]:
    """Convert None parameters to empty lists."""
    parameters = [] if parameters is None else parameters
    parameter_mapping = [] if parameter_mapping is None else parameter_mapping
    gen_coeffs = [] if gen_coeffs is None else gen_coeffs
    return parameters, parameter_mapping, gen_coeffs


def wrap_functional_call(
    fn: Callable,
    transform: Callable | None = None,
) -> Callable:
    """Wrap simulator functionals so ``None`` parameters map to empty lists."""

    def _fn(
        parameters: list[float] | np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray] | np.ndarray:
        result = fn([] if parameters is None else parameters)
        if transform is None:
            return result
        return transform(result)

    return _fn


def jordan_wigner_basis_change(n_qubits: int) -> list[list[int]]:
    """Generate a basis change for Jordan-Wigner representation.

    This function returns a list of lists, where each inner list represents a basis vector in the Jordan-Wigner
    representation in terms of Majoranas.

    Args:
        n_qubits: The number of qubits.

    Returns:
        A list of lists representing the basis change.
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

    Args:
        basis_change: The basis change to validate.
        num_modes: The number of modes.

    Raises:
        ValueError: If the basis change is invalid.
    """
    if basis_change is not None and len(basis_change) != 2 * num_modes:
        raise ValueError(
            f"Basis change must have length {2 * num_modes}, but got {len(basis_change)}."
        )
