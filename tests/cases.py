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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from msgpack import unpackb
from pytest_cases import case

from monoprop.circuit import Circuit
from monoprop.majorana import MajoranaOperator

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import ndarray


@dataclass
class DenseMajoranaArrays:
    """Flat per-monomial arrays for a Majorana gate sequence (test transport).

    Mirrors the on-disk msgpack-fixture layout; :meth:`to_circuit` groups it into a
    :class:`~monoprop.circuit.Circuit` via
    :meth:`~monoprop.circuit.Circuit.from_dense_arrays`.
    """

    initial_state: list[int] | ndarray
    majoranas: list[tuple[int, ...]] | ndarray
    parameters: list[float] | ndarray
    gen_coeffs: list[float] | ndarray
    param_inds: list[int] | ndarray

    def to_circuit(self) -> Circuit:
        """Group the dense arrays into a :class:`~monoprop.circuit.Circuit`."""
        return Circuit.from_dense_arrays(
            majoranas=self.majoranas,
            gen_coeffs=self.gen_coeffs,
            param_inds=self.param_inds,
            parameters=self.parameters,
            initial_state=self.initial_state,
        )


class FermionicProblem:
    """Data class for Fermionic problems."""

    def __init__(
        self,
        monomial_circuit: DenseMajoranaArrays,
        operator: MajoranaOperator,
        exact_expval: float,
        exact_gradient: np.ndarray,
        n_modes: int,
    ) -> None:
        self.monomial_circuit = monomial_circuit
        self.operator = operator
        self.exact_expval = exact_expval
        self.exact_gradient = exact_gradient
        self.n_modes = n_modes


def load_problem(path: Path) -> FermionicProblem:
    """Load a fermionic test case from a minimal-schema msgpack file.

    See ``tests/data/README.md`` for the on-disk schema.

    Args:
        path: Path to the ``.msgpack`` fixture.

    Returns:
        A :class:`FermionicProblem` built from the dense msgpack arrays and
        :class:`MajoranaOperator`.
    """
    with path.open("rb") as fh:
        data = unpackb(fh.read())

    monomial_circuit = DenseMajoranaArrays(
        initial_state=data["hartree_fock"],
        majoranas=[tuple(mono) for mono in data["majoranas"]],
        gen_coeffs=np.asarray(data["gen_coeffs"]),
        param_inds=np.asarray(data["param_inds"], dtype=int),
        parameters=np.asarray(data["parameters"]),
    )

    ham = data["hamiltonian"]
    terms = {
        tuple(key): complex(real, imag)
        for key, real, imag in zip(ham["keys"], ham["real"], ham["imag"], strict=True)
    }
    quantum_operator = MajoranaOperator(terms, data["num_modes"])

    return FermionicProblem(
        monomial_circuit=monomial_circuit,
        operator=quantum_operator,
        exact_expval=data["actual_energy"],
        exact_gradient=np.asarray(data["actual_gradient"]),
        n_modes=data["num_modes"],
    )


def _create_case(pth: Path, fname: str) -> FermionicProblem:
    return load_problem(pth / f"{fname}.msgpack")


class CasesFermionicProblemOrbitalRotations:
    """
    A class to represent a fermionic problem case for testing purposes.
    """

    @case(id="S0_8e8o", tags=["molecule", "only_rotate_len_k"])
    def case_s0_8e8o(self, lazy_shared_datadir: Path) -> FermionicProblem:
        """
        A test case for the S0_8e8o fermionic problem.
        """
        return _create_case(lazy_shared_datadir, "S0_8e8o_majoranic_c8")


class CasesFermionicProblem:
    """
    A class to represent a fermionic problem case for testing purposes.
    """

    @case(id="LiH_fermionic_spin", tags=["molecule", "has_commutator_data"])
    def case_lih_fermionic_spin(self, lazy_shared_datadir: Path) -> FermionicProblem:
        """
        A test case for the LiH fermionic problem.
        """
        return _create_case(lazy_shared_datadir, "lih_fermionic_spin_exact")

    @case(id="rx_rz_ry_rz")
    def case_rx_rz_ry_rz(self, lazy_shared_datadir: Path) -> FermionicProblem:
        """
        A simple test case for a 1q circuit containing a RX and RZ rotations.
        """
        return _create_case(lazy_shared_datadir, "rx_rz_ry_rz_exact")

    @case(id="random_circuit")
    def case_random_circuit(self, lazy_shared_datadir: Path) -> FermionicProblem:
        """
        A simple test case for a 1q circuit containing a random rotations.
        """
        return _create_case(lazy_shared_datadir, "random_exact")
