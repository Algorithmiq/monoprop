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
import pytest
from msgpack import unpackb

from monoprop.circuit import Circuit
from monoprop.majorana import MajoranaOperator

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import ndarray


@dataclass
class DenseMajoranaArrays:
    """Flat per-monomial arrays for a Majorana gate sequence, mirroring the msgpack layout."""

    initial_state: list[int] | ndarray
    majoranas: list[tuple[int, ...]] | ndarray
    parameters: list[float] | ndarray
    gen_coeffs: list[float] | ndarray
    param_inds: list[int] | ndarray
    num_modes: int

    def to_circuit(self) -> Circuit:
        return Circuit.from_dense_arrays(
            majoranas=self.majoranas,
            gen_coeffs=self.gen_coeffs,
            param_inds=self.param_inds,
            system_size=self.num_modes,
            parameters=self.parameters,
            initial_state=self.initial_state,
        )


class FermionicProblem:
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
    """Load a fermionic test case from a ``.msgpack`` fixture (schema: ``tests/data/README.md``)."""
    with path.open("rb") as fh:
        data = unpackb(fh.read())

    monomial_circuit = DenseMajoranaArrays(
        initial_state=data["hartree_fock"],
        majoranas=[tuple(mono) for mono in data["majoranas"]],
        gen_coeffs=np.asarray(data["gen_coeffs"]),
        param_inds=np.asarray(data["param_inds"], dtype=int),
        parameters=np.asarray(data["parameters"]),
        num_modes=int(data["num_modes"]),
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


MOLECULE_PROBLEMS = [
    pytest.param("lih_fermionic_spin_exact", id="LiH_fermionic_spin"),
]
COMMUTATOR_PROBLEMS = [*MOLECULE_PROBLEMS]
FERMIONIC_PROBLEMS = [
    *MOLECULE_PROBLEMS,
    pytest.param("rx_rz_ry_rz_exact", id="rx_rz_ry_rz"),
    pytest.param("random_exact", id="random_circuit"),
]
ORBITAL_ROTATION_PROBLEMS = [
    pytest.param("S0_8e8o_majoranic_c8", id="S0_8e8o"),
]
