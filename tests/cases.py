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

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from msgpack import unpackb
from pytest_cases import case

from monoprop.majorana_data import MajoranaOperator, MajoranaSequence

if TYPE_CHECKING:
    from pathlib import Path


class SplitOrbitalRotations(NamedTuple):
    """Data class for Majorana propagator fermionic test data."""

    majs: list[tuple[int, ...]]
    param_inds: np.ndarray
    gen_coeffs: np.ndarray
    parameters: np.ndarray
    majs_orb: list[tuple[int, ...]]
    param_inds_orb: np.ndarray
    gen_coeffs_orb: np.ndarray
    parameters_orb: np.ndarray


class FermionicProblem:
    """Data class for Fermionic problems."""

    def __init__(
        self,
        monomial_circuit: MajoranaSequence,
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

    def split_only_rotate_len_k(self) -> SplitOrbitalRotations:
        """Split the Majorana operators into non-orbital and orbital rotation parts."""
        majoranas = self.monomial_circuit.majoranas
        param_inds = self.monomial_circuit.param_inds
        gen_coeffs = self.monomial_circuit.gen_coeffs
        parameters = self.monomial_circuit.parameters

        # Find split index
        idx = None
        lens = [len(maj) for maj in majoranas]
        for i in range(len(lens)):
            if all(length == 2 for length in lens[i:]):
                idx = i
                break

        # Split the lists at the appropriate index
        if idx is None:
            raise ValueError("No single excitations found")

        m1, m2 = majoranas[:idx], majoranas[idx:]
        pi1, pi2 = param_inds[:idx], param_inds[idx:] - idx  # adjust indices
        gc1, gc2 = gen_coeffs[:idx], gen_coeffs[idx:]
        p1, p2 = parameters[: param_inds[idx]], parameters[param_inds[idx] :]
        return SplitOrbitalRotations(m1, pi1, gc1, p1, m2, pi2, gc2, p2)


def load_problem(path: Path) -> FermionicProblem:
    """Load a fermionic test case from a minimal-schema msgpack file.

    See ``tests/data/README.md`` for the on-disk schema.

    Args:
        path: Path to the ``.msgpack`` fixture.

    Returns:
        A :class:`FermionicProblem` built directly from the public API
        (:class:`MajoranaSequence` and :class:`MajoranaOperator`).
    """
    with path.open("rb") as fh:
        data = unpackb(fh.read())

    monomial_circuit = MajoranaSequence(
        initial_state=data["hartree_fock"],
        majoranas=[tuple(maj) for maj in data["majoranas"]],
        gen_coeffs=np.asarray(data["gen_coeffs"]),
        param_inds=np.asarray(data["param_inds"], dtype=int),
        parameters=np.asarray(data["parameters"]),
    )

    ham = data["hamiltonian"]
    terms = {
        tuple(key): complex(real, imag)
        for key, real, imag in zip(ham["keys"], ham["real"], ham["imag"], strict=True)
    }
    quantum_operator = MajoranaOperator.from_dict(terms, data["num_modes"])

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
