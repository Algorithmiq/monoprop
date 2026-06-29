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

from pytest_cases import case

from monoprop import MPData
from monoprop.monomial_data import MonomialCircuit, MonomialOperator

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class MPFermData(NamedTuple):
    """Data class for Monomial Propagator Fermionic data."""

    majoranas: list[tuple[int, ...]]
    gen_coeffs: np.ndarray
    param_inds: np.ndarray
    parameters: np.ndarray


class SplitOrbitalRotations(NamedTuple):
    """Data class for Monomial Propagator Fermionic data."""

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
        monomial_circuit: MonomialCircuit,
        initial_state: list[int],
        operator: MonomialOperator,
        exact_expval: float,
        exact_gradient: np.ndarray,
        n_modes: int,
        fermi_com_data: list[list[tuple[tuple[int, ...], complex]]] | None = None,
        exact_commutator_gradient: np.ndarray | None = None,
    ) -> None:
        self.monomial_circuit = monomial_circuit
        self.initial_state = initial_state
        self.operator = operator
        self.exact_expval = exact_expval
        self.exact_gradient = exact_gradient
        self.n_modes = n_modes
        self.fermi_com_data = fermi_com_data
        self.exact_commutator_gradient = exact_commutator_gradient

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


def _create_case(pth: Path, fname: str) -> FermionicProblem:
    monoprop_data = MPData.from_msgpack(filepath=pth / f"{fname}.msgpack")
    initial_state = monoprop_data.hartree_fock
    monomial_circuit = MonomialCircuit(
        majoranas=monoprop_data.majoranas,
        gen_coeffs=monoprop_data.gen_coeffs,
        param_inds=monoprop_data.param_inds,
        parameters=monoprop_data.parameters,
    )
    quantum_operator = MonomialOperator.from_dict(
        monoprop_data.fermionic_hamiltonian, monoprop_data.num_modes
    )
    return FermionicProblem(
        monomial_circuit=monomial_circuit,
        initial_state=initial_state,
        operator=quantum_operator,
        exact_expval=monoprop_data.actual_energy,
        exact_gradient=monoprop_data.actual_gradient,
        n_modes=monoprop_data.num_modes,
        fermi_com_data=monoprop_data.fermionic_pool,
        exact_commutator_gradient=monoprop_data.commutator_gradient,
    )


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
