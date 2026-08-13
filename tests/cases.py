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


@dataclass(frozen=True)
class ModeEmbedding:
    """A monotone injection of a fixture's modes into the modes of a wider system.

    Padding a problem with unoccupied, ungated modes and relabelling the ones it uses is a canonical
    transformation: the map is strictly increasing, so a sorted Majorana index tuple stays sorted and
    no anticommutation sign appears, and the physics -- exact energy and exact gradient alike -- is
    the same problem's. That is what turns a narrow fixture into a wide one at no cost in provenance:
    no second reference calculation, and no second checked-in blob whose contents are a permutation of
    an existing one.

    Attributes:
        num_modes: Width of the embedding system.
        modes: Where source mode ``m`` lands; strictly increasing and below ``num_modes``.
    """

    num_modes: int
    modes: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject a map that is not a monotone injection into the target system."""
        if any(x >= y for x, y in zip(self.modes, self.modes[1:], strict=False)):
            raise ValueError(f"Embedding modes must be increasing; got {self.modes}.")
        if self.modes and (self.modes[0] < 0 or self.modes[-1] >= self.num_modes):
            raise ValueError(
                f"Embedding modes must lie in [0, {self.num_modes}); got {self.modes}."
            )

    def majorana(self, index: int) -> int:
        """Map one Majorana index of the source system into the embedding system."""
        return 2 * self.modes[index // 2] + (index % 2)


# 260 logical modes store as 288 (nine whole 32-mode blocks, nine 64-bit words). Two regimes at once,
# both of which a released wheel reaches on its own and neither of which any checked-in fixture does:
#
#   * 288 >= monoprop_SPARSE_ROW_MIN_MODES for a wheel (256 without architecture flags), so this is the
#     support-form row store as it ships, not a code path a test forced on.
#   * nine words is past Bitset::kInlineWords (8), so every by-value monomial *spills to the heap*.
#     Nothing else propagates at a spilled width -- bitset_tests.cpp covers spilled bitsets in
#     isolation, which is not the same as running a scan, a fold and a wire exchange on them.
#
# 260 rather than 288 leaves the active window 28 modes short of its storage, exercising the offset
# arithmetic (MSb0: logical modes sit at the HIGH physical positions) rather than the flush case.
#
# The 12 positions are chosen, not spread evenly. With the 28-mode offset, logical L sits at physical
# L + 28, so: 0 and 259 are the two ends of the active window; 3/4, 35/36 and 67/68 straddle physical
# modes 32, 64 and 96, the storage-word boundaries; and 227/228 straddles physical mode 256 -- the
# *inline-to-heap* boundary, where the pair's two bits land in the last inline word and the first
# spilled one.
WIDE_EMBEDDING = ModeEmbedding(
    num_modes=260, modes=(0, 3, 4, 35, 36, 67, 68, 227, 228, 257, 258, 259)
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


def load_problem(
    path: Path, embedding: ModeEmbedding | None = None
) -> FermionicProblem:
    """Load a fermionic test case from a ``.msgpack`` fixture (schema: ``tests/data/README.md``).

    Args:
        path: The fixture to read.
        embedding: Optional [ModeEmbedding][] relabelling the problem into a wider system. The
            reference energy and gradient carry over unchanged -- see the class docstring.
    """
    with path.open("rb") as fh:
        data = unpackb(fh.read())

    if embedding is None:  # the identity embedding: the fixture as it is on disk
        width = int(data["num_modes"])
        embedding = ModeEmbedding(num_modes=width, modes=tuple(range(width)))
    gamma = embedding.majorana

    monomial_circuit = DenseMajoranaArrays(
        initial_state=[embedding.modes[m] for m in data["hartree_fock"]],
        majoranas=[tuple(gamma(i) for i in mono) for mono in data["majoranas"]],
        gen_coeffs=np.asarray(data["gen_coeffs"]),
        param_inds=np.asarray(data["param_inds"], dtype=int),
        parameters=np.asarray(data["parameters"]),
        num_modes=embedding.num_modes,
    )

    ham = data["hamiltonian"]
    terms = {
        tuple(gamma(i) for i in key): complex(real, imag)
        for key, real, imag in zip(ham["keys"], ham["real"], ham["imag"], strict=True)
    }
    quantum_operator = MajoranaOperator(terms, embedding.num_modes)

    return FermionicProblem(
        monomial_circuit=monomial_circuit,
        operator=quantum_operator,
        exact_expval=data["actual_energy"],
        exact_gradient=np.asarray(data["actual_gradient"]),
        n_modes=embedding.num_modes,
    )


def _create_case(
    pth: Path, fname: str, embedding: ModeEmbedding | None = None
) -> FermionicProblem:
    return load_problem(pth / f"{fname}.msgpack", embedding)


class CasesFermionicProblemOrbitalRotations:
    @case(id="S0_8e8o", tags=["molecule", "only_rotate_len_k"])
    def case_s0_8e8o(self, lazy_shared_datadir: Path) -> FermionicProblem:
        return _create_case(lazy_shared_datadir, "S0_8e8o_majoranic_c8")


class CasesFermionicProblem:
    @case(id="LiH_fermionic_spin", tags=["molecule", "has_commutator_data"])
    def case_lih_fermionic_spin(self, lazy_shared_datadir: Path) -> FermionicProblem:
        return _create_case(lazy_shared_datadir, "lih_fermionic_spin_exact")

    @case(id="rx_rz_ry_rz")
    def case_rx_rz_ry_rz(self, lazy_shared_datadir: Path) -> FermionicProblem:
        """A 1q circuit of RX and RZ rotations."""
        return _create_case(lazy_shared_datadir, "rx_rz_ry_rz_exact")

    @case(id="random_circuit")
    def case_random_circuit(self, lazy_shared_datadir: Path) -> FermionicProblem:
        """A 1q circuit of random rotations."""
        return _create_case(lazy_shared_datadir, "random_exact")


class CasesWideFermionicProblem:
    """The fixtures above, at a storage width that runs the support-form row store in a wheel.

    Deliberately a class of its own rather than more entries in [CasesFermionicProblem][]: this is not
    more physics, it is the same physics at a width no checked-in fixture reaches, and every test that
    sweeps that class would otherwise pay for a second, wider copy of every problem it already runs.
    """

    @case(id="LiH_fermionic_spin_wide", tags=["molecule", "wide"])
    def case_lih_fermionic_spin_wide(
        self, lazy_shared_datadir: Path
    ) -> FermionicProblem:
        """LiH, its 12 modes spread across a 260-mode system -- see [WIDE_EMBEDDING][]."""
        return _create_case(
            lazy_shared_datadir, "lih_fermionic_spin_exact", WIDE_EMBEDDING
        )
