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

"""Coverage for the cutoff basis change.

The front-ends construct with ``basis_change=None``, so the only way in is the engine's
``basis_change`` setter, which writes straight through to the cutoff regeneration.
"""

from __future__ import annotations

import numpy as np
import pytest

from monoprop import (
    Circuit,
    MajoranaOperator,
    MajoranaPropagator,
    PauliOperator,
    PauliPropagator,
    jordan_wigner_basis_change,
)

N_MODES = 6


def _propagator(comm=None) -> MajoranaPropagator:
    return MajoranaPropagator(
        MajoranaOperator({(0,): 1.0}, N_MODES), [], cutoff=N_MODES // 2, comm=comm
    )


def test_basis_change(serial_comm) -> None:
    """A Jordan-Wigner cutoff basis reproduces the exact single-rotation result.

    serial_comm because evolved_operator() is rank-local: on COMM_WORLD the single term lives on
    whichever rank owns its hash partition.
    """
    propagator = _propagator(serial_comm)
    propagator._simulator.basis_change = jordan_wigner_basis_change(N_MODES)
    propagator.propagate(
        Circuit.from_dense_arrays(
            majoranas=[(5,)],
            gen_coeffs=[-1.0],
            param_inds=[0],
            parameters=[1.0],
            system_size=N_MODES,
        )
    )

    evolved = propagator.evolved_operator()

    assert list(evolved.terms) == [(0,)]
    assert np.isclose(evolved.terms[(0,)].real, np.cos(2 * 1.0))


@pytest.mark.parametrize(
    ("basis_change", "match"),
    [
        ([[0]], "exactly 2\\*num_modes"),
        ([[0]] * (2 * N_MODES - 1), "exactly 2\\*num_modes"),
        ([[2 * N_MODES]] * (2 * N_MODES), "out of range"),
    ],
)
def test_malformed_basis_change_rejected(basis_change, match) -> None:
    """Validation lives in the engine setter, so a short table or one naming a slot outside the
    system must raise rather than index out of bounds."""
    with pytest.raises((ValueError, RuntimeError), match=match):
        _propagator()._simulator.basis_change = basis_change


def test_pauli_propagator_rejects_basis_change_and_length_cutoff() -> None:
    """A Pauli propagator must reject post-construction the configurations its constructor
    rejects."""
    propagator = PauliPropagator(PauliOperator({"ZZ": 1.0}, num_qubits=2), [], cutoff=2)

    with pytest.raises(ValueError, match="does not accept a basis_change"):
        propagator._simulator.basis_change = jordan_wigner_basis_change(2)
    with pytest.raises(ValueError, match="requires cutoff_type == Support"):
        propagator._simulator.cutoff_type = "length"
