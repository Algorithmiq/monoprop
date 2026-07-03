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

import numpy as np

from monoprop import (
    MajoranaCircuit,
    MajoranaPropagator,
    jordan_wigner_basis_change,
)
from monoprop.fermi_data import MajoranaOperator


def test_basis_change(serial_comm):
    n_modes = 6
    cutoff = 3
    initial_op = MajoranaOperator([(0,)], [1.0], n_modes)

    sequence = MajoranaCircuit.from_dense_arrays(
        initial_state=[],
        majoranas=[(5,)],
        parameters=[1.0],
        gen_coeffs=[-1.0],
        param_inds=[0],
    )
    circuit = sequence
    pauli_basis = jordan_wigner_basis_change(n_modes)
    mp = MajoranaPropagator(
        initial_op,
        sequence.initial_state,
        cutoff=cutoff,
        basis_change=pauli_basis,
        comm=serial_comm,
    )
    mp.build_graph(circuit)
    tes_op = mp.evolved_operator(circuit)
    act_op = {(0,): np.cos(2 * 1.0)}
    assert len(tes_op) == 1, f"Expected 1 operator, got {len(tes_op)}: {tes_op}"
    for (k1, v1), (k2, v2) in zip(tes_op.items(), act_op.items(), strict=True):
        assert k1 == k2, f"Key mismatch in evolved operator: {k1} vs {k2}"
        assert np.isclose(v1, v2), (
            f"Value mismatch in evolved operator: {k1} -> {v1} vs {k2} -> {v2}"
        )
