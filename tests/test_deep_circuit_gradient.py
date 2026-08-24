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

"""Gradient accuracy on circuits deep enough for per-layer error to compound.

The reverse pass must recover each layer's pre-layer operator from the forward pass's record
rather than by dividing out the layer's cosine: the division is exact on paper but scales any
error already in the operator by 1/|cos| once per layer, so it compounds with depth even at
angles that look harmless one layer at a time. No angle here is near-singular -- |cos| stays
around 0.5 -- and the depth alone is what breaks a divide-based reverse pass. The cutoff has to be
low enough to prune, or there is no error for the division to amplify and the check passes vacuously:
at 8 qubits and cutoff 4, 8 repetitions put a divide-based reverse pass 12 orders of magnitude out.

Regression test for issue #146.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from monoprop import Circuit, ExpGate, Pauli, PauliOperator, PauliPropagator


def _central_differences(expval, params, eps=1e-5):
    grad = np.zeros_like(params)
    for i in range(params.size):
        plus, minus = params.copy(), params.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (expval(plus) - expval(minus)) / (2 * eps)
    return grad


def _brickwork(num_qubits, layers, observable):
    """`layers` repetitions of (every ZZ term of `observable`, then an X on every qubit)."""
    gates = []
    for k in range(layers):
        gates += [
            ExpGate(PauliOperator({pauli: v}, num_qubits=num_qubits), index=k)
            for pauli, v in observable.terms.items()
        ]
        gates += [
            ExpGate(
                PauliOperator({Pauli("X", i): -1.0}, num_qubits=num_qubits),
                index=layers + k,
            )
            for i in range(num_qubits)
        ]
    return Circuit(gates=gates, system_size=num_qubits)


@pytest.mark.parametrize("layers", [4, 8])
def test_deep_pauli_gradient_matches_finite_differences(layers):
    num_qubits = 8
    cutoff = 4
    observable = PauliOperator(
        {
            Pauli("ZZ", [i, j]): 0.5
            for i, j in itertools.combinations(range(num_qubits), 2)
        },
        num_qubits=num_qubits,
    )

    propagator = PauliPropagator(observable, [], cutoff=cutoff)
    propagator.build_graph(_brickwork(num_qubits, layers, observable))

    params = np.ones(2 * layers)
    gradient = propagator.gradient(params)
    reference = _central_differences(propagator.expectation_value, params)

    # A divide-based reverse pass overshoots by orders of magnitude here, not by a small tolerance.
    assert np.allclose(gradient, reference, atol=1e-6)
