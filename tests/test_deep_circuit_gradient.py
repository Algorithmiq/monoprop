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

"""Gradient accuracy where the reverse pass cannot simply divide the layer cosine back out.

Two ways that division goes wrong. Depth: it is exact on paper, but a coefficient divided at more
layers than its rotation partner is mis-scaled against it, and the rotation then drops the
partner's low bits -- no angle here is near-singular, |cos| stays around 0.5, and 8 repetitions
alone put a divide-based reverse pass 12 orders of magnitude out. Vanishing cosine: theta = pi/4 at
unit generator coefficient gives cos ~ 6.1e-17, so 1/cos is ~1.6e+16 and the gradient comes back
finite and wrong by more than 100%. The repository's own kicked-Ising notebook uses that angle.

The cutoff must actually prune, or there is nothing for the division to amplify and the checks pass
vacuously. Regression tests for issues #146 and #190.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from monoprop import Circuit, ExpGate, Pauli, PauliOperator, PauliPropagator

NUM_QUBITS = 8
CUTOFF = 4


def _central_differences(expval, params, eps=1e-5):
    grad = np.zeros_like(params)
    for i in range(params.size):
        plus, minus = params.copy(), params.copy()
        plus[i] += eps
        minus[i] -= eps
        grad[i] = (expval(plus) - expval(minus)) / (2 * eps)
    return grad


def _built_propagator(layers):
    """A ZZ observable propagated through `layers` of (every ZZ term, then an X on every qubit)."""
    observable = PauliOperator(
        {
            Pauli("ZZ", [i, j]): 0.5
            for i, j in itertools.combinations(range(NUM_QUBITS), 2)
        },
        num_qubits=NUM_QUBITS,
    )
    gates = []
    for k in range(layers):
        gates += [
            ExpGate(PauliOperator({pauli: v}, num_qubits=NUM_QUBITS), index=k)
            for pauli, v in observable.terms.items()
        ]
        gates += [
            ExpGate(
                PauliOperator({Pauli("X", i): -1.0}, num_qubits=NUM_QUBITS),
                index=layers + k,
            )
            for i in range(NUM_QUBITS)
        ]

    propagator = PauliPropagator(observable, [], cutoff=CUTOFF)
    propagator.build_graph(Circuit(gates=gates, system_size=NUM_QUBITS))
    assert propagator.graph_size()[0] > 0, (
        "cutoff did not prune; the checks would be vacuous"
    )
    return propagator


def _check_against_finite_differences(propagator, params):
    gradient = propagator.gradient(params)
    assert np.all(np.isfinite(gradient))
    assert np.allclose(
        gradient, _central_differences(propagator.expectation_value, params), atol=1e-6
    )


@pytest.mark.parametrize("layers", [4, 8])
def test_deep_pauli_gradient_matches_finite_differences(layers):
    _check_against_finite_differences(_built_propagator(layers), np.ones(2 * layers))


# layers=1 is the control: it passes on main, so a failure there is the harness, not the fix.
@pytest.mark.parametrize("layers", [1, 2, 4])
def test_gradient_where_the_layer_cosine_vanishes(layers):
    _check_against_finite_differences(
        _built_propagator(layers), np.full(2 * layers, np.pi / 4)
    )


# The other side of the bound. These angles put Sum log2|sec| at ~0.9 bits over the whole circuit
# against a 53-bit threshold, so every layer skips its record; theta=1 above reaches 280.
@pytest.mark.parametrize("layers", [4, 8])
def test_gradient_where_no_layer_can_amplify(layers):
    _check_against_finite_differences(
        _built_propagator(layers), np.full(2 * layers, 0.05)
    )
