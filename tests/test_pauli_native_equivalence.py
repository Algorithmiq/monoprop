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

"""Python R1 equivalence pin: native Pauli engine vs Jordan-Wigner fallback.

The definitive Python correctness arbiter for the Pauli-native basis. For the SAME
``PauliPropagator`` problem built twice (``engine_basis="pauli"`` vs
``engine_basis="majorana-jw"``) this asserts, across pictures, cutoffs, and truncation
tolerances:

- **identical expectation value** (rel/abs ~1e-12), and
- **identical stored ``size()`` after every layer** (driven per-layer via successive
  ``propagate`` calls, so the two representations track term-for-term).

If the native generator-coefficient sign were wrong, the expectation value would flip sign;
the tests would catch it here (the fix would be the single ``gen_coeff`` knob in circuit.py).
In practice the pinned convention holds and no flip is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from monoprop import Circuit, Exp, Pauli, PauliOperator, PauliPropagator

_EV_RTOL = 1e-12
_EV_ATOL = 1e-12
_GRAD_ATOL = 1e-10


# -- problem builders -----------------------------------------------------------------------


def _chain(num_qubits: int) -> list[tuple[int, int]]:
    """A linear-chain coupling map."""
    return [(i, i + 1) for i in range(num_qubits - 1)]


def _heavy_hex_ish(num_qubits: int) -> list[tuple[int, int]]:
    """A small branched (heavy-hex-flavoured) coupling map on ``num_qubits`` qubits."""
    edges = [(i, i + 1) for i in range(num_qubits - 1)]  # backbone
    # A few branch rungs, mimicking heavy-hex's degree-3 vertices.
    edges.extend((a, a + 3) for a in range(0, num_qubits - 4, 4))
    return [(i, j) for i, j in edges if j < num_qubits]


def _x_layer(num_qubits: int, angle: float) -> list[tuple[Exp, float]]:
    """Single-qubit X rotations on every qubit (odd-popcount generators)."""
    return [
        (Exp(PauliOperator({Pauli("X", (q,)): 1.0}, num_qubits=num_qubits)), -angle)
        for q in range(num_qubits)
    ]


def _zz_layer(
    num_qubits: int, angle: float, topology: list[tuple[int, int]]
) -> list[tuple[Exp, float]]:
    """Two-qubit ZZ rotations on every edge of ``topology``."""
    return [
        (Exp(PauliOperator({Pauli("ZZ", (i, j)): 1.0}, num_qubits=num_qubits)), -angle)
        for i, j in topology
    ]


def _kicked_ising_gates(
    num_qubits: int, topology: list[tuple[int, int]], num_layers: int
) -> list[tuple[Exp, float]]:
    """Alternating X-rotation / ZZ-rotation layers (a kicked-Ising circuit)."""
    gate_angles: list[tuple[Exp, float]] = []
    for _ in range(num_layers):
        gate_angles.extend(_x_layer(num_qubits, np.pi / 8))
        gate_angles.extend(_zz_layer(num_qubits, np.pi / 8, topology))
    return gate_angles


def _single_z_observable(num_qubits: int, qubit: int) -> PauliOperator:
    """A single-Z observable on one qubit (the kicked-Ising order parameter)."""
    return PauliOperator({Pauli("Z", (qubit,)): 1.0}, num_qubits=num_qubits)


def _random_pauli_gates(
    num_qubits: int, num_gates: int, seed: int
) -> list[tuple[Exp, float]]:
    """Seeded single-Pauli-string gates (weight 1 or 2) with random angles.

    Each generator is a single Pauli term, so it trivially commutes with itself and the
    commuting-generator validation passes.
    """
    rng = np.random.default_rng(seed)
    letters = "XYZ"
    gate_angles: list[tuple[Exp, float]] = []
    for _ in range(num_gates):
        weight = int(rng.integers(1, 3))
        qubits = tuple(
            sorted(rng.choice(num_qubits, size=weight, replace=False).tolist())
        )
        string = "".join(rng.choice(list(letters)) for _ in range(weight))
        gen = PauliOperator({Pauli(string, qubits): 1.0}, num_qubits=num_qubits)
        angle = float(rng.uniform(-0.8, 0.8))
        gate_angles.append((Exp(gen), angle))
    return gate_angles


def _random_observable(num_qubits: int, num_terms: int, seed: int) -> PauliOperator:
    """A seeded random Hermitian Pauli observable (real coefficients)."""
    rng = np.random.default_rng(seed)
    letters = "XYZ"
    terms: dict[Pauli, float] = {}
    while len(terms) < num_terms:
        weight = int(rng.integers(1, 4))
        qubits = tuple(
            sorted(rng.choice(num_qubits, size=weight, replace=False).tolist())
        )
        string = "".join(rng.choice(list(letters)) for _ in range(weight))
        terms[Pauli(string, qubits)] = float(rng.standard_normal())
    return PauliOperator(terms, num_qubits=num_qubits)


# -- comparison core ------------------------------------------------------------------------


def _make(
    comm,
    observable,
    initial_state,
    *,
    cutoff,
    schrodinger_cutoff,
    lower_atol,
    engine_basis,
):
    return PauliPropagator(
        observable,
        list(initial_state),
        cutoff=cutoff,
        schrodinger_cutoff=schrodinger_cutoff,
        lower_atol=lower_atol,
        comm=comm,
        engine_basis=engine_basis,
    )


def _assert_arms_equivalent(
    comm,
    observable,
    gate_angles,
    *,
    cutoff,
    schrodinger_cutoff=None,
    lower_atol=None,
    initial_state=(),
):
    """Assert the native and JW arms match on expectation value and per-layer term count."""
    circuit = Circuit(
        gates=tuple(gate for gate, _ in gate_angles),
        parameters=tuple(angle for _, angle in gate_angles),
        initial_state=initial_state,
    )

    def _build(engine_basis):
        propagator = _make(
            comm,
            observable,
            initial_state,
            cutoff=cutoff,
            schrodinger_cutoff=schrodinger_cutoff,
            lower_atol=lower_atol,
            engine_basis=engine_basis,
        )
        propagator.build_graph(circuit)
        return propagator

    native = _build("pauli")
    jw = _build("majorana-jw")

    ev_native = native.expectation_value(circuit.parameters)
    ev_jw = jw.expectation_value(circuit.parameters)
    assert np.isclose(ev_native, ev_jw, rtol=_EV_RTOL, atol=_EV_ATOL), (
        ev_native,
        ev_jw,
    )
    assert native.size() == jw.size()

    # Per-layer term-count tracking: both arms are driven identically, one gate per
    # propagate call, and must hold the same number of stored terms at every step.
    step_native = _make(
        comm,
        observable,
        initial_state,
        cutoff=cutoff,
        schrodinger_cutoff=schrodinger_cutoff,
        lower_atol=lower_atol,
        engine_basis="pauli",
    )
    step_jw = _make(
        comm,
        observable,
        initial_state,
        cutoff=cutoff,
        schrodinger_cutoff=schrodinger_cutoff,
        lower_atol=lower_atol,
        engine_basis="majorana-jw",
    )
    for layer, (gate, angle) in enumerate(gate_angles):
        one = Circuit(gates=(gate,), parameters=(angle,), initial_state=initial_state)
        step_native.propagate(one)
        step_jw.propagate(one)
        assert step_native.size() == step_jw.size(), (
            f"size diverged at layer {layer}: "
            f"native={step_native.size()} jw={step_jw.size()}"
        )

    return ev_native


# -- (a) kicked-Ising chain + (c) picture + (e) atol (finite cutoff) ------------------------


@pytest.mark.parametrize("schrodinger", [False, True], ids=["heisenberg", "schrodinger"])
@pytest.mark.parametrize("lower_atol", [None, 1e-4], ids=["atol_off", "atol_on"])
@pytest.mark.parametrize("num_qubits", [8, 12], ids=["nq8", "nq12"])
def test_kicked_ising_chain_equivalence(
    serial_comm, schrodinger, lower_atol, num_qubits
):
    topology = _chain(num_qubits)
    gate_angles = _kicked_ising_gates(num_qubits, topology, num_layers=2)
    observable = _single_z_observable(num_qubits, num_qubits // 2)
    cutoff = 4
    schrodinger_cutoff = cutoff + 2 if schrodinger else None
    _assert_arms_equivalent(
        serial_comm,
        observable,
        gate_angles,
        cutoff=cutoff,
        schrodinger_cutoff=schrodinger_cutoff,
        lower_atol=lower_atol,
    )


def test_kicked_ising_16q_scale(serial_comm):
    """A 16-qubit instance with a tight support cutoff (Heisenberg, atol off)."""
    num_qubits = 16
    gate_angles = _kicked_ising_gates(num_qubits, _chain(num_qubits), num_layers=2)
    observable = _single_z_observable(num_qubits, num_qubits // 2)
    _assert_arms_equivalent(serial_comm, observable, gate_angles, cutoff=3)


# -- (d) effectively-infinite cutoff on a small system --------------------------------------


@pytest.mark.parametrize("schrodinger", [False, True], ids=["heisenberg", "schrodinger"])
def test_effectively_infinite_cutoff_equivalence(serial_comm, schrodinger):
    # cutoff = num_qubits is full Pauli weight, so no structural truncation happens: the two
    # arms must agree on the (untruncated) evolved operator term-for-term. Kept small so the
    # untruncated operator stays tractable.
    num_qubits = 6
    gate_angles = _kicked_ising_gates(num_qubits, _chain(num_qubits), num_layers=2)
    observable = _single_z_observable(num_qubits, num_qubits // 2)
    cutoff = num_qubits
    schrodinger_cutoff = cutoff + 2 if schrodinger else None
    _assert_arms_equivalent(
        serial_comm,
        observable,
        gate_angles,
        cutoff=cutoff,
        schrodinger_cutoff=schrodinger_cutoff,
    )


@pytest.mark.parametrize("schrodinger", [False, True], ids=["heisenberg", "schrodinger"])
def test_kicked_ising_heavy_hex_equivalence(serial_comm, schrodinger):
    num_qubits = 12
    topology = _heavy_hex_ish(num_qubits)
    gate_angles = _kicked_ising_gates(num_qubits, topology, num_layers=2)
    observable = _single_z_observable(num_qubits, num_qubits // 2)
    cutoff = 4
    schrodinger_cutoff = cutoff + 2 if schrodinger else None
    _assert_arms_equivalent(
        serial_comm,
        observable,
        gate_angles,
        cutoff=cutoff,
        schrodinger_cutoff=schrodinger_cutoff,
    )


# -- (b) random commuting single-Pauli-gate circuits ----------------------------------------


@pytest.mark.parametrize("schrodinger", [False, True], ids=["heisenberg", "schrodinger"])
@pytest.mark.parametrize("seed", [0, 5])
@pytest.mark.parametrize("num_qubits", [6, 9], ids=["nq6", "nq9"])
def test_random_pauli_circuit_equivalence(
    serial_comm, schrodinger, seed, num_qubits
):
    num_gates = 30 + 6 * seed  # 30 (seed 0) .. 60 (seed 5)
    gate_angles = _random_pauli_gates(num_qubits, num_gates, seed)
    observable = _random_observable(num_qubits, num_terms=10, seed=seed + 100)
    cutoff = 4
    schrodinger_cutoff = cutoff + 2 if schrodinger else None
    _assert_arms_equivalent(
        serial_comm,
        observable,
        gate_angles,
        cutoff=cutoff,
        schrodinger_cutoff=schrodinger_cutoff,
        lower_atol=1e-6,
    )


# -- (f) gradient equality ------------------------------------------------------------------


@pytest.mark.parametrize("schrodinger", [False, True], ids=["heisenberg", "schrodinger"])
def test_gradient_equivalence(serial_comm, schrodinger):
    num_qubits = 8
    gate_angles = _kicked_ising_gates(num_qubits, _chain(num_qubits), num_layers=2)
    observable = _random_observable(num_qubits, num_terms=8, seed=3)
    circuit = Circuit(
        gates=tuple(gate for gate, _ in gate_angles),
        parameters=tuple(angle for _, angle in gate_angles),
        initial_state=[],
    )
    cutoff = 5
    schrodinger_cutoff = cutoff + 2 if schrodinger else None

    def _value_and_grad(engine_basis):
        propagator = _make(
            serial_comm,
            observable,
            [],
            cutoff=cutoff,
            schrodinger_cutoff=schrodinger_cutoff,
            lower_atol=None,
            engine_basis=engine_basis,
        )
        propagator.build_graph(circuit)
        return propagator.expectation_value_and_gradient(circuit.parameters)

    value_native, grad_native = _value_and_grad("pauli")
    value_jw, grad_jw = _value_and_grad("majorana-jw")
    assert np.isclose(value_native, value_jw, rtol=_EV_RTOL, atol=_EV_ATOL)
    assert grad_native.shape == grad_jw.shape
    assert np.allclose(grad_native, grad_jw, atol=_GRAD_ATOL)


# -- (g) from_circuit parity ----------------------------------------------------------------


def test_from_circuit_parity(serial_comm):
    num_qubits = 8
    gate_angles = _kicked_ising_gates(num_qubits, _chain(num_qubits), num_layers=3)
    observable = _single_z_observable(num_qubits, num_qubits // 2)
    circuit = Circuit(
        gates=tuple(gate for gate, _ in gate_angles),
        parameters=tuple(angle for _, angle in gate_angles),
        initial_state=[],
    )
    native = PauliPropagator.from_circuit(
        circuit, observable, cutoff=5, comm=serial_comm, engine_basis="pauli"
    )
    jw = PauliPropagator.from_circuit(
        circuit, observable, cutoff=5, comm=serial_comm, engine_basis="majorana-jw"
    )
    assert np.isclose(
        native.expectation_value(), jw.expectation_value(), rtol=_EV_RTOL, atol=_EV_ATOL
    )
    assert native.size() == jw.size()


# -- (h) pared functional -------------------------------------------------------------------


def test_pared_functional_equivalence(serial_comm):
    num_qubits = 8
    gate_angles = _kicked_ising_gates(num_qubits, _chain(num_qubits), num_layers=2)
    observable = _single_z_observable(num_qubits, num_qubits // 2)
    circuit = Circuit(
        gates=tuple(gate for gate, _ in gate_angles),
        parameters=tuple(angle for _, angle in gate_angles),
        initial_state=[],
    )

    def _build(engine_basis):
        propagator = _make(
            serial_comm,
            observable,
            [],
            cutoff=5,
            schrodinger_cutoff=None,
            lower_atol=None,
            engine_basis=engine_basis,
        )
        propagator.build_graph(circuit)
        return propagator

    native = _build("pauli")
    jw = _build("majorana-jw")

    # A tiny pare threshold keeps the masked replay effectively exact.
    native_fn = native.expectation_value_functional(pare_threshold=1e-12)
    ev_native_pared = native_fn(circuit.parameters)
    ev_jw = jw.expectation_value(circuit.parameters)
    ev_native_unpared = native.expectation_value(circuit.parameters)

    assert np.isclose(ev_native_pared, ev_native_unpared, atol=1e-9)
    assert np.isclose(ev_native_pared, ev_jw, atol=1e-9)
