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

"""Tests for the Circuit authoring type and graph accumulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monoprop import (
    Circuit,
    MajoranaGate,
    MajoranaPropagator,
    PauliGate,
    Term,
)
from monoprop.pauli_data import PauliOperator
from tests.cases import load_problem

DATA = Path(__file__).parent / "data"
FIXTURES = ["rx_rz_ry_rz_exact", "random_exact", "lih_fermionic_spin_exact"]


def _propagator(problem):
    return MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
    )


# -- the Circuit type -----------------------------------------------------------


def test_to_circuit_round_trips_sequence() -> None:
    """MajoranaSequence.to_circuit reproduces the dense arrays and mapping."""
    problem = load_problem(DATA / "lih_fermionic_spin_exact.msgpack")
    mc = problem.monomial_circuit
    circuit = mc.to_circuit()

    # Expanding the gates against the mapping reproduces the dense per-monomial arrays.
    majoranas = [tuple(term.majorana) for gate in circuit for term in gate.terms]
    assert majoranas == [tuple(m) for m in mc.majoranas]
    assert list(circuit.resolved_mapping) != []  # sanity: mapping is populated
    np.testing.assert_allclose(
        circuit.parameters, np.asarray(mc.parameters, dtype=float)
    )
    assert circuit.initial_state == tuple(int(i) for i in mc.initial_state)


def test_default_mapping_is_identity() -> None:
    """Omitting the mapping gives each gate its own distinct angle."""
    circuit = Circuit(
        (MajoranaGate((Term((0, 1), 1.0),)), MajoranaGate((Term((2, 3), 1.0),)))
    )
    assert circuit.resolved_mapping == (0, 1)
    assert circuit.n_parameters == 2


def test_shared_mapping_index_ties_gates() -> None:
    """Reusing one index in the mapping ties gates to one angle."""
    gates = (
        MajoranaGate((Term((0, 1), 1.0),)),
        MajoranaGate((Term((2, 3), 1.0),)),
        MajoranaGate((Term((0, 3), 1.0),)),  # same angle as the first gate
    )
    circuit = Circuit(gates, parameter_mapping=(0, 1, 0))
    assert circuit.resolved_mapping == (0, 1, 0)
    assert circuit.n_parameters == 2


def test_circuit_rejects_non_contiguous_mapping() -> None:
    """A mapping with an index gap is rejected (it would invent a phantom parameter)."""
    gates = (MajoranaGate((Term((0, 1), 1.0),)), MajoranaGate((Term((2, 3), 1.0),)))
    with pytest.raises(ValueError, match="contiguous"):
        Circuit(gates, parameter_mapping=(0, 2))


def test_circuit_rejects_mapping_length_mismatch() -> None:
    """A mapping whose length differs from the gate count is rejected."""
    gates = (MajoranaGate((Term((0, 1), 1.0),)), MajoranaGate((Term((2, 3), 1.0),)))
    with pytest.raises(ValueError, match="entries but there are 2 gates"):
        Circuit(gates, parameter_mapping=(0,))


def test_circuit_rejects_wrong_parameter_length() -> None:
    """A bound circuit must supply exactly one value per distinct angle."""
    gates = (MajoranaGate((Term((0, 1), 1.0),)),)
    with pytest.raises(ValueError, match="1 parameters"):
        Circuit(gates, parameters=(0.1, 0.2))


def test_circuit_add_offsets_second_axis() -> None:
    """Concatenating circuits appends the second's angles on a fresh axis."""
    a = Circuit(
        (MajoranaGate((Term((0,), 1.0),)), MajoranaGate((Term((1,), 1.0),))),
        parameters=(0.1, 0.2),
    )
    b = Circuit((MajoranaGate((Term((2,), 1.0),)),), parameters=(0.3,))
    combined = a + b
    assert combined.resolved_mapping == (0, 1, 2)
    assert combined.parameters == (0.1, 0.2, 0.3)
    assert combined.n_parameters == 3


# -- evaluation contracts -------------------------------------------------------


@pytest.mark.parametrize("fixture", FIXTURES)
def test_expectation_value_matches_exact(fixture: str) -> None:
    """Building the graph then evaluating reproduces the exact expectation value."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.propagate_build_graph(circuit)

    assert prop.n_parameters == circuit.n_parameters
    # The circuit carries its own parameters, so it can be evaluated directly.
    np.testing.assert_allclose(prop.expectation_value(circuit), problem.exact_expval)
    np.testing.assert_allclose(
        prop.expectation_value(circuit.parameters), problem.exact_expval
    )
    np.testing.assert_allclose(
        prop.gradient(circuit), problem.exact_gradient, atol=1e-9
    )


def test_eval_rejects_wrong_parameter_length() -> None:
    """expectation_value validates the parameter vector length against the graph."""
    problem = load_problem(DATA / "rx_rz_ry_rz_exact.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.propagate_build_graph(circuit)
    with pytest.raises(RuntimeError):
        prop.expectation_value([0.1])  # graph has more than one parameter


@pytest.mark.parametrize("fixture", FIXTURES)
def test_pared_functional_matches_unpared(fixture: str) -> None:
    """A pared functional agrees with the exact (unpared) evaluation."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.propagate_build_graph(circuit)

    pared = prop.expectation_value_functional(pare_threshold=1e-12)
    np.testing.assert_allclose(
        pared(circuit.parameters), problem.exact_expval, atol=1e-9
    )


def test_from_circuit_matches_manual_build() -> None:
    """from_circuit wires the initial state and graph in one step."""
    problem = load_problem(DATA / "lih_fermionic_spin_exact.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = MajoranaPropagator.from_circuit(
        circuit, problem.operator, cutoff=2 * problem.n_modes
    )
    np.testing.assert_allclose(prop.expectation_value(circuit), problem.exact_expval)


# -- the graph-owned parameter mapping ------------------------------------------


def test_parameter_mapping_getter_reflects_graph() -> None:
    """The propagator exposes the graph's per-layer mapping; re-setting it is a no-op."""
    problem = load_problem(DATA / "lih_fermionic_spin_exact.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.propagate_build_graph(circuit)

    mapping = prop.parameter_mapping
    assert len(mapping) == prop.graph_layers  # per-layer (per-monomial) granularity
    assert max(mapping) + 1 == prop.n_parameters
    # Round-trip: writing the mapping back changes nothing.
    prop.parameter_mapping = mapping
    assert prop.parameter_mapping == mapping
    np.testing.assert_allclose(prop.expectation_value(circuit), problem.exact_expval)


def test_parameter_mapping_setter_ties_parameters() -> None:
    """Re-wiring the graph mapping ties layers to a shared angle without rebuilding."""
    problem = load_problem(DATA / "lih_fermionic_spin_exact.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.propagate_build_graph(circuit)
    n_layers = prop.graph_layers

    # Tie every layer to one shared angle: evaluating at [theta] must equal the untied
    # graph evaluated with that same theta broadcast across every parameter.
    prop.parameter_mapping = [0] * n_layers
    assert prop.n_parameters == 1
    tied = prop.expectation_value([0.3])

    prop.parameter_mapping = list(range(n_layers))
    assert prop.n_parameters == n_layers
    reference = prop.expectation_value([0.3] * n_layers)
    np.testing.assert_allclose(tied, reference)


def test_parameter_mapping_setter_validates() -> None:
    """The setter rejects a wrong-length or non-contiguous mapping."""
    problem = load_problem(DATA / "rx_rz_ry_rz_exact.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.propagate_build_graph(circuit)

    with pytest.raises(ValueError, match="layers"):
        prop.parameter_mapping = [0]  # too short
    with pytest.raises(ValueError, match="contiguous"):
        prop.parameter_mapping = [i + 1 for i in range(prop.graph_layers)]  # no 0


def test_majorana_propagator_rejects_pauli_gate() -> None:
    """A PauliGate in a circuit fed to MajoranaPropagator raises a clear TypeError."""
    problem = load_problem(DATA / "rx_rz_ry_rz_exact.msgpack")
    prop = _propagator(problem)
    circuit = Circuit(gates=(PauliGate((0,), PauliOperator(["Z"], [1.0])),))

    with pytest.raises(TypeError, match="MajoranaGate"):
        prop.propagate(circuit)


# -- building the graph incrementally -------------------------------------------


def _schrodinger_propagator(problem):
    return MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
        schrodinger_cutoff=2 * problem.n_modes,
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_build_graph_accumulates_layers_and_parameters(fixture: str) -> None:
    """Two propagate_build_graph calls accumulate the graph (layers + parameters)."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    gates = circuit.gates
    split = len(gates) // 2

    single = _propagator(problem)
    single.propagate_build_graph(circuit)

    twice = _propagator(problem)
    twice.propagate_build_graph(Circuit(gates[:split]))
    layers_after_first = twice.graph_layers
    twice.propagate_build_graph(Circuit(gates[split:]))

    assert 0 < layers_after_first < twice.graph_layers
    assert twice.graph_layers == single.graph_layers
    assert twice.n_parameters == single.n_parameters


@pytest.mark.parametrize("fixture", FIXTURES)
def test_compose_then_single_build_matches_single_call(fixture: str) -> None:
    """Composing two circuit halves with ``+`` and building in one call matches.

    Composition is picture-independent: because the whole sequence is fed in a single
    propagate_build_graph call, there is no back-to-front reordering across calls to
    reason about (in either picture).
    """
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    gates = circuit.gates
    split = len(gates) // 2
    params = list(map(float, problem.monomial_circuit.parameters))
    a = Circuit(gates[:split], parameters=tuple(params[:split]))
    b = Circuit(gates[split:], parameters=tuple(params[split:]))
    composed = a + b

    single = _propagator(problem)
    single.propagate_build_graph(circuit)

    fused = _propagator(problem)
    fused.propagate_build_graph(composed)

    np.testing.assert_allclose(
        fused.expectation_value(composed), single.expectation_value(circuit)
    )
    np.testing.assert_allclose(fused.expectation_value(composed), problem.exact_expval)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_build_graph_in_two_calls_schrodinger(fixture: str) -> None:
    """In Schrodinger picture, building in two forward calls equals a single call.

    (In Heisenberg picture each call applies its gates back-to-front, so a forward
    split is deliberately *not* equivalent to one call -- compose with ``+`` instead.)
    """
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    gates = circuit.gates
    params = list(map(float, problem.monomial_circuit.parameters))
    split = len(gates) // 2

    single = _schrodinger_propagator(problem)
    single.propagate_build_graph(circuit)

    twice = _schrodinger_propagator(problem)
    twice.propagate_build_graph(Circuit(gates[:split]))
    twice.propagate_build_graph(Circuit(gates[split:]))

    np.testing.assert_allclose(
        twice.expectation_value(params), single.expectation_value(params)
    )
    np.testing.assert_allclose(twice.expectation_value(params), problem.exact_expval)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_build_graph_twice_with_seed_regeneration(fixture: str) -> None:
    """Extending a non-empty graph with seed_parameters regenerates the seed internally."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    gates = circuit.gates
    params = list(map(float, problem.monomial_circuit.parameters))
    split = len(gates) // 2

    prop = _schrodinger_propagator(problem)
    prop.propagate_build_graph(Circuit(gates[:split]))
    # seed_parameters on the second call exercises the internal seed regeneration
    # (the former operator_coeffs round-trip) used for coefficient-informed truncation.
    prop.propagate_build_graph(Circuit(gates[split:]), seed_parameters=params)

    np.testing.assert_allclose(prop.expectation_value(params), problem.exact_expval)
