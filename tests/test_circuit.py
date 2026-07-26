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
from pytest_cases import case, parametrize_with_cases

from monoprop import (
    Circuit,
    ExpGate,
    MajoranaPropagator,
)
from monoprop.fermi import FermiOperator
from monoprop.majorana import Majorana, MajoranaOperator
from monoprop.pauli import Pauli, PauliOperator
from tests.cases import load_problem

DATA = Path(__file__).parent / "data"
FIXTURES = ["rx_rz_ry_rz_exact", "random_exact", "lih_fermionic_spin_exact"]


def _propagator(problem):
    return MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
    )


def _rebase(gates):
    """Drop each gate's explicit ``index`` so a gate slice gets the default identity mapping.

    Slicing a circuit's gates keeps their absolute ``index`` indices, which are no longer
    contiguous from 0 in a tail slice; re-basing to the identity restores a valid sub-circuit.
    ``_with_index`` preserves each gate's ``_structural`` flag -- these gates come from a
    ``from_dense_arrays`` circuit (already-structural coefficients), so a plain
    ``ExpGate(gate.generator)`` would re-antihermitian-normalize them and reject the structural
    real coefficients of weight-2 monomials.
    """
    return tuple(ExpGate._with_index(gate, None) for gate in gates)


# -- the Circuit type -----------------------------------------------------------


def test_exp_rejects_bare_term() -> None:
    """ExpGate accepts only operator objects; a bare Majorana/Pauli term is a clear TypeError."""
    with pytest.raises(TypeError, match="not a bare term"):
        ExpGate(Majorana(0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a bare term"):
        ExpGate(Pauli("X", 0))  # type: ignore[arg-type]


def test_exp_equality_and_repr() -> None:
    """Two ExpGate gates are equal when generator, param, family, and structural flag match."""
    gen = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
    assert ExpGate(gen, index=0) == ExpGate(gen, index=0)
    assert ExpGate(gen, index=0) != ExpGate(gen, index=1)
    assert ExpGate(gen) != "not an ExpGate"
    assert repr(ExpGate(gen, index=0)).startswith("ExpGate(")


def test_exp_from_fermi_generator_becomes_majorana() -> None:
    """A FermiOperator generator is converted to its Majorana form and the gate is 'majorana'."""
    gate = ExpGate(
        FermiOperator(
            [[(0, "+"), (1, "-")], [(1, "+"), (0, "-")]], [1.0, -1.0], num_modes=2
        )
    )
    assert gate.family == "majorana"
    assert isinstance(gate.generator, MajoranaOperator)


class ExpGateAtolCases:
    @case(id="single_excitation")
    def case_single_excitation(self):
        """Without truncation, single excitation can produce up to 4 terms."""
        generator = FermiOperator(
            [
                [(0, "+"), (1, "-")],
                [(1, "+"), (0, "-")],
            ],
            [1.0, -1.0 + 1e-10],
            num_modes=2,
        )
        expected = MajoranaOperator({(0, 2): 0.5, (1, 3): 0.5}, num_modes=2)
        return generator, 1e-8, expected

    @case(id="double_excitation")
    def case_double_excitation(self):
        """Without truncation, double excitation can produce up to 4 terms."""
        generator = FermiOperator(
            [
                [(0, "+"), (1, "+"), (2, "-"), (3, "-")],
                [(3, "+"), (2, "+"), (1, "-"), (0, "-")],
            ],
            [1.0, -1.0 + 1e-10],
            num_modes=4,
        )
        expected = MajoranaOperator(
            {
                (0, 2, 4, 7): 0.125j,
                (0, 2, 5, 6): 0.125j,
                (0, 3, 4, 6): -0.125j,
                (0, 3, 5, 7): 0.125j,
                (1, 2, 4, 6): -0.125j,
                (1, 2, 5, 7): 0.125j,
                (1, 3, 4, 7): -0.125j,
                (1, 3, 5, 6): -0.125j,
            },
            num_modes=4,
        )
        return generator, 1e-8, expected

    @case(id="majorana_operator")
    def case_majorana_operator(self):
        generator = MajoranaOperator({(0, 1): 1.0j, (2, 3): 1e-10j}, num_modes=2)
        expected = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
        return generator, 1e-8, expected

    @case(id="pauli_operator")
    def case_pauli_operator(self):
        generator = PauliOperator(
            {Pauli("XX"): 1.0, Pauli("ZZ"): 1e-4, Pauli("YY"): 1e-6},
            num_qubits=2,
        )
        expected = PauliOperator({Pauli("XX"): 1.0, Pauli("ZZ"): 1e-4}, num_qubits=2)
        return generator, 1e-5, expected


@parametrize_with_cases("generator, atol, expected", cases=ExpGateAtolCases)
def test_exp_gate_applies_atol_truncation(
    generator: FermiOperator | MajoranaOperator | PauliOperator,
    atol: float,
    expected: MajoranaOperator | PauliOperator,
) -> None:
    """ExpGate(atol=...) keeps only terms above the provided threshold."""
    truncated = ExpGate(generator, atol=atol).generator

    assert isinstance(truncated, type(expected))
    assert truncated.terms.keys() == expected.terms.keys()
    assert truncated.isclose(expected, atol=1e-10, rtol=0.0)


def test_circuit_equality() -> None:
    """Circuits are equal on gates/parameters/initial_state; family is derived, not compared."""
    gen = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
    a = Circuit((ExpGate(gen),), parameters=(0.3,), initial_state=(0,))
    b = Circuit((ExpGate(gen),), parameters=(0.3,), initial_state=(0,))
    assert a == b
    assert a != Circuit((ExpGate(gen),), parameters=(0.9,), initial_state=(0,))
    assert a != "not a circuit"


def test_circuit_rejects_non_exp_gate() -> None:
    """A gate that is not an ExpGate is rejected with a clear TypeError."""
    with pytest.raises(TypeError, match="Circuit gates must be ExpGate"):
        Circuit(("not a gate",))  # type: ignore[arg-type]


def test_to_circuit_round_trips_sequence() -> None:
    """Circuit.from_dense_arrays reproduces the dense arrays and mapping."""
    problem = load_problem(DATA / "lih_fermionic_spin_exact.msgpack")
    mc = problem.monomial_circuit
    circuit = mc.to_circuit()

    # Expanding the gates against the mapping reproduces the dense per-monomial arrays.
    majoranas = [maj for gate in circuit for maj in gate.generator.terms]
    assert majoranas == [tuple(m) for m in mc.majoranas]
    assert list(circuit.resolved_mapping) != []  # sanity: mapping is populated
    np.testing.assert_allclose(
        circuit.parameters, np.asarray(mc.parameters, dtype=float)
    )
    assert circuit.initial_state == tuple(int(i) for i in mc.initial_state)


def test_default_mapping_is_identity() -> None:
    """Omitting the mapping gives each gate its own distinct angle."""
    circuit = Circuit(
        (
            ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2)),
            ExpGate(MajoranaOperator({(2, 3): 1.0j}, num_modes=2)),
        )
    )
    assert circuit.resolved_mapping == (0, 1)
    assert circuit.n_parameters == 2


def test_shared_mapping_index_ties_gates() -> None:
    """Reusing one index in the mapping ties gates to one angle."""
    gates = (
        ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2), index=0),
        ExpGate(MajoranaOperator({(2, 3): 1.0j}, num_modes=2), index=1),
        ExpGate(
            MajoranaOperator({(0, 3): 1.0j}, num_modes=2), index=0
        ),  # ties to the first
    )
    circuit = Circuit(gates)
    assert circuit.resolved_mapping == (0, 1, 0)
    assert circuit.n_parameters == 2


def test_circuit_rejects_non_contiguous_mapping() -> None:
    """A mapping with an index gap is rejected (it would invent a phantom parameter)."""
    gates = (
        ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2), index=0),
        ExpGate(MajoranaOperator({(2, 3): 1.0j}, num_modes=2), index=2),
    )
    with pytest.raises(ValueError, match="contiguous"):
        Circuit(gates)


def test_circuit_rejects_mixed_param_scheme() -> None:
    """Setting `index` on some gates but not others is rejected as ambiguous."""
    gates = (
        ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2), index=0),
        ExpGate(MajoranaOperator({(2, 3): 1.0j}, num_modes=2)),
    )
    with pytest.raises(ValueError, match="every gate must set"):
        Circuit(gates)


def test_circuit_rejects_wrong_parameter_length() -> None:
    """A bound circuit must supply exactly one value per distinct angle."""
    gates = (ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2)),)
    with pytest.raises(ValueError, match="1 parameters"):
        Circuit(gates, parameters=(0.1, 0.2))


def test_circuit_add_offsets_second_axis() -> None:
    """Concatenating circuits appends the second's angles on a fresh axis."""
    a = Circuit(
        (
            ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2)),
            ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2)),
        ),
        parameters=(0.1, 0.2),
    )
    b = Circuit(
        (ExpGate(MajoranaOperator({(2,): 1.0}, num_modes=2)),), parameters=(0.3,)
    )
    combined = a + b
    assert combined.resolved_mapping == (0, 1, 2)
    assert combined.parameters == (0.1, 0.2, 0.3)
    assert combined.n_parameters == 3


def test_circuit_add_rejects_mixed_families() -> None:
    """Concatenating a Majorana circuit with a qubit circuit raises a clear TypeError."""
    maj = Circuit((ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2)),))
    qubit = Circuit((ExpGate(PauliOperator({Pauli("X", 0): 1.0}, num_qubits=2)),))
    with pytest.raises(TypeError, match="gate families differ"):
        _ = maj + qubit


def test_circuit_add_rejects_different_initial_states() -> None:
    """Concatenating circuits with conflicting initial states raises a ValueError."""
    a = Circuit(
        (ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2)),), initial_state=(0,)
    )
    b = Circuit(
        (ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2)),), initial_state=(1,)
    )
    with pytest.raises(ValueError, match="different initial states"):
        _ = a + b


def test_bound_circuit_with_identity_gate_wrong_param_count_raises() -> None:
    """Under the default mapping, a bound circuit's params must match the pre-drop gate count."""
    gates = (
        ExpGate(MajoranaOperator({}, num_modes=2)),  # identity gate (dropped)
        ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2)),
    )
    with pytest.raises(ValueError, match="2 gates"):
        Circuit(gates, parameters=(0.1,))  # 1 value, but 2 gates before the drop


def test_non_hermitian_majorana_generator_rejected() -> None:
    """A Majorana gate whose generator is not Hermitian is rejected on ingestion.

    A Majorana generator carries the *Hermitian* operator (the same convention as an
    observable): a weight-2 monomial takes an imaginary coefficient. A *real* weight-2
    coefficient is not Hermitian, so it is rejected rather than silently normalized.
    """
    obs = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
    prop = MajoranaPropagator(obs, [0, 1], cutoff=4)
    bad = Circuit(
        (ExpGate(MajoranaOperator({(0, 1): 1.0}, num_modes=2)),), parameters=(0.3,)
    )
    with pytest.raises(ValueError, match="not Hermitian"):
        prop.propagate(bad)


def test_hermitian_majorana_generator_matches_structural() -> None:
    """A Hermitian Majorana generator reproduces the equivalent structural (dense) gate.

    The Hermitian generator ``i·m_4 m_5`` (coefficient ``1j`` on the bare product) is
    antihermitian-normalized to the structural coefficient ``g = -1.0``, so
    ``ExpGate(MajoranaOperator({(4, 5): 1j}))`` must evolve an observable identically to the
    dense/wire gate carrying ``g = -1.0`` directly.
    """
    obs = MajoranaOperator({(0, 1, 2, 4): 1.0}, 8)
    hermitian = Circuit(
        gates=(ExpGate(MajoranaOperator({(4, 5): 1j}, num_modes=8)),), parameters=(0.5,)
    )
    structural = Circuit.from_dense_arrays([[4, 5]], [-1.0], [0], parameters=[0.5])

    from_hermitian = MajoranaPropagator.from_circuit(
        hermitian, obs, cutoff=16
    ).evolved_operator()
    from_structural = MajoranaPropagator.from_circuit(
        structural, obs, cutoff=16
    ).evolved_operator()

    assert from_hermitian.keys() == from_structural.keys()
    for key, value in from_hermitian.items():
        np.testing.assert_allclose(value, from_structural[key])


# -- evaluation contracts -------------------------------------------------------


@pytest.mark.parametrize("fixture", FIXTURES)
def test_expectation_value_matches_exact(fixture: str) -> None:
    """Building the graph then evaluating reproduces the exact expectation value."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.build_graph(circuit)

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
    prop.build_graph(circuit)
    with pytest.raises(RuntimeError):
        prop.expectation_value([0.1])  # graph has more than one parameter


@pytest.mark.parametrize("fixture", FIXTURES)
def test_pared_functional_matches_unpared(fixture: str) -> None:
    """A pared functional agrees with the exact (unpared) evaluation."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.build_graph(circuit)

    pared = prop.expectation_value_functional(pare_threshold=1e-12)
    np.testing.assert_allclose(
        pared(circuit.parameters), problem.exact_expval, atol=1e-9
    )


def test_from_circuit_propagates_in_place() -> None:
    """from_circuit constructs and propagates in one step, in-place with no graph."""
    problem = load_problem(DATA / "lih_fermionic_spin_exact.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = MajoranaPropagator.from_circuit(
        circuit, problem.operator, cutoff=2 * problem.n_modes
    )
    # No graph is stored; the circuit's angles are already applied, so the value is read
    # off with no parameters (and matches a manual construct + propagate).
    assert prop.n_parameters == 0
    np.testing.assert_allclose(prop.expectation_value(), problem.exact_expval)

    manual = _propagator(problem)
    manual.propagate(circuit)
    np.testing.assert_allclose(prop.expectation_value(), manual.expectation_value())


# -- the graph-owned parameter mapping ------------------------------------------


def test_parameter_mapping_getter_reflects_graph() -> None:
    """The propagator exposes the graph's per-layer mapping; re-setting it is a no-op."""
    problem = load_problem(DATA / "lih_fermionic_spin_exact.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    prop = _propagator(problem)
    prop.build_graph(circuit)

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
    prop.build_graph(circuit)
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
    prop.build_graph(circuit)

    with pytest.raises(ValueError, match=r"per graph layer.*per gate"):
        prop.parameter_mapping = [0]  # length matches neither layers nor gates
    with pytest.raises(ValueError, match="contiguous"):
        prop.parameter_mapping = [i + 1 for i in range(prop.graph_layers)]  # no 0


def _multi_term_gate_propagator():
    """A propagator whose graph has a multi-term gate (n_gates < graph_layers)."""
    op = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
    prop = MajoranaPropagator(op, [0, 1], cutoff=4)
    # two monomials -> two layers
    g0 = ExpGate(MajoranaOperator({(0, 2): 1.0j, (1, 3): 1.0j}, num_modes=2))
    g1 = ExpGate(MajoranaOperator({(2,): 1.0}, num_modes=2))
    prop.build_graph(Circuit((g0, g1)))
    return prop


def test_n_gates_tracks_ingested_gates() -> None:
    """n_gates counts authoring gates; a multi-term gate spans several layers."""
    prop = _multi_term_gate_propagator()
    assert prop.n_gates == 2
    assert prop.graph_layers > prop.n_gates  # gate 0 expands to multiple layers


def test_n_gates_accumulates_across_builds() -> None:
    """Each build_graph call extends the gate count on the accumulated axis."""
    op = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
    prop = MajoranaPropagator(op, [0, 1], cutoff=4)
    prop.build_graph(Circuit((ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2)),)))
    assert prop.n_gates == 1
    prop.build_graph(Circuit((ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2)),)))
    assert prop.n_gates == 2


def test_parameter_mapping_setter_accepts_per_gate() -> None:
    """A per-gate mapping (length n_gates) ties a multi-term gate's layers together."""
    prop = _multi_term_gate_propagator()

    # Per-gate mapping tying both gates to one angle: equals the per-layer-tied graph
    # evaluated at the same broadcast angle.
    prop.parameter_mapping = [0, 0]  # length == n_gates
    assert prop.n_parameters == 1
    tied_by_gate = prop.expectation_value([0.3])

    prop.parameter_mapping = [0] * prop.graph_layers  # length == graph_layers
    tied_by_layer = prop.expectation_value([0.3])
    np.testing.assert_allclose(tied_by_gate, tied_by_layer)


def test_majorana_propagator_rejects_pauli_circuit() -> None:
    """A Circuit fed to MajoranaPropagator raises a clear TypeError."""
    problem = load_problem(DATA / "rx_rz_ry_rz_exact.msgpack")
    prop = _propagator(problem)
    circuit = Circuit(
        gates=(ExpGate(PauliOperator({"Z": 1.0}, num_qubits=1)),),
    )

    with pytest.raises(TypeError, match="qubit"):
        prop.propagate(circuit)


def test_propagate_rejects_mismatched_initial_state() -> None:
    """A circuit whose initial_state disagrees with the propagator's raises ValueError."""
    problem = load_problem(DATA / "rx_rz_ry_rz_exact.msgpack")
    prop = _propagator(problem)  # built with the fixture's initial state ([])
    gate = ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2))
    circuit = Circuit((gate,), parameters=(0.1,), initial_state=(0, 1))

    with pytest.raises(ValueError, match="initial_state"):
        prop.propagate(circuit)


def test_propagate_accepts_empty_initial_state() -> None:
    """An empty circuit.initial_state defers to the propagator's reference state."""
    problem = load_problem(DATA / "rx_rz_ry_rz_exact.msgpack")
    prop = _propagator(problem)
    gate = ExpGate(MajoranaOperator({(0, 1): 1.0j}, num_modes=2))
    circuit = Circuit((gate,), parameters=(0.1,))  # empty initial_state

    prop.propagate(circuit)  # does not raise


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
    """Two build_graph calls accumulate the graph (layers + parameters)."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    gates = circuit.gates
    split = len(gates) // 2

    single = _propagator(problem)
    single.build_graph(circuit)

    twice = _propagator(problem)
    twice.build_graph(Circuit(_rebase(gates[:split])))
    layers_after_first = twice.graph_layers
    twice.build_graph(Circuit(_rebase(gates[split:])))

    assert 0 < layers_after_first < twice.graph_layers
    assert twice.graph_layers == single.graph_layers
    assert twice.n_parameters == single.n_parameters


@pytest.mark.parametrize("fixture", FIXTURES)
def test_compose_then_single_build_matches_single_call(fixture: str) -> None:
    """Composing two circuit halves with ``+`` and building in one call matches.

    Composition is picture-independent: because the whole sequence is fed in a single
    build_graph call, there is no back-to-front reordering across calls to
    reason about (in either picture).
    """
    problem = load_problem(DATA / f"{fixture}.msgpack")
    circuit = problem.monomial_circuit.to_circuit()
    gates = circuit.gates
    split = len(gates) // 2
    params = list(map(float, problem.monomial_circuit.parameters))
    a = Circuit(_rebase(gates[:split]), parameters=tuple(params[:split]))
    b = Circuit(_rebase(gates[split:]), parameters=tuple(params[split:]))
    composed = a + b

    single = _propagator(problem)
    single.build_graph(circuit)

    fused = _propagator(problem)
    fused.build_graph(composed)

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
    single.build_graph(circuit)

    twice = _schrodinger_propagator(problem)
    twice.build_graph(Circuit(_rebase(gates[:split])))
    twice.build_graph(Circuit(_rebase(gates[split:])))

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
    prop.build_graph(Circuit(_rebase(gates[:split])))
    # seed_parameters on the second call exercises the internal seed regeneration used
    # for coefficient-informed truncation.
    prop.build_graph(Circuit(_rebase(gates[split:])), seed_parameters=params)

    np.testing.assert_allclose(prop.expectation_value(params), problem.exact_expval)


# --- Regression tests for the interface-refactor review ---------------------------------

_OBS = MajoranaOperator({(0, 1, 2, 4): 1.0}, 8)


def _small_propagator(**kwargs: object) -> MajoranaPropagator:
    return MajoranaPropagator(_OBS, initial_state=[], cutoff=16, **kwargs)  # type: ignore[arg-type]


def test_empty_default_mapping_gate_dropped_and_evaluable() -> None:
    """A zero-coefficient (identity) gate under the default mapping is dropped with its
    aligned angle, so the circuit stays evaluable instead of carrying a phantom parameter."""
    circuit = Circuit(
        gates=(
            ExpGate(MajoranaOperator({(4, 5): -1.0j}, num_modes=8)),
            ExpGate(
                MajoranaOperator({(6, 7): 0.0}, num_modes=8)
            ),  # identity generator: dropped
        ),
        parameters=(0.5, 0.3),
    )
    assert len(circuit.gates) == 1
    assert circuit.n_parameters == 1
    assert circuit.parameters == (0.5,)

    prop = _small_propagator()
    prop.build_graph(circuit)
    assert prop.graph_layers == 1
    prop.expectation_value(circuit.parameters)  # no parameter-length mismatch


def test_empty_gate_in_middle_builds_contiguously() -> None:
    """An identity gate between real gates is dropped, keeping the gate indices contiguous."""
    circuit = Circuit(
        gates=(
            ExpGate(MajoranaOperator({(4, 5): -1.0j}, num_modes=8)),
            ExpGate(
                MajoranaOperator({(6, 7): 0.0}, num_modes=8)
            ),  # identity in the middle
            ExpGate(MajoranaOperator({(2, 3): -1.0j}, num_modes=8)),
        ),
        parameters=(0.5, 0.3, 0.2),
    )
    assert len(circuit.gates) == 2
    prop = _small_propagator()
    prop.build_graph(circuit)  # gate indices stay contiguous across the drop
    assert prop.graph_layers == 2


def test_surplus_parameters_raise_not_truncated() -> None:
    """More angle values than gates raises rather than silently truncating (fermi path)."""
    generator = FermiOperator(
        [[(0, "+"), (1, "-")], [(1, "+"), (0, "-")]], [1.0, 1.0], num_modes=4
    )
    with pytest.raises(ValueError, match="1 parameter"):
        Circuit(gates=(ExpGate(generator),), parameters=(1.0, 2.0))


def test_non_commuting_pauli_generator_rejected() -> None:
    """A multi-term Pauli gate whose terms anticommute is rejected at construction."""
    with pytest.raises(ValueError, match="anticommute"):
        ExpGate(PauliOperator({Pauli("X", 0): 1.0, Pauli("Z", 0): 1.0}, num_qubits=1))


def test_commuting_pauli_generator_accepted() -> None:
    """A multi-term Pauli gate whose terms mutually commute is accepted."""
    ExpGate(
        PauliOperator(
            {Pauli("XX", (0, 1)): 1.0, Pauli("ZZ", (0, 1)): 1.0}, num_qubits=2
        )
    )


@pytest.mark.parametrize(
    ("terms", "should_raise"),
    [
        pytest.param({(0,): 1.0, (1,): 1.0}, True, id="odd_odd_anticommuting"),
        pytest.param({(0, 1): 1.0j, (2, 3): 1.0j}, False, id="even_even_commuting"),
        pytest.param(
            {(0, 2, 3, 5): 1.0, (0, 2, 4, 6): 1.0},
            False,
            id="even_even_commuting_overlap",
        ),
        pytest.param(
            {(0,): 1.0, (0, 1): 1.0}, True, id="mixed_parity_lengths_anticommuting"
        ),
        pytest.param(
            {(0,): 1.0, (1, 2): 1.0}, False, id="mixed_parity_lengths_commuting"
        ),
    ],
)
def test_majorana_generator_commutation_validation(
    terms: dict[tuple[int, ...], complex], *, should_raise: bool
) -> None:
    """Majorana multi-term gates are accepted iff all terms pairwise commute."""
    if should_raise:
        with pytest.raises(ValueError, match="anticommute"):
            ExpGate(MajoranaOperator(terms, num_modes=10))
    else:
        ExpGate(MajoranaOperator(terms, num_modes=10))


def test_build_graph_seed_parameters_accepts_numpy() -> None:
    """seed_parameters may be a NumPy array (an accepted ParameterValues type)."""
    circuit = Circuit(
        gates=(
            ExpGate(MajoranaOperator({(4, 5): -1.0j}, num_modes=8)),
            ExpGate(MajoranaOperator({(2, 3): -1.0j}, num_modes=8)),
        ),
        parameters=(0.5, 0.3),
    )
    prop = _small_propagator(lower_atol=1e-12)
    prop.build_graph(circuit, seed_parameters=np.array([0.5, 0.3]))


def test_build_graph_rejects_too_short_seed() -> None:
    """A too-short explicit seed raises a clear length error, not an out-of-bounds read."""
    circuit = Circuit(
        gates=(
            ExpGate(MajoranaOperator({(4, 5): -1.0j}, num_modes=8)),
            ExpGate(MajoranaOperator({(2, 3): -1.0j}, num_modes=8)),
        ),
        parameters=(0.5, 0.3),
    )
    prop = _small_propagator()
    with pytest.raises(RuntimeError, match="length of parameters"):
        prop.build_graph(circuit, seed_parameters=[0.5])


def test_extend_without_seed_builds_structurally() -> None:
    """Extending a non-empty graph without a seed builds the new layers structurally (no
    raise, no silent corruption); the result matches a single-call build, and an explicit
    full-axis seed is still accepted."""
    c1 = Circuit(
        gates=(ExpGate(MajoranaOperator({(4, 5): -1.0j}, num_modes=8)),),
        parameters=(0.3,),
    )
    c2 = Circuit(
        gates=(ExpGate(MajoranaOperator({(2, 3): -1.0j}, num_modes=8)),),
        parameters=(0.4,),
    )
    params = [0.3, 0.4]

    single = _small_propagator(lower_atol=1e-15)
    single.build_graph(c1 + c2)
    reference = single.expectation_value(params)

    extended = _small_propagator(lower_atol=1e-15)
    extended.build_graph(c1)
    extended.build_graph(c2)  # no seed_parameters: structural extension, no raise
    assert extended.n_parameters == 2
    np.testing.assert_allclose(extended.expectation_value(params), reference)

    seeded = _small_propagator(lower_atol=1e-15)
    seeded.build_graph(c1)
    seeded.build_graph(c2, seed_parameters=params)  # explicit full-axis seed
    np.testing.assert_allclose(seeded.expectation_value(params), reference)


def test_propagate_after_build_graph_rejected() -> None:
    """propagate() on top of a build_graph() graph raises rather than corrupting it."""
    c1 = Circuit(
        gates=(ExpGate(MajoranaOperator({(4, 5): -1.0j}, num_modes=8)),),
        parameters=(0.3,),
    )
    c2 = Circuit(
        gates=(ExpGate(MajoranaOperator({(2, 3): -1.0j}, num_modes=8)),),
        parameters=(0.4,),
    )
    prop = _small_propagator()
    prop.build_graph(c1)
    with pytest.raises(RuntimeError, match="non-empty graph"):
        prop.propagate(c2)
