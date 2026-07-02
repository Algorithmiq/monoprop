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

"""Tests for the circuit authoring types and graph accumulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monoprop import (
    Gate,
    MajoranaPropagator,
    Parameter,
    ParameterVector,
    Term,
    gates_from_monomial_sequence,
    to_engine_arrays,
)
from tests.cases import load_problem

DATA = Path(__file__).parent / "data"
FIXTURES = ["rx_rz_ry_rz_exact", "random_exact", "lih_fermionic_spin_exact"]


def _propagator(problem):
    return MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
    )


# -- authoring types ------------------------------------------------------------


def test_to_engine_arrays_round_trips_sequence() -> None:
    """gates_from_monomial_sequence + to_engine_arrays reproduce the dense arrays."""
    problem = load_problem(DATA / "lih_fermionic_spin_exact.msgpack")
    mc = problem.monomial_circuit
    gates, params = gates_from_monomial_sequence(mc)
    majoranas, gen_coeffs, parameter_mapping = to_engine_arrays(gates, params)

    assert [tuple(m) for m in majoranas] == [tuple(m) for m in mc.majoranas]
    np.testing.assert_allclose(gen_coeffs, np.asarray(mc.gen_coeffs, dtype=float))
    assert parameter_mapping == [int(p) for p in mc.param_inds]


def test_shared_parameter_reduces_n_parameters() -> None:
    """Reusing one Parameter across gates ties them: fewer params, repeated mapping."""
    params = ParameterVector()
    shared = params.new()
    other = params.new()
    gates = [
        Gate(shared, (Term((0, 1), 1.0),)),
        Gate(other, (Term((2, 3), 1.0),)),
        Gate(shared, (Term((0, 3), 1.0),)),  # same angle as the first gate
    ]
    _, _, parameter_mapping = to_engine_arrays(gates, params)
    assert len(params) == 2
    assert parameter_mapping == [0, 1, 0]


def test_parameter_identity_is_by_object() -> None:
    """Two parameters with the same name are still distinct axes."""
    assert Parameter("theta") != Parameter("theta")
    p = Parameter("theta")
    assert len({p, p}) == 1  # same handle hashes/compares equal to itself


def test_bind_rejects_wrong_length() -> None:
    """ParameterVector.bind enforces the axis length for sequences."""
    params = ParameterVector([Parameter(), Parameter()])
    with pytest.raises(ValueError, match="Expected 2 parameters"):
        params.bind([1.0])


# -- evaluation contracts -------------------------------------------------------


@pytest.mark.parametrize("fixture", FIXTURES)
def test_expectation_value_matches_exact(fixture: str) -> None:
    """Building the graph then evaluating reproduces the exact expectation value."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    gates, _ = gates_from_monomial_sequence(problem.monomial_circuit)
    prop = _propagator(problem)
    prop.propagate_build_graph(gates)
    params = list(map(float, problem.monomial_circuit.parameters))

    assert prop.n_parameters == len(params)
    np.testing.assert_allclose(prop.expectation_value(params), problem.exact_expval)
    np.testing.assert_allclose(prop.gradient(params), problem.exact_gradient, atol=1e-9)


def test_eval_rejects_wrong_parameter_length() -> None:
    """expectation_value validates the parameter vector length against the graph."""
    problem = load_problem(DATA / "rx_rz_ry_rz_exact.msgpack")
    gates, _ = gates_from_monomial_sequence(problem.monomial_circuit)
    prop = _propagator(problem)
    prop.propagate_build_graph(gates)
    with pytest.raises(RuntimeError):
        prop.expectation_value([0.1])  # graph has more than one parameter


@pytest.mark.parametrize("fixture", FIXTURES)
def test_pared_functional_matches_unpared(fixture: str) -> None:
    """A pared functional agrees with the exact (unpared) evaluation."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    gates, _ = gates_from_monomial_sequence(problem.monomial_circuit)
    prop = _propagator(problem)
    prop.propagate_build_graph(gates)
    params = list(map(float, problem.monomial_circuit.parameters))

    pared = prop.expectation_value_functional(pare_threshold=1e-12)
    np.testing.assert_allclose(pared(params), problem.exact_expval, atol=1e-9)


# -- building the graph by propagating twice ------------------------------------


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
    gates, _ = gates_from_monomial_sequence(problem.monomial_circuit)
    split = len(gates) // 2

    single = _propagator(problem)
    single.propagate_build_graph(gates)

    twice = _propagator(problem)
    twice.propagate_build_graph(gates[:split])
    layers_after_first = twice.graph_layers
    twice.propagate_build_graph(gates[split:])

    assert 0 < layers_after_first < twice.graph_layers
    assert twice.graph_layers == single.graph_layers
    assert twice.n_parameters == single.n_parameters


@pytest.mark.parametrize("fixture", FIXTURES)
def test_build_graph_in_two_calls_schrodinger(fixture: str) -> None:
    """In Schrodinger picture, building in two forward calls equals a single call.

    (In Heisenberg picture each call applies its gates back-to-front, so a forward
    split is deliberately *not* equivalent to one call.)
    """
    problem = load_problem(DATA / f"{fixture}.msgpack")
    gates, _ = gates_from_monomial_sequence(problem.monomial_circuit)
    params = list(map(float, problem.monomial_circuit.parameters))
    split = len(gates) // 2

    single = _schrodinger_propagator(problem)
    single.propagate_build_graph(gates)

    twice = _schrodinger_propagator(problem)
    twice.propagate_build_graph(gates[:split])
    twice.propagate_build_graph(gates[split:])

    np.testing.assert_allclose(
        twice.expectation_value(params), single.expectation_value(params)
    )
    np.testing.assert_allclose(twice.expectation_value(params), problem.exact_expval)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_build_graph_twice_with_seed_regeneration(fixture: str) -> None:
    """Extending a non-empty graph with parameters regenerates the seed internally."""
    problem = load_problem(DATA / f"{fixture}.msgpack")
    gates, _ = gates_from_monomial_sequence(problem.monomial_circuit)
    params = list(map(float, problem.monomial_circuit.parameters))
    split = len(gates) // 2

    prop = _schrodinger_propagator(problem)
    prop.propagate_build_graph(gates[:split])
    # Passing parameters on the second call exercises the internal seed regeneration
    # (the former operator_coeffs round-trip) used for coefficient-informed truncation.
    prop.propagate_build_graph(gates[split:], params)

    np.testing.assert_allclose(prop.expectation_value(params), problem.exact_expval)
