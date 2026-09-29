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

"""Coverage for ``evolved_operator_coefficients``, the term-probing companion to ``evolved_operator``.

Oracle throughout: ``evolved_operator(atol=0.0)``, which enumerates the whole index. Both are
rank-local, so every propagator here is built on ``serial_comm`` (see the note in ``test_basis.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monoprop import (
    Circuit,
    ExpGate,
    MajoranaPropagator,
    PauliPropagator,
)
from monoprop.majorana import Majorana, MajoranaOperator
from monoprop.pauli import Pauli, PauliOperator
from tests.cases import load_problem

DATA = Path(__file__).parent / "data"

N_QUBITS = 6


@pytest.fixture(scope="module")
def problem():
    """The Majorana case the tests below share; they read it and never mutate it."""
    return load_problem(DATA / "random_exact.msgpack")


@pytest.fixture(scope="module")
def prop_and_params(problem, serial_comm):
    """A built propagator over ``problem`` and its parameters, shared across the module.

    ``evolved_operator`` and ``evolved_operator_coefficients`` leave simulator state alone, so one
    graph build serves every test that needs it.
    """
    prop = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
        comm=serial_comm,
    )
    prop.build_graph(problem.monomial_circuit.to_circuit())
    return prop, problem.monomial_circuit.parameters


def _pauli_propagator(serial_comm, schrodinger_cutoff=None):
    """A Pauli propagator with a graph whose evolution spreads the operator over many terms."""
    prop = PauliPropagator(
        PauliOperator({Pauli("ZZ", (0, 1)): 1.0, Pauli("XY", (2, 3)): 0.5}, N_QUBITS),
        initial_state=[],
        cutoff=N_QUBITS,
        schrodinger_cutoff=schrodinger_cutoff,
        comm=serial_comm,
    )
    circuit = Circuit(
        tuple(
            ExpGate(PauliOperator({Pauli(letters, qubits): 1.0}, N_QUBITS))
            for letters, qubits in (
                ("XX", (0, 1)),
                ("YZ", (1, 2)),
                ("ZX", (2, 3)),
                ("XY", (3, 4)),
                ("YY", (4, 5)),
            )
        ),
        N_QUBITS,
    )
    prop.build_graph(circuit)
    return prop, np.linspace(0.1, 0.9, prop.n_parameters)


def test_pauli_coefficients_match_evolved_operator(serial_comm) -> None:
    """Every term the evolved Pauli operator carries reads back with the same coefficient."""
    prop, parameters = _pauli_propagator(serial_comm)

    evolved = prop.evolved_operator(parameters, atol=0.0)
    terms = list(evolved.terms)
    assert len(terms) > 1

    coefficients = prop.evolved_operator_coefficients(terms, parameters)

    assert coefficients.shape == (len(terms),)
    np.testing.assert_allclose(
        coefficients, [evolved.terms[term] for term in terms], atol=1e-12
    )


def test_majorana_coefficients_match_evolved_operator(prop_and_params) -> None:
    """Same property for the Majorana front-end, whose keys are raw index tuples."""
    prop, parameters = prop_and_params

    evolved = prop.evolved_operator(parameters, atol=0.0)
    terms = list(evolved.terms)
    assert len(terms) > 1

    coefficients = prop.evolved_operator_coefficients(terms, parameters)

    np.testing.assert_allclose(
        coefficients, [evolved.terms[term] for term in terms], atol=1e-12
    )


def test_majorana_accepts_majorana_terms(prop_and_params) -> None:
    """A Majorana term object keys the same coefficient its raw index tuple does."""
    prop, parameters = prop_and_params

    evolved = prop.evolved_operator(parameters, atol=0.0)
    terms = list(evolved.terms)[:4]

    from_tuples = prop.evolved_operator_coefficients(terms, parameters)
    from_objects = prop.evolved_operator_coefficients(
        [Majorana(*term) for term in terms], parameters
    )

    np.testing.assert_array_equal(from_tuples, from_objects)


def test_absent_term_reads_back_zero(serial_comm) -> None:
    """A term the evolved operator does not carry is 0, not a raise and not noise."""
    prop = PauliPropagator(
        PauliOperator({Pauli("Z", (0,)): 1.0}, N_QUBITS),
        initial_state=[],
        cutoff=N_QUBITS,
        comm=serial_comm,
    )

    absent = Pauli("X", (5,))
    assert absent not in prop.evolved_operator(atol=0.0).terms

    coefficients = prop.evolved_operator_coefficients([absent])

    assert coefficients.shape == (1,)
    assert coefficients[0] == 0


def test_schrodinger_coefficients_match_evolved_state(serial_comm) -> None:
    """The motivating case: reading a few amplitudes out of an evolved state."""
    prop, parameters = _pauli_propagator(serial_comm, schrodinger_cutoff=4)

    evolved = prop.evolved_operator(parameters, atol=0.0)
    terms = list(evolved.terms)
    assert len(terms) > 1

    coefficients = prop.evolved_operator_coefficients(terms, parameters)

    np.testing.assert_allclose(
        coefficients, [evolved.terms[term] for term in terms], atol=1e-12
    )


def test_query_order_is_preserved(prop_and_params) -> None:
    """``out[i]`` belongs to ``terms[i]`` whatever order the terms arrive in.

    Majorana rather than Pauli so the shuffled query spans many of the engine's batch-probe
    prefetch groups, not just the first.
    """
    prop, parameters = prop_and_params

    evolved = prop.evolved_operator(parameters, atol=0.0)
    terms = list(evolved.terms)
    assert len(terms) > 100

    shuffled = list(np.random.default_rng(0).permutation(len(terms)))
    reordered = [terms[i] for i in shuffled]

    coefficients = prop.evolved_operator_coefficients(reordered, parameters)

    np.testing.assert_allclose(
        coefficients, [evolved.terms[term] for term in reordered], atol=1e-12
    )


def test_identity_term_is_the_core_term(problem, serial_comm) -> None:
    """In the Heisenberg picture the empty term is the core term ``evolved_operator`` re-adds.

    The fixture Hamiltonian carries no identity term, so one is injected: with a zero core term the
    check could not tell the core term apart from the absent-term answer.
    """
    with_identity = MajoranaOperator(
        {**problem.operator.terms, (): 0.75}, problem.n_modes
    )
    prop = MajoranaPropagator(
        with_identity,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
        comm=serial_comm,
    )
    prop.build_graph(problem.monomial_circuit.to_circuit())
    parameters = problem.monomial_circuit.parameters

    evolved = prop.evolved_operator(parameters, atol=0.0)
    assert evolved.terms[()] == pytest.approx(0.75)

    coefficients = prop.evolved_operator_coefficients([()], parameters)

    assert coefficients[0] == pytest.approx(evolved.terms[()])


def test_schrodinger_identity_is_a_state_amplitude(serial_comm) -> None:
    """In the Schrodinger picture the identity is an ordinary term, not a core term."""
    prop, parameters = _pauli_propagator(serial_comm, schrodinger_cutoff=4)

    evolved = prop.evolved_operator(parameters, atol=0.0)
    identity = Pauli("", ())
    assert identity in evolved.terms

    coefficients = prop.evolved_operator_coefficients([identity], parameters)

    assert coefficients[0] == pytest.approx(evolved.terms[identity])


def test_empty_term_list_returns_empty_array(prop_and_params) -> None:
    """An empty query is an empty answer, not a degenerate probe."""
    prop, parameters = prop_and_params

    coefficients = prop.evolved_operator_coefficients([], parameters)

    assert coefficients.shape == (0,)


def test_out_of_range_term_raises(problem, prop_and_params) -> None:
    """The engine encodes with the checked path, so a term outside the system is rejected."""
    prop, parameters = prop_and_params

    with pytest.raises(RuntimeError, match="out of range"):
        prop.evolved_operator_coefficients([(2 * problem.n_modes,)], parameters)


@pytest.mark.parametrize(
    "term",
    [
        pytest.param((14, 0), id="unsorted"),
        pytest.param((0, 0), id="repeated"),
        pytest.param((-1,), id="negative"),
    ],
)
def test_non_canonical_majorana_term_is_rejected(prop_and_params, term) -> None:
    """A raw tuple that is not a canonical monomial raises instead of answering wrongly.

    The engine keys terms by an order-insensitive bitset, so ``(14, 0)`` would resolve to the row
    of ``(0, 14)`` and read back its coefficient *without* the anticommutation sign -- and a lookup
    has no coefficient of its own to put that sign on. ``(0, 0)`` would likewise resolve to the
    identity row. Both are silent wrong answers, so the front-end validates as
    [Majorana][monoprop.majorana.Majorana] does.
    """
    prop, parameters = prop_and_params

    with pytest.raises(ValueError, match="Majorana indices must be"):
        prop.evolved_operator_coefficients([term], parameters)


def test_canonicalizing_an_unsorted_majorana_term_recovers_the_sign(
    prop_and_params,
) -> None:
    """``Majorana.from_unsorted`` is the supported way to ask for a non-canonical product."""
    prop, parameters = prop_and_params

    evolved = prop.evolved_operator(parameters, atol=0.0)
    canonical = next(term for term in evolved.terms if len(term) == 2)

    term, sign = Majorana.from_unsorted(*reversed(canonical))
    assert term.indices == canonical
    assert sign == -1.0

    coefficient = sign * prop.evolved_operator_coefficients([term], parameters)[0]
    assert coefficient == pytest.approx(-evolved.terms[canonical])


def test_majorana_accepts_a_numpy_index_array(prop_and_params) -> None:
    """An index array is a valid term: the guard is on iterability, and ndarray is not a Sequence."""
    prop, parameters = prop_and_params

    evolved = prop.evolved_operator(parameters, atol=0.0)
    term = next(iter(evolved.terms))

    coefficients = prop.evolved_operator_coefficients([np.array(term)], parameters)

    assert coefficients[0] == pytest.approx(evolved.terms[term])


@pytest.mark.parametrize(
    "term",
    [
        pytest.param("", id="empty_str"),
        pytest.param(b"", id="empty_bytes"),
        pytest.param("01", id="digits"),
    ],
)
def test_string_like_majorana_term_is_rejected(prop_and_params, term) -> None:
    """A string is iterable, so ``""`` would expand to the identity term's coefficient."""
    prop, parameters = prop_and_params

    with pytest.raises(TypeError, match="Majorana objects or index sequences"):
        prop.evolved_operator_coefficients([term], parameters)


def test_majorana_front_end_rejects_a_pauli_term(prop_and_params) -> None:
    """A term from the other front-end's vocabulary raises TypeError, not an obscure failure."""
    prop, parameters = prop_and_params
    paulis = [Pauli("X", (0,))]

    with pytest.raises(TypeError, match="Majorana objects or index sequences"):
        prop.evolved_operator_coefficients(paulis, parameters)


def test_pauli_front_end_rejects_a_raw_slot_tuple(serial_comm) -> None:
    """Symplectic slots are an engine-internal encoding, so the Pauli front-end takes Pauli only."""
    prop, parameters = _pauli_propagator(serial_comm)

    with pytest.raises(TypeError, match="Pauli objects"):
        prop.evolved_operator_coefficients([(0, 1)], parameters)


def test_repeated_term_is_answered_once_per_occurrence(serial_comm) -> None:
    """The query is a list, not a set: each occurrence gets its own slot with the same value."""
    prop, parameters = _pauli_propagator(serial_comm)

    evolved = prop.evolved_operator(parameters, atol=0.0)
    term = next(iter(evolved.terms))

    coefficients = prop.evolved_operator_coefficients([term, term], parameters)

    assert coefficients.shape == (2,)
    assert coefficients[0] == coefficients[1] == pytest.approx(evolved.terms[term])
