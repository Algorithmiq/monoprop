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

"""Max-sins truncation (``max_sins=`` / ``build_graph(..., count_sins=...)``) against a dict oracle."""

from __future__ import annotations

import numpy as np
import pytest

from monoprop import (
    Circuit,
    ExpGate,
    MajoranaPropagator,
    PauliOperator,
    PauliPropagator,
)
from tests import blockwise_oracle as oracle
from tests.test_blockwise_truncation import (
    N_MODES,
    STATE,
    _excitation_gates,
    _random_commuting_gate,
    _random_hamiltonian,
    _random_runs,
)
from tests.test_only_rotate_k import _split_orbital_gates


def _build(
    hamiltonian,
    gates,
    theta,
    cutoff,
    max_sins,
    *,
    blocks=None,
    seeded=False,
    comm=None,
    **kwargs,
):
    mp = MajoranaPropagator(
        hamiltonian, STATE, cutoff=cutoff, comm=comm, max_sins=max_sins, **kwargs
    )
    circuit = Circuit(
        gates, N_MODES, parameters=theta if seeded else (), initial_state=STATE
    )
    mp.build_graph(circuit, truncation_blocks=blocks)
    return mp


def _live(op):
    """Rows the engine stores: the identity is kept as a scalar core term, not a row."""
    return sum(1 for term in op if term)


def _reindexed(gates, n_modes, parameters=()):
    """Circuit from a slice of another circuit's gates, with fresh parameter indices."""
    return Circuit(
        tuple(ExpGate._with_index(gate, None) for gate in gates),
        system_size=n_modes,
        parameters=tuple(parameters),
    )


def _energy_and_grad(mp, theta):
    energy, grad = mp.expval_and_grad(theta)
    return energy, np.asarray(grad)


def _oracle_state(hamiltonian, schrodinger_cutoff):
    if schrodinger_cutoff is None:
        return {tuple(t): complex(v) for t, v in hamiltonian.terms.items()}, {}
    state = oracle.initial_state_terms(N_MODES, STATE, schrodinger_cutoff)
    fresh = {"fresh_paired": lambda term: oracle.reference_coefficient(term, STATE)}
    return state, fresh


def _oracle_evolve(hamiltonian, circuit, theta, schrodinger_cutoff=None, **kwargs):
    """Evolved oracle operator/state; ``kwargs`` go to ``oracle.evolve``."""
    start, fresh = _oracle_state(hamiltonian, schrodinger_cutoff)
    return oracle.evolve(
        start,
        circuit,
        theta,
        schrodinger=schrodinger_cutoff is not None,
        **fresh,
        **kwargs,
    )


@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("cutoff", [2, 4, 6])
@pytest.mark.parametrize("max_sins", [0, 1, 2, 3, 5])
def test_matches_oracle_on_excitations(max_sins, cutoff, seeded, comm):
    """Energy, gradient and term count against the oracle, standard structural truncation."""
    rng = np.random.default_rng(1000 + 10 * cutoff + max_sins)
    hamiltonian = _random_hamiltonian(rng, cutoff)
    gates = _excitation_gates(rng, 7)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    mp = _build(hamiltonian, gates, theta, cutoff, max_sins, seeded=seeded, comm=comm)
    energy, grad = _energy_and_grad(mp, theta)
    circuit = Circuit(gates, N_MODES)
    kwargs = {"cutoff": cutoff, "max_sins": max_sins}
    assert energy == pytest.approx(
        oracle.energy(hamiltonian, circuit, theta, STATE, **kwargs), abs=1e-12
    )
    fd = oracle.central_difference_gradient(
        hamiltonian, circuit, theta, STATE, **kwargs
    )
    np.testing.assert_allclose(grad, fd, atol=1e-7)
    if comm is None:
        assert mp.size() == _live(_oracle_evolve(hamiltonian, circuit, theta, **kwargs))


@pytest.mark.parametrize("seed", range(6))
def test_matches_oracle_with_truncation_blocks(seed):
    """Max-sins is checked per rotation, inside blocks too, and AND-ed with the block rule."""
    rng = np.random.default_rng(2000 + seed)
    cutoff = int(rng.choice([2, 4]))
    max_sins = int(rng.integers(1, 4))
    hamiltonian = _random_hamiltonian(rng, cutoff)
    gates = [_random_commuting_gate(rng) for _ in range(8)]
    labels = _random_runs(rng, len(gates))
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    mp = _build(hamiltonian, gates, theta, cutoff, max_sins, blocks=labels)
    energy, grad = _energy_and_grad(mp, theta)
    circuit = Circuit(gates, N_MODES)
    kwargs = {"cutoff": cutoff, "blocks": labels, "max_sins": max_sins}
    assert energy == pytest.approx(
        oracle.energy(hamiltonian, circuit, theta, STATE, **kwargs), abs=1e-12
    )
    fd = oracle.central_difference_gradient(
        hamiltonian, circuit, theta, STATE, **kwargs
    )
    np.testing.assert_allclose(grad, fd, atol=1e-7)


@pytest.mark.parametrize("max_sins", [0, 1, 2, 4])
@pytest.mark.parametrize(("cutoff", "schrodinger_cutoff"), [(2, 2), (4, 4), (4, 6)])
def test_schrodinger_matches_oracle(cutoff, schrodinger_cutoff, max_sins, comm):
    """In Schrodinger a paired partner is part of the reference state: never declined."""
    rng = np.random.default_rng(3000 + 10 * cutoff + schrodinger_cutoff + max_sins)
    hamiltonian = _random_hamiltonian(rng)
    gates = _excitation_gates(rng, 6) + [_random_commuting_gate(rng) for _ in range(3)]
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    mp = _build(
        hamiltonian,
        gates,
        theta,
        cutoff,
        max_sins,
        comm=comm,
        schrodinger_cutoff=schrodinger_cutoff,
    )
    energy, grad = _energy_and_grad(mp, theta)
    circuit = Circuit(gates, N_MODES)
    kwargs = {
        "cutoff": cutoff,
        "schrodinger_cutoff": schrodinger_cutoff,
        "max_sins": max_sins,
    }
    assert energy == pytest.approx(
        oracle.energy(hamiltonian, circuit, theta, STATE, **kwargs), abs=1e-12
    )
    if schrodinger_cutoff != cutoff:
        # With schrodinger_cutoff > cutoff the analytic gradient already differs from
        # finite differences on main (max_sins=None), so only the energy is compared.
        return
    fd = oracle.central_difference_gradient(
        hamiltonian, circuit, theta, STATE, **kwargs
    )
    np.testing.assert_allclose(grad, fd, atol=1e-7)


def test_extending_calls_continue_the_counts():
    """Counts persist across build_graph calls; count_sins=False makes one call's sines free."""
    rng = np.random.default_rng(41)
    cutoff, max_sins = 4, 2
    hamiltonian = _random_hamiltonian(rng, cutoff)
    first = _excitation_gates(rng, 4)
    second = _excitation_gates(rng, 3)
    third = _excitation_gates(rng, 3)
    mp = MajoranaPropagator(hamiltonian, STATE, cutoff=cutoff, max_sins=max_sins)
    calls = [(first, True), (second, False), (third, True)]
    for gates, count in calls:
        mp.build_graph(Circuit(gates, N_MODES, initial_state=STATE), count_sins=count)
    # Heisenberg: build_graph(a); build_graph(b) is the circuit b + a.
    order = [g for gates, _ in reversed(calls) for g in gates]
    theta = rng.uniform(0, 2 * np.pi, len(order))
    energy = mp.expval_and_grad(theta)[0]

    op, _ = _oracle_state(hamiltonian, None)
    counts: dict[tuple[int, ...], int] = {}
    offset = len(order)
    for gates, count in calls:
        offset -= len(gates)
        op = oracle.evolve(
            op,
            Circuit(gates, N_MODES),
            theta[offset : offset + len(gates)],
            cutoff=cutoff,
            max_sins=max_sins,
            count_sins=count,
            sin_counts=counts,
        )
    assert energy == pytest.approx(oracle.expectation(op, STATE), abs=1e-12)
    assert mp.size() == _live(op)


def test_large_limit_is_bit_identical_to_off(problem, comm):
    """No term is ever declined at the largest limit, so the graph is the untruncated one."""
    circuit = problem.monomial_circuit.to_circuit()
    parameters = problem.monomial_circuit.parameters
    results = []
    for max_sins in (None, 254):
        mp = MajoranaPropagator(
            problem.operator,
            problem.monomial_circuit.initial_state,
            cutoff=4,
            comm=comm,
            max_sins=max_sins,
        )
        mp.build_graph(circuit)
        energy, grad = mp.expval_and_grad(parameters)
        results.append((energy, list(grad), mp.size(), mp.graph_size()))
    assert results[0] == results[1]


def test_size_is_monotone_in_the_limit(lih_problem):
    """A larger limit never keeps fewer terms; 0 creates none."""
    circuit = lih_problem.monomial_circuit.to_circuit()
    sizes = []
    for max_sins in (0, 1, 2, 3, 4, None):
        mp = MajoranaPropagator(
            lih_problem.operator,
            lih_problem.monomial_circuit.initial_state,
            cutoff=6,
            max_sins=max_sins,
        )
        initial = mp.size()
        mp.build_graph(circuit)
        if max_sins == 0:
            assert mp.size() == initial
        sizes.append(mp.size())
    assert sizes == sorted(sizes)
    assert sizes[1] < sizes[-1]


@pytest.mark.parametrize("partitions", [1, 3])
def test_partition_count_does_not_change_results(partitions, monkeypatch):
    """The cross-partition path carries counts in the query records."""
    monkeypatch.setenv("monoprop_PARTITIONS", str(partitions))
    rng = np.random.default_rng(51)
    hamiltonian = _random_hamiltonian(rng, 2)
    gates = _excitation_gates(rng, 8)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    mp = _build(hamiltonian, gates, theta, 2, 1, seeded=True)
    energy, grad = _energy_and_grad(mp, theta)
    circuit = Circuit(gates, N_MODES)
    kwargs = {"cutoff": 2, "max_sins": 1}
    assert energy == pytest.approx(
        oracle.energy(hamiltonian, circuit, theta, STATE, **kwargs), abs=1e-12
    )
    assert mp.size() == _live(_oracle_evolve(hamiltonian, circuit, theta, **kwargs))
    np.testing.assert_allclose(
        grad,
        oracle.central_difference_gradient(
            hamiltonian, circuit, theta, STATE, **kwargs
        ),
        atol=1e-7,
    )


def _orbital_split(problem):
    circuit = problem.monomial_circuit.to_circuit()
    parameters = problem.monomial_circuit.parameters
    head, tail = _split_orbital_gates(circuit.gates)
    split = len(head)
    assert tail, "fixture has no orbital-rotation tail"

    return (
        _reindexed(head, problem.n_modes, parameters[:split]),
        _reindexed(tail, problem.n_modes, parameters[split:]),
        parameters,
    )


def test_uncounted_candidates_are_never_declined(orbital_rotation_problem):
    """Candidate orbital rotations built with count_sins=False create every term they would
    without max-sins, on top of the max-sins-truncated base graph."""
    problem = orbital_rotation_problem
    head, tail, parameters = _orbital_split(problem)
    kwargs = {"cutoff": 6, "schrodinger_cutoff": 8, "max_sins": 2}
    state = problem.monomial_circuit.initial_state

    free = MajoranaPropagator(problem.operator, state, **kwargs)
    free.build_graph(head)
    free.build_graph(tail, count_sins=False)

    reference = MajoranaPropagator(problem.operator, state, **kwargs)
    reference.build_graph(head)
    reference.max_sins = None
    reference.build_graph(tail)

    counted = MajoranaPropagator(problem.operator, state, **kwargs)
    counted.build_graph(head)
    counted.build_graph(tail)

    assert free.size() == reference.size()
    assert free.graph_size() == reference.graph_size()
    assert free.expval_functional()(parameters) == reference.expval_functional()(
        parameters
    )
    assert counted.size() < free.size()

    # Lowering the limit below the counts the head left does not make free gates decline.
    lowered = MajoranaPropagator(problem.operator, state, **kwargs)
    lowered.build_graph(head)
    lowered.max_sins = 0
    lowered.build_graph(tail, count_sins=False)
    assert lowered.graph_size() == reference.graph_size()


def test_reset_and_toggle(lih_problem):
    """reset_sin_counts() and switching max_sins on both restart every count at 0."""
    circuit = lih_problem.monomial_circuit.to_circuit()
    state = lih_problem.monomial_circuit.initial_state
    gates = circuit.gates
    half = len(gates) // 2
    first = _reindexed(gates[:half], circuit.system_size)
    second = _reindexed(gates[half:], circuit.system_size)

    def run(setup):
        mp = MajoranaPropagator(lih_problem.operator, state, cutoff=6, max_sins=1)
        mp.build_graph(first)
        setup(mp)
        mp.build_graph(second)
        return mp.size(), mp.graph_size()

    reset = run(lambda mp: mp.reset_sin_counts())
    toggled = run(
        lambda mp: (setattr(mp, "max_sins", None), setattr(mp, "max_sins", 1))
    )
    kept = run(lambda _mp: None)
    assert reset == toggled
    assert reset[0] > kept[0]

    mp = MajoranaPropagator(lih_problem.operator, state, cutoff=4)
    breakdown = mp._simulator.operator_memory_breakdown()
    assert breakdown["sin_count_bytes"] == 0
    mp.max_sins = 3
    assert mp.max_sins == 3
    mp.build_graph(first)
    assert mp._simulator.operator_memory_breakdown()["sin_count_bytes"] >= mp.size()
    mp.max_sins = None
    assert mp._simulator.operator_memory_breakdown()["sin_count_bytes"] == 0
    mp.reset_sin_counts()  # a no-op when off


@pytest.mark.parametrize("schrodinger_cutoff", [None, 4])
def test_pauli_propagator(schrodinger_cutoff):
    """The rule is basis-independent; on Pauli it is checked by its invariants."""
    n = 4
    hamiltonian = PauliOperator(
        {"ZZII": 0.7, "IXXI": -0.4, "IIYY": 0.3, "XIIY": 0.2, "IZIZ": 0.5}, num_qubits=n
    )
    generators = [
        {"XYII": 0.5, "YXII": -0.5},
        {"IXYI": 0.8, "IYXI": -0.8},
        {"IIXY": 0.6, "IIYX": -0.6},
        {"XXYY": 0.3, "YYXX": -0.3, "XYYX": 0.4, "YXXY": -0.4},
        {"XZZY": 0.7, "YZZX": -0.7},
    ]
    gates = [ExpGate(PauliOperator(t, num_qubits=n)) for t in generators]
    circuit = Circuit(gates, n, initial_state=(0, 2))
    theta = np.linspace(0.3, 1.7, len(gates))
    results = {}
    for max_sins in (None, 254, 0, 1, 2):
        mp = PauliPropagator(
            hamiltonian,
            (0, 2),
            cutoff=4,
            schrodinger_cutoff=schrodinger_cutoff,
            max_sins=max_sins,
        )
        initial = mp.size()
        mp.build_graph(circuit)
        energy = mp.expval_and_grad(theta)[0]
        results[max_sins] = (energy, mp.size())
        if max_sins == 0 and schrodinger_cutoff is None:
            assert mp.size() == initial
    assert results[254] == results[None]
    assert results[1][1] <= results[2][1] <= results[None][1]
    assert abs(results[1][0] - results[None][0]) > 1e-6


def test_propagate_rejects_max_sins():
    rng = np.random.default_rng(0)
    gates = _excitation_gates(rng, 3)
    theta = tuple(rng.uniform(0, 2 * np.pi, len(gates)))
    circuit = Circuit(gates, N_MODES, parameters=theta, initial_state=STATE)
    mp = MajoranaPropagator(_random_hamiltonian(rng), STATE, cutoff=2, max_sins=2)
    with pytest.raises(RuntimeError, match="build_graph"):
        mp.propagate(circuit)
    mp.max_sins = None
    mp.propagate(circuit)


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        (255, RuntimeError),
        (1000, RuntimeError),
        (-1, TypeError),
        (1.5, TypeError),
        ("2", TypeError),
    ],
)
def test_rejects_malformed_max_sins(bad, error):
    """Like the other truncation settings: the engine rejects a bad value, the binding a bad type."""
    rng = np.random.default_rng(0)
    hamiltonian = _random_hamiltonian(rng)
    with pytest.raises(error):
        MajoranaPropagator(hamiltonian, STATE, cutoff=2, max_sins=bad)
    mp = MajoranaPropagator(hamiltonian, STATE, cutoff=2)
    with pytest.raises(error):
        mp.max_sins = bad
    assert mp.max_sins is None


def test_constructor_and_property_agree():
    """max_sins given at construction behaves exactly like setting the property afterwards."""
    rng = np.random.default_rng(7)
    hamiltonian = _random_hamiltonian(rng, 4)
    gates = _excitation_gates(rng, 6)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    built = _build(hamiltonian, gates, theta, 4, 2)
    late = MajoranaPropagator(hamiltonian, STATE, cutoff=4)
    late.max_sins = 2
    late.build_graph(Circuit(gates, N_MODES, initial_state=STATE))
    assert built.max_sins == late.max_sins == 2
    assert built.graph_size() == late.graph_size()
    assert built.expval_and_grad(theta)[0] == late.expval_and_grad(theta)[0]
