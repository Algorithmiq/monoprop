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

"""Blockwise truncation (``build_graph(..., truncation_blocks=...)``) against a dict oracle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from monoprop import (
    Circuit,
    ExpGate,
    FermiOperator,
    MajoranaOperator,
    MajoranaPropagator,
    PauliOperator,
    PauliPropagator,
)
from tests import blockwise_oracle as oracle

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

N_MODES = 5
STATE = (0, 1)


def _random_hamiltonian(
    rng: np.random.Generator,
    cutoff: int | None = None,
    cutoff_type: str = "length",
    n: int = N_MODES,
    n_terms: int = 14,
) -> MajoranaOperator:
    """Hermitian observable of weight-2 and weight-4 terms (molecular-like, length <= 4).

    With ``cutoff``, every unpaired term is within it (a failing draw is replaced by a paired
    weight-4 term). That is the regime where blockwise truncation is the literal rule the
    oracle implements: a block's last layer keeps the emit-time cutoff, which differs from it
    only for unpaired terms over the cutoff that exist before the block (see the docs).
    """
    terms: dict[tuple[int, ...], complex] = {(): 0.25}
    for _ in range(n_terms):
        weight = int(rng.choice([2, 4]))
        term = tuple(sorted(rng.choice(2 * n, size=weight, replace=False).tolist()))
        if cutoff is not None and not oracle.passes(term, cutoff, cutoff_type):
            a, b = sorted(int(m) for m in rng.choice(n, 2, replace=False))
            term = (2 * a, 2 * a + 1, 2 * b, 2 * b + 1)
            weight = 4
        value = rng.normal()
        terms[term] = 1j * value if weight == 2 else value
    return MajoranaOperator(terms, n)


def _excitation(
    n: int, p: int, q: int, r: int | None = None, s: int | None = None
) -> FermiOperator:
    """Hermitian generator ``i (T - T^dag)`` of a single (p<-q) or double (pq<-rs) excitation."""
    if r is None or s is None:
        strings = [((p, "+"), (q, "-")), ((q, "+"), (p, "-"))]
    else:
        strings = [
            ((p, "+"), (q, "+"), (r, "-"), (s, "-")),
            ((s, "+"), (r, "+"), (q, "-"), (p, "-")),
        ]
    return FermiOperator(strings, [1j, -1j], n)


def _excitation_gates(
    rng: np.random.Generator, count: int, n: int = N_MODES
) -> list[ExpGate]:
    gates = []
    for _ in range(count):
        if rng.random() < 0.5:
            p, q = (int(x) for x in rng.choice(n, 2, replace=False))
            gates.append(ExpGate(_excitation(n, p, q)))
        else:
            p, q, r, s = (int(x) for x in rng.choice(n, 4, replace=False))
            gates.append(ExpGate(_excitation(n, p, q, r, s)))
    return gates


def _random_commuting_gate(
    rng: np.random.Generator, n: int = N_MODES, max_terms: int = 3
) -> ExpGate:
    """A generator of up to ``max_terms`` pairwise-commuting Hermitian Majorana strings."""
    chosen: list[tuple[int, ...]] = []
    for _ in range(40):
        weight = int(rng.choice([2, 4]))
        term = tuple(sorted(rng.choice(2 * n, size=weight, replace=False).tolist()))
        if term not in chosen and not any(
            oracle._anticommute(term, other) for other in chosen
        ):
            chosen.append(term)
        if len(chosen) == max_terms:
            break
    terms = {t: (1j if len(t) == 2 else 1.0) * rng.uniform(0.3, 1.2) for t in chosen}
    return ExpGate(MajoranaOperator(terms, n))


def _random_runs(rng: np.random.Generator, count: int) -> list[str]:
    """Random contiguous block labels, one per gate."""
    labels: list[str] = []
    block = 0
    while len(labels) < count:
        labels.extend([f"b{block}"] * int(rng.integers(1, 4)))
        block += 1
    return labels[:count]


def _split_into_layers(circuit: Circuit) -> Circuit:
    """One single-term gate per generator term, sharing the original gate's angle.

    Built with ``truncation_blocks="gate"``, this is one block per layer: Invariant A.
    """
    gates = []
    for gate, index in zip(circuit.gates, circuit.resolved_mapping, strict=True):
        for term, coeff in gate.generator.terms.items():
            gates.append(
                ExpGate(
                    MajoranaOperator({term: coeff}, circuit.system_size),
                    index=index,
                    _structural=gate._structural,
                )
            )
    return Circuit(
        gates,
        circuit.system_size,
        parameters=circuit.parameters,
        initial_state=circuit.initial_state,
    )


def _permute_terms(
    gates: Sequence[ExpGate], rng: np.random.Generator, n: int = N_MODES
) -> list[ExpGate]:
    """The same gates with each generator's (commuting) terms in a shuffled order."""
    out = []
    for gate in gates:
        items = list(gate.generator.terms.items())
        order = rng.permutation(len(items))
        out.append(ExpGate(MajoranaOperator(dict(items[i] for i in order), n)))
    return out


def _propagator(hamiltonian, cutoff, *, cutoff_type="length", comm=None, **kwargs):
    return MajoranaPropagator(
        hamiltonian, STATE, cutoff=cutoff, cutoff_type=cutoff_type, comm=comm, **kwargs
    )


def _energy_and_grad(
    hamiltonian: MajoranaOperator,
    gates: Sequence[ExpGate],
    theta: np.ndarray,
    cutoff: int,
    blocks: Sequence[Hashable] | str | None,
    *,
    cutoff_type: str = "length",
    comm=None,
    seeded: bool = False,
    **kwargs,
) -> tuple[float, np.ndarray, MajoranaPropagator]:
    mp = _propagator(hamiltonian, cutoff, cutoff_type=cutoff_type, comm=comm, **kwargs)
    circuit = Circuit(
        gates, N_MODES, parameters=theta if seeded else (), initial_state=STATE
    )
    mp.build_graph(circuit, truncation_blocks=blocks)
    energy, grad = mp.expval_and_grad(theta)
    return energy, np.asarray(grad), mp


def _retired_rows(mp) -> int:
    return mp._simulator.operator_memory_breakdown()["d_retired_rows"]


@pytest.mark.parametrize("cutoff_type", ["length", "support"])
@pytest.mark.parametrize("cutoff", [2, 3, 4, 6])
def test_oracle_matches_standard_truncation(cutoff, cutoff_type):
    """The oracle's per-layer mode reproduces monoprop's standard truncation (oracle self-check)."""
    rng = np.random.default_rng(10 + cutoff)
    hamiltonian = _random_hamiltonian(rng, cutoff, cutoff_type)
    gates = _excitation_gates(rng, 6)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    energy, _, _ = _energy_and_grad(
        hamiltonian, gates, theta, cutoff, None, cutoff_type=cutoff_type
    )
    circuit = Circuit(gates, N_MODES)
    expected = oracle.energy(
        hamiltonian, circuit, theta, STATE, cutoff=cutoff, cutoff_type=cutoff_type
    )
    assert energy == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("cutoff_type", ["length", "support"])
@pytest.mark.parametrize("cutoff", [2, 3, 4, 6])
def test_gate_blocks_match_oracle_on_excitations(cutoff, cutoff_type, seeded, comm):
    """The main use case: one block per single/double excitation gate."""
    rng = np.random.default_rng(100 + cutoff)
    hamiltonian = _random_hamiltonian(rng, cutoff, cutoff_type)
    gates = _excitation_gates(rng, 7)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    energy, grad, _ = _energy_and_grad(
        hamiltonian,
        gates,
        theta,
        cutoff,
        "gate",
        cutoff_type=cutoff_type,
        comm=comm,
        seeded=seeded,
    )
    circuit = Circuit(gates, N_MODES)
    kwargs = {"cutoff": cutoff, "cutoff_type": cutoff_type, "blocks": "gate"}
    assert energy == pytest.approx(
        oracle.energy(hamiltonian, circuit, theta, STATE, **kwargs), abs=1e-12
    )
    fd = oracle.central_difference_gradient(
        hamiltonian, circuit, theta, STATE, **kwargs
    )
    np.testing.assert_allclose(grad, fd, atol=1e-7)


@pytest.mark.parametrize("seed", range(6))
def test_multi_gate_blocks_match_oracle(seed):
    """Random commuting multi-term gates grouped into random multi-gate blocks."""
    rng = np.random.default_rng(seed)
    cutoff = int(rng.choice([2, 4, 6]))
    cutoff_type = str(rng.choice(["length", "support"]))
    hamiltonian = _random_hamiltonian(rng, cutoff, cutoff_type)
    gates = [_random_commuting_gate(rng) for _ in range(8)]
    labels = _random_runs(rng, len(gates))
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    energy, grad, _ = _energy_and_grad(
        hamiltonian, gates, theta, cutoff, labels, cutoff_type=cutoff_type
    )
    circuit = Circuit(gates, N_MODES)
    kwargs = {"cutoff": cutoff, "cutoff_type": cutoff_type, "blocks": labels}
    assert energy == pytest.approx(
        oracle.energy(hamiltonian, circuit, theta, STATE, **kwargs), abs=1e-12
    )
    fd = oracle.central_difference_gradient(
        hamiltonian, circuit, theta, STATE, **kwargs
    )
    np.testing.assert_allclose(grad, fd, atol=1e-7)


@pytest.mark.parametrize("pare_threshold", [None, 1e-10])
def test_one_block_per_layer_is_standard_truncation(problem, pare_threshold, comm):
    """Invariant A: every layer its own block is standard truncation, bit for bit."""
    circuit = problem.monomial_circuit.to_circuit()
    parameters = problem.monomial_circuit.parameters
    results = []
    for build_circuit, blocks in [
        (circuit, None),
        (_split_into_layers(circuit), "gate"),
    ]:
        mp = MajoranaPropagator(
            problem.operator, circuit.initial_state, cutoff=4, comm=comm
        )
        mp.build_graph(build_circuit, truncation_blocks=blocks)
        energy = mp.expval_functional(pare_threshold=pare_threshold)(parameters)
        _, grad = mp.expval_and_grad(parameters)
        results.append((energy, np.asarray(grad), mp.size(), _retired_rows(mp)))
    (e_std, g_std, n_std, _), (e_blk, g_blk, n_blk, retired) = results
    assert retired == 0
    assert n_blk == n_std
    assert e_blk == e_std
    np.testing.assert_array_equal(g_blk, g_std)


def test_one_block_per_layer_is_standard_truncation_on_random_gates():
    """Invariant A, with multi-term gates and a structural (unseeded) build."""
    rng = np.random.default_rng(7)
    hamiltonian = _random_hamiltonian(rng)
    circuit = Circuit(
        [_random_commuting_gate(rng) for _ in range(8)], N_MODES, initial_state=STATE
    )
    theta = rng.uniform(0, 2 * np.pi, circuit.n_parameters)
    energies = []
    for build_circuit, blocks in [
        (circuit, None),
        (_split_into_layers(circuit), "gate"),
    ]:
        mp = _propagator(hamiltonian, 3)
        mp.build_graph(build_circuit, truncation_blocks=blocks)
        energies.append(mp.expval_and_grad(theta))
    assert energies[0][0] == energies[1][0]
    np.testing.assert_array_equal(energies[0][1], energies[1][1])


def test_one_block_over_the_whole_circuit_is_exact(problem, comm):
    """Invariant B: the only cut is at the end, where unpaired terms contribute nothing."""
    circuit = problem.monomial_circuit.to_circuit()
    parameters = problem.monomial_circuit.parameters
    mp = MajoranaPropagator(
        problem.operator, circuit.initial_state, cutoff=2, comm=comm
    )
    mp.build_graph(circuit, truncation_blocks=[0] * len(circuit.gates))
    energy = mp.expval_functional()(parameters)
    assert energy == pytest.approx(problem.exact_expval, abs=1e-12)

    huge = MajoranaPropagator(
        problem.operator, circuit.initial_state, cutoff=2 * problem.n_modes, comm=comm
    )
    huge.build_graph(circuit)
    assert energy == pytest.approx(huge.expval_functional()(parameters), abs=1e-12)


def test_permuting_terms_inside_a_gate_only_matters_without_blocks():
    """Commuting strings of one gate: order-independent per gate, order-dependent per layer."""
    rng = np.random.default_rng(3)
    hamiltonian = _random_hamiltonian(rng, 2)
    gates = _excitation_gates(rng, 8)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    standard_spread = 0.0
    for cutoff in (2, 3, 4):
        blocked = []
        standard = []
        for perm_seed in range(4):
            permuted = (
                gates
                if perm_seed == 0
                else _permute_terms(gates, np.random.default_rng(perm_seed))
            )
            blocked.append(
                _energy_and_grad(hamiltonian, permuted, theta, cutoff, "gate")[0]
            )
            standard.append(
                _energy_and_grad(hamiltonian, permuted, theta, cutoff, None)[0]
            )
        np.testing.assert_allclose(blocked, blocked[0], atol=1e-12)
        standard_spread = max(standard_spread, max(standard) - min(standard))
    # The test can tell: standard truncation does depend on the order at these cutoffs.
    assert standard_spread > 1e-6


def test_retired_monomial_regenerated_in_a_later_block():
    """A monomial retired at one block end and produced again later must restart from zero.

    Searches fixed seeds for a circuit where this happens and affects the energy, i.e. where the
    naive implementation (a retired monomial keeps its dead slot) gives a different answer.
    """
    for seed in range(60):
        rng = np.random.default_rng(1000 + seed)
        hamiltonian = _random_hamiltonian(rng, 2)
        gates = _excitation_gates(rng, 8)
        theta = rng.uniform(0, 2 * np.pi, len(gates))
        circuit = Circuit(gates, N_MODES)
        stats: dict[str, int] = {}
        kwargs = {"cutoff": 2, "blocks": "gate"}
        correct = oracle.energy(
            hamiltonian, circuit, theta, STATE, stats=stats, **kwargs
        )
        naive = oracle.energy(
            hamiltonian, circuit, theta, STATE, naive_regeneration=True, **kwargs
        )
        if stats["regenerated"] > 0 and abs(correct - naive) > 1e-6:
            break
    else:
        pytest.fail("no seed produced a regeneration that changes the energy")

    energy, grad, mp = _energy_and_grad(hamiltonian, gates, theta, 2, "gate")
    assert energy == pytest.approx(correct, abs=1e-12)
    fd = oracle.central_difference_gradient(
        hamiltonian, circuit, theta, STATE, **kwargs
    )
    np.testing.assert_allclose(grad, fd, atol=1e-7)
    # Every row is either reachable through the lookup or retired, and some retired monomial is
    # held again by a live row: the store repeats it.
    live_terms = mp.evolved_operator(theta, atol=0.0).terms
    assert mp.size() == len([t for t in live_terms if t]) + _retired_rows(mp)


def test_evolved_operator_matches_oracle_without_duplicates():
    rng = np.random.default_rng(21)
    hamiltonian = _random_hamiltonian(rng, 2)
    gates = _excitation_gates(rng, 7)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    _, _, mp = _energy_and_grad(hamiltonian, gates, theta, 2, "gate")
    assert _retired_rows(mp) > 0
    evolved = mp.evolved_operator(theta, atol=0.0).terms
    expected = oracle.evolve(
        hamiltonian, Circuit(gates, N_MODES), theta, cutoff=2, blocks="gate"
    )
    assert set(evolved) == set(expected)
    for term, value in expected.items():
        assert complex(evolved[term]) == pytest.approx(value, abs=1e-12)


def test_term_expectations_and_reweighting_after_retirement():
    rng = np.random.default_rng(22)
    hamiltonian = _random_hamiltonian(rng, 2)
    gates = _excitation_gates(rng, 7)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    energy, _, mp = _energy_and_grad(hamiltonian, gates, theta, 2, "gate")
    assert _retired_rows(mp) > 0
    terms = [t for t in hamiltonian.terms if t]
    per_term = mp.term_expectation_values(terms, theta)
    contributions = sum(
        complex(hamiltonian.terms[t]) * v for t, v in zip(terms, per_term, strict=True)
    )
    assert hamiltonian.terms[()] + contributions.real == pytest.approx(
        energy, abs=1e-12
    )

    doubled = MajoranaOperator(
        {t: 2 * v for t, v in hamiltonian.terms.items()}, N_MODES
    )
    mp.update_initial_operator(doubled)
    assert mp.expval_functional()(theta) == pytest.approx(2 * energy, abs=1e-12)


def test_upper_atol_keeps_large_long_terms_at_block_ends():
    rng = np.random.default_rng(23)
    hamiltonian = _random_hamiltonian(rng, 2)
    gates = _excitation_gates(rng, 7)
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    circuit = Circuit(gates, N_MODES)
    upper_atol = 0.05
    energy, _, _ = _energy_and_grad(
        hamiltonian, gates, theta, 2, "gate", seeded=True, upper_atol=upper_atol
    )
    expected = oracle.energy(
        hamiltonian,
        circuit,
        theta,
        STATE,
        cutoff=2,
        blocks="gate",
        upper_atol=upper_atol,
    )
    assert energy == pytest.approx(expected, abs=1e-12)
    without = oracle.energy(hamiltonian, circuit, theta, STATE, cutoff=2, blocks="gate")
    assert abs(expected - without) > 1e-6  # the rescue changed the outcome


@pytest.mark.parametrize("seeded", [False, True])
def test_extension_across_build_graph_calls(seeded):
    """build_graph(a); build_graph(b) is the single build of b + a; every call closes its blocks."""
    rng = np.random.default_rng(31)
    hamiltonian = _random_hamiltonian(rng, 2)
    gates_a = _excitation_gates(rng, 4)
    gates_b = _excitation_gates(rng, 3)
    theta_a = rng.uniform(0, 2 * np.pi, len(gates_a))
    theta_b = rng.uniform(0, 2 * np.pi, len(gates_b))
    theta = np.concatenate([theta_b, theta_a])  # the accumulated axis of b + a
    labels_a = _random_runs(rng, len(gates_a))
    labels_b = _random_runs(rng, len(gates_b))

    split = _propagator(hamiltonian, 2)
    circuit_a = Circuit(
        gates_a, N_MODES, parameters=theta_a if seeded else (), initial_state=STATE
    )
    circuit_b = Circuit(
        gates_b, N_MODES, parameters=theta_b if seeded else (), initial_state=STATE
    )
    split.build_graph(circuit_a, truncation_blocks=labels_a)
    split.build_graph(
        circuit_b, truncation_blocks=labels_b, seed_parameters=theta if seeded else None
    )

    joined = _propagator(hamiltonian, 2)
    joined.build_graph(
        circuit_b + circuit_a,
        truncation_blocks=[*labels_b, *(f"a{x}" for x in labels_a)],
    )
    assert split.expval_functional()(theta) == pytest.approx(
        joined.expval_functional()(theta), abs=1e-12
    )
    expected = oracle.energy(
        hamiltonian,
        Circuit(gates_b + gates_a, N_MODES),
        theta,
        STATE,
        cutoff=2,
        blocks=[*labels_b, *(f"a{x}" for x in labels_a)],
    )
    assert split.expval_functional()(theta) == pytest.approx(expected, abs=1e-12)


def test_pared_functional_matches_exact_graph():
    rng = np.random.default_rng(41)
    hamiltonian = _random_hamiltonian(rng, 2)
    gates = [_random_commuting_gate(rng) for _ in range(8)]
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    energy, grad, mp = _energy_and_grad(
        hamiltonian, gates, theta, 2, _random_runs(rng, len(gates))
    )
    assert _retired_rows(mp) > 0
    assert mp.expval_functional(pare_threshold=1e-12)(theta) == pytest.approx(
        energy, abs=1e-10
    )
    pared_energy, pared_grad = mp.expval_and_grad_functional(pare_threshold=1e-12)(
        theta
    )
    assert pared_energy == pytest.approx(energy, abs=1e-10)
    np.testing.assert_allclose(pared_grad, grad, atol=1e-10)


@pytest.mark.parametrize("partitions", [1, 3])
def test_partition_count_does_not_change_blockwise_results(partitions, monkeypatch):
    monkeypatch.setenv("monoprop_PARTITIONS", str(partitions))
    rng = np.random.default_rng(51)
    hamiltonian = _random_hamiltonian(rng, 2)
    gates = [_random_commuting_gate(rng) for _ in range(8)]
    labels = _random_runs(rng, len(gates))
    theta = rng.uniform(0, 2 * np.pi, len(gates))
    energy, grad, mp = _energy_and_grad(
        hamiltonian, gates, theta, 2, labels, seeded=True
    )
    assert _retired_rows(mp) > 0
    circuit = Circuit(gates, N_MODES)
    kwargs = {"cutoff": 2, "blocks": labels}
    assert energy == pytest.approx(
        oracle.energy(hamiltonian, circuit, theta, STATE, **kwargs), abs=1e-12
    )
    fd = oracle.central_difference_gradient(
        hamiltonian, circuit, theta, STATE, **kwargs
    )
    np.testing.assert_allclose(grad, fd, atol=1e-7)


def _pauli_problem():
    n = 4
    # Every non-Z-only term has weight <= 2, the cutoff below (the regime of the literal rule).
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
    return n, hamiltonian, generators


def _pauli_energy(cutoff, blocks, *, reverse_terms=False):
    n, hamiltonian, generators = _pauli_problem()
    gates = []
    for terms in generators:
        items = list(terms.items())
        if reverse_terms:
            items.reverse()
        gates.append(ExpGate(PauliOperator(dict(items), num_qubits=n)))
    circuit = Circuit(gates, n, initial_state=(0, 2))
    if blocks == "all":
        blocks = [0] * len(gates)
    mp = PauliPropagator(hamiltonian, (0, 2), cutoff=cutoff)
    mp.build_graph(circuit, truncation_blocks=blocks)
    return mp.expval_and_grad(np.linspace(0.3, 1.7, len(gates)))[0]


def test_pauli_propagator_blocks():
    """Invariant B and per-gate order independence on the Pauli basis (support = Pauli weight)."""
    exact = _pauli_energy(4, None)
    assert _pauli_energy(2, "all") == pytest.approx(exact, abs=1e-12)
    per_gate = _pauli_energy(2, "gate")
    assert per_gate == pytest.approx(
        _pauli_energy(2, "gate", reverse_terms=True), abs=1e-12
    )
    assert abs(per_gate - exact) > 1e-6  # the blocks did truncate


def test_rejects_malformed_blocks():
    rng = np.random.default_rng(0)
    gates = _excitation_gates(rng, 3)
    circuit = Circuit(gates, N_MODES, initial_state=STATE)
    mp = _propagator(_random_hamiltonian(rng), 2)
    with pytest.raises(ValueError, match="has 2 labels but the circuit has 3 gates"):
        mp.build_graph(circuit, truncation_blocks=[0, 0])
    with pytest.raises(ValueError, match="recurs at gate 2"):
        mp.build_graph(circuit, truncation_blocks=["a", "b", "a"])
    with pytest.raises(ValueError, match='must be None, "gate"'):
        mp.build_graph(circuit, truncation_blocks="layer")
    assert mp.graph_layers == 0


def test_rejects_blocks_in_schrodinger_picture():
    rng = np.random.default_rng(0)
    mp = _propagator(_random_hamiltonian(rng), 4, schrodinger_cutoff=4)
    circuit = Circuit(_excitation_gates(rng, 2), N_MODES, initial_state=STATE)
    with pytest.raises(ValueError, match="Heisenberg picture only"):
        mp.build_graph(circuit, truncation_blocks="gate")


def test_engine_rejects_blocks_in_schrodinger_picture():
    rng = np.random.default_rng(0)
    mp = _propagator(_random_hamiltonian(rng), 4, schrodinger_cutoff=4)
    with pytest.raises(ValueError, match="Heisenberg picture only"):
        mp._simulator.build_graph([(0, 1)], [0], [1.0], block_ids=[0])


def test_engine_rejects_non_contiguous_block_ids():
    rng = np.random.default_rng(0)
    mp = _propagator(_random_hamiltonian(rng), 4)
    with pytest.raises(RuntimeError, match="block_ids must be contiguous runs"):
        mp._simulator.build_graph(
            [(0, 1), (2, 3)], [0, 1], [1.0, 1.0], block_ids=[0, 2]
        )
