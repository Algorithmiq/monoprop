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

"""Intra-process shard equivalence at the Python API level.

A propagator built with ``shards>1`` (S single-threaded shard partitions over an in-process
ShmComm) must agree with the ordinary single-partition propagator within floating-point
accumulation tolerance — the same standard the MPI rank-count tests use. Exercises the full
Python -> bindings -> ShardGroup path. Fast (small fermionic fixture, S in {1,2,4})."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monoprop import MajoranaPropagator
from tests.cases import load_problem

_FIXTURE = Path(__file__).parent / "data" / "S0_8e8o_majoranic_c6.msgpack"


def _run(problem, shards: int) -> tuple[float, int]:
    mp = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
        shards=shards,
    )
    circuit = problem.monomial_circuit.to_circuit()
    mp.propagate(circuit)
    return mp.expectation_value(), mp.size()


@pytest.mark.parametrize("shards", [2, 4])
def test_shard_energy_matches_single_partition(shards: int) -> None:
    problem = load_problem(_FIXTURE)
    e1, n1 = _run(problem, 1)
    e, n = _run(problem, shards)
    # ULP-level accumulation differences are expected across shard counts (as across MPI ranks);
    # the operator size (hash-partitioned, non-overlapping) is exactly invariant.
    assert np.isclose(e1, e, rtol=1e-7, atol=1e-9), f"shards={shards}: {e} vs {e1}"
    assert n1 == n, f"shards={shards}: size {n} vs {n1}"


def test_shard_energy_is_deterministic() -> None:
    problem = load_problem(_FIXTURE)
    e_a, _ = _run(problem, 4)
    e_b, _ = _run(problem, 4)
    # Fixed rank-order ShmComm reduction ⇒ a given shard count has no run-to-run jitter.
    assert e_a == e_b


def test_shard_deep_copy_matches() -> None:
    import copy

    problem = load_problem(_FIXTURE)
    mp = MajoranaPropagator(
        problem.operator,
        problem.monomial_circuit.initial_state,
        cutoff=2 * problem.n_modes,
        shards=4,
    )
    circuit = problem.monomial_circuit.to_circuit()
    mp.propagate(circuit)
    e = mp.expectation_value()
    mp2 = copy.deepcopy(mp)  # clones the whole shard group (fresh threads + ShmComm)
    assert mp2.expectation_value() == e
    assert mp2.size() == mp.size()
