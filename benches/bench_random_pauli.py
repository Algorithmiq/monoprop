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

"""Random-Pauli benchmarks (time + peak memory), native Pauli engine.

The Pauli counterpart of ``bench_random.py``: random fixed-weight Pauli-rotation
generators and a random Hermitian Pauli observable, run through the native Pauli
engine (``engine_basis="pauli"``) in the Heisenberg picture. Unlike the fixed
kicked-Ising model (a mostly-dead Clifford-point operator whose per-gate work is
tiny), a random Pauli operator is dense and grows with the cutoff, so it exercises
the find-scan with genuinely thread-parallel per-gate work -- the regime where the
scan's parallel decomposition has to earn its keep.

Operations are barrier-wrapped so the measured time is the makespan across MPI
ranks. Timing uses ``pytest-benchmark``; peak memory is the per-test physical
footprint (PSS), recorded by the ``record_memory`` fixture in ``conftest.py``.
"""

from __future__ import annotations

from _builders import barriered

PARE_THRESHOLD = 1e-10
# In-place random bench keeps its own truncation constant (see design non-goals).
INPLACE_LOWER_ATOL = 1e-5


def test_random_pauli_build_graph(
    benchmark, make_random_pauli_propagator, bench_comm, bench_rounds
):
    """Benchmark building the propagation graph from a fresh Pauli propagator."""

    def setup():
        return (make_random_pauli_propagator(),), {}

    def build(built):
        propagator, circuit = built
        propagator.build_graph(circuit)

    benchmark.pedantic(
        barriered(build, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )


def test_random_pauli_pare(benchmark, built_pauli_graph, bench_comm, bench_rounds):
    """Benchmark paring the Pauli graph into a masked execution plan."""

    def pare():
        return built_pauli_graph.expectation_value_and_gradient_functional(
            pare_threshold=PARE_THRESHOLD,
        )

    benchmark.pedantic(barriered(pare, bench_comm), rounds=bench_rounds, iterations=1)


def test_random_pauli_energy(
    benchmark, built_pauli_graph, random_pauli_problem, bench_comm, bench_rounds
):
    """Benchmark evaluating the Pauli expectation-value functional."""
    functional = built_pauli_graph.expectation_value_functional()
    result = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(random_pauli_problem.parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    assert isinstance(result, float)


def test_random_pauli_gradient(
    benchmark, built_pauli_graph, random_pauli_problem, bench_comm, bench_rounds
):
    """Benchmark evaluating the Pauli expectation-value-and-gradient functional."""
    functional = built_pauli_graph.expectation_value_and_gradient_functional()
    _value, gradient = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(random_pauli_problem.parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    assert len(gradient) == len(random_pauli_problem.parameters)


def test_random_pauli_inplace(
    benchmark, make_random_pauli_propagator, bench_comm, bench_rounds
):
    """Benchmark in-place Pauli evolution + expectation value (no graph stored)."""

    def setup():
        return (make_random_pauli_propagator(lower_atol=INPLACE_LOWER_ATOL),), {}

    def run(built):
        propagator, circuit = built
        propagator.propagate(circuit)
        return propagator.expectation_value()

    result = benchmark.pedantic(
        barriered(run, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )
    assert isinstance(result, float)
