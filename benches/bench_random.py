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

"""Random-problem benchmarks (time + peak memory), both physical pictures.

Random fixed-length Majorana generators and a random Hermitian observable, run in
the Heisenberg and Schrödinger pictures (``schrodinger_cutoff = cutoff + 2``).
Configurable from the command line -- see ``benches/README.md``.
"""

from __future__ import annotations

from _builders import barriered

PARE_THRESHOLD = 1e-10
INPLACE_LOWER_ATOL = 1e-5


def test_random_build_graph(
    benchmark, make_random_propagator, bench_comm, bench_rounds
):
    """Benchmark building the propagation graph from a fresh propagator."""

    def setup():
        return (make_random_propagator(),), {}

    def build(built):
        propagator, gates, _parameters = built
        propagator.propagate_build_graph(gates)

    benchmark.pedantic(
        barriered(build, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )


def test_random_pare(benchmark, built_graph, bench_comm, bench_rounds):
    """Benchmark paring the graph into a masked execution plan."""

    def pare():
        return built_graph.expectation_value_and_gradient_functional(
            pare_threshold=PARE_THRESHOLD,
        )

    benchmark.pedantic(barriered(pare, bench_comm), rounds=bench_rounds, iterations=1)


def test_random_energy(
    benchmark, built_graph, random_problem, bench_comm, bench_rounds
):
    """Benchmark evaluating the expectation-value functional."""
    functional = built_graph.expectation_value_functional(
        pare_threshold=PARE_THRESHOLD,
    )
    result = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(random_problem.parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    assert isinstance(result, float)


def test_random_gradient(
    benchmark, built_graph, random_problem, bench_comm, bench_rounds
):
    """Benchmark evaluating the expectation-value-and-gradient functional."""
    functional = built_graph.expectation_value_and_gradient_functional(
        pare_threshold=PARE_THRESHOLD,
    )
    _value, gradient = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(random_problem.parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    assert len(gradient) == len(random_problem.parameters)


def test_random_inplace(benchmark, make_random_propagator, bench_comm, bench_rounds):
    """Benchmark in-place evolution + expectation value (no graph stored)."""

    def setup():
        return (make_random_propagator(lower_atol=INPLACE_LOWER_ATOL),), {}

    def run(built):
        propagator, gates, parameters = built
        propagator.propagate(gates, parameters)
        return propagator.expectation_value()

    result = benchmark.pedantic(
        barriered(run, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )
    assert isinstance(result, float)
