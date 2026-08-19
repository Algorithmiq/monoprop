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

"""Random-problem benchmarks (time + peak memory), both physical pictures."""

from __future__ import annotations

from monoprop_bench_tools.models import barrier_setup, barriered

PARE_THRESHOLD = 1e-10
INPLACE_LOWER_ATOL = 1e-5


def test_random_build_graph(
    benchmark,
    make_random_propagator,
    bench_comm,
    bench_rounds,
    op_memory,
    record_opsize,
):
    """Benchmark building the propagation graph from a fresh propagator."""
    last = []

    def setup():
        built = make_random_propagator()
        last[:] = [built[0]]
        # Opened here rather than around pedantic() so the window covers the timed call
        # and not the propagator construction, which dwarfs it.
        op_memory.open()
        return (built,), {}

    def build(built):
        propagator, circuit = built
        propagator.build_graph(circuit)

    benchmark.pedantic(
        barriered(build, bench_comm),
        setup=barrier_setup(bench_comm, setup),
        rounds=bench_rounds,
        iterations=1,
    )
    op_memory.close(last[0])
    record_opsize(last[0])


def test_random_propagate(
    benchmark,
    make_random_propagator,
    bench_comm,
    bench_rounds,
    op_memory,
    record_opsize,
):
    """Benchmark in-place evolution alone, with no expectation value and no graph.

    Separate from ``test_random_inplace``, which fuses ``propagate`` with an expectation
    value and truncates at a coefficient threshold: neither is wrong, but neither isolates
    the propagation. This one leaves ``lower_atol`` at the propagator default so it carries
    the same operator as ``test_random_build_graph`` and the two are comparable.

    A fresh propagator per round because ``propagate`` refuses to run on an instance that
    already holds a graph.
    """
    last = []

    def setup():
        built = make_random_propagator()
        last[:] = [built[0]]
        op_memory.open()
        return (built,), {}

    def run(built):
        propagator, circuit = built
        propagator.propagate(circuit)

    benchmark.pedantic(
        barriered(run, bench_comm),
        setup=barrier_setup(bench_comm, setup),
        rounds=bench_rounds,
        iterations=1,
    )
    op_memory.close(last[0])
    record_opsize(last[0])


def test_random_pare(benchmark, built_graph, bench_comm, bench_rounds):
    """Benchmark paring the graph."""

    def pare():
        return built_graph.expectation_value_and_gradient_functional(
            pare_threshold=PARE_THRESHOLD,
        )

    benchmark.pedantic(
        barriered(pare, bench_comm),
        setup=barrier_setup(bench_comm),
        rounds=bench_rounds,
        iterations=1,
    )


def test_random_energy(
    benchmark, built_graph, random_problem, bench_comm, bench_rounds, op_memory
):
    """Benchmark evaluating the expectation-value functional.

    The functional is built outside the timed region on purpose: ``expectation_value()``
    rebuilds it on every call, which copies the whole operator, so timing that would time
    the copy rather than the evaluation.
    """
    functional = built_graph.expectation_value_functional()
    op_memory.open()

    # The args ride a setup callable rather than pedantic's args=, because pedantic rejects
    # both at once and the entry barrier has to live in a setup to stay untimed.
    def setup():
        return (random_problem.parameters,), {}

    result = benchmark.pedantic(
        barriered(functional, bench_comm),
        setup=barrier_setup(bench_comm, setup),
        rounds=bench_rounds,
        iterations=1,
    )
    # Small, and legitimately so: the graph this walks is already resident, so the delta is
    # scratch only. Read it next to opbytes.graph or it looks like the operation is free.
    op_memory.close(built_graph)
    assert isinstance(result, float)


def test_random_gradient(
    benchmark, built_graph, random_problem, bench_comm, bench_rounds, op_memory
):
    """Benchmark evaluating the expectation-value-and-gradient functional.

    Contains the whole energy forward pass -- there is no API for the reverse pass alone --
    so read this against ``test_random_energy`` rather than as a standalone cost.
    """
    functional = built_graph.expectation_value_and_gradient_functional()
    op_memory.open()

    def setup():
        return (random_problem.parameters,), {}

    _value, gradient = benchmark.pedantic(
        barriered(functional, bench_comm),
        setup=barrier_setup(bench_comm, setup),
        rounds=bench_rounds,
        iterations=1,
    )
    op_memory.close(built_graph)
    assert len(gradient) == len(random_problem.parameters)


def test_random_inplace(benchmark, make_random_propagator, bench_comm, bench_rounds):
    """Benchmark in-place evolution + expectation value (no graph stored)."""

    def setup():
        return (make_random_propagator(lower_atol=INPLACE_LOWER_ATOL),), {}

    def run(built):
        propagator, circuit = built
        propagator.propagate(circuit)
        return propagator.expectation_value()

    result = benchmark.pedantic(
        barriered(run, bench_comm),
        setup=barrier_setup(bench_comm, setup),
        rounds=bench_rounds,
        iterations=1,
    )
    assert isinstance(result, float)
