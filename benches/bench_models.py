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

"""Fixed-model benchmarks: heavy, Heisenberg-only simulations.

The 120-qubit Fermi-Hubbard trajectory and the 127-qubit Pauli-basis kicked-Ising
circuit, at fixed sizes. The registry (config class, builder, steps-per-run) lives
in ``_builders.MODELS``; each config field is overridable via ``--<model>-<field>``.

``test_model`` is the original fused in-place run (propagate + expectation value) and is
kept as-is: it is a tracked Bencher series, and renaming or replacing it would orphan its
history. The four operations beside it split that work the way ``bench_random.py`` splits
the random problem, so build_graph, propagate, energy and gradient each get their own time
and their own ``op_memory`` window.

The two groups do not belong in one process. ``build_graph`` and ``propagate`` each hold
their own operator, while ``energy`` and ``gradient`` share the one :func:`model_graph`
builds; running all four together holds two operators per rank and does not fit a node at
the sizes this is used at. Select one group at a time (``-k "build_graph or propagate"``).
"""

from __future__ import annotations

from typing import Any

import pytest
from _builders import MODELS, barriered
from _memory_cpu import resting_rss_bytes


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model(
    benchmark,
    bench_comm,
    model_configs,
    model,
    record_model_config,
    record_model_stats,
):
    """Benchmark a fixed in-place model simulation (Heisenberg picture)."""
    _config_cls, build_fn, steps_fn = MODELS[model]
    config = model_configs[model]
    steps = steps_fn(config)
    record_model_config(model, config)

    state: dict[str, Any] = {}

    def setup():
        state["baseline_rss"] = resting_rss_bytes()
        state["built"] = build_fn(config, comm=bench_comm)
        return (state["built"], steps), {}

    def run(built, n_steps):
        propagator, circuit = built
        for _ in range(n_steps):
            propagator.propagate(circuit)
        return propagator.expectation_value()

    result = benchmark.pedantic(
        barriered(run, bench_comm), setup=setup, rounds=1, iterations=1
    )
    assert isinstance(result, float)

    propagator, _circuit = state["built"]
    record_model_stats(model, propagator, state["baseline_rss"])


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model_build_graph(
    benchmark,
    bench_comm,
    bench_rounds,
    model_configs,
    model,
    record_model_config,
    op_memory,
    record_opsize,
):
    """Benchmark building a fixed model's propagation graph, from a fresh propagator.

    ``steps`` successive ``build_graph`` calls, because Hubbard's circuit is one Trotter step
    the driver re-applies: fewer would build a shorter graph than ``test_model_propagate``
    evolves, and the two would not be comparable. Pauli's step count is 1, so the same code
    attributes per-call cost 29x against 1x between the two models for free.
    """
    _config_cls, build_fn, steps_fn = MODELS[model]
    config = model_configs[model]
    steps = steps_fn(config)
    record_model_config(model, config)

    last = []

    def setup():
        built = build_fn(config, comm=bench_comm)
        last[:] = [built[0]]
        # Opened here rather than around pedantic() so the window covers the timed call
        # and not the propagator construction, which dwarfs it.
        op_memory.open()
        return (built, steps), {}

    def build(built, n_steps):
        propagator, circuit = built
        for _ in range(n_steps):
            propagator.build_graph(circuit)

    benchmark.pedantic(
        barriered(build, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )
    op_memory.close(last[0])
    record_opsize(last[0])


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model_propagate(
    benchmark,
    bench_comm,
    bench_rounds,
    model_configs,
    model,
    record_model_config,
    op_memory,
    record_opsize,
):
    """Benchmark a fixed model's in-place evolution alone -- no expectation value, no graph.

    Separate from :func:`test_model`, which fuses the same evolution with an expectation
    value: neither is wrong, but only this one isolates the propagation. A fresh propagator
    per round because ``propagate`` refuses to run on an instance that already holds a graph.
    """
    _config_cls, build_fn, steps_fn = MODELS[model]
    config = model_configs[model]
    steps = steps_fn(config)
    record_model_config(model, config)

    last = []

    def setup():
        built = build_fn(config, comm=bench_comm)
        last[:] = [built[0]]
        op_memory.open()
        return (built, steps), {}

    def run(built, n_steps):
        propagator, circuit = built
        for _ in range(n_steps):
            propagator.propagate(circuit)

    benchmark.pedantic(
        barriered(run, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )
    op_memory.close(last[0])
    record_opsize(last[0])


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model_energy(
    benchmark, model_graph, model, bench_comm, bench_rounds, op_memory
):
    """Benchmark evaluating a fixed model's expectation-value functional.

    The functional is built outside the timed region on purpose: ``expectation_value()``
    rebuilds it on every call, which copies the whole operator, so timing that would time the
    copy rather than the evaluation.
    """
    propagator, parameters = model_graph(model)
    functional = propagator.expectation_value_functional()
    op_memory.open()
    result = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    # Small, and legitimately so: the graph this walks is already resident, so the delta is
    # scratch only. Read it next to opbytes.graph or it looks like the operation is free.
    op_memory.close(propagator)
    assert isinstance(result, float)


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model_gradient(
    benchmark, model_graph, model, bench_comm, bench_rounds, op_memory
):
    """Benchmark evaluating a fixed model's expectation-value-and-gradient functional.

    Contains the whole energy forward pass -- there is no API for the reverse pass alone --
    so read this against :func:`test_model_energy` rather than as a standalone cost.
    """
    propagator, parameters = model_graph(model)
    functional = propagator.expectation_value_and_gradient_functional()
    op_memory.open()
    _value, gradient = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    op_memory.close(propagator)
    assert len(gradient) == len(parameters)
