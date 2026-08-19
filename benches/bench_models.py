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
in :data:`monoprop_bench_tools.models.MODELS`; each config field is overridable via
``--<model>-<field>``.

``test_model`` is the original fused run (propagate + expectation value), kept as-is because
it is a tracked Bencher series. Beside it, build_graph, propagate, energy and gradient each
get their own time and ``op_memory`` window.

Select one group at a time (``-k "build_graph or propagate"``): ``build_graph`` and
``propagate`` each hold their own operator while ``energy`` and ``gradient`` share the one
:func:`model_graph` builds, and all four together do not fit a node at these sizes.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from monoprop_bench_tools.memory.cpu import resting_rss_bytes
from monoprop_bench_tools.models import MODELS, barrier_setup, barriered

# `build_graph` extends the graph, so a driver that re-applies its circuit retains one
# layer-set per step: 3 steps of hubbard exceed a 242 GiB node where 2 fit in 15.4 GiB.
# Skipping beats OOM-killing the machine of whoever runs `just bench`, which does not
# pass -m "not slow". Pauli's step count is 1 and is unaffected.
MAX_GRAPH_STEPS = 2


def skip_if_graph_will_not_fit(model: str, steps: int) -> None:
    """Skip a graph-holding benchmark whose retained graph is known not to fit."""
    allow = os.environ.get("monoprop_BENCH_ALLOW_BIG_GRAPH")  # noqa: SIM112
    if steps > MAX_GRAPH_STEPS and allow != "1":
        pytest.skip(
            f"{model}: {steps} successive build_graph calls retain {steps} layer-sets, "
            f"measured to exceed 242 GiB. Set monoprop_BENCH_ALLOW_BIG_GRAPH=1 to run it."
        )


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
        barriered(run, bench_comm),
        setup=barrier_setup(bench_comm, setup),
        rounds=1,
        iterations=1,
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

    ``steps`` successive calls, because Hubbard's circuit is one Trotter step the driver
    re-applies: fewer would build a shorter graph than ``test_model_propagate`` evolves.
    """
    _config_cls, build_fn, steps_fn = MODELS[model]
    config = model_configs[model]
    steps = steps_fn(config)
    skip_if_graph_will_not_fit(model, steps)
    record_model_config(model, config)

    last = []

    def setup():
        built = build_fn(config, comm=bench_comm)
        last[:] = [built[0]]
        op_memory.open()  # inside setup, so the window excludes the construction
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

    Isolates the propagation that :func:`test_model` fuses with an expectation value. A fresh
    propagator per round, because ``propagate`` refuses an instance that holds a graph.
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
    benchmark, model_graph, model, model_configs, bench_comm, bench_rounds, op_memory
):
    """Benchmark evaluating a fixed model's expectation-value functional.

    The functional is built outside the timed region: ``expectation_value()`` rebuilds it per
    call, copying the whole operator, so timing that would time the copy.
    """
    skip_if_graph_will_not_fit(model, MODELS[model][2](model_configs[model]))
    propagator, parameters = model_graph(model)
    functional = propagator.expectation_value_functional()
    op_memory.open()
    result = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    # Legitimately small: the graph is already resident, so the delta is scratch only.
    # Read it next to opbytes.graph or the operation looks free.
    op_memory.close(propagator)
    assert isinstance(result, float)


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model_gradient(
    benchmark, model_graph, model, model_configs, bench_comm, bench_rounds, op_memory
):
    """Benchmark evaluating a fixed model's expectation-value-and-gradient functional.

    Contains the whole energy forward pass -- there is no API for the reverse pass alone --
    so read it against :func:`test_model_energy`, not as a standalone cost.
    """
    skip_if_graph_will_not_fit(model, MODELS[model][2](model_configs[model]))
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
