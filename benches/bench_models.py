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

Run one group per pytest process: build_graph/propagate and energy/gradient do not fit a node together.
"""

from __future__ import annotations

import os

import pytest
from monoprop_bench_tools.models import MODELS, barrier_setup, barriered

# `build_graph` extends the graph, so a driver that re-applies its circuit retains one layer-set per step
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
    """Benchmark building a fixed model's propagation graph, from a fresh propagator."""
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
        barriered(build, bench_comm),
        setup=barrier_setup(bench_comm, setup),
        rounds=bench_rounds,
        iterations=1,
    )
    op_memory.close(last[0])
    assert record_opsize(last[0]) > 0
    assert last[0].graph_layers > 0


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
    """Benchmark a fixed model's in-place evolution alone -- no expectation value, no graph."""
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
        barriered(run, bench_comm),
        setup=barrier_setup(bench_comm, setup),
        rounds=bench_rounds,
        iterations=1,
    )
    op_memory.close(last[0])
    assert record_opsize(last[0]) > 0


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model_energy(
    benchmark,
    model_graph,
    model,
    model_configs,
    bench_comm,
    bench_rounds,
    op_memory,
    pare_threshold,
):
    """Benchmark evaluating a fixed model's expectation-value functional."""
    skip_if_graph_will_not_fit(model, MODELS[model][2](model_configs[model]))
    propagator, parameters = model_graph(model)
    functional = propagator.expectation_value_functional(pare_threshold)
    op_memory.open()
    result = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(parameters,),
        setup=barrier_setup(bench_comm),
        rounds=bench_rounds,
        iterations=1,
    )
    op_memory.close(propagator)
    assert isinstance(result, float)


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model_gradient(
    benchmark,
    model_graph,
    model,
    model_configs,
    bench_comm,
    bench_rounds,
    op_memory,
    pare_threshold,
):
    """Benchmark evaluating a fixed model's expectation-value-and-gradient functional."""
    skip_if_graph_will_not_fit(model, MODELS[model][2](model_configs[model]))
    propagator, parameters = model_graph(model)
    functional = propagator.expectation_value_and_gradient_functional(pare_threshold)
    op_memory.open()
    _value, gradient = benchmark.pedantic(
        barriered(functional, bench_comm),
        args=(parameters,),
        setup=barrier_setup(bench_comm),
        rounds=bench_rounds,
        iterations=1,
    )
    op_memory.close(propagator)
    assert len(gradient) == len(parameters)
