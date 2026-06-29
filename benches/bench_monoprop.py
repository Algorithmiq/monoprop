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

"""Unified monoprop benchmark suite (time + peak memory).

Random benchmarks are configurable (see ``benches/README.md``) and run in both
the Heisenberg and Schrödinger pictures (the latter with
``schrodinger_cutoff = cutoff + 2``). Static benchmarks are fixed, heavy,
Heisenberg-only in-place simulations: the 120-qubit Fermi-Hubbard trajectory and
the 127-qubit Pauli-basis kicked-Ising circuit.

Operations are barrier-wrapped so the measured time is the makespan across MPI
ranks. Timing uses ``pytest-benchmark``; peak memory uses ``pytest-memray``
(``--memray``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from _builders import (
    HubbardConfig,
    KickedIsingConfig,
    barriered,
    build_hubbard_problem,
    build_kicked_ising_problem,
    build_random_propagator,
)

if TYPE_CHECKING:
    from _builders import RandomProblem

    from monoprop import MonomialPropagator

PARE_THRESHOLD = 1e-10
# In-place random bench keeps its own truncation constant (see design non-goals).
INPLACE_LOWER_ATOL = 1e-5


# --------------------------------------------------------------------------- #
# Random benchmarks (configurable; both pictures)
# --------------------------------------------------------------------------- #
@pytest.mark.bench
def test_random_build_graph(
    benchmark: object,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
    picture: str,
) -> None:
    """Benchmark building the propagation graph from a fresh propagator."""

    def setup() -> tuple[tuple[MonomialPropagator], dict]:
        mp = build_random_propagator(
            random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
        )
        return (mp,), {}

    def build(mp: MonomialPropagator) -> None:
        mp.propagate()

    benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(build, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )


@pytest.mark.bench
def test_random_pare(
    benchmark: object,
    built_graph: MonomialPropagator,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
) -> None:
    """Benchmark paring the graph into a masked execution plan."""

    def pare() -> object:
        return built_graph.expectation_value_and_gradient_functional(
            parameter_mapping=random_problem.parameter_mapping,
            gen_coeffs=random_problem.gen_coeffs,
            pare_threshold=PARE_THRESHOLD,
        )

    benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(pare, bench_comm), rounds=bench_rounds, iterations=1
    )


@pytest.mark.bench
def test_random_energy(
    benchmark: object,
    built_graph: MonomialPropagator,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
) -> None:
    """Benchmark evaluating the expectation-value functional."""
    functional = built_graph.expectation_value_functional(
        parameter_mapping=random_problem.parameter_mapping,
        gen_coeffs=random_problem.gen_coeffs,
        pare_threshold=PARE_THRESHOLD,
    )
    result = benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(functional, bench_comm),
        args=(random_problem.parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    assert isinstance(result, float)


@pytest.mark.bench
def test_random_gradient(
    benchmark: object,
    built_graph: MonomialPropagator,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
) -> None:
    """Benchmark evaluating the expectation-value-and-gradient functional."""
    functional = built_graph.expectation_value_and_gradient_functional(
        parameter_mapping=random_problem.parameter_mapping,
        gen_coeffs=random_problem.gen_coeffs,
        pare_threshold=PARE_THRESHOLD,
    )
    _value, gradient = benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(functional, bench_comm),
        args=(random_problem.parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    assert len(gradient) == len(random_problem.parameters)


@pytest.mark.bench
def test_random_inplace(
    benchmark: object,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
    picture: str,
) -> None:
    """Benchmark in-place evolution + expectation value (no graph stored)."""

    def setup() -> tuple[tuple[MonomialPropagator], dict]:
        mp = build_random_propagator(
            random_problem,
            comm=bench_comm,
            lower_atol=INPLACE_LOWER_ATOL,
            schrodinger=picture == "schrodinger",
        )
        return (mp,), {}

    def run(mp: MonomialPropagator) -> float:
        mp.propagate(evolve_with_coeffs=True)
        return mp.expectation_value()

    result = benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(run, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )
    assert isinstance(result, float)


# --------------------------------------------------------------------------- #
# Static benchmarks (fixed, heavy, Heisenberg-only, in-place)
# --------------------------------------------------------------------------- #
# Mapping name -> (builder, config factory, Trotter steps per measured run).
# Hubbard re-applies its one-step circuit ``trotter_steps`` times; the Pauli
# circuit already contains all layers, so a single propagate suffices.
STATIC_MODELS: dict[str, tuple] = {
    "hubbard": (build_hubbard_problem, HubbardConfig, HubbardConfig().trotter_steps),
    "pauli": (build_kicked_ising_problem, KickedIsingConfig, 1),
}


@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("model", list(STATIC_MODELS))
def test_static(
    benchmark: object,
    bench_comm: Any,
    lower_atol: float | None,
    model: str,
    record_model_config: Any,
) -> None:
    """Benchmark a fixed in-place static simulation (Heisenberg picture)."""
    build_fn, config_cls, steps = STATIC_MODELS[model]
    config = config_cls()
    if lower_atol is not None:
        config = replace(config, lower_atol=lower_atol)
    record_model_config(model, config)

    def setup() -> tuple[tuple[MonomialPropagator, int], dict]:
        return (build_fn(config, comm=bench_comm), steps), {}

    def run(mp: MonomialPropagator, n_steps: int) -> float:
        value = 0.0
        for _ in range(n_steps):
            mp.propagate(evolve_with_coeffs=True)
            value = mp.expectation_value()
        return value

    result = benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(run, bench_comm), setup=setup, rounds=1, iterations=1
    )
    assert isinstance(result, float)
