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

"""Microbenchmarks isolating Python/C++ binding overhead."""

from __future__ import annotations

from itertools import combinations, islice

import pytest

import monoprop
from monoprop import MajoranaPropagator
from monoprop.fermi import MajoranaOperator

BOUNDARY_ROUNDS = 20
CHEAP_CALL_ITERATIONS = 10_000
FUNCTIONAL_ITERATIONS = 1_000
LARGE_TERM_COUNT = 1_024
NUM_MODES = 32
OUTPUT_ATOL = 1e-300


@pytest.fixture(scope="module", autouse=True)
def _serial_only(bench_comm):
    if bench_comm is not None and bench_comm.Get_size() > 1:
        pytest.skip("binding microbenchmarks are serial-only")


@pytest.fixture(scope="module")
def binding_propagators():
    """Build fixed small and large propagators outside timed regions."""
    small = MajoranaPropagator(
        MajoranaOperator({(0, 1, 2, 3): 1.0}, NUM_MODES),
        [],
        cutoff=2 * NUM_MODES,
    )
    large_terms = dict.fromkeys(
        islice(combinations(range(2 * NUM_MODES), 4), LARGE_TERM_COUNT), 1.0
    )
    large = MajoranaPropagator(
        MajoranaOperator(large_terms, NUM_MODES),
        [],
        cutoff=2 * NUM_MODES,
    )
    return {"small": small, "large": large}


def test_binding_is_antihermitian_positional(benchmark):
    """Benchmark positional free-function dispatch."""
    result = benchmark.pedantic(
        monoprop.is_antihermitian,
        args=([0, 1],),
        rounds=BOUNDARY_ROUNDS,
        iterations=CHEAP_CALL_ITERATIONS,
    )
    assert result is True


def test_binding_is_antihermitian_keyword(benchmark):
    """Benchmark keyword free-function dispatch."""
    result = benchmark.pedantic(
        monoprop.is_antihermitian,
        kwargs={"indices": [0, 1]},
        rounds=BOUNDARY_ROUNDS,
        iterations=CHEAP_CALL_ITERATIONS,
    )
    assert result is True


def test_binding_size(benchmark, binding_propagators):
    """Benchmark a bound method with a scalar return value."""
    simulator = binding_propagators["small"]._simulator
    result = benchmark.pedantic(
        simulator.size,
        rounds=BOUNDARY_ROUNDS,
        iterations=CHEAP_CALL_ITERATIONS,
    )
    assert result == 1


def test_binding_expectation_value_functional(benchmark, binding_propagators):
    """Benchmark calls through an already-created functional."""
    functional = binding_propagators["small"].expectation_value_functional()
    result = benchmark.pedantic(
        functional,
        args=([],),
        rounds=BOUNDARY_ROUNDS,
        iterations=FUNCTIONAL_ITERATIONS,
    )
    assert isinstance(result, float)


@pytest.mark.parametrize("operator_size", ["small", "large"])
def test_binding_evolved_operator(benchmark, binding_propagators, operator_size):
    """Benchmark raw evolved-operator dictionary conversion."""
    simulator = binding_propagators[operator_size]._simulator
    iterations = 100 if operator_size == "small" else 1
    expected_terms = 1 if operator_size == "small" else LARGE_TERM_COUNT
    result = benchmark.pedantic(
        simulator.evolved_operator,
        args=([], OUTPUT_ATOL),
        rounds=BOUNDARY_ROUNDS,
        iterations=iterations,
    )
    assert len(result) == expected_terms
