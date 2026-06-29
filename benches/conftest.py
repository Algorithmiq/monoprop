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

"""Pytest configuration and fixtures for the monoprop benchmark suite.

The random benchmarks are parameterised through command-line options so the
generator length, number of observable terms, number of generators,
mode count, cutoff, and RNG seed can all be varied without editing code, e.g.::

    uv run pytest benches --num-generators 200 --num-modes 64 --cutoff 10

MPI: the benchmarks are communicator-aware. When ``mpi4py`` is available the
``bench_comm`` fixture yields ``MPI.COMM_WORLD`` (size 1 for a serial run, size
R under ``mpiexec -n R``); otherwise it yields ``None``. Each measured operation
is wrapped in barriers (see :func:`_builders.barriered`) so the timed cost
reflects the slowest rank, and only rank 0 writes the timing JSON.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from _builders import RandomProblem, build_random_propagator, make_random_problem

if TYPE_CHECKING:
    from monoprop import MonomialPropagator

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - depends on optional MPI build
    MPI = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register benchmark configuration options for the random benchmarks."""
    group = parser.getgroup("monoprop-bench", "monoprop random benchmark sizing")
    group.addoption(
        "--gen-length",
        type=int,
        default=4,
        help="Number of Majorana operators per generator.",
    )
    group.addoption(
        "--obs-terms", type=int, default=10000, help="Number of observable terms."
    )
    group.addoption(
        "--num-generators",
        type=int,
        default=100,
        help="Number of random generators (circuit gates).",
    )
    group.addoption(
        "--num-modes", type=int, default=128, help="Number of fermionic modes."
    )
    group.addoption("--cutoff", type=int, default=6, help="Truncation cutoff.")
    group.addoption(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    group.addoption(
        "--bench-rounds",
        type=int,
        default=5,
        help="Fixed number of rounds for the random benchmarks (MPI-safe).",
    )
    group.addoption(
        "--lower-atol",
        type=float,
        default=None,
        help="Override lower_atol for the static (hubbard, pauli) benchmarks.",
    )


# Resolved random-problem sizes (and run knobs) recorded for the report so the
# hyperparameters a run actually used are visible alongside its numbers. The
# ``--`` CLI names map to plain keys here.
_RECORDED_OPTIONS = (
    "gen_length",
    "obs_terms",
    "num_generators",
    "num_modes",
    "cutoff",
    "seed",
    "bench_rounds",
    "lower_atol",
)


def _write_run_params(config: pytest.Config) -> None:
    """Record the resolved hyperparameters for this run label, if requested.

    ``benches/run.py`` sets ``MONOPROP_BENCH_LABEL`` and
    ``MONOPROP_BENCH_RESULTS`` so the report can show, per label, the sizes the
    benchmarks actually used -- defaults included, not just CLI overrides.
    Written only by rank 0; the timing and memory passes overwrite identically.
    """
    label = os.environ.get("MONOPROP_BENCH_LABEL")
    results = os.environ.get("MONOPROP_BENCH_RESULTS")
    if not label or not results:
        return
    params = {
        opt: config.getoption(f"--{opt.replace('_', '-')}") for opt in _RECORDED_OPTIONS
    }
    Path(results, f"params-{label}.json").write_text(json.dumps(params, indent=2))


def _record_graph_size(picture: str, mp: MonomialPropagator) -> None:
    """Record the number of terms (and graph metrics) reached for one picture.

    Merged into ``graphsize-<label>.json`` (one entry per picture) so the report
    can show how large the evolved operator actually grew in each picture.
    Written only by rank 0 when ``benches/run.py`` requested recording.
    """
    if MPI is not None and MPI.COMM_WORLD.Get_rank() != 0:
        return
    label = os.environ.get("MONOPROP_BENCH_LABEL")
    results = os.environ.get("MONOPROP_BENCH_RESULTS")
    if not label or not results:
        return
    path = Path(results, f"graphsize-{label}.json")
    data = json.loads(path.read_text()) if path.exists() else {}
    n_cos_indices, n_cycles = mp.graph_size()
    data[picture] = {
        "terms": mp.size(),
        "n_cos_indices": n_cos_indices,
        "n_cycles": n_cycles,
    }
    path.write_text(json.dumps(data, indent=2))


def pytest_configure(config: pytest.Config) -> None:
    """Rank-0 bookkeeping: guard the benchmark JSON and record hyperparameters.

    Under MPI every rank would otherwise write the benchmark JSON to the same
    path, racing on the file. Rank 0 holds the makespan timings (operations are
    barrier-wrapped), so the other ranks simply do not save. Rank 0 also records
    the run's resolved hyperparameters for the report.
    """
    rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()
    size = 1 if MPI is None else MPI.COMM_WORLD.Get_size()
    if size > 1 and rank != 0 and getattr(config.option, "benchmark_json", None):
        config.option.benchmark_json = None
    if rank == 0:
        _write_run_params(config)


@pytest.fixture(scope="session")
def bench_comm() -> Any:
    """Return the benchmark communicator (``MPI.COMM_WORLD`` or ``None``)."""
    return None if MPI is None else MPI.COMM_WORLD


@pytest.fixture(scope="session")
def bench_rounds(request: pytest.FixtureRequest) -> int:
    """Return the fixed round count for the random benchmarks."""
    return int(request.config.getoption("--bench-rounds"))


@pytest.fixture(scope="session")
def lower_atol(request: pytest.FixtureRequest) -> float | None:
    """Return the static-benchmark ``lower_atol`` override (``None`` if unset)."""
    return request.config.getoption("--lower-atol")


@pytest.fixture(params=["heisenberg", "schrodinger"])
def picture(request: pytest.FixtureRequest) -> str:
    """Parametrize the random benchmarks over the physical picture."""
    return request.param


@pytest.fixture(scope="session")
def random_problem(request: pytest.FixtureRequest) -> RandomProblem:
    """Build the random observable/circuit problem from the CLI options."""
    opt = request.config.getoption
    return make_random_problem(
        gen_length=opt("--gen-length"),
        obs_terms=opt("--obs-terms"),
        num_generators=opt("--num-generators"),
        num_modes=opt("--num-modes"),
        cutoff=opt("--cutoff"),
        seed=opt("--seed"),
    )


@pytest.fixture
def built_graph(
    random_problem: RandomProblem, bench_comm: Any, picture: str
) -> MonomialPropagator:
    """Return a propagator whose graph has been built (no coefficients contracted).

    Built fresh per test, in the requested picture, so its propagation graph is
    not shared between measurements; the build cost happens in fixture setup so
    it is excluded from the memory profile of the operation under test.
    """
    mp = build_random_propagator(
        random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
    )
    mp.propagate()
    _record_graph_size(picture, mp)
    return mp
