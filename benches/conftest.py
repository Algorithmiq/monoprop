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
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from _builders import RandomProblem, build_random_propagator, make_random_problem

if TYPE_CHECKING:
    from collections.abc import Callable

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
        default=1,
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


def _record_operator_size(picture: str, mp: MonomialPropagator, comm: Any) -> None:
    """Record the total number of terms in the evolved operator for one picture.

    Merged into ``opsize-<label>.json`` (one entry per picture) so the report
    can show how large the evolved operator actually grew in each picture.

    Under MPI the operator is partitioned disjointly across ranks, so ``mp.size()``
    returns only this rank's shard; the global total is the sum of the per-rank
    sizes (an allreduce). The size is deterministic in the hyperparameters, so a
    serial run and an MPI run at the same hyperparameters report the same total.
    Only rank 0 writes the file.
    """
    total = mp.size()
    rank = 0
    if comm is not None and comm.Get_size() > 1:
        total = comm.allreduce(total, op=MPI.SUM)  # collective: all ranks must call
        rank = comm.Get_rank()
    if rank != 0:
        return
    label = os.environ.get("MONOPROP_BENCH_LABEL")
    results = os.environ.get("MONOPROP_BENCH_RESULTS")
    if not label or not results:
        return
    path = Path(results, f"opsize-{label}.json")
    data = json.loads(path.read_text()) if path.exists() else {}
    data[picture] = {"terms": total}
    path.write_text(json.dumps(data, indent=2))


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Make rank 0 the sole writer and printer under MPI; record hyperparameters.

    Every rank runs the whole pytest session, so without intervention all R ranks
    interleave their progress output and each opens the shared ``--benchmark-json``
    file (``pytest-benchmark`` opens it at parse time), racing on it and leaking
    the handle. Rank 0 holds the makespan timings (operations are barrier-wrapped)
    and is the only rank that prints and writes the JSON; the others go silent and
    close their JSON handle. Per-rank memory dumps (``--memray``) are unaffected.

    Runs ``trylast`` so the terminal reporter is already registered (it registers
    in its own ``pytest_configure``) by the time the non-root ranks unregister it.
    """
    rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        _write_run_params(config)
        return

    # Non-root MPI rank: unregister the terminal reporter so it does not
    # interleave its output with rank 0's, and close+drop the benchmark-JSON file
    # it opened at parse time so it neither writes nor leaks the handle.
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        config.pluginmanager.unregister(reporter)
    bench_json = getattr(config.option, "benchmark_json", None)
    if bench_json is not None:
        if hasattr(bench_json, "close"):
            bench_json.close()
        config.option.benchmark_json = None


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


@pytest.fixture
def record_model_config() -> Callable[[str, Any], None]:
    """Return a callable recording a static model's resolved config for the report.

    ``benches/run.py`` sets ``MONOPROP_BENCH_LABEL`` and
    ``MONOPROP_BENCH_RESULTS``; the recorder merges the model's dataclass fields
    into ``configs-<label>.json`` (one entry per model) so the report can show
    the configuration each static model actually ran with. The config is pure
    input and identical across ranks, so only rank 0 writes (mirroring
    :func:`_write_run_params`); the timing and memory passes overwrite
    identically.

    Returns:
        A callable ``record(model, config)`` taking the model name and its
        resolved (frozen) configuration dataclass.
    """

    def _record(model: str, config: Any) -> None:
        rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            return
        label = os.environ.get("MONOPROP_BENCH_LABEL")
        results = os.environ.get("MONOPROP_BENCH_RESULTS")
        if not label or not results:
            return
        path = Path(results, f"configs-{label}.json")
        data = json.loads(path.read_text()) if path.exists() else {}
        data[model] = asdict(config)
        path.write_text(json.dumps(data, indent=2))

    return _record


@pytest.fixture(scope="session", params=["heisenberg", "schrodinger"])
def picture(request: pytest.FixtureRequest) -> str:
    """Parametrize the random benchmarks over the physical picture.

    Session-scoped so the (expensive) shared :func:`built_graph` can be built
    once per picture and reused across the graph-based benchmarks.
    """
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


@pytest.fixture(scope="session")
def built_graph(
    random_problem: RandomProblem, bench_comm: Any, picture: str
) -> MonomialPropagator:
    """Return a propagator whose graph has been built (no coefficients contracted).

    Session-scoped per picture so the graph is built **once** and shared across
    the graph-based benchmarks (``pare``, ``energy``, ``gradient``) instead of
    rebuilt for each. This is safe because those operations only read the graph:
    constructing a (pared) functional and evaluating it do not mutate it -- the
    ``pare`` benchmark already re-runs the construction many rounds on one
    instance with stable results.

    The build happens in fixture setup, which ``pytest-memray`` does not track
    (it hooks the call phase only), so the build cost stays out of each
    operation's memory profile while the resident graph still counts toward its
    peak -- the numbers are the same as a per-test build, only built once.
    """
    mp = build_random_propagator(
        random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
    )
    mp.propagate()
    _record_operator_size(picture, mp, bench_comm)
    return mp
