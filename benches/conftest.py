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

The random benchmarks take their sizes from CLI options, e.g.::

    uv run pytest benches --num-generators 200 --num-modes 64 --cutoff 10

Non-timing results accumulate in ``_RESULTS`` and are written at session end to
``results/<label>.json`` (rank 0 only); pytest-benchmark writes timings to
``time-<label>.json``; ``report.py`` merges them. Recording is on only when the
``just bench`` recipe exports ``monoprop_BENCH_LABEL`` / ``monoprop_BENCH_RESULTS``.

The ``bench_comm`` fixture yields ``MPI.COMM_WORLD`` when ``mpi4py`` is available
(``None`` otherwise). Operations are barrier-wrapped so the timed cost reflects
the slowest rank, and only rank 0 writes results.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
import pytest
from _builders import (
    MODELS,
    RandomProblem,
    build_random_propagator,
    make_random_problem,
)
from _memory import PssSampler, merge_peak_of_sum, resting_pss_bytes

import monoprop

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from _builders import Built

    from monoprop import MajoranaPropagator

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - depends on optional MPI build
    MPI = None


def _rank() -> int:
    """Return this process's MPI rank (``0`` without MPI)."""
    return 0 if MPI is None else MPI.COMM_WORLD.Get_rank()


def _size() -> int:
    """Return the number of MPI ranks (``1`` without MPI)."""
    return 1 if MPI is None else MPI.COMM_WORLD.Get_size()


def _reduce_sum(comm: Any, value: int) -> int:
    """Sum ``value`` across ranks. Collective; a serial run returns ``value``."""
    if comm is not None and comm.Get_size() > 1:
        return comm.allreduce(value, op=MPI.SUM)
    return value


# Random-benchmark options as ``(name, default, help)`` (all int). This is the
# single source of truth: the CLI options and the recorded hyperparameters (in
# this display order) are both derived from it.
_RANDOM_OPTIONS = (
    ("gen-length", 4, "Majorana operators per generator."),
    ("obs-terms", 10000, "Observable terms."),
    ("num-generators", 100, "Random generators (circuit gates)."),
    ("num-modes", 128, "Fermionic modes."),
    ("cutoff", 6, "Truncation cutoff."),
    ("seed", 0, "Random seed."),
    ("bench-rounds", 1, "Fixed timing rounds (MPI-safe)."),
)

# Non-timing results, written at session end (rank 0) to ``results/<label>.json``.
_RESULTS: dict[str, Any] = {
    "meta": {},  # run configuration (ranks, threads, host, ...)
    "params": {},  # resolved random-problem hyperparameters
    "mem": {},  # node id -> peak PSS bytes (per operation)
    "opsize": {},  # picture -> {"terms": n}
    "memrest": {},  # picture -> resting PSS bytes
    "storage": {},  # picture -> {"operator": bytes, "graph": bytes}
    "configs": {},  # fixed model -> config dataclass fields
}


def _record(section: str, key: str, value: Any) -> None:
    """Store one value into the in-memory results (rank-0 only)."""
    if _rank() == 0:
        _RESULTS[section][key] = value


def _results_path() -> Path | None:
    """Return ``results/<label>.json``, or ``None`` when recording is off.

    Recording is on only when ``just bench`` exported ``monoprop_BENCH_LABEL``
    and ``monoprop_BENCH_RESULTS``.
    """
    label = os.environ.get("monoprop_BENCH_LABEL")  # noqa: SIM112
    results = os.environ.get("monoprop_BENCH_RESULTS")  # noqa: SIM112
    if not label or not results:
        return None
    return Path(results, f"{label}.json")


def _peak_of_sum(comm: Any, samples: list[tuple[float, int]]) -> int:
    """Reduce per-rank PSS timelines to the job's peak summed PSS (bytes).

    Gathers every rank's ``(wall_clock, pss)`` samples to rank 0 and merges them
    via :func:`_memory.merge_peak_of_sum`. Collective; returns ``0`` off root.
    """
    if comm is None or comm.Get_size() == 1:
        return merge_peak_of_sum([samples])
    gathered = comm.gather(samples, root=0)
    if comm.Get_rank() != 0:
        return 0
    return merge_peak_of_sum(gathered)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register benchmark configuration options for the random benchmarks."""
    group = parser.getgroup("monoprop-bench", "monoprop random benchmark sizing")
    for name, default, help_text in _RANDOM_OPTIONS:
        group.addoption(f"--{name}", type=int, default=default, help=help_text)

    # One override option per model config field, defaulting to the field's own
    # default so the config classes stay the source of truth.
    models = parser.getgroup("monoprop-models", "monoprop fixed-model overrides")
    for model, (config_cls, _builder, _steps) in MODELS.items():
        for field in fields(config_cls):
            models.addoption(
                f"--{model}-{field.name.replace('_', '-')}",
                type=type(field.default),
                default=field.default,
                help=f"{config_cls.__name__}.{field.name} (default: {field.default}).",
            )


def _meta() -> dict[str, Any]:
    """Return this run's configuration metadata for the report."""
    return {
        "label": os.environ.get("monoprop_BENCH_LABEL", "?"),  # noqa: SIM112
        "ranks": _size(),
        "monoprop_threads": os.environ.get("monoprop_NUM_THREADS", "default"),  # noqa: SIM112
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "hostname": socket.gethostname(),
        "monoprop_version": monoprop.__version__,
    }


def _params(config: pytest.Config) -> dict[str, Any]:
    """Return the resolved random-problem hyperparameters (defaults included)."""
    return {
        name.replace("-", "_"): config.getoption(f"--{name}")
        for name, _default, _help in _RANDOM_OPTIONS
    }


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Record run metadata on rank 0; silence the other ranks under MPI.

    Every rank runs the whole session, so without intervention they interleave
    output and race on the shared ``--benchmark-json`` file. Rank 0 prints, writes
    the JSON, and records metadata; the others go silent. The memory fixture still
    runs everywhere (for the collective reduce), but only rank 0 records.

    ``trylast`` so the terminal reporter exists before non-root ranks unregister it.
    """
    if _rank() == 0:
        _RESULTS["meta"] = _meta()
        _RESULTS["params"] = _params(config)
        return

    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        config.pluginmanager.unregister(reporter)
    # pytest-benchmark opens --benchmark-json at parse time and writes it at
    # session finish. Null the session handle to skip the write, then close the
    # file so it does not leak (closing alone leaves ``with self.json`` raising).
    bench_session = getattr(config, "_benchmarksession", None)
    if bench_session is not None:
        bench_session.json = None
    bench_json = getattr(config.option, "benchmark_json", None)
    if bench_json is not None:
        if hasattr(bench_json, "close"):
            bench_json.close()
        config.option.benchmark_json = None


def pytest_sessionfinish() -> None:
    """Write the accumulated results to ``results/<label>.json`` (rank 0 only)."""
    if _rank() != 0:
        return
    path = _results_path()
    if path is not None:
        path.write_text(json.dumps(_RESULTS, indent=2))


@pytest.fixture(scope="session")
def bench_comm() -> Any:
    """Return the benchmark communicator (``MPI.COMM_WORLD`` or ``None``)."""
    return None if MPI is None else MPI.COMM_WORLD


@pytest.fixture(scope="session")
def bench_rounds(request: pytest.FixtureRequest) -> int:
    """Return the fixed round count for the random benchmarks."""
    return int(request.config.getoption("--bench-rounds"))


@pytest.fixture(scope="session")
def model_configs(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Return each fixed model's config, every field resolved from the CLI.

    Each ``--<model>-<field>`` option defaults to the dataclass field's default,
    so an unoverridden config reproduces the dataclass default.
    """
    opt = request.config.getoption
    return {
        model: config_cls(
            **{
                field.name: opt(f"--{model}-{field.name.replace('_', '-')}")
                for field in fields(config_cls)
            }
        )
        for model, (config_cls, _builder, _steps) in MODELS.items()
    }


@pytest.fixture
def record_model_config() -> Callable[[str, Any], None]:
    """Return ``record(model, config)`` recording a model's resolved config."""

    def _do(model: str, config: Any) -> None:
        _record("configs", model, asdict(config))

    return _do


@pytest.fixture(autouse=True)
def record_memory(request: pytest.FixtureRequest, bench_comm: Any) -> Iterator[None]:
    """Record each benchmark's peak physical-memory footprint (PSS) for the report.

    A background :class:`PssSampler` samples this rank's live PSS while the test
    runs. It is a footprint: it includes structures already resident when the
    operation starts (e.g. the shared :func:`built_graph`). Under MPI the per-rank
    timelines are merged into the peak-of-sum (see :func:`_peak_of_sum`); the
    gather is collective, but only rank 0 records.
    """
    with PssSampler() as sampler:
        yield
    mem = _peak_of_sum(bench_comm, sampler.samples)
    if mem:  # 0 => non-root rank or /proc unavailable: nothing to record
        _record("mem", request.node.nodeid.split("/")[-1], mem)


@pytest.fixture(scope="session", params=["heisenberg", "schrodinger"])
def picture(request: pytest.FixtureRequest) -> str:
    """Parametrize the random benchmarks over the physical picture.

    Session-scoped so the shared :func:`built_graph` is built once per picture.
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


@pytest.fixture
def make_random_propagator(
    random_problem: RandomProblem, bench_comm: Any, picture: str
) -> Callable[..., Built]:
    """Return a factory building a fresh ``(propagator, circuit)`` tuple.

    Wraps the picture/communicator wiring so a benchmark ``setup`` can just call
    ``make_random_propagator(lower_atol=...)`` for a fresh build each round.
    """

    def _make(*, lower_atol: float | None = None) -> Built:
        return build_random_propagator(
            random_problem,
            comm=bench_comm,
            lower_atol=lower_atol,
            schrodinger=picture == "schrodinger",
        )

    return _make


@pytest.fixture(scope="session")
def built_graph(
    random_problem: RandomProblem, bench_comm: Any, picture: str
) -> MajoranaPropagator:
    """Return a propagator whose graph has been built (no coefficients contracted).

    Session-scoped per picture so the graph is built once and shared across the
    read-only graph benchmarks (``pare``, ``energy``, ``gradient``).

    Also records the operator size, operator-vs-graph storage breakdown, and
    resting footprint for this picture while the graph is resident.
    """
    mp, circuit = build_random_propagator(
        random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
    )
    mp.propagate_build_graph(circuit)

    # Under MPI the operator is partitioned, so sum the shards.
    _record("opsize", picture, {"terms": _reduce_sum(bench_comm, mp.size())})

    # Structural byte totals from the C++ accounting, which a process-wide PSS
    # reading cannot attribute to a structure.
    sim = mp._simulator
    _record(
        "storage",
        picture,
        {
            "operator": _reduce_sum(bench_comm, sim.operator_memory_bytes()),
            "graph": _reduce_sum(bench_comm, sim.graph_memory_bytes()),
        },
    )

    # Settled PSS once the build's transients are released -- the persistent
    # footprint the per-operation peak cannot see.
    resting = _reduce_sum(bench_comm, resting_pss_bytes())
    if resting:  # 0 => /proc unavailable; skip rather than record 0 MiB
        _record("memrest", picture, resting)

    return mp
