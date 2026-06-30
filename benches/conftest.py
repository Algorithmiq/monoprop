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
generator length, number of observable terms, number of generators, mode count,
cutoff, and RNG seed can all be varied without editing code, e.g.::

    uv run pytest benches --num-generators 200 --num-modes 64 --cutoff 10

All non-timing results a run produces -- run metadata, resolved hyperparameters,
per-operation peak memory, per-picture operator sizes and footprints, static
configs -- are accumulated in ``_RESULTS`` and written once, at session end, to
``results/<label>.json`` (rank 0 only). Timing is written separately by
pytest-benchmark to ``time-<label>.json``. ``report.py`` merges the two into
``REPORT.md``. Recording is enabled only when the ``just bench`` recipe exports
``MONOPROP_BENCH_LABEL`` / ``MONOPROP_BENCH_RESULTS``.

MPI: the benchmarks are communicator-aware. When ``mpi4py`` is available the
``bench_comm`` fixture yields ``MPI.COMM_WORLD`` (size 1 for a serial run, size
R under ``mpiexec -n R``); otherwise it yields ``None``. Each measured operation
is wrapped in barriers (see :func:`_builders.barriered`) so the timed cost
reflects the slowest rank, and only rank 0 writes results.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import gc
import json
import os
import socket
from dataclasses import asdict, fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from _builders import (
    STATIC_MODELS,
    RandomProblem,
    build_random_propagator,
    make_random_problem,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from monoprop import MonomialPropagator

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


def _reduce_sum(comm: Any, value: int) -> tuple[int, int]:
    """Sum ``value`` across ranks and return ``(total, rank)``.

    Collective: every rank must call it when multi-rank. A serial run (no
    communicator or size 1) returns ``(value, 0)`` without communicating.
    """
    if comm is not None and comm.Get_size() > 1:
        return comm.allreduce(value, op=MPI.SUM), comm.Get_rank()
    return value, 0


# Resolved hyperparameters recorded for the report (the ``--`` CLI names map to
# these plain keys). Kept here as the single source of truth; report.py displays
# them in this order.
RECORDED_OPTIONS = (
    "gen_length",
    "obs_terms",
    "num_generators",
    "num_modes",
    "cutoff",
    "seed",
    "bench_rounds",
)

# All non-timing results for this run, written once at session end (rank 0) to
# ``results/<label>.json``. The fixtures and hooks below fill the sections.
_RESULTS: dict[str, Any] = {
    "meta": {},  # run configuration (ranks, threads, host, ...)
    "params": {},  # resolved random-problem hyperparameters
    "mem": {},  # node id -> peak PSS bytes (per operation)
    "opsize": {},  # picture -> {"terms": n}
    "memrest": {},  # picture -> resting PSS bytes
    "storage": {},  # picture -> {"operator": bytes, "graph": bytes}
    "configs": {},  # static model -> config dataclass fields
}


def _record(section: str, key: str, value: Any) -> None:
    """Store one value into the in-memory results (rank-0 only)."""
    if _rank() == 0:
        _RESULTS[section][key] = value


def _results_path() -> Path | None:
    """Return ``results/<label>.json``, or ``None`` when recording is off.

    Recording is enabled only when the ``just bench`` recipe exported
    ``MONOPROP_BENCH_LABEL`` and ``MONOPROP_BENCH_RESULTS``; run directly, the
    suite has nowhere to write and the results stay in memory only.
    """
    label = os.environ.get("MONOPROP_BENCH_LABEL")
    results = os.environ.get("MONOPROP_BENCH_RESULTS")
    if not label or not results:
        return None
    return Path(results, f"{label}.json")


def _monoprop_version() -> str:
    """Return the installed monoprop version, or 'unknown'."""
    try:
        return version("monoprop")
    except PackageNotFoundError:
        return "unknown"


def _reset_peak_rss() -> None:
    """Reset this process's peak-RSS high-water mark to its current RSS.

    Linux resets ``VmHWM`` when ``5`` (``CLEAR_REFS_MM_HIWATER_RSS``, kernel
    >= 4.0) is written to ``/proc/self/clear_refs``. A no-op where ``/proc`` is
    unavailable (the benchmark environment is Linux).
    """
    with contextlib.suppress(OSError):  # non-Linux or restricted /proc
        Path("/proc/self/clear_refs").write_text("5\n")


def _proc_field(path: str, key: str) -> int:
    """Return a ``/proc/self`` size field (kB → bytes); 0 if unavailable."""
    try:
        text = Path(path).read_text()
    except OSError:  # pragma: no cover - non-Linux or restricted /proc
        return 0
    for line in text.splitlines():
        if line.startswith(key):
            return int(line.split()[1]) * 1024  # values are in kB
    return 0


def _peak_pss_bytes() -> int:
    """Return this operation's peak proportional set size (PSS) in bytes.

    PSS (shared pages split across their sharers) is the honest per-process share
    of physical RAM, so summing it across MPI ranks gives the job's true footprint
    -- unlike RSS, which counts shared library/code pages at full size in every rank.

    PSS has no kernel high-water mark, so derive the peak from ``VmHWM`` (peak RSS,
    reset per test) minus RSS's shared-page over-count (``VmRSS - Pss``). That
    over-count is near-constant over the process lifetime, so reading it at
    teardown still recovers ``peak PSS = peak RSS - shared double-count``.
    """
    hwm = _proc_field("/proc/self/status", "VmHWM:")
    rss = _proc_field("/proc/self/status", "VmRSS:")
    pss = _proc_field("/proc/self/smaps_rollup", "Pss:")
    overcount = max(rss - pss, 0)  # shared pages counted >1x in RSS but not PSS
    return max(hwm - overcount, 0)


def _malloc_trim() -> None:
    """Return free heap pages held by the C allocator to the OS (glibc only).

    ``malloc_trim`` is what makes a *resting* PSS reading meaningful: glibc keeps
    freed pages in its per-arena heaps, so without trimming the resident footprint
    still includes transient build buffers that are logically gone. A no-op (and
    silently ignored) on non-glibc libc.
    """
    with contextlib.suppress(Exception):  # non-glibc libc or no malloc_trim symbol
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.malloc_trim(ctypes.c_size_t(0))


def _resting_pss_bytes() -> int:
    """Return current PSS after collecting garbage and trimming the C heap.

    Unlike :func:`_peak_pss_bytes` (a high-water mark reached mid-operation, e.g.
    while transient build buffers are live), this is the *settled* footprint once
    those transients are released -- the metric that reveals persistent-memory
    wins (a smaller index, recomputed-vs-stored data) that peak RSS cannot see.
    """
    gc.collect()
    _malloc_trim()
    return _proc_field("/proc/self/smaps_rollup", "Pss:")


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
    # Static-model overrides: one option per dataclass field, defaulting to that
    # field's own default, so the config classes stay the single source of truth.
    static = parser.getgroup("monoprop-static", "monoprop static-model overrides")
    for model, (config_cls, _builder, _steps) in STATIC_MODELS.items():
        for field in fields(config_cls):
            static.addoption(
                f"--{model}-{field.name.replace('_', '-')}",
                type=type(field.default),
                default=field.default,
                help=f"{config_cls.__name__}.{field.name} (default: {field.default}).",
            )


def _meta() -> dict[str, Any]:
    """Return this run's configuration metadata for the report."""
    return {
        "label": os.environ.get("MONOPROP_BENCH_LABEL", "?"),
        "ranks": _size(),
        # monoprop's thread knob is genuinely lower-case (read at import).
        "monoprop_threads": os.environ.get("monoprop_NUM_THREADS", "default"),  # noqa: SIM112
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "monoprop_version": _monoprop_version(),
    }


def _params(config: pytest.Config) -> dict[str, Any]:
    """Return the resolved random-problem hyperparameters (defaults included)."""
    return {
        opt: config.getoption(f"--{opt.replace('_', '-')}") for opt in RECORDED_OPTIONS
    }


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Record run metadata on rank 0; silence the other ranks under MPI.

    Every rank runs the whole session, so without intervention all ranks interleave
    their output and race on the shared ``--benchmark-json`` file. Rank 0 (holding
    the makespan timings) prints, writes the JSON, and records the run's metadata;
    the others go silent and disable their JSON output. The memory fixture still
    runs on every rank (for the collective reduce); only rank 0 writes results.

    ``trylast`` so the terminal reporter is already registered by the time the
    non-root ranks unregister it.
    """
    if _rank() == 0:
        _RESULTS["meta"] = _meta()
        _RESULTS["params"] = _params(config)
        return

    # Non-root rank: drop the terminal reporter so it does not interleave with
    # rank 0's output.
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        config.pluginmanager.unregister(reporter)
    # pytest-benchmark opens --benchmark-json at parse time and writes it at session
    # finish. Null the session handle to skip the write, then close the file so it
    # does not leak -- closing alone leaves the session's ``with self.json`` raising
    # on a closed file.
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
def static_configs(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Return each static model's config, with every field resolved from the CLI.

    Each ``--<model>-<field>`` option defaults to the dataclass field's own
    default (see :func:`pytest_addoption`), so reconstructing the config from
    those options reproduces the dataclass default unless the user overrode a
    field.
    """
    opt = request.config.getoption
    return {
        model: config_cls(
            **{
                field.name: opt(f"--{model}-{field.name.replace('_', '-')}")
                for field in fields(config_cls)
            }
        )
        for model, (config_cls, _builder, _steps) in STATIC_MODELS.items()
    }


@pytest.fixture
def record_model_config() -> Callable[[str, Any], None]:
    """Return ``record(model, config)`` recording a static model's resolved config.

    The config is identical across ranks, so :func:`_record` keeps only rank 0's.
    """

    def _do(model: str, config: Any) -> None:
        _record("configs", model, asdict(config))

    return _do


@pytest.fixture(autouse=True)
def record_memory(request: pytest.FixtureRequest, bench_comm: Any) -> Iterator[None]:
    """Record each benchmark's peak physical-memory footprint (PSS) for the report.

    Resets the peak-RSS high-water mark before the test and derives peak PSS (see
    :func:`_peak_pss_bytes`) at teardown -- no allocation tracking, so timing in the
    same sweep is undistorted. It is a footprint: it includes structures already
    resident when the operation starts (e.g. the shared :func:`built_graph`).

    Under MPI the per-rank PSS values are summed via a collective reduce (every rank
    runs every test), giving the job's true physical RAM. Kept rank-0 only.
    """
    _reset_peak_rss()
    yield
    mem, _ = _reduce_sum(bench_comm, _peak_pss_bytes())
    _record("mem", request.node.nodeid.split("/")[-1], mem)


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

    Session-scoped per picture so the graph is built once and shared across the
    graph-based benchmarks (``pare``, ``energy``, ``gradient``) -- safe because
    those operations only read it. The build runs in fixture setup, before each
    test resets the peak-RSS counter, so its transient cost stays out of each
    operation's memory profile while the resident graph still counts toward its peak.

    Records the operator size, operator-vs-graph storage breakdown, and the
    settled resting footprint for this picture while the graph is resident.
    """
    mp = build_random_propagator(
        random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
    )
    mp.propagate()

    # Operator size: under MPI the operator is partitioned, so sum the shards.
    total, _ = _reduce_sum(bench_comm, mp.size())
    _record("opsize", picture, {"terms": total})

    # Storage breakdown from the C++ structural accounting (capacity-based byte
    # totals), which a process-wide PSS reading cannot attribute to a structure.
    sim = mp._simulator
    operator_total, _ = _reduce_sum(bench_comm, sim.operator_memory_bytes())
    graph_total, _ = _reduce_sum(bench_comm, sim.graph_memory_bytes())
    _record("storage", picture, {"operator": operator_total, "graph": graph_total})

    # Resting footprint: settled PSS once the build's transients are released, the
    # persistent-memory metric the per-operation peak cannot see. Summed across ranks.
    resting, _ = _reduce_sum(bench_comm, _resting_pss_bytes())
    _record("memrest", picture, resting)

    return mp
