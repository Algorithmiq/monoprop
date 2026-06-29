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

import contextlib
import ctypes
import ctypes.util
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from _builders import RandomProblem, build_random_propagator, make_random_problem

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


def _reduce_sum(comm: Any, value: int) -> tuple[int, int]:
    """Sum ``value`` across ranks and return ``(total, rank)``.

    Collective: every rank must call it when multi-rank. A serial run (no
    communicator or size 1) returns ``(value, 0)`` without communicating.
    """
    if comm is not None and comm.Get_size() > 1:
        return comm.allreduce(value, op=MPI.SUM), comm.Get_rank()
    return value, 0


def _record_path(prefix: str) -> Path | None:
    """Return ``results/<prefix>-<label>.json``, or ``None`` if recording is off.

    Recording is enabled only when ``run.py`` exported ``MONOPROP_BENCH_LABEL``
    and ``MONOPROP_BENCH_RESULTS``; otherwise the suite was run directly and has
    nowhere to write.
    """
    label = os.environ.get("MONOPROP_BENCH_LABEL")
    results = os.environ.get("MONOPROP_BENCH_RESULTS")
    if not label or not results:
        return None
    return Path(results, f"{prefix}-{label}.json")


def _merge_record(prefix: str, key: str, value: Any) -> None:
    """Merge ``{key: value}`` into this label's ``prefix`` artifact (rank-0 callers)."""
    path = _record_path(prefix)
    if path is None:
        return
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2))


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
        "--hubbard-lower-atol",
        type=float,
        default=None,
        help="Override lower_atol for the hubbard static benchmark "
        "(default: HubbardConfig's 1e-4).",
    )
    group.addoption(
        "--pauli-lower-atol",
        type=float,
        default=None,
        help="Override lower_atol for the pauli static benchmark "
        "(default: KickedIsingConfig's 1e-4).",
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
    """Record this label's resolved hyperparameters (defaults included) for the report.

    Called on rank 0 only; a no-op when recording is off (see :func:`_record_path`).
    """
    path = _record_path("params")
    if path is None:
        return
    params = {
        opt: config.getoption(f"--{opt.replace('_', '-')}") for opt in _RECORDED_OPTIONS
    }
    path.write_text(json.dumps(params, indent=2))


def _record_operator_size(picture: str, mp: MonomialPropagator, comm: Any) -> None:
    """Record the evolved operator's total term count for one picture.

    Under MPI the operator is partitioned across ranks, so ``mp.size()`` is only
    this rank's shard; the global total is their sum (allreduce). Merged into
    ``opsize-<label>.json`` (one entry per picture) by rank 0.
    """
    total, rank = _reduce_sum(comm, mp.size())
    if rank == 0:
        _merge_record("opsize", picture, {"terms": total})


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


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Make rank 0 the sole writer and printer under MPI; record hyperparameters.

    Every rank runs the whole session, so without intervention all ranks interleave
    their output and race on the shared ``--benchmark-json`` file. Rank 0 (holding
    the makespan timings) prints and writes the JSON; the others go silent and
    disable their JSON output. The memory fixture still runs on every rank (for the
    collective reduce); only rank 0 writes its ``mem-<label>.json``.

    ``trylast`` so the terminal reporter is already registered by the time the
    non-root ranks unregister it.
    """
    if _rank() == 0:
        _write_run_params(config)
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


@pytest.fixture(scope="session")
def bench_comm() -> Any:
    """Return the benchmark communicator (``MPI.COMM_WORLD`` or ``None``)."""
    return None if MPI is None else MPI.COMM_WORLD


@pytest.fixture(scope="session")
def bench_rounds(request: pytest.FixtureRequest) -> int:
    """Return the fixed round count for the random benchmarks."""
    return int(request.config.getoption("--bench-rounds"))


@pytest.fixture(scope="session")
def static_lower_atol(request: pytest.FixtureRequest) -> dict[str, float | None]:
    """Return per-model ``lower_atol`` CLI overrides (``None`` = use config default)."""
    return {
        "hubbard": request.config.getoption("--hubbard-lower-atol"),
        "pauli": request.config.getoption("--pauli-lower-atol"),
    }


@pytest.fixture
def record_model_config() -> Callable[[str, Any], None]:
    """Return ``record(model, config)`` recording a static model's resolved config.

    Merges the config dataclass fields into ``configs-<label>.json`` (one entry per
    model) for the report. The config is identical across ranks, so rank 0 only.
    """

    def _record(model: str, config: Any) -> None:
        if _rank() == 0:
            _merge_record("configs", model, asdict(config))

    return _record


@pytest.fixture(autouse=True)
def record_memory(request: pytest.FixtureRequest, bench_comm: Any) -> Iterator[None]:
    """Record each benchmark's peak physical-memory footprint (PSS) for the report.

    Resets the peak-RSS high-water mark before the test and derives peak PSS (see
    :func:`_peak_pss_bytes`) at teardown -- no allocation tracking, so timing in the
    same sweep is undistorted. It is a footprint: it includes structures already
    resident when the operation starts (e.g. the shared :func:`built_graph`).

    Under MPI the per-rank PSS values are summed via a collective reduce (every rank
    runs every test), giving the job's true physical RAM. Written env-gated and
    rank-0 only into ``mem-<label>.json``.
    """
    _reset_peak_rss()
    yield
    mem, rank = _reduce_sum(bench_comm, _peak_pss_bytes())
    if rank == 0:
        _merge_record("mem", request.node.nodeid.split("/")[-1], mem)


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
    """
    mp = build_random_propagator(
        random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
    )
    mp.propagate()
    _record_operator_size(picture, mp, bench_comm)
    _record_storage_breakdown(picture, mp, bench_comm)
    _record_resting_footprint(picture, bench_comm)
    return mp


def _record_storage_breakdown(picture: str, mp: MonomialPropagator, comm: Any) -> None:
    """Record the built footprint split into operator vs graph storage.

    Uses the C++ structural memory accounting (capacity-based byte totals) rather
    than PSS, so it attributes the footprint to its two structures -- which a
    process-wide PSS reading cannot. Under MPI each value is this rank's shard, so
    they are summed across ranks (allreduce), matching :func:`_record_operator_size`.
    Merged into ``storage-<label>.json`` (one entry per picture) by rank 0.
    """
    sim = mp._simulator  # noqa: SLF001 - the bound C++ object holds the accounting
    operator_total, rank = _reduce_sum(comm, sim.operator_memory_bytes())
    graph_total, _ = _reduce_sum(comm, sim.graph_memory_bytes())
    if rank == 0:
        _merge_record("storage", picture, {"operator": operator_total, "graph": graph_total})


def _record_resting_footprint(picture: str, comm: Any) -> None:
    """Record the built operator+graph's *resting* PSS for one picture.

    Measured here, while the session-scoped graph is resident and its build
    transients have been released, so it captures the persistent footprint the
    per-operation peak metric cannot. Summed across ranks (true physical RAM) and
    merged into ``memrest-<label>.json`` by rank 0, one entry per picture.
    """
    total, rank = _reduce_sum(comm, _resting_pss_bytes())
    if rank == 0:
        _merge_record("memrest", picture, total)
