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
``time-<label>.json``. This module is the writer of that artifact schema; the
readers are :mod:`monoprop_bench_tools.report` and :mod:`monoprop_bench_tools.bmf`,
so a change to ``_RESULTS`` has to land in both places. Recording is on only when
the ``just bench`` recipe exports ``monoprop_BENCH_LABEL`` / ``monoprop_BENCH_RESULTS``.

The ``bench_comm`` fixture yields ``MPI.COMM_WORLD`` when ``mpi4py`` is available
(``None`` otherwise). Operations are barrier-wrapped so the timed cost reflects
the slowest rank, and only rank 0 writes results.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import socket
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
import pytest
from monoprop_bench_tools.memory.cpu import (
    HighWaterMark,
    pinned_thread_summary,
    resting_rss_bytes,
)
from monoprop_bench_tools.models import (
    MODELS,
    RandomProblem,
    build_random_propagator,
    make_random_problem,
)

import monoprop

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from monoprop_bench_tools.models import Built

    from monoprop import MajoranaPropagator

try:
    from mpi4py import MPI
except (ImportError, OSError, RuntimeError):  # pragma: no cover - optional MPI build
    # mpi4py may be absent, or (the ABI wheel) present but unable to dlopen libmpi
    # on a serial node with no MPI module loaded.
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


def _reduce_max(comm: Any, value: int) -> int:
    """Return the largest ``value`` over ranks. Collective; serial returns ``value``."""
    if comm is not None and comm.Get_size() > 1:
        return comm.allreduce(value, op=MPI.MAX)
    return value


def _reduce_min(comm: Any, value: int) -> int:
    """Return the smallest ``value`` over ranks. Collective; serial returns ``value``."""
    if comm is not None and comm.Get_size() > 1:
        return comm.allreduce(value, op=MPI.MIN)
    return value


def _gather_lists(comm: Any, values: list[int]) -> list[list[int]]:
    """Gather per-rank CPU-id lists to rank 0. Collective; off root returns ``[]``."""
    if comm is None or comm.Get_size() == 1:
        return [values]
    gathered = comm.gather(values, root=0)
    return gathered if gathered is not None else []


def _spread(comm: Any, value: int) -> dict[str, int]:
    """Reduce a per-rank number to ``sum`` (bounds the job) and ``max`` (bounds a node)."""
    return {"sum": _reduce_sum(comm, value), "max": _reduce_max(comm, value)}


_RANDOM_OPTIONS = (
    ("gen-length", 4, "Majorana operators per generator."),
    ("obs-terms", 10000, "Observable terms."),
    ("num-generators", 100, "Random generators (circuit gates)."),
    ("num-modes", 128, "Fermionic modes."),
    ("cutoff", 6, "Truncation cutoff."),
    ("seed", 0, "Random seed."),
    ("bench-rounds", 1, "Fixed timing rounds (MPI-safe)."),
)

_RESULTS: dict[str, Any] = {
    "meta": {},  # run configuration (ranks, threads, host, ...)
    "params": {},  # resolved random-problem hyperparameters
    "memhwm": {},  # node id -> summed peak RSS, whole test, setup() included
    "opsize": {},  # picture / model / node id -> {"terms": n}
    "memrest": {},  # picture / model -> resting RSS bytes
    "membase": {},  # fixed model -> resting RSS bytes before the model is built
    "configs": {},  # fixed model -> config dataclass fields
    "opmem": {},  # fixed model -> per-field operator memory split (bytes)
    # Timed call only (see ``OpMemory``), each {"sum", "max"}.
    "opmemdelta": {},  # node id -> peak above the operation's floor
    "opmempeak": {},  # node id -> peak including resident bytes
    "opmembase": {},  # node id -> that floor
    "opbytes": {},  # node id -> {"operator": n, "graph": n}
    "opmembreak": {},  # node id -> operator memory split (bytes)
}


def _record(section: str, key: str, value: Any) -> None:
    """Store one value into the in-memory results (rank-0 only)."""
    if _rank() == 0:
        _RESULTS[section][key] = value


def _results_path() -> Path | None:
    """Return ``results/<label>.json``, or ``None`` when recording is off."""
    label = os.environ.get("monoprop_BENCH_LABEL")  # noqa: SIM112
    results = os.environ.get("monoprop_BENCH_RESULTS")  # noqa: SIM112
    if not label or not results:
        return None
    return Path(results, f"{label}.json")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register benchmark configuration options for the random benchmarks."""
    group = parser.getgroup("monoprop-bench", "monoprop random benchmark sizing")
    for name, default, help_text in _RANDOM_OPTIONS:
        group.addoption(f"--{name}", type=int, default=default, help=help_text)

    models = parser.getgroup("monoprop-models", "monoprop fixed-model overrides")
    for model, (config_cls, _builder, _steps) in MODELS.items():
        for field in fields(config_cls):
            models.addoption(
                f"--{model}-{field.name.replace('_', '-')}",
                type=type(field.default),
                default=field.default,
                help=f"{config_cls.__name__}.{field.name} (default: {field.default}).",
            )


def _core_md5() -> str:
    """Return the md5 of the extension module this run imported.

    The install-time version stamp is not rewritten by a later ``cmake --build`` and
    site-packages holds its own copies, so hash what ``_core`` actually resolved to.
    """
    try:
        path = Path(monoprop._core.__file__)
        return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324
    except (AttributeError, OSError, TypeError):
        return "unavailable"


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
        "monoprop_core_md5": _core_md5(),
        "monoprop_variant": monoprop.__variant__,
        "monoprop_compiler_flags": monoprop.__compiler_flags__,
        "monoprop_max_num_modes": monoprop.MAX_NUM_MODES,
        "malloc_arena_max": os.environ.get("MALLOC_ARENA_MAX", "default"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "default"),
        # Filled by _record_placement: the threads exist only once a propagator does.
        "pinning": {},
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

    Every rank runs the whole session, so rank 0 alone prints, writes and records.
    ``trylast`` so the terminal reporter exists before non-root ranks unregister it.
    """
    if _rank() == 0:
        _RESULTS["meta"] = _meta()
        _RESULTS["params"] = _params(config)
        return

    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        config.pluginmanager.unregister(reporter)
    # Nulling the session handle skips the write; closing alone leaves ``with self.json``
    # raising, so do both.
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


def _record_placement(comm: Any) -> None:
    """Record engine thread placement over ranks; every rank must enter the reductions and gather.

    min/max because a partial failure -- some ranks placed, some not -- is the shape a cpuset
    confinement takes and is invisible in rank 0's view alone.
    """
    summary = pinned_thread_summary()
    placed = summary["single_cpu_threads"]
    mask = summary["affinity_cpus"]
    spread = {
        "single_cpu_threads_min": _reduce_min(comm, placed),
        "single_cpu_threads_max": _reduce_max(comm, placed),
        "affinity_cpus_min": _reduce_min(comm, mask),
        "affinity_cpus_max": _reduce_max(comm, mask),
    }
    pinned_cpus_by_rank = _gather_lists(comm, summary["pinned_cpus"])
    if _rank() == 0:
        _RESULTS["meta"]["pinning"] = {
            **summary,
            **spread,
            "pinned_cpus_by_rank": pinned_cpus_by_rank,
        }


@pytest.fixture
def record_model_config() -> Callable[[str, Any], None]:
    """Return ``record(model, config)`` recording a model's resolved config."""

    def _do(model: str, config: Any) -> None:
        _record("configs", model, asdict(config))

    return _do


def _record_model_stats(
    comm: Any, key: str, propagator: Any, baseline_rss: int | None = None
) -> None:
    """Record term count, operator memory breakdown and footprint under ``key``."""
    _record("opsize", key, {"terms": _reduce_sum(comm, propagator.size())})

    # Placement is only observable while the propagator's threads are alive.
    _record_placement(comm)

    # The Python front-end does not re-export the C++ accounting.
    breakdown = getattr(propagator._simulator, "operator_memory_breakdown", None)
    if breakdown is not None:
        _record(
            "opmem",
            key,
            {k: _reduce_sum(comm, v) for k, v in breakdown().items()},
        )

    resting = _reduce_sum(comm, resting_rss_bytes())
    if resting:  # 0 => /proc unavailable; skip rather than record 0 MiB
        _record("memrest", key, resting)

    if baseline_rss is not None:
        baseline = _reduce_sum(comm, baseline_rss)
        if baseline:
            _record("membase", key, baseline)


@pytest.fixture
def record_model_stats(bench_comm: Any) -> Callable[..., None]:
    """Return ``record(model, propagator, baseline_rss)`` for fixed-model runs."""

    def _do(model: str, propagator: Any, baseline_rss: int) -> None:
        _record_model_stats(bench_comm, model, propagator, baseline_rss)

    return _do


class OpMemory:
    """Records one benchmarked operation's memory over the timed call alone.

    The window opens in ``setup`` and closes when ``pedantic`` returns, so ``delta`` is the
    operation's own cost, not the construction transient. ``settle=False`` deliberately:
    settling is seconds of ``gc.collect()``/``malloc_trim`` beside a timed region.
    """

    def __init__(self, key: str, comm: Any) -> None:
        """Bind the recorder to one benchmark's report key and communicator."""
        self._key = key
        self._comm = comm
        self._window: HighWaterMark | None = None

    def open(self) -> None:
        """Start the window. Safe to call once per round; the last round wins."""
        self._window = HighWaterMark(settle=False)
        self._window.start()

    def close(self, propagator: Any = None) -> None:
        """Record the window and ``propagator``'s byte counts; collective, so every rank calls it equally often."""
        if self._window is None:
            return
        self._window.stop()
        window, self._window = self._window, None

        _record("opmemdelta", self._key, _spread(self._comm, window.delta_bytes))
        _record("opmempeak", self._key, _spread(self._comm, window.peak_bytes))
        _record("opmembase", self._key, _spread(self._comm, window.baseline_bytes))

        _record_placement(self._comm)

        if propagator is None:
            return
        simulator = getattr(propagator, "_simulator", None)
        counts = {
            name: getattr(simulator, f"{name}_memory_bytes", None)
            for name in ("operator", "graph")
        }
        if all(fn is not None for fn in counts.values()):
            _record(
                "opbytes",
                self._key,
                {
                    name: _reduce_sum(self._comm, int(fn()))
                    for name, fn in counts.items()
                },
            )
        breakdown = getattr(simulator, "operator_memory_breakdown", None)
        if breakdown is not None:
            _record(
                "opmembreak",
                self._key,
                {k: _reduce_sum(self._comm, v) for k, v in breakdown().items()},
            )


@pytest.fixture
def op_memory(request: pytest.FixtureRequest, bench_comm: Any) -> Iterator[OpMemory]:
    """Return this test's :class:`OpMemory` recorder, keyed as the timing report keys it."""
    recorder = OpMemory(request.node.nodeid.split("/")[-1], bench_comm)
    yield recorder
    # A test that raised before closing would desync the collectives.
    recorder.close()


@pytest.fixture
def record_opsize(
    request: pytest.FixtureRequest, bench_comm: Any
) -> Callable[[Any], int]:
    """Return ``record(propagator)`` storing and returning the global term count."""
    key = request.node.nodeid.split("/")[-1]

    def _do(propagator: Any) -> int:
        terms = _reduce_sum(bench_comm, propagator.size())
        _record("opsize", key, {"terms": terms})
        return terms

    return _do


@pytest.fixture(autouse=True)
def record_memory(request: pytest.FixtureRequest, bench_comm: Any) -> Iterator[None]:
    """Record ``memhwm``: peak RSS over the whole test, summed over ranks.

    It spans ``setup``, so it predicts an OOM kill but is the wrong number for comparing
    operations -- ``opmemdelta`` is that. The reduce is collective; only rank 0 records.
    """
    with HighWaterMark() as window:
        yield
    key = request.node.nodeid.split("/")[-1]
    hwm = _reduce_sum(bench_comm, window.peak_bytes)
    if hwm:
        _record("memhwm", key, hwm)


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
    problem = make_random_problem(
        gen_length=opt("--gen-length"),
        obs_terms=opt("--obs-terms"),
        num_generators=opt("--num-generators"),
        num_modes=opt("--num-modes"),
        cutoff=opt("--cutoff"),
        seed=opt("--seed"),
    )
    # Keeps every later HighWaterMark's gc.collect() off these session-lifetime tuples.
    gc.freeze()
    return problem


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

    Also records the operator size and resting footprint for this picture while the
    graph is resident.
    """
    mp, circuit = build_random_propagator(
        random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
    )
    mp.build_graph(circuit)

    # Under MPI the operator is partitioned, so sum the partitions.
    _record("opsize", picture, {"terms": _reduce_sum(bench_comm, mp.size())})

    # Settled RSS once the build's transients are released -- the persistent
    # footprint the per-operation peak cannot see.
    resting = _reduce_sum(bench_comm, resting_rss_bytes())
    if resting:  # 0 => /proc unavailable; skip rather than record 0 MiB
        _record("memrest", picture, resting)

    return mp


@pytest.fixture(scope="session")
def model_graph(
    model_configs: dict[str, Any], bench_comm: Any
) -> Callable[[str], tuple[Any, list[float]]]:
    """Return ``get(model)`` giving a fixed model's built graph and its parameter vector.

    Hubbard's circuit is one Trotter step re-applied, so the graph takes ``steps`` successive
    ``build_graph`` calls and the parameter vector is the circuit's repeated to match.
    """
    cache: dict[str, tuple[Any, list[float]]] = {}

    def _get(model: str) -> tuple[Any, list[float]]:
        if model not in cache:
            _config_cls, build_fn, steps_fn = MODELS[model]
            config = model_configs[model]
            steps = steps_fn(config)
            propagator, circuit = build_fn(config, comm=bench_comm)
            for _ in range(steps):
                propagator.build_graph(circuit)
            parameters = list(circuit.parameters) * steps
            # A mismatch benchmarks a different graph than the build_graph cell measured.
            assert len(parameters) == propagator.n_parameters

            # No test here calls record_model_config, so record the config from this side.
            _record("configs", model, asdict(config))
            _record_model_stats(bench_comm, model, propagator)
            gc.freeze()  # same reason as random_problem
            cache[model] = (propagator, parameters)
        return cache[model]

    return _get
