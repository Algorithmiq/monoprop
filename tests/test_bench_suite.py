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
"""Tests for the benchmark suite's own preflight and evidence fields.

They launch ``pytest benches`` at tiny sizes in fresh processes and check the evidence fields the
suite writes and the declarations it refuses, not performance.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import monoprop
from tests.mpi_support import LAUNCHER_PREFIXES, inside_multirank_job

# Repository tooling, not the package: a wheel's test run skips it.
pytestmark = pytest.mark.wheel_redundant

_ROOT = Path(__file__).resolve().parents[1]

# Tiny sizes for every benchmark family; the fixed models are shrunk through their own overrides.
_TINY = [
    "--num-generators=8",
    "--num-modes=8",
    "--cutoff=6",
    "--obs-terms=16",
    "--bench-rounds=1",
    "--hubbard-num-sites=4",
    "--hubbard-observable-site=1",
    "--hubbard-trotter-steps=2",
    "--pauli-num-layers=1",
    "--pauli-cutoff=2",
    "--pauli-lower-atol=0.1",
]
_FAMILIES = ("build_graph", "propagate", "energy", "gradient")

# The imported installation's build mode, as the suite records it in ``has_mpi``.
HAS_MPI = bool(monoprop.MPI_ABIS)

# These tests launch their own processes (pytest, mpiexec). Inside a multi-rank MPI test job every
# rank would repeat those launches nested in the parent job, so they run outside one.
_launches = pytest.mark.skipif(
    inside_multirank_job(),
    reason="launches its own processes; run outside a multi-rank MPI job",
)


def _bench_env(**extra: str) -> dict[str, str]:
    """Return a clean launch environment for a benchmark subprocess."""
    # Launcher variables are dropped so a child never joins an enclosing MPI job.
    dropped = ("monoprop_", "OMP_", "PYTEST_", *LAUNCHER_PREFIXES)
    env = {
        key: value for key, value in os.environ.items() if not key.startswith(dropped)
    }
    env.update(extra)
    return env


def _run_benches(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``pytest benches`` in a fresh process from the repository root."""
    return subprocess.run(  # noqa: S603 - trusted: this interpreter, repository paths
        [sys.executable, "-m", "pytest", "benches", "-p", "no:cacheprovider", *args],
        cwd=_ROOT,
        env=env if env is not None else _bench_env(),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _needs_bench_tools() -> None:
    pytest.importorskip("monoprop_bench_tools")
    pytest.importorskip("pytest_benchmark")


@_launches
def test_suite_records_independent_exactness_for_all_four_families(
    tmp_path: Path,
) -> None:
    _needs_bench_tools()
    results = tmp_path / "results"
    results.mkdir()
    env = _bench_env(
        monoprop_BENCH_LABEL="tiny",
        monoprop_BENCH_RESULTS=str(results),
        monoprop_NUM_THREADS="1",
    )
    proc = _run_benches(
        *_TINY,
        f"--benchmark-json={results / 'time-tiny.json'}",
        env=env,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    data = json.loads((results / "tiny.json").read_text())

    nodes = [
        f"bench_random.py::test_random_{family}[{picture}]"
        for family in _FAMILIES
        for picture in ("heisenberg", "schrodinger")
    ] + [
        f"bench_models.py::test_model_{family}[{model}]"
        for family in _FAMILIES
        for model in ("hubbard", "pauli")
    ]
    for node in nodes:
        for section in ("opmempeak", "opmembase", "opmemdelta"):
            assert set(data[section][node]) == {"sum", "max"}, (section, node)
        assert data["opmemexact"][node] is True, node
        assert data["memhwmexact"][node] is True, node
        assert data["memhwm"][node] >= data["opmempeak"][node]["max"] > 0, node
    assert data["meta"]["has_mpi"] is HAS_MPI


def _mpi4py_trap(tmp_path: Path) -> tuple[Path, Path]:
    """Return a PYTHONPATH entry whose ``mpi4py`` records any import, and that marker file."""
    marker = tmp_path / "mpi4py-imported"
    package = tmp_path / "trap" / "mpi4py"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "raise RuntimeError('mpi4py must not be imported in an MPI-off run')\n"
    )
    return package.parent, marker


@_launches
@pytest.mark.skipif(HAS_MPI, reason="needs an MPI-off extension")
def test_mpi_off_suite_never_imports_mpi4py(tmp_path: Path) -> None:
    _needs_bench_tools()
    trap, marker = _mpi4py_trap(tmp_path)
    env = _bench_env(PYTHONPATH=os.pathsep.join([str(trap), *sys.path]))
    proc = _run_benches("--collect-only", "-q", *_TINY, env=env)

    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert not marker.exists()


@_launches
@pytest.mark.skipif(HAS_MPI, reason="needs an MPI-off extension")
@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (["--runtime-shape=openmp"], {"monoprop_NUM_THREADS": "0"}, "positive"),
        (["--build-mode=mpi"], {}, "has_mpi"),
    ],
)
def test_suite_rejects_contradictory_declarations(
    args: list[str], env: dict[str, str], message: str
) -> None:
    _needs_bench_tools()
    proc = _run_benches("--collect-only", "-q", *args, env=_bench_env(**env))

    assert proc.returncode != 0
    assert message in proc.stdout + proc.stderr


@_launches
@pytest.mark.skipif(HAS_MPI, reason="needs an MPI-off extension")
@pytest.mark.parametrize(
    ("args", "env"),
    [
        (["--runtime-shape=openmp"], {"monoprop_NUM_THREADS": "2"}),
        (["--runtime-shape=openmp"], {"OMP_NUM_THREADS": "2"}),
        (["--build-mode=mpi-off"], {}),
        ([], {}),
    ],
)
def test_suite_accepts_valid_declarations(args: list[str], env: dict[str, str]) -> None:
    _needs_bench_tools()
    proc = _run_benches("--collect-only", "-q", *args, env=_bench_env(**env))

    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]


@_launches
@pytest.mark.skipif(
    not HAS_MPI or shutil.which("mpiexec") is None,
    reason="needs an MPI extension and mpiexec",
)
@pytest.mark.parametrize(
    ("args", "env"),
    [
        (["--runtime-shape=openmp"], {"monoprop_NUM_THREADS": "1"}),
        ([], {}),
    ],
)
def test_multi_rank_shape_preflight(args: list[str], env: dict[str, str]) -> None:
    _needs_bench_tools()
    exported = [flag for key in env for flag in ("-x", key)]
    proc = subprocess.run(  # noqa: S603 - trusted: this interpreter, repository paths
        [
            shutil.which("mpiexec"),
            "--map-by",
            ":OVERSUBSCRIBE",
            "-n",
            "2",
            *exported,
            sys.executable,
            "-m",
            "pytest",
            "benches",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            *args,
        ],
        cwd=_ROOT,
        env=_bench_env(**env),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
