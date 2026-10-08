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

"""MPI thread support through Python hosts: mpi4py initializes MPI before ``monoprop`` is imported.

The library requests MPI_THREAD_FUNNELED when it initializes MPI itself and requires at least FUNNELED from a host
that did; only the initializing thread (the primary of the library's OpenMP team) ever calls MPI. Each case launches
``tests/mpi_thread_support_probe.py`` in fresh interpreters, directly (one rank) or under the MPI launcher (two ranks of
two threads each, splitmix routing, so every gate exchanges across ranks), and reads the requested and the provided
level back from the probe. MPI may provide more support than requested: a run whose provided level does not exercise
the case's route is skipped and says so, never passed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from monoprop import _core

_PROBE = Path(__file__).resolve().parent / "mpi_thread_support_probe.py"
# Environment a launcher gives its ranks: a child that inherits it would try to join the parent's job.
_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PRTE_", "PMI_", "HYDRA_", "MPIR_")
_TIMEOUT = 300


def _inside_multirank_job() -> bool:
    size = os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE") or "1"
    return int(size) > 1


def _launcher() -> str | None:
    return shutil.which("mpiexec") or shutil.which("mpirun")


pytestmark = [
    pytest.mark.skipif(not _core.has_mpi, reason="needs an MPI build"),
    pytest.mark.skipif(
        _inside_multirank_job(),
        reason="launches fresh interpreters, which cannot run inside a multi-rank job",
    ),
]
needs_launcher = pytest.mark.skipif(
    _launcher() is None, reason="needs an MPI launcher on PATH"
)


def _env(threads: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_PREFIXES)}
    env.update(
        {
            "monoprop_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "OMP_DYNAMIC": "FALSE",
            "monoprop_ROUTING": "splitmix",
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            # Leave binding to OpenMP, so a rank's threads are not squeezed onto one core.
            "OMPI_MCA_hwloc_base_binding_policy": "none",
        }
    )
    return env


def _single(
    level: str, mode: str, threads: int = 2
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    result = subprocess.run(  # noqa: S603 - trusted: this interpreter and a fixed probe
        [sys.executable, str(_PROBE), level, mode],
        env=_env(threads),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return result, (json.loads(lines[-1]) if lines else {})


def _ranks(
    level: str, mode: str, tmp_path: Path, ranks: int = 2, threads: int = 2
) -> tuple[subprocess.CompletedProcess[str], list[dict[str, Any]]]:
    launcher = _launcher()
    assert launcher is not None
    out = tmp_path / uuid.uuid4().hex[:8]
    out.mkdir()
    result = subprocess.run(  # noqa: S603 - trusted: a known launcher, this interpreter and a fixed probe
        [
            launcher,
            "-n",
            str(ranks),
            sys.executable,
            str(_PROBE),
            level,
            mode,
            str(out),
        ],
        env=_env(threads),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    docs = [json.loads(path.read_text()) for path in sorted(out.glob("rank*.json"))]
    return result, docs


def _require_exact(doc: dict[str, Any], level: str) -> None:
    if doc.get("provided") != level:
        pytest.skip(
            f"not exercised: requested {level} but the MPI library provided {doc.get('provided')}"
        )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 + 1e-7 * max(abs(a), abs(b))


def _check_operations(doc: dict[str, Any]) -> None:
    assert "error" not in doc, doc["error"]
    world, reference = doc["world"], doc["self"]
    for key in ("energy", "value", "functional"):
        assert _close(world[key], reference["energy"]), (key, world[key], reference)
    for key in ("gradient", "functional_gradient"):
        assert len(world[key]) == len(reference["gradient"])
        assert all(
            _close(a, b) for a, b in zip(world[key], reference["gradient"], strict=True)
        ), (key, world[key], reference["gradient"])
    assert any(abs(g) > 0 for g in reference["gradient"])


@pytest.mark.parametrize("level", ["funneled", "serialized", "multiple"])
def test_host_initialized_level_is_accepted_on_one_rank(level: str) -> None:
    result, doc = _single(level, "operations")
    assert result.returncode == 0, result.stderr[-4000:]
    assert doc["requested"] == level
    _require_exact(doc, level)
    _check_operations(doc)


@needs_launcher
@pytest.mark.parametrize("level", ["funneled", "serialized", "multiple"])
def test_host_initialized_level_is_accepted_on_two_ranks(
    level: str, tmp_path: Path
) -> None:
    result, docs = _ranks(level, "operations", tmp_path)
    assert result.returncode == 0, result.stderr[-4000:]
    assert [doc["rank"] for doc in docs] == [0, 1]
    for doc in docs:
        _require_exact(doc, level)
        _check_operations(doc)
    # Global results agree on every rank.
    assert docs[0]["world"]["energy"] == docs[1]["world"]["energy"]
    assert docs[0]["world"]["gradient"] == docs[1]["world"]["gradient"]


def test_single_level_is_refused_on_one_rank() -> None:
    result, doc = _single("single", "construct")
    assert result.returncode == 0, result.stderr[-4000:]
    if doc.get("provided") != "single":
        pytest.skip(
            f"not exercised: requested single but the MPI library provided {doc.get('provided')}"
        )
    assert "requires MPI_THREAD_FUNNELED" in doc.get("error", ""), doc
    assert "MPI_THREAD_SINGLE" in doc["error"]
    assert "size" not in doc


@needs_launcher
def test_single_level_aborts_two_ranks_promptly(tmp_path: Path) -> None:
    # A rank that refused would strand its peers in the construction's collectives, so the job is aborted, within
    # the timeout, with the refusal in the diagnostic.
    result, docs = _ranks("single", "construct", tmp_path)
    output = result.stdout + result.stderr
    if docs and all(doc.get("provided") != "single" for doc in docs):
        pytest.skip(
            "not exercised: the MPI library provided more than MPI_THREAD_SINGLE"
        )
    assert result.returncode != 0, output[-4000:]
    assert "requires MPI_THREAD_FUNNELED" in output, output[-4000:]
    assert "aborting the communicator" in output, output[-4000:]


@pytest.mark.parametrize("level", ["funneled", "multiple"])
def test_wrong_thread_entry_fails_fast(level: str) -> None:
    # Higher thread support never allows a call from a thread other than the initializing one.
    result, doc = _single(level, "wrong-thread")
    if doc.get("provided") != level:
        pytest.skip(
            f"not exercised: requested {level} but the MPI library provided {doc.get('provided')}"
        )
    assert "error" not in doc, doc
    assert "main_thread_energy" in doc
    assert result.returncode != 0
    assert "thread that initialized MPI" in result.stderr, result.stderr[-4000:]
    assert "completed" not in doc
    assert "aborting the communicator" not in result.stderr
