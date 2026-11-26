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
``tests/mpi_thread_support_probe.py`` in a fresh interpreter of two threads and reads the requested and the provided
level back from the probe. MPI may provide more support than requested: a run whose provided level does not exercise
the case's route is skipped and says so, never passed. The C++ failure driver (``cpp/tests/mpi_failure_driver.cpp``)
covers the refusal below FUNNELED and two-rank launches.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.mpi_support import (
    HAS_MPI_BACKEND,
    MPI_LAUNCH_ENV,
    close,
    run_probe,
    skip_in_multirank_job,
)

if TYPE_CHECKING:
    import subprocess

_PROBE = Path(__file__).resolve().parent / "mpi_thread_support_probe.py"


pytestmark = [
    # The probe initializes MPI through mpi4py, which an MPI-enabled wheel's serial test environment lacks.
    pytest.mark.skipif(not HAS_MPI_BACKEND, reason="needs an MPI build and mpi4py"),
    skip_in_multirank_job,
]


def _single(
    level: str, mode: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    return run_probe(
        _PROBE, level, mode, threads=2, extra_env=MPI_LAUNCH_ENV, timeout=300
    )


def _require_exact(doc: dict[str, Any], level: str) -> None:
    if doc.get("provided") != level:
        pytest.skip(
            f"not exercised: requested {level} but the MPI library provided {doc.get('provided')}"
        )


def _check_operations(doc: dict[str, Any]) -> None:
    assert "error" not in doc, doc["error"]
    world, reference = doc["world"], doc["self"]
    for key in ("energy", "value", "functional"):
        assert close(world[key], reference["energy"]), (key, world[key], reference)
    for key in ("gradient", "functional_gradient"):
        assert len(world[key]) == len(reference["gradient"])
        assert all(
            close(a, b) for a, b in zip(world[key], reference["gradient"], strict=True)
        ), (key, world[key], reference["gradient"])
    assert any(abs(g) > 0 for g in reference["gradient"])


def test_host_initialized_funneled_level_is_accepted() -> None:
    result, doc = _single("funneled", "operations")
    assert result.returncode == 0, result.stderr[-4000:]
    assert doc["requested"] == "funneled"
    _require_exact(doc, "funneled")
    _check_operations(doc)


@pytest.mark.parametrize("level", ["funneled", "multiple"])
def test_wrong_thread_entry_fails_fast(level: str) -> None:
    # Higher thread support never allows a call from a thread other than the initializing one.
    result, doc = _single(level, "wrong-thread")
    # No report at all means the probe died before the check: a failure, not an unexercised level.
    assert doc, result.stderr[-4000:]
    _require_exact(doc, level)
    assert "error" not in doc, doc
    assert "main_thread_energy" in doc
    assert result.returncode != 0
    assert "thread that initialized MPI" in result.stderr, result.stderr[-4000:]
    assert "completed" not in doc
    assert "aborting the communicator" not in result.stderr
