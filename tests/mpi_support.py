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

"""Whether this test run can exercise monoprop's MPI backend, and helpers for tests that launch fresh processes."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

import monoprop

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# An installation without MPI never initializes MPI, even where mpi4py happens to be installed:
# importing mpi4py.MPI would initialize it as a side effect of collecting the suite.
MPI = None
if monoprop.MPI_ABIS:
    with contextlib.suppress(ImportError):
        from mpi4py import MPI

HAS_MPI_BACKEND: bool = MPI is not None and bool(monoprop.MPI_ABIS)
"""True when mpi4py is importable and this monoprop installation ships an MPI backend."""

# Environment a launcher gives its ranks: a child that inherits it would try to join the parent's job.
LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PRTE_", "PMI_", "HYDRA_", "MPIR_")

# For a child launched with mpiexec: Open MPI may run as root, binding is left to OpenMP so a rank's threads are not
# squeezed onto one core, and oversubscription lets 3- and 4-rank runs fit runners with fewer cores (Open MPI 4 and 5
# spellings, ignored by MPICH).
MPI_LAUNCH_ENV = {
    "OMPI_ALLOW_RUN_AS_ROOT": "1",
    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    "OMPI_MCA_hwloc_base_binding_policy": "none",
    "OMPI_MCA_rmaps_base_oversubscribe": "1",
    "PRTE_MCA_rmaps_default_mapping_policy": ":oversubscribe",
}


def inside_multirank_job() -> bool:
    """Return whether this process is one rank of a multi-rank MPI job."""
    size = os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE") or "1"
    return int(size) > 1


def mpi_launcher() -> str | None:
    """Return the MPI launcher on ``PATH``, if any."""
    return shutil.which("mpiexec") or shutil.which("mpirun")


skip_in_multirank_job = pytest.mark.skipif(
    inside_multirank_job(),
    reason="launches fresh processes, which cannot run inside a multi-rank MPI job",
)

needs_mpi_launch = pytest.mark.skipif(
    not HAS_MPI_BACKEND or mpi_launcher() is None,
    reason="needs an MPI build, mpi4py and an MPI launcher on PATH",
)


def close(a: float, b: float) -> bool:
    """Return whether ``a`` and ``b`` agree within ``1e-9 + 1e-7 * max(|a|, |b|)``."""
    return abs(a - b) <= 1e-9 + 1e-7 * max(abs(a), abs(b))


def inherited_env(*, drop: tuple[str, ...] = ()) -> dict[str, str]:
    """Return this environment without launcher variables or variables starting with any of ``drop``."""
    prefixes = (*LAUNCHER_PREFIXES, *drop)
    return {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}


def child_env(threads: int, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return this environment without launcher variables, at an exact team of ``threads``, updated by ``extra``."""
    return {
        **inherited_env(),
        "monoprop_NUM_THREADS": str(threads),
        "OMP_NUM_THREADS": str(threads),
        "OMP_DYNAMIC": "FALSE",
        **(extra or {}),
    }


def run_probe(
    probe: Path,
    *args: str,
    threads: int,
    extra_env: Mapping[str, str] | None = None,
    timeout: float,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Run ``probe`` with ``args`` in a fresh interpreter at ``child_env(threads, extra_env)``.

    Returns:
        The finished process and the last JSON document it printed, empty when it printed none.
    """
    result = subprocess.run(  # noqa: S603 - trusted: this interpreter and a test probe
        [sys.executable, str(probe), *args],
        env=child_env(threads, extra_env),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return result, (json.loads(lines[-1]) if lines else {})
