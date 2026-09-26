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

"""Runtime-shape and build-mode preflight shared by the benchmark suite and the parity driver.

A measurement declares how its process is laid out -- ``partitions`` (the legacy runtime, which
reads ``monoprop_PARTITIONS``) or ``openmp`` (one store per rank, sized by
``monoprop_NUM_THREADS`` or, when that is unset, the OpenMP runtime default). The preflight checks
that declaration against the launch environment, the observed rank count and the build mode of
the extension actually imported, so that neither an arm label nor the absence of ``mpiexec`` can
stand in for evidence.

This is measurement policy, not library configuration: the library parses its own settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The two process layouts a measurement may declare.
RUNTIME_SHAPES = ("partitions", "openmp")

#: The runtime shape each comparison arm must declare.
ARM_SHAPES = {"baseline": "partitions", "candidate": "openmp"}

# Launch settings that change what a measurement runs; everything else in the environment is
# irrelevant to the comparison and is kept out of recorded evidence.
_SETTING_PREFIXES = ("monoprop_", "OMP_", "GOMP_", "KMP_", "MALLOC_")
# Output routing only; ``monoprop_BENCH_ALLOW_BIG_GRAPH`` stays because it changes what runs.
_NOT_SETTINGS = ("monoprop_BENCH_LABEL", "monoprop_BENCH_RESULTS")


class PreflightError(ValueError):
    """A declared measurement shape or build mode contradicts the observed process."""


def _positive_decimal(name: str, value: str) -> int:
    """Return ``value`` as a positive decimal integer, rejecting signs, spaces and lists."""
    if not value.isascii() or not value.isdigit() or int(value) < 1:
        msg = f"{name}={value!r} is not a positive decimal integer."
        raise PreflightError(msg)
    return int(value)


def settings_snapshot(env: Mapping[str, str]) -> dict[str, str]:
    """Return the library, OpenMP and allocator settings present in ``env``."""
    return {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(_SETTING_PREFIXES) and key not in _NOT_SETTINGS
    }


def require_arm_shape(arm: str, runtime_shape: str) -> None:
    """Reject an arm label that does not match its runtime shape.

    Raises:
        PreflightError: If ``arm`` is unknown or declares the other arm's shape.
    """
    expected = ARM_SHAPES.get(arm)
    if expected is None:
        msg = f"Unknown arm {arm!r}; expected one of {sorted(ARM_SHAPES)}."
        raise PreflightError(msg)
    if runtime_shape != expected:
        msg = (
            f"The {arm} arm must declare runtime shape {expected!r}, not "
            f"{runtime_shape!r}; relabelling an arm cannot change what it measures."
        )
        raise PreflightError(msg)


def _partition_threads(env: Mapping[str, str]) -> tuple[int, str]:
    """Return the legacy runtime's declared per-rank thread count."""
    partitions = env.get("monoprop_PARTITIONS")
    threads = env.get("monoprop_NUM_THREADS")
    if partitions is None or threads is None:
        msg = (
            "Runtime shape 'partitions' needs both monoprop_PARTITIONS=T and "
            "monoprop_NUM_THREADS=T; the legacy engine otherwise picks its own count."
        )
        raise PreflightError(msg)
    count = _positive_decimal("monoprop_PARTITIONS", partitions)
    if _positive_decimal("monoprop_NUM_THREADS", threads) != count:
        msg = (
            f"Runtime shape 'partitions' needs monoprop_PARTITIONS ({partitions}) equal to "
            f"monoprop_NUM_THREADS ({threads})."
        )
        raise PreflightError(msg)
    return count, "monoprop_PARTITIONS"


def _openmp_threads(env: Mapping[str, str]) -> tuple[int, str]:
    """Return the one-store runtime's declared per-rank team budget."""
    if "monoprop_PARTITIONS" in env:
        msg = (
            "Runtime shape 'openmp' has no partitions: unset monoprop_PARTITIONS rather than "
            "leave it in the measurement environment."
        )
        raise PreflightError(msg)
    configured = env.get("monoprop_NUM_THREADS")
    if configured is not None:
        return _positive_decimal(
            "monoprop_NUM_THREADS", configured
        ), "monoprop_NUM_THREADS"
    fallback = env.get("OMP_NUM_THREADS", "")
    if not fallback.isascii() or not fallback.isdigit() or int(fallback) < 1:
        msg = (
            "Runtime shape 'openmp' with monoprop_NUM_THREADS unset cannot establish the "
            f"declared team size from OMP_NUM_THREADS={fallback!r}; set either to one "
            "positive integer."
        )
        raise PreflightError(msg)
    return int(fallback), "OMP_NUM_THREADS"


def declared_shape(
    runtime_shape: str,
    env: Mapping[str, str],
    *,
    ranks: int,
    has_mpi: bool | None,
    expected_has_mpi: bool | None = None,
    expected_ranks: int | None = None,
    expected_threads: int | None = None,
) -> dict[str, Any]:
    """Validate a declared runtime shape and return it with its evidence.

    Args:
        runtime_shape: ``"partitions"`` or ``"openmp"``.
        env: The launch environment of this process.
        ranks: The observed number of ranks in the measurement communicator.
        has_mpi: ``monoprop.has_mpi`` of the extension actually imported; ``None`` when it could
            not be read.
        expected_has_mpi: The build mode the measurement requires, if declared.
        expected_ranks: The rank count the measurement requires, if declared.
        expected_threads: The per-rank thread count the measurement requires, if declared.

    Returns:
        ``runtime_shape``, ``ranks``, ``threads``, ``threads_source``, ``has_mpi`` and the
        relevant ``settings`` of ``env``.

    Raises:
        PreflightError: On any missing, malformed or contradictory declaration.
    """
    if runtime_shape not in RUNTIME_SHAPES:
        msg = f"Unknown runtime shape {runtime_shape!r}; expected one of {RUNTIME_SHAPES}."
        raise PreflightError(msg)
    if has_mpi is None:
        msg = "The imported extension's has_mpi could not be read; the build mode is unproven."
        raise PreflightError(msg)
    if expected_has_mpi is not None and has_mpi != expected_has_mpi:
        msg = (
            f"The imported extension has has_mpi={has_mpi}, but the measurement requires "
            f"has_mpi={expected_has_mpi}."
        )
        raise PreflightError(msg)
    if ranks < 1:
        msg = f"Observed {ranks} ranks; a measurement needs at least one."
        raise PreflightError(msg)
    if not has_mpi and ranks != 1:
        msg = f"An MPI-off binary cannot run {ranks} ranks; MPI-off cells have R=1."
        raise PreflightError(msg)
    if expected_ranks is not None and ranks != expected_ranks:
        msg = f"Observed {ranks} ranks, but the measurement declares {expected_ranks} ranks."
        raise PreflightError(msg)

    reader = _partition_threads if runtime_shape == "partitions" else _openmp_threads
    try:
        threads, source = reader(env)
    except PreflightError as exc:
        if str(exc).startswith("Runtime shape"):
            raise
        msg = f"Runtime shape '{runtime_shape}': {exc}"
        raise PreflightError(msg) from exc
    if expected_threads is not None and threads != expected_threads:
        msg = (
            f"The environment declares {threads} threads per rank, but the measurement "
            f"requires {expected_threads} threads."
        )
        raise PreflightError(msg)
    return {
        "runtime_shape": runtime_shape,
        "ranks": ranks,
        "threads": threads,
        "threads_source": source,
        "has_mpi": has_mpi,
        "settings": settings_snapshot(env),
    }


def import_mpi(*, has_mpi: bool) -> Any:
    """Return ``mpi4py.MPI`` for an MPI build, without touching mpi4py otherwise.

    Importing ``mpi4py.MPI`` initializes MPI, so an MPI-off run must never reach it, even when
    the package is installed. An MPI build whose mpi4py is absent or cannot load libmpi runs
    serially, as before.

    Args:
        has_mpi: ``monoprop.has_mpi`` of the imported extension.
    """
    if not has_mpi:
        return None
    try:
        from mpi4py import MPI  # noqa: PLC0415 - deliberately gated on the build mode
    except (ImportError, OSError, RuntimeError):
        return None
    return MPI
