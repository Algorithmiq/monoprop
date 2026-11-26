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

"""Runtime-shape and build-mode preflight for the benchmark suite.

A measurement declares how its process is laid out: ``openmp``, the sharded OpenMP root, whose
shard count is ``monoprop_NUM_THREADS`` or, when that is unset, the OpenMP runtime default. The
preflight checks that declaration against the launch environment, the observed rank count and the
build mode of the extension actually imported, so that neither an environment setting nor the
absence of ``mpiexec`` can stand in for evidence.

This is measurement policy, not library configuration: the library parses its own settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import ModuleType

#: The process layouts a measurement may declare.
RUNTIME_SHAPES = ("openmp",)

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


def _openmp_threads(env: Mapping[str, str]) -> tuple[int, str]:
    """Return the one-store runtime's declared per-rank team budget."""
    configured = env.get("monoprop_NUM_THREADS")
    if configured is not None:
        return _positive_decimal(
            "monoprop_NUM_THREADS", configured
        ), "monoprop_NUM_THREADS"
    fallback = env.get("OMP_NUM_THREADS", "")
    try:
        return _positive_decimal("OMP_NUM_THREADS", fallback), "OMP_NUM_THREADS"
    except PreflightError as exc:
        msg = (
            "Runtime shape 'openmp' with monoprop_NUM_THREADS unset cannot establish the "
            f"declared team size from OMP_NUM_THREADS={fallback!r}; set either to one "
            "positive integer."
        )
        raise PreflightError(msg) from exc


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
        runtime_shape: ``"openmp"``.
        env: The launch environment of this process.
        ranks: The observed number of ranks in the measurement communicator.
        has_mpi: The build mode of the installation actually imported (:func:`build_has_mpi`);
            ``None`` when it could not be read.
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

    try:
        threads, source = _openmp_threads(env)
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


def build_has_mpi(monoprop: ModuleType) -> bool | None:
    """Return whether an imported ``monoprop`` installation ships an MPI backend.

    Current builds report it as a non-empty ``monoprop.MPI_ABIS``; builds from before the MPI ABI
    variants as ``monoprop.has_mpi``.

    Args:
        monoprop: The imported ``monoprop`` package.

    Returns:
        The build mode, or ``None`` when the package reports neither.
    """
    abis = getattr(monoprop, "MPI_ABIS", None)
    if abis is not None:
        return bool(abis)
    legacy = getattr(monoprop, "has_mpi", None)
    return legacy if isinstance(legacy, bool) else None


def measured_extension(monoprop: ModuleType, comm: Any) -> ModuleType:
    """Return the extension module that runs propagators built on ``comm``.

    Current builds serve ``comm=None`` from the serial ``_core`` and a communicator from the MPI
    ABI variant ``_core_mpi``; builds from before the variants run everything on ``_core``.

    Args:
        monoprop: The imported ``monoprop`` package.
        comm: The measurement's mpi4py communicator, or ``None``.

    Returns:
        The module whose file a measurement must hash.
    """
    backend = getattr(monoprop, "_backend", None)
    if backend is None:
        return monoprop._core
    return backend.core_module(comm)


def import_mpi(*, has_mpi: bool) -> Any:
    """Return ``mpi4py.MPI`` for an MPI build, without touching mpi4py otherwise.

    Importing ``mpi4py.MPI`` initializes MPI, so an MPI-off run must never reach it, even when
    the package is installed. An MPI build whose mpi4py is absent or cannot load libmpi runs
    serially, as before.

    Args:
        has_mpi: The build mode of the imported installation (:func:`build_has_mpi`).
    """
    if not has_mpi:
        return None
    try:
        from mpi4py import MPI  # noqa: PLC0415 - deliberately gated on the build mode
    except (ImportError, OSError, RuntimeError):
        return None
    return MPI
