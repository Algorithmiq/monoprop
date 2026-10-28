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

"""Pytest configuration for monoprop tests.

MPI selection is handled here (``--with-mpi``, ``@pytest.mark.mpi``).

``comm`` is parametrized over ``None``, the serial run, and, with an MPI backend, ``comm_world``.
The world case carries the ``mpi`` marker, so it runs only under ``--with-mpi`` and is what
``-m mpi`` selects. Use ``comm`` for results every rank agrees on: energies, gradients,
``term_expectation_values``. Test rank-local state (``evolved_operator``, ``contract_partially``,
``size()``, ``graph_size()``) serially, with ``comm=None``.

``comm_world`` is a session-wide duplicate of COMM_WORLD, never the predefined handle itself, as
the parallelism docs tell users to do: monoprop's point-to-point traffic then cannot match
messages that conftest or mpi4py exchange on COMM_WORLD. It is created on first use and freed at
session teardown: ``Dup`` is collective, and doing it per test would add a collective to every
one of them for no extra isolation.

Under more than one rank each rank gets its own ``--basetemp``; see ``_rank_basetemp``.
"""

from __future__ import annotations

import getpass
import os
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.mpi_support import HAS_MPI_BACKEND, MPI
from tests.reference_data import load_problem

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mpi4py.MPI import Intracomm

    from tests.reference_data import FermionicProblem


def _rank_basetemp() -> Path | None:
    """Per-rank base temporary directory under ``mpiexec``, or ``None`` on a single rank.

    Every rank is its own pytest process, and without ``--basetemp`` they all prune the same
    ``pytest-of-<user>`` numbered directories at exit. One rank then deletes a directory another
    is scanning, and the resulting ``ResourceWarning`` fails the run under
    ``filterwarnings = error``. An explicit basetemp skips that pruning altogether.
    """
    if MPI is None or MPI.COMM_WORLD.size == 1:
        return None
    try:
        user = getpass.getuser()
    except (ImportError, OSError, KeyError):
        user = "unknown"
    temproot = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir())
    return temproot / f"pytest-of-{user}" / f"mpi-rank-{MPI.COMM_WORLD.rank}"


def _make_private_dir(path: Path) -> None:
    """Create ``path`` if needed and make sure only the current user can use it.

    The checks pytest applies to its own ``pytest-of-<user>``, which an explicit ``--basetemp``
    skips: on a shared temporary root another user could create the predictable name first.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Not following symlinks: a swapped-in link would otherwise pass every check below.
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise OSError(
            f"The temporary directory {path} is a symbolic link. Fix this and try again."
        )
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise OSError(
            f"The temporary directory {path} is not owned by the current user. "
            "Fix this and try again."
        )
    if info.st_mode & 0o077:
        path.chmod(0o700)


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Give each MPI rank its own ``--basetemp`` unless one was passed explicitly.

    Not in ``pytest_configure``: the tmpdir plugin reads the basetemp in its own
    ``pytest_configure``, and that hook here must run last to unregister the terminal reporter.
    """
    if config.option.basetemp is None and (basetemp := _rank_basetemp()) is not None:
        # pytest creates only the basetemp itself when one is given, not its parent.
        _make_private_dir(basetemp.parent)
        config.option.basetemp = basetemp


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``mpi``-marked tests when monoprop has no MPI backend, even under ``--with-mpi``."""
    if HAS_MPI_BACKEND:
        return
    skip = pytest.mark.skip(
        reason="this monoprop installation has no MPI backend (MPI_ABIS is empty)"
    )
    for item in items:
        if item.get_closest_marker("mpi") is not None:
            item.add_marker(skip)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in for tests launched under MPI."""
    parser.getgroup("mpi").addoption(
        "--with-mpi",
        action="store_true",
        default=False,
        help="Run MPI tests; launch pytest with mpiexec.",
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Enable ``--with-mpi`` under the VS Code adapter, which bypasses
    wrapper-level argument injection and would otherwise skip every ``mpi``-marked test.
    """
    invocation_args = tuple(str(arg) for arg in config.invocation_params.args)
    is_vscode_run = config.pluginmanager.hasplugin("vscode_pytest") or any(
        "vscode_pytest" in arg for arg in invocation_args
    )
    if is_vscode_run:
        config.option.with_mpi = True

    if config.option.with_mpi and MPI is not None and MPI.COMM_WORLD.rank != 0:
        reporter = config.pluginmanager.getplugin("terminalreporter")
        if reporter is not None:
            config.pluginmanager.unregister(reporter)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Enforce MPI opt-in and rank requirements before setting up fixtures."""
    for marker in item.iter_markers("mpi"):
        if not item.config.getoption("--with-mpi"):
            pytest.skip("need --with-mpi option to run")
        if MPI is None:
            pytest.fail("MPI tests require mpi4py")
        min_size = marker.kwargs.get("min_size", 1)
        if MPI.COMM_WORLD.size < min_size:
            pytest.skip(f"Test requires {min_size} MPI processes")


@pytest.fixture(scope="session")
def comm_world() -> Iterator[Intracomm | None]:
    """Session-wide duplicate of COMM_WORLD; ``None`` when monoprop has no MPI backend."""
    if not HAS_MPI_BACKEND:
        yield None
        return
    duplicate = MPI.COMM_WORLD.Dup()
    yield duplicate
    duplicate.Free()


_COMM_PARAMS = [pytest.param(None, id="comm_none")]
if HAS_MPI_BACKEND:
    _COMM_PARAMS.append(
        pytest.param("comm_world", id="comm_world", marks=pytest.mark.mpi)
    )


@pytest.fixture(params=_COMM_PARAMS)
def comm(request: pytest.FixtureRequest) -> Intracomm | None:
    """``None`` (serial), or ``comm_world`` under ``--with-mpi`` when monoprop has an MPI backend."""
    if request.param is None:
        return None
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        pytest.param("lih_fermionic_spin_exact", id="LiH_fermionic_spin"),
        pytest.param("rx_rz_ry_rz_exact", id="rx_rz_ry_rz"),
        pytest.param("random_exact", id="random_circuit"),
    ]
)
def problem(
    request: pytest.FixtureRequest, lazy_shared_datadir: Path
) -> FermionicProblem:
    """Load each general reference problem, freshly for every test invocation."""
    return load_problem(lazy_shared_datadir / f"{request.param}.msgpack")


@pytest.fixture
def lih_problem(lazy_shared_datadir: Path) -> FermionicProblem:
    """Load the LiH molecule with commutator reference data."""
    return load_problem(lazy_shared_datadir / "lih_fermionic_spin_exact.msgpack")


@pytest.fixture
def orbital_rotation_problem(lazy_shared_datadir: Path) -> FermionicProblem:
    """Load the S0 molecule used to check orbital rotations."""
    return load_problem(lazy_shared_datadir / "S0_8e8o_majoranic_c8.msgpack")
