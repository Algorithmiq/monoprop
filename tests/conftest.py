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

``comm`` is parametrized over COMM_SELF and COMM_WORLD: use it for energies and gradients, which
allreduce internally. ``serial_comm`` is always COMM_SELF: use it for rank-local state such as
``evolved_operator``, ``contract_partially``, ``size()``, or ``graph_size()``.

Under more than one rank each rank gets its own ``--basetemp``; see ``_rank_basetemp``.
"""

from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.mpi_support import HAS_MPI_BACKEND, MPI, SERIAL_COMM
from tests.reference_data import load_problem

if TYPE_CHECKING:
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


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Give each MPI rank its own ``--basetemp`` unless one was passed explicitly.

    Not in ``pytest_configure``: the tmpdir plugin reads the basetemp in its own
    ``pytest_configure``, and that hook here must run last to unregister the terminal reporter.
    """
    if config.option.basetemp is None and (basetemp := _rank_basetemp()) is not None:
        # pytest creates only the basetemp itself when one is given, not its parent.
        basetemp.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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


_COMM_PARAMS = (
    [pytest.param(None, id="comm_none")]
    if not HAS_MPI_BACKEND
    else [
        pytest.param(MPI.COMM_SELF, id="comm_self"),
        pytest.param(MPI.COMM_WORLD, id="comm_world"),
    ]
)


@pytest.fixture(params=_COMM_PARAMS)
def comm(request: pytest.FixtureRequest) -> Any:  # noqa: ANN401
    """Parametrized communicator; ``None`` when mpi4py is unavailable or monoprop has no MPI backend."""
    return request.param


@pytest.fixture
def serial_comm() -> Any:  # noqa: ANN401
    """Single-rank communicator for tests that inspect rank-local operator state."""
    return SERIAL_COMM


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
