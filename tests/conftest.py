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

MPI selection is handled by ``pytest-mpi`` (``--with-mpi``, ``@pytest.mark.mpi``).

``comm`` is parametrized over COMM_SELF and COMM_WORLD: use it for energies and gradients, which
allreduce internally. ``serial_comm`` is always COMM_SELF: use it for rank-local state such as
``evolved_operator``, ``contract_partially``, ``size()``, or ``graph_size()``.
"""

from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.reference_data import load_problem

if TYPE_CHECKING:
    from tests.reference_data import FermionicProblem

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - exercised in wheel-test environments
    MPI = None


def _rank_basetemp() -> Path | None:
    """Per-rank base temporary directory under ``mpiexec``, or ``None`` when serial.

    Every rank is its own pytest process, and without ``--basetemp`` they all prune the same
    ``pytest-of-<user>`` numbered directories at exit. One rank then deletes a directory another
    is scanning, and the resulting ``ResourceWarning`` fails the run under
    ``filterwarnings = error``. An explicit basetemp skips that pruning altogether.
    """
    if MPI is None or MPI.COMM_WORLD.Get_size() == 1:
        return None
    try:
        user = getpass.getuser()
    except (ImportError, OSError, KeyError):
        user = "unknown"
    temproot = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir())
    return temproot / f"pytest-of-{user}" / f"mpi-rank-{MPI.COMM_WORLD.Get_rank()}"


# tryfirst: pytest's tmpdir plugin reads --basetemp in its own pytest_configure.
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Give each MPI rank its own ``--basetemp``, unless one was passed explicitly, and force
    pytest-mpi's ``--with-mpi`` under the VS Code adapter, which bypasses wrapper-level argument
    injection and would otherwise skip every ``mpi``-marked test.
    """
    if config.option.basetemp is None and (basetemp := _rank_basetemp()) is not None:
        config.option.basetemp = basetemp
    invocation_args = tuple(str(arg) for arg in config.invocation_params.args)
    is_vscode_run = config.pluginmanager.hasplugin("vscode_pytest") or any(
        "vscode_pytest" in arg for arg in invocation_args
    )
    if is_vscode_run and hasattr(config.option, "with_mpi"):
        config.option.with_mpi = True


_COMM_PARAMS = (
    [pytest.param(None, id="comm_none")]
    if MPI is None
    else [
        pytest.param(MPI.COMM_SELF, id="comm_self"),
        pytest.param(MPI.COMM_WORLD, id="comm_world"),
    ]
)


@pytest.fixture(params=_COMM_PARAMS)
def comm(request: pytest.FixtureRequest) -> Any:  # noqa: ANN401
    """Parametrized communicator; ``None`` when ``mpi4py`` is unavailable (the non-MPI wheel)."""
    return request.param


@pytest.fixture
def serial_comm() -> Any:  # noqa: ANN401
    """Single-rank communicator for tests that inspect rank-local operator state."""
    return None if MPI is None else MPI.COMM_SELF


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
