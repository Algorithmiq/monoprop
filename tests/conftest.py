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

MPI test selection is handled by ``pytest-mpi`` via the ``--with-mpi`` flag
and ``@pytest.mark.mpi`` markers.

Two communicator fixtures are available:

- ``comm`` — parametrized over COMM_SELF and COMM_WORLD.  Every test that
  uses it appears twice in the VS Code sidebar (``[comm_self]`` and
  ``[comm_world]``).  Use for tests that compare **energies or gradients**
  (methods that internally do MPI allreduce).

- ``serial_comm`` — always COMM_SELF.  Use for tests that inspect
  rank-local state such as ``evolved_operator``, ``contract_partially``,
  ``size()``, or ``graph_size()``.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - exercised in wheel-test environments
    MPI = None


def pytest_configure(config: pytest.Config) -> None:
    """Force pytest-mpi mode for VS Code adapter runs.

    The VS Code adapter invokes pytest with ``-p vscode_pytest`` and can bypass
    wrapper-level argument injection. In that mode we explicitly enable
    pytest-mpi's ``--with-mpi`` behaviour to avoid skipping all ``@pytest.mark.mpi`` tests.
    """
    invocation_args = tuple(str(arg) for arg in config.invocation_params.args)
    is_vscode_run = config.pluginmanager.hasplugin("vscode_pytest") or any(
        "vscode_pytest" in arg for arg in invocation_args
    )
    if is_vscode_run and hasattr(config.option, "with_mpi"):
        config.option.with_mpi = True
    config.addinivalue_line(
        "markers",
        "unsharded: build propagators single-partition (sets monoprop_SHARDS=off). Operator sharding "
        "is the auto-default, but tests that inspect raw per-partition internals (indexing/mp_op/graph, "
        "which have no single value on a shard facade) must run unsharded.",
    )


@pytest.fixture(autouse=True)
def _shard_policy(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force single-partition for tests marked ``unsharded``.

    Sharding is the default parallelism (``monoprop_SHARDS`` unset ⇒ one shard per core), so every
    propagator built in the suite is shard-backed by default — which is exactly what we want to
    exercise. The exception is white-box tests that reach into raw engine internals; those opt out
    with ``@pytest.mark.unsharded`` (or a module-level ``pytestmark``). The env is read at propagator
    construction, so setting it in an autouse fixture (before the test body) is sufficient.
    """
    if request.node.get_closest_marker("unsharded"):
        monkeypatch.setenv("monoprop_SHARDS", "off")


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
    """Parametrized communicator fixture.

    Falls back to ``None`` when ``mpi4py`` is unavailable, which matches the
    non-MPI wheel configuration used in cibuildwheel tests.
    """
    return request.param


@pytest.fixture
def serial_comm() -> Any:  # noqa: ANN401
    """Single-rank communicator for tests that inspect rank-local operator state."""
    return None if MPI is None else MPI.COMM_SELF
