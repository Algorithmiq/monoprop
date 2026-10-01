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

from typing import Any

import pytest

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - exercised in wheel-test environments
    MPI = None


def pytest_configure(config: pytest.Config) -> None:
    """Force pytest-mpi's ``--with-mpi`` under the VS Code adapter, which bypasses
    wrapper-level argument injection and would otherwise skip every ``mpi``-marked test.
    """
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
