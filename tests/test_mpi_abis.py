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

"""``monoprop.MPI_ABIS``: the MPI ABI families an installation was built for."""

from __future__ import annotations

import os

import pytest

import monoprop

_KNOWN_ABIS = {"mpich", "openmpi", "mpiabi"}
_ON = {"on", "true", "yes", "1"}


def test_mpi_abis_names_known_families_once() -> None:
    assert isinstance(monoprop.MPI_ABIS, tuple)
    assert set(monoprop.MPI_ABIS) <= _KNOWN_ABIS
    assert len(set(monoprop.MPI_ABIS)) == len(monoprop.MPI_ABIS)


def test_has_mpi_was_removed() -> None:
    assert not hasattr(monoprop, "has_mpi")


def test_mpi_abis_follow_the_build_switch() -> None:
    """CI exports monoprop_ENABLE_MPI to both the build and the test steps."""
    switch = os.environ.get("monoprop_ENABLE_MPI")  # noqa: SIM112
    if switch is None:
        pytest.skip("monoprop_ENABLE_MPI is not set in the test environment")
    assert bool(monoprop.MPI_ABIS) == (switch.lower() in _ON)
