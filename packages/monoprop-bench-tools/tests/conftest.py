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

"""Pytest configuration for the monoprop-bench-tools test suite.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - exercised in non-MPI environments
    MPI = None


@pytest.fixture
def serial_comm() -> Any:
    """Single-rank communicator for builders whose state is rank-local."""
    return None if MPI is None else MPI.COMM_SELF
