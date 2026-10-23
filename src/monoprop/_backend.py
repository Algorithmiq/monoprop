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

"""Backend selection for monoprop's extension modules."""

from __future__ import annotations

from monoprop._mpi_abis import ABIS as _BUILT_MPI_ABIS

MPI_ABIS: tuple[str, ...] = _BUILT_MPI_ABIS
"""MPI ABI families this installation can drive, for example ``("mpich", "openmpi")``.

``"mpich"`` covers MPICH, Intel MPI, MVAPICH and Cray MPICH; ``"openmpi"`` covers Open MPI 4 and
5; ``"mpiabi"`` is the MPI-5 standard ABI. The tuple is empty for an installation built without
MPI. It states what was built, not that a usable MPI library is present: that depends on the MPI
library mpi4py loads at runtime.
"""
