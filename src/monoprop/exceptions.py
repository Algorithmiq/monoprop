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

"""Custom exceptions for monoprop."""

from __future__ import annotations


class NumberOfModesInvalidError(ValueError):
    """Raised when the number of Fermionic modes is invalid."""


class MPIUnavailableError(RuntimeError):
    """Raised when a communicator is passed to a monoprop installation built without MPI."""


class MPIAbiMismatchError(RuntimeError):
    """Raised when mpi4py uses an MPI ABI family this monoprop installation was not built for."""


class MPIVersionError(RuntimeError):
    """Raised when the MPI library mpi4py uses is older than monoprop supports."""
