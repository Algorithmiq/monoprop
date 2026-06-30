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

"""Preflight check that the loaded ``monoprop`` extension was built with MPI."""

from __future__ import annotations

import sys

import monoprop


def main() -> int:
    """Report whether the loaded extension was built with MPI; return an exit code."""
    if monoprop.has_mpi:
        sys.stdout.write("[mpi-check] OK: monoprop was built with MPI.\n")
        return 0

    sys.stderr.write(
        "[mpi-check] FAIL: the loaded monoprop extension was built without MPI "
        "(monoprop.has_mpi = False).\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
