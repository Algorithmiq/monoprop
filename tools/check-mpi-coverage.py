#!/usr/bin/env python3
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

"""Fail when an MPI coverage run did not actually exercise the MPI sources.

Reads a gcovr JSON report and requires covered lines in ``MPICompat.cpp`` and in at least
two sources under ``detail/mpi/``. Without this, an MPI lane that silently ran without MPI
would still produce a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def covered_mpi_sources(report: dict) -> list[str]:
    """Return the ``detail/mpi/`` sources with at least one covered line.

    Args:
        report: A parsed gcovr JSON report.

    Returns:
        The covered source paths, with forward slashes.
    """
    covered = []
    for entry in report["files"]:
        path = entry["file"].replace("\\", "/")
        if "/detail/mpi/" not in f"/{path}":
            continue
        if any(line.get("count", 0) > 0 for line in entry.get("lines", [])):
            covered.append(path)
    return covered


def main() -> None:
    """Check the report named on the command line.

    Raises:
        SystemExit: If the MPI sources are not covered.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="gcovr JSON report")
    args = parser.parse_args()

    covered = covered_mpi_sources(json.loads(args.report.read_text()))
    if not any(path.endswith("/MPICompat.cpp") for path in covered):
        raise SystemExit("MPICompat.cpp has no covered lines")
    if len(covered) < 2:
        raise SystemExit(
            f"expected coverage in at least two MPI sources, found: {covered}"
        )
    sys.stdout.write(
        "Covered MPI sources:\n" + "".join(f"  {path}\n" for path in covered)
    )


if __name__ == "__main__":
    main()
