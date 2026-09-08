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

"""Record the benchmark runner's physical core count for the ladder's later steps.

Every rung is sized against this number, so it must be the cores the run can actually use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psutil

import monoprop

TOPOLOGY = "/sys/devices/system/cpu/cpu{}/topology"


def visible_physical_cores(cpus):
    """Count distinct (package, core) pairs among CPUS; None if /sys is unreadable."""
    pairs = set()
    for cpu in cpus:
        topology = Path(TOPOLOGY.format(cpu))
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return None
        pairs.add((package, core))
    return len(pairs)


def main() -> str | None:
    if not monoprop.has_mpi:
        return "monoprop was built without MPI; every rung shares this binary"

    # PHYSICAL cores, and only the ones this process may run on. Physical, because that is what
    # the engine enumerates and the logical count would cross the `partitions > visible cores`
    # threshold on any SMT machine. Visible, because psutil counts the whole machine: under a
    # mask -- a cpuset, taskset, or a shared runner -- that number oversubscribes every rung and
    # names the testbed after cores the run never had.
    visible = os.sched_getaffinity(0)
    logical = psutil.cpu_count(logical=True)
    cores = visible_physical_cores(visible)
    if cores is None:
        # Without the mapping the machine's count is the only number left, and it is only safe
        # when nothing is masked.
        if len(visible) < logical:
            return (
                f"cannot read CPU topology and only {len(visible)} of {logical} CPUs are "
                "visible; export BENCH_CORES with the physical cores this mask covers"
            )
        cores = psutil.cpu_count(logical=False)
    if not cores:
        return "cannot determine the physical core count on this runner"

    print("variant ", monoprop.__variant__)
    print("cores   ", cores, "visible physical of", psutil.cpu_count(logical=False))
    print("affinity", len(visible), "of", logical, "logical")
    # Outside Actions there is no GITHUB_ENV to hand the next step: print the assignment
    # instead, so a local caller can `export` it and run the rungs by hand.
    if path := os.environ.get("GITHUB_ENV"):
        with open(path, "a") as env:
            print(f"BENCH_CORES={cores}", file=env)
    else:
        print(f"BENCH_CORES={cores}")
    return None


if __name__ == "__main__":
    sys.exit(main())
