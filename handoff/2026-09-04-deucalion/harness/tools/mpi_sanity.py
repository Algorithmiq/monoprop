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

"""Smallest useful MPI check: launcher, thread level, pinning, and the extension.

Run under srun before trusting any larger job:
    srun --mpi=pmix -N2 --ntasks-per-node=8 --cpus-per-task=16 \
         --cpu-bind=cores ./.venv/bin/python hpc/deucalion/tools/mpi_sanity.py
"""

from __future__ import annotations

import os
import socket

from mpi4py import MPI

_LEVELS = {
    MPI.THREAD_SINGLE: "SINGLE",
    MPI.THREAD_FUNNELED: "FUNNELED",
    MPI.THREAD_SERIALIZED: "SERIALIZED",
    MPI.THREAD_MULTIPLE: "MULTIPLE",
}


def main() -> None:
    """Report the launcher's MPI thread level, collective health, and per-rank pinning."""
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # monoprop calls MPI_Abort if the library cannot provide at least
    # SERIALIZED: its partition-0 masters, not the main thread, make the calls.
    level = MPI.Query_thread()

    checksum = comm.allreduce(rank, op=MPI.SUM)
    gathered = comm.gather(
        (rank, socket.gethostname(), len(os.sched_getaffinity(0))), root=0
    )

    if rank != 0:
        return

    # Deferred, and past the rank-0 early return on purpose: importing the extension pulls in
    # the whole engine, and only rank 0 reports.
    import monoprop as mp  # noqa: PLC0415

    print(f"ranks            : {size}")
    print(f"thread level     : {_LEVELS.get(level, level)} (need >= SERIALIZED)")
    print(
        f"allreduce        : {'ok' if checksum == size * (size - 1) // 2 else 'WRONG'}"
    )
    print(f"monoprop.has_mpi : {mp.has_mpi}")
    print(f"MAX_NUM_MODES    : {mp.MAX_NUM_MODES}")
    hosts: dict[str, list[int]] = {}
    for r, host, cores in gathered or []:
        hosts.setdefault(f"{host} ({cores} cores/rank)", []).append(r)
    for host, ranks in sorted(hosts.items()):
        print(f"  {host}: ranks {min(ranks)}-{max(ranks)}")


if __name__ == "__main__":
    main()
