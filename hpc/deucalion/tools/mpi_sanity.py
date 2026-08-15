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
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # monoprop calls MPI_Abort if the library cannot provide at least
    # SERIALIZED: its partition-0 masters, not the main thread, make the calls.
    level = MPI.Query_thread()

    checksum = comm.allreduce(rank, op=MPI.SUM)
    gathered = comm.gather((rank, socket.gethostname(), len(os.sched_getaffinity(0))), root=0)

    if rank != 0:
        return

    import monoprop as mp

    print(f"ranks            : {size}")
    print(f"thread level     : {_LEVELS.get(level, level)} (need >= SERIALIZED)")
    print(f"allreduce        : {'ok' if checksum == size * (size - 1) // 2 else 'WRONG'}")
    print(f"monoprop.has_mpi : {mp.has_mpi}")
    print(f"MAX_NUM_MODES    : {mp.MAX_NUM_MODES}")
    hosts: dict[str, list[int]] = {}
    for r, host, cores in gathered or []:
        hosts.setdefault(f"{host} ({cores} cores/rank)", []).append(r)
    for host, ranks in sorted(hosts.items()):
        print(f"  {host}: ranks {min(ranks)}-{max(ranks)}")


if __name__ == "__main__":
    main()
