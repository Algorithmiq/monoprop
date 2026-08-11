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

"""Distributed counterpart of ``monoprop_single_layer.py``.

Same physics -- the builders are imported from that module, so there is one
definition of the circuit and the observable -- but the propagator is handed
``MPI.COMM_WORLD`` and the measurements are collective. This is the driver for
problem sizes whose operator does not fit one node.

Differences from the serial driver, all forced by distribution:

* **Timing is a makespan.** The timed region is barrier-bracketed on both sides,
  so what is recorded is the slowest rank, not rank 0's private view.
* **Memory is summed across ranks.** ``operator_memory_bytes()`` is documented as
  "total bytes held by the operator *on this rank*"; the job-wide figure is its
  allreduce. (``operator_memory_usage()`` handles the partitioned case internally
  via ``partitioned_operator_memory_usage_()``, so no special casing is needed --
  contrary to the note in ``paper_plots/README.md``, which describes older
  behaviour where the accessor raised once the propagator sharded.)
* **Term count is summed across ranks.** ``size()`` is rank-local; terms are
  assigned to ranks by a stateless hash, so the sum is the operator size.
* **One output file per job, written by rank 0 only.** The serial drivers append
  to a shared JSONL, which is not safe from many concurrent jobs on Lustre.
  Concatenate afterwards.

Example::

    srun --mpi=pmix -N16 --ntasks-per-node=8 --cpus-per-task=16 \\
         --cpu-bind=cores --distribution=block:block \\
         ./.venv/bin/python paper_plots/scripts/monoprop_multinode.py \\
         --basis pauli --num-qubits 1024 --cutoff 6 --layers 5 \\
         --lower-atol 0 --outdir runs/multinode
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from pathlib import Path
from time import perf_counter

from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monoprop_single_layer import build_majorana, build_pauli  # noqa: E402


def _partition_count() -> str:
    """What monoprop will resolve S to, as far as the environment reveals it."""
    for var in ("monoprop_PARTITIONS", "monoprop_NUM_THREADS"):
        value = os.environ.get(var)
        if value:
            return f"{var}={value}"
    # A multi-rank run with neither set silently gets ONE partition per rank.
    return "unset (=> 1 partition per rank)"


def _vm_hwm_bytes() -> int:
    """This rank's peak resident set, from the kernel rather than from sampling.

    VmHWM is a watermark the kernel maintains, so it cannot miss a transient the
    way a periodic RSS read can. Returns 0 off Linux rather than raising -- the
    metric is diagnostic, and a missing one should not fail the run.
    """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices=["majorana", "pauli"], required=True)
    parser.add_argument("--num-qubits", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument(
        "--observable", choices=["extensive", "local"], default="extensive"
    )
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lower-atol", type=float, default=1e-8)
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="fresh rebuild+apply repetitions; the min makespan is kept",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="directory for the JSONL record; rank 0 writes one file per job",
    )
    parser.add_argument(
        "--tag", default="", help="extra label folded into the output filename"
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # Explicit, always. In an MPI build `comm=None` would mean COMM_WORLD anyway,
    # but relying on that makes the serial/distributed distinction invisible.
    builder = build_majorana if args.basis == "majorana" else build_pauli
    factory, circuit = builder(
        args.num_qubits, args.cutoff, args.lower_atol, args.observable, comm=comm
    )

    best = float("inf")
    num_terms = 0
    memory_bytes = 0
    expectation = float("nan")

    for _ in range(max(1, args.rounds)):
        sim = factory()

        # Bracket the timed region so every rank measures the same interval and
        # the result is the makespan rather than one rank's local view.
        comm.Barrier()
        t0 = perf_counter()
        for _ in range(args.layers):
            sim.propagate(circuit)
        comm.Barrier()
        dt = perf_counter() - t0
        best = min(best, dt)

        # size() and operator_memory_bytes() are both rank-local.
        num_terms = comm.allreduce(int(sim.size()), op=MPI.SUM)
        memory_bytes = comm.allreduce(
            int(sim._simulator.operator_memory_bytes()), op=MPI.SUM
        )
        # expectation_value() allreduces internally, so it is already global.
        expectation = float(sim.expectation_value())
        del sim

    # Resident footprint alongside the operator bytes, because the two answer
    # different questions and the gap between them is the interesting number.
    # operator_memory_bytes() counts the operator's own allocation; VmHWM counts
    # everything the rank ever touched, including the interpreter, the MPI
    # transport and any per-rank bookkeeping that scales with the world size.
    # On a fixed problem the first should be flat in R and the second should not
    # be -- that difference is what tells you whether growth is data or overhead.
    rss_bytes = comm.allreduce(_vm_hwm_bytes(), op=MPI.SUM)
    rss_max = comm.allreduce(_vm_hwm_bytes(), op=MPI.MAX)

    if rank != 0:
        return

    record = {
        "engine": "monoprop",
        "mode": "multinode",
        "basis": args.basis,
        "num_qubits": args.num_qubits,
        "cutoff": args.cutoff,
        "observable": args.observable,
        "layers": args.layers,
        "lower_atol": args.lower_atol,
        "ranks": size,
        "partitions_env": _partition_count(),
        "nodes": int(os.environ.get("SLURM_JOB_NUM_NODES", 0)) or None,
        "ntasks_per_node": os.environ.get("SLURM_NTASKS_PER_NODE"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "num_terms": num_terms,
        "memory_bytes": memory_bytes,
        "bytes_per_term": (memory_bytes / num_terms) if num_terms else 0.0,
        "rss_bytes": rss_bytes,
        "rss_max_per_rank": rss_max,
        "rss_per_rank": rss_bytes / size if size else 0,
        # The overhead each rank carries beyond its share of the operator. If this
        # climbs with the rank count on a fixed problem, per-rank bookkeeping is
        # scaling with the world size and will cap how far the job can be spread.
        "overhead_per_rank": (rss_bytes - memory_bytes) / size if size else 0,
        "seconds": best,
        "expectation": expectation,
        "host": socket.gethostname(),
        "python": platform.python_version(),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.basis}_N{args.num_qubits}_c{args.cutoff}"
        f"_r{size}_job{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    if args.tag:
        stem += f"_{args.tag}"
    (outdir / f"{stem}.jsonl").write_text(json.dumps(record) + "\n")

    print(
        f"[monoprop/{args.basis}] N={args.num_qubits} cutoff={args.cutoff} "
        f"ranks={size} terms={num_terms} "
        f"mem={memory_bytes / 1024**3:.2f}GiB "
        f"b/term={record['bytes_per_term']:.1f} "
        f"t={best:.4f}s exp={expectation:.6g}"
    )


if __name__ == "__main__":
    main()
