"""Track W reproducer: Hubbard propagate, ledger, VmHWM, optional MPI comm.

usage: w_repro.py <steps> <cutoff> <lower_atol> <label> [--mpi]

One RESULT line per rank plus the whole operator ledger, so a shell loop can collect either.
Mirrors storage-logs/ab_reproducer.py; the only additions are VmHWM and the --mpi arm.
"""

import hashlib
import pathlib
import sys
import time

steps = int(sys.argv[1])
cutoff = int(sys.argv[2])
atol = float(sys.argv[3])
label = sys.argv[4]
use_mpi = "--mpi" in sys.argv[5:]

comm = None
rank = 0
if use_mpi:
    from mpi4py import MPI  # noqa: E402

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

import monoprop  # noqa: E402
import monoprop._core  # noqa: E402
from monoprop_bench_tools.models import HubbardConfig, build_hubbard_problem  # noqa: E402


def vmhwm_kib():
    for line in pathlib.Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return -1


cfg = HubbardConfig(cutoff=cutoff, lower_atol=atol, trotter_steps=steps)
prop, circuit = build_hubbard_problem(cfg, comm=comm)

t0 = time.perf_counter()
for _ in range(steps):
    prop.propagate(circuit)
t1 = time.perf_counter()

so = pathlib.Path(monoprop._core.__file__)
md5 = hashlib.md5(so.read_bytes()).hexdigest()
core = prop._simulator
b = core.operator_memory_breakdown()
print(
    f"RESULT label={label} rank={rank} wall={t1 - t0:.4f} size={prop.size()} md5={md5} "
    f"total_bytes={b['total_bytes']} gate_scratch_bytes={b['gate_scratch_bytes']} "
    f"d_gate_buffers_hwm_bytes={b['d_gate_buffers_hwm_bytes']} vmhwm_kib={vmhwm_kib()}",
    flush=True,
)
for k in sorted(b):
    print(f"  LEDGER rank={rank} {k} = {b[k]}", flush=True)
