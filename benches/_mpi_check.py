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

"""Preflight check that the MPI bench actually distributes work across ranks.

monoprop splits operator terms across ranks by hashing each term to a rank
(``find_rank``), keyed on the communicator size the **C++** extension sees. If
the loaded ``monoprop`` extension was built without MPI (``monoprop_ENABLE_MPI``
defaults to ``OFF``, and a plain ``uv run`` / ``uv sync`` rebuilds it that way,
clobbering ``just bench-build-mpi``), the C++ side reads ``size == 1`` and every
rank keeps the **entire** operator. The benchmark then runs N redundant serial
copies that each hold the full graph -- no speedup, and N x the memory, which
OOMs at large sizes.

This module detects that by building a small problem twice on each rank -- once
distributed (``comm=COMM_WORLD``) and once serial (``comm=None``) -- and checking
that the distributed per-rank term count is a *fraction* of the serial count.
Run it under ``mpiexec`` before the heavy passes::

    mpiexec -n 5 python benches/_mpi_check.py

Exit code 0 means distribution is working (or it is a serial run); exit code 1
means the loaded build is not distributing and the bench should not proceed.
"""

from __future__ import annotations

import sys

from _builders import build_random_propagator, make_random_problem

# Small, fast probe problem. Distribution behaviour is size-independent, so keep
# it cheap; Heisenberg keeps the graph tiny.
_PROBE = {
    "gen_length": 4,
    "obs_terms": 2000,
    "num_generators": 40,
    "num_modes": 24,
    "cutoff": 6,
}
# A rank is "not distributing" if it still holds at least this fraction of the
# full serial operator. Working distribution gives ~1/size per rank; a non-MPI
# build gives 1.0 on every rank.
_REPLICATED_FRACTION = 0.9


def verify_distribution(comm: object) -> tuple[bool, str]:
    """Check that ``monoprop`` distributes the operator across ``comm``.

    Args:
        comm: An MPI communicator, or ``None`` for a serial run.

    Returns:
        ``(ok, message)``. ``ok`` is ``True`` when distribution is working (or
        the run is serial); ``False`` when every rank holds the full operator,
        which means the loaded extension is not MPI-distributing.
    """
    size = 1 if comm is None else comm.Get_size()
    if size == 1:
        return True, "serial run (size 1): nothing to distribute"

    from mpi4py import MPI  # noqa: PLC0415

    problem = make_random_problem(**_PROBE)
    distributed = build_random_propagator(problem, comm=comm, schrodinger=False)
    distributed.propagate()
    local_terms = distributed.size()

    # Serial reference: a size-1 communicator forces the *full* operator onto
    # this rank, giving a ground-truth global term count. (``comm=None`` would
    # NOT work -- the binding maps it to MPI_COMM_WORLD, so under mpiexec it
    # would distribute too and the comparison would be meaningless.)
    serial = build_random_propagator(problem, comm=MPI.COMM_SELF, schrodinger=False)
    serial.propagate()
    full_terms = serial.size()

    total_terms = comm.allreduce(local_terms, op=MPI.SUM)
    fraction = local_terms / full_terms if full_terms else 1.0
    ok = fraction < _REPLICATED_FRACTION
    message = (
        f"size={size}: per-rank terms={local_terms:,} of serial total "
        f"{full_terms:,} (fraction {fraction:.2f}, expected ~{1 / size:.2f}); "
        f"allreduce sum={total_terms:,}"
    )
    if not ok:
        message += (
            " -> NOT DISTRIBUTING: every rank holds the full operator. The "
            "loaded monoprop extension was built without MPI. Rebuild with "
            "`just bench-build-mpi` and run the bench so `uv` does not "
            "re-sync a non-MPI build over it (the bench recipes pass "
            "`--no-sync`)."
        )
    return ok, message


def main() -> int:
    """Run the distribution check under MPI and return a process exit code."""
    try:
        from mpi4py import MPI  # noqa: PLC0415
    except ImportError:
        sys.stdout.write("[mpi-check] mpi4py not available; skipping.\n")
        return 0

    comm = MPI.COMM_WORLD
    ok, message = verify_distribution(comm)
    if comm.Get_rank() == 0:
        status = "OK" if ok else "FAIL"
        sys.stdout.write(f"[mpi-check] {status}: {message}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
