==========
Benchmarks
==========

monoprop ships a pytest-based benchmark suite under ``benches/`` that measures
the **wall-clock time** and **peak physical memory** of the core operations.
Timing uses `pytest-benchmark <https://pytest-benchmark.readthedocs.io>`_; memory
is read from the kernel (peak RSS corrected to proportional set size, PSS),
capturing the C++-side allocations that dominate the library at no measurable
cost, so timing and memory share one pass.

The suite is **not** part of the normal test run. One command runs the sweep and
regenerates the report:

.. code-block:: bash

   just bench         # serial run -> benches/results/REPORT.md
   just bench-smoke   # quick sanity check with tiny sizes

The driver (``benches/run.py``) consumes ``--ranks``, ``--mpiexec-args``,
``--env`` and ``--label``; everything else is forwarded to pytest (sizing
options, ``--bench-rounds``, test selection).

Configuration (MPI, threads, pinning)
=====================================

Configuration is set on the command line and recorded in the report:

- ``--ranks N`` runs under ``mpiexec -n N``; ``--mpiexec-args`` passes the rest
  (e.g. core pinning with ``--bind-to core``).
- ``--env KEY=VALUE`` (repeatable) sets environment variables. ``monoprop`` reads
  its thread count from ``monoprop_NUM_THREADS``; add ``OMP_NUM_THREADS`` /
  ``OMP_PROC_BIND`` / ``OMP_PLACES`` for OpenMP pinning.
- ``--label`` names the run (default ``np<ranks>``) — use a custom label to
  compare thread/pinning variants at the same rank count.

MPI needs an MPI-enabled build:

.. code-block:: bash

   just bench-build-mpi                                       # one-time MPI rebuild
   just bench --ranks 4 --mpiexec-args="--bind-to core" \
       --env monoprop_NUM_THREADS=2 --label r4t2

Each operation is barrier-wrapped so the measured time is the makespan across
ranks; rank 0 writes the timing JSON and memory is the per-rank peak PSS summed
across ranks (true physical RAM). Every run adds a column to ``REPORT.md`` —
**Configuration**, **Hyperparameters** and **Operator size** tables, then a
**Heisenberg** and a **Schrödinger** section each with a **Time** and a
**Memory (PSS)** table (Schrödinger omitted when unused).

Benchmarks
==========

All benchmarks live in ``benches/bench_monoprop.py``.

**Random** benchmarks evolve a random observable through configurable random
fixed-length Majorana generators, in both the Heisenberg and Schrödinger pictures
(the latter with ``schrodinger_cutoff = cutoff + 2``). Generator length,
observable terms, generator count, mode count, cutoff and RNG seed are
command-line options:

.. code-block:: bash

   just bench --num-generators 200 --num-modes 64 --cutoff 10

They cover the graph-based path (``build_graph``, ``pare``, ``energy``,
``gradient``) and the in-place truncation path (``inplace``), which never
materialises the propagation graph.

**Static** benchmarks are fixed, heavy, Heisenberg-only in-place simulations:

- ``test_static[hubbard]`` -- a 120-qubit (60-site) Fermi-Hubbard Trotter
  trajectory.
- ``test_static[pauli]`` -- a 127-qubit Pauli-basis kicked-Ising circuit on the
  IBM Eagle heavy-hex topology.

Both use a ``lower_atol`` of 1e-4; ``--hubbard-lower-atol`` / ``--pauli-lower-atol``
override each model's coefficient-truncation tolerance.

See ``benches/README.md`` for the full list of options and details.
