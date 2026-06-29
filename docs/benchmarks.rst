==========
Benchmarks
==========

monoprop ships a pytest-based benchmark suite under ``benches/`` that measures
the **wall-clock time** and **peak memory** of the core operations. Timing uses
`pytest-benchmark <https://pytest-benchmark.readthedocs.io>`_ and memory uses
`pytest-memray <https://pytest-memray.readthedocs.io>`_, whose allocator-level
tracking captures the C++-side graph and coefficient allocations that dominate
the library.

The suite is **not** part of the normal test run (``testpaths`` is ``tests/``).
A single command runs the timing pass, the memory pass, and regenerates the
report:

.. code-block:: bash

   just bench         # serial run -> benches/results/REPORT.md
   just bench-smoke   # quick sanity check with tiny sizes

Arguments after ``just bench`` are handled by the driver (``benches/run.py``):
``--ranks``, ``--mpiexec-args``, ``--env``, ``--label`` and ``--no-mem`` are
consumed, everything else is forwarded to pytest (sizing options,
``--bench-rounds``, test selection).

Configuration (MPI, threads, pinning)
=====================================

Configuration is set on the command line and recorded in the report:

- ``--ranks N`` runs under ``mpiexec -n N``; ``--mpiexec-args`` passes the rest
  (e.g. core pinning with ``--bind-to core``).
- ``--env KEY=VALUE`` (repeatable) sets environment variables for the run.
  ``monoprop`` reads its thread count from ``monoprop_NUM_THREADS``; add
  ``OMP_NUM_THREADS`` / ``OMP_PROC_BIND`` / ``OMP_PLACES`` for OpenMP pinning.
- ``--label`` names the run (default ``np<ranks>``) — use a custom label to
  compare thread/pinning variants at the same rank count.

MPI needs an MPI-enabled build (built once, as ``just test-py-mpi`` does):

.. code-block:: bash

   just bench-build-mpi                                       # one-time MPI rebuild
   just bench --ranks 4 --mpiexec-args="--bind-to core" \
       --env monoprop_NUM_THREADS=2 --label r4t2

The benchmarks are communicator-aware: each operation is barrier-wrapped so the
measured time is the makespan across ranks, only rank 0 writes the timing JSON,
and peak memory is captured per rank.

``REPORT.md`` opens with a **Configuration** table (ranks, threads, launcher,
host per run label), a **Hyperparameters** table (the resolved random-problem
sizes and run knobs each run used) and a **Graph size** table (the number of
terms reached per picture), then a **Heisenberg** and a **Schrödinger** section,
each holding a **Time** and a **Peak memory** table. Every run is a column, so
configurations sit side by side; the Schrödinger section is omitted when no
Schrödinger benchmarks were run.

Random benchmarks
=================

All benchmarks live in a single file, ``benches/bench_monoprop.py``. The random
benchmarks evolve a random observable through a configurable number of random
fixed-length Majorana generators, and run in **both** the Heisenberg and
Schrödinger pictures (the latter with ``schrodinger_cutoff = cutoff + 2``). The
generator length, number of observable terms, number of generators, mode count,
cutoff, and RNG seed are all command-line options, for example:

.. code-block:: bash

   just bench --num-generators 200 --num-modes 64 --cutoff 10

- ``test_random_build_graph``, ``test_random_pare``, ``test_random_energy`` and
  ``test_random_gradient`` measure the graph-based path: building the graph,
  paring it into a masked execution plan, and evaluating the energy and the
  gradient.
- ``test_random_inplace`` measures the in-place coefficient-truncation path,
  which never materialises the propagation graph.

Static benchmarks
=================

- ``test_static[hubbard]`` -- a 120-qubit (60-site) Fermi-Hubbard model with the
  sandbox default input, run as an in-place Trotter trajectory.
- ``test_static[pauli]`` -- a 127-qubit Pauli-basis kicked-Ising simulation on the
  IBM Eagle heavy-hex topology.

Both run in the Heisenberg picture; pass ``--lower-atol VALUE`` to override their
coefficient-truncation tolerance.

See ``benches/README.md`` for the full list of options and details.
