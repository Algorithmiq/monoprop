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

``REPORT.md`` has three sections — **Configuration** (ranks, threads, launcher,
host per run label), **Time**, and **Peak memory** — so every configuration is a
column and they sit side by side.

Random benchmarks
=================

The random benchmarks evolve a random observable through ``x`` random
length-``k`` Majorana generators. The generator length, number of observable
terms, number of generators, mode count, cutoff, and RNG seed are all
command-line options, for example:

.. code-block:: bash

   just bench --num-generators 200 --num-modes 64 --cutoff 10

- ``bench_random_evolve.py`` measures the graph-based path as four separate
  operations: building the graph, paring it into a masked execution plan, and
  evaluating the energy and the gradient.
- ``bench_random_inplace.py`` measures the in-place coefficient-truncation path,
  which never materialises the propagation graph.

Static benchmarks
=================

- ``bench_hubbard.py`` -- a 120-qubit (60-site) Fermi-Hubbard model with the
  sandbox default input, run as an in-place Trotter trajectory.
- ``bench_pauli.py`` -- a 127-qubit Pauli-basis kicked-Ising simulation on the
  IBM Eagle heavy-hex topology.

See ``benches/README.md`` for the full list of options and details.
