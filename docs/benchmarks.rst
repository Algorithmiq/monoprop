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

Profiling the bound C++ code
============================

The benchmarks drive the compiled ``monoprop._core`` extension, so a sampling
CPU profiler (``perf``, Intel VTune) attributes time straight to the C++
kernels. Two properties of the *installed* extension make it unprofilable as
shipped, both fixed by a one-off rebuild:

- nanobind **strips** the symbol table from ``Release`` builds, so samples land
  on bare addresses with no function names.
- ``Release`` carries no debug info, so there is no line-level attribution.

Rebuild ``_core`` as ``RelWithDebInfo`` — the same ``-O3 -DNDEBUG`` codegen as
the shipped wheel, plus ``-g`` and an unstripped symbol table:

.. code-block:: bash

   env -u VIRTUAL_ENV uv sync --all-extras --group bench \
       --reinstall-package monoprop --no-cache \
       --config-settings-package="monoprop:cmake.build-type=RelWithDebInfo"

Capture one benchmark under ``perf``. Invoke pytest directly (not through
``just bench``) so the profiler's target is the Python process, pin to a single
thread for clean attribution, and unwind with DWARF — the build omits frame
pointers, so ``--call-graph fp`` will not work:

.. code-block:: bash

   monoprop_NUM_THREADS=1 perf record -g --call-graph dwarf,16384 -F 999 \
       -o build.perf -- .venv/bin/python -m pytest \
       benches/bench_random.py::test_random_build_graph \
       --num-modes 64 --cutoff 8 --bench-rounds 3 -q

``perf`` resolves the full mixed stack (Python → nanobind trampoline → the
templated ``MonomialPropagator<N>`` kernels, inlined frames included). Read it
back as a flat hotspot list, by source line, or as a flame graph (the latter
needs `FlameGraph <https://github.com/brendangregg/FlameGraph>`_ on your
``PATH``):

.. code-block:: bash

   perf report -i build.perf --stdio --no-children -g none   # flat hot functions
   perf report -i build.perf --stdio --sort srcline          # hot source lines
   perf script -i build.perf | stackcollapse-perf.pl | flamegraph.pl > build.svg

Sampling user space needs no special privileges (only kernel symbols stay
restricted, which does not affect the C++ kernels). Intel VTune profiles the
same rebuilt extension — ``vtune -collect hotspots`` for the hotspot view, or
``vtune -collect memory-access`` for per-object bandwidth and loaded latency.

.. note::

   The profile reflects whatever flags ``_core`` was actually compiled with;
   confirm them in ``build/editable/*/compile_commands.json``. Vectorisation
   flags (``-march``) on the binding translation units in particular change
   which kernels dominate — e.g. hardware versus software ``popcount``.

Worked example: vectorising the ``_core`` kernels
==================================================

The profiling pass above surfaced a concrete defect. On ``build_graph`` the
flat profile was ~31% ``fused_find_and_collect`` and **~18% libgcc
``__popcountdi2``** — a *software* popcount. The cause: the hot
``MonomialPropagator<N>`` kernels are header templates instantiated in the
``_core`` extension's own translation units, but the architecture flag
(``-march=native``) was applied only to the ``monoprop`` library, not to
``_core``. So those translation units compiled to scalar baseline ``x86-64``
and popcounts fell back to libgcc.

The fix applies the same flag block to the ``_core`` target (see
``src/monoprop/bindings/CMakeLists.txt``); the extension then emits hardware
``popcnt`` and vectorises the scan/bitset kernels. Measured on a large random
system — **500 generators**, 128 modes, cutoff 8, single thread, on an
i9-10900X — ``build_graph`` in the Heisenberg picture (median of 5 builds):

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - ``_core`` build
     - median
     - min
     - mean
   * - baseline (``-march=x86-64``)
     - 16.56 s
     - 16.20 s
     - 17.55 s
   * - ``-march=native`` (fixed)
     - 11.82 s
     - 11.60 s
     - 12.21 s
   * - **speedup**
     - **−28.6% (1.40×)**
     - −28.4%
     - −30.5%

The change is codegen-only and correctness-preserving: the core test suite
(``pytest tests/ -m "not mpi and not slow"``) gives an identical 405 passed
before and after. The win is larger than the popcount share alone because
``-march=native`` also vectorises the bitset/scan kernels.
