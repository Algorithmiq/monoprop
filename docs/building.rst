====================
Building from source
====================

monoprop has two build products, each with its own build system:

- the **Python bindings** — the nanobind extension behind ``import monoprop``,
  built with scikit-build-core and driven by ``uv`` (or ``pip``);
- the **standalone C++ library and executables** — built directly with CMake
  presets.

MPI is **off by default** in every build path; you enable it explicitly. The
mechanism differs by build:

.. list-table::
   :header-rows: 1

   * - Build
     - Enable MPI with
   * - Python bindings (scikit-build / ``uv`` / ``pip``)
     - ``--config-settings=cmake.define.monoprop_ENABLE_MPI=ON`` (or export
       ``SKBUILD_CMAKE_ARGS="-Dmonoprop_ENABLE_MPI=ON"``)
   * - C++ (CMake presets)
     - use the ``*-mpi`` preset

The prebuilt wheels published to PyPI (``pip install monoprop``) are also built
without MPI, so a from-source build is required for multi-rank runs.


Prerequisites
=============

- a C++23 compiler (GCC 13+ or Clang 17+)
- CMake and Ninja
- Python 3.11 or newer and the ``uv`` package manager (for the bindings)
- an MPI implementation such as Open MPI (only for MPI builds)

The repository ships a `DevContainer <https://containers.dev/>`_ with all of the
above pre-configured; opening the folder in VS Code and rebuilding the container is
the quickest route to a working environment.

Building the Python bindings
============================

``uv`` creates a virtual environment, installs the Python dependencies, and
compiles the nanobind extension in editable mode. Re-run the sync command whenever
the dependency graph or the C++ sources change.

.. note::

   These commands deliberately do not use ``--all-groups``: the ``docs`` group
   requires Python 3.12 (we develop against 3.11), so it is excluded from the
   default environment and built separately via ``just build-docs`` (which uses
   ``--python 3.12``).

Without MPI (default)
---------------------

.. code-block:: bash

   uv sync --all-extras -v

This produces a single-process build with no MPI dependency.

With MPI
--------

Pass a ``config-settings`` override to enable MPI:

.. code-block:: bash

   uv sync --all-extras -v \
       --config-settings=cmake.define.monoprop_ENABLE_MPI=ON

The same override works with ``pip`` when installing from a checkout:

.. code-block:: bash

   pip install . --config-settings=cmake.define.monoprop_ENABLE_MPI=ON

Verify the install
------------------

.. code-block:: bash

   uv run python -c "import monoprop as mp; print(mp.__version__)"

Running the bindings
--------------------

A serial run is just a normal Python invocation:

.. code-block:: bash

   uv run python your_script.py

For a multi-rank run, launch the same script under ``mpiexec`` (requires an MPI
build) and pass ``comm=MPI.COMM_WORLD`` to the simulator:

.. code-block:: bash

   mpiexec -n 8 uv run python your_script.py

See :doc:`/features/parallelism` for the communicator options and the
shared-memory thread controls.

Building the C++ library and executables
========================================

The standalone build uses CMake presets (Ninja generator). It produces the
``monoprop`` library plus the example, benchmark, and test executables under
``build/<preset>/bin/``.

Default preset (no MPI)
-----------------------

.. code-block:: bash

   cmake --preset release-gcc
   cmake --build --preset release-gcc

MPI preset
----------

Use the ``-mpi`` preset, which sets ``monoprop_ENABLE_MPI=ON``:

.. code-block:: bash

   cmake --preset release-gcc-mpi
   cmake --build --preset release-gcc-mpi

``debug-gcc`` / ``debug-gcc-mpi`` presets are available for debug builds. To build a
single target, pass ``--target``:

.. code-block:: bash

   cmake --build --preset release-gcc --target example.x

Running the tests
=================

Python tests
------------

The Python tests run with ``pytest``. Without MPI:

.. code-block:: bash

   uv run python -m pytest -m "not mpi"   # or: just test-py

The MPI tests need an MPI-enabled build, which the default source build does not
produce. The ``just`` recipes build an MPI-enabled extension and launch the suite
under ``mpiexec``:

.. code-block:: bash

   just test-py-mpi          # full suite under MPI
   just test-py-mpi-matrix   # MPI-marked tests across a rank matrix

To do it by hand, build with MPI on, then run under ``mpiexec`` with ``--no-sync``
so each rank reuses that build. ``--reinstall-package`` and ``--no-cache`` force a
genuine rebuild, since uv does not key its build cache on ``SKBUILD_CMAKE_ARGS``:

.. code-block:: bash

   SKBUILD_CMAKE_ARGS="-Dmonoprop_ENABLE_MPI=ON" \
       uv sync --all-extras --reinstall-package monoprop --no-cache
   mpiexec --allow-run-as-root -n 2 uv run --no-sync python -m pytest tests --with-mpi

C++ unit tests
--------------

The C++ tests run through CTest, against whichever preset you built. Without MPI:

.. code-block:: bash

   cmake --preset release-gcc
   cmake --build --preset release-gcc
   ctest --preset release-gcc

With MPI, use the ``-mpi`` preset, which also runs the multi-rank tests:

.. code-block:: bash

   cmake --preset release-gcc-mpi
   cmake --build --preset release-gcc-mpi
   ctest --preset release-gcc-mpi

The full MPI rank matrix (several rank counts) is driven by a helper script:

.. code-block:: bash

   ctest -S tools/ctest-mpi-matrix.cmake -VV

See also
========

- :doc:`/getting-started` — installing a prebuilt release from PyPI.
- :doc:`/features/parallelism` — running across MPI ranks and shared-memory threads.
- :doc:`/how-to-contribute` — the full test and documentation workflow.
