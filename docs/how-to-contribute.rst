=================
How to Contribute
=================

Install prerequisites
---------------------

- C++23 compiler (GCC 13+ or Clang 17+)
- Python 3.11 or newer
- ``uv`` package manager
- MPI implementation for multi-rank runs (optional)

This repository ships a `DevContainer <https://containers.dev/>`_ that provides
all of the above pre-configured. Opening the folder in Visual Studio Code and
rebuilding the container is the fastest way to reach a working environment.
Without a DevContainer, the prerequisites must be installed by hand before
proceeding.

Build from source
-----------------

Once the repository is cloned, run the following command from the root directory:

.. code-block:: bash

   uv sync --all-groups --all-extras -v

``uv`` creates a virtual environment, installs Python dependencies (including
``mpi4py``), and compiles the nanobind extension in editable mode with MPI
enabled. Re-run this command whenever the dependency graph or C++ sources
change.

To skip the MPI installation (for example on platforms without an MPI toolchain) add:

.. code-block:: bash

   --config-settings=cmake.define.monoprop_ENABLE_MPI=OFF

Verify installation
-------------------

To verify that the package is installed correctly, run the following command:

.. code-block:: bash

   uv run python -c "import monoprop as mp; print(mp.__version__)"


Python tests
------------

To run Python tests locally:

.. code-block:: bash

   uv run pytest -v

To run more tests using MPI:

.. code-block:: bash

   mpiexec --allow-run-as-root -n 2 uv run python -m pytest tests --with-mpi

C++ tests
---------

.. code-block:: bash

   cmake --preset release-gcc
   cmake --build --preset release-gcc --target unit_tests.x
   ctest --preset release-gcc -R unit_tests

The CTest matrix is driven by:

.. code-block:: bash

   ctest -S tools/ctest-mpi-matrix.cmake -VV

Documentation
-------------

To build the documentation locally:

.. code-block:: bash

   just build-docs

The rendered HTML index is written to ``build/docs/html/index.html``.
