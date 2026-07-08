=================
How to Contribute
=================


Development setup
-----------------

The repository's `DevContainer <https://containers.dev/>`_ provides a ready-made development environment.

Alternatively, set up a local development build — prerequisites,
the editable ``uv`` build of the Python bindings (with or without MPI),
the standalone C++ build, and verifying the install — as described
in :doc:`building`.

The sections below cover the parts specific to contributing: running the tests and
building the documentation.

Running the tests
-----------------

Running the Python and C++ test suites --- with and without MPI --- is documented in
:doc:`building`.

Code quality checks
-------------------

The project uses automated code quality checks via pre-commit hooks.
To run all pre-commit checks manually:

.. code-block:: bash

   prek run --all
   # or for specific hook
   prek run --hook aislop

Documentation
-------------

To build the documentation locally:

.. code-block:: bash

   just build-docs

The rendered HTML index is written to ``build/docs/html/index.html``.
