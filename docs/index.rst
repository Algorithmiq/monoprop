======================
monoprop Documentation
======================

monoprop is a high-performance C++ library with Python bindings for Majorana and Pauli propagation that provides a backend for simulating and variationally optimising quantum circuits.
It provides support for large scale simulations through multithreading and multi-node support on HPC clusters through MPI and shared-memory parallelism.
The simulator backend represents quantum operators and states in the Majorana basis.
Two front-ends build on it: ``MajoranaPropagator`` for native Majorana and fermionic problems, and ``PauliPropagator`` for qubit (Pauli) problems.

New here? Install monoprop and run the minimal example in
:doc:`getting-started`, then read :doc:`concepts` to understand how propagation
works. The documentation is organised into four parts:

**Getting started** — install monoprop and get it running.

- :doc:`getting-started` — installation and minimal serial and distributed examples.
- :doc:`building` — building the Python bindings and the C++ library and executables, with and without MPI, and how to run them.

**User guide** — understand the method and drive a simulation.

- :doc:`concepts` — the Majorana basis, the propagation algorithm, truncation, and the simulation modes.
- :doc:`features` — the simulator's capabilities: configuration, cutoffs, expectation value and gradient evaluation, and parallel execution.
- :doc:`tutorials/index` — step-by-step notebooks for concrete problems and workflows.

**Reference** — look up details.

- :doc:`python-api` — full API reference for the Python bindings.
- :doc:`benchmarks` — performance benchmarks and comparisons to other libraries.
- :doc:`zreferences` — bibliography of the papers and methods behind monoprop.

**Contributing** — work on monoprop itself.

- :doc:`how-to-contribute` — contribution workflow, coding standards, and testing requirements.

.. toctree::
   :hidden:
   :caption: Getting started
   :maxdepth: 2

   getting-started
   building

.. toctree::
   :hidden:
   :caption: User guide
   :maxdepth: 2

   concepts
   features
   tutorials/index

.. toctree::
   :hidden:
   :caption: Reference
   :maxdepth: 2

   python-api
   benchmarks
   zreferences

.. toctree::
   :hidden:
   :caption: Contributing
   :maxdepth: 2

   how-to-contribute
