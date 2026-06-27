======================
monoprop Documentation
======================

monoprop is a high-performance C++ and Python library for Majorana and Pauli
propagation. It provides a branch-based simulator that evolves quantum
operators through a sequence of Hamiltonian terms, tracking the full tree
of Majorana branches and pruning it according to user-supplied truncation
thresholds. The core engine is written in C++ with Python bindings.
Distributed runs on HPC clusters are supported through MPI and shared-memory
parallelism.

The documentation is organized into the following sections:

- :doc:`getting-started` — installation instructions and minimal serial and distributed examples
- :doc:`concepts` — the underlying algorithms, operator representations, simulation modes, truncation strategies, and distributed execution model
- :doc:`features` — supported capabilities and usage examples, including cutoff configuration, expectation value and gradient evaluation, and parallel execution
- :doc:`tutorials/index` — step-by-step notebooks for specific tasks and workflows
- :doc:`benchmarks` — performance benchmarks and comparisons to other libraries
- :doc:`python-api` — full API reference for the Python bindings
- :doc:`how-to-contribute` — contribution workflow, coding standards, and testing requirements
- :doc:`zreferences` — bibliography of papers and resources related to the algorithms and techniques used in monoprop

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   concepts
   features
   tutorials/index
   benchmarks
   python-api
   how-to-contribute
   zreferences

.. toctree::
   :hidden:
   :caption: Internals
   :maxdepth: 1

   internals/runtime_architecture
   internals/data_structures
   internals/graph
   internals/evolution_path
   internals/graph_evolution
   internals/single_rank_evolution
   internals/mpi_and_threading
   internals/masked_execution_plan
