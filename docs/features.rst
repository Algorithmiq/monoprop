========
Features
========

This section summarises the capabilities of monoprop and explains how to use each
one. For the ideas behind them see :doc:`/concepts`; for the full constructors and
arguments see :doc:`/python-api`.

The pages below group the features by stage of a workflow:

- :doc:`/features/simulators` — the two simulators (``MajoranaPropagator`` and
  ``PauliPropagator``), the operator/gate/circuit layers they share, and how to
  construct and drive each one.
- :doc:`/features/initialisation` — constructing the simulator, choosing the
  Heisenberg or Schrödinger picture, and updating the Hamiltonian in place.
- :doc:`/features/cutoff` — controlling operator growth: the structural cutoff
  strategies and coefficient-tolerance filtering.
- :doc:`/features/evaluation` — replaying the propagated graph to evaluate
  expectation values and gradients, including paring.
- :doc:`/features/parallelism` — scaling across MPI ranks and shared-memory
  threads.

.. toctree::
   :hidden:
   :maxdepth: 1

   features/simulators
   features/initialisation
   features/cutoff
   features/evaluation
   features/parallelism
