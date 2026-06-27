========
Features
========

This page summarises the capabilities of monoprop and explains how to use each one.

.. contents:: On this page
   :local:
   :depth: 2

Simulator construction and configuration
=========================================

The top-level class is ``MonomialPropagator`` (Python) /
``MonomialPropagator<NumModes>`` (C++). Every significant aspect of the
simulation can be tuned at construction time and updated at runtime without
rebuilding.

Heisenberg and Schrödinger pictures
-------------------------------------

Pass ``schrodinger_cutoff`` to switch from the default Heisenberg picture
(operator evolution) to Schrödinger picture (state evolution). The two pictures
use different graph-traversal orders and sign conventions; see
:doc:`/concepts` for what this means algorithmically.

Cutoff type and value
----------------------

Two structural cutoff strategies are available:

``LengthPairingDistance`` (default)
   Truncates Majorana monomials whose string length plus pairing distance from
   the reference Slater determinant exceeds the cutoff. This is the
   physically motivated choice for quantum chemistry Hamiltonians.

``Mode``
   Truncates by the span of mode indices in the bitset. Useful when orbital
   ordering encodes spatial locality.

Both the cutoff value and the cutoff type can be changed at any point during a
simulation:

.. code-block:: python

   sim.update_cutoff(8)
   sim.update_cutoff_type("LengthPairingDistance")

Coefficient tolerance filtering
---------------------------------

Two absolute-tolerance thresholds prune terms by coefficient magnitude,
independently of the structural cutoff:

- ``lower_atol``: discard terms with ``|coeff| < lower_atol`` during gate
  application. Primary tool for suppressing negligible branches in long
  ADAPT loops.
- ``upper_atol``: discard terms with ``|coeff| > upper_atol``. Useful for
  numerical stability in special regimes.

.. code-block:: python

   sim.update_lower_atol(1e-9)
   sim.update_upper_atol(1e-3)

Hamiltonian updates
--------------------

The initial Hamiltonian can be patched after construction without rebuilding
the simulator or the graph:

.. code-block:: python

   sim.update_initial_operator(new_hamiltonian)

In Heisenberg picture this patches the coefficient arrays in place. In
Schrödinger picture it also maintains the gradient Hamiltonian required for
gradient calculations.

See :doc:`/internals/runtime_architecture` for the full C++ constructor signature and
the internal layout of the simulator object.

Expectation value evaluation
============================

After calling ``propagate``, the graph can be replayed at any
parameter vector without re-running Majorana algebra.
``expectation_value_functional`` returns a callable that accepts a parameter vector:

.. code-block:: python

   expval_fn = sim.expectation_value_functional(param_inds, gen_coeffs)
   expval = expval_fn(parameters)

Exact paring
------------

Passing a ``pare_threshold`` to ``expectation_value_functional`` activates *exact
paring*: monoprop builds a masked ``MPExecutionPlan`` over the stored graph
that skips edges whose contribution to the expectation value is provably below the
threshold. The original graph is not modified; unchanged layers share their
storage. This can dramatically reduce replay cost for sparse graphs:

.. code-block:: python

   pared_expval_fn = sim.expectation_value_functional(param_inds, gen_coeffs, pare_threshold=1e-7)

See :doc:`/internals/masked_execution_plan` for the masked-plan construction
algorithm.

Gradient estimation
====================

``expectation_value_and_gradient_functional`` computes both the expectation value and the full
parameter gradient in a single backward pass over the graph:

.. code-block:: python

   expval_grad_fn = sim.expectation_value_and_gradient_functional(
       param_inds, gen_coeffs, pare_threshold=1e-7
   )
   expval, gradient = expval_grad_fn(parameters)

The derivative pass walks each graph layer in reverse, accumulates per-layer
gradient contributions rank-locally, and finishes with one
``allreduce_sum_inplace`` over the gradient vector — a single MPI collective
regardless of circuit depth. The cross-rank derivative exchange derives a
2× layout from the cached evolution layout on-the-fly, so no extra layout
storage is needed per layer.

Both ``expectation_value_functional`` and ``expectation_value_and_gradient_functional`` accept the
same optional ``pare_threshold`` to enable exact paring during gradient
evaluation.

Distributed MPI runs
=====================

monoprop distributes the Majorana operator across MPI ranks using a deterministic
hash: ``rank(M) = hash(M) % num_ranks``. Any rank can compute the owner of any
term without communication, which allows gate compilation to resolve cross-rank
dependencies with a two-phase ``alltoallv`` and replay to use cached per-layer
exchange layouts.

Single-node (``MPI.COMM_SELF``)
---------------------------------

.. code-block:: python

   from mpi4py import MPI
   sim = MonomialPropagator(..., comm=MPI.COMM_SELF)

Multi-node (``MPI.COMM_WORLD``)
---------------------------------

Replace the communicator and launch with ``mpiexec``:

.. code-block:: python

   from mpi4py import MPI
   sim = MonomialPropagator(..., comm=MPI.COMM_WORLD)

.. code-block:: bash

   mpiexec -n 8 uv run python your_script.py

MPI is optional at build time. When compiled without
``monoprop_ENABLE_MPI``, all MPI types become stubs and the library runs
as a single-rank program with no MPI dependency.

The MPI communication pattern per operation is summarised in
:doc:`/internals/mpi_and_threading`. Each functional evaluation uses at most one
``allreduce`` collective for the expectation value scalar and one ``allreduce_sum_inplace``
for the gradient vector, regardless of circuit depth.

Shared-memory parallelism
==========================

Within each MPI rank, monoprop uses Intel oneTBB for shared-memory parallelism.
The main patterns are:

- ``parallel_for_indices`` over independent coefficient updates during gate
  replay
- ``parallel_reduce_indices`` for gradient accumulation over cosine and cycle
  index ranges

The thread count is read from the ``TBB_NUM_THREADS`` environment variable (or
TBB's hardware concurrency detection). The ``ShardedIndexMap`` is sized to one
shard per hardware thread to minimise contention during concurrent inserts.
