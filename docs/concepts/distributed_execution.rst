Distributed Execution
======================

monoprop scales to distributed-memory HPC clusters through MPI and uses Intel
oneTBB for shared-memory parallelism within each rank. This page describes the
distribution model, the communication patterns, and parallelism configuration.

Operator partitioning
----------------------

The Majorana operator is partitioned at the monomial level. Each monomial
:math:`\hat{M}_\alpha` is permanently assigned to exactly one MPI rank by a
deterministic hash:

.. math::

   \mathrm{rank}(M_\alpha) = \mathrm{hash}(M_\alpha) \bmod P,

where :math:`P` is the number of MPI ranks. The hash is stateless and
deterministic: any rank can compute the owner of any monomial without
communication. No rank table is ever broadcast or stored.

Gate compilation
-----------------

When a gate is compiled, each rank runs the Majorana algebra locally. New
monomials owned by the *same* rank are inserted directly. Monomials owned by a
*different* rank create a **half-cycle** and are resolved by a two-round
exchange: first the new monomial is sent to its target rank, then the assigned
index is returned. After this exchange every cycle entry has a valid
``(source, target)`` index pair and cycle classification proceeds rank-locally.
A communication layout is cached in the compiled layer and reused at every
subsequent replay call.

Gate replay
-----------

Replaying a stored layer at a concrete parameter :math:`\theta` requires one
MPI exchange per layer: each rank sends the coefficient values needed by remote
cycle partners and applies the received values to complete its local rotations.
This exchange uses the pre-built layout from compilation, so no MPI setup is
needed at replay time.

Expectation value and gradient collectives
------------------------------------------

Beyond the per-layer cross-rank exchanges, the evaluation kernels use two global
reductions:

- **Expectation value**: one collective sum of rank-local inner-product contributions
  after the full graph has been replayed.
- **Gradient**: rank-local layer derivatives are accumulated throughout the
  backward pass and reduced in a single collective sum at the end — no global
  communication inside the layer loop.

For the complete communication cost table broken down by operation, see
:doc:`/internals/mpi_and_threading`.

Shared-memory parallelism
--------------------------

Within each MPI rank, monoprop uses Intel oneTBB for shared-memory parallelism.
MPI calls are always issued from the main thread after TBB tasks complete.
The main parallel patterns are:

- Independent coefficient updates during cosine scaling and commutator scans.
- Reduction over cycle index ranges for gradient accumulation.
- Per-thread partial maps during commutator algebra, merged after the parallel
  loop.

The thread count is controlled by the ``TBB_NUM_THREADS`` environment variable,
falling back to hardware-concurrency detection.

Single-rank and no-MPI configurations
---------------------------------------

For serial development and testing, pass ``comm=MPI.COMM_SELF`` to restrict the
simulator to a single rank with no inter-rank communication:

.. code-block:: python

   from mpi4py import MPI
   sim = MonomialPropagator(..., comm=MPI.COMM_SELF)

When compiled without the ``monoprop_ENABLE_MPI`` flag, all MPI types and functions
are replaced by no-op stubs. The library then requires no MPI installation and
runs as a pure single-rank program, with the core logic unchanged.

Scaling considerations
-----------------------

The hash-based partitioning distributes operator terms uniformly across ranks
in expectation for chemistry-scale operators. Compilation cost grows with the
number of new cross-rank terms; replay cost grows with the number of cross-rank
cycle pairs per layer. Both are bounded by the total graph size, so the
per-evaluation communication cost is independent of the number of functional
evaluations once the graph is compiled.
