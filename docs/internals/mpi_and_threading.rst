MPI and Threading
=================

.. admonition:: Developer reference

   This page documents internal implementation details for contributors modifying
   performance-sensitive runtime code.

MPI initialisation
------------------

``mpi::init()`` must be called once before any simulator work. It calls
``MPI_Init_thread`` requesting ``MPI_THREAD_FUNNELED`` — only the main thread may
call MPI functions. TBB tasks never call MPI directly; they prepare data, and then
the main thread issues the collective.

.. code-block:: cpp

   monoprop::mpi::init(&argc, &argv);
   // ... run simulations ...
   monoprop::mpi::finalize();

When compiled without ``-Dmonoprop_ENABLE_MPI``, ``MPICompat.h`` provides stub types
(``MPI_Comm = int``, ``MPI_COMM_WORLD = 0``) and no-op implementations of every MPI
function, so the entire library compiles and runs as a single-rank program without
an MPI installation.

Rank assignment for Majorana terms
----------------------------------

Every ``MajoranaSet`` is permanently assigned to exactly one rank by:

.. code-block:: cpp

   size_t find_rank(const MajoranaSet<N>& maj, size_t n_ranks) {
       return MPHash<N>{}(maj) % n_ranks;
   }

``MPHash`` uses ``SplitmixHash`` (64-bit avalanche mix over each ``uint64_t``
word). Because the hash is deterministic and stateless, any rank can compute the
owner of any term without communication. This property is used pervasively — for
example, during Majorana algebra the calling rank only retains the result terms
that hash to its own rank number.

MPI collective wrappers
-----------------------

All collectives are thin wrappers in ``mpi::`` namespace (``MPIUtils.h``).

alltoallv
~~~~~~~~~

The generic wrappers in ``MPIUtils.h`` use ``alltoallv_into`` with contiguous
scratch buffers so nested per-rank payloads do not allocate on every call:

.. code-block::

   alltoallv_into(send_data, recv_data, comm)
     1. Count send sizes → MPI_Alltoall for recv sizes
     2. Pack send_data into contiguous send_buffer (thread-local)
     3. MPI_Alltoallv: full personalized exchange
     4. Unpack recv_buffer into recv_data per source rank

Three typed overloads exist for ``VecD``, ``VecI``, and ``VecZ``
(``double``, ``int``, ``uint64_t``).

The replay and derivative hot paths go further than this generic wrapper. Each
full ``Layer`` caches one ``LayerExchangeLayout`` for evolution replay.
``Evolution.cpp`` derives the derivative layout from that cached evolution layout
in thread-local scratch space, so there is no second persistent derivative layout
in ``LayerStorage``. When exact paring filters cross-rank edges,
``LayerExecutionPlan`` builds a reduced evolution layout for that filtered view
only. This avoids rebuilding nested ``vector<VecD>`` payloads inside every layer
update.

allgatherv / allreduce
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Call
     - Use case
   * - ``allgatherv(VecZ, comm)``
     - Gather Majorana word buffers (operator exchange)
   * - ``allreduce_sum(double, comm)``
     - Reduce scalar expectation value contributions to a global sum
   * - ``allreduce_sum_inplace(VecD, comm)``
     - Reduce gradient vectors in place

Bitset serialisation for MPI
------------------------------

``MajoranaSet<N>`` is a ``Bitset<2N>`` backed by ``ceil(2N/64)`` contiguous
``uint64_t`` words. Because it is trivially copyable, MPI transfers use ``memcpy``
directly into/out of the word array:

.. code-block:: cpp

   // Pack a MajoranaSet into a VecZ buffer (one word per element)
   mpi_detail::append_majorana_words<N>(maj, buffer);

   // Unpack from a word buffer at offset `start`
   auto maj = mpi_detail::read_majorana_from_words<N>(buffer, start);

The MPI datatype used is ``MPI_UNSIGNED_LONG_LONG`` (same width as ``uint64_t``).
``kWords<N>`` holds the word count as a compile-time constant so the serialisation
loops are fully unrollable for small ``N``.

update_mp: the operator expansion protocol
------------------------------------------

When ``propagate_one`` encounters a Majorana product ``gen ^ op[i]`` whose result
is not yet on the calling rank, it produces a "half-cycle" — a pending
(source, new_term) pair where ``new_term`` must be inserted on its target rank.

``update_mp`` resolves all half-cycles across all ranks in a **two-phase alltoallv
protocol**:

.. code-block::

   Phase 1 — send new terms to their target ranks
     Each rank packs half_ham[target_rank] as uint64_t words
     → mpi::alltoallv(send_ham_data, comm)

   Phase 2 — insert received terms, send back indices
     Each rank inserts received MajoranaSets into its local op/indexing
     (with duplicate detection — terms may already exist)
     Packs the assigned indices into response_new_indices[source_rank]
     → mpi::alltoallv(response_new_indices, comm)

   Phase 3 — complete half-cycles with resolved target indices
     For each remote rank tr:
       For each half-cycle i in half_cycles[tr]:
         cycles[tr].push_back({src_idx, recv_new_indices[tr][i] + rank_offset})

   Phase 4 — handle local half-cycles (target_rank == my_rank) inline
     Same duplicate-safe insert; no MPI needed.

     mpi::barrier(comm)
     → return empty SplitCycleResult
     (caller invokes split_and_exchange_cycles to build CrossRankCycles)

For a single rank (``num_ranks == 1``), the function inserts local half-ham terms
directly and returns without any MPI calls.

The rank offset (``all_op1_sizes[tr]``) allows candidate evolutions to maintain
separate coefficient index spaces for the base operator and the candidate extension.

Masked-plan builder exchanges
------------------------------

``build_masked_execution_plan(...)`` exchanges keep flags rather than coefficients.
The builder uses two small ``MPI_Alltoallv`` phases per layer when remote cross-rank
edges are present:

1. **Source-keep exchange**: ``pack_source_keep_flags(...)`` sends one ``0/1`` flag
   per outgoing source index to the remote rank that owns the corresponding
   incoming target.
2. **Selection exchange**: after local filtering, the target-owning rank sends back
   the newly selected source positions that must now be kept.

If that second exchange activates additional local sources, the layer is refiltered
once locally. The builder therefore avoids a global fixpoint loop while still
preserving exact support across ranks.

Evolution layer replay: synchronize_cross_rank_operator
---------------------------------------------------------

During ``evolve_step`` (replaying a single layer at parameter ``θ``), after local
updates are applied, cross-rank coefficient dependencies are resolved via one
``alltoallv`` using the cached layer layout:

.. code-block::

   layout = layer.evolution_exchange_layout()
   resize flat send/recv buffers to layout.total_count

   pack_cross_rank_evolution_payload(op, layer, my_rank, layout, send_buffer)
     For each remote rank r, write into the rank's flat slice:
       Outgoing (we own src): send_buffer[...] = op[out_indices[*]]
       Incoming (we own tgt): send_buffer[...] = op[in_indices[*]]

   → flat recv_buffer from MPI_Alltoallv           ← ONE collective call

   apply_cross_rank_evolution_exchange(op, layer, layout, cos_val, sin_val,
                                       recv_buffer, my_rank)
     Outgoing updates (we own src, remote owns tgt):
       op[out_indices[i]] = cos·op[i] - sin·phase·recv[r][in_size + i]
     Incoming updates (we own tgt, remote owns src):
       op[in_indices[i]] = cos·op[i] + sin·phase·recv[r][i]

The update formulas implement the symplectic cycle transformation:

.. code-block::

     new[src] =  cos·old[src] - sin·phase·old[tgt]
     new[tgt] =  cos·old[tgt] + sin·phase·old[src]

where ``src`` and ``tgt`` are on different ranks and ``phase ∈ {-1, +1}``.

Derivative pass: accumulate_cross_rank_derivatives
----------------------------------------------------

The gradient computation simultaneously walks both ``state`` and ``hamiltonian``
vectors through each layer. The cross-rank derivative exchange derives a 2×  layout
from the stored evolution layout (on-the-fly, no extra storage) and packs **both**
vectors into one flat payload:

.. code-block::

   layout = acquire_derivative_layout(layer.evolution_exchange_layout())
            // 2× scale, thread-local
   pack_cross_rank_derivative_payload(state, ham, layer, my_rank, layout, send_buffer)
     For each remote rank r, the flat payload stores 2*(out_size + in_size) values:
       Outgoing: [state[out_i], ham[out_i]] pairs
       Incoming: [state[in_i],  ham[in_i]] pairs

After the ``alltoallv``, ``apply_cross_rank_derivative_exchange`` applies the
coupled update to both vectors and accumulates the partial gradient contribution
``(cos_contrib, sin_contrib)``.

``state_hamiltonian_derivative`` can still reduce one scalar layer derivative
immediately, but the optimized ``expectation_value_and_gradient_functional`` path keeps those
per-layer contributions local and finishes with one vector reduction:

.. code-block::

   gradient[param_ind] += local_layer_derivative
   ...
   allreduce_sum_inplace(gradient, comm)

where each local layer derivative uses ``der_cos_val = -2g·sin(2gθ)`` and
``der_sin_val = 2g·cos(2gθ)`` from ``TrigValues``.

TBB threading
-------------

Thread-level parallelism uses Intel TBB; MPI calls are always made from the main
thread after TBB tasks complete.

Parallel patterns used
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Pattern
     - Where
     - Description
   * - ``parallel_for_indices(n, fn)``
     - ``evolve_step``
     - Independent index updates; grain size 256
   * - ``parallel_reduce_indices(n, init, fn, combine)``
     - ``accumulate_cosine_derivative``, ``accumulate_cycle_derivative``
     - Reduction over independent index ranges; accepts an explicit grain size on
       hot paths
   * - ``parallel_for_cross_rank<Dir>(cross_rank, ...)``
     - Pack/apply cross-rank payloads
     - Iterates outgoing or incoming cycle entries per rank
   * - ``parallel_reduce_cross_rank<Dir>(...)``
     - Derivative exchange
     - Reduce over outgoing/incoming cycle entries
   * - ``tbb::enumerable_thread_specific``
     - ``MPOperator::get_operator``
     - Per-thread accumulators merged after parallel scan
   * - ``tbb::combinable<MajoranaOperator>``
     - Commutator algebra
     - Per-thread partial maps, merged after parallel loop

Thread count
~~~~~~~~~~~~

``threading::effective_parallelism()`` returns TBB's configured thread count
(respects ``TBB_NUM_THREADS``). It is used to size ``ShardedIndexMap`` shards so
there is roughly one shard per hardware thread, minimising contention during
shard-local updates.

Communication summary per operation
-----------------------------------

.. list-table::
   :header-rows: 1

   * - Operation
     - MPI calls
     - What is exchanged
   * - Constructor / ``propagate_one``
     - 2× ``alltoallv`` per new term
     - half-ham words → target rank; assigned indices → source rank
   * - ``evolve_step`` (one layer replay)
     - 1× ``alltoallv``
     - coefficient values for cross-rank cycle sources and targets
   * - ``state_hamiltonian_derivative`` (one layer)
     - 1× ``alltoallv`` + 1× ``allreduce_sum``
     - state+ham values for cross-rank cycles; scalar gradient sum
   * - ``expectation_value_functional`` evaluation
     - 1× ``allreduce_sum`` per call
     - scalar expectation value
   * - ``expectation_value_and_gradient_functional`` evaluation
     - replay/derivative layer exchanges + 1× ``allreduce_sum``
       + 1× ``allreduce_sum_inplace``
     - per-layer cross-rank replay, per-layer cross-rank derivative exchange,
       scalar expectation value sum, final gradient vector sum
