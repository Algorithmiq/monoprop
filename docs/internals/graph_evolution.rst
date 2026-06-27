Graph Evolution
===============

.. admonition:: Developer reference

   This page documents internal implementation details for contributors modifying
   performance-sensitive runtime code.

This page explains how monoprop compiles a Majorana generator into graph layers and how
those layers replay during evaluation. For the public simulator surface, see the
:doc:`/features` page. For the object model and top-level call path
around these kernels, see :doc:`/internals/runtime_architecture`.

This page covers two complementary operations:

1. **Gate compilation** — how ``propagate_one`` translates a Majorana generator
   into a ``Layer`` stored in the graph.
2. **Layer replay** — how ``evolve_step`` / ``evolve_operator`` re-applies a stored
   layer to a coefficient vector with a concrete parameter value.

For the graph object model and the concrete storage layout of ``LayerStorage``,
``LayerTraversal``, ``MPGraphView``, and ``MPOExecutionPlan``, see :doc:`/internals/graph`.

Gate compilation: propagate_one
---------------------------------

Each call to ``propagate_one(gen, orbital_rot, [coeffs], [param])`` compiles one
unitary gate ``e^{i θ G}`` into the graph.

Step 1 — Majorana algebra (commutator scan)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each existing term ``op[i]`` in ``mp_op_.op``, the evolution planner measures
the overlap with the generator and uses ``should_process_anticommuting(...)`` to
decide whether that term contributes a sine branch.

The transformed support is:

.. code-block::

   result_maj = op[i] XOR gen      (symmetric difference of bit sets)

If the source term commutes with the generator, the layer receives only a cosine
entry for that source index. If it anticommutes, the sine branch uses ``result_maj``
as the target support and computes the multiplicative sign with
``get_multiplicative_phase(...)``, which combines the interleave sign and the
Hermitian-basis correction described in :doc:`/concepts/algebra`.

The cutoff function is applied to ``result_maj``. Terms that violate the cutoff
(string too long, modes out of range, etc.) are discarded. Surviving new terms
become **half-cycles** if their target rank differs from the local rank, or **full
cycles** if both ``op[i]`` and ``result_maj`` live on the same rank.

The single-rank planner adds one more layer of filtering here: lower/upper
tolerances can suppress small sine branches before lookup and can keep large cosine
branches alive even if the structural cutoff would reject the target. That flow is
spelled out in :doc:`/internals/single_rank_evolution`.

Step 2 — Half-cycle resolution via update_mp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the result ``result_maj`` hashes to a remote rank, the calling rank cannot
insert it locally. It records:

.. code-block::

   half_ham[target_rank].push_back(result_maj)
   half_cycles[target_rank].push_back({src_idx, placeholder})
   half_phases[target_rank].push_back(phase)

``update_mp`` then runs the two-phase MPI protocol (see
:doc:`/internals/mpi_and_threading`) to:

1. Send ``result_maj`` bitset words to the target rank.
2. Receive back the assigned coefficient index.
3. Complete the cycle entry:
   ``cycles[target_rank].push_back({src_idx, resolved_tgt_idx})``.

After ``update_mp``, all cycles are complete (each has a valid source and target
index).

Step 3 — Cycle classification (split_and_exchange_cycles)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cycles are classified into:

- **LocalCycle** — both ``src`` and ``tgt`` indices live on this rank.
- **CrossRankCycles** — one index is remote.

  - *Outgoing* (``out_indices``): we own ``src``; the remote rank owns ``tgt``.
  - *Incoming* (``in_indices``): we own ``tgt``; the remote rank owns ``src``.

This classification is stored in ``SplitCycleResult`` which is then passed directly
to ``graph_.append(...)``.

Step 4 — Cosine index filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every local coefficient index that is not updated through a retained local cycle or
an incoming cross-rank cycle needs a standalone cosine scaling during replay. These
indices are collected into ``cos_inds``. In the MPI case, ``append_to_graph(...)``
removes incoming cross-rank targets before the layer is stored, because those
indices are updated by the cross-rank exchange itself.

.. code-block::

   cos_inds = all_local_indices - {cycle targets} - {incoming cross-rank targets}

Before the layer is stored, ``cos_inds`` is compressed into
``CompressedCosineData``: a list of contiguous ``CosineSpan`` runs plus the total
logical cosine count. Replay kernels iterate spans directly instead of walking a
raw ``VecZ`` of indices.

Step 5 — Append layer to graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   graph_.append(compressed_cos_data, local_cycles, cross_rank_per_remote_rank);

Appending a layer stores everything in shared ``LayerStorage`` and caches the flat
evolution exchange layout once. The derivative path derives its 2x-scaled layout on
demand from that stored evolution layout. In **Heisenberg** picture this appends to
the back of the layer vector. In **Schrödinger** picture it inserts at
``front_offset_`` so new layers appear at the logical front. The concrete packed
storage format is described in :doc:`/internals/graph`.

Layer replay: evolve_step
--------------------------

``evolve_step(op, graph, param, layer_idx, comm)`` applies one layer to the
coefficient vector ``op`` at angle ``θ = param``.

Trig precomputation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   double cos_val = cos(2 * param);
   double sin_val = sin(2 * param);

The factor of 2 comes from the convention ``U(θ) = e^{iθG}`` where ``G² = I``,
giving expectation value ``f(θ) = A cos(2θ) + B sin(2θ) + C``.

Cosine pass (local, parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   for (const auto& span : layer.cos_spans()) {
     for (size_t idx = span.start; idx < span.start + span.count; ++idx) {
       op[idx] *= cos_val;
     }
   }

Every term that commutes with the generator gets scaled by ``cos(2θ)``. The runtime
picks a span-grained or coefficient-grained TBB traversal depending on the span
shape, but the logical effect is the same.

Local cycle pass (local, parallel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   parallel_for_indices(local_cycles.size(), [&](size_t i) {
       const auto& cyc = local_cycles[i];
       double p = sin_val * cyc.phase;    // phase ∈ {-1, +1}
       double o1 = op[cyc.src];
       double o2 = op[cyc.tgt];
       op[cyc.src] = o1 * cos_val - p * o2;
       op[cyc.tgt] = o2 * cos_val + p * o1;
   });

This is the symplectic rotation for a pair of anticommuting terms:

.. code-block::

   |src'⟩ = cos(2θ)·|src⟩ - sin(2θ)·phase·|tgt⟩
   |tgt'⟩ = cos(2θ)·|tgt⟩ + sin(2θ)·phase·|src⟩

Cross-rank synchronisation (one alltoallv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``synchronize_cross_rank_operator`` handles all remote dependencies in a single MPI
collective using the cached ``LayerExchangeLayout`` for that layer:

.. code-block::

   1. Read counts/displacements from layer.evolution_exchange_layout().
   2. Pack a flat send buffer of size layout.total_count:
     For rank r, the buffer contains the outgoing src values followed by the
     incoming tgt values required by the remote rank.

   3. MPI_Alltoallv over that flat payload   ← one collective call

   4. Apply received values:
        Outgoing entries (we own src, remote owns tgt):
          op[out_indices[r][i]] = cos·op[i] - sin·phase·recv[r][in_size + i]
                                               ↑ remote's tgt value
        Incoming entries (we own tgt, remote owns src):
          op[in_indices[r][i]] = cos·op[i] + sin·phase·recv[r][i]
                                              ↑ remote's src value

After step 4, all coefficient indices are in the correct post-rotation state. No
barrier is needed; ``alltoallv`` is already synchronising.

evolve_operator: full graph replay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   auto evolve_operator(const VecD& coeffs, const GraphType& graph,
                        const VecD& params, MPI_Comm comm) -> VecD {
       auto evolved = coeffs;
       for (size_t i = 0; i < graph.layers(); ++i) {
           evolve_step(evolved, graph, params[i], i, comm);
       }
       return evolved;
   }

``GraphType`` can be `MPGraph``, ``MPGraphView``, or ``MPExecutionPlan``.
``params`` is the vector of mapped parameter values (one per layer), pre-computed in
logical replay order. The loop applies layers in logical order regardless of
picture.

Exact paring during functional evaluation
-----------------------------------------

When ``expectation_value_functional(..., pare_threshold)`` or
``expectation_value_and_gradient_functional(..., pare_threshold)`` is called with a threshold,
monoprop builds an ``MPExecutionPlan`` instead of copying a second pared graph.

- seed collection stays rank-local
- each layer exchanges only the keep information needed to preserve exact support
  across ranks
- untouched layers reuse their original ``LayerStorage``
- filtered layers overlay only the masked replay data that differs from the stored
  layer; :doc:`/internals/graph` covers that representation

Replay, derivative accumulation, and MPI exchange then run directly against that
execution plan through the same ``get_layer(i)`` interface used by the full graph.

Derivative pass: state_hamiltonian_derivative
----------------------------------------------

The gradient ``∂E/∂θ_k`` is computed analytically, not by finite differences or a
parameter-shift wrapper. The simulator walks both ``state`` and ``hamiltonian``
vectors through a single layer and accumulates the exact chain-rule contribution.

TrigValues
~~~~~~~~~~~

.. code-block:: cpp

   struct TrigValues {
       double cos_val;      // cos(2g·θ)
       double sin_val;      // sin(2g·θ)
       double sec_val;      // 1/cos(2g·θ)   (for inverse cosine scaling)
       double der_cos_val;  // -2g·sin(2g·θ)
       double der_sin_val;  //  2g·cos(2g·θ)
   };

``g = gen_coeff`` is the coefficient from ``gen_coeffs``. The factor of 2 comes from
the doubled-angle convention.

Cosine derivative accumulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   for (const auto& span : layer.cos_spans()) {
     for (size_t idx = span.start; idx < span.start + span.count; ++idx) {
       hamiltonian[idx] *= sec_val;  // undo the cos scaling (H -> H/cos)
       local += state[idx] * hamiltonian[idx];
       state[idx] *= cos_val;        // re-apply cos to state
     }
   }

Each cos-index contributes ``state[i] * (hamiltonian[i] / cos) = state[i] *
hamiltonian[i] * sec`` to the ``cos_contrib`` partial sum. The ``hamiltonian`` is
updated in place so subsequent cycle accumulations use the partially-differentiated
value.

Cycle derivative accumulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each local cycle ``(src, tgt, phase)``:

.. code-block:: cpp

   double ps = sin_val * phase;
   double s0 = state[src],  s1 = state[tgt];
   double h0 = ham[src],    h1 = ham[tgt];
   double nh0 = h0 * cos_val + ps * h1;
   double nh1 = h1 * cos_val - ps * h0;
   ham[src] = nh0;   ham[tgt] = nh1;
   cos_contrib += nh0 * s0 + nh1 * s1;
   sin_contrib += (nh0 * s1 - nh1 * s0) * phase;
   state[src] = s0 * cos_val + ps * s1;
   state[tgt] = s1 * cos_val - ps * s0;

Both ``hamiltonian`` and ``state`` receive the same cycle rotation, and their
bilinear product contributes to the gradient.

Cross-rank derivative exchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same structure as the evolution exchange but packs ``[state[i], ham[i]]`` pairs:

.. code-block::

   send_vals[r] has 2*(out_size + in_size) elements:
     Outgoing pairs: [state[out_i], ham[out_i]] for each out_i
     Incoming pairs: [state[in_i],  ham[in_i]]  for each in_i

After the ``alltoallv``, the received values complete the cycle updates for
cross-rank pairs, accumulating into ``(cos_contrib, sin_contrib)``.

Final gradient value
~~~~~~~~~~~~~~~~~~~~~

The rank-local contribution returned by
``state_hamiltonian_derivative_local`` is:

.. code-block:: cpp

   double local_grad = (cos_contrib + cyc_cos_contrib) * trig.der_cos_val
                     + cyc_sin_contrib                 * trig.der_sin_val;

``state_hamiltonian_derivative`` wraps that in an immediate ``allreduce_sum`` for
the scalar layer-derivative API. The hot ``expectation_value_and_gradient_functional`` path
calls ``state_hamiltonian_derivative_local`` for each layer, accumulates into the
parameter-indexed gradient vector locally, and finishes with one
``allreduce_sum_inplace(gradient, comm)``.

contract_partially: inplace vs non-inplace
--------------------------------------------

.. code-block:: cpp

   auto evolved = sim.contract_partially(params, param_map, gen_coeffs, inplace);

Both paths call ``evolve_operator``. The difference is how the source graph is
obtained:

.. list-table::
   :header-rows: 1

   * - ``inplace``
     - Graph source
     - Effect on graph
     - Effect on internal coeffs
   * - ``true``
     - ``graph_.slice_graph(k, contract=true)``
     - Removes replayed layers from graph
     - Updates ``ham_coeffs``/``state_coeffs``
   * - ``false``
     - ``graph_.slice_view(k)``
     - Graph unchanged (shared-ptr view)
     - No internal update

``slice_view`` produces a ``MPGraphView`` — a zero-copy window over the layer
vector. ``evolve_operator`` accepts both ``MPOperator`` and ``MPGraphView`` via the
templated ``evolve_operator_impl``.

Functional exact paring is separate from this slicing step: it keeps the graph
immutable and builds an ``MPExecutionPlan`` only when ``pare_threshold`` is
enabled.

Benchmark reporting
-------------------

The example benchmark reports only the top-level wall-clock phases it executes,
such as initial evolution, functional setup, and repeated evaluation time.
Fine-grained internal evolution timing hooks are not part of the current runtime
path.
