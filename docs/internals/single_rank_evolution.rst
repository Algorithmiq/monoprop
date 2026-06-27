Single-Rank Evolution Path
==========================

.. admonition:: Developer reference

   This page documents internal implementation details for contributors modifying
   performance-sensitive runtime code.

This page describes the exact control flow used when
``MonomialPropagator::propagate_one(...)`` runs with a single MPI rank.

The relevant call chain is:

.. code-block:: cpp

   MonomialPropagator::propagate_one
     -> evolve_maj
       -> evolve_maj_single_rank   // num_ranks == 1 fast path
     -> update_mp
     -> split_and_exchange_cycles
     -> graph_.append

Even in the single-rank case, the code still uses the same high-level pipeline as
the MPI path. The difference is that all lookup resolution and basis growth happen
locally, so there is no inter-rank lookup exchange and no remote cycle traffic.

Main data structures
--------------------

- ``mp_op_.op``: the local vector of Majorana basis terms.
- ``mp_op_.indexing``: a ``ShardedIndexMap`` mapping ``MajoranaSet`` to its local
  coefficient index.
- ``EvolveMajResult``: the temporary planner output for one generator application.
- ``MPGraph::Layer``: the compiled graph layer later replayed by ``evolve_step``.

The planner partitions each source term into one of three buckets:

- cosine-only: the source coefficient is multiplied by ``cos(2 * theta)`` during
  replay
- cycle: the source term rotates with an already-existing target term
- half-term: the transformed target does not exist yet and must be inserted into
  the basis first

Step 1: Entry from propagate_one
------------------------------------

``MonomialPropagator::propagate_one`` converts the incoming generator index
list to a ``MajoranaSet`` bitset and calls:

.. code-block:: cpp

   auto evolve_result = evolve_maj<NumModes>(...);

``evolve_maj`` computes ``num_ranks = mpi::size(comm)`` and forwards to
``evolve_maj_single_rank``. When ``num_ranks == 1``, the single-rank fast path runs
immediately.

Step 2: Build cutoff context
------------------------------

At the top of ``evolve_maj_single_rank``, the code derives the values that control
truncation and the trigonometric weights used later:

.. code-block:: cpp

   sin_val = param.has_value() ? std::sin(2 * param.value()) : 1.0;
   cos_val = param.has_value() ? std::cos(2 * param.value()) : 1.0;

From ``atol``, ``upper_atol``, ``param``, and the optional coefficient vector, it
builds a ``CutoffContext`` with two important predicates:

- ``is_above_upper(abs_coeff)``: keep a term alive even if the structural cutoff
  rejects it, because the cosine contribution is still large enough
- ``is_below_sin(abs_coeff)``: treat a lookup miss as cosine-only because the sine
  branch would be too small to keep

These checks are only active when the caller supplied both coefficients and the
corresponding tolerance.

Step 3: Parallel scan over the existing basis
----------------------------------------------

The hot loop scans every local term ``maj = ham[i]`` in parallel with TBB. Each
worker writes to its own ``LocalScanResult`` so there is no contention inside the
scan.

For each source term, the logic is:

3.1 Anticommutation test
~~~~~~~~~~~~~~~~~~~~~~~~~

The code computes:

.. code-block:: cpp

   maj_pop = maj.count();
   overlap = maj.count_and(gen_maj);

Then it calls
``should_process_anticommuting(maj_pop, gen_maj_pop, overlap, only_rotate_len_k)``.

This does two things:

- if ``only_rotate_len_k > 0`` and the source term length is greater than ``k``,
  skip it immediately
- otherwise call ``majs_anticommute(...)``

If this returns false, the term commutes with the generator and does not create any
work in this layer.

3.2 Form the transformed Majorana term
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For anticommuting terms, the transformed support is:

.. code-block:: cpp

   new_maj = maj ^ gen_maj;

This is the Majorana string that would appear in the sine branch of the rotation.

3.3 Structural cutoff before lookup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code next evaluates the cutoff function on ``new_maj``.

If both of these are true:

- ``!cutoff_ctx.is_above_upper(abs_coeff)``
- ``!cutoff_eval(new_maj)``

then the term is classified as cosine-only and the scan does not query the hash map
at all.

This is the cheapest early-exit path in the loop.

3.4 Local lookup in the current basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the transformed term survives the structural cutoff stage, the code probes the
local index map:

.. code-block:: cpp

   const auto found_it = indexing.find(new_maj);

There are two cases.

**Lookup miss**

If ``new_maj`` is not already in the basis:

- if ``is_below_sin(abs_coeff)`` is true, the sine contribution is dropped and the
  source term becomes cosine-only
- otherwise the code computes the Majorana multiplication sign with
  ``get_multiplicative_phase(...)`` and emits a half-term

The half-term is stored as:

- ``half_ham.push_back(new_maj)``
- ``half_phases.push_back(phase)``
- ``half_cycles.push_back({source_index, 0})``

The zero is a placeholder target index that will be resolved later by
``update_mp``.

**Lookup hit**

If ``new_maj`` already exists in the basis, the code decides whether this source
should emit the cycle or whether the partner term will emit it instead.

That symmetry break is handled by ``asymmetric_bitset_compare(new_maj, maj)``. It
compares the first differing bit between the two bitsets and returns true for
exactly one orientation of the pair.

Only the winning side emits a cycle:

.. code-block:: cpp

   phases.push_back(phase);
   cycles.push_back({source_index, target_index});

This avoids storing the same two-term rotation twice.

Step 4: Worker-local cosine storage
-------------------------------------

Cosine-only terms are stored in one of two formats.

Without ``upper_atol``
~~~~~~~~~~~~~~~~~~~~~~~

The worker directly appends indices into a ``CompressedCosineData`` builder using
``append_cosine_index(...)`` and a thread-local pending run. This keeps cosine data
compressed as contiguous spans while scanning.

With ``upper_atol``
~~~~~~~~~~~~~~~~~~~~

The worker stores raw ``cos_inds`` instead.

That slower representation is used because ``evolve_maj`` later removes any indices
that also appear as cycle targets. If a target survives as part of a cycle, it must
not also be scaled again as a standalone cosine index.

Step 5: Merge thread-local scan results
-----------------------------------------

After the TBB scan finishes, the function gathers all ``LocalScanResult`` objects
and computes total sizes for:

- ``cycles``
- ``half_cycles``
- cosine storage

It then merges worker-local buffers into one ``EvolveMajResult``.

Two details matter here:

- cycles and half-terms are copied into double-buffered persistent vectors so
  capacity survives across calls
- compressed cosine data is concatenated blockwise when ``upper_atol`` is off;
  otherwise raw ``cos_inds`` are merged directly

At this point, the planner output contains complete local cycle candidates,
unresolved half-terms, and cosine indices for this one generator.

Step 6: Single-rank cleanup in evolve_maj
-------------------------------------------

Control returns to ``evolve_maj``.

If ``upper_atol`` is active, ``evolve_maj`` gathers the target indices of the cycles
retained on this rank and removes those targets from ``cos_inds``.

The reason is simple: a cycle replay already applies the cosine factor to both
endpoints, so a retained target must not also be present in the standalone cosine
pass.

When this cleanup runs, ``compressed_cos_data`` is cleared and the layer will be
appended from raw ``cos_inds`` instead.

Step 7: Resolve half-terms with update_mp
-----------------------------------------

Back in ``MonomialPropagator::propagate_one``, the next call is:

.. code-block:: cpp

   update_mp(mp_op_, evolve_result.half_ham, ...);

For a single rank, ``update_mp`` does not perform any MPI communication. It just
walks the local half-term list and, for each ``new_maj``:

- probe ``mp_op_.indexing``
- if the term is absent, append it to ``mp_op_.op`` and insert it into the index
  map
- obtain the resolved target index
- convert the stored half-cycle ``(src, 0)`` into a real cycle ``(src, new_idx)``

After this step, every surviving sine contribution refers to a valid local basis
index.

Step 8: Pack cycles for graph storage
-------------------------------------

The next call is:

.. code-block:: cpp

   split = split_and_exchange_cycles(evolve_result.cycles, evolve_result.phases,
                                     comm_);

In the single-rank case, ``split_and_exchange_cycles`` is trivial:

- every cycle becomes a ``LocalCycle {src, tgt, phase}``
- ``cross_rank`` stays empty
- no MPI send or receive buffers are built

This preserves the same downstream graph API as the multi-rank path.

Step 9: Append the layer to the graph
---------------------------------------

Finally ``append_to_graph`` stores the compiled layer in ``graph_``.

There are two cases:

- if ``compressed_cos_data`` is present, call
  ``graph.append(compressed_cos_data, local_cycles, cross_rank)``
- otherwise call ``graph.append(cos_inds, local_cycles, cross_rank)`` and let
  ``Layer`` compress the cosine indices when it builds storage

So the single-rank planner always produces the same logical layer contents:

- cosine indices or cosine spans
- local cycles
- no cross-rank traffic

Replay semantics
----------------

Later, ``evolve_step`` replays that stored layer at a concrete parameter value.

The runtime meaning of the planner buckets is:

Cosine-only bucket
~~~~~~~~~~~~~~~~~~~

For every cosine index ``i``:

.. code-block:: cpp

   op[i] *= cos(2 * theta);

Cycle bucket
~~~~~~~~~~~~~

For every local cycle ``(src, tgt, phase)``:

.. code-block:: cpp

   const double p = sin(2 * theta) * phase;
   const double o1 = op[src];
   const double o2 = op[tgt];

   op[src] = o1 * cos(2 * theta) - p * o2;
   op[tgt] = o2 * cos(2 * theta) + p * o1;

where ``phase`` is ``+1`` or ``-1`` from Majorana multiplication.

This is the stored two-dimensional rotation for the anticommuting pair.

Summary
-------

The single-rank path is a local gate compiler.

For one generator, it:

1. scans the current basis in parallel
2. classifies anticommuting terms into cosine-only, cycle, or half-term buckets
3. inserts any missing local targets into the basis
4. converts the result into one graph layer that can later be replayed without
   repeating any algebra or hash lookups

The main distinction from the MPI path is not different math. It is that every
lookup, insertion, and cycle resolution happens in one local ``MPOperator``, so the
expensive parts are the branch-heavy scan and local hash-map probes rather than
message exchange.
