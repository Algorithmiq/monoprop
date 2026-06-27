Graph
=====

.. admonition:: Developer reference

   This page documents internal implementation details for contributors modifying
   performance-sensitive runtime code.

``include/monoprop/MPGraph.h`` · ``src/monoprop/MPGraph.cpp``

Overview
--------

``MPGraph`` records the sequence of compiled evolution gates for one rank. Each
gate corresponds to one logical ``Layer``, but the graph does not store layers as
raw cosine-index lists and cycle tuples. Instead it stores a replay-oriented
encoding: shared per-layer payload, compressed cosine spans, packed cycle data, and
cached cross-rank exchange layouts.

For the full compile-and-replay story, including how a generator is turned into a
layer and then replayed against coefficients, start with :doc:`/internals/graph_evolution`.
This page focuses on the graph object model and the concrete storage layout.

MPGraph
-------

``MPGraph`` is the owning container for the logical replay program of one rank.
It stores:

- ``std::vector<Layer> layers_``
- a ``schrodinger_`` flag that determines logical ordering
- a ``front_offset_`` that lets Heisenberg-prefix contractions avoid O(N) erasures
  on every slice

Appending a layer stores the payload in shared ``LayerStorage`` and caches the flat
evolution exchange layout once. The derivative path later derives its 2x-scaled
exchange layout on demand from that cached evolution layout.

.. code-block:: cpp

   // Schrödinger: new layers prepended at the logical front
   // Heisenberg:  new layers appended at the logical back
   MPGraph graph(/*schrodinger=*/false);

   graph.append(cos_inds, local_cycles, cross_rank);  // compress on append
   graph.append(cos_data, local_cycles, cross_rank);  // already-compressed cosine data
   graph.layers();                                     // active logical layer count
   graph.get_layer(i);                                 // Layer& at logical index i

Picture-specific storage order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Heisenberg**: the build loop walks the supplied generator list from end to
  beginning and appends each compiled layer.
- **Schrödinger**: the build loop walks the supplied list from beginning to end
  and inserts each new layer at the logical front.

Raw storage order is therefore an implementation detail. Replay always uses logical
layer order through ``get_layer(i)`` / ``get_layer_traversal(i)``.

Prefix consumption and compaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a Heisenberg graph is partially contracted, ``front_offset_`` advances instead
of erasing the consumed prefix immediately. ``src/MPGraph.cpp`` compacts the
underlying vector only when the dead prefix is large enough to justify the copy:

.. code-block::

   front_offset_ >= 4096 && front_offset_ * 2 >= layers_.size()

That keeps repeated prefix contractions cheap while still bounding wasted storage.

Layer
-----

A ``Layer`` is just a shared handle to one ``LayerStorage`` payload for a single
evolution gate. The stored payload is already compressed:

.. list-table::
   :header-rows: 1

   * - Component
     - Type
     - Meaning
   * - ``cos_data``
     - ``CompressedCosineData``
     - Coefficient indices stored as contiguous ``CosineSpan`` runs plus the total
       logical cosine count
   * - ``local_cycles``
     - ``PackedLocalCycleStorage``
     - Packed ``(src, tgt, phase)`` rotations between indices on the same rank
   * - ``cross_rank``
     - ``PackedCrossRankStorage``
     - Flattened per-rank outgoing/incoming cycle payloads plus offsets
   * - ``evolution_exchange_layout``
     - ``LayerExchangeLayout``
     - Cached counts/displacements for flat replay exchanges; the derivative
       exchange (2× scale) is derived on-the-fly from this

A logical local cycle is ``{src, tgt, phase}``. During replay the local rotation is:

.. code-block::

   src' = cos(2θ) * src - phase * sin(2θ) * tgt
   tgt' = cos(2θ) * tgt + phase * sin(2θ) * src

``Layer`` itself only owns a ``shared_ptr<LayerStorage>``. ``union_with()`` and
``slice_view()`` therefore share the full layer payload without copying it.
``materialize_cos_inds()`` expands cosine spans back into a ``VecZ`` only for
debugging or serialization.

Cosine encoding
~~~~~~~~~~~~~~~~

The cosine bucket is stored as ``CompressedCosineData``, which is a chunked
run-length encoding rather than a raw list of indices.

- ``chunk_bases`` stores the high bits of each active chunk
- ``chunk_span_starts`` locates that chunk's spans in the flat span arrays
- ``span_offsets`` stores 16-bit offsets inside the chunk
- ``span_counts`` stores span lengths as 8-bit run lengths
- ``total_count`` stores the total logical number of cosine indices

Each logical span start is reconstructed as:

.. code-block::

   span_start = chunk_base | span_offset

This is why replay kernels can iterate cosine spans directly without first
materializing a list of indices.

Note that these spans are rank-local and involve no communication.

Example: raw cosine indices → CompressedCosineData
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

   before:
   cos_inds = [12, 13, 14, 70000, 70001, 70003, 131072, 131073]

   after:
   total_count       = 8
   chunk_bases       = [0, 65536, 131072]
   chunk_span_starts = [0, 1, 3]
   span_offsets      = [12, 4464, 4467, 0]
   span_counts       = [3, 2, 1, 2]

This decodes to four logical cosine spans:

.. code-block::

   (12, 3)       -> [12, 13, 14]
   (70000, 2)    -> [70000, 70001]
   (70003, 1)    -> [70003]
   (131072, 2)   -> [131072, 131073]

So replay can multiply four contiguous ranges instead of reading eight separate
indices from a ``VecZ``.

Local and cross-rank cycle encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local cycles are stored in ``PackedLocalCycleStorage``:

- if ``src`` and ``tgt`` fit in 32 bits, they are packed into one ``uint64_t``
- otherwise they fall back to separate wide arrays
- phases use ``PackedPhaseStorage``, which bit-packs the common binary ``±1`` case

Cross-rank cycles are stored in ``PackedCrossRankStorage``:

- each remote rank gets a ``CrossRankStorageRange`` describing its outgoing and
  incoming slice inside shared flat arrays
- indices use a 32-bit packed representation when possible and wide arrays when
  necessary
- outgoing and incoming phases again bit-pack when all phases are binary

This avoids one allocation per remote rank per layer while keeping the public
per-rank traversal API intact.

Example: one layer before and after packing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The planner-facing form of one layer might look like this before encoding:

.. code-block::

   cos_inds = [3, 4, 5, 11]

   local_cycles = [
       {src: 1, tgt: 9, phase: +1},
       {src: 8, tgt: 10, phase: -1},
   ]

   cross_rank = [
       rank 0: {out: [], out_phases: [], in: [], in_phases: []},
       rank 1: {out: [20, 23], out_phases: [+1, -1], in: [8], in_phases: [+1]},
       rank 2: {out: [4], out_phases: [-1], in: [11, 15], in_phases: [+1, -1]},
   ]

After ``build_layer_storage(...)``, the same logical layer becomes:

.. code-block::

   cos_data = {
       total_count: 4,
       chunk_bases: [0],
       chunk_span_starts: [0],
       span_offsets: [3, 11],
       span_counts: [3, 1],
   }

   local_cycles = {
       uses_wide_indices: false,
       compact_pairs: [(1 << 32) | 9, (8 << 32) | 10],
       phases: {uses_binary_phases: true, phase_words: [0b10]},
   }

   cross_rank = {
       ranges: [
           {out_offset: 0, out_count: 0, in_offset: 0, in_count: 0},
           {out_offset: 0, out_count: 2, in_offset: 0, in_count: 1},
           {out_offset: 2, out_count: 1, in_offset: 1, in_count: 2},
       ],
       out_indices: [20, 23, 4],
       out_phases: {uses_binary_phases: true, phase_words: [0b110]},
       in_indices: [8, 11, 15],
       in_phases: {uses_binary_phases: true, phase_words: [0b100]},
   }

The important change is that the logical rank-partitioned vectors have been flattened
into one packed storage block per category. ``ranges`` is what lets
``LayerTraversal`` reconstruct the per-rank view without storing separate
``std::vector``\s for every remote rank.

If any index exceeds ``uint32_t``, monoprop switches that category to its wide array
representation instead of using ``compact_pairs`` / ``out_indices`` /
``in_indices``.

Example: before and after MPI structuring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same ``cross_rank`` example has two different shapes depending on where you
look at it.

Before MPI structuring, replay still thinks in per-rank logical groups:

.. code-block::

   rank 1: out = [20, 23], in = [8]
   rank 2: out = [4],      in = [11, 15]

After ``build_layer_exchange_layout(cross_rank, 1)``, the layer gets one flat MPI
layout:

.. code-block::

   counts     = [0, 3, 3]
   displs     = [0, 0, 3]
   total_count = 6

Those counts come directly from ``out_size(rank) + in_size(rank)`` for each rank.
For this layer, rank 1 owns a 3-value slice and rank 2 owns a second 3-value slice.

On the sending side, ``pack_cross_rank_evolution_payload_impl(...)`` writes one flat
buffer with outgoing values first and incoming values second inside each rank slice:

.. code-block::

   send_buffer = [
       op[20], op[23], op[8],
       op[4], op[11], op[15],
   ]

After ``MPI_Alltoallv``, the received slice for each rank is interpreted in the
opposite logical order: incoming shadow values first, then outgoing shadow values.

.. code-block::

   recv_buffer = [
       src_shadow_for_8, tgt_shadow_for_20, tgt_shadow_for_23,
       src_shadow_for_11, src_shadow_for_15, tgt_shadow_for_4,
   ]

That is why replay applies the two parts differently:

.. code-block::

   rank 1 incoming edge 8    uses recv_buffer[0]
   rank 1 outgoing edges     use recv_buffer[1], recv_buffer[2]
   rank 2 incoming edges     use recv_buffer[3], recv_buffer[4]
   rank 2 outgoing edge 4    uses recv_buffer[5]

The derivative path keeps the same logical slicing but uses ``scale = 2``, so the
layout becomes ``counts = [0, 6, 6]``, ``displs = [0, 0, 6]``, and each logical
entry expands to a ``[state, hamiltonian]`` pair instead of one coefficient.

Traversal surface
~~~~~~~~~~~~~~~~~~

Replay code does not poke raw storage directly. It consumes ``LayerTraversal``,
which exposes one logical API over either a full stored layer or a masked
execution-plan layer:

- cosine spans via ``cos_data()`` / ``for_each_cos_span(...)``
- local cycles via ``for_each_local_cycle_range(...)``
- cross-rank cycles via ``for_each_cross_rank_out_range(...)`` and
  ``for_each_cross_rank_in_range(...)``
- cached MPI metadata via ``evolution_exchange_layout()``

That shared traversal contract is the reason the same replay kernels can operate on
``MPGraph``, ``MPGraphView``, and ``MPExecutionPlan``.

LayerExecutionPlan and MPExecutionPlan
-----------------------------------------

``LayerExecutionPlan`` is the filtered replay view used when exact paring is enabled
for a functional. A layer plan either:

- reuses the original ``LayerStorage`` unchanged, or
- points at shared execution-plan storage containing only the surviving cosine spans
  and/or compressed position maps.

Execution plans do **not** clone the whole graph. They keep pointers to the original
``LayerStorage`` and store only the filtered pieces:

- replacement ``CompressedCosineData`` blocks for filtered cosine buckets
- ``CompressedPositionData`` blocks that remap logical local-cycle positions back to
  the original stored cycles
- ``CompressedPositionData`` blocks plus rebuilt per-rank ranges for filtered
  cross-rank edges

Filtered plans also build a reduced ``LayerExchangeLayout`` when cross-rank edges
have been masked away. ``MPExecutionPlan`` is therefore an ordered vector of
``LayerExecutionPlan`` objects that reuses the original storage wherever possible.

Slicing
~~~~~~~~

.. code-block:: cpp

   // slice_graph: returns a new graph containing the k earliest layers
   // contract=true removes those layers from this graph in-place
   MPGraph sub = graph.slice_graph(/*key=*/3, /*contract=*/true);

   // slice_view: zero-copy view over the same shared layer data
   MPGraphView view = graph.slice_view(3);

Union
~~~~~~

.. code-block:: cpp

   // Combine two graphs; layers share their LayerStorage via shared_ptr
   MPGraph combined = graph_a.union_with(graph_b);

In Heisenberg picture ``graph_a``'s layers come first (applied first). In
Schrödinger picture ``graph_b``'s layers come first (most recent).

MPGraphView
-----------

A lightweight, non-owning window over a range of layers. It stores:

- a pointer to the original ``std::vector<Layer>``
- ``base_`` and ``count_`` for the logical window
- a ``reverse_`` flag so Schrödinger slices can preserve logical replay order

It supports ``get_layer(i)`` with bounds checking and presents the same traversal
surface as the owning graph.

Used by ``contract_partially(inplace=false)`` and ``evolve_operator()`` to avoid
materialising a copy of the graph.

The execution-plan path uses the same logical indexing contract, but with
``LayerExecutionPlan`` in place of ``Layer``.

Inspecting graph data
----------------------

``graph_data()`` on the simulator returns a Python-friendly vector of per-layer
tuples for debugging and serialisation:

.. code-block:: cpp

   auto layers = sim.graph_data();
   for (const auto& [cos_inds, local_cycs, out_data, in_data] : layers) {
       // cos_inds:  VecZ
       // local_cycs: vector<tuple<src, tgt, phase>>
       // out_data / in_data: vector<tuple<VecZ indices, VecI phases>> indexed by remote rank
   }

Internally, ``graph_data()`` materializes everything through ``LayerTraversal``, not
through raw storage members. That means the debugging format stays stable even
though the underlying graph uses compressed cosine spans and packed cycle storage.

Memory inspection
------------------

Both the full graph and masked execution plans can report storage usage:

- ``MPGraph::storage_memory_usage()``
- ``MPExecutionPlan::storage_memory_usage()``

These walk unique shared storage blocks so layer-sharing via ``shared_ptr`` is not
double-counted.
