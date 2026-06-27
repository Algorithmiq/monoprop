=====================
Masked Execution Plan
=====================

The current optimization path keeps the evolution graph immutable and moves
exact paring into a lightweight execution plan. This avoids rebuilding large
graph copies while still letting functional evaluation skip inactive work.

Overview
========

The optimized path has three main pieces:

- Layer payload is shared.
- Full layers cache one evolution exchange layout; derivative layouts are
  derived on demand and filtered plans build reduced layouts only when needed.
- Exact paring builds an execution plan over the original graph instead of a
  second graph.

At a high level the flow is:

- Graph construction stores each layer in a shared LayerStorage object.
- Cosine indices are compressed into contiguous CosineSpan runs.
- Each full layer caches flat Alltoallv counts and displacements for evolution
  replay.
- Functional construction builds an MPExecutionPlan that references the
  original layer storage and only materializes filtered position lists for the
  layers that actually change.
- Masked-plan construction exchanges keep information once per layer, then
  performs at most one local refilter pass instead of iterating a global
  fixpoint.
- Evaluation replays the plan directly, maps parameters in reverse order
  without an extra reversal copy, reuses flat send/recv buffers, and finishes
  gradients with one vector allreduce.

Builder Walk Order
==================

The builder walks layers in the direction needed to propagate support back to
their sources:

- Heisenberg picture iterates from the newest stored layer back toward the
  oldest (`num_layers - 1` down to `0`).
- Schrödinger picture iterates from `0` upward.

That ordering matches `build_masked_execution_plan(...)` in
`src/masked_execution_plan/BuildExecutionPlan.cpp` and keeps the local
`nodes_to_keep` bitset aligned with the currently active side of the circuit.

Shared Layer Storage
====================

Layer now owns a shared LayerStorage backing object that contains:

- compressed cosine data
- local cycles
- cross-rank cycles
- cached evolution exchange layout

This removes repeated vector and control-block overhead when multiple views of
the same layer are needed. Unfiltered execution plans can point back to the
original layer storage without copying any per-layer payload.

Compressed Cosine Runs
======================

Raw cosine index lists are stored as contiguous spans rather than individual
size_t indices. Each span keeps:

- a 32-bit absolute start index on the common path
- an out-of-line wide-start fallback for spans above 2^32
- a 16-bit run length

The runtime kernels in Evolution.cpp iterate spans directly. That reduces graph
memory footprint and lowers index bandwidth in the cosine kernels.

Cached Flat MPI Exchanges
=========================

Each full layer precomputes flat Alltoallv counts and displacements for
evolution replay only.

`Evolution.cpp` derives the derivative layout in thread-local scratch storage by
doubling those cached counts and displacements. Filtered `LayerExecutionPlan`
objects build a reduced evolution layout only when cross-rank positions are
actually masked out.

The hot path therefore no longer rebuilds per-rank payload vectors on every
layer update. It packs into reusable flat buffers, performs one collective, and
applies the received payload using either the cached full-layer layout or the
filtered execution-plan layout.

Masked Exact Paring
===================

When pare_threshold is enabled, monoprop does not build a separate pared graph.
Instead it creates an MPExecutionPlan over the original graph:

- local seed indices are collected from the active state or Hamiltonian on each
  rank
- each layer exchanges only the keep flags needed to decide which cross-rank
  edges remain active
- if a layer is untouched, the plan reuses the original LayerStorage directly
- if a layer is filtered, the plan stores compact position lists for the
  surviving cosine spans and cycles

The important detail is that support discovery stays rank-local. The builder does not allgather global nonzero indices; it exchanges only the keep decisions needed to preserve exact support across rank boundaries.

This preserves the exact-support semantics of the old pared-graph path while
avoiding a large fraction of the copied-graph memory cost.

Layer Filtering And Refilter Pass
=================================

The builder logic in `BuildExecutionPlan.cpp` and `LayerFiltering.cpp` runs in
three stages for each layer:

- `pack_source_keep_flags(...)` sends one keep flag per outgoing source index
  to the remote owner of the matching incoming target.
- `filter_layer_execution_plan(...)` filters cosine spans, local cycles, and
  cross-rank positions against the local `nodes_to_keep` set and the received
  remote keep flags.
- `merge_remote_selected_sources(...)` applies the second exchange of selected
  remote sources. If that activates new local sources, the layer is filtered
  one more time locally.

This is why the builder needs at most one local refilter pass per layer rather
than a global fixpoint loop.

Filtered storage is recorded as:

- compressed cosine blocks for masked cosine spans
- compressed position blocks for local cycles and cross-rank edges
- `CrossRankMaskRange` metadata so traversal can map each rank's logical
  outgoing/incoming positions back into the shared compressed blocks

Traversal Surface
=================

The replay kernels do not care whether they are walking a full graph layer or a
filtered execution-plan layer. `LayerTraversal` hides that distinction.

For untouched layers, traversal reads directly from the original `LayerStorage`.
For filtered layers, it remaps logical positions through the compressed
position-data blocks and uses the execution plan's reduced evolution exchange
layout.

That common traversal surface is what lets `evolve_operator(...)` and
`state_hamiltonian_derivative_local(...)` work against `MPGraph`,
`MPGraphView`, and `MPExecutionPlan` without branching on storage format.

Gradient Path
=============

The expectation-value-and-gradient functional uses two key runtime optimizations:

- state_hamiltonian_derivative_local computes each local contribution without a
  scalar allreduce inside the layer loop
- the full gradient vector is reduced once with allreduce_sum_inplace at the
  end
- parameter mapping uses map_params(..., reverse=true) so replay gets logical
  reverse order directly without building an intermediate reversed vector

The derivative path still reuses the evolution-side exchange information. It
does not store a second derivative layout per layer; `Evolution.cpp` derives the
2x-scaled derivative layout from the active evolution layout when needed.

That removes hundreds of small collectives from typical workloads and keeps the
derivative path aligned with the lighter execution-plan replay.

Invariants
==========

When changing this path, keep these invariants intact:

- graph and coefficient indices are rank-local
- exact paring should exchange only per-edge keep information, not allgather
  the full active support
- unchanged layers should stay on the original shared LayerStorage path
- runtime kernels should prefer flat reusable buffers over per-rank nested
  allocations

If you benchmark a change, measure setup time, evaluation time, and retained RSS
separately. The design intentionally trades a modest amount of plan metadata for
substantially lower copied-graph memory overhead.
