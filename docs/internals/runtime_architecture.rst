====================
Runtime Architecture
====================

.. note::

   Developer reference. This page documents internal implementation details
   for contributors modifying performance-sensitive runtime code.

This page connects the public simulator API to the main runtime subsystems and
is the right starting point for locating where a particular behaviour lives
before diving into the more focused internals pages.

Subsystem Map
=============

- `include/monoprop/MonomialPropagator.h` and `src/monoprop/detail/monomial_propagator/*`
  own public API validation, evolution-mode dispatch, functional construction,
  and candidate-functional plumbing.
- `include/monoprop/MPGraph.h` and `src/monoprop/detail/graph/*` own layer storage,
  graph slicing/views, and the shared traversal surface used by both full graphs
  and masked execution plans.
- `include/monoprop/MPFunctions.h` and `src/MPFunctions.cpp` own mapped-parameter
  preparation plus the top-level expectation value and gradient kernels.
- `include/monoprop/Evolution.h` and `src/Evolution.cpp` own replay,
  cross-rank exchange, and analytic derivative accumulation.
- `src/masked_execution_plan/*` owns exact paring over an immutable graph.
- `src/monoprop/monomial_propagator.py` is the Python wrapper that validates
  arguments and forwards into the nanobind-backed simulator.

Simulator Object Model
======================

At construction time `MonomialPropagator<NumModes>` partitions the initial
operator by rank, extracts the constant term into `core_term_`, builds the
rank-local basis/index map in `mp_op_`, regenerates the cutoff function, and
materializes the current Hamiltonian/state caches.

The important owned state is:

- `mp_op_`: the rank-local basis terms, coefficient vectors, indexing map, and
  staged initial Hamiltonian map.
- `graph_`: the persistent evolution graph built from compiled Majorana gates.
- `cutoff_fn_`: the active truncation predicate, rebuilt whenever cutoff type,
  cutoff value, or basis change changes.
- `core_term_`: the constant expectation value offset that stays outside the dense
  coefficient vectors.

Two small helpers define a lot of the runtime behaviour:

- `current_picture_coeffs_()` selects `ham_coeffs` in Heisenberg picture and
  `state_coeffs` in Schrödinger picture.
- `extend_coeffs_from_current_picture_if_needed_()` backfills coefficient
  vectors after graph growth so immediate replay paths can keep running without
  rebuilding the operator from scratch.

Evolution Control Flow
======================

`propagate(...)` is the public dispatcher. It validates the optional
parameter/coefficient arguments, determines one of the three propagation modes,
checks graph-state constraints, and forwards into `execute_evolution_mode(...)`.

The three modes are implemented as separate internal paths:

- graph only: compile layers into `graph_`, do not touch coefficient vectors
- graph with coeffs: compile each layer and immediately replay it into a caller
  supplied coefficient vector
- contract immediately: compile one layer, replay it through a one-layer view,
  then consume that layer so the graph does not accumulate

Order Bookkeeping
-----------------

The supplied Majorana list is not replayed verbatim.

- In Heisenberg picture, `propagate_with_timing(...)` walks the supplied
  list from end to beginning.
- In Schrödinger picture, it walks the supplied list from beginning to end.
- Heisenberg appends compiled layers to the graph.
- Schrödinger inserts each new layer at the logical front of the storage
  vector.

This split keeps insertion and consumption cheap while the replay helpers do the
remaining bookkeeping through `slice_graph(...)`, `slice_view(...)`, and mapped
parameter order.

Functional Construction
=======================

`expectation_value_functional(...)` and `expectation_value_and_gradient_functional(...)` both go
through `make_functional(...)`.

At callable-construction time monoprop captures:

- the current state coefficients
- the current Hamiltonian coefficients
- the current core term
- the communicator
- the expected layer count of the graph at construction time

That snapshotting is deliberate: a functional call should not silently observe a
later simulator mutation. The returned closure validates the parameter length
on every call and also checks that the graph still has the layer count it was
built against.

If `pare_threshold` is set, `make_functional(...)` builds an
`MPExecutionPlan` once and captures that plan instead of the full graph. The
state and Hamiltonian snapshots stay the same in either case.

Expectation value And Gradient Path
===================================

The top-level evaluation kernels live in `src/MPFunctions.cpp`.

The expectation value path is:

1. `fill_mapped_params(...)` writes replay-order parameters into a scratch
   vector.
2. `prepare_evolved_hamiltonian(...)` copies the Hamiltonian snapshot and calls
   `evolve_operator(...)`.
3. `inner_product(state, evolved_hamiltonian)` computes the rank-local overlap.
4. `mpi::allreduce_sum(...)` forms the global expectation value.

The gradient path reuses the evolved Hamiltonian and then loops over the mapped
generators, calling `state_hamiltonian_derivative_local(...)` once per layer.
Those local derivatives accumulate directly into the optimizer-indexed gradient
vector and the code finishes with one `allreduce_sum_inplace(...)` over the full
gradient.

The important consequence is that the hot path does not do a scalar allreduce
inside the layer loop anymore.

Candidate Functionals
=====================

Candidate functionals are built by `make_candidate_functional(...)`.

That path does not mutate the base simulator. Instead it:

1. builds a temporary candidate graph and candidate operator with
   `evolve_candidate_majoranas(...)`
2. materializes candidate Hamiltonian/state coefficient blocks
3. concatenates those blocks after the base state/Hamiltonian vectors
4. optionally unions the base graph with the candidate graph when `use_graph=true`
5. optionally builds a masked execution plan over that combined graph

This is why candidate expectation value and gradient calls can evaluate prospective gates
without committing them to the simulator state.

Where To Change What
====================

- Change argument validation, mode dispatch, or functional capture rules in
  `src/monoprop/detail/monomial_propagator/*`.
- Change replay math or derivative accumulation in `src/Evolution.cpp`.
- Change mapped-parameter handling or expectation value/gradient orchestration in
  `src/MPFunctions.cpp`.
- Change graph storage, views, or traversal indirection in
  `src/monoprop/detail/graph/*` and `src/MPGraph.cpp`.
- Change exact paring or filtered execution plans in
  `src/masked_execution_plan/*`.
