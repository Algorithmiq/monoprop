# Rank-local OpenMP parallelization design

## Status and authority

This is the design accompanying the implementation plan at `../plans/2026-09-18-rank-local-openmp.md`. It records the
direction discussed with the owner; it is not a claim that the proposed implementation meets its performance gates.

Source inspected: branch `perf/linear-routing-on-wire`, commit `290112c8289ab8015eb9a2c7651ac409839c5a88`.

The owner requested a written implementation plan, not implementation or publication. The owner has approved removing
monoprop's direct hwloc dependency and making process/thread counts and placement user responsibilities. Approval of the
full plan is still required before executing it; the remaining compatibility policy is not an assertion that a breaking
C++ API change has already been approved.

## Objective

Replace thread-owned operator partitions and their in-process communication protocols with one compact operator store
per MPI rank, operated on by OpenMP worksharing. Preserve the mathematical features, memory footprint, and performance
of the existing library. Each MPI rank's assigned CPUs are assumed to lie within one NUMA domain. The user chooses the
number of processes P and requested threads per process T for the resources available, and configures the launcher and
OpenMP runtime to bind/pin processes and threads correctly. The library neither discovers topology nor repairs
placement.

## Global constraints

- C++23; retain the GCC 14 / Clang 18 minimum compiler versions and Python 3.11 floor.
- MPI remains optional and OFF by default; retain working non-MPI wheels.
- OpenMP becomes a required implementation dependency for the replacement runtime; support Linux x86_64, Linux aarch64,
  and macOS builds covered by the current project.
- Each MPI rank owns exactly one operator store and one graph; threads are never communicator ranks.
- Remove monoprop's direct hwloc dependency and topology/affinity machinery; do not replace it with another
  library-owned hardware-discovery or pinning mechanism.
- Users choose P processes and T threads per process and configure resource allocation, binding and pinning; monoprop
  does not clamp T to physical cores or validate NUMA placement.
- Preserve packed operator rows, the keyless hash index, fixed uint32_t TermIndex, sparse state storage, and
  inverted-index cosine recomputation.
- Do not introduce boost::concurrent_flat_map, hidden shards, a dense duplicate store, a second persistent key store, a
  full per-thread operator, or full-vector coefficient double buffering.
- Preserve Majorana and Pauli bases, Heisenberg and Schrödinger pictures, cutoffs, graph build/replay/paring, gradients,
  partial contraction, copying, and initial-operator updates.
- MPI calls execute on the controlling thread outside library-created OpenMP worksharing regions; retain
  MPI_THREAD_SERIALIZED rather than changing the initialization contract.
- No exception escapes an OpenMP structured region; distributed operations fail coherently rather than leaving peers in
  collectives.
- No global omp_set_num_threads/omp_set_dynamic/omp_set_nested mutation; each worksharing region receives an explicit
  per-object thread budget.
- Use bitmap traversal or anticommutation pass for the existing operation; reserve prefix offsets for cumulative counts
  used to allocate output ranges.
- Do not change numerical tolerances, truncation rules, generator order, source ordering, or MPI routing to make a
  regression disappear.
- Do not declare success without target-hardware runtime and peak-memory measurements at the same rank count, CPU
  allocation, workload, and fixed 32-bit TermIndex as the baseline.
- No automatic regression allowance: the acceptance target is candidate/baseline <= 1.00 for the median runtime and
  median peak memory of every required workload cell. Noise triggers more measurement, not a silent waiver.
- Update AGENTS.md, README.md, and affected user/contributor docs in the same implementation changes that alter their
  documented behavior.
- No permanent legacy/new-backend selector. Temporary coexistence is confined to the development branch; remove it
  before accepting the replacement.

## Architecture

One rank owns its `MPOperator`, `MPGraph`, and ordinary MPI communicator. A small header-only worksharing helper
supplies deterministic block IDs, explicit thread budgets, and exception collection. It owns no threads, queues,
communicators, or operator data. OpenMP owns worker teams.

Local kernels read an immutable structure or write preallocated, non-overlapping output ranges. The controlling thread
resizes shared containers and publishes structural changes between phases. Initially, insertion into the keyless hash
table and inverted-index maintenance remain single-writer. This is an explicit performance gate, not an assumption that
insertion is negligible.

Only recognized pure `LengthCutoff`/`SupportCutoff` targets may run in worker traversal initially. Arbitrary cutoff
callbacks and current basis-change closures retain serial invocation; a const `std::function` does not establish that
its captured state is thread-safe.

A gate remains sequential with respect to other gates:

1. Prepare lazy caches and snapshot the pre-gate row count.
2. Traverse disjoint inverted-index word ranges, preserving the fused cosine update and capturing pre-update source
   coefficients in the existing record streams.
3. Resolve the leader pass: local resolution, remote query exchange, owner resolution/insertion, and response handling
   in the existing order.
4. Resolve the follower pass after leader matches are visible.
5. Publish deferred local misses after both passes, preserving existing insertion ordering.
6. Finalize the graph layer or apply disjoint fused rotation records.

The freeze boundary in step 1 controls which source rows generate work. Do not change the current post-insertion
`LayerCore::scaled_count` convention when finalizing a graph layer.

Replay uses the existing graph and coefficient arrays. Parallelize cosine blocks, endpoint packing, local rotation
pairs, remote endpoint application, and ordered local reductions. Do not parallelize gate order or parameter order. Keep
MPI replay transport behavior unchanged. EXISTING: `Evolution.cpp` derives rank-uniform `wire_bits`;
`detail/mpi/Exchange.h::post_flat_alltoallv` uses a dense `PeerPlan` but posts only nonzero Isend/Irecv legs when
`wire_bits>0`, otherwise MPI_Ialltoallv. HybridComm derives its sparse plan from all local partitions' rows. Preserve
plain-MPI sparse on-wire replay at S=1; a new generator-derived replay plan is outside this refactor.

## Ownership rules

| Structure | Parallel access allowed | Structural mutation owner |
| --- | --- | --- |
| Packed rows / keyless table | Frozen `find_batch_positions` with disjoint ID/hash outputs | Controlling thread initially |
| Inverted index / row parity | Reads after all lazy initialization finishes | Controlling thread initially |
| Existing coefficients | Disjoint cosine indices or proven-disjoint rotation records | Caller reserves/resizes; workers update assigned elements |
| Query records | Reachable-window and self-position streams; merged in source-range order | Each private buffer's owner, then controlling thread |
| Incoming query probe arrays | Disjoint deserialization/probe output ranges | Caller sizes once |
| Missing-row IDs | Assigned in canonical sender/query or source/query order | Prefix offsets and allocation on controlling thread |
| MatchedEpochSet | Distinct element writes by leader records, followed by a barrier before follower reads | Caller grows/resets; workers may mark unique targets |
| Graph endpoint arrays | Disjoint preallocated records | Caller sizes and finalizes |
| Packed phase words | Not concurrently written merely because record IDs differ | Keep packing serial initially |
| MPI buffers and requests | Pack before posting; do not mutate in-flight buffers | Controlling thread posts/waits and sizes buffers |

EXISTING: `FusedScanResult<N>` in `Scan.h:208–227` has six `WindowVec` streams plus `leader_self`/`follower_self`.
`SlotWindow` indices are not flat slots, including when base is nonzero. Preserve the precomputed `gen_shift`, variable
`QueryWire` records and `QueryForm`; self stages are positions, never serialized wire. Merge only their logical lengths
and rebase position offsets while preserving source/value alignment. Private query buffers have at most
O(T*window.count) empty-container metadata, not O(number_of_bitmap_blocks*R). Per-record payload across those buffers
must be O(total emitted records), never O(T*operator_size). Release/move drained temporary buffers before starting the
next memory-heavy phase. Before parallel decode, the caller checks variable record boundaries, canonical escaped headers
(`h.bits == QueryWire<N>::header_bits_for(h.k)`) and position-count prefix sums, sizes arrays once, then workers decode
into disjoint spans. Frozen probes retain `find_batch_positions` and cached hashes; publication remains serial
`set_positions` → `bulk_insert_hashed` → reindex. No per-rank or full-store dense monomial reconstruction is allowed.
Existing bounded paired-only state scoring may still reconstruct its one required monomial; do not turn that exception
into a batch/store. Task 6 defines the checked preparation sequence.

## Determinism and numerical contract

- Bitmap traversal output is concatenated by increasing source-word range, never completion order.
- Each destination rank's query order remains its source-order subsequence.
- Incoming misses retain sender-rank/query order; local misses retain source-query order.
- Stable row IDs do not use an atomic allocation counter whose results depend on scheduling.
- Reductions use fixed logical blocks independent of thread count and combine partials in block order.
- Repeated calls with a fixed rank geometry must be deterministic. Cross-thread/rank numerical equivalence uses the
  existing project tolerances; do not promise bitwise equality to the old floating-point reduction tree.
- Preserve the exact serial behavior of standalone `inner_product` and `EvalState::dot`; introduce explicitly named
  `inner_product_threaded` / `EvalState::dot_threaded` helpers rather than silently changing their existing contracts.
  Evaluation adopts the new association only after the numerical policy gate. Dot blocks span fixed logical row ranges
  in both sparse and dense representations; cosine blocks span fixed logical word ranges in both lazy and stored-mask
  representations.
- Preserve `LayerAngle`, `CosRecordView`, all three `CosCallbacks` (`LayerCosScale`, `LayerCosAccumulate`,
  `LayerCosIndices`) and current pre-layer-unit derivative formulas. Forward record append and reverse predivide/restore
  remain serial: record indices may duplicate. Threaded gates include deep non-singular circuits, vanishing cosine and
  no-record cases from `tests/test_deep_circuit_gradient.py`, plus exact record restoration in
  `cpp/tests/combined_recompute_equivalence.cpp`.
- Near-cutoff tests compare retained operator terms, not just final energies.
- Preserve streamed `for_each_paired_monomial` initialization order and local reservation/cardinality checks in
  `MonomialPropagator.inl:126–151,197–227`. At S=1 estimates use real ranks, never T. Reject impossible cardinality
  before workers; never materialize the global paired basis per rank.

## Resource ownership and deployment

- The user is responsible for choosing P processes and T threads per process to match available resources, including any
  deliberate SMT use. Normally set P with the scheduler/MPI launcher and T with `OMP_NUM_THREADS=T`.
- The user is responsible for binding/pinning processes and threads to the correct hardware using, for example,
  `OMP_PLACES` + `OMP_PROC_BIND`, `srun --cpu-bind`, and/or `mpiexec` options. Scheduler allocation, rank binding and
  OpenMP places must agree: binding a rank to one core while requesting many threads does not allocate additional cores.
  Each rank's allocation must accommodate its workers within one NUMA domain; avoid unintended overlap with other ranks.
- Monoprop validates thread-count representability, not hardware suitability. It does not discover physical cores,
  choose SMT policy, redistribute CPUs, change affinity, validate NUMA containment or prevent oversubscription. OpenMP's
  own limits and small-work/nested serial paths may use fewer workers; they do not certify adequate resource allocation.
- Delete `CpuTopology` with the partition runtime and remove direct hwloc discovery/link/install requirements. Do not
  retain hwloc for mandatory diagnostics. Benchmark operators still verify placement with runtime/launcher diagnostics
  and external tools; those are evidence requirements, not library dependencies. MPI or the OpenMP runtime may
  independently depend on hwloc.

## Thread selection and compatibility policy

The replacement stores an immutable `parallel::Options` per propagator; copies copy that budget. Without a library
override, capture the calling context's OpenMP default via `omp_get_max_threads()` at construction, normally configured
by `OMP_NUM_THREADS`. Do not parse that environment variable independently; OpenMP also supports list-valued settings
and application-level runtime configuration. Later runtime-setting changes do not alter an existing object's captured
budget. A worksharing region may use fewer workers than the budget. Thread count never changes ownership or graph
layout. MPI ranks may use different budgets.

During the prototype only, add a final C++ constructor argument `size_t num_threads = 0` after the existing
`child_factory` argument. A nonzero value is allowed only when the constructor resolves one legacy partition; it enables
threaded kernels on that single store. Existing legacy children use one kernel thread. This gives a testable transition
without nesting runtimes.

At cutover:

1. Keep the existing `partitions` constructor slot and low-level binding keyword for a deprecation window. Explicit
   positive values become a deprecated thread-budget request, with a once-per-process warning. `partitions=1` remains
   serial. There are no actual thread partitions.
2. Keep `monoprop_PARTITIONS=off|auto|N` as a deprecated, warned thread-budget alias. `off` means one thread; positive N
   requests that budget. `auto` and malformed values fall through to a valid `monoprop_NUM_THREADS` override, otherwise
   the OpenMP default. They no longer discover hardware.
3. A positive explicit C++ `num_threads` wins over environment settings, but supplying it alongside positive explicit
   `partitions` is an error. Otherwise precedence is: explicit positive partitions, then monoprop_PARTITIONS
   off/positive N, then valid monoprop_NUM_THREADS, then the captured OpenMP default. Preserve the existing
   monoprop_NUM_THREADS parser/range; recommend OMP_NUM_THREADS for normal configuration and unset library overrides
   when using it.
4. The OpenMP-default path is the same for one or multiple MPI ranks. Do not retain the old automatic physical-core
   selection or multi-rank default of one thread. If OMP_NUM_THREADS is unset, use the runtime's default, without a
   library promise about that default's suitability. Explicit positive requests must fit `int`; reject unrepresentable
   counts instead of truncating or hardware-clamping them. Public constructor zero remains the no-explicit-request
   sentinel; resolved Options always contains a positive count. Document the changed default-selection policy.
5. `OMP_PLACES` and `OMP_PROC_BIND` control OpenMP placement; monoprop never modifies affinity. Explicit library
   overrides take precedence over the runtime's default thread budget. Runtime limits such as OMP_THREAD_LIMIT and
   dynamic-team adjustment may reduce actual teams even when a larger count was requested; correctness must not depend
   on receiving exactly T workers. No place-union/physical-core query is needed to select or validate T. Verify actual
   placement externally for performance acceptance, not during ordinary propagator construction.
6. Retain the `PartitionChildFactory` type and constructor argument temporarily for source diagnostics, but reject a
   non-null factory with `PropagatorConfigError` explaining that thread-owned children no longer exist. Do not ignore it
   or invoke it once with changed semantics.
7. Remove protected partition-only helpers and facade-specific accessor errors. This is an explicit C++
   extension-interface breaking change requiring a release note; do not describe the refactor as universally source/ABI
   compatible. Keep the virtual destructor, virtual `update_initial_operator`, its protected nonvirtual
   `apply_initial_operator_` helper, and independent virtual `clone_` hook (`MonomialPropagator.h:286–305`).
8. Python mathematical constructors/methods remain unchanged. Low-level `_core` gains the optional `num_threads` keyword
   used by tests; the public frontends continue to use environment configuration.
9. Returned coefficient/term ordering can differ from the old partition-block order. Candidate query/row ordering is
   deterministic across thread budgets at fixed rank geometry; this does not promise the old raw export layout. Validate
   key/value associations, not coefficient-only sorted lists.
10. EXISTING routing is boolean linear/splitmix (`Routing.h::linear_requested`, `linear_bits_for`, `make_router`).
    Linear is the default and requires power-of-two R; unsupported geometry throws `UnroutableGeometry`, not a fallback.
    Preserve linear rank mapping, generator `rank_shift`/`dest_from_shift` and peer selection. Splitmix previously
    placed a term on `floor((hash % (R*S))/S)`; S=1 uses `hash % R`. Obtain explicit approval of that ownership change,
    including rank-local export/load/reduction effects; do not retain hidden shards. New thread budgets never change
    ownership. Test default-linear R=3 rejection separately. Every successful R=3 suite or benchmark explicitly sets
    `monoprop_ROUTING=splitmix`. Compare global maps and per-rank loads/peaks at identical routing/seed/rank geometry.

The plan's final removal task may proceed only after the owner approves this policy as part of the plan, or provides a
replacement policy. Implementation agents must not invent another migration.

## Safety and failure policy

The worksharing helper catches each worker exception, joins the region, and rethrows on the calling thread. Caller-side
distributed phase guards print the underlying failure and call MPI_Abort when communicator size exceeds one. Catch
failures inside the lifetime of active exchange tickets, before stack unwinding can wait on peers that never posted; an
outer catch alone is insufficient. Successful paths explicitly wait as before. For single-rank operation, rethrow the
original exception. An operation that failed after mutations is not transactional and the propagator must not be reused.
No rank-local early return may bypass a collective peers will enter. Validate MPI_THREAD_SERIALIZED also when MPI was
initialized externally. Distributed entry from an external OpenMP team is not a supported calling contract; numerical
helpers may serialize when nested, but this does not authorize MPI from workers or concurrent evaluation on a shared
communicator.

Lazy initialization, allocation, and cache mutation must be outside worker bodies unless storage is strictly
worker-private. Do not assume `const` means thread-safe: both the inverted index and parity cache have lazy mutation
today.

## Performance acceptance and stopping

Keep the baseline revision separately built. The benchmark operator must set P/T and verify the allocation and actual
worker placement using OpenMP/launcher diagnostics and external tools, not a monoprop topology service. Use identical
MPI rank placement and physical cores; changing ranks per domain or widening indices is not a valid comparison. Measure
at least five fresh-process observations per required cell, alternating baseline/candidate. Each observation has one
benchmark round. Compare medians per cell, both runtime and kernel VmHWM (sum across ranks and maximum rank peak).
EXISTING operation windows and nested-window peak preservation already work (`benches/conftest.py` and benchmark
`memory/cpu.py`). Extend them, do not reimplement them. Separate fresh-process whole-construction measurements remain a
deliberate policy to include graph creation without unrelated resident session graphs, not a workaround for broken
nesting. PROPOSED artifacts independently aggregate `opmemexact` for operation metrics, `memhwmexact` for outer
`memhwm`/`memhwm_max`, and `construction_exact` for the separate construction window. Never certify outer peaks with
`opmemexact`; legacy absent flags are unknown. Renderers label each metric using its own provenance. The parity gate has
exactly FIVE metrics: runtime, operation peak sum/max, and whole-construction peak sum/max. Operation and construction
exactness must both be true; exit-RSS fallbacks cannot pass. Outer peaks/ratios are diagnostics only: non-exact or
unknown outer evidence must be labeled honestly but cannot fail an otherwise passing five-metric comparison. All five
required ratios must be <=1.00. Rank-peak sums are upper bounds, not simultaneous node-footprint measurements.

If a ratio exceeds 1.00, repeat the cell for five additional observations on each arm. If the combined median still
exceeds 1.00, the gate is not passed. Profile and use only the bounded optimization menu in the plan; otherwise stop
with a regression report. No geometric-mean improvement cancels a failing required cell. Missing target hardware is a
validation blocker, not permission to claim parity.

Benchmark type/length assertions do not establish numerical parity. A separate untimed validation process records
energies, gradients where applicable and decoded global term/value maps, compared against the identical baseline
operation with the existing project's tolerance policy. Paring compares the same pared callable/threshold on both arms.
Missing/non-finite evidence fails acceptance; replicated Heisenberg core terms are checked once rather than summed
across ranks. Keep exact small-system and finite-difference tests as independent oracles. Preserve all four current
pytest families: build_graph, graph-free propagate, energy and gradient. Record random `lower_atol=None`, fixed-model
cutoffs/atols and both unpared/default and `pare_threshold=1e-10` evaluation profiles. Keep original paring-construction
performance coverage through an explicitly NEW bounded driver measurement, `pare_functional_construct`, specified in
Task 1: graph ready before timing/op window, threshold 1e-10, returned gradient callable retained through close, untimed
scalar/full-gradient validation, and separate whole-construction peak scope. This is proposed measurement work under
full-plan approval, not an existing pytest node or instrumentation-only edit. Use identical driver code on both arms; do
not build a general benchmark framework.

Freeze and archive an owner-approved campaign inventory and digest before baseline collection. It lists every required
profile/node/geometry/routing cell independently of the observations. Comparator input must match that inventory, so
removing a whole cell from both arms cannot pass. Each sample references one fresh timed artifact, one separate fresh
construction artifact, and a matching untimed validation artifact. Join only matching cell/arm/binary/configuration,
parameters, geometry/routing/allocation and campaign provenance. Timed/construction run IDs and files cannot be reused;
require five distinct pairs per arm/cell, or ten after prescribed repetition. Validation and placement diagnostics can
be reused only for the same cell/arm/binary/configuration. Task 1 specifies the CLI, file schema, identity checks and
independent placement references; no caller boolean or synthetic-data bypass substitutes for that evidence.

The shared shape-preflight overlay must accept declared baseline `monoprop_PARTITIONS=T` and candidate
`OMP_NUM_THREADS=T` with library overrides unset, on the same allocation. Verify declarations against observed shape;
arm labels alone cannot establish shape. Extend existing ladder/provenance/term-count fields without renaming historical
nodes. `tools/capture-baseline.py` supplies supplementary small-fixture evidence only; it cannot replace global,
ownership-aware benchmark validation with gradients and provenance.

The full-rank store must fit the fixed uint32_t TermIndex ceiling (`TypeAliases.h:39`). Row IDs exclude UINT32_MAX,
which is the index sentinel; row counts and some graph counts may equal it. Extend existing overflow guards/tests rather
than claim there are two index-width builds. The store must also fit capacity arithmetic and MPI count limits. If not,
report that capacity incompatibility; do not widen indices, change rank geometry, or add local shards without a new
design decision.

## Non-goals

No new executor framework, custom concurrent hash table, GPU backend, load-balancing/repartitioning algorithm,
asynchronous gate pipeline, GIL-policy change, graph-format redesign, new replay transport, or replacement
topology/affinity-management subsystem. Do not remove useful lifetime/ordering comments merely to lower line count.

## Execution handoff and supported build routes

All future code/test/build commands in the companion plan are planned, not executed during this document refresh. Only
these two Markdown documents are authorized now. Before implementation, recheck exact HEAD and the plan's baseline-drift
table; stop on changed signatures, routing, capacity, numerical or benchmark contracts. Keep 13 gated tasks and record
RED/GREEN evidence for new behavior, plus baseline results for preservation tests that may already pass. Do not
manufacture RED by removing earlier plumbing. Use kernel-specific worker observations for threading participation. Do
not execute past a failed gate or infer approval from this refresh.

Required OpenMP must link PUBLIC on both `monoprop-objs` and the shared `monoprop` target, plus installed
`find_dependency(OpenMP REQUIRED COMPONENTS CXX)`; INTERFACE alone does not link the shared library itself. Cover wheel
hooks in `pyproject.toml`, actual wheel jobs in `.github/workflows/deploy.yml`, `tools/install-deps.sh`, shared
`tools/packages/{apt,brew}.txt` and setup action, `.devcontainer/Dockerfile`, `nix/{monoprop,devshell}.nix`, `flake.nix`
package/check entrypoints and the Nix workflow. Remove direct hwloc requirements/checks LAST, with the partition
runtime; legitimate transitive dependencies remain. Use environment-driven MPI recipes from `justfile` so isolated
builds receive mpi4py; `--all-extras` alone cannot do so. Migrate centralized TSan selection before retiring partition
tests, preserve the unit-only export probe, and extend `just test-find-package` rather than create a second consumer
harness. MPI OFF clean wheels remain mandatory. Before any separately authorized push, run `prek run --all-files`,
relevant tests and docs generation/build.
