# Sharded OpenMP execution design

## Status and authority

This document records the owner-approved redesign following the failed one-store experiment, including the subsequent
interface and numerical-policy interview. It supersedes the production architecture in the
[one-store design](2026-09-18-rank-local-openmp-design.md). Its implementation is organized in the
[sharded OpenMP plan](../plans/2026-09-29-sharded-openmp.md).

Source checkpoint: `a6a995a4352bc5a9494acc4c13df33aa61ef84ef` on `refactor-parallelization`.
The [historical plan](../plans/2026-09-18-rank-local-openmp.md) preserves Tasks 0–6 and their evidence; its Tasks 7–13
are superseded, not the next implementation steps. Task 1's frozen campaign and formal partition baseline remain the
acceptance reference. Historical requirements that conflict with the explicit decisions here have no execution
authority.

The owner authorized documenting this design and plan. This does not authorize implementation, builds, measurements,
remote expenditure, delegation, commits or publication. The written documents remain available for owner review before
separately authorized task execution. Proposed modules below do not exist at this checkpoint.

## Why retain sharding

The recorded Task 6 diagnostic covered 96 frozen cells with three alternating fresh observation pairs per cell.
Numerics matched, but none of the 16 reference nodes reached runtime parity at full machine; recorded ratios were
2–62 times the partition baseline. In two construction cases, the prototype's parallel traversal alone exceeded the
baseline's whole operation: 0.465 s versus 0.399 s, and 1.34 s versus 0.50 s. Removing the remaining serial phases
would not repair those results. The separate libgomp regression from alternating 16- and 96-worker regions is not an
adequate explanation for the overall gap.

These are archived findings, not measurements reproduced while writing this design. They reject that implementation
path, not OpenMP generally. Shards allow independent owners to perform the whole pipeline, including row insertion,
index maintenance and graph construction. Retain that ownership while replacing its custom execution machinery.

## Objective and fixed constraints

One rank-level propagator owns T internal shard states. OpenMP supplies a coarse-grained team for an entire operation;
workers execute local phases against their own shards. Direct shared-memory handoffs replace in-process communicators.
MPI's initializing thread performs physical-rank communication as the OpenMP primary thread.

- C++23; GCC 14 / Clang 18 minimum; Python 3.11 minimum.
- OpenMP CXX is required, including MPI-off builds and installed consumers. MPI remains optional and OFF by default.
- Retain Linux x86_64, Linux aarch64 and macOS packaging coverage; report untested platforms as pending.
- Keep packed position-list rows, keyless indexing, sparse state, inverted indices and packed graph representation.
- Keep `TermIndex = std::uint32_t`, with its reserved sentinel, per shard. Retain checked count/allocation arithmetic.
- Preserve both bases and pictures, cutoffs, basis changes, propagation, graphs, gradients, paring, copies and updates.
- Remove the custom worker pool, in-process communicator abstractions, custom barriers and direct hwloc dependency.
- Allocation, placement and pinning belong to the launcher/OpenMP configuration. Recommend one worker per physical core.
- Numerical preservation and the frozen runtime/memory gates govern this refactor; neither is waived by a faster kernel.

Terminology: **traversal** means the bitmap/anticommutation pass that emits partner queries. **Prefix-sum** means
cumulative counts used to assign offsets. Existing source identifiers containing `Scan` are historical names, not a
reason to use ambiguous terminology in the redesign. Retained-operator terminology follows `CONTEXT.md`.

## Launch and ownership contract

The supported calculation is P MPI processes with exactly T OpenMP threads each, configured through environment
variables before `mpirun`/`srun` or a direct MPI-off launch. P and T are fixed for the calculation; T is uniform across
MPI ranks. This is a caller precondition, not a condition the library must establish defensively.

Capture `monoprop_NUM_THREADS` at construction, or use `omp_get_max_threads()` when it is absent. Reuse the existing
scalar parser; there is no independent `OMP_NUM_THREADS` parser, hardware clamp, setter or operation-time reload.
Copies preserve the captured budget. The shard count is internal and equals T; there is no separate shard-count knob.

The owner explicitly accepts uncontrolled failure for inconsistent thread geometry. Add no thread-count consistency
collective, per-region actual-team check, reduced-team fallback, resharding, heterogeneous-rank support or friendly
mismatch diagnostic. Runtime settings that supply a different team are outside the supported contract. Document an
exact-team launch (`OMP_DYNAMIC=FALSE`, no limiting setting below T) without mutating global OpenMP settings in code.
Placement/participation observation in tests and performance diagnostics is evidence gathering, not library enforcement.

Worker t exclusively mutates shard t during ordinary parallel phases. This is an OpenMP team index, not a promise of
persistent OS-thread identity between calls. A flat owner ID `rank * T + shard` remains useful for routing and graph
slots; it is not an MPI communicator rank. No shard owns a fake communicator or an independent public propagator.

Use the existing router with `(P,T)`, preserving its linear/splitmix behavior, seed, full-rank basis and rejection of
unsupported linear process counts. In particular, preserve splitmix ownership through `hash % (P*T)` rather than the
abandoned `hash % P` cutover. Successful three-process tests use splitmix. Changing T between separate launches changes
ownership geometry; identical row IDs across those launches are not required.

Keep routing mode/seed agreement and communicator-lifetime caching on the primary/caller. Thread-budget agreement is
not part of the new contract. The existing partition-count agreement field must not become a new T-consistency check.

## State and execution

### Rank-level owner

`MonomialPropagator` remains the mathematical interface and owns configuration, the ordinary MPI communicator, T shard
states and operation/functional validity. It is not a facade dispatching calls to T child propagators.

Each shard owns its `MPOperator`, primary `MPGraph`, matched marks and other shard-local mutable mathematical state.
Local IDs, coefficients, initial-state data and graph endpoints refer to that shard's store. Immutable graph cores may
be shared where they already are; independent propagator copies own independent stores. Existing retained snapshots and
pared graphs remain legitimate mathematical objects, not prohibited duplication.

Initialize and copy substantial shard storage on its owner worker for first-touch locality. Preserve streamed paired
basis enumeration and per-owner reservation; do not materialize the whole initial basis once per shard. Metadata can
be sized on the caller. Scratch is bounded and reused where appropriate; moving existing thread-local scratch into
per-object storage must account for simultaneous instances, not silently multiply resident memory.

### Operation-scoped team

One OpenMP region spans a build, propagation or evaluation call, including its gate loop or reverse-gradient loop.
Construction/copy can have their own initialization region. There are no object-owned threads, background jobs or
persistent custom executors. Initially, numerical kernels are serial within each shard: parallelism is across owners.
Avoid reintroducing the one-store prototype's succession of small and large nested teams.

The orchestration exposes explicit local phases and primary-only MPI phases. Local bodies contain no MPI calls or
hidden synchronization. All workers, including empty shards, reach the same phase boundaries. OpenMP barriers provide
publication and reuse guarantees; ordinary shared buffers replace communicator verbs and barrier state machines.

A small operation-local exception/checkpoint helper is permitted. It owns no scheduling policy, job queue, threads or
transport. `Workshare.h::for_blocks` is not a replacement for the collective, long-lived-team phase protocol.

### Primary-thread MPI

The thread that initialized MPI enters the operation and is the OpenMP primary thread. It may call MPI inside the
parallel region. Use primary-thread execution, never an arbitrary `omp single` winner, for posting, waiting,
collectives, routing-cache queries and communication cleanup. Higher MPI thread support does not change that ownership.

Retain SERIALIZED while the legacy Hybrid path remains executable. Request/require FUNNELED after removing it. Validate
host-provided support without reinitializing MPI. Public MPI entry from a different host thread or an active host team
remains unsupported; the newly approved in-team calls are made by the library's own initializing primary thread.

Workers still perform payload preparation and consumption. The primary owns MPI calls, not all packing, decoding,
lookup, insertion or graph work. Reuse the existing physical-MPI request ownership and sparse/dense transport semantics.

## Construction and handoffs

Preserve the current gate protocol and arithmetic. Extraction of the phases in `layer_build/Engine.h` changes
orchestration, not which queries become rows.

1. Each owner prepares its private lazy caches, snapshots its source boundary, traverses its shard and emits queries.
2. Publish stable outgoing buffers at a team checkpoint. They cannot resize or die while another owner reads them.
3. Resolve same-shard queries with the existing position-stage fast path. **Self means the same shard**, not the same
   physical process. Deferred self misses remain deferred.
4. Make other local owners' query blocks available by immutable views. Workers pack off-process blocks into disjoint
   staging slices; the primary performs the physical-rank exchange. Check counts and byte extents before posting.
5. Each destination owner consumes local and received blocks in the original ascending flat-source-owner order, with
   each source's query order unchanged. Local blocks must not jump ahead of earlier remote senders merely because they
   arrived first. Each owner probes, assigns IDs, fills rows, publishes its index and updates its inverted index
   serially within its shard; these operations run concurrently across shards.
6. Publish responses with source/query alignment intact. Process responses and leader marks before starting followers.
7. Repeat for followers, preserving stable matched filtering. Publish deferred self leader then self follower misses.
8. Finalize each shard's graph or apply its fused records, preserving pre-update values and post-insertion
   `scaled_count`.

The insertion order remains cross-owner leader misses, cross-owner follower misses, same-shard leader misses and
same-shard follower misses. Cross-owner includes other shards in the same process. Preserve query distinctness from
XOR with a fixed generator, matched-mark bounds and unique coefficient-update ownership.

Keep the compact existing query representation initially; same-shard position stages remain unencoded. Reading another
local shard's published buffer does not require a second serialization or a transport copy. Decoding may still need its
existing bounded private output. Generalize the decoder's input to read-only spans as needed; retain Task 6's checked
record boundaries, canonical escaped headers and position-before-narrowing checks.

A physical-rank exchange module handles only inter-process requests and staging layout. It is not a replacement
`ShmComm`/`HybridComm` exposing local allreduce/alltoall/barrier methods to shards. Keep source/destination ownership
explicit at the orchestration site. Payload processing stays owner-parallel and source order is independent of timing.

All store growth and ordinary `set_positions` calls remain single-writer **per shard**, not single-writer per rank.
No concurrent index, shared overflow-map mutation, coefficient atomics or graph-format redesign is needed for this path.

## Replay, derivatives and reductions

Publish endpoint snapshots before any owner changes coefficients needed by a peer. Same-process consumers read those
snapshots, not changing live coefficient arrays. Workers prepare remote payloads; the primary posts MPI, independent
local work proceeds, then the primary completes requests before dependent reads. Preserve the current overlap where
safe. Replay uses the communicator-agreed transport choice and the actual peer summaries of every shard, including
multi-peer and empty layouts; do not infer its transport from one shard or one generator.

Preserve whole self pairs, `LayerAngle`, record decisions, predivide/restore ordering, duplicate record indices and
pre-layer-unit derivative formulas. Keep gate order, reverse order and repeated-parameter accumulation order unchanged.
Paring and retained functionals use the same shard geometry and lifetime rules as unpared evaluation.

For this refactor:

- Retain serial per-shard arithmetic and existing standalone dot/scatter contracts, including unsorted/duplicate cases.
- Combine scalar shard contributions in ascending shard ID, then perform the existing physical-rank MPI reduction.
- For vectors, different output positions may be combined independently, but each position folds shards in that order.
- Count replicated Heisenberg identity/core contributions once, not once per shard or MPI rank.
- Keep compiler floating-point flags and the Task 4 contraction workaround until their removal is proven neutral.
  Do not substitute OpenMP floating-point reductions or introduce a new explicit FMA association during the refactor.
- Require deterministic fixed-P/T results and preserve existing bitwise fixtures. Cross-geometry validation compares
  global retained keys and corresponding coefficients with the existing tolerances, plus energies and full gradients.
  Near-cutoff terms and stored zeros count. Observable agreement alone is insufficient.

These are refactor-isolation rules, not permanent restrictions on future numerical optimization.

## Lifecycle and failure handling

Thread-geometry misuse is outside the supported contract. Ordinary allocation, decode and algorithmic failures still
follow the approved validity and distributed-failure policy.

Catch exceptions inside each local/primary phase, before its synchronization point. Every worker reaches the same
checkpoints, observes a checkpoint-stable failure decision and skips subsequent work coherently. A later phase's failure
must not change an earlier checkpoint's branch decision for a slower worker. Do not catch only around an entire shard
operation containing barriers: peers could otherwise wait for a worker that abandoned the protocol.

Preallocate operation-local error slots. Keep MPI request owners and all referenced buffers alive outside the parallel
region so an exceptional exit can join without running a request-draining destructor first. After the join, the caller
invalidates an owner whose mutation began, then aborts a multi-rank communicator or rethrows the original exception for
single-rank/MPI-off operation. Successful paths explicitly wait before reusing buffers. Preserve move-only request
ownership and destination draining on move assignment.

Pre-mutation validation failures leave the object usable. Dependent functionals and copies reject invalid owners;
earlier independent copies stay valid. Destruction is allowed. Constructor/copy failures clean up partially initialized
shard state after joining. Add no rollback, recovery scheduler, custom poisoned barrier or per-phase MPI error-agreement
collective. Integrate error agreement into the explicit team handoffs rather than adding another communication protocol.

Opaque C++ callbacks are not made thread-safe by this redesign. Preserve their supported behavior and conservative
serialization where necessary; builtin cutoffs and basis-change semantics remain unchanged. A traversal that must invoke
a caller-thread callback can use a primary-only phase over stable shard data while other owners wait, with exclusive
access for that phase. This is not a reduced team or resharding. Audit protected extension uses before moving state;
partition-child-factory extensions are explicitly removed below, ordinary virtual hooks are not.

## Public interface and migration

Keep the high-level mathematical Python/C++ operations. Remove `partitions`, `child_factory`, `PartitionChildFactory`,
partition facade helpers and the `monoprop_PARTITIONS` reader without aliases. Add no public `num_threads` or
shard-count constructor parameter. Retain virtual destruction, `clone_`, `update_initial_operator` and the protected
nonvirtual `apply_initial_operator_` hook, adapting their implementation to the rank-level owner.

`mp_op()`, `indexing()`, `graph()` and `graph_data()` remain single-shard-only, now selected by launch-time T=1.
For T>1 they reject access; they must not return shard zero or manufacture a merged store. Update the error terminology
and remedy instead of preserving facade machinery for the old diagnostic. Counts, memory reporting and decoded exports
continue to cover all local shards with their existing rank-local/global semantics. Partial contraction concatenates
shard-local coefficient blocks in shard order.

The source audit found these required migrations:

| Use | Required change |
| --- | --- |
| `MonomialPropagator.inl::evolved_operator_terms` | Iterate each internal shard's index with its evolved coefficients; never call the restricted root accessor |
| `binder.h::evolved_operator` | Preserve decoded Python output and identity handling; raw accessors themselves are not bound |
| `fused_cos_sweep_tests.cpp`, `mpi_fresh_insert_equivalence.cpp` | Replace one-store cross-budget raw-ID comparisons with fixed-geometry shard checks and global-map comparisons |
| `mpi_failure_driver.cpp` | Inject failures into real shard phases and live request lifetimes, without calling T>1 raw accessors |
| Paring, recomputation, Pauli, copy and other single-store tests | Launch with T=1; retain their raw-layout assertions |
| `unit_tests.cpp` and test launch recipes | Replace `monoprop_PARTITIONS=off` policy with launch-time T=1; dedicated team tests use separate fixed-T processes |
| `PropagatorTestAccess.h` | Provide private shard-aware inspection/injection; add no public shard interface solely for tests |

The old sharded runtime already rejects raw access on a facade. The new break is for one-store prototype consumers
that combined `partitions=1` with T>1, and for partition-specific extension/configuration users. Record the C++ ABI
change and rebuild library, generated bindings and installed consumers together. Generated files are not hand-edited.

## Evidence, migration gates and packaging

Retain the production partition baseline at `90d57177c2b0cd93503f88dd931d2f8dd409aa23` separately built. Task 6 is an
experiment checkpoint, not a new performance baseline. Preserve Task 1's campaign/workloads and formal artifacts;
[historical Task 1][task-1] is the authoritative evidence schema and inventory. Source changes to the shared measurement
tool need explicit overlay provenance on both arms, not rewritten historical results.

Run an early integrated checkpoint covering construction, propagation, energy and gradients before deleting the old
runtime. Include actual MPI-off full-machine operation and matched single-thread controls; demonstrate MPI+OpenMP
before completing cutover. The old full-machine result shows why leaf-kernel speedups or a prototype's own T=1 are
insufficient. Keep performance experiments separately authorized and bounded; unused earlier task budgets do not carry.

Final acceptance uses the unchanged 250-cell inventory, its selected size tiers/build modes/geometries, global retained
operator validation and five independent median ratios: runtime, operation peak sum/max and construction peak sum/max.
Each must be <=1.00. Use five fresh pairs initially and five additional pairs on failing cells, as Task 1 specifies.
No geometric-mean masking, profile resizing, MPI-enabled substitution for MPI-off, or implicit memory allowance.
Outer peaks remain diagnostic; the recorded VmHWM limitation does not relax the gates.

Use the already selected c8a.metal-24xl and verify the allocation externally. Task 1 observed D=1 and 96 physical cores;
recheck rather than assume a current full allocation. HPC-family and proper-interconnect multi-node qualification remain
pending. Performance observations need actual shard-worker participation and physical-core placement from the measured
binary, collected outside normal timed paths. Preserve the allocation-before-import measurement fix.

After pre-removal parity, remove the old runtime and direct hwloc discovery/link/install paths, then rerun affected
parity and correctness on the final binary. A transitive MPI/OpenMP runtime dependency on hwloc is allowed. Retain
PUBLIC OpenMP linkage on both object/shared targets and installed `find_dependency(OpenMP REQUIRED COMPONENTS CXX)`.
Qualify wheels, installed consumers, Nix and supported compiler/runtime routes; a platform not run is not passed.

## Post-landing numerical optimization remains open

Once this refactor has landed, the owner wants the option to change **arithmetic operation ordering, combination of
shard contributions, floating-point reduction schemes and FMA policy** for further performance improvements. These
changes may be investigated **individually or in combination**; a beneficial combination need not have individually
beneficial constituent changes.

The initial preservation requirements isolate the runtime refactor. They are not a permanent promise of one reduction
association or a ban on explicit FMA. Do not make these follow-up experiments prerequisites for landing the refactor,
and do not restart the superseded one-store Task 9 as part of this plan.

A separately scoped follow-up should retain the landed refactor as its control, specify the numerical/reproducibility
contract it proposes, and measure end-to-end runtime and memory on matched workloads/resources. Evaluate retained keys,
coefficients, energies and full gradients, particularly cancellation, near-cutoff decisions, deep derivatives and
vanishing cosine. Include relevant compiler flags and hardware/software FMA behavior. Differences in bits are not by
themselves a veto on an explicitly reviewed new association; silently changing tolerances or weakening correctness gates
is not authorized. Record the adopted policy and its evidence, whether the winning change is isolated or combined.
Permission to keep this option open is not permission to run or adopt an optimization without its own approval.

## Non-goals for this refactor

No new concurrent hash table, numerical association, dynamic load balancing, variable/heterogeneous team support,
work-stealing runtime, GPU path, graph-format redesign, global dense operator mirror, affinity service or permanent
legacy/new selector. Temporary development selection is permitted only to compare and migrate the implementations;
it is removed before acceptance. The companion plan defines gates and the explicit post-landing follow-up boundary.

[task-1]: ../plans/2026-09-18-rank-local-openmp.md#task-1-freeze-baselines-and-repair-measurement-trustworthiness
