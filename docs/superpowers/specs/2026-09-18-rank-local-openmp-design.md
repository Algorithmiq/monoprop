# Rank-local OpenMP parallelization design

## Status and authority

This is the design accompanying the implementation plan at `../plans/2026-09-18-rank-local-openmp.md`. It records the
direction discussed with the owner; it is not a claim that the proposed implementation meets its performance gates.

Source inspected for this owner-requested current-main refresh: `origin/main` at
`90d57177c2b0cd93503f88dd931d2f8dd409aa23`, present in rebased `refactor-parallelization` at
`35f359f095fa3f1d628ca2fdc66c93fef6ea7564`. Only CONTEXT.md and the two planning documents differ between those commits.
EXISTING means this refreshed source snapshot; PROPOSED means future implementation.

The original planning publication was `38eb13d80a95dd1ac60e762f14f68c9a5925f755` in `refactor-parallelization`.
This source refresh preserves the subsequent owner-approved policy revisions and uncommitted edits. The owner approved
mandatory OpenMP, removal of direct hwloc and partition configuration without deprecation aliases, the associated
API/ABI and capacity breaks, environment-only construction-time thread budgets, and the failure/calling contracts below.
These are design approvals, not implementation or publication authority for this revision.

**Architecture decision reopened:** the owner now requires Task 0, a small MPI-off index mini-app comparing per-thread
stores, one shared non-concurrent store and one shared concurrent index. Its evidence can overturn the one-store choice.
The one-store design and Tasks 1–13 below are conditional, not an instruction to proceed regardless of that result.
After Task 0, stop for the owner's architecture decision and reconcile both documents before any downstream execution.
The follow-up interview approved a staged index-phase study and required full-machine MPI-off library acceptance,
including selected large profiles. These requirements are fixed before results, not deferred to the winning design.
This amendment authorizes planning only, not mini-app implementation, measurements or remote expenditure.

The owner selected one AWS EC2 `c8a.metal-24xl` for implementation/testing and the initial acceptance campaign.
Production deployments are dual-socket HPC nodes using Intel Sapphire Rapids or AMD Zen3/Zen4/Zen5. Hardware SMT may be
on or off, but the recommended configuration always uses one worker per physical core, never extra SMT-sibling workers.
Passing the frozen EC2 campaign is sufficient for initial acceptance; production-HPC performance remains unverified
until separately qualified. Multi-node qualification is pending a proper HPC cluster/interconnect, not a second EC2
instance. Observed EC2 topology/software and representative workload sizes still need recording/approval before freezing
that campaign. The owner approved baseline-only, unscored feasibility calibration before the freeze, once execution is
separately authorized. Small/medium profiles cover the full geometry sweep plus full-machine MPI-off operation;
selected large profiles cover the two full-node MPI decompositions and full-machine MPI-off operation. This revision
authorizes no runtime work or implementation.

**Source-drift audit completed:** the owner-requested refresh replaces the superseded
`290112c8289ab8015eb9a2c7651ac409839c5a88` pin with the main revision above. The audit covered all 43 changed paths,
including routing/MPI, construction, Python parameter-axis semantics, tests, version/build tooling and documentation.
The previous d93d63f source-refresh blocker is resolved by this audit and the corresponding plan updates, not by the
rebase alone. Both measurement arms must start from this refreshed pre-OpenMP engine; do not reuse superseded-engine
samples as its baseline. Baseline build/calibration authorization, campaign freeze and implementation approval remain
separate gates. Later semantic source drift still requires review against the plan's refreshed anchor table.

### Findings incorporated from current main

- Routing walks a width-bound, full-rank GF(2) basis; it no longer uses a transposed plane table. Linear geometry
  requires power-of-two R and log2(R)<=min(2*NumModes,64). A deterministic bounded redraw can change the basis
  relative to the superseded source, so old physical-rank maps are not a reference for this refactor. Preserve the
  current map/seed.
- MPI replay selects a communicator-agreed boolean via mpi::routes_pairwise, not a recomputed bit-count selector.
  Exact agreement, communicator-scoped caching, sparse-plan validation and multi-peer replay must survive cutover.
- Construction exchanges accept WindowVec blocks only. LayerBuildEngine receives both plan and window explicitly;
  each FusedScanResult stream carries its own window. IncomingProbe exposes sender_index, not sender_slot.
- Both replay Ticket and construction PendingAlltoallv own and drain their requests. Failure guards must protect both
  lifetimes, including move-assignment and destruction, rather than accidentally wait during exception unwinding.
- Python incremental Heisenberg graph builds now renumber the parameter axis to match the equivalent composed circuit;
  preserve this behavior, post-call seed interpretation and the new regression tests for both Python propagators.
- Packed storage, fixed32 indices, streamed initialization, stable-gradient formulas, benchmark families/windows and
  centralized build recipes are unchanged from the inspected source. The plan retains their gates and adds the current
  transport/routing/axis regressions. Source inspection is not runtime, race-freedom or performance evidence.

## Objective

First investigate MPI-off index scaling with Task 0 before choosing the ownership architecture. The conditional
proposal is to replace thread-owned operator partitions and their in-process communication protocols with one compact
operator store per MPI rank, operated on by OpenMP worksharing. Preserve mathematical features, memory footprint and
performance. Simplification without regressions is the objective; neither one-store variant is a predetermined winner.
Required MPI-enabled acceptance cells keep each rank's CPUs within one NUMA domain; MPI-enabled full-node cross-NUMA
L2a remains a diagnostic. Separately, a real MPI-disabled build using the full machine is a required library gate,
even across NUMA domains. A NUMA-local win cannot excuse an MPI-off full-machine regression. The user chooses
processes P and threads per process T and configures the launcher and OpenMP runtime to bind/pin correctly. The library
neither discovers topology nor repairs placement.

## Preliminary architecture experiment (Task 0)

Keep the mini-app an index-phase pipeline experiment, not a second propagation engine: one C++ translation unit,
standalone CMake file and short README under tools/index-miniapp/. Reuse the current OperatorIndex and monomial hashing
headers. No library
build/import, MPI, ShmComm, graph, coefficients, inverted index, physics/cutoff logic, trace capture/replay service,
custom scheduler or benchmark framework. A few local adapter functions, flat batch buffers, self-tests and CSV output
are sufficient. Stop and reduce scope if implementing the experiment requires a production storage refactor.

Use one process and one OpenMP team of T workers, including its primary thread. Compare exactly three variants:

| Variant | Stores for one global input | Lookup and publication |
| --- | --- | --- |
| per-thread | T disjoint current OperatorIndex stores, not T copies of all keys | Each owner probes and inserts serially; owners run in parallel |
| shared-serial | One current OperatorIndex | Parallel frozen probes; caller assigns IDs, initializes rows and inserts |
| shared-concurrent | One packed row store and one Boost concurrent index | Parallel probes; caller assigns IDs/initializes rows; parallel index publication |

The per-thread variant models current storage ownership, not the legacy runtime's performance. All variants use the
same OpenMP harness and global keys/queries. Ownership transfers and packed-row initialization belong in batch timing;
this is not an isolated container-throughput contest. Shared rows remain caller-initialized, so a concurrent index
need not fix a row-initialization bottleneck. Keep the real packed-row representation, batched position lookup, cached
32-bit hashes, exact collision checks and fixed32 row references. For Boost use stable row handles and heterogeneous
query lookup, not persistent full monomial keys. A row-only use of OperatorIndex may retain its constant empty
16-slot table, accounted as harness overhead; never populate it or allocate a second size-dependent index. This narrow
mini-app convenience is not an approved production layout. If the supported Boost floor cannot express the adapter,
report that limitation rather than building another table or silently changing dependencies.

Exercise frozen lookup and repeated probe → ordered missing-ID assignment → row growth/fill → publication → join
batches. Keep rows immutable during concurrent index operations; do not benchmark simultaneous readers/inserters when
the proposed engine uses separate phases. Include a small flat-buffer owner-bucketing/result-scatter path for the
sharded variant and report its cost separately. Do not simulate the old message transport or claim those transfers are
free. Local IDs may differ between sharding geometries; decoded keys/results and within-geometry ID order must agree.

Stage the study. The first performance pass uses 128 modes, inline length-six keys, and exactly three cases: frozen
all-hit lookup, growth with 10% misses, and growth with 100% misses. Use two working-set sizes, fixed global row/query
counts across thread counts, deterministic synthetic position lists and natural growth. No trace corpus or synthetic
physics. Keep small collision, spilled-row and row-growth correctness checks; defer wider-key, spilled-row performance,
small-batch and other distribution sensitivities until initial results justify an owner-approved follow-up.

Use a compact physical-core sweep within one NUMA domain and across the allocated machine. Both series matter; do not
replace the full-machine MPI-off series with an MPI run. Verify placement externally and use identical external memory
policies across variants. Natural first-touch differences are real consequences of storage ownership/construction and
count in the comparison, not noise to normalize away. Report locality separately from synchronization effects. An
explicitly matched interleaved-memory rerun may diagnose a result, but cannot replace it. No app-owned affinity service.

Report complete-batch time, phase times, throughput, speedup against each variant's own T=1 and the per-thread variant
at the same T, actual workers, per-store imbalance, and fresh-process peak RSS including construction/growth. State
which costs are omitted, which byte counts are estimates, and when memory placement or noise prevents a conclusion.
Call exact key/ID lookup checks index-result agreement, not retained-operator equivalence: the latter also requires
coefficients. Synthetic index results cannot establish propagation/replay/gradient scaling or full-library parity.

Task 0 has an approved calibration/measurement limit of 2 hours total, including each process's initialization, input
generation and validation, and phase-profile runs. Setup/build time and costs are accounted separately and need their
own authorization. Cap each fresh process's peak RSS at min(8 GiB, 25% of observed usable allocation RAM), including
setup/growth/checking. Record a fixed RAM basis and byte ceiling before trials; retain consumed/remaining time across
resumption. Stop with incomplete evidence if a limit prevents completion; do not silently extend the budget or claim
DRAM scaling if the permitted working set cannot establish it. These are external study limits, not a library clamp.
Approval of limits does not authorize execution.

Task 0 ends with raw results and an owner decision: retain a shared-store path, retain/reconsider sharding, or declare
the evidence inconclusive. Do not auto-select the fastest lookup kernel or invent a fourth architecture. Any revised
production design needs an updated spec/plan and approval; it must preserve the now-required MPI-off acceptance
coverage and strict library gates unless the owner explicitly revises them. Task 0 neither consumes Task 1's separate
two-hour baseline pilot nor supplies its calibration or acceptance samples. The later Task 11 trial is integration
validation of a retained candidate, not a second container search.

## Global constraints

The storage constraints describe the conditional one-store path in Tasks 1–13. Task 0 alone permits the explicit
per-thread-store experiment, not production shards. Full-machine MPI-off measurements are required both in Task 0
and in full-library acceptance, including cross-NUMA placement. Reconcile the storage design with the owner's
post-experiment decision without silently weakening that coverage or any numerical/performance gate.

- C++23; retain the GCC 14 / Clang 18 minimum compiler versions and Python 3.11 floor.
- MPI remains optional and OFF by default; retain working non-MPI wheels.
- OpenMP becomes a required implementation dependency for the replacement runtime; support Linux x86_64, Linux aarch64,
  and macOS builds covered by the current project.
- Each independent propagator instance owns one packed operator store per MPI rank; threads are never communicator
  ranks. Preserve independent copies, immutable graph-core sharing, retained coefficient/state snapshots and pared
  graphs.
- Remove monoprop's direct hwloc dependency and topology/affinity machinery; do not replace it with another
  library-owned hardware-discovery or pinning mechanism.
- Users choose P processes and T threads per process and configure resource allocation, binding and pinning; monoprop
  does not clamp T to physical cores or validate NUMA placement.
- SMT use is not recommended: allocate at most one monoprop worker per physical core, including on SMT-enabled hosts.
  Enforce this through user launch/binding settings and acceptance evidence, not a library hardware policy.
- Preserve packed rows, keyless indexing through row references, fixed uint32_t TermIndex, sparse state storage, and
  inverted-index cosine recomputation. A bounded Boost concurrent-index experiment may replace the index, not rows.
- Do not introduce hidden operator shards, a dense duplicate store, a second persistent full-key store, per-thread
  operators, or full-vector coefficient double buffering. Select any concurrent index only after correctness/memory/time
  evidence; its container name is neither a rejection criterion nor proof of suitability.
- Required MPI-enabled cells keep each rank within one NUMA domain; MPI-enabled cross-NUMA L2a is diagnostic-only.
  Actual MPI-disabled full-machine operation is required for small/medium and selected large profiles, even across
  NUMA domains. Diagnostic results cannot replace, excuse or average away any required cell; correctness still applies.
- Preserve Majorana and Pauli bases, Heisenberg and Schrödinger pictures, cutoffs, graph build/replay/paring, gradients,
  partial contraction, copying, and initial-operator updates.
- Final MPI calls execute on the MPI-initializing thread, outside library-created OpenMP regions. Request/validate at
  least MPI_THREAD_FUNNELED, accepting higher provided levels without relaxing the caller rule. Retain
  MPI_THREAD_SERIALIZED during legacy Hybrid coexistence; lower the minimum only with its removal in Task 12.
- No exception escapes an OpenMP structured region; distributed operations fail coherently rather than leaving peers in
  collectives. Enforce invalidation after failed mutation; dependent functionals cannot reuse an invalid owner.
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

This section describes the one-store candidate, conditional on the Task 0 decision.

Each independent propagator has its rank-local `MPOperator`, primary `MPGraph`, and ordinary MPI communicator. This is
not a process-wide singleton: copies own independent packed stores; retained functionals may own coefficient/state
snapshots and pared graphs without cloning the packed store. Keep existing C++ owner-lifetime requirements and Python
owner retention. The prohibition on redundant stores/buffering concerns new parallelization machinery, not these
existing mathematical objects. A small header-only worksharing helper
supplies deterministic block IDs, explicit thread budgets, and exception collection. It owns no threads, queues,
communicators, or operator data. OpenMP owns worker teams.

Local kernels read an immutable structure or write preallocated, non-overlapping output ranges. The controlling thread
resizes shared containers and publishes structural changes between phases. Initially, insertion into the keyless hash
table and inverted-index maintenance remain single-writer. This is an explicit performance gate, not an assumption that
insertion is negligible.

### Bounded concurrent-index experiment

The previous blanket exclusion of boost::concurrent_flat_map is superseded, not a measured rejection. Task 0 first
compares the three ownership/index designs in isolation. If retained by the owner, compare the packed keyless path with
that same bounded Boost variant in Task 11 after the full-library baseline campaign is frozen. This later trial tests
integration and the unchanged acceptance gates; it does not repeat container selection or bypass Task 0. A direct
`concurrent_flat_map<Monomial,TermIndex>` duplicates full keys; investigate stable row handles and heterogeneous lookup
into packed rows. Container synchronization does not protect coefficients, overflow storage, inverted indices or
graph packing. The
[Boost concurrency guide](https://www.boost.org/doc/libs/1_89_0/libs/unordered/doc/html/unordered/concurrent.html)
describes visitation and blocking rehash; neither is a drop-in substitute for the current batched query/result contract.

Assign IDs in canonical order, reserve all needed storage on the caller, initialize rows before publication, then join
before dependent reads. Only the index's insertion/publication is allowed to become concurrent in this experiment;
other shared structural mutation remains separately governed. Prove exact collision handling, stable external-row
lifetimes, per-query result alignment and clone ownership. Never store query-buffer views as long-lived keys, allocate
IDs by worker arrival order, or add a second persistent index beside the reference implementation. Compare separate
build variants, not two resident stores or a permanent backend selector. Measure lookup/insertion, growth peaks and
end-to-end construction plus the full affected acceptance cells; an insertion microbenchmark alone cannot select it.
Task 11 specifies the bounded trial and its tests. Keep or reject it from evidence, without weakening parity gates.

### Kernel and phase sequencing

Only recognized pure `LengthCutoff`/`SupportCutoff` targets may run in worker traversal initially. Ordinary Python and
C++ constructors accept built-in cutoff configuration and basis-change data, not arbitrary cutoff functions. Opaque
predicates are reachable through the existing protected C++ `cutoff_fn_` extension point and lower-level helpers;
retain their conservative serial path without inventing a Python callback API or an artificial callback benchmark.
Current basis-change closures capture data by value; audit their callees rather than claim a mutable capture exists.
Their initial serial path is a conservative implementation choice, not proof that they are unsafe or a performance
exemption. Actual lazy cosine/parity caches still require caller-side preparation.

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
MPI replay transport behavior unchanged. EXISTING: `Evolution.cpp::begin_flat_exchange` passes
`mpi::routes_pairwise(comm)` to `detail/mpi/Exchange.h::post_flat_alltoallv(..., bool pairwise=false)`. This
rank-uniform boolean selects Isend/Irecv or MPI_Ialltoallv; a local layout must never select the transport. Plain MPI
walks a dense PeerPlan and posts only nonempty remote legs, copying self locally. HybridComm folds every partition's
send/receive peer summaries: one compatible peer narrows the set, while multiple/disagreeing peers retain every peer
on the SAME pairwise transport. Empty traffic must not send one rank into a collective. Preserve multi-peer/hand-built
layer replay and S=1 sparse on-wire replay; a new generator-derived replay plan is outside this refactor.

EXISTING: `Pairwise.h::agree_routes_pairwise` checks raw routing mode, partition count and seed exactly with one
MPI_MAX allreduce over values and complements. `MPICompat.cpp::routes_pairwise` caches ordinary-MPI results as a
communicator attribute with no copy callback; new/duplicated communicators agree independently rather than inheriting
stale raw-handle cache entries. HybridComm agrees during construction; Shm/non-MPI/single-rank ordinary MPI return
false. `MPIUtils.h::check_routing_agreement` delegates to this path. First use can be collective, and even a cache hit
queries MPI attributes: all of it belongs on the caller, never workers. At S=1 agreement still covers routing/seed,
not per-object thread budgets. The error type is `routing::RoutingDisagreement`.

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

EXISTING: `FusedScanResult<N>` in `Scan.h:208–226` has six `WindowVec` streams plus `leader_self`/`follower_self`,
with no separate result.window member. `build_layer` supplies the same explicit plan/window to scan and engine;
value streams remain empty when capture is disabled. `SlotWindow::indices()` yields typed WindowIndex values in order;
they are not flat slots, including when base is nonzero. Resolve a sender as `pr.window.slot(pr.sender_index(g))`.
Preserve the precomputed `gen_shift`, variable `QueryWire` records and `QueryForm`; self stages are positions, never
serialized wire. Production `begin_alltoallv`/`wait_into` accept WindowVec blocks, not vector-of-vectors; the latter's
adapter is test-only in `cpp/tests/SlotBlocks.h`. Known receive counts remain flat-slot indexed and must cover the
window's stop; short arrays throw rather than silently truncate. Sparse PeerPlan validation rejects invalid geometry
or shifts before indexing/posting. Do not remove these checks when retiring Hybrid.

Merge self stages by their logical lengths and rebase position offsets while preserving source/value alignment.
Private query buffers have at most
O(T*window.count) empty-container metadata, not O(number_of_bitmap_blocks*R). Per-record payload across those buffers
must be O(total emitted records), never O(T*operator_size). Release/move drained temporary buffers before starting the
next memory-heavy phase. Before parallel decode, the caller checks variable record boundaries, canonical escaped headers
(`h.bits == QueryWire<N>::header_bits_for(h.k)`) and position-count prefix sums, sizes arrays once, then workers decode
into disjoint spans. Frozen probes retain `find_batch_positions` and cached hashes; the initial reference publication is
serial `set_positions` → `bulk_insert_hashed` → reindex. The bounded Task 11 index trial must preserve these phase and
row-ID contracts even if hash publication becomes concurrent. No per-rank or full-store dense monomial reconstruction is
allowed. Existing bounded paired-only scoring may reconstruct its one required monomial; do not turn that exception into
a batch/store. Task 6 defines the checked preparation sequence.

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
- Preserve Python's incremental parameter-axis contract (`src/monoprop/monomial_propagator.py::build_graph`):
  build_graph(a) then build_graph(b) denotes b+a in Heisenberg and a+b in Schrödinger. Heisenberg moves each new block
  to the low indices, lifting the existing indices; seed_parameters uses the post-call axis. Python rotates that seed
  for the low-level build and relabels after success. Gate arrival IDs do not change, and multiple monomial layers of
  one gate keep their shared index. C++ build_graph still consumes parameter_mapping as given; do not move Python's
  renumbering into it. Preserve tests/test_circuit.py axis, off-build-angle value/gradient, Pauli and seeded-truncation
  regressions across budgets. This behavior is already in the baseline, not a new OpenMP compatibility break.
- Near-cutoff tests compare retained operator terms, not just final energies.
- Preserve streamed `for_each_paired_monomial` initialization order and local reservation/cardinality checks in
  `MonomialPropagator.inl:126–151,197–227`. At S=1 estimates use real ranks, never T. Reject impossible cardinality
  before workers; never materialize the global paired basis per rank.

## Resource ownership and deployment

### Selected implementation/test platform

AWS documents `c8a.metal-24xl` as AMD EPYC 9R45, 96 physical cores/96 vCPUs, one thread per core and 192 GiB RAM in its
[exact-size specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html#co_hardware). The
[C8a page](https://aws.amazon.com/ec2/instance-types/c8a/) identifies fifth-generation EPYC (Turin) and explicitly
states that there is no SMT. Do not call a 192-worker run on this instance an SMT-on test; it would oversubscribe its
physical cores. The exact socket/NUMA layout was not established by the reviewed AWS specifications and must be observed
on the actual allocation before selecting rank/thread geometries. Do not assume eight NUMA domains or claim a
96-core process is NUMA-contained. Verify actual placement for required MPI-off full-machine operation and classify
MPI-enabled cross-NUMA L2a as diagnostic; both still require distinct physical cores.

These published specifications are not measurements or evidence of parity. C8a does not establish performance on
Sapphire Rapids, Zen3/Zen4 or HPC hosts with hardware SMT enabled. Those remain separate qualification targets using the
same one-worker-per-physical-core recommendation; no SMT-sibling worker sweep is required. Initial acceptance is scoped
to the frozen single-instance C8a campaign. Multi-node qualification requires a proper HPC cluster/interconnect and
remains pending; do not substitute a cloud-network experiment or claim HPC fabric scaling from local MPI.

### User-owned allocation and placement

- The user chooses P processes and T threads per process for the allocated physical cores. SMT use is not recommended:
  do not count SMT siblings as extra worker capacity. Set P with the launcher and normally T with
  `monoprop_NUM_THREADS=T` at launch.
  When it is unset, use the OpenMP runtime default, normally configured by `OMP_NUM_THREADS`. No runtime reconfiguration
  API or environment reload is provided for an existing object.
- The user is responsible for binding/pinning processes and threads to the correct hardware using, for example,
  `OMP_PLACES` + `OMP_PROC_BIND`, `srun --cpu-bind`, and/or `mpiexec` options. Scheduler allocation, rank binding and
  OpenMP places must agree: binding a rank to one core while requesting many threads does not allocate additional cores.
  Required MPI-enabled acceptance keeps each rank's workers within one NUMA domain; avoid overlap with other ranks.
  MPI-enabled cross-NUMA L2a remains diagnostic. Required MPI-off full-machine operation spans the allocated domains
  in one process and must verify distinct physical-core use without claiming NUMA containment.
  For MPI deployments, advise at least one MPI process per NUMA domain used by the allocation; multiple processes may
  share a domain using disjoint CPU allocations. The production nodes commonly have four domains per socket, hence
  eight per full dual-socket node, but use the actual configured topology rather than a hard-coded count. Do not infer
  the EC2 topology from that HPC convention. Partial-node runs cover only their allocated domains.
- Monoprop validates thread-count representability, not hardware suitability. The executable does not discover physical
  cores, enforce the documented no-SMT-worker recommendation, redistribute CPUs, change affinity, validate NUMA
  containment or prevent oversubscription. OpenMP's
  own limits and small-work/nested serial paths may use fewer workers; they do not certify adequate resource allocation.
- Delete `CpuTopology` with the partition runtime and remove direct hwloc discovery/link/install requirements. Do not
  retain hwloc for mandatory diagnostics. Benchmark operators still verify placement with runtime/launcher diagnostics
  and external tools; those are evidence requirements, not library dependencies. MPI or the OpenMP runtime may
  independently depend on hwloc.

### Physical-core launch recipes

These are documentation templates, not executed/qualified launch commands. T is the number of assigned physical cores
per rank and the total OpenMP team size, including its primary thread. Set both launch-time budgets explicitly:

```bash
export monoprop_NUM_THREADS="$T" OMP_NUM_THREADS="$T"
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_DYNAMIC=FALSE
```

A core place may contain both hardware threads of a physical core. With exactly T distinct core places and T workers,
`close` assigns one worker per place; it does not launch a worker on each sibling. `OMP_PLACES=cores` alone is not a
thread-count limit: requesting more workers than places can share cores. Verify the runtime's actual places and worker
masks, and remove conflicting runtime/site affinity overrides. Do not recommend SMT with `OMP_PLACES=threads`.
See the [OpenMP places](https://openmp.org/spec-html/5.1/openmpse62.html) and
[close binding](https://openmp.org/spec-html/5.1/openmpsu41.html) definitions.

**Open MPI 4.1/5.x:** for K=1 or K=2 ranks per participating NUMA domain and R total ranks, each domain must provide at
least K*T available physical cores. Obtain a matching allocation and select R/K/T consistently with its actual topology:

```bash
mpirun -n "$R" --map-by "ppr:${K}:numa:PE=${T}" \
  --bind-to core --nooversubscribe --report-bindings ./application
```

Processing elements are cores by default. Do not use `--use-hwthread-cpus`, `HWTCPUS` or binding-overload options.
Open MPI 5 documents `CORECPUS`; it is absent from the 4.1 manual, so the shared example omits it and requires the
core-counting default without conflicting overrides. `--nooversubscribe` checks process slots, not proof of disjoint
physical-core ownership. Verify binding reports: T distinct cores per rank, no physical core shared across ranks, and
one NUMA domain per rank. Do not assume a partial allocation exposes exactly the desired domains. Sources:
[Open MPI 4.1](https://www.open-mpi.org/doc/v4.1/man1/mpirun.1.php) and
[Open MPI 5.0](https://docs.open-mpi.org/en/v5.0.0/man-openmpi/man1/mpirun.1.html).

**Slurm:** the following single-node template requires matching job/step CPU requests, task/affinity support and a
site-approved MPI plugin (select MPI_PLUGIN with site guidance; `srun --mpi=list` lists installed choices). It is valid
only when observed site ordering/allocation makes each resulting T-core rank mask NUMA-local:

```bash
srun --nodes=1 --ntasks="$R" --ntasks-per-node="$R" --mpi="$MPI_PLUGIN" \
  --cpus-per-task="$T" --threads-per-core=1 \
  --distribution=block:block --cpu-bind=verbose,cores ./application
```

`--threads-per-core=1` restricts task layout to one thread/core; explicit core binding overrides its implied thread
binding. Core masks may still include sibling CPUs, but the OpenMP settings above use one worker/core. Do not combine
these explicit options with `--hint=nomultithread`: Slurm documents conflicts/overrides. That hint is an alternative,
not an extra safeguard. Sources: [srun](https://slurm.schedmd.com/srun.html) and its
[CPU-binding options](https://slurm.schedmd.com/srun.html#OPT_cpu-bind).

Slurm's block distribution describes nodes/sockets/cores, **not NUMA domains**. Neither this command nor adding a third
`block` field proves containment on four-domains-per-socket machines. If masks cross domains, use a site-supported
NUMA-aware launcher or replace automatic binding with `--cpu-bind=verbose,mask_cpu:$MASKS`. Build hexadecimal masks from
observed CPU/sibling topology: T distinct physical cores within one domain, with disjoint physical cores across
ranks. Slurm requires every available CPU on the node to belong to the **job step** for explicit masks to be honored;
job-level exclusivity alone is insufficient, and a reduced `--cpus-per-task` step is not automatically eligible. Resolve
that allocation with the site before using masks. Mask lists repeat by node-local rank, so identical masks cannot be
assumed valid on differently numbered/configured nodes. Binding two ranks to the same entire locality domain does not
partition its cores. No automatic topology/binding fallback is added to monoprop.

Archive rank/worker masks, physical-core sibling identities and, for required MPI-enabled cells, NUMA containment.
For required MPI-off full-machine runs and diagnostic MPI-enabled cross-NUMA L2a, record the full allocation and actual
multi-domain placement explicitly; do not claim single-domain containment or substitute multiple MPI ranks.
These recipes document HPC deployment; they do not convert pending proper-interconnect multi-node qualification into
completed evidence.

## Thread selection and compatibility policy

The replacement stores an immutable `parallel::Options` per propagator; copies and retained functionals preserve it.
The supported deployment is launch-configured: capture the request exclusively at construction, never reread it during
operations, and provide no setter or hot-reconfiguration mechanism. A worksharing region may use fewer workers than the
budget; this is not a budget change. Thread count never changes ownership or graph layout. MPI ranks may use different
launch-time budgets.

At cutover:

1. Remove `partitions` from the C++ constructor and low-level binding, `PartitionChildFactory`, partition-specific
   helpers/error types, and every partition configuration environment variable. There is no deprecation window, alias,
   compatibility warning shim or new public `num_threads` constructor argument. Existing `monoprop_PARTITIONS` has no
   effect in the candidate: do not retain a getenv reader merely to diagnose it. Remove it from launch scripts/tests.
2. A present `monoprop_NUM_THREADS` must contain a positive decimal integer representable as `int`. Reject empty,
   malformed, zero, negative and overflowing values with an actionable configuration error; never silently fall back.
   Only an unset variable selects `omp_get_max_threads()`. Preserve the chosen positive value exactly, without hardware
   clamping. Replace the old cached permissive parser for this setting; routing settings retain their own contract.
3. Do not independently parse `OMP_NUM_THREADS`. OpenMP supports list-valued settings and application initialization;
   its runtime default is the fallback at any MPI rank count. If it is also unset, use that default without promising
   hardware suitability. Do not preserve automatic physical-core discovery or a multi-rank default of one thread.
4. `OMP_PLACES` and `OMP_PROC_BIND` control placement; monoprop never modifies affinity. OpenMP limits/dynamic-team
   adjustment may reduce actual teams; correctness must not depend on receiving exactly T workers. Verify actual
   placement and participation externally for performance acceptance, not during ordinary construction.
5. Keep the virtual destructor, virtual `update_initial_operator`, its protected nonvirtual `apply_initial_operator_`
   helper, and independent virtual `clone_` hook (`MonomialPropagator.h:288–306`). The owner accepts immediate C++
   source/ABI breaks for partition removal in this 0.x library; record release notes and rebuild library/bindings.
   High-level Python mathematical signatures remain unchanged; `_core` loses `partitions` and gains no thread keyword.
6. Returned term/coefficient ordering may differ from old partition-block order. Validate identical global retained term
   keys and coefficient associations within existing tolerances, not coefficient-only sorted lists. Equal energies and
   gradients alone are insufficient. Keep candidate ordering deterministic across budgets at fixed rank geometry.
7. Preserve boolean linear/splitmix routing, default linear rejection of unsupported R, generator shifts and peer rules.
   Keep the full-rank basis draw and width bound log2(R)<=min(2*NumModes,64); the bound uses the C++ template width.
   The owner accepts the splitmix ownership change from `floor((hash % (R*S))/S)` to `hash % R`; do not emulate S with
   hidden shards. Keep `monoprop_ROUTING` and `monoprop_ROUTE_SEED`: these control MPI routing, not thread partitions.
   Preserve the seed's strict decimal uint64 parser (no sign/whitespace; unset/empty selects its default), separately
   from the proposed present-empty thread-budget error. COMMROUTE reports ranks_per_coset/rank_cosets, not measured
   idle ranks; a full-rank basis or spanning generator set alone does not prove an actual retained operator is balanced.
   Test default-linear R=3 rejection separately; successful R=3 suites/benchmarks explicitly use splitmix. Compare
   global maps and per-rank loads/peaks at identical routing/seed/rank geometry.
8. MPI-using entry must run on MPI's initializing thread, even with SERIALIZED or MULTIPLE support. This deliberately
   removes the old freedom to move the controlling caller between threads while serializing calls. It does not add a
   dispatcher, change the GIL policy or permit overlapping dependent-state calls.

Development-only coexistence uses the EXISTING explicit `partitions=1` argument plus `comm.kind==mpi::Comm::Kind::Mpi`
to select the one-store prototype and capture its environment budget. Legacy Shm/Hybrid children and multi-partition
facades retain serial kernel options. No new constructor keyword, environment switch or backend selector is needed.
Task 10 removes that transitional selection and captures the budget for every independent instance. Tests use explicit
internal kernel Options or separately launched processes, not a production budget setter.

These compatibility policies are approved. Execution and final removal still require implementation authorization and
all earlier gates; they are not permission to bypass performance or packaging failures.

## Safety and failure policy

The worksharing helper catches each worker exception, joins the region, and rethrows on the calling thread. Caller-side
distributed phase guards print the underlying failure and call MPI_Abort when communicator size exceeds one. Catch
failures inside the lifetime of both `Exchange.h::Ticket` and `MPICompat.h::PendingAlltoallv`, before stack unwinding
can wait on peers that never posted; an outer catch alone is insufficient. Both handles are move-only and self-draining;
move assignment drains the destination before adopting incoming requests. Keep request/buffer ownership together and
never resize live request storage. `Pairwise.h::sparse_pairwise` pre-sizes from actual nonempty remote send/receive
legs, not a caller-supplied active-leg estimate. Successful paths explicitly wait as before. For single-rank operation,
rethrow the original exception. Enforce post-mutation invalidation on the affected propagator: later
state-consuming/mutating calls, copy/clone and dependent retained functionals reject it before touching state.
Existing independent copies remain valid.
Pre-mutation validation failures leave a valid object usable; destruction remains safe. There is no rollback or reset
that revives a failed object. Task 3 defines the invalid flag, entry checks and failure-injection tests.

No rank-local early return may bypass a collective peers will enter. In the final runtime, validate provided support
with MPI_Query_thread, including externally initialized MPI, against MPI_THREAD_FUNNELED. Higher levels are accepted,
but the owner-approved contract remains one controlling caller per rank: the thread that initialized MPI, outside active
host OpenMP teams. Serialize dependent-state use and host MPI calls. A different thread making calls one at a time is
not FUNNELED. Check caller identity before MPI-using entry, including constructors, retained functionals and cleanup.
Wrong-thread entry is a host-contract violation, not an ordinary distributed operation failure: diagnose and fail fast
locally without calling MPI_Abort or querying communicators from that thread. Task 3 defines the launcher-supervised
negative tests. Valid-caller operation failures retain the join/abort/invalidation rules above.

Keep the legacy SERIALIZED requirement while Hybrid workers still call MPI. Task 12 removes that path and lowers the
requested/required level, updating tests in the same change. Never reinitialize or downgrade host-owned MPI. mpi4py
normally requests MULTIPLE; record actual provided support rather than claim FUNNELED from the request alone. Numerical
helpers may serialize when nested, but public distributed entry from an active host OpenMP team remains unsupported.
Add no process-wide locking service, caller-thread dispatcher or nested-parallel execution machinery. See the
[MPI initialization/thread contract](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node267.htm), including
MPI_Is_thread_main; identifying a wrong caller does not permit that caller to communicate or invoke MPI_Abort.

Lazy initialization, allocation, and cache mutation must be outside worker bodies unless storage is strictly
worker-private. Do not assume `const` means thread-safe: both the inverted index and parity cache have lazy mutation
today.

## Performance acceptance and stopping

### Run the existing ladder on both revisions

[benches/LADDER.md](../../../benches/LADDER.md) is the measurement roadmap, not merely a source of historical labels.
Reuse its benchmark functions, model builders, CLI sizing controls and report outputs. The additional driver is a thin
execution/validation wrapper for the agreed evidence requirements, not a replacement benchmark suite.

Here use ladder notation: N nodes, C usable physical cores/node, D NUMA domains/node, R ranks/node and P old partitions
per rank. The candidate uses T OpenMP workers/rank; matched comparisons use T=P. For balanced allocations:

| Rung/control | Single-node candidate shape | Role |
| --- | --- | --- |
| L1 | R=1, T=1 | Required single-thread control for small/medium profiles |
| NUMA-local shared-memory control | R=1, T=C/D | Required small/medium control; do not call it full-node L2a |
| MPI-only control | R=D, T=1 | Required small/medium control, not a fixed-core comparison with L1 |
| L2b | R=D, T=C/D | Required production shape: one rank/domain |
| L2b decomposition variant | R=2D, T=floor(C/(2D)) | Required two-ranks/domain comparison where resources permit |
| L2a (MPI-enabled) | R=1, T=C | Full-node diagnostic, allowed to span NUMA domains |
| MPI-off full machine | R=1, T=C, has_mpi=false | Required small/medium and selected large profiles, including cross-NUMA |
| L3/L4 | Multiple nodes | Pending proper HPC cluster/interconnect qualification |

Use actual balanced subsets when allocation is uneven and record unused cores. Small/medium profiles span the required
controls plus MPI-off full-machine operation. Selected large profiles use the two full-node MPI decompositions and
MPI-off full-machine operation, not a new all-sizes Cartesian product. Calibrate shared profiles against the actual
MPI-disabled baseline too; after freeze, a required baseline/candidate capacity failure blocks acceptance, not grounds
to shrink a workload or drop a cell. MPI-enabled diagnostic L2a retains the same flags even if it cannot fit.
If D=1, L2a and L2b coincide: deduplicate within the same build mode without removing a required gate. Never deduplicate
MPI-off evidence against an MPI-enabled run, even at identical R/T/allocation.

After the Task 0 architecture decision and reconciliation of these conditional production gates, the sequence is:

1. On the baseline only, calibrate each ladder row using its preferred control: Hubbard/Pauli `lower_atol`, random
   Heisenberg `obs_terms`, and random Schrödinger `num_generators` (or the documented mode-count sweep). Keep model
   structure/seed fixed within each sweep; changing generator count also changes gradient length. Record every override.
2. Size graph rows against the largest build/energy/gradient peak, not build alone; also account for numerical export
   and validation. Use existing grouping where it actually shares a graph, not an assumed sharing property.
3. Obtain approval and freeze exact flags, seeds, routing, placements and required cells. Apply the same flags to each
   matched rung/arm; pre-freeze workload sizing is not permission to relax oracle tolerances or tune after a regression.
4. Run the existing pytest nodes on baseline and candidate with unique labels, fresh processes and `--bench-rounds=1`.
   Preserve the plan's per-operation isolation for formal observations; grouped pilot/diagnostic runs do not certify
   fresh construction peaks. Add separate construction and numerical-validation passes using the same model builders.
5. Compare five pairs per required cell, adding five more after a failure, using the five metrics defined below. Reuse
   existing reports and record actual term maps/values as well as counts; a term-count match is not numerical proof.

Keep MPI-enabled cross-NUMA L2a output in a separately labelled diagnostic inventory/directory, outside the frozen
required-cell manifest consumed by the acceptance comparator. Its ratios are informative, never a waiver or an
acceptance verdict. MPI-off full-machine cells belong IN that required manifest and pass the same five metrics and
numerical checks; they must not inherit L2a's diagnostic classification merely from their R=1,T=C geometry.
Diagnose numerical mismatches even there; diagnostic-only describes the performance gate, not permission for wrong
results. Full-node L2a/L2b contrasts include process/decomposition and local MPI effects. Replacing L2a with a smaller
NUMA-local run changes resources and cannot claim that same contrast. Historical ladder timings/sizes are examples,
not measurements on the target machine.

### Acceptance evidence and stopping rules

The initial acceptance gate is a frozen campaign on one c8a.metal-24xl, with at most one worker per physical core.
Record the actual socket/NUMA/core layout before selecting required geometries. Include one-rank-per-domain operation
and an additional two-ranks-per-domain configuration with disjoint physical cores, where resources permit. Both arms use
identical decomposition/allocation in every comparison; changing ranks only on the candidate cannot rescue a regression.
Also require full-machine single-process MPI-disabled operation in every size tier's selected profiles; an MPI-enabled
binary at one rank cannot substitute. Source-reference workloads remain candidates, not approved production sizes.
After separate execution authorization, use baseline-only, unscored calibration to choose feasible sizes. The approved
pilot budget is 2 hours total, excluding
builds/setup, with 15 minutes per trial including construction, evaluation and numerical export/validation. If the time
budget is insufficient, stop and report rather than silently extend it. Target at most 75% of observed usable allocation
RAM across all ranks, including construction and validation/export temporary storage. Record the RAM basis and byte
ceiling before the pilot; do not continuously rescale it from falling free memory during a trial. Exceeding the memory
target requires owner review before larger trials. These are operator-side sizing limits, not library memory/clamping
policies or guarantees against out-of-memory failures. Estimate formal campaign time/cost from the pilot before seeking
approval; do not reduce required samples or coverage to meet an unapproved budget.

Archive each trial's config, placement, binary identity, elapsed time, memory observations and outcome, including
rejected sizes. Do not consult candidate performance when selecting the campaign. Baseline numerical inconsistencies
stop calibration rather than justify a smaller test. Resource failures may inform pre-freeze sizing; they are not failed
acceptance cells. The approved calibration limits do not authorize builds, setup or runtime execution.

Small/medium profiles cover the full geometry sweep with the existing routing requirements, plus MPI-off full-machine
operation. Selected large profiles cover the full-node one- and two-ranks-per-domain MPI configurations and MPI-off
full-machine operation, not one-worker runs. This coverage is fixed before results, not chosen by the winning design.
Select the actual large workloads and common sizes using baseline-only calibration, including the MPI-off baseline,
and obtain owner approval. Task 1 keeps its 2-hour/15-minute/75% pilot limits; Task 0's 8-GiB cap does not apply here.
Freeze the complete size-band-specific inventory before formal baseline collection. Calibration results are not
acceptance samples: collect fresh formal baseline runs after the freeze, even for unchanged configurations. Every
required cell must pass, but this does not claim parity for omitted workload/geometry combinations or CPU families.
Correctness still covers the supported API. A noisy failure means parity was not demonstrated, not proof of a slowdown,
and still blocks acceptance without a waiver.

Intel/AMD HPC qualification remains separate from initial acceptance and must not be reported as already passed.
Multi-node performance qualification is pending access to a proper HPC cluster and interconnect; same-instance MPI
correctness/performance does not substitute for that evidence. Do not add two-instance EC2 networking to the initial
campaign or require sibling-worker SMT profiles.

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
For MPI-off single-process cells, sum and max peaks coincide; retain all five fields and gates rather than weakening
the schema or substituting Task 0's whole-process RSS measurements.

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

Freeze and archive an owner-approved campaign inventory and digest after unscored calibration and before formal baseline
collection. It lists every required profile/node/build-mode/geometry/routing cell for its size band, independently of
observations. Include has_mpi in cross-arm identity and verify it from monoprop.has_mpi in each imported binary, with
binary-linked provenance. Missing/mismatched build modes fail; a single observed rank does not prove MPI is disabled.
Gate the shared harness's mpi4py import on has_mpi so an installed MPI package cannot initialize MPI in an MPI-off run.
MPI support/version fields are not applicable there, not invented FUNNELED results. Comparator input must match the
inventory, so removing a whole cell from both arms cannot pass. Each sample references one fresh timed artifact, one
separate fresh
construction artifact, and a matching untimed validation artifact. Join only matching cell/arm/binary/configuration,
parameters, geometry/routing/allocation and campaign provenance. Timed/construction run IDs and files cannot be reused;
require five distinct pairs per arm/cell, or ten after prescribed repetition. Validation and placement diagnostics can
be reused only for the same cell/arm/binary/configuration. Task 1 specifies the CLI, file schema, identity checks and
independent placement references; no caller boolean or synthetic-data bypass substitutes for that evidence.

The shared shape-preflight overlay must accept declared baseline `monoprop_PARTITIONS=T` and candidate
`monoprop_NUM_THREADS=T` with removed partition settings unset, on the same allocation. Test the candidate's
unset-library fallback to OpenMP separately; both configuration paths must resolve the declared budget. Verify
declarations against observed shape;
arm labels alone cannot establish shape. Extend existing ladder/provenance/term-count fields without renaming historical
nodes. `tools/capture-baseline.py` supplies supplementary small-fixture evidence only; it cannot replace global,
ownership-aware benchmark validation with gradients and provenance.

The owner accepts reduced per-rank capacity: one independent propagator's store must fit fixed uint32_t TermIndex
(`TypeAliases.h:39`), whereas several old partitions could collectively exceed that ceiling. Row IDs exclude
UINT32_MAX, the index sentinel; row counts and some graph counts may equal it. Extend existing guards, including
allocation arithmetic and MPI counts. A required campaign cell that does not fit still blocks acceptance; never widen
indices, change ranks or add shards silently.

The representative 512-GiB/eight-domain example does not select the campaign or impose a hardware limit. If evenly
budgeted at 64 GiB/rank, current coefficient-bearing storage is memory-limited before the ceiling. Approximately 2^32
four-byte IDs occupy 16 GiB, not a full operator: at that row count, a consolidated current hash table alone is 64 GiB,
coefficients 32 GiB, and packed rows at least 8 GiB, before other storage. Across old partitions the corresponding
optimistic aggregate lower bound exceeds 85 GiB. These are source-derived bounds from `OperatorIndex.h`/`MPOperator.h`,
not measurements; CPU binding alone does not enforce a RAM budget.

## Non-goals

Beyond the bounded existing-library index experiment, no custom concurrent hash table or new executor framework.
No GPU backend, load-balancing/repartitioning algorithm, asynchronous gate pipeline, GIL-policy change, graph-format
redesign, new replay transport, or replacement
topology/affinity-management subsystem. Do not remove useful lifetime/ordering comments merely to lower line count.

## Execution handoff and supported build routes

All future code/test/build commands in the companion plan are planned, not executed during this document refresh.
This current-main refresh changes the two planning documents only; the existing glossary remains applicable. No
implementation or publication is authorized. The previously reported source drift is audited and incorporated above.
Before runtime work, verify the pinned main revision and the plan's refreshed anchor table; stop on subsequent changes
to signatures, routing, capacity, numerical or benchmark contracts. Keep Task 0 followed by conditional Tasks 1–13;
record
RED/GREEN evidence for new behavior, plus baseline results for preservation tests that may already pass. Do not
manufacture RED by removing earlier plumbing. Use kernel-specific worker observations for threading participation. Do
not execute past a failed gate or infer approval from this refresh.

The first execution handoff is the separately authorized Task 0 mini-app, ending at the architecture decision. Apply
its staged first pass and 2-hour/min(8 GiB, 25% RAM) limits; account for setup/build separately. These approvals do not
authorize launch. Only after that gate does the baseline pilot travel with Task 1 to the authorized host; neither
experiment must run on the planning host before handoff. Transfer the latest spec, plan and CONTEXT.md, not only the
historical published planning commit. The Task 1 portion of the remote entry point defines the subsequent baseline-only
phase ending with a proposed campaign/budget for owner approval, not candidate engine work or a completed Task 1 gate.
A different execution host does not silently change the agreed
c8a.metal-24xl acceptance target. No machine provisioning, execution or publication is authorized by this clarification.

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
