# Rank-local OpenMP parallelization design

## Status and authority

This is the design accompanying the implementation plan at `../plans/2026-09-18-rank-local-openmp.md`. It records the
direction discussed with the owner; it is not a claim that the proposed implementation meets its performance gates.

Source inspected: branch `perf/linear-routing-on-wire`, commit `290112c8289ab8015eb9a2c7651ac409839c5a88`.

The planning documents were published in `refactor-parallelization` at `38eb13d80a95dd1ac60e762f14f68c9a5925f755`;
this policy revision follows the owner's design interview. The owner approved mandatory OpenMP, removal of direct hwloc
and partition configuration without deprecation aliases, the associated API/ABI and capacity breaks, environment-only
construction-time thread budgets, and the failure/calling contracts below. These are design approvals, not
implementation or publication authority for this revision.

The owner selected one AWS EC2 `c8a.metal-24xl` for implementation/testing and the initial acceptance campaign.
Production deployments are dual-socket HPC nodes using Intel Sapphire Rapids or AMD Zen3/Zen4/Zen5. Hardware SMT may be
on or off, but the recommended configuration always uses one worker per physical core, never extra SMT-sibling workers.
Passing the frozen EC2 campaign is sufficient for initial acceptance; production-HPC performance remains unverified
until separately qualified. Multi-node qualification is pending a proper HPC cluster/interconnect, not a second EC2
instance. Observed EC2 topology/software and representative workload sizes still need recording/approval before freezing
that campaign. The owner approved baseline-only, unscored feasibility calibration before the freeze, once execution is
separately authorized. Small/medium profiles cover the full geometry sweep; large profiles cover only the full-node
one- and two-ranks-per-domain configurations. This revision authorizes no runtime work or implementation.

## Objective

Replace thread-owned operator partitions and their in-process communication protocols with one compact operator store
per MPI rank, operated on by OpenMP worksharing. Preserve the mathematical features, memory footprint, and performance
of the existing library. Simplification without regressions is the objective; the one-store/OpenMP candidate must earn
acceptance, not be rescued with prohibited replacement machinery if its gates fail. Each MPI rank's assigned CPUs are
assumed to lie within one NUMA domain. The user chooses the
number of processes P and requested threads per process T for the resources available, and configures the launcher and
OpenMP runtime to bind/pin processes and threads correctly. The library neither discovers topology nor repairs
placement.

## Global constraints

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
- Preserve packed operator rows, the keyless hash index, fixed uint32_t TermIndex, sparse state storage, and
  inverted-index cosine recomputation.
- Do not introduce boost::concurrent_flat_map, hidden shards, a dense duplicate store, a second persistent key store, a
  full per-thread operator, or full-vector coefficient double buffering.
- Preserve Majorana and Pauli bases, Heisenberg and Schrödinger pictures, cutoffs, graph build/replay/paring, gradients,
  partial contraction, copying, and initial-operator updates.
- MPI calls execute on the controlling thread outside library-created OpenMP worksharing regions; retain
  MPI_THREAD_SERIALIZED rather than changing the initialization contract.
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

### Selected implementation/test platform

AWS documents `c8a.metal-24xl` as AMD EPYC 9R45, 96 physical cores/96 vCPUs, one thread per core and 192 GiB RAM in its
[exact-size specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/co.html#co_hardware). The
[C8a page](https://aws.amazon.com/ec2/instance-types/c8a/) identifies fifth-generation EPYC (Turin) and explicitly
states that there is no SMT. Do not call a 192-worker run on this instance an SMT-on test; it would oversubscribe its
physical cores. The exact socket/NUMA layout was not established by the reviewed AWS specifications and must be observed
on the actual allocation before selecting rank/thread geometries. In particular, do not assume eight NUMA domains or
approve one rank spanning all 96 cores without checking containment.

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
  Each rank's allocation must accommodate its workers within one NUMA domain; avoid unintended overlap with other ranks.
  Documentation will advise at least one MPI process per NUMA domain used by the allocation; multiple processes may
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

Archive observed rank/worker masks, physical-core sibling identities and NUMA containment before accepting measurements.
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
   helper, and independent virtual `clone_` hook (`MonomialPropagator.h:286–305`). The owner accepts immediate C++
   source/ABI breaks for partition removal in this 0.x library; record release notes and rebuild library/bindings.
   High-level Python mathematical signatures remain unchanged; `_core` loses `partitions` and gains no thread keyword.
6. Returned term/coefficient ordering may differ from old partition-block order. Validate identical global retained term
   keys and coefficient associations within existing tolerances, not coefficient-only sorted lists. Equal energies and
   gradients alone are insufficient. Keep candidate ordering deterministic across budgets at fixed rank geometry.
7. Preserve boolean linear/splitmix routing, default linear rejection of unsupported R, generator shifts and peer rules.
   The owner accepts the splitmix ownership change from `floor((hash % (R*S))/S)` to `hash % R`; do not emulate S with
   hidden shards. Keep `monoprop_ROUTING` and `monoprop_ROUTE_SEED`: these control MPI routing, not thread partitions.
   Test default-linear R=3 rejection separately; successful R=3 suites/benchmarks explicitly use splitmix. Compare
   global maps and per-rank loads/peaks at identical routing/seed/rank geometry.

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
failures inside the lifetime of active exchange tickets, before stack unwinding can wait on peers that never posted; an
outer catch alone is insufficient. Successful paths explicitly wait as before. For single-rank operation, rethrow the
original exception. Enforce post-mutation invalidation on the affected propagator: later state-consuming/mutating calls,
copy/clone and dependent retained functionals reject it before touching state. Existing independent copies remain valid.
Pre-mutation validation failures leave a valid object usable; destruction remains safe. There is no rollback or reset
that revives a failed object. Task 3 defines the invalid flag, entry checks and failure-injection tests.

No rank-local early return may bypass a collective peers will enter. Validate MPI_THREAD_SERIALIZED also when MPI was
initialized externally. The owner confirms one controlling caller per rank, outside active host OpenMP teams, no
overlapping calls on the same propagator/dependent functionals and no overlapping host MPI calls. Distributed entry
from an external OpenMP team remains unsupported; numerical helpers may serialize when nested, but that does not
expand the public calling contract. Add no process-wide locking service or nested-parallel execution machinery.

Lazy initialization, allocation, and cache mutation must be outside worker bodies unless storage is strictly
worker-private. Do not assume `const` means thread-safe: both the inverted index and parity cache have lazy mutation
today.

## Performance acceptance and stopping

The initial acceptance gate is a frozen campaign on one c8a.metal-24xl, with at most one worker per physical core.
Record the actual socket/NUMA/core layout before selecting geometries. Include one-rank-per-allocated-domain operation
and an additional two-ranks-per-domain configuration with disjoint physical cores, where resources permit. Both arms use
identical decomposition/allocation in every comparison; changing ranks only on the candidate cannot rescue a regression.
Source-reference workloads remain candidates, not approved production sizes. After separate execution authorization,
use baseline-only, unscored calibration to choose feasible sizes. The approved pilot budget is 2 hours total, excluding
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

Small/medium profiles cover the full geometry sweep with the existing routing requirements. Large profiles cover only
the full-node one- and two-ranks-per-NUMA-domain configurations, not one-worker runs. This coverage is chosen before
formal measurements, not used to
excuse a failed cell later. Select the actual large workloads and sizes from calibration and obtain owner approval.
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
collection. It lists every required profile/node/geometry/routing cell for its size band, independently of observations.
Comparator input must match that inventory, so
removing a whole cell from both arms cannot pass. Each sample references one fresh timed artifact, one separate fresh
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

No new executor framework, custom concurrent hash table, GPU backend, load-balancing/repartitioning algorithm,
asynchronous gate pipeline, GIL-policy change, graph-format redesign, new replay transport, or replacement
topology/affinity-management subsystem. Do not remove useful lifetime/ordering comments merely to lower line count.

## Execution handoff and supported build routes

All future code/test/build commands in the companion plan are planned, not executed during this document refresh.
This revision changes planning documentation and the agreed glossary only; no implementation or publication is
authorized. Before implementation, recheck exact HEAD and the plan's baseline-drift
table; stop on changed signatures, routing, capacity, numerical or benchmark contracts. Keep 13 gated tasks and record
RED/GREEN evidence for new behavior, plus baseline results for preservation tests that may already pass. Do not
manufacture RED by removing earlier plumbing. Use kernel-specific worker observations for threading participation. Do
not execute past a failed gate or infer approval from this refresh.

The baseline-only pilot travels with Task 1 to the authorized execution machine; it need not run on the planning host
before handoff. Transfer the latest spec, plan and CONTEXT.md, not only the historical published planning commit.
The plan's remote entry point defines a bounded first phase ending with a proposed campaign/budget for owner approval,
not candidate engine work or a completed Task 1 gate. A different execution host does not silently change the agreed
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
