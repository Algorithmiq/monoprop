# Open optimization points and "for later" items (2026-10-11)

State at commit `1eaebd4` on `refactor-parallelization` (PR #398): S8, round 5, the merges of `origin/main` and step 6's
fixes F1/F2. Sources: the [sharded OpenMP plan](2026-09-29-sharded-openmp.md), and outside the checkout
`/home/ubuntu/s8-artifacts/`: `LEDGER.md`, `opt/FINDINGS-ROUND5.md`, `opt/FINDINGS-STEP6.md`.

**Caveat on the numbers.** Unless a row says otherwise, figures come from the targeted re-measures on the current
binaries (step 6: 20 cells, 3 frozen-driver pairs each, against fresh baseline samples, page-cache protocol). The
campaign verdict (145 of 250 cells pass) is still from the pre-round-5 binaries. Cells not re-measured since then are
uncertain in both directions.

## Part 1: open optimization points

### Memory gates (excess over the fresh baseline, MiB summed over ranks)

| # | Point | Current excess | What is known | Possible remedy |
| --- | --- | --- | --- | --- |
| M1 | Hubbard energy at T = 96 (reference 1x96 and MPI-off, reference-pared MPI-off) | +4.3 to +5.0 | Mostly the partner staging run itself (4096 values, 32 KiB per owner, about 3 MiB at T = 96), plus some arena growth in the first call | Drop or shrink the staging run (cost gradient Hubbard 1.8–4 % runtime before), or stage into a buffer each owner already holds (never tried) |
| M2 | Hubbard hair-lines: energy 2x48 and pared 2x48 (construction max), pared gradient 1x96, gradient 1x96 | ratios 1.0005–1.0015 (+0.4 to +1 MiB) | Same mechanism as M1, plus the 2x48 exchange buffers | M1's fix would probably cover these |
| M3 | Heisenberg/Schrödinger evaluation at T = 96 (energy-heis MPI-off; gradient-schr 1x96) | +1.7; +1.6 | Partly the per-step publication buffers (+1.4 MiB live) and partner staging (+0.7) seen in round 5; not attributed on the current binary | Attribution pass first (phase probe, live allocation trace) |
| M4 | Cells not re-measured since the campaign: large gradient-hubb (2x48 was +30 MiB), large gradient-heis (hair-line +0.9), reference build-graph-paul and propagate-heis at 1x1 (construction +1 to +3) | unknown now | Round 5 and the merge probably moved them, but nothing is measured | Include them in the next re-measure or in the campaign |
| M5 | libgomp's own pages | +0.3 per process | OpenMP is a required dependency | None. Most of the tiny 1x1 memory excess; those cells also fail the 1x1 runtime gate the owner accepted. |

### Runtime gates

| # | Point | Current | What is known |
| --- | --- | --- | --- |
| R1 | Pared Schrödinger construction (`reference-pared--pare-construct-schr`) at T = 96 and 2x48 (option A of round 5, not chosen) | 1.28–1.40 in the campaign | Bimodal first call (run-queue waits) on both arms, plus a real excess: about 5,900 more first-call page faults and barrier waits across 96 pinned threads. Ideas: merge barrier phases in paring; pre-size the paring buffers. |
| R2 | Tiny 2x48 constructions (option C, proposed as accepted) | 1.0–1.2, noisy | +0.1–0.35 ms on about 1.3 ms: per-gate cross-rank rounds. Skipping empty rounds needs rank agreement. |
| R3 | Tiny 1x1 evaluation, warm calls | 13.4 µs against round 5's 12.2 µs (the first call is back to parity) | The remaining per-call cost of main's fold changes |
| R4 | MPI 1x1 tail, **accepted by the owner** | tiny 1.09–1.10 (campaign 1.16); larger cells 1.00–1.05 | The larger cells' +6 % cycles were never attributed. The merge improved Hubbard propagate 1x1 to 0.94. |
| R5 | Bimodal first calls at T = 96, **left to the medians by the owner** | — | Host noise, on both arms |

### Opportunities from the merge of main (not measured)

| # | Point |
| --- | --- |
| O1 | Positions emit with several owners. Main's N-independent per-term emit (#383) runs only at P = T = 1, because the partner's owner hash reads the dense bitset. Hashing from the position lists would extend it to all T, which matters for wide operators at T = 96. |
| O2 | Hit-coefficient prefetch in the sharded self-resolve. Main added it to the single-owner engine (`Engine.h`); the sharded resolve path does not have it. |
| O3 | Posting fold for mid-size shards. F2 sends single-block cosine folds (up to 65,536 rows per shard) through the direct sweep, which is safe for memory and tiny calls. Whether the posting fold would pay off on sparse mid-size shards is not measured. |

### Not optimization, but open

- Final evidence: no current binary has campaign evidence. A full re-run needs about 29 GB of free disk (27 GB free on
  2026-10-11); a partial re-run would mix binaries.
- TSan: the `mpi_failure` host-`MULTIPLE` and `wrong_thread_entry_*` scenarios time out under TSan (also without the
  latest changes) and stay inconclusive. They pass in the release build.

### Suggested order for a next round

1. M1/M2, with the runtime trade-off measured: they cover the most remaining memory failures.
2. The campaign re-run, which settles M4 and gives the final evidence.
3. R1, if runtime parity at T = 96 matters.
4. O1, as a performance item of its own rather than for parity.

## Part 2: items recorded "for later"

### 1. Deferred by design until after the refactor lands (separate approval)

The plan's last section, "Explicit post-landing follow-up: numerical performance policy", keeps open, individually or
in combination:

1. arithmetic operation ordering in local kernels and in derivative/replay calculations;
2. the order or method used to combine shard contributions;
3. floating-point reduction schemes, including alternatives to the ascending-shard fold;
4. FMA policy (explicit FMA, compiler contraction).

The global constraints also say "No nested kernel parallelism initially", another lever kept for later. All of these
are off-limits now because they would break the bitwise-numerics rule.

### 2. Optimizations flagged in S5 ("none started")

| Item | Status now |
| --- | --- |
| Replay remote step: 4 checkpoints → 2 (fold the pack into the publishing phase; lay out step p + 1 during step p's post) | **Still open.** Later rounds made the layout cheaper (owner-parallel layout, contiguous columns) but did not reduce the checkpoint count. |
| Primary lays every round out serially, O(P × T²); cache the replay layout per layer | **Partly done:** owner-parallel layout (`bb71b9a`), and Step 2's A (no per-step allocation, no false sharing). A per-graph layout cache was not built. |
| Construction: up to 6 checkpoints per remote pass (merge Q1 into Q2; plan the answers with the query post) | **Closed:** tried three times as the "announced-count answer round" and rejected (no better anywhere, worse at 4x24). Owner-parallel query placement was also rejected. Both are on the do-not-repeat list. |
| Profile gradient Schrödinger at 2x48 (+18 ms) | Superseded: the cell no longer fails on runtime. |
| Robustness: an optional count round per remote pass, so a wrong answer count aborts instead of hanging | **Still open** (not a speed item) |
| Minor: each worker rebuilds its gate's peer vector per gate | **Probably still open** in construction. The replay side's per-step `other_ranks()` allocation was removed in Step 2. |

### 3. S4/S5 deviations accepted "to be re-evaluated at a later stage"

They were re-evaluated in S7 and the S8 campaign. Their candidate remedies ended up as:

- smaller staging run: rejected in S6's B-memory work (−2.6 MiB at most, 1.8–4 % slower). Round 5 removed only the
  widening to the largest block;
- staging into an existing owner buffer: never implemented. A probe bounded the saving at 1.1–2.0 MiB. It is the open
  idea behind M1;
- allocation tracer at a small threshold inside the window: done in round 5;
- profiling energy Heisenberg's cold first call (thread-local scratch warm-up): never done specifically. Round 4 found
  the T = 96 first calls bimodal on both arms, and the owner left that to the medians.

### 4. Qualification still pending (verification, not optimization)

- Nix, macOS, Linux aarch64, CI wheels (manylinux images), minimum-version compilers.
- Multi-node runs: never qualified.
- The TSan `mpi_failure` scenarios above.

### 5. Discussed on 2026-10-10/11, not yet in the plan

- An HPC cluster cross-check: a pre-registered reduced ladder covering multi-NUMA nodes, a real interconnect, and many
  repetitions with confidence intervals.
- An equivalence-style gate: a margin agreed in advance instead of the strict median ≤ 1.00.
- The merge opportunities O1–O3 above.

### 6. OpenMP usage today (the baseline for items 7 and 8)

The library uses three OpenMP directives:

- one fixed team, `#pragma omp parallel num_threads(T)` in `detail/sharded/Team.h`: each owner works on its own
  shard through phases separated by barriers;
- one `#pragma omp barrier`, the team's phase checkpoint;
- one worksharing loop, `parallel for schedule(static)` in `detail/parallel/Workshare.h::for_blocks`. It runs
  serially when called inside the team, so kernels inside an owner's phase are serial.

There are no `simd` or `task` directives. Vectorization relies on `-O3 -march=native` auto-vectorization.

### 7. Evaluate explicit `#pragma omp simd`

Where it could matter, from the hot spots in the Step 1 and round 5 profiles:

- Word-wise XOR folds (`combine_columns_block`, the fold's word sweep) and `Monomial` word loops: probably
  auto-vectorized already, so little to gain.
- Bit-iteration loops such as `for_each_cos_index` (count trailing zeros, then index `coeff[i]`): inherently scalar.
  With AVX-512 they could become "expand the mask bits into indices, then gather, multiply and scatter". The largest
  candidate, for cosine scaling of sparse sets.
- Batched hash probing (`OperatorIndex::find_batch`, `probe_frozen_positions`): several keys hashed at once.
- Fused-apply record updates (`c[t] += s * c[src]`): risky. Targets can repeat, and a wrong `simd` independence claim
  is a silent bug.

Constraints:

- `simd` reductions reorder sums (for example the dot product in the cosine accumulation): they break the bitwise
  rule, so they belong under item 1 (numerical-performance policy), not here.
- Floating-point contraction: check that explicit `simd` does not change which multiply-adds GCC fuses.
- `-march=native` against portable wheels: generic x86-64 builds lose auto-vectorization. That is where explicit
  `simd`, `declare simd` or function multiversioning would pay off most, which ties in with the pending CI-wheel
  qualification (item 4).

How to evaluate: read the compiler's vectorization reports (`-fopt-info-vec-missed`) for the hot kernels to see what is
already vectorized; try one kernel (likely cosine scaling) with a gather/scatter version; keep only measured wins, with
the bitwise differential as the gate. Measure both native and generic-ISA builds.

### 8. Evaluate OpenMP tasking

Where it could matter:

- Fewer barriers: replace some phase barriers with task dependencies (`depend`). One late thread out of 96 stalls every
  checkpoint, which is behind R1 (paring barriers), the tiny 2x48 per-gate rounds and part of the bimodal first calls.
- Pipelining replay steps (S5's "4 checkpoints → 2", item 2): lay out step p + 1 while step p's exchange is in flight,
  as a task graph instead of fixed phases.
- Load balance: per-gate work differs between owners even when rows are balanced. Splitting owner work into finer
  tasks, or a `taskloop` with a grain size instead of `schedule(static)`, could fill idle waits.
- Irregular kernels: per-layer paring (`pare_graph_owner`), resolve probes, scans with uneven ranges.

Constraints (the hard part):

- MPI only on the primary thread (FUNNELED). A task may run on any thread, so tasks must never call MPI and MPI stays
  in primary-only phases; untied tasks are out.
- Owner-computes memory. Each shard's data is allocated by its owner (one-writer stores, per-thread malloc arenas,
  per-thread scratch). Tasks migrating between threads would scatter allocations across arenas, which moves the memory
  gates, and on multi-NUMA nodes would break first-touch placement.
- Determinism. Task completion order must never decide arithmetic order: results go into private slots and are folded
  in fixed order, as the threaded range kernels do now.
- Failure handling. The `phase()`/`TeamFailure` contract (catch on the worker, decide at the checkpoint) needs an
  equivalent for tasks, including cancellation and the abort under MPI.
- Overhead and tooling. A task costs about a microsecond, too much for the tiny cells' microsecond-scale calls.
  libgomp's and libomp's task schedulers differ, and TSan/Archer support for tasks needs checking.

How to evaluate: measure barrier wait time per phase first (R1's step-1 diagnostic, never run); prototype on one
bounded path (paring, or the replay pipeline) behind the existing bitwise tests; measure tiny and T = 96 cells, and on
a multi-NUMA node if the cluster cross-check (item 5) happens.
