# 100M-term A/B: `perf/multinode-comm-scaling` vs `main`

2026-08-13, Deucalion x86 (`normal-x86`, exclusive). Jobs `1821294` (N=1), `1821295` (N=2).
Harness `sbatch/ab-100m.sh`, summaries in `$PROJ/runs/ab-100m-N{1,2}/AB-SUMMARY.md`.

| | |
| --- | --- |
| arms | `main` @ `1bd8dc7` (0.8.1.dev34) vs branch @ `fa65466` (0.8.1.dev53) |
| problem | 24M observable terms → **99,441,369 propagated terms**, 250 modes, cutoff 6, 100 generators |
| layout | B_8x16 — 8 ranks/node × 16 partitions (world 128 at N=1, 256 at N=2) |
| reps | 4, interleaved, `--bench-rounds=1` |
| identical | `MAX_NUM_MODES=1024`, `MALLOC_ARENA_MAX=16`, `OMP_NUM_THREADS=1`, MPI on, same `benches/` |

Both builds passed `ctest -L serial` (207/207 main, 224/224 branch) and the Python MPI
suite across all four layouts before measuring.

## Result

**The branch restores thread placement and that does not make it faster at this size.**

| operation | N=1 port/main | agree | N=2 port/main | agree |
| --- | ---: | ---: | ---: | ---: |
| `build_graph` | *unresolved* (spread 6.0×) | 2/4 | *unresolved* (spread 2.1×) | 2/4 |
| `propagate` | **1.34× slower** | 4/4 | 1.15× faster | 4/4 |
| `energy` | *unresolved* (spread 1.3×) | 2/4 | **1.32× faster** | 4/4 |
| `gradient` | 0.97× flat | 3/4 | 0.97× flat | 3/4 |

Memory is flat everywhere except `energy` at N=2 (1.23× less peak above its own floor).
`op+graph` agrees across arms to the byte — 11.48 GiB at N=1, 11.87 GiB at N=2 — so both
arms really did walk the same operator.

Ratios are **medians of per-rep paired ratios**, not ratios of per-side medians. This is
not a refinement, it changes conclusions: `build_graph` at N=1 has per-rep ratios of
2.44, 0.44, 0.41, 2.11 — one arm near 1.7 s and the other near 4 s in every rep, alternating
which. Those two sets have nearly equal medians, so the unpaired statistic reads `0.95×
flat`, indistinguishable from a genuine null. It is not a null; it is noise 6× wider than
any effect claimed here.

## The placement fix works and is not the bottleneck

`main`'s `partition_cpusets` has no `node_mask` parameter, so at layout B it asks for 8×16
cores out of the 16 a Slurm-confined rank can see, `placement_order` refuses, and every
rank runs unpinned. The build-agnostic `/proc` probe confirms it on both node counts —
`main` 0 single-CPU threads per rank, branch 16, minimum over all ranks and reps. This is
the one prediction that held exactly.

Why it buys nothing: `COMMPROF` (branch only) shows `barrier_peers ≈ mpi` in every cell.

| | barriers | `mpi_s` | `barrier_peers_s` | per-sync |
| --- | ---: | ---: | ---: | ---: |
| N=1 `fresh` | 2350 | 0.39 | 0.65 | 274 µs |
| N=1 `graph` | 4408 | 2.53 | 2.82 | 604 µs |
| N=2 `fresh` | 2350 | 0.60 | 0.88 | 359 µs |
| N=2 `graph` | 4408 | 2.27 | 2.50 | 536 µs |

That 274–604 µs is **not** barrier latency, and it must not be compared against the
437 → 15.5 µs/sync in `RESULTS-threading-baseline.md`. `barrier_per_sync_us` is not
comparable across problem sizes: it is peers idling while partition 0 is inside MPI, so it
grows with the work being waited *for* (15–22 µs at 0.8M terms, 150–435 µs at 29M, 274–604 µs
here). Placement fixes barrier latency; it cannot fix a `MPI_THREAD_SERIALIZED` funnel, and
at 100M terms the funnel is what there is. See the ranked list of what would actually move
this in the 29M cost-split analysis — wider ranks-per-node, a local self-rank leg, and the
unreached `MPI_Ialltoallv` overlap.

`energy` at N=2 is the exception that fits: it is the lowest-work, most sync-dominated
operation, so it is the only place the fix shows.

## What needs following up

`propagate` is **1.34× slower on the branch at N=1**, in all four reps, in a run where the
same operation is 1.15× faster at N=2. The sign flip with node count is reproducible within
this run but unexplained. A plausible reading — untested — is that the branch's barrier
restructuring costs something on the intra-node shm path that only pays off once there is
an inter-node component. Do not ship this as a win without resolving it.

## Reading rules and limits

- **`dmem` next to `op+graph`, always.** `energy`/`gradient` add 0.04–0.18 GiB above their
  own floor and that will be misread as "gradient is nearly free". It is scratch on top of
  an 11.9 GiB graph that is already resident.
- **`main` emits no `COMMPROF`** — `monoprop_COMM_PROFILE` does not exist on that branch.
  Every cross-arm claim here rests on wall time and the `/proc` probe. There is no
  measurement of `main`'s barrier cost, only of the branch's.
- **The RNG stream changed.** `_random_terms` was vectorised (9.8 → 1.22 µs/term) to make
  24M terms affordable, so nothing here shares an axis with `RESULTS-threading-baseline.md`
  or any earlier table, even at matching parameters.
- **`pare` and `inplace` were not run.** The 1.37× `pare` regression in
  `RESULTS-threading-baseline.md` is therefore untested at this size, not fixed.
- **4 reps is few.** Three of eight operation/node-count cells are unresolved. `build_graph`
  needs more reps or a quieter node before anything can be said about it at all.

## Cost

Whole campaign ≈ 2.1 node-hours: builds 2 × 4 min, correctness 2 × (5–9 min) × 2 nodes,
two sizing rungs 4 min each, then 57 min at N=1 and 58 min at N=2. Memory at full size was
predicted to 1% from the two rungs (116 GiB/node predicted, 115 GiB measured), which is
what made a separate full-size pilot unnecessary.
