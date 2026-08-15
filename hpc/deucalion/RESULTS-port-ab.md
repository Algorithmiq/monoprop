# A/B after porting `perf/multinode-comm-scaling` onto main

> **Archival.** The job scripts this file names (`ab-interleaved.sh, layout-bakeoff.sh`) were removed when
> `sbatch/ab-100m.sh` replaced them; see README §10. The measurements stand, but
> they predate the vectorized `benches/_builders.py`, which changed the RNG stream —
> so these term sets cannot be reproduced and these numbers share no axis with any
> run made after it.

Deliverable 3 of `RESULTS-layout-bakeoff.md` ("re-run this bake-off after the port").
Two nodes, `normal-x86`, `--exclusive`. Problem identical to the bake-off: 256 modes,
cutoff 6, 200k observable terms, 100 generators, Heisenberg only.

Under test: `s2-comm-to-main` @ `22631cd` — PR #166 merged onto main `5836a06`, with its
`HybridComm` test cases moved onto main's argument bundles. Baseline: the primary
checkout, whose `cpp/` tree is byte-identical to `origin/main`.

## Method, and the trap the first attempt fell into

The first before/after pair ran as **two sbatch jobs** (13:20 and 13:22, same node) and
compared the **means** that `benches/report.py` prints. It reported layout B's `pare` at
**1.86x slower** after the port. That was an artifact, twice over:

- **Means.** `pare`'s distribution is badly skewed — one run showed min 1.07 ms, median
  2.03 ms, mean 2.95 ms. The mean tracks stragglers, not the code.
- **Separate allocations.** Any per-allocation slowdown lands entirely on one side and
  reads as an effect of the change. Recomputing on min/median did **not** rescue it,
  because the skew came from the allocation, not from the choice of statistic.

`ab-interleaved.sh` fixes both: both venvs run inside **one** allocation, the two sides of
a given layout run **back to back**, and the order flips on every (rep, layout) cell so
drift cannot accumulate on one side. `ab_summary.py` reports the median across reps and
carries `min` alongside — when those two disagree, the difference is the machine.

## Result: layout A wins big, nothing regresses

Interleaved, 3 reps x 3 bench-rounds (`ab-interleaved-port-vs-main-N2`):

| layout | operation | main med | port med | port/main | min ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| **A** 1x128 | gradient | 180.2 ms | 52.7 ms | **3.42x faster** | 3.48x |
| **A** 1x128 | inplace | 432.2 ms | 141.9 ms | **3.04x faster** | 3.16x |
| **A** 1x128 | energy | 81.6 ms | 31.8 ms | **2.57x faster** | 4.26x |
| **A** 1x128 | build_graph | 512.6 ms | 318.4 ms | **1.61x faster** | 2.47x |
| B 8x16 | all five | — | — | 0.91–1.16x | disagree |
| C 2x64 | build_graph | 632.4 ms | 478.9 ms | 1.32x faster | 1.34x |
| C 2x64 | rest | — | — | 0.96–1.13x | disagree |

Layout A's four large operations improve with median **and** min agreeing — that is the
PR's claim, reproduced on Deucalion under the stricter methodology. Layouts B and C are
flat: every apparent difference there has median and min pointing opposite ways.

`pare` needed a second pass because its spread exceeds the effect: its *same-build* min
moved 4x between jobs (6.07 ms vs 1.47 ms for main). At 6 reps x 10 rounds
(`ab-interleaved-pare-focus-N2`):

| layout | operation | main med | port med | port/main | main min | port min | min ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A 1x128 | pare | 1.408 ms | 1.659 ms | 1.18x slower | 0.799 | 0.781 | **0.98x** |
| A 1x128 | energy | 65.8 ms | 24.5 ms | **2.69x faster** | 55.7 | 15.8 | 3.52x |
| B 8x16 | pare | 1.284 ms | 1.229 ms | 0.96x flat | 0.682 | 0.705 | 1.03x |
| B 8x16 | energy | 22.2 ms | 26.0 ms | 1.17x slower | 14.1 | 12.9 | 0.91x |

With 10 rounds `pare` settles at 1.2–1.7 ms on both sides and the two builds' minima land
within 2% of each other. **No regression is established at any layout.**

## The real finding: layouts B and C run unpinned

`monoprop_COMM_PROFILE=1` (`barrier-groups-probe.sh`) reports the L3 domain count the
two-level `PartitionBarrier` actually grouped into:

| layout | transport | `barrier_groups` | `barrier_per_sync_us` |
| --- | --- | ---: | ---: |
| A 1x128 | ShmComm | **32** | 32.7 |
| B 8x16 | HybridComm | **0** (flat) | 290.1 |
| C 2x64 | HybridComm | **0** (flat) | 571–695 |

`barrier_groups` is wired up on both transports (`HybridComm.h:96`, `ShmComm.h:51`), so 0
is a measurement, not a reporting gap: **the two-level barrier never engages in a
multi-rank run here.** The cause is in `partition_cpusets`
(`cpp/monoprop/detail/partition/CpuTopology.h`), which PR #166 does not touch — since fixed;
see `RESULTS-threading-baseline.md` for the before/after:

```cpp
const auto cores = enumerate_physical_cores();   // cores THIS PROCESS may use
if (cores.empty() || group_count * n > cores.size()) {
    return {};                                   // => unpinned, no L3 domains
}
```

`enumerate_physical_cores()` filters by the affinity mask. Under
`srun --cpus-per-task=16 --cpu-bind=cores` a rank's mask holds 16 CPUs, so
`cores.size() == 16` while `group_count * n == node_size * S == 8 * 16 == 128`. The guard
trips. Slurm has *already* carved the node into per-rank masks and the rank then divides
by `node_size` a second time.

So on this cluster:

1. **Layout B — the default recommended in `RESULTS-layout-bakeoff.md` — has been running
   with no partition pinning at all**, on main and on the port alike. The 290 vs 32.7
   us-per-sync gap, at 8x *fewer* barrier participants than layout A, is threads yielding
   into syscalls instead of spinning on-core — precisely what `PartitionBarrier`'s
   comments say pinning is there to prevent. This is the most likely source of the wide
   spread that made the original bake-off hard to read.
2. Layout A (1 rank/node, mask = all 128 cores) is the only layout where pinning engages —
   and the only one showing a clean speedup. The two mechanisms are the same mechanism.
3. PR #166's two-level barrier is therefore **inert for multi-rank runs on this cluster**.
   Its measured wins here come from the other three changes (parallel prefix,
   `alltoallv_reverse`, empty-block veto).

This is a pre-existing main-side bug, not something the PR introduced, and it wants its own
issue rather than being folded into #166. A fix has to decide whose job the division is:
when Slurm has already confined the rank, the rank should partition *its own* mask
(`group_count` effectively 1), not re-divide the machine.

## Reproduce

```bash
# full ladder, 3 reps
sbatch -N2 --chdir="$PWD" --export=ALL,RESULTS_TAG=port-vs-main \
    hpc/deucalion/sbatch/ab-interleaved.sh

# one noisy operation, more reps
sbatch -N2 --chdir="$PWD" \
    --export=ALL,RESULTS_TAG=pare-focus,REPS=6,BENCH_ROUNDS=10,\
BENCH_K='heisenberg and (pare or energy)',LAYOUTS_ONLY='A_1x128 B_8x16' \
    hpc/deucalion/sbatch/ab-interleaved.sh

# which barrier actually ran
sbatch -N1 --chdir="$PWD" hpc/deucalion/sbatch/barrier-groups-probe.sh
```

Do not read `REPORT.md`'s means for anything whose spread is comparable to the effect;
read `AB-SUMMARY.md`, and believe a difference only when the median and min columns agree.
