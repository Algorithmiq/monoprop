# Threading portability fixes: measured before/after

> **Archival.** The job scripts this file names (`funnel-width.sh, barrier-groups-probe.sh`) were removed when
> `sbatch/ab-100m.sh` replaced them; see README §10. The measurements stand, but
> they predate the vectorized `benches/_builders.py`, which changed the RNG stream —
> so these term sets cannot be reproduced and these numbers share no axis with any
> run made after it.

Under test: three fixes to the intra-node partition threading, against pristine `c1e034c`.
Both sides are git worktrees on `/projects`, built in the same session with the same
toolchain (GCC 14.3.0, Open MPI 5.0.8, `znver2`, `MONOPROP_MAX_NUM_MODES=250`), so only the
compiled extension differs.

The reported problem was *performance that degrades from system to system*. It was not a
correctness bug — every equivalence and determinism test passed before and after.

## The three mechanisms

1. **Pinning silently turned itself off under Slurm.** `enumerate_physical_cores()` already
   filters by the rank's affinity mask, and `partition_cpusets()` then divided by the number
   of co-located ranks a *second* time. `group_count * n` exceeded the share, the guard read
   that as "host too small", and every multi-rank layout ran **unpinned** — taking the
   two-level barrier's domains with it, because `cpuset_domains()` derives them from the
   cpusets.
2. **The spin budget was a fixed iteration count with an arch-dependent cost.** `cpu_relax()`
   is one `PAUSE` on x86 and one `YIELD` on aarch64, which differ by more than an order of
   magnitude, so 2048 iterations spent a wholly different budget on each. Past the budget the
   barrier yielded in an unbounded loop.
3. **Locality-group discovery keyed on `cache/index3` only.** A part without an L3 made every
   core its own domain: a flat barrier carrying S extra cache lines, while `barrier_groups`
   still reported S and read as if the optimisation had engaged.

## Method

`bgroups-ab.sh` runs **both** builds over the same layouts in **one** allocation on **one**
node, so the before/after is read off the same hardware and the same `--cpu-bind`. That
matters more than usual here: the mechanism under test is triggered by the launcher's mask,
so a difference in allocation would *be* the variable.

`--cpus-per-task=$((128 / RPN))` with `--cpu-bind=cores` is the trigger — it is what confines
each rank to its own slice.

## Result: the barrier was 28-70x more expensive than it needed to be

`monoprop_COMM_PROFILE=1`, one node, job 1818392. `barrier_per_sync_us` is per rank; the range
covers all ranks.

| layout | `barrier_groups` base → fixed | µs/sync base → fixed |
| --- | --- | --- |
| A `1x128` | 32 → 32 | 24.9 → 21.5 |
| B `8x16` | **0 → 4** | **432-443 → 10.9-19.2** |
| C `2x64` | **0 → 16** | **837-844 → 11.5-49.3** |

- **Layout A is the control.** Its mask is the whole node, so it never hit the bug. Grouping is
  identical (32 domains) and the timing is flat — the fix did not perturb the path that already
  worked.
- **Layouts B and C are the bug.** `barrier_groups=0` is pinning having disabled itself; the
  8-16x fewer barrier participants were *more* expensive per sync than layout A's 128, which is
  the signature of a `sched_yield` storm rather than a coherence cost.
- 4 domains for 16 cores and 16 for 64 is exactly one 4-core CCX per domain on EPYC 7742.

Cross-checked at layout B on a separate job (1818397) with output preserved verbatim:
`barrier_groups=4`, 9.3-12.4 µs/sync across 8 ranks.

## The generalised discovery is a no-op on x86 and load-bearing on ARM

x86 (`cnx001`) reports `index2` (L2) shared by one CPU and `index3` (L3) shared 4-wide, so
"deepest level shared with another core" selects the same CCX the old L3-keyed code did — 128
cores / 4 = the 32 domains layout A reports. Unchanged by construction.

A64FX (`cna0001`, job 1818396) is the opposite extreme, and worse than expected:

```
/sys/devices/system/cpu/cpu0/cache/     <- does not exist at all
thread_siblings_list   0                <- present, so core enumeration works
core_siblings_list     0-11
physical_package_id    0                (0..3 across the 48 cpus)
node possible: 0-3   node0 0-11  node1 12-23  node2 24-35  node3 36-47
Model name: A64FX    48 cpus, 1 thread/core, 12 cores/cluster, 4 NUMA nodes
```

No cache level is reported — not merely no L3. The cache walk therefore finds nothing and the
**NUMA fallback is the only signal left**, giving 4 domains of 12 cores where the old code
would have produced 48 singletons. This is the case the fallback was written for; it is not a
nicety.

`sbatch/barrier-groups-probe.sh` hardcodes `cache/index3` in its diagnostic echo and will print
empty on ARM. Repoint it at the NUMA cpulist before using it there.

## Two corrections to earlier assumptions in this directory

- **`MONOPROP_MAX_NUM_MODES=250` does not cover `--num-modes=256`.** The generated bindings do
  use 32-mode storage blocks, so 250 and 256 compile to the same template — but the Python
  dispatch validates the *mode count* against `MAX_NUM_MODES` and raises
  `NumberOfModesInvalidError` for 256. An A/B built at 250 must pass `NUM_MODES=250`, which
  keeps both arms internally consistent but not comparable with the historical
  `bench-layout-*` runs at 256.
- **`nproc` honours `OMP_NUM_THREADS`**, which `env.sh:126` pins to 1 for the BLAS pools. Every
  job here logs `cores: 1` while holding a full 128-CPU allocation. `hardware_concurrency()` is
  unaffected, so only the diagnostic misleads.

## Placement is now measured, not inferred

Fixing mechanism 1 by collapsing whenever the mask is narrower than the machine introduced a
regression of its own: mask *width* cannot distinguish "8 ranks holding 16 cores each" from "8
ranks sharing one 16-core mask" — both leave a rank seeing 16 of 128 CPUs — and the two need
opposite placement. Collapsing a genuinely shared mask points every co-located rank at the
*same* cores, which is worse than not pinning at all, and it breaks the invariant that two
ranks must never share a core.

`PartitionGroup::discover_node_peers_` therefore allgathers the raw affinity masks over the
node-local communicator it already opens, and `classify_node_mask` answers `PerRank` only for
masks that are pairwise disjoint. Anything else — identical masks, partial overlap, an
unreadable mask — answers `Shared`, which is the pre-existing behaviour and the safe direction.

## At production scale (~29M terms)

Everything above ran on a 828k-term operator, which is small enough that the barrier is most of
the collective. The sizing probe (`obs-terms=7000000 cutoff=6 --num-generators=100
--num-modes=250`) gives **28 965 342 terms** at 7.4 GiB peak RSS, so the A/B was repeated there:
2 nodes, `REPS=3`, `BENCH_ROUNDS=1`.

> **The ordering in this first 29M-term run was not interleaved, and the table below is therefore
> only partly trustworthy.** `ab-interleaved.sh` derived the A/B order from a single counter
> incremented once per cell, which gives layout *i* the values *i, i+N, i+2N, …* — alternating only
> when the layout count *N* is **odd**. The default three layouts hide this; passing
> `LAYOUTS_ONLY="A_1x128 B_8x16"` reduced *N* to 2 and froze the parity, so layout B ran `main`
> first in **every** rep and layout A ran `port` first in every rep. A first-vs-second position
> effect (page cache, clock ramp) is then indistinguishable from an effect of the code, which is the
> exact confound the script exists to remove — and it fails silently, since the summary still
> prints. Fixed by deriving the parity from `(rep + layout_idx)`, which is independent of how many
> layouts survive the filter. Re-run with `REPS=4` so each layout gets two of each ordering.
>
> Treat the two large *ratios* below as sound — a 1.45–1.56× gap is far outside any plausible
> position effect — and treat everything within a few percent of 1.00 as unresolved.
>
> **The clean re-run has since landed — see "The clean-order re-run" below, which supersedes this
> table.** It confirms the two large wins (and makes them larger), clears both small "regressions"
> as noise, and turns up one regression this table missed entirely.

| layout | operation | main med | port med | port/main | min agrees? |
|---|---|---:|---:|---|---|
| A 1×128 | build_graph | 4947 ms | 4592 ms | 0.93× | no |
| A 1×128 | energy | 210.9 ms | 197.2 ms | 0.93× | yes |
| A 1×128 | gradient | 406.6 ms | 381.0 ms | 0.94× | yes |
| A 1×128 | inplace | 4965 ms | 5313 ms | 1.07× | no |
| A 1×128 | pare | 15.1 ms | 14.5 ms | 0.96× | yes |
| B 8×16 | build_graph | 10902 ms | 10413 ms | 0.96× | no |
| B 8×16 | **energy** | 77.1 ms | 49.5 ms | **1.56× faster** | yes |
| B 8×16 | **gradient** | 103.0 ms | 71.0 ms | **1.45× faster** | yes |
| B 8×16 | inplace | 3665 ms | 3759 ms | 1.03× | no |
| B 8×16 | **pare** | 9.9 ms | 13.6 ms | **1.37× slower** | **yes (1.40×)** |

`pare` is **not** an evaluation criterion for this work: `build_graph`, `energy`, `gradient` and
`inplace` are what matter, and `pare` is a 10 ms operation against `build_graph`'s 10 s. It is kept
in the table because it is the most barrier-dominated operation in the suite, which makes it a
useful *diagnostic* for the two-level barrier even though it is not a target.

Two things changed relative to the 828k-term run, and both matter:

- **The layout A `build_graph` regression does not reproduce.** At 828k terms it was 1.10× on the
  median and 1.11× on the min — median and min agreeing, so it was recorded as real. At 29M terms
  the same cell is 0.93×. A `monoprop_SPIN_BUDGET_US` sweep over {5, 10, 30, 60, 120} on that
  exact cell moves it by at most ~5%, and median and min disagree at every setting except 60 µs
  (1.04× / 1.06×, i.e. mildly worse than the 30 µs default). So the spin budget is **not** the
  mechanism, the package-power hypothesis is unsupported, and the effect is size-specific rather
  than a property of the code. `kDefaultSpinBudgetUs = 30` stands, on x86, as a near-no-op.
- **Layout B `pare` regressed by 1.37× (1.40× on min).** Median and min agree. Not a target, but
  the mechanism is worth identifying because it is most likely the second barrier level rather than
  placement: grouping trades one `fetch_add` for two sequential hops, which can only pay where
  there is contention to relieve, and `pare` is short enough that the barrier is most of it.

On the operations that *are* targets, the first run leaves two questions open, and both are on the
two most expensive operations rather than the cheap ones:

- **Layout B `build_graph`** — `main` measured 10.902, 11.000, **7.660** s; `port` measured 10.366,
  10.413, 10.696 s. The median ratio is 0.96× (port faster) but `main`'s min is 26% below port's.
  `main` came in 30% under its own median exactly once while `port` stayed within 0.5% of its median
  in all three reps. One outlier in a three-rep sample against a tight distribution is not evidence
  of a regression, but it is not dismissable either, and `build_graph` is the largest operation in
  the suite. `main` also ran first in every one of those reps (see the ordering warning above).
- **Layout B `inplace`** — 1.026× on the median and 1.035× on the min. Median and min agree, which
  by the rule used throughout this directory makes it real; but the ranges overlap heavily
  (`main` 3.616–3.881, `port` 3.744–3.845) and `main` again held first position every rep. At 3.7 s
  a 3% regression costs more in absolute terms than the 1.56× `energy` win recovers, so this needs
  settling rather than waving through.

## Where a collective's time actually goes at 29M terms

Raw `COMMPROF` lines, 2 nodes, one bench invocation per layout (`costsplit.sh`). Every field is
already a per-rank aggregate over that rank's partitions; these are means over ranks.

| field | A 1×128 (2 ranks) | B 8×16 (16 ranks) |
|---|---:|---:|
| `barrier_groups` | 32 | 4 |
| `verbs` / `barriers` | 701 / 3202 | 701 / 3202 |
| `table_p0_s` | 0.005 | 0.001 |
| `table_par_s` | 0.046 | 0.008 |
| `table_move_s` | 0.158 | 0.026 |
| `mpi_s` | 1.207 | 0.482 |
| `barrier_p0_s` | 0.345 | 0.031 |
| `barrier_peers_s` | 1.400 | 0.511 |

Three conclusions, in descending order of how much they should change what happens next.

**1. `barrier_peers_s ≈ mpi_s`, and that is the dominant intra-node cost.** 1.400 vs 1.207 on
layout A; 0.511 vs 0.482 on layout B. Because MPI is funnelled through partition 0 to satisfy
`MPI_THREAD_SERIALIZED`, the other S−1 partitions sit in a barrier for almost exactly as long as
partition 0 spends inside MPI. This is the funnel's cost, it is the largest single item in the
table, and it is *not* what the plan's Stage 4 proposed to attack. Fewer, larger MPI calls reduce
it; overlapping the MPI leg with local work removes it. Note also that this is what inflates
`barrier_per_sync_us` to 150–435 µs here versus 15–22 µs at 828k terms — the barrier mechanism did
not get slower, the thing being waited *for* got bigger. Reading that number as a barrier
regression would be a mistake.

**2. The exchange is strongly latency-bound: `mpi_s / table_move_s` is 7.6× on layout A and 18.3×
on layout B.** This is the decisive measurement for collapsing the leader and follower exchange
passes into one. `build_layer` currently runs the exchange twice per gate — 20 barriers and 6 MPI
collectives, roughly 4× the plan's estimate of "4–6 barriers per verb". Merging them halves the
collective count while roughly doubling query bytes, which is a clear win at these ratios and
would have been a straight regression had the ratio come out near 1.

**3. The plan's headline Stage 4 item is dead, by measurement.** `table_p0_s / table_par_s = 0.12`
in *both* layouts — 5 ms of a 2.8 s wall on layout A, 1 ms on layout B. The residual serial
partition-0 prefix is not worth attacking. `RESULTS-scaling.md`'s 0.997 s floor at R=64 was
measured on `main` before this branch's `size_staging_parallel_`, and it names `pack_count_matrix_`,
a function that no longer exists; the prefix is now O(R·S) rather than O(R·S²). That item is
retired rather than deferred.

## The two-level barrier can now be measured, and never could be before

Its domains are derived from the cpusets, so the only way to obtain a flat barrier from outside
the process was to disable pinning — which also unpins. Every before/after in this directory
therefore confounds "grouped vs flat" with "pinned vs unpinned", *including* the evidence
originally used to justify the second level. `monoprop_BARRIER_GROUPING=0` (default on) forces the
flat path and leaves pinning alone.

The immediate question is layout B's `pare`. The prediction is directional, which is what makes
`grouping-ab.sh` a test rather than a fishing trip: flat should win on `pare` (short,
latency-critical, partitions arriving together, so no contention for the second level to relieve),
and grouped should win or tie on `energy` and `gradient`. If flat wins across the board, deleting
the second level is a simplification with a measurement behind it — roughly 270 production and 145
test lines.

---

## The clean-order re-run

`REPS=4`, parity derived from `(rep + layout_idx)`, 2 nodes, 29M terms. Per-rep **minima** in
seconds, in rep order — the medians and the summary ratios are in
`runs/ab-interleaved-30M-fixorder-N2/AB-SUMMARY.md`. Per-rep numbers are given because two cells
turn on the *shape* of the distribution rather than on a ratio.

| layout | operation | main | port | verdict |
|---|---|---|---|---|
| B 8×16 | **energy** | 0.078 0.088 0.099 0.067 | **0.036 0.036 0.046 0.040** | **2.1× faster** |
| B 8×16 | **gradient** | 0.095 0.106 0.105 0.098 | **0.072 0.068 0.072 0.073** | **1.36× faster** |
| B 8×16 | inplace | 3.974 3.770 3.871 3.668 | 3.808 3.751 3.714 3.687 | tie |
| B 8×16 | build_graph | 10.63 11.21 7.18 10.55 | 10.25 10.51 10.43 7.24 | unresolved |
| A 1×128 | build_graph | 5.632 5.089 4.651 4.433 | 4.729 5.828 5.291 6.299 | **~1.1× slower** |
| A 1×128 | energy | 0.369 0.207 0.195 0.226 | 0.217 0.209 0.197 0.177 | tie |
| A 1×128 | gradient | 0.489 0.372 0.369 0.409 | 0.407 0.421 0.397 0.460 | tie |
| A 1×128 | inplace | 8.641 4.781 5.153 4.686 | 4.489 5.093 4.767 4.878 | tie |

**`energy` and `gradient` on layout B are settled.** The distributions do not overlap at all —
main's *best* rep is worse than port's *worst*. No ordering artefact can produce that.

**Both earlier "regressions" were noise.** `inplace` B (previously 1.03×) is flat. And layout B's
`build_graph` had a 7.66 s outlier against a ~11 s cluster that looked alarming; it now appears in
**both** arms (main rep 3 = 7.18, port rep 4 = 7.24), so it is a bimodal machine effect, not code.
That is a useful reminder that a single outlier in one arm is not evidence until the other arm has
been given the same number of chances to produce one.

**One regression this table found that the frozen-order run missed: layout A `build_graph`.** The
raw ratio is ~1.14× on the median, but the interesting part is that *both arms run faster in first
position*, so the comparison has to be position-matched:

| | main | port | port/main |
|---|---:|---:|---:|
| ran first (reps 2, 4) | 4.76 | 5.01 | 1.05× |
| ran second (reps 1, 3) | 5.14 | 6.06 | 1.18× |

Port is slower in both halves, so the effect survives the position correction — but the position
effect itself (~1.1×) is the same size as the signal, on 4 reps. Call it **~1.1× slower, real but
not tightly bounded**. This is one of the four evaluated operations and the only open regression in
the suite. Note it is also the same regression that appeared at 828k terms, was reported, and was
then withdrawn as non-reproducing at 29M — **that withdrawal was premature**; it was the frozen
ordering that hid it. Mechanism is still unidentified: layout A is the one layout where pinning
already engaged on `main`, so the mask fix cannot explain it, and the spin-budget sweep moves it
≤5%.

## Grouped vs flat barrier: keep the second level

`monoprop_BARRIER_GROUPING=0` forces the flat path without touching pinning, which is what makes
this measurable at all. `REPS=4`, 29M terms, 2 nodes. `flat/grouped > 1` means grouped is faster.

| layout | operation | flat/grouped (med) | (min) | verdict |
|---|---|---:|---:|---|
| A 1×128 | build_graph | 1.20 | 1.15 | **grouped wins** |
| A 1×128 | gradient | 1.10 | 1.11 | **grouped wins** |
| A 1×128 | energy | 1.05 | 1.03 | grouped, small |
| A 1×128 | inplace | 1.14 | 0.89 | unresolved |
| B 8×16 | gradient | 1.11 | 1.13 | **grouped wins** |
| B 8×16 | everything else | 0.84–1.01 | 0.96–1.12 | tie |

**No cell shows the flat barrier robustly faster.** The second level pays at S=128 and is a wash at
S=16, which is what the mechanism predicts: contention on the root word scales with participant
count, so there is little to relieve at 16 and a lot at 128. The prediction recorded before the run
— that flat would win on `pare` — was **wrong**; `pare` is a tie in both layouts.

So the ~270 production and ~145 test lines stay. This also independently confirms the risk the plan
flagged against adopting OpenMP: a centralised barrier would lose at S≈128 pinned.

## Funnel width: the cheap lever is not free, and it is not uniform

Zero code change, sweeping ranks-per-node at a fixed 128 cores/node (`funnel-width.sh`). Prediction
recorded in the script beforehand: 8×16 sits at the minimum and both directions lose. Wall clock,
min over 2 reps, seconds:

| layout | R | S | build_graph | energy | gradient | inplace |
|---|---:|---:|---:|---:|---:|---:|
| 4×32 | 8 | 32 | **9.593** | 0.071 | 0.120 | 4.506 |
| 8×16 | 16 | 16 | 10.465 | 0.051 | **0.070** | **3.584** |
| 16×8 | 32 | 8 | 11.347 | **0.043** | 0.093 | 3.639 |
| 32×4 | 64 | 4 | 13.840 | 0.074 | 0.154 | 4.311 |

The prediction **holds for `gradient` and `inplace`** and **fails for `energy`** (16×8 wins by
1.19×) and **`build_graph`** (4×32 wins by 1.09×). Different operations prefer different funnel
widths, in opposite directions, which a single documented default cannot express.

This is deliberately *not* yet a docs change. Two reps is thin, and the cost decomposition that
would explain it was lost to the instrumentation defect below — so the observation is recorded and
the recommendation is left alone until the re-run attributes it.

## The instrument was silently returning nothing

`COMMPROF` is the primary instrument for all of the above, and three separate 29M-term measurements
came back with empty cost tables before the cause was found.

**pytest's default capture is fd-level.** It replaces file descriptor 2 for the duration of each
test and discards the buffer when the test **passes**. `CommProfile::dump()` runs in the transport
*destructor* and writes with `std::print(stderr, ...)` — a direct fd-2 write from C++ that never
passes through Python's `sys.stderr`. Measured on one command and one tree: **0 `COMMPROF` lines
without `-s`, 3 with it.** Redirecting `>log 2>&1` does not help; an earlier version of
`funnel-width.sh` carried a confident comment claiming combined redirection was the fix, and that
comment is why the sweep above ran and produced an empty cost table.

`-s` is now passed by `funnel-width.sh` and `barrier-groups-probe.sh`, and each cell asserts **one
`COMMPROF` line per rank**, printing `COST DATA MISSING` on a short count. That assertion matters
more than the flag: without it, *"the two arms are identical"* and *"the instrument never fired"*
are the same observation.

**Method rule, earned the hard way: never diagnose by comparing two zeros.** `--benchmark-json` was
"exonerated", then `cwd` was "exonerated", each time by observing that both arms produced 0 — both
comparisons were between two failures and established nothing. What localised the fault in a single
step was a **positive control**: the C++ transport tests emit 34 `COMMPROF` lines under
`monoprop_COMM_PROFILE=1`, which placed the fault in the Python path immediately. The same trap
appears in shell, where `binary … 2>&1 | grep -c PAT` prints `0` when the binary never started —
a missing `module load` had killed it on a `libboost` link error, and `$?` reports grep's status,
not the binary's.

## Funnel width, with the instrument working — and a result that reorders the plan

Same sweep re-run with `-s` (job 1819604, `REPS=2`, 29M terms, 2 nodes). `COMMPROF` now captures:
3 lines per rank, since each rank tears down several transports per run.

| layout | R | S | `mpi_s` | `barrier_peers_s` | `table_move_s` | `barrier_groups` |
|---|---:|---:|---:|---:|---:|---:|
| 4×32 | 8 | 32 | 1.143 | 1.200 | 0.045 | 8 |
| 8×16 | 16 | 16 | 0.419 | 0.449 | 0.027 | 4 |
| 16×8 | 32 | 8 | **0.405** | 0.433 | 0.016 | 2 |
| 32×4 | 64 | 4 | 0.871 | 0.889 | 0.006 | 0 |

`barrier_groups` = S/4 as expected on EPYC's 4-core CCX, reaching 0 at S=4 where one group is
degenerate and the flat path correctly takes over. `pinned` reads `nan` because the built tree
predates the pin-count change — the parser reporting a missing field rather than inventing a value
is the intended behaviour.

### `barrier_peers_s ≈ mpi_s` is now confirmed four more times

The ratio is **1.02–1.07 across every layout**. This is the most robust finding in this directory:
the S−1 non-funnel partitions idle for as long as partition 0 spends inside MPI, at every funnel
width tested.

### Prediction scorecard

The prediction was recorded in the script before the run, which is the only reason it can be scored.

| layout | predicted `mpi_s` | observed | error |
|---|---:|---:|---:|
| 4×32 | 0.62 | 1.143 | **1.84×** |
| 8×16 | 0.48 | 0.419 | 0.87× |
| 16×8 | 0.56 | 0.405 | **0.72×** |
| 32×4 | 0.90 | 0.871 | 0.97× |

The qualitative claim — *the recommended layout is already in the optimal basin, so there is no cheap
configuration win* — **holds**: 8×16 and 16×8 are within 3% of each other at the bottom, and both
outer points are 2.1–2.7× worse. The sharp form of it was **wrong**: the minimum is a broad basin
spanning 8×16–16×8, not a point at 8×16, and 16×8 was predicted 1.17× worse than 8×16 when it is
marginally better.

The linear two-term model is too crude, which is what a 2-point fit deserves. Neither term is
linear: 4×32 carries 1.87× the bytes for **2.73×** the `mpi_s` (super-linear), while 32×4 carries
4.20× the messages for **2.08×** (sub-linear). The *ordering* the model was used for — bytes bind at
low rank counts, messages at high ones — survives; the magnitudes do not.

### The finding that matters most: comm is a few percent of wall time

`COMMPROF` accumulates over the whole process lifetime, across all benchmarks, so `mpi_s` must be
compared against the **sum** of the operations and never against one of them. Against the sum of the
four evaluated operations:

| layout | evaluated wall | `mpi_s` | comm share |
|---|---:|---:|---:|
| 4×32 | 13.20 s | 1.143 | 8.7% |
| 8×16 | 14.02 s | 0.419 | **3.0%** |
| 16×8 | 14.56 s | 0.405 | 2.8% |
| 32×4 | 17.97 s | 0.871 | 4.8% |

**At the recommended layout the entire MPI leg is 3% of the work.** Perfectly eliminating the funnel
would therefore buy ~3% at this scale — which puts a ceiling on the whole Stage 4 programme, and on
the candidate ranking recorded earlier in this file. The candidates are real (the self-rank leg does
go through `MPI_Alltoallv`; the `Ialltoallv` overlap is genuinely discarded for `Kind::Hybrid`) but
at 2 nodes they are chasing a few percent.

**Layout choice is worth more than the comm rewrite here.** `build_graph` is 1.25× faster at 4×32
than at 8×16, consistently across both runs (9.59 → 8.31 vs 10.46 → 10.38), *despite* 4×32 having
2.7× the comm cost — so `build_graph`, the largest operation in the suite, is not comm-bound at all.
It improves with more partitions per rank. `inplace` prefers 8×16 in both runs (3.49 vs 4.66), so
the two largest operations genuinely conflict, and 4×32 wins the total only because `build_graph`
dominates the sum.

Note also that the earlier run's apparent "`energy` prefers 16×8" **did not reproduce** (0.043 →
0.063, with 8×16 now better at 0.054). Two reps is not enough for the small operations; treat only
`build_graph` and `inplace` as having a consistent layout preference.

### What this leaves open

The decisive question for Stage 4 is no longer *which* comm candidate to implement but *whether the
comm layer matters at the node counts that are actually used*. `RESULTS-scaling.md` measures a
failure to scale past roughly 8 nodes, so the comm share should grow with node count — but that is
an inference, and every inference in this directory that went unmeasured has so far been wrong. A
node-count sweep reporting comm share at N = 2, 4, 8, 16 would settle it, and it is the only
measurement that can justify or retire Stage 4 as a whole.
