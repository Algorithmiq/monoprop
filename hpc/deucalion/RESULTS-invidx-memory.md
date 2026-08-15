# Inverted-index memory: `perf/invidx-memory` vs its base

2026-08-15, Deucalion x86. Harness `sbatch/ab-100m.sh`, job `1825647`, summary under
`$PROJ/runs/ab-100m-invidx-vsmain-N1/AB-SUMMARY.md`.

| | |
| --- | --- |
| arms | `origin/main` @ `6abd839` (0.8.1.dev37) vs the branch, all four changes present |
| problem | 7M observable terms → **29,007,110 propagated terms**, 250 modes, cutoff 6, 100 generators |
| layout | B_8x16 — 8 ranks × 16 partitions, world 128, so ~226k rows per inverted index |
| reps | 6, interleaved, order flipped per (rep, cell), `--bench-rounds=1` |
| identical | `MAX_NUM_MODES=1024`, `MALLOC_ARENA_MAX=16`, `OMP_NUM_THREADS=1`, MPI on, same `benches/` |
| placement | 16 threads pinned per rank on **both** arms, under `--cpu-bind=none` |

Term counts agree to the term across all cells.

> **This supersedes the 2026-08-14 measurement** (jobs `1821536`/`1821568`, dirs
> `ab-100m-invidx{,2,3}-N1`), which read 5.17× on the index and 1.40× on the operator. That
> measurement is wrong twice over: its base was `bench/hpc-harness` @ `97c1fb6` rather than
> main, and its branch arm predated three of the four changes — visible in its own table,
> where `indexing_bytes` and `operator_terms_bytes` came back byte-identical across the arms,
> which cannot happen once the dedup slot and the chunked row store are present. Those runs
> were also refused for unplaced threads. The numbers below replace them.

## Result — Majorana, the regime the change targets

**The operator is 2.97× smaller: 129.67 → 43.59 B/term, 3.76 GB → 1.26 GB, so 2.33 GiB off a
29M-term operator.** The four changes do not contribute in the expected order:

| field | main B/term | branch B/term | main/branch | Δ B/term | share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `init_operator_bytes` | 39.58 | **0.00** | — | 39.58 | **46%** |
| `inverted_index_bytes` | 46.20 | **7.87** | **5.87×** | 38.33 | **45%** |
| `indexing_bytes` | 18.51 | 11.57 | 1.60× | 6.94 | 8% |
| `operator_terms_bytes` | 17.38 | 14.17 | 1.23× | 3.21 | 4% |
| `matched_scratch_bytes` | — | 1.98 | — | −1.98 | −2%, the branch pays |
| **`total_bytes`** | **129.67** | **43.59** | **2.97×** | 86.08 | 100% |

The largest single win is **not the index**. It is releasing `init_op_map`, which holds
**1.15 GB** of empty buckets here that `erase` never returns. That is a large-observable
effect — 7M observable terms — and it disappears on the fixed models, where
`init_operator_bytes` is a few kilobytes on both arms and the total ratio is 1.35×. Quote
2.97× only with the observable size beside it.

The index falls from 36% of the operator to 18%, so the monomial→row hash (11.57) and the row
store (14.17) are now the two largest structures, and the two worth attacking next.

### What is actually resident

Capacity is not residency, so the ledger is quoted beside the suite's own per-rank `VmHWM`,
`_reduce_sum`-ed over the 8 ranks and medianed over 6 reps. Both arms share a geometry, so
the ratio is meaningful even though the summed total is not comparable across layouts:

| operation | main GiB | branch GiB | main/branch | GiB saved |
| --- | ---: | ---: | ---: | ---: |
| `build_graph` | 34.99 | 32.79 | 1.07× | 2.20 |
| `propagate` | 34.57 | 33.03 | 1.05× | 1.54 |
| `energy` | 14.71 | 12.16 | **1.21×** | 2.55 |
| `gradient` | 14.94 | 12.46 | **1.20×** | 2.48 |

The ledger's 2.33 GiB and the kernel's 1.5–2.6 GiB agree here, which they do **not** on the
fixed models. The whole-job ratio is 1.05–1.21×, not 2.97×, because the operator is about a
tenth of the footprint at this size; `energy` and `gradient` show the most because they do not
also hold the transient build structures.

Where the branch's index bytes go, from the earlier run's breakdown (8.94 B/term there,
7.87 here): 8.43 byte-delta payload (of which 3.04 is the unsealed
tail), 0.43 of bitmap cells on the handful of columns dense enough to earn one, 0.07 directory,
and 0.01 arena slack. The base's 46.20 is 33.77 of full-height bitmaps plus 12.43 of 4-byte
postings — and that bitmap figure is the point, since only ~95 of 500 columns qualify as dense
by their *final* density while promotion is one-way, so most of those 33.77 bytes are held by
columns that no longer deserve them.

## The fold is 1.4–1.9× slower, and that is the whole cost

Measured directly rather than inferred: the same 200 random 4-column generators folded over a
synthetic operator carrying the *measured* Majorana density skew (p25 0.0066, p50 0.0119, p90
0.0179, max 0.0346, w̄ 5.71), on one pinned core, arms interleaved and the order flipped per rep.
`$PROJ/runs/invidx-build/fold2-1821565.log`.

| rows per index | base ms/gen | branch ms/gen | branch/base | agree | spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| 226,618 *(the production per-partition size)* | 0.0091 | 0.0175 | **1.92×** | 8/8 | 1.30× |
| 776,886 | 0.0354 | 0.0605 | **1.72×** | 8/8 | 1.10× |
| 4,969,038 | 0.2825 | 0.3980 | **1.41×** | 8/8 | 1.08× |

This is the quantity actually at stake, it is resolved 8/8, and it is measured on a
single-threaded kernel whose output is bit-identical between the arms — none of which is true
of the whole-operation A/B below. The density skew is load-bearing: on a *uniform* synthetic the
base has no dense columns at all, both arms scatter postings, and the ratio flatters the branch
to 1.05×.

## Timing is not resolved here, and the reason is not the change

`ab_summary.py` **REFUSED** every one of these runs: neither arm placed a single thread. `main`
has no `node_mask` on `partition_cpusets`, so under `srun --cpu-bind=cores` at 8 ranks/node
every rank runs unpinned — a property of the base commit, applying equally to both arms (the
`/proc` probe reports 0 single-CPU threads per rank on both). Unpinned, per-rep spreads run
1.6–2.6× and almost every operation comes back *unresolved* at 6 reps.

Two independent 6-rep runs of the identical comparison land on **opposite signs**:

| operation | run 1 branch/base | agree | run 2 branch/base | agree |
| --- | ---: | ---: | ---: | ---: |
| `build_graph` | 1.10× slower | 3/6 | 0.77× faster | 4/6 |
| `propagate` | 0.95× | 4/6 | 0.82× faster | 5/6 |
| `energy` | 0.92× | 4/6 | 0.94× | 5/6 |
| `gradient` | 1.02× flat | 6/6 | 1.06× flat | 5/6 |

That disagreement is the result: at this rep count the harness cannot see an effect of either
sign, and neither can anyone reading it. It is emphatically not evidence that the branch is
faster, which is what run 2 alone would suggest. The fold measurement above is the timing
claim; this table is only here to show it was checked end to end.

### Placement was not the cause — tested, and refuted

The paragraph above blames the spread on unpinned threads. **That explanation is wrong.** The
2026-08-15 re-run (job `1825647`) places 16 threads per rank on both arms under
`--cpu-bind=none`, and the timing is *still* unresolved, with spreads no better:

| operation | branch/main | agree | per-rep spread |
| --- | ---: | ---: | ---: |
| `build_graph` | 1.01× | 3/6 | 4.4× |
| `propagate` | 1.31× | 4/6 | 4.6× |
| `energy` | 1.08× | 5/6 | 1.3× |
| `gradient` | 1.06× | 6/6 | — |

Unpinned the spreads were 1.6–2.6×; pinned they are 1.3–4.6×. Pinning fixed a real bug and
changed nothing about the variance, so the residual is the harness, not the placement — which
is what the direct fold measurement already implied by resolving cleanly on one pinned core.
The fixed-model cells resolve small effects at the same rep count (`propagate` 1.05× at 6/6 on
pauli c12), so this is a property of the random problem's per-rep setup, not of six reps being
too few everywhere.

One thing does reproduce across both runs: `gradient`'s peak above its own floor is ~20 MB
higher on the branch (0.03 → 0.05–0.06 GiB, on a 3.74 GiB operator), while the operator is
1.08 GiB smaller at rest. Some of that was per-seal churn in the tail buffers — reshaping
`kNumColumns` buffers on every seal to move a capacity that barely moved — which this change
now skips unless the buffer is genuinely the wrong size.

## Pauli: the regime that corrected the design three times

127-qubit kicked Ising, cutoff 8, 20 layers, 223,372 terms, serial. Every partition here holds
~1.7k rows — far short of a 65,536-row chunk — so **nothing is ever sealed** and the index is
*entirely* the growing tail. Pauli column densities run to 0.44, an order of magnitude past
anything Majorana reaches.

| field | base B/term | branch B/term | base/branch |
| --- | ---: | ---: | ---: |
| `inverted_index_bytes` | 11.06 | **7.22** | **1.53×** |
| `total_bytes` | 60.93 | 57.08 | 1.07× |

The index is only the third-largest structure here (row store 22.96, hash 18.86), so 1.53× on
it is 1.07× on the operator. Quote that pair, not the Majorana one.

Getting there took three corrections, each forced by a measurement and none visible from the
Majorana side:

| tail design | B/term | vs base |
| --- | ---: | ---: |
| one container, a byte-delta stream | 14.81 | **0.75× — a regression** |
| delta or bitmap, switching only past 4096 rows | 11.95 | 0.93× — still a regression |
| delta or bitmap, switching on cost | **7.22** | **1.53×** |

The first was the plan's own "the tail is delta-encoded too, and that is the simpler choice",
and at density 0.44 a byte per posting loses badly to a bit per row. The second added the
bitmap but gated it on having seen 4096 rows, to keep a briefly-busy column from buying a
bitmap it would not fill — and every partition here holds 1.7k rows, so the gate never opened
and `d_invidx_bitmap_chunks` came back 0. The row gate was the wrong shape: a tail bitmap only
grows *on a push*, so what needs bounding is its cost, not its age. A delta stream never
exceeds 3 bytes per posting, so a bitmap past 4·m is provably the worse of the two and converts
back — an O(1) test with a provable bound, and the row threshold disappears.

This is also why the design doc's Pauli estimate could not be trusted: it costed the *sealed*
menu, and in this regime nothing seals.

## What the measurements changed about the plan

1. **The plan predicted 6.36 B/term for Majorana; the measurement is 8.94.** The model costed
   container payloads and left out the growing tail's capacity, the directory, and the true
   escape rate under a non-uniform density. 8.94 against a 46.20 baseline is still 5.17×.
2. **The plan's "the tail is delta-encoded too, and that is the simpler choice" was wrong**,
   and only Pauli showed it. See above.
3. **`std::vector` growth had to be taken off three separate paths**, each found by
   measurement, not inspection: the arena (projected once at the first seal, worth 0.81
   B/term at 5M rows), the per-chunk tail buffer (sized from the previous chunk, 2.2 B/term
   at 234k rows), and the tail's own growth (quartered rather than doubled, which is the
   whole index before anything seals). The arena projection needs a 1/32 margin, and that is
   not decoration: an exact projection lands ~0.01% short, and coming up short hands the whole
   reservation to the geometric fallback, which costs the full 12.5% — measured, the
   "improvement" made slack *worse* until the margin went in.
4. **`monoprop_WIDE_TERM_INDEX` stops costing anything.** Chunk-local 16-bit ids make the
   index byte-for-byte identical in both builds, where the base doubles: 24.00 → 48.00
   B/term on a matched synthetic against 6.98 → 6.98.

## Bit-identity

The fold is bit-identical to the base at every size tested. A synthetic harness folding 200
random 4-column generators over 234k / 777k / 4.97M rows returns the same XOR-accumulated
sink from both arms, and the same sink again with `monoprop_INVIDX_BITMAP_M_STAR` forced to
256, 700, 1024 and 4096 — which is the claim that matters, since container choice is
supposed to be invisible to the answer. That escape hatch trades as advertised: at M\*=256
every chunk becomes a bitmap, 8.7× the memory for a 2.4× faster fold.

---

# Round 2: hold the memory, take the time back

2026-08-14, same machine and same harness. Jobs `1822224` (end-to-end), `1822348` (four-arm
fold), `1822387` (fold under concurrency), `1822266`/`1822376` (memory, both layouts).

Round 2 keeps every round-1 container decision and changes three things: a whole-chunk fast
path in the fold kernel (**T1**), a chunk height decoupled from the fold block and set to 512
words (**M1**), and the deletion of the `U16` container (**S1**). It also settles the timing
question round 1 left open.

## Two instruments, and why the numbers look different

`inverted_index_bytes` counts the sealed arena and the directory. It does **not** count the
unsealed tail, which at the 8×16 layout is a quarter of the structure. Round 1 quoted that
field; this section quotes it too, for continuity, and gives the all-in figure beside it.
Read one or the other consistently — mixing them looks like a regression that is not there.

A third trap, and it bit this report once: on the **base** arm `d_invidx_sparse_bytes` is a real
component (its 4-byte postings), while on the branch it is an *alias* of `d_invidx_delta_bytes`.
Summing the components without testing for that alias double-counts the branch or, worse, silently
drops 12.4 B/term from the base. Detect the alias; do not assume either shape.

**B/term also depends on the partition layout**, because the tail and the directory are
per-partition costs amortised over that partition's rows. The same code reads 6.02 B/term at
16 partitions and 7.87 at 128. Quote the layout with the number.

| layout | field | base | round 1 | **round 2** | r2 vs base | r2 vs r1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1×16 | `inverted_index_bytes` | 46.23 | 6.78 | **6.02** | **7.68×** | 1.13× |
| 1×16 | index, all-in (incl. tail) | 46.23 | 7.10 | **6.18** | **7.48×** | 1.15× |
| 1×16 | `total_bytes` | 130.58 | 91.14 | **90.38** | **1.445×** | 1.008× |
| 8×16 | `inverted_index_bytes` | 46.20 | 8.86 | **7.87** | **5.87×** | 1.13× |
| 8×16 | index, all-in (incl. tail) | 46.20 | 11.82 | **9.78** | **4.73×** | 1.21× |
| 8×16 | `total_bytes` | 129.67 | 92.33 | **91.34** | **1.420×** | 1.011× |

Memory is deterministic here — every rep of every run agreed to the byte.

## Where the round-2 index memory went (1×16, 29,007,110 terms)

| component | B/term | share |
| --- | ---: | ---: |
| byte-delta payload | 4.637 | 75.1% |
| bitmap cells (6,979 chunks) | 0.980 | 15.9% |
| arena slack | 0.258 | 4.2% |
| unsealed tail | 0.157 | 2.5% |
| directory | 0.145 | 2.3% |

The delta payload sits at 1.12× the gap entropy, so three quarters of what remains is
provably near its floor. **Removing everything else in the index would save 1.7% of the
operator.** The index is finished as a memory target — see "where the memory is now" below.

## The fold: what T1 bought, what M1 cost

Four arms, same binary set, one pinned core, 8 reps, order rotated per rep. Every row 8/8
paired, p=0.0078, spread ≤1.16×. `foldab2-1822348.log`.

| step | 226k rows | 777k | 4.97M | what it isolates |
| --- | ---: | ---: | ---: | --- |
| base → round 1 | 2.076× | 1.794× | 1.438× | the round-1 regression |
| **round 1 → T1** | **0.885×** | **0.883×** | **0.880×** | T1 alone |
| T1 → M1 (C=1024→512) | 1.022× | 1.039× | 1.066× | what M1's memory costs in time |
| base → round 2 | 1.907× | 1.634× | 1.351× | net |

T1 is a flat **1.13× on every size**, which is what a per-posting µop saving should look like.
It is less than the 1.45× the plan estimated: the estimate assumed the bounds checks were ~2
of ~9 µops and that removing them scaled through, and it did not.

## The solo fold measurement systematically flatters the base

This is the round's most consequential correction. The A/B above runs **one** thread on an
idle node. Production runs 16 partitions per rank, sharing L3 and memory bandwidth, and the
two arms differ 3.8× in footprint — so the isolated measurement is precisely the regime that
favours the larger index. Re-measured at n=1.94M (the per-partition size of the 31M-term
end-to-end run), 6 reps, 6/6 on every row. `foldconc-1822387.log`.

| arm | B/term | solo vs base | **16 concurrent vs base** | contention penalty |
| --- | ---: | ---: | ---: | ---: |
| round 2 (M\*=0) | 6.87 | 1.181× | **1.084×** | 1.115× |
| M\*=850 | 7.25 | 1.146× | 1.076× | 1.145× |
| M\*=700 | 9.66 | 1.122× | 1.058× | 1.132× |
| M\*=600 | 11.81 | 1.084× | 1.043× | 1.169× |
| base | 26.54 | 1.000× | 1.000× | 1.214× |

Two conclusions. **The fold regression is 1.084×, not 1.181×**, once the arms compete for the
memory system the way they actually do. And **the M\* frontier is nearly flat in production**:
buying back the last 8.4% of fold time costs 3.9× the index, and even 1.7× the memory (M\*=600)
recovers only 4.1%. The escape hatch stays off, and now for a measured reason rather than an
argued one.

## What did not work

- **The branchless escape decode.** The delta reader's one remaining branch is the 0xFF escape,
  taken ~4.8% of the time and unpredictable in principle. Reading the wide gap unconditionally
  and selecting with a cmov measured **1.015–1.04× slower** (4/4 and 4/4, identical sink): the
  cmov puts the pointer advance on the loop-carried dependency chain, and a 4.8% branch is
  predicted well enough that there was little to recover. Not shipped; recorded in the kernel.
- **T2, the budgeted fold cache — rejected, not deferred.** It caches folds across repeated
  `ev`/`ev_and_grad` calls, so it can only help `energy` and `gradient`. The end-to-end
  measurement shows those two are *already faster* on the branch (0.993×, 0.994×), while the
  entire penalty sits in `build_graph` and `propagate`, which fold once and have nothing to
  reuse. It would have cost 12.5–23.5 B/term to speed up the two operations that did not need it.
- **`M_STAR` as the tuning knob.** See the table above.

## Index construction got faster

`build_graph` and `propagate` both rebuild the index, so the build path needed its own
instrument (`bench_build.cpp`, interleaved, 4 reps).

| rows | base | round 2 | |
| --- | ---: | ---: | --- |
| 226,618 | 30.6 ms | 20.0 ms | **1.53× faster** |
| 776,886 | 106.5 ms | 65.5 ms | **1.63× faster** |

The arms are indistinguishable across `r1`/`c1024`/`c512`, so this is not T1 or the chunk
height — it is that the base writes 26.67 B/term and round 2 writes 7.02. This is why the
end-to-end penalty is far smaller than the fold ratio alone predicts.

## T4 — prefetch the next column's chunk: a real kernel win, rejected

The fold walks a generator's columns for one chunk at a time. The arena is chunk-major, so those
columns land far apart inside the chunk's segment, and the next column's stream is a cold line by
the time the current one is decoded. A single `__builtin_prefetch` of the next column's payload,
issued from the directory entry the loop is about to read anyway, costs nothing and hides it.

| n per index | solo | **16 concurrent** | agree |
| --- | ---: | ---: | ---: |
| 226,618 | 0.987× | **0.959×** | 5/6, 6/6 |
| 1,940,000 | 0.958× | **0.902×** | 6/6, 6/6 |

The contended case is the bigger win, which is what a latency fix should look like, and it is the
case production runs. Bit-identical sinks at 226k / 1.94M / 4.97M rows, and identical B/term.

Composed with the fold gap this predicts closure: 1.084 x 0.902 = 0.978x, i.e. a round-2 fold
*faster* than the base it replaced at 3.9x less memory.

**The prediction did not survive, and the change is not shipped.** Measured end-to-end at three
points, it moved nothing and two of the three went slightly the wrong way:

| workload | without prefetch | with prefetch |
| --- | ---: | ---: |
| Hubbard c10, 26.6M | 1.011x | 1.010x |
| kicked Ising c10, 2.7M | 1.088x | 1.100x |
| kicked Ising c14, 31M | 1.0128x (6/6) | 1.0142x (6/6) |

Eight lines that a kernel benchmark loves and no user can measure are eight lines that do not go in.
This is the same standard T3 was rejected under, applied to a change that was much more tempting.

## T3 — the dense 16-byte scan word, rejected

`EvenParityNzWord` was 24 bytes `{base, overlap, foll}`, and `base` is derivable from the entry's
position. Rebuilt as a **dense** 16-byte `{overlap, foll}` array indexed by word position (a
compacted array cannot drop `base`; a dense one can, and at ~94% word density dense is the smaller
of the two). It is correct — 209/209 `ctest -L serial` and 609/609 pytest — and it measured
**nothing**: hubbard at 26.6M terms gives `propagate` 1.000× with **3/6 reps agreeing**, i.e.
unresolved, and Pauli 1.004–1.006×, marginally the wrong way. Reverted. A rewrite of the fused
pass-1/pass-2 loop needs a reason, and "no measurable effect" is not one.

## End-to-end: the penalty is real, small, and depends on operator size

The acceptance criterion is summed wall clock (`propagate` + `build_graph` + `energy` + `gradient`),
paired per rep, on a pinned 16-partition process — the same per-rank shape as the production layout.
Six reps per point, arm order flipped per rep.

| workload | terms | index base→r2 | operator | **summed wall clock** | agree |
| --- | ---: | ---: | ---: | ---: | ---: |
| kicked Ising c6 | 2,978 | 7.40 → 6.35 | 1.022× | 0.994× | 4/6 *(no signal)* |
| kicked Ising c8 | 223,372 | 11.40 → 6.52 | 1.089× | 1.018× | 5/6 |
| kicked Ising c10 | 2,655,112 | 14.50 → 7.41 | 1.129× | **1.088×** | 6/6 |
| kicked Ising c14 | 30,992,523 | 16.67 → 8.35 | 1.119× | **1.013×** | 6/6 |
| Hubbard c6 | 9,519,091 | 15.22 → 6.83 | 1.225× | 1.000× | 3/6 *(no signal)* |
| Hubbard c8 | 20,883,753 | 13.85 → 6.81 | 1.182× | 1.016× | 5/6 |
| Hubbard c10 | 26,607,878 | 12.28 → 6.27 | 1.121× | **1.011×** | 6/6 |

**The index is ~2× smaller on every workload at every size, and the time cost is not monotone in
size — it peaks in the middle.** At 2.7M terms the penalty is 8.8%; at 31M it is 1.3%. The
mechanism is cache residency: at 166k rows per partition the base's 3.8×-larger index still fits,
so its bitmaps are free and it wins on decode cost; by 1.9M rows per partition it no longer fits,
its bandwidth disadvantage dominates, and the gap closes. The same effect appears directly in the
solo-vs-concurrent fold table above.

At the ~30M sizes this round was asked to optimise, the end-to-end cost is **1.1–1.3%**. It is not
flat, and it is resolved rather than noise (6/6, p=0.031) — so it is a cost, not an absence of one.

## The fold microbenchmark does not predict end-to-end, and that is a finding about the instrument

This round steered its kernel work with a synthetic fold benchmark. Two changes it endorsed
strongly produced almost nothing end-to-end:

| change | fold microbenchmark | end-to-end |
| --- | ---: | ---: |
| T1 (whole-chunk fast path) | 0.880–0.885×, 8/8 | not separable from noise |
| T4 (prefetch) | 0.902× concurrent, 6/6 | nothing; two of three points slightly the wrong way — rejected |

Both are real, resolved gains *in the kernel*. Neither moved the workload. The fold is simply a
smaller share of `propagate` than the microbenchmark's ratios imply, and a benchmark that folds 200
generators back-to-back over a freshly built index has better locality than the real sweep does.
**Quote the microbenchmark for kernel work and the end-to-end for acceptance; do not convert one
into the other.** Both T3 and T4 were rejected on that principle; T1 and M1 survive on the memory
and correctness evidence, not on their fold ratios.

## Where the memory is now — the index is no longer the target

The point of this table is that the two rounds finished the job they were given. At 1×16,
29,007,110 terms:

| component | B/term | share of operator |
| --- | ---: | ---: |
| `init_operator_bytes` | 39.58 | 43.8% |
| `indexing_bytes` (monomial → row hash) | 18.51 | 20.5% |
| `operator_terms_bytes` (the row store) | 18.27 | 20.2% |
| `op_coeffs_bytes` | 8.00 | 8.9% |
| **`inverted_index_bytes`** | **6.02** | **6.7%** |
| `d_terms_slack_bytes` | 4.27 | 4.7% |

The inverted index went from 25.9% of the operator to 6.7%, and three quarters of what is left
is a delta stream at 1.12× its entropy. Two things follow, and both are measurements rather
than proposals:

1. **`d_terms_slack_bytes` is 123.7 MB of pure `std::vector` capacity overshoot** — 4.27 B/term,
   now *larger than the entire inverted index minus its delta payload*. Round 1 already showed
   that taking growth off three separate index paths was worth 3 B/term; this is the same defect
   in the row store, and it is the single cheapest remaining win in the operator.
2. **`init_operator_bytes` at 39.58 B/term is the largest structure and has never been looked at.**
   It is 6.6× the inverted index.

Neither is inverted-index work, and neither was in scope for this round. They are where a third
round should start.

---

# Round 3: the index was finished, so the round went next door

2026-08-14. Scripts under `$PROJ/runs/invidx-r3/`; jobs `1823006` (V0), `1823018` (ledger at 1×16),
`1823032` (round 3a), `1823141` (round-2 arm), `1823254` (3a timing), `1823288` (round 3b + timing).

Round 3 was asked to re-scan the inverted index for a simpler, smaller, faster integration, and to
change the algorithm if that is what it took. **The answer is that the index is finished and the
algorithm should not change.** What the scan found instead is that the operator keeps *four*
representations of its monomials, that the index is the smallest and best-engineered of them, and
that a large part of the operator was capacity no container had earned.

## The instrument came first, and it corrected the plan twice

Two fields were added before anything was optimised, and both immediately mattered.

- **`d_init_operator_entries`** — live entries behind `init_operator_bytes`, which measures
  `bucket_count()`, not `size()`.
- **`matched_scratch_bytes`** — the follower-marking epoch array, one stamp per term, owned by the
  propagator rather than the operator. **No telemetry field had ever counted it**, so every
  `total_bytes` figure earlier in this document under-reports by that amount — 7.58 B/term at the
  Heisenberg point, i.e. round 2's headline 90.38 is **97.96 all-in**. It is counted from round 3 on.
  Subtract it when comparing against a build that predates the field; the base arm has no such field,
  which is why the A/B harness's `total_b_per_term` column understates every round-3 ratio below.

**First correction.** `init_operator_bytes` came back at **0.0 B/term on Hubbard and kicked Ising**.
Both carry a single-site observable, so `init_op_map` never grows and there is nothing to release.
The 39.58 B/term in the tables above is the *random-Heisenberg* workload, whose observable is 7M terms
against 29M propagated ones. **That number tracks the observable, not the operator.** Round 3's plan
had quoted it as a property of the operator; it is not.

**Second correction.** The same run put `matched_scratch_bytes` at 6.0–7.6 B/term — ~10% of the
operator, larger than the whole inverted index, and a target that was not in the plan because nothing
had ever measured it.

Where `init_op_map` *is* large, the diagnosis was exact: at the 7M-observable point it measured
**1,148,190,448 bytes with `d_init_operator_entries` = 0** — a fully drained map holding its entire
bucket array, because `erase` never shrinks `bucket_count()`. It is now **896 bytes**.

## What shipped

| | change | mechanism |
| --- | --- | --- |
| **R1** | release `init_op_map` once drained | swap-with-empty instead of leaving `clear()`'s buckets behind; the drain also lost a `std::vector<Monomial>` of keys that was 448 MB of transient peak across 16 partitions |
| **R3** | `MatchedEpochSet::Stamp` u32 → u16, plus a gated shrink | one O(n) fill per 65535 gates instead of per 2^32 |
| **S** | delete `monoprop_INVIDX_BITMAP_M_STAR` | its own measured frontier killed it: 1.7× the index for 4% of a fold that is a minority of `propagate` |
| ~~R2~~ | ~~shrink `OperatorIndex::rows_` per propagate~~ | **measured and reverted — see below** |

The three that shipped hang off one invariant, now stated operator-wide in `AGENTS.md`: **no container
holds capacity it has not earned.** That started as the inverted-index arena's rule in round 1, and it
turned out to be worth more outside the index than in it.

## R2 was measured and reverted: 3.0% of `propagate` for 8% of the operator

`OperatorIndex::rows_` is the only container in the operator with no shrink path, and it carries
4.1–6.0 B/term of 1.5×-growth overshoot. Shrinking it at the existing quiescent point
(`initialize_operator_caches_`, beside the `op_coeffs` and state shrinks) collects all of it — and
costs real time, because **that function runs after every `propagate()`, not once at the end of a run.**

| | Hubbard c10 (29 `propagate` calls) | kicked Ising c10 (1 call) |
| --- | ---: | ---: |
| round 3a — with the row shrink | `propagate` **1.030×**, 6/6, p=0.031 | 1.008×, 4/6 — unresolved |
| round 3b — with it removed | `propagate` **0.999×**, 4/6 — unresolved | 1.011×, 4/6 — unresolved |

The Hubbard/Pauli contrast is the attribution: the workload that calls `propagate` 29 times pays,
the one that calls it once does not. Removing that single call returns `propagate` to flat, and no
other operation moves in either arm (the one resolved row anywhere is Pauli `build_graph` at 0.996×,
6/6 — 0.4%, and in the good direction).

Geometric growth plus shrink-to-fit churns whenever both run repeatedly: each shrink reallocs the
whole 293 MB row array, and the next layer's growth reallocs it straight back. **A better-timed
shrink does not fix this** — any rule that leaves low slack *at rest* has to shrink near the end, and
"the end" is not observable to the library. Collecting this 4.1–6.0 B/term needs a row store whose
growth does not copy at all (a segmented one, chunk table plus fixed-size blocks), which is a hot-path
change to `row_eq_key`/`for_each_position` and wants its own round.

`shrink_rows_to_fit()` is kept, tested, and left uncalled, for a caller that knows the operator is final.

## Memory — 1×16, all-in, deterministic to the byte

| workload | terms | round 2 | **round 3** | ratio | *(3a, rejected)* |
| --- | ---: | ---: | ---: | ---: | ---: |
| random Heisenberg, 7M observable | 29,007,110 | 97.96 | **52.78** | **1.86×** | *48.51 → 2.02×* |
| Hubbard c10 | 26,607,878 | 55.58 | **51.58** | 1.08× | *47.45 → 1.17×* |
| kicked Ising c10 | 2,655,112 | 61.25 | **57.13** | 1.07× | *51.11 → 1.20×* |

The rejected column is what R2 would have added, and is the size of the prize a segmented row store
would collect at no time cost.

Component detail, random Heisenberg (B/term):

| component | round 2 | round 3 | |
| --- | ---: | ---: | --- |
| `init_operator_bytes` | 39.583 | **0.000** | R1 |
| `indexing_bytes` (dedup hash) | 18.508 | 18.508 | untouched — the next target |
| `operator_terms_bytes` | 18.265 | 18.265 | R2 reverted; 4.265 of this is slack |
| `op_coeffs_bytes` | 8.000 | 8.000 | the only actual payload |
| `matched_scratch_bytes` | 7.578 | **1.985** | R3 |
| `inverted_index_bytes` | 6.020 | 6.020 | unchanged by design |
| **total** | **97.955** | **52.779** | **1.856×** |

Against the original base (`97c1fb6`: 130.58 published + ~7.58 then-uncounted scratch ≈ **138.2**
all-in), the three rounds together are **≈2.62×** on the operator at this workload, and **7.68×** on
the index.

The ranking has inverted again. The dedup hash is now the largest structure at 18.51 B/term (35% of
the operator) and sits at **43.2% occupancy** — `bit_ceil` on a 0.7 load threshold, i.e. the same
unearned-capacity defect, in the one structure whose probe is the hottest random access in `propagate`.

## Why the index itself was left alone

- **Memory.** 6.02 B/term, three quarters of it a delta stream at 1.12× gap entropy. Deleting the
  index outright would save 6.7% of the operator; releasing one empty hash map saved 43.8%.
- **Time.** `M_STAR` had already measured that 1.7× the index buys 4% of the fold under contention,
  and the fold is a minority of `propagate` — so the decode is a minority of a minority. Round 2
  confirmed it twice more: T1 (0.88× kernel, 8/8) and T4 (0.902× kernel, 6/6) both moved nothing
  end-to-end.

### Four restructurings were evaluated and rejected, each on a number

| idea | why it dies |
| --- | --- |
| **Delete the persistent index; fuse a blocked transpose into each fold** | The index is built incrementally (`append_rows`) and folded **L ≈ 5,420 times** per `propagate` on kicked Ising, ~10⁴ on Hubbard — one `build_layer` per generator. ~10⁴ folds amortise one build; re-transposing per fold is four orders of magnitude the wrong way. |
| **Gray-code incremental folds**, `fold(gᵢ₊₁) = fold(gᵢ) ⊕ fold(gᵢ Δ gᵢ₊₁)` | Needs more than half the support shared. \|G\| ∈ {1,2,4} and consecutive layers share **0–1** columns, so the delta is usually larger than the target. It also needs a full n/64-word running mask (3.3 MB at 26.6M terms), destroying the deliberate 8 KB L1-resident blocking. |
| **Cross-generator CSE at replay** — 500 columns, ~13,550 decodes per pass, so each column is decoded ~27× | The masks genuinely are independent (the index is frozen at replay), but application is strictly sequential: sine drains and cross-rank exchange run between layers, so the loop nest cannot be inverted. The legal residue saves at most one column decode per gate. |
| **Prune columns no generator names** | Measured against the real models it prunes nothing: kicked Ising `X_q → {2q}` and `ZZ → {2q, 2q+1}` covers all 254 columns; Hubbard hopping plus on-site interaction covers all 4 indices of every site. |

Unifying the row store with the index is rejected too. They are exact transposes information-
theoretically, but `row_eq_key` — the dedup confirm, on the hottest path in `propagate` — needs O(1)
*row* access, which a column-major structure cannot serve without touching all 500 columns. At 14.00
B/term live the row store is already at the floor for a structure that must answer row queries.

## Where a round 4 should start

1. **The dedup hash, 18.51 B/term at 43.2% occupancy.** Two independent levers: leave power-of-2
   sizing (multiply-shift range reduction, ×1.5 growth) for ~4.7 B/term, and narrow the slot from 8 B
   `{TermIndex, uint32 hash}` to 5 B `{idx[], uint8 fingerprint[]}` for ~7 B/term. Both are also
   *time* levers on the hottest random access in `propagate`, so this is the one place where memory
   and the residual fold penalty pull the same way.
2. **A segmented `rows_`**, to collect the 4.1–6.0 B/term R2 had to give back.

# Round 4a — the occupancy lever, measured and rejected

Round 4 started at item 1 above, with its first lever: size `OperatorIndex::Table` exactly for its 0.7
load factor instead of rounding up with `bit_ceil`, range-reduce with a multiply-shift (Lemire) in
place of the AND mask, and grow by 1.5× instead of doubling. The stored 32-bit hash was left alone on
purpose, so a rehash stays a pure memory shuffle and never re-materialises a row — that is what makes
this lever separable from narrowing the slot.

Base arm `ba79e77` (round 3) in `src/mp-r3`; port arm the same tree with the change. 1×16, pinned,
`--bench-rounds=1`, 6 paired reps, arm order flipped per rep.

## It delivered the memory, on three workloads out of four

| workload | terms | index base → port | occupancy | `total_bytes` base → port | |
| --- | ---: | --- | --- | --- | ---: |
| Hubbard c10 | 26,607,878 | 20.18 → **16.32** | 0.396 → 0.490 | 51.58 → **47.73** | 1.081× |
| kicked Ising c10 | 2,655,112 | 12.64 → **14.36** | 0.633 → 0.557 | 57.13 → **58.85** | **0.971×** |
| kicked Ising c8 | 223,372 | 18.79 → **14.99** | 0.426 → 0.534 | 56.54 → **52.75** | 1.072× |
| random Heisenberg, 7M obs | 29,007,110 | 18.51 → **14.91** | 0.432 → 0.537 | 52.78 → **49.18** | 1.073× |

The kicked Ising c10 row is not a bug and it is the useful half of the table. `bit_ceil` leaves the
realised load **uniform in [0.35, 0.7]**, and on that one workload it drew a lucky 0.633; exact sizing
with 1.5× growth bounds the load to **[0.467, 0.7]**, mean 0.583. Every port occupancy above falls
inside that band, so the mechanism did exactly what it claims — it just cannot beat a good draw from a
wider distribution. The change is a *distributional* improvement (worst case 0.35 → 0.467, mean
0.525 → 0.583), not a uniform one, and one sample landing the wrong way is the honest price.

## And it cost `propagate`, on both real models, resolved

| model | metric | base | port | ratio | agree | p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hubbard c10 | `propagate` | 22967.52 ms | 23306.20 ms | **1.015×** | **6/6** | **0.031** |
| Hubbard c10 | `build_graph` | 8.90 ms | 9.00 ms | 1.017× | 4/6 | 0.69 |
| kicked Ising c10 | `propagate` | 1037.04 ms | 1055.44 ms | **1.019×** | **6/6** | **0.031** |
| kicked Ising c10 | `build_graph` | 852.44 ms | 872.35 ms | **1.023×** | **6/6** | **0.031** |
| kicked Ising c10 | `energy` | 125.12 ms | 125.14 ms | 0.999× | 3/6 | 1.00 |

Rep spread was 1.01–1.05×, so this is signal, not noise, and it is the same standard that rejected R2
(1.030×, 6/6).

## Why — and why this does not condemn the other lever

The cost is **not** the multiply-shift; it is the density that the memory win is made of. Linear-probe
chain length goes as (1 + 1/(1−α))/2 for a hit and (1 + 1/(1−α)²)/2 for a miss, so moving Hubbard's
load from 0.396 to 0.490 buys 7% of the operator at the price of **11% more probe steps on hits and
29% more on misses**. Occupancy and probe length are the same variable read two ways.

So the round-3 note above — "the one place where memory and the residual fold penalty pull the same
way" — is **wrong for this lever specifically**, and the correction is worth stating plainly: on the
occupancy lever memory and time are strictly opposed, and no implementation of range reduction can
change that.

It remains right for the *other* lever. Narrowing the slot from 8 B to 5 B cuts bytes at **constant α**
— probe chains stay exactly as long, and a 64-byte line holds 12.8 slots instead of 8 — so it is a
different trade in kind, and it is the bigger half (~7 B/term against 3.6). Its own cost is that an
8-bit fingerprint cannot survive a rehash: the full hash has to be recomputed from the row, which is
~2n row materialisations across a build under doubling. Iterate the old slots with the same
group-prefetch pipeline `find_batch` already uses, rather than reading rows in slot order and eating a
DRAM miss per entry.

## What was kept

The change itself was reverted. Kept: `index_slot_count()` / `index_entry_count()`, because slot count
is what `indexing_bytes` is really measuring and occupancy is the number this round turned on; and
`probing_is_correct_across_rehashes`, which grows a store through ~8 doublings and pins `find`,
`find_batch` and insertion against each other at every table size on the way. A probe or wraparound
bug strands keys near the top of the table silently, and the suite had no case large enough to catch
one. The rejection is recorded on `Table` itself so the next reader finds it before rebuilding it.

# Round 4b — the slot-width lever: all of the memory, and it still cost time

R4a's closing note said to reach for the other lever instead, and it was right about the memory to the
byte. The slot goes from 8 B `{TermIndex idx, uint32_t hash}` to 5 B — a **1-byte fingerprint** in
place of the hash — with power-of-2 sizing, the 0.7 ceiling and the doubling growth all untouched, so
the load factor is unchanged by construction. R4b implemented it as **two parallel arrays**,
`uint8_t fp[]` and `TermIndex idx[]`, on the Swiss-table argument that a probe should walk only the
byte array: 64 slots to a cache line instead of 8.

Same harness as R4a: base `ba79e77` in `src/mp-r3`, 1×16 pinned, `--bench-rounds=1`, 6 paired reps,
order flipped per rep. Job `1823924`. `ctest -L serial` 212/212 and the 609-case Python suite green on
the port arm, including a new `rehash_preserves_overflow_rows` — the rehash now re-reads rows, so
overflow rows had to be pinned across one.

## The memory came in at exactly 8/5, on every workload

| workload | terms | index base → port | ratio | `total_bytes` base → port | |
| --- | ---: | --- | ---: | --- | ---: |
| Hubbard c10 | 26,607,878 | 20.18 → **12.61** | 1.600× | 51.58 → **44.02** | 1.172× |
| kicked Ising c10 | 2,655,112 | 12.64 → **7.90** | 1.600× | 57.13 → **52.39** | 1.090× |
| kicked Ising c8 | 223,372 | 18.79 → **11.75** | 1.599× | 56.54 → **49.50** | 1.142× |
| random Heisenberg, 7M obs | 29,007,110 | 18.51 → **11.57** | 1.600× | 52.78 → **45.84** | 1.151× |

The ratio is 8/5 to three digits on all four points, which is the check that matters: it says the slot
count did not move at all and the entire delta is the width. Contrast R4a, where the win varied
workload to workload — and went backwards on one — because it was made of occupancy. This lever has no
distributional component to argue about. The prediction stated before the run was 18.51 → ~11.6 on the
Majorana point; measured 11.57.

## And it cost time anyway

| model | metric | base | port | ratio | agree | p | spread |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hubbard c10 | `propagate` | 22956.84 ms | 23793.23 ms | **1.036×** | **6/6** | **0.031** | 1.00× |
| Hubbard c10 | `build_graph` | 8.92 ms | 9.02 ms | 1.002× | 3/6 | 1.00 | 10.5× |
| kicked Ising c10 | `propagate` | 1041.12 ms | 1070.30 ms | 1.030× | 5/6 | 0.22 | 1.10× |
| kicked Ising c10 | `build_graph` | 854.90 ms | 886.81 ms | **1.036×** | **6/6** | **0.031** | 1.01× |
| kicked Ising c10 | `energy` | 125.42 ms | 125.56 ms | 1.004× | 3/6 | 1.00 | 1.02× |
| kicked Ising c10 | `gradient` | 300.13 ms | 299.78 ms | 0.998× | 4/6 | 0.69 | 1.03× |

Hubbard's `propagate` spread is **1.00×** across six reps — the tightest signal in this file. This is
worse than R4a's 1.015×/1.019×, on a lever that was supposed to be free.

## The diagnosis is the split, not the narrowing

The Swiss-table argument needs a long probe chain to amortise, and this table has none. It is sized to
≤0.7 load and realises **α ∈ [0.35, 0.7]**, where the expected hit chain is 1.3–2 slots: a probe
essentially never reads a second fingerprint byte, so the wide scan has nothing to scan. What the split
does deliver, on every single lookup, is a **second cache line** — `fp[s]` and `idx[s]` are in
different allocations — where the 8-byte slot answered both the probe and the confirm from one line.
`find_batch` was given a second prefetch to cover it, which hides the latency but doubles the table's
line traffic, and the table is far larger than LLC on both models. Trading one line for two on the
operator's hottest random-access structure is the 3.6%.

Worth recording that R4a's own forward note had the right arithmetic and R4b's implementation walked
away from it: *"a 64-byte line holds 12.8 slots instead of 8"* is true only if the slot is **one
record**. The moment the fields were split into parallel arrays that sentence stopped describing the
code, and nothing in the change flagged it.

So this is not "narrowing the slot costs time". It is "splitting the slot costs time", and the two were
conflated in the same change. Which is exactly the experiment round 4c runs: same 5 bytes, interleaved.

# Round 4c — the same 5 bytes as one record: a third of the penalty, not all of it

One record per slot, `[fingerprint][TermIndex]`, in a single byte array; the index is read and written
by `memcpy` rather than through a packed struct member, because GCC lowers packed-member access to
byte-wise reads on aarch64 and this project builds for both partitions. On x86-64 GCC 14.3 both forms
emit a single `movl`, so the portable form costs nothing here. Job `1824017`; 213/213 ctest (the extra
case pins the table at exactly `slots × slot width`, no allocator slack) and 609 pytest green.

Memory is **identical to R4b to the milligram** — same slot count, same width, so the table below is
R4b's table again: 20.18 → 12.61 on Hubbard, 18.51 → 11.57 on the Majorana point, 1.600× on all four
workloads. Interleaving is a pure layout change and the ledger proves it.

| model | metric | base | R4b | R4c | agree (R4c) | p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hubbard c10 | `propagate` | 22959.67 ms | 1.036× | **1.024×** | **6/6** | **0.031** |
| Hubbard c10 | `build_graph` | 9.02 ms | 1.002× | 1.012× | 4/6 | 0.69 |
| kicked Ising c10 | `propagate` | 1047.03 ms | 1.030× | 1.017× | 5/6 | 0.22 |
| kicked Ising c10 | `build_graph` | 855.59 ms | 1.036× | **1.027×** | **6/6** | **0.031** |
| kicked Ising c10 | `energy` | 125.65 ms | 1.004× | 0.996× | 4/6 | 0.69 |

The diagnosis was **right, and partial**. Interleaving removed about a third of the penalty on both
models — Hubbard `propagate` 1.036× → 1.024×, kicked Ising `build_graph` 1.036× → 1.027× — which is
the second cache line being paid back, and confirms the R4b post-mortem. But the remaining ~2.4% is
still resolved at 6/6, p=0.031, so it fails the same gate that rejected R2 (1.030×) and R4a (1.015×),
and R4c is reverted on the same rule.

Rejecting it on the rule is not the same as understanding it, and the residual is now the whole
question: **1.024× stands between this project and 7.6 B/term, 13–15% of the whole operator.** Both
5-byte arms carry exactly one cost the 8-byte base does not — an 8-bit fingerprint cannot survive a
rehash, so every doubling re-reads each live row and recomputes `fold_hash`, ~2n row materialisations
across a build. An order-of-magnitude estimate puts that in the right range to be the entire residual,
which is precisely the kind of reasoning this file has been wrong about before, so it is being
measured rather than assumed: job `1824068` runs a **null control** — R4c with the slot widened to 9 B
`[fp][idx][uint32 hash]`, wider than base and therefore unshippable, identical to R4c on the find path,
with the rehash carrying the stored hash the way base does. One mechanism swapped, nothing else.

- Flat against base ⇒ the residual is the recompute, and a **6-byte slot** — 8 prefilter bits plus 8
  spare hash bits, which buys 8 doublings between recomputes because a doubling consumes exactly one
  hash bit — recovers most of R4c's memory at base's rehash cost.
- Still ~1.02× ⇒ the residual is the fingerprint find path itself and the slot-width lever is closed.

## The null control came back flat: it is the recompute

| model | metric | R4c | null control | agree | p |
| --- | --- | ---: | ---: | ---: | ---: |
| Hubbard c10 | `propagate` | 1.024× | **1.004×** | 6/6 | 0.031 |
| Hubbard c10 | `build_graph` | 1.012× | 0.994× | 3/6 | 1.00 |
| kicked Ising c10 | `propagate` | 1.017× | 0.998× | 3/6 | 1.00 |
| kicked Ising c10 | `build_graph` | 1.027× | 1.003× | 6/6 | 0.031 |

Swapping one mechanism — and nothing else — takes 2.4% to 0.4%. **Four fifths of the slot-narrowing
penalty is the rehash's row re-read**, and the 5-byte find path (fingerprint prefilter, one-line
probe+confirm, 1/255 false positives) is very nearly free. 213/213 and 609 green on this arm too, so
the rehash rewrite it required is also correct.

This also kills the 6-byte slot idea before it was built, which is what a null control is for. Spare
hash bits would let the table skip 7 of every 8 recomputes, but total recompute work under doubling is
≈2n and is dominated by whichever recompute lands last — skipping intermediate ones takes 2n to ≈1n, a
**2× reduction, not 8×**. Half the penalty for a byte per slot is a bad trade.

## Round 4d — the cost is the order, not the work

The recompute visits live entries in **old-slot order**, which scatters the row reads at random across
a `rows_` array far larger than L3: one DRAM stall per entry, which the group prefetch only partly
hides. But term indices are dense in [0, `size_`), so the identical set of rows can be visited in
**index order** instead, and the read side collapses into one sequential stream. The live set is
carried by a bitmap — one bit per term, 3.3 MB at 26.6M terms, cache-resident — filled by a sequential
walk of the old table. The random writes into the new table remain, and those are what the 8-byte
baseline already pays.

This keeps the whole 5-byte slot, so the prize is R4c's undiminished: index 20.18 → 12.61 B/term on
Hubbard, 18.51 → 11.57 on the Majorana point, `total_bytes` 51.58 → 44.02. Job `1824165`.

**And it came back worse where it mattered**: Hubbard `propagate` **1.037×**, against R4c's 1.024×,
both 6/6. kicked Ising moved the other way by less (`build_graph` 1.027× → 1.020×, `propagate` 1.017×
→ 1.015×), which is the same story at a tenth the size.

Old-slot order was not arbitrary, and that is what the hypothesis missed. Under doubling a slot's new
home is its old home with at most one bit set above the old mask, so **walking old slots in order
writes the new table in two interleaved ascending runs — nearly sequentially.** R4d bought sequential
reads by spending write locality it had not noticed it had, and at Hubbard's size (a 335 MB table
against a 400 MB row array) the scattered writes cost more than the scattered reads saved.

## Round 4e — overlap both sides, and the floor comes into view

Keep the bitmap and the sequential row scan, then treat the writes exactly as the old code treated the
reads: compute each hash a group ahead and `__builtin_prefetch(..., 1, 0)` the destination line, so the
scattered writes are overlapped 16-deep instead of taken one stall at a time. Job `1824252`.

| arm | Hubbard `propagate` | kicked Ising `propagate` | kicked Ising `build_graph` |
| --- | ---: | ---: | ---: |
| R4b — split 5 B | 1.036× | 1.030× | 1.036× |
| R4c — interleaved 5 B | 1.024× | 1.017× | 1.027× |
| R4d — index order | 1.037× | 1.015× | 1.020× |
| **R4e — index order + write prefetch** | **1.017×** | **1.012×** | **1.018×** |
| null control — hash stored, no recompute | 1.004× | 0.998× | 1.003× |

All 6/6 at p=0.031 except the null control's kicked Ising `propagate` (3/6, no signal). 214/214 ctest
and 609 pytest on every arm.

The series is now closed as an optimisation problem: **3.6% → 2.4% → 1.7%, against a 0.4% floor.** What
remains is not memory latency — both sides are overlapped — but arithmetic. The loop materialises each
row into a `Monomial` and runs SplitmixHash across all its words, ~53M times over a Hubbard build at
roughly 22 cycles each, which is ≈0.39 s and accounts for essentially the whole gap to base. There is
no more stalling left to hide; the only way down is to stop doing the work.

The one lever left for that is a **6-byte slot** carrying 8 spare hash bits, so a doubling consumes one
bit and only every 8th rehash recomputes. Total recompute work goes 2n → ~1n — a 2× cut, not 8×,
because the cost is dominated by whichever recompute lands last — so it projects to ~1.0–1.2%, bought
by giving back a byte per slot: the memory win drops from 7.57 to ~5.05 B/term, 14.7% of the operator
to ~10%. It trades the thing this round was asked to maximise for the thing it was asked to hold
constant, and once the budget widened to 3% it was dominated outright and never built.

# Rounds 4f/4g — the two levers do not stack, and the reason is the growth factor

With the acceptance budget widened to <3%, the occupancy lever came back: it is **orthogonal** to slot
width — one sets slots per term, the other bytes per slot — so stacking them is arithmetic, not luck.

| | slots/term | B/slot | Hubbard index | operator saved |
| --- | ---: | ---: | ---: | ---: |
| base | 2.522 | 8 | 20.18 | — |
| R4e — width only | 2.522 | 5 | 12.61 | 14.7% |
| **R4f — both, 1.5× growth** | **2.041** | **5** | **9.12** | **21.4%** |
| R4g — both, 2× growth | 2.522 | 5 | 12.61 | 14.7% |

R4f's memory beat its own projection: index **2.21× smaller** on Hubbard, 2.24× on kicked Ising c8,
1.99× on the 29M-term Majorana point, and `total_bytes` 51.58 → **40.52**. On Hubbard it also met the
budget — `propagate` **1.024×** for 21.4% of the operator, exactly the trade that was asked for.

It fails on the other model: kicked Ising c10 `propagate` **1.056×** and `build_graph` **1.072×**,
both 6/6. That row diagnoses itself — its load moved 0.633 → **0.623**, i.e. the occupancy lever did
nothing there, because `bit_ceil` had already dealt it a good draw. All of the cost, none of the win,
on the same workload that went backwards in R4a for the same reason.

## The cost is the growth factor, and the win is also the growth factor

The excess is bigger than the mechanism predicts: 1.5× growth takes rehash work 2n → 3n, which should
scale R4e's 1.8 points to ~2.7, not 7.2. The rest is allocation churn — ~30 resizes instead of ~18,
each freeing and reallocating a multi-megabyte buffer on top of re-materialising every live row.

So R4g kept exact sizing and multiply-shift and dropped growth back to 2×, on the reasoning that the
table is sized once from a projection (`MonomialPropagator.inl:170`) and that exact sizing *at that
point* was where the win came from. **That reasoning was wrong, and the measurement is unambiguous:**

| workload | load base → R4g |
| --- | --- |
| Hubbard c10 | 0.396 → **0.396** |
| kicked Ising c10 | 0.633 → **0.633** |
| kicked Ising c8 | 0.426 → **0.426** |
| random Heisenberg | 0.432 → **0.363** |

Identical on three, worse on the fourth. Once growth doubles, exact sizing lands in the same
[0.35, 0.7] band `bit_ceil` does and can land below it. **The occupancy win lives entirely in the
growth factor**, which is precisely the part that is unaffordable — the sizing arithmetic contributes
nothing on its own. The two levers therefore do not stack at any price the budget allows.

# What shipped from round 4

R4e: the 5-byte interleaved slot with the index-order, both-sides-prefetched rehash. It is the only
arm that meets the <3% rule on every workload measured.

| workload | terms | index B/term | `total_bytes` | operator saved |
| --- | ---: | --- | --- | ---: |
| Hubbard c10 | 26,607,878 | 20.18 → **12.61** | 51.58 → 44.02 | **14.7%** |
| random Heisenberg, 7M obs | 29,007,110 | 18.51 → **11.57** | 52.78 → 45.84 | 13.2% |
| kicked Ising c8 | 223,372 | 18.79 → **11.75** | 56.54 → 49.50 | 12.5% |
| kicked Ising c10 | 2,655,112 | 12.64 → **7.90** | 57.13 → 52.39 | 8.3% |

`propagate` 1.017× on Hubbard and 1.012× on kicked Ising; `build_graph` 1.018×; `energy` and `gradient`
flat. 214/214 ctest and 609 pytest.

Rejected and recorded, each with a number: R4a and R4f/R4g (occupancy, at any growth factor), R4b (the
SoA split), R4d (index order without write prefetch), and the 6-byte spare-hash-bits slot (dominated
before it was built). The remaining frontier for a round 5 is **not** this table — it is `rows_` slack
at 4.1–6.0 B/term, which needs a row store whose growth does not copy, and which R2 could not collect
with a better-timed `shrink_to_fit`.

Rebuilding in index order changes insertion order into the new table. That is legal here — the
determinism contract is bit-identity at fixed (R, S), which no table ordering participates in, and
nothing serializes or iterates this table for output — but "legal" is not "tested", so
`rehash_rebuilds_exactly_the_indexed_subset` pins it on a store where the indexed rows are a strict
scattered subset of the rows present (every third row indexed, ~9 doublings). A bitmap that drops a
live index silently loses a key and one that invents an index fabricates an entry; neither crashes,
both only show up as a wrong `find()`.

## Round 5 — the row-store slack was mostly virtual

Round 4 closed the dedup table and named `rows_` slack (4.1-6.0 B/term) as the next frontier, needing
"a row store whose growth does not copy". That store was built and measured. It works exactly as
designed, and the prize turned out to be much smaller than this document has been claiming — because
every B/term figure in it is **capacity-based, and capacity is not residency**.

### The change

`rows_` (a flat `DefaultInitVector<PosT>` grown 1.5x) became `std::vector<std::unique_ptr<PosT[]>>`:
fixed `kChunkRows`-row blocks, allocated with `make_unique_for_overwrite` so the non-zeroing property
of `default_init_allocator` survives, addressed by `chunks_[i >> kChunkShift].get() + ((i & kChunkMask)
* stride_)`. Chunk geometry is counted in rows, not bytes, because `stride_` is a runtime value.
Nothing outside the class could observe contiguity: `rows_` is private, has no `friend`s, and the class
exposes no `data()`, span, reference or iterator — all ~20 call sites go through `row()` /
`for_each_position()` / `popcount()` / `set()` / `find*()`, by index, returning values.

`shrink_rows_to_fit()` and `kShrinkSlackDenom` were **deleted**. They existed to hand back geometric
overshoot; with chunked growth capacity exceeds live by less than one chunk at all times, so there is
nothing to hand back. `grow_rows_geometric` became `grow_rows`.

### Capacity says one thing, RSS says another

`d_terms_slack_bytes` is `capacity - size`. For `rows_` those pages are **never faulted in**: growth
memcpys only the live prefix into the new buffer, and `DefaultInitVector` exists precisely so `resize`
does not touch the tail. At ~18 MB per partition the allocation is mmap'd, so untouched is not resident.

Job 1824515, `/usr/bin/time -v` Maximum RSS, same workloads, both arms, no rebuild:

| workload | ledger claims saved | **actual peak RSS saved** |
| --- | ---: | ---: |
| Hubbard c10 | 110 MB (9.4%) | **4.8 MB (0.24%)** |
| kicked Ising c10 | 15.6 MB | **43.7 MB (8.9%)** |
| kicked Ising c8 | 0.8 MB | **7.0 MB (2.45%)** |

Wrong in both directions, for one reason each:

* **Overstates** where the row store is not the peak. Hubbard's peak is 2.0 GB and is set by some other
  phase, so removing a 293 MB growth transient never reaches the high-water mark.
* **Understates** where it is. The real win is the GROWTH TRANSIENT — the old path holds both buffers
  with the live prefix touched in each — and no telemetry field has ever counted it. On kicked Ising
  c10 live rows are 55.8 MB and the measured saving is 43.7 MB, i.e. essentially the whole transient.

This retroactively explains R2 (round 3): a gated `shrink_to_fit` "collected" 4.27 B/term, cost
`propagate` 1.030x, and no real memory improvement was ever visible. There was none to see.

It does **not** touch round 4's dedup win: a hash table's slots are written scattered across the whole
buffer, so `table_.buf.capacity()` genuinely is resident. Same for `init_op_map`'s buckets. It is
`rows_` slack, and only that, that was fictional.

### Time: the cost is the chunk TABLE, not the chunks

Job 1824461, 6 paired reps, order flipped per rep, medians of per-rep ratios:

| workload | `propagate` | `build_graph` |
| --- | --- | --- |
| Hubbard c10 | **1.024x** (6/6, p=0.031) | 1.047x (5/6) |
| kicked Ising c10 | 1.0004x (3/6, no signal) | 1.008x (6/6) |
| kicked Ising c8 | 0.9966x (1/6, no signal) | 1.021x (6/6) |
| random Heisenberg | 0.9605x (2/6, unresolved) | **0.989x (6/6, faster)** |

Heisenberg's `build_graph` is the only fully-resolved improvement, and it is where the row store
actually grows — consistent with deleting ~3x live of memcpy traffic per partition.

Chunk-size sweep, job 1824589, Hubbard `propagate`, 6/6 at every point:

| kChunkRows | Hubbard propagate | total_bytes hub / heis / c8 / c10 |
| ---: | ---: | --- |
| 1024 | 1.0248 | 39.89 / 41.59 / 45.75 / 46.44 |
| 2048 | 1.0228 | 39.89 / 41.59 / 45.74 / 46.49 |
| 4096 | **1.0158** | 39.90 / 41.59 / 48.23 / 46.62 |
| 8192 | 1.0155 | 39.91 / 41.62 / 48.23 / 47.14 |

Two hypotheses died here. Allocation count is **not** the cost: 1024 doubles the number of allocations
against 2048 and is no worse in that direction. And the per-access dependent load is **not** intrinsic:
if it were, chunk size would not matter, and 1024 -> 4096 moves it 0.9 points. What moves is the chunk
TABLE's cache footprint — at Hubbard's 1.66M rows/partition it is 13 KB at 1024, 6.5 KB at 2048, 3.2 KB
at 4096, against a 32 KB L1 shared with everything else `propagate` touches. The curve plateaus once
the table stops mattering, which sets Hubbard's floor at **1.016x**, not zero.

Larger chunks then trade the small workloads' memory back: c8 goes 45.74 -> 48.23 B/term at 4096
(13,961 rows/partition rounds up to 16,384), c10 goes 46.49 -> 47.14 at 8192.

### Scale: the split is workload SHAPE, not size

The first RSS job measured 26.6M / 2.66M / 223k terms, so "the saving is small" and "the saving is
small *here*" were not yet distinguishable. Job 1824735 ran random Heisenberg at both 29M and ~100M
propagated terms, `/usr/bin/time -v`, same arms, no rebuild:

| point | terms | base peak | port peak | saved | ledger claimed |
| --- | ---: | ---: | ---: | ---: | ---: |
| random Heisenberg 29M | 29,007,110 | 6,288,028 KB | 6,278,356 KB | 9.7 MB (**0.15%**) | 123 MB |
| random Heisenberg 100M | 99,441,369 | 19,480,332 KB | 19,422,388 KB | 56.6 MB (**0.30%**) | 423 MB |

Wall clock at the 100M point is 156.58 s vs 156.32 s — flat, and worth stating because the 29M pair
read 61.1 s vs 52.1 s and that was pure noise (n=1, unpaired, construction included). The paired 6-rep
measurement above is the one to quote.

Live row data at 99.4M terms is 1.39 GB (`stride_ = 7` x `uint16_t` at `num_modes = 250`). Removing the
growth transient should therefore have been worth ~1.39 GB if the row store set the high-water mark. It
moved 56.6 MB, 4% of that. **The row store is not the peak on this model at any size**, and growing the
operator 3.4x did not change that — the saving stayed pinned near zero in *both* absolute-fraction terms
and relative to what the transient is worth.

So the five points do not split by size. They split by observable:

| workload | observable | ledger claims | actual peak RSS saved |
| --- | --- | ---: | ---: |
| Hubbard c10 | 60 sites | 110 MB (9.4%) | 4.8 MB (**0.24%**) |
| random Heisenberg 29M | 7M terms | 123 MB | 9.7 MB (**0.15%**) |
| random Heisenberg 100M | 24M terms | 423 MB | 56.6 MB (**0.30%**) |
| kicked Ising c8 | 1 qubit | 0.8 MB | 7.0 MB (**2.45%**) |
| kicked Ising c10 | 1 qubit | 15.6 MB | 43.7 MB (**8.9%**) |

Where the observable is large, some other phase sets the peak and the row store never reaches it. Where
the observable is a single site, the operator *is* the rows and the transient binds.

### On the row-dominated workloads the saving is ERRATIC, not proportional

Both kicked Ising points above are small — 223k and 2.66M terms, 285 MB and 489 MB peak — so "the row
store is the peak because the observable is one site" and "the row store is the peak because the run is
small" were still confounded. Job 1824754 ran c12 and c14 on the same model:

| cutoff | terms | base peak | port peak | saved | vs live rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| c8 | 223,372 | 284,556 KB | 277,584 KB | 7.0 MB (**2.45%**) | 149% |
| c10 | 2,655,112 | 488,876 KB | 445,220 KB | 43.7 MB (**8.9%**) | 78% |
| c12 | 12,688,327 | 1,302,508 KB | 1,276,608 KB | 25.3 MB (**1.99%**) | 9.5% |
| c14 | 30,992,523 | 2,932,240 KB | 2,657,232 KB | 268.6 MB (**9.38%**) | 41% |

(The "vs live rows" column is against `terms x 21.0 B/term`, c10's measured `d_terms_rows_bytes`; the
job truncates its LEDGER lines at 260 chars so the other three rows are not measured directly. It can
exceed 100% because the transient is *two* buffers, not one.)

**The fraction does not converge and it is not monotone in size.** c12 read at first like a decay from
c10 — it is not; c14 is back at 9.4%. What sets the base arm's transient is where the *last* 1.5x growth
step happened to land relative to the final row count: fire a growth just before the end and both
buffers are near-full at the peak, finish comfortably inside the last allocation and there is almost
nothing to save. That phase is arbitrary with respect to anything a user controls.

So the honest statement of what this change buys is **not** a percentage. It removes an allocation spike
whose size is unpredictable and which reached 268 MB / 9.4% on the largest row-dominated point measured,
and it cost memory on none of the seven points.

### 4096 makes the largest win bigger, not smaller

Every RSS figure above was taken at `kChunkRows = 2048` (the sweep's `-U` restore). The ship gate, job
1824757, rebuilt at the 4096 operating point and re-measured the biggest point. The prediction was that
4096's extra quantized slack — one more chunk per partition, 4096 rows x 21 B x 16 = 1.4 MB — would
erode the win slightly. **It did the opposite:**

| kicked Ising c14 | peak RSS | vs base |
| --- | ---: | ---: |
| base (1.5x flat vector) | 2,932,240 KB | — |
| port @ 2048 | 2,657,232 KB | 268.6 MB (9.38%) |
| port @ 4096 | **2,576,992 KB** | **347.0 MB (12.12%)** |

4096 is 78.4 MB better than 2048 on the same workload, ~55x the quantized slack it was supposed to cost.
The quantization argument was right in isolation and swamped by allocator behaviour: at c14 there are
1.94M rows per partition, so 2048 asks glibc for 946 blocks of 43 KB per partition where 4096 asks for
473 of 86 KB — both under the 128 KB mmap threshold, so both come off the heap, and the smaller blocks
fragment it harder. This was not predicted and is not modelled anywhere; it is recorded because it says
the chunk-size choice has a *third* term (allocator packing) beyond the two that were swept (L1 footprint
of the table, quantized slack).

Ledger at 4096, for the record: `total_bytes` 39.896 (Hubbard c10) / 46.623 (kicked Ising c10) / 48.234
(c8) / 41.595 (random Heisenberg), with `d_terms_slack_bytes` 0.015 / 0.252 / 2.951 / 0.020 — matching
the sweep row to three digits.

### Where this leaves it

At the 4096 operating point: Hubbard `propagate` 1.016x, against a memory spike removed that ranges from
nothing (Hubbard, Heisenberg — 0.15-0.30%) to 347 MB (kicked Ising c14, 12.12%). The workload that costs
23 seconds a call is the one that gains nothing resident, and that asymmetry is the reason this was a
judgement call rather than an obvious win. 215/215 ctest and 609 pytest on both arms at every swept size,
and again on the 4096 build that ships.

**Shipped** at `kChunkRows = 4096`: the time cost is inside the accepted budget, the change never costs
memory anywhere measured, and it deletes `shrink_rows_to_fit()` and `kShrinkSlackDenom` along with the
growth-vs-shrink churn that round 3's R2 ran into. The append path is now O(1) worst case with no
realloc, which is a robustness property independent of any of these numbers.

Patch preserved at `runs/invidx-r5/r5-chunked-rows.patch`; harness and all raw output under
`runs/invidx-r5/`.

**The durable result of this round is the measurement, not the container: no B/term claim in this file
should be believed without a peak-RSS figure beside it.**
