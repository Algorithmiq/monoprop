# The graph is quadratic in the flat world size, not in the rank count

2026-08-15, Deucalion `dev-x86`. Harness `sbatch/campaign.sh geometry`, jobs `1825759`–`1825763`;
the P sweep is job `1825752`. Summaries under `$PROJ/runs/models-pauli-gws-*/AB-SUMMARY.md`.

| | |
| --- | --- |
| arms | `main` @ `6abd839` (0.8.1.dev37+g6abd839e6) vs `perf/graph-world-size` @ `1d62773` (0.8.1.dev38+g1d6277344) |
| problem | `pauli` kicked-Ising, 127 qubits, 20 layers. c12 → **29,385,590 terms**; c14 → **91,273,861 terms** |
| geometry | R and S swept independently: 1x16, 8x2 (P=16); 1x128, 8x16, 2x64, 4x32 (P=128) |
| reps | 6, interleaved, order flipped per (rep, cell), `--bench-rounds=1` |
| identical | `MAX_NUM_MODES=1024`, `MALLOC_ARENA_MAX=PARTITIONS`, `OMP_NUM_THREADS=1`, MPI on, same `benches/` |
| placement | `single_cpu_threads == PARTITIONS` on **both** arms, every cell, under `--cpu-bind=none` |

Term counts agree to the term across all cells of a rung.

## The reviewer's observation was two reporting artifacts and one real defect

The investigation started from a note that "the whole 8-rank job costs 2.3× the memory of the
single-rank one" for the same problem. Two corrections before the real finding:

- **The 2.3× is a sum-vs-max comparison.** `memhwm` is `_reduce_sum` over ranks
  (`benches/conftest.py:507`), so it is a **job total** being read against one process's RSS. Eight
  ranks for 2.3× the total is **0.29× per rank** — sub-linear, which is good. Every `max`-reduced
  metric agrees: `opmempeak.max` is 1.28 GiB at 8 ranks against 4.27 GiB at 1.
- **The paired timing cells were not controlled.** `-serial-N1` is 1×16 and `-c12-N1` is 8×16, so
  cores, MPI rank count and world size all move at once, on two different nodes.

The real defect is underneath, and nothing in this directory had named it.

## Result: the axis is P = ranks × partitions

Inside the engine `rank_count` is `mpi::size(comm)`. On a partitioned run the comm is
`Kind::Hybrid`, whose `size()` returns the **flat world `P = mpi_ranks × partitions`**
(`HybridComm.h:81`), not the MPI rank count. So a per-layer "per-rank" array is `P` long, each
process holds one per partition, and the graph retains one per layer — `O(P²)` across the job.

Sweeping partitions at **one MPI rank** (job `1825752`, c10, 4,751,695 terms):

| P (= 1 rank × S) | slot records | slot-proportional bytes | graph `total_bytes` | occupancy |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1,387,520 | 33,300,480 | 366,649,352 | 27.99% |
| 32 | 5,550,080 | 133,201,920 | 510,098,112 | 25.27% |
| 64 | 22,200,320 | 532,807,680 | 996,798,160 | 21.86% |
| 128 | 88,801,280 | 2,131,230,720 | 2,769,389,680 | 16.62% |

Every doubling of `P` multiplies both by **exactly 4** — 1,387,520 × 4 = 5,550,080 × 4 =
22,200,320 × 4 = 88,801,280. This is not a fit through four points.

### It tracks P, not the MPI rank count — the point that could have refuted it

Every other wave in this directory sweeps ranks at a fixed 16 partitions, so `P` and the rank count
move together and no cell here could tell which one a cost tracks. Holding `P` fixed and moving the
rank count across its whole range:

| P | geometries | graph `total_bytes` |
| ---: | --- | ---: |
| 16 | 1x16, 8x2 | 366,649,352 — **identical** |
| 128 | 1x128, 8x16, 2x64, 4x32 | 2,769,389,680 — **identical** |

Six geometries, two values. A **single** MPI rank at 1x128 pays 7.6× what the same single rank pays
at 1x16. The rank-count explanation predicts that cell is cheap; it is the expensive one. MPI is
exonerated and the cost is in the graph encoding.

### The coefficient is predicted with nothing fitted

`b = L × bytes_per_slot`, where `L` is the layer count — one layer per gate, so
`20 × (127 X + 144 heavy-hex ZZ)` = **5,420**. The instrument recovers exactly that number
independently: 86,720 layer-cores over 16 partitions is 5,420, which also confirms the layer count
does **not** grow with `P` (a per-layer `O(P)` cost would otherwise read as `P²` with nothing
wasted). At 24 B/slot after the fix, `5,420 × 24 × 128² = 2,131,230,720` — the measured row **to the
byte**.

## Before/after: the record was storing the same range twice

`CrossRankPartnerRange` carried an offset *and* a count for each of B and D, which are the two
endpoints of the same rotation set and therefore always equal. Keeping one pair takes the record
from **32 B to 16 B**.

| cell | P | main | port | saved | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| c12 1x16 | 16 | 1.442 GiB | 1.421 GiB | 0.021 GiB | 1.01× |
| c12 8x2 | 16 | 1.442 GiB | 1.421 GiB | 0.021 GiB | 1.01× |
| c12 1x128 | 128 | 4.982 GiB | 3.659 GiB | **1.323 GiB** | **1.36×** |
| c12 8x16 | 128 | 4.982 GiB | 3.659 GiB | **1.323 GiB** | **1.36×** |
| c14 8x16 | 128 | 7.091 GiB | 5.768 GiB | **1.323 GiB** | **1.23×** |

Three things to read off it:

- **Exactly 16.00 B per slot record** — 1.323 GiB over 88,801,280 records.
- **The saving does not move with the problem.** c12 and c14 differ by 3.1× in terms and save the
  *identical* 1.323 GiB. A traffic-proportional cost could not; only a slot-proportional one can.
  The ratio differs only because the larger operator dilutes it.
- **The operator is byte-identical between arms on every cell** (`port/main = 1.0000`), so the
  change is confined to the graph.

Fitting on the c12 pair, where the term count is fixed and only `P` moves:

```
b_main = 235,680 B/P²    b_port = 148,998 B/P²    delta = 86,682 B/P²
predicted delta = L × 16 B = 5,420 × 16 = 86,720 B/P²        agreement: 0.04%
```

**The absolute coefficient is not fully explained** and is not claimed: 130,080 is predicted for the
port arm against 148,998 measured, so ~19,000 B/P² of the graph is P-proportional content this
change does not touch. The *difference* is what the change claims, and that is predicted to 0.04%.

## Phase 1 time is flat, and is reported as flat

| cell | build_graph | propagate | energy | gradient |
| --- | --- | --- | --- | --- |
| c12 1x16 | 1.00× (3/6) | 0.99× (6/6) | 1.00× (3/6) | 1.01× (3/6) |
| c12 8x2 | 1.00× (3/6) | 1.00× (4/6) | 1.00× (4/6) | 1.00× (5/6) |
| c12 1x128 | 1.14× faster (5/6) | 1.01× (3/6) | 0.96× (6/6) | 1.06× slower (6/6) |
| c12 8x16 | 0.98× (5/6) | 1.00× (3/6) | 1.10× slower (5/6) | 1.04× (5/6) |
| c14 8x16 | 1.00× (4/6) | 0.98× (5/6) | 0.95× (5/6) | 0.99× (5/6) |

**No operation resolves as a time change.** At six reps a sign test bottoms out at p=0.031, which
needs 6/6; 5/6 is p=0.109 and does not clear. So the 1.14× on `build_graph` at 1x128 — the largest
and most attractive number in the table — is **not a result at this rep count**.

Two cells do reach 6/6, in opposite directions and both inside the harness's own flat band: `energy`
0.96× and `gradient` 1.06×, at 1x128. The `gradient` figure is the honest negative. The other four
cells put gradient at 0.99–1.04× and none reach 6/6, so it is probably node state — but one cell did
clear the bar in the slower direction and suppressing that would be cherry-picking.

This is the expected shape: the change removes retained bytes, not work from any hot loop.

## Phase 2: deriving the layouts saves 2.26× at P=512, and moves the gradient

The second change stops **retaining** the exchange layouts. `counts[r]` is a function of the slot
records and `displs` is its prefix sum, so both are derived into per-thread scratch at the call
site; the derivative round is the evolution round at scale 2, and scaling commutes with the
transpose, so it needs no collective of its own.

Arms: `main` `0.8.1.dev37+g6abd839e6` against this branch @ `5f33a71`, confirmed distinct by
**`.so` hash** (`f73d17f2…` vs `d37b0f1e…`) rather than by version string — the port venv's stamp
reads `dev38+g1d6277344.d20260815` because the gate built it before the commit existed, which is
exactly the orphaned-stamp trap. 7 cells, 6 reps, up to 4 nodes.

| cell | P | main | port | saved | ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| c12 1x16 | 16 | 1.442 GiB | 1.397 GiB | 0.045 GiB | 1.03× |
| c12 8x2 | 16 | 1.442 GiB | 1.397 GiB | 0.045 GiB | 1.03× |
| c12 1x128 | 128 | 4.982 GiB | 2.888 GiB | 2.093 GiB | 1.72× |
| c12 8x16 | 128 | 4.982 GiB | 2.888 GiB | 2.093 GiB | 1.72× |
| c14 8x16, 1 node | 128 | 7.091 GiB | 4.998 GiB | 2.093 GiB | 1.42× |
| c14 8x16, 2 nodes | 256 | 17.340 GiB | 9.183 GiB | 8.157 GiB | 1.89× |
| **c14 8x16, 4 nodes** | **512** | **57.686 GiB** | **25.494 GiB** | **32.192 GiB** | **2.26×** |

The paired geometries are byte-identical again (1x16 = 8x2, 1x128 = 8x16), so `P` is still the axis.

### The saving is two terms, and predicts every cell to the byte

Job-total slots = `L·P²` and layer-cores = `L·P`, so with `L = 5,420`:

```
saved = 24 B × slots  +  168 B × layer_cores
```

| cell | P | predicted | measured | ratio |
| --- | ---: | ---: | ---: | ---: |
| c12 | 16 | 47,869,440 B | 47,869,440 B | **1.00000** |
| c12 | 128 | 2,247,782,400 B | 2,247,782,400 B | **1.00000** |
| c14 | 128 | 2,247,782,400 B | 2,247,782,400 B | **1.00000** |
| c14 | 256 | 8,758,026,240 B | 8,758,026,240 B | **1.00000** |
| c14 | 512 | 34,565,898,240 B | 34,565,898,240 B | **1.00000** |

Zero free parameters, across a 4× range in `P` and a 3.1× range in term count. The 24 B/slot is the
record fields plus `counts`/`displs`; the 168 B/core is `sizeof(LayerCore)` falling **416 → 248 B**
as two `LayerExchangeLayout` structs and their embedded caches leave it.

> **The second term is why a one-term reading looked inconsistent.** Per slot the saving reads
> 34.50 B at P=16 and 25.31 B at P=128, which invites "the effect is not slot-proportional". It is:
> the *other* term is linear in `P`, so it is a larger share of the total when `P` is small. Divide
> a two-term law by one of its variables and it will look unstable in exactly this way.

### The ledger understates it

The derivative layout was a diagnostic outside `total_bytes()`, so neither it nor its removal can
appear in `graph`. At P=512, uncounted: `recv_cache` **10.59 GiB measured** on the port arm
(`d_recv_cache_bytes` = 11,366,563,840 = exactly 8 B/slot) against ~31.76 GiB computed for `main`
(8 B/slot cache + 16 B/slot derivative layout). That is ~21 GiB more than the table shows.

`main`'s figure is **computed from struct layout, not measured** — `main` has no such binding.
Independent corroboration: `gradient` peak transient memory on the worst rank falls 0.88 → 0.04 GiB
at P=512, and a per-rank derivative layout works out to ~0.66 GiB. Same order, from a metric that
knows nothing about the ledger.

`recv_cache` is now the largest remaining P² term, and it is the one piece that cannot be derived.

### Time — this one does move, and the win grows with P

`*` = 6/6, clearing a sign test at p=0.031. 5/6 is p=0.109 and does not clear.

| cell | P | build_graph | propagate | energy | gradient |
| --- | ---: | --- | --- | --- | --- |
| c12 1x16 | 16 | 1.00× (4/6) | 1.03× faster (5/6) | **1.04× slower** (6/6\*) | 1.01× slower (6/6\*) |
| c12 8x2 | 16 | 1.01× slower (5/6) | 1.00× (4/6) | **1.04× slower** (6/6\*) | 1.02× faster (6/6\*) |
| c12 1x128 | 128 | 1.23× faster (5/6) | 1.11× faster (3/6) | 1.01× faster (5/6) | **2.10× faster** (6/6\*) |
| c12 8x16 | 128 | 1.02× slower (5/6) | 1.01× faster (3/6) | 1.00× (3/6) | **1.15× faster** (6/6\*) |
| c14 N1 | 128 | 1.02× faster (6/6\*) | 1.00× (3/6) | 1.01× slower (4/6) | 1.03× faster (5/6) |
| c14 N2 | 256 | 1.02× faster (5/6) | 1.00× (4/6) | **1.06× faster** (6/6\*) | **1.13× faster** (6/6\*) |
| c14 N4 | 512 | 1.01× faster (5/6) | 1.01× faster (4/6) | **1.26× faster** (6/6\*) | **1.43× faster** (6/6\*) |

`gradient` is the operation this moves, and the win grows with `P`: 1.15× at P=128, 1.13× at P=256,
1.43× at P=512, and **2.10× at 1x128** where 128 partitions sit inside one rank (per-rep ratios
0.466–0.486, the tightest cell in the campaign). The mechanism is that `main` resolved a *separate*
transpose for the derivative round — one extra `alltoall_counts` per layer, 5,420 of them, on the
first gradient. The derivative transpose is now the evolution one scaled by 2.

`build_graph` and `propagate` are flat everywhere. The 1.23× on `build_graph` at 1x128 is the
largest number in the table and is **not a result** — 5/6.

### The honest negative, and one thing that is unresolved

At **P=16 this is a small loss**: `energy` is 1.04× slower on *both* geometries at 6/6. Expected
rather than surprising — deriving counts is a fixed cost paid per exchange while the saving grows
with `P`, so a small world pays without earning.

`gradient` at P=16 is **unresolved, not a regression**: 1.01× slower on 1x16 and 1.02× faster on
8x2, each at 6/6, in opposite directions. Two geometries at the same `P` disagreeing in direction
while both reach 6/6 means the effect is per-geometry, not per-`P`. Neither figure is the P=16
gradient result, and quoting either alone would be picking the one that suits the story.

## Occupancy — measured for the first time, and initially over-read

| cell | occupancy |
| --- | ---: |
| c12, P=16 | 31.5% |
| c12, P=128 | 23.9% |
| c14, P=128 | 27.8% |

**Roughly three quarters of world slots carry no traffic**, and tripling the term count moves
occupancy only 23.9% → 27.8%, so this is structural rather than an artifact of small problems.
Routing is a uniform hash with no locality, so it is not skew — a layer's terms simply do not reach
most of the world.

> **The inference first drawn from this was wrong, and the error is worth keeping.** This table
> originally carried a second column — "slot bytes a sparse layout could drop", 68.5% / 76.1% /
> 72.2% — and on that basis sparsifying `ranges` was reported as the largest remaining lever,
> worth about 3× the record shrink. Low occupancy bounds what sparsifying could reclaim **as a
> fraction of itself**, which says nothing about its size beside the other levers. Ranked in bytes
> per slot, sparsifying `ranges` saves ~12 B — *less* than the 16 B the record shrink had already
> shipped — and it is the one option that changes a structure three consumers index by global
> slot. The retained exchange layouts were the larger prize (24 B/slot, 40 once gradients run) and
> needed no such change. The column is removed because it invites the same mistake; the occupancy
> figures themselves stand.

Rank a lever by its bytes against the other levers, not by the percentage of itself it reclaims.

## The next lever: the transpose cache survives 550M comparisons without a single mismatch

Phase 2 kept `evolution_recv_cache` on the grounds that it is "the one part that cannot be derived
locally — it takes a collective". Reading the sink that builds the partner data says otherwise.

On rank *m*, slot *r* holds `in_entries` (the queries *r* sent *m*) followed by `out_entries` (the
responses *m* got for the queries it sent *r*), and the stored count is `P + Q`
(`Engine.h:145-175`). On rank *r*, slot *m* holds those two swapped. So `count(m→r) == count(r→m)`
by construction, and since displacements are prefix sums of the same array, **the recv layout is
the send layout**. If that holds, `resolve_recv`'s `alltoall_counts` is unnecessary and the cache
is deletable.

Measured with a temporary probe (`sbatch/symcheck.sh` + `sbatch/symcheck-probe.patch`) that
compares the alltoall's answer against the send counts on every resolve, job 1826390:

| stage | world P | resolves | slots compared | mismatched |
| --- | ---: | ---: | ---: | ---: |
| pytest `--with-mpi` | 32 | 280,160 | 5,743,104 | **0** |
| pytest `--with-mpi` | 256 | 2,241,280 | 189,190,144 | **0** |
| pauli c12 energy + gradient | 256 | 1,387,520 | 355,205,120 | **0** |

The counters are the point. A probe that only fires on mismatch reports nothing on a clean run and
nothing when it never ran, and the first attempt at this job did exactly that — see below.

**The one path that could have broken it does not.** Symmetry is a property of the send pattern, so
any rank-local edit to `cross_rank` after the build would end it, and the failure would be a
distributed hang rather than a wrong answer. The only place a `LayerCore` is copied and modified is
`set_parameter_mapping`'s relabel (`MonomialPropagator.inl:815-830`), which writes `param_index` and
nothing else.

Worth, if implemented: **8 B/slot — 10.59 GiB at P=512** (measured, port arm), plus one collective
per layer per first evaluation. It is uncounted by `total_bytes`, so it would not move the headline.

> **Two gates in this job failed for reasons that had nothing to do with the claim**, both worth
> keeping. `strings … | grep -q` under `set -o pipefail` reports *failure* on a match — `-q` exits
> early, `strings` takes SIGPIPE, and the pipeline status is the failure. And the gate was checking
> `build/editable/Release/…/_core.so` while Python imports its own copy under
> `site-packages/monoprop/`: `cmake --build` does not update the imported binary, so a whole job ran
> the previous day's `.so` and reported "no probe files" on all three stages. Ask the interpreter
> which file it loaded (`from monoprop import _core; _core.__file__`) and gate on that one.

Still a hypothesis about *all* patterns, not a proof — the evidence is 550M slot comparisons plus a
construction argument. Implementing it should keep an assertion on the equality in debug builds
rather than delete the check with the cache.

## Reading rules and limits

- **`memhwm` is a sum over ranks; `dmem` and `opmempeak.max` are maxima.** Comparing one against the
  other is what produced the 2.3× that started this. Quote a job total only against another job
  total.
- **`graph_memory_bytes()` deliberately did not change meaning** in the fix: the new diagnostics
  (`d_slot_record_bytes`, `d_recv_cache_bytes`, `d_derivative_layout_bytes`, occupancy) are excluded
  from `total_bytes()`, so an A/B against an older build still compares one quantity.
- **The reported graph sits below process RSS**, and the gap widens with `P`, because the
  `resolve_recv` transpose cache and the retained derivative layout are resident memory
  `total_bytes()` has never counted. The ledger is a **floor**.
- **The per-field split and occupancy exist on the port arm only** — the binding postdates `main`.
  `conftest` records nothing rather than zeros for the baseline, by design: a flat field and an
  absent instrument must not look alike.
- **The harness now refuses an A/B whose arms are the same build** (`_check_versions`), which it did
  not when the cells above were run — it recorded `monoprop_version` and never asserted on it, and
  the default `PORT_VENV` points at `mp-invidx`, a different branch. The check has a known blind
  spot: the version is a git describe of the worktree, not a fingerprint of the extension, and an
  editable install serves whatever `.so` was last built into that tree. Distinct versions prove the
  *checkouts* differ; only the `.so`'s hash proves the *builds* do. Still read the `=== main:` /
  `=== port:` banner.
- `$PROJ/runs` is shared between sessions. Collate from an explicit directory list; a `models-*`
  glob has swept another campaign's cells into a table before.

## Reproduce

```bash
cd $PROJ/src/mp-gwsbench
# Phase 1: the record shrink.       Phase 2: the retained layouts, + a node ladder.
PORT_VENV=$PROJ/src/mp-gws/.venv hpc/deucalion/sbatch/campaign.sh geometry
PORT_VENV=$PROJ/src/mp-gws/.venv hpc/deucalion/sbatch/campaign.sh layout

hpc/deucalion/tools/graph_world_report.py $PROJ/runs/models-pauli-gws2-c12-g1x16-N1 \
                                          $PROJ/runs/models-pauli-gws2-c12-g1x128-N1
```

`PORT_VENV` is **not optional**: its default points at another branch's worktree. `ab_summary` now
refuses an A/B whose two arms report the same build, but that check cannot see a stale `.so` — the
arms here were confirmed by hashing the extension, not by reading `__version__`.

Use a venv python for the tools: the login node's `python3` is 3.6.8.

Cost: **~13 node-hours of 3 000 (0.4 %)** — 6 for phase 1, ~7 for phase 2's seven cells (the
4-node rung is 4 nodes × 21 min).
