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

## Time is flat, and is reported as flat

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

## Occupancy — the number that picks the next fix, and had never been measured

| cell | occupancy | slot bytes a sparse layout could drop |
| --- | ---: | ---: |
| c12, P=16 | 31.5% | 68.5% |
| c12, P=128 | 23.9% | 76.1% |
| c14, P=128 | 27.8% | 72.2% |

**Roughly three quarters of world slots carry no traffic**, and tripling the term count moves
occupancy only 23.9% → 27.8%, so this is structural rather than an artifact of small problems.
Routing is a uniform hash with no locality, so it is not skew — a layer's terms simply do not reach
most of the world.

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
- **Nothing in the harness asserts the two arms are different builds.** `check_provenance` refuses
  on term counts, placement and environment, but only *prints* `monoprop_version`. The default
  `PORT_VENV` points at `mp-invidx`, a different branch — read the `=== main:` / `=== port:` banner
  in the job log before trusting any table here.
- `$PROJ/runs` is shared between sessions. Collate from an explicit directory list; a `models-*`
  glob has swept another campaign's cells into a table before.

## Reproduce

```bash
cd $PROJ/src/mp-gwsbench
PORT_VENV=$PROJ/src/mp-gws/.venv hpc/deucalion/sbatch/campaign.sh geometry
hpc/deucalion/tools/graph_world_report.py $PROJ/runs/models-pauli-gws-c12-g1x16-N1 \
                                          $PROJ/runs/models-pauli-gws-c12-g1x128-N1
```

Use a venv python for the tools: the login node's `python3` is 3.6.8.

Cost: **6 node-hours of 3 000 node-hours (0.2 %).**
