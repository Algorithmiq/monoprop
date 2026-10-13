# Paper Fig. 3 — inverse time scaling with qubit number, at three circuit depths

One page-wide row of three panels (paper `fig:inverse-scaling`, a `figure*`):

| | (a) kicked Ising chain (Pauli) | (b) 2D Hubbard lattice (Majorana) |
| --- | --- | --- |
| engines | monoprop, `PauliPropagation.jl` 0.9.0, `ppvm` | monoprop, `MajoranaPropagation.jl` 0.4.1 |
| N | 32 … 1024 qubits | 32 … 1024 modes (3 layers: to 256) |
| depths (small / medium / large) | 7 / 10 / 13 layers | 1 / 2 / 3 layers (one Trotter step each) |
| cutoff | Pauli weight 6 | Majorana length 4 |

- **(a), (b) cost:** nanoseconds for one gate to act on one term, against N. Colour, line style
  and marker are the engine (monoprop blue, solid, filled; the Julia reference vermillion,
  dashed, open; `ppvm` green, dotted, open triangles), and each engine is named in ink beside its
  lines; there is no figure legend. Shade is the depth, light to dark, with a grey key at the
  lower left of each panel; the three depths of one engine lie close together, so only the
  medium depth carries markers. The near-black dashed line, keyed at the lower right, is 1/N
  through monoprop's first medium-depth point. A point over the 30-min limit, or not run because
  a smaller workload already exceeded it, has no cost to plot and is left out; the caption lists
  them. The fitted exponents are in the caption and below, not on the figure.
- **(c) terms a gate touches:** from counts rather than timings, the share of the K terms one
  gate touches, at the medium depth of each model. A dense code tests all of them (the dashed
  100% line); monoprop visits only those that anticommute with the gate, a share that falls
  from 15% (kicked Ising) and 23% (Hubbard) at N = 32 to 0.46% at N = 1024 and 1.6% at N = 512.
  The grey band is what the index skips.

The design went through a busier eight-panel version (one panel per depth, plus K and the
anticommuting count as separate lines); since the depths of one engine nearly coincide, they
are folded into one panel per model, and the two counts into the share touched. Colours are
Okabe-Ito, checked with a CVD validator: the three engine colours pass the colour-blind and
normal-vision separation checks over all three pairs, and each three-shade depth ramp is
monotone in lightness with its light end at ≥ 2:1 against white. Text is ink, never a series
colour. The engine names sit at a fixed side of each bundle (`NAME_AT`), placed from the data.

A gate anticommutes with ~cK/N terms (c ≈ 4.4–4.9 for kicked Ising; 3.8–3.9 after one Hubbard
layer and 7.4–8.0 after two and three), and at fixed depth K grows ∝ N, so monoprop's work per
gate is flat in N while a dense code's grows ∝ N; and a dense code's per-term test also grows
with the machine words a string spans. Hence monoprop's cost per term falls as 1/N and the
references' does not. Where K is small and strings fit one word (kicked Ising at N ≤ 64),
`PauliPropagation.jl` is faster.

## Result

Every number below comes from the rows; `make_figure.py` prints the exponents and ratios and
writes them to `figures/captions.txt`. ns per gate per term:

| N | mono 7 | PP.jl 7 | ppvm 7 | mono 10 | PP.jl 10 | ppvm 10 | mono 13 | PP.jl 13 | ppvm 13 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.82 | **0.72** | 3.20 | 2.19 | **0.898** | 5.42 | 2.47 | **0.933** | 7.17 |
| 64 | 1.00 | **0.856** | 3.86 | 1.21 | **0.902** | 6.00 | 1.32 | **0.948** | 10.4 |
| 128 | **0.52** | 1.25 | 4.43 | **0.607** | 1.36 | 6.13 | **0.635** | 1.44 | 7.26 |
| 256 | **0.275** | 1.98 | 5.71 | **0.315** | 2.19 | 7.51 | **0.329** | 2.32 | > 30 min |
| 512 | **0.140** | 3.40 | 7.59 | 0.161 | > 30 min | > 30 min² | 0.169 | > 30 min² | > 30 min² |
| 1024 | 0.0707 | > 30 min¹ | > 30 min | 0.0807 | > 30 min² | > 30 min² | 0.0879 | > 30 min² | > 30 min² |

| N | mono 1 | MP.jl 1 | mono 2 | MP.jl 2 | mono 3 | MP.jl 3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | **17.0** | 26.9 | **2.61** | 5.48 | **6.02** | 15.6 |
| 64 | **6.90** | 19.3 | **1.01** | 4.19 | **3.03** | 14.8 |
| 128 | **3.10** | 16.6 | **0.522** | 4.18 | **1.26** | 12.9 |
| 256 | **1.54** | 21.3 | **0.245** | 5.52 | 0.427 | > 30 min |
| 512 | **0.876** | 29.9 | 0.118 | > 30 min | — | — |
| 1024 | **0.553** | 54.1 | 0.0579 | > 30 min² | — | — |

¹ One round finished past the limit (2278 s): left out of the figure and the fits. ² Not run: the
engine already exceeded the limit on a shallower circuit at the same N, or at the next smaller N
at the same depth (`implied` rows). — Three Hubbard layers stop at N = 256 (6.4e8 terms); at
N = 512 they would not fit a node.

- **Kicked Ising.** monoprop falls as N^−0.93 / −0.94 / −0.98 at 7 / 10 / 13 layers.
  `PauliPropagation.jl` grows as N^+0.57 / +0.44 / +0.31 and `ppvm` as N^+0.31 / +0.14 / +0.01
  (each fitted over the widest N every engine finished: 512, 256, 128). `PauliPropagation.jl` is
  2.4–2.6× faster than monoprop at N = 32 at every depth and still faster at N = 64; monoprop
  leads from N = 128 (2.2–2.4×) to 24× at N = 512, 7 layers. `ppvm` is 1.8–2.9× slower than
  monoprop at N = 32 and 54× slower at N = 512, 7 layers. Neither dense code finishes N = 1024
  at any depth; monoprop takes 26 s (7 layers, 2.5e7 terms) to 868 s (13 layers, 3.7e8 terms).
- **2D Hubbard.** monoprop falls as N^−0.99 / −1.12 / −1.13 at 1 / 2 / 3 layers;
  `MajoranaPropagation.jl` goes as N^+0.21 / +0.00 / −0.14 (fitted over N ≤ 1024 / 256 / 128).
  monoprop is faster at every point: 1.6–2.6× at N = 32, 23× at N = 256 after 2 layers and 98×
  at N = 1024 after 1. One layer holds only 512–32,384 terms, so its cost per term is mostly
  per-gate overhead (monoprop 100–145 ns per visited term against 8–26 ns at 2–3 layers). At 2
  layers and N = 1024 (2.8e8 terms) monoprop takes 79 s. On the two smallest lattices three
  layers saturate the length-4 cutoff.
- **Work per gate.** The anticommuting terms per gate are 4.43–4.92 K/N for kicked Ising and
  3.75–7.95 K/N for Hubbard, at every counted point; so the ratio K / anticommuting grows ∝ N, to
  216–231× (kicked Ising, N = 1024) and 65–130× (Hubbard, N = 512).
- **Even support** (`occupancy_figure.py`, `figures/fig3_occupancy.*`, paper appendix
  `app:measured-occupancy`). After every layer, from the index list lengths: the longest list
  over the mean, c, and the average list a gate reads over one layer, over the mean. The second
  is what the runtime pays: every position is read by at most r gates of a layer (r = 2 kicked
  Ising, 5 Hubbard), so a layer reads at most r × the whole index and the average gate costs
  O(K/N) whatever c is. Measured, it stays within 1.5% (kicked Ising) and 2.5% (Hubbard) of
  the mean at every depth and N. c only bounds the worst single gate: kicked Ising c ≤ 1.56
  (N = 32), falling to 1.04–1.05 at N = 1024; Hubbard c grows to 1.82 (N = 512, 2 layers), from
  the open boundary. The per-site bars (panels a–c) show the edge strip below the mean and the
  interior above it; after two layers the strip is ~7 sites wide, half the 16×16 lattice, so the
  interior has not reached its bulk value. After one layer (a 2-site strip) c turns over, 1.44 at
  N = 128 to 1.14 at N = 1024.

## Workloads (identical for all engines)

- **Kicked Ising.** Open chain of N qubits; each layer `Rx(π/4)` on every site and `Rzz(π/4)` on
  every bond (2N−1 gates). Observable `Σᵢ Zᵢ`, Heisenberg picture, Pauli-weight cutoff 6.
- **2D Hubbard.** Open `lx × ly` lattice, N = 2·lx·ly modes (4×4, 8×4, 8×8, 16×8, 16×16, 32×16),
  t = 1, U = 1.5, dt = 0.07. A Trotter step is hop-up on every bond, hop-down on every bond, then
  `n↑n↓` on every site. The bonds come in four commuting classes (horizontal from even x, from odd
  x, vertical from even y, from odd y): ordered row by row instead, one step's non-commuting hops
  carry a mode across the whole lattice and K grows as a power of N. Observable: the staggered
  magnetisation `Σ (−1)^(x+y) (n↑ − n↓)`; reference state: the Néel checkerboard. Truncation:
  a monomial survives if it is a product of at most 4 Majoranas or is fully paired (monoprop's
  `cutoff_type="length"`, the drivers' `--cutoff-type length`; even parity makes 4 the only
  useful length below 6, which holds 2.9e8 terms at N = 64 after two layers).
  `MajoranaPropagation.jl`'s own `max_weight` has no fully-paired exception, so it gets the same
  rule as a `customtruncfunc`; the two engines' term sets and expectations agree exactly on 4×2,
  4×4, 8×4 and 8×8 lattices at 1–3 layers (job 2028185). It also gets
  `truncate_each_mr = true`: by default it truncates a hopping gate once, after both of its
  Majorana rotations, and keeps more terms (20,888 against 18,831 on a 4×2 lattice in an early test).
- **No coefficient truncation anywhere** (`lower_atol = 0`): the cutoff alone fixes the term set.
  The Hubbard drivers take no tolerance at all, and the figure refuses a row with one.

`make_figure.py` **refuses to build** unless, at every point, every engine reports the same term
count, gate count and expectation value (agreeing to 1.2e-13), one thread, and one Deucalion x86
node, and every count saw the timed operator's term count.

## Timing rules

- **One node, packed.** All points ran on `cnx002`. Kicked Ising: the 7 layers in job 2018658
  (15-min limit) and the points it left unsettled under 30 min in job 2026810; 10 and 13 layers
  in job 2027835. Hubbard at length 4: every point in job 2028187. `scripts/slurm/campaign.sbatch` runs up to eight points at once,
  one per NUMA domain (CPUs 8, 24, …, 120, `taskset`), longest first. The pilot (job 2016941) timed
  four points alone, beside seven copies, and alone again: packed/alone medians were
  0.999–1.005, and 1.037 for one Julia point whose two alone runs differed by 4.7%.
- **Same repetition rule for every engine** (`keep_timing` in each driver): the minimum of up to 10
  rounds on a freshly built operator; always a second round when two fit in the limit, since a
  fresh process's first round can be far slower (`PauliPropagation.jl` at N=32: 0.36 s vs 0.19 s);
  further rounds while the timed total is under 60 s. Rows record `rounds`, `max_rounds` and
  `time_limit`. (The previous version of this figure took monoprop's best of 10 against one round of
  each reference, which flattered monoprop at small N.)
- **Time limit.** 30 min of timed propagation per point, the same for every engine; the process is
  killed at 30 min + 10 min of setup. A killed point is a `timeout` row; a row whose single round
  ran past 30 min (one here) is no better than a bound either. Both are left out of the figure
  and the fits, and the caption lists them. An implied point is written as a timeout row without running, with an
  `implied` reason, when the same engine exceeded the limit on a shallower circuit at the same N
  or at a smaller N at the same depth (`IMPLIED`, or `IMPLIED_IF` decided in flight once that
  point finishes).
- **Reuse across limits.** A row measured under the earlier 15-min limit is reused only if it is
  *settled* under 30 min, i.e. a rerun under the longer limit would take the same minimum:
  it finished all its rounds, or ran for the 60 s round budget, or stopped at two rounds with
  room for a third under its own limit, or its one round alone takes more than half the new
  limit. A 15-min timeout is never reused. `scripts/slurm/settled.py` filters the campaign queue
  by this rule (`SETTLED_IN`), and `merge_octave_runs.py` applies it again, refusing an
  unsettled row; a later run directory replaces an earlier one per engine, depth and N.
- **Counting is separate and untimed.** `scripts/count_anticommuting.py` (jobs 2026811, 2027833
  and 2028188)
  propagates the same circuit with monoprop, exports the operator, and counts exactly, for each
  gate of one layer, the terms anticommuting with it: a term anticommutes with a multi-term
  generator (a hopping gate's two bilinears, the interaction's diagonal monomials) when it
  anticommutes with any of them. Validated against a gate-by-gate brute force on small lattices
  (kicked Ising N = 8–16, Hubbard 2×2 to 4×4).

## Data and provenance

| file | what | rows |
| --- | --- | --- |
| `data/pauli.jsonl` | panel a: monoprop, `PauliPropagation.jl` (`julia_pauli`), `ppvm` | one per engine, depth (`layers`) and N |
| `data/majorana.jsonl` | panel b: monoprop, `MajoranaPropagation.jl` (`julia_majorana`) | one per engine, depth (`layers`, one Trotter step each) and N |
| `data/counts.jsonl` | panel c: `num_terms`, `anticommuting_per_gate` | one per depth and N that monoprop finished, up to 2e8 terms |
| `data/occupancy.jsonl` | the appendix figure: c and the list-length summary; Hubbard rows keep all 2N list lengths | one per model, depth and N, up to 1.6e8 terms (`scripts/occupancy.py`) |

| engine | version |
| --- | --- |
| monoprop | `0.9.3.dev13+g67e7c4ca0` (`Algorithmiq/monoprop` main 67e7c4ca), `_core.abi3.so` MD5 `cee8f20abf39fa6f9b5028dee9d93b7c` |
| `PauliPropagation.jl` | 0.9.0, `VectorPauliSum` (rows record `result_type`), Julia 1.12.7 |
| `MajoranaPropagation.jl` | 0.4.1 with `PauliPropagation.jl` 0.8.2, `VectorMajoranaSum`, Julia 1.12.7 |
| `ppvm` | `0.1.0+git.2570637b1459` |

Container choice: `VectorPauliSum` is 2.7–5.6× faster than `PauliSum` on this workload (N = 16–64,
0.9.0), and `VectorMajoranaSum` 2.1–2.3× faster than `MajoranaSum` on the Hubbard one (N = 16–64), so
both references run at their more favourable representation.

## Reproduce

Figure and caption from the shipped data (matplotlib and numpy only):

```bash
python make_figure.py            # -> figures/fig3_inverse_scaling.{pdf,png}, captions.txt
python occupancy_figure.py       # -> figures/fig3_occupancy.{pdf,png}, and its numbers
```

The measurement. Drivers in `scripts/`: `monoprop_single_layer.py` (both bases),
`ppvm_single_layer.py`, `julia_pauli_single_layer.jl` (project `scripts/julia-pauli/`) and
`julia_majorana_hubbard2d.jl` (project `scripts/julia-majorana/`); each Julia project carries a
`Manifest.toml`, and the two cannot share one (MajoranaPropagation 0.4.1 requires
PauliPropagation 0.8). Fill in a copy of `scripts/slurm/site.env.example` (monoprop built with
`monoprop_MAX_NUM_MODES >= 1024`; Julia ≥ 1.11; instantiate both Julia projects where there is
internet), then from this directory:

```bash
sbatch -A <account> -p <partition> -t 04:00:00 --chdir="$PWD" \
    scripts/slurm/campaign.sbatch /path/to/site.env          # every timed point, one node
sbatch -A <account> -p <partition> -t 03:00:00 --chdir="$PWD" \
    scripts/slurm/count.sbatch /path/to/site.env             # panel c, any node
python merge_octave_runs.py <campaign RUN_DIR> <count RUN_DIR>   # checks, then writes data/
python make_figure.py
```

`campaign.sbatch` takes `ROWS`, `NS`, `PAULI_LAYERS`, `HUBBARD_LAYERS`, `HUBBARD_CUTOFF`,
`HUBBARD_CUTOFF_TYPE`,
`CEILING_S`, `MAX_N`, `IMPLIED`, `IMPLIED_IF`, `SETTLED_IN` and `MAX_CONCURRENCY` from the
environment (see `site.env.example`); `scripts/slurm/octave_point.sbatch`
runs one engine at one depth and one or more N. `count.sbatch` takes the same depth knobs and runs
the points above ~1.5e7 terms (`BIG`) one at a time. `merge_octave_runs.py` takes any number of
run directories, keys rows by the depth they record (both file-name styles), and refuses on more
than one row per engine, depth and N, a wrong engine field or library version, differing workload
keys, more than one thread, term or gate mismatches, expectation spreads above 1e-9, a row at another
cutoff, a reused row that is not settled under the 30-min limit, or a count whose term count
differs from the timed row. The figure's data came from

```bash
R=<runs>/fig3
python merge_octave_runs.py $R/campaign-2018658 $R/campaign-2023575 $R/probe-s3 \
    $R/campaign-2026810 $R/campaign-2027835 $R/campaign-2028187 \
    $R/count-2026811 $R/count-2027833 $R/count-2028188
HUBBARD_CUTOFF=4 HUBBARD_CUTOFF_TYPE=length ...   # the Hubbard campaign, count and occupancy
python occupancy_figure.py --merge $R/occupancy-2028177 $R/occupancy-len4
```

Rows at another cutoff (Hubbard support 3) or depth (kicked Ising 5 and 9) stay in their run
directories and are left out. `--dry-run` writes nothing; `--allow-partial --outdir
<dir>` previews a campaign in flight, and `make_figure.py --allow-partial --data <dir> --outdir
<dir>` draws it.

## Method notes

- **Fit window.** Each depth's exponents are fitted over the widest N range every engine finished
  there (kicked Ising N ≤ 512 / 256 / 128, Hubbard N ≤ 1024 / 256 / 128). Points above it are drawn
  but enter no fit.
- **Why PauliPropagation.jl wins at small N.** Up to N = 64 a Pauli string is one or two machine
  words, so the dense anticommutation test is a single AND and popcount; monoprop's index lookup
  and its higher per-gate setup (~3.5 µs against ~1 µs) only pay off once strings span several
  words. `PauliPropagation.jl`'s per-term cost then grows with the word count while monoprop's falls.
  Profiled at N = 32 (callgrind, 7 layers): monoprop spends ~620 instructions on each
  anticommuting term (position-list partner merge, term rebuild, sign, position-comparing hash
  lookup); finding the anticommuting terms is under 0.5%. The same algorithm on one-word terms
  matches `PauliPropagation.jl` (0.10 s against 0.107 s for the layer, monoprop 0.30 s). Of the
  ~3.5 µs per gate, ~1.6 µs is Python-side circuit preparation and ~1.2 µs C++ per-gate
  allocation and setup.
- **Why monoprop sits above 1/N (kicked Ising).** Per-term instruction counts are flat in N
  (full-size callgrind, PR #383). The gap accrues below N=256 as one-time steps: the operator
  outgrows a 16 MiB L3 slice, and above N=128 stored term positions widen from 8 to 16 bits.

## Files

```
make_figure.py          the figure, the like-for-like checks and the caption
occupancy_figure.py     the appendix figure of how evenly the operators are supported
merge_octave_runs.py    checks and merges campaign run directories into data/
data/                   pauli.jsonl, majorana.jsonl, counts.jsonl, occupancy.jsonl
scripts/                the four drivers, count_anticommuting.py, occupancy.py, julia-pauli/
                        and julia-majorana/ projects, slurm/ (campaign, octave_point, count,
                        occupancy)
figures/                fig3_inverse_scaling.{pdf,png}, fig3_occupancy.{pdf,png}, captions.txt
```
