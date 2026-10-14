# Paper Fig. 3 — inverse time scaling with qubit number, at three circuit depths

One page-wide row of three panels (paper `fig:inverse-scaling`, a `figure*`):

| | (a) kicked Ising chain (Pauli) | (b) 2D Hubbard lattice (Majorana) |
| --- | --- | --- |
| engines | monoprop, `PauliPropagation.jl` 0.9.0, `ppvm` | monoprop, `MajoranaPropagation.jl` 0.4.1 |
| N | 32 … 1024 qubits | 32, 72, 128, 242, 512, 968 modes (L × L, L = 4 … 22) |
| depths (small / medium / large) | 7 / 10 / 13 layers | 3 / 4 / 5 layers (one whole Trotter step each) |
| cutoff | Pauli weight 6 | support 3 (modes a monomial acts on) |

- **(a), (b) cost:** nanoseconds for one gate to act on one term, against N. Colour is the
  engine, each its own (monoprop blue, `PauliPropagation.jl` vermillion, `MajoranaPropagation.jl`
  wine, `ppvm` green), every series the same solid line at full colour. The marker alone is the
  depth, the same for every engine: small a circle, medium a square, large a triangle, at every
  point. The near-black dash-dot line is 1/N through monoprop's first medium-depth point. Each
  panel keys its own engines and the guide in its lower left, over one row of its depth markers
  with that panel's layer counts; the two panels share the y axis, which reaches six times below
  the cheapest point to leave that corner clear. A point over the 30-min limit, or not run because a
  smaller workload already exceeded it, has no cost to plot and is left out; the caption lists
  them. The fitted exponents are in the caption and below, not on the figure.
- **(c) work per gate:** from counts rather than timings, the percentage of the K terms one
  gate reads and rotates, at the medium depth of each model (squares, filled for kicked Ising
  and open for Hubbard, keyed "(a) 10 layers" and "(b) 4 layers" above the lines' key). A dense code reads all of
  them (the grey dashed 100% line, the top of the axis). monoprop reads the index lists of the
  gate's anticommutation support (solid blue: every entry of every list its scans fold, 30% of K at
  N = 32 for both models, 0.94% at N = 1024 and 1.0% at N = 968) and rotates only their
  symmetric difference, the anticommuting set (dashed blue: 15% and 16% at N = 32, 0.46% at N = 1024
  and 0.58% at N = 968). The grey band, from the higher read line up to 100%, is what the index
  never reads, named in its middle ("never read by monoprop"). Each panel's plotting box is
  square (7.0 × 2.42 in).

The design went through a busier eight-panel version (one panel per depth, plus K and the
anticommuting count as separate lines); since the depths of one engine nearly coincide, they
are folded into one panel per model, and the two counts into the share touched. A later version
told the depths apart by shade and named the engines beside their lines; shades read as
different transparencies, so every series is now the same solid line at full colour, the depth
is the marker alone. The engines were first keyed by one legend under the row; each panel now
keys its own engines and depths in its lower left, and the row is shorter (2.3 in) so the
panels are not tall and narrow. One colour per engine: the two
Julia references never share a panel but are different codes, so `MajoranaPropagation.jl` has
its own colour rather than sharing vermillion. The other three are Okabe-Ito; Okabe-Ito's
reddish purple (#CC79A7) fell short of the colour-blind separation target against `ppvm` green
(ΔE 7.6 under deuteranopia, against a target of 8) and is only just 3:1 on white, so it is
Tol's wine (#882255), which passes every check against the other three (one entry of
`ENGINE_COLOR`). Text is ink, never a
series colour; in (c) the two models are told apart by the marker's fill, keyed in the panel.

A gate anticommutes with ~cK/N terms (c ≈ 4.4–4.9 for kicked Ising, 5.2–5.7 for Hubbard), and at fixed depth K grows ∝ N, so monoprop's work per
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

| N | lattice | mono 3 | MP.jl 3 | mono 4 | MP.jl 4 | mono 5 | MP.jl 5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4×4 | **5.37** | 17.7 | **6.67** | 23.5 | **7.43** | 26.5 |
| 72 | 6×6 | **2.12** | 15.8 | **3.05** | 24.6 | **3.61** | 29.9 |
| 128 | 8×8 | **0.891** | 13.8 | **1.34** | 22.6 | **1.71** | 30.3 |
| 242 | 11×11 | **0.445** | 17.3 | **0.581** | 24.7 | **0.784** | 35.5 |
| 512 | 16×16 | **0.173** | 25.8 | 0.203 | > 30 min | 0.248 | > 30 min |
| 968 | 22×22 | 0.0946 | > 30 min | 0.106 | > 30 min² | 0.118 | > 30 min |

¹ One round finished past the limit (2278 s): left out of the figure and the fits. ² Not run: the
engine already exceeded the limit on a shallower circuit at the same N, or at the next smaller N
at the same depth (`implied` rows). monoprop finishes every Hubbard point.

- **Kicked Ising.** monoprop falls as N^−0.93 / −0.94 / −0.98 at 7 / 10 / 13 layers.
  `PauliPropagation.jl` grows as N^+0.57 / +0.44 / +0.31 and `ppvm` as N^+0.31 / +0.14 / +0.01
  (each fitted over the widest N every engine finished: 512, 256, 128). `PauliPropagation.jl` is
  2.4–2.6× faster than monoprop at N = 32 at every depth and still faster at N = 64; monoprop
  leads from N = 128 (2.2–2.4×) to 24× at N = 512, 7 layers. `ppvm` is 1.8–2.9× slower than
  monoprop at N = 32 and 54× slower at N = 512, 7 layers. Neither dense code finishes N = 1024
  at any depth; monoprop takes 26 s (7 layers, 2.5e7 terms) to 868 s (13 layers, 3.7e8 terms).
- **2D Hubbard.** monoprop falls as N^−1.25 / −1.22 / −1.12 at 3 / 4 / 5 layers;
  `MajoranaPropagation.jl` goes as N^+0.12 / +0.01 / +0.13 (fitted over N ≤ 512 / 242 / 242).
  monoprop is faster at every point: 3.3–3.6× at N = 32, 43–45× at N = 242 after 4 and 5 layers
  and 149× at N = 512 after 3. Its slope is a little steeper than 1/N because its own cost per
  visited term falls, 33–46 ns at N = 32 to 16–20 ns at N = 968; `MajoranaPropagation.jl` spends
  14–36 ns per tested term. At 5 layers and N = 968 (2.0e8 terms) monoprop takes 271 s. The
  cutoff saturates the small lattices (4×4 from 3 layers, 6×6 and 8×8 from 4), so the depths hold
  the same terms there and separate from N = 242.
- **Work per gate.** The anticommuting terms per gate are 4.43–4.92 K/N for kicked Ising and
  5.22–5.66 K/N for Hubbard, at every counted point; so the ratio K / anticommuting grows ∝ N, to
  216–231× (kicked Ising, N = 1024) and 171× (Hubbard, N = 968). At the medium depth a gate reads
  9.6–9.8 K/N index entries for both models, 2.03–2.04× its anticommuting set for kicked Ising
  (an X gate reads exactly its anticommuting set, a ZZ gate four lists) and 1.73–1.86× for
  Hubbard.
- **Even support** (`occupancy_figure.py`, not in the paper; its data are the earlier
  length-4 Hubbard workload). After every layer, from the index list lengths: the longest list
  over the mean, c, and the average list a gate reads over one layer, over the mean.

## Workloads (identical for all engines)

- **Kicked Ising.** Open chain of N qubits; each layer `Rx(π/4)` on every site and `Rzz(π/4)` on
  every bond (2N−1 gates). Observable `Σᵢ Zᵢ`, Heisenberg picture, Pauli-weight cutoff 6.
- **2D Hubbard.** Open square `L × L` lattice, N = 2·L² modes (L = 4, 6, 8, 11, 16, 22, the
  most nearly square lattices that about double N under the 1024-mode build),
  t = 1, U = 1.5, dt = 0.07. A Trotter step is hop-up on every bond, hop-down on every bond, then
  `n↑n↓` on every site. The bonds come in four commuting classes (horizontal from even x, from odd
  x, vertical from even y, from odd y): ordered row by row instead, one step's non-commuting hops
  carry a mode across the whole lattice and K grows as a power of N. Observable: the staggered
  magnetisation `Σ (−1)^(x+y) (n↑ − n↓)`; reference state: the Néel checkerboard. Truncation:
  a monomial survives if it acts on at most 3 modes, whatever its length, or is fully paired
  (monoprop's `cutoff_type="support"`, the drivers' `--cutoff-type support`). Each layer is one
  whole Trotter step. The Majorana-length cutoff 4 was tried first: its K grows ~10⁴× per step, so
  2 layers reach 2.8e8 terms at N = 1024 and 3 layers stop fitting a node past N = 256, while at
  support 3 the three depths hold about as many terms per mode as the kicked-Ising ones at the
  top of the grid and monoprop finishes all of them.
  `MajoranaPropagation.jl` has no truncation with this rule, so it gets it as a
  `customtruncfunc`; the two engines' term sets and expectations agree at every point. It also
  gets `truncate_each_mr = true`: by default it truncates a hopping gate once, after both of its
  Majorana rotations, and keeps more terms (20,888 against 18,831 on a 4×2 lattice in an early test).
- **No coefficient truncation anywhere** (`lower_atol = 0`): the cutoff alone fixes the term set.
  The Hubbard drivers take no tolerance at all, and the figure refuses a row with one.

`make_figure.py` **refuses to build** unless, at every point, every engine reports the same term
count, gate count and expectation value (agreeing to 1.2e-13), one thread, and one Deucalion x86
node, and every count saw the timed operator's term count.

## Timing rules

- **One node, packed.** All points ran on `cnx002`. Kicked Ising: the 7 layers in job 2018658
  (15-min limit) and the points it left unsettled under 30 min in job 2026810; 10 and 13 layers
  in job 2027835. Hubbard: every point in job 2029402 (square lattices, support 3). `scripts/slurm/campaign.sbatch` runs up to eight points at once,
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
- **Counting is separate and untimed.** `scripts/count_anticommuting.py` (jobs 2026811, 2027833,
  2029403 and 2029624; the medium depths again in 2030444, adding the index reads)
  propagates the same circuit with monoprop, exports the operator, and counts exactly, for each
  gate of one layer, the terms anticommuting with it: a term anticommutes with a multi-term
  generator (a hopping gate's two bilinears, the interaction's diagonal monomials) when it
  anticommutes with any of them. Validated against a gate-by-gate brute force on small lattices
  (kicked Ising N = 8–16, Hubbard 2×2 to 4×4). Beside it, the index entries the same gates read:
  for each scan (one generator monomial) the lengths of the lists it XOR-folds in full
  (`scripts/read_weights.py`), summed over the layer and divided by its gates; the pivot list,
  read only where the fold is nonzero, is left out, and a list held as a dense bitmap is read as
  K/64 words, counted here by its entries. Checked against a per-scan brute force (Hubbard 4×4,
  kicked Ising N = 12).

## Data and provenance

| file | what | rows |
| --- | --- | --- |
| `data/pauli.jsonl` | panel a: monoprop, `PauliPropagation.jl` (`julia_pauli`), `ppvm` | one per engine, depth (`layers`) and N |
| `data/majorana.jsonl` | panel b: monoprop, `MajoranaPropagation.jl` (`julia_majorana`) | one per engine, depth (`layers`, one Trotter step each) and N |
| `data/counts.jsonl` | panel c: `num_terms`, `anticommuting_per_gate`, `index_reads_per_gate` (medium depths) | one per depth and N that monoprop finished, up to 2e8 terms |
| `data/occupancy.jsonl` | `occupancy_figure.py` (not in the paper; earlier length-4 Hubbard workload): c and the list-length summary | one per model, depth and N, up to 1.6e8 terms (`scripts/occupancy.py`) |

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
python occupancy_figure.py       # -> figures/fig3_occupancy.{pdf,png}; not in the paper
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
python merge_octave_runs.py $R/campaign-2018658 $R/campaign-2023575 $R/campaign-2026810 \
    $R/campaign-2027832 $R/campaign-2027835 $R/count-2026811 $R/count-2027833 \
    $R/campaign-2029402 $R/count-2029403 $R/count-2029624 $R/count-2030444
```

with the Hubbard campaign submitted as `ROWS=majorana NS="32 72 128 242 512 968"
HUBBARD_LAYERS="3 4 5" HUBBARD_CUTOFF=3 HUBBARD_CUTOFF_TYPE=support`, and `IMPLIED_IF` chaining
each larger `julia-majorana` point after the smaller one at the same depth. Rows at another depth
(kicked Ising 5 and 9, Hubbard 2), cutoff (Hubbard length 4) or lattice (the earlier 2:1 Hubbard
lattices 8×4, 16×8, 32×16, and their rows at N = 32, 128, 512, replaced by the later run) stay in
their run directories and are left out. `--dry-run` writes nothing; `--allow-partial --outdir
<dir>` previews a campaign in flight, and `make_figure.py --allow-partial --data <dir> --outdir
<dir>` draws it.

## Method notes

- **Fit window.** Each depth's exponents are fitted over the widest N range every engine finished
  there (kicked Ising N ≤ 512 / 256 / 128, Hubbard N ≤ 512 / 242 / 242). Points above it are drawn
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
occupancy_figure.py     how evenly the operators are supported (not in the paper)
merge_octave_runs.py    checks and merges campaign run directories into data/
data/                   pauli.jsonl, majorana.jsonl, counts.jsonl, occupancy.jsonl
scripts/                the four drivers, count_anticommuting.py, occupancy.py, julia-pauli/
                        and julia-majorana/ projects, slurm/ (campaign, octave_point, count,
                        occupancy)
figures/                fig3_inverse_scaling.{pdf,png}, fig3_occupancy.{pdf,png}, captions.txt
```
