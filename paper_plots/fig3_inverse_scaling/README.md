# Paper Fig. 3 — inverse time scaling with qubit number, at three circuit depths

One page-wide row of three panels (paper `fig:inverse-scaling`, a `figure*`):

| | (a) kicked Ising chain (Pauli) | (b) 2D Hubbard lattice (Majorana) |
| --- | --- | --- |
| engines | monoprop, `PauliPropagation.jl` 0.9.0, `ppvm` | monoprop, `MajoranaPropagation.jl` 0.4.1 |
| N | 32 … 1024 qubits | 32 … 1024 modes |
| depths (small / medium / large) | 5 / 7 / 9 layers | 2 / 3 / 4 Trotter steps |
| cutoff | Pauli weight 6 | support 3 |

- **(a), (b) cost:** nanoseconds for one gate to act on one term, against N. Colour, line style
  and marker are the engine (monoprop blue, solid, filled; the Julia reference vermillion,
  dashed, open; `ppvm` green, dotted, open triangles). Shade is the depth, light to dark, with
  a grey key in the panel; the three depths of one engine lie close together, so only the
  medium depth carries markers. The near-black dashed line is 1/N through monoprop's first
  medium-depth point. A point over the 30-min limit is an open marker on the "> 30 min" rail
  at the top. The fitted exponents are in the caption and below, not on the figure.
- **(c) work per gate:** from counts rather than timings, the K terms a dense code tests per
  gate divided by the terms that anticommute with it (the only ones monoprop's index visits),
  against N. Both models and all depths (medium bold, others faint) lie on lines ∝ N; the
  labels give the medium depth's ratio at N = 1024.

The design went through a busier eight-panel version (one panel per depth, plus K and the
anticommuting count as separate lines); since the depths of one engine nearly coincide, they
are folded into one panel per model, and the two counts into their ratio. Colours are
Okabe-Ito, checked with a CVD validator: the three engine colours pass the colour-blind and
normal-vision separation checks over all three pairs, and each three-shade depth ramp is
monotone in lightness with its light end at ≥ 2:1 against white. Text is ink, never a series
colour.

A gate anticommutes with ~cK/N terms (c ≈ 4–4.7 for kicked Ising, 5.2–5.7 for Hubbard), and at
fixed depth K grows ∝ N, so monoprop's work per gate is flat in N while a dense code's grows ∝ N;
and a dense code's per-term test also grows with the machine words a string spans. Hence
monoprop's cost per term falls as 1/N and the references' does not. Where K is small and strings
fit one word (kicked Ising at N ≤ 64), `PauliPropagation.jl` is faster.

## Result

Every number below comes from the rows; `make_figure.py` prints the exponents and ratios and
writes them to `figures/captions.txt`. ns per gate per term:

| N | mono 5 | PP.jl 5 | ppvm 5 | mono 7 | PP.jl 7 | ppvm 7 | mono 9 | PP.jl 9 | ppvm 9 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.48 | **0.47** | 1.38 | 1.82 | **0.72** | 3.20 | 2.10 | **0.87** | 4.77 |
| 64 | 0.80 | **0.48** | 1.46 | 1.00 | **0.86** | 3.86 | 1.16 | **0.92** | 5.27 |
| 128 | **0.43** | 1.11 | 2.92 | **0.52** | 1.25 | 4.43 | **0.58** | 1.34 | 5.61 |
| 256 | **0.229** | 1.73 | 4.13 | **0.275** | 1.98 | 5.71 | **0.307** | 2.14 | 7.04 |
| 512 | **0.115** | 2.99 | 5.72 | **0.140** | 3.40 | 7.59 | **0.157** | 3.67 | > 30 min |
| 1024 | **0.059** | 5.05 | 8.70 | **0.071** | > 30 min¹ | > 30 min | **0.079** | > 30 min | > 30 min² |

| N | mono 2 | MP.jl 2 | mono 3 | MP.jl 3 | mono 4 | MP.jl 4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | **3.52** | 9.46 | **5.38** | 17.9 | **6.68** | 23.4 |
| 64 | **1.49** | 7.96 | **2.21** | 12.9 | **3.19** | 19.6 |
| 128 | **0.656** | 8.87 | **0.885** | 13.8 | **1.33** | 22.6 |
| 256 | **0.348** | 14.7 | **0.392** | 17.7 | **0.488** | 24.0 |
| 512 | **0.171** | 23.9 | **0.174** | 25.8 | **0.203** | > 30 min |
| 1024 | **0.083** | 46.3 | **0.083** | > 30 min | **0.095** | > 30 min² |

¹ One round finished past the limit (2278 s): drawn on the rail, in no fit. ² Not run: the
engine already exceeded the limit one depth lower at the same N (`implied` rows).

- **Kicked Ising.** monoprop falls as N^−0.93 at every depth. `PauliPropagation.jl` grows as
  N^+0.73 / +0.57 / +0.44 and `ppvm` as N^+0.56 / +0.31 / +0.18 at 5 / 7 / 9 layers (each fitted
  over the widest N every engine finished: 1024, 512, 256). `PauliPropagation.jl` is 2.4–3.1×
  faster than monoprop at N = 32 at every depth and still faster at N = 64; monoprop leads from
  N = 128 (2.3–2.6×) to 86× at N = 1024, 5 layers. `ppvm` is 7% faster than monoprop at N = 32 and 5
  layers, and 150× slower at N = 1024. At 9 layers and N = 1024 (7.9e7 terms) monoprop takes 114 s.
- **2D Hubbard.** monoprop falls as N^−1.07 / −1.24 / −1.26 at 2 / 3 / 4 steps;
  `MajoranaPropagation.jl` goes as N^+0.48 / +0.15 / +0.03 (fitted over N ≤ 1024 / 512 / 256).
  monoprop is faster at every point: 2.7–3.5× at N = 32, and 560× at N = 1024 after 2 steps
  (1081 s against 1.9 s). At 4 steps and N = 1024 (9.3e7 terms) monoprop takes 87 s. At N ≤ 64
  three and four steps hold the same terms: the support-3 cutoff saturates a small lattice.
- **Work per gate.** The anticommuting terms per gate are 4.06–4.68 K/N for kicked Ising and
  5.22–5.66 K/N for Hubbard, at every timed point; so the ratio K / anticommuting grows ∝ N, to
  220–252× (kicked Ising) and 181–184× (Hubbard) at N = 1024.

## Workloads (identical for all engines)

- **Kicked Ising.** Open chain of N qubits; each layer `Rx(π/4)` on every site and `Rzz(π/4)` on
  every bond (2N−1 gates). Observable `Σᵢ Zᵢ`, Heisenberg picture, Pauli-weight cutoff 6.
- **2D Hubbard.** Open `lx × ly` lattice, N = 2·lx·ly modes (4×4, 8×4, 8×8, 16×8, 16×16, 32×16),
  t = 1, U = 1.5, dt = 0.07. A Trotter step is hop-up on every bond, hop-down on every bond, then
  `n↑n↓` on every site. The bonds come in four commuting classes (horizontal from even x, from odd
  x, vertical from even y, from odd y): ordered row by row instead, one step's non-commuting hops
  carry a mode across the whole lattice and K grows as a power of N. Observable: the staggered
  magnetisation `Σ (−1)^(x+y) (n↑ − n↓)`; reference state: the Néel checkerboard. Truncation:
  a monomial survives if it touches at most 3 modes or is fully paired (monoprop's
  `cutoff_type="support"`). Support 3 rather than 4 keeps every depth on a node to N = 1024
  (4 steps: 9.3e7 terms); at support 4, three steps already held 1e8 terms at N = 128. `MajoranaPropagation.jl` gets the same rule as a `customtruncfunc`, and
  `truncate_each_mr = true`: by default it truncates a hopping gate once, after both of its
  Majorana rotations, and keeps more terms (20,888 against 18,831 on a 4×2 lattice in an early test).
- **No coefficient truncation anywhere** (`lower_atol = 0`): the cutoff alone fixes the term set.
  The Hubbard drivers take no tolerance at all, and the figure refuses a row with one.

`make_figure.py` **refuses to build** unless, at every point, every engine reports the same term
count, gate count and expectation value (agreeing to 1.6e-11), one thread, and one Deucalion x86
node, and every count saw the timed operator's term count.

## Timing rules

- **One node, packed.** All points ran on `cnx002`. Kicked Ising: the 7 layers in job 2018658,
  5 and 9 layers in job 2023575 (both under a 15-min limit), and the points those left unsettled
  under 30 min in job 2026810. Hubbard: N = 256–1024 at 2–4 steps in the probe job 2025971
  (15 min), and the rest in job 2026810. `scripts/slurm/campaign.sbatch` runs up to eight points at once,
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
  ran past 30 min (one here) is no better than a bound either. Both are drawn on the "> 30 min"
  rail and enter no fit. An implied point is written as a timeout row without running, with an
  `implied` reason, when the same engine exceeded the limit one depth lower at the same N
  (`IMPLIED`, or `IMPLIED_IF` decided in flight once the shallower point finishes).
- **Reuse across limits.** A row measured under the earlier 15-min limit is reused only if it is
  *settled* under 30 min, i.e. a rerun under the longer limit would take the same minimum:
  it finished all its rounds, or ran for the 60 s round budget, or stopped at two rounds with
  room for a third under its own limit, or its one round alone takes more than half the new
  limit. A 15-min timeout is never reused. `scripts/slurm/settled.py` filters the campaign queue
  by this rule (`SETTLED_IN`), and `merge_octave_runs.py` applies it again, refusing an
  unsettled row; a later run directory replaces an earlier one per engine, depth and N.
- **Counting is separate and untimed.** `scripts/count_anticommuting.py` (job 2026811, `cnx001`)
  propagates the same circuit with monoprop, exports the operator, and counts exactly, for each
  gate of one layer, the terms anticommuting with it: a term anticommutes with a multi-term
  generator (a hopping gate's two bilinears, the interaction's diagonal monomials) when it
  anticommutes with any of them. Validated against a gate-by-gate brute force on small lattices
  (kicked Ising N = 8–16, Hubbard 2×2 to 4×4).

## Data and provenance

| file | what | rows |
| --- | --- | --- |
| `data/pauli.jsonl` | panel a: monoprop, `PauliPropagation.jl` (`julia_pauli`), `ppvm` | one per engine, depth (`layers`) and N |
| `data/majorana.jsonl` | panel b: monoprop, `MajoranaPropagation.jl` (`julia_majorana`) | one per engine, depth (Trotter steps, `layers`) and N |
| `data/counts.jsonl` | panel c: `num_terms`, `anticommuting_per_gate` | one per depth and N that monoprop finished |

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
    $R/campaign-2026810 $R/count-2026811
``` `--dry-run` writes nothing; `--allow-partial --outdir
<dir>` previews a campaign in flight, and `make_figure.py --allow-partial --data <dir> --outdir
<dir>` draws it.

## Method notes

- **Fit window.** Each depth's exponents are fitted over the widest N range every engine finished
  there (kicked Ising N ≤ 1024 / 512 / 256, Hubbard N ≤ 1024 / 512 / 256). Points above it are drawn
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
merge_octave_runs.py    checks and merges campaign run directories into data/
data/                   pauli.jsonl, majorana.jsonl, counts.jsonl
scripts/                the four drivers, count_anticommuting.py, julia-pauli/ and
                        julia-majorana/ projects, slurm/ (campaign, octave_point, count)
figures/                fig3_inverse_scaling.{pdf,png}, captions.txt
```
