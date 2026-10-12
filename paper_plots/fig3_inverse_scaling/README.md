# Paper Fig. 3 — inverse time scaling with qubit number, and where dense codes stay faster

Two rows, two columns (paper `fig:inverse-scaling`, a page-wide `figure*`):

| | (a, c) cost per term vs N | (b, d) cost per gate vs K |
| --- | --- | --- |
| **kicked Ising** (Pauli) | monoprop, `PauliPropagation.jl` 0.9.0, `ppvm`; N = 32 … 1024, 7 layers | monoprop vs `PauliPropagation.jl` at N = 32, 128, 512; 9 layers, one point per layer |
| **2D Hubbard** (Majorana) | monoprop, `MajoranaPropagation.jl` 0.4.1; N = 32 … 1024 modes, 2 Trotter steps | monoprop vs `MajoranaPropagation.jl` at N = 32, 128, 256; one point per commuting gate group |

monoprop's per-gate cost is ~`a + bK/N` while a dense engine pays ~`K` times a per-term cost that
grows with the number of machine words a string spans. So the left column falls as `1/N` for
monoprop and not for the references, and the right column shows the regime the left one hides:
at small N and small K, `PauliPropagation.jl` is faster.

## Result

Every number below is printed by `make_figure.py` and written to `figures/captions.txt`, computed
from the rows, never typed. ns per gate per term (panels a, c):

| N | monoprop (Pauli) | PP.jl | ppvm | monoprop (Majorana) | MP.jl |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1.82 | **0.72** | 3.20 | **2.60** | 5.20 |
| 64 | 1.00 | **0.86** | 3.87 | **1.16** | 4.31 |
| 128 | **0.52** | 1.25 | 4.43 | **0.55** | 4.58 |
| 256 | **0.275** | 1.98 | 5.71 | **0.267** | 6.25 |
| 512 | **0.140** | 3.40 | 7.61 | **0.125** | > 15 min |
| 1024 | **0.071** | > 15 min | > 15 min | **0.062** | > 15 min |

- **Kicked Ising.** Fitted over N ≤ 512: monoprop N^−0.93, `PauliPropagation.jl` N^+0.57, `ppvm`
  N^+0.31. `PauliPropagation.jl` is 2.5× faster than monoprop at N=32 and level at N=64; monoprop
  leads it 2.4× at N=128 and 24× at N=512, and leads `ppvm` 1.8× at N=32 and 54× at N=512. At
  N=1024 monoprop takes 26 s for 2.5e7 terms; both references exceed 15 min.
- **Crossover (b).** monoprop's floor is ~3.5 µs per gate against ~1 µs for `PauliPropagation.jl`.
  At N=32 `PauliPropagation.jl` is ~3× faster at every K up to 1.8e6 (no crossover); at N=128
  monoprop overtakes it at K* ≈ 1.1e3 and leads 2.3× at K ≈ 7e6; at N=512 monoprop is faster at
  every K, by 23× at K ≈ 3e7.
- **2D Hubbard.** Fitted over N ≤ 256: monoprop N^−1.09, `MajoranaPropagation.jl` N^+0.09. monoprop is
  faster at every N and every K measured: 2.0× per term at N=32, 23× at N=256. At N=1024 monoprop
  propagates 3.0e8 terms in 90 s.

## Workloads (identical for all engines)

- **Kicked Ising.** Open chain of N qubits; each layer `Rx(π/4)` on every site and `Rzz(π/4)` on
  every bond (2N−1 gates). Observable `Σᵢ Zᵢ`, Heisenberg picture, Pauli-weight cutoff 6.
- **2D Hubbard.** Open `lx × ly` lattice, N = 2·lx·ly modes (4×4, 8×4, 8×8, 16×8, 16×16, 32×16),
  t = 1, U = 1.5, dt = 0.07. A Trotter step is hop-up on every bond, hop-down on every bond, then
  `n↑n↓` on every site. The bonds come in four commuting classes (horizontal from even x, from odd
  x, vertical from even y, from odd y): ordered row by row instead, one step's non-commuting hops
  carry a mode across the whole lattice and K grows as a power of N. Observable: the staggered
  magnetisation `Σ (−1)^(x+y) (n↑ − n↓)`; reference state: the Néel checkerboard. Truncation:
  a monomial survives if it touches at most 4 modes or is fully paired (monoprop's
  `cutoff_type="support"`). `MajoranaPropagation.jl` gets the same rule as a `customtruncfunc`, and
  `truncate_each_mr = true`: by default it truncates a hopping gate once, after both of its
  Majorana rotations, and keeps more terms (20,888 against 18,831 on a 4×2 lattice in an early test).
- **No coefficient truncation anywhere** (`lower_atol = 0`): the cutoff alone fixes the term set.
  The Hubbard drivers take no tolerance at all, and the figure refuses a row with one.

`make_figure.py` **refuses to build** unless, at every point, every engine reports the same term
count (and the same per-step term sequence in the per-step sweeps), gate count and expectation value
(agreeing to 1.4e-11), one thread, and one Deucalion x86 node.

## Timing rules

- **One node, packed.** All points ran in one allocation on `cnx002` (job 2018658; the Hubbard N=256
  per-step pair in job 2020273 on the same node). `scripts/slurm/campaign.sbatch` runs up to eight
  points at once, one per NUMA domain (CPUs 8, 24, …, 120, `taskset`), longest first. The pilot
  (job 2016941) timed four points alone, beside seven copies, and alone again: packed/alone medians
  were 0.999–1.005, and 1.037 for one Julia point whose two alone runs differed by 4.7%.
- **Same repetition rule for every engine** (`keep_timing` in each driver): the minimum of up to 10
  rounds on a freshly built operator; always a second round when two fit in the limit, since a
  fresh process's first round can be far slower (`PauliPropagation.jl` at N=32: 0.36 s vs 0.19 s);
  further rounds while the timed total is under 60 s. Rows record `rounds`, `max_rounds` and
  `time_limit`. (The previous version of this figure took monoprop's best of 10 against one round of
  each reference, which flattered monoprop at small N.)
- **Time limit.** 15 min of timed propagation per point, the same for every engine; the process is
  killed at 15 min + 10 min of setup. A killed point is a `timeout` row, labelled "> 15 min" in the
  figure; a row whose single round ran past 15 min would be labelled the same. The `PauliPropagation.jl`
  per-step N=512 run (one round, 22 min) is the one point over the limit that is plotted: in the
  per-step sweep the steps, not the total, are the measurement.
- **Per step.** Panels b and d time each propagate call: one per layer for kicked Ising, one per
  commuting gate group (nine per Trotter step) for Hubbard, whose K rises a thousandfold within the
  second step. A step's K is the geometric mean of the terms entering and leaving it.

## Data and provenance

| file | panel | engines |
| --- | --- | --- |
| `data/pauli_nsweep.jsonl` | a | monoprop, `PauliPropagation.jl` (`julia_pauli`), `ppvm` |
| `data/pauli_perlayer.jsonl` | b | monoprop, `PauliPropagation.jl` |
| `data/majorana_nsweep.jsonl` | c | monoprop, `MajoranaPropagation.jl` (`julia_majorana`) |
| `data/majorana_perlayer.jsonl` | d | monoprop, `MajoranaPropagation.jl` |

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
sbatch -A <account> -p <partition> -t 01:45:00 --chdir="$PWD" \
    scripts/slurm/campaign.sbatch /path/to/site.env          # every point, one node
python merge_octave_runs.py "$RUN_DIR"                       # checks, then writes data/
python make_figure.py
```

`campaign.sbatch` takes `ROWS`, `SWEEPS`, `NSWEEP_NS`, `PERLAYER_NS` and `MAX_CONCURRENCY` from the
environment to run a subset; `scripts/slurm/octave_point.sbatch` runs one engine at one or more N.
`merge_octave_runs.py` takes several run directories, keeps each dataset's N grid (`GRID`), and
refuses on more than one row per engine and N, a wrong engine field or library version, differing
workload keys, more than one thread, term/gate/per-step mismatches, expectation spreads above 1e-9,
or a timeout in a per-step sweep. `--dry-run` writes nothing; `--allow-partial --outdir <dir>`
previews a campaign in flight.

## Method notes

- **Fit window.** Each row's exponents are fitted over the widest N range every engine finished
  (N ≤ 512 kicked Ising, N ≤ 256 Hubbard). Points above it are drawn but enter no fit.
- **Why PauliPropagation.jl wins at small N.** Up to N = 64 a Pauli string is one or two machine
  words, so the dense anticommutation test is a single AND and popcount; monoprop's index lookup
  and its higher per-gate setup (~3.5 µs against ~1 µs) only pay off once strings span several
  words. `PauliPropagation.jl`'s per-term cost then grows with the word count while monoprop's falls.
- **Why monoprop sits above 1/N (kicked Ising).** Per-term instruction counts are flat in N
  (full-size callgrind, PR #383). The gap accrues below N=256 as one-time steps: the operator
  outgrows a 16 MiB L3 slice, and above N=128 stored term positions widen from 8 to 16 bits.

## Files

```
make_figure.py          the figure, the like-for-like checks and the caption
merge_octave_runs.py    checks and merges campaign run directories into data/
data/                   the four merged JSONL files above
scripts/                the four drivers, julia-pauli/ and julia-majorana/ projects, slurm/
figures/                fig3_inverse_scaling.{pdf,png}, captions.txt
```
