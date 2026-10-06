# Paper Fig. 3 — inverse time scaling with qubit number

Cost of one gate acting on one term, in nanoseconds, against the qubit count `N`, for monoprop,
`ppvm` and `PauliPropagation.jl` on the same kicked-Ising workload (paper `fig:inverse-scaling`).
monoprop's per-gate cost is ~`K/N` while the reference engines pay ~`K`, so on this axis monoprop
falls and they rise.

## Result

Fitted over `N ≤ 512` (the range all three engines completed), ns per gate per term:

| engine | exponent | N=32 | N=512 | N=1024 |
| --- | ---: | ---: | ---: | ---: |
| monoprop | **N^-0.93** | 1.82 | 0.140 | 0.0700 |
| `ppvm` | N^+0.30 | 3.23 | 7.60 | 11.04 |
| `PauliPropagation.jl` | N^+0.35 | 4.02 | 10.26 | not measured (exceeded 3600 s) |

monoprop leads `ppvm` by 1.8× at N=32, 54× at N=512 and 158× at N=1024, and `PauliPropagation.jl`
by 2.2× and 73× at N=32 and N=512. At N=1024 monoprop sits 1.23× above the `1/N` guide drawn
through its N=32 point. Every number is printed by `make_figure.py` and written to
`figures/captions.txt`, which is computed from the rows, never typed.

## Workload (identical for all engines)

Open kicked-Ising chain of `N` qubits, 7 layers, each `Rzz(π/4)` on every bond then `Rx(π/4)` on
every site (2N−1 gates a layer). Observable `Σᵢ Zᵢ` (the unnormalised magnetisation; the paper's
text writes it with a `1/N`, which does not change any timing), Heisenberg picture, Pauli-weight
cutoff 6, `lower_atol = 0`, so no coefficient is pruned and the term set is fixed by gate supports.
`N = 32 … 1024` in powers of two, 607,054 to 25,254,286 terms. One thread, timed process pinned to
one core of an exclusive node.

`make_figure.py` **refuses to build** unless every engine reports the same term count, gate count
and expectation value (agreeing to 8.8e-14) at every N, one thread, and one declared node type.

## Data and provenance

All four series were measured in **one job on one node** (`cnx001`, Deucalion `dev-x86`, AMD EPYC
7742, 128 cores, SMT off, job 1984137, 2026-10-06), so every curve and every exponent comes from
the same machine. Rows record `host`, `library_version` and `rounds`.

| file | engine | version | notes |
| --- | --- | --- | --- |
| `data/monoprop_pauli_octave_l7_deucalion.jsonl` | monoprop (drawn) | `0.9.3.dev5+g664f84c00` | 10 rounds, minimum |
| `data/monoprop_main_pauli_octave_l7_deucalion.jsonl` | monoprop on `main` (not drawn) | `0.9.3.dev2+g508b536ec` | measured alongside, for the PR comparison |
| `data/ppvm_pauli_octave_l7_deucalion.jsonl` | `ppvm` | `0.1.0+git.2570637b1459` | 1 round; N=1024 takes 67 min |
| `data/julia_pauli_octave_v082_l7_deucalion.jsonl` | `PauliPropagation.jl` | 0.8.2 | `VectorPauliSum` (every row records `result_type`); 1 round; N=512 takes 15 min |

The Julia driver stops with an error unless it propagated a `VectorPauliSum`. Its N=1024 point is
absent: it hit the 3600 s ceiling in `scripts/slurm/octave_point.sbatch`.

## Reproduce

Figure and caption from the shipped data (matplotlib only; no monoprop build, no Julia):

```bash
python make_figure.py            # -> figures/fig3_inverse_scaling.{pdf,png}, captions.txt
```

The sweep (`scripts/`): `monoprop_single_layer.py`, `ppvm_single_layer.py` and
`julia_pauli_single_layer.jl` are the three drivers; `scripts/Project.toml` pins PauliPropagation
0.8.2 for Julia 1.10. `scripts/slurm/octave_point.sbatch` runs one engine at one or more `N` on a
whole node under a hard per-point timeout. Fill in a copy of `scripts/slurm/site.env.example`
first (monoprop must be built on a compute node with `monoprop_MAX_NUM_MODES >= 1024`), one per
monoprop build, since each build's venv is `PYTHON` and gets its own `RUN_DIR`. Submit from this
directory:

```bash
sbatch -A <project> --partition=dev-x86 --chdir="$PWD" scripts/slurm/octave_point.sbatch \
    /path/to/site.env 7 ppvm 32 64 128 256 512 1024        # likewise: julia, monoprop
python merge_octave_runs.py --monoprop "$RUN_DIR_NEW" --monoprop-main "$RUN_DIR_MAIN" \
    --ppvm "$RUN_DIR_THIRDPARTY" --julia "$RUN_DIR_THIRDPARTY"
python make_figure.py
```

To keep all series on one node, run the four engines back to back inside a single allocation.
`merge_octave_runs.py` checks and merges the per-job files into `data/*_octave_l7_deucalion.jsonl`.
It refuses on more than one row per N, a wrong engine field, any row not at 7 layers, cutoff 6 and
one thread, term, gate or expectation mismatches between series, two monoprop slots at the same
version, or `PauliPropagation.jl` not at 0.8.2. Every series must reach every `N ≤ 512`; a series
missing points above that is reported, not refused. `--dry-run` writes nothing.

## Method notes

- **Fit window.** Every exponent is fitted over `N ≤ 512` with the same window for all engines.
  Points above it are drawn but enter no fit. Fitted over every point each engine reached instead,
  the exponents are monoprop −0.94, `ppvm` +0.35 and `PauliPropagation.jl` +0.35.
- **y axis.** The top is 1.3× the highest point, capped at 20 ns (`Y_TOP_NS`), so a runaway point
  cannot squash the decade the `1/N` lives in; nothing reaches the cap in this data.
- **Why monoprop sits above 1/N.** Per-term instruction counts are flat in N (full-size callgrind,
  PR #383). The gap accrues below N=256 as one-time steps: the operator outgrows a 16 MiB L3 slice
  (simulated last-level misses per anticommuting term rise 0.63 → 2.33 from N=32 to N=256), and
  above N=128 stored term positions widen from 8 to 16 bits.
- **Small N.** At N ≤ 64 single-thread timings move by up to ±1.5% between builds whose callgrind
  profiles are equal or better; treat differences below that as build-to-build noise.

## Files

```
make_figure.py          the figure, the like-for-like checks and the caption
merge_octave_runs.py    checks and merges run directories into data/
data/                   the four merged JSONL files above
scripts/                the three drivers, Project.toml/Manifest.toml, slurm/
figures/                fig3_inverse_scaling.{pdf,png}, captions.txt
```
