# Paper figures — single-layer Pauli propagation: monoprop vs PauliPropagation.jl

Publication-ready, self-contained figure package for the single-layer scaling
comparison of monoprop's `PauliPropagator` against the reference Julia library
[`PauliPropagation.jl`](https://github.com/SparqleSim/PauliPropagation.jl) v0.7.3.

## Result

On the **same operator**, monoprop's per-term cost stays **bounded** while
PauliPropagation.jl's grows with the system size, so monoprop's advantage in both
runtime and memory **widens as a power law in the qubit count N** — the *divergence*.
The advantage has two compounding parts: (i) a smooth **algorithmic** divergence
(the overhead ratio grows as `N^1.2–1.6` in time over the clean window N=128–512),
and (ii) past `N≈512` a **key-width performance cliff** in PauliPropagation.jl — its
packed-integer Pauli key outgrows the fast fixed-width `BitInteger` and falls onto a
slow wide-integer path — that makes the divergence turn super-linear (time overhead
exceeds 1000× by N=1024). monoprop, storing each term in `O(cutoff)` symplectic
support, shows neither the growth nor the cliff.

## Method

- **Circuit:** kicked-Ising chain, `LAYERS = 5` layers of `Rx(π/4)` on every qubit
  plus `Rzz(π/4)` on every bond of a 1D chain.
- **Observable:** extensive `Σᵢ Zᵢ` (Heisenberg picture) — spreads under the circuit,
  giving a term count that grows *linearly* in N (single shallow-layer light cone).
- **Truncation trick — `lower_atol = 0`:** truncation is then purely by weight cutoff,
  so **both engines keep the identical term set** (verified exact at every point:
  e.g. N=1024 gives 6 140 / 369 226 / 4 646 768 terms at cutoff 2 / 4 / 6 on *both*
  engines). This makes bytes/term, total memory, time and term count all fair,
  same-operator comparisons.
- **Cutoffs** {2, 4, 6}; **N = 32 … 1024** (step 32).
- **Single thread, single shard** — a like-for-like serial comparison:
  - monoprop: `monoprop_NUM_THREADS=1` (one oneTBB worker; recorded on every row);
    the serial, non-MPI build (`has_mpi = False`); single-shard confirmed at runtime
    because `operator_memory_bytes()` returned a value at every point (that accessor
    *raises* once the propagator shards).
  - Julia: `JULIA_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.

### Coverage

monoprop covers all three cutoffs at all 32 sizes (96 points). PauliPropagation.jl
covers cutoffs 2 and 4 in full, but **cutoff 6 is missing N=832…992** (90 points):
the far end is expensive on the Julia side because of the cliff — the N=1024 c6 point
alone took ~2.7 h of isolated-node time (≈9 770 s) versus ~4 s for monoprop — so those
points were run as dedicated jobs and five of them were never filled in. The gap sits
*inside* the key-width region the headline claim is about, so the figures draw a
**break** there rather than a segment joining N=800 to N=1024; the c6 Julia point at
N=1024 stands alone. Filling it needs Leonardo, not a workstation.

Timings were taken on **exclusive** Leonardo DCGP nodes (`--exclusive --mem=0`);
shared nodes were found to be overloaded and corrupt timing.

### Provenance limits of the shipped data

Two things the shipped `data/*.jsonl` cannot tell you, both worth knowing before
citing a number from them:

- **No provenance fields.** The records carry no host, no monoprop version and no
  Julia library version, so they cannot be audited the way `node_scaling/` gates every
  rep on the built `_core.so` md5. The hardware claims above come from this README, not
  from the data. Any *new* sweep should record them.
- **The `expectation` column of `monoprop_pauli.jsonl` predates a fix and does not
  match Julia's.** The monoprop driver passed twice Julia's angle on the Rzz layer (see
  the angle-convention comment in `scripts/monoprop_single_layer.py`), so the two
  engines evolved different circuits. With `lower_atol = 0` the retained term set is
  fixed by the gate *supports*, not the angles, which is why the term counts still
  agree exactly at every point and why the time, memory and bytes/term comparisons are
  unaffected — but do not cite the shipped monoprop `expectation` values, and do not
  cite agreement between the two `expectation` columns as validation of anything. The
  driver is now correct: re-measured at N=32/64 and cutoff 2/4 the two engines agree to
  12 decimal places.

## Figures

All figures: colour = weight cutoff (Okabe–Ito, colourblind-safe, fixed order),
line style = engine (monoprop = solid line / filled marker, PauliPropagation.jl =
dashed line / open marker). Vector `.pdf` (for the paper) + a 150-dpi `.png` twin
(preview only) in `figures/`. Both rebuild byte-identically from the shipped data.

**`fig1_absolute_scaling`** — Absolute time (left) and total operator memory (right)
vs N, log–log. Faint grey guides show slopes `N¹/N²/N³` for reference. The monoprop
curves track `N^1.2–1.6` (time) while the Julia curves track `N^2.5–2.8` over the clean
pre-cliff window N=128–512, then visibly steepen past N≈512 (the key-width cliff).

**`fig2_divergence_scaling`** — *The headline.* Julia ÷ monoprop for time (left) and
memory (right) vs N, log–log, against the monoprop = 1× baseline. Each curve is the
cost multiplier at the same operator; its log–log slope is the **divergence exponent**
(= the gap between the two absolute exponents in Fig. 1). The steep jump past N≈512 is
the super-linear key-width cliff regime, reaching >1000× in time by N=1024.

**`fig3_per_term_memory`** — The *mechanism*: memory per term vs N. monoprop sits in a
bounded band (`O(cutoff)` symplectic support); PauliPropagation.jl climbs a staircase
as its 2-bits/qubit packed key widens across word boundaries (shaded `64→…→2048`-bit).

**`fig4_scaling_and_divergence`** — Figs. 1 and 2 merged into one 2×2 block: columns
are time (left) / memory (right), the top row is the absolute scaling and the bottom
row the matching Julia ÷ monoprop overhead ratio, sharing the N axis so each column
reads "absolute cost, then its overhead". Use this as the single combined figure when
one panel-of-four is preferred over two separate figures.

### Suggested LaTeX captions

> **Fig. 1.** Single-layer Pauli propagation on a kicked-Ising chain: absolute
> wall-clock time (left) and total operator memory (right) versus qubit count $N$, on
> the identical operator for both engines ($\texttt{atol}=0$ keeps the term sets
> equal), single-threaded. Colour encodes the weight cutoff; solid/filled marks are
> monoprop, dashed/open marks are PauliPropagation.jl. Grey dashed lines are
> $N^1,N^2,N^3$ slope guides. The Julia curves steepen past the key-width cliff at
> $N\approx512$.

> **Fig. 2.** The monoprop advantage diverges as a power law in $N$. Each curve is the
> PauliPropagation.jl cost divided by monoprop's on the same operator; monoprop is the
> $1\times$ baseline. Time (left) and memory (right). The log–log slope is the
> divergence exponent, equal to the difference of the two absolute exponents in Fig. 1.
> Beyond $N\approx512$, PauliPropagation.jl's packed key exceeds its fast integer width
> and the divergence becomes super-linear (the steep jump).

> **Fig. 3.** Per-term memory. monoprop stores each term in $O(\text{cutoff})$
> symplectic support (bounded band); PauliPropagation.jl packs each Pauli into a
> BitInteger key at 2 bits/qubit, stepping up at each word boundary (shaded). This is
> the mechanism behind Figs. 1–2.

> **Fig. 4.** (Figs. 1 and 2 combined.) Columns are time (left) and memory (right);
> the top row is absolute cost versus $N$ for both engines ($N^1,N^2,N^3$ slope
> guides), the bottom row the PauliPropagation.jl / monoprop overhead ratio (monoprop
> $=1\times$), on a shared $N$ axis. Same operator ($\texttt{atol}=0$), single-threaded;
> colour = cutoff, solid/filled = monoprop, dashed/open = PauliPropagation.jl. The
> overhead grows as a power law and turns super-linear past the key-width cliff at
> $N\approx512$.

## Fig. 5 — the single-thread per-term figure, and a better test than the lattice

Figs. 1–4 measure the full-width kicked-Ising chain. That is the realistic workload, but it
is the wrong instrument for a claim about *per-term* cost: the term count K, the gate count
and the mode-space width N all grow together, so every exponent in those figures is a
mixture of three N-dependencies and none can be read off alone. In particular there is no
way to vary N at fixed K at all.

**The idle-spectator sweep fixes that.** `--active-window M` confines the whole model — every
gate and every observable term — to qubits `0…M-1`, and pads the register to N with qubits
nothing ever touches. K, the gate count, the term supports *and the expectation value* are
then identical at every N, so the only variable left is the width of the mode space and the
curve measures per-term cost in N and nothing else. The invariance is checked, not assumed:
`make_single_thread_figure.py` refuses to plot a sweep whose terms, gates or expectation
value move, since a drift in any of them means a gate reached a spectator qubit.

It also makes the comparison exact in a way Figs. 1–4 cannot. At `M=32` all three engines
report the **same term count and the same expectation value** at every N and every cutoff
(188 / 10 122 / 119 280 terms at cutoff 2 / 4 / 6), so dividing by K is fair by construction
and no engine can be accused of winning by keeping fewer terms.

### Result

Fitted log-log exponent of **time per term** against N, over N = 32…1024 at `M=32`, one
thread, three timed rounds per point, minimum kept:

| cutoff | K | monoprop | QuEra ppvm | PauliPropagation.jl |
|---|---|---|---|---|
| 2 | 188 | `N^+0.17` (1.7× over a 32× rise in N) | `N^+0.21` (2.3×) | `N^+1.78` (123×) |
| 4 | 10 122 | `N^+0.24` (2.2×) | `N^+0.41` (5.1×) | `N^+1.54` (74×) |
| 6 | 119 280 | `N^+0.17` (1.6×) | `N^+0.57` (8.3×) | `N^+1.48` (61×) |

The single-thread claim rests on different evidence per engine, and the figure script prints
which: monoprop and ppvm are driven from Python and record `cpu_seconds`, so theirs is a
measured CPU-to-wall ratio that never exceeds one core busy. The Julia driver records the
thread count instead — Base exposes no per-interval process CPU clock accurate enough to
form the ratio (`clock()` deltas on this host report 1.3–4.1 cores busy for a provably
serial loop at `Threads.nthreads() == 1`), and Julia's parallelism is explicit and opt-in.

monoprop's per-term cost is essentially flat in N: a term is an entropy-packed position list
whose width comes from the cutoff rather than from N, and the commute/anticommute decision
for 64 terms is one word operation against a transposed index. The other two carry a
2-bits-per-qubit packed key per term and pay for the width they declare. So the totals are
`K` against `K·N`, and per gate — where a gate reaches only the ~K/N of the operator that
touches its modes — `K/N` against `K`. Panel (b) shows that per-gate view on the full-width
model, where the operator really is spread over all N modes.

The residual `N^+0.15…0.24` for monoprop is not noise and should not be reported as flat:
the emit path still materialises the dense partner and folds every word for the hash, which
is `O(N/32)` work on the branching fraction.

### Reproduce

Needs all three engines on one host. `ppvm` installs from the git pin in
`packages/bench-third-party/pyproject.toml`; the Julia project pins PauliPropagation 0.7.3
and wants Julia 1.10.

```bash
export monoprop_NUM_THREADS=1 RAYON_NUM_THREADS=1 JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# (a) idle-spectator: fixed operator, widening register
for C in 2 4 6; do for N in $(seq 32 32 1024); do
  "$PYTHON" scripts/monoprop_single_layer.py --basis pauli --num-qubits "$N" \
    --active-window 32 --cutoff "$C" --layers 5 --lower-atol 0 --rounds 3 \
    --out data/monoprop_pauli_spectator.jsonl
  "$PPVM_PYTHON" scripts/ppvm_single_layer.py --num-qubits "$N" \
    --active-window 32 --cutoff "$C" --layers 5 --lower-atol 0 --rounds 3 \
    --out data/ppvm_pauli_spectator.jsonl
  "$JULIA" --project=scripts scripts/julia_pauli_single_layer.jl --num-qubits "$N" \
    --active-window 32 --cutoff "$C" --layers 5 --lower-atol 0 --rounds 3 \
    --out data/julia_pauli_spectator.jsonl
done; done

# (b) full-width, all three engines: drop --active-window, N capped where Julia stays
# affordable on one thread (its cost is ~N^2.5-2.8 here)
for C in 2 4 6; do for N in $(seq 32 32 512); do ... ; done; done

python make_single_thread_figure.py \
  --spectator data/{monoprop,ppvm,julia}_pauli_spectator.jsonl \
  --lattice   data/{monoprop,ppvm,julia}_pauli_lattice.jsonl \
  --outdir figures

# Fig. 6 -- the octave sweep: cutoff 6 only, N = 32..1024 in powers of two, 10 rounds for
# monoprop and 1 for the reference engines. ~56 min, essentially all of it the single
# PauliPropagation.jl N=1024 point (5304.7 s).
for N in 32 64 128 256 512 1024; do
  "$PYTHON"      scripts/monoprop_single_layer.py --basis pauli --num-qubits "$N" \
    --cutoff 6 --layers 5 --lower-atol 0 --rounds 10 --out data/monoprop_pauli_octave.jsonl
  "$PPVM_PYTHON" scripts/ppvm_single_layer.py --num-qubits "$N" \
    --cutoff 6 --layers 5 --lower-atol 0 --rounds 1  --out data/ppvm_pauli_octave.jsonl
  "$JULIA" --project=scripts scripts/julia_pauli_single_layer.jl --num-qubits "$N" \
    --cutoff 6 --layers 5 --lower-atol 0 --rounds 1  --out data/julia_pauli_octave.jsonl
done

# Both panel shapes. --spectator and --cutoff-evidence draw no panel: they only supply the
# caption's Fig. 5a cross-reference and its cutoff-independence table. Add
# `--layout column` or `--layout page` for just one shape.
# Add `--layout column|page` or `--fit clean|cliff` to emit just one variant.
python make_inverse_scaling_figure.py \
  --lattice        data/{monoprop,ppvm,julia}_pauli_octave.jsonl \
  --spectator      data/{monoprop,ppvm,julia}_pauli_spectator.jsonl \
  --cutoff-evidence data/{monoprop,ppvm,julia}_pauli_lattice.jsonl \
  --outdir figures
```

### Fig. 6 — the same data on the axis where the claim is visible

One panel, three curves, nothing else — in two shapes from one code path, so they can
never disagree:

| `--layout` | file | saved size | intended slot |
|---|---|---|---|
| `column` | `fig6_inverse_scaling` | 3.40 × 3.45 in | one journal column (`\columnwidth`), legend below the axes |
| `page` | `fig6_inverse_scaling_wide` | 6.90 × 4.15 in | full-width banner at the top of a page (`figure*`, `\textwidth`), all three engines on one horizontal line beneath the axes |

The column legend is stacked because three entries each carrying an exponent need roughly
twice 3.4 in on one line. Both `figsize` values are calibrated rather than chosen, and in
opposite directions: with the legend centred under the axes the tight bbox *trims* the
unused right margin, so the saved width comes out ~0.9× of `figsize`. Re-measure the saved
`MediaBox` if the legend text or the tick labels change width. Neither shape is a
decade-square (1.51 decades in N against ~2.1 in the plotted time), so the rendered
*angle* of a true `-1` slope is a property of the frame — steeper than 45° in the
column, shallower in the banner. That is
what the `1/N` guide is for: the eye reads monoprop against the guide, never against the
frame, so the claim survives the reshaping.

It plots the wall-clock cost of
**one gate acting on one term**, in nanoseconds, against N = 32…1024 in powers of two. On
that axis monoprop falls as `N^-0.88` (0.571 → 0.0347 ns, a 16.4× reduction over a 32×
wider system) while both reference engines stay flat or rise (`N^+0.59` for ppvm, `N^+0.14`
for PauliPropagation.jl), which is the `K/N` against `K` statement made directly rather
than inferred from a ratio of exponents. Per layer of `2N-1` gates that reads as `K`
against `K·N`. Every sentence that used to sit on the canvas is now in the generated
`figures/fig6_caption.txt`; the panel itself carries only the fitted exponents and one
grey slope ruler.

That ruler is the ideal `1/N`, anchored *through* monoprop's first point rather than offset
from it, so monoprop starts on its guide and the drift away from it is the `N^-0.88` the
legend reports. The `N^0` and `N^+1` counterparts — what the reference engines ought to pay
per gate, `K` and `K·N` — are deliberately not drawn: their exponents are in the legend, and
three rulers among three curves read as furniture rather than as a reference.

**Two fit windows, distinguished by filename.** `--fit` selects:

| `--fit` | suffix | window | exponents | axis |
|---|---|---|---|---|
| `clean` | — | N ≤ 512 | `N^-0.88` / `N^+0.59` / `N^+0.14` | scaled to the fitted decade (2.1); N=1024 faded, PauliPropagation.jl's clipped and named at 112 ns |
| `cliff` | `_cliff` | all N | `N^-0.83` / `N^+0.57` / `N^+0.87` | opened to hold the post-cliff excursion (3.8 decades) |

Both are emitted by default, so there are four PDFs (two layouts × two fit modes). Either
way one window covers all three engines, so the three exponents are always like-for-like.
The panel carries no prose, so **nothing on the canvas says which fit you are looking at** —
only the `_cliff` suffix and the caption do. Keep the filenames straight when placing one in
the paper: `N^+0.14` and `N^+0.87` for PauliPropagation.jl are the same figure otherwise.

Beyond N=512 the reference engines cross a packed-key width boundary —
PauliPropagation.jl's is measured at N=576→608 (8.7× in one step) and ppvm has one near
N=1024 (2.14×). A power law fitted across a discontinuity measures where the step falls,
not per-term cost in N, which is what the `clean` window exists to avoid; the inflation
from `N^+0.14` to `N^+0.87` *is* the cliff. monoprop's exponent barely moves (`-0.88` to
`-0.83`), having no such boundary to cross, so `clean` is the conservative reading — it
gives up the widest part of monoprop's lead, 3210× over PauliPropagation.jl at N=1024.

**Why the normalisation is fair.** The y axis divides wall time by the gate count (∝ N) and
by the term count K. Neither divisor is engine-specific, and the script *checks* that rather
than assuming it: `check_grid`, `check_same_workload` and `check_single_thread` refuse to
build the figure unless all three engines cover the identical (cutoff, N) grid and report
the same term count, the same gate count and an expectation value agreeing to 1e-9 at every
point (the measured worst case is 1.6e-11 on the octave sweep). A shared divisor cannot
manufacture a difference between the curves, only reveal one.

**Timed repetitions differ by engine, and the rows now record it** (`rounds`): monoprop 10,
ppvm 1, PauliPropagation.jl 1. monoprop's points are sub-second, where timer granularity is
proportionally worst; the reference engines' N=1024 points run for minutes to an hour and
would lose more to thermal drift within one round than they could gain from a second.

**The divisor is not the effect.** A falling curve invites the objection that the gate count
is itself proportional to N. Stated with no gate division at all, the same rows read *a
monoprop layer of 1023 gates costs 1.42× what a layer of 63 gates costs* (36.8 → 52.1 ms
per 10⁶ terms) — 16.2× the gates for 1.42× the time. The identical divisor applied to the
other two engines leaves them flat or rising.

**Nor is the cutoff.** The axis is an absolute time, so nothing is extrapolated to a term
count that was not measured and the choice of arm is not load-bearing. Fitted per-gate,
per-term exponent at every cutoff on this sweep:

| cutoff | K (N=32→512) | monoprop | QuEra ppvm | PauliPropagation.jl |
|---|---|---|---|---|
| 2 | 188 → 3,068 | `N^-0.96` | `N^-0.06` | `N^+0.38` |
| 4 | 10,122 → 183,882 | `N^-0.81` | `N^+0.31` | `N^+0.31` |
| 6 | 119,280 → 2,310,000 | `N^-0.86` | `N^+0.58` | `N^+0.23` |

monoprop falls at all three and neither reference engine is ever below `N^-0.06`. Cutoff 6
is drawn for continuity with Figs. 1–5.

The mechanism is selectivity rather than batching: the operator is stored transposed, one
column per bit position, and a column below 1/64 density is held as an ascending set-row
list instead of a full-height bit-vector (`cpp/monoprop/detail/operator/InvertedIndex.h`),
with `combine_columns_block` narrowing even a dense column to a word range. A gate on qubit
`i` therefore costs work proportional to the terms that actually touch qubit `i` — at a
fixed weight cutoff a `~w/N` fraction of the operator — instead of a scan over all K terms.

**Where the sweep is generous to the reference engines, and where it is not.** Both
reference engines' N=1024 points sit outside every fit, which discards monoprop's widest
lead (3210× over PauliPropagation.jl, 78× over ppvm there). They also ran for minutes to an
hour on a **fanless laptop** and throttled, while monoprop's N=1024 point finishes in 1.65 s
and does not — biasing their apparent cost upward, against them. `num_terms` is the final
term count while K grows through the five layers, so the absolute ns/term understates
per-term cost — identically for all three engines. And at N=32 ppvm is the fastest of the
three (0.459 ns against monoprop's 0.571), so the ordering is earned over the sweep and not
assumed at the origin. N=1024 is the ceiling, not a choice: `monoprop_MAX_NUM_MODES`
defaults to 1024 with no headroom, so a wider sweep needs a rebuild.

The Fig. 5 and 6 data was taken on a 10-core workstation, not on Leonardo, and the records
say so — `host` and `library_version` on every row, `cpu_seconds` and `busy_cores` on the
two Python engines. It is a *shape* measurement (an exponent in N), which is what makes a
workstation acceptable here where it would not be for Figs. 1–4's absolute times; the two
datasets are kept in separate files and never plotted in the same panel.

## Reproduce

Figures from the shipped data (matplotlib only; no monoprop build, no Julia):

```bash
python make_paper_figures.py data/monoprop_pauli.jsonl data/julia_pauli.jsonl \
  --outdir figures
```

Regenerate the raw data (`scripts/` holds copies of the canonical study drivers):

```bash
# Point these at your own toolchain. On Leonardo, PYTHON is a venv holding an
# arch-native monoprop built on a compute node and JULIA is a 1.10.11 install; the
# Manifest pins PauliPropagation@0.7.3.
PYTHON=${PYTHON:-python}
JULIA=${JULIA:-julia}

export monoprop_NUM_THREADS=1 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# monoprop must be built with monoprop_MAX_NUM_MODES >= 1024 for the N=1024 end; that
# is now the default (see the repository CMakeLists.txt), with no headroom above it.
for C in 2 4 6; do for N in $(seq 32 32 1024); do
  "$PYTHON" scripts/monoprop_single_layer.py --basis pauli \
    --num-qubits "$N" --cutoff "$C" --layers 5 --lower-atol 0 --rounds 1 \
    --out data/monoprop_pauli.jsonl
  "$JULIA" --project=scripts scripts/julia_pauli_single_layer.jl \
    --num-qubits "$N" --cutoff "$C" --layers 5 --lower-atol 0 --rounds 1 \
    --out data/julia_pauli.jsonl
done; done
```

Terms and memory are contention-immune; only timing needs an isolated node. On
Leonardo the sweep is run as parallel single-threaded, arch-native, **exclusive**
compute-node jobs.

## Caveats

- **Timing** compares compiled C++ (monoprop) against JIT Julia; both measure
  steady-state propagation (Julia is JIT-warmed at the true qubit count, min of rounds).
- **Memory** metrics differ in construction: monoprop reports its C++ operator
  accounting (`operator_memory_bytes()`), Julia reports `Base.summarysize` of the
  `PauliSum`. Both are operator storage — the reported comparison is the *trend/slope*
  and the per-term staircase, not an absolute byte-for-byte equality.

## Files

```
make_paper_figures.py     Figs. 1-4 (PDF + PNG)
make_single_thread_figure.py  Fig. 5, the supporting per-term view (all three cutoffs)
make_inverse_scaling_figure.py Fig. 6, the headline single-panel per-gate figure
                          (--layout column | page, --fit clean | cliff; all four
                          combinations by default)
ruff.toml                 lint scope for this package (extends the repository config)
data/*_pauli.jsonl        Figs. 1-4, Leonardo, N=32..1024
data/*_spectator.jsonl    Fig. 5a, idle-spectator sweep at M=32, three engines
                          (Fig. 6 uses it only for caption cross-reference numbers)
data/*_lattice.jsonl      Fig. 5b, full-width step-32 sweep, three engines, three
                          cutoffs (Fig. 6 uses it for its cutoff-independence table)
data/*_octave.jsonl       Fig. 6, full-width octave sweep, cutoff 6, N=32..1024
scripts/                  reproduction drivers (copies of the canonical study files)
  monoprop_single_layer.py, julia_pauli_single_layer.jl, ppvm_single_layer.py,
  Project.toml, Manifest.toml
figures/                  fig1_absolute_scaling, fig2_divergence_scaling,
                          fig3_per_term_memory, fig4_scaling_and_divergence,
                          fig5_single_thread_per_term, fig6_inverse_scaling{,_cliff},
                          fig6_inverse_scaling_wide{,_cliff}
                          (each .pdf + .png); captions.txt (LaTeX-ready captions),
                          fig5_caption.txt and fig6_caption.txt (generated)
```
