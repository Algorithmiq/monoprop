# Paper figures — single-layer Pauli propagation: monoprop vs PauliPropagation.jl

Publication-ready, self-contained figure package for the single-layer scaling
comparison of monoprop's `PauliPropagator` against the reference Julia library
[`PauliPropagation.jl`](https://github.com/SparqleSim/PauliPropagation.jl).

The shipped data spans two library versions and the rows say which: Figs. 1-5 were taken
against v0.7.3 on its dictionary-backed `PauliSum`, Fig. 6 against **v0.8.2** on the
vectorised `VectorPauliSum`. `scripts/` pins v0.8.2 and the driver refuses to run on
anything else, so re-running Figs. 1-5 means checking out the commit that pinned v0.7.3.

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
`packages/bench-third-party/pyproject.toml`; the Julia project pins PauliPropagation 0.8.2
and wants Julia 1.10. The driver always propagates the vectorised `VectorPauliSum`; the rows
of Fig. 5 shipped here predate it and were taken on the dictionary-backed `PauliSum` at
v0.7.3.

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
    --cutoff 6 --layers 5 --lower-atol 0 --rounds 1 \
    --out data/julia_pauli_octave_v082.jsonl
done

# Both panel shapes. --spectator and --cutoff-evidence draw no panel: they only supply the
# caption's Fig. 5a cross-reference and its cutoff-independence table. Add
# `--layout column` or `--layout page` for just one shape.
# Add `--layout column` or `--layout page` to emit just one shape.
python make_inverse_scaling_figure.py \
  --lattice        data/{monoprop,ppvm}_pauli_octave.jsonl data/julia_pauli_octave_v082.jsonl \
  --spectator      data/{monoprop,ppvm}_pauli_spectator.jsonl data/julia_pauli_spectator_v082.jsonl \
  --cutoff-evidence data/{monoprop,ppvm}_pauli_lattice.jsonl data/julia_pauli_lattice_v082.jsonl \
  --outdir figures
```

**What v0.8.2 and the vectorised backend changed, and why it is not what one would guess.**
Re-measuring the PauliPropagation.jl arm on v0.8.2's `VectorPauliSum` makes it look *worse*
on this axis, not better: `N^+0.87` → `N^+1.12` over the drawn range, and `N^+0.14` →
`N^+0.32` in the clean N ≤ 512 window. Absolute times did improve over most of the sweep —
1.91× at N=32, 1.87× at N=128, 1.20× at N=512 — but the gain decays monotonically and
inverts at the top, where v0.8.2 is 0.83× (6385 s against 5305 s at N=1024). Speeding up
small N more than large N steepens a slope in N, which is why the exponent worsens even in
the window where every point got faster; the effect therefore does not rest on the single
N=1024 point or on laptop throttling. Operator memory fell 1.8-2.3× at every point
(1170 MB against 2120 MB at N=1024), consistent with dropping the dictionary's hash-table
slack — that quantity is not plotted here, and it is why Figs. 1-3 were left at v0.7.3
rather than re-measured alongside.

### Fig. 6 — the same data on the axis where the claim is visible

One panel, three curves, nothing else — in two shapes from one code path, so they can
never disagree:

| `--layout` | file | intended slot |
|---|---|---|
| `column` | `fig6_inverse_scaling` | one journal column (`\columnwidth`), legend inside the lower-left corner |
| `page` | `fig6_inverse_scaling_wide` | full-width banner at the top of a page (`figure*`, `\textwidth`), all three engines on one horizontal line beneath the axes |

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
that axis monoprop falls as `N^-0.83` (0.571 → 0.0347 ns, a 16.4× reduction over a 32×
wider system) while both reference engines stay flat or rise (`N^+0.57` for ppvm, `N^+1.12`
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

**The panel fits every point it draws**, N = 32…1024, so the drawn range and the fitted
range are the same and there is no window a reader cannot see. The *view* is cropped
though: the y axis stops at 10 ns per gate per term (`Y_TOP_NS`), because letting
PauliPropagation.jl's 134 ns post-cliff point set the ceiling stretches the axis to 3.8
decades and squashes the decade the `1/N` lives in. Its line runs off the top instead; the
point is still in the fit and in the caption. The legend carries engine
names only, so the exponents themselves live in `figures/fig6_caption.txt` and in the
build's stdout — those are the only record of them.

That choice has a cost and it lands on the reference engines. Beyond N=512 both cross a
packed-key width boundary — PauliPropagation.jl's is measured at N=576→608 (8.5× in one
step at v0.8.2, 8.7× at v0.7.3) and ppvm has one near N=1024 (2.14×) — and a power law
fitted across a discontinuity
partly measures where the step falls rather than per-term cost in N. Restricted to the clean
N ≤ 512 window the same rows give monoprop `N^-0.88`, ppvm `N^+0.59` and
PauliPropagation.jl `N^+0.32`. PauliPropagation.jl is the one that moves, `+0.32` → `+1.12`:
that difference *is* the cliff, and it flatters monoprop, so **`N^+0.32` is the conservative
number to quote for per-term cost** and `N^+1.12` is per-term cost plus a discontinuity.
monoprop's own exponent barely moves (`-0.88` → `-0.83`), having no such boundary to cross.
The caption reports both windows; `FIT_ALL_POINTS` in the script selects which one the panel
draws.

The mechanism is selectivity rather than batching: the operator is stored transposed, one
column per bit position, and a column below 1/64 density is held as an ascending set-row
list instead of a full-height bit-vector (`cpp/monoprop/detail/operator/InvertedIndex.h`),
with `combine_columns_block` narrowing even a dense column to a word range. A gate on qubit
`i` therefore costs work proportional to the terms that actually touch qubit `i` — at a
fixed weight cutoff a `~w/N` fraction of the operator — instead of a scan over all K terms.

**Where the sweep is generous to the reference engines, and where it is not.** Both
reference engines' N=1024 points sit outside every fit, which discards monoprop's widest
lead (3864× over PauliPropagation.jl, 78× over ppvm there). They also ran for minutes to an
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
# arch-native monoprop built on a compute node and JULIA is a 1.10.11 install. The
# Manifest now pins PauliPropagation@0.8.2 and the driver propagates VectorPauliSum, so
# these commands no longer reproduce the shipped Figs. 1-4 rows, which were taken at
# v0.7.3 on the dictionary-backed PauliSum; they re-measure at v0.8.2.
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
                          (--layout column | page, both by default)
ruff.toml                 lint scope for this package (extends the repository config)
data/*_pauli.jsonl        Figs. 1-4, Leonardo, N=32..1024
data/*_spectator.jsonl    Fig. 5a, idle-spectator sweep at M=32, three engines
                          (Fig. 6 uses it only for caption cross-reference numbers)
data/*_lattice.jsonl      Fig. 5b, full-width step-32 sweep, three engines, three
                          cutoffs (Fig. 6 uses it for its cutoff-independence table)
data/*_octave.jsonl       Fig. 6, full-width octave sweep, cutoff 6, N=32..1024
data/julia_*_v082.jsonl   Fig. 6's PauliPropagation.jl arm re-measured at v0.8.2 on the
                          vectorised VectorPauliSum backend: octave (drawn), lattice and
                          spectator (caption only), plus cliff, the N=576/608 probe that
                          dates the key-width step and is fed to no figure
scripts/                  reproduction drivers (copies of the canonical study files)
  monoprop_single_layer.py, julia_pauli_single_layer.jl, ppvm_single_layer.py,
  Project.toml, Manifest.toml (PauliPropagation 0.8.2)
figures/                  fig1_absolute_scaling, fig2_divergence_scaling,
                          fig3_per_term_memory, fig4_scaling_and_divergence,
                          fig5_single_thread_per_term, fig6_inverse_scaling,
                          fig6_inverse_scaling_wide
                          (each .pdf + .png); captions.txt (LaTeX-ready captions),
                          fig5_caption.txt and fig6_caption.txt (generated)
```
