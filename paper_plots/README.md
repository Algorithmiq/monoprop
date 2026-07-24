# Paper figures — single-layer Pauli propagation: monoprop vs PauliPropagation.jl

Publication-ready, self-contained figure package for the single-layer scaling
comparison of monoprop's `PauliPropagator` (branch `perf/align-with-paper`) against
the reference Julia library
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

Both engines cover all cutoffs to N=1024. The far end is expensive on the Julia side
because of the cliff — the N=1024 c6 point alone took ~2.7 h of isolated-node time
(≈9 770 s) versus ~4 s for monoprop — so those points were run as dedicated jobs.
Timings were taken on **exclusive** Leonardo DCGP nodes (`--exclusive --mem=0`);
shared nodes were found to be overloaded and corrupt timing.

## Figures

All figures: colour = weight cutoff (Okabe–Ito, colourblind-safe, fixed order),
line style = engine (monoprop = solid line / filled marker, PauliPropagation.jl =
dashed line / open marker). Vector `.pdf` (for the paper) + `.png` twin (preview) in
`figures/`.

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

## Reproduce

Figures from the shipped data (matplotlib only; no monoprop build, no Julia):

```bash
python make_paper_figures.py data/monoprop_pauli.jsonl data/julia_pauli.jsonl \
  --outdir figures
```

Regenerate the raw data (`scripts/` holds copies of the canonical study drivers):

```bash
source ~/scripts/monoprop_leonardo_env.sh          # Spack GCC 15 toolchain
export monoprop_NUM_THREADS=1 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
JULIA=~/opt/julia/julia-1.10.11/bin/julia          # project pins PauliPropagation@0.7.3

# monoprop must be built with monoprop_MAX_NUM_MODES >= 1024 for the N=1024 end
# (arch-native on a compute node for timing). See the workspace CLAUDE.md.
for C in 2 4 6; do for N in $(seq 32 32 1024); do
  <monoprop-venv>/bin/python scripts/monoprop_single_layer.py --basis pauli \
    --num-qubits $N --cutoff $C --layers 5 --lower-atol 0 --rounds 1 \
    --out data/monoprop_pauli.jsonl
  $JULIA --project=scripts scripts/julia_pauli_single_layer.jl \
    --num-qubits $N --cutoff $C --layers 5 --lower-atol 0 --rounds 1 \
    --out data/julia_pauli.jsonl
done; done
```

On Leonardo the sweep is run as parallel single-threaded, arch-native, **exclusive**
compute-node jobs (see `~/scripts/sl_*.sbatch.sh`); memory/terms are contention-immune,
only timing needs an isolated node.

## Caveats

- **Timing** compares compiled C++ (monoprop) against JIT Julia; both measure
  steady-state propagation (Julia is JIT-warmed at the true qubit count, min of rounds).
- **Memory** metrics differ in construction: monoprop reports its C++ operator
  accounting (`operator_memory_bytes()`), Julia reports `Base.summarysize` of the
  `PauliSum`. Both are operator storage — the reported comparison is the *trend/slope*
  and the per-term staircase, not an absolute byte-for-byte equality.

## Files

```
make_paper_figures.py     publication figure script (PDF + PNG)
data/*.jsonl              merged, validated benchmark data (N=32..1024)
scripts/                  reproduction drivers (copies of the canonical study files)
  monoprop_single_layer.py, julia_pauli_single_layer.jl, Project.toml, Manifest.toml
figures/                  fig1_absolute_scaling, fig2_divergence_scaling,
                          fig3_per_term_memory  (each .pdf + .png)
```
