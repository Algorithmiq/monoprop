# Paper Fig. 1 — strong and weak scaling of monoprop on Deucalion x86

Publication-ready, self-contained figure package for the node-scaling study of monoprop's Hubbard
`propagate` operator, measured in the shipped default configuration (`routing=default`) on the
`w2-bench` tree at commit `a204dc13`, installed `_core.so` md5
`ce208a744cd5fa2225a6a458b789803b` — the identity every rep re-checks before its time is kept,
and the one pinned in `scripts/scaleladder.sh` and `scripts/collatescale.py`. The run directories
are named `*-hubbard-296ft-*`.

## Result

> These numbers are read off `data/SCALE-CELLS.tsv` with the same loader the figures use.
> The percentages match `figures/captions.txt`, which `captions.py` computes from the plotted
> rows; if this section and that file ever disagree, that file is right.

**The strong-scaling wall is set by the load a node carries, not by a core count, and it moves
right as the problem grows.** Every strong curve is monotone — none turns back up. At 1.57e9
terms the time falls 212.1 → 6.9 s from 128 to 8192 cores, but the last doubling buys almost
nothing (7.4 → 6.9 s): the curve has flattened, not reversed. At 6.13e9 terms it is still
halving cleanly to 4096 and gives up only the last step (28.4 → 17.0 s), and at 24.42e9 it is
still descending near-ideally at 8192 (453.8 → 225.6 → 110.2 → 57.3 s) with no turn in sight.
Parallel efficiency at 8192 cores is **48% / 81% / 99%** for the three sizes, and the departure
from ideal moves right with the problem: the 1.57e9 curve holds ≥97% to 2048 cores and breaks at
4096 (89%), the 6.13e9 curve holds ≥96% to 4096, and the 24.42e9 curve has not departed by 8192.

What that break tracks is the per-node load, not the core count. At 8192 cores the three sizes
leave each node holding 24.5M, 95.7M and 381.6M terms respectively — and that is the order of
the efficiencies.

Weak scaling says the same thing from the other side: efficiency at 8192 cores is **61%** at
96M terms/node, **81%** at 385M, and **94%** at 1529M. At roughly 1.5e9 terms per node monoprop
weak-scales close to ideal all the way to 8192 cores — the operator is not latency-limited at
scale, it is starved when the per-node load is small. (The weak percentages are measured against
a single-node baseline across a 64× span, so they are not directly comparable to the strong ones
at the same per-node load; each family is internally consistent.)

## Method

- **Operator:** Hubbard `propagate`, at a fixed weight cutoff of **10**.
- **Size knob — `lower_atol`, not the cutoff:** the cutoff is already saturated at these
  tolerances (at `atol=1.25e-05`, cutoffs 10/12/14 give 96.98M/105.79M/106.82M terms), so all
  growth comes from the tolerance and every curve stays inside one operator family.
- **One shared size sequence.** Eleven sizes stepping by ×1.90–2.07 (96.98M … 94.68e9 terms);
  *both* scaling families are built from it, so a cell shared between a strong and a weak curve
  is **exactly** the same problem. Nine such shared cells were measured in separate allocations
  and agree to ≤1.24%, most to ≤0.2%. Every size above the 6.1e9 rung was calibrated by
  measurement, not extrapolated: the local exponent `d(log terms)/d(-log atol)` drifts from 1.83
  to 1.40 across the range, so a power law fitted at one end does not predict the other.
- **Layout:** 8 MPI ranks per node × 16 partitions per rank
  (`--cpu-bind=cores --distribution=block:block`), so `R = 8N` ranks and `P = 128N` partitions at
  `N` nodes — one core per partition on 128-core nodes.
- **N = 1 … 64 nodes** (128 … 8192 cores), **five reps per rung**, median reported.
- **Gating.** Every rep re-checks the installed `_core.so` md5, the environment, and the
  resulting term count before its time is kept; `MALLOC_ARENA_MAX` is left unset and that is
  verified *from inside the python process*, not from the submitting shell (setting it to the
  partition count costs ~16% of wall). 197 reps across these 38 rungs were kept with **zero**
  gate failures (`data/SCALE-CELLS.tsv` carries 239 gate-clean reps over 45 rungs in total; the
  other 42 reps over 7 rungs are the `strong_s5` curve, which is not plotted here).

### Coverage

Six curves over 38 rungs (21 weak, 17 strong). The rung count is not simply 6 × 7 because the two
largest strong curves start at 2 and 8 nodes: at 8 ranks per node their problem does not fit in
one node's memory. Per-node footprint fits `GiB/node = 3.38 + 0.0633 × Mterms/node`
(≈68.0 bytes/term) across all rungs; the two memory-edge cells landed at 200.1 and 201.3 GiB/node
against a 242.0 GiB grant, so **no curve had to be shortened** for memory.

## Figure

**`fig1_strong_weak_scaling`** (paper Fig. 1, `fig:mpi-scaling`) is one page-wide 2×2 float. Columns
are the scaling family (strong left, weak right), rows the quantity (wall time above, parallel
efficiency below). Strong scaling fixes the number of tracked monomials (1.6B, 6.1B, 24.4B); weak
scaling fixes the monomials per node (96M, 385M, 1.5B).

- **Time panels:** log–log wall time against cores. The dotted line beside each curve is *that
  curve's own* ideal (`1/N` for strong, flat for weak), anchored at its first point. The y axis is
  clipped to the data, so ideal guides leave the frame rather than wasting half the panel.
- **Strong efficiency:** `t(N₀)N₀ / t(N)N`, each curve normalised to the narrowest run its problem
  fits in (128 / 256 / 1024 cores), so every curve starts at 100% and the single ideal line is
  honest for all three. The three baselines differ, so these curves describe how well each *size*
  scales from its own starting point and are **not** absolute comparisons between sizes.
- **Weak efficiency:** `t(128 cores) / t(N)`. The larger the load a node carries, the closer weak
  scaling stays to ideal.

Colour and marker encode problem size (strong) or load per node (weak) on a light-to-dark
**single-hue ordinal** ramp — the three curves differ in a *magnitude*, not in identity —
validated colourblind-safe against a white surface with `scripts/palette/validate_palette.js
--ordinal`. Marker shape carries the same information so identity survives greyscale print. No
titles, no in-plot annotations. The four panels share one x axis, **cores below and nodes above**
(they differ by exactly the 128 cores per node), drawn once per edge. Each row's key is stacked in
the lower-left corner of its efficiency panel, empty by construction, and serves the wall-time
panel above it. Type is DejaVu Serif, drawn at final size (**7.0 × 3.9 in**, a landscape banner).
Vector `.pdf` plus a `.png` twin for preview, in `figures/`.

The LaTeX-ready caption is **`figures/captions.txt`**, generated from `captions.py` by
`make_figure.py` — edit the Python, never the `.txt`. Every range and percentage in it is computed
from the rows the figure is drawn from, so it cannot describe rungs that were not plotted. The
legend and caption say *monomials*, as the paper does.

## Reproduce

Figures and captions from the shipped data (matplotlib only; no cluster, no monoprop build):

```bash
./build.sh                                        # -> figures/fig1_*.pdf, .png, captions.txt
# or explicitly:
python make_figure.py data/SCALE-CELLS.tsv figures
```

Output is **byte-reproducible** — the PDF creation stamp is suppressed — so a regenerated figure
that differs from the shipped one means the *data* changed, not the clock.

Regenerate the raw data (`scripts/` holds copies of the canonical campaign drivers; these need
Deucalion and a built worktree):

```bash
scripts/calibseq.sh                # calibrate lower_atol -> term count, writes SEQUENCE.tsv
scripts/submit_scale.sh            # submit the rungs; walltime/partition come from the ladder's
                                   # own rung table, so a submit line cannot disagree with it
scripts/scanlogs.sh                # grep -a the job logs (a NUL byte makes plain grep silently
                                   # print nothing, which reads as a clean job)
scripts/collatescale.py            # run dirs -> SCALE-CELLS.tsv, gate-checked
```

`scaleladder.sh` is the per-rung driver `submit_scale.sh` invokes; it must be submitted from
inside the worktree, because the job scripts `cd` to `SLURM_SUBMIT_DIR` and ignore `--chdir`.

## Caveats

- **`weak_1569m` at 64 nodes** sits 12.8 GiB/node *below* the memory fit, reproducibly on all five
  reps. The cause is not established; it is the one residual outside the −2.9…+4.6 GiB/node band
  that holds for the other 37 rungs.
- **Efficiency baselines differ between the three strong curves** (see above) — the panel
  compares each size against itself, not against the others.
- **Load along a weak curve drifts** by a few per cent, because the size sequence steps by
  ×1.90–2.07 rather than exactly ×2. Every quantity is normalised on the *measured* term count,
  and the captions quote the measured range, not the nominal target.

## Files

```
build.sh                  one command: data -> figures + captions (no cluster)
make_figure.py            the figure; also writes figures/captions.txt
captions.py               one definition per caption, shared by the figures, the results
                          document and the artifact page, so the three cannot drift
data/
  SCALE-CELLS.tsv         gate-clean per-rep rows: the only input the figures read
  SEQUENCE.tsv            calibrated lower_atol -> term count sequence
scripts/                  campaign drivers (copies of the canonical files)
  calibseq.sh, submit_scale.sh, scaleladder.sh, scanlogs.sh, collatescale.py
  palette/                colourblind-safety validator for the ramp
figures/                  fig1_strong_weak_scaling (.pdf + .png); captions.txt
```

The `retired/` and `ab-supporting/` figure sets and the `harness/RESULTS-296-*.md` prose
documents this file used to point at were never part of the package and are not in the
repository; the Result section above is the record.
