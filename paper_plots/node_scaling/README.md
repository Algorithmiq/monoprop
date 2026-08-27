# Paper figures — strong and weak scaling of monoprop on Deucalion x86

Publication-ready, self-contained figure package for the node-scaling study of monoprop's Hubbard
`propagate` operator, measured in the shipped default configuration on branch
`perf/linear-routing-on-wire` (tip `c2554554`, `_core.so` md5
`1157a5e2b421fb8bd18fc6d16fa39778`).

## Result

**The strong-scaling wall is set by the load a node carries, not by a core count, and it moves
right as the problem grows.** At 1.57e9 terms the curve bottoms out at 4096 cores (16.13 s) and
then *reverses* to 23.99 s at 8192; at 6.13e9 it is still flattening at 8192 (38.09 → 36.95 s);
at 24.42e9 it is still descending at 1.60× per doubling (471.0 → 235.2 → 127.1 → 79.6 s) with no
turn in sight. Parallel efficiency at 8192 cores is 14% / 38% / 74% for the three sizes, and the
departure from ideal moves from ~512 to ~1024 to ~2048 cores as the size grows.

Weak scaling says the same thing from the other side: efficiency at 8192 cores is **29.4%** at
96M terms/node, **59.7%** at 385M, and **83.9%** at 1529M. At roughly 1.5e9 terms per node
monoprop weak-scales close to ideal all the way to 8192 cores — the operator is not
latency-limited at scale, it is starved when the per-node load is small.

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
  partition count costs ~16% of wall). 208 reps across these 38 rungs were kept with **zero**
  gate failures (`data/SCALE-CELLS.tsv` carries 252 gate-clean rows in total; the rest belong to
  a curve not plotted here).

### Coverage

Six curves over 38 rungs (21 weak, 17 strong). The rung count is not simply 6 × 7 because the two
largest strong curves start at 2 and 8 nodes: at 8 ranks per node their problem does not fit in
one node's memory. Per-node footprint fits `GiB/node = 3.38 + 0.0633 × Mterms/node`
(≈68.0 bytes/term) across all rungs; the two memory-edge cells landed at 200.1 and 201.3 GiB/node
against a 242.0 GiB grant, so **no curve had to be shortened** for memory.

## Figures

All four: colour and marker encode problem size (strong) or load per node (weak) on a
light-to-dark **single-hue ordinal** ramp — the three curves differ in a *magnitude*, not in
identity — validated colourblind-safe against a white surface with
`scripts/palette/validate_palette.js --ordinal`. Marker shape carries the same information so
identity survives greyscale print. Deliberately minimal, conventional HPC style: no titles, no
`(a)`/`(b)` lettering, no secondary axes, no in-plot annotations. Drawn at final single-column
width (3.4 in) so the type is at true size in the paper. Vector `.pdf` plus a `.png` twin for
preview, in `figures/`.

**`fig-strong-time`** — wall time vs cores, log–log, at three fixed problem sizes. The dotted line
beside each curve is *that curve's own* ideal `1/N`, anchored at its first point. The y-axis is
clipped to the data, so the ideal guides leave the frame rather than wasting half the panel.

**`fig-strong-efficiency`** — parallel efficiency `t(N₀)N₀ / t(N)N` for the same runs. Each curve
is normalised to the narrowest run its problem fits in (128 / 256 / 1024 cores), so every curve
starts at 100% and the single ideal line is honest for all three. The price, stated in the
caption: the three baselines differ, so these curves describe how well each *size* scales from
its own starting point and are **not** absolute comparisons between sizes.

**`fig-weak-time`** — wall time vs cores holding the load per node constant. The dotted line
beside each curve is its own ideal, flat from its first point.

**`fig-weak-efficiency`** — weak efficiency `t(128 cores) / t(N)`. The headline plot: the larger
the load a node carries, the closer weak scaling stays to ideal.

LaTeX-ready captions for all four are in **`figures/captions.txt`**, generated from `captions.py`
by `make_paper_figures.py` — edit the Python, never the `.txt`. Every range and percentage in a
caption is computed from the rows the figure is drawn from, so a caption cannot describe rungs
that were not plotted.

## Reproduce

Figures and captions from the shipped data (matplotlib only; no cluster, no monoprop build):

```bash
./build.sh                                        # -> figures/*.pdf, *.png, captions.txt
# or explicitly, naming the interpreter:
PY=/projects/EEHPC-DEV-2026D08-260/venvs/plot296/bin/python ./build.sh    # on Deucalion
python make_paper_figures.py data/SCALE-CELLS.tsv figures
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
- **Efficiency baselines differ between the three strong curves** (see `fig-strong-efficiency`
  above) — that figure compares each size against itself, not against the others.
- **Load along a weak curve drifts** by a few per cent, because the size sequence steps by
  ×1.90–2.07 rather than exactly ×2. Every quantity is normalised on the *measured* term count,
  and the captions quote the measured range, not the nominal target.

## Files

```
build.sh                  one command: data -> figures + captions (no cluster)
make_paper_figures.py     the four figures; also writes figures/captions.txt
captions.py               one definition per caption, shared by the figures, the results
                          document and the artifact page, so the three cannot drift
data/
  SCALE-CELLS.tsv         gate-clean per-rep rows: the only input the figures read
  SEQUENCE.tsv            calibrated lower_atol -> term count sequence
scripts/                  campaign drivers (copies of the canonical files)
  calibseq.sh, submit_scale.sh, scaleladder.sh, scanlogs.sh, collatescale.py
  palette/                colourblind-safety validator for the ramp
figures/                  fig-strong-time, fig-strong-efficiency, fig-weak-time,
                          fig-weak-efficiency (each .pdf + .png); captions.txt
```

The prose results document (`RESULTS-296-scaling.md`), the superseded figure sets and the A/B
routing record this study deliberately excludes all live in the campaign harness on Deucalion
(`/projects/EEHPC-DEV-2026D08-260/harness/`), not in this repository.
