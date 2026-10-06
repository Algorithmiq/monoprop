# Paper figures

The data figures of the monoprop paper, one self-contained directory each, named by the figure's
position in the paper. Each holds its data, the one script that draws it, the figure and a
caption computed from the plotted rows. Nothing here is a figure the paper does not use.

| paper | label | directory | script | data | status |
| --- | --- | --- | --- | --- | --- |
| Fig. 1 | `fig:mpi-scaling` | [`fig1_strong_weak_scaling/`](fig1_strong_weak_scaling) | `make_figure.py` | `data/SCALE-CELLS.tsv` | rebuilds from data |
| Fig. 2 | `fig:inverted-index` | — | TikZ in the paper repo (`figures/inverted-index-majorana.tex`) | — | not a data plot |
| Fig. 3 | `fig:inverse-scaling` | [`fig3_inverse_scaling/`](fig3_inverse_scaling) | `make_figure.py` | `data/*_octave_l7_deucalion.jsonl` | rebuilds from data |
| Fig. 4 | `fig:graph-gradient` | [`fig4_graph_gradient/`](fig4_graph_gradient) | not recovered | not recovered | **frozen PDF** |
| Fig. 5 | `fig:selection-scaling` | [`fig5_selection_scaling/`](fig5_selection_scaling) | `make_figure.py` | `data/selection_scaling.jsonl` | rebuilds from data |

(The appendix's `fig:pauli-query` and `fig:state-side` are also TikZ.)

## Rebuilding

Figs. 1, 3 and 5 need only matplotlib, no cluster and no monoprop build:

```bash
for d in fig1_strong_weak_scaling fig3_inverse_scaling fig5_selection_scaling; do
    (cd "$d" && python make_figure.py)
done
```

Output is byte-reproducible for a given matplotlib version (the PDF creation stamp is
suppressed, but the version is written into the file). Each build writes `figures/<stem>.pdf`,
`figures/<stem>.png` and `figures/captions.txt`; the stem is the figure's directory name. Copy
the PDFs into the paper repository's `figures/` under the same names.

## Conventions

- **Captions are generated.** Edit the script, never `captions.txt`; every number in a caption
  comes from the plotted rows.
- **Notation follows the paper:** `monoprop`, `ppvm`, `PauliPropagation.jl`; "monomials" for
  tracked terms in Fig. 1; `(d, g)` for the observable and generator degrees in Fig. 5.
- **Shapes, not absolute times.** Fig. 3 is a single node of Deucalion's x86 partition and is
  cited for an exponent in N, not for a time.
- **Provenance:** every row of Fig. 3 records `host`, `library_version` and `rounds`; the details
  and the rerun recipe are in each directory's README.

## Layout

```
README.md
ruff.toml                  lint scope for this package (extends the repository configuration)
fig1_strong_weak_scaling/  Hubbard strong and weak scaling, 128-8192 cores (+ scripts/ for the campaign)
fig3_inverse_scaling/      kicked-Ising per-gate cost, monoprop vs ppvm vs PauliPropagation.jl
fig4_graph_gradient/       surrogate-graph build and gradient cost (frozen PDF + provenance)
fig5_selection_scaling/    gate-selection cost against the mode count
```
