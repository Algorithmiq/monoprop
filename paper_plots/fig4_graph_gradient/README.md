# Paper Fig. 4 — building and evaluating the surrogate graph (frozen)

`figures/fig4_graph_gradient.pdf` is the paper's figure (`fig:graph-gradient`), copied byte for
byte from the paper repository, where it was committed in `dc7c4fc` as `fig7_graph_gradient.pdf`.

**There is no generator and no data here.** That commit names `make_graph_gradient_figure.py` in
this package as the source, but neither the script nor its rows are on any branch of this
repository, so the figure cannot be regenerated or cross-checked from files. Treat it as an
imported artefact.

## What it shows

Single-threaded time (panel a) and peak memory above the interpreter's baseline (panel b) of
building the surrogate graph, evaluating the expectation value, and evaluating the parameter
gradient, each divided by plain propagation of the same circuit. The circuit is the 127-qubit IBM
Eagle heavy-hex kicked-Ising model with 4 to 12 layers (`P = 1084 … 3252` parameters), Pauli-weight
cutoff 6, no coefficient truncation.

## Numbers the paper text quotes from it

A rerun has to reproduce these:

- building the graph costs 1.02–1.12× propagation;
- an expectation value takes about 0.1× (0.04–0.08×) of propagation, a gradient about 0.2×
  (0.12–0.24×), so the graph is faster than re-propagation from the second evaluation;
- peak memory stays within 1.02–1.56× of propagation, and the gradient evaluation reaches 2.92× at
  12 layers.

To regenerate: write a benchmark over `benches/bench_models.py` for the heavy-hex circuit and a
drawing script in the style of `../fig3_inverse_scaling/make_figure.py`, then replace the PDF.
