# monoprop benchmarks

The `monoprop` repository includes a pytest suite measuring the **time** and
**peak resident memory (RSS)** of monoprop's core operations.
The suite is separated from the test suite and can be run with `just bench`.
See below for more detailed instructions.

See the [Benchmarks](../docs/content/docs/benchmarks.mdx) section of the
documentation for detailed instructions.

## What lives where

This directory holds only monoprop's own benchmarks — `conftest.py` (the fixtures
and the results schema), `bench_random.py`, `bench_models.py`, and `results/`.

Benchmark names are the key [Bencher](https://bencher.dev/) stores history under,
so they stay here rather than moving with a library release.

Everything reusable is in the `monoprop-bench-tools` package
([`../packages/monoprop-bench-tools`](../packages/monoprop-bench-tools)): the
memory instrumentation, the model builders, and the two renderers that turn a
run's artifacts into `REPORT.md` and Bencher Metric Format JSON.

Cross-engine comparisons against other propagation libraries live in
[`../packages/bench-third-party`](../packages/bench-third-party), a standalone uv
project with its own lockfile.

## Fixed-model sizing

At the default `--hubbard-lower-atol 1e-4`, both nominal Hubbard size axes are
saturated -- `--hubbard-cutoff` is flat from 10, `--hubbard-num-sites` is flat
from 60 -- so `--hubbard-lower-atol` is the only axis that reaches 100M terms.
Measured at `cutoff=10`: `lower_atol` 1e-4 / 5e-5 / 2.5e-5 / 1.25e-5 gives
1,887,255 / 7,156,480 / 26,607,878 / 96,981,051 terms, i.e. terms scale
roughly as `lower_atol^-1.9`.

`--hubbard-observable-site` (default 46) is a lattice position, not a
constant: sweeping `--hubbard-num-sites` without scaling it as 46/60 of the
lattice slides the observable off-centre and changes the light cone.

`build_graph` extends the graph, so Hubbard's 29 Trotter steps retain 29
layer-sets and exceed 229 GiB at 23.9M terms, where `propagate` runs the same
model to 97M terms in well under 2 GiB. Size a graph-holding benchmark from a
graph measurement, never from a `propagate` measurement.
