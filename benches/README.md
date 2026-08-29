# monoprop benchmarks

The `monoprop` repository includes a pytest suite measuring the **time** and
**peak resident memory (RSS)** of monoprop's core operations.
The suite is separated from the test suite and can be run with `just bench`.
See below for more detailed instructions.

See the [Benchmarks](../docs/content/docs/benchmarks.mdx) section of the
documentation for detailed instructions.

## What lives where

This directory holds only monoprop's own benchmarks — `conftest.py` (the fixtures
and the results schema), `bench_random.py`, `bench_models.py`, `rungs.toml` (the
benchmark set), [`RUNGS.md`](RUNGS.md) (how to run it) and `results/`.

Both bench modules measure the same four operations — `build_graph`, `propagate`,
`energy` and `gradient` — so a number means the same thing whichever problem
produced it.

Benchmark names are the key [Bencher](https://bencher.dev/) stores history under,
so they stay here rather than moving with a library release.

Everything reusable is in the `monoprop-bench-tools` package
([`../packages/monoprop-bench-tools`](../packages/monoprop-bench-tools)): the
memory instrumentation, the model builders, the two renderers that turn a run's
artifacts into `REPORT.md` and Bencher Metric Format JSON, and the runner that
executes one row of `rungs.toml` and gates the result on its term count.

Cross-engine comparisons against other propagation libraries live in
[`../packages/bench-third-party`](../packages/bench-third-party), a standalone uv
project with its own lockfile.

## The rung ladder

`rungs.toml` is the benchmark set: one row per cell, giving the picture, the model,
the operations, the geometry, the size knobs and the exact term count that
configuration produces. A row, not a loop, so a campaign cannot quietly run a
different grid and two campaigns' numbers are comparable row for row.

```bash
monoprop-bench-rung benches/rungs.toml list             # the set, and what each costs
monoprop-bench-rung benches/rungs.toml <id> --dry-run   # the plan, no allocation spent
monoprop-bench-rung benches/rungs.toml <id> --rep 1     # one rep
monoprop-bench-ladder benches/rungs.toml benches/results  # the block to paste into the PR
```

Nothing runs these for you. Run the rungs your change could plausibly move and put
the block in the pull request: it carries the timings, the peak memory, and the
resolved parameters of every problem measured.

`expect_terms` is a gate, not documentation: a result missing it by more than 0.1%
is refused, so a mistyped tolerance fails the cell instead of measuring a different
problem under the right name. A row nobody has calibrated says so twice --
`expect_terms = 0` and `TBD` on the unmeasured knob -- and refuses to run.

`cost_seconds` and `cost_gib_per_node` are the opposite: what one rep last cost, so
you can see what a rung takes before you spend it. Documentation, never a gate.
The machine and scheduler that run a rung are not in this repository; the table,
the runner and the gate are.

**[`RUNGS.md`](RUNGS.md)** is the guide: every model's parameters and what they do,
how the geometry maps onto a machine, example Slurm scripts, the pinning and
allocation-sizing traps, and how to calibrate a `TBD` row.

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
