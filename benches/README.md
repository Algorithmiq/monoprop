# monoprop benchmarks

The `monoprop` repository includes a pytest suite measuring the **time** and
**peak resident memory (RSS)** of monoprop's core operations.
The suite is separated from the test suite and can be run with `just bench`.
See below for more detailed instructions.

See the the [Benchmarks](../docs/content/docs/benchmarks.mdx) section of the
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
