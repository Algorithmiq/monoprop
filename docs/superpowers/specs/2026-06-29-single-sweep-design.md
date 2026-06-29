# Single timing+memory sweep — design

**Date:** 2026-06-29
**Status:** Approved

## Problem

`benches/run.py` runs the suite **twice** per label: a timing pass
(`--benchmark-json`) and then a separate memory pass
(`--memray --benchmark-disable`). The static models (`hubbard`, `pauli`) are
multi-second in-place simulations run at `rounds=1`, so running them twice
dominates the wall-clock of a full `just bench`.

The two passes exist on the assumption that `memray`'s allocation tracking would
inflate the timing numbers. Measurement on this suite disproves that:

| op | clean | +memray | ratio (min of 4 interleaved runs) |
|---|--:|--:|--:|
| build_graph | 9.26 ms | 8.26 ms | 0.89× |
| pare | 0.63 ms | 0.66 ms | 1.06× |
| energy | 0.050 ms | 0.051 ms | 1.01× |
| gradient | 0.062 ms | 0.066 ms | 1.06× |
| inplace | 9.28 ms | 9.69 ms | 1.04× |

All within the suite's run-to-run noise (~±10%), including the allocation-heavy
`build_graph`/`inplace`. The ops are compute-bound at the granularity memray
hooks, so timing under memray is not measurably corrupted. The combined run also
produced valid memray peaks (e.g. build_graph 24.7 MiB).

## Goal

Run each label **once** — a single sweep that records both `--benchmark-json`
timing and `--memray` peaks — halving the static-model wall-clock at no
measurable timing cost.

## Non-goals

- Changing what is recorded (same `time-<label>.json` and `memray-<label>/`).
- Changing round counts or the `barriered` makespan logic.
- Removing `--no-mem` (it still skips the memray instrumentation entirely).

## Design

In `benches/run.py::main`, replace the two `_launch` calls with one. The memray
flags are appended to the timing invocation instead of forming a second pass;
`--benchmark-disable` is dropped (timing must stay live in the combined run):

```python
    bench_json = results_dir / f"time-{label}.json"
    sweep_args = [*common, "--benchmark-json", str(bench_json)]
    if not args.no_mem:
        mem_dir = results_dir / f"memray-{label}"
        mem_dir.mkdir(exist_ok=True)
        sweep_args += ["--memray", "--memray-bin-path", str(mem_dir)]
    _launch(prefix, sweep_args, run_env)
```

Run-count effect per label:

| | before (2 passes) | after (1 sweep) |
|---|--:|--:|
| random op | 5 (timing) + 1 (mem) = 6 | 5 |
| static op | 1 + 1 = 2 | 1 |

Memory-peak correctness under multiple benchmark rounds: peak is the max across
rounds, which equals the per-round peak for these deterministic, round-isolated
ops (`setup` rebuilds the random graph each round; the graph-based ops free their
functional each round). Confirmed on the random ops; validated on the static
models as part of implementation.

Update the prose that describes "two passes":
- `run.py` module docstring ("runs the timing pass … and the memory pass …").
- `benches/README.md` Running section ("runs the timing pass, the memory pass").

## Verification

- Run one serial combined sweep over the static models only and confirm both
  artifacts are produced and the timings are sane:
  `python benches/run.py -m slow --results-dir <scratch>` → `time-np1.json` has
  `test_static[hubbard]`/`[pauli]` means, and `memray-np1/` has their `.bin`
  peaks.
- `uv run pytest tests/test_bench_report.py tests/test_bench_builders.py -q`
  still green (report/builders untouched).
