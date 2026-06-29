# Lightweight memory profiling — design

**Date:** 2026-06-29

## Goal

Replace `pytest-memray` with a near-zero-cost peak-RSS measurement so the
benchmark suite records timing and memory in a single sweep without the
allocation-tracking overhead distorting timing (memray inflated the malloc-bound
`hubbard` timing ~1.68× in a combined sweep).

## Background

Memory is currently captured with `pytest-memray` (`--memray`), which interposes
on every allocation. That interception is the source of the timing distortion: a
combined timing + memory sweep cannot be trusted for allocation-heavy benchmarks,
forcing either a slow two-pass run or an inflated single-pass number.

The Linux kernel already tracks each process's peak resident set size (`VmHWM`)
for free. It is normally monotonic for the process lifetime — useless for
per-test isolation — but writing `5` to `/proc/self/clear_refs`
(`CLEAR_REFS_MM_HIWATER_RSS`, Linux ≥ 4.0) resets it to the *current* RSS. That
makes an exact per-operation peak available with no allocation interception, no
sampling thread, and no measurable timing cost. RSS includes the C++/TBB
allocations that `tracemalloc` misses — the original reason for adopting memray.

## Metric semantics

The reported number changes meaning, deliberately:

- **Before (memray):** peak *heap allocated during the call* — excludes
  structures already resident when the call starts.
- **After (RSS):** peak *total resident footprint* of the process during the op.

For the graph-based ops (`pare`, `energy`, `gradient`) the shared session-scoped
`built_graph` is already resident when the op runs, so its bytes form the
baseline floor of the peak. These numbers will therefore read higher than the
old memray heap figures. This is the intended "peak total footprint" semantics;
the report header states it plainly.

The environment is Linux-only (container/CI), so the `clear_refs` mechanism is
always available; no cross-platform fallback is in scope.

## Components

### 1. `peak_memory` autouse fixture (`benches/conftest.py`)

A function-scoped `@pytest.fixture(autouse=True)` applied to every bench test,
depending on `bench_comm`.

- **Setup (before `yield`):** reset the high-water mark by writing `"5\n"` to
  `/proc/self/clear_refs`.
- **Teardown (after `yield`):** read this rank's `VmHWM` from
  `/proc/self/status` (reported in kB; convert to bytes). Under MPI, reduce with
  `comm.allreduce(local_peak, op=MPI.MAX)` so the recorded value is the
  worst-rank peak (collective — every rank runs every test, mirroring
  `_record_operator_size`). Only rank 0 writes.
- **Writing:** env-gated exactly like the other recorders — no-op unless
  `MONOPROP_BENCH_LABEL` and `MONOPROP_BENCH_RESULTS` are set. Merge
  `{op_key: peak_bytes}` into `mem-<label>.json` (read-modify-write JSON, one
  entry per op).
- **Key:** `op_key` is `request.node.nodeid` normalised with the existing
  `split("/")[-1]` rule (see `_op_key_from_fullname`) so it matches the timing
  keys exactly (`bench_monoprop.py::test_name[param]`).

Two small read helpers (peak read, hwm reset) live in `conftest.py` alongside the
fixture; they are trivial file reads/writes and need no separate module.

### 2. Report (`benches/report.py`)

- Rewrite `_load_memory(results_dir)` to read `mem-*.json` files into
  `{label: {op_key: peak_bytes}}`, replacing the memray-bin reader.
- Delete the now-unused `import memray`, `_op_key_from_bin`, and the `_HEX_PREFIX`
  regex it used.
- `_fmt_mem`, `_section`, and the `mem_cells` construction are unchanged.
- Reword the report header and the `_load_memory`/module docstrings: "peak heap"
  → "peak resident memory (RSS)".

### 3. Driver and dependencies

- **`benches/run.py`:** remove the `--no-mem` flag and all memray sweep args
  (`--memray`, `--memray-bin-path`, the `memray-<label>/` mkdir). The sweep is
  just `--benchmark-json <path>`; memory records itself via the fixture, always
  and for free. Update the module docstring.
- **`pyproject.toml`:** remove the `pytest-memray>=1.5` dependency.
- **`benches/bench_monoprop.py`:** update the module docstring (memray → peak
  RSS).
- **`benches/README.md`:** swap the memray description/link for the peak-RSS
  description; remove any `--no-mem` mention.
- **`benches/results/.gitignore`:** drop `memray-*/`, add `mem-*.json` to the
  artifact comment.

## Out of scope

- Cross-platform memory measurement (Linux-only by design).
- Keeping memray as an opt-in path (fully replaced).
- Allocation-breakdown / flamegraph reporting (memray's detailed output is
  dropped; only the peak number was ever consumed by the report).

## Testing

- Unit-test the new `_load_memory` against a hand-written `mem-np1.json`
  fixture in `tests/test_bench_report.py`, asserting the peak bytes render in the
  picture sections' "Peak memory" tables (mirrors the existing
  `test_build_report_includes_operator_size`).
- A smoke run (`just bench-smoke`) confirms `mem-<label>.json` is produced and
  the report renders memory rows.
- Existing report tests stay green after the memray reader is replaced.
