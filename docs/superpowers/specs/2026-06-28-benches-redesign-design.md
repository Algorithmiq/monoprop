# Benchmark suite redesign

## Goal

Simplify and restructure the `benches/` suite, add Schrödinger-picture coverage to
the random benchmarks, expose `lower_atol` as a CLI knob for the static benchmarks,
and split `REPORT.md` into per-picture sections — without touching the existing
MPI / driver / threading machinery.

## Requirements

1. **Consolidate** the four `bench_*.py` files into a single `bench_monoprop.py`.
2. **Schrödinger picture** for the **random** benchmarks only (graph-based evolve
   *and* in-place), run with `schrodinger_cutoff = cutoff + 2`. The static
   benchmarks (Hubbard, Pauli) stay **Heisenberg-only**.
3. **`--lower-atol`** CLI option overriding the built-in `lower_atol` on the
   Hubbard and Pauli configs (scoped to the static benches).
4. **`REPORT.md`** gains two top-level sections — `## Heisenberg` and
   `## Schrödinger` — each with `### Time` and `### Peak memory` tables.
5. Keep the MPI machinery (`barriered`, `bench_comm`, rank-0 JSON writing,
   `--bench-rounds`, `pedantic` timing) and the `run.py` driver unchanged.

## Non-goals

- No changes to `run.py`'s MPI / threads / pinning / multi-label logic.
- No Schrödinger variant for the static benches.
- No new `lower_atol` knob for the random benches (the in-place random bench keeps
  its existing constant; the graph-based random bench keeps `lower_atol=None`).

## Design

### File layout (after)

| File | Change |
|---|---|
| `bench_monoprop.py` | **NEW** — unified suite (replaces the four `bench_*.py`) |
| `bench_random_evolve.py`, `bench_random_inplace.py`, `bench_hubbard.py`, `bench_pauli.py` | **DELETED** |
| `_builders.py` | `build_random_propagator` gains `schrodinger: bool`; static configs accept a `lower_atol` override |
| `conftest.py` | add `--lower-atol` option + `picture` fixture |
| `report.py` | route ops into Heisenberg / Schrödinger sections by node-id tag |
| `run.py` | unchanged (doc string may mention picture coverage) |
| `README.md` | updated for the single-file layout, picture coverage, `--lower-atol` |

### Picture parametrization

A single auto-parametrized fixture in `conftest.py`:

```python
@pytest.fixture(params=["heisenberg", "schrodinger"])
def picture(request) -> str:
    return request.param
```

`build_random_propagator` selects the picture:

```python
def build_random_propagator(problem, *, comm=None, lower_atol=None, schrodinger=False):
    return MonomialPropagator(
        problem.observable, problem.circuit, problem.cutoff,
        schrodinger_cutoff=problem.cutoff + 2 if schrodinger else None,
        lower_atol=lower_atol, comm=comm,
    )
```

The `built_graph` fixture and the in-place random test depend on `picture`, so each
random operation runs in **both** pictures and its pytest node id carries
`[heisenberg]` / `[schrodinger]` automatically. Confirmed supported: the gradient
functional works in the Schrödinger picture (see `tests/test_infinite_cutoff.py`,
`test_gradient` parametrized over a `schrodinger` flag using `cutoff + 2`).

Random operations (each ×2 pictures):
`test_random_build_graph`, `test_random_pare`, `test_random_energy`,
`test_random_gradient`, `test_random_inplace`.

### Static benchmarks (table-driven, Heisenberg-only)

The two static models share one in-place run+expectation harness and differ only by
builder/config. They become a small parametrized pair:

```python
STATIC_MODELS = [
    ("hubbard", build_hubbard_problem, HubbardConfig),
    ("pauli",   build_kicked_ising_problem, KickedIsingConfig),
]
```

yielding `test_static[hubbard]` and `test_static[pauli]`, both `@pytest.mark.slow`,
`pedantic(rounds=1)`. They read `--lower-atol`: when provided it overrides the
config's built-in value; when omitted, each config's existing default is used
(Hubbard 1e-5, Pauli 1e-4).

### `--lower-atol` plumbing

`conftest.py` registers `--lower-atol` (float, default `None`) in the existing
`monoprop-bench` option group, exposed via a session fixture. The static test reads
it and constructs the config with the override applied (the frozen dataclass is
rebuilt via `dataclasses.replace` when the option is set).

### Report routing

`report.py` routes each op by the picture tag embedded in its node id, present in
both the pytest-benchmark `fullname` and the memray `.bin` filename (brackets
survive on disk):

- contains `[schrodinger]` → **Schrödinger** section
- otherwise (`[heisenberg]` or no tag, e.g. static ops) → **Heisenberg** section

Display names strip the picture tag (the section header already implies it) and the
file part, producing `random / energy`, `static / hubbard`, etc.

New `REPORT.md` structure:

```
# monoprop benchmark report
<intro line>
## Configuration            (unchanged)
## Heisenberg
### Time
### Peak memory             ← static (hubbard, pauli) + heisenberg random ops
## Schrödinger
### Time
### Peak memory             ← schrodinger random ops only
```

A section is omitted if it has no ops (keeps a Heisenberg-only run clean).

## Testing / verification

- `just bench-smoke` runs the full suite at tiny sizes (timing only) and exits 0,
  producing both pictures for the random ops.
- `uv run --group bench python benches/report.py` regenerates `REPORT.md` with the
  two-section layout from existing artifacts.
- Spot-check `REPORT.md`: random ops appear under both sections; `static/hubbard`
  and `static/pauli` appear only under Heisenberg.
- `--lower-atol 1e-6` changes the value recorded/used for the static benches.
- Python lints (pre-commit / ruff) pass on the changed files.
