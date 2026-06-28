# Random Benchmark Sizing & Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the random energy/gradient/pare benchmarks measure a meaningful Heisenberg graph by raising the default problem size, lock that with a regression test proving the graph is actually built, and replace the cryptic `k`/`x` labels with descriptive prose.

**Architecture:** The random benchmarks already build a full propagation graph in the `built_graph` fixture (it calls `mp.propagate()`), so energy/gradient/pare operate on a real graph — the build cost is deliberately in fixture setup, excluded from timing and memray. The only problem is the *defaults*: at 32 modes / cutoff 8 / 4 observable terms / 100 generators the Heisenberg graph is ~2.9k nodes (noise). This plan raises the defaults to 128 modes / cutoff 6 / 10 000 observable terms / 200 generators (Heisenberg graph ≈ 122k nodes, `size` ≈ 132k), adds a fast unit test that the path builds a non-empty graph, and renames the `k`/`x` shorthand in docstrings, `--help`, and the README to "generator length" / "number of generators".

**Tech Stack:** Python, pytest, pytest-benchmark, pytest-memray, uv, just, nanobind-bound `monoprop`.

## Global Constraints

- Conventional commits with gitmoji: `<type>(<scope>): <gitmoji> <description>` (see `CONTRIBUTING.md`).
- Python must pass pre-commit lints (ruff check + ruff format). The `insert-license` hook auto-adds the Apache 2.0 header to new files and aborts the commit; re-stage and re-commit when it fires.
- New Python files need the Apache 2.0 header (the hook adds it; or copy it from `tests/test_bench_report.py:1-13`).
- Python docstrings use Google style.
- `benches/` is on the pytest pythonpath (`pyproject.toml` `pythonpath = [".", "benches"]`), so `from _builders import ...` works from `tests/`.
- `benches/` is NOT collected by the normal `pytest` run (`testpaths = tests/`); tests that must run in CI live under `tests/`.
- Work stays on the existing `feat/python-benches` branch.
- "qubits" maps to fermionic modes 1:1 under Jordan–Wigner, so "128 qubits" = `num_modes = 128` (256 Majorana indices). "Initial operators" = observable terms (`obs_terms`). "Gates" = generators (`num_generators`).

**Working-tree note:** `benches/bench_monoprop.py` currently has one uncommitted line change (`INPLACE_LOWER_ATOL` `1e-6 → 1e-5`) unrelated to this plan. Before starting, commit it on its own or leave it staged-out so it does not get bundled into Task 1/2:

```bash
git add benches/bench_monoprop.py
git commit -m "test(benches): 🔧 align random in-place lower_atol with static default"
```

---

### Task 1: Raise random-benchmark default sizes (+ regression test)

Changes *what the benchmark runs by default* and pins it with a test. The Heisenberg graph goes from ~2.9k to ~122k nodes; CLI flags still override every value.

**Files:**
- Modify: `benches/_builders.py:124-132` (`make_random_problem` keyword defaults)
- Modify: `benches/conftest.py:49-64` (`--gen-length`/`--obs-terms`/`--num-generators`/`--num-modes`/`--cutoff` `default=` and `help=`)
- Create: `tests/test_bench_builders.py`

**Interfaces:**
- Consumes: `make_random_problem(*, gen_length, obs_terms, num_generators, num_modes, cutoff, seed) -> RandomProblem` and `build_random_propagator(problem, *, comm=None, lower_atol=None, schrodinger=False) -> MonomialPropagator` (both in `benches/_builders.py`); `MonomialPropagator.propagate()` and `MonomialPropagator.graph_size() -> tuple[int, int]` returning `(edges, nodes)`.
- Produces: new default sizes `gen_length=4, obs_terms=10000, num_generators=200, num_modes=128, cutoff=6, seed=0`; a guard test module Task 2 does not depend on.

- [ ] **Step 1: Write the failing test for the new defaults**

Create `tests/test_bench_builders.py` (the `insert-license` hook will prepend the Apache header on commit; or copy `tests/test_bench_report.py:1-13`):

```python
"""Tests for the benchmark model builders (``benches/_builders.py``).

``benches`` is on the pytest pythonpath (see ``pyproject.toml``), so the module
imports as ``_builders`` from the normal test suite.
"""

from __future__ import annotations

import inspect

from _builders import build_random_propagator, make_random_problem


def test_random_default_sizes_are_meaningful() -> None:
    defaults = {
        name: param.default
        for name, param in inspect.signature(make_random_problem).parameters.items()
    }
    assert defaults["gen_length"] == 4
    assert defaults["obs_terms"] == 10000
    assert defaults["num_generators"] == 200
    assert defaults["num_modes"] == 128
    assert defaults["cutoff"] == 6
    assert defaults["seed"] == 0


def test_built_graph_is_populated() -> None:
    # A deliberately tiny problem keeps this fast while proving the
    # energy/gradient/pare path operates on a real (non-empty) graph: the
    # benchmark builds the graph in its fixture before measuring. gen_length=4
    # (a length-4 Majorana monomial is Hermitian with real coefficients; a
    # length-2 one is anti-Hermitian and would be rejected as non-Hermitian).
    problem = make_random_problem(
        gen_length=4, obs_terms=3, num_generators=5, num_modes=6, cutoff=3, seed=0
    )
    propagator = build_random_propagator(problem)
    propagator.propagate()
    _edges, nodes = propagator.graph_size()
    assert nodes > 0
```

- [ ] **Step 2: Run the test to verify the defaults test fails**

Run: `uv run pytest tests/test_bench_builders.py -v`
Expected: `test_random_default_sizes_are_meaningful` FAILS — today's defaults are `obs_terms=4`, `num_generators=100`, `num_modes=32`, `cutoff=8`, so the first changed assertion (`obs_terms == 10000`) fails. `test_built_graph_is_populated` PASSES — the graph is already built today; this test is the guard that documents the invariant.

- [ ] **Step 3: Update `make_random_problem` defaults**

In `benches/_builders.py`, change the signature at lines 124-132 from:

```python
def make_random_problem(
    *,
    gen_length: int = 4,
    obs_terms: int = 4,
    num_generators: int = 100,
    num_modes: int = 32,
    cutoff: int = 8,
    seed: int = 0,
) -> RandomProblem:
```

to:

```python
def make_random_problem(
    *,
    gen_length: int = 4,
    obs_terms: int = 10000,
    num_generators: int = 200,
    num_modes: int = 128,
    cutoff: int = 6,
    seed: int = 0,
) -> RandomProblem:
```

- [ ] **Step 4: Update the CLI option defaults and help text**

In `benches/conftest.py`, change the option registrations (lines 49-64) so defaults match `make_random_problem` and the `help=` strings drop `k`/`x`:

```python
    group.addoption(
        "--gen-length",
        type=int,
        default=4,
        help="Number of Majorana operators per generator.",
    )
    group.addoption(
        "--obs-terms", type=int, default=10000, help="Number of observable terms."
    )
    group.addoption(
        "--num-generators",
        type=int,
        default=200,
        help="Number of random generators (circuit gates).",
    )
    group.addoption(
        "--num-modes", type=int, default=128, help="Number of fermionic modes."
    )
    group.addoption("--cutoff", type=int, default=6, help="Truncation cutoff.")
    group.addoption(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_bench_builders.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Confirm the bench suite still collects with the new defaults**

Run: `uv run --group bench pytest benches/bench_monoprop.py --collect-only -q`
Expected: collection succeeds, listing `test_random_*[heisenberg]`/`[schrodinger]` and `test_static[hubbard]`/`[pauli]` with no errors.

- [ ] **Step 7: Run ruff on the touched files**

Run: `uv run ruff check benches/_builders.py benches/conftest.py tests/test_bench_builders.py && uv run ruff format --check benches/_builders.py benches/conftest.py tests/test_bench_builders.py`
Expected: no errors (run `uv run ruff format benches/_builders.py benches/conftest.py tests/test_bench_builders.py` first if the format check reports a diff).

- [ ] **Step 8: Commit**

```bash
git add benches/_builders.py benches/conftest.py tests/test_bench_builders.py
git commit -m "test(benches): 📈 raise random-benchmark default sizes to a meaningful scale"
```

(If `insert-license` rewrites the new test file and aborts, re-run the same `git add` + `git commit`.)

---

### Task 2: Rename `k`/`x` to descriptive prose

Pure documentation: kill the cryptic single-letter labels in docstrings, the module docstring, and the README, and refresh the README's size table to the new defaults.

**Files:**
- Modify: `benches/_builders.py:81`, `benches/_builders.py:136`, `benches/_builders.py:138` (docstrings)
- Modify: `benches/conftest.py:18` (module docstring)
- Modify: `benches/README.md:91`, `benches/README.md:97-104` (prose + size table)

**Interfaces:**
- Consumes: nothing (prose only).
- Produces: no bare `` ``k`` `` / `` ``x`` `` / `` `k` `` / `` `x` `` labels remain in `benches/` documentation.

- [ ] **Step 1: Rename in `_builders.py` docstrings**

In `benches/_builders.py`, replace the `RandomProblem.circuit` attribute doc (line 81):

```python
        circuit: Monomial circuit of random fixed-length Majorana generators.
```

Replace the `make_random_problem` argument docs (lines 136 and 138):

```python
        gen_length: Number of Majorana operators in each random generator.
```

```python
        num_generators: Number of random generators in the circuit.
```

- [ ] **Step 2: Rename in the `conftest.py` module docstring**

In `benches/conftest.py`, replace line 18:

```
generator length, number of observable terms, number of generators,
```

(The surrounding sentence at lines 17-20 then reads: "... so the generator length, number of observable terms, number of generators, mode count, cutoff, and RNG seed can all be varied without editing code, e.g.::".)

- [ ] **Step 3: Rename and refresh the README**

In `benches/README.md`, replace the description line (line 91):

```markdown
`make_random_problem` builds `--num-generators` random fixed-length Majorana
generators and a random Hermitian observable with a configurable number of
terms. Each random operation is run in **both** the Heisenberg and Schrödinger
pictures (the latter with `schrodinger_cutoff = cutoff + 2`), shown as
`[heisenberg]` / `[schrodinger]` variants. All sizes are CLI options (defaults
in parentheses):
```

Then replace the size table (lines 97-104) so descriptions drop `k`/`x` and the default column matches the new defaults:

```markdown
| Option | Meaning | Default |
|---|---|---|
| `--gen-length` | Majorana operators per generator | 4 |
| `--obs-terms` | number of observable terms | 10000 |
| `--num-generators` | number of generators (circuit gates) | 200 |
| `--num-modes` | fermionic modes (Majorana indices = `2·modes`) | 128 |
| `--cutoff` | truncation cutoff | 6 |
| `--seed` | RNG seed | 0 |
```

- [ ] **Step 4: Verify no cryptic `k`/`x` labels remain**

Run: `grep -rn '``k``\|``x``\|length-`k`\|builds `x`\|generators `x`\|length `k`' benches/_builders.py benches/conftest.py benches/README.md`
Expected: no matches (exit status 1, no output).

- [ ] **Step 5: Sanity-check the docstrings import cleanly**

Run: `uv run --group bench python -c "import _builders, conftest" 2>&1 | tail -1`
Expected: no output / no traceback (modules import).

Run: `uv run ruff check benches/_builders.py benches/conftest.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add benches/_builders.py benches/conftest.py benches/README.md
git commit -m "docs(benches): 📝 rename k/x to descriptive names and refresh size table"
```

---

## Verification (after both tasks)

- [ ] Full normal test suite still passes (proves the new `tests/test_bench_builders.py` and the bench imports are healthy):

Run: `uv run pytest tests/test_bench_builders.py tests/test_bench_report.py -v`
Expected: all PASS.

- [ ] Optional heavy confirmation that the bigger defaults actually build (one Heisenberg build-graph round, no memory pass) — only if you want to see the new scale end to end:

Run: `uv run --group bench pytest "benches/bench_monoprop.py::test_random_build_graph[heisenberg]" --bench-rounds 1 -o filterwarnings=default -q`
Expected: PASS (builds the ~122k-node Heisenberg graph).
