# Benchmark Suite Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the four `bench_*.py` files into one parametrized suite, add Schrödinger-picture coverage to the random benchmarks, expose a `--lower-atol` knob for the static benchmarks, and split `REPORT.md` into per-picture sections.

**Architecture:** A single `bench_monoprop.py` holds all benchmarks. A `picture` fixture auto-parametrizes the random benches over Heisenberg and Schrödinger (`schrodinger_cutoff = cutoff + 2`); the static benches are a table-driven `test_static[hubbard|pauli]` pair, Heisenberg-only. `report.py` routes ops into `## Heisenberg` / `## Schrödinger` sections by the `[picture]` tag in each op's pytest node id. The MPI / driver / threading machinery (`run.py`, `barriered`, `bench_comm`, rank-0 JSON, `--bench-rounds`) is untouched.

**Tech Stack:** Python, pytest, pytest-benchmark, pytest-memray, monoprop, `just`, `uv`.

## Global Constraints

- Python code must pass the pre-commit lints (ruff check + ruff format).
- Python docstrings use Google style.
- Benchmark functions are marked `@pytest.mark.bench`; static (multi-second) benches additionally `@pytest.mark.slow`.
- The Schrödinger cutoff is always `cutoff + 2` (mirrors `tests/test_infinite_cutoff.py::test_gradient`).
- `benches/` is currently **untracked** in git; deletions of old bench files are plain `rm`, not `git rm`.
- Commits go on the `refactor/benches-schrodinger` branch (the repo blocks commits to `main`). Conventional-commit format: `<type>(<scope>): <gitmoji> <description>`.
- Do not modify `run.py`'s MPI / threads / pinning / multi-label logic.

---

### Task 1: Add Schrödinger flag to the random-propagator builder

**Files:**
- Modify: `benches/_builders.py:156-178` (`build_random_propagator`)

**Interfaces:**
- Produces: `build_random_propagator(problem, *, comm=None, lower_atol=None, schrodinger=False) -> MonomialPropagator`. When `schrodinger=True`, the propagator is constructed with `schrodinger_cutoff=problem.cutoff + 2`; otherwise `schrodinger_cutoff=None` (Heisenberg).

- [ ] **Step 1: Edit `build_random_propagator`**

Replace the existing function body (currently `benches/_builders.py:156-178`) with:

```python
def build_random_propagator(
    problem: RandomProblem,
    *,
    comm: Any | None = None,
    lower_atol: float | None = None,
    schrodinger: bool = False,
) -> MonomialPropagator:
    """Construct a propagator for a random problem (optionally MPI-aware).

    Args:
        problem: The random observable/circuit problem.
        comm: Optional MPI communicator (``None`` for a serial run).
        lower_atol: Optional coefficient-truncation tolerance.
        schrodinger: If ``True``, build in the Schrödinger picture with
            ``schrodinger_cutoff = problem.cutoff + 2``; otherwise Heisenberg.

    Returns:
        A fresh :class:`MonomialPropagator`.
    """
    return MonomialPropagator(
        problem.observable,
        problem.circuit,
        problem.cutoff,
        schrodinger_cutoff=problem.cutoff + 2 if schrodinger else None,
        lower_atol=lower_atol,
        comm=comm,
    )
```

- [ ] **Step 2: Verify both pictures construct correctly**

Run:
```bash
cd /workspaces/monoprop && uv run python -c "
import sys; sys.path.insert(0, 'benches')
from _builders import make_random_problem, build_random_propagator
p = make_random_problem(num_generators=8, num_modes=8, cutoff=6)
assert build_random_propagator(p).schrodinger is False
assert build_random_propagator(p, schrodinger=True).schrodinger is True
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add benches/_builders.py
git commit -m "refactor(benches): 🔧 add schrodinger flag to build_random_propagator"
```

---

### Task 2: Unified `bench_monoprop.py` + conftest fixtures

**Files:**
- Create: `benches/bench_monoprop.py`
- Modify: `benches/conftest.py` (add `--lower-atol` option, `lower_atol` + `picture` fixtures, make `built_graph` picture-aware)
- Delete: `benches/bench_random_evolve.py`, `benches/bench_random_inplace.py`, `benches/bench_hubbard.py`, `benches/bench_pauli.py`

**Interfaces:**
- Consumes: `build_random_propagator(..., schrodinger=...)` from Task 1; `barriered`, `build_hubbard_problem`, `build_kicked_ising_problem`, `HubbardConfig`, `KickedIsingConfig` from `_builders`.
- Produces pytest node ids (consumed by `report.py` in Task 3):
  - `bench_monoprop.py::test_random_build_graph[heisenberg|schrodinger]`
  - `bench_monoprop.py::test_random_pare[heisenberg|schrodinger]`
  - `bench_monoprop.py::test_random_energy[heisenberg|schrodinger]`
  - `bench_monoprop.py::test_random_gradient[heisenberg|schrodinger]`
  - `bench_monoprop.py::test_random_inplace[heisenberg|schrodinger]`
  - `bench_monoprop.py::test_static[hubbard|pauli]`

- [ ] **Step 1: Add the `--lower-atol` option to conftest**

In `benches/conftest.py`, inside `pytest_addoption`, after the `--bench-rounds` option (currently ends at line 56), add:

```python
    group.addoption(
        "--lower-atol",
        type=float,
        default=None,
        help="Override lower_atol for the static (hubbard, pauli) benchmarks.",
    )
```

- [ ] **Step 2: Add the `lower_atol` and `picture` fixtures to conftest**

In `benches/conftest.py`, after the `bench_rounds` fixture (currently ends at line 83), add:

```python
@pytest.fixture(scope="session")
def lower_atol(request: pytest.FixtureRequest) -> float | None:
    """Return the static-benchmark ``lower_atol`` override (``None`` if unset)."""
    return request.config.getoption("--lower-atol")


@pytest.fixture(params=["heisenberg", "schrodinger"])
def picture(request: pytest.FixtureRequest) -> str:
    """Parametrize the random benchmarks over the physical picture."""
    return request.param
```

- [ ] **Step 3: Make `built_graph` picture-aware in conftest**

Replace the existing `built_graph` fixture (currently `benches/conftest.py:100-110`) with:

```python
@pytest.fixture
def built_graph(
    random_problem: RandomProblem, bench_comm: Any, picture: str
) -> MonomialPropagator:
    """Return a propagator whose graph has been built (no coefficients contracted).

    Built fresh per test, in the requested picture, so its propagation graph is
    not shared between measurements; the build cost happens in fixture setup so
    it is excluded from the memory profile of the operation under test.
    """
    mp = build_random_propagator(
        random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
    )
    mp.propagate()
    return mp
```

- [ ] **Step 4: Create `benches/bench_monoprop.py`**

```python
"""Unified monoprop benchmark suite (time + peak memory).

Random benchmarks are configurable (see ``benches/README.md``) and run in both
the Heisenberg and Schrödinger pictures (the latter with
``schrodinger_cutoff = cutoff + 2``). Static benchmarks are fixed, heavy,
Heisenberg-only in-place simulations: the 120-qubit Fermi-Hubbard trajectory and
the 127-qubit Pauli-basis kicked-Ising circuit.

Operations are barrier-wrapped so the measured time is the makespan across MPI
ranks. Timing uses ``pytest-benchmark``; peak memory uses ``pytest-memray``
(``--memray``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from _builders import (
    HubbardConfig,
    KickedIsingConfig,
    barriered,
    build_hubbard_problem,
    build_kicked_ising_problem,
    build_random_propagator,
)

if TYPE_CHECKING:
    from _builders import RandomProblem

    from monoprop import MonomialPropagator

PARE_THRESHOLD = 1e-10
# In-place random bench keeps its own truncation constant (see design non-goals).
INPLACE_LOWER_ATOL = 1e-6


# --------------------------------------------------------------------------- #
# Random benchmarks (configurable; both pictures)
# --------------------------------------------------------------------------- #
@pytest.mark.bench
def test_random_build_graph(
    benchmark: object,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
    picture: str,
) -> None:
    """Benchmark building the propagation graph from a fresh propagator."""

    def setup() -> tuple[tuple[MonomialPropagator], dict]:
        mp = build_random_propagator(
            random_problem, comm=bench_comm, schrodinger=picture == "schrodinger"
        )
        return (mp,), {}

    def build(mp: MonomialPropagator) -> None:
        mp.propagate()

    benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(build, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )


@pytest.mark.bench
def test_random_pare(
    benchmark: object,
    built_graph: MonomialPropagator,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
) -> None:
    """Benchmark paring the graph into a masked execution plan."""

    def pare() -> object:
        return built_graph.expectation_value_and_gradient_functional(
            parameter_mapping=random_problem.parameter_mapping,
            gen_coeffs=random_problem.gen_coeffs,
            pare_threshold=PARE_THRESHOLD,
        )

    benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(pare, bench_comm), rounds=bench_rounds, iterations=1
    )


@pytest.mark.bench
def test_random_energy(
    benchmark: object,
    built_graph: MonomialPropagator,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
) -> None:
    """Benchmark evaluating the expectation-value functional."""
    functional = built_graph.expectation_value_functional(
        parameter_mapping=random_problem.parameter_mapping,
        gen_coeffs=random_problem.gen_coeffs,
        pare_threshold=PARE_THRESHOLD,
    )
    result = benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(functional, bench_comm),
        args=(random_problem.parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    assert isinstance(result, float)


@pytest.mark.bench
def test_random_gradient(
    benchmark: object,
    built_graph: MonomialPropagator,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
) -> None:
    """Benchmark evaluating the expectation-value-and-gradient functional."""
    functional = built_graph.expectation_value_and_gradient_functional(
        parameter_mapping=random_problem.parameter_mapping,
        gen_coeffs=random_problem.gen_coeffs,
        pare_threshold=PARE_THRESHOLD,
    )
    _value, gradient = benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(functional, bench_comm),
        args=(random_problem.parameters,),
        rounds=bench_rounds,
        iterations=1,
    )
    assert len(gradient) == len(random_problem.parameters)


@pytest.mark.bench
def test_random_inplace(
    benchmark: object,
    random_problem: RandomProblem,
    bench_comm: Any,
    bench_rounds: int,
    picture: str,
) -> None:
    """Benchmark in-place evolution + expectation value (no graph stored)."""

    def setup() -> tuple[tuple[MonomialPropagator], dict]:
        mp = build_random_propagator(
            random_problem,
            comm=bench_comm,
            lower_atol=INPLACE_LOWER_ATOL,
            schrodinger=picture == "schrodinger",
        )
        return (mp,), {}

    def run(mp: MonomialPropagator) -> float:
        mp.propagate(evolve_with_coeffs=True)
        return mp.expectation_value()

    result = benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(run, bench_comm), setup=setup, rounds=bench_rounds, iterations=1
    )
    assert isinstance(result, float)


# --------------------------------------------------------------------------- #
# Static benchmarks (fixed, heavy, Heisenberg-only, in-place)
# --------------------------------------------------------------------------- #
# Mapping name -> (builder, config factory, Trotter steps per measured run).
# Hubbard re-applies its one-step circuit ``trotter_steps`` times; the Pauli
# circuit already contains all layers, so a single propagate suffices.
STATIC_MODELS: dict[str, tuple] = {
    "hubbard": (build_hubbard_problem, HubbardConfig, HubbardConfig().trotter_steps),
    "pauli": (build_kicked_ising_problem, KickedIsingConfig, 1),
}


@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("model", list(STATIC_MODELS))
def test_static(
    benchmark: object,
    bench_comm: Any,
    lower_atol: float | None,
    model: str,
) -> None:
    """Benchmark a fixed in-place static simulation (Heisenberg picture)."""
    build_fn, config_cls, steps = STATIC_MODELS[model]
    config = config_cls()
    if lower_atol is not None:
        config = replace(config, lower_atol=lower_atol)

    def setup() -> tuple[tuple[MonomialPropagator, int], dict]:
        return (build_fn(config, comm=bench_comm), steps), {}

    def run(mp: MonomialPropagator, n_steps: int) -> float:
        value = 0.0
        for _ in range(n_steps):
            mp.propagate(evolve_with_coeffs=True)
            value = mp.expectation_value()
        return value

    result = benchmark.pedantic(  # type: ignore[attr-defined]
        barriered(run, bench_comm), setup=setup, rounds=1, iterations=1
    )
    assert isinstance(result, float)
```

- [ ] **Step 5: Delete the four old bench files**

```bash
cd /workspaces/monoprop && rm benches/bench_random_evolve.py benches/bench_random_inplace.py benches/bench_hubbard.py benches/bench_pauli.py
```

- [ ] **Step 6: Run the smoke suite (random ops, both pictures)**

```bash
cd /workspaces/monoprop && just bench-smoke
```
Expected: pytest collects the random ops twice each (`[heisenberg]` and `[schrodinger]`), all pass, and the run prints a generated `REPORT.md`. Static benches are skipped (`-m "not slow"`).

- [ ] **Step 7: Confirm both pictures were collected**

```bash
cd /workspaces/monoprop && uv run --group bench python -m pytest benches/bench_monoprop.py -m "not slow" \
    --collect-only -q --num-generators 8 --num-modes 8 --cutoff 6 | grep -E "random_(energy|inplace)\[" | sort
```
Expected output includes both:
```
benches/bench_monoprop.py::test_random_energy[heisenberg]
benches/bench_monoprop.py::test_random_energy[schrodinger]
benches/bench_monoprop.py::test_random_inplace[heisenberg]
benches/bench_monoprop.py::test_random_inplace[schrodinger]
```

- [ ] **Step 8: Lint**

```bash
cd /workspaces/monoprop && uv run pre-commit run ruff ruff-format --files benches/bench_monoprop.py benches/conftest.py
```
Expected: Passed (or auto-formatted; re-stage if so).

- [ ] **Step 9: Commit**

```bash
git add benches/bench_monoprop.py benches/conftest.py
git rm benches/bench_random_evolve.py benches/bench_random_inplace.py benches/bench_hubbard.py benches/bench_pauli.py 2>/dev/null || true
git commit -m "refactor(benches): ♻️ unify bench suite into bench_monoprop with picture parametrization"
```

> Note: the old bench files are untracked, so `git rm` may report "did not match any files"; the `|| true` and the `rm` in Step 5 handle that. The deletion is reflected simply by the files being gone.

---

### Task 3: Split `REPORT.md` into Heisenberg / Schrödinger sections

**Files:**
- Modify: `benches/report.py` (`_display_op`, `_table`, `build_report`; add `_picture_of`, `_section`)
- Create: `tests/test_bench_report.py`

**Interfaces:**
- Consumes: op keys of the form `bench_monoprop.py::test_random_energy[heisenberg]` and `bench_monoprop.py::test_static[hubbard]` (from Task 2).
- Produces: `_picture_of(op_key) -> str` (`"schrodinger"` iff the key contains `[schrodinger]`, else `"heisenberg"`); `build_report(results_dir) -> str` emitting `## Heisenberg` and `## Schrödinger` sections, each with `### Time` and `### Peak memory` sub-tables; a section is omitted when it has no ops.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bench_report.py`:

```python
"""Unit tests for the benchmark report builder (``benches/report.py``).

``benches`` is on the pytest pythonpath (see ``pyproject.toml``), so the module
imports as ``report`` from the normal test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import report


def test_picture_of_routes_by_tag() -> None:
    key = "bench_monoprop.py::test_random_energy"
    assert report._picture_of(f"{key}[schrodinger]") == "schrodinger"
    assert report._picture_of(f"{key}[heisenberg]") == "heisenberg"
    assert report._picture_of("bench_monoprop.py::test_static[hubbard]") == "heisenberg"


def test_display_op_strips_picture_and_names_static() -> None:
    assert (
        report._display_op("bench_monoprop.py::test_random_energy[heisenberg]")
        == "random / energy"
    )
    assert (
        report._display_op("bench_monoprop.py::test_random_build_graph[schrodinger]")
        == "random / build_graph"
    )
    assert (
        report._display_op("bench_monoprop.py::test_static[hubbard]")
        == "static / hubbard"
    )


def _write_timings(results_dir: Path) -> None:
    data = {
        "benchmarks": [
            {
                "fullname": "benches/bench_monoprop.py::test_random_energy[heisenberg]",
                "stats": {"mean": 0.001},
            },
            {
                "fullname": "benches/bench_monoprop.py::test_random_energy[schrodinger]",
                "stats": {"mean": 0.002},
            },
            {
                "fullname": "benches/bench_monoprop.py::test_static[hubbard]",
                "stats": {"mean": 1.5},
            },
        ]
    }
    (results_dir / "time-np1.json").write_text(json.dumps(data))


def test_build_report_has_two_sections(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    md = report.build_report(tmp_path)

    assert "## Heisenberg" in md
    assert "## Schrödinger" in md
    # Heisenberg precedes Schrödinger.
    assert md.index("## Heisenberg") < md.index("## Schrödinger")

    heis = md[md.index("## Heisenberg") : md.index("## Schrödinger")]
    schr = md[md.index("## Schrödinger") :]

    # Static op only under Heisenberg; random energy appears in both sections.
    assert "static / hubbard" in heis
    assert "static / hubbard" not in schr
    assert "random / energy" in heis
    assert "random / energy" in schr


def test_build_report_omits_empty_schrodinger_section(tmp_path: Path) -> None:
    data = {
        "benchmarks": [
            {
                "fullname": "benches/bench_monoprop.py::test_static[pauli]",
                "stats": {"mean": 2.0},
            }
        ]
    }
    (tmp_path / "time-np1.json").write_text(json.dumps(data))
    md = report.build_report(tmp_path)
    assert "## Heisenberg" in md
    assert "## Schrödinger" not in md
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cd /workspaces/monoprop && uv run --group bench python -m pytest tests/test_bench_report.py -v
```
Expected: FAIL — `AttributeError: module 'report' has no attribute '_picture_of'` (and the section assertions fail).

- [ ] **Step 3: Add `_picture_of` and rewrite `_display_op`**

In `benches/report.py`, replace the existing `_display_op` function (currently `benches/report.py:147-151`) with:

```python
_PICTURES = ("heisenberg", "schrodinger")


def _picture_of(op_key: str) -> str:
    """Return the physical picture an op belongs to (defaults to Heisenberg)."""
    return "schrodinger" if "[schrodinger]" in op_key else "heisenberg"


def _display_op(op_key: str) -> str:
    """Turn a node id into a ``group / op`` label, dropping the picture tag.

    ``...::test_random_energy[heisenberg]`` -> ``random / energy``;
    ``...::test_static[hubbard]``           -> ``static / hubbard``.
    """
    _file, _, test = op_key.partition("::")
    base, _, param = test.removeprefix("test_").partition("[")
    param = param.rstrip("]")
    if param in _PICTURES:
        param = ""  # the section header already states the picture
    group, _, op = base.partition("_")
    if not op:  # parametrized-only name, e.g. "static" + param "hubbard"
        return f"{group} / {param}" if param else group
    return f"{group} / {op}"
```

- [ ] **Step 4: Change `_table` to use an `H3` heading**

In `benches/report.py`, in `_table` (currently `benches/report.py:154-167`), change the first heading line from `## {title}` to `### {title}`:

```python
    lines = [f"### {title}", "", "| Operation | " + " | ".join(labels) + " |"]
```

- [ ] **Step 5: Add `_section` and rewrite `build_report`'s assembly**

In `benches/report.py`, add this helper directly above `build_report`:

```python
def _section(
    name: str,
    labels: list[str],
    ops: list[str],
    time_cells: dict[str, dict[str, str]],
    mem_cells: dict[str, dict[str, str]],
) -> list[str]:
    """Render one picture section (Time + Peak memory), or nothing if empty."""
    if not ops:
        return []
    return [
        f"## {name}",
        "",
        *_table("Time", labels, ops, time_cells),
        *_table("Peak memory", labels, ops, mem_cells),
    ]
```

Then in `build_report` (currently `benches/report.py:170-206`), replace the final assembly. Change the `ops` computation and the returned `lines` so the body reads:

```python
    labels = sorted(set(timings) | set(memory) | set(metas))
    all_ops = sorted(
        {op for table in timings.values() for op in table}
        | {op for table in memory.values() for op in table}
    )

    if not labels or not all_ops:
        return (
            "# monoprop benchmark report\n\nNo results found. Run `just bench` first.\n"
        )

    heisenberg_ops = [op for op in all_ops if _picture_of(op) == "heisenberg"]
    schrodinger_ops = [op for op in all_ops if _picture_of(op) == "schrodinger"]

    time_cells = {
        label: {op: _fmt_time(t) for op, t in table.items()}
        for label, table in timings.items()
    }
    mem_cells = {
        label: {op: _fmt_mem(m) for op, m in table.items()}
        for label, table in memory.items()
    }

    lines = [
        "# monoprop benchmark report",
        "",
        f"Run labels: **{', '.join(labels)}**. Times are the mean over rounds; "
        "memory is the peak heap (max across ranks under MPI).",
        "",
        *_config_table(labels, metas),
        *_section("Heisenberg", labels, heisenberg_ops, time_cells, mem_cells),
        *_section("Schrödinger", labels, schrodinger_ops, time_cells, mem_cells),
    ]
    return "\n".join(lines)
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
cd /workspaces/monoprop && uv run --group bench python -m pytest tests/test_bench_report.py -v
```
Expected: PASS (all five tests).

- [ ] **Step 7: Lint**

```bash
cd /workspaces/monoprop && uv run pre-commit run ruff ruff-format --files benches/report.py tests/test_bench_report.py
```
Expected: Passed (re-stage if auto-formatted).

- [ ] **Step 8: Commit**

```bash
git add benches/report.py tests/test_bench_report.py
git commit -m "feat(benches): 📊 split benchmark report into per-picture sections"
```

---

### Task 4: Update `README.md` and verify end-to-end

**Files:**
- Modify: `benches/README.md`

**Interfaces:**
- Consumes: the final behaviour from Tasks 1-3 (single bench file, picture coverage, `--lower-atol`, two-section report).

- [ ] **Step 1: Update the "Benchmarks" section of `benches/README.md`**

Replace the block from the `## Benchmarks` heading through the end of the "Static (fixed, in-place)" subsection (currently `benches/README.md:83-110`) with:

```markdown
## Benchmarks

All benchmarks live in a single file, `bench_monoprop.py`.

### Random (configurable, both pictures)

`make_random_problem` builds `x` random length-`k` Majorana generators and a
random Hermitian observable with a configurable number of terms. Each random
operation is run in **both** the Heisenberg and Schrödinger pictures (the latter
with `schrodinger_cutoff = cutoff + 2`), shown as `[heisenberg]` / `[schrodinger]`
variants. All sizes are CLI options (defaults in parentheses):

| Option | Meaning | Default |
|---|---|---|
| `--gen-length` | generator Majorana length `k` | 4 |
| `--obs-terms` | number of observable terms | 4 |
| `--num-generators` | number of generators `x` | 100 |
| `--num-modes` | fermionic modes (Majorana indices = `2·modes`) | 32 |
| `--cutoff` | truncation cutoff | 8 |
| `--seed` | RNG seed | 0 |

Operations: `test_random_build_graph`, `test_random_pare` (masked execution
plan), `test_random_energy`, `test_random_gradient` (graph-based path), and
`test_random_inplace` (in-place coefficient truncation, never materialising the
graph).

### Static (fixed, in-place, Heisenberg only)

- `test_static[hubbard]` — 120-qubit Fermi-Hubbard model (60 sites), sandbox
  default input, run as a 29-step in-place Trotter trajectory.
- `test_static[pauli]` — 127-qubit Pauli-basis kicked-Ising simulation (IBM Eagle
  heavy-hex, 20 layers, ⟨Z₆₂⟩).

Both are marked `slow`. Pass `--lower-atol VALUE` to override their coefficient
truncation tolerance (defaults: Hubbard 1e-5, Pauli 1e-4).
```

- [ ] **Step 2: Update the "Reporting" section to mention the two sections**

In `benches/README.md`, in the "## Reporting" section, replace the sentence listing the report sections (currently `benches/README.md:70-74`, beginning "Each run regenerates") with:

```markdown
Each run regenerates `REPORT.md`, which has:

- **Configuration** — one row per run label: ranks, thread counts, launcher (incl.
  pinning args), CPU count, host.
- **Heisenberg** and **Schrödinger** — one section each, every section holding a
  **Time** and a **Peak memory** table (one row per operation, one column per
  label). The Schrödinger section is omitted when no Schrödinger ops were run.
```

- [ ] **Step 3: Update the file list in "Notes"**

In `benches/README.md`, in the "## Notes" section, replace the final bullet (currently `benches/README.md:119-121`, "Files are organised as…") with:

```markdown
- Files are organised as: `run.py` (the driver), `report.py` (Markdown report),
  `conftest.py` (fixtures + CLI options), `_builders.py` (model construction),
  and `bench_monoprop.py` (the unified benchmark suite).
```

- [ ] **Step 4: End-to-end smoke + report check**

```bash
cd /workspaces/monoprop && just bench-smoke && grep -E "^## (Heisenberg|Schrödinger)|random / " benches/results/REPORT.md
```
Expected: smoke passes; `REPORT.md` shows both `## Heisenberg` and `## Schrödinger` headings and `random / ...` rows under each.

- [ ] **Step 5: Verify `--lower-atol` reaches the static config (no heavy run)**

```bash
cd /workspaces/monoprop && uv run --group bench python -m pytest benches/bench_monoprop.py::test_static -p no:cacheprovider \
    --lower-atol 1e-7 --collect-only -q | grep test_static
```
Expected: `test_static[hubbard]` and `test_static[pauli]` collected without error (confirms the option parses; a full timed run via `just bench` is optional and heavy).

- [ ] **Step 6: Commit**

```bash
git add benches/README.md
git commit -m "docs(benches): 📝 document unified suite, picture coverage, and --lower-atol"
```

---

## Self-Review

**Spec coverage:**
- Consolidate 4 → 1 file → Task 2 (create `bench_monoprop.py`, delete the four).
- Schrödinger for random benches, `cutoff + 2` → Task 1 (`schrodinger` flag) + Task 2 (`picture` fixture, both `test_random_*` paths).
- Static stays Heisenberg-only → Task 2 (`test_static` has no `picture` dependency).
- `--lower-atol` for static benches → Task 2 (option + fixture + `replace` in `test_static`).
- Two-section report, empty section omitted → Task 3 (`_section`, `build_report`, test `test_build_report_omits_empty_schrodinger_section`).
- MPI machinery untouched → no task modifies `run.py` or the barrier/comm logic.
- Docs → Task 4.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows full code.

**Type consistency:** `build_random_propagator(..., schrodinger=bool)` defined in Task 1, consumed identically in Task 2. `_picture_of` / `_display_op` / `_section` / `build_report` names and signatures consistent between Task 3's implementation and its test. Node-id formats produced in Task 2 match the strings parsed in Task 3.
