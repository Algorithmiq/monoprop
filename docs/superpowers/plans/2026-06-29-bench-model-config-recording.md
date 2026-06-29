# Benchmark model-configuration recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the resolved configuration of each static benchmark model (`hubbard`, `pauli`) per run label and surface it in `benches/results/REPORT.md`, and remove the misleading shared `lower_atol` row from the random Hyperparameters table.

**Architecture:** A new pytest fixture in `benches/conftest.py` records each static model's resolved dataclass fields to `configs-<label>.json` (rank 0 only, keyed by model). `benches/report.py` reads those files and renders a new `## Static model configuration` section with one sub-table per model (rows = fields, columns = labels). The `--lower-atol` CLI option — which only ever affected the static models — stops being recorded as a random hyperparameter.

**Tech Stack:** Python 3, pytest, `pytest-benchmark`, `dataclasses.asdict`, stdlib `json`. `benches/` is on the pytest pythonpath, so `report` imports directly in `tests/`.

## Global Constraints

- Python must pass the repo's pre-commit lints (ruff check + ruff format). Run `uv run pre-commit run --files <changed files>` before committing if available; otherwise `uv run ruff check <files>` and `uv run ruff format <files>`.
- Python docstrings use Google style.
- Conventional commits: `<type>(<scope>): <gitmoji> <description>`.
- No new files are created, so no license headers are needed.
- `benches/report.py` is standalone — it must not import from `benches/_builders.py` or `benches/conftest.py` (it is run as `python benches/report.py`). Model/field ordering in the report comes from the JSON, not from importing the dataclasses.

---

### Task 1: Render static model configs in the report

**Files:**
- Modify: `benches/report.py`
- Test: `tests/test_bench_report.py`

**Interfaces:**
- Consumes: nothing from other tasks (tests fabricate the `configs-*.json` input).
- Produces (read by no other task, but defines the on-disk contract Task 2 must write):
  - `configs-<label>.json` schema: a JSON object `{model_name: {field: value, ...}}`, e.g. `{"hubbard": {"num_sites": 60, ...}, "pauli": {"num_qubits": 127, ...}}`.
  - `report._load_configs(results_dir: Path) -> dict[str, dict]` → `{label: {model: {field: value}}}`.
  - `report._fmt_config_value(value: object) -> str`.
  - `report._static_config_section(labels: list[str], configs: dict[str, dict]) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_bench_report.py`, add a fabricator helper and new tests, and update the two `lower_atol` tests. Insert the helper near `_write_timings`:

```python
def _write_static_configs(results_dir: Path) -> None:
    (results_dir / "configs-np1.json").write_text(
        json.dumps(
            {
                "hubbard": {"num_sites": 60, "cutoff": 6, "lower_atol": 1e-5},
                "pauli": {"num_qubits": 127, "cutoff": 8, "lower_atol": 1e-4},
            }
        )
    )
```

Add these new tests:

```python
def test_fmt_config_value_formats_floats_compactly() -> None:
    assert report._fmt_config_value(1e-5) == "1e-05"
    assert report._fmt_config_value(1.0) == "1"
    assert report._fmt_config_value(0.2) == "0.2"
    assert report._fmt_config_value(60) == "60"
    assert report._fmt_config_value("up") == "up"


def test_build_report_includes_static_config(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    _write_static_configs(tmp_path)
    md = report.build_report(tmp_path)

    assert "## Static model configuration" in md
    assert "### hubbard" in md
    assert "### pauli" in md
    # Int field and compact-float field rendered in the right sub-tables.
    assert "| num_sites | 60 |" in md
    assert "| num_qubits | 127 |" in md
    assert "| lower_atol | 1e-05 |" in md
    # hubbard precedes pauli; the section precedes the picture sections.
    assert md.index("### hubbard") < md.index("### pauli")
    assert md.index("## Static model configuration") < md.index("## Heisenberg")


def test_static_config_section_absent_when_no_configs(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    md = report.build_report(tmp_path)
    assert "## Static model configuration" not in md
```

Replace the body of `test_build_report_includes_hyperparameters` so it no longer expects a `lower_atol` row (remove `"lower_atol": 0.0001` from the params dict and the `| lower_atol | 0.0001 |` assertion), and assert the row is gone:

```python
def test_build_report_includes_hyperparameters(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    (tmp_path / "params-np1.json").write_text(
        json.dumps(
            {
                "gen_length": 4,
                "obs_terms": 10000,
                "num_generators": 100,
                "num_modes": 128,
                "cutoff": 6,
                "seed": 0,
                "bench_rounds": 5,
            }
        )
    )
    md = report.build_report(tmp_path)

    assert "## Hyperparameters" in md
    assert "| num_generators | 100 |" in md
    assert "| cutoff | 6 |" in md
    # lower_atol is no longer a random hyperparameter (it is per-model now).
    assert "| lower_atol |" not in md
    # Hyperparameters sit between Configuration and the picture sections.
    assert md.index("## Hyperparameters") < md.index("## Heisenberg")
```

Delete `test_hyperparams_table_renders_none_lower_atol_as_default` entirely (the row no longer exists).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bench_report.py -q`
Expected: FAIL — `AttributeError: module 'report' has no attribute '_fmt_config_value'` (and `## Static model configuration` assertions failing).

- [ ] **Step 3: Implement the report changes**

In `benches/report.py`, drop `"lower_atol"` from `_PARAM_KEYS` so it reads:

```python
# Hyperparameter rows, in display order. Keys match params-*.json.
_PARAM_KEYS = (
    "gen_length",
    "obs_terms",
    "num_generators",
    "num_modes",
    "cutoff",
    "seed",
    "bench_rounds",
)
```

In `_hyperparams_table`, update the caption line (remove the `lower_atol` clause):

```python
        "Random-problem sizes and run knobs used for each label.",
```

Add a loader next to `_load_graphsize`:

```python
def _load_configs(results_dir: Path) -> dict[str, dict]:
    """Return ``{label: {model: config_fields}}`` from ``configs-*.json`` files."""
    configs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("configs-*.json")):
        label = path.stem.removeprefix("configs-")
        configs[label] = json.loads(path.read_text())
    return configs
```

Add the formatter and section renderers (place after `_graphsize_table`):

```python
# Static models in display order; any others sorted after.
_STATIC_MODEL_ORDER = ("hubbard", "pauli")


def _fmt_config_value(value: object) -> str:
    """Format a config field value compactly (floats via ``g``, else ``str``)."""
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _model_config_table(
    model: str, labels: list[str], configs: dict[str, dict]
) -> list[str]:
    """Render one static model's config sub-table (rows = fields, cols = labels)."""
    fields: list[str] = []
    for label in labels:
        for field in configs.get(label, {}).get(model, {}):
            if field not in fields:
                fields.append(field)
    lines = [
        f"### {model}",
        "",
        "| Parameter | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join(["---:"] * len(labels)) + " |",
    ]
    for field in fields:
        cells = []
        for label in labels:
            model_cfg = configs.get(label, {}).get(model, {})
            cells.append(
                _fmt_config_value(model_cfg[field]) if field in model_cfg else "—"
            )
        lines.append(f"| {field} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _static_config_section(labels: list[str], configs: dict[str, dict]) -> list[str]:
    """Render the static-model configuration section (one sub-table per model)."""
    present = {model for cfg in configs.values() for model in cfg}
    if not present:
        return []
    ordered = [m for m in _STATIC_MODEL_ORDER if m in present]
    ordered += sorted(present.difference(ordered))
    lines = [
        "## Static model configuration",
        "",
        "Resolved configuration of each static model for this run "
        "(``--lower-atol`` overrides the per-model truncation tolerance).",
        "",
    ]
    for model in ordered:
        lines += _model_config_table(model, labels, configs)
    return lines
```

In `build_report`, load configs, add them to the label union, and insert the section after Graph size:

```python
    timings = _load_timings(results_dir)
    memory = _load_memory(results_dir)
    metas = _load_metadata(results_dir)
    params = _load_params(results_dir)
    graphsize = _load_graphsize(results_dir)
    configs = _load_configs(results_dir)

    labels = sorted(
        set(timings)
        | set(memory)
        | set(metas)
        | set(params)
        | set(graphsize)
        | set(configs)
    )
```

and in the `lines = [...]` list, between `*_graphsize_table(...)` and the first `*_section(...)`:

```python
        *_graphsize_table(labels, graphsize),
        *_static_config_section(labels, configs),
        *_section("Heisenberg", labels, heisenberg_ops, time_cells, mem_cells),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bench_report.py -q`
Expected: PASS (all tests, including the new static-config tests).

- [ ] **Step 5: Lint**

Run: `uv run ruff check benches/report.py tests/test_bench_report.py && uv run ruff format benches/report.py tests/test_bench_report.py`
Expected: no errors; formatter makes no or trivial changes.

- [ ] **Step 6: Commit**

```bash
git add benches/report.py tests/test_bench_report.py
git commit -m "feat(benches): 📊 report static model configurations per label"
```

---

### Task 2: Record each static model's resolved config

**Files:**
- Modify: `benches/conftest.py`
- Modify: `benches/bench_monoprop.py:189-217` (the `test_static` test)

**Interfaces:**
- Consumes: the `configs-<label>.json` schema defined in Task 1 (`{model: {field: value}}`).
- Produces: a `record_model_config` pytest fixture returning a callable `(model: str, config: Any) -> None`; `test_static` calls it after resolving its config.

- [ ] **Step 1: Drop `lower_atol` from the recorded random hyperparameters**

In `benches/conftest.py`, remove `"lower_atol"` from `_RECORDED_OPTIONS` so it reads:

```python
_RECORDED_OPTIONS = (
    "gen_length",
    "obs_terms",
    "num_generators",
    "num_modes",
    "cutoff",
    "seed",
    "bench_rounds",
)
```

(The `--lower-atol` CLI option and the `lower_atol` fixture stay — they still drive the static models; they are simply no longer written to `params-*.json`.)

- [ ] **Step 2: Add the `asdict` import and a `Callable` type import**

In `benches/conftest.py`, add `from dataclasses import asdict` to the top-level imports (next to `import json` / `import os`), and add `Callable` to the `TYPE_CHECKING` block:

```python
if TYPE_CHECKING:
    from collections.abc import Callable

    from monoprop import MonomialPropagator
```

- [ ] **Step 3: Add the `record_model_config` fixture**

In `benches/conftest.py`, add this fixture (place it near `lower_atol`, before the `picture` fixture):

```python
@pytest.fixture
def record_model_config() -> Callable[[str, Any], None]:
    """Return a callable recording a static model's resolved config for the report.

    ``benches/run.py`` sets ``MONOPROP_BENCH_LABEL`` and
    ``MONOPROP_BENCH_RESULTS``; the recorder merges the model's dataclass fields
    into ``configs-<label>.json`` (one entry per model) so the report can show
    the configuration each static model actually ran with. The config is pure
    input and identical across ranks, so only rank 0 writes (mirroring
    :func:`_write_run_params`); the timing and memory passes overwrite
    identically.

    Returns:
        A callable ``record(model, config)`` taking the model name and its
        resolved (frozen) configuration dataclass.
    """

    def _record(model: str, config: Any) -> None:
        rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            return
        label = os.environ.get("MONOPROP_BENCH_LABEL")
        results = os.environ.get("MONOPROP_BENCH_RESULTS")
        if not label or not results:
            return
        path = Path(results, f"configs-{label}.json")
        data = json.loads(path.read_text()) if path.exists() else {}
        data[model] = asdict(config)
        path.write_text(json.dumps(data, indent=2))

    return _record
```

- [ ] **Step 4: Call the recorder from `test_static`**

In `benches/bench_monoprop.py`, add the fixture to the `test_static` signature and call it right after the config is resolved:

```python
@pytest.mark.bench
@pytest.mark.slow
@pytest.mark.parametrize("model", list(STATIC_MODELS))
def test_static(
    benchmark: object,
    bench_comm: Any,
    lower_atol: float | None,
    model: str,
    record_model_config: Any,
) -> None:
    """Benchmark a fixed in-place static simulation (Heisenberg picture)."""
    build_fn, config_cls, steps = STATIC_MODELS[model]
    config = config_cls()
    if lower_atol is not None:
        config = replace(config, lower_atol=lower_atol)
    record_model_config(model, config)
```

(The rest of `test_static` — `setup`, `run`, `benchmark.pedantic` — is unchanged.)

- [ ] **Step 5: Lint**

Run: `uv run ruff check benches/conftest.py benches/bench_monoprop.py && uv run ruff format benches/conftest.py benches/bench_monoprop.py`
Expected: no errors; formatter makes no or trivial changes.

- [ ] **Step 6: Integration-verify the recorder writes a configs file**

`benches/conftest` recorders are verified by running the suite (no unit test imports `conftest`). Run one static model (pauli is the cheapest) with the recording env vars set, into a scratch dir:

```bash
OUT=/tmp/claude-1000/-workspaces-monoprop/c4f6902e-b6a8-45c9-9a97-62e32c1b77b9/scratchpad/cfgcheck
mkdir -p "$OUT"
MONOPROP_BENCH_LABEL=verify MONOPROP_BENCH_RESULTS="$OUT" \
  uv run --group bench python -m pytest benches/bench_monoprop.py \
  -k "test_static and pauli" --benchmark-disable -o filterwarnings=default -q
cat "$OUT/configs-verify.json"
```

Expected: the test passes and `configs-verify.json` contains a `"pauli"` object with `num_qubits`, `cutoff`, `lower_atol`, etc.

- [ ] **Step 7: Commit**

```bash
git add benches/conftest.py benches/bench_monoprop.py
git commit -m "feat(benches): 🩺 record resolved static model configs per run"
```

---

### Task 3: End-to-end report render check

**Files:** none modified (verification only).

- [ ] **Step 1: Render the report against the recorded config**

Reuse the `configs-verify.json` from Task 2 (copy it alongside a timing file is not needed; just point the report at a dir that has it plus existing artifacts). Simplest: regenerate the real report and confirm no regressions, then render against the scratch dir:

```bash
# Real results dir still renders (no static configs recorded there yet -> section omitted).
uv run --group bench python benches/report.py benches/results >/dev/null && echo "real report OK"

# Scratch dir with the recorded pauli config renders the new section.
OUT=/tmp/claude-1000/-workspaces-monoprop/c4f6902e-b6a8-45c9-9a97-62e32c1b77b9/scratchpad/cfgcheck
uv run --group bench python benches/report.py "$OUT" | grep -A6 "Static model configuration"
```

Expected: "real report OK" prints, and the scratch render shows a `## Static model configuration` section with a `### pauli` sub-table.

- [ ] **Step 2: Full bench-report test pass**

Run: `uv run pytest tests/test_bench_report.py tests/test_bench_builders.py -q`
Expected: PASS (builders untouched; report tests green).

---

## Self-Review

- **Spec coverage:** Recording fixture (Task 2 §3–4) ✓; `configs-<label>.json` schema (Task 1 interface + Task 2) ✓; `_load_configs` (Task 1) ✓; `_static_config_section` per-model sub-tables, ordering, float formatting (Task 1) ✓; drop `lower_atol` from `_PARAM_KEYS` (Task 1) and `_RECORDED_OPTIONS` (Task 2) ✓; caption update (Task 1) ✓; label union includes configs (Task 1) ✓; section after Graph size (Task 1) ✓; tests new + updated + removed (Task 1) ✓; record every label / rank-0 only (Task 2 fixture) ✓; verification incl. builders still green (Task 3) ✓.
- **Placeholder scan:** none — every code step shows full code.
- **Type consistency:** `record_model_config` returns `Callable[[str, Any], None]`; `test_static` annotates the fixture param `Any` and calls `record_model_config(model, config)`. `_fmt_config_value`/`_load_configs`/`_static_config_section`/`_model_config_table` signatures match between definition (Task 1 Step 3) and tests (Task 1 Step 1).
