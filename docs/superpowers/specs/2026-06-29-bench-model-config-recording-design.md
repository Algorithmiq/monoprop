# Benchmark model-configuration recording — design

**Date:** 2026-06-29
**Status:** Approved

## Problem

`benches/run.py` writes a Markdown `REPORT.md` with one column per run label. It
already records, per label, the **random** benchmark hyperparameters
(`params-<label>.json` → "Hyperparameters" table) and the graph sizes reached.

It does **not** record the configurations of the **static** models — `hubbard`
(`HubbardConfig`) and `pauli` (`KickedIsingConfig`) — even though those configs
can change from the command line: `--lower-atol` overrides their truncation
tolerance. So a reader cannot tell what `num_sites`, `cutoff`, `lower_atol`, etc.
a static run actually used.

Worse, the random Hyperparameters table currently carries a single `lower_atol`
row sourced from the `--lower-atol` CLI option. That option only affects the
static models, and the two static models have *different* defaults (`hubbard`
`1e-5`, `pauli` `1e-4`), so one shared row cannot represent them and is
misleading next to the random parameters.

## Goal

Record the resolved configuration of every model a run actually used, per label,
and surface the static model configs in the report. Remove the misleading shared
`lower_atol` row from the random Hyperparameters table.

## Non-goals

- Recording derived quantities (e.g. `HubbardConfig.num_qubits`, a property).
  Only the dataclass fields that were set are recorded.
- Changing how the random problem is recorded beyond dropping the `lower_atol`
  row (its sizes are already captured).
- Recording static configs per-rank. The config is pure input, identical on all
  ranks, so rank 0 records it once per label.

## Design

### 1. Recording (`benches/conftest.py`)

Add a fixture that returns a recorder callable:

```python
@pytest.fixture
def record_model_config():
    """Return a callable recording a static model's resolved config for the report."""
    def _record(model: str, config: object) -> None:
        # rank-0 + env guard (mirrors _record_graph_size / _write_run_params)
        # merge {model: dataclasses.asdict(config)} into configs-<label>.json
    return _record
```

Behaviour, mirroring the existing `_record_graph_size` / `_write_run_params`
helpers:

- No-op unless `MONOPROP_BENCH_LABEL` and `MONOPROP_BENCH_RESULTS` are set.
- No-op on ranks other than rank 0 (the config is identical across ranks; no
  serial-only guard — MPI labels record too, so their columns are populated,
  unlike graph size which is distributed and left blank for MPI).
- Reads the existing `configs-<label>.json` if present, merges
  `{model: dataclasses.asdict(config)}`, and writes it back (pretty JSON). The
  timing and memory passes overwrite identically, as `params-*.json` already
  does.

`bench_monoprop.py::test_static` resolves its config today:

```python
config = config_cls()
if lower_atol is not None:
    config = replace(config, lower_atol=lower_atol)
```

It gains the `record_model_config` fixture and calls
`record_model_config(model, config)` immediately after resolving `config`, before
the build. No other production code paths change.

#### `configs-<label>.json` schema

```json
{
  "hubbard": {
    "num_sites": 60, "hopping": 1.0, "interaction": -2.0,
    "chemical_potential": 0.0, "trotter_dt": 0.2, "trotter_steps": 29,
    "observable_site": 46, "observable_spin": "up",
    "neel_start_spin": "down", "cutoff": 6, "lower_atol": 1e-05
  },
  "pauli": {
    "num_qubits": 127, "num_layers": 20, "observable_qubit": 62,
    "theta": 0.7853981633974483, "coupling": 0.7853981633974483,
    "cutoff": 8, "lower_atol": 0.0001
  }
}
```

Field order is the dataclass declaration order, preserved by `asdict` and JSON.

### 2. Reporting (`benches/report.py`)

- `_load_configs(results_dir) -> dict[str, dict]`: returns
  `{label: {model: {field: value}}}` from `configs-*.json` (mirrors
  `_load_params` / `_load_graphsize`).
- `_static_config_section(labels, configs) -> list[str]`: renders a
  `## Static model configuration` section. For each model present — in a fixed
  preferred order (`hubbard`, `pauli`), any others sorted after — a `### <model>`
  sub-table with `| Parameter | <label>… |` columns and one row per field. Field
  rows are the union of fields across labels, in recorded (declaration) order.
  Returns `[]` when no configs exist (section omitted, like graph size).
- Value formatting `_fmt_config_value(v)`: `format(v, "g")` for floats
  (`1e-5`→`1e-05`, `1.0`→`1`, `0.7853981633974483`→`0.785398`), `str(v)`
  otherwise; missing field in a label → `—`.
- Remove `"lower_atol"` from `report._PARAM_KEYS` and from
  `conftest._RECORDED_OPTIONS`; update the Hyperparameters caption to drop the
  "``lower_atol`` applies to the static benchmarks" clause.
- Include `set(configs)` in the `labels` union in `build_report` so a
  config-only run still renders.
- Section placement: after Graph size, before the picture sections.

### 3. Testing (`tests/test_bench_report.py`)

`benches` is on the pytest pythonpath, so tests fabricate JSON in `tmp_path` and
call `report.build_report` (existing pattern).

- New `test_build_report_includes_static_config`: write a `configs-np1.json`
  with `hubbard` and `pauli`, assert `## Static model configuration`,
  `### hubbard`, `### pauli`, a sample int row (`| num_sites | 60 |`), and a
  float row (`| lower_atol | 1e-05 |`).
- New `test_static_config_section_omitted_when_absent` (or assert within an
  existing no-config test) that the section is absent with no `configs-*.json`.
- New unit test for `_fmt_config_value` covering float/int/str.
- Update `test_build_report_includes_hyperparameters`: drop the
  `| lower_atol | 0.0001 |` assertion; assert the `lower_atol` row is **absent**
  from the Hyperparameters table.
- Remove `test_hyperparams_table_renders_none_lower_atol_as_default` (the row no
  longer exists).

### Verification

- `uv run pytest tests/test_bench_report.py` green.
- Regenerate the report against a hand-built `configs-*.json` to confirm
  end-to-end rendering (`bench-smoke` skips the static models, so it cannot
  exercise this path).
- `uv run pytest tests/test_bench_builders.py` still green (builders unchanged).

## Files touched

- `benches/conftest.py` — `record_model_config` fixture; drop `lower_atol` from
  `_RECORDED_OPTIONS`.
- `benches/bench_monoprop.py` — `test_static` records its resolved config.
- `benches/report.py` — `_load_configs`, `_static_config_section`,
  `_fmt_config_value`; drop `lower_atol` row; caption update; label union.
- `tests/test_bench_report.py` — new tests; update/remove `lower_atol` tests.
