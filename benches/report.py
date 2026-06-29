# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge benchmark timing and memory results into a single Markdown report.

Reads the artifacts produced by the ``just bench*`` recipes from a results
directory and writes ``REPORT.md`` combining, per operation:

- wall-clock time (mean of the rounds) from ``pytest-benchmark`` JSON files
  named ``time-<label>.json`` (e.g. ``time-np1.json``, ``time-np4.json``), and
- peak physical memory (PSS) per operation from ``mem-<label>.json``.

``<label>`` is the run label, conventionally ``np<N>`` for an N-rank MPI run
(``np1`` being the serial baseline). Under MPI, timing JSON is written by rank 0
(the makespan) and the recorded memory is the per-rank PSS summed across ranks.

Usage::

    uv run --group bench python benches/report.py [results_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as pretty JSON to ``path``."""
    path.write_text(json.dumps(obj, indent=2))


def _label_header(first_col: str, labels: list[str]) -> list[str]:
    """Return the two header rows for a table whose columns are run labels.

    The first column (``first_col``) is left-aligned; the per-label value columns
    are right-aligned.
    """
    return [
        f"| {first_col} | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join(["---:"] * len(labels)) + " |",
    ]


def _load_by_label(results_dir: Path, prefix: str) -> dict[str, dict]:
    """Return ``{label: parsed_json}`` from ``<prefix>-*.json`` files."""
    return {
        path.stem.removeprefix(f"{prefix}-"): json.loads(path.read_text())
        for path in sorted(results_dir.glob(f"{prefix}-*.json"))
    }


def _load_timings(results_dir: Path) -> dict[str, dict[str, float]]:
    """Return ``{label: {op_key: mean_seconds}}`` from ``time-*.json`` files."""
    return {
        label: {
            b["fullname"].split("/")[-1]: b["stats"]["mean"]
            for b in data.get("benchmarks", [])
        }
        for label, data in _load_by_label(results_dir, "time").items()
    }


def _config_table(labels: list[str], metas: dict[str, dict]) -> list[str]:
    """Render the run-configuration table (one row per run label)."""
    if not metas:
        return []
    header = [
        "Label",
        "Ranks",
        "monoprop threads",
        "OMP threads",
        "Launcher",
        "CPUs",
        "Host",
    ]
    lines = ["## Configuration", "", "| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for label in labels:
        meta = metas.get(label, {})
        env = meta.get("env", {})
        cells = [
            label,
            str(meta.get("ranks", "—")),
            str(env.get("monoprop_NUM_THREADS", "default")),
            str(env.get("OMP_NUM_THREADS", "default")),
            f"`{meta['launcher']}`" if meta.get("launcher") else "—",
            str(meta.get("cpu_count", "—")),
            str(meta.get("hostname", "—")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


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


def _hyperparams_table(labels: list[str], params: dict[str, dict]) -> list[str]:
    """Render the random-problem hyperparameters (one column per run label)."""
    if not params:
        return []
    lines = [
        "## Hyperparameters",
        "",
        "Random-problem sizes and run knobs used for each label.",
        "",
        *_label_header("Parameter", labels),
    ]
    for key in _PARAM_KEYS:
        cells = []
        for label in labels:
            value = params.get(label, {}).get(key, "—")
            cells.append("default" if value is None else str(value))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _resting_footprint_table(labels: list[str], resting: dict[str, dict]) -> list[str]:
    """Render the built operator+graph's resting PSS per picture (one col per label).

    Distinct from the per-operation *peak* memory tables: this is the settled
    footprint after the build's transient buffers are released (gc + malloc_trim),
    so it exposes persistent-memory differences (index size, recomputed-vs-stored
    data) that the mid-build peak metric masks.
    """
    if not resting:
        return []
    pictures = [
        p for p in _PICTURES if any(p in resting.get(lbl, {}) for lbl in labels)
    ]
    lines = [
        "## Operator resting footprint (PSS)",
        "",
        "Resident physical memory of the built operator and propagation graph at "
        "rest -- measured after the build's transient buffers are freed "
        "(`gc.collect()` + `malloc_trim`), with the graph still alive. Unlike the "
        "per-operation peak memory below (a high-water mark reached mid-build), this "
        "settled footprint reveals persistent-memory changes that peak hides. It is "
        "the whole process's footprint, so it includes any sibling-picture graph the "
        "shared session still holds (a near-constant offset that cancels in "
        "label-to-label comparisons). Summed across ranks under MPI.",
        "",
        *_label_header("Picture", labels),
    ]
    for picture in pictures:
        cells = []
        for label in labels:
            num_bytes = resting.get(label, {}).get(picture)
            cells.append(_fmt_mem(num_bytes))
        lines.append(f"| {_PICTURE_NAMES[picture]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _operator_size_table(labels: list[str], sizes: dict[str, dict]) -> list[str]:
    """Render the number of terms reached per picture (one column per label)."""
    if not sizes:
        return []
    pictures = [p for p in _PICTURES if any(p in sizes.get(lbl, {}) for lbl in labels)]
    lines = [
        "## Operator size",
        "",
        "Number of terms in the evolved operator reached per picture. Under MPI the "
        "operator is partitioned across ranks, so the per-rank shard sizes are summed "
        "(allreduce) into the full count. The size is deterministic in the "
        "hyperparameters, so runs at the same hyperparameters report the same count "
        "regardless of rank or thread count.",
        "",
        *_label_header("Picture", labels),
    ]
    for picture in pictures:
        cells = []
        for label in labels:
            terms = sizes.get(label, {}).get(picture, {}).get("terms")
            cells.append(f"{terms:,}" if terms is not None else "—")
        lines.append(f"| {_PICTURE_NAMES[picture]} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


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
    lines = [f"### {model}", "", *_label_header("Parameter", labels)]
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
        "(``--hubbard-lower-atol`` / ``--pauli-lower-atol`` override each "
        "model's truncation tolerance).",
        "",
    ]
    for model in ordered:
        lines += _model_config_table(model, labels, configs)
    return lines


def _fmt_time(seconds: float | None) -> str:
    """Format a duration adaptively (us / ms / s)."""
    if seconds is None:
        return "—"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


def _fmt_mem(num_bytes: int | None) -> str:
    """Format a byte count as MiB."""
    if num_bytes is None:
        return "—"
    return f"{num_bytes / 1024 / 1024:.2f} MiB"


_PICTURES = ("heisenberg", "schrodinger")
_PICTURE_NAMES = {"heisenberg": "Heisenberg", "schrodinger": "Schrödinger"}


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


def _table(
    title: str,
    labels: list[str],
    ops: list[str],
    values: dict[str, dict[str, str]],
) -> list[str]:
    """Render one Markdown table (rows = operations, columns = run labels)."""
    lines = [f"### {title}", "", *_label_header("Operation", labels)]
    for op in ops:
        cells = [values.get(label, {}).get(op, "—") for label in labels]
        lines.append(f"| {_display_op(op)} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _section(
    name: str,
    labels: list[str],
    ops: list[str],
    time_cells: dict[str, dict[str, str]],
    mem_cells: dict[str, dict[str, str]],
) -> list[str]:
    """Render one picture section (Time + Memory), or nothing if empty."""
    if not ops:
        return []
    return [
        f"## {name}",
        "",
        *_table("Time", labels, ops, time_cells),
        *_table("Memory (PSS)", labels, ops, mem_cells),
    ]


def build_report(results_dir: Path) -> str:
    """Build the Markdown report string from the artifacts in ``results_dir``."""
    timings = _load_timings(results_dir)
    memory = _load_by_label(results_dir, "mem")
    metas = _load_by_label(results_dir, "meta")
    params = _load_by_label(results_dir, "params")
    operator_sizes = _load_by_label(results_dir, "opsize")
    resting = _load_by_label(results_dir, "memrest")
    configs = _load_by_label(results_dir, "configs")

    labels = sorted(
        set(timings)
        | set(memory)
        | set(metas)
        | set(params)
        | set(operator_sizes)
        | set(resting)
        | set(configs)
    )
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
        "memory is the peak physical footprint (PSS) during each operation, "
        "including structures already resident when it runs. Under MPI it is "
        "summed across ranks (true physical RAM, with shared library pages counted "
        "once), so it is not lower just because the operator is distributed.",
        "",
        *_config_table(labels, metas),
        *_hyperparams_table(labels, params),
        *_operator_size_table(labels, operator_sizes),
        *_resting_footprint_table(labels, resting),
        *_static_config_section(labels, configs),
        *_section("Heisenberg", labels, heisenberg_ops, time_cells, mem_cells),
        *_section("Schrödinger", labels, schrodinger_ops, time_cells, mem_cells),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write ``REPORT.md`` into the results directory."""
    argv = sys.argv[1:] if argv is None else argv
    results_dir = Path(argv[0]) if argv else RESULTS_DIR
    report = build_report(results_dir)
    out_path = results_dir / "REPORT.md"
    out_path.write_text(report)
    sys.stdout.write(report)
    sys.stdout.write(f"\nWrote {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
