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

"""Merge benchmark results into a single side-by-side Markdown report.

Each label writes ``<label>.json`` (metadata etc.) and pytest-benchmark writes
``time-<label>.json``. This renders ``REPORT.md`` with one column per label, so
serial / MPI / thread variants sit side by side.

Usage::

    monoprop-bench-report <results_dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tabulate import tabulate

if TYPE_CHECKING:
    from collections.abc import Callable

_PICTURES = ("heisenberg", "schrodinger")
_PICTURE_NAMES = {"heisenberg": "Heisenberg", "schrodinger": "Schrödinger"}


def _read_json(path: Path) -> Any:
    """Return the parsed JSON at ``path``, or ``None`` if empty/malformed.

    Bad artifacts are skipped with a warning so one failed run doesn't poison the
    report for the others.
    """
    text = path.read_text()
    if not text.strip():
        print(f"report: skipping empty artifact {path.name}", file=sys.stderr)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(
            f"report: skipping malformed artifact {path.name}: {exc}", file=sys.stderr
        )
        return None


def _natural_key(label: str) -> list[object]:
    """Sort key ordering embedded integers numerically (so ``t2`` < ``t10``)."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", label)]


def _load_results(results_dir: Path) -> dict[str, dict]:
    """Return ``{label: results}`` from the ``<label>.json`` artifacts."""
    out: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("time-"):
            continue
        data = _read_json(path)
        if data is not None:
            out[path.stem] = data
    return out


def _load_timings(results_dir: Path) -> dict[str, dict[str, float]]:
    """Return ``{label: {op_key: mean_seconds}}`` from ``time-*.json`` files."""
    out: dict[str, dict[str, float]] = {}
    for path in sorted(results_dir.glob("time-*.json")):
        data = _read_json(path)
        if data is None:
            continue
        out[path.stem.removeprefix("time-")] = {
            b["fullname"].split("/")[-1]: b["stats"]["mean"]
            for b in data.get("benchmarks", [])
        }
    return out


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
    return "—" if num_bytes is None else f"{num_bytes / 1024 / 1024:.2f} MiB"


def _fmt_config(value: object) -> str:
    """Format a config field value compactly (floats via ``g``, else ``str``)."""
    return format(value, "g") if isinstance(value, float) else str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a GitHub-flavoured Markdown table (cells are pre-formatted strings).

    ``disable_numparse`` keeps tabulate from re-parsing/realigning our already
    formatted cells (durations, MiB, thousands-separated counts).
    """
    return [
        tabulate(rows, headers=headers, tablefmt="github", disable_numparse=True),
        "",
    ]


def _table(
    first_col: str,
    rows: list[tuple[Any, str]],
    labels: list[str],
    cell: Callable[[str, Any], str],
) -> list[str]:
    """Render a label-columned Markdown table.

    ``rows`` are ``(row_id, display)`` pairs; ``cell(label, row_id)`` returns the
    formatted value for one cell.
    """
    body = [
        [display, *(cell(label, row_id) for label in labels)]
        for row_id, display in rows
    ]
    return _md_table([first_col, *labels], body)


def _section(
    title: str,
    caption: str,
    first_col: str,
    rows: list[tuple[Any, str]],
    labels: list[str],
    cell: Callable[[str, Any], str],
    level: int = 2,
) -> list[str]:
    """Render a titled, captioned section, or nothing when ``rows`` is empty."""
    if not rows:
        return []
    lines = ["#" * level + f" {title}", ""]
    if caption:
        lines += [caption, ""]
    return lines + _table(first_col, rows, labels, cell)


def _pictures_present(by_label: dict[str, dict], labels: list[str]) -> list[str]:
    """Return the pictures (in display order) that any label recorded."""
    return [p for p in _PICTURES if any(p in by_label.get(lbl, {}) for lbl in labels)]


def _picture_of(op_key: str) -> str:
    """Return the physical picture an op belongs to (defaults to Heisenberg)."""
    return "schrodinger" if "[schrodinger]" in op_key else "heisenberg"


def _display_op(op_key: str) -> str:
    """Turn a node id into a ``group / op`` label, dropping the picture tag.

    ``...::test_random_energy[heisenberg]`` -> ``random / energy``;
    ``...::test_model[hubbard]``            -> ``model / hubbard``.
    """
    _file, _, test = op_key.partition("::")
    base, _, param = test.removeprefix("test_").partition("[")
    param = param.rstrip("]")
    if param in _PICTURES:
        param = ""  # the section header already states the picture
    group, _, op = base.partition("_")
    if not op:  # parametrized-only name, e.g. "model" + param "hubbard"
        return f"{group} / {param}" if param else group
    return f"{group} / {op}"


def _fmt_cpus(meta: dict) -> str:
    """Render a run's CPU counts as ``logical/physical`` (``—`` when missing)."""
    logical = meta.get("cpu_count_logical", "—")
    physical = meta.get("cpu_count_physical", "—")
    return f"{logical}/{physical}"


def _fmt_row_store(meta: dict) -> str:
    """Render the row backend as ``asked → ran`` (the two differ whenever the setting is ``auto``)."""
    asked = meta.get("monoprop_row_store", "—")
    ran = meta.get("row_store_effective")
    if ran is None:
        return str(asked)
    return str(asked) if asked == ran else f"{asked} → {ran}"


def _config_table(labels: list[str], results: dict[str, dict]) -> list[str]:
    """Render the run-configuration table (one row per run label)."""
    metas = {lbl: results.get(lbl, {}).get("meta", {}) for lbl in labels}
    if not any(metas.values()):
        return []
    headers = [
        "Label",
        "Python",
        "nanobind",
        "Backend",
        "Ranks",
        "Nodes",
        "Ranks/node",
        "Partitions (requested)",
        "monoprop threads",
        "Row store",
        "CPUs (logical/physical)",
        "Host",
    ]
    rows = [
        [
            label,
            str(metas[label].get("python_version", "—")),
            str(metas[label].get("nanobind_version", "—")),
            str(metas[label].get("nanobind_backend_version", "—")),
            str(metas[label].get("ranks", "—")),
            str(metas[label].get("nodes", "—")),
            str(metas[label].get("ranks_per_node", "—")),
            str(metas[label].get("partitions_env", "—")),
            str(metas[label].get("monoprop_threads", "default")),
            _fmt_row_store(metas[label]),
            _fmt_cpus(metas[label]),
            str(metas[label].get("hostname", "—")),
        ]
        for label in labels
    ]
    return ["## Configuration", "", *_md_table(headers, rows)]


def _model_config_section(labels: list[str], results: dict[str, dict]) -> list[str]:
    """Render the fixed-model configuration section (one sub-table per model)."""
    configs = {lbl: results.get(lbl, {}).get("configs", {}) for lbl in labels}
    # Order-preserving dedup across labels (dict keys keep insertion order).
    models = list(dict.fromkeys(model for cfg in configs.values() for model in cfg))
    if not models:
        return []
    lines = [
        "## Model configuration",
        "",
        "Resolved configuration of each fixed model "
        "(override any field with `--<model>-<field>`).",
        "",
    ]
    for model in models:
        fields = dict.fromkeys(
            f for cfg in configs.values() for f in cfg.get(model, {})
        )
        rows = [(f, f) for f in fields]
        lines += _section(
            model,
            "",
            "Parameter",
            rows,
            labels,
            lambda lbl, f, m=model: (
                _fmt_config(configs[lbl][m][f]) if f in configs[lbl].get(m, {}) else "—"
            ),
            level=3,
        )
    return lines


def _per_unit(total: float | None, count: int | None, unit: str) -> str:
    """Format ``total / count`` with ``unit``, or ``—`` when either is missing."""
    if not total or not count:
        return "—"
    return f"{total / count:.1f} {unit}"


# Rows of the per-model tables: (row id, display). Cost and footprint per term are the two
# width-comparable quantities -- the models grow their term count with their mode count, so the
# totals beside them are not comparable across widths on their own.
_MODEL_ROWS: list[tuple[str, str]] = [
    ("terms", "Terms"),
    ("time", "Time (mean)"),
    ("cost_per_term", "Cost per term"),
    ("peak_rss", "Peak RSS"),
    ("base_rss", "Baseline RSS (pre-build)"),
    ("rest_rss", "Resting RSS"),
    ("op_bytes", "Operator accounting"),
    ("bytes_per_term", "Operator bytes per term"),
]


def _model_cells(
    data: dict, model: str, seconds: float | None, peak_rss: int | None
) -> dict[str, str]:
    """Return one label's formatted cells for ``model``, keyed as in :data:`_MODEL_ROWS`."""
    terms = data.get("opsize", {}).get(model, {}).get("terms")
    op_bytes = data.get("opmem", {}).get(model, {}).get("total_bytes")
    return {
        "terms": f"{terms:,}" if terms else "—",
        "time": _fmt_time(seconds),
        "cost_per_term": _per_unit(seconds * 1e9 if seconds else None, terms, "ns"),
        "peak_rss": _fmt_mem(peak_rss),
        "base_rss": _fmt_mem(data.get("membase", {}).get(model)),
        "rest_rss": _fmt_mem(data.get("memrest", {}).get(model)),
        "op_bytes": _fmt_mem(op_bytes),
        "bytes_per_term": _per_unit(op_bytes, terms, "B"),
    }


def _model_results_section(
    labels: list[str],
    results: dict[str, dict],
    timings: dict[str, dict[str, float]],
    memory: dict[str, dict],
    all_ops: list[str],
) -> list[str]:
    """Render one table of cost and footprint per fixed model.

    The random benchmarks are sized from the CLI and reported by picture above; the fixed models
    each own a mode count, so their numbers are collected per model instead.
    """
    # Order-preserving dedup across labels, as in _model_config_section.
    models = list(
        dict.fromkeys(
            model for lbl in labels for model in results.get(lbl, {}).get("configs", {})
        )
    )
    if not models:
        return []

    # The op key a model's timing and peak RSS are recorded under, e.g.
    # "bench_models.py::test_model[hubbard]" -- matched rather than reconstructed, so a renamed
    # test file does not silently blank the section.
    op_keys = {
        model: next(
            (op for op in all_ops if _display_op(op) == f"model / {model}"), None
        )
        for model in models
    }

    cells = {
        model: {
            label: _model_cells(
                results.get(label, {}),
                model,
                timings.get(label, {}).get(op_keys[model]) if op_keys[model] else None,
                memory.get(label, {}).get(op_keys[model]) if op_keys[model] else None,
            )
            for label in labels
        }
        for model in models
    }

    lines = [
        "## Fixed models",
        "",
        "Cost and footprint of each fixed model. `Time` is the mean over `--model-rounds`; "
        "`Cost per term` divides it by the evolved operator's term count. `Peak RSS` is the "
        "process high-water mark over the whole benchmark, `Baseline RSS` the resting footprint "
        "before the model is built, and `Operator accounting` monoprop's own C++ accounting of "
        "the evolved operator (`operator_memory_breakdown`), which is the only one of the three "
        "that excludes the interpreter and the transient build buffers.",
        "",
    ]
    for model in models:
        lines += _section(
            model,
            "",
            "Quantity",
            _MODEL_ROWS,
            labels,
            lambda lbl, row_id, m=model: cells[m][lbl].get(row_id, "—"),
            level=3,
        )
    return lines


def build_report(results_dir: Path) -> str:
    """Build the Markdown report string from the artifacts in ``results_dir``."""
    results = _load_results(results_dir)
    timings = _load_timings(results_dir)
    labels = sorted(set(results) | set(timings), key=_natural_key)

    def sec(name: str) -> dict[str, dict]:
        return {lbl: results.get(lbl, {}).get(name, {}) for lbl in labels}

    params, opsize, memrest, memory, memory_max = (
        sec("params"),
        sec("opsize"),
        sec("memrest"),
        sec("memhwm"),
        sec("memhwm_max"),
    )

    all_ops = sorted(
        {op for table in timings.values() for op in table}
        | {op for table in memory.values() for op in table}
        | {op for table in memory_max.values() for op in table}
    )
    if not labels or not all_ops:
        return (
            "# monoprop benchmark report\n\nNo results found. Run `just bench` first.\n"
        )

    # Hyperparameter keys come from the data, so there is no list to keep in sync
    # with conftest.
    param_keys = next((list(params[lbl]) for lbl in labels if params.get(lbl)), [])
    pictures = _pictures_present(opsize, labels)

    def ops_section(name: str, picture: str) -> list[str]:
        ops = [(op, _display_op(op)) for op in all_ops if _picture_of(op) == picture]
        if not ops:
            return []
        return [
            f"## {name}",
            "",
            *_section(
                "Time",
                "",
                "Operation",
                ops,
                labels,
                lambda lbl, op: _fmt_time(timings.get(lbl, {}).get(op)),
                level=3,
            ),
            *_section(
                "Memory (peak RSS, summed across ranks)",
                "",
                "Operation",
                ops,
                labels,
                lambda lbl, op: _fmt_mem(memory.get(lbl, {}).get(op)),
                level=3,
            ),
            *_section(
                "Memory (peak RSS, max across ranks)",
                "",
                "Operation",
                ops,
                labels,
                lambda lbl, op: _fmt_mem(memory_max.get(lbl, {}).get(op)),
                level=3,
            ),
        ]

    lines = [
        "# monoprop benchmark report",
        "",
        f"Run labels: **{', '.join(labels)}**. Times are the mean over rounds; "
        "memory is the kernel's exact peak resident footprint (`VmHWM`) during each "
        "operation, measured from a window reset and settled per operation. Under MPI "
        "the summed figure counts ranks peaking at different moments together (an "
        "upper bound on the job total); the max figure is the single worst rank.",
        "",
        *_config_table(labels, results),
        *_section(
            "Hyperparameters",
            "Random-problem sizes and run knobs used for each label.",
            "Parameter",
            [(k, k) for k in param_keys],
            labels,
            lambda lbl, k: (
                "default" if params.get(lbl, {}).get(k) is None else str(params[lbl][k])
            ),
        ),
        *_section(
            "Operator size",
            "Terms in the evolved operator per picture (summed across ranks under MPI).",
            "Picture",
            [(p, _PICTURE_NAMES[p]) for p in pictures],
            labels,
            lambda lbl, p: (
                f"{opsize[lbl][p]['terms']:,}" if p in opsize.get(lbl, {}) else "—"
            ),
        ),
        *_section(
            "Operator resting footprint (RSS)",
            "Settled resident memory of the built operator + graph, after the "
            "build's transient buffers are freed (`gc.collect()` + `heap_trim`).",
            "Picture",
            [(p, _PICTURE_NAMES[p]) for p in _pictures_present(memrest, labels)],
            labels,
            lambda lbl, p: _fmt_mem(memrest.get(lbl, {}).get(p)),
        ),
        *_model_config_section(labels, results),
        *_model_results_section(labels, results, timings, memory, all_ops),
        *ops_section("Heisenberg", "heisenberg"),
        *ops_section("Schrödinger", "schrodinger"),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write ``REPORT.md`` into the results directory."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        msg = "usage: monoprop-bench-report <results_dir>"
        raise SystemExit(msg)
    results_dir = Path(argv[0])
    report = build_report(results_dir)
    out_path = results_dir / "REPORT.md"
    out_path.write_text(report)
    sys.stdout.write(report)
    sys.stdout.write(f"\nWrote {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
