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

For each run label the suite writes ``<label>.json`` (metadata, hyperparameters,
per-operation peak memory, per-picture sizes/footprints, model configs) and
pytest-benchmark writes ``time-<label>.json`` (timings). This reads every label
in a results directory and renders ``REPORT.md`` with one column per label, so
serial / MPI / thread variants sit side by side.

Usage::

    uv run --group bench python benches/report.py [results_dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

RESULTS_DIR = Path(__file__).parent / "results"

_PICTURES = ("heisenberg", "schrodinger")
_PICTURE_NAMES = {"heisenberg": "Heisenberg", "schrodinger": "Schrödinger"}


def _read_json(path: Path) -> Any:
    """Return the parsed JSON at ``path``, or ``None`` if empty/malformed.

    A run that fails before populating its JSON must not poison the report for
    the runs that did succeed, so bad artifacts are skipped with a warning.
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
    out = [
        f"| {first_col} | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join(["---:"] * len(labels)) + " |",
    ]
    for row_id, display in rows:
        cells = " | ".join(cell(label, row_id) for label in labels)
        out.append(f"| {display} | {cells} |")
    out.append("")
    return out


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


def _config_table(labels: list[str], results: dict[str, dict]) -> list[str]:
    """Render the run-configuration table (one row per run label)."""
    metas = {lbl: results.get(lbl, {}).get("meta", {}) for lbl in labels}
    if not any(metas.values()):
        return []
    header = ["Label", "Ranks", "monoprop threads", "CPUs", "Host"]
    lines = [
        "## Configuration",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for label in labels:
        meta = metas[label]
        cells = [
            label,
            str(meta.get("ranks", "—")),
            str(meta.get("monoprop_threads", "default")),
            str(meta.get("cpu_count", "—")),
            str(meta.get("hostname", "—")),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _model_config_section(labels: list[str], results: dict[str, dict]) -> list[str]:
    """Render the fixed-model configuration section (one sub-table per model)."""
    configs = {lbl: results.get(lbl, {}).get("configs", {}) for lbl in labels}
    models: list[str] = []
    for cfg in configs.values():
        for model in cfg:
            if model not in models:
                models.append(model)
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
        rows: list[tuple[Any, str]] = []
        for cfg in configs.values():
            for field in cfg.get(model, {}):
                if (field, field) not in rows:
                    rows.append((field, field))
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


def build_report(results_dir: Path) -> str:
    """Build the Markdown report string from the artifacts in ``results_dir``."""
    results = _load_results(results_dir)
    timings = _load_timings(results_dir)
    labels = sorted(set(results) | set(timings), key=_natural_key)

    def sec(name: str) -> dict[str, dict]:
        return {lbl: results.get(lbl, {}).get(name, {}) for lbl in labels}

    params, opsize, memrest, storage, memory = (
        sec("params"),
        sec("opsize"),
        sec("memrest"),
        sec("storage"),
        sec("mem"),
    )

    all_ops = sorted(
        {op for table in timings.values() for op in table}
        | {op for table in memory.values() for op in table}
    )
    if not labels or not all_ops:
        return (
            "# monoprop benchmark report\n\nNo results found. Run `just bench` first.\n"
        )

    # Hyperparameter keys come from the data (insertion order preserved), so there
    # is no duplicated key list to keep in sync with conftest.
    param_keys = next((list(params[lbl]) for lbl in labels if params.get(lbl)), [])
    pictures = _pictures_present(opsize, labels)
    storage_rows = [
        ((p, comp), f"{_PICTURE_NAMES[p]} / {comp}")
        for p in _pictures_present(storage, labels)
        for comp in ("operator", "graph")
    ]

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
                "Memory (PSS)",
                "",
                "Operation",
                ops,
                labels,
                lambda lbl, op: _fmt_mem(memory.get(lbl, {}).get(op)),
                level=3,
            ),
        ]

    lines = [
        "# monoprop benchmark report",
        "",
        f"Run labels: **{', '.join(labels)}**. Times are the mean over rounds; "
        "memory is the peak physical footprint (PSS) during each operation, summed "
        "across ranks under MPI (true physical RAM, shared library pages counted once).",
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
            "Operator resting footprint (PSS)",
            "Settled resident memory of the built operator + graph, after the "
            "build's transient buffers are freed (`gc.collect()` + `malloc_trim`).",
            "Picture",
            [(p, _PICTURE_NAMES[p]) for p in _pictures_present(memrest, labels)],
            labels,
            lambda lbl, p: _fmt_mem(memrest.get(lbl, {}).get(p)),
        ),
        *_section(
            "Storage breakdown: operator vs graph",
            "Structural memory of the built propagator (C++ capacity-based "
            "accounting, not PSS), split between the operator and the graph.",
            "Picture / component",
            storage_rows,
            labels,
            lambda lbl, key: _fmt_mem(storage.get(lbl, {}).get(key[0], {}).get(key[1])),
        ),
        *_model_config_section(labels, results),
        *ops_section("Heisenberg", "heisenberg"),
        *ops_section("Schrödinger", "schrodinger"),
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
