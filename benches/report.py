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
- peak memory from ``pytest-memray`` binary dumps under ``memray-<label>/``.

``<label>`` is the run label, conventionally ``np<N>`` for an N-rank MPI run
(``np1`` being the serial baseline). Under MPI, timing JSON is written by rank 0
(the makespan) while memory is reported per rank, so the peak shown is the
maximum across ranks.

Usage::

    uv run --group bench python benches/report.py [results_dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
_HEX_PREFIX = re.compile(r"^[0-9a-f]{32}-")


def write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as pretty JSON to ``path``."""
    path.write_text(json.dumps(obj, indent=2))


def _op_key_from_fullname(fullname: str) -> str:
    """Normalise a pytest-benchmark ``fullname`` to ``file.py::test`` form."""
    return fullname.split("/")[-1]


def _op_key_from_bin(filename: str) -> str | None:
    """Recover ``file.py::test`` from a pytest-memray ``.bin`` filename."""
    stem = _HEX_PREFIX.sub("", filename.removesuffix(".bin"))
    if ".py-" not in stem:
        return None
    file_part, test_part = stem.rsplit(".py-", 1)
    return f"{file_part.split('-')[-1]}.py::{test_part}"


def _load_timings(results_dir: Path) -> dict[str, dict[str, float]]:
    """Return ``{label: {op_key: mean_seconds}}`` from ``time-*.json`` files."""
    timings: dict[str, dict[str, float]] = {}
    for path in sorted(results_dir.glob("time-*.json")):
        label = path.stem.removeprefix("time-")
        data = json.loads(path.read_text())
        timings[label] = {
            _op_key_from_fullname(b["fullname"]): b["stats"]["mean"]
            for b in data.get("benchmarks", [])
        }
    return timings


def _load_memory(results_dir: Path) -> dict[str, dict[str, int]]:
    """Return ``{label: {op_key: peak_bytes}}`` from ``memray-*/`` directories."""
    try:
        import memray  # noqa: PLC0415
    except ImportError:
        return {}

    memory: dict[str, dict[str, int]] = {}
    for bin_dir in sorted(results_dir.glob("memray-*")):
        if not bin_dir.is_dir():
            continue
        label = bin_dir.name.removeprefix("memray-")
        peaks: dict[str, int] = {}
        for bin_path in sorted(bin_dir.glob("*.bin")):
            op_key = _op_key_from_bin(bin_path.name)
            if op_key is None:
                continue
            peak = memray.FileReader(str(bin_path)).metadata.peak_memory
            # Max across ranks (one dump per rank under MPI).
            peaks[op_key] = max(peaks.get(op_key, 0), peak)
        if peaks:
            memory[label] = peaks
    return memory


def _load_metadata(results_dir: Path) -> dict[str, dict]:
    """Return ``{label: metadata}`` from ``meta-*.json`` files."""
    metas: dict[str, dict] = {}
    for path in sorted(results_dir.glob("meta-*.json")):
        label = path.stem.removeprefix("meta-")
        metas[label] = json.loads(path.read_text())
    return metas


def _load_params(results_dir: Path) -> dict[str, dict]:
    """Return ``{label: hyperparameters}`` from ``params-*.json`` files."""
    params: dict[str, dict] = {}
    for path in sorted(results_dir.glob("params-*.json")):
        label = path.stem.removeprefix("params-")
        params[label] = json.loads(path.read_text())
    return params


def _load_graphsize(results_dir: Path) -> dict[str, dict]:
    """Return ``{label: {picture: metrics}}`` from ``graphsize-*.json`` files."""
    sizes: dict[str, dict] = {}
    for path in sorted(results_dir.glob("graphsize-*.json")):
        label = path.stem.removeprefix("graphsize-")
        sizes[label] = json.loads(path.read_text())
    return sizes


def _load_configs(results_dir: Path) -> dict[str, dict]:
    """Return ``{label: {model: config_fields}}`` from ``configs-*.json`` files."""
    configs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("configs-*.json")):
        label = path.stem.removeprefix("configs-")
        configs[label] = json.loads(path.read_text())
    return configs


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
        "| Parameter | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join(["---:"] * len(labels)) + " |",
    ]
    for key in _PARAM_KEYS:
        cells = []
        for label in labels:
            value = params.get(label, {}).get(key, "—")
            cells.append("default" if value is None else str(value))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _graphsize_table(labels: list[str], sizes: dict[str, dict]) -> list[str]:
    """Render the number of terms reached per picture (one column per label)."""
    if not sizes:
        return []
    pictures = [p for p in _PICTURES if any(p in sizes.get(lbl, {}) for lbl in labels)]
    lines = [
        "## Graph size",
        "",
        "Number of terms in the evolved operator reached per picture. Recorded on "
        "serial runs (it is deterministic in the hyperparameters, so MPI columns "
        "are blank).",
        "",
        "| Picture | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join(["---:"] * len(labels)) + " |",
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
    lines = [f"### {title}", "", "| Operation | " + " | ".join(labels) + " |"]
    lines.append("| --- | " + " | ".join(["---:"] * len(labels)) + " |")
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
    """Render one picture section (Time + Peak memory), or nothing if empty."""
    if not ops:
        return []
    return [
        f"## {name}",
        "",
        *_table("Time", labels, ops, time_cells),
        *_table("Peak memory", labels, ops, mem_cells),
    ]


def build_report(results_dir: Path) -> str:
    """Build the Markdown report string from the artifacts in ``results_dir``."""
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
        *_hyperparams_table(labels, params),
        *_graphsize_table(labels, graphsize),
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
