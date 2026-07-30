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

"""Plot total runtime and final operator memory against qubit count.

Reads the JSONL written by run_scaling.py (one or more files) and draws one curve per
backend. A backend's curve ends where it stopped completing points; the size it failed
at is marked with a hollow 'x' so a truncated curve cannot be mistaken for a finished one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed per backend so runtime and memory panels, and successive runs, stay comparable.
COLORS = {
    "monoprop": "#0072B2",
    "QuEra ppvm": "#D55E00",
    "Qiskit pauli-prop": "#009E73",
    "cuPauliProp (GPU)": "#CC79A7",
    "PauliPropagation.jl": "#E69F00",
}
ORDER = list(COLORS)


def load(paths: list[Path]) -> list[dict]:
    """Read the JSONL files, keeping one record per (backend, size).

    Later appends within a file win (a re-run supersedes the run it repeats), but the
    first *file* on the command line wins across files. That is what makes combining
    sides work: pass the CPU sweep before the GPU sweep and monoprop is taken from the
    CPU node, with only cuPauliProp picked up from the GPU node.
    """
    chosen: dict[tuple[str, int], dict] = {}
    claimed_by: dict[tuple[str, int], Path] = {}
    for path in paths:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record.get("label"), record.get("num_qubits"))
                if claimed_by.get(key, path) != path:
                    continue
                chosen[key] = record
                claimed_by[key] = path
    return list(chosen.values())


def warn_mixed_hosts(records: list[dict]) -> list[str]:
    """Flag any backend whose points were not all measured on the same machine.

    Combining sweeps is how a curve silently ends up half on one node and half on
    another — e.g. taking the large sizes from a GPU-node file because the CPU sweep had
    not reached them yet. Such a curve is not a scaling curve, so say so loudly.
    """
    seen: dict[str, set[tuple[str, int]]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        seen.setdefault(r["label"], set()).add((r.get("host"), r.get("threads")))
    warnings = []
    for label, hosts in seen.items():
        if len(hosts) > 1:
            detail = ", ".join(
                f"{host} ({threads} threads)"
                for host, threads in sorted(hosts, key=str)
            )
            warnings.append(f"WARNING: {label} mixes hosts: {detail}")
    return warnings


def _series(records: list[dict], label: str, key: str) -> tuple[list[int], list[float]]:
    """This backend's (lattice side, value) points, ordered by system size.

    The x coordinate is the lattice side, not the qubit count: the sweep steps the side by
    two, so plotting against N would bunch the small lattices together and stretch the large
    ones apart for what is a uniform ladder.
    """
    points = sorted(
        (r["num_qubits"], r["nx"], r[key])
        for r in records
        if r.get("label") == label
        and r.get("status") == "ok"
        and r.get(key) is not None
    )
    return [p[1] for p in points], [p[2] for p in points]


def _grid_ticks(records: list[dict]) -> tuple[list[int], list[str]]:
    """Tick positions (lattice side) and `nx x ny` labels, taken from the data itself.

    Built from every record, failed ones included, so all panels carry the same ticks
    whichever backends happen to reach which size.
    """
    grids = {
        r["nx"]: f"{r['nx']}x{r['ny']}" for r in sorted(records, key=lambda r: r["nx"])
    }
    return list(grids), list(grids.values())


def layers(records: list[dict]) -> str:
    """How many Trotter layers every curve carries, for the title.

    One layer is one `step_circuit` (all RZZ bonds, then RZ, then RX), applied once per point
    in `step_range` — so the count is `num_steps`, not `step_max`. Every backend must run the
    same depth for the comparison to mean anything; if a file ever mixes depths, say the range
    rather than quietly showing one.
    """
    depths = {r["num_steps"] for r in records if r.get("status") == "ok"}
    if len(depths) == 1:
        return f"{depths.pop()} Trotter layers"
    return f"{min(depths)}-{max(depths)} Trotter layers (MIXED)"


def truncation(records: list[dict]) -> str:
    """The truncation every curve was run under, for the title.

    Prefers the explicit `lower_atol` / `weight_cutoff` fields; falls back to parsing the
    `settings` prose so files written before those fields existed still label correctly. A
    weight cutoff equal to the qubit count cannot truncate anything and reads as none.
    """
    atols: set[str] = set()
    cutoffs: set[str] = set()
    for r in records:
        if r.get("status") != "ok":
            continue
        if "lower_atol" in r:
            atols.add(f"{r['lower_atol']:g}")
            cutoffs.add(
                "none" if r.get("weight_cutoff") is None else str(r["weight_cutoff"])
            )
            continue
        prose = r.get("settings", "")
        found_atol = re.search(r"atol=([0-9.eE+-]+)", prose)
        found_cut = re.search(r"weight cutoff=([0-9.]+|Inf)", prose)
        if found_atol:
            atols.add(f"{float(found_atol.group(1)):g}")
        if found_cut:
            raw = found_cut.group(1)
            binds = raw != "Inf" and float(raw) < r.get("num_qubits", float("inf"))
            cutoffs.add(raw if binds else "none")
    atol = atols.pop() if len(atols) == 1 else "MIXED"
    cutoff = cutoffs.pop() if len(cutoffs) == 1 else "MIXED"
    weight = "no weight cutoff" if cutoff == "none" else f"weight cutoff {cutoff}"
    return f"atol={atol}, {weight}"


def provenance(records: list[dict]) -> str:
    """Which machine, thread count and library version produced each backend's points.

    Deliberately not on the figure. A title states what was computed, which is the same
    wherever it runs; the host and the thread count qualify the *numbers*, so they belong
    next to them — here, and in every record of the JSONL these tables are built from.
    """
    rows: dict[str, tuple[str, object, object]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        # First ok record per backend: warn_mixed_hosts() is what catches a backend whose
        # points disagree, so one row per backend is enough here.
        rows.setdefault(
            r["label"],
            (r.get("host") or "?", r.get("threads"), r.get("library_version")),
        )
    lines = [
        "| backend | host | threads | version |",
        "| ------- | ---- | ------: | ------- |",
    ]
    for label in ORDER:
        if label not in rows:
            continue
        host, threads, version = rows[label]
        lines.append(f"| {label} | {host} | {threads or '?'} | {version or '-'} |")
    # The thread column is the cap the backend was *given*, which is not the same as the
    # cores it uses: ppvm and Qiskit propagate on one core whatever it is set to.
    lines.append("")
    lines.append(
        "`threads` is the cap each backend was given, not the cores it uses — "
        "QuEra ppvm and Qiskit pauli-prop propagate on a single core regardless "
        "(see ../README.md)."
    )
    return "\n".join(lines)


def _plot_panel(ax, records, key, ylabel, title) -> None:
    for label in ORDER:
        sides, values = _series(records, label, key)
        if not sides:
            continue
        # A curve simply ends at the last size the backend completed; the per-size status
        # (timeout / killed) is in the table and the JSONL, not drawn on the axes.
        ax.plot(sides, values, "o-", color=COLORS[label], label=label, markersize=5)
    ticks, tick_labels = _grid_ticks(records)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("lattice size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    # Outside the axes: the fastest backend's curve runs along the bottom-right, exactly where
    # a default legend lands, and it would hide the largest-size point.
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(0.0, -0.13), ncols=3)


def _table(records: list[dict]) -> str:
    lines = [
        "| backend | qubits | grid | total s | final MB | final terms | status |",
        "| ------- | -----: | ---- | ------: | -------: | ----------: | ------ |",
    ]
    for r in sorted(
        records,
        key=lambda r: (
            ORDER.index(r["label"]) if r.get("label") in ORDER else 99,
            r["num_qubits"],
        ),
    ):
        if r.get("status") == "ok":
            lines.append(
                f"| {r['label']} | {r['num_qubits']} | {r['nx']}x{r['ny']} | "
                f"{r['total_runtime_s']:.3f} | {r['final_memory_MB']:.1f} | "
                f"{r['final_num_terms']:,} | ok |"
            )
        else:
            lines.append(
                f"| {r['label']} | {r['num_qubits']} | {r['nx']}x{r['ny']} | - | - | - | "
                f"{r.get('status')} ({r.get('detail', '')}) |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="scaling JSONL file(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--memory-key",
        default="final_memory_MB",
        choices=["final_memory_MB", "operator_memory_MB", "peak_rss_MB"],
        help="Which memory column to plot. The default is each library's own final "
        "operator accounting, which is not the same quantity for every backend "
        "(see MEMORY_METRICS in backends.py).",
    )
    args = parser.parse_args()

    records = load(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for warning in warn_mixed_hosts(records):
        print(warning)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    _plot_panel(
        ax1, records, "total_runtime_s", "total runtime [s]", "Total runtime vs size"
    )
    _plot_panel(
        ax2, records, args.memory_key, "final memory [MB]", "Final memory vs size"
    )
    # Title = what was computed, and nothing about where: the machine goes in provenance().
    fig.suptitle(
        f"2D TFIM Trotter evolution: scaling with lattice size, {layers(records)}, "
        f"{truncation(records)}",
        fontsize="medium",
    )
    fig.tight_layout()
    out = args.output_dir / "pauli_scaling.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    sidecar = f"{_table(records)}\n\nMeasured on:\n\n{provenance(records)}\n"
    (args.output_dir / "pauli_scaling.md").write_text(sidecar)
    print(sidecar)


if __name__ == "__main__":
    main()
