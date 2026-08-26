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

"""Plot total runtime and final operator memory against lattice size.

Reads the JSONL written by run_scaling.py (one or more files) and draws one curve per
backend, as two standalone figures — `pauli_scaling_runtime.png` and
`pauli_scaling_memory.png` — plus a `pauli_scaling.md` sidecar holding the same numbers
as a table and the provenance of each backend's points. A backend's curve simply ends at
the last size it completed; the runtime figure marks the timeout threshold when the result
records specify one.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed per backend so the runtime and memory figures, successive runs, and the speed-up
# chart in plot_speedup.py all keep the same colour->backend mapping.
COLORS = {
    "monoprop": "#0072B2",
    "QuEra ppvm": "#D55E00",
    "Qiskit pauli-prop": "#009E73",
    "cuPauliProp (GPU)": "#CC79A7",
    "PauliPropagation.jl": "#E69F00",
}
ORDER = list(COLORS)

# Backends whose operator lives in device memory. Host RSS does not describe their
# footprint at all -- it stays flat while the card fills up -- so the working-set column
# reads them off the device instead. Membership is a property of the backend, not of a
# run, which is why it is a constant rather than something inferred per record.
DEVICE_BACKENDS = {"cuPauliProp (GPU)"}

# What `final_memory_MB` must have been measured with for the default figure's axis label
# to be true. Kept in sync with `backends.HOST_MEMORY_METRIC` by `check_memory_metric`
# rather than imported, so this script stays runnable without the backend dependencies.
HOST_MEMORY_METRIC = "peak process RSS over the step (kernel VmHWM)"

# Columns whose value is only a host-RSS figure if the record was written by the current
# schema. An older runner put each library's own accounting in `final_memory_MB`, which is
# indistinguishable from a host figure once it is a bare float on an axis.
HOST_MEMORY_COLUMNS = {"final_memory_MB"}

# Per memory column: axis label and headline fragment. The label has to follow the column
# actually plotted -- a figure titled "final memory" whichever key was passed is how an
# operator-memory curve gets read as a host-RSS one.
MEMORY_COLUMNS = {
    "final_memory_MB": (
        "final-step peak host RSS [MB]",
        "final-step peak host RSS",
    ),
    "peak_rss_MB": (
        "whole-run peak host RSS [MB]",
        "whole-run peak host RSS",
    ),
    "operator_memory_MB": (
        "library's own accounting [MB]",
        "each library's own memory accounting",
    ),
    "working_set_MB": (
        "peak working memory [MB]   (CPU: host RSS | GPU: device + host)",
        "peak working memory, per engine's own memory pool",
    ),
}


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


def add_working_set(records: list[dict]) -> None:
    """Annotate each record with ``working_set_MB``: the peak memory it actually needed.

    Both inputs are exact high-water marks taken by this suite rather than figures a
    library reports about itself -- the kernel's ``VmHWM`` on the host, CuPy's allocator
    hook on the device -- so they answer the same question for engines that answer it in
    different pools. The GPU's host RSS is added to its device peak because that job needs
    both at once.

    One asymmetry survives and is not correctable here: host RSS includes pages the
    allocator has cached but not returned to the OS, while the device figure counts bytes
    in use and excludes cached-free pool blocks. The host side is therefore the slightly
    more generous of the two.
    """
    for r in records:
        host = r.get("final_memory_MB")
        if r.get("label") not in DEVICE_BACKENDS:
            r["working_set_MB"] = host
            continue
        device = r.get("operator_memory_MB")
        r["working_set_MB"] = None if device is None else device + (host or 0.0)


def check_memory_metric(records: list[dict], memory_key: str) -> list[str]:
    """Flag records whose `final_memory_MB` was not measured as host RSS.

    The axis label for this column asserts a specific instrument. A record written before
    `final_memory_MB` became the host high-water mark carries its library's own accounting
    there instead -- a smaller number, on a different pool, that plots perfectly happily
    under an RSS title. `memory_metric` is the only thing that distinguishes them, so it is
    checked rather than assumed: regenerate the sweep, or pass an explicit `--memory-key`.
    """
    if memory_key not in HOST_MEMORY_COLUMNS:
        return []
    stale: dict[str, str] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        metric = r.get("memory_metric", "")
        if metric != HOST_MEMORY_METRIC:
            stale.setdefault(r["label"], metric or "(unrecorded)")
    if not stale:
        return []
    detail = ", ".join(
        f"{label}: {metric!r}" for label, metric in sorted(stale.items())
    )
    return [
        (
            f"WARNING: `{memory_key}` is labelled {HOST_MEMORY_METRIC!r} but these "
            f"records were measured otherwise -- {detail}. The memory figure is "
            "mislabelled; re-run run_scaling.py, or plot --memory-key "
            "operator_memory_MB explicitly."
        )
    ]


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

    Built from every record, failed ones included, so both figures carry the same ticks
    whichever backends happen to reach which size.
    """
    grids = {
        r["nx"]: f"{r['nx']}x{r['ny']}" for r in sorted(records, key=lambda r: r["nx"])
    }
    return list(grids), list(grids.values())


def _timeout_limit(records: list[dict]) -> float | None:
    """Return the common timeout limit recorded by failed runs, if unambiguous."""
    limits = {
        float(match.group(1))
        for record in records
        if record.get("status") == "timeout"
        and (match := re.search(r"exceeded ([0-9.]+)s", record.get("detail", "")))
    }
    return limits.pop() if len(limits) == 1 else None


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


def _figure(
    records: list[dict],
    key: str,
    ylabel: str,
    headline: str,
    out: Path,
) -> None:
    """One standalone figure for one measured quantity.

    Runtime and memory are separate files rather than two panels of one image: they are read
    and cited separately, and a shared canvas forces a shared size and one title for two
    different claims. Each figure therefore repeats the model line under its own headline.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    _plot_axes(ax, records, key, ylabel)
    # Title = what was computed, and nothing about where: the machine goes in provenance().
    ax.set_title(
        f"{headline}\n{layers(records)}, {truncation(records)}", fontsize="medium"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _plot_axes(ax, records, key, ylabel) -> None:
    for label in ORDER:
        sides, values = _series(records, label, key)
        if not sides:
            continue
        # A curve ends at the last size the backend completed; per-size status remains in
        # the table and JSONL, while the common timeout threshold is shown below.
        ax.plot(sides, values, "o-", color=COLORS[label], label=label, markersize=5)
    if key == "total_runtime_s" and (timeout := _timeout_limit(records)) is not None:
        ax.axhline(timeout, color="#555555", linestyle="--", linewidth=1.2)
        ax.annotate(
            "Timeout limit",
            xy=(1, timeout),
            xycoords=("axes fraction", "data"),
            xytext=(-6, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize="small",
            color="#555555",
        )
    ticks, tick_labels = _grid_ticks(records)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("lattice size")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    # Outside the axes: the fastest backend's curve runs along the bottom-right, exactly where
    # a default legend lands, and it would hide the largest-size point.
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(0.0, -0.13), ncols=3)


def _table(records: list[dict], memory_key: str = "final_memory_MB") -> str:
    # The memory column must be the one the figure plots, or the sidecar silently
    # contradicts it -- host RSS next to a curve of device bytes, for instance.
    heading = MEMORY_COLUMNS[memory_key][0]
    lines = [
        f"| backend | qubits | grid | total s | {heading} | final terms | status |",
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
            value = r.get(memory_key)
            shown = "-" if value is None else f"{value:.1f}"
            lines.append(
                f"| {r['label']} | {r['num_qubits']} | {r['nx']}x{r['ny']} | "
                f"{r['total_runtime_s']:.3f} | {shown} | "
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
        choices=[
            "final_memory_MB",
            "operator_memory_MB",
            "peak_rss_MB",
            "working_set_MB",
        ],
        help="Which memory column to plot. The default is the peak process RSS over the "
        "final step, the same quantity for every backend. `working_set_MB` reads each "
        "engine against the pool its operator actually lives in (host RSS, or device + "
        "host for GPU backends), which is the comparison to use when a GPU engine is in "
        "the field. `operator_memory_MB` is each library's own accounting and is not "
        "comparable across backends (see OPERATOR_MEMORY_METRICS in backends.py).",
    )
    args = parser.parse_args()

    records = load(args.results)
    add_working_set(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for warning in warn_mixed_hosts(records):
        print(warning)
    for warning in check_memory_metric(records, args.memory_key):
        print(warning)

    _figure(
        records,
        "total_runtime_s",
        "total runtime [s]",
        "2D TFIM Trotter evolution: total runtime vs lattice size",
        args.output_dir / "pauli_scaling_runtime.png",
    )
    ylabel, headline = MEMORY_COLUMNS[args.memory_key]
    # The default column keeps the published filename; any other key gets its own, so a
    # second run cannot quietly replace the default figure with a different quantity.
    memory_out = (
        "pauli_scaling_memory.png"
        if args.memory_key == "final_memory_MB"
        else f"pauli_scaling_memory_{args.memory_key.removesuffix('_MB')}.png"
    )
    _figure(
        records,
        args.memory_key,
        ylabel,
        f"2D TFIM Trotter evolution: {headline} vs lattice size",
        args.output_dir / memory_out,
    )

    sidecar = (
        f"{_table(records, args.memory_key)}\n\nMeasured on:\n\n{provenance(records)}\n"
    )
    (args.output_dir / "pauli_scaling.md").write_text(sidecar)
    print(sidecar)


if __name__ == "__main__":
    main()
