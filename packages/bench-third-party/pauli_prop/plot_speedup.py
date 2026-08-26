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

"""Bar chart of monoprop's speed-up over every other backend, grouped by qubit count.

Reads the same scaling JSONL as plot_scaling.py and divides each backend's total
runtime by monoprop's at the same lattice size, so every bar is "monoprop is this many
times faster here". A backend contributes a bar only at the sizes where both it and
monoprop finished; where it stopped finishing, its slot in the group is empty.

The y axis is linear and starts at zero — the bar lengths are the ratios. That is why
the GPU bars look short next to the CPU ones: being 15x behind really is a fifteenth of
being 230x behind, and flattening that difference onto a log axis would hide the whole
point of the chart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_scaling import (  # noqa: E402
    COLORS,
    layers,
    load,
    provenance,
    truncation,
    warn_mixed_hosts,
)

BASELINE = "monoprop"
# Fixed order and fixed colors, shared with plot_scaling.py so a reader moving between
# the two figures keeps the same colour->backend mapping.
CHALLENGERS = [
    "PauliPropagation.jl",
    "QuEra ppvm",
    "Qiskit pauli-prop",
    "cuPauliProp (GPU)",
]
# Secondary encoding for the GPU series. Its pink and Qiskit's green are the one pair in
# this palette that sits in the 6-8 dE band under simulated protanopia/deuteranopia, so
# colour alone would not separate them; the hatch also carries the honest distinction
# that this bar is a different device class.
HATCH = {"cuPauliProp (GPU)": "//"}


def speedups(records: list[dict]) -> tuple[list[int], dict[str, dict[int, float]]]:
    """(sizes, {backend: {num_qubits: ratio}}) for every size monoprop finished.

    Sizes come from monoprop's own successful runs: without a baseline at a size there
    is no ratio to state, so that size is not a group in the chart at all.
    """
    totals: dict[str, dict[int, float]] = {}
    for r in records:
        if r.get("status") != "ok" or r.get("total_runtime_s") is None:
            continue
        totals.setdefault(r["label"], {})[r["num_qubits"]] = r["total_runtime_s"]

    base = totals.get(BASELINE, {})
    if not base:
        raise SystemExit(
            f"no successful {BASELINE} runs in the input: nothing to divide by"
        )
    ratios = {
        label: {n: t / base[n] for n, t in sizes.items() if n in base}
        for label, sizes in totals.items()
        if label != BASELINE
    }
    return sorted(base), ratios


def _fmt(ratio: float) -> str:
    return f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×"


def _plot(records: list[dict], ax) -> list[int]:
    sizes, ratios = speedups(records)
    present = [label for label in CHALLENGERS if ratios.get(label)]
    # One fixed slot per backend inside each group, whether or not it has a bar there,
    # so a gap reads as "this one did not get this far" rather than shifting its
    # neighbours into its place.
    slot = 0.8 / max(len(present), 1)
    for column, label in enumerate(present):
        offset = (column - (len(present) - 1) / 2) * slot
        xs = [i + offset for i, n in enumerate(sizes) if n in ratios[label]]
        ys = [ratios[label][n] for n in sizes if n in ratios[label]]
        # 2px of surface between neighbouring bars, as the mark spec asks.
        bars = ax.bar(
            xs,
            ys,
            width=slot * 0.86,
            color=COLORS[label],
            label=label,
            hatch=HATCH.get(label),
            edgecolor="white",
            linewidth=0.6,
        )
        # Every bar is labelled: at four series the labels are what carry identity for a
        # colourblind reader, and they are the only way the short GPU bars stay readable.
        ax.bar_label(
            bars,
            labels=[_fmt(y) for y in ys],
            padding=2,
            fontsize=6.5,
            rotation=90,
            color="#333333",
        )

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel("qubits")
    ax.set_ylabel(f"{BASELINE} speed-up  (other backend's total time / {BASELINE}'s)")
    ax.margins(y=0.16)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize="small", loc="upper left", frameon=False, ncols=2)
    return sizes


def _table(records: list[dict]) -> str:
    sizes, ratios = speedups(records)
    present = [label for label in CHALLENGERS if ratios.get(label)]
    header = "| qubits | " + " | ".join(present) + " |"
    rule = "| -----: | " + " | ".join("-----:" for _ in present) + " |"
    lines = [header, rule]
    for n in sizes:
        cells = [
            _fmt(ratios[label][n]) if n in ratios[label] else "-" for label in present
        ]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="scaling JSONL file(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    records = load(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for warning in warn_mixed_hosts(records):
        print(warning)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    _plot(records, ax)
    # Derived from the records, like plot_scaling.py's — and carrying no hardware either.
    ax.set_title(
        f"How many times faster {BASELINE} is, per system size — 2D TFIM, "
        f"{layers(records)}, {truncation(records)}",
        fontsize="medium",
    )
    fig.tight_layout()

    out = args.output_dir / "pauli_speedup.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    sidecar = f"{_table(records)}\n\nMeasured on:\n\n{provenance(records)}\n"
    (args.output_dir / "pauli_speedup.md").write_text(sidecar)
    print(sidecar)


if __name__ == "__main__":
    main()
