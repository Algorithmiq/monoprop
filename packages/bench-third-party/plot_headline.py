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

"""Draw the headline figure: Pauli on the left, Majorana on the right.

Both panels are per-step wall-clock time on a log axis, so the vertical gap between two curves
reads directly as the speed-up. The two benchmarks are separate runs of separate models; the
panels share a figure, not an axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from palette import ENGINE_COLORS as COLORS

HERE = Path(__file__).parent

# The first Pauli steps are dominated by warm-up rather than by propagation; the first Majorana
# layers by an operator still too small to time. Both would otherwise spend decades of a log axis
# on points where nothing is being measured.
PAULI_MIN_STEP = 5
MAJORANA_MIN_LAYER = 4


def _panel(ax: plt.Axes, xlabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel("time per step [s]")
    ax.set_yscale("log")
    ax.set_title(title, fontsize="medium")
    ax.legend(fontsize=9)
    ax.grid(which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.get_major_locator().set_params(integer=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "headline_runtime.png")
    args = parser.parse_args()

    pauli = json.loads((HERE / "pauli_prop" / "results.json").read_text())
    majorana = json.loads((HERE / "majorana_prop" / "results.json").read_text())

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))

    steps = pauli["step_range"]
    for label, runtime in pauli["runtime"].items():
        points = [(s, v) for s, v in zip(steps, runtime) if s >= PAULI_MIN_STEP]
        left.plot(
            *zip(*points), color=COLORS[label], label=label, marker="o", markersize=4
        )
    _panel(left, "Trotter step", "Pauli: 144-qubit 2D transverse-field Ising")

    layers = majorana["step_range"]
    for label, runtime in majorana["runtime_seconds"].items():
        points = [(x, v) for x, v in zip(layers, runtime) if x >= MAJORANA_MIN_LAYER]
        right.plot(
            *zip(*points), color=COLORS[label], label=label, marker="o", markersize=4
        )
    _panel(
        right,
        "circuit layer",
        f"Majorana: {majorana['n_spinful_sites']}-site 1D Fermi-Hubbard",
    )

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
