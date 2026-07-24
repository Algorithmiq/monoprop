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

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

STYLES = {"monoprop": "-o", "MajoranaPropagation.jl": "--x"}


def plot_metric(ax, step_range: list[int], metric_dict: dict[str, list[float]], ylabel: str) -> None:
    """Plot ``metric_dict[source]`` vs. ``step_range`` for each source onto ``ax``."""
    for source, values in metric_dict.items():
        ax.plot(step_range, values, STYLES.get(source, "-o"), label=source)
    ax.set_xlabel("layers")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).with_name("results.json"),
        help="Path to the shared benchmark results JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save the generated plots into.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively instead of only saving them.",
    )
    args = parser.parse_args()

    with args.results.open() as file:
        data = json.load(file)

    step_range = data["step_range"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_metric(axes[0, 0], step_range, data["runtime_seconds"], "time (seconds)")
    axes[0, 0].set_title(f"Runtime vs layers (n_spin={data['n_spinful_sites']})")

    plot_metric(axes[0, 1], step_range, data["num_terms"], "number of terms")
    axes[0, 1].set_title("Number of terms vs layers")

    plot_metric(axes[1, 0], step_range, data["memory_MB"], "memory (MB)")
    axes[1, 0].set_title("Memory vs layers")

    plot_metric(axes[1, 1], step_range, data["expectation_value"], "expectation value")
    axes[1, 1].set_title("Expectation value vs layers")

    fig.tight_layout()
    fig.savefig(args.output_dir / "majorana_results.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

