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


def plot_metric(
    ax,
    step_range: list[int],
    metric_dict: dict[str, list[float]],
    ylabel: str,
    secondary_dict: dict[str, list[float]] | None = None,
) -> None:
    """Plot ``metric_dict[source]`` vs. ``step_range`` for each source onto ``ax``.

    ``secondary_dict``, where given, is drawn as a faint unlabeled line reusing each source's own
    color and linestyle (only reduced alpha, no marker, distinguishes it) — a different
    measurement of the same source, not a new series.
    """
    colors_by_source = {}
    for source, values in metric_dict.items():
        (line,) = ax.plot(step_range, values, STYLES.get(source, "-o"), label=source)
        colors_by_source[source] = line.get_color()
    if secondary_dict:
        for source, values in secondary_dict.items():
            linestyle = "--" if STYLES.get(source, "-o").startswith("--") else "-"
            ax.plot(
                step_range,
                values,
                linestyle=linestyle,
                color=colors_by_source.get(source),
                alpha=0.4,
                linewidth=1,
            )
    ax.set_xlabel("layers")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def plot_runtime_figure(
    step_range: list[int], runtimes: dict[str, list[float]], out: Path
) -> None:
    """Save the runtime on its own canvas, both axes linear.

    Runtime is the quantity that gets cited on its own, so it gets its own file rather than a
    quarter of the combined grid. Both axes are linear: seconds are read as seconds, so the
    time the Julia engine spends at the largest depth is drawn at its true multiple of
    monoprop's instead of being compressed into a decade's width. That flattens monoprop's own
    curve against the axis, which is the finding, not a defect of the axis.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    plot_metric(ax, step_range, runtimes, "time (seconds)")
    ax.set_ylim(bottom=0)
    ax.set_title("1D Hubbard runtime vs circuit depth", fontsize="medium")
    # On this axis monoprop's curve lies along the bottom, where a reader cannot tell 4 s from
    # 0 s. State the band's top so the flat line reads as small rather than as absent.
    monoprop = runtimes.get("monoprop")
    if monoprop:
        ax.annotate(
            f"monoprop: all points ≤ {max(monoprop):.1f} s",
            xy=(step_range[-1], max(monoprop)),
            xytext=(-6, 14),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


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

    native_memory_dict = data.get("native_memory_MB", {})
    plot_metric(
        axes[1, 0],
        step_range,
        data["memory_MB"],
        "memory (MB)",
        secondary_dict=native_memory_dict,
    )
    axes[1, 0].set_title("Memory vs layers")

    plot_metric(axes[1, 1], step_range, data["expectation_value"], "expectation value")
    axes[1, 1].set_title("Expectation value vs layers")

    fig.tight_layout()
    if native_memory_dict:
        fig.text(
            0.5,
            0.005,
            "Faint lines: each engine's own native memory accounting (reference only, not the plotted peak)",
            ha="center",
            fontsize=8,
            color="gray",
        )
    fig.savefig(args.output_dir / "majorana_results.png", bbox_inches="tight")

    plot_runtime_figure(
        step_range, data["runtime_seconds"], args.output_dir / "majorana_runtime.png"
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
