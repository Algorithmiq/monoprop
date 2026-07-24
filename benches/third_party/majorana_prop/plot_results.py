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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_benchmark(path: Path, source: str) -> pd.DataFrame:
    """Load a benchmark JSONL file into a DataFrame of runtime/term-count rows."""
    df = pd.read_json(path, lines=True)
    df = df.rename(
        columns={
            "n_spinful_sites": "n_spin",
            "n_layers": "layers",
            "runtime_seconds": "seconds",
            "memory_MB": "memory",
            "final_overlap": "overlap",
        }
    )
    df["source"] = source
    return df[
        ["n_spin", "layers", "num_terms", "seconds", "memory", "overlap", "source"]
    ].sort_values(["n_spin", "layers"])


def plot_metric(ax, data: pd.DataFrame, metric: str, ylabel: str) -> None:
    """Plot ``metric`` vs. layers for each n_spin/source combination onto ``ax``."""
    styles = {"monoprop": "-o", "MajoranaPropagation.jl": "--x"}
    colors = plt.cm.tab10.colors
    for i, n_spin in enumerate(sorted(data["n_spin"].unique())):
        color = colors[i % len(colors)]
        for source, style in styles.items():
            subset = data[(data["n_spin"] == n_spin) & (data["source"] == source)]
            if subset.empty:
                continue
            ax.plot(
                subset["layers"],
                subset[metric],
                style,
                color=color,
                label=f"n={n_spin} ({source})",
            )
    ax.set_xlabel("layers")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="small", ncol=2)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monoprop-results",
        type=Path,
        default=Path("monoprop_hubbard1d_benchmark_results.jsonl"),
        help="Path to the monoprop benchmark results JSONL file.",
    )
    parser.add_argument(
        "--julia-results",
        type=Path,
        default=Path("julia_hubbard1d_benchmark_results.jsonl"),
        help="Path to the julia benchmark results JSONL file.",
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

    monoprop_df = load_benchmark(args.monoprop_results, "monoprop")
    julia_df = load_benchmark(args.julia_results, "MajoranaPropagation.jl")
    df = pd.concat([monoprop_df, julia_df], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_metric(axes[0, 0], df, "seconds", "time (seconds)")
    axes[0, 0].set_title("Runtime vs layers")

    plot_metric(axes[0, 1], df, "num_terms", "number of terms")
    axes[0, 1].set_title("Number of terms vs layers")

    plot_metric(axes[1, 0], df, "memory", "memory (MB)")
    axes[1, 0].set_title("Memory vs layers")

    plot_metric(axes[1, 1], df, "overlap", "final overlap")
    axes[1, 1].set_title("Final overlap vs layers")

    fig.tight_layout()
    fig.savefig(args.output_dir / "majorana_results.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
