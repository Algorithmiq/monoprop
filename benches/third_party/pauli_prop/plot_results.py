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

import json
from pathlib import Path

import matplotlib.pyplot as plt

# Only entries whose step is >= min_step are plotted to skip the early noisy steps
min_step = 5

colors = {
    "monoprop": "tab:purple",
    "QuEra ppvm": "tab:orange",
    "Qiskit pauli-prop": "tab:blue",
    "cuPauliProp (GPU)": "tab:green",
    "PauliPropagation.jl": "tab:red",
}

with open(Path(__file__).parent / "results.json") as file:
    data = json.load(file)
step_range = data["step_range"]
runtime_dict = data["runtime"]
memory_dict = data["memory"]
expvals_dict = data["expvals"]


def _filter_from_min_step(
    steps: list[int], values: list[float]
) -> tuple[list[int], list[float]]:
    """Keep only the entries whose step is >= min_step."""
    filtered = [(step, value) for step, value in zip(steps, values) if step >= min_step]
    filtered_steps = [step for step, _ in filtered]
    filtered_values = [value for _, value in filtered]
    return filtered_steps, filtered_values


def _style_axes(ax: plt.Axes, ylabel: str) -> None:
    """Apply the shared log-scale styling used by both plots."""
    ax.set_xlabel("Trotter step")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for label, runtime in runtime_dict.items():
    steps, values = _filter_from_min_step(step_range, runtime)
    ax1.plot(steps, values, color=colors[label], label=label)
_style_axes(ax1, "Time per step [s]")

for label, memory in memory_dict.items():
    steps, values = _filter_from_min_step(step_range, memory)
    ax2.plot(steps, values, color=colors[label], label=label)
_style_axes(ax2, "Memory per step [MB]")

fig.tight_layout()
fig.savefig(Path(__file__).parent / "results.png", dpi=150)
