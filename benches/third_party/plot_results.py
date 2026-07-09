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

import pylab as plt

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
runtimes_dict = data["runtimes"]
expvals_dict = data["expvals"]
memory_dict = data["memory"]

MIN_STEP = 5


def _filter_from_min_step(steps: list[int], values: list[float]) -> tuple[list[int], list[float]]:
    """Keep only the entries whose step is >= MIN_STEP."""
    filtered = [(step, value) for step, value in zip(steps, values) if step >= MIN_STEP]
    filtered_steps = [step for step, _ in filtered]
    filtered_values = [value for _, value in filtered]
    return filtered_steps, filtered_values


fig1, ax1 = plt.subplots()
runtime_step_range = step_range[1:]
for label, runtimes in runtimes_dict.items():
    color = colors[label]
    steps, values = _filter_from_min_step(runtime_step_range, runtimes)
    ax1.plot(steps, values, color=color, label=label)
ax1.set_xlabel("Trotter step")
ax1.set_ylabel("Time per step [s]")
ax1.set_yscale("log")
ax1.legend(fontsize=10)
ax1.grid(which="both", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.savefig(Path(__file__).parent / "runtime.png", dpi=150)

fig2, ax2 = plt.subplots()
for label, memory in memory_dict.items():
    color = colors[label]
    steps, values = _filter_from_min_step(step_range, memory)
    ax2.plot(steps, values, color=color, label=label)
ax2.set_xlabel("Trotter step")
ax2.set_ylabel("Memory per step [MB]")
ax2.set_yscale("log")
ax2.legend(fontsize=10)
ax2.grid(which="both", alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.savefig(Path(__file__).parent / "memory.png", dpi=150)
