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
    "cuPauliProp (GPU)": "tab:green",
    "PauliPropagation.jl": "tab:red",
}

with open(Path(__file__).parent / "results.json") as file:
    data = json.load(file)
step_range = data["step_range"]
runtimes_dict = data["runtimes"]
expvals_dict = data["expvals"]

fig1, ax = plt.subplots()
runtime_step_range = step_range[1:]
for label, runtimes in runtimes_dict.items():
    color = colors[label]
    ax.plot(runtime_step_range, runtimes, marker=".", color=color, label=label)
ax.set_xlabel("Trotter step")
ax.set_ylabel("Time per step [s]")
ax.set_yscale("log")
ax.legend(fontsize=10)
ax.grid(which="both", alpha=0.3)
plt.savefig(Path(__file__).parent / "runtime.png", dpi=150)
