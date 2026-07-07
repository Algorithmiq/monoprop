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

import pylab as plt

colors = {
    "monoprop": "tab:purple",
    "QuEra ppvm": "tab:orange",
    "Qiskit pauli-prop": "tab:blue",
    "cuPauliProp (GPU)": "tab:green",
    "PauliPropagation.jl": "tab:red",
}

with open("trotter_ising_results.json") as file:
    data = json.load(file)
qubit_range = data["qubit_range"]
runtimes_dict = data["runtimes"]

fig1, ax1 = plt.subplots()
for label, runtimes in runtimes_dict.items():
    color = colors[label]
    ax1.plot(qubit_range, runtimes, marker=".", color=color, label=label)
ax1.set_xlabel("Num qubits")
ax1.set_ylabel("Runtime [s]")
ax1.legend(fontsize=10)
ax1.grid(which="both", alpha=0.3)
plt.savefig("trotter_ising_figure.png", dpi=150)
