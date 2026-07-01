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
"""Benchmark comparing to other Pauli Propagation simulators on Trotter-evolved Ising model."""

from __future__ import annotations

import time

import pylab as plt
from ppvm import PauliSum
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm

from monoprop import MonomialPropagator, jordan_wigner_basis_change
from monoprop.qiskit_conversion import from_qiskit_circuit, from_qiskit_operator

########################### SETTINGS ##########################
h = 1.0
j = 1.5 * h

dt = 0.002 / h
tot_time = 1.0 / h
num_steps = int(tot_time / dt)

qubit_range = range(5, 121, 5)
sv_max_qubits = 20
max_pauli_weight = 8
lower_atol = 1e-8
###############################################################

labels = [
    "monoprop",
    "ppvm",
]
colors = {
    "monoprop": "tab:green",
    "ppvm": "tab:orange",
}
runtimes_dict = {label: [] for label in labels}

for nq in tqdm(qubit_range, desc="Running simulations", ncols=80):
    # -------------------------------- monoprop --------------------------------
    # define qiskit circuit
    circ = QuantumCircuit(nq)
    theta_x = dt * h
    theta_zz = dt * j
    for _ in range(num_steps):
        for i in range(nq):
            circ.rx(theta_x, i)
        for i in range(nq - 1):
            circ.rzz(theta_zz, i, i + 1)

    # define qiskit observable
    obs = SparsePauliOp.from_sparse_list(
        [("Z", [i], 1.0) for i in range(nq)], num_qubits=nq
    )

    mp_circ = from_qiskit_circuit(circ, initial_state=[])
    mp_obs = from_qiskit_operator(obs)
    mp = MonomialPropagator(
        initial_operator=mp_obs,
        quantum_circuit=mp_circ,
        cutoff_type="support",
        cutoff=max_pauli_weight,
        lower_atol=lower_atol,
        basis_change=jordan_wigner_basis_change(nq),
    )
    t1 = time.perf_counter()
    mp.propagate(evolve_with_coeffs=True)
    expval = mp.expectation_value()
    t2 = time.perf_counter()
    runtimes_dict["monoprop"].append(t2 - t1)

    # ---------------------------------- ppvm ----------------------------------
    # define observable
    ppvm_obs = PauliSum.new(
        n_qubits=nq,
        terms=[f"Z{i}" for i in range(nq)],
        min_abs_coeff=lower_atol,
        max_pauli_weight=max_pauli_weight,
    )

    # evolve observable
    t1 = time.perf_counter()
    for _ in range(num_steps):
        for i in range(nq - 1):
            ppvm_obs.rzz(i, i + 1, theta_zz)
        for i in range(nq):
            ppvm_obs.rx(i, theta_x)
    ppvm_expval = ppvm_obs.overlap_with_zero()
    t2 = time.perf_counter()

    runtimes_dict["ppvm"].append(t2 - t1)


fig1, ax1 = plt.subplots()
for label in labels:
    color = colors[label]
    ax1.plot(qubit_range, runtimes_dict[label], marker=".", color=color, label=label)
ax1.set_xlabel("Num qubits")
ax1.set_ylabel("Runtime [s]")
ax1.legend(fontsize=12)
ax1.grid(which="both", alpha=0.3)
plt.savefig("trotter_ising.png", dpi=150)
