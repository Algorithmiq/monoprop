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
"""Run benchmark comparing to other Pauli Propagation simulators on Trotter-evolved Ising model."""

from __future__ import annotations

import json
import time

from monoprop import MonomialPropagator, jordan_wigner_basis_change
from monoprop.qiskit_conversion import from_qiskit_circuit, from_qiskit_operator
from pauli_prop import propagate_through_circuit
from ppvm import PauliSum
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm

########################### SETTINGS ##########################
h = 1.0
j = 1.5 * h

dt = 0.002 / h
tot_time = 1.0 / h
num_steps = int(tot_time / dt)

qubit_range = range(5, 121, 5)
max_pauli_weight = 8
lower_atol = 1e-8

results_file = "trotter_ising_results.json"
###############################################################

labels = [
    "monoprop",
    "QuEra ppvm",
    "Qiskit pauli-prop",
    "cuPauliProp",
]
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

    runtimes_dict["QuEra ppvm"].append(t2 - t1)

    # ------------------------------- pauli-prop -------------------------------
    t1 = time.perf_counter()
    evolved_obs, _ = propagate_through_circuit(
        obs, circ, max_terms=10_000, atol=lower_atol, frame="h"
    )
    expval = float(evolved_obs.coeffs[~evolved_obs.paulis.x.any(axis=1)].sum())
    t2 = time.perf_counter()
    runtimes_dict["Qiskit pauli-prop"].append(t2 - t1)


with open("trotter_ising_results.json", "w") as file:
    json.dump(
        {"qubit_range": list(qubit_range), "runtimes": runtimes_dict}, file, indent=4
    )
