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

import time

from _common import HighWaterMark, init_results, load_settings, update_results
from monoprop import PauliPropagator
from monoprop.qiskit_conversion import from_qiskit_circuit, from_qiskit_operator
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm

LABEL = "monoprop"

settings = load_settings()
init_results(settings)

step_circ = QuantumCircuit(settings.nq)
for i, k in settings.grid_edges:
    step_circ.rzz(settings.theta_zz, i, k)
for i in range(settings.nq):
    step_circ.rz(settings.theta_z, i)
for i in range(settings.nq):
    step_circ.rx(settings.theta_x, i)

obs = SparsePauliOp.from_sparse_list(
    [("ZZ", list(settings.obs_qubits), 1.0)], num_qubits=settings.nq
)

mp_circ = from_qiskit_circuit(step_circ, initial_state=[])
mp_obs = from_qiskit_operator(obs)
mp = PauliPropagator(
    initial_operator=mp_obs,
    initial_state=mp_circ.initial_state,
    cutoff=settings.max_pauli_weight,
    lower_atol=settings.lower_atol,
)

runtime: list[float] = []
memory: list[float] = []
expvals: list[float] = []
num_terms: list[int] = []
native_memory: list[float] = []

for step_idx, _ in enumerate(tqdm(settings.step_range, desc=LABEL)):
    with HighWaterMark() as window:
        t1 = time.perf_counter()
        mp.propagate(mp_circ)
        expval = mp.expectation_value()
        t2 = time.perf_counter()

    if step_idx > 0:
        runtime.append(t2 - t1)
    expvals.append(expval)
    num_terms.append(mp.size())
    memory.append(window.peak_mb)
    native_memory.append(
        (mp._simulator.operator_memory_bytes() + mp._simulator.graph_memory_bytes())
        / 1024**2
    )

update_results(
    LABEL,
    runtime=runtime,
    memory=memory,
    expvals=expvals,
    num_terms=num_terms,
    native_memory=native_memory,
)
