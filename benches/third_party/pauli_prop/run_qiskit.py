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
import time

from pauli_prop import propagate_through_circuit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm

from _common import (
    RESULTS_FILE,
    RssPeakSampler,
    ensure_results_file,
    load_settings,
    update_results,
)

LABEL = "Qiskit pauli-prop"

settings = load_settings()
ensure_results_file(settings)

step_circ = QuantumCircuit(settings.nq)
for i, k in settings.grid_edges:
    step_circ.rzz(settings.theta_zz, i, k)
for i in range(settings.nq):
    step_circ.rz(settings.theta_z, i)
for i in range(settings.nq):
    step_circ.rx(settings.theta_x, i)

qiskit_obs = SparsePauliOp.from_sparse_list(
    [("ZZ", list(settings.obs_qubits), 1.0)], num_qubits=settings.nq
)

# Qiskit's propagate_through_circuit API has no weight-based cutoff, only a mandatory positive
# term budget, so we reuse monoprop's own term count at each step (already in results.json,
# since run_model.sh always runs run_monoprop.py first) to keep the comparison apples-to-apples.
with open(RESULTS_FILE) as file:
    monoprop_num_terms = json.load(file)["num_terms"].get("monoprop")
if monoprop_num_terms is None:
    raise RuntimeError(
        "run_qiskit.py needs monoprop's per-step term counts, which aren't in results.json yet — "
        "run run_monoprop.py (or run_model.sh) first."
    )

runtime: list[float] = []
memory: list[float] = []
expvals: list[float] = []
num_terms: list[int] = []

with RssPeakSampler() as sampler:
    for step_idx, _ in enumerate(tqdm(settings.step_range, desc=LABEL)):
        max_terms = monoprop_num_terms[step_idx]
        sampler.reset()
        t1 = time.perf_counter()
        qiskit_obs, _ = propagate_through_circuit(
            qiskit_obs,
            step_circ,
            max_terms=max_terms,
            atol=settings.lower_atol,
            frame="h",
        )
        qiskit_expval = float(qiskit_obs.coeffs[~qiskit_obs.paulis.x.any(axis=1)].sum())
        t2 = time.perf_counter()

        if step_idx > 0:
            runtime.append(t2 - t1)
        expvals.append(qiskit_expval)
        num_terms.append(len(qiskit_obs))
        memory.append(sampler.peak_mb())

update_results(
    LABEL, runtime=runtime, memory=memory, expvals=expvals, num_terms=num_terms
)
