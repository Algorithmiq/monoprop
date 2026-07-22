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
from pathlib import Path

import cupy as cp
import numpy as np
import psutil
from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliExpansionOptions,
    PauliRotationGate,
    Truncation,
    get_num_packed_integers,
)
from monoprop import PauliPropagator
from monoprop.qiskit_conversion import from_qiskit_circuit, from_qiskit_operator
from pauli_prop import propagate_through_circuit
from ppvm import PauliSum
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm


def _pauli_string_to_packed_integers(
    paulis: list[str], qubits: list[int], num_qubits: int
) -> np.ndarray:
    """Pack a Pauli string into the (x, z) bitfield layout expected by cuPauliProp."""
    num_packed_ints = get_num_packed_integers(num_qubits)
    out = np.zeros(num_packed_ints * 2, dtype=np.uint64)
    x_ptr = out[:num_packed_ints]
    z_ptr = out[num_packed_ints:]
    for pauli, qubit in zip(paulis, qubits):
        int_ind = qubit // 64
        bit_ind = qubit % 64
        if pauli in ("X", "Y"):
            x_ptr[int_ind] |= 1 << bit_ind
        if pauli in ("Z", "Y"):
            z_ptr[int_ind] |= 1 << bit_ind
    return out


def _grid_edges(nx: int, ny: int) -> list[tuple[int, int]]:
    """Nearest-neighbor edges of an nx-by-ny grid, row-major qubit indexing."""
    edges = []
    for row in range(ny):
        for col in range(nx):
            idx = row * nx + col
            if col + 1 < nx:
                edges.append((idx, idx + 1))
            if row + 1 < ny:
                edges.append((idx, idx + nx))
    return edges


# --- Simulation parameters (shared by every engine, see settings.json) ---
with open(Path(__file__).parent / "settings.json") as settings_file:
    settings = json.load(settings_file)

nx, ny = settings["nx"], settings["ny"]
nq = nx * ny
hx = settings["hx"]
hz = settings["hz"]
j = settings["j"]
dt = settings["dt"]

step_range = range(settings["step_min"],
                   settings["step_max"] + 1,
                   settings["step_size"])
lower_atol = settings["lower_atol"]
max_pauli_weight = nq if settings["cutoff"] is None else settings["cutoff"]
obs_qubits = tuple(settings["obs_qubits"])

labels = [
    "monoprop",
    "QuEra ppvm",
    "Qiskit pauli-prop",
    "cuPauliProp (GPU)",
]
runtime_dict = {label: [] for label in labels}
expvals_dict = {label: [] for label in labels}
num_terms_dict = {label: [] for label in labels}
memory_dict = {label: [] for label in labels}

process = psutil.Process()
ppvm_mem_bytes = 0
qiskit_mem_bytes = 0

theta_x = dt * hx
theta_z = dt * hz
theta_zz = dt * j
grid_edges = _grid_edges(nx, ny)

# One Trotter step of the tilted TFIM: ZZ couplings, then the Z field, then the X field.
step_circ = QuantumCircuit(nq)
for i, k in grid_edges:
    step_circ.rzz(theta_zz, i, k)
for i in range(nq):
    step_circ.rz(theta_z, i)
for i in range(nq):
    step_circ.rx(theta_x, i)

obs = SparsePauliOp.from_sparse_list([("ZZ", list(obs_qubits), 1.0)], num_qubits=nq)

# --- monoprop ---
mp_circ = from_qiskit_circuit(step_circ, initial_state=[])
mp_obs = from_qiskit_operator(obs)
mp = PauliPropagator(
    initial_operator=mp_obs,
    quantum_circuit=mp_circ,
    cutoff_type="support",
    cutoff=max_pauli_weight,
    lower_atol=lower_atol,
)

# --- QuEra ppvm ---
ppvm_obs = PauliSum.new(
    n_qubits=nq,
    terms=[f"Z{obs_qubits[0]}Z{obs_qubits[1]}"],
    min_abs_coeff=lower_atol,
    max_pauli_weight=max_pauli_weight,
)

# --- Qiskit pauli-prop (reuses the qiskit circuit and observable built above) ---
qiskit_obs = obs

# --- cuPauliProp (GPU, via cuquantum) ---
cupp_handle = LibraryHandle()

num_packed = get_num_packed_integers(nq)
cupp_xz = cp.zeros((1, 2 * num_packed), dtype=cp.uint64)
cupp_coefs = cp.ones((1,), dtype=cp.float64)
cupp_xz[0] = cp.asarray(
    _pauli_string_to_packed_integers(["Z", "Z"], list(obs_qubits), nq)
)

cupp_expansion = PauliExpansion(
    library_handle=cupp_handle,
    num_qubits=nq,
    num_terms=1,
    xz_bits=cupp_xz,
    coeffs=cupp_coefs,
    options=PauliExpansionOptions(memory_limit="80%", blocking=True),
)
cupp_truncation = Truncation(
    pauli_coeff_cutoff=lower_atol,
    pauli_weight_cutoff=max_pauli_weight,
)

cupp_step_gates = [
    PauliRotationGate(theta_zz, ["Z", "Z"], [i, k]) for i, k in grid_edges
]
cupp_step_gates += [PauliRotationGate(theta_z, ["Z"], [i]) for i in range(nq)]
cupp_step_gates += [PauliRotationGate(theta_x, ["X"], [i]) for i in range(nq)]

for step_idx, _ in enumerate(tqdm(step_range, desc="Running simulations")):
    # --- monoprop --- (exposes its own C++ operator-memory accounting)
    t1 = time.perf_counter()
    mp.propagate(evolve_with_coeffs=True)
    expval = mp.expectation_value()
    t2 = time.perf_counter()
    if step_idx > 0:
        runtime_dict["monoprop"].append(t2 - t1)
    expvals_dict["monoprop"].append(expval)
    num_terms_dict["monoprop"].append(mp.size())
    memory_dict["monoprop"].append(mp._simulator.operator_memory_bytes() / 1024**2)

    # --- QuEra ppvm --- (no memory accounting exposed: approximate via RSS growth over this step)
    mem_before = process.memory_info().rss
    t1 = time.perf_counter()
    for i, k in grid_edges:
        ppvm_obs.rzz(i, k, theta_zz)
    for i in range(nq):
        ppvm_obs.rz(i, theta_z)
    for i in range(nq):
        ppvm_obs.rx(i, theta_x)
    ppvm_expval = ppvm_obs.overlap_with_zero()
    t2 = time.perf_counter()
    ppvm_mem_bytes += max(0, process.memory_info().rss - mem_before)

    if step_idx > 0:
        runtime_dict["QuEra ppvm"].append(t2 - t1)
    expvals_dict["QuEra ppvm"].append(ppvm_expval)
    num_terms_dict["QuEra ppvm"].append(len(ppvm_obs))
    memory_dict["QuEra ppvm"].append(ppvm_mem_bytes / 1024**2)

    # --- Qiskit pauli-prop --- (max_terms tracks monoprop's own term count at this step, since
    # this API has no weight-based cutoff, only a mandatory positive term budget)
    mem_before = process.memory_info().rss
    max_terms = mp.size()
    t1 = time.perf_counter()
    qiskit_obs, _ = propagate_through_circuit(
        qiskit_obs, step_circ, max_terms=max_terms, atol=lower_atol, frame="h"
    )
    qiskit_expval = float(qiskit_obs.coeffs[~qiskit_obs.paulis.x.any(axis=1)].sum())
    t2 = time.perf_counter()
    qiskit_mem_bytes += max(0, process.memory_info().rss - mem_before)

    if step_idx > 0:
        runtime_dict["Qiskit pauli-prop"].append(t2 - t1)
    expvals_dict["Qiskit pauli-prop"].append(qiskit_expval)
    num_terms_dict["Qiskit pauli-prop"].append(len(qiskit_obs))
    memory_dict["Qiskit pauli-prop"].append(qiskit_mem_bytes / 1024**2)

    # --- cuPauliProp (GPU) --- (exposes its own cupy device-memory pool accounting)
    t1 = time.perf_counter()
    for gate in reversed(cupp_step_gates):
        cupp_expansion = cupp_expansion.apply_gate(
            gate,
            truncation=cupp_truncation,
            adjoint=True,
            sort_order=None,
            keep_duplicates=False,
        )
    trace_significand, trace_exponent = cupp_expansion.trace_with_zero_state()
    cupp_expval = float(trace_significand * np.exp2(trace_exponent))
    t2 = time.perf_counter()
    if step_idx > 0:
        runtime_dict["cuPauliProp (GPU)"].append(t2 - t1)
    expvals_dict["cuPauliProp (GPU)"].append(cupp_expval)
    num_terms_dict["cuPauliProp (GPU)"].append(cupp_expansion.num_terms)
    memory_dict["cuPauliProp (GPU)"].append(
        cp.get_default_memory_pool().used_bytes() / 1024**2
    )

with open(Path(__file__).parent / "results.json", "w") as file:
    json.dump(
        {
            "step_range": list(step_range),
            "num_terms": num_terms_dict,
            "runtime": runtime_dict,
            "memory": memory_dict,
            "expvals": expvals_dict,
        },
        file,
        indent=4,
    )
