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

import cupy as cp
import numpy as np
from cuquantum.pauliprop.experimental import (
    LibraryHandle,
    PauliExpansion,
    PauliExpansionOptions,
    PauliRotationGate,
    Truncation,
    get_num_packed_integers,
)
from monoprop import MonomialPropagator, jordan_wigner_basis_change
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
    "cuPauliProp (GPU)",
]
runtimes_dict = {label: [] for label in labels}
expvals_dict = {label: [] for label in labels}

cupp_handle = LibraryHandle()

for nq in tqdm(qubit_range, desc="Running simulations", ncols=80):
    # -------------------------------- monoprop ---------------------------------
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
    expvals_dict["monoprop"].append(expval)

    # ---------------------------------- ppvm -----------------------------------
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
    expvals_dict["QuEra ppvm"].append(ppvm_expval)

    # ------------------------------- pauli-prop --------------------------------
    t1 = time.perf_counter()
    evolved_obs, _ = propagate_through_circuit(
        obs, circ, max_terms=10_000, atol=lower_atol, frame="h"
    )
    expval = float(evolved_obs.coeffs[~evolved_obs.paulis.x.any(axis=1)].sum())
    t2 = time.perf_counter()
    runtimes_dict["Qiskit pauli-prop"].append(t2 - t1)
    expvals_dict["Qiskit pauli-prop"].append(expval)

    # ------------------------------- cuPauliProp -------------------------------
    # define observable
    num_packed = get_num_packed_integers(nq)
    cupp_xz = cp.zeros((nq, 2 * num_packed), dtype=cp.uint64)
    cupp_coefs = cp.ones((nq,), dtype=cp.float64)
    for i in range(nq):
        cupp_xz[i] = cp.asarray(_pauli_string_to_packed_integers(["Z"], [i], nq))

    cupp_expansion = PauliExpansion(
        library_handle=cupp_handle,
        num_qubits=nq,
        num_terms=nq,
        xz_bits=cupp_xz,
        coeffs=cupp_coefs,
        options=PauliExpansionOptions(memory_limit="80%", blocking=True),
    )
    cupp_truncation = Truncation(
        pauli_coeff_cutoff=lower_atol,
        pauli_weight_cutoff=max_pauli_weight,
    )

    # build the circuit
    cupp_circuit = []
    for _ in range(num_steps):
        cupp_circuit.extend(PauliRotationGate(theta_x, ["X"], [i]) for i in range(nq))
        cupp_circuit.extend(
            PauliRotationGate(theta_zz, ["Z", "Z"], [i, i + 1]) for i in range(nq - 1)
        )

    # back-propagate the observable
    t1 = time.perf_counter()
    for gate_index in range(len(cupp_circuit) - 1, -1, -1):
        cupp_expansion = cupp_expansion.apply_gate(
            cupp_circuit[gate_index],
            truncation=cupp_truncation,
            adjoint=True,
            sort_order=None,
            keep_duplicates=False,
        )
    trace_significand, trace_exponent = cupp_expansion.trace_with_zero_state()
    cupp_expval = float(trace_significand * np.exp2(trace_exponent))
    t2 = time.perf_counter()
    runtimes_dict["cuPauliProp (GPU)"].append(t2 - t1)
    expvals_dict["cuPauliProp (GPU)"].append(cupp_expval)

    # ---------------------------------------------------------------------------

with open("trotter_ising_results.json", "w") as file:
    json.dump(
        {
            "qubit_range": list(qubit_range),
            "runtimes": runtimes_dict,
            "expvals": expvals_dict,
        },
        file,
        indent=4,
    )
