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

"""One propagation backend per function, each measured on the same TFIM model.

Every backend imports its own dependencies *inside* its function: cupy/cuquantum need
a GPU, ppvm needs its compiled extension, and a scaling sweep must be able to run the
backends that work on the current host without the others' imports failing first.

Each returns a `BackendResult` holding the per-step series. Both entry points build on
that: `run_model.py` plots the series, `run_one.py` reduces it to totals.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import psutil

from model import Settings, grid_edges, observable, pauli_rotations, step_circuit

# What each backend's `memory` series actually measures. They are not the same
# quantity — say so wherever these numbers are plotted or tabulated.
MEMORY_METRICS = {
    "monoprop": "operator memory (reported by the library)",
    "QuEra ppvm": "process RSS growth (no memory accounting exposed)",
    "Qiskit pauli-prop": "process RSS growth (no memory accounting exposed)",
    "cuPauliProp (GPU)": "GPU memory pool in use",
    "PauliPropagation.jl": "Base.summarysize of the Pauli sum",
    "mlxQ statevector": "statevector size (2^n complex64 amplitudes)",
}


@dataclass
class BackendResult:
    """Per-step series for one backend over one model."""

    label: str
    runtime: list[float] = field(default_factory=list)
    expvals: list[float] = field(default_factory=list)
    num_terms: list[int] = field(default_factory=list)
    memory: list[float] = field(default_factory=list)
    # Set when the library reports its own operator footprint, so the scaling
    # summary can prefer it over a whole-process RSS proxy.
    operator_memory_mb: float | None = None

    @property
    def memory_metric(self) -> str:
        return MEMORY_METRICS.get(self.label, "unknown")


def _run_steps(
    settings: Settings,
    label: str,
    step: Callable[[int], tuple[float, int, float]],
) -> BackendResult:
    """Drive `step` once per Trotter point, timing it and recording what it returns.

    `step(step_idx)` advances the backend by one benchmark step and returns
    (expectation value, term count, memory MB). The timer brackets the propagation
    *and* the expectation value, matching what every backend reports.
    """
    result = BackendResult(label=label)
    for step_idx, _ in enumerate(settings.step_range):
        t1 = time.perf_counter()
        expval, num_terms, memory_mb = step(step_idx)
        t2 = time.perf_counter()
        result.runtime.append(t2 - t1)
        result.expvals.append(expval)
        result.num_terms.append(num_terms)
        result.memory.append(memory_mb)
    return result


def run_monoprop(settings: Settings) -> BackendResult:
    from monoprop import PauliPropagator
    from monoprop.qiskit_conversion import from_qiskit_circuit, from_qiskit_operator

    circ = from_qiskit_circuit(step_circuit(settings), initial_state=[])
    propagator = PauliPropagator(
        initial_operator=from_qiskit_operator(observable(settings)),
        initial_state=circ.initial_state,
        cutoff=settings.max_pauli_weight,
        lower_atol=settings.lower_atol,
    )

    def step(_step_idx: int) -> tuple[float, int, float]:
        propagator.propagate(circ)
        expval = propagator.expectation_value()
        mem = propagator._simulator.operator_memory_bytes() / 1024**2
        return expval, propagator.size(), mem

    result = _run_steps(settings, "monoprop", step)
    result.operator_memory_mb = result.memory[-1]
    return result


def run_ppvm(settings: Settings) -> BackendResult:
    from ppvm import PauliSum

    nq = settings.num_qubits
    obs_qubits = settings.observable_qubits
    pauli_sum = PauliSum.new(
        n_qubits=nq,
        terms=[f"Z{obs_qubits[0]}Z{obs_qubits[1]}"],
        min_abs_coeff=settings.lower_atol,
        max_pauli_weight=settings.max_pauli_weight,
    )
    edges = grid_edges(settings.nx, settings.ny)
    process = psutil.Process()
    accumulated_bytes = 0

    def step(_step_idx: int) -> tuple[float, int, float]:
        # ppvm exposes no memory accounting: approximate it via RSS growth over its
        # own step. The timer is already running, so keep this to two cheap reads.
        nonlocal accumulated_bytes
        before = process.memory_info().rss
        for i, k in edges:
            pauli_sum.rzz(i, k, settings.theta_zz)
        for i in range(nq):
            pauli_sum.rz(i, settings.theta_z)
        for i in range(nq):
            pauli_sum.rx(i, settings.theta_x)
        expval = pauli_sum.overlap_with_zero()
        accumulated_bytes += max(0, process.memory_info().rss - before)
        return expval, len(pauli_sum), accumulated_bytes / 1024**2

    return _run_steps(settings, "QuEra ppvm", step)


def run_qiskit(settings: Settings, max_terms: int | Sequence[int]) -> BackendResult:
    """Qiskit's pauli-prop, which truncates on a term budget rather than a weight.

    `max_terms` is that mandatory budget. `run_model.py` passes monoprop's term count
    at each step so the two track each other; the scaling driver passes monoprop's
    final term count for the size as a single constant (recorded alongside the
    result, since it makes the budget more generous in the early steps).
    """
    from pauli_prop import propagate_through_circuit

    operator = observable(settings)
    circ = step_circuit(settings)
    process = psutil.Process()
    accumulated_bytes = 0

    def step(step_idx: int) -> tuple[float, int, float]:
        nonlocal operator, accumulated_bytes
        budget = max_terms if isinstance(max_terms, int) else max_terms[step_idx]
        before = process.memory_info().rss
        operator, _ = propagate_through_circuit(
            operator,
            circ,
            max_terms=int(budget),
            atol=settings.lower_atol,
            frame="h",
        )
        expval = float(operator.coeffs[~operator.paulis.x.any(axis=1)].sum())
        accumulated_bytes += max(0, process.memory_info().rss - before)
        return expval, len(operator), accumulated_bytes / 1024**2

    return _run_steps(settings, "Qiskit pauli-prop", step)


def run_cupauliprop(settings: Settings) -> BackendResult:
    import cupy as cp
    from cuquantum.pauliprop.experimental import (
        LibraryHandle,
        PauliExpansion,
        PauliExpansionOptions,
        PauliRotationGate,
        Truncation,
        get_num_packed_integers,
    )

    nq = settings.num_qubits
    num_packed = get_num_packed_integers(nq)
    xz_bits = cp.zeros((1, 2 * num_packed), dtype=cp.uint64)
    xz_bits[0] = cp.asarray(
        _pack_pauli_string(["Z", "Z"], list(settings.observable_qubits), nq)
    )

    expansion = PauliExpansion(
        library_handle=LibraryHandle(),
        num_qubits=nq,
        num_terms=1,
        xz_bits=xz_bits,
        coeffs=cp.ones((1,), dtype=cp.float64),
        options=PauliExpansionOptions(memory_limit="80%", blocking=True),
    )
    truncation = Truncation(
        pauli_coeff_cutoff=settings.lower_atol,
        pauli_weight_cutoff=settings.max_pauli_weight,
    )
    gates = [
        PauliRotationGate(theta, paulis, qubits)
        for theta, paulis, qubits in pauli_rotations(settings)
    ]

    def step(_step_idx: int) -> tuple[float, int, float]:
        nonlocal expansion
        for gate in reversed(gates):
            expansion = expansion.apply_gate(
                gate,
                truncation=truncation,
                adjoint=True,
                sort_order=None,
                keep_duplicates=False,
            )
        significand, exponent = expansion.trace_with_zero_state()
        expval = float(significand * np.exp2(exponent))
        mem = cp.get_default_memory_pool().used_bytes() / 1024**2
        return expval, expansion.num_terms, mem

    result = _run_steps(settings, "cuPauliProp (GPU)", step)
    result.operator_memory_mb = result.memory[-1]
    return result


def _pack_pauli_string(
    paulis: list[str], qubits: list[int], num_qubits: int
) -> np.ndarray:
    """Pack a Pauli string into the (x, z) bitfield layout cuPauliProp expects."""
    from cuquantum.pauliprop.experimental import get_num_packed_integers

    num_packed = get_num_packed_integers(num_qubits)
    out = np.zeros(num_packed * 2, dtype=np.uint64)
    x_ptr, z_ptr = out[:num_packed], out[num_packed:]
    for pauli, qubit in zip(paulis, qubits):
        int_ind, bit_ind = qubit // 64, qubit % 64
        if pauli in ("X", "Y"):
            x_ptr[int_ind] |= 1 << bit_ind
        if pauli in ("Z", "Y"):
            z_ptr[int_ind] |= 1 << bit_ind
    return out


def run_mlxq(settings: Settings) -> BackendResult:
    """mlxQ: an exact statevector baseline on the Apple-silicon GPU (via MLX).

    Not a Pauli-propagation engine — it bounds them from below: wherever the
    2^n statevector fits, its per-step cost is size-independent of the operator,
    so it locates the size under which propagation is not worth running (about
    25-28 qubits against monoprop on an M1 Max). Install on Apple silicon with
    `pip install "mlxq @ git+https://github.com/BoltzmannEntropy/Qupertino"`;
    MLXQ_METAL_KERNELS=1 selects its hand-written Metal kernels. Expectation
    values are exact up to complex64 (~1e-6); the memory column is the exact
    statevector footprint.
    """
    import mlx.core as mx
    from mlxq import shaders
    from mlxq.gates import RX
    from mlxq.sim import StateVectorSimulator

    nq = settings.num_qubits
    sim = StateVectorSimulator(nq)
    edges = grid_edges(settings.nx, settings.ny)
    a, b = settings.observable_qubits
    metal = shaders.metal_enabled()
    rx_gate = RX(settings.theta_x)

    # The all-qubit RZ layer as one cached diagonal (the same construction the
    # simulator uses internally for its fused ZZ layer): the phase per basis
    # state is exp(-i*(theta_z/2)*sum_q s_q) with s_q = +-1 per bit.
    idx = mx.arange(1 << nq, dtype=mx.uint32)
    acc = mx.zeros((1 << nq,), dtype=mx.int32)
    for q in range(nq):
        acc = acc + (1 - 2 * ((idx >> (nq - 1 - q)) & 1).astype(mx.int32))
    ang = (-settings.theta_z / 2.0) * acc.astype(mx.float32)
    z_phase = mx.cos(ang).astype(mx.complex64) + 1j * mx.sin(ang).astype(mx.complex64)
    mx.eval(z_phase)

    q0, q1 = sorted((a, b))
    state_mb = (1 << nq) * 8 / 1024**2

    def step(_step_idx: int) -> tuple[float, int, float]:
        # apply_zz_layer(theta) applies exp(-i*theta*sum Z_a Z_b): Qiskit's
        # RZZ(theta_zz) is theta_zz/2 in that convention.
        sim.apply_zz_layer(settings.theta_zz / 2.0, edges)
        sim.state = sim.state * z_phase
        if metal:
            sim.state = shaders.rx_layer_all(sim.state, nq, settings.theta_x)
        else:
            for q in range(nq):
                sim.apply_single(rx_gate, q)
        # <Z_a Z_b> = P(bits agree) - P(bits disagree). MLX is lazy; float()
        # forces the whole step's evaluation, so the timer sees the real work.
        t = mx.reshape(sim.state, (1 << q0, 2, 1 << (q1 - q0 - 1), 2, -1))
        p = mx.abs(t) ** 2
        expval = float(
            mx.sum(p[:, 0, :, 0, :])
            + mx.sum(p[:, 1, :, 1, :])
            - mx.sum(p[:, 0, :, 1, :])
            - mx.sum(p[:, 1, :, 0, :])
        )
        if nq >= 26:
            # State-sized temporaries otherwise accumulate in MLX's buffer pool
            # and push a 32 GB machine into unified-memory thrashing.
            mx.clear_cache()
        return expval, 1 << nq, state_mb

    result = _run_steps(settings, LABELS["mlxq"], step)
    result.operator_memory_mb = state_mb
    return result


# Backend name (as passed on a command line) -> runner. The Julia backend is not
# here: it is a separate process driven by run_scaling.jl.
CPU_BACKENDS = ("monoprop", "ppvm", "qiskit")
GPU_BACKENDS = ("cupauliprop",)
# Apple-silicon GPU via MLX. Not a Pauli-propagation engine: an exact statevector
# baseline that locates the size below which propagation is not worth running.
APPLE_BACKENDS = ("mlxq",)
JULIA_BACKEND = "juliapp"
ALL_BACKENDS = (*CPU_BACKENDS, *GPU_BACKENDS, *APPLE_BACKENDS, JULIA_BACKEND)

# The environment variable each backend takes its thread count from. Read both ways: the
# sweep driver sets these to apply a per-backend cap, and the worker reads its own to record
# what it ran with. A backend must consult only *its* variable -- the others are exported
# node-wide, so a shared scan would report another backend's setting.
#
# Only monoprop and the Julia backend respond to theirs. ppvm propagates through its serial
# indexmap `PauliSum` (rayon sizes a pool that gets no work) and Qiskit's `_accelerate`
# extension has no parallel path at all; measured, both sit at 0.99 busy cores. Their entries
# bound idle pools only -- see the threading note in ../README.md.
THREAD_VARS = {
    "monoprop": ("monoprop_NUM_THREADS",),
    "ppvm": ("RAYON_NUM_THREADS",),
    "qiskit": ("OMP_NUM_THREADS", "MKL_NUM_THREADS"),
    "cupauliprop": ("OMP_NUM_THREADS",),
    "mlxq": (),  # Apple GPU: no CPU thread knob
    JULIA_BACKEND: ("JULIA_NUM_THREADS",),
}

LABELS = {
    "monoprop": "monoprop",
    "ppvm": "QuEra ppvm",
    "qiskit": "Qiskit pauli-prop",
    "cupauliprop": "cuPauliProp (GPU)",
    "mlxq": "mlxQ statevector",
    JULIA_BACKEND: "PauliPropagation.jl",
}
