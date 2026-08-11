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

import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from model import Settings, grid_edges, observable, pauli_rotations, step_circuit

# The repository's own benchmark suite owns the memory instrumentation; this directory is a
# separate uv project, so reach it by path rather than by dependency.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _memory_cpu import HighWaterMark  # noqa: E402
from _memory_gpu import DeviceHighWaterMark  # noqa: E402

# cuPauliProp sizes its workspace as a fraction of the card, so this is a confounder for
# any device-memory figure: raise it and the library may hold more without needing more.
# Kept at the upstream default so the runtime is the configuration users would hit.
CUPAULIPROP_MEMORY_LIMIT = "80%"

# Every backend's `memory` series is this one quantity, measured the same way, so the
# curves may share an axis.
HOST_MEMORY_METRIC = "peak process RSS over the step (kernel VmHWM)"

# What a backend reports about its *own* footprint, where it reports anything. Reference
# only: these are not commensurable with each other (one counts an operator, another an
# object graph, another device memory), so they must never be compared across backends.
# The GPU backend sets its own string at run time -- see `_memory_gpu.DEVICE_MEMORY_METRICS`,
# which depends on the allocator actually in use.
OPERATOR_MEMORY_METRICS = {
    "monoprop": "operator memory (reported by the library)",
    "PauliPropagation.jl": "Base.summarysize of the Pauli sum",
}


@dataclass
class BackendResult:
    """Per-step series for one backend over one model."""

    label: str
    runtime: list[float] = field(default_factory=list)
    expvals: list[float] = field(default_factory=list)
    num_terms: list[int] = field(default_factory=list)
    # Peak host RSS over each step; see HOST_MEMORY_METRIC.
    memory: list[float] = field(default_factory=list)
    # Empty for backends that expose no accounting of their own.
    operator_memory: list[float] = field(default_factory=list)
    # Set by backends whose own metric is only known at run time; see the property below.
    operator_metric: str = ""

    @property
    def memory_metric(self) -> str:
        return HOST_MEMORY_METRIC

    @property
    def operator_memory_metric(self) -> str:
        if self.operator_metric:
            return self.operator_metric
        return OPERATOR_MEMORY_METRICS.get(self.label, "none reported")

    @property
    def operator_memory_mb(self) -> float | None:
        """Return the library's own final footprint, where it reports one."""
        return self.operator_memory[-1] if self.operator_memory else None


def _run_steps(
    settings: Settings,
    label: str,
    step: Callable[[int], tuple[float, int, float | None]],
) -> BackendResult:
    """Drive `step` once per Trotter point, timing it and recording what it returns.

    `step(step_idx)` advances the backend by one benchmark step and returns (expectation
    value, term count, the library's own footprint in MB or None). The timer brackets the
    propagation *and* the expectation value, matching what every backend reports.

    Peak memory is taken here rather than inside each backend so that every engine is
    measured by the same instrument over exactly the timed region. Opening the window
    settles the process, which is why it wraps the timer instead of the reverse.
    """
    result = BackendResult(label=label)
    for step_idx, _ in enumerate(settings.step_range):
        with HighWaterMark() as window:
            t1 = time.perf_counter()
            expval, num_terms, operator_mb = step(step_idx)
            t2 = time.perf_counter()
        result.runtime.append(t2 - t1)
        result.expvals.append(expval)
        result.num_terms.append(num_terms)
        result.memory.append(window.peak_mb)
        if operator_mb is not None:
            result.operator_memory.append(operator_mb)
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

    def step(_step_idx: int) -> tuple[float, int, float | None]:
        propagator.propagate(circ)
        expval = propagator.expectation_value()
        mem = propagator._simulator.operator_memory_bytes() / 1024**2
        return expval, propagator.size(), mem

    return _run_steps(settings, "monoprop", step)


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

    # ppvm exposes no memory accounting of its own; _run_steps measures it from outside.
    def step(_step_idx: int) -> tuple[float, int, float | None]:
        for i, k in edges:
            pauli_sum.rzz(i, k, settings.theta_zz)
        for i in range(nq):
            pauli_sum.rz(i, settings.theta_z)
        for i in range(nq):
            pauli_sum.rx(i, settings.theta_x)
        expval = pauli_sum.overlap_with_zero()
        return expval, len(pauli_sum), None

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

    # pauli-prop exposes no memory accounting of its own; _run_steps measures it from outside.
    def step(step_idx: int) -> tuple[float, int, float | None]:
        nonlocal operator
        budget = max_terms if isinstance(max_terms, int) else max_terms[step_idx]
        operator, _ = propagate_through_circuit(
            operator,
            circ,
            max_terms=int(budget),
            atol=settings.lower_atol,
            frame="h",
        )
        expval = float(operator.coeffs[~operator.paulis.x.any(axis=1)].sum())
        return expval, len(operator), None

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
        options=PauliExpansionOptions(
            memory_limit=CUPAULIPROP_MEMORY_LIMIT, blocking=True
        ),
    )
    truncation = Truncation(
        pauli_coeff_cutoff=settings.lower_atol,
        pauli_weight_cutoff=settings.max_pauli_weight,
    )
    gates = [
        PauliRotationGate(theta, paulis, qubits)
        for theta, paulis, qubits in pauli_rotations(settings)
    ]

    device_metric = ""

    def step(_step_idx: int) -> tuple[float, int, float | None]:
        nonlocal expansion, device_metric
        # The window synchronizes the device on both edges, so it belongs inside the
        # timed region: GPU work the timer must count is work the reading must cover.
        with DeviceHighWaterMark() as device_window:
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
        device_metric = device_window.metric
        return expval, expansion.num_terms, device_window.peak_mb

    result = _run_steps(settings, "cuPauliProp (GPU)", step)
    result.operator_metric = device_metric
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


# Backend name (as passed on a command line) -> runner. The Julia backend is not
# here: it is a separate process driven by run_scaling.jl.
CPU_BACKENDS = ("monoprop", "ppvm", "qiskit")
GPU_BACKENDS = ("cupauliprop",)
JULIA_BACKEND = "juliapp"
ALL_BACKENDS = (*CPU_BACKENDS, *GPU_BACKENDS, JULIA_BACKEND)

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
    JULIA_BACKEND: ("JULIA_NUM_THREADS",),
}

LABELS = {
    "monoprop": "monoprop",
    "ppvm": "QuEra ppvm",
    "qiskit": "Qiskit pauli-prop",
    "cupauliprop": "cuPauliProp (GPU)",
    JULIA_BACKEND: "PauliPropagation.jl",
}
