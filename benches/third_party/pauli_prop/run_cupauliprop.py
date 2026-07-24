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

import threading
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
from tqdm import tqdm

from _common import load_settings, update_results

LABEL = "cuPauliProp (GPU)"


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


class GpuMemPeakSampler:
    """Tracks the GPU's peak used device memory (driver-level, via cudaMemGetInfo) within a
    resettable window.

    This is the GPU analogue of RssPeakSampler: rather than trusting one library's own pool
    accounting (which can miss allocations that bypass that pool, e.g. CUDA context or
    library-handle overhead), it polls the CUDA driver's own free/total memory report from a
    background thread, so it reflects everything actually resident on the device. Caveat: if
    another process shares this GPU concurrently, its usage is included too.
    """

    def __init__(self, interval_s: float = 1e-3) -> None:
        self._interval_s = interval_s
        self._peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)

    @staticmethod
    def _used_bytes() -> int:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return total_bytes - free_bytes

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            used = self._used_bytes()
            if used > self._peak_bytes:
                self._peak_bytes = used

    def __enter__(self) -> GpuMemPeakSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop_event.set()
        self._thread.join()

    def reset(self) -> None:
        self._peak_bytes = self._used_bytes()

    def peak_mb(self) -> float:
        return self._peak_bytes / 1024**2


class _PoolPeakHook(cp.cuda.MemoryHook):
    """Event-driven peak of cupy's default memory pool, for reference alongside the GPU-wide
    peak. Fires on every allocation, so — unlike sampling on a timer — it cannot miss a spike
    that is allocated and freed within a single step. Only sees allocations routed through
    cupy's own pool (which cuQuantum's scratch workspace uses by default, but CUDA
    context/library-handle overhead does not)."""

    name = "PoolPeakHook"

    def __init__(self) -> None:
        self.peak_bytes = 0

    def malloc_postprocess(self, **kwargs: object) -> None:
        used = cp.get_default_memory_pool().used_bytes()
        if used > self.peak_bytes:
            self.peak_bytes = used

    def reset(self) -> None:
        self.peak_bytes = cp.get_default_memory_pool().used_bytes()


settings = load_settings()

cupp_handle = LibraryHandle()

num_packed = get_num_packed_integers(settings.nq)
cupp_xz = cp.zeros((1, 2 * num_packed), dtype=cp.uint64)
cupp_coefs = cp.ones((1,), dtype=cp.float64)
cupp_xz[0] = cp.asarray(
    _pauli_string_to_packed_integers(["Z", "Z"], list(settings.obs_qubits), settings.nq)
)

cupp_expansion = PauliExpansion(
    library_handle=cupp_handle,
    num_qubits=settings.nq,
    num_terms=1,
    xz_bits=cupp_xz,
    coeffs=cupp_coefs,
    options=PauliExpansionOptions(memory_limit="80%", blocking=True),
)
cupp_truncation = Truncation(
    pauli_coeff_cutoff=settings.lower_atol,
    pauli_weight_cutoff=settings.max_pauli_weight,
)

cupp_step_gates = [
    PauliRotationGate(settings.theta_zz, ["Z", "Z"], [i, k])
    for i, k in settings.grid_edges
]
cupp_step_gates += [
    PauliRotationGate(settings.theta_z, ["Z"], [i]) for i in range(settings.nq)
]
cupp_step_gates += [
    PauliRotationGate(settings.theta_x, ["X"], [i]) for i in range(settings.nq)
]

runtime: list[float] = []
memory: list[float] = []
expvals: list[float] = []
num_terms: list[int] = []
native_memory: list[float] = []

pool_hook = _PoolPeakHook()

with GpuMemPeakSampler() as gpu_sampler, pool_hook:
    for step_idx, _ in enumerate(tqdm(settings.step_range, desc=LABEL)):
        gpu_sampler.reset()
        pool_hook.reset()
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
            runtime.append(t2 - t1)
        expvals.append(cupp_expval)
        num_terms.append(cupp_expansion.num_terms)
        memory.append(gpu_sampler.peak_mb())
        native_memory.append(pool_hook.peak_bytes / 1024**2)

update_results(
    LABEL,
    runtime=runtime,
    memory=memory,
    expvals=expvals,
    num_terms=num_terms,
    native_memory=native_memory,
)
