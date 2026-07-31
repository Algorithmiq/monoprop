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

from ppvm import PauliSum
from tqdm import tqdm

from _common import RssPeakSampler, ensure_results_file, load_settings, update_results

LABEL = "QuEra ppvm"

settings = load_settings()
ensure_results_file(settings)

ppvm_obs = PauliSum.new(
    n_qubits=settings.nq,
    terms=[f"Z{settings.obs_qubits[0]}Z{settings.obs_qubits[1]}"],
    min_abs_coeff=settings.lower_atol,
    max_pauli_weight=settings.max_pauli_weight,
)

runtime: list[float] = []
memory: list[float] = []
expvals: list[float] = []
num_terms: list[int] = []

with RssPeakSampler() as sampler:
    for step_idx, _ in enumerate(tqdm(settings.step_range, desc=LABEL)):
        sampler.reset()
        t1 = time.perf_counter()
        for i, k in settings.grid_edges:
            ppvm_obs.rzz(i, k, settings.theta_zz)
        for i in range(settings.nq):
            ppvm_obs.rz(i, settings.theta_z)
        for i in range(settings.nq):
            ppvm_obs.rx(i, settings.theta_x)
        ppvm_expval = ppvm_obs.overlap_with_zero()
        t2 = time.perf_counter()

        if step_idx > 0:
            runtime.append(t2 - t1)
        expvals.append(ppvm_expval)
        num_terms.append(len(ppvm_obs))
        memory.append(sampler.peak_mb())

update_results(
    LABEL, runtime=runtime, memory=memory, expvals=expvals, num_terms=num_terms
)
