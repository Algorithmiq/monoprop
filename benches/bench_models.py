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

"""Fixed-model benchmarks: heavy, Heisenberg-only, in-place simulations.

The 120-qubit Fermi-Hubbard trajectory and the 127-qubit Pauli-basis kicked-Ising
circuit, at fixed sizes. The registry (config class, builder, steps-per-run) lives
in ``_builders.MODELS``; each config field is overridable via ``--<model>-<field>``.
"""

from __future__ import annotations

import pytest
from _builders import MODELS, barriered


@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_model(benchmark, bench_comm, model_configs, model, record_model_config):
    """Benchmark a fixed in-place model simulation (Heisenberg picture)."""
    _config_cls, build_fn, steps_fn = MODELS[model]
    config = model_configs[model]
    steps = steps_fn(config)
    record_model_config(model, config)

    def setup():
        return (build_fn(config, comm=bench_comm), steps), {}

    def run(built, n_steps):
        propagator, circuit = built
        value = 0.0
        for _ in range(n_steps):
            propagator.propagate(circuit)
            value = propagator.expectation_value()
        return value

    result = benchmark.pedantic(
        barriered(run, bench_comm), setup=setup, rounds=1, iterations=1
    )
    assert isinstance(result, float)
