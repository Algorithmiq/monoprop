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

"""Tests for the benchmark model builders (``benches/_builders.py``).

``benches`` is on the pytest pythonpath (see ``pyproject.toml``), so it imports as ``_builders``.
"""

from __future__ import annotations

import inspect

from _builders import build_random_propagator, make_random_problem


def test_random_default_sizes_are_meaningful() -> None:
    defaults = {
        name: param.default
        for name, param in inspect.signature(make_random_problem).parameters.items()
    }
    assert defaults["gen_length"] == 4
    assert defaults["obs_terms"] == 10000
    assert defaults["num_generators"] == 100
    assert defaults["num_modes"] == 128
    assert defaults["cutoff"] == 6
    # The bench CLI fixes the seed via ``--seed``.
    assert defaults["seed"] is None


def test_built_graph_is_populated(serial_comm) -> None:
    # gen_length=4: a length-4 Majorana monomial is Hermitian with real coefficients, while a
    # length-2 one is anti-Hermitian and would be rejected.
    # serial_comm because graph_size() is rank-local: on COMM_WORLD a problem this small leaves
    # some ranks with no local cycles.
    problem = make_random_problem(
        gen_length=4, obs_terms=3, num_generators=5, num_modes=6, cutoff=3, seed=0
    )
    propagator, circuit = build_random_propagator(problem, comm=serial_comm)
    propagator.build_graph(circuit)
    _n_cos_indices, n_cycles = propagator.graph_size()
    assert n_cycles > 0
