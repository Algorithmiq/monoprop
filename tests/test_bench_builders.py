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

``benches`` is on the pytest pythonpath (see ``pyproject.toml``), so the module
imports as ``_builders`` from the normal test suite.
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
    # The seed is left to the caller (``None`` draws fresh entropy); the bench
    # CLI fixes it via ``--seed`` for reproducible recorded runs.
    assert defaults["seed"] is None


def test_built_graph_is_populated() -> None:
    # A deliberately tiny problem keeps this fast while proving the
    # energy/gradient/pare path operates on a real (non-empty) graph: the
    # benchmark builds the graph in its fixture before measuring. gen_length=4
    # (a length-4 Majorana monomial is Hermitian with real coefficients; a
    # length-2 one is anti-Hermitian and would be rejected as non-Hermitian).
    problem = make_random_problem(
        gen_length=4, obs_terms=3, num_generators=5, num_modes=6, cutoff=3, seed=0
    )
    propagator, gates, _parameters = build_random_propagator(problem)
    propagator.propagate_build_graph(gates)
    _n_cos_indices, n_cycles = propagator.graph_size()
    assert n_cycles > 0
