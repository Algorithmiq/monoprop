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

"""Parameter validation and error handling for the propagators."""

from __future__ import annotations

import pytest

from monoprop import Gate, MajoranaPropagator, Parameter, ParameterVector, Term
from monoprop.monomial_data import MonomialOperator


def _two_gate_graph(serial_comm):
    """A propagator with a two-layer, two-parameter graph already built."""
    operator = MonomialOperator.from_dict(
        terms_dict={(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2
    )
    mp = MajoranaPropagator(operator, [0, 1], cutoff=4, comm=serial_comm)
    params = ParameterVector()
    gates = [
        Gate(params.new(), (Term((0,), 1.0),)),
        Gate(params.new(), (Term((1,), 1.0),)),
    ]
    mp.propagate_build_graph(gates)
    return mp, gates


class TestConstructorValidation:
    """Validation performed at construction time."""

    def test_invalid_basis_change_length(self, serial_comm):
        operator = MonomialOperator.from_dict(terms_dict={(0, 1): 1.0j}, num_modes=2)
        with pytest.raises(ValueError, match="Basis change must have length 4"):
            MajoranaPropagator(
                operator,
                [0, 1],
                cutoff=4,
                basis_change=[[0], [1], [2]],
                comm=serial_comm,
            )

    def test_invalid_tolerances(self, serial_comm):
        operator = MonomialOperator.from_dict(terms_dict={(0, 1): 1.0j}, num_modes=2)
        with pytest.raises(
            RuntimeError,
            match=r"upper_atol \(0\.1\) must be greater than or equal to lower_atol \(0\.5\)",
        ):
            MajoranaPropagator(
                operator,
                [0, 1],
                cutoff=4,
                upper_atol=0.1,
                lower_atol=0.5,
                comm=serial_comm,
            )


class TestGraphAndParameterValidation:
    """Validation around the graph-owned gate information and parameters."""

    def test_build_graph_accumulates_layers(self, serial_comm):
        mp, _ = _two_gate_graph(serial_comm)
        assert mp.graph_layers == 2
        assert mp.n_parameters == 2

    @pytest.mark.parametrize("parameters", [[1.0], [1.0, 2.0, 3.0]])
    def test_wrong_parameter_length_raises(self, serial_comm, parameters):
        mp, _ = _two_gate_graph(serial_comm)
        with pytest.raises(RuntimeError, match="Parameter length"):
            mp.expectation_value(parameters)

    def test_functional_invalidated_after_graph_mutation(self, serial_comm):
        operator = MonomialOperator.from_dict(
            terms_dict={(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2
        )
        mp = MajoranaPropagator(operator, [0, 1], cutoff=4, comm=serial_comm)
        params = ParameterVector()
        shared = params.new()
        other = params.new()
        mp.propagate_build_graph(
            [Gate(shared, (Term((0,), 1.0),)), Gate(other, (Term((1,), 1.0),))]
        )
        functional = mp.expectation_value_functional()
        # Add a layer reusing an existing parameter so n_parameters is unchanged and the
        # stale-graph guard (not the parameter-length check) is what fires.
        mp.propagate_build_graph([Gate(shared, (Term((2,), 1.0),))])
        with pytest.raises(RuntimeError, match=r"MP object has been modified"):
            functional([1.0, 2.0])

    def test_bind_rejects_wrong_length(self):
        params = ParameterVector([Parameter(), Parameter()])
        with pytest.raises(ValueError, match="Expected 2 parameters"):
            params.bind([1.0])


class TestEvolvedOperatorPictureGuard:
    """evolved_operator is Heisenberg-only."""

    def test_schrodinger_error(self, serial_comm):
        operator = MonomialOperator.from_dict(terms_dict={(0, 1): 1.0j}, num_modes=2)
        mp = MajoranaPropagator(
            operator, [0, 1], cutoff=4, schrodinger_cutoff=2, comm=serial_comm
        )
        with pytest.raises(
            ValueError, match="Cannot call evolved_operator in Schrodinger picture"
        ):
            mp.evolved_operator()
