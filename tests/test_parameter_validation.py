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

from monoprop import Circuit, ExpGate, MajoranaPropagator
from monoprop.majorana import MajoranaOperator


def _two_gate_graph(serial_comm):
    """A propagator with a two-layer, two-parameter graph already built."""
    operator = MajoranaOperator({(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2)
    mp = MajoranaPropagator(operator, [0, 1], cutoff=4, comm=serial_comm)
    circuit = Circuit(
        (
            ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2)),
            ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2)),
        )
    )
    mp.build_graph(circuit)  # identity mapping -> two distinct angles
    return mp, circuit


class TestConstructorValidation:
    def test_invalid_tolerances(self, serial_comm):
        operator = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
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
    def test_build_graph_accumulates_layers(self, serial_comm):
        mp, _ = _two_gate_graph(serial_comm)
        assert mp.graph_layers == 2
        assert mp.n_parameters == 2

    @pytest.mark.parametrize("parameters", [[1.0], [1.0, 2.0, 3.0]])
    def test_wrong_parameter_length_raises(self, serial_comm, parameters):
        mp, _ = _two_gate_graph(serial_comm)
        with pytest.raises(RuntimeError, match="Parameter length"):
            mp.expval(parameters)

    def test_non_contiguous_mapping_raises(self):
        gates = (
            ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2), index=0),
            ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2), index=2),
        )
        with pytest.raises(ValueError, match="contiguous"):
            Circuit(gates)

    def test_mixed_param_scheme_rejected(self):
        gates = (
            ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2), index=0),
            ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2)),
        )
        with pytest.raises(ValueError, match="every gate must set"):
            Circuit(gates)

    def test_shared_mapping_index_ties_gates(self, serial_comm):
        operator = MajoranaOperator({(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2)
        mp = MajoranaPropagator(operator, [0, 1], cutoff=4, comm=serial_comm)
        circuit = Circuit(
            (
                ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2), index=0),
                ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2), index=0),
            ),
        )
        mp.build_graph(circuit)
        assert mp.graph_layers == 2
        assert mp.n_parameters == 1

    def test_functional_invalidated_after_graph_mutation(self, serial_comm):
        operator = MajoranaOperator({(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2)
        mp = MajoranaPropagator(operator, [0, 1], cutoff=4, comm=serial_comm)
        mp.build_graph(
            Circuit(
                (
                    ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2)),
                    ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2)),
                )
            )
        )
        functional = mp.expval_functional()
        # Appending another layer mutates the graph, so the previously-built functional
        # must reject being called against the stale plan.
        mp.build_graph(Circuit((ExpGate(MajoranaOperator({(2,): 1.0}, num_modes=2)),)))
        # Two parameters, so the stale-graph guard fires rather than the length check.
        with pytest.raises(RuntimeError, match=r"MP object has been modified"):
            functional([1.0, 2.0])


class TestEvolvedOperatorBothPictures:
    def test_schrodinger_returns_state_dict(self, serial_comm):
        operator = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
        mp = MajoranaPropagator(
            operator, [0, 1], cutoff=4, schrodinger_cutoff=2, comm=serial_comm
        )
        result = mp.evolved_operator()
        assert isinstance(result, dict)
