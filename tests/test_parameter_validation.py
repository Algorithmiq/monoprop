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

from monoprop import (
    Circuit,
    ExpGate,
    MajoranaPropagator,
    PauliPropagator,
    jordan_wigner_basis_change,
)
from monoprop.majorana import MajoranaOperator
from monoprop.pauli import PauliOperator


def _two_gate_graph(serial_comm):
    """A propagator with a two-layer, two-parameter graph already built."""
    operator = MajoranaOperator({(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2)
    mp = MajoranaPropagator(operator, [0, 1], cutoff=4, comm=serial_comm)
    circuit = Circuit(
        (
            ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2)),
            ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2)),
        ),
        2,
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
            Circuit(gates, 2)

    def test_mixed_param_scheme_rejected(self):
        gates = (
            ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2), index=0),
            ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2)),
        )
        with pytest.raises(ValueError, match="every gate must set"):
            Circuit(gates, 2)

    def test_shared_mapping_index_ties_gates(self, serial_comm):
        operator = MajoranaOperator({(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2)
        mp = MajoranaPropagator(operator, [0, 1], cutoff=4, comm=serial_comm)
        circuit = Circuit(
            (
                ExpGate(MajoranaOperator({(0,): 1.0}, num_modes=2), index=0),
                ExpGate(MajoranaOperator({(1,): 1.0}, num_modes=2), index=0),
            ),
            2,
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
                ),
                2,
            ),
        )
        functional = mp.expectation_value_functional()
        # Appending another layer mutates the graph, so the previously-built functional
        # must reject being called against the stale plan.
        mp.build_graph(
            Circuit((ExpGate(MajoranaOperator({(2,): 1.0}, num_modes=2)),), 2)
        )
        # Call with the parameter count the functional was built with (2), so the
        # stale-graph guard fires rather than the parameter-length check.
        with pytest.raises(RuntimeError, match=r"MP object has been modified"):
            functional([1.0, 2.0])

    @pytest.mark.parametrize(
        (
            "propagator_cls",
            "initial_operator",
            "updated_operator",
            "gate_generators",
            "cutoff",
        ),
        [
            pytest.param(
                MajoranaPropagator,
                MajoranaOperator({(0, 1): 1.0j, (2, 3): 0.5j}, num_modes=2),
                MajoranaOperator({(0, 1): 2.0j, (2, 3): 0.5j}, num_modes=2),
                (
                    MajoranaOperator({(0,): 1.0}, num_modes=2),
                    MajoranaOperator({(1,): 1.0}, num_modes=2),
                ),
                4,
                id="majorana",
            ),
            pytest.param(
                PauliPropagator,
                PauliOperator({"ZZ": 1.0, "XX": 0.5}, num_qubits=2),
                PauliOperator({"ZZ": 2.0, "XX": 0.5}, num_qubits=2),
                (
                    PauliOperator({"XI": 1.0}, num_qubits=2),
                    PauliOperator({"IY": 1.0}, num_qubits=2),
                ),
                2,
                id="pauli",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "schrodinger_cutoff", [None, 2], ids=["heisenberg", "schrodinger"]
    )
    @pytest.mark.parametrize(
        "functional_name",
        [
            "expectation_value_functional",
            "expectation_value_and_gradient_functional",
        ],
    )
    def test_functional_follows_initial_operator_update(
        self,
        serial_comm,
        propagator_cls,
        initial_operator,
        updated_operator,
        gate_generators,
        cutoff,
        schrodinger_cutoff,
        functional_name,
    ):
        mp = propagator_cls(
            initial_operator,
            [0, 1],
            cutoff=cutoff,
            schrodinger_cutoff=schrodinger_cutoff,
            comm=serial_comm,
        )
        mp.build_graph(
            Circuit(tuple(ExpGate(generator) for generator in gate_generators), 2)
        )
        functional = getattr(mp, functional_name)()
        parameters = [0.3, 0.7]

        def call(fn):
            result = fn(parameters)
            return result[0] if isinstance(result, tuple) else result

        before = call(functional)

        mp.update_initial_operator(updated_operator)

        # A re-weight moves no structure, so the functional built before it follows the new
        # coefficients instead of refusing the call: it now answers what the propagator answers.
        after = call(functional)
        assert after != pytest.approx(before)
        assert after == pytest.approx(mp.expval(parameters))
        assert after == call(getattr(mp, functional_name)())


class TestFunctionalValidityTable:
    """What each public mutator does to a functional built before it ran.

    The Python mirror of ``cpp/tests/functional_validity.cpp``: same rows, same expectations, run
    over both partition settings. The build-time coverage gate lives on the C++ side
    (``MonomialPropagator::num_mutating_methods``); here the roster is asserted as data.
    """

    _MODES = 2
    _CUTOFF = 4
    _PARAMS = (0.3, 0.7)
    _PARE_THRESHOLD = 1e-12

    # One gate per Hamiltonian term, and the terms carry different weights, so the answer is not
    # symmetric under swapping the two angles -- which is what makes the parameter_mapping row bite.
    @staticmethod
    def _generator(index):
        return MajoranaOperator({(index,): 1.0}, num_modes=2)

    @classmethod
    def _base_circuit(cls):
        return Circuit(
            (ExpGate(cls._generator(0)), ExpGate(cls._generator(2))),
            cls._MODES,
            cls._PARAMS,
        )

    @classmethod
    def _propagator(cls, comm, *, schrodinger, with_graph, first_weight=1.0):
        mp = MajoranaPropagator(
            MajoranaOperator(
                {(0, 1): first_weight * 1j, (2, 3): 0.5j}, num_modes=cls._MODES
            ),
            [0, 1],
            cutoff=cls._CUTOFF,
            schrodinger_cutoff=cls._CUTOFF if schrodinger else None,
            comm=comm,
        )
        if with_graph:
            mp.build_graph(cls._base_circuit())
        return mp

    # The mutators, one per public mutating method.
    @classmethod
    def _mutate_build_graph(cls, mp):
        mp.build_graph(Circuit((ExpGate(cls._generator(1)),), cls._MODES, (0.4,)))

    @classmethod
    def _mutate_propagate(cls, mp):
        mp.propagate(Circuit((ExpGate(cls._generator(0)),), cls._MODES, (0.4,)))

    @classmethod
    def _mutate_contract_partially(cls, mp):
        mp.contract_partially(list(cls._PARAMS), inplace=True)

    @classmethod
    def _mutate_update_initial_operator(cls, mp):
        mp.update_initial_operator(
            MajoranaOperator({(0, 1): 2.75j, (2, 3): 0.5j}, num_modes=cls._MODES)
        )

    @staticmethod
    def _mutate_parameter_mapping(mp):
        mp.parameter_mapping = [1, 0]

    @staticmethod
    def _mutate_cutoff(mp):
        mp.cutoff = 2

    @staticmethod
    def _mutate_cutoff_type(mp):
        mp.cutoff_type = "length"

    @classmethod
    def _mutate_basis_change(cls, mp):
        # No front-end setter; the engine property is the only way in (see tests/test_basis.py).
        mp._simulator.basis_change = jordan_wigner_basis_change(cls._MODES)

    @staticmethod
    def _mutate_lower_atol(mp):
        mp.lower_atol = 1e-12

    @staticmethod
    def _mutate_upper_atol(mp):
        mp.upper_atol = 1e-3

    # (method, mutator, needs_empty_graph, exact, pared, pared_schrodinger, rationale).
    # "stale" = the call must throw that the propagator moved, "answers" = it must return exactly what
    # it returned before the mutation, "refreshes" = it must return what a functional built after the
    # mutation returns, "refuses-refresh" = it must throw that it cannot follow the new weights.
    ROWS = (
        (
            "build_graph",
            "_mutate_build_graph",
            False,
            "stale",
            "stale",
            "stale",
            "Appending a layer moves the structure revision, which a pared plan reads as readily as "
            "an exact one.",
        ),
        (
            "propagate",
            "_mutate_propagate",
            True,
            "stale",
            "stale",
            "stale",
            "Re-evolves the operator in place. It leaves the layer count at zero, which is why a "
            "layer count was never enough to see it.",
        ),
        (
            "contract_partially",
            "_mutate_contract_partially",
            False,
            "stale",
            "stale",
            "stale",
            "Consumes the folded layers and rewrites the coefficients. Only inplace=True bumps.",
        ),
        (
            "update_initial_operator",
            "_mutate_update_initial_operator",
            False,
            "refreshes",
            "refreshes",
            "refuses-refresh",
            "A re-weight moves no structure, so the functional follows the new coefficients -- "
            "unless its keep-set was thresholded from those very coefficients, which is "
            "Schrodinger with a pare threshold.",
        ),
        (
            "parameter_mapping",
            "_mutate_parameter_mapping",
            False,
            "stale",
            "stale",
            "stale",
            "Relabels the layers in place, which changes neither the layer count nor the operator -- "
            "the revision is the only thing that sees it.",
        ),
        (
            "cutoff",
            "_mutate_cutoff",
            False,
            "answers",
            "answers",
            "answers",
            "Intended: a cutoff gates the next build and changes nothing the plan holds.",
        ),
        (
            "cutoff_type",
            "_mutate_cutoff_type",
            False,
            "answers",
            "answers",
            "answers",
            "Intended: as cutoff.",
        ),
        (
            "basis_change",
            "_mutate_basis_change",
            False,
            "answers",
            "answers",
            "answers",
            "Intended: as cutoff.",
        ),
        (
            "lower_atol",
            "_mutate_lower_atol",
            False,
            "answers",
            "answers",
            "answers",
            "Intended: as cutoff.",
        ),
        (
            "upper_atol",
            "_mutate_upper_atol",
            False,
            "answers",
            "answers",
            "answers",
            "Intended: as cutoff.",
        ),
    )

    def test_table_covers_every_public_mutator(self):
        assert {row[0] for row in self.ROWS} == {
            "build_graph",
            "propagate",
            "contract_partially",
            "update_initial_operator",
            "parameter_mapping",
            "cutoff",
            "cutoff_type",
            "basis_change",
            "lower_atol",
            "upper_atol",
        }

    @pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
    @pytest.mark.parametrize("partitions", ["off", "auto"])
    @pytest.mark.parametrize("pared", [False, True], ids=["exact", "pared"])
    @pytest.mark.parametrize(
        "schrodinger", [False, True], ids=["heisenberg", "schrodinger"]
    )
    @pytest.mark.parametrize(
        "functional_name",
        [
            "expectation_value_functional",
            "expectation_value_and_gradient_functional",
        ],
    )
    def test_mutator_effect_on_live_functional(
        self,
        monkeypatch,
        serial_comm,
        functional_name,
        schrodinger,
        pared,
        partitions,
        row,
    ):
        (
            method,
            mutator,
            needs_empty_graph,
            exact,
            pared_outcome,
            pared_schrodinger,
            rationale,
        ) = row
        monkeypatch.setenv("monoprop_PARTITIONS", partitions)

        mp = self._propagator(
            serial_comm, schrodinger=schrodinger, with_graph=not needs_empty_graph
        )
        parameters = [] if needs_empty_graph else list(self._PARAMS)
        threshold = self._PARE_THRESHOLD if pared else None
        functional = getattr(mp, functional_name)(threshold)

        def call(fn=None):
            result = (fn or functional)(parameters)
            return result[0] if isinstance(result, tuple) else result

        before = call()
        getattr(self, mutator)(mp)

        expected = exact
        if pared:
            expected = pared_schrodinger if schrodinger else pared_outcome
        context = f"{method}: {rationale}"
        if expected == "stale":
            with pytest.raises(RuntimeError, match=r"MP object has been modified"):
                call()
        elif expected == "refuses-refresh":
            with pytest.raises(RuntimeError, match=r"cannot follow the new weights"):
                call()
        elif expected == "refreshes":
            after = call()
            assert after == call(getattr(mp, functional_name)(threshold)), context
            assert after != pytest.approx(before), context
        else:
            assert call() == pytest.approx(before), context

    @pytest.mark.parametrize("partitions", ["off", "auto"])
    @pytest.mark.parametrize(
        "factory",
        [
            "expectation_value_functional",
            "expectation_value_and_gradient_functional",
        ],
    )
    def test_bound_functional_reports_its_parameter_axis(
        self, monkeypatch, serial_comm, factory, partitions
    ):
        monkeypatch.setenv("monoprop_PARTITIONS", partitions)
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        functional = getattr(mp._simulator, factory)(None)
        assert functional.num_params == len(self._PARAMS)

    @pytest.mark.parametrize("partitions", ["off", "auto"])
    def test_parameter_mapping_invalidates_functional_it_desynchronises(
        self, monkeypatch, serial_comm, partitions
    ):
        """The relabel moves the propagator's own answer, and the functional refuses rather than
        following it half way -- it used to keep returning the pre-relabel number.
        """
        monkeypatch.setenv("monoprop_PARTITIONS", partitions)
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        functional = mp.expectation_value_functional()
        parameters = list(self._PARAMS)
        before = functional(parameters)

        self._mutate_parameter_mapping(mp)

        assert mp.expval(parameters) != pytest.approx(before)
        with pytest.raises(RuntimeError, match=r"set_parameter_mapping"):
            functional(parameters)

    @pytest.mark.parametrize("partitions", ["off", "auto"])
    @pytest.mark.parametrize(
        "factory",
        [
            "expectation_value_functional",
            "expectation_value_and_gradient_functional",
        ],
    )
    def test_bound_functional_reports_whether_it_follows_weights(
        self, monkeypatch, serial_comm, factory, partitions
    ):
        """The contract read off the object: only a Schrodinger plan pared against the operator's own
        coefficients refuses to follow a re-weight.
        """
        monkeypatch.setenv("monoprop_PARTITIONS", partitions)
        for schrodinger in (False, True):
            mp = self._propagator(serial_comm, schrodinger=schrodinger, with_graph=True)
            for threshold in (None, self._PARE_THRESHOLD):
                follows = not (schrodinger and threshold is not None)
                engine = getattr(mp._simulator, factory)(threshold)
                assert engine.follows_weights is follows
                # The front end wraps the engine functional in a callable, which has to carry the
                # rule through: the engine object is not part of the public surface.
                assert getattr(mp, factory)(threshold).follows_weights is follows

    @pytest.mark.parametrize("partitions", ["off", "auto"])
    @pytest.mark.parametrize("pared", [False, True], ids=["exact", "pared"])
    @pytest.mark.parametrize(
        "functional_name",
        [
            "expectation_value_functional",
            "expectation_value_and_gradient_functional",
        ],
    )
    def test_reweighted_functional_matches_a_fresh_propagator(
        self, monkeypatch, serial_comm, functional_name, pared, partitions
    ):
        """The refresh at full precision: a re-weighted propagator's functional must answer what a
        propagator built with those coefficients answers, to the last bit. Both run the same
        arithmetic over the same store order, so there is no rounding to hide behind.
        """
        monkeypatch.setenv("monoprop_PARTITIONS", partitions)
        threshold = self._PARE_THRESHOLD if pared else None
        parameters = list(self._PARAMS)

        def value(result):
            return result[0] if isinstance(result, tuple) else result

        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        functional = getattr(mp, functional_name)(threshold)
        before = value(functional(parameters))
        self._mutate_update_initial_operator(mp)

        fresh = self._propagator(
            serial_comm, schrodinger=False, with_graph=True, first_weight=2.75
        )
        expected = value(getattr(fresh, functional_name)(threshold)(parameters))

        after = value(functional(parameters))
        assert after == expected
        assert after != pytest.approx(before)

    @pytest.mark.parametrize("partitions", ["off", "auto"])
    def test_pared_schrodinger_functional_refuses_to_follow_a_reweight(
        self, monkeypatch, serial_comm, partitions
    ):
        """Schrodinger thresholds its keep-set from the operator coefficients, so new coefficients
        select a different keep-set: the plan says so rather than replaying a paring nobody asked
        for. The unpared functional over the same propagator follows the re-weight, which is what
        makes the refusal the paring's and not the picture's.
        """
        monkeypatch.setenv("monoprop_PARTITIONS", partitions)
        mp = self._propagator(serial_comm, schrodinger=True, with_graph=True)
        parameters = list(self._PARAMS)
        pared = mp.expectation_value_functional(self._PARE_THRESHOLD)
        exact = mp.expectation_value_functional()
        pared(parameters)

        self._mutate_update_initial_operator(mp)

        with pytest.raises(RuntimeError, match=r"cannot follow the new weights"):
            pared(parameters)
        assert exact(parameters) == pytest.approx(mp.expval(parameters))

    @pytest.mark.parametrize("partitions", ["off", "auto"])
    def test_pared_functional_is_invalidated_by_build_graph(
        self, monkeypatch, serial_comm, partitions
    ):
        """A pared plan owns its layers, so its layer count cannot move; the structure revision is
        what sees the appended layer.
        """
        monkeypatch.setenv("monoprop_PARTITIONS", partitions)
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        functional = mp.expectation_value_functional(self._PARE_THRESHOLD)
        parameters = list(self._PARAMS)
        before = functional(parameters)

        self._mutate_build_graph(mp)

        # The appended gate claims a fresh angle, so the propagator's own axis is now three long.
        assert mp.expval([*parameters, 0.4]) != pytest.approx(before)
        with pytest.raises(RuntimeError, match=r"build_graph"):
            functional(parameters)


class TestEvolvedOperatorBothPictures:
    def test_schrodinger_returns_state_dict(self, serial_comm):
        operator = MajoranaOperator({(0, 1): 1.0j}, num_modes=2)
        mp = MajoranaPropagator(
            operator, [0, 1], cutoff=4, schrodinger_cutoff=2, comm=serial_comm
        )
        result = mp.evolved_operator()
        assert isinstance(result, MajoranaOperator)
