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

import numpy as np
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


def _components(result):
    """A functional's answer as ``(value, gradient)``; ``None`` gradient for the value-only kind."""
    return result if isinstance(result, tuple) else (result, None)


def _assert_answers_match(actual, expected, *, exact, context=""):
    """Assert both components of two answers agree, bit-exactly or to ``pytest.approx``.

    The two functional kinds return different shapes, so a test parametrized over both has to
    compare whatever the kind under test returned -- a gradient dropped here is a gradient nobody
    checks.
    """
    value, gradient = _components(actual)
    expected_value, expected_gradient = _components(expected)
    assert (gradient is None) == (expected_gradient is None), context
    if exact:
        assert value == expected_value, context
        if gradient is not None:
            assert np.array_equal(gradient, expected_gradient), context
    else:
        assert value == pytest.approx(expected_value), context
        if gradient is not None:
            assert gradient == pytest.approx(expected_gradient), context


def _assert_answers_differ(actual, other, context=""):
    """Assert both components of two answers moved (a gradient counts as moved if any entry did)."""
    value, gradient = _components(actual)
    other_value, other_gradient = _components(other)
    assert value != pytest.approx(other_value), context
    if gradient is not None:
        assert gradient != pytest.approx(other_gradient), context


# The propagator method each functional factory is the reusable form of, so a test can compare a
# functional's answer against the direct call of the same shape.
_DIRECT_CALL = {
    "expectation_value_functional": "expectation_value",
    "expectation_value_and_gradient_functional": "expectation_value_and_gradient",
}


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
    @pytest.mark.parametrize("functional_name", list(_DIRECT_CALL))
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
        before = functional(parameters)

        mp.update_initial_operator(updated_operator)

        # A re-weight moves no structure, so the functional built before it follows the new
        # coefficients instead of refusing the call: it now answers what the propagator answers.
        after = functional(parameters)
        _assert_answers_differ(after, before)
        direct = getattr(mp, _DIRECT_CALL[functional_name])(parameters)
        _assert_answers_match(after, direct, exact=False)
        _assert_answers_match(
            after, getattr(mp, functional_name)()(parameters), exact=True
        )


class TestFunctionalValidityTable:
    """What each public mutator does to a functional built before it ran.

    The Python mirror of ``cpp/tests/functional_validity.cpp``: same rows, same expectations, run
    over both partition settings. The gate that a new mutator gets a row is the C++ static_assert
    against ``MonomialPropagator::num_mutating_methods``; these rows mirror that table.
    """

    _MODES = 2
    _CUTOFF = 4
    _PARAMS = (0.3, 0.7)
    _PARE_THRESHOLD = 1e-12
    # What _mutate_update_initial_operator() writes onto term (0, 1); a re-weighted propagator must
    # answer exactly like one built with it from the start, so both sides read it from here.
    _REWEIGHTED_FIRST_WEIGHT = 2.75

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
    def _propagator(
        cls, comm, *, schrodinger, with_graph, first_weight=1.0, core_term=None
    ):
        # The identity row is only carried by the core-term case; every other case leaves it out,
        # which is what makes a re-weight that also leaves it out a no-op there.
        terms = {(0, 1): first_weight * 1j, (2, 3): 0.5j}
        if core_term is not None:
            terms[()] = core_term
        mp = MajoranaPropagator(
            MajoranaOperator(terms, num_modes=cls._MODES),
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
            MajoranaOperator(
                {(0, 1): cls._REWEIGHTED_FIRST_WEIGHT * 1j, (2, 3): 0.5j},
                num_modes=cls._MODES,
            )
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

    # (method, mutator, needs_empty_graph, outcome, pared_schrodinger, rationale). Paring only changes
    # the verdict where the keep-set came from the operator coefficients, so the Schrodinger-pared
    # column is the only one that can differ from the general one -- as in the C++ table.
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
            "Appending a layer moves the structure revision, which a pared plan reads as readily as "
            "an exact one.",
        ),
        (
            "propagate",
            "_mutate_propagate",
            True,
            "stale",
            "stale",
            "Re-evolves the operator in place. It leaves the layer count at zero, so the revision "
            "is the only thing that sees it.",
        ),
        (
            "contract_partially",
            "_mutate_contract_partially",
            False,
            "stale",
            "stale",
            "Consumes the folded layers and rewrites the coefficients. Only inplace=True bumps.",
        ),
        (
            "update_initial_operator",
            "_mutate_update_initial_operator",
            False,
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
            "Relabels the layers in place, which changes neither the layer count nor the operator -- "
            "the revision is the only thing that sees it.",
        ),
        (
            "cutoff",
            "_mutate_cutoff",
            False,
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
            "Intended: as cutoff.",
        ),
        (
            "basis_change",
            "_mutate_basis_change",
            False,
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
            "Intended: as cutoff.",
        ),
        (
            "upper_atol",
            "_mutate_upper_atol",
            False,
            "answers",
            "answers",
            "Intended: as cutoff.",
        ),
    )

    @pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
    @pytest.mark.parametrize("pared", [False, True], ids=["exact", "pared"])
    @pytest.mark.parametrize(
        "schrodinger", [False, True], ids=["heisenberg", "schrodinger"]
    )
    @pytest.mark.parametrize("functional_name", list(_DIRECT_CALL))
    @pytest.mark.usefixtures("partitions")
    def test_mutator_effect_on_live_functional(
        self,
        serial_comm,
        functional_name,
        schrodinger,
        pared,
        row,
    ):
        (
            method,
            mutator,
            needs_empty_graph,
            outcome,
            pared_schrodinger,
            rationale,
        ) = row

        mp = self._propagator(
            serial_comm, schrodinger=schrodinger, with_graph=not needs_empty_graph
        )
        parameters = [] if needs_empty_graph else list(self._PARAMS)
        threshold = self._PARE_THRESHOLD if pared else None
        functional = getattr(mp, functional_name)(threshold)

        before = functional(parameters)
        getattr(self, mutator)(mp)

        expected = pared_schrodinger if pared and schrodinger else outcome
        context = f"{method}: {rationale}"
        if expected == "stale":
            with pytest.raises(RuntimeError, match=r"MP object has been modified"):
                functional(parameters)
        elif expected == "refuses-refresh":
            with pytest.raises(RuntimeError, match=r"cannot follow the new weights"):
                functional(parameters)
        elif expected == "refreshes":
            after = functional(parameters)
            fresh = getattr(mp, functional_name)(threshold)
            _assert_answers_match(after, fresh(parameters), exact=True, context=context)
            _assert_answers_differ(after, before, context=context)
        else:
            _assert_answers_match(
                functional(parameters), before, exact=False, context=context
            )

    @pytest.mark.parametrize("factory", list(_DIRECT_CALL))
    @pytest.mark.usefixtures("partitions")
    def test_bound_functional_reports_its_parameter_axis(self, serial_comm, factory):
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        assert getattr(mp._simulator, factory)(None).num_params == len(self._PARAMS)
        # The front end hands back its own callable, which forwards what the engine object exposes.
        assert getattr(mp, factory)(None).num_params == len(self._PARAMS)

    @pytest.mark.usefixtures("partitions")
    def test_parameter_mapping_invalidates_functional_it_desynchronises(
        self, serial_comm
    ):
        """The relabel moves the propagator's own answer, and the functional refuses the call rather
        than following it half way.
        """
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        functional = mp.expectation_value_functional()
        parameters = list(self._PARAMS)
        before = functional(parameters)

        self._mutate_parameter_mapping(mp)

        assert mp.expval(parameters) != pytest.approx(before)
        with pytest.raises(RuntimeError, match=r"set_parameter_mapping"):
            functional(parameters)

    @pytest.mark.parametrize("factory", list(_DIRECT_CALL))
    @pytest.mark.usefixtures("partitions")
    def test_bound_functional_reports_whether_it_follows_weights(
        self, serial_comm, factory
    ):
        """The contract read off the object: only a Schrodinger plan pared against the operator's own
        coefficients refuses to follow a re-weight.
        """
        for schrodinger in (False, True):
            mp = self._propagator(serial_comm, schrodinger=schrodinger, with_graph=True)
            for threshold in (None, self._PARE_THRESHOLD):
                follows = not (schrodinger and threshold is not None)
                engine = getattr(mp._simulator, factory)(threshold)
                assert engine.follows_weights is follows
                # The engine object is not part of the public surface, so the rule has to be readable
                # off what the front end returns.
                assert getattr(mp, factory)(threshold).follows_weights is follows

    @pytest.mark.parametrize("pared", [False, True], ids=["exact", "pared"])
    @pytest.mark.parametrize("functional_name", list(_DIRECT_CALL))
    @pytest.mark.usefixtures("partitions")
    def test_reweighted_functional_matches_a_fresh_propagator(
        self, serial_comm, functional_name, pared
    ):
        """The refresh at full precision: a re-weighted propagator's functional must answer what a
        propagator built with those coefficients answers, to the last bit. Both run the same
        arithmetic over the same store order, so there is no rounding to hide behind.
        """
        threshold = self._PARE_THRESHOLD if pared else None
        parameters = list(self._PARAMS)

        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        functional = getattr(mp, functional_name)(threshold)
        before = functional(parameters)
        self._mutate_update_initial_operator(mp)

        fresh = self._propagator(
            serial_comm,
            schrodinger=False,
            with_graph=True,
            first_weight=self._REWEIGHTED_FIRST_WEIGHT,
        )
        expected = getattr(fresh, functional_name)(threshold)(parameters)

        after = functional(parameters)
        _assert_answers_match(after, expected, exact=True)
        _assert_answers_differ(after, before)

    @pytest.mark.usefixtures("partitions")
    def test_pared_schrodinger_functional_refuses_to_follow_a_reweight(
        self, serial_comm
    ):
        """Schrodinger thresholds its keep-set from the operator coefficients, so new coefficients
        select a different keep-set: the plan says so rather than replaying a paring nobody asked
        for. The unpared functional over the same propagator follows the re-weight, which is what
        makes the refusal the paring's and not the picture's.
        """
        mp = self._propagator(serial_comm, schrodinger=True, with_graph=True)
        parameters = list(self._PARAMS)
        pared = mp.expectation_value_functional(self._PARE_THRESHOLD)
        exact = mp.expectation_value_functional()
        pared(parameters)

        self._mutate_update_initial_operator(mp)

        with pytest.raises(RuntimeError, match=r"cannot follow the new weights"):
            pared(parameters)
        assert exact(parameters) == pytest.approx(mp.expval(parameters))

    @pytest.mark.usefixtures("partitions")
    def test_no_op_mutators_keep_a_functional_valid(self, serial_comm):
        """A call that appends or folds nothing is not a mutation, on either partition setting.

        The facade decides that before it fans out, so ``off`` and ``auto`` must agree -- an
        unconditional bump on the fan-out is exactly the divergence this table exists to rule out.
        """
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=False)
        functional = mp.expectation_value_functional()
        before = functional([])

        # No graph, so there is nothing to fold and no layer to retire; the return is the current
        # coefficients either way.
        folded = mp.contract_partially([], inplace=True)
        assert folded == pytest.approx(mp.contract_partially([], inplace=False))
        empty = Circuit((), self._MODES, ())
        mp.build_graph(empty)
        mp.propagate(empty)
        assert mp.n_parameters == 0

        assert functional([]) == pytest.approx(before)

    # A gate generator is bounds-checked only as the gate loop reaches it, so a bad one in a
    # multi-gate call throws with the earlier gates already committed. The front end validates
    # generators against num_modes, so the engine is the only way to express one -- as in
    # _mutate_basis_change.
    _PART_WAY_GATES = ((0,), (2 * _MODES + 1,), (2,))
    _PART_WAY_MAPPING = (0, 1, 2)
    _PART_WAY_COEFFS = (1.0, 1.0, 1.0)

    def test_part_way_build_graph_failure_invalidates_the_functional(
        self, serial_comm, partitions
    ):
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        functional = mp.expectation_value_functional()
        parameters = list(self._PARAMS)
        functional(parameters)
        layers_before = mp.graph_layers

        with pytest.raises(RuntimeError):
            mp._simulator.build_graph(
                self._PART_WAY_GATES, self._PART_WAY_MAPPING, self._PART_WAY_COEFFS
            )

        # The graph grew, so the call did mutate on its way to the throw. Only the single-partition
        # shape can witness that: the bad generator is rejected independently on every partition, so
        # whichever reaches it first poisons the collective the slower ones are still inside, and a
        # poisoned partition unwinds before committing the layer. Whether that happened decides what
        # the facade read even means -- it answers when all of them committed and refuses when a
        # poisoned one did not (see test_part_way_fan_out_failure_refuses_the_facade).
        if partitions == "off":
            assert mp.graph_layers > layers_before

        with pytest.raises(RuntimeError, match=r"build_graph"):
            functional(parameters)

    def test_part_way_fan_out_failure_refuses_the_facade(
        self, monkeypatch, serial_comm
    ):
        """A fan-out that throws part-way leaves the partitions holding different state.

        A term the operator does not hold is rejected by the one partition that would own it, with
        its siblings already re-weighted -- the one mutation that diverges deterministically rather
        than by whichever partition loses a race. No read can reconcile the two, so the facade
        latches the fault and refuses every later call rather than answering for a partitioning that
        never existed. Two partitions are asked for by number rather than through ``auto``, which
        resolves to one -- and so to no facade at all -- on a single-core host.
        """
        monkeypatch.setenv("monoprop_PARTITIONS", "2")
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        parameters = list(self._PARAMS)
        functional = mp.expectation_value_functional()
        functional(parameters)

        with pytest.raises(RuntimeError, match=r"not found in the operator"):
            mp.update_initial_operator(
                MajoranaOperator(
                    {
                        (0, 1): self._REWEIGHTED_FIRST_WEIGHT * 1j,
                        (0, 2): 1.0j,  # absent: only its owning partition rejects it
                    },
                    num_modes=self._MODES,
                )
            )

        refuses = r"update_initial_operator\(\) failed part-way"
        with pytest.raises(RuntimeError, match=refuses):
            _ = mp.graph_layers
        with pytest.raises(RuntimeError, match=refuses):
            mp.size()
        with pytest.raises(RuntimeError, match=refuses):
            mp.expval(parameters)
        with pytest.raises(RuntimeError, match=refuses):
            mp.expectation_value_functional()

        # The functional built beforehand reports the mutation rather than the fault: its plan checks
        # the revision, which the same unwind bumped, before it reaches a partition.
        with pytest.raises(RuntimeError, match=r"update_initial_operator"):
            functional(parameters)

    def test_identically_rejected_fan_out_leaves_the_facade_usable(
        self, monkeypatch, serial_comm
    ):
        """The other half of the same rule, and why the fault is not just "a fan-out threw".

        Every partition rejects this call for the same reason and none of them gets as far as
        changing anything, so the facade is still whole and the fixed-up retry has to go through.
        """
        monkeypatch.setenv("monoprop_PARTITIONS", "2")
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=True)
        parameters = list(self._PARAMS)
        before = mp.expval(parameters)

        # A coefficient-informed extension one parameter short of replaying the stored graph.
        extension = Circuit((ExpGate(self._generator(1)),), self._MODES, (0.4,))
        with pytest.raises(RuntimeError):
            mp.build_graph(extension, seed_parameters=[parameters[0]])

        assert mp.graph_layers == 2
        assert mp.expval(parameters) == pytest.approx(before)
        # The retry seeds the whole accumulated axis, the appended gate's angle included.
        mp.build_graph(extension, seed_parameters=[*parameters, 0.4])
        assert mp.graph_layers == 3

    @pytest.mark.usefixtures("partitions")
    def test_part_way_propagate_failure_invalidates_the_functional(self, serial_comm):
        """propagate() folds into the operator instead of appending, so nothing counts the
        mutation: without the failure-path bump the functional would keep answering for
        coefficients that had moved.
        """
        mp = self._propagator(serial_comm, schrodinger=False, with_graph=False)
        functional = mp.expectation_value_functional()
        functional([])

        with pytest.raises(RuntimeError):
            mp._simulator.propagate(
                self._PART_WAY_GATES,
                self._PART_WAY_MAPPING,
                self._PART_WAY_COEFFS,
                [0.3, 0.7, 0.5],
            )

        with pytest.raises(RuntimeError, match=r"propagate"):
            functional([])

    @pytest.mark.parametrize("functional_name", list(_DIRECT_CALL))
    @pytest.mark.usefixtures("partitions")
    def test_reweight_that_drops_the_core_term_zeroes_it(
        self, serial_comm, functional_name
    ):
        """The identity row describes the dict that committed, like every other row: a re-weight
        that leaves it out zeroes it, which is what the store does with every row a dict omits.
        """
        parameters = list(self._PARAMS)
        mp = self._propagator(
            serial_comm, schrodinger=False, with_graph=True, core_term=0.25
        )
        functional = getattr(mp, functional_name)(None)
        before = functional(parameters)

        self._mutate_update_initial_operator(mp)  # carries no identity row

        fresh = self._propagator(
            serial_comm,
            schrodinger=False,
            with_graph=True,
            first_weight=self._REWEIGHTED_FIRST_WEIGHT,
        )
        after = functional(parameters)
        _assert_answers_match(
            after, getattr(fresh, functional_name)(None)(parameters), exact=True
        )
        _assert_answers_match(
            after, getattr(mp, _DIRECT_CALL[functional_name])(parameters), exact=False
        )
        _assert_answers_differ(after, before)

    @pytest.mark.usefixtures("partitions")
    def test_pared_functional_is_invalidated_by_build_graph(self, serial_comm):
        """A pared plan owns its layers, so its layer count cannot move; the structure revision is
        what sees the appended layer.
        """
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
