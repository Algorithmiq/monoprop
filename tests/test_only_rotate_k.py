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

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from monoprop import Circuit, ExpGate, MajoranaPropagator, PauliPropagator
from monoprop.majorana import MajoranaOperator
from monoprop.pauli import PauliOperator
from tests.cases import CasesFermionicProblemOrbitalRotations


@pytest.fixture
def serial_mp_kwargs(serial_comm):
    """Common MP constructor arguments — always single-rank."""
    return {"cutoff": 6, "schrodinger_cutoff": 8, "comm": serial_comm}


def _split_orbital_gates(gates):
    """Split gates into (non-orbital, orbital), where orbital gates are the tail run
    whose generated monomials are all length-2 (free-fermion rotations)."""

    def is_orbital(gate):
        return all(len(majorana) == 2 for majorana in gate.generator.terms)

    for i in range(len(gates)):
        if all(is_orbital(gate) for gate in gates[i:]):
            return gates[:i], gates[i:]
    return gates, []


@pytest.mark.parametrize(
    ("propagator_cls", "operator", "gate_generator", "system_size"),
    [
        (
            MajoranaPropagator,
            MajoranaOperator({(0,): 1.0}, num_modes=1),
            MajoranaOperator({(0, 1): 1.0j}, num_modes=1),
            1,
        ),
        (
            PauliPropagator,
            PauliOperator({"Z": 1.0}, num_qubits=1),
            PauliOperator({"X": 1.0}, num_qubits=1),
            1,
        ),
    ],
)
@pytest.mark.parametrize(
    ("builder_method", "result_method")
    [
        ("build_graph", "expval_functional"),
        ("propagate", "expval"),
    ]
)
def test_only_rotate_len_k_none_is_uncapped(
    propagator_cls, operator, gate_generator, system_size, serial_mp_kwargs, builder_method, result_method
):
    circuit = Circuit(
        (ExpGate(gate_generator, index=0),),
        initial_state=(),
        system_size=system_size,
        parameters=(0.321,),
    )
    parameters = circuit.parameters

    mp_default = propagator_cls(operator, circuit.initial_state, **serial_mp_kwargs)
    getattr(mp_default, builder_method)(circuit)
    expval_default = getattr(mp_default, result_method)(parameters)

    mp_none = propagator_cls(operator, circuit.initial_state, **serial_mp_kwargs)
    getattr(mp_none, builder_method)(circuit, only_rotate_len_k=None)
    expval_default = getattr(mp_none, result_method)(parameters)

    assert mp_none.graph_size() == mp_default.graph_size()
    assert mp_none.size() == mp_default.size()
    assert np.isclose(expval_none, expval_default, atol=1e-12)


def test_basic_orbital_rotation(serial_comm):
    n_modes = 4

    operator = MajoranaOperator({}, n_modes)
    sequence = Circuit.from_dense_arrays(
        initial_state=[],
        majoranas=[(1, 2)],
        parameters=[np.pi / 4],
        gen_coeffs=[1.0],
        param_inds=[0],
        system_size=n_modes,
    )
    circuit = sequence
    kwargs = {"cutoff": 6, "schrodinger_cutoff": 8, "comm": serial_comm}

    mp_act = MajoranaPropagator(operator, sequence.initial_state, **kwargs)
    initial_state = mp_act.evolved_operator()
    mp_act.propagate(circuit)
    rotated_state = mp_act.evolved_operator()

    mp_orb = MajoranaPropagator(operator, sequence.initial_state, **kwargs)
    mp_orb.propagate(circuit, only_rotate_len_k=4)
    orb_rotated_state = mp_orb.evolved_operator()

    for key in initial_state.terms:
        if len(key) > 4:
            assert np.isclose(
                initial_state.terms[key], orb_rotated_state.terms[key], atol=1e-12
            )

    for key in rotated_state.terms:
        if len(key) <= 4:
            assert np.isclose(
                rotated_state.terms[key], orb_rotated_state.terms[key], atol=1e-12
            )


@parametrize_with_cases(
    "problem", cases=CasesFermionicProblemOrbitalRotations, has_tag="only_rotate_len_k"
)
@pytest.mark.parametrize("inplace", [False, True])
def test_only_rotate_len_k(problem, inplace, serial_mp_kwargs):
    """Tests size/graph_size (rank-local) so uses serial_comm."""
    circuit = problem.monomial_circuit.to_circuit()
    gates = circuit.gates
    parameters = problem.monomial_circuit.parameters
    non_orbital_gates, orbital_gates = _split_orbital_gates(gates)
    # Consecutive prefix/suffix split, so the (identity) parameter values split the same way.
    split = len(non_orbital_gates)
    # _with_index rather than ExpGate(gate.generator) so _structural is preserved -- see
    # tests/test_circuit.py::_rebase.
    non_orbital = Circuit(
        tuple(ExpGate._with_index(gate, None) for gate in non_orbital_gates),
        system_size=problem.n_modes,
        parameters=tuple(parameters[:split]),
    )
    orbital = Circuit(
        tuple(ExpGate._with_index(gate, None) for gate in orbital_gates),
        system_size=problem.n_modes,
        parameters=tuple(parameters[split:]),
    )

    mp_act = MajoranaPropagator(
        problem.operator, problem.monomial_circuit.initial_state, **serial_mp_kwargs
    )
    mp_act.build_graph(circuit)
    act_ener = mp_act.expval_functional()(parameters)

    mp = MajoranaPropagator(
        problem.operator, problem.monomial_circuit.initial_state, **serial_mp_kwargs
    )

    if inplace:
        mp.propagate(non_orbital)
        mp.propagate(orbital, only_rotate_len_k=4)
        test_expval = mp.expval()
    else:
        mp.build_graph(non_orbital)
        mp.build_graph(orbital, only_rotate_len_k=4)
        test_expval = mp.expval_functional()(parameters)
        assert sum(mp.graph_size()) < sum(mp_act.graph_size())

    assert mp.size() < mp_act.size()
    assert np.isclose(test_expval, act_ener, atol=1e-12)


@pytest.mark.parametrize(
    ("only_rotate_len_k", "err"),
    [
        (
            -1,
            pytest.raises(
                ValueError,
                match=r"only_rotate_len_k=-1 is out of range; must be 0 < k <= 2\*num_qubits",
            ),
        ),
        (
            0,
            pytest.raises(
                ValueError,
                match=r"only_rotate_len_k=0 is out of range; must be 0 < k <= 2\*num_qubits",
            ),
        ),
        (None, does_not_raise()),
        (
            9,
            pytest.raises(
                ValueError,
                match=r"only_rotate_len_k=9 is out of range; must be 0 < k <= 2\*num_qubits",
            ),
        ),
        (8, does_not_raise()),
    ],
)
@pytest.mark.parametrize("method_name", ["build_graph", "propagate"])
def test_only_rotate_len_k_errors_majorana(only_rotate_len_k, err, method_name):
    # MajoranaPropagator has no qubit count, so only the k > 0 half of the check applies.
    mp = MajoranaPropagator(MajoranaOperator({}, 4), [], cutoff=6, schrodinger_cutoff=8)
    with err:
        getattr(mp, method_name)(
            Circuit((), initial_state=(), system_size=4),
            only_rotate_len_k=only_rotate_len_k,
        )


@pytest.mark.parametrize(
    ("only_rotate_len_k", "err"),
    [
        (
            -1,
            pytest.raises(
                ValueError,
                match=r"only_rotate_len_k=-1 is out of range; must be 0 < k <= 2\*num_qubits",
            ),
        ),
        (
            0,
            pytest.raises(
                ValueError,
                match=r"only_rotate_len_k=0 is out of range; must be 0 < k <= 2\*num_qubits",
            ),
        ),
        (None, does_not_raise()),
        (
            9,
            pytest.raises(
                ValueError,
                match=r"only_rotate_len_k=9 is out of range; must be 0 < k <= 2\*num_qubits",
            ),
        ),
        (8, does_not_raise()),
    ],
)
@pytest.mark.parametrize("method_name", ["build_graph", "propagate"])
def test_only_rotate_len_k_errors_pauli(only_rotate_len_k, err, method_name):
    mp = PauliPropagator(PauliOperator({}, 4), [], cutoff=6, schrodinger_cutoff=8)
    with err:
        getattr(mp, method_name)(
            Circuit((), initial_state=(), system_size=4),
            only_rotate_len_k=only_rotate_len_k,
        )
