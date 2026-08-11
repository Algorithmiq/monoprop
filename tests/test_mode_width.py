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

"""There is no compile-time ceiling on the mode count.

The engine used to be one C++ class template per 32-mode block, capped at build time, with a Python
dispatch table routing a mode count to the right instantiation. It is now a single class that takes its
logical width as an argument. These tests pin the part of that which is observable from Python: mode
counts past the old cap work, they agree with a narrow run of the same physics, and the storage width the
engine picks rounds up rather than being fixed by a template argument.
"""

from __future__ import annotations

import pytest

from monoprop import Circuit, ExpGate, MajoranaPropagator, PauliPropagator
from monoprop.majorana import MajoranaOperator
from monoprop.pauli import PauliOperator

# 250 was the shipped ceiling, so 260 is past it and still cheap. 4096 is where a monomial no longer
# fits Bitset's 8 inline words and spills to the heap -- correct, just slower.
_PAST_THE_OLD_CEILING = 260
_WIDE = 4096
_NARROW = 8  # the reference width, inside every old mode tier


def _energy(num_modes, angle, comm):
    """One rotation applied to a weight-2 observable, all inside Majoranas 0..3.

    Padding the system with unoccupied, ungated modes cannot change the answer, so this is the same
    number at every num_modes >= 2 -- which is what makes it a check of the width handling rather than
    of a value that happens to have been recorded.
    """
    observable = MajoranaOperator({(0, 1): 1j}, num_modes)
    generator = MajoranaOperator({(2, 3): 1j}, num_modes)
    circuit = Circuit(
        gates=[ExpGate(generator)], system_size=num_modes, initial_state=[]
    )

    mp = MajoranaPropagator(observable, [], cutoff=2 * _NARROW, comm=comm)
    assert mp.num_modes == num_modes
    mp.build_graph(circuit)
    return mp.expectation_value([angle])


@pytest.mark.parametrize("num_modes", [_PAST_THE_OLD_CEILING, _WIDE])
def test_wide_run_agrees_with_a_narrow_run_of_the_same_physics(num_modes, serial_comm):
    reference = _energy(_NARROW, 0.37, serial_comm)
    assert _energy(num_modes, 0.37, serial_comm) == pytest.approx(reference, abs=1e-12)


@pytest.mark.parametrize(
    ("num_modes", "expected_storage"),
    [(5, 32), (32, 32), (33, 64), (64, 64), (65, 96), (260, 288)],
)
def test_storage_width_rounds_up_to_a_whole_block(
    num_modes, expected_storage, serial_comm
):
    """Monomials are stored at a whole 32-mode block, with a one-block floor."""
    observable = MajoranaOperator({(0, 1): 1j}, num_modes)
    mp = MajoranaPropagator(observable, [], cutoff=4, comm=serial_comm)
    assert mp.num_modes == num_modes
    assert mp._simulator.storage_num_modes == expected_storage


def test_pauli_propagator_runs_past_the_old_compile_time_ceiling(serial_comm):
    num_qubits = _PAST_THE_OLD_CEILING
    observable = PauliOperator({"Z" + "I" * (num_qubits - 1): 1.0}, num_qubits)
    mp = PauliPropagator(observable, [], cutoff=4, comm=serial_comm)
    assert mp.num_modes == num_qubits
    # Z on qubit 0 against the all-zero reference state.
    assert mp.expectation_value([]) == pytest.approx(1.0)
