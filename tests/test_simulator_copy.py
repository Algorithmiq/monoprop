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

"""copy.deepcopy on the simulator produces a fully independent deep copy.

Deep copy is supported end-to-end: the public ``MonomialPropagator`` wrapper, the
``_SimulatorAdapter``, and the bound C++ ``_core`` all implement ``__deepcopy__``. The C++ operator
store owns a self back-pointer, so the copy rebuilds it with the pointer repaired. The MPI
communicator is shared (not duplicated), which is correct both with and without MPI.
"""

from __future__ import annotations

import copy

from monoprop import MonomialPropagator
from monoprop.fermi_data import FermiCircuit, MajoranaOperator


def _build(serial_comm):
    initial_op = MajoranaOperator([(0, 1, 2, 4)], [1], 8)
    quantum_circuit = FermiCircuit(initial_state=[], gates=[])
    return MonomialPropagator(initial_op, quantum_circuit, 16, comm=serial_comm)


def test_public_deepcopy_is_independent_and_shares_comm(serial_comm):
    mp = _build(serial_comm)
    clone = copy.deepcopy(mp)

    # Independent objects all the way down, but the communicator is shared (with or without MPI).
    assert clone is not mp
    assert clone._simulator is not mp._simulator
    assert clone._simulator._core is not mp._simulator._core
    assert clone._comm is mp._comm

    # Same content to start.
    assert clone.evolved_operator_dict() == mp.evolved_operator_dict() == {(0, 1, 2, 4): 1}

    # Mutating the source's operator must not affect the deep copy.
    mp.update_coeffs({(0, 1, 2, 4): 5.0 + 0j})
    assert mp.evolved_operator_dict() != {(0, 1, 2, 4): 1}
    assert clone.evolved_operator_dict() == {(0, 1, 2, 4): 1}


def test_core_deepcopy_index_backpointer_valid(serial_comm):
    """A deep-copied _core must keep a usable index: update_initial_operator looks each term up via
    find(), which only works if the store's self back-pointer was repaired during the clone."""
    mp = _build(serial_comm)
    clone_core = copy.deepcopy(mp._simulator._core)

    def evolved(core):
        return core.evolved_operator_dict(
            parameters=[], parameter_mapping=[], gen_coeffs=[], atol=1e-12
        )

    assert evolved(clone_core) == {(0, 1, 2, 4): 1}

    # Looks the term up in the clone's own rows, then rewrites its coefficient.
    clone_core.update_initial_operator({(0, 1, 2, 4): 9.0 + 0j})
    assert evolved(clone_core)[(0, 1, 2, 4)] != 1
