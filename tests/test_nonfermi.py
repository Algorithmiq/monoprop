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

import numpy as np

from monoprop import Circuit, MajoranaPropagator
from monoprop.majorana import MajoranaOperator


def test_nonfermi(serial_comm):
    # Oracle: a hand-computed evolution of a non-fermionic operator.
    num_modes = 2
    majoranas = [(0,), (0, 1), (0, 1, 3), (2, 3)]
    gen_coeffs = np.array([-1.0, 1.0, -1.0, 1.0])
    param_inds = np.array([0, 1, 2, 3])
    parameters = np.array([0.1, 0.2, 0.2, 0.2])

    fermionic_operator = MajoranaOperator({(1, 2, 3): -1j}, num_modes=num_modes)
    circuit = Circuit.from_dense_arrays(
        majoranas=majoranas,
        gen_coeffs=gen_coeffs,
        param_inds=param_inds,
        system_size=num_modes,
        parameters=parameters,
        initial_state=[],
    )

    exact_evolved_op = MajoranaOperator(
        {
            (1, 2, 3): -0.8314427691150754j,
            (0, 1, 2, 3): (-0.16854179325074592 + 0j),
            (0, 2, 3): 0.35867804544976145j,
            (0, 2): 0.35152836455073294j,
            (2,): (0.07125832726038464 + 0j),
            (1, 2): 0.1516466453264173j,
        },
        2,
    )

    mp = MajoranaPropagator(
        fermionic_operator,
        circuit.initial_state,
        cutoff=4,
        comm=serial_comm,
    )
    mp.propagate(circuit)
    test_evolved_op = mp.evolved_operator()

    assert len(test_evolved_op) == len(exact_evolved_op)
    assert test_evolved_op.isclose(exact_evolved_op, atol=1e-12)
