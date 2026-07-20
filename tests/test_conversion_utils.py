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

import numpy as np
import pytest

from monoprop.conversion_utils import _majorana_to_pauli, _pauli_to_majorana


@pytest.mark.parametrize(
    "pauli",
    ["IIII", "X", "Y", "Z", "IXXZI", "XZYYYXX", "IZZI"],
)
def test_pauli_majorana_roundtrip(pauli: str) -> None:
    majorana, coeff = _pauli_to_majorana(pauli)
    decoded, inv_coeff = _majorana_to_pauli(majorana, n_qubits=len(pauli))

    assert decoded == pauli
    np.testing.assert_allclose(inv_coeff, np.conjugate(coeff))
