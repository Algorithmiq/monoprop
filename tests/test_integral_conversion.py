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

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from monoprop.fermi import FermiOperator, FermiString
from monoprop.integral_conversion import integrals_to_fermion


def _read_openfermion(path: Path) -> FermiOperator:
    with path.open() as f:
        terms = json.load(f)
    coeffs = []
    fermi_strings = []
    for key, value in terms.items():
        fixed_key = ast.literal_eval(key) if key != "()" else ()
        coeffs.append(value)
        fermi_terms = [(el[0], "-" if el[1] == 0 else "+") for el in fixed_key]
        fermi_strings.append(FermiString(fermi_terms))
    return FermiOperator(fermi_strings, coeffs)


@pytest.fixture
def fermion_operator(lazy_shared_datadir):
    return _read_openfermion(lazy_shared_datadir / "h2o_fermion.json")


@pytest.fixture
def integral_hamiltonian(lazy_shared_datadir):
    data = np.load(lazy_shared_datadir / "integrals_H2O.npz")
    return (float(data["h0"]), data["h1"], data["h2"])


def test_to_fermion(integral_hamiltonian, fermion_operator):
    fermion_ham = integrals_to_fermion(integral_hamiltonian)
    assert fermion_operator.isclose(fermion_ham)
