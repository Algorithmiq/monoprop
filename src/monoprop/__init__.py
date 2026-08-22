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

"""monoprop: classical Majorana and Pauli monomial propagation."""

from __future__ import annotations

import importlib.util

# Must precede ._core: on a fat-binary wheel there is no monoprop/_core to import until this module
# has bound one of the shipped ISA variants to that name. The module name sorts ahead of _core so
# alphabetical import ordering keeps it there.
from ._bootstrap import available_variants, supported_variants
from ._core import (
    __build_type__,
    __compiler_flags__,
    __nanobind_version__,
    __variant__,
    antihermitian_generator_correction,
    has_mpi,
    is_antihermitian,
)
from ._version import version as __version__
from .circuit import (
    Circuit,
    ExpGate,
    expand_monomials,
    validate_parameter_mapping,
)
from .fermi import FermiOperator, FermiString
from .integral_conversion import integrals_to_fermion
from .majorana import Majorana, MajoranaOperator
from .majorana_propagator import MajoranaPropagator
from .monomial_propagator import MonomialPropagator
from .pauli import Pauli, PauliOperator
from .pauli_propagator import PauliPropagator
from .utils import jordan_wigner_basis_change

__all__ = [
    "Circuit",
    "ExpGate",
    "FermiOperator",
    "FermiString",
    "Majorana",
    "MajoranaOperator",
    "MajoranaPropagator",
    "MonomialPropagator",
    "Pauli",
    "PauliOperator",
    "PauliPropagator",
    "__build_type__",
    "__compiler_flags__",
    "__nanobind_version__",
    "__variant__",
    "__version__",
    "antihermitian_generator_correction",
    "available_variants",
    "expand_monomials",
    "has_mpi",
    "integrals_to_fermion",
    "is_antihermitian",
    "jordan_wigner_basis_change",
    "supported_variants",
    "validate_parameter_mapping",
]

if importlib.util.find_spec("qiskit") is not None:
    from .qiskit_conversion import (
        from_qiskit_circuit,
        from_qiskit_operator,
        to_qiskit_circuit,
        to_qiskit_operator,
    )

    __all__ += [
        "from_qiskit_circuit",
        "from_qiskit_operator",
        "to_qiskit_circuit",
        "to_qiskit_operator",
    ]
