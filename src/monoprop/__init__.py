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

from ._core import (
    MAX_NUM_MODES,
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
    "MAX_NUM_MODES",
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
    "expand_monomials",
    "has_mpi",
    "integrals_to_fermion",
    "is_antihermitian",
    "jordan_wigner_basis_change",
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


import importlib.metadata


def _load_plugins() -> None:
    """Attach every installed monoprop plugin's public API onto this namespace.

    A plugin is any distribution registering a `monoprop.plugins` entry point whose
    value is an importable module exposing `__all__`; every name in it is copied here,
    the same way `from module import *` would. See `monoprop_pennylane` for the
    reference implementation.
    """
    for entry_point in importlib.metadata.entry_points(group="monoprop.plugins"):
        plugin = entry_point.load()
        names = getattr(plugin, "__all__", ())
        for name in names:
            globals()[name] = getattr(plugin, name)
        # extend() mutates the existing list in place, so `__all__` stays a read (not a
        # rebind) of the module-level global -- unlike `+=`, this needs no `global` statement.
        __all__.extend(names)  # noqa: PYI056


_load_plugins()
del _load_plugins
