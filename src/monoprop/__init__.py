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

"""Copyright (c) 2025 Algorithmiq. All rights reserved.

monoprop: A great package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._core import (
    MAX_NUM_MODES,
    __build_type__,
    __compiler_flags__,
    antihermitian_generator_correction,
    has_mpi,
    is_antihermitian,
)
from ._version import version as __version__
from .circuit import (
    Circuit,
    Exp,
    expand_monomials,
    validate_parameter_mapping,
)
from .exceptions import NumberOfModesInvalidError
from .fermi_data import FermiOperator, FermiString
from .integral_conversion import integrals_to_fermion
from .majorana_data import Majorana, MajoranaOperator
from .majorana_propagator import MajoranaPropagator
from .pauli_data import Pauli, PauliOperator
from .pauli_propagator import PauliPropagator
from .quantum_data import IQuantumOperator
from .utils import jordan_wigner_basis_change

__all__ = [
    "MAX_NUM_MODES",
    "Circuit",
    "Exp",
    "FermiOperator",
    "FermiString",
    "IQuantumOperator",
    "Majorana",
    "MajoranaOperator",
    "MajoranaPropagator",
    "NumberOfModesInvalidError",
    "Pauli",
    "PauliOperator",
    "PauliPropagator",
    "__build_type__",
    "__compiler_flags__",
    "__version__",
    "antihermitian_generator_correction",
    "expand_monomials",
    "has_mpi",
    "integrals_to_fermion",
    "is_antihermitian",
    "jordan_wigner_basis_change",
    "validate_parameter_mapping",
]

# The qiskit conversion helpers live behind an optional dependency: importing
# ``monoprop.qiskit_conversion`` raises if qiskit is absent. Expose them lazily so that
# ``import monoprop`` never requires qiskit, while ``from monoprop import to_qiskit_circuit``
# still works (raising the helpful ImportError only when actually accessed). They are kept out
# of ``__all__`` so ``from monoprop import *`` stays qiskit-free for installs without the extra.
_QISKIT_EXPORTS = frozenset(
    {
        "from_qiskit_circuit",
        "from_qiskit_operator",
        "to_qiskit_circuit",
        "to_qiskit_operator",
    }
)

if TYPE_CHECKING:
    # Re-exported for type checkers / IDEs; resolved at runtime by ``__getattr__`` below.
    from .qiskit_conversion import (
        from_qiskit_circuit as from_qiskit_circuit,
    )
    from .qiskit_conversion import (
        from_qiskit_operator as from_qiskit_operator,
    )
    from .qiskit_conversion import (
        to_qiskit_circuit as to_qiskit_circuit,
    )
    from .qiskit_conversion import (
        to_qiskit_operator as to_qiskit_operator,
    )


def __getattr__(name: str) -> object:
    """Lazily resolve the optional qiskit conversion helpers (PEP 562)."""
    if name in _QISKIT_EXPORTS:
        from . import qiskit_conversion  # noqa: PLC0415  (lazy: optional dependency)

        return getattr(qiskit_conversion, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exposed qiskit helpers in ``dir(monoprop)``."""
    return sorted({*__all__, *_QISKIT_EXPORTS})
