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

import importlib.metadata
import importlib.util
import warnings

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


def _load_plugins() -> None:
    """Attach every installed monoprop plugin's public API onto this namespace.

    A plugin is any distribution registering a `monoprop.plugins` entry point whose
    value is an importable module exposing `__all__`; every name in it is copied here,
    the same way `from module import *` would. See `monoprop_pennylane` for the
    reference implementation. Plugins load only after every name above this function is
    already bound, so a plugin can safely `import monoprop` at its own import time (as
    `monoprop_pennylane` does implicitly).

    Dunder names (e.g. a plugin's own `__version__`) are never copied, so a plugin can
    never shadow monoprop core's own identity metadata. A non-dunder name that would
    replace an existing, different object already bound here (core's own, or an
    earlier-loaded plugin's) is still copied -- last-write-wins, per this mechanism's
    load-and-splat design -- but raises a warning so the collision is never silent, and is
    never duplicated in `__all__`. A plugin that fails to load (e.g. a missing or broken
    runtime dependency) is skipped with a warning rather than breaking `import monoprop`
    for every consumer in the environment, not just the ones using that plugin.
    """
    for entry_point in importlib.metadata.entry_points(group="monoprop.plugins"):
        try:
            plugin = entry_point.load()
        except Exception as exc:  # noqa: BLE001 - one broken plugin must not break `import monoprop`
            warnings.warn(
                f"monoprop plugin {entry_point.name!r} failed to load: {exc}",
                stacklevel=2,
            )
            continue
        names = [
            name for name in getattr(plugin, "__all__", ()) if not name.startswith("__")
        ]
        for name in names:
            value = getattr(plugin, name)
            # Compare identity, not mere presence: a name already bound to this exact
            # object (e.g. re-loading the same plugin) is not a collision, only a
            # different object replacing it is.
            if globals().get(name, value) is not value:
                warnings.warn(
                    f"monoprop plugin {entry_point.name!r} overwrites existing name {name!r}",
                    stacklevel=2,
                )
            globals()[name] = value
            if name not in __all__:
                __all__.append(name)  # noqa: PYI056


_load_plugins()
del _load_plugins
