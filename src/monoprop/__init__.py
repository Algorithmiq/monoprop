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

from ._core import (
    MAX_NUM_MODES,
    __build_type__,
    __compiler_flags__,
    antihermitian_generator_correction,
    has_mpi,
    is_antihermitian,
)
from ._version import version as __version__
from .monomial_propagator import MonomialPropagator
from .utils import jordan_wigner_basis_change

__all__ = [
    "MAX_NUM_MODES",
    "MonomialPropagator",
    "__build_type__",
    "__compiler_flags__",
    "__version__",
    "antihermitian_generator_correction",
    "has_mpi",
    "is_antihermitian",
    "jordan_wigner_basis_change",
]
