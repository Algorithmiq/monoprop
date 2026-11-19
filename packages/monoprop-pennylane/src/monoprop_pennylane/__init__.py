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

"""Convert PennyLane circuits and operators to and from monoprop's native representations."""

from __future__ import annotations

from monoprop_pennylane._version import __version__
from monoprop_pennylane.conversion import (
    from_pennylane_circuit,
    from_pennylane_operator,
    to_pennylane_circuit,
    to_pennylane_operator,
)

__all__ = [
    "__version__",
    "from_pennylane_circuit",
    "from_pennylane_operator",
    "to_pennylane_circuit",
    "to_pennylane_operator",
]
