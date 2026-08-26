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

"""The engine to colour mapping shared by every figure in this directory.

One mapping for the whole suite, so an engine keeps its colour as a reader moves between the
headline, per-step, scaling and speed-up figures. The hues are the Okabe-Ito qualitative
palette, which stays separable under the common forms of colour blindness.
"""

from __future__ import annotations

ENGINE_COLORS = {
    "monoprop": "#0072B2",
    "QuEra ppvm": "#D55E00",
    "Qiskit pauli-prop": "#009E73",
    "cuPauliProp (GPU)": "#CC79A7",
    "PauliPropagation.jl": "#E69F00",
    # Each family's Julia reference engine; the two never appear on the same axes.
    "MajoranaPropagation.jl": "#E69F00",
}

# Legend and bar order for the Pauli figures. Also the membership test that keeps a Majorana
# engine out of a Pauli figure's legend, so it cannot be widened by adding a key above.
PAULI_ENGINES = [
    "monoprop",
    "QuEra ppvm",
    "Qiskit pauli-prop",
    "cuPauliProp (GPU)",
    "PauliPropagation.jl",
]
