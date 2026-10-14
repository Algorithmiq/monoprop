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

from importlib.metadata import entry_points

import monoprop_cirq


def test_cirq_entry_point_loads_package():
    """monoprop's plugin loader reads this entry point and attaches the names in ``__all__``."""
    (entry_point,) = entry_points(group="monoprop.plugins", name="cirq")
    assert entry_point.value == "monoprop_cirq"
    assert entry_point.load() is monoprop_cirq


def test_public_names_resolve():
    public = [name for name in monoprop_cirq.__all__ if not name.startswith("__")]
    assert public
    assert all(callable(getattr(monoprop_cirq, name)) for name in public)
