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

"""Tests for the generic `monoprop.plugins` entry-point loading mechanism.

These exercise the mechanism in isolation, independent of whether any real plugin
(e.g. `monoprop_pennylane`) happens to be installed, by monkeypatching
`importlib.metadata.entry_points` to return a fake entry point pointing at a
throwaway module.
"""

from __future__ import annotations

import importlib.metadata
import types
from typing import TYPE_CHECKING

import monoprop

if TYPE_CHECKING:
    import pytest


def test_load_plugins_attaches_names_from_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `monoprop.plugins` entry point's `__all__` names land on the `monoprop` namespace."""
    fake_plugin = types.SimpleNamespace(greet=lambda: "hello", __all__=["greet"])
    fake_entry_point = types.SimpleNamespace(load=lambda: fake_plugin)

    def fake_entry_points(*, group: str | None = None) -> list[types.SimpleNamespace]:
        return [fake_entry_point] if group == "monoprop.plugins" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        importlib.reload(monoprop)
        assert monoprop.greet is fake_plugin.greet
        assert monoprop.greet() == "hello"
        assert "greet" in monoprop.__all__
    finally:
        # Undo the patch before reloading again, so monoprop ends up back in the
        # state a normal import would leave it in (real plugins, if any installed).
        monkeypatch.undo()
        importlib.reload(monoprop)
