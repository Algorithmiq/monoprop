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


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, fake_entry_point: types.SimpleNamespace
) -> None:
    def fake_entry_points(*, group: str | None = None) -> list[types.SimpleNamespace]:
        return [fake_entry_point] if group == "monoprop.plugins" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


def test_load_plugins_attaches_names_from_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `monoprop.plugins` entry point's `__all__` names land on the `monoprop` namespace."""
    fake_plugin = types.SimpleNamespace(greet=lambda: "hello", __all__=["greet"])
    _patch_entry_points(monkeypatch, types.SimpleNamespace(load=lambda: fake_plugin))
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
        # `reload` re-executes the module body but never clears an attribute the reloaded
        # code doesn't re-set, so the fake plugin's `greet` would otherwise linger on the
        # real `monoprop` module for the rest of the test session.
        if hasattr(monoprop, "greet"):
            delattr(monoprop, "greet")


def test_load_plugins_never_shadows_dunder_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin's own `__version__` (or any dunder) must never overwrite monoprop core's.

    `monoprop_pennylane.__all__` legitimately includes `"__version__"` (so
    `monoprop_pennylane.__version__` keeps working standalone), so the loader itself, not
    the plugin, is responsible for never letting a dunder cross the namespace boundary.
    """
    original_version = monoprop.__version__
    sentinel_version = object()
    fake_plugin = types.SimpleNamespace(
        __version__=sentinel_version, __all__=["__version__"]
    )
    _patch_entry_points(monkeypatch, types.SimpleNamespace(load=lambda: fake_plugin))
    try:
        importlib.reload(monoprop)
        assert monoprop.__version__ is not sentinel_version
        assert monoprop.__version__ == original_version
        # Also not duplicated in __all__ (core already lists "__version__" once).
        assert monoprop.__all__.count("__version__") == 1
    finally:
        monkeypatch.undo()
        importlib.reload(monoprop)
