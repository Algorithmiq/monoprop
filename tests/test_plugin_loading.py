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
`importlib.metadata.entry_points` to return fake entry points pointing at throwaway
modules.
"""

from __future__ import annotations

import importlib.metadata
import types

import pytest

import monoprop


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, *fake_entry_points: types.SimpleNamespace
) -> None:
    def fake_entry_points_fn(
        *, group: str | None = None
    ) -> list[types.SimpleNamespace]:
        return list(fake_entry_points) if group == "monoprop.plugins" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points_fn)


def _delete_injected(*names: str) -> None:
    """Remove attributes a fake plugin left on the real `monoprop` module.

    `importlib.reload` re-executes the module body in place but never clears an
    attribute the reloaded code doesn't re-set, so anything a fake plugin injected
    would otherwise linger on the real `monoprop` module for the rest of the test
    session.
    """
    for name in names:
        if hasattr(monoprop, name):
            delattr(monoprop, name)


def test_load_plugins_attaches_names_from_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `monoprop.plugins` entry point's `__all__` names land on the `monoprop` namespace."""
    fake_plugin = types.SimpleNamespace(greet=lambda: "hello", __all__=["greet"])
    _patch_entry_points(
        monkeypatch, types.SimpleNamespace(name="fake", load=lambda: fake_plugin)
    )
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
        _delete_injected("greet")


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
    _patch_entry_points(
        monkeypatch, types.SimpleNamespace(name="fake", load=lambda: fake_plugin)
    )
    try:
        importlib.reload(monoprop)
        assert monoprop.__version__ is not sentinel_version
        assert monoprop.__version__ == original_version
        # Also not duplicated in __all__ (core already lists "__version__" once).
        assert monoprop.__all__.count("__version__") == 1
    finally:
        monkeypatch.undo()
        importlib.reload(monoprop)


def test_load_plugins_skips_failing_plugin_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin whose entry point fails to load must not break `import monoprop` for everyone.

    Reachable in practice: `monoprop_pennylane.conversion` raises `ImportError` at import
    time whenever pennylane is missing or broken (uninstalled, version mismatch, partial
    install) -- that must not take down every other consumer of `monoprop` in the same
    environment, including ones using other, working plugins.
    """
    original_circuit = monoprop.Circuit

    def _raise_on_load() -> types.SimpleNamespace:
        raise ImportError("pennylane is required to use this plugin")

    working_plugin = types.SimpleNamespace(
        still_works=object(), __all__=["still_works"]
    )
    _patch_entry_points(
        monkeypatch,
        types.SimpleNamespace(name="broken", load=_raise_on_load),
        types.SimpleNamespace(name="working", load=lambda: working_plugin),
    )
    try:
        with pytest.warns(UserWarning, match="broken.*failed to load"):
            importlib.reload(monoprop)
        # The broken plugin didn't crash the loader...
        assert monoprop.Circuit is original_circuit
        # ...and the working plugin listed after it still loaded normally.
        assert monoprop.still_works is working_plugin.still_works
        assert "still_works" in monoprop.__all__
    finally:
        monkeypatch.undo()
        importlib.reload(monoprop)
        _delete_injected("still_works")


def test_load_plugins_with_missing_all_loads_without_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin module exposing no `__all__` at all is a silent no-op, not an error."""
    fake_plugin = types.SimpleNamespace()  # deliberately no `__all__`
    _patch_entry_points(
        monkeypatch, types.SimpleNamespace(name="fake", load=lambda: fake_plugin)
    )
    try:
        importlib.reload(monoprop)  # must not raise
    finally:
        monkeypatch.undo()
        importlib.reload(monoprop)


def test_load_plugins_warns_and_overwrites_on_name_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name collision (with core or an earlier-loaded plugin) still overwrites, but warns.

    Keeps the load-and-splat/no-registry design -- the overwrite is not refused -- but
    makes an otherwise-silent collision noisy, and never leaves a duplicate entry in
    `monoprop.__all__`.
    """
    first_value = object()
    second_value = object()
    first_plugin = types.SimpleNamespace(
        plugin_shared_value=first_value, __all__=["plugin_shared_value"]
    )
    second_plugin = types.SimpleNamespace(
        plugin_shared_value=second_value, __all__=["plugin_shared_value"]
    )
    _patch_entry_points(
        monkeypatch,
        types.SimpleNamespace(name="first", load=lambda: first_plugin),
        types.SimpleNamespace(name="second", load=lambda: second_plugin),
    )
    try:
        with pytest.warns(
            UserWarning, match="second.*overwrites existing name 'plugin_shared_value'"
        ):
            importlib.reload(monoprop)
        # Last-write-wins: the second plugin's value is the one that stuck.
        assert monoprop.plugin_shared_value is second_value
        assert monoprop.__all__.count("plugin_shared_value") == 1
    finally:
        monkeypatch.undo()
        importlib.reload(monoprop)
        _delete_injected("plugin_shared_value")
