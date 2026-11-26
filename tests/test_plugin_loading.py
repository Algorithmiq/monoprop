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

"""Tests for the `monoprop.plugins` entry-point loader, using fake entry points."""

from __future__ import annotations

import importlib.metadata
import types
from typing import TYPE_CHECKING

import pytest

import monoprop

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture
def load_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., None]]:
    """Reload `monoprop` against fake entry points, then restore the real module."""
    baseline = set(vars(monoprop))

    def load(**loaders: Callable[[], object]) -> None:
        entry_points = [
            types.SimpleNamespace(name=name, load=loader)
            for name, loader in loaders.items()
        ]
        monkeypatch.setattr(
            importlib.metadata,
            "entry_points",
            lambda *, group=None: entry_points if group == "monoprop.plugins" else [],
        )
        importlib.reload(monoprop)

    yield load
    monkeypatch.undo()
    importlib.reload(monoprop)
    # reload never clears attributes the module body no longer sets.
    for name in set(vars(monoprop)) - baseline:
        delattr(monoprop, name)


def test_load_plugins_attaches_names_from_entry_point(
    load_plugins: Callable[..., None],
) -> None:
    """A `monoprop.plugins` entry point's `__all__` names land on the `monoprop` namespace."""
    fake_plugin = types.SimpleNamespace(greet=lambda: "hello", __all__=["greet"])
    load_plugins(fake=lambda: fake_plugin)
    assert monoprop.greet is fake_plugin.greet
    assert monoprop.greet() == "hello"
    assert "greet" in monoprop.__all__


def test_load_plugins_never_shadows_dunder_names(
    load_plugins: Callable[..., None],
) -> None:
    """A plugin's own `__version__` (or any dunder) must never overwrite monoprop core's."""
    original_version = monoprop.__version__
    sentinel_version = object()
    fake_plugin = types.SimpleNamespace(
        __version__=sentinel_version, __all__=["__version__"]
    )
    load_plugins(fake=lambda: fake_plugin)
    assert monoprop.__version__ == original_version
    assert monoprop.__all__.count("__version__") == 1


def test_load_plugins_skips_failing_plugin_and_continues(
    load_plugins: Callable[..., None],
) -> None:
    """A plugin that fails to load warns, and later plugins still load."""
    original_circuit = monoprop.Circuit

    def _raise_on_load() -> types.SimpleNamespace:
        raise ImportError("pennylane is required to use this plugin")

    working_plugin = types.SimpleNamespace(
        still_works=object(), __all__=["still_works"]
    )
    with pytest.warns(UserWarning, match="broken.*failed to load"):
        load_plugins(broken=_raise_on_load, working=lambda: working_plugin)
    assert monoprop.Circuit is original_circuit
    assert monoprop.still_works is working_plugin.still_works
    assert "still_works" in monoprop.__all__


def test_load_plugins_with_missing_all_loads_without_error(
    load_plugins: Callable[..., None],
) -> None:
    """A plugin module exposing no `__all__` at all is a silent no-op, not an error."""
    load_plugins(fake=types.SimpleNamespace)


def test_load_plugins_warns_and_overwrites_on_name_collision(
    load_plugins: Callable[..., None],
) -> None:
    """A name collision still overwrites (last write wins), but warns."""
    first_plugin = types.SimpleNamespace(
        plugin_shared_value=object(), __all__=["plugin_shared_value"]
    )
    second_plugin = types.SimpleNamespace(
        plugin_shared_value=object(), __all__=["plugin_shared_value"]
    )
    with pytest.warns(
        UserWarning, match="second.*overwrites existing name 'plugin_shared_value'"
    ):
        load_plugins(first=lambda: first_plugin, second=lambda: second_plugin)
    assert monoprop.plugin_shared_value is second_plugin.plugin_shared_value
    assert monoprop.__all__.count("plugin_shared_value") == 1
