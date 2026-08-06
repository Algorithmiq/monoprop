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

"""Tests for the ``gcc -Q --help=target`` output cleaner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "tools" / "gcc-target-help-clean.py"
_spec = importlib.util.spec_from_file_location("gcc_target_help_clean", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
clean_target_help = _module.clean_target_help


def _wrap(body: str) -> str:
    return (
        "The following options are target specific:\n"
        f"{body}\n"
        "\n"
        "  Known assembler dialects (for use with the -masm= option):\n"
        "    att intel\n"
    )


def test_enabled_keeps_only_name() -> None:
    text = _wrap("  -m64                        \t\t[enabled]")
    assert clean_target_help(text) == "-m64"


def test_disabled_is_dropped() -> None:
    text = _wrap("  -m16                        \t\t[disabled]")
    assert clean_target_help(text) == ""


def test_equals_joins_with_value() -> None:
    text = _wrap("  -mabi=                      \t\tsysv")
    assert clean_target_help(text) == "-mabi=sysv"


def test_equals_empty_value_is_dropped() -> None:
    text = _wrap("  -mcpu=                      \t\t")
    assert clean_target_help(text) == ""


def test_equals_default_value_is_dropped() -> None:
    text = _wrap("  -mcmodel=                   \t\t[default]")
    assert clean_target_help(text) == ""


def test_alias_line_keeps_both_fields() -> None:
    text = _wrap("  -msse5                      \t\t-mavx")
    assert clean_target_help(text) == "-msse5 -mavx"


def test_range_hint_is_stripped() -> None:
    text = _wrap("  -mbranch-cost=<0,5>         \t\t3")
    assert clean_target_help(text) == "-mbranch-cost=3"


def test_only_section_between_markers_is_used() -> None:
    text = (
        "-mignored-before                \t\t[enabled]\n"
        "The following options are target specific:\n"
        "  -m64                        \t\t[enabled]\n"
        "  Known assembler dialects (for use with the -masm= option):\n"
        "  -mignored-after                 \t\t[enabled]\n"
    )
    assert clean_target_help(text) == "-m64"


def test_missing_start_marker_returns_empty() -> None:
    assert clean_target_help("nothing relevant here") == ""


def test_multiple_entries_joined_by_space() -> None:
    text = _wrap(
        "  -m64                        \t\t[enabled]\n"
        "  -m16                        \t\t[disabled]\n"
        "  -mabi=                      \t\tsysv\n"
        "  -msse5                      \t\t-mavx"
    )
    assert clean_target_help(text) == "-m64 -mabi=sysv -msse5 -mavx"
