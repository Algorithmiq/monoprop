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

"""Tests for the ``clang -###`` output cleaner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[1] / "tools" / "clang-target-help-clean.py"
_spec = importlib.util.spec_from_file_location("clang_target_help_clean", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
clean_target_help = _module.clean_target_help


def test_target_cpu_and_feature_are_emitted() -> None:
    text = (
        "Apple clang version 16.0.0\n"
        "Target: arm64-apple-darwin\n"
        ' "/usr/bin/clang++" "-cc1" "-triple" "arm64-apple-macosx14.0.0" '
        '"-target-cpu" "apple-m4" "-target-feature" "+neon"\n'
    )
    assert clean_target_help(text) == "-target-cpu=apple-m4 -target-feature=+neon"


def test_only_selected_m_flags_are_kept() -> None:
    text = (
        ' "/usr/bin/clang++" "-cc1" "-mframe-pointer=non-leaf" '
        '"-march=armv8.6-a" "-mtune=apple-m4" "-mllvm" "-something"\n'
    )
    assert clean_target_help(text) == "-march=armv8.6-a -mtune=apple-m4"


def test_spaced_flag_forms_are_normalized() -> None:
    text = "-mcpu apple-m3 -mtune generic -march native -target-feature +crc"
    assert (
        clean_target_help(text)
        == "-mcpu=apple-m3 -mtune=generic -march=native -target-feature=+crc"
    )


def test_duplicates_are_removed_preserving_order() -> None:
    text = (
        "-march=native -target-cpu apple-m3 -target-feature +neon "
        "-target-feature +neon -march=native"
    )
    assert (
        clean_target_help(text)
        == "-march=native -target-cpu=apple-m3 -target-feature=+neon"
    )


def test_realistic_appleclang_output_shape() -> None:
    text = (
        "Apple clang version 21.0.0 (clang-2100.0.123.102)\n"
        "Target: arm64-apple-darwin25.3.0\n"
        '"/Library/Developer/CommandLineTools/usr/bin/clang" "-cc1" '
        '"-target-cpu" "apple-m1" '
        '"-target-feature" "+v8.5a" '
        '"-target-feature" "+dotprod" '
        '"-target-feature" "+neon"\n'
    )
    assert (
        clean_target_help(text) == "-target-cpu=apple-m1 -target-feature=+v8.5a "
        "-target-feature=+dotprod -target-feature=+neon"
    )


def test_empty_input_returns_empty() -> None:
    assert clean_target_help("") == ""
