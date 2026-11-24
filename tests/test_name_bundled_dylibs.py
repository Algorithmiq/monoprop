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

"""The macOS wheel repair step that names bundled libraries ``@rpath/<file>``."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "cibw" / "name_bundled_dylibs.py"
)
_spec = importlib.util.spec_from_file_location("name_bundled_dylibs", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
name_bundled_dylibs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(name_bundled_dylibs)


def test_only_delocate_bundled_libraries_are_selected() -> None:
    assert name_bundled_dylibs.is_bundled_dylib("monoprop/.dylibs/libomp.dylib")
    assert not name_bundled_dylibs.is_bundled_dylib("monoprop/lib/libmonoprop.dylib")
    assert not name_bundled_dylibs.is_bundled_dylib("monoprop/.dylibs/README")
    assert not name_bundled_dylibs.is_bundled_dylib("monoprop/_core.abi3.so")


def test_record_row_matches_the_wheel_format() -> None:
    data = b"libomp"
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    assert name_bundled_dylibs.record_row("a/b.dylib", data) == [
        "a/b.dylib",
        f"sha256={digest.decode()}",
        "6",
    ]


def test_rewrite_record_only_touches_the_changed_rows() -> None:
    record = (
        "monoprop/__init__.py,sha256=abc,10\n"
        "monoprop/.dylibs/libomp.dylib,sha256=old,3\n"
        "monoprop-1.dist-info/RECORD,,\n"
    )
    out = name_bundled_dylibs.rewrite_record(
        record, {"monoprop/.dylibs/libomp.dylib": b"new!"}
    )
    lines = out.splitlines()
    assert lines[0] == "monoprop/__init__.py,sha256=abc,10"
    assert lines[1] == ",".join(
        name_bundled_dylibs.record_row("monoprop/.dylibs/libomp.dylib", b"new!")
    )
    assert lines[2] == "monoprop-1.dist-info/RECORD,,"
