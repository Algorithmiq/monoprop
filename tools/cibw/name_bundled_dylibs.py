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

# ruff: noqa: INP001, T201

"""Give every library delocate bundled into a macOS wheel the install name ``@rpath/<file>``.

delocate names each library it copies into ``<package>/.dylibs`` with a placeholder (``/DLC/...``). The wheel's own
binaries load those copies through ``@loader_path`` and never read that name, but a C++ consumer of the wheel's CMake
package links the bundled libomp itself (see ``cmake/monopropConfig.cmake.in``) and records the name it linked: with
``@rpath/<file>`` it loads the very file libmonoprop loads, where the placeholder would not load at all. LLVM's libomp
aborts when a second copy initializes in one process.

The renamed libraries are ad-hoc re-signed (arm64 refuses unsigned code) and their ``RECORD`` rows rewritten.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def is_bundled_dylib(name: str) -> bool:
    """Whether the wheel member ``name`` is a library delocate bundled."""
    path = PurePosixPath(name)
    return ".dylibs" in path.parts[:-1] and path.suffix == ".dylib"


def record_row(name: str, data: bytes) -> list[str]:
    """The ``RECORD`` row of the wheel member ``name`` holding ``data``."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return [name, f"sha256={digest.decode()}", str(len(data))]


def rewrite_record(record: str, contents: dict[str, bytes]) -> str:
    """Return ``record`` with the rows of the members in ``contents`` recomputed."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for row in csv.reader(io.StringIO(record)):
        if row and row[0] in contents:
            writer.writerow(record_row(row[0], contents[row[0]]))
        else:
            writer.writerow(row)
    return out.getvalue()


def _rename(path: Path) -> None:
    subprocess.run(  # noqa: S603
        ["install_name_tool", "-id", f"@rpath/{path.name}", str(path)],  # noqa: S607
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["codesign", "--force", "--sign", "-", str(path)],  # noqa: S607
        check=True,
    )


def main() -> int:
    """Rename the bundled libraries of the wheel named on the command line, in place."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()

    with zipfile.ZipFile(args.wheel) as wheel:
        infos = wheel.infolist()
        members = {info.filename: wheel.read(info) for info in infos}
    bundled = [name for name in members if is_bundled_dylib(name)]
    if not bundled:
        print(f"{args.wheel.name}: no bundled libraries")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        changed: dict[str, bytes] = {}
        for name in bundled:
            path = Path(tmp, PurePosixPath(name).name)
            path.write_bytes(members[name])
            _rename(path)
            changed[name] = path.read_bytes()
    records = [name for name in members if name.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        print(
            f"{args.wheel.name}: expected one RECORD, found {records}", file=sys.stderr
        )
        return 1
    members.update(changed)
    members[records[0]] = rewrite_record(members[records[0]].decode(), changed).encode()

    staged = args.wheel.with_suffix(".tmp")
    with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as out:
        for info in infos:
            out.writestr(info, members[info.filename])
    staged.replace(args.wheel)
    print(f"{args.wheel.name}: named {', '.join(bundled)} @rpath/<file>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
