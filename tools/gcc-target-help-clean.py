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

# ruff: noqa: INP001

"""Clean up the output of ``gcc -Q --help=target``.

Reads the command's output from stdin, extracts the target-specific options
section, normalizes each line, and prints a single space-separated string.
"""

from __future__ import annotations

import re
import sys

_START_MARKER = "The following options are target specific:"
_END_MARKER = "Known assembler dialects (for use with the -masm= option):"
_MULTISPACE = re.compile(r"\s{2,}")
_HINT = re.compile(r"<[^>]*>")


def clean_target_help(text: str) -> str:
    """Normalize ``gcc -Q --help=target`` output into a single string.

    Args:
        text: The full stdout of ``gcc -Q --help=target``.

    Returns:
        A single space-separated string of the cleaned options. Rules:
        lines containing ``[disabled]`` are dropped; lines whose value is
        ``[enabled]`` keep only the option name; options ending in ``=`` are
        joined to their value unless the value is empty or ``[default]`` (in
        which case the line is dropped); any remaining line keeps both fields
        joined by a single space.
    """
    start = text.find(_START_MARKER)
    if start == -1:
        return ""
    end = text.find(_END_MARKER, start)
    section = text[start + len(_START_MARKER) : end if end != -1 else None]

    entries: list[str] = []
    for raw in section.splitlines():
        line = raw.strip()
        if not line or "[disabled]" in line:
            continue
        fields = _MULTISPACE.split(line, maxsplit=1)
        name = _HINT.sub("", fields[0])
        value = fields[1].strip() if len(fields) > 1 else ""

        if "[enabled]" in value:
            entries.append(name)
        elif "=" in name:
            if value and value != "[default]":
                entries.append(name + value)
        else:
            entries.append(f"{name} {value}".strip())

    return " ".join(entries)


def main() -> None:
    """Read stdin and print the cleaned target options."""
    sys.stdout.write(clean_target_help(sys.stdin.read()) + "\n")


if __name__ == "__main__":
    main()
