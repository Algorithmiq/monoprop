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

"""Normalize compiler target-help output into stable machine-flag strings.

The script reads compiler output from stdin and cleans it according to the
selected mode:

- ``gcc`` for ``gcc -Q --help=target`` output
- ``clang`` for ``clang -###`` output
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_PAIR_FLAGS = (
    "-march",
    "-mcpu",
    "-mtune",
    "-target-cpu",
    "-target-feature",
)
_ALLOWED_PREFIXES = tuple(f"{flag}=" for flag in _PAIR_FLAGS)

_START_MARKER = "The following options are target specific:"
_MULTISPACE = re.compile(r"\s{2,}")
_HINT = re.compile(r"<[^>]*>")


def _gcc_option_lines(text: str) -> Iterator[str]:
    """Yield option lines from GCC's target-specific section."""
    _, marker, section = text.partition(_START_MARKER)
    if not marker:
        return

    options_started = False
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if line.startswith("-"):
            options_started = True
            yield line
        elif options_started:
            return


def _normalize_gcc_option(line: str) -> str | None:
    """Normalize one GCC target option, omitting inactive values."""
    if "[disabled]" in line:
        return None

    fields = _MULTISPACE.split(line, maxsplit=1)
    name = _HINT.sub("", fields[0])
    value = fields[1].strip() if len(fields) > 1 else ""

    if "[enabled]" in value:
        return name
    if "=" not in name:
        return f"{name} {value}".strip()
    if not value or value == "[default]":
        return None
    return name + value


def _clang_machine_flags(text: str) -> Iterator[str]:
    """Yield normalized machine flags from Clang's command-line trace."""
    tokens = iter(shlex.split(text.replace("\n", " ")))
    for token in tokens:
        if token.startswith(_ALLOWED_PREFIXES):
            yield token
        elif token in _PAIR_FLAGS and (value := next(tokens, None)) is not None:
            yield f"{token}={value}"


def clean_clang_target_help(text: str) -> str:
    """Normalize ``clang -###`` output into a machine-flag string."""
    return " ".join(dict.fromkeys(_clang_machine_flags(text)))


def clean_gcc_target_help(text: str) -> str:
    """Normalize ``gcc -Q --help=target`` output into a machine-flag string.

    Parsing starts after the target-specific marker and stops at the first
    blank separator line. This keeps section detection architecture-agnostic.
    """
    entries = (
        normalized
        for line in _gcc_option_lines(text)
        if (normalized := _normalize_gcc_option(line)) is not None
    )
    return " ".join(entries)


def clean_target_help(text: str, mode: str) -> str:
    """Dispatch target-help normalization according to compiler mode."""
    if mode == "gcc":
        return clean_gcc_target_help(text)
    if mode == "clang":
        return clean_clang_target_help(text)

    raise ValueError(f"Unsupported mode: {mode}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("gcc", "clang"),
        help="Parser mode matching the compiler output format.",
    )
    return parser


def main() -> None:
    """Read stdin and print normalized target flags for the selected mode."""
    args = _build_parser().parse_args()
    sys.stdout.write(clean_target_help(sys.stdin.read(), mode=args.mode) + "\n")


if __name__ == "__main__":
    main()
