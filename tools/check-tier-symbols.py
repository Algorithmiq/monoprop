#!/usr/bin/env python3
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

"""Whether a narrow-seam fat binary actually kept its ISA tiers, or lost them to the linker.

The failure this exists for. A tiered translation unit reaches the compiler once per ISA tier, so its
objects genuinely differ -- one has ``vpopcntq`` in it and another does not. But almost everything in
those objects is a *template instantiation*, emitted as a weak COMDAT symbol whose mangled name says
nothing about the tier. The linker keeps one definition per name and discards the rest, so four tiers
collapse to whichever one the link line happened to list first, and the run-time dispatch then selects
between four entry points that all call the same code.

It is invisible from the outside: the module loads, every tier reports its own identity, every tier
returns bit-identical numbers, and the whole suite passes. Only a benchmark notices, and only if you
already suspect it -- which is why this is a build-time check and not a test.

Run it against a build tree::

    tools/check-tier-symbols.py build/editable/Release

Exit status is 0 when every tier's tiered objects define a disjoint set of weak symbols, 1 otherwise.
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

# One directory per tier, named by CMake from the tier id: monoprop-tier-<slug>.dir
_TIER_DIR = re.compile(r"^monoprop-tier-(?P<slug>.+)\.dir$")

# Instructions that mark a tier above the baseline. Counted, not just detected, because the useful
# question is how much widened code is in an object, not whether any is.
_WIDE = {
    "zmm": re.compile(r"%zmm"),
    "ymm": re.compile(r"%ymm"),
    "vpopcnt": re.compile(r"\bvpopcnt[bwdq]\b"),
    "popcnt": re.compile(r"\bpopcnt\b"),
    "kmask": re.compile(r"\bk(?:mov[bwdq]|and[bwdq]|or[bwdq])\b"),
}


def _run(argv: list[str]) -> str:
    completed = subprocess.run(  # noqa: S603 - argv is built here, never from input
        argv, capture_output=True, text=True, check=True
    )
    return completed.stdout


def weak_symbols(obj: Path) -> set[str]:
    """Names of the weak definitions in one object -- i.e. everything the linker may deduplicate."""
    out = _run(["nm", "--defined-only", str(obj)])
    names = set()
    for line in out.splitlines():
        fields = line.split()
        # "<addr> <type> <name>"; W/w is a weak definition, which for C++ means a COMDAT
        if len(fields) == 3 and fields[1] in {"W", "w", "V", "v"}:
            names.add(fields[2])
    return names


def wide_census(target: Path) -> dict[str, int]:
    """How many instructions in one object or module belong to a tier above the baseline.

    Counted per disassembly line, not per match, so a three-operand AVX instruction counts once.
    """
    text = _run(["objdump", "-d", "--no-show-raw-insn", str(target)])
    lines = text.splitlines()
    return {
        name: sum(1 for line in lines if pattern.search(line))
        for name, pattern in _WIDE.items()
    }


def tier_objects(build_dir: Path) -> dict[str, list[Path]]:
    """The tiered objects, per tier id, found by the directory names CMake generates."""
    found: dict[str, list[Path]] = {}
    for entry in sorted((build_dir / "CMakeFiles").glob("monoprop-tier-*.dir")):
        match = _TIER_DIR.match(entry.name)
        if match is None:
            continue
        objects = sorted(entry.rglob("*.o"))
        if objects:
            found[match.group("slug")] = objects
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="a configured build tree, e.g. build/editable/Release")
    parser.add_argument(
        "--module",
        type=Path,
        default=None,
        help="also census the linked module (default: the _core extension under this build tree)",
    )
    args = parser.parse_args()

    tiers = tier_objects(args.build_dir)
    if not tiers:
        print(
            f"no monoprop-tier-*.dir under {args.build_dir}/CMakeFiles: this is not a narrow-seam "
            "build, and there is nothing for the linker to collapse",
            file=sys.stderr,
        )
        return 0

    print(f"{'tier':24s} {'objects':>8s} {'weak syms':>10s} " + " ".join(f"{k:>9s}" for k in _WIDE))
    per_tier: dict[str, set[str]] = {}
    for slug, objects in tiers.items():
        weak: set[str] = set()
        counts: collections.Counter[str] = collections.Counter()
        for obj in objects:
            weak |= weak_symbols(obj)
            counts.update(wide_census(obj))
        per_tier[slug] = weak
        print(
            f"{slug:24s} {len(objects):8d} {len(weak):10d} "
            + " ".join(f"{counts[k]:9d}" for k in _WIDE)
        )

    module = args.module
    if module is None:
        candidates = sorted(args.build_dir.rglob("_core*.so"))
        module = candidates[0] if candidates else None
    if module is not None:
        counts = wide_census(module)
        print(
            f"\n{'linked ' + module.name:24s} {'':8s} {'':10s} "
            + " ".join(f"{counts[k]:9d}" for k in _WIDE)
        )

    # The check. Every name defined weakly in two tiers is a name the linker will resolve once, so the
    # losing tier's copy of it -- and every widened instruction inside it -- is discarded.
    owners: dict[str, list[str]] = collections.defaultdict(list)
    for slug, weak in per_tier.items():
        for name in weak:
            owners[name].append(slug)
    shared = {name: slugs for name, slugs in owners.items() if len(slugs) > 1}

    if not shared:
        print("\nOK: every tier's weak symbols are its own; the linker keeps all of them.")
        return 0

    print(
        f"\nFAIL: {len(shared)} weak symbol(s) are defined by more than one tier. The linker keeps one "
        "definition per name, so those tiers run the same code -- whichever the link line lists first "
        "-- whatever the run-time dispatch selected.",
        file=sys.stderr,
    )
    for name in sorted(shared)[:5]:
        demangled = _run(["c++filt", name]).strip()
        print(f"  {', '.join(sorted(shared[name]))}: {demangled[:150]}", file=sys.stderr)
    if len(shared) > 5:
        print(f"  ... and {len(shared) - 5} more", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
