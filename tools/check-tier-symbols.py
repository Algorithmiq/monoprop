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

"""Whether a seam-mode fat binary actually kept its ISA tiers, or lost them to the linker.

The failure this exists for. A tiered translation unit reaches the compiler once per ISA tier, so its
objects genuinely differ -- one has ``vpopcntq`` in it and another does not. But almost everything in
those objects is a *template instantiation*, emitted as a weak COMDAT symbol whose mangled name says
nothing about the tier. The linker keeps one definition per name and discards the rest, so if all four
tiers arrive in one link they collapse to whichever the link line listed first, and the run-time
dispatch then selects between four entry points that all call the same code.

It is invisible from the outside: the module loads, every tier reports its own identity, every tier
returns bit-identical numbers, and the whole suite passes. Only a benchmark notices, and only if you
already suspect it -- which is why this is a build-time check and not a test.

The two seam packagings therefore get different checks, chosen by the mode in the tree's CMakeCache:

``tier-dso``    each tier is its own shared object, so each is its own link and the deduplication
                happens inside a tier instead of across them. Checked structurally: no tier exports a
                monoprop symbol another one does (nothing to interpose at run time either), every
                monoprop symbol a tier imports is exported by the module that will load it (or it dies
                at load with a symbol lookup error), the baseline tier and the shared module hold no
                widened instruction at all, and the tiers above baseline do.

``narrow-seam`` all four tiers arrive in one link as object libraries. Checked by weak-symbol
                disjointness, which is the thing that fails -- this mode is a recorded negative result,
                not a configuration to fix.

Run it against a build tree::

    tools/check-tier-symbols.py build/editable/Release

Exit status is 0 when the tiers are intact, 1 otherwise.
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

# One directory per tier in narrow-seam mode, named by CMake from the tier id.
_TIER_DIR = re.compile(r"^monoprop-tier-(?P<slug>.+)\.dir$")

# One shared object per tier in tier-dso mode, named by CMake from the tier id itself.
_TIER_DSO = re.compile(r"^libmonoprop-tier-(?P<tier>.+)\.so$")

# An objdump -d instruction line: leading blanks, hex address, colon, tab, mnemonic, operands.
_INSN = re.compile(r"^[ \t]+[0-9a-f]+:\t(?P<mnemonic>\S+)[ \t]*(?P<operands>.*)$")

# What marks a tier above the baseline. Counted, not just detected, because the useful question is how
# much widened code is in a module, not whether any is. Matched against a parsed instruction rather
# than a raw line: a tier's own namespace slug is "x86_64_v4_vpopcntdq", so every mangled symbol name
# objdump prints in that tier would otherwise register as a vector popcount.
_WIDE: dict[str, tuple[str, re.Pattern[str]]] = {
    "zmm": ("operands", re.compile(r"%zmm")),
    "ymm": ("operands", re.compile(r"%ymm")),
    "vpopcnt": ("mnemonic", re.compile(r"^vpopcnt")),
    "popcnt": ("mnemonic", re.compile(r"^popcnt")),
    "kmask": ("operands", re.compile(r"%k[0-7]\b")),
}

# What must be absent from anything every machine runs, and present in a tier above the baseline. All
# five, popcnt included: POPCNT is a v2 instruction, and on this codebase -- std::popcount word loops
# throughout -- it is the single biggest step in the whole ladder, while v2 adds no vector width at all.
# A ladder of v1 and v2 alone would look collapsed to a vector-only check.
_BASELINE_FORBIDDEN = tuple(_WIDE)


def _run(argv: list[str], *, check: bool = True) -> str:
    completed = subprocess.run(  # noqa: S603 - argv is built here, never from input
        argv, capture_output=True, text=True, check=check
    )
    return completed.stdout


def _out(text: str = "") -> None:
    """One line to stdout. sys.stdout.write and not print, which is the convention under tools/."""
    sys.stdout.write(f"{text}\n")


def _err(text: str) -> None:
    """One line to stderr, for the findings that make this exit non-zero."""
    sys.stderr.write(f"{text}\n")


def wide_census(target: Path) -> collections.Counter[str]:
    """How many instructions in one object or module belong to a tier above the baseline."""
    counts: collections.Counter[str] = collections.Counter()
    for line in _run(["objdump", "-d", "--no-show-raw-insn", str(target)]).splitlines():
        insn = _INSN.match(line)
        if insn is None:
            continue
        for name, (field, pattern) in _WIDE.items():
            if pattern.search(insn.group(field)):
                counts[name] += 1
    return counts


def weak_symbols(obj: Path) -> set[str]:
    """Names of the weak definitions in one object -- i.e. everything the linker may deduplicate."""
    names = set()
    for line in _run(["nm", "--defined-only", str(obj)]).splitlines():
        fields = line.split()
        # "<addr> <type> <name>"; W/w is a weak definition, which for C++ means a COMDAT
        if len(fields) == 3 and fields[1] in {"W", "w", "V", "v"}:
            names.add(fields[2])
    return names


def dynamic_symbols(target: Path) -> tuple[dict[str, str], set[str]]:
    """``({exported name: type letter}, {imported name})`` from one module's dynamic symbol table."""
    exported: dict[str, str] = {}
    imported: set[str] = set()
    for line in _run(["nm", "-D", str(target)]).splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"U", "w"}:
            imported.add(fields[1])
        elif len(fields) == 3:
            exported[fields[2]] = fields[1]
    return exported, imported


def _demangle(name: str) -> str:
    return _run(["c++filt", name], check=False).strip() or name


def _census_row(label: str, counts: collections.Counter[str]) -> str:
    return f"{label:44s} " + " ".join(f"{counts[k]:>9d}" for k in _WIDE)


def _census_header() -> str:
    return f"{'module':44s} " + " ".join(f"{k:>9s}" for k in _WIDE)


def find_tier_dsos(build_dir: Path) -> dict[str, Path]:
    """One shared object per tier, keyed by tier id -- non-empty only in tier-dso mode."""
    found: dict[str, Path] = {}
    for path in sorted(build_dir.rglob("libmonoprop-tier-*.so")):
        match = _TIER_DSO.match(path.name)
        if match is not None:
            found.setdefault(match.group("tier"), path)
    return found


def find_tier_objects(build_dir: Path) -> dict[str, list[Path]]:
    """The tiered objects per tier slug, from the directory names CMake generates in narrow-seam mode."""
    found: dict[str, list[Path]] = {}
    for entry in sorted((build_dir / "CMakeFiles").glob("monoprop-tier-*.dir")):
        match = _TIER_DIR.match(entry.name)
        if match is None:
            continue
        objects = sorted(entry.rglob("*.o"))
        if objects:
            found[match.group("slug")] = objects
    return found


def find_module(build_dir: Path) -> Path | None:
    """The one shared extension module a seam-mode build produces, if it has been linked yet."""
    candidates = sorted(build_dir.rglob("_core*.so"))
    return candidates[0] if candidates else None


def _check_tier_widths(census: dict[str, collections.Counter[str]]) -> list[str]:
    """The baseline tier must be at the floor, and something above it must not be."""
    problems: list[str] = []
    baseline = min(census)  # tier ids sort v1 < v2 < v3 < v4-*
    leak = {k: census[baseline][k] for k in _BASELINE_FORBIDDEN if census[baseline][k]}
    if leak:
        problems.append(
            f"the baseline tier {baseline} holds {leak}, which its own predicate does not guarantee "
            "the CPU can run"
        )
    if not [
        t
        for t in census
        if t != baseline and any(census[t][k] for k in _BASELINE_FORBIDDEN)
    ]:
        problems.append(
            "no tier above the baseline holds a single widened instruction: the tiers are compiled "
            "but carry the same code, so the run-time dispatch has nothing to choose between"
        )
    return problems


def _check_tier_exports(exports: dict[str, dict[str, str]]) -> list[str]:
    """Nothing a tier exports may be interposable: that is the static collapse arriving later."""
    problems: list[str] = []
    owners: dict[str, list[str]] = collections.defaultdict(list)
    for tier, exported in exports.items():
        for name in exported:
            if "monoprop" in name:
                owners[name].append(tier)
    collisions = {n: t for n, t in owners.items() if len(t) > 1}
    if collisions:
        problems.append(
            f"{len(collisions)} monoprop symbol(s) are exported by more than one tier, so the dynamic "
            "linker binds one copy for all of them: "
            + "; ".join(
                f"{_demangle(n)[:90]} ({', '.join(sorted(t))})"
                for n, t in sorted(collisions.items())[:3]
            )
        )
    weak = sorted(
        (tier, name)
        for tier, exported in exports.items()
        for name, kind in exported.items()
        if "monoprop" in name and kind in {"W", "w", "V", "v"}
    )
    if weak:
        problems.append(
            f"{len(weak)} monoprop symbol(s) are exported *weakly* from a tier and can be interposed "
            "by another module: "
            + "; ".join(f"{t}: {_demangle(n)[:80]}" for t, n in weak[:3])
        )
    return problems


def _report_seam_abi(
    dsos: dict[str, Path],
    exports: dict[str, dict[str, str]],
    imports: dict[str, set[str]],
    module: Path | None,
    module_exports: dict[str, str],
) -> list[str]:
    """Print the per-tier symbol counts and the seam ABI; a tier may import only what the loader has.

    Not a preference: a monoprop symbol a tier imports and the loading module does not export kills the
    process at load time on a symbol lookup.
    """
    problems: list[str] = []
    _out()
    for tier in sorted(dsos):
        wanted = sorted(n for n in imports[tier] if "monoprop" in n)
        exported = sum(1 for n in exports[tier] if "monoprop" in n)
        _out(
            f"{tier:28s} exports {exported:2d} monoprop symbol(s), imports {len(wanted):2d}"
        )
        missing = [n for n in wanted if n not in module_exports]
        if module is not None and missing:
            problems.append(
                f"tier {tier} imports {len(missing)} monoprop symbol(s) that {module.name} does not "
                "export, so loading it fails at run time: "
                + "; ".join(_demangle(n)[:90] for n in missing[:3])
            )
    if module_exports:
        abi = sorted({n for tier in dsos for n in imports[tier] if "monoprop" in n})
        _out(f"\nthe seam ABI is {len(abi)} symbol(s) wide:")
        for name in abi:
            _out(f"  {_demangle(name)[:120]}")
    return problems


def check_tier_dso(dsos: dict[str, Path], module: Path | None) -> int:
    """The structural checks for one-shared-object-per-tier."""
    failures: list[str] = []

    _out(_census_header())
    census: dict[str, collections.Counter[str]] = {}
    exports: dict[str, dict[str, str]] = {}
    imports: dict[str, set[str]] = {}
    for tier, path in dsos.items():
        census[tier] = wide_census(path)
        exports[tier], imports[tier] = dynamic_symbols(path)
        _out(_census_row(path.name, census[tier]))

    module_exports: dict[str, str] = {}
    if module is not None:
        module_census = wide_census(module)
        module_exports, _ = dynamic_symbols(module)
        _out(
            _census_row(
                module.name + "  (shared, every machine runs it)", module_census
            )
        )
        # The safety property, and the one narrow-seam could not offer: what every machine executes
        # regardless of its CPU must hold nothing above the baseline. Reachability arguments are not
        # good enough here -- a wheel's ISA floor has to be structural.
        leaked = {k: module_census[k] for k in _BASELINE_FORBIDDEN if module_census[k]}
        if leaked:
            failures.append(
                f"the shared module {module.name} holds instructions above the baseline ISA "
                f"({', '.join(f'{k}={v}' for k, v in leaked.items())}); every machine the wheel "
                "installs on executes this module, whatever tier was selected"
            )

    failures += _check_tier_widths(census)
    failures += _check_tier_exports(exports)
    failures += _report_seam_abi(dsos, exports, imports, module, module_exports)
    if failures:
        _err(f"\nFAIL: {len(failures)} problem(s) with this tier-dso build.")
        for problem in failures:
            _err(f"  - {problem}")
        return 1
    _out(
        "\nOK: every tier is its own link, keeps its own code, exports nothing another tier does, "
        "and imports only what the loading module exports. The shared module is at the baseline."
    )
    return 0


def check_narrow_seam(objects: dict[str, list[Path]], module: Path | None) -> int:
    """The weak-symbol disjointness check for all-four-tiers-in-one-link. Expected to fail."""
    _out(
        f"{'tier':24s} {'objects':>8s} {'weak syms':>10s} "
        + " ".join(f"{k:>9s}" for k in _WIDE)
    )
    per_tier: dict[str, set[str]] = {}
    for slug, objs in objects.items():
        weak: set[str] = set()
        counts: collections.Counter[str] = collections.Counter()
        for obj in objs:
            weak |= weak_symbols(obj)
            counts.update(wide_census(obj))
        per_tier[slug] = weak
        _out(
            f"{slug:24s} {len(objs):8d} {len(weak):10d} "
            + " ".join(f"{counts[k]:9d}" for k in _WIDE)
        )

    if module is not None:
        counts = wide_census(module)
        _out(
            f"\n{'linked ' + module.name:24s} {'':8s} {'':10s} "
            + " ".join(f"{counts[k]:9d}" for k in _WIDE)
        )

    owners: dict[str, list[str]] = collections.defaultdict(list)
    for slug, weak in per_tier.items():
        for name in weak:
            owners[name].append(slug)
    shared = {name: slugs for name, slugs in owners.items() if len(slugs) > 1}

    if not shared:
        _out(
            "\nOK: every tier's weak symbols are its own; the linker keeps all of them."
        )
        return 0

    _err(
        f"\nFAIL: {len(shared)} weak symbol(s) are defined by more than one tier. The linker keeps one "
        "definition per name, so those tiers run the same code -- whichever the link line lists first "
        "-- whatever the run-time dispatch selected. This is the known narrow-seam defect; build with "
        "monoprop_FAT_BINARY_MODE=tier-dso instead."
    )
    for name in sorted(shared)[:5]:
        _err(f"  {', '.join(sorted(shared[name]))}: {_demangle(name)[:150]}")
    if len(shared) > 5:
        _err(f"  ... and {len(shared) - 5} more")
    return 1


def configured_mode(build_dir: Path) -> str:
    """The fat-binary shape this tree was *configured* for, from its own CMakeCache.

    Read rather than inferred from which artefacts are on disk. A build tree is incremental and mode
    switches leave the previous shape's outputs in place -- a narrow-seam build over a tier-dso one
    still has four libmonoprop-tier-*.so lying beside it -- so file presence identifies the last mode
    that ran, not the one that produced the module being checked.
    """
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return "unconfigured"
    settings: dict[str, str] = {}
    for line in cache.read_text().splitlines():
        match = re.match(r"^(?P<key>monoprop_[A-Z_]+):[A-Z]+=(?P<value>.*)$", line)
        if match is not None:
            settings[match.group("key")] = match.group("value")
    if settings.get("monoprop_ENABLE_FAT_BINARY", "OFF").upper() not in {
        "ON",
        "TRUE",
        "1",
        "YES",
    }:
        return "single-isa"
    return settings.get("monoprop_FAT_BINARY_MODE", "whole-library")


def main() -> int:
    """Read which seam packaging this build tree was configured for, and run that shape's checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "build_dir",
        type=Path,
        help="a configured build tree, e.g. build/editable/Release",
    )
    parser.add_argument(
        "--module",
        type=Path,
        default=None,
        help="the shared module to check against (default: the _core extension under this build tree)",
    )
    args = parser.parse_args()

    module = args.module if args.module is not None else find_module(args.build_dir)
    mode = configured_mode(args.build_dir)
    _out(f"{args.build_dir}: monoprop_FAT_BINARY_MODE={mode}\n")

    if mode == "tier-dso":
        dsos = find_tier_dsos(args.build_dir)
        if not dsos:
            _err(
                f"{args.build_dir} is configured tier-dso but holds no libmonoprop-tier-*.so; build it first"
            )
            return 1
        return check_tier_dso(dsos, module)

    if mode == "narrow-seam":
        objects = find_tier_objects(args.build_dir)
        if not objects:
            _err(
                f"{args.build_dir} is configured narrow-seam but holds no tiered objects; build it first"
            )
            return 1
        return check_narrow_seam(objects, module)

    _err(
        f"nothing to check: {mode} puts each tier in its own module (or has only one), where there is "
        "no single link for the tiers to collapse inside"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
