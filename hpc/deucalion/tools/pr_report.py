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

"""Collate one PR's A/B cells into a single report, in a format identical across PRs.

WHY THIS EXISTS AT ALL. Every campaign on this project grew its own collator -- collate.py,
collate2.py, graph_layout_collate.py, sparse_collate.py, sparse_stack_collate.py,
nocache_collate.py -- so no two results were in the same format and none of them could be put
side by side. This is the one collator. It calls ab_summary.py per cell (which owns the paired
ratios, the sign test and every refusal) and does nothing but arrange the answers.

WHAT IT REFUSES. A cell whose ab_summary exited non-zero is reported as VOID and makes this
tool exit non-zero too. It does not omit it: a report that silently drops a failed cell reads
exactly like a report of fewer cells, and "the arms measured the same" and "the instrument
never fired" are otherwise the same observation.

WHY IT TAKES DIRECTORIES AND NOT A GLOB. $PROJ/runs is shared. A campaign that collated by
globbing it once swept another session's A/B into the table and printed a 1.17x 6/6 regression
that was not its own. Every input is named explicitly -- and a named input that does not exist
is an abort, because the point of naming them is defeated by silently skipping one.

WHAT IT READS BESIDES THE CELLS.

  <root>/EXPECTED-CELLS      one cell-directory name per line, written by pr-ab.sh at SUBMIT
                             time. Without it, a cell the scheduler cancelled is simply absent
                             and the report reads clean; with it, absence is MISSING and fails.
  <cell>/SUMMARY-ARGS        the ab_summary flags that cell actually ran under, so re-running
                             ab_summary here reproduces its refusals rather than loosening them.
  <cell>/MISSING-ARTIFACTS   artifacts an srun failed to write. ab_summary cannot see these: it
                             pairs on the reps present, so lost reps shrink the count silently.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ab.sh names each cell directory ab-<workload>[-<tag>]-N<nodes>. The tag is the cell label,
# so `grid-hubbard` and `bind-cores` sort into the common grid and the highlights.
CELL_DIR_RE = re.compile(r"^ab-(?P<workload>[a-z]+)(?:-(?P<tag>.+))?-N(?P<nodes>\d+)$")

GRID_PREFIX = "grid-"


def summary_args(cell: Path, *, force_rearm: bool) -> tuple[list[str], str | None]:
    """Return the ab_summary flags for ``cell``, plus a note when they had to be guessed.

    This tool RE-RUNS ab_summary rather than reading the cell's own AB-SUMMARY.md, so it must
    reproduce the flags that cell ran under. ab.sh records them in ``SUMMARY-ARGS``. It did not
    always, and the fallback used to be an unconditional ``--allow-both-placed`` -- which
    disarmed the both-arms-placed refusal in the report for every cell, including the one cell
    a placement PR adds precisely to re-arm it. A disarmed refusal reads as a clean run, so when
    the flags are not recorded the report says so rather than quietly assuming the loose one.

    ``force_rearm`` (``--placement-pr``) applies ONLY where nothing was recorded. A placement PR
    adds one ``ALLOW_BOTH_PLACED=0`` cell; its other cells are ordinary grid cells that ran with
    the refusal disarmed on purpose, because both arms placing is the healthy state at layout B.
    Overriding every cell's recorded flags voided all four cells of a four-cell campaign in
    test -- a whole-report VOID that says nothing about the PR. Per-cell truth beats a
    campaign-wide switch, and the cell recorded its own.
    """
    recorded = cell / "SUMMARY-ARGS"
    if recorded.is_file():
        return [a for a in recorded.read_text().split() if a], None
    if force_rearm:
        return [], None
    return (
        ["--allow-both-placed"],
        f"{cell.name}: no `SUMMARY-ARGS` recorded, so the both-arms-placed refusal was "
        "DISARMED for this cell. Re-run it, or read its own `AB-SUMMARY.md`.",
    )


def run_ab_summary(cell: Path, argv_extra: list[str]) -> tuple[int, str]:
    """Return (exit code, markdown) from ab_summary.py over one cell directory."""
    argv = [sys.executable, str(HERE / "ab_summary.py"), str(cell), *argv_extra]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def demote(body: str, levels: int = 3) -> str:
    """Push ab_summary's own headings below this report's, so the outline nests."""
    return "\n".join(
        ("#" * levels + line) if line.startswith("#") else line for line in body.splitlines()
    )


def cell_core_md5s(cell: Path) -> dict[str, set[str]]:
    """Map side -> the set of _core.so md5s its result files recorded.

    Arm identity is that hash and nothing else. __version__ is a git describe stamped into the
    dist-info at install time and a later `cmake --build` + copy into the venv does not rewrite
    it, so an arm can advertise one commit while serving a binary from several commits later.
    Keying on the version has already refused five valid cells and would equally have passed
    two arms sharing a .so.
    """
    out: dict[str, set[str]] = {}
    for path in sorted(cell.glob("N*_*.json")):
        side = "port" if "_port_" in path.name else "main" if "_main_" in path.name else None
        if side is None:
            continue
        try:
            meta = json.loads(path.read_text()).get("meta", {})
        except (OSError, ValueError):
            continue
        md5 = meta.get("monoprop_core_md5")
        if md5:
            out.setdefault(side, set()).add(md5)
    return out


def check_manifests(roots: list[Path]) -> tuple[list[str], list[str]]:
    """Return (cells that were submitted and produced nothing, roots with no manifest).

    A CELL THAT NEVER RAN LEAVES NO TRACE, so the expectation has to come from the submitter.
    pr-ab.sh writes one cell-directory name per line into ``EXPECTED-CELLS`` as it submits.
    Without it this tool reports on whatever is on disk: a cell the scheduler cancelled, or one
    whose ``afterok`` dependency was never satisfied, simply vanishes and the report is a clean
    green report of fewer cells -- while pr-ab.sh's own comment claims "pr_report.py exits
    non-zero if any cell is missing or void". It does now.
    """
    missing: list[str] = []
    unverified: list[str] = []
    for root in roots:
        manifest = root / "EXPECTED-CELLS"
        if not manifest.is_file():
            unverified.append(
                f"{root}: no `EXPECTED-CELLS` manifest, so a cell that never ran cannot be "
                "distinguished from a cell that was never submitted. Check the count below "
                "against the cells you actually submitted."
            )
            continue
        missing += [
            f"{root.name}/{name}: submitted, but produced no directory"
            for name in (n.strip() for n in manifest.read_text().splitlines())
            if name and not (root / name).is_dir()
        ]
    return missing, unverified


def lost_artifacts(cell: Path) -> int:
    """Return how many artifacts ab.sh recorded as never written for ``cell``.

    ab_summary cannot see these: it pairs on the reps that ARE present, so a cell that lost two
    of six reps prints a confident 4/4 and never mentions the four that are gone.
    """
    lost = cell / "MISSING-ARTIFACTS"
    return len(lost.read_text().split()) if lost.is_file() else 0


def render_cell(
    cell: Path, label: str, workload: str, nodes: int, *, force_rearm: bool
) -> tuple[list[str], dict[str, list[str]]]:
    """Render one cell's section, and the refusals it raises.

    Returns ``(markdown lines, {"void": [...], "md5": [...], "args": [...]})``. A cell is VOID
    twice over here: once when ab_summary refuses it, and once when ab.sh recorded artifacts an
    srun never wrote -- those two are different failures and the report says which.
    """
    notes: dict[str, list[str]] = {"void": [], "md5": [], "args": []}
    lines = [f"### `{label}` -- {workload}, N={nodes}", ""]

    # `flags`, not `extra`: this used to rebind the highlight-cell LIST, one name for two
    # things. It survived only because the `for` target tuple is evaluated once before the loop
    # body ever runs, i.e. by luck rather than by design.
    flags, note = summary_args(cell, force_rearm=force_rearm)
    if note:
        notes["args"].append(note)

    lost = lost_artifacts(cell)
    if lost:
        notes["void"].append(f"{label} / {workload} / N={nodes} ({lost} artifact(s) never written)")
        lines += [
            f"> **VOID -- {lost} expected artifact(s) missing. The reps below are a SUBSET "
            "of the reps requested, and the `agree` count does not know it.**",
            "",
        ]

    rc, body = run_ab_summary(cell, flags)
    if rc != 0:
        notes["void"].append(f"{label} / {workload} / N={nodes} (ab_summary exit {rc})")
        lines += [
            f"> **VOID -- `ab_summary.py` exited {rc}. This is a diagnostic, not a result. "
            "Do not read the absence of a difference below as an absence of a difference.**",
            "",
        ]

    md5s = cell_core_md5s(cell)
    main_h, port_h = md5s.get("main", set()), md5s.get("port", set())
    if len(main_h) > 1 or len(port_h) > 1:
        notes["md5"].append(f"{label}/N{nodes}: an arm served more than one binary")
    if main_h and main_h == port_h:
        notes["md5"].append(f"{label}/N{nodes}: both arms share a `_core.so` -- not an A/B")
    lines += [
        "    main _core.so: " + ", ".join(sorted(main_h) or ["(unrecorded)"]) + "\n"
        "    port _core.so: " + ", ".join(sorted(port_h) or ["(unrecorded)"]),
        "",
        demote(body).rstrip(),
        "",
    ]
    return lines, notes


def main(argv: list[str]) -> int:
    """Collate every named campaign directory into one report; non-zero if any cell is void."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="campaign directories, named explicitly")
    ap.add_argument("--pr", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--placement-pr",
        action="store_true",
        help="this PR IS the placement fix, so re-arm the both-arms-placed refusal",
    )
    args = ap.parse_args(argv)

    # A ROOT THAT IS NOT THERE IS AN ABORT, not an empty iteration. `root.iterdir() if
    # root.is_dir() else []` silently contributed zero cells, so a mistyped or never-created
    # campaign directory passed alongside a good one produced a clean report of the good one
    # and exit 0 -- the report's own docstring promises the opposite. Every input is named
    # explicitly here precisely so that a name that names nothing is an error.
    for root in args.roots:
        if not root.is_dir():
            print(f"refusing: {root} is not a directory", file=sys.stderr)
            return 2

    cells: list[tuple[str, int, str, Path]] = []
    for root in args.roots:
        for d in sorted(root.iterdir()):
            m = CELL_DIR_RE.match(d.name)
            if m and d.is_dir():
                cells.append((m["workload"], int(m["nodes"]), m["tag"] or "", d))
    if not cells:
        print(f"refusing: no ab-*-N* cell directories under {args.roots}", file=sys.stderr)
        return 2

    missing_cells, unverified = check_manifests(args.roots)

    lines: list[str] = []
    lines.append(f"# A/B: `{args.pr}` vs `main`")
    lines.append("")
    if args.ref:
        lines.append(
            f"Port ref: `{args.ref}`. Baseline: `origin/main` -- one shared build, "
            "re-run interleaved inside every allocation."
        )
        lines.append("")
    lines.append(
        "Every cell is one allocation holding both arms, order flipped per `(rep, cell)`, "
        "`--bench-rounds=1`, `-s`, paired per-rep ratios with a sign-test p. Read `agree` "
        "before the ratio: 4/4 is p=0.125, the best a four-rep run can do."
    )
    lines.append("")

    void: list[str] = []
    md5_notes: list[str] = []
    arg_notes: list[str] = []
    grid = [c for c in cells if c[2].startswith(GRID_PREFIX)]
    highlights = [c for c in cells if not c[2].startswith(GRID_PREFIX)]

    for title, group in (("The common grid", grid), ("Highlight cells", highlights)):
        if not group:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for workload, nodes, tag, d in sorted(group, key=lambda c: (c[2], c[0], c[1])):
            body, notes = render_cell(
                d, tag or workload, workload, nodes, force_rearm=args.placement_pr
            )
            lines += body
            void += notes["void"]
            md5_notes += notes["md5"]
            arg_notes += notes["args"]

    sections = (
        ("VOID", void),
        ("MISSING", missing_cells),
        ("PROVENANCE", md5_notes),
        ("DISARMED", arg_notes),
        ("UNVERIFIED", unverified),
    )
    if any(items for _, items in sections):
        lines.append("## Refusals")
        lines.append("")
        for kind, items in sections:
            lines += [f"- **{kind}** {item}" for item in items]
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {args.out} ({len(cells)} cells, {len(void)} void, "
        f"{len(missing_cells)} never ran)"
    )
    # `unverified` alone does not fail: a hand-collated directory from before the manifest
    # existed is still a legitimate input, and the report says so in the text. A cell that WAS
    # submitted and produced nothing is a definite failure and does fail.
    return 1 if (void or md5_notes or arg_notes or missing_cells) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
