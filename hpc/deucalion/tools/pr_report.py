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
that was not its own. Every input is named explicitly.
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


def run_ab_summary(cell: Path, *, allow_both_placed: bool) -> tuple[int, str]:
    """Return (exit code, markdown) from ab_summary.py over one cell directory."""
    argv = [sys.executable, str(HERE / "ab_summary.py"), str(cell)]
    if allow_both_placed:
        argv.append("--allow-both-placed")
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


def main(argv: list[str]) -> int:
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

    cells: list[tuple[str, int, str, Path]] = []
    for root in args.roots:
        for d in sorted(root.iterdir() if root.is_dir() else []):
            m = CELL_DIR_RE.match(d.name)
            if m and d.is_dir():
                cells.append((m["workload"], int(m["nodes"]), m["tag"] or "", d))
    if not cells:
        print(f"refusing: no ab-*-N* cell directories under {args.roots}", file=sys.stderr)
        return 2

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
    grid = [c for c in cells if c[2].startswith(GRID_PREFIX)]
    extra = [c for c in cells if not c[2].startswith(GRID_PREFIX)]

    for title, group in (("The common grid", grid), ("Highlight cells", extra)):
        if not group:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for workload, nodes, tag, d in sorted(group, key=lambda c: (c[2], c[0], c[1])):
            label = tag or workload
            lines.append(f"### `{label}` -- {workload}, N={nodes}")
            lines.append("")
            rc, body = run_ab_summary(d, allow_both_placed=not args.placement_pr)
            if rc != 0:
                void.append(f"{label} / {workload} / N={nodes} (ab_summary exit {rc})")
                lines.append(
                    f"> **VOID -- `ab_summary.py` exited {rc}. This is a diagnostic, not a "
                    "result. Do not read the absence of a difference below as an absence of "
                    "a difference.**"
                )
                lines.append("")
            md5s = cell_core_md5s(d)
            main_h, port_h = md5s.get("main", set()), md5s.get("port", set())
            if len(main_h) > 1 or len(port_h) > 1:
                md5_notes.append(f"{label}/N{nodes}: an arm served more than one binary")
            if main_h and main_h == port_h:
                md5_notes.append(f"{label}/N{nodes}: both arms share a `_core.so` -- not an A/B")
            lines.append(
                "    main _core.so: " + ", ".join(sorted(main_h) or ["(unrecorded)"]) + "\n"
                "    port _core.so: " + ", ".join(sorted(port_h) or ["(unrecorded)"])
            )
            lines.append("")
            lines.append(demote(body).rstrip())
            lines.append("")

    if void or md5_notes:
        lines.append("## Refusals")
        lines.append("")
        for v in void:
            lines.append(f"- **VOID** {v}")
        for n in md5_notes:
            lines.append(f"- **PROVENANCE** {n}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(cells)} cells, {len(void)} void)")
    return 1 if (void or md5_notes) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
