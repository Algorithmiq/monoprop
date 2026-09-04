# Four-arm knob-sweep summary for knob-sweep-n1.sh.
#
# ab_summary.py answers "is the branch faster than main". This answers "which of the
# branch's mechanisms is responsible", by running one binary under different runtime knobs
# so the arms differ in exactly one thing each:
#
#   main      the baseline build, cannot pin at layout B at all
#   grouped   the branch as shipped -- pinned, two-level barrier
#   flat      monoprop_BARRIER_GROUPING=0: pinned, FLAT barrier
#   unpinned  monoprop_PARTITION_PINNING=0: no placement
#
# Three of the four are the same binary, so COMMPROF is directly comparable across them --
# which a main-vs-branch comparison can never be, because main has no monoprop_COMM_PROFILE
# and emits nothing at all.
#
# Every statistic here is imported from ab_summary rather than reimplemented. The paired
# per-rep ratio and the `agree` count are the reason its tables can be trusted, and a second
# hand-rolled copy would drift from the first exactly when it mattered.
#
#   python hpc/deucalion/tools/knob_summary.py <results-dir>

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_summary import (
    Cell,
    agreement,
    collect,
    fmt_agree,
    fmt_ms,
    medians,
    op_rank,
    paired_ratios,
    short_op,
    verdict,
)

# Same shape as ab_summary's, widened from main|port to the four arm names.
LABEL_RE = re.compile(
    r"^(?P<nodes>\d+)_(?P<layout>[A-Z]_\d+x\d+)_(?P<group>fresh|graph)"
    r"_(?P<side>main|grouped|flat|unpinned)_r(?P<rep>\d+)$"
)

ARMS = ("main", "grouped", "flat", "unpinned")

# `grouped` is the branch as shipped, so it is what every variant is measured against;
# `main` is the anchor the original regression was reported in.
BASELINE = "grouped"
ANCHOR = "main"

# What each arm's knobs claim to have done, checked against COMMPROF rather than assumed.
# A knob that silently did nothing yields four identical arms, which reads exactly like
# "the mechanism is not responsible" -- the one conclusion this tool exists to support.
EXPECTED_STATE = {
    "grouped": {"barrier_groups": "> 0", "pinned": "> 0"},
    "flat": {"barrier_groups": "== 0", "pinned": "> 0"},
    "unpinned": {"pinned": "== 0"},
}


def commprof_rows(results_dir: Path, side: str) -> list[dict[str, str]]:
    """Return every COMMPROF record this arm emitted, across all reps."""
    rows = []
    for path in sorted(results_dir.glob(f"N*_{side}_r*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("COMMPROF"):
                fields = (kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                rows.append(dict(fields))
    return rows


def profile_median(rows: list[dict[str, str]], field: str) -> float | None:
    """Median of one COMMPROF field over every rank and rep, or None."""
    values = [float(r[field]) for r in rows if field in r]
    return statistics.median(values) if values else None


def check_state(
    results_dir: Path, present: set[str]
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    """Verify each arm's knobs actually took effect. Returns (refusals, per-arm rows).

    Only arms that actually ran are checked. A sweep legitimately drops an arm once it has
    answered its question -- `flat` did, at N=1 -- and demanding all four forever would mean
    every follow-up run refuses itself. What is *not* optional is the baseline and the
    anchor: without both there is nothing to take a ratio against.
    """
    refusals: list[str] = []
    rows_by_arm = {arm: commprof_rows(results_dir, arm) for arm in ARMS}

    refusals += [
        f"`{required}` did not run; there is nothing to compare against"
        for required in (BASELINE, ANCHOR)
        if required not in present
    ]

    if rows_by_arm[ANCHOR]:
        refusals.append(
            f"`{ANCHOR}` emitted COMMPROF, so it is not a baseline build: "
            "monoprop_COMM_PROFILE does not exist there. A venv is mislabelled."
        )

    for arm, expected in EXPECTED_STATE.items():
        if arm not in present:
            continue
        rows = rows_by_arm[arm]
        if not rows:
            refusals.append(
                f"`{arm}` emitted no COMMPROF at all -- its runtime state is unverifiable. "
                "Check that -s was passed and monoprop_COMM_PROFILE=1 reached the ranks."
            )
            continue
        for field, condition in expected.items():
            seen = sorted({r[field] for r in rows if field in r})
            if not seen:
                refusals.append(f"`{arm}`: COMMPROF carries no `{field}`")
                continue
            ok = all((int(v) > 0 if condition == "> 0" else int(v) == 0) for v in seen)
            if not ok:
                refusals.append(
                    f"`{arm}`: expected {field} {condition}, saw {{{', '.join(seen)}}} -- "
                    "the knob did not take effect, so this arm is not the arm it is named."
                )
    return refusals, rows_by_arm


def check_work(cells: dict[str, Cell]) -> list[str]:
    """Every arm must have propagated the same operator, and used one round."""
    refusals: list[str] = []
    terms = {
        entry["terms"]
        for cell in cells.values()
        for entry in cell.results.get("opsize", {}).values()
        if isinstance(entry, dict) and "terms" in entry
    }
    if terms and (max(terms) - min(terms)) / max(terms) > 0.001:
        refusals.append(f"term counts differ across arms: {sorted(terms)}")

    rounds = {
        cell.results.get("meta", {}).get("bench_rounds")
        for cell in cells.values()
        if cell.results
    } - {None}
    if rounds - {1}:
        refusals.append(
            f"bench_rounds={sorted(rounds)}, must be 1: pedantic builds round k+1's "
            "arguments before releasing round k's, which doubles peak memory."
        )
    return refusals


def emit_op(op: str, by_arm: dict[str, list[Cell]], rows_by_arm: dict) -> None:
    """Print one operation's four-arm table."""
    print(f"### {short_op(op)}\n")
    headers = [
        "arm",
        "med (ms)",
        "min (ms)",
        f"vs {BASELINE}",
        "agree",
        f"vs {ANCHOR}",
        "agree",
        "groups",
        "pinned",
        "mpi_s",
        "peers_s",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")

    for arm in ARMS:
        cells = by_arm.get(arm, [])
        runs = [c.times[op] for c in cells if op in c.times]
        if not runs:
            continue
        med = medians([r["median"] for r in runs])
        low = min((r["min"] for r in runs), default=None)

        cols = []
        for base in (BASELINE, ANCHOR):
            if arm == base or base not in by_arm:
                cols += ["-", "-"]
                continue
            pr = paired_ratios(by_arm[base], cells, op, "min")
            ratio, agree, compared = agreement(pr)
            cols += [verdict(ratio), fmt_agree(agree, compared, pr)]

        rows = rows_by_arm.get(arm, [])
        groups = sorted({r["barrier_groups"] for r in rows if "barrier_groups" in r})
        pinned = sorted({r["pinned"] for r in rows if "pinned" in r})
        mpi = profile_median(rows, "mpi_s")
        peers = profile_median(rows, "barrier_peers_s")

        print(
            f"| `{arm}` | {fmt_ms(med)} | {fmt_ms(low)} "
            f"| {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} "
            f"| {','.join(groups) or '-'} | {','.join(pinned) or '-'} "
            f"| {'-' if mpi is None else f'{mpi:.2f}'} "
            f"| {'-' if peers is None else f'{peers:.2f}'} |"
        )
    print()


def main(argv: list[str]) -> int:
    """Render the sweep, refusing when an arm is not in the state it claims."""
    if len(argv) < 2:
        print("usage: knob_summary.py <results-dir>", file=sys.stderr)
        return 2
    results_dir = Path(argv[1])
    cells, problems = collect(results_dir, LABEL_RE)
    if not cells:
        print(f"no sweep artifacts found in {results_dir}", file=sys.stderr)
        return 1

    print("# N=1 knob sweep: which mechanism costs `propagate` its 1.34x?\n")
    print(f"Results: `{results_dir}`\n")
    print(
        "Ratios are medians of per-rep paired ratios, below 1.00 meaning the arm is "
        "faster than the column's baseline. Read `agree` first: anything short of all "
        "reps means the arm is unresolved, whatever the ratio says.\n"
    )
    print(
        "`groups`/`pinned` come from COMMPROF and are what prove each knob took effect. "
        f"`{ANCHOR}` is silent there by construction -- it has no "
        "monoprop_COMM_PROFILE.\n"
    )

    by_arm: dict[str, list[Cell]] = {}
    for cell in cells.values():
        by_arm.setdefault(cell.side, []).append(cell)

    refusals, rows_by_arm = check_state(results_dir, set(by_arm))
    refusals += check_work(cells)
    refusals += problems

    for op in sorted({op for cell in cells.values() for op in cell.times}, key=op_rank):
        emit_op(op, by_arm, rows_by_arm)

    if refusals:
        print("## REFUSED\n")
        print("\n".join(f"- {r}" for r in refusals))
        print("\nThe tables above are a diagnostic, not a result.")
        return 1
    print("## Provenance\n")
    print("- every arm reached the runtime state its knobs claim")
    print(f"- arms compared: {', '.join(sorted(by_arm))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
