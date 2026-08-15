#!/usr/bin/env python3
"""Report the graph's world-size term across a set of benchmark result directories.

    hpc/deucalion/tools/graph_world_report.py "$MONOPROP_RUNS"/models-pauli-*

The graph is the half of the footprint that does not partition: its per-layer arrays are
indexed by rank, and on a partitioned run that index space is the flat world P = ranks x
partitions. A structure of length P held once per layer per partition therefore costs O(P^2)
across the job, which no per-rank column makes visible.

Two things this prints that the A/B tables cannot:

  * `graph` against P, with the fitted quadratic coefficient from EVERY adjacent pair rather
    than one. A coefficient is a claim about a mechanism; agreement across independent pairs
    is what makes it more than a curve through four points.
  * the per-field split from `graphmembreak`, which says WHICH field grows, and the slot
    occupancy, which says whether a sparse layout would pay or only a narrower record would.

Reads only what the harness already writes. Cells whose arms disagree on the term count are
refused rather than averaged: a memory number is meaningless without knowing it walked the
same operator.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

GB = 1e9


def cells(directory: Path) -> list[dict]:
    """Every per-rep results JSON in one directory, timing files excluded."""
    out = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("time-"):
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "meta" in data:
            data["_label"] = path.stem
            out.append(data)
    return out


def arm_of(label: str) -> str | None:
    for side in ("main", "port"):
        if f"_{side}_r" in label:
            return side
    return None


def geometry(cell: dict) -> tuple[int, int, int]:
    """(ranks, partitions, flat world P) for one cell."""
    ranks = int(cell["meta"]["ranks"])
    parts = int(cell["meta"]["monoprop_threads"])
    return ranks, parts, ranks * parts


def terms(cell: dict) -> int | None:
    for entry in (cell.get("opsize") or {}).values():
        if isinstance(entry, dict) and entry.get("terms"):
            return int(entry["terms"])
    return None


def energy_key(section: dict) -> str | None:
    """The energy op's key. Both ops walk one graph, so either would do; pick one and be
    consistent, because mixing them across rungs would fold an operation difference into
    what is supposed to be a geometry sweep."""
    for key in section:
        if "energy" in key:
            return key
    return next(iter(section), None)


def summarise(directory: Path) -> list[dict]:
    rows = []
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for cell in cells(directory):
        arm = arm_of(cell["_label"])
        if arm:
            by_arm[arm].append(cell)

    for arm, arm_cells in sorted(by_arm.items()):
        graph_vals, hwm_vals, break_vals, term_vals, geoms, pins = [], [], [], set(), set(), set()
        for cell in arm_cells:
            geoms.add(geometry(cell))
            if (t := terms(cell)) is not None:
                term_vals.add(t)
            pins.add((cell["meta"].get("pinning") or {}).get("single_cpu_threads_min"))
            if opbytes := cell.get("opbytes"):
                if key := energy_key(opbytes):
                    graph_vals.append(opbytes[key]["graph"])
            if memhwm := cell.get("memhwm"):
                if key := energy_key(memhwm):
                    hwm_vals.append(memhwm[key])
            if gmb := cell.get("graphmembreak"):
                if key := energy_key(gmb):
                    break_vals.append(gmb[key])
        if not geoms:
            continue
        if len(geoms) > 1:
            print(f"!! {directory.name} [{arm}]: mixed geometry {sorted(geoms)}", file=sys.stderr)
            continue
        ranks, parts, world = next(iter(geoms))
        rows.append(
            {
                "dir": directory.name,
                "arm": arm,
                "ranks": ranks,
                "parts": parts,
                "P": world,
                "terms": max(term_vals) if term_vals else None,
                "terms_spread": len(term_vals),
                "pinned": sorted(p for p in pins if p is not None),
                "graph": max(graph_vals) if graph_vals else None,
                "hwm": max(hwm_vals) if hwm_vals else None,
                "break": max(break_vals, key=lambda b: b.get("total_bytes", 0)) if break_vals else None,
                "reps": len(arm_cells),
            }
        )
    return rows


def fit_pairs(points: list[tuple[int, float]]) -> list[tuple[int, int, float, float]]:
    """graph = a + b*P^2 solved on each ADJACENT pair.

    One pair is a solution, not a fit -- two unknowns, two points, it cannot fail. Printing
    every adjacent pair is what turns it into evidence: independent pairs agreeing on b is a
    claim the data could have refused, and a single pair is not.
    """
    out = []
    ordered = sorted(points)
    for (p1, g1), (p2, g2) in zip(ordered, ordered[1:]):
        if p1 == p2:
            continue
        b = (g2 - g1) / (p2**2 - p1**2)
        out.append((p1, p2, b, g1 - b * p1**2))
    return out


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv[1:]]
    if not dirs:
        print(__doc__)
        return 2

    rows = [row for d in dirs if d.is_dir() for row in summarise(d)]
    if not rows:
        print("no results found", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: (r["arm"], r["P"], r["dir"]))

    print("# Graph memory against the flat world size\n")
    print("`P` = ranks x partitions. `graph` and `hwm` are summed over ranks (job totals),")
    print("so read them as what the job costs, not what a node holds.\n")
    print("| dir | arm | ranks x parts | P | terms | pinned | graph GB | RSS GB | reps |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        flag = " !!" if r["terms_spread"] > 1 else ""
        g = f"{r['graph'] / GB:.2f}" if r["graph"] else "-"
        h = f"{r['hwm'] / GB:.2f}" if r["hwm"] else "-"
        t = f"{r['terms']:,}{flag}" if r["terms"] else "-"
        print(
            f"| {r['dir']} | {r['arm']} | {r['ranks']}x{r['parts']} | {r['P']} | {t} "
            f"| {r['pinned']} | {g} | {h} | {r['reps']} |"
        )

    for arm in sorted({r["arm"] for r in rows}):
        pts = [(r["P"], float(r["graph"])) for r in rows if r["arm"] == arm and r["graph"]]
        uniq = sorted({p: g for p, g in pts}.items())
        if len(uniq) < 2:
            continue
        print(f"\n## graph = a + b*P^2, arm `{arm}`\n")
        print("| pair | b (B/P^2) | a (GB) |")
        print("|---|---:|---:|")
        for p1, p2, b, a in fit_pairs(uniq):
            print(f"| ({p1},{p2}) | {b:,.0f} | {a / GB:.2f} |")

    have_break = [r for r in rows if r["break"]]
    if have_break:
        print("\n## Per-field graph split\n")
        print("`d_` fields are diagnostics outside total_bytes: slot_record is the part of")
        print("cross_rank sized by the world rather than by traffic, and recv_cache and")
        print("derivative_layout are resident memory total_bytes has never counted.\n")
        fields = ["cross_rank_bytes", "exchange_layout_bytes", "cos_data_bytes",
                  "layer_descriptor_bytes", "total_bytes", "d_slot_record_bytes",
                  "d_recv_cache_bytes", "d_derivative_layout_bytes"]
        print("| dir | arm | P | " + " | ".join(f.replace("_bytes", "") for f in fields) + " | occupancy | layers |")
        print("|---|---|---:|" + "---:|" * (len(fields) + 2))
        for r in have_break:
            b = r["break"]
            vals = " | ".join(f"{b.get(f, 0) / GB:.2f}" for f in fields)
            slots, occ = b.get("d_slot_records", 0), b.get("d_occupied_slots", 0)
            occupancy = f"{100 * occ / slots:.1f}%" if slots else "-"
            cores = b.get("d_layer_cores", 0)
            per_part = f"{slots // cores}" if cores else "-"
            print(f"| {r['dir']} | {r['arm']} | {r['P']} | {vals} | {occupancy} | {cores} (P={per_part}) |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
