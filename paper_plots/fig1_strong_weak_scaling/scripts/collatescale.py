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

"""Flatten every single-arm scaling cell into one long TSV -- the only file the figures read.

The figures must regenerate with no cluster access, which is the check that the plots and the
data cannot drift apart. So everything a figure or caption needs lands here, one row per rep:
identity (ladder, nodes, R, P), the measurement (seconds, terms, peak RSS), and the gate columns
that say whether the row is quotable at all. A row that fails its gate is KEPT and flagged, never
dropped -- a silently short cell reads as a clean one.

Reps are keyed by (dir, rep) because a rung can be spread over several allocations: the existing
ladders have r1/r3/r10 directories for the same node count, and topping a rung up to 5 reps adds
another. Merging them by rep number alone would collide rep 1 of four different jobs.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path(os.environ.get("MONOPROP_RUNS", "/projects/EEHPC-DEV-2026D08-260/runs"))
PINNED_MD5 = "1157a5e2b421fb8bd18fc6d16fa39778"
RANKS_PER_NODE, PARTITIONS = 8, 16
MIN_REPS_FOR_SPREAD = 3  # fewer reps than this is a shakedown, not a measurement

# The two finished ladders were measured by the two-arm driver; only its `port` side is the
# shipped configuration. They enter the sequence under the names their size actually is.
LEGACY = {"weak": "weak_97m", "strong": "strong_s5"}
DIR_RE = re.compile(r"^(ab|scale)-hubbard-296-(.+?)-r(\d+)-N(\d+)$")


def read_tsv(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as fh:
        rows = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    if not rows:
        return []
    head = rows[0]
    return [dict(zip(head, r, strict=True)) for r in rows[1:] if len(r) == len(head)]


def seconds(path):
    """The rep's wall time. rounds=1 so median is the single measurement; a cell still filling
    leaves a half-written file, which is a missing rep and not a zero."""
    try:
        with Path(path).open() as fh:
            doc = json.load(fh)
        return float(doc["benchmarks"][0]["stats"]["median"])
    except (ValueError, OSError, KeyError, IndexError):
        return None


def expected_terms(d):
    """The cell's pinned term count, 0 when the cell never wrote its metadata."""
    cell_meta = d / "CELL-META.tsv"
    if not cell_meta.exists():
        return 0
    meta = {}
    for ln in cell_meta.read_text().splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2:
            meta[parts[0]] = parts[1]
    return int(meta.get("expect_terms") or 0)


def peak_rss(d, kind):
    """rep -> (sum_kb, worst_kb, ranks). The two-arm driver's `port` side only."""
    return {
        r["rep"]: (int(r["sum_kb"]), int(r["worst_kb"]), int(r["ranks"]))
        for r in read_tsv(d / "PEAK-RSS.tsv")
        if not (kind == "ab" and r.get("side") != "port")
    }


def gate(chk, terms, expect):
    """One gate, evaluated in one place so no downstream consumer can forget a clause."""
    bad = []
    if chk.get("so_md5") != PINNED_MD5:
        bad.append("md5")
    if chk.get("arena_seen") not in ("default", "UNSET"):
        bad.append("arena")
    if chk.get("routing_seen") not in ("linear", "default"):
        bad.append("routing")
    if expect and terms != expect:
        bad.append("terms")
    if chk.get("rc") not in ("0", ""):
        bad.append(f"rc{chk.get('rc')}")
    return "ok" if not bad else "+".join(bad)


def collect():
    rows = []
    for d in sorted(RUNS.iterdir()):
        name = d.name
        m = DIR_RE.match(name)
        if not m:
            continue
        kind, tag, _reps, nodes = (
            m.group(1),
            m.group(2),
            int(m.group(3)),
            int(m.group(4)),
        )
        ladder = LEGACY.get(tag) if kind == "ab" else tag
        if ladder is None:  # an unrelated 296 directory
            continue
        expect = expected_terms(d)
        rss = peak_rss(d, kind)

        for chk in read_tsv(d / "CELL-CHECKS.tsv"):
            if kind == "ab" and chk.get("side") != "port":
                continue
            label, rep = chk["label"], chk["rep"]
            secs = seconds(d / f"time-{label}.json")
            if secs is None:
                continue
            terms = int(chk.get("terms") or 0)
            sum_kb, worst_kb, ranks = rss.get(rep, (0, 0, 0))
            rows.append(
                {
                    "driver": kind,
                    "ladder": ladder,
                    "family": "weak" if ladder.startswith("weak") else "strong",
                    "nodes": nodes,
                    "ranks": nodes * RANKS_PER_NODE,
                    "world": nodes * RANKS_PER_NODE * PARTITIONS,
                    "srcdir": name,
                    "rep": rep,
                    "seconds": f"{secs:.4f}",
                    "terms": terms,
                    "terms_per_node": f"{terms / nodes:.0f}" if nodes else "0",
                    "rss_sum_kb": sum_kb,
                    "rss_worst_kb": worst_kb,
                    "rss_ranks": ranks,
                    "gate": gate(chk, terms, expect),
                }
            )
    return rows


def prefer_one_driver(rows):
    """A rung must not mix drivers. Two drivers measured the same configuration -- the two-arm
    A/B driver with monoprop_ROUTING=linear set explicitly, and the single-arm driver with it
    unset -- and the in-process gate proves the engine took the same path either way. But rep
    counts and allocations differ, so a rung takes ALL its reps from the single-arm driver where
    one ran, and falls back to the A/B driver's `port` side only where none did.

    Rows the rule discards are returned separately, not dropped silently: cells where both
    drivers ran are the equivalence check that justifies the fallback at all.
    """
    has_scale = {(r["ladder"], r["nodes"]) for r in rows if r["driver"] == "scale"}
    keep = [
        r
        for r in rows
        if r["driver"] == "scale" or (r["ladder"], r["nodes"]) not in has_scale
    ]
    shadowed = [r for r in rows if r not in keep]
    return keep, shadowed


def median(vals):
    v = sorted(vals)
    if not v:
        return 0.0
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def shared_cells(all_rows):
    """The same problem reached through different ladder definitions.

    "Same problem" is (term count, node count) -- NOT (ladder, node count). A cell is shared
    across the two families precisely when two curves need the same total at the same width, and
    those pairs carry the two curve names, so keying on the name would miss every one of them.
    Each group is a free consistency check: separate allocations, separate jobs, sometimes a
    different driver, and the medians must still agree.
    """
    per = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        if r["gate"] == "ok":
            key = (int(r["terms"]), r["nodes"])
            per[key][(r["ladder"], r["driver"], r["srcdir"])].append(
                float(r["seconds"])
            )
    out = []
    for (terms, n), grp in sorted(per.items()):
        if len(grp) < 2:
            continue
        meds = [(k, median(v), len(v)) for k, v in sorted(grp.items())]
        # A one- or two-rep source is a shakedown -- it exists to prove the cell fits in memory,
        # and the plan says no ratio from it is quotable. Listing it is right; letting it set the
        # spread is not: one shakedown rep read 41.56 s against two real medians at 36.6-37.0 and
        # would have printed a 14% disagreement between measurements that agree to 1%.
        real = [m for _, m, k in meds if k >= MIN_REPS_FOR_SPREAD]
        spread = max(real) / min(real) if len(real) >= 2 else 0.0
        if spread:
            out.append((terms, n, meds, spread))
    return out


def main():
    every = collect()
    if not every:
        sys.exit(f"no cells found under {RUNS}")
    rows, shadowed = prefer_one_driver(every)
    cols = list(rows[0])
    out = os.environ.get("OUT", "SCALE-CELLS.tsv")
    with Path(out).open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        fh.writelines("\t".join(str(r[c]) for c in cols) + "\n" for r in rows)

    per = defaultdict(list)
    for r in rows:
        per[(r["ladder"], r["nodes"])].append(r)
    bad = [r for r in rows if r["gate"] != "ok"]
    print(
        f"{out}: {len(rows)} reps over {len(per)} rungs, {len(bad)} failing their gate"
    )
    print(
        f"{'ladder':<12} {'N':>4} {'reps':>5} {'ok':>4} {'terms':>14} "
        f"{'M/node':>8} {'median s':>9} {'GiB/node':>9}"
    )
    for (lad, n), rs in sorted(per.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ok = [r for r in rs if r["gate"] == "ok"]
        s = sorted(float(r["seconds"]) for r in ok)
        med = (
            s[len(s) // 2]
            if len(s) % 2
            else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2
            if s
            else 0
        )
        gib = max((r["rss_sum_kb"] for r in rs), default=0) / 1024 / 1024 / n
        print(
            f"{lad:<12} {n:>4} {len(rs):>5} {len(ok):>4} {rs[0]['terms']:>14} "
            f"{int(rs[0]['terms']) / n / 1e6:>8.1f} {med:>9.2f} {gib:>9.1f}"
        )
    for r in bad:
        print(f"  GATE {r['ladder']} N={r['nodes']} rep {r['rep']} -> {r['gate']}")
    if shadowed:
        print(
            f"\n{len(shadowed)} A/B-driver reps shadowed by a single-arm run of the same rung "
            f"(kept out of the rung, used for the check below)"
        )
    sh = shared_cells(every)
    if sh:
        print(
            "\nshared cells -- the same (terms, nodes) reached through different ladders,"
        )
        print(
            "separate allocations, sometimes a different driver. Spread should be ~1%."
        )
        for terms, n, meds, spread in sh:
            flag = "" if spread <= 1.02 else "   <<< DISAGREES >2%"
            print(f"  {terms:>14} terms at N={n:<3} spread {spread:>6.4f}{flag}")
            for (lad, drv, src), m, k in meds:
                tag = (
                    ""
                    if k >= MIN_REPS_FOR_SPREAD
                    else "  shakedown, excluded from spread"
                )
                print(f"      {m:>9.2f} s  ({k} reps)  {lad:<11} {drv:<5} {src}{tag}")


main()
