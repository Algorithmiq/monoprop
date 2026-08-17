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

"""Collate a membisect run into a per-arm memory ledger plus paired time ratios.

    membisect_summary.py [--baseline main] [--arm-order a,b,c] [--cumulative a,b,c] <dir> [dir ...]

Explicit directories only, never a glob. `$PROJ/runs` is shared by every concurrent session on the
account, and a `models-*` glob has already swept another session's A/B into a table here and printed a
1.17x 6/6 "regression" that belonged to a different branch. The tell was a ledger column of exactly
1.00x; check the point count against the cells you actually submitted.

Three things this reconciles, each of which has misled a reading of these numbers before:

1. TWO KEY SETS. Arms predating a layout change emit different `d_*` keys and carry neither
   `matched_scratch_bytes` nor an arena-slack field; current arms emit the current set. Both are mapped
   onto one row shape rather than one being dropped.

2. SLACK IS VIRTUAL. `d_terms_slack_bytes` and `d_invidx_arena_slack_bytes` are `capacity() - size()`
   -- untouched, never faulted in. `total_bytes` counts them; resident memory does not. The TOUCHED
   column is the honest one and is what the per-arm deltas are taken over. Reading `total_bytes`
   instead is what made a commit that saves nothing at rest look like a campaign's biggest win.

3. RATIOS ARE PAIRED. Time ratios are the median of PER-REP ratios, never the ratio of two medians --
   the two arms of a rep run back to back on the same node, so a node-state swing hits both and
   cancels in the quotient. Ratio-of-medians once printed 6x noise as a clean "0.95x flat". The
   `agree` count is the sign test: 3/3 is p=0.25, 4/4 is p=0.125, 6/6 is p=0.031. Budget reps against
   what a sign test can resolve, and check whether a dissenting rep is a bad ARM or a bad
   DENOMINATOR (a baseline rep that moved makes every arm disagree at once).

The epoch-stamp adjustment is DERIVED, not measured, and is printed in its own column so it can never
be mistaken for a reading: arms without `matched_scratch_bytes` carry one u32 stamp per term, hence
exactly 4.0 B/term of payload.

Arm display order is `--arm-order` if given, else first-seen. It is deliberately NOT a hardcoded list:
one used to live here, it silently DROPPED any arm not in it, and a 16-cell job printed a one-row
table containing only the baseline -- indistinguishable from an arm that failed to run.
"""

from __future__ import annotations

import os
import re
import sys
from collections import OrderedDict

SLACK_KEYS = ("d_terms_slack_bytes", "d_invidx_arena_slack_bytes")
U32_STAMP_BYTES_PER_TERM = 4.0  # derived: one u32 MatchedEpochSet stamp per term

STRUCTS = OrderedDict(
    [
        ("indexing_bytes", "dedup"),
        ("operator_terms_bytes", "rows"),
        ("inverted_index_bytes", "invidx"),
        ("op_coeffs_bytes", "coeffs"),
        ("matched_scratch_bytes", "stamp"),
    ]
)

LOG_RE = re.compile(r"^(?P<cell>[A-Za-z0-9]+)_rep(?P<rep>\d+)_(?P<arm>.+)\.log$")


def parse_dir(path):
    """{(cell, arm): {'terms', 'rss_max', 'rss_sum', 'mem': {...}, 'reps': {rep: seconds}}}"""
    out = {}
    for name in sorted(os.listdir(path)):
        m = LOG_RE.match(name)
        if not m:
            continue
        cell, rep, arm = m.group("cell"), int(m.group("rep")), m.group("arm")
        rec = {"mem": {}, "terms": 0, "rss_max": 0.0, "rss_sum": 0.0, "reps": {}}
        with open(os.path.join(path, name)) as fh:
            for line in fh:
                if line.startswith("PROFMEM terms="):
                    for tok in line.split()[1:]:
                        k, _, v = tok.partition("=")
                        rec["mem"][k] = int(v)
                    rec["terms"] = rec["mem"].pop("terms", 0)
                    rec["mem"].pop("ranks", None)
                elif line.startswith("PROFRUN "):
                    for tok in line.split():
                        if tok.startswith("terms="):
                            rec["terms"] = rec["terms"] or int(tok[6:])
                        elif tok.startswith("rssmax_mb="):
                            rec["rss_max"] = float(tok[10:])
                        elif tok.startswith("rsssum_mb="):
                            rec["rss_sum"] = float(tok[10:])
                        elif tok.startswith("elapsed="):
                            rec["reps"][rep] = float(tok[8:])
        if not rec["mem"]:
            print("!! no PROFMEM in %s -- skipped" % name)
            continue
        # Keep the FIRST rep as the ledger (it is deterministic) and the MIN peak RSS across reps,
        # which is the least allocator-contaminated high-water mark.
        key = (cell, arm)
        if key in out:
            out[key]["rss_max"] = min(out[key]["rss_max"], rec["rss_max"]) or rec["rss_max"]
            out[key]["rss_sum"] = min(out[key]["rss_sum"], rec["rss_sum"]) or rec["rss_sum"]
            out[key]["reps"].update(rec["reps"])
            if out[key]["mem"] != rec["mem"]:
                print(
                    "!! %s/%s rep%d ledger differs between reps -- NOT deterministic, investigate"
                    % (cell, arm, rep)
                )
        else:
            out[key] = rec
    return out


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def arms_in_order(recs, cell, arm_order):
    """Arms present for `cell`: `arm_order` first, then anything unrecognised -- appended, never
    dropped, and announced on stderr so a new arm cannot silently vanish from the tables."""
    present = [a for (c, a) in recs if c == cell]
    known = [a for a in arm_order if a in present]
    unknown = [a for a in sorted(set(present) - set(arm_order))]
    if unknown and arm_order:
        sys.stderr.write("note: arm(s) not in --arm-order, appended: %s\n" % ", ".join(unknown))
    return known + unknown


def cells_in_order(recs):
    """Cells present, first-seen order preserved."""
    seen = []
    for cell, _arm in recs:
        if cell not in seen:
            seen.append(cell)
    return seen


def report_times(recs, cell, arm_order, base):
    """Per-rep PAIRED ratios against `base`, then the median. See the module docstring."""
    b = recs.get((cell, base))
    if not b or len(b["reps"]) < 2:
        return
    rows = [(a, recs[(cell, a)]) for a in arms_in_order(recs, cell, arm_order) if a != base]
    if not rows:
        return
    print(
        "\n**time vs %s** (per-rep paired ratios, R=%s)\n"
        % (base, os.environ.get("STAGE_RANKS", "?"))
    )
    print("| arm | median s | ratio | agree | per-rep |")
    print("|---|---:|---:|---:|---|")
    print("| %s | %.3f | 1.000x | -- | %s |"
          % (base, median(list(b["reps"].values())),
             " ".join("%.2f" % b["reps"][r] for r in sorted(b["reps"]))))
    for arm, rec in rows:
        shared = sorted(set(rec["reps"]) & set(b["reps"]))
        if not shared:
            continue
        ratios = [rec["reps"][r] / b["reps"][r] for r in shared]
        agree = sum(1 for x in ratios if x > 1.0)
        agree = max(agree, len(ratios) - agree)
        print(
            "| %s | %.3f | **%.4fx** | %d/%d | %s |"
            % (
                arm,
                median([rec["reps"][r] for r in shared]),
                median(ratios),
                agree,
                len(ratios),
                " ".join("%.4f" % x for x in ratios),
            )
        )
    print(
        "\nThe baseline's own per-rep seconds are printed above so a dissenting `agree` count can be "
        "read as a bad ARM or a bad DENOMINATOR. Never quote an agree count without checking which."
    )


def touched(rec):
    """total_bytes less the two capacity-only fields, plus the derived stamp where uncounted."""
    mem, terms = rec["mem"], rec["terms"]
    total = mem.get("total_bytes", 0)
    for k in SLACK_KEYS:
        total -= mem.get(k, 0)
    derived = 0.0
    if "matched_scratch_bytes" not in mem:
        derived = U32_STAMP_BYTES_PER_TERM * terms
    return total + derived, derived


def report(recs, cell, arm_order, cumulative, base):
    rows = [(a, recs[(cell, a)]) for a in arms_in_order(recs, cell, arm_order)]
    if not rows:
        return
    terms = rows[0][1]["terms"]
    bad = [a for a, r in rows if r["terms"] != terms]
    if bad:
        print(
            "!! %s: term counts differ across arms %s -- ratios are between different problems"
            % (cell, bad)
        )

    print("\n### %s  (%d terms, B/term)\n" % (cell, terms))
    hdr = ["arm"] + list(STRUCTS.values()) + ["slack*", "TOUCHED", "step", "rssmax_mb"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))

    prev = None
    for arm, rec in rows:
        mem, t = rec["mem"], float(rec["terms"])
        tch, derived = touched(rec)
        cells = []
        for key in STRUCTS:
            if key == "matched_scratch_bytes" and key not in mem:
                cells.append("%.3f*" % (derived / t))
            else:
                cells.append("%.3f" % (mem.get(key, 0) / t))
        slack = sum(mem.get(k, 0) for k in SLACK_KEYS) / t
        # Steps are meaningful only along a cumulative chain of arms; arms that differ by an env knob
        # pair with each other instead, so they get no step.
        step = ""
        if arm in cumulative and prev is not None:
            step = "%+.3f" % ((tch - prev) / t)
        if arm in cumulative:
            prev = tch
        print(
            "| %s | %s | %.3f | **%.3f** | %s | %.1f |"
            % (arm, " | ".join(cells), slack, tch / t, step, rec["rss_max"])
        )

    print(
        "\n`slack*` is capacity, not resident, and is excluded from TOUCHED. "
        "A `*` on the stamp column marks the derived u32 value, not a measurement."
    )

    b = recs.get((cell, base))
    if not b:
        return
    bt, _ = touched(b)
    for arm, rec in rows:
        if arm == base:
            continue
        at, _ = touched(rec)
        print(
            "- **%s vs %s** -- touched **%.3fx** (%.3f -> %.3f B/term); as printed incl. slack: "
            "%.3fx; peak RSS/rank: %.3fx"
            % (
                arm,
                base,
                bt / at if at else float("nan"),
                bt / b["terms"],
                at / rec["terms"],
                b["mem"]["total_bytes"] / float(rec["mem"]["total_bytes"] or 1),
                b["rss_max"] / rec["rss_max"] if rec["rss_max"] else float("nan"),
            )
        )


def main():
    args = sys.argv[1:]
    base, arm_order, cumulative, dirs = "main", [], [], []
    while args:
        a = args.pop(0)
        if a == "--baseline":
            base = args.pop(0)
        elif a == "--arm-order":
            arm_order = [x for x in args.pop(0).split(",") if x]
        elif a == "--cumulative":
            cumulative = [x for x in args.pop(0).split(",") if x]
        elif a.startswith("--"):
            raise SystemExit("unknown option %s\n\n%s" % (a, __doc__))
        else:
            dirs.append(a)
    if not dirs:
        raise SystemExit(__doc__)
    recs = {}
    for d in dirs:
        recs.update(parse_dir(d))
    print("# membisect: %s" % ", ".join(dirs))
    for cell in cells_in_order(recs):
        report(recs, cell, arm_order, cumulative, base)
        report_times(recs, cell, arm_order, base)


if __name__ == "__main__":
    main()
