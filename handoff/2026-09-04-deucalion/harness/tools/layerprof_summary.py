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

"""Collate LAYERPROF/LAYERPOP lines into a per-phase table, one block per cell.

    layerprof_summary.py [--baseline main] [--arm port] <dir> [dir ...]

Reads both filename shapes the harness produces: `<cell>_<arm>.log` and membisect's
`<cell>_rep<N>_<arm>.log`.

Every number here is a SUM over partitions -- partition-seconds, not wall seconds. That is the right
unit for a phase split (it is what the partitions actually spent) but it is NOT comparable to a
stopwatch: with S partitions running concurrently the wall is roughly layer_s/S. Both are printed so
neither can be quoted as the other.

`unattributed` is layer_s minus the phases. It is printed first among the derived rows on purpose: a
phase split that accounts for a third of the layer is a statement about the timers, not about the
workload, and it should be impossible to read the table without seeing that.

With several reps, ONE rep is reported -- the median by `layer_s` -- and the spread across reps is
printed beside it. Per-field medians across reps are deliberately not used: the fields would come from
different runs and would no longer sum to `layer_s`, which is exactly the invariant the
`unattributed` row exists to expose.

A caution that cost this project a whole attribution: the two arms must differ ONLY in the thing being
attributed. Pairing an instrumented baseline against an instrumented *superset* of the change under
test produces an exact table that answers the wrong question -- 69% of one regression landed in phases
the candidate did not contain. Check what each tree actually holds before quoting a row.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# Phases in the order they occur within a layer, so the table reads as the code runs.
PHASES = [
    "fold",
    "scan",
    "emit",
    "index",
    "resolve",
    "sendbuf",
    "exchange",
    "exchangeresp",
    "incoming",
    "insert",
    "apply",
    "contract",
]
COUNTERS = ["gates", "anti", "foll", "emit", "atol", "reject", "push", "hit", "miss", "qbytes",
            "words", "livewords"]
# Everything outside build_layer. Printed as its own table because the two have different
# denominators: layer_s is a phase of a gate, total_s is the whole propagate.
OUTER = ["gate", "contract", "cacheop", "cachestate", "cacheshrink"]

FIELD_RE = re.compile(r"(\w+)=([-\d.]+)")
LOG_RE = re.compile(r"^(?P<cell>[A-Za-z0-9]+)(?:_rep(?P<rep>\d+))?_(?P<arm>.+)$")


def parse(path: Path) -> tuple[dict[str, float], int, list[dict]]:
    """Return (summed fields, partition-line count, LAYERPOP samples)."""
    total: dict[str, float] = defaultdict(float)
    parts = 0
    pops: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("LAYERPROF "):
            parts += 1
            for key, value in FIELD_RE.findall(line):
                if key != "part":
                    total[key] += float(value)
        elif line.startswith("LAYERPOP "):
            rec = {k: v for k, v in FIELD_RE.findall(line)}
            hists = re.findall(r"(\w+_hist)=([\d,]+)", line)
            for name, body in hists:
                rec[name] = [int(x) for x in body.split(",") if x != ""]
            pops.append(rec)
    return total, parts, pops


def pick_median_rep(samples: list[tuple[dict, int, list]]) -> tuple[tuple[dict, int, list], str]:
    """Return (the median-by-layer_s sample, a note on the spread across reps)."""
    if len(samples) == 1:
        return samples[0], ""
    ordered = sorted(samples, key=lambda s: s[0].get("layer_s", 0.0))
    chosen = ordered[len(ordered) // 2]
    lows, highs = ordered[0][0].get("layer_s", 0.0), ordered[-1][0].get("layer_s", 0.0)
    return chosen, " (median of %d reps; layer_s spanned %.2f-%.2f part-s)" % (
        len(samples), lows, highs)


def ratio(port: float, main: float) -> str:
    if main <= 0:
        return "--"
    return f"{port / main:.3f}x"


def emit_cell(name: str, arms: dict[str, tuple[dict, int, list]], base: str, other: str,
              notes: dict[str, str]) -> None:
    if base not in arms or other not in arms:
        print(f"\n## {name}\n\n  incomplete: have {sorted(arms)}, need {base} and {other}")
        return
    (m, m_parts, m_pops) = arms[base]
    (p, p_parts, _) = arms[other]
    print(f"\n## {name}\n")
    print(f"Partitions reporting: {base} {m_parts}, {other} {p_parts}. Gates: {int(m['gates'])}.")
    for arm in (base, other):
        if notes.get(arm):
            print(f"  {arm}{notes[arm]}")

    # The counters must agree exactly or the two arms did different work and no timing below means
    # anything. This is the same rule the A/B harness applies to term counts.
    disagree = [c for c in ("anti", "emit", "atol", "reject", "push") if m[c] != p[c]]
    if disagree:
        print(f"\n**ARMS DID DIFFERENT WORK** -- {', '.join(disagree)} differ. Timings below are void.")
        for c in disagree:
            print(f"  {c}: {base} {int(m[c])} {other} {int(p[c])}")
    else:
        print("\nCounters agree exactly across arms (anti, emit, atol, reject, push).")

    print(f"\n| phase | {base} part-s | {other} part-s | {other}/{base} | {base} share | {other} share |")
    print("|---|---:|---:|---:|---:|---:|")
    m_layer = m["layer_s"] or 1.0
    p_layer = p["layer_s"] or 1.0
    m_sum = p_sum = 0.0
    for ph in PHASES:
        key = f"{ph}_s"
        if key not in m and key not in p:
            continue
        mv, pv = m.get(key, 0.0), p.get(key, 0.0)
        m_sum += mv
        p_sum += pv
        print(
            f"| {ph} | {mv:.2f} | {pv:.2f} | {ratio(pv, mv)} "
            f"| {100 * mv / m_layer:.1f}% | {100 * pv / p_layer:.1f}% |"
        )
    print(
        f"| **unattributed** | {m['layer_s'] - m_sum:.2f} | {p['layer_s'] - p_sum:.2f} "
        f"| {ratio(p['layer_s'] - p_sum, m['layer_s'] - m_sum)} "
        f"| {100 * (m_layer - m_sum) / m_layer:.1f}% | {100 * (p_layer - p_sum) / p_layer:.1f}% |"
    )
    print(f"| **layer** | {m['layer_s']:.2f} | {p['layer_s']:.2f} | {ratio(p['layer_s'], m['layer_s'])} | | |")
    if m_parts:
        print(
            f"\nPer-partition wall equivalent (layer_s / partitions): "
            f"{base} {m['layer_s'] / m_parts:.2f} s, {other} {p['layer_s'] / p_parts:.2f} s."
        )

    if "total_s" in m:
        print("\n### Outside build_layer\n")
        print(f"| region | {base} part-s | {other} part-s | {other}/{base} | share of total |")
        print("|---|---:|---:|---:|---:|")
        m_tot = m["total_s"] or 1.0
        for r in OUTER:
            k = f"{r}_s"
            if k in m or k in p:
                mv, pv = m.get(k, 0.0), p.get(k, 0.0)
                print(f"| {r} | {mv:.2f} | {pv:.2f} | {ratio(pv, mv)} | {100 * mv / m_tot:.1f}% |")
        print(
            f"| **layer (of which)** | {m['layer_s']:.2f} | {p['layer_s']:.2f} "
            f"| {ratio(p['layer_s'], m['layer_s'])} | {100 * m['layer_s'] / m_tot:.1f}% |"
        )
        gate_rem_m = m.get("gate_s", 0.0) - m["layer_s"] - m.get("apply_s", 0.0)
        gate_rem_p = p.get("gate_s", 0.0) - p["layer_s"] - p.get("apply_s", 0.0)
        print(
            f"| **per-gate remainder** | {gate_rem_m:.2f} | {gate_rem_p:.2f} "
            f"| {ratio(gate_rem_p, gate_rem_m)} | {100 * gate_rem_m / m_tot:.1f}% |"
        )
        print(f"| **total** | {m['total_s']:.2f} | {p['total_s']:.2f} | {ratio(p['total_s'], m['total_s'])} | |")
        print(
            f"\nPer-partition wall equivalent of total: {base} {m['total_s'] / max(m_parts, 1):.2f} s, "
            f"{other} {p['total_s'] / max(p_parts, 1):.2f} s."
        )

    print(f"\n| counter | {base} | {other} |")
    print("|---|---:|---:|")
    for c in COUNTERS:
        if c in m or c in p:
            print(f"| {c} | {int(m.get(c, 0)):,} | {int(p.get(c, 0)):,} |")
    if m.get("words"):
        print(
            f"\nWord liveness: {int(m['livewords']):,} of {int(m['words']):,} scanned 64-term words "
            f"held at least one term surviving the atol gate (**{100 * m['livewords'] / m['words']:.1f}%**). "
            f"Below ~50% the operator could be scanned sparsely; near 100% the survivors are scattered "
            f"and no row ordering will help."
        )
    if m.get("emit"):
        print(
            f"\nReject split: {100 * m['atol'] / max(m['anti'], 1):.1f}% of anticommuting terms refused "
            f"by atol before any row is read; {100 * m['reject'] / max(m['emit'], 1):.1f}% of the "
            f"survivors refused by the structural cutoff AFTER the partner row was materialised."
        )

    # Last population sample: the k distribution and the paired fraction.
    if m_pops:
        last_gate = max(int(r["gate"]) for r in m_pops)
        tail = [r for r in m_pops if int(r["gate"]) == last_gate]
        rows = sum(int(r["rows"]) for r in tail)
        paired = sum(int(r["paired"]) for r in tail)
        overflow = sum(int(r["overflow"]) for r in tail)
        khist: dict[int, int] = defaultdict(int)
        for r in tail:
            for k, n in enumerate(r.get("k_hist", [])):
                khist[k] += n
        mean_k = sum(k * n for k, n in khist.items()) / max(rows, 1)
        top = sorted(khist.items(), key=lambda kv: -kv[1])[:8]
        print(
            f"\nPopulation at gate {last_gate} ({len(tail)} partitions, {rows:,} rows): "
            f"**fully paired {100 * paired / max(rows, 1):.1f}%**, overflow {overflow:,}, "
            f"mean k {mean_k:.2f}, width {tail[0].get('width')}."
        )
        print("  k histogram (top): " + ", ".join(f"k={k}:{100 * n / max(rows, 1):.1f}%" for k, n in top if n))


def main() -> None:
    args = sys.argv[1:]
    base, other, dirs = "main", "port", []
    while args:
        a = args.pop(0)
        if a == "--baseline":
            base = args.pop(0)
        elif a == "--arm":
            other = args.pop(0)
        elif a.startswith("--"):
            raise SystemExit(f"unknown option {a}\n\n{__doc__}")
        else:
            dirs.append(a)
    if not dirs:
        raise SystemExit(__doc__)

    # cell -> arm -> [sample per rep]
    reps: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        for path in sorted(Path(d).glob("*.log")):
            m = LOG_RE.match(path.stem)
            if not m or m.group("arm") not in (base, other):
                continue
            sample = parse(path)
            if sample[1]:  # at least one LAYERPROF line, else the instrument never fired
                reps[m.group("cell")][m.group("arm")].append(sample)

    print("# LAYERPROF phase attribution\n")
    print(
        "All phase figures are PARTITION-SECONDS summed over every reporting partition, not wall "
        "clock. Read `unattributed` before any ratio: it bounds what the split can explain."
    )
    if not reps:
        print(
            f"\nNo LAYERPROF lines found for arms {base!r}/{other!r}. The instrument is opt-in: "
            "the run needs monoprop_LAYER_PROFILE=1, and the arm must be a build that carries it."
        )
        return
    for name in sorted(reps):
        arms, notes = {}, {}
        for arm, samples in reps[name].items():
            arms[arm], notes[arm] = pick_median_rep(samples)
        emit_cell(name, arms, base, other, notes)


if __name__ == "__main__":
    main()
