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

"""Paired N-arm summary for ab-hashfree-n1.sh and ab-hashfree-nodes.sh cells.

Every non-reference arm is paired against the reference arm (the first arm recorded in
CELL-META.tsv, or --ref). Per (layout, model, group) row and per (reference, arm) pair: median of
the PER-REP paired ratios for wall time (the benchmark's own timing), peak RSS (PEAK-RSS.tsv) and
the engine's operator ledger (opmembreak), the agreement count with a two-sided sign test, and a
term-count check that refuses to call the arms comparable when they did different work. A compact
row x arm table closes the report.

PEAK-RSS.tsv holds one number per cell whose meaning depends on the launcher, so it is labelled
from CELL-META.tsv's ``ranks_per_node``. At one rank per node it is the kernel high-water mark of
the single process (/usr/bin/time -v, ab-hashfree-n1.sh). Under srun with several ranks per node
there is no such process -- the launcher's own RSS is meaningless -- so ab-hashfree-nodes.sh writes
the bench JSON's ``memhwm``, the ranks' VmHWM summed, and the row says so. The sum charges shared
pages to every rank, so it bounds a node rather than describing one process; the same run's
worst-rank ``memhwm_max`` is kept beside it in WORST-RANK-RSS.tsv.

CELL-META.tsv is read for arm identity and order: the current ``arm_<n>_name`` keys (n = 1..N, arm
1 is the reference), or the legacy two-arm ``main_venv``/``port_venv`` keys (arms ["main", "port"],
"main" the reference) written by older runs.

  python ab_pairs.py <results-dir> [--ref NAME]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Layout is "<rung>_<ranks>x<partitions>", with an optional "<nodes>n_" between them for the
# multi-node rungs (A_1x1, C_1x128, B_8x16, C_4n_8x16).
LABEL_RE = re.compile(
    r"^N(?P<nodes>\d+)_(?P<layout>[A-Z]_(?:\d+n_)?\d+x\d+)_(?P<model>[a-z0-9]+)"
    r"_(?P<group>fresh|graph)_(?P<side>[A-Za-z0-9]+)_r(?P<rep>\d+)$"
)


def sign_test_p(k: int, n: int) -> float:
    """Two-sided sign test: probability of a split at least as lopsided as k of n."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def read_meta(meta: Path) -> dict[str, str]:
    """Return CELL-META.tsv as a key -> value dict (empty when the file is missing)."""
    if not meta.exists():
        return {}
    kv = {}
    for line in meta.read_text().splitlines()[1:]:
        if "\t" not in line:
            continue
        k, v = line.split("\t", 1)
        kv[k] = v
    return kv


def rss_row_label(kv: dict[str, str]) -> str:
    """Return the PEAK-RSS.tsv row's name for this run's layout.

    Multi-rank cells run under ``srun``, where no single process's high-water mark describes the
    cell; ab-hashfree-nodes.sh records the ranks' summed VmHWM instead, so say so rather than
    let it read as one process's kernel peak.
    """
    try:
        ranks_per_node = int(kv.get("ranks_per_node", "1"))
    except ValueError:
        ranks_per_node = 1
    return "peak RSS (ranks summed)" if ranks_per_node > 1 else "peak RSS (kernel)"


def read_arms(meta: Path) -> list[str]:
    """Return arm names in CELL-META.tsv order, reference (arm 1) first.

    Falls back to the legacy ``main_venv``/``port_venv`` two-arm format when no ``arm_<n>_name``
    keys are present. Returns [] if the file is missing or carries neither shape.
    """
    kv = read_meta(meta)
    if not kv:
        return []
    idx = {}
    for k, v in kv.items():
        m = re.match(r"^arm_(\d+)_name$", k)
        if m:
            idx[int(m.group(1))] = v
    if idx:
        return [idx[i] for i in sorted(idx)]
    if "main_venv" in kv and "port_venv" in kv:
        return ["main", "port"]
    return []


def load(results: Path) -> tuple[dict, int]:
    """Return the parsed cells and the number of cells skipped for an unusable time JSON.

    A cell whose pytest run failed leaves an empty (or missing) ``time-<label>.json``; such a cell
    is dropped with one warning line on stderr so the rest of the run still pairs up.
    """
    cells: dict[tuple, dict] = {}
    skipped = 0
    for tf in sorted(results.glob("time-*.json")):
        m = LABEL_RE.match(tf.stem.removeprefix("time-"))
        if not m:
            continue
        key = (m["layout"], m["model"], m["group"], m["side"], int(m["rep"]))
        cell = {"ops": {}, "rss_kb": None, "ledger": None, "terms": None}
        try:
            text = tf.read_text()
            data = json.loads(text) if text.strip() else None
        except (OSError, json.JSONDecodeError) as exc:
            data = None
            reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = "empty time JSON (pytest did not write results)"
        if not isinstance(data, dict):
            print(f"!! skipping cell {tf.stem.removeprefix('time-')}: {reason}", file=sys.stderr)
            skipped += 1
            continue
        for b in data.get("benchmarks", []):
            op = b.get("fullname") or b.get("name")
            cell["ops"][op] = b["stats"]["median"]
        mf = results / f"{tf.stem.removeprefix('time-')}.json"
        if mf.exists():
            mem = json.loads(mf.read_text())
            brk = mem.get("opmembreak", {})
            if brk:
                first = next(iter(brk.values()))
                ledger = {k: v for k, v in first.items() if not k.startswith("d_")}
                ledger.setdefault("total_bytes", sum(v for k, v in ledger.items() if k.endswith("_bytes")))
                cell["ledger"] = ledger
            terms = []
            def walk(x):
                if isinstance(x, dict):
                    if "terms" in x and isinstance(x["terms"], int):
                        terms.append(x["terms"])
                    for v in x.values():
                        walk(v)
            walk(mem.get("opsize", {}))
            cell["terms"] = tuple(sorted(terms))
        cells[key] = cell
    rss = results / "PEAK-RSS.tsv"
    if rss.exists():
        for line in rss.read_text().splitlines()[1:]:
            label, kb, _wall = line.split("\t")
            m = LABEL_RE.match(label)
            if m and kb != "NA":
                k = (m["layout"], m["model"], m["group"], m["side"], int(m["rep"]))
                if k in cells:
                    cells[k]["rss_kb"] = int(kb)
    return cells, skipped


def emit_pair(pairs: dict, ref: str, arm: str, rss_label: str = "peak RSS (kernel)") -> tuple[int, dict]:
    """Print one (ref, arm) comparison table for a paired-reps dict; return (bad_count, metrics).

    ``metrics`` carries the representative ratios used by the compact summary table: "time" (the
    first op alphabetically), "rss" (peak RSS, named by ``rss_label``) and "bpt" (ledger
    bytes/term, when the row's term count is a single value).
    """
    bad = 0
    for r, s in pairs.items():
        if s[ref]["terms"] != s[arm]["terms"]:
            print(f"!! rep {r}: term counts differ {ref}={s[ref]['terms']} {arm}={s[arm]['terms']}")
            bad += 1
    any_pair = next(iter(pairs.values()))
    terms = any_pair[arm]["terms"]
    print(f"terms: {terms}\n")
    print(f"| metric | {ref} median | {arm} median | {arm}/{ref} (paired median) | agree | p |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")

    metrics: dict = {}

    def row(name, get, fmt, key=None):
        refs, arms_, ratios = [], [], []
        for r, s in sorted(pairs.items()):
            a, b = get(s[ref]), get(s[arm])
            if a is not None:
                refs.append(a)
            if b is not None:
                arms_.append(b)
            if a is None or b is None or a == 0:
                continue
            ratios.append(b / a)
        if not ratios:
            # A ledger field one arm has and another lacks (scratch buffers named differently) is
            # still worth seeing; only a field neither arm reports is dropped.
            if refs or arms_:
                fa = fmt(statistics.median(refs)) if refs else "—"
                fb = fmt(statistics.median(arms_)) if arms_ else "—"
                print(f"| {name} | {fa} | {fb} | — | — | — |")
            return
        med = statistics.median(ratios)
        agree = sum(1 for x in ratios if (x < 1) == (med < 1)) if abs(med - 1) > 1e-9 else len(ratios)
        p = sign_test_p(sum(1 for x in ratios if x < 1), len(ratios))
        tag = " **" if p < 0.05 and abs(med - 1) > 0.01 else ""
        print(f"| {name} | {fmt(statistics.median(refs))} | {fmt(statistics.median(arms_))} | {med:.3f}{tag} | {agree}/{len(ratios)} | {p:.3f} |")
        if key is not None:
            metrics[key] = med

    ops = sorted({op for s in pairs.values() for c in s.values() for op in c["ops"]})
    for op in ops:
        short = op.rsplit("::", 1)[-1]
        row(f"time {short}", lambda c, op=op: c["ops"].get(op), lambda v: f"{v:.3f} s", key=("time", op))
    row(rss_label, lambda c: c["rss_kb"], lambda v: f"{v/1024**2:.3f} GiB", key="rss")
    row("operator ledger total", lambda c: (c["ledger"] or {}).get("total_bytes"), lambda v: f"{v/1024**3:.3f} GiB")
    for field in ("operator_terms_bytes", "indexing_bytes", "inverted_index_bytes", "op_coeffs_bytes", "row_keys_bytes", "gate_scratch_bytes", "matched_scratch_bytes"):
        row(f"ledger {field}", lambda c, f=field: (c["ledger"] or {}).get(f), lambda v: f"{v/1024**2:.1f} MiB")
    # The per-gate scratch is `matched_scratch_bytes` on one arm and `gate_scratch_bytes` on
    # another: same role, so pair them under one name to get a ratio.
    row("ledger scratch (matched|gate)",
        lambda c: next((v for k in ("gate_scratch_bytes", "matched_scratch_bytes") if (v := (c["ledger"] or {}).get(k)) is not None), None),
        lambda v: f"{v/1024**2:.1f} MiB")
    if terms and len(terms) == 1:
        n = terms[0]
        row("ledger bytes/term", lambda c: (c["ledger"] or {}).get("total_bytes"), lambda v: f"{v/n:.2f} B/term", key="bpt")
        bpt_label = "peak bytes/term (ranks summed)" if "summed" in rss_label else "kernel peak bytes/term"
        row(bpt_label, lambda c: c["rss_kb"], lambda v: f"{v*1024/n:.2f} B/term")
    print()

    time_keys = sorted(k for k in metrics if isinstance(k, tuple) and k[0] == "time")
    return bad, {
        "time": metrics[time_keys[0]] if time_keys else None,
        "rss": metrics.get("rss"),
        "bpt": metrics.get("bpt"),
    }


def main(results: Path, ref_arg: str | None = None) -> int:
    cells, skipped = load(results)
    meta = results / "CELL-META.tsv"
    rss_label = rss_row_label(read_meta(meta))
    arms = read_arms(meta)
    if not arms:
        arms = sorted({side for (*_rest, side, _rep) in cells})
    ref = ref_arg or (arms[0] if arms else "main")
    other_arms = [a for a in arms if a != ref]

    print(f"# A/B summary: {results.name}\n")
    if meta.exists():
        print("```\n" + meta.read_text().strip() + "\n```\n")
    print(f"cells: {len(cells)} loaded, {skipped} skipped (failed cells, see stderr)\n")

    groups = defaultdict(dict)
    for (layout, model, group, side, rep), cell in cells.items():
        groups[(layout, model, group)].setdefault(rep, {})[side] = cell

    bad = 0
    compact: list[tuple[str, str, dict]] = []
    for (layout, model, group), reps in sorted(groups.items()):
        present_arms = [a for a in other_arms if any(a in s for s in reps.values())]
        row_label = f"{layout} {model} {group}"
        if len(present_arms) <= 1:
            # Exactly the two-arm shape (or a row only one other arm ran): one table, same
            # heading and columns as the original two-arm report.
            arm = present_arms[0] if present_arms else (other_arms[0] if other_arms else None)
            pairs = {r: s for r, s in reps.items() if arm and ref in s and arm in s}
            print(f"## {row_label}  ({len(pairs)} paired reps)\n")
            if not pairs:
                continue
            b, metrics = emit_pair(pairs, ref, arm, rss_label)
            bad += b
            compact.append((row_label, arm, metrics))
        else:
            print(f"## {row_label}\n")
            for arm in present_arms:
                pairs = {r: s for r, s in reps.items() if ref in s and arm in s}
                print(f"### {ref} vs {arm}  ({len(pairs)} paired reps)\n")
                if not pairs:
                    continue
                b, metrics = emit_pair(pairs, ref, arm, rss_label)
                bad += b
                compact.append((row_label, arm, metrics))

    if compact:
        print("## Compact summary\n")
        rss_col = "RSS ratio (summed)" if "summed" in rss_label else "RSS ratio"
        print(f"| row | arm | time ratio | {rss_col} | bytes/term ratio |")
        print("| --- | --- | ---: | ---: | ---: |")

        def f(v):
            return f"{v:.3f}" if v is not None else "—"

        for row_label, arm, m in compact:
            print(f"| {row_label} | {arm} | {f(m['time'])} | {f(m['rss'])} | {f(m['bpt'])} |")
        print()

    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--ref", default=None, help="reference arm name (default: first arm in CELL-META.tsv)")
    args = ap.parse_args()
    sys.exit(main(args.results, args.ref))
