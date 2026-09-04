"""Compare two golden.py dumps cell by cell: term counts, term sets, coefficient/expval/gradient ULP.

Usage: python golden_diff.py <a.json> <b.json>
"""
from __future__ import annotations

import json
import struct
import sys
from collections import defaultdict


def ulp(a: float, b: float) -> int:
    if a == b:
        return 0
    if (a < 0) != (b < 0):
        return 1 << 62
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    return abs(ia - ib)


A = json.load(open(sys.argv[1]))
B = json.load(open(sys.argv[2]))
print("A core_md5:", A["core_md5"])
print("B core_md5:", B["core_md5"])
per_cfg: dict[str, dict] = defaultdict(lambda: {"cells": 0, "size_mismatch": 0, "term_set_mismatch": 0, "max_ulp_coeff": 0, "max_ulp_expval": 0, "max_ulp_grad": 0, "graph_size_mismatch": 0, "coeff_ne": 0, "coeffs": 0})
missing = set(A["cases"]) ^ set(B["cases"])
if missing:
    print("cells only in one file:", sorted(missing))
worst_cells = []
for key in sorted(set(A["cases"]) & set(B["cases"])):
    cfg = key.split("/", 1)[1]
    a, b = A["cases"][key], B["cases"][key]
    s = per_cfg[cfg]
    s["cells"] += 1
    if a["propagate"]["size"] != b["propagate"]["size"] or a["graph"]["size"] != b["graph"]["size"]:
        s["size_mismatch"] += 1
        print(f"SIZE {key}: propagate {a['propagate']['size']} vs {b['propagate']['size']}, graph {a['graph']['size']} vs {b['graph']['size']}")
    if a["graph"]["graph_size"] != b["graph"]["graph_size"]:
        s["graph_size_mismatch"] += 1
        print(f"GRAPH_SIZE {key}: {a['graph']['graph_size']} vs {b['graph']['graph_size']}")
    ta = {tuple(t[0]): (float(t[1]), float(t[2])) for t in a["propagate"]["terms"]}
    tb = {tuple(t[0]): (float(t[1]), float(t[2])) for t in b["propagate"]["terms"]}
    if set(ta) != set(tb):
        s["term_set_mismatch"] += 1
        print(f"TERMS {key}: {len(set(ta) - set(tb))} only in A, {len(set(tb) - set(ta))} only in B")
    cell_max = 0
    for k in set(ta) & set(tb):
        u = max(ulp(ta[k][0], tb[k][0]), ulp(ta[k][1], tb[k][1]))
        s["coeffs"] += 1
        if u:
            s["coeff_ne"] += 1
        cell_max = max(cell_max, u)
    s["max_ulp_coeff"] = max(s["max_ulp_coeff"], cell_max)
    ue = ulp(float(a["propagate"]["expval"]), float(b["propagate"]["expval"]))
    ug = ulp(float(a["graph"]["expval"]), float(b["graph"]["expval"]))
    s["max_ulp_expval"] = max(s["max_ulp_expval"], ue, ug)
    grad = max((ulp(float(x), float(y)) for x, y in zip(a["graph"]["grad"], b["graph"]["grad"])), default=0)
    if len(a["graph"]["grad"]) != len(b["graph"]["grad"]):
        grad = 1 << 62
    s["max_ulp_grad"] = max(s["max_ulp_grad"], grad)
    worst_cells.append((cell_max, ue, ug, grad, key))
print()
print(f"{'config':24} {'cells':>5} {'size!=':>6} {'terms!=':>7} {'gsize!=':>7} {'coeffs':>7} {'coeff!=':>7} {'maxULP coeff':>12} {'maxULP expval':>13} {'maxULP grad':>11}")
for cfg, s in sorted(per_cfg.items()):
    print(f"{cfg:24} {s['cells']:5} {s['size_mismatch']:6} {s['term_set_mismatch']:7} {s['graph_size_mismatch']:7} {s['coeffs']:7} {s['coeff_ne']:7} {s['max_ulp_coeff']:12} {s['max_ulp_expval']:13} {s['max_ulp_grad']:11}")
print()
print("worst cells (coeff ULP, expval ULP propagate/graph, grad ULP):")
for row in sorted(worst_cells, reverse=True)[:12]:
    print("  ", row)
