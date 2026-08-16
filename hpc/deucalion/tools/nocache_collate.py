"""Before/after for dropping the retained transpose cache, on THREE arms rather than two.

The usual A/B (main vs port) is not enough here, because the quantity that moved most is not
in the shipped metric at all. Three numbers are needed per cell:

    main            origin/main, the same arm the `layout` wave used
    port (gws2)     the retained-layouts build -- the previous port, read from its own run dir
    port (gws3)     this change

`graph` should separate main from gws2 by a lot and gws2 from gws3 by exactly 80 B per layer
core: sizeof(LayerCore) 248 -> 168. It should NOT move per slot. The 8 B/slot the transpose
cache did cost was never inside total_bytes, so it shows up only as d_recv_cache_bytes going to
zero, and in RSS.

Stating the prediction as an assertion rather than eyeballing the table afterwards: a number
that merely looks plausible is how a wrong model survives.

Collates from an EXPLICIT dir list -- $PROJ/runs is shared between sessions and a `models-*`
glob has swept another campaign's cells into a table here before now.
"""

import json
import pathlib
import sys

RUNS = pathlib.Path("/projects/EEHPC-DEV-2026D08-260/runs")
GIB = 1024.0**3
LAYERS = 5420  # 20 x (127 X + 144 heavy-hex ZZ), recovered independently by the instrument

# (label, flat world P, gws3 dir, gws2 dir for the same cell)
CELLS = [
    ("c12 1x128", 128, "models-pauli-gws3-c12-g1x128-N1", "models-pauli-gws2-c12-g1x128-N1"),
    ("c14 8x16 N4", 512, "models-pauli-gws3-c14-g8x16-N4", "models-pauli-gws2-c14-g8x16-N4"),
]


def collect(d):
    """arm -> {metric: [values]}, plus term counts and rep counts."""
    out = {"main": {}, "port": {}}
    terms = {"main": set(), "port": set()}
    reps = {"main": 0, "port": 0}
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("time-"):
            continue
        arm = "main" if "_main_" in f.name else "port" if "_port_" in f.name else None
        if arm is None:
            continue
        try:
            blob = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  !! unreadable {f.name}: {exc}")
            continue
        reps[arm] += 1
        for rec in (blob.get("opsize") or {}).values():
            if isinstance(rec, dict) and "terms" in rec:
                terms[arm].add(rec["terms"])
        for group in ("opbytes", "graphmembreak"):
            for rec in (blob.get(group) or {}).values():
                for field, val in rec.items():
                    out[arm].setdefault(f"{group}.{field}", []).append(val)
    return out, terms, reps


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


failures = []
print("=" * 88)
print("GRAPH BYTES: main -> retained layouts (gws2) -> derived transpose (gws3)")
print("=" * 88)

for label, world, new_dir, old_dir in CELLS:
    dn, do = RUNS / new_dir, RUNS / old_dir
    print(f"\n## {label}  (P={world})")
    if not dn.is_dir():
        print(f"   MISSING {new_dir} -- cell not run")
        failures.append(f"{label}: {new_dir} missing")
        continue

    new, new_terms, new_reps = collect(dn)
    print(f"   gws3 reps: main {new_reps['main']}  port {new_reps['port']}   terms {sorted(new_terms['port'])}")

    g_main = med(new["main"].get("opbytes.graph", []))
    g_new = med(new["port"].get("opbytes.graph", []))
    cores = med(new["port"].get("graphmembreak.d_layer_cores", []))
    slots = med(new["port"].get("graphmembreak.d_slot_records", []))

    g_old = None
    if do.is_dir():
        old, old_terms, _ = collect(do)
        g_old = med(old["port"].get("opbytes.graph", []))
        old_cache = med(old["port"].get("graphmembreak.d_recv_cache_bytes", []))
        # A memory comparison across run dirs is meaningless if the operators differ.
        if new_terms["port"] and old_terms["port"] and new_terms["port"] != old_terms["port"]:
            failures.append(f"{label}: gws2/gws3 term counts differ {old_terms['port']} vs {new_terms['port']}")
            print("   !! REFUSED: term counts differ between the two port arms")
    else:
        old_cache = None
        print(f"   (no gws2 dir {old_dir}: the increment cannot be shown, only main->gws3)")

    for name, val in (("main", g_main), ("port gws2", g_old), ("port gws3", g_new)):
        if val is not None:
            print(f"   graph {name:<10} {val / GIB:9.3f} GiB")

    if g_main and g_new:
        print(f"   -> main/gws3 = {g_main / g_new:.2f}x   saved {(g_main - g_new) / GIB:.3f} GiB")

    # The prediction: gws2 -> gws3 is 80 B per layer core, and nothing per slot.
    if g_old and g_new and cores:
        delta = g_old - g_new
        per_core = delta / cores
        print(f"   -> gws2/gws3 delta {delta / GIB:.3f} GiB = {per_core:.2f} B per layer core "
              f"(cores {cores:,.0f}, slots {slots:,.0f})")
        print(f"      predicted 80.00 B/core (sizeof(LayerCore) 248 -> 168); "
              f"ratio measured/predicted = {per_core / 80.0:.4f}")
        if abs(per_core - 80.0) > 0.5:
            failures.append(f"{label}: {per_core:.2f} B/core, predicted 80.00 -- the model of "
                            f"what total_bytes contains is wrong")
        # Slot count is unchanged by this commit, so any per-slot movement is unexplained.
        if slots and abs(delta - 80.0 * cores) / slots > 0.05:
            failures.append(f"{label}: {(delta - 80.0 * cores) / slots:.3f} B/slot unaccounted for")

    # The saving that total_bytes cannot see.
    new_cache = med(new["port"].get("graphmembreak.d_recv_cache_bytes", []))
    if old_cache is not None or new_cache is not None:
        print(f"   [uncounted] recv_cache  gws2 {(old_cache or 0) / GIB:8.3f} GiB"
              f"  ->  gws3 {(new_cache or 0) / GIB:8.3f} GiB")
        if old_cache and slots:
            print(f"      that is {old_cache / slots:.2f} B/slot removed, invisible to `graph`")
        if new_cache:
            failures.append(f"{label}: gws3 still reports {new_cache} B of recv cache -- it should be 0")

    if cores and slots:
        print(f"      check: cores/L = {cores / LAYERS:,.1f} (= P), slots/cores = {slots / cores:,.1f} (= P)")

print()
if failures:
    print("=" * 88)
    print("PREDICTION FAILED -- do not quote the table above without explaining these")
    print("=" * 88)
    for f in failures:
        print(f"- {f}")
    sys.exit(1)
print("All predictions held: 80 B/core counted, 8 B/slot uncounted and now zero.")
