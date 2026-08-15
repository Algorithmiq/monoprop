"""Before/after for the retained-exchange-layout change, on the ledger rather than RSS.

Reads the engine's own `opbytes`/`graphmembreak`, not peak RSS. RSS carries the allocator's
growth transient, which is counted by nothing; the ledger is what the change is sized against.

Collates from an EXPLICIT dir list. $PROJ/runs is shared between sessions and a `models-*`
glob has swept another campaign's cells into a table here before now.

Two fields carry the claim, and they are not the same kind of number:

  exchange_layout_bytes   counted inside total_bytes, so removing it MOVES the headline
  d_derivative_layout_bytes / d_recv_cache_bytes
                          diagnostics that total_bytes has never counted, so the derivative
                          layout's removal does NOT show up in `graph` at all

That second line is why this prints both. Reporting only the headline would understate the
change; reporting only the diagnostics would claim a saving the shipped metric cannot see.
And d_derivative_layout is 0 on both arms unless a GRADIENT ran in the cell -- it is lazily
allocated -- so a zero there is "not exercised", not "nothing was saved".
"""

import json
import pathlib
import sys

RUNS = pathlib.Path("/projects/EEHPC-DEV-2026D08-260/runs")
GIB = 1024.0**3

# (label, flat world P, dirname). The c12 four break P apart from the rank count; the c14
# three are the node ladder at fixed per-node geometry, where P is the only thing moving.
CELLS = [
    ("c12 1x16", 16, "models-pauli-gws2-c12-g1x16-N1"),
    ("c12 8x2", 16, "models-pauli-gws2-c12-g8x2-N1"),
    ("c12 1x128", 128, "models-pauli-gws2-c12-g1x128-N1"),
    ("c12 8x16", 128, "models-pauli-gws2-c12-g8x16-N1"),
    ("c14 8x16 N1", 128, "models-pauli-gws2-c14-g8x16-N1"),
    ("c14 8x16 N2", 256, "models-pauli-gws2-c14-g8x16-N2"),
    ("c14 8x16 N4", 512, "models-pauli-gws2-c14-g8x16-N4"),
]


def collect(d):
    """arm -> {metric: [values]} pooled over reps, plus term counts and versions seen."""
    out = {"main": {}, "port": {}}
    terms = {"main": set(), "port": set()}
    versions = {"main": set(), "port": set()}
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
        meta = blob.get("meta", {})
        if "monoprop_version" in meta:
            versions[arm].add(meta["monoprop_version"])
        for rec in (blob.get("opsize") or {}).values():
            if isinstance(rec, dict) and "terms" in rec:
                terms[arm].add(rec["terms"])
        for rec in (blob.get("opbytes") or {}).values():
            for field, val in rec.items():
                out[arm].setdefault(f"opbytes.{field}", []).append(val)
        for rec in (blob.get("graphmembreak") or {}).values():
            for field, val in rec.items():
                out[arm].setdefault(f"gmb.{field}", []).append(val)
    return out, terms, versions, reps


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def gib(x):
    return "-" if x is None else f"{x / GIB:8.3f}"


rows = []
refusals = []

print("=" * 82)
print("GRAPH BYTES, engine ledger, median over reps")
print("=" * 82)

for label, world, dirname in CELLS:
    d = RUNS / dirname
    if not d.is_dir():
        print(f"\n## {label}: MISSING {d.name}")
        continue
    data, terms, versions, reps = collect(d)
    print(f"\n## {label}  (P={world})  dir={dirname}")
    print(f"   reps: main {reps['main']}  port {reps['port']}")
    for arm in ("main", "port"):
        print(f"   {arm}: v{sorted(versions[arm]) or 'unrecorded'} terms {sorted(terms[arm])}")

    # A memory number is meaningless without knowing both arms walked the same operator.
    all_terms = terms["main"] | terms["port"]
    if len(all_terms) > 1:
        refusals.append(f"{label}: arms/reps disagree on term count {sorted(all_terms)}")
        print("   !! REFUSED: term counts differ")
    if versions["main"] and versions["main"] == versions["port"]:
        refusals.append(f"{label}: both arms are the same build {sorted(versions['main'])}")
        print("   !! REFUSED: both arms are the same build")

    g_main = med(data["main"].get("opbytes.graph", []))
    g_port = med(data["port"].get("opbytes.graph", []))
    o_main = med(data["main"].get("opbytes.operator", []))
    o_port = med(data["port"].get("opbytes.operator", []))

    if g_main and g_port:
        print(f"   graph    main {gib(g_main)} GiB  port {gib(g_port)} GiB"
              f"  port/main {g_port / g_main:.4f}  saved {(g_main - g_port) / GIB:.3f} GiB")
        rows.append((label, world, g_main, g_port))
    if o_main and o_port:
        print(f"   operator main {gib(o_main)} GiB  port {gib(o_port)} GiB"
              f"  port/main {o_port / o_main:.4f}")

    # The counted field this change zeroes, both arms, plus what it works out to per slot.
    for field in ("gmb.exchange_layout_bytes", "gmb.cross_rank_bytes", "gmb.total_bytes"):
        m, p = med(data["main"].get(field, [])), med(data["port"].get(field, []))
        if m is not None or p is not None:
            name = field.split(".")[1].replace("_bytes", "")
            print(f"   {name:<18} main {gib(m)} GiB  port {gib(p)} GiB")

    slots = med(data["port"].get("gmb.d_slot_records", []))
    if slots and g_main and g_port:
        print(f"   -> saved per slot record: {(g_main - g_port) / slots:.2f} B"
              f"   (slots {slots:,.0f})")
    occ = med(data["port"].get("gmb.d_occupied_slots", []))
    if slots and occ is not None:
        print(f"   [port] occupancy {100 * occ / slots:.2f}%")

    # Uncounted by total_bytes on BOTH arms, so state them separately rather than folding in.
    for k in ("gmb.d_recv_cache_bytes", "gmb.d_derivative_layout_bytes"):
        m, p = med(data["main"].get(k, [])), med(data["port"].get(k, []))
        if m is not None or p is not None:
            name = k.split(".")[1]
            note = "  (lazy: 0 unless a gradient ran)" if "derivative" in k else ""
            print(f"   [uncounted] {name:<26} main {gib(m)} GiB  port {gib(p)} GiB{note}")

print()
print("=" * 82)
print("SUMMARY: graph bytes before -> after")
print("=" * 82)
print(f"{'cell':<14}{'P':>5}{'main GiB':>11}{'port GiB':>11}{'saved GiB':>11}{'ratio':>9}")
for label, world, gm, gp in rows:
    print(f"{label:<14}{world:>5}{gm / GIB:>11.3f}{gp / GIB:>11.3f}"
          f"{(gm - gp) / GIB:>11.3f}{gm / gp:>8.2f}x")

# graph = a + b*P^2 on each adjacent pair, per arm. One pair is a solution, not a fit -- two
# unknowns from two points cannot fail -- so independent pairs agreeing is what makes it
# evidence. Fitted only where the TERM COUNT is fixed; across c12/c14 it is not.
for tag, keep in (("c12", lambda l: l.startswith("c12")), ("c14 ladder", lambda l: l.startswith("c14"))):
    pts = {}
    for label, world, gm, gp in rows:
        if keep(label):
            pts[world] = (gm, gp)
    if len(pts) < 2:
        continue
    print(f"\n## graph = a + b*P^2, {tag} (term count fixed within this group)")
    print(f"{'pair':<14}{'b_main':>14}{'b_port':>14}{'delta':>14}")
    ordered = sorted(pts.items())
    for (p1, (m1, o1)), (p2, (m2, o2)) in zip(ordered, ordered[1:]):
        bm = (m2 - m1) / (p2**2 - p1**2)
        bp = (o2 - o1) / (p2**2 - p1**2)
        print(f"({p1},{p2}){'':<6}{bm:>14,.0f}{bp:>14,.0f}{bm - bp:>14,.0f}")

if refusals:
    print("\n" + "=" * 82)
    print("REFUSED -- do not quote the numbers above")
    print("=" * 82)
    for r in refusals:
        print(f"- {r}")
    sys.exit(1)
