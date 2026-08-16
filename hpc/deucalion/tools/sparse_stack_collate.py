"""Sparse slot records measured on top of the derived exchange layout (#238 stacked on #237).

Different question from sparse_collate.py, and a different baseline. There the arm was `main` and
the headline was a job-total ratio. Here the arm is perf/graph-world-size, and the claim is about
SCALING: per-rank graph memory must go DOWN as the world grows. It did not before -- #238 alone
left it improving to P=256 and then reversing, because the retained exchange layout is 8*L*P per
rank and grows faster than the payload falls.

Two things are asserted, both before the cells ran:

  * the accounting identity, exactly. Rather than hardcode record widths, it is stated over
    measured fields only, so it cannot drift when a struct changes:

        total drop  ==  (slot record bytes freed)  -  (LayerCore growth)

    Widths are checked separately, as the MODEL: 16 B per dense slot on the arm, 12 B per occupied
    slot on the port. If the identity misses by a byte the ratio is not quotable, however good.

  * per-rank monotonicity. total_bytes / P must strictly decrease across P = 128, 256, 512 on the
    port. That is the whole point of the stack and the one assertion that fails if it did not work.

Collates from an EXPLICIT dir list -- $PROJ/runs is shared and a glob has swept another session's
cells into a table here before.
"""

import json
import os
import pathlib
import sys

RUNS = pathlib.Path("/projects/EEHPC-DEV-2026D08-260/runs")
GIB = 1024.0**3
MIB = 1024.0**2
LAYERS = 5420  # 20 x (127 X + 144 heavy-hex ZZ), recovered independently by the instrument

ARM_RECORD = 16  # sizeof(CrossRankPartnerRange) on perf/graph-world-size: offset + 2 counts
PORT_RECORD = 12  # sizeof(CrossRankOccupiedSlot): slot id + 2 counts, offset derived

TAG = os.environ.get("SPARSE_TAG", "sp3")
# (label, flat world P, run dir)
CELLS = [
    ("c12 1x16", 16, f"models-pauli-{TAG}-c12-g1x16-N1"),
    ("c12 1x128", 128, f"models-pauli-{TAG}-c12-g1x128-N1"),
    ("c14 8x16 N1", 128, f"models-pauli-{TAG}-c14-g8x16-N1"),
    ("c14 8x16 N2", 256, f"models-pauli-{TAG}-c14-g8x16-N2"),
    ("c14 8x16 N4", 512, f"models-pauli-{TAG}-c14-g8x16-N4"),
]

FIELDS = ("total_bytes", "cross_rank_bytes", "exchange_layout_bytes", "layer_storage_object_bytes",
          "layer_descriptor_bytes", "d_slot_record_bytes", "d_layer_cores", "d_slot_records",
          "d_occupied_slots", "d_cross_rank_endpoints")


def collect(d):
    """arm -> {field: [values]} from the graphmembreak block, plus rep counts, terms and md5s."""
    out = {"main": {}, "port": {}}
    reps = {"main": 0, "port": 0}
    terms = {"main": set(), "port": set()}
    md5s = {"main": set(), "port": set()}
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
        meta = blob.get("meta") or {}
        if meta.get("monoprop_core_md5"):
            md5s[arm].add(meta["monoprop_core_md5"])
        for rec in (blob.get("opsize") or {}).values():
            if isinstance(rec, dict) and "terms" in rec:
                terms[arm].add(rec["terms"])
        for rec in (blob.get("graphmembreak") or {}).values():
            for field, val in rec.items():
                out[arm].setdefault(field, []).append(val)
    return out, reps, terms, md5s


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


failures = []
rows = []
print("=" * 100)
print("SPARSE SLOT RECORDS ON THE DERIVED EXCHANGE LAYOUT  (#238 stacked on #237)")
print("=" * 100)

for label, world, dirname in CELLS:
    d = RUNS / dirname
    print(f"\n## {label}  (P={world})")
    if not d.is_dir():
        print(f"   MISSING {dirname} -- cell not run")
        failures.append(f"{label}: {dirname} missing")
        continue

    data, reps, terms, md5s = collect(d)
    print(f"   reps: main {reps['main']}  port {reps['port']}   terms {sorted(terms['port'])}")

    got = {arm: {f: med(data[arm].get(f, [])) for f in FIELDS} for arm in ("main", "port")}

    if got["main"]["total_bytes"] is None or got["port"]["total_bytes"] is None:
        print("   !! REFUSED: an arm reports no graphmembreak")
        failures.append(f"{label}: an arm has no graphmembreak block")
        continue

    # Identity is the binary, never the version stamp.
    if md5s["main"] and md5s["port"] and (md5s["main"] & md5s["port"]):
        failures.append(f"{label}: both arms ran the same binary {sorted(md5s['main'] & md5s['port'])}")
    if terms["main"] and terms["port"] and terms["main"] != terms["port"]:
        failures.append(f"{label}: term counts differ {terms['main']} vs {terms['port']}")
        continue

    cores = got["main"]["d_layer_cores"]
    slots = got["main"]["d_slot_records"]
    occ_main = got["main"]["d_occupied_slots"]
    occ_port = got["port"]["d_occupied_slots"]

    # The exchange layout must already be gone on BOTH arms -- #237 is the baseline here, not main.
    for arm in ("main", "port"):
        if got[arm]["exchange_layout_bytes"]:
            failures.append(f"{label}: {arm} still retains {got[arm]['exchange_layout_bytes']} B of "
                            f"exchange layout -- the baseline is supposed to be #237, not main")

    # Geometry, recovered from the instrument rather than trusted from the config.
    if cores and abs(slots / cores - world) > 1e-9:
        failures.append(f"{label}: slots/cores = {slots / cores:.1f}, expected P={world}")
    if cores and abs(cores / LAYERS - world) > 1e-9:
        failures.append(f"{label}: cores/L = {cores / LAYERS:.1f}, expected P={world}")

    # Occupancy is a property of the traffic; the format must not move it.
    if occ_main != occ_port:
        failures.append(f"{label}: occupied moved with the format, {occ_main} -> {occ_port}")

    # THE accounting identity, over measured fields only.
    freed = got["main"]["d_slot_record_bytes"] - got["port"]["d_slot_record_bytes"]
    core_growth = got["port"]["layer_storage_object_bytes"] - got["main"]["layer_storage_object_bytes"]
    drop = got["main"]["total_bytes"] - got["port"]["total_bytes"]
    predicted = freed - core_growth

    print(f"   graph  main {got['main']['total_bytes'] / GIB:9.3f} GiB"
          f"   port {got['port']['total_bytes'] / GIB:9.3f} GiB"
          f"   -> {got['main']['total_bytes'] / got['port']['total_bytes']:.2f}x")
    print(f"   slot records  {got['main']['d_slot_record_bytes'] / GIB:8.3f} -> "
          f"{got['port']['d_slot_record_bytes'] / GIB:7.3f} GiB   "
          f"({got['main']['d_slot_record_bytes'] / max(got['port']['d_slot_record_bytes'], 1):.2f}x)")
    print(f"   drop measured {drop / GIB:9.3f} GiB   predicted {predicted / GIB:9.3f} GiB"
          f"   (freed {freed / GIB:.3f} - core growth {core_growth / GIB:.3f})")
    if drop != predicted:
        failures.append(f"{label}: drop off by {drop - predicted} B ({drop} vs {predicted}) -- "
                        f"the model of what moved is wrong")

    # The widths, checked as a separate claim from the identity above.
    if slots and abs(got["main"]["d_slot_record_bytes"] / slots - ARM_RECORD) > 1e-6:
        failures.append(f"{label}: arm record {got['main']['d_slot_record_bytes'] / slots:.2f} B/slot, "
                        f"expected {ARM_RECORD}")
    if occ_port and abs(got["port"]["d_slot_record_bytes"] / occ_port - PORT_RECORD) > 1e-6:
        failures.append(f"{label}: port record {got['port']['d_slot_record_bytes'] / occ_port:.2f} "
                        f"B/occupied, expected {PORT_RECORD}")

    per_rank_main = got["main"]["total_bytes"] / world / MIB
    per_rank_port = got["port"]["total_bytes"] / world / MIB
    occupancy = occ_port / slots
    print(f"   per-rank  main {per_rank_main:7.2f} MiB   port {per_rank_port:7.2f} MiB"
          f"   occupancy {occupancy:7.3%}")
    rows.append((label, world, per_rank_main, per_rank_port, occupancy,
                 got["port"]["d_cross_rank_endpoints"]))

# The claim the whole stack exists to make.
ladder = [r for r in rows if r[0].startswith("c14")]
if ladder:
    print()
    print("=" * 100)
    print("PER-RANK GRAPH MEMORY vs P -- the c14 8x16 ladder, only the node count moving")
    print("=" * 100)
    print(f"{'cell':<14}{'P':>6}{'#237 MiB':>12}{'+#238 MiB':>12}{'ratio':>9}{'occupancy':>12}")
    for label, world, pm, pp, occ, _ in ladder:
        print(f"{label:<14}{world:>6}{pm:>12.2f}{pp:>12.2f}{pm / pp:>8.2f}x{occ:>12.2%}")
    for arm_idx, name in ((2, "#237 alone"), (3, "with #238")):
        seq = [r[arm_idx] for r in ladder]
        deltas = [f"x{b / a:.2f}" for a, b in zip(seq, seq[1:])]
        falling = all(b < a for a, b in zip(seq, seq[1:]))
        print(f"\n{name:<12} per doubling: {' '.join(deltas)}   "
              f"{'MONOTONICALLY DOWN' if falling else 'NOT monotonic -- adding nodes stops helping'}")
        if name == "with #238" and not falling:
            failures.append("per-rank graph memory does not fall monotonically in P on the port arm "
                            "-- the stack did not restore scaling, which is the whole claim")

    endpoints = {r[5] for r in ladder if r[5] is not None}
    if len(endpoints) > 1:
        spread = (max(endpoints) - min(endpoints)) / max(endpoints)
        print(f"\nendpoints across P vary by {spread:.4%} ({min(endpoints):,} .. {max(endpoints):,})")
        if spread > 0.01:
            failures.append(f"endpoints moved {spread:.2%} across P -- they are a property of the operator")

print()
if failures:
    print("=" * 100)
    print("PREDICTION FAILED -- do not quote the tables above without explaining these")
    print("=" * 100)
    for f in failures:
        print(f"- {f}")
    sys.exit(1)
print("All predictions held: the accounting identity is exact, occupancy is unchanged by the")
print("format, and per-rank graph memory now falls monotonically as the world grows.")
