"""Before/after for storing only the world slots that carry traffic.

The claim is not "the graph got smaller". It is that the slot array stopped being a
function of the flat world size P at all, and became a function of the traffic -- which is
measured flat in P. So the table has to show two different things:

  * the byte law, asserted exactly. Baseline holds 32 B per world slot (main stores the D
    range twice); the port holds 12 B per OCCUPIED slot and pays 24 B more per layer core
    for the world size and the self-slot position:

        graph drop = 32*slots - 12*occupied - 24*cores

    If that misses by a byte, the model of what moved is wrong and the ratio below is not
    quotable, however good it looks.

  * occupancy against P. `occupied` must be IDENTICAL on both arms -- it is a property of
    the traffic, not of the format -- and it is the ceiling argument's whole substance:
    occupied <= endpoints, and endpoints does not depend on P.

Stating predictions as assertions rather than eyeballing the table afterwards: a number
that merely looks plausible is how a wrong model survives.

Collates from an EXPLICIT dir list -- $PROJ/runs is shared between sessions and a
`models-*` glob has swept another campaign's cells into a table here before now.
"""

import json
import pathlib
import sys

RUNS = pathlib.Path("/projects/EEHPC-DEV-2026D08-260/runs")
GIB = 1024.0**3
LAYERS = 5420  # 20 x (127 X + 144 heavy-hex ZZ), recovered independently by the instrument

BASE_RECORD = 32  # sizeof(CrossRankPartnerRange) on the baseline arm
PORT_RECORD = 12  # sizeof(CrossRankOccupiedSlot)
CORE_GROWTH = 24  # sizeof(LayerCore): world_size + self_pos + self_offset

# (label, flat world P, run dir)
CELLS = [
    ("c12 1x16", 16, "models-pauli-sp-c12-g1x16-N1"),
    ("c12 1x128", 128, "models-pauli-sp-c12-g1x128-N1"),
    ("c14 8x16 N1", 128, "models-pauli-sp-c14-g8x16-N1"),
    ("c14 8x16 N2", 256, "models-pauli-sp-c14-g8x16-N2"),
    ("c14 8x16 N4", 512, "models-pauli-sp-c14-g8x16-N4"),
]

FIELDS = ("total_bytes", "cross_rank_bytes", "d_slot_record_bytes", "d_layer_cores",
          "d_slot_records", "d_occupied_slots", "d_cross_rank_endpoints")


def collect(d):
    """arm -> {field: [values]} from the graphmembreak block, plus rep counts and terms."""
    out = {"main": {}, "port": {}}
    reps = {"main": 0, "port": 0}
    terms = {"main": set(), "port": set()}
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
        for rec in (blob.get("graphmembreak") or {}).values():
            for field, val in rec.items():
                out[arm].setdefault(field, []).append(val)
    return out, reps, terms


def med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


failures = []
rows = []
print("=" * 96)
print("SLOT RECORDS: dense over every possible partner -> sparse over the occupied ones")
print("=" * 96)

for label, world, dirname in CELLS:
    d = RUNS / dirname
    print(f"\n## {label}  (P={world})")
    if not d.is_dir():
        print(f"   MISSING {dirname} -- cell not run")
        failures.append(f"{label}: {dirname} missing")
        continue

    data, reps, terms = collect(d)
    print(f"   reps: main {reps['main']}  port {reps['port']}   terms {sorted(terms['port'])}")

    got = {}
    for arm in ("main", "port"):
        got[arm] = {f: med(data[arm].get(f, [])) for f in FIELDS}

    if got["main"]["d_slot_records"] is None or got["port"]["d_slot_records"] is None:
        print("   !! REFUSED: one arm reports no graphmembreak -- it predates the instrument")
        failures.append(f"{label}: an arm has no graphmembreak block")
        continue

    if terms["main"] and terms["port"] and terms["main"] != terms["port"]:
        print("   !! REFUSED: the two arms ran different operators")
        failures.append(f"{label}: term counts differ {terms['main']} vs {terms['port']}")
        continue

    slots = got["main"]["d_slot_records"]
    cores = got["main"]["d_layer_cores"]
    occ_main = got["main"]["d_occupied_slots"]
    occ_port = got["port"]["d_occupied_slots"]
    endpoints = got["port"]["d_cross_rank_endpoints"]

    # Structure: the instrument must recover the geometry independently of the config.
    if cores and abs(slots / cores - world) > 1e-9:
        failures.append(f"{label}: slots/cores = {slots / cores:.1f}, expected P={world}")
    if cores and abs(cores / LAYERS - world) > 1e-9:
        failures.append(f"{label}: cores/L = {cores / LAYERS:.1f}, expected P={world}")

    # THE load-bearing check. Occupancy is a property of the traffic; if the format changed
    # it, the layout is not addressing the same sends.
    if occ_main != occ_port:
        failures.append(f"{label}: occupied moved with the format, {occ_main} -> {occ_port}")

    # The ceiling the whole argument rests on.
    if endpoints is not None and occ_port > endpoints:
        failures.append(f"{label}: occupied {occ_port} exceeds endpoints {endpoints} -- impossible")

    # Per-arm record widths.
    if slots and abs(got["main"]["d_slot_record_bytes"] / slots - BASE_RECORD) > 1e-6:
        failures.append(f"{label}: baseline record {got['main']['d_slot_record_bytes'] / slots:.2f} B/slot, "
                        f"expected {BASE_RECORD}")
    if occ_port and abs(got["port"]["d_slot_record_bytes"] / occ_port - PORT_RECORD) > 1e-6:
        failures.append(f"{label}: port record {got['port']['d_slot_record_bytes'] / occ_port:.2f} B/occupied, "
                        f"expected {PORT_RECORD}")

    # The byte law, exact.
    drop = got["main"]["total_bytes"] - got["port"]["total_bytes"]
    predicted = BASE_RECORD * slots - PORT_RECORD * occ_port - CORE_GROWTH * cores
    print(f"   graph  main {got['main']['total_bytes'] / GIB:9.3f} GiB"
          f"   port {got['port']['total_bytes'] / GIB:9.3f} GiB"
          f"   -> {got['main']['total_bytes'] / got['port']['total_bytes']:.2f}x")
    print(f"   slot records  {got['main']['d_slot_record_bytes'] / GIB:9.3f} -> "
          f"{got['port']['d_slot_record_bytes'] / GIB:9.3f} GiB   "
          f"({got['main']['d_slot_record_bytes'] / max(got['port']['d_slot_record_bytes'], 1):.2f}x)")
    print(f"   drop measured {drop / GIB:9.3f} GiB   predicted {predicted / GIB:9.3f} GiB   "
          f"ratio {drop / predicted if predicted else float('nan'):.5f}")
    if predicted and abs(drop - predicted) > 0:
        failures.append(f"{label}: drop off by {drop - predicted} B "
                        f"({drop} measured vs {predicted} predicted) -- the model of what moved is wrong")

    occupancy = occ_port / slots
    print(f"   occupancy {occupancy:8.3%}   peers per (rank,layer) {occ_port / cores:8.2f}"
          f"   endpoints {endpoints:,}   occupied/endpoints {occ_port / endpoints:.3f}")
    rows.append((label, world, cores, slots, occ_port, endpoints, occupancy,
                 got["main"]["total_bytes"], got["port"]["total_bytes"]))

# The reason this is not merely a smaller constant: the ceiling has no P in it.
if rows:
    print()
    print("=" * 96)
    print("OCCUPANCY vs P -- occupied is bounded by endpoints, and endpoints does not depend on P")
    print("=" * 96)
    print(f"{'cell':<14}{'P':>6}{'occupancy':>11}{'peers/core':>12}{'endpoints':>16}{'occ/endpt':>11}{'graph x':>9}")
    for label, world, cores, slots, occ, endpoints, occupancy, tb_main, tb_port in rows:
        print(f"{label:<14}{world:>6}{occupancy:>10.2%}{occ / cores:>12.2f}"
              f"{endpoints:>16,}{occ / endpoints:>11.3f}{tb_main / tb_port:>8.2f}x")
    # Endpoints are a property of the operator, so cells sharing a workload must agree on them
    # regardless of P. That is the measurement the ceiling argument stands on.
    for workload in ("c12", "c14"):
        vals = {e for lbl, _, _, _, _, e, _, _, _ in rows if lbl.startswith(workload)}
        if len(vals) > 1:
            spread = (max(vals) - min(vals)) / max(vals)
            print(f"\n{workload}: endpoints across P vary by {spread:.4%} "
                  f"({min(vals):,} .. {max(vals):,})")
            if spread > 0.01:
                failures.append(f"{workload}: endpoints moved {spread:.2%} across P -- "
                                f"they are supposed to be a property of the operator")

print()
if failures:
    print("=" * 96)
    print("PREDICTION FAILED -- do not quote the tables above without explaining these")
    print("=" * 96)
    for f in failures:
        print(f"- {f}")
    sys.exit(1)
print("All predictions held: the byte law is exact, occupancy is unchanged by the format,")
print("and occupied slots stay under the P-independent endpoint ceiling.")
