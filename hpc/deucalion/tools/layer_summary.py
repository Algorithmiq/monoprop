# Phase attribution summary for monoprop_LAYER_PROFILE=1 runs.
#
# Reads the LAYERPROF and LAYERPOP lines LayerProfile.h writes to stderr (so the producing
# pytest run needs `-s` -- its capture is fd-level and swallows C++ stderr otherwise) and
# reports the per-phase split, the derived counter ratios, and the store population.
#
# Every phase is summed over partitions and reported as a share of `layer_s`, NOT of the sum
# of the phases. The two differ by the unattributed remainder, and that remainder is the
# number that says whether the instrumentation covers the wall time at all: a split whose
# phases sum to 6% of `layer_s` describes almost nothing, however precise each phase is.
# It is printed on its own line for exactly that reason.
#
# Phase time is in PARTITION-seconds. Dividing by the partition count gives the per-core wall
# equivalent, which is the only form comparable to a bench timing. In particular `exchange_s`
# counts partitions *blocked* on the MPI_THREAD_SERIALIZED funnel, not partitions doing MPI --
# near-identical median and max across partitions is the signature of that blocking, so both
# are printed rather than the mean.

import statistics as st
import sys


def parse_kv(line):
    """LAYERPROF/LAYERPOP lines are `TAG k=v k=v ...`; values are int, float or a bare string."""
    out = {}
    for token in line.split()[1:]:
        key, _, value = token.partition("=")
        try:
            out[key] = float(value) if "." in value else int(value)
        except ValueError:
            out[key] = value
    return out


BASE_PHASES = ["fold_s", "emit_s", "index_s", "resolve_s", "insert_s", "contract_s"]
CROSS_PHASES = ["sendbuf_s", "exchange_s", "incoming_s"]
COUNTERS = ["gates", "anti", "foll", "emit", "cutoff", "reject", "push", "hit", "miss", "qbytes"]


def report_phases(rows):
    phases = BASE_PHASES + [p for p in CROSS_PHASES if p in rows[0]]
    totals = {k: sum(r[k] for r in rows) for k in phases}
    layer = sum(r.get("layer_s", 0.0) for r in rows)

    print("\n--- summed over %d partitions (partition-seconds) ---" % len(rows))
    for k in phases:
        share = (100 * totals[k] / layer) if layer else 0.0
        print("  %-11s %9.2f s   %5.1f%% of layer" % (k, totals[k], share))
    print("  %-11s %9.2f s" % ("phases sum", sum(totals.values())))
    if not layer:
        print("  layer_s absent -- shares above are meaningless; rebuild with the layer bracket")
        return
    remainder = layer - sum(totals.values())
    print("  %-11s %9.2f s" % ("layer_s", layer))
    print("  %-11s %9.2f s   %5.1f%% of layer  <- unattributed inside build_layer"
          % ("remainder", remainder, 100 * remainder / layer))
    print("  per-core wall equivalent: %.3f s over %d partitions" % (layer / len(rows), len(rows)))

    # Median next to max: a funnel makes every partition wait the same amount, so the two
    # converging is evidence about the mechanism, not redundant description.
    print("\n--- per-partition median / max seconds ---")
    for k in phases:
        values = [r[k] for r in rows]
        print("  %-11s med %8.3f   max %8.3f" % (k, st.median(values), max(values)))


def report_counters(rows):
    totals = {k: sum(r[k] for r in rows) for k in COUNTERS}
    print("\n--- counters (all partitions) ---")
    for k in COUNTERS:
        print("  %-8s %s" % (k, format(totals[k], ",")))

    emit = max(1, totals["emit"])
    push = max(1, totals["push"])
    probes = max(1, totals["hit"] + totals["miss"])
    print("\n  emit/anti       %.3f" % (totals["emit"] / max(1, totals["anti"])))
    print("  cutoff/emit     %.3f   <- share of emits paying the full cutoff_sums" % (totals["cutoff"] / emit))
    print("  reject/emit     %.3f   <- structural rejection rate" % (totals["reject"] / emit))
    print("  push/emit       %.3f" % (totals["push"] / emit))
    print("  hit rate        %.6f   <- near zero means the store is in a growth regime" % (totals["hit"] / probes))
    print("  query bytes     %.2f GiB   (%.1f B per pushed record)"
          % (totals["qbytes"] / 2 ** 30, totals["qbytes"] / push))


def report_population(pops):
    last_gate = max(p["gate"] for p in pops)
    final = [p for p in pops if p["gate"] == last_gate]
    rows_n = sum(p["rows"] for p in final)
    paired = sum(p["paired"] for p in final)
    overflow = sum(p["overflow"] for p in final)

    print("\n--- population at gate %d (%d partitions) ---" % (last_gate, len(final)))
    print("  rows           %s" % format(rows_n, ","))
    print("  fully paired   %s  (%.4f%%)" % (format(paired, ","), 100 * paired / max(1, rows_n)))
    print("  overflow rows  %s  (%.4f%%)  inline_width=%d"
          % (format(overflow, ","), 100 * overflow / max(1, rows_n), final[0]["width"]))

    hist = [0] * 34
    for p in final:
        for i, v in enumerate(str(p["k_hist"]).split(",")):
            hist[i] += int(v)
    total = max(1, sum(hist))
    print("  k histogram (share of rows):")
    for i, v in enumerate(hist):
        if v:
            print("    k=%-2d %12s  %6.2f%%" % (i, format(v, ","), 100 * v / total))
    print("  mean k         %.3f" % (sum(i * v for i, v in enumerate(hist)) / total))

    # The sweep runs on power-of-two gate indices, so the log carries the growth curve for free.
    print("\n--- growth (summed over partitions) ---")
    for gate in sorted({p["gate"] for p in pops}):
        sel = [p for p in pops if p["gate"] == gate]
        r_ = sum(p["rows"] for p in sel)
        print("  gate %-4d rows %14s  paired %8.4f%%  overflow %8.4f%%"
              % (gate, format(r_, ","), 100 * sum(p["paired"] for p in sel) / max(1, r_),
                 100 * sum(p["overflow"] for p in sel) / max(1, r_)))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: layer_summary.py <run.out>")
    lines = open(sys.argv[1], errors="replace").read().splitlines()

    rows = [parse_kv(l) for l in lines if l.startswith("LAYERPROF")]
    if not rows:
        sys.exit("no LAYERPROF lines: was monoprop_LAYER_PROFILE=1 set, and did pytest run with -s?")
    print("partitions reporting: %d" % len(rows))

    report_phases(rows)
    report_counters(rows)
    pops = [parse_kv(l) for l in lines if l.startswith("LAYERPOP")]
    if pops:
        report_population(pops)


if __name__ == "__main__":
    main()
