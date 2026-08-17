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

"""Paired A/B of monoprop_LAYER_PROFILE phase timers over an ab.sh results directory.

WHY THIS EXISTS ALONGSIDE ab_summary.py. `ab_summary.py` reads pytest-benchmark's JSON, i.e.
the end-to-end wall. Measured on this machine that instrument has a ~1% floor and needs ~54
paired reps to resolve 0.8%; the phase timers resolve 8% in ~5. For any change that lands
inside one phase, the phase timer is the instrument and the wall is the null control.

THREE RULES, EACH LEARNED BY GETTING IT WRONG HERE:

* **Divide per rep, then take the median.** A ratio of the two sides' medians printed 6x
  noise as "0.95x flat" on the 100M run. The arms of one rep run back to back, so a
  node-state swing hits both and cancels only in the paired form.
* **An agree count needs a p-value.** 4/4 reps agreeing is p=0.125 under a sign test -- the
  best a 4-rep run can do, and not significant. The p column is what stops a small run from
  reading as a result.
* **Print the counters.** For a knob that must not change WHICH work happens, arms that
  disagree on emit/reject/push mean the timing comparison is between two different
  workloads. That is a refusal, not a footnote.

Phase time is in PARTITION-seconds summed over every partition, which is what LayerProfile
reports; `exchange_s` counts partitions *blocked* on the MPI_THREAD_SERIALIZED funnel rather
than partitions doing MPI, so it is a cost even though it is not work.

    python hpc/deucalion/tools/layer_ab.py <results-dir>
"""

from __future__ import annotations

import math
import pathlib
import re
import statistics
import sys

# Written by ab.sh as N{nodes}_{layout}_{cell}_{side}_r{rep}.log. The layout field itself
# contains underscores ("B_8x16"), so it has to be the greedy one and everything else anchored
# against the tail -- a non-greedy layout silently matches nothing and the tool reports "no
# LAYERPROF data", which reads as an instrument that never fired rather than a parser that missed.
LABEL = re.compile(r"^N(?P<nodes>\d+)_(?P<layout>.+)_(?P<cell>[^_]+)_(?P<side>main|port)_r(?P<rep>\d+)$")

PHASES = [
    "fold_s",
    "emit_s",
    "index_s",
    "resolve_s",
    "insert_s",
    "contract_s",
    "sendbuf_s",
    "exchange_s",
    "incoming_s",
    "layer_s",
]
COUNTERS = ["gates", "anti", "foll", "emit", "cutoff", "reject", "push", "hit", "miss", "qbytes"]


def parse_kv(line: str) -> dict:
    out: dict = {}
    for token in line.split()[1:]:
        key, _, value = token.partition("=")
        try:
            out[key] = float(value) if "." in value else int(value)
        except ValueError:
            out[key] = value
    return out


def read_log(path: pathlib.Path) -> dict | None:
    """Sum every LAYERPROF field over the partitions in one run's log."""
    totals: dict[str, float] = {}
    n = 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("LAYERPROF"):
            continue
        n += 1
        for key, value in parse_kv(line).items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + value
    if n == 0:
        return None
    totals["_partitions"] = n
    return totals


def sign_test_p(agree: int, n: int) -> float:
    """Two-sided exact sign test. 4/4 is 0.125 -- the best a four-rep run can report."""
    if n == 0:
        return 1.0
    k = max(agree, n - agree)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: layer_ab.py <results-dir>\n")
        return 2
    root = pathlib.Path(argv[1])
    if not root.is_dir():
        sys.stderr.write("not a directory: %s\n" % root)
        return 2

    runs: dict[tuple[str, str, int], dict] = {}
    skipped = []
    for path in sorted(root.glob("*.log")):
        m = LABEL.match(path.stem)
        if m is None:
            continue
        totals = read_log(path)
        if totals is None:
            skipped.append(path.name)
            continue
        runs[(m["cell"], m["side"], int(m["rep"]))] = totals

    if not runs:
        sys.stderr.write(
            "no LAYERPROF data found in %s.\n"
            "The producing run needs monoprop_LAYER_PROFILE=1 AND pytest -s: capture is\n"
            "fd-level, so without -s a live instrument looks like one that never fired.\n" % root
        )
        return 1
    if skipped:
        print("!! %d log(s) held no LAYERPROF lines: %s" % (len(skipped), ", ".join(skipped[:6])))

    cells = sorted({key[0] for key in runs})
    refused: list[str] = []

    for cell in cells:
        reps = sorted(
            {key[2] for key in runs if key[0] == cell and (cell, "main", key[2]) in runs and (cell, "port", key[2]) in runs}
        )
        if not reps:
            continue
        print()
        print("## cell `%s` -- %d paired rep(s)" % (cell, len(reps)))
        print()

        # Counters first. A timing comparison between two different workloads is not a
        # comparison, so this decides whether the table below is readable at all.
        for name in COUNTERS:
            if name == "qbytes":
                continue  # the one counter this class of change is SUPPOSED to move
            mismatched = [
                r
                for r in reps
                if runs[(cell, "main", r)].get(name) != runs[(cell, "port", r)].get(name)
            ]
            if mismatched:
                refused.append(
                    "cell %s: counter `%s` differs between the arms in rep(s) %s -- the arms did "
                    "different work, so the phase ratios below compare two workloads."
                    % (cell, name, ",".join(str(r) for r in mismatched))
                )

        print("| phase | main (part-s) | port (part-s) | ratio | agree | p |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for phase in PHASES:
            ratios = []
            mains, ports = [], []
            for r in reps:
                a = runs[(cell, "main", r)].get(phase, 0.0)
                b = runs[(cell, "port", r)].get(phase, 0.0)
                mains.append(a)
                ports.append(b)
                if a and b:
                    ratios.append(b / a)
            # A phase that is zero on both sides is printed rather than skipped. Dropping the
            # row would make a phase that went to zero on ONE side look like a phase that was
            # never instrumented -- and sendbuf_s legitimately reads 0.00 under GraphSink,
            # which is a fact about the code, not a gap in the table.
            if not ratios:
                print("| `%s` | %.3f | %.3f | -- | | |" % (phase, statistics.median(mains), statistics.median(ports)))
                continue
            med = statistics.median(ratios)
            direction = 1 if med >= 1.0 else -1
            agree = sum(1 for x in ratios if (1 if x >= 1.0 else -1) == direction)
            print(
                "| `%s` | %.3f | %.3f | **%.4fx** | %d/%d | %.3g |"
                % (
                    phase,
                    statistics.median(mains),
                    statistics.median(ports),
                    med,
                    agree,
                    len(ratios),
                    sign_test_p(agree, len(ratios)),
                )
            )

        # qbytes is the witness that the codec is live at all; print it as bytes per pushed
        # record, which is the number the design is stated in.
        print()
        for side in ("main", "port"):
            qb = statistics.median([runs[(cell, side, r)].get("qbytes", 0) for r in reps])
            pu = statistics.median([runs[(cell, side, r)].get("push", 0) for r in reps])
            print("- `%s`: qbytes %.3f GiB over %d pushes = **%.2f B/record**"
                  % (side, qb / (1 << 30), pu, (qb / pu) if pu else 0.0))

    if refused:
        print()
        print("## REFUSED")
        print()
        for line in refused:
            print("- " + line)
        print()
        print("A non-zero exit means the tables above are a diagnostic, not a result.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
