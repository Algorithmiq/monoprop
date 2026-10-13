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

"""Drop from a campaign queue the points that earlier runs already settled.

Reads queue lines ``<estimate> <layers> <engine> <N>`` on stdin and writes back those that
still need a run: a point is settled when one of the given run directories holds a row for it
at the figure's cutoff that ``merge_octave_runs.settled`` accepts under the time limit. Skipped
points are reported on stderr.

    settled.py --ceiling 1800 <run dir> ... < queue > queue.todo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from merge_octave_runs import CUTOFF, ROWS, gather, settled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--ceiling", type=float, required=True)
    args = ap.parse_args()

    done = set()
    for row_kind, (_, engines) in ROWS.items():
        rows, _ = gather(args.run_dirs, engines, CUTOFF[row_kind])
        for (engine, layers), entries in rows.items():
            done |= {
                (engine, layers, n)
                for n, _, row, _ in entries
                if settled(row, args.ceiling)
            }
    for line in sys.stdin:
        _, layers, engine, n = line.split()[:4]
        if (engine, int(layers), int(n)) in done:
            print(f"[settled] {engine} L{layers} N={n}", file=sys.stderr)
        else:
            sys.stdout.write(line)


if __name__ == "__main__":
    main()
