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

"""Collate many A/B result directories into one table per model.

`ab_summary.py` compares two arms at ONE point and is the authority on whether that point
may be read at all. Nothing aggregated across points: `benches/report.py` reads whole-test
`memhwm` and means, and a sweep over cutoff, system size and rank count lives in a dozen
separate directories whose numbers have to be put beside each other to say anything.

This walks those directories and emits, per model, one row per (cell, operation): the size
the cell ran at, the term count both arms reached, each arm's median time, the PAIRED ratio
with its agreement count, each arm's per-operation memory, and peak RSS.

Three things it deliberately does not do:

- It does not recompute the ratio from the two arms' medians. It reuses ab_summary's
  `paired_ratios`, because the pairing is the entire reason the runs interleave.
- It does not hide a refused cell. A directory whose provenance check failed is listed
  under VOID and its rows are omitted, rather than printed as though they meant something.
- It does not pick the size axis by knowing what the models are. It reports whichever
  config fields actually VARY across the collated directories, so the axis names itself.

    python hpc/deucalion/tools/campaign_summary.py <results-dir> [<results-dir> ...]
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_summary import (
    GIB,
    agreement,
    check_provenance,
    collect,
    fmt_gib,
    fmt_ms,
    medians,
    op_rank,
    paired_ratios,
    short_op,
    verdict,
)

# `/usr/bin/time -v` writes "Maximum resident set size (kbytes): N" once per rank into the
# cell's log. This is the peak-RSS control that does NOT come from the suite's own
# instrumentation -- the ledger has been wrong against the kernel by up to 23x in both
# directions, so the two are reported side by side rather than one standing in for the other.
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def time_v_peak(results_dir: Path, side: str) -> tuple[float | None, float | None]:
    """Return (summed, worst-rank) peak RSS in bytes for one arm, from its cell logs."""
    total = 0.0
    worst = 0.0
    seen = False
    for log in sorted(results_dir.glob(f"*_{side}_r*.log")):
        with contextlib.suppress(OSError):
            for match in RSS_RE.finditer(log.read_text(errors="replace")):
                kib = float(match.group(1))
                total += kib * 1024
                worst = max(worst, kib * 1024)
                seen = True
    if not seen:
        return None, None
    # Summed over every rank of every rep; per-rep is what a node has to hold.
    reps = len(
        {p.name.rsplit("_r", 1)[-1] for p in results_dir.glob(f"*_{side}_r*.log")}
    )
    return (total / reps if reps else total), worst


def config_of(cells: dict) -> dict[str, object]:
    """Return the resolved model config a directory ran at, or {} if it recorded none."""
    for cell in cells.values():
        configs = cell.results.get("configs") or {}
        for fields in configs.values():
            if fields:
                return dict(fields)
    return {}


def terms_of(cells: dict, side: str) -> int | None:
    """Return the term count one arm reached, as the max over its recorded entries."""
    counts = [
        entry["terms"]
        for cell in cells.values()
        if cell.side == side
        for entry in (cell.results.get("opsize") or {}).values()
        if isinstance(entry, dict) and entry.get("terms")
    ]
    return max(counts) if counts else None


class Point:
    """One results directory: a single (model, size, layout, nodes) cell of the campaign."""

    def __init__(self, path: Path) -> None:
        """Read the directory and evaluate its provenance without printing it."""
        self.path = path
        self.cells, self.problems = collect(path)
        # check_provenance prints its own section; here only its verdict is wanted.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            self.refusals = (
                check_provenance(self.cells, allow_both_placed=True)
                if self.cells
                else []
            )
        if not self.cells:
            self.refusals = [f"no parseable cells in {path.name}"]
        self.config = config_of(self.cells)
        self.model = (
            next(
                (c.model for c in self.cells.values() if c.model),
                path.name.split("-")[1],
            )
            if self.cells
            else "?"
        )
        first = next(iter(self.cells.values()), None)
        self.nodes = first.nodes if first else 0
        self.layout = first.layout if first else "?"

    @property
    def ok(self) -> bool:
        """Whether this point's numbers may be read as a result."""
        return not self.refusals


def emit_model(model: str, points: list[Point]) -> None:
    """Print one model's campaign table."""
    # The axis names itself: report only the config fields that actually move across the
    # collated points, so a cutoff sweep shows cutoff and a size sweep shows the size.
    keys = sorted({k for p in points for k in p.config})
    axis = [k for k in keys if len({repr(p.config.get(k)) for p in points}) > 1]
    if not axis:  # a single point, or every point at the same size
        axis = [k for k in ("cutoff", "lower_atol") if k in keys]

    print(f"\n## {model}\n")
    print(f"Axis: {', '.join(axis) if axis else '(single size)'}\n")

    headers = [
        *axis,
        "N",
        "terms",
        "op",
        "main ms",
        "port ms",
        "port/main",
        "agree",
        "main dmem",
        "port dmem",
        "peak RSS",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for point in sorted(
        points, key=lambda p: [repr(p.config.get(k)) for k in axis] + [p.nodes]
    ):
        cells = list(point.cells.values())
        by_side = {s: [c for c in cells if c.side == s] for s in ("main", "port")}
        ops = sorted({op for c in cells for op in c.times}, key=op_rank)
        terms = terms_of(point.cells, "port") or terms_of(point.cells, "main")
        rss_port, _ = time_v_peak(point.path, "port")

        for i, op in enumerate(ops):
            ratios = paired_ratios(by_side, op, "min")
            med, same, total = agreement(ratios)
            row = [
                *((str(point.config.get(k, "-")) if i == 0 else "") for k in axis),
                str(point.nodes) if i == 0 else "",
                f"{terms:,}" if (terms and i == 0) else "",
                short_op(op),
                fmt_ms(
                    medians(
                        [
                            c.times[op]["median"]
                            for c in by_side["main"]
                            if op in c.times
                        ]
                    )
                ),
                fmt_ms(
                    medians(
                        [
                            c.times[op]["median"]
                            for c in by_side["port"]
                            if op in c.times
                        ]
                    )
                ),
                verdict(med),
                f"{same}/{total}" if total else "-",
                fmt_gib(
                    medians(
                        [
                            e["max"]
                            for c in by_side["main"]
                            for e in [(c.results.get("opmemdelta") or {}).get(op)]
                            if isinstance(e, dict) and "max" in e
                        ]
                    )
                ),
                fmt_gib(
                    medians(
                        [
                            e["max"]
                            for c in by_side["port"]
                            for e in [(c.results.get("opmemdelta") or {}).get(op)]
                            if isinstance(e, dict) and "max" in e
                        ]
                    )
                ),
                (f"{rss_port / GIB:.1f}" if (rss_port and i == 0) else ""),
            ]
            print("| " + " | ".join(row) + " |")


def main(argv: list[str]) -> int:
    """Render the campaign tables; return non-zero if any collated point was refused."""
    paths = [Path(a) for a in argv if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 2

    points = [Point(p) for p in paths if p.is_dir()]
    good = [p for p in points if p.ok]
    void = [p for p in points if not p.ok]

    print("# Campaign summary\n")
    print(
        f"{len(good)} point(s) collated, {len(void)} refused. Ratios are the median of the "
        "PER-REP PAIRED port/main ratios on `min`; read `agree` before the ratio. `dmem` is "
        "the worst rank's peak above the operation's own floor -- the engine's own "
        "accounting. `peak RSS` is the port arm's per-rep sum over ranks from "
        "`/usr/bin/time -v`, which is the kernel's answer and not the engine's; quote a "
        "B/term figure only beside it.\n"
    )

    for model in sorted({p.model for p in good}):
        emit_model(model, [p for p in good if p.model == model])

    print("\n## Void\n")
    if void:
        for point in void:
            print(f"- `{point.path.name}`: " + "; ".join(point.refusals))
        print(
            "\nThese are NOT rows with no effect; they are cells whose arms cannot be "
            "compared. Their numbers are omitted above on purpose."
        )
    else:
        print("None: every collated point passed its provenance check.")
    print()
    return 1 if void else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
