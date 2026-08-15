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

# The operator ledger fields that exist on BOTH arms, so a ratio means something. The index
# is the change under test; `indexing_bytes` is the dedup table beside it; the row store and
# the total are what a reader actually budgets against. Fields carried by one arm only --
# the `d_invidx_*` breakdown keys, which this branch renamed -- are omitted rather than
# printed against a blank, and `matched_scratch_bytes` gets its own footnote: it is summed
# into the port's total and into no main total at all, which makes the total ratio a FLOOR.
LEDGER_FIELDS = (
    "inverted_index_bytes",
    "indexing_bytes",
    "operator_terms_bytes",
    "total_bytes",
)
PORT_ONLY_IN_TOTAL = "matched_scratch_bytes"

# The graph ledger. Deliberately NOT normalised by terms the way the operator ledger is: these
# fields scale with the flat world size P = ranks x partitions, not with the operator, so
# bytes-per-term would divide the effect away and print a shrinking number for a growing cost.
# Reported as raw GiB against P instead.
#
# The `d_` fields are diagnostics that total_bytes has never included -- see
# GraphMemoryBreakdown, which keeps them out on purpose so `graph` means the same thing across
# builds. slot_record is the part of cross_rank sized by the world rather than by the traffic;
# recv_cache and derivative_layout are resident memory the total has never counted, which is why
# the reported graph sits below RSS.
GRAPH_FIELDS = (
    "cross_rank_bytes",
    "exchange_layout_bytes",
    "cos_data_bytes",
    "layer_descriptor_bytes",
    "total_bytes",
    "d_slot_record_bytes",
    "d_recv_cache_bytes",
    "d_derivative_layout_bytes",
)


def graph_field(cells: list, field: str) -> float | None:
    """Return one arm's median bytes for one graph-ledger field.

    Absent on any build predating the graph_memory_breakdown binding, which includes every
    `main` arm -- conftest records nothing rather than zeros there, so an absent instrument and
    a genuinely flat field cannot be confused. None means "not reported", not "zero".
    """
    values = [
        fields[field]
        for cell in cells
        for _op, fields in (cell.results.get("graphmembreak") or {}).items()
        if field in fields
    ]
    return medians(values)


def graph_occupancy(cells: list) -> tuple[float | None, int | None]:
    """(occupied fraction, layer cores) for one arm, or (None, None) if unreported.

    Occupancy is the number that decides whether a sparse slot layout would pay; nothing
    reported it before this instrument existed.
    """
    slots = graph_field(cells, "d_slot_records")
    occupied = graph_field(cells, "d_occupied_slots")
    cores = graph_field(cells, "d_layer_cores")
    if not slots or occupied is None:
        return None, None
    return occupied / slots, int(cores) if cores else None


def ledger_bpt(cells: list, field: str) -> float | None:
    """Return one arm's median bytes-per-term for one ledger field.

    Every operation in a cell reports the same operator, so this medians over operations
    and reps alike: they are repeated measurements of one quantity, not a distribution.
    """
    values = [
        fields[field] / terms
        for cell in cells
        for op, fields in (cell.results.get("opmembreak") or {}).items()
        for terms in [((cell.results.get("opsize") or {}).get(op) or {}).get("terms")]
        if terms and field in fields
    ]
    return medians(values)


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
        # The flat world size, which is the axis the graph ledger lives on. Not a config field:
        # it is ranks x partitions, and `monoprop_threads` is where the partition count is
        # recorded. 0 when unrecorded, so a graph table can say so rather than print a wrong P.
        meta = first.results.get("meta", {}) if first else {}
        try:
            self.world = int(meta.get("ranks", 0)) * int(meta.get("monoprop_threads", 0))
        except (TypeError, ValueError):
            self.world = 0

    @property
    def ok(self) -> bool:
        """Whether this point's numbers may be read as a result."""
        return not self.refusals


def axis_of(points: list[Point]) -> list[str]:
    """Return the config fields that actually move across these points.

    The axis names itself: a cutoff sweep shows cutoff and a size sweep shows the size,
    without this file knowing what the models are.
    """
    keys = sorted({k for p in points for k in p.config})
    axis = [k for k in keys if len({repr(p.config.get(k)) for p in points}) > 1]
    if not axis:  # a single point, or every point at the same size
        axis = [k for k in ("cutoff", "lower_atol") if k in keys]
    return axis


# Below this the effect is too small to care about even when every rep agrees on it.
NOISE_BAND = 0.01


def verdict_with_agreement(ratio: float | None, same: int, total: int) -> str:
    """Render a ratio, but do not call a UNANIMOUS small difference "flat".

    `verdict` labels everything inside 0.90..1.10 flat, which is right when the reps
    disagree and wrong when they do not. A 1.05x that every one of six reps points the
    same way is p=0.031 under a sign test -- the best six reps can do, and the bar this
    project already set. Calling it flat asserts the absence of an effect the data
    resolved. "consistent" says small AND real, which is the honest reading.
    """
    base = verdict(ratio)
    if ratio is None or not base.endswith("(flat)"):
        return base
    if total >= 6 and same == total and abs(ratio - 1.0) > NOISE_BAND:
        return f"{ratio:.2f}x (consistent)"
    return base


def emit_model(model: str, points: list[Point], axis: list[str]) -> None:
    """Print one model's campaign table."""
    print(f"\n## {model}\n")
    print(f"Axis: {', '.join(axis) if axis else '(single size)'}\n")

    headers = [
        *axis,
        "N",
        "layout",
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
        points,
        key=lambda p: [repr(p.config.get(k)) for k in axis] + [p.layout, p.nodes],
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
                point.layout if i == 0 else "",
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
                verdict_with_agreement(med, same, total),
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


def emit_ledger(model: str, points: list[Point], axis: list[str]) -> None:
    """Print one model's operator-memory table, in bytes per term.

    Separate from the timing table because it is a different measurement with a different
    failure mode: these are the engine's own CAPACITY figures, which have disagreed with
    the kernel by more than an order of magnitude in both directions, so BOTH arms' peak
    RSS is printed beside them. One arm's RSS alone cannot settle the question the ledger
    raises; only the ratio of the two can, and the two ratios have already disagreed in
    sign -- a cell whose ledger said the port was 7% larger measured 2.7% smaller.
    """
    print(f"\n### {model} -- operator memory (B/term)\n")
    print(
        "`main/port` above 1.00x means the port is smaller, for the ledger columns and "
        "for RSS alike. The ledger is CAPACITY and the RSS is the kernel's answer for the "
        "whole job: they answer different questions and are allowed to disagree. Where "
        "they do, the RSS ratio is the one a user feels.\n"
    )

    headers = [
        *axis,
        "N",
        "layout",
        "terms",
        "field",
        "main",
        "port",
        "main/port",
        "main RSS",
        "port RSS",
        "RSS main/port",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    floors = []
    for point in sorted(
        points,
        key=lambda p: [repr(p.config.get(k)) for k in axis] + [p.layout, p.nodes],
    ):
        cells = list(point.cells.values())
        by_side = {s: [c for c in cells if c.side == s] for s in ("main", "port")}
        terms = terms_of(point.cells, "port") or terms_of(point.cells, "main")
        rss_main, _ = time_v_peak(point.path, "main")
        rss_port, _ = time_v_peak(point.path, "port")
        rss_ratio = f"{rss_main / rss_port:.3f}x" if (rss_main and rss_port) else "-"

        extra = ledger_bpt(by_side["port"], PORT_ONLY_IN_TOTAL)
        if extra:
            floors.append((point.path.name, extra))

        for i, field in enumerate(LEDGER_FIELDS):
            main_bpt = ledger_bpt(by_side["main"], field)
            port_bpt = ledger_bpt(by_side["port"], field)
            if main_bpt is None and port_bpt is None:
                continue
            ratio = f"{main_bpt / port_bpt:.2f}x" if (main_bpt and port_bpt) else "-"
            print(
                "| "
                + " | ".join(
                    [
                        *(
                            (str(point.config.get(k, "-")) if i == 0 else "")
                            for k in axis
                        ),
                        str(point.nodes) if i == 0 else "",
                        point.layout if i == 0 else "",
                        f"{terms:,}" if (terms and i == 0) else "",
                        field,
                        f"{main_bpt:.2f}" if main_bpt is not None else "-",
                        f"{port_bpt:.2f}" if port_bpt is not None else "-",
                        ratio,
                        (f"{rss_main / GIB:.1f}" if (rss_main and i == 0) else ""),
                        (f"{rss_port / GIB:.1f}" if (rss_port and i == 0) else ""),
                        rss_ratio if i == 0 else "",
                    ]
                )
                + " |"
            )

    if floors:
        worst = max(v for _, v in floors)
        print(
            f"\n`total_bytes` understates the saving: the port sums "
            f"`{PORT_ONLY_IN_TOTAL}` (up to {worst:.2f} B/term here) into its total and "
            f"main counts it in no field at all. The `total_bytes` ratios above are a "
            f"floor, not a best case."
        )


def emit_graph_ledger(model: str, points: list[Point], axis: list[str]) -> None:
    """Print one model's graph-memory table, in GiB against the flat world size P.

    Separate from the operator ledger because it lives on a different axis. The operator
    partitions -- it is flat in P at a fixed problem -- while the graph is the half that does
    not, and its per-layer arrays are indexed by P = ranks x partitions. Printing the two in
    one table invites reading a P-driven growth as a term-driven one, which is the mistake
    RESULTS-scaling.md section 4 made.
    """
    reported = [p for p in points if graph_field(list(p.cells.values()), "total_bytes") is not None]
    if not reported:
        return

    print(f"\n### {model} -- graph memory (GiB) against the flat world P\n")
    print(
        "`P` = ranks x partitions, which is what the engine's `rank_count` returns on a "
        "partitioned run -- not the MPI rank count. Raw GiB, NOT bytes per term: these fields "
        "scale with P, so normalising by terms would divide the effect away. Blank where the "
        "arm's binding predates `graph_memory_breakdown`, which is every `main` arm -- absent "
        "is not zero.\n"
    )

    headers = [*axis, "N", "layout", "P", "arm", *(f.replace("_bytes", "") for f in GRAPH_FIELDS), "occupancy"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for point in sorted(reported, key=lambda p: [repr(p.config.get(k)) for k in axis] + [p.world, p.nodes]):
        cells = list(point.cells.values())
        # Keyed on the first row actually PRINTED, not on the first arm: `main` is skipped
        # whenever its binding predates the instrument, which is the common case, and keying on
        # the arm index would then blank the identity columns of every row in the table.
        first_row = True
        for side in ("main", "port"):
            side_cells = [c for c in cells if c.side == side]
            if graph_field(side_cells, "total_bytes") is None:
                continue
            occupied, _cores = graph_occupancy(side_cells)
            values = [graph_field(side_cells, f) for f in GRAPH_FIELDS]
            lead = first_row
            first_row = False
            print(
                "| "
                + " | ".join(
                    [
                        *((str(point.config.get(k, "-")) if lead else "") for k in axis),
                        str(point.nodes) if lead else "",
                        point.layout if lead else "",
                        (str(point.world) if point.world else "?") if lead else "",
                        side,
                        *(f"{v / GIB:.3f}" if v is not None else "-" for v in values),
                        f"{100 * occupied:.1f}%" if occupied is not None else "-",
                    ]
                )
                + " |"
            )

    print(
        "\nOccupancy is the fraction of world slots carrying any traffic. It decides whether a "
        "sparse slot layout would pay: a slot record is retained whether or not anything crosses "
        "to it, so `1 - occupancy` is the share of `d_slot_record` describing nothing."
    )


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
        chosen = [p for p in good if p.model == model]
        axis = axis_of(chosen)
        emit_model(model, chosen, axis)
        emit_ledger(model, chosen, axis)
        emit_graph_ledger(model, chosen, axis)

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
