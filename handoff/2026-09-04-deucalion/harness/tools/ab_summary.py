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

"""Paired time+memory summary for ab.sh.

Reads the two artifacts an interleaved run drops per cell -- pytest-benchmark's
`time-<label>.json` and the suite's own `<label>.json` -- and reports, per node count and
operation, each side's median across repetitions plus the port/main ratio for time and
for memory.

The headline ratio is the median of the PER-REP PAIRED ratios, not the ratio of the two
sides' medians. The two arms of a rep run back to back on the same node, so a node-state
swing hits both and cancels in the quotient; taking medians per side first throws that
pairing away and lets a swing that touched only one arm survive into the result. The
pairing is the entire reason ab.sh interleaves, and computing the ratio the obvious
way spends it for nothing.

`agree` -- how many reps point the same way as their median -- is reported next to every
ratio with its two-sided sign-test p, and an operation is RESOLVED when that p is below
`P_RESOLVE`. Median across reps rather than mean: an interleaved run is deliberately
exposed to whatever else is in the allocation, and one slow rep should move nothing.

The Provenance section is not decoration. It exits non-zero when the two arms did not do
the same work, or when thread placement did not match what the campaign DECLARED -- states
that otherwise print as a clean table full of meaningless ratios.

MEMORY IS TRACKED BY KERNEL PEAK RSS, from `/usr/bin/time -v` around every rank, which
ab.sh persists per rep into `PEAK-RSS.tsv`. That figure does not come from the code under
test, and that is the entire point: the engine's own byte ledger has been wrong by up to
23x in BOTH directions against the kernel. The ledger is still printed, clearly labelled,
as a secondary diagnostic -- with the operator and graph halves kept apart, because a
graph-memory win is not representable in their sum.

NO FLAGS. Everything this tool needs to know about how the cell was run it reads from
`CELL-META.tsv`, written by ab.sh beside the measurements: the launcher recipe, the
geometry, the harness revision, and the expected thread placement. A flag would put that
declaration in an argv the artifact does not record, which is what let pr_report.py re-run
this tool with a GUESSED `--allow-both-placed` and disarm a refusal in the report.

  python hpc/deucalion/tools/ab_summary.py <results-dir>
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

# time-N2_B_8x16_fresh_port_r3.json          -> nodes 2, layout "B_8x16", no model
# time-N1_B_8x16_hubbard_fresh_port_r3.json  -> the same, model "hubbard"
#
# Anchored on the layout's own shape rather than a greedy `.+`: with the group appended, a
# greedy layout group swallows "B_8x16_fresh" and every cell lands in its own table.
#
# The workload segment is optional so pre-ab.sh runs keep parsing unchanged; ab.sh always
# emits it, and it becomes part of the table key. Without it every
# model in a campaign directory collates into one table and the ratios average across
# workloads that are not comparable.
LABEL_RE = re.compile(
    r"^(?P<nodes>\d+)_(?P<layout>[A-Z]_\d+x\d+)(?:_(?P<model>[a-z][a-z0-9]*))?"
    r"_(?P<group>fresh|graph)_(?P<side>main|port)_r(?P<rep>\d+)$"
)

# Above/below these the difference is called out rather than left for the reader to spot.
# Deliberately wide: a handful of reps on a shared machine do not resolve 5%.
REGRESS = 1.10
IMPROVE = 0.90

# Resolution is this per-test sign-test p, not unanimity: unanimity gets HARDER as reps rise,
# so 10/10 (p=0.002) read as unresolved beside 6/6 (p=0.031). Holm across the campaign family
# stays in pr_report.py and is reported, never gating.
P_RESOLVE = 0.05

# Inside this band of 1.00 the ratio is called flat -- a magnitude statement, which needs no
# direction and so does not consult the sign test.
NOISE_BAND = 0.01

# Report order, not alphabetical: it keeps the two cells legible, since the first two
# operations build their own propagator and the last two share one graph.
OP_ORDER = ("build_graph", "propagate", "energy", "gradient")

# Term counts are compared as a ratio rather than for equality: they are identical by
# construction (same seed, same benches/), so any drift at all means the arms diverged.
TERM_TOLERANCE = 0.001

GIB = 1024**3

# Below this, a per-operation memory delta is page-granularity noise rather than a
# measurement, and dividing one by another manufactures dramatic ratios out of nothing.
# Operations that legitimately allocate almost nothing (energy over a resident graph) sit
# here on purpose -- their cost is the graph, which the op+graph column carries.
MEM_FLOOR = 16 * 1024**2


def op_rank(op: str) -> tuple[int, str]:
    """Return a sort key placing ``op`` in :data:`OP_ORDER`, unknown names last."""
    for index, name in enumerate(OP_ORDER):
        if name in op:
            return (index, op)
    return (len(OP_ORDER), op)


def short_op(op: str) -> str:
    """Return the display form of an op key, e.g. ``build_graph[heisenberg]``.

    The full key is kept everywhere it is used to join the two artifact families; only the
    table shows this, because the shared prefix pushes every number off the screen.
    """
    name = op.rsplit("::", maxsplit=1)[-1]
    for prefix in ("test_random_", "test_model_"):
        name = name.removeprefix(prefix)
    return name


def label_of(path: Path) -> re.Match[str] | None:
    """Return the parsed label of a results file, or ``None`` if it is not one."""
    stem = path.stem.removeprefix("time-").removeprefix("N")
    return LABEL_RE.match(stem)


class Cell:
    """One (nodes, layout, model, group, side, rep) run's timing and memory artifacts."""

    def __init__(self, match: re.Match[str]) -> None:
        """Record the label fields; payloads are attached as the files are read."""
        self.nodes = int(match["nodes"])
        self.side = match["side"]
        self.rep = int(match["rep"])
        self.layout = match["layout"]
        self.model = match["model"] or ""
        self.group = match["group"]
        self.times: dict[str, dict[str, float]] = {}
        self.results: dict[str, dict] = {}

    @property
    def table_key(self) -> tuple[int, str, str]:
        """The table this cell belongs in. Cells from different keys must not be pooled."""
        return (self.nodes, self.layout, self.model)


# CELL-META.tsv keys this tool cannot proceed without. `expect_placement` decides a refusal, so
# its absence is itself a refusal rather than a default; the launcher keys are what make "both
# arms ran one recipe" provable from the artifact instead of assumed.
REQUIRED_META = ("expect_placement", "cpu_bind", "distribution", "cpus_per_task", "harness_sha")


def read_meta(results_dir: Path) -> dict[str, str]:
    """Return the cell's CELL-META.tsv as key -> value, or ``{}`` when it is absent.

    Written by ab.sh at measurement time. It is deliberately a separate file rather than an
    argument: a declaration that lives only in an argv is not recorded anywhere the artifact
    keeps, so a later re-run of this tool has to guess it -- and the guess that shipped was the
    LOOSE gate, which disarmed the placement refusal in the report for every cell.
    """
    path = results_dir / "CELL-META.tsv"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition("\t")
        out[key.strip()] = value.strip()
    return out


def read_peak_rss(results_dir: Path) -> dict[tuple[int, str, str, str, str, int], tuple[int, int, int]]:
    """Return ``(nodes, layout, model, group, side, rep) -> (sum_kb, worst_kb, ranks)``.

    Kernel-truth peak RSS, one row per rank-group written by ab.sh from `/usr/bin/time -v`.
    Rows whose label does not parse are skipped rather than guessed at: this file is appended
    to by an awk inside the measurement loop, so a truncated final line is a real possibility
    and inventing a cell from it would be worse than losing it.
    """
    path = results_dir / "PEAK-RSS.tsv"
    out: dict[tuple[int, str, str, str, str, int], tuple[int, int, int]] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) != 7:
            continue
        label, side, rep, group, sum_kb, worst_kb, ranks = fields
        match = LABEL_RE.match(label.removeprefix("N"))
        if match is None:
            continue
        try:
            key = (
                int(match["nodes"]),
                match["layout"],
                match["model"] or "",
                group,
                side,
                int(rep),
            )
            out[key] = (int(sum_kb), int(worst_kb), int(ranks))
        except ValueError:
            continue
    return out


def collect(results_dir: Path) -> tuple[dict[str, Cell], list[str]]:
    """Return ``label -> Cell`` for every readable artifact, plus any read failures."""
    cells: dict[str, Cell] = {}
    problems: list[str] = []

    def load(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"unreadable {path.name}: {exc}")
            return None

    for path in sorted(results_dir.glob("*.json")):
        match = label_of(path)
        if match is None:
            continue
        label = path.stem.removeprefix("time-")
        cell = cells.setdefault(label, Cell(match))
        payload = load(path)
        if payload is None:
            continue
        if path.name.startswith("time-"):
            for bench in payload.get("benchmarks", []):
                # Same op key as benches/report.py, so the two agree on naming.
                op = bench["fullname"].split("/")[-1]
                cell.times[op] = {
                    "median": bench["stats"]["median"],
                    "min": bench["stats"]["min"],
                }
        else:
            cell.results = payload
    return cells, problems


def medians(values: list[float]) -> float | None:
    """Return the median of ``values``, or ``None`` when there are none."""
    return statistics.median(values) if values else None


def gather(cells: list[Cell], op: str, section: str, field: str) -> list[float]:
    """Return one number per rep for ``op`` out of a ``<label>.json`` section."""
    out = []
    for cell in cells:
        entry = cell.results.get(section, {}).get(op)
        if isinstance(entry, dict) and field in entry:
            out.append(float(entry[field]))
    return out


def paired_ratios(by_side: dict[str, list[Cell]], op: str, key: str) -> list[float]:
    """Return one ``port/main`` ratio per rep that both sides ran.

    This, and not the ratio of the two sides' medians, is what the interleaving buys.
    The two arms of a rep run back to back on the same node, so a node-state swing hits
    both and cancels in the quotient; comparing medians computed independently across
    reps throws that pairing away and lets a swing that touched only one arm survive
    into the ratio.

    It also changes conclusions rather than just tightening them. At 100M terms
    `build_graph` on one node gave per-rep ratios of 2.44, 0.44, 0.41 and 2.11 -- one
    arm near 1.7 s and the other near 4 s in every rep, alternating which. The medians
    of those two sets are nearly equal, so the unpaired ratio reads 0.95x "flat", which
    is indistinguishable from a genuine null. The paired view shows four reps that do
    not agree on a direction, i.e. no resolvable effect at all.
    """
    out = []
    for rep in sorted(
        {c.rep for c in by_side["main"]} & {c.rep for c in by_side["port"]}
    ):
        pair = {}
        for side, cells in by_side.items():
            run = next(
                (c.times[op] for c in cells if c.rep == rep and op in c.times), None
            )
            if run is not None:
                pair[side] = run[key]
        if len(pair) == 2 and pair["main"]:
            out.append(pair["port"] / pair["main"])
    return out


def sign_p_agree(agree: int, compared: int) -> float:
    """Two-sided sign-test p from an ``(agree, compared)`` count alone.

    The only copy of the formula in this file; `pr_report.py`'s `sign_p_from_agree` is the same
    closed form over the integers it parses back out of the rendered table.
    """
    if compared <= 0:
        return 1.0
    return min(
        1.0,
        2 * sum(math.comb(compared, k) for k in range(agree, compared + 1)) / 2**compared,
    )


def agreement(ratios: list[float]) -> tuple[float | None, int, int]:
    """Return (median ratio, reps agreeing with its direction, reps compared)."""
    if not ratios:
        return None, 0, 0
    med = statistics.median(ratios)
    same = sum(1 for r in ratios if (r >= 1.0) == (med >= 1.0))
    return med, same, len(ratios)


def sign_p(ratios: list[float]) -> float | None:
    """Two-sided sign-test p for "the arms differ in direction at all".

    Reported next to `agree` because the bare count invites over-reading, and did: a 4-rep
    run showing 4/4 was called solid here, when 4/4 is the *best attainable* outcome at that
    rep count and still only reaches p=0.125. **No four-rep run can be significant by sign
    alone.** Three separate 100M runs each landed at p=0.13-0.18 on `propagate` at N=1;
    only pooling their 17 paired observations (15/17 slower, p=0.002) established it.

    The test deliberately throws away magnitude, so it is a floor rather than the whole
    story: several reps agreeing at a consistent, large ratio carry information this does
    not see. Use it to know when a count is doing less work than it appears to, not as the
    sole verdict.
    """
    if not ratios:
        return None
    # agreement() splits the reps on the same 1.0 threshold, so its count IS this test's k
    # (checked over every sign pattern to n=12; only a ratio of exactly 1.0 could split them).
    _, agree, compared = agreement(ratios)
    return sign_p_agree(agree, compared)


# The p rendered by fmt_agree is uncorrected: it is one sign test out of the whole campaign's
# family (24 in the common shape -- 4 operations each for the pauli cells, 1 for hubbard), and
# pr_report.py's "Multiplicity correction" section Holm-adjusts across that entire family, never
# per test. A bare `p=.03` in a per-cell table reads as evidence on its own to anyone skimming
# straight to it, so every rendered p carries this marker and OP_TABLE_FOOTNOTE is printed once
# under each operation table it appears in. This file cannot know the family size -- that is a
# property of everything run-campaign.sh submitted under one --pr, i.e. pr_report.py's job, not
# this one's -- so the footnote says only that the value is uncorrected and where the correction
# lives, never a specific corrected number.
P_UNCORRECTED_MARK = "†"
OP_TABLE_FOOTNOTE = (
    f"`{P_UNCORRECTED_MARK}` uncorrected p (one sign test in isolation) -- see the campaign "
    "report's Multiplicity correction section before reading any single one of these on its own."
)


def fmt_agree(agree: int, compared: int, ratios: list[float]) -> str:
    """Render the agreement count with its sign-test p, e.g. ``7/9 p=.18†``.

    See :data:`OP_TABLE_FOOTNOTE` -- the trailing mark is not decoration, it is the pointer from
    a per-cell table back to where the multiplicity correction for this exact number lives.
    """
    if not compared:
        return "-"
    p = sign_p(ratios)
    return f"{agree}/{compared}" + (
        "" if p is None else f" p={p:.2f}".replace("0.", ".") + P_UNCORRECTED_MARK
    )


def fmt_ms(seconds: float | None) -> str:
    """Render seconds as milliseconds, or a dash."""
    return "-" if seconds is None else f"{seconds * 1e3:.1f}"


def fmt_gib(value: float | None) -> str:
    """Render bytes as GiB, or a dash."""
    return "-" if value is None else f"{value / GIB:.2f}"


def verdict(ratio: float | None, agree: int | None = None, compared: int | None = None) -> str:
    """Render a port/main ratio as a human verdict, qualified by whether the reps resolve it.

    REGRESS/IMPROVE decide the headline word from magnitude alone. The ratio is RESOLVED when
    `agree`/`compared` -- from the same `agreement()` pairing as every other per-rep number here
    -- give a sign-test p below :data:`P_RESOLVE`; anything else prints `(unresolved)`, large
    ratios included. `agree=None` (the byte ledger's `dmem_ratio`, two unpaired side medians)
    means there is nothing to resolve, and that site prints exactly as before: `(resolved)` there
    would claim a test that was never run on it.

    In band a PAIRED ratio is `(flat)` within :data:`NOISE_BAND` of 1.00 and `(resolved)`/
    `(unresolved)` outside it. `(flat)` is a magnitude claim and asserts no direction, so it does
    not consult the sign test: a genuine null scatters its reps either side of 1.00 and cannot
    pass one by construction, which is what made the word unreachable where it was true.
    """
    if ratio is None or ratio <= 0:
        return "-"
    unpaired = agree is None or not compared
    resolved = unpaired or sign_p_agree(agree, compared) < P_RESOLVE
    if ratio >= REGRESS:
        base = f"**{ratio:.2f}x slower**"
        return base if resolved else f"{base} (unresolved)"
    if ratio <= IMPROVE:
        # BOTH numbers, deliberately. This is the only band whose headline figure is the ratio
        # INVERTED (`1/ratio`, the speedup); every other band prints the raw port/main, and the
        # column heading says `port/main` for all of them. So a table holding `0.91x (resolved)`
        # beside `1.14x faster` mixes two different quantities under one heading, and the
        # larger-LOOKING number is the SMALLER win -- 1.14x faster is port/main 0.88, a bigger
        # win than the 0.91 the neighbouring row shows as a raw ratio. These rows are quoted
        # verbatim into PR bodies, where that reading cannot be recovered from context, so the
        # raw ratio travels with the speedup rather than being replaced by it.
        base = f"{1 / ratio:.2f}x faster ({ratio:.2f}x)"
        return base if resolved else f"{base} (unresolved)"
    if unpaired or abs(ratio - 1.0) <= NOISE_BAND:
        return f"{ratio:.2f}x (flat)"
    return f"{ratio:.2f}x (resolved)" if resolved else f"{ratio:.2f}x (unresolved)"


def ratio_of(port: float | None, main: float | None) -> float | None:
    """Return ``port / main`` when both exist and ``main`` is non-zero."""
    if port is None or main is None or not main:
        return None
    return port / main


def emit_rss(
    key: tuple[int, str, str], cells: list[Cell], rss: dict, flagged: list[str]
) -> None:
    """Print the kernel peak-RSS table for one (nodes, layout, model).

    THIS IS THE CAMPAIGN'S MEMORY METRIC. It comes from `/usr/bin/time -v` around every rank,
    i.e. from the kernel and not from the code under test -- which matters because the engine's
    own byte ledger has been wrong by up to 23x in BOTH directions against it.

    BOTH FIGURES, and they answer different questions. `sum` is the node total and is the only
    one comparable ACROSS LAYOUTS: at layout A there is one rank per node, at layout B there are
    eight, so a per-rank figure at A is ~8x the same node memory at B. `worst` is the largest
    single rank, which is what an OOM is decided by.

    Ratios are paired per rep, as everywhere else here: the two arms of a rep run back to back
    on one node, so a node-state swing hits both and cancels in the quotient.
    """
    nodes, layout, model = key
    groups = sorted({c.group for c in cells})
    rows = []
    for group in groups:
        per_side = {
            side: {
                rep: rss[(nodes, layout, model, group, side, rep)]
                for rep in sorted({c.rep for c in cells if c.group == group})
                if (nodes, layout, model, group, side, rep) in rss
            }
            for side in ("main", "port")
        }
        if not per_side["main"] and not per_side["port"]:
            continue
        shared = sorted(set(per_side["main"]) & set(per_side["port"]))
        row = {"group": group, "reps": (len(per_side["main"]), len(per_side["port"]))}
        for index, name in ((0, "sum"), (1, "worst")):
            for side in ("main", "port"):
                row[f"{side}_{name}"] = medians(
                    [v[index] * 1024 for v in per_side[side].values()]
                )
            ratios = [
                per_side["port"][rep][index] / per_side["main"][rep][index]
                for rep in shared
                if per_side["main"][rep][index]
            ]
            row[f"{name}_ratio"], row[f"{name}_agree"], row[f"{name}_n"] = agreement(ratios)
            row[f"{name}_ratios"] = ratios
        rows.append(row)

    print("### Peak RSS -- kernel truth (`/usr/bin/time -v`), the memory metric\n")
    if not rows:
        print(
            "No `PEAK-RSS.tsv` rows for this table. `/usr/bin/time -v` did not run, or its\n"
            "output did not reach the cell's log. **The memory numbers below are the engine's\n"
            "own ledger only, which is not an independent check of the engine.**\n"
        )
        return
    headers = [
        "cell",
        "main sum (GiB)",
        "port sum (GiB)",
        "sum ratio",
        "agree",
        "main worst (GiB)",
        "port worst (GiB)",
        "worst ratio",
        "reps",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for row in rows:
        print(
            f"| {row['group']} | {fmt_gib(row['main_sum'])} | {fmt_gib(row['port_sum'])} "
            f"| {verdict(row['sum_ratio'], row['sum_agree'], row['sum_n'])} "
            f"| {fmt_agree(row['sum_agree'], row['sum_n'], row['sum_ratios'])} "
            f"| {fmt_gib(row['main_worst'])} | {fmt_gib(row['port_worst'])} "
            f"| {verdict(row['worst_ratio'], row['worst_agree'], row['worst_n'])} "
            f"| {row['reps'][0]}/{row['reps'][1]} |"
        )
    print(
        "\n`sum` is the NODE TOTAL over all ranks and is the only figure comparable across\n"
        "layouts; `worst` is the largest single rank, which is what an OOM is decided by. At\n"
        "layout A (1 rank/node) the two are equal; at layout B (8 ranks/node) sum is ~8x worst.\n"
    )
    where = f"N={nodes} / {layout}" + (f" / {model}" if model else "")
    for row in rows:
        for name in ("sum", "worst"):
            ratio = row[f"{name}_ratio"]
            if ratio is not None and ratio > 0 and (ratio >= REGRESS or ratio <= IMPROVE):
                direction = "larger" if ratio >= REGRESS else "smaller"
                flagged.append(
                    f"- `{where}` / `{row['group']}` / peak RSS {name}: {ratio:.2f}x "
                    f"{direction} (kernel, paired, "
                    f"{fmt_agree(row[f'{name}_agree'], row[f'{name}_n'], row[f'{name}_ratios'])} "
                    "reps agree)"
                )


def emit_ledger(key: tuple[int, str, str], cells: list[Cell]) -> None:
    """Print the engine's own byte ledger, per arm, as a SECONDARY diagnostic.

    Kept apart from the RSS table above and labelled as secondary on purpose: the owner tracks
    memory by RSS because this ledger has been wrong by up to 23x in both directions against the
    kernel -- it overstated Hubbard 23x and understated the kicked Ising 2.8x, and `rows_` slack
    telemetry reports capacity rather than resident bytes.

    The operator and graph halves are printed SEPARATELY and PER ARM. The column this replaces
    printed `max(main, port)` of their SUM, which destroyed both splits at once: a genuine
    graph-memory win was not representable in it, and the per-arm difference it was meant to
    summarise was the thing it discarded.
    """
    nodes, layout, model = key
    by_side = {side: [c for c in cells if c.side == side] for side in ("main", "port")}
    ops = sorted({op for cell in cells for op in cell.times}, key=op_rank)
    print("### Engine byte ledger -- SECONDARY, not the memory metric\n")
    headers = [
        "operation",
        "main dmem/rank (GiB)",
        "port dmem/rank (GiB)",
        "dmem ratio",
        "main operator (GiB)",
        "port operator (GiB)",
        "main graph (GiB)",
        "port graph (GiB)",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")
    for op in ops:
        stats = {}
        for side, side_cells in by_side.items():
            stats[side] = {
                "dmem": medians(gather(side_cells, op, "opmemdelta", "max")),
                "operator": medians(gather(side_cells, op, "opbytes", "operator")),
                "graph": medians(gather(side_cells, op, "opbytes", "graph")),
            }
        main, port = stats["main"], stats["port"]
        mem_ratio = None
        if min(main["dmem"] or 0, port["dmem"] or 0) >= MEM_FLOOR:
            mem_ratio = ratio_of(port["dmem"], main["dmem"])
        print(
            f"| {short_op(op)} | {fmt_gib(main['dmem'])} | {fmt_gib(port['dmem'])} "
            f"| {verdict(mem_ratio)} "
            f"| {fmt_gib(main['operator'])} | {fmt_gib(port['operator'])} "
            f"| {fmt_gib(main['graph'])} | {fmt_gib(port['graph'])} |"
        )
    print(
        "\n**`dmem/rank` is a MAX over ranks, so it is NOT comparable across layouts.** At\n"
        "layout A there is one rank per node and max == the node total; at layout B there are\n"
        "eight and max is roughly the node total over 8. A prints ~8x B's for identical node\n"
        "memory. Paired ratios are unaffected -- both arms of a cell share a layout -- so only\n"
        "the absolute columns mislead. Read the peak-RSS `sum` above for a cross-layout figure.\n"
    )


def emit_table(
    key: tuple[int, str, str], cells: list[Cell], rss: dict, flagged: list[str]
) -> None:
    """Print one (nodes, layout, model) operation table, then its memory sections."""
    nodes, layout, model = key
    by_side = {side: [c for c in cells if c.side == side] for side in ("main", "port")}
    ops = sorted({op for cell in cells for op in cell.times}, key=op_rank)

    where = f"N={nodes} / {layout}" + (f" / {model}" if model else "")
    print(f"## {where}\n")
    headers = [
        "operation",
        "main med (ms)",
        "port med (ms)",
        "port/main",
        "agree",
        "main min",
        "port min",
        "reps",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")

    for op in ops:
        stats = {}
        for side, side_cells in by_side.items():
            runs = [c.times[op] for c in side_cells if op in c.times]
            stats[side] = {
                "median": medians([r["median"] for r in runs]),
                "min": min((r["min"] for r in runs), default=None),
                "reps": len(runs),
                "dmem": medians(gather(side_cells, op, "opmemdelta", "max")),
            }
        main, port = stats["main"], stats["port"]
        tratios = paired_ratios(by_side, op, "min")
        time_ratio, agree, compared = agreement(tratios)
        mem_ratio = None
        if min(main["dmem"] or 0, port["dmem"] or 0) >= MEM_FLOOR:
            mem_ratio = ratio_of(port["dmem"], main["dmem"])

        print(
            f"| {short_op(op)} | {fmt_ms(main['median'])} | {fmt_ms(port['median'])} "
            f"| {verdict(time_ratio, agree, compared)} | {fmt_agree(agree, compared, tratios)} "
            f"| {fmt_ms(main['min'])} | {fmt_ms(port['min'])} "
            f"| {main['reps']}/{port['reps']} |"
        )

        # Reps that do not agree on a direction are the loudest thing the table can say,
        # and the ratio column cannot say it: a split 2/4 still prints some number. Flag
        # it on its own so an unresolvable operation is never read as a measured null.
        #
        # Failing to resolve is not enough to fire, though. An operation that is genuinely
        # flat scatters its reps either side of 1.00 by construction, so direction alone
        # would flag every null result as unresolved and drown the real ones. Require the
        # spread to be wider than the effects being claimed: reps landing in 0.93..1.01
        # agree that nothing happened, while 0.41..2.44 agree on nothing at all.
        #
        # The gate is the SAME p as the table's own label, or a tight 9/10 earns a bullet
        # calling it unresolved beside a row that resolved it.
        ratios = tratios
        spread = max(ratios) / min(ratios) if ratios and min(ratios) > 0 else 1.0
        if (
            compared
            and sign_p_agree(agree, compared) >= P_RESOLVE
            and spread > REGRESS + 0.15
        ):
            flagged.append(
                f"- `{where}` / `{short_op(op)}` / time: **unresolved** -- only "
                f"{agree} of {compared} reps agree on a direction, spread {spread:.1f}x "
                f"(per-rep {', '.join(f'{r:.2f}' for r in ratios)})"
            )

        for name, ratio, extra in (
            ("time", time_ratio, f"paired, {fmt_agree(agree, compared, tratios)} reps agree"),
            (
                "ledger dmem/rank",
                mem_ratio,
                "ENGINE LEDGER, secondary -- read the peak-RSS table before this",
            ),
        ):
            if (
                ratio is not None
                and ratio > 0
                and (ratio >= REGRESS or ratio <= IMPROVE)
            ):
                direction = "slower/larger" if ratio >= REGRESS else "faster/smaller"
                flagged.append(
                    f"- `{where}` / `{short_op(op)}` / {name}: "
                    f"{ratio:.2f}x {direction} ({extra})"
                )
    print(f"\n{OP_TABLE_FOOTNOTE}\n")
    emit_rss(key, cells, rss, flagged)
    emit_ledger(key, cells)
    print()


def _check_terms(cells: dict[str, Cell]) -> list[str]:
    """Refuse unless both arms propagated the same operator."""
    refusals: list[str] = []
    terms = {
        (cell.side, op): entry["terms"]
        for cell in cells.values()
        for op, entry in cell.results.get("opsize", {}).items()
        if isinstance(entry, dict) and "terms" in entry
    }
    if terms:
        low, high = min(terms.values()), max(terms.values())
        print(f"- term count: {low:,} .. {high:,} across all cells")
        if low and (high - low) / low > TERM_TOLERANCE:
            refusals.append(
                f"term counts differ by more than {TERM_TOLERANCE:.1%} "
                f"({low:,} vs {high:,}): the two arms did not do the same work"
            )
    else:
        refusals.append(
            "no term counts recorded: cannot show the arms did the same work"
        )

    return refusals


def _check_placement(cells: dict[str, Cell], meta: dict[str, str]) -> list[str]:
    """Refuse when thread placement does not match what the campaign DECLARED.

    The declaration comes from `CELL-META.tsv`'s `expect_placement`, which ab.sh wrote from
    campaigns.tsv (run-campaign.sh picks `expect_a` or `expect_b` there by the cell's grid.tsv
    layout letter, because the expectation is a function of the LAYOUT, not of the campaign --
    see campaigns.tsv's column comment). There are exactly four legal declarations and each
    admits exactly one outcome:

        both        the baseline already carries the placement fix, so BOTH arms must place.
        port-only   the port introduces placement, so main must place NOTHING and port must.
        main-only   the port REMOVES placement, so main must place and port must place NOTHING.
                    port-only's mirror, and deliberately a separate declaration rather than a
                    reuse of `neither`: a campaign that deletes the engine's placement ends at
                    zero pinned threads on the port arm at BOTH layouts, including layout A,
                    where there is no cgroup confinement for `neither` to assert. Declaring
                    such a campaign `neither` would refuse it for the absence of a confinement
                    it was never supposed to have -- and would also stop checking the main arm,
                    which is the arm that has to still be placing for the comparison to mean
                    "with placement versus without".
        neither     Slurm's cgroup confines the rank and NEITHER arm places anything on top of
                    that confinement. This is a LEGITIMATE, CORRECT state, not a failure: at
                    layout B (8 ranks/node, 16 partitions) Slurm confines each rank to 16 cores
                    and the engine places nothing further, measured main=0/port=0 in 6 of 6
                    null-control cells. It is not, however, a free pass -- see below.

    THIS IS STRICTLY STRONGER THAN THE FLAG IT REPLACES. `--allow-both-placed` could only widen
    the gate to tolerate both arms placing. It could not distinguish main=0/port=16 -- the
    placement fix working -- from main=16/port=0 -- the placement fix REGRESSING -- because
    neither is "both", and both therefore passed silently. A one-sided result is now checked
    against which side was supposed to be the placed one.

    The signal is `single_cpu_threads_min` from `/proc/self/task/*/status`'s
    `Cpus_allowed_list`, MPI-reduced with MIN, produced by benches/_memory_cpu.py. It is
    DELIBERATELY not the engine's COMMPROF `pinned=` field: that exists only on builds carrying
    monoprop_COMM_PROFILE, so it cannot be compared across the boundary that added it, and it
    reports pinned=0 for idle single-partition transports, which would void good cells.

    `neither` MUST NOT BECOME A GATE THAT ASSERTS NOTHING. "Both arms placed 0 threads" is also
    what a cell would show if `/proc` were unreadable, or if Slurm granted the whole node and
    the engine simply chose not to place -- neither of which is the confined state `neither`
    declares. So this also asserts the confinement itself: `affinity_cpus` (the process mask's
    width, from the same `pinned_thread_summary()` that produces `single_cpu_threads_min`) must
    equal `cpus_per_task` from CELL-META on BOTH arms. Zero pinned threads over a mask that is
    NOT confined to `cpus_per_task` proves nothing about the layout-dependence this state
    exists to describe, and is refused same as a nonzero placement count would be.
    """
    refusals: list[str] = []
    placed = {}
    for side in ("main", "port"):
        values = [
            cell.results["meta"]["pinning"].get("single_cpu_threads_min")
            for cell in cells.values()
            if cell.side == side and cell.results.get("meta", {}).get("pinning")
        ]
        values = [v for v in values if v is not None]
        placed[side] = min(values) if values else None
        print(
            f"- {side}: {placed[side]} thread(s) pinned per rank (min over ranks/reps)"
        )

    expect = meta.get("expect_placement")
    print(f"- expected placement: {expect or 'NOT DECLARED'}")
    if expect is None:
        # Not a default. A cell that does not say what it expected cannot be checked, and an
        # unchecked placement gate is indistinguishable from a passing one.
        refusals.append(
            "no `expect_placement` in CELL-META.tsv, so the declared placement outcome for "
            "this cell is unknown and cannot be checked. It is not assumed: an unchecked "
            "gate reads exactly like a gate that passed. Re-run the cell under "
            "run-campaign.sh, which writes the declaration from campaigns.tsv."
        )
        return refusals

    if None in placed.values():
        refusals.append("placement was not recorded on at least one arm")
        return refusals

    if expect == "both":
        if not (placed["main"] and placed["port"]):
            refusals.append(
                f"this campaign declares expect=both, but main={placed['main']} and "
                f"port={placed['port']} threads pinned per rank. The baseline is supposed to "
                "carry the placement fix already, so an arm at 0 is either a venv that is not "
                "the build it is labelled as, or placement regressing on that arm -- and the "
                "timings either way compare a placed build against an unplaced one."
            )
    elif expect == "port-only":
        if placed["main"] != 0 or not placed["port"]:
            refusals.append(
                f"this campaign declares expect=port-only, i.e. main must place NOTHING and "
                f"port must place -- but main={placed['main']} and port={placed['port']}. "
                + (
                    "main placing threads means the baseline already carries the placement fix "
                    "and this campaign's baseline is mis-declared."
                    if placed["main"]
                    else "port not placing means the fix under test did not take effect."
                )
            )
    elif expect == "main-only":
        if not placed["main"] or placed["port"] != 0:
            refusals.append(
                f"this campaign declares expect=main-only, i.e. main must place and port must "
                f"place NOTHING -- but main={placed['main']} and port={placed['port']}. "
                + (
                    "port placing threads means the removal under test did not take effect, so "
                    "both arms place and the cell measures nothing about placement."
                    if placed["port"]
                    else "main not placing means the baseline was already unplaced, so there "
                    "was no placement for this port to remove and the cell is mis-declared."
                )
            )
    elif expect == "neither":
        if placed["main"] or placed["port"]:
            refusals.append(
                f"this campaign declares expect=neither, i.e. NEITHER arm is expected to place "
                f"any thread (Slurm's cgroup confines the rank and the engine places nothing "
                f"on top) -- but main={placed['main']} and port={placed['port']} threads "
                "pinned per rank. A layout declared confined that placed threads anyway is not "
                "the state this campaign recorded."
            )
        refusals.extend(_check_confinement(cells, meta))
    else:
        # Fails CLOSED. This used to be a bare `else` standing in for "neither", so any value
        # run-campaign.sh's case list accepted and this function did not know about was
        # silently checked as a confined layout -- the strictest gate, applied to the wrong
        # declaration. An unrecognised declaration is a harness inconsistency, not a cell
        # result, and must not resolve to any gate at all.
        refusals.append(
            f"CELL-META.tsv declares expect_placement='{expect}', which this tool has no "
            "check for. run-campaign.sh accepted it, so the two lists have diverged; the "
            "cell is refused rather than checked against whichever gate happens to be last."
        )
    return refusals


def _check_confinement(cells: dict[str, Cell], meta: dict[str, str]) -> list[str]:
    """Refuse an `expect=neither` cell whose zero placement is not PROVABLY Slurm confinement.

    Both arms placing 0 threads is consistent with two very different physical situations: (1)
    Slurm confined the rank and the engine deferred to it -- the state `neither` declares -- or
    (2) the rank's mask was never confined at all and the engine simply chose not to place,
    which is a DIFFERENT, unrelated null and not evidence for the layout-dependence this state
    exists to describe. Distinguishing them is the entire reason this function exists: a
    `neither` branch that only checked "both arms placed 0" would be a gate that asserts
    nothing, the exact shape of bug this project has already shipped once (a guarded test that
    passed having asserted zero assertions).

    `affinity_cpus` -- the width of `os.sched_getaffinity(0)`, i.e. the process's own CPU mask --
    is compared against `cpus_per_task` from CELL-META.tsv (`128 / RANKS_PER_NODE`, the size of
    the cgroup Slurm's `--cpus-per-task` should have handed each rank). Equal on both arms is
    the confined state; anything else means the zero was not confinement.

    IT IS THE REDUCED `affinity_cpus_min`/`_max` THAT ARE READ, NOT `affinity_cpus`. The
    unreduced key carries RANK 0's mask alone, so a cgroup that confined rank 0 and left the
    other ranks of the node wide open would satisfy this check while being precisely the
    partial failure it exists to catch -- and `min != max` IS that partial failure, so an
    uneven mask is refused separately from a uniformly-wrong one. An artifact predating the
    reduction is refused rather than silently falling back to the rank-0 key: a fallback here
    would restore the vacuous check under a name that claims otherwise.
    """
    refusals: list[str] = []
    cpus_per_task_raw = meta.get("cpus_per_task")
    try:
        want_cpus = int(cpus_per_task_raw) if cpus_per_task_raw is not None else None
    except ValueError:
        want_cpus = None
    if want_cpus is None:
        refusals.append(
            "expect=neither cannot verify Slurm confinement: CELL-META.tsv has no parseable "
            "`cpus_per_task` to compare `affinity_cpus` against, so the zero placement count "
            "is unproven as confinement rather than an unconfined rank the engine simply did "
            "not place on."
        )
        return refusals

    for side in ("main", "port"):
        pinnings = [
            cell.results["meta"]["pinning"]
            for cell in cells.values()
            if cell.side == side and cell.results.get("meta", {}).get("pinning")
        ]
        stale = [p for p in pinnings if "affinity_cpus_min" not in p]
        seen = {v for p in pinnings for k in ("affinity_cpus_min", "affinity_cpus_max")
                if (v := p.get(k)) is not None}
        print(f"- {side}: affinity_cpus (reduced min/max over ranks) = "
              f"{sorted(seen) if seen else '(none recorded)'}")
        if stale:
            refusals.append(
                f"expect=neither but {len(stale)} of {side}'s reps carry an unreduced "
                "`affinity_cpus` only (no `affinity_cpus_min`/`_max`). That key is RANK 0's "
                "mask width, so it cannot distinguish a node-wide confinement from one that "
                "confined rank 0 alone. Re-run against benches that reduce it."
            )
        elif not seen:
            refusals.append(
                f"expect=neither but {side} recorded no `affinity_cpus`, so nothing proves "
                "Slurm confined this arm rather than the arm simply not placing threads for "
                "an unrelated reason."
            )
        elif len(seen) > 1:
            refusals.append(
                f"expect=neither but {side}'s reduced affinity_cpus spans {sorted(seen)}: the "
                "ranks were NOT confined uniformly. A min below the max is a partial cgroup "
                "failure -- some ranks confined, some free -- which is the exact shape this "
                "check exists to catch, and it voids the cell rather than widening it."
            )
        elif seen != {want_cpus}:
            refusals.append(
                f"expect=neither declares Slurm confines each rank to {want_cpus} cpus "
                f"(cpus_per_task), but {side}'s affinity_cpus was {sorted(seen)}. Both-zero "
                "placement here does not prove confinement -- it is equally consistent with "
                "an unconfined rank the engine chose not to pin, which is not the state this "
                "campaign declared."
            )
    return refusals


def _check_recipe(meta: dict[str, str]) -> list[str]:
    """Print the launcher recipe and the harness revision, and refuse when they are missing.

    THE RESOLVED --cpu-bind USED TO BE RECORDED IN NO ARTIFACT: not here, not in the per-rep
    JSON meta, only in a job stdout deleted with the port worktree. The only proxy was the
    pinning counts -- which conflate the LAUNCHER's binding with the ENGINE's own affinity
    calls, the two things a placement campaign exists to tell apart. It matters: measured at
    8x16, `--cpu-bind=none` spreads a rank across four NUMA domains and costs 1.45x against
    `=cores`, and `--distribution`'s default second field is cyclic, which reproduces the
    scattered arrangement while the script still says `=cores`.

    The whole campaign rests on both arms sharing one launcher recipe, so it is printed here,
    beside the numbers it explains.
    """
    print("### Launcher recipe and harness revision\n")
    if not meta:
        print("- **no `CELL-META.tsv` in this cell directory**\n")
        return [
            "no CELL-META.tsv: the launcher recipe (--cpu-bind, --distribution, "
            "--cpus-per-task), the geometry and the harness revision that produced these "
            "numbers are unrecorded, so nothing here proves the two arms shared a recipe."
        ]
    for key in (
        "campaign",
        "harness_sha",
        "benches_sha",
        "cpu_bind",
        "distribution",
        "cpus_per_task",
        "ranks_per_node",
        "partitions",
        "nodes",
        "ntasks",
        "world",
        "layout",
        "workload",
        "size_args",
        "model_args",
        "cells",
        "reps",
    ):
        if key in meta:
            print(f"- {key}: `{meta[key] or '(none)'}`")
    print()
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        return [
            "CELL-META.tsv is missing "
            + ", ".join(f"`{k}`" for k in missing)
            + " -- this cell is not fully provenanced."
        ]
    return []


def _check_environment(cells: dict[str, Cell]) -> list[str]:
    """Refuse when the two builds' environments, or the round count, differ from the protocol."""
    refusals: list[str] = []
    for field in (
        "monoprop_max_num_modes",
        "malloc_arena_max",
        "omp_num_threads",
        "monoprop_threads",
    ):
        seen = {
            cell.results.get("meta", {}).get(field)
            for cell in cells.values()
            if cell.results.get("meta")
        }
        seen.discard(None)
        print(f"- {field}: {sorted(map(str, seen))}")
        if len(seen) > 1:
            refusals.append(f"{field} differs across cells: {sorted(map(str, seen))}")

    # 4. More than one timed round per rep doubles peak memory, because pedantic builds
    #    the next round's arguments before releasing the previous round's.
    rounds = {
        cell.results.get("params", {}).get("bench_rounds")
        for cell in cells.values()
        if cell.results.get("params")
    }
    rounds.discard(None)
    print(f"- bench_rounds: {sorted(rounds)}")
    if rounds - {1}:
        refusals.append(
            f"bench_rounds must be 1 for the memory arm, got {sorted(rounds)}"
        )

    for side in ("main", "port"):
        variants = {
            cell.results.get("meta", {}).get("monoprop_variant")
            for cell in cells.values()
            if cell.side == side and cell.results.get("meta")
        }
        variants.discard(None)
        print(f"- {side} variant: {sorted(map(str, variants))}")

    return refusals


def _check_builds(cells: dict[str, Cell]) -> list[str]:
    """Refuse when the two arms are not actually two different builds.

    The version was recorded but never asserted on, so an A/B that pointed both arms at the
    same venv -- or at two checkouts of the same commit -- printed a full table of 1.00x and
    called it flat. That is the most expensive way to fail here, because the output looks
    exactly like a real negative result. ab.sh's PORT_VENV is required and is a DIFFERENT
    branch's worktree, so getting this wrong takes only forgetting to pass it.

    Absent is not a refusal. Only present-and-equal is: an arm can legitimately record nothing
    (an older build with no such field), and refusing on absence would refuse the very
    baseline comparisons this tool exists for.

    Identity is the .so's md5, NOT ``__version__``. The version is a git describe of the
    worktree stamped into the dist-info at install time; a later ``cmake --build`` and copy
    into the venv does not rewrite it. So an arm can advertise the commit it was installed at
    while serving a binary from several commits later. Keying on the version refused a run
    whose two arms were provably distinct (different md5s, and a 3.17x graph-memory gap that
    one binary cannot produce), and it would equally have PASSED two arms that shared a .so
    across differing checkouts. The hash catches both directions; the version catches neither.

    Versions are still reported, as a checkout-level breadcrumb. They are never refused on
    when hashes are available. When no arm records a hash (artifacts older than this field),
    fall back to the version -- a weak check is better than none, and absence is not a refusal.
    """
    refusals: list[str] = []
    md5s: dict[str, set[str]] = {}
    versions: dict[str, set[str]] = {}
    for side in ("main", "port"):
        metas = [
            cell.results["meta"]
            for cell in cells.values()
            if cell.side == side and cell.results.get("meta")
        ]
        versions[side] = {str(m["monoprop_version"]) for m in metas if m.get("monoprop_version")}
        md5s[side] = {
            str(m["monoprop_core_md5"])
            for m in metas
            if m.get("monoprop_core_md5") and m["monoprop_core_md5"] != "unavailable"
        }
        print(f"- {side} version: {sorted(versions[side])}")
        print(f"- {side} _core.so md5: {sorted(md5s[side]) or ['not recorded']}")
        if len(md5s[side]) > 1:
            refusals.append(
                f"{side} arm ran {len(md5s[side])} different binaries across its reps: "
                f"{sorted(md5s[side])} -- the arm was rebuilt mid-campaign"
            )

    if md5s["main"] and md5s["port"]:
        shared = md5s["main"] & md5s["port"]
        if shared:
            refusals.append(
                f"both arms ran the same binary (md5 {sorted(shared)}); there is nothing being "
                "compared. Pass PORT_VENV explicitly -- its default points at another branch's "
                "worktree."
            )
        elif versions["main"] & versions["port"]:
            print(
                "- note: the arms share a version string but differ in md5 -- an editable "
                "install's stamp is stale, not its binary. Identity is the hash."
            )
    elif versions["main"] and versions["port"] and (versions["main"] & versions["port"]):
        # No hashes recorded anywhere: pre-md5 artifacts, so the weak check is all there is.
        refusals.append(
            f"both arms report one version ({sorted(versions['main'] & versions['port'])}) and "
            "neither records a binary hash, so they cannot be told apart. Re-run: the harness "
            "now records _core.so's md5."
        )
    return refusals


def check_provenance(cells: dict[str, Cell], meta: dict[str, str]) -> list[str]:
    """Return the reasons this run's tables must not be read as a result."""
    print("## Provenance\n")
    refusals = [
        *_check_recipe(meta),
        *_check_terms(cells),
        *_check_placement(cells, meta),
        *_check_environment(cells),
        *_check_builds(cells),
    ]
    print()
    return refusals


def main(argv: list[str]) -> int:
    """Render the summary; return non-zero when it must not be read as a result."""
    # NO FLAGS, and an unrecognised one is an error rather than an ignored word. The tool used
    # to take --allow-both-placed and filtered every leading-dash argument out of its positional
    # list, so a misspelt flag was silently dropped and the run proceeded under the other
    # posture. Everything this tool needs is now in the cell's CELL-META.tsv.
    unknown = [a for a in argv if a.startswith("-")]
    if unknown or len(argv) != 1:
        print("usage: ab_summary.py <results-dir>", file=sys.stderr)
        if unknown:
            print(
                f"  no flags are accepted; got {unknown}. The expected placement, the launcher"
                "\n  recipe and the harness revision are read from <results-dir>/CELL-META.tsv,"
                "\n  which ab.sh writes beside the measurements.",
                file=sys.stderr,
            )
        return 2
    results_dir = Path(argv[0])
    cells, problems = collect(results_dir)
    if not cells:
        print(f"no A/B artifacts found in {results_dir}", file=sys.stderr)
        return 1

    # The port branch is whatever PORT_VENV was built from, so naming one here goes stale
    # silently; the run's own directory is the thing that identifies it.
    print("# Paired A/B: main vs port\n")
    print(f"Results: `{results_dir}`\n")
    print("`port/main` below 1.00 means the port is better; a win past 10% is headlined as")
    print("its inverse (`1.14x faster`) with the raw ratio kept beside it in brackets. It is")
    print("the **median of the")
    print(
        "per-rep paired ratios**, not the ratio of the two sides' medians: the arms of"
    )
    print(
        "one rep run back to back, so a node-state swing hits both and cancels. `agree`"
    )
    print(
        "is how many of those reps point the same way as the median, and the operation is"
    )
    print(f"resolved when their two-sided sign-test p is below {P_RESOLVE:.2f} -- not when every")
    print("rep agrees, which gets harder as reps rise. The med and min columns are per-side")
    print("figures.\n")
    print(
        "**Memory is the peak-RSS table under each operation table**, from `/usr/bin/time -v`"
    )
    print(
        "around every rank -- kernel truth, produced outside the code under test. Its `sum`"
    )
    print(
        "column is the node total and is the only memory figure comparable across layouts."
    )
    print(
        "The engine's own byte ledger follows it, labelled as a secondary diagnostic: it has"
    )
    print("been wrong by up to 23x in both directions against the kernel.\n")

    meta = read_meta(results_dir)
    rss = read_peak_rss(results_dir)
    refusals = check_provenance(cells, meta)
    flagged: list[str] = []
    for key in sorted({cell.table_key for cell in cells.values()}):
        emit_table(key, [c for c in cells.values() if c.table_key == key], rss, flagged)

    print("## Flagged\n")
    if flagged:
        print("\n".join(flagged))
        print(
            f"\nRead `agree` before the ratio. A ratio whose sign-test p is below "
            f"{P_RESOLVE:.2f} is a result; one carrying an **unresolved** line above is a "
            "statement about the machine, and the fix is more reps."
        )
    else:
        print(f"Nothing outside {IMPROVE:.2f}x..{REGRESS:.2f}x.")
    print()

    if problems:
        print("## Unreadable artifacts\n")
        print("\n".join(f"- {p}" for p in problems))
        print()
    if refusals:
        print("## REFUSED\n")
        print("\n".join(f"- {r}" for r in refusals))
        print("\nThe tables above are a diagnostic, NOT a result.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
