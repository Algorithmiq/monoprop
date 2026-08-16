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

"""Paired time+memory summary for ab-100m.sh.

Reads the two artifacts an interleaved run drops per cell -- pytest-benchmark's
`time-<label>.json` and the suite's own `<label>.json` -- and reports, per node count and
operation, each side's median across repetitions plus the port/main ratio for time and
for memory.

The headline ratio is the median of the PER-REP PAIRED ratios, not the ratio of the two
sides' medians. The two arms of a rep run back to back on the same node, so a node-state
swing hits both and cancels in the quotient; taking medians per side first throws that
pairing away and lets a swing that touched only one arm survive into the result. The
pairing is the entire reason ab-100m.sh interleaves, and computing the ratio the obvious
way spends it for nothing.

`agree` -- how many reps point the same way as their median -- is reported next to every
ratio, because the ratio alone cannot distinguish "flat" from "unresolvable". Median
across reps rather than mean, for the same reason: an interleaved run is deliberately
exposed to whatever else is in the allocation, and one slow rep should move nothing.

The Provenance section is not decoration. It exits non-zero when the two arms did not do
the same work, or when the run is void because thread placement failed -- states that
otherwise print as a clean table full of meaningless ratios.

  python hpc/deucalion/tools/ab_summary.py <results-dir> [--allow-both-placed]
"""

from __future__ import annotations

import json
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
# The model segment is optional so the random-problem runs (ab-100m.sh) keep parsing
# unchanged; models-ab.sh emits it, and it becomes part of the table key. Without it every
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
        self.times: dict[str, dict[str, float]] = {}
        self.results: dict[str, dict] = {}

    @property
    def table_key(self) -> tuple[int, str, str]:
        """The table this cell belongs in. Cells from different keys must not be pooled."""
        return (self.nodes, self.layout, self.model)


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


def agreement(ratios: list[float]) -> tuple[float | None, int, int]:
    """Return (median ratio, reps agreeing with its direction, reps compared)."""
    if not ratios:
        return None, 0, 0
    med = statistics.median(ratios)
    same = sum(1 for r in ratios if (r >= 1.0) == (med >= 1.0))
    return med, same, len(ratios)


def fmt_ms(seconds: float | None) -> str:
    """Render seconds as milliseconds, or a dash."""
    return "-" if seconds is None else f"{seconds * 1e3:.1f}"


def fmt_gib(value: float | None) -> str:
    """Render bytes as GiB, or a dash."""
    return "-" if value is None else f"{value / GIB:.2f}"


def verdict(ratio: float | None) -> str:
    """Render a port/main ratio as a human verdict."""
    if ratio is None or ratio <= 0:
        return "-"
    if ratio >= REGRESS:
        return f"**{ratio:.2f}x slower**"
    if ratio <= IMPROVE:
        return f"{1 / ratio:.2f}x faster"
    return f"{ratio:.2f}x (flat)"


def ratio_of(port: float | None, main: float | None) -> float | None:
    """Return ``port / main`` when both exist and ``main`` is non-zero."""
    if port is None or main is None or not main:
        return None
    return port / main


def emit_table(
    key: tuple[int, str, str], cells: list[Cell], flagged: list[str]
) -> None:
    """Print one (nodes, layout, model) operation table."""
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
        "main dmem (GiB)",
        "port dmem (GiB)",
        "dmem ratio",
        "op+graph (GiB)",
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
                "bytes": medians(
                    [
                        sum(entry.values())
                        for entry in (
                            c.results.get("opbytes", {}).get(op) for c in side_cells
                        )
                        if isinstance(entry, dict)
                    ]
                ),
            }
        main, port = stats["main"], stats["port"]
        time_ratio, agree, compared = agreement(paired_ratios(by_side, op, "min"))
        mem_ratio = None
        if min(main["dmem"] or 0, port["dmem"] or 0) >= MEM_FLOOR:
            mem_ratio = ratio_of(port["dmem"], main["dmem"])

        # One column, not two: both arms propagate the same operator, so a disagreement
        # here is a provenance failure rather than a result. It is checked below.
        sizes = [s["bytes"] for s in (main, port) if s["bytes"] is not None]
        print(
            f"| {short_op(op)} | {fmt_ms(main['median'])} | {fmt_ms(port['median'])} "
            f"| {verdict(time_ratio)} | {agree}/{compared} "
            f"| {fmt_ms(main['min'])} | {fmt_ms(port['min'])} "
            f"| {fmt_gib(main['dmem'])} | {fmt_gib(port['dmem'])} "
            f"| {verdict(mem_ratio)} | {fmt_gib(max(sizes) if sizes else None)} "
            f"| {main['reps']}/{port['reps']} |"
        )

        # Reps that do not agree on a direction are the loudest thing the table can say,
        # and the ratio column cannot say it: a split 2/4 still prints some number. Flag
        # it on its own so an unresolvable operation is never read as a measured null.
        #
        # Disagreement alone is not enough to fire, though. An operation that is genuinely
        # flat scatters its reps either side of 1.00 by construction, so direction alone
        # would flag every null result as unresolved and drown the real ones. Require the
        # spread to be wider than the effects being claimed: reps landing in 0.93..1.01
        # agree that nothing happened, while 0.41..2.44 agree on nothing at all.
        ratios = paired_ratios(by_side, op, "min")
        spread = max(ratios) / min(ratios) if ratios and min(ratios) > 0 else 1.0
        if compared and agree < compared and spread > REGRESS + 0.15:
            flagged.append(
                f"- `{where}` / `{short_op(op)}` / time: **unresolved** -- only "
                f"{agree} of {compared} reps agree on a direction, spread {spread:.1f}x "
                f"(per-rep {', '.join(f'{r:.2f}' for r in ratios)})"
            )

        for name, ratio, extra in (
            ("time", time_ratio, f"paired, {agree}/{compared} reps agree"),
            ("memory", mem_ratio, "peak above the operation's own floor"),
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


def _check_placement(cells: dict[str, Cell], *, allow_both_placed: bool) -> list[str]:
    """Refuse when thread placement makes the timings meaningless.

    The engine's own COMMPROF `pinned=` field exists only on builds that have
    monoprop_COMM_PROFILE, so it cannot be compared across the boundary that added it;
    this comes from /proc and works identically on both arms.
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

    if placed["main"] == 0 and placed["port"] == 0:
        refusals.append(
            "neither arm placed a single thread: every rank ran unpinned, so this "
            "measures two unplaced builds and not the change"
        )
    elif placed["main"] and placed["port"] and not allow_both_placed:
        refusals.append(
            f"both arms placed threads (main={placed['main']}, port={placed['port']}); "
            "the baseline is not expected to place at this layout, so a venv is probably "
            "not the build it is labelled as. Pass --allow-both-placed if this is real"
        )
    elif None in placed.values():
        refusals.append("placement was not recorded on at least one arm")

    return refusals


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
    exactly like a real negative result. models-ab.sh's PORT_VENV default is a DIFFERENT
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


def check_provenance(cells: dict[str, Cell], *, allow_both_placed: bool) -> list[str]:
    """Return the reasons this run's tables must not be read as a result."""
    print("## Provenance\n")
    refusals = [
        *_check_terms(cells),
        *_check_placement(cells, allow_both_placed=allow_both_placed),
        *_check_environment(cells),
        *_check_builds(cells),
    ]
    print()
    return refusals


def main(argv: list[str]) -> int:
    """Render the summary; return non-zero when it must not be read as a result."""
    args = [a for a in argv if not a.startswith("-")]
    if not args:
        print(
            "usage: ab_summary.py <results-dir> [--allow-both-placed]", file=sys.stderr
        )
        return 2
    results_dir = Path(args[0])
    cells, problems = collect(results_dir)
    if not cells:
        print(f"no A/B artifacts found in {results_dir}", file=sys.stderr)
        return 1

    # The port branch is whatever PORT_VENV was built from, so naming one here goes stale
    # silently; the run's own directory is the thing that identifies it.
    print("# Paired A/B: main vs port\n")
    print(f"Results: `{results_dir}`\n")
    print("`port/main` below 1.00 means the port is better. It is the **median of the")
    print(
        "per-rep paired ratios**, not the ratio of the two sides' medians: the arms of"
    )
    print(
        "one rep run back to back, so a node-state swing hits both and cancels. `agree`"
    )
    print(
        "is how many of those reps point the same way as the median -- anything short of"
    )
    print("all of them means the operation is unresolved at this rep count, however")
    print("confident the ratio looks. The med and min columns are the raw per-side")
    print(
        "figures. `dmem` is the peak RSS the operation added above its own floor, on the"
    )
    print(
        "worst rank -- read it next to `op+graph`, the operator and graph it walked.\n"
    )

    refusals = check_provenance(cells, allow_both_placed="--allow-both-placed" in argv)
    flagged: list[str] = []
    for key in sorted({cell.table_key for cell in cells.values()}):
        emit_table(key, [c for c in cells.values() if c.table_key == key], flagged)

    print("## Flagged\n")
    if flagged:
        print("\n".join(flagged))
        print(
            "\nRead `agree` before the ratio. A ratio backed by all reps is a result; one "
            "carrying an **unresolved** line above is a statement about the machine, and "
            "the fix is more reps, not a smaller threshold."
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
