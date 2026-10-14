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

"""Check the Fig. 3 campaign's per-point JSONL files and merge them into data/.

A timing run directory (``RUN_DIR`` of ``scripts/slurm/campaign.sbatch``) holds ``nsweep/``
with one ``<engine>_L<layers>_N<n>.jsonl`` per point (``<engine>_N<n>.jsonl`` in campaigns
that ran one depth); a counting run directory (``scripts/slurm/count.sbatch``) holds ``count/``.
Rows are keyed by the depth they record, not by their file name, and a row at another cutoff
than the figure's (``CUTOFF``) belongs to an earlier version of the workload and is left out.
When run directories hold the same point, the one given last wins: list a rerun after the
campaign it repeats. They merge into:

    row        engines (file prefix)                 data/ file
    pauli      monoprop, julia, ppvm                 pauli.jsonl      (panel a)
    majorana   monoprop-majorana, julia-majorana     majorana.jsonl   (panel b)
    counts     count_anticommuting.py                counts.jsonl     (panel c)
    held       count_held.py                         held.jsonl       (the cost denominators)

Each (row, depth) is checked as one dataset. Refused on failure: more than one row per engine,
depth and N; an engine field that does not match its file; cutoff, lower_atol, observable or
basis differing within a dataset; any row not single-threaded; PauliPropagation.jl not at
v0.9.0 or MajoranaPropagation.jl not at v0.4.1; and, between the engines at one point, a
different term or gate count, or expectations further apart than 1e-9. A point killed at its
time limit is a ``timeout`` row, drawn as a lower bound; an ``implied`` timeout row was not run,
because the engine already exceeded the limit on a shallower circuit at the same N. Every
engine must reach every N of its dataset's grid (``grid``), as a row or a timeout row.

Every point is held to one time limit, ``CEILING_S``. A row measured under a shorter one is
kept only if it is ``settled``: the repetition rule would have produced the same row under
``CEILING_S`` (see :func:`settled`). A timeout under a shorter limit is not settled.

A count must report the same number of terms as monoprop's timed row at its point, and every
point monoprop finished needs one, unless it holds more than ``COUNT_MAX_TERMS`` terms, past
which the counter does not fit a node. A held count (the terms the operator holds when each gate
is applied, summed over the run) must likewise match monoprop's term count, and every point
monoprop finished needs one: the cost panels divide by it.

    python merge_octave_runs.py ../../../runs/fig3/campaign-<jobid> ... ../../../runs/fig3/count-<jobid>

``--dry-run`` writes nothing. ``--allow-partial`` turns missing points into notes, for a
campaign still in flight, and must write outside data/ (``--outdir``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Imported by make_figure.py, so the merge and the figure cannot disagree on names.
ROWS = {
    "pauli": ("pauli.jsonl", ("monoprop", "julia", "ppvm")),
    "majorana": ("majorana.jsonl", ("monoprop-majorana", "julia-majorana")),
}
COUNTS = "counts.jsonl"
HELD = "held.jsonl"
# Small, medium and large, in layers (a Hubbard layer is one whole Trotter step). At the
# Hubbard support cutoff 3, three to five steps hold about as many terms per mode at the top
# of the grid as 7 to 13 kicked-Ising layers, and monoprop finishes every point.
DEPTHS = {"pauli": (7, 10, 13), "majorana": (3, 4, 5)}
NS = (32, 64, 128, 256, 512, 1024)
# The Hubbard lattices are square, L x L with N = 2 L^2 modes: 4, 6, 8, 11, 16 and 22 sites a
# side, about doubling N each step under the 1024-mode build.
HUBBARD_NS = (32, 72, 128, 242, 512, 968)
HUBBARD_LATTICE = {2 * side * side: (side, side) for side in (4, 6, 8, 11, 16, 22)}
GRID = {"pauli": NS, "majorana": HUBBARD_NS}
# (row, depth) -> largest N, for a depth that does not fit a node at the top of its grid.
MAX_N: dict[tuple[str, int], int] = {}
# (cutoff, cutoff_type, boundary): Pauli weight on the open chain; Hubbard support (modes
# touched) or length (Majoranas), fully paired monomials kept under either, on an open lattice
# or a torus.
CUTOFF = {"pauli": (6, None, None), "majorana": (3, "support", "open")}
# Timed propagation allowed per point, the same for every engine (octave_point's CEILING_S).
CEILING_S = 1800
# octave_point's ROUND_BUDGET_S: rounds continue while the timed total is under it.
ROUND_BUDGET_S = 60
# ~1 KB per term at the counter's peak, against a 223 GiB node.
COUNT_MAX_TERMS = 2e8
# The engine field each driver writes, by the file prefix the campaign gives it.
ENGINE_FIELD = {
    "monoprop": "monoprop",
    "monoprop-majorana": "monoprop",
    "julia": "julia_pauli",
    "julia-majorana": "julia_majorana",
    "ppvm": "ppvm",
}
LIBRARY_VERSION = {"julia": "0.9.0", "julia-majorana": "0.4.1"}
EXPECTATION_TOL = 1e-9
# Must be the same on every row of a dataset.
WORKLOAD_KEYS = ("cutoff", "lower_atol", "observable", "basis")
FILE_NAME = re.compile(r"^(?P<engine>.+?)_(?:L\d+_)?N[\d-]+\.jsonl$")


def grid(row_kind, layers):
    """The N values of one (row, depth) dataset."""
    ns = GRID[row_kind]
    cap = MAX_N.get((row_kind, layers), max(ns))
    return tuple(n for n in ns if n <= cap)


def settled(row, ceiling=CEILING_S):
    """Whether a finished row would be the same under ``ceiling`` as under its own limit.

    The repetition rule (``keep_timing`` in the drivers) reads the limit only to refuse a round
    that would end past it. So a row is unaffected by a longer limit when its limit never
    refused a round: it ran ``max_rounds``, or stopped on the ``ROUND_BUDGET_S`` budget (two
    rounds or more, totalling at least the budget), or ran far inside its own limit, or its one
    round is longer than half of ``ceiling``, so that no second round fits under ``ceiling``
    either. A row whose one round already ran past ``ceiling`` is the same bound under both.
    A timeout, run or implied, settles a point only under the limit it was recorded at.
    """
    if row.get("timeout"):
        return row.get("ceiling_seconds") == ceiling
    # Past the limit in its first round, a row is the same bound under either limit.
    if row.get("time_limit") == ceiling or row["seconds"] > ceiling:
        return True
    rounds, t = row["rounds"], row["seconds"]
    # ``seconds`` is the fastest round; a fresh process's first round can take ~2x as long, so
    # the one more round a limit could have refused ends by 2 * (rounds + 1) * t.
    return (
        rounds >= row["max_rounds"]
        or (rounds >= 2 and rounds * t >= ROUND_BUDGET_S)
        or (rounds >= 2 and 2 * (rounds + 1) * t <= row["time_limit"])
        or (rounds == 1 and 2 * t > ceiling)
    )


def cutoff_of(row):
    """(cutoff, cutoff_type, boundary) of a row.

    Hubbard rows from before a field are support-truncated, on an open lattice.
    """
    kind, boundary = row.get("cutoff_type"), row.get("boundary")
    if row["basis"] == "majorana":
        kind, boundary = kind or "support", boundary or "open"
    return row["cutoff"], kind, boundary


def on_lattice(row):
    """Whether a Hubbard row is on the figure's square lattice for its N (Pauli rows always are).

    Earlier campaigns ran 2:1 lattices (8 x 4, 16 x 8, 32 x 16) at the N between the squares;
    their rows are another workload. A timeout row names no lattice, so its N alone decides.
    """
    if row["basis"] != "majorana":
        return True
    want = HUBBARD_LATTICE.get(row["num_qubits"])
    return want is not None and (
        row.get("lattice") is None or tuple(row["lattice"]) == want
    )


def _read(path):
    return [
        (raw.strip(), json.loads(raw))
        for raw in path.read_text().splitlines()
        if raw.strip()
    ]


def _take(points, notes, rank, f, engine, cutoff):
    """Add one file's rows to ``points`` under ``gather``'s rule; the rows left out by cutoff
    or lattice."""
    other_cutoff = 0
    for raw, row in _read(f):
        if cutoff_of(row) != cutoff or not on_lattice(row):
            other_cutoff += 1
            continue
        key = (engine, row["layers"], row["num_qubits"])
        held = points.get(key)
        if held is None or held[0] < rank:
            if held is not None:
                notes.append(
                    f"{engine} depth {key[1]} N={key[2]}: {f} replaces {held[1][0][3]}"
                )
            points[key] = (rank, [])
        if points[key][0] == rank:
            points[key][1].append((row["num_qubits"], raw, row, f))
    return other_cutoff


def gather(run_dirs, engines, cutoff):
    """({(engine, layers): [(N, raw line, row, source)]}, notes) from every run's nsweep/.

    Rows at another cutoff are dropped; a point found in a later run directory replaces the
    same point from an earlier one (duplicates within one directory are kept, and refused).
    """
    points, notes, other_cutoff = {}, [], 0
    for rank, run_dir in enumerate(run_dirs):
        for f in sorted((run_dir / "nsweep").glob("*.jsonl")):
            m = FILE_NAME.match(f.name)
            if m and m["engine"] in engines:
                other_cutoff += _take(points, notes, rank, f, m["engine"], cutoff)
    if other_cutoff:
        notes.append(
            f"left out {other_cutoff} rows at a cutoff other than {cutoff} or another lattice"
        )
    out = {}
    for (engine, layers, _), (_, entries) in points.items():
        out.setdefault((engine, layers), []).extend(entries)
    return out, notes


def _row_problems(engine, n, row, src):
    """What is wrong with one row on its own."""
    out = []
    if row["engine"] != ENGINE_FIELD[engine]:
        out.append(f"{engine}: N={n} in {src} has engine {row['engine']!r}")
    if row.get("timeout"):
        if not row.get("implied") and row["ceiling_seconds"] != CEILING_S:
            out.append(
                f"{engine}: N={n} timed out at {row['ceiling_seconds']} s, not {CEILING_S}"
            )
        return out
    if not settled(row):
        out.append(
            f"{engine}: N={n} measured under a {row.get('time_limit')} s limit and not settled "
            f"under {CEILING_S} s (rounds={row['rounds']}, {row['seconds']:.0f} s): rerun"
        )
    if str(row["num_threads"]) != "1":
        out.append(f"{engine}: N={n} num_threads={row['num_threads']!r}")
    want = LIBRARY_VERSION.get(engine)
    if want and row.get("library_version") != want:
        out.append(
            f"{engine}: N={n} library_version {row.get('library_version')!r}, want {want}"
        )
    return out


def check_dataset(ns, by_engine, *, allow_partial):
    """(problems, notes) for one (row, depth) dataset, {engine: rows}."""
    problems, notes = [], []
    partial = notes if allow_partial else problems
    for engine, rows in by_engine.items():
        by_n = {}
        for n, _, row, src in rows:
            by_n.setdefault(n, []).append(src)
            problems += _row_problems(engine, n, row, src)
        problems += [
            f"{engine}: {len(srcs)} rows at N={n} ({', '.join(map(str, srcs))})"
            for n, srcs in sorted(by_n.items())
            if len(srcs) > 1
        ]
        if missing := [n for n in ns if n not in by_n]:
            partial.append(f"{engine}: missing N={missing}")

    done = [
        row
        for rows in by_engine.values()
        for *_, row, _ in rows
        if not row.get("timeout")
    ]
    for key in WORKLOAD_KEYS:
        vals = sorted({json.dumps(r.get(key)) for r in done})
        if len(vals) > 1:
            problems.append(f"{key} differs across rows: {', '.join(vals)}")
    for n in ns:
        at = {
            engine: row
            for engine, rows in by_engine.items()
            for m, _, row, _ in rows
            if m == n and not row.get("timeout")
        }
        if len(at) > 1:
            problems += _agreement_problems(n, at)
    return problems, notes


def _agreement_problems(n, at):
    """What differs between the engines' rows {engine: row} at one N."""
    problems = []
    for key in ("num_terms", "gates"):
        vals = {engine: json.dumps(row.get(key)) for engine, row in at.items()}
        if len(set(vals.values())) != 1:
            detail = ", ".join(f"{e}={v[:80]}" for e, v in vals.items())
            problems.append(f"N={n}: {key} differs -- {detail}")
    exps = [row["expectation"] for row in at.values()]
    if max(exps) - min(exps) > EXPECTATION_TOL:
        problems.append(f"N={n}: expectation spread {max(exps) - min(exps):.3e}")
    return problems


def _summary(engine, rows):
    ns = sorted(n for n, *_ in rows)
    timeouts = sorted(
        n for n, _, row, _ in rows if row.get("timeout") and not row.get("implied")
    )
    implied = sorted(n for n, _, row, _ in rows if row.get("implied"))
    rounds = sorted(
        {str(row.get("rounds")) for _, _, row, _ in rows if not row.get("timeout")}
    )
    hosts = sorted({row["host"].split(".")[0] for _, _, row, _ in rows})
    return (
        f"    {engine:<18} N={ns}  timeouts={timeouts}  implied={implied}  "
        f"rounds={'/'.join(rounds)}  hosts={', '.join(hosts)}"
    )


def _stray(rows, engines, layers, ns):
    """The points of one depth outside its N grid."""
    return sorted(
        {
            f"{e} N={t[0]}"
            for e in engines
            for t in rows.get((e, layers), [])
            if t[0] not in ns
        }
    )


def _collect(by_engine, engines, layers, lines, terms):
    """Append one depth's raw lines in engine and N order, and record monoprop's term counts."""
    for e in engines:
        for n, raw, row, _ in sorted(by_engine.get(e, []), key=lambda t: t[0]):
            lines.append(raw)
            if e.startswith("monoprop") and not row.get("timeout"):
                terms[(layers, n)] = row["num_terms"]


def merge_row(args, row_kind, engines):
    """(problems, merged lines, {(layers, N): monoprop terms}) for one row of the figure."""
    rows, gather_notes = gather(args.run_dirs, engines, CUTOFF[row_kind])
    for note in gather_notes:
        print(f"  {row_kind}: {note}")
    problems, lines, terms = [], [], {}
    for layers in DEPTHS[row_kind]:
        ns = grid(row_kind, layers)
        by_engine = {
            e: [t for t in rows.get((e, layers), []) if t[0] in ns] for e in engines
        }
        by_engine = {e: r for e, r in by_engine.items() if r}
        print(f"{row_kind} at depth {layers}:")
        stray = _stray(rows, engines, layers, ns)
        if stray:
            problems.append(
                f"depth {layers}: rows outside the grid {ns}: {', '.join(stray)}"
            )
        if not by_engine:
            (
                print("  note: no rows")
                if args.allow_partial
                else problems.append(f"depth {layers}: no rows")
            )
            continue
        for engine, r in by_engine.items():
            print(_summary(engine, r))
        p, notes = check_dataset(ns, by_engine, allow_partial=args.allow_partial)
        problems += [f"depth {layers}: {x}" for x in p]
        for note in notes:
            print(f"  note: {note}")
        _collect(by_engine, engines, layers, lines, terms)
    # Earlier campaigns measured other depths too; they stay in their run directories.
    extra = sorted({layers for _, layers in rows} - set(DEPTHS[row_kind]))
    if extra:
        print(
            f"  {row_kind}: left out rows at depths outside {DEPTHS[row_kind]}: {extra}"
        )
    return problems, lines, terms


def gather_counts(run_dirs, subdir="count"):
    """{(basis, layers, N): [(raw line, row, source)]} from every run's ``subdir``/, at ``CUTOFF``.

    A point found in a later run directory replaces the same point from an earlier one.
    """
    seen, rank_of = {}, {}
    for rank, run_dir in enumerate(run_dirs):
        for f in sorted((run_dir / subdir).glob("*.jsonl")):
            for raw, row in _read(f):
                # As for timed rows, other cutoffs and depths are earlier workloads.
                if (
                    cutoff_of(row) != CUTOFF[row["basis"]]
                    or not on_lattice(row)
                    or row["layers"] not in DEPTHS[row["basis"]]
                ):
                    continue
                key = (row["basis"], row["layers"], row["num_qubits"])
                # As for timed rows, a later run directory replaces an earlier one.
                if rank_of.setdefault(key, rank) < rank:
                    rank_of[key] = rank
                    seen.pop(key, None)
                seen.setdefault(key, []).append((raw, row, f))
    return seen


def merge_counts(args, timed):
    """(problems, lines) of the anticommutation counts, checked against the timed term counts."""
    problems, notes, lines = [], [], []
    partial = notes if args.allow_partial else problems
    seen = gather_counts(args.run_dirs)
    for (basis, layers, n), found in sorted(seen.items()):
        where = f"{basis} depth {layers} N={n}"
        if len(found) > 1:
            problems.append(f"{where}: {len(found)} count rows")
        raw, row, _ = found[0]
        want = timed.get(basis, {}).get((layers, n))
        if want is None:
            partial.append(f"{where}: count with no finished monoprop row")
        elif row["num_terms"] != want:
            problems.append(
                f"{where}: count saw {row['num_terms']} terms, timed {want}"
            )
        lines.append(raw)
    uncounted = [
        (basis, layers, n, terms)
        for basis, points in timed.items()
        for (layers, n), terms in sorted(points.items())
        if (basis, layers, n) not in seen
    ]
    for basis, layers, n, terms in uncounted:
        if terms > COUNT_MAX_TERMS:
            print(
                f"  counts: {basis} depth {layers} N={n} left out ({terms:.2e} terms)"
            )
        else:
            partial.append(f"{basis} depth {layers} N={n}: no count")
    print(f"counts: {len(lines)} rows")
    for note in notes:
        print(f"  note: {note}")
    return problems, lines


def merge_held(args, timed):
    """(problems, lines) of the held-term counts: one per point monoprop finished, same K."""
    problems, notes, lines = [], [], []
    partial = notes if args.allow_partial else problems
    seen = gather_counts(args.run_dirs, "held")
    for (basis, layers, n), found in sorted(seen.items()):
        where = f"{basis} depth {layers} N={n}"
        if len(found) > 1:
            problems.append(f"{where}: {len(found)} held rows")
        raw, row, _ = found[0]
        want = timed.get(basis, {}).get((layers, n))
        if want is None:
            partial.append(f"{where}: held count with no finished monoprop row")
        elif row["num_terms"] != want:
            problems.append(
                f"{where}: held count ended at {row['num_terms']} terms, timed {want}"
            )
        lines.append(raw)
    partial += [
        f"{basis} depth {layers} N={n}: no held count"
        for basis, points in timed.items()
        for layers, n in sorted(points)
        if (basis, layers, n) not in seen
    ]
    print(f"held: {len(lines)} rows")
    for note in notes:
        print(f"  note: {note}")
    return problems, lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run_dirs", nargs="+", type=Path, help="campaign and count RUN_DIRs"
    )
    ap.add_argument("--outdir", type=Path, default=DATA)
    ap.add_argument("--dry-run", action="store_true", help="check and report only")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="missing points become notes; for previews, never into data/",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    if args.allow_partial and not args.dry_run and outdir == DATA:
        raise SystemExit(
            "--allow-partial output is a preview: pass --outdir outside data/"
        )

    all_problems, writes, timed = [], {}, {}
    for row_kind, (name, engines) in ROWS.items():
        problems, lines, timed[row_kind] = merge_row(args, row_kind, engines)
        all_problems += [f"{row_kind}: {p}" for p in problems]
        writes[outdir / name] = lines
    problems, lines = merge_counts(args, timed)
    all_problems += [f"counts: {p}" for p in problems]
    writes[outdir / COUNTS] = lines
    problems, lines = merge_held(args, timed)
    all_problems += [f"held: {p}" for p in problems]
    writes[outdir / HELD] = lines
    if all_problems:
        print("refused:", *all_problems, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)
    print(
        "  checks passed: one row per point, identical terms/gates/expectation, 1 thread"
    )

    clobber = [p for p in writes if p.exists()]
    if clobber and not args.force and not args.dry_run:
        raise SystemExit(
            f"exists (pass --force to overwrite): {', '.join(map(str, clobber))}"
        )
    print("\nwould write:" if args.dry_run else "\nwrote:")
    for path, lines in writes.items():
        if not args.dry_run:
            outdir.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n")
        print(f"  {path}  ({len(lines)} rows)")


if __name__ == "__main__":
    main()
