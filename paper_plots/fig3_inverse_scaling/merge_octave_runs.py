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

A campaign run directory (``RUN_DIR`` of ``scripts/slurm/campaign.sbatch``) holds
``nsweep/`` and ``perlayer/``, each with one ``<engine>_N<n>.jsonl`` per point. They merge
into one file per panel source:

    dataset             panels   engines (file prefix)                 data/ file
    pauli/nsweep        (a)      monoprop, julia, ppvm                 pauli_nsweep.jsonl
    pauli/perlayer      (b)      monoprop, julia                       pauli_perlayer.jsonl
    majorana/nsweep     (c)      monoprop-majorana, julia-majorana     majorana_nsweep.jsonl
    majorana/perlayer   (d)      monoprop-majorana, julia-majorana     majorana_perlayer.jsonl

Refused on failure: more than one row per engine and N; an engine field that does not match
its file; layers, cutoff, lower_atol or observable differing within a dataset; any row not
single-threaded; PauliPropagation.jl not at v0.9.0 or MajoranaPropagation.jl not at v0.4.1;
and, between the engines at one N, a different term count, gate count or (per layer) term
sequence, or expectations further apart than 1e-9. A point killed at its time limit is a
``timeout`` row: allowed in an N sweep, where the figure draws it as a lower bound, but not
in a per-layer sweep, whose rows must be complete. Every engine must reach every N of its
dataset's grid (``GRID``), as a row or a timeout row; rows at other N are reported and left out.

    python merge_octave_runs.py ../../../runs/fig3/campaign-<jobid>

``--dry-run`` writes nothing. ``--allow-partial`` turns missing points into notes, for a
campaign still in flight, and must write outside data/ (``--outdir``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Imported by make_figure.py, so the merge and the figure cannot disagree on names.
DATASETS = {
    ("pauli", "nsweep"): ("pauli_nsweep.jsonl", ("monoprop", "julia", "ppvm")),
    ("pauli", "perlayer"): ("pauli_perlayer.jsonl", ("monoprop", "julia")),
    ("majorana", "nsweep"): (
        "majorana_nsweep.jsonl",
        ("monoprop-majorana", "julia-majorana"),
    ),
    ("majorana", "perlayer"): (
        "majorana_perlayer.jsonl",
        ("monoprop-majorana", "julia-majorana"),
    ),
}
# The engine field each driver writes, by the file prefix the campaign gives it.
ENGINE_FIELD = {
    "monoprop": "monoprop",
    "monoprop-majorana": "monoprop",
    "julia": "julia_pauli",
    "julia-majorana": "julia_majorana",
    "ppvm": "ppvm",
}
LIBRARY_VERSION = {"julia": "0.9.0", "julia-majorana": "0.4.1"}
NSWEEP_GRID = (32, 64, 128, 256, 512, 1024)
# Per-layer sweeps need every engine to finish, so the top N is the largest that the
# reference completes within the time limit: MajoranaPropagation.jl does not at N=512.
GRID = {
    ("pauli", "nsweep"): NSWEEP_GRID,
    ("majorana", "nsweep"): NSWEEP_GRID,
    ("pauli", "perlayer"): (32, 128, 512),
    ("majorana", "perlayer"): (32, 128, 256),
}
EXPECTATION_TOL = 1e-9
# Must be the same on every row of a dataset.
WORKLOAD_KEYS = ("cutoff", "layers", "lower_atol", "observable", "basis")


def _gather(run_dirs, sweep, engine):
    """[(N, raw line, row, source)] for one engine of one sweep."""
    out = []
    for run_dir in run_dirs:
        for f in sorted((run_dir / sweep).glob(f"{engine}_N*.jsonl")):
            for raw in f.read_text().splitlines():
                if raw.strip():
                    row = json.loads(raw)
                    out.append((row["num_qubits"], raw.strip(), row, f))
    return out


def _collect(run_dirs, sweep, engines, grid):
    """({engine: rows on the grid}, ["engine N=n" for each row off it])."""
    by_engine, ignored = {}, []
    for e in engines:
        rows = _gather(run_dirs, sweep, e)
        ignored += sorted({f"{e} N={n}" for n, *_ in rows if n not in grid})
        if rows := [t for t in rows if t[0] in grid]:
            by_engine[e] = rows
    return by_engine, ignored


def _row_problems(engine, n, row, src):
    """What is wrong with one row on its own."""
    out = []
    if row["engine"] != ENGINE_FIELD[engine]:
        out.append(f"{engine}: N={n} in {src} has engine {row['engine']!r}")
    if row.get("timeout"):
        return out
    if str(row["num_threads"]) != "1":
        out.append(f"{engine}: N={n} num_threads={row['num_threads']!r}")
    want = LIBRARY_VERSION.get(engine)
    if want and row.get("library_version") != want:
        out.append(
            f"{engine}: N={n} library_version {row.get('library_version')!r}, want {want}"
        )
    return out


def _engine_problems(sweep, grid, engine, rows):
    """(problems, missing N) for one engine's rows of one dataset."""
    problems, by_n = [], {}
    for n, _, row, src in rows:
        by_n.setdefault(n, []).append(src)
        problems += _row_problems(engine, n, row, src)
        if row.get("timeout") and sweep == "perlayer":
            problems.append(f"{engine}: N={n} timed out in the per-layer sweep")
    problems += [
        f"{engine}: {len(srcs)} rows at N={n}"
        for n, srcs in sorted(by_n.items())
        if len(srcs) > 1
    ]
    stray = sorted(set(by_n) - set(grid))
    if stray:
        problems.append(f"{engine}: N outside the {sweep} grid: {stray}")
    return problems, [n for n in grid if n not in by_n]


def check_dataset(sweep, grid, by_engine, *, allow_partial):
    """(problems, notes) for one dataset, {engine: rows}."""
    problems, notes = [], []
    partial = notes if allow_partial else problems
    for engine, rows in by_engine.items():
        engine_problems, missing = _engine_problems(sweep, grid, engine, rows)
        problems += engine_problems
        if missing:
            partial.append(f"{engine}: missing N={missing}")

    done = [
        row
        for rows in by_engine.values()
        for _, _, row, _ in rows
        if not row.get("timeout")
    ]
    for key in WORKLOAD_KEYS:
        vals = sorted({json.dumps(r.get(key)) for r in done})
        if len(vals) > 1:
            problems.append(f"{key} differs across rows: {', '.join(vals)}")
    for n in grid:
        at = {
            engine: row
            for engine, rows in by_engine.items()
            for m, _, row, _ in rows
            if m == n and not row.get("timeout")
        }
        if len(at) > 1:
            problems += _agreement_problems(sweep, n, at)
    return problems, notes


def _agreement_problems(sweep, n, at):
    """What differs between the engines' rows {engine: row} at one N."""
    problems = []
    keys = ["num_terms", "gates"]
    if sweep == "perlayer":
        keys += [
            "gates_per_layer",
            "initial_terms",
            "layer_terms",
            "step_gates",
            "step_terms",
        ]
    for key in keys:
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
    timeouts = sorted(n for n, _, row, _ in rows if row.get("timeout"))
    hosts = sorted({row["host"] for _, _, row, _ in rows})
    versions = sorted(
        {
            str(row.get("library_version"))
            for _, _, row, _ in rows
            if not row.get("timeout")
        }
    )
    rounds = sorted({str(row.get("rounds")) for _, _, row, _ in rows})
    return (
        f"    {engine:<18} N={ns}  timeouts={timeouts}  rounds={'/'.join(rounds)}  "
        f"version={', '.join(versions)}  hosts={', '.join(hosts)}"
    )


def _merge_dataset(args, row_kind, sweep, engines):
    """(problems, merged lines or None) for one dataset, reporting as it goes."""
    grid = GRID[(row_kind, sweep)]
    by_engine, ignored = _collect(args.run_dirs, sweep, engines, grid)
    print(f"{row_kind}/{sweep}:")
    if not by_engine:
        if args.allow_partial:
            print("  note: no rows")
            return [], None
        return ["no rows"], None
    for engine, rows in by_engine.items():
        print(_summary(engine, rows))
    if ignored:
        print(
            f"  ignored, outside the {row_kind}/{sweep} grid {grid}: {', '.join(ignored)}"
        )
    problems, notes = check_dataset(
        sweep, grid, by_engine, allow_partial=args.allow_partial
    )
    for note in notes:
        print(f"  note: {note}")
    lines = [
        raw
        for e in engines
        for _, raw, _, _ in sorted(by_engine.get(e, []), key=lambda t: t[0])
    ]
    return problems, lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dirs", nargs="+", type=Path, help="campaign RUN_DIRs")
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

    all_problems, writes = [], {}
    for (row_kind, sweep), (name, engines) in DATASETS.items():
        problems, lines = _merge_dataset(args, row_kind, sweep, engines)
        all_problems += [f"{row_kind}/{sweep}: {p}" for p in problems]
        if lines is not None:
            writes[outdir / name] = lines
    if all_problems:
        print("refused:", *all_problems, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)
    print("  checks passed: one row per N, identical terms/gates/expectation, 1 thread")

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
