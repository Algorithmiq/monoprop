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

"""Check a Fig. 6 octave sweep's per-job JSONL files and merge them into data/.

Each slot's rows are written, as recorded and ordered by N, to the file the figure reads:

    slot           engine field   data/ file
    monoprop       monoprop       monoprop_pauli_octave_l7_deucalion.jsonl
    monoprop-main  monoprop       monoprop_main_pauli_octave_l7_deucalion.jsonl
    ppvm           ppvm           ppvm_pauli_octave_l7_deucalion.jsonl
    julia          julia_pauli    julia_pauli_octave_v082_l7_deucalion.jsonl

The two monoprop builds share an engine field, so the slot, not the row, tells them apart.

Refused on failure: more than one row per N per slot, an engine field not matching its
slot, any row not at 7 layers, cutoff 6 and one thread, mismatched lower_atol or
observable, term or gate counts differing between slots, expectations differing by more
than 1e-9, both monoprop slots at the same version, and PauliPropagation.jl not at v0.8.2.
Every slot must reach every N <= FIT_NMAX. A slot missing points above that is reported,
not refused. Hostnames are reported only; FLEETS in the figure script judges them.

    python merge_octave_runs.py \\
        --monoprop      ../../runs/fig6/octave-mono-ref-664f84c0 \\
        --monoprop-main ../../runs/fig6/octave-mono-ref-508b536e-r2 \\
        --ppvm          ../../runs/fig6/octave-deucalion \\
        --julia         ../../runs/fig6/octave-deucalion

A slot takes run directories or JSONL files. ``--dry-run`` writes nothing.
``--allow-partial`` turns missing points and slots into notes, for a sweep still in
flight, and must write outside data/ (``--outdir``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# The slot -> (engine field the rows must carry, run-dir glob, data/ file name). The figure
# imports DEUCALION_FILES, so the merge and the default figure build cannot disagree on names.
SLOTS = {
    "monoprop": ("monoprop", "monoprop_N*.jsonl"),
    "monoprop-main": ("monoprop", "monoprop_N*.jsonl"),
    "ppvm": ("ppvm", "ppvm_N*.jsonl"),
    "julia": ("julia_pauli", "julia_N*.jsonl"),
}
DEUCALION_FILES = {
    "monoprop": "monoprop_pauli_octave_l7_deucalion.jsonl",
    "monoprop-main": "monoprop_main_pauli_octave_l7_deucalion.jsonl",
    "ppvm": "ppvm_pauli_octave_l7_deucalion.jsonl",
    "julia": "julia_pauli_octave_v082_l7_deucalion.jsonl",
}
# Slots the figure cannot be built without. monoprop-main only adds the "before" reference.
REQUIRED = ("monoprop", "ppvm", "julia")

GRID = (32, 64, 128, 256, 512, 1024)
# Must equal FIT_NMAX in make_inverse_scaling_figure.py (not imported: that module pulls in
# matplotlib, and this one runs anywhere). Inside it a missing point would move an exponent.
FIT_NMAX = 512
JULIA_VERSION = "0.8.2"
# The file names say l7, and the figure draws one cutoff.
LAYERS, CUTOFF = 7, 6
EXPECTATION_TOL = 1e-9
# Must be the same on every row of every slot.
WORKLOAD_KEYS = (
    "cutoff",
    "layers",
    "lower_atol",
    "observable",
    "basis",
    "active_window",
)


def _gather(slot, sources, *, allow_partial):
    """[(N, raw line, row, source)] for one slot, from run directories and/or files.

    A run directory holding none of the slot's files yet (a job still pending) is an error,
    or under --allow-partial simply contributes nothing.
    """
    _, pattern = SLOTS[slot]
    files = []
    for src in sources:
        if src.is_dir():
            found = sorted(src.glob(pattern))
            if not found and not allow_partial:
                raise SystemExit(f"--{slot}: no {pattern} in {src}")
            files += found
        elif src.is_file():
            files.append(src)
        else:
            raise SystemExit(f"--{slot}: {src} does not exist")
    out = []
    for f in files:
        for raw in f.read_text().splitlines():
            if raw.strip():
                row = json.loads(raw)
                out.append((row["num_qubits"], raw.strip(), row, f))
    return out


def _row_problems(slot, n, row, src):
    """What is wrong with one row on its own."""
    engine, _ = SLOTS[slot]
    out = []
    if row["engine"] != engine:
        out.append(f"{slot}: N={n} in {src} has engine {row['engine']!r}")
    if (row["layers"], row["cutoff"]) != (LAYERS, CUTOFF):
        out.append(
            f"{slot}: N={n} has layers={row['layers']} cutoff={row['cutoff']}, "
            f"expected {LAYERS} and {CUTOFF}"
        )
    if str(row["num_threads"]) != "1":
        out.append(f"{slot}: N={n} num_threads={row['num_threads']!r}")
    if slot == "julia" and row.get("library_version") != JULIA_VERSION:
        out.append(
            f"julia: N={n} library_version {row.get('library_version')!r}, the file name "
            f"says v{JULIA_VERSION}"
        )
    return out


def _check_slot(slot, rows):
    """(problems, gaps inside the fit window, gaps above it) for one slot."""
    problems, by_n = [], {}
    for n, _, row, src in rows:
        by_n.setdefault(n, []).append(src)
        problems += _row_problems(slot, n, row, src)
    problems += [
        f"{slot}: {len(srcs)} rows at N={n} ({', '.join(map(str, srcs))})"
        for n, srcs in sorted(by_n.items())
        if len(srcs) > 1
    ]
    stray = sorted(set(by_n) - set(GRID))
    if stray:
        problems.append(f"{slot}: N outside the octave grid: {stray}")
    missing = [n for n in GRID if n not in by_n]
    return (
        problems,
        [n for n in missing if n <= FIT_NMAX],
        [n for n in missing if n > FIT_NMAX],
    )


def _check_across(slots):
    """Problems between slots: one workload, the same operator at every N, two builds."""
    problems = []
    every = [row for rows in slots.values() for _, _, row, _ in rows]
    for key in WORKLOAD_KEYS:
        vals = sorted({json.dumps(r.get(key)) for r in every})
        if len(vals) > 1:
            problems.append(f"{key} differs across rows: {', '.join(vals)}")
    for n in GRID:
        at = {
            slot: row for slot, rows in slots.items() for m, _, row, _ in rows if m == n
        }
        if len(at) < 2:
            continue
        for key in ("num_terms", "gates"):
            vals = {slot: row[key] for slot, row in at.items()}
            if len(set(vals.values())) != 1:
                detail = ", ".join(f"{s}={v}" for s, v in vals.items())
                problems.append(f"N={n}: {key} differs -- {detail}")
        exps = [row["expectation"] for row in at.values()]
        if max(exps) - min(exps) > EXPECTATION_TOL:
            problems.append(f"N={n}: expectation spread {max(exps) - min(exps):.3e}")
    if "monoprop" in slots and "monoprop-main" in slots:
        new, old = (
            {row.get("library_version") for _, _, row, _ in slots[s]}
            for s in ("monoprop", "monoprop-main")
        )
        if new & old:
            problems.append(
                f"monoprop and monoprop-main share library_version {sorted(new & old)}: "
                "the same build (or run directory) passed twice?"
            )
    return problems


def check(slots, *, allow_partial):
    """(problems, notes) for the gathered slots. Problems refuse the merge.

    Under ``allow_partial`` a missing slot or a gap inside the fit window is a note instead.
    """
    problems, notes = [], []
    partial = notes if allow_partial else problems
    partial += [f"{slot}: no rows given" for slot in REQUIRED if slot not in slots]
    for slot, rows in slots.items():
        slot_problems, inside, above = _check_slot(slot, rows)
        problems += slot_problems
        if inside:
            partial.append(f"{slot}: missing N={inside} (inside the fit window)")
        if above:
            notes.append(f"{slot}: missing N={above} (above N={FIT_NMAX}: drawn-only)")
    return problems + _check_across(slots), notes


def _summary(slot, rows):
    ns = sorted(n for n, *_ in rows)
    hosts = sorted({row["host"] for _, _, row, _ in rows})
    versions = sorted({str(row.get("library_version")) for _, _, row, _ in rows})
    rounds = sorted({str(row.get("rounds")) for _, _, row, _ in rows})
    return (
        f"  {slot:<14} N={ns}  rounds={'/'.join(rounds)}  version={', '.join(versions)}\n"
        f"  {'':<14} hosts: {', '.join(hosts)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for slot in SLOTS:
        ap.add_argument(
            f"--{slot}",
            nargs="+",
            type=Path,
            default=None,
            help=f"run directories (globbed for {SLOTS[slot][1]}) or JSONL files",
        )
    ap.add_argument("--outdir", type=Path, default=DATA)
    ap.add_argument("--dry-run", action="store_true", help="check and report only")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="missing points/slots become notes; for previews, never into data/",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    if args.allow_partial and not args.dry_run and outdir == DATA:
        raise SystemExit(
            "--allow-partial output is a preview: pass --outdir outside data/"
        )

    gathered = {
        slot: _gather(slot, srcs, allow_partial=args.allow_partial)
        for slot in SLOTS
        if (srcs := getattr(args, slot.replace("-", "_"))) is not None
    }
    slots = {slot: rows for slot, rows in gathered.items() if rows}
    problems, notes = check(slots, allow_partial=args.allow_partial)

    print("slots:")
    for slot, rows in slots.items():
        print(_summary(slot, rows))
    for note in notes:
        print(f"  note: {note}")
    if problems:
        print("refused:", *problems, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)
    print(
        "  checks passed: one row per N, identical num_terms/gates/expectation, 1 thread"
    )

    targets = {slot: outdir / DEUCALION_FILES[slot] for slot in slots}
    clobber = [p for p in targets.values() if p.exists()]
    if clobber and not args.force and not args.dry_run:
        raise SystemExit(
            f"exists (pass --force to overwrite): {', '.join(map(str, clobber))}"
        )
    print("\nwould write:" if args.dry_run else "\nwrote:")
    for slot, path in targets.items():
        lines = [raw for _, raw, _, _ in sorted(slots[slot], key=lambda t: t[0])]
        if not args.dry_run:
            outdir.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n")
        print(f"  {path}  ({len(lines)} rows)")


if __name__ == "__main__":
    main()
