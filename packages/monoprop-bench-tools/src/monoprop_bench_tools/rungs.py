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

"""The rung ladder: monoprop's benchmark set as a table, and the runner that executes it.

A *rung* is one cell of the benchmark set -- a picture, a model, a set of operations, a
geometry and a problem size, with the term count that size is known to produce. The rungs
live in ``benches/rungs.toml`` rather than in a loop, so that a campaign cannot quietly run
a different grid and two campaigns' numbers are comparable row for row.

Every rung declares ``expect_terms``, and the runner refuses a result whose term count
misses it: a mistyped tolerance then fails the cell instead of silently measuring a
different problem. A row that has never been calibrated declares ``expect_terms = 0`` and
refuses to run at all.

Nothing here runs automatically. These are the fixed benchmarks; running the ones a change
could plausibly move, and putting the numbers in the pull request, is the author's job.

Two entry points::

    monoprop-bench-rung <rungs.toml> <rung-id> [--rep N] [--results DIR] [--dry-run]
    monoprop-bench-ladder <rungs.toml> <results-dir> [--family strong]

The first runs one rung once; repetition comes from repeated invocations, because
``pytest-benchmark``'s ``pedantic`` builds round k+1's arguments before releasing round k's,
so more than one round holds two propagators and doubles peak memory. The second collates a
directory of rung artifacts into a markdown block -- the timings, the peak memory, and the
resolved parameters of every problem measured -- to paste into the pull request.

Each row also carries what it last cost, in wall seconds and GiB per node, so you can see
what a rung will take before you spend it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

BYTES_PER_GIB = 1 << 30

# A term count is reproducible to the digit at a fixed seed and tolerance, so this is a
# guard against a mistyped knob, not a statistical tolerance.
TERM_TOLERANCE = 0.001

# A size knob nobody has measured yet. It is not spelled 0 because 0 is not neutral --
# `lower_atol = 0` prunes nothing, so a placeholder 0 is the largest problem the model can
# pose and would burn the allocation the gate exists to protect.
UNSET = "TBD"

FAMILIES = frozenset({"size", "strong", "weak"})
PICTURES = frozenset({"heisenberg", "schrodinger"})
MODELS = frozenset({"random", "hubbard", "pauli"})
OPS = ("build_graph", "propagate", "energy", "gradient")
REQUIRED = frozenset(
    {
        "id",
        "family",
        "picture",
        "model",
        "ops",
        "nodes",
        "ranks_per_node",
        "partitions",
        "expect_terms",
        "reps",
    }
)

# The random problem is driven from the CLI options in benches/conftest.py and carries the
# picture axis; the fixed models are Heisenberg-only and carry their own config overrides.
_BENCH_FILE = {"random": "benches/bench_random.py"}
_TEST_PREFIX = {"random": "test_random_"}
_DEFAULT_FILE = "benches/bench_models.py"
_DEFAULT_PREFIX = "test_model_"


@dataclass(frozen=True, slots=True)
class Rung:
    """One cell of the benchmark set."""

    id: str
    family: str
    picture: str
    model: str
    ops: tuple[str, ...]
    nodes: int
    ranks_per_node: int
    partitions: int
    expect_terms: int
    reps: int
    args: tuple[str, ...] = ()
    cost_seconds: float = 0.0
    cost_gib_per_node: float = 0.0
    walltime: str = ""
    note: str = ""

    @property
    def ranks(self) -> int:
        """Return the MPI rank count this rung is launched with."""
        return self.nodes * self.ranks_per_node

    @property
    def world(self) -> int:
        """Return the flat world size, ranks x partitions -- the engine's own ``P``."""
        return self.ranks * self.partitions

    @property
    def bench_file(self) -> str:
        """Return the bench module holding this rung's operations."""
        return _BENCH_FILE.get(self.model, _DEFAULT_FILE)

    @property
    def cost(self) -> str:
        """Return what one rep of this rung last cost, or ``?`` if nobody has measured it."""
        if not self.cost_seconds:
            return "?"
        return f"{self.cost_seconds:.3g}s {self.cost_gib_per_node:.3g}GiB/node"

    @property
    def calibrated(self) -> bool:
        """Return whether this rung's size knobs and term count have both been measured."""
        return self.expect_terms > 0 and not any(
            a.endswith(f"={UNSET}") for a in self.args
        )

    def selector(self) -> str:
        """Return the pytest ``-k`` expression selecting this rung's cells."""
        prefix = _TEST_PREFIX.get(self.model, _DEFAULT_PREFIX)
        ops = " or ".join(f"{prefix}{op}" for op in self.ops)
        # The random tests carry the picture as a parameter id; the fixed models carry
        # the model name instead and have no picture axis.
        axis = self.picture if self.model == "random" else self.model
        return f"({ops}) and {axis}"


def _require(cond: bool, msg: str) -> None:  # noqa: FBT001
    if not cond:
        raise SystemExit(f"rungs: {msg}")


def _one(raw: dict[str, Any], index: int) -> Rung:
    """Return one validated rung from its table entry."""
    where = raw.get("id", f"rung #{index}")
    missing = REQUIRED - set(raw)
    _require(not missing, f"{where}: missing {sorted(missing)}")

    # Every enum is checked against an exhaustive set: a value nobody validated reaches the
    # launcher as a selector that matches nothing, and an empty run reads as a clean one.
    _require(
        raw["family"] in FAMILIES,
        f"{where}: family {raw['family']!r} not in {sorted(FAMILIES)}",
    )
    _require(
        raw["picture"] in PICTURES,
        f"{where}: picture {raw['picture']!r} not in {sorted(PICTURES)}",
    )
    _require(
        raw["model"] in MODELS,
        f"{where}: model {raw['model']!r} not in {sorted(MODELS)}",
    )
    _require(bool(raw["ops"]), f"{where}: ops is empty")
    unknown = [op for op in raw["ops"] if op not in OPS]
    _require(
        not unknown, f"{where}: unknown ops {unknown}, expected a subset of {list(OPS)}"
    )
    _require(
        raw["picture"] == "heisenberg" or raw["model"] == "random",
        f"{where}: only the random model has a picture axis",
    )
    for field in ("nodes", "ranks_per_node", "partitions", "reps"):
        _require(raw[field] >= 1, f"{where}: {field} must be >= 1, got {raw[field]}")
    _require(raw["expect_terms"] >= 0, f"{where}: expect_terms must be >= 0")
    unset = [a for a in raw.get("args", ()) if a.endswith(f"={UNSET}")]
    _require(
        not (unset and raw["expect_terms"]),
        f"{where}: carries {unset} yet declares a term count; a rung whose size knob is "
        f"unmeasured cannot have a measured size",
    )

    return Rung(
        id=raw["id"],
        family=raw["family"],
        picture=raw["picture"],
        model=raw["model"],
        ops=tuple(raw["ops"]),
        nodes=raw["nodes"],
        ranks_per_node=raw["ranks_per_node"],
        partitions=raw["partitions"],
        expect_terms=raw["expect_terms"],
        reps=raw["reps"],
        args=tuple(raw.get("args", ())),
        cost_seconds=float(raw.get("cost_seconds", 0.0)),
        cost_gib_per_node=float(raw.get("cost_gib_per_node", 0.0)),
        walltime=raw.get("walltime", ""),
        note=raw.get("note", ""),
    )


def load_rungs(path: Path) -> dict[str, Rung]:
    """Return the rung table at ``path``, keyed by id."""
    _require(path.is_file(), f"missing rung table {path}")
    raw = tomllib.loads(path.read_text())
    rungs: dict[str, Rung] = {}
    for index, entry in enumerate(raw.get("rung", [])):
        rung = _one(entry, index)
        _require(rung.id not in rungs, f"duplicate rung id {rung.id!r}")
        rungs[rung.id] = rung
    _require(bool(rungs), f"{path} declares no rungs")
    return rungs


def pytest_argv(rung: Rung, results_dir: Path, label: str) -> list[str]:
    """Return the pytest command line measuring one rep of ``rung``."""
    return [
        sys.executable,
        "-m",
        "pytest",
        rung.bench_file,
        "-k",
        rung.selector(),
        # Not a default: a second round holds two propagators at once and doubles peak RSS.
        "--bench-rounds=1",
        *rung.args,
        f"--benchmark-json={results_dir / f'time-{label}.json'}",
        "-o",
        "filterwarnings=default",
        "-q",
        "-p",
        "no:cacheprovider",
    ]


def bench_env(rung: Rung, results_dir: Path, label: str) -> dict[str, str]:
    """Return the environment overrides one rep of ``rung`` runs under."""
    return {
        "monoprop_BENCH_LABEL": label,
        "monoprop_BENCH_RESULTS": str(results_dir),
        "monoprop_NUM_THREADS": str(rung.partitions),
        "monoprop_PARTITIONS": str(rung.partitions),
    }


def artifact_terms(rung: Rung, results: dict[str, Any]) -> int | None:
    """Return the term count ``results`` recorded for ``rung``, or ``None`` if it has none."""
    opsize = results.get("opsize", {})
    axis = rung.picture if rung.model == "random" else rung.model
    if axis in opsize:
        return int(opsize[axis]["terms"])
    counts = [int(v["terms"]) for k, v in opsize.items() if "::" in k]
    return max(counts) if counts else None


def gate(rung: Rung, results: dict[str, Any]) -> list[str]:
    """Return the reasons ``results`` must not be kept as a measurement of ``rung``.

    An empty list is the only clean outcome. Each check answers "did this process measure
    the problem the table names", never "did it look plausible".
    """
    reasons: list[str] = []
    meta = results.get("meta", {})

    for field, want in (("nodes", rung.nodes), ("ranks_per_node", rung.ranks_per_node)):
        got = meta.get(field)
        if got != want:
            reasons.append(f"{field}: ran {got}, table says {want}")

    partitions_env = meta.get("partitions_env")
    if partitions_env != str(rung.partitions):
        reasons.append(
            f"partitions: ran {partitions_env}, table says {rung.partitions}"
        )

    rounds = results.get("params", {}).get("bench_rounds")
    if rounds != 1:
        reasons.append(
            f"bench_rounds: ran {rounds}, must be 1 or peak memory is doubled"
        )

    terms = artifact_terms(rung, results)
    if terms is None:
        reasons.append("terms: the run recorded no operator size")
    elif abs(terms - rung.expect_terms) > TERM_TOLERANCE * rung.expect_terms:
        reasons.append(f"terms: measured {terms:,}, table says {rung.expect_terms:,}")

    return reasons


def _run_one(rung: Rung, results_dir: Path, rep: int, *, dry_run: bool) -> int:
    label = f"{rung.id}-r{rep}"
    argv = pytest_argv(rung, results_dir, label)
    env = bench_env(rung, results_dir, label)

    if dry_run:
        print(f"rung      {rung.id}  ({rung.family}, {rung.picture}, {rung.model})")
        print(
            f"geometry  {rung.nodes} nodes x {rung.ranks_per_node} ranks x {rung.partitions} partitions"
            f"  => R={rung.ranks} P={rung.world}"
        )
        print(
            f"expect    {rung.expect_terms:,} terms, {rung.reps} reps, walltime {rung.walltime or '-'}"
        )
        print(f"costs     {rung.cost} per rep, x{rung.reps} reps x {rung.nodes} nodes")
        print("env       " + " ".join(f"{k}={v}" for k, v in env.items()))
        print("run       " + " ".join(argv))
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(argv, env={**os.environ, **env}, check=False)  # noqa: S603
    if completed.returncode != 0:
        print(f"rung {rung.id}: pytest exited {completed.returncode}", file=sys.stderr)
        return completed.returncode

    artifact = results_dir / f"{label}.json"
    if not artifact.is_file():
        print(f"rung {rung.id}: no artifact at {artifact}", file=sys.stderr)
        return 1

    reasons = gate(rung, json.loads(artifact.read_text()))
    if reasons:
        artifact.rename(artifact.with_suffix(".refused.json"))
        print(f"## REFUSED {rung.id} rep {rep}", file=sys.stderr)
        for reason in reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 1

    print(f"rung {rung.id} rep {rep}: kept {artifact}")
    return 0


def run_main(argv: Sequence[str] | None = None) -> int:
    """Run one rep of one rung, keeping the result only if it passes the gate."""
    parser = argparse.ArgumentParser(prog="monoprop-bench-rung", description=__doc__)
    parser.add_argument("table", type=Path, help="path to rungs.toml")
    parser.add_argument("rung_id", help="rung to run, or 'list' to print the table")
    parser.add_argument(
        "--rep", type=int, default=1, help="rep index, used to name the artifact"
    )
    parser.add_argument("--results", type=Path, default=Path("benches/results"))
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan without running it"
    )
    args = parser.parse_args(argv)

    rungs = load_rungs(args.table)
    if args.rung_id == "list":
        for rung in rungs.values():
            mark = " " if rung.calibrated else "*"
            print(
                f"{mark} {rung.id:<44} {rung.family:<6} N={rung.nodes:<3} P={rung.world:<5} "
                f"{rung.expect_terms:>14,}  {rung.cost}"
            )
        print(
            "\n* = uncalibrated (no term count, or a TBD size knob); the rung refuses to run."
            "\nThe last column is what one rep last cost; ? means nobody has timed it."
        )
        return 0

    rung = rungs.get(args.rung_id)
    if rung is None:
        print(f"rungs: no rung {args.rung_id!r}; try 'list'", file=sys.stderr)
        return 2
    if not rung.calibrated:
        print(
            f"rungs: {rung.id} is uncalibrated. Measure its size knobs and term count and "
            f"put them in the table before spending an allocation on it.",
            file=sys.stderr,
        )
        return 2

    return _run_one(rung, args.results, args.rep, dry_run=args.dry_run)


@dataclass(frozen=True, slots=True)
class Cell:
    """One rung's collated measurement across its reps."""

    rung: Rung
    seconds: tuple[float, ...]
    gib_per_node: tuple[float, ...]
    problem: dict[str, Any]

    @property
    def reps(self) -> int:
        """Return how many gate-clean reps were found."""
        return len(self.seconds)


def _artifacts(
    rung: Rung, results_dir: Path
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for artifact in sorted(results_dir.glob(f"{rung.id}-r*.json")):
        if artifact.name.endswith(".refused.json"):
            continue
        timing = results_dir / f"time-{artifact.stem}.json"
        if not timing.is_file():
            continue
        yield json.loads(artifact.read_text()), json.loads(timing.read_text())


def problem(rung: Rung, results: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved parameters of the problem a rung actually posed.

    Read back from the run rather than from the row's ``args``, because ``args`` names only
    the overrides: a reader given ``--hubbard-lower-atol=1.25e-05`` cannot tell what the
    other nine fields were, and a default that changes underneath the table would go unseen.
    """
    if rung.model == "random":
        return dict(results.get("params", {}))
    return dict(results.get("configs", {}).get(rung.model, {}))


def collate(rungs: dict[str, Rung], results_dir: Path) -> list[Cell]:
    """Return one :class:`Cell` per rung that has at least one gate-clean rep."""
    cells: list[Cell] = []
    for rung in rungs.values():
        seconds: list[float] = []
        gib: list[float] = []
        params: dict[str, Any] = {}
        for results, timing in _artifacts(rung, results_dir):
            if gate(rung, results):
                continue
            benches = timing.get("benchmarks", [])
            if not benches:
                continue
            # The reported cost of a rung is the sum of the operations it selected: one
            # rung is one workload, not a menu to be read a column at a time.
            seconds.append(sum(b["stats"]["median"] for b in benches))
            # Summed over ranks, so ranks peaking at different moments are added as though
            # they had peaked together -- an upper bound on the node, never an estimate.
            total = sum(results.get("memhwm", {}).values())
            gib.append(total / rung.nodes / BYTES_PER_GIB)
            params = params or problem(rung, results)
        if seconds:
            cells.append(Cell(rung, tuple(seconds), tuple(gib), params))
    return cells


_COLUMNS = (
    "rung",
    "nodes",
    "R",
    "P",
    "terms",
    "Mterms/node",
    "reps",
    "median s",
    "min s",
    "Mterms/s/node",
    "GiB/node",
    "declared s",
    "vs declared",
)


def _row(cell: Cell) -> list[str]:
    rung = cell.rung
    median = statistics.median(cell.seconds)
    per_node = rung.expect_terms / 1e6 / rung.nodes
    was = rung.cost_seconds
    return [
        rung.id,
        str(rung.nodes),
        str(rung.ranks),
        str(rung.world),
        f"{rung.expect_terms:,}",
        f"{per_node:.0f}",
        str(cell.reps),
        f"{median:.2f}",
        f"{min(cell.seconds):.2f}",
        f"{per_node / median:.2f}",
        f"{statistics.median(cell.gib_per_node):.1f}",
        f"{was:.2f}" if was else "—",
        f"{median / was:.3f}x" if was else "—",
    ]


def render(cells: Sequence[Cell]) -> str:
    """Return the measurement, as a block to paste into a pull request.

    Read on the median and the min. These distributions are skewed -- a straggling rank
    pulls a mean well off the body of the sample -- so a mean here tracks stragglers rather
    than the cost of the work. That is also why this does not reuse the mean that
    :mod:`monoprop_bench_tools.bmf` reports: Bencher's t-test consumes a mean and a spread,
    and the two statistics answer different questions.

    ``declared s`` is the cost the table carries for that rung, so the ratio says how this
    run compares to the last one recorded. It is context, not a verdict: it may have been
    taken on another machine, and nothing here gates on it.
    """
    lines: list[str] = []
    for family in sorted({c.rung.family for c in cells}):
        lines += [f"## {family}", ""]
        lines += [
            "| " + " | ".join(_COLUMNS) + " |",
            "| --- |" + " ---: |" * (len(_COLUMNS) - 1),
        ]
        for cell in sorted(
            (c for c in cells if c.rung.family == family),
            # The node count is the last field of the id, so a plain string sort would
            # print n1, n16, n2 -- a ladder read down the page out of order.
            key=lambda c: (c.rung.id.rsplit("-n", 1)[0], c.rung.nodes, c.rung.id),
        ):
            lines.append("| " + " | ".join(_row(cell)) + " |")
        lines.append("")

    # A number without its problem is not reproducible, and a reviewer should not have to
    # resolve the row's overrides against the model's defaults to learn what was measured.
    lines += ["## Problems measured", ""]
    for cell in sorted(cells, key=lambda c: c.rung.id):
        rung = cell.rung
        lines.append(
            f"- **{rung.id}** — {rung.model} / {rung.picture}, "
            f"{', '.join(rung.ops)}, {rung.nodes} x {rung.ranks_per_node} x "
            f"{rung.partitions} (nodes x ranks/node x partitions)"
        )
        if cell.problem:
            fields = ", ".join(f"{k}={v}" for k, v in sorted(cell.problem.items()))
            lines.append(f"  - {fields}")
    lines.append("")
    return "\n".join(lines)


def ladder_main(argv: Sequence[str] | None = None) -> int:
    """Collate a directory of rung artifacts into a block to paste into a pull request."""
    parser = argparse.ArgumentParser(
        prog="monoprop-bench-ladder", description=ladder_main.__doc__
    )
    parser.add_argument("table", type=Path, help="path to rungs.toml")
    parser.add_argument(
        "results", type=Path, help="directory holding the rung artifacts"
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(FAMILIES),
        help="restrict to one family",
    )
    args = parser.parse_args(argv)

    rungs = load_rungs(args.table)
    if args.family:
        rungs = {k: v for k, v in rungs.items() if v.family in set(args.family)}
    cells = collate(rungs, args.results)
    if not cells:
        print(
            f"ladder: no gate-clean rung artifacts in {args.results}", file=sys.stderr
        )
        return 1
    sys.stdout.write(render(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main())
