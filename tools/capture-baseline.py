#!/usr/bin/env python3
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

"""Capture golden baselines for bit-identity checks across a refactor.

For every fixture in ``tests/data/*.msgpack``, propagates through
:class:`~monoprop.MajoranaPropagator` (``Basis::Majorana``) at a handful of cutoffs and cutoff
types, and for a small set of hand-written qubit smoke problems through
:class:`~monoprop.PauliPropagator` (``Basis::Pauli``). For each run this dumps: the term count, the
full ``(monomial indices, coefficient)`` set (in the engine's own iteration order -- ordering is
itself a regression signal, see the module docstring on diffing below), and the expectation value.

Dual-basis note: the msgpack fixtures are all fermionic (Majorana). The public API only converts
Pauli -> Majorana (``PauliOperator.get_majorana_operator()``, the Jordan-Wigner image); there is no
Majorana -> Pauli operator converter to press a fixture into a qubit circuit, and hand-rolling that
transform for a test-only tool risks baking a silently-wrong transform into the "golden" baseline.
So ``Basis::Pauli`` coverage instead comes from ``_PAULI_SMOKE_CASES`` below: a few hardcoded native
``PauliOperator``/``Circuit`` problems, sized like the ones in ``tests/test_basis.py`` /
``tests/test_pauli.py``. Their values are not independently re-derived here -- as with the msgpack
fixtures' *non*-``_EXACT_FIXTURES``, they are simply frozen as-is for regression diffing.

Ordering and diffing: terms are dumped in the engine's own returned order, not sorted -- that order is
itself load-bearing (``SplitmixHash`` -> probe order -> MPI owner routing -> insertion order ->
floating-point accumulation order), so a silent reorder is exactly the kind of regression this tool
exists to catch, and ``just diff-baseline`` stays a byte-wise diff.

``--compare REF CAND [--tol]`` is the other mode, for a change that reorders terms *on purpose*: it
holds term membership and the term count exactly and compares coefficients and energies only to a
relative tolerance. The support-form row backend is the case it exists for -- it hashes rows
differently, so it accumulates in a different order (see ``monoprop_ROW_STORE`` and the plan's
Stage 6). Use it to check one backend against the other; do not use it to wave through a diff whose
ordering was supposed to be stable.

Usage (needs the `test` dependency group synced -- this reuses tests/cases.py, which imports
pytest-cases):
    uv sync --all-extras --group test
    uv run --no-sync python tools/capture-baseline.py --out .baseline-capture/<label>

Under MPI (one output tree per run; each rank writes its own files -- ``evolved_operator()`` is
rank-local, see ``tests/conftest.py``):
    mpiexec -n 2 uv run --no-sync python tools/capture-baseline.py --out .baseline-capture/<label>

Output layout under ``--out``:
    manifest-rank<r>.json  -- one entry per (case, basis, cutoff_type, cutoff) with its sha256
    cases/<case>-rank<r>.json -- the raw per-case dumps backing that manifest

See ``just diff-baseline`` (justfile) for the rebuild-and-compare instrument this feeds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monoprop import Circuit, ExpGate, MajoranaPropagator, PauliPropagator
from monoprop.pauli import PauliOperator

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - exercised in wheel-test environments
    MPI = None

if TYPE_CHECKING:
    from monoprop.monomial_propagator import MonomialPropagator

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "tests" / "data"

# Cutoffs to probe on every fixture, clipped to the fixture's valid range -- cheap and broad.
# Deliberately stops at 4: "support" grows combinatorially in num_modes (28 modes, support=6 ->
# ~13.7M surviving terms on S0_14e14o_majoranic_c6 -- seconds to propagate, but a full term-by-term
# JSON dump of that is exactly the kind of accidental multi-GB, multi-minute capture this tool must
# not become). Wider cutoffs belong to the dedicated benchmarks (notes/monomial-storage/bench/,
# benches/bench_models.py), not this smoke-level regression instrument.
_PROBE_CUTOFFS = (2, 4)

# Fixtures small enough that propagating with no truncation (cutoff = 2 * num_modes) is tractable,
# matching tests/test_circuit.py's FIXTURES -- also cross-checked here against the fixture's own
# `actual_energy` (an independently-computed exact value), not just captured for later diffing.
_EXACT_FIXTURES = frozenset(
    {"rx_rz_ry_rz_exact", "random_exact", "lih_fermionic_spin_exact"}
)


def _load_cases_module() -> Any:  # noqa: ANN401
    """Import tests/cases.py by path (tests/ is not a package; pytest uses --import-mode=importlib)."""
    spec = importlib.util.spec_from_file_location(
        "_baseline_tests_cases", REPO_ROOT / "tests" / "cases.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load a module spec for {REPO_ROOT / 'tests' / 'cases.py'}."
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _comm() -> Any:  # noqa: ANN401
    """COMM_WORLD if run under mpiexec with mpi4py available, else None (serial)."""
    return None if MPI is None else MPI.COMM_WORLD


def _cutoffs_for(num_modes: int, case_name: str) -> list[int]:
    cutoffs = {c for c in _PROBE_CUTOFFS if 1 <= c <= 2 * num_modes}
    if case_name in _EXACT_FIXTURES:
        cutoffs.add(2 * num_modes)
    return sorted(cutoffs)


def _term_key(term: Any) -> Any:  # noqa: ANN401
    """A JSON-safe key: index tuples dump as lists; a Pauli term dumps as ``[string, qubits]``."""
    if hasattr(term, "string"):  # monoprop.pauli.Pauli
        return [term.string, list(term.qubits)]
    return list(term)


def _term_dump(terms: dict[Any, complex]) -> list[list[Any]]:
    """Engine-native order preserved -- see the module docstring's "Ordering and diffing"."""
    return [
        [_term_key(key), repr(coeff.real), repr(coeff.imag)]
        for key, coeff in terms.items()
    ]


def _record(
    *,
    case: str,
    basis: str,
    cutoff_type: str,
    cutoff: int,
    propagator: MonomialPropagator,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dump one propagator run: term count, the ordered term set, and the expectation value."""
    evolved = propagator.evolved_operator(
        atol=0.0
    )  # atol=0.0: keep every term, nothing dropped.
    energy = propagator.expectation_value()
    rec = {
        "case": case,
        "basis": basis,
        "cutoff_type": cutoff_type,
        "cutoff": cutoff,
        "n_terms": len(evolved.terms),
        "terms": _term_dump(evolved.terms),
        "energy": repr(energy),
    }
    if extra:
        rec.update(extra)
    return rec


def _capture_majorana_case(
    cases_mod: Any,  # noqa: ANN401
    path: Path,
    comm: Any,  # noqa: ANN401
) -> list[dict[str, Any]]:
    """Capture every (cutoff_type, cutoff) combo for one msgpack fixture."""
    problem = cases_mod.load_problem(path)
    circuit = problem.monomial_circuit.to_circuit()
    case = path.stem
    cutoffs = _cutoffs_for(problem.n_modes, case)

    records = []
    for cutoff_type in ("length", "support"):
        for cutoff in cutoffs:
            propagator = MajoranaPropagator.from_circuit(
                circuit,
                problem.operator,
                cutoff=cutoff,
                cutoff_type=cutoff_type,
                comm=comm,
            )
            extra = None
            if case in _EXACT_FIXTURES and cutoff == 2 * problem.n_modes:
                extra = {"exact_energy": repr(problem.exact_expval)}
            records.append(
                _record(
                    case=case,
                    basis="majorana",
                    cutoff_type=cutoff_type,
                    cutoff=cutoff,
                    propagator=propagator,
                    extra=extra,
                )
            )
    return records


def _pauli_smoke_cases() -> list[tuple[str, Any, Any, int]]:
    """A few hand-written native-Pauli problems -- see the module docstring's dual-basis note."""
    cases = []

    observable_1q = PauliOperator({"Z": 1.0}, num_qubits=1)
    circuit_1q = Circuit(
        gates=(ExpGate(PauliOperator({"Y": 1.0}, num_qubits=1), index=0),),
        initial_state=(),
        system_size=1,
        parameters=(0.3,),
    )
    cases.append(("pauli_smoke_1q_y_rotation", observable_1q, circuit_1q, 1))

    observable_2q = PauliOperator({"ZZ": 1.0}, num_qubits=2)
    circuit_2q = Circuit(
        gates=(ExpGate(PauliOperator({"XY": 1.0}, num_qubits=2), index=0),),
        initial_state=(0,),
        system_size=2,
        parameters=(0.7,),
    )
    cases.append(("pauli_smoke_2q_xy_rotation", observable_2q, circuit_2q, 2))

    return cases


def _capture_pauli_smoke(comm: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Capture every probe cutoff for each of ``_pauli_smoke_cases``."""
    records = []
    for case, observable, circuit, num_qubits in _pauli_smoke_cases():
        for cutoff in sorted(
            {c for c in _PROBE_CUTOFFS if 1 <= c <= num_qubits} | {num_qubits}
        ):
            propagator = PauliPropagator.from_circuit(
                circuit, observable, cutoff=cutoff, comm=comm
            )
            records.append(
                _record(
                    case=case,
                    basis="pauli",
                    cutoff_type="support",
                    cutoff=cutoff,
                    propagator=propagator,
                )
            )
    return records


def _sha256_of(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _manifest_entry(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": rec["case"],
        "basis": rec["basis"],
        "cutoff_type": rec["cutoff_type"],
        "cutoff": rec["cutoff"],
        "n_terms": rec["n_terms"],
        "sha256": _sha256_of([rec]),
    }


def _hashable(key: Any) -> Any:  # noqa: ANN401
    """A term key as a nested tuple, so it can go in a dict. Pauli keys nest one level."""
    if isinstance(key, list):
        return tuple(_hashable(k) for k in key)
    return key


def _load_records(tree: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Every record in a capture tree, keyed by (case, basis, cutoff_type, cutoff, rank)."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted((tree / "cases").glob("*-rank*.json")):
        rank = int(path.stem.rsplit("rank", 1)[1])
        for rec in json.loads(path.read_text()):
            out[
                (rec["case"], rec["basis"], rec["cutoff_type"], rec["cutoff"], rank)
            ] = rec
    return out


def _close(a: float, b: float, tol: float) -> bool:
    """Relative comparison, floored at 1 so near-zero coefficients are held to an absolute tol."""
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _compare_terms(
    ref: list[list[Any]], cand: list[list[Any]], tol: float
) -> list[str]:
    """The term sets of one record: membership exactly, coefficients only to ``tol``."""
    problems = []
    ref_terms = {_hashable(k): (float(re_), float(im)) for k, re_, im in ref}
    cand_terms = {_hashable(k): (float(re_), float(im)) for k, re_, im in cand}
    missing = sorted(map(repr, ref_terms.keys() - cand_terms.keys()))
    extra = sorted(map(repr, cand_terms.keys() - ref_terms.keys()))
    if missing:
        problems.append(f"{len(missing)} term(s) absent, e.g. {missing[0]}")
    if extra:
        problems.append(f"{len(extra)} unexpected term(s), e.g. {extra[0]}")
    worst = 0.0
    for key in ref_terms.keys() & cand_terms.keys():
        for a, b in zip(ref_terms[key], cand_terms[key], strict=True):
            if not _close(a, b, tol):
                worst = max(worst, abs(a - b))
    if worst > 0.0:
        problems.append(f"coefficient(s) differ by up to {worst:.3e} (tol {tol:.1e})")
    return problems


def _compare_record(ref: dict[str, Any], cand: dict[str, Any], tol: float) -> list[str]:
    """Term sets and energies of one record. Terms compare as sets: order is not preserved here."""
    problems = []
    if ref["n_terms"] != cand["n_terms"]:
        problems.append(f"n_terms {ref['n_terms']} != {cand['n_terms']}")
    problems.extend(_compare_terms(ref["terms"], cand["terms"], tol))
    for field in ("energy", "exact_energy"):
        if field in ref or field in cand:
            a, b = float(ref.get(field, "nan")), float(cand.get(field, "nan"))
            if not _close(a, b, tol):
                problems.append(f"{field} {a!r} != {b!r} (tol {tol:.1e})")
    return problems


def _compare_trees(ref_dir: Path, cand_dir: Path, tol: float) -> int:
    """Compare two capture trees as term *sets* to a relative tolerance. Returns an exit code."""
    ref, cand = _load_records(ref_dir), _load_records(cand_dir)
    if ref.keys() != cand.keys():
        only_ref = sorted(map(repr, ref.keys() - cand.keys()))
        only_cand = sorted(map(repr, cand.keys() - ref.keys()))
        sys.stdout.write(
            f"record sets differ: {len(only_ref)} only in {ref_dir}, "
            f"{len(only_cand)} only in {cand_dir}\n"
        )
        return 1
    failures = 0
    for key in sorted(ref.keys(), key=repr):
        problems = _compare_record(ref[key], cand[key], tol)
        if problems:
            failures += 1
            sys.stdout.write(f"{key}: {'; '.join(problems)}\n")
    verdict = "differ" if failures else "agree"
    sys.stdout.write(
        f"{len(ref)} records {verdict} between {ref_dir} and {cand_dir} "
        f"(term sets exact, values to rtol {tol:.1e}); {failures} failing\n"
    )
    return 1 if failures else 0


def main() -> int:
    """Capture every fixture/smoke case into ``--out``, or compare two capture trees."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, help="Output directory for a capture (created if absent)."
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("REF", "CAND"),
        help=(
            "Compare two capture trees instead of capturing: term sets must match exactly, "
            "values only to --tol. Use when a change reorders terms on purpose."
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Relative tolerance for --compare (default 1e-10).",
    )
    args = parser.parse_args()

    if args.compare:
        return _compare_trees(args.compare[0], args.compare[1], args.tol)
    if args.out is None:
        parser.error("one of --out or --compare is required")

    comm = _comm()
    rank = 0 if comm is None else comm.Get_rank()

    cases_mod = _load_cases_module()
    all_records: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.msgpack")):
        all_records.extend(_capture_majorana_case(cases_mod, path, comm))
    all_records.extend(_capture_pauli_smoke(comm))

    out_dir = args.out
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    by_case: dict[str, list[dict[str, Any]]] = {}
    for rec in all_records:
        by_case.setdefault(rec["case"], []).append(rec)

    manifest = []
    for case, recs in by_case.items():
        case_path = cases_dir / f"{case}-rank{rank}.json"
        case_path.write_text(json.dumps(recs, sort_keys=True, indent=2) + "\n")
        manifest.extend(_manifest_entry(rec) for rec in recs)
    manifest.sort(key=lambda m: (m["case"], m["basis"], m["cutoff_type"], m["cutoff"]))
    (out_dir / f"manifest-rank{rank}.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )

    sys.stdout.write(
        f"[rank {rank}] captured {len(all_records)} records across {len(by_case)} cases -> {out_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
