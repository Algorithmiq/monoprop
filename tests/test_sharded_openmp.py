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

"""The sharded OpenMP runtime, through the public API, in fresh fixed-T processes.

Every case launches ``tests/sharded_openmp_probe.py`` in its own interpreter with
``monoprop_NUM_THREADS=T OMP_NUM_THREADS=T OMP_DYNAMIC=FALSE`` set before import, because T is captured when a
propagator is constructed and is never changed inside a process.

Comparisons:

- Across T = 1, 2, 4: identical global retained keys, coefficients within ``1e-9 + 1e-7 * max(|a|, |b|)``, and the
  same tolerance for unrounded energies and every gradient component.
- Against the removed partition runtime at the same (1, T) geometry, when ``monoprop_TEST_LEGACY_PYTHON`` names the
  interpreter of a separately built legacy environment (a pre-removal checkout): bitwise equality of energies,
  gradients, decoded maps, contraction blocks (concatenated in shard / partition order) and aggregate counts. The
  legacy process is a different build; the two class definitions never share a process.
- Over P MPI processes (an MPI build and a launcher on ``PATH``), each rank writing its own document: at (P, T) against
  the legacy runtime at P ranks x ``monoprop_PARTITIONS=T`` rank by rank and bit for bit (when the legacy interpreter
  is an MPI build); across the geometries (1, 4), (2, 2) and (4, 1), and three processes under splitmix against
  (1, 6), with the gathered global maps, energies and gradients within the tolerance above. Linear routing over three
  processes is rejected on every rank.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from monoprop import _core

_PROBE = Path(__file__).resolve().parent / "sharded_openmp_probe.py"
# The runtime identity this build's extension embeds, and the identities of the two pre-removal runtimes, which only a
# separately built external interpreter can carry. The embedded literals are NUL-terminated, so the final marker is
# matched exactly and never as a prefix of the prototype's.
_RUNTIME = "monoprop-runtime=sharded-openmp"
_PROTOTYPE = "monoprop-runtime=sharded-openmp-prototype"
_LEGACY = "monoprop-runtime=legacy-partitions"
# Environment a launcher gives its ranks: a child that inherits it would try to join the parent's job.
_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PRTE_", "PMI_", "HYDRA_", "MPIR_")
_TEAMS = (1, 2, 4)
_FIXTURES = [
    ("random_exact", "heisenberg"),
    ("random_exact", "schrodinger"),
    ("lih", "heisenberg"),
    ("lih", "schrodinger"),
    ("s0_8e8o", "heisenberg"),
    ("pauli_deep", "heisenberg"),
    ("pauli_deep", "schrodinger"),
    ("pauli_deep_vanishing", "heisenberg"),
]


def _inside_multirank_job() -> bool:
    size = os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE") or "1"
    return int(size) > 1


pytestmark = pytest.mark.skipif(
    _inside_multirank_job(),
    reason="launches fresh interpreters, which cannot run inside a multi-rank job",
)


def _child_env(threads: int, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_PREFIXES)}
    env.pop("monoprop_PARTITIONS", None)
    env.update(
        {
            "monoprop_NUM_THREADS": str(threads),
            "OMP_NUM_THREADS": str(threads),
            "OMP_DYNAMIC": "FALSE",
        }
    )
    env.update(extra or {})
    return env


def _run(
    spec: dict[str, Any],
    threads: int,
    *,
    python: str = sys.executable,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result = subprocess.run(  # noqa: S603 - trusted: a known interpreter and a fixed probe
        [python, str(_PROBE), json.dumps(spec)],
        env=_child_env(threads, extra_env),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 + 1e-7 * max(abs(a), abs(b))


def _f(text: str) -> float:
    return float.fromhex(text)


def _assert_map_close(
    left: dict[str, list[str]], right: dict[str, list[str]], what: str
) -> None:
    assert left.keys() == right.keys(), (
        f"{what}: retained keys differ "
        f"({len(left.keys() - right.keys())} only left, {len(right.keys() - left.keys())} only right)"
    )
    for key, (re_a, im_a) in left.items():
        re_b, im_b = right[key]
        assert _close(_f(re_a), _f(re_b)), (what, key)
        assert _close(_f(im_a), _f(im_b)), (what, key)


def _assert_values_close(left: Any, right: Any, what: str) -> None:
    if isinstance(left, dict):
        for key in left:
            if key.startswith(
                ("map", "contract", "aggregates", "python_terms", "parameter_mapping")
            ):
                continue
            _assert_values_close(left[key], right[key], f"{what}.{key}")
    elif isinstance(left, list):
        assert len(left) == len(right), what
        for a, b in zip(left, right, strict=True):
            _assert_values_close(a, b, what)
    elif isinstance(left, str) and left.startswith(("0x", "-0x")):
        assert _close(_f(left), _f(right)), (what, left, right)
    else:
        assert left == right, (what, left, right)


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Paths at which two probe documents differ; compact, so a failure never renders the whole document."""
    if isinstance(left, dict) and isinstance(right, dict):
        out = [
            f"{path}.{key}: missing on one side" for key in left.keys() ^ right.keys()
        ]
        for key in left.keys() & right.keys():
            out += _differences(left[key], right[key], f"{path}.{key}")
        return out
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [f"{path}: lengths {len(left)} != {len(right)}"]
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out += _differences(a, b, f"{path}[{index}]")
        return out
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def test_build_identity_matches_the_extension() -> None:
    # Provenance, not a selector: the attribute and the NUL-terminated literal in the extension file agree, and neither
    # pre-removal marker is present. A mismatch fails here instead of letting other cases skip or run the wrong code.
    assert getattr(_core, "__runtime_identity__", None) == _RUNTIME
    data = Path(_core.__file__).read_bytes()
    assert _RUNTIME.encode() + b"\0" in data
    assert _PROTOTYPE.encode() not in data
    assert _LEGACY.encode() not in data


_ARGS: dict[str, Any] = {
    "initial_operator": {(0, 1): 1j, (2, 3): 0.5j, (1, 2): 0.25j},
    "cutoff": 4,
    "initial_state": [0, 1],
}
# The low-level constructor's whole positional signature, up to and including ``basis``.
_POSITIONAL = [
    _ARGS["initial_operator"],
    4,
    [0, 1],
    None,
    None,
    None,
    None,
    "length",
    None,
    32,
    "majorana",
]
# Removed (partitions, child_factory) and never-added (thread or shard counts) constructor arguments.
_UNSUPPORTED_KEYWORDS = (
    "partitions",
    "child_factory",
    "num_threads",
    "threads",
    "shards",
    "num_shards",
)


def _high_level_constructors() -> dict[str, Any]:
    from monoprop import (  # noqa: PLC0415
        MajoranaOperator,
        MajoranaPropagator,
        Pauli,
        PauliOperator,
        PauliPropagator,
    )
    from monoprop._dispatch import dispatch  # noqa: PLC0415

    majorana = MajoranaOperator({(0, 1): 1j, (2, 3): 0.5j}, 4)
    pauli = PauliOperator({Pauli("ZZ", [0, 1]): 0.5}, num_qubits=3)
    return {
        "MajoranaPropagator": lambda **kw: MajoranaPropagator(
            majorana, [0, 1], cutoff=4, **kw
        ),
        "PauliPropagator": lambda **kw: PauliPropagator(pauli, [0], cutoff=3, **kw),
        "dispatch adapter": lambda **kw: dispatch(32)(**_ARGS, **kw),
    }


@pytest.mark.parametrize("value", [0, 1, 2])
def test_low_level_constructor_has_no_partitions(value: int) -> None:
    # Even the old default is not accepted: the parameter is gone, not merely restricted.
    with pytest.raises(TypeError):
        _core.MonomialPropagator032(**_ARGS, partitions=value)


def test_low_level_constructor_rejects_the_old_positional_tail() -> None:
    assert _core.MonomialPropagator032(*_POSITIONAL).size() >= 0
    for tail in ([0], [1], [0, None]):
        with pytest.raises(TypeError):
            _core.MonomialPropagator032(*_POSITIONAL, *tail)


def test_low_level_constructor_documents_no_partitions() -> None:
    doc = _core.MonomialPropagator032.__init__.__doc__ or ""
    assert "logical_num_modes" in doc
    assert "basis" in doc
    assert "partitions" not in doc
    # The generated typing stubs, where the build installs them (sanitizer trees skip them).
    stub = Path(_core.__file__).with_name("_core.pyi")
    if stub.exists():
        text = stub.read_text()
        assert "logical_num_modes: int" in text
        assert "partitions" not in text


@pytest.mark.parametrize("keyword", _UNSUPPORTED_KEYWORDS)
def test_no_thread_or_shard_keyword_is_accepted(keyword: str) -> None:
    # The high-level constructors and generated adapters never took any of them; the low-level one takes none either.
    for name, construct in _high_level_constructors().items():
        with pytest.raises(TypeError):
            construct(**{keyword: 2})
        assert construct() is not None, name
    with pytest.raises(TypeError):
        _core.MonomialPropagator032(**_ARGS, **{keyword: 2})


@pytest.fixture(scope="module")
def unset_partition_variable_run() -> dict[str, Any]:
    return _run(_full_spec("random_exact", "heisenberg"), 2)


@pytest.mark.parametrize(
    "value", ["off", "auto", "1", "2", "3", "0", "", "not-a-count"]
)
def test_obsolete_partition_variable_is_ignored(
    value: str, unset_partition_variable_run: dict[str, Any]
) -> None:
    # Set before the interpreter starts, as a launch would: the library never reads it, so every answer, aggregate
    # and shard-ordered block is bitwise what the same T gives with the variable unset.
    run = _run(
        _full_spec("random_exact", "heisenberg"),
        2,
        extra_env={"monoprop_PARTITIONS": value},
    )
    assert run["runtime"] == _RUNTIME
    differences = _differences(run["full"], unset_partition_variable_run["full"])
    assert not differences, differences[:10]


@pytest.mark.parametrize(
    "budget", ["", "0", "-1", "abc", "2,3", " 3", "+3", "2147483648"]
)
def test_invalid_budget_is_rejected(budget: str) -> None:
    out = _run(
        {"scenario": "controls", "construction_only": True},
        1,
        extra_env={"monoprop_NUM_THREADS": budget},
    )["controls"]
    assert out["default"].startswith("RuntimeError"), out["default"]
    assert "monoprop_NUM_THREADS" in out["default"]


def test_controls_and_lifecycle() -> None:
    results = {t: _run({"scenario": "controls"}, t) for t in _TEAMS}
    for t, result in results.items():
        assert result["runtime"] == _RUNTIME
        out = result["controls"]
        assert out["default"] == "ok"
        for key in (
            "logical_modes_zero",
            "logical_modes_too_large",
            "logical_modes_index",
        ):
            assert out[key].startswith("RuntimeError"), (t, key, out[key])
        assert out["logical_generator_index"].startswith(
            ("RuntimeError", "ValueError", "IndexError")
        ), out
        assert out["empty_operator"]["size"] == 0
        assert out["empty_operator"]["map"] == "ok"
        assert out["identity_generator"]["layers"] == 1
        # Rejected before mutation: the object stays usable.
        assert out["bad_parameter_count"] != "ok"
        assert out["usable_after_rejection"] == "ok"
        assert out["propagate_on_graph"].startswith("RuntimeError"), out[
            "propagate_on_graph"
        ]
        assert "non-empty graph" in out["propagate_on_graph"]
        # A missing Heisenberg term fails after the update started: the object is invalidated, copies refused.
        assert "not found" in out["missing_term_update"], out["missing_term_update"]
        assert "no longer usable" in out["after_missing_term"], out[
            "after_missing_term"
        ]
        assert "no longer usable" in out["copy_of_invalid"], out["copy_of_invalid"]
        # The replicated identity is added once, at every T.
        assert _f(out["core_only"]["energy"]) == 2.5
        assert out["core_only"]["map"] == {"[]": (2.5).hex()}
    for key in ("logical_energy",):
        values = [_f(results[t]["controls"][key]) for t in _TEAMS]
        assert all(_close(v, values[0]) for v in values), (key, values)
    for key in ("energy", "gradient"):
        values = [results[t]["controls"]["identity_generator"][key] for t in _TEAMS]
        assert all(v == values[0] for v in values)


def _full_spec(fixture: str, picture: str) -> dict[str, Any]:
    return {
        "scenario": "full",
        "fixture": fixture,
        "picture": picture,
        "informed_atol": 1e-3,
    }


def _check_single_run(full: dict[str, Any], picture: str) -> None:
    graph = full["graph"]
    # atol = 0 keeps stored zeros: one decoded entry per stored row, plus the Heisenberg core added by the binding.
    # A duplicated nonidentity row would collapse in the dictionary and break this count.
    core_entry = 1 if picture == "heisenberg" else 0
    assert len(graph["map0"]) == graph["aggregates"]["size"] + core_entry
    assert graph["python_terms"] == len(graph["map0"])
    assert len(graph["contract"]) == graph["aggregates"]["size"]
    assert (
        len(full["propagate"]["map0"])
        == full["propagate"]["aggregates"]["size"] + core_entry
    )
    for label in ("unpared", "pared"):
        assert graph[f"{label}_repeat"] is True
    # Direct evaluation, unpared functional and the energy-only path agree bitwise.
    assert graph["energy"] == graph["value"] == graph["unpared_value"]
    assert graph["gradient"] == graph["unpared_gradient"]
    assert graph["after_contract_energy"] == graph["energy"]
    # Independent copies: mutating the copy leaves the source's answers unchanged.
    assert full["copy"]["energy"] == graph["energy"]
    assert full["copy"]["source_energy"] == graph["energy"]
    assert full["incremental"]["after_contract_layers"] == 0
    assert "initial operator" in full["stale_functional"], full["stale_functional"]
    assert full["updated"]["energy"] == full["updated"]["functional"]


@pytest.mark.parametrize(("fixture", "picture"), _FIXTURES)
def test_full_api_agrees_across_team_sizes(fixture: str, picture: str) -> None:
    runs = {t: _run(_full_spec(fixture, picture), t) for t in _TEAMS}
    for t, run in runs.items():
        assert run["runtime"] == _RUNTIME, t
        _check_single_run(run["full"], picture)
    reference = runs[1]["full"]
    for t in _TEAMS[1:]:
        full = runs[t]["full"]
        for section in reference:
            if isinstance(reference[section], dict):
                for key, value in reference[section].items():
                    if key.startswith("map"):
                        _assert_map_close(
                            value, full[section][key], f"T={t} {section}.{key}"
                        )
            _assert_values_close(reference[section], full[section], f"T={t} {section}")
        # Global retained operators have one row per key at any geometry.
        for section in ("graph", "informed", "propagate"):
            assert (
                full[section]["aggregates"]["size"]
                == reference[section]["aggregates"]["size"]
            )
            assert (
                full[section]["aggregates"]["graph_layers"]
                == reference[section]["aggregates"]["graph_layers"]
            )


def _legacy_python() -> str | None:
    return os.environ.get("monoprop_TEST_LEGACY_PYTHON")  # noqa: SIM112 - the suite's names


@pytest.mark.skipif(
    _legacy_python() is None,
    reason="set monoprop_TEST_LEGACY_PYTHON to a separately built legacy environment's interpreter",
)
@pytest.mark.parametrize("threads", _TEAMS)
@pytest.mark.parametrize(("fixture", "picture"), _FIXTURES)
def test_matches_the_legacy_runtime_at_the_same_geometry(
    fixture: str, picture: str, threads: int
) -> None:
    legacy_python = _legacy_python()
    assert legacy_python is not None
    candidate = _run(_full_spec(fixture, picture), threads)
    # The legacy facade at partitions = T routes over the same (1, T) flat owners.
    legacy = _run(
        _full_spec(fixture, picture),
        threads,
        python=legacy_python,
        extra_env={"monoprop_PARTITIONS": str(threads)},
    )
    assert candidate["runtime"] == _RUNTIME
    assert legacy["runtime"] in (_LEGACY, None)
    assert candidate["core_file"] != legacy["core_file"]
    differences = _differences(candidate["full"], legacy["full"])
    assert not differences, differences[:10]


def _launcher() -> str | None:
    return shutil.which("mpiexec") or shutil.which("mpirun")


needs_mpi_launch = pytest.mark.skipif(
    not _core.has_mpi or _launcher() is None,
    reason="needs an MPI build and an MPI launcher on PATH",
)


def _run_ranks(
    spec: dict[str, Any],
    ranks: int,
    threads: int,
    out_root: Path,
    *,
    python: str = sys.executable,
    extra_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Launch the probe on `ranks` processes of `threads` threads; one document per rank, in rank order."""
    launcher = _launcher()
    assert launcher is not None
    out = out_root / f"p{ranks}-t{threads}-{uuid.uuid4().hex[:8]}"
    out.mkdir()
    env = _child_env(
        threads,
        {
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            # Leave binding to OpenMP: by default each rank would be bound to one core and its T threads squeezed.
            "OMPI_MCA_hwloc_base_binding_policy": "none",
            **(extra_env or {}),
        },
    )
    result = subprocess.run(  # noqa: S603 - trusted: a known launcher, interpreter and probe
        [
            launcher,
            "-n",
            str(ranks),
            python,
            str(_PROBE),
            json.dumps({**spec, "out": str(out)}),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    docs = [json.loads((out / f"rank{r}.json").read_text()) for r in range(ranks)]
    for r, doc in enumerate(docs):
        assert (doc["rank"], doc["ranks"]) == (r, ranks)
    return docs


_RANK_LOCAL = ("contract", "aggregates", "python_terms")


def _merge_ranks(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """One process's view of a multi-rank run: maps united, global values checked equal on every rank.

    Non-identity keys must be owned by exactly one rank; the replicated identity must agree. Aggregate sizes and layer
    counts are summed and checked equal, respectively; other rank-local values are dropped.
    """
    merged: dict[str, Any] = {}
    for section, first in docs[0]["full"].items():
        if not isinstance(first, dict):
            assert all(d["full"][section] == first for d in docs), section
            merged[section] = first
            continue
        out: dict[str, Any] = {}
        for key, value in first.items():
            if key.startswith("map"):
                union: dict[str, Any] = {}
                for d in docs:
                    for term, coeff in d["full"][section][key].items():
                        if term in union:
                            assert term == "[]", (section, key, term)
                            assert union[term] == coeff, (section, key, term)
                        union[term] = coeff
                out[key] = union
            elif key == "aggregates":
                out[key] = {
                    "size": sum(d["full"][section][key]["size"] for d in docs),
                    "graph_layers": value["graph_layers"],
                }
                assert all(
                    d["full"][section][key]["graph_layers"] == value["graph_layers"]
                    for d in docs
                )
            elif key in _RANK_LOCAL:
                continue
            else:
                # Energies, gradients and repeat flags are global: every rank returns the same bits.
                assert all(d["full"][section][key] == value for d in docs), (
                    section,
                    key,
                )
                out[key] = value
        merged[section] = out
    return merged


def _assert_full_close(
    reference: dict[str, Any], other: dict[str, Any], what: str
) -> None:
    for section, values in reference.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if key.startswith("map"):
                    _assert_map_close(
                        value, other[section][key], f"{what} {section}.{key}"
                    )
        _assert_values_close(values, other[section], f"{what} {section}")
    for section in ("graph", "informed", "propagate"):
        assert (
            other[section]["aggregates"]["size"]
            == reference[section]["aggregates"]["size"]
        ), (what, section)
        assert (
            other[section]["aggregates"]["graph_layers"]
            == reference[section]["aggregates"]["graph_layers"]
        ), (what, section)


def _single_process(fixture: str, picture: str, threads: int) -> dict[str, Any]:
    full = _run(_full_spec(fixture, picture), threads)["full"]
    return {
        section: (
            {
                **values,
                "aggregates": {
                    k: values["aggregates"][k] for k in ("size", "graph_layers")
                },
            }
            if isinstance(values, dict) and "aggregates" in values
            else values
        )
        for section, values in full.items()
    }


@needs_mpi_launch
@pytest.mark.parametrize(("fixture", "picture"), _FIXTURES)
def test_multirank_agrees_across_geometries(
    fixture: str, picture: str, tmp_path: Path
) -> None:
    reference = _single_process(fixture, picture, 4)
    for ranks, threads in ((2, 2), (4, 1)):
        docs = _run_ranks(_full_spec(fixture, picture), ranks, threads, tmp_path)
        for doc in docs:
            assert doc["runtime"] == _RUNTIME
            _check_single_run(doc["full"], picture)
        _assert_full_close(reference, _merge_ranks(docs), f"P={ranks} T={threads}")


@needs_mpi_launch
@pytest.mark.parametrize(
    ("fixture", "picture"), [_FIXTURES[0], _FIXTURES[3], _FIXTURES[6]]
)
def test_multirank_splitmix_three_processes(
    fixture: str, picture: str, tmp_path: Path
) -> None:
    reference = _single_process(fixture, picture, 6)
    docs = _run_ranks(
        _full_spec(fixture, picture),
        3,
        2,
        tmp_path,
        extra_env={"monoprop_ROUTING": "splitmix"},
    )
    _assert_full_close(reference, _merge_ranks(docs), "P=3 T=2 splitmix")


@needs_mpi_launch
def test_multirank_linear_routing_rejects_three_processes(tmp_path: Path) -> None:
    docs = _run_ranks(
        {"scenario": "controls", "construction_only": True},
        3,
        1,
        tmp_path,
    )
    for doc in docs:
        outcome = doc["controls"]["default"]
        assert outcome.startswith("RuntimeError"), outcome
        assert "power-of-two rank count" in outcome, outcome


@needs_mpi_launch
@pytest.mark.skipif(
    _legacy_python() is None,
    reason="set monoprop_TEST_LEGACY_PYTHON to a separately built legacy environment's interpreter",
)
@pytest.mark.parametrize("threads", _TEAMS)
@pytest.mark.parametrize(("fixture", "picture"), _FIXTURES)
def test_multirank_matches_the_legacy_runtime_at_the_same_geometry(
    fixture: str, picture: str, threads: int, tmp_path: Path
) -> None:
    legacy_python = _legacy_python()
    assert legacy_python is not None
    probe = subprocess.run(  # noqa: S603 - trusted interpreter
        [legacy_python, "-c", "import monoprop; print(monoprop.has_mpi)"],
        capture_output=True,
        text=True,
        check=True,
        env=_child_env(1),
    )
    if probe.stdout.strip() != "True":
        pytest.skip("the legacy interpreter is an MPI-off build")
    candidate = _run_ranks(_full_spec(fixture, picture), 2, threads, tmp_path)
    # The legacy runtime at P ranks x monoprop_PARTITIONS=T routes over the same (P, T) flat owners.
    legacy = _run_ranks(
        _full_spec(fixture, picture),
        2,
        threads,
        tmp_path,
        python=legacy_python,
        extra_env={"monoprop_PARTITIONS": str(threads)},
    )
    for rank, (ours, theirs) in enumerate(zip(candidate, legacy, strict=True)):
        assert ours["runtime"] == _RUNTIME
        assert theirs["runtime"] in (_LEGACY, None)
        differences = _differences(ours["full"], theirs["full"])
        assert not differences, (rank, differences[:10])
