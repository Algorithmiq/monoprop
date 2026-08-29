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

"""Unit tests for the rung table, its gate, and the ladder collator.

Every gate is tested by being made to fire. A gate that has only ever been seen to pass has
not been shown to work: "the run was clean" and "the check never ran" are the same
observation from outside.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from monoprop_bench_tools import rungs

REPO = Path(__file__).resolve().parents[3]
TABLE = REPO / "benches" / "rungs.toml"

_CLEAN = {
    "id": "n1-clean",
    "family": "size",
    "picture": "heisenberg",
    "model": "hubbard",
    "ops": ["propagate"],
    "nodes": 1,
    "ranks_per_node": 8,
    "partitions": 16,
    "expect_terms": 1_000_000,
    "reps": 5,
}


def _table(tmp_path: Path, *entries: dict) -> Path:
    """Write a rung table holding ``entries`` and return its path."""
    body = "\n".join(
        "[[rung]]\n" + "\n".join(f"{k} = {json.dumps(v)}" for k, v in entry.items())
        for entry in entries
    )
    path = tmp_path / "rungs.toml"
    path.write_text(body + "\n")
    return path


def _results(**overrides) -> dict:
    """Return an artifact that passes every gate, before ``overrides`` are applied."""
    results = {
        "meta": {"nodes": 1, "ranks_per_node": 8, "partitions_env": "16"},
        "params": {"bench_rounds": 1},
        "opsize": {"hubbard": {"terms": 1_000_000}},
    }
    for section, value in overrides.items():
        results[section] = {**results.get(section, {}), **value}
    return results


# --------------------------------------------------------------------------- the table


def test_the_shipped_table_loads() -> None:
    table = rungs.load_rungs(TABLE)
    assert table
    assert {r.family for r in table.values()} <= rungs.FAMILIES


def test_the_shipped_table_carries_the_full_scaling_ladder() -> None:
    table = rungs.load_rungs(TABLE)
    scaling = [r for r in table.values() if r.family in {"strong", "weak"}]
    assert len(scaling) == 38, "the 2026-08-27 campaign is 21 weak and 17 strong rungs"
    assert all(r.calibrated for r in scaling), (
        "a ported rung carries its measured term count"
    )


def test_every_scaling_rung_declares_what_it_cost() -> None:
    """The ported campaign's measured cost travels with the row, not in a second file."""
    scaling = [r for r in rungs.load_rungs(TABLE).values() if r.family != "size"]
    assert len(scaling) == 38
    assert all(r.cost_seconds > 0 and r.cost_gib_per_node > 0 for r in scaling)
    assert all(r.cost != "?" for r in scaling)


def test_duplicate_ids_are_refused(tmp_path: Path) -> None:
    path = _table(tmp_path, _CLEAN, _CLEAN)
    with pytest.raises(SystemExit, match="duplicate rung id"):
        rungs.load_rungs(path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("family", "sizes", "family"),
        ("picture", "interaction", "picture"),
        ("model", "heisenberg", "model"),
        ("ops", ["pare"], "unknown ops"),
        ("ops", [], "ops is empty"),
        ("nodes", 0, "nodes must be"),
    ],
)
def test_an_unknown_value_is_refused(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    path = _table(tmp_path, {**_CLEAN, field: value})
    with pytest.raises(SystemExit, match=match):
        rungs.load_rungs(path)


def test_a_picture_is_refused_on_a_fixed_model(tmp_path: Path) -> None:
    path = _table(tmp_path, {**_CLEAN, "picture": "schrodinger"})
    with pytest.raises(SystemExit, match="only the random model has a picture axis"):
        rungs.load_rungs(path)


def test_a_missing_field_names_itself(tmp_path: Path) -> None:
    entry = {k: v for k, v in _CLEAN.items() if k != "reps"}
    path = _table(tmp_path, entry)
    with pytest.raises(SystemExit, match=r"missing.*reps"):
        rungs.load_rungs(path)


# ----------------------------------------------------------------------- the invocation


def test_the_selector_names_the_random_tests_and_the_picture(tmp_path: Path) -> None:
    entry = {**_CLEAN, "model": "random", "ops": ["energy", "gradient"]}
    rung = rungs.load_rungs(_table(tmp_path, entry))["n1-clean"]
    assert rung.bench_file == "benches/bench_random.py"
    assert (
        rung.selector() == "(test_random_energy or test_random_gradient) and heisenberg"
    )


def test_the_selector_names_the_model_tests_and_the_model(tmp_path: Path) -> None:
    rung = rungs.load_rungs(_table(tmp_path, _CLEAN))["n1-clean"]
    assert rung.bench_file == "benches/bench_models.py"
    assert rung.selector() == "(test_model_propagate) and hubbard"


def test_the_geometry_is_derived_not_stored(tmp_path: Path) -> None:
    rung = rungs.load_rungs(_table(tmp_path, {**_CLEAN, "nodes": 8}))["n1-clean"]
    assert (rung.ranks, rung.world) == (64, 1024)


def test_one_round_is_forced_on_the_command_line(tmp_path: Path) -> None:
    rung = rungs.load_rungs(_table(tmp_path, _CLEAN))["n1-clean"]
    argv = rungs.pytest_argv(rung, tmp_path, "n1-clean-r1")
    assert "--bench-rounds=1" in argv, (
        "a second round holds two propagators and doubles peak RSS"
    )


def test_the_partition_count_reaches_the_engine(tmp_path: Path) -> None:
    rung = rungs.load_rungs(_table(tmp_path, _CLEAN))["n1-clean"]
    env = rungs.bench_env(rung, tmp_path, "n1-clean-r1")
    assert env["monoprop_PARTITIONS"] == "16"
    assert env["monoprop_NUM_THREADS"] == "16"


# ------------------------------------------------------------------------------ the gate


def test_a_clean_artifact_passes(tmp_path: Path) -> None:
    rung = rungs.load_rungs(_table(tmp_path, _CLEAN))["n1-clean"]
    assert rungs.gate(rung, _results()) == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"meta": {"nodes": 2}}, "nodes"),
        ({"meta": {"ranks_per_node": 4}}, "ranks_per_node"),
        ({"meta": {"partitions_env": "128"}}, "partitions"),
        ({"params": {"bench_rounds": 5}}, "bench_rounds"),
        ({"opsize": {"hubbard": {"terms": 1_002_000}}}, "terms"),
    ],
)
def test_the_gate_fires(tmp_path: Path, overrides: dict, match: str) -> None:
    rung = rungs.load_rungs(_table(tmp_path, _CLEAN))["n1-clean"]
    reasons = rungs.gate(rung, _results(**overrides))
    assert any(match in reason for reason in reasons), reasons


def test_a_term_count_inside_the_tolerance_passes(tmp_path: Path) -> None:
    rung = rungs.load_rungs(_table(tmp_path, _CLEAN))["n1-clean"]
    assert rungs.gate(rung, _results(opsize={"hubbard": {"terms": 1_000_500}})) == []


def test_a_run_that_recorded_no_size_is_refused(tmp_path: Path) -> None:
    rung = rungs.load_rungs(_table(tmp_path, _CLEAN))["n1-clean"]
    results = _results()
    results["opsize"] = {}
    assert any("no operator size" in r for r in rungs.gate(rung, results))


def test_a_node_id_keyed_size_is_read_when_no_axis_key_exists(tmp_path: Path) -> None:
    rung = rungs.load_rungs(_table(tmp_path, _CLEAN))["n1-clean"]
    results = _results()
    results["opsize"] = {
        "bench_models.py::test_model_propagate[hubbard]": {"terms": 1_000_000}
    }
    assert rungs.gate(rung, results) == []


def test_an_uncalibrated_rung_refuses_to_run(tmp_path: Path) -> None:
    path = _table(tmp_path, {**_CLEAN, "expect_terms": 0})
    assert rungs.run_main([str(path), "n1-clean", "--dry-run"]) == 2


def test_an_unmeasured_size_knob_refuses_to_run(tmp_path: Path) -> None:
    """`TBD` on a knob is uncalibrated even where a term count was somehow filled in."""
    entry = {**_CLEAN, "args": ["--hubbard-cutoff=10", "--hubbard-lower-atol=TBD"]}
    path = _table(tmp_path, {**entry, "expect_terms": 0})
    assert rungs.run_main([str(path), "n1-clean", "--dry-run"]) == 2


def test_a_measured_size_with_an_unmeasured_knob_is_refused(tmp_path: Path) -> None:
    entry = {**_CLEAN, "args": ["--hubbard-lower-atol=TBD"]}
    with pytest.raises(SystemExit, match="unmeasured cannot have a measured size"):
        rungs.load_rungs(_table(tmp_path, entry))


def test_no_shipped_rung_pairs_a_term_count_with_an_unmeasured_knob() -> None:
    """The loader enforces it; this states it of the table that ships."""
    for rung in rungs.load_rungs(TABLE).values():
        unset = [a for a in rung.args if a.endswith(f"={rungs.UNSET}")]
        assert not unset or not rung.calibrated, rung.id


def test_a_calibrated_rung_dry_runs(tmp_path: Path) -> None:
    path = _table(tmp_path, _CLEAN)
    assert rungs.run_main([str(path), "n1-clean", "--dry-run"]) == 0


def test_an_unknown_rung_id_is_an_error(tmp_path: Path) -> None:
    path = _table(tmp_path, _CLEAN)
    assert rungs.run_main([str(path), "no-such-rung"]) == 2


# --------------------------------------------------------------------------- the ladder


def _rep(
    results_dir: Path, rung_id: str, rep: int, seconds: float, peak: int, **overrides
) -> None:
    """Write the two artifacts one rep of ``rung_id`` leaves behind."""
    label = f"{rung_id}-r{rep}"
    results = _results(**overrides)
    results["memhwm"] = {"x::y": peak}
    (results_dir / f"{label}.json").write_text(json.dumps(results))
    (results_dir / f"time-{label}.json").write_text(
        json.dumps({"benchmarks": [{"fullname": "x::y", "stats": {"median": seconds}}]})
    )


def test_the_ladder_collates_reps_on_the_median(tmp_path: Path) -> None:
    table = _table(tmp_path, _CLEAN)
    for rep, seconds in enumerate([10.0, 11.0, 40.0], start=1):
        _rep(tmp_path, "n1-clean", rep, seconds, peak=rungs.BYTES_PER_GIB)
    (cell,) = rungs.collate(rungs.load_rungs(table), tmp_path)
    assert cell.reps == 3
    md = rungs.render([cell])
    assert "| 11.00 | 10.00 |" in md, "median then min, never the mean"
    assert "| 1.0 | — | — |" in md, "one summed GiB over one node, and no declared cost"


def test_the_ladder_drops_a_rep_the_gate_refuses(tmp_path: Path) -> None:
    table = _table(tmp_path, _CLEAN)
    _rep(tmp_path, "n1-clean", 1, 10.0, peak=rungs.BYTES_PER_GIB)
    _rep(
        tmp_path,
        "n1-clean",
        2,
        10.0,
        peak=rungs.BYTES_PER_GIB,
        opsize={"hubbard": {"terms": 9}},
    )
    (cell,) = rungs.collate(rungs.load_rungs(table), tmp_path)
    assert cell.reps == 1


def test_the_ladder_shows_the_declared_cost(tmp_path: Path) -> None:
    table = _table(tmp_path, {**_CLEAN, "cost_seconds": 10.0, "cost_gib_per_node": 4.0})
    _rep(tmp_path, "n1-clean", 1, 12.0, peak=rungs.BYTES_PER_GIB)
    cells = rungs.collate(rungs.load_rungs(table), tmp_path)
    assert "| 10.00 | 1.200x |" in rungs.render(cells)


def test_a_rung_nobody_has_timed_shows_no_ratio(tmp_path: Path) -> None:
    table = _table(tmp_path, _CLEAN)
    _rep(tmp_path, "n1-clean", 1, 12.0, peak=rungs.BYTES_PER_GIB)
    md = rungs.render(rungs.collate(rungs.load_rungs(table), tmp_path))
    assert "| — | — |" in md, "no declared cost must print no ratio, never 0.000x"


def test_the_ladder_describes_the_problem_it_measured(tmp_path: Path) -> None:
    """A number without its parameters is not reproducible from the block alone."""
    table = _table(tmp_path, _CLEAN)
    _rep(
        tmp_path,
        "n1-clean",
        1,
        10.0,
        peak=rungs.BYTES_PER_GIB,
        configs={"hubbard": {"num_sites": 12, "cutoff": 10, "lower_atol": 1.25e-05}},
    )
    md = rungs.render(rungs.collate(rungs.load_rungs(table), tmp_path))
    assert "## Problems measured" in md
    assert "cutoff=10" in md
    assert "lower_atol=1.25e-05" in md
    assert "num_sites=12" in md


def test_the_random_problem_is_described_from_params(tmp_path: Path) -> None:
    """The random model's parameters live in `params`, not in `configs`."""
    entry = {**_CLEAN, "model": "random", "args": []}
    table = _table(tmp_path, entry)
    _rep(
        tmp_path,
        "n1-clean",
        1,
        10.0,
        peak=rungs.BYTES_PER_GIB,
        opsize={"heisenberg": {"terms": 1_000_000}},
        params={"bench_rounds": 1, "num_generators": 1000, "cutoff": 6},
    )
    md = rungs.render(rungs.collate(rungs.load_rungs(table), tmp_path))
    assert "num_generators=1000" in md


def test_the_ladder_orders_a_rung_by_node_count(tmp_path: Path) -> None:
    """A ladder is read down the page, so n2 must not sort between n1 and n16."""
    counts = (1, 2, 16)
    entries = [{**_CLEAN, "id": f"strong-n{n}", "nodes": n} for n in counts]
    table = _table(tmp_path, *entries)
    for n in counts:
        _rep(
            tmp_path,
            f"strong-n{n}",
            1,
            10.0,
            peak=rungs.BYTES_PER_GIB,
            meta={"nodes": n},
        )
    md = rungs.render(rungs.collate(rungs.load_rungs(table), tmp_path))
    order = [
        line.split("|")[1].strip()
        for line in md.splitlines()
        if line.startswith("| strong-n")
    ]
    assert order == ["strong-n1", "strong-n2", "strong-n16"]


def test_the_table_is_valid_toml() -> None:
    assert tomllib.loads(TABLE.read_text())["rung"]
