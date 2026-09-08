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

"""Unit tests for the Bencher Metric Format exporter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from monoprop_bench_tools import bmf

if TYPE_CHECKING:
    from pathlib import Path

# The node id conftest and pytest-benchmark record, and the name it is exported under.
_ENERGY = "bench_random.py::test_random_energy[heisenberg]"
_ENERGY_NAME = "random/energy[heisenberg]"


def _write(results_dir: Path, label: str = "ci", **sections: object) -> None:
    """Write both artifacts a labelled run leaves behind."""
    timings = {
        "benchmarks": [
            {
                "fullname": f"benches/{_ENERGY}",
                "stats": {"mean": 0.5, "stddev": 0.01},
            }
        ]
    }
    (results_dir / f"time-{label}.json").write_text(json.dumps(timings))
    (results_dir / f"{label}.json").write_text(json.dumps(sections))


def test_latency_is_nanoseconds_with_stddev_bounds(tmp_path: Path) -> None:
    _write(tmp_path)
    latency = bmf.build_bmf(tmp_path, "ci")[_ENERGY_NAME]["latency"]

    assert latency == {
        "value": 0.5e9,
        "lower_value": 0.49e9,
        "upper_value": 0.51e9,
    }


def test_latency_omits_bounds_without_spread(tmp_path: Path) -> None:
    _write(tmp_path)
    timings = json.loads((tmp_path / "time-ci.json").read_text())
    timings["benchmarks"][0]["stats"]["stddev"] = 0.0
    (tmp_path / "time-ci.json").write_text(json.dumps(timings))

    assert bmf.build_bmf(tmp_path, "ci")[_ENERGY_NAME]["latency"] == {"value": 0.5e9}


def test_latency_lower_bound_floors_at_zero(tmp_path: Path) -> None:
    _write(tmp_path)
    timings = json.loads((tmp_path / "time-ci.json").read_text())
    timings["benchmarks"][0]["stats"]["stddev"] = 2.0
    (tmp_path / "time-ci.json").write_text(json.dumps(timings))

    assert bmf.build_bmf(tmp_path, "ci")[_ENERGY_NAME]["latency"]["lower_value"] == 0.0


def test_memory_joins_the_benchmark_of_the_same_operation(tmp_path: Path) -> None:
    _write(tmp_path, memhwm={_ENERGY: 1024})

    # conftest keys memhwm by the same node id pytest-benchmark reports, so both
    # measures must land on one benchmark rather than splitting into two.
    assert bmf.build_bmf(tmp_path, "ci")[_ENERGY_NAME] == {
        "latency": {"value": 0.5e9, "lower_value": 0.49e9, "upper_value": 0.51e9},
        "peak-memory": {"value": 1024.0},
    }


def test_operator_metrics_are_grouped_per_picture_and_model(tmp_path: Path) -> None:
    _write(
        tmp_path,
        opsize={"heisenberg": {"terms": 51450866}, "hubbard": {"terms": 12}},
    )
    result = bmf.build_bmf(tmp_path, "ci")

    assert result["operator[heisenberg]"] == {"terms": {"value": 51450866.0}}
    assert result["operator[hubbard]"] == {"terms": {"value": 12.0}}


def test_operator_metrics_land_on_the_benchmark_that_owns_the_operator(
    tmp_path: Path,
) -> None:
    build_graph = "bench_random.py::test_random_build_graph[heisenberg]"
    build_graph_name = "random/build_graph[heisenberg]"
    _write(
        tmp_path,
        opsize={"heisenberg": {"terms": 12}, build_graph: {"terms": 34}},
    )
    result = bmf.build_bmf(tmp_path, "ci")

    # A node-id key names a benchmark holding its own operator, so dropping it leaves that
    # operator's term count untracked -- at L1 that is both fixed-model `propagate` rows.
    assert set(result) == {_ENERGY_NAME, "operator[heisenberg]", build_graph_name}
    assert result[build_graph_name] == {"terms": {"value": 34.0}}
    assert result["operator[heisenberg]"] == {"terms": {"value": 12.0}}


def test_operator_metrics_join_the_timings_of_the_same_benchmark(
    tmp_path: Path,
) -> None:
    _write(tmp_path, opsize={_ENERGY: {"terms": 34}})
    result = bmf.build_bmf(tmp_path, "ci")

    # One benchmark, both measures: Bencher keys history on (benchmark, measure), so the
    # count has to share the name its latency is recorded under.
    assert result[_ENERGY_NAME] == {
        "latency": {"value": 0.5e9, "lower_value": 0.49e9, "upper_value": 0.51e9},
        "terms": {"value": 34.0},
    }


def test_missing_sections_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path)

    assert set(bmf.build_bmf(tmp_path, "ci")) == {_ENERGY_NAME}


def test_missing_artifact_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="missing artifact"):
        bmf.build_bmf(tmp_path, "absent")


def test_malformed_artifact_is_fatal(tmp_path: Path) -> None:
    _write(tmp_path)
    (tmp_path / "ci.json").write_text("{not json")

    with pytest.raises(SystemExit, match="malformed artifact"):
        bmf.build_bmf(tmp_path, "ci")


@pytest.mark.parametrize(
    ("node_id", "expected"),
    [
        ("bench_models.py::test_model_propagate[hubbard]", "model/propagate[hubbard]"),
        (
            "bench_random.py::test_random_build_graph[heisenberg]",
            "random/build_graph[heisenberg]",
        ),
        # Not a parameterised pytest id: left alone rather than rewritten into a name
        # Bencher would then key its history on forever.
        ("operator[heisenberg]", "operator[heisenberg]"),
        ("bench_x.py::helper_thing", "bench_x.py::helper_thing"),
        ("no_separator_at_all", "no_separator_at_all"),
    ],
)
def test_benchmark_name_drops_the_module_and_test_prefix(
    node_id: str, expected: str
) -> None:
    assert bmf.benchmark_name(node_id) == expected
