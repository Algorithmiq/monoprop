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

"""Unit tests for the benchmark report builder (``benches/report.py``).

``benches`` is on the pytest pythonpath (see ``pyproject.toml``), so the module
imports as ``report`` from the normal test suite.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import report

if TYPE_CHECKING:
    from pathlib import Path


def test_picture_of_routes_by_tag() -> None:
    key = "bench_monoprop.py::test_random_energy"
    assert report._picture_of(f"{key}[schrodinger]") == "schrodinger"
    assert report._picture_of(f"{key}[heisenberg]") == "heisenberg"
    assert report._picture_of("bench_monoprop.py::test_static[hubbard]") == "heisenberg"


def test_display_op_strips_picture_and_names_static() -> None:
    assert (
        report._display_op("bench_monoprop.py::test_random_energy[heisenberg]")
        == "random / energy"
    )
    assert (
        report._display_op("bench_monoprop.py::test_random_build_graph[schrodinger]")
        == "random / build_graph"
    )
    assert (
        report._display_op("bench_monoprop.py::test_static[hubbard]")
        == "static / hubbard"
    )


def _write_timings(results_dir: Path) -> None:
    data = {
        "benchmarks": [
            {
                "fullname": "benches/bench_monoprop.py::test_random_energy[heisenberg]",
                "stats": {"mean": 0.001},
            },
            {
                "fullname": "benches/bench_monoprop.py::test_random_energy[schrodinger]",
                "stats": {"mean": 0.002},
            },
            {
                "fullname": "benches/bench_monoprop.py::test_static[hubbard]",
                "stats": {"mean": 1.5},
            },
        ]
    }
    (results_dir / "time-np1.json").write_text(json.dumps(data))


def _write_static_configs(results_dir: Path) -> None:
    (results_dir / "configs-np1.json").write_text(
        json.dumps(
            {
                "hubbard": {"num_sites": 60, "cutoff": 6, "lower_atol": 1e-5},
                "pauli": {"num_qubits": 127, "cutoff": 8, "lower_atol": 1e-4},
            }
        )
    )


def test_build_report_has_two_sections(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    md = report.build_report(tmp_path)

    assert "## Heisenberg" in md
    assert "## Schrödinger" in md
    # Heisenberg precedes Schrödinger.
    assert md.index("## Heisenberg") < md.index("## Schrödinger")

    heis = md[md.index("## Heisenberg") : md.index("## Schrödinger")]
    schr = md[md.index("## Schrödinger") :]

    # Static op only under Heisenberg; random energy appears in both sections.
    assert "static / hubbard" in heis
    assert "static / hubbard" not in schr
    assert "random / energy" in heis
    assert "random / energy" in schr


def test_build_report_includes_hyperparameters(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    (tmp_path / "params-np1.json").write_text(
        json.dumps(
            {
                "gen_length": 4,
                "obs_terms": 10000,
                "num_generators": 100,
                "num_modes": 128,
                "cutoff": 6,
                "seed": 0,
                "bench_rounds": 5,
            }
        )
    )
    md = report.build_report(tmp_path)

    assert "## Hyperparameters" in md
    # Each hyperparameter row is present with its value.
    assert "| num_generators | 100 |" in md
    assert "| cutoff | 6 |" in md
    # lower_atol is no longer a random hyperparameter (it is per-model now).
    assert "| lower_atol |" not in md
    # Hyperparameters sit between Configuration and the picture sections.
    assert md.index("## Hyperparameters") < md.index("## Heisenberg")


def test_fmt_config_value_formats_floats_compactly() -> None:
    assert report._fmt_config_value(1e-5) == "1e-05"
    assert report._fmt_config_value(1.0) == "1"
    assert report._fmt_config_value(0.2) == "0.2"
    assert report._fmt_config_value(60) == "60"
    assert report._fmt_config_value("up") == "up"


def test_build_report_includes_static_config(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    _write_static_configs(tmp_path)
    md = report.build_report(tmp_path)

    assert "## Static model configuration" in md
    assert "### hubbard" in md
    assert "### pauli" in md
    # Int field and compact-float field rendered in the right sub-tables.
    assert "| num_sites | 60 |" in md
    assert "| num_qubits | 127 |" in md
    assert "| lower_atol | 1e-05 |" in md
    # hubbard precedes pauli; the section precedes the picture sections.
    assert md.index("### hubbard") < md.index("### pauli")
    assert md.index("## Static model configuration") < md.index("## Heisenberg")


def test_static_config_section_absent_when_no_configs(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    md = report.build_report(tmp_path)
    assert "## Static model configuration" not in md


def test_build_report_includes_graph_size(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    (tmp_path / "graphsize-np1.json").write_text(
        json.dumps(
            {
                "heisenberg": {
                    "terms": 132220,
                    "n_cos_indices": 1051388,
                    "n_cycles": 122220,
                },
                "schrodinger": {
                    "terms": 11311942,
                    "n_cos_indices": 253573716,
                    "n_cycles": 294309,
                },
            }
        )
    )
    md = report.build_report(tmp_path)

    assert "## Graph size" in md
    # Term counts are rendered with thousands separators, one row per picture.
    assert "| Heisenberg | 132,220 |" in md
    assert "| Schrödinger | 11,311,942 |" in md


def test_build_report_omits_empty_schrodinger_section(tmp_path: Path) -> None:
    data = {
        "benchmarks": [
            {
                "fullname": "benches/bench_monoprop.py::test_static[pauli]",
                "stats": {"mean": 2.0},
            }
        ]
    }
    (tmp_path / "time-np1.json").write_text(json.dumps(data))
    md = report.build_report(tmp_path)
    assert "## Heisenberg" in md
    assert "## Schrödinger" not in md
