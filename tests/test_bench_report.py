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
