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

Each run contributes two artifacts to the results directory: ``time-<label>.json``
(pytest-benchmark timings) and ``<label>.json`` (everything else, keyed by section).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import report

if TYPE_CHECKING:
    from pathlib import Path


def _collapse(text: str) -> str:
    """Collapse runs of spaces so table assertions ignore tabulate's padding."""
    return re.sub(r" +", " ", text)


def test_picture_of_routes_by_tag() -> None:
    key = "bench_random.py::test_random_energy"
    assert report._picture_of(f"{key}[schrodinger]") == "schrodinger"
    assert report._picture_of(f"{key}[heisenberg]") == "heisenberg"
    assert report._picture_of("bench_models.py::test_model[hubbard]") == "heisenberg"


def test_display_op_strips_picture_and_names_model() -> None:
    assert (
        report._display_op("bench_random.py::test_random_energy[heisenberg]")
        == "random / energy"
    )
    assert (
        report._display_op("bench_random.py::test_random_build_graph[schrodinger]")
        == "random / build_graph"
    )
    assert (
        report._display_op("bench_models.py::test_model[hubbard]") == "model / hubbard"
    )


def _write_timings(results_dir: Path, label: str = "np1") -> None:
    data = {
        "benchmarks": [
            {
                "fullname": "benches/bench_random.py::test_random_energy[heisenberg]",
                "stats": {"mean": 0.001},
            },
            {
                "fullname": "benches/bench_random.py::test_random_energy[schrodinger]",
                "stats": {"mean": 0.002},
            },
            {
                "fullname": "benches/bench_models.py::test_model[hubbard]",
                "stats": {"mean": 1.5},
            },
        ]
    }
    (results_dir / f"time-{label}.json").write_text(json.dumps(data))


def _write_results(results_dir: Path, label: str = "np1", **sections: object) -> None:
    (results_dir / f"{label}.json").write_text(json.dumps(sections))


def test_build_report_has_two_sections(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    md = _collapse(report.build_report(tmp_path))

    assert "## Heisenberg" in md
    assert "## Schrödinger" in md
    assert md.index("## Heisenberg") < md.index("## Schrödinger")

    heis = md[md.index("## Heisenberg") : md.index("## Schrödinger")]
    schr = md[md.index("## Schrödinger") :]

    assert "model / hubbard" in heis
    assert "model / hubbard" not in schr
    assert "random / energy" in heis
    assert "random / energy" in schr


def test_build_report_includes_hyperparameters(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    _write_results(
        tmp_path,
        params={
            "gen_length": 4,
            "obs_terms": 10000,
            "num_generators": 100,
            "num_modes": 128,
            "cutoff": 6,
            "seed": 0,
            "bench_rounds": 5,
        },
    )
    md = _collapse(report.build_report(tmp_path))

    assert "## Hyperparameters" in md
    assert "| num_generators | 100 |" in md
    assert "| cutoff | 6 |" in md
    # lower_atol is per-model, not a sampled hyperparameter.
    assert "| lower_atol |" not in md
    assert md.index("## Hyperparameters") < md.index("## Heisenberg")


def test_fmt_config_formats_floats_compactly() -> None:
    assert report._fmt_config(1e-5) == "1e-05"
    assert report._fmt_config(1.0) == "1"
    assert report._fmt_config(0.2) == "0.2"
    assert report._fmt_config(60) == "60"
    assert report._fmt_config("up") == "up"


def test_build_report_includes_model_config(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    _write_results(
        tmp_path,
        configs={
            "hubbard": {"num_sites": 60, "cutoff": 6, "lower_atol": 1e-5},
            "pauli": {"num_qubits": 127, "cutoff": 8, "lower_atol": 1e-4},
        },
    )
    md = _collapse(report.build_report(tmp_path))

    assert "## Model configuration" in md
    assert "### hubbard" in md
    assert "### pauli" in md
    assert "| num_sites | 60 |" in md
    assert "| num_qubits | 127 |" in md
    assert "| lower_atol | 1e-05 |" in md
    assert md.index("### hubbard") < md.index("### pauli")
    assert md.index("## Model configuration") < md.index("## Heisenberg")


def test_model_config_section_absent_when_no_configs(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    md = _collapse(report.build_report(tmp_path))
    assert "## Model configuration" not in md


def test_build_report_includes_operator_size(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    _write_results(
        tmp_path,
        opsize={
            "heisenberg": {"terms": 132220},
            "schrodinger": {"terms": 11311942},
        },
    )
    md = _collapse(report.build_report(tmp_path))

    assert "## Operator size" in md
    assert "| Heisenberg | 132,220 |" in md
    assert "| Schrödinger | 11,311,942 |" in md


def test_build_report_includes_memory(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    _write_results(
        tmp_path,
        mem={
            "bench_random.py::test_random_energy[heisenberg]": 52428800,
            "bench_random.py::test_random_energy[schrodinger]": 104857600,
        },
    )
    md = _collapse(report.build_report(tmp_path))

    assert "Memory (RSS)" in md
    assert "50.00 MiB" in md
    assert "100.00 MiB" in md


def test_build_report_includes_resting(tmp_path: Path) -> None:
    _write_timings(tmp_path)
    _write_results(
        tmp_path,
        memrest={"heisenberg": 52428800},
    )
    md = _collapse(report.build_report(tmp_path))

    assert "## Operator resting footprint (RSS)" in md
    assert "| Heisenberg | 50.00 MiB |" in md


def test_build_report_sorts_labels_numerically(tmp_path: Path) -> None:
    for label in ("np1", "np2", "np10"):
        _write_timings(tmp_path, label)
    md = _collapse(report.build_report(tmp_path))
    # Natural sort: np2 before np10 (not lexicographic np1, np10, np2).
    assert md.index("np1") < md.index("np2") < md.index("np10")


def test_build_report_omits_empty_schrodinger_section(tmp_path: Path) -> None:
    data = {
        "benchmarks": [
            {
                "fullname": "benches/bench_models.py::test_model[pauli]",
                "stats": {"mean": 2.0},
            }
        ]
    }
    (tmp_path / "time-np1.json").write_text(json.dumps(data))
    md = _collapse(report.build_report(tmp_path))
    assert "## Heisenberg" in md
    assert "## Schrödinger" not in md
