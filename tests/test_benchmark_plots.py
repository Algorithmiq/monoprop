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

"""Regression tests for benchmark plot labels and point alignment."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

BENCHMARKS = Path(__file__).parents[1] / "packages" / "bench-third-party"


@pytest.fixture
def plot_script(monkeypatch):
    figures = []
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "savefig",
        lambda figure, *_args, **_kwargs: figures.append(figure),
    )

    def run(relative_path):
        path = BENCHMARKS / relative_path
        monkeypatch.syspath_prepend(str(path.parent))
        monkeypatch.setattr("sys.argv", [str(path)])
        runpy.run_path(str(path), run_name="__main__")
        return figures

    yield run
    plt.close("all")


def _assert_pauli_points(ax, metric):
    data = json.loads((BENCHMARKS / "pauli_prop" / "results.json").read_text())
    assert ax.get_xlabel() == "Trotter step"
    for line in ax.lines:
        assert list(line.get_xdata()) == list(range(5, 29))
        values = data[metric][line.get_label()]
        offset = 3 if metric == "runtime" else 4
        assert list(line.get_ydata()) == values[offset:]
        assert line.get_ydata()[-1] == values[-1]


def test_headline_labels_and_points(plot_script):
    (figure,) = plot_script("plot_headline.py")
    left, right = figure.axes
    _assert_pauli_points(left, "runtime")
    assert left.get_ylabel() == "time per step [s]"
    assert left.get_title() == "Pauli: 144-qubit 2D tilted-field Ising"
    assert right.get_ylabel() == "cumulative time [s]"
    data = json.loads((BENCHMARKS / "majorana_prop" / "results.json").read_text())
    for line in right.lines:
        assert list(line.get_xdata()) == data["step_range"][4:]
        assert list(line.get_ydata()) == data["runtime_seconds"][line.get_label()][4:]


def test_pauli_runtime_and_memory_alignment(plot_script):
    (figure,) = plot_script("pauli_prop/plot_results.py")
    runtime, memory = figure.axes
    _assert_pauli_points(runtime, "runtime")
    _assert_pauli_points(memory, "memory")


def test_majorana_runtime_labels_and_values(plot_script):
    combined, standalone = plot_script("majorana_prop/plot_results.py")
    data = json.loads((BENCHMARKS / "majorana_prop" / "results.json").read_text())
    for ax in (combined.axes[0], standalone.axes[0]):
        assert ax.get_ylabel() == "cumulative time (seconds)"
        assert "cumulative runtime" in ax.get_title().lower()
        for line in ax.lines:
            assert list(line.get_xdata()) == data["step_range"]
            assert list(line.get_ydata()) == data["runtime_seconds"][line.get_label()]
