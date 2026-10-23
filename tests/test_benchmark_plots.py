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

"""Snapshot and alignment checks for the benchmark figures shown in the docs.

Each plot script is re-run on the committed results and its output compared, as an image, with
the figure committed under ``docs/public/benchmarks``: a change to a script, or to its results,
that is not followed by regenerating the figures fails here. The alignment checks pin down which
measurement lands on which Trotter step, which a pixel comparison only catches as "different".
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
compare_images = pytest.importorskip("matplotlib.testing.compare").compare_images

ROOT = Path(__file__).parents[1]
BENCHMARKS = ROOT / "packages" / "bench-third-party"
COMMITTED_FIGURES = ROOT / "docs" / "public" / "benchmarks"

# Figures each script writes, by file name, as committed under COMMITTED_FIGURES.
FIGURES = {
    "plot_headline.py": ["headline_runtime.png"],
    "pauli_prop/plot_results.py": ["pauli_results.png"],
    "majorana_prop/plot_results.py": ["majorana_results.png", "majorana_runtime.png"],
}

# RMS over 0-255 pixel values. The docs environment and the benchmark package lock the same
# matplotlib and pillow, so the figures reproduce exactly; this only absorbs anti-aliasing noise.
TOLERANCE = 0.5


@pytest.fixture
def plot_script(monkeypatch, tmp_path):
    """Run a plot script as `__main__`, saving its figures into `tmp_path` by file name."""
    figures = {}
    savefig = matplotlib.figure.Figure.savefig

    def redirect(figure, fname, *args, **kwargs):
        out = tmp_path / Path(fname).name
        savefig(figure, out, *args, **kwargs)
        figures[out.name] = figure

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", redirect)
    monkeypatch.syspath_prepend(str(BENCHMARKS))

    def run(relative_path):
        path = BENCHMARKS / relative_path
        monkeypatch.syspath_prepend(str(path.parent))
        monkeypatch.setattr("sys.argv", [str(path)])
        # A local matplotlibrc must not restyle the figures being compared.
        with matplotlib.rc_context():
            matplotlib.rcdefaults()
            runpy.run_path(str(path), run_name="__main__")
        return figures

    yield run
    plt.close("all")


@pytest.mark.parametrize("script", FIGURES)
def test_figures_match_committed(plot_script, tmp_path, script):
    figures = plot_script(script)
    assert sorted(figures) == sorted(FIGURES[script])
    for name in FIGURES[script]:
        # On failure, compare_images leaves the diff image next to the regenerated figure.
        failure = compare_images(
            str(COMMITTED_FIGURES / name), str(tmp_path / name), tol=TOLERANCE
        )
        assert failure is None, (
            f"{name} no longer matches the committed figure; regenerate it with "
            f"{script} and copy it to docs/public/benchmarks.\n{failure}"
        )


def _assert_pauli_points(ax, metric):
    # Iteration i of the committed run is Trotter step i + 1; runtime has no entry for the
    # warm-up iteration, so its first value is step 2. Both plots start at step 5.
    data = json.loads((BENCHMARKS / "pauli_prop" / "results.json").read_text())
    offset = 3 if metric == "runtime" else 4
    for line in ax.lines:
        assert list(line.get_xdata()) == list(range(5, 29))
        assert list(line.get_ydata()) == data[metric][line.get_label()][offset:]


def test_headline_points(plot_script):
    (figure,) = plot_script("plot_headline.py").values()
    pauli, majorana = figure.axes
    _assert_pauli_points(pauli, "runtime")
    data = json.loads((BENCHMARKS / "majorana_prop" / "results.json").read_text())
    for line in majorana.lines:
        assert list(line.get_xdata()) == data["step_range"][4:]
        assert list(line.get_ydata()) == data["runtime_seconds"][line.get_label()][4:]


def test_pauli_runtime_and_memory_points(plot_script):
    (figure,) = plot_script("pauli_prop/plot_results.py").values()
    runtime, memory = figure.axes
    _assert_pauli_points(runtime, "runtime")
    _assert_pauli_points(memory, "memory")


def test_majorana_runtime_points(plot_script):
    figures = plot_script("majorana_prop/plot_results.py")
    data = json.loads((BENCHMARKS / "majorana_prop" / "results.json").read_text())
    for ax in (
        figures["majorana_results.png"].axes[0],
        figures["majorana_runtime.png"].axes[0],
    ):
        for line in ax.lines:
            assert list(line.get_xdata()) == data["step_range"]
            assert list(line.get_ydata()) == data["runtime_seconds"][line.get_label()]
