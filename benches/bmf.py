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

"""Convert one label's benchmark artifacts to Bencher Metric Format (BMF) JSON.

``report.py`` renders every label side by side for humans; this renders a single
label for `Bencher <https://bencher.dev/>`_, which tracks the metrics over time
and alerts on regressions. Both read the same two artifacts written by
``just bench <label>``: ``time-<label>.json`` (pytest-benchmark) and
``<label>.json`` (memory, operator sizes).

Bencher accepts one adapter per report, so timings and memory are merged here and
uploaded together under ``--adapter json``; the stock ``python_pytest`` adapter
would keep the timings and drop everything else.

Four measures are emitted:

``latency`` (nanoseconds)
    Per-operation mean, with the interval one standard deviation either side.
``peak-memory`` (bytes)
    Per-operation peak resident footprint, summed across ranks under MPI.
``terms`` (count)
    Terms in the evolved operator. Deterministic for a fixed seed and problem
    size, so it can carry a far tighter threshold than the timing measures.
``resting-memory`` (bytes)
    Settled footprint of the built operator + graph.

Usage::

    uv run --no-sync python benches/bmf.py benches/results <label>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

_NS_PER_S = 1e9

# Benchmark name prefix for the per-picture / per-model operator metrics, which
# describe a built operator rather than one timed call.
_OPERATOR = "operator"

Metric = dict[str, float]
Bmf = dict[str, dict[str, Metric]]


def _read_json(path: Path) -> Any:
    """Return the parsed JSON at ``path``.

    Unlike ``report.py``, a missing or malformed artifact is fatal: a partial
    upload would silently seed Bencher's history with the wrong baseline.
    """
    if not path.is_file():
        msg = f"bmf: missing artifact {path}"
        raise SystemExit(msg)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        msg = f"bmf: malformed artifact {path}: {exc}"
        raise SystemExit(msg) from exc


def _latency(stats: dict[str, float]) -> Metric:
    """Return the ``latency`` metric (ns) for one pytest-benchmark ``stats``.

    pytest-benchmark reports seconds; Bencher's latency measure is nanoseconds.
    The bounds are one standard deviation either side of the mean, floored at
    zero because a noisy round can otherwise push the lower bound negative.
    """
    mean = stats["mean"] * _NS_PER_S
    stddev = stats.get("stddev", 0.0) * _NS_PER_S
    if stddev <= 0.0:  # a single round has no spread to report
        return {"value": mean}
    return {
        "value": mean,
        "lower_value": max(0.0, mean - stddev),
        "upper_value": mean + stddev,
    }


def _timings(results_dir: Path, label: str) -> Iterator[tuple[str, Metric]]:
    """Yield ``(benchmark, latency)`` from ``time-<label>.json``.

    The leading directory is stripped from ``fullname`` so the benchmark name
    does not depend on the directory pytest was invoked from -- Bencher keys its
    history on the name, so it has to be stable across runners.
    """
    data = _read_json(results_dir / f"time-{label}.json")
    for bench in data.get("benchmarks", []):
        yield bench["fullname"].split("/")[-1], _latency(bench["stats"])


def build_bmf(results_dir: Path, label: str) -> Bmf:
    """Return the BMF JSON object for ``label``'s artifacts in ``results_dir``."""
    results = _read_json(results_dir / f"{label}.json")
    bmf: Bmf = {}

    def measure(benchmark: str, name: str, value: float) -> None:
        bmf.setdefault(benchmark, {})[name] = {"value": float(value)}

    for benchmark, latency in _timings(results_dir, label):
        bmf.setdefault(benchmark, {})["latency"] = latency

    # ``memhwm`` is the kernel's exact peak RSS per operation (summed over ranks
    # under MPI), which is what REPORT.md shows; ``mem`` is the peak-of-sum PSS
    # lower bound for the same window and would only duplicate the signal.
    for benchmark, peak in results.get("memhwm", {}).items():
        measure(benchmark, "peak-memory", peak)

    for key, size in results.get("opsize", {}).items():
        measure(f"{_OPERATOR}[{key}]", "terms", size["terms"])

    for key, resting in results.get("memrest", {}).items():
        measure(f"{_OPERATOR}[{key}]", "resting-memory", resting)

    return bmf


def main(argv: list[str]) -> None:
    """Write ``label``'s BMF JSON to stdout."""
    if len(argv) != 2:
        msg = "usage: bmf.py <results_dir> <label>"
        raise SystemExit(msg)
    results_dir, label = Path(argv[0]), argv[1]
    json.dump(build_bmf(results_dir, label), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
