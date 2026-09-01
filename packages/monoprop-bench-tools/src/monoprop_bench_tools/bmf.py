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

:mod:`monoprop_bench_tools.report` renders every label side by side for humans;
this renders a single label for `Bencher <https://bencher.dev/>`_, which tracks
the metrics over time and alerts on regressions. Both read the same two artifacts
written by ``just bench <label>``: ``time-<label>.json`` (pytest-benchmark) and
``<label>.json`` (memory, operator sizes).

Bencher accepts one adapter per report, so timings and memory are merged here and
uploaded together under ``--adapter json``; the stock ``python_pytest`` adapter
would keep the timings and drop everything else.

Three measures are emitted:

``latency`` (nanoseconds)
    Per-operation mean, with the interval one standard deviation either side.
``peak-memory`` (bytes)
    Per-operation peak resident footprint, summed across ranks under MPI. The
    sum bounds the job; ``memhwm_max`` bounds a node. The two coincide for single-rank jobs.
``terms`` (count)
    Terms in the evolved operator. Deterministic for a fixed seed and problem
    size, so it is held to an exact match: it is the only check that a timing
    win is not an accuracy change.

Usage::

    monoprop-bench-bmf <results_dir> <label>
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

# ``opsize`` is also keyed by pytest node id (``::``, as report.py reads it) for a
# private A/B harness; those are not operators, so they are skipped here.
_NODE_ID_SEP = "::"

Metric = dict[str, float]
Bmf = dict[str, dict[str, Metric]]


def _read_json(path: Path) -> Any:
    """Return the parsed JSON at ``path``.

    Unlike :mod:`monoprop_bench_tools.report`, a missing or malformed artifact is fatal: a partial
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

    # ``memhwm`` is the kernel's exact peak RSS per operation, summed over ranks under MPI.
    for benchmark, peak in results.get("memhwm", {}).items():
        measure(benchmark, "peak-memory", peak)

    for key, size in results.get("opsize", {}).items():
        if _NODE_ID_SEP not in key:
            measure(f"{_OPERATOR}[{key}]", "terms", size["terms"])

    return bmf


def main(argv: list[str] | None = None) -> None:
    """Write ``label``'s BMF JSON to stdout."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        msg = "usage: monoprop-bench-bmf <results_dir> <label>"
        raise SystemExit(msg)
    results_dir, label = Path(argv[0]), argv[1]
    json.dump(build_bmf(results_dir, label), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
