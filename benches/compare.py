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

"""A/B regression check between two sets of benchmark labels.

``report.py`` renders every label side by side but computes no deltas. This tool
compares a *baseline* set of labels (``--a``) against a *candidate* set (``--b``)
and decides pass/fail, filling the gap for validating a refactor has no perf
regression.

Label scheme (from the A/B sweep driver): ``<side>-<cfg>-r<rep>``, e.g.
``A-OV-r2``. Labels are grouped by ``cfg`` so the same operation measured at
different problem sizes is never pooled together; repeats (``-r<rep>``) of one
``cfg`` ARE pooled.

For each ``(cfg, operation)`` present on both sides it reports:

* **time**: ``min(candidate) / min(baseline)`` over the pooled per-round samples.
  The minimum is the estimator of choice — measurement noise on a shared,
  turbo-uncontrolled box is strictly additive, so minima converge to the true
  cost while means do not. A ratio above ``1 + --time-threshold`` is a
  regression (``1 + --model-threshold`` for configs with few pooled samples,
  e.g. the ``-m slow`` fixed models that run a single round).
* **memory**: median peak-RSS ratio across the pooled labels, flagged above
  ``1 + --mem-threshold``.

It also enforces a hard **equivalence gate**: with identical seeds both sides
must evolve the *exact* same number of terms per picture; any mismatch is a
semantic change in the candidate, not a perf question, and fails the run.

Exit status is ``1`` if any time regression or term-count mismatch is found,
so this doubles as a CI gate.

Usage::

    uv run --no-sync python benches/compare.py <results_dir> --a 'A-*' --b 'B-*'
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from tabulate import tabulate

# <side>-<cfg>-r<rep>; cfg may contain dashes, rep is the trailing -r<digits>.
_LABEL_RE = re.compile(r"^(?P<side>[^-]+)-(?P<cfg>.+)-r(?P<rep>\d+)$")


def _read_json(path: Path) -> Any:
    """Return the parsed JSON at ``path``, or ``None`` if empty/malformed."""
    text = path.read_text()
    if not text.strip():
        print(f"compare: skipping empty artifact {path.name}", file=sys.stderr)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(
            f"compare: skipping malformed artifact {path.name}: {exc}", file=sys.stderr
        )
        return None


def _cfg_of(label: str) -> str:
    """Return the ``cfg`` token of a ``<side>-<cfg>-r<rep>`` label.

    Falls back to the whole label (minus a trailing ``-r<rep>``) when it does not
    match the scheme, so ad-hoc labels still group sanely.
    """
    m = _LABEL_RE.match(label)
    if m:
        return m.group("cfg")
    return re.sub(r"-r\d+$", "", label)


def _labels_in(results_dir: Path) -> list[str]:
    """Every label with a ``time-<label>.json`` artifact in ``results_dir``."""
    return sorted(p.stem.removeprefix("time-") for p in results_dir.glob("time-*.json"))


def _match(labels: list[str], pattern: str) -> list[str]:
    return [lbl for lbl in labels if fnmatch.fnmatch(lbl, pattern)]


def _op_samples(results_dir: Path, label: str) -> dict[str, list[float]]:
    """Return ``{op_key: [per-round seconds]}`` for one label's ``time-*.json``."""
    data = _read_json(results_dir / f"time-{label}.json")
    if data is None:
        return {}
    out: dict[str, list[float]] = {}
    for b in data.get("benchmarks", []):
        op = b["fullname"].split("/")[-1]
        stats = b.get("stats", {})
        samples = stats.get("data") or ([stats["min"]] if "min" in stats else [])
        if samples:
            out[op] = [float(x) for x in samples]
    return out


def _label_json(results_dir: Path, label: str) -> dict:
    """Return the ``<label>.json`` metadata artifact (mem / opsize / ...)."""
    data = _read_json(results_dir / f"{label}.json")
    return data if isinstance(data, dict) else {}


def _pool(
    results_dir: Path, labels: list[str]
) -> tuple[
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], list[int]],
    dict[tuple[str, str], list[int]],
]:
    """Pool a side's labels into keyed maps.

    Returns ``(times, mems, sizes)`` where ``times[(cfg, op)]`` is every pooled
    per-round sample, ``mems[(cfg, op)]`` the per-label peak-RSS values, and
    ``sizes[(cfg, picture)]`` the per-label term counts.
    """
    times: dict[tuple[str, str], list[float]] = {}
    mems: dict[tuple[str, str], list[int]] = {}
    sizes: dict[tuple[str, str], list[int]] = {}
    for label in labels:
        cfg = _cfg_of(label)
        for op, samples in _op_samples(results_dir, label).items():
            times.setdefault((cfg, op), []).extend(samples)
        meta = _label_json(results_dir, label)
        for op, num_bytes in (meta.get("mem") or {}).items():
            mems.setdefault((cfg, op), []).append(int(num_bytes))
        for picture, info in (meta.get("opsize") or {}).items():
            if isinstance(info, dict) and "terms" in info:
                sizes.setdefault((cfg, picture), []).append(int(info["terms"]))
    return times, mems, sizes


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f}ms"
    return f"{seconds:.3f}s"


def _fmt_mem(num_bytes: float | None) -> str:
    return "—" if num_bytes is None else f"{num_bytes / 1024 / 1024:.1f}MiB"


def _ratio(candidate: float, baseline: float) -> float | None:
    return None if baseline <= 0 else candidate / baseline


_TIME_HEADERS = [
    "cfg",
    "operation",
    "n A/B",
    "A min",
    "B min",
    "time B/A",
    "A RSS",
    "B RSS",
    "RSS B/A",
    "verdict",
]
_SIZE_HEADERS = ["cfg", "picture", "A terms", "B terms", "verdict"]


def _timing_rows(
    a_times: dict[tuple[str, str], list[float]],
    b_times: dict[tuple[str, str], list[float]],
    a_mems: dict[tuple[str, str], list[int]],
    b_mems: dict[tuple[str, str], list[int]],
    time_threshold: float,
    model_threshold: float,
    mem_threshold: float,
    min_samples: int,
) -> tuple[list[list[str]], list[str]]:
    """Build the per-(cfg, op) timing/memory table rows; return ``(rows, regressions)``."""
    rows: list[list[str]] = []
    regressions: list[str] = []
    for key in sorted(set(a_times) & set(b_times)):
        cfg, op = key
        a_s, b_s = a_times[key], b_times[key]
        a_min, b_min = min(a_s), min(b_s)
        r = _ratio(b_min, a_min)
        thr = (
            time_threshold
            if min(len(a_s), len(b_s)) >= min_samples
            else model_threshold
        )
        a_mem = statistics.median(a_mems[key]) if key in a_mems else None
        b_mem = statistics.median(b_mems[key]) if key in b_mems else None
        mem_r = _ratio(b_mem, a_mem) if a_mem and b_mem else None
        if r is not None and r > 1 + thr:
            verdict = "REGRESSION"
            regressions.append(f"{cfg} / {op}: {r:.3f}x")
        elif mem_r is not None and mem_r > 1 + mem_threshold:
            verdict = "mem?"
        else:
            verdict = "OK"
        rows.append(
            [
                cfg,
                op,
                f"{len(a_s)}/{len(b_s)}",
                _fmt_time(a_min),
                _fmt_time(b_min),
                "—" if r is None else f"{r:.3f}x",
                _fmt_mem(a_mem),
                _fmt_mem(b_mem),
                "—" if mem_r is None else f"{mem_r:.3f}x",
                verdict,
            ]
        )
    return rows, regressions


def _size_rows(
    a_sizes: dict[tuple[str, str], list[int]], b_sizes: dict[tuple[str, str], list[int]]
) -> tuple[list[list[str]], list[str]]:
    """Build the term-count equivalence rows; return ``(rows, mismatches)``."""
    rows: list[list[str]] = []
    mismatches: list[str] = []
    for key in sorted(set(a_sizes) & set(b_sizes)):
        cfg, picture = key
        a_terms, b_terms = set(a_sizes[key]), set(b_sizes[key])
        equal = a_terms == b_terms and len(a_terms) == 1
        if not equal:
            mismatches.append(
                f"{cfg}/{picture}: A={sorted(a_terms)} B={sorted(b_terms)}"
            )
        rows.append(
            [
                cfg,
                picture,
                ",".join(f"{t:,}" for t in sorted(a_terms)),
                ",".join(f"{t:,}" for t in sorted(b_terms)),
                "OK" if equal else "MISMATCH",
            ]
        )
    return rows, mismatches


def compare(
    results_dir: Path,
    a_pattern: str,
    b_pattern: str,
    time_threshold: float,
    model_threshold: float,
    mem_threshold: float,
    min_samples: int,
) -> tuple[str, bool]:
    """Return ``(markdown_report, ok)``; ``ok`` is False on any regression/mismatch."""
    labels = _labels_in(results_dir)
    a_labels = _match(labels, a_pattern)
    b_labels = _match(labels, b_pattern)
    if not a_labels or not b_labels:
        return (
            f"No labels matched (A `{a_pattern}` -> {a_labels or 'none'}, "
            f"B `{b_pattern}` -> {b_labels or 'none'}). "
            f"Available: {', '.join(labels) or 'none'}.\n",
            False,
        )

    a_times, a_mems, a_sizes = _pool(results_dir, a_labels)
    b_times, b_mems, b_sizes = _pool(results_dir, b_labels)

    time_rows, regressions = _timing_rows(
        a_times,
        b_times,
        a_mems,
        b_mems,
        time_threshold,
        model_threshold,
        mem_threshold,
        min_samples,
    )
    size_rows, mismatches = _size_rows(a_sizes, b_sizes)
    ok = not regressions and not mismatches

    lines = [
        "# A/B benchmark comparison",
        "",
        f"Baseline **A** = `{a_pattern}` ({', '.join(a_labels)})  ",
        f"Candidate **B** = `{b_pattern}` ({', '.join(b_labels)})",
        "",
        f"Time verdict: min-ratio B/A > {1 + time_threshold:.2f} is a regression "
        f"({1 + model_threshold:.2f} for < {min_samples} pooled samples). "
        f"Memory: median RSS ratio > {1 + mem_threshold:.2f} flagged (advisory).",
        "",
        tabulate(
            time_rows, headers=_TIME_HEADERS, tablefmt="github", disable_numparse=True
        ),
        "",
    ]

    only = set(a_times) ^ set(b_times)
    if only:
        lines += [
            "> One-sided operations (not compared): "
            + ", ".join(f"{c}/{o}" for c, o in sorted(only)),
            "",
        ]

    lines += [
        "## Operator-size equivalence (hard gate)",
        "",
        tabulate(
            size_rows, headers=_SIZE_HEADERS, tablefmt="github", disable_numparse=True
        ),
        "",
        "## Summary",
        "",
    ]
    if ok:
        lines += [
            "**PASS** — no time regression and identical term counts on every compared config.",
            "",
        ]
    else:
        if regressions:
            lines += ["**Time regressions:**", *[f"- {r}" for r in regressions], ""]
        if mismatches:
            lines += [
                "**Term-count mismatches (semantic change!):**",
                *[f"- {m}" for m in mismatches],
                "",
            ]
    return "\n".join(lines), ok


def main(argv: list[str] | None = None) -> int:
    """Parse args, print the comparison report, and exit non-zero on any regression."""
    parser = argparse.ArgumentParser(description="A/B benchmark regression check.")
    parser.add_argument(
        "results_dir", type=Path, help="dir holding time-*.json and <label>.json"
    )
    parser.add_argument(
        "--a", required=True, help="glob for baseline labels, e.g. 'A-*'"
    )
    parser.add_argument(
        "--b", required=True, help="glob for candidate labels, e.g. 'B-*'"
    )
    parser.add_argument(
        "--time-threshold",
        type=float,
        default=0.05,
        help="fractional slowdown that fails (default 0.05)",
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=0.10,
        help="looser threshold for low-sample configs (default 0.10)",
    )
    parser.add_argument(
        "--mem-threshold",
        type=float,
        default=0.10,
        help="fractional RSS growth flagged, advisory (default 0.10)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="pooled samples below which --model-threshold applies (default 5)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="also write the markdown report here",
    )
    args = parser.parse_args(argv)

    report, ok = compare(
        args.results_dir,
        args.a,
        args.b,
        args.time_threshold,
        args.model_threshold,
        args.mem_threshold,
        args.min_samples,
    )
    sys.stdout.write(report + "\n")
    if args.out is not None:
        args.out.write_text(report + "\n")
        sys.stdout.write(f"\nWrote {args.out}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
