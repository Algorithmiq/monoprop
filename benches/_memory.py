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

"""RSS-based memory-measurement primitives for the benchmark suite.

What the per-test peak means: **peak-of-sum, not sum-of-peaks.** The peak is
``max over time`` of the summed RSS. Summing each rank's independently-timed peak
counts transients that never coexisted. Comparable wall-clock timestamps let
:func:`merge_peak_of_sum` recover the true peak-of-sum.
"""

from __future__ import annotations

import contextlib
import gc
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self

# RSS sampling cadence. monoprop's heavy work runs in C++ with the GIL released, so
# the background sampler costs an idle core, not the timed thread.
SAMPLE_INTERVAL_S = 0.005


def proc_field(path: str, key: str) -> int:
    """Return a ``/proc/self`` size field (kB → bytes); 0 if unavailable."""
    try:
        text = Path(path).read_text()
    except OSError:  # pragma: no cover - non-Linux or restricted /proc
        return 0
    for line in text.splitlines():
        if line.startswith(key):
            return int(line.split()[1]) * 1024  # values are in kB
    return 0


def rss_bytes() -> int:
    """Return this process's current resident set size (RSS) in bytes.

    Read from ``/proc/self/status`` (``VmRSS``), which is a cheap single-line lookup
    """
    return proc_field("/proc/self/status", "VmRSS:")


def heap_trim() -> None:
    """Ask the C allocator to return unused heap pages to the OS."""
    with contextlib.suppress(Exception):  # unsupported platform / allocator
        psutil.heap_trim()


def resting_rss_bytes() -> int:
    """Return current RSS after collecting garbage and trimming the C heap."""
    gc.collect()
    heap_trim()
    return rss_bytes()


class RssSampler:
    """Background thread sampling this process's live RSS over time.

    Records ``(wall_clock, rss_bytes)`` pairs while active. Uses ``time.time``
    (not ``time.monotonic``) so timestamps are comparable across ranks sharing a
    node's clock, which :func:`merge_peak_of_sum` needs to correlate readings.

    Use as a context manager around the operation; it samples on entry and exit,
    so even a sub-interval operation yields a usable timeline.
    """

    def __init__(self, interval: float = SAMPLE_INTERVAL_S) -> None:
        self._interval = interval
        self._samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append((time.time(), rss_bytes()))
            self._stop.wait(self._interval)

    def __enter__(self) -> Self:
        self._samples.append((time.time(), rss_bytes()))
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()
        self._samples.append((time.time(), rss_bytes()))

    @property
    def samples(self) -> list[tuple[float, int]]:
        """Return the recorded ``(wall_clock, rss_bytes)`` samples."""
        return self._samples


def merge_peak_of_sum(per_rank: list[list[tuple[float, int]]]) -> int:
    """Return the peak of the summed live RSS across ranks, in bytes.

    ``per_rank[i]`` is rank ``i``'s samples. Walks all samples in time order,
    step-holding each rank's most recent reading, and tracks the maximum of the
    running sum -- the largest summed footprint that actually coexisted.
    """
    # Seed each rank at its pre-op baseline sample; empty series contribute 0.
    current = [series[0][1] if series else 0 for series in per_rank]
    running = sum(current)
    peak = running
    events = sorted(
        (t, r, pss) for r, series in enumerate(per_rank) for t, pss in series
    )
    for _t, rank, pss in events:
        running += pss - current[rank]
        current[rank] = pss
        peak = max(peak, running)
    return peak
