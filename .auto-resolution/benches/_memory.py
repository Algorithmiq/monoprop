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

"""PSS-based memory-measurement primitives for the benchmark suite.

Import-only (no pytest, no MPI) so the logic stays unit-testable. Provides the
PSS readers, the resting-footprint reader, and the background :class:`PssSampler`
+ :func:`merge_peak_of_sum` used for the job's peak-of-sum physical memory.

Two choices make the per-test peak honest under MPI:

- **PSS, not peak RSS.** PSS splits shared pages (libraries, MPI's shared-memory
  transport segments) across their sharers, so summing across ranks counts them
  once; peak RSS counts them at full size in every rank and has no PSS high-water
  mark to correct from. Sampling PSS directly sidesteps that.
- **Peak-of-sum, not sum-of-peaks.** The peak is ``max over time`` of the summed
  PSS. Summing each rank's independently-timed peak counts transients that never
  coexisted. Comparable wall-clock timestamps let :func:`merge_peak_of_sum`
  recover the true peak-of-sum.
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

# PSS sampling cadence. monoprop's heavy work runs in C++/TBB with the GIL
# released, so the background sampler costs an idle core, not the timed thread.
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


def pss_bytes() -> int:
    """Return this process's current proportional set size (PSS) in bytes.

    PSS splits shared pages across their sharers, so summing it across ranks gives
    the job's true footprint -- unlike RSS, which counts them fully in every rank.
    """
    return proc_field("/proc/self/smaps_rollup", "Pss:")


def heap_trim() -> None:
    """Ask the C allocator to return unused heap pages to the OS.

    The allocator keeps freed pages in its per-arena heaps, so without trimming a
    resting reading still includes transient build buffers. Best-effort: modern
    allocators may decline, and the call is unsupported on some platforms.
    """
    with contextlib.suppress(Exception):  # unsupported platform / allocator
        psutil.heap_trim()


def resting_pss_bytes() -> int:
    """Return current PSS after collecting garbage and trimming the C heap.

    Unlike the per-operation peak (a mid-operation high-water mark), this is the
    settled footprint once transients are freed -- the persistent-memory metric
    the peak cannot see.
    """
    gc.collect()
    heap_trim()
    return pss_bytes()


class PssSampler:
    """Background thread sampling this process's live PSS over time.

    Records ``(wall_clock, pss_bytes)`` pairs while active. Uses ``time.time``
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
            self._samples.append((time.time(), pss_bytes()))
            self._stop.wait(self._interval)

    def __enter__(self) -> Self:
        self._samples.append((time.time(), pss_bytes()))  # baseline before the op
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
        self._samples.append((time.time(), pss_bytes()))  # final state after the op

    @property
    def samples(self) -> list[tuple[float, int]]:
        """Return the recorded ``(wall_clock, pss_bytes)`` samples."""
        return self._samples


def merge_peak_of_sum(per_rank: list[list[tuple[float, int]]]) -> int:
    """Return the peak of the summed live PSS across ranks, in bytes.

    ``per_rank[i]`` is rank ``i``'s samples. Walks all samples in time order,
    step-holding each rank's most recent reading, and tracks the maximum of the
    running sum -- the largest summed footprint that actually coexisted. A serial
    run passes one series and gets back its own peak.
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
