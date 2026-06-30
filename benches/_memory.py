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
proportional-set-size (PSS) readers, the resting-footprint reader, and the
background :class:`PssSampler` + :func:`merge_peak_of_sum` used to compute the
job's *peak-of-sum* physical memory.

Two things make the per-test peak honest under MPI:

- **PSS, not peak RSS.** PSS splits shared pages (libraries, and the shared-memory
  segments MPI uses for intra-node transport) across their sharers, so summing it
  across ranks counts them once. Peak RSS (``VmHWM``) counts those shared pages at
  full size in *every* rank, and PSS has no kernel high-water mark to correct it
  from -- a teardown-time correction misses the shared memory mapped at the peak and
  overestimates (the larger error in practice). Sampling PSS directly sidesteps it.
- **Peak-of-sum, not sum-of-peaks.** The per-test peak is ``max over time`` of the
  PSS summed across ranks. Summing each rank's independently-timed peak instead
  counts transients that never coexisted (worst when ranks peak at staggered times).
  Comparable wall-clock timestamps let :func:`merge_peak_of_sum` recover the true
  peak-of-sum after the fact.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import gc
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self

# PSS sampling cadence. monoprop's heavy work runs in C++/TBB with the GIL
# released, so a background sampler at this interval costs an otherwise-idle core
# rather than perturbing the timed main thread.
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

    PSS (shared pages split across their sharers) is the honest per-process share
    of physical RAM, so summing it across MPI ranks gives the job's true footprint
    -- unlike RSS, which counts shared library/code pages at full size in every rank.
    """
    return proc_field("/proc/self/smaps_rollup", "Pss:")


def malloc_trim() -> None:
    """Return free heap pages held by the C allocator to the OS (glibc only).

    ``malloc_trim`` is what makes a *resting* PSS reading meaningful: glibc keeps
    freed pages in its per-arena heaps, so without trimming the resident footprint
    still includes transient build buffers that are logically gone. A no-op (and
    silently ignored) on non-glibc libc.
    """
    with contextlib.suppress(Exception):  # non-glibc libc or no malloc_trim symbol
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.malloc_trim(ctypes.c_size_t(0))


def resting_pss_bytes() -> int:
    """Return current PSS after collecting garbage and trimming the C heap.

    Unlike the per-operation peak (a high-water mark reached mid-operation while
    transient build buffers are live), this is the *settled* footprint once those
    transients are released -- the metric that reveals persistent-memory wins (a
    smaller index, recomputed-vs-stored data) that the peak cannot see.
    """
    gc.collect()
    malloc_trim()
    return pss_bytes()


class PssSampler:
    """Background thread sampling this process's live PSS over time.

    Records ``(wall_clock, pss_bytes)`` pairs while active. Wall-clock time
    (``time.time``) is used rather than ``time.monotonic`` so timestamps are
    comparable across MPI ranks sharing a node's clock, which
    :func:`merge_peak_of_sum` needs to time-correlate the per-rank readings.

    Use as a context manager around the measured operation; a baseline sample is
    taken on entry and a final sample on exit, so even an operation shorter than
    one sampling interval yields a usable timeline.
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

    ``per_rank[i]`` is rank ``i``'s ``(wall_clock, pss_bytes)`` samples. Walks all
    samples in time order, step-holding each rank's most recent reading, and
    tracks the maximum of the running sum -- the job's true *peak-of-sum*: the
    largest summed footprint that actually coexisted. A serial run passes a
    single series and gets back its own peak.
    """
    # Seed each rank at its first (pre-op baseline) sample; ranks with no samples
    # contribute nothing.
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
