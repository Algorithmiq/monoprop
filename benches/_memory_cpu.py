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

"""Memory-measurement primitives for the benchmark suite.

Two metrics, deliberately kept apart:

:class:`HighWaterMark`
    The kernel's own peak RSS over a resettable window. Exact -- there is no sampling, so
    no transient can be missed -- and it needs no background thread, so the GIL cannot
    hide anything from it. This is the metric to quote, and the only one that is
    like-for-like against a non-Python library.

:class:`PssSampler`
    A sampled ``(wall_clock, pss)`` timeline. Needed only under MPI, where the job's
    footprint is the **peak-of-sum, not the sum-of-peaks**: summing each rank's
    independently-timed peak counts transients that never coexisted, so
    :func:`merge_peak_of_sum` replays the timelines instead. A scalar per-rank peak
    cannot reconstruct that, which is the one thing sampling still buys.
"""

from __future__ import annotations

import contextlib
import gc
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self

# Sampling cadence for the MPI timeline.
SAMPLE_INTERVAL_S = 0.005

# Reset VmHWM to the current RSS, starting a new measurement window
_CLEAR_REFS_MM_HIWATER_RSS = "5\n"


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


def pss_bytes() -> int:
    """Return this process's proportional set size (PSS) in bytes.

    PSS divides each shared page among the processes mapping it, so PSS summed over the
    ranks on a node counts every page exactly once. RSS charges a shared page (the Python
    interpreter, libstdc++, a shared graph) in full to every rank, which inflates an
    MPI sum by roughly the shared footprint times the rank count.
    """
    return proc_field("/proc/self/smaps_rollup", "Pss:")


def peak_rss_bytes() -> int:
    """Return the kernel's high-water mark of this process's RSS, in bytes.

    ``VmHWM`` is maintained by the kernel on every RSS increase, so it is exact.
    """
    return proc_field("/proc/self/status", "VmHWM:")


def reset_peak_rss() -> bool:
    """Reset ``VmHWM`` to the current RSS, starting a new measurement window.

    This also resets ``getrusage(...).ru_maxrss`` (and so Julia's ``Sys.maxrss()``): both
    report the same kernel field, ``mm->hiwater_rss``. Any process that opens a window
    therefore loses ``ru_maxrss`` as a whole-run ceiling, and must take the maximum over
    its windows instead.

    Returns:
        ``True`` if the reset took effect, ``False`` where ``/proc/self/clear_refs`` is
        unavailable (non-Linux, kernel < 4.0, or a restricted sandbox), in which case
        ``VmHWM`` keeps counting from process start and callers must fall back.
    """
    try:
        Path("/proc/self/clear_refs").write_text(_CLEAR_REFS_MM_HIWATER_RSS)
    except OSError:  # pragma: no cover - platform dependent
        return False
    return True


def heap_trim() -> None:
    """Ask the C allocator to return unused heap pages to the OS."""
    with contextlib.suppress(Exception):  # unsupported platform / allocator
        psutil.heap_trim()


def resting_rss_bytes() -> int:
    """Return current RSS after collecting garbage and trimming the C heap."""
    gc.collect()
    heap_trim()
    return rss_bytes()


def _parse_cpu_list(spec: str) -> set[int]:
    """Expand a kernel CPU list (``0-3,8``) into a set of CPU numbers."""
    cpus: set[int] = set()
    for part in spec.split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def pinned_thread_summary() -> dict[str, int | list[int]]:
    """Return how many of this process's threads are bound to a single CPU.

    Reads ``Cpus_allowed_list`` for every thread under ``/proc/self/task``. A thread the
    engine has placed sees exactly one CPU; an unplaced one sees the whole rank mask.

    This is deliberately independent of the engine: the alternative signal, the ``pinned=``
    field of a ``COMMPROF`` line, only exists on builds that have ``monoprop_COMM_PROFILE``,
    so it cannot be compared across a version boundary that added it. Without a
    build-agnostic probe, "the other build is much faster" and "placement silently failed
    on one arm, so the run is void" are the same observation.

    Returns:
        ``threads``, ``single_cpu_threads``, ``distinct_pinned_cpus`` (how many different
        CPUs those threads occupy -- equal to ``single_cpu_threads`` iff no two threads
        landed on the same core), ``affinity_cpus`` (the process mask's width), and
        ``pinned_cpus`` (sorted list of CPU ids that single-pinned threads occupy). All
        numeric values zero and ``pinned_cpus`` empty where ``/proc`` is unavailable.
    """
    try:
        tasks = list(Path("/proc/self/task").iterdir())
    except OSError:  # pragma: no cover - non-Linux or restricted /proc
        return {
            "threads": 0,
            "single_cpu_threads": 0,
            "distinct_pinned_cpus": 0,
            "affinity_cpus": 0,
            "pinned_cpus": [],
        }

    threads = 0
    pinned: set[int] = set()
    single = 0
    for task in tasks:
        try:
            text = (task / "status").read_text()
        except OSError:  # thread exited between listing and reading
            continue
        threads += 1
        for line in text.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                cpus = _parse_cpu_list(line.split(":", 1)[1].strip())
                if len(cpus) == 1:
                    single += 1
                    pinned |= cpus
                break
    return {
        "threads": threads,
        "single_cpu_threads": single,
        "distinct_pinned_cpus": len(pinned),
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "pinned_cpus": sorted(pinned),
    }


class HighWaterMark:
    """Exact peak RSS over the enclosed block, straight from the kernel.

    Settles the process on entry (``gc.collect()`` + ``malloc_trim``) and resets ``VmHWM``
    to that floor, so the peak reported is this block's own and not an earlier block's
    retained garbage. The settling is what makes the number comparable across languages:
    without it the figure tracks the allocator's or GC's willingness to return pages more
    than it tracks what the code needed.

    Use ``peak_bytes`` for the footprint (peak including everything already resident) and
    ``delta_bytes`` for what this block added on top of its floor. Report the floor too --
    an interpreter plus its imports is a 100+ MiB constant that has nothing to do with the
    code under test.

    ``exact`` is ``False`` when the kernel would not reset the window (see
    :func:`reset_peak_rss`); the peak then degrades to the RSS observed on exit, which is
    a lower bound. Callers that publish numbers should check it.
    """

    def __init__(self, *, settle: bool = True) -> None:
        self._settle = settle
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.exact = False

    def __enter__(self) -> Self:
        self.baseline_bytes = resting_rss_bytes() if self._settle else rss_bytes()
        self.exact = reset_peak_rss()
        self.peak_bytes = self.baseline_bytes
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        observed = peak_rss_bytes() if self.exact else rss_bytes()
        self.peak_bytes = max(self.baseline_bytes, observed)

    def start(self) -> Self:
        """Open the window explicitly (same as ``__enter__``).

        Exists because a ``pytest-benchmark`` pedantic run cannot be wrapped in a ``with``:
        the window has to open inside the benchmark's ``setup`` and close after
        ``pedantic()`` returns.
        """
        return self.__enter__()

    def stop(self) -> None:
        """Close the window explicitly (same as ``__exit__``)."""
        self.__exit__(None, None, None)

    @property
    def delta_bytes(self) -> int:
        """Return the peak measured above the block's own starting floor."""
        return self.peak_bytes - self.baseline_bytes

    @property
    def peak_mb(self) -> float:
        """Return :attr:`peak_bytes` in MiB."""
        return self.peak_bytes / 1024**2

    @property
    def baseline_mb(self) -> float:
        """Return :attr:`baseline_bytes` in MiB."""
        return self.baseline_bytes / 1024**2


class PssSampler:
    """Background thread sampling this process's live PSS over time.

    Records ``(wall_clock, pss_bytes)`` pairs while active. Uses ``time.time`` (not
    ``time.monotonic``) so timestamps are comparable across ranks sharing a node's clock,
    which :func:`merge_peak_of_sum` needs to correlate readings.

    Use as a context manager around the operation; it samples on entry and exit, so even a
    sub-interval operation yields a usable timeline.

    The timeline is only as dense as the timed thread's GIL releases allow, so treat it as
    a lower bound on the true peak-of-sum and :class:`HighWaterMark` as the exact per-rank
    figure. It is kept because no per-rank scalar can recover which peaks coexisted.
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
        self._samples.append((time.time(), pss_bytes()))
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
        self._samples.append((time.time(), pss_bytes()))

    @property
    def samples(self) -> list[tuple[float, int]]:
        """Return the recorded ``(wall_clock, pss_bytes)`` samples."""
        return self._samples


def merge_peak_of_sum(per_rank: list[list[tuple[float, int]]]) -> int:
    """Return the peak of the summed live PSS across ranks, in bytes.

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
