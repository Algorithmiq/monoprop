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

One metric, :class:`HighWaterMark`: the kernel's own peak RSS over a resettable window.
Exact -- there is no sampling, so no transient can be missed -- and it needs no background
thread, so the GIL cannot hide anything from it. This is the metric to quote, and the only
one that is like-for-like against a non-Python library.

Under MPI the ranks' peaks are summed, which is a sum-of-peaks: it counts transients that
never coexisted, and RSS charges each shared page in full to every rank mapping it. Both
err high, so the figure is an upper bound on the node's true footprint -- fine for
tracking regressions, but not a number to quote as a provisioning requirement.
"""

from __future__ import annotations

import contextlib
import gc
import os
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self

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


def peak_rss_bytes() -> int:
    """Return the kernel's high-water mark of this process's RSS, in bytes.

    ``VmHWM`` is maintained by the kernel on every RSS increase, so it is exact.
    """
    return proc_field("/proc/self/status", "VmHWM:")


# Every HighWaterMark between __enter__ and __exit__. mm->hiwater_rss is per-process, not
# per-window, so an inner window's reset would erase an outer window's peak: the reset folds
# the mark into these first. Weak so a window abandoned without stop() is still collected.
_OPEN_WINDOWS: weakref.WeakSet[HighWaterMark] = weakref.WeakSet()


def _fold_open_windows() -> None:
    """Carry the current mark into every open window, before a reset discards it."""
    observed = peak_rss_bytes()
    for window in list(_OPEN_WINDOWS):
        window.peak_bytes = max(window.peak_bytes, observed)


def reset_peak_rss() -> bool:
    """Reset ``VmHWM`` to the current RSS, starting a new measurement window.

    This also resets ``getrusage(...).ru_maxrss`` (and so Julia's ``Sys.maxrss()``): both
    report the same kernel field, ``mm->hiwater_rss``. Any process that opens a window
    therefore loses ``ru_maxrss`` as a whole-run ceiling, and must take the maximum over
    its windows instead.

    The field is per-process, so this would also erase the peak of any :class:`HighWaterMark`
    already open. Their marks are folded forward first, which is what makes windows nest --
    an inner window measuring one operation cannot hide the transient an outer window spans.

    Returns:
        ``True`` if the reset took effect, ``False`` where ``/proc/self/clear_refs`` is
        unavailable (non-Linux, kernel < 4.0, or a restricted sandbox), in which case
        ``VmHWM`` keeps counting from process start and callers must fall back.
    """
    _fold_open_windows()
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
    """Count this rank's own threads bound to a single CPU; all-zero means ``/proc`` was unreadable, not unpinned."""
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

    Windows nest: an inner one resets the same per-process kernel field, so it folds the
    mark into every enclosing window first. An outer window therefore still reports the
    peak of the whole block, including transients that fell before an inner window opened.
    """

    def __init__(self, *, settle: bool = True) -> None:
        """Prepare a window.

        Args:
            settle: Whether to settle the process (``gc.collect()`` + ``malloc_trim``)
                before taking the baseline. Pass ``False`` only when the caller has
                already settled and the extra pause would perturb the measurement.
        """
        self._settle = settle
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.exact = False

    def __enter__(self) -> Self:
        """Take the baseline and reset the kernel's peak-RSS window to it."""
        self.baseline_bytes = resting_rss_bytes() if self._settle else rss_bytes()
        # Before registering: this window's own reset must not fold a mark into itself.
        self.exact = reset_peak_rss()
        self.peak_bytes = self.baseline_bytes
        _OPEN_WINDOWS.add(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Read the peak back, never below the baseline. Exceptions propagate."""
        _OPEN_WINDOWS.discard(self)
        observed = peak_rss_bytes() if self.exact else rss_bytes()
        # peak_bytes already carries what any inner window's reset folded in, and starts at
        # the baseline, so the kernel's remaining mark can only raise it.
        self.peak_bytes = max(self.peak_bytes, observed)

    def start(self) -> Self:
        """Open the window explicitly; a ``pedantic`` run cannot be wrapped in a ``with``."""
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
