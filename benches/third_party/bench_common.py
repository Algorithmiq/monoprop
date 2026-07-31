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

"""Shared across the third-party benchmark scripts in this project (pauli_prop, majorana_prop)."""

from __future__ import annotations

import threading

import psutil


class RssPeakSampler:
    """Tracks this process's peak resident set size within a resettable window.

    The kernel-tracked high-water mark (``resource.getrusage().ru_maxrss``) is monotonic for the
    whole process lifetime and can never be reset, so it cannot isolate a single step's peak from
    the steps around it. This instead polls the *current* RSS from a background thread at a high
    fixed frequency and keeps the max seen since the last :meth:`reset`, so each step gets its own
    peak, independent of what earlier or later steps did.

    Caveat: this can only sample while the calling thread doesn't hold the GIL — a C extension
    call that never releases it (e.g. monoprop's ``propagate()``) blocks this thread out for its
    whole duration, so the sampler only catches whatever RSS growth is already visible by the time
    control returns to Python.
    """

    def __init__(self, interval_s: float = 1e-3) -> None:
        self._process = psutil.Process()
        self._interval_s = interval_s
        self._peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            rss = self._process.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss

    def __enter__(self) -> RssPeakSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop_event.set()
        self._thread.join()

    def reset(self) -> None:
        """Start a new measurement window, floored at the current RSS."""
        self._peak_bytes = self._process.memory_info().rss

    def peak_mb(self) -> float:
        return self._peak_bytes / 1024**2
