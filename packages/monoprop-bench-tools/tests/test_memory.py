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

"""Unit tests for the host-memory primitives.

The behaviour pinned throughout is that :class:`HighWaterMark` sees transients that are
gone before the window closes, and that each window starts from its own floor rather than
inheriting an earlier block's spike.
"""

from __future__ import annotations

import gc
import resource

import pytest
from monoprop_bench_tools.memory.cpu import (
    HighWaterMark,
    heap_trim,
    peak_rss_bytes,
    reset_peak_rss,
    rss_bytes,
)

MIB = 2**20


def test_high_water_mark_catches_a_freed_transient() -> None:
    if not reset_peak_rss():
        pytest.skip("/proc/self/clear_refs unavailable (non-Linux or kernel < 4.0)")
    with HighWaterMark() as window:
        blob = bytearray(80 * MIB)
        for i in range(0, len(blob), 4096):
            blob[i] = 1
        del blob  # gone before the window closes: only VmHWM still knows it existed

    assert window.exact
    assert window.delta_bytes >= 70 * MIB
    assert window.peak_bytes >= window.baseline_bytes


def test_high_water_mark_window_is_reset_per_block() -> None:
    if not reset_peak_rss():
        pytest.skip("/proc/self/clear_refs unavailable (non-Linux or kernel < 4.0)")
    with HighWaterMark() as first:
        blob = bytearray(80 * MIB)
        for i in range(0, len(blob), 4096):
            blob[i] = 1
        del blob
    with HighWaterMark() as second:
        pass

    # The second window must not inherit the first's spike
    assert first.delta_bytes >= 70 * MIB
    assert second.delta_bytes < 10 * MIB


def test_peak_rss_never_below_current_rss() -> None:
    if peak_rss_bytes() == 0:
        pytest.skip("/proc/self/status VmHWM unavailable (non-Linux)")
    assert peak_rss_bytes() >= rss_bytes()


def test_reset_also_clears_ru_maxrss() -> None:
    """``ru_maxrss`` shares ``mm->hiwater_rss`` with ``VmHWM``, so a window reset drops it.

    Benchmarks that record a whole-run ceiling alongside per-step windows have to take the
    maximum over the windows instead; this pins the behaviour that forces that.
    """
    if not reset_peak_rss():
        pytest.skip("/proc/self/clear_refs unavailable (non-Linux or kernel < 4.0)")

    blob = bytearray(80 * MIB)
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    del blob
    gc.collect()
    heap_trim()
    reset_peak_rss()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    assert before >= 70 * MIB
    assert after < before
