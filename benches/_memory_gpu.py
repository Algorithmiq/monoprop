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

"""Device-memory measurement, the GPU counterpart of ``benches/_memory_cpu.py``.

The host metric works because the kernel maintains ``VmHWM`` on every RSS increase: a
peak, not a sample, so no transient can hide between polls. This module looks for the
same guarantee on the device, and reports which one it found rather than pretending the
answer is uniform.

Which strategy applies depends on the allocator CuPy is using, so
:class:`DeviceHighWaterMark` records it in :attr:`~DeviceHighWaterMark.method`:

``async-pool``
    CuPy on ``malloc_async``. CUDA's stream-ordered pool keeps
    ``cudaMemPoolAttrUsedMemHigh``, a true high-water mark of bytes in use, and writing 0
    to it resets it to the current value -- a resettable window, exactly like
    ``clear_refs`` on the host. This is the only *exact* strategy.

``caching-pool``
    CuPy's default allocator. Its pool has no high-water counter, but ``total_bytes()``
    (bytes reserved from the driver) is monotone: freeing an array returns the block to
    the pool's free list without returning it to CUDA. So it never falls back down over a
    transient, which is what a sampler could not manage. It is an *upper bound* on peak
    bytes in use -- it carries the pool's fragmentation and cached slack with it -- and it
    is cumulative over the process rather than per-window, since nothing resets it.

``unavailable``
    No CuPy, or the driver refused the query. Everything reads zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self

# `method` values; see the module docstring.
ASYNC_POOL = "async-pool"
CACHING_POOL = "caching-pool"
UNAVAILABLE = "unavailable"

# How each strategy's number should be described wherever it is published.
DEVICE_MEMORY_METRICS = {
    ASYNC_POOL: "peak GPU bytes in use over the step (cudaMemPoolAttrUsedMemHigh)",
    CACHING_POOL: "GPU pool reservation high-water since process start "
    "(CuPy MemoryPool.total_bytes; upper bound on bytes in use)",
    UNAVAILABLE: "no GPU memory reading available",
}


def _async_pool() -> int | None:
    """Return the device's default CUDA memory pool, or None if it is not in use.

    The pool exists on any CUDA >= 11.2 device, but it only *observes* the allocations
    under test when CuPy routes them through it, so an allocator check comes first --
    otherwise the counters read a pool nobody is allocating from and report zero.
    """
    try:
        import cupy as cp
        from cupy.cuda import runtime
    except ImportError:
        return None
    try:
        if cp.cuda.get_allocator() is not cp.cuda.malloc_async:
            return None
        return runtime.deviceGetDefaultMemPool(runtime.getDevice())
    except (AttributeError, RuntimeError):  # old CuPy, or no device
        return None


def _pool_attr(pool: int, attr_name: str) -> int:
    """Read one ``cudaMemPoolAttr`` counter, or 0 if unavailable."""
    try:
        from cupy.cuda import runtime

        return int(runtime.memPoolGetAttribute(pool, getattr(runtime, attr_name)))
    except (AttributeError, RuntimeError):
        return 0


def reset_peak_device_bytes(pool: int) -> bool:
    """Reset the pool's used-memory high-water mark to its current occupancy.

    CUDA defines writing 0 to ``cudaMemPoolAttrUsedMemHigh`` as "set it to
    ``cudaMemPoolAttrUsedMemCurrent``", which is what opens a fresh window.
    """
    try:
        from cupy.cuda import runtime

        runtime.memPoolSetAttribute(pool, runtime.cudaMemPoolAttrUsedMemHigh, 0)
    except (AttributeError, RuntimeError):  # pragma: no cover - driver dependent
        return False
    return True


def device_synchronize() -> None:
    """Block until the device is idle, so a reading covers completed work.

    Launches are asynchronous: without this, a counter can be read before the allocations
    it is meant to cover have happened.
    """
    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
    except (ImportError, AttributeError, RuntimeError):
        pass


def _caching_pool_reserved_bytes() -> int:
    """Return bytes CuPy's default pool holds from the driver, 0 if unavailable."""
    try:
        import cupy as cp

        return int(cp.get_default_memory_pool().total_bytes())
    except (ImportError, AttributeError, RuntimeError):
        return 0


def _caching_pool_used_bytes() -> int:
    """Return bytes live in CuPy's default pool right now, 0 if unavailable."""
    try:
        import cupy as cp

        return int(cp.get_default_memory_pool().used_bytes())
    except (ImportError, AttributeError, RuntimeError):
        return 0


class DeviceHighWaterMark:
    """Peak device memory over the enclosed block.

    Mirrors :class:`_memory_cpu.HighWaterMark`: synchronizes on entry and exit so the reading
    covers completed work, and exposes the same ``peak``/``baseline``/``delta`` vocabulary.

    ``exact`` is ``True`` only under the ``async-pool`` strategy, where the driver keeps a
    real high-water mark. Otherwise the figure is an upper bound (``caching-pool``) or
    absent (``unavailable``), and :attr:`method` says which -- check it before publishing,
    because the three are not the same quantity.

    Entry is deliberately cheap under both strategies: a counter reset, or nothing at all.
    Releasing cached blocks would give the caching pool a per-window floor, but it calls
    ``cudaFree``, which synchronizes the device and would land in the caller's timing.
    """

    def __init__(self) -> None:
        self._pool = _async_pool()
        self.method = UNAVAILABLE
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.exact = False

    def __enter__(self) -> Self:
        device_synchronize()
        if self._pool is not None and reset_peak_device_bytes(self._pool):
            self.exact = True
            self.method = ASYNC_POOL
            self.baseline_bytes = _pool_attr(
                self._pool, "cudaMemPoolAttrUsedMemCurrent"
            )
        else:
            # A pool we cannot reset is a pool we cannot window; fall back rather than
            # report an all-time high-water mark as if it belonged to this step.
            self._pool = None
            self.baseline_bytes = _caching_pool_reserved_bytes()
            self.method = CACHING_POOL if self.baseline_bytes else UNAVAILABLE
        self.peak_bytes = self.baseline_bytes
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        device_synchronize()
        if self.method == ASYNC_POOL and self._pool is not None:
            observed = _pool_attr(self._pool, "cudaMemPoolAttrUsedMemHigh")
        elif self.method == CACHING_POOL:
            observed = _caching_pool_reserved_bytes()
        else:
            observed = _caching_pool_used_bytes()
            self.method = CACHING_POOL if observed else UNAVAILABLE
        self.peak_bytes = max(self.baseline_bytes, observed)

    @property
    def metric(self) -> str:
        """Return the description of what :attr:`peak_bytes` counts."""
        return DEVICE_MEMORY_METRICS[self.method]

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

    @property
    def delta_mb(self) -> float:
        """Return :attr:`delta_bytes` in MiB."""
        return self.delta_bytes / 1024**2


def _self_check() -> int:
    """Allocate a transient that only a high-water mark can see, and report.

    The transient is freed before the window closes, so an end-of-step occupancy reading
    misses it entirely -- that gap is the bug this module exists to close. Run this on the
    GPU host to find out which strategy is active and whether it catches the transient.
    """
    try:
        import cupy as cp
    except ImportError:
        print("cupy is not installed: nothing to check")
        return 1

    transient_mb, resident_mb = 512, 8
    keep = None
    with DeviceHighWaterMark() as window:
        blob = cp.zeros(transient_mb * 1024**2 // 8, dtype=cp.float64)
        blob += 1  # touch it, so the allocation is real
        del blob
        keep = cp.zeros(resident_mb * 1024**2 // 8, dtype=cp.float64)
    end_of_step_mb = _caching_pool_used_bytes() / 1024**2
    del keep

    print(f"strategy      : {window.method} (exact={window.exact})")
    print(f"metric        : {window.metric}")
    print(f"baseline      : {window.baseline_mb:8.1f} MB")
    print(f"peak          : {window.peak_mb:8.1f} MB")
    print(f"delta         : {window.delta_mb:8.1f} MB  (expected >= {transient_mb})")
    print(
        f"end-of-step   : {end_of_step_mb:8.1f} MB  (the old metric; misses the peak)"
    )

    if window.method == UNAVAILABLE:
        print("\nFAIL: no device reading available")
        return 1
    if window.delta_mb < transient_mb * 0.9:
        print(f"\nFAIL: the {transient_mb} MB transient was not captured")
        return 1
    print("\nPASS: the freed transient is visible in the peak")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
