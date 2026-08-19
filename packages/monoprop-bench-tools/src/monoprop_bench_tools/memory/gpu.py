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

"""Device-memory measurement, the GPU counterpart of :mod:`monoprop_bench_tools.memory.cpu`.

The host metric works because the kernel maintains ``VmHWM`` on every RSS increase: a
peak, thus accounting for transients as well. This module looks for the
same guarantee on the device, reporting which one it found.

Which strategy applies depends on the allocator CuPy is using, so
:class:`DeviceHighWaterMark` records it in :attr:`~DeviceHighWaterMark.method`:

``async-pool``
    CuPy on ``malloc_async``. CUDA's stream-ordered pool keeps
    ``cudaMemPoolAttrUsedMemHigh``, a true high-water mark of bytes in use, and writing 0
    to it resets it to the current value -- a resettable window, exactly like
    ``clear_refs`` on the host. Exact, and it needs no Python-side bookkeeping at all.

``hook``
    CuPy's default (caching) allocator, tracked via a :class:`cupy.cuda.MemoryHook`.
    The hook's callbacks fire synchronously, in-process, on every malloc/free CuPy makes
    -- not on a timer, so nothing can happen between polls the way it could with a
    background sampler. Each callback takes ``max(peak, pool.used_bytes())``, which is
    exact for the same reason ``VmHWM`` is: it is updated at the moment of the event, not
    read back later.

``unavailable``
    No CuPy, or the hook API is missing (very old CuPy) and the async pool is also absent.
    Everything reads zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Self

# `method` values; see the module docstring.
ASYNC_POOL = "async-pool"
HOOK = "hook"
UNAVAILABLE = "unavailable"

# How each strategy's number should be described wherever it is published.
DEVICE_MEMORY_METRICS = {
    ASYNC_POOL: "peak GPU bytes in use over the step (cudaMemPoolAttrUsedMemHigh)",
    HOOK: "peak GPU bytes in use over the step (CuPy MemoryHook on the default pool)",
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


def _caching_pool_used_bytes() -> int:
    """Return bytes live in CuPy's default pool right now, 0 if unavailable."""
    try:
        import cupy as cp

        return int(cp.get_default_memory_pool().used_bytes())
    except (ImportError, AttributeError, RuntimeError):
        return 0


_hook_class: type[Any] | None = None


def _peak_hook_class() -> type[Any] | None:
    """Build (once) and return a ``MemoryHook`` subclass that tracks a running peak.

    Built lazily and cached at module level: the base class only exists once CuPy is
    importable, so it cannot be defined at import time. The callbacks accept ``**kwargs``
    rather than CuPy's documented parameter names, so a signature change in some CuPy
    version does not silently stop the tracking -- it would raise loudly instead of the
    hook simply never firing.
    """
    if _peak_hook_class.cached is not None:
        return _peak_hook_class.cached
    try:
        from cupy.cuda import memory_hook
    except ImportError:
        return None

    class _PeakUsedBytesHook(memory_hook.MemoryHook):
        name = "monoprop_peak_used_bytes_hook"

        def __init__(self, pool: Any) -> None:
            self._pool = pool
            self.peak_bytes = int(pool.used_bytes())

        def _update(self, **_kwargs: object) -> None:
            self.peak_bytes = max(self.peak_bytes, int(self._pool.used_bytes()))

        def malloc_postprocess(self, **kwargs: object) -> None:
            self._update(**kwargs)

        def free_postprocess(self, **kwargs: object) -> None:
            self._update(**kwargs)

    _peak_hook_class.cached = _PeakUsedBytesHook
    return _PeakUsedBytesHook


_peak_hook_class.cached = None


class DeviceHighWaterMark:
    """Peak device memory over the enclosed block.

    Mirrors :class:`monoprop_bench_tools.memory.cpu.HighWaterMark`: synchronizes on entry and exit so the reading
    covers completed work, and exposes the same ``peak``/``baseline``/``delta`` vocabulary.

    ``exact`` is ``True`` under both the ``async-pool`` and ``hook`` strategies: the first
    reads a driver-maintained high-water counter, the second updates its own peak
    synchronously on every allocation event CuPy makes, so neither can miss a transient
    that lived and died inside the block. Only ``unavailable`` degrades the figure (to
    zero); :attr:`method` says which applies -- check it before publishing.
    """

    def __init__(self) -> None:
        """Prepare a window. The strategy is only chosen on entry, not here."""
        self._pool = _async_pool()
        self._hook: Any = None
        self.method = UNAVAILABLE
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.exact = False

    def __enter__(self) -> Self:
        """Synchronize, pick the strongest available strategy, and take the baseline."""
        device_synchronize()
        if self._pool is not None and reset_peak_device_bytes(self._pool):
            self.exact = True
            self.method = ASYNC_POOL
            self.baseline_bytes = _pool_attr(
                self._pool, "cudaMemPoolAttrUsedMemCurrent"
            )
            self.peak_bytes = self.baseline_bytes
            return self

        # Not on the async pool (or too old a CuPy to reset it): fall back to tracking
        # the default pool's `used_bytes()` synchronously, via a hook, rather than the
        # driver-native counter.
        self._pool = None
        hook_cls = _peak_hook_class()
        if hook_cls is None:
            self.baseline_bytes = _caching_pool_used_bytes()
            self.peak_bytes = self.baseline_bytes
            return self

        try:
            import cupy as cp

            pool = cp.get_default_memory_pool()
            self._hook = hook_cls(pool)
            self._hook.__enter__()
        except (ImportError, AttributeError, RuntimeError):
            self._hook = None
            self.baseline_bytes = _caching_pool_used_bytes()
            self.peak_bytes = self.baseline_bytes
            return self

        self.exact = True
        self.method = HOOK
        self.baseline_bytes = self._hook.peak_bytes
        self.peak_bytes = self.baseline_bytes
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Synchronize and read the peak back for the chosen strategy. Exceptions propagate."""
        device_synchronize()
        if self.method == ASYNC_POOL and self._pool is not None:
            observed = _pool_attr(self._pool, "cudaMemPoolAttrUsedMemHigh")
        elif self.method == HOOK and self._hook is not None:
            self._hook.__exit__(None, None, None)
            observed = self._hook.peak_bytes
        else:
            # Nothing tracked the block: this is the same single end-of-block sample that
            # proved unreliable for the caching pool. Report it, but the method stays
            # `unavailable` so callers do not mistake it for an exact reading.
            observed = _caching_pool_used_bytes()
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
