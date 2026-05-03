"""
lucid.metal: Apple Metal GPU utilities.
"""

import time
from typing import Any

import mlx.core as _mx

from lucid._C import engine as _C_engine


def is_available() -> bool:
    """Return True; Apple Silicon always has Metal GPU."""
    return True


def synchronize() -> None:
    """Wait for all pending Metal GPU operations to complete."""
    _mx.synchronize()


def empty_cache() -> None:
    """Release unused cached Metal memory back to the system."""
    _mx.clear_cache()


def manual_seed(seed: int) -> None:
    """Set the Metal GPU random number generator seed."""
    _C_engine.default_generator().set_seed(seed)


def memory_allocated() -> int:
    """Return bytes currently allocated on Metal GPU."""
    return int(_C_engine.memory_stats(_C_engine.Device.GPU).current_bytes)


def max_memory_allocated() -> int:
    """Return peak bytes allocated on Metal GPU since last reset."""
    return int(_C_engine.memory_stats(_C_engine.Device.GPU).peak_bytes)


def reset_peak_memory_stats() -> None:
    """Reset the peak memory counter for Metal GPU."""
    _C_engine.reset_peak_memory_stats(_C_engine.Device.GPU)


def get_cache_memory() -> int:
    """Return bytes held in the Metal memory cache (not yet freed to OS)."""
    return int(_mx.metal.get_cache_memory())


def get_device_name() -> str:
    """Return the name of the Metal GPU device."""
    info = _mx.metal.device_info()
    return str(info.get("device_name", "Apple Silicon Metal GPU"))


class MetalStream:
    """Metal command stream context manager.

    On entry, all subsequent MLX operations are submitted to this stream.
    On exit, the stream is synchronized before control returns.

    Args:
        priority: Ignored (MLX uses a single default stream per device).
                  Kept for API compatibility with multi-stream frameworks.
    """

    def __init__(self, priority: int = 0) -> None:
        self._priority = priority
        self._stream = _mx.default_stream(_mx.gpu)

    def __enter__(self) -> "MetalStream":
        return self

    def __exit__(self, *args: Any) -> None:
        self.synchronize()

    def synchronize(self) -> None:
        """Wait for all commands submitted to this stream to complete."""
        _mx.synchronize(self._stream)


class MetalEvent:
    """Metal GPU timing event.

    Records a point in time on the GPU timeline.  When ``enable_timing=True``,
    :meth:`elapsed_time` returns wall-clock milliseconds between two events
    (a coarse approximation — MLX does not expose GPU-side timestamps).

    Example::

        start = MetalEvent(enable_timing=True)
        end   = MetalEvent(enable_timing=True)
        start.record()
        model(x)
        end.record()
        end.synchronize()
        print(start.elapsed_time(end), "ms")
    """

    def __init__(self, enable_timing: bool = False) -> None:
        self._enable_timing = enable_timing
        self._t: float | None = None

    def record(self, stream: MetalStream | None = None) -> None:
        """Mark this event on the current stream (and snapshot wall clock)."""
        if stream is not None:
            _mx.synchronize(stream._stream)
        else:
            _mx.synchronize()
        if self._enable_timing:
            self._t = time.perf_counter()

    def synchronize(self) -> None:
        """Block until all GPU work preceding this event has completed."""
        _mx.synchronize()

    def elapsed_time(self, end_event: "MetalEvent") -> float:
        """Return wall-clock milliseconds between this event and *end_event*.

        Both events must have been recorded with ``enable_timing=True``.
        Returns 0.0 if timing was not enabled on either event.
        """
        if self._t is None or end_event._t is None:
            return 0.0
        return (end_event._t - self._t) * 1e3


__all__ = [
    "is_available",
    "synchronize",
    "empty_cache",
    "manual_seed",
    "memory_allocated",
    "max_memory_allocated",
    "reset_peak_memory_stats",
    "get_cache_memory",
    "get_device_name",
    "MetalStream",
    "MetalEvent",
]
