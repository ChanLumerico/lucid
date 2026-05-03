"""
lucid.metal: Apple Metal GPU utilities.
"""

from typing import Any
from lucid._C import engine as _C_engine


def is_available() -> bool:
    """Return True; Apple Silicon always has Metal GPU."""
    return True


def synchronize() -> None:
    """Wait for all pending Metal operations to complete."""
    # MLX evaluates lazily; calling data_as_python() on any tensor forces eval.
    # No global sync API in the current engine.
    pass


def empty_cache() -> None:
    """Release cached Metal memory (MLX manages this automatically)."""
    pass


def manual_seed(seed: int) -> None:
    """Set the Metal GPU random number generator seed."""
    _C_engine.default_generator().set_seed(seed)


def memory_allocated() -> int:
    """Return bytes currently allocated on Metal GPU."""
    return int(_C_engine.memory_stats(_C_engine.Device.GPU).current_bytes)


def max_memory_allocated() -> int:
    """Return peak bytes allocated on Metal GPU."""
    return int(_C_engine.memory_stats(_C_engine.Device.GPU).peak_bytes)


def reset_peak_memory_stats() -> None:
    """Reset the peak memory counter for Metal GPU."""
    _C_engine.reset_peak_memory_stats(_C_engine.Device.GPU)


def get_device_name() -> str:
    """Return the name of the Metal GPU device."""
    return "Apple Silicon Metal GPU"


class MetalStream:
    """
    Metal command stream abstraction.

    MLX manages streams internally; this is a no-op wrapper for API compatibility.
    """

    def __init__(self, priority: int = 0) -> None:
        pass

    def __enter__(self) -> "MetalStream":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def synchronize(self) -> None:
        """Wait for this stream's commands to complete."""
        pass


class MetalEvent:
    """Metal GPU timing event."""

    def __init__(self, enable_timing: bool = False) -> None:
        self._enable_timing = enable_timing

    def record(self, stream: "MetalStream | None" = None) -> None:
        """Record the current time on this event."""
        pass

    def synchronize(self) -> None:
        """Wait for this event to complete."""
        pass

    def elapsed_time(self, end_event: "MetalEvent") -> float:
        """Return elapsed time in milliseconds between events."""
        return 0.0


__all__ = [
    "is_available", "synchronize", "empty_cache", "manual_seed",
    "memory_allocated", "max_memory_allocated", "reset_peak_memory_stats",
    "get_device_name", "MetalStream", "MetalEvent",
]
