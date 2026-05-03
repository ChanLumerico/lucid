"""
lucid.profiler: performance profiling utilities.

Wraps the C++ engine's Profiler, OpEvent, and MemoryStats classes.

Usage:
    with lucid.profiler.profile() as prof:
        loss = model(x)
        loss.backward()

    for event in prof.events():
        print(event.name, event.time_ns)
"""

from contextlib import contextmanager
from typing import Any, Iterator, TYPE_CHECKING
from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    pass


class OpEvent:
    """A single recorded operation event."""

    def __init__(self, impl: Any) -> None:
        self._impl = impl

    @property
    def name(self) -> str:
        """Operation name (e.g. 'matmul', 'add')."""
        return str(self._impl.name)

    @property
    def time_ns(self) -> int:
        """Wall-clock duration of the operation in nanoseconds."""
        return int(self._impl.time_ns)

    @property
    def time_us(self) -> float:
        """Wall-clock duration in microseconds."""
        return self.time_ns / 1_000.0

    @property
    def time_ms(self) -> float:
        """Wall-clock duration in milliseconds."""
        return self.time_ns / 1_000_000.0

    @property
    def shape(self) -> list[Any]:
        """Output tensor shape."""
        return list(self._impl.shape)

    @property
    def flops(self) -> int:
        """Estimated FLOPs for this operation."""
        return int(self._impl.flops)

    @property
    def memory_delta_bytes(self) -> int:
        """Net memory allocated by this operation in bytes."""
        return int(self._impl.memory_delta_bytes)

    def __repr__(self) -> str:
        return (
            f"OpEvent(name={self.name!r}, time_us={self.time_us:.3f}, "
            f"shape={self.shape})"
        )


class ProfileSummary:
    """Aggregated summary of a named operation across multiple calls."""

    def __init__(self, name: str, events: list[OpEvent]) -> None:
        self.name = name
        self._events = events

    @property
    def count(self) -> int:
        """Number of times this operation was called."""
        return len(self._events)

    @property
    def total_time_us(self) -> float:
        """Total time in microseconds across all calls."""
        return sum(e.time_us for e in self._events)

    @property
    def avg_time_us(self) -> float:
        """Average time in microseconds per call."""
        return self.total_time_us / max(1, self.count)

    @property
    def total_flops(self) -> int:
        """Total FLOPs across all calls."""
        return sum(e.flops for e in self._events)

    def __repr__(self) -> str:
        return (
            f"ProfileSummary(name={self.name!r}, count={self.count}, "
            f"avg_us={self.avg_time_us:.3f}, total_flops={self.total_flops})"
        )


class MemoryStats:
    """Memory usage snapshot."""

    def __init__(self, impl: Any) -> None:
        self._impl = impl

    @property
    def current_bytes(self) -> int:
        """Currently allocated bytes."""
        return int(self._impl.current_bytes)

    @property
    def peak_bytes(self) -> int:
        """Peak allocated bytes since last reset."""
        return int(self._impl.peak_bytes)

    @property
    def alloc_count(self) -> int:
        """Number of allocations."""
        return int(self._impl.alloc_count)

    @property
    def free_count(self) -> int:
        """Number of deallocations."""
        return int(self._impl.free_count)

    def __repr__(self) -> str:
        return (
            f"MemoryStats(current={self.current_bytes / 1024:.1f}KB, "
            f"peak={self.peak_bytes / 1024:.1f}KB)"
        )


class Profiler:
    """Profile operation execution times and memory usage.

    Use as a context manager or call start()/stop() manually.

    Example:
        with lucid.profiler.profile() as prof:
            y = model(x)
        for ev in prof.events():
            print(ev)
    """

    def __init__(self, with_memory: bool = True) -> None:
        self._impl = _C_engine.Profiler()
        self._with_memory = with_memory
        self._active = False

    def start(self) -> None:
        """Start recording operations."""
        _C_engine.set_current_profiler(self._impl)
        self._impl.start()
        self._active = True

    def stop(self) -> None:
        """Stop recording operations."""
        self._impl.stop()
        _C_engine.set_current_profiler(None)
        self._active = False

    def __enter__(self) -> Profiler:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def events(self) -> list[OpEvent]:
        """Return all recorded OpEvents."""
        return [OpEvent(e) for e in self._impl.events]

    def key_averages(self) -> list[ProfileSummary]:
        """Return per-operation-name summaries sorted by total time."""
        from collections import defaultdict

        by_name: dict[str, list[OpEvent]] = defaultdict(list)
        for ev in self.events():
            by_name[ev.name].append(ev)
        summaries = [ProfileSummary(name, evs) for name, evs in by_name.items()]
        return sorted(summaries, key=lambda s: -s.total_time_us)

    def memory_stats(self) -> MemoryStats | None:
        """Return memory stats if available."""
        try:
            impl = _C_engine.memory_stats(_C_engine.Device.CPU)
            return MemoryStats(impl)
        except Exception:
            return None

    def export_chrome_trace(self, path: str) -> None:
        """Export events as a Chrome trace JSON file.

        Open with chrome://tracing or https://ui.perfetto.dev.
        """
        import json

        trace_events = []
        for i, ev in enumerate(self.events()):
            trace_events.append(
                {
                    "name": ev.name,
                    "ph": "X",  # complete event
                    "ts": i * 100,  # synthetic timestamp (µs)
                    "dur": ev.time_us,
                    "pid": 0,
                    "tid": 0,
                    "args": {"shape": str(ev.shape), "flops": ev.flops},
                }
            )
        with open(path, "w") as f:
            json.dump({"traceEvents": trace_events}, f)

    def clear(self) -> None:
        """Clear all recorded events."""
        self._impl.clear()


@contextmanager
def profile(
    activities: Any = None,
    with_memory: bool = True,
) -> Iterator[Profiler]:
    """Context manager for profiling a code block.

    Args:
        activities:   Ignored (for API compatibility).
        with_memory:  Include memory statistics.

    Yields:
        Profiler instance with recorded events.

    Example:
        with lucid.profiler.profile() as prof:
            output = model(input)
        print(prof.key_averages())
    """
    prof = Profiler(with_memory=with_memory)
    with prof:
        yield prof


@contextmanager
def record_function(name: str) -> Iterator[None]:
    """Mark a code block as a named event in the profiler.

    If no profiler is active, this is a no-op.

    Args:
        name: Label for this code block in the trace.

    Example:
        with lucid.profiler.record_function('my_forward'):
            y = model(x)
    """
    # If a profiler is active, we can't easily inject named regions without
    # engine support. This is a best-effort no-op that preserves the API.
    yield


__all__ = [
    "Profiler",
    "OpEvent",
    "ProfileSummary",
    "MemoryStats",
    "profile",
    "record_function",
]
