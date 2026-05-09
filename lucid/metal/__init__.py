"""
lucid.metal: Apple Metal GPU utilities.

Includes :func:`run_kernel` — the Metal Shader Escape Hatch (Phase 18) that
lets users execute arbitrary Metal Shading Language (MSL) compute shaders on
the GPU from Python with Lucid tensors as inputs/outputs.
"""

import time
from typing import TYPE_CHECKING

import mlx.core as _mx

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    import lucid


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

    def __exit__(self, *args: object) -> None:
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


# ── MetalKernelRunner (Phase 18) ─────────────────────────────────────────────

_DTYPE_TO_STR: dict[object, str] = {}


def _init_dtype_map() -> None:
    """Lazily populate the dtype→string map to avoid circular imports."""
    if _DTYPE_TO_STR:
        return
    from lucid._dtype import float16, float32, float64, int32, int64, bool_

    _DTYPE_TO_STR.update(
        {
            float16: "f16",
            float32: "f32",
            float64: "f64",
            int32: "i32",
            int64: "i64",
            bool_: "i32",  # bools are 32-bit ints in MSL
        }
    )


def run_kernel(
    source: str,
    function_name: str,
    inputs: "list[lucid.Tensor]",
    output_shape: tuple[int, ...] | list[int],
    dtype: object = None,
    grid: tuple[int, int, int] = (1, 1, 1),
    threads: tuple[int, int, int] = (1, 1, 1),
) -> "lucid.Tensor":
    """Run a custom Metal Shading Language (MSL) compute kernel.

    This is the **Metal Shader Escape Hatch** — it lets you write arbitrary
    GPU-accelerated code in MSL and call it from Python with Lucid tensors.

    Parameters
    ----------
    source : str
        Complete MSL source code (including ``#include <metal_stdlib>``).
        The kernel function should accept input buffers at indices 0..n-1
        and write its output to the buffer at index n.
    function_name : str
        Name of the ``kernel`` function inside *source* to dispatch.
    inputs : list[Tensor]
        Input tensors bound as read-only buffers in declaration order.
        Tensors on CPU are automatically copied to shared Metal memory;
        tensors already on Metal are bridged zero-copy when possible.
    output_shape : tuple[int, ...]
        Shape of the output tensor.
    dtype : lucid.dtype, optional
        Element dtype of the output.  Defaults to ``lucid.float32``.
    grid : tuple[int, int, int]
        Threadgroup grid dimensions ``(gx, gy, gz)``.
    threads : tuple[int, int, int]
        Threads-per-threadgroup ``(tx, ty, tz)``.

    Returns
    -------
    Tensor
        Output tensor on CPU (backed by shared Metal memory — zero-copy
        readable).  Call ``.to(device='metal')`` if you need an MLX tensor.

    Examples
    --------
    >>> MSL = '''
    ... #include <metal_stdlib>
    ... using namespace metal;
    ... kernel void relu(
    ...     device const float* x [[buffer(0)]],
    ...     device float* y       [[buffer(1)]],
    ...     uint gid [[thread_position_in_grid]])
    ... {
    ...     y[gid] = max(0.0f, x[gid]);
    ... }
    ... '''
    >>> x = lucid.tensor([-1.0, 2.0, -0.5, 3.0])
    >>> y = lucid.metal.run_kernel(MSL, 'relu', [x], (4,))
    >>> y   # tensor([0., 2., 0., 3.])
    """
    import lucid as _lucid

    _init_dtype_map()

    if dtype is None:
        dtype = _lucid.float32

    dtype_str: str = _DTYPE_TO_STR.get(dtype, "f32")

    impl_inputs = [_unwrap(t) for t in inputs]
    out_shape_list = list(output_shape)
    grid_arr = (int(grid[0]), int(grid[1]), int(grid[2]))
    threads_arr = (int(threads[0]), int(threads[1]), int(threads[2]))

    out_impl = _C_engine._run_metal_kernel(
        source,
        function_name,
        impl_inputs,
        out_shape_list,
        dtype_str,
        grid_arr,
        threads_arr,
    )
    return _wrap(out_impl)


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
    "run_kernel",
]
