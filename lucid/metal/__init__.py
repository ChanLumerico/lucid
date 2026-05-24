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
    """Return ``True`` — Apple Silicon always has a Metal GPU.

    Kept as a function (rather than a constant) for API parity with
    multi-device frameworks where availability is an actual runtime
    check.  On Lucid this is always true: the framework refuses to
    install on non-Apple-Silicon platforms, so by the time this call
    runs the Metal stack is guaranteed present.

    Returns
    -------
    bool
        Always ``True``.
    """
    return True


def synchronize() -> None:
    """Block until every pending Metal GPU operation has completed.

    MLX kernels dispatch asynchronously — calling ``synchronize`` makes
    the calling thread wait until the GPU command queue drains.  Use
    sparingly: it stalls the pipeline.  Typical places to call it are
    *before* timing measurements and *before* device→host transfers
    that read the result of recent kernels.

    See Also
    --------
    MetalStream.synchronize : sync a specific stream rather than the
        default device-wide stream.
    """
    _mx.synchronize()


def empty_cache() -> None:
    """Release unused cached Metal memory back to the OS.

    MLX caches recently-freed device buffers for reuse — this avoids
    repeated allocator round-trips in steady-state training but holds
    memory the system might want elsewhere (other processes, the
    desktop compositor).  Call this after a memory-hungry phase ends
    to give the OS back what you no longer need.

    Notes
    -----
    Cached buffers are otherwise freed automatically under memory
    pressure; manual eviction is a hint, not a requirement.
    """
    _mx.clear_cache()


def manual_seed(seed: int) -> None:
    """Set the Metal GPU random-number generator seed.

    Re-seeds the engine's default ``Generator`` so subsequent random
    ops (``rand``, ``randn``, dropout, weight init) produce a
    reproducible sequence.  Combine with :func:`lucid.manual_seed`
    when you need both CPU and GPU streams pinned.

    Parameters
    ----------
    seed : int
        Non-negative seed value.  Identical seeds produce identical
        sequences across runs on the same GPU.
    """
    _C_engine.default_generator().set_seed(seed)


def memory_allocated() -> int:
    """Return bytes currently allocated on the Metal GPU.

    Reflects live ``MTLBuffer`` storage owned by the engine — cached /
    pooled buffers held by MLX are *not* counted (see
    :func:`get_cache_memory`).

    Returns
    -------
    int
        Bytes currently allocated, excluding the MLX cache pool.
    """
    return int(_C_engine.memory_stats(_C_engine.Device.GPU).current_bytes)


def max_memory_allocated() -> int:
    """Return peak Metal GPU allocation observed since the last reset.

    The peak counter is updated on every allocation and is reset by
    :func:`reset_peak_memory_stats`.  Useful for sizing training jobs
    around the largest forward+backward footprint.

    Returns
    -------
    int
        Peak live-byte count since the last reset.
    """
    return int(_C_engine.memory_stats(_C_engine.Device.GPU).peak_bytes)


def reset_peak_memory_stats() -> None:
    """Reset the Metal-GPU peak-allocation counter to the current value.

    Call this at the start of a benchmark / profiling window so that
    :func:`max_memory_allocated` reports the peak observed during the
    window of interest rather than since process start.
    """
    _C_engine.reset_peak_memory_stats(_C_engine.Device.GPU)


def get_cache_memory() -> int:
    """Return bytes held in MLX's Metal allocator cache.

    These pages are reserved by the framework but not currently
    backing any live tensor — they sit in the allocator pool for fast
    reuse and are released back to the OS only on
    :func:`empty_cache` or under system memory pressure.

    Returns
    -------
    int
        Bytes in the MLX cache pool (not counted by
        :func:`memory_allocated`).
    """
    return int(_mx.metal.get_cache_memory())


def get_device_name() -> str:
    """Return the human-readable Metal GPU device name.

    Pulled from ``MTLDevice.name`` via MLX.  Examples: ``"Apple M1
    Max"``, ``"Apple M2 Ultra"``, ``"Apple M4 Pro"``.  Falls back to
    ``"Apple Silicon Metal GPU"`` if the platform layer refuses to
    answer.

    Returns
    -------
    str
        Marketing name of the GPU.
    """
    info = _mx.metal.device_info()
    return str(info.get("device_name", "Apple Silicon Metal GPU"))


class MetalStream:
    """Metal command stream context manager.

    On entry, all subsequent MLX operations are submitted to this stream.
    On exit, the stream is synchronized before control returns — every
    GPU command issued inside the ``with`` block is guaranteed complete
    when the block exits.

    Parameters
    ----------
    priority : int, optional
        Accepted for API parity with multi-stream frameworks but
        currently ignored — MLX exposes a single default stream per
        device on Apple Silicon and does not honor per-stream priority
        hints.  Default ``0``.

    Examples
    --------
    >>> import lucid
    >>> with lucid.metal.MetalStream():
    ...     y = model(x)             # all kernels submitted to this stream
    ...                              # stream sync happens on block exit
    """

    def __init__(self, priority: int = 0) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        self._priority = priority
        self._stream = _mx.default_stream(_mx.gpu)  # type: ignore[arg-type]

    def __enter__(self) -> MetalStream:
        """Enter the context.  Returns self so the value can be bound via ``with ... as``."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context, restoring any state that was modified on entry."""
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
        """Initialise the instance.  See the class docstring for parameter semantics."""
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

    def elapsed_time(self, end_event: MetalEvent) -> float:
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
    inputs: list[lucid.Tensor],
    output_shape: tuple[int, ...] | list[int],
    dtype: object = None,
    grid: tuple[int, int, int] = (1, 1, 1),
    threads: tuple[int, int, int] = (1, 1, 1),
) -> lucid.Tensor:
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
        dtype_str,  # type: ignore[arg-type]
        grid_arr,  # type: ignore[arg-type]
        threads_arr,  # type: ignore[arg-type]
    )
    return _wrap(out_impl)


def shared_tensor(
    shape: tuple[int, ...] | list[int],
    dtype: object = None,
    requires_grad: bool = False,
) -> lucid.Tensor:
    """Allocate a zero-filled tensor in Metal shared memory (no memcpy ever).

    The backing buffer is ``MTLResourceStorageModeShared`` — it is immediately
    readable and writable from CPU, and transferable to GPU via ``.to("metal")``
    with **zero memcpy** (the GPU reads the same physical pages).

    Parameters
    ----------
    shape:
        Desired shape of the tensor.
    dtype:
        Element dtype.  Defaults to ``lucid.float32``.
    requires_grad:
        Whether to track gradients.  Default is ``False``.

    Returns
    -------
    Tensor
        A ``device="cpu"`` tensor in shared Metal storage.

    Examples
    --------
    >>> buf = lucid.metal.shared_tensor((1024,))
    >>> buf.is_shared
    True
    >>> buf_gpu = buf.to("metal")   # zero-copy
    """
    import lucid as _lucid

    if dtype is None:
        dtype = _lucid.float32
    from lucid._dtype import to_engine_dtype as _to_eng

    impl = _C_engine.make_shared_tensor(list(shape), _to_eng(dtype), requires_grad)  # type: ignore[arg-type]
    return _wrap(impl)


def to_shared(tensor: lucid.Tensor) -> lucid.Tensor:
    """Promote a tensor to Metal shared memory (at most one memcpy).

    If *tensor* is already in shared storage this is a no-op.  After calling
    ``to_shared()``, both ``.to("metal")`` and ``.to("cpu")`` are zero-copy.

    Parameters
    ----------
    tensor:
        Source tensor on any device.

    Returns
    -------
    Tensor
        A tensor with ``is_shared == True`` on the same logical device as the
        input.

    Examples
    --------
    >>> x = lucid.randn(512, 512)
    >>> xs = lucid.metal.to_shared(x)   # one memcpy into shared buffer
    >>> xg = xs.to("metal")             # zero-copy
    >>> xs.is_shared
    True
    """
    impl = _unwrap(tensor)
    if impl.is_metal_shared:
        return tensor
    return _wrap(_C_engine.to_shared_storage(impl))


def is_shared(tensor: lucid.Tensor) -> bool:
    """Return ``True`` if *tensor* is backed by Metal shared memory.

    Parameters
    ----------
    tensor:
        Tensor to inspect.

    Returns
    -------
    bool
        ``True`` when the underlying storage is
        ``MTLResourceStorageModeShared``.

    Examples
    --------
    >>> lucid.metal.is_shared(lucid.randn(4))
    False
    >>> lucid.metal.is_shared(lucid.metal.to_shared(lucid.randn(4)))
    True
    """
    return _unwrap(tensor).is_metal_shared


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
    "shared_tensor",
    "to_shared",
    "is_shared",
]
