"""
Conversion utilities: Python objects / NumPy arrays -> TensorImpl.

This is one of the H4 numpy bridge sites — the only place where a
``np.ndarray`` is allowed to enter Lucid.  NumPy is imported lazily
(inside the functions that need it) so ``import lucid`` works
without numpy installed.

3.0.2: pure-Python inputs (``list``, ``tuple``, scalar) now take a
numpy-free fast path through ``struct.pack`` + ``TensorImpl.from_bytes``.
``lucid.tensor([1, 2, 3])`` no longer pulls numpy into the dependency
graph — only ``lucid.tensor(np_array)`` and explicit ``from_numpy``
keep the numpy bridge.
"""

import struct
from typing import TYPE_CHECKING, Sequence, SupportsFloat, SupportsInt, cast

from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    import numpy as np
    from lucid._tensor.tensor import Tensor


# struct format code + element size for each engine dtype that has a
# direct CPython struct representation.  BF16 and C64 are absent — they
# need numpy / explicit conversion, so the fast path falls through to
# the existing numpy bridge when those are the target dtype.
_DTYPE_STRUCT: dict[_C_engine.Dtype, tuple[str, int]] = {
    _C_engine.Dtype.F16: ("e", 2),
    _C_engine.Dtype.F32: ("f", 4),
    _C_engine.Dtype.F64: ("d", 8),
    _C_engine.Dtype.I8: ("b", 1),
    _C_engine.Dtype.I16: ("h", 2),
    _C_engine.Dtype.I32: ("i", 4),
    _C_engine.Dtype.I64: ("q", 8),
    _C_engine.Dtype.Bool: ("?", 1),
}


# Cached default-device engine enum for the ndarray fast path.
# Default device is stable per process (set once by user or defaults to CPU);
# resolving it on every ``lucid.tensor(np_array)`` call cost ~50 ns × 120 k
# calls = 6 ms / epoch — small but eliminable.
_CACHED_DEFAULT_DEVICE_ENUM: _C_engine.Device | None = None


def _default_device_enum_cached() -> _C_engine.Device:
    """Return the resolved default-device engine enum from a per-process cache.

    Invalidation: cleared by :func:`lucid._globals.set_default_device` —
    user explicitly switching the global default is rare.  Code that
    passes an explicit ``device=`` kwarg bypasses this helper and goes
    through ``_parse_device`` as before.
    """
    global _CACHED_DEFAULT_DEVICE_ENUM
    if _CACHED_DEFAULT_DEVICE_ENUM is None:
        from lucid._dispatch import _parse_device
        from lucid._globals import get_default_device

        _CACHED_DEFAULT_DEVICE_ENUM = _parse_device(get_default_device())
    return _CACHED_DEFAULT_DEVICE_ENUM


def _invalidate_default_device_cache() -> None:
    """Called by ``lucid._globals.set_default_device`` after a change."""
    global _CACHED_DEFAULT_DEVICE_ENUM
    _CACHED_DEFAULT_DEVICE_ENUM = None


def _flatten_with_shape(data: object) -> tuple[list[int], list[object]] | None:
    """Walk a (possibly nested) list/tuple, returning ``(shape, flat)``.

    Returns ``None`` when the structure is ragged (different sub-lengths
    at the same level), signalling the caller to fall back to numpy.
    """
    if not isinstance(data, (list, tuple)):
        # Scalar — 0-d tensor.
        return [], [data]
    shape: list[int] = [len(data)]
    if len(data) == 0:
        return shape, []
    if isinstance(data[0], (list, tuple)):
        # Nested — recurse on first element to pick up the sub-shape,
        # then validate every sibling against it.
        first = _flatten_with_shape(data[0])
        if first is None:
            return None
        sub_shape, _ = first
        flat: list[object] = []
        for item in data:
            sub = _flatten_with_shape(item)
            if sub is None or sub[0] != sub_shape:
                return None
            flat.extend(sub[1])
        return shape + sub_shape, flat
    # Leaf row — must be uniform scalars.
    for item in data:
        if isinstance(item, (list, tuple)):
            return None
    return shape, list(data)


def _infer_engine_dtype(flat: Sequence[object]) -> _C_engine.Dtype:
    """Default-dtype inference for Python scalars.

    Matches numpy's behaviour at the call site:
      * any ``complex`` element → ``C64``  (numpy promotes to complex128
        by default; lucid pins to complex64 because the engine only
        carries C64 today)
      * any ``float`` element → ``F32`` (lucid's default float dtype)
      * all ``bool`` elements → ``Bool``
      * otherwise (ints) → ``I64``  (numpy uses platform int, but
        lucid + reference frameworks both pin int → int64 for tensor
        literals to avoid 32-bit-vs-64-bit footguns).
    """
    if not flat:
        return _C_engine.Dtype.F32  # zero-length tensor — match numpy default
    # complex must be checked first — Python's numeric hierarchy means
    # `complex` is neither `float` nor `int`, but the dtype promotion
    # rules say a single complex value upgrades everything else.
    if any(isinstance(v, complex) for v in flat):
        return _C_engine.Dtype.C64
    has_float = any(isinstance(v, float) and not isinstance(v, bool) for v in flat)
    if has_float:
        return _C_engine.Dtype.F32
    # bool is a subclass of int in Python, check it first.
    if all(isinstance(v, bool) for v in flat):
        return _C_engine.Dtype.Bool
    return _C_engine.Dtype.I64


def _coerce_for_struct(v: object, dtype: _C_engine.Dtype) -> object:
    """Cast a Python scalar so ``struct.pack`` accepts it for ``dtype``."""
    if dtype == _C_engine.Dtype.Bool:
        return bool(v)
    if dtype in (
        _C_engine.Dtype.I8,
        _C_engine.Dtype.I16,
        _C_engine.Dtype.I32,
        _C_engine.Dtype.I64,
    ):
        return int(cast(SupportsInt, v))
    # F16 / F32 / F64
    return float(cast(SupportsFloat, v))


def _pack_complex64(flat: Sequence[object]) -> bytes:
    """Pack a flat sequence of complex/real scalars as little-endian
    interleaved (real, imag) float32 pairs — the on-disk layout the
    engine's C64 dtype expects (matches ``TensorImpl::tolist()``'s
    decoder in TensorImpl.cpp).
    """
    if not flat:
        return b""
    floats: list[float] = []
    for v in flat:
        if isinstance(v, complex):
            c = v
        else:
            c = complex(cast(SupportsFloat, v))
        floats.append(c.real)
        floats.append(c.imag)
    return struct.pack(f"={len(floats)}f", *floats)


def _try_numpy_free_to_impl(
    data: object,
    dtype_eng: _C_engine.Dtype | None,
    device_eng: _C_engine.Device,
    requires_grad: bool,
) -> _C_engine.TensorImpl | None:
    """Build a TensorImpl from Python scalars/lists/tuples without numpy.

    Returns ``None`` when the input can't be handled by ``struct.pack``
    (e.g. ragged nesting, BF16 target dtype); the caller then falls
    through to the numpy bridge.  All currently supported engine dtypes
    have a numpy-free path here — including C64 (packed as interleaved
    f32 pairs).
    """
    if isinstance(data, (list, tuple)) or isinstance(data, (int, float, bool, complex)):
        unpacked = _flatten_with_shape(data)
        if unpacked is None:
            return None  # ragged → numpy
        shape, flat = unpacked
        target = dtype_eng if dtype_eng is not None else _infer_engine_dtype(flat)

        # C64 isn't directly encodable as a single struct format code —
        # it ships as two f32 values per element (real, imag).  Handle
        # it explicitly so users can write ``lucid.tensor([1+2j, ...])``
        # without dragging numpy in just for complex literal support.
        if target == _C_engine.Dtype.C64:
            packed = _pack_complex64(flat)
            return _C_engine.TensorImpl.from_bytes(
                packed, shape, target, device_eng, requires_grad
            )

        fmt_entry = _DTYPE_STRUCT.get(target)
        if fmt_entry is None:
            return None  # BF16 (not yet in the enum) → numpy if it ever lands
        fmt, _ = fmt_entry
        n = len(flat)
        if n == 0:
            packed = b""
        else:
            packed = struct.pack(
                f"={n}{fmt}", *(_coerce_for_struct(v, target) for v in flat)
            )
        return _C_engine.TensorImpl.from_bytes(
            packed, shape, target, device_eng, requires_grad
        )
    return None


_NP_TO_ENGINE_DTYPE: dict[str, _C_engine.Dtype] = {
    "float16": _C_engine.Dtype.F16,
    "float32": _C_engine.Dtype.F32,
    "float64": _C_engine.Dtype.F64,
    "int8": _C_engine.Dtype.I8,
    "int16": _C_engine.Dtype.I16,
    "int32": _C_engine.Dtype.I32,
    "int64": _C_engine.Dtype.I64,
    "bool": _C_engine.Dtype.Bool,
    "complex64": _C_engine.Dtype.C64,
    "complex128": _C_engine.Dtype.C128,
}


def _np_dtype_to_engine(np_dtype: np.dtype) -> _C_engine.Dtype:
    name = np_dtype.name
    if name in _NP_TO_ENGINE_DTYPE:
        return _NP_TO_ENGINE_DTYPE[name]
    if name.startswith("float"):
        return _C_engine.Dtype.F64
    return _C_engine.Dtype.F32


def _is_ndarray(obj: object) -> bool:
    """Check ndarray-ness without importing numpy when it isn't loaded."""
    cls = type(obj)
    return cls.__module__ == "numpy" and cls.__name__ == "ndarray"


def _require_numpy(operation: str) -> object:
    """Lazy-import numpy with a clean error message when it isn't installed.

    Lucid runs without numpy by default; the bridge methods listed in H4
    (``tensor(np_array)``, ``Tensor.numpy()``, ``from_numpy``,
    ``from_dlpack`` / ``to_dlpack``) opt into numpy as the canonical
    interop library.  When the user reaches one of these without having
    installed numpy, raise an ``ImportError`` that points them at the
    correct extra rather than the generic ``ModuleNotFoundError``.
    """
    try:
        import numpy as np  # noqa: PLC0415 — bridge import
    except ImportError as e:
        raise ImportError(
            f"{operation} requires numpy, but numpy is not installed.\n"
            "Install it explicitly:\n"
            "    pip install lucid[numpy]\n"
            "or\n"
            "    pip install numpy"
        ) from e
    return np


def _to_impl(
    data: object,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> _C_engine.TensorImpl:
    """Convert list/scalar/ndarray/Tensor -> TensorImpl."""
    from lucid._tensor.tensor import Tensor as _Tensor

    # 3.1.1: hottest path — bare ``lucid.tensor(np_array)`` with no dtype
    # override, default device, no grad.  Skips ``normalize_factory_kwargs``,
    # ``_try_numpy_free_to_impl``'s isinstance gauntlet, the dead-code
    # ``_np_dtype_to_engine`` lookup, and ``np.ascontiguousarray`` (gated
    # on the C_CONTIGUOUS flag).  Profile of LeNet-5/MNIST training showed
    # ``lucid.tensor`` called 120k times per epoch via the DataLoader
    # per-sample tensorisation pattern — this fast path turns a ~9 µs
    # per-call hot loop into ~1 µs.
    if (
        dtype is None
        and device is None
        and not requires_grad
        and type(data).__module__ == "numpy"
        and type(data).__name__ == "ndarray"
    ):
        if data.flags["C_CONTIGUOUS"]:  # type: ignore[attr-defined]
            return _C_engine.TensorImpl(data, _default_device_enum_cached(), False)
        # Non-contiguous → still hot, just one more numpy call.
        np = _require_numpy("lucid.tensor() ndarray fast path")
        return _C_engine.TensorImpl(
            np.ascontiguousarray(data),  # type: ignore[attr-defined]
            _default_device_enum_cached(),
            False,
        )

    _dtype_eng, _device_eng, _rg = normalize_factory_kwargs(
        dtype, device, requires_grad
    )

    if isinstance(data, _Tensor):
        impl = data._impl
        if impl.requires_grad != _rg:
            from lucid._dispatch import _impl_with_grad

            impl = _impl_with_grad(impl, _rg)
        return impl

    if isinstance(data, _C_engine.TensorImpl):
        if data.requires_grad != _rg:
            from lucid._dispatch import _impl_with_grad

            data = _impl_with_grad(data, _rg)
        return data

    # 3.0.2: numpy-free fast path for pure-Python scalars / lists / tuples.
    # Uses ``struct.pack`` + ``TensorImpl.from_bytes`` so the most common
    # ``lucid.tensor([1, 2, 3])`` pattern doesn't transitively import
    # numpy.  Returns None for inputs the fast path can't handle (ragged
    # lists, BF16 / C64 dtype targets, ndarray) — caller falls through.
    fast = _try_numpy_free_to_impl(
        data, _dtype_eng if dtype is not None else None, _device_eng, _rg
    )
    if fast is not None:
        return fast

    # Numpy is the sanctioned conversion library for the remaining inputs
    # (ndarray, BF16/C64 targets, ragged sequences).  Imported lazily so
    # ``import lucid`` doesn't need numpy installed.  When numpy is
    # missing, ``_require_numpy`` raises a guidance-rich ImportError.
    np = _require_numpy("lucid.tensor() with non-Tensor input")

    numpy_input = isinstance(data, np.ndarray)  # type: ignore[attr-defined]

    if not numpy_input:
        # Python list/scalar -> convert to numpy with default dtype (float32)
        tmp = np.array(data)  # type: ignore[attr-defined]
        if dtype is None:
            target_eng = _dtype_eng
        else:
            target_eng = _dtype_eng
        arr = tmp.astype(_engine_dtype_to_np(target_eng), copy=False)
        _dtype_eng = target_eng
    else:
        if dtype is not None:
            arr = data.astype(_engine_dtype_to_np(_dtype_eng), copy=False)  # type: ignore[attr-defined]
        else:
            arr = data
            _dtype_eng = _np_dtype_to_engine(arr.dtype)  # type: ignore[attr-defined]

    arr = np.ascontiguousarray(arr)  # type: ignore[attr-defined]
    with np.errstate(invalid="ignore", over="ignore"):  # type: ignore[attr-defined]
        impl = _C_engine.TensorImpl(arr, _device_eng, _rg)
    # bfloat16 has no NumPy counterpart, so the array above is float32 and
    # the engine narrows afterwards.  Doing it this way round rather than
    # picking the nearest NumPy dtype matters: float16 is *not* a
    # substitute — a bfloat16 value can exceed float16's maximum, and
    # routing through it would turn a representable number into infinity
    # before the engine ever saw it.
    if _dtype_eng == _C_engine.Dtype.BF16 and impl.dtype != _C_engine.Dtype.BF16:
        impl = _C_engine.astype(impl, _C_engine.Dtype.BF16)
    return impl


def _engine_dtype_to_np(d: _C_engine.Dtype) -> str:
    _MAP: dict[_C_engine.Dtype, str] = {
        # bfloat16 is widened, not narrowed — see the note in
        # ``_to_impl``: float16 cannot hold every bfloat16 value.
        _C_engine.Dtype.BF16: "float32",
        _C_engine.Dtype.F16: "float16",
        _C_engine.Dtype.F32: "float32",
        _C_engine.Dtype.F64: "float64",
        _C_engine.Dtype.I8: "int8",
        _C_engine.Dtype.I16: "int16",
        _C_engine.Dtype.I32: "int32",
        _C_engine.Dtype.I64: "int64",
        _C_engine.Dtype.Bool: "bool",
        _C_engine.Dtype.C64: "complex64",
        _C_engine.Dtype.C128: "complex128",
    }
    return _MAP.get(d, "float32")


def _copy_tensor(
    src: Tensor,
    *,
    dtype: DTypeLike,
    device: DeviceLike,
    requires_grad: bool,
) -> _C_engine.TensorImpl:
    """A fresh copy of ``src`` for :func:`tensor`, as a leaf of its own.

    ``_to_impl`` hands a Tensor's impl back as it is, which is right for
    ``Tensor(t)`` and ``Parameter(t)`` — they wrap the data they are given
    — and wrong here: ``tensor`` is documented to copy, and passing the
    impl through made ``tensor(t, dtype=float64)`` a float32 alias of
    ``t``'s storage, with ``dtype`` and ``device`` silently dropped.

    ``None`` keeps the source's own dtype and device rather than falling
    back to the global defaults, as the reference framework does.
    """
    dt, dev, rg = normalize_factory_kwargs(
        dtype if dtype is not None else src.dtype,
        device if device is not None else src.device,
        requires_grad,
    )
    # Starting from a detached view means none of the steps below is
    # recorded, so the copy does not hang off ``src``'s graph.
    # ``contiguous`` always allocates, which is what makes this a copy even
    # when neither the dtype nor the device changes.
    impl = _C_engine.contiguous(src.detach()._impl)
    if impl.dtype != dt:
        impl = _C_engine.astype(impl, dt)
    if impl.device != dev:
        impl = impl.transfer_to_device(dev, False)
    return impl.clone_with_grad(True) if rg else impl


def tensor(
    data: object,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Construct a new :class:`Tensor` from Python data, a NumPy array, or another Tensor.

    Always allocates a fresh storage and **copies** the source bytes into
    Lucid-owned memory.  This is the canonical entry point for creating
    tensors from heterogeneous Python inputs: scalars (``int`` / ``float`` /
    ``bool``), nested lists, NumPy ``ndarray``\s, and existing Lucid
    ``Tensor``\s.  Dtype is inferred from the source unless ``dtype`` is
    given; device defaults to the global default (typically ``"cpu"``) unless
    overridden.

    Parameters
    ----------
    data : object
        Source data.  Accepted forms:

        * Python scalar (``int``, ``float``, ``bool``) — produces a 0-d
          tensor.
        * Nested ``list`` / ``tuple`` — recursively converted; element type
          must be uniform.
        * ``numpy.ndarray`` — bridge boundary
          (see :mod:`lucid._factories.converters`); the data is copied
          regardless of the source array's contiguity.
        * Existing :class:`Tensor` — copied to a new buffer and detached
          from its autograd graph, so the result is a leaf of its own (use
          :func:`as_tensor` to avoid the copy when dtype/device match).
    dtype : dtype | str | None, optional
        Target element type.  ``None`` (default) infers from ``data``:
        integers → ``int64``, floats → ``float32``, complex → ``complex64``;
        a Tensor keeps its own dtype.
    device : device | str | None, optional
        Target device (``"cpu"`` or ``"metal"``).  ``None`` uses
        :func:`lucid.get_default_device`, except that a Tensor stays on its
        own device.
    requires_grad : bool, optional
        Whether the resulting tensor should record autograd operations.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        A freshly-allocated Lucid tensor.

    Notes
    -----
    This factory is one of the six "bridge" entry points in the **H4** rule
    — the only places where external libraries (NumPy here) may legitimately
    cross into Lucid's compute path.  Outside the bridges, Lucid composites
    must use engine primitives directly.

    For zero-copy conversion when the source is already an ``ndarray`` on
    CPU and shares dtype, prefer :func:`as_tensor`.

    Examples
    --------
    >>> import lucid
    >>> lucid.tensor([1.0, 2.0, 3.0])
    tensor([1., 2., 3.])
    >>> lucid.tensor([[1, 2], [3, 4]], dtype=lucid.float32)
    tensor([[1., 2.],
            [3., 4.]])
    >>> import numpy as np
    >>> lucid.tensor(np.arange(6).reshape(2, 3))
    tensor([[0, 1, 2],
            [3, 4, 5]], dtype=lucid.int64)
    >>> src = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> copy = lucid.tensor(src, dtype=lucid.float64)
    >>> copy.dtype, copy.is_leaf, copy.requires_grad
    (lucid.float64, True, False)
    """
    from lucid._tensor.tensor import Tensor

    if isinstance(data, Tensor):
        return Tensor.__new_from_impl__(
            _copy_tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
        )
    return Tensor.__new_from_impl__(
        _to_impl(data, dtype=dtype, device=device, requires_grad=requires_grad)
    )


def as_tensor(
    data: object,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Convert data to a tensor, avoiding a copy when the source already matches.

    Unlike :func:`tensor`, ``as_tensor`` is "best-effort no-copy":

    * If ``data`` is already a :class:`Tensor` with the requested ``dtype``
      and ``device``, it is returned unchanged.
    * If ``data`` is a :class:`Tensor` whose dtype or device differs, it is
      converted as :meth:`Tensor.to` converts it, and the result stays
      connected to ``data``'s autograd graph — unlike :func:`tensor`, which
      copies into a detached leaf.
    * Anything else — Python data, NumPy arrays — goes through
      :func:`tensor`, which copies.

    Parameters
    ----------
    data : object
        Source data — Python scalar / list, NumPy array, or Tensor.
    dtype : dtype | str | None, optional
        Target element type.  ``None`` preserves the source dtype.
    device : device | str | None, optional
        Target device.  When the source already lives on a different device,
        a copy across the device boundary is performed.

    Returns
    -------
    Tensor
        ``data`` itself, a converted copy of it, or a freshly-constructed
        Lucid tensor.

    Notes
    -----
    ``as_tensor`` is the right choice in code that may be handed either a
    Tensor or raw data (e.g. a collate function): a Tensor that already has
    the requested dtype and device passes through with no allocation at
    all.  For semantic clarity in library code that should never share
    storage, use :func:`tensor`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> lucid.as_tensor(x) is x              # already a Tensor, returned as-is
    True
    >>> lucid.as_tensor(x, dtype=lucid.float64).dtype
    lucid.float64
    >>> w = lucid.tensor([1.0, 2.0], requires_grad=True)
    >>> lucid.as_tensor(w, dtype=lucid.float64).sum().backward()
    >>> w.grad                               # the conversion stays in the graph
    tensor([1., 1.])
    """
    from lucid._tensor.tensor import Tensor

    if isinstance(data, Tensor):
        # Nothing to convert: hand back the object itself, so identity,
        # storage and the autograd graph all carry over unchanged.
        _dt, _dev, _ = normalize_factory_kwargs(
            dtype if dtype is not None else data.dtype,
            device if device is not None else data.device,
        )
        if data._impl.dtype == _dt and data._impl.device == _dev:
            return data
        # A conversion goes through ``Tensor.to`` rather than ``tensor``:
        # ``tensor`` copies into a detached leaf, while ``as_tensor`` of a
        # Tensor is a cast that stays in the graph, as the reference
        # framework's is.
        return data.to(device=_dev, dtype=_dt)
    return tensor(data, dtype=dtype, device=device)


def from_numpy(arr: np.ndarray) -> Tensor:
    r"""Copy a NumPy ``ndarray`` into an owned Lucid tensor.

    The returned tensor owns a copy and inherits the array's dtype according
    to the canonical
    NumPy → Lucid mapping (``np.float32`` → ``lucid.float32``,
    ``np.int64`` → ``lucid.int64``, etc.). Mutations do not propagate between
    the source array and the result. Like :func:`tensor`, this bridge uses
    the active default device; use ``tensor(arr, device="cpu")`` to pin it.

    Parameters
    ----------
    arr : numpy.ndarray
        Source array.  Must reside in CPU memory.  Any layout (C / Fortran /
        strided) is accepted and copied to contiguous owned storage.

    Returns
    -------
    Tensor
        A tensor with copied values on the active default device.

    Raises
    ------
    RuntimeError
        If the array dtype has no corresponding Lucid dtype. The native
        ``DtypeMismatch`` exception derives from ``RuntimeError``; for
        example, object arrays and unsigned integer arrays are rejected.

    Notes
    -----
    This is one of the documented **H4** bridge boundaries — the only places
    where Lucid is allowed to take a NumPy array as input.  To move the
    result onto a Metal device, chain :meth:`Tensor.to`::

        t = lucid.from_numpy(arr).to("metal")

    Examples
    --------
    >>> import lucid, numpy as np
    >>> arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    >>> t = lucid.from_numpy(arr)
    >>> t.dtype
    lucid.float32
    >>> arr[0, 0] = 99.0          # mutate the source
    >>> t[0, 0].item()            # the owned copy is unchanged
    1.0
    """
    return tensor(arr)


# ── DLPack interop ─────────────────────────────────────────────────────────
#
# Lucid's DLPack bridge intentionally goes through NumPy:
#
#   * NumPy already implements the DLPack PyCapsule producer/consumer
#     correctly (lifetime, deleter, all dtypes).  A Lucid CPU tensor's
#     numpy view is itself zero-copy, so wrapping it via numpy's
#     ``__dlpack__`` adds no data movement.
#   * Metal tensors must download to CPU regardless — DLPack with
#     ``kDLMetal`` device type is supported by almost no consumers.
#   * Building our own DLPack ABI in C++ would duplicate ~350 lines of
#     subtle struct / lifetime code without any runtime benefit on top
#     of what numpy already provides.
#
# Calling ``from_dlpack`` / ``to_dlpack`` therefore opts the user into
# numpy as the canonical interop library — same H4 carve-out as
# ``tensor(np_array)``, ``Tensor.numpy()``, and ``from_numpy``.
# ``_require_numpy`` raises a clean ImportError pointing at
# ``pip install lucid[numpy]`` when numpy is absent.


# DLPack device type for Metal, as MLX tags its capsules and as
# ``Tensor.__dlpack_device__`` reports for a GPU-resident tensor.
_DLPACK_METAL = 8


def from_dlpack(ext_tensor: object) -> Tensor:
    """Construct a Lucid tensor from any object exposing the DLPack
    protocol (``__dlpack__``) or from a raw PyCapsule.

    Two dialects, picked by what the producer says it is:

    * **Metal** (``__dlpack_device__() == (8, 0)``) — adopted natively.
      The producer's ``MTLBuffer`` becomes this tensor's storage with no
      copy at all, which is what makes an ``mlx.core.array`` and a Lucid
      Metal tensor two views of one allocation. NumPy cannot read a
      capsule of this device type, so this path bypasses it entirely.
    * **Everything else** — through NumPy, which shares host memory
      where it can. The result lives on the CPU regardless of the
      producer's device; call ``.to("metal")`` after importing.

    Requires numpy for the host dialect — install via
    ``pip install lucid[numpy]`` if missing. The Metal dialect needs
    nothing.
    """
    device = getattr(ext_tensor, "__dlpack_device__", None)
    if callable(device):
        try:
            kind, _index = device()
        except Exception:  # noqa: BLE001 - a broken producer falls to numpy
            kind = None
        if kind == _DLPACK_METAL:
            from lucid._C import engine as _C_engine
            from lucid._dispatch import _wrap

            capsule = ext_tensor.__dlpack__()  # type: ignore[attr-defined]
            return _wrap(_C_engine.from_dlpack_metal(capsule))

    np = _require_numpy("lucid.from_dlpack")
    arr = np.from_dlpack(ext_tensor)  # type: ignore[attr-defined]
    return tensor(arr)


def to_dlpack(t: Tensor) -> object:
    """Export ``t`` as a DLPack PyCapsule.

    A Metal tensor exports natively — the capsule carries its
    ``MTLBuffer``, so an MLX consumer reads the same pages rather than a
    downloaded copy. A CPU tensor materialises through NumPy. Either
    capsule can be consumed exactly once.

    Requires numpy for the host dialect — install via
    ``pip install lucid[numpy]`` if missing.
    """
    if t.is_metal:
        return t.__dlpack__()
    _require_numpy("lucid.to_dlpack")
    return t.numpy().__dlpack__()
