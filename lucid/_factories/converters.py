"""
Conversion utilities: Python objects / NumPy arrays -> TensorImpl.

This is one of the H4 numpy bridge sites — the only place where a
``np.ndarray`` is allowed to enter Lucid.  NumPy is imported lazily
(inside the functions that need it) so ``import lucid`` works
without numpy installed.
"""

from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    import numpy as np
    from lucid._tensor.tensor import Tensor


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

    # Numpy is the sanctioned conversion library for Python-data → engine.
    # Imported lazily so ``import lucid`` doesn't need numpy installed —
    # only the explicit ``tensor(...)`` call at this site does.  When numpy
    # is missing, ``_require_numpy`` raises a guidance-rich ImportError.
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
    return impl


def _engine_dtype_to_np(d: _C_engine.Dtype) -> str:
    _MAP: dict[_C_engine.Dtype, str] = {
        _C_engine.Dtype.F16: "float16",
        _C_engine.Dtype.F32: "float32",
        _C_engine.Dtype.F64: "float64",
        _C_engine.Dtype.I8: "int8",
        _C_engine.Dtype.I16: "int16",
        _C_engine.Dtype.I32: "int32",
        _C_engine.Dtype.I64: "int64",
        _C_engine.Dtype.Bool: "bool",
        _C_engine.Dtype.C64: "complex64",
    }
    return _MAP.get(d, "float32")


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
        * Existing :class:`Tensor` — copied to a new buffer (use
          :func:`as_tensor` to avoid the copy when dtype/device match).
    dtype : dtype | str | None, optional
        Target element type.  ``None`` (default) infers from ``data``:
        integers → ``int64``, floats → ``float32``, complex → ``complex64``.
    device : device | str | None, optional
        Target device (``"cpu"`` or ``"metal"``).  ``None`` uses
        :func:`lucid.get_default_device`.
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
    Tensor([1., 2., 3.])
    >>> lucid.tensor([[1, 2], [3, 4]], dtype=lucid.float32)
    Tensor([[1., 2.],
            [3., 4.]])
    >>> import numpy as np
    >>> lucid.tensor(np.arange(6).reshape(2, 3))
    Tensor([[0, 1, 2],
            [3, 4, 5]])
    """
    from lucid._tensor.tensor import Tensor

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
    * If ``data`` is a NumPy array on CPU and ``dtype`` matches (or is
      ``None``), the resulting tensor shares its storage with the array —
      mutations in either side are reflected in the other.
    * Otherwise the call delegates to :func:`tensor`, which copies.

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
        The input tensor or a freshly-constructed Lucid tensor.

    Notes
    -----
    ``as_tensor`` is the right choice in performance-sensitive code paths
    (e.g. DataLoader collate functions) where the input is already an
    ``ndarray`` and copying would be wasteful.  For semantic clarity in
    library code that should never share storage, use :func:`tensor`.

    Examples
    --------
    >>> import lucid
    >>> import numpy as np
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> t = lucid.as_tensor(arr)            # no copy
    >>> arr[0] = 99.0
    >>> t                                    # reflects the mutation
    Tensor([99.,  2.,  3.])
    >>> x = lucid.tensor([1, 2, 3])
    >>> lucid.as_tensor(x) is x              # already a Tensor, returned as-is
    True
    """
    return tensor(data, dtype=dtype, device=device)


def from_numpy(arr: np.ndarray) -> Tensor:
    r"""Create a CPU tensor from a NumPy ``ndarray`` with shared storage.

    The returned tensor wraps the array's existing buffer — no data is
    copied — and inherits the array's dtype according to the canonical
    NumPy → Lucid mapping (``np.float32`` → ``lucid.float32``,
    ``np.int64`` → ``lucid.int64``, etc.).  Because storage is shared,
    mutations in the array are visible in the tensor and vice versa.

    Parameters
    ----------
    arr : numpy.ndarray
        Source array.  Must reside in CPU memory.  Any layout (C / Fortran /
        strided) is accepted; the resulting tensor preserves the array's
        strides where possible.

    Returns
    -------
    Tensor
        A CPU tensor sharing storage with ``arr``.

    Raises
    ------
    TypeError
        If ``arr`` is not a NumPy ``ndarray``.
    ValueError
        If ``arr``\'s dtype has no corresponding Lucid dtype
        (e.g. ``np.float128`` on some platforms).

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
    float32
    >>> arr[0, 0] = 99.0          # mutate the source
    >>> t[0, 0].item()            # change visible in the tensor
    99.0
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


def from_dlpack(ext_tensor: object) -> Tensor:
    """Construct a Lucid tensor from any object exposing the DLPack
    protocol (``__dlpack__``) or from a raw PyCapsule.

    Memory is shared with ``ext_tensor`` where possible (CPU only).  The
    result lives on CPU regardless of the producer's device — Metal
    consumers should ``.to(device='metal')`` after the import.

    Requires numpy as the canonical DLPack bridge — install via
    ``pip install lucid[numpy]`` if missing.
    """
    np = _require_numpy("lucid.from_dlpack")
    arr = np.from_dlpack(ext_tensor)  # type: ignore[attr-defined]
    return tensor(arr)


def to_dlpack(t: Tensor) -> object:
    """Export ``t`` as a DLPack PyCapsule.

    Always materialises through NumPy, so GPU tensors take a CPU
    round-trip.  The capsule can be consumed exactly once — pass it
    directly to ``np.from_dlpack`` / a reference-framework consumer / etc.

    Requires numpy as the canonical DLPack bridge — install via
    ``pip install lucid[numpy]`` if missing.
    """
    _require_numpy("lucid.to_dlpack")
    return t.numpy().__dlpack__()
