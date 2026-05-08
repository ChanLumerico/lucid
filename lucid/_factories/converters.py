"""
Conversion utilities: Python objects / NumPy arrays -> TensorImpl.

This is one of the H4 numpy bridge sites — the only place where a
``np.ndarray`` is allowed to enter Lucid.  NumPy is imported lazily
(inside the functions that need it) so ``import lucid`` works
without numpy installed.
"""

from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs, _parse_device, _unwrap
from lucid._dtype import dtype, to_engine_dtype
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


def _np_dtype_to_engine(np_dtype: "np.dtype") -> _C_engine.Dtype:  # type: ignore[type-arg]
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
    # only the explicit ``tensor(...)`` call at this site does.
    import numpy as np  # noqa: PLC0415 — bridge import

    numpy_input = isinstance(data, np.ndarray)

    if not numpy_input:
        # Python list/scalar -> convert to numpy with default dtype (float32)
        tmp = np.array(data)
        if dtype is None:
            target_eng = _dtype_eng
        else:
            target_eng = _dtype_eng
        arr = tmp.astype(_engine_dtype_to_np(target_eng), copy=False)
        _dtype_eng = target_eng
    else:
        if dtype is not None:
            arr = data.astype(_engine_dtype_to_np(_dtype_eng), copy=False)
        else:
            arr = data
            _dtype_eng = _np_dtype_to_engine(arr.dtype)

    arr = np.ascontiguousarray(arr)
    with np.errstate(invalid="ignore", over="ignore"):
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
    """Create a tensor from data (list, ndarray, scalar, or Tensor)."""
    from lucid._tensor.tensor import Tensor

    return Tensor.__new_from_impl__(
        _to_impl(data, dtype=dtype, device=device, requires_grad=requires_grad)
    )


def as_tensor(
    data: object,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Convert data to a tensor, sharing memory where possible."""
    return tensor(data, dtype=dtype, device=device)


def from_numpy(arr: np.ndarray) -> Tensor:  # type: ignore[type-arg]
    """Create a CPU tensor from a NumPy array with automatic dtype mapping."""
    return tensor(arr)


# ── DLPack interop ─────────────────────────────────────────────────────────
#
# Lucid's DLPack bridge runs through NumPy: a Lucid tensor exposes
# ``__dlpack__`` / ``__dlpack_device__`` by converting to NumPy first
# (which is a no-op for CPU tensors and a CPU round-trip for Metal
# tensors).  This is enough for the reference framework / JAX / SciPy / etc to consume
# Lucid tensors zero-copy on the CPU side.  A real engine-side DLPack
# implementation that avoids the NumPy hop is filed as future work.


def from_dlpack(ext_tensor: object) -> Tensor:  # type: ignore[type-arg]
    """Construct a Lucid tensor from any object exposing the DLPack
    protocol (``__dlpack__``) or from a raw PyCapsule.

    Memory is shared with ``ext_tensor`` where possible (CPU only).  The
    result lives on CPU regardless of the producer's device — Metal
    consumers should ``.to(device='metal')`` after the import.
    """
    import numpy as np  # noqa: PLC0415 — DLPack bridge is a numpy bridge
    arr = np.from_dlpack(ext_tensor)
    return tensor(arr)


def to_dlpack(t: Tensor) -> object:
    """Export ``t`` as a DLPack PyCapsule.

    Always materialises through NumPy, so GPU tensors take a CPU
    round-trip.  The capsule can be consumed exactly once — pass it
    directly to ``np.from_dlpack`` / a reference-framework consumer / etc.
    """
    return t.numpy().__dlpack__()
