"""
Conversion utilities: Python objects / NumPy arrays -> TensorImpl.
"""

from typing import TYPE_CHECKING
import numpy as np

from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs, _parse_device, _unwrap
from lucid._dtype import dtype, to_engine_dtype

if TYPE_CHECKING:
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


def _np_dtype_to_engine(np_dtype: np.dtype) -> _C_engine.Dtype:  # type: ignore[type-arg]
    name = np_dtype.name
    if name in _NP_TO_ENGINE_DTYPE:
        return _NP_TO_ENGINE_DTYPE[name]
    if name.startswith("float"):
        return _C_engine.Dtype.F64
    return _C_engine.Dtype.F32


def _to_impl(
    data: object,
    *,
    dtype: dtype | _C_engine.Dtype | str | None = None,
    device: str | None = None,
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

    numpy_input = isinstance(data, np.ndarray)

    # Convert data to numpy
    if not numpy_input:
        # Python list/scalar -> convert to numpy with default dtype (float32)
        tmp = np.array(data)
        if dtype is None:
            # Use default dtype for Python scalars/lists
            target_eng = (
                _dtype_eng  # already set to default by normalize_factory_kwargs
            )
        else:
            target_eng = _dtype_eng
        arr = tmp.astype(_engine_dtype_to_np(target_eng), copy=False)
        _dtype_eng = target_eng
    else:
        if dtype is not None:
            # Explicit dtype override
            arr = data.astype(_engine_dtype_to_np(_dtype_eng), copy=False)
        else:
            # Preserve numpy array's dtype
            arr = data
            _dtype_eng = _np_dtype_to_engine(arr.dtype)

    arr = np.ascontiguousarray(arr)
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
    dtype: dtype | _C_engine.Dtype | str | None = None,
    device: str | None = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor from data (list, ndarray, scalar, or Tensor)."""
    from lucid._tensor.tensor import Tensor

    return Tensor.__new_from_impl__(
        _to_impl(data, dtype=dtype, device=device, requires_grad=requires_grad)
    )


def as_tensor(
    data: object,
    dtype: dtype | _C_engine.Dtype | str | None = None,
    device: str | None = None,
) -> Tensor:
    """Convert data to a tensor, sharing memory where possible."""
    return tensor(data, dtype=dtype, device=device)


def from_numpy(arr: np.ndarray) -> Tensor:  # type: ignore[type-arg]
    """Create a CPU tensor from a NumPy array with automatic dtype mapping."""
    return tensor(arr)
