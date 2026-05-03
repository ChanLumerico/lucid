"""
Dunder methods injected into the Tensor class.

All arithmetic/comparison operators are implemented here and attached to
the Tensor class by _inject_dunders() at module import time.
"""

from typing import Any, TYPE_CHECKING
from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _unwrap_or_scalar(
    x: Any,
    ref_impl: _C_engine.TensorImpl | None = None,
) -> _C_engine.TensorImpl:
    """
    Return TensorImpl for Tensor; convert scalars to scalar TensorImpl.
    ref_impl is used to match dtype/device for scalar→TensorImpl conversion.
    """
    import numpy as np
    from lucid._tensor.tensor import Tensor as _Tensor

    if isinstance(x, _Tensor):
        return x._impl
    if isinstance(x, _C_engine.TensorImpl):
        return x

    # scalar → TensorImpl broadcast to ref_impl shape
    if isinstance(x, (int, float, bool)):
        if ref_impl is not None:
            dtype = ref_impl.dtype
            device = ref_impl.device
            shape = list(ref_impl.shape)
        else:
            dtype = _C_engine.Dtype.F32
            device = _C_engine.Device.CPU
            shape = []
        np_dtype = _DTYPE_TO_NP.get(dtype, "float32")
        if shape:
            arr = np.full(shape, x, dtype=np_dtype)
        else:
            arr = np.array(x, dtype=np_dtype)
        return _C_engine.TensorImpl(arr, device, False)

    raise TypeError(f"Cannot convert {type(x).__name__} to TensorImpl")


_DTYPE_TO_NP: dict[_C_engine.Dtype, str] = {
    _C_engine.Dtype.F16: "float16",
    _C_engine.Dtype.F32: "float32",
    _C_engine.Dtype.F64: "float64",
    _C_engine.Dtype.I8:  "int8",
    _C_engine.Dtype.I16: "int16",
    _C_engine.Dtype.I32: "int32",
    _C_engine.Dtype.I64: "int64",
    _C_engine.Dtype.Bool: "bool",
    _C_engine.Dtype.C64: "complex64",
}


def _inject_dunders(cls: type) -> None:
    """Attach all dunder methods to the Tensor class."""
    from lucid._dispatch import _wrap

    def __add__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.add(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __radd__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.add(_unwrap_or_scalar(other, self._impl), self._impl))

    def __iadd__(self: Tensor, other: Any) -> Tensor:
        self._impl = _C_engine.add_(self._impl, _unwrap_or_scalar(other, self._impl))
        return self

    def __sub__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.sub(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __rsub__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.sub(_unwrap_or_scalar(other, self._impl), self._impl))

    def __isub__(self: Tensor, other: Any) -> Tensor:
        self._impl = _C_engine.sub_(self._impl, _unwrap_or_scalar(other, self._impl))
        return self

    def __mul__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.mul(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __rmul__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.mul(_unwrap_or_scalar(other, self._impl), self._impl))

    def __imul__(self: Tensor, other: Any) -> Tensor:
        self._impl = _C_engine.mul_(self._impl, _unwrap_or_scalar(other, self._impl))
        return self

    def __truediv__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.div(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __rtruediv__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.div(_unwrap_or_scalar(other, self._impl), self._impl))

    def __itruediv__(self: Tensor, other: Any) -> Tensor:
        self._impl = _C_engine.div_(self._impl, _unwrap_or_scalar(other, self._impl))
        return self

    def __floordiv__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.floordiv(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __rfloordiv__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.floordiv(_unwrap_or_scalar(other, self._impl), self._impl))

    def __pow__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.pow(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __rpow__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.pow(_unwrap_or_scalar(other, self._impl), self._impl))

    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        return _wrap(_C_engine.matmul(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __rmatmul__(self: Tensor, other: Tensor) -> Tensor:
        return _wrap(_C_engine.matmul(_unwrap_or_scalar(other, self._impl), self._impl))

    def __neg__(self: Tensor) -> Tensor:
        return _wrap(_C_engine.neg(self._impl))

    def __abs__(self: Tensor) -> Tensor:
        return _wrap(_C_engine.abs(self._impl))

    def __invert__(self: Tensor) -> Tensor:
        return _wrap(_C_engine.invert(self._impl))

    def __and__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.bitwise_and(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __or__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.bitwise_or(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __xor__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.bitwise_xor(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __eq__(self: Tensor, other: Any) -> Tensor:  # type: ignore[override]
        return _wrap(_C_engine.equal(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __ne__(self: Tensor, other: Any) -> Tensor:  # type: ignore[override]
        return _wrap(_C_engine.not_equal(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __lt__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.less(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __le__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.less_equal(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __gt__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.greater(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __ge__(self: Tensor, other: Any) -> Tensor:
        return _wrap(_C_engine.greater_equal(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __getitem__(self: Tensor, idx: Any) -> Tensor:
        from lucid._tensor._indexing import _getitem
        return _getitem(self, idx)

    def __setitem__(self: Tensor, idx: Any, value: Any) -> None:
        from lucid._tensor._indexing import _setitem
        _setitem(self, idx, value)

    # attach all methods
    for _name, _fn in [
        ("__add__", __add__), ("__radd__", __radd__), ("__iadd__", __iadd__),
        ("__sub__", __sub__), ("__rsub__", __rsub__), ("__isub__", __isub__),
        ("__mul__", __mul__), ("__rmul__", __rmul__), ("__imul__", __imul__),
        ("__truediv__", __truediv__), ("__rtruediv__", __rtruediv__),
        ("__itruediv__", __itruediv__),
        ("__floordiv__", __floordiv__), ("__rfloordiv__", __rfloordiv__),
        ("__pow__", __pow__), ("__rpow__", __rpow__),
        ("__matmul__", __matmul__), ("__rmatmul__", __rmatmul__),
        ("__neg__", __neg__), ("__abs__", __abs__), ("__invert__", __invert__),
        ("__and__", __and__), ("__or__", __or__), ("__xor__", __xor__),
        ("__eq__", __eq__), ("__ne__", __ne__),
        ("__lt__", __lt__), ("__le__", __le__),
        ("__gt__", __gt__), ("__ge__", __ge__),
        ("__getitem__", __getitem__), ("__setitem__", __setitem__),
    ]:
        setattr(cls, _name, _fn)
