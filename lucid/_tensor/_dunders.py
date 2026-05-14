"""
Dunder methods injected into the Tensor class.

All arithmetic/comparison operators are implemented here and attached to
the Tensor class by _inject_dunders() at module import time.
"""

from typing import TYPE_CHECKING, cast
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._tensor._indexing import _getitem, _setitem

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid._types import TensorOrScalar, _IndexType


# ── Dtype promotion helpers ───────────────────────────────────────────────────
# Maps engine Dtype enum → (kind, width).
# kind: 0=bool, 1=int, 2=float, 3=complex  (higher kind wins across categories)
# width: bit-width (higher width wins within the same kind)
_D = _C_engine.Dtype
_DTYPE_KIND_WIDTH: dict[_C_engine.Dtype, tuple[int, int]] = {
    _D.Bool: (0, 1),
    _D.I8: (1, 8),
    _D.I16: (1, 16),
    _D.I32: (1, 32),
    _D.I64: (1, 64),
    _D.F16: (2, 16),
    _D.F32: (2, 32),
    _D.F64: (2, 64),
    _D.C64: (3, 64),
}


def _result_dtype(da: _C_engine.Dtype, db: _C_engine.Dtype) -> _C_engine.Dtype:
    """Return the type-promotion result dtype for two arithmetic operands."""
    if da == db:
        return da
    ka, wa = _DTYPE_KIND_WIDTH.get(da, (2, 32))
    kb, wb = _DTYPE_KIND_WIDTH.get(db, (2, 32))
    if ka != kb:
        return da if ka > kb else db
    return da if wa >= wb else db


def _maybe_promote(
    a_impl: _C_engine.TensorImpl, b_impl: _C_engine.TensorImpl
) -> tuple[_C_engine.TensorImpl, _C_engine.TensorImpl]:
    """Cast a_impl / b_impl to their common promoted dtype if they differ."""
    da, db = a_impl.dtype, b_impl.dtype
    if da == db:
        return a_impl, b_impl
    tgt = _result_dtype(da, db)
    if da != tgt:
        a_impl = _C_engine.astype(a_impl, tgt)
    if db != tgt:
        b_impl = _C_engine.astype(b_impl, tgt)
    return a_impl, b_impl


def _unwrap_or_scalar(
    x: TensorOrScalar,
    ref_impl: _C_engine.TensorImpl | None = None,
) -> _C_engine.TensorImpl:
    """
    Return TensorImpl for Tensor; convert scalars to scalar TensorImpl.
    ref_impl is used to match dtype/device for scalar→TensorImpl conversion.
    """
    # Avoid circular import: check duck-type attribute instead of isinstance
    impl = getattr(x, "_impl", None)
    if impl is not None and isinstance(impl, _C_engine.TensorImpl):
        return cast(_C_engine.TensorImpl, impl)
    if isinstance(x, _C_engine.TensorImpl):
        return x

    # scalar → TensorImpl broadcast to ref_impl shape, via engine ops only
    if isinstance(x, (int, float, bool)):
        if ref_impl is not None:
            dtype = ref_impl.dtype
            device = ref_impl.device
            shape = list(ref_impl.shape)
        else:
            dtype = _C_engine.Dtype.F32
            device = _C_engine.Device.CPU
            shape = []
        return _C_engine.full(shape, float(x), dtype, device)

    raise TypeError(f"Cannot convert {type(x).__name__} to TensorImpl")


def _inject_dunders(cls: type) -> None:
    """Attach all dunder methods to the Tensor class."""

    def __add__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise addition: ``self + other`` with dtype promotion and broadcasting."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.add(a, b))

    def __radd__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Reflected addition: ``other + self``."""
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.add(a, b))

    def __iadd__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """In-place addition: ``self += other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.add_(a, b)
        return self

    def __sub__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise subtraction: ``self - other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.sub(a, b))

    def __rsub__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Reflected subtraction: ``other - self``."""
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.sub(a, b))

    def __isub__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """In-place subtraction: ``self -= other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.sub_(a, b)
        return self

    def __mul__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise multiplication: ``self * other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.mul(a, b))

    def __rmul__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Reflected multiplication: ``other * self``."""
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.mul(a, b))

    def __imul__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """In-place multiplication: ``self *= other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.mul_(a, b)
        return self

    def __truediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise true division: ``self / other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.div(a, b))

    def __rtruediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Reflected true division: ``other / self``."""
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.div(a, b))

    def __itruediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """In-place true division: ``self /= other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.div_(a, b)
        return self

    def __floordiv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise floor division: ``self // other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.floordiv(a, b))

    def __rfloordiv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Reflected floor division: ``other // self``."""
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.floordiv(a, b))

    def __pow__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise exponentiation: ``self ** other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.pow(a, b))

    def __rpow__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Reflected exponentiation: ``other ** self``."""
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.pow(a, b))

    def __ipow__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """In-place exponentiation: ``self **= other``."""
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.pow_(a, b)
        return self

    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        """Matrix multiplication: ``self @ other`` with batched semantics."""
        return _wrap(_C_engine.matmul(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __rmatmul__(self: Tensor, other: Tensor) -> Tensor:
        """Reflected matrix multiplication: ``other @ self``."""
        return _wrap(_C_engine.matmul(_unwrap_or_scalar(other, self._impl), self._impl))

    def __neg__(self: Tensor) -> Tensor:
        """Unary negation: ``-self``."""
        return _wrap(_C_engine.neg(self._impl))

    def __abs__(self: Tensor) -> Tensor:
        """Element-wise absolute value: ``abs(self)``."""
        return _wrap(_C_engine.abs(self._impl))

    def __invert__(self: Tensor) -> Tensor:
        """Bitwise/logical NOT: ``~self`` (boolean negation for bool tensors)."""
        return _wrap(_C_engine.invert(self._impl))

    def __and__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise bitwise/logical AND: ``self & other``."""
        return _wrap(
            _C_engine.bitwise_and(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __or__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise bitwise/logical OR: ``self | other``."""
        return _wrap(
            _C_engine.bitwise_or(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __xor__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise bitwise/logical XOR: ``self ^ other``."""
        return _wrap(
            _C_engine.bitwise_xor(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __eq__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise equality comparison.  Returns a boolean Tensor."""
        return _wrap(_C_engine.equal(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __ne__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise inequality comparison.  Returns a boolean Tensor."""
        return _wrap(
            _C_engine.not_equal(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __lt__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise less-than comparison.  Returns a boolean Tensor."""
        return _wrap(_C_engine.less(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __le__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise less-than-or-equal comparison.  Returns a boolean Tensor."""
        return _wrap(
            _C_engine.less_equal(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __gt__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise greater-than comparison.  Returns a boolean Tensor."""
        return _wrap(
            _C_engine.greater(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __ge__(self: Tensor, other: TensorOrScalar) -> Tensor:
        """Element-wise greater-than-or-equal comparison.  Returns a boolean Tensor."""
        return _wrap(
            _C_engine.greater_equal(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __getitem__(self: Tensor, idx: _IndexType) -> Tensor:
        """Tensor indexing: dispatches to :func:`lucid._tensor._indexing._getitem`.

        Supports basic indexing (ints, slices, ``None``/``...``), advanced
        indexing (integer / boolean tensors), and combinations thereof.
        """
        return _getitem(self, idx)

    def __setitem__(self: Tensor, idx: _IndexType, value: TensorOrScalar) -> None:
        """In-place indexed assignment: ``self[idx] = value``.

        Dispatches to :func:`lucid._tensor._indexing._setitem` and supports
        the same index forms as :meth:`__getitem__`.
        """
        _setitem(self, idx, value)

    # attach all methods
    for _name, _fn in [
        ("__add__", __add__),
        ("__radd__", __radd__),
        ("__iadd__", __iadd__),
        ("__sub__", __sub__),
        ("__rsub__", __rsub__),
        ("__isub__", __isub__),
        ("__mul__", __mul__),
        ("__rmul__", __rmul__),
        ("__imul__", __imul__),
        ("__truediv__", __truediv__),
        ("__rtruediv__", __rtruediv__),
        ("__itruediv__", __itruediv__),
        ("__floordiv__", __floordiv__),
        ("__rfloordiv__", __rfloordiv__),
        ("__pow__", __pow__),
        ("__rpow__", __rpow__),
        ("__ipow__", __ipow__),
        ("__matmul__", __matmul__),
        ("__rmatmul__", __rmatmul__),
        ("__neg__", __neg__),
        ("__abs__", __abs__),
        ("__invert__", __invert__),
        ("__and__", __and__),
        ("__or__", __or__),
        ("__xor__", __xor__),
        ("__eq__", __eq__),
        ("__ne__", __ne__),
        ("__lt__", __lt__),
        ("__le__", __le__),
        ("__gt__", __gt__),
        ("__ge__", __ge__),
        ("__getitem__", __getitem__),
        ("__setitem__", __setitem__),
    ]:
        setattr(cls, _name, _fn)
