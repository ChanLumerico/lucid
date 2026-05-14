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
        r"""Element-wise addition: ``self + other`` with broadcasting and dtype promotion.

        Forwards to the engine ``add`` op.  The right operand may be another
        Tensor (broadcast-compatible) or a Python scalar (auto-promoted to
        the tensor's dtype and broadcast across the shape).

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            Sum with shape ``broadcast(self.shape, other.shape)`` and dtype
            ``promote_types(self.dtype, other.dtype)``.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \text{self}_i + \text{other}_i

        Mixed-dtype operands are promoted to their common type before the
        add.  Both operands receive gradient flow with unit Jacobian
        (:math:`\partial \text{out} / \partial \text{self} = 1`,
        :math:`\partial \text{out} / \partial \text{other} = 1`).

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> b = lucid.tensor([10.0, 20.0, 30.0])
        >>> a + b
        Tensor([11., 22., 33.])
        >>> a + 5
        Tensor([6., 7., 8.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.add(a, b))

    def __radd__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Reflected addition: ``other + self``.

        Triggered when the left operand does not implement ``__add__`` with
        a Tensor right operand (typically when ``other`` is a plain Python
        scalar).  Addition is commutative so the result equals
        :meth:`__add__`.

        Parameters
        ----------
        other : Tensor or scalar
            Left-hand operand of the original expression.

        Returns
        -------
        Tensor
            Sum with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \text{other}_i + \text{self}_i

        Order is preserved for symmetry with :meth:`__rsub__` even though
        addition commutes.  Dtype promotion and broadcasting follow the
        same rules as :meth:`__add__`.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> 5 + a
        Tensor([6., 7., 8.])
        """
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.add(a, b))

    def __iadd__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""In-place addition: ``self += other``.

        Mutates ``self._impl`` to hold the sum and returns ``self`` so the
        expression can be chained.  Useful for memory-tight loops; for
        leaves that participate in autograd prefer the out-of-place
        :meth:`__add__`.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            ``self`` after mutation.

        Notes
        -----
        Math (in place):

        .. math::

            \text{self}_i \leftarrow \text{self}_i + \text{other}_i

        Performing an in-place add on a leaf tensor with
        ``requires_grad=True`` is typically rejected by autograd because it
        would destroy the value needed for the backward pass.  Use the
        non-in-place form there.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> a += 10
        >>> a
        Tensor([11., 12., 13.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.add_(a, b)
        return self

    def __sub__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise subtraction: ``self - other`` with broadcasting.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            Difference with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \text{self}_i - \text{other}_i

        Dtype promotion follows the standard kind/width rules.  Gradients
        flow with Jacobian ``+1`` for ``self`` and ``-1`` for ``other``.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([10.0, 20.0, 30.0])
        >>> b = lucid.tensor([1.0, 2.0, 3.0])
        >>> a - b
        Tensor([9., 18., 27.])
        >>> a - 1
        Tensor([9., 19., 29.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.sub(a, b))

    def __rsub__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Reflected subtraction: ``other - self``.

        Triggered when the left operand does not implement ``__sub__`` with
        a Tensor right operand.  Subtraction is **not** commutative, so the
        order of operands matters.

        Parameters
        ----------
        other : Tensor or scalar
            Left-hand operand of the original expression.

        Returns
        -------
        Tensor
            Difference with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \text{other}_i - \text{self}_i

        Jacobians are ``+1`` for ``other`` and ``-1`` for ``self``.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> 10 - a
        Tensor([9., 8., 7.])
        """
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.sub(a, b))

    def __isub__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""In-place subtraction: ``self -= other``.

        Mutates ``self._impl`` and returns ``self``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            ``self`` after mutation.

        Notes
        -----
        Math (in place):

        .. math::

            \text{self}_i \leftarrow \text{self}_i - \text{other}_i

        In-place mutation of an autograd leaf with ``requires_grad=True``
        is typically rejected because it destroys the saved value needed
        for backward.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([10.0, 20.0, 30.0])
        >>> a -= 5
        >>> a
        Tensor([5., 15., 25.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.sub_(a, b)
        return self

    def __mul__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise multiplication: ``self * other`` (Hadamard product).

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            Product with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \text{self}_i \cdot \text{other}_i

        This is the **element-wise** product; for matrix multiplication
        use :meth:`__matmul__` (``@`` operator).  Jacobians are
        :math:`\partial \text{out}_i / \partial \text{self}_i = \text{other}_i`
        and
        :math:`\partial \text{out}_i / \partial \text{other}_i = \text{self}_i`.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> b = lucid.tensor([4.0, 5.0, 6.0])
        >>> a * b
        Tensor([4., 10., 18.])
        >>> a * 2
        Tensor([2., 4., 6.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.mul(a, b))

    def __rmul__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Reflected multiplication: ``other * self``.

        Triggered when the left operand does not implement ``__mul__``
        with a Tensor right operand (e.g. plain scalar on the left).
        Multiplication is commutative, so the result equals
        :meth:`__mul__`.

        Parameters
        ----------
        other : Tensor or scalar
            Left-hand operand of the original expression.

        Returns
        -------
        Tensor
            Product with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \text{other}_i \cdot \text{self}_i

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> 3 * a
        Tensor([3., 6., 9.])
        """
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.mul(a, b))

    def __imul__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""In-place multiplication: ``self *= other``.

        Mutates ``self._impl`` and returns ``self``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            ``self`` after mutation.

        Notes
        -----
        Math (in place):

        .. math::

            \text{self}_i \leftarrow \text{self}_i \cdot \text{other}_i

        In-place multiplication on an autograd leaf with
        ``requires_grad=True`` is typically rejected.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> a *= 10
        >>> a
        Tensor([10., 20., 30.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.mul_(a, b)
        return self

    def __truediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise true division: ``self / other``.

        Always produces a floating-point result.  Integer operands are
        promoted to float before the division.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            Quotient with broadcast shape and a float dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \frac{\text{self}_i}{\text{other}_i}

        Division by zero follows IEEE 754: positive / 0 -> ``+inf``,
        0 / 0 -> ``nan``.  Jacobians are
        :math:`\partial \text{out}_i / \partial \text{self}_i = 1 / \text{other}_i`
        and
        :math:`\partial \text{out}_i / \partial \text{other}_i = -\text{self}_i / \text{other}_i^2`.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([10.0, 20.0, 30.0])
        >>> b = lucid.tensor([2.0, 4.0, 5.0])
        >>> a / b
        Tensor([5., 5., 6.])
        >>> a / 10
        Tensor([1., 2., 3.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.div(a, b))

    def __rtruediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Reflected true division: ``other / self``.

        Triggered when the left operand does not implement ``__truediv__``
        with a Tensor right operand.  Division is **not** commutative.

        Parameters
        ----------
        other : Tensor or scalar
            Left-hand operand of the original expression.

        Returns
        -------
        Tensor
            Quotient with broadcast shape and float dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \frac{\text{other}_i}{\text{self}_i}

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([2.0, 4.0, 5.0])
        >>> 20 / a
        Tensor([10., 5., 4.])
        """
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.div(a, b))

    def __itruediv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""In-place true division: ``self /= other``.

        Mutates ``self._impl`` and returns ``self``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            ``self`` after mutation.

        Notes
        -----
        Math (in place):

        .. math::

            \text{self}_i \leftarrow \frac{\text{self}_i}{\text{other}_i}

        Requires ``self`` to already have a float dtype since the result
        cannot be cast back to integer in place.  Rejected on autograd
        leaves with ``requires_grad=True``.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([10.0, 20.0, 30.0])
        >>> a /= 10
        >>> a
        Tensor([1., 2., 3.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.div_(a, b)
        return self

    def __floordiv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise floor division: ``self // other``.

        Computes the largest integer less than or equal to the true
        quotient, broadcast element-wise.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand.

        Returns
        -------
        Tensor
            Floor-divided result with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \left\lfloor \frac{\text{self}_i}{\text{other}_i} \right\rfloor

        For integer operands the result is integer; for float operands
        the result is float but takes integral values.  Floor division by
        zero follows the engine's convention (typically raises or
        produces NaN/inf depending on dtype).  Not differentiable at
        integer-quotient boundaries.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([7.0, 8.0, 9.0])
        >>> a // 2
        Tensor([3., 4., 4.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.floordiv(a, b))

    def __rfloordiv__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Reflected floor division: ``other // self``.

        Triggered when the left operand does not implement
        ``__floordiv__`` with a Tensor right operand.

        Parameters
        ----------
        other : Tensor or scalar
            Left-hand operand of the original expression.

        Returns
        -------
        Tensor
            Floor-divided result with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \left\lfloor \frac{\text{other}_i}{\text{self}_i} \right\rfloor

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([2.0, 3.0, 4.0])
        >>> 10 // a
        Tensor([5., 3., 2.])
        """
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.floordiv(a, b))

    def __pow__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise exponentiation: ``self ** other``.

        Raises ``self`` to the ``other`` power element-by-element with
        broadcasting and dtype promotion.

        Parameters
        ----------
        other : Tensor or scalar
            Exponent.

        Returns
        -------
        Tensor
            Powered result with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \text{self}_i^{\text{other}_i}

        Gradients are
        :math:`\partial \text{out}_i / \partial \text{self}_i =
        \text{other}_i \cdot \text{self}_i^{\text{other}_i - 1}`
        and
        :math:`\partial \text{out}_i / \partial \text{other}_i =
        \text{self}_i^{\text{other}_i} \cdot \ln(\text{self}_i)`.
        Negative base with non-integer exponent produces NaN under IEEE
        754 unless complex dtype is used.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> a ** 2
        Tensor([1., 4., 9.])
        >>> a ** lucid.tensor([3.0, 2.0, 1.0])
        Tensor([1., 4., 3.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        return _wrap(_C_engine.pow(a, b))

    def __rpow__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Reflected exponentiation: ``other ** self``.

        Triggered when the left operand does not implement ``__pow__``
        with a Tensor right operand.  Exponentiation is **not**
        commutative.

        Parameters
        ----------
        other : Tensor or scalar
            Base of the original expression.

        Returns
        -------
        Tensor
            Powered result with broadcast shape and promoted dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \text{other}_i^{\text{self}_i}

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> 2 ** a
        Tensor([2., 4., 8.])
        """
        a, b = _maybe_promote(_unwrap_or_scalar(other, self._impl), self._impl)
        return _wrap(_C_engine.pow(a, b))

    def __ipow__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""In-place exponentiation: ``self **= other``.

        Mutates ``self._impl`` and returns ``self``.

        Parameters
        ----------
        other : Tensor or scalar
            Exponent.

        Returns
        -------
        Tensor
            ``self`` after mutation.

        Notes
        -----
        Math (in place):

        .. math::

            \text{self}_i \leftarrow \text{self}_i^{\text{other}_i}

        Rejected on autograd leaves with ``requires_grad=True`` because
        the operation cannot be reversed.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> a **= 2
        >>> a
        Tensor([1., 4., 9.])
        """
        a, b = _maybe_promote(self._impl, _unwrap_or_scalar(other, self._impl))
        self._impl = _C_engine.pow_(a, b)
        return self

    def __matmul__(self: Tensor, other: Tensor) -> Tensor:
        r"""Matrix multiplication: ``self @ other`` with batched semantics.

        Supports 2-D (matrix) and N-D (batched) inputs.  For 1-D vectors
        the usual NumPy/BLAS dot-product semantics apply (1-D @ 1-D is
        scalar, 1-D @ 2-D is a row vector, 2-D @ 1-D is a column vector).
        For N-D operands the trailing two dimensions are matrix-multiplied
        while leading dimensions broadcast.

        Parameters
        ----------
        other : Tensor
            Right-hand operand; inner dimension must equal ``self``'s last
            dimension.

        Returns
        -------
        Tensor
            Matrix product with shape determined by broadcasting batch
            dims and contracting the inner dimension.

        Notes
        -----
        Math (2-D case):

        .. math::

            \text{out}_{ij} = \sum_k \text{self}_{ik} \cdot \text{other}_{kj}

        For N-D inputs the same contraction is applied to the last two
        axes:

        .. math::

            \text{out}_{\dots ij} = \sum_k \text{self}_{\dots ik} \cdot \text{other}_{\dots kj}

        Backward passes both matmuls' transposes through the chain rule.
        Use :meth:`__mul__` (``*``) for element-wise multiplication.

        Examples
        --------
        >>> import lucid
        >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> B = lucid.tensor([[5.0, 6.0], [7.0, 8.0]])
        >>> A @ B
        Tensor([[19., 22.], [43., 50.]])
        """
        return _wrap(_C_engine.matmul(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __rmatmul__(self: Tensor, other: Tensor) -> Tensor:
        r"""Reflected matrix multiplication: ``other @ self``.

        Triggered when the left operand does not implement ``__matmul__``
        with a Tensor right operand.  Matrix multiplication is **not**
        commutative.

        Parameters
        ----------
        other : Tensor
            Left-hand operand of the original expression.

        Returns
        -------
        Tensor
            Matrix product with broadcasted batch dims and contracted
            inner dimension.

        Notes
        -----
        Math (2-D case):

        .. math::

            \text{out}_{ij} = \sum_k \text{other}_{ik} \cdot \text{self}_{kj}

        Examples
        --------
        >>> import lucid
        >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> B = lucid.tensor([[5.0, 6.0], [7.0, 8.0]])
        >>> B.__rmatmul__(A)
        Tensor([[19., 22.], [43., 50.]])
        """
        return _wrap(_C_engine.matmul(_unwrap_or_scalar(other, self._impl), self._impl))

    def __neg__(self: Tensor) -> Tensor:
        r"""Unary negation: ``-self``.

        Returns
        -------
        Tensor
            Element-wise negation with the same shape and dtype as
            ``self``.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = -\text{self}_i

        Defined for all signed numeric dtypes.  Applied to an unsigned
        integer dtype the engine raises ``TypeError`` because the result
        would not be representable.  Jacobian is ``-1``.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, -2.0, 3.0])
        >>> -a
        Tensor([-1., 2., -3.])
        """
        return _wrap(_C_engine.neg(self._impl))

    def __abs__(self: Tensor) -> Tensor:
        r"""Element-wise absolute value: ``abs(self)``.

        Returns
        -------
        Tensor
            Magnitude tensor with the same shape as ``self``.  For complex
            input the result has the corresponding real float dtype.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \lvert \text{self}_i \rvert

        For complex input :math:`\lvert z \rvert = \sqrt{\Re(z)^2 + \Im(z)^2}`.
        The derivative is :math:`\operatorname{sign}(\text{self}_i)` and is
        undefined at zero (the engine returns ``0`` there).

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([-1.0, 2.0, -3.0])
        >>> abs(a)
        Tensor([1., 2., 3.])
        """
        return _wrap(_C_engine.abs(self._impl))

    def __invert__(self: Tensor) -> Tensor:
        r"""Bitwise / logical NOT: ``~self``.

        For boolean tensors this is logical negation; for integer dtypes
        it is the standard two's-complement bitwise complement.  Float
        inputs are rejected by the engine.

        Returns
        -------
        Tensor
            Inverted tensor with the same shape and dtype as ``self``.

        Notes
        -----
        Math (boolean case):

        .. math::

            \text{out}_i = \neg\,\text{self}_i

        Math (integer case):

        .. math::

            \text{out}_i = \sim\!\text{self}_i = -\text{self}_i - 1

        Not differentiable; produced tensor never carries autograd
        information.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([True, False, True])
        >>> ~a
        Tensor([False, True, False])
        """
        return _wrap(_C_engine.invert(self._impl))

    def __and__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise bitwise / logical AND: ``self & other``.

        For boolean tensors this is logical AND; for integer dtypes it is
        bitwise AND.  Float dtypes are rejected.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; must share dtype kind (bool or integer)
            with ``self``.

        Returns
        -------
        Tensor
            AND-combined tensor with broadcast shape.

        Notes
        -----
        Math (boolean):

        .. math::

            \text{out}_i = \text{self}_i \wedge \text{other}_i

        Math (integer): bitwise AND across the two operands' binary
        representations.  Not differentiable.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([True, True, False, False])
        >>> b = lucid.tensor([True, False, True, False])
        >>> a & b
        Tensor([True, False, False, False])
        """
        return _wrap(
            _C_engine.bitwise_and(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __or__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise bitwise / logical OR: ``self | other``.

        For boolean tensors this is logical OR; for integer dtypes it is
        bitwise OR.  Float dtypes are rejected.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; must share dtype kind with ``self``.

        Returns
        -------
        Tensor
            OR-combined tensor with broadcast shape.

        Notes
        -----
        Math (boolean):

        .. math::

            \text{out}_i = \text{self}_i \vee \text{other}_i

        Not differentiable.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([True, True, False, False])
        >>> b = lucid.tensor([True, False, True, False])
        >>> a | b
        Tensor([True, True, True, False])
        """
        return _wrap(
            _C_engine.bitwise_or(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __xor__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise bitwise / logical XOR: ``self ^ other``.

        For boolean tensors this is logical exclusive-or; for integer
        dtypes it is bitwise XOR.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; must share dtype kind with ``self``.

        Returns
        -------
        Tensor
            XOR-combined tensor with broadcast shape.

        Notes
        -----
        Math (boolean):

        .. math::

            \text{out}_i = \text{self}_i \oplus \text{other}_i

        Not differentiable.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([True, True, False, False])
        >>> b = lucid.tensor([True, False, True, False])
        >>> a ^ b
        Tensor([False, True, True, False])
        """
        return _wrap(
            _C_engine.bitwise_xor(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __eq__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise equality comparison: ``self == other``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; broadcast against ``self``.

        Returns
        -------
        Tensor
            Boolean tensor with broadcast shape; ``True`` where the
            corresponding elements match.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \mathbb{1}\!\left[\text{self}_i = \text{other}_i\right]

        Comparison with NaN follows IEEE 754 — ``NaN == NaN`` is always
        ``False``.  Not differentiable; the output tensor never carries
        gradient.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> b = lucid.tensor([1.0, 5.0, 3.0])
        >>> a == b
        Tensor([True, False, True])
        """
        return _wrap(_C_engine.equal(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __ne__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise inequality comparison: ``self != other``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; broadcast against ``self``.

        Returns
        -------
        Tensor
            Boolean tensor with broadcast shape.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \mathbb{1}\!\left[\text{self}_i \neq \text{other}_i\right]

        Under IEEE 754, ``NaN != NaN`` evaluates to ``True`` — this is the
        only comparison operator that is "true" against NaN.  Not
        differentiable.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> b = lucid.tensor([1.0, 5.0, 3.0])
        >>> a != b
        Tensor([False, True, False])
        """
        return _wrap(
            _C_engine.not_equal(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __lt__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise less-than comparison: ``self < other``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; broadcast against ``self``.

        Returns
        -------
        Tensor
            Boolean tensor with broadcast shape.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \mathbb{1}\!\left[\text{self}_i < \text{other}_i\right]

        Any comparison involving NaN returns ``False`` per IEEE 754.  Not
        differentiable.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> a < 2.5
        Tensor([True, True, False])
        """
        return _wrap(_C_engine.less(self._impl, _unwrap_or_scalar(other, self._impl)))

    def __le__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise less-than-or-equal comparison: ``self <= other``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; broadcast against ``self``.

        Returns
        -------
        Tensor
            Boolean tensor with broadcast shape.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \mathbb{1}\!\left[\text{self}_i \leq \text{other}_i\right]

        Any comparison involving NaN returns ``False`` per IEEE 754.  Not
        differentiable.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> a <= 2.0
        Tensor([True, True, False])
        """
        return _wrap(
            _C_engine.less_equal(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __gt__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise greater-than comparison: ``self > other``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; broadcast against ``self``.

        Returns
        -------
        Tensor
            Boolean tensor with broadcast shape.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \mathbb{1}\!\left[\text{self}_i > \text{other}_i\right]

        Any comparison involving NaN returns ``False`` per IEEE 754.  Not
        differentiable.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> a > 1.5
        Tensor([False, True, True])
        """
        return _wrap(
            _C_engine.greater(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __ge__(self: Tensor, other: TensorOrScalar) -> Tensor:
        r"""Element-wise greater-than-or-equal comparison: ``self >= other``.

        Parameters
        ----------
        other : Tensor or scalar
            Right-hand operand; broadcast against ``self``.

        Returns
        -------
        Tensor
            Boolean tensor with broadcast shape.

        Notes
        -----
        Math:

        .. math::

            \text{out}_i = \mathbb{1}\!\left[\text{self}_i \geq \text{other}_i\right]

        Any comparison involving NaN returns ``False`` per IEEE 754.  Not
        differentiable.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0])
        >>> a >= 2.0
        Tensor([False, True, True])
        """
        return _wrap(
            _C_engine.greater_equal(self._impl, _unwrap_or_scalar(other, self._impl))
        )

    def __getitem__(self: Tensor, idx: _IndexType) -> Tensor:
        r"""Tensor indexing: ``self[idx]``.

        Dispatches to :func:`lucid._tensor._indexing._getitem`, which
        supports both basic and advanced indexing semantics.

        Parameters
        ----------
        idx : int, slice, None, Ellipsis, Tensor, or tuple of these
            Index specification.  Supported atoms:

            * ``int`` — pick a single position along an axis (reduces rank
              by 1).
            * ``slice`` — Python ``start:stop:step`` window (preserves
              rank).
            * ``None`` — insert a new axis of length 1.
            * ``...`` (Ellipsis) — fill in remaining axes with full slices.
            * integer Tensor — gather along an axis (advanced indexing).
            * boolean Tensor — mask selection (advanced indexing).
            * tuple — combine the above across multiple axes.

        Returns
        -------
        Tensor
            View or gather result.  Basic indexing produces a view that
            shares storage with ``self``; advanced indexing copies.

        Notes
        -----
        Math (basic case, single axis):

        .. math::

            \text{out}_{j_1 \dots j_{k}} = \text{self}_{j_1 \dots j_{r-1} \, i \, j_{r+1} \dots}

        Advanced indexing follows NumPy's gather semantics: integer
        tensors are broadcast against one another and used to fetch
        elements; boolean masks select a flat 1-D subset along the masked
        axes.  Indexing participates in autograd via a sparse scatter in
        the backward pass.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        >>> a[0]
        Tensor([1., 2., 3.])
        >>> a[:, 1]
        Tensor([2., 5.])
        >>> a[a > 3]
        Tensor([4., 5., 6.])
        """
        return _getitem(self, idx)

    def __setitem__(self: Tensor, idx: _IndexType, value: TensorOrScalar) -> None:
        r"""In-place indexed assignment: ``self[idx] = value``.

        Dispatches to :func:`lucid._tensor._indexing._setitem`.  Supports
        the same index forms as :meth:`__getitem__`.

        Parameters
        ----------
        idx : int, slice, None, Ellipsis, Tensor, or tuple of these
            Index specification — see :meth:`__getitem__` for the full
            list.
        value : Tensor or scalar
            Right-hand value to assign.  Broadcast against the shape of
            the selected region and promoted to ``self.dtype``.

        Returns
        -------
        None
            The assignment mutates ``self`` in place.

        Notes
        -----
        Math (basic case):

        .. math::

            \text{self}_{j_1 \dots j_{r-1} \, i \, j_{r+1} \dots} \leftarrow \text{value}_{j_1 \dots}

        The mutated region is disconnected from the autograd graph for
        the previous values — subsequent backward passes will see only
        the new values.  Assigning into a leaf tensor with
        ``requires_grad=True`` raises because it would invalidate saved
        activations.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        >>> a[1:3] = 0
        >>> a
        Tensor([1., 0., 0., 4.])
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
