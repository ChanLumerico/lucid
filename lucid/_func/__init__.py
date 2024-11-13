from typing import Any
import numpy as np

from lucid._func import bfunc, gfunc, ufunc
from lucid._tensor import Tensor
from lucid.types import _Scalar, _ShapeLike, _ArrayLike


def minimum(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the element-wise minimum of two tensors.

    Given two tensors :math:`a` and :math:`b`, this function returns a tensor
    containing the minimum value at each position.

    **Forward Calculation**:

    .. math::
       \text{output}_{i,j} = \min(a_{i,j}, b_{i,j})

    **Backward Calculation**:
    Gradient flows only through the minimum value at each position.

    Parameters
    ----------
    a : Tensor
        The first input tensor.
    b : Tensor
        The second input tensor.

    Returns
    -------
    Tensor
        A tensor with the element-wise minimum values of `a` and `b`.
    """
    return bfunc.minimum(a, b)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the element-wise maximum of two tensors.

    **Forward Calculation**:
    .. math::
        \text{output}_{i,j} = \max(a_{i,j}, b_{i,j})

    **Backward Calculation**:
    Gradient flows only through the maximum value at each position.

    Parameters
    ----------
    a : Tensor
        The first input tensor.
    b : Tensor
        The second input tensor.

    Returns
    -------
    Tensor
        A tensor with the element-wise maximum values of `a` and `b`.
    """
    return bfunc.maximum(a, b)


def power(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the element-wise power of one tensor raised to the other.

    **Forward Calculation**:
    .. math::
        \text{output}_{i,j} = a_{i,j}^{b_{i,j}}

    **Backward Calculation**:
    The gradient of the power operation is calculated using:
    .. math::
        \frac{\partial \text{output}_{i,j}}{\partial a_{i,j}} = b_{i,j} \cdot a_{i,j}^{(b_{i,j}-1)}
        \quad \text{and} \quad
        \frac{\partial \text{output}_{i,j}}{\partial b_{i,j}} = a_{i,j}^{b_{i,j}} \cdot \ln(a_{i,j})

    Parameters
    ----------
    a : Tensor
        The base tensor.
    b : Tensor
        The exponent tensor.

    Returns
    -------
    Tensor
        A tensor where each element is `a` raised to the power of `b`.
    """
    return bfunc.power(a, b)


def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes the dot product of two tensors.

    **Forward Calculation**:
    .. math::
        \text{output} = \sum_{k} a_{i,k} \cdot b_{k,j}

    **Backward Calculation**:
    Gradients for `a` and `b` are computed as:
    .. math::
        \frac{\partial \text{output}}{\partial a_{i,k}} = b_{k,j}
        \quad \text{and} \quad
        \frac{\partial \text{output}}{\partial b_{k,j}} = a_{i,k}

    Parameters
    ----------
    a : Tensor
        The first tensor (left operand).
    b : Tensor
        The second tensor (right operand).

    Returns
    -------
    Tensor
        The result of the dot product of `a` and `b`.
    """
    return bfunc.dot(a, b)


def inner(a: Tensor, b: Tensor) -> Tensor:
    """Inner product of two tensors."""
    return bfunc.inner(a, b)


def exp(a: Tensor) -> Tensor:
    """
    Computes the element-wise exponential of the tensor.

    **Forward Calculation**:
    .. math::
        \text{output}_{i,j} = e^{a_{i,j}}

    **Backward Calculation**:
    .. math::
        \frac{\partial \text{output}_{i,j}}{\partial a_{i,j}} = e^{a_{i,j}}

    Parameters
    ----------
    a : Tensor
        The input tensor.

    Returns
    -------
    Tensor
        A tensor where each element is the exponential of the corresponding element in `a`.
    """
    return ufunc.exp(a)


def log(a: Tensor) -> Tensor:
    """Natural logarithm"""
    return ufunc.log(a)


def sqrt(a: Tensor) -> Tensor:
    """Square root"""
    return ufunc.sqrt(a)


def sin(a: Tensor) -> Tensor:
    """Sine function"""
    return ufunc.sin(a)


def cos(a: Tensor) -> Tensor:
    """Cosine function"""
    return ufunc.cos(a)


def tan(a: Tensor) -> Tensor:
    """Tangent function"""
    return ufunc.tan(a)


def arcsin(a: Tensor) -> Tensor:
    """Arcsin function"""
    return ufunc.arcsin(a)


def arccos(a: Tensor) -> Tensor:
    """Arccos function"""
    return ufunc.arccos(a)


def arctan(a: Tensor) -> Tensor:
    """Arctan function"""
    return ufunc.arctan(a)


def sinh(a: Tensor) -> Tensor:
    """Hyperbolic sine function"""
    return ufunc.sinh(a)


def cosh(a: Tensor) -> Tensor:
    """Hyperbolic cosine function"""
    return ufunc.cosh(a)


def tanh(a: Tensor) -> Tensor:
    """Hyperbolic tangent function"""
    return ufunc.tanh(a)


def clip(a: Tensor, min_value: _Scalar, max_value: _Scalar) -> Tensor:
    """Clips the values of the tensor to a specified range element-wise."""
    return ufunc.clip(a, min_value, max_value)


def abs(a: Tensor) -> Tensor:
    """Element-wise absolute value."""
    return ufunc.abs(a)


def sign(a: Tensor) -> Tensor:
    """Element-wise sign function."""
    return ufunc.sign(a)


def reciprocal(a: Tensor) -> Tensor:
    """Element-wise reciprocal."""
    return ufunc.reciprocal(a)


def square(a: Tensor) -> Tensor:
    """Element-wise square."""
    return ufunc.square(a)


def cube(a: Tensor) -> Tensor:
    """Element-wise cube."""
    return ufunc.cube(a)


def transpose(a: Tensor, axes: list[int] | None = None) -> Tensor:
    """Transpose over axes."""
    return ufunc.transpose(a, axes)


def sum(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    """Sum along the specified axis."""
    return ufunc.sum(a, axis, keepdims)


def trace(a: Tensor) -> Tensor:
    """Trace operation."""
    return ufunc.trace(a)


def mean(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    """
    Computes the mean of the tensor along the specified axis.

    **Forward Calculation**:
    .. math::
        \text{output} = \frac{1}{N} \sum_{i=1}^{N} a_i

    where :math:`N` is the number of elements along the specified axis.

    **Backward Calculation**:
    The gradient is distributed evenly to each element:
    .. math::
        \frac{\partial \text{output}}{\partial a_i} = \frac{1}{N}

    Parameters
    ----------
    a : Tensor
        The input tensor.
    axis : int or tuple of int, optional
        The axis or axes along which to compute the mean.
    keepdims : bool, optional
        Whether to retain the reduced dimensions in the output tensor.

    Returns
    -------
    Tensor
        The mean of `a` along the specified axis.
    """
    return ufunc.mean(a, axis, keepdims)


def var(
    a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False
) -> Tensor:
    """Compute the variance along the specified axis; if `axis=None`,
    variance of the entire tensor is returned."""
    return ufunc.var(a, axis, keepdims)


def zeros(
    shape: _ShapeLike,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create a zero-tensor with the specified shape."""
    return gfunc.zeros(shape, dtype, requires_grad, keep_grad)


def zeros_like(
    a: Tensor | _ArrayLike,
    dtype: Any = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create a zero-tensor of shape same with the given tensor."""
    return zeros_like(a, dtype, requires_grad, keep_grad)


def ones(
    shape: _ShapeLike,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create an one-tensor with the specified shape."""
    return gfunc.ones(shape, dtype, requires_grad, keep_grad)


def ones_like(
    a: Tensor | _ArrayLike,
    dtype: Any = None,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create an one-tensor of shape same with the given tensor."""
    return gfunc.ones_like(a, dtype, requires_grad, keep_grad)


def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create an identical matrix of shape `(N, M)`."""
    return gfunc.eye(N, M, k, dtype, requires_grad, keep_grad)


def diag(
    v: Tensor | _ArrayLike,
    k: int = 0,
    dtype: Any = np.float32,
    requires_grad: bool = False,
    keep_grad: bool = False,
) -> Tensor:
    """Create a diagonal matrix from the given vector."""
    return gfunc.diag(v, k, dtype, requires_grad, keep_grad)


Tensor.__add__ = bfunc._add
Tensor.__radd__ = bfunc._radd
Tensor.__sub__ = bfunc._sub
Tensor.__rsub__ = bfunc._rsub
Tensor.__mul__ = bfunc._mul
Tensor.__rmul__ = bfunc._rmul
Tensor.__truediv__ = bfunc._truediv
Tensor.__rtruediv__ = bfunc._rtruediv
Tensor.__matmul__ = bfunc.matmul

Tensor.__eq__ = bfunc._equal
Tensor.__ne__ = bfunc._not_equal
Tensor.__gt__ = bfunc._greater
Tensor.__ge__ = bfunc._greater_or_equal
Tensor.__lt__ = bfunc._less
Tensor.__le__ = bfunc._less_or_equal

Tensor.__pow__ = ufunc._pow
Tensor.__neg__ = ufunc._neg

Tensor.T = ufunc._T
Tensor.dot = bfunc.dot
Tensor.sum = ufunc.sum
Tensor.mean = ufunc.mean
Tensor.var = ufunc.var
