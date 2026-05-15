"""Predicates and zero-cost identity ops surfaced for reference-framework parity."""

from typing import TYPE_CHECKING

import lucid
from lucid._ops.composite._shared import _is_tensor
from lucid._types import TensorLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def numel(x: Tensor) -> int:
    r"""Return the total number of elements in a tensor.

    Equivalent to the product of the shape dimensions, returned as a
    plain Python ``int`` so it can be used in pure-Python control flow
    (loops, conditionals) without triggering autograd tracing.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    int
        Total number of scalar elements:
        :math:`\prod_{d} \text{shape}[d]`.

    Notes
    -----
    For a tensor of shape :math:`(s_0, s_1, \dots, s_{n-1})`:

    .. math::

        \text{numel}(x) = \prod_{i = 0}^{n - 1} s_i.

    A 0-D (scalar) tensor has ``numel == 1``; a tensor with any dimension
    of size ``0`` has ``numel == 0``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.zeros((2, 3, 4))
    >>> lucid.numel(x)
    24
    """
    return int(x.numel())


def is_storage(x: Tensor) -> bool:
    r"""Predicate: is the object a separate Storage container?

    Lucid does not expose a distinct Storage type — tensor data ownership
    is folded into the :class:`Tensor` class itself — so this predicate
    always returns ``False``. Provided for API parity with reference
    frameworks that maintain a Tensor/Storage split.

    Parameters
    ----------
    x : Tensor
        Object to test.

    Returns
    -------
    bool
        Always ``False`` in Lucid.

    Notes
    -----
    Lucid uses an arc-buffer model where every :class:`Tensor` directly
    owns its byte buffer (see the ``arch-storage-ownership`` ADR). There
    is therefore no separate Storage handle to query.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_storage(lucid.zeros(3))
    False
    """
    return False


def is_nonzero(x: Tensor) -> bool:
    r"""Predicate: is a single-element tensor non-zero?

    Reduces a scalar (one-element) tensor to a Python ``bool``. Raises if
    the tensor has more than one element so that the caller is forced to
    pick an explicit reduction (e.g. ``.any()`` / ``.all()``) for
    multi-element inputs.

    Parameters
    ----------
    x : Tensor
        Single-element tensor (i.e. ``x.numel() == 1``).

    Returns
    -------
    bool
        ``True`` iff the (sole) element is non-zero.

    Raises
    ------
    RuntimeError
        If ``x.numel() != 1``.

    Notes
    -----
    Equivalent to:

    .. math::

        \text{out} = (x_0 \neq 0)

    where :math:`x_0` is the single scalar value extracted via
    :meth:`Tensor.item`. The strict shape check prevents the silent
    bug where one would accidentally call ``bool(tensor)`` on an
    unintentionally multi-element tensor.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_nonzero(lucid.tensor(0.0))
    False
    >>> lucid.is_nonzero(lucid.tensor(3.5))
    True
    """
    if x.numel() != 1:
        raise RuntimeError("is_nonzero is defined only for scalar tensors (numel == 1)")
    return bool(x.item() != 0)


def is_same_size(a: Tensor, b: Tensor) -> bool:
    r"""Predicate: do two tensors have identical shapes?

    Compares the shape tuples exactly — no broadcasting compatibility is
    implied. Use :func:`lucid.broadcast_shapes` if broadcastability is
    the question.

    Parameters
    ----------
    a : Tensor
        First tensor.
    b : Tensor
        Second tensor.

    Returns
    -------
    bool
        ``True`` iff ``tuple(a.shape) == tuple(b.shape)``.

    Notes
    -----
    Compares all dimensions including singleton axes:

    .. math::

        \text{out} = \bigwedge_{i} (\text{shape}_a[i] = \text{shape}_b[i]).

    A ``(1, 3)`` tensor is not the same size as a ``(3,)`` tensor, even
    though they broadcast.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.zeros(2, 3)
    >>> b = lucid.ones(2, 3)
    >>> lucid.is_same_size(a, b)
    True
    """
    return tuple(a.shape) == tuple(b.shape)


def is_neg(x: Tensor) -> bool:
    r"""Predicate: does the tensor carry a lazy "negated" flag?

    Some reference frameworks track negation as a lazy view flag that is
    materialised on access. Lucid eagerly materialises every operation,
    so this predicate is always ``False``.

    Parameters
    ----------
    x : Tensor
        Tensor to query.

    Returns
    -------
    bool
        Always ``False``.

    Notes
    -----
    Lucid's tensor model has no lazy-flag bits — negation is realised by
    the engine ``neg`` op which produces a fresh buffer (or, under
    fusion, an in-pipeline transformation that still yields concrete
    values). There is therefore nothing to query.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_neg(-lucid.tensor([1.0, 2.0]))
    False
    """
    return False


def is_conj(x: Tensor) -> bool:
    r"""Predicate: does the tensor carry a lazy "conjugated" flag?

    Some reference frameworks track complex conjugation as a lazy view
    flag. Lucid materialises every operation eagerly, so this predicate
    is always ``False``.

    Parameters
    ----------
    x : Tensor
        Tensor to query.

    Returns
    -------
    bool
        Always ``False``.

    Notes
    -----
    See :func:`is_neg` for the analogous discussion on lazy flag bits.
    Conjugation in Lucid is performed by :func:`lucid.conj`, which
    produces a materialised result rather than toggling a flag.

    Examples
    --------
    >>> import lucid
    >>> lucid.is_conj(lucid.tensor([1.0, 2.0]))
    False
    """
    return False


def isin(
    elements: Tensor | TensorLike,
    test_elements: Tensor | TensorLike,
    *,
    invert: bool = False,
) -> Tensor:
    r"""Per-element set-membership test.

    For each entry of ``elements``, checks whether the value appears
    anywhere in ``test_elements``. Implemented as a broadcasted
    equality compare followed by an OR reduction.

    Parameters
    ----------
    elements : Tensor | TensorLike
        Values to test.  Reshaped to 1-D internally; the output is
        reshaped back to ``elements.shape``.
    test_elements : Tensor | TensorLike
        The set of values to test against. Flattened internally.
    invert : bool, optional
        If ``True``, return the negation of the membership test.
        Defaults to ``False``.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``elements``.

    Notes
    -----
    Mathematical definition (with :math:`E = \text{elements}` and
    :math:`T = \text{test\_elements}`):

    .. math::

        \text{out}_i = \bigvee_{j} (E_i = T_j).

    Cost is :math:`\mathcal{O}(|E| \cdot |T|)`. For large ``test_elements``
    a sort-based implementation would be cheaper, but this composite
    form keeps the operation differentiable-by-default (the gradient
    is zero everywhere — the output dtype is boolean — but the structure
    is preserved).

    Examples
    --------
    >>> import lucid
    >>> e = lucid.tensor([1, 2, 3, 4])
    >>> t = lucid.tensor([2, 4])
    >>> lucid.isin(e, t)
    Tensor([False,  True, False,  True])
    """
    if not _is_tensor(elements):
        elements = lucid.tensor(elements)
    if not _is_tensor(test_elements):
        test_elements = lucid.tensor(test_elements)
    e_flat = elements.reshape(-1)
    t_flat = test_elements.reshape(-1)
    n, m = e_flat.shape[0], t_flat.shape[0]
    e_bc = e_flat.unsqueeze(1).expand(n, m)
    t_bc = t_flat.unsqueeze(0).expand(n, m)
    matches = (e_bc == t_bc).to(dtype=lucid.float32).sum(1)
    out = (matches > 0.0).reshape(elements.shape)
    return ~out if invert else out


def isneginf(x: Tensor) -> Tensor:
    r"""Element-wise test for negative infinity.

    Identifies entries that are exactly :math:`-\infty` in the IEEE 754
    floating-point sense. Finite values and :math:`+\infty` both report
    ``False``.

    Parameters
    ----------
    x : Tensor
        Floating-point input tensor.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``x``.

    Notes
    -----
    Defined as the conjunction of "is infinite" and "is negative":

    .. math::

        \text{out}_i =
        \operatorname{isinf}(x_i) \wedge (x_i < 0).

    NaN inputs report ``False`` because NaN compares false against any
    real value, including in ``isinf``.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([-math.inf, -1.0, 0.0, math.inf])
    >>> lucid.isneginf(x)
    Tensor([ True, False, False, False])
    """
    return lucid.logical_and(lucid.isinf(x), x < 0.0)


def isposinf(x: Tensor) -> Tensor:
    r"""Element-wise test for positive infinity.

    Identifies entries that are exactly :math:`+\infty` in the IEEE 754
    floating-point sense. Finite values and :math:`-\infty` both report
    ``False``.

    Parameters
    ----------
    x : Tensor
        Floating-point input tensor.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``x``.

    Notes
    -----
    Defined as the conjunction of "is infinite" and "is positive":

    .. math::

        \text{out}_i =
        \operatorname{isinf}(x_i) \wedge (x_i > 0).

    NaN inputs report ``False``.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> x = lucid.tensor([-math.inf, -1.0, 0.0, math.inf])
    >>> lucid.isposinf(x)
    Tensor([False, False, False,  True])
    """
    return lucid.logical_and(lucid.isinf(x), x > 0.0)


def isreal(x: Tensor) -> Tensor:
    r"""Element-wise test for real-valued entries.

    Lucid currently supports only real-valued floating dtypes (and
    complex tensors via the dedicated ``complex64`` dtype), so for real
    inputs this predicate is uniformly ``True``. Provided for API parity
    with reference frameworks that include both real and complex dtypes.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``x``; every entry is
        ``True`` for real-valued inputs.

    Notes
    -----
    Mathematical definition (for real dtypes):

    .. math::

        \text{out}_i = \text{True}.

    For complex inputs (when supported), ``isreal`` reports ``True``
    only where the imaginary part is exactly zero. The current
    implementation returns ``isfinite | isnan | isinf``, which is a
    tautology for real dtypes.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, float('nan'), float('inf')])
    >>> lucid.isreal(x)
    Tensor([ True,  True,  True])
    """
    return lucid.isfinite(x) | lucid.isnan(x) | lucid.isinf(x)


def conj_physical(x: Tensor) -> Tensor:
    r"""Eagerly materialised complex conjugate.

    Some reference frameworks distinguish ``conj`` (which may set a lazy
    "conjugated" flag on the tensor's view) from ``conj_physical`` (which
    always writes a new buffer with the conjugated values).  Lucid does
    not carry any lazy conjugation metadata, so both call into the same
    engine op — ``conj_physical`` is provided for API parity and as a
    clear signal to readers that the conjugation is materialised.

    Parameters
    ----------
    x : Tensor
        Input tensor.  Typically ``complex64``; for real dtypes the
        conjugate is the identity.

    Returns
    -------
    Tensor
        Element-wise complex conjugate of ``x``, same shape and dtype.

    Notes
    -----
    Mathematical definition for a complex element :math:`z = a + ib`:

    .. math::

        \overline{z} = a - i\,b.

    Use this spelling when porting code that explicitly depends on the
    conjugation being physically present in the storage (e.g. for
    interop with external libraries that would otherwise see the lazy
    flag).  Functionally identical to :func:`lucid.conj` in Lucid.

    Examples
    --------
    >>> import lucid
    >>> z = lucid.tensor([1+2j, 3-4j])
    >>> lucid.conj_physical(z)
    Tensor([1.-2.j, 3.+4.j])
    """
    return lucid.conj(x)


def resolve_conj(x: Tensor) -> Tensor:
    r"""Materialise any pending lazy conjugation — no-op in Lucid.

    Reference frameworks that maintain a lazy "is_conj" flag on complex
    tensors use ``resolve_conj`` to force the flag to be applied to the
    underlying buffer.  Lucid has no such flag (see
    :func:`conj_physical`), so this routine simply returns the input
    unchanged.  It is provided for API parity so that ported code keeps
    working without behavioural drift.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        The input ``x`` returned unchanged (no copy, no autograd op
        inserted).

    Notes
    -----
    Equivalent to the identity map:

    .. math::

        \text{resolve\_conj}(x) = x.

    See :func:`conj_physical` for an explicit, always-materialised
    conjugation.

    Examples
    --------
    >>> import lucid
    >>> z = lucid.tensor([1+2j, 3-4j])
    >>> lucid.resolve_conj(z) is z
    True
    """
    return x


def resolve_neg(x: Tensor) -> Tensor:
    r"""Materialise any pending lazy negation — no-op in Lucid.

    Reference frameworks that maintain a lazy-negation flag use this op
    to force the flag to be applied to the underlying buffer. Lucid has
    no such flag (see :func:`is_neg`), so this returns the input
    unchanged.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        The input ``x``, returned unchanged.

    Notes
    -----
    Equivalent to the identity map:

    .. math::

        \text{out} = x.

    Provided strictly for API parity. Calling this op is free — no copy
    is made, no autograd node is inserted.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0])
    >>> lucid.resolve_neg(x) is x
    True
    """
    return x


def allclose(
    a: Tensor,
    b: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> bool:
    """Return True if all ``|a - b| <= atol + rtol * |b|`` element-wise."""
    diff = lucid.abs(a - b)
    tol = atol + rtol * lucid.abs(b)
    close = diff <= tol
    if equal_nan:
        both_nan = lucid.isnan(a) & lucid.isnan(b)
        close = close | both_nan
    return bool(close.all().item())


__all__ = [
    "numel",
    "is_storage",
    "is_nonzero",
    "is_same_size",
    "is_neg",
    "is_conj",
    "isin",
    "isneginf",
    "isposinf",
    "isreal",
    "conj_physical",
    "resolve_conj",
    "resolve_neg",
    "allclose",
]
