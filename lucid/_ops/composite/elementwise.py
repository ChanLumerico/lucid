"""Element-wise composite ops layered on the engine.

Three subgroups:

1. **Aliases** for engine ops the reference framework exposes under multiple names
   (``absolute`` ↔ ``abs``, ``subtract`` ↔ ``sub``, etc.).
2. **Inverse-hyperbolic** functions composed from ``log`` and ``sqrt``.
3. **Specials** — ``expm1``, ``sinc``, ``heaviside``, ``xlogy``, ``logit``,
   ``signbit``, ``float_power``, ``fmax`` / ``fmin`` — each implemented as
   a small expression over engine primitives, so autograd works without
   any extra wiring.
"""

import math
import math as _math
from typing import TYPE_CHECKING, cast

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops.composite._shared import _is_tensor
from lucid._types import Scalar

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Aliases ────────────────────────────────────────────────────────────────


def absolute(x: Tensor) -> Tensor:
    r"""Element-wise absolute value.

    Verbose alias of :func:`lucid.abs`, exposed for parity with reference
    frameworks that publish both ``abs`` and ``absolute`` spellings of the
    same operation.

    Parameters
    ----------
    x : Tensor
        Input tensor. Any numeric dtype.

    Returns
    -------
    Tensor
        Tensor of the same shape and dtype as ``x`` containing
        :math:`|x_i|` element-wise.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = |x_i|

    The gradient is :math:`\operatorname{sign}(x_i)`; the sub-gradient at
    :math:`x_i = 0` is conventionally taken to be ``0``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.5, 0.0, 2.0])
    >>> lucid.absolute(x)
    Tensor([1.5, 0., 2.])
    """
    return lucid.abs(x)


def negative(x: Tensor) -> Tensor:
    r"""Element-wise negation.

    Verbose alias of :func:`lucid.neg`. Equivalent to the unary ``-`` operator
    on a tensor.

    Parameters
    ----------
    x : Tensor
        Input tensor. Any numeric dtype.

    Returns
    -------
    Tensor
        Tensor of the same shape and dtype as ``x`` whose entries are
        :math:`-x_i`.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = -x_i

    The gradient with respect to ``x`` is :math:`-1` everywhere.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, -2.0, 3.5])
    >>> lucid.negative(x)
    Tensor([-1.,  2., -3.5])
    """
    return lucid.neg(x)


def positive(x: Tensor) -> Tensor:
    r"""Element-wise identity — returns the input unchanged.

    Provided as the unary-``+`` counterpart of :func:`negative`, matching
    the convention that ``+x`` is a no-op on numeric data.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        The input ``x``, returned without modification (no copy, no
        autograd op inserted).

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = +x_i = x_i

    Useful only for symmetry with frameworks that treat ``+x`` as a unary
    operator.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, -2.0])
    >>> lucid.positive(x) is x
    True
    """
    return x


def subtract(a: Tensor, b: Tensor | Scalar, *, alpha: float = 1.0) -> Tensor:
    r"""Element-wise subtraction with an optional scalar multiplier.

    Computes :math:`a - \alpha \cdot b` element-wise, broadcasting ``a``
    and ``b`` to a common shape and following the standard dtype-promotion
    rules.

    Parameters
    ----------
    a : Tensor
        Minuend.
    b : Tensor | Scalar
        Subtrahend. Python scalars are broadcast against ``a``.
    alpha : float, optional
        Scalar multiplier applied to ``b`` before subtraction.
        Defaults to ``1.0`` (plain subtraction).

    Returns
    -------
    Tensor
        Element-wise difference of the broadcast shape of ``a`` and ``b``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = a_i - \alpha \cdot b_i

    Dtype follows standard promotion: e.g. ``int64 - float32 → float32``.
    Gradients flow as :math:`\partial / \partial a = 1` and
    :math:`\partial / \partial b = -\alpha`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([4.0, 5.0])
    >>> b = lucid.tensor([1.0, 2.0])
    >>> lucid.subtract(a, b, alpha=2.0)
    Tensor([2., 1.])
    """
    if alpha == 1.0:
        return a - b
    return a - (b * alpha)


def multiply(a: Tensor, b: Tensor | Scalar) -> Tensor:
    r"""Element-wise product.

    Verbose alias of the ``*`` operator. Broadcasts ``a`` and ``b`` and
    applies the standard dtype-promotion rules.

    Parameters
    ----------
    a : Tensor
        First operand.
    b : Tensor | Scalar
        Second operand.

    Returns
    -------
    Tensor
        Element-wise product of the broadcast shape of ``a`` and ``b``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = a_i \cdot b_i

    Gradients: :math:`\partial / \partial a = b`,
    :math:`\partial / \partial b = a`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> lucid.multiply(a, 2.0)
    Tensor([2., 4., 6.])
    """
    return a * b


def divide(a: Tensor, b: Tensor | Scalar) -> Tensor:
    r"""Element-wise true division.

    Verbose alias of the ``/`` operator. Always performs floating-point
    division — integer operands are promoted to a floating dtype before
    the division so that the result preserves fractional parts.

    Parameters
    ----------
    a : Tensor
        Numerator.
    b : Tensor | Scalar
        Denominator.

    Returns
    -------
    Tensor
        Element-wise quotient of the broadcast shape of ``a`` and ``b``,
        in a floating-point dtype.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = \frac{a_i}{b_i}

    Division by zero produces :math:`\pm\infty` (or NaN for ``0/0``)
    rather than raising. Gradients are
    :math:`\partial / \partial a = 1/b` and
    :math:`\partial / \partial b = -a / b^{2}`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([6.0, 9.0])
    >>> lucid.divide(a, 3.0)
    Tensor([2., 3.])
    """
    return a / b


def true_divide(a: Tensor, b: Tensor | Scalar) -> Tensor:
    r"""Element-wise true (floating-point) division.

    Equivalent to :func:`divide`; named ``true_divide`` for parity with
    NumPy / reference-framework spellings that distinguish it from
    integer-flavored ``floor_divide``.

    Parameters
    ----------
    a : Tensor
        Numerator.
    b : Tensor | Scalar
        Denominator.

    Returns
    -------
    Tensor
        Element-wise quotient in a floating-point dtype.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = \frac{a_i}{b_i}

    Even for integer inputs, the output dtype is a floating type — this is
    the defining property that separates ``true_divide`` from
    ``floor_divide``.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([7, 8], dtype=lucid.int32)
    >>> lucid.true_divide(a, 2)
    Tensor([3.5, 4.])
    """
    return a / b


def rsub(a: Tensor, b: Tensor | Scalar, *, alpha: float = 1.0) -> Tensor:
    r"""Reverse subtraction: :math:`b - \alpha \cdot a`.

    Useful when ``b`` is a Python scalar that cannot left-bind the ``-``
    operator against a tensor in a particular calling convention.

    Parameters
    ----------
    a : Tensor
        Right operand (subtrahend, multiplied by ``alpha``).
    b : Tensor | Scalar
        Left operand (minuend).
    alpha : float, optional
        Scalar multiplier applied to ``a``. Defaults to ``1.0``.

    Returns
    -------
    Tensor
        Element-wise difference of the broadcast shape of ``a`` and ``b``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = b_i - \alpha \cdot a_i

    Equivalent to :func:`subtract` with the operands reversed (and ``alpha``
    re-attached to the new second argument). Gradient w.r.t. ``a`` is
    :math:`-\alpha`; w.r.t. ``b`` is :math:`1`.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0])
    >>> lucid.rsub(a, 10.0)
    Tensor([9., 8.])
    """
    if alpha == 1.0:
        return b - a
    return b - (a * alpha)


def arctan2(y: Tensor, x: Tensor) -> Tensor:
    r"""Quadrant-correct two-argument arctangent.

    Verbose alias of :func:`lucid.atan2`.  Returns the polar angle
    :math:`\theta` of the point :math:`(x, y)` in the Cartesian plane,
    resolving the correct quadrant from the signs of both arguments.

    Parameters
    ----------
    y : Tensor
        Ordinate (imaginary / y-coordinate). Must broadcast with ``x``.
    x : Tensor
        Abscissa (real / x-coordinate). Must broadcast with ``y``.

    Returns
    -------
    Tensor
        Element-wise angle in radians on :math:`[-\pi, \pi]`, with the
        broadcast shape of ``y`` and ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \arctan2(y, x) =
        \begin{cases}
            \arctan(y/x),          & x > 0, \\
            \arctan(y/x) + \pi,    & x < 0,\ y \geq 0, \\
            \arctan(y/x) - \pi,    & x < 0,\ y < 0, \\
            +\pi/2,                & x = 0,\ y > 0, \\
            -\pi/2,                & x = 0,\ y < 0, \\
            0,                     & x = 0,\ y = 0.
        \end{cases}

    The naive :math:`\arctan(y / x)` collapses both half-planes into
    :math:`(-\pi/2, \pi/2)`; ``arctan2`` instead inspects the sign of
    each argument to recover the full :math:`[-\pi, \pi]` range, which
    is essential for converting Cartesian to polar coordinates.

    Examples
    --------
    >>> import lucid
    >>> y = lucid.tensor([1.0,  1.0, -1.0, -1.0])
    >>> x = lucid.tensor([1.0, -1.0, -1.0,  1.0])
    >>> lucid.arctan2(y, x)
    Tensor([ 0.7854,  2.3562, -2.3562, -0.7854])
    """
    return lucid.atan2(y, x)


# ── Inverse hyperbolic (composed) ──────────────────────────────────────────


def arccosh(x: Tensor) -> Tensor:
    r"""Inverse hyperbolic cosine.

    Verbose alias of :func:`acosh`.  Computes the principal branch
    :math:`\operatorname{arccosh}(x) = \log\!\bigl(x + \sqrt{x^{2} - 1}\bigr)`,
    defined for real :math:`x \geq 1`.

    Parameters
    ----------
    x : Tensor
        Input tensor; element-wise values must satisfy :math:`x \geq 1`
        for a real result.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic cosine, same shape as ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{arccosh}(x) =
        \log\!\bigl(x + \sqrt{x^{2} - 1}\bigr), \qquad x \geq 1.

    Inputs with :math:`x < 1` produce NaN through the real ``sqrt``
    branch.  The derivative is :math:`1 / \sqrt{x^{2} - 1}`, singular
    at :math:`x = 1`.  Both ``arccosh`` and :func:`acosh` refer to the
    same composite; either spelling may be used for parity with NumPy
    or reference-framework code.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 10.0])
    >>> lucid.arccosh(x)
    Tensor([0.    , 1.3170, 2.9932])
    """
    return lucid.log(x + lucid.sqrt(x * x - 1.0))


def acosh(x: Tensor) -> Tensor:
    r"""Inverse hyperbolic cosine.

    Short-name alias of :func:`arccosh`. Computes the principal branch of
    :math:`\operatorname{arccosh}(x)` for real inputs.

    Parameters
    ----------
    x : Tensor
        Input tensor with values in the domain :math:`x \geq 1`.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic cosine.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{acosh}(x) = \log\!\bigl(x + \sqrt{x^{2} - 1}\bigr),
        \qquad x \geq 1.

    Inputs with :math:`x < 1` produce NaN. The derivative is
    :math:`1 / \sqrt{x^{2} - 1}`, which is singular at :math:`x = 1`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 10.0])
    >>> lucid.acosh(x)
    Tensor([0.    , 1.3170, 2.9932])
    """
    return arccosh(x)


def arcsinh(x: Tensor) -> Tensor:
    r"""Inverse hyperbolic sine.

    Verbose alias of :func:`asinh`.  Computes
    :math:`\operatorname{arcsinh}(x) = \log\!\bigl(x + \sqrt{x^{2} + 1}\bigr)`,
    defined for all real ``x``.

    Parameters
    ----------
    x : Tensor
        Input tensor; domain is all of :math:`\mathbb{R}`.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic sine, same shape as ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{arcsinh}(x) =
        \log\!\bigl(x + \sqrt{x^{2} + 1}\bigr).

    Odd function: :math:`\operatorname{arcsinh}(-x) = -\operatorname{arcsinh}(x)`.
    Derivative is :math:`1 / \sqrt{x^{2} + 1}`.  Both ``arcsinh`` and
    :func:`asinh` refer to the same composite.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> lucid.arcsinh(x)
    Tensor([-0.8814,  0.    ,  0.8814])
    """
    return lucid.log(x + lucid.sqrt(x * x + 1.0))


def asinh(x: Tensor) -> Tensor:
    r"""Inverse hyperbolic sine.

    Short-name alias of :func:`arcsinh`. Defined for all real inputs.

    Parameters
    ----------
    x : Tensor
        Input tensor. Domain is all of :math:`\mathbb{R}`.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic sine.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{asinh}(x) = \log\!\bigl(x + \sqrt{x^{2} + 1}\bigr).

    The function is odd: :math:`\operatorname{asinh}(-x) = -\operatorname{asinh}(x)`.
    Derivative is :math:`1 / \sqrt{x^{2} + 1}`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> lucid.asinh(x)
    Tensor([-0.8814,  0.    ,  0.8814])
    """
    return arcsinh(x)


def arctanh(x: Tensor) -> Tensor:
    r"""Inverse hyperbolic tangent.

    Verbose alias of :func:`atanh`.  Computes
    :math:`\operatorname{arctanh}(x) =
    \tfrac{1}{2}\log\!\bigl((1+x)/(1-x)\bigr)`, defined for
    :math:`|x| < 1`.

    Parameters
    ----------
    x : Tensor
        Input tensor; element-wise values must satisfy :math:`|x| < 1`
        for a finite real result.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic tangent, same shape as ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{arctanh}(x) =
        \tfrac{1}{2}\log\!\Bigl(\tfrac{1 + x}{1 - x}\Bigr),
        \qquad |x| < 1.

    Boundary values :math:`x = \pm 1` produce :math:`\pm\infty`;
    :math:`|x| > 1` produces NaN.  Derivative is :math:`1 / (1 - x^{2})`.
    Both ``arctanh`` and :func:`atanh` refer to the same composite.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-0.5, 0.0, 0.5])
    >>> lucid.arctanh(x)
    Tensor([-0.5493,  0.    ,  0.5493])
    """
    return lucid.log((1.0 + x) / (1.0 - x)) * 0.5


def atanh(x: Tensor) -> Tensor:
    r"""Inverse hyperbolic tangent.

    Short-name alias of :func:`arctanh`. Domain is the open interval
    :math:`(-1, 1)`; values at the boundary diverge to :math:`\pm\infty`.

    Parameters
    ----------
    x : Tensor
        Input tensor with values in :math:`|x| < 1`.

    Returns
    -------
    Tensor
        Element-wise inverse hyperbolic tangent.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{atanh}(x) = \tfrac{1}{2}\,
        \log\!\Bigl(\tfrac{1 + x}{1 - x}\Bigr),
        \qquad |x| < 1.

    Values outside :math:`(-1, 1)` produce NaN; :math:`x = \pm 1` produces
    :math:`\pm\infty`. The derivative is :math:`1 / (1 - x^{2})`.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-0.5, 0.0, 0.5])
    >>> lucid.atanh(x)
    Tensor([-0.5493,  0.    ,  0.5493])
    """
    return arctanh(x)


# ── Specials ───────────────────────────────────────────────────────────────


def expm1(x: Tensor) -> Tensor:
    r"""Element-wise :math:`e^{x} - 1`.

    Computes the exponential-minus-one function. The current composite
    implementation evaluates the naive form ``exp(x) - 1``; a dedicated
    engine primitive would additionally preserve relative accuracy for
    :math:`|x| \ll 1`, but the algebraic result is identical.

    Parameters
    ----------
    x : Tensor
        Input tensor. Any floating-point dtype.

    Returns
    -------
    Tensor
        Element-wise :math:`e^{x} - 1`.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = e^{x_i} - 1.

    The function is useful in probability and finance where :math:`x` is
    a small log-difference, since ``log(1 + y) = expm1⁻¹(y)`` and the pair
    avoids subtractive cancellation when chained.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([0.0, 1.0, 2.0])
    >>> lucid.expm1(x)
    Tensor([0.    , 1.71828, 6.38906])
    """
    return lucid.exp(x) - 1.0


def sinc(x: Tensor) -> Tensor:
    r"""Normalised sinc function with the standard removable singularity at 0.

    The "normalised" sinc carries a factor of :math:`\pi` inside the sine,
    so its zeros coincide with the non-zero integers — the convention used
    in signal processing.

    Parameters
    ----------
    x : Tensor
        Input tensor. Any floating-point dtype.

    Returns
    -------
    Tensor
        Element-wise normalised sinc with ``sinc(0) = 1``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{sinc}(x) =
        \begin{cases}
            \dfrac{\sin(\pi x)}{\pi x}, & x \neq 0, \\
            1,                          & x = 0.
        \end{cases}

    The composite uses :func:`lucid.where` to substitute the limit value
    at ``x = 0`` so the gradient flows cleanly through the removable
    singularity.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([0.0, 0.5, 1.0])
    >>> lucid.sinc(x)
    Tensor([1.    , 0.6366, 0.    ])
    """
    px = x * math.pi
    is_zero = x == 0.0
    safe_px = lucid.where(is_zero, lucid.full_like(x, 1.0), px)
    val = lucid.sin(safe_px) / safe_px
    return lucid.where(is_zero, lucid.full_like(x, 1.0), val)


def heaviside(x: Tensor, values: Tensor | Scalar) -> Tensor:
    r"""Heaviside step function with a user-selectable value at :math:`x = 0`.

    Produces ``0`` for negative inputs and ``1`` for positive inputs; the
    value at exactly zero is taken from ``values`` (typically ``0``,
    ``0.5``, or ``1`` depending on convention).

    Parameters
    ----------
    x : Tensor
        Input tensor.
    values : Tensor | Scalar
        Value(s) used wherever ``x == 0``. If a scalar, it is broadcast to
        the shape of ``x``; if a tensor, it must broadcast against ``x``.

    Returns
    -------
    Tensor
        Element-wise Heaviside step.

    Notes
    -----
    Mathematical definition:

    .. math::

        H(x_i) =
        \begin{cases}
            0,            & x_i < 0, \\
            \text{values}_i, & x_i = 0, \\
            1,            & x_i > 0.
        \end{cases}

    The gradient is zero almost everywhere — the derivative is a Dirac
    delta in the distributional sense, which autograd does not propagate.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> lucid.heaviside(x, 0.5)
    Tensor([0. , 0.5, 1. ])
    """
    if not _is_tensor(values):
        values = lucid.full_like(x, float(cast(float, values)))
    return lucid.where(
        x == 0.0,
        values,
        lucid.where(x > 0.0, lucid.full_like(x, 1.0), lucid.full_like(x, 0.0)),
    )


def xlogy(x: Tensor | Scalar, y: Tensor | Scalar) -> Tensor:
    r"""Compute :math:`x \log y` with the convention :math:`0 \cdot \log 0 = 0`.

    Frequently used to compute cross-entropy losses where the limit
    :math:`\lim_{x \to 0^{+}} x \log x = 0` must be honored to avoid NaN
    contamination from probability values that are exactly zero.

    Parameters
    ----------
    x : Tensor | Scalar
        Left operand (multiplier).
    y : Tensor | Scalar
        Argument of the logarithm. Must be non-negative for a real result.

    Returns
    -------
    Tensor
        Element-wise :math:`x \log y` with the zero-times-zero convention
        applied.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{xlogy}(x, y) =
        \begin{cases}
            0,            & x = 0, \\
            x \cdot \log y, & x \neq 0.
        \end{cases}

    Gradient with respect to ``y`` is :math:`x / y`; with respect to ``x``
    is :math:`\log y`. Both are masked to zero wherever ``x == 0``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([0.0, 1.0, 2.0])
    >>> y = lucid.tensor([0.0, 2.0, 3.0])
    >>> lucid.xlogy(x, y)
    Tensor([0.    , 0.6931, 2.1972])
    """
    if not _is_tensor(x):
        x = lucid.tensor(float(cast(float, x)))
    if not _is_tensor(y):
        y = lucid.tensor(float(cast(float, y)))
    safe_y = lucid.where(y == 0.0, lucid.full_like(y, 1.0), y)
    out = x * lucid.log(safe_y)
    return lucid.where(x == 0.0, lucid.full_like(out, 0.0), out)


def logit(x: Tensor, eps: float | None = None) -> Tensor:
    r"""Logit function — inverse of the logistic sigmoid.

    Maps a probability in :math:`(0, 1)` to a log-odds value in
    :math:`\mathbb{R}`. Optionally clamps the input to
    :math:`[\epsilon, 1 - \epsilon]` first to avoid divergence at the
    boundaries.

    Parameters
    ----------
    x : Tensor
        Input tensor with values in :math:`(0, 1)`.
    eps : float | None, optional
        If provided, ``x`` is clamped to ``[eps, 1 - eps]`` before the
        log-ratio is taken, preventing infinities. Defaults to ``None``
        (no clamping).

    Returns
    -------
    Tensor
        Element-wise logit value.

    Notes
    -----
    Mathematical definition:

    .. math::

        \operatorname{logit}(x) = \log\!\Bigl(\tfrac{x}{1 - x}\Bigr).

    Inverse of the sigmoid:
    :math:`\operatorname{logit}(\sigma(z)) = z`. Without ``eps``, inputs
    of exactly ``0`` or ``1`` produce :math:`-\infty` and :math:`+\infty`
    respectively.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([0.1, 0.5, 0.9])
    >>> lucid.logit(x)
    Tensor([-2.1972,  0.    ,  2.1972])
    """
    if eps is not None:
        x = lucid.clamp(x, eps, 1.0 - eps)
    return lucid.log(x / (1.0 - x))


def signbit(x: Tensor) -> Tensor:
    r"""Element-wise sign-bit predicate.

    Returns a boolean tensor that is ``True`` wherever ``x`` is strictly
    negative. The current implementation uses the strict inequality
    ``x < 0``, so ``-0.0`` reports ``False`` (i.e. the value is treated
    as zero rather than tested via IEEE 754 sign-bit inspection).

    Parameters
    ----------
    x : Tensor
        Input tensor. Any numeric dtype.

    Returns
    -------
    Tensor
        Boolean tensor of the same shape as ``x``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{signbit}(x_i) =
        \begin{cases}
            \text{True},  & x_i < 0, \\
            \text{False}, & x_i \geq 0.
        \end{cases}

    Non-differentiable: the gradient is zero almost everywhere and the
    output dtype is boolean.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([-1.0, 0.0, 1.0])
    >>> lucid.signbit(x)
    Tensor([ True, False, False])
    """
    return x < 0.0


def float_power(x: Tensor | Scalar, y: Tensor | Scalar) -> Tensor:
    r"""Element-wise power that always promotes operands to ``float64``.

    Unlike the regular :func:`lucid.pow`, ``float_power`` upcasts both
    operands to ``float64`` before exponentiation, so that negative bases
    raised to non-integer powers and very large exponents behave
    consistently without integer overflow or domain errors.

    Parameters
    ----------
    x : Tensor | Scalar
        Base.
    y : Tensor | Scalar
        Exponent.

    Returns
    -------
    Tensor
        Element-wise ``x ** y`` evaluated in ``float64``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = x_i^{y_i}.

    Promotion is mandatory — even integer inputs are cast to ``float64``.
    The result is always ``float64`` regardless of input dtype.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.tensor([2, 3], dtype=lucid.int32)
    >>> lucid.float_power(x, 0.5).dtype
    lucid.float64
    """
    if _is_tensor(x):
        x = x.to(dtype=lucid.float64)
    if _is_tensor(y):
        y = y.to(dtype=lucid.float64)
    if not _is_tensor(x) and _is_tensor(y):
        x = lucid.tensor(float(cast(float, x)), dtype=lucid.float64, device=y.device)
    if not _is_tensor(y) and _is_tensor(x):
        y = lucid.tensor(float(cast(float, y)), dtype=lucid.float64, device=x.device)
    return lucid.pow(cast(Tensor, x), cast(Tensor, y))


def fmax(a: Tensor, b: Tensor) -> Tensor:
    r"""Element-wise NaN-quiet maximum.

    Like :func:`lucid.maximum`, but treats NaN as missing data: if exactly
    one operand is NaN at a given position, the non-NaN value is returned.
    Only when both are NaN does the result remain NaN.

    Parameters
    ----------
    a : Tensor
        First operand.
    b : Tensor
        Second operand. Must broadcast with ``a``.

    Returns
    -------
    Tensor
        Element-wise NaN-quiet maximum.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{fmax}(a_i, b_i) =
        \begin{cases}
            b_i,           & a_i = \text{NaN}, \\
            a_i,           & b_i = \text{NaN}, \\
            \max(a_i, b_i),& \text{otherwise.}
        \end{cases}

    Contrast with :func:`lucid.maximum`, which propagates NaN per IEEE 754.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> a = lucid.tensor([1.0, math.nan, 3.0])
    >>> b = lucid.tensor([2.0, 2.0, math.nan])
    >>> lucid.fmax(a, b)
    Tensor([2., 2., 3.])
    """
    a_is_nan = lucid.isnan(a)
    b_is_nan = lucid.isnan(b)
    m = lucid.maximum(a, b)
    m = lucid.where(a_is_nan, b, m)
    m = lucid.where(b_is_nan, a, m)
    return m


def fmin(a: Tensor, b: Tensor) -> Tensor:
    r"""Element-wise NaN-quiet minimum.

    Like :func:`lucid.minimum`, but treats NaN as missing data: if exactly
    one operand is NaN at a given position, the non-NaN value is returned.
    Only when both are NaN does the result remain NaN.

    Parameters
    ----------
    a : Tensor
        First operand.
    b : Tensor
        Second operand. Must broadcast with ``a``.

    Returns
    -------
    Tensor
        Element-wise NaN-quiet minimum.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{fmin}(a_i, b_i) =
        \begin{cases}
            b_i,           & a_i = \text{NaN}, \\
            a_i,           & b_i = \text{NaN}, \\
            \min(a_i, b_i),& \text{otherwise.}
        \end{cases}

    Contrast with :func:`lucid.minimum`, which propagates NaN per IEEE 754.

    Examples
    --------
    >>> import lucid
    >>> import math
    >>> a = lucid.tensor([1.0, math.nan, 3.0])
    >>> b = lucid.tensor([2.0, 2.0, math.nan])
    >>> lucid.fmin(a, b)
    Tensor([1., 2., 3.])
    """
    a_is_nan = lucid.isnan(a)
    b_is_nan = lucid.isnan(b)
    m = lucid.minimum(a, b)
    m = lucid.where(a_is_nan, b, m)
    m = lucid.where(b_is_nan, a, m)
    return m


# ── Special math functions ─────────────────────────────────────────────────


def erfc(x: Tensor) -> Tensor:
    """Complementary error function: ``erfc(x) = 1 - erf(x)``."""
    return lucid.full_like(x, 1.0) - lucid.erf(x)


def copysign(x: Tensor, y: Tensor) -> Tensor:
    """Return a tensor with magnitudes from ``x`` and signs from ``y``."""
    return lucid.where(y < 0.0, -lucid.abs(x), lucid.abs(x))


def ldexp(input: Tensor, exponent: Tensor | Scalar) -> Tensor:
    """``input * 2 ** exponent`` element-wise (differentiable w.r.t. both)."""
    exp_t: Tensor = (
        exponent
        if _is_tensor(exponent)
        else lucid.tensor(cast(float, exponent), dtype=input.dtype, device=input.device)
    )
    return input * lucid.exp(exp_t * math.log(2.0))


def frexp(input: Tensor) -> tuple[Tensor, Tensor]:
    """Decompose ``input`` into mantissa ``m`` and exponent ``e`` such that
    ``input = m * 2**e`` with ``|m|`` in ``[0.5, 1)`` (or ``m == 0``).

    Returns a ``(mantissa, exponent)`` tuple.  Exponent dtype is int32.
    Non-finite inputs propagate: NaN → (NaN, 0); ±inf → (±inf, 0); 0 → (0, 0).
    """
    abs_x = lucid.abs(input)
    is_zero = abs_x == 0.0
    is_nonfinite = lucid.logical_or(lucid.isinf(input), lucid.isnan(input))
    safe_abs = lucid.where(
        lucid.logical_or(is_zero, is_nonfinite),
        lucid.full_like(input, 1.0),
        abs_x,
    )
    # e = floor(log2(|x|)) + 1 puts |m| in [0.5, 1).
    e_float = lucid.floor(lucid.log2(safe_abs)) + 1.0
    e_float = lucid.where(
        lucid.logical_or(is_zero, is_nonfinite),
        lucid.full_like(e_float, 0.0),
        e_float,
    )
    m = input * lucid.exp(-e_float * math.log(2.0))
    m = lucid.where(is_zero, lucid.full_like(m, 0.0), m)
    m = lucid.where(is_nonfinite, input, m)
    e_i32 = e_float.to(dtype=lucid.int32)
    return m, e_i32


# ── Integer math (non-differentiable) ────────────────────────────────────


def gcd(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise greatest common divisor (integer tensors)."""
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    n = int(flat_x.shape[0])
    result = [_math.gcd(int(flat_x[i].item()), int(flat_y[i].item())) for i in range(n)]
    return lucid.tensor(result, dtype=x.dtype, device=x.device).reshape(x.shape)


def lcm(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise least common multiple (integer tensors)."""
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    n = int(flat_x.shape[0])
    result = [_math.lcm(int(flat_x[i].item()), int(flat_y[i].item())) for i in range(n)]
    return lucid.tensor(result, dtype=x.dtype, device=x.device).reshape(x.shape)


# ── Log-gamma and Digamma ─────────────────────────────────────────────────
#
# Both are implemented as pure composites over engine primitives so that
# autograd flows through them automatically.
#
# lgamma: Lanczos approximation (Numerical Recipes, g=7, n=9).
#   Valid for x > 0.  Using the full Lanczos series keeps relative error
#   below 1.5e-15 for real x > 0 (float64).
#
# digamma: shift via recurrence ψ(x) = ψ(x+8) − Σ 1/(x+k) to push the
#   argument above 8, then apply the asymptotic series.

_LANCZOS_G = 7.0
_LANCZOS_P: list[float] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
]


def lgamma(x: Tensor) -> Tensor:
    """Natural log of the gamma function via the Lanczos approximation."""
    z = x - 1.0
    t = z + (_LANCZOS_G + 0.5)
    series = lucid.full_like(x, _LANCZOS_P[0])
    for k in range(1, len(_LANCZOS_P)):
        series = series + _LANCZOS_P[k] / (z + float(k))
    return (
        0.5 * math.log(2.0 * math.pi) + (z + 0.5) * lucid.log(t) - t + lucid.log(series)
    )


def digamma(x: Tensor) -> Tensor:
    """Digamma function ψ(x) = d/dx ln Γ(x) via recurrence + asymptotic series."""
    # Shift to xr = x + 8 accumulating the correction sum.
    # ψ(x) = ψ(x+8) − 1/x − 1/(x+1) − … − 1/(x+7)
    correction = lucid.zeros_like(x)
    for k in range(8):
        correction = correction + 1.0 / (x + float(k))
    xr = x + 8.0
    # Asymptotic expansion: ψ(xr) ≈ ln(xr) − 1/(2xr) − B2/(2xr²) + B4/(4xr⁴) − B6/(6xr⁶)
    # Bernoulli-based: B2=1/6, B4=-1/30, B6=1/42
    r = 1.0 / xr
    r2 = r * r
    psi = lucid.log(xr) - 0.5 * r - r2 / 12.0 + r2 * r2 / 120.0 - r2 * r2 * r2 / 252.0
    return psi - correction


# ── Modified Bessel function I₀ ───────────────────────────────────────────
#
# Uses the Abramowitz & Stegun polynomial approximation (Table 9.8.1):
#   |x| ≤ 3.75 : series in (x/3.75)²
#   |x| > 3.75 : series in 3.75/|x|, multiplied by exp(|x|)/√|x|

_I0_SMALL_COEFFS: list[float] = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.0360768,
    0.0045813,
]
_I0_LARGE_COEFFS: list[float] = [
    0.39894228,
    0.01328592,
    0.00225319,
    -0.00157565,
    0.00916281,
    -0.02057706,
    0.02635537,
    -0.01647633,
    0.00392377,
]


def i0(x: Tensor) -> Tensor:
    """Modified Bessel function of the first kind, order 0."""
    ax = lucid.abs(x)
    # Guard: avoid division by zero in large-argument branch (ax == 0 → use small branch)
    ax_safe = lucid.where(ax == lucid.zeros_like(ax), lucid.full_like(ax, 1.0), ax)

    # Small-argument branch: polynomial in t = (x / 3.75)²
    t_small = (x * (1.0 / 3.75)) ** 2
    val_small = lucid.full_like(x, _I0_SMALL_COEFFS[0])
    t_pow = lucid.ones_like(x)
    for c in _I0_SMALL_COEFFS[1:]:
        t_pow = t_pow * t_small
        val_small = val_small + c * t_pow

    # Large-argument branch: polynomial in y = 3.75 / |x|
    y_large = 3.75 / ax_safe
    val_large = lucid.full_like(x, _I0_LARGE_COEFFS[0])
    y_pow = lucid.ones_like(x)
    for c in _I0_LARGE_COEFFS[1:]:
        y_pow = y_pow * y_large
        val_large = val_large + c * y_pow
    val_large = val_large * lucid.exp(ax_safe) / lucid.sqrt(ax_safe)

    return lucid.where(ax <= 3.75, val_small, val_large)


def softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Softmax along ``dim`` (default: last axis)."""
    axis = dim if dim is not None else -1
    return _wrap(_C_engine.softmax(_unwrap(x), axis))


def log_softmax(x: Tensor, dim: int | None = None) -> Tensor:
    """Log-softmax along ``dim`` (default: last axis)."""
    axis = dim if dim is not None else -1
    sm = _C_engine.softmax(_unwrap(x), axis)
    return _wrap(_C_engine.log(sm))


def floor_divide(a: Tensor, b: Tensor | Scalar) -> Tensor:
    """Element-wise floor division: ``floor(a / b)``."""
    return (a / b).floor()


def diag_embed(
    x: Tensor,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Tensor:
    """Embed the last dimension of ``x`` as the diagonal of a new matrix.

    For a 1-D input of length ``n`` returns shape ``(n+|offset|, n+|offset|)``.
    For batch inputs the last dimension is embedded; ``dim1``/``dim2`` select
    which two axes of the *output* carry the matrix (default: last two).
    """
    n = int(x.shape[-1])
    size = n + abs(offset)
    batch_shape = list(x.shape[:-1])
    row_off = max(0, -offset)
    col_off = max(0, offset)

    # Build (size, size) diagonal mask by padding eye(n) with zero rows/cols.
    eye_n = lucid.eye(n, dtype=x.dtype, device=x.device)
    if col_off > 0:
        eye_n = lucid.cat(
            [lucid.zeros((n, col_off), dtype=x.dtype, device=x.device), eye_n], dim=1
        )
    right = size - n - col_off
    if right > 0:
        eye_n = lucid.cat(
            [eye_n, lucid.zeros((n, right), dtype=x.dtype, device=x.device)], dim=1
        )
    if row_off > 0:
        eye_n = lucid.cat(
            [lucid.zeros((row_off, size), dtype=x.dtype, device=x.device), eye_n], dim=0
        )
    bot = size - n - row_off
    if bot > 0:
        eye_n = lucid.cat(
            [eye_n, lucid.zeros((bot, size), dtype=x.dtype, device=x.device)], dim=0
        )
    # eye_n is now (size, size) with the diagonal at (row_off+i, col_off+i).

    # Broadcast x (..., n) against diagonal mask (size, size).
    # For offset==0: n==size, so x.reshape(..., n, 1) * eye_n.reshape(1,...,n,n).
    # For offset!=0: pad x to (..., size) by prepending/appending zeros.
    if offset == 0:
        diag_view = eye_n.reshape([1] * len(batch_shape) + [size, size])
        return x.reshape(batch_shape + [n, 1]) * diag_view
    else:
        # Expand x to (..., size) with zeros at padded positions.
        pad_before = lucid.zeros(
            tuple(batch_shape + [row_off]), dtype=x.dtype, device=x.device
        )
        pad_after = lucid.zeros(
            tuple(batch_shape + [size - n - row_off]), dtype=x.dtype, device=x.device
        )
        x_pad = lucid.cat([pad_before, x, pad_after], dim=-1)  # (..., size)
        diag_view = eye_n.reshape([1] * len(batch_shape) + [size, size])
        return x_pad.reshape(batch_shape + [size, 1]) * diag_view


__all__ = [
    "absolute",
    "negative",
    "positive",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "rsub",
    "arctan2",
    "arccosh",
    "acosh",
    "arcsinh",
    "asinh",
    "arctanh",
    "atanh",
    "expm1",
    "sinc",
    "heaviside",
    "xlogy",
    "logit",
    "signbit",
    "float_power",
    "fmax",
    "fmin",
    "erfc",
    "copysign",
    "ldexp",
    "frexp",
    "gcd",
    "lcm",
    "lgamma",
    "digamma",
    "i0",
    "softmax",
    "log_softmax",
    "floor_divide",
    "diag_embed",
]
