"""
Gradient clipping utilities.
All computation goes through the C++ engine — no numpy.
"""

from typing import Iterable, TYPE_CHECKING, cast
import math
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._factories.creation import zeros

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.parameter import Parameter


_D = _C_engine.Dtype
_NORMABLE = frozenset({_D.F16, _D.BF16, _D.F32, _D.F64, _D.C64})


def _total_norm(
    impls: list[_C_engine.TensorImpl], norm_type: float, error_if_nonfinite: bool
) -> float:
    """The :math:`\\ell_p` norm of ``impls`` taken jointly, as a Python float."""
    dt = impls[0].dtype
    dev = impls[0].device
    if norm_type == math.inf:
        max_val = _C_engine.full([1], float("-inf"), dt, dev)
        for impl in impls:
            m = _C_engine.reshape(_C_engine.max(_C_engine.abs(impl), [], False), [1])
            mv = _C_engine.reshape(max_val, [1])
            max_val = _C_engine.max(_C_engine.stack([mv, m], 0), [0], False)
        total_norm = float(_wrap(max_val).item())
    else:
        acc = _C_engine.zeros([1], dt, dev)
        for impl in impls:
            pow_g = _C_engine.pow_scalar(_C_engine.abs(impl), norm_type)
            s = _C_engine.reshape(_C_engine.sum(pow_g, [], False), [1])
            acc = _C_engine.add(acc, s)
        total_norm = float(_wrap(acc).item()) ** (1.0 / norm_type)

    if error_if_nonfinite and (math.isnan(total_norm) or math.isinf(total_norm)):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients is "
            f"non-finite ({total_norm})."
        )
    return total_norm


def clip_grad_norm_(
    parameters: Iterable[Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tensor:
    r"""Clip the global gradient norm of ``parameters`` in place.

    Rescales every gradient so that the total :math:`\ell_p` norm —
    computed across all parameters jointly, as if they were one long
    concatenated vector — is at most ``max_norm``.  A staple of stable
    Transformer / RNN training: prevents the occasional huge gradient
    from derailing optimisation.

    Parameters
    ----------
    parameters : iterable of Parameter
        Parameters whose ``.grad`` should be clipped.  Entries with
        ``grad is None`` are silently skipped.
    max_norm : float
        Maximum allowed norm of the combined gradient vector.  The
        scaling factor never exceeds ``1`` — gradients smaller than
        ``max_norm`` are untouched.
    norm_type : float, optional
        Order :math:`p` of the norm.  Default ``2.0`` (Euclidean).  Pass
        ``math.inf`` for the max-norm (element-wise absolute maximum
        across all gradients).
    error_if_nonfinite : bool, optional
        If ``True``, raise :class:`RuntimeError` when the computed total
        norm is ``inf`` or ``nan`` instead of silently scaling by a
        non-finite coefficient.

    Returns
    -------
    Tensor
        Scalar tensor holding the *pre-clipping* total norm.  Useful for
        logging the gradient magnitude during training even when no
        actual clipping took place.

    Notes
    -----
    With combined norm :math:`\|g\|_p = \left(\sum_i |g_i|^p\right)^{1/p}`
    taken over every element of every gradient, the update is

    .. math::

        g \;\mapsto\; g \cdot \min\!\left(1,\,
            \frac{\text{max\_norm}}{\|g\|_p + \epsilon}\right),

    where the :math:`\epsilon = 10^{-6}` guards against division by zero
    when all gradients vanish.  Because every parameter is scaled by the
    *same* coefficient the direction of the global update is preserved —
    only its magnitude is bounded.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import clip_grad_norm_
    >>> model = nn.Linear(4, 2)
    >>> (model(lucid.randn(8, 4)) * 100.0).sum().backward()
    >>> total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    >>> total_norm.shape                        # a 0-d scalar
    ()
    >>> float(total_norm) > 1.0                 # the norm before clipping
    True
    >>> after = clip_grad_norm_(model.parameters(), max_norm=1.0)
    >>> abs(float(after) - 1.0) < 1e-3          # and after it
    True
    """
    params = list(parameters)
    params_with_grad = [p for p in params if p.grad is not None]
    if not params_with_grad:
        # Preserve the parameters' device/dtype for the zero-norm result —
        # defaulting to CPU/float32 here breaks multi-device / mixed-precision
        # callers.  Fall back to the global default only when there are no
        # parameters at all (nothing to infer from).
        if not params:
            return zeros()
        impl0 = params[0]._impl
        return _wrap(_C_engine.full([], 0.0, impl0.dtype, impl0.device))

    dev = params_with_grad[0]._impl.device
    dt = params_with_grad[0]._impl.dtype
    grads = []
    for p in params_with_grad:
        assert p.grad is not None
        grads.append(_unwrap(p.grad))
    total_norm = _total_norm(grads, norm_type, error_if_nonfinite)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p, g_impl in zip(params_with_grad, grads):
            coef = _C_engine.full(
                list(g_impl.shape), clip_coef, g_impl.dtype, g_impl.device
            )
            p._impl.set_grad(_C_engine.mul(g_impl, coef))

    # 0-d, as documented and as the reference framework returns it: a
    # ``(1,)`` norm puts a stray axis into whatever it is stacked or
    # broadcast with.
    return _wrap(_C_engine.full([], total_norm, dt, dev))


def clip_grad_value_(
    parameters: Iterable[Parameter],
    clip_value: float,
) -> None:
    r"""Clamp every gradient element to :math:`[-\text{clip\_value}, \text{clip\_value}]` in place.

    Unlike :func:`clip_grad_norm_`, which preserves the direction of the
    full gradient vector, this operates element-wise — each scalar entry
    is independently clipped to the symmetric interval.  Cheap, and a
    useful safety net when only a handful of weights tend to blow up
    (e.g. embedding tables on rare tokens).

    Parameters
    ----------
    parameters : iterable of Parameter
        Parameters whose ``.grad`` should be clipped.  Entries with
        ``grad is None`` are skipped.
    clip_value : float
        Symmetric magnitude bound.  Must be non-negative; gradients are
        clamped to ``[-clip_value, +clip_value]``.

    Returns
    -------
    None
        The clipping happens in place via ``Parameter.grad``.

    Notes
    -----
    The element-wise update is

    .. math::

        g_i \;\mapsto\; \mathrm{clip}(g_i,\, -c,\, +c),

    with :math:`c = \text{clip\_value}`.  Because each component is
    treated independently the direction of the gradient vector is *not*
    preserved — large coordinates are flattened toward zero while small
    ones pass through unchanged.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import clip_grad_value_
    >>> model = nn.Linear(4, 2)
    >>> (model(lucid.randn(8, 4)) * 100.0).sum().backward()
    >>> clip_grad_value_(model.parameters(), clip_value=0.5)
    >>> max(float(p.grad.abs().max()) for p in model.parameters()) <= 0.5
    True
    """
    for p in parameters:
        if p.grad is not None:
            assert p.grad is not None
            g_impl = _unwrap(p.grad)
            p._impl.set_grad(_C_engine.clip(g_impl, -clip_value, clip_value))


def get_total_norm(
    tensors: Iterable[Tensor] | Tensor,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> Tensor:
    r"""Compute the :math:`\ell_p` norm of ``tensors``, taken jointly.

    The measurement step of :func:`clip_grad_norm_`, which hands it the
    parameters' gradients.  It measures the tensors it is given, as the
    reference framework's does — it used to read ``.grad`` from them, so
    the usual call, ``get_total_norm([p.grad for p in model.parameters()])``,
    read the gradients' own (absent) gradients and answered 0.

    Parameters
    ----------
    tensors : iterable of Tensor, or Tensor
        The tensors whose elements make up the norm — typically the
        gradients.  ``None`` entries are skipped; the empty case returns a
        zero tensor.
    norm_type : float, optional
        Order :math:`p` of the norm.  Default ``2.0``; use ``math.inf``
        for the max-norm.
    error_if_nonfinite : bool, optional
        Raise on ``inf`` / ``nan`` results instead of returning them.
    foreach : bool, optional
        Accepted for API compatibility; ignored — Lucid processes the
        tensors sequentially through the C++ engine.

    Returns
    -------
    Tensor
        Scalar tensor with the combined norm.

    Notes
    -----
    The returned scalar is

    .. math::

        \|g\|_p \;=\; \left(\sum_i |g_i|^p\right)^{1/p}

    taken over every element of every tensor — the quantity
    :func:`clip_grad_norm_` thresholds against ``max_norm``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils.clip_grad import get_total_norm
    >>> get_total_norm([lucid.tensor([3.0, 4.0])]).item()
    5.0
    >>> model = nn.Linear(4, 2)
    >>> model(lucid.randn(8, 4)).sum().backward()
    >>> g_norm = get_total_norm([p.grad for p in model.parameters()])
    >>> g_norm.shape
    ()
    >>> float(g_norm) > 0.0
    True
    """
    items: list[Tensor | None] = (
        [cast("Tensor", tensors)] if hasattr(tensors, "_impl") else list(tensors)
    )
    impls = [_unwrap(t) for t in items if t is not None]
    if not impls:
        return zeros()
    # A norm is a real number, and integer tensors have no norm kernel on
    # every device — the CPU refused int64 while Metal computed it.  The
    # reference refuses them too; so does this, the same way everywhere.
    for impl in impls:
        if impl.dtype not in _NORMABLE:
            raise TypeError(
                f"get_total_norm: expected floating-point or complex tensors, "
                f"got {impl.dtype}"
            )
    total_norm = _total_norm(impls, norm_type, error_if_nonfinite)
    # 0-d, like :func:`clip_grad_norm_`'s result.
    return _wrap(_C_engine.full([], total_norm, impls[0].dtype, impls[0].device))


clip_grad_norm = clip_grad_norm_
