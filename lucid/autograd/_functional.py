"""
Higher-order autograd utilities: jacobian, hessian, vjp, jvp.

Every derivative here is taken through the graph with
:func:`lucid.autograd.grad` — reverse mode, and reverse mode twice for
``jvp`` — so each is exact up to rounding, and under ``create_graph`` each
result is itself differentiable.
"""

from collections.abc import Sequence
from typing import Any, Callable, TYPE_CHECKING

import lucid
from lucid.autograd._backward import grad as _grad
from lucid.autograd._grad_mode import enable_grad

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _require_differentiable(inputs: Tensor | tuple[Tensor, ...], where: str) -> None:
    """Refuse to differentiate with respect to a discrete input.

    A derivative with respect to an integer is not a small number, it is
    not a number — there is no neighbouring value to take a limit
    towards.  These entry points accepted one and produced whatever the
    ops underneath happened to do with it, which differed by device:
    ``jacobian`` ran on Metal for int8 and raised on the CPU, because the
    two backends refuse different narrow widths for unrelated reasons.

    The reference states it as "only Tensors of floating point dtype can
    require gradients", and that is the same rule.
    """
    candidates = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
    for item in candidates:
        if hasattr(item, "dtype") and not lucid.is_floating_point(item):
            raise TypeError(
                f"{where}: only a floating-point input can be differentiated, "
                f"got {item.dtype}"
            )


def _differentiable_inputs(
    inputs: tuple[Tensor, ...], create_graph: bool
) -> list[Tensor]:
    """The tensors to differentiate ``func`` at, leaving the caller's alone.

    Each is a fresh leaf that requires grad and shares the input's
    storage, so neither the caller's ``requires_grad`` nor its ``.grad``
    is touched.  Under ``create_graph`` an input that already requires
    grad is passed as a view instead, so the result stays differentiable
    with respect to it.
    """
    return [
        (
            x.view_as(x)
            if create_graph and x.requires_grad
            else x.detach().requires_grad_(True)
        )
        for x in inputs
    ]


def _as_tuple(value: Tensor | Sequence[Tensor]) -> tuple[tuple[Tensor, ...], bool]:
    """``value`` as a tuple of tensors, and whether it was one tensor."""
    from lucid._tensor.tensor import Tensor as _Tensor

    if isinstance(value, _Tensor):
        return (value,), True
    return tuple(value), False


def _or_zeros(grads: Sequence[Tensor | None], like: Sequence[Tensor]) -> list[Tensor]:
    """Each gradient, or zeros shaped like its tensor where it is unused."""
    return [
        lucid.zeros_like(t) if g is None else g
        for g, t in zip(grads, like, strict=True)
    ]


def _jacobian_blocks(
    output: Tensor, xs: Sequence[Tensor], create_graph: bool
) -> list[Tensor]:
    """``d output / d x`` for each ``x`` in ``xs``, flattened to
    ``(output.numel(), x.numel())``.

    One reverse pass per element of ``output``, each seeded with a row of
    the identity, so row ``k`` is exactly ``d output[k] / d x``.  The graph
    is kept between passes; under ``create_graph`` each row is
    differentiable.
    """
    n = output.numel()
    basis = lucid.eye(n, dtype=output.dtype, device=output.device)
    rows: list[list[Tensor]] = [[] for _ in xs]
    for k in range(n):
        grads = _grad(
            output,
            list(xs),
            grad_outputs=basis[k].reshape(output.shape),
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        for row, g in zip(rows, _or_zeros(grads, xs), strict=True):
            row.append(g.reshape(-1))
    return [
        (
            lucid.stack(row)
            if row
            else lucid.zeros(0, x.numel(), dtype=x.dtype, device=x.device)
        )
        for row, x in zip(rows, xs, strict=True)
    ]


def jacobian(
    func: Callable[..., Tensor],
    inputs: Tensor | tuple[Tensor, ...],
    create_graph: bool = False,
    strict: bool = False,
    vectorize: bool = False,
) -> Tensor | tuple[Tensor, ...]:
    r"""Compute the Jacobian matrix of ``func`` with respect to each input.

    The Jacobian of a vector-valued function
    :math:`f : \mathbb{R}^n \to \mathbb{R}^m` is

    .. math::

        J_{ij} = \frac{\partial f_i}{\partial x_j}, \qquad
        J \in \mathbb{R}^{m \times n}.

    Lucid evaluates it row by row with reverse-mode passes — one per
    output element — seeding each pass with a row of the identity so the
    resulting input gradient is exactly the corresponding Jacobian row.
    The cost therefore scales with the output dimension :math:`m`; prefer
    :func:`vjp` when only :math:`v^\top J` is needed and :func:`jvp` when
    only :math:`J v` is needed.

    Parameters
    ----------
    func : callable
        A function mapping ``Tensor`` inputs to a ``Tensor`` (or
        tuple of ``Tensor``). Must be differentiable w.r.t. each
        positional input.
    inputs : Tensor or tuple of Tensor
        Input tensor(s) at which the Jacobian is evaluated. They are
        left as they are: the Jacobian is taken at copies that require
        grad, so the inputs' ``requires_grad`` and ``.grad`` do not change.
    create_graph : bool, optional
        If ``True`` the Jacobian itself is differentiable — with respect
        to the inputs that require grad — enabling higher-order
        derivatives. Defaults to ``False``.
    strict : bool, optional
        Reserved for stricter shape/dtype validation. Currently
        unused.
    vectorize : bool, optional
        Reserved for a future vmap-based implementation. Currently
        unused.

    Returns
    -------
    Tensor or tuple of Tensor
        For a single input ``x`` the returned tensor has shape
        ``(numel(out), numel(x))``, the outputs' rows stacked in order
        when ``func`` returns several. For a tuple of inputs a tuple is
        returned, one Jacobian block per input.

    Notes
    -----
    Reverse-mode differentiation makes the cost per row
    :math:`O(\text{cost}(f))`; the full Jacobian therefore costs
    :math:`O(m \cdot \text{cost}(f))`.  An input the output does not
    depend on gets a block of zeros.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import jacobian
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> def f(x):
    ...     return x * x
    >>> jacobian(f, x)
    tensor([[2., 0., 0.], [0., 4., 0.], [0., 0., 6.]])
    """
    _require_differentiable(inputs, "jacobian")
    inputs_t, one_input = _as_tuple(inputs)
    xs = _differentiable_inputs(inputs_t, create_graph)
    with enable_grad():
        outputs, _ = _as_tuple(func(*xs))
        per_output = [_jacobian_blocks(o, xs, create_graph) for o in outputs]
    blocks = [
        (
            lucid.cat([b[j] for b in per_output], 0)
            if len(per_output) > 1
            else per_output[0][j]
        )
        for j in range(len(xs))
    ]
    return blocks[0] if one_input else tuple(blocks)


def hessian(
    func: Callable[..., Tensor],
    inputs: Tensor | tuple[Tensor, ...],
    create_graph: bool = False,
    strict: bool = False,
    vectorize: bool = False,
) -> Tensor | tuple[tuple[Tensor, ...], ...]:
    r"""Compute the Hessian matrix of a scalar-valued ``func``.

    The Hessian of a scalar function
    :math:`f : \mathbb{R}^n \to \mathbb{R}` is

    .. math::

        H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j},
        \qquad H \in \mathbb{R}^{n \times n}.

    Implemented as the :func:`jacobian` of the gradient of ``func`` —
    one forward pass, one backward pass that records its own graph, and
    one backward pass through that graph per gradient coordinate.  Cost
    is therefore :math:`O(n \cdot \text{cost}(\nabla f))`.

    Parameters
    ----------
    func : callable
        Function of one or more ``Tensor`` inputs returning a tensor of
        one element.
    inputs : Tensor or tuple of Tensor
        Inputs at which :math:`H` is evaluated. They are left as they
        are: the Hessian is taken at copies that require grad, so the
        inputs' ``requires_grad`` and ``.grad`` do not change.
    create_graph : bool, optional
        If ``True`` the Hessian itself is differentiable with respect to
        the inputs that require grad (third-order derivatives). Defaults
        to ``False``.
    strict : bool, optional
        Reserved for stricter validation. Currently unused.
    vectorize : bool, optional
        Reserved for a future vmap-based implementation. Currently
        unused.

    Returns
    -------
    Tensor or tuple of tuple of Tensor
        For a single input the returned tensor has shape
        ``(numel(x), numel(x))``. For multiple inputs a nested
        tuple of cross-Hessian blocks is returned, with
        ``H[i][j]`` of shape ``(numel(x_i), numel(x_j))`` containing
        :math:`\partial^2 f / (\partial x_i \, \partial x_j)`.

    Raises
    ------
    RuntimeError
        ``func`` does not return a single tensor of one element.

    Notes
    -----
    Symmetry :math:`H_{ij} = H_{ji}` holds in exact arithmetic
    when :math:`f` is :math:`C^2`. In floating-point the result is
    only approximately symmetric; symmetrize as
    :math:`\tfrac{1}{2}(H + H^\top)` if a strictly symmetric
    matrix is required.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import hessian
    >>> x = lucid.tensor([1.0, 2.0])
    >>> def f(x):
    ...     return (x ** 3).sum()
    >>> hessian(f, x).tolist()
    [[6.0, 0.0], [0.0, 12.0]]
    """
    _require_differentiable(inputs, "hessian")
    from lucid._tensor.tensor import Tensor as _Tensor

    inputs_t, one_input = _as_tuple(inputs)
    xs = _differentiable_inputs(inputs_t, create_graph)
    with enable_grad():
        # ``object``: a callable typed to return a tensor may return a tuple.
        out: object = func(*xs)
        if not isinstance(out, _Tensor) or out.numel() != 1:
            got = (
                f"shape {tuple(out.shape)}"
                if isinstance(out, _Tensor)
                else type(out).__name__
            )
            raise RuntimeError(
                f"hessian: func must return a single tensor of one element, got {got}"
            )
        firsts = _or_zeros(_grad(out, xs, create_graph=True, allow_unused=True), xs)
        blocks = tuple(tuple(_jacobian_blocks(g, xs, create_graph)) for g in firsts)
    return blocks[0][0] if one_input else blocks


def vjp(
    func: Callable[..., Tensor],
    inputs: Tensor | tuple[Tensor, ...],
    v: Tensor | tuple[Tensor, ...],
    create_graph: bool = False,
    strict: bool = False,
) -> tuple[Tensor, tuple[Tensor | None, ...]]:
    r"""Vector-Jacobian product :math:`v^\top J` (reverse-mode AD).

    Given :math:`f : \mathbb{R}^n \to \mathbb{R}^m` with Jacobian
    :math:`J \in \mathbb{R}^{m \times n}` and a cotangent vector
    :math:`v \in \mathbb{R}^m`, returns

    .. math::

        v^\top J \in \mathbb{R}^{n}

    along with the primal output :math:`y = f(x)`. This is the
    operation that backpropagation performs on every node: when
    a scalar loss :math:`\mathcal{L}(y)` is being differentiated
    against an intermediate :math:`y`, the upstream cotangent is
    :math:`v = \partial \mathcal{L} / \partial y` and the result
    is :math:`\partial \mathcal{L} / \partial x`.

    Computing a full VJP costs the same as one backward pass —
    much cheaper than materialising :math:`J` when only the
    product is needed.

    Parameters
    ----------
    func : callable
        Function mapping ``Tensor`` inputs to a ``Tensor`` (or
        tuple thereof).
    inputs : Tensor or tuple of Tensor
        Primal point :math:`x` at which :math:`J` is evaluated. Left as
        it is: the product is taken at copies that require grad.
    v : Tensor or tuple of Tensor
        Cotangent vector(s), one per output of ``func``, each of its
        output's shape and device — the seed rules of
        :func:`lucid.autograd.grad`.
    create_graph : bool, optional
        If ``True`` the returned VJP is itself differentiable,
        enabling double-backward. Defaults to ``False``.
    strict : bool, optional
        Reserved for stricter validation. Currently unused.

    Returns
    -------
    tuple of (Tensor, tuple of (Tensor or None))
        ``(output, vjp_grads)`` where ``output = func(*inputs)`` —
        detached unless ``create_graph`` — and ``vjp_grads[i]`` is
        :math:`v^\top J` projected onto input ``i`` (or ``None`` if that
        input has no gradient path).

    Raises
    ------
    RuntimeError
        The number of entries of ``v`` differs from the number of
        outputs.
    ShapeMismatch, DeviceMismatch
        An entry of ``v`` does not have its output's shape or device.

    Notes
    -----
    The dual to :func:`vjp` is :func:`jvp`, which computes :math:`J v`
    by differentiating this product with respect to :math:`v`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import vjp
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> v = lucid.tensor([1.0, 1.0, 1.0])
    >>> def f(x):
    ...     return x * x
    >>> y, (grad_x,) = vjp(f, x, v)
    >>> grad_x
    tensor([2., 4., 6.])
    """
    _require_differentiable(inputs, "vjp")
    inputs_t, _ = _as_tuple(inputs)
    v_t, _ = _as_tuple(v)
    xs = _differentiable_inputs(inputs_t, create_graph)
    with enable_grad():
        outputs = func(*xs)
        outs, _ = _as_tuple(outputs)
        grads = _grad(
            list(outs),
            xs,
            grad_outputs=list(v_t),
            retain_graph=create_graph,
            create_graph=create_graph,
            allow_unused=True,
        )

    # ``retain_graph`` follows ``create_graph``, so by default the graph
    # behind ``outputs`` has just been freed — while ``outputs`` itself
    # still advertises ``requires_grad=True``.  Handing that back invites
    # a second backward through storages that no longer exist, and it does
    # not raise: ``MulBackward`` multiplied a live gradient by a released
    # operand and read off the end of it.  ``(out * out).sum().backward()``
    # after a plain ``vjp`` was a segfault in four lines.
    #
    # Detached only when the graph was not kept.  Under ``create_graph``
    # the graph is alive and differentiating the output again is exactly
    # what the flag is for.
    if not create_graph:
        outputs = _detach_outputs(outputs)
    return outputs, grads


def _detach_outputs(outputs: Any) -> Any:
    """Detach ``outputs``, preserving whether it was a single tensor.

    ``vjp`` and its callers accept one tensor or a tuple of them, and the
    return type has to keep that distinction.
    """
    if isinstance(outputs, (list, tuple)):
        return type(outputs)(o.detach() if hasattr(o, "detach") else o for o in outputs)
    return outputs.detach() if hasattr(outputs, "detach") else outputs


def jvp(
    func: Callable[..., Tensor],
    inputs: Tensor | tuple[Tensor, ...],
    v: Tensor | tuple[Tensor, ...],
    create_graph: bool = False,
    strict: bool = False,
) -> tuple[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]]:
    r"""Jacobian-vector product :math:`J v` (forward-mode directional derivative).

    Given :math:`f : \mathbb{R}^n \to \mathbb{R}^m` with Jacobian
    :math:`J \in \mathbb{R}^{m \times n}` and a tangent vector
    :math:`v \in \mathbb{R}^n`, returns

    .. math::

        J v = \left.\frac{d}{dt} f(x + t v)\right|_{t=0}
            \in \mathbb{R}^{m}

    along with the primal output :math:`y = f(x)`.  JVPs are useful for
    propagating tangent information (sensitivities) through a network,
    for directional derivatives, and as a building block for
    second-order methods.

    Computed exactly with two reverse-mode passes (the double-vjp
    trick): :math:`g(u) = J^\top u` is linear in a dummy cotangent
    :math:`u`, so its vector-Jacobian product with :math:`v` is
    :math:`(J^\top)^\top v = J v`.  No finite differences are involved.

    Parameters
    ----------
    func : callable
        Function mapping ``Tensor`` inputs to a ``Tensor`` (or
        tuple thereof).
    inputs : Tensor or tuple of Tensor
        Primal point :math:`x`.  Left as it is: the product is taken at
        copies that require grad.
    v : Tensor or tuple of Tensor
        Tangent vector(s), one per input, each of its input's shape and
        device.
    create_graph : bool, optional
        If ``True`` the returned outputs and tangents are differentiable
        with respect to the inputs that require grad (and ``v``).
        Defaults to ``False``.
    strict : bool, optional
        Reserved for stricter validation. Currently unused.

    Returns
    -------
    tuple of (Tensor or tuple of Tensor, Tensor or tuple of Tensor)
        ``(primals_out, tangents_out)`` where
        ``primals_out = func(*inputs)`` — detached unless
        ``create_graph`` — and ``tangents_out`` has the same structure
        and shapes as ``primals_out`` and holds :math:`J v`.  An output
        that does not depend on the inputs has a zero tangent.

    Raises
    ------
    RuntimeError
        The number of entries of ``v`` differs from the number of
        inputs, or an op on the path has no differentiable backward
        (the second pass differentiates the first).
    ShapeMismatch, DeviceMismatch
        An entry of ``v`` does not have its input's shape or device.

    Notes
    -----
    The complementary operation is :func:`vjp`, which computes
    :math:`v^\top J` with one reverse pass.  This JVP costs two.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import jvp
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> v = lucid.tensor([1.0, 0.0, 0.0])
    >>> def f(x):
    ...     return x * x
    >>> y, tangent = jvp(f, x, v)
    >>> tangent
    tensor([2., 0., 0.])
    """
    _require_differentiable(inputs, "jvp")
    inputs_t, _ = _as_tuple(inputs)
    v_t, _ = _as_tuple(v)
    xs = _differentiable_inputs(inputs_t, create_graph)
    with enable_grad():
        outputs = func(*xs)
        outs, one_output = _as_tuple(outputs)
        # ``u`` enters as grad_outputs, whose graph ``grad`` keeps under
        # create_graph: the first pass is J^T u as a function of u.
        us = [lucid.zeros_like(o).requires_grad_(True) for o in outs]
        firsts = _grad(
            list(outs), xs, grad_outputs=us, create_graph=True, allow_unused=True
        )
        tangents = _grad(
            _or_zeros(firsts, xs),
            us,
            grad_outputs=list(v_t),
            create_graph=create_graph,
            allow_unused=True,
        )
    tangents_out = _or_zeros(tangents, outs)
    if not create_graph:
        outputs = _detach_outputs(outputs)
    return outputs, tangents_out[0] if one_output else tuple(tangents_out)
