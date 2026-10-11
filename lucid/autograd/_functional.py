"""
Higher-order autograd utilities: jacobian, hessian, vjp, jvp, hvp, vhp.

Every derivative here is taken through the graph with
:func:`lucid.autograd.grad` — reverse mode, and reverse mode twice for
``jvp`` — so each is exact up to rounding, and under ``create_graph`` each
result is itself differentiable.

All six pack their results by one rule, the reference's: a result per
input comes back bare when ``inputs`` was one bare tensor and as a tuple
when it was a tuple, and a result per output follows what ``func``
returned the same way.  ``strict`` refuses, with the reference's reasons,
a result that would otherwise be filled in with zeros.
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


def _pack(items: Sequence[Tensor], one: bool) -> Tensor | tuple[Tensor, ...]:
    """The packing rule: the one result bare, or every result as a tuple."""
    return items[0] if one else tuple(items)


def _require_tracked(tensors: Sequence[Tensor | None], kind: str, strict: bool) -> None:
    """Under ``strict``, refuse a ``kind`` result that is ``None`` or has no
    graph back to the inputs — the reference's ``strict`` checks."""
    if not strict:
        return
    for i, t in enumerate(tensors):
        if t is None:
            raise RuntimeError(
                "The output of the user-provided function is independent of "
                f"input {i}. This is not allowed in strict mode."
            )
        if t.requires_grad:
            continue
        if kind == "outputs":
            raise RuntimeError(
                f"Output {i} of the user-provided function does not require "
                "gradients. The outputs must be computed in a differentiable "
                "manner from the input when running in strict mode."
            )
        if kind == "grad_inputs":
            raise RuntimeError(
                f"The gradient with respect to input {i} is independent of the "
                "inputs of the user-provided function. This is not allowed in "
                "strict mode."
            )
        raise RuntimeError(
            "While computing the hessian, found that the jacobian of the "
            f"user-provided function with respect to input {i} is independent "
            "of the input. This is not allowed in strict mode. You should ensure "
            "that your function is twice differentiable and that the jacobian "
            "depends on the inputs (this would be violated by a linear function "
            "for example)."
        )


#: Why ``strict`` refuses a result that is ``None``, by the pass it came from.
_INDEPENDENT = {
    "back": "The output of the user-provided function is independent of input {i}.",
    "back_trick": (
        "The gradient with respect to the input is independent of entry {i} in "
        "the grad_outputs when using the double backward trick to compute "
        "forward mode gradients."
    ),
    "double_back": (
        "The jacobian of the user-provided function is independent of input {i}."
    ),
    "double_back_trick": (
        "The hessian of the user-provided function is independent of entry {i} "
        "in the grad_jacobian."
    ),
}


def _filled(
    grads: Sequence[Tensor | None],
    like: Sequence[Tensor],
    stage: str,
    strict: bool,
    create_graph: bool,
) -> list[Tensor]:
    """``grads`` with zeros for the unused entries — or, under ``strict``, a
    refusal of an unused entry and, with ``create_graph``, of one that is a
    constant."""
    if strict:
        for i, g in enumerate(grads):
            if g is None:
                raise RuntimeError(
                    f"{_INDEPENDENT[stage].format(i=i)} This is not allowed in "
                    "strict mode."
                )
            if create_graph and not g.requires_grad:
                what = "hessian" if stage.startswith("double") else "jacobian"
                raise RuntimeError(
                    f"The {what} of the user-provided function is independent of "
                    f"input {i}. This is not allowed in strict mode when "
                    "create_graph=True."
                )
    return _or_zeros(grads, like)


def _single_element(out: object, where: str) -> Tensor:
    """``out`` when it is one tensor of one element; refused otherwise."""
    from lucid._tensor.tensor import Tensor as _Tensor

    if not isinstance(out, _Tensor) or out.numel() != 1:
        got = (
            f"shape {tuple(out.shape)}"
            if isinstance(out, _Tensor)
            else type(out).__name__
        )
        raise RuntimeError(
            f"{where}: func must return a single tensor of one element, got {got}"
        )
    return out


def _seeds(
    v: Tensor | Sequence[Tensor] | None, like: tuple[Tensor, ...], subject: str
) -> list[Tensor]:
    """``v`` as a list, or ones for a ``like`` that is one single-element tensor.

    One tensor, bare or in a 1-tuple: the reference accepts either.

    The reference lets ``v`` be left out exactly there — a scalar loss, a
    scalar input — where a vector of ones is the only seed that makes sense.
    """
    if v is not None:
        return list(_as_tuple(v)[0])
    if len(like) != 1 or like[0].numel() != 1:
        raise RuntimeError(
            f"The vector v can only be None if {subject} a single Tensor with "
            "a single element."
        )
    return [lucid.ones_like(like[0])]


def _jacobian_blocks(
    output: Tensor,
    xs: Sequence[Tensor],
    create_graph: bool,
    strict: bool,
    index: int,
) -> list[Tensor]:
    """``d output / d x`` for each ``x`` in ``xs``, flattened to
    ``(output.numel(), x.numel())``.

    One reverse pass per element of ``output``, each seeded with a row of
    the identity, so row ``k`` is exactly ``d output[k] / d x``.  The graph
    is kept between passes; under ``create_graph`` each row is
    differentiable.  ``index`` is the output's position, for ``strict``'s
    refusals.
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
        if strict:
            _refuse_independent_rows(grads, index, create_graph)
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


def _refuse_independent_rows(
    grads: Sequence[Tensor | None], index: int, create_graph: bool
) -> None:
    """``strict``'s refusal of a Jacobian row that does not reach an input."""
    for j, g in enumerate(grads):
        if g is None:
            raise RuntimeError(
                f"Output {index} of the user-provided function is independent "
                f"of input {j}. This is not allowed in strict mode."
            )
        if create_graph and not g.requires_grad:
            raise RuntimeError(
                "The jacobian of the user-provided function is independent of "
                f"input {j}. This is not allowed in strict mode when "
                "create_graph=True."
            )


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
        If ``True``, refuse an output that does not require grad, or an
        input some output does not depend on, instead of giving it a
        block of zeros — and, under ``create_graph``, a block that is a
        constant.  Defaults to ``False``.
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
        _require_tracked(outputs, "outputs", strict)
        per_output = [
            _jacobian_blocks(o, xs, create_graph, strict, i)
            for i, o in enumerate(outputs)
        ]
    blocks = [
        (
            lucid.cat([b[j] for b in per_output], 0)
            if len(per_output) > 1
            else per_output[0][j]
        )
        for j in range(len(xs))
    ]
    return _pack(blocks, one_input)


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
        If ``True``, refuse a gradient of ``func`` that does not depend on
        the inputs (a linear ``func``, say) and a Hessian block of an
        input the gradient does not depend on, instead of filling them
        with zeros.  Defaults to ``False``.
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
    inputs_t, one_input = _as_tuple(inputs)
    xs = _differentiable_inputs(inputs_t, create_graph)
    with enable_grad():
        _, firsts = _first_gradients(func, xs, strict, "hessian")
        blocks = tuple(
            tuple(_jacobian_blocks(g, xs, create_graph, strict, i))
            for i, g in enumerate(firsts)
        )
    return blocks[0][0] if one_input else blocks


def _first_gradients(
    func: Callable[..., Tensor], xs: Sequence[Tensor], strict: bool, where: str
) -> tuple[Tensor, list[Tensor]]:
    """``func(*xs)`` and its gradient in each ``x``, with the graph kept, for
    the second-order functions; ``func`` must return one single-element
    tensor.

    Call under :func:`enable_grad`.  Zeros stand in for an input ``func``
    does not reach, unless ``strict`` refuses it.
    """
    # ``object``: a callable typed to return a tensor may return a tuple.
    returned: object = func(*xs)
    out = _single_element(returned, where)
    _require_tracked([out], "outputs", strict)
    firsts = _or_zeros(_grad(out, list(xs), create_graph=True, allow_unused=True), xs)
    _require_tracked(firsts, "jacobian", strict)
    return out, firsts


def vjp(
    func: Callable[..., Tensor],
    inputs: Tensor | tuple[Tensor, ...],
    v: Tensor | tuple[Tensor, ...] | None = None,
    create_graph: bool = False,
    strict: bool = False,
) -> tuple[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]]:
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
    v : Tensor or tuple of Tensor, optional
        Cotangent vector(s), one per output of ``func``, each of its
        output's shape and device — the seed rules of
        :func:`lucid.autograd.grad`.  May be left out when ``func``
        returns one single-element tensor, which is then seeded with 1.
    create_graph : bool, optional
        If ``True`` the returned VJP is itself differentiable,
        enabling double-backward. Defaults to ``False``.
    strict : bool, optional
        If ``True``, refuse an output that does not require grad and an
        input the outputs do not depend on, instead of giving it a zero
        gradient — and, under ``create_graph``, a gradient that is a
        constant.  Defaults to ``False``.

    Returns
    -------
    tuple of (Tensor or tuple of Tensor, Tensor or tuple of Tensor)
        ``(output, vjp)`` where ``output = func(*inputs)`` — detached
        unless ``create_graph`` — and ``vjp`` is :math:`v^\top J`, shaped
        and packed like ``inputs``: one tensor for one tensor, a tuple for
        a tuple.  An input the outputs do not depend on gets zeros.

    Raises
    ------
    RuntimeError
        The number of entries of ``v`` differs from the number of
        outputs, ``v`` is left out and ``func`` does not return one
        single-element tensor, or ``strict`` refuses a result.
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
    >>> y, grad_x = vjp(f, x, v)
    >>> grad_x
    tensor([2., 4., 6.])
    """
    _require_differentiable(inputs, "vjp")
    inputs_t, one_input = _as_tuple(inputs)
    xs = _differentiable_inputs(inputs_t, create_graph)
    with enable_grad():
        outputs = func(*xs)
        outs, _ = _as_tuple(outputs)
        _require_tracked(outs, "outputs", strict)
        seeds = _seeds(v, outs, "the user-provided function returns")
        grads = _grad(
            list(outs),
            xs,
            grad_outputs=seeds,
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
    vjps = _filled(grads, inputs_t, "back", strict, create_graph)
    if not create_graph:
        outputs = _detach_outputs(outputs)
    return outputs, _pack(vjps, one_input)


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
    v: Tensor | tuple[Tensor, ...] | None = None,
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
    v : Tensor or tuple of Tensor, optional
        Tangent vector(s), one per input, each of its input's shape and
        device.  May be left out when ``inputs`` is one single-element
        tensor, which is then seeded with 1.
    create_graph : bool, optional
        If ``True`` the returned outputs and tangents are differentiable
        with respect to the inputs that require grad (and ``v``).
        Defaults to ``False``.
    strict : bool, optional
        If ``True``, refuse an output that does not require grad, an input
        the outputs do not depend on and an output whose tangent would be
        zeros for want of a path — and, under ``create_graph``, a tangent
        that is a constant.  Defaults to ``False``.

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
        inputs, ``v`` is left out and ``inputs`` is not one single-element
        tensor, or an op on the path has no differentiable backward
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
    v_t = _seeds(v, inputs_t, "the input to the user-provided function is")
    xs = _differentiable_inputs(inputs_t, create_graph)
    with enable_grad():
        outputs = func(*xs)
        outs, one_output = _as_tuple(outputs)
        _require_tracked(outs, "outputs", strict)
        tangents = _forward_by_double_back(outs, xs, v_t, create_graph, strict)
    tangents_out = _filled(tangents, outs, "back_trick", strict, create_graph)
    if not create_graph:
        outputs = _detach_outputs(outputs)
    return outputs, _pack(tangents_out, one_output)


def _forward_by_double_back(
    outs: Sequence[Tensor],
    xs: Sequence[Tensor],
    v: Sequence[Tensor],
    create_graph: bool,
    strict: bool,
) -> tuple[Tensor | None, ...]:
    """``J v`` for the ``J`` of ``outs`` in ``xs``, by two reverse passes.

    ``g(u) = Jᵀ u`` is linear in a dummy cotangent ``u``, so its own
    vector-Jacobian product with ``v`` is ``J v``.  ``u`` enters as
    ``grad_outputs``, whose graph ``grad`` keeps under ``create_graph``.
    Call under :func:`enable_grad`; shared by :func:`jvp` and :func:`hvp`.
    """
    us = [lucid.zeros_like(o).requires_grad_(True) for o in outs]
    firsts = _grad(
        list(outs), list(xs), grad_outputs=us, create_graph=True, allow_unused=True
    )
    _require_tracked(firsts, "grad_inputs", strict)
    return _grad(
        _or_zeros(firsts, xs),
        us,
        grad_outputs=list(v),
        create_graph=create_graph,
        allow_unused=True,
    )


def _hessian_product(
    name: str,
    func: Callable[..., Tensor],
    inputs: Tensor | tuple[Tensor, ...],
    v: Tensor | tuple[Tensor, ...] | None,
    create_graph: bool,
    strict: bool,
) -> tuple[Tensor, Tensor | tuple[Tensor, ...]]:
    """The body of :func:`hvp` (``name == "hvp"``) and :func:`vhp`.

    Both start from the gradient of ``func`` with its graph kept.  ``vhp``
    is that gradient's vector-Jacobian product with ``v`` — one more
    reverse pass.  ``hvp`` is its Jacobian-vector product, taken as
    :func:`jvp` takes one; the two agree when the Hessian is symmetric.
    """
    _require_differentiable(inputs, name)
    inputs_t, one_input = _as_tuple(inputs)
    v_t = _seeds(v, inputs_t, "the input to the user-provided function is")
    xs = _differentiable_inputs(inputs_t, create_graph)
    with enable_grad():
        out, firsts = _first_gradients(func, xs, strict, name)
        if name == "hvp":
            products = _forward_by_double_back(firsts, xs, v_t, create_graph, False)
            stage = "double_back_trick"
        else:
            products = _grad(
                firsts,
                xs,
                grad_outputs=v_t,
                create_graph=create_graph,
                allow_unused=True,
            )
            stage = "double_back"
    result = _filled(products, inputs_t, stage, strict, create_graph)
    if not create_graph:
        out = out.detach()
    return out, _pack(result, one_input)


def vhp(
    func: Callable[..., Tensor],
    inputs: Tensor | tuple[Tensor, ...],
    v: Tensor | tuple[Tensor, ...] | None = None,
    create_graph: bool = False,
    strict: bool = False,
) -> tuple[Tensor, Tensor | tuple[Tensor, ...]]:
    r"""Vector-Hessian product :math:`v^\top H` of a scalar-valued ``func``.

    For :math:`f : \mathbb{R}^n \to \mathbb{R}` with Hessian
    :math:`H_{ij} = \partial^2 f / \partial x_i \partial x_j` and a vector
    :math:`v \in \mathbb{R}^n`, returns

    .. math::

        v^\top H = \nabla_x \big( v^\top \nabla_x f(x) \big)

    along with :math:`f(x)`, without forming :math:`H`: one backward pass
    that records its own graph, and one backward pass through it.

    Parameters
    ----------
    func : callable
        Function of one or more ``Tensor`` inputs returning a tensor of
        one element.
    inputs : Tensor or tuple of Tensor
        Point :math:`x`.  Left as it is: the product is taken at copies
        that require grad.
    v : Tensor or tuple of Tensor, optional
        One vector per input, each of its input's shape and device.  May
        be left out when ``inputs`` is one single-element tensor, which is
        then seeded with 1.
    create_graph : bool, optional
        If ``True`` the output and the product are differentiable with
        respect to the inputs that require grad (and ``v``).  Defaults to
        ``False``.
    strict : bool, optional
        If ``True``, refuse an output that does not require grad, a
        gradient that does not depend on the inputs (a linear ``func``,
        say) and a product that would be zeros for want of a path — and,
        under ``create_graph``, a product that is a constant.  Defaults
        to ``False``.

    Returns
    -------
    tuple of (Tensor, Tensor or tuple of Tensor)
        ``(output, vhp)`` where ``output = func(*inputs)`` — detached
        unless ``create_graph`` — and ``vhp`` is shaped and packed like
        ``inputs``: one tensor for one tensor, a tuple for a tuple.

    Raises
    ------
    RuntimeError
        ``func`` does not return a single tensor of one element, ``v`` is
        left out and ``inputs`` is not one single-element tensor, or
        ``strict`` refuses a result.
    ShapeMismatch, DeviceMismatch
        An entry of ``v`` does not have its input's shape or device.

    Notes
    -----
    For a twice continuously differentiable ``func`` the Hessian is
    symmetric and :math:`v^\top H = (H v)^\top`, so this and :func:`hvp`
    agree up to rounding; this one costs a pass less.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import vhp
    >>> x = lucid.tensor([1.0, 2.0])
    >>> def f(x):
    ...     return (x ** 3).sum()
    >>> out, product = vhp(f, x, lucid.tensor([1.0, 1.0]))
    >>> product
    tensor([6., 12.])
    """
    return _hessian_product("vhp", func, inputs, v, create_graph, strict)


def hvp(
    func: Callable[..., Tensor],
    inputs: Tensor | tuple[Tensor, ...],
    v: Tensor | tuple[Tensor, ...] | None = None,
    create_graph: bool = False,
    strict: bool = False,
) -> tuple[Tensor, Tensor | tuple[Tensor, ...]]:
    r"""Hessian-vector product :math:`H v` of a scalar-valued ``func``.

    For :math:`f : \mathbb{R}^n \to \mathbb{R}` with Hessian :math:`H`
    and a vector :math:`v \in \mathbb{R}^n`, returns

    .. math::

        H v = \left.\frac{d}{dt} \nabla_x f(x + t v)\right|_{t=0}

    along with :math:`f(x)`, without forming :math:`H` — the directional
    derivative of the gradient, taken exactly as :func:`jvp` takes one.

    Parameters
    ----------
    func : callable
        Function of the inputs returning one single-element tensor, as
        for :func:`vhp`.
    inputs : Tensor or tuple of Tensor
        Where :math:`H` is taken; the caller's tensors are not modified.
    v : Tensor or tuple of Tensor, optional
        The direction, one tensor per input, packed like ``inputs``.
        Optional only for one single-element input (seeded with 1).
    create_graph : bool, optional
        Keep the output and the product differentiable.  Defaults to
        ``False``.
    strict : bool, optional
        Refuse what :func:`vhp` refuses under ``strict``.  Defaults to
        ``False``.

    Returns
    -------
    tuple of (Tensor, Tensor or tuple of Tensor)
        ``(output, hvp)``, the product packed like ``inputs`` and both
        detached unless ``create_graph`` — as :func:`vhp` returns them.

    Raises
    ------
    RuntimeError, ShapeMismatch, DeviceMismatch
        For the reasons :func:`vhp` gives.

    Notes
    -----
    Three reverse passes against :func:`vhp`'s two; prefer :func:`vhp`
    when ``func`` is twice continuously differentiable, where the Hessian
    is symmetric and the two agree up to rounding.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import hvp
    >>> x = lucid.tensor([1.0, 2.0])
    >>> def f(x):
    ...     return (x ** 3).sum()
    >>> out, product = hvp(f, x, lucid.tensor([1.0, 0.0]))
    >>> product
    tensor([6., 0.])
    """
    return _hessian_product("hvp", func, inputs, v, create_graph, strict)
