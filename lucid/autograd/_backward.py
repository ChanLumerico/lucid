"""
autograd.backward() and autograd.grad() free functions.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.autograd._grad_mode import enable_grad

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _refuse_trace() -> None:
    """Mark an active compile trace as one no executable can stand for.

    A backward pass inside a traced function — MeanFlow's ``jvp``, a
    gradient penalty — computes storage the tracer never records: compiled,
    it would replay the trace-time values, and ``make_step`` hung building
    a VJP of the recorded backward.  The trace is marked, every compile
    entry point refuses it, and the call runs eager.
    """
    tracer = _C_engine.compile.current_tracer()
    if tracer is not None:
        tracer.mark_unsupported("a backward pass ran inside the traced function")


def _as_tensors(value: Tensor | Sequence[Tensor]) -> list[Tensor]:
    """One tensor or a sequence of them, as a list."""
    from lucid._tensor.tensor import Tensor

    return [value] if isinstance(value, Tensor) else list(value)


def _check_seed(where: str, index: int, out: Tensor, seed: object) -> Tensor | None:
    """Hold one seed gradient to the output it is for.

    ``None`` stands for an implicit ``1`` and is allowed only for a real
    output of one element.  A tensor must have the output's shape and
    device, and be complex exactly when the output is; another real dtype
    is cast to the output's, as the reference casts it.
    """
    from lucid._tensor.tensor import Tensor

    if seed is None:
        if out.numel() != 1:
            raise RuntimeError(
                f"{where}(): grad can be implicitly created only for scalar "
                f"outputs, but output {index} has shape {tuple(out.shape)}; "
                "pass its gradient explicitly"
            )
        if out.is_complex():
            raise RuntimeError(
                f"{where}(): grad can be implicitly created only for real "
                f"scalar outputs, but output {index} is {out.dtype}"
            )
        return None
    if not isinstance(seed, Tensor):
        raise TypeError(
            f"{where}(): gradient {index} must be a Tensor or None, "
            f"got {type(seed).__name__}"
        )
    if tuple(seed.shape) != tuple(out.shape):
        raise _C_engine.ShapeMismatch(
            f"{where}(): mismatch in shape: gradient {index} has shape "
            f"{tuple(seed.shape)} and output {index} has shape {tuple(out.shape)}"
        )
    if seed.is_complex() != out.is_complex():
        raise _C_engine.DtypeMismatch(
            f"{where}(): a complex output and its gradient must both be "
            f"complex: gradient {index} is {seed.dtype} and output {index} "
            f"is {out.dtype}"
        )
    if seed.device != out.device:
        raise _C_engine.DeviceMismatch(
            f"{where}(): gradient {index} is on {seed.device.type} but "
            f"output {index} is on {out.device.type}"
        )
    return seed if seed.dtype == out.dtype else seed.to(out.dtype)


def _make_seeds(
    where: str,
    outputs: Sequence[Tensor],
    grads: Tensor | Sequence[Tensor | None] | None,
) -> list[Tensor | None]:
    """Match the seed gradients a caller passed to the outputs, one each.

    The one boundary for ``grad_outputs`` / ``grad_tensors`` /
    ``Tensor.backward(gradient)``: a single tensor or ``None`` stands for
    a one-element sequence, the count must equal the number of outputs,
    and each seed is checked by :func:`_check_seed`.

    Raises
    ------
    TypeError
        ``grads`` or one of its entries is neither a tensor nor ``None``.
    RuntimeError
        The counts differ, or ``None`` stands for the gradient of an
        output that is not a real scalar.
    ShapeMismatch, DtypeMismatch, DeviceMismatch
        A seed does not fit its output.
    """
    from lucid._tensor.tensor import Tensor

    seeds: list[object]
    if grads is None:
        seeds = [None] * len(outputs)
    elif isinstance(grads, Tensor):
        seeds = [grads]
    elif isinstance(grads, (list, tuple)):
        seeds = list(grads)
    else:
        raise TypeError(
            f"{where}(): gradients must be a Tensor, a sequence of Tensors "
            f"or None, got {type(grads).__name__}"
        )
    if len(seeds) != len(outputs):
        raise RuntimeError(
            f"{where}(): got {len(outputs)} tensors and {len(seeds)} gradients"
        )
    return [
        _check_seed(where, i, out, seed)
        for i, (out, seed) in enumerate(zip(outputs, seeds, strict=True))
    ]


def _seeded_root(
    out: Tensor, seed: Tensor | None, keep_seed_graph: bool
) -> _C_engine.TensorImpl:
    """The scalar the engine differentiates: ``out``, or ``sum(out * seed)``.

    Built with grad mode on, so a backward started under ``no_grad`` still
    reaches ``out``'s graph.  ``keep_seed_graph`` keeps a seed's own graph
    attached, so that under ``create_graph`` a gradient is differentiable
    with respect to whatever computed the seed.
    """
    if seed is None:
        return _unwrap(out)
    with enable_grad():
        return _unwrap((out * (seed if keep_seed_graph else seed.detach())).sum())


def backward(
    tensors: Tensor | Sequence[Tensor],
    grad_tensors: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    inputs: list[Tensor] | None = None,
) -> None:
    r"""Compute gradients of ``tensors`` w.r.t. the leaf variables in their graph.

    Top-level entry point that triggers reverse-mode automatic
    differentiation across the computation graph rooted at
    ``tensors``. For every leaf tensor ``x`` reachable from
    ``tensors`` whose ``requires_grad`` is ``True``, this function
    accumulates :math:`\partial \mathcal{L} / \partial x` into
    ``x.grad``, where :math:`\mathcal{L}` is the (possibly weighted)
    sum of the root tensors.

    The chain rule is applied edge-by-edge during a topological
    walk of the graph in reverse order, so each intermediate
    Jacobian-vector product fires exactly once.

    Parameters
    ----------
    tensors : Tensor or list of Tensor
        Root tensors at which the backward pass starts. When more
        than one root is supplied each receives its own seed and the
        contributions are summed at every shared leaf.
    grad_tensors : Tensor, sequence of (Tensor or None), or None, optional
        Seed cotangent vectors, one per root tensor, each of its root's
        shape and device (another real dtype is cast to the root's).
        A single tensor stands for a one-element sequence.  ``None`` —
        for the whole argument or one entry — is an implicit ``1``,
        allowed only for a real root of one element.
    retain_graph : bool, optional
        If ``True`` the intermediate saved tensors are not freed after
        the backward pass, so the same graph can be traversed again.
        Necessary when calling :func:`backward` multiple times on
        overlapping graphs or when ``create_graph`` is also ``True``.
    create_graph : bool, optional
        If ``True`` the operations performed during backward are
        themselves recorded in the graph, enabling higher-order
        differentiation (e.g. Hessian-vector products, meta-learning).
        Implies stronger memory usage. Defaults to ``False``.
    inputs : list of Tensor or None, optional
        Reserved for the future ability to restrict gradient
        accumulation to a specified subset of leaves. Currently
        unused.

    Returns
    -------
    None
        Gradients are accumulated in-place onto each leaf tensor's
        ``.grad`` attribute. Existing ``.grad`` values are added to,
        not overwritten — call :meth:`Tensor.zero_grad` (or the
        optimizer's ``zero_grad``) between successive backward passes
        if accumulation is undesired.

    Raises
    ------
    RuntimeError
        A root does not require grad, the number of gradients differs
        from the number of roots, or a non-scalar root has no gradient.
    TypeError
        A gradient is neither a tensor nor ``None``.
    ShapeMismatch, DtypeMismatch, DeviceMismatch
        A gradient's shape or device differs from its root's, or only
        one of the two is complex.

    Notes
    -----
    Reverse-mode AD computes

    .. math::

        \frac{\partial \mathcal{L}}{\partial x}
        = \sum_{i}
            \left(
                \frac{\partial \mathcal{L}}{\partial t_i}
            \right)^{\!\top}
            \frac{\partial t_i}{\partial x},

    propagating cotangents :math:`\bar t_i = \partial \mathcal{L} /
    \partial t_i` from the roots through each saved op contract
    until every reachable leaf has received its contribution.

    Memory/compute trade-off:

    * ``retain_graph=False`` (default) is the cheapest mode — once
      the walk finishes, every saved tensor is freed.
    * ``retain_graph=True, create_graph=False`` keeps activations
      so the same graph can be traversed again.
    * ``create_graph=True`` additionally records the backward ops
      in a new graph, doubling memory in the worst case but
      enabling :math:`\nabla^2 \mathcal{L}` and beyond.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import backward
    >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = (x * x).sum()
    >>> backward(y)
    >>> x.grad
    tensor([2., 4., 6.])
    """
    roots = _as_tensors(tensors)
    for i, t in enumerate(roots):
        if not t.requires_grad:
            # Returning quietly turned "this loss is not connected to
            # anything trainable" into a training loop that never moved a
            # parameter.
            raise RuntimeError(
                f"backward(): element {i} of tensors does not require grad "
                "and has no grad_fn, so nothing upstream of it can receive "
                "a gradient. If it is a loss, check that it was not computed "
                "under lucid.no_grad() and that every op on its path tracks "
                "gradients (requires_grad is True on its inputs)."
            )
    seeds = _make_seeds("backward", roots, grad_tensors)
    for root, seed in zip(roots, seeds, strict=True):
        impl = _seeded_root(root, seed, keep_seed_graph=False)
        _refuse_trace()
        _C_engine.engine_backward(
            impl, retain_graph=retain_graph, create_graph=create_graph
        )


def grad(
    outputs: Tensor | Sequence[Tensor],
    inputs: Tensor | Sequence[Tensor],
    grad_outputs: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
) -> tuple[Tensor | None, ...]:
    r"""Compute gradients of outputs w.r.t. inputs, returning them as a tuple.

    The "functional" gradient interface — invoke once to get the partial
    derivatives back without touching ``.grad`` on any leaf tensor.
    Useful for higher-order differentiation, gradient-based meta-learning,
    or any pattern where you want to use the gradients as input to a new
    computation rather than to update parameters in-place.

    Parameters
    ----------
    outputs : Tensor or list of Tensor
        Output tensors to differentiate.  Each must have ``requires_grad``
        set in the graph that produced it.
    inputs : Tensor or list of Tensor
        Input tensors w.r.t. which gradients are requested.  Each must be
        a leaf (or non-leaf with ``requires_grad=True`` if you want grads
        flowing into intermediate nodes).
    grad_outputs : Tensor, sequence of (Tensor or None), or None, optional
        Seed gradients :math:`\partial \mathcal{L} / \partial \text{outputs}`,
        one per output, each of its output's shape and device (another
        real dtype is cast to the output's).  A single tensor stands for
        a one-element sequence.  ``None`` — for the whole argument or one
        entry — is an implicit ``1``, allowed only for a real output of
        one element.
    retain_graph : bool, optional
        Keep the autograd graph alive after this call so additional
        backward passes are possible.  Defaults to ``create_graph``.
    create_graph : bool, optional
        If ``True``, build the autograd graph of the gradient itself so
        the returned tensors are differentiable — used by
        :func:`gradgradcheck` and other higher-order recipes.
    only_inputs : bool, optional
        Reserved for reference-framework compatibility; gradients are
        always restricted to the requested ``inputs``.
    allow_unused : bool, optional
        If ``True``, return ``None`` for any ``inputs`` entry that lies
        outside the computation graph of ``outputs``.  Otherwise raise.

    Returns
    -------
    tuple[Tensor or None, ...]
        One gradient per element of ``inputs``, in the same order.
        Entries are ``None`` only when ``allow_unused=True`` and the
        input is disconnected from ``outputs``.

    Raises
    ------
    RuntimeError
        The number of ``grad_outputs`` differs from the number of
        outputs, a non-scalar output has no seed, or (without
        ``allow_unused``) an input is unreachable from ``outputs``.
    TypeError
        A seed is neither a tensor nor ``None``.
    ShapeMismatch, DtypeMismatch, DeviceMismatch
        A seed's shape or device differs from its output's, or only one
        of the two is complex.

    Notes
    -----
    Mathematically, ``grad`` computes the vector-Jacobian product

    .. math::

        \frac{\partial}{\partial \mathbf{x}}
        \left(\sum_k \text{grad\_outputs}_k \cdot \text{outputs}_k\right)

    via one reverse-mode pass.  Unlike :meth:`Tensor.backward`, it does
    NOT accumulate into ``.grad`` — leaf tensors' existing ``.grad``
    values are preserved across the call.  For chained gradient
    computations (Hessian-vector products, MAML inner loops, etc.) this
    is the right primitive.

    Examples
    --------
    Scalar output — no seed needed:

    >>> import lucid
    >>> from lucid.autograd import grad
    >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = (x * x).sum()
    >>> (gx,) = grad(y, [x])
    >>> gx                         # equals 2 * x
    tensor([2., 4., 6.])

    Vector output — explicit seed:

    >>> z = x * x
    >>> seed = lucid.ones_like(z)
    >>> (gx,) = grad(z, [x], grad_outputs=[seed])

    Higher-order with ``create_graph=True``:

    >>> y = (x ** 3).sum()
    >>> (g,) = grad(y, [x], create_graph=True)
    >>> (gg,) = grad(g.sum(), [x])    # second derivative: 6x
    >>> gg
    tensor([6., 12., 18.])
    """
    outs = _as_tensors(outputs)
    ins = _as_tensors(inputs)
    seeds = _make_seeds("grad", outs, grad_outputs)

    _retain = retain_graph if retain_graph is not None else create_graph

    # One engine call per output, summing the contributions.  The engine
    # returns the gradients rather than accumulating them, so no tensor's
    # ``.grad`` is read or written — not the requested inputs', and not any
    # other leaf's.
    totals: list[Tensor | None] = [None] * len(ins)
    impls = [_unwrap(inp) for inp in ins]
    for index, (out, seed) in enumerate(zip(outs, seeds, strict=True)):
        # Under ``create_graph`` the seed stays attached: the returned
        # gradient is ``J^T seed``, and a seed computed from something
        # upstream (the double-backward form of a JVP, a learned weighting)
        # must be differentiable through it.
        root = _seeded_root(out, seed, keep_seed_graph=create_graph)
        # Every output but the last needs the graph kept for the next call.
        keep = _retain or index < len(outs) - 1
        _refuse_trace()
        partials = _C_engine.engine_grad(root, impls, None, keep, create_graph)
        for slot, impl in enumerate(partials):
            if impl is None:
                continue
            piece = _wrap(impl)
            current = totals[slot]
            totals[slot] = piece if current is None else current + piece

    if not allow_unused and any(g is None for g in totals):
        raise RuntimeError(
            "One of the differentiated tensors does not require grad "
            "and is not reachable from outputs. "
            "Set allow_unused=True to suppress this error."
        )

    return tuple(totals)
