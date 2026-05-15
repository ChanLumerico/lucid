"""
autograd.backward() and autograd.grad() free functions.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def backward(
    tensors: Tensor | list[Tensor],
    grad_tensors: list[Tensor] | None = None,
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
    grad_tensors : list of Tensor or None, optional
        Seed cotangent vectors, one per root tensor and matching the
        corresponding root's shape. Required when a root is
        non-scalar. Defaults to ``ones_like(t)`` for each root, which
        is correct for scalar losses.
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
    Tensor([2., 4., 6.])
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    if grad_tensors is None:
        for t in tensors:
            t.backward(retain_graph=retain_graph, create_graph=create_graph)
    else:
        for t, g in zip(tensors, grad_tensors):
            t.backward(gradient=g, retain_graph=retain_graph, create_graph=create_graph)


def grad(
    outputs: Tensor | list[Tensor],
    inputs: Tensor | list[Tensor],
    grad_outputs: list[Tensor] | None = None,
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
    grad_outputs : list of Tensor, optional
        Seed gradients :math:`\partial \mathcal{L} / \partial \text{outputs}`
        for non-scalar outputs.  If omitted, ``outputs`` is expected to be
        scalar and an implicit ``ones_like`` seed is used.
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
    >>> x = lucid.randn(3, requires_grad=True)
    >>> y = (x * x).sum()
    >>> (gx,) = grad(y, [x])
    >>> gx                         # equals 2 * x
    Tensor([...])

    Vector output — explicit seed:

    >>> z = x * x
    >>> seed = lucid.ones_like(z)
    >>> (gx,) = grad(z, [x], grad_outputs=[seed])

    Higher-order with ``create_graph=True``:

    >>> y = (x ** 3).sum()
    >>> (g,) = grad(y, [x], create_graph=True)
    >>> (gg,) = grad(g.sum(), [x])    # second derivative: 6x
    """
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    # Save existing .grad values so we can restore them after
    saved_grads: list[Tensor | None] = [inp.grad for inp in inputs]
    # Zero out existing grads on inputs so we can read the fresh ones.
    # ``grad_to_tensor`` is the numpy-free probe — returns a TensorImpl
    # for any accumulated grad (graph-mode or detached), or ``None`` when
    # truly absent.  ``grad_as_impl`` alone would be too strict (it only
    # finds graph-mode grads).
    for inp in inputs:
        if inp._impl.grad_to_tensor() is not None:
            inp._impl.zero_grad()

    _retain = retain_graph if retain_graph is not None else create_graph

    # Run backward
    if grad_outputs is None:
        for out in outputs:
            out.backward(retain_graph=_retain, create_graph=create_graph)
    else:
        for out, g in zip(outputs, grad_outputs):
            out.backward(gradient=g, retain_graph=_retain, create_graph=create_graph)

    # Collect computed gradients.  Two numpy-free accessors, used in order:
    #   1. ``grad_as_impl()`` — graph-mode grad with ``grad_fn`` intact
    #      (needed when ``create_graph=True`` so the grad-of-grad graph
    #      isn't severed).  Returns None for detached / non-graph grads.
    #   2. ``grad_to_tensor()`` — wraps the accumulated grad Storage as a
    #      fresh TensorImpl regardless of graph-mode tracking.  Returns
    #      None only when no grad was accumulated.
    # Both replace the prior ``grad_as_python()`` + ``TensorImpl(np.ndarray,
    # ...)`` round-trip, which forced numpy in even for detached grads.
    result: list[Tensor | None] = []
    for inp in inputs:
        g_impl = inp._impl.grad_as_impl()
        if g_impl is None:
            g_impl = inp._impl.grad_to_tensor()
        if g_impl is not None:
            result.append(_wrap(g_impl))
            continue
        if not allow_unused:
            raise RuntimeError(
                "One of the differentiated tensors does not require grad "
                "and is not reachable from outputs. "
                "Set allow_unused=True to suppress this error."
            )
        result.append(None)

    # Restore original .grad values using set_grad (no numpy).
    for inp, saved in zip(inputs, saved_grads):
        inp._impl.zero_grad()
        if saved is not None:
            inp._impl.set_grad(_unwrap(saved))

    return tuple(result)
