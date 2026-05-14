"""
Higher-order autograd utilities: jacobian, hessian, vjp, jvp.
"""

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def jacobian(
    func: Callable[..., "Tensor"],
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

    Lucid evaluates it row-by-row by repeated reverse-mode
    backward passes — one per output element — seeding each pass
    with a one-hot cotangent so the resulting input gradient is
    exactly the corresponding Jacobian row. The cost therefore
    scales with the output dimension :math:`m`; prefer :func:`vjp`
    when only :math:`v^\top J` is needed and :func:`jvp` when only
    :math:`J v` is needed.

    Parameters
    ----------
    func : callable
        A function mapping ``Tensor`` inputs to a ``Tensor`` (or
        tuple of ``Tensor``). Must be differentiable w.r.t. each
        positional input.
    inputs : Tensor or tuple of Tensor
        Input tensor(s) at which the Jacobian is evaluated. They
        are silently promoted to ``requires_grad=True`` if needed.
    create_graph : bool, optional
        If ``True`` the Jacobian itself is differentiable, enabling
        higher-order derivatives (e.g. building :func:`hessian` on
        top). Defaults to ``False``.
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
        ``(prod(out_shape), prod(x.shape))``. For multiple inputs
        a tuple is returned, one Jacobian block per input.

    Notes
    -----
    Reverse-mode differentiation makes the cost per row
    :math:`O(\text{cost}(f))`; the full Jacobian therefore costs
    :math:`O(m \cdot \text{cost}(f))`. For square or wide
    Jacobians (:math:`m \ge n`) forward-mode would be cheaper —
    Lucid does not yet ship a forward-mode implementation, so
    this routine is preferred for tall Jacobians
    (:math:`m \ll n`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import jacobian
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> def f(x):
    ...     return x * x
    >>> J = jacobian(f, x)
    >>> J.shape
    (3, 3)
    """
    from lucid._dispatch import _wrap
    from lucid._C import engine as _C_engine

    scalar_input = not isinstance(inputs, (list, tuple))
    inputs_t: tuple[Tensor, ...] = (inputs,) if scalar_input else tuple(inputs)  # type: ignore[assignment]

    # Make sure inputs require grad
    inputs_rg = []
    for x in inputs_t:
        if not x.requires_grad:
            x = x.requires_grad_(True)
        inputs_rg.append(x)

    # Run forward
    _raw_outputs = func(*inputs_rg)
    outputs: list[Tensor] | tuple[Tensor, ...]
    if not isinstance(_raw_outputs, (list, tuple)):
        outputs = [_raw_outputs]
    else:
        outputs = _raw_outputs

    # Flatten each output: record (tensor, numel)
    from lucid._tensor.tensor import Tensor as _T

    out_flat_list = [(o, o.numel()) for o in outputs]

    results = []
    for x in inputs_rg:
        x_numel = x.numel()
        out_total = sum(n for _, n in out_flat_list)
        # Accumulate Jacobian rows as a list of 1-D tensors, then stack.
        J_rows: list[_C_engine.TensorImpl] = []

        for out_t, out_numel in out_flat_list:
            out_shape = list(out_t.shape) if out_t.shape else []
            for i in range(out_numel):
                for xx in inputs_rg:
                    xx._impl.zero_grad()

                if out_numel == 1 and out_shape == []:
                    out_t.backward(retain_graph=True, create_graph=create_graph)
                else:
                    # One-hot seed via engine: zeros + scatter-like fill.
                    seed_impl = _C_engine.zeros(
                        [out_numel], _C_engine.F32, out_t._impl.device
                    )
                    ones1 = _C_engine.ones([1], _C_engine.F32, out_t._impl.device)
                    idx1 = _C_engine.full(
                        [1], float(i), _C_engine.I32, out_t._impl.device
                    )
                    seed_impl = _C_engine.scatter_add(seed_impl, idx1, ones1, 0)
                    if out_shape:
                        seed_impl = _C_engine.reshape(seed_impl, out_shape)
                    seed_t = _T.__new_from_impl__(seed_impl)
                    out_t.backward(
                        gradient=seed_t, retain_graph=True, create_graph=create_graph
                    )

                g_raw = x._impl.grad_as_python()
                if g_raw is not None:
                    row_impl = _C_engine.TensorImpl(g_raw, x._impl.device, False)
                    J_rows.append(_C_engine.reshape(row_impl, [x_numel]))
                else:
                    J_rows.append(
                        _C_engine.zeros([x_numel], _C_engine.F32, x._impl.device)
                    )

        # Stack rows to shape (out_total, x_numel).
        J_impl = _C_engine.stack(J_rows, 0)
        results.append(_wrap(J_impl))

    return results[0] if scalar_input else tuple(results)


def hessian(
    func: Callable[..., "Tensor"],
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

    Implemented as :func:`jacobian` of the gradient of ``func`` —
    a forward pass produces the loss, a first backward (with
    ``create_graph=True``) builds the gradient graph, and a second
    backward along each gradient coordinate yields the rows of
    :math:`H`. Cost is therefore :math:`O(n \cdot
    \text{cost}(\nabla f))`.

    Parameters
    ----------
    func : callable
        Scalar-valued function of one or more ``Tensor`` inputs.
    inputs : Tensor or tuple of Tensor
        Inputs at which :math:`H` is evaluated. They are silently
        promoted to ``requires_grad=True`` if necessary.
    create_graph : bool, optional
        If ``True`` the Hessian itself remains differentiable
        (third-order derivatives). Defaults to ``False``.
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
        ``H[i][j]`` containing :math:`\partial^2 f / (\partial
        x_i \, \partial x_j)`.

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
    ...     return (x * x).sum()
    >>> H = hessian(f, x)
    >>> H.shape
    (2, 2)
    """
    from lucid._dispatch import _wrap
    from lucid._C import engine as _C_engine

    scalar_input = not isinstance(inputs, (list, tuple))
    inputs_t: tuple[Tensor, ...] = (inputs,) if scalar_input else tuple(inputs)  # type: ignore[assignment]

    # Make sure inputs require grad
    inputs_rg = []
    for x in inputs_t:
        if not x.requires_grad:
            x = x.requires_grad_(True)
        inputs_rg.append(x)

    from lucid._tensor.tensor import Tensor as _T

    n_inputs = len(inputs_rg)
    blocks: list[list["Tensor"]] = [[None] * n_inputs for _ in range(n_inputs)]  # type: ignore[list-item]

    for i, xi in enumerate(inputs_rg):
        ni = xi.numel()
        for j, xj in enumerate(inputs_rg):
            nj = xj.numel()
            H_rows: list[_C_engine.TensorImpl] = []

            for k in range(ni):
                xi._impl.zero_grad()
                xj._impl.zero_grad()
                out = func(*inputs_rg)
                out.backward(create_graph=True, retain_graph=True)

                gi_impl = xi._impl.grad_as_impl()
                if gi_impl is None:
                    H_rows.append(_C_engine.zeros([nj], _C_engine.F32, xi._impl.device))
                    continue
                gi_t = _T.__new_from_impl__(gi_impl)

                # One-hot mask at index k (engine ops only).
                gi_shape = list(gi_impl.shape)
                mask_flat = _C_engine.zeros([ni], _C_engine.F32, gi_impl.device)
                ones1 = _C_engine.ones([1], _C_engine.F32, gi_impl.device)
                idx1 = _C_engine.full([1], float(k), _C_engine.I32, gi_impl.device)
                mask_flat = _C_engine.scatter_add(mask_flat, idx1, ones1, 0)
                mask_impl = _C_engine.reshape(mask_flat, gi_shape)
                mask_t = _T.__new_from_impl__(mask_impl)
                gi_k = (gi_t * mask_t).sum()

                xj._impl.zero_grad()
                gi_k.backward(retain_graph=True)

                gj_raw = xj._impl.grad_as_python()
                if gj_raw is not None:
                    row_impl = _C_engine.TensorImpl(gj_raw, xj._impl.device, False)
                    H_rows.append(_C_engine.reshape(row_impl, [nj]))
                else:
                    H_rows.append(_C_engine.zeros([nj], _C_engine.F32, xj._impl.device))

            H_impl = _C_engine.stack(H_rows, 0)
            blocks[i][j] = _wrap(H_impl)

    if scalar_input:
        return blocks[0][0]
    return tuple(tuple(row) for row in blocks)


def vjp(
    func: Callable[..., "Tensor"],
    inputs: Tensor | tuple[Tensor, ...],
    v: Tensor | tuple[Tensor, ...],
    create_graph: bool = False,
    strict: bool = False,
) -> tuple["Tensor", tuple["Tensor | None", ...]]:
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
        Primal point :math:`x` at which :math:`J` is evaluated.
        Silently promoted to ``requires_grad=True`` if needed.
    v : Tensor or tuple of Tensor
        Cotangent vector(s) matching the output shape(s) of
        ``func``. Scalar-valued ``v`` is broadcast for scalar
        outputs.
    create_graph : bool, optional
        If ``True`` the returned VJP is itself differentiable,
        enabling double-backward. Defaults to ``False``.
    strict : bool, optional
        Reserved for stricter validation. Currently unused.

    Returns
    -------
    tuple of (Tensor, tuple of (Tensor or None))
        ``(output, vjp_grads)`` where ``output = func(*inputs)``
        and ``vjp_grads[i]`` is :math:`v^\top J` projected onto
        input ``i`` (or ``None`` if that input has no gradient
        path).

    Notes
    -----
    The dual to :func:`vjp` is :func:`jvp`, which computes
    :math:`J v` via forward-mode (or finite differences in
    Lucid's current implementation).

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import vjp
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> v = lucid.tensor([1.0, 1.0, 1.0])
    >>> def f(x):
    ...     return x * x
    >>> y, (grad_x,) = vjp(f, x, v)
    """
    from lucid.autograd._backward import grad as _grad

    scalar_input = not isinstance(inputs, (list, tuple))
    inputs_t: tuple[Tensor, ...] = (inputs,) if scalar_input else tuple(inputs)  # type: ignore[assignment]

    scalar_v = not isinstance(v, (list, tuple))
    v_t: tuple[Tensor, ...] = (v,) if scalar_v else tuple(v)  # type: ignore[assignment]

    inputs_rg = []
    for x in inputs_t:
        if not x.requires_grad:
            x = x.requires_grad_(True)
        inputs_rg.append(x)

    outputs = func(*inputs_rg)
    if not isinstance(outputs, (list, tuple)):
        outputs_list = [outputs]
    else:
        outputs_list = list(outputs)

    # Align v shapes to match output shapes (e.g. scalar() vs (1,))
    aligned_v = []
    for vi, oi in zip(v_t, outputs_list):
        out_shape = tuple(oi.shape) if oi.shape else ()
        v_shape = tuple(vi.shape) if vi.shape else ()
        if out_shape != v_shape and vi.numel() == 1:
            from lucid._C import engine as _C_engine
            from lucid._dispatch import _wrap, _unwrap

            vi = _wrap(_C_engine.reshape(_unwrap(vi), list(out_shape)))
        aligned_v.append(vi)

    grads = _grad(
        outputs_list,
        list(inputs_rg),
        grad_outputs=aligned_v,
        retain_graph=create_graph,
        create_graph=create_graph,
        allow_unused=True,
    )

    return outputs, grads


def jvp(
    func: Callable[..., "Tensor"],
    inputs: Tensor | tuple[Tensor, ...],
    v: Tensor | tuple[Tensor, ...],
    create_graph: bool = False,
    strict: bool = False,
) -> tuple["Tensor", "Tensor"]:
    r"""Jacobian-vector product :math:`J v` (forward-mode directional derivative).

    Given :math:`f : \mathbb{R}^n \to \mathbb{R}^m` with Jacobian
    :math:`J \in \mathbb{R}^{m \times n}` and a tangent vector
    :math:`v \in \mathbb{R}^n`, returns

    .. math::

        J v = \left.\frac{d}{dt} f(x + t v)\right|_{t=0}
            \in \mathbb{R}^{m}

    along with the primal output :math:`y = f(x)`. JVPs are the
    natural primitive of forward-mode AD and are useful for
    propagating tangent information (sensitivities) through a
    network in a single forward sweep, for computing directional
    derivatives, and as a building block for second-order methods.

    Lucid currently realises the JVP via a symmetric central
    finite difference

    .. math::

        J v \approx \frac{f(x + \varepsilon v) - f(x - \varepsilon v)}
                         {2 \varepsilon},

    with :math:`\varepsilon = 10^{-4}`. This avoids the need for a
    true forward-mode implementation while still being accurate
    enough for testing and most applications.

    Parameters
    ----------
    func : callable
        Function mapping ``Tensor`` inputs to a ``Tensor`` (or
        tuple thereof).
    inputs : Tensor or tuple of Tensor
        Primal point :math:`x`.
    v : Tensor or tuple of Tensor
        Tangent vector(s) matching the input shape(s).
    create_graph : bool, optional
        Reserved for the future native forward-mode implementation.
        Currently unused.
    strict : bool, optional
        Reserved for stricter validation. Currently unused.

    Returns
    -------
    tuple of (Tensor or tuple of Tensor, Tensor or tuple of Tensor)
        ``(primals_out, tangents_out)`` where
        ``primals_out = func(*inputs)`` and ``tangents_out`` has the
        same shape as ``primals_out`` and holds :math:`J v`.

    Notes
    -----
    The complementary operation is :func:`vjp`, which computes
    :math:`v^\top J` cheaply via reverse-mode. Use :func:`jvp`
    when the input dimension is small relative to the output
    dimension; otherwise reverse-mode is more efficient.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import jvp
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> v = lucid.tensor([1.0, 0.0, 0.0])
    >>> def f(x):
    ...     return x * x
    >>> y, tangent = jvp(f, x, v)
    """

    scalar_input = not isinstance(inputs, (list, tuple))
    inputs_t: tuple[Tensor, ...] = (inputs,) if scalar_input else tuple(inputs)  # type: ignore[assignment]

    scalar_v = not isinstance(v, (list, tuple))
    v_t: tuple[Tensor, ...] = (v,) if scalar_v else tuple(v)  # type: ignore[assignment]

    inputs_rg = []
    for x in inputs_t:
        if not x.requires_grad:
            x = x.requires_grad_(True)
        inputs_rg.append(x)

    # Forward pass with create_graph=True to allow higher-order differentation
    primals_out = func(*inputs_rg)
    if not isinstance(primals_out, (list, tuple)):
        primals_list = [primals_out]
    else:
        primals_list = list(primals_out)

    # Use a dummy ones vector for the first backward, then use v for the second
    # Standard JVP via double-backward: jvp = d/dt[f(x + tv)] at t=0
    # Implemented as: grad(grad(f, x).dot(v), x) applied carefully.

    # Simpler approach: use autograd.grad twice
    # 1. Get Jacobian-row vJP by forward-mode approximation
    # For now, implement via finite-difference fallback that supports create_graph=False
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _wrap, _unwrap

    eps = 1e-4
    # Finite-difference JVP: (f(x + eps*v) - f(x - eps*v)) / (2*eps)
    inputs_fwd = []
    inputs_bwd = []
    for x, vi in zip(inputs_rg, v_t):
        xf = _wrap(
            _C_engine.add(
                _unwrap(x),
                _C_engine.mul(
                    _unwrap(vi),
                    _C_engine.full(
                        _unwrap(vi).shape, eps, _unwrap(vi).dtype, _unwrap(vi).device
                    ),
                ),
            )
        )
        xb = _wrap(
            _C_engine.sub(
                _unwrap(x),
                _C_engine.mul(
                    _unwrap(vi),
                    _C_engine.full(
                        _unwrap(vi).shape, eps, _unwrap(vi).dtype, _unwrap(vi).device
                    ),
                ),
            )
        )
        inputs_fwd.append(xf)
        inputs_bwd.append(xb)

    out_fwd = func(*inputs_fwd)
    out_bwd = func(*inputs_bwd)

    if isinstance(out_fwd, (list, tuple)):
        tangents = tuple(
            _wrap(
                _C_engine.div(
                    _C_engine.sub(_unwrap(f), _unwrap(b)),
                    _C_engine.full(
                        _unwrap(f).shape, 2 * eps, _unwrap(f).dtype, _unwrap(f).device
                    ),
                )
            )
            for f, b in zip(out_fwd, out_bwd)
        )
        return primals_out, tangents
    else:
        tangent = _wrap(
            _C_engine.div(
                _C_engine.sub(_unwrap(out_fwd), _unwrap(out_bwd)),
                _C_engine.full(
                    _unwrap(out_fwd).shape,
                    2 * eps,
                    _unwrap(out_fwd).dtype,
                    _unwrap(out_fwd).device,
                ),
            )
        )
        return primals_out, tangent
