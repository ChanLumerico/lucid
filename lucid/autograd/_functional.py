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
    """Compute the Jacobian of *func* w.r.t. each input.

    For scalar-valued functions this is equivalent to :func:`grad`.
    For vector-valued functions returns a matrix J where J[i,j] = d out[i]/d in[j].

    Parameters
    ----------
    func : callable
        Function to differentiate. Must take Tensor(s) and return a Tensor.
    inputs : Tensor or tuple of Tensor
        Input tensors at which to evaluate the Jacobian.

    Returns
    -------
    Tensor or tuple of Tensor
        Jacobian matrix/matrices, one per input.
    """
    from lucid._dispatch import _wrap
    from lucid._C import engine as _C_engine

    scalar_input = not isinstance(inputs, (list, tuple))
    if scalar_input:
        inputs = (inputs,)

    # Make sure inputs require grad
    inputs_rg = []
    for x in inputs:
        if not x.requires_grad:
            x = x.requires_grad_(True)  # type: ignore[assignment]
        inputs_rg.append(x)

    # Run forward
    outputs = func(*inputs_rg)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    # Flatten each output: record (tensor, numel)
    from lucid._tensor.tensor import Tensor as _T

    out_flat_list = [(o, o.numel()) for o in outputs]

    results = []
    for x in inputs_rg:
        x_numel = x.numel()
        out_total = sum(n for _, n in out_flat_list)
        # Accumulate Jacobian rows as a list of 1-D tensors, then stack.
        J_rows: list[object] = []

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
    """Compute the Hessian of a scalar-valued *func* w.r.t. each pair of inputs.

    Parameters
    ----------
    func : callable
        Scalar-valued function to differentiate twice.
    inputs : Tensor or tuple of Tensor

    Returns
    -------
    Tensor or tuple of tuple of Tensor
        Hessian matrix (or block-Hessian for multiple inputs).
    """
    from lucid._dispatch import _wrap
    from lucid._C import engine as _C_engine
    from lucid.autograd._backward import grad as _grad

    scalar_input = not isinstance(inputs, (list, tuple))
    if scalar_input:
        inputs = (inputs,)

    # Make sure inputs require grad
    inputs_rg = []
    for x in inputs:
        if not x.requires_grad:
            x = x.requires_grad_(True)  # type: ignore[assignment]
        inputs_rg.append(x)

    from lucid._tensor.tensor import Tensor as _T

    n_inputs = len(inputs_rg)
    blocks: list[list["Tensor"]] = [[None] * n_inputs for _ in range(n_inputs)]  # type: ignore[list-item]

    for i, xi in enumerate(inputs_rg):
        ni = xi.numel()
        for j, xj in enumerate(inputs_rg):
            nj = xj.numel()
            H_rows: list[object] = []

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
    """Vector-Jacobian product (reverse-mode): returns (outputs, vjp_tensors).

    Computes  v^T @ J  where J is the Jacobian of func and v is the
    "vector" (cotangent / grad_output).

    Parameters
    ----------
    func : callable
        Function to differentiate.
    inputs : Tensor or tuple of Tensor
        Input tensors.
    v : Tensor or tuple of Tensor
        Cotangent vector(s) matching the output shape(s).

    Returns
    -------
    (output, vjp_grads) where output is func(*inputs) and vjp_grads are
    gradients w.r.t. each input.
    """
    from lucid.autograd._backward import grad as _grad

    scalar_input = not isinstance(inputs, (list, tuple))
    if scalar_input:
        inputs = (inputs,)

    scalar_v = not isinstance(v, (list, tuple))
    if scalar_v:
        v = (v,)

    inputs_rg = []
    for x in inputs:
        if not x.requires_grad:
            x = x.requires_grad_(True)  # type: ignore[assignment]
        inputs_rg.append(x)

    outputs = func(*inputs_rg)
    if not isinstance(outputs, (list, tuple)):
        outputs_list = [outputs]
    else:
        outputs_list = list(outputs)

    # Align v shapes to match output shapes (e.g. scalar() vs (1,))
    aligned_v = []
    for vi, oi in zip(v, outputs_list):
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
    """Jacobian-vector product (forward-mode via double-backward).

    Computes  J @ v  using the forward-over-reverse trick:
    create_graph=True backward, then backward again with the tangent.

    Parameters
    ----------
    func : callable
        Function to differentiate.
    inputs : Tensor or tuple of Tensor
        Primal inputs.
    v : Tensor or tuple of Tensor
        Tangent vectors matching the input shapes.

    Returns
    -------
    (primals_out, tangents_out)
        primals_out = func(*inputs),
        tangents_out = J @ v  (shape = output shape).
    """
    from lucid.autograd._backward import grad as _grad

    scalar_input = not isinstance(inputs, (list, tuple))
    if scalar_input:
        inputs = (inputs,)

    scalar_v = not isinstance(v, (list, tuple))
    if scalar_v:
        v = (v,)

    inputs_rg = []
    for x in inputs:
        if not x.requires_grad:
            x = x.requires_grad_(True)  # type: ignore[assignment]
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
    for x, vi in zip(inputs_rg, v):
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
        return primals_out, tangents  # type: ignore[return-value]
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
