"""
Numerical gradient checker — implemented entirely with the Lucid C++ engine.

No numpy is used.  Finite differences are built by perturbing elements
via scatter_add, computing forward passes through the engine, and comparing
analytical gradients from backward() with the numerical Jacobian columns.
"""

from typing import Callable, Sequence
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap, _unwrap
from lucid._tensor.tensor import Tensor


def gradcheck(
    func: Callable[..., "Tensor | tuple[Tensor, ...]"],
    inputs: "Sequence[Tensor]",
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """Compare analytical gradients from backward() against finite-difference Jacobians.

    Implemented entirely with the Lucid C++ engine — no numpy.

    Parameters
    ----------
    func : callable
        Scalar-valued function mapping Tensor inputs to a single-element Tensor.
    inputs : sequence of Tensor
        Inputs to func that require grad (must be float tensors).
    eps : float
        Finite-difference step size (default 1e-6).
    atol, rtol : float
        Tolerances for the gradient comparison.
    raise_exception : bool
        Raise AssertionError on mismatch (default True), else return False.

    Returns
    -------
    bool  True if all gradients agree within tolerance.
    """

    def _clone_leaf(t: Tensor) -> Tensor:
        """Deep-copy t into a new leaf tensor with requires_grad=True."""
        return _wrap(_C_engine.contiguous(_unwrap(t)).clone_with_grad(True))

    def _detach_copy(t: Tensor) -> Tensor:
        """Deep-copy t as a non-grad leaf (for building perturbed inputs)."""
        return _wrap(_C_engine.contiguous(_unwrap(t)).clone_with_grad(False))

    # ── Analytical gradients ─────────────────────────────────────────────────
    inputs_clone = [_clone_leaf(t) for t in inputs]
    out = func(*inputs_clone)
    if _unwrap(out).numel() != 1:
        raise ValueError(
            f"gradcheck requires a scalar-valued function "
            f"(got shape {tuple(_unwrap(out).shape)})"
        )
    out.backward()

    analytical: list[Tensor] = []
    for t in inputs_clone:
        g = t.grad
        if g is None:
            raise RuntimeError("gradcheck: an input has no gradient after backward()")
        # Cast to F64 for precise comparison.
        g64 = _wrap(_C_engine.astype(_unwrap(g), _C_engine.F64))
        analytical.append(g64)

    # ── Numerical gradients (central differences) ────────────────────────────
    numerical: list[Tensor] = []
    inputs_base = [_detach_copy(t) for t in inputs]  # clean copies for each sweep

    for inp_idx, inp in enumerate(inputs):
        dev = _unwrap(inp).device
        numel = _unwrap(inp).numel()
        shape = list(_unwrap(inp).shape)
        # Work in F64 for numerical stability.
        flat64 = _C_engine.astype(
            _C_engine.reshape(_unwrap(inp), [numel]),
            _C_engine.F64,
        )

        grad_values: list[float] = []

        for k in range(numel):
            # Perturbation vector: eps at position k, 0 elsewhere.
            e_k = _C_engine.zeros([numel], _C_engine.F64, dev)
            k_idx = _C_engine.full([1], float(k), _C_engine.I32, dev)
            eps_v = _C_engine.full([1], float(eps), _C_engine.F64, dev)
            e_k = _C_engine.scatter_add(e_k, k_idx, eps_v, 0)

            # Build perturbed inputs list.
            # Keep all perturbed tensors in F64 so the denominator eps is exact.
            # Casting back to the original dtype would round the perturbation,
            # making the effective step ≠ eps and corrupting the gradient estimate.
            def _make_inputs_f64(sign: float) -> list[Tensor]:
                result = []
                for j, base in enumerate(inputs_base):
                    if j == inp_idx:
                        flat_b = _C_engine.astype(
                            _C_engine.reshape(_unwrap(base), [numel]),
                            _C_engine.F64,
                        )
                        if sign > 0:
                            perturbed_flat = _C_engine.add(flat_b, e_k)
                        else:
                            perturbed_flat = _C_engine.sub(flat_b, e_k)
                        # Reshape to original shape; stay in F64 for accuracy.
                        perturbed = _C_engine.reshape(perturbed_flat, shape)
                        result.append(_wrap(perturbed))
                    else:
                        # Other inputs: cast to F64 for consistent dtype.
                        other_f64 = _C_engine.astype(
                            _C_engine.contiguous(_unwrap(inputs[j])),
                            _C_engine.F64,
                        )
                        result.append(_wrap(other_f64))
                return result

            f_plus = float(func(*_make_inputs_f64(+1.0)).item())
            f_minus = float(func(*_make_inputs_f64(-1.0)).item())
            grad_values.append((f_plus - f_minus) / (2.0 * eps))

        # Build numerical gradient tensor from Python list (interop boundary).
        from lucid._factories.converters import tensor as _tensor_fn

        num_t = _tensor_fn(grad_values)
        num64 = _wrap(
            _C_engine.astype(
                _C_engine.reshape(_unwrap(num_t), shape),
                _C_engine.F64,
            )
        )
        numerical.append(num64)

    # ── Compare analytical vs numerical ──────────────────────────────────────
    for i, (an_t, nu_t) in enumerate(zip(analytical, numerical)):
        an = _unwrap(an_t)
        nu = _unwrap(nu_t)
        diff = _C_engine.abs(_C_engine.sub(an, nu))
        thresh = _C_engine.add(
            _C_engine.full(list(diff.shape), atol, _C_engine.F64, diff.device),
            _C_engine.mul(
                _C_engine.full(list(diff.shape), rtol, _C_engine.F64, diff.device),
                _C_engine.abs(nu),
            ),
        )
        ok = bool(_wrap(_C_engine.all(_C_engine.less_equal(diff, thresh))).item())
        if not ok:
            max_diff = float(_wrap(_C_engine.max(diff, [], False)).item())
            max_thresh = float(_wrap(_C_engine.max(thresh, [], False)).item())
            diff_flat = _C_engine.reshape(diff, [diff.numel()])
            worst_k = int(_wrap(_C_engine.argmax(diff_flat, 0, False)).item())
            k_idx2 = _C_engine.full([1], float(worst_k), _C_engine.I32, diff.device)
            an_flat = _C_engine.reshape(an, [an.numel()])
            nu_flat = _C_engine.reshape(nu, [nu.numel()])
            an_val = float(_wrap(_C_engine.gather(an_flat, k_idx2, 0)).item())
            nu_val = float(_wrap(_C_engine.gather(nu_flat, k_idx2, 0)).item())

            msg = (
                f"Gradient check failed for input {i} "
                f"(atol={atol}, rtol={rtol}):\n"
                f"  Max |diff|    = {max_diff:.6g}\n"
                f"  Max threshold = {max_thresh:.6g}\n"
                f"  Worst element [flat {worst_k}]: "
                f"analytical={an_val:.6g}, numerical={nu_val:.6g}, "
                f"|diff|={abs(an_val - nu_val):.6g}"
            )
            if raise_exception:
                raise AssertionError(msg)
            return False

    return True


def gradgradcheck(
    func: Callable[..., "Tensor | tuple[Tensor, ...]"],
    inputs: "Sequence[Tensor]",
    grad_outputs: "Sequence[Tensor] | None" = None,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """Verify second-order gradients by gradchecking the gradient itself.

    Wraps ``func`` so its scalar output produces a first-order gradient
    sum, then runs ``gradcheck`` on that wrapped function.  This catches
    errors in custom ``Function.backward`` implementations that only show
    up at the second-derivative level.

    The signature mirrors ``reference framework.autograd.gradgradcheck`` so
    user code that imports ``from lucid.autograd import gradgradcheck``
    works the same way.  ``grad_outputs`` is currently ignored — we always
    use ``ones_like`` upstream gradients, matching the most common use.
    """
    import lucid

    if grad_outputs is not None:
        # The reference framework allows custom upstream gradients; we
        # accept the kwarg for source compatibility but the simple
        # ones-grad path is always sufficient for verifying ``backward``.
        # A future pass can wire the supplied tensors through.
        pass

    def _scalar_grad_fn(*args: Tensor) -> Tensor:
        # gradcheck builds finite-difference inputs that don't have
        # ``requires_grad`` set, so re-enable it on fresh leaf copies before
        # we differentiate.  The leaves are the things the outer gradcheck
        # is finite-differencing against — ``func`` runs on them, the
        # first-order grad is taken with ``create_graph=True``, and the
        # returned scalar depends on them through the grad-of-grad graph.
        leaves: list[Tensor] = []
        for a in args:
            if isinstance(a, Tensor):
                if not a.requires_grad:
                    a = a.detach().requires_grad_(True)
                leaves.append(a)
            else:
                leaves.append(a)  # non-tensor passthrough
        out = func(*leaves)
        scalar: Tensor = (
            out.sum() if isinstance(out, Tensor) else sum(o.sum() for o in out)
        )
        tensor_leaves = [a for a in leaves if isinstance(a, Tensor)]
        grads = lucid.autograd.grad(
            scalar, tensor_leaves, create_graph=True, retain_graph=True
        )
        return sum(g.sum() for g in grads)

    return gradcheck(
        _scalar_grad_fn,
        inputs,
        eps=eps,
        atol=atol,
        rtol=rtol,
        raise_exception=raise_exception,
    )
