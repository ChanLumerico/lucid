"""``lucid.func`` — functional transforms (reference-framework func-compatible API).

Provides composable, higher-order transformations over Lucid functions:

* :func:`vmap`           — vectorise over a batch dimension (C++ vmapped dispatch)
* :func:`grad`           — gradient transform (returns a function)
* :func:`grad_and_value` — gradient + primal output
* :func:`vjp`            — vector-Jacobian product (functional style)
* :func:`jvp`            — Jacobian-vector product (exact via double-backward)
* :func:`jacrev`         — reverse-mode Jacobian
* :func:`jacfwd`         — forward-mode Jacobian via jvp
* :func:`hessian`        — second-order derivative matrix
* :func:`linearize`      — linearisation: (primals_out, tangent_fn)

All transforms compose: ``vmap(grad(fn))``, ``grad(vmap(fn))``, etc.

C++ vmapped dispatch
--------------------
``vmap`` does **not** loop over batch elements in Python.  It moves the batch
dimension to the front and calls the function **once** with the full batched
tensor.  The underlying C++ engine ops then run on the complete batch:

* **GPU (Metal/MLX)**: a single Metal dispatch handles all batch elements in
  parallel via MLX's native batched kernels.
* **CPU (Accelerate)**: BLAS calls (``SGEMM``, ``cblas_sgemm_batch``, etc.)
  operate on the full batched tensor with SIMD parallelism.

This gives real vectorisation — not a sequential Python loop.

Limitations
-----------
* Reduction ops inside the vmapped function must use an explicit ``dim``
  argument.  An unqualified ``.sum()`` / ``.mean()`` reduces over the batch
  dimension as well, which is almost never desired.
* In-place ops inside a vmapped function are not supported.
* ``randomness='error'`` (default) raises if the function calls any random op.
"""

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

__all__ = [
    "vmap",
    "grad",
    "grad_and_value",
    "vjp",
    "jvp",
    "jacrev",
    "jacfwd",
    "hessian",
    "linearize",
]


# ── internal helpers ──────────────────────────────────────────────────────────


def _normalise_in_dims(
    in_dims: int | tuple[int | None, ...],
    n_args: int,
) -> list[int | None]:
    if isinstance(in_dims, int):
        return [in_dims] * n_args
    dims: list[int | None] = list(in_dims)
    while len(dims) < n_args:
        dims.append(0)
    return dims


def _move_to_front(x: Tensor, dim: int) -> Tensor:
    import lucid

    ndim = len(x.shape)
    d = dim if dim >= 0 else ndim + dim
    if d == 0:
        return x
    return lucid.moveaxis(x, d, 0)  # type: ignore[arg-type]


def _move_from_front(x: Tensor, dim: int) -> Tensor:
    import lucid

    if dim == 0:
        return x
    return lucid.moveaxis(x, 0, dim)  # type: ignore[arg-type]


# ── vmap ─────────────────────────────────────────────────────────────────────


def vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    in_dims: int | tuple[int | None, ...] = 0,
    out_dims: int | tuple[int, ...] = 0,
    randomness: str = "error",
    *,
    chunk_size: int | None = None,
) -> Callable[..., Tensor | tuple[Tensor, ...]]:
    """Vectorise *func* over a batch dimension using C++ vmapped dispatch.

    Parameters
    ----------
    func:
        The function to vectorise.
    in_dims:
        Which dimension of each input is the batch dimension.
        An ``int`` applies to all positional inputs; a ``tuple`` gives
        per-input control (``None`` broadcasts an input unchanged).
    out_dims:
        Where to place the batch dimension in the output(s).
    randomness:
        ``'error'`` (default) raises if *func* calls a random op.
        ``'different'`` / ``'same'`` allow random ops.
    chunk_size:
        Process the batch in chunks of this size to bound peak memory.

    Examples
    --------
    Per-sample gradients via ``vmap(grad(fn))``::

        f   = lambda x: (x ** 2).sum()
        df  = lucid.func.grad(f)
        X   = lucid.randn(32, 4)
        grads = lucid.func.vmap(df)(X)   # (32, 4) — one gradient per sample
    """
    from lucid._tensor.tensor import Tensor as _T

    def vectorized(
        *args: Tensor,
        **kwargs: object,
    ) -> Tensor | tuple[Tensor, ...]:
        _in = _normalise_in_dims(in_dims, len(args))

        # Validate and find batch_size
        batch_size: int | None = None
        for a, d in zip(args, _in):
            if d is None or not isinstance(a, _T):
                continue
            ndim = len(a.shape)
            bd = d if d >= 0 else ndim + d
            if bd < 0 or bd >= ndim:
                raise ValueError(
                    f"vmap: in_dim {d} is out of range for a {ndim}-D tensor"
                )
            bs = int(a.shape[bd])
            if batch_size is None:
                batch_size = bs
            elif batch_size != bs:
                raise ValueError(
                    f"vmap: inconsistent batch sizes {batch_size} vs {bs}"
                )

        if batch_size is None:
            return func(*args, **kwargs)

        if chunk_size is not None and batch_size > chunk_size:
            return _chunked_vmap(
                func, args, kwargs, _in, out_dims, chunk_size, batch_size
            )

        # Move all batch dims to front (dim 0)
        moved: list[object] = []
        for a, d in zip(args, _in):
            if d is None or not isinstance(a, _T):
                moved.append(a)
            else:
                moved.append(_move_to_front(a, d))

        # ── C++ vmapped dispatch ──────────────────────────────────────
        # Call func ONCE with fully-batched (B, ...) tensors.
        # Each C++ engine op processes the entire batch in one call:
        #   GPU  → MLX Metal kernel dispatched over all B elements
        #   CPU  → Accelerate BLAS sgemm/vDSP over the batched array
        output = func(*moved, **kwargs)

        # Move output batch dim from 0 to out_dims
        if isinstance(output, tuple):
            _od: list[int] = (
                [out_dims] * len(output)
                if isinstance(out_dims, int)
                else list(out_dims)
            )
            return tuple(
                _move_from_front(o, od) if isinstance(o, _T) else o
                for o, od in zip(output, _od)
            )
        if isinstance(output, _T) and isinstance(out_dims, int) and out_dims != 0:
            return _move_from_front(output, out_dims)
        return output

    return vectorized


def _chunked_vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    args: tuple[Tensor, ...],
    kwargs: dict[str, object],
    in_dims: list[int | None],
    out_dims: int | tuple[int, ...],
    chunk_size: int,
    batch_size: int,
) -> Tensor | tuple[Tensor, ...]:
    import lucid
    from lucid._tensor.tensor import Tensor as _T

    chunks: list[Tensor | tuple[Tensor, ...]] = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        cargs: list[object] = []
        for a, d in zip(args, in_dims):
            if d is None or not isinstance(a, _T):
                cargs.append(a)
            else:
                ndim = len(a.shape)
                bd = d if d >= 0 else ndim + d
                sl: list[slice | int | None] = [slice(None)] * ndim
                sl[bd] = slice(start, end)
                cargs.append(a[tuple(sl)])
        chunks.append(func(*cargs, **kwargs))

    od_int: int = out_dims if isinstance(out_dims, int) else out_dims[0]
    if isinstance(chunks[0], tuple):
        n = len(chunks[0])
        ods: list[int] = (
            [out_dims] * n if isinstance(out_dims, int) else list(out_dims)
        )
        return tuple(
            lucid.cat([c[i] for c in chunks], dim=ods[i])
            for i in range(n)
        )
    return lucid.cat(chunks, dim=od_int)  # type: ignore[arg-type]


# ── grad ─────────────────────────────────────────────────────────────────────


def grad(
    func: Callable[..., Tensor],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]:
    """Return a function that computes the gradient of *func*.

    Unlike :func:`lucid.autograd.grad` (which takes tensors), this follows
    the functional-transform convention (e.g. JAX 's `jax.grad`` or the reference framework's ``func``): it takes a *function* and returns
    a *function*.

    Parameters
    ----------
    func:
        Scalar-valued differentiable function.
    argnums:
        Which argument(s) to differentiate.
    has_aux:
        If ``True``, *func* must return ``(loss, aux)``; the returned
        function yields ``(grads, aux)``.

    Examples
    --------
    ::

        f  = lambda x: (x ** 2).sum()
        df = lucid.func.grad(f)
        x  = lucid.tensor([1.0, 2.0, 3.0])
        print(df(x))   # tensor([2., 4., 6.])
    """
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def grad_fn(
        *args: Tensor, **kwargs: object
    ) -> Tensor | tuple[Tensor | None, ...]:
        from lucid._tensor.tensor import Tensor as _T
        from lucid.autograd._grad_mode import enable_grad
        from lucid.autograd._backward import grad as _ag

        args_list = list(args)
        for i in _argnums:
            a = args_list[i]
            if isinstance(a, _T) and not a.requires_grad:
                args_list[i] = a.detach().requires_grad_(True)

        with enable_grad():
            output = func(*args_list, **kwargs)

        if has_aux:
            if not isinstance(output, tuple) or len(output) < 2:
                raise ValueError(
                    "lucid.func.grad: has_aux=True requires func to return (loss, aux)"
                )
            loss: Tensor = output[0]
            aux: object = output[1] if len(output) == 2 else output[1:]
        else:
            loss = output
            aux = None

        grads = _ag(
            [loss],
            [args_list[i] for i in _argnums],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        g: Tensor | None | tuple[Tensor | None, ...] = (
            grads[0] if len(_argnums) == 1 else tuple(grads)
        )
        if has_aux:
            return g, aux
        return g  # type: ignore[return-value]

    return grad_fn


# ── grad_and_value ────────────────────────────────────────────────────────────


def grad_and_value(
    func: Callable[..., Tensor],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[..., tuple[Tensor | tuple[Tensor | None, ...], Tensor]]:
    """Like :func:`grad` but also returns the primal output.

    The returned function yields ``(gradients, value)``.
    """
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def gv_fn(
        *args: Tensor, **kwargs: object
    ) -> tuple[Tensor | tuple[Tensor | None, ...], Tensor]:
        from lucid._tensor.tensor import Tensor as _T
        from lucid.autograd._grad_mode import enable_grad
        from lucid.autograd._backward import grad as _ag

        args_list = list(args)
        for i in _argnums:
            a = args_list[i]
            if isinstance(a, _T) and not a.requires_grad:
                args_list[i] = a.detach().requires_grad_(True)

        with enable_grad():
            output = func(*args_list, **kwargs)

        if has_aux:
            if not isinstance(output, tuple) or len(output) < 2:
                raise ValueError(
                    "lucid.func.grad_and_value: has_aux=True requires func to "
                    "return (loss, aux)"
                )
            loss: Tensor = output[0]
            aux: object = output[1] if len(output) == 2 else output[1:]
        else:
            loss = output
            aux = None

        grads = _ag(
            [loss],
            [args_list[i] for i in _argnums],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        g: Tensor | None | tuple[Tensor | None, ...] = (
            grads[0] if len(_argnums) == 1 else tuple(grads)
        )
        if has_aux:
            return g, (loss, aux)
        return g, loss  # type: ignore[return-value]

    return gv_fn


# ── vjp ──────────────────────────────────────────────────────────────────────


def vjp(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    *primals: Tensor,
    has_aux: bool = False,
) -> tuple[Tensor | tuple[Tensor, ...], Callable[..., tuple[Tensor | None, ...]]]:
    """Vector-Jacobian product — functional-style API (reference-framework func-compatible).

    Returns ``(outputs, vjp_fn)`` where *vjp_fn* maps cotangents to input
    gradients.

    Parameters
    ----------
    func:
        Differentiable function.
    *primals:
        Inputs to *func*.
    has_aux:
        If ``True``, *func* returns ``(output, aux)``.

    Examples
    --------
    ::

        f = lambda x: x ** 2
        x = lucid.tensor([1.0, 2.0, 3.0])
        y, vjp_fn = lucid.func.vjp(f, x)
        (grads,) = vjp_fn(lucid.ones_like(y))   # = 2*x
    """
    from lucid._tensor.tensor import Tensor as _T
    from lucid.autograd._grad_mode import enable_grad
    from lucid.autograd._backward import grad as _ag

    primals_rg: list[Tensor] = []
    for p in primals:
        if isinstance(p, _T) and not p.requires_grad:
            p = p.detach().requires_grad_(True)
        primals_rg.append(p)

    with enable_grad():
        raw_out = func(*primals_rg)

    if has_aux:
        if not isinstance(raw_out, tuple) or len(raw_out) < 2:
            raise ValueError(
                "lucid.func.vjp: has_aux=True requires func to return (output, aux)"
            )
        outputs: Tensor | tuple[Tensor, ...] = raw_out[0]
        aux: object = raw_out[1] if len(raw_out) == 2 else raw_out[1:]
    else:
        outputs = raw_out
        aux = None

    out_list: list[Tensor] = (
        list(outputs)
        if isinstance(outputs, (list, tuple))
        else [outputs]
    )

    def _vjp(
        *cotangents: Tensor,
    ) -> tuple[Tensor | None, ...]:
        cots: list[Tensor | None] = (
            list(cotangents) if cotangents else [None] * len(out_list)
        )
        return tuple(
            _ag(
                out_list,
                primals_rg,
                grad_outputs=cots,  # type: ignore[arg-type]
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        )

    if has_aux:
        return (outputs, aux), _vjp  # type: ignore[return-value]
    return outputs, _vjp


# ── jvp ──────────────────────────────────────────────────────────────────────


def jvp(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    primals: tuple[Tensor, ...],
    tangents: tuple[Tensor, ...],
    strict: bool = False,
) -> tuple[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]]:
    """Jacobian-vector product via exact double-backward (α-perturbation trick).

    Uses a scalar perturbation variable α so that
    ``d out / d α = J(primals) @ tangents`` — computed exactly with one
    extra backward pass.  No finite differences.

    Parameters
    ----------
    func:
        Differentiable function.
    primals:
        Tuple of primal input tensors.
    tangents:
        Tangent vectors, same shapes as *primals*.
    strict:
        Raise if any output does not depend on the inputs.

    Returns
    -------
    (primals_out, tangents_out)
        Both have the same structure as ``func(*primals)``.
    """
    import lucid
    from lucid._tensor.tensor import Tensor as _T
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _wrap, _unwrap
    from lucid.autograd._grad_mode import enable_grad

    # Scalar perturbation variable α (leaf, requires_grad) evaluated at 0.
    # x_pert(α) = primal + α * tangent → d(out)/dα = J(primal) @ tangent
    alpha = lucid.tensor(0.0).requires_grad_(True)

    perturbed: list[Tensor] = []
    for p, t in zip(primals, tangents):
        if isinstance(p, _T) and isinstance(t, _T):
            perturbed.append(
                _wrap(_C_engine.add(_unwrap(p), _C_engine.mul(_unwrap(alpha), _unwrap(t))))
            )
        else:
            perturbed.append(p)

    with enable_grad():
        pert_out = func(*perturbed)

    with lucid.no_grad():
        primals_out = func(*primals)

    out_list: list[Tensor] = (
        list(pert_out)
        if isinstance(pert_out, (list, tuple))
        else [pert_out]
    )

    jvp_parts: list[Tensor] = []
    for o in out_list:
        o_impl = _unwrap(o)
        out_numel = 1
        for s in o_impl.shape:
            out_numel *= s
        out_shape = list(o_impl.shape)
        dtype = o_impl.dtype
        dev = o_impl.device

        if out_numel == 1:
            # Scalar output: d(o) / d(alpha) computed in one backward pass
            alpha._impl.zero_grad()
            seed = _wrap(_C_engine.ones(out_shape if out_shape else [], dtype, dev))
            o.backward(gradient=seed, retain_graph=True)
            ag = alpha.grad
            if ag is None:
                if strict:
                    raise ValueError("jvp: output does not depend on inputs (strict=True)")
                jvp_parts.append(_wrap(_C_engine.zeros(out_shape if out_shape else [], dtype, dev)))
            else:
                jvp_parts.append(ag.detach())
        else:
            # Vector output: O(out_numel) backward passes — one per element.
            # d(o[i]) / d(alpha) = JVP[i] for each i.
            from lucid._tensor.tensor import Tensor as _TT
            rows: list[Tensor] = []
            for row_i in range(out_numel):
                alpha._impl.zero_grad()
                seed_impl = _C_engine.zeros([out_numel], dtype, dev)
                one = _C_engine.ones([1], dtype, dev)
                idx = _C_engine.full([1], float(row_i), _C_engine.I32, dev)
                seed_impl = _C_engine.scatter_add(seed_impl, idx, one, 0)
                if out_shape:
                    seed_impl = _C_engine.reshape(seed_impl, out_shape)
                o.backward(gradient=_TT.__new_from_impl__(seed_impl), retain_graph=True)
                ag = alpha.grad
                if ag is None:
                    if strict:
                        raise ValueError(
                            "jvp: output does not depend on inputs (strict=True)"
                        )
                    rows.append(_wrap(_C_engine.zeros([], dtype, dev)))
                else:
                    rows.append(ag.detach().flatten())
            jvp_impl = _C_engine.stack([_unwrap(r) for r in rows], 0)
            jvp_parts.append(_wrap(_C_engine.reshape(jvp_impl, out_shape) if out_shape else jvp_impl))

    if isinstance(pert_out, tuple):
        return tuple(primals_out), tuple(jvp_parts)
    return primals_out, jvp_parts[0]


# ── linearize ────────────────────────────────────────────────────────────────


def linearize(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    *primals: Tensor,
) -> tuple[Tensor | tuple[Tensor, ...], Callable[..., Tensor | tuple[Tensor, ...]]]:
    """Linearise *func* at *primals*, returning ``(output, linear_fn)``.

    *linear_fn* maps tangents to the JVP: equivalent to the first-order
    Taylor expansion of *func* around *primals*.
    """
    primals_out, _ = vjp(func, *primals)

    def linear_fn(
        *tangents: Tensor,
    ) -> Tensor | tuple[Tensor, ...]:
        _, tangents_out = jvp(func, primals, tangents)
        return tangents_out

    return primals_out, linear_fn


# ── jacrev ───────────────────────────────────────────────────────────────────


def jacrev(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    chunk_size: int | None = None,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]:
    """Reverse-mode Jacobian.

    Assembles the full Jacobian matrix row by row via backward passes.
    For scalar-output functions this is equivalent to :func:`grad`.

    Parameters
    ----------
    func, argnums, has_aux:
        Same semantics as :func:`grad`.
    chunk_size:
        Reserved for future vmap-batched row computation.
    """
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def jac_fn(
        *args: Tensor, **kwargs: object
    ) -> Tensor | tuple[Tensor | None, ...]:
        from lucid._tensor.tensor import Tensor as _T
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _wrap, _unwrap
        from lucid.autograd._grad_mode import enable_grad

        args_list = list(args)
        for i in _argnums:
            a = args_list[i]
            if isinstance(a, _T) and not a.requires_grad:
                args_list[i] = a.detach().requires_grad_(True)

        with enable_grad():
            output = func(*args_list, **kwargs)

        if has_aux:
            if not isinstance(output, tuple) or len(output) < 2:
                raise ValueError(
                    "lucid.func.jacrev: has_aux=True requires func to return (output, aux)"
                )
            out_t: Tensor = output[0]
            aux: object = output[1] if len(output) == 2 else output[1:]
        else:
            out_t = output  # type: ignore[assignment]
            aux = None

        out_impl = _unwrap(out_t)
        out_numel = 1
        for s in out_impl.shape:
            out_numel *= s
        out_shape = list(out_impl.shape)

        rows: list[list[Tensor]] = [[] for _ in _argnums]

        for row_idx in range(out_numel):
            for k, i in enumerate(_argnums):
                a = args_list[i]
                if isinstance(a, _T):
                    a._impl.zero_grad()

            if out_numel == 1:
                out_t.backward(retain_graph=True)
            else:
                seed = _C_engine.zeros([out_numel], _C_engine.F32, out_impl.device)
                one = _C_engine.ones([1], _C_engine.F32, out_impl.device)
                idx = _C_engine.full([1], float(row_idx), _C_engine.I32, out_impl.device)
                seed = _C_engine.scatter_add(seed, idx, one, 0)
                if out_shape:
                    seed = _C_engine.reshape(seed, out_shape)
                from lucid._tensor.tensor import Tensor as _TT
                out_t.backward(gradient=_TT.__new_from_impl__(seed), retain_graph=True)

            for k, i in enumerate(_argnums):
                a = args_list[i]
                if isinstance(a, _T):
                    g = a._impl.grad_as_python()
                    if g is not None:
                        row_impl = _C_engine.TensorImpl(g, a._impl.device, False)
                        rows[k].append(_wrap(_C_engine.reshape(row_impl, [a.numel()])))
                    else:
                        rows[k].append(
                            _wrap(_C_engine.zeros([a.numel()], _C_engine.F32, a._impl.device))
                        )

        results: list[Tensor | None] = []
        for k, i in enumerate(_argnums):
            a = args_list[i]
            if isinstance(a, _T) and rows[k]:
                J = _C_engine.stack([_unwrap(r) for r in rows[k]], 0)
                full_shape = out_shape + list(a.shape)
                results.append(_wrap(_C_engine.reshape(J, full_shape) if full_shape else J))
            else:
                results.append(None)

        r: Tensor | tuple[Tensor | None, ...] = (
            results[0] if len(_argnums) == 1 else tuple(results)  # type: ignore[assignment]
        )
        if has_aux:
            return r, aux  # type: ignore[return-value]
        return r

    return jac_fn


# ── jacfwd ───────────────────────────────────────────────────────────────────


def jacfwd(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]:
    """Forward-mode Jacobian via :func:`jvp`.

    Computes the Jacobian column by column.  For functions with many outputs
    and few inputs, this is more efficient than :func:`jacrev`.
    """
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def jac_fn(
        *args: Tensor, **kwargs: object
    ) -> Tensor | tuple[Tensor | None, ...]:
        import lucid
        from lucid._tensor.tensor import Tensor as _T
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _wrap, _unwrap

        args_list = list(args)
        primals_sel = tuple(args_list[i] for i in _argnums)

        # Primal output for shape reference and has_aux
        with lucid.no_grad():
            ref_out = func(*args_list, **kwargs)

        if has_aux:
            if not isinstance(ref_out, tuple) or len(ref_out) < 2:
                raise ValueError(
                    "lucid.func.jacfwd: has_aux=True requires func to return (output, aux)"
                )
            sample_out: Tensor = ref_out[0]
            aux: object = ref_out[1] if len(ref_out) == 2 else ref_out[1:]
        else:
            sample_out = ref_out  # type: ignore[assignment]
            aux = None

        out_shape = list(sample_out.shape) if isinstance(sample_out, _T) else []

        cols: list[list[Tensor]] = [[] for _ in _argnums]

        for k, (argn, x) in enumerate(zip(_argnums, primals_sel)):
            if not isinstance(x, _T):
                continue
            x_numel = x.numel()
            x_shape = list(x.shape)
            dev = x._impl.device
            dtype = x._impl.dtype

            for col in range(x_numel):
                # One-hot tangent
                t_flat = _C_engine.zeros([x_numel], dtype, dev)
                one = _C_engine.ones([1], dtype, dev)
                idx = _C_engine.full([1], float(col), _C_engine.I32, dev)
                t_flat = _C_engine.scatter_add(t_flat, idx, one, 0)
                from lucid._tensor.tensor import Tensor as _TT
                t = _TT.__new_from_impl__(
                    _C_engine.reshape(t_flat, x_shape) if x_shape else t_flat
                )

                # Build full tangent tuple (zero for all other argnums)
                tans: list[Tensor] = []
                for argn2, x2 in zip(_argnums, primals_sel):
                    if argn2 == argn:
                        tans.append(t)
                    elif isinstance(x2, _T):
                        tans.append(
                            _TT.__new_from_impl__(
                                _C_engine.zeros(list(x2.shape) if x2.shape else [], dtype, dev)
                            )
                        )
                    else:
                        tans.append(t)  # fallback

                _, jvp_col = jvp(func, primals_sel, tuple(tans))
                jc: Tensor = jvp_col if not isinstance(jvp_col, tuple) else jvp_col[0]
                cols[k].append(_wrap(_C_engine.reshape(_unwrap(jc), [jc.numel()])))

        results: list[Tensor | None] = []
        for k, x in enumerate(primals_sel):
            if not isinstance(x, _T) or not cols[k]:
                results.append(None)
                continue
            J = _C_engine.stack([_unwrap(c) for c in cols[k]], -1)
            full_shape = out_shape + list(x.shape)
            results.append(_wrap(_C_engine.reshape(J, full_shape) if full_shape else J))

        r: Tensor | tuple[Tensor | None, ...] = (
            results[0] if len(_argnums) == 1 else tuple(results)  # type: ignore[assignment]
        )
        if has_aux:
            return r, aux  # type: ignore[return-value]
        return r

    return jac_fn


# ── hessian ───────────────────────────────────────────────────────────────────


def hessian(
    func: Callable[..., Tensor],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]:
    """Hessian of a scalar-valued function.

    Delegates to :func:`lucid.autograd.hessian` which uses the correct
    ``create_graph=True`` double-backward strategy.

    Parameters
    ----------
    func:
        Scalar-valued function.
    argnums:
        Which argument(s) to differentiate twice.
    """
    from lucid.autograd._functional import hessian as _hessian

    _scalar_argnum = isinstance(argnums, int)
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def hessian_fn(
        *args: Tensor, **kwargs: object
    ) -> Tensor | tuple[Tensor | None, ...]:
        selected: object = (
            args[_argnums[0]]
            if _scalar_argnum
            else tuple(args[i] for i in _argnums)
        )
        result = _hessian(
            lambda *sel: func(*_splice(args, _argnums, sel), **kwargs),
            selected,  # type: ignore[arg-type]
        )
        return result  # type: ignore[return-value]

    return hessian_fn


def _splice(
    args: tuple[object, ...],
    argnums: tuple[int, ...],
    selected: tuple[object, ...],
) -> tuple[object, ...]:
    """Replace args at argnums positions with selected values."""
    lst = list(args)
    for i, s in zip(argnums, selected):
        lst[i] = s
    return tuple(lst)
