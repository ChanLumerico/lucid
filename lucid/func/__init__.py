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
from lucid._vmap_ctx import _RandomnessGuard as _RandGuard

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


# ── isolation marker ─────────────────────────────────────────────────────────

# Attribute set on functions returned by jacrev / jacfwd / hessian so that
# vmap knows to use per-element isolation instead of the move-batch-dim path.
# Those transforms materialise per-output backward passes that must not see
# the batch dimension as part of the input — Stage 1 would give wrong shapes
# like (B, out, B, in) instead of (B, out, in).
_ISOLATION_ATTR = "_lucid_needs_vmap_isolation"


def _mark_isolation(fn: Callable[..., object]) -> Callable[..., object]:
    """Tag *fn* so :func:`vmap` uses the per-element isolation path.

    Used by :func:`jacrev` / :func:`jacfwd` / :func:`hessian` whose
    materialised backward passes must not see the batch dimension as part
    of the input.
    """
    setattr(fn, _ISOLATION_ATTR, True)
    return fn


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
    strategy: str = "auto",
) -> Callable[..., Tensor | tuple[Tensor, ...]]:
    r"""Vectorise ``func`` over a batch axis to produce a batched function.

    Lifts a function operating on a single example into one operating on a
    whole batch in a single dispatch, without writing an explicit Python
    loop. Mathematically, if :math:`f : \mathbb{R}^d \to \mathbb{R}^e`, then
    ``vmap(f)`` realises :math:`F : \mathbb{R}^{B \times d} \to
    \mathbb{R}^{B \times e}` such that ``F(x)[b] = f(x[b])`` for every
    batch index :math:`b`. The transform composes with :func:`grad`,
    :func:`jacrev`, :func:`jvp`, and :func:`vjp` to yield per-sample
    gradients, batched Jacobians, and higher-order constructs.

    Parameters
    ----------
    func : Callable
        Function to vectorise. Must accept and return tensors (or tuples of
        tensors).
    in_dims : int or tuple of (int or None), optional
        Batch axis for each input. An ``int`` applies to all positional
        arguments; a tuple gives per-argument control. Use ``None`` to
        broadcast an argument unchanged across the batch. Default ``0``.
    out_dims : int or tuple of int, optional
        Where the batch axis appears in the output(s). Default ``0``.
    randomness : str, optional
        ``"error"`` (default) forbids random ops inside ``func``;
        ``"different"`` and ``"same"`` allow them with shared RNG state.
    chunk_size : int, optional
        If set, process the batch in chunks of this size to cap peak
        memory. Applies in both vectorised and isolated strategies.
    strategy : str, optional
        ``"auto"`` (default) picks isolated mode for transforms that
        materialise per-output backward passes (jacrev/jacfwd/hessian) and
        falls back to vectorised mode otherwise. ``"vectorized"`` always
        moves the batch axis to the front and calls ``func`` once.
        ``"isolated"`` always loops per-element in Python.

    Returns
    -------
    Callable
        A new function with the batched semantics described above.

    Notes
    -----
    In vectorised mode there is exactly one underlying engine dispatch:
    on GPU this becomes a single Metal kernel launch across all batch
    elements via MLX; on CPU it becomes an Accelerate BLAS / vDSP call
    over the fully batched tensor. Reductions inside ``func`` must
    specify ``dim`` — an unqualified ``.sum()`` would also collapse the
    batch axis, which is rarely desired. In-place ops inside the
    vectorised function are unsupported.

    Examples
    --------
    Per-sample gradients:

    >>> import lucid
    >>> from lucid.func import grad, vmap
    >>> f = lambda x: (x ** 2).sum()
    >>> X = lucid.randn(32, 4)
    >>> per_sample = vmap(grad(f))(X)  # shape (32, 4)
    """
    if strategy not in ("auto", "isolated", "vectorized"):
        raise ValueError(
            f"vmap: strategy must be 'auto', 'isolated', or 'vectorized'; "
            f"got {strategy!r}"
        )

    from lucid._tensor.tensor import Tensor as _T

    def vectorized(
        *args: Tensor,
        **kwargs: object,
    ) -> Tensor | tuple[Tensor, ...]:
        """Run ``func`` once across the batched axis of each input.

        Inputs whose ``in_dims`` entry is not ``None`` must share the
        same batch length on the indicated axis. The unwrapped outputs
        are stacked along ``out_dims`` to form the vectorised result.
        """
        _in = _normalise_in_dims(in_dims, len(args))

        # ── validate inputs and find batch_size ───────────────────────
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
                raise ValueError(f"vmap: inconsistent batch sizes {batch_size} vs {bs}")

        # No batching — propagate randomness context then call directly.
        with _RandGuard(randomness):
            if batch_size is None:
                return func(*args, **kwargs)

            # ── dispatch: isolated vs vectorized ──────────────────────
            _isolate = strategy == "isolated" or (
                strategy == "auto" and getattr(func, _ISOLATION_ATTR, False)
            )

            if _isolate:
                return _isolated_vmap(
                    func, args, kwargs, _in, out_dims, batch_size, chunk_size
                )

            if chunk_size is not None and batch_size > chunk_size:
                return _chunked_vmap(
                    func, args, kwargs, _in, out_dims, chunk_size, batch_size
                )

            # ── Stage 1: move-batch-dim + call-once ──────────────────
            # Move all batch dims to front (dim 0).
            moved: list[object] = []
            for a, d in zip(args, _in):
                if d is None or not isinstance(a, _T):
                    moved.append(a)
                else:
                    moved.append(_move_to_front(a, d))

            # One C++ call with fully-batched (B, ...) tensors:
            #   GPU → MLX Metal kernel over all B elements in parallel
            #   CPU → Accelerate BLAS sgemm/vDSP over the batched array
            output = func(*moved, **kwargs)

            # Move output batch dim from 0 → out_dims.
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
        ods: list[int] = [out_dims] * n if isinstance(out_dims, int) else list(out_dims)
        return tuple(lucid.cat([c[i] for c in chunks], dim=ods[i]) for i in range(n))
    return lucid.cat(chunks, dim=od_int)  # type: ignore[arg-type]


def _isolated_vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    args: tuple[Tensor, ...],
    kwargs: dict[str, object],
    in_dims: list[int | None],
    out_dims: int | tuple[int, ...],
    batch_size: int,
    chunk_size: int | None = None,
) -> Tensor | tuple[Tensor, ...]:
    """Per-element isolation: call func(x[b]) for b in 0..batch_size-1.

    Unlike Stage 1 (move-batch-dim + call-once), each element is sliced out
    independently before calling func, so transforms that materialise backward
    passes (jacrev, jacfwd, hessian) cannot observe the batch axis as input.

    When *chunk_size* is given, results are partially stacked after each chunk
    to free intermediate autograd-graph memory before the next chunk starts.
    """
    import lucid
    from lucid._tensor.tensor import Tensor as _T

    def _run_element(b: int) -> Tensor | tuple[Tensor, ...]:
        sliced: list[object] = []
        for a, d in zip(args, in_dims):
            if d is None or not isinstance(a, _T):
                sliced.append(a)
            else:
                nd = len(a.shape)
                bd = d if d >= 0 else nd + d
                idx: list[int | slice] = [slice(None)] * nd
                idx[bd] = b
                sliced.append(a[tuple(idx)])
        return func(*sliced, **kwargs)

    def _stack(items: list[Tensor | tuple[Tensor, ...]]) -> Tensor | tuple[Tensor, ...]:
        od_i: int = out_dims if isinstance(out_dims, int) else out_dims[0]
        if isinstance(items[0], tuple):
            n_o = len(items[0])
            ods_l: list[int] = (
                [out_dims] * n_o if isinstance(out_dims, int) else list(out_dims)
            )
            return tuple(
                lucid.stack([r[i] for r in items], dim=ods_l[i])  # type: ignore[index,arg-type]
                for i in range(n_o)
            )
        return lucid.stack(list(items), dim=od_i)  # type: ignore[arg-type]

    def _cat(parts: list[Tensor | tuple[Tensor, ...]]) -> Tensor | tuple[Tensor, ...]:
        od_i = out_dims if isinstance(out_dims, int) else out_dims[0]
        if isinstance(parts[0], tuple):
            n_o = len(parts[0])
            ods_c: list[int] = (
                [out_dims] * n_o if isinstance(out_dims, int) else list(out_dims)
            )
            return tuple(
                lucid.cat([p[i] for p in parts], dim=ods_c[i])  # type: ignore[index,arg-type]
                for i in range(n_o)
            )
        return lucid.cat(list(parts), dim=od_i)  # type: ignore[arg-type]

    if chunk_size is not None and batch_size > chunk_size:
        # Process B elements in chunks; partially stack after each chunk to
        # release intermediate autograd-graph memory before the next chunk.
        parts: list[Tensor | tuple[Tensor, ...]] = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_res = [_run_element(b) for b in range(start, end)]
            parts.append(_stack(chunk_res))
        return _cat(parts)

    results = [_run_element(b) for b in range(batch_size)]
    return _stack(results)


# ── grad ─────────────────────────────────────────────────────────────────────


def grad(
    func: Callable[..., Tensor],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]:
    r"""Build a function returning the gradient of ``func``.

    The cornerstone of functional-style autograd: rather than calling
    ``.backward()`` and reading ``.grad``, ``grad(func)`` produces a new
    callable that, when invoked, returns the gradient tensor directly.
    Transforms compose, so ``grad(grad(func))`` yields a second
    derivative and ``vmap(grad(func))`` computes per-sample gradients.

    Parameters
    ----------
    func : Callable
        Function returning a **scalar** Tensor (or ``(scalar, aux)`` when
        ``has_aux=True``).
    argnums : int or tuple of int, optional
        Positional argument index/indices to differentiate. Default ``0``.
    has_aux : bool, optional
        If ``True``, ``func`` must return ``(loss, aux)``; the wrapped
        callable then returns ``(grads, aux)`` with ``aux`` forwarded
        through without differentiation. Default ``False``.

    Returns
    -------
    Callable
        Function with the same signature as ``func`` returning the
        gradient tensor — or a tuple of gradients when ``argnums`` is a
        tuple.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}`, ``grad(f)`` realises

    .. math::

        \nabla f : \mathbb{R}^n \to \mathbb{R}^n,
        \quad (\nabla f)_i (x) = \frac{\partial f}{\partial x_i}(x).

    Implementation is reverse-mode AD: one forward + one backward pass
    through ``func``, independent of input dimensionality.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import grad
    >>> f = lambda x: (x ** 3).sum()
    >>> df = grad(f)
    >>> df(lucid.tensor([1.0, 2.0, 3.0]))  # 3 * x ** 2
    Tensor([ 3., 12., 27.])
    """
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def grad_fn(*args: Tensor, **kwargs: object) -> Tensor | tuple[Tensor | None, ...]:
        """Closure that computes the gradient via reverse-mode autograd."""
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
    r"""Return a function computing both the gradient and the primal value.

    Equivalent to combining :func:`grad` with an evaluation of ``func``,
    but evaluates the forward pass exactly once. This is the canonical
    pattern for training loops where the loss value is also needed for
    logging or learning-rate scheduling.

    Parameters
    ----------
    func : Callable
        Function returning a scalar Tensor (or ``(scalar, aux)`` when
        ``has_aux=True``).
    argnums : int or tuple of int, optional
        Positional argument index/indices to differentiate. Default ``0``.
    has_aux : bool, optional
        If ``True``, ``func`` must return ``(loss, aux)`` and the wrapped
        callable returns ``(grads, (loss, aux))``. Default ``False``.

    Returns
    -------
    Callable
        Function returning ``(grads, value)`` — or ``(grads, (value, aux))``
        when ``has_aux=True``.

    Notes
    -----
    Mathematically computes both :math:`f(x)` and :math:`\nabla f(x)` in
    a single forward + backward sweep, saving redundant work compared to
    calling ``func`` and ``grad(func)`` separately.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import grad_and_value
    >>> f = lambda x: (x ** 2).sum()
    >>> gv = grad_and_value(f)
    >>> grads, value = gv(lucid.tensor([1.0, 2.0, 3.0]))
    >>> value  # 14.0
    Tensor(14.)
    """
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def gv_fn(
        *args: Tensor, **kwargs: object
    ) -> tuple[Tensor | tuple[Tensor | None, ...], Tensor]:
        """Closure that computes the gradient-vector product."""
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
    r"""Compute the vector-Jacobian product of ``func`` at ``primals``.

    Evaluates ``func`` at the supplied primal inputs and returns both the
    output and a callable ``vjp_fn`` that, given cotangent vectors
    :math:`v`, returns :math:`v^\top J(\mathrm{primals})` — the
    backward-mode contraction of the Jacobian against ``v``. This is the
    workhorse used internally by :func:`grad` and :func:`jacrev`, and is
    useful directly when many backward passes are needed against the same
    forward computation.

    Parameters
    ----------
    func : Callable
        Differentiable function taking the primals as positional inputs.
    *primals : Tensor
        Points at which to linearise ``func``.
    has_aux : bool, optional
        If ``True``, ``func`` must return ``(output, aux)``; the call
        then yields ``((output, aux), vjp_fn)``. Default ``False``.

    Returns
    -------
    tuple
        ``(outputs, vjp_fn)``. ``vjp_fn(*cotangents)`` returns a tuple of
        input-gradient tensors of the same shapes as ``primals``.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}^m` and cotangent
    :math:`v \in \mathbb{R}^m`,

    .. math::

        \mathrm{vjp\_fn}(v) = v^\top J_f(x), \quad
        J_f(x) \in \mathbb{R}^{m \times n}.

    Cost is one backward pass per call to ``vjp_fn`` — the forward
    graph is retained so multiple cotangents can be applied cheaply.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import vjp
    >>> f = lambda x: x ** 2
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> y, vjp_fn = vjp(f, x)
    >>> (grads,) = vjp_fn(lucid.ones_like(y))  # 2 * x
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
        list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
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
    r"""Compute the Jacobian-vector product of ``func`` at ``primals``.

    Returns both the primal output ``func(*primals)`` and the directional
    derivative :math:`J(\mathrm{primals}) \cdot \mathrm{tangents}` — the
    forward-mode contraction of the Jacobian against a tangent vector.
    Forward-mode is preferred over reverse-mode when the input dimension
    is small relative to the output dimension.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor or tuple of Tensors.
    primals : tuple of Tensor
        Points at which to evaluate ``func`` and its Jacobian.
    tangents : tuple of Tensor
        Tangent vectors, each matching the shape of the corresponding
        primal.
    strict : bool, optional
        If ``True``, raise when an output is independent of any input.
        Default ``False``.

    Returns
    -------
    tuple
        ``(primals_out, tangents_out)`` with the same nested structure as
        ``func(*primals)``.

    Notes
    -----
    Implemented via the exact :math:`\alpha`-perturbation trick: introduce
    a scalar :math:`\alpha` with ``requires_grad=True``, substitute
    ``x + alpha * t`` for each primal, and read
    :math:`\partial \text{out} / \partial \alpha` at :math:`\alpha = 0` from
    a single backward pass. No finite differences are used, so the result
    is exact up to floating-point rounding.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import jvp
    >>> f = lambda x: x ** 2
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> t = lucid.ones_like(x)
    >>> y, dy = jvp(f, (x,), (t,))  # dy = 2 * x
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
                _wrap(
                    _C_engine.add(_unwrap(p), _C_engine.mul(_unwrap(alpha), _unwrap(t)))
                )
            )
        else:
            perturbed.append(p)

    with enable_grad():
        pert_out = func(*perturbed)

    with lucid.no_grad():
        primals_out = func(*primals)

    out_list: list[Tensor] = (
        list(pert_out) if isinstance(pert_out, (list, tuple)) else [pert_out]
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
                    raise ValueError(
                        "jvp: output does not depend on inputs (strict=True)"
                    )
                jvp_parts.append(
                    _wrap(_C_engine.zeros(out_shape if out_shape else [], dtype, dev))
                )
            else:
                # alpha may have shape (1,) due to lucid.tensor(0.0) semantics;
                # reshape to match the actual output element shape.
                jvp_parts.append(
                    _wrap(
                        _C_engine.reshape(
                            _unwrap(ag.detach()), out_shape if out_shape else []
                        )
                    )
                )
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
            jvp_parts.append(
                _wrap(_C_engine.reshape(jvp_impl, out_shape) if out_shape else jvp_impl)
            )

    if isinstance(pert_out, tuple):
        return tuple(primals_out), tuple(jvp_parts)
    return primals_out, jvp_parts[0]


# ── linearize ────────────────────────────────────────────────────────────────


def linearize(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    *primals: Tensor,
) -> tuple[Tensor | tuple[Tensor, ...], Callable[..., Tensor | tuple[Tensor, ...]]]:
    r"""Linearise ``func`` around ``primals`` for reuse across many tangents.

    Returns the primal output together with a callable ``linear_fn`` that
    applies the first-order Taylor expansion of ``func`` at ``primals``.
    Mathematically ``linear_fn(t)`` evaluates :math:`J_f(\mathrm{primals})
    \, t` — the same quantity as :func:`jvp` — but the linearisation cost
    is paid once even if many tangent vectors are queried subsequently.

    Parameters
    ----------
    func : Callable
        Differentiable function.
    *primals : Tensor
        Points at which to linearise ``func``.

    Returns
    -------
    tuple
        ``(primals_out, linear_fn)``. Calling ``linear_fn(*tangents)``
        returns the JVP at ``primals`` against ``tangents``.

    Notes
    -----
    Conceptually equivalent to a Taylor expansion truncated at first
    order:

    .. math::

        f(x + t) \approx f(x) + J_f(x) \, t.

    The returned ``linear_fn`` is flagged for vmap isolation so that
    composing ``vmap(linear_fn)`` slices tangents one at a time.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import linearize
    >>> f = lambda x: x ** 2
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> y, lin = linearize(f, x)
    >>> lin(lucid.ones_like(x))  # 2 * x
    """
    primals_out, _ = vjp(func, *primals)

    def linear_fn(
        *tangents: Tensor,
    ) -> Tensor | tuple[Tensor, ...]:
        """Closure that applies the linear approximation."""
        _, tangents_out = jvp(func, primals, tangents)
        return tangents_out

    # linear_fn closes over fixed primals and maps tangents through jvp.
    # When vmapped with strategy='auto', isolation ensures each tangent
    # vector is passed individually so jvp sees the right (n,) shape.
    setattr(linear_fn, _ISOLATION_ATTR, True)
    return primals_out, linear_fn


# ── jacrev ───────────────────────────────────────────────────────────────────


def jacrev(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    chunk_size: int | None = None,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]:
    r"""Build a function returning the reverse-mode Jacobian of ``func``.

    Assembles the full Jacobian matrix one row at a time, each row
    obtained from a backward pass seeded with a one-hot cotangent. For a
    scalar-output ``func`` this collapses to :func:`grad`.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor (or ``(output, aux)``
        when ``has_aux=True``).
    argnums : int or tuple of int, optional
        Argument(s) to differentiate with respect to. Default ``0``.
    has_aux : bool, optional
        Whether ``func`` returns auxiliary data. Default ``False``.
    chunk_size : int, optional
        Reserved for future vmap-batched row computation.

    Returns
    -------
    Callable
        Function returning the Jacobian tensor (or tuple of tensors)
        with shape ``output.shape + arg.shape``.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}^m` produces

    .. math::

        J_{ij} = \frac{\partial f_i}{\partial x_j}, \quad
        J \in \mathbb{R}^{m \times n}.

    Cost scales with the output dimension :math:`m` (one backward pass
    per row). Prefer :func:`jacfwd` when :math:`n < m`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import jacrev
    >>> f = lambda x: lucid.stack([x.sum(), (x ** 2).sum()])
    >>> jacrev(f)(lucid.tensor([1.0, 2.0, 3.0]))  # shape (2, 3)
    """
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def jac_fn(*args: Tensor, **kwargs: object) -> Tensor | tuple[Tensor | None, ...]:
        """Closure that computes the Jacobian."""
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
                idx = _C_engine.full(
                    [1], float(row_idx), _C_engine.I32, out_impl.device
                )
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
                            _wrap(
                                _C_engine.zeros(
                                    [a.numel()], _C_engine.F32, a._impl.device
                                )
                            )
                        )

        results: list[Tensor | None] = []
        for k, i in enumerate(_argnums):
            a = args_list[i]
            if isinstance(a, _T) and rows[k]:
                J = _C_engine.stack([_unwrap(r) for r in rows[k]], 0)
                full_shape = out_shape + list(a.shape)
                results.append(
                    _wrap(_C_engine.reshape(J, full_shape) if full_shape else J)
                )
            else:
                results.append(None)

        r: Tensor | tuple[Tensor | None, ...] = (
            results[0] if len(_argnums) == 1 else tuple(results)  # type: ignore[assignment]
        )
        if has_aux:
            return r, aux  # type: ignore[return-value]
        return r

    return _mark_isolation(jac_fn)  # type: ignore[return-value]


# ── jacfwd ───────────────────────────────────────────────────────────────────


def jacfwd(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]:
    r"""Build a function returning the forward-mode Jacobian of ``func``.

    Materialises the Jacobian one column at a time by repeatedly calling
    :func:`jvp` with one-hot tangent vectors. Each column corresponds to
    the partial derivative of every output with respect to a single input
    coordinate.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor (or ``(output, aux)``
        when ``has_aux=True``).
    argnums : int or tuple of int, optional
        Argument(s) to differentiate with respect to. Default ``0``.
    has_aux : bool, optional
        Whether ``func`` returns auxiliary data. Default ``False``.
    randomness : str, optional
        Randomness policy forwarded to internal vmap calls. Default
        ``"error"``.

    Returns
    -------
    Callable
        Function returning the Jacobian tensor (or tuple of tensors)
        with shape ``output.shape + arg.shape``.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}^m` produces

    .. math::

        J_{ij} = \frac{\partial f_i}{\partial x_j}, \quad
        J \in \mathbb{R}^{m \times n}.

    Cost scales with the input dimension :math:`n` (one forward / JVP per
    column). Prefer this transform over :func:`jacrev` when
    :math:`n \ll m`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import jacfwd
    >>> f = lambda x: lucid.stack([x.sum(), (x ** 2).sum()])
    >>> jacfwd(f)(lucid.tensor([1.0, 2.0, 3.0]))  # shape (2, 3)
    """
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def jac_fn(*args: Tensor, **kwargs: object) -> Tensor | tuple[Tensor | None, ...]:
        """Closure that computes the Jacobian."""
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
                                _C_engine.zeros(
                                    list(x2.shape) if x2.shape else [], dtype, dev
                                )
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

    return _mark_isolation(jac_fn)  # type: ignore[return-value]


# ── hessian ───────────────────────────────────────────────────────────────────


def hessian(
    func: Callable[..., Tensor],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[..., Tensor | tuple[Tensor | None, ...]]:
    r"""Build a function returning the Hessian of a scalar-valued ``func``.

    Computes the matrix of second partial derivatives by composing
    forward-mode over reverse-mode differentiation
    (``jacfwd(jacrev(func))``). The result captures the local curvature
    used by Newton-style optimisers, natural-gradient methods, and
    second-order analyses such as eigenvalue spectra of the loss.

    Parameters
    ----------
    func : Callable
        Function returning a scalar Tensor.
    argnums : int or tuple of int, optional
        Argument(s) to differentiate twice. Default ``0``.

    Returns
    -------
    Callable
        Function returning the Hessian tensor with shape
        ``arg.shape + arg.shape``.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}`,

    .. math::

        H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j},
        \quad H \in \mathbb{R}^{n \times n}.

    Cost is dominated by :math:`n` JVPs over the reverse-mode gradient
    graph, retained via ``create_graph=True``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import hessian
    >>> f = lambda x: (x ** 3).sum()  # H = diag(6 * x)
    >>> hessian(f)(lucid.tensor([1.0, 2.0, 3.0]))
    """
    from lucid.autograd._functional import hessian as _hessian

    _scalar_argnum = isinstance(argnums, int)
    _argnums: tuple[int, ...] = (
        (argnums,) if isinstance(argnums, int) else tuple(argnums)
    )

    def hessian_fn(
        *args: Tensor, **kwargs: object
    ) -> Tensor | tuple[Tensor | None, ...]:
        """Closure that computes the Hessian (second-order derivative tensor)."""
        selected: object = (
            args[_argnums[0]] if _scalar_argnum else tuple(args[i] for i in _argnums)
        )
        result = _hessian(
            lambda *sel: func(*_splice(args, _argnums, sel), **kwargs),
            selected,  # type: ignore[arg-type]
        )
        return result  # type: ignore[return-value]

    return _mark_isolation(hessian_fn)  # type: ignore[return-value]


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
