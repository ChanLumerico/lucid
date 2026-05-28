"""
lucid.compile._function — single-step compile+run helper for Phase 1.3.

The Plan's `CompiledStepFunction(autograd.Function)` design routes the
compiled fwd+bwd through Lucid's autograd graph so a ``loss.backward()``
call replays our cached executable.  That full integration lives with
Phase 1.4's ``CompiledModule`` (cache + ownership semantics + recompile
triggers all in one place).

For Phase 1.3's acceptance gate (ResNet-18 + CE + SGD, 5 steps via
compile, param drift < 5e-3) the minimal surface is a direct
``compiled_step(model, x, loss_fn)`` function that:

  1. Runs ``model(x); loss_fn(out)`` once under ``_tracing()`` to record
     the op DAG.
  2. Calls ``compile_trace_with_backward`` to get a single executable
     that produces ``loss`` + per-parameter gradients in one shot.
  3. Runs that executable.
  4. Writes the per-parameter gradients directly to ``param.grad``, so
     downstream ``optimizer.step()`` works exactly like an eager
     training step.

This bypasses Lucid's autograd graph entirely for the compiled step.
That's fine for Phase 1.3's deliverable (5-step training parity); we
keep the autograd-integrated surface for Phase 1.4.
"""

from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor

if TYPE_CHECKING:
    from lucid.nn.module import Module

__all__ = ["compiled_step"]


def compiled_step(
    model: Module,
    x: Tensor,
    loss_fn: Callable[[Tensor], Tensor],
    *,
    use_grad_mode: bool = False,
) -> Tensor:
    """Run one compiled training step: forward + backward + grad assignment.

    Parameters
    ----------
    model : nn.Module
        Module whose ``forward(x)`` produces the model output.
    x : Tensor
        Input tensor (single forward pass).
    loss_fn : callable
        Maps ``model(x)`` → scalar loss tensor.  Typical: ``lambda y: F.cross_entropy(y, target)``.
    use_grad_mode : bool, optional
        Whether to keep autograd's GradMode on during tracing.  Default
        ``False`` — the compiled step bypasses Lucid's autograd graph
        entirely; we only need the trace hook to fire (it fires under
        no_grad too because the kernel hooks run before the GradMode
        short-circuit).  Set ``True`` if you want the eager autograd
        graph to also build (useful for verification).

    Returns
    -------
    Tensor
        The loss tensor (compiled result).  After this call,
        ``param.grad`` is populated for every parameter of ``model``.
        The caller can then run an optimizer step.

    Notes
    -----
    The trace is captured fresh on every call; cache integration
    (compile-or-cached) lives in Phase 1.4's ``CompiledModule``.

    Phase 1.3 acceptance: ``ResNet-18 + CE + SGD, 5 steps``, param
    drift < 5e-3 vs eager.

    Examples
    --------
    >>> import lucid, lucid.nn as nn, lucid.nn.functional as F
    >>> import lucid.optim as optim
    >>> from lucid.compile._function import compiled_step
    >>> model = nn.Linear(8, 4).to('metal')
    >>> opt = optim.SGD(model.parameters(), lr=0.1)
    >>> for batch in batches:                       # doctest: +SKIP
    ...     opt.zero_grad()
    ...     loss = compiled_step(model, batch.x,
    ...                          lambda y: F.cross_entropy(y, batch.target))
    ...     opt.step()

    See Also
    --------
    lucid.compile.make_step : the cached entry point preferred for
        repeated training loops — wraps this in a signature-keyed
        cache so the compile cost amortises.
    lucid.compile.compile : module-level entrypoint.
    """
    # Local imports to avoid circulars during package init.
    from lucid._dispatch import _unwrap, _wrap
    from lucid.autograd._grad_mode import no_grad
    from lucid.compile import _tracing

    params = list(model.parameters())
    if not params:
        raise ValueError("compiled_step: model has no trainable parameters")

    grad_ctx = nullcontext() if use_grad_mode else no_grad()
    with grad_ctx:
        with _tracing() as tracer:
            out = model(x)
            # Module.__call__ widens to Tensor|tuple; loss_fn expects Tensor.
            if not isinstance(out, Tensor):
                raise TypeError(
                    "compiled_step: model must return a Tensor for the loss_fn "
                    f"to consume, got {type(out).__name__}"
                )
            loss_fn(out)

    g = tracer.graph
    ext = tracer.external_feeds
    if not g.ops:
        raise RuntimeError("compiled_step: empty trace (model produced no ops)")
    loss_id = g.ops[-1].outputs[0].id

    # Match each Parameter (by TensorImpl pointer identity) to its
    # trace feed id.  Using ``impl is target`` (not equality) — every
    # external feed in the trace is the exact same Python object as
    # the user-side Parameter.
    p_impls = [_unwrap(p) for p in params]
    param_ids: list[int] = []
    for p_impl in p_impls:
        found = None
        for tid, impl in ext.items():
            if impl is p_impl:
                found = tid
                break
        if found is None:
            raise RuntimeError(
                "compiled_step: a model parameter was not observed in the trace "
                "(model.forward did not use it, or the parameter was wrapped/copied)"
            )
        param_ids.append(found)

    exe = _C_engine.compile.compile_trace_with_backward(g, ext, loss_id, param_ids)
    if exe is None:
        raise RuntimeError(
            "compiled_step: compile_trace_with_backward returned None — "
            "the trace contains an op without an emitter (eager fallback)"
        )

    feed_impls = [ext[tid] for tid in exe.input_ids]
    outs = _C_engine.compile.run_executable(exe, feed_impls)

    # First output is the loss; the rest are per-parameter gradients
    # (in the same order we passed param_ids).
    if len(outs) != 1 + len(params):
        raise RuntimeError(
            f"compiled_step: executable produced {len(outs)} outputs but expected "
            f"{1 + len(params)} (loss + {len(params)} grads)"
        )
    loss_compiled = _wrap(outs[0])

    # Direct grad assignment — the eager optimizer reads ``param.grad``
    # the same way as after ``loss.backward()``.
    for param, grad_impl in zip(params, outs[1:]):
        param.grad = _wrap(grad_impl)

    return loss_compiled
