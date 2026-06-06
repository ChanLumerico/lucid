"""
lucid.compile._entry.step — Phase 1.5 step 5.1.

``make_step(model, loss_fn)`` returns a callable that traces +
compiles ``model(x); loss_fn(out, target)`` into a single
:class:`MPSGraphExecutable` (forward + backward) and wraps the result
as a Lucid :class:`autograd.Function`.  The user can then call
``loss.backward()`` and the cached fwd+bwd executable supplies the
parameter gradients exactly the way an eager step would — but the
Lucid autograd engine is responsible for wiring those gradients into
the parameter ``.grad`` slots.

This is the autograd-integrated successor to Phase 1.3's
:func:`compiled_step`.  Both surfaces co-exist for the duration of
Phase 1.5: ``compiled_step`` remains the direct call helper for
benchmarks / smoke tests, while ``make_step`` is the production entry
that plays well with eager Lucid models that already populate the
forward call site with `lucid.Tensor` graph state.

Acceptance:

- For a single step, ``loss.backward(); opt.step()`` with ``make_step``
  yields the same parameter trajectory as the eager ``model(x);
  loss_fn(out, target); loss.backward(); opt.step()`` baseline within
  fp32 round-off.
- Cache hit on identical signature avoids re-tracing.
- A graph-incompatible op (e.g. fused ``F.mse_loss``) falls through to
  eager + records the signature as ``eager-only``.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast, final, override

from lucid._C import engine as _C_engine
from lucid.autograd.function import Function, FunctionCtx


class _ExecutableLike(Protocol):
    """Surface of pybind11 ``CompiledExecutable`` consumed by _step.py."""

    input_ids: list[int]
    output_ids: list[int]
    num_inputs: int


if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["make_step"]


@final
@dataclass(slots=True)
class _StepEntry:
    """One compiled fwd+bwd executable + the I/O plan to invoke it."""

    exe: object  # _C_engine.compile.PyCompiledExecutable
    external_feeds: dict[int, object]
    # Per ``exe.input_ids[i]`` slot: ``None`` → pinned param/constant in
    # ``external_feeds``; integer → index into the positional ``args``
    # tuple of the wrapped ``step`` callable.
    input_source: tuple[int | None, ...]
    # JSON-serialised TraceGraph for observability (graph_dump).
    graph_json: str = ""
    n_hits: int = 0
    compile_ms: float = 0.0
    last_run_ms: float = 0.0
    # 3.5 BatchNorm running-stats write-back.  Fused-momentum BN traces as a
    # 5-input/3-output node whose 2 extra outputs are the EMA-updated
    # running_mean / running_var.  They are surfaced as extra executable
    # outputs (trailing [loss, *grads]) and, after each run, copy_'d back into
    # these live module buffers so ``model.eval()`` reads fresh stats instead
    # of the frozen trace-time value.  Empty when the model has no such BN.
    bn_stat_buffers: list[Tensor] = field(default_factory=list)
    bn_stat_out_count: int = 0


def make_step(
    model: Module,
    loss_fn: Callable[..., Tensor],
    *,
    dynamic: bool = False,
) -> Callable[..., Tensor]:
    """Return a callable that runs one compiled training step.

    The returned ``step(x, *extra_inputs)`` callable:

    1. Calls ``loss_fn(model(x), *extra_inputs)`` once under
       :func:`_tracing` to capture the op DAG (forward + loss).
    2. Compiles a single MPSGraph executable that produces the loss
       and the gradient of the loss w.r.t. every model parameter, via
       :func:`compile_trace_with_backward`.
    3. Runs the executable, populates a Lucid :class:`autograd.Function`
       with the resulting (loss, grads) so the returned scalar loss has
       a working ``grad_fn`` and ``loss.backward()`` drives ``.grad`` on
       every parameter.
    4. Caches the executable so a second call with the same input /
       parameter signature is a pure run.

    Parameters
    ----------
    model : nn.Module
        Model whose forward will be traced + compiled.
    loss_fn : callable
        ``loss_fn(model_output, *extra_inputs) -> Tensor`` scalar loss.
    dynamic : bool, optional
        Reserved for Phase 1.6's symbolic batch axis.  Today the value
        is recorded in the signature but otherwise unused.

    Returns
    -------
    callable
        ``step(x, *extra_inputs) -> Tensor`` — the scalar loss.
        ``loss.backward()`` works the same way as the eager baseline.

    Examples
    --------
    >>> import lucid, lucid.nn as nn, lucid.nn.functional as F
    >>> import lucid.optim as optim
    >>> from lucid.compile import make_step
    >>> model = nn.Linear(8, 4).to('metal')
    >>> step = make_step(model, lambda y, t: F.cross_entropy(y, t))
    >>> opt = optim.SGD(model.parameters(), lr=0.1)
    >>> for batch in batches:                          # doctest: +SKIP
    ...     opt.zero_grad()
    ...     loss = step(batch.x, batch.target)
    ...     loss.backward()
    ...     opt.step()

    See Also
    --------
    lucid.compile.compile : module-level compile entrypoint.
    lucid.compile.compile_optimizer : even tighter form that fuses
        the optimizer update into the same MPSGraph executable.
    lucid.compile._entry.module.CompiledModule.step : per-instance
        cached version of the same dispatch.
    """
    from lucid._dispatch import _unwrap, _wrap
    from lucid._tensor.tensor import Tensor
    from lucid.autograd._grad_mode import no_grad
    from lucid.compile import _tracing
    from lucid.compile._core.bn_runstats import (
        bn_writeback_targets,
        model_has_cumulative_bn,
    )
    from lucid.compile._core.fallback import EagerFallbackSet
    from lucid.compile._core.signature import signature_of

    # Phase 1.6 dynamic-batch: gated behind ``LUCID_COMPILE_DYNAMIC=1``
    # alongside the matching surface in :class:`CompiledModule`.  Until
    # macOS 25 ``gradientForPrimaryTensor:`` aborted on symbolic-shape
    # primaries; reopened 2026-05-25 to characterise the macOS 26
    # / MPSGraph SDK state.  Falls back to NotImplementedError unless
    # the env opt-in is set.
    import os as _os_step

    _dyn_opt_in = _os_step.environ.get(
        "LUCID_COMPILE_DYNAMIC",
        "0",
    ) in ("1", "true", "True")
    if dynamic and not _dyn_opt_in:
        raise NotImplementedError(
            "lucid.compile.make_step(..., dynamic=True) is gated behind "
            "LUCID_COMPILE_DYNAMIC=1 while macOS 26's MPSGraph backward "
            "support for symbolic-shape primaries is being characterised. "
            "Set the env var to opt in; otherwise use static shapes."
        )

    cache: dict[Any, _StepEntry] = {}
    eager_only = EagerFallbackSet()

    # Snapshot the parameter list once.  ``make_step`` is meant to be
    # called per-model-per-training-loop, not on every step, so the
    # parameter set is stable across calls (mutating it invalidates
    # the cache — the caller must rebuild via a fresh ``make_step``).
    params = list(model.parameters())
    if not params:
        raise ValueError("make_step: model has no trainable parameters")

    def _compile_for(
        x_args: tuple[Tensor, ...],
    ) -> _StepEntry | None:
        """Trace ``model`` + ``loss_fn`` once for this signature and compile.

        Parameters
        ----------
        x_args : tuple of Tensor
            ``(model_input, *loss_extra_inputs)`` — the same positional
            tuple the returned ``step`` callable expects.

        Returns
        -------
        _StepEntry or None
            Compiled executable + feed plan, or ``None`` on any
            failure condition (empty trace, unresolved parameter
            id, builder rejection) — caller marks the signature
            eager-only.
        """
        t0 = time.perf_counter()
        with no_grad():
            with _tracing() as tracer:
                x_out = model(*x_args[:1])
                loss_fn(x_out, *x_args[1:])

        graph = tracer.graph
        if not graph.ops:
            return None
        ext = tracer.external_feeds
        loss_id = graph.ops[-1].outputs[0].id

        # 3.5 BatchNorm running-stats. A cumulative-MA BN (track_running_stats=
        # True + momentum=None) can't be lowered into the graph (its update reads
        # num_batches_tracked as a host scalar) → eager-fallback the whole step.
        # A track_running_stats=False BN keeps no buffers and compiles unchanged
        # (3-input node, nothing to write back). A fused-momentum BN traces as a
        # 5-input/3-output node; surface its EMA outputs (new_rm/new_rv) and
        # write them into the live module buffers after each run, else the
        # running stats freeze at their trace-time value and eval() reads stale
        # stats. The cumulative-vs-stateless distinction is invisible in the
        # trace IR, so the model is the discriminator (see _bn_runstats).
        if model_has_cumulative_bn(model):
            return None
        _bn_targets = bn_writeback_targets(graph, ext)
        bn_stat_out_ids: list[int] = [out_id for _, out_id, _ in _bn_targets]
        bn_stat_buffers: list[Tensor] = [_wrap(impl) for _, _, impl in _bn_targets]

        # param_ids: match each Parameter (by TensorImpl identity)
        # against ext.  Re-fetched on every compile in case the inner
        # model mutated (e.g. .to() moved params).
        p_impls = [_unwrap(p) for p in params]
        param_ids: list[int] = []
        for p_impl in p_impls:
            found = None
            for tid, impl in ext.items():
                if impl is p_impl:
                    found = tid
                    break
            if found is None:
                return None
            param_ids.append(found)

        try:
            exe = _C_engine.compile.compile_trace_with_backward(
                graph,
                ext,
                loss_id,
                param_ids,
                dynamic,
                extra_output_ids=bn_stat_out_ids,
            )
        except RuntimeError:
            # compile_trace_with_backward surfaces a RuntimeError when
            # an op has no emitter or another invariant fails.  Either
            # way it's an eager-only signature.
            return None
        if exe is None:
            return None

        # Build the positional-feed plan.  ``args`` to ``step`` is the
        # user-supplied input tuple; everything else is a pinned param
        # or a graph constant.
        positional_impls: dict[int, int] = {}
        for i, a in enumerate(x_args):
            if isinstance(a, Tensor):
                positional_impls[id(_unwrap(a))] = i

        input_source: list[int | None] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                return None
            input_source.append(positional_impls.get(id(impl)))

        from lucid.compile._debug.trace_dump import trace_to_json

        return _StepEntry(
            exe=exe,
            external_feeds=dict(ext),
            input_source=tuple(input_source),
            graph_json=trace_to_json(graph),
            compile_ms=(time.perf_counter() - t0) * 1000.0,
            bn_stat_buffers=bn_stat_buffers,
            bn_stat_out_count=len(bn_stat_out_ids),
        )

    def _run(
        entry: _StepEntry, x_args: tuple[Tensor, ...]
    ) -> tuple[_C_engine.TensorImpl, list[_C_engine.TensorImpl]]:
        """Execute one cached step entry; return ``(loss_impl, [grad_impls])``.

        The executable returns ``len(params) + 1`` storages — the loss
        scalar followed by the per-parameter gradients in the same
        order ``params`` was captured.  The caller wraps these into
        Lucid Tensors and feeds them into the autograd graph.
        """
        feed_impls: list[object] = []
        exe = cast(_ExecutableLike, entry.exe)
        for tid, src in zip(exe.input_ids, entry.input_source):
            if src is None:
                feed_impls.append(entry.external_feeds[tid])
            else:
                feed_impls.append(_unwrap(x_args[src]))
        t0 = time.perf_counter()
        outs = _C_engine.compile.run_executable(entry.exe, feed_impls)
        entry.last_run_ms = (time.perf_counter() - t0) * 1000.0
        entry.n_hits += 1
        # 3.5 BatchNorm running-stats write-back: the BN EMA outputs trail
        # [loss, *grads] (one pair per fused-momentum BN).  copy_ each into its
        # live module buffer so eval() reads fresh stats.  The count assert
        # guards against a C++/Python output-slice drift.
        n_loss_grad = 1 + len(params)
        if len(outs) != n_loss_grad + entry.bn_stat_out_count:
            raise RuntimeError(
                f"make_step: run_executable returned {len(outs)} outputs, "
                f"expected {n_loss_grad + entry.bn_stat_out_count} "
                f"(1 loss + {len(params)} grads + {entry.bn_stat_out_count} BN stats)"
            )
        for _i, _buf in enumerate(entry.bn_stat_buffers):
            _buf.copy_(_wrap(cast(_C_engine.TensorImpl, outs[n_loss_grad + _i])))
        # outs = [loss_impl, grad_param_0_impl, …]
        loss_impl = cast(_C_engine.TensorImpl, outs[0])
        grad_impls = [cast(_C_engine.TensorImpl, g) for g in outs[1 : 1 + len(params)]]
        return loss_impl, grad_impls

    @final
    class _CompiledStepFunction(Function):
        """Inner autograd.Function — only the *compiled* path runs through here.

        The eager fallback is handled by ``step()`` itself BEFORE we
        enter the Function, so this static method can assume a valid
        cache entry was just looked up by the caller and passed in
        via the first positional argument.
        """

        @override
        @staticmethod
        def forward(  # type: ignore[override]  # reason: takes a non-Tensor ``entry_holder`` 1st positional (the _StepEntry holder); Function.forward base types *args as Tensor. Documented as an exception per CompiledModule's design.
            ctx: FunctionCtx,
            entry_holder: list[_StepEntry],
            *step_args: Tensor,
        ) -> Tensor:
            """Run the cached executable; save grads for backward.

            ``entry_holder[0]`` is the :class:`_StepEntry` selected
            by the outer ``step()`` cache lookup.  ``step_args`` is
            laid out as ``(*x_args, *params)`` so the autograd graph
            sees both the user inputs and the model parameters in
            its next-edge slot list (the parameter gradients are
            wired in :meth:`backward`).
            """
            # ``entry_holder`` is a 1-element list owning the
            # ``_StepEntry``.  Wrapping it in a list keeps it out of
            # the *Tensor*-only positional contract of Function.apply
            # — the meta-class still passes it through to ``forward``
            # unchanged because non-Tensor positional args are
            # allowed (see :meth:`Function.forward` docstring).
            #
            # Layout of ``step_args``: (x, *extra_inputs, *params).
            entry = entry_holder[0]
            n_user = len(step_args) - len(params)
            x_args = step_args[:n_user]

            loss_impl, grad_impls = _run(entry, x_args)
            ctx.saved_grad_impls = grad_impls
            ctx.n_user = n_user
            return _wrap(loss_impl)

        @override
        @staticmethod
        def backward(  # type: ignore[override]  # reason: returns tuple[Tensor|None, ...] (one slot per next-edge: None for user inputs, Tensors for params); base returns Tensor|tuple[Tensor,...].
            ctx: FunctionCtx, grad_loss: Tensor
        ) -> tuple[Tensor | None, ...]:
            """Chain ``grad_loss`` into each saved per-parameter gradient.

            Returns one slot per autograd next-edge: ``None`` for
            every user-input position (so backprop doesn't propagate
            into data / targets) followed by the scaled gradients
            for each parameter.
            """
            # Chain rule: ``grad_param_i = grad_loss * dL/dparam_i``.
            # Returning ``None`` for the user-supplied inputs declines
            # to back-propagate further into them (data / targets the
            # optimizer ignores).  The ``entry_holder`` non-Tensor
            # input does not appear in ``next_edges`` so no slot is
            # produced for it here.
            saved_grad_impls = cast(list[_C_engine.TensorImpl], ctx.saved_grad_impls)
            n_user = cast(int, ctx.n_user)
            scaled = [grad_loss * _wrap(g) for g in saved_grad_impls]
            return tuple([None] * n_user + scaled)

    def step(*x_args: Tensor) -> Tensor:
        """Run one fwd+bwd compiled training step (or eager fallback).

        Parameters
        ----------
        *x_args : Tensor
            ``(model_input, *loss_extra_inputs)`` — the same shape
            ``model(input)`` + ``loss_fn(out, *extras)`` accept
            together.

        Returns
        -------
        Tensor
            The scalar loss carrying a ``grad_fn`` that, when
            ``.backward()`` is called, runs the captured gradients
            into each model parameter's ``.grad``.  On compile
            failure the loss tensor instead comes from the eager
            forward + ``loss_fn`` and uses the regular autograd
            graph for backward.
        """
        # Cache lookup / compile / eager-fallback decision happens
        # OUTSIDE the autograd Function so the eager path doesn't
        # accidentally end up with a "compiled" wrapper around it.
        try:
            key = signature_of(model, x_args, {}, dynamic=dynamic)
        except TypeError, AttributeError:
            key = None  # un-hashable → fresh per-call compile only

        if key is not None and key in eager_only:
            out = model(*x_args[:1])
            return loss_fn(out, *x_args[1:])

        entry = cache.get(key) if key is not None else None
        if entry is None:
            entry = _compile_for(x_args)
            if entry is None:
                if key is not None:
                    eager_only.add(key)
                out = model(*x_args[:1])
                return loss_fn(out, *x_args[1:])
            if key is not None:
                cache[key] = entry

        return cast(Tensor, _CompiledStepFunction.apply([entry], *x_args, *params))

    # Stash the cache / introspection handles on the returned callable
    # so tests + telemetry can poke at them without grabbing them via
    # closure.
    step.cache = cache  # type: ignore[attr-defined]
    step.eager_only = eager_only  # type: ignore[attr-defined]

    def clear_cache() -> None:
        """Drop every cached executable + eager-fallback blacklist for this step callable."""
        cache.clear()
        eager_only.clear()

    def graph_dump() -> str:
        """Return a JSON view of every cached signature's :class:`TraceGraph`.

        Used by `tools/inspect_trace.py` and the test harness to diff
        consecutive compiles and pinpoint divergent ops.
        """
        import json as _json

        items = [
            {"sig_hash": hash(k), "graph": _json.loads(e.graph_json)}
            for k, e in cache.items()
        ]
        return _json.dumps(items, indent=2)

    step.clear_cache = clear_cache  # type: ignore[attr-defined]
    step.graph_dump = graph_dump  # type: ignore[attr-defined]

    return step
