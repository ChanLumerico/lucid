"""
lucid.compile._fused_step — Phase 1.8 generic fused training step.

``fused_step(model, loss_fn, optimizer)`` returns a callable that runs
the entire training step (forward + loss + auto-derived gradients +
optimizer.step) inside a single MPSGraph executable.  One ``run`` call
per training iteration, all writes to parameters / optimizer state
happen in-place via ``run_executable_inplace``.

Architecture
------------
The optimizer update math lives in the corresponding
:mod:`lucid.compile._optim` subclass's ``_trace_update`` method
(``_CompiledSGD``, ``_CompiledAdam``, ..., ``_CompiledNAdam``).  At
``fused_step`` construction time:

1. Reuse the optimizer subclass to allocate state buffers + scalar
   feeds.  It carries the math AND the in-place-target plan.
2. Open a single :class:`Tracer`.  Run ``model(x); loss_fn(out, t)``
   (the forward + loss path).  Then call the optimizer subclass's
   ``_trace_update`` with **ghost grad placeholders** (zero tensors
   that take the slot of "param_i.grad" in the trace).  The trace
   now contains forward+loss+opt-update ops.
3. Call ``compile_generic_fused_step`` with the ghost grad ids.
   C++ side derives gradients via MPSGraph autograd after emitting
   the forward, binds each ghost grad id to its derived gradient
   tensor, then continues emitting the opt-update ops (which now
   resolve their grad reads correctly).
4. Each ``step(x, t)`` call: refresh scalars (if any), then
   ``run_executable_inplace`` writes new params / new state
   directly into the corresponding tensors.

Supported optimizers
--------------------
Every optimizer that :func:`compile_optimizer` accepts: SGD, Adam,
AdamW, RMSprop, Adagrad, Adadelta, Adamax, NAdam.  The same rejection
messages apply for LBFGS / SparseAdam / Rprop / RAdam / ASGD.

Usage
-----
::

    model = MyModel().to('metal')
    opt   = optim.Adam(model.parameters(), lr=0.001)
    step  = lucid.compile.fused_step(model, F.mse_loss, opt)

    for x, t in loader:
        loss = step(x, t)        # forward + bwd + opt.step in one go
        # No backward(), no opt.step() — already done.
        if want_logging:
            print(float(loss.item()))

Limitations
-----------
* Single param_group only (the underlying compile_optimizer constraint).
* No dynamic batch — shape-locked to the first call's input signature.
* The loss tensor returned has no ``grad_fn`` (the backward has
  already run inside the executable).  ``loss.backward()`` is a no-op.
"""

import threading
from typing import TYPE_CHECKING, Callable

from lucid._C import engine as _C_engine

# Thread-local flag flipped on while ``_FusedStep._build_executable``
# is actively tracing.  ``lucid.nn.functional.dropout`` checks it to
# decide whether to route training-mode dispatch through the
# ``dropout_stateful`` engine op (which only works when the compile
# path uses ``compile_generic_fused_step_with_vars`` to promote the
# state buffer to an MPSGraph variable).  The forward-only
# :class:`CompiledModule` path doesn't set this flag — its compile
# entry (``compile_trace``) has no variable-promotion hook, so
# training-mode dropout there continues to fall back to eager.
_tls = threading.local()


def _is_fused_step_tracing() -> bool:
    """Return True while inside ``_FusedStep._build_executable``'s trace."""
    return bool(getattr(_tls, "active", False))


# Hot-path imports — hoisted out of ``_FusedStep._run`` so the
# per-call ``from ... import _unwrap`` / ``import lucid as _lucid``
# bookkeeping (≈ 30-50 μs on M-series) doesn't run every training
# step.  Phase 1.10 per-call overhead reduction.
from lucid._dispatch import _unwrap as _unwrap_hot
import lucid as _lucid_hot

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module
    from lucid.optim.optimizer import Optimizer

__all__ = ["fused_step"]


def fused_step(
    model: Module,
    loss_fn: Callable[..., Tensor],
    optimizer: Optimizer,
) -> Callable[..., Tensor]:
    """Return a callable that runs one fused training step.

    The first call traces ``model(*x); loss_fn(out, *targets);
    optimizer._trace_update(...)`` once under a single :class:`Tracer`,
    plumbs the ghost-grad placeholders, and compiles the resulting
    graph into one :class:`MPSGraphExecutable` that runs forward +
    loss + backward (via MPSGraph autodiff) + optimizer update in a
    single submission.  Subsequent calls reuse the cached executable.

    Delegates the optimizer math to the matching
    :func:`compile_optimizer` subclass — so all 8 supported optimizers
    (SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax, NAdam)
    work in fused mode automatically.

    Parameters
    ----------
    model : Module
        The trainable :class:`nn.Module`.  Parameters from
        ``optimizer.param_groups`` must reference these tensors.
    loss_fn : Callable
        Scalar-loss-returning callable invoked as
        ``loss_fn(model_output, *targets)``.  Captured by identity in
        the trace, so passing a fresh closure each call defeats the
        cache.
    optimizer : Optimizer
        One of the 8 supported optimizers.  Unsupported optimizers
        raise :class:`NotImplementedError` from
        :func:`compile_optimizer` with the structural reason
        (line-search / sign branch / per-step coefficient / …).

    Returns
    -------
    Callable[..., Tensor]
        A callable ``step(*args)`` where ``args`` is
        ``(model_input, *loss_targets)``.  Returns the scalar loss
        :class:`Tensor` from the just-completed step (the parameter
        and optimizer-state buffers have already been updated
        in-place inside the executable).

    Examples
    --------
    >>> import lucid, lucid.nn as nn, lucid.nn.functional as F
    >>> import lucid.optim as optim
    >>> from lucid.compile import fused_step
    >>> model = nn.Linear(8, 4).to('metal')
    >>> opt = optim.Adam(model.parameters(), lr=1e-3)
    >>> step = fused_step(model, F.cross_entropy, opt)
    >>> for batch in batches:                        # doctest: +SKIP
    ...     loss = step(batch.x, batch.target)       # one executable submission
    ...     # opt.step() is implicit — parameters already updated in-place

    See Also
    --------
    lucid.compile.make_step : forward+backward only, no optimizer fusion
        (lets the caller drive the optimizer in Python).
    lucid.compile.compile_optimizer : the underlying optimizer-side
        translator that produces the update graph.
    """
    return _FusedStep(model, loss_fn, optimizer)


def _zeros_like(t: Tensor) -> Tensor:
    """Allocate a same-shape, same-dtype, same-device zero-filled Tensor.

    Used to materialise the *ghost-grad placeholders* fed into the
    trace: the executable will overwrite each placeholder with the
    actual gradient computed by MPSGraph's
    ``gradientForPrimaryTensor:`` at compile time, so the runtime
    value doesn't matter — only the shape + dtype + device that
    pin the trace placeholder's identity.
    """
    import lucid as _lucid

    return _lucid.zeros(*t.shape, dtype=t.dtype, device=t.device)


class _FusedStep:
    r"""Driver behind :func:`fused_step` — one executable, whole training step.

    Composes :mod:`lucid.compile._optim` (which already knows how to
    trace each optimizer's update arithmetic and identify the in-
    place output targets) with a trace of the model's forward pass +
    loss function.  The result is one :class:`MPSGraphExecutable`
    whose op DAG covers:

    .. code-block:: text

        x ─▶ model(x) ─▶ loss_fn(out, *targets) ─▶ loss
                              │
                              ▼
                ∂loss/∂param  (auto-derived inside C++ via
                              gradientForPrimaryTensor:withTensors:)
                              │
                              ▼
                opt_update(param, grad, state, scalars) ─▶ new_param, new_state
                              │
                              ▼
                          (in-place write back)

    Ghost-grad mechanism
    --------------------
    The challenge is that the optimizer update reads
    ``param.grad``, but the trace runs under :func:`no_grad` so no
    eager gradient exists yet.  We sidestep this by:

    1. Allocating one **ghost-grad placeholder** Tensor per
       parameter (a same-shape, same-dtype zero tensor).
    2. Feeding the placeholders into ``copt._trace_update(...)`` so
       the optimizer math captures them as graph inputs and writes
       the update math that reads them.
    3. Calling ``compile_generic_fused_step`` with the
       ``ghost_grad_ids``.  The C++ builder, before emitting the
       optimizer ops, asks MPSGraph to derive
       ``gradientForPrimaryTensor: loss withTensors: params`` and
       binds each ``ghost_grad_id`` to the corresponding derived
       gradient.  When the opt-update ops are then emitted, their
       ghost-grad reads resolve to the actual auto-derived
       gradients.

    The executable's I/O therefore looks like:

    * Inputs: positional model args (per call), pinned parameters,
      state buffers, per-step scalars, and the **ghost-grad
      placeholders** (their values don't matter because the C++
      side rebinds them).
    * Outputs (all in-place writes): loss scalar, new parameters,
      new state buffers.

    Attributes
    ----------
    _model : nn.Module
        The compute unit being traced.
    _loss_fn : Callable
        Scalar-returning callable invoked as
        ``loss_fn(model_output, *targets)``.
    _copt : _CompiledStepBase
        The optimizer subclass driving the update math and
        state-buffer plan.
    _params : list[Tensor]
        Flat parameter list (shared with ``_copt._params``).
    _exe : object or None
        Cached executable, lazily compiled on the first call.
    _input_resolvers : list[Callable]
        One zero-arg getter per input slot in
        ``exe.input_ids`` order — each returns the live Tensor to
        bind for that slot at the current step.
    _output_targets : list[Tensor]
        Tensors that receive the executable's in-place writes
        (params + state buffers, in trace return order).
    _loss_shape / _loss_dtype / _loss_device
        Captured at compile time so each step allocates a fresh
        scalar loss tensor with matching meta.

    See Also
    --------
    :func:`fused_step` : user-facing constructor.
    :func:`compile_optimizer` : the underlying optimizer compile
        path whose ``_trace_update`` hook is reused here.
    :class:`CompiledModule` : forward-only compile (no
        backward+update fusion).
    """

    def __init__(
        self,
        model: Module,
        loss_fn: Callable[..., Tensor],
        optimizer: Optimizer,
    ) -> None:
        """Initialise the driver; the executable itself is lazy.

        Parameters
        ----------
        model, loss_fn, optimizer
            See :func:`fused_step` for semantics.

        Raises
        ------
        ValueError
            If ``optimizer`` exposes no trainable parameters (every
            param_group is empty).
        """
        from lucid.compile._optim import compile_optimizer

        self._model = model
        self._loss_fn = loss_fn
        # The compile_optimizer subclass owns the optimizer math + state
        # buffers + scalar feeds + output_targets.  Any optimizer it
        # supports (or rejects) propagates automatically here.
        self._copt = compile_optimizer(optimizer)
        self._params = self._copt._params
        if not self._params:
            raise ValueError("fused_step: optimizer has no trainable parameters")

        self._exe: object | None = None
        # Per-call args stash so positional resolvers can pick them up.
        self._current_args: tuple = ()
        # Built at compile time, indexed by exe.input_ids.
        self._input_resolvers: list[Callable[[], Tensor]] = []
        # output_targets list parallel to exe.grad_output_ids (the opt
        # outputs in order: new_params + new_state buffers).
        self._output_targets: list[Tensor] = []
        # Loss meta for fresh allocation each step.
        self._loss_shape: tuple = ()
        self._loss_dtype: object = None
        self._loss_device: object = None

    # ── Public API ──────────────────────────────────────────────

    def __call__(self, *args: Tensor) -> Tensor:
        """Run one fused training step; lazy-compile on first call.

        Parameters
        ----------
        *args : Tensor
            ``(model_input, *loss_targets)`` — same positional tuple
            that ``loss_fn(model(input), *targets)`` would consume.

        Returns
        -------
        Tensor
            Scalar loss tensor freshly allocated for this step.  All
            parameter and optimizer-state buffers have already been
            updated **in-place** before this returns.
        """
        if self._exe is None:
            self._build_executable(args)
        # Refresh per-step scalars (e.g. Adam bias correction).  The
        # optimizer subclass owns this.
        self._copt._refresh_scalars()
        return self._run(args)

    def recompile(self) -> None:
        """Drop the cached executable so the next call retraces from scratch.

        Useful after manual surgery on the model or optimizer state
        (e.g. resizing a parameter buffer) where the captured tensor
        identities no longer match the live ones.  Normal training
        loops never need to call this — the executable amortises
        compile cost across every subsequent step.
        """
        self._exe = None

    # ── Internals ───────────────────────────────────────────────

    def _build_executable(self, args: tuple[Tensor, ...]) -> None:
        """First-call trace + compile sequence.

        Walks the optimizer subclass to (a) prime its state and
        scalar buffers, (b) materialise ghost-grad placeholders to
        plumb into the trace, (c) record forward + loss +
        optimizer-update as a single :class:`TraceGraph`, then (d)
        call into ``compile_generic_fused_step`` which threads the
        ghost-grad ids through MPSGraph's autodiff and returns one
        executable producing loss + parameter updates in one shot.

        Mutates ``self._exe``, ``self._input_resolvers``,
        ``self._output_targets``, and ``self._loss_*`` on success.
        Raises :class:`RuntimeError` with a structural reason on any
        failure (empty trace, missing parameter id, builder
        rejection) — fused_step intentionally has no eager fallback
        path because every step would silently lose the speedup.
        """
        from lucid._dispatch import _unwrap
        from lucid._tensor.tensor import Tensor
        from lucid.autograd._grad_mode import no_grad
        from lucid.compile import _tracing

        copt = self._copt
        # The compile_optimizer subclass allocates its state buffers in
        # __init__.  Force its scalar holders to exist now (they're
        # registered inside ``_register_scalars``, called from
        # ``_build_executable`` of the base — but we're not going
        # through that path; replicate the relevant parts here).
        scalars = copt._register_scalars(lambda kind, idx, t: None)
        # Allocate ghost grad placeholders that stand in for the
        # gradient inputs.  Their TensorImpl identity goes into the
        # trace as external feeds; the C++ compile binds them to the
        # MPSGraph-derived gradients before emitting the opt-update
        # portion of the graph.
        ghost_grads = [_zeros_like(p) for p in self._params]

        # Trace forward + loss + opt update in one block.  Flip the
        # thread-local flag so the dropout wrapper knows it can route
        # training-mode dispatch through ``dropout_stateful`` (we have
        # the variable-promotion machinery downstream).
        #
        # AMP scoping (X4.4): the user may wrap the entire
        # ``step(x, t)`` call in ``with autocast()``.  Per the standard
        # PyTorch convention, autocast applies only to forward + loss
        # — backward and optimizer.step run on F32 master weights so
        # the update math stays numerically stable.  We split the
        # scope explicitly here: the captured ``with _tracing()``
        # block respects the user's autocast for model + loss (so
        # the trace records the right ``astype`` casts that the
        # ``astype`` VJP from P1 cleanly differentiates), then
        # temporarily installs a neutral ``AutocastGuard(F32)``
        # before ``copt._trace_update`` to ensure the optimizer
        # arithmetic runs in F32.  Without this split, the
        # optimizer's reads of F32 master weights would get autocast
        # to F16 → F16 ``new_param`` → ``run_executable_inplace``
        # dtype mismatch with the F32 param buffer.
        from lucid._C import engine as _C_engine

        _autocast_was_active = _C_engine.amp_is_active()
        _autocast_prev_dtype = _C_engine.amp_active_dtype()

        _tls.active = True
        try:
            with no_grad():
                with _tracing() as tracer:
                    out = self._model(*args[:1])
                    loss = self._loss_fn(out, *args[1:])
                    # Disable autocast for the optimizer math.  The
                    # F32 guard is the canonical "off" sentinel — the
                    # engine has no ``disable_amp()`` primitive, so
                    # we install a guard that targets F32 (identity
                    # for F32 weights, which is what we want).
                    if _autocast_was_active:
                        _opt_guard = _C_engine.AutocastGuard(_C_engine.F32)
                        _opt_guard.__enter__()
                    else:
                        _opt_guard = None
                    try:
                        # Pass ghost grads as the "grad" inputs to the
                        # optimizer math.  The subclass's _trace_update
                        # emits Lucid tensor ops referencing them; those
                        # become ghost-grad-consuming ops in the trace,
                        # which C++ will handle in the second emit phase.
                        opt_outputs = copt._trace_update(
                            None,  # all_inputs unused by current _trace_update impls
                            ghost_grads,
                            scalars,
                        )
                    finally:
                        if _opt_guard is not None:
                            # Restore user's autocast dtype after the
                            # opt scope.  AutocastGuard destructor
                            # restores the prev_active state captured
                            # at __enter__ time, but the Python wrapper
                            # is RAII via context manager — replicate
                            # that pattern.
                            if (
                                _autocast_prev_dtype is not None
                                and _autocast_was_active
                            ):
                                _restore = _C_engine.AutocastGuard(_autocast_prev_dtype)
                                _restore.__enter__()
        finally:
            _tls.active = False

        graph = tracer.graph
        ext = dict(tracer.external_feeds)
        if not graph.ops:
            raise RuntimeError("fused_step: empty trace")

        loss_id = int(tracer.lookup_id(_unwrap(loss)))
        self._loss_shape = tuple(loss.shape)
        self._loss_dtype = loss.dtype
        self._loss_device = loss.device

        # Resolve param ids.
        param_ids: list[int] = []
        for p in self._params:
            tid = tracer.lookup_id(_unwrap(p))
            if tid is None:
                raise RuntimeError(
                    "fused_step: a parameter was not observed in the "
                    "trace (likely it isn't used in forward)"
                )
            param_ids.append(int(tid))

        # Resolve ghost grad ids — must be in same order as param_ids.
        ghost_grad_ids: list[int] = []
        for g in ghost_grads:
            tid = tracer.lookup_id(_unwrap(g))
            if tid is None:
                raise RuntimeError(
                    "fused_step: ghost grad placeholder missing from trace"
                )
            ghost_grad_ids.append(int(tid))

        # Resolve opt-output ids (new_params + new_state, in the order
        # the subclass produced them).
        output_target_ids: list[int] = []
        for o in opt_outputs:
            tid = tracer.lookup_id(_unwrap(o))
            if tid is None:
                raise RuntimeError(
                    "fused_step: optimizer output tensor missing from trace"
                )
            output_target_ids.append(int(tid))

        # Append every training-mode dropout's ``state_out`` id so the
        # ``compile_generic_fused_step_with_vars`` call below can pair
        # them with their ``state_in`` feeds — required by that API
        # because every ``write_id`` in ``variable_pairs`` must also
        # appear in ``output_target_ids``.  The parallel Python-side
        # target Tensor (the same buffer that was ``state_in``) is
        # appended to ``self._output_targets`` further down so
        # ``run_executable_inplace`` has somewhere to flush the
        # readVariable output (harmless for dropout state since we
        # never read it from Python — the buffer rotation is purely
        # what advances the RNG sequence across dispatches).
        dropout_state_target_pairs: list[tuple[int, int]] = []
        for _node in graph.ops:
            if _node.name == "dropout_stateful":
                if len(_node.inputs) >= 2 and len(_node.outputs) >= 2:
                    _sin = int(_node.inputs[1])
                    _sout = int(_node.outputs[1].id)
                    output_target_ids.append(_sout)
                    dropout_state_target_pairs.append((_sin, _sout))

        # ``compile_generic_fused_step_with_vars`` (MPSGraph stateful
        # variables variant) is now functional after the source-graph
        # retention fix in CompiledExecutable.mm — previously the
        # source ``MPSGraph`` was released at the end of the compile
        # autoreleasepool, freeing the variable's MTLBuffer and causing
        # a SIGSEGV inside ``GPU::VarHandleOpHandler::encodeOp`` on the
        # first ``runWithMTLCommandQueue:`` (manifested as an
        # indefinite hang because the GPU command buffer never
        # completed).  Gated behind ``LUCID_COMPILE_VARS=1`` until a
        # follow-up PR validates performance + memory characteristics
        # at model-zoo scale; the default path stays on the in/out-feed
        # ``compile_generic_fused_step`` so the long-run + training
        # regression matrix continues to exercise the production code.
        import os as _os

        # Force the ``_with_vars`` path whenever any training-mode
        # dropout is present in the trace — every ``dropout_stateful``
        # op must promote its ``(state_in, state_out)`` to an MPSGraph
        # variable so per-dispatch RNG state advances correctly.
        # Without variable promotion the state buffer would be
        # re-initialised on every dispatch and every call would emit
        # the same mask (the very regression the prior X2 prototype
        # ran into — see ``test_dropout_training_produces_random_outputs``).
        _use_vars = _os.environ.get("LUCID_COMPILE_VARS", "0") in (
            "1",
            "true",
            "True",
        ) or bool(dropout_state_target_pairs)
        if _use_vars:
            # Build (feed_id, write_id) pairs: each parameter feed
            # becomes a variable, paired with the matching opt-output
            # id (the "new_param" tensor for that parameter).  The
            # opt-output order matches param_ids order in every
            # ``_trace_update`` implementation — the first N opt
            # outputs are the new params, followed by new state
            # buffers.  Only the param-tier entries get promoted to
            # variables; momenta / m / v stay as input/output feeds.
            #
            # State-buffer promotion was investigated (2026-05-25,
            # tracked alongside Tier 2-A): replacing the swap-buffer
            # dance for m / v / momenta with ``assignVariable:``
            # produced a **regression** of +10–20% per step on every
            # measured workload (mlp / deep_mlp / Adam / SGD).  The
            # MPSGraph variable path appears to serialise around
            # ``assignVariable`` writes in a way the in/out-feed
            # double-buffer doesn't on M-series.  Conclusion: keep
            # variables for the param tier only; state stays on the
            # feed/output path.  See ``obsidian/perf/perf-state-vars-regression.md``
            # for the bench data.
            variable_pairs: list[tuple[int, int]] = []
            # Param-tier promotion runs only when LUCID_COMPILE_VARS=1
            # opts in.  When the only reason ``_use_vars`` flipped is
            # dropout state plumbing, params stay on the in/out feed
            # path (matches the perf measurement in
            # ``obsidian/perf/perf-state-vars-regression.md`` that
            # found promoting large state buffers to variables
            # regressed 10-20 % per step).
            _params_as_vars = _os.environ.get("LUCID_COMPILE_VARS", "0") in (
                "1",
                "true",
                "True",
            )
            if _params_as_vars:
                for i, pid in enumerate(param_ids):
                    if i < len(output_target_ids):
                        variable_pairs.append((pid, output_target_ids[i]))
            # Dropout-train state ALWAYS goes through variable
            # promotion when present.  These pairs are tiny
            # (int32[7] per dropout site, 28 bytes) so the
            # serialisation overhead documented for large optimizer
            # state in ``perf-state-vars-regression.md`` doesn't
            # materially apply.
            variable_pairs.extend(dropout_state_target_pairs)
            exe = _C_engine.compile.compile_generic_fused_step_with_vars(
                graph,
                ext,
                loss_id,
                param_ids,
                ghost_grad_ids,
                output_target_ids,
                variable_pairs,
            )
        else:
            exe = _C_engine.compile.compile_generic_fused_step(
                graph,
                ext,
                loss_id,
                param_ids,
                ghost_grad_ids,
                output_target_ids,
            )
        if exe is None:
            raise RuntimeError(
                "fused_step: compile_generic_fused_step returned None — "
                "an op in the combined trace has no emitter, or the "
                "trace is otherwise incompatible with the fused path."
            )

        # Build per-input resolvers (impl identity → live tensor getter).
        impl_to_resolver: dict[int, Callable[[], Tensor]] = {}
        for i, p in enumerate(self._params):
            impl_to_resolver[id(_unwrap(p))] = lambda _i=i: self._params[_i]
        # State buffers + scalar holders from the compile_optimizer.
        for (kind, idx), getter in copt._buffer_table.items():
            impl_to_resolver[id(_unwrap(getter()))] = getter
        for name, t in copt._scalar_slots.items():
            impl_to_resolver[id(_unwrap(t))] = lambda _n=name: copt._scalar_slots[_n]
        # Positional model inputs (keyed by impl id captured here, but
        # read from self._current_args at run time).
        for slot, a in enumerate(args):
            if isinstance(a, Tensor):
                impl_to_resolver[id(_unwrap(a))] = lambda _s=slot: self._current_args[
                    _s
                ]

        # Pinned constants (mirror of CompiledModule's ``input_source``
        # treatment): if a trace external_feed isn't a known
        # param/state/scalar/positional, save the original TensorImpl
        # and return it verbatim every call.  Covers ad-hoc tensors
        # the model materialises inside ``forward()`` and whose impl
        # identity isn't stable across calls (``Conv2d(bias=False)``
        # produces a fresh zero-bias tensor on each forward via
        # :func:`conv_bias_or_zero`; ``BatchNorm`` running-stats
        # buffers; SiLU's constant scalars; etc.).  Using the
        # first-trace impl is correct for these because they're
        # semantic constants — no per-call updates happen on them
        # through the fused step.
        from lucid._tensor.tensor import Tensor as _TensorT  # noqa: PLC0415

        resolvers: list[Callable[[], Tensor]] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                raise RuntimeError(f"fused_step: input id {tid} not in external_feeds")
            r = impl_to_resolver.get(id(impl))
            if r is None:
                # Pin the impl directly — wrap as a Tensor view (no
                # grad, no autograd hook) and return it verbatim every
                # call.
                pinned_tensor = _TensorT(impl, requires_grad=False)
                r = lambda _t=pinned_tensor: _t
            resolvers.append(r)
        self._input_resolvers = resolvers

        # Output targets: the compile_optimizer subclass already knows
        # which Tensors receive the opt outputs.  Same order as
        # opt_outputs in _trace_update.  Then append one Tensor per
        # dropout state_out_tid we added to ``output_target_ids`` —
        # the readVariable target for each promoted variable.  We wrap
        # the same TensorImpl that was the ``state_in`` feed so the
        # buffer write-back lands in the same Python-side tensor
        # (harmless overhead since we never read it; the actual RNG
        # advancement happens inside the variable's internal storage).
        self._output_targets = list(copt._outputs_to_targets(opt_outputs))
        from lucid._tensor.tensor import Tensor as _TensorT_for_state

        for _state_in_tid, _state_out_tid in dropout_state_target_pairs:
            _impl = ext.get(_state_in_tid)
            if _impl is None:
                raise RuntimeError(
                    "fused_step: dropout state feed id "
                    f"{_state_in_tid} not in external_feeds"
                )
            self._output_targets.append(_TensorT_for_state(_impl, requires_grad=False))

        if len(self._output_targets) != len(output_target_ids):
            raise RuntimeError(
                "fused_step: output_targets count "
                f"({len(self._output_targets)}) doesn't match "
                f"output_target_ids count ({len(output_target_ids)})"
            )
        self._exe = exe
        # Phase 1.10 per-call overhead reduction: pre-unwrap the
        # opt-output target TensorImpls.  These are stable across calls
        # (params + state buffers live for the lifetime of the model),
        # so we pay the unwrap once at compile time rather than every
        # ``_run``.  The list is rebuilt with a fresh loss_impl at
        # slot 0 each call (loss is the only non-stable target).
        from lucid._dispatch import _unwrap as _unwrap_hot

        self._opt_target_impls = tuple(_unwrap_hot(t) for t in self._output_targets)

    def _run(self, args: tuple[Tensor, ...]) -> Tensor:
        """Bind feeds + targets and invoke the cached executable in-place.

        Uses ``run_executable_inplace`` so the optimizer's
        parameter and state buffers are mutated directly inside the
        executable (no allocation churn between steps).  Allocates a
        fresh scalar loss tensor each call — the caller usually
        reads it once and drops it, and the cost is a single 0-D
        allocation per step.

        Parameters
        ----------
        args : tuple of Tensor
            Same per-step arguments forwarded from :meth:`__call__`.

        Returns
        -------
        Tensor
            Scalar loss for this step.
        """
        self._current_args = args
        feeds = [_unwrap_hot(r()) for r in self._input_resolvers]

        # Fresh loss tensor (single-element output that flows back to
        # Python; not in-place since callers want a clean handle).
        loss_tensor = _lucid_hot.zeros(
            *self._loss_shape if self._loss_shape else (),
            dtype=self._loss_dtype,
            device=self._loss_device,
        )

        # Output target list: [loss_scratch, opt_target_impls...].
        # ``_opt_target_impls`` was pre-unwrapped at compile time;
        # using ``list(tuple)`` then ``append`` is faster than the
        # previous append-in-loop pattern.
        output_targets = [_unwrap_hot(loss_tensor), *self._opt_target_impls]

        _C_engine.compile.run_executable_inplace(self._exe, feeds, output_targets)
        return loss_tensor
