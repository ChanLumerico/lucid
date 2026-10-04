"""
lucid.compile._entry.fused_step — Phase 1.8 generic fused training step.

``fused_step(model, loss_fn, optimizer)`` returns a callable that runs
the entire training step (forward + loss + auto-derived gradients +
optimizer.step) inside a single MPSGraph executable.  One ``run`` call
per training iteration, all writes to parameters / optimizer state
happen in-place via ``run_executable_inplace``.

Architecture
------------
The optimizer update math lives in the corresponding
:mod:`lucid.compile._optim.compiler` subclass (``_CompiledSGD``,
``_CompiledAdam``, ..., ``_CompiledNAdam``).  On the first call:

1. Open a single :class:`Tracer`.  Run ``model(x); loss_fn(out, t)``
   (the forward + loss path).
2. Find the parameters the loss depends on (its ancestors in the trace)
   that also require grad — the parameters eager ``backward()`` would
   give a gradient.  Only those take part: MPSGraph's autodiff aborts
   the process when asked for the gradient of a tensor that does not
   precede the loss, and eager leaves such a parameter untouched.
3. Emit the optimizer update of those parameters with **ghost grad
   placeholders** (zero tensors that take the slot of "param_i.grad").
4. Call ``compile_generic_fused_step`` with the ghost grad ids.  C++
   derives gradients via MPSGraph autograd (or the manual VJPs) after
   emitting the forward, binds each ghost grad id to its derived
   gradient, then emits the update ops.
5. Each ``step(x, t)`` call: pack this step's hyper-parameters and
   per-parameter scalars (read from ``param_groups`` — LR schedulers
   take effect), then ``run_executable_inplace`` writes new params /
   new state directly into the corresponding tensors.

Freezing or unfreezing a parameter (``requires_grad_``) or flipping a
structural hyper-parameter (weight decay on/off …) builds another
executable on the next call; every executable built is kept, so
switching back is free.

Supported optimizers
--------------------
Every optimizer that :func:`compile_optimizer` accepts — all 13 of
Lucid's eager optimizers, any number of parameter groups.

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
* No dynamic batch — shape-locked to the first call's input signature.
* The loss tensor returned has no ``grad_fn`` (the backward has
  already run inside the executable).  ``loss.backward()`` is a no-op.
"""

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, final

from contextlib import contextmanager

from lucid._C import engine as _C_engine
from lucid._device import device as _device_cls
from lucid._dtype import dtype as _dtype_cls
from lucid.compile._core.bn_runstats import advance_bn_counters, bn_counter_targets

# Thread-local flag flipped on while ``_FusedStep._build_plan``
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
    """Return True while inside ``_FusedStep._build_plan``'s trace."""
    return bool(getattr(_tls, "active", False))


# Hot-path imports — hoisted out of ``_FusedStep._run`` so the
# per-call ``from ... import _unwrap`` / ``import lucid as _lucid``
# bookkeeping (≈ 30-50 μs on M-series) doesn't run every training
# step.  Phase 1.10 per-call overhead reduction.
from lucid._dispatch import _unwrap as _unwrap_hot
import lucid as _lucid_hot

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.amp.grad_scaler import GradScaler
    from lucid.compile._optim.compiler import _Flags, _ScalarLayout
    from lucid.nn.module import Module
    from lucid.optim.optimizer import Optimizer

__all__ = ["fused_step"]

# Executables a single fused step keeps (oldest dropped first).
_MAX_PLANS: int = 8


def fused_step(
    model: Module,
    loss_fn: Callable[..., Tensor],
    optimizer: Optimizer,
    *,
    grad_scaler: GradScaler | None = None,
) -> Callable[..., Tensor]:
    """Return a callable that runs one fused training step.

    The first call traces ``model(*x); loss_fn(out, *targets)`` and the
    optimizer update once under a single :class:`Tracer`, plumbs the
    ghost-grad placeholders, and compiles the resulting graph into one
    :class:`MPSGraphExecutable` that runs forward + loss + backward (via
    MPSGraph autodiff) + optimizer update in a single submission.
    Subsequent calls reuse the cached executable.

    The step matches an eager ``zero_grad(); loss.backward();
    optimizer.step()`` loop: a parameter the loss does not depend on, or
    one with ``requires_grad=False``, gets no gradient and is left
    untouched (state and step count included), and every hyper-parameter
    is read from ``optimizer.param_groups`` on each call, so LR
    schedulers work.  Checkpoint the optimizer through
    :attr:`optimizer` — the compiled optimizer whose ``state_dict``
    holds the state this step updates.

    Delegates the optimizer math to the matching
    :func:`compile_optimizer` subclass — so every Lucid eager optimizer
    (all 13: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax,
    NAdam, SparseAdam, Rprop, ASGD, RAdam, LBFGS) compiles cleanly
    into a fused step.  LBFGS is the closure-less single-step variant
    (per-element Barzilai-Borwein direction); full closure-driven
    line search remains eager-only.

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
        Any optimizer :func:`compile_optimizer` supports, with any
        number of parameter groups.
    grad_scaler : GradScaler, optional
        When provided and enabled, the fused step replicates the
        eager :class:`lucid.amp.GradScaler` contract entirely inside
        the compiled executable:

        1. The loss is multiplied by the current ``scaler._scale``
           before the backward derivation, so MPSGraph autograd
           produces scaled gradients that won't underflow in F16.
        2. Each scaled gradient is unscaled in F32 by
           ``1/scaler._scale`` before the optimizer reads it.
        3. ``found_inf = OR(any(!isfinite(g_unscaled)))`` is computed
           across all params; each new_param / new_state is wrapped
           with ``where(found_inf, old, new)`` so an overflow step
           leaves params + state buffers unchanged — and, as with
           ``scaler.step(optimizer)``, the step counts do not advance.
        4. After the executable runs, ``found_inf`` is read back to
           Python and ``scaler.update()`` is invoked — the scale
           halves on overflow, doubles after ``growth_interval``
           clean steps.

        The user-visible loss returned by ``step(*args)`` is always
        the **unscaled** loss (the trace divides by scale on the
        return path), matching the eager API exactly.

    Returns
    -------
    Callable[..., Tensor]
        A callable ``step(*args)`` where ``args`` is
        ``(model_input, *loss_targets)``.  Returns the scalar loss
        :class:`Tensor` from the just-completed step (the parameter
        and optimizer-state buffers have already been updated
        in-place inside the executable).

    Raises
    ------
    RuntimeError
        On the first call, when no parameter both requires grad and
        reaches the loss (there would be nothing to step).

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
    return _FusedStep(model, loss_fn, optimizer, grad_scaler=grad_scaler)


def _prod_shape(shape: tuple[int, ...]) -> int:
    """Product of the dims in ``shape`` (treating 0-D as 1).

    Used by the GradScaler integration: the ``found_inf`` reduction
    sums ``isfinite(g)`` to a scalar and compares against the
    expected total (the param's numel).  Computed at trace time
    from the trace-recorded shape, not from a graph reduction, so
    the comparison constant is a pure Python int / float baked into
    the trace via ``lucid.tensor(expected_total)``.
    """
    n = 1
    for d in shape:
        n *= int(d)
    return n


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


@contextmanager
def _untraced(tracer: object) -> Iterator[None]:
    """Detach ``tracer`` for the block, then put it back.

    Optimizer state and placeholders are allocated in the middle of the
    trace (only once the forward shows which parameters take part); a
    factory call recorded there would turn a feed into a traced op.
    """
    _C_engine.compile.set_current_tracer(None)
    try:
        yield
    finally:
        _C_engine.compile.set_current_tracer(tracer)


def _loss_ancestors(graph: object, loss_id: int) -> set[int]:
    """Every trace id the tensor ``loss_id`` depends on (itself included).

    One reverse walk over the ops recorded so far — the forward and the
    loss, since the optimizer update has not been traced yet.
    """
    needed = {loss_id}
    for node in reversed(graph.ops):  # type: ignore[attr-defined]
        if any(int(meta.id) in needed for meta in node.outputs):
            needed.update(int(i) for i in node.inputs if int(i) >= 0)
    return needed


@final
@dataclass
class _FusedPlan:
    """One compiled fused step: its executable and how to feed / drain it."""

    exe: object
    members: tuple[int, ...]
    layout: _ScalarLayout
    resolvers: list[Callable[[], _C_engine.TensorImpl]]
    target_impls: tuple[_C_engine.TensorImpl, ...]
    bn_counters: list[tuple[Module, int]]
    loss_shape: tuple[int, ...]
    loss_dtype: _dtype_cls
    loss_device: _device_cls


@final
class _FusedStep:
    r"""Driver behind :func:`fused_step` — one executable, whole training step.

    Composes :mod:`lucid.compile._optim.compiler` (which already knows how to
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
       participating parameter (a same-shape, same-dtype zero tensor).
    2. Feeding the placeholders into the optimizer's update emitter
       so the optimizer math captures them as graph inputs.
    3. Calling ``compile_generic_fused_step`` with the
       ``ghost_grad_ids``.  The C++ builder, before emitting the
       optimizer ops, derives the gradient of the loss with respect to
       each participating parameter and binds each ``ghost_grad_id`` to
       it.

    Which parameters take part
    --------------------------
    Those the loss depends on that also require grad — exactly the ones
    an eager ``backward()`` leaves a gradient on.  The set is part of
    the executable's key: ``requires_grad_`` between calls selects (or
    builds) another executable.

    Attributes
    ----------
    _model : nn.Module
        The compute unit being traced.
    _loss_fn : Callable
        Scalar-returning callable invoked as
        ``loss_fn(model_output, *targets)``.
    _copt : _CompiledStepBase
        The compiled optimizer driving the update math and owning the
        optimizer state (exposed as :attr:`optimizer`).
    _plans : dict
        Compiled executables keyed by (participating parameters,
        structural flags of every group, per-parameter scalar classes).
    _reachable : frozenset[int] or None
        Flat indices of the parameters the loss depends on; found on the
        first trace.

    See Also
    --------
    :func:`fused_step` : user-facing constructor.
    :func:`compile_optimizer` : the underlying optimizer compile
        path whose update emitter is reused here.
    :class:`CompiledModule` : forward-only compile (no
        backward+update fusion).
    """

    def __init__(
        self,
        model: Module,
        loss_fn: Callable[..., Tensor],
        optimizer: Optimizer,
        *,
        grad_scaler: GradScaler | None = None,
    ) -> None:
        """Initialise the driver; the executable itself is lazy.

        Parameters
        ----------
        model, loss_fn, optimizer
            See :func:`fused_step` for semantics.
        grad_scaler : GradScaler, optional
            See :func:`fused_step`.

        Raises
        ------
        ValueError
            If ``optimizer`` exposes no trainable parameters (every
            param_group is empty).
        """
        from lucid.compile._optim.compiler import compile_optimizer

        self._model = model
        self._loss_fn = loss_fn
        # The compiled optimizer owns the optimizer math + state buffers
        # + per-step scalars.  Any optimizer it supports (or rejects)
        # propagates automatically here.
        self._copt = compile_optimizer(optimizer)
        if not self._copt._params:
            raise ValueError("fused_step: optimizer has no trainable parameters")
        self._params_seen: list[Tensor] = self._copt._params

        # GradScaler integration (X4.3).  None ⇒ unscaled path.
        self._grad_scaler: GradScaler | None = grad_scaler
        # Stable 0-D scalar holders refreshed each step via ``copy_``
        # so the cached executable keeps its TensorImpl identity.
        # Allocated lazily at the first build.
        self._scale_holder: Tensor | None = None
        self._inv_scale_holder: Tensor | None = None
        # Persistent F32 0-D holder for the found_inf output.  Read
        # back after each run to drive ``scaler.update()``.
        self._found_inf_target: Tensor | None = None

        self._plans: dict[tuple[object, ...], _FusedPlan] = {}
        self._reachable: frozenset[int] | None = None
        # Per-call args stash so positional resolvers can pick them up.
        self._current_args: tuple[Tensor, ...] = ()
        # This call's packed scalar vectors (read by their resolvers).
        self._step_vectors: list[_C_engine.TensorImpl] = []
        # The "scale that was applied this step" — captured before
        # ``run`` so the user-visible loss can be unscaled afterwards
        # even if the scaler's schedule advances between calls.
        self._last_applied_scale: float = 1.0

    # ── Public API ──────────────────────────────────────────────

    @property
    def optimizer(self) -> object:
        """The compiled optimizer this step updates — checkpoint through it.

        Its ``state_dict()`` / ``load_state_dict()`` carry the state the
        fused executable reads and writes, in the eager format.
        """
        return self._copt

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
        copt = self._copt
        copt._sync_params()
        if copt._params is not self._params_seen:
            # ``add_param_group`` re-flattened the parameters.
            self._params_seen = copt._params
            self._plans.clear()
            self._reachable = None
        flags = copt._flags_now()
        plan: _FusedPlan | None = None
        if self._reachable is not None:
            plan = self._plans.get(self._key(self._members(), flags))
        if plan is None:
            plan = self._build_plan(args, flags)
            # Keyed after the build: it initialised the new members' state.
            self._plans[self._key(plan.members, flags)] = plan
            while len(self._plans) > _MAX_PLANS:
                self._plans.pop(next(iter(self._plans)))
        # GradScaler scalar refresh — write the current ``scaler._scale``
        # + ``1/scale`` into the persistent 0-D holders before run.
        self._refresh_scaler_scalars()
        return self._run(plan, args, flags)

    def _refresh_scaler_scalars(self) -> None:
        """Copy ``scaler._scale`` + ``1/scale`` into the persistent feeds.

        ``copy_`` writes through to the holder's existing buffer so
        the cached executable keeps hitting the same input slot.
        Captures ``self._last_applied_scale`` so the loss returned by
        ``_run`` can be unscaled by exactly the value that was applied
        in this step (even if ``scaler.update()`` mutates ``_scale``
        between calls).
        """
        if self._grad_scaler is None or not self._grad_scaler._enabled:
            return
        if self._scale_holder is None or self._inv_scale_holder is None:
            return

        import lucid as _lucid

        scale = float(self._grad_scaler.get_scale())
        self._last_applied_scale = scale
        inv = 1.0 / scale
        dt = self._scale_holder.dtype
        dev = self._scale_holder.device
        self._scale_holder.copy_(_lucid.tensor(scale, dtype=dt, device=dev))
        self._inv_scale_holder.copy_(_lucid.tensor(inv, dtype=dt, device=dev))

    def recompile(self) -> None:
        """Drop every cached executable so the next call retraces from scratch.

        Useful after manual surgery on the model (e.g. resizing a
        parameter buffer) where the captured tensor identities no longer
        match the live ones.  Normal training loops never need to call
        this — freezing / unfreezing parameters and LR schedules are
        picked up on their own.
        """
        self._plans.clear()
        self._reachable = None

    # ── Internals ───────────────────────────────────────────────

    def _key(
        self, members: tuple[int, ...], flags: tuple[_Flags, ...]
    ) -> tuple[object, ...]:
        """Plan key: participants, structural flags, per-parameter classes."""
        return (members, flags, self._copt._partition(members))

    def _members(self) -> tuple[int, ...]:
        """Parameters that step: reach the loss and require grad (eager's rule)."""
        assert self._reachable is not None
        params = self._copt._params
        return tuple(i for i in sorted(self._reachable) if params[i].requires_grad)

    def _build_plan(
        self, args: tuple[Tensor, ...], flags: tuple[_Flags, ...]
    ) -> _FusedPlan:
        """Trace + compile the fused step for the current participants.

        Records forward + loss, decides which parameters take part (the
        loss's ancestors that require grad), prepares their optimizer
        state, records their update with ghost-grad placeholders, then
        calls ``compile_generic_fused_step``, which threads the ghost-
        grad ids through the gradient derivation and returns one
        executable producing loss + parameter updates in one shot.

        Raises :class:`RuntimeError` with a structural reason on any
        failure (empty trace, no participating parameter, builder
        rejection) — fused_step intentionally has no eager fallback
        path because every step would silently lose the speedup.
        """
        from lucid._dispatch import _unwrap
        from lucid._tensor.tensor import Tensor
        from lucid.autograd._grad_mode import no_grad
        from lucid.compile import _tracing
        from lucid.compile._core.bn_runstats import (
            bn_writeback_targets,
            model_has_cumulative_bn,
        )

        copt = self._copt
        params = copt._params
        groups = copt._opt.param_groups

        # AMP scoping (X4.4): the user may wrap the entire
        # ``step(x, t)`` call in ``with autocast()``.  Autocast applies
        # only to forward + loss — the optimizer math runs on F32 master
        # weights, so a neutral ``AutocastGuard(F32)`` is installed
        # around the update emission below.  Without this split the
        # optimizer's reads of F32 master weights would get autocast to
        # F16 → F16 ``new_param`` → dtype mismatch with the F32 buffer.
        _autocast_was_active = _C_engine.amp_is_active()
        _autocast_prev_dtype = _C_engine.amp_active_dtype()

        # GradScaler holders (allocated once, outside any trace).  These
        # tensors are external feeds in the trace; their *values* are
        # refreshed each step via ``copy_``.
        scaler_enabled = self._grad_scaler is not None and self._grad_scaler._enabled
        if scaler_enabled and self._scale_holder is None:
            import lucid as _lucid

            p0 = params[0]
            self._scale_holder = _lucid.zeros(
                (), dtype=_lucid.float32, device=p0.device
            )
            self._inv_scale_holder = _lucid.zeros(
                (), dtype=_lucid.float32, device=p0.device
            )
            self._found_inf_target = _lucid.zeros(
                (), dtype=_lucid.float32, device=p0.device
            )

        members: tuple[int, ...] = ()
        ghost_grads: dict[int, Tensor] = {}
        found_inf_f32: Tensor | None = None
        _tls.active = True
        try:
            with no_grad():
                with _tracing() as tracer:
                    out = self._model(*args[:1])
                    loss = self._loss_fn(out, *args[1:])

                    # GradScaler step 1 — scale loss before backward.
                    # ``loss_id`` (the backward source) points at the
                    # scaled loss; the unscaled loss is divided out on
                    # the return path.
                    if scaler_enabled:
                        assert self._scale_holder is not None
                        loss_for_bwd = loss * self._scale_holder.to(loss.dtype)
                    else:
                        loss_for_bwd = loss

                    # Who takes part: the loss's ancestors that require
                    # grad.  Asking MPSGraph for the gradient of anything
                    # else aborts the process ("Not a predecessor of
                    # primaryTensor"), and eager would leave it alone.
                    if self._reachable is None:
                        loss_tid = tracer.lookup_id(_unwrap(loss_for_bwd))
                        if loss_tid is None:
                            raise RuntimeError("fused_step: loss missing from trace")
                        ancestors = _loss_ancestors(tracer.graph, int(loss_tid))
                        reachable: set[int] = set()
                        for i, p in enumerate(params):
                            tid = tracer.lookup_id(_unwrap(p))
                            if tid is not None and int(tid) in ancestors:
                                reachable.add(i)
                        self._reachable = frozenset(reachable)
                    members = self._members()
                    if not members:
                        raise RuntimeError(
                            "fused_step: no optimizer parameter both requires "
                            "grad and reaches the loss — there is nothing to step"
                        )
                    with _untraced(tracer):
                        for i in members:
                            copt._activate(i, flags, groups)
                        ghost_grads = {i: _zeros_like(params[i]) for i in members}
                        layout = copt._new_layout(members, False, flags)
                        vectors = copt._trace_vectors(layout)

                    if _autocast_was_active:
                        _opt_guard = _C_engine.AutocastGuard(_C_engine.F32)
                        _opt_guard.__enter__()
                    else:
                        _opt_guard = None
                    try:
                        if scaler_enabled:
                            import lucid as _lucid

                            # GradScaler step 2 — unscale grads (in F32)
                            # before the optimizer sees them.  F32 is the
                            # eager-GradScaler convention: F16 inv_scale
                            # at ``2**-16`` is subnormal and Metal
                            # flushes it to zero.
                            unscaled: dict[int, Tensor] = {}
                            for i, g in ghost_grads.items():
                                g_f32 = (
                                    g
                                    if g.dtype == _lucid.float32
                                    else g.to(_lucid.float32)
                                )
                                unscaled[i] = g_f32 * self._inv_scale_holder

                            # GradScaler step 3a — found_inf over the
                            # unscaled gradients: Σ isfinite < Σ numel.
                            finite_counts: list[Tensor] = []
                            expected_total = 0.0
                            for g in unscaled.values():
                                finite_counts.append(
                                    _lucid.isfinite(g).to(_lucid.float32).sum()
                                )
                                expected_total += float(int(_prod_shape(g.shape)))
                            total_finite = finite_counts[0]
                            for nf in finite_counts[1:]:
                                total_finite = total_finite + nf
                            expected_t = _lucid.tensor(
                                expected_total,
                                dtype=_lucid.float32,
                                device=params[0].device,
                            )
                            found_inf_bool = total_finite < expected_t
                            found_inf_f32 = found_inf_bool.to(_lucid.float32)

                            opt_outputs, slots = copt._emit(
                                members, unscaled, layout, vectors, flags, False
                            )

                            # GradScaler step 3b — conditional update:
                            # ``where(found_inf, old, new)`` keeps params +
                            # state on an overflow step (eager skips
                            # ``optimizer.step()``).
                            final_outputs: list[Tensor] = []
                            for slot, new in zip(slots, opt_outputs):
                                old = copt._slot_tensor(slot)
                                if old.dtype != new.dtype:
                                    old = old.to(new.dtype)
                                final_outputs.append(
                                    _lucid.where(found_inf_bool, old, new)
                                )
                            opt_outputs = final_outputs
                        else:
                            opt_outputs, slots = copt._emit(
                                members, ghost_grads, layout, vectors, flags, False
                            )
                    finally:
                        if _opt_guard is not None:
                            # Restore the user's autocast dtype after the
                            # optimizer scope.
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
        bn_counters = bn_counter_targets(self._model, graph, ext)
        if not graph.ops:
            raise RuntimeError("fused_step: empty trace")

        # Resolve the fused-attention workaround capability flag before the
        # emitters run, if this step's graph contains attention (probes once).
        from lucid.compile._core.attention_probe import maybe_probe_for_graph

        maybe_probe_for_graph(graph)

        # ``loss_id`` is BOTH the backward source for autograd AND the
        # tid bound to output[0].  Under GradScaler, both purposes
        # need the SCALED loss (so grads come out scaled).
        loss_id = int(tracer.lookup_id(_unwrap(loss_for_bwd)))

        param_ids: list[int] = []
        for i in members:
            tid = tracer.lookup_id(_unwrap(params[i]))
            if tid is None:
                raise RuntimeError("fused_step: a parameter vanished from the trace")
            param_ids.append(int(tid))

        # Ghost grad ids — same order as param_ids.
        ghost_grad_ids: list[int] = []
        for i in members:
            tid = tracer.lookup_id(_unwrap(ghost_grads[i]))
            if tid is None:
                raise RuntimeError(
                    "fused_step: ghost grad placeholder missing from trace"
                )
            ghost_grad_ids.append(int(tid))

        # Opt-output ids (new params first, then new state buffers).
        output_target_ids: list[int] = []
        for o in opt_outputs:
            tid = tracer.lookup_id(_unwrap(o))
            if tid is None:
                raise RuntimeError(
                    "fused_step: optimizer output tensor missing from trace"
                )
            output_target_ids.append(int(tid))

        # GradScaler step 4 — found_inf as an extra output so Python can
        # read it back after each step and drive ``scaler.update()``.
        if scaler_enabled and found_inf_f32 is not None:
            found_inf_tid = tracer.lookup_id(_unwrap(found_inf_f32))
            if found_inf_tid is None:
                raise RuntimeError("fused_step: found_inf scalar missing from trace")
            output_target_ids.append(int(found_inf_tid))

        # Every training-mode dropout's ``state_out`` id, paired with its
        # ``state_in`` feed — ``compile_generic_fused_step_with_vars``
        # requires each ``write_id`` in ``variable_pairs`` to also appear
        # in ``output_target_ids``.
        dropout_state_target_pairs: list[tuple[int, int]] = []
        for _node in graph.ops:
            if _node.name == "dropout_stateful":
                if len(_node.inputs) >= 2 and len(_node.outputs) >= 2:
                    _sin = int(_node.inputs[1])
                    _sout = int(_node.outputs[1].id)
                    output_target_ids.append(_sout)
                    dropout_state_target_pairs.append((_sin, _sout))

        # 3.5 BatchNorm running-stats (Path A swap-buffer write-back). A
        # cumulative-MA BN (track_running_stats=True + momentum=None) can't be
        # lowered into the graph (its update reads num_batches_tracked as a host
        # scalar) and fused_step has no eager fallback → raise. A
        # track_running_stats=False BN keeps no buffers → compiles unchanged. A
        # fused-momentum BN traces 5-input; pair each running-stat FEED with its
        # EMA OUTPUT and route as PLAIN output-feed targets.
        if model_has_cumulative_bn(self._model):
            raise NotImplementedError(
                "fused_step: BatchNorm with momentum=None (cumulative moving "
                "average) can't have its running-stats update lowered into the "
                "compiled graph, and fused_step has no eager fallback. Use "
                "lucid.compile.make_step (which falls back to eager for it) or a "
                "momentum-based BatchNorm (the default, momentum=0.1)."
            )
        bn_stat_target_pairs: list[tuple[int, int]] = [
            (feed_id, out_id) for feed_id, out_id, _ in bn_writeback_targets(graph, ext)
        ]
        for _bn_feed_id, _bn_new_id in bn_stat_target_pairs:
            output_target_ids.append(_bn_new_id)  # LAST in output_target_ids

        # ``compile_generic_fused_step_with_vars`` (MPSGraph stateful
        # variables) is gated behind ``LUCID_COMPILE_VARS=1`` for the
        # parameter tier, and forced whenever a training-mode dropout is
        # present (its RNG state must advance across dispatches).
        import os as _os

        _params_as_vars = _os.environ.get("LUCID_COMPILE_VARS", "0") in (
            "1",
            "true",
            "True",
        )
        if _params_as_vars or dropout_state_target_pairs:
            # Parameters become variables only on opt-in: promoting large
            # state regressed 10-20 % per step (perf-state-vars-regression).
            # The first len(members) opt outputs are the new parameters,
            # in param_ids order.
            variable_pairs: list[tuple[int, int]] = []
            if _params_as_vars:
                for i, pid in enumerate(param_ids):
                    variable_pairs.append((pid, output_target_ids[i]))
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

        # Per-input resolvers (impl identity → live impl getter).
        impl_to_resolver: dict[int, Callable[[], _C_engine.TensorImpl]] = {}

        def _param_resolver(i: int) -> Callable[[], _C_engine.TensorImpl]:
            return lambda: _unwrap_hot(params[i])

        def _state_resolver(i: int, name: str) -> Callable[[], _C_engine.TensorImpl]:
            return lambda: _unwrap_hot(copt._state[i][name])

        def _vector_resolver(k: int) -> Callable[[], _C_engine.TensorImpl]:
            return lambda: self._step_vectors[k]

        def _arg_resolver(slot: int) -> Callable[[], _C_engine.TensorImpl]:
            return lambda: _unwrap_hot(self._current_args[slot])

        def _pinned(impl: _C_engine.TensorImpl) -> Callable[[], _C_engine.TensorImpl]:
            return lambda: impl

        for i, p in enumerate(params):
            impl_to_resolver[id(_unwrap(p))] = _param_resolver(i)
        for i in members:
            for name, buf in copt._state[i].items():
                impl_to_resolver[id(_unwrap(buf))] = _state_resolver(i, name)
        for k, dt in enumerate(layout.dtypes):
            impl_to_resolver[id(_unwrap(vectors[dt]))] = _vector_resolver(k)
        if self._scale_holder is not None and self._inv_scale_holder is not None:
            impl_to_resolver[id(_unwrap(self._scale_holder))] = _pinned(
                _unwrap(self._scale_holder)
            )
            impl_to_resolver[id(_unwrap(self._inv_scale_holder))] = _pinned(
                _unwrap(self._inv_scale_holder)
            )
        for pos, a in enumerate(args):
            if isinstance(a, Tensor):
                impl_to_resolver[id(_unwrap(a))] = _arg_resolver(pos)

        # 3.5 BatchNorm: the running-stat FEEDS (rm/rv) resolve to the LIVE
        # module buffer — the read half of the per-step read-modify-write.
        for _bn_feed_id, _bn_new_id in bn_stat_target_pairs:
            _bn_impl = ext.get(_bn_feed_id)
            if _bn_impl is not None and id(_bn_impl) not in impl_to_resolver:
                impl_to_resolver[id(_bn_impl)] = _pinned(_bn_impl)

        # Anything else (ad-hoc constants the forward materialises, e.g.
        # ``Conv2d(bias=False)``'s zero bias) is pinned to its trace impl.
        resolvers: list[Callable[[], _C_engine.TensorImpl]] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                raise RuntimeError(f"fused_step: input id {tid} not in external_feeds")
            r = impl_to_resolver.get(id(impl))
            resolvers.append(r if r is not None else _pinned(impl))

        # Output targets in the SAME order as ``output_target_ids``:
        # [opt outputs, found_inf, dropout state, BN running-stats].
        output_targets: list[Tensor] = [copt._slot_tensor(s) for s in slots]
        if scaler_enabled and self._found_inf_target is not None:
            output_targets.append(self._found_inf_target)
        for _state_in_tid, _state_out_tid in dropout_state_target_pairs:
            _impl = ext.get(_state_in_tid)
            if _impl is None:
                raise RuntimeError(
                    "fused_step: dropout state feed id "
                    f"{_state_in_tid} not in external_feeds"
                )
            output_targets.append(Tensor(_impl, requires_grad=False))
        for _bn_feed_id, _bn_new_id in bn_stat_target_pairs:
            _bn_impl = ext.get(_bn_feed_id)
            if _bn_impl is None:
                raise RuntimeError(
                    f"fused_step: BN running-stat feed id {_bn_feed_id} "
                    "not in external_feeds"
                )
            output_targets.append(Tensor(_bn_impl, requires_grad=False))

        if len(output_targets) != len(output_target_ids):
            raise RuntimeError(
                "fused_step: output_targets count "
                f"({len(output_targets)}) doesn't match "
                f"output_target_ids count ({len(output_target_ids)})"
            )

        # Per-slot shape guard: any misordering (e.g. a found_inf/dropout/BN
        # swap) would silently write the wrong buffer.
        _id_to_shape: dict[int, tuple[int, ...]] = {}
        for _n in graph.ops:
            for _m in _n.outputs:
                _id_to_shape[int(_m.id)] = tuple(_m.shape)
        for _slot, (_oid, _tgt) in enumerate(zip(output_target_ids, output_targets)):
            _exp = _id_to_shape.get(int(_oid))
            if _exp is not None and tuple(_tgt.shape) != _exp:
                raise RuntimeError(
                    f"fused_step: output target slot {_slot} (id {_oid}) shape "
                    f"{tuple(_tgt.shape)} != trace meta {_exp} — "
                    "output_target_ids / output_targets ordering drift"
                )

        # Phase 1.10: the targets are stable across calls (a state
        # buffer keeps its identity for life), so unwrap them once.
        return _FusedPlan(
            exe=exe,
            members=members,
            layout=layout,
            resolvers=resolvers,
            target_impls=tuple(_unwrap(t) for t in output_targets),
            bn_counters=bn_counters,
            loss_shape=tuple(loss_for_bwd.shape),
            loss_dtype=loss_for_bwd.dtype,
            loss_device=loss_for_bwd.device,
        )

    def _run(
        self, plan: _FusedPlan, args: tuple[Tensor, ...], flags: tuple[_Flags, ...]
    ) -> Tensor:
        """Bind feeds + targets and invoke the cached executable in-place.

        Packs this step's scalars (hyper-parameters read from the live
        ``param_groups`` and per-parameter step-dependent values), runs
        the executable, and advances the step counts of the parameters
        that stepped — unless the GradScaler found an overflow, in which
        case nothing stepped.

        Parameters
        ----------
        plan : _FusedPlan
            The executable for this step's participants.
        args : tuple of Tensor
            Same per-step arguments forwarded from :meth:`__call__`.
        flags : tuple
            Every group's structural flags (the plan was built for them).

        Returns
        -------
        Tensor
            Scalar loss for this step.
        """
        copt = self._copt
        self._current_args = args
        self._step_vectors = copt._scalar_vectors(plan.layout, flags, None)
        feeds = [r() for r in plan.resolvers]

        # Fresh loss tensor (single-element output that flows back to
        # Python; not in-place since callers want a clean handle).
        loss_tensor = _lucid_hot.zeros(
            *plan.loss_shape if plan.loss_shape else (),
            dtype=plan.loss_dtype,
            device=plan.loss_device,
        )
        output_targets = [_unwrap_hot(loss_tensor), *plan.target_impls]
        _C_engine.compile.run_executable_inplace(plan.exe, feeds, output_targets)
        advance_bn_counters(plan.bn_counters)

        skipped = False
        if (
            self._grad_scaler is not None
            and self._grad_scaler._enabled
            and self._found_inf_target is not None
        ):
            # ``item()`` forces a CPU sync — unavoidable since the
            # scaler's update logic needs the bool.
            skipped = bool(float(self._found_inf_target.item()))
            self._grad_scaler._found_inf = skipped
            self._grad_scaler.update()
            # Unscale the loss — the user expects to see the original
            # unscaled loss value, matching ``scaler.scale(loss)``'s
            # eager contract.
            loss_tensor = loss_tensor / self._last_applied_scale
        if not skipped:
            copt._commit(plan.members)
        return loss_tensor
