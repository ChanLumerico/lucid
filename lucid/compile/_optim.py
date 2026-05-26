"""
lucid.compile._optim — compiled optimizer wrappers (in-place output path).

Wraps an eager :class:`lucid.optim.Optimizer` so ``step()`` becomes a
single MPSGraph executable that fuses the per-parameter update math AND
writes its outputs directly back into the parameter / state buffers via
``run_executable_inplace``.

Why in-place
------------
A first revision routed the update through ``lucid.compile(update_fn)``
and ``param.copy_(new_param)`` per parameter.  That version compiled
correctly (bit-exact parity) but ran ~50 % SLOWER than the eager C++
optim, because the per-output ``copy_`` calls each forced an
``mlx::copy`` sync.  For a 22-parameter Adam this added ~2.3 ms / step
on top of the ~2.3 ms MPSGraph dispatch — entirely defeating the
fusion benefit.

The current implementation skips the ``copy_`` round-trip entirely:

1. **One-time trace + compile** — runs the update function under a
   :class:`Tracer`, records params / state / grads / bias-corrections
   as inputs and the new params / state as outputs, then calls
   ``compile_or_cached`` to mint a :class:`MPSGraphExecutable`.
2. **Per-step run** — collects fresh grads + bias-correction scalars,
   calls ``run_executable_inplace`` with the parameter and state
   tensors as the output targets.  MPSGraph writes directly into
   their existing MTLBuffers; no fresh allocation, no per-output
   ``copy_``.

End result: the per-step cost drops from ~4.6 ms to roughly the
MPSGraph dispatch time alone, beating eager once the parameter
count is large enough to amortise the per-call overhead.
"""

import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.optim.optimizer import Optimizer

__all__ = ["compile_optimizer"]


def compile_optimizer(opt: Optimizer) -> object:
    r"""Wrap ``opt`` so ``opt.step()`` runs as a single MPSGraph executable.

    Dispatches on the concrete optimizer subclass and returns the
    matching :class:`_CompiledStepBase` subclass instance.  The
    returned object exposes the same eager-optimizer API surface
    (``step`` / ``zero_grad`` / ``param_groups`` / ``state_dict`` /
    ``load_state_dict``) so existing training loops slot it in
    without code changes — the only difference is that ``step()``
    runs as one compiled GPU kernel instead of N sequential per-
    parameter element-wise updates.

    Supported optimizers (8):
        * :class:`~lucid.optim.SGD` — classical, with optional
          momentum / Nesterov / weight-decay branches.
        * :class:`~lucid.optim.Adam`, :class:`~lucid.optim.AdamW` —
          bias-corrected first + second moments; coupled vs
          decoupled weight decay respectively.  AMSGrad variant of
          Adam is rejected at construct time (a planned follow-up).
        * :class:`~lucid.optim.RMSprop` — exponentially-smoothed
          squared gradient.  ``centered=True`` is rejected loudly
          here even though the eager backend silently drops it.
        * :class:`~lucid.optim.Adagrad` — per-parameter cumulative
          squared gradient; ``lr_decay`` is fed as a per-step
          scalar so the trace stays signature-stable.
        * :class:`~lucid.optim.Adadelta` — running RMS-ratio
          adaptive step (no manual LR).
        * :class:`~lucid.optim.Adamax` — Adam variant with L∞-norm
          second-moment estimate.
        * :class:`~lucid.optim.NAdam` — Adam with Nesterov lookahead
          + momentum-decay schedule (closed-form ``μ_t`` series).

    Structurally unsupported (raise :class:`NotImplementedError`
    with the reason at construct time):
        * :class:`~lucid.optim.LBFGS` — line search depends on
          data-dependent iteration count (no static-shape MPSGraph
          equivalent).
        * :class:`~lucid.optim.SparseAdam` — index-driven update
          requires runtime ``nonzero`` / ``scatter``.
        * :class:`~lucid.optim.Rprop` — sign-based per-element
          conditional branching.
        * :class:`~lucid.optim.RAdam` — ``ρ_t > 4`` branch can't be a
          static MPSGraph op.
        * :class:`~lucid.optim.ASGD` — averaging coefficient
          depends on iteration count past a warmup threshold.

    Multi-param-group optimizers are not yet supported (a future
    pass will emit one update kernel per group); single-group
    optimizers cover essentially every practical training recipe.

    Parameters
    ----------
    opt : Optimizer
        Concrete optimizer instance whose ``step()`` is being
        lifted.  Held by reference; LR-scheduler callbacks and
        other state mutations on ``opt`` continue to take effect
        because the compiled subclass reads them at scalar-refresh
        time, not at construct time.

    Returns
    -------
    _CompiledStepBase
        Drop-in optimizer replacement.  The underlying class is one
        of the ``_Compiled<Name>`` subclasses listed above.

    Raises
    ------
    NotImplementedError
        With a structural reason when ``opt`` is one of the
        unsupported families above (or AMSGrad / multi-group / etc.).
    TypeError
        When ``opt`` is not a Lucid :class:`Optimizer` subclass at
        all — the message lists the supported set.

    Examples
    --------
    Drop-in replacement::

        opt = lucid.optim.Adam(model.parameters(), lr=1e-3)
        copt = compile_optimizer(opt)

        for batch, target in loader:
            copt.zero_grad()
            loss = F.cross_entropy(model(batch), target)
            loss.backward()
            copt.step()             # one MPSGraph executable

    Inspecting the rejection reason::

        try:
            copt = compile_optimizer(lucid.optim.LBFGS(params))
        except NotImplementedError as e:
            print(e)
            # "compile_optimizer: LBFGS is not supported.  LBFGS performs
            #  a line search inside step() whose iteration count depends
            #  on tensor values; ..."

    See Also
    --------
    :func:`fused_step` : single-executable forward + loss + backward
        + update via the ghost-grad mechanism.  Use this when the
        whole step is the unit of work and the model's forward
        compiles cleanly.
    :func:`make_step` : autograd-graph-aware fwd+bwd compile that
        still lets ``loss.backward()`` run a regular eager
        optimizer step.
    """
    from lucid.optim.sgd import SGD
    from lucid.optim.adam import Adam, AdamW
    from lucid.optim.others import (
        RMSprop,
        Adagrad,
        Adadelta,
        Adamax,
        NAdam,
        ASGD,
        RAdam,
        Rprop,
        SparseAdam,
    )
    from lucid.optim.lbfgs import LBFGS

    # Multi-group dispatch — wrap one _Compiled* per group, each
    # backed by a synthetic single-group clone of the parent optimizer
    # (so the per-group compile sees its own hyperparams).  The wrapper
    # delegates lifecycle (zero_grad / param_groups / state_dict /
    # load_state_dict) back to the parent.  Useful for backbone+head
    # training recipes with distinct LRs per group.
    if len(opt.param_groups) > 1:
        return _MultiGroupCompiledOptimizer(opt)

    # Supported — elementwise update math, single-dispatch friendly.
    if isinstance(opt, SGD):
        return _CompiledSGD(opt)
    if isinstance(opt, AdamW):
        return _CompiledAdamW(opt)
    if isinstance(opt, Adam):
        return _CompiledAdam(opt)
    if isinstance(opt, RMSprop):
        return _CompiledRMSprop(opt)
    if isinstance(opt, Adagrad):
        return _CompiledAdagrad(opt)
    if isinstance(opt, Adadelta):
        return _CompiledAdadelta(opt)
    if isinstance(opt, Adamax):
        return _CompiledAdamax(opt)
    if isinstance(opt, NAdam):
        return _CompiledNAdam(opt)

    # Structurally unsupported — explain *why* per family so the
    # caller knows whether to wait for a future variant or pick
    # a different optimizer.
    if isinstance(opt, LBFGS):
        raise NotImplementedError(
            "compile_optimizer: LBFGS is not supported.  LBFGS performs "
            "a line search inside ``step()`` whose iteration count "
            "depends on tensor values; that's incompatible with a "
            "single MPSGraph executable (no data-dependent loops in "
            "the IR).  Use eager LBFGS."
        )
    if isinstance(opt, SparseAdam):
        raise NotImplementedError(
            "compile_optimizer: SparseAdam is not supported.  The "
            "update is defined only over non-zero gradient indices and "
            "needs a runtime nonzero / scatter dispatch that the "
            "compile path can't express as a dense MPSGraph op."
        )
    if isinstance(opt, Rprop):
        raise NotImplementedError(
            "compile_optimizer: Rprop is not supported.  Each element's "
            "step size is updated via a sign-based conditional branch "
            "(grow / shrink / hold), which can be expressed as MPSGraph "
            "``select`` chains but introduces a control-flow shape that "
            "the current emit pipeline isn't designed for.  Possible "
            "future work."
        )
    if isinstance(opt, RAdam):
        raise NotImplementedError(
            "compile_optimizer: RAdam is not supported.  The rectified "
            "Adam update branches at ``ρ_t > 4`` between a SGD-like "
            "fallback and the bias-corrected step — that data-dependent "
            "switch can't live inside a single MPSGraph executable.  "
            "Possible future work via a precomputed schedule."
        )
    if isinstance(opt, ASGD):
        raise NotImplementedError(
            "compile_optimizer: ASGD is not supported.  The averaged "
            "parameter trajectory uses a per-step scalar coefficient "
            "``μ_t`` that depends on the iteration count past a "
            "warmup threshold.  Possible future work via per-step "
            "scalar feeds."
        )

    raise TypeError(
        f"compile_optimizer: unsupported optimizer class "
        f"{type(opt).__name__!r}.  Supported: SGD, Adam, AdamW, "
        f"RMSprop, Adagrad, Adadelta, Adamax, NAdam."
    )


# ── Common helpers ──────────────────────────────────────────────────


def _zeros_like(t: Tensor) -> Tensor:
    """Allocate a same-shape zero-filled tensor on the same device/dtype."""
    import lucid as _lucid

    return _lucid.zeros(*t.shape, dtype=t.dtype, device=t.device)


def _flatten_params(opt: Optimizer) -> list[Tensor]:
    """Flatten every Parameter across every ``param_group`` into one list.

    The compile path treats parameters as a flat sequence — the
    ``_trace_update`` / ``_outputs_to_targets`` hooks index into
    ``self._params`` by integer slot.  Multi-group optimizers are
    rejected upstream in :meth:`_CompiledStepBase.__init__`; this
    helper is therefore single-group in practice and just flattens
    one ``group["params"]`` list.

    Returns
    -------
    list[Tensor]
        Parameters in the same order they appear in
        ``opt.param_groups[0]["params"]``.
    """
    out: list[Tensor] = []
    for group in opt.param_groups:
        for p in group["params"]:
            out.append(p)
    return out


def _zero_scalar(dt: object, dev: object) -> Tensor:
    """Allocate a 0-D zero tensor.  Used as a stable host for the bias
    correction scalars that the Adam-family compiled executables read
    on every step — we update the *value* of these tensors each step
    via ``copy_``, keeping their TensorImpl identity stable so the
    cache hits.
    """
    import lucid as _lucid

    return _lucid.zeros((), dtype=dt, device=dev)


class _CompiledStepBase:
    r"""Abstract base for the compiled-optimizer wrappers.

    Each concrete subclass (one per supported optimizer family)
    plugs into a fixed five-hook lifecycle and inherits the build /
    cache / run loop from this class.  The hooks together describe
    *what state buffers the optimizer maintains*, *which extra
    scalars it needs per step*, and *how the update math composes*
    inside a tracer-recorded :class:`TraceGraph`.

    Lifecycle
    ---------
    The first :meth:`step` call triggers :meth:`_build_executable`,
    which assembles the input/output plan, runs the trace, calls
    ``compile_or_cached``, and stores the executable.  Every
    subsequent :meth:`step` reuses that executable via
    ``run_executable_inplace`` — the param + state buffers are
    written *in place* by the GPU so no per-output ``copy_`` syncs
    fire between MPSGraph and Python.

    Subclass hooks (override these)
    -------------------------------
    :meth:`_register_state_in_inputs(register)`
        Append every optimizer-owned state buffer (velocity buffers
        for SGD, ``m``/``v`` for Adam, ``square_avg`` for RMSprop,
        ...) into the input plan via the supplied ``register(kind,
        index, tensor)`` callback.  The ``(kind, index)`` pair must
        match an entry in ``_buffer_table`` so
        :meth:`_resolve_input` can find the live tensor at run time.
    :meth:`_register_scalars(register)` *(optional)*
        Materialise stable 0-D placeholders for per-step scalars
        whose *value* varies across steps but whose *identity*
        must stay constant for the cache to hit (e.g. Adam's bias-
        correction factors, NAdam's three coefficient scalars).
        Returns a ``dict[name, Tensor]`` consumed by
        :meth:`_trace_update`.  Default: no scalars.
    :meth:`_refresh_scalars()` *(optional)*
        Copy fresh values into the placeholders allocated by
        :meth:`_register_scalars` before each :meth:`step`.  Default:
        no-op for optimizers without per-step scalars.
    :meth:`_trace_update(all_inputs, grads, scalars)`
        Emit the actual update math under the active :class:`Tracer`.
        Returns the ordered list of result tensors
        ``[new_params..., new_state...]``.  This ordering pins the
        executable's output order, so :meth:`_outputs_to_targets`
        must mirror it exactly.
    :meth:`_outputs_to_targets(outputs)`
        Map each executable-output slot to the parameter / state-
        buffer Tensor whose storage receives the in-place write.
        Same ordering as :meth:`_trace_update`'s return.

    Attributes
    ----------
    _opt : Optimizer
        The wrapped eager optimizer.  Held by reference; LR
        schedulers / state mutations flow through naturally.
    _params : list[Tensor]
        Flat ordered list of every parameter across every
        ``param_group`` (rejected at construct time when more than
        one group is present).
    _exe : object or None
        The cached ``PyCompiledExecutable``, lazily allocated
        on the first :meth:`step`.  Set to ``None`` by
        :meth:`load_state_dict` to force a retrace.
    _input_plan : list[tuple]
        Ordered ``(kind, index)`` pairs naming each placeholder the
        executable reads in ``exe.input_ids`` order.  Resolved
        to live tensors via :meth:`_resolve_input`.
    _output_targets : list[Tensor]
        Tensors that receive the executable's in-place writes; same
        order as :meth:`_trace_update`'s return.
    _buffer_table : dict[(str, int), Callable[[], Tensor]]
        Subclass-extensible registry mapping every state-buffer
        ``(kind, index)`` to a zero-arg getter that returns the
        *live* tensor.  Using a callable (rather than storing the
        tensor directly) lets :meth:`load_state_dict` swap the
        underlying list without touching the table.
    _scalar_slots : dict[str, Tensor]
        Name → stable 0-D placeholder for per-step scalars.  Refresh
        via :meth:`_refresh_scalars`.

    Notes
    -----
    *In-place output writes.*  A first revision routed the update
    through ``lucid.compile(update_fn)`` and explicit
    ``param.copy_(new_param)`` per parameter.  That ran ~50 %
    *slower* than eager because every ``copy_`` forced an
    ``mlx::copy`` sync — for a 22-parameter Adam this added ~2.3 ms
    on top of the ~2.3 ms MPSGraph dispatch.  The current path uses
    ``run_executable_inplace`` so MPSGraph writes directly into
    the parameter MTLBuffers; per-step cost drops to the dispatch
    time alone.

    See Also
    --------
    :func:`compile_optimizer` : the user-facing entry that
        dispatches on optimizer type and returns the right
        subclass instance.
    """

    def __init__(self, opt: Optimizer) -> None:
        """Set up the shared compile-step plumbing on top of ``opt``.

        Parameters
        ----------
        opt : Optimizer
            The eager optimizer instance whose math is being lifted
            into a compiled executable.  Held by reference so
            scheduler callbacks (LR updates, etc.) still flow
            through naturally.

        Raises
        ------
        ValueError
            If ``opt`` has no parameters.
        NotImplementedError
            If ``opt`` has more than one ``param_group`` — multi-
            group support is deferred (the trace would need to emit
            one update kernel per group, which the current generic
            entry point doesn't model).
        """
        self._opt = opt
        self._params = _flatten_params(opt)
        if not self._params:
            raise ValueError("compile_optimizer: optimizer has no trainable parameters")
        # Multi-group dispatch is handled one level up in
        # ``compile_optimizer()`` via ``_MultiGroupCompiledOptimizer``.
        # By the time we reach a concrete ``_Compiled*`` subclass, the
        # optimizer is guaranteed to have exactly one ``param_group``.
        if len(opt.param_groups) > 1:
            raise NotImplementedError(
                "_CompiledStepBase: expected single-group optimizer "
                "(multi-group should be wrapped by _MultiGroupCompiledOptimizer)."
            )
        # Filled by the concrete subclass before _build_executable().
        self._exe: object | None = None
        # Per-input-slot category descriptor, populated at compile time.
        # Each entry is a tuple ``(kind, index)`` matched against
        # ``_buffer_table``.  Built-in kinds: "param", "grad",
        # "scalar".  Concrete subclasses extend the table with their
        # own state-buffer kinds (e.g. "m"/"v" for Adam,
        # "square_avg"/"acc_delta" for Adadelta).
        self._input_plan: list[tuple] = []
        # Output targets in trace return order; built once after compile.
        self._output_targets: list[Tensor] = []
        # Per-step scratch (re-bound in step()).
        self._current_grads: list[Tensor] = []
        self._scalar_slots: dict[str, Tensor] = {}
        # Subclass-extensible buffer-table for ``_resolve_input``.  Each
        # entry maps ``(kind, index)`` to a zero-arg callable returning
        # the *live* Tensor for that slot — using a callable lets the
        # subclass swap the underlying tensor list (e.g. on
        # ``load_state_dict``) without touching the table.
        self._buffer_table: dict[tuple[str, int], Callable[[], Tensor]] = {}

    # ── Drop-in API surface ──────────────────────────────────────

    @property
    def param_groups(self) -> list[dict[str, object]]:
        """Delegate to the wrapped optimizer's parameter groups."""
        return self._opt.param_groups

    @property
    def state(self) -> dict[int, dict[str, object]]:
        """Delegate to the wrapped optimizer's per-parameter state dict."""
        return self._opt.state

    @property
    def defaults(self) -> dict[str, object]:
        """Delegate to the wrapped optimizer's hyperparameter defaults."""
        return self._opt.defaults

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Forward to the wrapped optimizer's ``zero_grad`` — drop-in API."""
        self._opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, object]:
        """Forward to the wrapped optimizer's ``state_dict`` for checkpointing."""
        return self._opt.state_dict()

    def load_state_dict(self, sd: dict[str, object]) -> None:
        """Restore optimizer state and **drop the compiled executable**.

        Loading a checkpoint may replace state buffer tensors with
        fresh objects, changing their TensorImpl identity.  The
        cached executable's input plan keys feeds by identity, so
        the safe default is to retrace on the next step.

        Parameters
        ----------
        sd : dict
            ``optimizer.state_dict()`` payload — passed straight to
            the wrapped optimizer's own ``load_state_dict``.

        See Also
        --------
        lucid.optim.Optimizer.load_state_dict : the underlying call.
        """
        self._opt.load_state_dict(sd)
        # Recompile next step — state buffer identity may have moved.
        self._exe = None

    # ── Internals ────────────────────────────────────────────────

    def _resolve_input(self, plan_entry: tuple) -> Tensor:
        """Map one ``(kind, index)`` plan slot to its live tensor.

        Built-in ``kind`` values: ``"param"`` (parameter slot),
        ``"grad"`` (per-step gradient, re-bound each ``step()``),
        ``"scalar"`` (the bias-correction holders).  Anything else is
        an optimizer-subclass-defined state buffer registered through
        ``_buffer_table`` (e.g. ``"m"``/``"v"`` for Adam,
        ``"square_avg"`` for RMSprop).
        """
        kind = plan_entry[0]
        if kind == "param":
            return self._params[plan_entry[1]]
        if kind == "grad":
            return self._current_grads[plan_entry[1]]
        if kind == "scalar":
            # plan_entry[1] is the scalar name (or integer index for
            # older Adam path); look up via subclass-populated dict.
            key = plan_entry[1]
            if isinstance(key, int):
                key = list(self._scalar_slots.keys())[key]
            return self._scalar_slots[key]
        # Anything else is a subclass-defined state buffer kind.  Look
        # it up in the buffer table; callable returns the live tensor.
        getter = self._buffer_table.get((kind, plan_entry[1]))
        if getter is None:
            raise RuntimeError(
                f"_resolve_input: unknown plan slot ({kind!r}, "
                f"{plan_entry[1]!r}) — subclass forgot to register "
                f"this state buffer in ``_buffer_table``"
            )
        return getter()

    def _build_executable(self) -> None:
        """Trace + compile the update once.  Populates ``self._exe``,
        ``self._input_plan``, and ``self._output_targets``.
        """
        from lucid._dispatch import _unwrap
        from lucid._tensor.tensor import Tensor
        from lucid.autograd._grad_mode import no_grad
        from lucid.compile import _tracing

        # Build a trace-time inputs registry.  Each entry: (kind, idx,
        # tensor) — tracked by *object identity* of the underlying
        # TensorImpl since that's what the tracer's impl_to_id_ keys on.
        inputs_registry: list[tuple[str, int, Tensor]] = []

        def register(kind: str, idx: int, t: Tensor) -> None:
            inputs_registry.append((kind, idx, t))

        # Allocate one-shot grad placeholders + scalar tensors so they
        # have stable TensorImpl identity across the trace and all
        # future step() calls.  At step time we use ``copy_`` (or a
        # direct storage swap) to refresh their values; the impl
        # pointer never changes, so the trace's external_feeds always
        # resolves to the same slot.
        for i, p in enumerate(self._params):
            register("param", i, p)
        self._register_state_in_inputs(register)
        grad_placeholders = [_zeros_like(p) for p in self._params]
        for i, g in enumerate(grad_placeholders):
            register("grad", i, g)
        scalar_holders = self._register_scalars(register)

        # Build the trace.
        with no_grad():
            with _tracing() as tracer:
                outputs = self._trace_update(
                    [t for (_kind, _idx, t) in inputs_registry],
                    grad_placeholders,
                    scalar_holders,
                )

        graph = tracer.graph
        if not graph.ops:
            raise RuntimeError(
                "compile_optimizer: empty trace — update function emitted "
                "no ops (unexpected)."
            )
        ext = tracer.external_feeds

        # Map each known tensor to its trace id.
        impl_to_kind: dict[int, tuple] = {}
        for kind, idx, t in inputs_registry:
            impl_to_kind[id(_unwrap(t))] = (kind, idx)

        # Build the input_plan in exe.input_ids order — first compile,
        # then look up by impl identity.
        explicit_outputs: list[int] = []
        for out_t in outputs:
            tid = tracer.lookup_id(_unwrap(out_t))
            if tid is None:
                raise RuntimeError(
                    "compile_optimizer: trace output tensor has no id — "
                    "the update function produced a tensor that wasn't "
                    "captured by the tracer (bug)."
                )
            explicit_outputs.append(int(tid))

        try:
            exe = _C_engine.compile.compile_or_cached(
                graph, ext, False, [], explicit_outputs
            )
        except RuntimeError as e:
            raise RuntimeError(f"compile_optimizer: compile_or_cached failed: {e}")
        if exe is None:
            raise RuntimeError(
                "compile_optimizer: compile_or_cached returned None — "
                "an op in the update graph has no emitter."
            )

        # Now resolve exe.input_ids → input_plan entries.
        input_plan: list[tuple] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                raise RuntimeError(
                    f"compile_optimizer: input id {tid} not in external_feeds"
                )
            kind_pair = impl_to_kind.get(id(impl))
            if kind_pair is None:
                raise RuntimeError(
                    f"compile_optimizer: input id {tid} not in our "
                    "registry — trace captured an unexpected tensor"
                )
            input_plan.append(kind_pair)

        # Resolve exe.output_ids → output_targets list (must match
        # ``outputs`` ordering since we passed explicit_outputs).
        targets = self._outputs_to_targets(outputs)
        if len(targets) != len(exe.output_ids):
            raise RuntimeError(
                f"compile_optimizer: target count {len(targets)} != "
                f"executable output count {len(exe.output_ids)}"
            )

        # Stash everything for step().
        self._exe = exe
        self._input_plan = input_plan
        self._output_targets = targets
        # Make grad / scalar placeholders accessible for value refresh.
        self._grad_placeholders = grad_placeholders

    # Subclass hooks — override these.

    def _register_state_in_inputs(
        self, register: Callable[[str, int, Tensor], None]
    ) -> None:
        """Register state buffers (momenta / m / v) as trace inputs."""
        raise NotImplementedError

    def _register_scalars(
        self, register: Callable[[str, int, Tensor], None]
    ) -> dict[str, Tensor]:
        """Register per-step scalar tensors (e.g. bias-correction) as
        trace inputs.  Returns a dict mapping scalar name to its
        placeholder tensor, so the trace function can reference them
        by name and ``step()`` can refresh their values.
        """
        return {}

    def _trace_update(
        self,
        all_inputs: Sequence[Tensor],
        grads: Sequence[Tensor],
        scalars: dict[str, Tensor],
    ) -> list[Tensor]:
        """Run the update math under the active tracer.  Returns the
        ordered list of result tensors (new_params + new_state).  The
        ordering here determines the executable's output ordering, so
        it must match ``_outputs_to_targets``.
        """
        raise NotImplementedError

    def _outputs_to_targets(self, outputs: list[Tensor]) -> list[Tensor]:
        """Map each trace-output position to the param / state buffer
        whose storage receives the executable's write.  Same ordering
        as :meth:`_trace_update`'s return.
        """
        raise NotImplementedError

    def _refresh_grads(self) -> None:
        """Copy current ``param.grad`` values into the trace-time grad
        placeholders.  Done via ``copy_`` so the placeholders keep
        their TensorImpl identity (and therefore their executable
        input slot) across steps.
        """
        import lucid as _lucid

        for i, p in enumerate(self._params):
            if p.grad is None:
                # No gradient — zero-fill the placeholder.
                self._grad_placeholders[i].copy_(_zeros_like(p))
            else:
                self._grad_placeholders[i].copy_(p.grad)
        self._current_grads = self._grad_placeholders

    def _refresh_scalars(self) -> None:
        """Subclass hook to refresh bias-correction or other per-step
        scalar tensors.  Default: nothing to refresh."""
        return None

    # ── Public step() ────────────────────────────────────────────

    def step(self, closure: Callable[..., Tensor] | None = None) -> Tensor | None:
        """Run one compiled update step; returns the (optional) closure loss.

        Parameters
        ----------
        closure : callable, optional
            Match the eager-optim signature — invoked once before the
            update and its return value is bubbled back to the
            caller.  ``None`` (the common path) means there's no
            closure and the return is ``None``.

        Returns
        -------
        Tensor or None
            Whatever ``closure()`` returned, or ``None`` when no
            closure was supplied.  Parameter and optimizer-state
            buffers are mutated in-place by the cached executable
            before this returns.
        """
        loss: Tensor | None = closure() if closure is not None else None
        if self._exe is None:
            self._build_executable()
        self._refresh_grads()
        self._refresh_scalars()

        from lucid._dispatch import _unwrap

        # Build the input feed list in exe.input_ids order.
        feeds = [_unwrap(self._resolve_input(p)) for p in self._input_plan]
        targets = [_unwrap(t) for t in self._output_targets]

        _C_engine.compile.run_executable_inplace(self._exe, feeds, targets)
        return loss


# ── SGD ─────────────────────────────────────────────────────────────


class _CompiledSGD(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.SGD` (with optional momentum / nesterov / weight_decay).

    Implements the classical Polyak heavy-ball + Nesterov-lookahead
    update lifted into a single MPSGraph executable.  When ``momentum
    == 0`` the velocity buffer is never allocated, so plain SGD costs
    no extra memory over the parameters themselves.

    Update rule
    -----------
    Per parameter :math:`\theta` with gradient :math:`g_t`, momentum
    coefficient :math:`\mu`, dampening :math:`\tau`, weight decay
    :math:`\lambda`, learning rate :math:`\eta`:

    .. math::

        g_t       &\leftarrow g_t + \lambda \theta_t \\
        v_{t+1}   &= \mu v_t + (1 - \tau) g_t \\
        \tilde g  &= \begin{cases}
                        g_t + \mu v_{t+1} & \text{Nesterov}\\
                        v_{t+1}           & \text{otherwise}
                     \end{cases} \\
        \theta_{t+1} &= \theta_t - \eta \tilde g

    With ``momentum == 0`` the velocity branch is skipped entirely
    and the update reduces to :math:`\theta_{t+1} = \theta_t - \eta g_t`.

    Notes
    -----
    Nesterov is implemented faithfully here, unlike the eager SGD
    which historically silently dropped the flag.  See
    ``test_compile_optimizer_nesterov_correctness`` for the
    hand-rolled formula verification.

    See Also
    --------
    :class:`lucid.optim.SGD` : eager counterpart.
    """

    def __init__(self, opt: Optimizer) -> None:
        """Capture SGD hyperparameters + allocate the velocity buffers.

        Hyperparameter extraction is delegated to
        :func:`OptimizerSpec.from_optim` (see :file:`_optim_spec.py`)
        which provides a single source of truth across this Python
        path and the C++ ``OptimizerSpec`` struct.  The velocity
        buffer (one tensor per parameter) is allocated only when
        ``momentum != 0`` — plain SGD therefore costs zero extra
        memory.  Nesterov is implemented faithfully (the eager SGD
        silently drops it; see ``test_optimizer.py``'s
        nesterov-correctness test).

        Raises
        ------
        TypeError
            If ``opt`` is not an :class:`~lucid.optim.sgd.SGD`
            instance.
        """
        from lucid.optim.sgd import SGD
        from lucid.compile._optim_spec import OptimizerSpec

        if not isinstance(opt, SGD):
            raise TypeError(f"_CompiledSGD: expected SGD, got {type(opt).__name__}")
        super().__init__(opt)
        spec = OptimizerSpec.from_optim(opt)
        self._spec = spec
        self._lr = spec.lr
        self._momentum = spec.momentum
        self._dampening = spec.dampening
        self._weight_decay = spec.weight_decay
        self._nesterov = spec.nesterov
        if self._momentum != 0.0:
            self._momenta = [_zeros_like(p) for p in self._params]
        else:
            self._momenta = []
        for i in range(len(self._momenta)):
            self._buffer_table[("mom", i)] = lambda _i=i: self._momenta[_i]

    def _register_state_in_inputs(self, register):
        """Register the velocity buffers (``"mom"`` kind) as trace inputs."""
        for i, m in enumerate(self._momenta):
            register("mom", i, m)

    def _trace_update(self, all_inputs, grads, scalars):
        """Emit the SGD update math under the active tracer.

        Implements the classical Polyak-momentum / Nesterov-lookahead
        form (see :class:`lucid.optim.SGD` docstring for the closed
        form).  Returns ``new_params + new_momenta`` in that order;
        :meth:`_outputs_to_targets` must mirror this order.
        """
        params = self._params
        momenta = self._momenta
        lr = self._lr
        mu = self._momentum
        dampening = self._dampening
        wd = self._weight_decay
        nesterov = self._nesterov
        new_params: list[Tensor] = []
        new_momenta: list[Tensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd != 0.0:
                g = g + wd * p
            if mu != 0.0:
                m = momenta[i]
                new_m = mu * m + (1.0 - dampening) * g
                eff_g = (g + mu * new_m) if nesterov else new_m
                new_momenta.append(new_m)
            else:
                eff_g = g
            new_p = p - lr * eff_g
            new_params.append(new_p)
        return new_params + new_momenta

    def _outputs_to_targets(self, outputs):
        """Map executable outputs to ``params`` (first N) then ``momenta`` (next N)."""
        # First N outputs → self._params, next N → self._momenta.
        n = len(self._params)
        return list(self._params) + list(self._momenta)


# ── Adam ────────────────────────────────────────────────────────────


class _CompiledAdam(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Adam` (no AMSGrad).

    Lifts the bias-corrected adaptive-moment update into one
    MPSGraph executable.  Maintains the standard ``m`` (first-moment)
    and ``v`` (second-moment / squared-gradient) running averages
    plus two stable 0-D scalar holders for the bias-correction
    factors :math:`1-\beta_1^t` and :math:`1-\beta_2^t`.  The factor
    *values* are refreshed via :meth:`_refresh_scalars` each step
    (advancing ``t`` is the only per-step CPU work); their *Tensor
    identities* stay constant so the cached executable continues to
    bind them as the same placeholders.

    Update rule
    -----------
    With first-moment decay :math:`\beta_1`, second-moment decay
    :math:`\beta_2`, learning rate :math:`\eta`, weight decay
    :math:`\lambda`, and step index :math:`t`:

    .. math::

        g_t      &\leftarrow g_t + \lambda \theta_t \\
        m_t      &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
        v_t      &= \beta_2 v_{t-1} + (1-\beta_2) g_t^{2} \\
        \hat m_t &= m_t / (1 - \beta_1^t) \\
        \hat v_t &= v_t / (1 - \beta_2^t) \\
        \theta_t &= \theta_{t-1}
                   - \eta \, \hat m_t / (\sqrt{\hat v_t} + \varepsilon)

    Weight decay is folded into the gradient *before* the moment
    update (coupled L₂), matching the eager :class:`~lucid.optim.Adam`.
    Use :class:`_CompiledAdamW` for the decoupled variant.

    Raises (at construct time)
    --------------------------
    NotImplementedError
        When ``amsgrad=True`` is set.  The maximum-history variant
        keeps an extra running max of the second moment — a
        straightforward extension once the test surface for it
        lands.

    See Also
    --------
    :class:`lucid.optim.Adam` : eager counterpart.
    :class:`_CompiledAdamW` : decoupled-weight-decay variant.
    """

    def __init__(self, opt: Optimizer) -> None:
        """Capture Adam hyperparameters + allocate ``m`` / ``v`` buffers.

        Pre-allocates the first-moment (``m``) and second-moment
        (``v``) running averages and the two bias-correction scalar
        holders.  Bias-correction values are refreshed via
        :meth:`_refresh_scalars` each step.

        Raises
        ------
        TypeError
            If ``opt`` is not an :class:`~lucid.optim.adam.Adam`
            instance.
        NotImplementedError
            If ``amsgrad=True`` is set — the maximum-history variant
            isn't supported on the compile path.
        """
        from lucid.optim.adam import Adam

        from lucid.compile._optim_spec import OptimizerSpec

        if not isinstance(opt, Adam):
            raise TypeError(f"_CompiledAdam: expected Adam, got {type(opt).__name__}")
        super().__init__(opt)
        g = opt.param_groups[0]
        if g.get("amsgrad", False):
            raise NotImplementedError(
                "compile_optimizer: AMSGrad variant not yet supported."
            )
        spec = OptimizerSpec.from_optim(opt)
        self._spec = spec
        self._lr = spec.lr
        self._beta1 = spec.beta1
        self._beta2 = spec.beta2
        self._eps = spec.eps
        self._weight_decay = spec.weight_decay
        self._m_buf = [_zeros_like(p) for p in self._params]
        self._v_buf = [_zeros_like(p) for p in self._params]
        self._t = 0
        for i in range(len(self._params)):
            self._buffer_table[("m", i)] = lambda _i=i: self._m_buf[_i]
            self._buffer_table[("v", i)] = lambda _i=i: self._v_buf[_i]

    def _register_state_in_inputs(self, register):
        """Register the ``m`` and ``v`` running-moment buffers as trace inputs."""
        for i, m in enumerate(self._m_buf):
            register("m", i, m)
        for i, v in enumerate(self._v_buf):
            register("v", i, v)

    def _register_scalars(self, register):
        """Register stable 0-D placeholders for the bias-correction factors.

        Returns a ``{"bias1", "bias2"}`` dict that the trace body
        reads — :meth:`_refresh_scalars` copies fresh values into
        the same placeholders each step so executable cache identity
        is preserved.
        """
        # Stable 0-D tensors for the bias-correction factors; values
        # refreshed via ``copy_`` each step.
        dt = self._params[0].dtype
        dev = self._params[0].device
        bias1 = _zero_scalar(dt, dev)
        bias2 = _zero_scalar(dt, dev)
        register("scalar", 0, bias1)
        register("scalar", 1, bias2)
        scalars = {"bias1": bias1, "bias2": bias2}
        self._scalar_slots = scalars
        return scalars

    def _trace_update(self, all_inputs, grads, scalars):
        """Emit the Adam update math (coupled weight decay; bias-corrected moments).

        Returns ``new_params + new_m + new_v`` in that order;
        :meth:`_outputs_to_targets` must mirror it.
        """
        bias1 = scalars["bias1"]
        bias2 = scalars["bias2"]
        params = self._params
        m_buf = self._m_buf
        v_buf = self._v_buf
        lr = self._lr
        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        wd = self._weight_decay
        new_params: list[Tensor] = []
        new_m: list[Tensor] = []
        new_v: list[Tensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd != 0.0:
                g = g + wd * p
            m_t = beta1 * m_buf[i] + (1.0 - beta1) * g
            v_t = beta2 * v_buf[i] + (1.0 - beta2) * (g * g)
            m_hat = m_t / bias1
            v_hat = v_t / bias2
            denom = v_hat.sqrt() + eps
            p_t = p - lr * m_hat / denom
            new_params.append(p_t)
            new_m.append(m_t)
            new_v.append(v_t)
        return new_params + new_m + new_v

    def _outputs_to_targets(self, outputs):
        """Map outputs to ``params`` then ``m_buf`` then ``v_buf`` in order."""
        return list(self._params) + list(self._m_buf) + list(self._v_buf)

    def _refresh_scalars(self) -> None:
        """Advance ``t`` and recompute ``bias1 = 1-β₁^t`` / ``bias2 = 1-β₂^t``.

        Writes the fresh values into the existing scalar holders via
        ``copy_`` so their TensorImpl identity stays stable and the
        cached executable continues to find them.
        """
        import lucid as _lucid

        self._t += 1
        bias1 = 1.0 - self._beta1**self._t
        bias2 = 1.0 - self._beta2**self._t
        dt = self._params[0].dtype
        dev = self._params[0].device
        # ``copy_`` writes through to the placeholder's existing buffer;
        # TensorImpl identity is preserved so the executable hits cache.
        self._scalar_slots["bias1"].copy_(_lucid.tensor(bias1, dtype=dt, device=dev))
        self._scalar_slots["bias2"].copy_(_lucid.tensor(bias2, dtype=dt, device=dev))


# ── AdamW ───────────────────────────────────────────────────────────


class _CompiledAdamW(_CompiledAdam):
    r"""Compiled :class:`~lucid.optim.AdamW` — Adam with decoupled weight decay.

    Identical state-buffer + scalar plumbing as :class:`_CompiledAdam`;
    differs only inside the trace body where weight decay is applied
    *directly to the parameter* rather than folded into the gradient.
    This decoupling is what makes AdamW the recommended optimizer
    for transformer-style training where weight decay needs to act
    as true regularisation rather than as an adaptive-LR-scaled
    perturbation.

    Update rule
    -----------
    .. math::

        m_t      &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
        v_t      &= \beta_2 v_{t-1} + (1-\beta_2) g_t^{2} \\
        \hat m_t &= m_t / (1 - \beta_1^t) \\
        \hat v_t &= v_t / (1 - \beta_2^t) \\
        \theta_t &= \theta_{t-1}
                   - \eta \, \big( \hat m_t / (\sqrt{\hat v_t} + \varepsilon)
                                   + \lambda \theta_{t-1} \big)

    The final ``λ·θ`` term is the decoupled decay; it never enters
    ``m_t`` / ``v_t`` so the adaptive scaling stays unbiased by the
    regularisation strength.

    Notes
    -----
    Construction bypasses :meth:`_CompiledAdam.__init__` (which
    would reject AdamW's runtime type) and reaches
    :meth:`_CompiledStepBase.__init__` directly.  The
    ``_register_scalars`` / ``_refresh_scalars`` / state-buffer
    inputs hooks are inherited verbatim from :class:`_CompiledAdam`.

    See Also
    --------
    :class:`lucid.optim.AdamW` : eager counterpart.
    :class:`_CompiledAdam` : coupled-weight-decay variant.
    """

    def __init__(self, opt: Optimizer) -> None:
        """Capture AdamW hyperparameters + allocate ``m`` / ``v`` buffers.

        Bypasses :meth:`_CompiledAdam.__init__` (which rejects
        AdamW's class type) and goes straight to the base
        :meth:`_CompiledStepBase.__init__`.
        """
        from lucid.optim.adam import AdamW
        from lucid.compile._optim_spec import OptimizerSpec

        if not isinstance(opt, AdamW):
            raise TypeError(f"_CompiledAdamW: expected AdamW, got {type(opt).__name__}")
        # Skip _CompiledAdam.__init__ (it rejects AdamW); reach the
        # _CompiledStepBase init directly.
        _CompiledStepBase.__init__(self, opt)
        spec = OptimizerSpec.from_optim(opt)
        self._spec = spec
        self._lr = spec.lr
        self._beta1 = spec.beta1
        self._beta2 = spec.beta2
        self._eps = spec.eps
        # AdamW default weight_decay is 0.01 (not 0); spec handles via param_group.
        g = opt.param_groups[0]
        self._weight_decay = float(g.get("weight_decay", 0.01))
        self._m_buf = [_zeros_like(p) for p in self._params]
        self._v_buf = [_zeros_like(p) for p in self._params]
        self._t = 0
        for i in range(len(self._params)):
            self._buffer_table[("m", i)] = lambda _i=i: self._m_buf[_i]
            self._buffer_table[("v", i)] = lambda _i=i: self._v_buf[_i]

    def _trace_update(self, all_inputs, grads, scalars):
        """Emit the AdamW update — decoupled weight decay variant.

        Same moment + bias-correction math as Adam, but weight decay
        is applied directly to the parameter (``p - lr * wd * p``)
        rather than folded into the gradient.  This avoids skewing
        the second-moment estimate by the decay term.
        """
        bias1 = scalars["bias1"]
        bias2 = scalars["bias2"]
        params = self._params
        m_buf = self._m_buf
        v_buf = self._v_buf
        lr = self._lr
        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        wd = self._weight_decay
        new_params: list[Tensor] = []
        new_m: list[Tensor] = []
        new_v: list[Tensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m_t = beta1 * m_buf[i] + (1.0 - beta1) * g
            v_t = beta2 * v_buf[i] + (1.0 - beta2) * (g * g)
            m_hat = m_t / bias1
            v_hat = v_t / bias2
            denom = v_hat.sqrt() + eps
            # Decoupled weight decay: ``p - lr * (m_hat / denom + wd * p)``
            p_t = p - lr * (m_hat / denom + wd * p)
            new_params.append(p_t)
            new_m.append(m_t)
            new_v.append(v_t)
        return new_params + new_m + new_v


# ── RMSprop ─────────────────────────────────────────────────────────


class _CompiledRMSprop(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.RMSprop`.

    Lifts Hinton's RMSProp update — an exponentially-smoothed
    running second moment normalises the step magnitude — into one
    MPSGraph executable.  An optional Polyak momentum buffer is
    layered on top when ``momentum != 0``; otherwise the
    parameter step is the raw scaled gradient.

    Update rule
    -----------
    With smoothing rate :math:`\alpha`, momentum :math:`\mu`,
    weight decay :math:`\lambda`, learning rate :math:`\eta`:

    .. math::

        g_t       &\leftarrow g_t + \lambda \theta_t \\
        s_t       &= \alpha s_{t-1} + (1-\alpha) g_t^{2} \\
        \tilde g  &= g_t / (\sqrt{s_t} + \varepsilon) \\
        b_t       &= \mu b_{t-1} + \tilde g
                       \quad\text{(only when } \mu \ne 0 \text{)} \\
        \theta_t  &= \theta_{t-1} - \eta \,
                       (b_t \text{ or } \tilde g)

    The ``centered=True`` variant (which subtracts a running gradient
    mean to compute a *centered* second moment) is rejected at
    construct time — the eager backend silently drops the flag, so
    surfacing it loudly here keeps compile + eager in agreement.

    Raises (at construct time)
    --------------------------
    NotImplementedError
        When ``centered=True`` is set.

    See Also
    --------
    :class:`lucid.optim.RMSprop` : eager counterpart.
    """

    def __init__(self, opt: Optimizer) -> None:
        """Capture RMSprop hyperparameters + allocate ``square_avg`` (+ momentum) buffers.

        Raises :class:`NotImplementedError` for ``centered=True`` —
        the eager backend silently drops that flag, so surfacing it
        loudly here keeps the compile + eager paths in agreement.
        """
        from lucid.optim.others import RMSprop

        if not isinstance(opt, RMSprop):
            raise TypeError(
                f"_CompiledRMSprop: expected RMSprop, got {type(opt).__name__}"
            )
        super().__init__(opt)
        g = opt.param_groups[0]
        if g.get("centered", False):
            raise NotImplementedError(
                "compile_optimizer: RMSprop(centered=True) is not yet "
                "supported.  The Lucid eager backend silently drops the "
                "flag too; we surface it as a compile-time error so "
                "callers know to switch to centered=False."
            )
        self._lr = float(g["lr"])
        self._alpha = float(g.get("alpha", 0.99))
        self._eps = float(g.get("eps", 1e-8))
        self._weight_decay = float(g.get("weight_decay", 0.0))
        self._momentum = float(g.get("momentum", 0.0))
        # State: square_avg (always); momentum buffer when momentum != 0.
        self._square_avg = [_zeros_like(p) for p in self._params]
        if self._momentum != 0.0:
            self._momenta = [_zeros_like(p) for p in self._params]
        else:
            self._momenta = []
        for i in range(len(self._params)):
            self._buffer_table[("square_avg", i)] = lambda _i=i: self._square_avg[_i]
        for i in range(len(self._momenta)):
            self._buffer_table[("mom", i)] = lambda _i=i: self._momenta[_i]

    def _register_state_in_inputs(self, register):
        """Register the ``square_avg`` (and optional ``mom``) buffers as trace inputs."""
        for i, sa in enumerate(self._square_avg):
            register("square_avg", i, sa)
        for i, m in enumerate(self._momenta):
            register("mom", i, m)

    def _trace_update(self, all_inputs, grads, scalars):
        """Emit the RMSprop update — exponentially smoothed squared gradient.

        When ``momentum != 0`` a Polyak-momentum buffer is also
        updated and used in the parameter step.  Returns
        ``new_params + new_sq + new_mom`` in that order.
        """
        params = self._params
        sq = self._square_avg
        mom = self._momenta
        lr = self._lr
        alpha = self._alpha
        eps = self._eps
        wd = self._weight_decay
        mu = self._momentum
        new_params: list[Tensor] = []
        new_sq: list[Tensor] = []
        new_mom: list[Tensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd != 0.0:
                g = g + wd * p
            new_v = alpha * sq[i] + (1.0 - alpha) * (g * g)
            denom = new_v.sqrt() + eps
            new_sq.append(new_v)
            if mu != 0.0:
                new_b = mu * mom[i] + g / denom
                new_mom.append(new_b)
                new_p = p - lr * new_b
            else:
                new_p = p - lr * g / denom
            new_params.append(new_p)
        return new_params + new_sq + new_mom

    def _outputs_to_targets(self, outputs):
        """Map outputs to ``params`` then ``square_avg`` then (optional) ``momenta``."""
        return list(self._params) + list(self._square_avg) + list(self._momenta)


# ── Adagrad ─────────────────────────────────────────────────────────


class _CompiledAdagrad(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Adagrad`.

    Duchi's per-parameter adaptive LR: each parameter scales its
    step by the inverse square root of its own historical squared-
    gradient sum.  Effective LR decays monotonically across steps,
    which makes Adagrad well-suited to sparse-gradient regimes
    (NLP feature embeddings, RecSys-style models) but typically
    too aggressive for dense vision training.

    Update rule
    -----------
    With base LR :math:`\eta_0`, LR-decay rate :math:`\gamma`,
    weight decay :math:`\lambda`, step :math:`t`:

    .. math::

        g_t       &\leftarrow g_t + \lambda \theta_t \\
        s_t       &= s_{t-1} + g_t^{2}                 \\
        \eta_t    &= \eta_0 / (1 + (t-1)\gamma)        \\
        \theta_t  &= \theta_{t-1}
                       - \eta_t \, g_t / (\sqrt{s_t} + \varepsilon)

    ``lr_decay`` is folded into a per-step scalar feed
    (``eff_lr = η_0 / (1 + (t-1)γ)``) rather than the trace body,
    so the executable signature stays constant across steps even as
    ``t`` advances.  :meth:`_refresh_scalars` writes the fresh value
    via ``copy_`` so the placeholder identity is preserved.

    See Also
    --------
    :class:`lucid.optim.Adagrad` : eager counterpart.
    """

    def __init__(self, opt: Optimizer) -> None:
        """Capture Adagrad hyperparameters + allocate ``state_sum`` accumulators.

        ``lr_decay`` is not folded into the trace — instead the
        effective LR (``lr / (1 + (t-1)*lr_decay)``) is fed as a 0-D
        scalar refreshed in :meth:`_refresh_scalars` each step.  This
        keeps the executable signature constant across steps.
        """
        from lucid.optim.others import Adagrad

        if not isinstance(opt, Adagrad):
            raise TypeError(
                f"_CompiledAdagrad: expected Adagrad, got {type(opt).__name__}"
            )
        super().__init__(opt)
        g = opt.param_groups[0]
        self._lr = float(g["lr"])
        self._lr_decay = float(g.get("lr_decay", 0.0))
        self._weight_decay = float(g.get("weight_decay", 0.0))
        self._eps = float(g.get("eps", 1e-10))
        self._state_sum = [_zeros_like(p) for p in self._params]
        self._t = 0
        for i in range(len(self._params)):
            self._buffer_table[("state_sum", i)] = lambda _i=i: self._state_sum[_i]

    def _register_state_in_inputs(self, register):
        """Register the ``state_sum`` accumulators as trace inputs."""
        for i, s in enumerate(self._state_sum):
            register("state_sum", i, s)

    def _register_scalars(self, register):
        """Register the effective-LR scalar placeholder (refreshed each step)."""
        # Effective LR: ``lr / (1 + (t-1)*lr_decay)`` — t-dependent, so
        # passed as a 0-D feed and refreshed each step.
        dt = self._params[0].dtype
        dev = self._params[0].device
        eff_lr = _zero_scalar(dt, dev)
        register("scalar", 0, eff_lr)
        scalars = {"eff_lr": eff_lr}
        self._scalar_slots = scalars
        return scalars

    def _refresh_scalars(self) -> None:
        """Advance ``t`` and copy fresh ``eff_lr`` value into the scalar holder."""
        import lucid as _lucid

        self._t += 1
        eff = self._lr / (1.0 + (self._t - 1) * self._lr_decay)
        dt = self._params[0].dtype
        dev = self._params[0].device
        self._scalar_slots["eff_lr"].copy_(_lucid.tensor(eff, dtype=dt, device=dev))

    def _trace_update(self, all_inputs, grads, scalars):
        """Emit the Adagrad update — running sum of squared gradients normalises LR."""
        eff_lr = scalars["eff_lr"]
        params = self._params
        state_sum = self._state_sum
        wd = self._weight_decay
        eps = self._eps
        new_params: list[Tensor] = []
        new_state: list[Tensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd != 0.0:
                g = g + wd * p
            new_s = state_sum[i] + g * g
            denom = new_s.sqrt() + eps
            new_p = p - eff_lr * g / denom
            new_params.append(new_p)
            new_state.append(new_s)
        return new_params + new_state

    def _outputs_to_targets(self, outputs):
        """Map outputs to ``params`` then ``state_sum``."""
        return list(self._params) + list(self._state_sum)


# ── Adadelta ────────────────────────────────────────────────────────


class _CompiledAdadelta(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Adadelta` — auto-adaptive LR.

    Zeiler's Adadelta maintains *two* running averages — one of
    squared gradients and one of squared parameter deltas — and
    uses the ratio of their RMS values as the adaptive step size.
    The ``lr`` parameter acts only as a multiplicative scaling on
    the resulting delta (default ``1.0``), so the optimizer is
    effectively learning-rate-free.

    Update rule
    -----------
    With smoothing rate :math:`\rho`, weight decay :math:`\lambda`,
    final scaling :math:`\eta`:

    .. math::

        g_t       &\leftarrow g_t + \lambda \theta_t \\
        v_t       &= \rho v_{t-1} + (1-\rho) g_t^{2}                 \\
        \Delta_t  &= \frac{\sqrt{u_{t-1} + \varepsilon}}
                          {\sqrt{v_t + \varepsilon}} \, g_t          \\
        u_t       &= \rho u_{t-1} + (1-\rho) \Delta_t^{2}            \\
        \theta_t  &= \theta_{t-1} - \eta \, \Delta_t

    where :math:`v_t` accumulates squared gradients and :math:`u_t`
    accumulates squared deltas.  Both buffers are initialised to
    zero, so the first few steps take small conservative updates
    until ``u`` warms up.

    See Also
    --------
    :class:`lucid.optim.Adadelta` : eager counterpart.
    """

    def __init__(self, opt: Optimizer) -> None:
        """Capture Adadelta hyperparameters + allocate ``square_avg`` / ``acc_delta`` buffers.

        Adadelta uses *two* running averages — squared gradients and
        squared parameter deltas — so the ratio of their RMS values
        serves as an adaptive step size that needs no manual LR
        tuning.  The ``lr`` argument acts as a multiplicative scaling
        on the final delta only.
        """
        from lucid.optim.others import Adadelta

        if not isinstance(opt, Adadelta):
            raise TypeError(
                f"_CompiledAdadelta: expected Adadelta, got {type(opt).__name__}"
            )
        super().__init__(opt)
        g = opt.param_groups[0]
        self._lr = float(g["lr"])
        self._rho = float(g.get("rho", 0.9))
        self._eps = float(g.get("eps", 1e-6))
        self._weight_decay = float(g.get("weight_decay", 0.0))
        self._square_avg = [_zeros_like(p) for p in self._params]
        self._acc_delta = [_zeros_like(p) for p in self._params]
        for i in range(len(self._params)):
            self._buffer_table[("square_avg", i)] = lambda _i=i: self._square_avg[_i]
            self._buffer_table[("acc_delta", i)] = lambda _i=i: self._acc_delta[_i]

    def _register_state_in_inputs(self, register):
        """Register the ``square_avg`` and ``acc_delta`` running buffers as trace inputs."""
        for i, sa in enumerate(self._square_avg):
            register("square_avg", i, sa)
        for i, ad in enumerate(self._acc_delta):
            register("acc_delta", i, ad)

    def _trace_update(self, all_inputs, grads, scalars):
        """Emit the Adadelta update — RMS(delta) / RMS(grad) as adaptive step size."""
        params = self._params
        sq = self._square_avg
        ad = self._acc_delta
        lr = self._lr
        rho = self._rho
        eps = self._eps
        wd = self._weight_decay
        new_params: list[Tensor] = []
        new_sq: list[Tensor] = []
        new_ad: list[Tensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd != 0.0:
                g = g + wd * p
            new_v = rho * sq[i] + (1.0 - rho) * (g * g)
            delta = ((ad[i] + eps).sqrt() / (new_v + eps).sqrt()) * g
            new_d = rho * ad[i] + (1.0 - rho) * (delta * delta)
            new_p = p - lr * delta
            new_params.append(new_p)
            new_sq.append(new_v)
            new_ad.append(new_d)
        return new_params + new_sq + new_ad

    def _outputs_to_targets(self, outputs):
        """Map outputs to ``params`` then ``square_avg`` then ``acc_delta``."""
        return list(self._params) + list(self._square_avg) + list(self._acc_delta)


# ── Adamax ──────────────────────────────────────────────────────────


class _CompiledAdamax(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.Adamax` — Adam with L∞-norm second moment.

    Variant of Adam that replaces the L² second-moment estimate
    :math:`v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2` with an
    L∞-norm running max
    :math:`u_t = \max(\beta_2 u_{t-1}, |g_t|)`.  More robust to
    occasional gradient outliers than vanilla Adam — useful when
    training signals can spike (sparse rewards in RL, rare-class
    losses in long-tailed classification).

    Update rule
    -----------
    .. math::

        g_t       &\leftarrow g_t + \lambda \theta_t \\
        m_t       &= \beta_1 m_{t-1} + (1-\beta_1) g_t                 \\
        u_t       &= \max\bigl(\beta_2 u_{t-1},\, |g_t|\bigr)          \\
        \eta_t    &= \eta / (1 - \beta_1^{t})                          \\
        \theta_t  &= \theta_{t-1} - \eta_t \, m_t / (u_t + \varepsilon)

    The bias-corrected effective LR :math:`\eta_t` is fed as a 0-D
    scalar refreshed by :meth:`_refresh_scalars` each step so the
    trace stays signature-stable.

    Notes
    -----
    The element-wise ``maximum(β₂ u_{t-1}, |g|)`` step requires the
    ``maximum`` op's compile emitter (real-emit, see
    ``OpEmitters/elementwise/Arith.mm``).  Without it the trace
    would abort during compile and the optimizer would silently
    bail to eager — :func:`compile_optimizer` instead surfaces a
    construct-time error if the emitter were ever removed.

    See Also
    --------
    :class:`lucid.optim.Adamax` : eager counterpart.
    """

    def __init__(self, opt: Optimizer) -> None:
        """Capture Adamax hyperparameters + allocate ``m`` (first-moment) / ``u`` (L∞) buffers.

        Replaces Adam's L² second-moment estimate with an L∞-norm
        running max, which is more robust to gradient outliers.  The
        bias-corrected effective LR is fed as a scalar refreshed by
        :meth:`_refresh_scalars`.
        """
        from lucid.optim.others import Adamax

        if not isinstance(opt, Adamax):
            raise TypeError(
                f"_CompiledAdamax: expected Adamax, got {type(opt).__name__}"
            )
        super().__init__(opt)
        g = opt.param_groups[0]
        self._lr = float(g["lr"])
        self._beta1 = float(g.get("beta1", 0.9))
        self._beta2 = float(g.get("beta2", 0.999))
        self._eps = float(g.get("eps", 1e-8))
        self._weight_decay = float(g.get("weight_decay", 0.0))
        self._m_buf = [_zeros_like(p) for p in self._params]
        self._u_buf = [_zeros_like(p) for p in self._params]
        self._t = 0
        for i in range(len(self._params)):
            self._buffer_table[("m", i)] = lambda _i=i: self._m_buf[_i]
            self._buffer_table[("u", i)] = lambda _i=i: self._u_buf[_i]

    def _register_state_in_inputs(self, register):
        """Register the ``m`` (first-moment) and ``u`` (L∞-norm) buffers as trace inputs."""
        for i, m in enumerate(self._m_buf):
            register("m", i, m)
        for i, u in enumerate(self._u_buf):
            register("u", i, u)

    def _register_scalars(self, register):
        """Register the bias-corrected effective-LR scalar placeholder."""
        # ``eff_lr = lr / (1 - beta1^t)`` — refreshed each step.
        dt = self._params[0].dtype
        dev = self._params[0].device
        eff_lr = _zero_scalar(dt, dev)
        register("scalar", 0, eff_lr)
        scalars = {"eff_lr": eff_lr}
        self._scalar_slots = scalars
        return scalars

    def _refresh_scalars(self) -> None:
        """Advance ``t`` and copy fresh ``lr / (1 - β₁^t)`` into the scalar holder."""
        import lucid as _lucid

        self._t += 1
        eff = self._lr / (1.0 - self._beta1**self._t)
        dt = self._params[0].dtype
        dev = self._params[0].device
        self._scalar_slots["eff_lr"].copy_(_lucid.tensor(eff, dtype=dt, device=dev))

    def _trace_update(self, all_inputs, grads, scalars):
        """Emit the Adamax update — L∞-norm running max replaces Adam's L² moment."""
        import lucid as _lucid

        eff_lr = scalars["eff_lr"]
        params = self._params
        m_buf = self._m_buf
        u_buf = self._u_buf
        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        wd = self._weight_decay
        new_params: list[Tensor] = []
        new_m: list[Tensor] = []
        new_u: list[Tensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd != 0.0:
                g = g + wd * p
            m_t = beta1 * m_buf[i] + (1.0 - beta1) * g
            u_t = _lucid.maximum(beta2 * u_buf[i], g.abs())
            new_p = p - eff_lr * m_t / (u_t + eps)
            new_params.append(new_p)
            new_m.append(m_t)
            new_u.append(u_t)
        return new_params + new_m + new_u

    def _outputs_to_targets(self, outputs):
        """Map outputs to ``params`` then ``m_buf`` then ``u_buf`` (Adamax)."""
        return list(self._params) + list(self._m_buf) + list(self._u_buf)


# ── NAdam ───────────────────────────────────────────────────────────


class _CompiledNAdam(_CompiledStepBase):
    r"""Compiled :class:`~lucid.optim.NAdam` — Adam with Nesterov lookahead.

    Combines Adam's adaptive moments with Nesterov's anticipatory
    gradient correction, using the closed-form momentum-decay
    schedule introduced by Dozat (2016).  In practice converges
    slightly faster than Adam on well-conditioned objectives while
    keeping the same robustness to gradient scale.

    Update rule
    -----------
    With momentum-decay rate :math:`d` (constant ``0.004`` in
    Lucid, matching the eager C++ default), step :math:`t`, base
    LR :math:`\eta`, decays :math:`(\beta_1, \beta_2)`, weight
    decay :math:`\lambda`:

    .. math::

        \mu_t       &= \beta_1 \bigl( 1 - 0.5 \cdot 0.96^{td} \bigr) \\
        \mu_{t+1}   &= \beta_1 \bigl( 1 - 0.5 \cdot 0.96^{(t+1)d} \bigr) \\
        \Pi_t       &= \prod_{k \le t} \mu_k \\
        g_t         &\leftarrow g_t + \lambda \theta_t \\
        m_t         &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t         &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^{2} \\
        \mathrm{denom} &= \sqrt{v_t / (1 - \beta_2^{t})} + \varepsilon \\
        \theta_t    &= \theta_{t-1}
                       - c_1 \, g_t / \mathrm{denom}
                       - c_2 \, m_t / \mathrm{denom}

    where the per-step coefficients fed in as 0-D scalars are:

    * ``c1``      = :math:`\eta (1 - \mu_t) / (1 - \Pi_t)`
    * ``c2``      = :math:`\eta \mu_{t+1}   / (1 - \Pi_t \mu_{t+1})`
    * ``inv_bc2`` = :math:`1 / (1 - \beta_2^{t})`

    All three are recomputed CPU-side every step in
    :meth:`_refresh_scalars` and copied into the stable placeholders
    via ``copy_``.

    Notes
    -----
    Lucid's NAdam holds ``momentum_decay`` at the C++ default
    ``0.004`` and doesn't expose it on the Python constructor; this
    wrapper mirrors that constant verbatim.  Exposing the knob would
    be a small follow-up if anyone ever needs a different schedule.

    See Also
    --------
    :class:`lucid.optim.NAdam` : eager counterpart.
    """

    # Lucid's eager NAdam fixes momentum_decay at the C++ default; the
    # Python constructor doesn't accept it.  Mirror that here.
    _MOMENTUM_DECAY: float = 0.004

    def __init__(self, opt: Optimizer) -> None:
        """Capture NAdam hyperparameters + allocate ``m`` / ``v`` buffers and ``μ_product``.

        The ``μ_product`` is a single CPU-side accumulator (not a
        per-parameter tensor) because every parameter multiplies by
        the same ``μ(t)`` schedule each step — Lucid's NAdam holds
        ``momentum_decay`` at the C++ default (``0.004``) and doesn't
        expose it via the Python constructor; we mirror that
        verbatim.
        """
        from lucid.optim.others import NAdam

        if not isinstance(opt, NAdam):
            raise TypeError(f"_CompiledNAdam: expected NAdam, got {type(opt).__name__}")
        super().__init__(opt)
        g = opt.param_groups[0]
        self._lr = float(g["lr"])
        self._beta1 = float(g.get("beta1", 0.9))
        self._beta2 = float(g.get("beta2", 0.999))
        self._eps = float(g.get("eps", 1e-8))
        self._weight_decay = float(g.get("weight_decay", 0.0))
        self._m_buf = [_zeros_like(p) for p in self._params]
        self._v_buf = [_zeros_like(p) for p in self._params]
        # mu_product accumulator — shared across params because each
        # param multiplies by the same μ(t) every step starting from
        # 1.0, so the per-param vector is degenerate.
        self._mu_product: float = 1.0
        self._t = 0
        for i in range(len(self._params)):
            self._buffer_table[("m", i)] = lambda _i=i: self._m_buf[_i]
            self._buffer_table[("v", i)] = lambda _i=i: self._v_buf[_i]

    def _register_state_in_inputs(self, register):
        """Register the NAdam ``m`` and ``v`` running-moment buffers as trace inputs."""
        for i, m in enumerate(self._m_buf):
            register("m", i, m)
        for i, v in enumerate(self._v_buf):
            register("v", i, v)

    def _register_scalars(self, register):
        """Register the three NAdam coefficient scalars (``c1``, ``c2``, ``inv_bc2``)."""
        dt = self._params[0].dtype
        dev = self._params[0].device
        c1 = _zero_scalar(dt, dev)
        c2 = _zero_scalar(dt, dev)
        inv_bc2 = _zero_scalar(dt, dev)
        register("scalar", 0, c1)
        register("scalar", 1, c2)
        register("scalar", 2, inv_bc2)
        scalars = {"c1": c1, "c2": c2, "inv_bc2": inv_bc2}
        self._scalar_slots = scalars
        return scalars

    def _refresh_scalars(self) -> None:
        """Advance ``t``, recompute ``μ_t`` / ``μ_{t+1}`` / ``μ_product``, refresh c1 / c2 / inv_bc2.

        See the class docstring for the closed-form expressions.
        """
        import lucid as _lucid

        self._t += 1
        t = self._t
        beta1 = self._beta1
        mom_decay = self._MOMENTUM_DECAY
        mu_t = beta1 * (1.0 - 0.5 * (0.96 ** (t * mom_decay)))
        mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((t + 1) * mom_decay)))
        self._mu_product *= mu_t
        mu_prod_next = self._mu_product * mu_next
        bc2 = 1.0 - self._beta2**t
        c1 = self._lr * (1.0 - mu_t) / (1.0 - self._mu_product)
        c2 = self._lr * mu_next / (1.0 - mu_prod_next)
        inv_bc2 = 1.0 / bc2
        dt = self._params[0].dtype
        dev = self._params[0].device
        self._scalar_slots["c1"].copy_(_lucid.tensor(c1, dtype=dt, device=dev))
        self._scalar_slots["c2"].copy_(_lucid.tensor(c2, dtype=dt, device=dev))
        self._scalar_slots["inv_bc2"].copy_(
            _lucid.tensor(inv_bc2, dtype=dt, device=dev)
        )

    def _trace_update(self, all_inputs, grads, scalars):
        """Emit the NAdam update — Nesterov lookahead applied to Adam's bias-corrected step.

        Uses the three pre-computed scalar coefficients (``c1`` /
        ``c2`` / ``inv_bc2``) so the trace stays signature-stable
        across steps even as ``t`` advances inside the closed-form
        ``μ_t`` schedule.
        """
        c1 = scalars["c1"]
        c2 = scalars["c2"]
        inv_bc2 = scalars["inv_bc2"]
        params = self._params
        m_buf = self._m_buf
        v_buf = self._v_buf
        beta1 = self._beta1
        beta2 = self._beta2
        eps = self._eps
        wd = self._weight_decay
        new_params: list[Tensor] = []
        new_m: list[Tensor] = []
        new_v: list[Tensor] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if wd != 0.0:
                g = g + wd * p
            m_t = beta1 * m_buf[i] + (1.0 - beta1) * g
            v_t = beta2 * v_buf[i] + (1.0 - beta2) * (g * g)
            denom = (v_t * inv_bc2).sqrt() + eps
            new_p = p - c1 * (g / denom) - c2 * (m_t / denom)
            new_params.append(new_p)
            new_m.append(m_t)
            new_v.append(v_t)
        return new_params + new_m + new_v

    def _outputs_to_targets(self, outputs):
        """Map outputs to ``params`` then ``m_buf`` then ``v_buf`` (NAdam)."""
        return list(self._params) + list(self._m_buf) + list(self._v_buf)


# ── Multi-group wrapper ─────────────────────────────────────────────


class _MultiGroupCompiledOptimizer:
    """Drop-in compiled-optimizer wrapper for multi-``param_group`` setups.

    Each parameter group becomes its own compiled-optimizer instance
    (one MPSGraph executable per group), constructed from a synthetic
    single-group clone of the parent optimizer that carries only that
    group's parameters + hyperparameters.  At ``step()`` time we
    iterate the per-group compiled wrappers in order — eager has the
    same shape (eager's ``step()`` loops over engine optimizers, one
    per group), so this preserves the user-visible contract.

    Lifecycle delegation (``zero_grad`` / ``param_groups`` /
    ``state_dict`` / ``load_state_dict``) routes back to the parent
    optimizer; the per-group wrappers do not own optimizer state.

    Motivation: backbone-vs-head training recipes use distinct LRs
    per group (e.g. lr=1e-3 for the pretrained backbone, lr=1e-2 for
    the freshly initialised classification head).  Without this
    wrapper, ``compile_optimizer`` would reject such setups and the
    user would fall back to eager training step.

    See [[retro-3-5-phase-vjp-priorities-p1-p7]] for the design
    rationale and the alternative "per-group LR baked into one
    trace" approach that was rejected (would require dynamic indexing
    into a per-parameter LR table which MPSGraph doesn't express
    cleanly).
    """

    def __init__(self, opt: Optimizer) -> None:
        self._opt = opt
        self._per_group: list[object] = [
            compile_optimizer(_clone_single_group(opt, i))
            for i in range(len(opt.param_groups))
        ]

    # ── Drop-in API surface ──────────────────────────────────────

    @property
    def param_groups(self) -> list[dict[str, object]]:
        return self._opt.param_groups

    @property
    def defaults(self) -> dict[str, object]:
        return self._opt.defaults

    @property
    def state(self) -> dict[int, dict[str, object]]:
        return self._opt.state

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, object]:
        return self._opt.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._opt.load_state_dict(state)
        # Drop per-group caches so the next ``step()`` retraces with
        # the freshly loaded buffers.
        for cg in self._per_group:
            if hasattr(cg, "load_state_dict"):
                cg.load_state_dict(state)
            elif hasattr(cg, "_exe"):
                cg._exe = None

    def step(self) -> None:
        """Run each per-group compiled optimizer in turn."""
        for cg in self._per_group:
            cg.step()


def _clone_single_group(opt: Optimizer, group_idx: int) -> Optimizer:
    """Construct a synthetic single-group optimizer from a single group.

    The clone is the same concrete class as ``opt`` (so
    ``compile_optimizer`` dispatches to the same ``_Compiled*``
    subclass), but carries only ``opt.param_groups[group_idx]``'s
    parameters and hyperparameters.  The clone shares parameter
    tensor identities with the original — no copies — so updates
    written by the clone's ``step()`` flow back to the user's
    parameters automatically.

    Hyperparameter forwarding: the group dict's keys (excluding
    ``"params"``) are passed as keyword arguments to the optimizer's
    ``__init__``.  Each Lucid optimizer class accepts the standard
    set (lr / momentum / betas / eps / etc.) by name, so this works
    uniformly across SGD / Adam / RMSprop / etc.
    """
    group = opt.param_groups[group_idx]
    params = list(group["params"])  # type: ignore[arg-type]
    kwargs = {k: v for k, v in group.items() if k != "params"}
    return type(opt)(params, **kwargs)
