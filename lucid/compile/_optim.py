"""
lucid.compile._optim — compiled optimizer wrappers (in-place output path).

Wraps an eager :class:`lucid.optim.Optimizer` so ``step()`` becomes a
single MPSGraph executable that fuses the per-parameter update math AND
writes its outputs directly back into the parameter / state buffers via
:func:`run_executable_inplace`.

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
   :func:`compile_or_cached` to mint a :class:`MPSGraphExecutable`.
2. **Per-step run** — collects fresh grads + bias-correction scalars,
   calls :func:`run_executable_inplace` with the parameter and state
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
    """Wrap ``opt`` so ``opt.step()`` runs as a single MPSGraph executable.

    Parameters
    ----------
    opt : lucid.optim.Optimizer
        Concrete optimizer instance (SGD / Adam / AdamW).

    Returns
    -------
    object
        Drop-in replacement exposing ``step()``, ``zero_grad()``,
        ``param_groups``, ``state_dict()`` / ``load_state_dict()``.

    Raises
    ------
    TypeError
        When ``opt``'s class isn't one of the supported families.
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
    """Return every Parameter across every param_group, in order."""
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
    """Shared compile + run plumbing for the optim wrappers.

    Concrete subclasses (SGD / Adam / AdamW) supply two things:

    * ``_collect_inputs()`` — returns the ordered list of tensors that
      the update function reads (params, state buffers, grads, bias
      corrections — in whichever order the trace recorded them).
    * ``_outputs_targets()`` — returns the ordered list of target
      tensors the executable's outputs should be written into.  The
      ordering must match the trace's return order.
    """

    def __init__(self, opt: Optimizer) -> None:
        self._opt = opt
        self._params = _flatten_params(opt)
        if not self._params:
            raise ValueError("compile_optimizer: optimizer has no trainable parameters")
        if len(opt.param_groups) > 1:
            raise NotImplementedError(
                "compile_optimizer: multi-group optimizers not yet supported.  "
                "Use a single param_group."
            )
        # Filled by the concrete subclass before _build_executable().
        self._exe: object | None = None
        # Per-input-slot category descriptor, populated at compile time.
        # Each entry is a tuple ``(kind, index)`` matched against
        # :attr:`_buffer_table`.  Built-in kinds: "param", "grad",
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
        return self._opt.param_groups

    @property
    def state(self) -> dict[int, dict[str, object]]:
        return self._opt.state

    @property
    def defaults(self) -> dict[str, object]:
        return self._opt.defaults

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, object]:
        return self._opt.state_dict()

    def load_state_dict(self, sd: dict[str, object]) -> None:
        self._opt.load_state_dict(sd)
        # Recompile next step — state buffer identity may have moved.
        self._exe = None

    # ── Internals ────────────────────────────────────────────────

    def _resolve_input(self, plan_entry: tuple) -> Tensor:
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
    """SGD wrapper (with optional momentum / nesterov / weight_decay)."""

    def __init__(self, opt: Optimizer) -> None:
        from lucid.optim.sgd import SGD

        if not isinstance(opt, SGD):
            raise TypeError(f"_CompiledSGD: expected SGD, got {type(opt).__name__}")
        super().__init__(opt)
        g = opt.param_groups[0]
        self._lr = float(g["lr"])
        self._momentum = float(g.get("momentum", 0.0))
        self._dampening = float(g.get("dampening", 0.0))
        self._weight_decay = float(g.get("weight_decay", 0.0))
        self._nesterov = bool(g.get("nesterov", False))
        if self._momentum != 0.0:
            self._momenta = [_zeros_like(p) for p in self._params]
        else:
            self._momenta = []
        for i in range(len(self._momenta)):
            self._buffer_table[("mom", i)] = lambda _i=i: self._momenta[_i]

    def _register_state_in_inputs(self, register):
        for i, m in enumerate(self._momenta):
            register("mom", i, m)

    def _trace_update(self, all_inputs, grads, scalars):
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
        # First N outputs → self._params, next N → self._momenta.
        n = len(self._params)
        return list(self._params) + list(self._momenta)


# ── Adam ────────────────────────────────────────────────────────────


class _CompiledAdam(_CompiledStepBase):
    """Adam wrapper (no AMSGrad)."""

    def __init__(self, opt: Optimizer) -> None:
        from lucid.optim.adam import Adam

        if not isinstance(opt, Adam):
            raise TypeError(f"_CompiledAdam: expected Adam, got {type(opt).__name__}")
        super().__init__(opt)
        g = opt.param_groups[0]
        self._lr = float(g["lr"])
        self._beta1 = float(g.get("beta1", 0.9))
        self._beta2 = float(g.get("beta2", 0.999))
        self._eps = float(g.get("eps", 1e-8))
        self._weight_decay = float(g.get("weight_decay", 0.0))
        if g.get("amsgrad", False):
            raise NotImplementedError(
                "compile_optimizer: AMSGrad variant not yet supported."
            )
        self._m_buf = [_zeros_like(p) for p in self._params]
        self._v_buf = [_zeros_like(p) for p in self._params]
        self._t = 0
        for i in range(len(self._params)):
            self._buffer_table[("m", i)] = lambda _i=i: self._m_buf[_i]
            self._buffer_table[("v", i)] = lambda _i=i: self._v_buf[_i]

    def _register_state_in_inputs(self, register):
        for i, m in enumerate(self._m_buf):
            register("m", i, m)
        for i, v in enumerate(self._v_buf):
            register("v", i, v)

    def _register_scalars(self, register):
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
        return list(self._params) + list(self._m_buf) + list(self._v_buf)

    def _refresh_scalars(self) -> None:
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
    """AdamW wrapper — Adam with decoupled weight decay.

    Differs from Adam only in the trace body: weight decay is applied
    directly to the parameter outside the gradient (decoupled), not
    folded into ``g`` before the moment updates.  All state-buffer +
    scalar plumbing is identical to :class:`_CompiledAdam`.
    """

    def __init__(self, opt: Optimizer) -> None:
        from lucid.optim.adam import AdamW

        if not isinstance(opt, AdamW):
            raise TypeError(f"_CompiledAdamW: expected AdamW, got {type(opt).__name__}")
        # Skip _CompiledAdam.__init__ (it rejects AdamW); reach the
        # _CompiledStepBase init directly.
        _CompiledStepBase.__init__(self, opt)
        g = opt.param_groups[0]
        self._lr = float(g["lr"])
        self._beta1 = float(g.get("beta1", 0.9))
        self._beta2 = float(g.get("beta2", 0.999))
        self._eps = float(g.get("eps", 1e-8))
        self._weight_decay = float(g.get("weight_decay", 0.01))
        self._m_buf = [_zeros_like(p) for p in self._params]
        self._v_buf = [_zeros_like(p) for p in self._params]
        self._t = 0
        for i in range(len(self._params)):
            self._buffer_table[("m", i)] = lambda _i=i: self._m_buf[_i]
            self._buffer_table[("v", i)] = lambda _i=i: self._v_buf[_i]

    def _trace_update(self, all_inputs, grads, scalars):
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
    """RMSprop wrapper.  Centered variant rejected (eager backend
    silently ignores the flag — better to fail loudly here)."""

    def __init__(self, opt: Optimizer) -> None:
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
        for i, sa in enumerate(self._square_avg):
            register("square_avg", i, sa)
        for i, m in enumerate(self._momenta):
            register("mom", i, m)

    def _trace_update(self, all_inputs, grads, scalars):
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
        return list(self._params) + list(self._square_avg) + list(self._momenta)


# ── Adagrad ─────────────────────────────────────────────────────────


class _CompiledAdagrad(_CompiledStepBase):
    """Adagrad wrapper.  lr_decay applied via a per-step scalar feed
    so the executable stays signature-stable across steps."""

    def __init__(self, opt: Optimizer) -> None:
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
        for i, s in enumerate(self._state_sum):
            register("state_sum", i, s)

    def _register_scalars(self, register):
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
        import lucid as _lucid

        self._t += 1
        eff = self._lr / (1.0 + (self._t - 1) * self._lr_decay)
        dt = self._params[0].dtype
        dev = self._params[0].device
        self._scalar_slots["eff_lr"].copy_(_lucid.tensor(eff, dtype=dt, device=dev))

    def _trace_update(self, all_inputs, grads, scalars):
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
        return list(self._params) + list(self._state_sum)


# ── Adadelta ────────────────────────────────────────────────────────


class _CompiledAdadelta(_CompiledStepBase):
    """Adadelta wrapper — auto-adaptive LR via running ratio of
    delta-RMS over gradient-RMS."""

    def __init__(self, opt: Optimizer) -> None:
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
        for i, sa in enumerate(self._square_avg):
            register("square_avg", i, sa)
        for i, ad in enumerate(self._acc_delta):
            register("acc_delta", i, ad)

    def _trace_update(self, all_inputs, grads, scalars):
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
        return list(self._params) + list(self._square_avg) + list(self._acc_delta)


# ── Adamax ──────────────────────────────────────────────────────────


class _CompiledAdamax(_CompiledStepBase):
    """Adamax wrapper — Adam with L∞ second-moment estimate.

    Uses :func:`lucid.maximum` for the element-wise max between
    ``β2 * u_{t-1}`` and ``|g|`` — that op must have a real-emit
    emitter on the compile side (it does — see
    ``OpEmitters/core/Elementwise.mm``).
    """

    def __init__(self, opt: Optimizer) -> None:
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
        for i, m in enumerate(self._m_buf):
            register("m", i, m)
        for i, u in enumerate(self._u_buf):
            register("u", i, u)

    def _register_scalars(self, register):
        # ``eff_lr = lr / (1 - beta1^t)`` — refreshed each step.
        dt = self._params[0].dtype
        dev = self._params[0].device
        eff_lr = _zero_scalar(dt, dev)
        register("scalar", 0, eff_lr)
        scalars = {"eff_lr": eff_lr}
        self._scalar_slots = scalars
        return scalars

    def _refresh_scalars(self) -> None:
        import lucid as _lucid

        self._t += 1
        eff = self._lr / (1.0 - self._beta1**self._t)
        dt = self._params[0].dtype
        dev = self._params[0].device
        self._scalar_slots["eff_lr"].copy_(_lucid.tensor(eff, dtype=dt, device=dev))

    def _trace_update(self, all_inputs, grads, scalars):
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
        return list(self._params) + list(self._m_buf) + list(self._u_buf)


# ── NAdam ───────────────────────────────────────────────────────────


class _CompiledNAdam(_CompiledStepBase):
    """NAdam wrapper — Nesterov-Adam with the momentum decay schedule
    used by Lucid's eager backend.

    Per-step scalar feeds, all CPU-computed:

    * ``c1``      = ``lr * (1 - μ_t)         / (1 - Π_{k≤t} μ_k)``
    * ``c2``      = ``lr * μ_{t+1}          / (1 - Π_{k≤t} μ_k · μ_{t+1})``
    * ``inv_bc2`` = ``1 / (1 - β2^t)``

    where ``μ_t = β1 · (1 - 0.5 · 0.96^(t · momentum_decay))``.  The
    momentum_decay default is ``0.004`` — the public Python ``NAdam``
    constructor doesn't expose it; we mirror the C++ default verbatim.
    """

    # Lucid's eager NAdam fixes momentum_decay at the C++ default; the
    # Python constructor doesn't accept it.  Mirror that here.
    _MOMENTUM_DECAY: float = 0.004

    def __init__(self, opt: Optimizer) -> None:
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
        for i, m in enumerate(self._m_buf):
            register("m", i, m)
        for i, v in enumerate(self._v_buf):
            register("v", i, v)

    def _register_scalars(self, register):
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
        return list(self._params) + list(self._m_buf) + list(self._v_buf)
