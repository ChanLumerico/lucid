"""
Optimizer base class.
"""

import operator
import warnings
from typing import Any, ClassVar, Iterable, Protocol, cast, override

import lucid as _lucid
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid.nn.parameter import Parameter

from lucid._C import engine as _C_engine

_INTEGER_DTYPES = frozenset(
    {
        _C_engine.Dtype.I8,
        _C_engine.Dtype.I16,
        _C_engine.Dtype.I32,
        _C_engine.Dtype.I64,
        _C_engine.Dtype.Bool,
    }
)


class _EngineRules(Protocol):
    """An engine optimizer class, as the base class reads its rules."""

    @staticmethod
    def check_hyperparams(values: dict[str, float]) -> None: ...


class _TunableEngine(Protocol):
    """An engine optimizer, as the base class hands it edited hyper-parameters."""

    def set_hyperparams(self, values: dict[str, float]) -> None: ...


class _GroupReader(Protocol):
    """Reads a class's ``_HYPERPARAMS`` out of a param group, as one tuple.

    A callable object rather than a function, so a reader stored on the
    class is not bound to the instances that call it.
    """

    def __call__(self, group: dict[str, object], /) -> Any: ...


class _OneKey:
    """Reads one param-group key as a 1-tuple.

    ``operator.itemgetter`` with a single key returns the bare value; this
    keeps the tuple every hyper-parameter reader returns.
    """

    __slots__ = ("_key",)

    def __init__(self, key: str) -> None:
        self._key = key

    def __call__(self, group: dict[str, object]) -> tuple[object, ...]:
        return (group[self._key],)


def _hyperparam_reader(
    keys: tuple[str, ...],
) -> _GroupReader | None:
    """Return a reader of ``keys`` out of a param group, as one tuple.

    ``operator.itemgetter`` does the reading in C, so comparing a group with
    the values last handed over costs one call and one tuple comparison.
    ``None`` when there are no keys to read.  Neither reader is a function,
    so stored on a class it is not bound to the instances that read it.
    """
    if not keys:
        return None
    if len(keys) == 1:
        return _OneKey(keys[0])
    return operator.itemgetter(*keys)


def _state_like(state: object, param: object) -> object | None:
    """Return optimizer state entry ``state`` ready for ``param``'s new impl.

    Both are engine ``TensorImpl`` objects.  A buffer shaped like the
    parameter (``exp_avg``, ``momentum_buffer`` ...) is cast to the
    parameter's dtype and moved to its device.  A per-parameter scalar — a
    0-d entry of a parameter that is not 0-d, such as ``step`` or NAdam's
    ``mu_product`` — and any integer entry (a count) pass unchanged: the
    engine reads them as numbers wherever they live, and ``step`` stays an
    integer.  ``None`` when a buffer's shape no longer matches: it belonged
    to a differently shaped parameter and cannot be reused.
    """
    src = cast(_C_engine.TensorImpl, state)
    dst = cast(_C_engine.TensorImpl, param)
    if src.dtype in _INTEGER_DTYPES or (not src.shape and dst.shape):
        return src
    if list(src.shape) != list(dst.shape):
        return None
    out: _C_engine.TensorImpl = src
    if out.dtype != dst.dtype:
        out = _C_engine.astype(out, dst.dtype)
    if out.device != dst.device:
        out = out.transfer_to_device(dst.device, False)
    return out


class Optimizer:
    """Base class for all optimizers.

    Every concrete optimizer must subclass ``Optimizer`` and implement
    :meth:`step`.  The base class manages parameter groups, provides
    :meth:`zero_grad`, and handles :meth:`state_dict` / :meth:`load_state_dict`
    checkpointing.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Either a flat iterable of :class:`~lucid.nn.Parameter` objects, or
        a list of *parameter-group dicts*.  Each group dict must contain at
        least a ``"params"`` key; any other keys override the corresponding
        entry in ``defaults`` for that group.
    defaults : dict
        Default hyperparameter values (e.g. ``{"lr": 1e-3, "weight_decay": 0}``).
        These are copied into each param group that does not override them.

    Attributes
    ----------
    param_groups : list of dict
        List of parameter groups.  Each entry is a dict with at least
        ``"params": list[Parameter]`` plus all hyperparameter keys from
        ``defaults`` (possibly overridden per group).
    defaults : dict
        The default hyperparameters passed at construction.
    state : dict
        Per-parameter optimizer state indexed by flat parameter index.
        Populated by :meth:`state_dict` and consumed by
        :meth:`load_state_dict`.

    Notes
    -----
    Every concrete ``step()`` override is automatically wrapped at class
    definition time (via :meth:`__init_subclass__`) to flush all parameter
    tensors to the Metal GPU after each update.  This ensures that
    parameter values are materialised before the next forward pass.

    Each step reads the hyper-parameters from ``param_groups``, so a value
    assigned there takes effect at the next :meth:`step`, the way a
    hand-written warm-up or fine-tuning schedule expects:

    .. code-block:: python

        for group in optimizer.param_groups:
            group["lr"] = 1e-4

    Each group is checked against the optimizer's rules when it is added and
    again when an edited group is next used.  A value the constructor would
    reject (a negative ``lr``, a beta outside ``[0, 1)``) raises
    ``InvalidArgument``, a ``ValueError``, and nothing of that group is
    applied.

    Parameter groups allow different hyperparameters per group:

    .. code-block:: python

        optimizer = optim.SGD(
            [
                {"params": model.backbone.parameters(), "lr": 1e-3},
                {"params": model.head.parameters(),     "lr": 1e-2},
            ],
            lr=1e-4,   # default, overridden per group above
        )

    Examples
    --------
    ``Optimizer`` is not used directly.  Use a concrete subclass:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2)
    >>> loss = model(lucid.randn(1, 4)).sum()
    >>> import lucid.optim as optim
    >>> optimizer = optim.Adam(model.parameters(), lr=1e-3)
    >>> optimizer.zero_grad()
    >>> loss.backward()
    >>> optimizer.step()
    """

    # Class-level flag controlling whether ``step()`` forces an eval of every
    # parameter tensor after the update.  Historical default was True — meant
    # as a "safety flush" for Metal params — but profiling on M4 Max
    # (May 2026) showed it costs ~37% of training throughput at small batch
    # sizes because it shatters the MLX lazy-graph pipeline that would
    # otherwise chain forward → backward → step into one submission.
    #
    # The reference framework's optimizers do NOT force-eval after step();
    # they rely on the next ``.item()`` / ``.cpu()`` / numpy bridge call
    # to materialise.
    # We mirror that default — set this class attribute to True (per-instance
    # via ``optimizer.AUTO_EVAL_AFTER_STEP = True`` or globally on a
    # subclass) only when you need step() to act as a synchronisation point.
    AUTO_EVAL_AFTER_STEP: ClassVar[bool] = False

    # The hyper-parameter table: the ``param_groups`` keys this optimizer's
    # rules and engine take.  It drives both the checks of a group (when it
    # is added, and when it is next used after an edit) and the hand-over of
    # edited values to the group's engine.  Empty for an optimizer with
    # nothing to check.
    _HYPERPARAMS: ClassVar[tuple[str, ...]] = ()

    # The engine optimizer class whose ``check_hyperparams`` states the
    # rules — the same checks its constructor and ``set_hyperparams`` run.
    # ``None`` for an optimizer that runs in Python; that one overrides
    # :meth:`_check_hyperparams`.
    _ENGINE_RULES: ClassVar[_EngineRules | None] = None

    # The reader of ``_HYPERPARAMS`` out of a group, derived once per class
    # by :meth:`__init_subclass__`.  On the class rather than the instance,
    # so an optimizer pickles without it.
    _read_hparams: ClassVar[_GroupReader | None] = None

    @override
    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap every concrete ``step()`` *only when* ``AUTO_EVAL_AFTER_STEP``
        is set on the subclass.

        3.2.2 change: the default (``AUTO_EVAL_AFTER_STEP = False``) keeps
        the user's overridden ``step()`` as-is — no closure layer, no
        per-call ``type(self).AUTO_EVAL_AFTER_STEP`` branch.  The wrapper
        existed historically to support the optional eager flush, but
        when the flag is False (default since 3.0.3) the wrapper just
        adds dispatch overhead with no semantic effect.

        To opt back into the synchronous flush, set
        ``AUTO_EVAL_AFTER_STEP = True`` on the subclass *before* it is
        first declared; the wrapper is then installed.  Toggling the
        flag at runtime is still possible — see ``step()``'s post-hoc
        wrapping in subclasses that need it.

        Also derives the class's ``_HYPERPARAMS`` reader.
        """
        super().__init_subclass__(**kwargs)
        cls._read_hparams = _hyperparam_reader(cls._HYPERPARAMS)
        if not cls.AUTO_EVAL_AFTER_STEP:
            # Default path: no wrapper, user's step() runs raw.
            return
        if "step" in cls.__dict__:
            _orig = cls.__dict__["step"]

            def _step_with_eval(
                self: Optimizer, closure: _OptimizerClosure = None
            ) -> Tensor | None:
                result = _orig(self, closure)
                if type(self).AUTO_EVAL_AFTER_STEP:
                    self._metal_eval_params()
                return cast(Tensor | None, result)

            _step_with_eval.__name__ = "step"
            _step_with_eval.__doc__ = _orig.__doc__
            cls.step = _step_with_eval  # type: ignore[method-assign]

    def _metal_eval_params(self) -> None:
        """Flush all parameter tensors via C++ eval_tensors() — no mlx import.

        Called automatically after every optimizer step when
        ``AUTO_EVAL_AFTER_STEP`` is True.  GPU tensors are flushed in one
        C++ call; CPU tensors are ignored.  Most training loops never need
        this — the lazy graph evaluates implicitly on the next
        ``loss.item()`` / ``.cpu()`` / numpy bridge call.
        """

        impls: list[object] = [
            p._impl
            for group in self.param_groups
            for p in group["params"]  # type: ignore[attr-defined]
            if isinstance(p, Parameter)
        ]
        if impls:
            _C_engine.eval_tensors(impls)  # type: ignore[arg-type]

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        defaults: dict[str, object],
    ) -> None:
        """Initialise the Optimizer.  See the class docstring for parameter semantics."""
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            param_groups: list[dict[str, object]] = list(params)
        else:
            param_groups = [{"params": list(params)}]

        # An optimiser over nothing steps nothing, and the loss still goes
        # down because the rest of the model is learning — so the usual way
        # to find out is a model that trains worse than it should for
        # reasons nobody can name.  The commonest cause is a lazy layer
        # whose parameters did not exist yet when ``parameters()`` was read;
        # ``Module.parameters`` warns about that case separately, and this
        # catches it when the whole model was lazy.
        if not any(group.get("params") for group in param_groups):
            raise ValueError(
                "optimizer got an empty parameter list.  If the model has "
                "lazy layers, run one forward pass before building the "
                "optimizer so their parameters exist."
            )

        self.param_groups: list[dict[str, object]] = []
        self._engines: list[object] = []
        # Parallel to ``_engines``: the parameters each engine optimizer was
        # built over, and the impls they held at that moment.
        self._engine_binds: list[tuple[list[Parameter], list[object]]] = []
        self._engine_state_reset_warned: bool = False
        self._engines_built: bool = False
        # Parallel to ``param_groups``: each group's hyper-parameters as they
        # were last checked or handed to its engine.
        self._group_hparams: list[tuple[object, ...]] = []
        self.state: dict[int, dict[str, object]] = {}
        self.defaults: dict[str, object] = defaults

        for group in param_groups:
            self.add_param_group(group)

    @property
    def _engine_optims(self) -> list[object]:
        """The engine optimizers, built on first use rather than in ``__init__``.

        Each engine optimizer captures ``TensorImpl`` pointers when it is
        constructed and steps those objects directly.  Building them in
        ``__init__`` therefore freezes whatever impls the parameters
        happened to hold at that moment — and a lazy layer replaces its
        impls at the first forward, which is *after* the optimizer is
        usually built.  The engine kept stepping the old ones, so the
        layer never moved and nothing said why.

        Deferring to first use costs nothing (the first ``step`` is after
        the first forward by construction) and fixes the general case,
        not only lazy layers: any parameter whose impl is replaced
        between construction and the first step is now picked up.

        Impls are also replaced *after* the first step — ``module.to()``,
        ``.double()``, ``.half()`` and any other conversion through
        ``Module._apply`` give each parameter a new buffer under the same
        ``Parameter`` object.  An engine built before that kept stepping
        the old buffer, so the parameter silently stopped training.  Every
        use therefore re-checks each group's impls against the ones its
        engine holds and rebuilds a group's engine when they differ,
        carrying its state across (see :meth:`_rebuild_engine_optim`).
        """
        if not self._engines_built:
            # Every group is checked before anything is built, so a group
            # edited to a rejected value fails the step with no engine half
            # made.  The engines are then built from these same values.
            hparams = [self._hyperparams_of(group) for group in self.param_groups]
            for values in hparams:
                self._check_hyperparams(self._hyperparam_dict(values))
            # Set first: ``_append_engine_optim`` appends through this
            # same property, and a re-entrant build would recurse.
            self._engines_built = True
            for group in self.param_groups:
                self._bind_engine_optim(group)
            self._group_hparams = hparams
        else:
            self._rebind_replaced_params()
            self._sync_group_hyperparams()
        return self._engines

    def add_param_group(self, group: dict[str, object]) -> None:
        """Add a parameter group, creating one new engine optimizer for it.

        The group's hyper-parameters are checked first, against the rules
        the optimizer's constructor applies; a group that breaks one raises
        ``InvalidArgument`` and is not added.
        """
        merged: dict[str, object] = {**self.defaults, **group}
        merged["params"] = list(merged["params"])  # type: ignore[call-overload]
        values: tuple[object, ...] = self._hyperparams_of(merged)
        self._check_hyperparams(self._hyperparam_dict(values))
        self.param_groups.append(merged)
        self._group_hparams.append(values)
        # Before the build, the group is simply on the list the build reads;
        # after it, the new group needs an engine of its own right now.
        if self._engines_built:
            self._bind_engine_optim(merged)

    # ── hyper-parameters ──────────────────────────────────────────────────────
    #
    # ``_HYPERPARAMS`` names the group keys; the engine class (or a Python
    # optimizer's override of ``_check_hyperparams``) holds the rules.  A
    # group is checked when it is added, and every use of the engines
    # compares each group with the values last handed over — one tuple
    # comparison per group — and hands over only a group that changed.

    def _hyperparams_of(self, group: dict[str, object]) -> tuple[object, ...]:
        """Return ``group``'s hyper-parameters in ``_HYPERPARAMS`` order."""
        read = self._read_hparams
        return () if read is None else tuple(read(group))

    def _hyperparam_dict(self, values: tuple[object, ...]) -> dict[str, Any]:
        """Name ``values`` by ``_HYPERPARAMS``, as the engine bindings take them."""
        return dict(zip(self._HYPERPARAMS, values))

    @classmethod
    def _check_hyperparams(cls, values: dict[str, Any]) -> None:
        """Raise ``InvalidArgument`` when a group's hyper-parameters break a rule.

        The engine class states the rules, so a group is held to exactly the
        checks its engine's constructor and ``set_hyperparams`` run.  An
        optimizer that runs in Python overrides this with its own.

        Parameters
        ----------
        values : dict
            The group's ``_HYPERPARAMS`` entries, by name.  Empty for an
            optimizer without a table, which has nothing to check.
        """
        if values and cls._ENGINE_RULES is not None:
            cls._ENGINE_RULES.check_hyperparams(values)

    def _sync_group_hyperparams(self) -> None:
        """Hand every edited group's hyper-parameters to where they apply.

        An engine optimizer receives them through ``set_hyperparams``, which
        checks the whole set before applying any of it; an optimizer without
        an engine, which reads its groups as it steps, has them checked.  A
        group equal to what was last handed over is skipped, so a step after
        no edit makes no binding call.  A rejected group raises and stays
        marked as changed, so every later step raises until it is fixed.
        """
        read = self._read_hparams
        if read is None:
            return
        seen = self._group_hparams
        for idx, (group, last) in enumerate(zip(self.param_groups, seen)):
            values = read(group)
            if values == last:
                continue
            self._apply_hyperparams(idx, self._hyperparam_dict(tuple(values)))
            seen[idx] = tuple(values)

    def _apply_hyperparams(self, idx: int, values: dict[str, Any]) -> None:
        """Hand group ``idx``'s edited hyper-parameters to its engine.

        A group without an engine (not built yet, or an optimizer that runs
        in Python) has them checked instead.
        """
        engines = self._engines
        engine = engines[idx] if idx < len(engines) else None
        if engine is None:
            self._check_hyperparams(values)
        else:
            cast(_TunableEngine, engine).set_hyperparams(values)

    @staticmethod
    def _group_binding(
        group: dict[str, object],
    ) -> tuple[list[Parameter], list[object]]:
        """Snapshot a group's parameters and the impls they hold right now."""
        params: list[Parameter] = list(group["params"])  # type: ignore[call-overload]
        return params, [p._impl for p in params]

    def _bind_engine_optim(self, group: dict[str, object]) -> None:
        """Build a group's engine optimizer and remember what it was built over."""
        built: int = len(self._engines)
        self._append_engine_optim(group)
        if len(self._engines) > built:
            self._engine_binds.append(self._group_binding(group))

    def _build_engine_optim(self, group: dict[str, object]) -> object | None:
        """Build one engine optimizer for ``group`` without appending it.

        ``_append_engine_optim`` is the per-optimizer hook and appends to
        ``_engines``; it runs against a scratch list here so the result can
        replace a group's existing engine in place.
        """
        live: list[object] = self._engines
        self._engines = []
        try:
            self._append_engine_optim(group)
            built: list[object] = self._engines
        finally:
            self._engines = live
        return built[-1] if built else None

    def _rebind_replaced_params(self) -> None:
        """Rebuild the engine of every group whose parameter impls changed.

        One identity comparison per parameter, so the common case — nothing
        replaced — stays a cheap scan.  An optimizer without engines (LBFGS,
        SparseAdam) has no bindings and is skipped.
        """
        binds = self._engine_binds
        if len(binds) != len(self.param_groups):
            return
        impl_of = operator.attrgetter("_impl")
        for idx, group in enumerate(self.param_groups):
            params: list[Parameter] = group["params"]  # type: ignore[assignment]
            impls: list[object] = binds[idx][1]
            if len(params) == len(impls) and all(
                map(operator.is_, map(impl_of, params), impls)
            ):
                continue
            self._rebuild_engine_optim(idx, group)

    def _rebuild_engine_optim(self, idx: int, group: dict[str, object]) -> None:
        """Replace group ``idx``'s engine with one over its current impls.

        The engine optimizer binds ``TensorImpl`` pointers at construction
        and has no way to re-point them, so a new one is built from the
        group (whose hyperparameters are the live ones — schedulers write
        there) and the old one's state is carried over through
        ``state_buffers`` / ``load_state_buffers``: entries are matched by
        ``Parameter`` identity, buffers are cast to the new impl's dtype and
        moved to its device, and per-parameter scalars (``step``,
        ``mu_product``) pass as they are (see :func:`_state_like`).

        Moving the state is a deliberate superset of the reference
        framework, which leaves state where it was after ``module.to()``
        and then fails at the next step for most optimizers.  State
        following the parameter is its own ``load_state_dict`` policy too.
        A buffer whose shape no longer matches its parameter is dropped.
        """
        old: object = self._engines[idx]
        old_params: list[Parameter] = self._engine_binds[idx][0]
        fresh: object | None = self._build_engine_optim(group)
        self._engines[idx] = fresh
        self._engine_binds[idx] = self._group_binding(group)
        # Built from the group as it is now, edits included.
        self._group_hparams[idx] = self._hyperparams_of(group)
        if old is None or fresh is None:
            return

        buffers: list[tuple[str, list[object | None]]] = old.state_buffers()  # type: ignore[attr-defined]
        if not buffers:
            # Nothing exported: no state yet, or an engine that keeps state
            # it cannot hand over — which only this optimizer can tell.
            if self._engine_holds_state(group):
                self._warn_engine_state_reset()
            return

        slot_of: dict[int, int] = {id(p): j for j, p in enumerate(old_params)}
        params: list[Parameter] = group["params"]  # type: ignore[assignment]
        carried: list[tuple[str, list[object | None]]] = []
        for name, tensors in buffers:
            moved: list[object | None] = []
            for p in params:
                j: int | None = slot_of.get(id(p))
                state = tensors[j] if j is not None and j < len(tensors) else None
                moved.append(None if state is None else _state_like(state, p._impl))
            carried.append((name, moved))
        if any(t is not None for _, moved in carried for t in moved):
            fresh.load_state_buffers(carried)  # type: ignore[attr-defined]

    def _engine_holds_state(self, group: dict[str, object]) -> bool:
        """Whether ``group``'s engine may hold state while exporting none.

        Read only when a rebuilt engine's ``state_buffers()`` came back
        empty, to tell "nothing to carry" from "state that could not be
        carried".  Every Lucid engine but SGD names its buffers even before
        it has any, so an empty export from one of them would be an engine
        without the hooks; the default assumes that.  SGD overrides it: its
        engine exports nothing until a momentum buffer exists.
        """
        return True

    def _warn_engine_state_reset(self) -> None:
        """Say once that a rebuilt engine started over from empty state."""
        if self._engine_state_reset_warned:
            return
        self._engine_state_reset_warned = True
        warnings.warn(
            f"{type(self).__name__}: a parameter's buffer was replaced after "
            "the optimizer had stepped (module.to(), .double(), .half() or "
            "another conversion), and this optimizer cannot carry its "
            "per-parameter state over, so that state restarted from empty.  "
            "Convert the model before building the optimizer to keep it.",
            RuntimeWarning,
            # warn ← rebuild ← rebind ← _engine_optims ← step() ← caller
            stacklevel=6,
        )

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        """Create and append one engine optimizer for a single param group.

        Override in subclasses. Base is a no-op so Optimizer can be subclassed
        without a C++ backend.
        """
        pass

    def _sync_hyperparams(self) -> None:
        """Hand edited ``param_groups`` hyper-parameters over now.

        Preserves all accumulated optimizer state (e.g. Adam first/second
        moments).  Every step does this anyway (see
        :meth:`_sync_group_hyperparams`); LR schedulers call it so the rate
        they write is in the engine as soon as they return, and an optimizer
        that runs in Python calls it at the top of its ``step``.

        A no-op for an engine optimizer before its engines exist — a
        scheduler constructed alongside the optimizer must not be what
        forces them into being, or the deferral above buys nothing.  The
        build reads ``param_groups``, which is where the new values already
        are.
        """
        if self._engines_built or self._ENGINE_RULES is None:
            self._sync_group_hyperparams()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients of all parameters."""
        for group in self.param_groups:
            for p in group["params"]:  # type: ignore[attr-defined]
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        # ``TensorImpl.zero_grad`` *resets* the gradient — it
                        # drops the buffer rather than filling it — so this
                        # branch used to be indistinguishable from the one
                        # above.  ``set_to_none=False`` exists precisely to
                        # keep an allocated zero buffer: gradient accumulation
                        # across micro-batches reads ``p.grad`` between steps,
                        # and a ``None`` there is an AttributeError at best and
                        # a silently skipped update wherever the code guards
                        # with ``if p.grad is not None``.
                        p.grad = _lucid.zeros_like(p)

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single optimization step.

        Subclasses must override this to update parameters from their
        current gradients.  Optionally accepts a *closure* that re-evaluates
        the model and returns the loss (required by some optimizers, e.g.
        LBFGS; ignored by most first-order methods).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.step() is not implemented. "
            "Subclasses of Optimizer must override step()."
        )

    # ── state_dict round-trip ─────────────────────────────────────────────────
    #
    # Format follows reference framework: ``param_groups`` mirrors the live
    # groups but each ``params`` entry is replaced with a list of integer
    # parameter ids; ``state`` is keyed by those same ids. Parameter tensors
    # themselves live on the model — saving them inside the optimizer would
    # double-checkpoint the weights and break partial restores.
    #
    # Engine optimizers (Adam, AdamW, SGD) expose their per-parameter mutable
    # state via the C++ ``state_buffers``/``load_state_buffers`` hooks; the base
    # class harvests those automatically below. Subclasses that own additional
    # Python-side state (e.g. LBFGS history) should override _save_state /
    # _load_state to round-trip it.

    def _save_state(self) -> dict[int, dict[str, object]]:
        """Snapshot per-parameter state. Default: pull from engine optimizers."""
        return self._save_engine_state()

    def _load_state(self, state: dict[int, dict[str, object]]) -> None:
        """Restore per-parameter state. Default: push back to engine optimizers."""
        self._load_engine_state(state)

    def _save_engine_state(self) -> dict[int, dict[str, object]]:
        """Snapshot every engine optimizer's per-parameter state.

        Output is keyed by flat parameter index.  Each entry holds one
        numpy array per entry the engine exports for that parameter, keyed
        by the reference framework's names: buffers shaped like the
        parameter (``exp_avg``, ``exp_avg_sq``, ``momentum_buffer`` ...)
        and per-parameter scalars as 0-d arrays — ``step``, the parameter's
        own step count (int64), and e.g. NAdam's ``mu_product``.  A
        parameter that has never been updated has no entry.

        Before the engines exist there is no state to save, and building
        them here would bind whatever impls the parameters hold now — a
        model converted afterwards then needs a rebuild at the first step.
        """
        if not self._engines_built:
            return {}
        out: dict[int, dict[str, object]] = {}
        flat_idx: int = 0
        for group, eng in zip(self.param_groups, self._engine_optims):
            params: list[Parameter] = group["params"]  # type: ignore[assignment]
            if eng is None:
                flat_idx += len(params)
                continue
            buffers: list[tuple[str, list[object]]] = eng.state_buffers()  # type: ignore[attr-defined]
            step_count: int = int(getattr(eng, "step_count", 0) or 0)
            for slot, _ in enumerate(params):
                snapshot: dict[str, object] = {}
                for name, tensors in buffers:
                    if slot < len(tensors) and tensors[slot] is not None:
                        # tensors[slot] is a TensorImpl — round-trip via numpy
                        # so the saved checkpoint is portable across processes
                        # (no shared C++ pointers across pickling).
                        import numpy as _np

                        snapshot[name] = _np.asarray(
                            tensors[slot].data_as_python()  # type: ignore[attr-defined]
                        ).copy()
                if step_count != 0 and "step" not in snapshot:
                    # An engine with one count per group, not per parameter.
                    snapshot["step"] = step_count
                if snapshot:
                    out[flat_idx + slot] = snapshot
            flat_idx += len(params)
        return out

    def _load_engine_state(self, state: dict[int, dict[str, object]]) -> None:
        """Push numpy-backed state entries back into each engine optimizer.

        Every entry, ``step`` included, goes back to its own parameter
        through ``load_state_buffers``.  A ``step`` that is a plain number
        is the older layout, one count for the whole group, and is set as
        that group's count.
        """
        if not state:
            return
        flat_idx: int = 0
        for group, eng in zip(self.param_groups, self._engine_optims):
            params: list[Parameter] = group["params"]  # type: ignore[assignment]
            if eng is None:
                flat_idx += len(params)
                continue
            # Collect per-buffer-name lists running parallel to params.
            by_name: dict[str, list[object | None]] = {}
            step_count: int = 0
            for slot, p in enumerate(params):
                snapshot: dict[str, object] = state.get(flat_idx + slot, {})
                for k, v in snapshot.items():
                    if k == "step" and isinstance(v, (int, float)):
                        # Folding per-parameter arrays to their maximum here
                        # gave every parameter the furthest one's count.
                        step_count = max(step_count, int(v))
                        continue
                    by_name.setdefault(k, [None] * len(params))
                    # Wrap as TensorImpl on the param's device so the engine
                    # can copy it back into its buffer slot.
                    by_name[k][slot] = _C_engine.TensorImpl(v, p._impl.device, False)
            if by_name:
                eng.load_state_buffers([(k, v) for k, v in by_name.items()])  # type: ignore[attr-defined]
            if step_count and hasattr(eng, "step_count"):
                eng.step_count = step_count
            flat_idx += len(params)

    def _param_id_map(self) -> dict[int, int]:
        """Map ``id(param)`` → flat integer index across all param groups."""
        out: dict[int, int] = {}
        idx: int = 0
        for group in self.param_groups:
            for p in group["params"]:  # type: ignore[attr-defined]
                out[id(p)] = idx
                idx += 1
        return out

    def state_dict(self) -> dict[str, object]:
        """Return a checkpointable snapshot of the optimizer.

        Mirrors reference framework's optimizer state_dict layout:

        - ``state``: ``{param_index: {key: value, ...}}`` — each parameter's
          optimizer state under the reference framework's keys: the engine's
          buffers and per-parameter scalars (``exp_avg``,
          ``momentum_buffer``, ``step`` ...) as numpy arrays, or the
          Python-side state of an optimizer that keeps its own (LBFGS).
        - ``param_groups``: list of group dicts; each group's ``params`` is a
          list of integer indices into the flat parameter list.
        """
        id_map: dict[int, int] = self._param_id_map()
        groups_out: list[dict[str, object]] = []
        for group in self.param_groups:
            g: dict[str, object] = {k: v for k, v in group.items() if k != "params"}
            g["params"] = [id_map[id(p)] for p in group["params"]]  # type: ignore[attr-defined]
            groups_out.append(g)
        self.state = self._save_state()
        return {"state": self.state, "param_groups": groups_out}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore from a state_dict produced by :meth:`state_dict`.

        Hyperparameters in ``param_groups`` are restored, and each
        parameter's state (returned from :meth:`_save_state`) is restored
        via :meth:`_load_state` — for the engine optimizers, every buffer
        and per-parameter ``step`` goes back into the engine.
        """
        loaded_groups: list[dict[str, object]] = state_dict["param_groups"]  # type: ignore[assignment]
        if len(loaded_groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state_dict has {len(loaded_groups)} param_groups but "
                f"optimizer has {len(self.param_groups)}"
            )
        for g_new, g_old in zip(self.param_groups, loaded_groups):
            for k, v in g_old.items():
                if k == "params":
                    continue
                g_new[k] = v
        loaded_state: dict[int, dict[str, object]] = state_dict.get("state", {})  # type: ignore[assignment]
        self.state = loaded_state
        self._load_state(loaded_state)
        self._sync_hyperparams()
