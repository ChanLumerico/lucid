"""
lucid.compile._compiled_module — Phase 1.4 CompiledModule.

Wraps a regular :class:`nn.Module` so that subsequent calls with the
same input signature reuse a single :class:`MPSGraphExecutable` rather
than re-dispatching through eager.  Mirrors the delegation pattern of
:class:`lucid.nn.functional.linear.FusedLinear` — the wrapper is
itself a :class:`nn.Module` and re-exposes the inner model's
parameters / state_dict / training mode, but does NOT register the
inner model under ``_modules`` (that would double-count parameters on
walks).

Phase 1.4 ships **forward-only graph caching**: the cached executable
captures the forward pass only, and the result tensor is returned
without a backward graph (``requires_grad=False``).  Training-step
compile — the autograd-integrated path that runs ``loss.backward()``
against a cached fwd+bwd executable — is the dedicated
:func:`lucid.compile.compiled_step` entrypoint until Phase 1.5 lands
the autograd-integrated :class:`CompiledStepBackward` node.

Acceptance gate (Plan §1.4):

  * 100 calls same signature → 1 compile + 99 cache hits.
  * Shape change → recompile.
  * Unsupported op → eager fallback with correct result + signature
    remembered in :class:`EagerFallbackSet` so we don't re-attempt.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from lucid._C import engine as _C_engine

from lucid.compile._fallback import EagerFallbackSet, run_eager
from lucid.compile._signature import CacheKey, signature_of
from lucid.nn.parameter import Parameter

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module


# Return-spec descriptors: small tags + nested children so that the
# executable's flat output list can be re-packed into the structure
# the user's forward returned.  ``("tensor", slot_index)`` is a leaf —
# replace with the executable output at ``slot_index``.  Other tags
# are container nodes whose children are themselves return-spec
# descriptors.
#
# Examples:
#   * ``return out`` (Tensor)               →  ("tensor", 0)
#   * ``return out, h_n`` (tuple)           →  ("tuple", [("tensor", 0), ("tensor", 1)])
#   * ``return {"loss": l, "logits": x}``  →  ("dict", [("loss", ("tensor", 0)),
#                                                       ("logits", ("tensor", 1))])


def _extract_return_impls(value: object, tracer: object) -> tuple[object, list[int]]:
    """Walk ``value`` and replace every Tensor with a slot descriptor.

    Returns ``(return_spec, explicit_outputs)`` where:

    * ``return_spec`` is the same structure as ``value`` but with each
      Tensor swapped for a ``("tensor", slot_index)`` leaf;
    * ``explicit_outputs`` is the flat ordered list of trace TensorIds
      that the slot indices reference.

    A Tensor whose impl wasn't observed during the trace is left in
    place (its impl appears as an external feed) — we still allocate a
    slot for it so the executable's flat output list lines up with the
    spec.  The Python compile harness handles that case by falling
    back to eager when the tensor isn't reachable in the graph.
    """
    from lucid._dispatch import _unwrap
    from lucid._tensor.tensor import Tensor

    flat_ids: list[int] = []

    def visit(v: object) -> object:
        """Recursive walk producing the spec tree + side-effecting ``flat_ids``.

        Tensors become ``("tensor", slot)`` leaves whose ``slot`` is
        an index into ``flat_ids`` (the per-output trace id); every
        other container type (tuple / list / dict / dataclass) becomes
        a tagged node carrying its children; scalars passthrough.
        """
        if isinstance(v, Tensor):
            impl = _unwrap(v)
            tid = tracer.lookup_id(impl)
            if tid is None:
                # Tensor that flows out untouched (external feed
                # returned as-is, or a host-precomputed factory the
                # builder skips).  Mark with a sentinel so the run
                # path knows to passthrough.
                return ("passthrough", v)
            slot = len(flat_ids)
            flat_ids.append(int(tid))
            return ("tensor", slot)
        if isinstance(v, tuple):
            return ("tuple", [visit(x) for x in v])
        if isinstance(v, list):
            return ("list", [visit(x) for x in v])
        if isinstance(v, Mapping):
            return ("dict", [(k, visit(x)) for k, x in v.items()])
        # Dataclass or BaseModelOutput-style record: walk its
        # __dict__-equivalent in declaration order.  We require
        # ``__dataclass_fields__`` to avoid blindly mutating arbitrary
        # objects.
        fields = getattr(v, "__dataclass_fields__", None)
        if fields is not None:
            kvs = []
            for fname in fields:
                kvs.append((fname, visit(getattr(v, fname))))
            return ("dataclass", type(v), kvs)
        # Anything else (None, scalar, str, …) passes through as-is.
        return ("scalar", v)

    spec = visit(value)
    return spec, flat_ids


def _repack_outputs(spec: object, outs_wrapped: list[object]) -> object:
    """Inverse of :func:`_extract_return_impls`.

    ``outs_wrapped`` is the flat list of wrapped Tensors returned by
    :func:`_C_engine.compile.run_executable` followed by :func:`_wrap`.
    Walks ``spec`` and substitutes each ``("tensor", slot)`` leaf with
    ``outs_wrapped[slot]``.  Reassembles tuples / lists / dicts /
    dataclasses in the original order.
    """
    tag = spec[0]
    if tag == "tensor":
        return outs_wrapped[spec[1]]
    if tag == "passthrough":
        return spec[1]
    if tag == "scalar":
        return spec[1]
    if tag == "tuple":
        return tuple(_repack_outputs(child, outs_wrapped) for child in spec[1])
    if tag == "list":
        return [_repack_outputs(child, outs_wrapped) for child in spec[1]]
    if tag == "dict":
        return {k: _repack_outputs(child, outs_wrapped) for (k, child) in spec[1]}
    if tag == "dataclass":
        cls = spec[1]
        kvs = spec[2]
        kwargs = {k: _repack_outputs(child, outs_wrapped) for (k, child) in kvs}
        return cls(**kwargs)
    raise RuntimeError(f"_repack_outputs: unknown spec tag {tag!r}")
    from lucid.nn.parameter import Parameter


__all__ = ["CompiledModule"]


@dataclass
class _CacheEntry:
    """One executable + the I/O plan needed to invoke it.

    The compiled MPSGraphExecutable is shape-fixed (placeholders are
    sized from the trace), so we record:
      * the ordered list of feed :type:`TensorId`s that the executable
        expects (``exe.input_ids``);
      * a mapping from those ids back to (a) external-feed Python
        objects pinned in the trace, or (b) the slot in ``args`` /
        ``kwargs`` that supplies the feed at call time.
    """

    exe: object  # _C_engine.compile.PyCompiledExecutable
    external_feeds: dict[int, object]  # TensorId -> TensorImpl
    # For each ``exe.input_ids[i]`` slot: either an integer index into
    # ``args`` (the positional input tensor at call time) OR ``None``
    # meaning "use the pinned external_feeds entry" (parameters,
    # constants).
    input_source: tuple[int | None, ...]
    # Structure descriptor used to re-pack the executable's flat output
    # list back into the shape the user's forward returned.  Each entry
    # is the path (sequence of keys / indices) to insert that output at
    # within a nested Python structure of tuples/lists/dicts.  When
    # ``return_spec`` is None, the executable returns a single Tensor.
    # See :func:`_extract_return_impls` for the format.
    return_spec: object = None
    # JSON-serialised TraceGraph used to build this executable —
    # cheap to compute at compile time and useful for ``graph_dump``
    # observability after the fact.  Kept as a string (not as a live
    # TraceGraph) so the entry can outlive the original C++ object.
    graph_json: str = ""
    n_hits: int = 0
    compile_ms: float = 0.0
    last_run_ms: float = 0.0


class CompiledModule:
    r"""User-facing wrapper returned by :func:`lucid.compile`.

    Wraps an :class:`nn.Module` (or any callable, via the internal
    :class:`_CallableModule` adapter) so the model's ``__call__`` is
    routed through a per-signature cache of compiled
    :class:`MPSGraphExecutable` objects.  Behaves like the underlying
    model for every non-call attribute (``parameters`` / ``state_dict``
    / ``train`` / ``eval`` / ``to`` / submodule attribute access), so
    optimizers, schedulers and checkpointing pipelines see the
    compiled module as a drop-in replacement.

    Lifecycle
    ---------
    On the first call with a given input signature
    ``(shape × dtype × device × training × param_fingerprint)`` the
    wrapper:

    1. Traces ``model(*args)`` under :func:`no_grad` while the
       :class:`Tracer` records every op into a :class:`TraceGraph`.
    2. Lowers the graph to a single :class:`MPSGraphExecutable` via
       ``compile_or_cached``.
    3. Builds an *input plan* mapping each placeholder in
       ``exe.input_ids`` to either a positional :class:`Tensor` slot
       from the call site or a pinned external feed (parameter /
       constant).
    4. Records a ``return_spec`` so the executable's flat output list
       can be re-packed into the user's original return structure
       (tuple, dataclass, dict, ...).

    On subsequent calls with the same signature step (1)–(3) are
    skipped and the cached executable runs directly.  Cache lookup
    cost is one ``__hash__`` + ``__eq__`` on the :class:`CacheKey`
    plus an integer-slot resolution per feed — typically O(μs).

    Two cache layers cooperate here:

    * **Python-side ``self._cache``** — unbounded ``dict`` keyed by
      :class:`CacheKey`, holds the :class:`_CacheEntry` + return spec
      + timing.  Dropped on :meth:`train` / :meth:`eval` / :meth:`to`
      / :meth:`load_state_dict` / :meth:`clear_cache`.
    * **C++ ``ExecutableCache::session()``** — process-global LRU
      bounded by ``LUCID_COMPILE_MAX_CACHE`` (default 32).  Holds
      one ``CompiledExecutable`` per *trace structure* so two
      :class:`CompiledModule` instances tracing the same op DAG
      share the underlying Metal kernels.  Cache key includes per-op
      attributes (since 2026-05-24), so ``dropout(p=0)`` and
      ``dropout(p=0.5)`` no longer collide.

    Attributes
    ----------
    model : nn.Module
        The wrapped model (read via the ``.model`` property).
    dynamic : bool
        ``True`` if the wrapper was constructed with the symbolic
        batch axis opt-in (Phase 1.6).  Forced to ``False`` until
        the MPSGraph SDK stabilises dynamic-shape lowering.
    training : bool
        Mirrors ``self._model.training``; flipping invalidates the
        cache (BN / Dropout codepaths diverge between modes).

    Notes
    -----
    * **Metal-only.**  Compile is meaningful only when both the model
      and its inputs live on the ``metal`` device.  On CPU every call
      silently routes through eager — the cache stays empty and
      ``eager_only`` grows.
    * **kwargs force eager.**  The current MPSGraph builder is
      positional-only; any call with non-empty ``kwargs`` falls back
      to :func:`run_eager` and the signature is blacklisted as
      ``eager_only`` so subsequent identical-shaped calls skip the
      recompile attempt.
    * **Dropout in training mode falls back per signature.**  See
      :file:`OpEmitters/nn/Dropout.mm` — the emitter returns nullptr
      when ``training=True and p>0`` to preserve dropout's
      step-to-step randomness (the stateless RNG path would apply
      the same mask every call).

    Examples
    --------
    Inference path::

        cm = lucid.compile(model.to('metal').eval())
        for batch in loader:
            y = cm(batch.to('metal'))

    Training step compile (forward only — backward stays eager)::

        cm = lucid.compile(model.to('metal'))
        opt = lucid.optim.Adam(cm.parameters(), lr=1e-3)
        for batch, target in loader:
            opt.zero_grad()
            loss = F.cross_entropy(cm(batch), target)
            loss.backward()
            opt.step()

    Fully fused (forward + backward + update in one executable)::

        cm = lucid.compile(model.to('metal'))
        opt = lucid.optim.Adam(cm.parameters(), lr=1e-3)
        for batch, target in loader:
            loss = cm.step(batch, target, loss_fn=F.cross_entropy)
            loss.backward()
            opt.step()

    See Also
    --------
    :func:`lucid.compile` : the user-facing constructor.
    :func:`fused_step` : single-executable fwd+bwd+update.
    :func:`compile_optimizer` : compile only the update step.
    """

    def __init__(self, model: object, *, dynamic: bool = False) -> None:
        """Wrap ``model`` for cached graph-capture execution.

        Parameters
        ----------
        model : nn.Module or _CallableModule
            The compute unit to compile.  Stored unwrapped so
            attribute access falls through to it.
        dynamic : bool, optional
            Opt-in to symbolic-batch-axis (Phase 1.6).  Currently
            forbidden — raises :class:`NotImplementedError` because
            the MPSGraph SDK rejects dynamic-shape lowering for
            conv-heavy graphs.  Default ``False``.

        Raises
        ------
        NotImplementedError
            When ``dynamic=True`` is passed.
        """
        # Phase 1.6 (symbolic batch axis): the C++ side (compile_or_cached
        # / compile_trace) already accepts ``dynamic_batch=True``.  Until
        # macOS 25 the MPSGraph SDK aborted on conv-heavy graphs (MLIR
        # pass manager) and drifted numerically on MLP placeholders, so
        # this surface stayed locked off.  Re-enabled 2026-05-25 behind
        # ``LUCID_COMPILE_DYNAMIC=1`` opt-in to characterize the macOS
        # 26 / MPSGraph SDK state — flip the default to ``True`` once
        # the model-zoo regression sweep is clean.
        import os as _os

        _dyn_opt_in = _os.environ.get(
            "LUCID_COMPILE_DYNAMIC",
            "0",
        ) in ("1", "true", "True")
        if dynamic and not _dyn_opt_in:
            raise NotImplementedError(
                "lucid.compile(..., dynamic=True) is gated behind "
                "LUCID_COMPILE_DYNAMIC=1 while macOS 26's MPSGraph "
                "symbolic-shape support is being characterised.  "
                "Set the env var to opt in; otherwise use a static "
                "shape and let the per-BS cache populate organically — "
                "the cache entry per shape is cheap because MPSGraph's "
                "own kernel cache is shared across them."
            )
        # CRITICAL: bypass nn.Module's __setattr__ tracking so the
        # inner model is NOT registered as a submodule (otherwise
        # parameters / state_dict double-count).
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_dynamic", bool(dynamic))
        object.__setattr__(self, "_cache", {})  # type: ignore[var-annotated]
        object.__setattr__(self, "_eager_only", EagerFallbackSet())
        object.__setattr__(self, "_call_counter", {})  # type: ignore[var-annotated]
        # Param fingerprint cache — invalidated by train()/eval()/to()/
        # load_state_dict() (those all call clear_cache).  Without this
        # signature_of would walk model.parameters() on every __call__,
        # which dominates the Python overhead for deep models.
        object.__setattr__(self, "_param_fp_cache", None)
        # Lazy training-step callables, keyed by ``id(loss_fn)``.  The
        # first ``cm.step(..., loss_fn=fn)`` call creates a make_step
        # callable and stashes it here; subsequent calls with the same
        # loss_fn re-use the cached fwd+bwd executable inside.  We
        # intentionally key by object identity, not value equality —
        # a fresh closure each call would defeat the cache, which the
        # docstring warns about.
        object.__setattr__(self, "_step_callables", {})  # type: ignore[var-annotated]

    # ── Delegation surface ────────────────────────────────────────

    @property
    def model(self) -> Module:
        """The wrapped :class:`nn.Module` (or duck-typed callable wrapper)."""

        return self._model

    @property
    def dynamic(self) -> bool:
        """Whether the compile path treats batch dim as symbolic (Phase 1.6)."""

        return self._dynamic

    @property
    def training(self) -> bool:
        """Mirrors the inner model's ``training`` flag (cache-invalidating)."""
        return self._model.training

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Delegate to ``model.parameters`` so optimizers see the live tensors."""
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, recurse: bool = True) -> Iterator[tuple[str, Parameter]]:
        """Delegate to ``model.named_parameters`` (no recompile, read-only)."""
        return self._model.named_parameters(recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Delegate to ``model.buffers``; buffers are pinned trace constants."""
        return self._model.buffers(recurse=recurse)

    def state_dict(self, *args: object, **kwargs: object) -> Mapping[str, object]:
        """Delegate to ``model.state_dict`` — the cache holds no separate state."""
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args: object, **kwargs: object) -> object:
        """Forward to ``model.load_state_dict`` and **drop the cache**.

        Loading new weights doesn't change the graph topology, but
        captured tensor *identities* may shift (e.g. when the loader
        replaces parameter storage in-place); clearing the cache is
        the safe default so the next call traces against the new
        state.
        """
        # Loading new weights doesn't invalidate the graph itself —
        # only its captured constants would — but the safe default is
        # to drop the cache so the user never sees stale graph state.
        result = self._model.load_state_dict(*args, **kwargs)
        self.clear_cache()
        return result

    def train(self, mode: bool = True) -> CompiledModule:
        """Flip training mode; **clears the cache** since BN / Dropout diverge."""
        # train ↔ eval flips force different BN / Dropout codepaths
        # → recompile.  Clearing on every flip is the safe default.
        self._model.train(mode)
        self.clear_cache()
        return self

    def eval(self) -> CompiledModule:
        """Shorthand for ``self.train(False)``; also clears the cache."""
        return self.train(False)

    def to(self, *args: object, **kwargs: object) -> CompiledModule:
        """Forward to ``model.to`` and **drop the cache**.

        Placeholders inside the cached MPSGraph executables are
        device-locked, so any device or dtype move invalidates every
        entry.
        """
        # Device / dtype move ⇒ every cached executable is invalid
        # (placeholders are device-locked).  Drop the cache.
        self._model.to(*args, **kwargs)
        self.clear_cache()
        return self

    def __getattr__(self, name: str) -> object:
        """Forward unknown attribute lookups to the wrapped model.

        Lets ``compiled.fc1`` reach a regular submodule and
        ``compiled.named_modules()`` return the inner model's tree.
        The construction-time guard on ``_model`` prevents infinite
        recursion before ``__init__`` finishes.
        """
        # Any attribute we don't override is forwarded to the inner
        # model (so e.g. ``compiled.fc1`` reaches a regular submodule).
        # Guard against recursion during construction (``_model`` not
        # yet set).
        if name == "_model":
            raise AttributeError(name)
        inner = object.__getattribute__(self, "_model")
        return getattr(inner, name)

    # ── Cache introspection ──────────────────────────────────────

    def cache_info(self) -> dict[str, object]:
        """Snapshot the cache state for tests + telemetry.

        Examples
        --------
        >>> compiled = lucid.compile(model)
        >>> compiled(x1); compiled(x2)
        >>> info = compiled.cache_info()
        >>> info["entries"]                 # one slot per unique signature
        2

        See Also
        --------
        timing : per-signature wall-time breakdown.
        clear_cache : drop every cached entry.

        Returns
        -------
        dict
            Four fields:

            * ``entries`` (int) — number of cached executables.
            * ``keys`` (tuple[CacheKey]) — every signature that
              currently has a compiled executable.
            * ``eager_only`` (tuple[CacheKey]) — signatures that
              failed to compile and were blacklisted; future calls
              with these signatures route straight to eager.
            * ``n_calls`` (dict[CacheKey, int]) — per-signature call
              counter, useful for spotting hot signatures.
        """

        return {
            "entries": len(self._cache),
            "keys": tuple(self._cache.keys()),
            "eager_only": self._eager_only.snapshot(),
            "n_calls": dict(self._call_counter),
        }

    def timing(self) -> list[dict[str, object]]:
        """Per-signature compile + run cost breakdown.

        Returns
        -------
        list[dict]
            One entry per cached signature with fields ``key``
            (CacheKey), ``compile_ms`` (float — first-call compile
            cost), ``last_run_ms`` (float — most recent execution
            time), and ``n_hits`` (int — total runs of this entry).
            Useful to verify the cache amortises compile cost over
            subsequent calls.
        """

        return [
            {
                "key": key,
                "compile_ms": entry.compile_ms,
                "last_run_ms": entry.last_run_ms,
                "n_hits": entry.n_hits,
            }
            for key, entry in self._cache.items()
        ]

    def clear_cache(self) -> None:
        """Drop every cached executable + retry blacklist + call counter.

        Invalidates the param-fingerprint cache and the lazy training
        step callables too, so the next call re-fingerprints the
        model.  Called automatically by :meth:`train` / :meth:`eval`
        / :meth:`to` / :meth:`load_state_dict` — direct user calls
        are useful only when forcing a recompile after manual
        parameter buffer surgery.
        """

        self._cache.clear()
        self._eager_only.clear()
        self._call_counter.clear()
        # Invalidate the param fingerprint cache so the next call
        # recomputes it (param dtype/device may have changed).
        object.__setattr__(self, "_param_fp_cache", None)
        # Drop the lazy training-step callables too — their internal
        # caches reference parameter impls that may have moved.
        self._step_callables.clear()

    # ── Training-step compile surface ────────────────────────────

    def step(
        self,
        *args: object,
        loss_fn: object,
    ) -> object:
        """Run one compiled training step.

        Equivalent to::

            step_fn = make_step(self.model, loss_fn)
            loss = step_fn(*args)

        but the underlying ``make_step`` callable is cached on this
        :class:`CompiledModule` so repeated training-step calls with
        the same loss function share the fwd+bwd executable.  The
        returned scalar loss tensor carries a ``grad_fn`` — calling
        ``loss.backward()`` populates ``.grad`` on every parameter
        from the cached gradient outputs.

        Parameters
        ----------
        *args : Tensor
            ``(model_input, *extra_loss_inputs)`` — the same positional
            tuple ``make_step`` expects.  Typically ``(x, target)``.
        loss_fn : callable
            ``loss_fn(model_output, *extra_inputs) -> Tensor`` scalar
            loss.  Keyed by object identity for caching; pass the same
            function object across iterations to hit the cache.

        Returns
        -------
        Tensor
            Scalar loss with a working ``grad_fn``.

        Examples
        --------
        >>> cm = lucid.compile(model)
        >>> opt = lucid.optim.SGD(cm.parameters(), lr=0.1)
        >>> for x, t in loader:
        ...     opt.zero_grad()
        ...     loss = cm.step(x, t, loss_fn=F.cross_entropy)
        ...     loss.backward()
        ...     opt.step()
        """

        if loss_fn is None:
            raise TypeError(
                "CompiledModule.step(): loss_fn keyword argument is "
                "required.  Pass the scalar-returning callable that "
                "produces the training loss, e.g. "
                "``cm.step(x, target, loss_fn=F.cross_entropy)``."
            )

        key = id(loss_fn)
        step_callable = self._step_callables.get(key)
        if step_callable is None:
            from lucid.compile._step import make_step

            step_callable = make_step(self._model, loss_fn, dynamic=self._dynamic)
            self._step_callables[key] = step_callable

        return step_callable(*args)

    def graph_dump(self, key: CacheKey | None = None) -> str:
        """Return a JSON view of the captured trace(s).

        Without ``key``: a JSON list of ``{sig, graph}`` pairs covering
        every cached signature.  With ``key``: just the graph string
        for that one signature.

        Useful for offline debugging — pipe the result through
        :file:`tools/inspect_trace.py` (Phase 1.6 deliverable) or
        diff two consecutive compiles to see what changed.

        Parameters
        ----------
        key : CacheKey, optional
            When supplied, dump only the trace for that signature.
            When ``None`` (default), dump every cached signature in
            one combined JSON array.

        Returns
        -------
        str
            JSON-encoded trace dump.  Empty string when ``key`` was
            provided but no entry matches.

        See Also
        --------
        cache_info : structural snapshot of every cached signature.
        lucid.compile._trace_dump.trace_to_json : the underlying
            serialiser called per cache entry.
        """

        if key is not None:
            entry = self._cache.get(key)
            return entry.graph_json if entry is not None else ""

        import json as _json

        items = [
            {"sig_hash": hash(k), "graph": _json.loads(e.graph_json)}
            for k, e in self._cache.items()
        ]
        return _json.dumps(items, indent=2)

    # ── Call surface ─────────────────────────────────────────────

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Route the call through the per-signature executable cache.

        On cache hit: bind the input tensors to the cached
        executable's feed slots, run, and re-pack the flat output
        list into the user's return structure (Tensor / tuple / dict
        / dataclass).  On miss: trace, compile, store, run — and on
        compile failure mark the signature ``eager_only`` so future
        calls skip the recompile attempt.

        Parameters
        ----------
        *args, **kwargs
            Whatever the wrapped model's ``forward`` accepts.  Only
            positional :class:`Tensor` arguments become executable
            feeds; non-tensor positionals (and any keyword arguments)
            force the call to eager (today's compile pipeline doesn't
            bind kwargs by name).

        Returns
        -------
        object
            Same structure the underlying ``model(*args, **kwargs)``
            produces — Tensor, tuple, dataclass, etc.
        """
        # Hashable signature.  If the signature itself is un-hashable
        # (e.g. an exotic input that we can't fingerprint), fall back
        # to eager without poisoning the cache.
        #
        # Perf: cache the param fingerprint so we don't walk
        # ``model.parameters()`` on every call.  Cache is invalidated
        # by train()/eval()/to()/load_state_dict() via clear_cache.
        try:
            fp = self._param_fp_cache
            if fp is None:
                from lucid.compile._signature import _param_fingerprint

                fp = _param_fingerprint(self._model)
                object.__setattr__(self, "_param_fp_cache", fp)
            key = signature_of(
                self._model,
                args,
                kwargs,
                dynamic=self._dynamic,
                param_fingerprint=fp,
            )
        except TypeError, AttributeError:
            return run_eager(self._model, args, kwargs)

        self._call_counter[key] = self._call_counter.get(key, 0) + 1

        if key in self._eager_only:
            return run_eager(self._model, args, kwargs)

        entry = self._cache.get(key)
        if entry is None:
            entry = self._compile_for(key, args, kwargs)
            if entry is None:
                # Compile aborted — remember this signature is
                # eager-only and route to eager.
                self._eager_only.add(key)
                return run_eager(self._model, args, kwargs)
            self._cache[key] = entry
        return self._run(entry, args, kwargs)

    # ── Internals ────────────────────────────────────────────────

    def _compile_for(
        self,
        key: CacheKey,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> _CacheEntry | None:
        """Trace + compile + wrap one signature into a :class:`_CacheEntry`.

        Parameters
        ----------
        key : CacheKey
            The signature key for this call (used only for the
            return-spec record; not consulted internally).
        args, kwargs : tuple, dict
            The user's call arguments to feed through the trace.

        Returns
        -------
        _CacheEntry or None
            ``None`` on compile failure or any abort condition
            (kwarg-bound tensors, empty trace, unresolvable feed,
            backend rejection); the caller marks the signature as
            ``eager_only`` and routes to eager.
        """
        # Local imports keep ``lucid.compile`` import-cheap.
        from lucid._dispatch import _unwrap
        from lucid._tensor.tensor import Tensor
        from lucid.autograd._grad_mode import no_grad
        from lucid.compile import _tracing

        if kwargs:
            # The MPSGraph builder is positional-only.  Compose kwargs
            # into the call as-is (the trace records them) but the
            # feed-binding logic below indexes args by integer slot
            # only.  Fall back to eager when keyword-bound tensor
            # inputs are present until we add kw-aware feed binding.
            return None

        # Trace under no_grad — kernel hooks fire before the GradMode
        # short-circuit so the captured DAG is identical to the
        # autograd-on path, minus the (unused) backward wiring.
        t_compile_start = time.perf_counter()
        with no_grad():
            with _tracing() as tracer:
                return_value = self._model(*args, **kwargs)

        graph = tracer.graph
        if not graph.ops:
            return None

        ext = tracer.external_feeds

        # Extract the user's return value structure into:
        #   * return_spec: a tree that mirrors ``return_value`` but
        #     with every Tensor swapped for a slot index;
        #   * explicit_outputs: ordered list of trace ids the executable
        #     must expose so the slot indices line up.
        # Falling back to eager is safer than guessing if return_value
        # contains an un-traceable type, but the helper tolerates
        # arbitrary scalars / strings / Nones via the "scalar" leaf.
        return_spec, explicit_outputs = _extract_return_impls(return_value, tracer)

        # Phase 1.6 dynamic-batch path: tell the builder which feeds
        # are model parameters (so their first dim stays static) and
        # bypass the C++ session cache (the Python cache here holds
        # the dynamic-aware key).
        param_ids: list[int] = []
        if self._dynamic:
            p_impls = [_unwrap(p) for p in self._model.parameters()]
            id_to_tid = {id(impl): tid for tid, impl in ext.items()}
            for p_impl in p_impls:
                tid = id_to_tid.get(id(p_impl))
                if tid is not None:
                    param_ids.append(tid)

        try:
            exe = _C_engine.compile.compile_or_cached(
                graph, ext, self._dynamic, param_ids, explicit_outputs
            )
        except RuntimeError:
            return None
        if exe is None:
            return None

        # Figure out, for each feed in exe.input_ids, whether it comes
        # from a positional arg (Tensor in ``args``) or from a pinned
        # parameter / constant.  Index by ``TensorImpl is`` identity.
        positional_impls: dict[int, int] = {}
        for i, a in enumerate(args):
            if isinstance(a, Tensor):
                positional_impls[id(_unwrap(a))] = i

        input_source: list[int | None] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                # Should not happen — the builder only emits placeholder
                # ids that are in the external_feeds set.  Treat as a
                # compile failure.
                return None
            slot = positional_impls.get(id(impl))
            input_source.append(slot)

        from lucid.compile._trace_dump import trace_to_json

        entry = _CacheEntry(
            exe=exe,
            external_feeds=dict(ext),
            input_source=tuple(input_source),
            return_spec=return_spec,
            graph_json=trace_to_json(graph),
            compile_ms=(time.perf_counter() - t_compile_start) * 1000.0,
        )
        return entry

    def _run(
        self,
        entry: _CacheEntry,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """Execute a cached entry and re-pack the output into the user's structure.

        Parameters
        ----------
        entry : _CacheEntry
            The cached executable + feed plan returned by
            :meth:`_compile_for`.
        args, kwargs : tuple, dict
            The user's call arguments; positional :class:`Tensor`
            slots become the per-call feeds, pinned external feeds
            are bound from ``entry.external_feeds``.

        Returns
        -------
        object
            The user's return structure, with each Tensor leaf
            substituted by the matching output of the executable.
            Falls back to :func:`run_eager` if any positional slot
            isn't a Tensor at this call site.
        """
        from lucid._dispatch import _unwrap, _wrap
        from lucid._tensor.tensor import Tensor

        # Build the feed list in exe.input_ids order.
        feed_impls: list[object] = []
        for tid, src in zip(entry.exe.input_ids, entry.input_source):
            if src is None:
                # Pinned parameter / constant — use the trace-time impl.
                feed_impls.append(entry.external_feeds[tid])
            else:
                # Fresh input tensor at this call site.
                arg = args[src]
                if not isinstance(arg, Tensor):
                    # User changed the argument type between calls.
                    # Treat as a compile failure for this call.
                    return run_eager(self._model, args, kwargs)
                feed_impls.append(_unwrap(arg))

        t0 = time.perf_counter()
        outs = _C_engine.compile.run_executable(entry.exe, feed_impls)
        entry.last_run_ms = (time.perf_counter() - t0) * 1000.0
        entry.n_hits += 1

        # If the trace recorded the user's return structure, re-pack the
        # flat outs list into that shape (single Tensor, tuple, dict,
        # dataclass, …).  Falls back to the legacy single/tuple
        # heuristic if the spec is missing (older cached entries).
        outs_wrapped = [_wrap(o) for o in outs]
        if entry.return_spec is not None:
            return _repack_outputs(entry.return_spec, outs_wrapped)

        # Legacy: single forward output → plain Tensor.  Multi-output →
        # tuple.  No grad_fn — the compiled forward graph is its own
        # ground truth.
        if not outs_wrapped:
            return None
        if len(outs_wrapped) == 1:
            return outs_wrapped[0]
        return tuple(outs_wrapped)
