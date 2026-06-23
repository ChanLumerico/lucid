"""
lucid.compile._entry.decode_step — compiled single-token decode for StaticCache.

``compiled_decode_step(forward_fn, static_cache)`` returns a ``step(input_ids,
cache_position)`` callable that compiles the decode forward ONCE into a reused
:class:`MPSGraphExecutable` and rolls the cache forward in place.

Why it compiles where :class:`DynamicCache` does not
----------------------------------------------------
``DynamicCache`` grows its K/V by concatenation, so the tensor shape changes
every token and ``lucid.compile``'s shape-keyed signature forces a recompile per
step.  :class:`StaticCache` writes each token into a fixed
``(B, H, max_cache_len, D)`` buffer at a runtime ``cache_position`` via the
traceable :func:`lucid.index_copy`, so the signature is constant and one
executable serves every step (the ``cache_position`` is a fixed-shape ``(T,)``
int64 runtime feed, not a Python int — an int would be baked into the signature
and recompile every step).

Buffer write-back
-----------------
The ``index_copy`` write produces a NEW buffer tensor inside the executable; it
is surfaced as an explicit output (alongside the model output) and ``copy_``'d
back into the live ``StaticCache`` buffer after each run — the same shape as the
BatchNorm running-stats write-back, but forward-only (generation is ``no_grad``).
Because the live buffer is also the executable's pinned feed, the next step reads
the rolled-forward value.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast, final

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.utils.cache import StaticCache

__all__ = ["compiled_decode_step"]


class _ExecutableLike(Protocol):
    """Surface of the pybind ``CompiledExecutable`` consumed here."""

    input_ids: list[int]
    output_ids: list[int]
    num_inputs: int


@final
@dataclass(slots=True)
class _DecodeEntry:
    """One compiled decode executable + the plan to invoke / write back."""

    exe: object
    external_feeds: dict[int, object]
    # Per ``exe.input_ids[i]``: 0 → input_ids arg, 1 → cache_position arg,
    # None → a pinned param / cache buffer in ``external_feeds``.
    input_source: tuple[int | None, ...]
    # The model output is exe output[0]; the remaining outputs are updated cache
    # buffers to copy back into these live buffer Tensors (same order).
    writeback_buffers: list[Tensor] = field(default_factory=list)
    n_hits: int = 0
    compile_ms: float = 0.0
    last_run_ms: float = 0.0


def compiled_decode_step(
    forward_fn: Callable[[Tensor, Tensor], Tensor],
    static_cache: StaticCache,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Return a ``step(input_ids, cache_position)`` running one compiled decode.

    Parameters
    ----------
    forward_fn : callable
        ``forward_fn(input_ids, cache_position) -> output`` — one decode forward
        that reads/writes ``static_cache`` internally (e.g. a model's
        single-token path).  Must return a single Tensor (the decode logits /
        hidden state).
    static_cache : StaticCache
        The cache whose per-layer buffers ``forward_fn`` writes via
        :func:`lucid.index_copy`.  Its buffers are rolled forward in place.

    Returns
    -------
    callable
        ``step(input_ids, cache_position) -> Tensor``.  ``input_ids`` is the new
        token(s); ``cache_position`` is a fixed-shape ``(T,)`` int64 Tensor of
        absolute write indices.  On a signature miss it (re)traces + compiles;
        otherwise it is a pure run + buffer write-back.  Falls back to a plain
        eager ``forward_fn`` call if the decode graph cannot be compiled.
    """
    from lucid._dispatch import _unwrap, _wrap
    from lucid._tensor.tensor import Tensor
    from lucid.autograd._grad_mode import no_grad
    from lucid.compile import _tracing
    from lucid.compile._core.attention_probe import maybe_probe_for_graph
    from lucid.compile._core.signature import signature_of

    cache: dict[Any, _DecodeEntry] = {}
    eager_sigs: set[Any] = set()

    def _buffer_impls() -> list[object]:
        """Flat list of the cache's live key/value buffer impls (key then value,
        per layer) — the write-back targets + pinned feeds."""
        impls: list[object] = []
        for i in range(len(static_cache.key_cache)):
            impls.append(_unwrap(static_cache.key_cache[i]))
            impls.append(_unwrap(static_cache.value_cache[i]))
        return impls

    def _restore(snap: tuple[Any, ...]) -> None:
        """Undo every cache mutation the trace caused (buffer rebinds + counters),
        so the real run starts from the pre-trace state."""
        orig_key, orig_val, orig_cumlen, orig_seen = snap
        static_cache.key_cache[:] = orig_key
        static_cache.value_cache[:] = orig_val
        static_cache._cumulative_length[:] = orig_cumlen
        static_cache._seen_tokens = orig_seen

    def _compile_for(input_ids: Tensor, cache_position: Tensor) -> _DecodeEntry | None:
        """Trace + compile one decode step for this signature."""
        t0 = time.perf_counter()

        # Snapshot ALL cache state before tracing.  forward_fn's ``update`` rebinds
        # key_cache[i]/value_cache[i] to the index_copy write outputs AND advances
        # the length counters — both must be rolled back so the live buffers stay
        # the pinned feeds and the real run isn't double-counted.
        n_layers = len(static_cache.key_cache)
        snap = (
            list(static_cache.key_cache),
            list(static_cache.value_cache),
            list(static_cache._cumulative_length),
            static_cache._seen_tokens,
        )
        orig_key, orig_val = snap[0], snap[1]

        with no_grad():
            with _tracing() as tracer:
                out = forward_fn(input_ids, cache_position)

        graph = tracer.graph
        if not graph.ops or not isinstance(out, Tensor):
            _restore(snap)
            return None

        maybe_probe_for_graph(graph)
        ext = tracer.external_feeds

        # The decode output + every cache buffer's post-write id become the
        # executable's explicit outputs (output first, then key/value per layer).
        out_id = tracer.lookup_id(_unwrap(out))
        if out_id is None:
            _restore(snap)
            return None
        write_ids: list[int] = [int(out_id)]
        writeback_buffers: list[Tensor] = []
        ok = True
        for i in range(n_layers):
            for buf_after, buf_before in (
                (static_cache.key_cache[i], orig_key[i]),
                (static_cache.value_cache[i], orig_val[i]),
            ):
                wid = tracer.lookup_id(_unwrap(buf_after))
                if wid is None:
                    ok = False
                    break
                write_ids.append(int(wid))
                writeback_buffers.append(buf_before)
            if not ok:
                break

        _restore(snap)
        if not ok:
            return None

        try:
            exe = _C_engine.compile.compile_or_cached(
                graph, dict(ext), False, [], write_ids
            )
        except RuntimeError:
            return None
        if exe is None:
            return None

        # Feed plan: input_ids → slot 0, cache_position → slot 1, everything
        # else (model params + pinned cache buffers) → None (from external_feeds).
        ids_impl = id(_unwrap(input_ids))
        cp_impl = id(_unwrap(cache_position))
        input_source: list[int | None] = []
        for tid in exe.input_ids:
            impl = ext.get(tid)
            if impl is None:
                return None
            if id(impl) == ids_impl:
                input_source.append(0)
            elif id(impl) == cp_impl:
                input_source.append(1)
            else:
                input_source.append(None)

        return _DecodeEntry(
            exe=exe,
            external_feeds=dict(ext),
            input_source=tuple(input_source),
            writeback_buffers=writeback_buffers,
            compile_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _run(entry: _DecodeEntry, input_ids: Tensor, cache_position: Tensor) -> Tensor:
        """Run one cached decode step + write the rolled buffers back."""
        feed_impls: list[object] = []
        exe = cast(_ExecutableLike, entry.exe)
        runtime = (_unwrap(input_ids), _unwrap(cache_position))
        for tid, src in zip(exe.input_ids, entry.input_source):
            feed_impls.append(runtime[src] if src is not None else entry.external_feeds[tid])

        t0 = time.perf_counter()
        outs = _C_engine.compile.run_executable(entry.exe, feed_impls)
        entry.last_run_ms = (time.perf_counter() - t0) * 1000.0
        entry.n_hits += 1

        n_expected = 1 + len(entry.writeback_buffers)
        if len(outs) != n_expected:
            raise RuntimeError(
                f"compiled_decode_step: executable returned {len(outs)} outputs, "
                f"expected {n_expected} (1 output + {len(entry.writeback_buffers)} "
                f"cache buffers)"
            )
        # Roll each cache buffer forward in place; the live buffer is also the
        # pinned feed, so the next step reads the updated value.
        for buf, new_impl in zip(entry.writeback_buffers, outs[1:]):
            buf.copy_(_wrap(cast(Any, new_impl)))
        return _wrap(cast(Any, outs[0]))

    def step(input_ids: Tensor, cache_position: Tensor) -> Tensor:
        """Run one compiled decode step (or eager fallback)."""
        # The cache buffers must be pre-allocated (the prefill step does this) so
        # they are persistent external feeds rather than in-graph ``zeros``
        # re-created each run.  Until then, run eager — this allocates the cache
        # WITHOUT recording the signature as eager-only (so the next, allocated,
        # call still attempts to compile).
        if len(static_cache.key_cache) == 0:
            return forward_fn(input_ids, cache_position)

        try:
            # Params + cache buffers are pinned in the executable, so the cache
            # key only needs to distinguish input shapes — pass an empty param
            # fingerprint (and no model) to skip the parameter walk.
            key = signature_of(
                cast(Any, None),
                (input_ids, cache_position),
                {},
                dynamic=False,
                param_fingerprint=(),
            )
        except (TypeError, AttributeError):
            key = None

        if key is not None and key in eager_sigs:
            return forward_fn(input_ids, cache_position)

        entry = cache.get(key) if key is not None else None
        if entry is None:
            entry = _compile_for(input_ids, cache_position)
            if entry is None:
                if key is not None:
                    eager_sigs.add(key)
                return forward_fn(input_ids, cache_position)
            if key is not None:
                cache[key] = entry
        return _run(entry, input_ids, cache_position)

    step.cache = cache  # type: ignore[attr-defined]
    return step
