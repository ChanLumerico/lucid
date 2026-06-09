"""Compiled single-token decode driver for ``StaticCache``-backed generation.

``DynamicCache`` grows its key/value tensors by concatenation every step, so the
decode forward changes shape each token and ``lucid.compile`` re-keys its
executable cache on every call — generation stays eager.  :class:`StaticCache`
fixes the shape (a pre-allocated ``(B, H, max_cache_len, D)`` buffer written in
place at a runtime ``cache_position``), which lets a single-token decode forward
compile **once** into a reused MPSGraph executable.

The wiring constraint: ``lucid.compile`` traces a module whose call args are all
positional Tensors, but a ``StaticCache`` object cannot be one.  So
:class:`_StaticDecodeWrap` threads the per-layer buffers as positional tensor
args/returns and rebuilds a cache *view* over them inside ``forward`` — the
executable reads the buffers, writes the new token's K/V, and returns the
updated buffers as extra outputs.  :class:`_CompiledStaticDecoder` then drives
the loop, ``copy_``-ing each step's output buffers back into the cache's
**persistent** storage so the re-fed inputs stay identity- and signature-stable
— which lets the executable compile exactly **once** and be reused for every
position.  (Rebinding to the executable's output tensors would drop that
~13 %-of-step copy, but the outputs carry a different compile signature than the
zero-allocated buffers, triggering a second compile that outweighs the copy
saving for realistic decode lengths — measured and rejected.)

The compiled executable is also reused across ``generate()`` calls (cached on
the model by a shape signature via :func:`_cached_compile`), so the one-time
MPSGraph compile is paid once per shape, not per call.

Measured reality (Apple Silicon, MLX): even with the compile amortised and the
``copy_``-back removed, a warm compiled decode step ties — does *not* beat —
eager ``StaticCache``, because MLX eager already fuses these ops well and both
``StaticCache`` paths attend over the **full** ``max_cache_len`` buffer (more
work than ``DynamicCache``'s actual-length attention).  ``DynamicCache`` (eager)
remains the fastest path; ``compile_decode`` is an opt-in for when a single
reused MPSGraph executable is wanted, not a guaranteed speedup.
"""

from typing import TYPE_CHECKING, Callable, cast, final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.compile import CompiledModule
from lucid.utils.cache import EncoderDecoderCache, StaticCache

if TYPE_CHECKING:
    from lucid.models.text.transformer._model import TransformerForSeq2SeqLM

# Attribute under which a model caches its compiled decode executables, keyed by
# a shape signature.  Stored on the model instance (not a module-level dict) so
# the cache's lifetime tracks the model's — the model<->wrap reference cycle is
# collected together by the cyclic GC when the model is dropped, no leak.
_DECODE_CACHE_ATTR = "_lucid_compiled_decode"


def _cached_compile(
    model: nn.Module, key: tuple[object, ...], build_wrap: Callable[[], nn.Module]
) -> CompiledModule[..., tuple[Tensor, ...]]:
    """Return a ``CompiledModule`` for ``key``, reusing it across ``generate()``
    calls instead of recompiling every call.

    The compiled executable depends only on the model's parameters (identity-
    stable) and the per-step input *shapes* captured in ``key`` — the decode
    buffers are passed positionally each step, so one executable serves every
    same-shape ``StaticCache``.  Rebuilding a fresh ``lucid.compile(wrap)`` per
    call (the previous behaviour) discarded the executable and re-paid the
    one-time MPSGraph compile on every call; caching it amortises that cost.

    Parameters
    ----------
    model : nn.Module
        The decoder model; owns the per-key executable cache.
    key : tuple of object
        Shape signature distinguishing executables (layers / buffer dims /
        dtype / device / mask presence).
    build_wrap : callable returning nn.Module
        Factory for the wrap module to compile on a cache miss.

    Returns
    -------
    CompiledModule
        The cached (or freshly compiled) executable wrapper.
    """
    cache: dict[tuple[object, ...], CompiledModule[..., tuple[Tensor, ...]]] | None = (
        getattr(model, _DECODE_CACHE_ATTR, None)
    )
    if cache is None:
        cache = {}
        setattr(model, _DECODE_CACHE_ATTR, cache)
    compiled = cache.get(key)
    if compiled is None:
        compiled = cast(
            CompiledModule[..., tuple[Tensor, ...]], lucid.compile(build_wrap())
        )
        cache[key] = compiled
    return compiled


def _ceil_pow2(n: int) -> int:
    """Smallest power of two ``>= n`` (for ``n >= 1``).

    Drives the compiled-decode read-window ladder: a decode step that has written
    ``filled`` positions attends over ``ceil_pow2(filled + 1)`` keys, so the live
    write index ``filled`` is always inside ``[0, bucket)`` and the executable is
    reused across every position sharing a bucket (≤ ``log2(max_cache_len) + 1``
    distinct widths total).  ``n + 1`` (not ``n``) is the caller's responsibility.
    """
    return 1 << (n - 1).bit_length() if n > 1 else 1


@final
class _StaticDecodeWrap(nn.Module):
    """Thread a ``StaticCache``'s buffers through a model's single-token forward.

    The wrapped model must accept ``(input_ids, past_key_values=, cache_position=,
    use_cache=)`` and return either a ``ModelOutput`` with ``.logits`` or the
    logits tensor directly (the decoder-only zoo contract).

    Parameters
    ----------
    model : nn.Module
        The decoder model (e.g. a ``GPT2LMHeadModel``).  Registered as a
        submodule so ``lucid.compile`` discovers its parameters as graph feeds.
    num_layers : int
        Number of cached layers (the buffer list is split key/value at this
        index).
    max_cache_len : int
        Fixed sequence capacity of every buffer.
    read_len : int
        Baked read-window width — the attention narrows its q·kᵀ to this many
        keys (a power-of-two bucket ``>= filled + 1``).  Baked as a constant so it
        keys the executable (one compile per bucket); ``max_cache_len`` ⇒ full
        width (no narrowing).
    """

    def __init__(
        self, model: nn.Module, num_layers: int, max_cache_len: int, read_len: int
    ) -> None:
        super().__init__()
        self.model = model
        self.num_layers = int(num_layers)
        self.max_cache_len = int(max_cache_len)
        self.read_len = int(read_len)

    @override
    def forward(  # type: ignore[override]  # reason: variadic buffer pass-through; the base Module.forward types *args loosely.
        self, token: Tensor, cache_position: Tensor, *buffers: Tensor
    ) -> tuple[Tensor, ...]:
        cache = StaticCache.from_buffers(
            list(buffers[: self.num_layers]),
            list(buffers[self.num_layers :]),
            self.max_cache_len,
            read_len=self.read_len,
        )
        out = self.model(
            token,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = cast(Tensor, out.logits if hasattr(out, "logits") else out)
        return (logits, *cache.key_cache, *cache.value_cache)


@final
class _CompiledStaticDecoder:
    """Drive compiled single-token decode steps over a prefilled ``StaticCache``.

    The cache's per-layer buffers are the persistent decode state: the compiled
    executable reads them, the driver ``copy_``-s each step's output back into
    them (keeping inputs signature-stable for a single compile).  Construct
    *after* the prompt has been prefilled into ``past`` (the buffers must already
    hold the prompt's K/V) — ``step`` continues from the next position.

    Parameters
    ----------
    model : nn.Module
        The decoder model, same instance used for the eager prefill.
    past : StaticCache
        The prefilled cache; its ``key_cache`` / ``value_cache`` lists are
        retained by reference and updated in place each step.
    """

    def __init__(self, model: nn.Module, past: StaticCache) -> None:
        self._model = model
        self._num_layers = len(past.key_cache)
        self._key_cache = past.key_cache
        self._value_cache = past.value_cache
        self._max_cache_len = past.max_cache_len
        b, h, _, d = past.key_cache[0].shape
        # Shape signature shared by every bucket's executable (the bucket itself is
        # appended per step), so each distinct read-window width compiles once.
        self._sig: tuple[object, ...] = (
            "decoder-only",
            self._num_layers,
            past.max_cache_len,
            int(b),
            int(h),
            int(d),
            str(past.key_cache[0].dtype),
            past.key_cache[0].device.type,
        )

    def step(self, token: Tensor, cache_position: Tensor, cur_pos: int) -> Tensor:
        """Run one compiled decode step; return ``(B, 1, vocab)`` logits.

        Parameters
        ----------
        token : Tensor
            ``(B, 1)`` int next-token ids.
        cache_position : Tensor
            ``(1,)`` int64 absolute write position for this step (shared across
            the batch — all rows decode in lockstep).
        cur_pos : int
            The absolute write index of this step (== ``cache_position`` value).
            The read window is the power-of-two bucket ``ceil_pow2(cur_pos + 1)``
            (capped at ``max_cache_len``), so attention covers the written prefix
            ``[0, cur_pos]`` and the executable is reused across the bucket.
        """
        bucket = min(_ceil_pow2(cur_pos + 1), self._max_cache_len)
        compiled = _cached_compile(
            self._model,
            (*self._sig, bucket),
            lambda: _StaticDecodeWrap(
                self._model, self._num_layers, self._max_cache_len, bucket
            ),
        )
        result = compiled(token, cache_position, *self._key_cache, *self._value_cache)
        logits = result[0]
        # ``copy_`` the updated K/V back into the cache's persistent buffers
        # rather than rebinding to the executable's output tensors.  This keeps
        # the re-fed inputs *identity- and signature-stable* across steps, so the
        # executable compiles exactly ONCE (the whole point of StaticCache
        # compiled decode).  Rebinding to the outputs would save this ~13%-of-
        # step copy but the output tensors carry a different compile signature
        # than the zero-allocated buffers → a second compile fires (~one-time,
        # but it dwarfs the per-step copy saving for any realistic decode
        # length, and is non-deterministic) — measured and rejected.
        for layer_idx in range(self._num_layers):
            self._key_cache[layer_idx].copy_(result[1 + layer_idx])
            self._value_cache[layer_idx].copy_(result[1 + self._num_layers + layer_idx])
        return logits


@final
class _Seq2SeqDecodeWrap(nn.Module):
    """Thread an encoder-decoder ``StaticCache`` pair through a decode step.

    The encoder ``memory`` and the cross-attention key/value buffers are
    **constant** after the prompt is encoded, so they are threaded in as
    read-only positional inputs and are absent from the return tuple; only the
    growing self-attention buffers are written back.  Pre-set ``is_updated=True``
    for every layer so the traced cross-attention takes the **reuse** branch:
    it reads the cached cross K/V instead of attending freshly-projected memory,
    making the step's output independent of ``memory``.  (The shared attention
    still projects ``memory`` to K/V before discarding it on the reuse branch,
    so ``memory`` remains a — graph-pruned — positional feed; a future
    projection short-circuit in ``MultiheadAttention`` could drop it entirely.)

    Parameters
    ----------
    model : nn.Module
        The ``TransformerForSeq2SeqLM`` (exposes ``transformer.decode`` and
        ``lm_head``).  Registered as a submodule so ``lucid.compile`` discovers
        its parameters as graph feeds.
    num_layers : int
        Number of decoder layers.
    self_max_len : int
        Fixed sequence capacity of the self-attention buffers.
    cross_len : int
        Encoder source length (fixed capacity of the cross-attention buffers).
    self_read_len : int
        Baked read-window width for the SELF-attention (a power-of-two bucket);
        the decoder narrows its self-attention q·kᵀ to this many keys.  The
        cross-attention always reads the full ``cross_len`` (constant source).
    has_memory_mask : bool, keyword-only
        Whether a source padding mask is threaded as an extra positional input.
        Omitted for an unpadded source so cross-attention keeps the fused SDPA
        fast path (an all-attend mask would force the slower explicit path).
    """

    def __init__(
        self,
        model: TransformerForSeq2SeqLM,
        num_layers: int,
        self_max_len: int,
        cross_len: int,
        self_read_len: int,
        *,
        has_memory_mask: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_layers = int(num_layers)
        self.self_max_len = int(self_max_len)
        self.cross_len = int(cross_len)
        self.self_read_len = int(self_read_len)
        self.has_memory_mask = has_memory_mask

    @override
    def forward(  # type: ignore[override]  # reason: variadic buffer pass-through; the base Module.forward types *args loosely.
        self, token: Tensor, cache_position: Tensor, memory: Tensor, *rest: Tensor
    ) -> tuple[Tensor, ...]:
        # The source padding mask (when present) is the first of ``rest``; the
        # rest are the 4N self/cross key/value buffers.
        memory_mask: Tensor | None
        if self.has_memory_mask:
            memory_mask = rest[0]
            buffers = rest[1:]
        else:
            memory_mask = None
            buffers = rest
        n = self.num_layers
        cache = EncoderDecoderCache(
            StaticCache.from_buffers(
                list(buffers[:n]),
                list(buffers[n : 2 * n]),
                self.self_max_len,
                read_len=self.self_read_len,  # narrow self-attn to the bucket
            ),
            StaticCache.from_buffers(
                list(buffers[2 * n : 3 * n]),
                list(buffers[3 * n : 4 * n]),
                self.cross_len,  # cross-attn stays full (constant source)
            ),
            is_updated={layer_idx: True for layer_idx in range(n)},
        )
        decoded = self.model.transformer.decode(
            token,
            memory,
            memory_attention_mask=memory_mask,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
        )
        logits = cast(Tensor, self.model.lm_head(decoded))
        # Only the self-attention buffers grow; cross + memory are constant.
        self_cache = cache.self_attention_cache
        return (logits, *self_cache.key_cache, *self_cache.value_cache)


@final
class _CompiledSeq2SeqDecoder:
    """Drive compiled single-token decode steps over a prefilled enc-dec cache.

    Construct *after* the BOS token has been decoded eagerly into ``past`` (so
    the cross-attention buffers hold the encoder projection and the
    self-attention buffer holds position 0).  ``memory`` and the cross-attention
    buffers are held by reference as constant graph feeds; each ``step`` writes
    only the self-attention buffers back.

    Parameters
    ----------
    model : nn.Module
        The ``TransformerForSeq2SeqLM`` used for the eager prefill.
    past : EncoderDecoderCache
        The prefilled cache (both sub-caches are ``StaticCache``).
    memory : Tensor
        Encoder output ``(B, S, d_model)`` — constant across decode steps.
    memory_mask : Tensor or None
        ``(B, S)`` source attention mask (1 = attend, 0 = pad), constant across
        steps; ``None`` for an unpadded source (keeps the fused SDPA path).
    """

    def __init__(
        self,
        model: TransformerForSeq2SeqLM,
        past: EncoderDecoderCache,
        memory: Tensor,
        memory_mask: Tensor | None,
    ) -> None:
        self_cache = past.self_attention_cache
        cross_cache = past.cross_attention_cache
        if not isinstance(self_cache, StaticCache) or not isinstance(
            cross_cache, StaticCache
        ):
            raise TypeError(
                "compiled seq2seq decode requires StaticCache self/cross caches"
            )
        self._model = model
        self._num_layers = len(self_cache.key_cache)
        self._self_key = self_cache.key_cache
        self._self_value = self_cache.value_cache
        self._self_max_len = self_cache.max_cache_len
        # memory + cross buffers + the source mask are constant graph feeds.
        self._memory = memory
        self._memory_mask = memory_mask
        self._cross_key = cross_cache.key_cache
        self._cross_value = cross_cache.value_cache
        self._cross_len = cross_cache.max_cache_len
        b, h, _, d = self_cache.key_cache[0].shape
        # Shape signature shared by every self-attn bucket's executable (the bucket
        # is appended per step), so each distinct self read-window compiles once.
        self._sig: tuple[object, ...] = (
            "seq2seq",
            self._num_layers,
            self_cache.max_cache_len,
            cross_cache.max_cache_len,
            int(b),
            int(h),
            int(d),
            str(self_cache.key_cache[0].dtype),
            self_cache.key_cache[0].device.type,
            memory_mask is not None,
        )

    def step(self, token: Tensor, cache_position: Tensor, cur_pos: int) -> Tensor:
        """Run one compiled decode step; return ``(B, 1, vocab)`` logits.

        Parameters
        ----------
        token : Tensor
            ``(B, 1)`` int next-token ids.
        cache_position : Tensor
            ``(1,)`` int64 absolute write position for this step.
        cur_pos : int
            The absolute self-attention write index of this step.  The self-attn
            read window is the power-of-two bucket ``ceil_pow2(cur_pos + 1)``
            (capped at ``self_max_len``); cross-attention stays full.
        """
        self_bucket = min(_ceil_pow2(cur_pos + 1), self._self_max_len)
        compiled = _cached_compile(
            self._model,
            (*self._sig, self_bucket),
            lambda: _Seq2SeqDecodeWrap(
                self._model,
                self._num_layers,
                self._self_max_len,
                self._cross_len,
                self_bucket,
                has_memory_mask=self._memory_mask is not None,
            ),
        )
        mask: tuple[Tensor, ...] = (
            (self._memory_mask,) if self._memory_mask is not None else ()
        )
        result = compiled(
            token,
            cache_position,
            self._memory,
            *mask,
            *self._self_key,
            *self._self_value,
            *self._cross_key,
            *self._cross_value,
        )
        logits = result[0]
        # ``copy_`` back only the self-attention buffers (cross + memory are
        # constant read-only feeds), keeping inputs signature-stable for a single
        # compile — see :meth:`_CompiledStaticDecoder.step` for why rebinding to
        # the executable outputs is rejected.
        for layer_idx in range(self._num_layers):
            self._self_key[layer_idx].copy_(result[1 + layer_idx])
            self._self_value[layer_idx].copy_(result[1 + self._num_layers + layer_idx])
        return logits
