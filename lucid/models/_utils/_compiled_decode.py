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
**persistent, freshly-allocated** storage.  That ``copy_`` (1) keeps the re-fed
input tensors *identity-stable* across steps and (2) lands each step's K/V into
contiguous owned buffers rather than re-feeding the executable's own output
tensors back as inputs — feeding a view / shared-storage tensor as a compiled
input reads a stale compile-time offset (silent garbage), so the inputs every
step must be independent materialised tensors.

For long-context decoding (the real KV-cache use case) the flat fixed-buffer
cost beats ``DynamicCache``'s growing concat + widening attention; for short
prompts the two are roughly even.
"""

from typing import TYPE_CHECKING, cast, final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.utils.cache import EncoderDecoderCache, StaticCache

if TYPE_CHECKING:
    from lucid.models.text.transformer._model import TransformerForSeq2SeqLM


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
    """

    def __init__(self, model: nn.Module, num_layers: int, max_cache_len: int) -> None:
        super().__init__()
        self.model = model
        self.num_layers = int(num_layers)
        self.max_cache_len = int(max_cache_len)

    @override
    def forward(  # type: ignore[override]  # reason: variadic buffer pass-through; the base Module.forward types *args loosely.
        self, token: Tensor, cache_position: Tensor, *buffers: Tensor
    ) -> tuple[Tensor, ...]:
        cache = StaticCache.from_buffers(
            list(buffers[: self.num_layers]),
            list(buffers[self.num_layers :]),
            self.max_cache_len,
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
    them.  Construct *after* the prompt has been prefilled into ``past`` (the
    buffers must already hold the prompt's K/V) — ``step`` continues from the
    next position.

    Parameters
    ----------
    model : nn.Module
        The decoder model, same instance used for the eager prefill.
    past : StaticCache
        The prefilled cache; its ``key_cache`` / ``value_cache`` lists are
        retained by reference and updated in place each step.
    """

    def __init__(self, model: nn.Module, past: StaticCache) -> None:
        self._num_layers = len(past.key_cache)
        self._key_cache = past.key_cache
        self._value_cache = past.value_cache
        wrap = _StaticDecodeWrap(model, self._num_layers, past.max_cache_len)
        self._compiled = lucid.compile(wrap)

    def step(self, token: Tensor, cache_position: Tensor) -> Tensor:
        """Run one compiled decode step; return ``(B, 1, vocab)`` logits.

        Parameters
        ----------
        token : Tensor
            ``(B, 1)`` int next-token ids.
        cache_position : Tensor
            ``(1,)`` int64 absolute write position for this step (shared across
            the batch — all rows decode in lockstep).
        """
        result = self._compiled(
            token, cache_position, *self._key_cache, *self._value_cache
        )
        logits = result[0]
        # Land the updated K/V into the cache's persistent, contiguous buffers
        # (identity-stable, owned storage) rather than re-feeding the
        # executable's output tensors as next-step inputs.
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
        *,
        has_memory_mask: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_layers = int(num_layers)
        self.self_max_len = int(self_max_len)
        self.cross_len = int(cross_len)
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
                list(buffers[:n]), list(buffers[n : 2 * n]), self.self_max_len
            ),
            StaticCache.from_buffers(
                list(buffers[2 * n : 3 * n]),
                list(buffers[3 * n : 4 * n]),
                self.cross_len,
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
        self._num_layers = len(self_cache.key_cache)
        self._self_key = self_cache.key_cache
        self._self_value = self_cache.value_cache
        # memory + cross buffers + the source mask are constant graph feeds.
        self._memory = memory
        self._memory_mask = memory_mask
        self._cross_key = cross_cache.key_cache
        self._cross_value = cross_cache.value_cache
        wrap = _Seq2SeqDecodeWrap(
            model,
            self._num_layers,
            self_cache.max_cache_len,
            cross_cache.max_cache_len,
            has_memory_mask=memory_mask is not None,
        )
        self._compiled = lucid.compile(wrap)

    def step(self, token: Tensor, cache_position: Tensor) -> Tensor:
        """Run one compiled decode step; return ``(B, 1, vocab)`` logits.

        Parameters
        ----------
        token : Tensor
            ``(B, 1)`` int next-token ids.
        cache_position : Tensor
            ``(1,)`` int64 absolute write position for this step.
        """
        mask: tuple[Tensor, ...] = (
            (self._memory_mask,) if self._memory_mask is not None else ()
        )
        result = self._compiled(
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
        # Write back only the self-attention buffers (cross + memory constant).
        for layer_idx in range(self._num_layers):
            self._self_key[layer_idx].copy_(result[1 + layer_idx])
            self._self_value[layer_idx].copy_(result[1 + self._num_layers + layer_idx])
        return logits
