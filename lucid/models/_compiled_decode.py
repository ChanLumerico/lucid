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

from typing import cast, final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.utils.cache import StaticCache


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
