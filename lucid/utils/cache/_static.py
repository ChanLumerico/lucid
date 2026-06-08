"""``StaticCache`` — fixed pre-allocated key/value cache for compiled decoding.

Unlike :class:`~lucid.utils.cache.DynamicCache` (which grows by concatenation
and so changes shape every step, defeating ``lucid.compile``'s static-shape
signature), ``StaticCache`` allocates a fixed ``(B, num_heads, max_cache_len,
head_dim)`` buffer once and writes each new token in place at ``cache_position``
via the traceable functional :func:`lucid.index_copy`.  The buffer shape is
constant every step, so a single-token decode forward compiles **once** into a
reused MPSGraph executable.

Class / method / argument names mirror the reference ``cache_utils.StaticCache``
for portability.  The write goes through ``lucid.index_copy`` (NOT in-place
``__setitem__``, which rebinds ``._impl`` and is invisible to the compile
tracer).
"""

from typing import Iterator, final, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.utils.cache._base import Cache

# Buffers are ``(B, num_heads, T, head_dim)`` — the sequence axis is dim 2.
_SEQ_DIM = 2


@final
class StaticCache(Cache):
    """A bounded cache backed by fixed pre-allocated buffers written in place.

    Parameters
    ----------
    max_cache_len : int
        Maximum sequence length the buffers can hold.  Must not exceed the
        model's positional range.

    Attributes
    ----------
    max_cache_len : int
        Fixed sequence capacity of every layer's buffer.
    key_cache : list[Tensor]
        Per-layer key buffers, each ``(B, num_heads, max_cache_len, head_dim)``
        (lazily allocated on the first :meth:`update`).
    value_cache : list[Tensor]
        Per-layer value buffers, same shapes.

    Notes
    -----
    ``get_seq_length`` is tracked by a per-layer counter (not inferred from the
    buffer shape, which is always ``max_cache_len``).  ``update`` returns the
    **full** buffer; the attention masks out the not-yet-written (future) tail
    via the usual causal mask, so the zero-filled slots never contribute.
    """

    def __init__(self, max_cache_len: int) -> None:
        """Create an empty static cache with capacity ``max_cache_len``."""
        self.max_cache_len: int = int(max_cache_len)
        self.key_cache: list[Tensor] = []
        self.value_cache: list[Tensor] = []
        self._cumulative_length: list[int] = []
        self._seen_tokens: int = 0

    @property
    def seen_tokens(self) -> int:
        """Total number of tokens written (tracked from ``layer_idx == 0``)."""
        return self._seen_tokens

    def __len__(self) -> int:
        """Number of layers currently allocated."""
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return the full ``(keys, values)`` buffers for ``layer_idx``."""
        if 0 <= layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        raise KeyError(
            f"Cache has {len(self.key_cache)} layers; no entry for layer {layer_idx}."
        )

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate the per-layer ``(keys, values)`` buffers."""
        for layer_idx in range(len(self.key_cache)):
            yield self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _lazy_init(self, layer_idx: int, key_states: Tensor) -> None:
        b, h, _, d = key_states.shape
        dev = key_states.device.type
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(
                lucid.zeros(
                    int(b),
                    int(h),
                    self.max_cache_len,
                    int(d),
                    dtype=key_states.dtype,
                    device=dev,
                )
            )
            self.value_cache.append(
                lucid.zeros(
                    int(b),
                    int(h),
                    self.max_cache_len,
                    int(d),
                    dtype=key_states.dtype,
                    device=dev,
                )
            )
            self._cumulative_length.append(0)

    @override
    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if len(self.key_cache) <= layer_idx:
            self._lazy_init(layer_idx, key_states)

        n = int(key_states.shape[_SEQ_DIM])
        cp_raw = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        if isinstance(cp_raw, Tensor):
            cache_position = cp_raw
        else:
            start = self._cumulative_length[layer_idx]
            cache_position = lucid.arange(
                start, start + n, device=key_states.device.type
            ).long()

        self.key_cache[layer_idx] = lucid.index_copy(
            self.key_cache[layer_idx], _SEQ_DIM, cache_position, key_states
        )
        self.value_cache[layer_idx] = lucid.index_copy(
            self.value_cache[layer_idx], _SEQ_DIM, cache_position, value_states
        )

        if layer_idx == 0:
            self._seen_tokens += n
        self._cumulative_length[layer_idx] += n
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @override
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self._cumulative_length):
            return 0
        return self._cumulative_length[layer_idx]

    @override
    def get_max_cache_shape(self) -> int | None:
        return self.max_cache_len

    @override
    def reorder_cache(self, beam_idx: Tensor) -> None:
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx
            )

    def reset(self) -> None:
        """Zero every buffer and reset the length counters in place."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = lucid.zeros_like(self.key_cache[layer_idx])
            self.value_cache[layer_idx] = lucid.zeros_like(self.value_cache[layer_idx])
            self._cumulative_length[layer_idx] = 0
        self._seen_tokens = 0

    def crop(self, max_length: int) -> None:
        """Logically truncate the cache to ``max_length`` tokens.

        The buffer is fixed-size, so cropping only rewinds the length counters
        (cropped positions become 'future' and are masked by the causal mask).
        A negative ``max_length`` drops that many tokens off the end.
        """
        for layer_idx in range(len(self._cumulative_length)):
            cur = self._cumulative_length[layer_idx]
            target = cur - abs(max_length) if max_length < 0 else max_length
            self._cumulative_length[layer_idx] = max(0, min(cur, target))
        self._seen_tokens = self._cumulative_length[0] if self._cumulative_length else 0

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat every batch row ``repeats`` times along the batch dimension."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )

    def batch_select_indices(self, indices: Tensor) -> None:
        """Keep only the batch rows named by ``indices`` along the batch dim."""
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, indices
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, indices
            )
