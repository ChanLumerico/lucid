"""``EncoderDecoderCache`` — paired self-attention + cross-attention cache.

Mirrors the reference ``EncoderDecoderCache``: same class name, same
``self_attention_cache`` / ``cross_attention_cache`` / ``is_updated``
attributes and method surface.  Used by encoder-decoder models, where the
decoder's self-attention cache grows each step while the cross-attention
key/value (derived only from the constant encoder ``memory``) is computed
once and reused.
"""

from typing import TYPE_CHECKING, Self, final, override

from lucid.utils.cache._base import Cache
from lucid.utils.cache._dynamic import DynamicCache

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


@final
class EncoderDecoderCache(Cache):
    """Wraps a decoder self-attention cache and a cross-attention cache.

    Parameters
    ----------
    self_attention_cache : DynamicCache
        Grows by one position per decode step (decoder self-attention).
    cross_attention_cache : DynamicCache
        Filled once from the encoder ``memory`` then reused unchanged.

    Attributes
    ----------
    self_attention_cache : DynamicCache
    cross_attention_cache : DynamicCache
    is_updated : dict[int, bool]
        Per-layer flag marking whether the cross-attention key/value for that
        layer has already been computed; lets cross-attention skip the
        redundant ``memory`` projection on every step after the first.
    """

    def __init__(
        self,
        self_attention_cache: DynamicCache,
        cross_attention_cache: DynamicCache,
    ) -> None:
        """Pair an existing self-attention and cross-attention cache."""
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache
        self.is_updated: dict[int, bool] = {}
        for layer_idx in range(len(cross_attention_cache)):
            self.is_updated[layer_idx] = (
                cross_attention_cache.get_seq_length(layer_idx) > 0
            )

    def __len__(self) -> int:
        """Number of decoder layers (from the self-attention cache)."""
        return len(self.self_attention_cache)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(self_k, self_v, cross_k, cross_v)`` for ``layer_idx``."""
        self_k, self_v = self.self_attention_cache[layer_idx]
        cross_k, cross_v = self.cross_attention_cache[layer_idx]
        return self_k, self_v, cross_k, cross_v

    @override
    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Update the decoder **self-attention** cache for ``layer_idx``.

        Cross-attention key/value are written directly through
        :attr:`cross_attention_cache` by the attention module (guarded by
        :attr:`is_updated`), not through this method.
        """
        return self.self_attention_cache.update(
            key_states, value_states, layer_idx, cache_kwargs
        )

    @override
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.self_attention_cache.get_seq_length(layer_idx)

    @override
    def get_max_cache_shape(self) -> int | None:
        return self.self_attention_cache.get_max_cache_shape()

    @override
    def reorder_cache(self, beam_idx: Tensor) -> None:
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def to_legacy_cache(
        self,
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], ...]:
        """Convert to the legacy per-layer ``(self_k, self_v, cross_k, cross_v)``
        tuple format."""
        legacy: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for layer_idx in range(len(self.self_attention_cache)):
            self_k, self_v = self.self_attention_cache[layer_idx]
            cross_k, cross_v = self.cross_attention_cache[layer_idx]
            legacy.append((self_k, self_v, cross_k, cross_v))
        return tuple(legacy)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: (
            tuple[tuple[Tensor, Tensor, Tensor, Tensor], ...] | None
        ) = None,
    ) -> Self:
        """Build an :class:`EncoderDecoderCache` from the legacy 4-tuple format."""
        self_cache = DynamicCache()
        cross_cache = DynamicCache()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                self_k, self_v, cross_k, cross_v = past_key_values[layer_idx]
                self_cache.update(self_k, self_v, layer_idx)
                cross_cache.update(cross_k, cross_v, layer_idx)
        return cls(self_cache, cross_cache)
