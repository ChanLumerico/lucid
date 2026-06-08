"""``DynamicCache`` — the default growing key/value cache.

Mirrors the reference ``DynamicCache``: same class name, same ``key_cache`` /
``value_cache`` / ``_seen_tokens`` attributes, same ``update`` /
``get_seq_length`` / ``to_legacy_cache`` / ``from_legacy_cache`` surface, so
existing generation code ports verbatim.  Pure ``lucid`` core ops only
(``lucid.cat`` along the sequence dimension); no engine work required.
"""

from typing import TYPE_CHECKING, Iterator, Self, final, override

import lucid
from lucid.utils.cache._base import Cache

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# Per-layer key/value tensors are laid out ``(B, num_heads, T, head_dim)`` in
# every Lucid attention, so the sequence axis is the second-to-last (dim -2).
_SEQ_DIM = -2


@final
class DynamicCache(Cache):
    """A cache that grows its key/value tensors by concatenation each step.

    Holds one key tensor and one value tensor per layer in :attr:`key_cache`
    and :attr:`value_cache`.  The first :meth:`update` for a layer seeds the
    entry; subsequent updates concatenate the new token(s) along the sequence
    dimension.  This is the eager default for autoregressive decoding.

    Attributes
    ----------
    key_cache : list[Tensor]
        Per-layer accumulated keys, each ``(B, num_heads, T, head_dim)``.
    value_cache : list[Tensor]
        Per-layer accumulated values, same shapes.
    _seen_tokens : int
        Running count of tokens the cache has processed (tracked from
        ``layer_idx == 0``), exposed via :attr:`seen_tokens`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.cache import DynamicCache
    >>> cache = DynamicCache()
    >>> k = lucid.zeros(1, 2, 3, 4)   # (B, H, T, D)
    >>> v = lucid.zeros(1, 2, 3, 4)
    >>> ck, cv = cache.update(k, v, layer_idx=0)
    >>> cache.get_seq_length()
    3
    """

    def __init__(self) -> None:
        """Create an empty cache with no layers populated."""
        self._seen_tokens: int = 0
        self.key_cache: list[Tensor] = []
        self.value_cache: list[Tensor] = []

    @property
    def seen_tokens(self) -> int:
        """Total number of tokens processed across all :meth:`update` calls."""
        return self._seen_tokens

    def __len__(self) -> int:
        """Number of layers currently held in the cache."""
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return the ``(keys, values)`` pair cached for ``layer_idx``."""
        if 0 <= layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        raise KeyError(
            f"Cache has {len(self.key_cache)} layers; no entry for layer {layer_idx}."
        )

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate ``(keys, values)`` pairs in layer order."""
        for layer_idx in range(len(self.key_cache)):
            yield self.key_cache[layer_idx], self.value_cache[layer_idx]

    @override
    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if layer_idx == 0:
            self._seen_tokens += int(key_states.shape[_SEQ_DIM])

        if len(self.key_cache) <= layer_idx:
            # First time we see this layer — seed the entry.  Layers are
            # expected to update in order (0, 1, 2, ...), matching the trunk's
            # forward loop.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = lucid.cat(
                [self.key_cache[layer_idx], key_states], dim=_SEQ_DIM
            )
            self.value_cache[layer_idx] = lucid.cat(
                [self.value_cache[layer_idx], value_states], dim=_SEQ_DIM
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @override
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return int(self.key_cache[layer_idx].shape[_SEQ_DIM])

    @override
    def get_max_cache_shape(self) -> int | None:
        return None

    @override
    def reorder_cache(self, beam_idx: Tensor) -> None:
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0, beam_idx
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0, beam_idx
            )

    def to_legacy_cache(self) -> tuple[tuple[Tensor, Tensor], ...]:
        """Convert to the legacy ``tuple(tuple(key, value), ...)`` format.

        Returns one ``(key, value)`` pair per layer, in layer order — the
        representation older decoder ``forward`` signatures consumed before
        :class:`Cache` objects existed.
        """
        return tuple(
            (self.key_cache[layer_idx], self.value_cache[layer_idx])
            for layer_idx in range(len(self.key_cache))
        )

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None
    ) -> Self:
        """Build a :class:`DynamicCache` from the legacy tuple format.

        ``past_key_values`` is a per-layer tuple of ``(key, value)`` tensors;
        ``None`` yields an empty cache.
        """
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache
