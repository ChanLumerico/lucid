"""Abstract ``Cache`` interface for incremental key/value attention caching.

The public API — class name ``Cache``, the ``update`` / ``get_seq_length`` /
``get_max_cache_shape`` / ``get_usable_length`` / ``reorder_cache`` method
names, and their argument names — mirrors the reference ``cache_utils``
module exactly, so generation code ports across with only an import change.
"""

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class Cache(abc.ABC):
    """Abstract base class for all key/value caches used in attention.

    A cache stores, per transformer layer, the key and value projections of
    every token seen so far so that an incremental (one-token-at-a-time)
    forward pass only has to project the *new* token's key/value and attend
    against the accumulated history — turning an :math:`O(T^2)` re-encode of
    the whole prefix at every step into an :math:`O(T)` update.

    Subclasses implement :meth:`update` (the per-layer write/read), plus the
    length-introspection and beam-reordering hooks.  Concrete subclasses are
    :class:`~lucid.utils.cache.DynamicCache` (the default growing cache) and
    :class:`~lucid.utils.cache.EncoderDecoderCache` (self- + cross-attention).
    """

    @abc.abstractmethod
    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Append ``key_states`` / ``value_states`` for ``layer_idx`` and return
        the full cached key/value tensors.

        Parameters
        ----------
        key_states : Tensor
            New key projections of shape ``(B, num_heads, T_new, head_dim)``.
        value_states : Tensor
            New value projections of the same shape.
        layer_idx : int
            Index of the transformer layer this update belongs to.
        cache_kwargs : dict or None, optional
            Extra per-cache arguments (e.g. rotary ``cache_position``).
            Ignored by the dynamic cache; reserved for parity with other
            cache implementations.

        Returns
        -------
        tuple of (Tensor, Tensor)
            The accumulated ``(keys, values)`` for ``layer_idx``, each of
            shape ``(B, num_heads, T_total, head_dim)``.
        """

    @abc.abstractmethod
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the number of cached tokens for ``layer_idx`` (0 if empty)."""

    @abc.abstractmethod
    def get_max_cache_shape(self) -> int | None:
        """Return the maximum sequence length the cache can hold.

        ``None`` for an unbounded (dynamically growing) cache.
        """

    @abc.abstractmethod
    def reorder_cache(self, beam_idx: Tensor) -> None:
        """Reindex the cache along the batch dimension with ``beam_idx``.

        Used by beam search to keep each beam's cache aligned after the
        per-step beam permutation.  Mutates the cache in place.
        """

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Return how many cached tokens can be used given ``new_seq_length``.

        For a bounded cache this clamps so that ``previous + new`` never
        exceeds :meth:`get_max_cache_shape`; for an unbounded cache it is
        simply the current sequence length.
        """
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length
