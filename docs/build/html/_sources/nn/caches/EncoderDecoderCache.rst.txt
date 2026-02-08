nn.EncoderDecoderCache
======================

.. autoclass:: lucid.nn.EncoderDecoderCache

Overview
--------

`EncoderDecoderCache` is a container cache for encoder-decoder style models.
It bundles two KV cache instances:

- `self_attention_cache` for decoder self-attention
- `cross_attention_cache` for decoder cross-attention

The class routes cache reads and writes to one of the two internal caches
using `is_cross_attention`.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.EncoderDecoderCache(
        self_attention_cache: nn.KVCache | None = None,
        cross_attention_cache: nn.KVCache | None = None,
    )

If either cache is not provided, `DynamicKVCache` is created by default.

Key methods
-----------

.. automethod:: lucid.nn.EncoderDecoderCache.update

.. automethod:: lucid.nn.EncoderDecoderCache.get

.. automethod:: lucid.nn.EncoderDecoderCache.get_seq_length

.. automethod:: lucid.nn.EncoderDecoderCache.reset

.. automethod:: lucid.nn.EncoderDecoderCache.batch_select_indices

.. automethod:: lucid.nn.EncoderDecoderCache.batch_repeat_interleave

.. automethod:: lucid.nn.EncoderDecoderCache.crop

`is_updated`
------------

`EncoderDecoderCache` exposes `is_updated: dict[int, bool]` to track whether
cross-attention cache has been updated per layer.

Minimal example
---------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.EncoderDecoderCache(
        self_attention_cache=nn.DynamicKVCache(),
        cross_attention_cache=nn.DynamicKVCache(),
    )

    # decoder self-attention cache update
    k_self = lucid.randn(2, 8, 1, 64)
    v_self = lucid.randn(2, 8, 1, 64)
    cache.update(k_self, v_self, layer_idx=0, is_cross_attention=False)

    # decoder cross-attention cache update
    k_cross = lucid.randn(2, 8, 10, 64)
    v_cross = lucid.randn(2, 8, 10, 64)
    cache.update(k_cross, v_cross, layer_idx=0, is_cross_attention=True)

    print(cache.get_seq_length(0, is_cross_attention=False))  # 1
    print(cache.get_seq_length(0, is_cross_attention=True))   # 10
