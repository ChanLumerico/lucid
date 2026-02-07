Neural Caches
=============

The :mod:`lucid.nn.cache` module provides cache classes for Transformer-style
attention. These caches store past key/value (KV) tensors per layer so decoding
can reuse previous attention states instead of recomputing full history.

.. toctree::
    :maxdepth: 1
    :hidden:

    KVCache.rst
    DynamicKVCache.rst
    StaticKVCache.rst

Why KV cache matters
--------------------

In autoregressive decoding, each new token attends to all previous tokens.
Without cache, each step recomputes every historical key/value tensor.
With cache, only the new key/value tensors are produced and merged into
the existing cache.

Common cache workflow
---------------------

1. Create a cache object before decoding.
2. Pass the cache into attention-enabled model forward calls.
3. Update cache with each decoding step.
4. Reorder/select cache during beam search.
5. Optionally crop cache for memory control.
6. Reset cache when sequence generation is finished.

Quick start
-----------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()

    # key/value shape example: (batch, heads, seq_len, head_dim)
    key = lucid.randn(2, 8, 1, 64)
    value = lucid.randn(2, 8, 1, 64)

    # append into layer 0
    k_all, v_all = cache.update(key, value, layer_idx=0)
    print(k_all.shape, v_all.shape)  # (2, 8, 1, 64), (2, 8, 1, 64)

    # next token append
    key2 = lucid.randn(2, 8, 1, 64)
    value2 = lucid.randn(2, 8, 1, 64)
    k_all, v_all = cache.update(key2, value2, layer_idx=0)
    print(cache.get_seq_length(0))   # 2

Cache types
-----------

- :class:`lucid.nn.KVCache`
  Abstract base interface for all KV cache implementations.

- :class:`lucid.nn.DynamicKVCache`
  Grows sequence length dynamically (append/expand on demand).

- :class:`lucid.nn.StaticKVCache`
  Uses fixed preallocated storage per layer (`max_cache_len`).

Choosing Dynamic vs Static
--------------------------

- Use `DynamicKVCache` when:
  - sequence length is highly variable,
  - you want simpler bring-up and debugging.

- Use `StaticKVCache` when:
  - you know an upper bound on decode length,
  - stable memory behavior is important.

Supported utility operations
----------------------------

All cache classes support:

- `update(...)` for layer KV update.
- `get(layer_idx)` and `get_seq_length(layer_idx)`.
- `reorder_cache(beam_idx)` for beam search.
- `batch_select_indices(indices)` for general batch reindexing.
- `batch_repeat_interleave(repeats)` for beam expansion.
- `crop(max_length)` for keeping recent tokens only.
- `reset()` for clearing states.

`cache_position` modes
----------------------

Both dynamic and static caches support:

- `None`: append mode.
- 0-D tensor: write a single token to a single position.
- 1-D tensor: write tokens to positions (length must match seq_len).
- 2-D tensor: batch-wise position mapping `(batch, seq_len)`.

Beam search snippets
--------------------

.. code-block:: python

    import lucid.nn as nn
    import lucid

    cache = nn.DynamicKVCache()
    # ... prefill/update cache first ...

    num_beams = 4
    cache.batch_repeat_interleave(num_beams)   # B -> B*num_beams

    # after beam pruning
    beam_idx = lucid.Tensor([3, 1, 0, 2], dtype=lucid.Int32)
    cache.reorder_cache(beam_idx)

Memory control snippet
----------------------

.. code-block:: python

    # keep only most recent 1024 tokens
    cache.crop(1024)

    # clear everything between requests
    cache.reset()
