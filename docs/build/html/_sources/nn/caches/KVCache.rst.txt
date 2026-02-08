nn.KVCache
==========

.. autoclass:: lucid.nn.KVCache

Overview
--------

`KVCache` is the abstract base class for Transformer KV cache implementations.
It extends :class:`lucid.nn.Cache` with KV-specific APIs used by attention
modules and generation loops.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.KVCache()

All concrete caches must implement:

- `update(key, value, layer_idx, cache_position=None)`
- `get(layer_idx)`
- `get_seq_length(layer_idx=0)`
- `reset()`
- internal crop behavior used by `crop(max_length)`

Common public methods
---------------------

`KVCache` also provides shared utility methods:

- `reorder_cache(beam_idx)`
- `batch_select_indices(indices)`
- `batch_repeat_interleave(repeats)`
- `crop(max_length)`
- `get_max_cache_shape()` (returns `None` by default)

Shape convention
----------------

Typical key/value shape is:

.. math::

    (B, H, T, D_h)

- :math:`B`: batch size
- :math:`H`: number of attention heads
- :math:`T`: sequence length in cache axis
- :math:`D_h`: per-head dimension

Minimal API example
-------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache: nn.KVCache = nn.DynamicKVCache()
    key = lucid.randn(1, 8, 1, 64)
    value = lucid.randn(1, 8, 1, 64)

    cache.update(key, value, layer_idx=0)
    kv = cache.get(0)
    print(cache.get_seq_length(0))  # 1

Beam utility example
--------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()
    # ... cache is already populated ...

    # Expand batch for beam search
    cache.batch_repeat_interleave(4)

    # Select/reorder active beams
    active = lucid.Tensor([3, 1, 0, 2], dtype=lucid.Int32)
    cache.reorder_cache(active)

    # equivalent generic call
    cache.batch_select_indices(active)

Crop and reset example
----------------------

.. code-block:: python

    cache.crop(512)  # keep only the latest 512 tokens
    cache.reset()    # clear all layers

When to use this class directly
-------------------------------

Use `KVCache` as a type annotation and public API contract. Instantiate
`DynamicKVCache` or `StaticKVCache` for actual runtime behavior.
