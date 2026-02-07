nn.StaticKVCache
================

.. autoclass:: lucid.nn.StaticKVCache

Overview
--------

`StaticKVCache` preallocates fixed-size cache storage for each layer.
This is useful when decode length bounds are known and predictable memory
behavior is desired.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.StaticKVCache(
        max_cache_len: int,
        num_layers: int,
    )

Parameters
----------

- **max_cache_len** (*int*):
  Maximum sequence length storable per layer in the cache axis.

- **num_layers** (*int*):
  Number of layer slots maintained by this cache instance.

Constructor
-----------

.. code-block:: python

    import lucid.nn as nn

    cache = nn.StaticKVCache(
        max_cache_len=2048,  # per-layer max sequence length
        num_layers=24,       # number of Transformer layers
    )

Basic append example
--------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.StaticKVCache(max_cache_len=16, num_layers=2)

    k = lucid.randn(2, 8, 4, 64)
    v = lucid.randn(2, 8, 4, 64)
    cache.update(k, v, layer_idx=0)

    print(cache.get_seq_length(0))     # 4
    print(cache.get_max_cache_shape()) # 16

Position update example (0-D)
-----------------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.StaticKVCache(max_cache_len=16, num_layers=1)

    k = lucid.randn(1, 8, 1, 64)
    v = lucid.randn(1, 8, 1, 64)
    pos = lucid.Tensor(10, dtype=lucid.Int32)

    cache.update(k, v, layer_idx=0, cache_position=pos)
    print(cache.get_seq_length(0))  # 11

Position update example (1-D)
-----------------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.StaticKVCache(max_cache_len=16, num_layers=1)

    k = lucid.randn(1, 8, 3, 64)
    v = lucid.randn(1, 8, 3, 64)
    pos = lucid.Tensor([0, 4, 7], dtype=lucid.Int32)

    cache.update(k, v, layer_idx=0, cache_position=pos)
    print(cache.get_seq_length(0))  # 8

Batch-wise position update (2-D)
--------------------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.StaticKVCache(max_cache_len=32, num_layers=1)

    k = lucid.randn(2, 8, 2, 64)
    v = lucid.randn(2, 8, 2, 64)
    pos = lucid.Tensor(
        [[0, 5],
         [1, 7]],
        dtype=lucid.Int32,
    )

    cache.update(k, v, layer_idx=0, cache_position=pos)
    print(cache.get_seq_length(0))  # 8

Out-of-bounds behavior
----------------------

`StaticKVCache` raises `ValueError` when writes exceed `max_cache_len`.

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.StaticKVCache(max_cache_len=4, num_layers=1)
    k = lucid.randn(1, 8, 5, 64)
    v = lucid.randn(1, 8, 5, 64)

    # ValueError: exceeded max_cache_len
    cache.update(k, v, layer_idx=0)

Beam search utilities
---------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.StaticKVCache(max_cache_len=64, num_layers=4)
    cache.update(lucid.randn(2, 8, 8, 64), lucid.randn(2, 8, 8, 64), layer_idx=0)

    cache.batch_repeat_interleave(3)  # B=2 -> B=6
    alive = lucid.Tensor([4, 1, 5], dtype=lucid.Int32)
    cache.reorder_cache(alive)        # B=3

Cropping example
----------------

For static cache, `crop` keeps recent valid tokens and updates sequence length.

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.StaticKVCache(max_cache_len=32, num_layers=1)
    cache.update(lucid.randn(1, 8, 20, 64), lucid.randn(1, 8, 20, 64), layer_idx=0)
    cache.crop(8)
    print(cache.get_seq_length(0))  # 8

Practical guidance
------------------

- Choose `max_cache_len` based on your longest expected decode context.
- Use `reset()` between independent requests.
- Use `crop()` when implementing sliding-window style decoding.
