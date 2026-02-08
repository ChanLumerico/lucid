nn.DynamicKVCache
=================

.. autoclass:: lucid.nn.DynamicKVCache

Overview
--------

`DynamicKVCache` grows cache length dynamically. This is convenient for
variable-length decoding and quick experimentation.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.DynamicKVCache()

Key properties
--------------

- No predefined max cache length.
- Sequence dimension grows as updates arrive.
- Supports `cache_position` modes: `None`, 0-D, 1-D, 2-D.

Basic append example
--------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()

    k1 = lucid.randn(2, 8, 1, 64)
    v1 = lucid.randn(2, 8, 1, 64)
    cache.update(k1, v1, layer_idx=0)

    k2 = lucid.randn(2, 8, 1, 64)
    v2 = lucid.randn(2, 8, 1, 64)
    cache.update(k2, v2, layer_idx=0)

    print(cache.get_seq_length(0))  # 2
    k_all, v_all = cache.get(0)
    print(k_all.shape)              # (2, 8, 2, 64)

Scalar position update (0-D)
----------------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()
    k = lucid.randn(1, 8, 1, 64)
    v = lucid.randn(1, 8, 1, 64)

    # write token to position 5
    cache.update(
        k, v,
        layer_idx=0,
        cache_position=lucid.Tensor(5, dtype=lucid.Int32),
    )
    print(cache.get_seq_length(0))  # 6 (auto-expanded)

Vector position update (1-D)
----------------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()
    k = lucid.randn(1, 8, 3, 64)
    v = lucid.randn(1, 8, 3, 64)
    pos = lucid.Tensor([0, 2, 4], dtype=lucid.Int32)

    cache.update(k, v, layer_idx=0, cache_position=pos)
    print(cache.get_seq_length(0))  # 5

Batch-wise position update (2-D)
--------------------------------

Useful when each sample in the batch writes to different positions.

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()

    # batch=2, seq_len=2
    k = lucid.randn(2, 8, 2, 64)
    v = lucid.randn(2, 8, 2, 64)
    pos = lucid.Tensor(
        [[0, 2],   # sample 0 writes to positions 0 and 2
         [1, 3]],  # sample 1 writes to positions 1 and 3
        dtype=lucid.Int32,
    )

    cache.update(k, v, layer_idx=0, cache_position=pos)
    print(cache.get_seq_length(0))  # 4

Multi-layer example
-------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()

    for layer_idx in range(6):
        k = lucid.randn(1, 8, 1, 64)
        v = lucid.randn(1, 8, 1, 64)
        cache.update(k, v, layer_idx=layer_idx)

    print(cache.get_seq_length(0))  # 1
    print(cache.get_seq_length(5))  # 1

Beam search utilities
---------------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()
    cache.update(lucid.randn(2, 8, 4, 64), lucid.randn(2, 8, 4, 64), layer_idx=0)

    # B=2 -> B*4=8 beams
    cache.batch_repeat_interleave(4)

    # pick/reorder surviving beams
    beam_idx = lucid.Tensor([7, 2, 5, 1], dtype=lucid.Int32)
    cache.reorder_cache(beam_idx)

Crop example
------------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    cache = nn.DynamicKVCache()
    cache.update(lucid.randn(1, 8, 16, 64), lucid.randn(1, 8, 16, 64), layer_idx=0)
    cache.crop(8)
    print(cache.get_seq_length(0))  # 8

Notes
-----

- `DynamicKVCache` is often easier to integrate first.
- For strict upper-bound memory control, prefer `StaticKVCache`.
