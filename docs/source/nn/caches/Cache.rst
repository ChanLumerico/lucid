nn.Cache
========

.. autoclass:: lucid.nn.Cache

Overview
--------

`Cache` is the top-level abstract interface for cache objects in `lucid.nn`.
It standardizes cache lifecycle and batch manipulation APIs that can be shared
across cache families.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.Cache()

All concrete cache classes must implement:

- `reset()`
- `batch_select_indices(indices)`
- `batch_repeat_interleave(repeats)`
- `crop(max_length)`

Common public methods
---------------------

`Cache` provides:

- `reorder_cache(beam_idx)`:
  Delegates to `batch_select_indices(...)` for beam-style reindexing.
- `get_max_cache_shape()`:
  Returns `None` by default. Subclasses can override with a fixed capacity.

Design note
-----------

Use `Cache` as the broad API contract when your component should accept
multiple cache implementations. Use specialized subclasses (for example
`KVCache`) when your logic depends on domain-specific methods.
