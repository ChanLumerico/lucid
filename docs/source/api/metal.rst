Metal (GPU)
===========

.. currentmodule:: lucid

Lucid's GPU backend uses Apple's Metal Performance Shaders (MPS) layer
via the MLX library. Operations dispatched to ``"metal"`` are evaluated
lazily — no computation runs until :func:`lucid.eval` (or an implicit
sync point such as ``.item()`` or ``.numpy()``) is called.

Device API
----------

.. autoclass:: device
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: eval

Metal synchronisation
---------------------

.. autofunction:: synchronize
.. autofunction:: is_available

Memory utilities
----------------

.. autofunction:: metal_memory_allocated
.. autofunction:: metal_max_memory_allocated
.. autofunction:: reset_peak_memory_stats

Stream API
----------

.. autoclass:: Stream
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: current_stream
.. autofunction:: default_stream

Example
-------

.. code-block:: python

   import lucid

   x = lucid.randn(1024, 1024, device="metal")
   y = x @ x.T
   lucid.eval(y)         # flush lazy Metal graph
   print(y.shape)        # (1024, 1024)
