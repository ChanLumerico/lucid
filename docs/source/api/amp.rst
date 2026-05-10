lucid.amp
=========

.. currentmodule:: lucid.amp

Automatic Mixed Precision (AMP) — runs forward passes in ``bfloat16`` or
``float16`` while keeping master weights in ``float32``.

Context manager
---------------

.. autoclass:: autocast
   :members:
   :undoc-members:
   :show-inheritance:

Gradient scaler
---------------

.. autoclass:: GradScaler
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. autofunction:: is_autocast_enabled
.. autofunction:: get_autocast_dtype
