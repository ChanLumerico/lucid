lucid.einops
============

.. currentmodule:: lucid.einops

Einstein-notation tensor operations.  Provides concise, readable
rearrangements without materialising intermediate tensors.

.. note::

   These operations are only accessible via the ``lucid.einops`` namespace.
   No ``lucid.einsum`` / ``lucid.rearrange`` shortcuts exist at the top level.

.. autofunction:: rearrange
.. autofunction:: reduce
.. autofunction:: repeat
.. autofunction:: einsum
.. autofunction:: asnumpy
.. autofunction:: parse_shape
