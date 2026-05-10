lucid.Tensor
============

.. currentmodule:: lucid

The :class:`Tensor` class is the central data structure in Lucid — a
multi-dimensional array that lives on either the CPU (Accelerate backend)
or the Metal GPU (MLX backend).  Every forward computation and gradient
is represented as a ``Tensor``.

.. autoclass:: Tensor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __add__, __sub__, __mul__, __truediv__,
                     __neg__, __pow__, __matmul__, __getitem__, __setitem__,
                     __len__, __repr__, __iter__
