lucid.autograd
==============

.. currentmodule:: lucid

Lucid uses dynamic reverse-mode automatic differentiation.  Every
:class:`Tensor` with ``requires_grad=True`` records operations in a
computation graph; calling ``.backward()`` traverses that graph to
accumulate gradients.

Context managers
----------------

.. autofunction:: no_grad
.. autofunction:: enable_grad
.. autofunction:: set_grad_enabled

Custom functions
----------------

.. currentmodule:: lucid.autograd

.. autoclass:: Function
   :members: forward, backward, apply
   :undoc-members:
   :show-inheritance:

Gradient utilities
------------------

.. autofunction:: lucid.autograd.grad
.. autofunction:: lucid.autograd.checkpoint
.. autofunction:: lucid.autograd.detect_anomaly
