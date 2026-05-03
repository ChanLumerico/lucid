lucid.autograd
==============

.. currentmodule:: lucid.autograd

Gradient computation
--------------------

.. autofunction:: backward
.. autofunction:: grad

Gradient mode
-------------

.. autoclass:: no_grad
   :members:
.. autoclass:: enable_grad
   :members:
.. autofunction:: set_grad_enabled
.. autofunction:: is_grad_enabled
.. autofunction:: inference_mode

Custom functions
----------------

.. autoclass:: Function
   :members:

.. autoclass:: FunctionCtx
   :members:

Gradient checking
-----------------

.. automodule:: lucid.autograd.gradcheck
   :members:
