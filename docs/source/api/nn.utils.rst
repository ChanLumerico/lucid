lucid.nn.utils
==============

.. currentmodule:: lucid.nn.utils

Utilities for inspecting and manipulating :class:`~lucid.nn.Module`
parameters and gradients.

Gradient clipping
-----------------

.. autofunction:: clip_grad_norm_
.. autofunction:: clip_grad_value_

Parameter utilities
-------------------

.. autofunction:: parameters_to_vector
.. autofunction:: vector_to_parameters

Weight initialisation
---------------------

.. currentmodule:: lucid.nn.init

.. autofunction:: uniform_
.. autofunction:: normal_
.. autofunction:: constant_
.. autofunction:: ones_
.. autofunction:: zeros_
.. autofunction:: eye_
.. autofunction:: xavier_uniform_
.. autofunction:: xavier_normal_
.. autofunction:: kaiming_uniform_
.. autofunction:: kaiming_normal_
.. autofunction:: orthogonal_
.. autofunction:: trunc_normal_
.. autofunction:: calculate_gain
