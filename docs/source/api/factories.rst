Tensor Creation
===============

.. currentmodule:: lucid

Functions for constructing :class:`Tensor` objects from Python scalars,
sequences, or by sampling from distributions.

From data
---------

.. autofunction:: tensor
.. autofunction:: from_numpy
.. autofunction:: from_dlpack

Constant / fill
---------------

.. autofunction:: zeros
.. autofunction:: ones
.. autofunction:: full
.. autofunction:: zeros_like
.. autofunction:: ones_like
.. autofunction:: full_like
.. autofunction:: empty
.. autofunction:: empty_like

Range / grid
------------

.. autofunction:: arange
.. autofunction:: linspace
.. autofunction:: logspace
.. autofunction:: meshgrid

Random
------

.. autofunction:: rand
.. autofunction:: randn
.. autofunction:: randint
.. autofunction:: rand_like
.. autofunction:: randn_like
.. autofunction:: manual_seed
