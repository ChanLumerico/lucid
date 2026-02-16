lucid.where
===========

.. autofunction:: lucid.where

The `where` function performs element-wise selection based on a condition tensor, 
returning elements chosen from either `a` or `b` based on whether the condition is true.

Function Signature
------------------

.. code-block:: python

    def where(condition: Tensor, a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **condition** (*Tensor*): 
  A boolean tensor indicating which elements to choose from `a` (if True) or `b` (if False).

- **a** (*Tensor*): Values selected at positions where `condition` is True.
- **b** (*Tensor*): Values selected at positions where `condition` is False.

Returns
-------

- **Tensor**:
  A new tensor composed of values from `a` and `b` selected according to `condition`.

Gradient Calculation
--------------------

.. math::

    \frac{\partial \text{out}}{\partial a} = \text{where}(\text{cond}, \text{grad}, 0) \\
    \frac{\partial \text{out}}{\partial b} = \text{where}(\neg \text{cond}, \text{grad}, 0) \\
    \frac{\partial \text{out}}{\partial \text{cond}} = 0

.. note::

   The gradient for `condition` is always zero since it is not differentiable.
   Gradients for `a` and `b` are masked accordingly.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> cond = lucid.tensor([[True, False], [False, True]])
    >>> a = lucid.tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> b = lucid.tensor([[10, 20], [30, 40]], requires_grad=True)
    >>> out = lucid.where(cond, a, b)
    >>> print(out)
    Tensor([[ 1, 20], [30,  4]], grad=None)
