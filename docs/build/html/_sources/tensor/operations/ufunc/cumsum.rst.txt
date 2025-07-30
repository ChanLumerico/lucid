lucid.cumsum
============

.. autofunction:: lucid.cumsum

The `cumsum` function computes the cumulative sum (inclusive prefix sum) 
of elements in the input tensor along a specified axis.

Function Signature
------------------

.. code-block:: python

    def cumsum(a: Tensor, axis: int = -1) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor whose cumulative sums are to be computed.  
- **axis** (*int*, optional): The axis along which to perform the cumulative sum. 
  Defaults to the last axis (:math:`-1`).

Returns
-------

- **Tensor**  
  A tensor of the same shape as **a**, where each element at position :math:`i` 
  along the specified axis is the sum of all elements from the start of that axis 
  up to :math:`i`. If **a** requires gradients, the returned tensor will also require 
  gradients.

Forward Calculation
-------------------

For each index :math:`k` along the chosen axis:

.. math::

   \mathrm{out}_{k} = \sum_{j=0}^{k} a_{j}

where :math:`a_{j}` are the elements of **a** along the specified axis.

Backward Gradient Calculation
-----------------------------

The Jacobian of the cumulative-sum operation yields, for an upstream gradient 
:math:`\nabla \mathrm{out}`:

.. math::

   \frac{\partial \mathrm{out}_{k}}{\partial a_{i}}
   =
   \begin{cases}
     1, & i \le k,\\
     0, & i > k.
   \end{cases}

Hence the gradient w.r.t. each input element :math:`a_{i}` is

.. math::

   \nabla a_{i}
   = \sum_{k=i}^{n-1} \nabla \mathrm{out}_{k}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 3, 4], requires_grad=True)
    >>> out = lucid.cumsum(a, axis=0)
    >>> print(out)
    Tensor([ 1  3  6 10], grad=None)
    >>> s = out.sum()
    >>> s.backward()
    >>> print(a.grad)
    [4., 3., 2., 1.]

.. note::

   When **axis** is negative, it counts from the last dimension 
   (e.g., :math:`axis=-1` refers to the final axis of **a**).
