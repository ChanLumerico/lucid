lucid.cumprod
=============

.. autofunction:: lucid.cumprod

The `cumprod` function computes the cumulative product (inclusive prefix product) 
of elements in the input tensor along a specified axis.

Function Signature
------------------

.. code-block:: python

    def cumprod(a: Tensor, axis: int = -1) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor whose cumulative products are to be computed.  
- **axis** (*int*, optional): The axis along which to perform the cumulative product. 
  Defaults to the last axis (:math:`-1`).

Returns
-------

- **Tensor**  
  A tensor of the same shape as **a**, where each element at position :math:`i` 
  along the specified axis is the product of all elements from the start of that 
  axis up to :math:`i`. If **a** requires gradients, the returned tensor will also 
  require gradients.

Forward Calculation
-------------------

For each index :math:`k` along the chosen axis:

.. math::

   \mathrm{out}_{k} = \prod_{j=0}^{k} a_{j}

where :math:`a_{j}` are the elements of **a** along the specified axis.

Backward Gradient Calculation
-----------------------------

The Jacobian of the cumulative-product operation yields, for an upstream gradient 
:math:`\nabla \mathrm{out}`:

.. math::

   \frac{\partial \mathrm{out}_{k}}{\partial a_{i}}
   = 
   \begin{cases}
     \displaystyle \prod_{\substack{j=0 \\ j \neq i}}^{k} a_{j}, & i \le k,\\
     0, & i > k.
   \end{cases}

Hence the gradient w.r.t. each input element :math:`a_{i}` is

.. math::

   \nabla a_{i}
   = \sum_{k=i}^{n-1} \nabla \mathrm{out}_{k} \times
     \prod_{\substack{j=0 \\ j \neq i}}^{k} a_{j}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 3, 4], requires_grad=True)
    >>> out = lucid.cumprod(a, axis=0)
    >>> print(out)
    Tensor([ 1  2  6 24], grad=None)
    >>> s = out.sum()
    >>> s.backward()
    >>> print(a.grad)
    [33., 16., 10.,  6.]

.. note::

   When **axis** is negative, it counts from the last dimension 
   (e.g., :math:`axis=-1` refers to the final axis of **a**).
