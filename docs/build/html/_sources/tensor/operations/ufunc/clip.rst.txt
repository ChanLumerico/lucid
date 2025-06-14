lucid.clip
==========

.. autofunction:: lucid.clip

The `clip` function limits the values in the input tensor to a specified range element-wise.

Function Signature
------------------

.. code-block:: python

    def clip(
        a: Tensor,
        min_value: _Scalar | None = None,
        max_value: _Scalar | None = None,
    ) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor whose values are to be clipped.

- **min_value** (*_Scalar | None*, optional):
  The lower bound for clipping. If ``None`` (default), the minimum of
  ``a`` is used.

- **max_value** (*_Scalar | None*, optional):
  The upper bound for clipping. If ``None`` (default), the maximum of
  ``a`` is used.

Returns
-------

- **Tensor**:  
    A new tensor where each element is clipped to the range  
    :math:`[\text{min\_value}, \text{max\_value}]`.  
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for `clip` is defined as:

.. math::

    \mathbf{out}_i =
    \begin{cases} 
      \text{min\_value} & \text{if } \mathbf{a}_i < \text{min\_value} \\
      \mathbf{a}_i & \text{if } \text{min\_value} \leq \mathbf{a}_i \leq \text{max\_value} \\
      \text{max\_value} & \text{if } \mathbf{a}_i > \text{max\_value}
    \end{cases}

Backward Gradient Calculation
-----------------------------

The gradient of the `clip` function with respect to the input tensor **a** is:

.. math::

    \frac{\partial \mathbf{out}_i}{\partial \mathbf{a}_i} =
    \begin{cases} 
      0 & \text{if } \mathbf{a}_i \notin [\text{min\_value}, \text{max\_value}] \\
      1 & \text{if } \mathbf{a}_i \in [\text{min\_value}, \text{max\_value}]
    \end{cases}

This means the gradient is non-zero only for elements within the clipping range.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([-1, 0, 1, 2, 3], requires_grad=True)
    >>> out = lucid.clip(a, min_value=0, max_value=2)
    >>> print(out)
    Tensor([0 0 1 2 2], grad=None)

The function supports tensors of arbitrary shapes:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[-1, 0, 3], [2, -2, 4]], requires_grad=True)
    >>> out = lucid.clip(a, min_value=0, max_value=3)
    >>> print(out)
    Tensor([[0 0 3] [2 0 3]], grad=None)
