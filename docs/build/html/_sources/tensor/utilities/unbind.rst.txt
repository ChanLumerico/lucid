lucid.unbind
============

.. autofunction:: lucid.unbind

The `unbind` function removes a specified axis by slicing the tensor into 
multiple sub-tensors along that axis. It is functionally equivalent to 
splitting and squeezing the axis.

Function Signature
------------------

.. code-block:: python

    def unbind(a: Tensor, /, axis: int = 0) -> tuple[Tensor, ...]

Parameters
----------

- **a** (*Tensor*):  
  The input tensor to be unbound.

- **axis** (*int*, optional):  
  The axis along which to unbind. Default is `0`.

Returns
-------

- **tuple[Tensor, ...]**:  
  A tuple of sub-tensors where each tensor corresponds to one slice along the specified axis.  
  Each output tensor will have one fewer dimension than the input tensor.

.. tip::

    This operation is similar to `split` followed by `squeeze` on the target axis.

Backward Gradient Calculation
-----------------------------

For input tensor :math:`\mathbf{a}` unbound along an axis, and each output 
:math:`\mathbf{y}_i`, the gradient is propagated by embedding :math:`\nabla \mathbf{y}_i` 
back into its slice:

.. math::

    \nabla \mathbf{a}_{\text{slice}_i} = \nabla \mathbf{y}_i

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    >>> y0, y1, y2 = lucid._util.unbind(x, axis=0)
    >>> print(y0)
    Tensor([1, 2], grad=None)
    >>> y0.backward()
    >>> print(x.grad)
    [[1. 1.]
     [0. 0.]
     [0. 0.]]

.. warning::

    The returned tensors are views and not copies. 
    Modifying them directly may interfere with gradient propagation.
