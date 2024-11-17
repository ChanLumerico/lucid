lucid.ravel
===========

.. autofunction:: lucid.ravel

The `ravel` function flattens a tensor into a one-dimensional array. 
It is equivalent to reshaping the tensor into a contiguous 1D array.

Function Signature
------------------

.. code-block:: python

    def ravel(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The input tensor to flatten.

Returns
-------

- **Tensor**: 
    A one-dimensional tensor with the same data as the original. 
    If **a** requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The `ravel` operation flattens the input tensor into a 1D tensor.

.. math::

    \mathbf{out} = \text{ravel}(\mathbf{a})

Backward Gradient Calculation
-----------------------------

The gradient for the ravel operation is passed through unchanged as it 
only affects the shape, not the tensor values.

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \mathbf{I}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> flattened = lucid.ravel(a)  # or a.ravel()
    >>> print(flattened)
    Tensor([1. 2. 3. 4.], grad=None)

.. note::

    - The `ravel` operation returns a flattened tensor without changing the underlying data.
    - The resulting tensor will always be one-dimensional, regardless of the original shape.
