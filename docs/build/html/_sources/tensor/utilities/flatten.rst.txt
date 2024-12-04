lucid.flatten
==============

.. autofunction:: lucid.flatten

The `flatten` function flattens the input tensor into a one-dimensional tensor.

Function Signature
------------------

.. code-block:: python

    def flatten(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*):
    The input tensor of any shape to be flattened.

Returns
-------

- **Tensor**:
    A one-dimensional tensor containing all the elements of the input tensor.

Examples
--------

Flattening a 2D tensor:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
    >>> out = lucid.flatten(a)
    >>> print(out)
    Tensor([1, 2, 3, 4])

