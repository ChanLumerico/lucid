lucid.unique
============

.. autofunction:: lucid.unique

The `unique` function returns the unique elements of the input tensor.  
It optionally sorts the result or preserves the order of first appearance.

Function Signature
------------------

.. code-block:: python

    def unique(a: Tensor, sorted: bool = True, axis: int | None = None) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor from which to extract unique values.

- **sorted** (*bool*, optional):  
  Whether to return the sorted unique values. Defaults to `True`.

- **axis** (*int or None*, optional):  
  The axis along which to apply uniqueness. If `None`, 
  the tensor is flattened. Defaults to `None`.

Returns
-------

- **Tensor**:  
  A tensor of unique elements along the specified axis.  
  If `axis=None`, the result is 1D. Otherwise, 
  it will have shape determined by unique slices.

  The result does not track gradients and has the same dtype as the input.

.. note::

   This operation is non-differentiable. Gradients will not be propagated through `unique()`.

.. tip::

   The default behavior returns sorted unique values.  
   To preserve the original order of appearance, set `sorted=False`.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.tensor([3, 1, 2, 3, 4, 2])
    >>> lucid.unique(x)
    Tensor([1, 2, 3, 4], grad=None)

    >>> lucid.unique(x, sorted=False)
    Tensor([3, 1, 2, 4], grad=None)

    >>> y = lucid.tensor([[1, 2], [2, 1], [3, 4]])
    >>> lucid.unique(y, axis=0)
    Tensor([[1, 2],
            [2, 1],
            [3, 4]], grad=None)
