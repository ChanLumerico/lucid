lucid.concatenate
==================

.. autofunction:: lucid.concatenate

The `concatenate` function joins a sequence of tensors along an existing dimension. 
It concatenates the input tensors along the specified axis, 
combining them into a single tensor.

Function Signature
------------------

.. code-block:: python

    def concatenate(arr: tuple[Tensor, ...], axis: int = 0) -> Tensor

Parameters
----------

- **arr** (*tuple[Tensor, ...]*): 
    A sequence of tensors to concatenate. All tensors must have the same shape 
    except in the dimension specified by `axis`.
    
- **axis** (*int*, optional): 
    The axis along which the tensors will be concatenated. Defaults to `0`.

Returns
-------

- **Tensor**: 
    A new tensor resulting from concatenating the input tensors along the specified axis. 
    The resulting tensor has the same number of dimensions as the input tensors, 
    with the size along the `axis` dimension being the sum of the sizes of the input 
    tensors along that axis.

Forward Calculation
-------------------

The `concatenate` operation joins the input tensors along the specified 
existing dimension `axis`.

.. math::

    \mathbf{out} = \text{concatenate}(\mathbf{a}_1, 
    \mathbf{a}_2, \dots, \mathbf{a}_n; \text{axis} = k)

Where each \(\mathbf{a}_i\) is a tensor and \(k\) is the axis along which concatenation occurs.

Backward Gradient Calculation
-----------------------------

The gradient for the concatenate operation is split and passed to 
each of the input tensors along the concatenated axis.

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}_i} = \mathbf{I}_i

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0]], requires_grad=True)
    >>> b = Tensor([[3.0, 4.0]], requires_grad=True)
    >>> concatenated = lucid.concatenate((a, b), axis=0)
    >>> print(concatenated)
    Tensor([[1. 2.],
            [3. 4.]], grad=None)
    
    >>> concatenated = lucid.concatenate((a, b), axis=1)
    >>> print(concatenated)
    Tensor([[1. 2. 3. 4.]], grad=None)

.. note::

    - All input tensors must have the same shape except in the dimension specified by `axis`.
    - The `axis` parameter specifies the dimension along which the tensors will be joined.
    - The resulting tensor will have the same number of dimensions as the input tensors.
    - If any of the input tensors require gradients, the resulting tensor will also require gradients.
    - Negative values for `axis` are supported and are interpreted as counting 
      dimensions from the end (e.g., `-1` refers to the last dimension).