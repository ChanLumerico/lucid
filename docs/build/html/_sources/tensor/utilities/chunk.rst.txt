lucid.chunk
===========

.. autofunction:: lucid.chunk

The `chunk` function splits a tensor into multiple sub-tensors along a specified axis.
If the tensor cannot be evenly divided, the last chunk will be smaller.

Function Signature
------------------

.. code-block:: python

    def chunk(input_: Tensor, chunks: int, axis: int = 0) -> tuple[Tensor, ...]:

Parameters
----------
- **input_** (*Tensor*):
  The input tensor to be split.

- **chunks** (*int*):
  The number of chunks to split the tensor into. Must be greater than 0.

- **axis** (*int*, optional):
  The dimension along which to split the tensor. Default is `0`.

Returns
-------
- **tuple[Tensor, ...]**:
  A tuple containing `chunks` number of tensors, 
  where each tensor is a portion of `input_` along the specified axis.

Gradient Computation
--------------------
Each returned tensor retains a corresponding `compute_grad` function, 
which ensures that gradients are correctly mapped back to their original positions 
in `input_`.

Example
-------

Splitting a tensor into two equal parts along the first axis:

.. code-block:: python

    >>> import lucid
    >>> x = lucid.ones(4, 6, requires_grad=True)
    >>> y1, y2 = x.chunk(2, axis=0)
    >>> y1 *= 2
    >>> y2 *= 3
    >>> y1.backward()
    >>> y2.backward()
    >>> print(x.grad)
    [[2. 2. 2. 2. 2. 2.]
     [2. 2. 2. 2. 2. 2.]
     [3. 3. 3. 3. 3. 3.]
     [3. 3. 3. 3. 3. 3.]]

.. note::

    - If `chunks` is larger than the size of the axis being split, 
      some returned tensors may be empty.

    - This function ensures proper gradient propagation, 
      preserving the backpropagation mechanism in `lucid`.
