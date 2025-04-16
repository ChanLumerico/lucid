nn.Unfold
=========

.. autoclass:: lucid.nn.Unfold

.. versionadded:: 2.0.8

The `Unfold` module extracts sliding local blocks from a batched input tensor 
and flattens them into columns, replicating the behavior of `torch.nn.Unfold`. 
This operation is useful in implementing lower-level convolution operations, 
patch-based models, or custom spatial transformations.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.Unfold(
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...],
        dilation: int | tuple[int, ...] = 1,
    )

Parameters
----------
- **kernel_size** (*int* or *tuple[int, ...]*): 
  The size of the sliding blocks. Can be a single integer or a tuple of integers 
  specifying the size per spatial dimension.

- **stride** (*int* or *tuple[int, ...]*): 
  The stride of the sliding blocks in each spatial dimension. Can be a single 
  integer or a tuple.

- **padding** (*int* or *tuple[int, ...]*): 
  Amount of implicit zero padding on both sides for each spatial dimension.

- **dilation** (*int* or *tuple[int, ...]*, optional): 
  Spacing between elements within the sliding block. Default is `1`.

Forward Calculation
-------------------
Given an input tensor of shape:

.. math::

    (N, C, H_{in}, W_{in})

The output shape will be:

.. math::

    (N, C \cdot K_H \cdot K_W, L)

Where:
- :math:`K_H, K_W` are the height and width of the kernel.
- :math:`L` is the number of sliding blocks extracted from each image, computed as:

.. math::

    L = \left\lfloor \frac{H_{in} + 2 \cdot p_H - d_H (K_H - 1) - 1}{s_H} + 1 \right\rfloor
        \cdot \left\lfloor \frac{W_{in} + 2 \cdot p_W - d_W (K_W - 1) - 1}{s_W} + 1 \right\rfloor

Where :math:`p`, :math:`s`, and :math:`d` are padding, stride, and dilation respectively.

Examples
--------
**Using `Unfold` to extract 3x3 patches:**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], requires_grad=True)
    >>> unfold = nn.Unfold(kernel_size=2, stride=1, padding=0)
    >>> out = unfold(x)
    >>> print(out.shape)
    (1, 4, 4)

**Using `Unfold` with dilation:**

.. code-block:: python

    >>> unfold = nn.Unfold(kernel_size=2, stride=1, padding=0, dilation=2)
    >>> out = unfold(x)
    >>> print(out.shape)
    (1, 4, 1)

**Gradient Backpropagation:**

.. code-block:: python

    >>> out.sum().backward()
    >>> print(x.grad)
    [...]  # Gradient with respect to input

.. note::

  - This module is useful for implementing convolution manually as matrix multiplication.
  
  - You can combine `nn.Unfold` with `lucid.matmul` or `nn.Linear` to perform custom 
    convolution operations.
  
  - For N-dimensional generalization, consider extending this module to support 
    1D/3D unfold as well.
