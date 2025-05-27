nn.functional.affine_grid
=========================

.. autofunction:: lucid.nn.functional.affine_grid

The `affine_grid` function generates a 2D sampling grid, given a batch of 2x3 
affine transformation matrices. This grid can be used with `grid_sample` to 
perform spatial transformations on image tensors.

Function Signature
------------------

.. code-block:: python

    def affine_grid(
        theta: Tensor,
        size: tuple[int, int, int, int],
        align_corners: bool = True
    ) -> Tensor

Parameters
----------

- **theta** (*Tensor*):  
  A tensor of shape `(N, 2, 3)` representing the affine transformation matrices for 
  each sample in the batch.

- **size** (*tuple[int, int, int, int]*):  
  The target output size in the format `(N, C, H, W)`. The function only uses 
  `N`, `H`, and `W`.

- **align_corners** (*bool*, optional):  
  If True, the grid aligns its extrema (-1 and 1) exactly with the corner 
  pixels of the image. If False, it aligns -1 and 1 with the centers of the corner pixels. 
  Default is True.

Returns
-------

- **Tensor**:  
  A tensor of shape `(N, H, W, 2)` representing the sampling grid in normalized 
  coordinates (values range in `[-1, 1]`). This output can be used as the `grid` 
  argument for `grid_sample`.

Forward Calculation
-------------------

The function creates a grid of shape `(H, W, 3)` with 1s in the third channel and 
applies the affine transform:

.. math::

    \mathbf{G}_{n, h, w, :} = \mathbf{A}_n \cdot 
    \begin{bmatrix}
        x \\
        y \\
        1
    \end{bmatrix}

where:

- :math:`\mathbf{G}` is the output grid.
- :math:`\mathbf{A}_n` is the affine matrix for sample :math:`n`, shape `(2, 3)`.
- :math:`x, y` are the normalized coordinates on the output grid (in range `[-1, 1]`).

Backward Gradient Calculation
-----------------------------

Gradients can propagate back through `theta` as long as the resulting grid is passed into  
`grid_sample` and used in a differentiable context. The gradients w.r.t. `theta` will reflect  
how changes in the transform affect the sampled output.

Examples
--------

**Generate a normalized grid for identity affine transform:**

.. code-block:: python

    >>> import lucid
    >>> from lucid import Tensor
    >>> import lucid.nn.functional as F

    >>> theta = Tensor([[[1, 0, 0], [0, 1, 0]]], requires_grad=True)
    >>> grid = F.affine_grid(theta, size=(1, 1, 3, 3))
    >>> print(grid.shape)
    (1, 3, 3, 2)

**Use with `grid_sample`:**

.. code-block:: python

    >>> input_ = Tensor([[[[1.0, 2.0, 3.0],
    ...                    [4.0, 5.0, 6.0],
    ...                    [7.0, 8.0, 9.0]]]])
    >>> out = F.grid_sample(input_, grid, mode='bilinear', align_corners=True)
    >>> print(out.shape)
    (1, 1, 3, 3)

**Backpropagate through affine transform:**

.. code-block:: python

    >>> loss = out.sum()
    >>> loss.backward()
    >>> print(theta.grad)
    # Should return non-zero gradients with respect to translation
