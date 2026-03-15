nn.functional.interpolate
=========================

.. autofunction:: lucid.nn.functional.interpolate

The `interpolate` function resizes an input tensor to a specified size using different
interpolation modes. It supports both 2D image tensors :math:`(N, C, H, W)` and 3D
volumetric tensors :math:`(N, C, D, H, W)`, with modes including bilinear, trilinear,
nearest, and area interpolation.

Function Signature
------------------

.. code-block:: python

    def interpolate(
        input_: Tensor,
        size: tuple[int, int] | tuple[int, int, int],
        mode: str = 'bilinear',
        align_corners: bool = False,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
    The input tensor of shape `(N, C, H, W)` for 2D interpolation or
    `(N, C, D, H, W)` for 3D (trilinear) interpolation.

- **size** (*tuple[int, int] | tuple[int, int, int]*):
    The target output spatial size. Pass `(height, width)` for 2D modes or
    `(depth, height, width)` for trilinear mode.

- **mode** (*str*, optional):
    The interpolation mode to use. Options are `'bilinear'`, `'trilinear'`,
    `'nearest'`, and `'area'`. Default is `'bilinear'`.

- **align_corners** (*bool*, optional):
    If True, aligns the corners of the input and output tensors.
    Relevant for `'bilinear'` and `'trilinear'` modes. Default is False.

Returns
-------

- **Tensor**:
    A new `Tensor` containing the result of the interpolation. For 2D modes the
    output shape is `(N, C, size[0], size[1])`; for trilinear mode it is
    `(N, C, size[0], size[1], size[2])`. If the input requires gradients,
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `interpolate` operation varies depending on the selected mode:

- **Bilinear Interpolation** (2D):

  .. math::

      \mathbf{out}_{ij} = (1 - h_{lerp}) (1 - w_{lerp}) \cdot \mathbf{I}_{top\_left}
      + h_{lerp} (1 - w_{lerp}) \cdot \mathbf{I}_{bottom\_left}
      + (1 - h_{lerp}) w_{lerp} \cdot \mathbf{I}_{top\_right}
      + h_{lerp} w_{lerp} \cdot \mathbf{I}_{bottom\_right}

  Where :math:`h_{lerp}` and :math:`w_{lerp}` are the interpolation coefficients along
  the height and width dimensions.

- **Trilinear Interpolation** (3D):

  Linear interpolation extended to three spatial dimensions. For each output voxel,
  the eight surrounding input voxels are gathered and blended using three independent
  lerp weights :math:`d_{lerp}`, :math:`h_{lerp}`, :math:`w_{lerp}`:

  .. math::

      \mathbf{out}_{ijk} = \sum_{a,b,c \;\in\; \{0,1\}}
          w^{d}_{a} \cdot w^{h}_{b} \cdot w^{w}_{c}
          \cdot \mathbf{I}_{d_a,\, h_b,\, w_c}

  where :math:`w^{d}_0 = 1 - d_{lerp}`,  :math:`w^{d}_1 = d_{lerp}`, and similarly
  for the height and width axes.

- **Nearest Neighbor Interpolation**:

  The value from the nearest neighbor in the input is assigned to the corresponding
  location in the output. Supports both 4D and 5D input tensors.

- **Area Interpolation**:

  The output pixel is computed as the average of input values within
  the corresponding region (2D only).

Backward Gradient Calculation
------------------------------

For tensors **input_** involved in the `interpolate` operation,
the gradients with respect to the output (**out**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

- **Bilinear Interpolation**: Gradients are propagated back according to the interpolation
  weights used in the forward pass.
- **Trilinear Interpolation**: Gradients are propagated back according to all eight
  corner weights, following the trilinear blending formula.
- **Nearest Neighbor Interpolation**: The gradient is passed directly to the nearest neighbor
  used in the forward pass.
- **Area Interpolation**: The gradient is distributed equally to all input pixels
  that contributed to the corresponding output pixel.

Examples
--------

**Using `interpolate` with bilinear interpolation:**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)  # Shape: (1, 1, 2, 2)
    >>> out = F.interpolate(input_, size=(4, 4), mode='bilinear', align_corners=True)  # Shape: (1, 1, 4, 4)
    >>> print(out)
    Tensor([[[[1.0, 1.5, 2.0, 2.0],
              [2.0, 2.5, 3.0, 3.0],
              [3.0, 3.5, 4.0, 4.0],
              [3.0, 3.5, 4.0, 4.0]]]])

Backpropagation propagates gradients through the input:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    # Gradient values corresponding to bilinear interpolation

**Using `interpolate` with trilinear interpolation:**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> import lucid
    >>> # Shape: (1, 1, 2, 2, 2) — batch=1, channels=1, depth=2, height=2, width=2
    >>> input_ = lucid.random.randn(1, 1, 2, 2, 2)
    >>> out = F.interpolate(input_, size=(4, 4, 4), mode='trilinear')  # Shape: (1, 1, 4, 4, 4)
    >>> print(out.shape)
    (1, 1, 4, 4, 4)

**Using `interpolate` with nearest neighbor interpolation:**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)  # Shape: (1, 1, 2, 2)
    >>> out = F.interpolate(input_, size=(4, 4), mode='nearest')  # Shape: (1, 1, 4, 4)
    >>> print(out)
    Tensor([[[[1.0, 1.0, 2.0, 2.0],
              [1.0, 1.0, 2.0, 2.0],
              [3.0, 3.0, 4.0, 4.0],
              [3.0, 3.0, 4.0, 4.0]]]])

Backpropagation propagates gradients through the input:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    # Gradient values corresponding to nearest interpolation

**Using `interpolate` with area interpolation:**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)  # Shape: (1, 1, 2, 2)
    >>> out = F.interpolate(input_, size=(1, 1), mode='area')  # Shape: (1, 1, 1, 1)
    >>> print(out)
    Tensor([[[[2.5]]]])
