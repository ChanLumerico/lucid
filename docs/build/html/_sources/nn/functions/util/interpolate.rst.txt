nn.functional.interpolate
=========================

.. autofunction:: lucid.nn.functional.interpolate

The `interpolate` function resizes an input tensor to a specified size using different 
interpolation modes. It supports multiple modes, including bilinear, nearest, and area 
interpolation.

Function Signature
------------------

.. code-block:: python

    def interpolate(
        input_: Tensor, size: tuple[int, int], mode: str = 'bilinear', align_corners: bool = False
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape `(N, C, H, W)`, where `N` is the batch size, 
    `C` is the number of channels, `H` is the height, and `W` is the width.
    
- **size** (*tuple[int, int]*): 
    The target output size as `(height, width)`. 
    
- **mode** (*str*, optional): 
    The interpolation mode to use. Options are `'bilinear'`, `'nearest'`, and `'area'`. 
    Default is `'bilinear'`.
    
- **align_corners** (*bool*, optional): 
    If True, aligns the corners of the input and output tensors. 
    This parameter is only relevant for `'bilinear'` mode. Default is False.

Returns
-------

- **Tensor**: 
    A new `Tensor` containing the result of the interpolation. The shape of the output tensor 
    is `(N, C, size[0], size[1])`. If the input requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `interpolate` operation varies depending on the selected mode:

- **Bilinear Interpolation**:

  .. math::
      
      \mathbf{out}_{ij} = (1 - h_lerp) (1 - w_lerp) \cdot \mathbf{I}_{top\_left} 
      + h_lerp (1 - w_lerp) \cdot \mathbf{I}_{bottom\_left} 
      + (1 - h_lerp) w_lerp \cdot \mathbf{I}_{top\_right} 
      + h_lerp w_lerp \cdot \mathbf{I}_{bottom\_right}

  Where \( h_{lerp} \) and \( w_{lerp} \) are the interpolation coefficients along 
  the height and width dimensions.

- **Nearest Neighbor Interpolation**:
  
  The value from the nearest neighbor in the input is assigned to the corresponding 
  location in the output.
  
- **Area Interpolation**:
  
  The output pixel is computed as the average of input values within 
  the corresponding region.

Backward Gradient Calculation
-----------------------------

For tensors **input_** involved in the `interpolate` operation, 
the gradients with respect to the output (**out**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

- **Bilinear Interpolation**: Gradients are propagated back according to the interpolation 
  weights used in the forward pass.
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
