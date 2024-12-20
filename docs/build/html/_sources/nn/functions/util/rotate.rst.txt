nn.functional.rotate
====================

.. autofunction:: lucid.nn.functional.rotate

The `rotate` function rotates an input tensor (typically an image) by a specified angle 
around a specified center point. This is a common operation in image processing and data 
augmentation for training neural networks.

Function Signature
------------------

.. code-block:: python

    def rotate(x: Tensor, angle: float, center: tuple[float, float] | None = None) -> Tensor

Parameters
----------

- **x** (*Tensor*):
  The input image tensor of shape `(N, C, H, W)`, where `N` is the batch size, `C` is the number 
  of channels, `H` is the height, and `W` is the width.

- **angle** (*float*):
  The rotation angle in degrees. Positive values rotate counterclockwise, 
  and negative values rotate clockwise.

- **center** (*tuple[float, float] | None*, optional):
  The center of rotation `(x, y)` in the image. If `None`, the center of the image 
  (`(W/2, H/2)`) is used as the rotation center. Default is `None`.

Returns
-------

- **Tensor**:
  The rotated image tensor with the same shape `(N, C, H, W)` as the input tensor `x`.

Usage
-----

The `rotate` function is commonly used in image preprocessing and data augmentation 
to improve the robustness of neural network models. It allows images to be randomly 
or deterministically rotated during training.

Example Usage
-------------

**Rotating an image by 45 degrees (center default to image center)**

.. code-block:: python

    import lucid.nn.functional as F
    
    x = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    rotated_image = F.rotate(x, angle=45)
    print(rotated_image)

**Rotating an image by -30 degrees around a custom center point (2, 2)**

.. code-block:: python

    import lucid.nn.functional as F
    
    x = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    rotated_image = F.rotate(x, angle=-30, center=(2, 2))
    print(rotated_image)

.. note::

    - **Shape Preservation**: The shape of the output tensor is identical to the input tensor. 
      The content of the image is rotated within its original size, which may result in 
      some content being clipped.

    - **Center of Rotation**: By default, the rotation is centered at `(W/2, H/2)`. If a custom center 
      is provided, the rotation occurs around that point.

    - **Batch Support**: The `rotate` function operates on batches of images, so it can process 
      multiple images simultaneously if `N > 1`.

    - **Data Augmentation**: This function is commonly used for data augmentation to increase 
      the diversity of training datasets and improve model generalization.
