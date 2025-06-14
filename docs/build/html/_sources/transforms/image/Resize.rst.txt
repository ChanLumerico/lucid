transforms.Resize
=================

.. autoclass:: lucid.transforms.Resize

The `Resize` class is a transformation that resizes an input tensor to 
the specified dimensions. This transformation is useful for ensuring that 
input tensors have a consistent size, which is often required in image 
processing and neural network applications.

Class Signature
---------------

.. code-block:: python

    class lucid.transforms.Resize(size: tuple[int, int])


Parameters
----------
- **size** (*tuple[int, int]*):
  The target size to which the input tensor will be resized. 
  The tuple represents the height and width of the output tensor.

Attributes
----------
- **size** (*tuple[int, int]*):
  The size to which the input tensor will be resized, stored as an internal attribute.

Forward Calculation
-------------------

The resizing of an input tensor :math:`\mathbf{x}` is performed using 
interpolation or down-sampling techniques, depending on the relationship between 
the original size and the target size.

If the target size is larger than the original size, up-sampling is performed.

If the target size is smaller than the original size, down-sampling is performed.

The transformation is applied independently for each channel.

Examples
--------

**Example 1: Basic Usage**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[1.0, 2.0], [3.0, 4.0]]])  # Shape: (1, 2, 2)
    >>> resize = T.Resize(size=(4, 4))  # Resize to 4x4
    >>> output = resize(input_tensor)
    >>> print(output)
    Tensor([[[1.0, 1.5, 2.0, 2.0],
             [2.0, 2.5, 3.0, 3.0],
             [3.0, 3.5, 4.0, 4.0],
             [3.0, 3.5, 4.0, 4.0]]], grad=None)

**Example 2: Resizing an Image Batch**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])  # Shape: (1, 2, 2, 2)
    >>> resize = T.Resize(size=(4, 4))  # Resize each image in the batch to 4x4
    >>> output = resize(input_tensor)
    >>> print(output)
    Tensor([[[[1.0, 1.5, 2.0, 2.0],
              [2.0, 2.5, 3.0, 3.0],
              [3.0, 3.5, 4.0, 4.0],
              [3.0, 3.5, 4.0, 4.0]],

             [[5.0, 5.5, 6.0, 6.0],
              [6.0, 6.5, 7.0, 7.0],
              [7.0, 7.5, 8.0, 8.0],
              [7.0, 7.5, 8.0, 8.0]]]], grad=None)

.. note::

    - The resize operation maintains the aspect ratio if both dimensions 
      are scaled proportionally.
    - This transformation is typically applied to images to ensure a 
      consistent input size for neural networks.
