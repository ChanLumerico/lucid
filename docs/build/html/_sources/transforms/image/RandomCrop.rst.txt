transforms.RandomCrop
=====================

.. autoclass:: lucid.transforms.RandomCrop

The `RandomCrop` class is a transformation that randomly crops an input tensor 
to the specified dimensions. This transformation is commonly used in data 
augmentation for image-based neural networks, as it introduces positional 
variability to the training samples.

Class Signature
---------------

.. code-block:: python

    class lucid.transforms.RandomCrop(size: tuple[int, int])


Parameters
----------
- **size** (*tuple[int, int]*):
  The target size to which the input tensor will be cropped. The tuple represents 
  the height and width of the cropped output tensor.

Attributes
----------
- **size** (*tuple[int, int]*):
  The size to which the input tensor will be cropped, stored as an internal attribute.

Forward Calculation
-------------------

The cropping of an input tensor :math:`\mathbf{x}` is performed as follows:

1. The dimensions of the input tensor are checked to ensure they are larger than or equal to the specified crop size.
2. Random starting indices :math:`(i, j)` for the height and width are generated to determine the top-left corner of the crop.
3. The sub-tensor of size :math:`(h, w)` is extracted from :math:`\mathbf{x}`, where :math:`h` and :math:`w` are the specified crop sizes.

This process can be represented mathematically as:

.. math::
    
   \mathbf{x}_{\text{cropped}} = \mathbf{x}[:, i:i+h, j:j+w]

Where :math:`i` and :math:`j` are the starting indices for the crop, 
and :math:`h` and :math:`w` are the specified crop sizes.

Examples
--------

**Example 1: Basic Usage**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])  # Shape: (1, 3, 3)
    >>> crop = T.RandomCrop(size=(2, 2))  # Crop to 2x2
    >>> output = crop(input_tensor)
    >>> print(output)
    Tensor([[[4.0, 5.0],
             [7.0, 8.0]]], grad=None)  # Example crop, actual result depends on randomness

**Example 2: Cropping an Image Batch**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                                      [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]])  # Shape: (1, 2, 3, 3)
    >>> crop = T.RandomCrop(size=(2, 2))  # Crop each image in the batch to 2x2
    >>> output = crop(input_tensor)
    >>> print(output)
    Tensor([[[[4.0, 5.0],
              [7.0, 8.0]],
             [[13.0, 14.0],
              [16.0, 17.0]]]], grad=None)  # Example crop, actual result depends on randomness

.. note::

    - The crop size must be smaller than or equal to the dimensions of the input tensor.
    - The random cropping introduces positional variation, which is useful for 
      improving the robustness of neural networks during training.

