transforms.RandomRotation
=========================

.. autoclass:: lucid.transforms.RandomRotation

The `RandomRotation` class is a transformation that randomly rotates an input tensor 
by a specified range of degrees. This transformation is commonly used in data augmentation 
to introduce rotational variability in training samples for image-based neural networks.

Class Signature
---------------

.. code-block:: python

    class lucid.transforms.RandomRotation(degrees: float)


Parameters
----------
- **degrees** (*float*):
  The maximum range of rotation in degrees. The input tensor will be randomly 
  rotated by an angle drawn from the range [-degrees, degrees].

Attributes
----------
- **degrees** (*float*):
  The maximum degree range within which the input tensor will be randomly rotated, 
  stored as an internal attribute.

Forward Calculation
-------------------

The random rotation of an input tensor :math:`\mathbf{x}` is applied as follows:

1. A random angle :math:`\theta` is sampled from the uniform distribution in the range 
   :math:`[-d, d]`, where :math:`d` is the specified degree range.
2. The input tensor is rotated counterclockwise by :math:`\theta` degrees. 
   The rotation is typically applied using affine transformations for 2D data.

The rotation can be represented mathematically as:

.. math::
   \mathbf{x}_{\text{rotated}} = R(\mathbf{x}, \theta)

Where :math:`R(\mathbf{x}, \theta)` represents the rotation of tensor :math:`\mathbf{x}` 
by an angle :math:`\theta`.

Examples
--------

**Example 1: Basic Usage**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])  # Shape: (1, 3, 3)
    >>> rotate = T.RandomRotation(degrees=30)  # Randomly rotate by up to 30 degrees
    >>> output = rotate(input_tensor)
    >>> print(output)
    Tensor(...)  # The exact output will vary due to the randomness of the rotation

**Example 2: Rotating an Image Batch**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                                      [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]])  # Shape: (1, 2, 3, 3)
    >>> rotate = T.RandomRotation(degrees=45)  # Randomly rotate each image by up to 45 degrees
    >>> output = rotate(input_tensor)
    >>> print(output)
    Tensor(...)  # The exact output will vary due to the randomness of the rotation

.. note::

    - This transformation introduces rotational variance, 
      which can improve the robustness of neural networks during training.
    - The degree of rotation is sampled independently for each input image 
      or sample in a batch.
