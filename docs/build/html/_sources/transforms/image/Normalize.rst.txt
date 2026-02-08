transforms.Normalize
====================

.. autoclass:: lucid.transforms.Normalize

The `Normalize` class is a transformation that normalizes an input tensor 
by subtracting the specified mean and dividing by the specified standard 
deviation for each channel independently. 

This operation is commonly used in image processing pipelines to ensure 
that inputs have a standard range, facilitating better training stability 
and convergence in neural networks.

Class Signature
---------------

.. code-block:: python

    class lucid.transforms.Normalize(mean: tuple[float, ...], std: tuple[float, ...])


Parameters
----------
- **mean** (*tuple[float, ...]*):
  The mean value(s) to subtract from each channel of the input tensor. 
  Each channel will have its own mean value. 
  The tuple length must match the number of channels in the input.

- **std** (*tuple[float, ...]*):
  The standard deviation value(s) used to divide each channel of the input tensor. 
  Each channel will have its own standard deviation value. 
  The tuple length must match the number of channels in the input.

Attributes
----------
- **mean** (*tuple[float, ...]*):
  The mean values used for normalization, stored as an internal attribute.

- **std** (*tuple[float, ...]*):
  The standard deviation values used for normalization, stored as an internal attribute.

Forward Calculation
-------------------

The normalization of an input tensor :math:`\mathbf{x}` is calculated as 
follows for each channel :math:`c`:

.. math::
   \mathbf{x}_{\text{normalized}, c} = \frac{\mathbf{x}_c - \mu_c}{\sigma_c}

Where:

- :math:`\mathbf{x}_c` is the input tensor channel :math:`c`.
- :math:`\mu_c` is the mean for channel :math:`c`.
- :math:`\sigma_c` is the standard deviation for channel :math:`c`.

Examples
--------

**Example 1: Basic Usage**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (2, 3)
    >>> normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    >>> output = normalize(input_tensor)
    >>> print(output)
    Tensor([[ 1.0,  3.0,  5.0],
            [ 7.0,  9.0, 11.0]], grad=None)

**Example 2: Normalizing an Image Batch**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[0.0, 0.5, 1.0], [0.1, 0.2, 0.3]]])  # Shape: (1, 2, 3)
    >>> normalize = T.Normalize(mean=(0.5,), std=(0.5,))  # Normalize using a single mean and std for all channels
    >>> output = normalize(input_tensor)
    >>> print(output)
    Tensor([[[-1.0, 0.0, 1.0],
             [-0.8, -0.6, -0.4]]], grad=None)

.. note::

    - The **mean** and **std** must have the same length as the number of channels 
      in the input tensor.
    - This transform is typically applied to images before inputting them into a 
      neural network to standardize the distribution of pixel intensities.
