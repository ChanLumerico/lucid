transforms.RandomGrayscale
==========================

.. autoclass:: lucid.transforms.RandomGrayscale

The `RandomGrayscale` class is a transformation that randomly converts an 
input tensor to grayscale with a given probability. This transformation is 
often used in data augmentation to increase robustness in training image-based 
neural networks.

Class Signature
---------------

.. code-block:: python

    class lucid.transforms.RandomGrayscale(p: float = 0.1)


Parameters
----------
- **p** (*float*, optional):
  The probability of converting the input tensor to grayscale. 
  Must be a value between 0 and 1. Default is 0.1.

Attributes
----------
- **p** (*float*):
  The probability with which the input tensor will be converted to grayscale, 
  stored as an internal attribute.

Forward Calculation
-------------------

The conversion of an input tensor :math:`\mathbf{x}` to grayscale is applied 
with a probability :math:`p` as follows:

1. Generate a random number :math:`r` from a uniform distribution over [0, 1].
2. If :math:`r < p`, convert the input tensor to grayscale using the following 
   formula for each pixel (or channel) :math:`(r, g, b)`:

.. math::

   y = 0.299r + 0.587g + 0.114b

where :math:`r`, :math:`g`, and :math:`b` are the red, green, and blue channels, respectively. The resulting value :math:`y` replaces the color channels with a single grayscale intensity.

3. If the tensor has multiple channels (e.g., an image batch), 
   the grayscale transformation is applied independently to each image.

Examples
--------

**Example 1: Basic Usage**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                      [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                      [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]])  # Shape: (1, 3, 2, 3)
    >>> grayscale = T.RandomGrayscale(p=1.0)  # Always convert to grayscale
    >>> output = grayscale(input_tensor)
    >>> print(output)
    Tensor(...)  # The exact output will be a grayscale version of the input

**Example 2: Randomized Grayscale**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                      [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                                      [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]]])  # Shape: (1, 3, 2, 3)
    >>> grayscale = T.RandomGrayscale(p=0.5)  # 50% chance to convert to grayscale
    >>> output = grayscale(input_tensor)  # Random outcome

.. note::

    - This transformation is useful for data augmentation, especially in scenarios 
      where robustness to color changes is essential.
    - Setting **p=1.0** ensures that the input tensor is always converted to grayscale, 
      while **p=0.0** ensures that the input tensor is never converted.
    - The grayscale transformation maintains the shape of the tensor, 
      but the number of channels may be reduced depending on the implementation.
