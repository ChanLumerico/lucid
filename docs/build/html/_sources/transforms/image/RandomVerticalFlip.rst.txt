transforms.RandomVerticalFlip
=============================

.. autoclass:: lucid.transforms.RandomVerticalFlip

The `RandomVerticalFlip` class is a transformation that randomly flips an 
input tensor vertically with a given probability. This transformation is 
commonly used in data augmentation to increase the diversity of training samples 
for image-based neural networks.

Class Signature
---------------

.. code-block:: python

    class lucid.transforms.RandomVerticalFlip(p: float = 0.5)


Parameters
----------
- **p** (*float*, optional):
  The probability of flipping the input tensor vertically. 
  Must be a value between 0 and 1. Default is 0.5.

Attributes
----------
- **p** (*float*):
  The probability with which the input tensor will be flipped vertically, 
  stored as an internal attribute.

Forward Calculation
-------------------

The vertical flip of an input tensor :math:`\mathbf{x}` is applied with a 
probability :math:`p` as follows:

1. Generate a random number :math:`r` from a uniform distribution over [0, 1].
2. If :math:`r < p`, flip the tensor along its vertical axis.

This can be mathematically represented as:

.. math::
   \mathbf{x}_{\text{flipped}} = \mathbf{x}[..., ::-1, :]

where the second-to-last axis of the tensor is reversed to achieve the vertical flip.

Examples
--------

**Example 1: Basic Usage**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[1.0, 2.0], [3.0, 4.0]]])  # Shape: (1, 2, 2)
    >>> flip = T.RandomVerticalFlip(p=1.0)  # Always flip
    >>> output = flip(input_tensor)
    >>> print(output)
    Tensor([[[3.0, 4.0],
             [1.0, 2.0]]], grad=None)

**Example 2: Randomized Flip**

.. code-block:: python

    >>> import lucid.transforms as T
    >>> input_tensor = lucid.Tensor([[[1.0, 2.0], [3.0, 4.0]]])  # Shape: (1, 2, 2)
    >>> flip = T.RandomVerticalFlip(p=0.5)  # 50% chance to flip
    >>> output = flip(input_tensor)  # Random outcome
