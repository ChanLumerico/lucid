nn.functional.dropout3d
=======================

.. autofunction:: lucid.nn.functional.dropout3d

The `dropout3d` function applies a 3D dropout regularization 
operation to the input tensor. 

It randomly zeroes out entire channels with a specified probability 
during training, helping to prevent overfitting in 3D data such as 
volumetric data or video frames.

Function Signature
------------------

.. code-block:: python

    def dropout3d(
        input_: Tensor,
        p: float = 0.5,
        training: bool = True
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
    The input tensor of shape (N, C, D, H, W), where N is the batch size, 
    C is the number of channels, D is the depth, H is the height, and W is the width.

- **p** (*float*, optional):
    Probability of an element to be zeroed out. Default: 0.5.

- **training** (*bool*, optional):
    If True, apply dropout. If False, return the input unchanged. Default: True.

Returns
-------

- **Tensor**:
    The output tensor after applying dropout, with the same shape as the input.

Examples
--------

Applying 3D dropout in training mode:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])  # Shape: (1, 1, 2, 2, 2)
    >>> out = F.dropout3d(input_, p=0.5, training=True)
    >>> print(out)
    Tensor([...])  # Randomly zeroed-out channels

