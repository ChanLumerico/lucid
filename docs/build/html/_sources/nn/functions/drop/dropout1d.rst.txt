nn.functional.dropout1d
=======================

.. autofunction:: lucid.nn.functional.dropout1d

The `dropout1d` function randomly zeroes entire channels of the 
input tensor with a probability `p` during training.

Function Signature
------------------

.. code-block:: python

    def dropout1d(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape (N, C, L), where N is the batch size, 
    C is the number of channels, and L is the length.

- **p** (*float*, optional): 
    Probability of an element to be zeroed. Default: 0.5.

- **training** (*bool*, optional): 
    Apply dropout if True; do nothing if False. Default: True.

Returns
-------

- **Tensor**: 
    A tensor of the same shape as the input, with entire channels randomly 
    zeroed with probability `p` if training is True.

Examples
--------

Applying dropout1d during training:

.. code-block:: python

    >>> input_ = Tensor([[[1.0, 2.0], [3.0, 4.0]]])  # Shape: (1, 2, 2)
    >>> out = F.dropout1d(input_, p=0.5, training=True)
    >>> print(out)
    # Example output: Tensor([[[0.0, 0.0], [3.0, 4.0]]])