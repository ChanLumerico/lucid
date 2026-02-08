nn.functional.dropout
=====================

.. autofunction:: lucid.nn.functional.dropout

The `dropout` function randomly zeroes some elements of the input 
tensor with a probability `p` during training to prevent overfitting.

Function Signature
------------------

.. code-block:: python

    def dropout(input_: Tensor, p: float = 0.5, training: bool = True) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of any shape.

- **p** (*float*, optional): 
    Probability of an element to be zeroed. Default: 0.5.

- **training** (*bool*, optional): 
    Apply dropout if True; do nothing if False. Default: True.

Returns
-------

- **Tensor**: 
    A tensor of the same shape as the input, with elements randomly 
    zeroed with probability `p` if training is True.

Examples
--------

Applying dropout during training:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([1.0, 2.0, 3.0])
    >>> out = F.dropout(input_, p=0.5, training=True)
    >>> print(out)
    # Example output: Tensor([1.0, 0.0, 3.0])