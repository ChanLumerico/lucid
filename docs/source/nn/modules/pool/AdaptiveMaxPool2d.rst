nn.AdaptiveMaxPool2d
====================

.. autoclass:: lucid.nn.AdaptiveMaxPool2d

The `AdaptiveMaxPool2d` applies adaptive max pooling on a 2D input, computing 
appropriate pooling parameters to reach the target output shape.

Class Signature
---------------

.. code-block:: python

    class AdaptiveMaxPool2d(nn.Module):
        def __init__(self, output_size: int | tuple[int, int])

Parameters
----------

- **output_size** (*int or tuple of int*):
  The target output spatial size :math:`(H_{out}, W_{out})`.

Attributes
----------

- **output_size** (*tuple of int*):
  Stored target output size after adaptation.

Forward Calculation
-------------------

Takes input of shape :math:`(N, C, H, W)` and produces output of shape 
:math:`(N, C, H_{out}, W_{out})`. Kernel size and stride are computed automatically.

Behavior
--------

Wraps `adaptive_max_pool2d`, computing parameters that evenly partition the 
input height and width into the target output dimensions.

Examples
--------

.. code-block:: python

    import lucid.nn as nn

    input_ = Tensor.ones((1, 1, 6, 6))
    pool = nn.AdaptiveMaxPool2d(output_size=(3, 3))
    output = pool(input_)
    print(output.shape)  # (1, 1, 3, 3)
