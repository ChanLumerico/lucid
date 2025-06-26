nn.AdaptiveMaxPool3d
====================

.. autoclass:: lucid.nn.AdaptiveMaxPool3d

The `AdaptiveMaxPool3d` performs adaptive max pooling over 3D volumetric input, 
automatically adjusting kernel size and stride to achieve the specified output dimensions.

Class Signature
---------------

.. code-block:: python

    class AdaptiveMaxPool3d(nn.Module):
        def __init__(self, output_size: int | tuple[int, int, int])

Parameters
----------

- **output_size** (*int or tuple of int*):
  Target output volume size :math:`(D_{out}, H_{out}, W_{out})`.

Attributes
----------

- **output_size** (*tuple of int*):
  Stores the output spatial size as a 3-element tuple.

Forward Calculation
-------------------

Transforms input of shape :math:`(N, C, D, H, W)` into output of shape 
:math:`(N, C, D_{out}, H_{out}, W_{out})`, adapting pooling parameters per dimension.

Behavior
--------

Encapsulates `adaptive_max_pool3d` and dynamically computes pooling regions 
to match the given output volume dimensions.

Examples
--------

.. code-block:: python

    import lucid.nn as nn

    input_ = Tensor.ones((1, 1, 8, 8, 8))
    pool = nn.AdaptiveMaxPool3d(output_size=(2, 2, 2))
    output = pool(input_)
    print(output.shape)  # (1, 1, 2, 2, 2)
