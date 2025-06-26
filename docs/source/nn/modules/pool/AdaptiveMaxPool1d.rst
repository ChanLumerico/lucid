nn.AdaptiveMaxPool1d
====================

.. autoclass:: lucid.nn.AdaptiveMaxPool1d

The `AdaptiveMaxPool1d` performs adaptive max pooling on a 1D input tensor, 
automatically computing kernel size, stride, and padding to produce a specified output size.

Class Signature
---------------

.. code-block:: python

    class AdaptiveMaxPool1d(nn.Module):
        def __init__(self, output_size: int)

Parameters
----------

- **output_size** (*int*):
  The desired output length :math:`L_{out}`.

Attributes
----------

- **output_size** (*int*):
  Stores the target output length for the pooling operation.

Forward Calculation
-------------------

Applies adaptive max pooling on input of shape :math:`(N, C, L)`, producing 
an output of shape :math:`(N, C, L_{out})`. Pooling parameters are dynamically calculated 
based on input length.

Behavior
--------

This module wraps the functional `adaptive_max_pool1d` API to provide a module interface, 
automatically adjusting pooling parameters to meet the target output size.

Examples
--------

.. code-block:: python

    import lucid.nn as nn

    input_ = Tensor([[[1, 4, 2, 3, 6, 5, 7, 0, 8, 1]]])
    pool = nn.AdaptiveMaxPool1d(output_size=5)
    output = pool(input_)
    print(output)  # Shape: (1, 1, 5)
