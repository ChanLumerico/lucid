nn.SelectiveKernel
==================

.. autoclass:: lucid.nn.SelectiveKernel

The `SelectiveKernel` class implements a dynamic selection mechanism among 
multiple convolutional branches with different kernel sizes. This allows the model 
to adaptively prioritize specific receptive fields based on the input data.

Class Signature
---------------

.. code-block:: python

    class SelectiveKernel(
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        reduction: int = 16,
    )

Parameters
----------
- **in_channels** (*int*):
  Number of input channels for the convolutional layers.

- **out_channels** (*int*):
  Number of output channels for each branch.

- **kernel_sizes** (*list[int]*):
  A list of kernel sizes for the convolutional branches (e.g., [3, 5, 7]).

- **stride** (*int*, optional):
  The stride for the convolutional layers. Default is 1.

- **padding** (*int | None*, optional):
  The padding for the convolutional layers. If `None`, it defaults to `kernel_size // 2` 
  for each branch. Default is `None`.

- **groups** (*int*, optional):
  Number of blocked connections from input channels to output channels. Default is 1.

- **reduction** (*int*, optional):
  Reduction ratio for the attention mechanism. Default is 16.

Forward Calculation
-------------------

The `SelectiveKernel` processes the input tensor through multiple convolutional branches, 
calculates attention weights for each branch using a lightweight attention mechanism, 
and combines the outputs using these weights.

**Steps**:

1. Each branch applies a convolutional operation with its respective kernel size.
2. The outputs of all branches are stacked along a new dimension.
3. Attention weights are computed based on the stacked outputs using an 
   adaptive average pooling layer followed by a two-layer MLP.
4. Attention weights are applied to each branch's output, 
   and the results are summed to form the final output.

Returns
-------
- **Tensor**: The output tensor after dynamically selecting and combining branch outputs. 
  The shape is `(batch_size, out_channels, height, width)`.

Examples
--------

**Using `SelectiveKernel` with multiple kernel sizes:**

.. code-block:: python

    import lucid
    import lucid.nn as nn

    # Create an input tensor
    x = lucid.random.randn(1, 64, 32, 32)  # Shape: (batch_size, channels, height, width)

    # Initialize SelectiveKernel
    sk = nn.SelectiveKernel(64, 128, kernel_sizes=[3, 5, 7], stride=1)

    # Forward pass
    output = sk(x)
    print(output.shape)  # Output shape: (1, 128, 32, 32)

.. note::

  - The class leverages a lightweight attention mechanism for dynamic kernel selection, 
    which can enhance model performance for tasks requiring multi-scale feature aggregation.
    
  - The `reduction` parameter controls the dimensionality reduction in the attention 
    mechanism and can be adjusted based on the task or computational constraints.

Attention Mechanism
-------------------

The attention module computes weights for each branch using the following steps:

.. math::
    \text{Weights} = \text{Softmax}(\text{MLP}(\text{AvgPool}(\text{Branch Outputs})))

where the MLP consists of two fully connected layers with a ReLU activation in between.
