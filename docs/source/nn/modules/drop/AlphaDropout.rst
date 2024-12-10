nn.AlphaDropout
===============

.. autoclass:: lucid.nn.AlphaDropout

The `AlphaDropout` module applies Alpha Dropout to the input tensor. Alpha Dropout is a 
specialized dropout technique designed to maintain the self-normalizing properties of 
activation functions like SELU (Scaled Exponential Linear Unit). Unlike standard Dropout, 
which randomly zeroes out elements, Alpha Dropout randomly sets elements to a specific 
value that preserves the mean and variance of the input, thereby ensuring stable 
forward and backward passes.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.AlphaDropout(
        p: float = 0.1
    ) -> None

Parameters
----------
- **p** (*float*, optional):
  The probability of an element to be set to the dropout value. Must be between `0` and `1`. 
  Default is `0.1`.

Attributes
----------
- **mask** (*Tensor* or *None*):
  A binary mask tensor of the same shape as the input, where each element is `1` with 
  probability `1 - p` and set to the dropout value with probability `p`. This mask is 
  used to modify the input during the forward pass in training mode.

Forward Calculation
-------------------
The `AlphaDropout` module performs the following operations:

**During Training:**

1. **Mask Generation:**

   .. math::

       \mathbf{mask} \sim \text{Bernoulli}(1 - p)

   Each element of the mask tensor is sampled independently from a Bernoulli distribution 
   with probability `1 - p` of being `1`.

2. **Applying Alpha Dropout:**

   .. math::

       \mathbf{y} = \frac{\mathbf{x} \odot \mathbf{mask} + a \cdot (1 - \mathbf{mask})}{1 - p}

   Where:

   - :math:`\mathbf{x}` is the input tensor.
   - :math:`\mathbf{mask}` is the binary mask tensor.
   - :math:`a` is the dropout value calculated to preserve the mean and variance of the input.
   - :math:`\odot` denotes element-wise multiplication.
   - The division by `1 - p` ensures that the expected value of the activations remains the same.

   **Note:** The dropout value :math:`a` is specifically chosen based on the activation function 
   (e.g., SELU) to maintain the self-normalizing properties of the network.

**During Evaluation:**

.. math::

    \mathbf{y} = \mathbf{x}

   Dropout is not applied during evaluation; the input is passed through unchanged.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient of the loss with respect to the input tensor is computed as follows:

.. math::

    \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\mathbf{mask}}{1 - p} 
    \odot \frac{\partial \mathcal{L}}{\partial \mathbf{y}}

Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`\mathbf{mask}` is the binary mask tensor applied during the forward pass.
- :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{y}}` 
  is the gradient of the loss with respect to the output.

These gradients ensure that only the non-dropped elements contribute to the weight updates, 
maintaining the robustness and self-normalizing properties introduced by Alpha Dropout.

Examples
--------
**Using `AlphaDropout` with a simple input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> input_tensor = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)  # Shape: (1, 4)
    >>> alpha_dropout = nn.AlphaDropout(p=0.1)
    >>> output = alpha_dropout(input_tensor)  # Shape: (1, 4)
    >>> print(output)
    Tensor([[0.9091, 2.0, 3.0, -4.5455]], grad=None)  # Example output with some elements set to dropout value and others scaled

    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[0.9091, 1.0, 1.0, -4.5455]]  # Gradients are scaled and adjusted where dropout was applied

**Using `AlphaDropout` within a simple neural network:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> class AlphaDropoutModel(nn.Module):
    ...     def __init__(self):
    ...         super(AlphaDropoutModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=4, out_features=2)
    ...         self.alpha_dropout = nn.AlphaDropout(p=0.1)
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.alpha_dropout(x)
    ...         return x
    ...
    >>> model = AlphaDropoutModel()
    >>> input_data = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[-0.4545, 1.8182]], grad=None)  # Example output after passing through the model

    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    [[-0.4545, 1.8182, 0.0, 0.0]]  # Gradients with respect to input_data, scaled and adjusted appropriately
