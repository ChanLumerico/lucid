nn.LeakyReLU
============

.. autoclass:: lucid.nn.LeakyReLU

The `LeakyReLU` (Leaky Rectified Linear Unit) module applies the leaky rectified linear activation 
function to the input tensor. 

Unlike the standard `ReLU`, which sets all negative input values to zero, `LeakyReLU` allows a small, 
non-zero gradient for negative input values. This helps mitigate the "dying ReLU" problem by ensuring 
that neurons can continue to learn during training.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.LeakyReLU(negative_slope: float = 0.01)

Parameters
----------
- **negative_slope** (*float*, optional):
    Slope of the function for input values less than zero. Default is `0.01`.

Attributes
----------
- **None**

Forward Calculation
-------------------
The `LeakyReLU` module performs the following operation:

.. math::

    \mathbf{y} = \begin{cases}
        \mathbf{x} & \text{if } \mathbf{x} > 0 \\
        \text{negative\_slope} \times \mathbf{x} & \text{otherwise}
    \end{cases}

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor after applying the leaky ReLU activation.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
    \begin{cases}
        1 & \text{if } \mathbf{x} > 0 \\
        \text{negative\_slope} & \text{otherwise}
    \end{cases}

This means that the gradient of the loss with respect to the input is passed through 
unchanged for positive input values and scaled by `negative_slope` for negative input values.

Examples
--------
**Applying `LeakyReLU` to a single input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 2.0, -0.5, 3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> leaky_relu = nn.LeakyReLU()
    >>> output = leaky_relu(input_tensor)
    >>> print(output)
    Tensor([[-0.01, 2.0, -0.005, 3.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.01, 1.0, 0.01, 1.0]])  # Gradients with respect to input_tensor

**Using `LeakyReLU` with a custom negative slope:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-2.0, 0.0, 1.5, -3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> leaky_relu = nn.LeakyReLU(negative_slope=0.1)
    >>> output = leaky_relu(input_tensor)
    >>> print(output)
    Tensor([[-0.2, 0.0, 1.5, -0.3]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.1, 0.0, 1.0, 0.1]])  # Gradients with respect to input_tensor

**Integrating `LeakyReLU` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class SimpleLeakyReLUModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleLeakyReLUModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=3, out_features=2)
    ...         self.leaky_relu = nn.LeakyReLU(negative_slope=0.05)
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.leaky_relu(x)
    ...         return x
    ...
    >>> model = SimpleLeakyReLUModel()
    >>> input_data = Tensor([[0.5, -1.2, 3.3]], requires_grad=True)  # Shape: (1, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...], [...]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[...], [...]])  # Gradients with respect to input_data
