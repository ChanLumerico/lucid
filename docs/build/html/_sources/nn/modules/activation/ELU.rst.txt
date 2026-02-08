nn.ELU
======

.. autoclass:: lucid.nn.ELU

The `ELU` (Exponential Linear Unit) module applies the exponential linear activation 
function to the input tensor. 

Unlike the standard `ReLU`, which outputs zero for negative input values, 
`ELU` allows for a smooth, non-zero output when the input is negative. 
This helps mitigate the vanishing gradient problem and allows the model to 
learn more effectively by maintaining a mean activation closer to zero.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ELU(alpha: float = 1.0)

Parameters
----------
- **alpha** (*float*, optional):
    Slope of the function for input values less than zero. Controls the value to which an
    ELU saturates for negative inputs. Default is `1.0`.

Attributes
----------
- **alpha** (*float*):
    The negative slope parameter used in the ELU activation function.

Forward Calculation
-------------------
The `ELU` module performs the following operation:

.. math::

    \mathbf{y} = 
    \begin{cases}
        \mathbf{x} & \text{if } \mathbf{x} > 0 \\
        \alpha (\exp(\mathbf{x}) - 1) & \text{otherwise}
    \end{cases}

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{y}` is the output tensor after applying the ELU activation.
- :math:`\alpha` is the negative slope parameter.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient with respect to the input is computed as:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
    \begin{cases}
        1 & \text{if } \mathbf{x} > 0 \\
        \mathbf{y} + \alpha & \text{otherwise}
    \end{cases}

This means that the gradient of the loss with respect to the input is passed through 
unchanged for positive input values and scaled by :math:`\alpha` for negative input values.

Examples
--------
**Applying `ELU` to a single input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-1.0, 2.0, -0.5, 3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> elu = nn.ELU(alpha=1.0)
    >>> output = elu(input_tensor)
    >>> print(output)
    Tensor([[-0.6321, 2.0, -0.3935, 3.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.3679, 1.0, 0.6065, 1.0]])  # Gradients with respect to input_tensor

**Using `ELU` with a custom alpha:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[-2.0, 0.0, 1.5, -3.0]], requires_grad=True)  # Shape: (1, 4)
    >>> elu = nn.ELU(alpha=0.5)
    >>> output = elu(input_tensor)
    >>> print(output)
    Tensor([[-0.4323, 0.0, 1.5, -0.4979]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    Tensor([[0.4323, 0.0, 1.0, 0.2512]])  # Gradients with respect to input_tensor

**Integrating `ELU` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class ELUModel(nn.Module):
    ...     def __init__(self):
    ...         super(ELUModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=3, out_features=2)
    ...         self.elu = nn.ELU(alpha=1.0)
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.elu(x)
    ...         return x
    ...
    >>> model = ELUModel()
    >>> input_data = Tensor([[0.5, -1.2, 3.3]], requires_grad=True)  # Shape: (1, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...], [...]], grad=None)  # Output tensor after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    Tensor([[...], [...]])  # Gradients with respect to input_data
