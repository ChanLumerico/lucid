nn.ReLU6
========

.. autoclass:: lucid.nn.ReLU6

The `ReLU6` class is an activation function module that limits the output of the 
ReLU activation to the range [0, 6]. It is typically used to introduce non-linearity 
in neural networks while capping the maximum activation value.

Class Signature
---------------

.. code-block:: python

    class ReLU6(nn.Module):
        def __init__(self) -> None

Forward Calculation
--------------------
The `ReLU6` module performs the following operation:

.. math::

    \text{ReLU6}(x) = \min(\max(x, 0), 6)

Where:

- :math:`x` is the input tensor.
- The output is clipped between 0 and 6.

Returns
-------
- **Tensor**: 
  The tensor resulting from applying the `ReLU6` activation. 
  It has the same shape as the input tensor.

Examples
--------

**Using `ReLU6` in a neural network model:**

.. code-block:: python

    import lucid.nn as nn
    import lucid.nn.functional as F

    # Define a simple model using ReLU6
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 5)
            self.activation = nn.ReLU6()

        def forward(self, x):
            x = self.fc1(x)
            return self.activation(x)

    # Create an instance of the model
    model = SimpleModel()

    # Input tensor
    input_tensor = lucid.Tensor([[1.0, -2.0, 3.0]])

    # Forward pass
    output = model(input_tensor)
    print(output)

**Standalone Example:**

.. code-block:: python

    import lucid
    import lucid.nn.functional as F
    from lucid.nn import ReLU6

    input_tensor = lucid.Tensor([-3.0, 0.0, 4.0, 7.0])
    activation = ReLU6()

    output = activation(input_tensor)
    print(output)  # Output will be [0.0, 0.0, 4.0, 6.0]

.. note::

  - `ReLU6` is commonly used in lightweight models such as MobileNet to enhance efficiency.
  - It prevents overly large activations, which can help with numerical stability.

