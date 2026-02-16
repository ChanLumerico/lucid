nn.Linear
=========

.. autoclass:: lucid.nn.Linear

The `Linear` module applies a linear transformation to the incoming data: 
it multiplies the input tensor by a weight matrix and, if enabled, adds a bias vector. 
This operation is fundamental in neural networks, particularly in fully connected 
layers.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Linear(in_features: int, out_features: int, bias: bool = True)

Parameters
----------
- **in_features** (*int*): 
    Size of each input sample. Represents the number of input features.
    
- **out_features** (*int*): 
    Size of each output sample. Represents the number of output features.
    
- **bias** (*bool*, optional): 
    If set to `True`, the layer will learn an additive bias. Default is `True`.

Attributes
----------
- **weight** (*Tensor*): 
    The learnable weight matrix of shape `(out_features, in_features)`. 
    Initialized from a uniform distribution.
    
- **bias** (*Tensor* or *None*): 
    The learnable bias vector of shape `(out_features)`. 
    If `bias` is set to `False`, this attribute is `None`.

Forward Calculation
-------------------
The `Linear` module performs the following operation:

.. math::

    \mathbf{y} = \mathbf{x} \cdot \mathbf{W}^\top + \mathbf{b}

Where:

- :math:`\mathbf{x}` is the input tensor of shape `(N, *, in\_features)`.
- :math:`\mathbf{W}` is the weight matrix of shape `(out\_features, in\_features)`.
- :math:`\mathbf{b}` is the bias vector of shape `(out\_features)`, if applicable.
- :math:`\mathbf{y}` is the output tensor of shape `(N, *, out\_features)`.

Backward Gradient Calculation
-----------------------------
For tensors **input**, **weight**, and **bias** involved in the `Linear` operation, 
the gradients with respect to the output (**y**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{x}`:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \mathbf{W}

**Gradient with respect to** :math:`\mathbf{W}`:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{W}} = \mathbf{x}^\top

**Gradient with respect to** :math:`\mathbf{b}` (if bias is used):

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{b}} = \mathbf{1}

This means that during backpropagation, gradients flow through the weights and biases 
according to these derivatives.

Examples
--------

**Using `Linear` for a simple linear transformation without bias:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # Shape: (1, 3)
    >>> linear = nn.Linear(in_features=3, out_features=2, bias=False)
    >>> print(linear.weight)
    Tensor([[4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]], requires_grad=True)  # Shape: (2, 3)
    >>> output = linear(input_tensor)  # Shape: (1, 2)
    >>> print(output)
    Tensor([[32.0, 50.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[4.0, 5.0, 6.0],
     [7.0, 8.0, 9.0]]  # Corresponding to weight
    >>> print(linear.weight.grad)
    [[1.0, 2.0, 3.0],
     [1.0, 2.0, 3.0]]  # Corresponding to input_tensor

**Using `Linear` with bias for a batch of inputs:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape: (2, 2)
    >>> linear = nn.Linear(in_features=2, out_features=2, bias=True)
    >>> print(linear.weight)
    Tensor([[5.0, 6.0],
            [7.0, 8.0]], requires_grad=True)  # Shape: (2, 2)
    >>> print(linear.bias)
    Tensor([9.0, 10.0], requires_grad=True)  # Shape: (2,)
    >>> output = linear(input_tensor)  # Shape: (2, 2)
    >>> print(output)
    Tensor([[26.0, 33.0],
            [48.0, 75.0]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[5.0, 6.0],
     [7.0, 8.0]]  # Corresponding to weight
    >>> print(linear.weight.grad)
    [[1.0, 2.0],
     [3.0, 4.0]]  # Corresponding to input_tensor
    >>> print(linear.bias.grad)
    [1.0, 1.0]  # Corresponding to ones

**Integrating `Linear` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class SimpleModel(nn.Module):
    ...     def __init__(self):
    ...         super(SimpleModel, self).__init__()
    ...         self.fc1 = nn.Linear(in_features=4, out_features=3)
    ...         self.relu = nn.ReLU()
    ...         self.fc2 = nn.Linear(in_features=3, out_features=1)
    ...
    ...     def forward(self, x):
    ...         x = self.fc1(x)
    ...         x = self.relu(x)
    ...         x = self.fc2(x)
    ...         return x
    >>>
    >>> model = SimpleModel()
    >>> input_data = Tensor([[0.5, -1.2, 3.3, 0.7]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...]], grad=None)  # Output tensor after passing through the model

