nn.Softmax
==========

.. autoclass:: lucid.nn.Softmax

The `Softmax` module applies the softmax activation function to the input tensor along a specified axis. 

Softmax is commonly used in neural networks, particularly in the output layer for multi-class classification tasks, 
to convert raw logits into probabilities that sum to one. This normalization allows the model to interpret the 
output as a probability distribution over different classes.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Softmax(axis: int = -1)

Parameters
----------
- **axis** (*int*, optional):
    The axis along which softmax will be computed. Default is `-1`, 
    which typically corresponds to the last dimension of the tensor.

Attributes
----------
- **None**

Forward Calculation
-------------------
The `Softmax` module computes the softmax of each element along the specified axis using 
the following formula:

.. math::

    \text{softmax}(\mathbf{x})_i = \frac{e^{\mathbf{x}_i}}{\sum_{j} e^{\mathbf{x}_j}}

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{x}_i` is the ith element along the specified axis.
- The exponential function is applied element-wise, and the results are normalized by 
  dividing by the sum of exponentials along the specified axis.

Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient of the loss with respect to the input tensor 
is computed as follows:

.. math::

    \frac{\partial \text{softmax}(\mathbf{x})_i}{\partial \mathbf{x}_j} =
    \text{softmax}(\mathbf{x})_i \left( \delta_{ij} - \text{softmax}(\mathbf{x})_j \right)

Where:

- :math:`\delta_{ij}` is the Kronecker delta, which is 1 if :math:`i = j` and 0 otherwise.
- This derivative ensures that gradients are properly scaled and normalized, facilitating 
  effective learning during training.

Examples
--------
**Applying `Softmax` to a single input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)  # Shape: (1, 3)
    >>> softmax = nn.Softmax(axis=1)
    >>> output = softmax(input_tensor)
    >>> print(output)
    Tensor([[0.6590, 0.2424, 0.0986]], grad=None)
    
    # Backpropagation
    >>> output.backward(Tensor([[1.0, 1.0, 1.0]]))
    >>> print(input_tensor.grad)
    Tensor([[0.6590, -0.2424, -0.0986]])  # Gradients with respect to input_tensor

**Applying `Softmax` along a different axis:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([
    ...     [2.0, 1.0, 0.1],
    ...     [1.0, 3.0, 0.2]
    ... ], requires_grad=True)  # Shape: (2, 3)
    >>> softmax = nn.Softmax(axis=0)
    >>> output = softmax(input_tensor)
    >>> print(output)
    Tensor([
        [0.7311, 0.1192, 0.7311],
        [0.2689, 0.8808, 0.2689]
    ], grad=None)
    
    # Backpropagation
    >>> output.backward(Tensor([
    ...     [1.0, 1.0, 1.0],
    ...     [1.0, 1.0, 1.0]
    ... ]))
    >>> print(input_tensor.grad)
    Tensor([
        [0.1966, -0.1192, 0.1966],
        [-0.1966, 0.1192, -0.1966]
    ])  # Gradients with respect to input_tensor

**Integrating `Softmax` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> import lucid.nn.functional as F
    >>> class SoftmaxModel(nn.Module):
    ...     def __init__(self, input_size, num_classes):
    ...         super(SoftmaxModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=input_size, out_features=num_classes)
    ...
    ...     def forward(self, x):
    ...         logits = self.linear(x)
    ...         probabilities = F.softmax(logits, axis=1)
    ...         return probabilities
    ...
    >>> model = SoftmaxModel(input_size=4, num_classes=3)
    >>> input_data = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[0.0900, 0.2447, 0.6652]], grad=None)  # Output probabilities after softmax
    
    # Backpropagation
    >>> output.backward(Tensor([[1.0, 1.0, 1.0]]))
    >>> print(input_data.grad)
    Tensor([[...], [...], [...], [...]])  # Gradients with respect to input_data
