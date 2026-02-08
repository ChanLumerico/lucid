nn.functional.softmax
=====================

.. autofunction:: lucid.nn.functional.softmax

The `softmax` function applies the softmax activation to the input tensor along a 
specified axis. Softmax is commonly used in neural networks, particularly in the 
output layer for multi-class classification tasks, to convert raw logits into 
probabilities that sum to one. This normalization allows the model to interpret 
the output as a probability distribution over different classes.

Function Signature
------------------
.. code-block:: python

    def softmax(input_: Tensor, axis: int = -1) -> Tensor:
        return _activation.softmax(input_, axis)

Parameters
----------
- **input_** (*Tensor*):
    The input tensor containing raw scores (logits) that need to be normalized 
    into probabilities.

- **axis** (*int*, optional):
    The axis along which softmax will be computed. Default is `-1`, which typically 
    corresponds to the last dimension of the tensor.

Returns
-------
- **Tensor**:
    A tensor of the same shape as `input_` with softmax applied along the specified axis. 
    The values along the specified axis sum to one.

Forward Calculation
-------------------
The `softmax` function computes the softmax of each element along the specified axis using
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
During backpropagation, the gradient of the loss with respect to the input tensor is 
computed as follows:

.. math::

    \frac{\partial \text{softmax}(\mathbf{x})_i}{\partial \mathbf{x}_j} =
    \text{softmax}(\mathbf{x})_i \left( \delta_{ij} - \text{softmax}(\mathbf{x})_j \right)

Where:

- :math:`\delta_{ij}` is the Kronecker delta, which is 1 if :math:`i = j` and 0 otherwise.
- This derivative ensures that gradients are properly scaled and normalized, facilitating 
  effective learning during training.

Examples
--------
**Applying `softmax` to a single input tensor:**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_tensor = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)  # Shape: (1, 3)
    >>> output = F.softmax(input_tensor, axis=1)
    >>> print(output)
    Tensor([[0.6590, 0.2424, 0.0986]], grad=None)
    
    # Backpropagation
    >>> output.backward(Tensor([[1.0, 1.0, 1.0]]))
    >>> print(input_tensor.grad)
    Tensor([[0.6590, -0.2424, -0.0986]])  # Gradients with respect to input_tensor

**Applying `softmax` along a different axis:**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_tensor = Tensor([
    ...     [2.0, 1.0, 0.1],
    ...     [1.0, 3.0, 0.2]
    ... ], requires_grad=True)  # Shape: (2, 3)
    >>> output = F.softmax(input_tensor, axis=0)
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
