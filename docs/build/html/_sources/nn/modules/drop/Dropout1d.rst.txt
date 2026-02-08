nn.Dropout1d
============
    
.. autoclass:: lucid.nn.Dropout1d
    
The `Dropout1d` module applies Dropout to a one-dimensional input tensor. 

Dropout is a regularization technique used to prevent overfitting in neural networks by randomly 
zeroing out a subset of activations during training. This encourages the network to 
learn more robust features that are not reliant on specific activations.
    
Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.Dropout1d(p: float = 0.5) -> None
    
Parameters
----------
- **p** (*float*, optional):
  The probability of an element to be zeroed. Must be between `0` and `1`. Default is `0.5`.
    
Attributes
----------
- **mask** (*Tensor* or *None*):
  A binary mask tensor of the same shape as the input, where each element is `0` with 
  probability `p` and `1` otherwise. This mask is used to zero out elements during the 
  forward pass in training mode.
    
Forward Calculation
-------------------
The `Dropout1d` module performs the following operations:
    
**During Training:**
    
1. **Mask Generation:**
   
   .. math::
   
       \mathbf{mask} \sim \text{Bernoulli}(1 - p)
   
   Each element of the mask tensor is sampled independently from a Bernoulli distribution 
   with probability `1 - p` of being `1`.
    
2. **Applying Dropout:**
   
   .. math::
   
       \mathbf{y} = \frac{\mathbf{x} \odot \mathbf{mask}}{1 - p}
   
   Where:

   - :math:`\mathbf{x}` is the input tensor.
   - :math:`\mathbf{mask}` is the binary mask tensor.
   - :math:`\odot` denotes element-wise multiplication.
   - The division by `1 - p` ensures that the expected value of the activations remains the same.
    
**During Evaluation:**
    
.. math::
    
    \mathbf{y} = \mathbf{x}
    
Dropout is not applied during evaluation; the input is passed through unchanged.
    
Backward Gradient Calculation
-----------------------------
During backpropagation, the gradient of the loss with respect to the input tensor 
is computed as follows:
    
**During Training:**
    
.. math::
    
    \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\mathbf{mask}}{1 - p} 
    \odot \frac{\partial \mathcal{L}}{\partial \mathbf{y}}
    
**During Evaluation:**
    
.. math::
    
    \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}
    
Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`\mathbf{mask}` is the binary mask tensor applied during the forward pass.
- :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{y}}` 
  is the gradient of the loss with respect to the output.
    
These gradients ensure that only the non-dropped elements contribute to the weight updates, 
maintaining the robustness introduced by Dropout.
    
Examples
--------
**Using `Dropout1d` with a simple input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> input_tensor = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)  # Shape: (1, 4)
    >>> dropout1d = nn.Dropout1d(p=0.5)
    >>> output = dropout1d(input_tensor)  # Shape: (1, 4)
    >>> print(output)
    Tensor([[2.0, 0.0, 6.0, 8.0]], grad=None)  # Example output with some elements zeroed
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[2.0, 0.0, 6.0, 8.0]]  # Gradients are scaled and zeroed where dropout was applied
    
**Using `Dropout1d` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> class Dropout1dModel(nn.Module):
    ...     def __init__(self):
    ...         super(Dropout1dModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=4, out_features=2)
    ...         self.dropout = nn.Dropout1d(p=0.5)
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.dropout(x)
    ...         return x
    ...
    >>> model = Dropout1dModel()
    >>> input_data = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[...], [...]], grad=None)  # Example output after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    # Gradients with respect to input_data, scaled and zeroed appropriately
