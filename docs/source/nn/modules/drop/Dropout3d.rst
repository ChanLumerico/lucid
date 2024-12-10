nn.Dropout3d
============
    
.. autoclass:: lucid.nn.Dropout3d
    
The `Dropout3d` module applies Dropout to a three-dimensional input tensor. 

Dropout is a regularization technique used to prevent overfitting in neural networks by randomly 
zeroing out entire channels during training. This encourages the network to learn more 
robust features that are not reliant on specific channels.
    
Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.Dropout3d(p: float = 0.5) -> None
    
Parameters
----------
- **p** (*float*, optional):
  The probability of an entire channel to be zeroed. Must be between `0` and `1`. Default is `0.5`.
    
Attributes
----------
- **mask** (*Tensor* or *None*):
  A binary mask tensor of shape `(N, C, 1, 1, 1)` where each channel is set to `0` with 
  probability `p` and `1` otherwise. This mask is broadcasted to the input shape during the 
  forward pass in training mode.
    
Forward Calculation
-------------------
The `Dropout3d` module performs the following operations:
    
**During Training:**
    
1. **Mask Generation:**
   
   .. math::
   
       \mathbf{mask} \sim \text{Bernoulli}(1 - p)^{(N, C, 1, 1, 1)}
   
   Each channel in the mask tensor is sampled independently from a Bernoulli distribution 
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
During backpropagation, the gradient of the loss with respect to the input tensor is computed as follows:
    
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
    
These gradients ensure that only the non-dropped channels contribute to the weight updates, 
maintaining the robustness introduced by Dropout.
    
Examples
--------
**Using `Dropout3d` with a simple input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> input_tensor = Tensor([[
    ...     [[[1.0, 2.0],
    ...       [3.0, 4.0]],
    ...      [[5.0, 6.0],
    ...       [7.0, 8.0]]]
    ... ]], requires_grad=True)  # Shape: (1, 2, 2, 2, 2)
    >>> dropout3d = nn.Dropout3d(p=0.5)
    >>> output = dropout3d(input_tensor)  # Shape: (1, 2, 2, 2, 2)
    >>> print(output)
    Tensor([[
        [[[0.0, 0.0],
          [0.0, 0.0]],
         [[10.0, 12.0],
          [14.0, 16.0]]]
    ]], grad=None)  # Example output with entire channels zeroed
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    # Gradients are scaled and zeroed where dropout was applied
