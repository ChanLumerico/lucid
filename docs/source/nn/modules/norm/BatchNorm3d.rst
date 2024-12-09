nn.BatchNorm3d
==============
    
.. autoclass:: lucid.nn.BatchNorm3d
    
The `BatchNorm3d` module applies Batch Normalization over a three-dimensional input 
(a batch of 3D inputs with optional additional dimensions). 
    
Batch Normalization normalizes the input features by maintaining the mean and variance 
of each feature across the batch and spatial dimensions. This stabilization of activations 
helps in accelerating the training process, reducing sensitivity to network initialization, 
and mitigating issues like vanishing or exploding gradients.
    
Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.BatchNorm3d(
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None
    
Parameters
----------
- **num_features** (*int*):
    - The number of features or channels in the input tensor. 
      For a 3D input, this typically corresponds to the number of channels.
    
- **eps** (*float*, optional):
    - A small value added to the denominator for numerical stability. Default is `1e-5`.
    
- **momentum** (*float* or *None*, optional):
    - The value used for the running mean and running variance computation. 
      When set to `None`, it defaults to `1 - momentum` in some frameworks. Default is `0.1`.
    
- **affine** (*bool*, optional):
    - If `True`, the module has learnable affine parameters (scale and shift). 
      Default is `True`.
    
- **track_running_stats** (*bool*, optional):
    - If `True`, the module tracks the running mean and variance, 
      which are not trainable but are updated during training and used during evaluation. 
      Default is `True`.
    
Attributes
----------
- **weight** (*Tensor* or *None*):
  The learnable scale parameter :math:`\gamma` of shape `(num_features)`. 
  Only present if `affine` is `True`.
    
- **bias** (*Tensor* or *None*):
  The learnable shift parameter :math:`\beta` of shape `(num_features)`. 
  Only present if `affine` is `True`.
    
- **running_mean** (*Tensor*):
  The running mean of shape `(num_features)`. Updated during training 
  if `track_running_stats` is `True`.
    
- **running_var** (*Tensor*):
  The running variance of shape `(num_features)`. Updated during training 
  if `track_running_stats` is `True`.
    
Forward Calculation
-------------------
The `BatchNorm3d` module normalizes the input tensor and optionally applies 
a scale and shift transformation. The normalization is performed as follows:
    
.. math::
    
    \mathbf{y} = \gamma \left( \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta
    
Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C, D, H, W)` where:
  - :math:`N` is the batch size.
  - :math:`C` is the number of channels/features.
  - :math:`D`, :math:`H`, and :math:`W` are the depth, height, and width of the input.

- :math:`\mu` is the mean of the input over the batch and spatial dimensions.
- :math:`\sigma^2` is the variance of the input over the batch and spatial dimensions.
- :math:`\epsilon` is a small constant for numerical stability.
- :math:`\gamma` and :math:`\beta` are the learnable scale and shift parameters, respectively.
    
Backward Gradient Calculation
-----------------------------
During backpropagation, gradients are computed with respect to the input, 
scale (:math:`\gamma`), and shift (:math:`\beta`) parameters. 
    
The gradient calculations are as follows:
    
**Gradient with respect to** :math:`\mathbf{x}`:
    
.. math::
    
    \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 
    \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( \mathbf{1} - \frac{1}{N \cdot D \cdot H \cdot W} \right) 
    \left( \mathbf{y} \cdot \left(1 - \mathbf{y}\right) \right)
    
**Gradient with respect to** :math:`\gamma`:
    
.. math::
    
    \frac{\partial \mathcal{L}}{\partial \gamma} = 
    \sum_{i=1}^{N} \sum_{j=1}^{D} \sum_{k=1}^{H} \sum_{l=1}^{W} \mathbf{y}_{i,j,k,l} \cdot 
    \left( \frac{\mathbf{x}_{i,j,k,l} - \mu}{\sqrt{\sigma^2 + \epsilon}} \right)
    
**Gradient with respect to** :math:`\beta`:
    
.. math::
    
    \frac{\partial \mathcal{L}}{\partial \beta} = 
    \sum_{i=1}^{N} \sum_{j=1}^{D} \sum_{k=1}^{H} \sum_{l=1}^{W} \mathbf{y}_{i,j,k,l}
    
Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`\mathbf{1}` is a tensor of ones with the same shape as :math:`\mathbf{y}`.
    
These gradients ensure that the normalization process adjusts the parameters to 
minimize the loss function effectively.
    
Examples
--------
**Using `BatchNorm3d` with a simple input tensor:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> input_tensor = Tensor([[
    ...     [[1.0, 2.0],
    ...      [3.0, 4.0]],
    ...     [[5.0, 6.0],
    ...      [7.0, 8.0]]
    ... ]], requires_grad=True)  # Shape: (1, 2, 2, 2, 2)
    >>> batch_norm = nn.BatchNorm3d(num_features=2)
    >>> output = batch_norm(input_tensor)  # Shape: (1, 2, 2, 2, 2)
    >>> print(output)
    Tensor([[
        [[-1.2247, 0.0],
         [1.2247, 2.4495]],
        [[-1.2247, 0.0],
         [1.2247, 2.4495]]
    ]], grad=None)
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[
        [[-0.6124,  0.0],
         [ 0.6124,  1.2247]],
        [[-0.6124,  0.0],
         [ 0.6124,  1.2247]]
    ]]  # Gradients with respect to input_tensor
    
**Using `BatchNorm3d` within a simple neural network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> class BatchNorm3dModel(nn.Module):
    ...     def __init__(self):
    ...         super(BatchNorm3dModel, self).__init__()
    ...         self.conv = nn.Conv3D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    ...         self.batch_norm = nn.BatchNorm3d(num_features=16)
    ...
    ...     def forward(self, x):
    ...         x = self.conv(x)
    ...         x = self.batch_norm(x)
    ...         return x
    ...
    >>> model = BatchNorm3dModel()
    >>> input_data = Tensor([[
    ...     [[1.0, 2.0, 3.0],
    ...      [4.0, 5.0, 6.0],
    ...      [7.0, 8.0, 9.0]],
    ...     [[10.0, 11.0, 12.0],
    ...      [13.0, 14.0, 15.0],
    ...      [16.0, 17.0, 18.0]],
    ...     [[19.0, 20.0, 21.0],
    ...      [22.0, 23.0, 24.0],
    ...      [25.0, 26.0, 27.0]]
    ... ]], requires_grad=True)  # Shape: (1, 3, 3, 3, 3)
    >>> output = model(input_data)
    >>> print(output)
    # Example output after passing through the model
    
    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    # Gradients with respect to input_data
