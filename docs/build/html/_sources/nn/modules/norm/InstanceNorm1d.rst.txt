nn.InstanceNorm1d
=================

.. autoclass:: lucid.nn.InstanceNorm1d

The `InstanceNorm1d` module applies Instance Normalization over a one-dimensional input 
(a batch of 1D inputs with optional additional dimensions). 

Instance Normalization normalizes the input features for each instance and channel 
separately by maintaining the mean and variance of each feature across the spatial dimension. 
This normalization technique is particularly useful in tasks like style transfer and 
speech processing, where per-instance normalization can lead to better performance 
by reducing instance-specific contrast variations.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.InstanceNorm1d(
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
      For a 1D input, this typically corresponds to the number of channels.

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
The `InstanceNorm1d` module normalizes the input tensor and optionally applies 
a scale and shift transformation. The normalization is performed as follows:

.. math::

    \mathbf{y} = \gamma \left( \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta

Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(N, C, L)` where:
  - :math:`N` is the batch size.
  - :math:`C` is the number of channels/features.
  - :math:`L` is the length of the signal.

- :math:`\mu` is the mean of the input over the spatial dimension :math:`L` for each instance and channel.
- :math:`\sigma^2` is the variance of the input over the spatial dimension :math:`L` for each instance and channel.
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
    \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( \mathbf{1} - \frac{1}{L} \right) 
    \left( \mathbf{y} \cdot \left(1 - \mathbf{y}\right) \right)

**Gradient with respect to** :math:`\gamma`:

.. math::

    \frac{\partial \mathcal{L}}{\partial \gamma} = 
    \sum_{i=1}^{N} \sum_{j=1}^{L} \mathbf{y}_{i,j} \cdot 
    \left( \frac{\mathbf{x}_{i,j} - \mu_i}{\sqrt{\sigma^2_i + \epsilon}} \right)

**Gradient with respect to** :math:`\beta`:

.. math::

    \frac{\partial \mathcal{L}}{\partial \beta} = 
    \sum_{i=1}^{N} \sum_{j=1}^{L} \mathbf{y}_{i,j}

Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`\mathbf{1}` is a tensor of ones with the same shape as :math:`\mathbf{y}`.
- :math:`\mu_i` and :math:`\sigma^2_i` are the mean and variance for the :math:`i`-th instance and channel.

These gradients ensure that the normalization process adjusts the parameters to 
minimize the loss function effectively.

Examples
--------
**Using `InstanceNorm1d` with a simple input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> input_tensor = Tensor([[
    ...     [1.0, 2.0, 3.0],
    ...     [4.0, 5.0, 6.0]
    ... ]], requires_grad=True)  # Shape: (1, 2, 3)
    >>> instance_norm = nn.InstanceNorm1d(num_features=2)
    >>> output = instance_norm(input_tensor)  # Shape: (1, 2, 3)
    >>> print(output)
    Tensor([[
        [-1.2247, 0.0, 1.2247],
        [-1.2247, 0.0, 1.2247]
    ]], grad=None)

    # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    [[
        [-0.6124,  0.0,  0.6124],
        [-0.6124,  0.0,  0.6124]
    ]]  # Gradients with respect to input_tensor

**Using `InstanceNorm1d` within a simple neural network:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> class InstanceNorm1dModel(nn.Module):
    ...     def __init__(self):
    ...         super(InstanceNorm1dModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=3, out_features=2)
    ...         self.instance_norm = nn.InstanceNorm1d(num_features=2)
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.instance_norm(x)
    ...         return x
    ...
    >>> model = InstanceNorm1dModel()
    >>> input_data = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # Shape: (1, 3)
    >>> output = model(input_data)
    >>> print(output)
    Tensor([[
        [-1.2247,  1.2247]
    ]], grad=None)  # Example output after passing through the model

    # Backpropagation
    >>> output.backward()
    >>> print(input_data.grad)
    [[-0.6124,  0.0,  0.6124]]  # Gradients with respect to input_data
