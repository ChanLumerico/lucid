nn.LayerNorm
============

.. autoclass:: lucid.nn.LayerNorm

The `LayerNorm` module applies Layer Normalization over a specified shape of input 
(tensor) by normalizing the activations within each individual instance. Unlike Batch 
Normalization, which normalizes across the batch dimension, Layer Normalization 
normalizes across the features and spatial dimensions for each sample independently. 
This makes it particularly useful in scenarios where batch sizes are small or 
dynamic, such as in recurrent neural networks and transformers.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.LayerNorm(
        normalized_shape: _ShapeLike | int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None

Parameters
----------
- **normalized_shape** (*tuple[int]* or *int*):
    - Input shape from an expected input. If a single integer is used, it is treated 
      as a singleton tuple for compatibility.

- **eps** (*float*, optional):
    - A value added to the denominator for numerical stability. Default is `1e-5`.

- **elementwise_affine** (*bool*, optional):
    - If `True`, this module has learnable per-element affine parameters 
      initialized to ones (for weights) and zeros (for biases). Default is `True`.

- **bias** (*bool*, optional):
    - If `True` and `elementwise_affine` is `True`, adds a learnable bias 
      to the output. Default is `True`.

Attributes
----------
- **weight** (*Tensor* or *None*):
  The learnable scale parameter :math:`\gamma` of shape `normalized_shape`. 
  Only present if `elementwise_affine` is `True`.

- **bias** (*Tensor* or *None*):
  The learnable shift parameter :math:`\beta` of shape `normalized_shape`. 
  Only present if `bias` and `elementwise_affine` are `True`.

Forward Calculation
-------------------
The `LayerNorm` module normalizes the input tensor and optionally applies 
a scale and shift transformation. The normalization is performed as follows:

.. math::

    \mathbf{y} = \gamma \left( \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \right) + \beta

Where:

- :math:`\mathbf{x}` is the input tensor of shape :math:`(*, D_1, D_2, \dots, D_N)` where:
  - :math:`*` represents any number of leading dimensions.
  - :math:`D_1, D_2, \dots, D_N` are the dimensions specified by `normalized_shape`.

- :math:`\mu` is the mean of the input over the `normalized_shape` dimensions.
- :math:`\sigma^2` is the variance of the input over the `normalized_shape` dimensions.
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
    \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( \mathbf{1} - \frac{1}{D} \right) 
    \left( \mathbf{y} \cdot \left(1 - \mathbf{y}\right) \right)

**Gradient with respect to** :math:`\gamma`:

.. math::

    \frac{\partial \mathcal{L}}{\partial \gamma} = 
    \sum_{i=1}^{D} \mathbf{y}_i \cdot 
    \left( \frac{\mathbf{x}_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \right)

**Gradient with respect to** :math:`\beta`:

.. math::

    \frac{\partial \mathcal{L}}{\partial \beta} = 
    \sum_{i=1}^{D} \mathbf{y}_i

Where:

- :math:`\mathcal{L}` is the loss function.
- :math:`\mathbf{1}` is a tensor of ones with the same shape as :math:`\mathbf{y}`.
- :math:`D` is the product of the dimensions specified by `normalized_shape`.

These gradients ensure that the normalization process adjusts the parameters to 
minimize the loss function effectively.

Examples
--------
**Using `LayerNorm` with a simple input tensor:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> input_tensor = Tensor([[
    ...     [1.0, 2.0, 3.0],
    ...     [4.0, 5.0, 6.0]
    ... ]], requires_grad=True)  # Shape: (1, 2, 3)
    >>> layer_norm = nn.LayerNorm(normalized_shape=3)
    >>> output = layer_norm(input_tensor)  # Shape: (1, 2, 3)
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

**Using `LayerNorm` within a simple neural network:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> class LayerNormModel(nn.Module):
    ...     def __init__(self):
    ...         super(LayerNormModel, self).__init__()
    ...         self.linear = nn.Linear(in_features=3, out_features=2)
    ...         self.layer_norm = nn.LayerNorm(normalized_shape=2)
    ...
    ...     def forward(self, x):
    ...         x = self.linear(x)
    ...         x = self.layer_norm(x)
    ...         return x
    ...
    >>> model = LayerNormModel()
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
