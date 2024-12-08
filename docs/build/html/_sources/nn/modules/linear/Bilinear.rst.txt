nn.Bilinear
============

.. autoclass:: lucid.nn.Bilinear

The `Bilinear` module applies a bilinear transformation to two input tensors. 
It computes the output by performing a bilinear product of the inputs with a 
learnable weight tensor and, if enabled, adds a bias vector. This operation is 
useful in models that need to capture interactions between two different input 
features, such as in certain types of neural network architectures for natural 
language processing or computer vision.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Bilinear(
        in1_features: int, in2_features: int, out_features: int, bias: bool = True
    )

Parameters
----------
- **in1_features** (*int*): 
    Size of each input sample from the first input tensor. Represents the number 
    of features in the first input.

- **in2_features** (*int*): 
    Size of each input sample from the second input tensor. Represents the number 
    of features in the second input.

- **out_features** (*int*): 
    Size of each output sample. Represents the number of output features.

- **bias** (*bool*, optional): 
    If set to `True`, the layer will learn an additive bias. Default is `True`.

Attributes
----------
- **weight** (*Tensor*): 
    The learnable weight tensor of shape `(out_features, in1_features, in2_features)`. 
    Initialized from a uniform distribution.

- **bias** (*Tensor* or *None*): 
    The learnable bias vector of shape `(out_features)`. 
    If `bias` is set to `False`, this attribute is `None`.

Forward Calculation
-------------------
The `Bilinear` module performs the following operation:

.. math::

    \mathbf{y} = \mathbf{x}_1 \cdot \mathbf{W} \cdot \mathbf{x}_2^\top + \mathbf{b}

Where:

- :math:`\mathbf{x}_1` is the first input tensor of shape `(N, *, in1\_features)`.
- :math:`\mathbf{x}_2` is the second input tensor of shape `(N, *, in2\_features)`.
- :math:`\mathbf{W}` is the weight tensor of shape `(out\_features, in1\_features, in2\_features)`.
- :math:`\mathbf{b}` is the bias vector of shape `(out\_features)`, if applicable.
- :math:`\mathbf{y}` is the output tensor of shape `(N, *, out\_features)`.

Backward Gradient Calculation
-----------------------------
For tensors **x₁**, **x₂**, **weight**, and **bias** involved in the `Bilinear` operation, 
the gradients with respect to the output (**y**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{x}_1`:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}_1} = \mathbf{W} \cdot \mathbf{x}_2^\top

**Gradient with respect to** :math:`\mathbf{x}_2`:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{x}_2} = \mathbf{W}^\top \cdot \mathbf{x}_1

**Gradient with respect to** :math:`\mathbf{W}`:

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{W}} = \mathbf{x}_1^\top \otimes \mathbf{x}_2

**Gradient with respect to** :math:`\mathbf{b}` (if bias is used):

.. math::

    \frac{\partial \mathbf{y}}{\partial \mathbf{b}} = \mathbf{1}

This means that during backpropagation, gradients flow through the inputs and weights 
according to these derivatives.

Examples
--------
**Using `Bilinear` for a simple bilinear transformation without bias:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input1 = Tensor([[1.0, 2.0]], requires_grad=True)  # Shape: (1, 2)
    >>> input2 = Tensor([[3.0, 4.0]], requires_grad=True)  # Shape: (1, 2)
    >>> bilinear = nn.Bilinear(in1_features=2, in2_features=2, out_features=1, bias=False)
    >>> print(bilinear.weight)
    Tensor([[[5.0, 6.0],
             [7.0, 8.0]]], requires_grad=True)  # Shape: (1, 2, 2)
    >>> output = bilinear(input1, input2)  # Shape: (1, 1)
    >>> print(output)
    Tensor([[(1*5 + 2*7) * 3 + (1*6 + 2*8) * 4]], grad=None)  # Example calculation

    # Backpropagation
    >>> output.backward()
    >>> print(input1.grad)
    Tensor([[ (5*3 + 6*4), (7*3 + 8*4) ]])  # Corresponding to weight
    >>> print(input2.grad)
    Tensor([[ (5*1 + 7*2), (6*1 + 8*2) ]])  # Corresponding to weight
    >>> print(bilinear.weight.grad)
    Tensor([[[3.0, 4.0],
             [3.0, 4.0]]])  # Corresponding to input1 and input2

**Using `Bilinear` with bias for a batch of inputs:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape: (2, 2)
    >>> input2 = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)  # Shape: (2, 2)
    >>> bilinear = nn.Bilinear(in1_features=2, in2_features=2, out_features=2, bias=True)
    >>> print(bilinear.weight)
    Tensor([[[9.0, 10.0],
             [11.0, 12.0]],
            [[13.0, 14.0],
             [15.0, 16.0]]], requires_grad=True)  # Shape: (2, 2, 2)
    >>> print(bilinear.bias)
    Tensor([17.0, 18.0], requires_grad=True)  # Shape: (2,)
    >>> output = bilinear(input1, input2)  # Shape: (2, 2)
    >>> print(output)
    Tensor([[...], 
            [...]], grad=None)  # Computed output after bilinear transformation and bias addition

    # Backpropagation
    >>> output.backward()
    >>> print(input1.grad)
    Tensor([[...], 
            [...]])  # Gradients with respect to input1
    >>> print(input2.grad)
    Tensor([[...], 
            [...]])  # Gradients with respect to input2
    >>> print(bilinear.weight.grad)
    Tensor([[[...], 
             [...]],
            [[...], 
             [...]]])  # Gradients with respect to weight
    >>> print(bilinear.bias.grad)
    Tensor([1.0, 1.0])  # Gradients with respect to bias

**Integrating `Bilinear` into a Neural Network Model:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> class BilinearModel(nn.Module):
    ...     def __init__(self):
    ...         super(BilinearModel, self).__init__()
    ...         self.bilinear = nn.Bilinear(in1_features=4, in2_features=4, out_features=2)
    ...         self.relu = nn.ReLU()
    ...         self.output = nn.Linear(in_features=2, out_features=1)
    ...
    ...     def forward(self, x1, x2):
    ...         x = self.bilinear(x1, x2)
    ...         x = self.relu(x)
    ...         x = self.output(x)
    ...         return x
    >>>
    >>> model = BilinearModel()
    >>> input1 = Tensor([[0.5, -1.2, 3.3, 0.7]], requires_grad=True)  # Shape: (1, 4)
    >>> input2 = Tensor([[1.5, 2.2, -0.3, 4.1]], requires_grad=True)  # Shape: (1, 4)
    >>> output = model(input1, input2)
    >>> print(output)
    Tensor([[...]], grad=None)  # Output tensor after passing through the model

