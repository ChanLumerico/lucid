nn.functional.bilinear
========================

.. autofunction:: lucid.nn.functional.bilinear

The `bilinear` function applies a bilinear transformation to two input tensors. 
It computes an output tensor based on the interactions between the two input tensors 
and a weight tensor, and optionally adds a bias tensor. This operation is commonly 
used in neural networks for tasks that require interactions between two different inputs, 
such as in certain types of attention mechanisms or feature interactions.

Function Signature
------------------

.. code-block:: python

    def bilinear(
        input_1: Tensor, input_2: Tensor, weight: Tensor, bias: Tensor | None = None
    ) -> Tensor

Parameters
----------

- **input\_1** (*Tensor*): 
    The first input tensor of shape `(N, *, in1_features)`, where `*` represents 
    any number of additional dimensions.

- **input\_2** (*Tensor*): 
    The second input tensor of shape `(N, *, in2_features)`, where `*` represents 
    any number of additional dimensions.

- **weight** (*Tensor*): 
    The weight tensor of shape `(out_features, in1_features, in2_features)`. 
    Each slice `weight[i]` corresponds to the weights for the `i`-th output feature.

- **bias** (*Tensor*, optional): 
    The bias tensor of shape `(1, out_features)`. If `None`, no bias is added.

Returns
-------

- **Tensor**: 
    A new `Tensor` containing the result of the bilinear transformation. 
    The shape of the output tensor is `(N, *, out_features)`. 
    If any of the inputs (`input_1`, `input_2`, `weight`, or `bias`) 
    require gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `bilinear` operation is:

.. math::

    \mathbf{out} = \mathbf{input\_1} \cdot \mathbf{W} \cdot \mathbf{input\_2}^\top + \mathbf{bias}

Where:

- For each pair of input vectors :math:`\mathbf{x}_1` from `input_1` and 
  :math:`\mathbf{x}_2` from `input_2`:

  .. math::

      \mathbf{out}_i = \mathbf{x}_1 \cdot \mathbf{W}_i \cdot \mathbf{x}_2^\top + \mathbf{b}_i

  Here, :math:`\mathbf{W}_i` is the weight matrix for the `i`-th output feature, 
  and :math:`\mathbf{b}_i` is the corresponding bias term.

Backward Gradient Calculation
-----------------------------

For tensors **input\_1**, **input\_2**, **weight**, and **bias** involved in the `bilinear` operation, 
the gradients with respect to the output (**out**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_1}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_1}} = \mathbf{W} \cdot \mathbf{input\_2}^\top

**Gradient with respect to** :math:`\mathbf{input\_2}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_2}} = \mathbf{W}^\top \cdot \mathbf{input\_1}^\top

**Gradient with respect to** :math:`\mathbf{weight}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{weight}} = \mathbf{input\_1} \otimes \mathbf{input\_2}

**Gradient with respect to** :math:`\mathbf{bias}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{bias}} = \mathbf{1}

Examples
--------

Using `bilinear` for a simple bilinear transformation without bias:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_1 = Tensor([[1.0, 2.0]], requires_grad=True)  # Shape: (1, 2)
    >>> input_2 = Tensor([[3.0, 4.0]], requires_grad=True)  # Shape: (1, 2)
    >>> weight = Tensor([[[5.0, 6.0], 
                         [7.0, 8.0]]], requires_grad=True)  # Shape: (1, 2, 2)
    >>> out = F.bilinear(input_1, input_2, weight)  # Shape: (1, 1)
    >>> print(out)
    Tensor([[1*5*3 + 1*6*4 + 2*7*3 + 2*8*4]], grad=None)
    Tensor([[15 + 24 + 42 + 64]])
    Tensor([[145.0]], grad=None)

Backpropagation computes gradients for `input_1`, `input_2`, and `weight`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_1.grad)
    [[(5*3 + 6*4), (7*3 + 8*4)]]  # Corresponding to weight and input_2
    [[27.0, 53.0]]
    
    >>> print(input_2.grad)
    [[(5*1 + 7*2), (6*1 + 8*2)]]  # Corresponding to weight and input_1
    [[19.0, 22.0]]
    
    >>> print(weight.grad)
    [[[1*3, 1*4],
      [2*3, 2*4]]]
    # Corresponding to input_1 and input_2
    [[[3.0, 4.0],
      [6.0, 8.0]]]

Using `bilinear` with bias for a batch of inputs:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_1 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape: (2, 2)
    >>> input_2 = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)  # Shape: (2, 2)
    >>> weight = Tensor([[[9.0, 10.0], 
                         [11.0, 12.0]]], requires_grad=True)  # Shape: (1, 2, 2)
                         
    >>> bias = Tensor([[13.0]], requires_grad=True)  # Shape: (1, 1)
    >>> out = F.bilinear(input_1, input_2, weight, bias)  # Shape: (2, 1)
    >>> print(out)
    Tensor([[372.0], [1134.0]], grad=None)

Backpropagation propagates gradients through the inputs, weights, and bias:

.. code-block:: python

    >>> out.backward()
    >>> print(input_1.grad)
    Tensor([[ 45.0,  82.0],
            [119.0, 170.0]])

    >>> print(input_2.grad)
    Tensor([[31.0, 34.0],
            [57.0, 62.0]])

    >>> print(weight.grad)
    Tensor([[[5.0, 6.0],
             [10.0, 12.0]],
            [[21.0, 24.0],
             [28.0, 32.0]]])

    >>> print(bias.grad)
    [[1.0], [1.0]]  # Corresponding to ones
