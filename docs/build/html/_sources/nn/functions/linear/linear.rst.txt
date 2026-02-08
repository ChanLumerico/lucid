nn.functional.linear
=====================

.. autofunction:: lucid.nn.functional.linear

The `linear` function applies a linear transformation to the incoming data: 
it multiplies the input tensor by a weight tensor and, if provided, adds a bias tensor. 
This operation is fundamental in neural networks, particularly in fully connected 
layers.

Function Signature
------------------

.. code-block:: python

    def linear(input_: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of shape `(N, *, in_features)`, where `*` represents any number 
    of additional dimensions.
    
- **weight** (*Tensor*): 
    The weight tensor of shape `(out_features, in_features)`. Each row of the 
    weight tensor represents the weights for one output feature.
    
- **bias** (*Tensor*, optional): 
    The bias tensor of shape `(1, out_features)`. If `None`, no bias is added.

Returns
-------

- **Tensor**: 
    A new `Tensor` containing the result of the linear transformation. 
    The shape of the output tensor is `(N, *, out_features)`. 
    If either `input_`, `weight`, or `bias` requires gradients, the resulting 
    tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `linear` operation is:

.. math::

    \mathbf{out} = \mathbf{input\_} \cdot \mathbf{weight}^\top + \mathbf{bias}

Where:

- For each input vector :math:`\mathbf{x}` in `input_`:
  
  .. math::
  
      \mathbf{out} = \mathbf{x} \cdot \mathbf{W}^\top + \mathbf{b}

Backward Gradient Calculation
-----------------------------

For tensors **input_**, **weight**, and **bias** involved in the `linear` operation, 
the gradients with respect to the output (**out**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_}} = \mathbf{weight}

**Gradient with respect to** :math:`\mathbf{weight}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{weight}} = \mathbf{input\_}^\top

**Gradient with respect to** :math:`\mathbf{bias}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{bias}} = \mathbf{1}

Examples
--------

Using `linear` for a simple linear transformation without bias:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # Shape: (1, 3)
    >>> weight = Tensor([[4.0, 5.0, 6.0], 
                        [7.0, 8.0, 9.0]], requires_grad=True)  # Shape: (2, 3)
    >>> out = F.linear(input_, weight)  # Shape: (1, 2)
    >>> print(out)
    Tensor([[32.0, 50.0]], grad=None)

Backpropagation computes gradients for both `input_` and `weight`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]  # Corresponding to weight
    >>> print(weight.grad)
    [[1.0, 2.0, 3.0],
     [1.0, 2.0, 3.0]]  # Corresponding to input_

Using `linear` with bias for a batch of inputs:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # Shape: (2, 2)
    >>> weight = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)  # Shape: (2, 2)
    >>> bias = Tensor([[9.0, 10.0]], requires_grad=True).reshape(1, -1)  # Shape: (1, 2)
    >>> out = F.linear(input_, weight, bias)  # Shape: (2, 2)
    >>> print(out)
    Tensor([[26.0, 33.0],
            [48.0, 75.0]], grad=None)

Backpropagation propagates gradients through the inputs, weights, and bias:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [[5.0, 6.0],
     [7.0, 8.0]]  # Corresponding to weight
    >>> print(weight.grad)
    [[1.0, 2.0],
     [3.0, 4.0]]  # Corresponding to input_
    >>> print(bias.grad)
    [1.0, 1.0]  # Corresponding to ones
