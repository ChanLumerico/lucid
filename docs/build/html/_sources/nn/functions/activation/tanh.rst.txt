nn.functional.tanh
==================

.. autofunction:: lucid.nn.functional.tanh

The `tanh` function applies the hyperbolic tangent activation function element-wise 
to the input tensor. This function maps input values to the range (-1, 1) and is 
commonly used to normalize data within neural networks.

Function Signature
------------------

.. code-block:: python

    def tanh(input_: Tensor) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of any shape.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the result of applying the Tanh function 
    to the corresponding element in `input_`. If `input_` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `tanh` operation is:

.. math::

    \mathbf{out} = \tanh(\mathbf{input\_}) = \frac{\exp(\mathbf{input\_}) - 
    \exp(-\mathbf{input\_})}{\exp(\mathbf{input\_}) + \exp(-\mathbf{input\_})}

Backward Gradient Calculation
-----------------------------

For the tensor **input_** involved in the `tanh` operation, 
the gradient with respect to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_}} = 1 - \mathbf{out}^2

Examples
--------

Using `tanh` on a tensor:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    >>> out = F.tanh(input_)
    >>> print(out)
    Tensor([-0.7616, 0.0, 0.7616], grad=None)

Backpropagation computes gradients for `input_`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [0.419974, 1.0, 0.419974]
