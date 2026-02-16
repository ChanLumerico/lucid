nn.functional.sigmoid
=====================

.. autofunction:: lucid.nn.functional.sigmoid

The `sigmoid` function applies the Sigmoid activation function element-wise to the input tensor. 
This function maps input values to the range (0, 1) and is commonly used in binary classification 
tasks and as activation functions in neural networks.

Function Signature
------------------

.. code-block:: python

    def sigmoid(input_: Tensor) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of any shape.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the result of applying the Sigmoid function 
    to the corresponding element in `input_`. If `input_` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `sigmoid` operation is:

.. math::

    \mathbf{out} = \frac{1}{1 + \exp(-\mathbf{input\_})}

Backward Gradient Calculation
-----------------------------

For the tensor **input_** involved in the `sigmoid` operation, 
the gradient with respect to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_}} = 
    \mathbf{out} \cdot (1 - \mathbf{out})

Examples
--------

Using `sigmoid` on a tensor:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([-2.0, 0.0, 3.0], requires_grad=True)
    >>> out = F.sigmoid(input_)
    >>> print(out)
    Tensor([0.1192, 0.5, 0.9526], grad=None)

Backpropagation computes gradients for `input_`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [0.10499, 0.25, 0.0474259]
