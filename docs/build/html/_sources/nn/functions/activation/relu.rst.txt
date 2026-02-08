nn.functional.relu
==================

.. autofunction:: lucid.nn.functional.relu

The `relu` function applies the Rectified Linear Unit activation function element-wise 
to the input tensor. This non-linear activation function is widely used in neural networks 
to introduce non-linearity, allowing the network to learn complex patterns.

Function Signature
------------------

.. code-block:: python

    def relu(input_: Tensor) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of any shape.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the result of applying the ReLU function 
    to the corresponding element in `input_`. If `input_` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `relu` operation is:

.. math::

    \mathbf{out} = \max(0, \mathbf{input\_})

Backward Gradient Calculation
-----------------------------

For the tensor **input_** involved in the `relu` operation, the gradient with respect 
to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_}} = 
    \begin{cases}
        1 & \text{if } \mathbf{input\_} > 0 \\
        0 & \text{otherwise}
    \end{cases}

Examples
--------

Using `relu` on a tensor:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    >>> out = F.relu(input_)
    >>> print(out)
    Tensor([0.0, 0.0, 2.0], grad=None)

Backpropagation computes gradients for `input_`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [0.0, 0.0, 1.0]
