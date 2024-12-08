nn.functional.leaky_relu
========================

.. autofunction:: lucid.nn.functional.leaky_relu

The `leaky_relu` function applies the Leaky Rectified Linear Unit activation function 
element-wise to the input tensor. Unlike the standard ReLU, Leaky ReLU allows a small, 
non-zero gradient when the unit is not active, which can help mitigate the "dying ReLU" 
problem.

Function Signature
------------------

.. code-block:: python

    def leaky_relu(input_: Tensor, negative_slope: float = 0.01) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of any shape.

- **negative_slope** (*float*, optional): 
    The slope of the function for input values less than zero. Default is `0.01`.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the result of applying the Leaky ReLU function 
    to the corresponding element in `input_`. If `input_` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `leaky_relu` operation is:

.. math::

    \mathbf{out} = 
    \begin{cases}
        \mathbf{input\_} & \text{if } \mathbf{input\_} > 0 \\
        \alpha \cdot \mathbf{input\_} & \text{otherwise}
    \end{cases}

Where :math:`\alpha` is the `negative_slope`.

Backward Gradient Calculation
-----------------------------

For the tensor **input_** involved in the `leaky_relu` operation, 
the gradient with respect to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_}} = 
    \begin{cases}
        1 & \text{if } \mathbf{input\_} > 0 \\
        \alpha & \text{otherwise}
    \end{cases}

Examples
--------

Using `leaky_relu` on a tensor:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    >>> out = F.leaky_relu(input_, negative_slope=0.1)
    >>> print(out)
    Tensor([-0.1, 0.0, 2.0], grad=None)

Backpropagation computes gradients for `input_`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [0.1, 0.0, 1.0]
