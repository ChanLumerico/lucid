nn.functional.selu
==================

.. autofunction:: lucid.nn.functional.selu

The `selu` function applies the Scaled Exponential Linear Unit activation function element-wise 
to the input tensor. SELU is designed to induce self-normalizing properties in neural networks, 
helping to maintain the mean and variance of activations throughout the network.

Function Signature
------------------

.. code-block:: python

    def selu(input_: Tensor) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of any shape.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the result of applying the SELU function 
    to the corresponding element in `input_`. If `input_` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `selu` operation is:

.. math::

    \mathbf{out} = \lambda \cdot 
    \begin{cases}
        \mathbf{input\_} & \text{if } \mathbf{input\_} > 0 \\
        \alpha \cdot (\exp(\mathbf{input\_}) - 1) & \text{otherwise}
    \end{cases}

Where:
- :math:`\lambda \approx 1.0507`
- :math:`\alpha \approx 1.6733`

Backward Gradient Calculation
-----------------------------

For the tensor **input_** involved in the `selu` operation, 
the gradient with respect to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_}} = \lambda \cdot 
    \begin{cases}
        1 & \text{if } \mathbf{input\_} > 0 \\
        \alpha \cdot \exp(\mathbf{input\_}) & \text{otherwise}
    \end{cases}

Examples
--------

Using `selu` on a tensor:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    >>> out = F.selu(input_)
    >>> print(out)
    Tensor([-1.7581, 0.0, 2.1014], grad=None)

Backpropagation computes gradients for `input_`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [0.3679, 1.0507, 1.0507]
