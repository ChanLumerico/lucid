nn.functional.elu
=================

.. autofunction:: lucid.nn.functional.elu

The `elu` function applies the Exponential Linear Unit activation function element-wise 
to the input tensor. ELU is designed to combine the benefits of ReLUs and alleviate some 
of their drawbacks by allowing negative values when the input is less than zero.

Function Signature
------------------

.. code-block:: python

    def elu(input_: Tensor, alpha: float = 1.0) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of any shape.

- **alpha** (*float*, optional): 
    The value to scale the negative factor. Default is `1.0`.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the result of applying the ELU function 
    to the corresponding element in `input_`. If `input_` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `elu` operation is:

.. math::

    \mathbf{out} = 
    \begin{cases}
        \mathbf{input\_} & \text{if } \mathbf{input\_} > 0 \\
        \alpha \cdot (\exp(\mathbf{input\_}) - 1) & \text{otherwise}
    \end{cases}

Backward Gradient Calculation
-----------------------------

For the tensor **input_** involved in the `elu` operation, 
the gradient with respect to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_}} = 
    \begin{cases}
        1 & \text{if } \mathbf{input\_} > 0 \\
        \mathbf{out} + \alpha & \text{otherwise}
    \end{cases}

Examples
--------

Using `elu` on a tensor:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([-1.0, 0.0, 2.0], requires_grad=True)
    >>> out = F.elu(input_, alpha=1.0)
    >>> print(out)
    Tensor([-0.6321, 0.0, 2.0], grad=None)

Backpropagation computes gradients for `input_`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [0.3679, 1.0, 1.0]
