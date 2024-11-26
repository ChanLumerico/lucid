nn.functional.gelu
==================

.. autofunction:: lucid.nn.functional.gelu

The `gelu` function applies the Gaussian Error Linear Unit activation function element-wise 
to the input tensor. GELU is a smooth, non-linear activation function that combines properties 
of dropout and activation functions, providing better performance in certain neural network architectures.

Function Signature
------------------

.. code-block:: python

    def gelu(input_: Tensor) -> Tensor

Parameters
----------

- **input_** (*Tensor*): 
    The input tensor of any shape.

Returns
-------

- **Tensor**: 
    A new `Tensor` where each element is the result of applying the GELU function 
    to the corresponding element in `input_`. If `input_` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `gelu` operation is:

.. math::

    \mathbf{out} = \mathbf{input\_} \cdot \Phi(\mathbf{input\_})

Where :math:`\Phi` is the cumulative distribution function of the 
standard normal distribution. 

An approximate formulation commonly used is:

.. math::

    \mathbf{out} = 0.5 \cdot \mathbf{input\_} \cdot 
    \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(\mathbf{input\_} + 
    0.044715 \cdot \mathbf{input\_}^3\right)\right)\right)

Backward Gradient Calculation
-----------------------------

For the tensor **input_** involved in the `gelu` operation, 
the gradient with respect to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{input\_}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{input\_}} = 
    \Phi(\mathbf{input\_}) + \mathbf{input\_} \cdot \phi(\mathbf{input\_})

Where :math:`\phi` is the probability density function of the standard normal distribution.

Examples
--------

Using `gelu` on a tensor:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    >>> out = F.gelu(input_)
    >>> print(out)
    Tensor([-0.1588, 0.0, 0.8413], grad=None)

Backpropagation computes gradients for `input_`:

.. code-block:: python

    >>> out.backward()
    >>> print(input_.grad)
    [0.1588, 0.5, 0.8413]
