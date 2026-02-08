lucid.max
=========

.. autofunction:: lucid.max

The `max` function computes the maximum value of a tensor along specified axes. 
It is used to perform reduction operations, extracting the largest elements from the input tensor 
based on the provided axis or axes. This function is essential in various neural network operations, 
such as finding the maximum activation or during certain pooling operations.

Function Signature
------------------

.. code-block:: python

    def max(a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False) -> Tensor

Parameters
----------

- **a** (*Tensor*): 
    The input tensor from which to compute the maximum values.

- **axis** (*int* or *tuple of int*, optional): 
    The axis or axes along which to compute the maximum. If `None`, 
    the maximum is computed over all elements 
    of the input tensor. Default is `None`.

- **keepdims** (*bool*, optional): 
    Whether to retain the reduced dimensions with size one. 
    If `True`, the output tensor will have the same 
    number of dimensions as the input tensor, with the reduced dimensions set to size one. 
    Default is `False`.

Returns
-------

- **Tensor**: 
    A new `Tensor` containing the maximum values. 
    The shape of the output tensor depends on the `axis` and `keepdims` parameters. 
    If any of `a` requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `max` operation is:

.. math::

    \mathbf{out} = \max(\mathbf{a}, \text{axis}= \text{axis}, 
    \text{keepdims}= \text{keepdims})

Where the maximum is computed along the specified axis or axes.

Backward Gradient Calculation
-----------------------------

For the tensor **a** involved in the `max` operation, 
the gradient with respect to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{a}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = 
    \begin{cases}
        1 & \text{if } \mathbf{a} \text{ is the maximum in its reduction window} \\
        0 & \text{otherwise}
    \end{cases}

This means that the gradient is propagated only to the 
elements that are the maximum in their respective reduction windows.

Examples
--------

Using `max` to find the maximum value in a tensor:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([3.0, 1.0, 2.0], requires_grad=True)
    >>> out = lucid.max(a)
    >>> print(out)
    Tensor(3.0, grad=None)

Backpropagation computes gradients for `a`:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [1.0, 0.0, 0.0]

Using `max` along a specific axis without `keepdims`:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[3.0, 1.0, 2.0],
    ...            [4.0, 5.0, 6.0]], requires_grad=True)
    >>> out = lucid.max(a, axis=0)
    >>> print(out)
    Tensor([4.0, 5.0, 6.0], grad=None)

Backpropagation computes gradients for `a`:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [[0.0, 0.0, 0.0],
     [1.0, 1.0, 1.0]]
