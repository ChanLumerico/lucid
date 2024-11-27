lucid.tile
==========

.. autofunction:: lucid.tile

The `tile` function constructs a new tensor by repeating elements of the input tensor 
along specified axes. It is analogous to `numpy.tile` and is useful for data augmentation, 
expanding tensor dimensions, or preparing tensors for operations that require specific 
dimensionalities.

Function Signature
------------------

.. code-block:: python

    def tile(a: Tensor, reps: int | Sequence[int]) -> Tensor

Parameters
----------

- **a** (*Tensor*): 
    The input tensor to be repeated. It can be of any shape.

- **reps** (*int* or *Sequence[int]*): 
    The number of repetitions for each element. If an integer is provided, the tensor 
    is repeated that number of times along each axis. If a sequence is provided, 
    it must match the number of dimensions of the input tensor, and each element in the 
    sequence specifies the number of repetitions for the corresponding axis.

Returns
-------

- **Tensor**: 
    A new `Tensor` with elements repeated as specified. The shape of the output tensor 
    depends on the `reps` parameter. If any of the input tensor `a` requires gradients, 
    the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `tile` operation is:

.. math::

    \mathbf{out} = \text{tile}(\mathbf{a}, \mathbf{reps})

Where each element of the input tensor `a` is repeated `reps` times along the specified axes.

Backward Gradient Calculation
-----------------------------

For the tensor **a** involved in the `tile` operation, the gradient with respect to the output (**out**) 
is computed by aggregating gradients from all the repeated elements back to their original positions 
in the input tensor.

**Gradient with respect to** :math:`\mathbf{a}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \sum \mathbf{grad\_out}

Where the gradients from all repeated instances are summed to form the gradient for each original element.

Examples
--------

Using `tile` to repeat elements of a 1D tensor:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1, 2, 3], requires_grad=True)
    >>> out = lucid.tile(a, reps=2)
    >>> print(out)
    Tensor([1, 1, 2, 2, 3, 3], grad=None)

Backpropagation computes gradients for `a` by summing the gradients from each repetition:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [2, 2, 2]

Using `tile` to repeat elements of a 2D tensor along specific axes:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2],
    ...            [3, 4]], requires_grad=True)
    >>> out = lucid.tile(a, reps=(2, 3))
    >>> print(out)
    Tensor([
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4]
    ], grad=None)

Backpropagation computes gradients for `a` by summing the gradients from 
each repetition along each axis:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [[3, 3],
     [7, 7]]
