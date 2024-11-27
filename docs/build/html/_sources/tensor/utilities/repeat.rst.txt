lucid.repeat
============
    
.. autofunction:: lucid.repeat
    
The `repeat` function constructs a new tensor by repeating elements of the input tensor 
along specified axes. This operation is analogous to `numpy.repeat` and is useful for 
data augmentation, expanding tensor dimensions, or preparing tensors for operations 
that require specific dimensionalities.
    
Function Signature
------------------
    
.. code-block:: python
    
    def repeat(a: Tensor, repeats: int | Sequence[int], axis: int | None = None) -> Tensor
    
Parameters
----------
    
- **a** (*Tensor*): 
    The input tensor to be repeated. It can be of any shape.
    
- **repeats** (*int* or *Sequence[int]*): 
    The number of repetitions for each element. If an integer is provided, the elements 
    are repeated that number of times along the specified axis. If a sequence is provided, 
    it must match the number of dimensions of the input tensor, and each element in the 
    sequence specifies the number of repetitions for the corresponding axis.
    
- **axis** (*int*, optional): 
    The axis along which to repeat the elements. If `axis` is `None`, the input tensor is 
    flattened, and the repetition is applied to the flattened array. Default is `None`.
    
Returns
-------
    
- **Tensor**: 
    A new `Tensor` with elements repeated as specified. The shape of the output tensor depends 
    on the `repeats` and `axis` parameters. If any of the input tensor `a` requires gradients, 
    the resulting tensor will also require gradients.
    
Forward Calculation
-------------------
    
The forward calculation for the `repeat` operation is:
    
.. math::
    
    \mathbf{out} = \text{repeat}(\mathbf{a}, \text{repeats}, \text{axis})
    
Where each element of the input tensor `a` is repeated `repeats` times along the specified `axis`.
    
Backward Gradient Calculation
-----------------------------
    
For the tensor **a** involved in the `repeat` operation, the gradient with respect to the output (**out**) 
is computed by aggregating gradients from all the repeated elements back to their original positions 
in the input tensor.
    
**Gradient with respect to** :math:`\mathbf{a}`:
    
.. math::
    
    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \sum_{i} \mathbf{grad\_out}_i
    
Where the gradients from all repeated instances are summed to form the gradient 
for each original element.
    
Examples
--------
    
Using `repeat` to repeat elements of a 1D tensor:
    
.. code-block:: python
    
    >>> import lucid
    >>> a = Tensor([1, 2, 3], requires_grad=True)
    >>> out = lucid.repeat(a, repeats=2, axis=0)
    >>> print(out)
    Tensor([1, 1, 2, 2, 3, 3], grad=None)
    
Backpropagation computes gradients for `a` by summing the gradients from each repetition:
    
.. code-block:: python
    
    >>> out.backward()
    >>> print(a.grad)
    [2, 2, 2]
    
Using `repeat` to repeat elements of a 2D tensor along a specific axis:
    
.. code-block:: python
    
    >>> import lucid
    >>> a = Tensor([[1, 2],
    ...            [3, 4]], requires_grad=True)
    >>> out = lucid.repeat(a, repeats=3, axis=1)
    >>> print(out)
    Tensor([
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4]
    ], grad=None)
    
Backpropagation computes gradients for `a` by summing the gradients from 
each repetition along the specified axis:
    
.. code-block:: python
    
    >>> out.backward()
    >>> print(a.grad)
    [[3, 3],
     [6, 6]]
