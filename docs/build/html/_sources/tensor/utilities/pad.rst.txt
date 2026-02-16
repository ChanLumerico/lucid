lucid.pad
=========
    
.. autofunction:: lucid.pad
    
The `pad` function adds constant padding to the edges of a tensor. It is similar to 
`numpy.pad` but with the padding mode fixed to `constant`. This operation is useful 
in various neural network applications, such as preparing input data for convolutional 
layers or ensuring consistent tensor dimensions.

Function Signature
------------------
    
.. code-block:: python
    
    def pad(a: Tensor, pad_width: _ArrayLikeInt) -> Tensor

Parameters
----------
    
- **a** (*Tensor*): 
    The input tensor to be padded. It can be of any shape.
    
- **pad_width** (*_ArrayLikeInt*): 
    A sequence of integers specifying the number of values padded to the edges of each axis. 
    The length of `pad_width` must be twice the number of dimensions of `a`, as it specifies 
    padding before and after each dimension. For example, for a 2D tensor, `pad_width` should 
    include four elements: `(pad_before_dim1, pad_after_dim1, pad_before_dim2, pad_after_dim2)`.
    
Returns
-------
    
- **Tensor**: 
    A new `Tensor` with the specified padding applied. The shape of the output tensor 
    will be larger than the input tensor based on the `pad_width`. If the input tensor 
    `a` requires gradients, the resulting tensor will also require gradients.
    
Forward Calculation
-------------------
    
The forward calculation for the `pad` operation is:

.. math::
    
    \mathbf{out} = \text{constant\_pad}(\mathbf{a}, \mathbf{pad\_width}, \text{constant\_value}=0)

Where `constant_pad` adds zeros (by default) to the specified edges of the input tensor 
`a` based on `pad_width`.

Backward Gradient Calculation
-----------------------------
    
For the tensor **a** involved in the `pad` operation, the gradient with respect to the output (**out**) 
is computed by propagating gradients only to the unpadded regions of the input tensor. The padded 
regions do not receive any gradients.

**Gradient with respect to** :math:`\mathbf{a}`:

.. math::
    
    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = 
    \begin{cases}
        1 & \text{for elements corresponding to the original tensor} \\
        0 & \text{for padded elements}
    \end{cases}

Examples
--------
    
Using `pad` to add padding to a 1D tensor:
    
.. code-block:: python
    
    >>> import lucid
    >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> out = lucid.pad(a, pad_width=(2, 3))  # Pads 2 zeros before and 3 zeros after
    >>> print(out)
    Tensor([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0], grad=None)
    
Backpropagation computes gradients for `a`:
    
.. code-block:: python
    
    >>> out.backward()
    >>> print(a.grad)
    [1.0, 1.0, 1.0]

Using `pad` to add padding to a 2D tensor:
    
.. code-block:: python
    
    >>> import lucid
    >>> a = Tensor([[1.0, 2.0],
    ...            [3.0, 4.0]], requires_grad=True)
    >>> out = lucid.pad(a, pad_width=((1, 1), (2, 2)))  
    >>> print(out)
    Tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], grad=None)
    
Backpropagation computes gradients for `a`:
    
.. code-block:: python
    
    >>> out.backward()
    >>> print(a.grad)
    [[1.0, 1.0],
     [1.0, 1.0]]
