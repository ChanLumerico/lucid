lucid.nn.Buffer
===============

.. autoclass:: lucid.nn.Buffer

The `Buffer` class is a specialized tensor used within neural network modules 
to store tensors that are not intended to be trained (i.e., they do not require gradients). 

Buffers are useful for maintaining state information that should be part of the 
model's state but should not be updated during training, such as running averages, 
masks, or other auxiliary data.

Unlike parameters, buffers are not updated by the optimizer. However, they are saved 
and loaded alongside the model's parameters, ensuring consistency during model serialization 
and deserialization.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.Buffer(data: Tensor | _ArrayOrScalar, dtype=np.float32)

Parameters
----------
- **data** (*Tensor* or *_ArrayOrScalar*):
    The initial data to store in the buffer. This can be a tensor or any 
    array-like structure that can be converted to a tensor.

- **dtype** (*numpy.dtype*, optional):
    The desired data type of the tensor. Default is `numpy.float32`.

Attributes
----------
- **data** (*Tensor*):
    The tensor stored in the buffer. This tensor does not require gradients 
    and is not updated during training.

Usage
-----
Buffers are typically used within custom neural network modules to store state 
information that should persist across training iterations but should not be 
treated as learnable parameters.

Examples
--------

**Creating a Buffer and accessing its data:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> # Initialize a buffer with a tensor
    >>> buffer = nn.Buffer(Tensor([1.0, 2.0, 3.0]))
    >>> print(buffer)
    Tensor([1.0, 2.0, 3.0], requires_grad=False)

**Using `Buffer` within a custom neural network module:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> class CustomModule(nn.Module):
    ...     def __init__(self):
    ...         super(CustomModule, self).__init__()
    ...         # Register a buffer to store running mean
    ...         self.register_buffer('running_mean', nn.Buffer(Tensor([0.0, 0.0, 0.0])))
    ...     
    ...     def forward(self, x):
    ...         # Example operation using the buffer
    ...         return x + self.running_mean
    ...
    >>> model = CustomModule()
    >>> input_tensor = Tensor([1.0, 2.0, 3.0], requires_grad=True)  # Shape: (3,)
    >>> output = model(input_tensor)
    >>> print(output)
    Tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Backpropagation
    >>> output.backward(Tensor([1.0, 1.0, 1.0]))
    >>> print(input_tensor.grad)
    [1.0, 1.0, 1.0]  # Gradients with respect to input_tensor
    >>> print(model.running_mean)
    Tensor([0.0, 0.0, 0.0], requires_grad=False)  # Buffer remains unchanged
