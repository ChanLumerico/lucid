Tensor
======

The `Tensor` class is a custom implementation resembling PyTorch's `Tensor`, designed to support automatic differentiation with a similar interface, using NumPy as a backend. This class provides functionality for tensor operations, tracking gradients, and constructing a computational graph.

Class Definition
----------------

.. autoclass:: lucid.Tensor
    :members:
    :undoc-members:
    :show-inheritance:

Usage
-----

This `Tensor` class is initialized with data, and optionally allows setting flags for gradient requirements and data type. It includes gradient tracking and a `backward` method to compute gradients for operations that use the tensor.

Constructor Parameters
----------------------

- **data** (`_ArrayOrScalar`): The initial data for the tensor, converted to a NumPy array if not already.
- **requires_grad** (`bool`, optional): Whether to track gradients for this tensor. Defaults to `False`.
- **keep_grad** (`bool`, optional): Whether to retain gradient information after the `backward` pass. Defaults to `False`.
- **dtype** (`Any`, optional): Data type of the tensor. Defaults to `np.float32`.

Attributes
----------

- **data** (`_NumPyArray`): The tensor’s underlying data, stored as a NumPy array.
- **requires_grad** (`bool`): Indicates if this tensor requires gradients.
- **keep_grad** (`bool`): Determines whether gradients are retained after `backward`.
- **dtype** (`Any`): Data type of the tensor.
- **grad** (`Optional[_NumPyArray]`): Holds the gradient of the tensor, if applicable.
- **_backward_op** (`callable`): A function that computes the gradient for this tensor.
- **_prev** (`list[Tensor]`): List of tensors from which this tensor was derived, for constructing the computational graph.

Properties
----------

- **is_leaf** (`bool`): Returns `True` if the tensor is a leaf node in the computational graph, meaning it was not derived from other tensors.
- **shape** (`tuple[int, ...]`): Shape of the tensor data.
- **ndim** (`int`): Number of dimensions of the tensor.
- **size** (`int`): Total number of elements in the tensor.

Methods
-------

- **backward(keep_grad: bool = False) -> None**
  
  Computes gradients for all tensors involved in producing this tensor. Builds the computational graph in topological order and calls `_backward_op` for each node.

  - **keep_grad** (`bool`, optional): If `False`, clears the gradient after the backward pass, unless `keep_grad` is `True` for this tensor. Defaults to `False`.

- **zero_grad() -> None**
  
  Clears the gradient for this tensor. If `keep_grad` is `False`, `grad` is set to `None`.

- **__getitem__(idx: SupportsIndex) -> Tensor**
  
  Returns a new tensor representing a slice of the original tensor, retaining gradient tracking for backward operations.

  - **idx** (`SupportsIndex`): The index or slice to retrieve.

- **__iter__() -> Iterator[Tensor]**
  
  Allows iteration over the tensor along the first dimension, yielding individual tensor slices.

- **__repr__() -> str**
  
  Returns a string representation of the tensor, displaying data and gradient.

- **__str__() -> str**
  
  Returns a string representation of the tensor data.

- **__hash__() -> int**
  
  Provides a unique hash based on the tensor’s id.

Examples
--------

Creating a tensor, setting gradients, and computing a backward pass:

.. code-block:: python

    import numpy as np
    from lucid import Tensor

    # Initialize tensor with data and gradient tracking
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
    
    # Perform an operation
    c = a + b

    # Compute gradients
    c.backward()

    # Access gradients
    print(a.grad)  # Displays gradient with respect to `a`
    print(b.grad)  # Displays gradient with respect to `b`

.. note::

    If `requires_grad` is set to `False`, the tensor will not track gradients or participate in the computational graph. Leaf tensors in the graph are nodes that are not derived from other tensors, and gradients can only be directly assigned to these leaf nodes.

"""
