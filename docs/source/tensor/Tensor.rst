lucid.Tensor
============

.. autoclass:: lucid.Tensor

The `Tensor` class is a core component in the `lucid` library, providing support for data storage, gradient tracking, and automatic differentiation. The class encapsulates data in an `ndarray` and supports backpropagation, making it a foundation for neural network operations.

Class Signature
---------------

.. code-block:: python

    class Tensor(_TensorOps):
        def __init__(
            data: _ArrayOrScalar,
            requires_grad: bool = False,
            keep_grad: bool = False,
            dtype: Any = np.float32,
        ) -> None

Parameters
----------

- **data** (*_ArrayOrScalar*): Input data to be encapsulated as a Tensor. Accepts data types that can be converted to a NumPy array.
- **requires_grad** (*bool*, optional): Enables gradient tracking if set to True. Defaults to False.
- **keep_grad** (*bool*, optional): If True, retains gradient after each backward pass. Defaults to False.
- **dtype** (*Any*, optional): Data type of the tensor elements. Defaults to `np.float32`.

Attributes
----------

- **data** (*np.ndarray*): The actual data stored in the tensor.
- **requires_grad** (*bool*): Indicates whether this tensor requires gradient calculation.
- **keep_grad** (*bool*): If set, gradients are retained after each backpropagation pass.
- **grad** (*Optional[np.ndarray]*): Gradient of the tensor, computed during backpropagation if `requires_grad` is True.
- **is_leaf** (*bool*): True if the tensor is a leaf node in the computation graph, with no other tensors feeding into it.

Methods
-------

- **backward(keep_grad: bool = False) -> None**: Performs backpropagation from this tensor, computing gradients for each preceding tensor. Clears gradients unless `keep_grad` is True.

- **zero_grad() -> None**: Resets the gradient to None if `keep_grad` is False.

- **__getitem__(idx: SupportsIndex) -> Self**: Slices the tensor and returns a new `Tensor` for the specified indices. Supports gradient propagation for sliced tensors.

- **__iter__() -> Iterator[Self]**: Allows iteration over the tensor, yielding one element at a time.

Properties
----------

- **shape** (*tuple[int, ...]*): Shape of the tensor.
- **ndim** (*int*): Number of dimensions of the tensor.
- **size** (*int*): Total number of elements in the tensor.

Examples
--------

Creating a tensor with gradient tracking:

.. code-block:: python

    >>> import lucid
    >>> t = lucid.Tensor([1, 2, 3], requires_grad=True)
    >>> print(t)
    [1. 2. 3.]

Performing backpropagation:

.. code-block:: python

    >>> t.backward()
    >>> print(t.grad)
    [1. 1. 1.]

Indexing and slicing:

.. code-block:: python

    >>> t[0]
    Tensor(1, grad=None)

Iterating through a tensor:

.. code-block:: python

    >>> for sub_tensor in t:
    >>>     print(sub_tensor)
    Tensor(1, grad=None)
    Tensor(2, grad=None)
    Tensor(3, grad=None)

