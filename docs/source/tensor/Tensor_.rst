lucid.Tensor
============

.. autoclass:: lucid.Tensor

The `Tensor` class is the fundamental data structure in the `lucid` library. 
It wraps raw data and provides support for automatic differentiation, gradient tracking, 
and GPU acceleration using Apple's Metal backend via MLX arrays. 

This class serves as the core component for nearly all mathematical and 
neural network operations in Lucid.

Class Signature
---------------

.. code-block:: python

    class Tensor(_TensorOps):
        def __init__(
            data: _ArrayOrScalar | _MLXArray,
            requires_grad: bool = False,
            keep_grad: bool = False,
            dtype: Numeric | _BuiltinNumeric | None = None,
            device: Literal["cpu", "gpu"] = "cpu",
        ) -> None

Parameters
----------

- **data** (*_ArrayOrScalar | _MLXArray*): 
  The data to be encapsulated. Can be a scalar, list, NumPy array, or MLX array.

- **requires_grad** (*bool*, optional): 
  Whether to track gradients for this tensor. Defaults to False.

- **keep_grad** (*bool*, optional): 
  If True, retains gradients after backpropagation. Defaults to False.

- **dtype** (*Numeric | _BuiltinNumeric*, optional): 
  The numeric type for tensor elements (e.g., Float32, Int64). 
  If None, inferred from input.

- **device** (*Literal["cpu", "gpu"]*, optional): 
  Specifies device placement for the tensor. Defaults to 'cpu'.

Attributes
----------

- **data** (*np.ndarray | mx.array*): 
  The underlying data representation.

- **requires_grad** (*bool*): 
  Indicates whether this tensor is being tracked by the autodiff engine.

- **keep_grad** (*bool*): 
  Controls whether gradients are retained after each backward pass.

- **grad** (*Optional[np.ndarray | mx.array]*): 
  The gradient accumulated during backpropagation.

- **device** (*str*): 
  The device where the tensor resides, either 'cpu' or 'gpu'.

- **is_leaf** (*bool*): 
  True if the tensor has no parent operations and is a leaf node 
  in the computation graph.

- **is_free** (*bool*): 
  Indicates that the tensor is *free* from any fixed device. 
  Such tensors can automatically adopt the device of their operand 
  tensor during an operation. This is useful for enabling flexible 
  device context resolution in multi-device environments.

Methods
-------

- **backward(keep_grad: bool = False) -> None**: 
  Performs reverse-mode differentiation starting from this tensor.

- **zero_grad() -> None**: 
  Resets gradients to None.

- **to(device: Literal["cpu", "gpu"]) -> Self**: 
  Moves the tensor to the specified device.

- **astype(dtype: type | Numeric) -> Self**: 
  Changes the data type of the tensor.

- **eval() -> Self**: 
  For GPU tensors, explicitly evaluates MLX operations.

- **register_hook(hook: Callable) -> Callable**: 
  Registers a hook to be executed after backpropagation.

Properties
----------

- **shape** (*tuple[int, ...]*): Shape of the tensor.

- **ndim** (*int*): Number of dimensions.

- **size** (*int*): Total number of elements.

Examples
--------

.. admonition:: Creating a tensor on GPU with gradient tracking

    .. code-block:: python

        >>> import lucid
        >>> t = lucid.Tensor([1.0, 2.0, 3.0], requires_grad=True, device="gpu")
        >>> print(t)
        [1. 2. 3.]

.. admonition:: Performing backpropagation

    .. code-block:: python

        >>> t.sum().backward()
        >>> print(t.grad)
        [1. 1. 1.]

.. admonition:: Moving tensor between devices

    .. code-block:: python

        >>> t_cpu = t.to("cpu")
        >>> print(t_cpu.device)
        'cpu'

.. admonition:: Gradient hooks

    .. code-block:: python

        >>> def print_grad(tensor, grad):
        >>>     print("Grad hooked:", grad)
        >>> t.register_hook(print_grad)

.. admonition:: Device-based dtype inference

    .. code-block:: python

        >>> t = lucid.Tensor([1, 2, 3], dtype=lucid.Float32, device="gpu")
        >>> print(t.data.dtype)
        float32

.. admonition:: Using free tensors

    .. code-block:: python

        >>> a = lucid.Tensor([1, 2, 3]).free()
        >>> b = lucid.Tensor([4, 5, 6], device="gpu")
        >>> result = a + b
        >>> print(result.device)
        'gpu'


.. note::
   Lucid tensors are hashable and indexable, support Python-style slicing,
   and are fully integrated with Lucid's reverse-mode autodiff system.

.. tip::
   Use `Tensor.is_all_free(t1, t2, ...)` to check if tensors are not bound
   to a computation graph or fixed device.

.. warning::
   Tensors with `requires_grad=True` should not be modified in-place.


Mathematical Representation
---------------------------

.. math::
    \text{grad}_i = \frac{\partial y}{\partial x_i}

Where :math:`y` is the final output and :math:`x_i` are intermediate tensors 
in the computation graph.
