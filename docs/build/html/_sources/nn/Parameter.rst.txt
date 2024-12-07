lucid.nn.Parameter
==================

.. autoclass:: lucid.nn.Parameter

The `Parameter` class is a specialized subclass of `Tensor`, 
designed to represent learnable parameters in a neural network. 

By default, `Parameter` instances have `requires_grad` and 
`keep_grad` set to `True`, ensuring that gradients are calculated 
and retained during backpropagation.

Class Signature
---------------

.. code-block:: python

    class Parameter(Tensor):
        def __init__(
            data: Tensor | _ArrayOrScalar, dtype: Any = np.float32
        ) -> None

Parameters
----------

- **data** (*Tensor | _ArrayOrScalar*):  
  Input data to initialize the parameter. 
  Can be a `Tensor` or any type convertible to a NumPy array.

- **dtype** (*Any*, optional):  
  Data type of the parameter's elements. Defaults to `np.float32`.

Attributes
----------

- **data** (*np.ndarray*):  
  The actual data stored in the parameter.

- **requires_grad** (*bool*):  
  Always set to `True`, indicating that this parameter will participate 
  in gradient computation.

- **keep_grad** (*bool*):  
  Always set to `True`, retaining gradients after each backpropagation pass.

- **grad** (*Optional[np.ndarray]*):  
  Gradient of the parameter, computed during backpropagation.

Methods
-------

Inherited from `Tensor`:

- **backward(keep_grad: bool = False) -> None**:  
  Performs backpropagation from this parameter, computing gradients for preceding tensors.

- **zero_grad() -> None**:  
  Resets the gradient to `None`.

Properties
----------

Inherited from `Tensor`:

- **shape** (*tuple[int, ...]*): Shape of the parameter.

- **ndim** (*int*): Number of dimensions of the parameter.

- **size** (*int*): Total number of elements in the parameter.

Examples
--------

.. tip:: **Creating a Parameter**

    Use `Parameter` to define learnable parameters in your models. 
    These parameters automatically enable gradient computation.

    .. code-block:: python

        >>> import lucid.nn as nn
        >>> p = nn.Parameter([1, 2, 3])
        >>> print(p)
        [1. 2. 3.]

.. note:: **Gradient Retention**

    Unlike regular tensors, `Parameter` objects retain their gradients 
    after each backpropagation pass, enabling easier inspection of gradients.

    .. code-block:: python

        >>> p.backward(keep_grad=False)
        >>> print(p.grad)
        [1. 1. 1.]

.. important:: **Using Parameters in Models**

    Parameters can be seamlessly integrated into models and accessed as 
    attributes for modularity and simplicity.

    .. code-block:: python

        >>> class MyModel:
        >>>     def __init__(self):
        >>>         self.param = nn.Parameter([0.5, -0.5])
        >>>
        >>> model = MyModel()
        >>> print(model.param)
        [ 0.5 -0.5]

.. hint:: **Indexing and Slicing**

    Like tensors, `Parameter` objects support indexing and slicing operations.

    .. code-block:: python

        >>> print(p[0])
        Tensor(1.0, grad=None)
