nn.Parameter
============

.. autoclass:: lucid.nn.Parameter

The `Parameter` class is a specialized subclass of `Tensor`, 
designed to represent learnable parameters in a neural network. 

By default, `Parameter` instances have `requires_grad` and `keep_grad` set to `True`, 
ensuring that gradients are calculated and retained during backpropagation.

The `Parameter` class allows parameters to be easily registered in `Module` classes 
and included in their state dictionaries for saving and loading models.

Class Signature
---------------

.. code-block:: python

    class Parameter(Tensor):
        def __init__(
            data: Tensor | _ArrayOrScalar,
            dtype: Any = np.float32
        ) -> None

Parameters
----------

- **data** (*Tensor | _ArrayOrScalar*):  
  Input data to initialize the parameter. Can be a `Tensor` or any type convertible to a NumPy array.

- **dtype** (*Any*, optional):  
  Data type of the parameter's elements. Defaults to `np.float32`.

Attributes
----------

- **data** (*np.ndarray*):  
  The actual data stored in the parameter.

- **requires_grad** (*bool*):  
  Always set to `True`, indicating that this parameter will participate in gradient computation.

- **keep_grad** (*bool*):  
  Always set to `True`, retaining gradients after each backpropagation pass.

- **grad** (*Optional[np.ndarray]*):  
  Gradient of the parameter, computed during backpropagation.

Methods
-------

.. code-block:: python

    def backward(self, keep_grad: bool = False) -> None

Performs backpropagation from this parameter, computing gradients for preceding tensors.

**Parameters**:

- **keep_grad** (*bool*, optional): 
  Whether to retain the gradient after backpropagation. Defaults to `False`.

.. code-block:: python

    def zero_grad(self) -> None

Resets the gradient to `None`. Useful for clearing gradients before a new optimization step.

Properties
----------

- **shape** (*tuple[int, ...]*):  
  Shape of the parameter.

- **ndim** (*int*):  
  Number of dimensions of the parameter.

- **size** (*int*):  
  Total number of elements in the parameter.

Examples
--------

.. admonition:: **Creating a Parameter**
   :class: note

   .. code-block:: python

       import lucid.nn as nn

       p = nn.Parameter([1.0, 2.0, 3.0])
       print(p)
       # Output: [1.0, 2.0, 3.0]

.. tip:: **Using parameters in a model**

   Parameters can be seamlessly integrated into models and accessed as attributes.

   .. code-block:: python

       class MyModel(nn.Module):
           def __init__(self):
               super().__init__()
               self.param = nn.Parameter([0.5, -0.5])

       model = MyModel()
       print(model.param)
       # Output: [0.5, -0.5]

.. important:: **Retaining gradients**

   By default, `Parameter` retains gradients during backpropagation, 
   enabling inspection or re-use.

   .. code-block:: python

       p = nn.Parameter([1.0, 2.0, 3.0])
       p.backward()
       print(p.grad)
       # Output: [1.0, 1.0, 1.0]

.. hint:: **State dictionary inclusion**

   Parameters are automatically included in a `Module`'s `state_dict`, 
   allowing for easy saving and loading of model parameters.

   .. code-block:: python

       model = MyModel()
       state_dict = model.state_dict()
       print(state_dict)
       # Output: {'param': nn.Parameter([...])}
