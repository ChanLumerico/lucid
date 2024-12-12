nn.ParameterDict
================

.. autoclass:: lucid.nn.ParameterDict

The `ParameterDict` module is a container designed to hold a dictionary of `Parameter` objects. 
It provides a key-based access method, allowing for intuitive storage and retrieval of parameters 
by their string identifiers. This container is particularly useful for managing multiple parameters 
that are not associated with specific sub-modules, enabling their inclusion in the model's parameter 
list for optimization. 

`ParameterDict` enhances the flexibility and organization of neural network architectures, 
especially in scenarios where parameters need to be dynamically added or accessed by name.

.. warning::

    All keys in a `ParameterDict` must be unique strings. 
    Adding parameters with duplicate keys will overwrite existing parameters without warning.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ParameterDict(
        parameters: dict[str, nn.Parameter] | None = None
    ) -> None

Parameters
----------
- **parameters** (*dict[str, nn.Parameter]* | *None*, optional):
  An optional dictionary of `Parameter` objects to be added to the `ParameterDict` container upon 
  initialization. Each key must be a unique string, and each value must be an instance of 
  `nn.Parameter`. If provided, parameters are added in the order of the dictionary's keys. 
  Default is `None`.

Attributes
----------
- **parameters** (*Dict[str, nn.Parameter]*):
  The internal dictionary that stores all `Parameter` objects added to the `ParameterDict` container. 
  Parameters can be accessed by their assigned string keys, allowing for easy retrieval and 
  manipulation.

Forward Calculation
-------------------
The `ParameterDict` container itself does not define a specific forward pass. 

Instead, it serves as a dictionary of parameters that can be accessed and utilized within the 
forward method of a custom `Module`. The forward computation is defined by the user, 
who can retrieve and apply each parameter using its corresponding key.

.. math::

    \mathbf{y} = f(\mathbf{x}, \mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n)

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n` are the parameters retrieved 
  from the `ParameterDict` using their respective keys.
- :math:`f` is a user-defined function that utilizes the parameters to compute the output.
- :math:`\mathbf{y}` is the final output tensor after applying the parameters.

.. note::

    The order in which parameters are applied must be explicitly defined within the 
    forward method, as `ParameterDict` does not enforce any particular sequence.

Backward Gradient Calculation
-----------------------------
Gradients are automatically handled during backpropagation for each `Parameter` 
contained within the `ParameterDict`. Since `ParameterDict` is a container, 
it does not directly modify gradients; instead, each `Parameter` computes its own 
gradients based on the loss.

.. note::

    All `Parameter` objects added to a `ParameterDict` must support gradient computation 
    for backpropagation to function correctly. Failure to do so will prevent the model 
    from learning effectively.

Examples
--------
**Using `ParameterDict` within a custom neural network module:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a custom neural network model using ParameterDict
    >>> class CustomParameterModel(nn.Module):
    ...     def __init__(self):
    ...         super(CustomParameterModel, self).__init__()
    ...         # Initialize a dictionary of parameters
    ...         params = {
    ...             'weight1': nn.Parameter(Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)),
    ...             'bias1': nn.Parameter(Tensor([1.0, 2.0], requires_grad=True)),
    ...             'weight2': nn.Parameter(Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)),
    ...             'bias2': nn.Parameter(Tensor([3.0, 4.0], requires_grad=True))
    ...         }
    ...         self.param_dict = nn.ParameterDict(params)
    ...     
    ...     def forward(self, x):
    ...         # Apply first linear transformation
    ...         x = x @ self.param_dict['weight1'] + self.param_dict['bias1']
    ...         x = nn.ReLU()(x)
    ...         # Apply second linear transformation
    ...         x = x @ self.param_dict['weight2'] + self.param_dict['bias2']
    ...         return x
    ...
    >>> 
    >>> # Initialize the model
    >>> model = CustomParameterModel()
    >>> 
    >>> # Define input tensor
    >>> input_tensor = Tensor([[1.0, 2.0]], requires_grad=True)  # Shape: (1, 2)
    >>> 
    >>> # Forward pass
    >>> output = model(input_tensor)
    >>> print(output)
    Tensor([[21.0, 26.0]], grad=None)  # Output after applying the transformations
    >>> 
    >>> # Backpropagation
    >>> output.sum().backward()
    >>> print(input_tensor.grad)
    [[12.0, 16.0]]  # Gradients with respect to input_tensor

.. note::

    Using `ParameterDict` allows for intuitive access to parameters by their names, 
    enhancing code readability and maintainability, especially in larger models.

.. warning::

    Unlike `ModuleDict`, `ParameterDict` is specifically intended for storing parameters 
    rather than modules. It does not handle the forward pass of sub-modules. 
    
    Ensure that parameters are appropriately utilized within the forward method to avoid errors.

