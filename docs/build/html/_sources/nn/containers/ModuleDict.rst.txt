nn.ModuleDict
=============

.. autoclass:: lucid.nn.ModuleDict

The `ModuleDict` module is a container that holds sub-modules in a dictionary. 
It allows for the storage and management of modules with string keys, 
facilitating easier access and organization, especially in complex neural network architectures. 

Unlike `Sequential`, which uses an ordered list, `ModuleDict` provides a key-based access method, 
enabling more flexible and intuitive module manipulation.

.. warning::
    
    `ModuleDict` requires that all keys are unique strings. 
    Attempting to add modules with duplicate keys will overwrite existing modules without warning.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ModuleDict(modules: dict[str, nn.Module] | None = None) -> None

Parameters
----------
- **modules** (*dict[str, Module]* | *None*, optional):
  An optional dictionary of modules to be added to the `ModuleDict` container upon initialization. 
  Each key must be a unique string, and each value must be an instance of a subclass of `Module`. If provided, modules are added in the order of the dictionary's keys. Default is `None`.

Attributes
----------
- **modules** (*Dict[str, Module]*):
  The internal dictionary that stores all the modules added to the `ModuleDict` container. 
  Modules can be accessed by their assigned string keys, allowing for easy retrieval and manipulation.

Forward Calculation
-------------------
The `ModuleDict` container itself does not define a specific forward pass. 
Instead, it serves as a dictionary of modules that can be accessed and applied within the 
forward method of a custom `Module`. The forward computation is defined by the user, 
who can retrieve and apply each module using its corresponding key.

.. math::

    \mathbf{y} = \mathbf{M}_{\text{key}_n}(\mathbf{M}_{\text{key}_{n-1}}(\dots \mathbf{M}_{\text{key}_1}(\mathbf{x}) \dots))

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{M}_{\text{key}_1}, \mathbf{M}_{\text{key}_2}, \dots, \mathbf{M}_{\text{key}_n}` 
  are the modules retrieved from the `ModuleDict` using their respective keys.
- :math:`\mathbf{y}` is the final output tensor after passing through all modules.

.. note::
    
    The order in which modules are applied must be explicitly defined within the forward method, 
    as `ModuleDict` does not enforce any particular sequence.

Backward Gradient Calculation
-----------------------------
Gradients are automatically handled during backpropagation for each module contained 
within the `ModuleDict`. Since `ModuleDict` is a container, it does not directly modify gradients; 
instead, each sub-module computes its own gradients based on the loss.

.. note::
    
    All modules added to a `ModuleDict` must support gradient computation for 
    backpropagation to function correctly. Failure to do so will prevent the model from learning effectively.

Examples
--------
**Using `ModuleDict` with named modules:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> from collections import OrderedDict
    >>> 
    >>> # Define a custom neural network model using ModuleDict
    >>> class CustomNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(CustomNetwork, self).__init__()
    ...         self.layers = nn.ModuleDict({
    ...             'fc1': nn.Linear(in_features=10, out_features=20),
    ...             'activation': nn.ReLU(),
    ...             'fc2': nn.Linear(in_features=20, out_features=5)
    ...         })
    ...     
    ...     def forward(self, x):
    ...         x = self.layers['fc1'](x)
    ...         x = self.layers['activation'](x)
    ...         x = self.layers['fc2'](x)
    ...         return x
    ...
    >>> 
    >>> # Initialize the model
    >>> model = CustomNetwork()
    >>> 
    >>> # Define input tensor
    >>> input_tensor = Tensor([[1.0] * 10], requires_grad=True)  # Shape: (1, 10)
    >>> 
    >>> # Forward pass
    >>> output = model(input_tensor)
    >>> print(output)
    # Example output after passing through the model
    >>> 
    >>> # Backpropagation
    >>> output.sum().backward()
    >>> print(input_tensor.grad)
    # Gradients with respect to input_tensor

.. note::
    
    Using `ModuleDict` allows for intuitive access to modules by their names, 
    which can enhance code readability and maintainability, especially in larger models.

.. warning::
    
    Unlike `Sequential`, `ModuleDict` does not automatically apply modules in any order. 
    It is the user's responsibility to ensure that modules are applied in the correct sequence 
    within the forward method.

