nn.Sequential
=============
    
.. autoclass:: lucid.nn.Sequential
    
The `Sequential` module is a container that allows for the stacking of 
multiple neural network modules in a sequential manner. 

It facilitates the creation of complex models by chaining together layers 
and other modules, ensuring that the output of one module serves as the input to the next. 
This modular approach promotes code reusability and simplifies the architecture of neural 
networks, making it easier to build, understand, and maintain models.
    
**Admonition: Note**
    
    `Sequential` is ideal for models where each layer has exactly one input tensor 
    and one output tensor. It does not support models with multiple inputs or outputs, 
    or layers that require additional arguments during the forward pass.

Class Signature
---------------
.. code-block:: python
    
    class lucid.nn.Sequential(
        *modules: Module
    ) -> None
    
    class lucid.nn.Sequential(
        ordered_dict: OrderedDict[str, Module]
    ) -> None
    
Parameters
----------
- **\*modules** (*Module*):
  A variable number of modules to be added to the `Sequential` container. 
  Each module is appended in the order they are passed.
    
- **ordered_dict** (*OrderedDict[str, Module]*):
  An ordered dictionary of modules where each key is a string representing 
  the name of the module, and each value is the corresponding module instance. 
  This allows for explicit naming and ordering of modules within the container.
    
Attributes
----------
- **modules** (*OrderedDict[str, Module]*):
  An ordered dictionary containing all the modules added to the `Sequential` container. 
  Modules can be accessed by their assigned names or indices.
    
Forward Calculation
-------------------
The `Sequential` module applies each contained module in the order they were added. 

The forward pass is defined as:
    
.. math::
    
    \mathbf{y} = \mathbf{M}_n(\mathbf{M}_{n-1}(\dots \mathbf{M}_1(\mathbf{x}) \dots))
    
Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{M}_1, \mathbf{M}_2, \dots, \mathbf{M}_n` 
  are the modules contained within the `Sequential` container.
- :math:`\mathbf{y}` is the final output tensor after passing through all modules.
    
.. warning::
    
    Ensure that the output shape of each module matches the expected input shape 
    of the subsequent module to prevent runtime errors.

Backward Gradient Calculation
-----------------------------
During backpropagation, gradients are propagated through each module in the reverse order 
of their addition. This sequential gradient flow ensures that each module's parameters are 
updated appropriately to minimize the loss function.
    
.. important::
    
    The `Sequential` container does not alter the gradient flow. 
    All contained modules must support gradient computation for backpropagation 
    to function correctly.

Examples
--------
**Using `Sequential` with positional modules:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a simple Sequential model
    >>> model = nn.Sequential(
    ...     nn.Linear(in_features=4, out_features=3),
    ...     nn.ReLU(),
    ...     nn.Linear(in_features=3, out_features=2)
    ... )
    >>> 
    >>> # Define input tensor
    >>> input_tensor = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)  # Shape: (1, 4)
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
    
**Using `Sequential` with an `OrderedDict`:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> from collections import OrderedDict
    >>> 
    >>> # Define a Sequential model using OrderedDict
    >>> model = nn.Sequential(OrderedDict([
    ...     ('fc1', nn.Linear(in_features=5, out_features=3)),
    ...     ('activation', nn.Tanh()),
    ...     ('fc2', nn.Linear(in_features=3, out_features=1))
    ... ]))
    >>> 
    >>> # Define input tensor
    >>> input_tensor = Tensor([[0.5, -1.2, 3.3, 0.0, 2.1]], requires_grad=True)  # Shape: (1, 5)
    >>> 
    >>> # Forward pass
    >>> output = model(input_tensor)
    >>> print(output)
    # Example output after passing through the model
    >>> 
    >>> # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    # Gradients with respect to input_tensor
    
.. tip::
    
    Using `OrderedDict` allows you to assign meaningful names to each module, 
    which can be beneficial for debugging and model introspection.

**Using `Sequential` within a Complex Neural Network:**
    
.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> from collections import OrderedDict
    >>> 
    >>> # Define a more complex Sequential model
    >>> model = nn.Sequential(OrderedDict([
    ...     ('conv1', nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)),
    ...     ('batch_norm1', nn.BatchNorm2d(num_features=16)),
    ...     ('relu1', nn.ReLU()),
    ...     ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
    ...     ('conv2', nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)),
    ...     ('batch_norm2', nn.BatchNorm2d(num_features=32)),
    ...     ('relu2', nn.ReLU()),
    ...     ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
    ...     ('flatten', nn.Flatten()),
    ...     ('fc1', nn.Linear(in_features=32 * 8 * 8, out_features=128)),
    ...     ('relu3', nn.ReLU()),
    ...     ('fc2', nn.Linear(in_features=128, out_features=10))
    ... ]))
    >>> 
    >>> # Define input tensor
    >>> input_tensor = Tensor([
    ...     [
    ...         [[1.0, 2.0, 3.0, ..., 32.0],
    ...          [33.0, 34.0, 35.0, ..., 64.0],
    ...          ...
    ...          [1025.0, 1026.0, 1027.0, ..., 1056.0]]
    ...     ]
    ... ], requires_grad=True)  # Shape: (1, 3, 32, 32)
    >>> 
    >>> # Forward pass
    >>> output = model(input_tensor)
    >>> print(output)
    # Output tensor after passing through the model
    >>> 
    >>> # Backpropagation
    >>> output.sum().backward()
    >>> print(input_tensor.grad)
    # Gradients with respect to input_tensor
    
.. warning::
    
    The `Sequential` container does not support modules that have multiple inputs or outputs. 
    Attempting to include such modules will result in errors during the forward pass. 
    
    For more complex architectures that require multiple inputs or outputs, 
    consider using custom `Module` classes instead.

.. note::
    
    You can nest `Sequential` containers within each other to create more organized and 
    hierarchical model architectures. This can be particularly useful for dividing models 
    into smaller, manageable sub-modules.

    While `Sequential` primarily supports static architectures defined at initialization, 
    you can dynamically add modules to an existing `Sequential` container using methods 
    like `add_module`. This allows for more flexibility in model design.

.. tip::
    
    When debugging models built with `Sequential`, you can access individual modules by 
    their names (if using `OrderedDict`) or by their indices. This facilitates targeted 
    inspection and modification of specific layers within the model.

**Advanced Usage: Dynamic Module Addition**

.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Initialize an empty Sequential container
    >>> model = nn.Sequential()
    >>> 
    >>> # Dynamically add modules
    >>> model.add_module('fc1', nn.Linear(in_features=10, out_features=5))
    >>> model.add_module('relu', nn.ReLU())
    >>> model.add_module('fc2', nn.Linear(in_features=5, out_features=2))
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

.. caution::
    
    When using `Sequential` with modules that have side effects or maintain internal state, 
    be mindful of how these states are managed across different forward passes. 
    
    Improper handling can lead to unexpected behaviors during training and inference.

**Integration with Other Modules**
    
.. note::
    
    `Sequential` can be seamlessly integrated with other container modules like 
    `ModuleList` and `ModuleDict` to build more complex and flexible model architectures. 
    Combining these containers allows for sophisticated designs while maintaining modularity and readability.

.. code-block:: python
    
    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> from collections import OrderedDict
    >>> 
    >>> # Define a ModuleList
    >>> layers = nn.ModuleList([
    ...     nn.Linear(in_features=10, out_features=20),
    ...     nn.ReLU(),
    ...     nn.Linear(in_features=20, out_features=10)
    ... ])
    >>> 
    >>> # Define a Sequential container that includes the ModuleList
    >>> model = nn.Sequential(OrderedDict([
    ...     ('layer_sequence', layers),
    ...     ('output_layer', nn.Linear(in_features=10, out_features=1))
    ... ]))
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
    >>> output.backward()
    >>> print(input_tensor.grad)
    # Gradients with respect to input_tensor
