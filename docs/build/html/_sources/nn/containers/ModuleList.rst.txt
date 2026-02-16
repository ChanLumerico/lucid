nn.ModuleList
=============

.. autoclass:: lucid.nn.ModuleList

The `ModuleList` module is a container that holds sub-modules in a list. 
It allows for the dynamic addition and management of modules, making it easier to 
construct complex neural network architectures where the number of layers or components may vary. 

Unlike `Sequential`, `ModuleList` does not define a fixed forward pass; instead, 
it provides a list-like interface to store and iterate over modules during the forward computation.

 `ModuleList` is useful when you need to manage an arbitrary number of modules, 
 such as in dynamic architectures, recurrent neural networks with varying lengths, 
 or when modules need to be accessed individually for operations beyond sequential execution.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ModuleList(modules: list[nn.Module] | None = None) -> None

Parameters
----------
- **modules** (*list[Module]* | *None*, optional):
  An optional list of modules to be added to the `ModuleList` container upon initialization. 
  If provided, each module in the list is appended in the order they are passed. Default is `None`.

Attributes
----------
- **modules** (*List[Module]*):
  The internal list that stores all the modules added to the `ModuleList` container. 
  Modules can be accessed by their indices or iterated over during the forward pass.

Forward Calculation
-------------------
The `ModuleList` container itself does not define a specific forward pass. 
Instead, it serves as a list of modules that can be iterated over in the forward method of 
a custom `Module`. The forward computation is defined by the user, who can loop through the 
`ModuleList` and apply each module to the input sequentially or in a specified manner.

.. math::

    \mathbf{y} = \mathbf{M}_n(\mathbf{M}_{n-1}(\dots \mathbf{M}_1(\mathbf{x}) \dots))

Where:

- :math:`\mathbf{x}` is the input tensor.
- :math:`\mathbf{M}_1, \mathbf{M}_2, \dots, \mathbf{M}_n` are the modules contained within the `ModuleList` container.
- :math:`\mathbf{y}` is the final output tensor after passing through all modules.

.. warning::

    Ensure that the output shape of each module matches the expected input shape of the subsequent 
    module to prevent runtime errors.

Backward Gradient Calculation
-----------------------------
Gradients are automatically handled during backpropagation for each module contained within 
the `ModuleList`. Since `ModuleList` is a container, it does not directly modify gradients; 
instead, each sub-module computes its own gradients based on the loss.

.. important::

    All modules added to a `ModuleList` must support gradient computation for backpropagation 
    to function correctly. Failure to do so will prevent the model from learning effectively.

Examples
--------
**Using `ModuleList` with a dynamic number of layers:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a custom neural network model using ModuleList
    >>> class DynamicNetwork(nn.Module):
    ...     def __init__(self, input_size, hidden_sizes, output_size):
    ...         super(DynamicNetwork, self).__init__()
    ...         self.layers = nn.ModuleList()
    ...         # Dynamically add hidden layers
    ...         for hidden_size in hidden_sizes:
    ...             self.layers.append(nn.Linear(input_size, hidden_size))
    ...             self.layers.append(nn.ReLU())
    ...             input_size = hidden_size
    ...         # Add the output layer
    ...         self.layers.append(nn.Linear(input_size, output_size))
    ...     
    ...     def forward(self, x):
    ...         for layer in self.layers:
    ...             x = layer(x)
    ...         return x
    ...
    >>> 
    >>> # Initialize the model with a dynamic architecture
    >>> model = DynamicNetwork(input_size=10, hidden_sizes=[20, 30], output_size=5)
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

**Using `ModuleList` to store convolutional layers:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a convolutional neural network using ModuleList
    >>> class ConvNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(ConvNetwork, self).__init__()
    ...         self.conv_layers = nn.ModuleList([
    ...             nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    ...             nn.ReLU(),
    ...             nn.MaxPool2D(kernel_size=2, stride=2),
    ...             nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    ...             nn.ReLU(),
    ...             nn.MaxPool2D(kernel_size=2, stride=2)
    ...         ])
    ...         self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=10)
    ...     
    ...     def forward(self, x):
    ...         for layer in self.conv_layers:
    ...             x = layer(x)
    ...         x = x.view(x.size(0), -1)
    ...         x = self.fc(x)
    ...         return x
    ...
    >>> 
    >>> # Initialize the model
    >>> model = ConvNetwork()
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

.. tip::

    Using `ModuleList` allows for the dynamic construction of neural network architectures, enabling the creation of models with varying depths and structures without the need for predefining the exact number of layers.

**Advanced Usage: Nested ModuleLists**

.. note::

    You can nest `ModuleList` containers within each other to create hierarchical and modular model architectures. This is particularly useful for building complex models where sub-components can be managed independently.

.. code-block:: python

    >>> import lucid.nn as nn
    >>> from lucid import Tensor
    >>> 
    >>> # Define a hierarchical neural network with nested ModuleLists
    >>> class HierarchicalNetwork(nn.Module):
    ...     def __init__(self):
    ...         super(HierarchicalNetwork, self).__init__()
    ...         self.layer_groups = nn.ModuleList()
    ...         
    ...         # Define first group of layers
    ...         group1 = nn.ModuleList([
    ...             nn.Linear(in_features=10, out_features=20),
    ...             nn.ReLU(),
    ...             nn.Linear(in_features=20, out_features=10)
    ...         ])
    ...         
    ...         # Define second group of layers
    ...         group2 = nn.ModuleList([
    ...             nn.Linear(in_features=10, out_features=5),
    ...             nn.Tanh(),
    ...             nn.Linear(in_features=5, out_features=1)
    ...         ])
    ...         
    ...         # Add groups to the main ModuleList
    ...         self.layer_groups.append(group1)
    ...         self.layer_groups.append(group2)
    ...     
    ...     def forward(self, x):
    ...         for group in self.layer_groups:
    ...             for layer in group:
    ...                 x = layer(x)
    ...         return x
    ...
    >>> 
    >>> # Initialize the model
    >>> model = HierarchicalNetwork()
    >>> 
    >>> # Define input tensor
    >>> input_tensor = Tensor([[1.0] * 10], requires_grad=True)  # Shape: (1, 10)
    >>> 
    >>> # Forward pass
    >>> output = model(input_tensor)
    >>> print(output)
    Tensor([[...]], grad=None)  # Example output after passing through the model
    >>> 
    >>> # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    # Gradients with respect to input_tensor

**Integration with Other Modules**
    
.. note::
    
    `ModuleList` can be seamlessly integrated with other container modules like `Sequential` and 
    `ModuleDict` to build more complex and flexible model architectures. 
    
    Combining these containers allows for sophisticated designs while maintaining modularity and 
    readability.

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
    Tensor([[...]], grad=None)  # Example output after passing through the model
    >>> 
    >>> # Backpropagation
    >>> output.backward()
    >>> print(input_tensor.grad)
    # Gradients with respect to input_tensor

.. caution::
    
    When using `ModuleList` with modules that have side effects or maintain internal state, 
    be mindful of how these states are managed across different forward passes. 
    Improper handling can lead to unexpected behaviors during training and inference.

.. note::
    
    You can nest `ModuleList` containers within each other to create more organized and 
    hierarchical model architectures. This can be particularly useful for dividing models 
    into smaller, manageable sub-modules.
