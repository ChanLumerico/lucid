lucid.optim
===========

The `lucid.optim` package provides a collection of optimization algorithms 
for training neural networks within the `lucid` library. 

It includes a foundational `Optimizer` base class and various optimizer 
implementations that adjust model parameters based on computed gradients.

Overview
--------

Optimization is a critical component in training neural networks, 
responsible for minimizing the loss function by updating the model's parameters. 
The `lucid.optim` package offers a flexible and extensible framework for implementing 
and using different optimization strategies.

Key Features
------------

- **Base Optimizer Class**: An abstract `Optimizer` class that defines the interface 
  and common functionality for all optimizers.
- **Parameter Management**: Handles parameter groups and state management, 
  facilitating complex optimization techniques.
- **State Serialization**: Supports saving and loading optimizer states, 
  enabling checkpointing and resuming training.

Getting Started
---------------

To use the `lucid.optim` package, you typically start by defining your model using 
`lucid.nn.Module`, then initialize an optimizer with the model's parameters. 

Here's a simple example:

.. code-block:: python

    import lucid.optim as optim
    import lucid.nn as nn

    # Define a simple model
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter([1.0, 2.0, 3.0])

        def forward(self, x):
            return x * self.param

    # Initialize model and optimizer
    model = MyModel()
    optimizer = optim.MyOptimizer(model.parameters(), lr=0.01)

    # Training loop
    for input, target in data:
        output = model(input)
        loss = compute_loss(output, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

Examples
--------

.. admonition:: Defining a Custom Optimizer
   :class: note

   .. code-block:: python

       import lucid.optim as optim
       import lucid.nn as nn

       class MyOptimizer(optim.Optimizer):
           def __init__(self, params, lr=0.01):
               defaults = {'lr': lr}
               super().__init__(params, defaults)

           def step(self, closure=None):
               for group in self.param_groups:
                   for param in group['params']:
                       if param.grad is not None:
                           param.data -= group['lr'] * param.grad

       # Usage
       model = nn.Module()
       # Assume model has parameters
       optimizer = MyOptimizer(model.parameters(), lr=0.01)

.. admonition:: Inspecting Optimizer State
   :class: tip

   Use the `state_dict()` and `load_state_dict()` methods to save and load 
   the optimizer state.

   .. code-block:: python

       # Save state
       optimizer_state = optimizer.state_dict()

       # Load state
       optimizer.load_state_dict(optimizer_state)

See Also
--------

- `lucid.nn.Module` - Base class for all neural network modules.
- `lucid.Parameter` - Represents a parameter in a neural network module.

References
----------

For more detailed information, refer to the documentation of individual classes 
within the `lucid.optim` package.

