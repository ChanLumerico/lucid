optim.SGD
=========

.. autoclass:: lucid.optim.SGD

The `SGD` class implements the Stochastic Gradient Descent optimization algorithm 
with optional momentum and weight decay. It inherits from the abstract `Optimizer` 
base class and provides a straightforward method for updating model parameters based 
on their gradients.

Class Signature
---------------

.. code-block:: python

    class SGD(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1e-3,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
        ) -> None

Parameters
----------

- **Learning Rate (`lr`)**:
  Controls the step size during parameter updates. 
  A higher learning rate can speed up training but may cause instability, 
  while a lower learning rate ensures more stable convergence.

- **Momentum (`momentum`)**:
  Accelerates SGD in the relevant direction and dampens oscillations. 
  Momentum values typically range between `0.0` (no momentum) and `1.0`.

- **Weight Decay (`weight_decay`)**:
  Adds a regularization term to prevent overfitting by penalizing large weights. 
  This corresponds to L2 regularization.

Algorithm
---------

The Stochastic Gradient Descent (SGD) algorithm updates the model parameters 
based on the following formulas:

.. math::

    v_{t} &= \text{momentum} \times v_{t-1} + \text{gradient}_{t} + \text{weight_decay} \times \theta_{t}
    
    \theta_{t+1} &= \theta_{t} - \text{lr} \times v_{t}

Where:

- :math:`\theta_{t}` are the parameters at iteration :math:`t`.
- :math:`\text{gradient}_{t}` is the gradient of the loss with respect to :math:`\theta_{t}`.
- :math:`v_{t}` is the velocity (momentum buffer) at iteration :math:`t`.
- :math:`\text{lr}` is the learning rate.
- :math:`\text{momentum}` is the momentum factor.
- :math:`\text{weight_decay}` is the weight decay coefficient.

Examples
--------

.. admonition:: **Using the SGD Optimizer**
   :class: note

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

       # Initialize model and SGD optimizer
       model = MyModel()
       optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

       # Training loop
       for input, target in data:
           optimizer.zero_grad()
           output = model(input)
           loss = compute_loss(output, target)
           loss.backward()
           optimizer.step()

.. admonition:: **Inspecting Optimizer State**
   :class: tip

   Use the `state_dict()` and `load_state_dict()` methods to save and load the 
   optimizer state.

   .. code-block:: python

       # Save state
       optimizer_state = optimizer.state_dict()

       # Load state
       optimizer.load_state_dict(optimizer_state)

.. seealso::

    - `lucid.optim.Optimizer` - Abstract base class for all optimizers.
    - `lucid.nn.Module` - Base class for all neural network modules.
    - `lucid.Parameter` - Represents a parameter in a neural network module.

