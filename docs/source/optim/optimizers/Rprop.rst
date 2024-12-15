optim.Rprop
===========

.. autoclass:: lucid.optim.Rprop

The `Rprop` class implements the Resilient Propagation (Rprop) optimization algorithm.
It adjusts the step size for each parameter individually based on the sign of the gradient change
from the previous iteration. Unlike standard gradient-based methods, Rprop only considers the
direction of the gradient, not its magnitude, leading to more stable updates.

Class Signature
---------------

.. code-block:: python

    class Rprop(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1e-2,
            etas: tuple[float, float] = (0.5, 1.2),
            step_sizes: tuple[float, float] = (1e-6, 50.0),
        ) -> None

Parameters
----------

- **Learning Rate (`lr`)**:
  Initial step size for the parameter updates. While the step size changes adaptively,
  this initial value serves as the starting step size for all parameters.

- **Etas (`etas`)**:
  A tuple of two factors controlling the adjustment of the step size.
  When the sign of the gradient change is positive, the step size is multiplied by `etas[1]`.
  When the sign of the gradient change is negative, the step size is multiplied by `etas[0]`.

- **Step Sizes (`step_sizes`)**:
  A tuple of two values representing the lower and upper bounds for the step size.
  These bounds ensure that the step size does not become too small or too large.

Algorithm
---------

The Resilient Propagation (Rprop) algorithm updates each parameter based 
on the following formulas:

.. math::

    s_{t+1} &= \begin{cases}
      s_{t} \cdot \eta_{+}, & \text{if} \ \nabla_{t} \cdot \nabla_{t-1} > 0 \\
      s_{t} \cdot \eta_{-}, & \text{if} \ \nabla_{t} \cdot \nabla_{t-1} < 0 \\
      s_{t}, & \text{otherwise}
    \end{cases}
    
    \nabla_{t} &= \text{gradient at iteration} \ t
    
    \theta_{t+1} &= \theta_{t} - \text{sign}(\nabla_{t}) \cdot s_{t+1}

Where:

- :math:`\nabla_{t}` is the gradient of the parameter at iteration :math:`t`.
- :math:`s_{t}` is the current step size for the parameter.
- :math:`\eta_{+}` and :math:`\eta_{-}` are the factors used to increase or decrease the step size.
- :math:`\theta_{t}` is the value of the parameter at iteration :math:`t`.

If the sign of the gradient changes (i.e., from positive to negative or vice versa), the step size is reduced
by multiplying it by :math:`\eta_{-}`. If the gradient sign remains the same, the step size is increased
by multiplying it by :math:`\eta_{+}`. The step size is clipped between the `step_sizes` bounds.

Examples
--------

.. admonition:: **Using the Rprop Optimizer**
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

       # Initialize model and Rprop optimizer
       model = MyModel()
       optimizer = optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0))

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
