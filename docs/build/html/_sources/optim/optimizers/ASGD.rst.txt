optim.ASGD
==========
    
.. autoclass:: lucid.optim.ASGD

The `ASGD` class implements the Averaged Stochastic Gradient Descent optimization algorithm 
with optional momentum and weight decay. It inherits from the abstract `Optimizer` 
base class and provides an advanced method for updating model parameters by averaging 
past gradients to improve convergence stability.

Class Signature
---------------
    
.. code-block:: python
    
    class ASGD(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1e-3,
            momentum: float = 0.0,
            weight_decay: float = 0.0,
            alpha: float = 0.75,
            t0: float = 1e6,
            lambd: float = 1e-4,
        ) -> None

Algorithm
---------
    
The Averaged Stochastic Gradient Descent (ASGD) algorithm updates the model parameters 
based on the following formulas:
    
.. math::
    
    v_{t} &= \text{momentum} \times v_{t-1} + \text{gradient}_{t} + \text{weight\_decay} \times \theta_{t}
    
    \theta_{t+1} &= \theta_{t} - \text{lr} \times v_{t}
    
    \bar{\theta}_{t+1} &= \alpha \times \bar{\theta}_{t} + (1 - \alpha) \times \theta_{t+1}
    
Where:

- :math:`\theta_{t}` are the parameters at iteration :math:`t`.
- :math:`\text{gradient}_{t}` is the gradient of the loss with respect to :math:`\theta_{t}`.
- :math:`v_{t}` is the velocity (momentum buffer) at iteration :math:`t`.
- :math:`\bar{\theta}_{t}` is the averaged parameter at iteration :math:`t`.
- :math:`\text{lr}` is the learning rate.
- :math:`\text{momentum}` is the momentum factor.
- :math:`\alpha` is the averaging factor.
- :math:`\text{weight_decay}` is the weight decay coefficient.
- :math:`t0` and :math:`\lambda` are additional hyperparameters controlling the averaging schedule.

**Explanation**:

1. **Velocity Update**:
   The velocity :math:`v_{t}` accumulates the gradients with momentum. The weight decay 
   term :math:`\text{weight_decay} \times \theta_{t}` applies L2 regularization, 
   penalizing large weights to prevent overfitting.

2. **Parameter Update**:
   The parameters :math:`\theta_{t}` are updated by moving them in the direction 
   opposite to the velocity, scaled by the learning rate.

3. **Averaging Parameters**:
   The averaged parameters :math:`\bar{\theta}_{t}` are computed as a weighted average 
   of the current parameters and the previous averaged parameters, controlled by the 
   averaging factor :math:`\alpha`. This averaging helps in stabilizing the convergence 
   by smoothing out the parameter updates over time.

Examples
--------
    
.. admonition:: **Using the ASGD Optimizer**
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

       # Initialize model and ASGD optimizer
       model = MyModel()
       optimizer = optim.ASGD(
           model.parameters(),
           lr=0.01,
           momentum=0.9,
           weight_decay=1e-4,
           alpha=0.75,
           t0=1e6,
           lambd=1e-4
       )

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

- **Averaging Factor (`alpha`)**:
  Controls the contribution of the current parameters to the averaged parameters. 
  A higher value gives more weight to past parameters, leading to smoother averaging.

- **Averaging Schedule (`t0`)**:
  The number of iterations before averaging starts. A larger value delays the averaging process.

- **Lambda (`lambd`)**:
  A regularization parameter that controls the decay rate of the averaging process.

.. seealso::

    - `lucid.optim.Optimizer` - Abstract base class for all optimizers.
    - `lucid.nn.Module` - Base class for all neural network modules.
    - `lucid.Parameter` - Represents a parameter in a neural network module.
