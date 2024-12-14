optim.RMSprop
=============
    
.. autoclass:: lucid.optim.RMSprop
    
The `RMSprop` class implements the Root Mean Square Propagation optimization algorithm 
with optional momentum and weight decay. It inherits from the abstract `Optimizer` 
base class and provides an adaptive learning rate method that adjusts the learning rate 
for each parameter based on the magnitude of recent gradients.
    
Class Signature
---------------
    
.. code-block:: python
    
    class RMSprop(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1e-2,
            alpha: float = 0.99,
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            momentum: float = 0.0,
            centered: bool = False,
        ) -> None
    
Algorithm
---------
    
The Root Mean Square Propagation (RMSprop) algorithm updates the model parameters 
based on the following formulas:
    
.. math::
    
    E[g^2]_t &= \alpha \times E[g^2]_{t-1} + (1 - \alpha) \times g_t^2
    
    \theta_{t+1} &= \theta_t - \frac{\text{lr} \times g_t}{\sqrt{E[g^2]_t + \epsilon}}
    

**If momentum is used:**

.. math::

    v_t &= \text{momentum} \times v_{t-1} + \frac{\text{lr} \times g_t}{\sqrt{E[g^2]_t + \epsilon}}
    
    \theta_{t+1} &= \theta_t - v_t
    
**If centered:**

.. math::

    E[g]_t &= \alpha \times E[g]_{t-1} + (1 - \alpha) \times g_t

    E[g^2]_t &= \alpha \times E[g^2]_{t-1} + (1 - \alpha) \times g_t^2

    \theta_{t+1} &= \theta_t - \frac{\text{lr} \times g_t}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}}
    
Where:

- :math:`\theta_{t}` are the parameters at iteration :math:`t`.
- :math:`g_t` is the gradient of the loss with respect to :math:`\theta_{t}`.
- :math:`E[g^2]_t` is the moving average of the squared gradients.
- :math:`v_t` is the velocity (momentum buffer) at iteration :math:`t`.
- :math:`\text{lr}` is the learning rate.
- :math:`\alpha` is the smoothing constant.
- :math:`\epsilon` is a small constant to prevent division by zero.
- :math:`\text{momentum}` is the momentum factor.
- :math:`\text{weight_decay}` is the weight decay coefficient.
- :math:`\text{centered}` determines whether to use the centered version of RMSprop.

Examples
--------
    
.. admonition:: **Using the RMSprop Optimizer**
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
    
       # Initialize model and RMSprop optimizer
       model = MyModel()
       optimizer = optim.RMSprop(
           model.parameters(),
           lr=0.01,
           alpha=0.99,
           eps=1e-8,
           weight_decay=1e-4,
           momentum=0.9,
           centered=True
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
    
- **Smoothing Constant (`alpha`)**:
  Decay rate for the moving average of squared gradients. Typically set close to 1 (e.g., 0.99) to give more weight to past gradients.
    
- **Epsilon (`eps`)**:
  A small constant added to the denominator to improve numerical stability and prevent division by zero.
    
- **Weight Decay (`weight_decay`)**:
  Adds a regularization term to prevent overfitting by penalizing large weights. 
  This corresponds to L2 regularization.
    
- **Momentum (`momentum`)**:
  Accelerates RMSprop in the relevant direction and dampens oscillations. 
  Momentum values typically range between `0.0` (no momentum) and `1.0`.
    
- **Centered (`centered`)**:
  If `True`, computes the centered RMSprop, which maintains an estimate of the mean of the gradients 
  and uses it to normalize the gradients, potentially leading to better convergence.
    
.. seealso::
    
    - `lucid.optim.Optimizer` - Abstract base class for all optimizers.
    - `lucid.nn.Module` - Base class for all neural network modules.
    - `lucid.Parameter` - Represents a parameter in a neural network module.