optim.Adadelta
==============

.. autoclass:: lucid.optim.Adadelta

The `Adadelta` class implements the Adadelta optimization algorithm.
Adadelta is a more robust alternative to Adagrad that limits the accumulation of 
past gradients by using a windowed accumulation of past squared gradients. 
This prevents the learning rate from shrinking to zero over time, as seen in **Adagrad**, 
while still providing adaptive learning rates.

Class Signature
---------------

.. code-block:: python

    class Adadelta(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1.0,
            rho: float = 0.9,
            eps: float = 1e-6,
            weight_decay: float = 0.0,
        ) -> None

Parameters
----------

- **Learning Rate (`lr`)**:
  Controls the step size during parameter updates. A higher learning rate
  can speed up convergence but may lead to instability.

- **Rho (`rho`)**:
  Coefficient used to calculate a running average of squared gradients and squared updates.
  A higher value keeps more of the past information.

- **Epsilon (`eps`)**:
  A small constant added to the denominator to prevent division by zero.
  This improves numerical stability during training.

- **Weight Decay (`weight_decay`)**:
  This coefficient controls the L2 penalty applied to the model parameters.
  Unlike traditional L2 regularization, weight decay in Adadelta is decoupled from the gradient calculation
  and applied directly to the parameter values.

Algorithm
---------

The Adadelta optimization algorithm updates each parameter according to the following formulas:

.. math::

    E[g^2]_{t} &= \rho \cdot E[g^2]_{t-1} + (1 - \rho) \cdot \nabla_{t}^2

    \Delta \theta_{t} &= - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}
    {\sqrt{E[g^2]_{t} + \epsilon}} \cdot \nabla_{t}

    E[\Delta \theta^2]_{t} &= \rho \cdot E[\Delta \theta^2]_{t-1} + (1 - \rho) \cdot 
    (\Delta \theta_{t})^2

    \theta_{t+1} &= \theta_{t} + \Delta \theta_{t}

Where:

- :math:`\nabla_{t}` is the gradient of the loss with respect to the parameter at iteration :math:`t`.
- :math:`E[g^2]_{t}` is the running average of the squared gradients.
- :math:`E[\Delta \theta^2]_{t}` is the running average of squared parameter updates.
- :math:`\Delta \theta_{t}` is the update to the parameter.
- :math:`\theta_{t}` is the parameter at iteration :math:`t`.
- :math:`\rho` is the decay rate for the running averages.
- :math:`\epsilon` is a small constant added for numerical stability.

The running averages :math:`E[g^2]_{t}` and :math:`E[\Delta \theta^2]_{t}` ensure that
Adadelta adapts its learning rate for each parameter, while still allowing for parameter updates
even when the gradient becomes very small.

Examples
--------

.. admonition:: **Using the Adadelta Optimizer**
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

       # Initialize model and Adadelta optimizer
       model = MyModel()
       optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.01)

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
