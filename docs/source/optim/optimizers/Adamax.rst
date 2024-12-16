optim.Adamax
============

.. autoclass:: lucid.optim.Adamax

The `Adamax` class implements the Adamax optimization algorithm.
Adamax is a variant of Adam that uses the infinity norm (maximum absolute value)
instead of the L2 norm to control parameter updates, making it more stable for
scenarios with large gradients.

Class Signature
---------------

.. code-block:: python

    class Adamax(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 2e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
        ) -> None

Parameters
----------

- **Learning Rate (`lr`)**:
  Controls the step size during parameter updates. A higher learning rate
  can speed up convergence but may lead to instability.

- **Betas (`betas`)**:
  A tuple of two coefficients controlling the exponential decay of moment estimates.
  The first value (`beta1`) controls the decay rate for the first moment (mean of gradients),
  while the second value (`beta2`) controls the decay rate for the second moment (variance of gradients).

- **Epsilon (`eps`)**:
  A small constant added to the denominator to prevent division by zero.
  This improves numerical stability during training.

- **Weight Decay (`weight_decay`)**:
  This coefficient controls the L2 penalty applied to the model parameters.
  Unlike traditional L2 regularization, weight decay in Adamax is decoupled from the gradient calculation
  and applied directly to the parameter values.

Algorithm
---------

The Adamax optimization algorithm updates each parameter according to the following formulas:

.. math::

    m_{t} &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{t}

    u_{t} &= \max(\beta_2 \cdot u_{t-1}, |\nabla_{t}|)

    \hat{m}_{t} &= \frac{m_{t}}{1 - \beta_1^t}
    
    \theta_{t+1} &= \theta_{t} - \frac{\text{lr} \cdot \hat{m}_{t}}{u_{t} + \epsilon}

Where:

- :math:`\nabla_{t}` is the gradient of the loss with respect to the parameter at iteration :math:`t`.
- :math:`m_{t}` is the exponentially decaying first moment estimate.
- :math:`u_{t}` is the exponentially weighted infinity norm.
- :math:`\hat{m}_{t}` is the bias-corrected first moment estimate.
- :math:`\theta_{t}` is the parameter at iteration :math:`t`.
- :math:`\text{lr}` is the learning rate.

Examples
--------

.. admonition:: **Using the Adamax Optimizer**
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

       # Initialize model and Adamax optimizer
       model = MyModel()
       optimizer = optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

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
