optim.NAdam
===========

.. autoclass:: lucid.optim.NAdam

The `NAdam` class implements the Nesterov-accelerated Adaptive Moment Estimation (NAdam) 
optimization algorithm. NAdam combines the benefits of Adam and Nesterov momentum to achieve 
faster convergence and better generalization.

It also includes support for momentum decay, which applies an additional decay to the 
momentum buffer, leading to better long-term convergence.

Class Signature
---------------

.. code-block:: python

    class NAdam(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            momentum_decay: float = 0.004,
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
  Unlike traditional L2 regularization in Adam, weight decay in NAdam is decoupled 
  from the gradient calculation and applied directly to the parameter values.

- **Momentum Decay (`momentum_decay`)**:
  This value represents the amount of decay applied to the momentum buffer (first moment estimate)
  before it is updated. This decay helps to reduce the impact of older momentum contributions,
  leading to better long-term convergence. Default is `0.004`.

Algorithm
---------

The NAdam optimization algorithm updates each parameter according to the following formulas:

.. math::

    m_{t} &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{t}

    v_{t} &= \beta_2 v_{t-1} + (1 - \beta_2) \nabla_{t}^2

    \hat{m}_{t} &= \frac{m_{t}}{1 - \beta_1^t}

    \hat{v}_{t} &= \frac{v_{t}}{1 - \beta_2^t}

    \theta_{t+1} &= \theta_{t} - \frac{\text{lr} \cdot (\beta_1 \cdot \hat{m}_{t} + 
    (1 - \beta_1) \cdot \nabla_{t})}{\sqrt{\hat{v}_{t}} + \epsilon}

Where:

- :math:`\nabla_{t}` is the gradient of the loss with respect to the parameter at iteration :math:`t`.
- :math:`m_{t}` and :math:`v_{t}` are the exponentially decaying first and second moment estimates.
- :math:`\hat{m}_{t}` and :math:`\hat{v}_{t}` are bias-corrected estimates of the first and second moments.
- :math:`\theta_{t}` is the parameter at iteration :math:`t`.
- :math:`\text{lr}` is the learning rate.

Additionally, momentum decay is applied to the first moment estimate (`m_t`) as follows:

.. math::

    m_{t} \leftarrow m_{t} \cdot (1 - \text{momentum\_decay})

This ensures that older momentum contributions are reduced, leading to better 
long-term convergence.

Examples
--------

.. admonition:: **Using the NAdam Optimizer**
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

       # Initialize model and NAdam optimizer
       model = MyModel()
       optimizer = optim.NAdam(
          model.parameters(), 
          lr=0.001, 
          betas=(0.9, 0.999), 
          eps=1e-8, 
          weight_decay=0.01, 
          momentum_decay=0.004,
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
