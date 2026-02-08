optim.AdamW
===========

.. autoclass:: lucid.optim.AdamW

The `AdamW` class implements the AdamW optimization algorithm.
AdamW (Adam with Weight Decay) decouples the weight decay term from the gradient-based updates,
making the weight decay act directly on the parameters rather than on the gradient. 
This approach leads to better generalization and more stable convergence, 
particularly in large-scale training scenarios.

Class Signature
---------------

.. code-block:: python

    class AdamW(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
            amsgrad: bool = False,
        ) -> None

Parameters
----------

- **Learning Rate (`lr`)**:
  Controls the step size during parameter updates. 
  A higher learning rate can speed up convergence but may lead to instability.

- **Betas (`betas`)**:
  A tuple of two coefficients controlling the exponential decay of moment estimates.
  The first value (`beta1`) controls the decay rate for the first moment (mean of gradients),
  while the second value (`beta2`) controls the decay rate for the second moment (variance of gradients).

- **Epsilon (`eps`)**:
  A small constant added to the denominator to prevent division by zero.
  This improves numerical stability during training.

- **Weight Decay (`weight_decay`)**:
  This coefficient controls the L2 penalty applied to the model parameters.
  Unlike Adam, AdamW decouples the weight decay from the gradient update. 
  Instead, it is applied directly to the parameter values after the gradient step.

- **AMSGrad (`amsgrad`)**:
  If True, uses the AMSGrad variant of Adam, which ensures that the denominator
  does not decrease over time. This helps address some issues with non-convergence 
  in certain settings.

Algorithm
---------

The AdamW optimization algorithm updates each parameter according to the following formulas:

.. math::

    m_{t} &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{t}

    v_{t} &= \beta_2 v_{t-1} + (1 - \beta_2) \nabla_{t}^2

    \hat{m}_{t} &= \frac{m_{t}}{1 - \beta_1^t}

    \hat{v}_{t} &= \frac{v_{t}}{1 - \beta_2^t}
    
    \theta_{t+1} &= \theta_{t} - \frac{\text{lr} \cdot \hat{m}_{t}}
    {\sqrt{\hat{v}_{t}} + \epsilon} - \lambda \cdot \theta_{t}

Where:

- :math:`\nabla_{t}` is the gradient of the loss with respect to the parameter at iteration :math:`t`.
- :math:`m_{t}` and :math:`v_{t}` are the exponentially decaying first and second moment estimates.
- :math:`\hat{m}_{t}` and :math:`\hat{v}_{t}` are bias-corrected estimates of the first and second moments.
- :math:`\theta_{t}` is the parameter at iteration :math:`t`.
- :math:`\text{lr}` is the learning rate.
- :math:`\lambda` is the weight decay coefficient.

Examples
--------

.. admonition:: **Using the AdamW Optimizer**
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

       # Initialize model and AdamW optimizer
       model = MyModel()
       optimizer = optim.AdamW(
           model.parameters(), 
           lr=0.001, 
           betas=(0.9, 0.999), 
           eps=1e-8, 
           weight_decay=0.01, 
           amsgrad=True,
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

