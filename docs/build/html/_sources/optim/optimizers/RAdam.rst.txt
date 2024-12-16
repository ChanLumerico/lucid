optim.RAdam
===========

.. autoclass:: lucid.optim.RAdam

The `RAdam` class implements the Rectified Adam (RAdam) optimization algorithm. 
RAdam addresses the slow convergence problem in Adam during early training steps 
by rectifying the variance of the adaptive learning rate. By doing so, RAdam combines 
the fast convergence of Adam with the stability and generalization properties of SGD.

Class Signature
---------------

.. code-block:: python

    class RAdam(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1e-3,
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
  Unlike traditional L2 regularization, weight decay in RAdam is decoupled from the gradient calculation 
  and applied directly to the parameter values.

Algorithm
---------

The RAdam optimization algorithm updates each parameter according to the following formulas:

.. math::

    m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{t} 

    v_t = \beta_2 v_{t-1} + (1 - \beta_2) \nabla_{t}^2 

    \hat{m}_t = \frac{m_t}{1 - \beta_1^t} 

    \hat{v}_t = \frac{v_t}{1 - \beta_2^t} 

    \rho_{\infty} = \frac{2}{1 - \beta_2} - 1 

    \rho_t = \rho_{\infty} - 2 \cdot t \cdot \frac{\beta_2^t}{1 - \beta_2^t} 

    r_t = \sqrt{\frac{(\rho_t - 4) \cdot (\rho_t - 2) \cdot 
    \rho_{\infty}}{(\rho_{\infty} - 4) \cdot (\rho_{\infty} - 2)}} 

    \theta_{t+1} = \begin{cases} 
      \theta_t - \text{lr} \cdot \hat{m}_t, & \text{if} \ \rho_t \leq 4 \\
      \theta_t - \frac{\text{lr} \cdot r_t \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}, & \text{if} \ \rho_t > 4 
    \end{cases}

Where:

- :math:`\nabla_t` is the gradient of the loss with respect to the parameter at iteration :math:`t`.
- :math:`m_t` and :math:`v_t` are the exponentially decaying first and second moment estimates.
- :math:`\hat{m}_t` and :math:`\hat{v}_t` are bias-corrected estimates of the first and second moments.
- :math:`\rho_t` represents the variance rectification term.
- :math:`\theta_t` is the parameter being updated at iteration :math:`t`.
- :math:`\text{lr}` is the learning rate.

The variance rectification factor :math:`r_t` ensures that the step size is adjusted dynamically 
for small batch sizes or early steps, improving the convergence properties.

Examples
--------

.. admonition:: **Using the RAdam Optimizer**
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

       # Initialize model and RAdam optimizer
       model = MyModel()
       optimizer = optim.RAdam(
          model.parameters(), 
          lr=0.001, 
          betas=(0.9, 0.999), 
          eps=1e-8, 
          weight_decay=0.01,
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
