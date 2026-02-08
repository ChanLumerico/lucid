optim.Adagrad
=============

.. autoclass:: lucid.optim.Adagrad

The `Adagrad` class implements the Adagrad optimization algorithm.
Adagrad adapts the learning rate for each parameter individually, scaling it
inversely with the square root of the sum of squared historical gradients.
This makes Adagrad suitable for sparse data or features, as it dynamically
scales the learning rate for each parameter.

Class Signature
---------------

.. code-block:: python

    class Adagrad(optim.Optimizer):
        def __init__(
            self,
            params: Iterable[nn.Parameter],
            lr: float = 1e-2,
            eps: float = 1e-10,
            weight_decay: float = 0.0,
            initial_accumulator_value: float = 0.0,
        ) -> None

Parameters
----------

- **Learning Rate (`lr`)**:
  Controls the step size during parameter updates. A higher learning rate
  can speed up convergence but may lead to instability.

- **Epsilon (`eps`)**:
  A small constant added to the denominator to prevent division by zero.
  This improves numerical stability during training.

- **Weight Decay (`weight_decay`)**:
  This coefficient controls the L2 penalty applied to the model parameters.
  Unlike traditional L2 regularization, weight decay in Adagrad is decoupled from the gradient calculation
  and applied directly to the parameter values.

- **Initial Accumulator Value (`initial_accumulator_value`)**:
  The starting value for the sum of squared gradients. This allows users to
  control the initial learning rate scaling for each parameter.

Algorithm
---------

The Adagrad optimization algorithm updates each parameter according to the following formulas:

.. math::

    G_{t} &= G_{t-1} + \nabla_{t}^2 
    
    \theta_{t+1} &= \theta_{t} - \frac{\text{lr} \cdot \nabla_{t}}{\sqrt{G_{t}} + \epsilon}

Where:

- :math:`\nabla_{t}` is the gradient of the loss with respect to the parameter at iteration :math:`t`.
- :math:`G_{t}` is the accumulated sum of squared gradients.
- :math:`\theta_{t}` is the parameter at iteration :math:`t`.
- :math:`\text{lr}` is the learning rate.

The accumulated squared gradients :math:`G_{t}` increase over time, leading to smaller updates
as training progresses. This property allows Adagrad to converge faster in some cases, but it
may suffer from diminishing learning rates for dense datasets.

Examples
--------

.. admonition:: **Using the Adagrad Optimizer**
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

       # Initialize model and Adagrad optimizer
       model = MyModel()
       optimizer = optim.Adagrad(model.parameters(), lr=0.01, eps=1e-10, weight_decay=0.01, initial_accumulator_value=0.1)

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
