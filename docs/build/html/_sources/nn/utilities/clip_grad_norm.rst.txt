nn.util.clip_grad_norm
======================

.. autofunction:: lucid.nn.util.clip_grad_norm

Function Signature
------------------

.. code-block:: python

    def clip_grad_norm(
        parameters: Iterable[Tensor] | Tensor,
        max_norm: _Scalar,
        norm_type: int = 2,
        eps: float = 1e-7,
    ) -> Tensor

Parameters
----------

- **parameters** (*Iterable[Tensor] | Tensor*):
  Model parameters whose gradients will be clipped. 
  Parameters with `grad is None` are skipped.

- **max_norm** (*_Scalar*):
  Maximum allowed norm for the global gradient vector. If the current global p-norm exceeds
  this value, all gradients are rescaled by the same factor to ensure the global norm equals 
  `max_norm`.

- **norm_type** (*int*, optional):
  Order of the p-norm to compute. 
  Use a positive integer for :math:`p` (e.g., `2` for L2 norm). Default is `2`.

- **eps** (*float*, optional):
  Small constant added to the denominator to prevent division by zero during normalization.
  Default is `1e-7`.

Return Value
------------

- **Tensor**:
  The total gradient norm **before** clipping. 
  Returned as a scalar tensor on the same device as the first parameter.

Mathematical Definition
-----------------------

Let :math:`\{\theta_i\}_{i=1}^N` be model parameters with gradients
:math:`\{g_i\}_{i=1}^N`. Define :math:`\operatorname{vec}(g_i)` as the flattened gradient
vector of parameter :math:`i`.

**Global p-norm**:

.. math::

   \|g\|_p
   \,=\,
   \left(\sum_{i=1}^{N} \big\|\operatorname{vec}(g_i)\big\|_p^{\,p}\right)^{\!1/p},
   \quad p \in (0, \infty)

**Clipping coefficient**:

If :math:`\|g\|_p > \text{max\_norm}`, then compute a scaling factor:

.. math::

   c
   \,=\,
   \frac{\text{max\_norm}}{\|g\|_p + \varepsilon},

and rescale all gradients:

.. math::

   g_i \leftarrow c \cdot g_i.

Otherwise, gradients remain unchanged.

.. note::

   Clipping is performed **in-place** on each parameter’s `grad` field.
   This means the gradients are directly modified without reallocating memory.


Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 4)
        def forward(self, x):
            x = lucid.nn.functional.relu(self.fc1(x))
            return self.fc2(x)

    model = Tiny()
    x = lucid.random.randn(32, 8)  # (N, D)
    y = lucid.random.randint(0, 4, size=(32,))

    out = model(x)
    loss = lucid.nn.functional.cross_entropy(out, y)
    loss.backward()

    # Clip gradient norms to have a maximum L2 norm of 1.0
    total_norm = nn.util.clip_grad_norm(model.parameters(), max_norm=1.0)
    print("Pre-clipping norm:", total_norm)


Usage Tips
----------

.. tip::

   Use :func:`clip_grad_norm` to stabilize training and prevent *exploding gradients*.
   This is especially useful in RNNs or deep networks where backpropagated gradients
   can grow exponentially.

.. warning::

   Always call :func:`clip_grad_norm` **after** `loss.backward()` and **before**
   `optimizer.step()`. Clipping before backpropagation has no effect on computed gradients.

.. admonition:: Common Practice

   Gradient clipping is often combined with monitoring the norm via
   :func:`lucid.nn.util.grad_norm` to decide when clipping is necessary.
