nn.utils.get_total_norm
=======================

.. autofunction:: lucid.nn.utils.get_total_norm

Function Signature
------------------

.. code-block:: python

    def get_total_norm(parameters: Iterable[Tensor] | Tensor, norm_type: int = 2) -> Tensor

Parameters
----------

- **parameters** (*Iterable[Tensor] | Tensor*):
  Model parameters whose gradients will be measured.
  Parameters with `grad is None` are skipped.

- **norm_type** (*int*, optional):
  Order of the p-norm. Use a positive integer number for
  :math:`p` (e.g., `2` for L2). Default is `2`.

Return Value
------------

- **Tensor**:
  The global gradient p-norm computed across all provided parameters.
  Returned as a scalar tensor on the same device as the first parameter.

Mathematical Definition
-----------------------

Let the parameter set be :math:`\{\theta_i\}_{i=1}^N` with associated gradient tensors
:math:`\{g_i\}_{i=1}^N`, where each :math:`g_i` has the same shape as :math:`\theta_i`.
Define :math:`\operatorname{vec}(g_i)` as the flattened vector of :math:`g_i`.

**Global p-norm**:

.. math::

   \|g\|_p
   \,=\,
   \left(\sum_{i=1}^{N} \big\|\operatorname{vec}(g_i)\big\|_p^{\,p}\right)^{\!1/p},
   \quad p \in (0, \infty)

where for each parameter :math:`i`,

.. math::

   \big\|\operatorname{vec}(g_i)\big\|_p
   \,=\,
   \left(\sum_{j} \big| (g_i)_j \big|^{\,p} \right)^{\!1/p}.

.. note::

   The gradients are **not** modified by :func:`get_total_norm`;
   it only measures the global magnitude.

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

    n2 = nn.utils.get_total_norm(model.parameters(), norm_type=2)  # L2 norm
    print("L2:", n2)

Usage Tips
----------

.. tip::

   Use :func:`get_total_norm` when you want to inspect the gradient magnitude
   without applying any clipping. Combine it with
   :func:`lucid.nn.utils.clip_grad_norm` when clipping is needed.
