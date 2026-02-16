nn.utils.clip_grad_value
========================

.. autofunction:: lucid.nn.utils.clip_grad_value

Function Signature
------------------

.. code-block:: python

    def clip_grad_value(parameters: Iterable[Tensor] | Tensor, clip_value: _Scalar) -> None

Parameters
----------

- **parameters** (*Iterable[Tensor] | Tensor*):
  Model parameters whose gradients will be clipped by **value**.
  Parameters with `grad is None` are skipped.

- **clip_value** (*_Scalar*):
  The absolute threshold for clipping each gradient element.
  Every gradient entry :math:`g_{ij}` is clamped to lie within
  :math:`[-\text{clip\_value}, \text{clip\_value}]`.

Return Value
------------

- **None**:
  The operation modifies gradients **in-place** and does not return a value.

Mathematical Definition
-----------------------

Let the set of parameter gradients be :math:`\{g_i\}_{i=1}^N`, where each
:math:`g_i` is a tensor of the same shape as its parameter.

**Value-based clipping** replaces every element of each gradient tensor according to:

.. math::

   (g_i)_{jk} \leftarrow
   \operatorname{clip}\Big((g_i)_{jk},\,-v,\,v\Big)
   \,=\,
   \min(\max((g_i)_{jk}, -v), v),

where :math:`v = \text{clip\_value}`.

Thus, each scalar gradient entry is independently restricted to the closed interval
:math:`[-v, v]`, without changing its relative scaling with respect to other parameters.

Computation Details
-------------------

1. Iterate over all parameters with non-`None` gradients.
2. For each gradient tensor `p.grad`:
   - Compute `p.grad = lucid.clip(p.grad, -clip_value, clip_value)`.

3. Operation is performed **in-place**, so `p.grad` storage is updated directly.

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
    x = lucid.random.randn(32, 8)
    y = lucid.random.randint(0, 4, size=(32,))

    out = model(x)
    loss = lucid.nn.functional.cross_entropy(out, y)
    loss.backward()

    # Clip each gradient value individually to [-0.1, 0.1]
    nn.utils.clip_grad_value(model.parameters(), clip_value=0.1)

Usage Tips
----------

.. tip::

   Use :func:`clip_grad_value` when you want **per-element clipping**, rather than
   rescaling the entire gradient vector as in :func:`clip_grad_norm`.

   It ensures that no individual gradient component exceeds a given threshold.

.. warning::

   This method does **not** preserve the global gradient direction. It truncates
   extreme values directly, which can lead to more abrupt optimization steps.
   Use cautiously when training large networks.

.. admonition:: Comparison to Norm Clipping

   - :func:`clip_grad_value` limits *each gradient element*.
   - :func:`clip_grad_norm` limits the *overall gradient magnitude*.

   Both can be combined if desired:
   first apply value clipping, then apply norm clipping for strict bounds.
