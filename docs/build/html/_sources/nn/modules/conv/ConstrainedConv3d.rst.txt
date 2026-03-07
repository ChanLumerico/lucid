nn.ConstrainedConv3d
====================

.. autoclass:: lucid.nn.ConstrainedConv3d

The `ConstrainedConv3d` module applies constrained convolution on volumetric
inputs (or spatio-temporal tensors). It is useful when 3D kernels should satisfy
explicit priors such as normalization, zero-mean, or bounded energy.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ConstrainedConv3d(
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        constraint: Literal[
            "none", "nonneg", "sum_to_one", "zero_mean", "nonneg_sum1",
            "unit_l2", "max_l2", "fixed_center"
        ] = "none",
        enforce: Literal["forward", "post_step"] = "forward",
        eps: float = 1e-12,
        max_l2: float | None = None,
        center_value: float = -1.0,
        neighbor_sum: float = 1.0,
    )

Parameters
----------
- **in_channels** (*int*): Number of channels in the input volume.
- **out_channels** (*int*): Number of output channels.
- **kernel_size** (*int* or *tuple[int, ...]*): 3D kernel size.
- **stride** (*int* or *tuple[int, ...]*, optional): Stride. Default is `1`.
- **padding** (*_PaddingStr* or *int* or *tuple[int, ...]*, optional):
  Padding. Supports `"same"` and `"valid"`. Default is `0`.
- **dilation** (*int* or *tuple[int, ...]*, optional): Dilation. Default is `1`.
- **groups** (*int*, optional): Grouped convolution factor. Default is `1`.
- **bias** (*bool*, optional): If `True`, adds learnable bias. Default is `True`.
- **constraint** (*str*, optional): Kernel constraint mode.
- **enforce** (*str*, optional): `"forward"` or `"post_step"`.
- **eps** (*float*, optional): Stability constant.
- **max_l2** (*float | None*, optional): Required for `"max_l2"`.
- **center_value** (*float*, optional): Center coefficient for `"fixed_center"`.
- **neighbor_sum** (*float*, optional): Sum target of non-center coefficients.

Mathematical Formulation
------------------------
For input :math:`x` and kernel :math:`W`:

.. math::

    y = \operatorname{conv3d}(x, \tilde{W}) + b

where constrained kernel is

.. math::

    \tilde{W} = \mathcal{C}(W)

for `enforce="forward"`, and

.. math::

    W \leftarrow \mathcal{C}(W)

is executed explicitly with :meth:`project_` for `enforce="post_step"`.

For input shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, output shape is
:math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

.. math::

    D_{out} = \left\lfloor
    \frac{D_{in} + 2P_D - \Delta_D(K_D-1) - 1}{S_D} + 1
    \right\rfloor,

.. math::

    H_{out} = \left\lfloor
    \frac{H_{in} + 2P_H - \Delta_H(K_H-1) - 1}{S_H} + 1
    \right\rfloor,

.. math::

    W_{out} = \left\lfloor
    \frac{W_{in} + 2P_W - \Delta_W(K_W-1) - 1}{S_W} + 1
    \right\rfloor.

Constraint Modes
----------------
Each mode is applied to each :math:`(K_D, K_H, K_W)` kernel block:

- **none**

  .. math::

      \tilde{W} = W

- **nonneg**

  .. math::

      \tilde{W} = \max(W, 0)

- **sum_to_one**

  .. math::

      \tilde{W} = \frac{W}{\sum_{p,q,r} W_{p,q,r} + \varepsilon}

- **zero_mean**

  .. math::

      \tilde{W} = W - \frac{1}{K_D K_H K_W}\sum_{p,q,r} W_{p,q,r}

- **nonneg_sum1**

  .. math::

      \tilde{W} =
      \frac{\max(W, 0)}{\sum_{p,q,r} \max(W_{p,q,r},0) + \varepsilon}

- **unit_l2**

  .. math::

      \tilde{W} =
      \frac{W}{\sqrt{\sum_{p,q,r} W_{p,q,r}^2 + \varepsilon}}

- **max_l2**

  .. math::

      \tilde{W} = W \cdot
      \min\left(1, \frac{c}{\sqrt{\sum_{p,q,r} W_{p,q,r}^2 + \varepsilon}}\right)

- **fixed_center** (odd kernel sizes required)

  Let center index be :math:`(p_c, q_c, r_c)`:

  .. math::

      \tilde{W}_{p_c,q_c,r_c} = v_c,
      \quad
      \sum_{(p,q,r)\neq(p_c,q_c,r_c)} \tilde{W}_{p,q,r} = s_n

Examples
--------
**1) Constrained volumetric convolution**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(2, 4, 16, 32, 32)
    >>> conv = nn.ConstrainedConv3d(
    ...     4, 8, kernel_size=3, padding=1,
    ...     constraint="unit_l2",
    ... )
    >>> y = conv(x)
    >>> y.shape
    (2, 8, 16, 32, 32)

**2) Spatio-temporal residual modeling with zero-mean kernels**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(1, 3, 8, 64, 64)
    >>> conv = nn.ConstrainedConv3d(3, 6, kernel_size=3, padding=1, constraint="zero_mean")
    >>> y = conv(x)
    >>> y.shape
    (1, 6, 8, 64, 64)

**3) Hard projected max-L2 constrained training step**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> conv = nn.ConstrainedConv3d(
    ...     6, 6, kernel_size=3, padding=1,
    ...     constraint="max_l2", max_l2=0.8,
    ...     enforce="post_step",
    ... )
    >>> opt = optim.SGD(conv.parameters(), lr=1e-2)
    >>> x = lucid.random.randn(2, 6, 6, 24, 24)
    >>> loss = conv(x).mean()
    >>> loss.backward()
    >>> opt.step()
    >>> conv.project_()
    >>> opt.zero_grad()

**4) Fixed-center 3D kernels**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(1, 1, 10, 20, 20)
    >>> conv = nn.ConstrainedConv3d(
    ...     1, 4, kernel_size=5, padding=2,
    ...     constraint="fixed_center",
    ...     center_value=-1.0,
    ...     neighbor_sum=1.0,
    ... )
    >>> y = conv(x)
    >>> y.shape
    (1, 4, 10, 20, 20)

**5) Grouped constrained Conv3d**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(2, 8, 8, 16, 16)
    >>> conv = nn.ConstrainedConv3d(
    ...     8, 8, kernel_size=3, padding=1,
    ...     groups=4,
    ...     constraint="nonneg",
    ... )
    >>> y = conv(x)
    >>> y.shape
    (2, 8, 8, 16, 16)
