nn.ConstrainedConv2d
====================

.. autoclass:: lucid.nn.ConstrainedConv2d

The `ConstrainedConv2d` module applies a 2D convolution with an explicit kernel
constraint map :math:`\mathcal{C}`.

Compared to standard `Conv2d`, this module allows explicit structural priors on
filters, which is useful for image residual modeling, stable feature extraction,
and constrained physical operators.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ConstrainedConv2d(
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
- **in_channels** (*int*): Number of channels in the input feature map.
- **out_channels** (*int*): Number of channels produced by the convolution.
- **kernel_size** (*int* or *tuple[int, ...]*): 2D kernel size.
- **stride** (*int* or *tuple[int, ...]*, optional): Convolution stride. Default is `1`.
- **padding** (*_PaddingStr* or *int* or *tuple[int, ...]*, optional):
  Input padding. Supports `"same"` and `"valid"`. Default is `0`.
- **dilation** (*int* or *tuple[int, ...]*, optional): Dilation factor. Default is `1`.
- **groups** (*int*, optional): Grouped convolution factor. Default is `1`.
- **bias** (*bool*, optional): If `True`, adds learnable bias. Default is `True`.
- **constraint** (*str*, optional): Constraint mode for each spatial kernel slice.
- **enforce** (*str*, optional): `"forward"` or `"post_step"`.
- **eps** (*float*, optional): Stability constant in normalization denominators.
- **max_l2** (*float | None*, optional): Radius for `"max_l2"`.
- **center_value** (*float*, optional): Fixed center value for `"fixed_center"`.
- **neighbor_sum** (*float*, optional): Target sum of non-center coefficients.

Forward Calculation
-------------------
Let raw kernel be :math:`W` and constrained kernel be :math:`\tilde{W}`.

.. math::

    y = \operatorname{conv2d}(x, \tilde{W}) + b

where:

.. math::

    \tilde{W} =
    \begin{cases}
    \mathcal{C}(W), & \text{if enforce = "forward"} \\
    W, & \text{if enforce = "post_step"}
    \end{cases}

For hard projection mode:

.. math::

    W \leftarrow \mathcal{C}(W)

is applied by calling :meth:`project_` after optimizer step.

For input shape :math:`(N, C_{in}, H_{in}, W_{in})`, output shape is
:math:`(N, C_{out}, H_{out}, W_{out})` with

.. math::

    H_{out} = \left\lfloor
    \frac{H_{in} + 2P_H - D_H(K_H-1) - 1}{S_H} + 1
    \right\rfloor,

.. math::

    W_{out} = \left\lfloor
    \frac{W_{in} + 2P_W - D_W(K_W-1) - 1}{S_W} + 1
    \right\rfloor.

Constraint Families
-------------------
Each constraint is applied per kernel slice
:math:`W[o, i, :, :]`:

- **none**

  .. math::

      \tilde{W} = W

- **nonneg**

  .. math::

      \tilde{W} = \max(W, 0)

- **sum_to_one**

  .. math::

      \tilde{W} = \frac{W}{\sum_{u,v} W_{u,v} + \varepsilon}

- **zero_mean**

  .. math::

      \tilde{W} = W - \frac{1}{K_H K_W}\sum_{u,v} W_{u,v}

- **nonneg_sum1**

  .. math::

      \tilde{W} = \frac{\max(W, 0)}{\sum_{u,v} \max(W_{u,v}, 0) + \varepsilon}

- **unit_l2**

  .. math::

      \tilde{W} = \frac{W}{\sqrt{\sum_{u,v} W_{u,v}^2 + \varepsilon}}

- **max_l2**

  .. math::

      \tilde{W} = W \cdot
      \min\left(1, \frac{c}{\sqrt{\sum_{u,v} W_{u,v}^2 + \varepsilon}}\right)

  where :math:`c = \text{max\_l2}`.

- **fixed_center**

  For odd kernel sizes, center :math:`(u_c, v_c)` is fixed and neighbors are
  normalized:

  .. math::

      \tilde{W}_{u_c,v_c} = v_c,
      \quad
      \sum_{(u,v)\neq(u_c,v_c)} \tilde{W}_{u,v} = s_n

  where :math:`v_c=\text{center\_value}` and
  :math:`s_n=\text{neighbor\_sum}`.

Practical Notes
---------------
- Use `enforce="forward"` for simple end-to-end constrained training.
- Use `enforce="post_step"` + `project_()` when strict hard projection is required.
- `fixed_center` requires odd kernel sizes along both spatial dimensions.

Examples
--------
**1) Non-negative normalized 2D kernels**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(8, 3, 64, 64)
    >>> conv = nn.ConstrainedConv2d(
    ...     3, 16, kernel_size=5, padding=2,
    ...     constraint="nonneg_sum1",
    ... )
    >>> y = conv(x)
    >>> y.shape
    (8, 16, 64, 64)

**2) Zero-mean residual filter bank**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(2, 3, 128, 128)
    >>> conv = nn.ConstrainedConv2d(3, 12, kernel_size=3, padding=1, constraint="zero_mean")
    >>> y = conv(x)
    >>> y.shape
    (2, 12, 128, 128)

**3) Hard projection workflow**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> import lucid.optim as optim
    >>> conv = nn.ConstrainedConv2d(
    ...     16, 16, kernel_size=3, padding=1,
    ...     constraint="max_l2", max_l2=1.0,
    ...     enforce="post_step",
    ... )
    >>> opt = optim.SGD(conv.parameters(), lr=1e-2)
    >>> x = lucid.random.randn(4, 16, 32, 32)
    >>> y = conv(x).mean()
    >>> y.backward()
    >>> opt.step()
    >>> conv.project_()
    >>> opt.zero_grad()

**4) Fixed-center constrained filter (for residual/noise emphasis)**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(1, 1, 64, 64)
    >>> conv = nn.ConstrainedConv2d(
    ...     1, 8, kernel_size=5, padding=2,
    ...     constraint="fixed_center",
    ...     center_value=-1.0,
    ...     neighbor_sum=1.0,
    ... )
    >>> y = conv(x)
    >>> y.shape
    (1, 8, 64, 64)

**5) Grouped constrained convolution**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(2, 8, 32, 32)
    >>> conv = nn.ConstrainedConv2d(
    ...     8, 8, kernel_size=3, padding=1,
    ...     groups=4,
    ...     constraint="unit_l2",
    ... )
    >>> y = conv(x)
    >>> y.shape
    (2, 8, 32, 32)
