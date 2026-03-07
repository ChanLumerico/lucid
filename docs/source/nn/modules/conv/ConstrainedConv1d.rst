nn.ConstrainedConv1d
====================

.. autoclass:: lucid.nn.ConstrainedConv1d

The `ConstrainedConv1d` module extends `Conv1d` by enforcing a structural
constraint on each convolution kernel.

It is useful when you want to inject priors such as:

- non-negative filters,
- sum-normalized filters,
- zero-mean residual filters,
- norm-bounded filters,
- fixed-center residual-style filters.

Class Signature
---------------
.. code-block:: python

    class lucid.nn.ConstrainedConv1d(
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
- **in_channels** (*int*): Number of channels in the input signal.
- **out_channels** (*int*): Number of channels produced by the convolution.
- **kernel_size** (*int* or *tuple[int, ...]*): Kernel size.
- **stride** (*int* or *tuple[int, ...]*, optional): Convolution stride. Default is `1`.
- **padding** (*_PaddingStr* or *int* or *tuple[int, ...]*, optional):
  Input padding. Supports `"same"` and `"valid"`. Default is `0`.
- **dilation** (*int* or *tuple[int, ...]*, optional): Kernel dilation. Default is `1`.
- **groups** (*int*, optional): Grouped convolution factor. Default is `1`.
- **bias** (*bool*, optional): If `True`, adds learnable bias. Default is `True`.
- **constraint** (*str*, optional): Constraint mode for the weight. Default is `"none"`.
- **enforce** (*str*, optional): Constraint timing. `"forward"` or `"post_step"`.
- **eps** (*float*, optional): Numerical stability constant used in normalization.
- **max_l2** (*float | None*, optional): Upper bound for `"max_l2"` mode.
- **center_value** (*float*, optional): Center coefficient for `"fixed_center"` mode.
- **neighbor_sum** (*float*, optional): Target sum of non-center coefficients for
  `"fixed_center"` mode.

Attributes
----------
- **weight** (*Tensor*): Learnable weight of shape
  :math:`(C_{out}, C_{in}/\text{groups}, K)`.
- **bias** (*Tensor* or *None*): Learnable bias of shape :math:`(C_{out})`.
- **project_** (*method*): Hard-projection utility for post-step enforcement.

Mathematical Formulation
------------------------
Let raw learnable kernel be :math:`W` and constrained kernel be :math:`\tilde{W}`.

The 1D convolution is:

.. math::

    y = \operatorname{conv1d}(x, \tilde{W}) + b

When `enforce="forward"`:

.. math::

    \tilde{W} = \mathcal{C}(W)

When `enforce="post_step"`:

.. math::

    \tilde{W} = W

for forward, and after optimizer update you call:

.. math::

    W \leftarrow \mathcal{C}(W)

via :meth:`project_`.

Output length is:

.. math::

    L_{out} = \left\lfloor
    \frac{L_{in} + 2P - D(K-1) - 1}{S} + 1
    \right\rfloor

where :math:`P` is padding, :math:`D` dilation, :math:`K` kernel size, and
:math:`S` stride.

Supported Constraints
---------------------
For each output-channel and input-channel-group kernel slice:

- **none**

  .. math::

      \tilde{W} = W

- **nonneg**

  .. math::

      \tilde{W} = \max(W, 0)

- **sum_to_one**

  .. math::

      \tilde{W} = \frac{W}{\sum_i W_i + \varepsilon}

- **zero_mean**

  .. math::

      \tilde{W} = W - \frac{1}{K}\sum_i W_i

- **nonneg_sum1**

  .. math::

      \tilde{W} = \frac{\max(W, 0)}{\sum_i \max(W_i, 0) + \varepsilon}

- **unit_l2**

  .. math::

      \tilde{W} = \frac{W}{\sqrt{\sum_i W_i^2 + \varepsilon}}

- **max_l2**

  .. math::

      \tilde{W} = W \cdot \min\left(1,
      \frac{c}{\sqrt{\sum_i W_i^2 + \varepsilon}}
      \right), \quad c=\text{max\_l2}

- **fixed_center**

  Let center index be :math:`i_c = \lfloor K/2 \rfloor`.

  .. math::

      \tilde{W}_{i_c} = v_c

  .. math::

      \sum_{i \ne i_c} \tilde{W}_i = s_n

  where :math:`v_c=\text{center\_value}` and :math:`s_n=\text{neighbor\_sum}`.

Training Patterns
-----------------
**Differentiable enforcement (default):**

.. code-block:: python

    import lucid.nn as nn

    conv = nn.ConstrainedConv1d(
        16, 32, kernel_size=5,
        constraint="nonneg_sum1",
        enforce="forward",
    )

**Hard projection after step:**

.. code-block:: python

    import lucid.nn as nn
    import lucid.optim as optim

    conv = nn.ConstrainedConv1d(
        16, 32, kernel_size=3,
        constraint="max_l2",
        max_l2=1.5,
        enforce="post_step",
    )
    opt = optim.SGD(conv.parameters(), lr=1e-2)

    # loss.backward()
    # opt.step()
    conv.project_()  # enforce hard constraint
    opt.zero_grad()

Examples
--------
**1) Non-negative normalized smoothing kernel**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(4, 8, 128)
    >>> m = nn.ConstrainedConv1d(
    ...     8, 16, kernel_size=7, padding=3,
    ...     constraint="nonneg_sum1",
    ... )
    >>> y = m(x)
    >>> y.shape
    (4, 16, 128)

**2) Residual-style high-pass behavior with zero-mean kernels**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(2, 4, 64)
    >>> m = nn.ConstrainedConv1d(4, 4, kernel_size=5, padding=2, constraint="zero_mean")
    >>> y = m(x)
    >>> y.shape
    (2, 4, 64)

**3) Fixed center coefficient for constrained residual filters**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(1, 3, 32)
    >>> m = nn.ConstrainedConv1d(
    ...     3, 8, kernel_size=5, padding=2,
    ...     constraint="fixed_center",
    ...     center_value=-1.0,
    ...     neighbor_sum=1.0,
    ... )
    >>> y = m(x)
    >>> y.shape
    (1, 8, 32)

**4) Grouped constrained convolution**

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn as nn
    >>> x = lucid.random.randn(2, 8, 50)
    >>> m = nn.ConstrainedConv1d(
    ...     8, 8, kernel_size=3, padding=1,
    ...     groups=4,
    ...     constraint="unit_l2",
    ... )
    >>> y = m(x)
    >>> y.shape
    (2, 8, 50)
