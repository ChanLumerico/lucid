r"""
Normalization modules.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import ones, zeros
import lucid as _lucid
from lucid.nn.functional.normalization import (
    layer_norm,
    rms_norm,
    group_norm,
    batch_norm,
    instance_norm,
)


class LayerNorm(Module):
    r"""Layer normalization over the trailing dimensions of the input.

    Normalises each sample independently by computing mean and variance
    over the axes defined by ``normalized_shape``:

    .. math::

        y = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma + \beta

    where :math:`\mu` and :math:`\sigma^2` are computed over the last
    ``len(normalized_shape)`` dimensions of the input tensor.

    Unlike batch normalization, Layer Norm does not depend on the batch
    dimension, making it well-suited to variable-length sequences,
    transformer architectures, and settings where the batch size may be
    as small as 1.

    Parameters
    ----------
    normalized_shape : int or list[int] or tuple[int, ...]
        Shape of the trailing dimensions to normalize over.  If an
        integer ``d`` is given it is treated as ``(d,)``, normalizing
        only the last axis.  For a ``(N, T, C)`` input with
        ``normalized_shape=(C,)`` the mean and variance are computed
        independently for each ``(n, t)`` position.
    eps : float, optional
        Small constant added to the denominator for numerical stability.
        Default: ``1e-5``.
    elementwise_affine : bool, optional
        If ``True``, learns per-element scale :math:`\gamma` and
        (optionally) shift :math:`\beta` of shape ``normalized_shape``.
        If ``False``, no affine parameters are created and the output is
        purely normalized.  Default: ``True``.
    bias : bool, optional
        Only meaningful when ``elementwise_affine=True``.  If ``False``,
        the module learns only a scale :math:`\gamma` with no additive
        shift.  Default: ``True``.
    device : DeviceLike, optional
        Device on which to allocate the learnable parameters.
        Default: ``None`` (uses the default device).
    dtype : DTypeLike, optional
        Data type of the learnable parameters.  Default: ``None``
        (inherits from the input).

    Attributes
    ----------
    weight : Parameter or None
        Learnable per-element scale :math:`\gamma` of shape
        ``normalized_shape``.  ``None`` when
        ``elementwise_affine=False``.
    bias : Parameter or None
        Learnable per-element shift :math:`\beta` of shape
        ``normalized_shape``.  ``None`` when
        ``elementwise_affine=False`` or ``bias=False``.

    Shape
    -----
    - Input: :math:`(*, \text{normalized\_shape})` — any leading batch
      dimensions followed by the normalized trailing dimensions.
    - Output: same shape as the input.

    Notes
    -----
    - The mean and variance are computed with ``correction=0`` (biased
      estimator), consistent with the standard layer-norm convention.
    - Weights are initialised to 1 and biases to 0 so that the
      transformation is an identity at the start of training.
    - When using ``elementwise_affine=True, bias=False`` the module
      matches the "scale-only" layer norm used in some modern
      architectures.

    Examples
    --------
    Normalize the last dimension of a sequence model's hidden states:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> ln = nn.LayerNorm(512)
    >>> x = lucid.randn(32, 64, 512)   # (batch, seq_len, hidden_dim)
    >>> out = ln(x)
    >>> out.shape
    (32, 64, 512)

    Normalize over multiple trailing dimensions (e.g. height and width):

    >>> ln2d = nn.LayerNorm((28, 28))
    >>> x2d = lucid.randn(8, 1, 28, 28)
    >>> out2d = ln2d(x2d)
    >>> out2d.shape
    (8, 1, 28, 28)
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LayerNorm module. See the class docstring for parameter semantics."""
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: tuple[int, ...] = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight: Parameter | None = Parameter(
                ones(*self.normalized_shape, dtype=dtype, device=device)
            )
            if bias:
                self.bias: Parameter | None = Parameter(
                    zeros(*self.normalized_shape, dtype=dtype, device=device)
                )
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        r"""Apply normalisation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor whose shape is documented in the class docstring.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape as ``input``.
        """
        return layer_norm(
            x, list(self.normalized_shape), self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class RMSNorm(Module):
    r"""Root Mean Square Layer Normalization.

    Normalises the input by its root-mean-square value and applies a
    learnable per-element scale, but intentionally omits the mean
    subtraction step of standard layer norm:

    .. math::

        y = \frac{x}{\mathrm{RMS}(x)} \cdot \gamma, \qquad
        \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon}

    where :math:`d` is the number of elements in ``normalized_shape``.

    Skipping mean subtraction reduces computation while retaining the
    re-scaling invariance property that makes normalization useful.
    This formulation is widely used in large language model architectures
    such as LLaMA.

    Parameters
    ----------
    normalized_shape : int or list[int] or tuple[int, ...]
        Shape of the trailing dimensions to normalize over.  An integer
        ``d`` is equivalent to the single-element tuple ``(d,)``.
    eps : float, optional
        Small constant added inside the square root for numerical
        stability.  Default: ``1e-8``.
    device : DeviceLike, optional
        Device on which to allocate ``weight``.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type of ``weight``.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter
        Learnable per-element scale :math:`\gamma` of shape
        ``normalized_shape``, initialised to ones.  Unlike
        :class:`LayerNorm`, RMSNorm has no bias parameter by design.

    Shape
    -----
    - Input: :math:`(*, \text{normalized\_shape})`.
    - Output: same shape as the input.

    Notes
    -----
    - RMSNorm has no bias term; if a shift is needed, add a separate
      bias or use :class:`LayerNorm`.
    - The default ``eps`` (``1e-8``) is intentionally smaller than that
      of :class:`LayerNorm` (``1e-5``), since RMS values can be very
      small for zero-mean inputs.
    - The weight is initialised to all ones so the transformation starts
      as a pure normalisation.

    Examples
    --------
    Typical use in a transformer feed-forward block:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> norm = nn.RMSNorm(256)
    >>> x = lucid.randn(4, 32, 256)   # (batch, seq_len, dim)
    >>> out = norm(x)
    >>> out.shape
    (4, 32, 256)

    Normalize over two trailing dimensions:

    >>> norm2d = nn.RMSNorm((16, 16))
    >>> x2d = lucid.randn(2, 8, 16, 16)
    >>> out2d = norm2d(x2d)
    >>> out2d.shape
    (2, 8, 16, 16)
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-8,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the RMSNorm module. See the class docstring for parameter semantics."""
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(
            ones(*self.normalized_shape, dtype=dtype, device=device)
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        r"""Apply normalisation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor whose shape is documented in the class docstring.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape as ``input``.
        """
        return rms_norm(x, list(self.normalized_shape), self.weight, self.eps)

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"{self.normalized_shape}, eps={self.eps}"


class GroupNorm(Module):
    r"""Group normalization over the channel dimension.

    Divides the :math:`C` channels into ``num_groups`` contiguous groups
    of size :math:`C / \text{num\_groups}` and normalises each group
    independently over its spatial elements:

    .. math::

        y = \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \varepsilon}}
            \cdot \gamma + \beta

    where :math:`\mu_g` and :math:`\sigma_g^2` are the mean and variance
    computed over a single group (channels + spatial axes) for each
    sample in the batch.

    Group Norm sits between two extremes: ``num_groups=1`` recovers
    Layer Norm (normalize over all channels at once), while
    ``num_groups=num_channels`` recovers Instance Norm (each channel is
    its own group).  Unlike Batch Norm, Group Norm statistics are
    independent of the batch size, making it stable for small batches
    and well-suited to detection and segmentation models.

    Parameters
    ----------
    num_groups : int
        Number of groups to divide the channels into.
        ``num_channels`` must be divisible by ``num_groups``.
    num_channels : int
        Total number of channels :math:`C` expected in the input.
    eps : float, optional
        Small constant for numerical stability.  Default: ``1e-5``.
    affine : bool, optional
        If ``True``, learns per-channel scale :math:`\gamma` and shift
        :math:`\beta` of shape ``(num_channels,)``.  Default: ``True``.
    device : DeviceLike, optional
        Device for the learnable parameters.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type of the learnable parameters.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Learnable per-channel scale :math:`\gamma` of shape
        ``(num_channels,)``, initialised to ones.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Learnable per-channel shift :math:`\beta` of shape
        ``(num_channels,)``, initialised to zeros.
        ``None`` when ``affine=False``.

    Shape
    -----
    - Input: :math:`(N, C, *)` where :math:`*` denotes zero or more
      spatial dimensions and :math:`C = \text{num\_channels}`.
    - Output: same shape as the input.

    Notes
    -----
    - ``num_channels`` must be divisible by ``num_groups``; a
      ``ValueError`` is raised at the functional level if this is
      violated.
    - Despite sharing a name with batch-norm affine parameters, the
      ``weight`` and ``bias`` here have shape ``(num_channels,)`` rather
      than being element-wise over the full normalized region.

    Examples
    --------
    32-channel input split into 8 groups:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> gn = nn.GroupNorm(num_groups=8, num_channels=32)
    >>> x = lucid.randn(4, 32, 64, 64)
    >>> out = gn(x)
    >>> out.shape
    (4, 32, 64, 64)

    Layer-Norm equivalent (single group) on a 1-D sequence:

    >>> gn_layer = nn.GroupNorm(num_groups=1, num_channels=128)
    >>> x_seq = lucid.randn(16, 128, 200)   # (N, C, L)
    >>> out_seq = gn_layer(x_seq)
    >>> out_seq.shape
    (16, 128, 200)
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the GroupNorm module. See the class docstring for parameter semantics."""
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight: Parameter | None = Parameter(
                ones(num_channels, dtype=dtype, device=device)
            )
            self.bias: Parameter | None = Parameter(
                zeros(num_channels, dtype=dtype, device=device)
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        r"""Apply normalisation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor whose shape is documented in the class docstring.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape as ``input``.
        """
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"


class _BatchNormBase(Module):
    r"""Private base class for BatchNorm1d, BatchNorm2d, and BatchNorm3d.

    Implements the shared parameter layout, running-statistics bookkeeping,
    and checkpoint-migration logic that all batch normalization variants
    share.  Concrete subclasses differ only in the rank of input they
    accept — the actual normalization computation is delegated to
    :func:`lucid.nn.functional.batch_norm`.

    The normalization formula is:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}}
            \cdot \gamma + \beta

    **Running statistics behaviour:**

    * ``track_running_stats=True`` (default): during training,
      ``running_mean`` and ``running_var`` are updated as an exponential
      moving average and ``num_batches_tracked`` is incremented.
      During evaluation, the stored running stats are used as fixed
      constants for normalisation.
    * ``track_running_stats=False``: no buffers are maintained; both
      train and eval modes use the current batch statistics.
    * ``momentum=None``: cumulative moving average — the effective
      momentum becomes :math:`1 / \text{num\_batches\_tracked}`, giving
      equal weight to every observed batch.

    Parameters
    ----------
    num_features : int
        Number of channels :math:`C`.
    eps : float, optional
        Stability term added to the variance.  Default: ``1e-5``.
    momentum : float or None, optional
        EMA factor for updating ``running_mean`` / ``running_var``.
        ``None`` selects cumulative averaging.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, allocates learnable ``weight`` and ``bias``
        parameters.  Default: ``True``.
    track_running_stats : bool, optional
        If ``True``, maintains ``running_mean``, ``running_var``, and
        ``num_batches_tracked`` buffers.  Default: ``True``.
    device : DeviceLike, optional
        Device for parameters and buffers.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for parameters and buffers.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Per-channel scale :math:`\gamma`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Per-channel shift :math:`\beta`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    running_mean : Tensor or None
        Running per-channel mean estimate, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    running_var : Tensor or None
        Running per-channel variance estimate (Bessel-corrected for the
        update step), shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    num_batches_tracked : Tensor or None
        Scalar ``int64`` tensor counting batches seen during training.
        Drives cumulative averaging when ``momentum=None``.
        ``None`` when ``track_running_stats=False``.

    Notes
    -----
    - ``_version = 2`` marks checkpoints that include
      ``num_batches_tracked``; version-1 checkpoints are silently
      migrated by ``_load_from_state_dict`` to avoid spurious warnings.
    - Variance for the *running buffer update* uses the unbiased
      (Bessel-corrected) estimator, while the *normalisation itself*
      uses the biased estimator — both consistent with standard
      batch-norm conventions.
    """

    # Version 2 introduces `num_batches_tracked`.  Checkpoints saved with
    # version < 2 (or no metadata) are migrated by `_load_from_state_dict`.
    _version: int = 2  # type: ignore[misc]

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the _BatchNormBase module. See the class docstring for parameter semantics."""
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight: Parameter | None = Parameter(
                ones(num_features, dtype=dtype, device=device)
            )
            self.bias: Parameter | None = Parameter(
                zeros(num_features, dtype=dtype, device=device)
            )
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.register_buffer(
                "running_mean", zeros(num_features, dtype=dtype, device=device)
            )
            self.register_buffer(
                "running_var", ones(num_features, dtype=dtype, device=device)
            )
            # `num_batches_tracked` is int64 scalar regardless of the module's
            # float dtype.  When momentum is None this drives the cumulative
            # moving average via 1/num_batches_tracked.
            self.register_buffer(
                "num_batches_tracked",
                _lucid.zeros((), dtype=_lucid.int64, device=device),
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # Version-1 checkpoints predate `num_batches_tracked`.  Drop the
        # missing-key entry for it so users loading old weights aren't
        # spuriously warned.
        """Internal helper for the _BatchNormBase module."""
        version: int | None = (
            local_metadata.get("version") if local_metadata else None  # type: ignore[assignment]
        )
        if (version is None or version < 2) and self.track_running_stats:
            key: str = f"{prefix}num_batches_tracked"
            if key not in state_dict:
                # Pre-populate with zero so the default loader can copy it.
                state_dict[key] = _lucid.zeros((), dtype=_lucid.int64)
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        # Update running stats before the forward when training with
        # tracking enabled.  Detach to avoid linking the buffer into the
        # autograd graph; buffers are never differentiated through.
        r"""Apply normalisation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor whose shape is documented in the class docstring.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape as ``input``.
        """
        if self.training and self.track_running_stats:
            self._update_running_stats(x)

        # Pick which stats path the functional uses:
        #   - eval + tracking → precomputed running stats
        #   - everything else (training, or no tracking)  → batch stats
        use_running: bool = (not self.training) and self.track_running_stats
        running_mean: Tensor | None = (
            self._buffers.get("running_mean") if use_running else None
        )
        running_var: Tensor | None = (
            self._buffers.get("running_var") if use_running else None
        )

        return batch_norm(
            x,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            training=not use_running,
            momentum=self.momentum if self.momentum is not None else 0.0,
            eps=self.eps,
        )

    def _update_running_stats(self, x: Tensor) -> None:
        """Update ``running_mean`` / ``running_var`` from this batch.

        Matches the reference framework:
          running ← (1 − m) · running + m · batch
        Variance for the running buffer uses the unbiased (Bessel-corrected)
        estimator while the *normalisation* itself uses the biased one;
        both follow the reference framework's behaviour.
        """
        # Reduce over batch + spatial dims, keeping the channel dim.
        reduce_dims: list[int] = [d for d in range(x.ndim) if d != 1]
        n: int = 1
        for d in reduce_dims:
            n *= x.shape[d]
        with _lucid.no_grad():
            batch_mean: Tensor = x.mean(reduce_dims).detach()
            batch_var: Tensor = x.var(reduce_dims, correction=0).detach()

            # Increment the count first (matches reference framework order).
            self._buffers["num_batches_tracked"] = (
                self._buffers["num_batches_tracked"] + 1  # type: ignore[union-attr]
            ).detach()

            eff: float
            if self.momentum is None:
                # Cumulative moving average: equal weight on every batch.
                eff = 1.0 / float(self._buffers["num_batches_tracked"].item())  # type: ignore[union-attr]
            else:
                eff = float(self.momentum)

            # Unbiased correction n/(n-1) for the running variance, like the
            # reference framework — only meaningful when n > 1.
            unbiased_factor: float = n / (n - 1) if n > 1 else 1.0
            new_rm: Tensor = (1.0 - eff) * self._buffers[
                "running_mean"
            ] + eff * batch_mean
            new_rv: Tensor = (1.0 - eff) * self._buffers["running_var"] + (
                eff * unbiased_factor
            ) * batch_var
            self._buffers["running_mean"] = new_rm.detach()
            self._buffers["running_var"] = new_rv.detach()

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm1d(_BatchNormBase):
    r"""Batch normalization over a 2-D or 3-D input ``(N, C)`` or ``(N, C, L)``.

    Normalises each channel across the batch (and, for 3-D inputs, the
    length) dimension:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}}
            \cdot \gamma + \beta

    For a 3-D input :math:`(N, C, L)`, the statistics :math:`\mathrm{E}[x]`
    and :math:`\mathrm{Var}[x]` are computed over the :math:`(N, L)` axes for
    each channel :math:`c`.  For a 2-D input :math:`(N, C)` only the batch
    axis :math:`N` is reduced.

    During **training**, batch statistics are used and running statistics
    are updated via an exponential moving average:

    .. math::

        \hat{\mu} \leftarrow (1 - m)\,\hat{\mu} + m\,\mu_{\text{batch}}

    During **evaluation** (``model.eval()``), the stored
    :attr:`running_mean` and :attr:`running_var` are used instead, making
    inference independent of batch composition.

    Parameters
    ----------
    num_features : int
        Number of channels :math:`C`.
    eps : float, optional
        Small constant added to the variance for numerical stability.
        Default: ``1e-5``.
    momentum : float or None, optional
        Exponential moving average factor for running statistics.
        ``None`` uses a cumulative moving average.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, learns per-channel scale :math:`\gamma` and shift
        :math:`\beta`.  Default: ``True``.
    track_running_stats : bool, optional
        If ``True``, maintains :attr:`running_mean`, :attr:`running_var`,
        and :attr:`num_batches_tracked`.  Default: ``True``.
    device : DeviceLike, optional
        Device for parameters and buffers.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for parameters and buffers.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Learnable scale :math:`\gamma` of shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Learnable shift :math:`\beta` of shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    running_mean : Tensor or None
        Running per-channel mean, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    running_var : Tensor or None
        Running per-channel variance, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    num_batches_tracked : Tensor or None
        Scalar ``int64`` counting batches seen during training.
        ``None`` when ``track_running_stats=False``.

    Shape
    -----
    - Input: :math:`(N, C)` or :math:`(N, C, L)`
    - Output: same shape as the input.

    Examples
    --------
    2-D input (e.g. a linear layer's activations):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> bn = nn.BatchNorm1d(128)
    >>> x = lucid.randn(32, 128)
    >>> out = bn(x)
    >>> out.shape
    (32, 128)

    3-D input — temporal sequence with channels:

    >>> bn_seq = nn.BatchNorm1d(64)
    >>> x_seq = lucid.randn(16, 64, 200)   # (N, C, L)
    >>> out_seq = bn_seq(x_seq)
    >>> out_seq.shape
    (16, 64, 200)
    """


class BatchNorm2d(_BatchNormBase):
    r"""Batch normalization over a 4-D input ``(N, C, H, W)``.

    Normalises each channel across the batch and spatial dimensions:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}}
            \cdot \gamma + \beta

    where :math:`\mathrm{E}[x]` and :math:`\mathrm{Var}[x]` are computed
    over the :math:`(N, H, W)` axes for each channel :math:`c`.

    During **training**, batch statistics are used and running statistics
    are updated via an exponential moving average:

    .. math::

        \hat{\mu} \leftarrow (1 - m)\,\hat{\mu} + m\,\mu_{\text{batch}}

    During **evaluation** (``model.eval()``), the stored running statistics
    :attr:`running_mean` and :attr:`running_var` are used instead.

    Parameters
    ----------
    num_features : int
        Number of channels :math:`C`.
    eps : float, optional
        Small constant added to the variance for numerical stability.
        Default: ``1e-5``.
    momentum : float or None, optional
        Exponential moving average factor for running statistics. If
        ``None``, uses cumulative moving average. Default: ``0.1``.
    affine : bool, optional
        If ``True``, learns per-channel scale :math:`\gamma` and shift
        :math:`\beta`. Default: ``True``.
    track_running_stats : bool, optional
        If ``True``, maintains :attr:`running_mean`, :attr:`running_var`,
        and :attr:`num_batches_tracked`. Default: ``True``.
    device : DeviceLike, optional
        Device for parameters and buffers.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for parameters and buffers.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Learnable scale :math:`\gamma` of shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Learnable shift :math:`\beta` of shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    running_mean : Tensor or None
        Running estimate of the per-channel mean, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    running_var : Tensor or None
        Running estimate of the per-channel variance, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    num_batches_tracked : Tensor or None
        Scalar counting the number of batches seen during training.
        ``None`` when ``track_running_stats=False``.

    Shape
    -----
    - Input: :math:`(N, C, H, W)`
    - Output: :math:`(N, C, H, W)` — same shape.

    Notes
    -----
    - BatchNorm2d is the most commonly used normalization layer in
      convolutional neural networks.  It stabilises training by keeping
      activations in a well-scaled range after each convolutional block.
    - At small batch sizes (e.g. :math:`N < 8`) the batch statistics
      become noisy.  Consider :class:`GroupNorm` or
      :class:`InstanceNorm2d` in those settings.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> bn = nn.BatchNorm2d(64)
    >>> x = lucid.randn(8, 64, 32, 32)
    >>> out = bn(x)   # normalised per channel
    >>> out.shape
    (8, 64, 32, 32)

    >>> # Eval mode uses running statistics (no batch dependence)
    >>> bn.eval()
    >>> with lucid.no_grad():
    ...     out = bn(x)
    """


class BatchNorm3d(_BatchNormBase):
    r"""Batch normalization over a 5-D input ``(N, C, D, H, W)``.

    Extends batch normalization to volumetric data by normalising each
    channel across the batch and all three spatial dimensions:

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}}
            \cdot \gamma + \beta

    where :math:`\mathrm{E}[x]` and :math:`\mathrm{Var}[x]` are computed
    over the :math:`(N, D, H, W)` axes for each channel :math:`c`.

    The training/evaluation distinction is identical to
    :class:`BatchNorm2d` — running statistics are updated during training
    and used as fixed normalisation constants during evaluation.

    Parameters
    ----------
    num_features : int
        Number of channels :math:`C`.
    eps : float, optional
        Small constant added to the variance for numerical stability.
        Default: ``1e-5``.
    momentum : float or None, optional
        EMA factor for the running statistics.  ``None`` selects
        cumulative averaging.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, learns per-channel :math:`\gamma` and :math:`\beta`.
        Default: ``True``.
    track_running_stats : bool, optional
        If ``True``, maintains :attr:`running_mean`, :attr:`running_var`,
        and :attr:`num_batches_tracked`.  Default: ``True``.
    device : DeviceLike, optional
        Device for parameters and buffers.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for parameters and buffers.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Learnable scale :math:`\gamma` of shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Learnable shift :math:`\beta` of shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    running_mean : Tensor or None
        Running per-channel mean, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    running_var : Tensor or None
        Running per-channel variance, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    num_batches_tracked : Tensor or None
        Scalar ``int64`` counting training batches seen.
        ``None`` when ``track_running_stats=False``.

    Shape
    -----
    - Input: :math:`(N, C, D, H, W)`
    - Output: :math:`(N, C, D, H, W)` — same shape.

    Notes
    -----
    - Typical applications include 3-D convolutional networks for video
      understanding, medical image segmentation (CT/MRI), and any
      domain where the data has a depth axis in addition to height and
      width.
    - Because the :math:`(N, D, H, W)` reduction covers more elements
      than in :class:`BatchNorm2d`, the variance estimate is generally
      more stable at the same batch size.

    Examples
    --------
    Normalizing activations from a 3-D convolution:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> bn3d = nn.BatchNorm3d(32)
    >>> x = lucid.randn(4, 32, 16, 32, 32)   # (N, C, D, H, W)
    >>> out = bn3d(x)
    >>> out.shape
    (4, 32, 16, 32, 32)

    Disable affine parameters for a parameter-free normaliser:

    >>> bn_no_affine = nn.BatchNorm3d(32, affine=False)
    >>> out2 = bn_no_affine(x)
    >>> out2.shape
    (4, 32, 16, 32, 32)
    """


class _InstanceNormBase(Module):
    r"""Private base class for InstanceNorm1d, InstanceNorm2d, and InstanceNorm3d.

    Performs per-instance normalization: each ``(n, c)`` slice of the
    input is standardised against statistics computed over its own
    spatial axes only, making the result independent of both the batch
    composition and the other channels:

    .. math::

        y_{n,c} = \frac{x_{n,c} - \mu_{n,c}}
                       {\sqrt{\sigma_{n,c}^2 + \varepsilon}}
                  \cdot \gamma_c + \beta_c

    where :math:`\mu_{n,c}` and :math:`\sigma_{n,c}^2` are the mean and
    variance of sample :math:`n`, channel :math:`c` over its spatial
    dimensions.

    Unlike :class:`_BatchNormBase`, the defaults here are
    ``affine=False`` and ``track_running_stats=False``, matching the
    standard use-case in style transfer where the affine transform is
    replaced by learned style parameters from a separate path.

    Parameters
    ----------
    num_features : int
        Number of channels :math:`C`.
    eps : float, optional
        Stability constant.  Default: ``1e-5``.
    momentum : float, optional
        EMA factor for optional running statistics.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, allocates per-channel :math:`\gamma` and
        :math:`\beta`.  Default: ``False``.
    track_running_stats : bool, optional
        If ``True``, maintains ``running_mean`` and ``running_var``.
        Default: ``False``.
    device : DeviceLike, optional
        Device for parameters/buffers.  Default: ``None``.
    dtype : DTypeLike, optional
        Dtype for parameters/buffers.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Per-channel scale :math:`\gamma`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Per-channel shift :math:`\beta`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    running_mean : Tensor or None
        Optional running mean per channel, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    running_var : Tensor or None
        Optional running variance per channel, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.

    Notes
    -----
    - Subclasses set ``_expected_dim`` (3, 4, or 5) for input-rank
      validation in ``_check_input_dim``.
    - When ``track_running_stats=True`` in eval mode the running buffers
      are used, matching the behavior of :class:`_BatchNormBase`.
    """

    _expected_dim: int = 0  # subclass overrides: 3 / 4 / 5

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the _InstanceNormBase module. See the class docstring for parameter semantics."""
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats

        if affine:
            self.weight: Parameter | None = Parameter(
                ones(num_features, dtype=dtype, device=device)
            )
            self.bias: Parameter | None = Parameter(
                zeros(num_features, dtype=dtype, device=device)
            )
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.register_buffer(
                "running_mean", zeros(num_features, dtype=dtype, device=device)
            )
            self.register_buffer(
                "running_var", ones(num_features, dtype=dtype, device=device)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def _check_input_dim(self, x: Tensor) -> None:
        """Internal helper for the _InstanceNormBase module."""
        if self._expected_dim and x.ndim != self._expected_dim:
            raise ValueError(
                f"{type(self).__name__} expects a {self._expected_dim}-D input "
                f"(N, C{', *spatial' if self._expected_dim > 2 else ''}); "
                f"got ndim={x.ndim}"
            )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        r"""Apply normalisation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor whose shape is documented in the class docstring.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape as ``input``.
        """
        self._check_input_dim(x)
        # eval mode + track_running_stats=True ⇒ use running stats path.
        use_input_stats: bool = self.training or not self.track_running_stats
        rm: Tensor | None = (
            self._buffers.get("running_mean") if self.track_running_stats else None
        )
        rv: Tensor | None = (
            self._buffers.get("running_var") if self.track_running_stats else None
        )
        # Update running stats during training (tracks per-channel stats
        # averaged across the batch — matches the reference framework).
        if self.training and self.track_running_stats:
            self._update_running_stats(x)
        return instance_norm(
            x,
            running_mean=rm,
            running_var=rv,
            weight=self.weight,
            bias=self.bias,
            use_input_stats=use_input_stats,
            momentum=self.momentum,
            eps=self.eps,
        )

    def _update_running_stats(self, x: Tensor) -> None:
        """Update running stats from per-channel batch statistics."""
        # Reduce over batch + spatial → (C,).  Matches BatchNorm's reduction.
        reduce_dims: list[int] = [d for d in range(x.ndim) if d != 1]
        with _lucid.no_grad():
            batch_mean: Tensor = x.mean(reduce_dims).detach()
            batch_var: Tensor = x.var(reduce_dims, correction=0).detach()
            m: float = float(self.momentum)
            new_rm: Tensor = (1.0 - m) * self._buffers["running_mean"] + m * batch_mean
            new_rv: Tensor = (1.0 - m) * self._buffers["running_var"] + m * batch_var
            self._buffers["running_mean"] = new_rm.detach()
            self._buffers["running_var"] = new_rv.detach()

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class InstanceNorm1d(_InstanceNormBase):
    r"""Instance normalization for 3-D input ``(N, C, L)``.

    Normalises each ``(n, c)`` slice over the length dimension :math:`L`
    independently:

    .. math::

        y_{n,c} = \frac{x_{n,c} - \mu_{n,c}}
                       {\sqrt{\sigma_{n,c}^2 + \varepsilon}}
                  \cdot \gamma_c + \beta_c

    where :math:`\mu_{n,c}` and :math:`\sigma_{n,c}^2` are the mean and
    variance of the :math:`L` elements in sample :math:`n`, channel :math:`c`.

    This is useful for 1-D temporal or sequential data where the
    statistics should not be mixed across samples or channels.

    Parameters
    ----------
    num_features : int
        Number of channels :math:`C`.
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float, optional
        EMA factor used when ``track_running_stats=True``.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, learns per-channel scale and shift.  Default: ``False``.
    track_running_stats : bool, optional
        If ``True``, maintains running mean/variance buffers.
        Default: ``False``.
    device : DeviceLike, optional
        Device for parameters/buffers.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for parameters/buffers.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Per-channel scale :math:`\gamma`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Per-channel shift :math:`\beta`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    running_mean : Tensor or None
        Running mean per channel, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    running_var : Tensor or None
        Running variance per channel, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.

    Shape
    -----
    - Input: :math:`(N, C, L)`
    - Output: :math:`(N, C, L)` — same shape.

    Notes
    -----
    - With ``affine=True``, the learned ``weight`` and ``bias`` can be
      replaced at inference time by externally supplied style tensors,
      which is the core mechanism behind adaptive instance normalization
      (AdaIN) used in style transfer.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> norm = nn.InstanceNorm1d(32)
    >>> x = lucid.randn(8, 32, 100)   # (N, C, L)
    >>> out = norm(x)
    >>> out.shape
    (8, 32, 100)

    With learnable affine parameters:

    >>> norm_affine = nn.InstanceNorm1d(32, affine=True)
    >>> out2 = norm_affine(x)
    >>> out2.shape
    (8, 32, 100)
    """

    _expected_dim: int = 3


class InstanceNorm2d(_InstanceNormBase):
    r"""Instance normalization for 4-D input ``(N, C, H, W)``.

    Normalises each ``(n, c)`` slice over its spatial height and width
    independently:

    .. math::

        y_{n,c} = \frac{x_{n,c} - \mu_{n,c}}
                       {\sqrt{\sigma_{n,c}^2 + \varepsilon}}
                  \cdot \gamma_c + \beta_c

    where :math:`\mu_{n,c}` and :math:`\sigma_{n,c}^2` are computed over
    the :math:`H \times W` spatial positions of sample :math:`n`,
    channel :math:`c`.

    Unlike :class:`BatchNorm2d`, the statistics do not mix across the
    batch, making the output of each image entirely independent of the
    other images in the batch.  This property makes InstanceNorm2d
    well-suited to image style transfer, generative models, and any
    scenario where per-image normalization is desirable.

    Parameters
    ----------
    num_features : int
        Number of feature channels :math:`C`.
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float, optional
        EMA factor for optional running stats.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, learns per-channel :math:`\gamma` and :math:`\beta`.
        Default: ``False``.
    track_running_stats : bool, optional
        If ``True``, maintains running mean/variance buffers.
        Default: ``False``.
    device : DeviceLike, optional
        Device for parameters/buffers.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for parameters/buffers.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Per-channel scale :math:`\gamma`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Per-channel shift :math:`\beta`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    running_mean : Tensor or None
        Running mean per channel, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    running_var : Tensor or None
        Running variance per channel, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.

    Shape
    -----
    - Input: :math:`(N, C, H, W)`
    - Output: :math:`(N, C, H, W)` — same shape.

    Notes
    -----
    - Instance Norm was introduced for style transfer, where each image
      should be normalized independently so that stylistic statistics
      from other images do not bleed through.
    - At inference time, ``weight`` and ``bias`` can be replaced by
      style-conditioned tensors to implement Adaptive Instance
      Normalization (AdaIN).

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> norm = nn.InstanceNorm2d(64)
    >>> x = lucid.randn(4, 64, 128, 128)
    >>> out = norm(x)
    >>> out.shape
    (4, 64, 128, 128)

    With affine parameters for conditional style transfer:

    >>> norm_affine = nn.InstanceNorm2d(64, affine=True)
    >>> out2 = norm_affine(x)
    >>> out2.shape
    (4, 64, 128, 128)
    """

    _expected_dim: int = 4


class InstanceNorm3d(_InstanceNormBase):
    r"""Instance normalization for 5-D input ``(N, C, D, H, W)``.

    Normalises each ``(n, c)`` slice over the three spatial dimensions
    :math:`D \times H \times W` independently:

    .. math::

        y_{n,c} = \frac{x_{n,c} - \mu_{n,c}}
                       {\sqrt{\sigma_{n,c}^2 + \varepsilon}}
                  \cdot \gamma_c + \beta_c

    where :math:`\mu_{n,c}` and :math:`\sigma_{n,c}^2` are computed over
    the :math:`D \times H \times W` voxels of sample :math:`n`,
    channel :math:`c`.

    Applicable to volumetric data such as 3-D medical images or
    spatio-temporal video features, where per-sample independence is
    important and batch-level statistics are undesirable.

    Parameters
    ----------
    num_features : int
        Number of feature channels :math:`C`.
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float, optional
        EMA factor for optional running stats.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, learns per-channel :math:`\gamma` and :math:`\beta`.
        Default: ``False``.
    track_running_stats : bool, optional
        If ``True``, maintains running mean/variance buffers.
        Default: ``False``.
    device : DeviceLike, optional
        Device for parameters/buffers.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for parameters/buffers.  Default: ``None``.

    Attributes
    ----------
    weight : Parameter or None
        Per-channel scale :math:`\gamma`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    bias : Parameter or None
        Per-channel shift :math:`\beta`, shape ``(num_features,)``.
        ``None`` when ``affine=False``.
    running_mean : Tensor or None
        Running mean per channel, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.
    running_var : Tensor or None
        Running variance per channel, shape ``(num_features,)``.
        ``None`` when ``track_running_stats=False``.

    Shape
    -----
    - Input: :math:`(N, C, D, H, W)`
    - Output: :math:`(N, C, D, H, W)` — same shape.

    Examples
    --------
    3-D volumetric feature maps:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> norm = nn.InstanceNorm3d(16)
    >>> x = lucid.randn(2, 16, 8, 32, 32)   # (N, C, D, H, W)
    >>> out = norm(x)
    >>> out.shape
    (2, 16, 8, 32, 32)

    Enable running stats for a test-time normalization baseline:

    >>> norm_track = nn.InstanceNorm3d(16, track_running_stats=True)
    >>> out2 = norm_track(x)
    >>> out2.shape
    (2, 16, 8, 32, 32)
    """

    _expected_dim: int = 5


class LocalResponseNorm(Module):
    r"""Local Response Normalization (LRN) across adjacent channels.

    Normalises each activation by a sum of squared activations in a
    local neighbourhood of ``size`` channels centred on the same
    spatial position:

    .. math::

        y_c = \frac{x_c}{\left(k + \alpha \sum_{j=\max(0,\,c-\lfloor
            \text{size}/2\rfloor)}^{\min(C-1,\,c+\lfloor \text{size}/2\rfloor)}
            x_j^2\right)^{\beta}}

    where :math:`C` is the total number of channels, and the window
    :math:`[c - \lfloor \text{size}/2 \rfloor,\; c + \lfloor
    \text{size}/2 \rfloor]` is clipped at the channel boundaries.

    LRN was introduced in the AlexNet paper as a biologically-inspired
    lateral inhibition mechanism: strongly-activated neurons suppress
    their neighbors, encouraging competition across feature detectors at
    the same spatial location.  It has largely been superseded by Batch
    Normalization in modern architectures, but remains available for
    reproducing historical results.

    Parameters
    ----------
    size : int
        Number of adjacent channels included in the normalization
        window (the :math:`\text{size}` term in the formula above).
        Must be a positive integer.
    alpha : float, optional
        Scaling factor :math:`\alpha` for the squared sum.
        Default: ``1e-4``.
    beta : float, optional
        Exponent :math:`\beta` applied to the normalization term.
        Default: ``0.75``.
    k : float, optional
        Bias constant :math:`k` that prevents division by zero.
        Default: ``1.0``.

    Shape
    -----
    - Input: :math:`(N, C, *)` where :math:`*` denotes zero or more
      spatial dimensions.  The channel axis must be axis 1.
    - Output: same shape as the input.

    Notes
    -----
    - The implementation uses a sliding-window sum over the squared
      input, applied via ``unfold_dim`` on the padded channel axis.
      This avoids an explicit Python loop over channels and is
      fully differentiable through the Lucid engine.
    - Inputs with fewer than 2 dimensions are returned unchanged.
    - The original AlexNet used ``size=5``, ``alpha=1e-4``,
      ``beta=0.75``, ``k=2.0`` — adjusting ``k`` is important to
      control how strongly unsuppressed channels are penalized.

    Examples
    --------
    Reproduce the AlexNet LRN configuration:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
    >>> x = lucid.randn(2, 96, 27, 27)   # e.g. after first conv in AlexNet
    >>> out = lrn(x)
    >>> out.shape
    (2, 96, 27, 27)

    Minimal example with a 1-D channel tensor:

    >>> lrn_small = nn.LocalResponseNorm(size=3)
    >>> x2 = lucid.randn(4, 32, 64)   # (N, C, L)
    >>> out2 = lrn_small(x2)
    >>> out2.shape
    (4, 32, 64)
    """

    def __init__(
        self,
        size: int,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0,
    ) -> None:
        """Initialise the LocalResponseNorm module. See the class docstring for parameter semantics."""
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        r"""Apply normalisation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor whose shape is documented in the class docstring.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape as ``input``.
        """
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _unwrap, _wrap

        xi = _unwrap(x)
        if len(xi.shape) < 2:
            return x

        # x shape: (N, C, *spatial)  — channel axis is 1
        # LRN: y[c] = x[c] / (k + alpha * sum_{j=max(0,c-n/2)}^{min(C,c+n/2+1)} x[j]^2) ^ beta
        #
        # Implement via 1D average-pool across channels on the squared input.
        # Pad the channel axis (axis=1) by half on each side with zeros so the
        # sliding sum matches the reference convention.

        ndim = len(xi.shape)
        C = int(xi.shape[1])
        half = self.size // 2

        # x^2: same shape as x
        x_sq = _C_engine.mul(xi, xi)

        # Pad channel axis by (half, half) — produces (N, C+2*half, *)
        pad_pairs = [(0, 0)] + [(half, half)] + [(0, 0)] * (ndim - 2)
        x_sq_pad = _C_engine.pad(x_sq, pad_pairs, 0.0)  # zero-pad channels

        # For each channel c, sum x_sq_pad[:,c:c+size,...] over the window.
        # Use unfold_dim on axis=1 to get (N, C, *, size) windows, then sum.
        # unfold_dim(a, dim=1, size=self.size, step=1) → (N, C, *, size) iff
        # spatial dims follow channel. We need to handle arbitrary spatial dims.

        # Reshape to (N, C+2*half, -1) for 1D unfold, then sum the window.
        spatial_size = 1
        for d in range(2, ndim):
            spatial_size *= int(xi.shape[d])
        flat = _C_engine.reshape(
            x_sq_pad, [int(xi.shape[0]), C + 2 * half, spatial_size]
        )
        # Transpose to (N, spatial, C+2*half) for unfold along last dim
        flat_t = _C_engine.permute(flat, [0, 2, 1])  # (N, S, C+2h)
        unf = _C_engine.unfold_dim(flat_t, 2, self.size, 1)  # (N, S, C, size)
        window_sum = _C_engine.sum(unf, [3], False)  # (N, S, C)
        window_sum_t = _C_engine.permute(window_sum, [0, 2, 1])  # (N, C, S)
        # Restore spatial shape
        out_shape = list(xi.shape)
        window_sum_rs = _C_engine.reshape(window_sum_t, out_shape)  # (N, C, *)

        # scale = (k + alpha * window_sum) ^ beta
        k_t = _C_engine.full(out_shape, self.k, xi.dtype, xi.device)
        alpha_t = _C_engine.full(out_shape, self.alpha, xi.dtype, xi.device)
        scale = _C_engine.pow_scalar(
            _C_engine.add(k_t, _C_engine.mul(alpha_t, window_sum_rs)), self.beta
        )
        return _wrap(_C_engine.div(xi, scale))

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"size={self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k}"


# ── Lazy normalization variants ───────────────────────────────────────────────
#
# LazyBatchNorm{1,2,3}d and LazyInstanceNorm{1,2,3}d defer the
# ``num_features`` decision to the first forward call (which reads
# ``x.shape[1]``) or to ``_load_from_state_dict`` (which reads the leading
# axis of the saved ``weight`` / ``running_mean`` buffer).  All other
# behaviour — ``track_running_stats`` updates, ``momentum=None`` cumulative
# averaging, ``affine=False`` skipping the affine params — is inherited
# from ``_BatchNormBase``.


class _LazyBatchNormMixin(_BatchNormBase):
    r"""Shared lazy-init logic for LazyBatchNorm{1,2,3}d and LazyInstanceNorm*.

    Defers the allocation of all ``num_features``-sized parameters and
    buffers to the first ``forward()`` call (or to
    ``_load_from_state_dict`` when loading a checkpoint).  Until
    ``_initialize`` is called, ``weight``, ``bias``, ``running_mean``,
    ``running_var``, and ``num_batches_tracked`` are all ``None``
    placeholders.

    At initialization time ``num_features`` is read from ``x.shape[1]``
    and the full parameter/buffer set is created via ``_initialize``.
    All subsequent behavior is identical to the eager
    :class:`_BatchNormBase`.

    Parameters
    ----------
    eps : float, optional
        Stability term passed through to :class:`_BatchNormBase` after
        initialization.  Default: ``1e-5``.
    momentum : float or None, optional
        EMA factor.  ``None`` selects cumulative averaging.
        Default: ``0.1``.
    affine : bool, optional
        Whether to allocate learnable scale and shift parameters.
        Default: ``True``.
    track_running_stats : bool, optional
        Whether to maintain running mean, variance, and batch count.
        Default: ``True``.
    device : DeviceLike, optional
        Target device for lazily allocated tensors.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for lazily allocated tensors.  Default: ``None``.

    Notes
    -----
    - ``_load_from_state_dict`` probes the saved ``weight`` or
      ``running_mean`` key to infer ``num_features`` before delegating
      to the parent loader, enabling transparent checkpoint restore
      without a prior forward pass.
    - ``_lazy_label`` is a string tag overridden by each concrete
      subclass; it is used for display and debugging purposes.
    """

    _lazy_label: str = "_LazyBatchNormBase"

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        # Skip _BatchNormBase.__init__ — we don't have num_features yet.
        """Initialise the _LazyBatchNormMixin module. See the class docstring for parameter semantics."""
        Module.__init__(self)
        self.num_features: int | None = None  # type: ignore[assignment]
        self.eps: float = eps
        self.momentum: float | None = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        # Placeholder slots — actual buffers / params installed lazily.
        if affine:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def _initialize(self, num_features: int) -> None:
        """Internal helper for the _LazyBatchNormMixin module."""
        self.num_features = num_features
        if self.affine:
            self.weight = Parameter(
                ones(num_features, dtype=self._dtype, device=self._device)
            )
            self.bias = Parameter(
                zeros(num_features, dtype=self._dtype, device=self._device)
            )
        if self.track_running_stats:
            self._buffers["running_mean"] = zeros(
                num_features, dtype=self._dtype, device=self._device
            )
            self._buffers["running_var"] = ones(
                num_features, dtype=self._dtype, device=self._device
            )
            self._buffers["num_batches_tracked"] = _lucid.zeros(
                (), dtype=_lucid.int64, device=self._device
            )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Internal helper for the _LazyBatchNormMixin module."""
        if self.num_features is None:
            # Probe known persistent keys for the feature dim.
            probe_keys: tuple[str, ...] = ("weight", "running_mean", "bias")
            inferred: int | None = None
            for k in probe_keys:
                t: Tensor | None = state_dict.get(f"{prefix}{k}")
                if t is not None and len(t.shape) >= 1:
                    inferred = int(t.shape[0])
                    break
            if inferred is not None:
                # Pick a tensor for dtype/device fallback; avoid boolean
                # truthiness on a Tensor (ambiguous for multi-element).
                first_seen: Tensor | None = None
                w_t: Tensor | None = state_dict.get(f"{prefix}weight")
                if w_t is not None:
                    first_seen = w_t
                else:
                    rm_t: Tensor | None = state_dict.get(f"{prefix}running_mean")
                    if rm_t is not None:
                        first_seen = rm_t
                if first_seen is not None:
                    self._dtype = self._dtype or first_seen.dtype
                    self._device = self._device or first_seen.device
                self._initialize(inferred)
        # Fall through to parent (handles version<2 num_batches_tracked migration).
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        r"""Apply normalisation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor whose shape is documented in the class docstring.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape as ``input``.
        """
        if self.num_features is None:
            self._initialize(int(x.shape[1]))
        return _BatchNormBase.forward(self, x)

    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )


class LazyBatchNorm1d(_LazyBatchNormMixin):
    r"""BatchNorm1d with lazy ``num_features`` inference.

    Identical to :class:`BatchNorm1d` except that ``num_features`` need
    not be specified at construction time.  On the first call to
    ``forward(x)``, ``num_features`` is automatically set to
    ``x.shape[1]`` and all parameters and buffers are allocated.

    This is convenient when the number of channels is not known at model
    definition time, for example when building networks programmatically
    or when constructing sub-modules before the input shape is fixed.

    Parameters
    ----------
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float or None, optional
        EMA factor for running statistics.  ``None`` uses cumulative
        averaging.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, lazily allocates learnable scale and shift.
        Default: ``True``.
    track_running_stats : bool, optional
        If ``True``, lazily allocates running mean/variance and
        batch counter.  Default: ``True``.
    device : DeviceLike, optional
        Device for lazily allocated tensors.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for lazily allocated tensors.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(N, C)` or :math:`(N, C, L)` — same as
      :class:`BatchNorm1d`.  ``C`` is inferred on the first forward pass.
    - Output: same shape as the input.

    Examples
    --------
    Construct without specifying the channel count:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> bn = nn.LazyBatchNorm1d()
    >>> x = lucid.randn(16, 64, 100)   # (N, C, L)
    >>> out = bn(x)    # num_features=64 is inferred here
    >>> out.shape
    (16, 64, 100)

    Reload from a checkpoint saved by a :class:`BatchNorm1d`:

    >>> bn2 = nn.LazyBatchNorm1d()
    >>> # bn2.load_state_dict(state)  # num_features inferred from saved tensors
    """

    _lazy_label: str = "LazyBatchNorm1d"


class LazyBatchNorm2d(_LazyBatchNormMixin):
    r"""BatchNorm2d with lazy ``num_features`` inference.

    Identical to :class:`BatchNorm2d` except that ``num_features`` is
    inferred from ``x.shape[1]`` on the first forward pass.

    Parameters
    ----------
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float or None, optional
        EMA factor for running statistics.  ``None`` uses cumulative
        averaging.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, lazily allocates learnable scale and shift.
        Default: ``True``.
    track_running_stats : bool, optional
        If ``True``, lazily allocates running statistics buffers.
        Default: ``True``.
    device : DeviceLike, optional
        Device for lazily allocated tensors.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for lazily allocated tensors.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(N, C, H, W)` — same as :class:`BatchNorm2d`.
      ``C`` is inferred on the first forward pass.
    - Output: same shape as the input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> bn = nn.LazyBatchNorm2d()
    >>> x = lucid.randn(8, 128, 32, 32)
    >>> out = bn(x)    # num_features=128 inferred on first call
    >>> out.shape
    (8, 128, 32, 32)

    Suitable for dynamic architectures where channel count depends on
    a configuration value not available at module construction:

    >>> def make_norm(affine: bool = True) -> nn.LazyBatchNorm2d:
    ...     return nn.LazyBatchNorm2d(affine=affine)
    """

    _lazy_label: str = "LazyBatchNorm2d"


class LazyBatchNorm3d(_LazyBatchNormMixin):
    r"""BatchNorm3d with lazy ``num_features`` inference.

    Identical to :class:`BatchNorm3d` except that ``num_features`` is
    inferred from ``x.shape[1]`` on the first forward pass.

    Parameters
    ----------
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float or None, optional
        EMA factor for running statistics.  ``None`` uses cumulative
        averaging.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, lazily allocates learnable scale and shift.
        Default: ``True``.
    track_running_stats : bool, optional
        If ``True``, lazily allocates running statistics buffers.
        Default: ``True``.
    device : DeviceLike, optional
        Device for lazily allocated tensors.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for lazily allocated tensors.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(N, C, D, H, W)` — same as :class:`BatchNorm3d`.
      ``C`` is inferred on the first forward pass.
    - Output: same shape as the input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> bn = nn.LazyBatchNorm3d()
    >>> x = lucid.randn(2, 32, 8, 16, 16)   # (N, C, D, H, W)
    >>> out = bn(x)    # num_features=32 inferred on first call
    >>> out.shape
    (2, 32, 8, 16, 16)

    Useful for 3-D architectures (video, volumetric) where the feature
    count is determined by a preceding layer of variable depth:

    >>> bn_no_track = nn.LazyBatchNorm3d(track_running_stats=False)
    >>> out2 = bn_no_track(x)
    >>> out2.shape
    (2, 32, 8, 16, 16)
    """

    _lazy_label: str = "LazyBatchNorm3d"


class _LazyInstanceNormMixin(_InstanceNormBase):
    r"""Lazy ``num_features`` inference for InstanceNorm{1,2,3}d.

    Mirrors :class:`_LazyBatchNormMixin` but defers to
    :class:`_InstanceNormBase` so that the forward path performs
    per-instance normalisation rather than batch normalization.

    All parameters and buffers are ``None`` placeholders until the first
    ``forward(x)`` call, at which point ``num_features = x.shape[1]``
    is fixed and the full parameter set is allocated via ``_initialize``.
    A checkpoint restore via ``_load_from_state_dict`` can also trigger
    initialization before any forward pass.

    Defaults follow the eager InstanceNorm: ``affine=False``,
    ``track_running_stats=False``.

    Parameters
    ----------
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float, optional
        EMA factor for optional running statistics.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, lazily allocates per-channel scale and shift.
        Default: ``False``.
    track_running_stats : bool, optional
        If ``True``, lazily allocates running mean/variance buffers.
        Default: ``False``.
    device : DeviceLike, optional
        Device for lazily allocated tensors.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for lazily allocated tensors.  Default: ``None``.

    Notes
    -----
    - ``_load_from_state_dict`` probes ``weight`` or ``running_mean``
      in the incoming state dict to infer ``num_features``, then calls
      ``_initialize`` before delegating to the base loader.
    - ``_expected_dim`` is overridden by each concrete subclass (3, 4,
      or 5) for input-rank validation.
    """

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the _LazyInstanceNormMixin module. See the class docstring for parameter semantics."""
        Module.__init__(self)
        self.num_features: int | None = None  # type: ignore[assignment]
        self.eps: float = eps
        self.momentum: float = momentum
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
        self._device: DeviceLike = device
        self._dtype: DTypeLike = dtype
        if affine:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        else:
            self.weight = None
            self.bias = None
        self.register_buffer("running_mean", None)
        self.register_buffer("running_var", None)

    def _initialize(self, num_features: int) -> None:
        """Internal helper for the _LazyInstanceNormMixin module."""
        self.num_features = num_features
        if self.affine:
            self.weight = Parameter(
                ones(num_features, dtype=self._dtype, device=self._device)
            )
            self.bias = Parameter(
                zeros(num_features, dtype=self._dtype, device=self._device)
            )
        if self.track_running_stats:
            self._buffers["running_mean"] = zeros(
                num_features, dtype=self._dtype, device=self._device
            )
            self._buffers["running_var"] = ones(
                num_features, dtype=self._dtype, device=self._device
            )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Internal helper for the _LazyInstanceNormMixin module."""
        if self.num_features is None:
            for k in ("weight", "running_mean", "bias"):
                t: Tensor | None = state_dict.get(f"{prefix}{k}")
                if t is not None and len(t.shape) >= 1:
                    if self._dtype is None:
                        self._dtype = t.dtype
                    if self._device is None:
                        self._device = t.device
                    self._initialize(int(t.shape[0]))
                    break
        from lucid.nn._state_dict import _default_load_from_state_dict

        _default_load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        r"""Apply normalisation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor whose shape is documented in the class docstring.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape as ``input``.
        """
        if self.num_features is None:
            self._initialize(int(x.shape[1]))
        return _InstanceNormBase.forward(self, x)


class LazyInstanceNorm1d(_LazyInstanceNormMixin):
    r"""InstanceNorm1d with lazy ``num_features`` inference.

    Identical to :class:`InstanceNorm1d` except that ``num_features``
    is inferred from ``x.shape[1]`` on the first forward pass.

    Parameters
    ----------
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float, optional
        EMA factor for optional running statistics.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, lazily allocates per-channel scale and shift.
        Default: ``False``.
    track_running_stats : bool, optional
        If ``True``, lazily allocates running mean/variance buffers.
        Default: ``False``.
    device : DeviceLike, optional
        Device for lazily allocated tensors.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for lazily allocated tensors.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(N, C, L)` — same as :class:`InstanceNorm1d`.
      ``C`` is inferred on the first forward pass.
    - Output: same shape as the input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> norm = nn.LazyInstanceNorm1d()
    >>> x = lucid.randn(4, 48, 100)
    >>> out = norm(x)   # num_features=48 inferred here
    >>> out.shape
    (4, 48, 100)

    With affine parameters inferred lazily:

    >>> norm_affine = nn.LazyInstanceNorm1d(affine=True)
    >>> out2 = norm_affine(x)
    >>> out2.shape
    (4, 48, 100)
    """

    _expected_dim: int = 3


class LazyInstanceNorm2d(_LazyInstanceNormMixin):
    r"""InstanceNorm2d with lazy ``num_features`` inference.

    Identical to :class:`InstanceNorm2d` except that ``num_features``
    is inferred from ``x.shape[1]`` on the first forward pass.

    Parameters
    ----------
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float, optional
        EMA factor for optional running statistics.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, lazily allocates per-channel scale and shift.
        Default: ``False``.
    track_running_stats : bool, optional
        If ``True``, lazily allocates running mean/variance buffers.
        Default: ``False``.
    device : DeviceLike, optional
        Device for lazily allocated tensors.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for lazily allocated tensors.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(N, C, H, W)` — same as :class:`InstanceNorm2d`.
      ``C`` is inferred on the first forward pass.
    - Output: same shape as the input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> norm = nn.LazyInstanceNorm2d()
    >>> x = lucid.randn(2, 64, 128, 128)
    >>> out = norm(x)   # num_features=64 inferred here
    >>> out.shape
    (2, 64, 128, 128)

    Commonly used in generator networks where the channel dimension is
    determined at runtime by a preceding convolution:

    >>> norm_track = nn.LazyInstanceNorm2d(track_running_stats=True)
    >>> out2 = norm_track(x)
    >>> out2.shape
    (2, 64, 128, 128)
    """

    _expected_dim: int = 4


class LazyInstanceNorm3d(_LazyInstanceNormMixin):
    r"""InstanceNorm3d with lazy ``num_features`` inference.

    Identical to :class:`InstanceNorm3d` except that ``num_features``
    is inferred from ``x.shape[1]`` on the first forward pass.

    Parameters
    ----------
    eps : float, optional
        Numerical stability constant.  Default: ``1e-5``.
    momentum : float, optional
        EMA factor for optional running statistics.  Default: ``0.1``.
    affine : bool, optional
        If ``True``, lazily allocates per-channel scale and shift.
        Default: ``False``.
    track_running_stats : bool, optional
        If ``True``, lazily allocates running mean/variance buffers.
        Default: ``False``.
    device : DeviceLike, optional
        Device for lazily allocated tensors.  Default: ``None``.
    dtype : DTypeLike, optional
        Data type for lazily allocated tensors.  Default: ``None``.

    Shape
    -----
    - Input: :math:`(N, C, D, H, W)` — same as :class:`InstanceNorm3d`.
      ``C`` is inferred on the first forward pass.
    - Output: same shape as the input.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> norm = nn.LazyInstanceNorm3d()
    >>> x = lucid.randn(1, 16, 8, 32, 32)   # (N, C, D, H, W)
    >>> out = norm(x)   # num_features=16 inferred here
    >>> out.shape
    (1, 16, 8, 32, 32)

    Useful for 3-D segmentation networks built with unknown channel
    counts at construction time:

    >>> norm_affine = nn.LazyInstanceNorm3d(affine=True)
    >>> out2 = norm_affine(x)
    >>> out2.shape
    (1, 16, 8, 32, 32)
    """

    _expected_dim: int = 5
