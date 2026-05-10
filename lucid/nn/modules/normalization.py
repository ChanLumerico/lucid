"""
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
    """Layer normalization.

    Parameters
    ----------
    normalized_shape : int | tuple[int, ...]
        Trailing dimensions to normalise over.
    eps : float
        Numerical stability term.
    elementwise_affine : bool
        If True, learn per-element gain (and bias when ``bias=True``).
    bias : bool
        Only honoured when ``elementwise_affine=True``.  When False, the
        layer applies a learnable scale but no shift, matching the
        reference framework's 1.12+ behaviour.
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
        return layer_norm(
            x, list(self.normalized_shape), self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class RMSNorm(Module):
    """RMS normalization."""

    def __init__(
        self,
        normalized_shape: int | list[int] | tuple[int, ...],
        eps: float = 1e-8,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(
            ones(*self.normalized_shape, dtype=dtype, device=device)
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        return rms_norm(x, list(self.normalized_shape), self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class GroupNorm(Module):
    """Group normalization."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
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
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"


class _BatchNormBase(Module):
    """Common implementation for BatchNorm1d/2d/3d.

    Running statistics behaviour matches the reference framework:

    * ``track_running_stats=True`` (default): in training mode, ``running_mean``
      / ``running_var`` are updated via the momentum formula and
      ``num_batches_tracked`` increments by 1.  In eval mode, the precomputed
      running stats normalise the input.
    * ``track_running_stats=False``: no running buffers; both train and eval
      use batch statistics.
    * ``momentum=None``: cumulative moving average — the effective momentum
      becomes ``1 / num_batches_tracked``, so all batches contribute equally.
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
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm1d(_BatchNormBase):
    """Batch normalization for 2D or 3D input (N, C) or (N, C, L)."""


class BatchNorm2d(_BatchNormBase):
    """Batch normalization for 4D input (N, C, H, W)."""


class BatchNorm3d(_BatchNormBase):
    """Batch normalization for 5D input (N, C, D, H, W)."""


class _InstanceNormBase(Module):
    """Common implementation for InstanceNorm1d/2d/3d.

    Per-instance normalisation: every ``(n, c)`` slice is standardised
    against its own statistics computed over the spatial axes only.
    Defaults follow the reference framework — ``affine=False`` and
    ``track_running_stats=False`` — so the layer is parameter-free out
    of the box.
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
        if self._expected_dim and x.ndim != self._expected_dim:
            raise ValueError(
                f"{type(self).__name__} expects a {self._expected_dim}-D input "
                f"(N, C{', *spatial' if self._expected_dim > 2 else ''}); "
                f"got ndim={x.ndim}"
            )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
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
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class InstanceNorm1d(_InstanceNormBase):
    """Instance normalization for 3-D input ``(N, C, L)``."""

    _expected_dim: int = 3


class InstanceNorm2d(_InstanceNormBase):
    """Instance normalization for 4-D input ``(N, C, H, W)``."""

    _expected_dim: int = 4


class InstanceNorm3d(_InstanceNormBase):
    """Instance normalization for 5-D input ``(N, C, D, H, W)``."""

    _expected_dim: int = 5


class LocalResponseNorm(Module):
    """Local Response Normalization (LRN) across channels.

    For each element x_i, normalizes by the sum of squares of neighboring
    elements: x_i / (k + alpha * sum(x_j^2))^beta, where the sum is over
    *size* neighboring channels.

    Parameters match ``the reference LocalResponseNorm API``.
    """

    def __init__(
        self,
        size: int,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0,
    ) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
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
    """Shared lazy-init logic for LazyBatchNorm{1,2,3}d & LazyInstanceNorm*."""

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
        if self.num_features is None:
            self._initialize(int(x.shape[1]))
        return _BatchNormBase.forward(self, x)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )


class LazyBatchNorm1d(_LazyBatchNormMixin):
    """BatchNorm1d with lazy ``num_features`` inference."""

    _lazy_label: str = "LazyBatchNorm1d"


class LazyBatchNorm2d(_LazyBatchNormMixin):
    """BatchNorm2d with lazy ``num_features`` inference."""

    _lazy_label: str = "LazyBatchNorm2d"


class LazyBatchNorm3d(_LazyBatchNormMixin):
    """BatchNorm3d with lazy ``num_features`` inference."""

    _lazy_label: str = "LazyBatchNorm3d"


class _LazyInstanceNormMixin(_InstanceNormBase):
    """Lazy ``num_features`` inference for InstanceNorm{1,2,3}d.

    Mirrors the BatchNorm lazy mixin but defers to ``_InstanceNormBase``
    so the forward path actually performs per-instance normalisation.
    Defaults follow the eager InstanceNorm: ``affine=False``,
    ``track_running_stats=False``.
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
        if self.num_features is None:
            self._initialize(int(x.shape[1]))
        return _InstanceNormBase.forward(self, x)


class LazyInstanceNorm1d(_LazyInstanceNormMixin):
    """InstanceNorm1d with lazy ``num_features`` inference."""

    _expected_dim: int = 3


class LazyInstanceNorm2d(_LazyInstanceNormMixin):
    """InstanceNorm2d with lazy ``num_features`` inference."""

    _expected_dim: int = 4


class LazyInstanceNorm3d(_LazyInstanceNormMixin):
    """InstanceNorm3d with lazy ``num_features`` inference."""

    _expected_dim: int = 5
