"""Bijective ``Transform`` system for ``lucid.distributions``.

A :class:`Transform` is an invertible map ``T : X → Y`` with a known
log-absolute-determinant Jacobian, used to push a base distribution
through a measurable bijection (e.g. ``ExpTransform`` turns a Normal
into a LogNormal).

Composition + inversion are handled by :class:`ComposeTransform` and
:class:`Transform.__invert__` so users can build Lucid-side bijectors
without worrying about Jacobian bookkeeping.
"""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import as_tensor as _as_tensor


class Transform:
    """Abstract bijection.  Subclasses override ``_call``, ``_inverse``
    and ``log_abs_det_jacobian``."""

    bijective: bool = True
    sign: int = 1  # +1 if monotone increasing, −1 if decreasing.
    event_dim: int = 0

    def __init__(self) -> None:
        self._inv: Transform | None = None

    @property
    def inv(self) -> "Transform":
        """Lazy inverse — caches an :class:`_InverseTransform` view."""
        if self._inv is None:
            self._inv = _InverseTransform(self)
        return self._inv

    def __invert__(self) -> "Transform":
        return self.inv

    def __call__(self, x: Tensor) -> Tensor:
        return self._call(x)

    def _call(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}._call")

    def _inverse(self, y: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}._inverse")

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.log_abs_det_jacobian")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class _InverseTransform(Transform):
    """View of the inverse of an underlying transform."""

    def __init__(self, base: Transform) -> None:
        super().__init__()
        self._base = base
        self.event_dim = base.event_dim
        self.sign = base.sign
        self._inv = base

    @property
    def inv(self) -> Transform:  # type: ignore[override]
        return self._base

    def _call(self, x: Tensor) -> Tensor:
        return self._base._inverse(x)

    def _inverse(self, y: Tensor) -> Tensor:
        return self._base._call(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return -self._base.log_abs_det_jacobian(y, x)


class ExpTransform(Transform):
    """``y = exp(x)`` — maps ℝ → (0, ∞)."""

    def _call(self, x: Tensor) -> Tensor:
        return x.exp()

    def _inverse(self, y: Tensor) -> Tensor:
        return y.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x  # |dy/dx| = exp(x), so log|dy/dx| = x.


class SigmoidTransform(Transform):
    """``y = σ(x) = 1 / (1 + exp(−x))`` — maps ℝ → (0, 1)."""

    def _call(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def _inverse(self, y: Tensor) -> Tensor:
        return y.log() - (1.0 - y).log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # log|y · (1 − y)|.
        return y.log() + (1.0 - y).log()


class TanhTransform(Transform):
    """``y = tanh(x)`` — maps ℝ → (−1, 1)."""

    def _call(self, x: Tensor) -> Tensor:
        return x.tanh()

    def _inverse(self, y: Tensor) -> Tensor:
        return 0.5 * ((1.0 + y) / (1.0 - y)).log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # log(1 − tanh²(x)) = 2·(log 2 − x − softplus(−2x)) — the stable
        # form used by the reference framework.
        return 2.0 * (math.log(2.0) - x - (-2.0 * x).exp().log1p())


class AffineTransform(Transform):
    """``y = loc + scale · x``."""

    def __init__(self, loc: Tensor | float, scale: Tensor | float) -> None:
        super().__init__()
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.sign = 1  # Caller is responsible for ``scale > 0``.

    def _call(self, x: Tensor) -> Tensor:
        return self.loc + self.scale * x

    def _inverse(self, y: Tensor) -> Tensor:
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.scale.abs().log() + lucid.zeros(
            tuple(x.shape), dtype=x.dtype, device=x.device
        )


class PowerTransform(Transform):
    """``y = x^exponent`` (element-wise) for ``x > 0``."""

    def __init__(self, exponent: Tensor | float) -> None:
        super().__init__()
        self.exponent = _as_tensor(exponent)

    def _call(self, x: Tensor) -> Tensor:
        return x ** self.exponent

    def _inverse(self, y: Tensor) -> Tensor:
        return y ** (1.0 / self.exponent)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # |dy/dx| = exponent · x^(exponent − 1) ⇒ log|dy/dx| =
        #   log|exponent| + (exponent − 1)·log(x).
        return self.exponent.abs().log() + (self.exponent - 1.0) * x.log()


class SoftmaxTransform(Transform):
    """``y = softmax(x)`` along the last axis — pushes ℝ^K onto the
    open K-simplex.  Not a true bijection (loses 1-DOF to the
    constraint), but standard in normalising-flow stacks."""

    event_dim: int = 1

    def _call(self, x: Tensor) -> Tensor:
        from lucid.nn.functional.activations import softmax

        return softmax(x, dim=-1)

    def _inverse(self, y: Tensor) -> Tensor:
        # Standard convention: take the un-normalised log-probabilities.
        # Any constant shift gives the same softmax — we anchor at log y.
        return y.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # log |det J| = sum_k log y_k — pseudo-Jacobian for the
        # over-parameterised softmax, matching the reference framework.
        return y.log().sum(dim=-1)


class StickBreakingTransform(Transform):
    """Logistic stick-breaking from ℝ^(K-1) to the K-simplex.

    Maps ``x ∈ ℝ^(K-1)`` to ``y ∈ Δ^K`` by
    ``y_k = σ(x_k − log(K−k)) · ∏_{j<k} (1 − y_j)``,
    with the last component ``y_{K-1}`` taking the residual stick.
    """

    event_dim: int = 1

    def _call(self, x: Tensor) -> Tensor:
        # ``x`` shape (..., K-1).  We extend with a final 0 so the
        # cumulative-product step yields the residual stick automatically.
        K_minus_1: int = int(x.shape[-1])
        offsets: Tensor = lucid.arange(
            K_minus_1, 0, -1, dtype=x.dtype, device=x.device
        )
        # σ(x_k − log(K−k)) for k = 0..K-2.
        z: Tensor = (x - offsets.log()).sigmoid()
        # Stick-breaking: y_k = z_k · ∏_{j<k} (1 − z_j).
        # Compute via cumulative product of (1 − z) and pad with 1 in front.
        one_minus_z: Tensor = 1.0 - z
        # Manual prefix-product since lucid has no cumprod-from-1 op handy:
        # build [1, 1−z₀, (1−z₀)(1−z₁), …].
        prods: list[Tensor] = []
        running: Tensor = lucid.ones_like(z.narrow(-1, 0, 1)).squeeze(-1)
        prods.append(running)
        for k in range(K_minus_1 - 1):
            running = running * one_minus_z.narrow(-1, k, 1).squeeze(-1)
            prods.append(running)
        prefix: Tensor = lucid.stack(prods, dim=-1)  # shape (..., K-1)
        head: Tensor = z * prefix  # y_0 .. y_{K-2}
        # Last component is the residual: y_{K-1} = ∏_{j<K-1} (1 − z_j).
        last: Tensor = prefix.narrow(-1, K_minus_1 - 1, 1) * one_minus_z.narrow(
            -1, K_minus_1 - 1, 1
        )
        return lucid.cat([head, last], dim=-1)

    def _inverse(self, y: Tensor) -> Tensor:
        # Recover x_k from y via z_k = y_k / (1 − Σ_{j<k} y_j),
        # then x_k = logit(z_k) + log(K−k).
        K: int = int(y.shape[-1])
        K_minus_1: int = K - 1
        # Cumulative tail sums: stick remaining before drawing y_k.
        cum: Tensor = y.narrow(-1, 0, K_minus_1).cumsum(dim=-1)
        # remaining_before_k = 1 − Σ_{j<k} y_j  → shifted version.
        remaining: list[Tensor] = []
        ones: Tensor = lucid.ones_like(y.narrow(-1, 0, 1))
        remaining.append(ones)
        for k in range(1, K_minus_1):
            remaining.append(
                (1.0 - cum.narrow(-1, k - 1, 1))
            )
        rem: Tensor = lucid.cat(remaining, dim=-1).squeeze(-1) if K_minus_1 == 1 \
            else lucid.cat(remaining, dim=-1)
        # rem has shape (..., K-1) by construction above.
        z: Tensor = y.narrow(-1, 0, K_minus_1) / rem
        offsets: Tensor = lucid.arange(
            K_minus_1, 0, -1, dtype=y.dtype, device=y.device
        )
        return (z.log() - (1.0 - z).log()) + offsets.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # log|det J| = Σ_k log(y_k) + log(remaining stick before y_k) —
        # the standard simplex-to-ℝ^(K-1) Jacobian.
        K_minus_1: int = int(x.shape[-1])
        z: Tensor = y.narrow(-1, 0, K_minus_1)
        # remaining-before for each k.
        remaining: list[Tensor] = []
        cum: Tensor = z.cumsum(dim=-1)
        ones: Tensor = lucid.ones_like(z.narrow(-1, 0, 1))
        remaining.append(ones)
        for k in range(1, K_minus_1):
            remaining.append((1.0 - cum.narrow(-1, k - 1, 1)))
        rem: Tensor = lucid.cat(remaining, dim=-1)
        return (z.log() + rem.log() + (1.0 - z / rem).log()).sum(dim=-1)


class LowerCholeskyTransform(Transform):
    """ℝ^(D·(D+1)/2) → lower-triangular matrices with positive diagonal.

    The standard reparameterisation used to recover Cholesky factors
    from an unconstrained vector: off-diagonal entries pass through
    unchanged, diagonal entries go through ``softplus`` so they stay
    positive.
    """

    event_dim: int = 2

    def _call(self, x: Tensor) -> Tensor:
        from lucid.nn.functional.activations import softplus

        # ``x`` is already a ``(*batch, D, D)`` matrix; we mask out the
        # upper triangle (lucid has no native vec→tril helper, so the
        # caller pre-shapes the input).
        D: int = int(x.shape[-1])
        # Build a lower-triangular mask.
        mask: Tensor = _tril_mask(D, x.dtype, x.device)
        diag_mask: Tensor = _eye_mask(D, x.dtype, x.device)
        off_mask: Tensor = mask - diag_mask
        # Diagonal: softplus; off-diagonal: identity; rest: zero.
        diag: Tensor = softplus(x * diag_mask) * diag_mask
        off: Tensor = x * off_mask
        return diag + off

    def _inverse(self, y: Tensor) -> Tensor:
        # Inverse: invert softplus on the diagonal, off-diagonal as-is,
        # zero out the upper triangle.
        D: int = int(y.shape[-1])
        diag_mask: Tensor = _eye_mask(D, y.dtype, y.device)
        tril_mask: Tensor = _tril_mask(D, y.dtype, y.device)
        off_mask: Tensor = tril_mask - diag_mask
        # ``softplus^{-1}(z) = log(exp(z) − 1)`` — stable for z > 0.
        diag_in: Tensor = (y * diag_mask).exp().log1p() if False else (
            (y * diag_mask).exp() - 1.0
        ).log() * diag_mask
        off_in: Tensor = y * off_mask
        return diag_in + off_in

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # Jacobian: diag entries pass through softplus' = σ(x_diag);
        # off-diagonal entries have unit Jacobian.  log|det J| =
        # Σ log σ(x_diag) over the diagonal.
        D: int = int(x.shape[-1])
        diag_mask: Tensor = _eye_mask(D, x.dtype, x.device)
        # σ(x) = 1/(1+exp(-x)) ⇒ log σ(x) = −softplus(−x).
        from lucid.nn.functional.activations import softplus

        log_sig: Tensor = -softplus(-(x * diag_mask))
        return (log_sig * diag_mask).sum(dim=(-2, -1))


def _eye_mask(D: int, dtype, device) -> Tensor:
    """``D×D`` identity mask as a Lucid tensor."""
    return lucid.eye(D, dtype=dtype, device=device)


def _tril_mask(D: int, dtype, device) -> Tensor:
    """``D×D`` lower-triangular indicator (1 on/below diag, 0 above)."""
    ones: Tensor = lucid.ones(D, D, dtype=dtype, device=device)
    return lucid.tril(ones)


class ComposeTransform(Transform):
    """``T = T_n ∘ … ∘ T_2 ∘ T_1`` — applied left-to-right."""

    def __init__(self, parts: list[Transform]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("ComposeTransform: parts cannot be empty.")
        self.parts = list(parts)
        self.event_dim = max(p.event_dim for p in parts)

    def _call(self, x: Tensor) -> Tensor:
        for p in self.parts:
            x = p(x)
        return x

    def _inverse(self, y: Tensor) -> Tensor:
        for p in reversed(self.parts):
            y = p._inverse(y)
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        # Need intermediate values to evaluate each Jacobian.
        total = lucid.zeros(tuple(x.shape), dtype=x.dtype, device=x.device)
        cur = x
        for p in self.parts:
            nxt = p(cur)
            total = total + p.log_abs_det_jacobian(cur, nxt)
            cur = nxt
        return total


# ── TransformedDistribution ──────────────────────────────────────────────

from lucid.distributions.distribution import Distribution


class TransformedDistribution(Distribution):
    """Push ``base_distribution`` through a (possibly composite) bijector.

    ``rsample = transform(base_dist.rsample())`` and ``log_prob`` accounts
    for the Jacobian via the change-of-variable formula.
    """

    def __init__(
        self,
        base_distribution: Distribution,
        transforms: Transform | list[Transform],
        validate_args: bool | None = None,
    ) -> None:
        if isinstance(transforms, Transform):
            self.transforms: list[Transform] = [transforms]
        else:
            self.transforms = list(transforms)
        self.base_dist = base_distribution
        super().__init__(
            batch_shape=tuple(base_distribution.batch_shape),
            event_shape=tuple(base_distribution.event_shape),
            validate_args=validate_args,
        )

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        return self.base_dist.has_rsample

    def _push(self, x: Tensor) -> Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self._push(self.base_dist.rsample(sample_shape))

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self._push(self.base_dist.sample(sample_shape))

    def log_prob(self, value: Tensor) -> Tensor:
        # Walk transforms in reverse, building the inverse chain and the
        # Jacobian correction.
        log_det = lucid.zeros(
            tuple(value.shape), dtype=value.dtype, device=value.device
        )
        cur = value
        for t in reversed(self.transforms):
            prev = t._inverse(cur)
            log_det = log_det + t.log_abs_det_jacobian(prev, cur)
            cur = prev
        return self.base_dist.log_prob(cur) - log_det
