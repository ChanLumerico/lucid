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
