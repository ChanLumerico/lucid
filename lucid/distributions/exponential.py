"""``Exponential``, ``Laplace``, and ``Cauchy`` — icdf-sampled families."""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    positive,
    real,
)
from lucid.distributions.distribution import Distribution, ExponentialFamily


from lucid.distributions._util import as_tensor as _as_tensor, broadcast_pair as _broadcast_pair


class Exponential(ExponentialFamily):
    """``Exp(rate)`` with density ``rate · exp(-rate · x)`` on ``x ≥ 0``."""

    arg_constraints = {"rate": positive}
    support: Constraint | None = positive
    has_rsample = True

    def __init__(
        self,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.rate = _as_tensor(rate)
        super().__init__(
            batch_shape=tuple(self.rate.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return 1.0 / self.rate

    @property
    def mode(self) -> Tensor:
        return lucid.zeros_like(self.rate)

    @property
    def variance(self) -> Tensor:
        return 1.0 / (self.rate * self.rate)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # Inverse CDF sampling via U ~ Uniform(0, 1):  x = -log(1 - U) / rate.
        # Subtract from 1 to keep U away from 0 (which would give -inf).
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.rate.dtype, device=self.rate.device)
        return -(1.0 - u).log() / self.rate

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value

    def cdf(self, value: Tensor) -> Tensor:
        return 1.0 - (-self.rate * value).exp()

    def icdf(self, value: Tensor) -> Tensor:
        return -(1.0 - value).log() / self.rate

    def entropy(self) -> Tensor:
        return 1.0 - self.rate.log()


class Laplace(Distribution):
    """``Laplace(loc, scale)`` — symmetric double-exponential."""

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.loc, self.scale = _broadcast_pair(self.loc, self.scale)
        super().__init__(
            batch_shape=tuple(self.loc.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return 2.0 * self.scale * self.scale

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # icdf: loc - scale · sign(u-0.5) · log(1 - 2|u-0.5|).
        # Reparameterise as 2U-1 then split into sign+magnitude.
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device) - 0.5
        sign_u = u.sign()
        return self.loc - self.scale * sign_u * (1.0 - 2.0 * u.abs()).log()

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        return -((value - self.loc).abs()) / self.scale - (2.0 * self.scale).log()

    def cdf(self, value: Tensor) -> Tensor:
        z = (value - self.loc) / self.scale
        return 0.5 - 0.5 * z.sign() * (1.0 - (-z.abs()).exp())

    def entropy(self) -> Tensor:
        return 1.0 + (2.0 * self.scale).log()


class Cauchy(Distribution):
    """``Cauchy(loc, scale)`` — heavy-tailed; mean / variance undefined."""

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.loc, self.scale = _broadcast_pair(self.loc, self.scale)
        super().__init__(
            batch_shape=tuple(self.loc.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # icdf: loc + scale · tan(π·(U − 0.5)).
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale * (math.pi * (u - 0.5)).tan()

    def log_prob(self, value: Tensor) -> Tensor:
        z = (value - self.loc) / self.scale
        return (
            -math.log(math.pi)
            - self.scale.log()
            - (1.0 + z * z).log()
        )

    def cdf(self, value: Tensor) -> Tensor:
        z = (value - self.loc) / self.scale
        return 0.5 + z.arctan() / math.pi

    def entropy(self) -> Tensor:
        return math.log(4.0 * math.pi) + self.scale.log()
