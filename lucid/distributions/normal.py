"""Univariate ``Normal`` and ``LogNormal``."""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    positive,
    real,
)
from lucid.distributions.distribution import Distribution, ExponentialFamily

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


from lucid.distributions._util import (
    as_tensor as _as_tensor,
    broadcast_pair as _broadcast_pair,
)


class Normal(ExponentialFamily):
    """Univariate Gaussian ``N(loc, scale²)``."""

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
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return self.scale * self.scale

    @property
    def stddev(self) -> Tensor:
        return self.scale

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = lucid.randn(*shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale * eps

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        var = self.variance
        log_scale = self.scale.log()
        return -((value - self.loc) ** 2) / (2.0 * var) - log_scale - _LOG_SQRT_2PI

    def cdf(self, value: Tensor) -> Tensor:
        return 0.5 * (
            1.0 + lucid.erf((value - self.loc) / (self.scale * math.sqrt(2.0)))
        )

    def icdf(self, value: Tensor) -> Tensor:
        return self.loc + self.scale * lucid.erfinv(2.0 * value - 1.0) * math.sqrt(2.0)

    def entropy(self) -> Tensor:
        return 0.5 + 0.5 * math.log(2.0 * math.pi) + self.scale.log()


class LogNormal(Distribution):
    """``X = exp(Y)`` where ``Y ∼ Normal(loc, scale²)``."""

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = positive
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self._base = Normal(loc, scale, validate_args=False)
        self.loc = self._base.loc
        self.scale = self._base.scale
        super().__init__(
            batch_shape=tuple(self.loc.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return (self.loc + 0.5 * self.scale * self.scale).exp()

    @property
    def mode(self) -> Tensor:
        return (self.loc - self.scale * self.scale).exp()

    @property
    def variance(self) -> Tensor:
        s2 = self.scale * self.scale
        return (s2.exp() - 1.0) * (2.0 * self.loc + s2).exp()

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self._base.rsample(sample_shape).exp()

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        log_v = value.log()
        return self._base.log_prob(log_v) - log_v

    def entropy(self) -> Tensor:
        return self._base.entropy() + self.loc
