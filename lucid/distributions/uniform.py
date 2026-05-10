"""Continuous ``Uniform(low, high)``."""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    real,
)
from lucid.distributions.distribution import Distribution


from lucid.distributions._util import (
    as_tensor as _as_tensor,
    broadcast_pair as _broadcast_pair,
)


class Uniform(Distribution):
    """Uniform distribution on the interval ``[low, high)``."""

    arg_constraints = {"low": real, "high": real}
    has_rsample = True

    def __init__(
        self,
        low: Tensor | float,
        high: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.low = _as_tensor(low)
        self.high = _as_tensor(high)
        self.low, self.high = _broadcast_pair(self.low, self.high)
        super().__init__(
            batch_shape=tuple(self.low.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]

        # Bounds may be tensors — fall back to a generic real constraint
        # rather than building an _Interval out of tensor bounds (which
        # the simple ``check`` doesn't handle).
        return real

    @property
    def mean(self) -> Tensor:
        return 0.5 * (self.low + self.high)

    @property
    def variance(self) -> Tensor:
        d = self.high - self.low
        return d * d / 12.0

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + (self.high - self.low) * u

    def log_prob(self, value: Tensor) -> Tensor:
        # log(1 / (high - low)) inside support, -inf outside.
        log_density = -(self.high - self.low).log()
        # Match value's shape for the comparison so 0-dim params + N-dim value work.
        low_b = self.low + value * 0
        high_b = self.high + value * 0
        in_support = (value >= low_b) & (value < high_b)
        return lucid.where(
            in_support,
            log_density + value * 0,
            lucid.full(
                tuple(value.shape),
                -math.inf,
                device=value.device,
                dtype=value.dtype,
            ),
        )

    def cdf(self, value: Tensor) -> Tensor:
        return ((value - self.low) / (self.high - self.low)).clip(0.0, 1.0)

    def icdf(self, value: Tensor) -> Tensor:
        return self.low + value * (self.high - self.low)

    def entropy(self) -> Tensor:
        return (self.high - self.low).log()
