"""``StudentT`` — Student's t with location-scale.

``T = loc + scale · z / sqrt(g / df)``  where  ``z ∼ N(0, 1)``  and
``g ∼ Chi²(df)``.  Reparameterised over ``z`` only (``g`` flows through
the rejection-based Gamma sampler so gradient is detached on it).
"""

import math

import lucid
import lucid.autograd
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import as_tensor as _as_tensor
from lucid.distributions._util import broadcast_pair as _broadcast_pair
from lucid.distributions.constraints import (
    Constraint,
    positive,
    real,
)
from lucid.distributions.distribution import Distribution


class StudentT(Distribution):
    """Three-parameter location-scale Student's t."""

    arg_constraints = {"df": positive, "loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample: bool = True

    def __init__(
        self,
        df: Tensor | float,
        loc: Tensor | float = 0.0,
        scale: Tensor | float = 1.0,
        validate_args: bool | None = None,
    ) -> None:
        self.df = _as_tensor(df)
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.df, self.loc = _broadcast_pair(self.df, self.loc)
        self.df, self.scale = _broadcast_pair(self.df, self.scale)
        self.loc, self.scale = _broadcast_pair(self.loc, self.scale)
        super().__init__(
            batch_shape=tuple(self.df.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        # Defined for df > 1 — we follow the reference framework and
        # return ``loc`` regardless.
        return self.loc + 0 * self.df

    @property
    def variance(self) -> Tensor:
        # Defined for df > 2:  scale² · df / (df − 2).
        return self.scale * self.scale * self.df / (self.df - 2.0)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample: gradient flows through the Normal variate.

        ``T = loc + scale · z / sqrt(g / df)``  where  ``z ~ N(0,1)``
        is differentiable and ``g ~ Chi²(df)`` is detached (standard
        practice — the marginal gradient w.r.t. df is not tracked).
        """
        from lucid.distributions.gamma import _sample_standard_gamma

        shape = self._extended_shape(sample_shape)
        z = lucid.randn(*shape, dtype=self.loc.dtype, device=self.loc.device)
        gamma_std = _sample_standard_gamma(self.df * 0.5, sample_shape).detach() * 2.0
        return self.loc + self.scale * z * (self.df / gamma_std).sqrt()

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        with lucid.autograd.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        z = (value - self.loc) / self.scale
        log_density = (
            lucid.lgamma((self.df + 1.0) * 0.5)
            - lucid.lgamma(self.df * 0.5)
            - 0.5 * (math.pi * self.df).log()
            - self.scale.log()
            - 0.5 * (self.df + 1.0) * (1.0 + z * z / self.df).log()
        )
        return log_density

    def entropy(self) -> Tensor:
        # H = log(scale·sqrt(df)·B(0.5, df/2)) + (df+1)/2 · (digamma((df+1)/2) − digamma(df/2)).
        half = lucid.tensor(0.5) + 0 * self.df  # broadcast 0.5 → df-shape.
        beta_term = (
            lucid.lgamma(half)
            + lucid.lgamma(self.df * 0.5)
            - lucid.lgamma((self.df + 1.0) * 0.5)
        )
        return (
            self.scale.log()
            + 0.5 * self.df.log()
            + beta_term
            + (self.df + 1.0)
            * 0.5
            * (lucid.digamma((self.df + 1.0) * 0.5) - lucid.digamma(self.df * 0.5))
        )
