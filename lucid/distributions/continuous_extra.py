"""Additional continuous distributions: ``Pareto``, ``Weibull``,
``HalfNormal``, ``HalfCauchy``, ``FisherSnedecor``.

All implemented as pure-Lucid composites — no engine work.  Each is
small and built on top of the existing ``Normal`` / ``Cauchy`` /
``Chi2`` infrastructure.
"""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import as_tensor as _as_tensor
from lucid.distributions._util import broadcast_pair as _broadcast_pair
from lucid.distributions.constraints import (
    Constraint,
    greater_than,
    nonnegative,
    positive,
)
from lucid.distributions.distribution import Distribution


class Pareto(Distribution):
    """``Pareto(scale, alpha)`` on ``[scale, ∞)``.  Density:
    ``α · scale^α / x^(α+1)``.
    """

    arg_constraints = {"scale": positive, "alpha": positive}
    has_rsample = True

    def __init__(
        self,
        scale: Tensor | float,
        alpha: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.scale = _as_tensor(scale)
        self.alpha = _as_tensor(alpha)
        self.scale, self.alpha = _broadcast_pair(self.scale, self.alpha)
        super().__init__(
            batch_shape=tuple(self.scale.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        # Lower bound varies per element with ``scale``; report a bare
        # positive constraint and rely on the user to know the family.
        return positive

    @property
    def mean(self) -> Tensor:
        # Defined only for α > 1.
        return self.alpha * self.scale / (self.alpha - 1.0)

    @property
    def variance(self) -> Tensor:
        # Defined only for α > 2.
        a: Tensor = self.alpha
        return self.scale * self.scale * a / ((a - 1.0) ** 2 * (a - 2.0))

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # icdf-trick: x = scale · (1 − U)^(−1/α).
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(*shape, dtype=self.scale.dtype, device=self.scale.device)
        return self.scale * (1.0 - u) ** (-1.0 / self.alpha)

    def log_prob(self, value: Tensor) -> Tensor:
        return (
            self.alpha.log()
            + self.alpha * self.scale.log()
            - (self.alpha + 1.0) * value.log()
        )

    def entropy(self) -> Tensor:
        return (self.scale / self.alpha).log() + 1.0 / self.alpha + 1.0


class Weibull(Distribution):
    """``Weibull(scale, concentration)`` on ``[0, ∞)``.  Density:
    ``(k/λ) · (x/λ)^(k−1) · exp(−(x/λ)^k)``.
    """

    arg_constraints = {"scale": positive, "concentration": positive}
    has_rsample = True

    def __init__(
        self,
        scale: Tensor | float,
        concentration: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.scale = _as_tensor(scale)
        self.concentration = _as_tensor(concentration)
        self.scale, self.concentration = _broadcast_pair(self.scale, self.concentration)
        super().__init__(
            batch_shape=tuple(self.scale.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        return nonnegative

    @property
    def mean(self) -> Tensor:
        # μ = λ · Γ(1 + 1/k).
        return self.scale * lucid.lgamma(1.0 + 1.0 / self.concentration).exp()

    @property
    def variance(self) -> Tensor:
        # Var = λ² · [Γ(1 + 2/k) − Γ(1 + 1/k)²].
        k_inv: Tensor = 1.0 / self.concentration
        g1: Tensor = lucid.lgamma(1.0 + k_inv).exp()
        g2: Tensor = lucid.lgamma(1.0 + 2.0 * k_inv).exp()
        return self.scale * self.scale * (g2 - g1 * g1)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # icdf: x = λ · (−log(1 − U))^(1/k).
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(*shape, dtype=self.scale.dtype, device=self.scale.device)
        return self.scale * (-(1.0 - u).log()) ** (1.0 / self.concentration)

    def log_prob(self, value: Tensor) -> Tensor:
        k: Tensor = self.concentration
        lam: Tensor = self.scale
        return k.log() - k * lam.log() + (k - 1.0) * value.log() - (value / lam) ** k

    def entropy(self) -> Tensor:
        # H = γ·(1 − 1/k) + log(λ/k) + 1, where γ is Euler-Mascheroni.
        EULER_GAMMA: float = 0.5772156649015329
        return (
            EULER_GAMMA * (1.0 - 1.0 / self.concentration)
            + (self.scale / self.concentration).log()
            + 1.0
        )


class HalfNormal(Distribution):
    """``|X|`` where ``X ∼ Normal(0, scale²)``.  Support ``[0, ∞)``."""

    arg_constraints = {"scale": positive}
    support: Constraint | None = nonnegative
    has_rsample = True

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        from lucid.distributions.normal import Normal

        self.scale = _as_tensor(scale)
        self._base = Normal(
            lucid.zeros_like(self.scale), self.scale, validate_args=False
        )
        super().__init__(
            batch_shape=tuple(self.scale.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return self.scale * math.sqrt(2.0 / math.pi)

    @property
    def variance(self) -> Tensor:
        return self.scale * self.scale * (1.0 - 2.0 / math.pi)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self._base.rsample(sample_shape).abs()

    def log_prob(self, value: Tensor) -> Tensor:
        # log p(x) = log 2 + log_normal(x | 0, σ).
        return math.log(2.0) + self._base.log_prob(value)

    def entropy(self) -> Tensor:
        # H = 0.5·log(π·e·σ²/2).
        return 0.5 * (math.pi * math.e / 2.0 * self.scale * self.scale).log()


class HalfCauchy(Distribution):
    """``|X|`` where ``X ∼ Cauchy(0, scale)``.  Support ``[0, ∞)``."""

    arg_constraints = {"scale": positive}
    support: Constraint | None = nonnegative
    has_rsample = True

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        from lucid.distributions.exponential import Cauchy

        self.scale = _as_tensor(scale)
        self._base = Cauchy(
            lucid.zeros_like(self.scale), self.scale, validate_args=False
        )
        super().__init__(
            batch_shape=tuple(self.scale.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        return self._base.rsample(sample_shape).abs()

    def log_prob(self, value: Tensor) -> Tensor:
        return math.log(2.0) + self._base.log_prob(value)


class FisherSnedecor(Distribution):
    """F-distribution: ratio of two independent ``Chi²`` r.v.s, scaled
    by their degrees of freedom.

    ``F(d1, d2) = (X/d1) / (Y/d2)``  where  ``X ∼ Chi²(d1)``  and
    ``Y ∼ Chi²(d2)``.
    """

    arg_constraints = {"df1": positive, "df2": positive}
    support: Constraint | None = positive

    def __init__(
        self,
        df1: Tensor | float,
        df2: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        from lucid.distributions.gamma import Chi2

        self.df1 = _as_tensor(df1)
        self.df2 = _as_tensor(df2)
        self.df1, self.df2 = _broadcast_pair(self.df1, self.df2)
        self._chi1 = Chi2(self.df1, validate_args=False)
        self._chi2 = Chi2(self.df2, validate_args=False)
        super().__init__(
            batch_shape=tuple(self.df1.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        # Defined for d2 > 2.
        return self.df2 / (self.df2 - 2.0)

    @property
    def variance(self) -> Tensor:
        # Defined for d2 > 4.
        d1, d2 = self.df1, self.df2
        return 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) ** 2 * (d2 - 4.0))

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        x: Tensor = self._chi1.sample(sample_shape)
        y: Tensor = self._chi2.sample(sample_shape)
        return (x / self.df1) / (y / self.df2)

    def log_prob(self, value: Tensor) -> Tensor:
        # log p(x) = 0.5·d1·log(d1·x/(d1·x+d2))
        #          + 0.5·d2·log(d2/(d1·x+d2)) − log(x·B(d1/2, d2/2))
        d1, d2 = self.df1, self.df2
        log_beta: Tensor = (
            lucid.lgamma(d1 * 0.5)
            + lucid.lgamma(d2 * 0.5)
            - lucid.lgamma((d1 + d2) * 0.5)
        )
        z: Tensor = d1 * value + d2
        return (
            0.5 * d1 * (d1 * value).log()
            + 0.5 * d2 * d2.log()
            - 0.5 * (d1 + d2) * z.log()
            - value.log()
            - log_beta
        )
