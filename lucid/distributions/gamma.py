"""``Gamma``, ``Chi2``, ``Beta``, ``Dirichlet`` — gamma-family distributions.

The reparameterised sampler for ``Gamma`` uses Marsaglia & Tsang's
acceptance-rejection algorithm wrapped in a fixed-cap retry loop: the
expected reject rate is well under 5 % per sample, so 8 rounds drives
the residual probability of any unsampled cell below ``2 ** −60``.
"""


import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    positive,
    simplex,
    unit_interval,
)
from lucid.distributions.distribution import ExponentialFamily

_MAX_GAMMA_RETRIES: int = 8


from lucid.distributions._util import (
    as_tensor as _as_tensor,
    broadcast_pair as _broadcast_pair,
)


def _sample_standard_gamma(
    concentration: Tensor, sample_shape: tuple[int, ...]
) -> Tensor:
    """Sample ``Gamma(concentration, 1)`` via Marsaglia & Tsang.

    Detached — the engine has no reparameterisation kernel for Gamma.
    Implemented in pure Lucid with a bounded retry loop (rejection rate
    is < 5 % so 8 rounds is overkill).
    """
    out_shape = tuple(sample_shape) + tuple(concentration.shape)
    # Branch for α ≥ 1; α < 1 uses Gamma(α + 1) ** (1/α) trick.
    alpha = concentration
    boost = (alpha < 1.0).to(alpha.dtype)
    alpha_eff = alpha + boost  # Use α + 1 when α < 1.

    d = alpha_eff - 1.0 / 3.0
    c = 1.0 / (9.0 * d).sqrt()

    accepted = lucid.zeros(out_shape, dtype=alpha.dtype, device=alpha.device)
    has_sample = lucid.zeros(out_shape, dtype=alpha.dtype, device=alpha.device)

    for _ in range(_MAX_GAMMA_RETRIES):
        z = lucid.randn(*out_shape, dtype=alpha.dtype, device=alpha.device)
        v = (1.0 + c * z) ** 3
        u = lucid.rand(*out_shape, dtype=alpha.dtype, device=alpha.device)
        u = u.clip(1e-30, 1.0)
        # Acceptance: v > 0  AND  log(u) < 0.5·z² + d − d·v + d·log(v).
        v_safe = v.clip(1e-30, float("inf"))
        log_v = v_safe.log()
        condition = (v > 0) & (u.log() < 0.5 * z * z + d - d * v + d * log_v)
        new_accept = condition & (has_sample == 0)
        new_mask = new_accept.to(alpha.dtype)
        accepted = accepted + new_mask * d * v
        has_sample = has_sample + new_mask
        if bool((has_sample == 1).all().item()):
            break

    # Apply the α<1 transform: x_α = x_{α+1} · U^{1/α}.
    boost_factor = boost * lucid.rand(
        *out_shape, dtype=alpha.dtype, device=alpha.device
    ).clip(1e-30, 1.0) ** (1.0 / alpha.clip(1e-30, float("inf"))) + (
        1.0 - boost
    )  # No-op factor when α ≥ 1.
    return accepted * boost_factor


class Gamma(ExponentialFamily):
    """``Gamma(concentration, rate)`` — shape-rate parameterisation."""

    arg_constraints = {"concentration": positive, "rate": positive}
    support: Constraint | None = positive
    has_rsample = False  # rejection-based; gradients don't flow.

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration = _as_tensor(concentration)
        self.rate = _as_tensor(rate)
        self.concentration, self.rate = _broadcast_pair(self.concentration, self.rate)
        super().__init__(
            batch_shape=tuple(self.concentration.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.rate

    @property
    def mode(self) -> Tensor:
        return ((self.concentration - 1.0).clip(0.0, float("inf"))) / self.rate

    @property
    def variance(self) -> Tensor:
        return self.concentration / (self.rate * self.rate)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        std = _sample_standard_gamma(self.concentration, sample_shape)
        return (std / self.rate).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        return (
            self.concentration * self.rate.log()
            + (self.concentration - 1.0) * value.log()
            - self.rate * value
            - lucid.lgamma(self.concentration)
        )

    def entropy(self) -> Tensor:
        return (
            self.concentration
            - self.rate.log()
            + lucid.lgamma(self.concentration)
            + (1.0 - self.concentration) * lucid.digamma(self.concentration)
        )


class Chi2(Gamma):
    """``Chi²(df)`` — equivalent to ``Gamma(df / 2, 1 / 2)``."""

    arg_constraints = {"df": positive}

    def __init__(
        self,
        df: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.df = _as_tensor(df)
        super().__init__(
            concentration=self.df * 0.5,
            # ``_as_tensor(0.5)`` produces a 0-D scalar; using
            # ``lucid.tensor(0.5)`` would yield a (1,) tensor and force
            # an unwanted broadcast that propagates into the batch shape.
            rate=_as_tensor(0.5),
            validate_args=validate_args,
        )


class Beta(ExponentialFamily):
    """``Beta(α, β)`` on ``[0, 1]``."""

    arg_constraints = {"concentration1": positive, "concentration0": positive}
    support: Constraint | None = unit_interval

    def __init__(
        self,
        concentration1: Tensor | float,
        concentration0: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration1 = _as_tensor(concentration1)
        self.concentration0 = _as_tensor(concentration0)
        self.concentration1, self.concentration0 = _broadcast_pair(
            self.concentration1, self.concentration0
        )
        super().__init__(
            batch_shape=tuple(self.concentration1.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self) -> Tensor:
        a = self.concentration1
        b = self.concentration0
        ab = a + b
        return (a * b) / (ab * ab * (ab + 1.0))

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # Beta(α, β) = X / (X + Y) with X ~ Gamma(α, 1), Y ~ Gamma(β, 1).
        x = _sample_standard_gamma(self.concentration1, sample_shape)
        y = _sample_standard_gamma(self.concentration0, sample_shape)
        return (x / (x + y)).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        a = self.concentration1
        b = self.concentration0
        log_b = lucid.lgamma(a) + lucid.lgamma(b) - lucid.lgamma(a + b)
        return (a - 1.0) * value.log() + (b - 1.0) * (1.0 - value).log() - log_b

    def entropy(self) -> Tensor:
        a = self.concentration1
        b = self.concentration0
        ab = a + b
        log_b = lucid.lgamma(a) + lucid.lgamma(b) - lucid.lgamma(ab)
        return (
            log_b
            - (a - 1.0) * lucid.digamma(a)
            - (b - 1.0) * lucid.digamma(b)
            + (ab - 2.0) * lucid.digamma(ab)
        )


class Dirichlet(ExponentialFamily):
    """``Dirichlet(α)`` on the K-simplex along the last axis."""

    arg_constraints = {"concentration": positive}
    support: Constraint | None = simplex

    def __init__(
        self,
        concentration: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.concentration = _as_tensor(concentration)
        shape = tuple(self.concentration.shape)
        super().__init__(
            batch_shape=shape[:-1],
            event_shape=shape[-1:],
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        s = self.concentration.sum(dim=-1, keepdim=True)
        return self.concentration / s

    @property
    def variance(self) -> Tensor:
        a = self.concentration
        s = a.sum(dim=-1, keepdim=True)
        m = a / s
        return m * (1.0 - m) / (s + 1.0)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        gammas = _sample_standard_gamma(self.concentration, sample_shape)
        return (gammas / gammas.sum(dim=-1, keepdim=True)).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        a = self.concentration
        log_b = lucid.lgamma(a).sum(dim=-1) - lucid.lgamma(a.sum(dim=-1))
        return ((a - 1.0) * value.log()).sum(dim=-1) - log_b

    def entropy(self) -> Tensor:
        a = self.concentration
        s = a.sum(dim=-1, keepdim=True)
        k = a.shape[-1]
        log_b = lucid.lgamma(a).sum(dim=-1, keepdim=True) - lucid.lgamma(s)
        digamma_diff = (a - 1.0) * (lucid.digamma(a) - lucid.digamma(s))
        return (
            log_b + (s - k) * lucid.digamma(s) - digamma_diff.sum(dim=-1, keepdim=True)
        ).squeeze(-1)
