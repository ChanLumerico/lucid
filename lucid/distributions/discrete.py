"""Discrete count distributions: ``Poisson``, ``Binomial``,
``NegativeBinomial``.

All three are pure-Lucid composites — sampling reuses
``lucid.poisson`` and the per-distribution helpers, log-prob is
closed-form via lgamma.
"""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import as_tensor as _as_tensor
from lucid.distributions._util import broadcast_pair as _broadcast_pair
from lucid.distributions.bernoulli import (
    _logits_to_probs,
    _probs_to_logits,
)
from lucid.distributions.constraints import (
    Constraint,
    nonnegative_integer,
    open_unit_interval,
    positive,
    real,
    unit_interval,
)
from lucid.distributions.distribution import Distribution, ExponentialFamily


class Poisson(ExponentialFamily):
    """``Poisson(rate)`` over non-negative integers."""

    arg_constraints = {"rate": positive}
    support: Constraint | None = nonnegative_integer

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
        return self.rate

    @property
    def mode(self) -> Tensor:
        # ``floor(rate)`` — when rate is integer, both floor and floor−1
        # are modes; we follow the reference framework and take floor.
        return self.rate.floor()

    @property
    def variance(self) -> Tensor:
        return self.rate

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # Top-level ``lucid.poisson`` does the Knuth + Normal-approx work
        # using the Lucid Philox PRNG, so ``manual_seed`` controls the
        # stream end-to-end.
        rate_b: Tensor = self.rate + lucid.zeros(
            self._extended_shape(sample_shape),
            dtype=self.rate.dtype,
            device=self.rate.device,
        )
        return lucid.poisson(rate_b)

    def log_prob(self, value: Tensor) -> Tensor:
        # log p(k | λ) = k·log(λ) − λ − lgamma(k+1).
        return value * self.rate.log() - self.rate - lucid.lgamma(value + 1.0)

    def entropy(self) -> Tensor:
        # Closed-form entropy is an infinite sum; the reference framework
        # returns NotImplemented.  We follow.
        raise NotImplementedError("Poisson.entropy: no closed form available")


class Binomial(Distribution):
    """``Binomial(total_count, probs|logits)`` — number of successes in
    ``total_count`` independent ``Bernoulli`` trials."""

    arg_constraints = {
        "total_count": nonnegative_integer,
        "probs": unit_interval,
        "logits": real,
    }
    has_enumerate_support = True

    _SMALL_N_CUTOFF: int = 25

    def __init__(
        self,
        total_count: Tensor | int = 1,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Binomial: pass exactly one of `probs` or `logits`."
            )
        self.total_count: Tensor = _as_tensor(total_count)
        if probs is not None:
            self.probs = _as_tensor(probs)
            self._is_logits = False
            shape = tuple(self.probs.shape)
        else:
            self.logits = _as_tensor(logits)
            self._is_logits = True
            shape = tuple(self.logits.shape)
        # Broadcast total_count against probs/logits.
        param: Tensor = self.logits if self._is_logits else self.probs
        self.total_count, _ = _broadcast_pair(self.total_count, param)
        super().__init__(
            batch_shape=shape,
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        # Per-element {0, 1, …, total_count}; we report nonneg-integer
        # since ``total_count`` may differ across the batch.
        return nonnegative_integer

    @property
    def _probs(self) -> Tensor:
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @property
    def _logits(self) -> Tensor:
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @property
    def mean(self) -> Tensor:
        return self.total_count * self._probs

    @property
    def variance(self) -> Tensor:
        p: Tensor = self._probs
        return self.total_count * p * (1.0 - p)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # Strategy:
        #   * If max(total_count) ≤ cutoff: sum of Bernoullis along an
        #     extra axis — exact.
        #   * Otherwise: Normal approximation rounded to the integer
        #     range [0, total_count].
        max_n: int = int(self.total_count.max().item())
        out_shape: tuple[int, ...] = self._extended_shape(sample_shape)
        device = self._probs.device
        dtype = self._probs.dtype

        if max_n <= self._SMALL_N_CUTOFF:
            # Build (max_n, *out_shape) uniform draws, mask by trial-wise
            # `i < total_count`, count where U < p.
            full_shape: tuple[int, ...] = (max_n,) + out_shape
            u: Tensor = lucid.rand(max_n, *out_shape, dtype=dtype, device=device)
            p_b: Tensor = self._probs + lucid.zeros(
                full_shape, dtype=dtype, device=device
            )
            tc_b_full: Tensor = self.total_count + lucid.zeros(
                full_shape, dtype=dtype, device=device
            )
            # Per-trial active mask: trial index i is active if i < tc.
            i_flat: Tensor = lucid.arange(
                0, max_n, 1, dtype=dtype, device=device
            )
            i_idx: Tensor = i_flat.reshape(
                [max_n] + [1] * len(out_shape)
            ) + lucid.zeros(full_shape, dtype=dtype, device=device)
            active: Tensor = (i_idx < tc_b_full).to(dtype)
            success: Tensor = (u < p_b).to(dtype) * active
            return success.sum(dim=0).to(lucid.int64).detach()

        # Normal approximation.
        mu: Tensor = self.mean + lucid.zeros(out_shape, dtype=dtype, device=device)
        sigma: Tensor = self.variance.sqrt() + lucid.zeros(
            out_shape, dtype=dtype, device=device
        )
        z: Tensor = lucid.randn(*out_shape, dtype=dtype, device=device)
        sample: Tensor = (mu + sigma * z + 0.5).floor()
        # Clamp to [0, total_count].
        zero: Tensor = lucid.zeros_like(sample)
        tc_b = self.total_count + zero
        sample = sample.maximum(zero).minimum(tc_b)
        return sample.to(lucid.int64).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        # log p(k) = lgamma(n+1) − lgamma(k+1) − lgamma(n−k+1)
        #           + k·log(p) + (n−k)·log(1−p).
        n: Tensor = self.total_count
        k: Tensor = value
        log_comb: Tensor = (
            lucid.lgamma(n + 1.0)
            - lucid.lgamma(k + 1.0)
            - lucid.lgamma(n - k + 1.0)
        )
        # Stable form via logits: k·l − n·softplus(l).
        l: Tensor = self._logits
        return log_comb + k * l - n * (1.0 + l.exp()).log()


class NegativeBinomial(Distribution):
    """``NegativeBinomial(total_count, probs|logits)`` — number of failures
    before ``total_count`` successes (Gamma-Poisson compound)."""

    arg_constraints = {
        "total_count": positive,
        "probs": open_unit_interval,
        "logits": real,
    }
    support: Constraint | None = nonnegative_integer

    def __init__(
        self,
        total_count: Tensor | float,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "NegativeBinomial: pass exactly one of `probs` or `logits`."
            )
        self.total_count = _as_tensor(total_count)
        if probs is not None:
            self.probs = _as_tensor(probs)
            self._is_logits = False
            shape = tuple(self.probs.shape)
        else:
            self.logits = _as_tensor(logits)
            self._is_logits = True
            shape = tuple(self.logits.shape)
        super().__init__(
            batch_shape=shape,
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def _probs(self) -> Tensor:
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @property
    def _logits(self) -> Tensor:
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @property
    def mean(self) -> Tensor:
        # E[X] = r · p / (1 − p).
        p: Tensor = self._probs
        return self.total_count * p / (1.0 - p)

    @property
    def variance(self) -> Tensor:
        # Var = r · p / (1 − p)².
        p: Tensor = self._probs
        return self.total_count * p / ((1.0 - p) ** 2)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # Gamma-Poisson compound: X = Poisson(λ),  λ = Gamma(r, (1−p)/p).
        from lucid.distributions.gamma import _sample_standard_gamma

        out_shape: tuple[int, ...] = self._extended_shape(sample_shape)
        p: Tensor = self._probs + lucid.zeros(
            out_shape, dtype=self._probs.dtype, device=self._probs.device
        )
        r: Tensor = self.total_count + lucid.zeros(
            out_shape, dtype=self._probs.dtype, device=self._probs.device
        )
        # Gamma(r, rate=(1−p)/p) — sample standard Gamma(r) then divide
        # by the rate.
        rate: Tensor = (1.0 - p) / p
        std_gamma: Tensor = _sample_standard_gamma(r, sample_shape)
        lam: Tensor = std_gamma / rate
        return lucid.poisson(lam).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        # log p(k) = lgamma(k+r) − lgamma(r) − lgamma(k+1)
        #           + r·log(1−p) + k·log(p).
        k: Tensor = value
        r: Tensor = self.total_count
        p: Tensor = self._probs
        return (
            lucid.lgamma(k + r)
            - lucid.lgamma(r)
            - lucid.lgamma(k + 1.0)
            + r * (1.0 - p).log()
            + k * p.log()
        )
