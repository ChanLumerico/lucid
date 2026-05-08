"""Discrete ``Bernoulli`` and ``Geometric``."""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    boolean,
    nonnegative_integer,
    open_unit_interval,
    real,
    unit_interval,
)
from lucid.distributions.distribution import Distribution, ExponentialFamily


from lucid.distributions._util import as_tensor as _as_tensor


def _probs_to_logits(probs: Tensor) -> Tensor:
    return probs.log() - (1.0 - probs).log()


def _logits_to_probs(logits: Tensor) -> Tensor:
    return logits.sigmoid()


class Bernoulli(ExponentialFamily):
    """Bernoulli distribution.  Specify *exactly one* of ``probs`` or
    ``logits`` — the other is derived lazily."""

    arg_constraints = {"probs": unit_interval, "logits": real}
    support: Constraint | None = boolean
    has_enumerate_support = True

    def __init__(
        self,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Bernoulli: pass exactly one of `probs` or `logits`.")
        if probs is not None:
            self.probs = _as_tensor(probs)
            self._is_logits = False
        else:
            self.logits = _as_tensor(logits)
            self._is_logits = True
        shape = (
            tuple(self.probs.shape) if not self._is_logits else tuple(self.logits.shape)
        )
        super().__init__(batch_shape=shape, event_shape=(), validate_args=validate_args)

    @property
    def param(self) -> Tensor:
        return self.logits if self._is_logits else self.probs

    @property
    def _probs(self) -> Tensor:
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @property
    def _logits(self) -> Tensor:
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @property
    def mean(self) -> Tensor:
        return self._probs

    @property
    def variance(self) -> Tensor:
        p = self._probs
        return p * (1.0 - p)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self._probs.dtype, device=self._probs.device)
        p = self._probs + lucid.zeros(
            shape, dtype=self._probs.dtype, device=self._probs.device
        )
        out = (u < p).to(self._probs.dtype)
        return out.detach()

    def log_prob(self, value: Tensor) -> Tensor:
        # Numerically stable form via logits + softplus identity:
        #   log p(x | l) = x · l − softplus(l)
        l = self._logits
        return value * l - (1.0 + l.exp()).log()

    def entropy(self) -> Tensor:
        # H = − p log p − (1−p) log(1−p), guarded by softplus form.
        l = self._logits
        return (1.0 + l.exp()).log() - self._probs * l


class Geometric(Distribution):
    """``Geometric(probs)`` over ``{0, 1, 2, ...}`` — number of failures
    before the first success of a series of Bernoulli trials with
    success probability ``probs``."""

    arg_constraints = {"probs": open_unit_interval, "logits": real}
    support: Constraint | None = nonnegative_integer

    def __init__(
        self,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Geometric: pass exactly one of `probs` or `logits`.")
        if probs is not None:
            self.probs = _as_tensor(probs)
        else:
            self.probs = _logits_to_probs(_as_tensor(logits))
        super().__init__(
            batch_shape=tuple(self.probs.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        return (1.0 - self.probs) / self.probs

    @property
    def variance(self) -> Tensor:
        return (1.0 - self.probs) / (self.probs * self.probs)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        # icdf-trick: floor(log(U) / log(1 - p)).  Detached — discrete.
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.probs.dtype, device=self.probs.device)
        # Avoid u=0 which would give -inf.  Lucid's rand draws from
        # [0, 1) so u=0 is admissible — clamp upward by epsilon.
        u_safe = u.clip(1e-7, 1.0 - 1e-7)
        return (u_safe.log() / (1.0 - self.probs).log()).floor().detach()

    def log_prob(self, value: Tensor) -> Tensor:
        return value * (1.0 - self.probs).log() + self.probs.log()

    def entropy(self) -> Tensor:
        p = self.probs
        return -((1.0 - p) * (1.0 - p).log() + p * p.log()) / p
