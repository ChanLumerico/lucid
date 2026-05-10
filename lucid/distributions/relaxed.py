"""Concrete (Gumbel-softmax) relaxed distributions.

* :class:`RelaxedBernoulli` — relaxation of ``Bernoulli`` whose samples
  live in ``(0, 1)`` and pass gradients through.
* :class:`RelaxedOneHotCategorical` — Concrete distribution over the
  open simplex; relaxation of :class:`OneHotCategorical`.

Both forward into :func:`lucid.nn.functional.gumbel_softmax` for the
sampling math, so the same Lucid Philox stream applies.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import as_tensor as _as_tensor
from lucid.distributions.bernoulli import (
    _logits_to_probs,
    _probs_to_logits,
)
from lucid.distributions.constraints import (
    Constraint,
    open_unit_interval,
    real,
    simplex,
)
from lucid.distributions.distribution import Distribution


class RelaxedBernoulli(Distribution):
    """Concrete relaxation of ``Bernoulli`` parameterised by a temperature.

    Samples ``y ∈ (0, 1)`` are differentiable wrt the underlying logits
    via the Gumbel-sigmoid reparameterisation:
    ``y = σ((logits + g₁ − g₂) / τ)`` where ``g₁, g₂ ∼ Gumbel(0, 1)``.
    """

    arg_constraints = {
        "temperature": real,
        "probs": open_unit_interval,
        "logits": real,
    }
    support: Constraint | None = open_unit_interval
    has_rsample = True

    def __init__(
        self,
        temperature: Tensor | float,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "RelaxedBernoulli: pass exactly one of `probs` or `logits`."
            )
        self.temperature = _as_tensor(temperature)
        if probs is not None:
            self.probs = _as_tensor(probs)
            self._is_logits = False
            shape = tuple(self.probs.shape)
        else:
            self.logits = _as_tensor(logits)  # type: ignore[arg-type]
            self._is_logits = True
            shape = tuple(self.logits.shape)
        super().__init__(batch_shape=shape, event_shape=(), validate_args=validate_args)

    @property
    def _logits(self) -> Tensor:
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @property
    def _probs(self) -> Tensor:
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        l: Tensor = self._logits + lucid.zeros(
            shape, dtype=self._logits.dtype, device=self._logits.device
        )
        u1: Tensor = lucid.rand(*shape, dtype=l.dtype, device=l.device).clip(
            1e-7, 1.0 - 1e-7
        )
        u2: Tensor = lucid.rand(*shape, dtype=l.dtype, device=l.device).clip(
            1e-7, 1.0 - 1e-7
        )
        # Gumbel(0,1) = −log(−log U).
        g1: Tensor = -(-(u1.log())).log()
        g2: Tensor = -(-(u2.log())).log()
        return ((l + g1 - g2) / self.temperature).sigmoid()

    def log_prob(self, value: Tensor) -> Tensor:
        # Density of the logistic-transformed Concrete (Maddison et al.):
        #   log p(y) = log τ + (logits − τ·log(y/(1−y))) − 2·log(...)
        # We implement the standard form via the unconstrained logit_y.
        logit_y: Tensor = value.log() - (1.0 - value).log()
        l: Tensor = self._logits
        tau: Tensor = self.temperature
        return (
            tau.log()
            + l
            - (tau + 1.0) * (logit_y * tau - l).exp().log1p()
            - tau * logit_y
        )


class RelaxedOneHotCategorical(Distribution):
    """Concrete distribution over the open simplex.

    Sampling delegates to :func:`lucid.nn.functional.gumbel_softmax` with
    ``hard=False``; ``log_prob`` uses the Maddison-Mnih-Teh density.
    """

    arg_constraints = {
        "temperature": real,
        "probs": simplex,
        "logits": real,
    }
    support: Constraint | None = simplex
    has_rsample = True

    def __init__(
        self,
        temperature: Tensor | float,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        from lucid.distributions.categorical import _normalize_probs

        if (probs is None) == (logits is None):
            raise ValueError(
                "RelaxedOneHotCategorical: pass exactly one of `probs` or `logits`."
            )
        self.temperature = _as_tensor(temperature)
        if probs is not None:
            self.probs = _normalize_probs(_as_tensor(probs))
            self._is_logits = False
            shape = tuple(self.probs.shape)
        else:
            self.logits = _as_tensor(logits)  # type: ignore[arg-type]
            self._is_logits = True
            shape = tuple(self.logits.shape)
        self._num_events = shape[-1]
        super().__init__(
            batch_shape=shape[:-1],
            event_shape=shape[-1:],
            validate_args=validate_args,
        )

    @property
    def _logits(self) -> Tensor:
        if self._is_logits:
            return self.logits
        return self.probs.log()

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        from lucid.nn.functional.activations import gumbel_softmax

        # Broadcast logits to (sample_shape + batch_shape + (K,)).
        out_shape = (
            tuple(sample_shape) + tuple(self._batch_shape) + tuple(self._event_shape)
        )
        l = self._logits + lucid.zeros(
            out_shape, dtype=self._logits.dtype, device=self._logits.device
        )
        return gumbel_softmax(l, tau=float(self.temperature.item()), hard=False)

    def log_prob(self, value: Tensor) -> Tensor:
        # Concrete density:  log p(y) = log Γ(K) + (K−1)·log τ
        #                   + Σ (logits − (τ+1)·log y) − K·logsumexp(logits − τ·log y).
        K: int = self._num_events
        l: Tensor = self._logits
        tau: Tensor = self.temperature
        log_y: Tensor = value.log()
        log_z: Tensor = lucid.logsumexp(l - tau * log_y, dim=-1)  # type: ignore[arg-type]
        return (
            float(_lgamma_int(K))
            + (K - 1.0) * tau.log()
            + (l - (tau + 1.0) * log_y).sum(dim=-1)
            - K * log_z
        )


def _lgamma_int(n: int) -> float:
    """``log(Γ(n)) = log((n-1)!)`` for small positive integers."""
    import math

    return math.lgamma(n)
