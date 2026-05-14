"""Additional distributions: ``Gumbel``, ``InverseGamma``, ``Kumaraswamy``,
``Multinomial``, ``ContinuousBernoulli``.

All are pure-Lucid composites — no engine work required.  Each distribution
is numerically stable in its main operating region, using Taylor
approximations or clamping where analytic expressions degenerate.
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
from lucid.distributions.distribution import Distribution

# ── Euler–Mascheroni constant ─────────────────────────────────────────────────
_EULER_GAMMA: float = 0.5772156649015329


# ── Gumbel ────────────────────────────────────────────────────────────────────


class Gumbel(Distribution):
    """Gumbel (Type-I extreme value) distribution on ``ℝ``.

    ``Gumbel(loc, scale)`` — density:
    ``(1/s) · exp(−(z + exp(−z)))``  where  ``z = (x − μ) / s``.

    Widely used for Gumbel-softmax / top-k sampling tricks.
    """

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample: bool = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Gumbel distribution.

        Parameters
        ----------
        loc : Tensor | float
            Location parameter :math:`\mu \in \mathbb{R}`.  Shifts the
            distribution along the real line.
        scale : Tensor | float
            Scale parameter :math:`s > 0`.  Stretches the distribution.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
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
        """``μ + s · γ`` where ``γ`` is the Euler–Mascheroni constant."""
        return self.loc + self.scale * _EULER_GAMMA

    @property
    def variance(self) -> Tensor:
        """``(π · s)² / 6``."""
        return (math.pi * self.scale) ** 2 / 6.0

    @property
    def mode(self) -> Tensor:
        """Mode of the Gumbel distribution.

        Returns
        -------
        Tensor
            The location parameter :math:`\\mu`, which coincides with the mode
            of the Gumbel PDF.
        """
        return self.loc

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample via the Gumbel icdf: ``μ − s · log(−log U)``."""
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device)
        # Clamp away from 0 and 1 to avoid log(0) = -inf.
        u = lucid.clamp(u, min=1e-6, max=1.0 - 1e-6)
        return self.loc - self.scale * (-(u.log())).log()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Gumbel distribution.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`x \in \mathbb{R}` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density :math:`\log p(x) = -(z + e^{-z}) - \log s`
            where :math:`z = (x - \mu) / s`.
        """
        z: Tensor = (value - self.loc) / self.scale
        return -(z + (-z).exp()) - self.scale.log()

    def entropy(self) -> Tensor:
        """``log(s) + γ + 1``."""
        return self.scale.log() + _EULER_GAMMA + 1.0


# ── InverseGamma ──────────────────────────────────────────────────────────────


class InverseGamma(Distribution):
    """Inverse-Gamma distribution: ``X = 1 / Y``  where  ``Y ~ Gamma(α, β)``.

    ``InverseGamma(concentration=α, rate=β)`` — density on ``(0, ∞)``:
    ``β^α / Γ(α) · x^(−α−1) · exp(−β / x)``.
    """

    arg_constraints = {"concentration": positive, "rate": positive}
    support: Constraint | None = positive
    # Lucid's Gamma sampler uses rejection sampling without a reparameterised
    # kernel, so samples are detached — has_rsample is False.
    has_rsample: bool = False

    def __init__(
        self,
        concentration: Tensor | float,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise an Inverse-Gamma distribution.

        Parameters
        ----------
        concentration : Tensor | float
            Shape parameter :math:`\alpha > 0`.
        rate : Tensor | float
            Rate (inverse-scale) parameter :math:`\beta > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        self.concentration = _as_tensor(concentration)
        self.rate = _as_tensor(rate)
        self.concentration, self.rate = _broadcast_pair(self.concentration, self.rate)
        # Lazily import Gamma to avoid a circular dependency at import time.
        from lucid.distributions.gamma import Gamma as _Gamma

        self._gamma = _Gamma(self.concentration, self.rate, validate_args=False)
        super().__init__(
            batch_shape=tuple(self.concentration.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        """Defined for ``concentration > 1``: ``β / (α − 1)``."""
        return self.rate / (self.concentration - 1.0)

    @property
    def variance(self) -> Tensor:
        """Defined for ``concentration > 2``:
        ``β² / ((α − 1)² (α − 2))``."""
        c, r = self.concentration, self.rate
        return r * r / ((c - 1.0) ** 2 * (c - 2.0))

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Sample by taking the reciprocal of a ``Gamma`` sample."""
        y: Tensor = self._gamma.sample(sample_shape)
        return (1.0 / y).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Inverse-Gamma distribution.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`x > 0` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density :math:`\alpha\log\beta - \log\Gamma(\alpha)
            - (\alpha+1)\log x - \beta/x`.
        """
        c, r = self.concentration, self.rate
        return c * r.log() - lucid.lgamma(c) - (c + 1.0) * value.log() - r / value

    def entropy(self) -> Tensor:
        """``α + log β + log Γ(α) − (1 + α) · ψ(α)``."""
        c, r = self.concentration, self.rate
        return c + r.log() + lucid.lgamma(c) - (c + 1.0) * lucid.digamma(c)


# ── Kumaraswamy ───────────────────────────────────────────────────────────────


class Kumaraswamy(Distribution):
    """Kumaraswamy distribution on ``(0, 1)`` — a Beta-like family that
    has a closed-form CDF and supports cheap reparameterised sampling.

    ``Kumaraswamy(concentration1=a, concentration0=b)`` — density:
    ``a · b · x^(a−1) · (1 − x^a)^(b−1)``.
    """

    arg_constraints = {"concentration1": positive, "concentration0": positive}
    support: Constraint | None = open_unit_interval
    has_rsample: bool = True

    def __init__(
        self,
        concentration1: Tensor | float,
        concentration0: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise a Kumaraswamy distribution.

        Parameters
        ----------
        concentration1 : Tensor | float
            First shape parameter :math:`a > 0` (controls the left tail).
        concentration0 : Tensor | float
            Second shape parameter :math:`b > 0` (controls the right tail).
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        self.concentration1 = _as_tensor(concentration1)  # a
        self.concentration0 = _as_tensor(concentration0)  # b
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
        """``E[X] = b · B(1 + 1/a, b)`` expressed via ``lgamma``."""
        a, b = self.concentration1, self.concentration0
        log_mean = (
            b.log()
            + lucid.lgamma(1.0 + 1.0 / a)
            + lucid.lgamma(b)
            - lucid.lgamma(1.0 + 1.0 / a + b)
        )
        return log_mean.exp()

    @property
    def variance(self) -> Tensor:
        """``Var[X] = E[X²] − E[X]²``  where ``E[X²] = b · B(1 + 2/a, b)``."""
        a, b = self.concentration1, self.concentration0
        log_sq_mean = (
            b.log()
            + lucid.lgamma(1.0 + 2.0 / a)
            + lucid.lgamma(b)
            - lucid.lgamma(1.0 + 2.0 / a + b)
        )
        return log_sq_mean.exp() - self.mean**2

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample via the icdf:
        ``x = (1 − (1 − U)^(1/b))^(1/a)``."""
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(
            *shape,
            dtype=self.concentration1.dtype,
            device=self.concentration1.device,
        )
        u = lucid.clamp(u, min=1e-6, max=1.0 - 1e-6)
        a, b = self.concentration1, self.concentration0
        return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Kumaraswamy distribution.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`x \in (0, 1)` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density :math:`\log a + \log b + (a-1)\log x + (b-1)\log(1-x^a)`.
        """
        a, b = self.concentration1, self.concentration0
        return (
            a.log()
            + b.log()
            + (a - 1.0) * value.log()
            + (b - 1.0) * (1.0 - value**a).log()
        )

    def entropy(self) -> Tensor:
        """Entropy via the Beta-distributed auxiliary variable Y = X^a.

        If X ~ Kumaraswamy(a, b) then Y = X^a ~ Beta(1, b), giving:
          E[log X]       = (1/a)(ψ(1) − ψ(b+1)) = −(1/a)(γ + ψ(b+1))
          E[log(1−X^a)] = ψ(b) − ψ(b+1)         = −1/b

        H = −log a − log b − (a−1)·E[log X] − (b−1)·E[log(1−X^a)]
          = −log a − log b + (1 − 1/a)(γ + ψ(b+1)) + (1 − 1/b)
        """
        a, b = self.concentration1, self.concentration0
        # ψ(b+1) = ψ(concentration0 + 1)
        harmonic: Tensor = _EULER_GAMMA + lucid.digamma(b + 1.0)
        return (1.0 - 1.0 / a) * harmonic + (1.0 - 1.0 / b) - a.log() - b.log()


# ── Multinomial ───────────────────────────────────────────────────────────────


class Multinomial(Distribution):
    """Multinomial distribution over ``K`` categories.

    ``Multinomial(total_count, probs)`` — counts of ``total_count`` draws
    from a Categorical with ``probs``.  The event dimension is the last axis
    of the probability vector.

    Parameters
    ----------
    total_count : int | Tensor
        Number of draws (scalar, non-negative integer).
    probs : Tensor | None
        Category probabilities (unnormalised; will be normalised internally).
        Mutually exclusive with ``logits``.
    logits : Tensor | None
        Log-unnormalised category probabilities.
    """

    arg_constraints = {
        "total_count": nonnegative_integer,
        "probs": unit_interval,
    }
    has_rsample: bool = False

    def __init__(
        self,
        total_count: Tensor | int = 1,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Multinomial distribution.

        Parameters
        ----------
        total_count : Tensor | int, optional
            Total number of draws :math:`n \geq 0`.  Must be a scalar.
            Default is ``1`` (reduces to a one-hot Categorical).
        probs : Tensor | None, optional
            Unnormalised probability vector(s) of shape ``(..., K)``.
            Normalised to sum to 1 internally.  Mutually exclusive with
            ``logits``.
        logits : Tensor | None, optional
            Unnormalised log-probability vector(s) of shape ``(..., K)``.
            Converted to probabilities via softmax.  Mutually exclusive with
            ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        if (probs is None) == (logits is None):
            raise ValueError("Multinomial: pass exactly one of `probs` or `logits`.")
        if probs is not None:
            self._param = _as_tensor(probs)
            self._is_logits = False
        else:
            self._param = _as_tensor(logits)  # type: ignore[arg-type]
            self._is_logits = True
        self._total_count: Tensor = _as_tensor(total_count)
        # batch_shape is everything but the last (category) dim.
        batch_shape: tuple[int, ...] = tuple(self._param.shape[:-1])
        event_shape: tuple[int, ...] = (int(self._param.shape[-1]),)
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        """Support of the distribution: non-negative integers.

        Returns
        -------
        Constraint
            ``nonnegative_integer`` — the multinomial event count vector lives
            in :math:`\\mathbb{Z}_{\\geq 0}^K`, subject to the additional
            constraint that the counts sum to ``total_count``.
        """
        return nonnegative_integer

    @property
    def _probs(self) -> Tensor:
        """Lazily resolved category probabilities (summing to 1 along the last axis).

        Applies a max-shifted softmax to the stored logits when the
        distribution was constructed from ``logits``; otherwise renormalises
        the stored raw probabilities along the final axis.
        """
        if self._is_logits:
            lp: Tensor = self._param
            # softmax along last dim
            lp_max = lp - lp.max(dim=-1, keepdim=True)
            exp_lp = lp_max.exp()
            return exp_lp / exp_lp.sum(dim=-1, keepdim=True)
        # Normalise raw probs.
        p = self._param
        return p / p.sum(dim=-1, keepdim=True)

    @property
    def _logits(self) -> Tensor:
        """Lazily resolved per-category logits.

        Returns the stored parameter when constructed from ``logits``;
        otherwise computes :math:`\\log(p/(1-p))` from the normalised probs.
        """
        if self._is_logits:
            return self._param
        return _probs_to_logits(self._probs)

    @property
    def total_count(self) -> Tensor:
        """Total number of trials :math:`n` per Multinomial draw.

        Returns
        -------
        Tensor
            The integer-valued ``n`` parameter broadcast over the batch shape.
            Each independent Multinomial sums to this many trials across the
            ``K`` categories.
        """
        return self._total_count

    @property
    def mean(self) -> Tensor:
        """``n · p`` (element-wise)."""
        return self._total_count * self._probs

    @property
    def variance(self) -> Tensor:
        """``n · p · (1 − p)`` (element-wise)."""
        p: Tensor = self._probs
        return self._total_count * p * (1.0 - p)

    def log_prob(self, value: Tensor) -> Tensor:
        """``log C(n; k₁,…,kK) + Σ kᵢ log pᵢ``."""
        n: Tensor = self._total_count
        # Multinomial coefficient: lgamma(n+1) - sum_i lgamma(k_i+1)
        log_coeff: Tensor = lucid.lgamma(n + 1.0) - lucid.lgamma(value + 1.0).sum(
            dim=-1
        )
        # Σ kᵢ log pᵢ — numerically stable via logits
        log_p_term: Tensor = (value * self._probs.log()).sum(dim=-1)
        return log_coeff + log_p_term

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw samples by summing one-hot Categorical draws."""
        from lucid.distributions.categorical import Categorical

        n: int = int(self._total_count.item())
        p: Tensor = self._probs  # (..., K)
        K: int = int(p.shape[-1])
        cat = Categorical(probs=p)
        # Draw n categorical samples: (n, *batch_shape)
        draws: list[Tensor] = [cat.sample(sample_shape) for _ in range(n)]
        # Count each category k in {0, ..., K-1}.
        counts: list[Tensor] = [
            lucid.stack([(d == k).to(p.dtype) for d in draws]).sum(dim=0)
            for k in range(K)
        ]
        # Stack along last dim: (*sample_shape, *batch_shape, K)
        return lucid.stack(counts, dim=-1).to(lucid.int64).detach()


# ── ContinuousBernoulli ───────────────────────────────────────────────────────


class ContinuousBernoulli(Distribution):
    """Continuous Bernoulli distribution on ``[0, 1]``.

    Like a Bernoulli but with a normalising constant ``C(p)`` that makes the
    density integrate to 1 over the continuous interval ``[0, 1]``:

    ``p(x | p) = C(p) · pˣ · (1−p)^(1−x)``

    where ``C(p) = |logit(p)| / |2p − 1|`` for ``p ≠ ½``, ``C(½) = 2``.

    Common in generative models with Bernoulli-like decoders over [0,1] data.
    """

    arg_constraints = {"probs": unit_interval, "logits": real}
    support: Constraint | None = unit_interval
    has_rsample: bool = True

    def __init__(
        self,
        probs: Tensor | float | None = None,
        logits: Tensor | float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Continuous Bernoulli distribution.

        Parameters
        ----------
        probs : Tensor | float | None, optional
            Parameter :math:`p \in [0, 1]`.  Mutually exclusive with
            ``logits``.
        logits : Tensor | float | None, optional
            Log-odds :math:`l = \log(p/(1-p)) \in \mathbb{R}`.  Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        if (probs is None) == (logits is None):
            raise ValueError(
                "ContinuousBernoulli: pass exactly one of `probs` or `logits`."
            )
        if probs is not None:
            self._param = _as_tensor(probs)
            self._is_logits = False
        else:
            self._param = _as_tensor(logits)  # type: ignore[arg-type]
            self._is_logits = True
        super().__init__(
            batch_shape=tuple(self._param.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def _probs(self) -> Tensor:
        """Lazily resolved probability parameter :math:`p \\in [0, 1]`.

        Returns the stored parameter when constructed from ``probs``;
        otherwise applies the sigmoid to the stored logits.
        """
        return self._param if not self._is_logits else _logits_to_probs(self._param)

    @property
    def _logits(self) -> Tensor:
        """Lazily resolved logit parameter :math:`\\ell = \\log(p/(1-p))`.

        Returns the stored parameter when constructed from ``logits``;
        otherwise computes the log-odds from the stored probs.
        """
        return self._param if self._is_logits else _probs_to_logits(self._param)

    # -- helpers ---------------------------------------------------------------

    def _log_normalizer(self) -> Tensor:
        """Stable log C(p):
        ``log(|logit(p)|) − log(|2p−1|)`` far from ½;
        Taylor ``log(2) + (2p−1)²/3`` near ½.
        """
        p: Tensor = _logits_to_probs(lucid.clamp(self._logits, min=-20.0, max=20.0))
        u: Tensor = 2.0 * p - 1.0  # u = 2p − 1 ∈ (−1, 1)
        abs_u: Tensor = u.abs()
        eps: float = 1e-4
        # For the stable branch, replace u with a safe value when near zero.
        safe_u: Tensor = lucid.where(abs_u < eps, lucid.full_like(u, eps), u)
        log_norm_stable: Tensor = (
            math.log(2.0) + lucid.atanh(safe_u).abs().log() - safe_u.abs().log()
        )
        log_norm_taylor: Tensor = math.log(2.0) + (u * u) / 3.0
        return lucid.where(abs_u < eps, log_norm_taylor, log_norm_stable)

    # -- distribution interface ------------------------------------------------

    @property
    def mean(self) -> Tensor:
        """``p / (2p−1) + 1 / (2 atanh(1−2p))``; ``½`` when ``p = ½``."""
        p: Tensor = self._probs
        u: Tensor = 2.0 * p - 1.0
        abs_u: Tensor = u.abs()
        eps: float = 1e-4
        safe_u: Tensor = lucid.where(abs_u < eps, lucid.full_like(u, eps), u)
        # mean = p/(2p−1) + 1/(2·atanh(1−2p))
        # Note: atanh(1−2p) = atanh(−u) = −atanh(u)
        stable: Tensor = p / u - 1.0 / (2.0 * lucid.atanh(safe_u))
        near: Tensor = lucid.full_like(p, 0.5)
        return lucid.where(abs_u < eps, near, stable)

    @property
    def variance(self) -> Tensor:
        """``E[X²] − E[X]²``; ``E[X²]`` computed via the same normaliser trick."""
        p: Tensor = self._probs
        u: Tensor = 2.0 * p - 1.0
        abs_u: Tensor = u.abs()
        eps: float = 1e-4
        safe_u: Tensor = lucid.where(abs_u < eps, lucid.full_like(u, eps), u)
        # E[X^2] = C(p)*(1-p)*(exp(l)*(l^2-2l+2) - 2) / l^3 — complex to derive;
        # numerically stable shortcut: Var = mean*(1-mean) + ... is not closed-form.
        # Use: Var = E[X^2] - E[X]^2 where E[X^2] from the normaliser.
        # Simpler direct formula (closed-form shortcut):
        # Var = mean - mean^2 - mean*(2p-1)/(2*atanh(1-2p)) ... equally complex.
        # Implement via the second moment integral result:
        # E[X^2] = p/(2p-1) + p*(p-1)/(l*(2p-1)) where l = logit(p)
        # = C*(1-p)*[l*e^l - e^l + 1] / l^2 ... let's use a direct stable form.
        mean_val: Tensor = self.mean
        l: Tensor = self._logits
        safe_l: Tensor = lucid.where(abs_u < eps, lucid.full_like(l, eps), l)
        # E[X^2] formula: 2nd moment = mean + (mean - 2*p*mean) / l
        # = mean*(1 + (1-2p)/l) = mean + mean*(1-2p)/l
        sq_mean_stable: Tensor = mean_val + mean_val * (-u) / safe_l
        sq_mean_near: Tensor = lucid.full_like(p, 1.0 / 3.0)
        sq_mean: Tensor = lucid.where(abs_u < eps, sq_mean_near, sq_mean_stable)
        return lucid.clamp(sq_mean - mean_val * mean_val, min=0.0)  # type: ignore[call-arg]

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample via the closed-form icdf."""
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(
            *shape, dtype=self._param.dtype, device=self._param.device
        )
        u = lucid.clamp(u, min=1e-6, max=1.0 - 1e-6)
        # Broadcast distribution parameters to the full sample shape so that
        # lucid.where (which requires same-shape operands) always succeeds.
        zeros_b: Tensor = lucid.zeros(
            *shape, dtype=self._param.dtype, device=self._param.device
        )
        p: Tensor = self._probs + zeros_b
        l: Tensor = self._logits + zeros_b
        d: Tensor = 2.0 * p - 1.0  # ∈ (−1, 1) \ {0}
        abs_d: Tensor = d.abs()
        eps: float = 1e-4
        safe_d: Tensor = lucid.where(abs_d < eps, lucid.full_like(d, eps), d)
        safe_l: Tensor = lucid.where(abs_d < eps, lucid.full_like(l, eps), l)
        # icdf: x = log(1 + u · (2p−1) / (1−p)) / logit(p)
        stable_x: Tensor = (1.0 + u * safe_d / (1.0 - p)).log() / safe_l
        near_x: Tensor = u  # uniform limit at p = ½
        return lucid.clamp(lucid.where(abs_d < eps, near_x, stable_x), min=0.0, max=1.0)

    def log_prob(self, value: Tensor) -> Tensor:
        """``x · l − softplus(l) + log C(p)``."""
        l: Tensor = self._logits
        # log p(x) = x * l - log(1 + exp(l)) + log C(p)
        return value * l - (1.0 + l.exp()).log() + self._log_normalizer()
