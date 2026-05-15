"""Discrete count distributions: ``Poisson``, ``Binomial``,
``NegativeBinomial``.

All three are pure-Lucid composites — sampling reuses
``lucid.poisson`` and the per-distribution helpers, log-prob is
closed-form via lgamma.
"""

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
    r"""Poisson distribution over the non-negative integers.

    Models the number of events occurring in a fixed interval when events
    happen independently at a constant mean rate :math:`\lambda`.  The
    Poisson distribution is the limiting case of a Binomial as
    :math:`n \to \infty` and :math:`p \to 0` with :math:`np = \lambda`
    fixed.

    Parameters
    ----------
    rate : Tensor | float
        Mean rate :math:`\lambda > 0`.  Determines both the mean and the
        variance of the distribution.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    rate : Tensor
        The mean rate parameter :math:`\lambda`.

    Notes
    -----
    **PMF** (probability mass function):

    .. math::

        P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!},
        \quad k \in \{0, 1, 2, \ldots\}

    **Moments**:

    - Mean: :math:`E[X] = \lambda`
    - Variance: :math:`\operatorname{Var}[X] = \lambda`
    - Mode: :math:`\lfloor \lambda \rfloor`

    The log-probability is computed as
    :math:`k \log \lambda - \lambda - \log\Gamma(k+1)`, which is
    numerically stable via the ``lgamma`` function.

    The Poisson has no closed-form entropy (it is an infinite sum over all
    non-negative integers), so :meth:`entropy` raises ``NotImplementedError``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Poisson
    >>> dist = Poisson(rate=3.5)
    >>> samples = dist.sample((100,))
    >>> samples.shape
    (100,)
    >>> # log-probability at k=4
    >>> dist.log_prob(lucid.tensor(4.0))
    """

    arg_constraints = {"rate": positive}
    support: Constraint | None = nonnegative_integer

    def __init__(
        self,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Poisson distribution.

        Parameters
        ----------
        rate : Tensor | float
            Mean rate parameter :math:`\lambda > 0`.  Determines both the
            mean and the variance.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
        self.rate = _as_tensor(rate)
        super().__init__(
            batch_shape=tuple(self.rate.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        r"""Mean of the Poisson distribution: :math:`E[X] = \lambda`.

        Returns
        -------
        Tensor
            Mean values equal to ``rate``, shape ``batch_shape``.
        """
        return self.rate

    @property
    def mode(self) -> Tensor:
        r"""Mode of the Poisson distribution: :math:`\lfloor \lambda \rfloor`.

        When :math:`\lambda` is an integer, both :math:`\lambda` and
        :math:`\lambda - 1` are modes; this property returns
        :math:`\lfloor \lambda \rfloor`.

        Returns
        -------
        Tensor
            Mode values of shape ``batch_shape``.
        """
        # ``floor(rate)`` — when rate is integer, both floor and floor−1
        # are modes; we follow the reference framework and take floor.
        return self.rate.floor()

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Poisson distribution: :math:`\operatorname{Var}[X] = \lambda`.

        Returns
        -------
        Tensor
            Variance values equal to ``rate``, shape ``batch_shape``.
        """
        return self.rate

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Poisson distribution.

        Delegates to :func:`lucid.poisson`, which uses the Knuth exact
        algorithm for small :math:`\lambda` and a Normal approximation
        with rounding for large :math:`\lambda`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative integer samples of shape
            ``(*sample_shape, *batch_shape)``.
        """
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
        r"""Log-probability of the given counts under the Poisson distribution.

        .. math::

            \log P(X = k) = k \log \lambda - \lambda - \log\Gamma(k + 1)

        Parameters
        ----------
        value : Tensor
            Non-negative integer counts :math:`k \geq 0`.

        Returns
        -------
        Tensor
            Log-probability values of the same shape as ``value``.
        """
        # log p(k | λ) = k·log(λ) − λ − lgamma(k+1).
        return value * self.rate.log() - self.rate - lucid.lgamma(value + 1.0)

    def entropy(self) -> Tensor:
        """Entropy of the Poisson distribution (not available in closed form).

        The Poisson entropy is an infinite sum over all non-negative integers
        and has no simple closed form.

        Raises
        ------
        NotImplementedError
            Always — use numerical approximation if needed.
        """
        # Closed-form entropy is an infinite sum; the reference framework
        # returns NotImplemented.  We follow.
        raise NotImplementedError("Poisson.entropy: no closed form available")


class Binomial(Distribution):
    r"""Binomial distribution over the number of successes in n independent trials.

    ``Binomial(total_count=n, probs=p)`` models the count of successes when
    each of :math:`n` i.i.d. Bernoulli trials has success probability
    :math:`p`.  Parameterisation is via either ``probs`` (in :math:`[0, 1]`)
    or ``logits`` (the log-odds :math:`\log(p/(1-p)) \in \mathbb{R}`);
    exactly one must be supplied.

    Parameters
    ----------
    total_count : Tensor | int, optional
        Number of trials :math:`n \geq 0`.  Default is ``1`` (reduces to
        Bernoulli).
    probs : Tensor | float | None, optional
        Success probability :math:`p \in [0, 1]`.  Mutually exclusive with
        ``logits``.
    logits : Tensor | float | None, optional
        Log-odds :math:`l = \log(p / (1-p)) \in \mathbb{R}`.  Mutually
        exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    total_count : Tensor
        Number of trials :math:`n`.
    probs : Tensor
        Success probability (present when constructed with ``probs``).
    logits : Tensor
        Log-odds (present when constructed with ``logits``).

    Notes
    -----
    **PMF**:

    .. math::

        P(X = k) = \binom{n}{k} p^k (1-p)^{n-k},
        \quad k \in \{0, 1, \ldots, n\}

    **Log-PMF** via logits (numerically stable form):

    .. math::

        \log P(X = k) = \log\binom{n}{k} + k \, l - n \log(1 + e^l)

    where :math:`l = \operatorname{logit}(p)` and the binomial coefficient is
    evaluated via :math:`\log \Gamma(n+1) - \log\Gamma(k+1) - \log\Gamma(n-k+1)`.

    **Moments**:

    - Mean: :math:`E[X] = np`
    - Variance: :math:`\operatorname{Var}[X] = np(1-p)`

    **Sampling strategy**: for :math:`n \leq 25` Bernoulli draws are summed
    exactly along a dedicated axis.  For larger :math:`n` a Normal
    approximation is used with the result rounded and clamped to
    :math:`[0, n]`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Binomial
    >>> dist = Binomial(total_count=10, probs=0.3)
    >>> samples = dist.sample((50,))
    >>> samples.shape
    (50,)
    >>> # PMF at k=3
    >>> dist.log_prob(lucid.tensor(3.0)).exp()
    """

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
        r"""Initialise a Binomial distribution.

        Parameters
        ----------
        total_count : Tensor | int, optional
            Number of trials :math:`n \geq 0`.  Default is ``1`` (reduces to
            Bernoulli).
        probs : Tensor | float | None, optional
            Success probability :math:`p \in [0, 1]`.  Mutually exclusive with
            ``logits``.
        logits : Tensor | float | None, optional
            Log-odds :math:`l = \log(p / (1-p)) \in \mathbb{R}`.  Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        if (probs is None) == (logits is None):
            raise ValueError("Binomial: pass exactly one of `probs` or `logits`.")
        self.total_count: Tensor = _as_tensor(total_count)
        if probs is not None:
            self.probs = _as_tensor(probs)
            self._is_logits = False
            shape = tuple(self.probs.shape)
        else:
            self.logits = _as_tensor(logits)  # type: ignore[arg-type]
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
        r"""Support of the Binomial distribution: non-negative integers.

        Although the strict support is :math:`\{0, 1, \ldots, n\}` per element,
        this property returns ``nonnegative_integer`` because ``total_count``
        may differ across the batch.

        Returns
        -------
        Constraint
            The ``nonnegative_integer`` constraint.
        """
        # Per-element {0, 1, …, total_count}; we report nonneg-integer
        # since ``total_count`` may differ across the batch.
        return nonnegative_integer

    @property
    def _probs(self) -> Tensor:
        """Success probability :math:`p`, computed from logits if necessary.

        Returns
        -------
        Tensor
            Probability values in :math:`[0, 1]`, shape ``batch_shape``.
        """
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @property
    def _logits(self) -> Tensor:
        r"""Log-odds :math:`\log(p/(1-p))`, computed from probs if necessary.

        Returns
        -------
        Tensor
            Log-odds values in :math:`\mathbb{R}`, shape ``batch_shape``.
        """
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @property
    def mean(self) -> Tensor:
        """Mean of the Binomial distribution: :math:`E[X] = np`.

        Returns
        -------
        Tensor
            Mean values of shape ``batch_shape``.
        """
        return self.total_count * self._probs

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Binomial distribution: :math:`\operatorname{Var}[X] = np(1-p)`.

        Returns
        -------
        Tensor
            Variance values of shape ``batch_shape``.
        """
        p: Tensor = self._probs
        return self.total_count * p * (1.0 - p)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Binomial distribution.

        Uses an exact strategy for small :math:`n` (sum of Bernoulli draws)
        and a Normal approximation with rounding for large :math:`n`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative integer samples in :math:`\{0, 1, \ldots, n\}`,
            shape ``(*sample_shape, *batch_shape)``.
        """
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
            i_flat: Tensor = lucid.arange(0, max_n, 1, dtype=dtype, device=device)
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
        r"""Log-probability of the given counts under the Binomial distribution.

        Computed via logits for numerical stability:

        .. math::

            \log P(X = k) = \log\binom{n}{k} + k \, l - n \log(1 + e^l)

        where :math:`l = \operatorname{logit}(p)` and the binomial coefficient
        is evaluated via :math:`\log\Gamma`.

        Parameters
        ----------
        value : Tensor
            Count values :math:`k \in \{0, 1, \ldots, n\}`.

        Returns
        -------
        Tensor
            Log-probability values of the same shape as ``value``.
        """
        # log p(k) = lgamma(n+1) − lgamma(k+1) − lgamma(n−k+1)
        #           + k·log(p) + (n−k)·log(1−p).
        n: Tensor = self.total_count
        k: Tensor = value
        log_comb: Tensor = (
            lucid.lgamma(n + 1.0) - lucid.lgamma(k + 1.0) - lucid.lgamma(n - k + 1.0)
        )
        # Stable form via logits: k·l − n·softplus(l).
        l: Tensor = self._logits
        return log_comb + k * l - n * (1.0 + l.exp()).log()


class NegativeBinomial(Distribution):
    r"""Negative Binomial distribution over the number of failures before r successes.

    ``NegativeBinomial(total_count=r, probs=p)`` models the count of failures
    :math:`k` before achieving :math:`r` successes in a sequence of independent
    Bernoulli trials each with success probability :math:`1 - p`.

    Equivalently, this is a **Gamma-Poisson compound**: draw
    :math:`\lambda \sim \operatorname{Gamma}(r, (1-p)/p)`, then
    :math:`X \sim \operatorname{Poisson}(\lambda)`.  This interpretation
    extends the distribution to real-valued ``total_count`` (the
    *generalised* Negative Binomial) and forms the basis of the sampler.

    Parameters
    ----------
    total_count : Tensor | float
        Dispersion / number of successes :math:`r > 0`.  May be non-integer
        for the generalised version.
    probs : Tensor | float | None, optional
        Probability of failure :math:`p \in (0, 1)`.  Mutually exclusive
        with ``logits``.
    logits : Tensor | float | None, optional
        Log-odds of failure :math:`l = \log(p / (1-p)) \in \mathbb{R}`.
        Mutually exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    total_count : Tensor
        Dispersion parameter :math:`r`.
    probs : Tensor
        Failure probability (present when constructed with ``probs``).
    logits : Tensor
        Log-odds of failure (present when constructed with ``logits``).

    Notes
    -----
    **PMF**:

    .. math::

        P(X = k) = \binom{k + r - 1}{k} (1-p)^r p^k,
        \quad k \in \{0, 1, 2, \ldots\}

    **Log-PMF** (via ``lgamma`` for numerical stability):

    .. math::

        \log P(X = k) = \log\Gamma(k+r) - \log\Gamma(r) - \log\Gamma(k+1)
                       + r \log(1-p) + k \log p

    **Moments**:

    - Mean: :math:`E[X] = r p / (1-p)`
    - Variance: :math:`\operatorname{Var}[X] = r p / (1-p)^2`

    **Sampler**: uses the Gamma-Poisson representation.  A standard-Gamma
    variate with concentration :math:`r` is drawn via rejection sampling,
    scaled by :math:`p/(1-p)`, and passed to the Poisson sampler.  This
    produces exact (non-approximate) samples for all real :math:`r > 0`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import NegativeBinomial
    >>> dist = NegativeBinomial(total_count=5.0, probs=0.4)
    >>> samples = dist.sample((100,))
    >>> samples.shape
    (100,)
    >>> dist.mean
    """

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
        r"""Initialise a Negative Binomial distribution.

        Parameters
        ----------
        total_count : Tensor | float
            Dispersion / number of successes :math:`r > 0`.  May be
            non-integer for the generalised version.
        probs : Tensor | float | None, optional
            Probability of failure :math:`p \in (0, 1)`.  Mutually exclusive
            with ``logits``.
        logits : Tensor | float | None, optional
            Log-odds of failure :math:`l = \log(p / (1-p))`.  Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
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
            self.logits = _as_tensor(logits)  # type: ignore[arg-type]
            self._is_logits = True
            shape = tuple(self.logits.shape)
        super().__init__(
            batch_shape=shape,
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def _probs(self) -> Tensor:
        """Failure probability :math:`p`, computed from logits if necessary.

        Returns
        -------
        Tensor
            Failure probability values in :math:`(0, 1)`, shape ``batch_shape``.
        """
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    @property
    def _logits(self) -> Tensor:
        r"""Log-odds of failure :math:`\log(p/(1-p))`, computed from probs if necessary.

        Returns
        -------
        Tensor
            Log-odds values in :math:`\mathbb{R}`, shape ``batch_shape``.
        """
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @property
    def mean(self) -> Tensor:
        """Mean of the Negative Binomial: :math:`E[X] = r p / (1-p)`.

        Returns
        -------
        Tensor
            Mean values of shape ``batch_shape``.
        """
        # E[X] = r · p / (1 − p).
        p: Tensor = self._probs
        return self.total_count * p / (1.0 - p)

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Negative Binomial: :math:`\operatorname{Var}[X] = r p / (1-p)^2`.

        Returns
        -------
        Tensor
            Variance values of shape ``batch_shape``.
        """
        # Var = r · p / (1 − p)².
        p: Tensor = self._probs
        return self.total_count * p / ((1.0 - p) ** 2)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples via the Gamma-Poisson compound representation.

        Draws :math:`\lambda \sim \operatorname{Gamma}(r, (1-p)/p)` then
        :math:`X \sim \operatorname{Poisson}(\lambda)`.  This produces exact
        (non-approximate) samples for all real :math:`r > 0`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative integer samples of shape
            ``(*sample_shape, *batch_shape)``.
        """
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
        r"""Log-probability of counts under the Negative Binomial distribution.

        .. math::

            \log P(X = k) = \log\Gamma(k+r) - \log\Gamma(r) - \log\Gamma(k+1)
                           + r \log(1-p) + k \log p

        Parameters
        ----------
        value : Tensor
            Non-negative integer counts :math:`k \geq 0`.

        Returns
        -------
        Tensor
            Log-probability values of the same shape as ``value``.
        """
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
