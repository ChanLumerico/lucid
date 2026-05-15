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
    nonnegative,
    positive,
)
from lucid.distributions.distribution import Distribution


class Pareto(Distribution):
    r"""Pareto distribution — a heavy-tailed power-law family on :math:`[x_m, \infty)`.

    ``Pareto(scale=x_m, alpha=α)`` models quantities that follow a power-law
    tail.  The 80/20 rule (Pareto principle) is a heuristic consequence of
    this distribution with :math:`\alpha \approx 1.16`.

    Parameters
    ----------
    scale : Tensor | float
        Scale parameter :math:`x_m > 0` — the minimum possible value of
        the random variable (the lower bound of the support).
    alpha : Tensor | float
        Shape (tail-index) parameter :math:`\alpha > 0`.  Smaller values
        produce heavier tails.  The mean exists only for :math:`\alpha > 1`;
        the variance only for :math:`\alpha > 2`.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    scale : Tensor
        Minimum value :math:`x_m`.
    alpha : Tensor
        Tail index :math:`\alpha`.

    Notes
    -----
    **PDF**:

    .. math::

        p(x; x_m, \alpha) = \frac{\alpha \, x_m^\alpha}{x^{\alpha+1}},
        \quad x \geq x_m

    **Log-PDF**:

    .. math::

        \log p(x) = \log\alpha + \alpha\log x_m - (\alpha+1)\log x

    **Moments** (when defined):

    - Mean (:math:`\alpha > 1`): :math:`E[X] = \alpha x_m / (\alpha - 1)`
    - Variance (:math:`\alpha > 2`):
      :math:`\operatorname{Var}[X] = x_m^2 \alpha / ((\alpha-1)^2 (\alpha-2))`

    **Entropy**:

    .. math::

        H[X] = \log(x_m / \alpha) + 1/\alpha + 1

    **Reparameterised sampling** uses the inverse-CDF trick:
    if :math:`U \sim \operatorname{Uniform}(0, 1)` then
    :math:`X = x_m (1-U)^{-1/\alpha} \sim \operatorname{Pareto}(x_m, \alpha)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Pareto
    >>> dist = Pareto(scale=1.0, alpha=2.0)
    >>> samples = dist.rsample((200,))
    >>> samples.min()  # always >= scale
    """

    arg_constraints = {"scale": positive, "alpha": positive}
    has_rsample = True

    def __init__(
        self,
        scale: Tensor | float,
        alpha: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Pareto distribution.

        Parameters
        ----------
        scale : Tensor | float
            Scale parameter :math:`x_m > 0` — the minimum value of the
            support (lower bound of the distribution).
        alpha : Tensor | float
            Shape (tail-index) parameter :math:`\alpha > 0`.  Smaller
            values produce heavier tails.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
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
        r"""Support of the Pareto distribution: positive reals :math:`(0, \infty)`.

        The true support is :math:`[x_m, \infty)` per element, but since the
        lower bound varies across the batch a bare ``positive`` constraint is
        returned.

        Returns
        -------
        Constraint
            The ``positive`` constraint.
        """
        # Lower bound varies per element with ``scale``; report a bare
        # positive constraint and rely on the user to know the family.
        return positive

    @property
    def mean(self) -> Tensor:
        r"""Mean of the Pareto distribution: :math:`E[X] = \alpha x_m / (\alpha - 1)`.

        Defined only for :math:`\alpha > 1`; returns ``inf`` or ``nan``
        for :math:`\alpha \leq 1`.

        Returns
        -------
        Tensor
            Mean values of shape ``batch_shape``.
        """
        # Defined only for α > 1.
        return self.alpha * self.scale / (self.alpha - 1.0)

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Pareto distribution.

        :math:`\operatorname{Var}[X] = x_m^2 \alpha / ((\alpha-1)^2 (\alpha-2))`

        Defined only for :math:`\alpha > 2`.

        Returns
        -------
        Tensor
            Variance values of shape ``batch_shape``.
        """
        # Defined only for α > 2.
        a: Tensor = self.alpha
        return self.scale * self.scale * a / ((a - 1.0) ** 2 * (a - 2.0))

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples via the inverse-CDF trick.

        Uses :math:`X = x_m (1 - U)^{-1/\alpha}` where
        :math:`U \sim \operatorname{Uniform}(0, 1)`.  Gradients propagate
        through both ``scale`` and ``alpha``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples :math:`\geq x_m` of shape ``(*sample_shape, *batch_shape)``.
        """
        # icdf-trick: x = scale · (1 − U)^(−1/α).
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(*shape, dtype=self.scale.dtype, device=self.scale.device)
        return self.scale * (1.0 - u) ** (-1.0 / self.alpha)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Pareto distribution.

        .. math::

            \log p(x) = \log\alpha + \alpha\log x_m - (\alpha+1)\log x

        Parameters
        ----------
        value : Tensor
            Points :math:`x \geq x_m` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density values of the same shape as ``value``.
        """
        return (
            self.alpha.log()
            + self.alpha * self.scale.log()
            - (self.alpha + 1.0) * value.log()
        )

    def entropy(self) -> Tensor:
        r"""Entropy of the Pareto distribution.

        .. math::

            H[X] = \log(x_m / \alpha) + 1/\alpha + 1

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).
        """
        return (self.scale / self.alpha).log() + 1.0 / self.alpha + 1.0


class Weibull(Distribution):
    r"""Weibull distribution — a flexible family for lifetime and survival analysis.

    ``Weibull(scale=λ, concentration=k)`` generalises both the Exponential
    (:math:`k = 1`) and Rayleigh (:math:`k = 2`) distributions.  It is widely
    used for modelling time-to-failure data because the hazard rate
    :math:`h(t) = (k/\lambda)(t/\lambda)^{k-1}` can increase, be constant,
    or decrease depending on :math:`k`.

    Parameters
    ----------
    scale : Tensor | float
        Scale parameter :math:`\lambda > 0` — a characteristic life value
        at which the CDF is :math:`1 - e^{-1} \approx 63.2\%`.
    concentration : Tensor | float
        Shape parameter :math:`k > 0`.  Values :math:`k < 1` give a
        decreasing hazard (infant mortality); :math:`k = 1` is constant
        (memoryless Exponential); :math:`k > 1` gives an increasing hazard
        (wear-out).
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    scale : Tensor
        Scale parameter :math:`\lambda`.
    concentration : Tensor
        Shape parameter :math:`k`.

    Notes
    -----
    **PDF**:

    .. math::

        p(x; \lambda, k) = \frac{k}{\lambda}
        \left(\frac{x}{\lambda}\right)^{k-1}
        \exp\!\left(-\left(\frac{x}{\lambda}\right)^k\right),
        \quad x \geq 0

    **Log-PDF**:

    .. math::

        \log p(x) = \log k - k\log\lambda + (k-1)\log x
                   - (x/\lambda)^k

    **Moments**:

    - Mean: :math:`E[X] = \lambda \,\Gamma(1 + 1/k)`
    - Variance: :math:`\operatorname{Var}[X] = \lambda^2
      \left[\Gamma(1+2/k) - \Gamma(1+1/k)^2\right]`

    **Entropy**:

    .. math::

        H[X] = \gamma\left(1 - \tfrac{1}{k}\right) + \log(\lambda/k) + 1

    where :math:`\gamma \approx 0.5772` is the Euler–Mascheroni constant.

    **Reparameterised sampling** uses the inverse-CDF:
    :math:`X = \lambda(-\log(1-U))^{1/k}` for :math:`U \sim \operatorname{Uniform}(0,1)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Weibull
    >>> # Exponential(rate=1) as a special case
    >>> dist_exp = Weibull(scale=1.0, concentration=1.0)
    >>> dist = Weibull(scale=2.0, concentration=1.5)
    >>> samples = dist.rsample((500,))
    """

    arg_constraints = {"scale": positive, "concentration": positive}
    has_rsample = True

    def __init__(
        self,
        scale: Tensor | float,
        concentration: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a Weibull distribution.

        Parameters
        ----------
        scale : Tensor | float
            Scale parameter :math:`\lambda > 0`.
        concentration : Tensor | float
            Shape parameter :math:`k > 0`.  Values :math:`k < 1` give a
            decreasing hazard; :math:`k = 1` gives the Exponential;
            :math:`k > 1` gives an increasing hazard.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
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
        r"""Support of the Weibull distribution: :math:`[0, \infty)`.

        Returns
        -------
        Constraint
            The ``nonnegative`` constraint.
        """
        return nonnegative

    @property
    def mean(self) -> Tensor:
        r"""Mean of the Weibull distribution: :math:`E[X] = \lambda \Gamma(1 + 1/k)`.

        Returns
        -------
        Tensor
            Mean values of shape ``batch_shape``.
        """
        # μ = λ · Γ(1 + 1/k).
        return self.scale * lucid.lgamma(1.0 + 1.0 / self.concentration).exp()

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Weibull distribution.

        :math:`\operatorname{Var}[X] = \lambda^2 [\Gamma(1+2/k) - \Gamma(1+1/k)^2]`

        Returns
        -------
        Tensor
            Variance values of shape ``batch_shape``.
        """
        # Var = λ² · [Γ(1 + 2/k) − Γ(1 + 1/k)²].
        k_inv: Tensor = 1.0 / self.concentration
        g1: Tensor = lucid.lgamma(1.0 + k_inv).exp()
        g2: Tensor = lucid.lgamma(1.0 + 2.0 * k_inv).exp()
        return self.scale * self.scale * (g2 - g1 * g1)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples via the inverse-CDF.

        Uses :math:`X = \lambda (-\log(1-U))^{1/k}` for
        :math:`U \sim \operatorname{Uniform}(0, 1)`.  Gradients propagate
        through both ``scale`` and ``concentration``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative samples of shape ``(*sample_shape, *batch_shape)``.
        """
        # icdf: x = λ · (−log(1 − U))^(1/k).
        shape = self._extended_shape(sample_shape)
        u: Tensor = lucid.rand(*shape, dtype=self.scale.dtype, device=self.scale.device)
        return self.scale * (-(1.0 - u).log()) ** (1.0 / self.concentration)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the Weibull distribution.

        .. math::

            \log p(x) = \log k - k\log\lambda + (k-1)\log x - (x/\lambda)^k

        Parameters
        ----------
        value : Tensor
            Non-negative points :math:`x \geq 0` at which to evaluate.

        Returns
        -------
        Tensor
            Log-density values of the same shape as ``value``.
        """
        k: Tensor = self.concentration
        lam: Tensor = self.scale
        return k.log() - k * lam.log() + (k - 1.0) * value.log() - (value / lam) ** k

    def entropy(self) -> Tensor:
        r"""Entropy of the Weibull distribution.

        .. math::

            H[X] = \gamma(1 - 1/k) + \log(\lambda/k) + 1

        where :math:`\gamma \approx 0.5772` is the Euler-Mascheroni constant.

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).
        """
        # H = γ·(1 − 1/k) + log(λ/k) + 1, where γ is Euler-Mascheroni.
        EULER_GAMMA: float = 0.5772156649015329
        return (
            EULER_GAMMA * (1.0 - 1.0 / self.concentration)
            + (self.scale / self.concentration).log()
            + 1.0
        )


class HalfNormal(Distribution):
    r"""Half-Normal distribution — the absolute value of a zero-mean Normal.

    If :math:`X \sim \mathcal{N}(0, \sigma^2)` then
    :math:`|X| \sim \operatorname{HalfNormal}(\sigma)`.
    The distribution is supported on :math:`[0, \infty)` and arises
    naturally as a scale prior in hierarchical Bayesian models (it is
    weakly informative while keeping probability mass away from zero).

    Parameters
    ----------
    scale : Tensor | float
        Scale parameter :math:`\sigma > 0` — the standard deviation of
        the underlying zero-mean Normal.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    scale : Tensor
        Scale parameter :math:`\sigma`.

    Notes
    -----
    **PDF**:

    .. math::

        p(x; \sigma) = \frac{\sqrt{2}}{\sigma\sqrt{\pi}}
        \exp\!\left(-\frac{x^2}{2\sigma^2}\right),
        \quad x \geq 0

    **Log-PDF**:

    .. math::

        \log p(x) = \log 2 + \log\mathcal{N}(x; 0, \sigma^2)

    where the right-hand side is the log-density of the full Normal evaluated
    at :math:`x`.

    **Moments**:

    - Mean: :math:`E[X] = \sigma\sqrt{2/\pi}`
    - Variance: :math:`\operatorname{Var}[X] = \sigma^2(1 - 2/\pi)`

    **Entropy**:

    .. math::

        H[X] = \tfrac{1}{2}\log\!\left(\frac{\pi e \sigma^2}{2}\right)

    **Reparameterised sampling** takes ``abs`` of a Normal sample, so
    gradients flow through :math:`\sigma` unobstructed.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import HalfNormal
    >>> dist = HalfNormal(scale=1.0)
    >>> samples = dist.rsample((300,))
    >>> (samples >= 0.0).all()
    """

    arg_constraints = {"scale": positive}
    support: Constraint | None = nonnegative
    has_rsample = True

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a HalfNormal distribution.

        Parameters
        ----------
        scale : Tensor | float
            Scale parameter :math:`\sigma > 0` — the standard deviation of
            the underlying zero-mean Normal distribution.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
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
        r"""Mean of the HalfNormal distribution: :math:`E[X] = \sigma\sqrt{2/\pi}`.

        Returns
        -------
        Tensor
            Mean values of shape ``batch_shape``.
        """
        return self.scale * math.sqrt(2.0 / math.pi)

    @property
    def variance(self) -> Tensor:
        r"""Variance of the HalfNormal distribution: :math:`\operatorname{Var}[X] = \sigma^2(1 - 2/\pi)`.

        Returns
        -------
        Tensor
            Variance values of shape ``batch_shape``.
        """
        return self.scale * self.scale * (1.0 - 2.0 / math.pi)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw reparameterised samples by folding a Normal sample.

        Computes ``abs`` of a :class:`Normal` sample, so gradients propagate
        through :math:`\sigma` unobstructed.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative samples of shape ``(*sample_shape, *batch_shape)``.
        """
        return self._base.rsample(sample_shape).abs()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the HalfNormal distribution.

        .. math::

            \log p(x) = \log 2 + \log\mathcal{N}(x; 0, \sigma^2)

        Parameters
        ----------
        value : Tensor
            Non-negative points :math:`x \geq 0` at which to evaluate.

        Returns
        -------
        Tensor
            Log-density values of the same shape as ``value``.
        """
        # log p(x) = log 2 + log_normal(x | 0, σ).
        return math.log(2.0) + self._base.log_prob(value)

    def entropy(self) -> Tensor:
        r"""Entropy of the HalfNormal distribution.

        .. math::

            H[X] = \tfrac{1}{2}\log\!\left(\frac{\pi e \sigma^2}{2}\right)

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).
        """
        # H = 0.5·log(π·e·σ²/2).
        return 0.5 * (math.pi * math.e / 2.0 * self.scale * self.scale).log()


class HalfCauchy(Distribution):
    r"""Half-Cauchy distribution — the absolute value of a zero-location Cauchy.

    If :math:`X \sim \operatorname{Cauchy}(0, s)` then
    :math:`|X| \sim \operatorname{HalfCauchy}(s)`.
    The distribution is supported on :math:`[0, \infty)` and has **no finite
    moments** (mean and variance are undefined, like the full Cauchy).

    It is a popular weakly-informative prior for scale parameters in Bayesian
    hierarchical models — heavy tails allow occasional large scales while the
    mode at zero permits near-zero scales.

    Parameters
    ----------
    scale : Tensor | float
        Scale parameter :math:`s > 0`.  This is the half-width at half-maximum
        of the Cauchy density folded onto :math:`[0, \infty)`.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    scale : Tensor
        Scale parameter :math:`s`.

    Notes
    -----
    **PDF**:

    .. math::

        p(x; s) = \frac{2}{\pi s \left(1 + (x/s)^2\right)},
        \quad x \geq 0

    **Log-PDF**:

    .. math::

        \log p(x) = \log 2 + \log\operatorname{Cauchy}(x; 0, s)

    where the right-hand side is the log-density of the full Cauchy evaluated
    at :math:`x`.

    Neither the mean nor the variance exists because the Cauchy distribution
    lacks finite moments of any positive order.

    **Reparameterised sampling** folds a Cauchy sample via ``abs``, so gradients
    propagate through the scale parameter.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import HalfCauchy
    >>> dist = HalfCauchy(scale=1.0)
    >>> samples = dist.rsample((200,))
    >>> (samples >= 0.0).all()
    """

    arg_constraints = {"scale": positive}
    support: Constraint | None = nonnegative
    has_rsample = True

    def __init__(
        self,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Initialise a HalfCauchy distribution.

        Parameters
        ----------
        scale : Tensor | float
            Scale parameter :math:`s > 0`.  The half-width at half-maximum
            of the Cauchy density folded onto :math:`[0, \infty)`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
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
        """Draw reparameterised samples by folding a Cauchy sample.

        Computes ``abs`` of a zero-location :class:`Cauchy` sample, so
        gradients propagate through the scale parameter.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Non-negative samples of shape ``(*sample_shape, *batch_shape)``.
        """
        return self._base.rsample(sample_shape).abs()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the HalfCauchy distribution.

        .. math::

            \log p(x) = \log 2 + \log\operatorname{Cauchy}(x; 0, s)

        Parameters
        ----------
        value : Tensor
            Non-negative points :math:`x \geq 0` at which to evaluate.

        Returns
        -------
        Tensor
            Log-density values of the same shape as ``value``.
        """
        return math.log(2.0) + self._base.log_prob(value)


class FisherSnedecor(Distribution):
    r"""Fisher-Snedecor F-distribution — ratio of two scaled Chi-squared variates.

    ``FisherSnedecor(df1=d1, df2=d2)`` is the distribution of the statistic

    .. math::

        F = \frac{X / d_1}{Y / d_2}

    where :math:`X \sim \chi^2(d_1)` and :math:`Y \sim \chi^2(d_2)` are
    independent.  The F-distribution is foundational in classical statistics:
    it arises in ANOVA F-tests and in the likelihood-ratio test for comparing
    nested linear models.

    Parameters
    ----------
    df1 : Tensor | float
        Numerator degrees of freedom :math:`d_1 > 0`.
    df2 : Tensor | float
        Denominator degrees of freedom :math:`d_2 > 0`.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    df1 : Tensor
        Numerator degrees of freedom :math:`d_1`.
    df2 : Tensor
        Denominator degrees of freedom :math:`d_2`.

    Notes
    -----
    **PDF**:

    .. math::

        p(x; d_1, d_2) = \frac{1}{x \, B(d_1/2,\, d_2/2)}
        \left(\frac{d_1 x}{d_1 x + d_2}\right)^{d_1/2}
        \left(\frac{d_2}{d_1 x + d_2}\right)^{d_2/2},
        \quad x > 0

    **Log-PDF** (stable form using lgamma):

    .. math::

        \log p(x) = \frac{d_1}{2}\log(d_1 x)
                   + \frac{d_2}{2}\log d_2
                   - \frac{d_1+d_2}{2}\log(d_1 x + d_2)
                   - \log x - \log B(d_1/2, d_2/2)

    **Moments** (when defined):

    - Mean (:math:`d_2 > 2`): :math:`E[X] = d_2 / (d_2 - 2)`
    - Variance (:math:`d_2 > 4`):
      :math:`\operatorname{Var}[X] = 2 d_2^2 (d_1 + d_2 - 2) /
      [d_1 (d_2 - 2)^2 (d_2 - 4)]`

    **Sampling** draws one :math:`\chi^2(d_1)` and one :math:`\chi^2(d_2)`
    sample and forms the ratio.  Both :math:`\chi^2` samples are drawn via
    the Gamma sampler.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import FisherSnedecor
    >>> dist = FisherSnedecor(df1=5.0, df2=10.0)
    >>> samples = dist.sample((100,))
    >>> dist.mean  # d2/(d2-2) = 10/8 = 1.25
    """

    arg_constraints = {"df1": positive, "df2": positive}
    support: Constraint | None = positive

    def __init__(
        self,
        df1: Tensor | float,
        df2: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise a Fisher-Snedecor F-distribution.

        Parameters
        ----------
        df1 : Tensor | float
            Numerator degrees of freedom :math:`d_1 > 0`.
        df2 : Tensor | float
            Denominator degrees of freedom :math:`d_2 > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.
        """
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
        """Mean of the F-distribution: :math:`E[X] = d_2 / (d_2 - 2)`.

        Defined only for :math:`d_2 > 2`; returns ``inf`` or ``nan`` otherwise.

        Returns
        -------
        Tensor
            Mean values of shape ``batch_shape``.
        """
        # Defined for d2 > 2.
        return self.df2 / (self.df2 - 2.0)

    @property
    def variance(self) -> Tensor:
        r"""Variance of the F-distribution.

        :math:`\operatorname{Var}[X] = 2 d_2^2 (d_1 + d_2 - 2) /
        [d_1 (d_2 - 2)^2 (d_2 - 4)]`

        Defined only for :math:`d_2 > 4`.

        Returns
        -------
        Tensor
            Variance values of shape ``batch_shape``.
        """
        # Defined for d2 > 4.
        d1, d2 = self.df1, self.df2
        return 2.0 * d2 * d2 * (d1 + d2 - 2.0) / (d1 * (d2 - 2.0) ** 2 * (d2 - 4.0))

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the F-distribution.

        Forms the ratio :math:`F = (X/d_1) / (Y/d_2)` where
        :math:`X \sim \chi^2(d_1)` and :math:`Y \sim \chi^2(d_2)` are
        independent samples drawn via the Gamma sampler.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Positive samples of shape ``(*sample_shape, *batch_shape)``.
        """
        x: Tensor = self._chi1.sample(sample_shape)
        y: Tensor = self._chi2.sample(sample_shape)
        return (x / self.df1) / (y / self.df2)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability density of the F-distribution.

        .. math::

            \log p(x) = \frac{d_1}{2}\log(d_1 x)
                       + \frac{d_2}{2}\log d_2
                       - \frac{d_1+d_2}{2}\log(d_1 x + d_2)
                       - \log x - \log B(d_1/2, d_2/2)

        Parameters
        ----------
        value : Tensor
            Positive points :math:`x > 0` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density values of the same shape as ``value``.
        """
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
