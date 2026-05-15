"""``Exponential``, ``Laplace``, and ``Cauchy`` — icdf-sampled families."""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    positive,
    real,
)
from lucid.distributions.distribution import Distribution, ExponentialFamily


from lucid.distributions._util import (
    as_tensor as _as_tensor,
    broadcast_pair as _broadcast_pair,
)


class Exponential(ExponentialFamily):
    r"""Exponential distribution on :math:`[0, \infty)`.

    Continuous distribution describing the waiting time between events of
    a Poisson process with rate :math:`\lambda`.  It is the continuous
    analogue of the :class:`~lucid.distributions.Geometric` distribution
    and the unique continuous distribution with the *memoryless* property.

    Parameters
    ----------
    rate : Tensor or float
        Rate parameter :math:`\lambda > 0`.  The mean of the distribution
        is :math:`1/\lambda`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density on :math:`x \geq 0`:

    .. math::

        p(x; \lambda) = \lambda e^{-\lambda x}

    Moments:

    .. math::

        \mathbb{E}[X] = \frac{1}{\lambda}, \qquad
        \mathrm{Var}[X] = \frac{1}{\lambda^2}, \qquad
        H[X] = 1 - \log \lambda

    **Memoryless property:**

    .. math::

        P(X > s + t \mid X > s) = P(X > t)

    Special cases / relations:

    * :math:`\mathrm{Exponential}(\lambda) = \mathrm{Gamma}(1, \lambda)`
    * The sum of :math:`k` IID :math:`\mathrm{Exponential}(\lambda)`
      variables is :math:`\mathrm{Gamma}(k, \lambda)` (Erlang).
    * :math:`-\log U / \lambda` with :math:`U \sim \mathrm{Uniform}(0, 1)`
      is the inverse-CDF sampler used by :meth:`rsample`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Exponential
    >>> d = Exponential(rate=2.0)
    >>> d.mean  # 1/rate
    Tensor(0.5)
    >>> d.rsample((4,))
    Tensor([...])
    >>> d.log_prob(lucid.tensor(1.0))
    Tensor(-1.3069)
    """

    arg_constraints = {"rate": positive}
    support: Constraint | None = positive
    has_rsample = True

    def __init__(
        self,
        rate: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct an Exponential distribution.

        Parameters
        ----------
        rate : Tensor | float
            Rate parameter :math:`\lambda > 0`.  The mean of the
            distribution is :math:`1/\lambda`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Exponential distribution with rate :math:`\lambda` has PDF:

        .. math::

            p(x; \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0

        It is the continuous analogue of the Geometric distribution and
        describes the waiting time between events in a Poisson process.

        Examples
        --------
        >>> from lucid.distributions import Exponential
        >>> d = Exponential(rate=2.0)
        >>> d.mean  # 1/rate = 0.5
        Tensor(0.5)
        """
        self.rate = _as_tensor(rate)
        super().__init__(
            batch_shape=tuple(self.rate.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Exponential distribution.

        .. math::

            E[X] = \frac{1}{\lambda}

        Returns
        -------
        Tensor
            Mean :math:`1/\lambda`, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=4.0).mean
        Tensor(0.25)
        """
        return 1.0 / self.rate

    @property
    def mode(self) -> Tensor:
        r"""Mode of the Exponential distribution.

        The Exponential distribution is monotonically decreasing on
        :math:`[0, \infty)`, so the mode is always zero regardless of the
        rate parameter.

        .. math::

            \text{mode} = 0

        Returns
        -------
        Tensor
            Zero tensor of shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=5.0).mode
        Tensor(0.0)
        """
        return lucid.zeros_like(self.rate)

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Exponential distribution.

        .. math::

            \operatorname{Var}[X] = \frac{1}{\lambda^2}

        Returns
        -------
        Tensor
            Variance :math:`1/\lambda^2`, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=2.0).variance
        Tensor(0.25)
        """
        return 1.0 / (self.rate * self.rate)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Reparameterised sample via the inverse-CDF method.

        Uses the transform :math:`X = -\log(1 - U) / \lambda` where
        :math:`U \sim \text{Uniform}(0, 1)`.  This is an exact
        reparameterisation — gradients flow through :math:`\lambda`.

        .. math::

            X = -\frac{\log(1 - U)}{\lambda}

        Subtracting from 1 (rather than directly using :math:`U`) ensures
        the argument to :math:`\log` is bounded away from zero.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Reparameterised samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = Exponential(rate=1.0)
        >>> x = d.rsample((500,))
        >>> x.mean()  # approximately 1.0
        """
        # Inverse CDF sampling via U ~ Uniform(0, 1):  x = -log(1 - U) / rate.
        # Subtract from 1 to keep U away from 0 (which would give -inf).
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.rate.dtype, device=self.rate.device)
        return -(1.0 - u).log() / self.rate

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Exponential distribution.

        .. math::

            \log p(x; \lambda) = \log \lambda - \lambda x

        Parameters
        ----------
        value : Tensor
            Non-negative real values :math:`x \geq 0`.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> d = Exponential(rate=1.0)
        >>> d.log_prob(lucid.tensor(1.0))  # -1.0
        Tensor(-1.0)
        """
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value

    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Exponential distribution.

        .. math::

            F(x; \lambda) = 1 - e^{-\lambda x}, \quad x \geq 0

        Parameters
        ----------
        value : Tensor
            Non-negative real values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            CDF values in :math:`[0, 1]`, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=1.0).cdf(lucid.tensor(1.0))  # 1 - e^-1 ≈ 0.632
        Tensor(0.6321)
        """
        return 1.0 - (-self.rate * value).exp()

    def icdf(self, value: Tensor) -> Tensor:
        r"""Inverse CDF (quantile function) of the Exponential distribution.

        .. math::

            F^{-1}(u; \lambda) = -\frac{\log(1 - u)}{\lambda}

        Parameters
        ----------
        value : Tensor
            Probability values :math:`u \in [0, 1)`.

        Returns
        -------
        Tensor
            Quantiles in :math:`[0, \infty)`, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=1.0).icdf(lucid.tensor(0.5))  # log(2) ≈ 0.693
        Tensor(0.6931)
        """
        return -(1.0 - value).log() / self.rate

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Exponential distribution (in nats).

        .. math::

            H(X) = 1 - \log \lambda

        The entropy decreases as the rate increases (concentrated
        distributions have lower uncertainty).

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Exponential(rate=1.0).entropy()  # 1 - log(1) = 1.0
        Tensor(1.0)
        """
        return 1.0 - self.rate.log()


class Laplace(Distribution):
    r"""Laplace (double-exponential) distribution on :math:`\mathbb{R}`.

    Symmetric continuous distribution composed of two mirrored Exponential
    densities placed back-to-back at the location :math:`\mu`.  It is the
    maximum-entropy distribution with a given mean and a given mean
    absolute deviation, and is the prior whose negative log-density induces
    the L1 (Lasso) penalty.

    Parameters
    ----------
    loc : Tensor or float
        Location parameter :math:`\mu \in \mathbb{R}` (mean = median = mode).
    scale : Tensor or float
        Scale / diversity parameter :math:`b > 0`.
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density:

    .. math::

        p(x; \mu, b) = \frac{1}{2b} \exp\!\left(-\frac{|x - \mu|}{b}\right)

    Moments:

    .. math::

        \mathbb{E}[X] = \mu, \qquad
        \mathrm{Var}[X] = 2 b^2, \qquad
        H[X] = 1 + \log(2b)

    The Laplace has heavier tails than the Normal: its kurtosis is 6 vs. 3
    for the Normal, and its tails decay exponentially rather than
    quadratically in log-space.

    Sampling uses the closed-form inverse CDF:

    .. math::

        X = \mu - b \operatorname{sgn}(U - \tfrac{1}{2})
            \log(1 - 2|U - \tfrac{1}{2}|),
        \quad U \sim \mathrm{Uniform}(0, 1)

    so :meth:`rsample` is exact and gradient-friendly through both
    :math:`\mu` and :math:`b`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Laplace
    >>> d = Laplace(loc=0.0, scale=1.0)
    >>> d.mean
    Tensor(0.0)
    >>> d.rsample((4,))
    Tensor([...])
    >>> d.log_prob(lucid.tensor(0.0))
    Tensor(-0.6931)
    """

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct a Laplace distribution.

        Parameters
        ----------
        loc : Tensor | float
            Location parameter :math:`\mu \in \mathbb{R}` (the mean and
            median of the distribution).
        scale : Tensor | float
            Scale (diversity) parameter :math:`b > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Laplace distribution has PDF:

        .. math::

            p(x; \mu, b) = \frac{1}{2b} \exp\!\left(-\frac{|x - \mu|}{b}\right)

        It is also called the double-exponential distribution because it
        resembles two mirrored Exponential densities placed back-to-back at
        the location :math:`\mu`.  It is frequently used in robust
        statistics and as the prior inducing L1 (Lasso) regularisation.

        Examples
        --------
        >>> from lucid.distributions import Laplace
        >>> d = Laplace(loc=0.0, scale=1.0)
        >>> d.mean
        Tensor(0.0)
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
        r"""Expected value of the Laplace distribution.

        .. math::

            E[X] = \mu

        Returns
        -------
        Tensor
            Location parameter :math:`\mu`, shape ``batch_shape``.

        Examples
        --------
        >>> Laplace(loc=3.0, scale=1.0).mean
        Tensor(3.0)
        """
        return self.loc

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Laplace distribution.

        .. math::

            \operatorname{Var}[X] = 2b^2

        Returns
        -------
        Tensor
            Variance :math:`2b^2`, shape ``batch_shape``.

        Examples
        --------
        >>> Laplace(loc=0.0, scale=1.0).variance
        Tensor(2.0)
        """
        return 2.0 * self.scale * self.scale

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Reparameterised sample via the inverse-CDF method.

        Uses the closed-form quantile function:

        .. math::

            X = \mu - b \operatorname{sgn}(u) \log(1 - 2|u|)

        where :math:`u = U - 0.5` and :math:`U \sim \text{Uniform}(0, 1)`.
        Gradients flow through both :math:`\mu` and :math:`b`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Reparameterised samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = Laplace(loc=0.0, scale=1.0)
        >>> x = d.rsample((200,))
        """
        # icdf: loc - scale · sign(u-0.5) · log(1 - 2|u-0.5|).
        # Reparameterise as 2U-1 then split into sign+magnitude.
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device) - 0.5
        sign_u = u.sign()
        return self.loc - self.scale * sign_u * (1.0 - 2.0 * u.abs()).log()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Laplace distribution.

        .. math::

            \log p(x; \mu, b) = -\frac{|x - \mu|}{b} - \log(2b)

        Parameters
        ----------
        value : Tensor
            Real-valued observations.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> Laplace(loc=0.0, scale=1.0).log_prob(lucid.tensor(0.0))
        Tensor(-0.6931)
        """
        if self._validate_args:
            self._validate_sample(value)
        return -((value - self.loc).abs()) / self.scale - (2.0 * self.scale).log()

    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Laplace distribution.

        .. math::

            F(x; \mu, b) = \frac{1}{2} - \frac{1}{2}
            \operatorname{sgn}(x - \mu)
            \left(1 - e^{-|x-\mu|/b}\right)

        Parameters
        ----------
        value : Tensor
            Real values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            CDF values in :math:`[0, 1]`, shape ``batch_shape``.
        """
        z = (value - self.loc) / self.scale
        return 0.5 - 0.5 * z.sign() * (1.0 - (-z.abs()).exp())

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Laplace distribution (in nats).

        .. math::

            H(X) = 1 + \log(2b)

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Laplace(loc=0.0, scale=1.0).entropy()  # 1 + log(2) ≈ 1.693
        Tensor(1.6931)
        """
        return 1.0 + (2.0 * self.scale).log()


class Cauchy(Distribution):
    r"""Cauchy distribution on :math:`\mathbb{R}` — heavy-tailed, all moments undefined.

    The canonical example of a *pathological* heavy-tailed distribution:
    its tails decay so slowly that no integer moment (mean, variance,
    skew, kurtosis) is defined.  The Cauchy is the Student's-t
    distribution with one degree of freedom and arises naturally as the
    ratio of two independent standard Normals.

    Parameters
    ----------
    loc : Tensor or float
        Location parameter :math:`x_0 \in \mathbb{R}` (median and mode).
        The "mean" does *not* exist.
    scale : Tensor or float
        Scale parameter :math:`\gamma > 0` (half-width at half-maximum).
    validate_args : bool, optional
        If ``True``, validate parameter constraints at construction time.

    Notes
    -----
    Probability density:

    .. math::

        p(x; x_0, \gamma) =
            \frac{1}{\pi \gamma}
            \left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]^{-1}

    Cumulative distribution:

    .. math::

        F(x; x_0, \gamma) =
            \frac{1}{\pi} \arctan\!\left(\frac{x - x_0}{\gamma}\right) + \frac{1}{2}

    Properties (no moments — *all* are undefined or infinite):

    * **Median / mode**: :math:`x_0`
    * **Mean / variance**: undefined (integrals diverge)
    * **Entropy**: :math:`H[X] = \log(4 \pi \gamma)`

    **Sample mean does not converge** — the strong law of large numbers
    fails for IID Cauchy variates.  Any finite-sample average of Cauchy
    draws is itself Cauchy-distributed with the same parameters.

    **Stability**: a sum of independent Cauchy variables remains Cauchy
    (it is an :math:`\alpha`-stable distribution with stability index 1).

    Reparameterised sampling uses the inverse-CDF
    :math:`X = x_0 + \gamma \tan(\pi(U - \tfrac{1}{2}))`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Cauchy
    >>> d = Cauchy(loc=0.0, scale=1.0)
    >>> d.rsample((4,))
    Tensor([...])
    >>> d.log_prob(lucid.tensor(0.0))
    Tensor(-1.1447)
    """

    arg_constraints = {"loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample = True

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct a Cauchy distribution.

        Parameters
        ----------
        loc : Tensor | float
            Location parameter (median and mode) :math:`x_0 \in \mathbb{R}`.
        scale : Tensor | float
            Scale (half-width at half-maximum) parameter :math:`\gamma > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Cauchy distribution has PDF:

        .. math::

            p(x; x_0, \gamma) = \frac{1}{\pi \gamma}
            \left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]^{-1}

        The Cauchy is the Student's t-distribution with one degree of
        freedom.  Because all moments of order :math:`\geq 1` are undefined,
        it is a canonical example of a heavy-tailed distribution where the
        sample mean does *not* converge to any value.

        Examples
        --------
        >>> from lucid.distributions import Cauchy
        >>> d = Cauchy(loc=0.0, scale=1.0)
        >>> x = d.rsample((1000,))
        """
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.loc, self.scale = _broadcast_pair(self.loc, self.scale)
        super().__init__(
            batch_shape=tuple(self.loc.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Reparameterised sample via the inverse-CDF method.

        Uses the quantile function of the Cauchy distribution:

        .. math::

            X = x_0 + \gamma \tan\!\bigl(\pi (U - \tfrac{1}{2})\bigr)

        where :math:`U \sim \text{Uniform}(0, 1)`.  Gradients flow
        through both :math:`x_0` and :math:`\gamma`.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Reparameterised samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = Cauchy(loc=0.0, scale=1.0)
        >>> x = d.rsample((500,))
        """
        # icdf: loc + scale · tan(π·(U − 0.5)).
        shape = self._extended_shape(sample_shape)
        u = lucid.rand(*shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + self.scale * (math.pi * (u - 0.5)).tan()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Cauchy distribution.

        .. math::

            \log p(x; x_0, \gamma) = -\log \pi - \log \gamma
            - \log\!\left(1 + \left(\frac{x - x_0}{\gamma}\right)^2\right)

        Parameters
        ----------
        value : Tensor
            Real-valued observations.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> Cauchy(loc=0.0, scale=1.0).log_prob(lucid.tensor(0.0))
        Tensor(-1.1447)
        """
        z = (value - self.loc) / self.scale
        return -math.log(math.pi) - self.scale.log() - (1.0 + z * z).log()

    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function of the Cauchy distribution.

        .. math::

            F(x; x_0, \gamma) = \frac{1}{\pi}
            \arctan\!\left(\frac{x - x_0}{\gamma}\right) + \frac{1}{2}

        Parameters
        ----------
        value : Tensor
            Real values at which to evaluate the CDF.

        Returns
        -------
        Tensor
            CDF values in :math:`(0, 1)`, shape ``batch_shape``.
        """
        z = (value - self.loc) / self.scale
        return 0.5 + z.arctan() / math.pi

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Cauchy distribution (in nats).

        .. math::

            H(X) = \log(4 \pi \gamma)

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Cauchy(loc=0.0, scale=1.0).entropy()  # log(4π) ≈ 2.531
        Tensor(2.5310)
        """
        return math.log(4.0 * math.pi) + self.scale.log()
