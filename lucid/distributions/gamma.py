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
        r"""Construct a Gamma distribution.

        Parameters
        ----------
        concentration : Tensor | float
            Shape (concentration) parameter :math:`\alpha > 0`.  Also known
            as the shape parameter :math:`k`.
        rate : Tensor | float
            Rate (inverse-scale) parameter :math:`\beta > 0`.  The scale
            is :math:`\theta = 1/\beta`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Gamma distribution has PDF:

        .. math::

            p(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)}
            x^{\alpha-1} e^{-\beta x}, \quad x > 0

        Special cases include:

        - :math:`\text{Gamma}(1, \beta) = \text{Exponential}(\beta)`
        - :math:`\text{Gamma}(k/2, 1/2) = \chi^2(k)`

        Sampling uses the Marsaglia–Tsang acceptance-rejection algorithm
        (see module docstring); the accept rate is > 95 % so eight retry
        rounds are more than sufficient.

        Examples
        --------
        >>> from lucid.distributions import Gamma
        >>> d = Gamma(concentration=2.0, rate=1.0)
        >>> d.mean  # α/β = 2.0
        Tensor(2.0)
        """
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
        r"""Expected value of the Gamma distribution.

        .. math::

            E[X] = \frac{\alpha}{\beta}

        Returns
        -------
        Tensor
            Mean :math:`\alpha/\beta`, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=3.0, rate=2.0).mean
        Tensor(1.5)
        """
        return self.concentration / self.rate

    @property
    def mode(self) -> Tensor:
        r"""Mode of the Gamma distribution.

        .. math::

            \text{mode} = \frac{\max(\alpha - 1,\; 0)}{\beta}

        The mode is zero for :math:`\alpha \leq 1` (the density is
        monotonically decreasing from infinity) and positive for
        :math:`\alpha > 1`.

        Returns
        -------
        Tensor
            Mode :math:`\max(\alpha-1, 0)/\beta`, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=3.0, rate=1.0).mode
        Tensor(2.0)
        """
        return ((self.concentration - 1.0).clip(0.0, float("inf"))) / self.rate

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Gamma distribution.

        .. math::

            \operatorname{Var}[X] = \frac{\alpha}{\beta^2}

        Returns
        -------
        Tensor
            Variance :math:`\alpha/\beta^2`, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=4.0, rate=2.0).variance
        Tensor(1.0)
        """
        return self.concentration / (self.rate * self.rate)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Gamma distribution.

        Delegates to ``_sample_standard_gamma`` (Marsaglia–Tsang algorithm)
        and scales by :math:`1/\beta` to obtain :math:`\text{Gamma}(\alpha, \beta)`
        samples.  The result is **detached** since the rejection-based
        sampler does not support reparameterisation.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Non-negative samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = Gamma(concentration=2.0, rate=1.0)
        >>> x = d.sample((500,))
        >>> x.mean()  # approximately 2.0
        """
        std = _sample_standard_gamma(self.concentration, sample_shape)
        return (std / self.rate).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Gamma distribution.

        .. math::

            \log p(x; \alpha, \beta) = \alpha \log \beta
            + (\alpha - 1) \log x - \beta x - \log \Gamma(\alpha)

        Parameters
        ----------
        value : Tensor
            Positive real values :math:`x > 0`.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=1.0, rate=1.0).log_prob(lucid.tensor(1.0))
        Tensor(-1.0)
        """
        return (
            self.concentration * self.rate.log()
            + (self.concentration - 1.0) * value.log()
            - self.rate * value
            - lucid.lgamma(self.concentration)
        )

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Gamma distribution (in nats).

        .. math::

            H(X) = \alpha - \log \beta + \log \Gamma(\alpha)
            + (1 - \alpha) \psi(\alpha)

        where :math:`\psi` is the digamma function.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> Gamma(concentration=1.0, rate=1.0).entropy()  # Exp(1): H = 1
        Tensor(1.0)
        """
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
        r"""Construct a Chi-squared distribution.

        Parameters
        ----------
        df : Tensor | float
            Degrees of freedom :math:`k > 0`.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        :math:`\chi^2(k)` is a special case of the Gamma distribution:

        .. math::

            \chi^2(k) = \text{Gamma}\!\left(\frac{k}{2},\; \frac{1}{2}\right)

        It arises as the distribution of the sum of squares of :math:`k`
        independent standard Normal variables, and is fundamental in
        hypothesis testing (e.g., goodness-of-fit tests, likelihood-ratio
        tests).

        Examples
        --------
        >>> from lucid.distributions import Chi2
        >>> d = Chi2(df=4.0)
        >>> d.mean  # k = 4
        Tensor(4.0)
        """
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
        r"""Construct a Beta distribution.

        Parameters
        ----------
        concentration1 : Tensor | float
            First shape parameter :math:`\alpha > 0` (also called
            ``concentration1``).  Controls how much mass is near 1.
        concentration0 : Tensor | float
            Second shape parameter :math:`\beta > 0` (also called
            ``concentration0``).  Controls how much mass is near 0.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Beta distribution has PDF:

        .. math::

            p(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)},
            \quad x \in [0, 1]

        where :math:`B(\alpha, \beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)`.

        Sampling uses the ratio-of-Gammas method:
        :math:`X = G_\alpha / (G_\alpha + G_\beta)` where
        :math:`G_\alpha \sim \text{Gamma}(\alpha, 1)`.

        Examples
        --------
        >>> from lucid.distributions import Beta
        >>> d = Beta(concentration1=2.0, concentration0=5.0)
        >>> d.mean  # α/(α+β) = 2/7 ≈ 0.286
        Tensor(0.2857)
        """
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
        r"""Expected value of the Beta distribution.

        .. math::

            E[X] = \frac{\alpha}{\alpha + \beta}

        Returns
        -------
        Tensor
            Mean :math:`\alpha/(\alpha+\beta)`, shape ``batch_shape``.

        Examples
        --------
        >>> Beta(concentration1=1.0, concentration0=1.0).mean
        Tensor(0.5)
        """
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Beta distribution.

        .. math::

            \operatorname{Var}[X] =
            \frac{\alpha \beta}{(\alpha+\beta)^2 (\alpha+\beta+1)}

        Returns
        -------
        Tensor
            Variance, shape ``batch_shape``.

        Examples
        --------
        >>> Beta(concentration1=2.0, concentration0=2.0).variance
        Tensor(0.05)
        """
        a = self.concentration1
        b = self.concentration0
        ab = a + b
        return (a * b) / (ab * ab * (ab + 1.0))

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Beta distribution.

        Uses the ratio-of-independent-Gamma-samples identity:

        .. math::

            X = \frac{G_\alpha}{G_\alpha + G_\beta},
            \quad G_\alpha \sim \text{Gamma}(\alpha, 1),\;
            G_\beta \sim \text{Gamma}(\beta, 1)

        The result is **detached** since the underlying Gamma sampler is
        rejection-based and does not support reparameterisation.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Samples in :math:`[0, 1]` of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = Beta(concentration1=2.0, concentration0=5.0)
        >>> x = d.sample((500,))
        >>> x.mean()  # approximately 2/7 ≈ 0.286
        """
        # Beta(α, β) = X / (X + Y) with X ~ Gamma(α, 1), Y ~ Gamma(β, 1).
        x = _sample_standard_gamma(self.concentration1, sample_shape)
        y = _sample_standard_gamma(self.concentration0, sample_shape)
        return (x / (x + y)).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Beta distribution.

        .. math::

            \log p(x; \alpha, \beta) =
            (\alpha - 1) \log x + (\beta - 1) \log(1-x)
            - \log B(\alpha, \beta)

        Parameters
        ----------
        value : Tensor
            Values in :math:`(0, 1)`.

        Returns
        -------
        Tensor
            Element-wise log-densities, shape ``batch_shape``.
        """
        a = self.concentration1
        b = self.concentration0
        log_b = lucid.lgamma(a) + lucid.lgamma(b) - lucid.lgamma(a + b)
        return (a - 1.0) * value.log() + (b - 1.0) * (1.0 - value).log() - log_b

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Beta distribution (in nats).

        .. math::

            H(X) = \log B(\alpha,\beta)
            - (\alpha-1)\psi(\alpha)
            - (\beta-1)\psi(\beta)
            + (\alpha+\beta-2)\psi(\alpha+\beta)

        where :math:`\psi` is the digamma function and
        :math:`B(\alpha,\beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)`.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.
        """
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
        r"""Construct a Dirichlet distribution.

        Parameters
        ----------
        concentration : Tensor
            Concentration vector :math:`\boldsymbol{\alpha}` with all
            entries :math:`> 0`.  The last dimension is the event (simplex)
            dimension :math:`K`; all preceding dimensions form the batch
            shape.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        The Dirichlet distribution with concentration :math:`\boldsymbol{\alpha}`
        has PDF over the :math:`K`-simplex:

        .. math::

            p(\mathbf{x}; \boldsymbol{\alpha}) =
            \frac{1}{B(\boldsymbol{\alpha})}
            \prod_{i=1}^{K} x_i^{\alpha_i - 1}

        where
        :math:`B(\boldsymbol{\alpha}) = \prod_i \Gamma(\alpha_i) / \Gamma(\sum_i \alpha_i)`.

        Sampling uses the normalised-Gamma trick: draw independent
        :math:`G_i \sim \text{Gamma}(\alpha_i, 1)` then return
        :math:`\mathbf{x} = \mathbf{G} / \sum_i G_i`.

        Examples
        --------
        >>> import lucid
        >>> from lucid.distributions import Dirichlet
        >>> d = Dirichlet(lucid.tensor([1.0, 2.0, 3.0]))
        >>> d.mean  # proportional to concentration
        Tensor([0.1667, 0.3333, 0.5000])
        """
        self.concentration = _as_tensor(concentration)
        shape = tuple(self.concentration.shape)
        super().__init__(
            batch_shape=shape[:-1],
            event_shape=shape[-1:],
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Dirichlet distribution.

        Each component of the mean equals the normalised concentration:

        .. math::

            E[X_i] = \frac{\alpha_i}{\sum_j \alpha_j}

        Returns
        -------
        Tensor
            Mean vector on the simplex, shape ``batch_shape + event_shape``.

        Examples
        --------
        >>> Dirichlet(lucid.tensor([2.0, 2.0])).mean
        Tensor([0.5, 0.5])
        """
        s = self.concentration.sum(dim=-1, keepdim=True)
        return self.concentration / s

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Dirichlet distribution (component-wise).

        .. math::

            \operatorname{Var}[X_i] =
            \frac{\mu_i (1 - \mu_i)}{\alpha_0 + 1},
            \quad \alpha_0 = \sum_j \alpha_j,\;
            \mu_i = \alpha_i / \alpha_0

        Returns
        -------
        Tensor
            Variance vector, shape ``batch_shape + event_shape``.
        """
        a = self.concentration
        s = a.sum(dim=-1, keepdim=True)
        m = a / s
        return m * (1.0 - m) / (s + 1.0)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Dirichlet distribution.

        Uses the normalised-Gamma method:

        .. math::

            \mathbf{x} = \frac{\mathbf{g}}{\sum_i g_i},
            \quad g_i \sim \text{Gamma}(\alpha_i, 1)

        The result lies on the probability simplex and is **detached**.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Simplex-valued samples of shape
            ``sample_shape + batch_shape + event_shape``.

        Examples
        --------
        >>> d = Dirichlet(lucid.tensor([1.0, 1.0, 1.0]))
        >>> x = d.sample((100,))
        >>> x.sum(dim=-1)  # all ones
        """
        gammas = _sample_standard_gamma(self.concentration, sample_shape)
        return (gammas / gammas.sum(dim=-1, keepdim=True)).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Dirichlet distribution.

        .. math::

            \log p(\mathbf{x}; \boldsymbol{\alpha}) =
            \sum_i (\alpha_i - 1) \log x_i - \log B(\boldsymbol{\alpha})

        Parameters
        ----------
        value : Tensor
            Simplex-valued observations, last dimension is :math:`K`.

        Returns
        -------
        Tensor
            Log-densities, shape ``batch_shape``.
        """
        a = self.concentration
        log_b = lucid.lgamma(a).sum(dim=-1) - lucid.lgamma(a.sum(dim=-1))
        return ((a - 1.0) * value.log()).sum(dim=-1) - log_b

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Dirichlet distribution (in nats).

        .. math::

            H = \log B(\boldsymbol{\alpha})
            + (\alpha_0 - K) \psi(\alpha_0)
            - \sum_i (\alpha_i - 1) \psi(\alpha_i)

        where :math:`\alpha_0 = \sum_i \alpha_i`, :math:`K` is the
        number of categories, and :math:`\psi` is the digamma function.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.
        """
        a = self.concentration
        s = a.sum(dim=-1, keepdim=True)
        k = a.shape[-1]
        log_b = lucid.lgamma(a).sum(dim=-1, keepdim=True) - lucid.lgamma(s)
        digamma_diff = (a - 1.0) * (lucid.digamma(a) - lucid.digamma(s))
        return (
            log_b + (s - k) * lucid.digamma(s) - digamma_diff.sum(dim=-1, keepdim=True)
        ).squeeze(-1)
