"""``StudentT`` — Student's t with location-scale.

``T = loc + scale · z / sqrt(g / df)``  where  ``z ∼ N(0, 1)``  and
``g ∼ Chi²(df)``.  Reparameterised over ``z`` only (``g`` flows through
the rejection-based Gamma sampler so gradient is detached on it).
"""

import math

import lucid
import lucid.autograd
from lucid._tensor.tensor import Tensor
from lucid.distributions._util import as_tensor as _as_tensor
from lucid.distributions._util import broadcast_pair as _broadcast_pair
from lucid.distributions.constraints import (
    Constraint,
    positive,
    real,
)
from lucid.distributions.distribution import Distribution


class StudentT(Distribution):
    r"""Student's t-distribution with location, scale, and degrees of freedom.

    ``StudentT(df=ν, loc=μ, scale=σ)`` defines the three-parameter
    location-scale generalisation of Student's t.  It arises naturally in:

    - **Bayesian inference**: the posterior predictive for a Normal likelihood
      with unknown mean and variance (Normal-InverseGamma conjugate model).
    - **Robust regression**: as a heavy-tailed alternative to the Normal for
      outlier-tolerant models.
    - **Limit behaviour**: as :math:`\nu \to \infty`, the t-distribution
      converges to :math:`\mathcal{N}(\mu, \sigma^2)`.

    Parameters
    ----------
    df : Tensor | float
        Degrees of freedom :math:`\nu > 0`.  Controls tail heaviness.
        :math:`\nu = 1` is the Cauchy distribution; :math:`\nu = \infty`
        is the Normal.
    loc : Tensor | float, optional
        Location parameter :math:`\mu \in \mathbb{R}`.  Default is ``0.0``.
    scale : Tensor | float, optional
        Scale parameter :math:`\sigma > 0`.  Default is ``1.0``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    df : Tensor
        Degrees of freedom :math:`\nu`.
    loc : Tensor
        Location parameter :math:`\mu`.
    scale : Tensor
        Scale parameter :math:`\sigma`.

    Notes
    -----
    **PDF**:

    .. math::

        p(x; \nu, \mu, \sigma) =
        \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}
             {\Gamma\!\left(\frac{\nu}{2}\right)
              \sqrt{\pi \nu}\, \sigma}
        \left(1 + \frac{(x-\mu)^2}{\nu \sigma^2}\right)^{-(\nu+1)/2}

    **Log-PDF**:

    .. math::

        \log p(x) = \log\Gamma\!\left(\frac{\nu+1}{2}\right)
                   - \log\Gamma\!\left(\frac{\nu}{2}\right)
                   - \tfrac{1}{2}\log(\pi\nu)
                   - \log\sigma
                   - \frac{\nu+1}{2}\log\!\left(1 + \frac{z^2}{\nu}\right)

    where :math:`z = (x - \mu)/\sigma`.

    **Moments**:

    - Mean (:math:`\nu > 1`): :math:`E[X] = \mu`
    - Variance (:math:`\nu > 2`):
      :math:`\operatorname{Var}[X] = \sigma^2 \nu / (\nu - 2)`
    - The distribution has no finite variance for :math:`\nu \leq 2` and no
      finite mean for :math:`\nu \leq 1`.

    **Reparameterised sampling** uses the representation

    .. math::

        T = \mu + \sigma \cdot z \cdot \sqrt{\nu / g}

    where :math:`z \sim \mathcal{N}(0,1)` is the **differentiable** variate
    and :math:`g \sim \chi^2(\nu)` is **detached** (the Gamma sampler uses
    rejection sampling whose path is not differentiable w.r.t. :math:`\nu`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import StudentT
    >>> # Heavy-tailed (Cauchy)
    >>> cauchy = StudentT(df=1.0, loc=0.0, scale=1.0)
    >>> # Near-Normal
    >>> approx_normal = StudentT(df=100.0, loc=0.0, scale=1.0)
    >>> samples = approx_normal.rsample((500,))
    """

    arg_constraints = {"df": positive, "loc": real, "scale": positive}
    support: Constraint | None = real
    has_rsample: bool = True

    def __init__(
        self,
        df: Tensor | float,
        loc: Tensor | float = 0.0,
        scale: Tensor | float = 1.0,
        validate_args: bool | None = None,
    ) -> None:
        r"""Construct a Student's t-distribution.

        Parameters
        ----------
        df : Tensor | float
            Degrees of freedom :math:`\nu > 0`.  Lower values produce
            heavier tails: :math:`\nu = 1` is the Cauchy distribution,
            while as :math:`\nu \to \infty` the distribution approaches
            :math:`\mathcal{N}(\mu, \sigma^2)`.
        loc : Tensor | float, optional
            Location parameter :math:`\mu \in \mathbb{R}`.  Default is
            ``0.0``.
        scale : Tensor | float, optional
            Scale parameter :math:`\sigma > 0`.  Default is ``1.0``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Notes
        -----
        All three parameters are broadcast against each other so that
        batches with mixed scalar / tensor parameters work naturally.
        The resulting ``batch_shape`` is the broadcast shape of
        ``(df, loc, scale)``.

        Examples
        --------
        >>> from lucid.distributions import StudentT
        >>> d = StudentT(df=5.0, loc=2.0, scale=0.5)
        >>> d.mean
        Tensor(2.0)
        """
        self.df = _as_tensor(df)
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.df, self.loc = _broadcast_pair(self.df, self.loc)
        self.df, self.scale = _broadcast_pair(self.df, self.scale)
        self.loc, self.scale = _broadcast_pair(self.loc, self.scale)
        super().__init__(
            batch_shape=tuple(self.df.shape),
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def mean(self) -> Tensor:
        r"""Expected value of the Student's t-distribution.

        .. math::

            E[X] = \mu \quad (\nu > 1)

        The mean is only mathematically defined for :math:`\nu > 1`.  For
        :math:`\nu \leq 1` (e.g., the Cauchy distribution) the first
        moment does not exist.  Following the convention of most
        distribution libraries, this property returns :math:`\mu`
        unconditionally — callers are responsible for checking
        :math:`\nu > 1` when that matters.

        Returns
        -------
        Tensor
            Location parameter :math:`\mu`, shape ``batch_shape``.

        Examples
        --------
        >>> StudentT(df=5.0, loc=3.0).mean
        Tensor(3.0)
        """
        # Defined for df > 1 — we follow the reference framework and
        # return ``loc`` regardless.
        return self.loc + 0 * self.df

    @property
    def variance(self) -> Tensor:
        r"""Variance of the Student's t-distribution.

        .. math::

            \operatorname{Var}[X] = \frac{\sigma^2 \nu}{\nu - 2}
            \quad (\nu > 2)

        The variance is only finite for :math:`\nu > 2`.  For
        :math:`1 < \nu \leq 2` the distribution has a defined mean but
        infinite variance.  For :math:`\nu \leq 1` neither moment
        exists.  This property computes :math:`\sigma^2 \nu / (\nu - 2)`
        algebraically; the caller must guard against :math:`\nu \leq 2`
        as the result will be negative or infinite in those cases.

        Returns
        -------
        Tensor
            Variance :math:`\sigma^2 \nu / (\nu - 2)`, shape ``batch_shape``.

        Examples
        --------
        >>> StudentT(df=4.0, scale=1.0).variance  # 4/(4-2) = 2.0
        Tensor(2.0)
        """
        # Defined for df > 2:  scale² · df / (df − 2).
        return self.scale * self.scale * self.df / (self.df - 2.0)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Reparameterised sample: gradient flows through the Normal variate.

        ``T = loc + scale · z / sqrt(g / df)``  where  ``z ~ N(0,1)``
        is differentiable and ``g ~ Chi²(df)`` is detached (standard
        practice — the marginal gradient w.r.t. df is not tracked).
        """
        from lucid.distributions.gamma import _sample_standard_gamma

        shape = self._extended_shape(sample_shape)
        z = lucid.randn(*shape, dtype=self.loc.dtype, device=self.loc.device)
        gamma_std = _sample_standard_gamma(self.df * 0.5, sample_shape).detach() * 2.0
        return self.loc + self.scale * z * (self.df / gamma_std).sqrt()

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw non-differentiable samples from the Student's t-distribution.

        Wraps :meth:`rsample` inside a ``no_grad`` context so that the
        returned samples have no gradient history.  Use :meth:`rsample`
        directly when gradient flow through the location-scale
        reparameterisation is required.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape dimensions for the sample batch.  Default is ``()``.

        Returns
        -------
        Tensor
            Detached samples of shape ``sample_shape + batch_shape``.

        Examples
        --------
        >>> d = StudentT(df=3.0, loc=0.0, scale=1.0)
        >>> x = d.sample((200,))
        """
        with lucid.autograd.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-density of ``value`` under the Student's t-distribution.

        .. math::

            \log p(x; \nu, \mu, \sigma) =
            \log\Gamma\!\left(\tfrac{\nu+1}{2}\right)
            - \log\Gamma\!\left(\tfrac{\nu}{2}\right)
            - \tfrac{1}{2}\log(\pi\nu)
            - \log\sigma
            - \tfrac{\nu+1}{2} \log\!\left(1 + \frac{z^2}{\nu}\right)

        where :math:`z = (x - \mu) / \sigma`.

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
        >>> d = StudentT(df=1.0, loc=0.0, scale=1.0)  # Cauchy
        >>> d.log_prob(lucid.tensor(0.0))  # -log(π) ≈ -1.145
        Tensor(-1.1447)
        """
        z = (value - self.loc) / self.scale
        log_density = (
            lucid.lgamma((self.df + 1.0) * 0.5)
            - lucid.lgamma(self.df * 0.5)
            - 0.5 * (math.pi * self.df).log()
            - self.scale.log()
            - 0.5 * (self.df + 1.0) * (1.0 + z * z / self.df).log()
        )
        return log_density

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Student's t-distribution (in nats).

        .. math::

            H = \log\!\left(\sigma\sqrt{\nu}\, B\!\left(\tfrac{1}{2},
            \tfrac{\nu}{2}\right)\right)
            + \frac{\nu+1}{2}
            \left[\psi\!\left(\tfrac{\nu+1}{2}\right)
             - \psi\!\left(\tfrac{\nu}{2}\right)\right]

        where :math:`B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)` is the
        Beta function and :math:`\psi` is the digamma function.

        As :math:`\nu \to \infty` this converges to the entropy of
        :math:`\mathcal{N}(\mu, \sigma^2)`, which is
        :math:`\tfrac{1}{2}\log(2\pi e\sigma^2)`.

        Returns
        -------
        Tensor
            Entropy in nats, shape ``batch_shape``.

        Examples
        --------
        >>> StudentT(df=1.0, loc=0.0, scale=1.0).entropy()  # Cauchy: log(4π) ≈ 2.531
        Tensor(2.5310)
        """
        # H = log(scale·sqrt(df)·B(0.5, df/2)) + (df+1)/2 · (digamma((df+1)/2) − digamma(df/2)).
        half = lucid.tensor(0.5) + 0 * self.df  # broadcast 0.5 → df-shape.
        beta_term = (
            lucid.lgamma(half)
            + lucid.lgamma(self.df * 0.5)
            - lucid.lgamma((self.df + 1.0) * 0.5)
        )
        return (
            self.scale.log()
            + 0.5 * self.df.log()
            + beta_term
            + (self.df + 1.0)
            * 0.5
            * (lucid.digamma((self.df + 1.0) * 0.5) - lucid.digamma(self.df * 0.5))
        )
