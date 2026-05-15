"""``Distribution`` base class for ``lucid.distributions``.

Concrete distributions subclass this and override the bits that apply
(``rsample`` for reparameterisable, ``log_prob`` for everything,
``mean`` / ``variance`` / ``entropy`` for closed-form moments).  The
default behaviours mimic the reference framework: methods that aren't
implemented raise :class:`NotImplementedError`, ``sample`` is
``rsample`` detached, and ``stddev = √variance``.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import Constraint


class Distribution:
    r"""Abstract base for a probability distribution.

    A distribution encodes a probability measure over a measurable space
    and exposes a standard interface for sampling, evaluating
    log-probabilities, and computing closed-form moments.  Every concrete
    distribution in ``lucid.distributions`` inherits from this class.

    Subclasses set:
      - ``arg_constraints`` — dict of param-name → :class:`Constraint`,
        used by ``validate_args`` and to spell out the parameter domain.
      - ``support`` — :class:`Constraint` for the random variable.
      - ``has_rsample`` — ``True`` for reparameterisable families.
      - ``batch_shape``, ``event_shape``.

    Either ``rsample`` or ``sample`` (or both) must be overridden.

    Parameters
    ----------
    batch_shape : tuple[int, ...], optional
        Shape of independent, non-identical draws.  For a scalar
        parameter this is ``()``, for a vector parameter of length ``n``
        it is ``(n,)``.  Default is ``()``.
    event_shape : tuple[int, ...], optional
        Shape of a *single* event (observation).  Univariate
        distributions have ``event_shape = ()``.  Multivariate
        distributions such as :class:`~lucid.distributions.Dirichlet`
        have a non-empty ``event_shape``.  Default is ``()``.
    validate_args : bool or None, optional
        If ``True``, parameter constraints and sample support are
        validated at construction time and in :meth:`log_prob`.  Useful
        during development; disable in production for speed.  ``None``
        inherits the class-level ``_validate_args`` flag.  Default is
        ``None``.

    Attributes
    ----------
    arg_constraints : dict[str, Constraint]
        Maps each constructor parameter name to the
        :class:`~lucid.distributions.constraints.Constraint` it must
        satisfy.  Populated by each concrete subclass.
    support : Constraint or None
        Constraint describing the set on which the distribution has
        positive density (or probability mass).  ``None`` means
        unconstrained.
    has_rsample : bool
        ``True`` when the distribution implements
        :meth:`rsample` — i.e., when the reparameterisation trick
        (Kingma & Welling, 2013) is available and gradients flow through
        sampled values.
    has_enumerate_support : bool
        ``True`` for finite discrete distributions that can enumerate
        every possible outcome, enabling exact marginalisation.

    Notes
    -----
    **Shape semantics**

    Every tensor returned by :meth:`sample` or :meth:`rsample` has shape

    .. math::

        \text{sample\_shape} + \text{batch\_shape} + \text{event\_shape}

    where ``+`` denotes tuple concatenation.  :meth:`log_prob` returns a
    tensor of shape ``sample_shape + batch_shape``, having reduced over
    ``event_shape``.

    **Reparameterisation**

    When ``has_rsample = True`` the sampler can be written as a
    deterministic transformation of a fixed-distribution noise variable
    :math:`\varepsilon`:

    .. math::

        X = g_{\theta}(\varepsilon), \quad \varepsilon \sim p(\varepsilon)

    This allows gradients :math:`\nabla_{\theta} \mathbb{E}[f(X)]` to
    be estimated with low variance via the pathwise derivative, which is
    the backbone of the VAE objective (ELBO) and stochastic computation
    graphs in general.

    Examples
    --------
    >>> import lucid.distributions as dist
    >>> d = dist.Normal(loc=0.0, scale=1.0)
    >>> d.batch_shape
    ()
    >>> d.event_shape
    ()
    >>> x = d.rsample((100,))  # shape (100,)
    >>> x.shape
    (100,)
    """

    arg_constraints: dict[str, Constraint] = {}
    support: Constraint | None = None
    has_rsample: bool = False
    has_enumerate_support: bool = False

    _validate_args: bool = False

    def __init__(
        self,
        batch_shape: tuple[int, ...] = (),
        event_shape: tuple[int, ...] = (),
        validate_args: bool | None = None,
    ) -> None:
        """Initialise batch/event shapes and optionally validate parameters.

        Parameters
        ----------
        batch_shape : tuple[int, ...], optional
            Shape of the batch of independent distributions.
        event_shape : tuple[int, ...], optional
            Shape of each individual event sample.
        validate_args : bool or None, optional
            When ``True``, ``_validate_params`` is called immediately so
            that out-of-constraint constructor arguments raise
            :class:`ValueError` at construction time rather than silently
            producing ``NaN`` values later.
        """
        self._batch_shape = tuple(batch_shape)
        self._event_shape = tuple(event_shape)
        if validate_args is not None:
            self._validate_args = validate_args
        if self._validate_args:
            self._validate_params()

    # ── shape introspection ────────────────────────────────────────────────

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shape of the batch of independent (but not identically parameterised)
        distributions.

        Returns
        -------
        tuple[int, ...]
            A tuple of integers.  ``()`` for a single scalar distribution.
        """
        return self._batch_shape

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Shape of a single observation drawn from the distribution.

        Returns
        -------
        tuple[int, ...]
            ``()`` for univariate distributions.  Non-empty for
            multivariate distributions such as
            :class:`~lucid.distributions.Dirichlet`.
        """
        return self._event_shape

    def _extended_shape(self, sample_shape: tuple[int, ...] = ()) -> tuple[int, ...]:
        """Compute the full tensor shape for a batch of samples.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Number of independent samples to draw at each batch position.

        Returns
        -------
        tuple[int, ...]
            ``sample_shape + batch_shape + event_shape``.
        """
        return tuple(sample_shape) + self._batch_shape + self._event_shape

    # ── moments / closed-form quantities ───────────────────────────────────

    @property
    def mean(self) -> Tensor:
        """Expected value of the distribution.

        Returns
        -------
        Tensor
            A tensor of shape ``batch_shape``.  Raises
            :class:`NotImplementedError` if the distribution has no
            closed-form mean (e.g. :class:`~lucid.distributions.Cauchy`).
        """
        raise NotImplementedError(f"{type(self).__name__}.mean")

    @property
    def mode(self) -> Tensor:
        """Most likely value of the distribution (the argmax of the density).

        Returns
        -------
        Tensor
            A tensor of shape ``batch_shape``.  Raises
            :class:`NotImplementedError` if not implemented by the
            concrete subclass.
        """
        raise NotImplementedError(f"{type(self).__name__}.mode")

    @property
    def variance(self) -> Tensor:
        r"""Variance of the distribution.

        Returns
        -------
        Tensor
            A tensor of shape ``batch_shape``.  The variance is the
            second central moment, :math:`\text{Var}[X] = \mathbb{E}[(X - \mu)^2]`.
            Raises :class:`NotImplementedError` if not provided by the
            concrete subclass.
        """
        raise NotImplementedError(f"{type(self).__name__}.variance")

    @property
    def stddev(self) -> Tensor:
        r"""Standard deviation of the distribution.

        Computed as :math:`\sigma = \sqrt{\text{Var}[X]}`.  Concrete
        subclasses may override this for numerical efficiency; the default
        implementation delegates to :attr:`variance`.

        Returns
        -------
        Tensor
            A tensor of shape ``batch_shape``.
        """
        return self.variance.sqrt()

    def entropy(self) -> Tensor:
        r"""Shannon differential (or discrete) entropy.

        Defined as

        .. math::

            H[X] = -\mathbb{E}_{x \sim p}[\log p(x)]

        for a continuous distribution and similarly for discrete ones.
        Measured in nats (natural logarithm base).

        Returns
        -------
        Tensor
            A tensor of shape ``batch_shape``.  Raises
            :class:`NotImplementedError` if not implemented by the
            concrete subclass.

        Notes
        -----
        Entropy quantifies the average amount of "surprise" or
        uncertainty in a single draw.  Higher entropy means the
        distribution is more spread out / less predictable.
        """
        raise NotImplementedError(f"{type(self).__name__}.entropy")

    # ── sampling ───────────────────────────────────────────────────────────

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw independent, identically distributed samples.

        The default implementation calls :meth:`rsample` and detaches the
        result from the autograd graph, so gradients do not flow through
        the returned tensor.  Discrete distributions that cannot be
        reparameterised must override this method directly instead.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Number of independent samples to draw.  The returned tensor
            has shape ``sample_shape + batch_shape + event_shape``.
            Default is ``()``, which returns a single sample with shape
            ``batch_shape + event_shape``.

        Returns
        -------
        Tensor
            A detached tensor (no gradient) of shape
            ``sample_shape + batch_shape + event_shape``.
        """
        return self.rsample(sample_shape).detach()

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Reparameterised sample — gradients flow through the noise.

        Unlike :meth:`sample`, ``rsample`` expresses the stochastic node as
        a *deterministic* transformation of a parameter-free noise variable:

        .. math::

            X = g_{\theta}(\varepsilon), \quad \varepsilon \sim p_0(\varepsilon)

        This factorisation allows the gradient
        :math:`\nabla_{\theta} \mathbb{E}_{X}[f(X)]` to be estimated
        cheaply via the pathwise (re-parameterisation) estimator, which
        has much lower variance than the REINFORCE estimator.

        Concrete distributions must override either this or :meth:`sample`.
        Only distributions that admit a differentiable sampler set
        ``has_rsample = True``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Number of independent samples to draw.  The returned tensor
            has shape ``sample_shape + batch_shape + event_shape``.

        Returns
        -------
        Tensor
            A tensor attached to the autograd graph through the
            distribution parameters.

        Raises
        ------
        NotImplementedError
            If the distribution does not support reparameterised
            sampling (``has_rsample = False``).  Use :meth:`sample`
            instead in that case.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.rsample is not reparameterisable; "
            f"use sample() for non-reparameterised draws."
        )

    # ── log-probabilities ──────────────────────────────────────────────────

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability (log-density) evaluated at ``value``.

        For a continuous distribution with density :math:`p(x)` this
        returns :math:`\log p(x)`.  For a discrete distribution with
        probability mass function :math:`P(X = k)` this returns
        :math:`\log P(X = k)`.

        Working in log-space is numerically preferable to evaluating the
        density directly: products of probabilities become sums of
        log-probabilities, avoiding underflow for long sequences.

        Parameters
        ----------
        value : Tensor
            Point(s) at which to evaluate the log-density.  Must be
            broadcastable with ``batch_shape + event_shape``.

        Returns
        -------
        Tensor
            Log-probability tensor of shape
            ``broadcast(value.shape, batch_shape + event_shape)[:-len(event_shape)]``.
            In the scalar / univariate case this simplifies to
            ``broadcast(value.shape, batch_shape)``.

        Raises
        ------
        NotImplementedError
            If not implemented by the concrete subclass.
        """
        raise NotImplementedError(f"{type(self).__name__}.log_prob")

    def cdf(self, value: Tensor) -> Tensor:
        r"""Cumulative distribution function (CDF) evaluated at ``value``.

        Returns the probability that a random variable :math:`X` drawn
        from this distribution is less than or equal to ``value``:

        .. math::

            F(x) = P(X \leq x) = \int_{-\infty}^{x} p(t)\, dt

        Parameters
        ----------
        value : Tensor
            Point(s) at which to evaluate the CDF.

        Returns
        -------
        Tensor
            Values in :math:`[0, 1]` with the same shape as
            ``broadcast(value, batch_shape)``.

        Raises
        ------
        NotImplementedError
            If not implemented by the concrete subclass.
        """
        raise NotImplementedError(f"{type(self).__name__}.cdf")

    def icdf(self, value: Tensor) -> Tensor:
        r"""Inverse CDF (quantile function / percent-point function).

        Given a probability :math:`p \in [0, 1]` returns the smallest
        :math:`x` such that :math:`F(x) \geq p`:

        .. math::

            Q(p) = F^{-1}(p) = \inf\{x \in \mathbb{R} : F(x) \geq p\}

        The quantile function is particularly useful for inverse-CDF
        (Smirnov transform) sampling: if :math:`U \sim \text{Uniform}(0,1)`
        then :math:`Q(U) \sim p`.

        Parameters
        ----------
        value : Tensor
            Probability values in :math:`[0, 1]`.

        Returns
        -------
        Tensor
            Quantiles with the same shape as
            ``broadcast(value, batch_shape)``.

        Raises
        ------
        NotImplementedError
            If not implemented by the concrete subclass.
        """
        raise NotImplementedError(f"{type(self).__name__}.icdf")

    def prob(self, value: Tensor) -> Tensor:
        r"""Probability density (or mass) at ``value``.

        Computed as :math:`\exp(\log p(x))` via :meth:`log_prob`.  For
        numerical stability prefer working with :meth:`log_prob` directly;
        use this method only when the raw density value is needed.

        Parameters
        ----------
        value : Tensor
            Point(s) at which to evaluate the density/mass.

        Returns
        -------
        Tensor
            Non-negative density or probability-mass values.
        """
        return self.log_prob(value).exp()

    # ── validation ─────────────────────────────────────────────────────────

    def _validate_params(self) -> None:
        """Check that all constructor arguments satisfy their constraints.

        Iterates over :attr:`arg_constraints` and calls
        ``constraint.check(param).all()`` for each tensor-valued parameter.
        Raises :class:`ValueError` on the first violation found.
        Called automatically during ``__init__`` when
        ``validate_args=True``.
        """
        for name, constraint in self.arg_constraints.items():
            if not hasattr(self, name):
                continue
            v = getattr(self, name)
            if not isinstance(v, lucid.Tensor):
                continue
            if not bool(constraint.check(v).all().item()):
                raise ValueError(
                    f"{type(self).__name__} parameter {name!r} "
                    f"out of constraint {constraint!r}"
                )

    def _validate_sample(self, value: Tensor) -> None:
        """Check that ``value`` lies within the distribution's support.

        Called inside :meth:`log_prob` when ``_validate_args`` is ``True``.
        For example, passing a negative value to an Exponential
        distribution would raise :class:`ValueError` here rather than
        silently returning ``-inf`` or ``NaN``.

        Parameters
        ----------
        value : Tensor
            Sample tensor to validate.

        Raises
        ------
        ValueError
            If any element of ``value`` is outside :attr:`support`.
        """
        if self.support is None:
            return
        if not bool(self.support.check(value).all().item()):
            raise ValueError(
                f"{type(self).__name__}: value out of support {self.support!r}"
            )

    # ── pretty-printing ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Concise string representation showing parameter shapes.

        Returns
        -------
        str
            E.g. ``"Normal(loc=(3,), scale=(3,))"`` or
            ``"Bernoulli(probs=0.3)"`` for a scalar.
        """
        params = []
        for name in self.arg_constraints:
            if not hasattr(self, name):
                continue
            v = getattr(self, name)
            if isinstance(v, lucid.Tensor):
                params.append(f"{name}={tuple(v.shape)}")
            else:
                params.append(f"{name}={v!r}")
        return f"{type(self).__name__}({', '.join(params)})"


class ExponentialFamily(Distribution):
    r"""Mix-in for exponential-family distributions.

    An exponential-family distribution has a density of the form

    .. math::

        p(x \mid \eta) = h(x)\, \exp\!\bigl(\eta^\top T(x) - A(\eta)\bigr)

    where:

    * :math:`\eta` are the *natural parameters* (also called canonical
      parameters),
    * :math:`T(x)` is the vector of *sufficient statistics*,
    * :math:`A(\eta) = \log Z(\eta)` is the *log-partition function*
      (log-normaliser), and
    * :math:`h(x)` is the *base measure*.

    This family encompasses most common distributions: Normal, Bernoulli,
    Gamma, Beta, Poisson, Dirichlet, and many more.

    Subclasses provide :attr:`_natural_params` and :meth:`_log_normalizer`
    so that entropy can be derived from the standard exponential-family
    identity

    .. math::

        H[X] = A(\eta) - \eta^\top \nabla_\eta A(\eta)

    Per-distribution overrides of :meth:`~Distribution.entropy` remain
    available for closed-form efficiency.

    Notes
    -----
    The exponential family structure implies:

    * The log-normalizer :math:`A(\eta)` is convex in :math:`\eta`.
    * :math:`\nabla_\eta A(\eta) = \mathbb{E}[T(X)]` (the mean
      parameter).
    * The Hessian :math:`\nabla^2_\eta A(\eta)` equals the covariance
      matrix of :math:`T(X)`, hence is always positive semi-definite.

    These properties underpin many elegant results in information geometry,
    variational inference, and natural gradient methods.

    **Members of this family in lucid.distributions** include
    :class:`~lucid.distributions.Normal`,
    :class:`~lucid.distributions.Bernoulli`,
    :class:`~lucid.distributions.Categorical`,
    :class:`~lucid.distributions.Gamma`,
    :class:`~lucid.distributions.Beta`,
    :class:`~lucid.distributions.Dirichlet`,
    :class:`~lucid.distributions.Exponential`, and
    :class:`~lucid.distributions.Poisson`.  Heavy-tailed families such
    as :class:`~lucid.distributions.Cauchy` and Student's-t are *not*
    members.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Bernoulli
    >>> d = Bernoulli(probs=0.7)  # ExponentialFamily subclass
    >>> isinstance(d, lucid.distributions.distribution.ExponentialFamily)
    True
    >>> d.entropy()
    Tensor(...)
    """

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        r"""Natural (canonical) parameters :math:`\eta` of the distribution.

        Returns
        -------
        tuple[Tensor, ...]
            A tuple of tensors, one per natural parameter dimension.
            Concrete subclasses must override this property.

        Raises
        ------
        NotImplementedError
            If the subclass has not provided an implementation.
        """
        raise NotImplementedError(f"{type(self).__name__}._natural_params")

    def _log_normalizer(self, *natural_params: Tensor) -> Tensor:
        r"""Log-partition function :math:`A(\eta)` evaluated at the natural parameters.

        The log-partition function is defined as

        .. math::

            A(\eta) = \log \int h(x)\, e^{\eta^\top T(x)}\, dx

        and is the normalising constant that ensures the density integrates
        to one.

        Parameters
        ----------
        *natural_params : Tensor
            The natural parameter tensors, matching the order of
            :attr:`_natural_params`.

        Returns
        -------
        Tensor
            Scalar or batch-shaped log-partition value.

        Raises
        ------
        NotImplementedError
            If the subclass has not provided an implementation.
        """
        raise NotImplementedError(f"{type(self).__name__}._log_normalizer")
