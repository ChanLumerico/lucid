"""``Distribution`` base class for ``lucid.distributions``.

Concrete distributions subclass this and override the bits that apply
(``rsample`` for reparameterisable, ``log_prob`` for everything,
``mean`` / ``variance`` / ``entropy`` for closed-form moments).  The
default behaviours mimic the reference framework: methods that aren't
implemented raise :class:`NotImplementedError`, ``sample`` is
``rsample`` detached, and ``stddev = √variance``.

Argument validation belongs to the base, not to each family.  A subclass
declares ``arg_constraints`` and ``support``; the base checks the parameters
in ``__init__`` and the value given to every ``log_prob`` / ``cdf`` /
``icdf`` a subclass defines (:meth:`Distribution.__init_subclass__` wraps
them), so no family can forget to.
"""

import functools
import inspect
from collections.abc import Callable
from contextvars import ContextVar
from typing import override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import Constraint, unit_interval

# The distribution whose validated ``log_prob`` / ``cdf`` / ``icdf`` call is
# running.  A subclass method that calls ``super().log_prob`` — or another of
# its own validated methods — has had its value checked once already, so the
# inner call skips the check.  A context variable rather than an instance
# flag, so two threads scoring with one distribution do not switch each
# other's checks off.
_VALIDATING: ContextVar[object] = ContextVar("_VALIDATING", default=None)


def _validated(
    method: Callable[..., Tensor], *, probability: bool
) -> Callable[..., Tensor]:
    """``method`` checking its value first, when its distribution validates.

    ``probability`` selects the check: an ``icdf`` takes a probability, any
    other scorer a point of the support.  With validation off the wrapper
    costs one attribute read.
    """

    @functools.wraps(method)
    def validated(self: Distribution, value: Tensor) -> Tensor:
        if not self._validate_args or _VALIDATING.get() is self:
            return method(self, value)
        if probability:
            self._validate_probability(value)
        else:
            self._validate_sample(value)
        token = _VALIDATING.set(self)
        try:
            return method(self, value)
        finally:
            _VALIDATING.reset(token)

    return validated


# Every wrapper ``_validated`` builds shares this code object, which is how a
# method already wrapped — inherited by aliasing, say — is told apart.
_VALIDATED_CODE = _validated(lambda _s, v: v, probability=False).__code__


def _is_validated(method: object) -> bool:
    """Whether ``method`` is a scorer the base validates."""
    return getattr(method, "__code__", None) is _VALIDATED_CODE


# The scorers the base validates, and whether each takes a probability.
_SCORERS: dict[str, bool] = {"log_prob": False, "cdf": False, "icdf": True}


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
        If ``True``, the parameters are checked against
        :attr:`arg_constraints` at construction, and the value given to
        :meth:`log_prob`, :meth:`cdf` and :meth:`icdf` is checked before it
        is scored; a violation raises :class:`ValueError`.  ``None`` takes
        the default, which is on and is changed for every distribution by
        :meth:`set_default_validate_args`.  Default is ``None``.

    Attributes
    ----------
    arg_constraints : dict[str, Constraint]
        Maps each constructor parameter name to the
        :class:`~lucid.distributions.constraints.Constraint` it must
        satisfy.  Populated by each concrete subclass.
    support : Constraint or None
        Constraint describing the set on which the distribution has
        positive density (or probability mass).  ``None`` means
        unconstrained.  A family whose support depends on its parameters
        (``Uniform``, ``Binomial``) answers it as a property.
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

    **Validation**

    With validation on, :meth:`log_prob` and :meth:`cdf` refuse a value
    outside :attr:`support`, and every scorer refuses a value whose shape
    does not end in ``event_shape`` or does not broadcast with
    ``batch_shape + event_shape``.  :meth:`icdf` takes a probability, so
    it refuses a value outside :math:`[0, 1]` instead.  That last check is
    a deliberate difference from the reference framework, which does not
    validate an ``icdf`` argument and answers ``nan`` there.  A family whose
    :attr:`support` is ``None`` has declared nothing to check a value
    against, and its ``log_prob`` / ``cdf`` values are not checked.  A
    subclass gets all of this by defining the scorer: it does not call the
    checks itself.

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
    has_rsample: bool = False
    has_enumerate_support: bool = False

    # On by default, matching the reference framework.
    #
    # Off, ``Beta(0.0, 1.0)`` constructed happily and every ``log_prob``
    # against it answered nan — a distribution whose concentration is
    # outside its own declared ``arg_constraints``, reporting nothing.  The
    # machinery to catch that was already here and simply was not armed.
    # A caller who wants the cost back can pass ``validate_args=False``.
    _validate_args: bool = True

    @override
    def __init_subclass__(cls) -> None:
        """Validate the value given to each scorer ``cls`` defines.

        Wrapping in place, rather than having subclasses implement a
        private hook behind a public template, keeps every family's
        ``log_prob`` its own: its signature, docstring and source location
        are unchanged.
        """
        super().__init_subclass__()
        for name, probability in _SCORERS.items():
            method = cls.__dict__.get(name)
            if inspect.isfunction(method) and not _is_validated(method):
                setattr(cls, name, _validated(method, probability=probability))

    @staticmethod
    def set_default_validate_args(value: bool) -> None:
        """Turn argument validation on or off for every distribution.

        Applies to distributions built afterwards without an explicit
        ``validate_args``.  Validation is on by default.

        Parameters
        ----------
        value : bool
            Whether to validate.

        Raises
        ------
        TypeError
            If ``value`` is not a ``bool``.

        Examples
        --------
        >>> import lucid.distributions as dist
        >>> dist.Distribution.set_default_validate_args(False)
        >>> dist.Normal(0.0, -1.0).scale  # not refused
        tensor(-1.)
        >>> dist.Distribution.set_default_validate_args(True)
        """
        if not isinstance(value, bool):
            raise TypeError(
                f"set_default_validate_args: expected a bool, got {type(value).__name__}"
            )
        Distribution._validate_args = value

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

    @property
    def support(self) -> Constraint | None:
        """The set the random variable takes values in.

        A family declares it as a class attribute, or as a property when it
        depends on the parameters.  ``None`` — this default — means no
        value is refused.

        Returns
        -------
        Constraint or None
            The support, or ``None`` when it is unconstrained.
        """
        return None

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
        ValueError
            With validation on, if ``value`` is outside :attr:`support` or
            its shape does not fit the distribution.
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
        ValueError
            Under validation, for a point the distribution cannot take or a
            shape that does not broadcast against it.
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
        ValueError
            With validation on, if ``value`` is outside :math:`[0, 1]` or
            its shape does not fit the distribution.  The reference
            framework does not check an ``icdf`` argument; refusing one
            that is not a probability is a deliberate difference.
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

    def _is_pending(self, name: str) -> bool:
        """Whether ``name`` is the dual of a parameter given in its place.

        ``probs`` of a ``Bernoulli`` built from ``logits`` exists — it is
        derived on access — but it was not an argument, so neither
        validation nor the repr should call it into being.  Validating it
        would also reject arguments the user never passed:
        ``Geometric(logits=20.0)`` derives a probability that rounds to 1 in
        float32, outside the open interval its ``probs`` must lie in.
        """
        from lucid.distributions._util import _lazy_param

        attr = getattr(type(self), name, None)
        return isinstance(attr, _lazy_param) and not attr.is_stored(self)

    def _validate_params(self) -> None:
        """Check that all constructor arguments satisfy their constraints.

        Iterates over :attr:`arg_constraints` and calls
        ``constraint.check(param).all()`` for each tensor-valued parameter.
        Raises :class:`ValueError` on the first violation found.
        Called automatically during ``__init__`` when
        ``validate_args=True``.
        """
        for name, constraint in self.arg_constraints.items():
            if self._is_pending(name) or not hasattr(self, name):
                continue
            v = getattr(self, name)
            if not isinstance(v, lucid.Tensor):
                continue
            if not bool(constraint.check(v).all().item()):
                raise ValueError(
                    f"Expected parameter {name} of shape {tuple(v.shape)} of "
                    f"distribution {type(self).__name__} to satisfy the "
                    f"constraint {constraint!r}"
                )

    def _validate_value_shape(self, value: Tensor) -> None:
        """Check that ``value`` is a tensor a scorer can broadcast.

        Its trailing axes must be ``event_shape``, and the rest must
        broadcast with ``batch_shape``.

        Raises
        ------
        TypeError
            If ``value`` is not a :class:`~lucid.Tensor`.
        ValueError
            If its shape does not end in ``event_shape`` or does not
            broadcast with ``batch_shape + event_shape``.
        """
        if not isinstance(value, Tensor):
            raise TypeError(
                f"{type(self).__name__}: the value to score must be a Tensor, "
                f"got {type(value).__name__}"
            )
        shape = tuple(value.shape)
        event = self._event_shape
        if shape[len(shape) - len(event) :] != event:
            raise ValueError(
                f"{type(self).__name__}: the right-most size of the value must "
                f"match event_shape: {shape} vs {event}"
            )
        expected = self._batch_shape + event
        for got, want in zip(reversed(shape), reversed(expected), strict=False):
            if got != 1 and want != 1 and got != want:
                raise ValueError(
                    f"{type(self).__name__}: the value is not broadcastable with "
                    f"batch_shape + event_shape: {shape} vs {expected}"
                )

    def _validate_sample(self, value: Tensor) -> None:
        """Check that ``value`` can be scored and lies in :attr:`support`.

        Run before every :meth:`log_prob` and :meth:`cdf` when validation is
        on, so a point outside the support raises rather than scoring as
        ``-inf`` or ``nan`` — or, worse, as a finite number.  A distribution
        that declares no support (``None``) declares nothing to check a value
        against, its shape included, and is not checked.

        Parameters
        ----------
        value : Tensor
            The value about to be scored.

        Raises
        ------
        TypeError
            If ``value`` is not a :class:`~lucid.Tensor`.
        ValueError
            If its shape does not fit the distribution, or any element of it
            is outside :attr:`support`.
        """
        support = self.support
        if support is None:
            return
        self._validate_value_shape(value)
        if not bool(support.check(value).all().item()):
            raise ValueError(
                f"Expected value argument of shape {tuple(value.shape)} to be "
                f"within the support ({support!r}) of the distribution {self!r}"
            )

    def _validate_probability(self, value: Tensor) -> None:
        """Check that ``value`` is a probability :meth:`icdf` can invert.

        Raises
        ------
        TypeError
            If ``value`` is not a :class:`~lucid.Tensor`.
        ValueError
            If its shape does not fit the distribution, or any element of it
            is outside :math:`[0, 1]`.
        """
        self._validate_value_shape(value)
        if not bool(unit_interval.check(value).all().item()):
            raise ValueError(
                f"Expected the icdf argument of shape {tuple(value.shape)} of "
                f"{self!r} to be a probability in [0, 1]"
            )

    # ── pretty-printing ────────────────────────────────────────────────────

    @override
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
            if self._is_pending(name) or not hasattr(self, name):
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
    tensor(0.6109)
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
