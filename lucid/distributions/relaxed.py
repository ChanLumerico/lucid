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
    r"""Concrete (Gumbel-sigmoid) relaxation of the Bernoulli distribution.

    ``RelaxedBernoulli(temperature=τ, probs=p)`` defines a continuous
    distribution over :math:`(0, 1)` whose samples are differentiable
    surrogates for Bernoulli samples.  As :math:`\tau \to 0` the distribution
    concentrates on :math:`\{0, 1\}`, recovering the discrete Bernoulli.
    As :math:`\tau \to \infty` the distribution approaches
    :math:`\operatorname{Uniform}(0, 1)`.

    The **Concrete distribution** (Maddison et al. 2017) / **Gumbel-softmax**
    (Jang et al. 2017) trick enables gradient-based optimisation through
    discrete latent variables in variational autoencoders and related models.

    Parameters
    ----------
    temperature : Tensor | float
        Temperature parameter :math:`\tau > 0`.  Controls the sharpness
        of the relaxation.  Small values give near-discrete samples;
        large values give near-uniform samples.
    probs : Tensor | float | None, optional
        Bernoulli success probability :math:`p \in (0, 1)`.  Mutually
        exclusive with ``logits``.
    logits : Tensor | float | None, optional
        Log-odds :math:`l = \log(p/(1-p)) \in \mathbb{R}`.  Mutually
        exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    temperature : Tensor
        Temperature :math:`\tau`.
    probs : Tensor
        Success probability (present when constructed with ``probs``).
    logits : Tensor
        Log-odds (present when constructed with ``logits``).

    Notes
    -----
    **Reparameterised sampling** (Gumbel-sigmoid trick):

    .. math::

        y = \sigma\!\left(\frac{l + g_1 - g_2}{\tau}\right),
        \quad g_1, g_2 \overset{\text{iid}}{\sim} \operatorname{Gumbel}(0, 1)

    where :math:`\sigma(\cdot)` is the sigmoid function.  Gradients
    propagate through both :math:`l` and :math:`\tau`.

    **Log-PDF** (Maddison et al. 2017, Eq. 2):

    .. math::

        \log p(y; l, \tau) = \log\tau + l
                             - (\tau+1)\log\!\left(e^{\tau\,\text{logit}(y) - l} + 1\right)
                             - \tau\,\text{logit}(y)

    where :math:`\text{logit}(y) = \log(y/(1-y))`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import RelaxedBernoulli
    >>> dist = RelaxedBernoulli(temperature=0.5, probs=0.7)
    >>> samples = dist.rsample((100,))
    >>> # Samples are in (0, 1)
    >>> ((samples > 0) & (samples < 1)).all()
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
        """Construct a RelaxedBernoulli (Binary Concrete) distribution.

        Parameters
        ----------
        temperature : Tensor | float
            Temperature :math:`\\lambda > 0` controlling the relaxation tightness.
            As :math:`\\lambda \\to 0`, samples concentrate on :math:`\\{0, 1\\}`
            recovering a hard Bernoulli; as :math:`\\lambda \\to \\infty`, samples
            concentrate around :math:`1/2`.
        probs : Tensor | float | None, optional
            Probability :math:`p \\in (0, 1)` parameter. Mutually exclusive
            with ``logits``.
        logits : Tensor | float | None, optional
            Log-odds :math:`\\log\\frac{p}{1-p}` parameter. Mutually exclusive
            with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
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
        """Lazily resolved log-odds tensor.

        Returns ``self.logits`` when the distribution was constructed from
        logits; otherwise computes :math:`\\log(p/(1-p))` from the stored probs.
        """
        return self.logits if self._is_logits else _probs_to_logits(self.probs)

    @property
    def _probs(self) -> Tensor:
        """Lazily resolved probability tensor in :math:`(0, 1)`.

        Returns ``self.probs`` when the distribution was constructed from
        probs; otherwise applies ``sigmoid`` to the stored logits.
        """
        return self.probs if not self._is_logits else _logits_to_probs(self.logits)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw a reparameterised sample via the Gumbel-sigmoid trick.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples :math:`y \in (0, 1)` of shape
            ``(*sample_shape, *batch_shape)``, with gradients flowing through
            both ``logits`` and ``temperature``.
        """
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
        r"""Log-probability density of the Concrete/RelaxedBernoulli distribution.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`y \in (0, 1)` at which to evaluate the density.

        Returns
        -------
        Tensor
            Log-density values of the same shape as ``value``.
        """
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
    r"""Concrete distribution — a continuous relaxation of OneHotCategorical.

    ``RelaxedOneHotCategorical(temperature=τ, probs=p)`` defines a
    distribution over the open :math:`K`-simplex whose samples are
    differentiable surrogates for one-hot categorical samples.  As
    :math:`\tau \to 0` the distribution concentrates on the :math:`K`
    vertices of the simplex (recovering one-hot samples); as
    :math:`\tau \to \infty` samples approach the uniform distribution over
    the simplex.

    This is the **Gumbel-softmax** distribution of Jang et al. (2017) and the
    **Concrete distribution** of Maddison et al. (2017).  The
    straight-through estimator (``hard=True`` in the sampler) rounds to
    one-hot in the forward pass while using the soft sample for gradients.

    Parameters
    ----------
    temperature : Tensor | float
        Temperature parameter :math:`\tau > 0`.  Smaller values give
        sharper (more discrete) samples.
    probs : Tensor | None, optional
        Probability vector(s) of shape ``(..., K)``.  Normalised internally.
        Mutually exclusive with ``logits``.
    logits : Tensor | None, optional
        Unnormalised log-probability vector(s) of shape ``(..., K)``.
        Mutually exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    temperature : Tensor
        Temperature :math:`\tau`.
    probs : Tensor
        Normalised probability vector (present when constructed with ``probs``).
    logits : Tensor
        Unnormalised log-probability vector (present when constructed with
        ``logits``).

    Notes
    -----
    **Reparameterised sampling** (Gumbel-softmax trick):

    .. math::

        y_k = \frac{\exp\!\bigl((l_k + g_k)/\tau\bigr)}
                    {\sum_j \exp\!\bigl((l_j + g_j)/\tau\bigr)},
        \quad g_k \overset{\text{iid}}{\sim} \operatorname{Gumbel}(0, 1)

    The result :math:`y \in \Delta^{K-1}` (open simplex) is differentiable
    w.r.t. :math:`l` and :math:`\tau`.

    **Log-PDF** (Maddison et al. 2017, Eq. 1):

    .. math::

        \log p(y; l, \tau) = \log\Gamma(K)
                             + (K-1)\log\tau
                             + \sum_k (l_k - (\tau+1)\log y_k)
                             - K \operatorname{logsumexp}_k(l_k - \tau\log y_k)

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import RelaxedOneHotCategorical
    >>> dist = RelaxedOneHotCategorical(
    ...     temperature=0.5,
    ...     probs=lucid.tensor([0.1, 0.4, 0.5]),
    ... )
    >>> samples = dist.rsample((50,))
    >>> samples.shape  # (50, 3) — lies on the open simplex
    (50, 3)
    >>> samples.sum(dim=-1)  # each row sums to ~1
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
        """Construct a RelaxedOneHotCategorical (Concrete) distribution.

        Parameters
        ----------
        temperature : Tensor | float
            Temperature :math:`\\lambda > 0` controlling relaxation tightness.
            As :math:`\\lambda \\to 0`, samples concentrate on the vertices of
            the simplex recovering a hard one-hot Categorical; as
            :math:`\\lambda \\to \\infty`, samples concentrate near the simplex
            centroid :math:`(1/K, \\ldots, 1/K)`.
        probs : Tensor | None, optional
            Probability vector of shape ``(..., K)`` on the K-simplex.
            Rows are automatically normalised to sum to 1.  Mutually exclusive
            with ``logits``.
        logits : Tensor | None, optional
            Unnormalised log-probabilities of shape ``(..., K)``. Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
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
        """Lazily resolved per-category logits.

        Returns ``self.logits`` when the distribution was constructed from
        logits; otherwise returns ``self.probs.log()``.
        """
        if self._is_logits:
            return self.logits
        return self.probs.log()

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw a reparameterised sample via the Gumbel-softmax trick.

        Delegates to :func:`lucid.nn.functional.gumbel_softmax` with
        ``hard=False``, ensuring the output lies strictly inside the open
        :math:`K`-simplex and gradients propagate through both ``logits`` and
        ``temperature``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Samples on the open simplex of shape
            ``(*sample_shape, *batch_shape, K)``.
        """
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
        """Log-probability density of the Concrete distribution over the simplex.

        Parameters
        ----------
        value : Tensor
            Point(s) :math:`y` on the open :math:`K`-simplex, shape
            ``(..., K)``.

        Returns
        -------
        Tensor
            Log-density values, shape ``(...,)`` (batch dimensions only).
        """
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
