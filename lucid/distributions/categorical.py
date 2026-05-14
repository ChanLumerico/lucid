"""``Categorical`` (discrete with K outcomes) and ``OneHotCategorical``."""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.distributions.constraints import (
    Constraint,
    integer_interval,
    real,
    simplex,
)
from lucid.distributions.distribution import Distribution


from lucid.distributions._util import as_tensor as _as_tensor


def _normalize_probs(probs: Tensor) -> Tensor:
    """Project a non-negative tensor onto the K-simplex along the last
    dim (matches the reference framework's preprocessing)."""
    return probs / probs.sum(dim=-1, keepdim=True)


class Categorical(Distribution):
    r"""Categorical distribution — a discrete distribution over K labelled outcomes.

    ``Categorical(probs=p)`` or ``Categorical(logits=l)`` defines a distribution
    over the integer set :math:`\{0, 1, \ldots, K-1\}` where :math:`K` is the
    number of categories.  Exactly one of ``probs`` or ``logits`` must be given.

    Parameters
    ----------
    probs : Tensor | None, optional
        Non-negative probability vector (or batch of vectors) of shape
        ``(..., K)``.  Rows are automatically normalised to sum to 1.
        Mutually exclusive with ``logits``.
    logits : Tensor | None, optional
        Unnormalised log-probabilities of shape ``(..., K)``.
        The distribution uses :math:`\text{softmax}` internally to convert to
        normalised probabilities.  Mutually exclusive with ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    probs : Tensor
        Normalised probability vector (shape ``(..., K)``; present when
        constructed with ``probs``).
    logits : Tensor
        Unnormalised log-probability vector (shape ``(..., K)``; present when
        constructed with ``logits``).

    Notes
    -----
    **PMF**:

    .. math::

        P(X = k) = p_k, \quad k \in \{0, 1, \ldots, K-1\},
        \quad \sum_{k} p_k = 1

    **Parameterisations** are related by:

    .. math::

        p_k = \frac{e^{l_k}}{\sum_j e^{l_j}}, \qquad
        l_k = \log p_k \;(\text{up to additive constant})

    **Entropy**:

    .. math::

        H[X] = -\sum_{k=0}^{K-1} p_k \log p_k

    **Mean** is not well-defined for a general Categorical (the labels have no
    canonical metric), so ``mean`` returns a ``NaN`` tensor of the batch shape.

    **Sampling** uses the **Gumbel-max trick**: add i.i.d.
    :math:`\operatorname{Gumbel}(0, 1)` noise to the log-probabilities and
    take the argmax.  This is equivalent to ancestral sampling and avoids
    cumulative-sum + binary-search.

    The batch dimensions of the input correspond to independent distributions.
    For example, ``probs`` of shape ``(B, K)`` yields a batch of :math:`B`
    Categorical distributions.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import Categorical
    >>> # Uniform over 4 categories
    >>> dist = Categorical(probs=lucid.tensor([0.25, 0.25, 0.25, 0.25]))
    >>> samples = dist.sample((10,))
    >>> # Batch of 2 distributions
    >>> dist_b = Categorical(logits=lucid.zeros(2, 5))
    >>> dist_b.batch_shape, dist_b.event_shape
    ((2,), ())
    """

    arg_constraints = {"probs": simplex, "logits": real}
    has_enumerate_support = True

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise a Categorical distribution.

        Parameters
        ----------
        probs : Tensor | None, optional
            Non-negative probability vector of shape ``(..., K)``.  Rows are
            automatically normalised to sum to 1.  Mutually exclusive with
            ``logits``.
        logits : Tensor | None, optional
            Unnormalised log-probabilities of shape ``(..., K)``.  Converted
            to probabilities via softmax internally.  Mutually exclusive with
            ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        if (probs is None) == (logits is None):
            raise ValueError("Categorical: pass exactly one of `probs` or `logits`.")
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
            event_shape=(),
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        r"""Support of the distribution: integer interval :math:`\{0, \ldots, K-1\}`.

        Returns
        -------
        Constraint
            An ``integer_interval`` constraint from ``0`` to ``K - 1``.
        """
        return integer_interval(0, self._num_events - 1)

    @property
    def _log_probs(self) -> Tensor:
        """Lazily resolved log-probability tensor over the ``K`` categories.

        Applies ``log_softmax`` to ``self.logits`` when the distribution was
        constructed from logits; otherwise returns ``self.probs.log()``.
        """
        if self._is_logits:
            from lucid.nn.functional.activations import log_softmax

            return log_softmax(self.logits, dim=-1)
        return self.probs.log()

    @property
    def _probs(self) -> Tensor:
        """Lazily resolved probability tensor (summing to 1 along the last axis).

        Applies ``softmax`` to ``self.logits`` when the distribution was
        constructed from logits; otherwise returns ``self.probs`` directly.
        """
        if self._is_logits:
            from lucid.nn.functional.activations import softmax

            return softmax(self.logits, dim=-1)
        return self.probs

    @property
    def mean(self) -> Tensor:
        """Mean of the Categorical distribution (undefined — returns NaN).

        The Categorical distribution assigns labels with no inherent ordering
        or metric, so the mean is not well-defined.  This property returns a
        ``NaN`` tensor of the batch shape to match expected behaviour.

        Returns
        -------
        Tensor
            Tensor of ``float('nan')`` values with shape ``batch_shape``.
        """
        # mean of Categorical isn't well-defined (no metric on labels)
        # but reference framework returns NaN with the right shape.
        return lucid.full(
            self._batch_shape,
            float("nan"),
            device=self._probs.device,
            dtype=self._probs.dtype,
        )

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        r"""Draw samples from the Categorical distribution.

        Uses the **Gumbel-max trick**: add i.i.d.
        :math:`\operatorname{Gumbel}(0, 1)` noise to the log-probabilities and
        take the argmax, which is equivalent to ancestral sampling but avoids
        cumulative-sum and binary search.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Integer tensor of shape ``(*sample_shape, *batch_shape)`` with
            values in :math:`\{0, 1, \ldots, K-1\}`.  The result is detached
            (no gradients flow through discrete samples).
        """
        # Gumbel-max trick, no replacement: draw G ~ −log(−log U), pick argmax.
        shape = tuple(sample_shape) + tuple(self._batch_shape) + (self._num_events,)
        u = lucid.rand(*shape, dtype=self._probs.dtype, device=self._probs.device)
        u = u.clip(1e-7, 1.0 - 1e-7)
        gumbel = -(-(u.log())).log()
        scores = self._log_probs + gumbel
        return scores.argmax(dim=-1).detach()

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of the given category indices.

        For a one-hot index :math:`k`, the log-probability is:

        .. math::

            \log P(X = k) = \log p_k

        Parameters
        ----------
        value : Tensor
            Integer tensor of category indices with shape compatible with
            ``batch_shape``.  Values must be in :math:`\{0, \ldots, K-1\}`.

        Returns
        -------
        Tensor
            Log-probabilities of shape ``batch_shape``.
        """
        log_p = self._log_probs
        # Reshape value so it matches log_p[:-1] exactly, then add a
        # trailing length-1 axis for the ``gather`` index.
        target_shape = tuple(log_p.shape[:-1])
        v = value
        if tuple(v.shape) != target_shape:
            if v.ndim == 0 or v.shape == (1,):
                v = lucid.full(
                    target_shape,
                    float(v.item()),
                    dtype=v.dtype,
                    device=v.device,
                )
            else:
                v = v + lucid.zeros(target_shape, dtype=v.dtype, device=v.device)
        v_long = v.to(lucid.int64).unsqueeze(-1)
        gathered = lucid.gather(log_p, v_long, dim=-1)
        return gathered.squeeze(-1)

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the Categorical distribution.

        .. math::

            H[X] = -\sum_{k=0}^{K-1} p_k \log p_k

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).
        """
        log_p = self._log_probs
        return -(self._probs * log_p).sum(dim=-1)


class OneHotCategorical(Distribution):
    r"""Categorical distribution with one-hot encoded samples.

    ``OneHotCategorical`` wraps a :class:`Categorical` and returns samples
    as one-hot vectors of shape ``(..., K)`` instead of integer indices.
    It is particularly useful for:

    - **REINFORCE-style gradient estimators** where you need a discrete
      sample but want to use it in differentiable downstream computation.
    - **Relaxations** — replacing with :class:`RelaxedOneHotCategorical`
      gives a differentiable approximation that converges to one-hot as
      temperature :math:`\to 0`.

    Parameters
    ----------
    probs : Tensor | None, optional
        Non-negative probability vector ``(..., K)``.  Normalised internally.
        Mutually exclusive with ``logits``.
    logits : Tensor | None, optional
        Unnormalised log-probabilities ``(..., K)``.  Mutually exclusive with
        ``probs``.
    validate_args : bool | None, optional
        If ``True``, validate parameter constraints at construction time.

    Attributes
    ----------
    probs : Tensor
        Normalised probability vector (present when constructed with ``probs``).
    logits : Tensor
        Unnormalised log-probability vector (present when constructed with
        ``logits``).

    Notes
    -----
    **Samples** are integer-valued one-hot vectors in :math:`\{0, 1\}^K`
    with exactly one 1 at the sampled category index.

    **Log-probability** for a one-hot vector :math:`e_k`:

    .. math::

        \log P(X = e_k) = \sum_{j} e_{kj} \log p_j = \log p_k

    which is simply the log-probability of the selected category.

    **Entropy** equals that of the underlying :class:`Categorical`:

    .. math::

        H[X] = -\sum_{k} p_k \log p_k

    The ``event_shape`` is ``(K,)`` whereas :class:`Categorical` has
    ``event_shape = ()``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.distributions import OneHotCategorical
    >>> dist = OneHotCategorical(probs=lucid.tensor([0.1, 0.5, 0.4]))
    >>> sample = dist.sample()
    >>> sample.shape  # (3,) — one-hot
    (3,)
    >>> sample.sum()  # always 1
    """

    arg_constraints = {"probs": simplex, "logits": real}

    def __init__(
        self,
        probs: Tensor | None = None,
        logits: Tensor | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Initialise a OneHotCategorical distribution.

        Parameters
        ----------
        probs : Tensor | None, optional
            Non-negative probability vector of shape ``(..., K)``.  Normalised
            internally.  Mutually exclusive with ``logits``.
        logits : Tensor | None, optional
            Unnormalised log-probabilities of shape ``(..., K)``.  Mutually
            exclusive with ``probs``.
        validate_args : bool | None, optional
            If ``True``, validate parameter constraints at construction time.

        Raises
        ------
        ValueError
            If both or neither of ``probs`` and ``logits`` are provided.
        """
        self._cat = Categorical(probs=probs, logits=logits, validate_args=False)
        if probs is not None:
            self.probs = self._cat.probs
        else:
            self.logits = self._cat.logits
        super().__init__(
            batch_shape=tuple(self._cat._batch_shape),
            event_shape=(self._cat._num_events,),
            validate_args=validate_args,
        )

    @property
    def support(self) -> Constraint:  # type: ignore[override]
        """Support of the distribution: the probability simplex.

        Returns
        -------
        Constraint
            The ``simplex`` constraint, as each sample is a one-hot vector
            whose entries are non-negative and sum to 1.
        """
        return simplex

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Draw one-hot encoded samples.

        Internally samples category indices from the underlying
        :class:`Categorical` distribution and converts them to one-hot
        vectors via ``one_hot``.

        Parameters
        ----------
        sample_shape : tuple[int, ...], optional
            Leading shape of the output sample batch.

        Returns
        -------
        Tensor
            Float tensor of shape ``(*sample_shape, *batch_shape, K)``
            containing one-hot vectors (exactly one 1 per row).
        """
        idx = self._cat.sample(sample_shape)
        from lucid.nn.functional.sparse import one_hot

        return one_hot(idx, num_classes=self._cat._num_events).to(
            self._cat._probs.dtype
        )

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Log-probability of a one-hot encoded sample.

        For a one-hot vector :math:`e_k` (all zeros except a 1 at position
        :math:`k`):

        .. math::

            \log P(X = e_k) = \sum_j e_{kj} \log p_j = \log p_k

        Parameters
        ----------
        value : Tensor
            One-hot tensor of shape ``(..., K)`` with ``float`` dtype.

        Returns
        -------
        Tensor
            Log-probabilities of shape ``batch_shape``.
        """
        # value is one-hot — log_prob = sum(value * log_probs).
        return (value * self._cat._log_probs).sum(dim=-1)

    def entropy(self) -> Tensor:
        r"""Shannon entropy of the OneHotCategorical distribution.

        Equal to the entropy of the underlying :class:`Categorical`:

        .. math::

            H[X] = -\sum_{k=0}^{K-1} p_k \log p_k

        Returns
        -------
        Tensor
            Entropy values of shape ``batch_shape`` (nats).
        """
        return self._cat.entropy()
