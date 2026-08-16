r"""The three quantities DreamerV3 changes about how a world model is scored.

Each one exists to remove a number the practitioner would otherwise have
had to supply per domain, and each is written here rather than inline
because each has a failure mode worth naming in one place.
"""

from typing import cast

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models.generative._rssm import RSSMState, categorical_kl

__all__ = ["free_bits_kl", "percentile", "ReturnNormaliser"]


def free_bits_kl(
    posterior: RSSMState,
    prior: RSSMState,
    *,
    dyn_scale: float = 1.0,
    rep_scale: float = 0.1,
    free_nats: float = 1.0,
) -> tuple[Tensor, Tensor]:
    r"""The divergence, split by direction and floored per step.

    Parameters
    ----------
    posterior, prior : RSSMState
        Both categorical, from the same step.
    dyn_scale, rep_scale : float, default=1.0, 0.1
        :math:`\beta_{\mathrm{dyn}}` and :math:`\beta_{\mathrm{rep}}`.
    free_nats : float, default=1.0
        The floor.  Below it a term contributes a constant, so no gradient
        is spent closing a gap that is already small.

    Returns
    -------
    dynamics, representation : Tensor
        Scalars, already scaled.  Returned apart because they are the two
        halves the paper reports separately and the pair is what a reader
        checks when a latent collapses.

    Raises
    ------
    ValueError
        If either state is Gaussian.

    Notes
    -----
    .. math::

        \mathcal{L}_{\mathrm{dyn}} = \max\big(\text{free},\;
            \mathrm{KL}[\mathrm{sg}(q) \,\|\, p]\big),
        \qquad
        \mathcal{L}_{\mathrm{rep}} = \max\big(\text{free},\;
            \mathrm{KL}[q \,\|\, \mathrm{sg}(p)]\big).

    The clip is applied **per step, before averaging**, which is the whole
    point and easy to get backwards.  Clipping the mean lets a batch where
    most steps are already below the floor be dragged by a few that are
    not; clipping each step first means those steps simply stop
    contributing.  DreamerV2's ``free_nats`` clipped the mean, and that
    difference is part of why this one does not need a tuned KL scale.
    """
    if not (posterior.is_discrete and prior.is_discrete):
        raise ValueError("free bits here expects two categorical states")
    if free_nats < 0.0:
        raise ValueError(f"free_nats must be non-negative, got {free_nats}")

    posterior_logits = cast(Tensor, posterior.logits)
    prior_logits = cast(Tensor, prior.logits)

    toward_prior = categorical_kl(posterior_logits.detach(), prior_logits)
    toward_posterior = categorical_kl(posterior_logits, prior_logits.detach())

    def floored(value: Tensor) -> Tensor:
        return value.clip(free_nats, None).mean()

    return dyn_scale * floored(toward_prior), rep_scale * floored(toward_posterior)


def percentile(values: Tensor, fraction: float) -> Tensor:
    r"""The ``fraction``-th percentile of a flattened tensor.

    Parameters
    ----------
    values : Tensor
        Any shape; flattened first.
    fraction : float
        In ``[0, 100]``.

    Returns
    -------
    Tensor
        A scalar, by nearest-rank on the sorted values.

    Notes
    -----
    Nearest-rank rather than interpolated.  The consumer is a scale
    estimate smoothed by a long moving average, so the difference between
    the two conventions is far below the noise the average is there to
    remove, and nearest-rank needs no gather.
    """
    if not 0.0 <= fraction <= 100.0:
        raise ValueError(f"fraction must be in [0, 100], got {fraction}")
    flat = values.reshape(-1)
    count = int(flat.shape[0])
    ordered = lucid.sort(flat)
    index = min(count - 1, max(0, int(round(fraction / 100.0 * (count - 1)))))
    return ordered[index]


class ReturnNormaliser:
    r"""A slow estimate of how wide the returns currently are.

    Parameters
    ----------
    decay : float, default=0.99
        Weight kept by the running estimate at each update.
    low, high : float, default=5.0, 95.0
        Percentiles whose difference is the spread.
    floor : float, default=1.0
        The divisor is ``max(floor, spread)``.

    Notes
    -----
    DreamerV3 divides the actor's objective by

    .. math::

        S = \mathrm{EMA}\big(\mathrm{Per}(R^\lambda, 95)
                           - \mathrm{Per}(R^\lambda, 5),\; 0.99\big),

    which is what lets a single entropy coefficient mean the same thing
    in a domain where returns are 0.01 and one where they are 10000 —
    the bonus is traded against a quantity that has been made
    dimensionless.

    Two details carry the design.  The **percentile range** rather than
    the standard deviation, because a handful of enormous returns should
    not shrink every other gradient.  And the **floor at one**, because
    dividing by a genuinely small spread would amplify what is mostly
    noise: below that width the returns are left in their own units.

    Not a module — it holds one float and no parameters, and making it
    one would put a non-learnable scalar into ``state_dict`` where a
    reader would reasonably expect weights.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer_v3._objectives import ReturnNormaliser
    >>> norm = ReturnNormaliser()
    >>> scaled = norm(lucid.randn((4, 8)))
    >>> scaled.shape
    (4, 8)
    """

    def __init__(
        self,
        decay: float = 0.99,
        low: float = 5.0,
        high: float = 95.0,
        floor: float = 1.0,
    ) -> None:
        """Initialise the estimate. See the class docstring for parameters."""
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(
                f"percentiles must satisfy 0 <= low < high <= 100, got {low}, {high}"
            )
        if floor <= 0.0:
            raise ValueError(f"floor must be positive, got {floor}")
        self.decay = decay
        self.low = low
        self.high = high
        self.floor = floor
        self.spread = 0.0

    @lucid.no_grad()
    def update(self, returns: Tensor) -> float:
        """Fold this batch's spread into the estimate.

        Parameters
        ----------
        returns : Tensor
            Lambda-returns, any shape.

        Returns
        -------
        float
            The divisor after the update — ``max(floor, spread)``.
        """
        observed = float(
            (percentile(returns, self.high) - percentile(returns, self.low)).item()
        )
        self.spread = self.decay * self.spread + (1.0 - self.decay) * observed
        return max(self.floor, self.spread)

    def __call__(self, returns: Tensor) -> Tensor:
        """Update the estimate and return the scaled returns.

        Parameters
        ----------
        returns : Tensor
            Lambda-returns.

        Returns
        -------
        Tensor
            ``returns / max(floor, spread)`` — the divisor is a plain
            float, so nothing differentiates through the normalisation.
        """
        return returns / self.update(returns)
