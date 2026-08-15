"""A truncated normal, for the actor that has to stay inside its box.

Dreamer squashed its Gaussian with ``tanh``; DreamerV2 truncates one
instead.  The difference matters for the entropy bonus this family adds:
a squashed normal has no closed-form entropy — the released implementation
estimates it by sampling — whereas a truncated one does, so the bonus is
exact and costs nothing.

Kept private to the family.  ``lucid.distributions`` is where a truncated
normal ought to eventually live, and it is missing one, but that module
has a contract of its own — constraints, registered divergences, transform
composition — and adding a half-conforming member to it would be worse
than this.  Promote it when a second consumer appears.
"""

import math

import lucid
from lucid._tensor.tensor import Tensor

__all__ = ["TruncatedNormal"]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _standard_pdf(x: Tensor) -> Tensor:
    """Standard normal density."""
    return lucid.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _standard_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF, via ``erf``."""
    return 0.5 * (1.0 + lucid.erf(x / _SQRT_2))


class TruncatedNormal:
    r"""A normal restricted to ``[low, high]`` and renormalised.

    Parameters
    ----------
    loc, scale : Tensor
        The underlying normal's parameters, broadcast together.
    low, high : float, default=-1.0, 1.0
        The interval kept.
    epsilon : float, default=1e-6
        Floor on the surviving probability mass, and the margin samples
        are held away from the endpoints.  Without it a ``scale`` driven
        small next to a bound leaves ``Z`` at zero and every quantity
        divides by it.

    Notes
    -----
    Writing :math:`\alpha = (a - \mu)/\sigma`, :math:`\beta = (b - \mu)/\sigma`
    and :math:`Z = \Phi(\beta) - \Phi(\alpha)`:

    .. math::

        \log p(x) = -\tfrac{1}{2}\Big(\tfrac{x - \mu}{\sigma}\Big)^2
                    - \log \sigma - \tfrac{1}{2}\log 2\pi - \log Z,

    .. math::

        H = \log\!\big(\sqrt{2\pi e}\, \sigma Z\big)
            + \frac{\alpha\,\phi(\alpha) - \beta\,\phi(\beta)}{2 Z}.

    Sampling inverts the CDF, so it is reparameterised: a uniform draw is
    mapped through :math:`\Phi^{-1}`, and the gradient reaches ``loc`` and
    ``scale`` through the affine step that follows.  That is what lets the
    actor's gradient come through the dynamics.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.dreamer_v2._dists import TruncatedNormal
    >>> dist = TruncatedNormal(lucid.zeros((2, 3)), lucid.ones((2, 3)))
    >>> sample = dist.rsample()
    >>> bool((sample.abs() <= 1.0).all().item())
    True
    >>> dist.entropy().shape
    (2, 3)
    """

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: float = -1.0,
        high: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialise the distribution. See the class docstring."""
        if high <= low:
            raise ValueError(f"high must exceed low, got low={low}, high={high}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.epsilon = epsilon

        self._alpha = (low - loc) / scale
        self._beta = (high - loc) / scale
        self._cdf_alpha = _standard_cdf(self._alpha)
        self._z = (_standard_cdf(self._beta) - self._cdf_alpha).clip(epsilon, None)

    @property
    def mean(self) -> Tensor:
        r"""The truncated mean, :math:`\mu + \sigma (\phi(\alpha) - \phi(\beta)) / Z`."""
        shift = (_standard_pdf(self._alpha) - _standard_pdf(self._beta)) / self._z
        return self.loc + self.scale * shift

    @property
    def mode(self) -> Tensor:
        """The density's peak — ``loc``, clipped into the interval."""
        return self.loc.clip(self.low + self.epsilon, self.high - self.epsilon)

    def rsample(self) -> Tensor:
        """Draw reparameterised, so gradients reach ``loc`` and ``scale``.

        Returns
        -------
        Tensor
            A sample strictly inside ``(low, high)``.

        Notes
        -----
        Inverse-CDF sampling: a uniform draw is placed inside the surviving
        mass and pushed back through :math:`\\Phi^{-1}`.  The result is
        clipped a hair away from the endpoints, because a sample landing
        exactly on one makes ``log_prob`` there finite but its gradient
        enormous.
        """
        shape = tuple(int(s) for s in self.loc.shape)
        uniform = lucid.rand(shape, device=self.loc.device, dtype=self.loc.dtype)
        probability = (self._cdf_alpha + uniform * self._z).clip(
            self.epsilon, 1.0 - self.epsilon
        )
        standard = _SQRT_2 * lucid.erfinv(2.0 * probability - 1.0)
        drawn = self.loc + self.scale * standard
        return drawn.clip(self.low + self.epsilon, self.high - self.epsilon)

    def log_prob(self, value: Tensor) -> Tensor:
        """Log density at ``value``.

        Parameters
        ----------
        value : Tensor
            Broadcastable against ``loc``.  Values outside the interval are
            not special-cased; they are the caller's error, and clipping
            them would hide it behind a plausible number.

        Returns
        -------
        Tensor
            Elementwise log density.
        """
        standard = (value - self.loc) / self.scale
        return (
            -0.5 * standard * standard
            - self.scale.log()
            - _LOG_SQRT_2PI
            - self._z.log()
        )

    def entropy(self) -> Tensor:
        r"""Differential entropy, in closed form.

        Returns
        -------
        Tensor
            Elementwise entropy.

        Notes
        -----
        This is why the family truncates rather than squashing: a
        ``tanh``-squashed normal has no closed-form entropy, so the
        released Dreamer estimates it from samples, and an entropy bonus
        built on an estimate is noisier than the thing it regularises.
        """
        correction = (
            self._alpha * _standard_pdf(self._alpha)
            - self._beta * _standard_pdf(self._beta)
        ) / (2.0 * self._z)
        return (
            0.5 * math.log(2.0 * math.pi * math.e)
            + self.scale.log()
            + self._z.log()
            + correction
        )
