r"""The three stochastic differential equations of Song et al. (2021).

Score-based generative modelling and denoising diffusion turn out to be
two discretisations of the same object.  This module is that object: a
forward SDE that carries data to noise, the reverse-time SDE that carries
it back, and the deterministic ODE which shares the reverse SDE's
marginals at every instant.

    dx = f(x, t) dt + g(t) dw                                    forward
    dx = [f(x, t) - g(t)^2 * score] dt + g(t) dw̄            reverse (6)
    dx = [f(x, t) - g(t)^2 * score / 2] dt          probability flow (13)

The point of writing it once is that ``VE`` and ``VP`` are not new models
— they are the continuous limits of NCSN and DDPM respectively, both of
which this zoo already ships.  Everything downstream (the samplers, the
loss, the likelihood) is written against the interface and does not know
which of the three it has.
"""

import math
from abc import ABC, abstractmethod
from typing import override

import lucid
from lucid._tensor.tensor import Tensor

__all__ = ["SDE", "VESDE", "VPSDE", "SubVPSDE", "make_sde"]


def _expand(scalar: Tensor, like: Tensor) -> Tensor:
    """Give a per-sample scalar the trailing axes needed to broadcast."""
    return scalar.reshape(int(scalar.shape[0]), *([1] * (like.ndim - 1)))


class SDE(ABC):
    """What the samplers and the loss need an SDE to provide.

    Notes
    -----
    Four things, and no more: the drift and diffusion of the forward
    process, the closed-form perturbation kernel that makes training a
    single network evaluation, and the prior the reverse process starts
    from.  Everything else — reverse SDE, probability-flow ODE,
    Predictor-Corrector — is derived from these and lives with the model.
    """

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """:math:`f(x, t)`, the deterministic part of the forward SDE."""

    @abstractmethod
    def diffusion(self, t: Tensor) -> Tensor:
        """:math:`g(t)`, per sample."""

    @abstractmethod
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        r"""The perturbation kernel :math:`p_{0t}(x(t) \mid x(0))`.

        Parameters
        ----------
        x : Tensor
            The clean sample, ``(N, ...)``.
        t : Tensor
            Times, ``(N,)``.

        Returns
        -------
        mean, std : Tensor
            Mean broadcast to ``x``'s shape, and the per-sample standard
            deviation.  Having this in closed form is what lets the loss
            perturb a sample in one step rather than integrating.
        """

    @abstractmethod
    def prior_sampling(self, shape: tuple[int, ...], device: str) -> Tensor:
        """Draw from the distribution the reverse process starts at."""

    @property
    @abstractmethod
    def t_min(self) -> float:
        """The earliest time the SDE is defined at."""


class VESDE(SDE):
    r"""Variance Exploding — the continuous limit of NCSN.

    Parameters
    ----------
    sigma_min, sigma_max : float
        The geometric noise range.

    Notes
    -----
    Song et al. (2021), equation 30:

    .. math::

        dx = \sigma_{\min}\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t
             \sqrt{2\log\frac{\sigma_{\max}}{\sigma_{\min}}}\; dw,
        \qquad t \in (0, 1],

    with the perturbation kernel of equation 31,
    :math:`\mathcal{N}(x(t); x(0), \sigma(t)^2 I)` — the mean does not
    move, which is what "variance exploding" means.

    Defined on the **open** interval.  The paper is explicit that
    :math:`\sigma` is not differentiable at zero because
    :math:`\sigma(0) = 0` by the discrete convention while
    :math:`\sigma(0^+) = \sigma_{\min}`, "causing the VE SDE ... undefined
    for t = 0".  Sampling therefore stops at :attr:`t_min` rather than
    running to zero.
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0) -> None:
        """Initialise the SDE. See the class docstring for parameters."""
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError(
                f"need 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}"
            )
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._ratio = sigma_max / sigma_min
        self._log_ratio = math.log(self._ratio)

    @override
    @property
    def t_min(self) -> float:
        """Sampling stops here — see the class docstring on ``t = 0``."""
        return 1e-5

    def sigma(self, t: Tensor) -> Tensor:
        r""":math:`\sigma(t) = \sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t`."""
        return self.sigma_min * (self._ratio**t)

    @override
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Zero — the VE process only adds variance."""
        return lucid.zeros_like(x)

    @override
    def diffusion(self, t: Tensor) -> Tensor:
        """Equation 30's coefficient."""
        return self.sigma(t) * math.sqrt(2.0 * self._log_ratio)

    @override
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Equation 31 — the mean is the sample itself."""
        return x, self.sigma(t)

    @override
    def prior_sampling(self, shape: tuple[int, ...], device: str) -> Tensor:
        """:math:`\\mathcal{N}(0, \\sigma_{\\max}^2 I)`."""
        return lucid.randn(shape, device=device) * self.sigma_max


class VPSDE(SDE):
    r"""Variance Preserving — the continuous limit of DDPM.

    Parameters
    ----------
    beta_min, beta_max : float
        The linear rate range.  The paper's values, "to match the
        settings in Ho et al. (2020)".

    Notes
    -----
    Song et al. (2021), equation 32:

    .. math::

        dx = -\tfrac{1}{2}\big(\bar\beta_{\min}
             + t(\bar\beta_{\max} - \bar\beta_{\min})\big) x\, dt
             + \sqrt{\bar\beta_{\min}
             + t(\bar\beta_{\max} - \bar\beta_{\min})}\; dw,

    with the perturbation kernel of equation 33.  Starting from unit
    variance the process keeps it, which is the difference from VE and
    the reason the prior is a standard normal rather than a wide one.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        """Initialise the SDE. See the class docstring for parameters."""
        if not 0.0 < beta_min < beta_max:
            raise ValueError(
                f"need 0 < beta_min < beta_max, got {beta_min}, {beta_max}"
            )
        self.beta_min = beta_min
        self.beta_max = beta_max

    @override
    @property
    def t_min(self) -> float:
        """The paper's sampling epsilon, chosen so the variance matches DDPM's."""
        return 1e-3

    def beta(self, t: Tensor) -> Tensor:
        r""":math:`\beta(t) = \bar\beta_{\min} + t(\bar\beta_{\max} - \bar\beta_{\min})`."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_mean_coeff(self, t: Tensor) -> Tensor:
        """The exponent of equation 33's mean scale."""
        return -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

    @override
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """:math:`-\\tfrac{1}{2}\\beta(t)x`."""
        return -0.5 * _expand(self.beta(t), x) * x

    @override
    def diffusion(self, t: Tensor) -> Tensor:
        """:math:`\\sqrt{\\beta(t)}`."""
        return self.beta(t) ** 0.5

    @override
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Equation 33."""
        log_coeff = self._log_mean_coeff(t)
        mean = _expand(log_coeff.exp(), x) * x
        std = (1.0 - (2.0 * log_coeff).exp()) ** 0.5
        return mean, std

    @override
    def prior_sampling(self, shape: tuple[int, ...], device: str) -> Tensor:
        """:math:`\\mathcal{N}(0, I)`."""
        return lucid.randn(shape, device=device)


class SubVPSDE(VPSDE):
    r"""Sub-VP — the SDE this paper introduces, and its best likelihoods.

    Notes
    -----
    Song et al. (2021), equation 12:

    .. math::

        dx = -\tfrac{1}{2}\beta(t)x\,dt
             + \sqrt{\beta(t)\big(1 - e^{-2\int_0^t\beta(s)ds}\big)}\; dw.

    Same drift as VP, a diffusion damped by how far the process has
    already travelled.  "The variance of the stochastic process induced
    by Eq. (12) is always bounded by the VP SDE at every intermediate
    time step", which is why the name and why the paper reports its best
    bits-per-dimension here.

    The perturbation kernel differs from VP's in exactly one place —
    equation 29 gives ``[1 - e^{-∫β}]^2`` where VP has ``1 - e^{-∫β}`` —
    so the standard deviation is that quantity rather than its square
    root.  Getting that wrong leaves a model that still trains.
    """

    def _integral_beta(self, t: Tensor) -> Tensor:
        r""":math:`\int_0^t \beta(s)\,ds`, in closed form."""
        return 0.5 * t**2 * (self.beta_max - self.beta_min) + t * self.beta_min

    @override
    def diffusion(self, t: Tensor) -> Tensor:
        """Equation 12's damped coefficient."""
        discount = 1.0 - (-2.0 * self._integral_beta(t)).exp()
        return (self.beta(t) * discount) ** 0.5

    @override
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Equation 29's sub-VP row — note the square, not a square root."""
        log_coeff = self._log_mean_coeff(t)
        mean = _expand(log_coeff.exp(), x) * x
        std = 1.0 - (2.0 * log_coeff).exp()
        return mean, std


def make_sde(
    kind: str,
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
) -> SDE:
    """Build one of the three by name.

    Parameters
    ----------
    kind : {"ve", "vp", "subvp"}
        Which SDE.
    sigma_min, sigma_max : float
        VE only.
    beta_min, beta_max : float
        VP and sub-VP.

    Returns
    -------
    SDE
        The requested process.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three.
    """
    if kind == "ve":
        return VESDE(sigma_min, sigma_max)
    if kind == "vp":
        return VPSDE(beta_min, beta_max)
    if kind == "subvp":
        return SubVPSDE(beta_min, beta_max)
    raise ValueError(f"sde_type must be 've', 'vp' or 'subvp', got {kind!r}")
