r"""The sampler — DDIM, and the noise schedule the released models use.

DDPM's ancestral sampler needs one network evaluation per training
timestep, so a thousand-step model costs a thousand forwards. DDIM
(Song et al., ICLR 2021) reinterprets the same trained network as
defining a *non-Markovian* process whose reverse can skip: the update

.. math::

    z_{t-1} = \sqrt{\bar\alpha_{t-1}}\,
        \underbrace{\frac{z_t - \sqrt{1-\bar\alpha_t}\,
        \epsilon_\theta}{\sqrt{\bar\alpha_t}}}_{\hat z_0}
        + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\; \epsilon_\theta
        + \sigma_t \epsilon

is valid for any decreasing subsequence of timesteps, and at
:math:`\eta = 0` the :math:`\sigma_t \epsilon` term vanishes and the
whole trajectory becomes deterministic. That is what makes fifty steps
enough, and what makes the same seed reproduce the same image.

**``scaled_linear`` is not linear.** The released schedule interpolates
linearly in :math:`\sqrt{\beta}` and squares the result:

.. math::

    \beta_t = \Big((1 - \tfrac{t}{T})\sqrt{\beta_0}
        + \tfrac{t}{T}\sqrt{\beta_T}\Big)^2 .

Using a plain linear ramp between the same endpoints gives a schedule
that looks nearly identical plotted, differs most where the signal is
still strong, and denoises to mush. It has its own name here for that
reason, and a test pins the two apart.
"""

from typing import cast, final

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models.generative.stable_diffusion._config import StableDiffusionConfig

__all__ = ["DDIMScheduler"]


@final
class DDIMScheduler:
    r"""Deterministic (or partly stochastic) sampling over a step subset.

    Parameters
    ----------
    config : StableDiffusionConfig
        Read for ``num_train_timesteps``, the betas and the schedule.

    Attributes
    ----------
    alphas_cumprod : Tensor
        :math:`\bar\alpha_t`, ``(num_train_timesteps,)``.

    Notes
    -----
    Reference: Song, Meng and Ermon, *"Denoising Diffusion Implicit
    Models"*, ICLR, 2021 (`arXiv:2010.02502
    <https://arxiv.org/abs/2010.02502>`_); the schedule constants are
    the released Stable Diffusion configuration.

    Kept as a plain object rather than a :class:`lucid.nn.Module`: it
    holds no parameters, and making it a Module would put its buffers in
    the checkpoint of every model that uses one.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     DDIMScheduler, StableDiffusionConfig)
    >>> scheduler = DDIMScheduler(StableDiffusionConfig())
    >>> len(scheduler.timesteps(50))
    50
    >>> scheduler.timesteps(50)[0] > scheduler.timesteps(50)[-1]
    True
    """

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Build the schedule. See the class docstring for parameters."""
        self.config = config
        steps = config.num_train_timesteps
        ramp = lucid.linspace(0.0, 1.0, steps)
        if config.beta_schedule == "scaled_linear":
            root = (1.0 - ramp) * (config.beta_start**0.5) + ramp * (
                config.beta_end**0.5
            )
            betas = root**2
        else:
            betas = (1.0 - ramp) * config.beta_start + ramp * config.beta_end
        self.betas = betas
        self.alphas_cumprod = lucid.cumprod(1.0 - betas, dim=0)

    def timesteps(self, num_inference_steps: int) -> list[int]:
        """The decreasing subsequence to sample over.

        Parameters
        ----------
        num_inference_steps : int
            How many network evaluations to spend.

        Returns
        -------
        list of int
            Descending indices into the training schedule.

        Raises
        ------
        ValueError
            If more steps are requested than the schedule has.
        """
        total = self.config.num_train_timesteps
        if not 1 <= num_inference_steps <= total:
            raise ValueError(
                f"num_inference_steps must be in [1, {total}], got "
                f"{num_inference_steps}"
            )
        stride = total // num_inference_steps
        return [int(i) for i in range(total - 1, -1, -stride)][:num_inference_steps]

    def add_noise(self, latent: Tensor, noise: Tensor, timestep: int) -> Tensor:
        r"""The forward process — :math:`z_t` from :math:`z_0`.

        Parameters
        ----------
        latent : Tensor
            ``(B, C, h, w)`` clean latent.
        noise : Tensor
            The same shape.
        timestep : int
            Index into the training schedule.

        Returns
        -------
        Tensor
            :math:`\sqrt{\bar\alpha_t} z_0 + \sqrt{1-\bar\alpha_t}\epsilon`.
        """
        alpha = float(self.alphas_cumprod[timestep].item())
        return cast(Tensor, alpha**0.5 * latent + (1.0 - alpha) ** 0.5 * noise)

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        previous_timestep: int,
        latent: Tensor,
        eta: float = 0.0,
    ) -> Tensor:
        r"""One reverse step.

        Parameters
        ----------
        model_output : Tensor
            :math:`\epsilon_\theta(z_t, t, \tau_\theta(y))`.
        timestep : int
            Current :math:`t`.
        previous_timestep : int
            The next index in the subsequence, or ``-1`` for the last
            step, where :math:`\bar\alpha` is taken as 1.
        latent : Tensor
            :math:`z_t`.
        eta : float, default=0.0
            Interpolates DDIM (0) to DDPM (1). At 0 the trajectory is
            deterministic, which is what makes a seed reproducible.

        Returns
        -------
        Tensor
            :math:`z_{t-1}`.

        Raises
        ------
        ValueError
            If ``eta`` is outside ``[0, 1]``.
        """
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must lie in [0, 1], got {eta}")
        alpha_t = float(self.alphas_cumprod[timestep].item())
        alpha_prev = (
            float(self.alphas_cumprod[previous_timestep].item())
            if previous_timestep >= 0
            else 1.0
        )

        predicted_x0 = (latent - (1.0 - alpha_t) ** 0.5 * model_output) / alpha_t**0.5
        sigma = (
            eta
            * ((1.0 - alpha_prev) / (1.0 - alpha_t)) ** 0.5
            * (1.0 - alpha_t / alpha_prev) ** 0.5
            if eta > 0.0 and previous_timestep >= 0
            else 0.0
        )
        direction = max(1.0 - alpha_prev - sigma**2, 0.0) ** 0.5 * model_output
        previous = alpha_prev**0.5 * predicted_x0 + direction
        if sigma > 0.0:
            previous = previous + sigma * lucid.randn(
                tuple(int(s) for s in latent.shape),
                device=latent.device.type,
                dtype=latent.dtype,
            )
        return cast(Tensor, previous)
