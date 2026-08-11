"""Noise schedulers for diffusion-family generative models.

Mirrors the diffusers-style separation: the *scheduler* owns the forward
process (``add_noise``) and the reverse-process step (``step``), while the
*model* owns the parametrised denoiser.  This keeps sampler swaps cheap —
DDPM vs DDIM vs Karras only requires re-instantiating the scheduler.

Today this module ships:

    * :class:`DiffusionScheduler` — abstract base (contract every scheduler honours)
    * :class:`DDPMScheduler` — Ho et al. (2020) ancestral sampler
"""

from abc import ABC, abstractmethod
from typing import override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._utils._generative import extract_into_tensor, make_beta_schedule

__all__ = ["DiffusionScheduler", "DDPMScheduler"]


class DiffusionScheduler(ABC):
    """Abstract base — every diffusion scheduler honours this contract.

    Attributes
    ----------
    num_train_timesteps : int
        Length of the training-time forward process ``T``.
    timesteps : Tensor
        Currently-active inference schedule (descending integer indices into
        the training schedule).  Set by :meth:`set_timesteps`.
    """

    num_train_timesteps: int
    timesteps: Tensor

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, *, device: str = "cpu") -> None:
        """Populate :attr:`timesteps` for the requested inference schedule.

        Most schedulers sub-sample the training schedule uniformly when
        ``num_inference_steps < num_train_timesteps`` (DDPM canonical)
        or construct a different grid entirely (Karras / Heun).

        Parameters
        ----------
        num_inference_steps : int
            How many denoising steps the sampler will perform.  Must be
            positive and (for sub-sampling schedulers) ``<=
            num_train_timesteps``.
        device : str, optional
            Placement of the materialised :attr:`timesteps` tensor.
            Default ``"cpu"``.
        """

    @abstractmethod
    def add_noise(self, original: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Forward process: ``q(x_t | x_0) = N(√ᾱ_t · x_0, (1 - ᾱ_t) · I)``."""

    @abstractmethod
    def step(self, model_output: Tensor, timestep: int, sample: Tensor) -> Tensor:
        """One reverse-process step: ``x_t → x_{t-1}`` given the network output."""


class DDPMScheduler(DiffusionScheduler):
    """Ho et al. (2020) ancestral DDPM sampler.

    Reverse process:
        ``x_{t-1} = (1/√α_t) · (x_t - (β_t / √(1-ᾱ_t)) · ε̂_θ(x_t, t)) + σ_t · z``
        where ``σ_t² = β_t`` (Algorithm 2).  Section 3.2 offers two choices;
    this is the *upper* bound — ``fixedlarge`` in the official code, and
    what ``run_cifar.py`` sets.  The lower bound
    ``σ_t² = β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t`` is the one other
    libraries call "fixed_small".

    Args:
        num_train_timesteps: Length of forward process ``T``.
        beta_start / beta_end / beta_schedule: see
            :func:`make_beta_schedule`.
        prediction_type: ``"epsilon"`` (the noise, default) or ``"sample"``
            (predict x_0 directly).  Both parameterisations are supported.

    Notes:
        The scheduler is *stateless w.r.t. weights* — it only owns the noise
        schedule.  Multiple schedulers (DDPM / DDIM / …) can sample from
        the same trained network without re-loading anything.
    """

    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor

    def __init__(
        self,
        num_train_timesteps: int = 1_000,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        clip_denoised: bool = True,
    ) -> None:
        if prediction_type not in ("epsilon", "sample"):
            raise ValueError(
                f"DDPMScheduler.prediction_type must be 'epsilon' or 'sample', "
                f"got {prediction_type!r}"
            )
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_denoised = clip_denoised
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = make_beta_schedule(
            num_train_timesteps,
            beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.alphas = 1.0 - self.betas
        # ᾱ_t = ∏_{s≤t} α_s — implemented as a Python prefix product because
        # cumulative-product over a 1-D 1000-entry tensor is one-shot cheap.
        a_list = [float(self.alphas[i].item()) for i in range(num_train_timesteps)]
        cum: list[float] = []
        acc = 1.0
        for a in a_list:
            acc *= a
            cum.append(acc)
        self.alphas_cumprod = lucid.tensor(cum)

        # Inference schedule defaults to the full training one.
        self.timesteps = lucid.arange(num_train_timesteps - 1, -1, -1).long()
        # Spacing between consecutive entries of ``self.timesteps``.  ``step()``
        # needs it: with a sub-sampled schedule the previous timestep is
        # ``t - step_ratio``, not ``t - 1``, and the per-step α/β must be the
        # ratio across that whole interval.
        self._step_ratio = 1

    @override
    def set_timesteps(self, num_inference_steps: int, *, device: str = "cpu") -> None:
        """Uniform-subsample the training schedule into ``num_inference_steps``."""
        if num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {num_inference_steps}"
            )
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) exceeds "
                f"num_train_timesteps ({self.num_train_timesteps})"
            )
        step = self.num_train_timesteps // num_inference_steps
        # Anchor the grid at 0 and count *up* by ``step``, then reverse:
        # [(n-1)·step, …, step, 0].  Anchoring at T-1 instead leaves the last
        # entry at ``T-1 - (n-1)·step``, which is only 0 when ``n`` divides
        # ``T``; otherwise the chain stops at some t > 0 and hands back a
        # sample that is still noisy — and, because ``prev_t`` is then >= 0,
        # adds one more σ·z on the way out.  Ending at exactly 0 makes the
        # final step noiseless for every ``n``, which is Algorithm 2 line 3
        # ("z = 0 if t == 1").
        idx = [i * step for i in range(num_inference_steps)][::-1]
        self.timesteps = lucid.tensor(idx, device=device).long()
        self._step_ratio = step

    @override
    def add_noise(self, original: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Forward process at arbitrary timesteps.

        Args:
            original:  ``(B, C, H, W)`` clean images ``x_0``.
            noise:     ``(B, C, H, W)`` Gaussian noise drawn from ``N(0, I)``.
            timesteps: ``(B,)`` int diffusion steps in ``[0, T)``.

        Returns:
            Noisy images ``x_t`` of the same shape as ``original``.
        """
        sqrt_ab = extract_into_tensor(
            self.alphas_cumprod**0.5, timesteps, original.shape
        )
        sqrt_one_minus_ab = extract_into_tensor(
            (1.0 - self.alphas_cumprod) ** 0.5, timesteps, original.shape
        )
        return sqrt_ab * original + sqrt_one_minus_ab * noise

    @override
    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
    ) -> Tensor:
        """Single reverse-process step at integer ``timestep``.

        Args:
            model_output: ``(B, C, H, W)`` denoiser prediction.  Interpreted
                per ``self.prediction_type``.
            timestep:     Current integer ``t``.
            sample:       Current noisy sample ``x_t``.

        Returns:
            ``x_{t-1}`` (still noisy except at ``t == 0``, where mean only).
        """
        t = timestep
        # The previous entry of the *inference* schedule, which is ``t - 1``
        # only when the schedule is the untouched training one.  Under
        # sub-sampling the reverse step spans ``step_ratio`` training steps, so
        # α and β for the step are the cumulative-product ratio across the whole
        # interval — using ``self.alphas[t]`` / ``self.betas[t]`` there would
        # under-denoise by the stride factor and return near-pure noise.
        prev_t = t - self._step_ratio
        ab_t = float(self.alphas_cumprod[t].item())
        ab_prev = float(self.alphas_cumprod[prev_t].item()) if prev_t >= 0 else 1.0

        alpha_t = ab_t / ab_prev
        beta_t = 1.0 - alpha_t

        if self.prediction_type == "epsilon":
            # x_0 reconstruction from predicted noise.
            x0_pred = (sample - (1.0 - ab_t) ** 0.5 * model_output) / (ab_t**0.5)
        else:  # "sample"
            x0_pred = model_output

        if self.clip_denoised:
            # DDPM samples live in [-1, 1]; without this the 1/sqrt(ᾱ_t) factor
            # (≈157 at t=999) lets an early x̂₀ blow the chain up.
            x0_pred = lucid.clip(x0_pred, -1.0, 1.0)

        # Posterior mean q(x_{t-1} | x_t, x_0).
        coef_x0 = (ab_prev**0.5) * beta_t / (1.0 - ab_t)
        coef_xt = (alpha_t**0.5) * (1.0 - ab_prev) / (1.0 - ab_t)
        mean: Tensor = coef_x0 * x0_pred + coef_xt * sample

        if prev_t < 0:
            return mean
        # Add Gaussian noise scaled by posterior variance σ_t² = β_t (fixed-small).
        sigma = beta_t**0.5
        z = lucid.randn(sample.shape, device=sample.device.type)
        out: Tensor = mean + sigma * z
        return out
