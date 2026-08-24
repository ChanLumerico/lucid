r"""Stable Diffusion — the two stages, assembled.

The pieces are in their own modules because each is a model in its own
right: :mod:`._autoencoder` learns the latent space, :mod:`._unet`
denoises inside it, :mod:`._scheduler` decides which timesteps to visit.
What this file adds is the wiring, and the wiring has two decisions in it
that no shape check can see.

**The latent is rescaled.**  The first stage's latents have a standard
deviation of roughly 5, and the diffusion process assumes something near
1.  So the encoder's output is multiplied by a scaling factor before the
U-Net ever sees it, and divided back before decoding.  Omit it and the
forward process's noise is negligible against the signal; the model
trains, and samples nothing.

**Classifier-free guidance runs the U-Net twice.**  Once on the prompt
and once on an empty conditioning, then extrapolates away from the
empty one:

.. math::

    \tilde\epsilon = \epsilon_\varnothing
        + s\,(\epsilon_{\text{cond}} - \epsilon_\varnothing).

At :math:`s = 1` this is exactly the conditional prediction, which is
the identity worth testing — a guidance implementation that ignores its
scale still produces images.
"""

from dataclasses import dataclass
from typing import ClassVar, cast, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ModelOutput
from lucid.models.generative.stable_diffusion._autoencoder import AutoencoderKL
from lucid.models.generative.stable_diffusion._config import StableDiffusionConfig
from lucid.models.generative.stable_diffusion._scheduler import DDIMScheduler
from lucid.models.generative.stable_diffusion._unet import UNet2DConditionModel

__all__ = [
    "StableDiffusionModel",
    "StableDiffusionForImageGeneration",
    "StableDiffusionOutput",
]

# The released first stage's scaling constant.  Not a tuned value: it is
# 1/std of the latents the published autoencoder produces.
_LATENT_SCALE = 0.18215


@dataclass(slots=True)
class StableDiffusionOutput(ModelOutput):
    """What :class:`StableDiffusionModel` returns.

    Attributes
    ----------
    noise_pred : Tensor
        ``(B, latent_channels, h, w)`` — :math:`\\epsilon_\\theta`.
    latent : Tensor
        The noised latent the prediction was made from.
    loss : Tensor or None
        The denoising objective, present only when a target was given.
    """

    noise_pred: Tensor
    latent: Tensor
    loss: Tensor | None = None


class StableDiffusionModel(PretrainedModel):
    r"""Latent diffusion: an autoencoder, a conditional U-Net, a sampler.

    Parameters
    ----------
    config : StableDiffusionConfig
        The variant to build.

    Attributes
    ----------
    vae : AutoencoderKL
        First stage.
    unet : UNet2DConditionModel
        Second stage.
    scheduler : DDIMScheduler
        Noise schedule and sampler.

    Notes
    -----
    Reference: Rombach et al., *"High-Resolution Image Synthesis with
    Latent Diffusion Models"*, CVPR, 2022
    (`arXiv:2112.10752 <https://arxiv.org/abs/2112.10752>`_).

    The text encoder is **not** held here.  The paper defines
    :math:`\tau_\theta` as a domain-specific encoder and evaluates
    several, and the released models freeze a CLIP text tower rather
    than training one — so conditioning arrives as an already-encoded
    ``(B, L, cross_attention_dim)`` sequence and the caller chooses what
    produced it.  At the default width that is
    :func:`~lucid.models.clip_vit_large_14`'s text tower.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     StableDiffusionConfig, StableDiffusionModel)
    >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
    ...                                vae_block_out_channels=(32, 64, 64),
    ...                                unet_block_out_channels=(32, 64),
    ...                                attention_head_dim=32,
    ...                                cross_attention_dim=16, context_length=4)
    >>> model = StableDiffusionModel(config).eval()
    >>> out = model(lucid.randn((1, 3, 32, 32)), lucid.randn((1, 4, 16)))
    >>> out.noise_pred.shape
    (1, 4, 8, 8)
    """

    config_class: ClassVar[type[StableDiffusionConfig]] = StableDiffusionConfig
    base_model_prefix = "stable_diffusion"

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.config: StableDiffusionConfig = config
        self.vae = AutoencoderKL(config)
        self.unet = UNet2DConditionModel(config)
        self.scheduler = DDIMScheduler(config)

    def encode_image(self, images: Tensor, sample: bool = True) -> Tensor:
        """Image to scaled latent.

        Parameters
        ----------
        images : Tensor
            ``(B, in_channels, H, W)``.
        sample : bool, default=True
            Draw from the posterior, or take its mode.

        Returns
        -------
        Tensor
            ``(B, latent_channels, H/f, W/f)``, already scaled for the
            diffusion process.
        """
        posterior = self.vae.encode(images)
        latent = posterior.sample() if sample else posterior.mode()
        return latent * _LATENT_SCALE

    def decode_latent(self, latent: Tensor) -> Tensor:
        """Scaled latent back to an image.

        Parameters
        ----------
        latent : Tensor
            ``(B, latent_channels, h, w)``.

        Returns
        -------
        Tensor
            ``(B, out_channels, h*f, w*f)``.
        """
        return self.vae.decode(latent / _LATENT_SCALE)

    @override
    def forward(  # type: ignore[override]
        self,
        images: Tensor,
        context: Tensor,
        timestep: Tensor | None = None,
        return_loss: bool = False,
    ) -> StableDiffusionOutput:
        r"""One training step's worth of the denoising objective.

        Parameters
        ----------
        images : Tensor
            ``(B, in_channels, H, W)``.
        context : Tensor
            ``(B, L, cross_attention_dim)`` — :math:`\tau_\theta(y)`.
        timestep : Tensor or None, optional
            ``(B,)``.  Drawn uniformly when omitted, which is what
            training does.
        return_loss : bool, default=False
            Also compute :math:`\|\epsilon - \epsilon_\theta\|_2^2`.

        Returns
        -------
        StableDiffusionOutput
            Prediction, the noised latent, and optionally the loss.
        """
        latent = self.encode_image(images)
        shape = tuple(int(s) for s in latent.shape)
        noise = lucid.randn(shape, device=latent.device.type, dtype=latent.dtype)

        if timestep is None:
            drawn = lucid.rand((shape[0],), device=latent.device.type)
            timestep = (drawn * self.config.num_train_timesteps).to(lucid.int64)

        # ``add_noise`` reads one scalar alpha, so a per-sample timestep
        # is applied one row at a time rather than gathered — the batch
        # here is small and a gather would need a data-dependent index.
        noised = lucid.stack(
            [
                self.scheduler.add_noise(
                    latent[i : i + 1], noise[i : i + 1], int(timestep[i].item())
                )[0]
                for i in range(shape[0])
            ],
            dim=0,
        )
        prediction = cast(Tensor, self.unet(noised, timestep.to(latent.dtype), context))

        loss: Tensor | None = None
        if return_loss:
            loss = ((prediction - noise) ** 2).mean()
        return StableDiffusionOutput(noise_pred=prediction, latent=noised, loss=loss)


class StableDiffusionForImageGeneration(PretrainedModel):
    """Stable Diffusion posed as a sampler.

    Parameters
    ----------
    config : StableDiffusionConfig
        The variant to build.

    Notes
    -----
    Reference: Rombach et al., CVPR 2022 (arXiv:2112.10752), §4.3.

    :meth:`generate` runs the reverse process and decodes; nothing is
    trained. Conditioning arrives already encoded, and for
    classifier-free guidance the caller supplies the unconditional
    sequence too — usually the text encoder's output for the empty
    string, which is not the same thing as zeros.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     StableDiffusionConfig, StableDiffusionForImageGeneration)
    >>> config = StableDiffusionConfig(sample_size=32, downsample_factor=4,
    ...                                vae_block_out_channels=(32, 64, 64),
    ...                                unet_block_out_channels=(32, 64),
    ...                                attention_head_dim=32,
    ...                                cross_attention_dim=16, context_length=4)
    >>> model = StableDiffusionForImageGeneration(config).eval()
    >>> image = model.generate(lucid.randn((1, 4, 16)), num_inference_steps=2)
    >>> image.shape
    (1, 3, 32, 32)
    """

    config_class: ClassVar[type[StableDiffusionConfig]] = StableDiffusionConfig
    base_model_prefix = "stable_diffusion"

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.stable_diffusion = StableDiffusionModel(config)

    @lucid.no_grad()
    def generate(
        self,
        context: Tensor,
        uncond_context: Tensor | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        latent: Tensor | None = None,
    ) -> Tensor:
        r"""Sample an image from the conditioning.

        Parameters
        ----------
        context : Tensor
            ``(B, L, cross_attention_dim)``.
        uncond_context : Tensor or None, optional
            The unconditional sequence. Guidance is skipped when absent,
            which is equivalent to ``guidance_scale = 1``.
        num_inference_steps : int, default=50
            Network evaluations along the reverse trajectory.
        guidance_scale : float, default=7.5
            :math:`s` in
            :math:`\epsilon_\varnothing + s(\epsilon_c - \epsilon_\varnothing)`.
        eta : float, default=0.0
            DDIM at 0, DDPM at 1.
        latent : Tensor or None, optional
            Starting noise. Drawn when absent.

        Returns
        -------
        Tensor
            ``(B, out_channels, sample_size, sample_size)``.

        Raises
        ------
        ValueError
            If ``guidance_scale`` is negative, or the two conditioning
            sequences disagree in shape.
        """
        if guidance_scale < 0.0:
            raise ValueError(
                f"guidance_scale must be non-negative, got {guidance_scale}"
            )
        inner = self.stable_diffusion
        config = inner.config
        batch = int(context.shape[0])
        side = config.latent_size

        if uncond_context is not None and tuple(uncond_context.shape) != tuple(
            context.shape
        ):
            raise ValueError(
                f"uncond_context {tuple(uncond_context.shape)} must match "
                f"context {tuple(context.shape)}"
            )

        if latent is None:
            latent = lucid.randn(
                (batch, config.latent_channels, side, side),
                device=context.device.type,
            )

        steps = inner.scheduler.timesteps(num_inference_steps)
        for index, step in enumerate(steps):
            previous = steps[index + 1] if index + 1 < len(steps) else -1
            timestep = lucid.full(
                (batch,), float(step), device=latent.device.type, dtype=latent.dtype
            )
            prediction = cast(Tensor, inner.unet(latent, timestep, context))
            if uncond_context is not None and guidance_scale != 1.0:
                unconditional = cast(
                    Tensor, inner.unet(latent, timestep, uncond_context)
                )
                prediction = unconditional + guidance_scale * (
                    prediction - unconditional
                )
            latent = inner.scheduler.step(prediction, step, previous, latent, eta)

        return inner.decode_latent(latent)

    @override
    def forward(  # type: ignore[override]
        self, context: Tensor, num_inference_steps: int = 50
    ) -> Tensor:
        """Alias for :meth:`generate` with default guidance disabled.

        Parameters
        ----------
        context : Tensor
            ``(B, L, cross_attention_dim)``.
        num_inference_steps : int, default=50
            Network evaluations.

        Returns
        -------
        Tensor
            ``(B, out_channels, sample_size, sample_size)``.
        """
        return cast(
            Tensor,
            self.generate(context, num_inference_steps=num_inference_steps),
        )
