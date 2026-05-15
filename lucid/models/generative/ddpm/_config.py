"""DDPM configuration — Ho et al., 2020 ("Denoising Diffusion Probabilistic Models").

The paper introduced the now-canonical U-Net + sinusoidal-timestep architecture
for image diffusion.  We expose every hyperparameter from Appendix B (channel
multipliers, ResBlock count, attention resolutions, dropout) plus a couple of
Improved-DDPM (Nichol & Dhariwal 2021) extras (``learn_sigma``,
``v_prediction``) — superset of the legacy port, intentionally.

Inherits noise-schedule knobs from :class:`DiffusionModelConfig`
(``num_train_timesteps`` / ``beta_start`` / ``beta_end`` / ``beta_schedule`` /
``prediction_type``).
"""

from dataclasses import dataclass, field
from typing import ClassVar

from lucid.models.generative._config import DiffusionModelConfig


@dataclass(frozen=True)
class DDPMConfig(DiffusionModelConfig):
    """Configuration for every DDPM variant.

    Args:
        base_channels: Channel width of the first U-Net stage ``C_0``.  All
            other stages are integer multiples (see ``channel_mult``).
            Paper default: 128.
        channel_mult: Per-stage channel multiplier.  ``(1, 2, 2, 2)`` for
            CIFAR (Table 9), ``(1, 1, 2, 2, 4, 4)`` for LSUN/CelebA-HQ.
            Length = number of downsampling stages; spatial resolution
            divides by ``2 ** (L - 1)`` (no downsample after the final
            stage).
        num_res_blocks: ResBlocks per stage (paper: 2).
        attention_resolutions: Spatial sizes at which self-attention is
            applied.  Ho 2020 uses ``(16,)`` — attention only at the
            16×16 feature scale.
        num_heads: Attention head count (paper: 1 for CIFAR, can scale).
        dropout: Dropout inside ResBlocks.  Paper: 0.1 (CIFAR), 0.0
            (CelebA / LSUN).
        resnet_groups: ``GroupNorm`` group count inside ResBlocks (paper
            default: 32).  Must divide every per-stage channel count.
        learn_sigma: Improved-DDPM (Nichol 2021) — when ``True``, the
            network predicts the variance in addition to the mean.  Output
            channels become ``2 * in_channels``.
        clip_denoised: Clip the predicted ``x_0`` to ``[-1, 1]`` during
            sampling (default True).
    """

    model_type: ClassVar[str] = "ddpm"

    # U-Net architectural knobs.
    base_channels: int = 128
    channel_mult: tuple[int, ...] = field(default_factory=lambda: (1, 2, 2, 2))
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = field(default_factory=lambda: (16,))
    num_heads: int = 1
    dropout: float = 0.1
    resnet_groups: int = 32

    # Improved DDPM / Imagen-style extras.
    learn_sigma: bool = False

    # Sampling.
    clip_denoised: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.base_channels <= 0:
            raise ValueError(
                f"base_channels must be positive, got {self.base_channels}"
            )
        if not self.channel_mult:
            raise ValueError("channel_mult must have at least one stage")
        if any(m <= 0 for m in self.channel_mult):
            raise ValueError(
                f"channel_mult entries must be positive, got {self.channel_mult}"
            )
        if self.num_res_blocks <= 0:
            raise ValueError(
                f"num_res_blocks must be positive, got {self.num_res_blocks}"
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.resnet_groups <= 0:
            raise ValueError(
                f"resnet_groups must be positive, got {self.resnet_groups}"
            )

        # Spatial divisibility: ``2 ** (L - 1)`` because the last stage has
        # no downsample.
        if isinstance(self.sample_size, tuple):
            h, w = self.sample_size
        else:
            h = w = self.sample_size
        factor = 2 ** (len(self.channel_mult) - 1)
        if h % factor != 0 or w % factor != 0:
            raise ValueError(
                f"sample_size {self.sample_size} must be divisible by "
                f"2 ** (len(channel_mult) - 1) = {factor}"
            )

        # Every per-stage channel count must be divisible by ``resnet_groups``
        # so GroupNorm doesn't error.
        for mult in self.channel_mult:
            stage_ch = self.base_channels * mult
            if stage_ch % self.resnet_groups != 0:
                raise ValueError(
                    f"Stage channels {stage_ch} (base_channels {self.base_channels}"
                    f" × mult {mult}) must be divisible by resnet_groups "
                    f"{self.resnet_groups}"
                )

    @property
    def out_channels_effective(self) -> int:
        """How many channels the U-Net actually emits (doubled when
        ``learn_sigma=True``)."""
        return 2 * self.in_channels if self.learn_sigma else self.in_channels
