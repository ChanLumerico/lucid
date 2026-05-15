"""NCSN configuration — Song & Ermon, 2019 ("Generative Modeling by
Estimating Gradients of the Data Distribution") + NCSNv2 (Song 2020).

Score-based generative model: learns ``s_θ(x̃, σ) ≈ ∇_x̃ log p_σ(x̃)`` for a
schedule of noise levels ``σ_1 > σ_2 > … > σ_L``, then samples via annealed
Langevin dynamics across that schedule.

Architecture-wise we reuse :class:`DDPMUNet` (NCSNv2 / NCSN++ converged on
the same modern U-Net as diffusion).  The differences live in:

    * Noise schedule: geometric ``σ`` levels instead of diffusion ``β``.
    * Loss:           denoising score matching (DSM) — see
                      :meth:`NCSNForImageGeneration.forward`.
    * Sampling:       annealed Langevin dynamics — see
                      :meth:`NCSNForImageGeneration.generate`.
"""

from dataclasses import dataclass, field
from typing import ClassVar

from lucid.models.generative._config import GenerativeModelConfig


@dataclass(frozen=True)
class NCSNConfig(GenerativeModelConfig):
    """Configuration for every NCSN variant.

    Args:
        base_channels: First U-Net stage width (paper: 128).
        channel_mult: Per-stage channel multiplier.  Same semantics as
            :class:`DDPMConfig`.
        num_res_blocks: ResBlocks per stage.
        attention_resolutions: Spatial sizes at which to insert
            self-attention.
        num_heads: Attention head count.
        dropout: Dropout inside ResBlocks.
        resnet_groups: GroupNorm group count.
        num_noise_levels: Number of σ levels ``L`` (paper L=10, NCSNv2 large
            L ≈ 200–500).
        sigma_max: Largest noise σ_1 (paper CIFAR-10: 50.0).
        sigma_min: Smallest noise σ_L (paper CIFAR-10: 0.01).
        langevin_steps: Sampler steps per σ level (paper T=100).
        langevin_eps: Sampler step-size base (paper ε=2e-5).  Per-σ step
            size is ``ε · σ_i² / σ_L²`` (Song 2019 §4.3).
    """

    model_type: ClassVar[str] = "ncsn"

    # U-Net architectural knobs (mirrored from DDPMConfig — same arch).
    base_channels: int = 128
    channel_mult: tuple[int, ...] = field(default_factory=lambda: (1, 2, 2, 2))
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = field(default_factory=lambda: (16,))
    num_heads: int = 1
    dropout: float = 0.0
    resnet_groups: int = 32

    # Score-based: sigma schedule + Langevin sampler.
    num_noise_levels: int = 10
    sigma_max: float = 50.0
    sigma_min: float = 0.01
    langevin_steps: int = 100
    langevin_eps: float = 2e-5

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
        if self.num_noise_levels <= 0:
            raise ValueError(
                f"num_noise_levels must be positive, got {self.num_noise_levels}"
            )
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise ValueError(
                f"require 0 < sigma_min < sigma_max, got "
                f"min={self.sigma_min}, max={self.sigma_max}"
            )
        if self.langevin_steps <= 0:
            raise ValueError(
                f"langevin_steps must be positive, got {self.langevin_steps}"
            )
        if self.langevin_eps <= 0:
            raise ValueError(f"langevin_eps must be positive, got {self.langevin_eps}")

        # Spatial divisibility — same rule as DDPM (no downsample after last stage).
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

        for mult in self.channel_mult:
            stage_ch = self.base_channels * mult
            if stage_ch % self.resnet_groups != 0:
                raise ValueError(
                    f"Stage channels {stage_ch} (base_channels {self.base_channels}"
                    f" × mult {mult}) must be divisible by resnet_groups "
                    f"{self.resnet_groups}"
                )
