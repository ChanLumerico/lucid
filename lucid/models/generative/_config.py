"""Shared base configs for ``lucid.models.generative``.

Mirrors the role of :class:`LanguageModelConfig` for text models: the common
fields every generative family needs are captured here once, so family
configs only add their unique knobs.

Two tiers:

    * :class:`GenerativeModelConfig` — every generative family (VAE / DDPM /
      NCSN / future flow models).  Holds the image shape and the trunk
      activation.
    * :class:`DiffusionModelConfig` — adds noise-schedule knobs used by
      every diffusion family.  VAE skips this tier.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models._base import ModelConfig

# Activation alias accepted by every generative family.  Diffusion U-Nets
# typically use SiLU (Swish); VAEs use either SiLU or ReLU.
GenerativeActivation = Literal["silu", "swish", "relu", "gelu"]

# Noise schedule shape — see ``make_beta_schedule`` in ``_utils/_generative``.
BetaSchedule = Literal["linear", "cosine"]


@dataclass(frozen=True)
class GenerativeModelConfig(ModelConfig):
    """Shared base for image-generative families.

    Args:
        sample_size: Output spatial resolution in pixels.  Square images use
            an int (``32`` → ``32 × 32``); rectangular targets pass a tuple.
        in_channels: Input image channels (3 for RGB, 1 for greyscale, 4
            for VAE latent samples, …).
        out_channels: Output channels — usually equal to ``in_channels``;
            diffusion models that predict variance use ``2 * in_channels``.
        act_fn: Activation used inside the trunk.  Most modern image
            generators default to ``"silu"``.
    """

    model_type: ClassVar[str] = "generative"

    sample_size: int | tuple[int, int] = 32
    in_channels: int = 3
    out_channels: int = 3
    act_fn: GenerativeActivation = "silu"

    def __post_init__(self) -> None:
        if isinstance(self.sample_size, tuple):
            if len(self.sample_size) != 2 or any(s <= 0 for s in self.sample_size):
                raise ValueError(
                    f"sample_size tuple must be (H, W) with both positive, got {self.sample_size}"
                )
        elif self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}")
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")
        if self.out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {self.out_channels}")


@dataclass(frozen=True)
class DiffusionModelConfig(GenerativeModelConfig):
    """Shared base for *diffusion*-family generative models.

    Args:
        num_train_timesteps: Number of forward-process steps ``T``.  DDPM
            uses 1000, NCSN traditionally a smaller grid.
        beta_start / beta_end: Endpoints of the linear noise schedule
            ``β_1 … β_T``.  Ignored when ``beta_schedule == "cosine"``.
        beta_schedule: ``"linear"`` (Ho et al., 2020) or ``"cosine"`` (Nichol
            & Dhariwal, 2021 — improves low-resolution sample quality).
        prediction_type: What the network predicts at each step.  ``"epsilon"``
            (the noise, default) is the canonical Ho parameterisation;
            ``"sample"`` predicts ``x_0`` directly; ``"v_prediction"`` is the
            Imagen / Progressive Distillation reparameterisation.
    """

    model_type: ClassVar[str] = "diffusion"

    num_train_timesteps: int = 1_000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: BetaSchedule = "linear"
    prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_train_timesteps <= 0:
            raise ValueError(
                f"num_train_timesteps must be positive, got {self.num_train_timesteps}"
            )
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError(
                f"Require 0 < beta_start < beta_end < 1, got "
                f"start={self.beta_start}, end={self.beta_end}"
            )


__all__ = [
    "GenerativeModelConfig",
    "DiffusionModelConfig",
    "GenerativeActivation",
    "BetaSchedule",
]
