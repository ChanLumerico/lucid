"""VAE configuration — Kingma & Welling, 2013 + Sønderby et al., 2016.

Single config covers three flavours:

    * **Vanilla VAE** — pass ``latent_dim=int``.  One bottleneck ``z`` from
      the flattened final encoder stage.
    * **β-VAE**       — pass ``latent_dim=int`` + ``kl_weight≠1`` (Higgins
      et al., 2017).
    * **Hierarchical VAE (HVAE)** — pass ``latent_dim=tuple[int, ...]``.
      Each entry is one latent ``z_l`` extracted at the corresponding
      encoder stage; the decoder injects ``z_l`` back at the matching
      upsampling spatial scale.  KL is summed across levels (Sønderby
      Ladder VAE, 2016 — independent posteriors variant).

Architecture knobs (channel widths, sample size) are shared across all
three modes.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from lucid.models.generative._config import GenerativeModelConfig


@dataclass(frozen=True)
class VAEConfig(GenerativeModelConfig):
    """Configuration for every VAE / β-VAE / HVAE variant.

    Args:
        latent_dim: Bottleneck spec.

            * ``int`` — vanilla / β-VAE: a single ``z`` of this dimension is
              projected from the flattened final encoder stage.
            * ``tuple[int, ...]`` — hierarchical: one ``z_l`` per encoder
              stage.  Length must equal ``len(down_block_channels)``.

        down_block_channels: Channel width *after* each encoder downsampling
            block.  Length determines the number of stride-2 blocks; the
            final spatial resolution is ``sample_size / 2 ** len(...)``.
        kl_weight: Multiplicative β in the β-VAE loss.  ``1.0`` recovers the
            vanilla ELBO; ``< 1`` weakens the KL pull (better
            reconstructions); ``> 1`` enforces stronger disentanglement
            (Higgins et al., 2017 β-VAE).  In the hierarchical case, β
            scales every level's KL identically.
        recon_loss: Reconstruction term — ``"mse"`` (Gaussian likelihood,
            default) or ``"bce"`` (Bernoulli likelihood for [0, 1] data).
    """

    model_type: ClassVar[str] = "vae"

    # VAE-base for CIFAR-10 — 32 × 32 × 3 → 4 × 4 × 256 → 128-dim z
    latent_dim: int | tuple[int, ...] = 128
    down_block_channels: tuple[int, ...] = field(default_factory=lambda: (64, 128, 256))

    kl_weight: float = 1.0
    recon_loss: Literal["mse", "bce"] = "mse"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.down_block_channels:
            raise ValueError("down_block_channels must have at least one stage")
        if any(c <= 0 for c in self.down_block_channels):
            raise ValueError(
                f"down_block_channels entries must be positive, got {self.down_block_channels}"
            )

        # Validate latent_dim per mode.
        if isinstance(self.latent_dim, tuple):
            L = len(self.down_block_channels)
            if len(self.latent_dim) != L:
                raise ValueError(
                    f"Hierarchical VAE: latent_dim tuple length must equal "
                    f"len(down_block_channels) ({len(self.latent_dim)} vs {L}); "
                    f"HVAE assigns one z per encoder stage."
                )
            if any(d <= 0 for d in self.latent_dim):
                raise ValueError(
                    f"latent_dim entries must be positive, got {self.latent_dim}"
                )
        else:
            if self.latent_dim <= 0:
                raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")

        if self.kl_weight < 0.0:
            raise ValueError(f"kl_weight must be ≥ 0, got {self.kl_weight}")

        # Sample size must divide cleanly by the total downsampling factor.
        if isinstance(self.sample_size, tuple):
            h, w = self.sample_size
        else:
            h = w = self.sample_size
        factor = 2 ** len(self.down_block_channels)
        if h % factor != 0 or w % factor != 0:
            raise ValueError(
                f"sample_size {self.sample_size} must be divisible by 2 ** "
                f"len(down_block_channels) = {factor}"
            )

    @property
    def is_hierarchical(self) -> bool:
        """True when ``latent_dim`` is a tuple — selects the HVAE topology."""
        return isinstance(self.latent_dim, tuple)

    @property
    def latent_dims(self) -> tuple[int, ...]:
        """Always-tuple view of ``latent_dim`` (length 1 in vanilla mode)."""
        if isinstance(self.latent_dim, tuple):
            return self.latent_dim
        return (self.latent_dim,)

    @property
    def total_latent_dim(self) -> int:
        """Sum across levels — what the decoder consumes at its top input."""
        return sum(self.latent_dims)
