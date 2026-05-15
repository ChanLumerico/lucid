"""Registry factories for DDPM (Ho et al., 2020).

The paper specifies one architecture template parametrised by dataset.  We
expose one factory per dataset config from Appendix B / Table 9:

    * ``ddpm_cifar``      — CIFAR-10 (32 × 32)
    * ``ddpm_lsun``       — LSUN / CelebA-HQ class (256 × 256)
    * ``ddpm_imagenet64`` — ImageNet 64 × 64

All three share the same U-Net code; only channel widths / multipliers /
dropout differ.  Smaller test configs go through
``create_model("ddpm_cifar", base_channels=..., channel_mult=...)``.
"""

from lucid.models._registry import register_model
from lucid.models.generative.ddpm._config import DDPMConfig
from lucid.models.generative.ddpm._model import DDPMForImageGeneration, DDPMModel

# Ho 2020 Table 9 — CIFAR-10.
_CFG_CIFAR = DDPMConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(16,),
    num_heads=1,
    dropout=0.1,
    num_train_timesteps=1_000,
    beta_schedule="linear",
)

# Ho 2020 Table 9 — LSUN church / bedroom / CelebA-HQ (all 256×256).
_CFG_LSUN = DDPMConfig(
    sample_size=256,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 1, 2, 2, 4, 4),
    num_res_blocks=2,
    attention_resolutions=(16,),
    num_heads=1,
    dropout=0.0,
    num_train_timesteps=1_000,
    beta_schedule="linear",
)

# Improved DDPM (Nichol 2021) Table 1 — ImageNet 64×64 with cosine schedule
# and learned variance.
_CFG_IMAGENET64 = DDPMConfig(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 2, 3, 4),
    num_res_blocks=3,
    attention_resolutions=(8, 16, 32),
    num_heads=4,
    dropout=0.0,
    num_train_timesteps=4_000,
    beta_schedule="cosine",
    learn_sigma=True,
)


def _apply(cfg: DDPMConfig, overrides: dict[str, object]) -> DDPMConfig:
    return DDPMConfig(**{**cfg.__dict__, **overrides}) if overrides else cfg


# ── Bare U-Net trunks ─────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMModel,
    default_config=_CFG_CIFAR,
)
def ddpm_cifar(pretrained: bool = False, **overrides: object) -> DDPMModel:
    """DDPM on CIFAR-10 (Ho et al., 2020 Appendix B.1)."""
    return DDPMModel(_apply(_CFG_CIFAR, overrides))


@register_model(
    task="base",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMModel,
    default_config=_CFG_LSUN,
)
def ddpm_lsun(pretrained: bool = False, **overrides: object) -> DDPMModel:
    """DDPM on LSUN / CelebA-HQ 256×256 (Ho et al., 2020 Appendix B.2)."""
    return DDPMModel(_apply(_CFG_LSUN, overrides))


@register_model(
    task="base",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMModel,
    default_config=_CFG_IMAGENET64,
)
def ddpm_imagenet64(pretrained: bool = False, **overrides: object) -> DDPMModel:
    """Improved DDPM on ImageNet 64×64 (Nichol & Dhariwal, 2021 Table 1)."""
    return DDPMModel(_apply(_CFG_IMAGENET64, overrides))


# ── Image-generation heads ───────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMForImageGeneration,
    default_config=_CFG_CIFAR,
)
def ddpm_cifar_gen(
    pretrained: bool = False, **overrides: object
) -> DDPMForImageGeneration:
    return DDPMForImageGeneration(_apply(_CFG_CIFAR, overrides))


@register_model(
    task="image-generation",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMForImageGeneration,
    default_config=_CFG_LSUN,
)
def ddpm_lsun_gen(
    pretrained: bool = False, **overrides: object
) -> DDPMForImageGeneration:
    return DDPMForImageGeneration(_apply(_CFG_LSUN, overrides))


@register_model(
    task="image-generation",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMForImageGeneration,
    default_config=_CFG_IMAGENET64,
)
def ddpm_imagenet64_gen(
    pretrained: bool = False, **overrides: object
) -> DDPMForImageGeneration:
    return DDPMForImageGeneration(_apply(_CFG_IMAGENET64, overrides))
