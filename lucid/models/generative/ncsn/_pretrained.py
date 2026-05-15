"""Registry factories for NCSN (Song & Ermon, 2019).

Two paper-faithful variants:

    * ``ncsn_cifar``  — CIFAR-10 setup from Song 2019 §4.1 / NCSNv2 Table 1
    * ``ncsn_celeba`` — CelebA 64×64 setup from NCSNv2 §C

Both reuse the DDPM U-Net architecture (NCSNv2 / NCSN++ uses the same
modern U-Net as diffusion).  ``sigma_max`` is dataset-specific — set
following NCSNv2 §3.2 "Technique 1" (use ``σ_max`` ≈ max pairwise data
distance).
"""

from lucid.models._registry import register_model
from lucid.models.generative.ncsn._config import NCSNConfig
from lucid.models.generative.ncsn._model import NCSNForImageGeneration, NCSNModel

# Song & Ermon 2019 CIFAR-10 setup (§4.1) — L=10, σ_1=1, σ_10=0.01.  NCSNv2
# bumped σ_max up to 50 for CIFAR; we follow the v2 recommendation as the
# default since it produces sharper samples (paper Table 1).
_CFG_CIFAR = NCSNConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attention_resolutions=(16,),
    num_heads=1,
    dropout=0.0,
    num_noise_levels=232,
    sigma_max=50.0,
    sigma_min=0.01,
    langevin_steps=5,  # NCSNv2 §3.3 — T=5 with σ-tuned step size
    langevin_eps=2e-5,
)

# NCSNv2 §C — CelebA 64×64.
_CFG_CELEBA = NCSNConfig(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    base_channels=128,
    channel_mult=(1, 1, 2, 2, 4, 4),
    num_res_blocks=2,
    attention_resolutions=(16,),
    num_heads=1,
    dropout=0.0,
    num_noise_levels=500,
    sigma_max=90.0,
    sigma_min=0.01,
    langevin_steps=5,
    langevin_eps=2e-5,
)


def _apply(cfg: NCSNConfig, overrides: dict[str, object]) -> NCSNConfig:
    return NCSNConfig(**{**cfg.__dict__, **overrides}) if overrides else cfg


# ── Bare score networks ───────────────────────────────────────────────────────


@register_model(
    task="base",
    family="ncsn",
    model_type="ncsn",
    model_class=NCSNModel,
    default_config=_CFG_CIFAR,
)
def ncsn_cifar(pretrained: bool = False, **overrides: object) -> NCSNModel:
    """NCSN on CIFAR-10 (Song & Ermon 2019 / NCSNv2 Table 1)."""
    return NCSNModel(_apply(_CFG_CIFAR, overrides))


@register_model(
    task="base",
    family="ncsn",
    model_type="ncsn",
    model_class=NCSNModel,
    default_config=_CFG_CELEBA,
)
def ncsn_celeba(pretrained: bool = False, **overrides: object) -> NCSNModel:
    """NCSN on CelebA 64×64 (NCSNv2 Appendix C)."""
    return NCSNModel(_apply(_CFG_CELEBA, overrides))


# ── Image-generation heads ───────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="ncsn",
    model_type="ncsn",
    model_class=NCSNForImageGeneration,
    default_config=_CFG_CIFAR,
)
def ncsn_cifar_gen(
    pretrained: bool = False, **overrides: object
) -> NCSNForImageGeneration:
    return NCSNForImageGeneration(_apply(_CFG_CIFAR, overrides))


@register_model(
    task="image-generation",
    family="ncsn",
    model_type="ncsn",
    model_class=NCSNForImageGeneration,
    default_config=_CFG_CELEBA,
)
def ncsn_celeba_gen(
    pretrained: bool = False, **overrides: object
) -> NCSNForImageGeneration:
    return NCSNForImageGeneration(_apply(_CFG_CELEBA, overrides))
