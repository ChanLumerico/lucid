"""Registry factories for VAE.

Our convolutional VAE doesn't map to a single canonical paper variant table
(the way ResNet does for sizes 18 / 34 / 50 / …) — Kingma & Welling (2013)
specified an MLP encoder, and downstream variants (β-VAE, Ladder VAE, …)
each use bespoke per-dataset configs.  Following the project rule
("paper-named only, otherwise nominal"), we expose just two factories:

    * ``vae``  — vanilla / β-VAE topology (single bottleneck ``z``).
    * ``hvae`` — hierarchical / Ladder-VAE topology (one ``z_l`` per stage).

Switch β-VAE on by passing ``kl_weight=...`` at ``create_model`` time.
Specific dataset configs (DSprites, CelebA, MNIST 5-level, …) belong in
user code or in follow-up paper-faithful factories when we ship pretrained
weights.
"""

from lucid.models._registry import register_model
from lucid.models.generative.vae._config import VAEConfig
from lucid.models.generative.vae._model import VAEForImageGeneration, VAEModel

# Vanilla / β-VAE default — single bottleneck ``z``.
_CFG_VAE = VAEConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    latent_dim=128,
    down_block_channels=(64, 128, 256),
)

# Hierarchical (Sønderby et al., 2016 / Ladder VAE) default — one ``z_l``
# per encoder stage.
_CFG_HVAE = VAEConfig(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    latent_dim=(32, 64, 128),
    down_block_channels=(64, 128, 256),
)


def _apply(cfg: VAEConfig, overrides: dict[str, object]) -> VAEConfig:
    return VAEConfig(**{**cfg.__dict__, **overrides}) if overrides else cfg


# ── Bare encoder-decoder ──────────────────────────────────────────────────────


@register_model(
    task="base",
    family="vae",
    model_type="vae",
    model_class=VAEModel,
    default_config=_CFG_VAE,
)
def vae(pretrained: bool = False, **overrides: object) -> VAEModel:
    """Convolutional VAE (Kingma & Welling, 2013) — single bottleneck.

    Override ``latent_dim`` with a tuple to switch to the hierarchical
    topology (or use :func:`hvae` instead, which defaults to a 3-level
    Sønderby-style stack).
    """
    return VAEModel(_apply(_CFG_VAE, overrides))


@register_model(
    task="base",
    family="vae",
    model_type="vae",
    model_class=VAEModel,
    default_config=_CFG_HVAE,
)
def hvae(pretrained: bool = False, **overrides: object) -> VAEModel:
    """Hierarchical VAE (Sønderby et al., 2016) — 3-level latent stack."""
    return VAEModel(_apply(_CFG_HVAE, overrides))


# ── Image-generation heads ───────────────────────────────────────────────────


@register_model(
    task="image-generation",
    family="vae",
    model_type="vae",
    model_class=VAEForImageGeneration,
    default_config=_CFG_VAE,
)
def vae_gen(pretrained: bool = False, **overrides: object) -> VAEForImageGeneration:
    """Vanilla VAE + ELBO loss + prior-sample ``.generate()``."""
    return VAEForImageGeneration(_apply(_CFG_VAE, overrides))


@register_model(
    task="image-generation",
    family="vae",
    model_type="vae",
    model_class=VAEForImageGeneration,
    default_config=_CFG_HVAE,
)
def hvae_gen(pretrained: bool = False, **overrides: object) -> VAEForImageGeneration:
    """Hierarchical VAE + ELBO (sum of per-level KLs) + ``.generate()``."""
    return VAEForImageGeneration(_apply(_CFG_HVAE, overrides))
