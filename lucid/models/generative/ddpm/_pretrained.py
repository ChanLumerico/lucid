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

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.generative.ddpm._config import DDPMConfig
from lucid.models.generative.ddpm._model import DDPMForImageGeneration, DDPMModel
from lucid.models.generative.ddpm._weights import (
    DDPMChurchWeights,
    DDPMCifarWeights,
)

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
    return replace(cfg, **cast(dict[str, Any], overrides)) if overrides else cfg


# ── Bare U-Net trunks ─────────────────────────────────────────────────────────


@register_model(  # type: ignore[arg-type]  # reason: ddpm_cifar adds a typed weights= kwarg (DDPMCifarWeights); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMModel,
    default_config=_CFG_CIFAR,
)
def ddpm_cifar(
    pretrained: bool | str = False,
    *,
    weights: DDPMCifarWeights | None = None,
    **overrides: object,
) -> DDPMModel:
    r"""Construct a DDPM U-Net trunk for the CIFAR-10 setup.

    Paper-faithful CIFAR-10 configuration from Ho, Jain, and Abbeel, 2020
    Appendix B.1 / Table 9: sample size 32x32, ``base_channels=128``,
    ``channel_mult=(1, 2, 2, 2)`` (four spatial stages — 32 -> 16 -> 8 ->
    4), 2 ResBlocks per stage, self-attention at the 16x16 feature map,
    1 attention head, dropout 0.1, and a 1000-step linear ``beta``
    schedule.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Weight selector.  ``False`` → random init; ``True`` → the official
        ``google/ddpm-cifar10-32`` checkpoint
        (:attr:`DDPMCifarWeights.CIFAR10`) — the trained noise predictor
        :math:`\epsilon_\theta(x_t, t)`.
    weights : DDPMCifarWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DDPMConfig` field overrides (e.g. ``dropout=...``,
        ``num_train_timesteps=...``) forwarded into the underlying config.
        Overrides that change the U-Net shape are incompatible with the
        pretrained checkpoint.

    Returns
    -------
    DDPMModel
        Bare U-Net trunk configured with the CIFAR-10 setup (pretrained
        when requested).

    Notes
    -----
    Reference: Ho, Jain, and Abbeel, *"Denoising Diffusion Probabilistic
    Models"*, NeurIPS, 2020 (arXiv:2006.11239), Appendix B.1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import ddpm_cifar
    >>> model = ddpm_cifar().eval()
    >>> x_t = lucid.randn((1, 3, 32, 32))
    >>> t = lucid.tensor([42]).long()
    >>> out = model(x_t, t)
    >>> out.sample.shape   # (1, 3, 32, 32) — predicted noise
    (1, 3, 32, 32)
    """
    entry = weights_mod.resolve_weights(DDPMCifarWeights, pretrained, weights)
    model = DDPMModel(_apply(_CFG_CIFAR, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="ddpm_cifar")
    return model


@register_model(  # type: ignore[arg-type]  # reason: ddpm_lsun adds a typed weights= kwarg (DDPMChurchWeights); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="base",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMModel,
    default_config=_CFG_LSUN,
)
def ddpm_lsun(
    pretrained: bool | str = False,
    *,
    weights: DDPMChurchWeights | None = None,
    **overrides: object,
) -> DDPMModel:
    r"""Construct a DDPM U-Net trunk for the LSUN / CelebA-HQ 256x256 setup.

    Paper-faithful 256x256 configuration from Ho, Jain, and Abbeel, 2020
    Appendix B.2 (LSUN church / bedroom and CelebA-HQ): sample size
    256x256, ``base_channels=128``, ``channel_mult=(1, 1, 2, 2, 4, 4)``
    (six spatial stages — 256 -> 128 -> ... -> 8), 2 ResBlocks per stage,
    self-attention at the 16x16 feature map, 1 attention head, no
    dropout, and a 1000-step linear ``beta`` schedule.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Weight selector.  ``False`` → random init; ``True`` → the official
        ``google/ddpm-church-256`` LSUN-Church checkpoint
        (:attr:`DDPMChurchWeights.LSUN_CHURCH`).
    weights : DDPMChurchWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DDPMConfig` field overrides forwarded into the
        underlying config.  Overrides that change the U-Net shape are
        incompatible with the pretrained checkpoint.

    Returns
    -------
    DDPMModel
        Bare U-Net trunk configured with the LSUN / CelebA-HQ setup
        (pretrained when requested).

    Notes
    -----
    Reference: Ho, Jain, and Abbeel, *"Denoising Diffusion Probabilistic
    Models"*, NeurIPS, 2020 (arXiv:2006.11239), Appendix B.2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import ddpm_lsun
    >>> model = ddpm_lsun().eval()
    >>> x_t = lucid.randn((1, 3, 256, 256))
    >>> t = lucid.tensor([500]).long()
    >>> out = model(x_t, t)
    >>> out.sample.shape   # (1, 3, 256, 256)
    (1, 3, 256, 256)
    """
    entry = weights_mod.resolve_weights(DDPMChurchWeights, pretrained, weights)
    model = DDPMModel(_apply(_CFG_LSUN, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="ddpm_lsun")
    return model


@register_model(
    task="base",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMModel,
    default_config=_CFG_IMAGENET64,
)
def ddpm_imagenet64(pretrained: bool = False, **overrides: object) -> DDPMModel:
    r"""Construct an Improved-DDPM U-Net trunk for ImageNet 64x64.

    Paper-faithful ImageNet 64x64 configuration from Nichol and Dhariwal,
    2021 Table 1: sample size 64x64, ``base_channels=128``,
    ``channel_mult=(1, 2, 3, 4)`` (four spatial stages — 64 -> 32 -> 16
    -> 8), 3 ResBlocks per stage, multi-scale self-attention at the 8 /
    16 / 32 feature maps, 4 attention heads, learned variance head
    enabled (``learn_sigma=True``), and a 4000-step cosine ``beta``
    schedule.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`DDPMConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    DDPMModel
        Bare U-Net trunk configured with the Improved-DDPM ImageNet 64x64
        setup and any overrides.

    Notes
    -----
    Reference: Nichol and Dhariwal, *"Improved Denoising Diffusion
    Probabilistic Models"*, ICML, 2021 (arXiv:2102.09672), Table 1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import ddpm_imagenet64
    >>> model = ddpm_imagenet64().eval()
    >>> x_t = lucid.randn((1, 3, 64, 64))
    >>> t = lucid.tensor([1234]).long()
    >>> out = model(x_t, t)
    >>> out.sample.shape   # learn_sigma=True -> 2 * 3 = 6 channels
    (1, 6, 64, 64)
    """
    return DDPMModel(_apply(_CFG_IMAGENET64, overrides))


# ── Image-generation heads ───────────────────────────────────────────────────


@register_model(  # type: ignore[arg-type]  # reason: ddpm_cifar_gen adds a typed weights= kwarg (DDPMCifarWeights); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="image-generation",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMForImageGeneration,
    default_config=_CFG_CIFAR,
)
def ddpm_cifar_gen(
    pretrained: bool | str = False,
    *,
    weights: DDPMCifarWeights | None = None,
    **overrides: object,
) -> DDPMForImageGeneration:
    r"""Construct a DDPM CIFAR-10 model with training loss and ``.generate()``.

    Same trunk as :func:`ddpm_cifar` (sample size 32x32, 4-stage U-Net),
    wrapped with the Ho 2020 simplified training objective and
    :meth:`DiffusionMixin.generate` for ancestral sampling.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Weight selector.  ``False`` → random init; ``True`` → the official
        ``google/ddpm-cifar10-32`` checkpoint
        (:attr:`DDPMCifarWeights.CIFAR10`), giving an inference-ready
        sampler — ``model.generate(...)`` draws CIFAR-10-like images.
    weights : DDPMCifarWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DDPMConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    DDPMForImageGeneration
        CIFAR-10 DDPM wrapped with the noise-prediction loss head
        (pretrained when requested).

    Notes
    -----
    Reference: Ho, Jain, and Abbeel, *"Denoising Diffusion Probabilistic
    Models"*, NeurIPS, 2020 (arXiv:2006.11239), Appendix B.1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import ddpm_cifar_gen
    >>> model = ddpm_cifar_gen().eval()
    >>> x_t = lucid.randn((1, 3, 32, 32))
    >>> t = lucid.tensor([42]).long()
    >>> out = model(x_t, t)
    >>> out.sample.shape   # (1, 3, 32, 32)
    (1, 3, 32, 32)
    """
    entry = weights_mod.resolve_weights(DDPMCifarWeights, pretrained, weights)
    model = DDPMForImageGeneration(_apply(_CFG_CIFAR, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="ddpm_cifar")
    return model


@register_model(  # type: ignore[arg-type]  # reason: ddpm_lsun_gen adds a typed weights= kwarg (DDPMChurchWeights); the ModelFactory protocol predates the weights system and names only pretrained + **overrides.
    task="image-generation",
    family="ddpm",
    model_type="ddpm",
    model_class=DDPMForImageGeneration,
    default_config=_CFG_LSUN,
)
def ddpm_lsun_gen(
    pretrained: bool | str = False,
    *,
    weights: DDPMChurchWeights | None = None,
    **overrides: object,
) -> DDPMForImageGeneration:
    r"""Construct a DDPM LSUN / CelebA-HQ model with training loss and ``.generate()``.

    Same trunk as :func:`ddpm_lsun` (sample size 256x256, 6-stage U-Net),
    wrapped with the Ho 2020 simplified training objective and
    :meth:`DiffusionMixin.generate` for ancestral sampling.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Weight selector.  ``False`` → random init; ``True`` → the official
        ``google/ddpm-church-256`` checkpoint
        (:attr:`DDPMChurchWeights.LSUN_CHURCH`), giving an inference-ready
        sampler — ``model.generate(...)`` draws LSUN-Church-like images.
    weights : DDPMChurchWeights, optional, keyword-only
        Explicit weights enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DDPMConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    DDPMForImageGeneration
        LSUN / CelebA-HQ DDPM wrapped with the noise-prediction loss head
        (pretrained when requested).

    Notes
    -----
    Reference: Ho, Jain, and Abbeel, *"Denoising Diffusion Probabilistic
    Models"*, NeurIPS, 2020 (arXiv:2006.11239), Appendix B.2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import ddpm_lsun_gen
    >>> model = ddpm_lsun_gen().eval()
    >>> x_t = lucid.randn((1, 3, 256, 256))
    >>> t = lucid.tensor([500]).long()
    >>> out = model(x_t, t)
    >>> out.sample.shape   # (1, 3, 256, 256)
    (1, 3, 256, 256)
    """
    entry = weights_mod.resolve_weights(DDPMChurchWeights, pretrained, weights)
    model = DDPMForImageGeneration(_apply(_CFG_LSUN, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="ddpm_lsun")
    return model


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
    r"""Construct an Improved-DDPM ImageNet 64x64 model with training loss and ``.generate()``.

    Same trunk as :func:`ddpm_imagenet64` (sample size 64x64, 4-stage U-Net
    with ``learn_sigma=True``).

    .. warning::
        This factory currently raises :class:`NotImplementedError` at
        construction time — the default config has ``learn_sigma=True``,
        which requires the Improved-DDPM hybrid
        :math:`L_{\text{simple}} + \lambda L_{\text{vlb}}` loss not yet
        implemented in Lucid.  Pass ``learn_sigma=False`` to override.

    Parameters
    ----------
    pretrained : bool, default=False
        Reserved for future weight registration; currently a no-op.
    **overrides : object
        Optional :class:`DDPMConfig` field overrides forwarded into the
        underlying config.

    Returns
    -------
    DDPMForImageGeneration
        Improved-DDPM ImageNet 64x64 wrapper.

    Notes
    -----
    Reference: Nichol and Dhariwal, *"Improved Denoising Diffusion
    Probabilistic Models"*, ICML, 2021 (arXiv:2102.09672), Table 1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ddpm import ddpm_imagenet64_gen
    >>> model = ddpm_imagenet64_gen(learn_sigma=False).eval()
    >>> x_t = lucid.randn((1, 3, 64, 64))
    >>> t = lucid.tensor([1234]).long()
    >>> out = model(x_t, t)
    >>> out.sample.shape   # (1, 3, 64, 64)
    (1, 3, 64, 64)
    """
    return DDPMForImageGeneration(_apply(_CFG_IMAGENET64, overrides))
