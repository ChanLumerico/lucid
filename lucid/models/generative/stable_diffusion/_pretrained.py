"""Registry factories for Stable Diffusion.

Two architectures, not five.  The released v1 line — 1.1 through 1.5 —
is one network trained for different numbers of steps on different
subsets; the configs are identical, so the versions are *weight tags*
rather than variants and registering five factories would invent
architecture where there is none.  v2 genuinely differs: OpenCLIP
ViT-H conditions it, so the cross-attention width is 1024 rather than
768, and it is native at 768 pixels.

No parameter counts are registered.  The paper quotes 1.45B for its own
text-to-image LDM, which is a different model from the released ones —
quoting it here would attribute the paper's figure to Stability's
network.  ``summary="auto"`` reports what is actually built.

v1 ships pretrained weights, converted by
:mod:`tools.convert_weights.stable_diffusion` and verified against the
reference to float32 round-off before publication.  v2 does not yet —
its checkpoint is a separate archive with a wider conditioning tower,
and an unconverted `pretrained=True` says no rather than failing
quietly.
"""

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models._utils._common import reject_unavailable_pretrained
from lucid.models.generative.stable_diffusion._config import StableDiffusionConfig
from lucid.models.generative.stable_diffusion._weights import (
    StableDiffusionV1Weights,
)
from lucid.models.generative.stable_diffusion._model import (
    StableDiffusionForImageGeneration,
    StableDiffusionModel,
)

__all__ = [
    "stable_diffusion_v1",
    "stable_diffusion_v2",
    "stable_diffusion_v1_gen",
    "stable_diffusion_v2_gen",
]

_CFG_V1 = StableDiffusionConfig()
_CFG_V2 = StableDiffusionConfig(
    sample_size=768, cross_attention_dim=1024, attention_head_dim=64
)


def _apply(
    cfg: StableDiffusionConfig, overrides: dict[str, object]
) -> StableDiffusionConfig:
    return replace(cfg, **cast(dict[str, Any], overrides)) if overrides else cfg


@register_model(
    task="base",
    family="stable_diffusion",
    model_type="stable_diffusion",
    model_class=StableDiffusionModel,
    default_config=_CFG_V1,
    summary="auto",
)
def stable_diffusion_v1(
    pretrained: bool = False, **overrides: object
) -> StableDiffusionModel:
    """Construct the v1 architecture — 512 pixels, CLIP ViT-L/14 width.

    Parameters
    ----------
    pretrained : bool, default=False
        Load the released v1 checkpoint from the Lucid hub.
    **overrides : object
        Optional :class:`StableDiffusionConfig` field overrides.

    Returns
    -------
    StableDiffusionModel
        Autoencoder, conditional U-Net and sampler.

    Notes
    -----
    Reference: Rombach et al., CVPR 2022 (arXiv:2112.10752).

    ``cross_attention_dim`` is 768, which is exactly
    :func:`~lucid.models.clip_vit_large_14`'s text width — that tower is
    what produced the conditioning these settings were trained against.

    Examples
    --------
    >>> from lucid.models import stable_diffusion_v1
    >>> model = stable_diffusion_v1()
    >>> model.config.latent_size, model.config.cross_attention_dim
    (64, 768)
    """
    model = StableDiffusionModel(_apply(_CFG_V1, overrides))
    entry = weights_mod.resolve_weights(StableDiffusionV1Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="stable_diffusion_v1")
    return model


@register_model(
    task="base",
    family="stable_diffusion",
    model_type="stable_diffusion",
    model_class=StableDiffusionModel,
    default_config=_CFG_V2,
    summary="auto",
)
def stable_diffusion_v2(
    pretrained: bool = False, **overrides: object
) -> StableDiffusionModel:
    """Construct the v2 architecture — 768 pixels, OpenCLIP ViT-H width.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`StableDiffusionConfig` field overrides.

    Returns
    -------
    StableDiffusionModel
        Autoencoder, conditional U-Net and sampler.

    Notes
    -----
    Reference: Rombach et al., CVPR 2022 (arXiv:2112.10752), with the
    v2 release's conditioning width.

    The only architectural differences from v1 are the conditioning
    width and the native resolution; the autoencoder is unchanged, which
    is why a v1 latent and a v2 latent are the same kind of object.

    Examples
    --------
    >>> from lucid.models import stable_diffusion_v2
    >>> model = stable_diffusion_v2()
    >>> model.config.cross_attention_dim, model.config.latent_size
    (1024, 96)
    """
    if pretrained:
        reject_unavailable_pretrained("stable_diffusion_v2")
    return StableDiffusionModel(_apply(_CFG_V2, overrides))


@register_model(
    task="image-generation",
    family="stable_diffusion",
    model_type="stable_diffusion",
    model_class=StableDiffusionForImageGeneration,
    default_config=_CFG_V1,
    summary="auto",
)
def stable_diffusion_v1_gen(
    pretrained: bool = False, **overrides: object
) -> StableDiffusionForImageGeneration:
    """v1 posed as a sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`StableDiffusionConfig` field overrides.

    Returns
    -------
    StableDiffusionForImageGeneration
        Exposes :meth:`generate` with classifier-free guidance.

    Notes
    -----
    Reference: Rombach et al., CVPR 2022 (arXiv:2112.10752), §4.3.

    Examples
    --------
    >>> from lucid.models import stable_diffusion_v1_gen
    >>> model = stable_diffusion_v1_gen()
    >>> model.config.sample_size
    512
    """
    model = StableDiffusionForImageGeneration(_apply(_CFG_V1, overrides))
    entry = weights_mod.resolve_weights(StableDiffusionV1Weights, pretrained, None)
    if entry is not None:
        weights_mod.load_weight_entry(
            model.stable_diffusion, entry, name="stable_diffusion_v1_gen"
        )
    return model


@register_model(
    task="image-generation",
    family="stable_diffusion",
    model_type="stable_diffusion",
    model_class=StableDiffusionForImageGeneration,
    default_config=_CFG_V2,
    summary="auto",
)
def stable_diffusion_v2_gen(
    pretrained: bool = False, **overrides: object
) -> StableDiffusionForImageGeneration:
    """v2 posed as a sampler.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights are published for this family; passing ``True`` raises.
    **overrides : object
        Optional :class:`StableDiffusionConfig` field overrides.

    Returns
    -------
    StableDiffusionForImageGeneration
        Exposes :meth:`generate` with classifier-free guidance.

    Notes
    -----
    Reference: Rombach et al., CVPR 2022 (arXiv:2112.10752), §4.3.

    Examples
    --------
    >>> from lucid.models import stable_diffusion_v2_gen
    >>> model = stable_diffusion_v2_gen()
    >>> model.config.sample_size
    768
    """
    if pretrained:
        reject_unavailable_pretrained("stable_diffusion_v2_gen")
    return StableDiffusionForImageGeneration(_apply(_CFG_V2, overrides))
