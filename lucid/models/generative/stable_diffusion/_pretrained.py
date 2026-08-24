"""Registry factories for Stable Diffusion.

One architecture, not five.  The released v1 line — 1.1 through 1.5 —
is one network trained for different numbers of steps on different
subsets; the configs are identical, so the versions are *weight tags*
rather than variants and registering five factories would invent
architecture where there is none.

**v2 is deliberately absent.**  It does differ — OpenCLIP ViT-H
conditions it at 1024 rather than 768 — but its published configuration
could not be read from the primary source when this was written, and
the field that decides its attention shape is named for the opposite of
what it means (see ``attention_head_dim`` in the config).  A variant
whose numbers cannot be cited is exactly what H11 forbids, so it waits
for the config rather than shipping a guess.

No parameter counts are registered.  The paper quotes 1.45B for its own
text-to-image LDM, which is a different model from the released ones —
quoting it here would attribute the paper's figure to Stability's
network.  ``summary="auto"`` reports what is actually built.

The weights are converted by
:mod:`tools.convert_weights.stable_diffusion` and were verified against
the reference to float32 round-off before publication.
"""

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
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
    "stable_diffusion_v1_gen",
]

_CFG_V1 = StableDiffusionConfig()
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


