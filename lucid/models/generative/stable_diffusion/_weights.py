"""Pretrained-weight declarations for Stable Diffusion.

One entry, holding both sub-models.  The released archive ships the
autoencoder and the U-Net separately and they are useless apart — a
latent from one decoder is not a latent for another — so they are
converted and published together under a single tag.

Checked numerically against the reference before publication rather
than merely loaded.  With these weights in place all three paths agree
to float32 round-off:

===================  ==========
path                 rel. error
===================  ==========
U-Net epsilon        9.2e-05
autoencoder encoder  2.7e-04
autoencoder decoder  1.4e-04
===================  ==========

That check is what this family needed most, because the conversion had
already passed a stricter-looking one.  The parameter counts matched the
release *tensor for tensor* while two defects remained: the attention
was built with forty heads of eight channels where the reference uses
eight of forty, and the autoencoder's downsampler padded symmetrically
where the reference pads bottom-right only.  Neither changes a shape,
a count, or anything a forward pass reports.

The text encoder is not here.  This family takes conditioning as an
already-encoded sequence, and the released models condition on a frozen
CLIP ViT-L/14 text tower — :func:`~lucid.models.clip_vit_large_14`,
whose own weights are published separately.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

__all__ = ["StableDiffusionV1Weights"]

# The first stage works in [-1, 1] rather than on dataset statistics —
# it is an autoencoder, not a classifier, and has no notion of a mean
# image to subtract.
_PRESET_512 = ImageClassification(
    crop_size=512,
    resize_size=512,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)


@register_weights("stable_diffusion_v1")
class StableDiffusionV1Weights(WeightsEnum):
    """Pretrained weights for :func:`lucid.models.stable_diffusion_v1`.

    The v1 line's released checkpoint, trained on LAION subsets.
    ``num_classes`` is 0: a diffusion model has no label set, and the
    conditioning it does take is supplied per call rather than baked in.

    Notes
    -----
    Loading these gives a model that denoises but cannot yet be
    prompted — the conditioning sequence is the caller's to produce, and
    at this configuration's width it comes from CLIP ViT-L/14.
    """

    COMPVIS_LAION = WeightEntry(
        url=(
            f"{HUB_BASE}/stable-diffusion-v1/resolve/main/"
            "CompVis_LAION/model.safetensors"
        ),
        sha256="2fa36eeea24af004402d07d8c1ff6731f81119d69bdf952e5d0da5ad42f2860e",
        num_classes=0,
        transforms=_PRESET_512,
        meta={
            "tag": "CompVis_LAION",
            "source": "stable-diffusion-v1-5 (diffusers layout)",
            "license": "creativeml-openrail-m",
            "num_params": 943_174_827,
            "file_size_mb": 3598.0,
            "conditioning": "clip_vit_large_14",
        },
    )
    DEFAULT = COMPVIS_LAION
