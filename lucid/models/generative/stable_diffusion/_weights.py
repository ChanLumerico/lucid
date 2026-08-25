"""Pretrained-weight declarations for Stable Diffusion.

One entry, holding both sub-models.  The released archive ships the
autoencoder and the U-Net separately and they are useless apart — a
latent from one decoder is not a latent for another — so they are
converted and published together under a single tag.

Checked numerically against the reference rather than merely loaded.
With these weights in place, on a 512-pixel image and the 64x64 latent
that implies:

===================  =========  =========
path                 Metal      CPU
===================  =========  =========
U-Net epsilon        4.5e-06    4.5e-06
autoencoder encoder  3.4e-05    2.7e-03
autoencoder decoder  5.9e-05    8.3e-04
===================  =========  =========

**The two columns differ for a reason that is not this conversion.**
Lucid's CPU ``group_norm`` kernel accumulates a group's mean and
variance in one ``float`` with a serial sum, so its error grows with
the size of the group rather than staying flat: 8e-07 over 4k elements
and 2e-04 over the million a 512-pixel autoencoder stage reduces.  MLX
does not, which is why the Metal column holds its accuracy while the
CPU column loses two digits between 256 and 512 pixels.  The U-Net is
largely spared because its widest group is 41k elements rather than a
million.  Read the Metal column as this conversion's accuracy; the gap
between them is a framework issue and is tracked apart from this family.

That check is what this family needed most, because the conversion had
already passed a stricter-looking one.  The parameter counts matched the
release *tensor for tensor* while three defects remained: the attention
was built with forty heads of eight channels where the reference uses
eight of forty, the autoencoder's downsampler padded symmetrically where
the reference pads bottom-right only, and every normalisation ran at the
framework's default epsilon where the release uses 1e-6 throughout the
autoencoder and inside the U-Net's spatial transformers.  None of the
three changes a shape, a count, or anything a forward pass reports; the
last one alone was a factor of twenty on the U-Net's agreement.

The text encoder is not here.  This family takes conditioning as an
already-encoded sequence, and the released models condition on a frozen
CLIP ViT-L/14 text tower — :func:`~lucid.models.clip_vit_large_14`,
whose own weights are published separately.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

__all__ = ["StableDiffusionWeights"]

# The first stage works in [-1, 1] rather than on dataset statistics —
# it is an autoencoder, not a classifier, and has no notion of a mean
# image to subtract.
_PRESET_512 = ImageClassification(
    crop_size=512,
    resize_size=512,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)


@register_weights("stable_diffusion")
class StableDiffusionWeights(WeightsEnum):
    """Pretrained weights for :func:`lucid.models.stable_diffusion`.

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
