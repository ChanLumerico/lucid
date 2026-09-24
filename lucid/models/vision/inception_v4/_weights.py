"""Pretrained-weight declarations for the Inception-v4 family.

One paper-cited variant (Szegedy et al., AAAI 2017) — sourced from
timm's ``tf_in1k`` checkpoint, the TensorFlow-Slim weights ported into
timm.  That preset is *not* the ImageNet default: it
evaluates at ``299×299`` with ``crop_pct=0.875`` (→ resize 341, floored
as the source pipeline floors it), **bicubic** interpolation, and
``(0.5, 0.5, 0.5)`` mean/std (the TF-Slim ``[-1, 1]`` normalisation) —
all replicated exactly in ``transforms`` below.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


@register_weights("inception_v4_cls")
class InceptionV4Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.inception_v4_cls`.

    Szegedy et al. AAAI 2017 (42.7 M params, top-1 80.14%).

    Attributes
    ----------
    TF_IN1K : WeightEntry
        TensorFlow-Slim ImageNet-1k checkpoint (top-1 80.144% / top-5
        94.982% at 299, per timm's ``results-imagenet.csv`` for this
        exact tag), sourced from ``timm/inception_v4.tf_in1k``.  The
        paper's Table 2 reports 80.0% / 95.0% single-crop for its own
        evaluation.
    DEFAULT : WeightEntry
        Alias for :attr:`TF_IN1K`.

    Notes
    -----
    Reference: Szegedy, Ioffe, Vanhoucke, Alemi, *"Inception-v4,
    Inception-ResNet and the Impact of Residual Connections on
    Learning"*, AAAI 2017 (arXiv:1602.07261).

    Examples
    --------
    >>> from lucid.models.weights import InceptionV4Weights
    >>> InceptionV4Weights.DEFAULT is InceptionV4Weights.TF_IN1K
    True
    >>> tf = InceptionV4Weights.TF_IN1K.transforms()
    >>> tf.crop_size, tf.resize_size, tf.interpolation
    (299, 341, 'bicubic')
    >>> tuple(tf.mean), tuple(tf.std)
    ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    """

    TF_IN1K = WeightEntry(
        url=f"{HUB_BASE}/inception-v4/resolve/main/TF_IN1K/model.safetensors",
        sha256="f7ea22ee0be7a9ae0d3dec69eefe3c9d5b3339ef1624f2d86ca3c3b596a63ee4",
        num_classes=1000,
        transforms=ImageClassification(
            crop_size=299,
            resize_size=341,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            interpolation="bicubic",
        ),
        meta={
            "tag": "TF_IN1K",
            "source": "timm/inception_v4.tf_in1k",
            "license": "apache-2.0",
            "num_params": 42_679_816,
            "metrics": {"ImageNet-1k": {"acc@1": 80.144, "acc@5": 94.982}},
        },
    )
    DEFAULT = TF_IN1K
