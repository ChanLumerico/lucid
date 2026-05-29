"""Pretrained-weight declarations for the Xception family.

One paper-cited variant (Chollet, CVPR 2017) — converted from timm's
``legacy_xception.tf_in1k`` checkpoint (originally the Cadene port of
Chollet's Keras weights).  Unlike the standard ImageNet eval pipeline,
this preset uses a 299×299 crop with ``crop_pct = 0.8975`` (→ 333
resize), **bicubic** interpolation, and a symmetric ``mean = std = 0.5``
normalisation (NOT ImageNet stats) — replicated exactly here from the
source ``default_cfg``.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Source preset: 299 crop / 333 resize / bicubic / mean=std=0.5.
_PRESET = ImageClassification(
    crop_size=299,
    resize_size=333,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    interpolation="bicubic",
)


@register_weights("xception_cls")
class XceptionWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.xception_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`TF_IN1K`) converted
    from timm's ``legacy_xception.tf_in1k``, re-hosted under
    ``huggingface.co/lucid-dl/xception``.
    """

    TF_IN1K = WeightEntry(
        url=f"{HUB_BASE}/xception/resolve/main/TF_IN1K/model.safetensors",
        sha256="8143bf6634e62129c20d48f0cb81ce45c67069709c1370ce7612506c4f8b5dc6",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "TF_IN1K",
            "source": "timm/legacy_xception.tf_in1k",
            "license": "apache-2.0",
            "num_params": 22_855_952,
            "metrics": {"ImageNet-1k": {"acc@1": 79.0}},
        },
    )
    DEFAULT = TF_IN1K
