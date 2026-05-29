"""Pretrained-weight declarations for the MobileNet v2 family.

One paper-cited variant (Sandler et al., CVPR 2018) ships an
ImageNet-1k checkpoint converted from torchvision's
``MobileNet_V2_Weights.IMAGENET1K_V1`` tag.  Preset is the standard
ImageNet eval pipeline (224 crop / 256 resize / bilinear / ImageNet
stats).

Only the full-width (:math:`\\alpha = 1.0`) classifier ships pretrained
weights — torchvision does not distribute a 0.75-width checkpoint, so
:func:`lucid.models.mobilenet_v2_075_cls` has no weights enum.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=256)


@register_weights("mobilenet_v2_cls")
class MobileNetV2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mobilenet_v2_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`IMAGENET1K_V1`) —
    the V1 weights distributed by the reference-framework model zoo,
    re-hosted under ``huggingface.co/lucid-dl/mobilenet-v2`` with the
    official ``acc@1=71.878 / acc@5=90.286`` validation metrics.
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/mobilenet-v2/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="1717e0d3229049bd60f9500a4b8e8a0dbfeae4a792558b062331f9651dd7b9db",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/MobileNet_V2_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 3_504_872,
            "metrics": {"ImageNet-1k": {"acc@1": 71.878, "acc@5": 90.286}},
        },
    )
    DEFAULT = IMAGENET1K_V1
