"""Pretrained-weight declarations for the ConvNeXt family.

ImageNet-1k weights for the four canonical paper-cited variants
(``convnext_tiny`` / ``convnext_small`` / ``convnext_base`` /
``convnext_large``).  Converted from torchvision's official
``ConvNeXt_*_Weights.IMAGENET1K_V1`` tag — which in turn re-distributes
Facebook AI Research's published checkpoints — re-hosted under the
``lucid-dl`` org so the per-family enum can pull a Lucid-native
SafeTensors blob without a torchvision dependency at load time.

The ``convnext_xlarge`` variant (paper Table 9, 350M params) ships in
a follow-up commit sourced from timm's ``fb_in22k_ft_in1k`` checkpoint;
torchvision does not publish a 1k-class xlarge head.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


@register_weights("convnext_tiny_cls")
class ConvNeXtTinyWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_tiny_cls`.

    Currently ships :attr:`IMAGENET1K_V1` — torchvision's ImageNet-1k
    checkpoint (acc@1 = 82.520 / acc@5 = 96.146) re-hosted under
    ``huggingface.co/lucid-dl/convnext-tiny``.
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/convnext-tiny/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="9bddab322a53aada74d717da131aba4c8fee460661209fe443c4bd743bd91c22",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=236),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
            "license": "mit",
            "num_params": 28_589_128,
            "gflops": 4.456,
            "file_size_mb": 109.1,
            "metrics": {"ImageNet-1k": {"acc@1": 82.520, "acc@5": 96.146}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("convnext_small_cls")
class ConvNeXtSmallWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_small_cls`.

    Currently ships :attr:`IMAGENET1K_V1` — torchvision's ImageNet-1k
    checkpoint (acc@1 = 83.616 / acc@5 = 96.650).
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/convnext-small/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="6412a3aab20fc5380288b68742a3e1b8015fcaadf0df3a0fc6ef3e4ab43f9212",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=230),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ConvNeXt_Small_Weights.IMAGENET1K_V1",
            "license": "mit",
            "num_params": 50_223_688,
            "gflops": 8.684,
            "file_size_mb": 191.6,
            "metrics": {"ImageNet-1k": {"acc@1": 83.616, "acc@5": 96.650}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("convnext_base_cls")
class ConvNeXtBaseWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_base_cls`.

    Currently ships :attr:`IMAGENET1K_V1` — torchvision's ImageNet-1k
    checkpoint (acc@1 = 84.062 / acc@5 = 96.870).
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/convnext-base/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="4c2c3389b534a134ed9c2531e9b89963aa0c8ab1484ce77b364d68eda437fcb7",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=232),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ConvNeXt_Base_Weights.IMAGENET1K_V1",
            "license": "mit",
            "num_params": 88_591_464,
            "gflops": 15.355,
            "file_size_mb": 338.0,
            "metrics": {"ImageNet-1k": {"acc@1": 84.062, "acc@5": 96.870}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("convnext_large_cls")
class ConvNeXtLargeWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.convnext_large_cls`.

    Currently ships :attr:`IMAGENET1K_V1` — torchvision's ImageNet-1k
    checkpoint (acc@1 = 84.414 / acc@5 = 96.976).
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/convnext-large/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="4f335d8136015448a3c62f58d4493a2dcc4adc5102e1c23dd6d3f57b7d695d26",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=232),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ConvNeXt_Large_Weights.IMAGENET1K_V1",
            "license": "mit",
            "num_params": 197_767_336,
            "gflops": 34.361,
            "file_size_mb": 754.4,
            "metrics": {"ImageNet-1k": {"acc@1": 84.414, "acc@5": 96.976}},
        },
    )
    DEFAULT = IMAGENET1K_V1
