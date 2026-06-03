"""Pretrained-weight declarations for the CSPNet family.

Three paper-cited variants (Wang et al., CVPRW 2020) — all sourced
from timm's ``ra_in1k`` tag (RandAugment / RandErasing recipe, no
ImageNet-22k pretraining for the canonical CSPNet line).

Preset: ``crop=224``, timm ``crop_pct=0.887``/0.882/0.887 (so
``resize=256``), ``bilinear`` for the CSPResNet line, ``bilinear``
for CSPDarknet too — exact values pulled from
``timm.create_model(...).default_cfg`` at conversion time.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


@register_weights("cspresnet_50_cls")
class CSPResNet50Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.cspresnet_50_cls`.

    Wang et al. CVPRW 2020 CSP-ResNet-50 (21.6 M params, top-1 76.74%).

    Attributes
    ----------
    RA_IN1K : WeightEntry
        ImageNet-1k RandAugment recipe checkpoint (top-1 76.74%),
        sourced from ``timm/cspresnet50.ra_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`RA_IN1K`.

    Notes
    -----
    Reference: Wang, Liao, Wu, Chen, Hsieh, Yeh, *"CSPNet: A New Backbone
    that can Enhance Learning Capability of CNN"*, CVPRW 2020
    (arXiv:1911.11929).

    Examples
    --------
    >>> from lucid.models import cspresnet_50_cls
    >>> model = cspresnet_50_cls(pretrained=True).eval()
    """

    RA_IN1K = WeightEntry(
        url=f"{HUB_BASE}/cspresnet-50/resolve/main/RA_IN1K/model.safetensors",
        sha256="f02d1a547ee3792f61a4510d171c04c2a3321e7c6818e7867a3f3f326b99e554",
        num_classes=1000,
        transforms=ImageClassification(
            crop_size=256, resize_size=288, interpolation="bilinear"
        ),
        meta={
            "tag": "RA_IN1K",
            "source": "timm/cspresnet50.ra_in1k",
            "license": "apache-2.0",
            "num_params": 21_620_000,
            "metrics": {"ImageNet-1k": {"acc@1": 76.74}},
        },
    )
    DEFAULT = RA_IN1K


@register_weights("cspresnext_50_cls")
class CSPResNeXt50Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.cspresnext_50_cls`.

    Wang et al. CVPRW 2020 CSP-ResNeXt-50 (20.6 M params, top-1 80.04%).

    Attributes
    ----------
    RA_IN1K : WeightEntry
        ImageNet-1k RandAugment recipe checkpoint (top-1 80.04%),
        sourced from ``timm/cspresnext50.ra_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`RA_IN1K`.

    Notes
    -----
    Reference: Wang, Liao, Wu, Chen, Hsieh, Yeh, *"CSPNet: A New Backbone
    that can Enhance Learning Capability of CNN"*, CVPRW 2020
    (arXiv:1911.11929).

    Examples
    --------
    >>> from lucid.models import cspresnext_50_cls
    >>> model = cspresnext_50_cls(pretrained=True).eval()
    """

    RA_IN1K = WeightEntry(
        url=f"{HUB_BASE}/cspresnext-50/resolve/main/RA_IN1K/model.safetensors",
        sha256="375b1e34949c133601cd65c1bcf2309715c9fef9684c65338a30f36b400eede4",
        num_classes=1000,
        transforms=ImageClassification(
            crop_size=224, resize_size=256, interpolation="bilinear"
        ),
        meta={
            "tag": "RA_IN1K",
            "source": "timm/cspresnext50.ra_in1k",
            "license": "apache-2.0",
            "num_params": 20_570_000,
            "metrics": {"ImageNet-1k": {"acc@1": 80.04}},
        },
    )
    DEFAULT = RA_IN1K


@register_weights("cspdarknet_53_cls")
class CSPDarknet53Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.cspdarknet_53_cls`.

    Wang et al. CVPRW 2020 CSP-Darknet-53 (27.6 M params, top-1 80.06%).

    Attributes
    ----------
    RA_IN1K : WeightEntry
        ImageNet-1k RandAugment recipe checkpoint (top-1 80.06%),
        sourced from ``timm/cspdarknet53.ra_in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`RA_IN1K`.

    Notes
    -----
    Reference: Wang, Liao, Wu, Chen, Hsieh, Yeh, *"CSPNet: A New Backbone
    that can Enhance Learning Capability of CNN"*, CVPRW 2020
    (arXiv:1911.11929).

    Examples
    --------
    >>> from lucid.models import cspdarknet_53_cls
    >>> model = cspdarknet_53_cls(pretrained=True).eval()
    """

    RA_IN1K = WeightEntry(
        url=f"{HUB_BASE}/cspdarknet-53/resolve/main/RA_IN1K/model.safetensors",
        sha256="d951deae3e98baa8e1fc33c18235ef399f872605a29b1a24f2d89a61f5d4312c",
        num_classes=1000,
        transforms=ImageClassification(
            crop_size=256, resize_size=288, interpolation="bilinear"
        ),
        meta={
            "tag": "RA_IN1K",
            "source": "timm/cspdarknet53.ra_in1k",
            "license": "apache-2.0",
            "num_params": 27_610_000,
            "metrics": {"ImageNet-1k": {"acc@1": 80.06}},
        },
    )
    DEFAULT = RA_IN1K
