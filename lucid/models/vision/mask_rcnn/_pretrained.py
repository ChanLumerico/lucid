"""Registry factories for Mask R-CNN variants."""

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.mask_rcnn._config import MaskRCNNConfig
from lucid.models.vision.mask_rcnn._model import MaskRCNNForObjectDetection
from lucid.models.vision.mask_rcnn._weights import MaskRCNNResNet50FPNWeights

# COCO config — the pretrained checkpoint uses num_classes=91 (90 COCO
# categories + background slot 0).
_CFG_R50_FPN = MaskRCNNConfig(num_classes=91)


def _seg(cfg: MaskRCNNConfig, kw: dict[str, object]) -> MaskRCNNForObjectDetection:
    return MaskRCNNForObjectDetection(
        replace(cfg, **cast(dict[str, Any], kw)) if kw else cfg
    )


# reason: mask_rcnn adds a typed weights= kwarg (per-model WeightsEnum); the
# ModelFactory protocol predates the v3.1 weights system and still names only
# pretrained + **overrides.
@register_model(  # type: ignore[arg-type]
    task="object-detection",
    family="mask_rcnn",
    model_type="mask_rcnn",
    model_class=MaskRCNNForObjectDetection,
    default_config=_CFG_R50_FPN,
)
def mask_rcnn(
    pretrained: bool | str = False,
    *,
    weights: MaskRCNNResNet50FPNWeights | None = None,
    **overrides: object,
) -> MaskRCNNForObjectDetection:
    r"""Mask R-CNN with a ResNet-50-FPN backbone (He et al., ICCV 2017).

    Alias of :func:`mask_rcnn_resnet50_fpn` — the canonical reference
    configuration: Faster R-CNN's ResNet-50 trunk (frozen BN) + Feature
    Pyramid Network + Region Proposal Network + Fast R-CNN box head, plus
    a parallel FCN mask branch.  The COCO config uses ``num_classes = 91``.
    Reaches COCO box AP 37.9 / mask AP 34.6 (reference
    ``maskrcnn_resnet50_fpn``).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`MaskRCNNResNet50FPNWeights.COCO_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : MaskRCNNResNet50FPNWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`MaskRCNNConfig`
        (``num_classes``, ``rpn_pre_nms_top_n``, ``score_thresh``, ...).

    Returns
    -------
    MaskRCNNForObjectDetection
        Detector with the ResNet-50-FPN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask_rcnn import mask_rcnn
    >>> model = mask_rcnn(num_classes=91)
    >>> model.eval()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.pred_masks.shape[-2:]
    (28, 28)
    """
    entry = weights_mod.resolve_weights(MaskRCNNResNet50FPNWeights, pretrained, weights)
    model = _seg(_CFG_R50_FPN, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mask_rcnn")
    return model


# reason: mask_rcnn_resnet50_fpn adds a typed weights= kwarg (per-model WeightsEnum);
# the ModelFactory protocol predates the v3.1 weights system and still names only
# pretrained + **overrides.
@register_model(  # type: ignore[arg-type]
    task="object-detection",
    family="mask_rcnn",
    model_type="mask_rcnn",
    model_class=MaskRCNNForObjectDetection,
    default_config=_CFG_R50_FPN,
)
def mask_rcnn_resnet50_fpn(
    pretrained: bool | str = False,
    *,
    weights: MaskRCNNResNet50FPNWeights | None = None,
    **overrides: object,
) -> MaskRCNNForObjectDetection:
    r"""Mask R-CNN with a ResNet-50-FPN backbone (COCO-pretrained).

    Builds the reference ``maskrcnn_resnet50_fpn`` detector: Faster
    R-CNN's ResNet-50 backbone with frozen batch-norm, a 5-level Feature
    Pyramid Network (P2-P5 + a parameter-free pool level for RPN), anchors
    at sizes ``((32,),(64,),(128,),(256,),(512,))`` and ratios
    ``(0.5, 1.0, 2.0)``, a Fast R-CNN box head (TwoMLPHead +
    FastRCNNPredictor) over MultiScale RoI Align (7×7), and a parallel
    mask branch (MaskRCNNHeads + MaskRCNNPredictor) over MultiScale RoI
    Align (14×14) emitting ``28×28`` per-class masks.  COCO box AP 37.9 /
    mask AP 34.6, ~44.4M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True`` →
        the ``DEFAULT`` tag (:attr:`MaskRCNNResNet50FPNWeights.COCO_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : MaskRCNNResNet50FPNWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`MaskRCNNConfig`.

    Returns
    -------
    MaskRCNNForObjectDetection
        Detector with the ResNet-50-FPN configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    Pretrained weights are converted from the reference
    ``MaskRCNN_ResNet50_FPN_Weights.COCO_V1`` checkpoint and hosted under
    ``lucid-dl/mask-rcnn-resnet-50-fpn``.  The detector expects an already
    resized + normalised image batch (the
    :class:`~lucid.utils.transforms.Detection` preset).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask_rcnn import mask_rcnn_resnet50_fpn
    >>> model = mask_rcnn_resnet50_fpn()
    >>> model.eval()
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.pred_masks.shape[-2:]
    (28, 28)
    """
    entry = weights_mod.resolve_weights(MaskRCNNResNet50FPNWeights, pretrained, weights)
    model = _seg(_CFG_R50_FPN, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="mask_rcnn_resnet50_fpn")
    return model
