"""Registry factories for DETR variants."""

from lucid.models._registry import register_model
from lucid.models.vision.detr._config import DETRConfig
from lucid.models.vision.detr._model import DETRForObjectDetection

_CFG_R50 = DETRConfig(
    num_classes=80,
    in_channels=3,
    backbone_layers=(3, 4, 6, 3),
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    num_bbox_layers=3,
    bbox_hidden_dim=256,
    score_thresh=0.7,
)

_CFG_R101 = DETRConfig(
    num_classes=80,
    in_channels=3,
    backbone_layers=(3, 4, 23, 3),  # ResNet-101
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    num_bbox_layers=3,
    bbox_hidden_dim=256,
    score_thresh=0.7,
)


def _det(cfg: DETRConfig, kw: dict[str, object]) -> DETRForObjectDetection:
    return DETRForObjectDetection(DETRConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


@register_model(
    task="object-detection",
    family="detr",
    model_type="detr",
    model_class=DETRForObjectDetection,
    default_config=_CFG_R50,
)
def detr_resnet50(
    pretrained: bool = False,
    **overrides: object,
) -> DETRForObjectDetection:
    r"""DETR with ResNet-50 backbone (Carion et al., ECCV 2020).

    Builds a :class:`DETRForObjectDetection` with the paper-cited
    ResNet-50 + transformer configuration: 100 object queries, a 6-layer
    encoder + 6-layer decoder with ``d_model = 256``, 8 attention heads,
    and ``dim_feedforward = 2048``.  Reaches COCO test-dev mAP of 42.0%
    (paper Table 1).  Approximately 41M parameters and a notable
    *no-anchor / no-NMS* design.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`DETRConfig`
        (``num_queries``, ``num_classes``, ``dropout``,
        ``num_encoder_layers``, ``num_decoder_layers``, ``d_model``, ...).

    Returns
    -------
    DETRForObjectDetection
        Detector with the DETR ResNet-50 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Carion et al., "End-to-End Object Detection with Transformers",
    ECCV 2020 (arXiv:2005.12872).  The key insight is bipartite Hungarian
    matching during training, which makes the set-prediction loss
    permutation-invariant in the query dimension.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.detr import detr_resnet50
    >>> model = detr_resnet50()
    >>> x = lucid.randn(1, 3, 800, 800)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 100, 81)
    """
    return _det(_CFG_R50, overrides)


@register_model(
    task="object-detection",
    family="detr",
    model_type="detr",
    model_class=DETRForObjectDetection,
    default_config=_CFG_R101,
)
def detr_resnet101(
    pretrained: bool = False,
    **overrides: object,
) -> DETRForObjectDetection:
    r"""DETR with ResNet-101 backbone (Carion et al., ECCV 2020).

    Builds a :class:`DETRForObjectDetection` with the same transformer
    head as :func:`detr_resnet50` but a deeper ResNet-101 backbone
    (``[3, 4, 23, 3]`` bottleneck blocks).  Approximately 60M parameters
    and COCO test-dev mAP of 43.5% (paper Table 1, DETR-R101 row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`DETRConfig`.

    Returns
    -------
    DETRForObjectDetection
        Detector with the DETR ResNet-101 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Carion et al., "End-to-End Object Detection with Transformers",
    ECCV 2020 (arXiv:2005.12872).  Switching to ResNet-101 buys ~1.5
    points of AP at the cost of ~50% more parameters in the backbone;
    the transformer head is unchanged.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.detr import detr_resnet101
    >>> model = detr_resnet101()
    >>> x = lucid.randn(1, 3, 800, 800)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 100, 81)
    """
    return _det(_CFG_R101, overrides)
