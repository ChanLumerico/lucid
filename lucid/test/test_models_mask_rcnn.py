import pytest

import lucid
import lucid.nn as nn

from lucid.models import (
    MaskRCNN,
    MaskRCNNConfig,
    mask_rcnn_resnet_50_fpn,
    mask_rcnn_resnet_101_fpn,
)


class _ToyBackbone(nn.Module):
    def __init__(self, out_channels: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.net(x)


def _small_mask_rcnn_config(**kwargs: object) -> MaskRCNNConfig:
    params = {
        "backbone": _ToyBackbone(out_channels=4),
        "feat_channels": 4,
        "num_classes": 3,
        "anchor_sizes": (8,),
        "aspect_ratios": (1.0,),
        "anchor_stride": 4,
        "pool_size": (2, 2),
        "hidden_dim": 8,
        "dropout": 0.0,
        "mask_pool_size": (2, 2),
        "mask_hidden_channels": 4,
        "mask_out_size": 4,
    }
    params.update(kwargs)
    return MaskRCNNConfig(**params)


def test_mask_rcnn_public_imports() -> None:
    assert MaskRCNN is not None
    assert MaskRCNNConfig is not None
    assert mask_rcnn_resnet_50_fpn is not None
    assert mask_rcnn_resnet_101_fpn is not None


def test_mask_rcnn_accepts_config_object() -> None:
    config = _small_mask_rcnn_config()

    model = MaskRCNN(config)

    assert model.config is config


def test_mask_rcnn_forward_with_custom_rois() -> None:
    model = MaskRCNN(_small_mask_rcnn_config())
    images = lucid.ones(2, 3, 16, 16)
    rois = lucid.Tensor([[0.1, 0.1, 0.6, 0.6], [0.2, 0.2, 0.8, 0.8]])
    roi_idx = lucid.Tensor([0, 1], dtype=lucid.Int32)

    cls_logits, bbox_deltas, mask_logits = model(images, rois=rois, roi_idx=roi_idx)

    assert cls_logits.shape == (2, 3)
    assert bbox_deltas.shape == (2, 12)
    assert mask_logits.shape == (2, 3, 4, 4)


def test_mask_rcnn_get_loss_returns_scalar_losses() -> None:
    model = MaskRCNN(_small_mask_rcnn_config())
    images = lucid.ones(1, 3, 16, 16)
    mask = lucid.zeros((1, 16, 16), dtype=lucid.Float32)
    mask[0, 2:10, 2:10] = 1.0
    targets = [
        {
            "boxes": lucid.Tensor([[2, 2, 10, 10]], dtype=lucid.Float32),
            "labels": lucid.Tensor([1], dtype=lucid.Int32),
            "masks": mask,
        }
    ]

    losses = model.get_loss(images, targets)

    assert set(losses.keys()) == {
        "rpn_cls_loss",
        "rpn_reg_loss",
        "roi_cls_loss",
        "roi_reg_loss",
        "mask_loss",
        "total_loss",
    }
    for loss in losses.values():
        assert loss.shape == ()


def test_mask_rcnn_predict_returns_detection_dicts() -> None:
    model = MaskRCNN(_small_mask_rcnn_config())

    detections = model.predict(lucid.ones(1, 3, 16, 16), top_k=5)

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert set(detections[0].keys()) == {"boxes", "scores", "labels", "masks"}


def test_mask_rcnn_factories_build_resnet_fpn_variants() -> None:
    model_50 = mask_rcnn_resnet_50_fpn(
        num_classes=3,
        anchor_sizes=(16,),
        aspect_ratios=(1.0,),
        anchor_stride=8,
        pool_size=(2, 2),
        dropout=0.0,
        mask_pool_size=(2, 2),
        mask_hidden_channels=8,
        mask_out_size=4,
    )
    model_101 = mask_rcnn_resnet_101_fpn(
        num_classes=3,
        anchor_sizes=(16,),
        aspect_ratios=(1.0,),
        anchor_stride=8,
        pool_size=(2, 2),
        dropout=0.0,
        mask_pool_size=(2, 2),
        mask_hidden_channels=8,
        mask_out_size=4,
    )

    assert model_50.config.use_fpn is True
    assert model_101.config.use_fpn is True

    cls_logits, bbox_deltas, mask_logits = model_50(lucid.ones(1, 3, 32, 32))
    cls_logits_101, bbox_deltas_101, mask_logits_101 = model_101(lucid.ones(1, 3, 32, 32))
    assert cls_logits.ndim == 2
    assert bbox_deltas.ndim == 2
    assert mask_logits.ndim == 4
    assert cls_logits.shape[1] == 3
    assert bbox_deltas.shape[1] == 12
    assert mask_logits.shape[1] == 3
    assert cls_logits_101.ndim == 2
    assert bbox_deltas_101.ndim == 2
    assert mask_logits_101.ndim == 4
    assert cls_logits_101.shape[1] == 3
    assert bbox_deltas_101.shape[1] == 12
    assert mask_logits_101.shape[1] == 3


@pytest.mark.parametrize(
    "kwargs",
    (
        {"backbone": object()},
        {"feat_channels": 0},
        {"num_classes": 0},
        {"use_fpn": "yes"},
        {"anchor_sizes": ()},
        {"anchor_sizes": (0,)},
        {"aspect_ratios": ()},
        {"aspect_ratios": (0.0,)},
        {"anchor_stride": 0},
        {"pool_size": (2,)},
        {"pool_size": (2, 0)},
        {"hidden_dim": 0},
        {"dropout": 1.0},
        {"mask_pool_size": (2,)},
        {"mask_pool_size": (2, 0)},
        {"mask_pool_size": (2, 3)},
        {"mask_hidden_channels": 0},
        {"mask_out_size": 0},
        {"mask_out_size": 6},
    ),
)
def test_mask_rcnn_config_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _small_mask_rcnn_config(**kwargs)
