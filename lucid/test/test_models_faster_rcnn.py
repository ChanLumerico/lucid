import pytest

import lucid
import lucid.nn as nn

from lucid.models import (
    FasterRCNN,
    FasterRCNNConfig,
    faster_rcnn_resnet_50_fpn,
    faster_rcnn_resnet_101_fpn,
)


class _ToyBackbone(nn.Module):
    def __init__(self, out_channels: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def _small_faster_rcnn_config() -> FasterRCNNConfig:
    return FasterRCNNConfig(
        backbone=_ToyBackbone(out_channels=4),
        feat_channels=4,
        num_classes=3,
        anchor_sizes=(8,),
        aspect_ratios=(1.0,),
        anchor_stride=4,
        pool_size=(2, 2),
        hidden_dim=8,
        dropout=0.0,
    )


def test_faster_rcnn_public_imports() -> None:
    assert FasterRCNN is not None
    assert FasterRCNNConfig is not None
    assert faster_rcnn_resnet_50_fpn is not None
    assert faster_rcnn_resnet_101_fpn is not None


def test_faster_rcnn_accepts_config_object() -> None:
    config = _small_faster_rcnn_config()

    model = FasterRCNN(config)

    assert model.config is config


def test_faster_rcnn_forward_with_custom_rois() -> None:
    model = FasterRCNN(_small_faster_rcnn_config())
    images = lucid.ones(2, 3, 16, 16)
    rois = lucid.Tensor([[0.1, 0.1, 0.6, 0.6], [0.2, 0.2, 0.8, 0.8]])
    roi_idx = lucid.Tensor([0, 1], dtype=lucid.Int32)

    cls_logits, bbox_deltas, feats = model(
        images, rois=rois, roi_idx=roi_idx, return_feats=True
    )

    assert cls_logits.shape == (2, 3)
    assert bbox_deltas.shape == (2, 12)
    assert feats.shape == (2, 4, 16, 16)


def test_faster_rcnn_get_loss_returns_scalar_losses() -> None:
    model = FasterRCNN(_small_faster_rcnn_config())
    images = lucid.ones(1, 3, 16, 16)
    targets = [
        {
            "boxes": lucid.Tensor([[2, 2, 10, 10]], dtype=lucid.Float32),
            "labels": lucid.Tensor([1], dtype=lucid.Int32),
        }
    ]

    losses = model.get_loss(images, targets)

    assert set(losses.keys()) == {
        "rpn_cls_loss",
        "rpn_reg_loss",
        "roi_cls_loss",
        "roi_reg_loss",
        "total_loss",
    }
    for loss in losses.values():
        assert loss.shape == ()


def test_faster_rcnn_factory_builds_resnet_fpn_variants() -> None:
    model_50 = faster_rcnn_resnet_50_fpn(
        num_classes=3,
        anchor_sizes=(16,),
        aspect_ratios=(1.0,),
        anchor_stride=8,
        pool_size=(2, 2),
        dropout=0.0,
    )
    model_101 = faster_rcnn_resnet_101_fpn(
        num_classes=3,
        anchor_sizes=(16,),
        aspect_ratios=(1.0,),
        anchor_stride=8,
        pool_size=(2, 2),
        dropout=0.0,
    )

    assert model_50.config.use_fpn is True
    assert model_101.config.use_fpn is True

    cls_logits, bbox_deltas = model_50(lucid.ones(1, 3, 32, 32))
    assert cls_logits.ndim == 2
    assert bbox_deltas.ndim == 2
    assert cls_logits.shape[1] == 3
    assert bbox_deltas.shape[1] == 12


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
    ),
)
def test_faster_rcnn_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "backbone": _ToyBackbone(out_channels=4),
        "feat_channels": 4,
        "num_classes": 3,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        FasterRCNNConfig(**params)
