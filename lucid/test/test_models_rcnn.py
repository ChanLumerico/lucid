import pytest

import lucid
import lucid.nn as nn

from lucid.models import RCNN, RCNNConfig


class _ToyBackbone(nn.Module):
    def __init__(self, out_channels: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.net(x)


class _FixedSelectiveSearch(nn.Module):
    def __init__(self, boxes) -> None:
        super().__init__()
        self.boxes = boxes

    def forward(self, _image):
        return self.boxes


def _small_rcnn_config() -> RCNNConfig:
    return RCNNConfig(
        backbone=_ToyBackbone(out_channels=8),
        feat_dim=8,
        num_classes=3,
        warper_output_size=(8, 8),
    )


def test_rcnn_public_imports() -> None:
    assert RCNN is not None
    assert RCNNConfig is not None


def test_rcnn_accepts_config_object() -> None:
    config = _small_rcnn_config()

    model = RCNN(config)

    assert model.config is config


def test_rcnn_forward_with_custom_rois() -> None:
    model = RCNN(_small_rcnn_config())
    images = lucid.ones(2, 3, 16, 16)
    rois = [
        lucid.Tensor([[1, 1, 10, 10]], dtype=lucid.Float32),
        lucid.Tensor([[2, 2, 12, 12]], dtype=lucid.Float32),
    ]

    cls_scores, bbox_deltas, feats = model(images, rois=rois, return_feats=True)

    assert cls_scores.shape == (2, 3)
    assert bbox_deltas.shape == (2, 3, 4)
    assert feats.shape == (2, 8)


def test_rcnn_predict_with_stubbed_selective_search() -> None:
    model = RCNN(_small_rcnn_config())
    model.ss = _FixedSelectiveSearch(
        lucid.Tensor([[1, 1, 10, 10], [3, 3, 12, 12]], dtype=lucid.Float32)
    )

    results = model.predict(lucid.ones(2, 3, 16, 16), max_det_per_img=4)

    assert len(results) == 2
    for result in results:
        assert set(result.keys()) == {"boxes", "scores", "labels"}
        assert result["boxes"].ndim == 2
        assert result["boxes"].shape[1] == 4
        assert result["scores"].ndim == 1
        assert result["labels"].ndim == 1
        assert result["boxes"].shape[0] == result["scores"].shape[0]
        assert result["boxes"].shape[0] == result["labels"].shape[0]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"backbone": object()},
        {"feat_dim": 0},
        {"num_classes": 0},
        {"image_means": (0.1, 0.2)},
        {"pixel_scale": 0.0},
        {"warper_output_size": (8,)},
        {"warper_output_size": (8, 0)},
        {"nms_iou_thresh": -0.1},
        {"nms_iou_thresh": 1.1},
    ),
)
def test_rcnn_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "backbone": _ToyBackbone(out_channels=8),
        "feat_dim": 8,
        "num_classes": 3,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        RCNNConfig(**params)
