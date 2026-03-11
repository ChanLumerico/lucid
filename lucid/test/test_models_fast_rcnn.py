import pytest

import lucid
import lucid.nn as nn

from lucid.models import FastRCNN, FastRCNNConfig


class _ToyBackbone(nn.Module):
    def __init__(self, out_channels: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class _FixedProposalGenerator:
    def __init__(self, boxes) -> None:
        self.boxes = boxes

    def __call__(self, _image):
        return self.boxes


def _small_fast_rcnn_config() -> FastRCNNConfig:
    return FastRCNNConfig(
        backbone=_ToyBackbone(out_channels=4),
        feat_channels=4,
        num_classes=3,
        pool_size=(2, 2),
        hidden_dim=8,
        dropout=0.0,
    )


def test_fast_rcnn_public_imports() -> None:
    assert FastRCNN is not None
    assert FastRCNNConfig is not None


def test_fast_rcnn_accepts_config_object() -> None:
    config = _small_fast_rcnn_config()

    model = FastRCNN(config)

    assert model.config is config


def test_fast_rcnn_forward_with_custom_rois() -> None:
    model = FastRCNN(_small_fast_rcnn_config())
    images = lucid.ones(2, 3, 16, 16)
    rois = lucid.Tensor([[0.1, 0.1, 0.6, 0.6], [0.2, 0.2, 0.8, 0.8]])
    roi_idx = lucid.Tensor([0, 1], dtype=lucid.Int32)

    cls_logits, bbox_deltas, feats = model(
        images, rois=rois, roi_idx=roi_idx, return_feats=True
    )

    assert cls_logits.shape == (2, 3)
    assert bbox_deltas.shape == (2, 12)
    assert feats.shape == (2, 4, 16, 16)


def test_fast_rcnn_predict_with_stubbed_proposal_generator() -> None:
    config = _small_fast_rcnn_config()
    config.proposal_generator = _FixedProposalGenerator(
        lucid.Tensor([[1, 1, 10, 10], [3, 3, 12, 12]], dtype=lucid.Float32)
    )
    model = FastRCNN(config)

    detections = model.predict(lucid.ones(1, 3, 16, 16), score_thresh=0.0, top_k=4)

    assert isinstance(detections, list)
    assert len(detections) > 0
    for detection in detections:
        assert set(detection.keys()) == {"boxes", "scores", "labels"}
        assert detection["boxes"].ndim == 2
        assert detection["boxes"].shape[1] == 4
        assert detection["scores"].ndim == 1
        assert detection["labels"].ndim == 1
        assert detection["boxes"].shape[0] == detection["scores"].shape[0]
        assert detection["boxes"].shape[0] == detection["labels"].shape[0]


def test_fast_rcnn_get_loss_returns_three_scalars() -> None:
    model = FastRCNN(_small_fast_rcnn_config())
    cls_logits = lucid.random.randn(2, 3)
    bbox_deltas = lucid.random.randn(2, 12)
    labels = lucid.Tensor([1, 0], dtype=lucid.Int32)
    reg_targets = lucid.random.randn(2, 4)

    total_loss, cls_loss, reg_loss = model.get_loss(
        cls_logits, bbox_deltas, labels, reg_targets
    )

    assert total_loss.shape == ()
    assert cls_loss.shape == ()
    assert reg_loss.shape == ()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"backbone": object()},
        {"feat_channels": 0},
        {"num_classes": 0},
        {"pool_size": (2,)},
        {"pool_size": (2, 0)},
        {"hidden_dim": 0},
        {"bbox_reg_means": (0.0, 0.0, 0.0)},
        {"bbox_reg_stds": (0.1, 0.1, 0.2)},
        {"bbox_reg_stds": (0.1, 0.1, 0.0, 0.2)},
        {"dropout": 1.0},
        {"proposal_generator": object()},
    ),
)
def test_fast_rcnn_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "backbone": _ToyBackbone(out_channels=4),
        "feat_channels": 4,
        "num_classes": 3,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        FastRCNNConfig(**params)
