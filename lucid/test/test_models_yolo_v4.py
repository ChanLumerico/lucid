import pytest

import lucid
import lucid.nn as nn

from lucid.models import YOLO_V4, YOLO_V4Config, yolo_v4


class _ToyBackbone(nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        return (
            lucid.ones(batch, 16, 4, 4),
            lucid.ones(batch, 32, 2, 2),
            lucid.ones(batch, 64, 1, 1),
        )


def _small_yolo_v4_config() -> YOLO_V4Config:
    return YOLO_V4Config(
        num_classes=3,
        anchors=[
            [(2, 2), (3, 3), (4, 4)],
            [(5, 5), (6, 6), (7, 7)],
            [(8, 8), (9, 9), (10, 10)],
        ],
        strides=[8, 16, 32],
        backbone=_ToyBackbone(),
        backbone_out_channels=(16, 32, 64),
        in_channels=(16, 32, 64),
        iou_aware_alpha=0.0,
        iou_branch_weight=0.0,
    )


def test_yolo_v4_public_imports() -> None:
    assert YOLO_V4 is not None
    assert YOLO_V4Config is not None
    assert yolo_v4 is not None


def test_yolo_v4_accepts_config_object() -> None:
    config = _small_yolo_v4_config()

    model = YOLO_V4(config)

    assert model.config is config


def test_yolo_v4_forward_outputs_three_scales() -> None:
    model = YOLO_V4(_small_yolo_v4_config())

    outputs = model(lucid.ones(1, 3, 32, 32))

    assert len(outputs) == 3
    assert outputs[0].shape == (1, 3 * (6 + 3), 4, 4)
    assert outputs[1].shape == (1, 3 * (6 + 3), 2, 2)
    assert outputs[2].shape == (1, 3 * (6 + 3), 1, 1)


def test_yolo_v4_get_loss_returns_scalar() -> None:
    model = YOLO_V4(_small_yolo_v4_config())

    loss = model.get_loss(lucid.ones(1, 3, 32, 32), [lucid.zeros(0, 5)])

    assert loss.shape == (1,)


def test_yolo_v4_predict_returns_detection_lists() -> None:
    model = YOLO_V4(_small_yolo_v4_config())

    detections = model.predict(lucid.ones(1, 3, 32, 32), conf_thresh=1.1)

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert detections[0] == []


def test_yolo_v4_factory_builds_and_runs() -> None:
    model = yolo_v4(num_classes=3)

    outputs = model(lucid.ones(1, 3, 64, 64))

    assert len(outputs) == 3
    assert all(out.shape[1] == 3 * (6 + 3) for out in outputs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_classes": 0},
        {"anchors": [[(1, 1)]]},
        {"anchors": [[(1, 1), (2, 2), (3, 3)]] * 2 + [[(0, 1), (2, 2), (3, 3)]]},
        {"strides": [8, 16]},
        {"strides": [8, 0, 32]},
        {"backbone": object()},
        {"backbone_out_channels": (16, 32, 64)},
        {"backbone": _ToyBackbone(), "backbone_out_channels": None},
        {"backbone": _ToyBackbone(), "backbone_out_channels": (16, 0, 64)},
        {"in_channels": (16, 32)},
        {"pos_iou_thr": 1.1},
        {"ignore_iou_thr": -0.1},
        {"obj_balance": (1.0, 1.0)},
        {"cls_label_smoothing": 1.0},
        {"iou_aware_alpha": 1.1},
        {"iou_branch_weight": -1.0},
    ),
)
def test_yolo_v4_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "num_classes": 3,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        YOLO_V4Config(**params)
