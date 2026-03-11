import pytest

import lucid
import lucid.nn as nn

from lucid.models import YOLO_V2, YOLO_V2Config, yolo_v2, yolo_v2_tiny


def _toy_darknet() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 1024, kernel_size=1),
        nn.ReLU(),
    )


def _small_yolo_v2_config() -> YOLO_V2Config:
    return YOLO_V2Config(
        num_classes=3,
        num_anchors=2,
        anchors=[(1.0, 1.0), (2.0, 2.0)],
        darknet=_toy_darknet(),
        route_layer=None,
        image_size=4,
        use_passthrough=False,
    )


def test_yolo_v2_public_imports() -> None:
    assert YOLO_V2 is not None
    assert YOLO_V2Config is not None
    assert yolo_v2 is not None
    assert yolo_v2_tiny is not None


def test_yolo_v2_accepts_config_object() -> None:
    config = _small_yolo_v2_config()

    model = YOLO_V2(config)

    assert model.config is config


def test_yolo_v2_forward_shape_with_custom_darknet() -> None:
    model = YOLO_V2(_small_yolo_v2_config())

    output = model(lucid.ones(2, 3, 4, 4))

    assert output.shape == (2, 2 * (5 + 3), 4, 4)


def test_yolo_v2_get_loss_returns_scalar() -> None:
    model = YOLO_V2(_small_yolo_v2_config())
    images = lucid.ones(1, 3, 4, 4)
    target = lucid.zeros(1, 4, 4, 2, 5 + 3)

    loss = model.get_loss(images, target)

    assert loss.shape == ()


def test_yolo_v2_predict_returns_detection_lists() -> None:
    model = YOLO_V2(_small_yolo_v2_config())

    detections = model.predict(lucid.ones(1, 3, 4, 4), conf_thresh=1.1)

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert detections[0] == []


def test_yolo_v2_factories_build_and_run() -> None:
    model = yolo_v2(num_classes=3)
    tiny_model = yolo_v2_tiny(num_classes=3)

    output = model(lucid.ones(1, 3, 64, 64))
    tiny_output = tiny_model(lucid.ones(1, 3, 64, 64))

    assert output.shape[1] == 5 * (5 + 3)
    assert tiny_output.shape[1] == 5 * (5 + 3)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_classes": 0},
        {"num_anchors": 0},
        {"anchors": [(1.0, 1.0)]},
        {"anchors": [(1.0, 1.0), (0.0, 2.0)]},
        {"lambda_coord": -1.0},
        {"lambda_noobj": -1.0},
        {"darknet": object()},
        {"route_layer": -1},
        {"image_size": 0},
        {"use_passthrough": "yes"},
    ),
)
def test_yolo_v2_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "num_classes": 3,
        "num_anchors": 2,
        "anchors": [(1.0, 1.0), (2.0, 2.0)],
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        YOLO_V2Config(**params)
