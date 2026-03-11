import pytest

import lucid
import lucid.nn as nn

from lucid.models import YOLO_V3, YOLO_V3Config, yolo_v3, yolo_v3_tiny


class _ToyDarknet(nn.Module):
    def forward(self, x):
        batch = x.shape[0]
        return (
            lucid.ones(batch, 16, 4, 4),
            lucid.ones(batch, 32, 2, 2),
            lucid.ones(batch, 64, 1, 1),
        )


def _small_yolo_v3_config() -> YOLO_V3Config:
    return YOLO_V3Config(
        num_classes=3,
        anchors=[
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
            (9, 9),
        ],
        image_size=32,
        darknet=_ToyDarknet(),
        darknet_out_channels_arr=[16, 32, 64],
    )


def test_yolo_v3_public_imports() -> None:
    assert YOLO_V3 is not None
    assert YOLO_V3Config is not None
    assert yolo_v3 is not None
    assert yolo_v3_tiny is not None


def test_yolo_v3_accepts_config_object() -> None:
    config = _small_yolo_v3_config()

    model = YOLO_V3(config)

    assert model.config is config


def test_yolo_v3_forward_outputs_three_scales() -> None:
    model = YOLO_V3(_small_yolo_v3_config())

    outputs = model(lucid.ones(1, 3, 32, 32))

    assert len(outputs) == 3
    assert outputs[0].shape == (1, 3 * (5 + 3), 1, 1)
    assert outputs[1].shape == (1, 3 * (5 + 3), 2, 2)
    assert outputs[2].shape == (1, 3 * (5 + 3), 4, 4)


def test_yolo_v3_get_loss_returns_scalar() -> None:
    model = YOLO_V3(_small_yolo_v3_config())
    images = lucid.ones(1, 3, 32, 32)
    targets = (
        lucid.zeros(1, 1, 1, 3, 5 + 3),
        lucid.zeros(1, 2, 2, 3, 5 + 3),
        lucid.zeros(1, 4, 4, 3, 5 + 3),
    )

    loss = model.get_loss(images, targets)

    assert loss.shape == ()


def test_yolo_v3_predict_returns_detection_lists() -> None:
    model = YOLO_V3(_small_yolo_v3_config())

    detections = model.predict(lucid.ones(1, 3, 32, 32), conf_thresh=1.1)

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert detections[0] == []


def test_yolo_v3_factories_build_and_run() -> None:
    model = yolo_v3(num_classes=3)
    tiny_model = yolo_v3_tiny(num_classes=3)

    output = model(lucid.ones(1, 3, 64, 64))
    tiny_output = tiny_model(lucid.ones(1, 3, 64, 64))

    assert len(output) == 3
    assert len(tiny_output) == 3
    assert all(out.shape[1] == 3 * (5 + 3) for out in output)
    assert all(out.shape[1] == 3 * (5 + 3) for out in tiny_output)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_classes": 0},
        {"anchors": [(1, 1)]},
        {"anchors": [(1, 1)] * 8 + [(0, 1)]},
        {"image_size": 0},
        {"darknet": object()},
        {"darknet_out_channels_arr": [16, 32, 64]},
        {"darknet": _ToyDarknet(), "darknet_out_channels_arr": None},
        {"darknet": _ToyDarknet(), "darknet_out_channels_arr": [16, 32]},
        {"darknet": _ToyDarknet(), "darknet_out_channels_arr": [16, 0, 64]},
    ),
)
def test_yolo_v3_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "num_classes": 3,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        YOLO_V3Config(**params)
