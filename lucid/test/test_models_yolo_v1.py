import pytest

import lucid

from lucid.models import YOLO_V1, YOLO_V1Config, yolo_v1, yolo_v1_tiny


def _small_yolo_v1_config() -> YOLO_V1Config:
    return YOLO_V1Config(
        in_channels=3,
        split_size=2,
        num_boxes=2,
        num_classes=3,
        conv_config=[(1024, 1, 1, 0)],
    )


def test_yolo_v1_public_imports() -> None:
    assert YOLO_V1 is not None
    assert YOLO_V1Config is not None
    assert yolo_v1 is not None
    assert yolo_v1_tiny is not None


def test_yolo_v1_accepts_config_object() -> None:
    config = _small_yolo_v1_config()

    model = YOLO_V1(config)

    assert model.config is config


def test_yolo_v1_forward_shape_matches_flat_detection_head() -> None:
    model = YOLO_V1(_small_yolo_v1_config())

    output = model(lucid.ones(2, 3, 2, 2))

    assert output.shape == (2, 2 * 2 * (2 * 5 + 3))


def test_yolo_v1_get_loss_returns_scalar() -> None:
    model = YOLO_V1(_small_yolo_v1_config())
    images = lucid.ones(2, 3, 2, 2)
    target = lucid.zeros(2, 2, 2, 2 * 5 + 3)

    loss = model.get_loss(images, target)

    assert loss.shape == ()
    assert loss.item() == loss.item()


def test_yolo_v1_factories_build_and_run() -> None:
    model = yolo_v1(num_classes=3, split_size=2)
    tiny_model = yolo_v1_tiny(num_classes=3, split_size=2)

    output = model(lucid.ones(1, 3, 128, 128))
    tiny_output = tiny_model(lucid.ones(1, 3, 128, 128))

    assert model.config.split_size == 2
    assert tiny_model.config.split_size == 2
    assert output.shape == (1, 2 * 2 * (2 * 5 + 3))
    assert tiny_output.shape == (1, 2 * 2 * (2 * 5 + 3))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"in_channels": 0},
        {"split_size": 0},
        {"num_boxes": 0},
        {"num_classes": 0},
        {"lambda_coord": -1.0},
        {"lambda_noobj": -0.1},
        {"conv_config": []},
        {"conv_config": ["X"]},
        {"conv_config": [(512, 1, 1, 0)]},
        {"conv_config": [(1024, 1, 1)]},
        {"conv_config": [[(512, 1, 1, 0), (1024, 3, 1, 1), 0]]},
    ),
)
def test_yolo_v1_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "in_channels": 3,
        "split_size": 2,
        "num_boxes": 2,
        "num_classes": 3,
        "conv_config": [(1024, 1, 1, 0)],
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        YOLO_V1Config(**params)
