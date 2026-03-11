import pytest

import lucid

from lucid.models import (
    EfficientDet,
    EfficientDetConfig,
    efficientdet_d0,
    efficientdet_d1,
    efficientdet_d7,
)


def test_efficientdet_public_imports() -> None:
    assert EfficientDet is not None
    assert EfficientDetConfig is not None
    assert efficientdet_d0 is not None
    assert efficientdet_d1 is not None
    assert efficientdet_d7 is not None


def test_efficientdet_accepts_config_object() -> None:
    config = EfficientDetConfig(compound_coef=0, num_classes=3)

    model = EfficientDet(config)

    assert model.config is config


def test_efficientdet_forward_outputs_cls_box_and_anchors() -> None:
    model = EfficientDet(EfficientDetConfig(compound_coef=0, num_classes=3))

    cls_preds, box_preds, anchors = model(lucid.ones(1, 3, 96, 96))

    assert cls_preds.ndim == 3
    assert box_preds.ndim == 3
    assert anchors.ndim == 3
    assert cls_preds.shape[0] == 1
    assert box_preds.shape[0] == 1
    assert anchors.shape[0] == 1
    assert cls_preds.shape[2] == 3
    assert box_preds.shape[2] == 4
    assert anchors.shape[2] == 4
    assert cls_preds.shape[1] == box_preds.shape[1] == anchors.shape[1]


def test_efficientdet_get_loss_returns_scalar_tensor() -> None:
    model = EfficientDet(EfficientDetConfig(compound_coef=0, num_classes=3))
    images = lucid.ones(1, 3, 96, 96)
    targets = [lucid.Tensor([[4, 4, 20, 20, 0]], dtype=lucid.Float32)]

    loss = model.get_loss(images, targets)

    assert loss.size == 1


def test_efficientdet_predict_returns_detection_lists() -> None:
    model = EfficientDet(EfficientDetConfig(compound_coef=0, num_classes=3))

    detections = model.predict(lucid.ones(1, 3, 96, 96))

    assert isinstance(detections, list)
    assert len(detections) == 1
    assert isinstance(detections[0], list)


def test_efficientdet_factories_build_and_run() -> None:
    model_d0 = efficientdet_d0(num_classes=3)
    model_d1 = efficientdet_d1(num_classes=3)
    model_d7 = efficientdet_d7(num_classes=3)

    out_d0 = model_d0(lucid.ones(1, 3, 96, 96))
    out_d1 = model_d1(lucid.ones(1, 3, 96, 96))

    assert model_d0.config.compound_coef == 0
    assert model_d1.config.compound_coef == 1
    assert model_d7.config.compound_coef == 7
    assert all(t.shape[0] == 1 for t in out_d0)
    assert all(t.shape[0] == 1 for t in out_d1)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"compound_coef": -1},
        {"compound_coef": 8},
        {"num_anchors": 0},
        {"num_anchors": 3},
        {"num_classes": 0},
    ),
)
def test_efficientdet_config_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    params = {}
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        EfficientDetConfig(**params)
