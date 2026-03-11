import pytest

import lucid
import lucid.nn as nn

from lucid.models import DETR, DETRConfig, detr_r50, detr_r101


class _ToyBackbone(nn.Module):
    def __init__(self, num_channels: int = 8) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.proj = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.pool(self.proj(x))


class _ToyTransformer(nn.Module):
    def __init__(self, d_model: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

    def forward(
        self, src: lucid.Tensor, mask: lucid.Tensor, query_embed: lucid.Tensor, pos_embed: lucid.Tensor
    ) -> tuple[lucid.Tensor, lucid.Tensor]:
        del mask
        memory = src + pos_embed
        global_context = memory.mean(axis=1, keepdims=True)
        hs = lucid.stack(
            [query_embed + global_context for _ in range(self.num_layers)], axis=0
        )
        return hs, memory


def _small_detr_config(**kwargs: object) -> DETRConfig:
    params = {
        "backbone": _ToyBackbone(),
        "transformer": _ToyTransformer(),
        "num_classes": 3,
        "num_queries": 6,
    }
    params.update(kwargs)
    return DETRConfig(**params)


def test_detr_public_imports() -> None:
    assert DETR is not None
    assert DETRConfig is not None
    assert detr_r50 is not None
    assert detr_r101 is not None


def test_detr_accepts_config_object_and_runs_forward_loss_predict() -> None:
    config = _small_detr_config()
    model = DETR(config)
    x = lucid.ones(1, 3, 32, 32)

    outputs = model(x)
    loss = model.get_loss(
        x,
        [
            {
                "class_id": lucid.Tensor([1], dtype=lucid.Int32),
                "box": lucid.Tensor([[0.5, 0.5, 0.25, 0.25]], dtype=lucid.Float32),
            }
        ],
    )
    detections = model.predict(x, k=5)

    assert model.config is config
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    final_logits, final_boxes = outputs[-1]
    assert final_logits.shape == (1, 6, 4)
    assert final_boxes.shape == (1, 6, 4)
    assert loss.size == 1
    assert isinstance(detections, list)
    assert len(detections) == 1
    assert isinstance(detections[0], list)
    assert len(detections[0]) == 5


def test_detr_factories_build_and_run() -> None:
    model_r50 = detr_r50(num_classes=3, num_queries=10, aux_loss=False)
    model_r101 = detr_r101(num_classes=3, num_queries=10, aux_loss=False)

    logits_r50, boxes_r50 = model_r50(lucid.ones(1, 3, 32, 32))
    logits_r101, boxes_r101 = model_r101(lucid.ones(1, 3, 32, 32))

    assert model_r50.config.num_queries == 10
    assert model_r101.config.num_queries == 10
    assert logits_r50.shape == (1, 10, 4)
    assert boxes_r50.shape == (1, 10, 4)
    assert logits_r101.shape == (1, 10, 4)
    assert boxes_r101.shape == (1, 10, 4)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"backbone": object()},
        {"backbone": nn.Identity()},
        {"transformer": object()},
        {"transformer": nn.Identity()},
        {"num_classes": 0},
        {"num_queries": 0},
        {"aux_loss": 1},
        {"matcher": object()},
        {"class_loss_coef": -1.0},
        {"bbox_loss_coef": -1.0},
        {"giou_loss_coef": -1.0},
        {"eos_coef": -0.1},
        {"eos_coef": 1.1},
    ),
)
def test_detr_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _small_detr_config(**kwargs)
