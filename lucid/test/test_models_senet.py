import pytest

import lucid

from lucid.models import (
    SENet,
    SENetConfig,
    se_resnet_18,
    se_resnet_50,
    se_resnext_50_32x4d,
)


def test_senet_public_imports() -> None:
    assert SENet is not None
    assert SENetConfig is not None
    assert se_resnet_18 is not None
    assert se_resnet_50 is not None
    assert se_resnext_50_32x4d is not None


def test_senet_accepts_config_object() -> None:
    config = SENetConfig(block="se_basic", layers=[2, 2, 2, 2])

    model = SENet(config)

    assert model.config is config


@pytest.mark.parametrize("factory", (se_resnet_18, se_resnet_50, se_resnext_50_32x4d))
def test_senet_factories_forward_default_shape(factory) -> None:
    model = factory()

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_senet_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = se_resnet_18(
        num_classes=10,
        in_channels=1,
        stem_type="deep",
        stem_width=32,
        avg_down=True,
    )
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == SENetConfig(
        block="se_basic",
        layers=[2, 2, 2, 2],
        reduction=16,
        cardinality=1,
        base_width=64,
        num_classes=10,
        in_channels=1,
        stem_type="deep",
        stem_width=32,
        avg_down=True,
    )
    assert output.shape == (1, 10)


def test_senet_factory_rejects_overriding_preset_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset block, layers, reduction, cardinality, or base_width",
    ):
        se_resnext_50_32x4d(cardinality=16)


@pytest.mark.parametrize(
    ("config_kwargs", "error_type", "message"),
    (
        (
            {"block": "unknown", "layers": [2, 2, 2, 2]},
            ValueError,
            "block must be 'se_basic' or 'bottleneck'",
        ),
        (
            {"block": "se_basic", "layers": [2, 2, 2]},
            ValueError,
            "layers must contain exactly 4 stage depths",
        ),
        (
            {"block": "se_basic", "layers": [2, 0, 2, 2]},
            ValueError,
            "layers values must be positive integers",
        ),
        (
            {"block": "se_basic", "layers": [2, 2, 2, 2], "reduction": 0},
            ValueError,
            "reduction must be greater than 0",
        ),
        (
            {"block": "bottleneck", "layers": [3, 4, 6, 3], "cardinality": 0},
            ValueError,
            "cardinality must be greater than 0",
        ),
        (
            {"block": "bottleneck", "layers": [3, 4, 6, 3], "base_width": 0},
            ValueError,
            "base_width must be greater than 0",
        ),
        (
            {"block": "se_basic", "layers": [2, 2, 2, 2], "num_classes": 0},
            ValueError,
            "num_classes must be greater than 0",
        ),
        (
            {"block": "se_basic", "layers": [2, 2, 2, 2], "in_channels": 0},
            ValueError,
            "in_channels must be greater than 0",
        ),
        (
            {"block": "se_basic", "layers": [2, 2, 2, 2], "stem_width": 0},
            ValueError,
            "stem_width must be greater than 0",
        ),
        (
            {"block": "se_basic", "layers": [2, 2, 2, 2], "stem_type": "wide"},
            ValueError,
            "stem_type must be None or 'deep'",
        ),
        (
            {"block": "se_basic", "layers": [2, 2, 2, 2], "channels": [64, 128, 256]},
            ValueError,
            "channels must contain exactly 4 stage widths",
        ),
        (
            {
                "block": "se_basic",
                "layers": [2, 2, 2, 2],
                "channels": [64, 0, 256, 512],
            },
            ValueError,
            "channels values must be positive integers",
        ),
        (
            {"block": "se_basic", "layers": [2, 2, 2, 2], "block_args": 1},
            TypeError,
            "block_args must be a dictionary",
        ),
    ),
)
def test_senet_config_rejects_invalid_values(
    config_kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        SENetConfig(**config_kwargs)
