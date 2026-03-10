import pytest

import lucid

from lucid.models import (
    ResNeXt,
    ResNeXtConfig,
    resnext_50_32x4d,
    resnext_101_32x4d,
    resnext_101_32x8d,
)


def test_resnext_public_imports() -> None:
    assert ResNeXt is not None
    assert ResNeXtConfig is not None
    assert resnext_50_32x4d is not None
    assert resnext_101_32x4d is not None
    assert resnext_101_32x8d is not None


def test_resnext_accepts_config_object() -> None:
    config = ResNeXtConfig(layers=[3, 4, 6, 3], cardinality=32, base_width=4)

    model = ResNeXt(config)

    assert model.config is config


@pytest.mark.parametrize(
    "factory",
    (resnext_50_32x4d, resnext_101_32x4d, resnext_101_32x8d),
)
def test_resnext_factories_forward_default_shape(factory) -> None:
    model = factory()

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_resnext_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = resnext_50_32x4d(
        num_classes=10,
        in_channels=1,
        stem_type="deep",
        stem_width=32,
        avg_down=True,
    )
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == ResNeXtConfig(
        layers=[3, 4, 6, 3],
        cardinality=32,
        base_width=4,
        num_classes=10,
        in_channels=1,
        stem_type="deep",
        stem_width=32,
        avg_down=True,
    )
    assert output.shape == (1, 10)


def test_resnext_factory_rejects_overriding_preset_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset layers, cardinality, or base_width",
    ):
        resnext_50_32x4d(cardinality=16)


@pytest.mark.parametrize(
    ("config_kwargs", "error_type", "message"),
    (
        (
            {"layers": [3, 4, 6], "cardinality": 32, "base_width": 4},
            ValueError,
            "layers must contain exactly 4 stage depths",
        ),
        (
            {"layers": [3, 0, 6, 3], "cardinality": 32, "base_width": 4},
            ValueError,
            "layers values must be positive integers",
        ),
        (
            {"layers": [3, 4, 6, 3], "cardinality": 0, "base_width": 4},
            ValueError,
            "cardinality must be greater than 0",
        ),
        (
            {"layers": [3, 4, 6, 3], "cardinality": 32, "base_width": 0},
            ValueError,
            "base_width must be greater than 0",
        ),
        (
            {
                "layers": [3, 4, 6, 3],
                "cardinality": 32,
                "base_width": 4,
                "num_classes": 0,
            },
            ValueError,
            "num_classes must be greater than 0",
        ),
        (
            {
                "layers": [3, 4, 6, 3],
                "cardinality": 32,
                "base_width": 4,
                "in_channels": 0,
            },
            ValueError,
            "in_channels must be greater than 0",
        ),
        (
            {
                "layers": [3, 4, 6, 3],
                "cardinality": 32,
                "base_width": 4,
                "stem_width": 0,
            },
            ValueError,
            "stem_width must be greater than 0",
        ),
        (
            {
                "layers": [3, 4, 6, 3],
                "cardinality": 32,
                "base_width": 4,
                "stem_type": "wide",
            },
            ValueError,
            "stem_type must be None or 'deep'",
        ),
        (
            {
                "layers": [3, 4, 6, 3],
                "cardinality": 32,
                "base_width": 4,
                "channels": [64, 128, 256],
            },
            ValueError,
            "channels must contain exactly 4 stage widths",
        ),
        (
            {
                "layers": [3, 4, 6, 3],
                "cardinality": 32,
                "base_width": 4,
                "channels": [64, 0, 256, 512],
            },
            ValueError,
            "channels values must be positive integers",
        ),
        (
            {
                "layers": [3, 4, 6, 3],
                "cardinality": 32,
                "base_width": 4,
                "block_args": 1,
            },
            TypeError,
            "block_args must be a dictionary",
        ),
    ),
)
def test_resnext_config_rejects_invalid_values(
    config_kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ResNeXtConfig(**config_kwargs)
