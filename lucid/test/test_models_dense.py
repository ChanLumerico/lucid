import pytest

import lucid
from lucid.models import (
    DenseNet,
    DenseNetConfig,
    densenet_121,
    densenet_169,
    densenet_201,
    densenet_264,
)


def test_densenet_public_imports() -> None:
    assert DenseNet is not None
    assert DenseNetConfig is not None
    assert densenet_121 is not None
    assert densenet_169 is not None
    assert densenet_201 is not None
    assert densenet_264 is not None


def test_densenet_accepts_config_object() -> None:
    config = DenseNetConfig(block_config=[6, 12, 24, 16])

    model = DenseNet(config)

    assert model.config is config


@pytest.mark.parametrize(
    "factory",
    (densenet_121, densenet_169, densenet_201, densenet_264),
)
def test_densenet_factories_forward_default_shape(factory) -> None:
    model = factory()

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_densenet_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = densenet_121(
        num_classes=10,
        in_channels=1,
        bottleneck=2,
        compression=0.75,
    )
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == DenseNetConfig(
        block_config=[6, 12, 24, 16],
        growth_rate=32,
        num_init_features=64,
        num_classes=10,
        in_channels=1,
        bottleneck=2,
        compression=0.75,
    )
    assert output.shape == (1, 10)


def test_densenet_factory_rejects_overriding_preset_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset block_config, growth_rate, or num_init_features",
    ):
        densenet_121(growth_rate=16)


@pytest.mark.parametrize(
    ("config_kwargs", "error_type", "message"),
    (
        (
            {"block_config": [6, 12, 24]},
            ValueError,
            "block_config must contain exactly 4 dense block depths",
        ),
        (
            {"block_config": [6, 12, 0, 16]},
            ValueError,
            "block_config values must be positive integers",
        ),
        (
            {"block_config": [6, 12, 24, 16], "growth_rate": 0},
            ValueError,
            "growth_rate must be greater than 0",
        ),
        (
            {"block_config": [6, 12, 24, 16], "num_init_features": 0},
            ValueError,
            "num_init_features must be greater than 0",
        ),
        (
            {"block_config": [6, 12, 24, 16], "num_classes": 0},
            ValueError,
            "num_classes must be greater than 0",
        ),
        (
            {"block_config": [6, 12, 24, 16], "in_channels": 0},
            ValueError,
            "in_channels must be greater than 0",
        ),
        (
            {"block_config": [6, 12, 24, 16], "bottleneck": 0},
            ValueError,
            "bottleneck must be greater than 0",
        ),
        (
            {"block_config": [6, 12, 24, 16], "compression": 0},
            ValueError,
            "compression must be in the range \\(0, 1\\]",
        ),
        (
            {"block_config": [6, 12, 24, 16], "compression": 1.5},
            ValueError,
            "compression must be in the range \\(0, 1\\]",
        ),
    ),
)
def test_densenet_config_rejects_invalid_values(
    config_kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        DenseNetConfig(**config_kwargs)
