import pytest

import lucid

from lucid.models import ResNet, ResNetConfig, resnet_18, resnet_50, resnet_200, wide_resnet_50


def test_resnet_public_imports() -> None:
    assert ResNet is not None
    assert ResNetConfig is not None
    assert resnet_18 is not None
    assert resnet_50 is not None
    assert resnet_200 is not None
    assert wide_resnet_50 is not None


def test_resnet_accepts_config_object() -> None:
    config = ResNetConfig(block="basic", layers=[2, 2, 2, 2])

    model = ResNet(config)

    assert model.config is config


@pytest.mark.parametrize("factory", (resnet_18, resnet_50, resnet_200, wide_resnet_50))
def test_resnet_factories_forward_default_shape(factory) -> None:
    model = factory()

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_resnet_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = resnet_50(
        num_classes=10,
        in_channels=1,
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == ResNetConfig(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        num_classes=10,
        in_channels=1,
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    assert output.shape == (1, 10)


def test_wide_resnet_factory_merges_block_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = wide_resnet_50(block_args={"dilation": 1})

    assert model.config.block_args == {"base_width": 128, "dilation": 1}


def test_resnet_factory_rejects_overriding_preset_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    with pytest.raises(
        TypeError, match="factory variants do not allow overriding preset block or layers"
    ):
        resnet_18(layers=[1, 1, 1, 1])


@pytest.mark.parametrize(
    ("config_kwargs", "error_type", "message"),
    (
        (
            {"block": "unknown", "layers": [2, 2, 2, 2]},
            ValueError,
            "block must be one of",
        ),
        (
            {"block": object, "layers": [2, 2, 2, 2]},
            TypeError,
            "block must be a ResNet block name",
        ),
        (
            {"block": "basic", "layers": [2, 2, 2]},
            ValueError,
            "layers must contain exactly 4 stage depths",
        ),
        (
            {"block": "basic", "layers": [2, 0, 2, 2]},
            ValueError,
            "layers values must be positive integers",
        ),
        (
            {"block": "basic", "layers": [2, 2, 2, 2], "num_classes": 0},
            ValueError,
            "num_classes must be greater than 0",
        ),
        (
            {"block": "basic", "layers": [2, 2, 2, 2], "in_channels": 0},
            ValueError,
            "in_channels must be greater than 0",
        ),
        (
            {"block": "basic", "layers": [2, 2, 2, 2], "stem_width": 0},
            ValueError,
            "stem_width must be greater than 0",
        ),
        (
            {"block": "basic", "layers": [2, 2, 2, 2], "stem_type": "wide"},
            ValueError,
            "stem_type must be None or 'deep'",
        ),
        (
            {"block": "basic", "layers": [2, 2, 2, 2], "channels": [64, 128, 256]},
            ValueError,
            "channels must contain exactly 4 stage widths",
        ),
        (
            {"block": "basic", "layers": [2, 2, 2, 2], "channels": [64, 0, 256, 512]},
            ValueError,
            "channels values must be positive integers",
        ),
        (
            {"block": "basic", "layers": [2, 2, 2, 2], "block_args": 1},
            TypeError,
            "block_args must be a dictionary",
        ),
    ),
)
def test_resnet_config_rejects_invalid_values(
    config_kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ResNetConfig(**config_kwargs)
