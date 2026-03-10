import pytest

import lucid

from lucid.models import (
    ResNeSt,
    ResNeStConfig,
    resnest_14,
    resnest_50,
    resnest_50_4s2x40d,
)


def test_resnest_public_imports() -> None:
    assert ResNeSt is not None
    assert ResNeStConfig is not None
    assert resnest_14 is not None
    assert resnest_50 is not None
    assert resnest_50_4s2x40d is not None


def test_resnest_accepts_config_object() -> None:
    config = ResNeStConfig(layers=[1, 1, 1, 1])

    model = ResNeSt(config)

    assert model.config is config


@pytest.mark.parametrize("factory", (resnest_14, resnest_50, resnest_50_4s2x40d))
def test_resnest_factories_forward_default_shape(factory) -> None:
    model = factory()

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_resnest_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = resnest_14(num_classes=10, in_channels=1, avg_down=True)
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == ResNeStConfig(
        layers=[1, 1, 1, 1],
        base_width=64,
        stem_width=32,
        cardinality=1,
        radix=2,
        avd=True,
        num_classes=10,
        in_channels=1,
        avg_down=True,
    )
    assert output.shape == (1, 10)


def test_resnest_factory_rejects_overriding_preset_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset layers, base_width, stem_width, cardinality, radix, or avd",
    ):
        resnest_50(radix=1)


@pytest.mark.parametrize(
    ("config_kwargs", "error_type", "message"),
    (
        ({"layers": [1, 1, 1]}, ValueError, "layers must contain exactly 4 stage depths"),
        (
            {"layers": [1, 0, 1, 1]},
            ValueError,
            "layers values must be positive integers",
        ),
        (
            {"layers": [1, 1, 1, 1], "base_width": 0},
            ValueError,
            "base_width must be greater than 0",
        ),
        (
            {"layers": [1, 1, 1, 1], "stem_width": 0},
            ValueError,
            "stem_width must be greater than 0",
        ),
        (
            {"layers": [1, 1, 1, 1], "cardinality": 0},
            ValueError,
            "cardinality must be greater than 0",
        ),
        (
            {"layers": [1, 1, 1, 1], "radix": -1},
            ValueError,
            "radix must be greater than or equal to 0",
        ),
        (
            {"layers": [1, 1, 1, 1], "num_classes": 0},
            ValueError,
            "num_classes must be greater than 0",
        ),
        (
            {"layers": [1, 1, 1, 1], "in_channels": 0},
            ValueError,
            "in_channels must be greater than 0",
        ),
        (
            {"layers": [1, 1, 1, 1], "channels": [64, 128, 256]},
            ValueError,
            "channels must contain exactly 4 stage widths",
        ),
        (
            {"layers": [1, 1, 1, 1], "channels": [64, 0, 256, 512]},
            ValueError,
            "channels values must be positive integers",
        ),
        (
            {"layers": [1, 1, 1, 1], "block_args": 1},
            TypeError,
            "block_args must be a dictionary",
        ),
    ),
)
def test_resnest_config_rejects_invalid_values(
    config_kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ResNeStConfig(**config_kwargs)
