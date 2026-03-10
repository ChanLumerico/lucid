import pytest

import lucid

from lucid.models import (
    CoAtNet,
    CoAtNetConfig,
    coatnet_0,
    coatnet_1,
    coatnet_2,
    coatnet_3,
    coatnet_4,
    coatnet_5,
    coatnet_6,
    coatnet_7,
)
from lucid.models.vision import coatnet as coatnet_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_coatnet_config() -> CoAtNetConfig:
    return CoAtNetConfig(
        img_size=(32, 32),
        in_channels=3,
        num_blocks=(1, 1, 1, 1, 1),
        channels=(8, 16, 32, 64, 128),
        num_classes=10,
        num_heads=4,
    )


def _small_scaled_coatnet_config() -> CoAtNetConfig:
    return CoAtNetConfig(
        img_size=(32, 32),
        in_channels=3,
        num_blocks=(1, 1, 1, 1, 1),
        channels=(8, 16, 32, 64, 128),
        num_classes=10,
        num_heads=4,
        scaled_num_blocks=(1, 1),
        scaled_channels=(48, 64),
    )


def test_coatnet_public_imports() -> None:
    assert CoAtNet is not None
    assert CoAtNetConfig is not None
    assert coatnet_0 is not None
    assert coatnet_1 is not None
    assert coatnet_2 is not None
    assert coatnet_3 is not None
    assert coatnet_4 is not None
    assert coatnet_5 is not None
    assert coatnet_6 is not None
    assert coatnet_7 is not None


def test_coatnet_accepts_config_object() -> None:
    config = _small_coatnet_config()

    model = CoAtNet(config)

    assert model.config is config


def test_coatnet_custom_config_forward_shape() -> None:
    model = CoAtNet(_small_coatnet_config())

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 10)


def test_coatnet_scaled_config_forward_shape() -> None:
    model = CoAtNet(_small_scaled_coatnet_config())

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 10)


def test_coatnet_0_factory_forward_shape() -> None:
    model = coatnet_0()

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_coatnet_0_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coatnet_module, "CoAtNet", _ConfigCapture)

    model = coatnet_0(num_classes=10, num_heads=4, block_types=("C", "T", "T", "T"))

    assert model.config == CoAtNetConfig(
        img_size=(224, 224),
        in_channels=3,
        num_blocks=(2, 2, 3, 5, 2),
        channels=(64, 96, 192, 384, 768),
        num_classes=10,
        num_heads=4,
        block_types=("C", "T", "T", "T"),
    )


@pytest.mark.parametrize(
    ("factory", "expected_num_blocks", "expected_channels", "expected_num_heads", "expected_scaled_num_blocks", "expected_scaled_channels"),
    (
        (coatnet_0, (2, 2, 3, 5, 2), (64, 96, 192, 384, 768), 32, None, None),
        (coatnet_1, (2, 2, 6, 14, 2), (64, 96, 192, 384, 768), 32, None, None),
        (coatnet_2, (2, 2, 6, 14, 2), (128, 128, 256, 512, 1024), 32, None, None),
        (coatnet_3, (2, 2, 6, 14, 2), (192, 192, 384, 768, 1536), 32, None, None),
        (coatnet_4, (2, 2, 12, 28, 2), (192, 192, 384, 768, 1536), 32, None, None),
        (coatnet_5, (2, 2, 12, 28, 2), (192, 256, 512, 1280, 2048), 64, None, None),
        (coatnet_6, (2, 2, 4, 8, 2), (192, 192, 384, 768, 2048), 128, (8, 42), (768, 1536)),
        (coatnet_7, (2, 2, 4, 8, 2), (192, 256, 512, 1024, 3072), 128, (8, 42), (1024, 2048)),
    ),
)
def test_coatnet_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_num_blocks: tuple[int, ...],
    expected_channels: tuple[int, ...],
    expected_num_heads: int,
    expected_scaled_num_blocks: tuple[int, int] | None,
    expected_scaled_channels: tuple[int, int] | None,
) -> None:
    monkeypatch.setattr(coatnet_module, "CoAtNet", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.img_size == (224, 224)
    assert model.config.in_channels == 3
    assert model.config.num_classes == 10
    assert model.config.num_blocks == expected_num_blocks
    assert model.config.channels == expected_channels
    assert model.config.num_heads == expected_num_heads
    assert model.config.scaled_num_blocks == expected_scaled_num_blocks
    assert model.config.scaled_channels == expected_scaled_channels


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    (
        (
            coatnet_0,
            {"img_size": (32, 32)},
            "factory variants do not allow overriding preset img_size, in_channels, num_blocks, or channels",
        ),
        (
            coatnet_5,
            {"num_heads": 8},
            "factory variants do not allow overriding preset img_size, in_channels, num_blocks, channels, or num_heads",
        ),
        (
            coatnet_6,
            {"scaled_channels": (32, 64)},
            "factory variants do not allow overriding preset img_size, in_channels, num_blocks, channels, num_heads, scaled_num_blocks, or scaled_channels",
        ),
    ),
)
def test_coatnet_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"img_size": (31, 32)},
        {"img_size": (32,)},
        {"in_channels": 0},
        {"num_blocks": (1, 1, 1, 1)},
        {"channels": (8, 16, 32, 64)},
        {"num_classes": 0},
        {"num_heads": 0},
        {"block_types": ("C", "T", "X", "T")},
        {"block_types": ("C", "T", "T")},
        {"scaled_num_blocks": (1, 1)},
        {"scaled_num_blocks": (1,), "scaled_channels": (32, 64)},
        {"scaled_num_blocks": (1, 1), "scaled_channels": (32,)},
    ),
)
def test_coatnet_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "img_size": (32, 32),
        "in_channels": 3,
        "num_blocks": (1, 1, 1, 1, 1),
        "channels": (8, 16, 32, 64, 128),
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        CoAtNetConfig(**params)
