import pytest

import lucid

from lucid.models import (
    MaxViT,
    MaxViTConfig,
    maxvit_base,
    maxvit_large,
    maxvit_small,
    maxvit_tiny,
    maxvit_xlarge,
)
from lucid.models.vision import maxvit as maxvit_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_maxvit_config() -> MaxViTConfig:
    return MaxViTConfig(
        in_channels=1,
        depths=(1, 1),
        channels=(16, 32),
        num_classes=10,
        embed_dim=16,
        num_heads=4,
        grid_window_size=(1, 1),
        drop_path=0.0,
        mlp_ratio=2.0,
    )


def test_maxvit_public_imports() -> None:
    assert MaxViT is not None
    assert MaxViTConfig is not None
    assert maxvit_tiny is not None
    assert maxvit_small is not None
    assert maxvit_base is not None
    assert maxvit_large is not None
    assert maxvit_xlarge is not None


def test_maxvit_accepts_config_object() -> None:
    config = _small_maxvit_config()

    model = MaxViT(config)

    assert model.config is config


def test_maxvit_custom_config_forward_shape() -> None:
    model = MaxViT(_small_maxvit_config())

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)


def test_maxvit_tiny_factory_forward_shape() -> None:
    model = maxvit_tiny(num_classes=10)

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 10)


def test_maxvit_tiny_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(maxvit_module, "MaxViT", _ConfigCapture)

    model = maxvit_tiny(
        in_channels=1,
        num_classes=10,
        num_heads=8,
        grid_window_size=(1, 1),
        attn_drop=0.1,
        drop=0.2,
        drop_path=0.3,
        mlp_ratio=2.0,
    )

    assert model.config.in_channels == 1
    assert model.config.num_classes == 10
    assert model.config.depths == (2, 2, 5, 2)
    assert model.config.channels == (64, 128, 256, 512)
    assert model.config.embed_dim == 64
    assert model.config.num_heads == 8
    assert model.config.grid_window_size == (1, 1)
    assert model.config.attn_drop == 0.1
    assert model.config.drop == 0.2
    assert model.config.drop_path == 0.3
    assert model.config.mlp_ratio == 2.0


@pytest.mark.parametrize(
    ("factory", "expected_depths", "expected_channels", "expected_embed_dim"),
    (
        (maxvit_tiny, (2, 2, 5, 2), (64, 128, 256, 512), 64),
        (maxvit_small, (2, 2, 5, 2), (96, 192, 384, 768), 64),
        (maxvit_base, (2, 6, 14, 2), (96, 192, 384, 768), 64),
        (maxvit_large, (2, 6, 14, 2), (128, 256, 512, 1024), 128),
        (maxvit_xlarge, (2, 6, 14, 2), (192, 384, 768, 1536), 192),
    ),
)
def test_maxvit_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_depths: tuple[int, ...],
    expected_channels: tuple[int, ...],
    expected_embed_dim: int,
) -> None:
    monkeypatch.setattr(maxvit_module, "MaxViT", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.in_channels == 3
    assert model.config.num_classes == 10
    assert model.config.depths == expected_depths
    assert model.config.channels == expected_channels
    assert model.config.embed_dim == expected_embed_dim


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    (
        (maxvit_tiny, {"depths": (1, 1)}),
        (maxvit_small, {"channels": (32, 64, 128, 256)}),
        (maxvit_base, {"embed_dim": 32}),
    ),
)
def test_maxvit_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset depths, channels, or embed_dim",
    ):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"in_channels": 0},
        {"depths": ()},
        {"depths": (1, 0)},
        {"channels": (16,)},
        {"channels": (16, 0)},
        {"channels": (18, 32), "num_heads": 4},
        {"num_classes": -1},
        {"embed_dim": 0},
        {"num_heads": 0},
        {"grid_window_size": (1,)},
        {"grid_window_size": (1, 0)},
        {"attn_drop": 1.0},
        {"drop": 1.0},
        {"drop_path": 1.0},
        {"mlp_ratio": 0.0},
    ),
)
def test_maxvit_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "in_channels": 1,
        "depths": (1, 1),
        "channels": (16, 32),
        "num_classes": 10,
        "embed_dim": 16,
        "num_heads": 4,
        "grid_window_size": (1, 1),
        "drop_path": 0.0,
        "mlp_ratio": 2.0,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        MaxViTConfig(**params)
