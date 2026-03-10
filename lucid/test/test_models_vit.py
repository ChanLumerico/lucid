import pytest

import lucid

from lucid.models import (
    ViT,
    ViTConfig,
    vit_base,
    vit_huge,
    vit_large,
    vit_small,
    vit_tiny,
)
from lucid.models.vision import vit as vit_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_vit_config() -> ViTConfig:
    return ViTConfig(
        image_size=32,
        patch_size=8,
        in_channels=1,
        num_classes=10,
        embedding_dim=64,
        depth=2,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.0,
    )


def test_vit_public_imports() -> None:
    assert ViT is not None
    assert ViTConfig is not None
    assert vit_tiny is not None
    assert vit_small is not None
    assert vit_base is not None
    assert vit_large is not None
    assert vit_huge is not None


def test_vit_accepts_config_object() -> None:
    config = _small_vit_config()

    model = ViT(config)

    assert model.config is config


def test_vit_custom_config_forward_shape() -> None:
    model = ViT(_small_vit_config())

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)


def test_vit_tiny_factory_forward_shape() -> None:
    model = vit_tiny(image_size=32, patch_size=8)

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 1000)


def test_vit_tiny_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vit_module, "ViT", _ConfigCapture)

    model = vit_tiny(
        image_size=32,
        patch_size=8,
        num_classes=10,
        in_channels=1,
        dropout_rate=0.0,
    )

    assert model.config == ViTConfig(
        image_size=32,
        patch_size=8,
        in_channels=1,
        num_classes=10,
        embedding_dim=192,
        depth=12,
        num_heads=3,
        mlp_dim=768,
        dropout_rate=0.0,
    )


@pytest.mark.parametrize(
    ("factory", "expected_embedding_dim", "expected_depth", "expected_num_heads", "expected_mlp_dim"),
    (
        (vit_tiny, 192, 12, 3, 768),
        (vit_small, 384, 12, 6, 1536),
        (vit_base, 768, 12, 12, 3072),
        (vit_large, 1024, 24, 16, 4096),
        (vit_huge, 1280, 32, 16, 5120),
    ),
)
def test_vit_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_embedding_dim: int,
    expected_depth: int,
    expected_num_heads: int,
    expected_mlp_dim: int,
) -> None:
    monkeypatch.setattr(vit_module, "ViT", _ConfigCapture)

    model = factory(image_size=32, patch_size=8, num_classes=10)

    assert model.config.image_size == 32
    assert model.config.patch_size == 8
    assert model.config.in_channels == 3
    assert model.config.num_classes == 10
    assert model.config.embedding_dim == expected_embedding_dim
    assert model.config.depth == expected_depth
    assert model.config.num_heads == expected_num_heads
    assert model.config.mlp_dim == expected_mlp_dim
    assert model.config.dropout_rate == 0.1


@pytest.mark.parametrize(
    "kwargs",
    (
        {"embedding_dim": 128},
        {"depth": 6},
        {"num_heads": 8},
        {"mlp_dim": 512},
    ),
)
def test_vit_factories_reject_overriding_preset_fields(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset embedding_dim, depth, num_heads, or mlp_dim",
    ):
        vit_tiny(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"image_size": 0},
        {"patch_size": 0},
        {"image_size": 30, "patch_size": 8},
        {"in_channels": 0},
        {"num_classes": 0},
        {"embedding_dim": 0},
        {"depth": 0},
        {"num_heads": 0},
        {"embedding_dim": 65, "num_heads": 4},
        {"mlp_dim": 0},
        {"dropout_rate": 1.0},
    ),
)
def test_vit_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "image_size": 32,
        "patch_size": 8,
        "in_channels": 3,
        "num_classes": 10,
        "embedding_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "mlp_dim": 128,
        "dropout_rate": 0.0,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        ViTConfig(**params)
