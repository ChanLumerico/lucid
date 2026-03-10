import pytest

import lucid

from lucid.models import (
    CrossViT,
    CrossViTConfig,
    crossvit_9,
    crossvit_9_dagger,
    crossvit_15,
    crossvit_15_dagger,
    crossvit_18,
    crossvit_18_dagger,
    crossvit_base,
    crossvit_small,
    crossvit_tiny,
)
from lucid.models.vision import crossvit as crossvit_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_crossvit_config() -> CrossViTConfig:
    return CrossViTConfig(
        img_size=(32, 32),
        patch_size=(8, 16),
        in_channels=1,
        num_classes=10,
        embed_dim=(32, 64),
        depth=((1, 1, 0), (1, 1, 0)),
        num_heads=(4, 4),
        mlp_ratio=(2.0, 2.0, 1.0),
        drop_path_rate=0.0,
    )


def test_crossvit_public_imports() -> None:
    assert CrossViT is not None
    assert CrossViTConfig is not None
    assert crossvit_tiny is not None
    assert crossvit_small is not None
    assert crossvit_base is not None
    assert crossvit_9 is not None
    assert crossvit_15 is not None
    assert crossvit_18 is not None
    assert crossvit_9_dagger is not None
    assert crossvit_15_dagger is not None
    assert crossvit_18_dagger is not None


def test_crossvit_accepts_config_object() -> None:
    config = _small_crossvit_config()

    model = CrossViT(config)

    assert model.config is config


def test_crossvit_custom_config_forward_shape() -> None:
    model = CrossViT(_small_crossvit_config())

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)


def test_crossvit_tiny_factory_forward_shape() -> None:
    model = crossvit_tiny(num_classes=10)

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 10)


def test_crossvit_tiny_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crossvit_module, "CrossViT", _ConfigCapture)

    model = crossvit_tiny(
        num_classes=10,
        in_channels=1,
        qk_scale=0.5,
        drop_rate=0.1,
        attn_drop_rate=0.2,
        drop_path_rate=0.3,
    )

    assert model.config.num_classes == 10
    assert model.config.in_channels == 1
    assert model.config.qk_scale == 0.5
    assert model.config.drop_rate == 0.1
    assert model.config.attn_drop_rate == 0.2
    assert model.config.drop_path_rate == 0.3
    assert model.config.img_size == (240, 224)
    assert model.config.patch_size == (12, 16)
    assert model.config.embed_dim == (96, 192)
    assert model.config.depth == ((1, 4, 0), (1, 4, 0), (1, 4, 0))
    assert model.config.num_heads == (3, 3)
    assert model.config.mlp_ratio == (4, 4, 1)
    assert model.config.multi_conv is False


@pytest.mark.parametrize(
    ("factory", "expected_embed_dim", "expected_depth", "expected_num_heads", "expected_mlp_ratio", "expected_multi_conv"),
    (
        (crossvit_tiny, (96, 192), ((1, 4, 0),) * 3, (3, 3), (4, 4, 1), False),
        (crossvit_small, (192, 384), ((1, 4, 0),) * 3, (6, 6), (4, 4, 1), False),
        (crossvit_base, (384, 768), ((1, 4, 0),) * 3, (12, 12), (4, 4, 1), False),
        (crossvit_9, (128, 256), ((1, 3, 0),) * 3, (4, 4), (3, 3, 1), False),
        (crossvit_15, (192, 384), ((1, 5, 0),) * 3, (6, 6), (3, 3, 1), False),
        (crossvit_18, (224, 448), ((1, 6, 0),) * 3, (7, 7), (3, 3, 1), False),
        (crossvit_9_dagger, (128, 256), ((1, 3, 0),) * 3, (4, 4), (3, 3, 1), True),
        (crossvit_15_dagger, (192, 384), ((1, 5, 0),) * 3, (6, 6), (3, 3, 1), True),
        (crossvit_18_dagger, (224, 448), ((1, 6, 0),) * 3, (7, 7), (3, 3, 1), True),
    ),
)
def test_crossvit_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_embed_dim: tuple[int, int],
    expected_depth: tuple[tuple[int, int, int], ...],
    expected_num_heads: tuple[int, int],
    expected_mlp_ratio: tuple[float, float, float],
    expected_multi_conv: bool,
) -> None:
    monkeypatch.setattr(crossvit_module, "CrossViT", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_classes == 10
    assert model.config.img_size == (240, 224)
    assert model.config.patch_size == (12, 16)
    assert model.config.embed_dim == expected_embed_dim
    assert model.config.depth == expected_depth
    assert model.config.num_heads == expected_num_heads
    assert model.config.mlp_ratio == expected_mlp_ratio
    assert model.config.multi_conv is expected_multi_conv


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    (
        (crossvit_tiny, {"embed_dim": (64, 128)}),
        (crossvit_small, {"patch_size": (8, 16)}),
        (crossvit_9, {"depth": ((1, 2, 0),) * 3}),
        (crossvit_9_dagger, {"multi_conv": False}),
    ),
)
def test_crossvit_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset img_size, patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, norm_layer, or multi_conv",
    ):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"img_size": (32,)},
        {"img_size": (32, 0)},
        {"patch_size": (8,)},
        {"patch_size": (8, 0)},
        {"in_channels": 0},
        {"num_classes": -1},
        {"embed_dim": (32,)},
        {"embed_dim": (32, 0)},
        {"depth": ()},
        {"depth": ((1, 1),)},
        {"depth": ((0, 1, 0),)},
        {"depth": ((1, 1, -1),)},
        {"num_heads": (4,)},
        {"num_heads": (4, 0)},
        {"mlp_ratio": (2.0, 2.0)},
        {"mlp_ratio": (2.0, 0.0, 1.0)},
        {"embed_dim": (30, 64), "num_heads": (4, 4)},
        {"drop_rate": 1.0},
        {"attn_drop_rate": 1.0},
        {"drop_path_rate": 1.0},
        {"patch_size": (8, 16), "multi_conv": True},
    ),
)
def test_crossvit_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "img_size": (32, 32),
        "patch_size": (8, 16),
        "in_channels": 1,
        "num_classes": 10,
        "embed_dim": (32, 64),
        "depth": ((1, 1, 0), (1, 1, 0)),
        "num_heads": (4, 4),
        "mlp_ratio": (2.0, 2.0, 1.0),
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        CrossViTConfig(**params)
