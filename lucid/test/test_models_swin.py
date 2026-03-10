import pytest
import numpy as np

import lucid

from lucid.models import (
    SwinTransformer,
    SwinTransformer_V2,
    SwinTransformerConfig,
    SwinTransformerV2Config,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    swin_v2_base,
    swin_v2_giant,
    swin_v2_huge,
    swin_v2_large,
    swin_v2_small,
    swin_v2_tiny,
)
from lucid.models.vision import swin as swin_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_swin_config() -> SwinTransformerConfig:
    return SwinTransformerConfig(
        img_size=32,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=8,
        depths=(2, 2),
        num_heads=(2, 4),
        window_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )


def _small_swin_v2_config(*, qkv_bias: bool = True) -> SwinTransformerV2Config:
    return SwinTransformerV2Config(
        img_size=32,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=8,
        depths=(2, 2),
        num_heads=(2, 4),
        window_size=7,
        qkv_bias=qkv_bias,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )


def test_swin_public_imports() -> None:
    assert SwinTransformer is not None
    assert SwinTransformerConfig is not None
    assert SwinTransformer_V2 is not None
    assert SwinTransformerV2Config is not None
    assert swin_tiny is not None
    assert swin_small is not None
    assert swin_base is not None
    assert swin_large is not None
    assert swin_v2_tiny is not None
    assert swin_v2_small is not None
    assert swin_v2_base is not None
    assert swin_v2_large is not None
    assert swin_v2_huge is not None
    assert swin_v2_giant is not None


def test_swin_accepts_config_object() -> None:
    config = _small_swin_config()

    model = SwinTransformer(config)

    assert model.config is config


def test_swin_v2_accepts_config_object() -> None:
    config = _small_swin_v2_config()

    model = SwinTransformer_V2(config)

    assert model.config is config


def test_swin_custom_config_forward_shape() -> None:
    model = SwinTransformer(_small_swin_config())

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)


def test_swin_v2_custom_config_forward_shape() -> None:
    model = SwinTransformer_V2(_small_swin_v2_config())

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)


def test_swin_v2_custom_config_supports_qkv_bias_disabled() -> None:
    model = SwinTransformer_V2(_small_swin_v2_config(qkv_bias=False))

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)
    assert np.isfinite(output.data).all()


def test_swin_tiny_factory_forward_shape() -> None:
    model = swin_tiny(img_size=32, num_classes=10)

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 10)


def test_swin_v2_tiny_factory_forward_shape() -> None:
    model = swin_v2_tiny(img_size=32, num_classes=10)

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 10)


def test_swin_tiny_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swin_module, "SwinTransformer", _ConfigCapture)

    model = swin_tiny(
        img_size=32,
        num_classes=10,
        patch_size=2,
        in_channels=1,
        window_size=5,
        abs_pos_emb=True,
        patch_norm=False,
        qkv_bias=False,
    )

    assert model.config == SwinTransformerConfig(
        img_size=32,
        patch_size=2,
        in_channels=1,
        num_classes=10,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=5,
        abs_pos_emb=True,
        patch_norm=False,
        qkv_bias=False,
    )


def test_swin_v2_tiny_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(swin_module, "SwinTransformer_V2", _ConfigCapture)

    model = swin_v2_tiny(
        img_size=32,
        num_classes=10,
        patch_size=2,
        in_channels=1,
        window_size=5,
        abs_pos_emb=True,
        patch_norm=False,
        qkv_bias=False,
    )

    assert model.config == SwinTransformerV2Config(
        img_size=32,
        patch_size=2,
        in_channels=1,
        num_classes=10,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=5,
        abs_pos_emb=True,
        patch_norm=False,
        qkv_bias=False,
    )


@pytest.mark.parametrize(
    ("factory", "expected_embed_dim", "expected_depths", "expected_num_heads"),
    (
        (swin_tiny, 96, (2, 2, 6, 2), (3, 6, 12, 24)),
        (swin_small, 96, (2, 2, 18, 2), (3, 6, 12, 24)),
        (swin_base, 128, (2, 2, 18, 2), (4, 8, 16, 32)),
        (swin_large, 192, (2, 2, 18, 2), (6, 12, 24, 48)),
    ),
)
def test_swin_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_embed_dim: int,
    expected_depths: tuple[int, ...],
    expected_num_heads: tuple[int, ...],
) -> None:
    monkeypatch.setattr(swin_module, "SwinTransformer", _ConfigCapture)

    model = factory(img_size=32, num_classes=10)

    assert model.config.img_size == 32
    assert model.config.num_classes == 10
    assert model.config.embed_dim == expected_embed_dim
    assert model.config.depths == expected_depths
    assert model.config.num_heads == expected_num_heads


@pytest.mark.parametrize(
    ("factory", "expected_embed_dim", "expected_depths", "expected_num_heads"),
    (
        (swin_v2_tiny, 96, (2, 2, 6, 2), (3, 6, 12, 24)),
        (swin_v2_small, 96, (2, 2, 18, 2), (3, 6, 12, 24)),
        (swin_v2_base, 128, (2, 2, 18, 2), (4, 8, 16, 32)),
        (swin_v2_large, 192, (2, 2, 18, 2), (6, 12, 24, 48)),
        (swin_v2_huge, 352, (2, 2, 18, 2), (11, 22, 44, 88)),
        (swin_v2_giant, 512, (2, 2, 42, 4), (16, 32, 64, 128)),
    ),
)
def test_swin_v2_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_embed_dim: int,
    expected_depths: tuple[int, ...],
    expected_num_heads: tuple[int, ...],
) -> None:
    monkeypatch.setattr(swin_module, "SwinTransformer_V2", _ConfigCapture)

    model = factory(img_size=32, num_classes=10)

    assert model.config.img_size == 32
    assert model.config.num_classes == 10
    assert model.config.embed_dim == expected_embed_dim
    assert model.config.depths == expected_depths
    assert model.config.num_heads == expected_num_heads


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    (
        (swin_tiny, {"embed_dim": 128}),
        (swin_tiny, {"depths": (1, 1, 1, 1)}),
        (swin_tiny, {"num_heads": (1, 2, 4, 8)}),
        (swin_v2_tiny, {"embed_dim": 128}),
        (swin_v2_tiny, {"depths": (1, 1, 1, 1)}),
        (swin_v2_tiny, {"num_heads": (1, 2, 4, 8)}),
    ),
)
def test_swin_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset embed_dim, depths, or num_heads",
    ):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"img_size": 0},
        {"patch_size": 0},
        {"img_size": 2, "patch_size": 4},
        {"in_channels": 0},
        {"num_classes": -1},
        {"embed_dim": 0},
        {"depths": ()},
        {"depths": (2, 2), "num_heads": (2,)},
        {"depths": (0, 2)},
        {"num_heads": (0, 2)},
        {"embed_dim": 10, "depths": (2, 2), "num_heads": (3, 4)},
        {"img_size": 16, "patch_size": 4, "depths": (2, 2, 2, 2), "num_heads": (1, 2, 4, 8)},
        {"window_size": 0},
        {"mlp_ratio": 0.0},
        {"drop_rate": 1.0},
        {"attn_drop_rate": 1.0},
        {"drop_path_rate": 1.0},
    ),
)
def test_swin_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "img_size": 32,
        "patch_size": 4,
        "in_channels": 1,
        "num_classes": 10,
        "embed_dim": 8,
        "depths": (2, 2),
        "num_heads": (2, 4),
        "window_size": 7,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        SwinTransformerConfig(**params)
