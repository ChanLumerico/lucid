import pytest

import lucid

from lucid.models import (
    PVT,
    PVT_V2,
    PVTConfig,
    PVTV2Config,
    pvt_huge,
    pvt_large,
    pvt_medium,
    pvt_small,
    pvt_tiny,
    pvt_v2_b0,
    pvt_v2_b1,
    pvt_v2_b2,
    pvt_v2_b2_li,
    pvt_v2_b3,
    pvt_v2_b4,
    pvt_v2_b5,
)
from lucid.models.vision import pvt as pvt_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_pvt_config() -> PVTConfig:
    return PVTConfig(
        img_size=32,
        num_classes=10,
        patch_size=4,
        in_channels=1,
        embed_dims=(8, 16, 32, 64),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(2.0, 2.0, 2.0, 2.0),
        depths=(1, 1, 1, 1),
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.0,
    )


def _small_pvt_v2_config() -> PVTV2Config:
    return PVTV2Config(
        img_size=32,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dims=(8, 16, 32, 64),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(2, 2, 2, 2),
        depths=(1, 1, 1, 1),
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.0,
    )


def test_pvt_public_imports() -> None:
    assert PVT is not None
    assert PVTConfig is not None
    assert PVT_V2 is not None
    assert PVTV2Config is not None
    assert pvt_tiny is not None
    assert pvt_small is not None
    assert pvt_medium is not None
    assert pvt_large is not None
    assert pvt_huge is not None
    assert pvt_v2_b0 is not None
    assert pvt_v2_b1 is not None
    assert pvt_v2_b2 is not None
    assert pvt_v2_b2_li is not None
    assert pvt_v2_b3 is not None
    assert pvt_v2_b4 is not None
    assert pvt_v2_b5 is not None


def test_pvt_accepts_config_object() -> None:
    config = _small_pvt_config()

    model = PVT(config)

    assert model.config is config


def test_pvt_v2_accepts_config_object() -> None:
    config = _small_pvt_v2_config()

    model = PVT_V2(config)

    assert model.config is config


def test_pvt_custom_config_forward_shape() -> None:
    model = PVT(_small_pvt_config())

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)


def test_pvt_v2_custom_config_forward_shape() -> None:
    model = PVT_V2(_small_pvt_v2_config())

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)


def test_pvt_tiny_factory_forward_shape() -> None:
    model = pvt_tiny(img_size=32, num_classes=10)

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 10)


def test_pvt_v2_b0_factory_forward_shape() -> None:
    model = pvt_v2_b0(img_size=32, num_classes=10)

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 10)


def test_pvt_tiny_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pvt_module, "PVT", _ConfigCapture)

    model = pvt_tiny(
        img_size=32,
        num_classes=10,
        in_channels=1,
        qk_scale=0.5,
        drop_rate=0.1,
        attn_drop_rate=0.2,
        drop_path_rate=0.3,
    )

    assert model.config.img_size == 32
    assert model.config.num_classes == 10
    assert model.config.in_channels == 1
    assert model.config.qk_scale == 0.5
    assert model.config.drop_rate == 0.1
    assert model.config.attn_drop_rate == 0.2
    assert model.config.drop_path_rate == 0.3
    assert model.config.patch_size == 4
    assert model.config.embed_dims == (64, 128, 320, 512)
    assert model.config.depths == (2, 2, 2, 2)


def test_pvt_v2_b0_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pvt_module, "PVT_V2", _ConfigCapture)

    model = pvt_v2_b0(
        img_size=32,
        num_classes=10,
        in_channels=1,
        qk_scale=0.5,
        drop_rate=0.1,
        attn_drop_rate=0.2,
        drop_path_rate=0.3,
        linear=True,
    )

    assert model.config.img_size == 32
    assert model.config.num_classes == 10
    assert model.config.in_channels == 1
    assert model.config.qk_scale == 0.5
    assert model.config.drop_rate == 0.1
    assert model.config.attn_drop_rate == 0.2
    assert model.config.drop_path_rate == 0.3
    assert model.config.linear is True
    assert model.config.patch_size == 7
    assert model.config.embed_dims == (32, 64, 160, 256)
    assert model.config.depths == (2, 2, 2, 2)


@pytest.mark.parametrize(
    ("factory", "expected_embed_dims", "expected_depths", "expected_num_heads"),
    (
        (pvt_tiny, (64, 128, 320, 512), (2, 2, 2, 2), (1, 2, 5, 8)),
        (pvt_small, (64, 128, 320, 512), (3, 4, 6, 3), (1, 2, 5, 8)),
        (pvt_medium, (64, 128, 320, 512), (3, 4, 18, 3), (1, 2, 5, 8)),
        (pvt_large, (64, 128, 320, 512), (3, 4, 27, 3), (1, 2, 5, 8)),
        (pvt_huge, (128, 256, 512, 768), (3, 10, 60, 3), (2, 4, 8, 12)),
    ),
)
def test_pvt_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_embed_dims: tuple[int, ...],
    expected_depths: tuple[int, ...],
    expected_num_heads: tuple[int, ...],
) -> None:
    monkeypatch.setattr(pvt_module, "PVT", _ConfigCapture)

    model = factory(img_size=32, num_classes=10)

    assert model.config.img_size == 32
    assert model.config.num_classes == 10
    assert model.config.patch_size == 4
    assert model.config.embed_dims == expected_embed_dims
    assert model.config.depths == expected_depths
    assert model.config.num_heads == expected_num_heads


@pytest.mark.parametrize(
    ("factory", "expected_embed_dims", "expected_depths", "expected_num_heads", "expected_linear"),
    (
        (pvt_v2_b0, (32, 64, 160, 256), (2, 2, 2, 2), (1, 2, 5, 8), False),
        (pvt_v2_b1, (64, 128, 320, 512), (2, 2, 2, 2), (1, 2, 5, 8), False),
        (pvt_v2_b2, (64, 128, 320, 512), (3, 4, 6, 3), (1, 2, 5, 8), False),
        (pvt_v2_b2_li, (64, 128, 320, 512), (3, 4, 6, 3), (1, 2, 5, 8), True),
        (pvt_v2_b3, (64, 128, 320, 512), (3, 4, 18, 3), (1, 2, 5, 8), False),
        (pvt_v2_b4, (64, 128, 320, 512), (3, 8, 27, 3), (1, 2, 5, 8), False),
        (pvt_v2_b5, (64, 128, 320, 512), (3, 6, 40, 3), (1, 2, 5, 8), False),
    ),
)
def test_pvt_v2_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_embed_dims: tuple[int, ...],
    expected_depths: tuple[int, ...],
    expected_num_heads: tuple[int, ...],
    expected_linear: bool,
) -> None:
    monkeypatch.setattr(pvt_module, "PVT_V2", _ConfigCapture)

    model = factory(img_size=32, num_classes=10)

    assert model.config.img_size == 32
    assert model.config.num_classes == 10
    assert model.config.patch_size == 7
    assert model.config.num_stages == 4
    assert model.config.embed_dims == expected_embed_dims
    assert model.config.depths == expected_depths
    assert model.config.num_heads == expected_num_heads
    assert model.config.linear is expected_linear


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    (
        (
            pvt_tiny,
            {"embed_dims": (32, 64, 128, 256)},
            "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, or sr_ratios",
        ),
        (
            pvt_huge,
            {"drop_path_rate": 0.1},
            "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or drop_path_rate",
        ),
        (
            pvt_v2_b0,
            {"num_stages": 5},
            "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or num_stages",
        ),
        (
            pvt_v2_b2_li,
            {"linear": False},
            "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, num_stages, or linear",
        ),
    ),
)
def test_pvt_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"img_size": 0},
        {"num_classes": -1},
        {"patch_size": 0},
        {"in_channels": 0},
        {"embed_dims": (8, 16, 32)},
        {"num_heads": (1, 2, 4, 0)},
        {"mlp_ratios": (2.0, 2.0, 0.0, 2.0)},
        {"depths": (1, 1, 1, 0)},
        {"sr_ratios": (8, 4, 2, 0)},
        {"embed_dims": (8, 16, 30, 64), "num_heads": (1, 2, 4, 8)},
        {"drop_rate": 1.0},
        {"attn_drop_rate": 1.0},
        {"drop_path_rate": 1.0},
        {"img_size": 8, "patch_size": 4},
    ),
)
def test_pvt_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "img_size": 32,
        "num_classes": 10,
        "patch_size": 4,
        "in_channels": 1,
        "embed_dims": (8, 16, 32, 64),
        "num_heads": (1, 2, 4, 8),
        "mlp_ratios": (2.0, 2.0, 2.0, 2.0),
        "depths": (1, 1, 1, 1),
        "sr_ratios": (8, 4, 2, 1),
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        PVTConfig(**params)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"img_size": 0},
        {"patch_size": 4},
        {"in_channels": 0},
        {"num_classes": -1},
        {"num_stages": 0},
        {"embed_dims": (8, 16, 32)},
        {"num_heads": (1, 2, 4, 0)},
        {"mlp_ratios": (2, 2, 0, 2)},
        {"depths": (1, 1, 1, 0)},
        {"sr_ratios": (8, 4, 2, 0)},
        {"embed_dims": (8, 16, 30, 64), "num_heads": (1, 2, 4, 8)},
        {"drop_rate": 1.0},
        {"attn_drop_rate": 1.0},
        {"drop_path_rate": 1.0},
    ),
)
def test_pvt_v2_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "img_size": 32,
        "patch_size": 7,
        "in_channels": 1,
        "num_classes": 10,
        "embed_dims": (8, 16, 32, 64),
        "num_heads": (1, 2, 4, 8),
        "mlp_ratios": (2, 2, 2, 2),
        "depths": (1, 1, 1, 1),
        "sr_ratios": (8, 4, 2, 1),
        "num_stages": 4,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        PVTV2Config(**params)
