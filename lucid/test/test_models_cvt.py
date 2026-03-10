from functools import partial

import pytest

import lucid
import lucid.nn as nn

from lucid.models import CvT, CvTConfig, cvt_13, cvt_21, cvt_w24
from lucid.models.vision import cvt as cvt_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_cvt_config() -> CvTConfig:
    return CvTConfig(
        num_stages=3,
        patch_size=(3, 3, 3),
        patch_stride=(2, 2, 2),
        patch_padding=(1, 1, 1),
        dim_embed=(16, 32, 64),
        num_heads=(1, 2, 4),
        depth=(1, 1, 1),
        in_channels=1,
        num_classes=10,
        act_layer=cvt_module._QuickGELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-5),
    )


def test_cvt_public_imports() -> None:
    assert CvT is not None
    assert CvTConfig is not None
    assert cvt_13 is not None
    assert cvt_21 is not None
    assert cvt_w24 is not None


def test_cvt_accepts_config_object() -> None:
    config = _small_cvt_config()

    model = CvT(config)

    assert model.config is config


def test_cvt_custom_config_forward_shape() -> None:
    model = CvT(_small_cvt_config())

    output = model(lucid.zeros(1, 1, 32, 32))

    assert output.shape == (1, 10)


def test_cvt_13_factory_forward_shape() -> None:
    model = cvt_13(num_classes=10)

    output = model(lucid.zeros(1, 3, 64, 64))

    assert output.shape == (1, 10)


def test_cvt_13_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cvt_module, "CvT", _ConfigCapture)

    model = cvt_13(
        num_classes=10,
        in_channels=1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_path_rate=(0.0, 0.0, 0.2),
        cls_token=(False, False, True),
        qkv_proj_method=("dw_bn", "avg", "lin"),
    )

    assert model.config == CvTConfig(
        num_stages=3,
        patch_size=(7, 3, 3),
        patch_stride=(4, 2, 2),
        patch_padding=(2, 1, 1),
        dim_embed=(64, 192, 384),
        num_heads=(1, 3, 6),
        depth=(1, 2, 10),
        in_channels=1,
        num_classes=10,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_path_rate=(0.0, 0.0, 0.2),
        cls_token=(False, False, True),
        qkv_proj_method=("dw_bn", "avg", "lin"),
    )


@pytest.mark.parametrize(
    ("factory", "expected_dim_embed", "expected_num_heads", "expected_depth", "expected_drop_path_rate"),
    (
        (cvt_13, (64, 192, 384), (1, 3, 6), (1, 2, 10), (0.0, 0.0, 0.1)),
        (cvt_21, (64, 192, 384), (1, 3, 6), (1, 4, 16), (0.0, 0.0, 0.1)),
        (cvt_w24, (192, 768, 1024), (3, 12, 16), (2, 2, 20), (0.0, 0.0, 0.3)),
    ),
)
def test_cvt_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_dim_embed: tuple[int, ...],
    expected_num_heads: tuple[int, ...],
    expected_depth: tuple[int, ...],
    expected_drop_path_rate: tuple[float, ...],
) -> None:
    monkeypatch.setattr(cvt_module, "CvT", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_stages == 3
    assert model.config.patch_size == (7, 3, 3)
    assert model.config.patch_stride == (4, 2, 2)
    assert model.config.patch_padding == (2, 1, 1)
    assert model.config.dim_embed == expected_dim_embed
    assert model.config.num_heads == expected_num_heads
    assert model.config.depth == expected_depth
    assert model.config.in_channels == 3
    assert model.config.num_classes == 10
    assert model.config.act_layer is cvt_module._QuickGELU
    assert isinstance(model.config.norm_layer, partial)
    assert model.config.norm_layer.func is nn.LayerNorm
    assert model.config.norm_layer.keywords == {"eps": 1e-5}
    assert model.config.drop_path_rate == expected_drop_path_rate


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_stages": 2},
        {"patch_size": (3, 3)},
        {"patch_stride": (2, 0, 2)},
        {"patch_padding": (1, -1, 1)},
        {"dim_embed": (16, 31, 64), "num_heads": (1, 2, 4)},
        {"depth": (1, 0, 1)},
        {"in_channels": 0},
        {"num_classes": -1},
        {"mlp_ratio": (4.0, 0.0, 4.0)},
        {"drop_rate": (0.0, 1.0, 0.0)},
        {"qkv_proj_method": ("dw_bn", "bad", "lin")},
        {"kernel_qkv": (3, 0, 3)},
    ),
)
def test_cvt_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "num_stages": 3,
        "patch_size": (3, 3, 3),
        "patch_stride": (2, 2, 2),
        "patch_padding": (1, 1, 1),
        "dim_embed": (16, 32, 64),
        "num_heads": (1, 2, 4),
        "depth": (1, 1, 1),
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        CvTConfig(**params)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    (
        (cvt_13, {"num_stages": 2}),
        (cvt_13, {"dim_embed": (32, 64, 128)}),
        (cvt_21, {"num_heads": (1, 2, 4)}),
        (cvt_w24, {"depth": (1, 1, 1)}),
    ),
)
def test_cvt_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset num_stages, patch_size, patch_stride, patch_padding, dim_embed, num_heads, or depth",
    ):
        factory(**kwargs)
