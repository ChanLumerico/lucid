import pytest

import lucid

from lucid.models import (
    EfficientNet,
    EfficientNet_V2,
    EfficientNetConfig,
    EfficientNetV2Config,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
    efficientnet_v2_xl,
)
from lucid.models.vision import efficient as efficient_module


_MINIMAL_V2_BLOCK_CFG = (
    (True, 24, 3, 1, 1, 1, 0),
    (False, 32, 3, 2, 4, 1, 4),
)


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def test_efficient_family_public_imports() -> None:
    assert EfficientNet is not None
    assert EfficientNetConfig is not None
    assert EfficientNet_V2 is not None
    assert EfficientNetV2Config is not None
    assert efficientnet_b0 is not None
    assert efficientnet_b1 is not None
    assert efficientnet_b2 is not None
    assert efficientnet_b3 is not None
    assert efficientnet_b4 is not None
    assert efficientnet_b5 is not None
    assert efficientnet_b6 is not None
    assert efficientnet_b7 is not None
    assert efficientnet_v2_s is not None
    assert efficientnet_v2_m is not None
    assert efficientnet_v2_l is not None
    assert efficientnet_v2_xl is not None


def test_efficientnet_accepts_config_object() -> None:
    config = EfficientNetConfig(
        num_classes=10,
        width_coef=0.75,
        depth_coef=1.1,
        scale=1.0,
        se_scale=8,
    )

    model = EfficientNet(config)

    assert model.config is config


def test_efficientnet_v2_accepts_config_object() -> None:
    config = EfficientNetV2Config(
        block_cfg=_MINIMAL_V2_BLOCK_CFG,
        num_classes=10,
        dropout=0.1,
        drop_path_rate=0.1,
    )

    model = EfficientNet_V2(config)

    assert model.config is config


def test_efficientnet_b0_forward_shape() -> None:
    model = efficientnet_b0()

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 1000)


def test_efficientnet_with_stochastic_depth_constructs_and_forwards() -> None:
    config = EfficientNetConfig(
        num_classes=10,
        stochastic_depth=True,
        p=0.8,
    )
    model = EfficientNet(config)
    model.eval()

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 10)


def test_efficientnet_v2_s_forward_shape() -> None:
    model = efficientnet_v2_s()

    output = model(lucid.zeros(1, 3, 32, 32))

    assert output.shape == (1, 1000)


def test_efficientnet_b0_factory_forwards_config_overrides() -> None:
    model = efficientnet_b0(num_classes=10, se_scale=8, stochastic_depth=True, p=0.7)

    assert model.config == EfficientNetConfig(
        num_classes=10,
        width_coef=1.0,
        depth_coef=1.0,
        scale=1.0,
        dropout=0.2,
        se_scale=8,
        stochastic_depth=True,
        p=0.7,
    )


def test_efficientnet_v2_s_factory_forwards_config_overrides() -> None:
    model = efficientnet_v2_s(num_classes=10)

    assert model.config == EfficientNetV2Config(
        block_cfg=(
            (True, 24, 3, 2, 1, 2, 0),
            (True, 48, 3, 2, 4, 4, 0),
            (True, 64, 3, 2, 4, 4, 0),
            (False, 128, 3, 2, 4, 6, 4),
            (False, 160, 3, 1, 6, 9, 4),
            (False, 256, 3, 2, 6, 15, 4),
        ),
        num_classes=10,
        dropout=0.2,
        drop_path_rate=0.2,
    )


@pytest.mark.parametrize(
    ("factory", "expected"),
    (
        (
            efficientnet_b0,
            {"width_coef": 1.0, "depth_coef": 1.0, "scale": 224 / 224, "dropout": 0.2},
        ),
        (
            efficientnet_b1,
            {"width_coef": 1.0, "depth_coef": 1.1, "scale": 240 / 224, "dropout": 0.2},
        ),
        (
            efficientnet_b2,
            {"width_coef": 1.1, "depth_coef": 1.2, "scale": 260 / 224, "dropout": 0.3},
        ),
        (
            efficientnet_b3,
            {"width_coef": 1.2, "depth_coef": 1.4, "scale": 300 / 224, "dropout": 0.3},
        ),
        (
            efficientnet_b4,
            {"width_coef": 1.4, "depth_coef": 1.8, "scale": 380 / 224, "dropout": 0.4},
        ),
        (
            efficientnet_b5,
            {"width_coef": 1.6, "depth_coef": 2.2, "scale": 456 / 224, "dropout": 0.4},
        ),
        (
            efficientnet_b6,
            {"width_coef": 1.8, "depth_coef": 2.6, "scale": 528 / 224, "dropout": 0.5},
        ),
        (
            efficientnet_b7,
            {"width_coef": 2.0, "depth_coef": 3.1, "scale": 600 / 224, "dropout": 0.5},
        ),
    ),
)
def test_efficientnet_b_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected: dict[str, float],
) -> None:
    monkeypatch.setattr(efficient_module, "EfficientNet", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_classes == 10
    assert model.config.width_coef == pytest.approx(expected["width_coef"])
    assert model.config.depth_coef == pytest.approx(expected["depth_coef"])
    assert model.config.scale == pytest.approx(expected["scale"])
    assert model.config.dropout == pytest.approx(expected["dropout"])


@pytest.mark.parametrize(
    ("factory", "expected_dropout", "expected_drop_path_rate", "expected_block_count"),
    (
        (efficientnet_v2_s, 0.2, 0.2, 6),
        (efficientnet_v2_m, 0.3, 0.2, 7),
        (efficientnet_v2_l, 0.4, 0.3, 7),
        (efficientnet_v2_xl, 0.5, 0.4, 7),
    ),
)
def test_efficientnet_v2_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_dropout: float,
    expected_drop_path_rate: float,
    expected_block_count: int,
) -> None:
    monkeypatch.setattr(efficient_module, "EfficientNet_V2", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_classes == 10
    assert model.config.dropout == pytest.approx(expected_dropout)
    assert model.config.drop_path_rate == pytest.approx(expected_drop_path_rate)
    assert len(model.config.block_cfg) == expected_block_count


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    (
        (
            efficientnet_b0,
            {"width_coef": 2.0},
            "factory variants do not allow overriding preset width_coef, depth_coef, scale, or dropout",
        ),
        (
            efficientnet_b0,
            {"dropout": 0.5},
            "factory variants do not allow overriding preset width_coef, depth_coef, scale, or dropout",
        ),
        (
            efficientnet_v2_s,
            {"block_cfg": []},
            "factory variants do not allow overriding preset block_cfg, dropout, or drop_path_rate",
        ),
        (
            efficientnet_v2_s,
            {"dropout": 0.5},
            "factory variants do not allow overriding preset block_cfg, dropout, or drop_path_rate",
        ),
    ),
)
def test_efficient_family_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_classes": 0},
        {"width_coef": 0},
        {"depth_coef": 0},
        {"scale": 0},
        {"dropout": 1.0},
        {"se_scale": 0},
        {"p": 1.5},
    ),
)
def test_efficientnet_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        EfficientNetConfig(**kwargs)


def test_efficientnet_config_rejects_non_boolean_stochastic_depth() -> None:
    with pytest.raises(TypeError, match="stochastic_depth must be a boolean"):
        EfficientNetConfig(stochastic_depth=1)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"block_cfg": []},
        {"block_cfg": [(True, 24, 3, 1, 1, 1)]},
        {"block_cfg": [(1, 24, 3, 1, 1, 1, 0)]},
        {"block_cfg": [(True, 24, 3, 1, 0, 1, 0)]},
        {"block_cfg": [(True, 24, 3, 1, 1, 1, -1)]},
        {"num_classes": 0},
        {"dropout": 1.0},
        {"drop_path_rate": 1.1},
    ),
)
def test_efficientnet_v2_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "block_cfg": _MINIMAL_V2_BLOCK_CFG,
        "num_classes": 1000,
        "dropout": 0.2,
        "drop_path_rate": 0.2,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        EfficientNetV2Config(**params)
