import pytest

import lucid

from lucid.models import (
    MobileNet,
    MobileNetConfig,
    MobileNet_V2,
    MobileNet_V3,
    MobileNet_V4,
    MobileNetV2Config,
    MobileNetV3Config,
    MobileNetV4Config,
    mobilenet,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
    mobilenet_v4_conv_large,
    mobilenet_v4_conv_medium,
    mobilenet_v4_conv_small,
    mobilenet_v4_hybrid_large,
    mobilenet_v4_hybrid_medium,
)


_MOBILENET_V2_STAGE_CONFIGS = (
    (32, 16, 1, 1, 1),
    (16, 24, 6, 2, 2),
    (24, 32, 6, 3, 2),
    (32, 64, 6, 4, 2),
    (64, 96, 6, 3, 1),
    (96, 160, 6, 3, 2),
    (160, 320, 6, 1, 1),
)

_MOBILENET_V3_SMALL_CFG = (
    (3, 16, 16, True, False, 2, 2),
    (3, 72, 24, False, False, 2, 4),
    (3, 88, 24, False, False, 1, 4),
    (5, 96, 40, True, True, 2, 4),
    (5, 240, 40, True, True, 1, 4),
    (5, 240, 40, True, True, 1, 4),
    (5, 120, 48, True, True, 1, 4),
    (5, 144, 48, True, True, 1, 4),
    (5, 288, 96, True, True, 2, 4),
    (5, 576, 96, True, True, 1, 4),
    (5, 576, 96, True, True, 1, 4),
)


def _minimal_v4_cfg() -> dict[str, dict[str, object]]:
    return {
        "conv0": {
            "block_name": "convbn",
            "num_blocks": 1,
            "block_specs": [[3, 16, 3, 2]],
        },
        "layer1": {
            "block_name": "convbn",
            "num_blocks": 1,
            "block_specs": [[16, 16, 1, 1]],
        },
        "layer2": {
            "block_name": "uib",
            "num_blocks": 1,
            "block_specs": [[16, 24, 3, 3, True, 2, 2]],
        },
        "layer3": {
            "block_name": "uib",
            "num_blocks": 1,
            "block_specs": [[24, 24, 3, 3, True, 1, 2]],
        },
        "layer4": {
            "block_name": "fused_ib",
            "num_blocks": 1,
            "block_specs": [[24, 32, 1, 4.0, True]],
        },
        "layer5": {
            "block_name": "convbn",
            "num_blocks": 1,
            "block_specs": [[32, 64, 1, 1]],
        },
    }


def test_mobile_family_public_imports() -> None:
    assert MobileNet is not None
    assert MobileNetConfig is not None
    assert MobileNet_V2 is not None
    assert MobileNetV2Config is not None
    assert MobileNet_V3 is not None
    assert MobileNetV3Config is not None
    assert MobileNet_V4 is not None
    assert MobileNetV4Config is not None
    assert mobilenet is not None
    assert mobilenet_v2 is not None
    assert mobilenet_v3_small is not None
    assert mobilenet_v3_large is not None
    assert mobilenet_v4_conv_small is not None
    assert mobilenet_v4_conv_medium is not None
    assert mobilenet_v4_conv_large is not None
    assert mobilenet_v4_hybrid_medium is not None
    assert mobilenet_v4_hybrid_large is not None


def test_mobilenet_accepts_config_object() -> None:
    config = MobileNetConfig(width_multiplier=0.75, num_classes=10, in_channels=1)

    model = MobileNet(config)

    assert model.config is config


def test_mobilenet_v2_accepts_config_object() -> None:
    config = MobileNetV2Config(
        stage_configs=_MOBILENET_V2_STAGE_CONFIGS,
        num_classes=10,
        in_channels=1,
    )

    model = MobileNet_V2(config)

    assert model.config is config


def test_mobilenet_v3_accepts_config_object() -> None:
    config = MobileNetV3Config(
        bottleneck_cfg=_MOBILENET_V3_SMALL_CFG,
        last_channels=1024,
        num_classes=10,
        in_channels=1,
    )

    model = MobileNet_V3(config)

    assert model.config is config


def test_mobilenet_v4_accepts_config_object() -> None:
    config = MobileNetV4Config(cfg=_minimal_v4_cfg(), num_classes=10)

    model = MobileNet_V4(config)

    assert model.config is config


@pytest.mark.parametrize(
    "factory",
    (
        mobilenet,
        mobilenet_v2,
        mobilenet_v3_small,
        mobilenet_v3_large,
        mobilenet_v4_conv_small,
        mobilenet_v4_conv_medium,
        mobilenet_v4_conv_large,
        mobilenet_v4_hybrid_medium,
        mobilenet_v4_hybrid_large,
    ),
)
def test_mobile_family_factories_forward_default_shape(factory) -> None:
    model = factory()

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_mobilenet_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = mobilenet(width_multiplier=0.5, num_classes=10, in_channels=1)
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == MobileNetConfig(
        width_multiplier=0.5,
        num_classes=10,
        in_channels=1,
    )
    assert output.shape == (1, 10)


def test_mobilenet_v2_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = mobilenet_v2(num_classes=10, in_channels=1, dropout=0.1)
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == MobileNetV2Config(
        stage_configs=_MOBILENET_V2_STAGE_CONFIGS,
        num_classes=10,
        in_channels=1,
        dropout=0.1,
    )
    assert output.shape == (1, 10)


def test_mobilenet_v3_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = mobilenet_v3_small(num_classes=10, in_channels=1, dropout=0.1)
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == MobileNetV3Config(
        bottleneck_cfg=_MOBILENET_V3_SMALL_CFG,
        last_channels=1024,
        num_classes=10,
        in_channels=1,
        dropout=0.1,
    )
    assert output.shape == (1, 10)


def test_mobilenet_v4_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = mobilenet_v4_conv_small(num_classes=10)
    output = model(lucid.zeros(1, 3, 224, 224))

    assert model.config.num_classes == 10
    assert output.shape == (1, 10)


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    (
        (
            mobilenet_v2,
            {"stage_configs": [(32, 16, 1, 1, 1)]},
            "factory variants do not allow overriding preset stage_configs, stem_channels, or last_channels",
        ),
        (
            mobilenet_v3_small,
            {"bottleneck_cfg": []},
            "factory variants do not allow overriding preset bottleneck_cfg, last_channels, or stem_channels",
        ),
        (
            mobilenet_v4_conv_small,
            {"cfg": {}},
            "factory variants do not allow overriding preset cfg",
        ),
    ),
)
def test_mobile_family_factories_reject_overriding_preset_fields(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    kwargs: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    with pytest.raises(TypeError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    ("config_cls", "config_kwargs", "error_type", "message"),
    (
        (
            MobileNetConfig,
            {"width_multiplier": 0},
            ValueError,
            "width_multiplier must be greater than 0",
        ),
        (
            MobileNetV2Config,
            {"stage_configs": [(32, 16, 1, 1)]},
            ValueError,
            "each stage config must contain exactly 5 values",
        ),
        (
            MobileNetV2Config,
            {"stage_configs": _MOBILENET_V2_STAGE_CONFIGS, "dropout": 1.0},
            ValueError,
            "dropout must be in the range \\[0, 1\\)",
        ),
        (
            MobileNetV3Config,
            {"bottleneck_cfg": [(3, 16, 16, True, False, 2)], "last_channels": 1024},
            ValueError,
            "each bottleneck spec must contain exactly 7 values",
        ),
        (
            MobileNetV3Config,
            {
                "bottleneck_cfg": [(3, 16, 16, 1, False, 2, 4)],
                "last_channels": 1024,
            },
            TypeError,
            "bottleneck spec use_se and use_hswish values must be booleans",
        ),
        (
            MobileNetV4Config,
            {"cfg": []},
            TypeError,
            "cfg must be a dictionary",
        ),
        (
            MobileNetV4Config,
            {
                "cfg": {
                    "conv0": {
                        "block_name": "convbn",
                        "num_blocks": 1,
                        "block_specs": [[3, 16, 3, 2]],
                    }
                }
            },
            ValueError,
            "cfg must define conv0 and layer1-layer5 blocks",
        ),
    ),
)
def test_mobile_family_config_rejects_invalid_values(
    config_cls,
    config_kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        config_cls(**config_kwargs)
