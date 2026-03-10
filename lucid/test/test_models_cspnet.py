from functools import partial

import pytest

import lucid
import lucid.nn as nn

from lucid.models import (
    CSPNet,
    CSPNetConfig,
    csp_darknet_53,
    csp_resnet_50,
    csp_resnext_50_32x4d,
)
from lucid.models.vision import cspnet as cspnet_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_csp_resnet_config() -> CSPNetConfig:
    return CSPNetConfig(
        stage_specs=((32, 1, False), (64, 1, True)),
        stack_type="resnet",
        in_channels=3,
        stem_channels=16,
        num_classes=10,
        feature_channels=32,
    )


def test_cspnet_public_imports() -> None:
    assert CSPNet is not None
    assert CSPNetConfig is not None
    assert csp_resnet_50 is not None
    assert csp_resnext_50_32x4d is not None
    assert csp_darknet_53 is not None


def test_cspnet_accepts_config_object() -> None:
    config = _small_csp_resnet_config()

    model = CSPNet(config)

    assert model.config is config


def test_cspnet_custom_config_forward_shape() -> None:
    model = CSPNet(_small_csp_resnet_config())

    output = model(lucid.zeros(1, 3, 64, 64))

    assert output.shape == (1, 10)


def test_cspnet_forward_features_can_return_stage_outputs() -> None:
    model = CSPNet(_small_csp_resnet_config())

    features = model.forward_features(lucid.zeros(1, 3, 64, 64), return_stage_out=True)

    assert isinstance(features, list)
    assert len(features) == 2


def test_csp_resnet_50_factory_forward_shape() -> None:
    model = csp_resnet_50()

    output = model(lucid.zeros(1, 3, 64, 64))

    assert output.shape == (1, 1000)


def test_csp_resnet_50_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cspnet_module, "CSPNet", _ConfigCapture)

    model = csp_resnet_50(
        num_classes=10,
        split_ratio=0.4,
        stem_channels=32,
        global_pool="max",
        dropout=0.1,
        in_channels=1,
    )

    assert model.config == CSPNetConfig(
        stage_specs=((256, 3, False), (512, 4, True), (1024, 6, True), (2048, 3, True)),
        stack_type="resnet",
        in_channels=1,
        stem_channels=32,
        num_classes=10,
        split_ratio=0.4,
        global_pool="max",
        dropout=0.1,
        feature_channels=1024,
        groups=1,
        base_width=64,
    )


@pytest.mark.parametrize(
    ("factory", "expected_stack_type", "expected_stage_specs", "expected_stem_channels", "expected_feature_channels", "expected_groups", "expected_base_width", "expected_pre_kernel_size"),
    (
        (
            csp_resnet_50,
            "resnet",
            ((256, 3, False), (512, 4, True), (1024, 6, True), (2048, 3, True)),
            64,
            1024,
            1,
            64,
            1,
        ),
        (
            csp_resnext_50_32x4d,
            "resnext",
            ((256, 3, False), (512, 4, True), (1024, 6, True), (2048, 3, True)),
            64,
            1024,
            32,
            4,
            1,
        ),
        (
            csp_darknet_53,
            "darknet",
            ((64, 1, True), (128, 2, True), (256, 8, True), (512, 8, True), (1024, 4, True)),
            32,
            1024,
            1,
            64,
            3,
        ),
    ),
)
def test_cspnet_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_stack_type: str,
    expected_stage_specs: tuple[tuple[int, int, bool], ...],
    expected_stem_channels: int,
    expected_feature_channels: int,
    expected_groups: int,
    expected_base_width: int,
    expected_pre_kernel_size: int,
) -> None:
    monkeypatch.setattr(cspnet_module, "CSPNet", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_classes == 10
    assert model.config.stack_type == expected_stack_type
    assert model.config.stage_specs == expected_stage_specs
    assert model.config.stem_channels == expected_stem_channels
    assert model.config.feature_channels == expected_feature_channels
    assert model.config.groups == expected_groups
    assert model.config.base_width == expected_base_width
    assert model.config.pre_kernel_size == expected_pre_kernel_size


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    (
        (
            csp_resnet_50,
            {"groups": 2},
            "factory variants do not allow overriding preset stage_specs, stack_type, groups, base_width, or feature_channels",
        ),
        (
            csp_resnext_50_32x4d,
            {"feature_channels": 512},
            "factory variants do not allow overriding preset stage_specs, stack_type, groups, base_width, or feature_channels",
        ),
        (
            csp_darknet_53,
            {"pre_kernel_size": 1},
            "factory variants do not allow overriding preset stage_specs, stack_type, feature_channels, or pre_kernel_size",
        ),
    ),
)
def test_cspnet_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"stage_specs": []},
        {"stage_specs": [(32, 1)]},
        {"stage_specs": [(32, 0, False)]},
        {"stage_specs": [(32, 1, 1)]},
        {"stack_type": "invalid"},
        {"in_channels": 0},
        {"stem_channels": 0},
        {"num_classes": 0},
        {"norm": "invalid"},
        {"act": "invalid"},
        {"split_ratio": 1.0},
        {"global_pool": "invalid"},
        {"dropout": 1.0},
        {"feature_channels": 0},
        {"pre_kernel_size": 0},
        {"groups": 0},
        {"base_width": 0},
    ),
)
def test_cspnet_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "stage_specs": ((32, 1, False),),
        "stack_type": "resnet",
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        CSPNetConfig(**params)


def test_cspnet_darknet_config_supports_custom_act() -> None:
    config = CSPNetConfig(
        stage_specs=((32, 1, True),),
        stack_type="darknet",
        act=partial(nn.LeakyReLU, negative_slope=0.1),
        num_classes=10,
    )

    model = CSPNet(config)

    output = model(lucid.zeros(1, 3, 64, 64))

    assert output.shape == (1, 10)
