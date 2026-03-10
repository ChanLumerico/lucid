import pytest

import lucid

from lucid.models import (
    ConvNeXt,
    ConvNeXt_V2,
    ConvNeXtConfig,
    ConvNeXtV2Config,
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    convnext_v2_atto,
    convnext_v2_base,
    convnext_v2_femto,
    convnext_v2_huge,
    convnext_v2_large,
    convnext_v2_nano,
    convnext_v2_pico,
    convnext_v2_tiny,
    convnext_xlarge,
)
from lucid.models.vision import convnext as convnext_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def test_convnext_family_public_imports() -> None:
    assert ConvNeXt is not None
    assert ConvNeXtConfig is not None
    assert ConvNeXt_V2 is not None
    assert ConvNeXtV2Config is not None
    assert convnext_tiny is not None
    assert convnext_small is not None
    assert convnext_base is not None
    assert convnext_large is not None
    assert convnext_xlarge is not None
    assert convnext_v2_atto is not None
    assert convnext_v2_femto is not None
    assert convnext_v2_pico is not None
    assert convnext_v2_nano is not None
    assert convnext_v2_tiny is not None
    assert convnext_v2_base is not None
    assert convnext_v2_large is not None
    assert convnext_v2_huge is not None


def test_convnext_accepts_config_object() -> None:
    config = ConvNeXtConfig(
        num_classes=10,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 256, 512),
        drop_path=0.1,
        layer_scale_init=0.0,
    )

    model = ConvNeXt(config)

    assert model.config is config


def test_convnext_v2_accepts_config_object() -> None:
    config = ConvNeXtV2Config(
        num_classes=10,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        drop_path=0.1,
    )

    model = ConvNeXt_V2(config)

    assert model.config is config


def test_convnext_tiny_forward_shape() -> None:
    model = convnext_tiny()

    output = model(lucid.zeros(1, 3, 64, 64))

    assert output.shape == (1, 1000)


def test_convnext_v2_atto_forward_shape() -> None:
    model = convnext_v2_atto()

    output = model(lucid.zeros(1, 3, 64, 64))

    assert output.shape == (1, 1000)


def test_convnext_tiny_factory_forwards_config_overrides() -> None:
    model = convnext_tiny(num_classes=10, drop_path=0.1, layer_scale_init=0.0)

    assert model.config == ConvNeXtConfig(
        num_classes=10,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path=0.1,
        layer_scale_init=0.0,
    )


def test_convnext_v2_atto_factory_forwards_config_overrides() -> None:
    model = convnext_v2_atto(num_classes=10, drop_path=0.1)

    assert model.config == ConvNeXtV2Config(
        num_classes=10,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        drop_path=0.1,
    )


@pytest.mark.parametrize(
    ("factory", "expected_depths", "expected_dims"),
    (
        (convnext_tiny, (3, 3, 9, 3), (96, 192, 384, 768)),
        (convnext_small, (3, 3, 27, 3), (96, 192, 364, 768)),
        (convnext_base, (3, 3, 27, 3), (128, 256, 512, 1024)),
        (convnext_large, (3, 3, 27, 3), (192, 384, 768, 1536)),
        (convnext_xlarge, (3, 3, 27, 3), (256, 512, 1024, 2048)),
    ),
)
def test_convnext_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_depths: tuple[int, int, int, int],
    expected_dims: tuple[int, int, int, int],
) -> None:
    monkeypatch.setattr(convnext_module, "ConvNeXt", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_classes == 10
    assert model.config.depths == expected_depths
    assert model.config.dims == expected_dims


@pytest.mark.parametrize(
    ("factory", "expected_depths", "expected_dims"),
    (
        (convnext_v2_atto, (2, 2, 6, 2), (40, 80, 160, 320)),
        (convnext_v2_femto, (2, 2, 6, 2), (48, 96, 192, 384)),
        (convnext_v2_pico, (2, 2, 6, 2), (64, 128, 256, 512)),
        (convnext_v2_nano, (2, 2, 8, 2), (80, 160, 320, 640)),
        (convnext_v2_tiny, (3, 3, 9, 3), (96, 192, 384, 768)),
        (convnext_v2_base, (3, 3, 27, 3), (128, 256, 512, 1024)),
        (convnext_v2_large, (3, 3, 27, 3), (192, 384, 768, 1536)),
        (convnext_v2_huge, (3, 3, 27, 3), (352, 704, 1408, 2816)),
    ),
)
def test_convnext_v2_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_depths: tuple[int, int, int, int],
    expected_dims: tuple[int, int, int, int],
) -> None:
    monkeypatch.setattr(convnext_module, "ConvNeXt_V2", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_classes == 10
    assert model.config.depths == expected_depths
    assert model.config.dims == expected_dims


@pytest.mark.parametrize(
    "factory",
    (
        convnext_tiny,
        convnext_v2_atto,
    ),
)
def test_convnext_family_factories_reject_overriding_preset_fields(factory) -> None:
    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset depths or dims",
    ):
        factory(depths=(1, 1, 1, 1))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_classes": 0},
        {"depths": (3, 3, 9)},
        {"dims": (96, 192, 384)},
        {"depths": (3, 0, 9, 3)},
        {"dims": (96, -1, 384, 768)},
        {"drop_path": 1.1},
        {"layer_scale_init": -1.0},
    ),
)
def test_convnext_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ConvNeXtConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_classes": 0},
        {"depths": (3, 3, 9)},
        {"dims": (96, 192, 384)},
        {"depths": (3, 0, 9, 3)},
        {"dims": (96, -1, 384, 768)},
        {"drop_path": 1.1},
    ),
)
def test_convnext_v2_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ConvNeXtV2Config(**kwargs)
