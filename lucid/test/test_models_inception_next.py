from functools import partial

import pytest

import lucid
import lucid.nn as nn

from lucid.models import (
    InceptionNeXt,
    InceptionNeXtConfig,
    inception_next_atto,
    inception_next_base,
    inception_next_small,
    inception_next_tiny,
)
from lucid.models.vision import inception_next as inception_next_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def test_inception_next_public_imports() -> None:
    assert InceptionNeXt is not None
    assert InceptionNeXtConfig is not None
    assert inception_next_atto is not None
    assert inception_next_tiny is not None
    assert inception_next_small is not None
    assert inception_next_base is not None


def test_inception_next_accepts_config_object() -> None:
    config = InceptionNeXtConfig(
        num_classes=10,
        depths=(2, 2),
        dims=(32, 64),
        token_mixers=partial(
            inception_next_module._InceptionDWConv2d,
            band_kernel_size=9,
        ),
        mlp_ratios=2,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )

    model = InceptionNeXt(config)

    assert model.config is config


def test_inception_next_atto_forward_shape() -> None:
    model = inception_next_atto()

    output = model(lucid.zeros(1, 3, 64, 64))

    assert output.shape == (1, 1000)


def test_inception_next_factory_supports_drop_path_rate() -> None:
    model = inception_next_atto(num_classes=10, drop_path_rate=0.1)

    output = model(lucid.zeros(1, 3, 64, 64))

    assert model.config.drop_path_rate == pytest.approx(0.1)
    assert output.shape == (1, 10)


def test_inception_next_tiny_factory_forwards_config_overrides() -> None:
    model = inception_next_tiny(
        num_classes=10,
        mlp_ratios=(4, 4, 4, 4),
        drop_rate=0.1,
        drop_path_rate=0.1,
    )

    assert model.config == InceptionNeXtConfig(
        num_classes=10,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        token_mixers=inception_next_module._InceptionDWConv2d,
        mlp_ratios=(4, 4, 4, 4),
        drop_rate=0.1,
        drop_path_rate=0.1,
    )


@pytest.mark.parametrize(
    ("factory", "expected_depths", "expected_dims"),
    (
        (inception_next_atto, (2, 2, 6, 2), (40, 80, 160, 320)),
        (inception_next_tiny, (3, 3, 9, 3), (96, 192, 384, 768)),
        (inception_next_small, (3, 3, 27, 3), (96, 192, 384, 768)),
        (inception_next_base, (3, 3, 27, 3), (128, 256, 512, 1024)),
    ),
)
def test_inception_next_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_depths: tuple[int, ...],
    expected_dims: tuple[int, ...],
) -> None:
    monkeypatch.setattr(inception_next_module, "InceptionNeXt", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_classes == 10
    assert model.config.depths == expected_depths
    assert model.config.dims == expected_dims


def test_inception_next_atto_uses_band_kernel_size_9_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inception_next_module, "InceptionNeXt", _ConfigCapture)

    model = inception_next_atto(num_classes=10)

    assert all(isinstance(token_mixer, partial) for token_mixer in model.config.token_mixers)
    assert all(
        token_mixer.keywords == {"band_kernel_size": 9}
        for token_mixer in model.config.token_mixers
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"depths": (1, 1, 1, 1)},
        {"token_mixers": nn.Identity},
    ),
)
def test_inception_next_factories_reject_overriding_preset_fields(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset depths, dims, or token_mixers",
    ):
        inception_next_atto(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"num_classes": 0},
        {"depths": ()},
        {"dims": (96, 192, 384)},
        {"depths": (3, 3, 9, 3), "dims": (96, 192, 384)},
        {"token_mixers": "invalid"},
        {"token_mixers": (nn.Identity,)},
        {"mlp_ratios": 0},
        {"mlp_ratios": (4, 4, 4)},
        {"head_fn": "invalid"},
        {"drop_rate": 1.0},
        {"drop_path_rate": 1.1},
        {"ls_init_value": -1.0},
    ),
)
def test_inception_next_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        InceptionNeXtConfig(**kwargs)
