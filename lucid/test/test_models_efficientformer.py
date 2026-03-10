import pytest

import lucid

from lucid.models import (
    EfficientFormer,
    EfficientFormerConfig,
    efficientformer_l1,
    efficientformer_l3,
    efficientformer_l7,
)
from lucid.models.vision import efficientformer as efficientformer_module


class _ConfigCapture:
    def __init__(self, config) -> None:
        self.config = config


def _small_efficientformer_config() -> EfficientFormerConfig:
    return EfficientFormerConfig(
        depths=(1, 1, 1, 1),
        embed_dims=(16, 32, 48, 64),
        in_channels=1,
        num_classes=10,
        num_vit=1,
        mlp_ratios=2.0,
        drop_path_rate=0.0,
    )


def test_efficientformer_public_imports() -> None:
    assert EfficientFormer is not None
    assert EfficientFormerConfig is not None
    assert efficientformer_l1 is not None
    assert efficientformer_l3 is not None
    assert efficientformer_l7 is not None


def test_efficientformer_accepts_config_object() -> None:
    config = _small_efficientformer_config()

    model = EfficientFormer(config)

    assert model.config is config


def test_efficientformer_custom_config_forward_shape() -> None:
    model = EfficientFormer(_small_efficientformer_config())

    output = model(lucid.zeros(1, 1, 224, 224))

    assert output.shape == (1, 10)


def test_efficientformer_l1_factory_forward_shape() -> None:
    model = efficientformer_l1(num_classes=10)

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 10)


def test_efficientformer_l1_factory_forwards_allowed_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(efficientformer_module, "EfficientFormer", _ConfigCapture)

    model = efficientformer_l1(
        num_classes=10,
        in_channels=1,
        global_pool=False,
        downsamples=(False, True, True, False),
        mlp_ratios=2.0,
        pool_size=5,
        layer_scale_init_value=1e-4,
        drop_rate=0.1,
        proj_drop_rate=0.2,
        drop_path_rate=0.3,
    )

    assert model.config.num_classes == 10
    assert model.config.in_channels == 1
    assert model.config.global_pool is False
    assert model.config.downsamples == (False, True, True, False)
    assert model.config.depths == (3, 2, 6, 4)
    assert model.config.embed_dims == (48, 96, 224, 448)
    assert model.config.num_vit == 1
    assert model.config.mlp_ratios == 2.0
    assert model.config.pool_size == 5
    assert model.config.layer_scale_init_value == 1e-4
    assert model.config.drop_rate == 0.1
    assert model.config.proj_drop_rate == 0.2
    assert model.config.drop_path_rate == 0.3


@pytest.mark.parametrize(
    ("factory", "expected_depths", "expected_embed_dims", "expected_num_vit"),
    (
        (efficientformer_l1, (3, 2, 6, 4), (48, 96, 224, 448), 1),
        (efficientformer_l3, (4, 4, 12, 6), (64, 128, 320, 512), 4),
        (efficientformer_l7, (6, 6, 18, 8), (96, 192, 384, 768), 8),
    ),
)
def test_efficientformer_factories_apply_expected_preset_config(
    monkeypatch: pytest.MonkeyPatch,
    factory,
    expected_depths: tuple[int, ...],
    expected_embed_dims: tuple[int, ...],
    expected_num_vit: int,
) -> None:
    monkeypatch.setattr(efficientformer_module, "EfficientFormer", _ConfigCapture)

    model = factory(num_classes=10)

    assert model.config.num_classes == 10
    assert model.config.depths == expected_depths
    assert model.config.embed_dims == expected_embed_dims
    assert model.config.num_vit == expected_num_vit


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    (
        (efficientformer_l1, {"depths": (1, 1, 1, 1)}),
        (efficientformer_l3, {"embed_dims": (32, 64, 128, 256)}),
        (efficientformer_l7, {"num_vit": 2}),
    ),
)
def test_efficientformer_factories_reject_overriding_preset_fields(
    factory,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset depths, embed_dims, or num_vit",
    ):
        factory(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"depths": ()},
        {"depths": (1, 0, 1, 1)},
        {"embed_dims": (16, 32, 48)},
        {"embed_dims": (16, 0, 48, 64)},
        {"in_channels": 0},
        {"num_classes": -1},
        {"downsamples": (False, True)},
        {"num_vit": -1},
        {"mlp_ratios": 0.0},
        {"pool_size": 0},
        {"layer_scale_init_value": -1e-5},
        {"drop_rate": 1.0},
        {"proj_drop_rate": 1.0},
        {"drop_path_rate": 1.0},
    ),
)
def test_efficientformer_config_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    params = {
        "depths": (1, 1, 1, 1),
        "embed_dims": (16, 32, 48, 64),
        "in_channels": 1,
        "num_classes": 10,
        "num_vit": 1,
        "mlp_ratios": 2.0,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        EfficientFormerConfig(**params)
