import pytest

import lucid

from lucid.models import (
    InceptionResNet,
    InceptionResNetConfig,
    inception_resnet_v1,
    inception_resnet_v2,
)


def test_inception_resnet_public_imports() -> None:
    assert InceptionResNet is not None
    assert InceptionResNetConfig is not None
    assert inception_resnet_v1 is not None
    assert inception_resnet_v2 is not None


def test_inception_resnet_accepts_config_object() -> None:
    config = InceptionResNetConfig(variant="v1")

    model = InceptionResNet(config)

    assert model.config is config


@pytest.mark.parametrize("factory", (inception_resnet_v1, inception_resnet_v2))
def test_inception_resnet_factories_forward_default_shape(
    monkeypatch: pytest.MonkeyPatch, factory
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = factory()
    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_inception_resnet_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = inception_resnet_v2(
        num_classes=10,
        in_channels=1,
        dropout_prob=0.25,
    )
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == InceptionResNetConfig(
        variant="v2",
        num_classes=10,
        in_channels=1,
        dropout_prob=0.25,
    )
    assert output.shape == (1, 10)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    (
        ({"variant": "v3"}, "variant must be one of"),
        ({"variant": "v1", "num_classes": 0}, "num_classes must be greater than 0"),
        ({"variant": "v1", "in_channels": 0}, "in_channels must be greater than 0"),
        ({"variant": "v1", "dropout_prob": 1.0}, "dropout_prob must be in the range"),
    ),
)
def test_inception_resnet_config_rejects_invalid_values(
    config_kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        InceptionResNetConfig(**config_kwargs)
