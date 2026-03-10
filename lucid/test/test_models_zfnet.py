import pytest

import lucid

from lucid.models import ZFNet, ZFNetConfig, zfnet


def test_zfnet_public_imports() -> None:
    assert ZFNet is not None
    assert ZFNetConfig is not None
    assert zfnet is not None


def test_zfnet_accepts_config_object() -> None:
    config = ZFNetConfig()

    model = ZFNet(config)

    assert model.config is config


def test_zfnet_forward_with_default_config() -> None:
    model = ZFNet(ZFNetConfig())

    output = model(lucid.zeros(2, 3, 224, 224))

    assert output.shape == (2, 1000)


def test_zfnet_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = zfnet(
        num_classes=10,
        in_channels=1,
        dropout=0.25,
        classifier_hidden_features=(512, 256),
    )
    output = model(lucid.zeros(2, 1, 224, 224))

    assert model.config == ZFNetConfig(
        num_classes=10,
        in_channels=1,
        dropout=0.25,
        classifier_hidden_features=(512, 256),
    )
    assert output.shape == (2, 10)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    (
        ({"num_classes": 0}, "num_classes must be greater than 0"),
        ({"in_channels": 0}, "in_channels must be greater than 0"),
        ({"dropout": -0.1}, "dropout must be in the range"),
        ({"dropout": 1.0}, "dropout must be in the range"),
        (
            {"classifier_hidden_features": (4096,)},
            "classifier_hidden_features must contain exactly 2 values",
        ),
        (
            {"classifier_hidden_features": (4096, 0)},
            "classifier_hidden_features values must be greater than 0",
        ),
    ),
)
def test_zfnet_config_rejects_invalid_values(
    config_kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ZFNetConfig(**config_kwargs)
