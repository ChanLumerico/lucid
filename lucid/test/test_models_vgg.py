import pytest

import lucid

from lucid.models import VGGNet, VGGNetConfig, vggnet_11, vggnet_13, vggnet_16, vggnet_19


def test_vggnet_public_imports() -> None:
    assert VGGNet is not None
    assert VGGNetConfig is not None
    assert vggnet_11 is not None
    assert vggnet_13 is not None
    assert vggnet_16 is not None
    assert vggnet_19 is not None


def test_vggnet_accepts_config_object() -> None:
    config = VGGNetConfig(conv_config=[64, "M", 128, "M", 256, 256, "M"])

    model = VGGNet(config)

    assert model.config is config


@pytest.mark.parametrize("factory", (vggnet_11, vggnet_13, vggnet_16, vggnet_19))
def test_vggnet_factories_forward_default_shape(factory) -> None:
    model = factory()

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_vggnet_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = vggnet_11(
        num_classes=10,
        in_channels=1,
        dropout=0.25,
        classifier_hidden_features=(512, 256),
    )
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == VGGNetConfig(
        conv_config=[64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        num_classes=10,
        in_channels=1,
        dropout=0.25,
        classifier_hidden_features=(512, 256),
    )
    assert output.shape == (1, 10)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    (
        ({"conv_config": []}, "conv_config must not be empty"),
        (
            {"conv_config": ["M"]},
            "conv_config must include at least 1 conv layer",
        ),
        (
            {"conv_config": [64, "X"]},
            "entries must be positive integers or 'M'",
        ),
        (
            {"conv_config": [64, 0]},
            "entries must be positive integers or 'M'",
        ),
        (
            {"conv_config": [64], "num_classes": 0},
            "num_classes must be greater than 0",
        ),
        (
            {"conv_config": [64], "in_channels": 0},
            "in_channels must be greater than 0",
        ),
        (
            {"conv_config": [64], "dropout": 1.0},
            "dropout must be in the range",
        ),
        (
            {"conv_config": [64], "classifier_hidden_features": (4096,)},
            "classifier_hidden_features must contain exactly 2 values",
        ),
        (
            {"conv_config": [64], "classifier_hidden_features": (4096, 0)},
            "classifier_hidden_features values must be greater than 0",
        ),
    ),
)
def test_vggnet_config_rejects_invalid_values(
    config_kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        VGGNetConfig(**config_kwargs)
