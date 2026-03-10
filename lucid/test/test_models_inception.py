import pytest

import lucid

from lucid.models import Inception, InceptionConfig, inception_v1, inception_v3, inception_v4


def test_inception_public_imports() -> None:
    assert Inception is not None
    assert InceptionConfig is not None
    assert inception_v1 is not None
    assert inception_v3 is not None
    assert inception_v4 is not None


def test_inception_accepts_config_object() -> None:
    config = InceptionConfig(variant="v4")

    model = Inception(config)

    assert model.config is config


def test_inception_direct_config_forward_for_v4() -> None:
    model = Inception(InceptionConfig(variant="v4"))

    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_inception_v1_factory_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = inception_v1(use_aux=False)
    logits, aux2, aux1 = model(lucid.zeros(1, 3, 224, 224))

    assert logits.shape == (1, 1000)
    assert aux2 is None
    assert aux1 is None


def test_inception_v3_factory_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = inception_v3(use_aux=False)
    logits, aux = model(lucid.zeros(1, 3, 224, 224))

    assert logits.shape == (1, 1000)
    assert aux.shape == (1, 1000)


def test_inception_v4_factory_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = inception_v4()
    output = model(lucid.zeros(1, 3, 224, 224))

    assert output.shape == (1, 1000)


def test_inception_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = inception_v4(
        num_classes=10,
        in_channels=1,
        dropout_prob=0.25,
    )
    output = model(lucid.zeros(1, 1, 224, 224))

    assert model.config == InceptionConfig(
        variant="v4",
        num_classes=10,
        in_channels=1,
        use_aux=False,
        dropout_prob=0.25,
    )
    assert output.shape == (1, 10)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    (
        ({"variant": "v2"}, "variant must be one of"),
        ({"variant": "v4", "num_classes": 0}, "num_classes must be greater than 0"),
        ({"variant": "v4", "in_channels": 0}, "in_channels must be greater than 0"),
        ({"variant": "v4", "dropout_prob": 1.0}, "dropout_prob must be in the range"),
        ({"variant": "v4", "use_aux": True}, "does not support auxiliary classifiers"),
        ({"variant": "v1", "use_aux": "invalid"}, "use_aux must be a boolean"),
    ),
)
def test_inception_config_rejects_invalid_values(
    config_kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        InceptionConfig(**config_kwargs)
