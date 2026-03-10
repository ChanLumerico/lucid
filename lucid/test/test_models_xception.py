import pytest

import lucid

from lucid.models import Xception, XceptionConfig, xception


def test_xception_public_imports() -> None:
    assert Xception is not None
    assert XceptionConfig is not None
    assert xception is not None


def test_xception_accepts_config_object() -> None:
    config = XceptionConfig(num_classes=10, in_channels=1)

    model = Xception(config)

    assert model.config is config


def test_xception_factory_forward_default_shape() -> None:
    model = xception()

    output = model(lucid.zeros(1, 3, 299, 299))

    assert output.shape == (1, 1000)


def test_xception_factory_forwards_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    model = xception(num_classes=10, in_channels=1)
    output = model(lucid.zeros(1, 1, 299, 299))

    assert model.config == XceptionConfig(num_classes=10, in_channels=1)
    assert output.shape == (1, 10)


def test_xception_factory_rejects_overriding_preset_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPHINX_BUILD", "1")

    with pytest.raises(
        TypeError,
        match="factory variants do not allow overriding preset stem_channels, entry_channels, middle_channels, middle_repeats, or exit_channels",
    ):
        xception(middle_repeats=4)


@pytest.mark.parametrize(
    ("config_kwargs", "error_type", "message"),
    (
        ({"num_classes": 0}, ValueError, "num_classes must be greater than 0"),
        ({"in_channels": 0}, ValueError, "in_channels must be greater than 0"),
        (
            {"stem_channels": [32]},
            ValueError,
            "stem_channels must contain exactly 2 channel values",
        ),
        (
            {"stem_channels": [32, 0]},
            ValueError,
            "stem_channels values must be positive integers",
        ),
        (
            {"entry_channels": [128, 256]},
            ValueError,
            "entry_channels must contain exactly 3 channel values",
        ),
        (
            {"entry_channels": [128, 0, 728]},
            ValueError,
            "entry_channels values must be positive integers",
        ),
        (
            {"middle_channels": 0},
            ValueError,
            "middle_channels must be greater than 0",
        ),
        (
            {"middle_repeats": 0},
            ValueError,
            "middle_repeats must be greater than 0",
        ),
        (
            {"exit_channels": [1024, 1536]},
            ValueError,
            "exit_channels must contain exactly 3 channel values",
        ),
        (
            {"exit_channels": [1024, 0, 2048]},
            ValueError,
            "exit_channels values must be positive integers",
        ),
    ),
)
def test_xception_config_rejects_invalid_values(
    config_kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        XceptionConfig(**config_kwargs)
