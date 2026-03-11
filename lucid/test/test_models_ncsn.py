import pytest

import lucid

from lucid.models import NCSN, NCSNConfig


def _small_ncsn_config(**kwargs: object) -> NCSNConfig:
    params = {
        "in_channels": 1,
        "nf": 8,
        "num_classes": 4,
        "dilations": (1, 1, 1, 1),
        "sigmas": (1.0, 0.5, 0.25, 0.1),
    }
    params.update(kwargs)
    return NCSNConfig(**params)


def test_ncsn_public_imports() -> None:
    assert NCSN is not None
    assert NCSNConfig is not None


def test_ncsn_accepts_config_and_runs_forward_loss_and_sample() -> None:
    config = _small_ncsn_config()
    model = NCSN(config)
    x = lucid.ones(2, 1, 8, 8)
    labels = lucid.Tensor([0, 1], dtype=lucid.Int32)

    score = model(x, labels)
    loss, sampled_labels = model.get_loss(x)
    samples = model.sample(
        n_samples=1,
        image_size=8,
        in_channels=1,
        n_steps_each=1,
        step_lr=1e-4,
        verbose=False,
    )

    assert model.config is config
    assert tuple(model.sigmas.shape) == (4,)
    assert score.shape == (2, 1, 8, 8)
    assert loss.size == 1
    assert sampled_labels.shape == (2,)
    assert samples.shape == (1, 1, 8, 8)


def test_ncsn_set_sigmas_updates_buffer() -> None:
    model = NCSN(_small_ncsn_config(sigmas=None))

    model.set_sigmas(lucid.Tensor([2.0, 1.0, 0.5, 0.25], dtype=lucid.Float32))

    assert model.sigmas.tolist() == pytest.approx([2.0, 1.0, 0.5, 0.25])


@pytest.mark.parametrize(
    "kwargs",
    (
        {"in_channels": 0},
        {"nf": 0},
        {"num_classes": 0},
        {"dilations": (1, 1, 1)},
        {"dilations": (1, 1, 1, 0)},
        {"scale_by_sigma": 1},
        {"sigmas": (1.0, 0.5)},
        {"sigmas": (1.0, 0.5, 0.25, 0.0)},
        {"sigmas": lucid.ones(2, 2)},
    ),
)
def test_ncsn_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _small_ncsn_config(**kwargs)
