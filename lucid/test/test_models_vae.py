import pytest

import lucid
import lucid.nn as nn

from lucid.models import VAE, VAEConfig


def _small_vae_config(**kwargs: object) -> VAEConfig:
    encoder = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )
    decoder = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.Sigmoid(),
    )
    params = {
        "encoders": [encoder],
        "decoders": [decoder],
        "reconstruction_loss": "bce",
        "kl_weight": 0.5,
        "beta_schedule": lambda step: 0.25 + 0.1 * step,
    }
    params.update(kwargs)
    return VAEConfig(**params)


def test_vae_public_imports() -> None:
    assert VAE is not None
    assert VAEConfig is not None


def test_vae_accepts_config_and_runs_forward_loss() -> None:
    config = _small_vae_config()
    model = VAE(config)
    x = lucid.ones(2, 4) * 0.5

    recon, mus, logvars, zs = model(x)
    loss, recon_loss, kl = model.get_loss(x, recon, mus, logvars, zs)

    assert model.config is config
    assert recon.shape == (2, 4)
    assert len(mus) == len(logvars) == len(zs) == 1
    assert mus[0].shape == (2, 2)
    assert logvars[0].shape == (2, 2)
    assert zs[0].shape == (2, 2)
    assert loss.size == 1
    assert recon_loss.size == 1
    assert kl.size == 1
    assert model.current_beta() == pytest.approx(0.35)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"encoders": []},
        {"decoders": []},
        {"encoders": [object()]},
        {"decoders": [object()]},
        {"depth": 0},
        {"depth": 2},
        {"priors": [nn.Linear(2, 4)]},
        {"reconstruction_loss": "l1"},
        {"kl_weight": -1.0},
        {"beta_schedule": 1},
        {"hierarchical_kl": 1},
    ),
)
def test_vae_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _small_vae_config(**kwargs)
